"""
Task plugin base protocol and abstractions.

This module defines the interface for task-specific training integration.
Tasks represent different use cases (e.g., scoring, classification,
information extraction) that can plug into the OPS training framework.

Key abstractions:
- OutputType: What kind of output the task produces (continuous, discrete, structured)
- ScaleDefinition: For continuous outputs, defines the scale range and semantics
- LabelDefinition: For discrete outputs, defines valid labels
- TaskConfig: Configuration for a task including its output type
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import dspy
    from ..core import TrainingDataSource, UnifiedTrainingExample
    from ..config import OracleIRRConfig
    from ..inference import Retriever

from treepo._research.core.prompting import (
    PromptBuilders,
    default_merge_prompt,
    default_summarize_prompt,
    clean_summary_text,
    parse_numeric_score,
)
logger = logging.getLogger(__name__)


# =============================================================================
# Output Type Abstractions
# =============================================================================

class OutputType(Enum):
    """Type of output the task produces."""
    CONTINUOUS_SCORE = "continuous"   # e.g., -1 to +1, 0 to 100, 0 to 1
    DISCRETE_LABEL = "discrete"       # e.g., "positive", "negative", "neutral"
    STRUCTURED = "structured"         # e.g., {"entities": [...], "relations": [...]}


@dataclass
class ScaleDefinition:
    """
    Definition for continuous score outputs.

    Used with OutputType.CONTINUOUS_SCORE to define the valid range
    and semantics of numeric outputs.
    """
    name: str                    # e.g., "quality", "sentiment", "relevance"
    min_value: float             # e.g., -100
    max_value: float             # e.g., 100
    description: str = ""        # Human-readable description
    higher_is_better: bool = True  # For optimization direction
    neutral_value: Optional[float] = None  # e.g., 0.0 or 0.5

    @property
    def range(self) -> float:
        """Get the range of the scale."""
        return self.max_value - self.min_value

    def normalize(self, value: float) -> float:
        """Normalize a value to 0-1 range."""
        return (value - self.min_value) / self.range

    def denormalize(self, normalized: float) -> float:
        """Convert from 0-1 range back to scale range."""
        return normalized * self.range + self.min_value

    def clamp(self, value: float) -> float:
        """Clamp a value to the valid range."""
        return max(self.min_value, min(self.max_value, value))

    def distance_to_score(self, distance: float) -> float:
        """
        Convert distance to normalized similarity score.

        Args:
            distance: Absolute distance between two values

        Returns:
            Score in [0.0, 1.0] where 1.0 = no distance (identical)
        """
        # Linear: score = 1 - (distance / range), clamped to [0, 1]
        return max(0.0, 1.0 - abs(distance) / self.range)

    def values_to_score(self, value_a: float, value_b: float) -> float:
        """
        Compute similarity score between two values on this scale.

        Args:
            value_a: First value
            value_b: Second value

        Returns:
            Score in [0.0, 1.0] where 1.0 = identical
        """
        return self.distance_to_score(abs(value_a - value_b))


@dataclass
class LabelDefinition:
    """
    Definition for discrete label outputs.

    Used with OutputType.DISCRETE_LABEL to define valid labels
    and their semantics.
    """
    name: str                           # e.g., "sentiment", "topic"
    labels: List[str] = field(default_factory=list)  # e.g., ["positive", "negative", "neutral"]
    descriptions: Dict[str, str] = field(default_factory=dict)  # label -> description
    default_label: Optional[str] = None  # Default if prediction fails

    def is_valid(self, label: str) -> bool:
        """Check if a label is valid."""
        return label in self.labels


@dataclass
class TaskConfig:
    """
    Configuration for a task.

    Captures all the metadata needed to describe what a task does
    and how its outputs should be interpreted.
    """
    name: str                                 # Unique task identifier
    output_type: OutputType                   # Type of output
    scale: Optional[ScaleDefinition] = None   # For CONTINUOUS_SCORE
    labels: Optional[LabelDefinition] = None  # For DISCRETE_LABEL
    rubric_template: str = ""                 # Template for generating rubrics
    task_context_template: str = ""           # Template for task context
    output_field_name: str = "score"          # Field name in predictions

    def __post_init__(self):
        # Validate that the appropriate definition is provided
        if self.output_type == OutputType.CONTINUOUS_SCORE and self.scale is None:
            raise ValueError("CONTINUOUS_SCORE output type requires a ScaleDefinition")
        if self.output_type == OutputType.DISCRETE_LABEL and self.labels is None:
            raise ValueError("DISCRETE_LABEL output type requires a LabelDefinition")


# Domain-specific scales should live in task modules.


# =============================================================================
# Unified Training Source (Generic, Task-Agnostic)
# =============================================================================

@dataclass
class UnifiedResult:
    """
    Generic result from processing a document through the tree pipeline.

    This is a task-agnostic format that can hold results from any task.
    Tasks can extend this or use duck-typing for their own result types.
    """
    doc_id: str
    final_summary: str
    reference_score: Optional[float] = None  # Ground truth if available
    estimated_score: Optional[float] = None  # Model prediction
    error: Optional[str] = None  # Error message if processing failed
    original_content: Optional[str] = None  # Original document text
    rubric: Optional[str] = None  # Task rubric used
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedTrainingSource:
    """
    Generic, task-agnostic training data source.

    This source extracts training examples from any processing results that have:
    - doc_id: Document identifier
    - final_summary: The summary text
    - reference_score: Ground truth score (if available)
    - estimated_score: Model's predicted score

    Labeling strategy (based on prediction error):
    - High error → POSITIVE example (violation - info lost)
    - Low error → NEGATIVE example (good preservation)
    - Mid-range errors → skipped (ambiguous)

    This class uses "Unified" naming to match UnifiedTrainingExample and provides
    a consistent vocabulary across the training framework.
    """

    def __init__(
        self,
        error_threshold_high: float = 0.3,  # 30% of scale range by default
        error_threshold_low: float = 0.1,   # 10% of scale range by default
        rubric: str = "",
        source_name: str = "unified",
        scale: Optional[ScaleDefinition] = None,
    ):
        """
        Initialize the training source.

        Args:
            error_threshold_high: Normalized error above which → positive example (violation)
            error_threshold_low: Normalized error below which → negative example (good)
            rubric: Task rubric for examples
            source_name: Identifier for this source (default: "unified")
            scale: Optional scale definition for error normalization
        """
        self.error_threshold_high = error_threshold_high
        self.error_threshold_low = error_threshold_low
        self.rubric = rubric
        self._source_name = source_name
        self.scale = scale

        self._results: List[UnifiedResult] = []
        self._processed_count = 0

    def add_result(self, result: Union[UnifiedResult, Dict[str, Any], Any]) -> None:
        """
        Add a processed document result.

        Accepts UnifiedResult, dict, or any object with compatible attributes.
        """
        if isinstance(result, UnifiedResult):
            self._results.append(result)
        elif isinstance(result, dict):
            self._results.append(UnifiedResult(
                doc_id=result.get('doc_id', result.get('id', str(len(self._results)))),
                final_summary=result.get('final_summary', result.get('summary', '')),
                reference_score=result.get('reference_score'),
                estimated_score=result.get('estimated_score'),
                error=result.get('error'),
                original_content=result.get('original_content', ''),
                rubric=result.get('rubric', self.rubric),
                metadata={k: v for k, v in result.items()
                         if k not in ('doc_id', 'id', 'final_summary', 'summary',
                                     'reference_score', 'estimated_score', 'error',
                                     'original_content', 'rubric')},
            ))
        else:
            # Duck-type from object attributes
            self._results.append(UnifiedResult(
                doc_id=getattr(result, 'doc_id', str(len(self._results))),
                final_summary=getattr(result, 'final_summary', ''),
                reference_score=getattr(result, 'reference_score', None),
                estimated_score=getattr(result, 'estimated_score', None),
                error=getattr(result, 'error', None),
                original_content=getattr(result, 'original_content', ''),
                rubric=getattr(result, 'rubric', self.rubric),
                metadata=getattr(result, 'metadata', {}),
            ))

    def add_results(self, results: List[Any]) -> None:
        """Add multiple processed document results."""
        for result in results:
            self.add_result(result)

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def source_type(self) -> str:
        """Implement TrainingDataSource protocol."""
        return self._source_name

    @property
    def source_confidence(self) -> float:
        """Base confidence for examples from this source."""
        return 0.85

    def _normalize_error(self, raw_error: float) -> float:
        """Normalize error to 0-1 range based on scale."""
        if self.scale:
            return raw_error / self.scale.range
        return raw_error  # Assume already normalized

    def get_examples(self) -> List['UnifiedTrainingExample']:
        """
        Extract training examples from document results.

        Strategy:
        - High prediction error → POSITIVE (violation - info lost)
        - Low prediction error → NEGATIVE (good summary)
        - Mid-range errors → skip (ambiguous)
        """
        from ..core import UnifiedTrainingExample, TrainingExampleLabel, ViolationType

        examples = []

        for result in self._results:
            # Skip if error or missing scores
            if result.error is not None:
                continue
            if result.estimated_score is None or result.reference_score is None:
                continue

            raw_error = abs(result.estimated_score - result.reference_score)
            error = self._normalize_error(raw_error)

            # Determine label based on prediction error
            if error >= self.error_threshold_high:
                # High error = info was lost
                label = TrainingExampleLabel.POSITIVE
                violation_type = ViolationType.SUFFICIENCY
                confidence = min(0.95, 0.7 + error)  # Higher error = higher confidence
            elif error <= self.error_threshold_low:
                # Low error = summary preserved info
                label = TrainingExampleLabel.NEGATIVE
                violation_type = ViolationType.NONE
                confidence = min(0.95, 0.7 + (self.error_threshold_low - error) / 0.2)
            else:
                # Mid-range error = ambiguous, skip
                continue

            example = UnifiedTrainingExample(
                example_id=f"{self._source_name}_{result.doc_id}",
                source_type=self._source_name,
                original_content=result.original_content or f"[Document: {result.doc_id}]",
                summary=result.final_summary,
                rubric=result.rubric or self.rubric,
                context={
                    'doc_id': result.doc_id,
                    'reference_score': result.reference_score,
                    'estimated_score': result.estimated_score,
                    'prediction_error': error,
                    'raw_error': raw_error,
                    **result.metadata,
                },
                label=label,
                violation_type=violation_type,
                human_reasoning=(
                    f"Reference: {result.reference_score:.3f}, "
                    f"Predicted: {result.estimated_score:.3f}, "
                    f"Error: {error:.3f}"
                ),
                confidence=confidence,
            )
            examples.append(example)

        self._processed_count = len(examples)
        return examples

    def get_positive_examples(self) -> List['UnifiedTrainingExample']:
        """Return positive (violation) examples only."""
        from ..core import TrainingExampleLabel
        return [e for e in self.get_examples() if e.label == TrainingExampleLabel.POSITIVE]

    def get_negative_examples(self) -> List['UnifiedTrainingExample']:
        """Return negative (good) examples only."""
        from ..core import TrainingExampleLabel
        return [e for e in self.get_examples() if e.label == TrainingExampleLabel.NEGATIVE]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the training source."""
        errors = []
        for r in self._results:
            if r.estimated_score is not None and r.reference_score is not None and r.error is None:
                raw_error = abs(r.estimated_score - r.reference_score)
                errors.append(self._normalize_error(raw_error))

        return {
            'total_results': len(self._results),
            'processed_examples': self._processed_count,
            'high_error_count': sum(1 for e in errors if e >= self.error_threshold_high),
            'low_error_count': sum(1 for e in errors if e <= self.error_threshold_low),
            'mean_error': sum(errors) / len(errors) if errors else 0,
            'error_threshold_high': self.error_threshold_high,
            'error_threshold_low': self.error_threshold_low,
        }


# Default rubric for generic document analysis (OPS-aligned)
DEFAULT_UNIFIED_RUBRIC = """
CONTENT PRESERVATION EVALUATION

Evaluate how well summaries preserve key information:

1. SUFFICIENCY: Does the summary contain enough information for the same conclusions?
2. MERGE CONSISTENCY: Are relationships and critical information preserved?
3. IDEMPOTENCE: Is the summary stable under re-summarization?

Score from 0.0 (poor) to 1.0 (excellent).
""".strip()


# =============================================================================
# Task Plugin Protocol
# =============================================================================

@runtime_checkable
class TaskPlugin(Protocol):
    """
    Protocol for task-specific training integration.

    A task plugin provides all the task-specific components needed to
    train and evaluate score predictors for a particular use case.

    Example tasks:
    - relevance_scoring: Document relevance scoring
    - sentiment: Sentiment score prediction
    - quality: Quality estimation for summaries
    """

    @property
    def name(self) -> str:
        """Unique identifier for this task."""
        ...

    def create_training_source(
        self,
        results: List[Any],
        **kwargs,
    ) -> 'TrainingDataSource':
        """
        Create a training data source from processing results.

        Args:
            results: List of task-specific result objects
            **kwargs: Task-specific configuration

        Returns:
            TrainingDataSource that can be added to UnifiedTrainingCollector
        """
        ...

    def create_metric(
        self,
        with_feedback: bool = True,
    ) -> Callable:
        """
        Create a DSPy-compatible metric function for this task.

        Args:
            with_feedback: Whether to include feedback for GEPA reflection

        Returns:
            Metric function compatible with DSPy optimizers
        """
        ...

    def create_predictor(
        self,
        retriever: Optional['Retriever'] = None,
        config: Optional['OracleIRRConfig'] = None,
    ) -> 'dspy.Module':
        """
        Create a score predictor module for this task.

        Args:
            retriever: Optional retriever for retrieval-augmented prediction
            config: Optional Oracle IRR configuration

        Returns:
            DSPy Module for score prediction
        """
        ...

    def create_rubric(self, **kwargs) -> str:
        """
        Create a rubric string for this task.

        Rubrics guide the summarization and prediction process.

        Args:
            **kwargs: Task-specific rubric configuration

        Returns:
            Rubric string
        """
        ...


# =============================================================================
# Abstract Base Class
# =============================================================================

class AbstractTask(ABC):
    """
    Abstract base class for task implementations.

    Provides common functionality and enforces the interface.
    Subclasses should implement the abstract methods.
    """

    def __init__(self):
        """Initialize task."""
        self._initialized = False
        self._config: Optional[TaskConfig] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this task."""
        pass

    @property
    def config(self) -> Optional[TaskConfig]:
        """Get the task configuration."""
        return self._config

    @property
    def output_type(self) -> OutputType:
        """Get the output type for this task."""
        if self._config:
            return self._config.output_type
        return OutputType.CONTINUOUS_SCORE  # Default for backward compatibility

    @property
    def scale(self) -> Optional[ScaleDefinition]:
        """Get the scale definition for continuous outputs."""
        if self._config:
            return self._config.scale
        return None

    @property
    def output_field_name(self) -> str:
        """Get the field name for output values."""
        if self._config:
            return self._config.output_field_name
        return "score"  # Default

    def normalize_score(self, value: Optional[float]) -> Optional[float]:
        """Normalize a raw score to 0-1 using the task scale."""
        if value is None:
            return None
        if self.scale:
            normalized = self.scale.normalize(float(value))
            return max(0.0, min(1.0, normalized))
        return float(value)

    def denormalize_score(self, normalized: Optional[float]) -> Optional[float]:
        """Convert a normalized 0-1 score back to the task scale."""
        if normalized is None:
            return None
        if self.scale:
            return self.scale.denormalize(float(normalized))
        return float(normalized)

    # =========================================================================
    # Data Field Configuration (for task-agnostic data loading)
    # =========================================================================

    @property
    def id_field(self) -> str:
        """Field name for document ID in samples.

        Override in subclasses if the task uses a different ID field name.
        Default: "doc_id"
        """
        return "doc_id"

    @property
    def label_field(self) -> str:
        """Field name for ground truth label/score in samples.

        Override in subclasses if the task uses a different label field name.
        Default: "reference_score"
        """
        return "reference_score"

    @property
    def text_field(self) -> str:
        """Field name for document text in samples.

        Override in subclasses if the task uses a different text field name.
        Default: "text"
        """
        return "text"

    # =========================================================================
    # Data Loading (for task-agnostic data access)
    # =========================================================================

    def get_data_loader(self) -> Any:
        """Get task-specific data loader instance.

        Override in subclasses to provide task-specific data loading.

        Returns:
            Data loader instance (task-specific type)

        Raises:
            NotImplementedError: If task doesn't implement data loading
        """
        raise NotImplementedError(
            f"get_data_loader not implemented for task '{self.name}'. "
            "Override this method to enable data loading."
        )

    def get_samples(
        self,
        splits: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get samples from the task's data source.

        Override in subclasses to provide task-specific sample loading.

        Args:
            splits: List of splits to load (e.g., ["train", "val"]).
                   If None, loads default splits.
            limit: Maximum number of samples to return. If None, returns all.

        Returns:
            List of sample dictionaries with at least id_field, text_field,
            and optionally label_field.

        Raises:
            NotImplementedError: If task doesn't implement sample loading
        """
        raise NotImplementedError(
            f"get_samples not implemented for task '{self.name}'. "
            "Override this method to enable sample loading."
        )

    def create_training_source(
        self,
        results: List[Any],
        **kwargs,
    ) -> 'TrainingDataSource':
        """
        Create a training data source from processing results.

        Default implementation uses UnifiedTrainingSource, which works with
        any result objects that have doc_id, final_summary, reference_score,
        and estimated_score fields.

        Subclasses can override this to provide task-specific handling.

        Args:
            results: List of result objects (UnifiedResult, dict, or duck-typed)
            **kwargs: Override error thresholds:
                - error_threshold_high: Override high error threshold
                - error_threshold_low: Override low error threshold
                - rubric: Override rubric text
                - source_name: Override source name

        Returns:
            UnifiedTrainingSource that implements TrainingDataSource protocol
        """
        # Get thresholds from kwargs or use defaults
        error_high = kwargs.get('error_threshold_high', 0.3)
        error_low = kwargs.get('error_threshold_low', 0.1)
        rubric = kwargs.get('rubric', self.create_rubric() if hasattr(self, 'create_rubric') else DEFAULT_UNIFIED_RUBRIC)
        source_name = kwargs.get('source_name', self.name if hasattr(self, 'name') else 'unified')

        source = UnifiedTrainingSource(
            error_threshold_high=error_high,
            error_threshold_low=error_low,
            rubric=rubric,
            source_name=source_name,
            scale=self.scale if hasattr(self, 'scale') else None,
        )
        source.add_results(results)
        return source

    @abstractmethod
    def create_metric(
        self,
        with_feedback: bool = True,
    ) -> Callable:
        """Create a DSPy-compatible metric function."""
        pass

    def create_predictor(
        self,
        retriever: Optional['Retriever'] = None,
        config: Optional['OracleIRRConfig'] = None,
    ) -> 'dspy.Module':
        """Create a score predictor module for this task.

        Default implementation raises NotImplementedError.
        Override in subclasses if score prediction is needed.
        """
        raise NotImplementedError("create_predictor not implemented for this task")

    def create_oracle_scorer(self) -> Callable[[str], float]:
        """
        Create an oracle scorer function for tournament of tournaments.

        The oracle scorer takes a summary text and returns a predicted score.
        This is used to determine which summaries lead to better downstream
        task performance during judge optimization.

        The oracle scorer should be fast enough to score many summaries
        (hundreds to thousands) during the training loop.

        Returns:
            Function(text) -> score

        Raises:
            NotImplementedError: If task doesn't support oracle scoring
        """
        raise NotImplementedError(
            f"create_oracle_scorer not implemented for task '{self.name}'. "
            "Override this method to enable tournament of tournaments."
        )

    def describe_local_law_oracle(self) -> Dict[str, Any]:
        """
        Describe the task-provided node-span oracle, if any.

        Tasks with exact/mechanical DGP-backed span verification should override
        this and set ``exact=True``. Real-data tasks may also expose model-backed
        task oracles, but those remain explicit fallback label sources.
        """
        return {
            "available": False,
            "exact": False,
            "model_backed": False,
            "kind": "none",
            "spec": None,
        }

    def create_local_law_oracle(self, **_: Any) -> Callable[[str], float]:
        """
        Create a node-span oracle for local-law supervision.

        Override in tasks/settings where the teacher already supplies a span
        oracle. For DGP-backed settings this should be an exact mechanical
        verifier; for real-data tasks this may be a task-specific fallback
        teacher labeler.
        """
        raise NotImplementedError(
            f"create_local_law_oracle not implemented for task '{self.name}'. "
            "Override this method to expose task-provided node-span labels."
        )

    def create_preference_labeler(self) -> Optional[Callable[['PreferencePair', float], Optional[str]]]:
        """
        Optional preference labeler for tournament of tournaments.

        Override to customize how preference pairs are labeled from metrics
        (e.g., lower error is better, higher score is better, domain-specific rules).
        """
        return None

    def create_merge_summarizer(self) -> 'dspy.Module':
        """Create a merge summarizer module for tree building."""
        return self.create_summarizer()

    def create_summarizer(self) -> 'dspy.Module':
        """Create a summarizer module for tree building."""
        from treepo._research.core.signatures import Summarizer
        return Summarizer()

    @abstractmethod
    def create_rubric(self, **kwargs) -> str:
        """Create a rubric string for this task."""
        pass

    def get_task_context(self) -> str:
        """Return the task context for scoring or evaluation."""
        return ""

    def create_prompt_builders(self) -> PromptBuilders:
        """Create prompt builders for summarization and merge."""
        return PromptBuilders(
            summarize=default_summarize_prompt,
            merge=default_merge_prompt,
            score=None,
            audit=None,
        )

    def parse_score(self, response: str) -> Optional[float]:
        """Parse a numeric score from an LLM response."""
        min_value = self.scale.min_value if self.scale else None
        max_value = self.scale.max_value if self.scale else None
        parsed = parse_numeric_score(response, min_value=min_value, max_value=max_value)
        return self.normalize_score(parsed)

    def normalize_prediction_output(self, result: Any) -> Any:
        """Normalize the output field value in a predictor result."""
        if result is None:
            return result

        output_field = self.output_field_name
        value = None

        if isinstance(result, dict):
            value = result.get(output_field)
        else:
            value = getattr(result, output_field, None)
            if value is None and hasattr(result, "__getitem__"):
                try:
                    value = result[output_field]
                except (KeyError, TypeError):
                    value = None

        if value is None:
            return result

        normalized = self.normalize_score(value)

        if isinstance(result, dict):
            result[output_field] = normalized
            return result

        if hasattr(result, output_field):
            try:
                setattr(result, output_field, normalized)
                return result
            except Exception:
                pass

        if hasattr(result, "__setitem__"):
            try:
                result[output_field] = normalized
                return result
            except Exception:
                pass

        return {output_field: normalized}

    def wrap_predictor(self, predictor: 'dspy.Module') -> 'dspy.Module':
        """Wrap a predictor to normalize its outputs to 0-1."""
        import dspy

        task = self

        class NormalizedPredictor(dspy.Module):
            def __init__(self, base: 'dspy.Module'):
                super().__init__()
                self.base = base

            def forward(self, *args, **kwargs):
                result = self.base(*args, **kwargs)
                return task.normalize_prediction_output(result)

        return NormalizedPredictor(predictor)

    def create_trainset(self, results: List[Any]) -> List['dspy.Example']:
        """
        Create a simple DSPy trainset from results.

        Expects result objects with final_summary and reference_score fields.
        """
        import dspy

        rubric = self.create_rubric()
        task_context = self.get_task_context()
        examples = []
        for result in results:
            if result is None or getattr(result, "error", None):
                continue
            summary = clean_summary_text(getattr(result, "final_summary", ""))
            reference = getattr(result, "reference_score", None)
            if not summary or reference is None:
                continue

            original_content = getattr(result, "original_content", None)
            if not original_content:
                metadata = getattr(result, "metadata", None)
                if isinstance(metadata, dict):
                    original_content = metadata.get("original_content")
            if not original_content:
                original_content = getattr(result, "doc_id", "")
            doc_id = getattr(result, "doc_id", None)
            metadata = getattr(result, "metadata", None)
            prompt_metadata = dict(metadata) if isinstance(metadata, dict) else None

            example = dspy.Example(
                doc_id=doc_id,
                original_content=original_content,
                summary=summary,
                rubric=rubric,
                task_context=task_context,
                reference_score=reference,
                metadata=prompt_metadata,
            ).with_inputs("original_content", "summary", "rubric")
            examples.append(example)

        return examples

    def validate(self) -> bool:
        """
        Validate that the task is properly configured.

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required properties
            _ = self.name
            return True
        except Exception as e:
            logger.error(f"Task validation failed: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this task.

        Returns:
            Dict with task metadata
        """
        info = {
            'name': self.name,
            'output_type': self.output_type.value,
        }
        if self.scale:
            info['scale'] = {
                'name': self.scale.name,
                'min': self.scale.min_value,
                'max': self.scale.max_value,
                'range': self.scale.range,
            }
        if self._config and self._config.labels:
            info['labels'] = self._config.labels.labels
        return info
