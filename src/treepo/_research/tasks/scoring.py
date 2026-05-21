"""
Generic scoring task for continuous score prediction.

This module provides a configurable ScoringTask that can be used for any
continuous scoring domain (sentiment, quality, position, etc.) by passing
appropriate configuration.

Example usage:
    from treepo._research.tasks.base import ScoringTask, ScaleDefinition

    # Create a quality scoring task (0-10 scale)
    quality_scale = ScaleDefinition(
        name="quality",
        min_value=0.0,
        max_value=10.0,
        description="Content quality score",
    )
    task = ScoringTask(
        name="quality_scoring",
        scale=quality_scale,
        rubric="Evaluate the overall quality...",
    )

    # Use with training pipeline
    metric = task.create_metric()
    training_source = task.create_training_source(results)
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .base import (
    AbstractTask,
    OutputType,
    TaskConfig,
    ScaleDefinition,
    UnifiedTrainingSource,
)
from .registry import register_task

if TYPE_CHECKING:
    import dspy
    from ..core import TrainingDataSource
    from ..config import OracleIRRConfig
    from ..inference import Retriever

from treepo._research.core.prompting import (
    PromptBuilders,
    default_summarize_prompt,
    default_merge_prompt,
)

logger = logging.getLogger(__name__)


@register_task("scoring")
class ScoringTask(AbstractTask):
    """
    Generic continuous scoring task.

    Configured entirely via constructor parameters - no domain-specific code.
    Pass a ScaleDefinition to define the score range.

    This class can be used directly with custom configuration, or via factory
    functions that provide domain-specific defaults.

    Args:
        name: Unique identifier for this task configuration
        scale: ScaleDefinition defining the score range (min, max, etc.)
        id_field: Field name for document ID in samples (default: "doc_id")
        label_field: Field name for ground truth score (default: "score")
        text_field: Field name for document text (default: "text")
        error_threshold_high: Normalized error threshold for "bad" examples (default: 0.15)
        error_threshold_low: Normalized error threshold for "good" examples (default: 0.05)
        rubric: Rubric text for evaluation guidance
        task_context: Task context for scoring/prediction
        prompt_builders: Optional custom prompt builders
        data_loader_factory: Optional factory function returning a data loader
        predictor_factory: Optional factory function returning a predictor module
        summarizer_factory: Optional factory function returning a summarizer module
        oracle_scorer_factory: Optional factory function returning an oracle scorer
    """

    def __init__(
        self,
        name: str,
        scale: ScaleDefinition,
        id_field: str = "doc_id",
        label_field: str = "score",
        text_field: str = "text",
        output_field_name: str = "score",
        error_threshold_high: float = 0.15,
        error_threshold_low: float = 0.05,
        rubric: str = "",
        task_context: str = "",
        prompt_builders: Optional[PromptBuilders] = None,
        data_loader_factory: Optional[Callable[[], Any]] = None,
        predictor_factory: Optional[Callable[[], 'dspy.Module']] = None,
        summarizer_factory: Optional[Callable[[], 'dspy.Module']] = None,
        oracle_scorer_factory: Optional[Callable[[], Callable[[str], float]]] = None,
    ):
        """Initialize the scoring task with the given configuration."""
        super().__init__()

        # Store configuration
        self._name = name
        self._scale = scale
        self._id_field = id_field
        self._label_field = label_field
        self._text_field = text_field
        self._error_threshold_high = error_threshold_high
        self._error_threshold_low = error_threshold_low
        self._rubric = rubric
        self._task_context = task_context

        # Store factories
        self._prompt_builders = prompt_builders or PromptBuilders(
            summarize=default_summarize_prompt,
            merge=default_merge_prompt,
        )
        self._data_loader_factory = data_loader_factory
        self._predictor_factory = predictor_factory
        self._summarizer_factory = summarizer_factory
        self._oracle_scorer_factory = oracle_scorer_factory

        # Create TaskConfig
        self._config = TaskConfig(
            name=name,
            output_type=OutputType.CONTINUOUS_SCORE,
            scale=scale,
            output_field_name=output_field_name,
            rubric_template=rubric,
            task_context_template=task_context,
        )

    # =========================================================================
    # Core Properties (AbstractTask interface)
    # =========================================================================

    @property
    def name(self) -> str:
        """Unique identifier for this task."""
        return self._name

    @property
    def id_field(self) -> str:
        """Field name for document ID in samples."""
        return self._id_field

    @property
    def label_field(self) -> str:
        """Field name for ground truth score."""
        return self._label_field

    @property
    def text_field(self) -> str:
        """Field name for document text."""
        return self._text_field

    # =========================================================================
    # Rubric and Context
    # =========================================================================

    def create_rubric(self, **kwargs) -> str:
        """Return the configured rubric."""
        return self._rubric

    def get_task_context(self) -> str:
        """Return the configured task context (or a generic scale-aware fallback)."""
        context = (self._task_context or "").strip()
        if context:
            return context

        scale_desc = (self._scale.description or "").strip()
        if scale_desc:
            scale_desc = f" {scale_desc}"

        output_field = (
            self._config.output_field_name
            if self._config and self._config.output_field_name
            else "score"
        )

        return (
            "Task: assign a numeric score to the provided text using the rubric/context.\n"
            f"Scale: [{self._scale.min_value:g}, {self._scale.max_value:g}].{scale_desc}\n"
            "Output requirements:\n"
            f"- Return exactly one numeric value for `{output_field}`.\n"
            "- Do not include explanations, extra fields, or alternate scales.\n"
            "- If uncertain, choose the closest valid value on the defined scale."
        )

    # =========================================================================
    # Prompt Builders
    # =========================================================================

    def create_prompt_builders(self) -> PromptBuilders:
        """Return the configured prompt builders."""
        return self._prompt_builders

    # =========================================================================
    # Data Loading
    # =========================================================================

    def get_data_loader(self) -> Any:
        """Get the data loader from the configured factory."""
        if self._data_loader_factory:
            return self._data_loader_factory()
        raise NotImplementedError(
            f"No data loader configured for task '{self._name}'. "
            "Pass a data_loader_factory to the constructor."
        )

    def get_samples(
        self,
        splits: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get samples from the data loader.

        Args:
            splits: List of splits to load (e.g., ["train", "val"])
            limit: Maximum number of samples to return

        Returns:
            List of sample dictionaries
        """
        loader = self.get_data_loader()

        # Try common data loading patterns
        if hasattr(loader, 'get_temporal_split'):
            train, val, test = loader.get_temporal_split()
            samples = []
            splits = splits or ["train", "val"]
            if "train" in splits:
                samples.extend(train)
            if "val" in splits:
                samples.extend(val)
            if "test" in splits:
                samples.extend(test)
        elif hasattr(loader, 'load'):
            samples = loader.load(splits=splits)
        elif hasattr(loader, '__iter__'):
            samples = list(loader)
        else:
            raise NotImplementedError(
                f"Data loader for task '{self._name}' does not have a recognized interface. "
                "Expected get_temporal_split(), load(), or __iter__."
            )

        if limit:
            samples = samples[:limit]

        # Convert to dictionaries if needed
        result = []
        for s in samples:
            if hasattr(s, '__dict__'):
                result.append({
                    self._id_field: getattr(s, self._id_field, None),
                    self._text_field: getattr(s, self._text_field, ''),
                    self._label_field: getattr(s, self._label_field, None),
                })
            else:
                result.append(s)

        return result

    # =========================================================================
    # Training Source
    # =========================================================================

    def create_training_source(
        self,
        results: List[Any],
        **kwargs,
    ) -> 'TrainingDataSource':
        """
        Create a training data source from processing results.

        Args:
            results: List of result objects
            **kwargs: Override thresholds or other settings

        Returns:
            UnifiedTrainingSource configured for this task's scale
        """
        error_high = kwargs.get('error_threshold_high', self._error_threshold_high)
        error_low = kwargs.get('error_threshold_low', self._error_threshold_low)

        source = UnifiedTrainingSource(
            error_threshold_high=error_high,
            error_threshold_low=error_low,
            rubric=self._rubric,
            source_name=self._name,
            scale=self._scale,
        )
        source.add_results(results)
        return source

    # =========================================================================
    # Metric
    # =========================================================================

    def create_metric(
        self,
        with_feedback: bool = True,
    ) -> Callable:
        """
        Create a DSPy-compatible metric for score prediction.

        Args:
            with_feedback: Whether to include feedback for optimization

        Returns:
            Metric function
        """
        from treepo._research.core.scoring import oracle_as_metric_with_feedback, oracle_as_metric

        if with_feedback:
            return oracle_as_metric_with_feedback
        return oracle_as_metric

    # =========================================================================
    # Predictor
    # =========================================================================

    def create_predictor(
        self,
        retriever: Optional['Retriever'] = None,
        config: Optional['OracleIRRConfig'] = None,
    ) -> 'dspy.Module':
        """
        Create a score predictor module.

        Args:
            retriever: Optional retriever for retrieval-augmented prediction
            config: Optional configuration

        Returns:
            DSPy Module for score prediction
        """
        if self._predictor_factory:
            return self._predictor_factory()
        raise NotImplementedError(
            f"No predictor configured for task '{self._name}'. "
            "Pass a predictor_factory to the constructor."
        )

    # =========================================================================
    # Oracle Scorer
    # =========================================================================

    def create_oracle_scorer(self) -> Callable[[str], float]:
        """
        Create an oracle scorer function for tournament optimization.

        Returns:
            Function(text) -> score
        """
        if self._oracle_scorer_factory:
            factory_or_fn = self._oracle_scorer_factory
            try:
                signature = inspect.signature(factory_or_fn)
                required_positional = [
                    p for p in signature.parameters.values()
                    if p.kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                    and p.default is inspect.Parameter.empty
                ]
            except (TypeError, ValueError):
                required_positional = []

            # Support both conventions:
            # 1) zero-arg factory returning scorer fn
            # 2) scorer fn directly (text -> score)
            if len(required_positional) == 0:
                scorer_fn = factory_or_fn()
                if callable(scorer_fn):
                    return scorer_fn
            return factory_or_fn

        # Try to create from predictor
        if self._predictor_factory:
            predictor = self._predictor_factory()
            task_context = self._task_context

            def oracle_predict(text: str) -> float:
                """Predict score for text using the predictor."""
                try:
                    result = predictor(text=text, task_context=task_context)
                    # Try common output patterns
                    if hasattr(result, 'score'):
                        return float(result.score)
                    if isinstance(result, dict) and 'score' in result:
                        return float(result['score'])
                    if hasattr(result, self._label_field):
                        return float(getattr(result, self._label_field))
                    return 0.0
                except Exception as e:
                    logger.warning(f"Oracle prediction failed: {e}")
                    neutral = self._scale.neutral_value if self._scale.neutral_value is not None else 0.0
                    return self.normalize_score(neutral)

            return oracle_predict

        raise NotImplementedError(
            f"No oracle scorer configured for task '{self._name}'. "
            "Pass an oracle_scorer_factory or predictor_factory to the constructor."
        )

    # =========================================================================
    # Summarizer
    # =========================================================================

    def create_summarizer(self) -> 'dspy.Module':
        """Create a summarizer module."""
        if self._summarizer_factory:
            return self._summarizer_factory()

        # Default to generic summarizer
        from treepo._research.core.signatures import Summarizer
        return Summarizer()

    def create_merge_summarizer(self) -> 'dspy.Module':
        """Create a merge summarizer module."""
        return self.create_summarizer()

    # =========================================================================
    # Info
    # =========================================================================

    def get_info(self) -> Dict[str, Any]:
        """Get information about this task configuration."""
        return {
            'name': self._name,
            'type': 'scoring',
            'output_type': 'continuous',
            'scale_name': self._scale.name,
            'scale_min': self._scale.min_value,
            'scale_max': self._scale.max_value,
            'scale_range': self._scale.range,
            'error_threshold_high': self._error_threshold_high,
            'error_threshold_low': self._error_threshold_low,
            'id_field': self._id_field,
            'label_field': self._label_field,
            'text_field': self._text_field,
            'has_data_loader': self._data_loader_factory is not None,
            'has_predictor': self._predictor_factory is not None,
            'has_summarizer': self._summarizer_factory is not None,
            'has_oracle_scorer': self._oracle_scorer_factory is not None,
        }
