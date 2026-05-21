"""
Document analysis task plugin.

This module implements the task plugin interface for generic document
analysis tasks - a general-purpose option for tree-based summarization and
information extraction without domain-specific scoring assumptions.

The task represents the end goal (what you're predicting), not the method:
- scoring_task → predicts a continuous score on a defined scale
- document_analysis → predicts content preservation quality (0 to 1)

The pipeline (tree building, summarization) is the METHOD that's constant
across tasks.

Uses:
- Content preservation evaluation (how well summaries preserve information)
- Generic quality rubrics aligned with OPS laws (sufficiency, merge consistency, idempotence)
- Simple 0-1 quality scores

Use this task for general summarization/extraction tasks or as a template
for creating custom tasks.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import dspy

from .base import (
    AbstractTask,
    OutputType,
    TaskConfig,
    ScaleDefinition,
    UnifiedTrainingSource,
    UnifiedResult,
)
from treepo._research.core.prompting import PromptBuilders, default_merge_prompt, default_summarize_prompt, parse_numeric_score
from .registry import register_task

if TYPE_CHECKING:
    from ..core import TrainingDataSource, UnifiedTrainingExample
    from ..config import OracleIRRConfig
    from ..inference import Retriever

logger = logging.getLogger(__name__)


# =============================================================================
# Scale Definition
# =============================================================================

# Generic 0-1 content preservation scale
PRESERVATION_SCALE = ScaleDefinition(
    name="preservation",
    min_value=0.0,
    max_value=1.0,
    description="Content preservation quality. 0 = information lost, 1 = fully preserved.",
    higher_is_better=True,
    neutral_value=0.5,
)


# =============================================================================
# Training Data Source
# =============================================================================


class DocumentAnalysisTrainingSource(UnifiedTrainingSource):
    """
    Training data source for document analysis results.

    This is a thin wrapper around UnifiedTrainingSource with domain-specific
    defaults for document analysis tasks.

    Extracts training examples from processed documents:
    - High prediction error → positive example (violation - info lost)
    - Low prediction error → negative example (good preservation)
    - Mid-range errors → skipped (ambiguous)
    """

    def __init__(
        self,
        error_threshold_high: float = 0.3,  # 30% of scale range
        error_threshold_low: float = 0.1,   # 10% of scale range
        rubric: str = "",
    ):
        """
        Initialize the training source.

        Args:
            error_threshold_high: Above this error → positive example (violation)
            error_threshold_low: Below this error → negative example (good)
            rubric: Task rubric for examples
        """
        super().__init__(
            error_threshold_high=error_threshold_high,
            error_threshold_low=error_threshold_low,
            rubric=rubric or OPS_PRESERVATION_RUBRIC,
            source_name="document_analysis",
            scale=PRESERVATION_SCALE,
        )

    @property
    def source_confidence(self) -> float:
        """Confidence depends on whether we have ground truth."""
        return 0.9  # Slightly higher than generic unified source


# =============================================================================
# Rubrics
# =============================================================================

OPS_PRESERVATION_RUBRIC = """
CONTENT PRESERVATION EVALUATION RUBRIC

This rubric evaluates how well summaries preserve key information from the original
content, aligned with the OPS (Oracle-Preserving Summarization) laws:

1. SUFFICIENCY (C1)
   - Does the summary contain enough information to support the same conclusions
     as the original?
   - Are key facts, arguments, and relationships preserved?
   - Would someone reading only the summary reach the same understanding?

2. MERGE CONSISTENCY (C3B)
   - When merging child summaries, is all critical information retained?
   - Are relationships between concepts preserved?
   - Is the merged result coherent and complete?

3. IDEMPOTENCE (C2)
   - Would re-summarizing the summary produce equivalent content?
   - Is the summary stable under repeated summarization?
   - Are there redundant or unstable elements?

SCORING:
- 1.0: Excellent - Full information preservation, stable, coherent
- 0.8: Good - Minor information gaps, mostly stable
- 0.6: Adequate - Some information loss but main points preserved
- 0.4: Poor - Significant information loss affecting conclusions
- 0.2: Very Poor - Critical information missing
- 0.0: Unacceptable - Summary fails to preserve essential content
""".strip()


# =============================================================================
# DSPy Signatures and Modules
# =============================================================================

class PreservationScore(dspy.Signature):
    """
    Compare original vs summary to measure information preservation.

    Evaluates how well a summary preserves key information from the original,
    aligned with OPS laws (sufficiency, merge consistency, idempotence).
    Uses pairwise comparison between original and summary for accurate assessment.
    """
    original_text: str = dspy.InputField(
        desc="Original document content to compare against"
    )
    summary: str = dspy.InputField(
        desc="Summary to evaluate for preservation quality"
    )
    task_context: str = dspy.InputField(
        desc="Evaluation criteria and domain context"
    )
    score: float = dspy.OutputField(
        desc="Preservation score from 0.0 (info lost) to 1.0 (fully preserved)"
    )
    key_preserved: str = dspy.OutputField(
        desc="Key information preserved in the summary"
    )
    key_lost: str = dspy.OutputField(
        desc="Key information lost or distorted in the summary"
    )
    reasoning: str = dspy.OutputField(
        desc="Detailed preservation assessment"
    )


class PreservationScorer(dspy.Module):
    """DSPy module for content preservation scoring via pairwise comparison."""

    def __init__(self, use_cot: bool = True):
        super().__init__()
        if use_cot:
            self.evaluate = dspy.ChainOfThought(PreservationScore)
        else:
            self.evaluate = dspy.Predict(PreservationScore)

    def forward(
        self,
        text: str = None,
        task_context: str = None,
        summary: str = None,
        rubric: str = None,
        original_content: str = None,
        original_text: str = None,
    ) -> dict:
        """
        Evaluate content preservation quality via pairwise comparison.

        Compares original text against summary to measure preservation.

        Accepts either:
        - original_text + summary + task_context (preferred format)
        - original_content + summary + rubric (training example format)
        - text + task_context (legacy format - uses text as summary, no original)

        Note: For accurate preservation scoring, original_text/original_content
        should be provided. Without it, only summary quality can be assessed.
        """
        from treepo._research.core.output_parser import NormalizedOutputAccessor

        # Support multiple calling conventions
        actual_summary = summary if summary is not None else text
        actual_context = task_context if task_context is not None else rubric
        actual_original = original_text if original_text is not None else original_content

        if actual_summary is None:
            raise ValueError("Either 'summary' or 'text' must be provided")
        if actual_context is None:
            raise ValueError("Either 'task_context' or 'rubric' must be provided")

        # If no original provided, use summary as stand-in (legacy behavior)
        # but log a warning since preservation can't be accurately measured
        if actual_original is None:
            import logging
            logging.getLogger(__name__).warning(
                "PreservationScorer called without original_text - "
                "cannot accurately measure preservation. Provide original_text "
                "or original_content for pairwise comparison."
            )
            actual_original = actual_summary  # Fallback: compare to self

        result = self.evaluate(
            original_text=actual_original,
            summary=actual_summary,
            task_context=actual_context
        )

        # Use normalized accessor for case-insensitive field access
        accessor = NormalizedOutputAccessor(result)

        return {
            'score': accessor.get('score', 0.5),
            'key_preserved': accessor.get('key_preserved', ''),
            'key_lost': accessor.get('key_lost', ''),
            'reasoning': accessor.get('reasoning', ''),
        }


def _build_preservation_score_prompt(
    summary: str,
    task_context: str
):
    """Build a prompt to score content preservation quality.

    Note: For scoring the final summary without the original text, we evaluate
    based on how well the summary appears to capture key information according
    to the task context/rubric.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a content quality evaluator for summarization systems. "
                "Evaluate how well the summary captures key information based on "
                "the task requirements."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{task_context}\n\n"
                f"SUMMARY:\n{summary}\n\n"
                "Evaluate this summary's quality and provide:\n"
                "SCORE: <0.0 to 1.0> (1.0 = excellent quality)\n"
                "KEY_ELEMENTS: <list key elements present>\n"
                "REASONING: <detailed assessment>"
            ),
        },
    ]


# =============================================================================
# Domain Implementation
# =============================================================================

@register_task(["document_analysis", "content_preservation"])
class DocumentAnalysisTask(AbstractTask):
    """
    Document analysis task implementation.

    This is a general-purpose task for document processing.
    It evaluates content preservation quality using OPS-aligned metrics.

    Quality scores range from 0.0 (poor preservation) to 1.0 (excellent).
    Uses continuous score prediction for quality assessment.
    """

    def __init__(
        self,
        error_threshold_high: float = 0.3,
        error_threshold_low: float = 0.1,
        max_preservation_scorer_text_length: Optional[int] = None,
    ):
        """
        Initialize document analysis domain.

        Args:
            error_threshold_high: Error above which is labeled as violation
            error_threshold_low: Error below which is labeled as good
        """
        super().__init__()
        self.error_threshold_high = error_threshold_high
        self.error_threshold_low = error_threshold_low
        self.max_preservation_scorer_text_length = max_preservation_scorer_text_length

        # Set up task configuration
        self._config = TaskConfig(
            name="document_analysis",
            output_type=OutputType.CONTINUOUS_SCORE,
            scale=PRESERVATION_SCALE,
            output_field_name="score",
            rubric_template="",
            task_context_template="",
        )

    @property
    def name(self) -> str:
        return "document_analysis"

    def create_training_source(
        self,
        results: List[Any],
        **kwargs,
    ) -> 'TrainingDataSource':
        """
        Create a training data source from document analysis results.

        Args:
            results: List of result objects (UnifiedResult/DocumentAnalysisResult,
                dicts, or objects with compatible attributes):
                - doc_id: Document identifier
                - final_summary: The summary text
                - reference_score: Ground truth score (0-1) if available
                - estimated_score: Model's predicted score
                - error: Error message if processing failed (optional)
                - metadata: Additional metadata dict (optional)
            **kwargs: Override error thresholds:
                - error_threshold_high: Override high error threshold
                - error_threshold_low: Override low error threshold
                - rubric: Override rubric text

        Returns:
            DocumentAnalysisTrainingSource that implements TrainingDataSource protocol
        """
        source = DocumentAnalysisTrainingSource(
            error_threshold_high=kwargs.get('error_threshold_high', self.error_threshold_high),
            error_threshold_low=kwargs.get('error_threshold_low', self.error_threshold_low),
            rubric=kwargs.get('rubric', self.create_rubric()),
        )

        # UnifiedTrainingSource.add_results handles all conversions
        source.add_results(results)
        return source

    def create_metric(
        self,
        with_feedback: bool = True,
    ) -> Callable:
        """
        Create a DSPy-compatible metric for content preservation.

        Args:
            with_feedback: Whether to include feedback for GEPA reflection

        Returns:
            Metric function
        """
        from treepo._research.core.scoring import oracle_as_metric_with_feedback, oracle_as_metric

        if with_feedback:
            return oracle_as_metric_with_feedback
        return oracle_as_metric

    def create_trainset(self, results: List[Any]) -> List['dspy.Example']:
        """
        Create a DSPy trainset, optionally filtering to short documents.

        Preservation scoring requires original text. To avoid context overflow,
        we can subset to shorter originals based on a configurable token limit.
        """
        max_length = self.max_preservation_scorer_text_length
        if not max_length:
            return super().create_trainset(results)

        from treepo._research.config.settings import load_settings
        from treepo._research.preprocessing.tokenizer import TokenCounter

        settings = load_settings()
        chunking_cfg = settings.get("chunking", {})
        model_name = chunking_cfg.get("model", "gpt-4")
        counter = TokenCounter(model=model_name)

        filtered_results = []
        skipped = 0
        total = len(results)

        for result in results:
            if result is None or getattr(result, "error", None):
                continue

            original_content = getattr(result, "original_content", None)
            if not original_content:
                metadata = getattr(result, "metadata", None)
                if isinstance(metadata, dict):
                    original_content = metadata.get("original_content")

            if not original_content:
                skipped += 1
                continue

            try:
                token_count = counter.count(original_content)
            except Exception:
                token_count = len(original_content)

            if token_count <= max_length:
                filtered_results.append(result)
            else:
                skipped += 1

        if skipped:
            logger.info(
                "DocumentAnalysisTask: skipped %d/%d results for preservation scoring "
                "(max_tokens=%d)",
                skipped,
                total,
                max_length,
            )

        return super().create_trainset(filtered_results)

    def create_prompt_builders(self) -> PromptBuilders:
        """Create prompt builders for document analysis tasks."""
        return PromptBuilders(
            summarize=default_summarize_prompt,
            merge=default_merge_prompt,
            score=_build_preservation_score_prompt,
            audit=None,
        )

    def parse_score(self, response: str) -> Optional[float]:
        """Parse a 0-1 preservation score."""
        parsed = parse_numeric_score(response, min_value=0.0, max_value=1.0)
        return self.normalize_score(parsed)

    def create_rubric(self, **kwargs) -> str:
        """
        Create a content preservation rubric.

        Args:
            **kwargs: Rubric customization options
                - focus_areas: List of specific areas to focus on
                - max_length: Target summary length guidance

        Returns:
            Rubric string
        """
        focus_areas = kwargs.get('focus_areas', [])
        max_length = kwargs.get('max_length', None)

        rubric = OPS_PRESERVATION_RUBRIC

        if focus_areas:
            rubric += "\n\nFOCUS AREAS:\n"
            for area in focus_areas:
                rubric += f"- {area}\n"

        if max_length:
            rubric += f"\n\nTARGET LENGTH: Approximately {max_length} words"

        return rubric

    def get_task_context(self) -> str:
        """
        Get the task context for content preservation evaluation.

        Returns:
            Task context string
        """
        return """
Evaluate the content preservation quality of this summary on a scale from 0.0 to 1.0.

Consider the OPS (Oracle-Preserving Summarization) laws:
- SUFFICIENCY: Does the summary contain enough information for the same conclusions?
- MERGE CONSISTENCY: Are relationships and critical information preserved?
- IDEMPOTENCE: Is the summary stable under re-summarization?

Output a single score between 0.0 (poor preservation) and 1.0 (excellent preservation).
""".strip()

    def create_predictor(
        self,
        retriever: Optional['Retriever'] = None,
        config: Optional['OracleIRRConfig'] = None,
    ) -> dspy.Module:
        """
        Create a content preservation predictor module.

        Args:
            retriever: Optional retriever for retrieval-augmented prediction
            config: Optional configuration

        Returns:
            DSPy Module for preservation prediction (PreservationScorer)
        """
        return PreservationScorer(use_cot=True)

    def get_info(self) -> Dict[str, Any]:
        """Get information about this domain."""
        base_info = super().get_info()
        base_info.update({
            'error_threshold_high': self.error_threshold_high,
            'error_threshold_low': self.error_threshold_low,
            'score_type': 'continuous',
            'score_range': '[0, 1]',
            'description': 'Document analysis with content preservation evaluation',
        })
        return base_info
