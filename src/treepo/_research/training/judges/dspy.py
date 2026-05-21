"""
DSPy-based pairwise comparison judge.

Wraps the existing PairwiseJudge DSPy module to conform to the BaseJudge protocol.
This judge uses DSPy's ChainOfThought for comparison reasoning.
"""

import logging
from typing import Any, Callable, Optional

import dspy

from treepo._research.core.signatures import PairwiseComparison
from treepo._research.core.output_parser import NormalizedOutputAccessor

from .base import BaseJudge, CompilableJudge, JudgeConfig, JudgeResult

logger = logging.getLogger(__name__)


class DSPyJudge(dspy.Module):
    """
    DSPy-based judge for pairwise summary comparison.

    Implements both BaseJudge and CompilableJudge protocols, allowing
    it to be optimized via DSPy's compile() mechanism.
    """

    def __init__(self, config: Optional[JudgeConfig] = None):
        """
        Initialize the DSPy judge.

        Args:
            config: Judge configuration (uses defaults if None)
        """
        super().__init__()
        self.config = config or JudgeConfig(type="dspy")

        # Build the comparison module
        if self.config.use_cot:
            self.compare_module = dspy.ChainOfThought(PairwiseComparison)
        else:
            self.compare_module = dspy.Predict(PairwiseComparison)

    def forward(
        self,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        reference_score: float = 0.0,
        **kwargs,
    ) -> JudgeResult:
        """
        DSPy forward pass - compare summaries.

        This method is called by DSPy's compile() mechanism.
        """
        return self.compare(
            context=context,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            reference_score=reference_score,
            **kwargs,
        )

    def compare(
        self,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
        extra_context: Optional[str] = None,
        reference_score: float = 0.0,
        **kwargs,
    ) -> JudgeResult:
        """
        Compare two summaries and return preference.

        Args:
            context: Description of what information to preserve (rubric)
            original_text: Original text being summarized
            summary_a: First candidate summary
            summary_b: Second candidate summary
            law_type: OPS law type ("sufficiency", "idempotence", "merge")
            extra_context: Additional context for the comparison
            reference_score: Ground truth score for original text
            **kwargs: Additional arguments (ignored)

        Returns:
            JudgeResult with preference and confidence
        """
        # Build full context if extra_context provided
        full_context = context
        if extra_context:
            full_context = f"{context}\n\nAdditional context: {extra_context}"

        # Call DSPy module
        result = self.compare_module(
            rubric=full_context,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            reference_score=reference_score,
        )

        # Parse result using normalized accessor (handles casing variations)
        accessor = NormalizedOutputAccessor(result)

        # Normalize preferred to uppercase
        preferred = str(accessor.get("preferred", "tie")).upper().strip()
        if preferred not in ["A", "B", "TIE"]:
            # Try to extract from string
            if "A" in preferred and "B" not in preferred:
                preferred = "A"
            elif "B" in preferred and "A" not in preferred:
                preferred = "B"
            else:
                preferred = "tie"
        elif preferred == "TIE":
            preferred = "tie"

        # Parse confidence
        try:
            confidence = float(accessor.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5

        # Parse score estimates
        score_a = None
        raw_score_a = accessor.get("score_estimate_a")
        if raw_score_a is not None:
            try:
                score_a = float(raw_score_a)
            except (ValueError, TypeError):
                pass

        score_b = None
        raw_score_b = accessor.get("score_estimate_b")
        if raw_score_b is not None:
            try:
                score_b = float(raw_score_b)
            except (ValueError, TypeError):
                pass

        return JudgeResult(
            preferred=preferred,
            confidence=confidence,
            reasoning=str(accessor.get("reasoning", "")),
            score_estimate_a=score_a,
            score_estimate_b=score_b,
            raw_result=result,
        )

    def compile(
        self,
        trainset: Any,
        metric: Optional[Callable] = None,
        optimizer_name: str = "bootstrap_random_search",
        **kwargs,
    ) -> "DSPyJudge":
        """
        Optimize the judge using DSPy.

        Args:
            trainset: Training examples
            metric: Evaluation metric
            optimizer_name: DSPy optimizer to use
            **kwargs: Additional optimizer arguments

        Returns:
            Optimized judge instance
        """
        from treepo._research.training.optimization import get_optimizer

        optimizer = get_optimizer(optimizer_name, **kwargs)
        compiled_module = optimizer.compile(
            self,
            trainset=trainset,
            metric=metric,
        )

        # Create new instance with compiled module
        new_judge = DSPyJudge(config=self.config)
        new_judge.compare_module = compiled_module.compare_module
        return new_judge


def create_dspy_judge(config: Optional[JudgeConfig] = None) -> DSPyJudge:
    """Factory function for creating DSPy judges."""
    return DSPyJudge(config=config)
