"""
Oracle-based pairwise comparison judge.

Uses oracle scoring functions to compare summaries by their
preservation of task-critical information.
"""

import logging
from typing import Callable, Optional

from .base import JudgeConfig, JudgeResult

logger = logging.getLogger(__name__)


class OracleJudge:
    """
    Judge that uses oracle predictions for comparison.

    Compares summaries by computing oracle scores for each and
    determining which has lower error relative to ground truth.
    """

    def __init__(
        self,
        oracle_fn: Callable[[str], float],
        config: Optional[JudgeConfig] = None,
    ):
        """
        Initialize the oracle judge.

        Args:
            oracle_fn: Function that scores text (returns task-specific value)
            config: Judge configuration (uses defaults if None)
        """
        self.oracle_fn = oracle_fn
        self.config = config or JudgeConfig(type="oracle")
        self.tie_margin = self.config.tie_margin

    def compare(
        self,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
        extra_context: Optional[str] = None,
        ground_truth: Optional[float] = None,
        scale_range: Optional[float] = None,
        **kwargs,
    ) -> JudgeResult:
        """
        Compare two summaries using oracle predictions.

        Args:
            context: Description of what information to preserve (rubric)
            original_text: Original text being summarized
            summary_a: First candidate summary
            summary_b: Second candidate summary
            law_type: OPS law type ("sufficiency", "idempotence", "merge")
            extra_context: Additional context (unused for oracle)
            ground_truth: Ground truth score for the original text
                         If None, computed from original_text
            scale_range: Range of the scale for normalization
                        If None, uses raw error difference
            **kwargs: Additional arguments (ignored)

        Returns:
            JudgeResult with preference based on error comparison
        """
        # Get ground truth if not provided
        if ground_truth is None:
            ground_truth = self.oracle_fn(original_text)

        # Score both summaries
        score_a = self.oracle_fn(summary_a)
        score_b = self.oracle_fn(summary_b)

        # Compute errors (absolute deviation from ground truth)
        error_a = abs(score_a - ground_truth)
        error_b = abs(score_b - ground_truth)

        # Normalize errors if scale_range provided
        if scale_range is not None and scale_range > 0:
            norm_error_a = error_a / scale_range
            norm_error_b = error_b / scale_range
        else:
            norm_error_a = error_a
            norm_error_b = error_b

        # Determine preference based on normalized error
        error_diff = norm_error_a - norm_error_b

        if abs(error_diff) <= self.tie_margin:
            # Within tie margin - no clear winner
            preferred = "tie"
            confidence = 0.5
            reasoning = (
                f"Tie: errors within margin ({self.tie_margin:.3f}). "
                f"Error A: {norm_error_a:.3f}, Error B: {norm_error_b:.3f}"
            )
        elif error_diff > 0:
            # A has higher error, B wins
            preferred = "B"
            confidence = min(0.95, 0.5 + abs(error_diff) * 2)
            reasoning = (
                f"B preferred: lower error ({norm_error_b:.3f} vs {norm_error_a:.3f}). "
                f"Scores: A={score_a:.2f}, B={score_b:.2f}, GT={ground_truth:.2f}"
            )
        else:
            # B has higher error, A wins
            preferred = "A"
            confidence = min(0.95, 0.5 + abs(error_diff) * 2)
            reasoning = (
                f"A preferred: lower error ({norm_error_a:.3f} vs {norm_error_b:.3f}). "
                f"Scores: A={score_a:.2f}, B={score_b:.2f}, GT={ground_truth:.2f}"
            )

        return JudgeResult(
            preferred=preferred,
            confidence=confidence,
            reasoning=reasoning,
            score_estimate_a=score_a,
            score_estimate_b=score_b,
            raw_result={
                "error_a": error_a,
                "error_b": error_b,
                "norm_error_a": norm_error_a,
                "norm_error_b": norm_error_b,
                "ground_truth": ground_truth,
                "scale_range": scale_range,
            },
        )


def create_oracle_judge(
    oracle_fn: Callable[[str], float],
    tie_margin: float = 0.05,
    config: Optional[JudgeConfig] = None,
) -> OracleJudge:
    """
    Factory function for creating oracle judges.

    Args:
        oracle_fn: Function that scores text
        tie_margin: Normalized error margin for ties (default 5%)
        config: Optional judge configuration

    Returns:
        Configured OracleJudge instance
    """
    if config is None:
        config = JudgeConfig(type="oracle", tie_margin=tie_margin)
    else:
        config.tie_margin = tie_margin

    return OracleJudge(oracle_fn=oracle_fn, config=config)
