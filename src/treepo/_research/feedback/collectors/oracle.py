"""
Oracle-based feedback collector.

Uses an oracle scoring function to produce scalar scores for single items,
and derives pairwise preferences from score differences.
"""

import asyncio
import logging
from typing import Any, Callable, Optional

from treepo._research.feedback.collector import register_collector
from treepo._research.feedback.types import FeedbackRequest, FeedbackResponse
from treepo._research.core.async_utils import to_thread

logger = logging.getLogger(__name__)


@register_collector("oracle")
class OracleCollector:
    """Feedback collector using an oracle scoring function.

    For single items: returns the oracle score.
    For pairwise items: scores both candidates and derives preference
    from the score difference (lower error = better, or higher score = better).

    Usage:
        collector = OracleCollector(oracle_predict=my_scorer)
        response = collector.collect(request)
    """

    def __init__(
        self,
        oracle_predict: Callable[[str], float],
        tie_margin: float = 0.05,
        prefer_lower: bool = True,
        scale_range: Optional[float] = None,
    ):
        """
        Args:
            oracle_predict: Function that scores text -> float
            tie_margin: Normalized difference below this is a tie
            prefer_lower: If True, lower score is better (error-based).
                If False, higher score is better (quality-based).
            scale_range: Range of the scale for normalization
        """
        self.oracle_predict = oracle_predict
        self.tie_margin = tie_margin
        self.prefer_lower = prefer_lower
        self.scale_range = scale_range

    def collect(
        self,
        request: FeedbackRequest,
        **kwargs: Any,
    ) -> FeedbackResponse:
        """Collect oracle-based feedback.

        For single items (text_b is None): scores text_a and returns scalar.
        For pairwise items: scores both and derives preference.
        """
        score_a = self.oracle_predict(request.text_a)

        if request.text_b is None:
            # Single item: return scalar score
            return FeedbackResponse(
                request_id=request.request_id,
                scores={"score": score_a},
                confidence=1.0,
                reasoning=f"Oracle score: {score_a:.4f}",
                score_estimate_a=score_a,
                source="oracle",
            )

        # Pairwise: score both and derive preference
        score_b = self.oracle_predict(request.text_b)

        # Compute errors relative to reference if available
        if request.reference_score is not None:
            error_a = abs(score_a - request.reference_score)
            error_b = abs(score_b - request.reference_score)
            diff_metric = error_a - error_b  # positive means A is worse
        elif self.prefer_lower:
            diff_metric = score_a - score_b  # positive means A is worse
        else:
            diff_metric = score_b - score_a  # positive means A is better (flip)

        # Normalize if scale_range provided
        if self.scale_range is not None and self.scale_range > 0:
            diff_metric = diff_metric / self.scale_range

        # Determine preference
        if abs(diff_metric) <= self.tie_margin:
            preferred = "tie"
            confidence = 0.5
            reasoning = f"Tie: difference {diff_metric:.4f} within margin {self.tie_margin}"
        elif diff_metric > 0:
            # A is worse (higher error or lower quality)
            preferred = "B"
            confidence = min(0.95, 0.5 + abs(diff_metric) * 2)
            reasoning = f"B preferred: diff={diff_metric:.4f}"
        else:
            preferred = "A"
            confidence = min(0.95, 0.5 + abs(diff_metric) * 2)
            reasoning = f"A preferred: diff={diff_metric:.4f}"

        return FeedbackResponse(
            request_id=request.request_id,
            preferred=preferred,
            scores={"score_a": score_a, "score_b": score_b},
            confidence=confidence,
            reasoning=reasoning,
            score_estimate_a=score_a,
            score_estimate_b=score_b,
            source="oracle",
            extra={
                "reference_score": request.reference_score,
                "diff_metric": diff_metric,
            },
        )

    async def collect_async(
        self,
        request: FeedbackRequest,
        **kwargs: Any,
    ) -> FeedbackResponse:
        return await to_thread(self.collect, request, **kwargs)
