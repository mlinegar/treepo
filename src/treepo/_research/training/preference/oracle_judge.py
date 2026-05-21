"""
Oracle-backed pairwise judge for tournament preference collection.

This provides a GenRM-free comparison backend by scoring summaries against
the task oracle/scorer and deriving pairwise preference from oracle error.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Optional

from treepo._research.core.async_utils import to_thread

@dataclass
class OracleJudgeResult:
    """Tournament-compatible pairwise result payload."""

    preferred: str
    reasoning: str
    confidence: float
    ranking_score: int
    score_estimate_a: float
    score_estimate_b: float
    helpfulness_a: float
    helpfulness_b: float
    error_message: Optional[str] = None
    error_type: Optional[str] = None


class OraclePairwiseJudge:
    """Pairwise judge that compares summaries by oracle-alignment error."""

    judge_backend = "oracle_scorer"

    def __init__(
        self,
        oracle_predict: Callable[[str], float],
        *,
        tie_margin: float = 0.01,
        score_range: float = 1.0,
    ) -> None:
        self.oracle_predict = oracle_predict
        self.tie_margin = max(0.0, float(tie_margin))
        self.score_range = max(1e-8, float(score_range))

    @staticmethod
    def _bounded_helpfulness(score_01: float) -> float:
        return max(1.0, min(5.0, 1.0 + 4.0 * float(score_01)))

    def compare(
        self,
        *,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
    ) -> OracleJudgeResult:
        del context, law_type  # Oracle path compares score preservation directly.
        try:
            reference = float(self.oracle_predict(str(original_text or "")))
            score_a = float(self.oracle_predict(str(summary_a or "")))
            score_b = float(self.oracle_predict(str(summary_b or "")))
            error_a = abs(score_a - reference)
            error_b = abs(score_b - reference)
            error_diff = error_a - error_b

            if abs(error_diff) <= self.tie_margin:
                preferred = "tie"
                ranking = 3
                confidence = 0.5
                reasoning = (
                    f"Oracle tie: errors within margin ({error_a:.4f} vs {error_b:.4f}, "
                    f"margin={self.tie_margin:.4f})."
                )
            elif error_diff < 0.0:
                preferred = "A"
                ranking = 1
                confidence = min(0.99, 0.5 + abs(error_diff) / self.score_range)
                reasoning = (
                    f"A preferred by oracle alignment: error_a={error_a:.4f} "
                    f"< error_b={error_b:.4f}."
                )
            else:
                preferred = "B"
                ranking = 6
                confidence = min(0.99, 0.5 + abs(error_diff) / self.score_range)
                reasoning = (
                    f"B preferred by oracle alignment: error_b={error_b:.4f} "
                    f"< error_a={error_a:.4f}."
                )

            return OracleJudgeResult(
                preferred=preferred,
                reasoning=reasoning,
                confidence=float(confidence),
                ranking_score=int(ranking),
                score_estimate_a=float(score_a),
                score_estimate_b=float(score_b),
                helpfulness_a=self._bounded_helpfulness(score_a),
                helpfulness_b=self._bounded_helpfulness(score_b),
            )
        except Exception as exc:
            return OracleJudgeResult(
                preferred="tie",
                reasoning=f"Oracle judge error fallback: {exc}",
                confidence=0.5,
                ranking_score=3,
                score_estimate_a=0.5,
                score_estimate_b=0.5,
                helpfulness_a=3.0,
                helpfulness_b=3.0,
                error_message=str(exc),
                error_type=type(exc).__name__,
            )

    async def compare_async(
        self,
        *,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
    ) -> OracleJudgeResult:
        return await to_thread(
            self.compare,
            context=context,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            law_type=law_type,
        )


__all__ = ["OracleJudgeResult", "OraclePairwiseJudge"]
