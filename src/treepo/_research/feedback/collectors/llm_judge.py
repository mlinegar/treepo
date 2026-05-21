"""
LLM-based feedback collector.

Uses an LLM (PairwiseJudge, GenRM, or generic DSPy module) to produce
multi-dimensional feedback. Can produce preference + scores + critique
simultaneously from a single LLM call.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from treepo._research.feedback.collector import register_collector
from treepo._research.feedback.types import FeedbackRequest, FeedbackResponse
from treepo._research.core.async_utils import to_thread

logger = logging.getLogger(__name__)


@register_collector("llm_judge")
class LLMJudgeCollector:
    """Feedback collector using an LLM judge.

    Wraps PairwiseJudge, GenRM, or any callable that accepts comparison
    arguments and returns a dict with preference/reasoning/scores.

    For pairwise requests: produces preference + scores + reasoning.
    For single-item requests: produces a scalar score + critique via
    a self-contained rating prompt.

    Usage:
        # With PairwiseJudge
        from treepo._research.training.supervision import PairwiseJudge
        collector = LLMJudgeCollector(judge=PairwiseJudge())

        # With GenRM
        from treepo._research.training.preference.genrm_dspy import GenRMComparisonModule
        collector = LLMJudgeCollector(judge=genrm_module, judge_type="genrm")

        # With any callable
        collector = LLMJudgeCollector(judge=my_judge_fn, judge_type="callable")
    """

    def __init__(
        self,
        judge: Optional[Any] = None,
        judge_type: str = "pairwise",
        model: str = "",
        use_cot: bool = True,
    ):
        """
        Args:
            judge: Judge module/callable. If None, creates a PairwiseJudge.
            judge_type: One of "pairwise", "genrm", "callable".
                - "pairwise": expects PairwiseJudge.forward() interface
                - "genrm": expects GenRMComparisonModule.forward() interface
                - "callable": expects (text_a, text_b, rubric, original_text) -> dict
            model: Model name for metadata.
            use_cot: Whether to use chain-of-thought (for PairwiseJudge).
        """
        self.judge = judge
        self.judge_type = judge_type
        self.model = model
        self.use_cot = use_cot

    def _ensure_judge(self) -> Any:
        """Lazily create a PairwiseJudge if no judge was provided."""
        if self.judge is None:
            from treepo._research.training.supervision import PairwiseJudge
            self.judge = PairwiseJudge(use_cot=self.use_cot)
            self.judge_type = "pairwise"
        return self.judge

    def collect(
        self,
        request: FeedbackRequest,
        **kwargs: Any,
    ) -> FeedbackResponse:
        """Collect LLM-based feedback.

        For pairwise requests: compares text_a vs text_b.
        For single-item requests: rates text_a against the rubric.
        """
        judge = self._ensure_judge()

        if request.is_pairwise and request.text_b is not None:
            return self._collect_pairwise(judge, request, **kwargs)
        else:
            return self._collect_single(judge, request, **kwargs)

    def _collect_pairwise(
        self,
        judge: Any,
        request: FeedbackRequest,
        **kwargs: Any,
    ) -> FeedbackResponse:
        """Handle pairwise comparison."""
        if self.judge_type == "genrm":
            result = judge.forward(
                context=request.rubric,
                original_text=request.original_text,
                summary_a=request.text_a,
                summary_b=request.text_b,
                law_type=request.law_type,
            )
            # GenRM returns an object with .preference, .reasoning, etc.
            preferred = getattr(result, "preference", "tie")
            reasoning = getattr(result, "reasoning", "")
            try:
                ranking_score = float(getattr(result, "ranking_score", 3.5))
                confidence = abs(ranking_score - 3.5) / 2.5
            except (ValueError, TypeError):
                confidence = 0.5
            score_a = getattr(result, "helpfulness_a", None)
            score_b = getattr(result, "helpfulness_b", None)
            raw_result = result

        elif self.judge_type == "callable":
            result = judge(
                request.text_a,
                request.text_b,
                request.rubric,
                request.original_text,
                **kwargs,
            )
            if isinstance(result, dict):
                preferred = result.get("preferred", "tie")
                reasoning = result.get("reasoning", "")
                confidence = result.get("confidence", 0.5)
                score_a = result.get("score_estimate_a")
                score_b = result.get("score_estimate_b")
                raw_result = result
            else:
                preferred = "tie"
                reasoning = str(result)
                confidence = 0.5
                score_a = score_b = None
                raw_result = result

        else:
            # Default: PairwiseJudge interface
            result = judge.forward(
                original_text=request.original_text,
                summary_a=request.text_a,
                summary_b=request.text_b,
                rubric=request.rubric,
                reference_score=request.reference_score or 0.0,
            )
            preferred = result.get("preferred", "tie")
            reasoning = result.get("reasoning", "")
            confidence = result.get("confidence", 0.5)
            score_a = result.get("score_estimate_a")
            score_b = result.get("score_estimate_b")
            raw_result = result

        scores: Dict[str, float] = {}
        if score_a is not None:
            scores["score_a"] = float(score_a)
        if score_b is not None:
            scores["score_b"] = float(score_b)

        extra: Dict[str, Any] = {}
        if self.judge_type == "genrm":
            try:
                extra["comparison_signal_value"] = float(getattr(result, "ranking_score"))
            except (AttributeError, TypeError, ValueError):
                pass
            extra.update({
                "comparison_signal_name": "genrm_ranking_score",
                "comparison_signal_min": 1.0,
                "comparison_signal_max": 6.0,
                "response_signal_name": "genrm_helpfulness",
                "response_signal_min": 1.0,
                "response_signal_max": 5.0,
            })
        elif score_a is not None or score_b is not None:
            extra["response_signal_name"] = "judge_score_estimate"

        return FeedbackResponse(
            request_id=request.request_id,
            preferred=preferred,
            scores=scores,
            critique="",
            reasoning=reasoning,
            confidence=confidence,
            score_estimate_a=score_a,
            score_estimate_b=score_b,
            extra=extra,
            source="llm_judge",
            judge_model=self.model,
            raw_result=raw_result,
        )

    def _collect_single(
        self,
        judge: Any,
        request: FeedbackRequest,
        **kwargs: Any,
    ) -> FeedbackResponse:
        """Handle single-item rating.

        Uses a self-comparison trick: compare text_a against original_text
        to get a quality estimate, or use a callable that supports single-item.
        """
        if self.judge_type == "callable":
            result = judge(
                request.text_a,
                None,
                request.rubric,
                request.original_text,
                **kwargs,
            )
            if isinstance(result, dict):
                return FeedbackResponse(
                    request_id=request.request_id,
                    scores={"score": result.get("score", 0.5)},
                    critique=result.get("feedback", result.get("critique", "")),
                    reasoning=result.get("reasoning", ""),
                    confidence=result.get("confidence", 0.5),
                    score_estimate_a=result.get("score"),
                    source="llm_judge",
                    judge_model=self.model,
                    raw_result=result,
                )

        # Fallback: use pairwise judge to compare text_a vs original
        # This gives us a "faithfulness" estimate
        if request.original_text:
            result = judge.forward(
                original_text=request.original_text,
                summary_a=request.text_a,
                summary_b=request.original_text,
                rubric=request.rubric,
                reference_score=request.reference_score or 0.0,
            )
            score_a = result.get("score_estimate_a")
            return FeedbackResponse(
                request_id=request.request_id,
                scores={"score": float(score_a) if score_a is not None else 0.5},
                critique="",
                reasoning=result.get("reasoning", ""),
                confidence=result.get("confidence", 0.5),
                score_estimate_a=score_a,
                source="llm_judge",
                judge_model=self.model,
                raw_result=result,
            )

        # No original text available -- return neutral
        return FeedbackResponse(
            request_id=request.request_id,
            scores={"score": 0.5},
            reasoning="No original text available for single-item rating",
            source="llm_judge",
            judge_model=self.model,
        )

    async def collect_async(
        self,
        request: FeedbackRequest,
        **kwargs: Any,
    ) -> FeedbackResponse:
        return await to_thread(self.collect, request, **kwargs)
