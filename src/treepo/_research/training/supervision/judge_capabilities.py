"""Shared judge capability adapters for pairwise and comparative supervision."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from treepo._research.core.async_utils import to_thread

def _call_with_supported_kwargs(method: Any, **kwargs: Any) -> Any:
    """Call a judge method with only the kwargs it declares."""
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(**kwargs)

    params = signature.parameters
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in params.values()
    )
    if accepts_var_kwargs:
        return method(**kwargs)

    filtered = {
        key: value
        for key, value in kwargs.items()
        if key in params
    }
    return method(**filtered)


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_preference(raw_value: Any) -> str:
    rendered = str(raw_value or "").strip().lower()
    if not rendered:
        return "tie"
    if rendered in {"a", "summary a", "response a", "1", "response 1", "summary 1"}:
        return "A"
    if rendered in {"b", "summary b", "response b", "2", "response 2", "summary 2"}:
        return "B"
    if rendered in {"tie", "equal", "neither", "both"}:
        return "tie"
    if "tie" in rendered or "equal" in rendered or "neither" in rendered or "both" in rendered:
        return "tie"
    if "a" in rendered and "b" not in rendered:
        return "A"
    if "b" in rendered and "a" not in rendered:
        return "B"
    return "tie"


def _judge_payload_to_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)

    result: Dict[str, Any] = {}
    for key in (
        "preferred",
        "preference",
        "reasoning",
        "confidence",
        "score_estimate_a",
        "score_estimate_b",
        "helpfulness_a",
        "helpfulness_b",
        "score_a",
        "score_b",
        "ranking_score",
        "comparison_signal_name",
        "comparison_signal_value",
        "comparison_signal_min",
        "comparison_signal_max",
        "response_signal_name",
        "response_signal_min",
        "response_signal_max",
        "ordered_candidate_ids",
        "candidate_scores",
        "candidate_scores_json",
        "error_message",
        "error_type",
    ):
        value = getattr(payload, key, None)
        if value is not None:
            result[key] = value
    return result


@dataclass(frozen=True)
class PairwiseJudgeResult:
    """Normalized pairwise judgment payload."""

    preferred: str
    reasoning: str = ""
    confidence: float = 0.5
    score_estimate_a: Optional[float] = None
    score_estimate_b: Optional[float] = None
    comparison_signal_name: Optional[str] = None
    comparison_signal_value: Optional[float] = None
    comparison_signal_min: Optional[float] = None
    comparison_signal_max: Optional[float] = None
    response_signal_name: Optional[str] = None
    response_signal_min: Optional[float] = None
    response_signal_max: Optional[float] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    raw_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparativeJudgeResult:
    """Normalized comparative judgment payload."""

    ordered_candidate_ids: List[str]
    candidate_scores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.5
    comparison_signal_name: Optional[str] = None
    comparison_signal_value: Optional[float] = None
    comparison_signal_min: Optional[float] = None
    comparison_signal_max: Optional[float] = None
    response_signal_name: Optional[str] = None
    response_signal_min: Optional[float] = None
    response_signal_max: Optional[float] = None
    raw_payload: Dict[str, Any] = field(default_factory=dict)


def judge_backend_name(judge: Any) -> str:
    """Return a stable backend label for a judge-like object."""
    if getattr(judge, "judge_backend", None):
        return str(getattr(judge, "judge_backend"))
    if getattr(judge, "model_name", None):
        return str(getattr(judge, "model_name"))
    if hasattr(judge, "use_dspy_prompt") and getattr(judge, "use_dspy_prompt", False):
        return "dspy-prompt-tuned"
    if hasattr(judge, "use_dspy_predictor") and getattr(judge, "use_dspy_predictor", False):
        return "dspy-optimizable"
    return type(judge).__name__


def supports_pairwise_judging(judge: Any) -> bool:
    """Whether a judge can directly produce a binary comparison."""
    return any(
        hasattr(judge, attribute)
        for attribute in ("compare", "compare_async", "forward")
    ) or callable(judge)


def supports_direct_comparative_judging(judge: Any) -> bool:
    """Whether a judge can directly rank/score multiple responses jointly."""
    return any(
        hasattr(judge, attribute)
        for attribute in ("rank_candidates", "rank_candidates_async", "score_responses")
    )


def _normalize_pairwise_result(payload: Any) -> PairwiseJudgeResult:
    raw = _judge_payload_to_dict(payload)
    preferred = _normalize_preference(raw.get("preference", raw.get("preferred")))

    score_estimate_a = _safe_float(
        raw.get("score_estimate_a", raw.get("helpfulness_a", raw.get("score_a")))
    )
    score_estimate_b = _safe_float(
        raw.get("score_estimate_b", raw.get("helpfulness_b", raw.get("score_b")))
    )
    confidence = _safe_float(raw.get("confidence"))
    if confidence is None:
        confidence = 0.5
    confidence = max(0.0, min(1.0, float(confidence)))

    comparison_signal_name = raw.get("comparison_signal_name")
    comparison_signal_value = _safe_float(raw.get("comparison_signal_value"))
    comparison_signal_min = _safe_float(raw.get("comparison_signal_min"))
    comparison_signal_max = _safe_float(raw.get("comparison_signal_max"))
    if comparison_signal_value is None and raw.get("ranking_score") is not None:
        comparison_signal_value = _safe_float(raw.get("ranking_score"))
        if comparison_signal_name is None:
            comparison_signal_name = "ranking_score"
        if comparison_signal_min is None:
            comparison_signal_min = 1.0
        if comparison_signal_max is None:
            comparison_signal_max = 6.0

    response_signal_name = raw.get("response_signal_name")
    response_signal_min = _safe_float(raw.get("response_signal_min"))
    response_signal_max = _safe_float(raw.get("response_signal_max"))
    if response_signal_name is None and (
        score_estimate_a is not None or score_estimate_b is not None
    ):
        response_signal_name = "response_score"

    return PairwiseJudgeResult(
        preferred=preferred,
        reasoning=str(raw.get("reasoning", "") or ""),
        confidence=confidence,
        score_estimate_a=score_estimate_a,
        score_estimate_b=score_estimate_b,
        comparison_signal_name=(
            str(comparison_signal_name)
            if comparison_signal_name is not None
            else None
        ),
        comparison_signal_value=comparison_signal_value,
        comparison_signal_min=comparison_signal_min,
        comparison_signal_max=comparison_signal_max,
        response_signal_name=(
            str(response_signal_name)
            if response_signal_name is not None
            else None
        ),
        response_signal_min=response_signal_min,
        response_signal_max=response_signal_max,
        error_message=(
            str(raw.get("error_message"))
            if raw.get("error_message") is not None
            else None
        ),
        error_type=(
            str(raw.get("error_type"))
            if raw.get("error_type") is not None
            else None
        ),
        raw_payload=raw,
    )


def invoke_pairwise_judgment_sync(
    judge: Any,
    *,
    context: str,
    original_text: str,
    summary_a: str,
    summary_b: str,
    law_type: str = "sufficiency",
    reference_score: float = 0.0,
    extra_context: Optional[str] = None,
) -> PairwiseJudgeResult:
    """Invoke a judge through the shared pairwise comparison surface."""
    if hasattr(judge, "compare"):
        payload = _call_with_supported_kwargs(
            judge.compare,
            context=context,
            rubric=context,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            law_type=law_type,
            reference_score=reference_score,
            extra_context=extra_context,
        )
        return _normalize_pairwise_result(payload)

    if hasattr(judge, "forward"):
        payload = _call_with_supported_kwargs(
            judge.forward,
            context=context,
            rubric=context,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            law_type=law_type,
            reference_score=reference_score,
            extra_context=extra_context,
        )
        return _normalize_pairwise_result(payload)

    if callable(judge):
        payload = _call_with_supported_kwargs(
            judge,
            context=context,
            rubric=context,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            law_type=law_type,
            reference_score=reference_score,
            extra_context=extra_context,
        )
        return _normalize_pairwise_result(payload)

    if supports_direct_comparative_judging(judge):
        comparative = invoke_comparative_judgment_sync(
            judge,
            context=context,
            original_text=original_text,
            candidate_summaries=[summary_a, summary_b],
            law_type=law_type,
            reference_score=reference_score,
        )
        order = list(comparative.ordered_candidate_ids or [])
        preferred = "tie"
        if len(order) >= 2:
            if order[0] == "C1" and order[1] == "C2":
                preferred = "A"
            elif order[0] == "C2" and order[1] == "C1":
                preferred = "B"
        score_a = comparative.candidate_scores.get("C1")
        score_b = comparative.candidate_scores.get("C2")
        comparison_signal_value = None
        comparison_signal_name = comparative.comparison_signal_name
        if score_a is not None and score_b is not None:
            comparison_signal_name = comparison_signal_name or "projected_response_signal_margin"
            comparison_signal_value = float(score_a) - float(score_b)
        return PairwiseJudgeResult(
            preferred=preferred,
            reasoning=comparative.reasoning,
            confidence=comparative.confidence,
            score_estimate_a=score_a,
            score_estimate_b=score_b,
            comparison_signal_name=comparison_signal_name,
            comparison_signal_value=comparison_signal_value,
            comparison_signal_min=comparative.comparison_signal_min,
            comparison_signal_max=comparative.comparison_signal_max,
            response_signal_name=comparative.response_signal_name,
            response_signal_min=comparative.response_signal_min,
            response_signal_max=comparative.response_signal_max,
            raw_payload=dict(comparative.raw_payload),
        )

    raise ValueError(f"Judge {judge!r} does not support pairwise comparison")


async def invoke_pairwise_judgment_async(
    judge: Any,
    *,
    context: str,
    original_text: str,
    summary_a: str,
    summary_b: str,
    law_type: str = "sufficiency",
    reference_score: float = 0.0,
    extra_context: Optional[str] = None,
) -> PairwiseJudgeResult:
    """Async pairwise invocation over the shared judge capability surface."""
    if hasattr(judge, "compare_async"):
        payload = await _call_with_supported_kwargs(
            judge.compare_async,
            context=context,
            rubric=context,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            law_type=law_type,
            reference_score=reference_score,
            extra_context=extra_context,
        )
        return _normalize_pairwise_result(payload)

    return await to_thread(
        invoke_pairwise_judgment_sync,
        judge,
        context=context,
        original_text=original_text,
        summary_a=summary_a,
        summary_b=summary_b,
        law_type=law_type,
        reference_score=reference_score,
        extra_context=extra_context,
    )


def _normalize_comparative_result(
    payload: Any,
    *,
    valid_ids: Sequence[str],
) -> ComparativeJudgeResult:
    raw = _judge_payload_to_dict(payload)
    ordered = [str(value).upper() for value in list(raw.get("ordered_candidate_ids", []) or [])]
    if not ordered:
        ordered = [str(value).upper() for value in list(raw.get("ordered_candidates", []) or [])]
    valid_ids = [str(candidate_id).upper() for candidate_id in valid_ids]
    normalized_order: List[str] = []
    seen = set()
    for candidate_id in ordered:
        if candidate_id in valid_ids and candidate_id not in seen:
            normalized_order.append(candidate_id)
            seen.add(candidate_id)
    for candidate_id in valid_ids:
        if candidate_id not in seen:
            normalized_order.append(candidate_id)

    raw_scores = raw.get("candidate_scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    candidate_scores: Dict[str, float] = {}
    for candidate_id, score in raw_scores.items():
        parsed = _safe_float(score)
        token = str(candidate_id).upper()
        if token in valid_ids and parsed is not None:
            candidate_scores[token] = parsed
    if candidate_scores:
        original_positions = {
            candidate_id: index
            for index, candidate_id in enumerate(normalized_order)
        }
        normalized_order = sorted(
            normalized_order,
            key=lambda candidate_id: (
                -float(candidate_scores.get(candidate_id, float("-inf"))),
                original_positions.get(candidate_id, len(original_positions)),
            ),
        )

    confidence = _safe_float(raw.get("confidence"))
    if confidence is None:
        confidence = 0.5
    confidence = max(0.0, min(1.0, float(confidence)))

    return ComparativeJudgeResult(
        ordered_candidate_ids=normalized_order,
        candidate_scores=candidate_scores,
        reasoning=str(raw.get("reasoning", "") or ""),
        confidence=confidence,
        comparison_signal_name=(
            str(raw.get("comparison_signal_name"))
            if raw.get("comparison_signal_name") is not None
            else None
        ),
        comparison_signal_value=_safe_float(raw.get("comparison_signal_value")),
        comparison_signal_min=_safe_float(raw.get("comparison_signal_min")),
        comparison_signal_max=_safe_float(raw.get("comparison_signal_max")),
        response_signal_name=(
            str(raw.get("response_signal_name"))
            if raw.get("response_signal_name") is not None
            else None
        ),
        response_signal_min=_safe_float(raw.get("response_signal_min")),
        response_signal_max=_safe_float(raw.get("response_signal_max")),
        raw_payload=raw,
    )


def _comparative_from_pairwise_fallback(
    judge: Any,
    *,
    context: str,
    original_text: str,
    candidate_summaries: Sequence[str],
    law_type: str = "sufficiency",
    reference_score: float = 0.0,
) -> ComparativeJudgeResult:
    valid_ids = [f"C{index}" for index in range(1, len(candidate_summaries) + 1)]
    if len(valid_ids) < 2:
        raise ValueError("Need at least two candidates for comparative judging")

    wins = {candidate_id: 0.0 for candidate_id in valid_ids}
    score_values: Dict[str, List[float]] = {candidate_id: [] for candidate_id in valid_ids}
    confidences: List[float] = []
    reasonings: List[str] = []
    response_signal_name: Optional[str] = None
    response_signal_min: Optional[float] = None
    response_signal_max: Optional[float] = None

    for left_index in range(len(candidate_summaries)):
        for right_index in range(left_index + 1, len(candidate_summaries)):
            result = invoke_pairwise_judgment_sync(
                judge,
                context=context,
                original_text=original_text,
                summary_a=str(candidate_summaries[left_index] or ""),
                summary_b=str(candidate_summaries[right_index] or ""),
                law_type=law_type,
                reference_score=reference_score,
            )
            left_id = valid_ids[left_index]
            right_id = valid_ids[right_index]
            confidence = max(0.0, min(1.0, float(result.confidence)))
            confidences.append(confidence)
            if result.reasoning:
                reasonings.append(str(result.reasoning))

            if result.score_estimate_a is not None:
                score_values[left_id].append(float(result.score_estimate_a))
            if result.score_estimate_b is not None:
                score_values[right_id].append(float(result.score_estimate_b))
            if result.response_signal_name and response_signal_name is None:
                response_signal_name = str(result.response_signal_name)
                response_signal_min = result.response_signal_min
                response_signal_max = result.response_signal_max

            if result.preferred == "A":
                wins[left_id] += 0.5 + 0.5 * confidence
                wins[right_id] += 0.5 - 0.5 * confidence
            elif result.preferred == "B":
                wins[left_id] += 0.5 - 0.5 * confidence
                wins[right_id] += 0.5 + 0.5 * confidence
            else:
                wins[left_id] += 0.5
                wins[right_id] += 0.5

    candidate_scores: Dict[str, float] = {}
    use_response_scores = any(values for values in score_values.values())
    for candidate_id in valid_ids:
        if score_values[candidate_id]:
            candidate_scores[candidate_id] = (
                sum(score_values[candidate_id]) / len(score_values[candidate_id])
            )
        else:
            candidate_scores[candidate_id] = wins[candidate_id]

    if not use_response_scores and response_signal_name is None:
        response_signal_name = "pairwise_win_score"
    ordered_ids = sorted(
        valid_ids,
        key=lambda candidate_id: (
            -float(candidate_scores.get(candidate_id, float("-inf"))),
            candidate_id,
        ),
    )

    return ComparativeJudgeResult(
        ordered_candidate_ids=ordered_ids,
        candidate_scores=candidate_scores,
        reasoning="\n".join(reasonings[:3]),
        confidence=(
            sum(confidences) / len(confidences)
            if confidences
            else 0.5
        ),
        response_signal_name=response_signal_name,
        response_signal_min=response_signal_min,
        response_signal_max=response_signal_max,
        raw_payload={
            "pairwise_fallback": True,
            "num_pairwise_calls": int(len(confidences)),
        },
    )


def invoke_comparative_judgment_sync(
    judge: Any,
    *,
    context: str,
    original_text: str,
    candidate_summaries: Sequence[str],
    law_type: str = "sufficiency",
    reference_score: float = 0.0,
) -> ComparativeJudgeResult:
    """Invoke a judge through the shared comparative supervision surface."""
    valid_ids = [f"C{index}" for index in range(1, len(candidate_summaries) + 1)]
    if len(valid_ids) < 2:
        raise ValueError("Need at least two candidates for comparative judging")

    if hasattr(judge, "rank_candidates"):
        payload = _call_with_supported_kwargs(
            judge.rank_candidates,
            context=context,
            rubric=context,
            original_text=original_text,
            candidate_summaries=list(candidate_summaries),
            law_type=law_type,
            reference_score=reference_score,
        )
        return _normalize_comparative_result(payload, valid_ids=valid_ids)

    if hasattr(judge, "score_responses"):
        payload = _call_with_supported_kwargs(
            judge.score_responses,
            context=context,
            rubric=context,
            original_text=original_text,
            candidate_summaries=list(candidate_summaries),
            law_type=law_type,
            reference_score=reference_score,
        )
        payload_dict = _judge_payload_to_dict(payload)
        scores = payload_dict.get("candidate_scores", {})
        if not isinstance(scores, dict):
            raise ValueError("score_responses must return a candidate_scores mapping")
        ordered_ids = sorted(
            valid_ids,
            key=lambda candidate_id: (
                -float(scores.get(candidate_id, float("-inf"))),
                candidate_id,
            ),
        )
        payload_dict.setdefault("ordered_candidate_ids", ordered_ids)
        return _normalize_comparative_result(payload_dict, valid_ids=valid_ids)

    if supports_pairwise_judging(judge):
        return _comparative_from_pairwise_fallback(
            judge,
            context=context,
            original_text=original_text,
            candidate_summaries=candidate_summaries,
            law_type=law_type,
            reference_score=reference_score,
        )

    raise ValueError(f"Judge {judge!r} does not support comparative supervision")


async def invoke_comparative_judgment_async(
    judge: Any,
    *,
    context: str,
    original_text: str,
    candidate_summaries: Sequence[str],
    law_type: str = "sufficiency",
    reference_score: float = 0.0,
) -> ComparativeJudgeResult:
    """Async comparative invocation over the shared judge capability surface."""
    valid_ids = [f"C{index}" for index in range(1, len(candidate_summaries) + 1)]
    if hasattr(judge, "rank_candidates_async"):
        payload = await _call_with_supported_kwargs(
            judge.rank_candidates_async,
            context=context,
            rubric=context,
            original_text=original_text,
            candidate_summaries=list(candidate_summaries),
            law_type=law_type,
            reference_score=reference_score,
        )
        return _normalize_comparative_result(payload, valid_ids=valid_ids)

    return await to_thread(
        invoke_comparative_judgment_sync,
        judge,
        context=context,
        original_text=original_text,
        candidate_summaries=list(candidate_summaries),
        law_type=law_type,
        reference_score=reference_score,
    )


__all__ = [
    "ComparativeJudgeResult",
    "PairwiseJudgeResult",
    "invoke_comparative_judgment_async",
    "invoke_comparative_judgment_sync",
    "invoke_pairwise_judgment_async",
    "invoke_pairwise_judgment_sync",
    "judge_backend_name",
    "supports_direct_comparative_judging",
    "supports_pairwise_judging",
]
