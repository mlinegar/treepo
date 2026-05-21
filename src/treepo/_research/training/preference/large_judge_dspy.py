"""DSPy pairwise judge module backed by the currently configured large LLM."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

import dspy

from treepo._research.ctreepo.sim.util import safe_float


def _normalize_preference(raw_preference: str) -> str:
    rendered = str(raw_preference or "").strip().lower()
    if not rendered:
        return "tie"
    if "tie" in rendered or "equal" in rendered or "neither" in rendered or "both" in rendered:
        return "tie"
    if rendered == "a" or rendered.startswith("a ") or "summary a" in rendered or "response a" in rendered:
        return "A"
    if rendered == "b" or rendered.startswith("b ") or "summary b" in rendered or "response b" in rendered:
        return "B"
    if rendered in {"1", "response 1", "summary 1"}:
        return "A"
    if rendered in {"2", "response 2", "summary 2"}:
        return "B"
    if "a" in rendered and "b" not in rendered:
        return "A"
    if "b" in rendered and "a" not in rendered:
        return "B"
    return "tie"


_safe_float = safe_float


def _ordered_candidate_ids(raw_ranking: object, valid_ids: List[str]) -> List[str]:
    if isinstance(raw_ranking, list):
        tokens = [str(value).strip().upper() for value in raw_ranking]
    else:
        rendered = str(raw_ranking or "")
        tokens = [token.upper() for token in re.findall(r"C\d+", rendered)]
        if not tokens:
            tokens = [f"C{int(token)}" for token in re.findall(r"\d+", rendered)]

    ordered: List[str] = []
    seen = set()
    valid = {candidate_id.upper() for candidate_id in valid_ids}
    for token in tokens:
        if token in valid and token not in seen:
            seen.add(token)
            ordered.append(token)
    for candidate_id in valid_ids:
        upper_id = candidate_id.upper()
        if upper_id not in seen:
            ordered.append(upper_id)
    return ordered


def _candidate_score_map(raw_scores: object, valid_ids: List[str]) -> Dict[str, float]:
    valid = {candidate_id.upper() for candidate_id in valid_ids}
    payload: Dict[str, object] = {}
    if isinstance(raw_scores, dict):
        payload = dict(raw_scores)
    else:
        rendered = str(raw_scores or "").strip()
        if rendered:
            try:
                parsed = json.loads(rendered)
                if isinstance(parsed, dict):
                    payload = dict(parsed)
            except Exception:
                payload = {
                    candidate_id.upper(): value
                    for candidate_id, value in re.findall(
                        r"(C\d+|\d+)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
                        rendered,
                    )
                }

    scores: Dict[str, float] = {}
    for key, value in payload.items():
        token = str(key).strip().upper()
        if token.isdigit():
            token = f"C{token}"
        if token not in valid:
            continue
        try:
            scores[token] = float(value)
        except (TypeError, ValueError):
            continue
    return scores


class LargeJudgeComparisonSignature(dspy.Signature):
    """Pairwise comparison signature for large-model judging."""

    context: str = dspy.InputField(desc="What information must be preserved.")
    original_text: str = dspy.InputField(desc="Original source text.")
    summary_a: str = dspy.InputField(desc="Candidate summary A.")
    summary_b: str = dspy.InputField(desc="Candidate summary B.")
    law_type: str = dspy.InputField(desc="Local law type: sufficiency, idempotence, merge.")

    preference: str = dspy.OutputField(desc="Preferred summary: A, B, or tie.")
    reasoning: str = dspy.OutputField(desc="Short comparison rationale.")
    score_a: str = dspy.OutputField(desc="Helpfulness score for summary A (1-5).")
    score_b: str = dspy.OutputField(desc="Helpfulness score for summary B (1-5).")
    confidence: str = dspy.OutputField(desc="Confidence in [0,1].")


class LargeJudgeListwiseSignature(dspy.Signature):
    """Listwise comparison signature for large-model judging."""

    context: str = dspy.InputField(desc="What information must be preserved.")
    original_text: str = dspy.InputField(desc="Original source text.")
    candidates_text: str = dspy.InputField(
        desc="Numbered candidate summaries C1..CK to rank jointly."
    )
    law_type: str = dspy.InputField(desc="Local law type: sufficiency, idempotence, merge.")

    ordered_candidates: str = dspy.OutputField(
        desc="Candidate identifiers from best to worst, e.g. C2 > C1 > C3."
    )
    candidate_scores_json: str = dspy.OutputField(
        desc="JSON object mapping candidate identifiers to comparable numeric scores."
    )
    reasoning: str = dspy.OutputField(desc="Short rationale for the overall ordering.")
    confidence: str = dspy.OutputField(desc="Confidence in [0,1].")


class LargeJudgeComparisonModule(dspy.Module):
    """Tournament-compatible pairwise judge using the active DSPy LM."""

    def __init__(self, *, use_cot: bool = True):
        super().__init__()
        self.use_dspy_predictor = True
        self.use_dspy_prompt = False
        self.judge_backend = "large_qwen"
        self.compare = (
            dspy.ChainOfThought(LargeJudgeComparisonSignature)
            if bool(use_cot)
            else dspy.Predict(LargeJudgeComparisonSignature)
        )

    def forward(
        self,
        *,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
    ) -> dspy.Prediction:
        result = self.compare(
            context=str(context or ""),
            original_text=str(original_text or ""),
            summary_a=str(summary_a or ""),
            summary_b=str(summary_b or ""),
            law_type=str(law_type or "sufficiency"),
        )

        preference = _normalize_preference(getattr(result, "preference", "tie"))
        score_a = _safe_float(getattr(result, "score_a", 3.0), 3.0)
        score_b = _safe_float(getattr(result, "score_b", 3.0), 3.0)
        score_a = min(5.0, max(1.0, score_a))
        score_b = min(5.0, max(1.0, score_b))
        parsed_confidence = _safe_float(getattr(result, "confidence", 0.5), 0.5)
        score_diff = abs(score_a - score_b)
        derived_confidence = min(1.0, 0.5 + score_diff * 0.125)
        if preference == "tie":
            confidence = 0.5
        else:
            confidence = max(0.5, min(1.0, max(parsed_confidence, derived_confidence)))

        if preference == "A":
            ranking_score = 1
        elif preference == "B":
            ranking_score = 6
        else:
            ranking_score = 3

        return dspy.Prediction(
            preference=preference,
            reasoning=str(getattr(result, "reasoning", "") or ""),
            confidence=confidence,
            score_a=str(score_a),
            score_b=str(score_b),
            helpfulness_a=score_a,
            helpfulness_b=score_b,
            ranking_score=ranking_score,
        )


class LargeJudgeListwiseModule(dspy.Module):
    """DSPy listwise judge using the active LM to rank K candidate summaries."""

    def __init__(self, *, use_cot: bool = True):
        super().__init__()
        self.use_dspy_predictor = True
        self.use_dspy_prompt = False
        self.judge_backend = "large_qwen_listwise"
        self.compare = (
            dspy.ChainOfThought(LargeJudgeListwiseSignature)
            if bool(use_cot)
            else dspy.Predict(LargeJudgeListwiseSignature)
        )

    @staticmethod
    def _render_candidates(candidate_summaries: List[str]) -> str:
        blocks: List[str] = []
        for idx, summary in enumerate(candidate_summaries, start=1):
            blocks.append(f"C{idx}:\n{str(summary or '').strip()}")
        return "\n\n".join(blocks)

    def rank_candidates(
        self,
        *,
        context: str,
        original_text: str,
        candidate_summaries: List[str],
        law_type: str = "sufficiency",
    ) -> Dict[str, object]:
        valid_ids = [f"C{idx}" for idx in range(1, len(candidate_summaries) + 1)]
        if len(valid_ids) < 2:
            raise ValueError("Need at least two candidates for listwise ranking")

        result = self.compare(
            context=str(context or ""),
            original_text=str(original_text or ""),
            candidates_text=self._render_candidates(candidate_summaries),
            law_type=str(law_type or "sufficiency"),
        )

        ordered_candidate_ids = _ordered_candidate_ids(
            getattr(result, "ordered_candidates", ""),
            valid_ids,
        )
        candidate_scores = _candidate_score_map(
            getattr(result, "candidate_scores_json", ""),
            valid_ids,
        )
        if candidate_scores:
            ordered_candidate_ids = sorted(
                ordered_candidate_ids,
                key=lambda candidate_id: (
                    -float(candidate_scores.get(candidate_id, float("-inf"))),
                    ordered_candidate_ids.index(candidate_id),
                ),
            )
        confidence = _safe_float(getattr(result, "confidence", 0.5), 0.5)
        confidence = min(1.0, max(0.0, confidence))
        return {
            "ordered_candidate_ids": ordered_candidate_ids,
            "candidate_scores": candidate_scores,
            "reasoning": str(getattr(result, "reasoning", "") or ""),
            "confidence": confidence,
            "response_signal_name": "listwise_candidate_score",
        }

    def forward(
        self,
        *,
        context: str,
        original_text: str,
        candidate_summaries: List[str],
        law_type: str = "sufficiency",
    ) -> dspy.Prediction:
        payload = self.rank_candidates(
            context=context,
            original_text=original_text,
            candidate_summaries=candidate_summaries,
            law_type=law_type,
        )
        return dspy.Prediction(**payload)


__all__ = [
    "LargeJudgeComparisonModule",
    "LargeJudgeListwiseModule",
]
