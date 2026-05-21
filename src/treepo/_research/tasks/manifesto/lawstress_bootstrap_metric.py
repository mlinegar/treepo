"""GEPA metric for LawStress local-law bootstrap.

The metric is designed to optimize a unified summarizer g to better satisfy:
- C1 (sufficiency): summary preserves doc-level directional signal
- C2 (idempotence): re-summarization is stable and stays on same side of neutral
- C3 (merge/substitution): merge output matches expected aggregate and joint-vs-disjoint agree

This metric uses a learned embedding proxy (synthetic oracle) to cheaply score
summaries during optimization. Teacher evaluation is performed separately.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from treepo._research.ctreepo.sim.util import safe_float
from treepo._research.tasks.manifesto.lawstress_eval import LawStressEvalConfig, RILE_RUBRIC, strict_same_side
from treepo._research.tasks.manifesto.lawstress_generator import length_weighted_mean, normalize_rile


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


_safe_float = lambda v, default=0.0: safe_float(v, default=default)


@dataclass
class LawStressBootstrapObjectiveConfig:
    """How per-example component signals are collapsed into one GEPA scalar."""

    # Bottleneck objective by default: optimize the weakest local-law component.
    aggregate_mode: str = "min"
    softmin_temperature: float = 0.08
    component_floor: float = 0.55


def _normalize_aggregate_mode(mode: str) -> str:
    value = str(mode or "").strip().lower()
    if value == "bottleneck_min":
        return "min"
    valid = {"weighted_mean", "min", "softmin", "floor_then_weighted"}
    if value in valid:
        return value
    return "min"


def _score_from_error(error: float, threshold: float) -> float:
    thr = float(threshold)
    if thr <= 0:
        return 1.0 if float(error) <= 0.0 else 0.0
    return _clamp01(1.0 - float(error) / thr)


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    total_weight = 0.0
    total_value = 0.0
    for value, weight in zip(values, weights):
        w = max(0.0, float(weight))
        total_weight += w
        total_value += w * float(value)
    if total_weight <= 0.0:
        return float(sum(values) / len(values))
    return float(total_value / total_weight)


def _softmin(values: Sequence[float], weights: Sequence[float], temperature: float) -> float:
    if not values:
        return 0.0
    tau = max(1e-6, float(temperature))
    max_arg = max((-float(v) / tau) for v in values)
    weighted_sum = 0.0
    total_weight = 0.0
    for value, weight in zip(values, weights):
        w = max(0.0, float(weight))
        if w <= 0.0:
            continue
        weighted_sum += w * math.exp((-float(value) / tau) - max_arg)
        total_weight += w
    if weighted_sum <= 0.0 or total_weight <= 0.0:
        return float(min(values))
    log_norm = math.log(weighted_sum / total_weight) + max_arg
    return float(-tau * log_norm)


def _aggregate_components(
    values: Sequence[float],
    *,
    weights: Sequence[float],
    objective: LawStressBootstrapObjectiveConfig,
) -> float:
    if not values:
        return 0.0
    clipped = [_clamp01(float(v)) for v in values]
    weight_values = [max(0.0, float(w)) for w in weights]
    if len(weight_values) != len(clipped):
        weight_values = [1.0] * len(clipped)

    mode = _normalize_aggregate_mode(objective.aggregate_mode)
    if mode == "min":
        return float(min(clipped))
    if mode == "softmin":
        return _clamp01(_softmin(clipped, weight_values, float(objective.softmin_temperature)))

    weighted = _clamp01(_weighted_mean(clipped, weight_values))
    if mode == "floor_then_weighted":
        floor = _clamp01(float(objective.component_floor))
        if min(clipped) < floor:
            return float(min(clipped))
    return float(weighted)


def _length_penalty(
    summary: str,
    input_text: str,
    *,
    min_chars: int = 80,
    min_frac: float = 0.05,
    max_frac: float = 0.40,
) -> Tuple[float, List[str]]:
    rendered = str(summary or "").strip()
    source = str(input_text or "")
    if not rendered:
        return 0.50, ["Empty summary."]

    penalty = 0.0
    notes: List[str] = []

    if len(rendered) < int(min_chars):
        penalty += 0.10
        notes.append(f"Too short (<{int(min_chars)} chars).")

    if source:
        ratio = len(rendered) / max(1, len(source))
        if ratio < float(min_frac):
            penalty += 0.10
            notes.append(f"Over-compressed ({ratio:.1%} < {float(min_frac):.1%}).")
        elif ratio > float(max_frac):
            penalty += 0.05
            notes.append(f"Under-compressed ({ratio:.1%} > {float(max_frac):.1%}).")

    if len(rendered.split()) < 10:
        penalty += 0.05
        notes.append("Very few words (<10).")

    return min(0.35, float(penalty)), notes


def _proxy_score_texts(proxy_model: Any, embedding_client: Any, texts: Sequence[str]) -> List[float]:
    embeddings = embedding_client.embed_texts([str(t or "") for t in texts])
    scores: List[float] = []
    predict = getattr(proxy_model, "predict_from_embedding")
    for emb in embeddings:
        scores.append(_clamp01(_safe_float(predict(emb), default=0.5)))
    return scores


def create_lawstress_bootstrap_metric(
    *,
    proxy_model: Any,
    embedding_client: Any,
    config: Optional[LawStressEvalConfig] = None,
    objective: Optional[LawStressBootstrapObjectiveConfig] = None,
    rubric: str = RILE_RUBRIC,
) -> Callable:
    """Return a GEPA-compatible metric(gold, pred, ...) -> {score, feedback}."""

    config = config or LawStressEvalConfig()
    objective = objective or LawStressBootstrapObjectiveConfig()

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> Dict[str, Any]:
        law_target = str(getattr(gold, "law_target", "") or "").strip().lower()

        feedback_parts: List[str] = []
        details: Dict[str, Any] = {"law_target": law_target}

        if law_target == "c1_sufficiency":
            summary1 = str(getattr(pred, "summary1", "") or "").strip()
            target_norm = _clamp01(_safe_float(getattr(gold, "y_doc_norm", 0.5), default=0.5))
            score_s1 = _proxy_score_texts(proxy_model, embedding_client, [summary1])[0]
            err = abs(score_s1 - target_norm)
            c1_base = _score_from_error(err, config.c1_threshold_norm)
            base = _aggregate_components([c1_base], weights=[1.0], objective=objective)

            penalty, notes = _length_penalty(summary1, str(getattr(gold, "text", "") or ""))
            final = _clamp01(base - penalty)

            if err > config.c1_threshold_norm:
                feedback_parts.append(f"C1 drift: |proxy(summary1)-y|={err:.3f} > {config.c1_threshold_norm:.3f}.")
            feedback_parts.extend(notes)

            details.update(
                {
                    "proxy_summary1": float(score_s1),
                    "y_doc_norm": float(target_norm),
                    "c1_error": float(err),
                    "c1_base": float(c1_base),
                    "base": float(base),
                    "objective_mode": _normalize_aggregate_mode(objective.aggregate_mode),
                    "length_penalty": float(penalty),
                }
            )
            return {"score": float(final), "feedback": " ".join(feedback_parts) or "OK", "details": details}

        if law_target == "c2_idempotence":
            summary1 = str(getattr(pred, "summary1", "") or "").strip()
            summary2 = str(getattr(pred, "summary2", "") or "").strip()

            score_s1, score_s2 = _proxy_score_texts(proxy_model, embedding_client, [summary1, summary2])
            err = abs(score_s2 - score_s1)
            c2_base = _score_from_error(err, config.c2_threshold_norm)

            # Auxiliary C1-on-summary1 (no extra model calls; score_s1 already computed).
            target_norm = _clamp01(_safe_float(getattr(gold, "y_doc_norm", 0.5), default=0.5))
            c1_err = abs(score_s1 - target_norm)
            c1_aux = _score_from_error(c1_err, config.c1_threshold_norm)
            base = _aggregate_components([c2_base, c1_aux], weights=[3.0, 1.0], objective=objective)

            same_side = strict_same_side(score_s2, score_s1, neutral_norm=config.neutral_norm)
            if not same_side:
                base = 0.0  # hard penalty
                feedback_parts.append("C2 sign flip / neutral collapse (strict same-side failed).")

            pen1, notes1 = _length_penalty(summary1, str(getattr(gold, "text", "") or ""))
            pen2, notes2 = _length_penalty(summary2, summary1)
            penalty = min(0.35, pen1 + 0.7 * pen2)
            final = _clamp01(base - penalty)

            if err > config.c2_threshold_norm:
                feedback_parts.append(f"C2 drift: |proxy(s2)-proxy(s1)|={err:.3f} > {config.c2_threshold_norm:.3f}.")
            feedback_parts.extend(notes1)
            feedback_parts.extend(notes2)

            details.update(
                {
                    "proxy_summary1": float(score_s1),
                    "proxy_summary2": float(score_s2),
                    "c2_error": float(err),
                    "c2_base": float(c2_base),
                    "c1_aux_error": float(c1_err),
                    "c1_aux": float(c1_aux),
                    "base": float(base),
                    "objective_mode": _normalize_aggregate_mode(objective.aggregate_mode),
                    "c2_same_side": bool(same_side),
                    "length_penalty": float(penalty),
                }
            )
            return {"score": float(final), "feedback": " ".join(feedback_parts) or "OK", "details": details}

        # Default: c3_merge
        summary_a = str(getattr(pred, "summary_a", "") or "").strip()
        summary_b = str(getattr(pred, "summary_b", "") or "").strip()
        merged_summary = str(getattr(pred, "merged_summary", "") or "").strip()
        joint_segments_summary = str(getattr(pred, "joint_segments_summary", "") or "").strip()

        # Proxy scores for predicted texts (single embedding batch call).
        score_a, score_b, score_merge, score_joint = _proxy_score_texts(
            proxy_model,
            embedding_client,
            [summary_a, summary_b, merged_summary, joint_segments_summary],
        )

        seg_a_raw = _safe_float(getattr(gold, "teacher_score_segment_a_raw", None), default=0.0)
        seg_b_raw = _safe_float(getattr(gold, "teacher_score_segment_b_raw", None), default=0.0)
        len_a = len(str(getattr(gold, "segment_a", "") or ""))
        len_b = len(str(getattr(gold, "segment_b", "") or ""))
        expected_raw = length_weighted_mean(seg_a_raw, seg_b_raw, len_a, len_b)
        expected_norm = _clamp01(normalize_rile(float(expected_raw)))

        err_merge = abs(score_merge - expected_norm)
        c3b = _score_from_error(err_merge, config.c3_threshold_norm)

        err_sub = abs(score_joint - score_merge)
        c3a = _score_from_error(err_sub, config.c3_threshold_norm)

        # Auxiliary: segment-level C1 preservation for summary_a/summary_b.
        seg_a_norm = _clamp01(normalize_rile(seg_a_raw))
        seg_b_norm = _clamp01(normalize_rile(seg_b_raw))
        err_seg_a = abs(score_a - seg_a_norm)
        err_seg_b = abs(score_b - seg_b_norm)
        c1_seg = 0.5 * (
            _score_from_error(err_seg_a, config.c1_threshold_norm)
            + _score_from_error(err_seg_b, config.c1_threshold_norm)
        )

        # Emphasize merge-consistency; keep substitution + segment sufficiency auxiliary.
        base = _aggregate_components([c3b, c3a, c1_seg], weights=[3.0, 1.0, 1.0], objective=objective)

        pen_a, notes_a = _length_penalty(summary_a, str(getattr(gold, "segment_a", "") or ""))
        pen_b, notes_b = _length_penalty(summary_b, str(getattr(gold, "segment_b", "") or ""))
        pen_m, notes_m = _length_penalty(
            merged_summary,
            f"{summary_a}\n\n{summary_b}",
            min_chars=120,
            min_frac=0.06,
            max_frac=0.50,
        )
        penalty = min(0.35, 0.5 * (pen_a + pen_b) + pen_m)
        final = _clamp01(base - penalty)

        if err_merge > config.c3_threshold_norm:
            feedback_parts.append(
                f"C3 merge drift: |proxy(merge)-expected|={err_merge:.3f} > {config.c3_threshold_norm:.3f}."
            )
        if err_sub > config.c3_threshold_norm:
            feedback_parts.append(
                f"C3 substitution drift: |proxy(joint)-proxy(disjoint)|={err_sub:.3f} > {config.c3_threshold_norm:.3f}."
            )
        if err_seg_a > config.c1_threshold_norm or err_seg_b > config.c1_threshold_norm:
            feedback_parts.append("Segment-level drift detected in summary_a/summary_b (aux C1).")
        feedback_parts.extend(notes_a)
        feedback_parts.extend(notes_b)
        feedback_parts.extend(notes_m)

        details.update(
            {
                "proxy_summary_a": float(score_a),
                "proxy_summary_b": float(score_b),
                "proxy_merge": float(score_merge),
                "proxy_joint": float(score_joint),
                "expected_merge_norm": float(expected_norm),
                "c3_merge_error": float(err_merge),
                "c3_sub_error": float(err_sub),
                "seg_a_norm": float(seg_a_norm),
                "seg_b_norm": float(seg_b_norm),
                "c1_seg_error_a": float(err_seg_a),
                "c1_seg_error_b": float(err_seg_b),
                "c1_seg": float(c1_seg),
                "c3b_base": float(c3b),
                "c3a_base": float(c3a),
                "base": float(base),
                "objective_mode": _normalize_aggregate_mode(objective.aggregate_mode),
                "length_penalty": float(penalty),
            }
        )
        return {"score": float(final), "feedback": " ".join(feedback_parts) or "OK", "details": details}

    return metric


__all__ = [
    "LawStressBootstrapObjectiveConfig",
    "create_lawstress_bootstrap_metric",
]
