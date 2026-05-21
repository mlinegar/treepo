from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


_SCENARIO_ID_RE = re.compile(
    r"^temporal_main_(?P<semantic>sem_on|sem_off)_(?P<weighting>learned|fixed)_(?P<windowing>uniform|chunker)$"
)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        rendered = float(value)
    except (TypeError, ValueError):
        return None
    if rendered != rendered:
        return None
    return float(rendered)


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_temporal_main_scenario_id(scenario_id: str) -> Optional[Dict[str, Any]]:
    match = _SCENARIO_ID_RE.match(str(scenario_id or "").strip())
    if not match:
        return None
    semantic = str(match.group("semantic"))
    weighting = str(match.group("weighting"))
    windowing = str(match.group("windowing"))
    semantic_on = semantic == "sem_on"
    learned_weighting = weighting == "learned"
    return {
        "semantic_memory_features": bool(semantic_on),
        "learn_loss_weights": bool(learned_weighting),
        "windowing_mode": windowing,
    }


def _candidate_sort_key(candidate: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    test_delta_improvement = _safe_float(candidate.get("test_delta_improvement"))
    test_rile_mae = _safe_float(candidate.get("test_rile_mae"))
    val_delta_improvement = _safe_float(candidate.get("val_delta_improvement"))
    test_delta_count = _safe_float(candidate.get("test_delta_count"))
    return (
        test_delta_improvement if test_delta_improvement is not None else float("-inf"),
        -(test_rile_mae if test_rile_mae is not None else float("inf")),
        val_delta_improvement if val_delta_improvement is not None else float("-inf"),
        test_delta_count if test_delta_count is not None else float("-inf"),
    )


def _normalize_result_rows(artifact: Mapping[str, Any], *, scenario_prefix: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    results = artifact.get("results", [])
    if not isinstance(results, Sequence):
        return out
    for raw in results:
        if not isinstance(raw, Mapping):
            continue
        scenario_id = str(raw.get("id", "") or "").strip()
        if not scenario_id.startswith(str(scenario_prefix)):
            continue
        metrics_raw = raw.get("metrics", {})
        metrics = metrics_raw if isinstance(metrics_raw, Mapping) else {}
        status = str(raw.get("status", "") or "").strip().lower()
        executed = status != "skipped"
        actual_outcome = str(raw.get("actual_outcome", "") or "").strip().lower()
        if not actual_outcome:
            actual_outcome = "pass" if status == "passed" else "fail"
        expectation_met = bool(raw.get("expectation_met", status == "passed"))
        parsed = parse_temporal_main_scenario_id(scenario_id) or {}
        out.append(
            {
                "id": scenario_id,
                "status": status,
                "executed": bool(executed),
                "actual_outcome": actual_outcome,
                "expectation_met": bool(expectation_met),
                "test_rile_mae": _safe_float(metrics.get("test_rile_mae")),
                "test_delta_count": _safe_int(metrics.get("test_delta_count")),
                "test_delta_improvement": _safe_float(metrics.get("test_delta_improvement")),
                "val_delta_improvement": _safe_float(metrics.get("val_delta_improvement")),
                **parsed,
            }
        )
    return out


def recommend_manifesto_memory_defaults(
    artifact: Mapping[str, Any],
    *,
    scenario_prefix: str = "temporal_main_",
    min_delta_count: int = 20,
    max_rile_mae: float = 0.20,
    fallback_scenario_id: str = "temporal_main_sem_on_learned_chunker",
) -> Dict[str, Any]:
    candidates = _normalize_result_rows(artifact, scenario_prefix=scenario_prefix)
    if not candidates:
        fallback_parsed = parse_temporal_main_scenario_id(fallback_scenario_id)
        if fallback_parsed is None:
            raise ValueError(f"No matrix candidates and invalid fallback scenario id: {fallback_scenario_id!r}")
        selected = {"id": fallback_scenario_id, **fallback_parsed}
        return _build_recommendation_payload(
            selected=selected,
            candidates=[],
            min_delta_count=min_delta_count,
            max_rile_mae=max_rile_mae,
            selection_reason="no_temporal_main_candidates_fallback",
        )

    eligible: List[Dict[str, Any]] = []
    for row in candidates:
        row["eligible"] = bool(
            row.get("executed")
            and row.get("expectation_met")
            and str(row.get("actual_outcome")) == "pass"
            and (row.get("test_delta_count") is not None and int(row["test_delta_count"]) >= int(min_delta_count))
            and (
                row.get("test_rile_mae") is None
                or float(row["test_rile_mae"]) <= float(max_rile_mae)
            )
        )
        if row["eligible"]:
            eligible.append(row)

    selected: Dict[str, Any]
    selection_reason = "eligible_candidates"
    if eligible:
        selected = max(eligible, key=_candidate_sort_key)
    else:
        valid = [
            row
            for row in candidates
            if row.get("executed") and row.get("expectation_met") and str(row.get("actual_outcome")) == "pass"
        ]
        if valid:
            selected = max(valid, key=_candidate_sort_key)
            selection_reason = "no_eligible_used_best_valid"
        else:
            fallback_parsed = parse_temporal_main_scenario_id(fallback_scenario_id)
            if fallback_parsed is None:
                raise ValueError(f"Invalid fallback scenario id: {fallback_scenario_id!r}")
            selected = {"id": fallback_scenario_id, **fallback_parsed}
            selection_reason = "no_valid_candidates_fallback"

    return _build_recommendation_payload(
        selected=selected,
        candidates=candidates,
        min_delta_count=min_delta_count,
        max_rile_mae=max_rile_mae,
        selection_reason=selection_reason,
    )


def _build_recommendation_payload(
    *,
    selected: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    min_delta_count: int,
    max_rile_mae: float,
    selection_reason: str,
) -> Dict[str, Any]:
    semantic_memory_features = bool(selected.get("semantic_memory_features", True))
    learn_loss_weights = bool(selected.get("learn_loss_weights", True))
    windowing_mode = str(selected.get("windowing_mode", "chunker") or "chunker")

    training_flags = [
        "--delta-head",
        "--max-windows 0",
        "--windowing-mode " + windowing_mode,
        "--semantic-memory-features" if semantic_memory_features else "--no-semantic-memory-features",
        "--learn-loss-weights" if learn_loss_weights else "--no-learn-loss-weights",
    ]

    pipeline_flags = [
        "--semantic-memory" if semantic_memory_features else "--no-semantic-memory",
        "--semantic-memory-model-features" if semantic_memory_features else "--no-semantic-memory-model-features",
        "--semantic-memory-inject-prompts",
        "--semantic-memory-max-windows 0",
        "--semantic-memory-update-policy post_score",
    ]

    return {
        "selected_scenario_id": str(selected.get("id", "")),
        "selection_reason": str(selection_reason),
        "selection_constraints": {
            "min_delta_count": int(min_delta_count),
            "max_rile_mae": float(max_rile_mae),
        },
        "recommended_defaults": {
            "delta_head": True,
            "semantic_memory_features": semantic_memory_features,
            "learn_loss_weights": learn_loss_weights,
            "windowing_mode": windowing_mode,
            "max_windows": 0,
            "semantic_memory": {
                "enabled": semantic_memory_features,
                "top_k": 5,
                "lambda_year": 0.08,
                "index_granularity": "doc_chunk",
                "max_windows": 0,
                "update_policy": "post_score",
                "inject_prompts": True,
                "model_features": semantic_memory_features,
            },
        },
        "training_flags": training_flags,
        "pipeline_flags": pipeline_flags,
        "candidates": list(candidates),
    }
