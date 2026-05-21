from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from treepo._research.ctreepo.contracts import (
    LAW_ID_LEAF_PRESERVATION,
    LAW_ID_MERGE_PRESERVATION,
    LAW_ID_ON_RANGE_IDEMPOTENCE,
)
from treepo._research.ctreepo.sim.util import safe_float
from treepo._research.ctreepo.sim.composite_objective import (
    CompositeObjectiveSpec,
    scalarize_objective_estimates,
)
from treepo._research.ctreepo.sim.learning_problem import attach_local_law_learning_problem
from treepo._research.ctreepo.sim.local_law_learnability import (
    DownstreamMetrics,
    LocalLawCounterexampleEvaluation,
    LocalLawMetrics,
    LocalLawPolicyEvaluation,
    LocalLawRunSummary,
    PolicyRole,
    SupportBudgetSummary,
)


_safe_float = safe_float


def _suite_role_from_path(path: str, *, family: str) -> str:
    raw = str(path)
    if family == "markov_ops_count" and ("/exact/" in raw or "exact_" in raw):
        return "failure_modes"
    mapping = {
        "suite_a_exact_controls": "positive_controls",
        "sanity_suite": "positive_controls",
        "suite_b_local_law_learnability": "support_scaling",
        "transition_map_suite": "support_scaling",
        "suite_c_mismatch_mediation": "relevance_mediation",
        "mechanism_suite": "relevance_mediation",
        "suite_d_ipw_sparse_labels": "failure_modes",
        "suite_e_hardness": "hardness",
        "capacity_appendix_suite": "hardness",
    }
    for key, value in mapping.items():
        if key in raw:
            return value
    return ""


def _split_id(prefix: str, *, seed: int, n_docs: int) -> str:
    return f"{prefix}:seed={int(seed)}:docs={int(n_docs)}"


def _resolve_legacy_lda_package_and_weights(
    payload: Mapping[str, Any],
    *,
    blank_package_fallback: str = "",
) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    cfg = dict(payload.get("config", {}) or {})
    local_law = dict(payload.get("local_law", {}) or {})
    local_cfg = dict(local_law.get("config", {}) or {})
    explicit_package = str(
        local_cfg.get("law_package", cfg.get("law_package", "")) or ""
    ).strip()
    package = str(explicit_package or blank_package_fallback).strip().lower()
    raw_weights = {
        "c1": _safe_float(
            local_cfg.get("law_c1_weight", cfg.get("law_c1_weight")),
            0.0,
        ),
        "c2_proxy": _safe_float(
            local_cfg.get("law_c2_proxy_weight", cfg.get("law_c2_proxy_weight")),
            0.0,
        ),
        "c3": _safe_float(
            local_cfg.get("law_c3_weight", cfg.get("law_c3_weight")),
            0.0,
        ),
    }
    resolved = dict(raw_weights)
    if package == "root_only":
        resolved = {"c1": 0.0, "c2_proxy": 0.0, "c3": 0.0}
    elif package == "c1_only":
        resolved = {"c1": raw_weights["c1"], "c2_proxy": 0.0, "c3": 0.0}
    elif package == "c2_only":
        resolved = {"c1": 0.0, "c2_proxy": raw_weights["c2_proxy"], "c3": 0.0}
    elif package == "c3_only":
        resolved = {"c1": 0.0, "c2_proxy": 0.0, "c3": raw_weights["c3"]}
    elif package == "c1c3":
        resolved = {"c1": raw_weights["c1"], "c2_proxy": 0.0, "c3": raw_weights["c3"]}
    return explicit_package or str(blank_package_fallback or ""), raw_weights, resolved


def _legacy_lda_objective_payload(
    metrics: Mapping[str, object],
    *,
    ipw_payload: Optional[Mapping[str, Any]],
    raw_weights: Mapping[str, float],
    resolved_weights: Mapping[str, float],
    law_package: str,
) -> Dict[str, Any]:
    active_weights = {
        str(name): float(weight)
        for name, weight in dict(resolved_weights).items()
        if float(weight) > 0.0
    }
    spec = CompositeObjectiveSpec(
        name="configured_local_law_objective",
        selection_metric_name="configured_local_law_objective",
        task_name="mean_aux_oracle_target_abs_error",
        task_weight=0.0,
        local_law_weights=active_weights,
        proxy_weights={},
        weighting_scheme="legacy_local_law_only_weighted_sum",
        task_weight_source="backfilled_zero_from_missing_legacy_task_weight",
        metadata={
            "task_metric_name": "mean_aux_oracle_target_abs_error",
            "backfilled_scope": "local_law_only",
            "law_package": str(law_package),
            "raw_local_law_weights": {
                str(name): float(value) for name, value in dict(raw_weights).items()
            },
            "resolved_local_law_weights": {
                str(name): float(value) for name, value in dict(resolved_weights).items()
            },
        },
    )
    task_raw = _safe_float(metrics.get("mean_aux_oracle_target_abs_error"), 0.0)
    task_estimates = {
        "exact": float(task_raw),
        "ht": float(task_raw),
        "hajek": float(task_raw),
        "eb_lo": float(task_raw),
        "eb_hi": float(task_raw),
    }
    ipw_source = dict(ipw_payload or {})

    def _metric_estimates(policy_key: str, exact_value: float) -> Dict[str, float]:
        branch = dict(ipw_source.get(policy_key, {}) or {})
        return {
            "exact": _safe_float(branch.get("population_exact_mean"), exact_value),
            "ht": _safe_float(branch.get("ht_mean")),
            "hajek": _safe_float(branch.get("hajek")),
            "eb_lo": _safe_float(branch.get("eb_lo")),
            "eb_hi": _safe_float(branch.get("eb_hi")),
        }

    # Backfill is a legacy reconstruction path for runs that pre-date the
    # normalized root/local objective. Forward-going runs resolve a single
    # lambda or explicit normalized root/law weights before scalarization.
    estimator_payload = scalarize_objective_estimates(
        spec,
        task_estimates=task_estimates,
        local_law_estimates={
            LAW_ID_LEAF_PRESERVATION: _metric_estimates(
                "c1", _safe_float(metrics.get("mean_c1"), 0.0)
            ),
            LAW_ID_ON_RANGE_IDEMPOTENCE: _metric_estimates(
                "c2_proxy", _safe_float(metrics.get("mean_c2_proxy"), 0.0)
            ),
            LAW_ID_MERGE_PRESERVATION: _metric_estimates(
                "c3", _safe_float(metrics.get("mean_c3"), 0.0)
            ),
        },
        proxy_estimates={},
        selection_preference="hajek",
    )
    local = LocalLawMetrics(
        c1=_safe_float(metrics.get("mean_c1")),
        c2=_safe_float(metrics.get("mean_c2_proxy")),
        c3=_safe_float(metrics.get("mean_c3")),
        combined=_safe_float(metrics.get("combined_law_score")),
        root_error=_safe_float(metrics.get("mean_root_c3_error")),
        c1_violation_rate=_safe_float(metrics.get("c1_violation_rate")),
        c2_violation_rate=_safe_float(metrics.get("c2_proxy_violation_rate")),
        c3_violation_rate=_safe_float(metrics.get("c3_violation_rate")),
    ).to_dict()
    downstream = DownstreamMetrics(
        oracle_target_abs_error=_safe_float(metrics.get("mean_aux_oracle_target_abs_error")),
        oracle_target_delta=_safe_float(metrics.get("mean_aux_oracle_target_delta")),
        root_error=_safe_float(metrics.get("mean_root_c3_error")),
    ).to_dict()
    objective = dict(spec.to_dict())
    objective.update(estimator_payload)
    objective["task_objective_value"] = float(task_raw)
    objective["regular_objective_value"] = float(task_raw)
    objective["value_source"] = "legacy_payload_local_law_only"
    legacy_metric = str(metrics.get("selection_metric_name", "") or "").strip()
    if legacy_metric:
        objective["legacy_selection_metric_name"] = str(legacy_metric)
    objective["legacy_combined_law_score"] = _safe_float(metrics.get("combined_law_score"))
    objective["reported_combined_law_score"] = _safe_float(metrics.get("combined_law_score"))
    objective["selection_metric_value"] = _safe_float(
        objective.get(str(objective.get("selection_metric_name", ""))),
        _safe_float(objective.get("full_objective_value")),
    )
    return {"test": {"local_law": local, "downstream": downstream, "objective": objective}}


def _backfill_legacy_lda(
    payload: Mapping[str, Any],
    *,
    source_path: str,
    blank_package_fallback: str = "",
) -> Optional[LocalLawRunSummary]:
    local_law = dict(payload.get("local_law", {}) or {})
    policy_metrics = dict(local_law.get("policy_metrics", {}) or {})
    if (
        not policy_metrics
        or "infer_identity" not in policy_metrics
        or "oracle_true_summary" not in policy_metrics
    ):
        return None
    cfg = dict(payload.get("config", {}) or {})
    training = dict(local_law.get("training", {}) or {})
    selection = dict(local_law.get("selection", {}) or {})
    ipw_evaluation = dict(local_law.get("ipw_evaluation", {}) or {})
    law_package, raw_weights, resolved_weights = _resolve_legacy_lda_package_and_weights(
        payload,
        blank_package_fallback=blank_package_fallback,
    )
    suite_role = str(
        cfg.get("suite_role", "")
        or _suite_role_from_path(source_path, family="tree_relevant_lda_local_law")
    )

    selected_candidate = str(selection.get("selected_candidate", "") or "").strip()
    if not selected_candidate:
        methods = dict(payload.get("methods", {}) or {})
        diagnostics = dict(
            methods.get("analysis_infer_law_calibrated_oracle_target", {}).get("diagnostics", {})
            or {}
        )
        selected_candidate = str(diagnostics.get("calibration_variant", "") or "").strip()
    if not selected_candidate:
        if "law_calibrated_ipw" in policy_metrics:
            selected_candidate = "law_calibrated_ipw"
        elif "law_calibrated_ipw_stabilized" in policy_metrics:
            selected_candidate = "law_calibrated_ipw_stabilized"
        elif "law_calibrated_naive" in policy_metrics:
            selected_candidate = "law_calibrated_naive"

    train_docs = int(cfg.get("train_docs", 0))
    val_docs = int(cfg.get("val_docs", 0) or 0)
    test_docs = int(cfg.get("test_docs", 0))
    seed = int(cfg.get("seed", 0))
    policies: Dict[str, LocalLawPolicyEvaluation] = {
        "oracle_true_summary": LocalLawPolicyEvaluation(
            name="oracle_true_summary",
            role=PolicyRole.ORACLE_G,
            split_metrics=_legacy_lda_objective_payload(
                dict(policy_metrics["oracle_true_summary"]),
                ipw_payload=dict(ipw_evaluation.get("oracle_true_summary", {}) or {}),
                raw_weights=raw_weights,
                resolved_weights=resolved_weights,
                law_package=law_package,
            ),
            metadata={"backfilled_from_legacy": True},
        ),
        "infer_identity": LocalLawPolicyEvaluation(
            name="infer_identity",
            role=PolicyRole.BASELINE_G,
            split_metrics=_legacy_lda_objective_payload(
                dict(policy_metrics["infer_identity"]),
                ipw_payload=dict(ipw_evaluation.get("infer_identity", {}) or {}),
                raw_weights=raw_weights,
                resolved_weights=resolved_weights,
                law_package=law_package,
            ),
            metadata={"backfilled_from_legacy": True},
        ),
    }
    for name in ("law_calibrated_naive", "law_calibrated_ipw", "law_calibrated_ipw_stabilized"):
        if name not in policy_metrics:
            continue
        policies[str(name)] = LocalLawPolicyEvaluation(
            name=str(name),
            role=PolicyRole.CANDIDATE_G,
            split_metrics=_legacy_lda_objective_payload(
                dict(policy_metrics[name]),
                ipw_payload=dict(ipw_evaluation.get(name, {}) or {}),
                raw_weights=raw_weights,
                resolved_weights=resolved_weights,
                law_package=law_package,
            ),
            metadata={"backfilled_from_legacy": True},
        )
    if selected_candidate and selected_candidate in policy_metrics:
        selected_policy_metrics = dict(policy_metrics[selected_candidate])
        selection_metric_name = str(
            selection.get("selection_metric", "") or "legacy_fixed_candidate"
        )
        selection_metric_value = _safe_float(selected_policy_metrics.get(selection_metric_name))
        if not math.isfinite(selection_metric_value):
            selection_metric_value = _safe_float(selected_policy_metrics.get("combined_law_score"))
        policies["learned_g"] = LocalLawPolicyEvaluation(
            name=str(selected_candidate),
            role=PolicyRole.LEARNED_G,
            selection_metric_value=selection_metric_value,
            split_metrics=_legacy_lda_objective_payload(
                selected_policy_metrics,
                ipw_payload=dict(ipw_evaluation.get(selected_candidate, {}) or {}),
                raw_weights=raw_weights,
                resolved_weights=resolved_weights,
                law_package=law_package,
            ),
            metadata={
                "backfilled_from_legacy": True,
                "legacy_selected_candidate": str(selected_candidate),
            },
        )

    thresholds = {
        "c1": _safe_float(dict(local_law.get("config", {}) or {}).get("law_c1_threshold"), 0.2),
        "c2": _safe_float(dict(local_law.get("config", {}) or {}).get("law_c2_threshold"), 0.2),
        "c3": _safe_float(dict(local_law.get("config", {}) or {}).get("law_c3_threshold"), 0.2),
    }
    total_queries = _safe_float(training.get("leaf_label_count"), 0.0) + _safe_float(
        training.get("internal_label_count"), 0.0
    )
    return LocalLawRunSummary(
        family="tree_relevant_lda_local_law",
        dgp="leaf_local_mixture_utility",
        oracle_name="oracle_true_summary",
        study_role=str(cfg.get("local_law_mode", "legacy_local_law")),
        split_ids={
            "train": _split_id("tree_relevant_lda:train", seed=seed, n_docs=train_docs),
            "val": _split_id("tree_relevant_lda:val", seed=seed, n_docs=val_docs),
            "test": _split_id("tree_relevant_lda:test", seed=seed, n_docs=test_docs),
        },
        support_budget=SupportBudgetSummary(
            train_docs=train_docs,
            val_docs=val_docs,
            test_docs=test_docs,
            leaf_query_rate=_safe_float(cfg.get("law_leaf_query_rate"), 0.0),
            internal_query_rate=_safe_float(cfg.get("law_internal_query_rate"), 0.0),
            mean_leaf_labels_per_doc=_safe_float(training.get("leaf_label_count"), 0.0)
            / float(max(1, train_docs)),
            mean_internal_labels_per_doc=_safe_float(training.get("internal_label_count"), 0.0)
            / float(max(1, train_docs)),
            mean_queries_per_doc=total_queries / float(max(1, train_docs)),
            total_queries_estimate=total_queries,
            metadata={"backfilled_from_legacy": True},
        ),
        selection={
            "selection_split": str(selection.get("selection_split", "") or "config"),
            "selection_metric": str(
                selection.get("selection_metric", "") or "legacy_fixed_candidate"
            ),
            "selected_candidate": str(selected_candidate),
            "test_metrics_used_for_selection": False,
            "backfilled_from_legacy": True,
        },
        policies=policies,
        counterexamples=[],
        thresholds=thresholds,
        suite_role=suite_role,
        metadata={
            "analysis_partition_mode": str(cfg.get("analysis_partition_mode", "")),
            "lambda_multiplier": _safe_float(cfg.get("lambda_multiplier")),
            "law_leaf_query_design": str(cfg.get("law_leaf_query_design", "")),
            "law_internal_query_design": str(cfg.get("law_internal_query_design", "")),
            "law_package": str(law_package),
            "raw_local_law_weights": {
                str(name): float(value) for name, value in dict(raw_weights).items()
            },
            "resolved_local_law_weights": {
                str(name): float(value) for name, value in dict(resolved_weights).items()
            },
            "backfilled_from_legacy": True,
            "source_path": str(source_path),
        },
    )


def _normalized_markov_value(
    block: Mapping[str, object], raw_key: str, norm_key: str, *, scale: float
) -> float:
    norm = _safe_float(block.get(norm_key))
    if math.isfinite(norm):
        return float(norm)
    raw = _safe_float(block.get(raw_key))
    if math.isfinite(raw) and scale > 0.0:
        return float(raw) / float(scale)
    return float("nan")


def _legacy_markov_policy_payload(
    block: Mapping[str, object], *, scale: float
) -> Dict[str, Dict[str, Any]]:
    c1 = _normalized_markov_value(block, "leaf_mae", "c1_leaf_mae_n", scale=scale)
    c2 = _normalized_markov_value(block, "c2_idempotence_mae", "c2_idempotence_mae_n", scale=scale)
    c3 = _normalized_markov_value(block, "merge_mae", "c3_merge_mae_n", scale=scale)
    root = _normalized_markov_value(block, "root_mae", "root_mae_n", scale=scale)
    spread = _normalized_markov_value(
        block, "schedule_spread_mean", "schedule_spread_mean_n", scale=scale
    )
    combined = _safe_float(block.get("theorem_bundle_score_n"))
    if not math.isfinite(combined):
        combined = _safe_float(block.get("test_theorem_bundle_score_n"))
    if not math.isfinite(combined):
        combined = _safe_float(block.get("val_theorem_bundle_score_n"))
    if not math.isfinite(combined):
        combined = float(sum(v for v in (c1, c2, c3) if math.isfinite(v)))
    local = LocalLawMetrics(
        c1=c1,
        c2=c2,
        c3=c3,
        combined=combined,
        root_error=root,
        schedule_spread=spread,
        c1_violation_rate=_safe_float(block.get("leaf_violation_rate")),
        c3_violation_rate=_safe_float(block.get("merge_violation_rate")),
    ).to_dict()
    downstream = DownstreamMetrics(root_error=root, schedule_spread=spread).to_dict()
    objective = {
        "objective_name": "configured_objective",
        "selection_metric_name": "configured_objective",
        "weighting_scheme": str(block.get("weighting_scheme", "legacy_theorem_bundle_proxy")),
        "task_metric_name": "root_error",
        "task_weight": _safe_float(block.get("objective_task_objective_weight"), float("nan")),
        "local_law_weight_total": _safe_float(
            block.get("objective_local_law_weight"), float("nan")
        ),
        "proxy_weight_total": _safe_float(
            block.get("objective_proxy_schedule_consistency_weight"),
            0.0,
        ),
        "total_weight_without_proxy": _safe_float(
            block.get("objective_optimization_weight_mass_no_proxy"),
            float("nan"),
        ),
        "local_law_weight": _safe_float(block.get("objective_local_law_weight"), float("nan")),
        "normalized_task_share": (
            _safe_float(
                block.get("objective_task_objective_weight"),
                float("nan"),
            )
            / _safe_float(block.get("objective_optimization_weight_mass_no_proxy"), float("nan"))
            if _safe_float(block.get("objective_optimization_weight_mass_no_proxy"), float("nan"))
            > 0.0
            else float("nan")
        ),
        "normalized_local_law_share": (
            _safe_float(
                block.get("objective_local_law_weight"),
                float("nan"),
            )
            / _safe_float(block.get("objective_optimization_weight_mass_no_proxy"), float("nan"))
            if _safe_float(block.get("objective_optimization_weight_mass_no_proxy"), float("nan"))
            > 0.0
            else float("nan")
        ),
        "full_objective_value": _safe_float(
            block.get("test_objective_full_labels"),
            combined,
        ),
        "task_objective_value": _safe_float(
            block.get("test_unweighted_objective_task_objective_term"),
            root,
        ),
        "task_objective_term": _safe_float(
            block.get("test_objective_task_objective_term"),
            root,
        ),
        "regular_objective_value": _safe_float(
            block.get("test_unweighted_objective_task_objective_term"),
            root,
        ),
        "regular_objective_term": _safe_float(
            block.get("test_objective_task_objective_term"),
            root,
        ),
        "local_law_objective_value": (
            float(
                sum(
                    _safe_float(block.get(key), 0.0)
                    for key in (
                        "test_unweighted_objective_leaf_term",
                        "test_unweighted_objective_c2_term",
                        "test_unweighted_objective_merge_term",
                    )
                )
            )
            if any(
                key in block
                for key in (
                    "test_unweighted_objective_leaf_term",
                    "test_unweighted_objective_c2_term",
                    "test_unweighted_objective_merge_term",
                )
            )
            else combined
        ),
        "local_law_objective_term": (
            float(
                sum(
                    _safe_float(block.get(key), 0.0)
                    for key in (
                        "test_objective_leaf_term",
                        "test_objective_c2_term",
                        "test_objective_merge_term",
                    )
                )
            )
            if any(
                key in block
                for key in (
                    "test_objective_leaf_term",
                    "test_objective_c2_term",
                    "test_objective_merge_term",
                )
            )
            else combined
        ),
        "proxy_objective_value": _safe_float(
            block.get("test_unweighted_objective_schedule_consistency_term"),
            0.0,
        ),
        "proxy_objective_term": _safe_float(
            block.get("test_objective_schedule_consistency_term"),
            0.0,
        ),
        "value_source": "legacy_payload",
    }
    return {
        "local_law_metrics": local,
        "downstream_metrics": downstream,
        "objective_metrics": objective,
    }


def _legacy_markov_selection_metric(block: Mapping[str, object]) -> Tuple[float, str]:
    for key in (
        "val_optimization_objective_full_labels",
        "val_objective_full_labels",
        "test_optimization_objective_full_labels",
        "test_objective_full_labels",
    ):
        value = _safe_float(block.get(key))
        if math.isfinite(value):
            return float(value), str(key)
    for key in (
        "val_theorem_bundle_score_n",
        "test_theorem_bundle_score_n",
        "theorem_bundle_score_n",
    ):
        value = _safe_float(block.get(key))
        if math.isfinite(value):
            return float(value), str(key)
    return float("nan"), "configured_objective"


def _backfill_legacy_markov(
    payload: Mapping[str, Any],
    *,
    source_path: str,
) -> Optional[LocalLawRunSummary]:
    cfg = dict(payload.get("config", {}) or {})
    metrics = dict(payload.get("metrics", {}) or {})
    geom = dict(payload.get("training_geometry", {}) or {})
    objective = dict(payload.get("objective", {}) or {})
    if not metrics or not geom or not cfg:
        return None
    if "learned" not in metrics and "stress_family" not in metrics:
        return None

    scale = float(max(1, int(cfg.get("max_segments", 1)) - 1))
    train_docs = int(cfg.get("train_docs", 0))
    val_docs = int(cfg.get("val_docs", 0))
    test_docs = int(cfg.get("test_docs", 0))
    data_seed = int(cfg.get("effective_data_seed", cfg.get("data_seed", cfg.get("seed", 0))))
    model_seed = int(cfg.get("effective_model_seed", cfg.get("model_seed", cfg.get("seed", 0))))
    val_seed = int(cfg.get("effective_val_seed", data_seed + 1))
    test_seed = int(cfg.get("effective_test_seed", data_seed + 2))
    law_package = str(objective.get("law_package", cfg.get("law_package", "")) or "").strip()
    exact_family = str(cfg.get("exact_family", "") or "").strip()
    suite_role = str(
        cfg.get("suite_role", "") or _suite_role_from_path(source_path, family="markov_ops_count")
    )

    policies: Dict[str, LocalLawPolicyEvaluation] = {}
    if "exact" in metrics:
        policies["oracle_g"] = LocalLawPolicyEvaluation(
            name="oracle_g",
            role=PolicyRole.ORACLE_G,
            split_metrics={
                "test": _legacy_markov_policy_payload(dict(metrics["exact"]), scale=scale)
            },
            metadata={"backfilled_from_legacy": True},
        )

    counterexamples = []
    counterexample_specs = [
        ("leaf_bucket", metrics.get("leaf_bucket"), ["C1"]),
        ("count_only", metrics.get("undersupported"), ["C3"]),
        ("flip_R2", metrics.get("flip_R2"), ["C2"]),
    ]
    for name, block, targeted_laws in counterexample_specs:
        if not isinstance(block, dict):
            continue
        counterexamples.append(
            LocalLawCounterexampleEvaluation(
                name=str(name),
                role=PolicyRole.COUNTEREXAMPLE_G,
                targeted_laws=[str(x) for x in targeted_laws],
                metrics={"test": _legacy_markov_policy_payload(block, scale=scale)},
                metadata={"backfilled_from_legacy": True},
            )
        )

    if "learned" in metrics:
        current_name = law_package or "learned_g"
        current_role = (
            PolicyRole.BASELINE_G if str(current_name) == "root_only" else PolicyRole.LEARNED_G
        )
        split_metrics: Dict[str, Dict[str, Any]] = {}
        if isinstance(metrics.get("learned_train"), dict):
            split_metrics["train"] = _legacy_markov_policy_payload(
                dict(metrics["learned_train"]), scale=scale
            )
        if isinstance(metrics.get("learned_val"), dict):
            split_metrics["val"] = _legacy_markov_policy_payload(
                dict(metrics["learned_val"]), scale=scale
            )
        if isinstance(metrics.get("learned_test"), dict):
            split_metrics["test"] = _legacy_markov_policy_payload(
                dict(metrics["learned_test"]), scale=scale
            )
        else:
            split_metrics["test"] = _legacy_markov_policy_payload(
                dict(metrics["learned"]), scale=scale
            )
        selection_metric, selection_metric_name = _legacy_markov_selection_metric(
            dict(metrics["learned"])
        )
        policies[str(current_name)] = LocalLawPolicyEvaluation(
            name=str(current_name),
            role=current_role,
            selection_metric_value=selection_metric,
            split_metrics=split_metrics,
            metadata={"law_package": str(law_package), "backfilled_from_legacy": True},
        )
        study_role = current_role.value
        selection = {
            "selection_split": "val" if val_docs > 0 else "config",
            "selection_metric": str(selection_metric_name),
            "selected_candidate": str(current_name),
            "uses_test_metrics": False,
            "backfilled_from_legacy": True,
        }
    else:
        stress = dict(metrics.get("stress_family", {}) or {})
        current_name = exact_family or str(
            stress.get("stress_family_name", "") or "counterexample_g"
        )
        policies[str(current_name)] = LocalLawPolicyEvaluation(
            name=str(current_name),
            role=PolicyRole.COUNTEREXAMPLE_G,
            split_metrics={"test": _legacy_markov_policy_payload(stress, scale=scale)},
            metadata={"exact_family": str(current_name), "backfilled_from_legacy": True},
        )
        if not counterexamples and stress:
            targeted = {
                "leaf_bucket": ["C1"],
                "count_only": ["C3"],
                "flip_R2": ["C2"],
            }.get(str(current_name), [])
            counterexamples.append(
                LocalLawCounterexampleEvaluation(
                    name=str(current_name),
                    role=PolicyRole.COUNTEREXAMPLE_G,
                    targeted_laws=[str(x) for x in targeted],
                    metrics={"test": _legacy_markov_policy_payload(stress, scale=scale)},
                    metadata={"backfilled_from_legacy": True},
                )
            )
        study_role = PolicyRole.COUNTEREXAMPLE_G.value
        selection = {
            "selection_split": "config",
            "selection_metric": "configured_exact_family",
            "selected_candidate": str(current_name),
            "uses_test_metrics": False,
            "backfilled_from_legacy": True,
        }

    return LocalLawRunSummary(
        family="markov_ops_count",
        dgp="markov_changepoint_ops_count",
        oracle_name="changepoint_count_exact_summary",
        study_role=study_role,
        split_ids={
            "train": _split_id("markov:train", seed=data_seed, n_docs=train_docs),
            "val": _split_id("markov:val", seed=val_seed, n_docs=val_docs),
            "test": _split_id("markov:test", seed=test_seed, n_docs=test_docs),
        },
        support_budget=SupportBudgetSummary(
            train_docs=train_docs,
            val_docs=val_docs,
            test_docs=test_docs,
            leaf_query_rate=_safe_float(cfg.get("leaf_query_rate"), 0.0),
            internal_query_rate=(
                _safe_float(geom.get("mean_internal_labels"), 0.0)
                / max(1.0, _safe_float(geom.get("mean_internal_nodes"), 1.0))
            ),
            root_query_rate=1.0 if bool(cfg.get("include_root_query", False)) else 0.0,
            mean_leaf_labels_per_doc=_safe_float(geom.get("mean_leaf_labels"), 0.0),
            mean_internal_labels_per_doc=_safe_float(geom.get("mean_internal_labels"), 0.0),
            mean_queries_per_doc=_safe_float(geom.get("mean_queries_per_doc"), 0.0),
            total_queries_estimate=_safe_float(geom.get("total_queries_estimate"), 0.0),
            metadata={
                "backfilled_from_legacy": True,
                "audit_fraction": _safe_float(cfg.get("audit_fraction")),
            },
        ),
        selection=selection,
        policies=policies,
        counterexamples=counterexamples,
        thresholds={
            "c1_tau": _safe_float(cfg.get("violation_tau"), 0.0) / max(1.0, scale),
            "c2_tau": _safe_float(cfg.get("violation_tau"), 0.0) / max(1.0, scale),
            "c3_tau": _safe_float(cfg.get("violation_tau"), 0.0) / max(1.0, scale),
        },
        suite_role=suite_role,
        metadata={
            "law_package": str(law_package),
            "exact_family": str(exact_family),
            "fixed_leaf_tokens": int(cfg.get("fixed_leaf_tokens", 0)),
            "feature_mode": str(cfg.get("feature_mode", "")),
            "model_family": str(cfg.get("model_family", "")),
            "n_regimes": int(cfg.get("n_regimes", 0)),
            "backfilled_from_legacy": True,
            "source_path": str(source_path),
        },
    )


def load_or_backfill_local_law_payload(
    payload: Mapping[str, Any],
    *,
    source_path: str = "",
    blank_lda_law_package_fallback: str = "",
) -> Optional[Tuple[LocalLawRunSummary, Dict[str, Any]]]:
    if not isinstance(payload, Mapping):
        return None
    direct = payload.get("local_law_learnability")
    if isinstance(direct, Mapping) and direct:
        try:
            summary = LocalLawRunSummary.from_dict(direct)
        except Exception:
            return None
        summary_payload = summary.to_dict()
        if not str(summary.suite_role):
            inferred_suite_role = _suite_role_from_path(source_path, family=str(summary.family))
            if inferred_suite_role:
                summary_payload["suite_role"] = str(inferred_suite_role)
                summary = LocalLawRunSummary.from_dict(summary_payload)
        summary = attach_local_law_learning_problem(summary, raw_payload=payload)
        augmented = dict(payload)
        augmented["local_law_learnability"] = summary.to_dict()
        if "g_artifacts" not in augmented:
            augmented["g_artifacts"] = {}
        return summary, augmented

    summary = _backfill_legacy_lda(
        payload,
        source_path=source_path,
        blank_package_fallback=blank_lda_law_package_fallback,
    )
    mode = "legacy_lda"
    if summary is None:
        summary = _backfill_legacy_markov(payload, source_path=source_path)
        mode = "legacy_markov"
    if summary is None:
        return None

    summary = attach_local_law_learning_problem(summary, raw_payload=payload)
    augmented = dict(payload)
    augmented["local_law_learnability"] = summary.to_dict()
    augmented.setdefault("g_artifacts", {})
    augmented["_local_law_backfill"] = {"mode": mode, "source_path": str(source_path)}
    return summary, augmented


def _summary_law_package(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> str:
    metadata = dict(summary.metadata or {})
    package = str(metadata.get("law_package", "") or "").strip()
    if package:
        return package
    objective = dict(metadata.get("objective", {}) or {})
    package = str(objective.get("law_package", "") or "").strip()
    if package:
        return package
    if raw_payload is not None:
        cfg = dict(raw_payload.get("config", {}) or {})
        local_law = dict(raw_payload.get("local_law", {}) or {})
        local_law_cfg = dict(local_law.get("config", {}) or {})
        raw_objective = dict(raw_payload.get("objective", {}) or {})
        package = str(local_law_cfg.get("law_package", "") or "").strip()
        if package:
            return package
        package = str(raw_objective.get("law_package", "") or "").strip()
        if package:
            return package
        package = str(cfg.get("law_package", "") or "").strip()
        if package:
            return package
        if str(summary.family) == "markov_ops_count":
            c1 = _safe_float(cfg.get("c1_relative_weight"))
            c2 = _safe_float(cfg.get("c2_relative_weight"))
            c3 = _safe_float(cfg.get("c3_relative_weight"))
            if math.isfinite(c1) and math.isfinite(c2) and math.isfinite(c3):
                exact_profiles = {
                    (0.0, 1.0, 0.0): "pure_c2",
                    (1.0, 0.0, 4.0): "no_c2",
                    (0.05, 1.0, 0.05): "c2_trace_c1c3",
                    (0.1, 1.0, 0.1): "c2_light_c1c3",
                    (0.25, 1.0, 0.25): "c2_mild_c1c3",
                    (0.5, 1.0, 0.5): "c2_moderate_c1c3",
                    (1.0, 8.0, 1.0): "c2_very_dominant",
                    (1.0, 4.0, 1.0): "c2_dominant",
                    (1.0, 2.0, 1.0): "c2_heavy",
                    (1.0, 1.0, 1.0): "equal",
                    (2.0, 1.0, 2.0): "c1c3_heavy",
                    (1.0, 1.0, 4.0): "c3_dominant",
                }
                rounded = (round(c1, 2), round(c2, 2), round(c3, 2))
                package = exact_profiles.get(rounded, "")
                if not package:
                    rounded = (round(c1, 1), round(c2, 1), round(c3, 1))
                    package = exact_profiles.get(rounded, "")
                if package:
                    return package
                rel_total = float(max(0.0, c1) + max(0.0, c2) + max(0.0, c3))
                if rel_total > 0.0:
                    return f"c1={c1:g}_c2={c2:g}_c3={c3:g}"
    if (
        str(summary.family) == "markov_ops_count"
        and str(summary.study_role) == PolicyRole.BASELINE_G.value
    ):
        return "root_only"
    return "unknown"


def _summary_exact_family(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> str:
    metadata = dict(summary.metadata or {})
    exact_family = str(metadata.get("exact_family", "") or "").strip()
    if exact_family:
        return exact_family
    if raw_payload is not None:
        cfg = dict(raw_payload.get("config", {}) or {})
        exact_family = str(cfg.get("exact_family", "") or "").strip()
        if exact_family:
            return exact_family
    return ""


def _selection_candidate_name(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> str:
    selection = dict(summary.selection or {})
    selected = str(selection.get("selected_candidate", "") or "").strip()
    if selected:
        return selected
    if raw_payload is not None:
        local_law = dict(raw_payload.get("local_law", {}) or {})
        raw_selection = dict(local_law.get("selection", {}) or {})
        selected = str(raw_selection.get("selected_candidate", "") or "").strip()
        if selected:
            return selected
    return ""


def _policy_by_key_or_name(
    summary: LocalLawRunSummary,
    name: str,
) -> Optional[LocalLawPolicyEvaluation]:
    policies = dict(summary.policies)
    if name in policies:
        return policies[str(name)]
    for key, policy in policies.items():
        if str(key) == str(name) or str(policy.name) == str(name):
            return policy
    return None


def _split_payload(
    summary: LocalLawRunSummary,
    policy: Optional[LocalLawPolicyEvaluation],
    *,
    split: str = "test",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if policy is None:
        return {}, {}
    split_metrics = dict(policy.split_metrics).get(split, {})
    if not isinstance(split_metrics, dict):
        return {}, {}
    local = dict(
        split_metrics.get("local_law", {}) or split_metrics.get("local_law_metrics", {}) or {}
    )
    downstream = dict(
        split_metrics.get("downstream", {}) or split_metrics.get("downstream_metrics", {}) or {}
    )
    return local, downstream


def _baseline_policy_metrics(
    summary: LocalLawRunSummary,
    *,
    split: str = "test",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    for policy in dict(summary.policies).values():
        role = str(policy.role.value if isinstance(policy.role, PolicyRole) else policy.role)
        if role != PolicyRole.BASELINE_G.value:
            continue
        local, downstream = _split_payload(summary, policy, split=split)
        if local or downstream:
            return local, downstream
    return {}, {}


def _selected_policy_metrics(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
    split: str = "test",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    learned = _policy_by_key_or_name(summary, "learned_g")
    local, downstream = _split_payload(summary, learned, split=split)
    if local or downstream:
        return local, downstream

    selected = _selection_candidate_name(summary, raw_payload=raw_payload)
    if selected:
        policy = _policy_by_key_or_name(summary, selected)
        local, downstream = _split_payload(summary, policy, split=split)
        if local or downstream:
            return local, downstream

    for policy in dict(summary.policies).values():
        role = str(policy.role.value if isinstance(policy.role, PolicyRole) else policy.role)
        if role != PolicyRole.CANDIDATE_G.value:
            continue
        local, downstream = _split_payload(summary, policy, split=split)
        if local or downstream:
            return local, downstream
    return {}, {}


def _primary_error(local: Mapping[str, Any], downstream: Mapping[str, Any]) -> float:
    for value in (
        downstream.get("oracle_target_abs_error"),
        downstream.get("root_error"),
        local.get("root_error"),
    ):
        metric = _safe_float(value)
        if math.isfinite(metric):
            return metric
    return float("nan")


def _classify_from_policy_metrics(
    *,
    baseline_local: Mapping[str, Any],
    baseline_downstream: Mapping[str, Any],
    selected_local: Mapping[str, Any],
    selected_downstream: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    from treepo._research.ctreepo.sim.core.law_stress_common import classify_law_stress

    baseline_primary = _primary_error(baseline_local, baseline_downstream)
    selected_primary = _primary_error(selected_local, selected_downstream)
    if not math.isfinite(baseline_primary) or not math.isfinite(selected_primary):
        return None

    try:
        assessment = classify_law_stress(
            baseline_c1=_safe_float(baseline_local.get("c1")),
            baseline_c2=_safe_float(baseline_local.get("c2")),
            baseline_c3=_safe_float(baseline_local.get("c3")),
            baseline_spread=_safe_float(
                baseline_local.get("schedule_spread", baseline_downstream.get("schedule_spread")),
                0.0,
            ),
            baseline_root_mae=baseline_primary,
            selected_c1=_safe_float(selected_local.get("c1")),
            selected_c2=_safe_float(selected_local.get("c2")),
            selected_c3=_safe_float(selected_local.get("c3")),
            selected_spread=_safe_float(
                selected_local.get("schedule_spread", selected_downstream.get("schedule_spread")),
                0.0,
            ),
            selected_root_mae=selected_primary,
        )
        return assessment.to_dict()
    except Exception:
        return None


def _markov_pair_key(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> str:
    cfg = dict(raw_payload.get("config", {}) or {}) if raw_payload is not None else {}
    metadata = dict(summary.metadata or {})
    key = {
        "family": str(summary.family),
        "suite_role": str(summary.suite_role),
        "n_regimes": int(cfg.get("n_regimes", metadata.get("n_regimes", 0)) or 0),
        "fixed_leaf_tokens": int(
            cfg.get("fixed_leaf_tokens", metadata.get("fixed_leaf_tokens", 0)) or 0
        ),
        "train_docs": int(summary.support_budget.train_docs),
        "val_docs": int(summary.support_budget.val_docs),
        "test_docs": int(summary.support_budget.test_docs),
        "audit_fraction": _safe_float(
            cfg.get(
                "audit_fraction",
                summary.support_budget.metadata.get("audit_fraction", float("nan")),
            )
        ),
        "root_weight": _safe_float(cfg.get("root_weight", float("nan"))),
        "state_dim": int(cfg.get("state_dim", 0) or 0),
        "hidden_dim": int(cfg.get("hidden_dim", 0) or 0),
        "n_epochs": int(cfg.get("n_epochs", 0) or 0),
        "feature_mode": str(cfg.get("feature_mode", metadata.get("feature_mode", "")) or ""),
        "model_family": str(cfg.get("model_family", metadata.get("model_family", "")) or ""),
        "effective_data_seed": int(
            cfg.get("effective_data_seed", cfg.get("data_seed", cfg.get("seed", 0))) or 0
        ),
        "effective_model_seed": int(
            cfg.get("effective_model_seed", cfg.get("model_seed", cfg.get("seed", 0))) or 0
        ),
        "effective_val_seed": int(cfg.get("effective_val_seed", 0) or 0),
        "effective_test_seed": int(cfg.get("effective_test_seed", 0) or 0),
    }
    return json.dumps(key, sort_keys=True)


def compute_law_stress_for_summary(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Compute law-stress classification from a unified ``LocalLawRunSummary``.

    Compares the ``learned_g`` (or ``candidate_g``) policy against the
    ``baseline_g`` policy using the shared ``classify_law_stress`` function.

    Returns ``None`` if the required policies are missing.
    """
    if raw_payload is not None:
        precomputed = dict(raw_payload.get("local_law", {}) or {}).get("law_stress")
        if isinstance(precomputed, dict):
            selected = _selection_candidate_name(summary, raw_payload=raw_payload)
            if selected:
                chosen = dict(precomputed.get(str(selected), {}) or {})
                if chosen and "bundle_status" in chosen:
                    return chosen
            learned = dict(precomputed.get("learned_g", {}) or {})
            if learned and "bundle_status" in learned:
                return learned
            valid = [
                dict(value)
                for value in precomputed.values()
                if isinstance(value, dict) and "bundle_status" in value
            ]
            if len(valid) == 1:
                return valid[0]

    baseline_local, baseline_downstream = _baseline_policy_metrics(summary)
    learned_local, learned_downstream = _selected_policy_metrics(summary, raw_payload=raw_payload)
    if not baseline_local or not learned_local:
        return None
    return _classify_from_policy_metrics(
        baseline_local=baseline_local,
        baseline_downstream=baseline_downstream,
        selected_local=learned_local,
        selected_downstream=learned_downstream,
    )


def collect_law_stress_assessments(
    records: Sequence[Tuple[str | Path, LocalLawRunSummary, Mapping[str, Any]]],
) -> Sequence[Dict[str, Any]]:
    assessments: list[Dict[str, Any]] = []
    markov_baselines: Dict[str, Dict[str, Any]] = {}
    markov_pending: list[Dict[str, Any]] = []

    for source_path, summary, raw_payload in records:
        payload = dict(raw_payload)
        family = str(summary.family)
        law_package = _summary_law_package(summary, raw_payload=payload)
        exact_family = _summary_exact_family(summary, raw_payload=payload)

        if family == "markov_ops_count":
            if exact_family:
                continue
            if law_package == "root_only":
                baseline_local, baseline_downstream = _baseline_policy_metrics(summary)
                if baseline_local:
                    markov_baselines[_markov_pair_key(summary, raw_payload=payload)] = {
                        "path": str(source_path),
                        "local": baseline_local,
                        "downstream": baseline_downstream,
                    }
                continue

            selected_local, selected_downstream = _selected_policy_metrics(
                summary, raw_payload=payload
            )
            if not selected_local:
                continue
            markov_pending.append(
                {
                    "family": family,
                    "law_package": law_package,
                    "scenario_key": _markov_pair_key(summary, raw_payload=payload),
                    "source_path": str(source_path),
                    "local": selected_local,
                    "downstream": selected_downstream,
                }
            )
            continue

        assessment = compute_law_stress_for_summary(summary, raw_payload=payload)
        if assessment is None:
            continue
        assessments.append(
            {
                "family": family,
                "law_package": law_package,
                "assessment": assessment,
                "source_path": str(source_path),
            }
        )

    for pending in markov_pending:
        baseline = markov_baselines.get(str(pending["scenario_key"]))
        if baseline is None:
            continue
        assessment = _classify_from_policy_metrics(
            baseline_local=dict(baseline["local"]),
            baseline_downstream=dict(baseline["downstream"]),
            selected_local=dict(pending["local"]),
            selected_downstream=dict(pending["downstream"]),
        )
        if assessment is None:
            continue
        assessments.append(
            {
                "family": str(pending["family"]),
                "law_package": str(pending["law_package"]),
                "assessment": assessment,
                "source_path": str(pending["source_path"]),
                "baseline_source_path": str(baseline["path"]),
            }
        )

    return assessments


__all__ = [
    "collect_law_stress_assessments",
    "compute_law_stress_for_summary",
    "load_or_backfill_local_law_payload",
]
