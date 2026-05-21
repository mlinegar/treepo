from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from treepo._research.ctreepo.sim.objective_semantics import (
    discrepancy_benchmark_objective_semantics,
    latent_lambda_objective_semantics,
    mergeable_target_objective_semantics,
)


def _safe_float(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    return float(out)


def safe_objective_backfill(payload: Mapping[str, Any]) -> Optional[Dict[str, object]]:
    if not isinstance(payload, Mapping):
        return None
    if isinstance(payload.get("objective"), Mapping) and dict(payload.get("objective", {}) or {}):
        return dict(payload.get("objective", {}) or {})

    family = str(payload.get("family", "") or "").strip()
    cfg_raw = payload.get("config", {})
    cfg = dict(cfg_raw or {}) if isinstance(cfg_raw, Mapping) else {}
    target_kind = str(payload.get("target_kind", "") or "").strip()

    if family == "leaf_local_mixture_utility":
        lam = _safe_float(cfg.get("lambda_multiplier"))
        if lam is None:
            return None
        return latent_lambda_objective_semantics(
            name="leaf_local_mixture_utility_target",
            optimized_against="document_level_local_mixture_utility",
            lambda_multiplier=float(lam),
            linear_component_name="topic_mixture_linear_term",
            interaction_component_name="local_topic_mixture_quadratic_term",
            weighting_scheme="linear_plus_lambda_local_quadratic_utility",
            metadata={
                "family": "leaf_local_mixture_utility",
                "target_kind": str(target_kind or "local_nonlinear_leaf_sum"),
                "latent_partition_mode": str(cfg.get("latent_partition_mode", "") or ""),
                "analysis_partition_mode": str(cfg.get("analysis_partition_mode", "") or ""),
            },
        )

    if family == "lda_tree_recovery":
        lam = _safe_float(cfg.get("lambda_multiplier"))
        if lam is None:
            return None
        return latent_lambda_objective_semantics(
            name="lda_document_utility_target",
            optimized_against="document_level_latent_utility",
            lambda_multiplier=float(lam),
            linear_component_name="topic_mixture_linear_term",
            interaction_component_name="topic_mixture_quadratic_term",
            weighting_scheme="linear_plus_lambda_quadratic_utility",
            metadata={"family": "lda_tree_recovery"},
        )

    if family == "lda_tree_recovery_learned":
        lam = _safe_float(cfg.get("lambda_multiplier"))
        if lam is None:
            return None
        return latent_lambda_objective_semantics(
            name="lda_document_utility_target",
            optimized_against="learned_approximation_to_document_utility",
            lambda_multiplier=float(lam),
            linear_component_name="topic_mixture_linear_term",
            interaction_component_name="topic_mixture_quadratic_term",
            weighting_scheme="linear_plus_lambda_quadratic_utility",
            metadata={"family": "lda_tree_recovery_learned"},
        )

    if family == "segment_lda_ops_weight_recovery":
        lam = _safe_float(cfg.get("lambda_multiplier"))
        if lam is None:
            return None
        return latent_lambda_objective_semantics(
            name="segment_lda_oracle_target",
            optimized_against="ridge_regression_on_oracle_span_labels",
            lambda_multiplier=float(lam),
            linear_component_name="latent_topic_counts",
            interaction_component_name="latent_topic_bigrams",
            metadata={
                "family": "segment_lda_ops_weight_recovery",
                "topic_process": str(cfg.get("topic_process", "") or ""),
                "topic_phi_estimator": str(cfg.get("topic_phi_estimator", "") or ""),
                "feature_inference": str(cfg.get("feature_inference", "") or ""),
            },
        )

    if family == "lda_tree_utility_vector":
        return mergeable_target_objective_semantics(
            name="lda_utility_vector_target",
            optimized_against="utility_vector_labels",
            target_kind=str(target_kind or "utility_vector"),
            metadata={"family": "lda_tree_utility_vector"},
        )

    if family == "segmented_lda_ctreepo":
        return discrepancy_benchmark_objective_semantics(
            name="segmented_lda_ctreepo_benchmark",
            optimized_against="ridge_calibration_on_queried_leaves",
            benchmark_metric_name="root_l1_mean",
            metadata={
                "family": "segmented_lda_ctreepo",
                "calibration_leaf_query_rate": _safe_float(cfg.get("calibration_leaf_query_rate")),
                "eval_leaf_query_rate": _safe_float(cfg.get("eval_leaf_query_rate")),
                "eval_internal_query_rate": _safe_float(cfg.get("eval_internal_query_rate")),
            },
        )

    if family == "tensor_lda_book_benchmark":
        return discrepancy_benchmark_objective_semantics(
            name="tensor_lda_book_benchmark",
            optimized_against="ridge_calibration_on_queried_chapters",
            benchmark_metric_name="root_l1_mean",
            metadata={
                "family": "tensor_lda_book_benchmark",
                "calibration_leaf_query_rate": _safe_float(cfg.get("calibration_leaf_query_rate")),
                "eval_leaf_query_rate": _safe_float(cfg.get("eval_leaf_query_rate")),
                "eval_internal_query_rate": _safe_float(cfg.get("eval_internal_query_rate")),
            },
        )

    return None


__all__ = ["safe_objective_backfill"]
