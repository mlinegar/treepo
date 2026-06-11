from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F

from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (
    AdditiveCountSketch,
    BUDGETED_SAMPLING_SCHEME_RANDOM_WITHOUT_REPLACEMENT,
    build_budgeted_train_supervision_manifest,
    budgeted_manifest_plan_maps,
    BudgetedTrainSupervisionDocPlan,
    BudgetedTrainSupervisionManifest,
    ChangepointMarkovDoc,
    DEFAULT_NORMALIZED_LOCAL_LAW_WEIGHT,
    MarkovOPSDataBundle,
    OPSCountConfig,
    SketchMetrics,
    TrainFitDiagnostics,
    _CountDoc,
    _build_objective_summary,
    _dense_doc_matrix_supervision_dataset,
    _doc_level_feature_matrix,
    _doc_level_supervision_dataset,
    _doc_root_targets,
    _doc_token_ngram_feature_matrix,
    _eval_root_predictions,
    _eval_exact_family,
    _exact_match_rate,
    _ExactState,
    _exact_from_span,
    _exact_merge,
    _leaf_spans,
    _oracle_count,
    _leaf_ridge_tree_supervision_dataset,
    _markov_corpus_signature,
    _prepare_count_docs,
    _prepare_doc_level_count_docs,
    _resolve_runtime_seeds,
    _sketch_metric_alias_payload,
    _set_global_seed,
    build_markov_changepoint_ops_count_data_bundle,
)
from treepo._research.ctreepo.sim.core.markov_comparison_surface import (
    FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS,
    apply_comparable_surface_to_mapping,
    comparable_surface_snapshot_from_mapping,
    comparison_surface_diff,
    infer_markov_comparison_mode,
    normalize_markov_comparison_mode,
    resolve_markov_benchmark_locked_fields,
    resolve_markov_comparable_surface,
)
from treepo._research.ctreepo.sim.core.markov_neural_operator_baselines import (
    FNOCountSketch,
    FNO_TREE_C2_EXACT_WITNESS_KIND,
    FNO_TREE_C2_METRIC_KIND,
    FNO_TREE_C2_PROXY_METRIC_KIND,
    HAS_NEURAL_OPERATOR,
    LEAN_APPROX_BUNDLE_OF_NODEWISE_REF,
    LEAN_MARKOV_COUNT_ONLY_INVALID_REF,
    LEAN_MARKOV_COUNT_SKETCH_REF,
    LEAN_MARKOV_OBSERVED_TOKEN_EXACT_SKETCH_REF,
    LEAN_MARKOV_OBSERVED_TOKEN_RECOVERABILITY_REF,
    LEAN_MARKOV_PATH_COUNT_SUPPORT_EXACT_REF,
    LEAN_MARKOV_PATH_SUPPORT_EXACT_REF,
    LEAN_MARKOV_REPRESENTATION_COUNT_TRANSPORT_REF,
    LEAN_MARKOV_REPRESENTATION_EXACT_PASS_REF,
    LEAN_MARKOV_REPRESENTATION_ZERO_ROOT_COUNT_ERROR_REF,
    LEAN_MARKOV_RUNTIME_AUDIT_STOCHASTIC_APPROX_REF,
    LEAN_MARKOV_SUFFICIENCY_DECODER_REF,
    LEAN_MARKOV_ZERO_BAYES_ERROR_REF,
    LEAN_SKETCH_CODEC_EXACT_ASSUMPTIONS_REF,
    PrototypeClassifier,
    _deterministic_sample_indices_from_ordering,
    _class_setup,
    _deterministic_sample_ordering,
    _exact_markov_root_error_decomposition,
    _exact_projected_root_count_from_states,
    _eval_fno_model,
    estimate_tree_gpu_batch_store_bytes,
    eval_scorefiber_root_probe_metrics,
    _eval_fno_teacher_first_decomposition_metrics,
    _FNOCountDoc,
    _fit_cnn1d_baseline_with_predictions,
    _fit_fno_baseline_with_predictions,
    _fit_mlp_bigram_baseline_with_predictions,
    _gpu_runtime_config_from_ops_config,
    _prepare_fno_count_docs,
    _trim_host_allocator,
    train_fno_tree,
)
from treepo._research.ctreepo.sim.core.markov_full_doc_provenance import (
    full_doc_baseline_provenance,
)
from treepo._research.ctreepo.sim.core.full_doc_config_codec import (
    runtime_config_overrides_from_config_like,
)
from treepo._research.ctreepo.sim.core.run_intent import (
    intent_diff,
    intent_hash,
    intent_is_complete,
    materialize_tree_run_intent,
)
from treepo._research.ctreepo.sim.suite.markov_observed_token_policy import (
    resolve_markov_observed_token_policy,
)
from treepo._research.ctreepo.sim.core.markov_hazard_panels import (
    DEFAULT_STICKY_STRUCTURAL_V2_CELL_ID,
    STICKY_STRUCTURAL_V2_CELL_SPECS,
    STICKY_STRUCTURAL_V2_LEGACY_ALIAS_MAP,
    canonicalize_structural_v2_cell_id as _canonicalize_structural_v2_cell_id,
    sticky_markov_mean_segment_length as _sticky_markov_mean_segment_length,
    sticky_markov_switch_probability as _sticky_markov_switch_probability,
    sticky_recoverable_config_overrides as _sticky_recoverable_config_overrides,
    sticky_recoverable_config_overrides_t2048 as _sticky_recoverable_config_overrides_t2048,
    sticky_structural_config_overrides as _sticky_structural_config_overrides,
)
from treepo._research.training.supervision import (
    DenseScalarRidgeModelConfig,
    DenseScalarRidgeTrainingConfig,
    fit_dense_scalar_ridge_regressor,
    predict_dense_scalar_ridge_regressor,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


def _canonical_diagnostic_bundle_path(output_dir_name: str) -> Path:
    return (
        REPO_ROOT
        / "outputs"
        / str(output_dir_name).strip()
        / "markov_data"
        / "observed_token_bundle.json"
    )


def _structural_diagnostic_bundle_path(grid_name: str, cell_id: str) -> Path:
    return _canonical_diagnostic_bundle_path(
        f"markov_observed_token_{str(grid_name).strip()}__{str(cell_id).strip()}"
    )


def _structural_expanded_diagnostic_bundle_path(
    grid_name: str,
    cell_id: str,
    expansion_tag: str,
) -> Path:
    return _canonical_diagnostic_bundle_path(
        "markov_observed_token_"
        f"{str(grid_name).strip()}_{str(expansion_tag).strip()}__{str(cell_id).strip()}"
    )


CANONICAL_DIAGNOSTIC_BUNDLES = {
    "demo_v1": _canonical_diagnostic_bundle_path("markov_observed_token_suite_demo_v1"),
    "recoverable_v4": _canonical_diagnostic_bundle_path("markov_observed_token_recoverable_v4"),
    "recoverable_10x_v1": _canonical_diagnostic_bundle_path("markov_observed_token_recoverable_10x_v1"),
    "recoverable_v5": _canonical_diagnostic_bundle_path("markov_observed_token_recoverable_v5"),
    "recoverable_10x_v2": _canonical_diagnostic_bundle_path("markov_observed_token_recoverable_10x_v2"),
    "recoverable_v4_t128": _canonical_diagnostic_bundle_path(
        "markov_observed_token_recoverable_v4_t128"
    ),
    "recoverable_v5_t128": _canonical_diagnostic_bundle_path(
        "markov_observed_token_recoverable_v5_t128"
    ),
    "recoverable_v5_t2048": _canonical_diagnostic_bundle_path(
        "markov_observed_token_recoverable_v5_t2048"
    ),
    "recoverable_10x_v1_t128": _canonical_diagnostic_bundle_path(
        "markov_observed_token_recoverable_10x_v1_t128"
    ),
    "recoverable_20x_v1_t128": _canonical_diagnostic_bundle_path(
        "markov_observed_token_recoverable_20x_v1_t128"
    ),
    "recoverable_20x_v2_t128": _canonical_diagnostic_bundle_path(
        "markov_observed_token_recoverable_20x_v2_t128"
    ),
}

VALID_BASELINE_FAMILIES = (
    "official_fno",
    "official_fno_sumlen",
    "cnn1d",
    "mlp_bigram",
    "palette_block_exact",
    "raw_token_ngram_ridge",
    "ridge_control",
    "tree_ridge_leaf",
    "tree_doc_ridge",
    "tree_neural_c2",
    "tree_neural_c2c3",
    "tree_neural",
)
VALID_HARDNESS_GRIDS = (
    "structural_core_v1",
    "structural_core_v1_t128",
    "structural_core_v2",
    "structural_core_v2_t128",
)
DEFAULT_DIAGNOSTIC_BASELINE_FAMILIES = (
    "official_fno",
    "official_fno_sumlen",
    "cnn1d",
    "mlp_bigram",
    "raw_token_ngram_ridge",
)
DEFAULT_STRUCTURAL_CORE_BASELINE_FAMILIES = (
    "official_fno",
    "official_fno_sumlen",
    "cnn1d",
    "palette_block_exact",
)
TREE_NEURAL_FAMILY_PROFILES: Dict[str, Dict[str, float]] = {
    "tree_neural_c2": {"c1_relative_weight": 0.0, "c2_relative_weight": 1.0, "c3_relative_weight": 0.0},
    "tree_neural_c2c3": {"c1_relative_weight": 0.0, "c2_relative_weight": 2.0, "c3_relative_weight": 1.0},
    "tree_neural": {"c1_relative_weight": 1.0, "c2_relative_weight": 1.0, "c3_relative_weight": 1.0},
}
TREE_NEURAL_BASELINE_FAMILIES = frozenset(TREE_NEURAL_FAMILY_PROFILES.keys())
TREE_NEURAL_LEGACY_PROFILE_FAMILIES = frozenset({"tree_neural_c2", "tree_neural_c2c3"})
TREE_NEURAL_EXACT_LOCKED_MODES = frozenset(
    {
        "official_fno_one_tree_identity",
        "official_fno_runtime_identity",
    }
)
TREE_NEURAL_SEMANTIC_CONFIG_FIELDS: tuple[str, ...] = (
    "pipeline_tree_reference_label",
    "local_law_weight",
    "c1_relative_weight",
    "c2_relative_weight",
    "c3_relative_weight",
    "leaf_supervision_kind",
    "leaf_label_rate",
    "leaf_exact_supervision",
    "internal_supervision_kind",
    "internal_label_rate",
    "task_objective_weight",
    "comparison_mode",
    "tree_exact_collapse_mode",
    "tree_c2_mode",
    "tree_training_schedule",
    "tree_supervision_source",
    "package_semantics",
)
TREE_NEURAL_OBJECTIVE_SNAPSHOT_FIELDS: tuple[str, ...] = (
    "parameterization",
    "weighting_scheme",
    "optimization_root_weight",
    "local_law_c1_weight",
    "local_law_c2_weight",
    "local_law_c3_weight",
    "task_objective_weight_source",
    "proxy_schedule_consistency_weight",
    "theorem_local_law_total_weight",
    "proxy_schedule_term_total_weight",
    "objective_surface_name",
    "objective_weights_active",
    "semantics_version",
)
CURRENT_TREE_NEURAL_SEMANTICS_VERSION = "tree_neural_objective_v4"
FAIR_FNO_PARITY_CONFIG_LABEL = "fair_fno_v1"
REPORTED_SPLITS = ("train", "val", "test")
PRIMARY_REPORT_METRIC = "test_root_mae_mean"
PRIMARY_REPORT_SPLIT = "test"
PRIMARY_REPORT_TARGET = "root_count"
PRIMARY_REPORT_WEIGHTING = "unweighted_mae"
DEV_SELECTION_METRIC = "val_root_mae_mean"
SPLIT_METRIC_BASES = (
    "root_mae",
    "leaf_mae",
    "merge_mae",
    "schedule_spread_mean",
    "c2_count_drift_r1_mae",
    "c2_count_drift_r2_mae",
    "c2_count_drift_r4_mae",
    "c2_root_count_drift_r1_mae",
    "c2_root_count_drift_r2_mae",
    "c2_root_count_drift_r4_mae",
    "c2_state_replay_mse",
    "c2_idempotence_mae",
)
DIAGNOSTIC_ONLY_METRIC_GROUPS = {
    "split_diagnostics": [
        "train_root_mae_mean",
        "val_root_mae_mean",
        "train_exact_match_rate_mean",
        "val_exact_match_rate_mean",
        "test_exact_match_rate_mean",
    ],
    "optimization_objective": [
        "fit_diagnostics.train_loss_curve",
        "fit_diagnostics.train_loss_final",
        "fit_diagnostics.selection_metric_curve",
    ],
    "unweighted_split_objectives": [
        "val_unweighted_full_law_objective_mean",
        "val_unweighted_active_objective_mean",
        "test_unweighted_full_law_objective_mean",
        "test_unweighted_active_objective_mean",
    ],
    "law_metrics": [
        "test_leaf_mae_mean",
        "test_c2_count_drift_r1_mae_mean",
        "test_merge_mae_mean",
        "test_schedule_spread_mean_mean",
    ],
    "c2_runtime_diagnostics": [
        "test_c2_count_drift_r2_mae_mean",
        "test_c2_count_drift_r4_mae_mean",
        "test_c2_root_count_drift_r1_mae_mean",
        "test_c2_root_count_drift_r2_mae_mean",
        "test_c2_root_count_drift_r4_mae_mean",
        "test_c2_state_replay_mse_mean",
    ],
}
FULL_DOC_ONLY_BUDGET_FAMILIES = frozenset(
    {
        "official_fno",
        "official_fno_sumlen",
        "tree_doc_ridge",
        "raw_token_ngram_ridge",
        "cnn1d",
        "mlp_bigram",
        "palette_block_exact",
    }
)
TREE_BUDGET_ELIGIBLE_FAMILIES = frozenset(TREE_NEURAL_BASELINE_FAMILIES)
ORACLE_BUDGET_STUDY_NAME = "oracle_budget_share_frontier"
MARKOV_FULL_DOC_OBJECTIVE_SURFACE = "markov_full_doc_normalized_task_local_law_surface"
TREEPO_REGULARIZED_OBJECTIVE_SURFACE = "treepo_regularized_objective"
PAPER_TO_LEAN_LOCAL_LAW_MAPPING = {
    "C1": "L1",
    "C2": "L3",
    "C3": "L2",
}

_METRIC_ALIAS_MAP: Dict[str, Tuple[str, ...]] = {
    "c2_count_drift_r1_mae": ("c2_idempotence_mae", "c2_r1_mae"),
    "c2_count_drift_r2_mae": ("c2_r2_mae",),
    "c2_count_drift_r4_mae": ("c2_r4_mae",),
    "c2_root_count_drift_r1_mae": ("resummary_root_drift_r1",),
    "c2_root_count_drift_r2_mae": ("resummary_root_drift_r2",),
    "c2_root_count_drift_r4_mae": ("resummary_root_drift_r4",),
    "c2_idempotence_mae": ("c2_count_drift_r1_mae", "c2_r1_mae"),
    "c2_r1_mae": ("c2_count_drift_r1_mae", "c2_idempotence_mae"),
    "c2_r2_mae": ("c2_count_drift_r2_mae",),
    "c2_r4_mae": ("c2_count_drift_r4_mae",),
    "resummary_root_drift_r1": ("c2_root_count_drift_r1_mae",),
    "resummary_root_drift_r2": ("c2_root_count_drift_r2_mae",),
    "resummary_root_drift_r4": ("c2_root_count_drift_r4_mae",),
}


def _report_metric_contract_payload() -> Dict[str, Any]:
    return {
        "primary_report_metric": PRIMARY_REPORT_METRIC,
        "primary_report_split": PRIMARY_REPORT_SPLIT,
        "primary_report_target": PRIMARY_REPORT_TARGET,
        "primary_report_weighting": PRIMARY_REPORT_WEIGHTING,
        "dev_selection_metric": DEV_SELECTION_METRIC,
        "markov_c2_primary_metric": "test_c2_count_drift_r1_mae_mean",
        "markov_c2_deprecated_alias": "test_c2_idempotence_mae_mean",
        "markov_c2_exact_witness_metric": FNO_TREE_C2_EXACT_WITNESS_KIND,
        "markov_c2_proxy_metric": "test_c2_state_replay_mse_mean",
        "diagnostic_metric_role": "diagnostic_only",
        "diagnostic_only_metric_groups": {
            key: list(values) for key, values in DIAGNOSTIC_ONLY_METRIC_GROUPS.items()
        },
    }


def _budget_manifest_metadata(
    manifest: BudgetedTrainSupervisionManifest | None,
) -> Dict[str, Any]:
    if manifest is None:
        return {
            "budget_total_calls": 0,
            "budget_total_calls_per_doc": 0.0,
            "budget_total_calls_used": 0,
            "budget_utilization": float("nan"),
            "full_doc_budget_share": 1.0,
            "full_doc_calls_requested": 0,
            "full_doc_calls_total": 0,
            "local_calls_requested": 0,
            "local_calls_total": 0,
            "doc_consumption_mode": "",
            "local_split_mode": "",
            "local_allocation_policy": "",
            "sampling_scheme": "",
            "effective_full_doc_mass_total": 0.0,
            "effective_full_doc_mass_per_doc": 0.0,
            "document_mass_share": 0.0,
            "leaf_mass_share": 0.0,
            "internal_mass_share": 0.0,
            "document_call_share": 0.0,
            "leaf_call_share": 0.0,
            "internal_call_share": 0.0,
            "actual_doc_tokens_mean": 0.0,
            "actual_doc_tokens_unique": [],
            "supervision_source": "",
            "local_estimand_mode": "",
            "package_semantics": "",
            "mass_target_per_doc": float("nan"),
            "requested_root_mass_per_doc": 0.0,
            "realized_root_mass_per_doc": 0.0,
            "realized_leaf_mass_per_doc": 0.0,
            "realized_internal_mass_per_doc": 0.0,
            "leaf_propensity_mean": 0.0,
            "internal_propensity_mean": 0.0,
            "local_weight_ess": 0.0,
            "local_weight_max": 0.0,
            "doc_touch_rate": 0.0,
            "mean_labels_per_touched_doc": 0.0,
            "touched_docs_total": 0,
            "doc_plans": [],
        }
    return {
        "budget_total_calls": int(manifest.budget_total_calls),
        "budget_total_calls_per_doc": float(manifest.budget_total_calls_per_doc),
        "budget_total_calls_used": int(manifest.budget_total_calls_used),
        "budget_utilization": float(manifest.budget_utilization),
        "full_doc_budget_share": float(manifest.full_doc_budget_share),
        "full_doc_calls_requested": int(manifest.full_doc_calls_requested),
        "full_doc_calls_total": int(manifest.full_doc_calls_total),
        "local_calls_requested": int(manifest.local_calls_requested),
        "local_calls_total": int(manifest.local_calls_total),
        "doc_consumption_mode": str(manifest.doc_consumption_mode),
        "local_split_mode": str(manifest.local_split_mode),
        "local_allocation_policy": str(manifest.local_allocation_policy),
        "sampling_scheme": str(manifest.sampling_scheme),
        "effective_full_doc_mass_total": float(manifest.effective_full_doc_mass_total),
        "effective_full_doc_mass_per_doc": float(manifest.effective_full_doc_mass_per_doc),
        "document_mass_share": float(manifest.document_mass_share),
        "leaf_mass_share": float(manifest.leaf_mass_share),
        "internal_mass_share": float(manifest.internal_mass_share),
        "document_call_share": float(manifest.document_call_share),
        "leaf_call_share": float(manifest.leaf_call_share),
        "internal_call_share": float(manifest.internal_call_share),
        "actual_doc_tokens_mean": float(manifest.actual_doc_tokens_mean),
        "actual_doc_tokens_unique": [
            int(value) for value in manifest.actual_doc_tokens_unique
        ],
        "supervision_source": str(manifest.supervision_source),
        "local_estimand_mode": str(manifest.local_estimand_mode),
        "package_semantics": str(manifest.package_semantics),
        "mass_target_per_doc": float(manifest.mass_target_per_doc),
        "requested_root_mass_per_doc": float(
            manifest.requested_root_mass_per_doc
        ),
        "realized_root_mass_per_doc": float(manifest.realized_root_mass_per_doc),
        "realized_leaf_mass_per_doc": float(manifest.realized_leaf_mass_per_doc),
        "realized_internal_mass_per_doc": float(
            manifest.realized_internal_mass_per_doc
        ),
        "leaf_propensity_mean": float(manifest.leaf_propensity_mean),
        "internal_propensity_mean": float(manifest.internal_propensity_mean),
        "local_weight_ess": float(manifest.local_weight_ess),
        "local_weight_max": float(manifest.local_weight_max),
        "doc_touch_rate": float(manifest.doc_touch_rate),
        "mean_labels_per_touched_doc": float(manifest.mean_labels_per_touched_doc),
        "touched_docs_total": int(manifest.touched_docs_total),
        "doc_plans": [
            {
                "doc_index": int(plan.doc_index),
                "doc_tokens": int(plan.doc_tokens),
                "document_mode": str(plan.document_mode),
                "leaf_indices": [int(value) for value in plan.leaf_indices],
                "internal_indices": [int(value) for value in plan.internal_indices],
                "raw_call_cost": int(plan.raw_call_cost),
                "document_mass": float(plan.document_mass),
                "leaf_mass": float(plan.leaf_mass),
                "internal_mass": float(plan.internal_mass),
                "leaf_propensity": float(plan.leaf_propensity),
                "internal_propensity": float(plan.internal_propensity),
                "effective_full_doc_mass": float(plan.effective_full_doc_mass),
            }
            for plan in manifest.doc_plans
        ],
    }


def _budget_metadata_for_payload(
    *,
    config: OPSCountConfig,
    fit_budget_manifest: Mapping[str, Any] | None,
    fallback_manifest: BudgetedTrainSupervisionManifest | None,
    train_doc_count: int,
) -> Dict[str, Any]:
    budget_metadata = dict(
        fit_budget_manifest or _budget_manifest_metadata(fallback_manifest)
    )
    if fit_budget_manifest is not None or fallback_manifest is not None:
        return budget_metadata

    budget_total_calls_per_doc = _clamp01(
        float(getattr(config, "budget_total_calls_per_doc", 0.0) or 0.0)
    )
    full_doc_budget_share = _clamp01(
        float(getattr(config, "full_doc_budget_share", 1.0) or 1.0)
    )
    requested_root_mass_per_doc = _clamp01(
        float(budget_total_calls_per_doc) * float(full_doc_budget_share)
    )
    mass_target_per_doc = _safe_float(
        getattr(config, "mass_target_per_doc", float("nan")),
        default=float("nan"),
    )
    doc_consumption_mode = str(getattr(config, "doc_consumption_mode", "") or "")
    local_split_mode = str(getattr(config, "local_split_mode", "") or "")
    local_allocation_policy = str(
        getattr(config, "local_allocation_policy", "") or ""
    )
    local_rates_active = bool(
        abs(float(getattr(config, "leaf_label_rate", 0.0) or 0.0)) > 1e-12
        or (
            str(getattr(config, "internal_supervision_kind", "none") or "none")
            .strip()
            .lower()
            != "none"
            and abs(float(getattr(config, "internal_label_rate", 0.0) or 0.0)) > 1e-12
        )
    )
    budget_metadata.update(
        {
            "budget_total_calls": int(getattr(config, "budget_total_calls", 0) or 0),
            "budget_total_calls_per_doc": float(budget_total_calls_per_doc),
            "mass_target_per_doc": float(mass_target_per_doc),
            "requested_root_mass_per_doc": float(requested_root_mass_per_doc),
            "full_doc_budget_share": float(full_doc_budget_share),
            "doc_consumption_mode": str(doc_consumption_mode),
            "local_split_mode": str(local_split_mode),
            "local_allocation_policy": str(local_allocation_policy),
        }
    )
    if (
        doc_consumption_mode
        and not local_rates_active
        and not math.isfinite(float(mass_target_per_doc))
        and requested_root_mass_per_doc > 0.0
    ):
        budget_metadata.update(
            {
                "effective_full_doc_mass_total": float(
                    float(train_doc_count) * float(requested_root_mass_per_doc)
                ),
                "effective_full_doc_mass_per_doc": float(requested_root_mass_per_doc),
                "document_mass_share": 1.0,
                "leaf_mass_share": 0.0,
                "internal_mass_share": 0.0,
                "document_call_share": 1.0,
                "leaf_call_share": 0.0,
                "internal_call_share": 0.0,
                "doc_touch_rate": float(requested_root_mass_per_doc),
                "mean_labels_per_touched_doc": 1.0,
                "touched_docs_total": int(
                    round(float(train_doc_count) * float(requested_root_mass_per_doc))
                ),
            }
    )
    return budget_metadata


def _requires_explicit_budget_manifest_for_run(
    *,
    baseline_family: str,
    config: OPSCountConfig,
) -> bool:
    normalized_family = _normalize_baseline_family(baseline_family)
    budget_active = bool(
        int(getattr(config, "budget_total_calls", 0) or 0) > 0
        or float(getattr(config, "budget_total_calls_per_doc", 0.0) or 0.0) > 0.0
    )
    if not budget_active:
        return False
    if not (
        normalized_family in TREE_BUDGET_ELIGIBLE_FAMILIES
        or normalized_family in FULL_DOC_ONLY_BUDGET_FAMILIES
    ):
        return False
    tree_supervision_source = str(
        getattr(config, "tree_supervision_source", "rate") or "rate"
    ).strip().lower()
    if normalized_family == "tree_neural" and tree_supervision_source == "manifest":
        return True
    return True


def _selected_document_indices_from_budget_manifest(
    manifest: BudgetedTrainSupervisionManifest | None,
    *,
    n_items: int | None = None,
    seed: int | None = None,
) -> tuple[int, ...]:
    if manifest is None:
        return tuple()
    explicit = tuple(
        int(plan.doc_index)
        for plan in manifest.doc_plans
        if str(plan.document_mode).strip()
    )
    if explicit:
        return explicit
    doc_count = int(n_items or 0)
    if doc_count <= 0 or seed is None:
        return tuple()
    requested_root_mass_per_doc = _clamp01(
        float(getattr(manifest, "requested_root_mass_per_doc", 0.0) or 0.0)
    )
    if requested_root_mass_per_doc <= 0.0:
        return tuple()
    ordering = _deterministic_sample_ordering(
        n_items=int(doc_count),
        seed=int(seed) + 71_000,
    )
    return _explicit_indices_from_rate(
        n_items=int(doc_count),
        rate=float(requested_root_mass_per_doc),
        ordering=ordering,
        seed=int(seed) + 71_000,
    )


def _subset_docs_by_indices(
    docs: Sequence[ChangepointMarkovDoc],
    indices: Sequence[int],
) -> tuple[ChangepointMarkovDoc, ...]:
    return tuple(
        docs[int(index)]
        for index in indices
        if 0 <= int(index) < len(docs)
    )


def _split_final_metric_fields(
    *,
    fit: Mapping[str, Any],
    fit_diag: TrainFitDiagnostics,
) -> Dict[str, float]:
    split_metrics = {
        "train": fit["train_metrics"],
        "val": fit["val_metrics"],
        "test": fit["test_metrics"],
    }
    split_exact_match = {
        "train": float(fit_diag.train_exact_match_rate),
        "val": float(fit_diag.val_exact_match_rate),
        "test": float(fit_diag.test_exact_match_rate),
    }
    fields: Dict[str, float] = {}
    for split, metrics in split_metrics.items():
        fields[f"{split}_root_mae"] = float(getattr(metrics, "root_mae", float("nan")))
        fields[f"{split}_exact_match_rate"] = float(split_exact_match[split])
    return fields


def _summary_stats(
    values: np.ndarray,
    *,
    prefix: str,
) -> Dict[str, float]:
    has_finite = bool(np.isfinite(values).any())
    if not has_finite:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_std": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.nanmean(values)),
        f"{prefix}_median": float(np.nanmedian(values)),
        f"{prefix}_min": float(np.nanmin(values)),
        f"{prefix}_max": float(np.nanmax(values)),
        f"{prefix}_std": float(np.nanstd(values)),
    }


def _is_missing_field(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (float, np.floating)):
        return not np.isfinite(float(value))
    return False


def _safe_float(value: Any, *, default: float = float("nan")) -> float:
    from treepo._research.ctreepo.sim.util import safe_float
    return safe_float(value, default=default)


def _clamp01(value: Any) -> float:
    return float(min(1.0, max(0.0, _safe_float(value, default=0.0))))


def _nominal_rate_propensity(
    *,
    n_items: int,
    rate: float,
) -> float:
    total = int(max(0, n_items))
    if total <= 0:
        return 0.0
    sample_rate = _clamp01(rate)
    if sample_rate <= 0.0:
        return 0.0
    if sample_rate >= 1.0:
        return 1.0
    return float(sample_rate)


def _explicit_indices_from_rate(
    *,
    n_items: int,
    rate: float,
    ordering: Sequence[int] | None,
    seed: int,
) -> Tuple[int, ...]:
    total = int(max(0, n_items))
    if total <= 0:
        return tuple()
    sample_rate = _clamp01(rate)
    if sample_rate <= 0.0:
        return tuple()
    resolved_ordering = (
        tuple(int(index) for index in ordering)
        if ordering is not None
        else _deterministic_sample_ordering(n_items=total, seed=int(seed))
    )
    sampled = _deterministic_sample_indices_from_ordering(
        ordering=resolved_ordering,
        rate=float(sample_rate),
        n_items=total,
    )
    if sampled is None:
        return tuple(range(total))
    return tuple(int(index) for index in sampled)


def _explicit_local_indices_from_rate(
    *,
    n_items: int,
    rate: float,
    ordering: Sequence[int] | None,
    seed: int,
) -> Tuple[int, ...]:
    total = int(max(0, n_items))
    if total <= 0:
        return tuple()
    sample_rate = _clamp01(rate)
    if sample_rate <= 0.0:
        return tuple()
    resolved_ordering = (
        tuple(int(index) for index in ordering)
        if ordering is not None
        else _deterministic_sample_ordering(n_items=total, seed=int(seed))
    )
    normalized = [int(index) for index in list(resolved_ordering)[:total]]
    if len(normalized) < total:
        raise ValueError(
            f"ordering shorter than n_items ({len(normalized)} < {total})"
        )
    if sample_rate >= 1.0:
        return tuple(range(total))
    desired_count = float(sample_rate) * float(total)
    base_count = int(math.floor(desired_count + 1e-12))
    residual = float(desired_count - float(base_count))
    sample_count = int(max(0, min(total, base_count)))
    if sample_count < total and residual > 1e-12:
        import random as _random

        if _random.Random(int(seed)).random() < residual:
            sample_count += 1
    if sample_count <= 0:
        return tuple()
    if sample_count >= total:
        return tuple(range(total))
    return tuple(sorted(int(index) for index in normalized[:sample_count]))


def _tree_local_diagnostic_weight(
    *,
    weighting_mode: str,
    span_mass: float,
    depth: int,
    propensity: float,
    gamma: float,
) -> float:
    mode = str(weighting_mode or "fixed_k_hajek").strip().lower() or "fixed_k_hajek"
    clipped_propensity = max(1e-12, float(propensity))
    if mode == "subset_mean":
        return 1.0
    if mode == "fixed_k_hajek":
        return 1.0 / clipped_propensity
    if mode == "span_mass_ipw_sum":
        return float(span_mass) * (float(gamma) ** int(max(0, depth))) / clipped_propensity
    raise ValueError(f"unsupported tree_local_weighting_mode={weighting_mode!r}")


def _build_tree_rate_driven_supervision_manifest(
    *,
    docs: Sequence[_FNOCountDoc],
    config: OPSCountConfig,
    leaf_sample_ordering_by_doc: Mapping[int, Sequence[int]] | None = None,
    internal_sample_ordering_by_doc: Mapping[int, Sequence[int]] | None = None,
) -> BudgetedTrainSupervisionManifest | None:
    train_docs = tuple(docs)
    if not train_docs:
        return None

    doc_mode = str(getattr(config, "doc_consumption_mode", "root_only") or "root_only")
    package_semantics = str(getattr(config, "package_semantics", "") or "")
    is_superset_package = bool(str(package_semantics) == "superset")
    local_split_mode = str(getattr(config, "local_split_mode", "balanced") or "balanced")
    allocation_policy = str(
        getattr(config, "local_allocation_policy", "breadth_first")
        or "breadth_first"
    )
    local_estimand_mode = str(
        getattr(config, "tree_local_weighting_mode", "fixed_k_hajek")
    )
    leaf_rate = _clamp01(getattr(config, "leaf_label_rate", 0.0))
    internal_rate = (
        _clamp01(getattr(config, "internal_label_rate", 0.0))
        if str(getattr(config, "internal_supervision_kind", "none")).strip().lower()
        != "none"
        else 0.0
    )
    configured_local_rates_active = bool(leaf_rate > 1e-12 or internal_rate > 1e-12)
    gamma = float(getattr(config, "depth_discount_gamma", 1.0))
    seed = int(getattr(config, "seed", 0))
    doc_count = int(len(train_docs))

    local_rows: List[Dict[str, Any]] = []
    local_weights: List[float] = []
    leaf_calls_total = 0
    internal_calls_total = 0
    leaf_mass_total = 0.0
    internal_mass_total = 0.0
    leaf_propensity_total = 0.0
    internal_propensity_total = 0.0

    for doc_idx, doc in enumerate(train_docs):
        n_leaves = int(len(doc.leaf_token_ids))
        n_internal = int(
            min(
                _internal_supervision_item_count(
                    n_leaves=n_leaves,
                    max_internal_depth=int(getattr(config, "max_internal_depth", 0)),
                ),
                len(doc.merge_token_lengths),
            )
        )
        leaf_indices = _explicit_local_indices_from_rate(
            n_items=n_leaves,
            rate=float(leaf_rate),
            ordering=(
                None
                if leaf_sample_ordering_by_doc is None
                else leaf_sample_ordering_by_doc.get(int(doc_idx))
            ),
            seed=int(seed) + 81_000 + int(doc_idx),
        )
        internal_indices = _explicit_local_indices_from_rate(
            n_items=n_internal,
            rate=float(internal_rate),
            ordering=(
                None
                if internal_sample_ordering_by_doc is None
                else internal_sample_ordering_by_doc.get(int(doc_idx))
            ),
            seed=int(seed) + 91_000 + int(doc_idx),
        )
        doc_tokens = int(max(1, int(doc.n_tokens)))
        leaf_propensity = _nominal_rate_propensity(
            n_items=n_leaves,
            rate=float(leaf_rate),
        )
        internal_propensity = _nominal_rate_propensity(
            n_items=n_internal,
            rate=float(internal_rate),
        )
        leaf_mass = float(
            sum(
                float(doc.leaf_token_lengths[int(index)]) / float(doc_tokens)
                for index in leaf_indices
                if 0 <= int(index) < len(doc.leaf_token_lengths)
            )
        )
        internal_mass = float(
            sum(
                float(doc.merge_token_lengths[int(index)]) / float(doc_tokens)
                for index in internal_indices
                if 0 <= int(index) < len(doc.merge_token_lengths)
            )
        )
        layout = FNOCountSketch._balanced_tree_layout(int(n_leaves))
        depth_by_global_idx = dict(layout.get("depth_by_global_idx") or {})
        for index in leaf_indices:
            if not (0 <= int(index) < len(doc.leaf_token_lengths)):
                continue
            span_mass = float(doc.leaf_token_lengths[int(index)]) / float(doc_tokens)
            local_weights.append(
                _tree_local_diagnostic_weight(
                    weighting_mode=local_estimand_mode,
                    span_mass=float(span_mass),
                    depth=int(depth_by_global_idx.get(int(index), 0)),
                    propensity=float(leaf_propensity),
                    gamma=float(gamma),
                )
            )
        for index in internal_indices:
            if not (0 <= int(index) < len(doc.merge_token_lengths)):
                continue
            span_mass = float(doc.merge_token_lengths[int(index)]) / float(doc_tokens)
            local_weights.append(
                _tree_local_diagnostic_weight(
                    weighting_mode=local_estimand_mode,
                    span_mass=float(span_mass),
                    depth=int(depth_by_global_idx.get(int(n_leaves) + int(index), 0)),
                    propensity=float(internal_propensity),
                    gamma=float(gamma),
                )
            )
        leaf_calls_total += int(len(leaf_indices))
        internal_calls_total += int(len(internal_indices))
        leaf_mass_total += float(leaf_mass)
        internal_mass_total += float(internal_mass)
        leaf_propensity_total += float(leaf_propensity)
        internal_propensity_total += float(internal_propensity)
        local_rows.append(
            {
                "doc_index": int(doc_idx),
                "doc_tokens": int(doc_tokens),
                "leaf_indices": tuple(int(index) for index in leaf_indices),
                "internal_indices": tuple(int(index) for index in internal_indices),
                "leaf_mass": float(leaf_mass),
                "internal_mass": float(internal_mass),
                "leaf_propensity": float(leaf_propensity),
                "internal_propensity": float(internal_propensity),
            }
        )

    local_mass_per_doc = float(
        (float(leaf_mass_total) + float(internal_mass_total)) / float(max(1, doc_count))
    )
    configured_mass_target_per_doc = _safe_float(
        getattr(config, "mass_target_per_doc", float("nan")),
        default=float("nan"),
    )
    configured_root_mass_per_doc = _clamp01(
        float(getattr(config, "budget_total_calls_per_doc", 0.0))
        * float(getattr(config, "full_doc_budget_share", 1.0))
    )
    if is_superset_package:
        requested_root_mass_per_doc = float(configured_root_mass_per_doc)
        if math.isfinite(configured_mass_target_per_doc) and not configured_local_rates_active:
            mass_target_per_doc = _clamp01(float(configured_mass_target_per_doc))
        elif configured_local_rates_active:
            mass_target_per_doc = float("nan")
        else:
            mass_target_per_doc = float(configured_root_mass_per_doc)
    elif math.isfinite(configured_mass_target_per_doc):
        mass_target_per_doc = _clamp01(float(configured_mass_target_per_doc))
        requested_root_mass_per_doc = _clamp01(
            float(mass_target_per_doc) - float(local_mass_per_doc)
        )
    else:
        mass_target_per_doc = float(configured_root_mass_per_doc)
        requested_root_mass_per_doc = _clamp01(
            float(mass_target_per_doc) - float(local_mass_per_doc)
        )
    doc_ordering = _deterministic_sample_ordering(
        n_items=int(doc_count),
        seed=int(seed) + 71_000,
    )
    selected_root_docs = _explicit_indices_from_rate(
        n_items=int(doc_count),
        rate=float(requested_root_mass_per_doc),
        ordering=doc_ordering,
        seed=int(seed) + 71_000,
    )
    selected_root_doc_set = {int(index) for index in selected_root_docs}

    doc_plans: List[BudgetedTrainSupervisionDocPlan] = []
    document_mass_total = 0.0
    touched_docs_total = 0
    for row in local_rows:
        doc_index = int(row["doc_index"])
        document_mode = str(doc_mode) if doc_index in selected_root_doc_set else ""
        document_mass = 1.0 if document_mode else 0.0
        raw_call_cost = (
            int(bool(document_mode))
            + int(len(row["leaf_indices"]))
            + int(len(row["internal_indices"]))
        )
        if raw_call_cost > 0:
            touched_docs_total += 1
        document_mass_total += float(document_mass)
        doc_plans.append(
            BudgetedTrainSupervisionDocPlan(
                doc_index=int(doc_index),
                doc_tokens=int(row["doc_tokens"]),
                document_mode=str(document_mode),
                leaf_indices=tuple(int(index) for index in row["leaf_indices"]),
                internal_indices=tuple(int(index) for index in row["internal_indices"]),
                raw_call_cost=int(raw_call_cost),
                document_mass=float(document_mass),
                leaf_mass=float(row["leaf_mass"]),
                internal_mass=float(row["internal_mass"]),
                leaf_propensity=float(row["leaf_propensity"]),
                internal_propensity=float(row["internal_propensity"]),
                effective_full_doc_mass=float(
                    float(document_mass)
                    + float(row["leaf_mass"])
                    + float(row["internal_mass"])
                ),
            )
        )

    budget_total_calls_used = int(
        len(selected_root_doc_set) + int(leaf_calls_total) + int(internal_calls_total)
    )
    effective_full_doc_mass_total = float(
        float(document_mass_total) + float(leaf_mass_total) + float(internal_mass_total)
    )
    realized_root_mass_per_doc = float(
        float(document_mass_total) / float(max(1, doc_count))
    )
    realized_effective_full_doc_mass_per_doc = float(
        float(effective_full_doc_mass_total) / float(max(1, doc_count))
    )
    mass_tolerance = float((1.0 / float(max(1, doc_count))) + 1e-9)
    if (
        configured_local_rates_active
        and not is_superset_package
        and math.isfinite(configured_mass_target_per_doc)
        and realized_effective_full_doc_mass_per_doc + mass_tolerance
        < float(mass_target_per_doc)
    ):
        raise ValueError(
            "manifest supervision under-shot the requested mass target: "
            f"target={float(mass_target_per_doc):.6f}, "
            f"requested_root={float(requested_root_mass_per_doc):.6f}, "
            f"realized_root={float(realized_root_mass_per_doc):.6f}, "
            f"realized_total={float(realized_effective_full_doc_mass_per_doc):.6f}, "
            f"local={float(local_mass_per_doc):.6f}, "
            f"doc_count={int(doc_count)}"
        )
    if (
        is_superset_package
        and realized_root_mass_per_doc + mass_tolerance
        < float(requested_root_mass_per_doc)
    ):
        raise ValueError(
            "manifest supervision under-shot the requested superset root coverage: "
            f"requested_root={float(requested_root_mass_per_doc):.6f}, "
            f"realized_root={float(realized_root_mass_per_doc):.6f}, "
            f"realized_total={float(realized_effective_full_doc_mass_per_doc):.6f}, "
            f"local={float(local_mass_per_doc):.6f}, "
            f"doc_count={int(doc_count)}"
        )
    sum_weights = float(sum(local_weights))
    sum_weights_sq = float(sum(weight * weight for weight in local_weights))
    local_weight_ess = (
        float((sum_weights * sum_weights) / max(1e-12, sum_weights_sq))
        if local_weights
        else 0.0
    )
    return BudgetedTrainSupervisionManifest(
        budget_total_calls=int(budget_total_calls_used),
        budget_total_calls_per_doc=float(
            float(budget_total_calls_used) / float(max(1, doc_count))
        ),
        budget_total_calls_used=int(budget_total_calls_used),
        budget_utilization=1.0 if int(budget_total_calls_used) > 0 else 0.0,
        full_doc_budget_share=float(getattr(config, "full_doc_budget_share", 1.0)),
        full_doc_calls_requested=int(len(selected_root_doc_set)),
        full_doc_calls_total=int(len(selected_root_doc_set)),
        local_calls_requested=int(leaf_calls_total + internal_calls_total),
        local_calls_total=int(leaf_calls_total + internal_calls_total),
        doc_consumption_mode=str(doc_mode),
        local_split_mode=str(local_split_mode),
        local_allocation_policy=str(allocation_policy),
        sampling_scheme=str(BUDGETED_SAMPLING_SCHEME_RANDOM_WITHOUT_REPLACEMENT),
        doc_touch_rate=float(float(touched_docs_total) / float(max(1, doc_count))),
        mean_labels_per_touched_doc=float(
            float(budget_total_calls_used) / float(max(1, touched_docs_total))
        ),
        touched_docs_total=int(touched_docs_total),
        effective_full_doc_mass_total=float(effective_full_doc_mass_total),
        effective_full_doc_mass_per_doc=float(
            float(effective_full_doc_mass_total) / float(max(1, doc_count))
        ),
        document_mass_share=float(
            float(document_mass_total) / float(max(1e-12, effective_full_doc_mass_total))
        ),
        leaf_mass_share=float(
            float(leaf_mass_total) / float(max(1e-12, effective_full_doc_mass_total))
        ),
        internal_mass_share=float(
            float(internal_mass_total) / float(max(1e-12, effective_full_doc_mass_total))
        ),
        document_call_share=float(
            float(len(selected_root_doc_set)) / float(max(1, budget_total_calls_used))
        ),
        leaf_call_share=float(
            float(leaf_calls_total) / float(max(1, budget_total_calls_used))
        ),
        internal_call_share=float(
            float(internal_calls_total) / float(max(1, budget_total_calls_used))
        ),
        actual_doc_tokens_mean=float(
            float(sum(int(doc.n_tokens) for doc in train_docs)) / float(max(1, doc_count))
        ),
        actual_doc_tokens_unique=tuple(
            sorted({int(doc.n_tokens) for doc in train_docs})
        ),
        supervision_source="manifest",
        local_estimand_mode=str(local_estimand_mode),
        package_semantics=str(package_semantics),
        mass_target_per_doc=float(mass_target_per_doc),
        requested_root_mass_per_doc=float(requested_root_mass_per_doc),
        realized_root_mass_per_doc=float(realized_root_mass_per_doc),
        realized_leaf_mass_per_doc=float(
            float(leaf_mass_total) / float(max(1, doc_count))
        ),
        realized_internal_mass_per_doc=float(
            float(internal_mass_total) / float(max(1, doc_count))
        ),
        leaf_propensity_mean=float(
            float(leaf_propensity_total) / float(max(1, doc_count))
        ),
        internal_propensity_mean=float(
            float(internal_propensity_total) / float(max(1, doc_count))
        ),
        local_weight_ess=float(local_weight_ess),
        local_weight_max=float(max(local_weights) if local_weights else 0.0),
        doc_plans=tuple(doc_plans),
    )


def _resolved_tree_supervision_manifest(
    *,
    docs: Sequence[_FNOCountDoc],
    config: OPSCountConfig,
    budget_manifest: BudgetedTrainSupervisionManifest | None,
    leaf_sample_ordering_by_doc: Mapping[int, Sequence[int]] | None = None,
    internal_sample_ordering_by_doc: Mapping[int, Sequence[int]] | None = None,
) -> BudgetedTrainSupervisionManifest | None:
    tree_supervision_source = str(
        getattr(config, "tree_supervision_source", "rate") or "rate"
    ).strip().lower() or "rate"
    if tree_supervision_source != "manifest":
        return budget_manifest
    leaf_rate = _clamp01(getattr(config, "leaf_label_rate", 0.0))
    internal_rate = (
        _clamp01(getattr(config, "internal_label_rate", 0.0))
        if str(getattr(config, "internal_supervision_kind", "none")).strip().lower()
        != "none"
        else 0.0
    )
    configured_local_rates_active = bool(leaf_rate > 1e-12 or internal_rate > 1e-12)
    configured_mass_target_per_doc = _safe_float(
        getattr(config, "mass_target_per_doc", float("nan")),
        default=float("nan"),
    )
    if budget_manifest is not None:
        local_calls_total = int(getattr(budget_manifest, "local_calls_total", 0) or 0)
        realized_effective_full_doc_mass_per_doc = _safe_float(
            getattr(budget_manifest, "effective_full_doc_mass_per_doc", float("nan")),
            default=float("nan"),
        )
        doc_count = max(1, len(tuple(getattr(budget_manifest, "doc_plans", tuple()) or tuple())))
        mass_tolerance = float((1.0 / float(doc_count)) + 1e-9)
        manifest_realizes_local_budget = (
            not configured_local_rates_active
            or local_calls_total > 0
        )
        manifest_meets_mass_target = (
            not math.isfinite(configured_mass_target_per_doc)
            or (
                math.isfinite(realized_effective_full_doc_mass_per_doc)
                and float(realized_effective_full_doc_mass_per_doc) + mass_tolerance
                >= float(configured_mass_target_per_doc)
            )
        )
        if manifest_realizes_local_budget and manifest_meets_mass_target:
            return budget_manifest
    return _build_tree_rate_driven_supervision_manifest(
        docs=docs,
        config=config,
        leaf_sample_ordering_by_doc=leaf_sample_ordering_by_doc,
        internal_sample_ordering_by_doc=internal_sample_ordering_by_doc,
    )


def _tree_supervision_contract_summary(
    *,
    config: OPSCountConfig,
    budget_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    source = str(getattr(config, "tree_supervision_source", "rate") or "rate")
    actual_doc_tokens_unique = tuple(
        int(value)
        for value in list(budget_metadata.get("actual_doc_tokens_unique") or [])
    )
    computed_assumed_doc_tokens = int(
        getattr(config, "computed_assumed_doc_tokens", 0) or 0
    )
    realized_local_mass_per_doc = float(
        _safe_float(budget_metadata.get("realized_leaf_mass_per_doc"), default=0.0)
        + _safe_float(budget_metadata.get("realized_internal_mass_per_doc"), default=0.0)
    )
    realized_effective_full_doc_mass_per_doc = float(
        _safe_float(
            budget_metadata.get("effective_full_doc_mass_per_doc"),
            default=float("nan"),
        )
    )
    requested_root_mass_per_doc = float(
        _safe_float(
            budget_metadata.get("requested_root_mass_per_doc"),
            default=0.0,
        )
    )
    configured_mass_target_per_doc = float(
        _safe_float(getattr(config, "mass_target_per_doc", float("nan")), default=float("nan"))
    )
    manifest_mass_target_per_doc = float(
        _safe_float(
            budget_metadata.get("mass_target_per_doc"),
            default=float("nan"),
        )
    )
    local_calls_total = int(
        _safe_float(budget_metadata.get("local_calls_total"), default=0.0)
    )
    doc_count = max(1, len(list(budget_metadata.get("doc_plans") or [])))
    mass_tolerance = float((1.0 / float(doc_count)) + 1e-9)
    configured_local_rates_active = bool(
        abs(float(getattr(config, "leaf_label_rate", 0.0) or 0.0)) > 1e-12
        or (
            str(getattr(config, "internal_supervision_kind", "none")).strip().lower()
            != "none"
            and abs(float(getattr(config, "internal_label_rate", 0.0) or 0.0)) > 1e-12
        )
    )
    package_semantics = str(getattr(config, "package_semantics", "") or "")
    geometry_match = bool(
        computed_assumed_doc_tokens <= 0
        or not actual_doc_tokens_unique
        or all(
            int(value) == int(computed_assumed_doc_tokens)
            for value in actual_doc_tokens_unique
        )
    )
    local_manifest_match = bool(
        not configured_local_rates_active
        or realized_local_mass_per_doc <= 1e-12
        or local_calls_total > 0
    )
    target_mass_to_match = (
        configured_mass_target_per_doc
        if math.isfinite(configured_mass_target_per_doc)
        else manifest_mass_target_per_doc
    )
    mass_target_match = bool(
        not configured_local_rates_active
        or not math.isfinite(target_mass_to_match)
        or realized_effective_full_doc_mass_per_doc + mass_tolerance
        >= float(target_mass_to_match)
    )
    passed = bool(geometry_match and local_manifest_match and mass_target_match)
    return {
        "required": bool(str(source).strip().lower() == "manifest"),
        "passed": bool(passed),
        "tree_supervision_source": str(source),
        "computed_assumed_doc_tokens": int(computed_assumed_doc_tokens),
        "actual_doc_tokens_unique": [
            int(value) for value in actual_doc_tokens_unique
        ],
        "realized_local_mass_per_doc": float(realized_local_mass_per_doc),
        "realized_effective_full_doc_mass_per_doc": float(
            realized_effective_full_doc_mass_per_doc
        ),
        "configured_mass_target_per_doc": float(configured_mass_target_per_doc),
        "manifest_mass_target_per_doc": float(manifest_mass_target_per_doc),
        "requested_root_mass_per_doc": float(requested_root_mass_per_doc),
        "local_calls_total": int(local_calls_total),
        "configured_local_rates_active": bool(configured_local_rates_active),
        "package_semantics": str(package_semantics),
        "checks": {
            "geometry_match": bool(geometry_match),
            "local_manifest_match": bool(local_manifest_match),
            "mass_target_match": bool(mass_target_match),
        },
    }


def _metric_value(metrics: Any, key: str) -> float:
    candidates = (key,) + tuple(_METRIC_ALIAS_MAP.get(key, ()))
    if isinstance(metrics, Mapping):
        for candidate in candidates:
            value = _safe_float(metrics.get(candidate, float("nan")))
            if np.isfinite(value):
                return value
        return float("nan")
    for candidate in candidates:
        value = _safe_float(getattr(metrics, candidate, float("nan")))
        if np.isfinite(value):
            return value
    return float("nan")


def _split_count_drift_value(run: Mapping[str, Any], *, split: str) -> float:
    value = _safe_float(run.get(f"{split}_c2_count_drift_r1_mae", float("nan")))
    if np.isfinite(value):
        return float(value)
    legacy = _safe_float(run.get(f"{split}_c2_idempotence_mae", float("nan")))
    if np.isfinite(legacy):
        return float(legacy)
    return 0.0


def _coerce_axis_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, str) and value == "":
        return ""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if numeric.is_integer():
            return int(numeric)
        return numeric
    text = str(value).strip()
    if not text:
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _leaf_count_stats(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    leaf_tokens: int,
) -> Dict[str, float]:
    if not docs:
        return {
            "mean_leaves_per_doc": float("nan"),
            "max_leaves_per_doc": float("nan"),
        }
    counts = np.asarray(
        [
            max(1, int(np.ceil(float(len(doc.token_regimes)) / float(max(1, leaf_tokens)))))
            for doc in docs
        ],
        dtype=np.float64,
    )
    return {
        "mean_leaves_per_doc": float(np.mean(counts)),
        "max_leaves_per_doc": float(np.max(counts)),
    }


def _resolved_tree_leaf_fno_hyperparameters(
    config: OPSCountConfig,
) -> Dict[str, int]:
    return {
        "tree_leaf_fno_width": int(
            config.tree_leaf_fno_width
            if config.tree_leaf_fno_width is not None
            else config.fno_width
        ),
        "tree_leaf_fno_n_modes": int(
            config.tree_leaf_fno_n_modes
            if config.tree_leaf_fno_n_modes is not None
            else config.fno_n_modes
        ),
        "tree_leaf_fno_n_layers": int(
            config.tree_leaf_fno_n_layers
            if config.tree_leaf_fno_n_layers is not None
            else config.fno_n_layers
        ),
        "tree_leaf_fno_pooling": str(
            getattr(config, "tree_leaf_fno_pooling", None) or "mean"
        ).strip().lower() or "mean",
    }


def _derived_split_objective_fields(
    run: Mapping[str, Any],
    *,
    split: str,
) -> Dict[str, float]:
    root = _safe_float(run.get(f"{split}_root_mae", float("nan")))
    if not np.isfinite(root):
        return {
            f"{split}_unweighted_full_law_objective": float("nan"),
            f"{split}_unweighted_active_objective": float("nan"),
        }
    leaf = _safe_float(run.get(f"{split}_leaf_mae", 0.0), default=0.0)
    c2 = _split_count_drift_value(run, split=split)
    merge = _safe_float(run.get(f"{split}_merge_mae", 0.0), default=0.0)
    full = float(root + leaf + c2 + merge)
    c1_active = _safe_float(run.get("local_law_c1_weight", 0.0), default=0.0) > 0.0
    c2_active = _safe_float(run.get("local_law_c2_weight", 0.0), default=0.0) > 0.0
    c3_active = _safe_float(run.get("local_law_c3_weight", 0.0), default=0.0) > 0.0
    active = float(root)
    if bool(run.get("objective_weights_active", False)):
        if c1_active:
            active += leaf
        if c2_active:
            active += c2
        if c3_active:
            active += merge
    return {
        f"{split}_unweighted_full_law_objective": float(full),
        f"{split}_unweighted_active_objective": float(active),
    }


def _derived_report_objective_fields(run: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for split in REPORTED_SPLITS:
        out.update(_derived_split_objective_fields(run, split=split))
    return out


def _backfill_loaded_run_fields(run: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(run)
    fit_diag = dict(normalized.get("fit_diagnostics") or {})
    config = dict(normalized.get("config") or {})
    merged_surface_source = {**config, **normalized}
    if _is_missing_field(normalized.get("comparison_mode")):
        normalized["comparison_mode"] = infer_markov_comparison_mode(
            requested_mode=str(config.get("comparison_mode", "") or ""),
            tree_exact_collapse_mode=str(
                normalized.get(
                    "tree_exact_collapse_mode",
                    config.get("tree_exact_collapse_mode", ""),
                )
                or ""
            ),
        )
    else:
        normalized["comparison_mode"] = normalize_markov_comparison_mode(
            str(normalized.get("comparison_mode", "") or "")
        )
    if _is_missing_field(normalized.get("comparison_surface_snapshot")):
        normalized["comparison_surface_snapshot"] = comparable_surface_snapshot_from_mapping(
            merged_surface_source
        )
    if _is_missing_field(normalized.get("comparison_surface_diff")):
        normalized["comparison_surface_diff"] = dict(
            config.get("comparison_surface_diff") or {}
        )
    budget_defaults = {
        "budget_total_calls": 0,
        "budget_total_calls_per_doc": 0.0,
        "budget_total_calls_used": 0,
        "budget_utilization": float("nan"),
        "full_doc_budget_share": _safe_float(
            config.get("full_doc_budget_share", 1.0),
            default=1.0,
        ),
        "full_doc_calls_total": 0,
        "local_calls_total": 0,
        "doc_consumption_mode": str(config.get("doc_consumption_mode", "")).strip(),
        "local_split_mode": str(config.get("local_split_mode", "")).strip(),
        "local_allocation_policy": str(
            config.get("local_allocation_policy", "")
        ).strip(),
        "full_doc_calls_requested": 0,
        "effective_full_doc_mass_total": 0.0,
        "effective_full_doc_mass_per_doc": 0.0,
        "document_mass_share": 0.0,
        "leaf_mass_share": 0.0,
        "internal_mass_share": 0.0,
        "document_call_share": 0.0,
        "leaf_call_share": 0.0,
        "internal_call_share": 0.0,
        "sampling_scheme": "",
        "doc_touch_rate": 0.0,
        "mean_labels_per_touched_doc": 0.0,
        "touched_docs_total": 0,
        "local_calls_requested": 0,
    }
    budget_manifest = dict(normalized.get("budget_manifest") or {})
    if budget_manifest:
        for key in (
            "budget_total_calls",
            "budget_total_calls_per_doc",
            "budget_total_calls_used",
            "budget_utilization",
            "full_doc_budget_share",
            "full_doc_calls_requested",
            "full_doc_calls_total",
            "local_calls_requested",
            "local_calls_total",
            "doc_consumption_mode",
            "local_split_mode",
            "local_allocation_policy",
            "sampling_scheme",
            "effective_full_doc_mass_total",
            "effective_full_doc_mass_per_doc",
            "document_mass_share",
            "leaf_mass_share",
            "internal_mass_share",
            "document_call_share",
            "leaf_call_share",
            "internal_call_share",
            "doc_touch_rate",
            "mean_labels_per_touched_doc",
            "touched_docs_total",
        ):
            if _is_missing_field(normalized.get(key)) and key in budget_manifest:
                normalized[key] = budget_manifest.get(key)
    if "doc_plans" not in budget_manifest:
        budget_manifest["doc_plans"] = []
    normalized["budget_manifest"] = budget_manifest
    for key, default_value in budget_defaults.items():
        if _is_missing_field(normalized.get(key)):
            normalized[key] = default_value
    resolved_objective = dict(
        normalized.get("resolved_objective")
        or config.get("resolved_objective")
        or {}
    )
    if resolved_objective:
        for key in (
            "parameterization",
            "weighting_scheme",
            "optimization_root_weight",
            "local_law_c1_weight",
            "local_law_c2_weight",
            "local_law_c3_weight",
            "task_objective_weight_source",
            "proxy_schedule_consistency_weight",
            "objective_surface_name",
            "objective_surface_distinct_from",
            "objective_surface_distinct_note",
            "objective_weights_active",
            "paper_to_lean_local_law_mapping",
            "theorem_terms",
            "proxy_terms",
            "formal_notes",
            "theorem_local_law_total_weight",
            "proxy_schedule_term_total_weight",
        ):
            if _is_missing_field(normalized.get(key)) and key in resolved_objective:
                normalized[key] = resolved_objective.get(key)
    if _is_missing_field(normalized.get("weighting_scheme")):
        normalized["weighting_scheme"] = str(
            resolved_objective.get("weighting_scheme", "")
        ).strip()
    if _is_missing_field(normalized.get("proxy_schedule_consistency_weight")):
        normalized["proxy_schedule_consistency_weight"] = float(
            _safe_float(
                resolved_objective.get("proxy_schedule_consistency_weight", 0.0),
                default=0.0,
            )
        )
    if "theorem_terms" not in normalized:
        normalized["theorem_terms"] = list(resolved_objective.get("theorem_terms") or [])
    if "proxy_terms" not in normalized:
        normalized["proxy_terms"] = list(resolved_objective.get("proxy_terms") or [])
    if _is_missing_field(normalized.get("objective_surface_name")):
        normalized["objective_surface_name"] = str(
            resolved_objective.get(
                "objective_surface_name",
                MARKOV_FULL_DOC_OBJECTIVE_SURFACE,
            )
        )
    if _is_missing_field(normalized.get("objective_variant")):
        normalized["objective_variant"] = str(
            config.get("doc_sequence_objective", "")
        ).strip()
    if _is_missing_field(normalized.get("device_requested")):
        normalized["device_requested"] = "cuda" if bool(config.get("use_cuda", False)) else "cpu"
    if _is_missing_field(normalized.get("device_resolved")):
        normalized["device_resolved"] = str(normalized.get("device_requested", "cpu")).strip()
    if _is_missing_field(normalized.get("objective_surface_distinct_from")):
        normalized["objective_surface_distinct_from"] = list(
            resolved_objective.get("objective_surface_distinct_from")
            or [TREEPO_REGULARIZED_OBJECTIVE_SURFACE]
        )
    if _is_missing_field(normalized.get("objective_surface_distinct_note")):
        normalized["objective_surface_distinct_note"] = str(
            resolved_objective.get(
                "objective_surface_distinct_note",
                (
                    "The Markov full-doc normalized task/local-law surface is not the "
                    "same objective family as the TreePO Regularized Objective note."
                ),
            )
        )
    if _is_missing_field(normalized.get("paper_to_lean_local_law_mapping")):
        normalized["paper_to_lean_local_law_mapping"] = dict(
            resolved_objective.get("paper_to_lean_local_law_mapping")
            or PAPER_TO_LEAN_LOCAL_LAW_MAPPING
        )
    if _is_missing_field(normalized.get("theorem_local_law_total_weight")):
        normalized["theorem_local_law_total_weight"] = float(
            _safe_float(
                resolved_objective.get(
                    "theorem_local_law_total_weight",
                    0.0,
                ),
                default=0.0,
            )
        )
    if _is_missing_field(normalized.get("proxy_schedule_term_total_weight")):
        normalized["proxy_schedule_term_total_weight"] = float(
            _safe_float(
                resolved_objective.get(
                    "proxy_schedule_term_total_weight",
                    resolved_objective.get("proxy_schedule_consistency_weight", 0.0),
                ),
                default=0.0,
            )
        )
    expected_theorem_total = float(
        _safe_float(normalized.get("local_law_c1_weight"), default=0.0)
        + _safe_float(normalized.get("local_law_c2_weight"), default=0.0)
        + _safe_float(normalized.get("local_law_c3_weight"), default=0.0)
    )
    current_theorem_total = _safe_float(
        normalized.get("theorem_local_law_total_weight"),
        default=float("nan"),
    )
    if not np.isfinite(current_theorem_total) or abs(current_theorem_total - expected_theorem_total) > 1e-12:
        normalized["theorem_local_law_total_weight"] = float(expected_theorem_total)
    expected_proxy_total = float(
        _safe_float(normalized.get("proxy_schedule_consistency_weight"), default=0.0)
    )
    current_proxy_total = _safe_float(
        normalized.get("proxy_schedule_term_total_weight"),
        default=float("nan"),
    )
    if not np.isfinite(current_proxy_total) or abs(current_proxy_total - expected_proxy_total) > 1e-12:
        normalized["proxy_schedule_term_total_weight"] = float(expected_proxy_total)
    if _is_missing_field(normalized.get("tree_root_supervision_kind")):
        root_supervision_kind = str(
            config.get("tree_root_supervision_kind", "mse")
        ).strip()
        normalized["tree_root_supervision_kind"] = root_supervision_kind or "mse"
    if _is_missing_field(normalized.get("tree_document_loss_normalization_mode")):
        normalized["tree_document_loss_normalization_mode"] = str(
            config.get("tree_document_loss_normalization_mode", "auto")
        ).strip() or "auto"
    if _is_missing_field(
        normalized.get("effective_tree_document_loss_normalization_mode")
    ):
        normalized["effective_tree_document_loss_normalization_mode"] = str(
            normalized.get("tree_document_loss_normalization_mode", "auto")
        ).strip() or "auto"
    if _is_missing_field(normalized.get("document_supervision_docs_total")):
        normalized["document_supervision_docs_total"] = int(
            _safe_float(
                dict(normalized.get("runtime_efficiency") or {}).get(
                    "document_supervision_docs_total",
                    0,
                ),
                default=0.0,
            )
        )
    if _is_missing_field(normalized.get("root_supervision_docs_total")):
        normalized["root_supervision_docs_total"] = int(
            _safe_float(
                dict(normalized.get("runtime_efficiency") or {}).get(
                    "root_supervision_docs_total",
                    0,
                ),
                default=0.0,
            )
        )
    if _is_missing_field(normalized.get("doc_sequence_supervision_docs_total")):
        normalized["doc_sequence_supervision_docs_total"] = int(
            _safe_float(
                dict(normalized.get("runtime_efficiency") or {}).get(
                    "doc_sequence_supervision_docs_total",
                    0,
                ),
                default=0.0,
            )
        )
    if _is_missing_field(normalized.get("document_supervision_coverage_rate")):
        normalized["document_supervision_coverage_rate"] = float(
            _safe_float(
                dict(normalized.get("runtime_efficiency") or {}).get(
                    "document_supervision_coverage_rate",
                    float("nan"),
                ),
                default=float("nan"),
            )
        )
    if _is_missing_field(normalized.get("document_loss_mean_batch_scale")):
        normalized["document_loss_mean_batch_scale"] = float(
            _safe_float(
                dict(normalized.get("runtime_efficiency") or {}).get(
                    "document_loss_mean_batch_scale",
                    float("nan"),
                ),
                default=float("nan"),
            )
        )
    if _is_missing_field(normalized.get("normalized_root_contribution_final")):
        normalized["normalized_root_contribution_final"] = float(
            _safe_float(
                dict(normalized.get("runtime_efficiency") or {}).get(
                    "normalized_root_contribution_final",
                    float("nan"),
                ),
                default=float("nan"),
            )
        )
    if _is_missing_field(normalized.get("tree_checkpoint_metric")):
        normalized["tree_checkpoint_metric"] = str(
            config.get("tree_checkpoint_metric", "")
        ).strip()
    if _is_missing_field(normalized.get("tree_stage1_checkpoint_metric")):
        normalized["tree_stage1_checkpoint_metric"] = str(
            config.get("tree_stage1_checkpoint_metric", "")
        ).strip()
    if _is_missing_field(normalized.get("tree_stage1_eval_mode")):
        normalized["tree_stage1_eval_mode"] = str(
            config.get("tree_stage1_eval_mode", "")
        ).strip()
    if _is_missing_field(normalized.get("tree_stage1_screen_doc_limit")):
        normalized["tree_stage1_screen_doc_limit"] = int(
            _safe_float(config.get("tree_stage1_screen_doc_limit", 0), default=0.0)
        )
    if _is_missing_field(normalized.get("tree_stage1_final_exact_doc_limit")):
        normalized["tree_stage1_final_exact_doc_limit"] = int(
            _safe_float(
                config.get("tree_stage1_final_exact_doc_limit", 0),
                default=0.0,
            )
        )
    if _is_missing_field(normalized.get("exact_metric_selection_doc_limit")):
        normalized["exact_metric_selection_doc_limit"] = int(
            _safe_float(
                config.get("exact_metric_selection_doc_limit", 0),
                default=0.0,
            )
        )
    if _is_missing_field(normalized.get("exact_metric_selection_interval")):
        normalized["exact_metric_selection_interval"] = int(
            _safe_float(
                config.get("exact_metric_selection_interval", 1),
                default=1.0,
            )
        )
    if _is_missing_field(normalized.get("exact_metric_final_doc_limit")):
        normalized["exact_metric_final_doc_limit"] = int(
            _safe_float(config.get("exact_metric_final_doc_limit", 0), default=0.0)
        )
    if _is_missing_field(normalized.get("tree_exact_eval_max_docs")):
        normalized["tree_exact_eval_max_docs"] = int(
            _safe_float(config.get("tree_exact_eval_max_docs", 0), default=0.0)
        )
    if _is_missing_field(normalized.get("tree_posttrain_train_doc_limit")):
        normalized["tree_posttrain_train_doc_limit"] = int(
            _safe_float(config.get("tree_posttrain_train_doc_limit", 0), default=0.0)
        )
    if _is_missing_field(normalized.get("posttrain_diagnostics_mode")):
        normalized["posttrain_diagnostics_mode"] = str(
            config.get("posttrain_diagnostics_mode", "")
        ).strip()
    if _is_missing_field(normalized.get("tree_batch_pack_mode")):
        normalized["tree_batch_pack_mode"] = str(
            config.get("tree_batch_pack_mode", "")
        ).strip()
    if _is_missing_field(normalized.get("tree_batch_token_budget")):
        normalized["tree_batch_token_budget"] = int(
            _safe_float(config.get("tree_batch_token_budget", 0), default=0.0)
        )
    if _is_missing_field(normalized.get("tree_batch_node_budget")):
        normalized["tree_batch_node_budget"] = int(
            _safe_float(config.get("tree_batch_node_budget", 0), default=0.0)
        )
    if _is_missing_field(normalized.get("tree_batch_autotune")):
        normalized["tree_batch_autotune"] = bool(
            config.get("tree_batch_autotune", False)
        )
    if _is_missing_field(normalized.get("tree_batch_structural_pad_limit")):
        normalized["tree_batch_structural_pad_limit"] = float(
            _safe_float(
                config.get("tree_batch_structural_pad_limit", 0.5),
                default=0.5,
            )
        )
    if _is_missing_field(normalized.get("tree_batch_auto_queue_min_docs")):
        normalized["tree_batch_auto_queue_min_docs"] = int(
            _safe_float(
                config.get("tree_batch_auto_queue_min_docs", 8),
                default=8.0,
            )
        )
    if _is_missing_field(normalized.get("tree_batch_auto_queue_min_fill_ratio")):
        normalized["tree_batch_auto_queue_min_fill_ratio"] = float(
            _safe_float(
                config.get("tree_batch_auto_queue_min_fill_ratio", 0.5),
                default=0.5,
            )
        )
    if _is_missing_field(normalized.get("tree_eval_workers_per_mig")):
        normalized["tree_eval_workers_per_mig"] = int(
            _safe_float(config.get("tree_eval_workers_per_mig", 0), default=0.0)
        )
    if _is_missing_field(normalized.get("tree_stage1_artifact_dir")):
        normalized["tree_stage1_artifact_dir"] = str(
            config.get("tree_stage1_artifact_dir", "")
        ).strip()
    if _is_missing_field(normalized.get("tree_stage1_artifact_root")):
        normalized["tree_stage1_artifact_root"] = str(
            config.get("tree_stage1_artifact_root", "")
        ).strip()
    if _is_missing_field(normalized.get("tree_stage1_resume_if_available")):
        normalized["tree_stage1_resume_if_available"] = bool(
            config.get("tree_stage1_resume_if_available", True)
        )
    if _is_missing_field(normalized.get("prepared_data_root")):
        normalized["prepared_data_root"] = str(
            config.get("prepared_data_root", "")
        ).strip()
    if _is_missing_field(normalized.get("prepared_data_allow_create")):
        normalized["prepared_data_allow_create"] = bool(
            config.get("prepared_data_allow_create", True)
        )
    if _is_missing_field(normalized.get("prepared_data_signature")):
        normalized["prepared_data_signature"] = str(
            config.get("prepared_data_signature", "")
        ).strip()
    if _is_missing_field(normalized.get("diagnostic_detail_mode")):
        normalized["diagnostic_detail_mode"] = str(
            config.get("diagnostic_detail_mode", "summary")
        ).strip()
    if _is_missing_field(normalized.get("raw_diagnostic_artifact_dir")):
        normalized["raw_diagnostic_artifact_dir"] = str(
            config.get("raw_diagnostic_artifact_dir", "")
        ).strip()
    if _is_missing_field(normalized.get("tree_stage1_root_weight")):
        normalized["tree_stage1_root_weight"] = float(
            _safe_float(config.get("tree_stage1_root_weight", 0.0), default=0.0)
        )
    if _is_missing_field(normalized.get("tree_training_schedule")):
        normalized["tree_training_schedule"] = str(
            config.get("tree_training_schedule", "")
        ).strip()
    if _is_missing_field(normalized.get("tree_stage1_epochs")):
        normalized["tree_stage1_epochs"] = int(
            _safe_float(config.get("tree_stage1_epochs", 0), default=0.0)
        )
    if _is_missing_field(normalized.get("tree_stage2_epochs")):
        normalized["tree_stage2_epochs"] = int(
            _safe_float(config.get("tree_stage2_epochs", 0), default=0.0)
        )
    if _is_missing_field(normalized.get("tree_task_head_mode")):
        normalized["tree_task_head_mode"] = str(
            config.get("tree_task_head_mode", "")
        ).strip()
    if _is_missing_field(normalized.get("tree_theorem_surface_mode")):
        normalized["tree_theorem_surface_mode"] = str(
            config.get("tree_theorem_surface_mode", "")
        ).strip()
    if _is_missing_field(normalized.get("tree_theorem_count_head_mode")):
        normalized["tree_theorem_count_head_mode"] = str(
            config.get("tree_theorem_count_head_mode", "")
        ).strip()
    if _is_missing_field(normalized.get("tree_theorem_feature_dim")):
        normalized["tree_theorem_feature_dim"] = int(
            _safe_float(config.get("tree_theorem_feature_dim", 0), default=0.0)
        )
    if _is_missing_field(normalized.get("tree_theorem_feature_hidden_dim")):
        normalized["tree_theorem_feature_hidden_dim"] = int(
            _safe_float(config.get("tree_theorem_feature_hidden_dim", 0), default=0.0)
        )
    if _is_missing_field(normalized.get("tree_merge_hidden_dim")):
        normalized["tree_merge_hidden_dim"] = int(
            _safe_float(config.get("tree_merge_hidden_dim", 0), default=0.0)
        )
    if _is_missing_field(normalized.get("tree_phi_compose_weight")):
        normalized["tree_phi_compose_weight"] = float(
            _safe_float(config.get("tree_phi_compose_weight", 0.0), default=0.0)
        )
    if _is_missing_field(normalized.get("tree_phi_contrastive_weight")):
        normalized["tree_phi_contrastive_weight"] = float(
            _safe_float(config.get("tree_phi_contrastive_weight", 0.0), default=0.0)
        )
    if _is_missing_field(normalized.get("tree_phi_alignment_loss")):
        normalized["tree_phi_alignment_loss"] = str(
            config.get("tree_phi_alignment_loss", "")
        ).strip()
    if _is_missing_field(normalized.get("tree_summary_spec_root_mode")):
        normalized["tree_summary_spec_root_mode"] = str(
            config.get("tree_summary_spec_root_mode", "")
        ).strip()
    if _is_missing_field(normalized.get("tree_join_bit_weight")):
        normalized["tree_join_bit_weight"] = float(
            _safe_float(config.get("tree_join_bit_weight", 0.0), default=0.0)
        )
    if _is_missing_field(normalized.get("tree_aux_doc_sequence_fraction")):
        aux_fraction = _safe_float(
            normalized.get(
                "tree_aux_doc_sequence_fraction",
                config.get("doc_sequence_train_fraction", 0.0),
            ),
            default=0.0,
        )
        normalized["tree_aux_doc_sequence_fraction"] = float(aux_fraction)
    if _is_missing_field(normalized.get("aligned_sketch_surface")):
        normalized["aligned_sketch_surface"] = str(
            config.get("aligned_sketch_surface", "")
        ).strip()
    if _is_missing_field(normalized.get("internal_supervision_kind")):
        normalized["internal_supervision_kind"] = str(
            config.get("internal_supervision_kind", "none")
        ).strip() or "none"
    if _is_missing_field(normalized.get("internal_label_rate")):
        normalized["internal_label_rate"] = float(
            _safe_float(config.get("internal_label_rate", 0.0), default=0.0)
        )
    if _is_missing_field(normalized.get("leaf_exact_supervision")):
        normalized["leaf_exact_supervision"] = bool(
            config.get("leaf_exact_supervision", False)
        )
    if _is_missing_field(normalized.get("leaf_supervision_kind")):
        normalized["leaf_supervision_kind"] = str(
            config.get("leaf_supervision_kind", "")
        ).strip()
    if _is_missing_field(normalized.get("summary_spec_name")):
        normalized["summary_spec_name"] = str(
            config.get("summary_spec_name", "")
        ).strip()
    if _is_missing_field(normalized.get("slot_count")):
        normalized["slot_count"] = int(
            _safe_float(config.get("slot_count", 0), default=0.0)
        )
    for field_name in (
        "tree_theorem_count_dim",
        "tree_theorem_first_dim",
        "tree_theorem_last_dim",
    ):
        if _is_missing_field(normalized.get(field_name)):
            normalized[field_name] = int(
                _safe_float(config.get(field_name, 0), default=0.0)
            )
    if _is_missing_field(normalized.get("leaf_label_rate")):
        normalized["leaf_label_rate"] = float(
            _safe_float(config.get("leaf_label_rate", 1.0), default=1.0)
        )
    resolved_tree_fno = {
        "tree_leaf_fno_width": config.get("tree_leaf_fno_width", config.get("fno_width")),
        "tree_leaf_fno_n_modes": config.get("tree_leaf_fno_n_modes", config.get("fno_n_modes")),
        "tree_leaf_fno_n_layers": config.get("tree_leaf_fno_n_layers", config.get("fno_n_layers")),
    }
    for field_name, raw_value in resolved_tree_fno.items():
        if _is_missing_field(normalized.get(field_name)) and raw_value not in {"", None}:
            normalized[field_name] = int(raw_value)
    if _is_missing_field(normalized.get("tree_leaf_fno_pooling")):
        raw_pool = config.get("tree_leaf_fno_pooling")
        if raw_pool not in {"", None}:
            normalized["tree_leaf_fno_pooling"] = str(raw_pool).strip().lower() or "mean"
    if _is_missing_field(normalized.get("fixed_leaf_tokens")) and "fixed_leaf_tokens" in config:
        normalized["fixed_leaf_tokens"] = int(config["fixed_leaf_tokens"])
    for split in REPORTED_SPLITS:
        metrics = normalized.get(f"{split}_metrics")
        for metric_base in SPLIT_METRIC_BASES:
            field_name = f"{split}_{metric_base}"
            if _is_missing_field(normalized.get(field_name)):
                value = _metric_value(metrics, metric_base)
                if np.isfinite(value):
                    normalized[field_name] = float(value)
        exact_field = f"{split}_exact_match_rate"
        if _is_missing_field(normalized.get(exact_field)):
            exact_value = _safe_float(
                fit_diag.get(exact_field, float("nan"))
            )
            if np.isfinite(exact_value):
                normalized[exact_field] = float(exact_value)
    for field_name, metric_key in (
        ("test_root_mae", "root_mae"),
        ("test_leaf_mae", "leaf_mae"),
        ("test_c2_count_drift_r1_mae", "c2_count_drift_r1_mae"),
        ("test_c2_count_drift_r2_mae", "c2_count_drift_r2_mae"),
        ("test_c2_count_drift_r4_mae", "c2_count_drift_r4_mae"),
        ("test_c2_root_count_drift_r1_mae", "c2_root_count_drift_r1_mae"),
        ("test_c2_root_count_drift_r2_mae", "c2_root_count_drift_r2_mae"),
        ("test_c2_root_count_drift_r4_mae", "c2_root_count_drift_r4_mae"),
        ("test_c2_idempotence_mae", "c2_idempotence_mae"),
        ("test_c2_state_replay_mse", "c2_state_replay_mse"),
        ("test_merge_mae", "merge_mae"),
        ("test_schedule_spread_mean", "schedule_spread_mean"),
    ):
        if _is_missing_field(normalized.get(field_name)):
            value = _metric_value(normalized.get("test_metrics"), metric_key)
            if np.isfinite(value):
                normalized[field_name] = float(value)
    if _is_missing_field(normalized.get("test_exact_match_rate")):
        exact_value = _safe_float(fit_diag.get("test_exact_match_rate", float("nan")))
        if np.isfinite(exact_value):
            normalized["test_exact_match_rate"] = float(exact_value)
    exact_sketch_diagnostics = dict(normalized.get("exact_sketch_diagnostics") or {})
    if exact_sketch_diagnostics:
        failure_attribution = dict(
            exact_sketch_diagnostics.get("failure_attribution") or {}
        )
        test_direct_metrics = dict(
            (exact_sketch_diagnostics.get("direct_selection_metrics") or {}).get(
                "test",
                {},
            )
            or {}
        )
        if _is_missing_field(normalized.get("exact_sketch_failure_bucket")):
            normalized["exact_sketch_failure_bucket"] = str(
                failure_attribution.get("bucket", "")
            )
        if _is_missing_field(normalized.get("exact_sketch_leaf_gap_score")):
            normalized["exact_sketch_leaf_gap_score"] = float(
                _safe_float(
                    failure_attribution.get("leaf_gap_score", float("nan")),
                    default=float("nan"),
                )
            )
        if _is_missing_field(normalized.get("exact_sketch_merge_gap_score")):
            normalized["exact_sketch_merge_gap_score"] = float(
                _safe_float(
                    failure_attribution.get("merge_gap_score", float("nan")),
                    default=float("nan"),
                )
            )
        if _is_missing_field(normalized.get("exact_sketch_phi_not_sufficient_score")):
            normalized["exact_sketch_phi_not_sufficient_score"] = float(
                _safe_float(
                    failure_attribution.get("phi_not_sufficient_score", float("nan")),
                    default=float("nan"),
                )
            )
        if _is_missing_field(normalized.get("exact_sketch_phi_not_compositional_score")):
            normalized["exact_sketch_phi_not_compositional_score"] = float(
                _safe_float(
                    failure_attribution.get(
                        "phi_not_compositional_score",
                        float("nan"),
                    ),
                    default=float("nan"),
                )
            )
        if _is_missing_field(normalized.get("exact_sketch_theorem_count_decode_gap_score")):
            normalized["exact_sketch_theorem_count_decode_gap_score"] = float(
                _safe_float(
                    failure_attribution.get(
                        "theorem_count_decode_gap_score",
                        float("nan"),
                    ),
                    default=float("nan"),
                )
            )
        if _is_missing_field(normalized.get("exact_sketch_markov_sufficiency_gap_score")):
            normalized["exact_sketch_markov_sufficiency_gap_score"] = float(
                _safe_float(
                    failure_attribution.get(
                        "markov_sufficiency_gap_score",
                        failure_attribution.get(
                            "theorem_count_decode_gap_score",
                            float("nan"),
                        ),
                    ),
                    default=float("nan"),
                )
            )
        if _is_missing_field(normalized.get("exact_sketch_subtree_label_value_gap_score")):
            normalized["exact_sketch_subtree_label_value_gap_score"] = float(
                _safe_float(
                    failure_attribution.get(
                        "subtree_label_value_gap_score",
                        float("nan"),
                    ),
                    default=float("nan"),
                )
            )
        if _is_missing_field(normalized.get("exact_sketch_internal_label_value_gap_score")):
            normalized["exact_sketch_internal_label_value_gap_score"] = float(
                _safe_float(
                    failure_attribution.get(
                        "internal_label_value_gap_score",
                        float("nan"),
                    ),
                    default=float("nan"),
                )
            )
        if _is_missing_field(normalized.get("exact_sketch_readout_gap_score")):
            normalized["exact_sketch_readout_gap_score"] = float(
                _safe_float(
                    failure_attribution.get("readout_gap_score", float("nan")),
                    default=float("nan"),
                )
            )
        for field_name, metric_key in (
            ("root_direct_count_mae", "root_direct_count_mae"),
            ("test_root_direct_count_mae", "root_direct_count_mae"),
            ("exact_projected_root_mae", "exact_projected_root_mae"),
            ("test_exact_projected_root_mae", "exact_projected_root_mae"),
            ("certified_projected_root_mae", "certified_projected_root_mae"),
            ("test_certified_projected_root_mae", "certified_projected_root_mae"),
            (
                "root_mae_predicted_counts_predicted_endpoints",
                "root_mae_predicted_counts_predicted_endpoints",
            ),
            (
                "test_root_mae_predicted_counts_predicted_endpoints",
                "root_mae_predicted_counts_predicted_endpoints",
            ),
            (
                "root_mae_oracle_counts_predicted_endpoints",
                "root_mae_oracle_counts_predicted_endpoints",
            ),
            (
                "test_root_mae_oracle_counts_predicted_endpoints",
                "root_mae_oracle_counts_predicted_endpoints",
            ),
            (
                "root_mae_predicted_counts_oracle_endpoints",
                "root_mae_predicted_counts_oracle_endpoints",
            ),
            (
                "test_root_mae_predicted_counts_oracle_endpoints",
                "root_mae_predicted_counts_oracle_endpoints",
            ),
            ("learned_merger_gap", "learned_merger_gap"),
            ("test_learned_merger_gap", "learned_merger_gap"),
            (
                "leaf_direct_exact_summary_match_rate",
                "leaf_direct_exact_match",
            ),
            (
                "test_leaf_direct_exact_summary_match_rate",
                "leaf_direct_exact_match",
            ),
            (
                "merge_direct_exact_summary_match_rate",
                "merge_direct_exact_match",
            ),
            (
                "test_merge_direct_exact_summary_match_rate",
                "merge_direct_exact_match",
            ),
            ("merge_join_bit_accuracy", "merge_join_bit_accuracy"),
            ("test_merge_join_bit_accuracy", "merge_join_bit_accuracy"),
            ("leaf_count_head_entropy_mean", "leaf_count_head_entropy_mean"),
            ("merge_count_head_entropy_mean", "merge_count_head_entropy_mean"),
            ("leaf_count_head_margin_mean", "leaf_count_head_margin_mean"),
            ("merge_count_head_margin_mean", "merge_count_head_margin_mean"),
            ("phi_merge_alignment", "phi_merge_alignment"),
            ("phi_within_class_variance", "phi_within_class_variance"),
            ("phi_between_class_margin", "phi_between_class_margin"),
            ("phi_direct_probe_leaf_gap", "phi_direct_probe_leaf_gap"),
            ("phi_direct_probe_merge_gap", "phi_direct_probe_merge_gap"),
            ("leaf_first_accuracy", "leaf_first_accuracy"),
            ("leaf_last_accuracy", "leaf_last_accuracy"),
            ("merge_first_accuracy", "merge_first_accuracy"),
            ("merge_last_accuracy", "merge_last_accuracy"),
        ):
            if _is_missing_field(normalized.get(field_name)):
                normalized[field_name] = float(
                    _safe_float(
                        test_direct_metrics.get(metric_key, float("nan")),
                        default=float("nan"),
                    )
                )
        if _is_missing_field(normalized.get("leaf_count_off_by_k_histogram")):
            normalized["leaf_count_off_by_k_histogram"] = dict(
                test_direct_metrics.get("leaf_count_off_by_k_histogram", {}) or {}
            )
        if _is_missing_field(normalized.get("merge_exact_summary_match_rate_by_depth")):
            normalized["merge_exact_summary_match_rate_by_depth"] = dict(
                test_direct_metrics.get("merge_exact_summary_match_rate_by_depth", {}) or {}
            )
    for key in (
        "study_name",
        "study_axis",
        "locked_tree_neural_config_label",
        "selection_metric",
    ):
        normalized[key] = str(normalized.get(key, "")).strip()
    normalized["axis_value"] = _coerce_axis_value(normalized.get("axis_value", ""))
    if not _is_missing_field(normalized.get("elapsed_s")):
        normalized["elapsed_s"] = float(_safe_float(normalized.get("elapsed_s", 0.0), default=0.0))
    if not _is_missing_field(normalized.get("elapsed_s_job_total")):
        normalized["elapsed_s_job_total"] = float(
            _safe_float(normalized.get("elapsed_s_job_total", 0.0), default=0.0)
        )
    normalized.update(_derived_report_objective_fields(normalized))
    return normalized


@dataclass(frozen=True)
class FullDocDiagnosticBenchmarkSpec:
    name: str
    description: str
    observed_token_profile: str
    canonical_bundle_path: str = ""
    expanded_bundle_path: str = ""
    canonical_train_docs_capacity: int = 0
    expanded_train_docs_capacity: int = 0
    degenerate: bool = False
    cell_id: str = ""
    grid_name: str = ""
    regime_count: int = 0
    segment_density_band: str = ""
    segment_min: int = 0
    segment_max: int = 0
    hazard_switch_prob: float = float("nan")
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    default_train_doc_counts: tuple[int, ...] = tuple()
    official_state_dim: int = 256
    official_hidden_dim: int = 1024
    official_epochs: int = 128
    official_batch_size: int = 64
    official_lr: float = 3e-4
    official_weight_decay: float = 0.0


def _normalized_selected_grid_cell_ids(
    normalized_grid: str,
    grid_cell_ids: Sequence[str],
) -> set[str]:
    selected_ids = {str(value).strip() for value in grid_cell_ids if str(value).strip()}
    if str(normalized_grid).startswith("structural_core_v2"):
        return {_canonicalize_structural_v2_cell_id(value) for value in selected_ids}
    return selected_ids


def resolve_full_doc_diagnostic_benchmark(
    benchmark_name: str = "recoverable_v4",
) -> FullDocDiagnosticBenchmarkSpec:
    key = str(benchmark_name or "recoverable_v4").strip().lower() or "recoverable_v4"
    for grid_name in VALID_HARDNESS_GRIDS:
        structural_grid_prefixes = (
            f"recoverable_{grid_name}__",
            f"{grid_name}::",
        )
        for prefix in structural_grid_prefixes:
            if not key.startswith(prefix):
                continue
            cell_id = str(key[len(prefix) :]).strip()
            if str(grid_name).startswith("structural_core_v2"):
                cell_id = _canonicalize_structural_v2_cell_id(cell_id)
            if not cell_id:
                break
            structural_cells = resolve_full_doc_diagnostic_grid(grid_name)
            benchmark = next(
                (
                    candidate
                    for candidate in structural_cells
                    if str(candidate.cell_id or "").strip().lower() == cell_id
                ),
                None,
            )
            if benchmark is None:
                valid_cells = ", ".join(
                    sorted(
                        str(candidate.cell_id or "").strip()
                        for candidate in structural_cells
                        if str(candidate.cell_id or "").strip()
                    )
                )
                raise ValueError(
                    "unknown full-doc structural benchmark cell: "
                    f"{benchmark_name!r}; expected one of {valid_cells}"
                )
            return benchmark
    if key in {"recoverable", "recoverable_v4"}:
        return FullDocDiagnosticBenchmarkSpec(
            name="recoverable_v4",
            description=(
                "Canonical nondegenerate recoverable full-document benchmark with "
                "root-count support {3,4,5}."
            ),
            observed_token_profile="recoverable",
            canonical_bundle_path=str(CANONICAL_DIAGNOSTIC_BUNDLES["recoverable_v4"]),
            expanded_bundle_path=str(CANONICAL_DIAGNOSTIC_BUNDLES["recoverable_10x_v1"]),
            canonical_train_docs_capacity=1024,
            expanded_train_docs_capacity=10240,
            degenerate=False,
            cell_id="recoverable_v4",
        )
    if key == "recoverable_v5":
        return FullDocDiagnosticBenchmarkSpec(
            name="recoverable_v5",
            description=(
                "Sticky recoverable full-document benchmark with disjoint token palettes, "
                "4 hidden regimes, stochastic regime switches under a simple "
                "stay/switch hazard process, and about 5 expected regime changes "
                "per 128-token document."
            ),
            observed_token_profile="recoverable",
            canonical_bundle_path=str(CANONICAL_DIAGNOSTIC_BUNDLES["recoverable_v5"]),
            expanded_bundle_path=str(CANONICAL_DIAGNOSTIC_BUNDLES["recoverable_10x_v2"]),
            canonical_train_docs_capacity=1024,
            expanded_train_docs_capacity=10240,
            degenerate=False,
            cell_id="recoverable_v5",
            hazard_switch_prob=float(
                _sticky_markov_switch_probability(
                    doc_tokens=128,
                    expected_boundaries=5.0,
                )
            ),
            config_overrides=_sticky_recoverable_config_overrides(doc_tokens=128),
        )
    if key in {"recoverable_t128", "recoverable_v4_t128"}:
        return FullDocDiagnosticBenchmarkSpec(
            name="recoverable_v4_t128",
            description=(
                "Canonical nondegenerate recoverable full-document benchmark with "
                "128-token geometry and root-count support {3,4,5}."
            ),
            observed_token_profile="recoverable",
            canonical_bundle_path=str(CANONICAL_DIAGNOSTIC_BUNDLES["recoverable_v4_t128"]),
            expanded_bundle_path=str(CANONICAL_DIAGNOSTIC_BUNDLES["recoverable_20x_v1_t128"]),
            canonical_train_docs_capacity=1024,
            expanded_train_docs_capacity=20480,
            degenerate=False,
            cell_id="recoverable_v4_t128",
        )
    if key in {"recoverable_v5_t128", "recoverable_sticky_t128"}:
        return FullDocDiagnosticBenchmarkSpec(
            name="recoverable_v5_t128",
            description=(
                "Sticky recoverable full-document benchmark with 128-token geometry, "
                "4 hidden regimes, disjoint token palettes, stochastic "
                "stay/switch regime changes, and about 5 expected regime changes "
                "per document."
            ),
            observed_token_profile="recoverable",
            canonical_bundle_path=str(CANONICAL_DIAGNOSTIC_BUNDLES["recoverable_v5_t128"]),
            expanded_bundle_path=str(CANONICAL_DIAGNOSTIC_BUNDLES["recoverable_20x_v2_t128"]),
            canonical_train_docs_capacity=1024,
            expanded_train_docs_capacity=20480,
            degenerate=False,
            cell_id="recoverable_v5_t128",
            hazard_switch_prob=float(
                _sticky_markov_switch_probability(
                    doc_tokens=128,
                    expected_boundaries=5.0,
                )
            ),
            config_overrides=_sticky_recoverable_config_overrides(doc_tokens=128),
        )
    if key in {"recoverable_v5_t2048", "recoverable_sticky_t2048"}:
        # Composition-stress benchmark: 2048-token docs scale boundary count
        # by sqrt(2048/128) ~ 4x (so ~20 boundaries/doc), keeping per-leaf
        # statistics recognizable at the small-leaf rungs while exposing a
        # much longer merge chain (up to 127 merges at fixed_leaf_tokens=16).
        return FullDocDiagnosticBenchmarkSpec(
            name="recoverable_v5_t2048",
            description=(
                "Sticky recoverable full-document benchmark with 2048-token geometry, "
                "4 hidden regimes, disjoint token palettes, stochastic stay/switch "
                "regime changes, and about 20 expected regime changes per document. "
                "Designed to stress merge composition under a leaf-size ladder."
            ),
            observed_token_profile="recoverable",
            canonical_bundle_path=str(CANONICAL_DIAGNOSTIC_BUNDLES["recoverable_v5_t2048"]),
            canonical_train_docs_capacity=10240,
            degenerate=False,
            cell_id="recoverable_v5_t2048",
            hazard_switch_prob=float(
                _sticky_markov_switch_probability(
                    doc_tokens=2048,
                    expected_boundaries=20.0,
                )
            ),
            config_overrides=_sticky_recoverable_config_overrides_t2048(),
        )
    if key == "recoverable_10x_v1_t128":
        return FullDocDiagnosticBenchmarkSpec(
            name="recoverable_10x_v1_t128",
            description=(
                "Expanded recoverable benchmark with 128-token geometry for large "
                "training-prefix studies."
            ),
            observed_token_profile="recoverable",
            canonical_bundle_path=str(CANONICAL_DIAGNOSTIC_BUNDLES["recoverable_10x_v1_t128"]),
            canonical_train_docs_capacity=10240,
            degenerate=False,
            cell_id="recoverable_10x_v1_t128",
        )
    if key == "recoverable_20x_v1_t128":
        return FullDocDiagnosticBenchmarkSpec(
            name="recoverable_20x_v1_t128",
            description=(
                "Expanded recoverable benchmark with 128-token geometry for xlarge "
                "training-prefix studies."
            ),
            observed_token_profile="recoverable",
            canonical_bundle_path=str(CANONICAL_DIAGNOSTIC_BUNDLES["recoverable_20x_v1_t128"]),
            canonical_train_docs_capacity=20480,
            degenerate=False,
            cell_id="recoverable_20x_v1_t128",
        )
    if key == "demo_v1":
        return FullDocDiagnosticBenchmarkSpec(
            name="demo_v1",
            description=(
                "Degenerate sanity endpoint where every document has the same root count."
            ),
            observed_token_profile="demo_v1",
            canonical_bundle_path=str(CANONICAL_DIAGNOSTIC_BUNDLES["demo_v1"]),
            canonical_train_docs_capacity=256,
            degenerate=True,
            cell_id="demo_v1",
        )
    if key == "smoke":
        return FullDocDiagnosticBenchmarkSpec(
            name="smoke",
            description="Tiny generated smoke benchmark for tests.",
            observed_token_profile="smoke",
            degenerate=False,
            cell_id="smoke",
            official_state_dim=16,
            official_hidden_dim=64,
            official_epochs=4,
            official_batch_size=8,
            official_lr=5e-4,
        )
    if key in {"scorefiber_toy", "scorefiber_toy_v1"}:
        return FullDocDiagnosticBenchmarkSpec(
            name="scorefiber_toy",
            description=(
                "Generic toy benchmark for factorized score-fiber models: "
                "scalar score target plus non-Markov length-bucket fiber labels."
            ),
            observed_token_profile="smoke",
            degenerate=False,
            cell_id="scorefiber_toy",
            official_state_dim=32,
            official_hidden_dim=128,
            official_epochs=8,
            official_batch_size=8,
            official_lr=5e-4,
            config_overrides={
                "theorem_feature_adapter": "scorefiber_length_bucket",
                "min_tokens": 16,
                "max_tokens": 96,
                "min_segments": 2,
                "max_segments": 6,
            },
        )
    raise ValueError(f"unknown full-doc diagnostic benchmark: {benchmark_name!r}")


def resolve_full_doc_diagnostic_grid(
    grid_name: str,
) -> tuple[FullDocDiagnosticBenchmarkSpec, ...]:
    key = str(grid_name or "").strip().lower()
    if key not in VALID_HARDNESS_GRIDS:
        raise ValueError(
            f"unknown full-doc diagnostic hardness grid: {grid_name!r}; "
            f"expected one of {VALID_HARDNESS_GRIDS}"
        )
    doc_tokens = 128 if key.endswith("_t128") else 96
    if key.startswith("structural_core_v2"):
        cells: List[FullDocDiagnosticBenchmarkSpec] = []
        for spec in STICKY_STRUCTURAL_V2_CELL_SPECS:
            switch_prob = _sticky_markov_switch_probability(
                doc_tokens=int(doc_tokens),
                expected_boundaries=float(spec["expected_boundaries"]),
            )
            cell_id = str(spec["cell_id"])
            n_regimes = int(spec["regime_count"])
            min_segments = int(spec["segment_min"])
            max_segments = int(spec["segment_max"])
            cells.append(
                FullDocDiagnosticBenchmarkSpec(
                    name=f"{key}::{cell_id}",
                    cell_id=cell_id,
                    grid_name=str(key),
                    description=(
                        "Sticky structural-hardening cell with disjoint token palettes, "
                        f"{int(n_regimes)} regimes, and a per-token switch probability of "
                        f"{float(switch_prob):.4f} to a uniformly random different regime."
                    ),
                    observed_token_profile="recoverable",
                    canonical_bundle_path=(
                        str(_structural_diagnostic_bundle_path(key, cell_id))
                        if key.endswith("_t128")
                        else ""
                    ),
                    expanded_bundle_path=(
                        str(
                            _structural_expanded_diagnostic_bundle_path(
                                key,
                                cell_id,
                                "20x",
                            )
                        )
                        if key.endswith("_t128")
                        else ""
                    ),
                    canonical_train_docs_capacity=10240 if key.endswith("_t128") else 0,
                    expanded_train_docs_capacity=20480 if key.endswith("_t128") else 0,
                    degenerate=False,
                    regime_count=int(n_regimes),
                    segment_density_band=str(spec["segment_density_band"]),
                    segment_min=int(min_segments),
                    segment_max=int(max_segments),
                    hazard_switch_prob=float(switch_prob),
                    default_train_doc_counts=(1024,),
                    config_overrides=_sticky_structural_config_overrides(
                        doc_tokens=int(doc_tokens),
                        n_regimes=int(n_regimes),
                        min_segments=int(min_segments),
                        max_segments=int(max_segments),
                        hazard_switch_prob=float(switch_prob),
                    ),
                )
            )
        return tuple(cells)
    density_bands = (
        ("low", 4, 6),
        ("mid", 7, 9),
        ("high", 10, 12),
    )
    cells: List[FullDocDiagnosticBenchmarkSpec] = []
    for n_regimes in (4, 8, 12):
        for band, min_segments, max_segments in density_bands:
            cell_id = f"r{int(n_regimes)}_seg{int(min_segments)}to{int(max_segments)}"
            min_distinct = int(min(int(n_regimes), int(min_segments)))
            max_distinct = int(min(int(n_regimes), int(max_segments)))
            cells.append(
                FullDocDiagnosticBenchmarkSpec(
                    name=(
                        f"recoverable_{key}__{cell_id}"
                        if key in {"structural_core_v1", "structural_core_v2"}
                        else f"{key}::{cell_id}"
                    ),
                    cell_id=cell_id,
                    grid_name=str(key),
                    description=(
                        (
                            "Sticky structural-hardening cell with disjoint token palettes, "
                            f"{int(n_regimes)} regimes, and a stay/switch hazard process "
                            f"calibrated to the {int(min_segments)}-{int(max_segments)} "
                            "segment-density band."
                            if key.startswith("structural_core_v2")
                            else "Recoverable structural-hardening cell with disjoint token palettes, "
                            f"{int(n_regimes)} regimes, and {int(min_segments)}-{int(max_segments)} "
                            "segments per document."
                        )
                    ),
                    observed_token_profile="recoverable",
                    canonical_bundle_path=(
                        str(_structural_diagnostic_bundle_path(key, cell_id))
                        if key.endswith("_t128")
                        else ""
                    ),
                    expanded_bundle_path=(
                        str(
                            _structural_expanded_diagnostic_bundle_path(
                                key,
                                cell_id,
                                "20x",
                            )
                        )
                        if key.endswith("_t128")
                        else ""
                    ),
                    canonical_train_docs_capacity=10240 if key.endswith("_t128") else 0,
                    expanded_train_docs_capacity=20480 if key.endswith("_t128") else 0,
                    degenerate=False,
                    regime_count=int(n_regimes),
                    segment_density_band=str(band),
                    segment_min=int(min_segments),
                    segment_max=int(max_segments),
                    default_train_doc_counts=(1024,),
                    config_overrides=(
                        _sticky_structural_config_overrides(
                            doc_tokens=int(doc_tokens),
                            n_regimes=int(n_regimes),
                            min_segments=int(min_segments),
                            max_segments=int(max_segments),
                        )
                        if key.startswith("structural_core_v2")
                        else {
                            "generator_profile": "piecewise_disjoint_palette",
                            "n_regimes": int(n_regimes),
                            "vocab_size": int(4 * int(n_regimes)),
                            "min_tokens": int(doc_tokens),
                            "max_tokens": int(doc_tokens),
                            "min_segments": int(min_segments),
                            "max_segments": int(max_segments),
                            "min_seg_len": 8,
                            "max_seg_len": 24,
                            "train_docs": 1024,
                            "val_docs": 128,
                            "test_docs": 256,
                            "min_distinct_regimes_per_doc": int(min_distinct),
                            "max_distinct_regimes_per_doc": int(max_distinct),
                        }
                    ),
                )
            )
    return tuple(cells)


def _resolved_markov_generator_profile(
    *,
    benchmark: FullDocDiagnosticBenchmarkSpec | None = None,
    config: OPSCountConfig | None = None,
) -> str:
    if config is not None and str(getattr(config, "generator_profile", "")).strip():
        return str(getattr(config, "generator_profile", "")).strip().lower()
    if benchmark is None:
        return ""
    policy = resolve_markov_observed_token_policy(
        profile_name=str(benchmark.observed_token_profile),
    )
    profile = str(getattr(policy, "generator_profile", "")).strip().lower()
    override = str(
        (benchmark.config_overrides or {}).get("generator_profile", "") or ""
    ).strip().lower()
    return override or profile


def _markov_observed_token_recoverability_contract(
    *,
    benchmark: FullDocDiagnosticBenchmarkSpec | None = None,
    config: OPSCountConfig | None = None,
) -> Dict[str, Any]:
    generator_profile = _resolved_markov_generator_profile(
        benchmark=benchmark,
        config=config,
    )
    recoverable = str(generator_profile) in {"piecewise_disjoint_palette", "hazard_topic"}
    return {
        "generator_profile": str(generator_profile),
        "lean_recoverable_in_principle": bool(recoverable),
        "lean_bayes_error_zero": bool(recoverable),
        "theorem_refs": {
            "observed_token_path_ref": LEAN_MARKOV_OBSERVED_TOKEN_RECOVERABILITY_REF,
            "observed_token_exact_sketch_ref": LEAN_MARKOV_OBSERVED_TOKEN_EXACT_SKETCH_REF,
            "zero_bayes_error_ref": LEAN_MARKOV_ZERO_BAYES_ERROR_REF,
            "representation_exact_pass_ref": LEAN_MARKOV_REPRESENTATION_EXACT_PASS_REF,
            "representation_zero_root_count_error_ref": (
                LEAN_MARKOV_REPRESENTATION_ZERO_ROOT_COUNT_ERROR_REF
            ),
            "representation_count_transport_ref": (
                LEAN_MARKOV_REPRESENTATION_COUNT_TRANSPORT_REF
            ),
        },
        "note": (
            "For the clean disjoint-palette Markov family, observed tokens identify "
            "the latent regime block at each position, so the latent path, exact "
            "Markov sketch, and changepoint-count target are recoverable in "
            "principle. Learnability experiments should therefore be interpreted as "
            "representation / optimization tests rather than identifiability tests."
            if recoverable
            else "This benchmark is not covered by the current disjoint-palette "
            "recoverability theorem surface."
        ),
    }


def estimate_tree_worker_runtime_preflight(
    *,
    benchmark_name: str = "recoverable_v4",
    hardness_grid: str = "",
    grid_cell_ids: Sequence[str] = tuple(),
    train_doc_count: int,
    config_overrides: Mapping[str, Any] | None = None,
    use_cuda: bool = True,
    torch_threads: int = 1,
    seed: int = 0,
) -> Dict[str, Any]:
    normalized_grid = str(hardness_grid or "").strip().lower()
    benchmarks: tuple[FullDocDiagnosticBenchmarkSpec, ...]
    if normalized_grid:
        benchmarks = resolve_full_doc_diagnostic_grid(normalized_grid)
        if grid_cell_ids:
            selected_ids = _normalized_selected_grid_cell_ids(
                normalized_grid,
                grid_cell_ids,
            )
            benchmarks = tuple(
                benchmark
                for benchmark in benchmarks
                if str(benchmark.cell_id) in selected_ids
            )
    else:
        benchmarks = (resolve_full_doc_diagnostic_benchmark(str(benchmark_name)),)
    if len(benchmarks) != 1:
        raise ValueError(
            "estimate_tree_worker_runtime_preflight requires exactly one benchmark cell"
        )
    benchmark = benchmarks[0]
    required_train_docs = int(train_doc_count)
    base_bundle, base_source = _materialize_base_bundle(
        benchmark=benchmark,
        required_train_docs=int(required_train_docs),
        output_dir=None,
    )
    bundle, _bundle_source = _bundle_with_fixed_eval_splits(
        base_bundle=base_bundle,
        base_source=base_source,
        train_doc_count=int(required_train_docs),
    )
    config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=int(required_train_docs),
        use_cuda=bool(use_cuda),
        cuda_device=0 if bool(use_cuda) else None,
        torch_threads=int(torch_threads),
        seed=int(seed),
        config_overrides=config_overrides,
    )
    fixed_leaf_tokens = int(getattr(config, "fixed_leaf_tokens", 0) or 0)
    if fixed_leaf_tokens <= 0:
        return {
            "available": False,
            "reason": "fixed_leaf_tokens_required",
            "benchmark": str(benchmark.name),
            "cell_id": str(benchmark.cell_id or ""),
            "train_doc_count": int(required_train_docs),
        }
    runtime_config = _gpu_runtime_config_from_ops_config(
        config,
        device=torch.device("cuda" if bool(use_cuda) else "cpu"),
    )
    train_fno_docs = _prepare_fno_count_docs(
        bundle.train_docs,
        leaf_tokens=int(fixed_leaf_tokens),
    )
    val_fno_docs = _prepare_fno_count_docs(
        bundle.val_docs,
        leaf_tokens=int(fixed_leaf_tokens),
    )
    split_estimates = {
        "train": estimate_tree_gpu_batch_store_bytes(
            docs=train_fno_docs,
            runtime_config=runtime_config,
            split_name="train",
            structural_pad_limit=float(
                getattr(config, "tree_batch_structural_pad_limit", 0.5)
            ),
            auto_queue_min_docs=int(
                getattr(config, "tree_batch_auto_queue_min_docs", 8)
            ),
            pad_id=0,
        ),
        "val": estimate_tree_gpu_batch_store_bytes(
            docs=val_fno_docs,
            runtime_config=runtime_config,
            split_name="val",
            structural_pad_limit=float(
                getattr(config, "tree_batch_structural_pad_limit", 0.5)
            ),
            auto_queue_min_docs=int(
                getattr(config, "tree_batch_auto_queue_min_docs", 8)
            ),
            pad_id=0,
        ),
    }
    total_resident_store_bytes = int(
        sum(
            int(dict(split_estimate).get("resident_store_bytes", 0) or 0)
            for split_estimate in split_estimates.values()
        )
    )
    return {
        "available": True,
        "benchmark": str(benchmark.name),
        "cell_id": str(benchmark.cell_id or ""),
        "hardness_grid": str(benchmark.grid_name or normalized_grid),
        "train_doc_count": int(required_train_docs),
        "fixed_leaf_tokens": int(fixed_leaf_tokens),
        "runtime": {
            "data_mode": str(getattr(config, "gpu_runtime_data_mode", "resident")),
            "bucket_mode": str(
                getattr(config, "gpu_runtime_bucket_mode", "exact_then_bucketed")
            ),
            "preload_splits": [
                str(value)
                for value in getattr(
                    config,
                    "gpu_runtime_preload_splits",
                    ("train", "val", "test"),
                )
            ],
            "preload_targets": bool(
                getattr(config, "gpu_runtime_preload_targets", True)
            ),
        },
        "split_estimates": split_estimates,
        "resident_store_bytes_total": int(total_resident_store_bytes),
    }


def default_train_doc_counts_for_benchmark(
    benchmark: FullDocDiagnosticBenchmarkSpec,
) -> tuple[int, ...]:
    if benchmark.default_train_doc_counts:
        return tuple(int(value) for value in benchmark.default_train_doc_counts)
    policy = resolve_markov_observed_token_policy(
        profile_name=str(benchmark.observed_token_profile),
    )
    base = int(benchmark.config_overrides.get("train_docs", policy.train_docs))
    return tuple(int(multiplier * base) for multiplier in (1, 2, 5, 10))


def _root_count_support(docs: Sequence[ChangepointMarkovDoc]) -> Dict[str, Any]:
    counts: Dict[int, int] = {}
    for doc in docs:
        value = int(round(float(len(doc.true_boundaries))))
        counts[int(value)] = int(counts.get(int(value), 0) + 1)
    values = sorted(counts.keys())
    return {
        "values": values,
        "histogram": {str(int(value)): int(counts[int(value)]) for value in values},
        "n_unique": int(len(values)),
        "is_constant": bool(len(values) <= 1),
    }


def _resolved_markov_target_scale(
    config: OPSCountConfig,
    *,
    observed_targets: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Return a process-level count bound for theorem-facing Markov metrics."""
    if str(getattr(config, "generator_profile", "")).strip().lower() == "hazard_topic":
        if observed_targets is not None:
            observed = np.asarray(list(observed_targets), dtype=np.float64).reshape(-1)
            if observed.size > 0:
                return float(max(1.0, float(np.max(observed))))
    max_segments = int(getattr(config, "max_segments", 0) or 0)
    if max_segments > 1:
        return float(max(1, max_segments - 1))
    if observed_targets is None:
        return 1.0
    observed = np.asarray(list(observed_targets), dtype=np.float64).reshape(-1)
    if observed.size == 0:
        return 1.0
    return float(max(1.0, float(np.max(observed))))


def _distinct_regime_support(docs: Sequence[ChangepointMarkovDoc]) -> Dict[str, Any]:
    counts: Dict[int, int] = {}
    for doc in docs:
        value = int(len(set(int(x) for x in doc.token_regimes)))
        counts[int(value)] = int(counts.get(int(value), 0) + 1)
    values = sorted(counts.keys())
    return {
        "values": values,
        "histogram": {str(int(value)): int(counts[int(value)]) for value in values},
        "n_unique": int(len(values)),
        "is_constant": bool(len(values) <= 1),
    }


def _resolve_device(config: OPSCountConfig) -> tuple[Dict[str, int], torch.device]:
    seeds = _resolve_runtime_seeds(config)
    _set_global_seed(int(seeds["effective_model_seed"]))
    if int(config.torch_threads) > 0:
        torch.set_num_threads(int(config.torch_threads))
    if bool(config.use_cuda) and torch.cuda.is_available():
        if config.cuda_device is not None:
            idx = int(config.cuda_device)
            if idx < 0 or idx >= int(torch.cuda.device_count()):
                raise ValueError(f"cuda_device={idx} out of range")
            torch.cuda.set_device(idx)
            return seeds, torch.device(f"cuda:{idx}")
        return seeds, torch.device("cuda")
    return seeds, torch.device("cpu")


def _base_config_for_benchmark(
    *,
    benchmark: FullDocDiagnosticBenchmarkSpec,
    train_docs: int,
    use_cuda: bool,
    cuda_device: int | None,
    torch_threads: int,
    seed: int,
    config_overrides: Mapping[str, Any] | None = None,
    baseline_families: Sequence[str] | None = None,
) -> OPSCountConfig:
    policy = resolve_markov_observed_token_policy(
        profile_name=str(benchmark.observed_token_profile),
    )
    raw_config_overrides = runtime_config_overrides_from_config_like(config_overrides)
    comparison_mode = infer_markov_comparison_mode(
        requested_mode=str(raw_config_overrides.get("comparison_mode", "") or ""),
        baseline_families=baseline_families,
        tree_exact_collapse_mode=str(
            raw_config_overrides.get("tree_exact_collapse_mode", "") or ""
        ),
    )
    requested_fixed_leaf_tokens_raw = raw_config_overrides.get("fixed_leaf_tokens", None)
    try:
        requested_fixed_leaf_tokens = int(requested_fixed_leaf_tokens_raw or 0)
    except Exception:
        requested_fixed_leaf_tokens = 0
    preserve_requested_leaf_tokens = bool(
        raw_config_overrides.get("preserve_requested_leaf_tokens", False)
        or raw_config_overrides.get(
            "official_fno_preserve_requested_leaf_tokens", False
        )
        or (
            requested_fixed_leaf_tokens_raw not in {"", None}
            and requested_fixed_leaf_tokens > 0
        )
        or (
            comparison_mode in {"comparable", "exact_collapse"}
            and requested_fixed_leaf_tokens_raw not in {"", None}
            and requested_fixed_leaf_tokens > 0
        )
    )
    if preserve_requested_leaf_tokens and requested_fixed_leaf_tokens > 0:
        raw_config_overrides.setdefault("preserve_requested_leaf_tokens", True)
        raw_config_overrides.setdefault(
            "official_fno_preserve_requested_leaf_tokens",
            True,
        )
    try:
        requested_leafgrid_tokens = int(
            raw_config_overrides.get("pipeline_supervision_recovery_leaf_tokens", 0)
            or 0
        )
    except Exception:
        requested_leafgrid_tokens = 0
    leafgrid_active = bool(
        raw_config_overrides.get("pipeline_supervision_recovery_leafgrid_active", False)
    ) or requested_leafgrid_tokens > 0
    if leafgrid_active and requested_leafgrid_tokens > 0:
        if not bool(raw_config_overrides.get("preserve_requested_leaf_tokens", False)):
            raise ValueError(
                "leaf-grid supervision-recovery tasks must set "
                "preserve_requested_leaf_tokens=True"
            )
        if not bool(
            raw_config_overrides.get(
                "official_fno_preserve_requested_leaf_tokens", False
            )
        ):
            raise ValueError(
                "leaf-grid supervision-recovery tasks must set "
                "official_fno_preserve_requested_leaf_tokens=True"
            )
        if (
            requested_fixed_leaf_tokens > 0
            and requested_fixed_leaf_tokens != requested_leafgrid_tokens
        ):
            raise ValueError(
                "leaf-grid supervision-recovery fixed_leaf_tokens drifted before "
                f"benchmark locking: requested={requested_leafgrid_tokens} "
                f"config_fixed_leaf_tokens={requested_fixed_leaf_tokens}"
            )
    observed_token_locked_fields = resolve_markov_benchmark_locked_fields(
        benchmark=benchmark,
        config=raw_config_overrides,
        comparison_mode=comparison_mode,
    )
    benchmark_overrides = dict(benchmark.config_overrides or {})
    if preserve_requested_leaf_tokens:
        benchmark_overrides.pop("fixed_leaf_tokens", None)
    cfg = OPSCountConfig(
        n_regimes=int(observed_token_locked_fields["n_regimes"]),
        vocab_size=int(observed_token_locked_fields["vocab_size"]),
        generator_profile=str(observed_token_locked_fields["generator_profile"]),
        min_tokens=int(observed_token_locked_fields["min_tokens"]),
        max_tokens=int(observed_token_locked_fields["max_tokens"]),
        min_segments=int(observed_token_locked_fields["min_segments"]),
        max_segments=int(observed_token_locked_fields["max_segments"]),
        min_seg_len=int(observed_token_locked_fields["min_seg_len"]),
        max_seg_len=int(observed_token_locked_fields["max_seg_len"]),
        fixed_leaf_tokens=int(observed_token_locked_fields["fixed_leaf_tokens"]),
        train_docs=int(train_docs),
        val_docs=int(policy.val_docs),
        test_docs=int(policy.test_docs),
        model_family="neural",
        feature_mode="token_full",
        state_dim=int(benchmark.official_state_dim),
        hidden_dim=int(benchmark.official_hidden_dim),
        n_epochs=int(benchmark.official_epochs),
        batch_size=int(benchmark.official_batch_size),
        lr=float(benchmark.official_lr),
        weight_decay=float(benchmark.official_weight_decay),
        use_cuda=bool(use_cuda),
        cuda_device=int(cuda_device) if cuda_device is not None else None,
        torch_threads=int(torch_threads),
        doc_sequence_objective="count_ce_only",
        doc_level_ridge_alpha=float(policy.doc_level_ridge_alpha),
        doc_level_ridge_breakdown_orders=tuple(
            int(x) for x in policy.doc_level_ridge_breakdown_orders
        ),
        rf_n_estimators=int(policy.rf_n_estimators),
        rf_max_depth=int(policy.rf_max_depth),
        rf_min_samples_leaf=int(policy.rf_min_samples_leaf),
        leaf_knn_neighbors=int(policy.leaf_knn_neighbors),
        seed=int(seed),
        data_seed=int(policy.seed),
        model_seed=int(seed),
        comparison_mode=str(comparison_mode),
    )
    benchmark_locked_keys = {
        "generator_profile",
        "n_regimes",
        "vocab_size",
        "min_tokens",
        "max_tokens",
        "min_segments",
        "max_segments",
        "min_seg_len",
        "max_seg_len",
        "fixed_leaf_tokens",
        "min_distinct_regimes_per_doc",
        "max_distinct_regimes_per_doc",
    }
    if preserve_requested_leaf_tokens:
        benchmark_locked_keys.discard("fixed_leaf_tokens")
    external_overrides = dict(raw_config_overrides)
    for key in benchmark_locked_keys:
        external_overrides.pop(key, None)
    overrides = {
        **benchmark_overrides,
        **external_overrides,
    }
    if overrides:
        merged = {**asdict(cfg), **overrides}
        merged["train_docs"] = int(train_docs)
        merged["seed"] = int(seed)
        merged["data_seed"] = int(merged.get("data_seed", policy.seed))
        merged["model_seed"] = int(merged.get("model_seed", seed))
        merged["comparison_mode"] = str(comparison_mode)
        if comparison_mode in {"comparable", "exact_collapse"}:
            surface = resolve_markov_comparable_surface(
                benchmark=benchmark,
                config=merged,
                comparison_mode=comparison_mode,
            )
            merged = apply_comparable_surface_to_mapping(
                benchmark=benchmark,
                config=merged,
                surface=surface,
            )
        known_fields = {f.name for f in OPSCountConfig.__dataclass_fields__.values()}
        merged = {k: v for k, v in merged.items() if k in known_fields}
        cfg = OPSCountConfig(**merged)
    elif comparison_mode in {"comparable", "exact_collapse"}:
        surface = resolve_markov_comparable_surface(
            benchmark=benchmark,
            config=cfg,
            comparison_mode=comparison_mode,
        )
        cfg = OPSCountConfig(
            **apply_comparable_surface_to_mapping(
                benchmark=benchmark,
                config=cfg,
                surface=surface,
            )
        )
    if leafgrid_active and requested_leafgrid_tokens > 0:
        resolved_fixed_leaf_tokens = int(getattr(cfg, "fixed_leaf_tokens", 0) or 0)
        if resolved_fixed_leaf_tokens != requested_leafgrid_tokens:
            raise ValueError(
                "leaf-grid supervision-recovery fixed_leaf_tokens drifted after "
                f"benchmark locking: requested={requested_leafgrid_tokens} "
                f"resolved={resolved_fixed_leaf_tokens}"
            )
    return cfg


def _official_fno_locked_config_for_benchmark(
    *,
    benchmark: FullDocDiagnosticBenchmarkSpec,
    config: OPSCountConfig,
) -> OPSCountConfig:
    comparison_mode = infer_markov_comparison_mode(
        requested_mode=str(getattr(config, "comparison_mode", "") or ""),
        baseline_families=("official_fno",),
        tree_exact_collapse_mode=str(
            getattr(config, "tree_exact_collapse_mode", "") or ""
        ),
    )
    if comparison_mode in {"comparable", "exact_collapse"}:
        surface = resolve_markov_comparable_surface(
            benchmark=benchmark,
            config=config,
            comparison_mode=comparison_mode,
        )
        comparable_mapping = apply_comparable_surface_to_mapping(
            benchmark=benchmark,
            config=config,
            surface=surface,
        )
        # Intentional comparator lock: official-FNO comparison rows are root-only
        # parity anchors, so all tree-local supervision semantics are zeroed here.
        locked = {
            "model_family": "neural",
            "feature_mode": "token_full",
            "doc_sequence_objective": "count_ce_only",
            "fixed_leaf_tokens": int(FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS),
            "law_package": "",
            "leaf_weight": 0.0,
            "c2_weight": 0.0,
            "c3_weight": 0.0,
            "local_law_weight": None,
            "task_objective_weight": 1.0,
            "c1_relative_weight": 0.0,
            "c2_relative_weight": 0.0,
            "c3_relative_weight": 0.0,
            "leaf_supervision_kind": "count_only",
            "leaf_label_rate": 0.0,
            "leaf_exact_supervision": False,
            "internal_supervision_kind": "none",
            "internal_label_rate": 0.0,
            "preserve_requested_leaf_tokens": False,
            "official_fno_preserve_requested_leaf_tokens": False,
            "comparison_mode": str(comparison_mode),
        }
        return OPSCountConfig(**{**asdict(config), **comparable_mapping, **locked})
    policy = resolve_markov_observed_token_policy(
        profile_name=str(benchmark.observed_token_profile),
    )
    observed_token_locked_fields = {
        "n_regimes": int(policy.n_regimes),
        "vocab_size": int(policy.vocab_size),
        "generator_profile": str(policy.generator_profile),
        "min_tokens": int(policy.min_tokens),
        "max_tokens": int(policy.max_tokens),
        "min_segments": int(policy.min_segments),
        "max_segments": int(policy.max_segments),
        "min_seg_len": int(policy.min_seg_len),
        "max_seg_len": int(policy.max_seg_len),
        "fixed_leaf_tokens": int(FULL_DOC_OFFICIAL_FNO_FIXED_LEAF_TOKENS),
        "min_distinct_regimes_per_doc": getattr(
            policy, "min_distinct_regimes_per_doc", None
        ),
        "max_distinct_regimes_per_doc": getattr(
            policy, "max_distinct_regimes_per_doc", None
        ),
    }
    benchmark_overrides = dict(benchmark.config_overrides or {})
    benchmark_overrides.pop("fixed_leaf_tokens", None)
    for key in tuple(observed_token_locked_fields):
        if key in benchmark_overrides and benchmark_overrides[key] is not None:
            observed_token_locked_fields[key] = benchmark_overrides[key]
    # Intentional comparator lock: standalone FNO rows are benchmark-locked
    # comparators rather than semantic-preserving tree runs.
    locked = {
        "model_family": "neural",
        "feature_mode": "token_full",
        "n_regimes": int(observed_token_locked_fields["n_regimes"]),
        "vocab_size": int(observed_token_locked_fields["vocab_size"]),
        "generator_profile": str(observed_token_locked_fields["generator_profile"]),
        "min_tokens": int(observed_token_locked_fields["min_tokens"]),
        "max_tokens": int(observed_token_locked_fields["max_tokens"]),
        "min_segments": int(observed_token_locked_fields["min_segments"]),
        "max_segments": int(observed_token_locked_fields["max_segments"]),
        "min_seg_len": int(observed_token_locked_fields["min_seg_len"]),
        "max_seg_len": int(observed_token_locked_fields["max_seg_len"]),
        "min_distinct_regimes_per_doc": observed_token_locked_fields[
            "min_distinct_regimes_per_doc"
        ],
        "max_distinct_regimes_per_doc": observed_token_locked_fields[
            "max_distinct_regimes_per_doc"
        ],
        "fixed_leaf_tokens": int(observed_token_locked_fields["fixed_leaf_tokens"]),
        "doc_sequence_objective": "count_ce_only",
        "seed": int(config.seed),
        "data_seed": int(config.data_seed if config.data_seed is not None else policy.seed),
        "model_seed": int(
            config.model_seed if config.model_seed is not None else config.seed
        ),
        "law_package": "",
        "leaf_weight": 0.0,
        "c2_weight": 0.0,
        "c3_weight": 0.0,
        "local_law_weight": None,
        "task_objective_weight": 1.0,
        "c1_relative_weight": 0.0,
        "c2_relative_weight": 0.0,
        "c3_relative_weight": 0.0,
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.0,
        "leaf_exact_supervision": False,
        "internal_supervision_kind": "none",
        "internal_label_rate": 0.0,
        "preserve_requested_leaf_tokens": False,
        "official_fno_preserve_requested_leaf_tokens": False,
        # Pass through tree_leaf_fno_* fields so the standalone FNO uses the
        # same width / n_modes / n_layers as the tree's leaf FNO.  None means
        # "use the standalone FNO's legacy default".
        "tree_leaf_fno_width": getattr(config, "tree_leaf_fno_width", None),
        "tree_leaf_fno_n_modes": getattr(config, "tree_leaf_fno_n_modes", None),
        "tree_leaf_fno_n_layers": getattr(config, "tree_leaf_fno_n_layers", None),
        "tree_leaf_fno_pooling": getattr(config, "tree_leaf_fno_pooling", None),
        "comparison_mode": str(comparison_mode),
    }
    return OPSCountConfig(**{**asdict(config), **locked})


def _effective_config_for_family(
    *,
    benchmark: FullDocDiagnosticBenchmarkSpec,
    baseline_family: str,
    config: OPSCountConfig,
) -> OPSCountConfig:
    family = _normalize_baseline_family(baseline_family)
    if family in {"official_fno", "official_fno_sumlen"}:
        return _official_fno_locked_config_for_benchmark(
            benchmark=benchmark,
            config=config,
        )
    return config


def _expected_epochs_for_config(config: OPSCountConfig) -> int:
    schedule = str(getattr(config, "tree_training_schedule", "") or "").strip().lower()
    if schedule == "two_stage":
        total = int(getattr(config, "tree_stage1_epochs", 0)) + int(
            getattr(config, "tree_stage2_epochs", 0)
        )
        if total > 0:
            return int(total)
    return int(getattr(config, "n_epochs", 0))


def _load_saved_bundle(path: str) -> MarkovOPSDataBundle:
    bundle_path = Path(str(path))
    if not bundle_path.exists():
        raise FileNotFoundError(f"bundle not found: {bundle_path}")
    return MarkovOPSDataBundle.load(bundle_path)


def _bundle_cache_path(
    output_dir: Path | None,
    *,
    benchmark: FullDocDiagnosticBenchmarkSpec,
    required_train_docs: int,
) -> Path | None:
    if output_dir is None:
        return None
    cell_dir = str(benchmark.cell_id or benchmark.name).strip().replace("/", "_")
    return (
        output_dir
        / "bundles"
        / cell_dir
        / f"observed_token_bundle_train_{int(required_train_docs)}.json"
    )


def _preferred_stable_bundle_target(
    *,
    benchmark: FullDocDiagnosticBenchmarkSpec,
    required_train_docs: int,
) -> tuple[Path | None, int]:
    canonical_capacity = int(getattr(benchmark, "canonical_train_docs_capacity", 0) or 0)
    expanded_capacity = int(getattr(benchmark, "expanded_train_docs_capacity", 0) or 0)
    canonical_path = str(getattr(benchmark, "canonical_bundle_path", "") or "").strip()
    expanded_path = str(getattr(benchmark, "expanded_bundle_path", "") or "").strip()
    if (
        expanded_path
        and expanded_capacity > 0
        and required_train_docs > max(0, canonical_capacity)
    ):
        return Path(expanded_path).expanduser(), int(expanded_capacity)
    if canonical_path:
        return Path(canonical_path).expanduser(), int(canonical_capacity)
    if expanded_path:
        return Path(expanded_path).expanduser(), int(expanded_capacity)
    return None, 0


def _materialize_base_bundle(
    *,
    benchmark: FullDocDiagnosticBenchmarkSpec,
    required_train_docs: int,
    output_dir: Path | None,
    base_bundle_path: str = "",
) -> tuple[MarkovOPSDataBundle, str]:
    required_train_docs = int(required_train_docs)

    # If an explicit bundle path is provided (e.g. from a prepared corpus),
    # load it directly — this takes precedence over all other sources.
    if str(base_bundle_path or "").strip():
        bundle_file = Path(str(base_bundle_path).strip())
        if bundle_file.exists():
            candidate = _load_saved_bundle(str(bundle_file))
            if int(len(candidate.train_docs)) >= int(required_train_docs):
                return candidate, str(bundle_file)

    cache_path = _bundle_cache_path(
        output_dir,
        benchmark=benchmark,
        required_train_docs=int(required_train_docs),
    )
    if cache_path is not None and cache_path.exists():
        return _load_saved_bundle(str(cache_path)), str(cache_path)

    base_bundle: MarkovOPSDataBundle | None = None
    base_source = ""
    if str(benchmark.canonical_bundle_path).strip():
        canonical_path = Path(str(benchmark.canonical_bundle_path)).expanduser()
        if canonical_path.exists():
            candidate = _load_saved_bundle(str(canonical_path))
            if int(required_train_docs) <= int(len(candidate.train_docs)):
                base_bundle = candidate
                base_source = str(canonical_path)

    if base_bundle is None and str(benchmark.expanded_bundle_path).strip():
        expanded_path = Path(str(benchmark.expanded_bundle_path)).expanduser()
        if expanded_path.exists():
            candidate = _load_saved_bundle(str(expanded_path))
            if int(required_train_docs) <= int(len(candidate.train_docs)):
                base_bundle = candidate
                base_source = str(expanded_path)

    if base_bundle is None:
        stable_target_path, stable_target_capacity = _preferred_stable_bundle_target(
            benchmark=benchmark,
            required_train_docs=int(required_train_docs),
        )
        target_train_docs = int(
            max(
                int(required_train_docs),
                int(stable_target_capacity),
            )
        )
        cfg = _base_config_for_benchmark(
            benchmark=benchmark,
            train_docs=int(target_train_docs),
            use_cuda=False,
            cuda_device=None,
            torch_threads=1,
            seed=0,
        )
        base_bundle = build_markov_changepoint_ops_count_data_bundle(cfg)
        base_source = "generated_base_bundle"
        save_path = stable_target_path or cache_path
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            base_bundle.save(save_path)
            base_source = str(save_path)
    return base_bundle, base_source


def _bundle_with_fixed_eval_splits(
    *,
    base_bundle: MarkovOPSDataBundle,
    base_source: str,
    train_doc_count: int,
) -> tuple[MarkovOPSDataBundle, str]:
    if int(train_doc_count) > int(len(base_bundle.train_docs)):
        raise ValueError(
            f"train_doc_count={train_doc_count} exceeds available train docs "
            f"{len(base_bundle.train_docs)}"
        )
    selected_train_docs = tuple(base_bundle.train_docs[: int(train_doc_count)])
    metadata = dict(getattr(base_bundle, "metadata", {}) or {})
    condition_ids_by_split = dict(metadata.get("condition_ids") or {})
    if condition_ids_by_split:
        sliced_condition_ids: Dict[str, List[str]] = {}
        for split_name in ("train", "val", "test"):
            values = [str(value) for value in list(condition_ids_by_split.get(split_name) or [])]
            if split_name == "train":
                values = values[: int(train_doc_count)]
            sliced_condition_ids[split_name] = values
        metadata["condition_ids"] = sliced_condition_ids
        condition_counts: Dict[str, Dict[str, int]] = {}
        for split_name, values in sliced_condition_ids.items():
            split_counts: Dict[str, int] = {}
            for value in values:
                split_counts[str(value)] = int(split_counts.get(str(value), 0)) + 1
            condition_counts[split_name] = split_counts
        metadata["condition_counts"] = condition_counts
    return (
        MarkovOPSDataBundle(
            train_docs=selected_train_docs,
            val_docs=tuple(base_bundle.val_docs),
            test_docs=tuple(base_bundle.test_docs),
            train_corpus_signature=str(_markov_corpus_signature(selected_train_docs)),
            val_corpus_signature=str(base_bundle.val_corpus_signature),
            test_corpus_signature=str(base_bundle.test_corpus_signature),
            metadata=metadata,
        ),
        f"{base_source}::train_prefix_{int(train_doc_count)}",
    )


@dataclass(frozen=True)
class _PreparedMarkovTreeData:
    root: Path
    signature: str
    benchmark_name: str
    cell_id: str
    required_train_docs: int
    fixed_leaf_tokens: int
    max_internal_depth: int
    train_prefix_counts: Tuple[int, ...]
    train_prefix_signatures: Mapping[int, str]
    train_fno_docs: Tuple[_FNOCountDoc, ...]
    val_fno_docs: Tuple[_FNOCountDoc, ...]
    test_fno_docs: Tuple[_FNOCountDoc, ...]
    leaf_orderings_by_seed: Mapping[int, Tuple[Tuple[int, ...], ...]]
    internal_orderings_by_seed: Mapping[int, Tuple[Tuple[int, ...], ...]]


def _default_prepared_data_root() -> Path:
    return REPO_ROOT / "outputs" / "_prepared_data" / "markov_full_doc_anchor_diagnostics"


def _internal_supervision_item_count(*, n_leaves: int, max_internal_depth: int) -> int:
    total = int(max(0, int(n_leaves) - 1))
    if total <= 0:
        return 0
    depth_limit = int(max_internal_depth)
    if depth_limit <= 0:
        return total
    n = int(max(0, n_leaves))
    merges = 0
    depth = 0
    while n > 1 and depth < depth_limit:
        depth += 1
        merges += int(n // 2)
        n = int((n + 1) // 2)
    return int(min(total, merges))


def _fno_doc_to_payload(doc: _FNOCountDoc) -> Dict[str, Any]:
    return {
        "n_tokens": int(doc.n_tokens),
        "leaf_token_ids": [
            [int(token) for token in leaf_tokens]
            for leaf_tokens in doc.leaf_token_ids
        ],
        "leaf_counts": [float(value) for value in doc.leaf_counts],
        "leaf_first_regimes": [int(value) for value in doc.leaf_first_regimes],
        "leaf_last_regimes": [int(value) for value in doc.leaf_last_regimes],
        "leaf_token_lengths": [int(value) for value in doc.leaf_token_lengths],
        "merge_counts_balanced": [float(value) for value in doc.merge_counts_balanced],
        "merge_sizes_balanced": [int(value) for value in doc.merge_sizes_balanced],
        "merge_token_lengths": [int(value) for value in doc.merge_token_lengths],
        "root_count": float(doc.root_count),
    }


def _fno_doc_from_payload(payload: Mapping[str, Any]) -> _FNOCountDoc:
    item = dict(payload or {})
    return _FNOCountDoc(
        n_tokens=int(item.get("n_tokens", 0)),
        leaf_token_ids=tuple(
            tuple(int(token) for token in list(leaf_tokens or ()))
            for leaf_tokens in list(item.get("leaf_token_ids") or ())
        ),
        leaf_counts=tuple(float(value) for value in list(item.get("leaf_counts") or ())),
        leaf_first_regimes=tuple(
            int(value) for value in list(item.get("leaf_first_regimes") or ())
        ),
        leaf_last_regimes=tuple(
            int(value) for value in list(item.get("leaf_last_regimes") or ())
        ),
        leaf_token_lengths=tuple(
            int(value) for value in list(item.get("leaf_token_lengths") or ())
        ),
        merge_counts_balanced=tuple(
            float(value) for value in list(item.get("merge_counts_balanced") or ())
        ),
        merge_sizes_balanced=tuple(
            int(value) for value in list(item.get("merge_sizes_balanced") or ())
        ),
        merge_token_lengths=tuple(
            int(value) for value in list(item.get("merge_token_lengths") or ())
        ),
        root_count=float(item.get("root_count", 0.0)),
    )


def _save_fno_docs(path: Path, docs: Sequence[_FNOCountDoc]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([_fno_doc_to_payload(doc) for doc in docs], indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_fno_docs(path: Path) -> Tuple[_FNOCountDoc, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(_fno_doc_from_payload(item) for item in list(payload or ()))


def _save_orderings(
    path: Path,
    payload: Mapping[int, Sequence[Sequence[int]]],
) -> None:
    serializable = {
        str(int(seed)): [[int(value) for value in ordering] for ordering in list(orderings)]
        for seed, orderings in payload.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")


def _load_orderings(path: Path) -> Dict[int, Tuple[Tuple[int, ...], ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(seed): tuple(
            tuple(int(value) for value in list(ordering or ()))
            for ordering in list(orderings or ())
        )
        for seed, orderings in dict(payload or {}).items()
    }


def _prepared_tree_data_identity_payload(
    *,
    benchmark: FullDocDiagnosticBenchmarkSpec,
    base_bundle: MarkovOPSDataBundle,
    required_train_docs: int,
    train_prefix_counts: Sequence[int],
    fixed_leaf_tokens: int,
    max_internal_depth: int,
    seeds: Sequence[int],
) -> Dict[str, Any]:
    return {
        "benchmark": str(benchmark.name),
        "cell_id": str(benchmark.cell_id or ""),
        "required_train_docs": int(required_train_docs),
        "fixed_leaf_tokens": int(fixed_leaf_tokens),
        "max_internal_depth": int(max_internal_depth),
        "train_prefix_counts": [int(value) for value in train_prefix_counts],
        "seeds": [int(seed) for seed in seeds],
        "train_corpus_signature": str(base_bundle.train_corpus_signature),
        "val_corpus_signature": str(base_bundle.val_corpus_signature),
        "test_corpus_signature": str(base_bundle.test_corpus_signature),
        "data_bundle_metadata": dict(getattr(base_bundle, "metadata", {}) or {}),
    }


def _prepared_tree_data_dir(
    *,
    benchmark: FullDocDiagnosticBenchmarkSpec,
    base_bundle: MarkovOPSDataBundle,
    required_train_docs: int,
    train_prefix_counts: Sequence[int],
    fixed_leaf_tokens: int,
    max_internal_depth: int,
    seeds: Sequence[int],
    prepared_data_root: str,
) -> tuple[Path, Dict[str, Any]]:
    root = (
        Path(str(prepared_data_root)).expanduser()
        if str(prepared_data_root).strip()
        else _default_prepared_data_root()
    )
    identity_payload = _prepared_tree_data_identity_payload(
        benchmark=benchmark,
        base_bundle=base_bundle,
        required_train_docs=required_train_docs,
        train_prefix_counts=train_prefix_counts,
        fixed_leaf_tokens=fixed_leaf_tokens,
        max_internal_depth=max_internal_depth,
        seeds=seeds,
    )
    encoded = json.dumps(
        identity_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    signature = hashlib.sha1(encoded).hexdigest()[:16]
    stem = str(benchmark.cell_id or benchmark.name)
    return root / stem / f"prepared_{signature}", {**identity_payload, "signature": signature}


def _requested_prepared_tree_prefix_signatures(
    *,
    base_bundle: MarkovOPSDataBundle,
    train_prefix_counts: Sequence[int],
) -> Dict[int, str]:
    requested_counts = tuple(dict.fromkeys(int(value) for value in train_prefix_counts))
    return {
        int(count): str(_markov_corpus_signature(base_bundle.train_docs[: int(count)]))
        for count in requested_counts
    }


def _find_compatible_prepared_markov_tree_data_root(
    *,
    benchmark: FullDocDiagnosticBenchmarkSpec,
    base_bundle: MarkovOPSDataBundle,
    required_train_docs: int,
    train_prefix_counts: Sequence[int],
    fixed_leaf_tokens: int,
    max_internal_depth: int,
    seeds: Sequence[int],
    prepared_data_root: str,
    preferred_root: Path,
) -> Path | None:
    root = (
        Path(str(prepared_data_root)).expanduser()
        if str(prepared_data_root).strip()
        else _default_prepared_data_root()
    )
    stem = str(benchmark.cell_id or benchmark.name)
    benchmark_root = root / stem
    if not benchmark_root.exists():
        return None
    requested_prefix_signatures = _requested_prepared_tree_prefix_signatures(
        base_bundle=base_bundle,
        train_prefix_counts=train_prefix_counts,
    )
    requested_seeds = {int(seed) for seed in seeds}
    candidates: List[Tuple[int, int, int, str, Path]] = []
    for metadata_path in sorted(benchmark_root.glob("prepared_*/metadata.json")):
        candidate_root = metadata_path.parent
        if candidate_root == preferred_root:
            continue
        prefix_manifest_path = candidate_root / "train_prefix_manifests.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            prefix_manifest = json.loads(prefix_manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
            continue
        if str(metadata.get("benchmark", "")) != str(benchmark.name):
            continue
        if str(metadata.get("cell_id", "")) != str(benchmark.cell_id or ""):
            continue
        if int(metadata.get("fixed_leaf_tokens", 0)) != int(fixed_leaf_tokens):
            continue
        if int(metadata.get("max_internal_depth", 0)) != int(max_internal_depth):
            continue
        if str(metadata.get("val_corpus_signature", "")) != str(base_bundle.val_corpus_signature):
            continue
        if str(metadata.get("test_corpus_signature", "")) != str(base_bundle.test_corpus_signature):
            continue
        candidate_required_train_docs = int(metadata.get("required_train_docs", 0))
        if candidate_required_train_docs < int(required_train_docs):
            continue
        candidate_prefix_signatures = {
            int(key): str(value)
            for key, value in dict(prefix_manifest.get("train_prefix_signatures") or {}).items()
        }
        if any(
            candidate_prefix_signatures.get(int(count)) != str(signature)
            for count, signature in requested_prefix_signatures.items()
        ):
            continue
        available_seeds = {
            int(seed) for seed in list(metadata.get("seeds") or ())
        }
        if not requested_seeds.issubset(available_seeds):
            continue
        candidates.append(
            (
                candidate_required_train_docs,
                len(candidate_prefix_signatures),
                len(available_seeds),
                str(metadata.get("signature", "")),
                candidate_root,
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return candidates[0][-1]


def _load_prepared_markov_tree_data(root: Path) -> _PreparedMarkovTreeData:
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    prefix_manifest = json.loads((root / "train_prefix_manifests.json").read_text(encoding="utf-8"))
    leaf_orderings_by_seed = _load_orderings(root / "leaf_orderings.json")
    internal_orderings_by_seed = _load_orderings(root / "internal_orderings.json")
    train_prefix_signatures = {
        int(key): str(value)
        for key, value in dict(prefix_manifest.get("train_prefix_signatures") or {}).items()
    }
    return _PreparedMarkovTreeData(
        root=root,
        signature=str(metadata.get("signature", "")),
        benchmark_name=str(metadata.get("benchmark", "")),
        cell_id=str(metadata.get("cell_id", "")),
        required_train_docs=int(metadata.get("required_train_docs", 0)),
        fixed_leaf_tokens=int(metadata.get("fixed_leaf_tokens", 0)),
        max_internal_depth=int(metadata.get("max_internal_depth", 0)),
        train_prefix_counts=tuple(
            int(value) for value in list(metadata.get("train_prefix_counts") or ())
        ),
        train_prefix_signatures=train_prefix_signatures,
        train_fno_docs=_load_fno_docs(root / "train_fno_docs.json"),
        val_fno_docs=_load_fno_docs(root / "val_fno_docs.json"),
        test_fno_docs=_load_fno_docs(root / "test_fno_docs.json"),
        leaf_orderings_by_seed=leaf_orderings_by_seed,
        internal_orderings_by_seed=internal_orderings_by_seed,
    )


def _acquire_prepared_data_lock(lock_dir: Path, *, ready_path: Path, timeout_s: float = 300.0) -> bool:
    deadline = time.time() + float(timeout_s)
    while True:
        try:
            lock_dir.mkdir(parents=False, exist_ok=False)
            return True
        except FileExistsError:
            if ready_path.exists():
                return False
            if time.time() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for prepared-data lock at {lock_dir}"
                )
            time.sleep(0.1)


def _ensure_prepared_markov_tree_data(
    *,
    benchmark: FullDocDiagnosticBenchmarkSpec,
    base_bundle: MarkovOPSDataBundle,
    required_train_docs: int,
    train_prefix_counts: Sequence[int],
    fixed_leaf_tokens: int,
    max_internal_depth: int,
    seeds: Sequence[int],
    prepared_data_root: str,
    allow_create: bool,
) -> _PreparedMarkovTreeData:
    root, identity_payload = _prepared_tree_data_dir(
        benchmark=benchmark,
        base_bundle=base_bundle,
        required_train_docs=required_train_docs,
        train_prefix_counts=train_prefix_counts,
        fixed_leaf_tokens=fixed_leaf_tokens,
        max_internal_depth=max_internal_depth,
        seeds=seeds,
        prepared_data_root=prepared_data_root,
    )
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        return _load_prepared_markov_tree_data(root)
    compatible_root = _find_compatible_prepared_markov_tree_data_root(
        benchmark=benchmark,
        base_bundle=base_bundle,
        required_train_docs=required_train_docs,
        train_prefix_counts=train_prefix_counts,
        fixed_leaf_tokens=fixed_leaf_tokens,
        max_internal_depth=max_internal_depth,
        seeds=seeds,
        prepared_data_root=prepared_data_root,
        preferred_root=root,
    )
    if compatible_root is not None:
        return _load_prepared_markov_tree_data(compatible_root)
    if not bool(allow_create):
        raise FileNotFoundError(
            f"prepared Markov tree data missing at {root}; run prepare_data first "
            "or enable prepared_data_allow_create"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = root.parent / f"{root.name}.lock"
    acquired = _acquire_prepared_data_lock(lock_dir, ready_path=metadata_path)
    if not acquired:
        return _load_prepared_markov_tree_data(root)
    try:
        if metadata_path.exists():
            return _load_prepared_markov_tree_data(root)
        root.mkdir(parents=True, exist_ok=True)
        bundle_path = root / "base_bundle.json"
        base_bundle.save(bundle_path)
        train_fno_docs = tuple(
            _prepare_fno_count_docs(base_bundle.train_docs, leaf_tokens=int(fixed_leaf_tokens))
        )
        val_fno_docs = tuple(
            _prepare_fno_count_docs(base_bundle.val_docs, leaf_tokens=int(fixed_leaf_tokens))
        )
        test_fno_docs = tuple(
            _prepare_fno_count_docs(base_bundle.test_docs, leaf_tokens=int(fixed_leaf_tokens))
        )
        _save_fno_docs(root / "train_fno_docs.json", train_fno_docs)
        _save_fno_docs(root / "val_fno_docs.json", val_fno_docs)
        _save_fno_docs(root / "test_fno_docs.json", test_fno_docs)
        leaf_orderings_payload: Dict[int, Tuple[Tuple[int, ...], ...]] = {}
        internal_orderings_payload: Dict[int, Tuple[Tuple[int, ...], ...]] = {}
        for seed in tuple(int(value) for value in seeds):
            leaf_orderings_payload[int(seed)] = tuple(
                _deterministic_sample_ordering(
                    n_items=len(doc.leaf_token_ids),
                    seed=int(seed) + 81_000 + int(doc_idx),
                )
                for doc_idx, doc in enumerate(train_fno_docs)
            )
            internal_orderings_payload[int(seed)] = tuple(
                _deterministic_sample_ordering(
                    n_items=_internal_supervision_item_count(
                        n_leaves=len(doc.leaf_token_ids),
                        max_internal_depth=int(max_internal_depth),
                    ),
                    seed=int(seed) + 91_000 + int(doc_idx),
                )
                for doc_idx, doc in enumerate(train_fno_docs)
            )
        _save_orderings(root / "leaf_orderings.json", leaf_orderings_payload)
        _save_orderings(root / "internal_orderings.json", internal_orderings_payload)
        prefix_manifest_payload = {
            "train_prefix_counts": [int(value) for value in train_prefix_counts],
            "train_prefix_signatures": {
                str(int(value)): str(
                    _markov_corpus_signature(base_bundle.train_docs[: int(value)])
                )
                for value in train_prefix_counts
            },
        }
        (root / "train_prefix_manifests.json").write_text(
            json.dumps(prefix_manifest_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        metadata_path.write_text(
            json.dumps(identity_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return _load_prepared_markov_tree_data(root)
    finally:
        try:
            lock_dir.rmdir()
        except Exception:
            pass


def _metrics_payload(metrics: SketchMetrics) -> Dict[str, Any]:
    return {
        **asdict(metrics),
        **_sketch_metric_alias_payload(metrics),
    }


def _config_like_value(config_like: Any, field: str) -> Any:
    if isinstance(config_like, Mapping):
        value = config_like.get(field)
    else:
        value = getattr(config_like, field, None)
    if isinstance(value, Path):
        return str(value)
    return value


def _tree_stage2_supervision_snapshot(config_like: Any) -> Dict[str, Any]:
    leaf_kind = str(_config_like_value(config_like, "leaf_supervision_kind") or "").strip().lower()
    internal_kind = str(
        _config_like_value(config_like, "internal_supervision_kind") or ""
    ).strip().lower()
    leaf_rate = _safe_float(_config_like_value(config_like, "leaf_label_rate"), default=0.0)
    internal_rate = _safe_float(
        _config_like_value(config_like, "internal_label_rate"),
        default=0.0,
    )
    leaf_active = bool(leaf_kind not in {"", "none"} and leaf_rate > 1e-12)
    internal_active = bool(
        internal_kind not in {"", "none"} and internal_rate > 1e-12
    )
    return {
        "stage2_leaf_supervision_active": bool(leaf_active),
        "stage2_internal_supervision_active": bool(internal_active),
        "stage2_local_supervision_active": bool(leaf_active or internal_active),
    }


def _tree_semantic_config_snapshot(config_like: Any) -> Dict[str, Any]:
    snapshot = {
        field: _config_like_value(config_like, field)
        for field in TREE_NEURAL_SEMANTIC_CONFIG_FIELDS
    }
    snapshot.update(_tree_stage2_supervision_snapshot(snapshot))
    return snapshot


def _tree_semantic_objective_snapshot(objective_like: Any) -> Dict[str, Any]:
    return {
        field: _config_like_value(objective_like, field)
        for field in TREE_NEURAL_OBJECTIVE_SNAPSHOT_FIELDS
    }


def _semantic_snapshot_values_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) is bool(right)
    if isinstance(left, (int, float, np.integer, np.floating)) and isinstance(
        right,
        (int, float, np.integer, np.floating),
    ):
        left_float = float(left)
        right_float = float(right)
        if not np.isfinite(left_float) or not np.isfinite(right_float):
            if np.isnan(left_float) and np.isnan(right_float):
                return True
            return left_float == right_float
        return bool(np.isclose(left_float, right_float, atol=1e-12, rtol=0.0))
    return left == right


def _semantic_snapshot_diff(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    diff: Dict[str, Dict[str, Any]] = {}
    for field in sorted(set(before) | set(after)):
        before_value = before.get(field)
        after_value = after.get(field)
        if _semantic_snapshot_values_equal(before_value, after_value):
            continue
        diff[field] = {
            "before": before_value,
            "after": after_value,
        }
    return diff


def _tree_semantic_validation_allowance(
    *,
    baseline_family: str,
    effective_config: OPSCountConfig,
) -> Dict[str, Any]:
    normalized_family = _normalize_baseline_family(baseline_family)
    collapse_mode = str(getattr(effective_config, "tree_exact_collapse_mode", "") or "").strip()
    if collapse_mode in TREE_NEURAL_EXACT_LOCKED_MODES:
        return {
            "requested_to_pre_allowed_fields": set(TREE_NEURAL_SEMANTIC_CONFIG_FIELDS),
            "pre_to_post_allowed_fields": set(TREE_NEURAL_SEMANTIC_CONFIG_FIELDS),
            "status": "locked_exception",
            "note": f"tree_exact_collapse_mode={collapse_mode}",
        }
    if normalized_family in TREE_NEURAL_LEGACY_PROFILE_FAMILIES:
        return {
            "requested_to_pre_allowed_fields": set(),
            "pre_to_post_allowed_fields": {
                "local_law_weight",
                "c1_relative_weight",
                "c2_relative_weight",
                "c3_relative_weight",
            },
            "status": "legacy_family_profile",
            "note": f"baseline_family={normalized_family}",
        }
    return {
        "requested_to_pre_allowed_fields": set(),
        "pre_to_post_allowed_fields": {"local_law_weight"},
        "status": "validated",
        "note": "",
    }


def _validate_tree_semantic_preservation(
    *,
    baseline_family: str,
    requested_config: OPSCountConfig,
    pre_family_config: OPSCountConfig,
    post_family_config: OPSCountConfig,
) -> Dict[str, Any]:
    normalized_family = _normalize_baseline_family(baseline_family)
    requested_snapshot = _tree_semantic_config_snapshot(requested_config)
    pre_family_snapshot = _tree_semantic_config_snapshot(pre_family_config)
    post_family_snapshot = _tree_semantic_config_snapshot(post_family_config)
    requested_to_pre = _semantic_snapshot_diff(requested_snapshot, pre_family_snapshot)
    pre_to_post = _semantic_snapshot_diff(pre_family_snapshot, post_family_snapshot)
    if normalized_family not in TREE_NEURAL_BASELINE_FAMILIES:
        return {
            "status": "inactive_for_family",
            "note": "",
            "requested_to_pre_family_normalization": requested_to_pre,
            "pre_to_post_family_normalization": pre_to_post,
            "unexpected_requested_to_pre_family_normalization": {},
            "unexpected_pre_to_post_family_normalization": {},
        }

    allowance = _tree_semantic_validation_allowance(
        baseline_family=normalized_family,
        effective_config=post_family_config,
    )
    unexpected_requested_to_pre = {
        field: payload
        for field, payload in requested_to_pre.items()
        if field not in allowance["requested_to_pre_allowed_fields"]
    }
    unexpected_pre_to_post = {
        field: payload
        for field, payload in pre_to_post.items()
        if field not in allowance["pre_to_post_allowed_fields"]
    }
    if unexpected_requested_to_pre or unexpected_pre_to_post:
        raise ValueError(
            "unexpected tree semantic drift detected for "
            f"{normalized_family}: requested_to_pre="
            f"{json.dumps(unexpected_requested_to_pre, sort_keys=True)}; "
            f"pre_to_post={json.dumps(unexpected_pre_to_post, sort_keys=True)}"
        )
    return {
        "status": str(allowance["status"]),
        "note": str(allowance["note"]),
        "requested_to_pre_family_normalization": requested_to_pre,
        "pre_to_post_family_normalization": pre_to_post,
        "unexpected_requested_to_pre_family_normalization": unexpected_requested_to_pre,
        "unexpected_pre_to_post_family_normalization": unexpected_pre_to_post,
    }


def _validate_tree_objective_snapshot(
    *,
    baseline_family: str,
    effective_config: OPSCountConfig,
    resolved_objective: Mapping[str, Any],
) -> Dict[str, Any]:
    expected = _tree_semantic_objective_snapshot(
        _resolved_objective_metadata_for_run(
            effective_config,
            baseline_family=baseline_family,
        )
    )
    actual = _tree_semantic_objective_snapshot(resolved_objective)
    diff = _semantic_snapshot_diff(expected, actual)
    if diff:
        raise ValueError(
            "resolved objective snapshot drifted from effective config for "
            f"{_normalize_baseline_family(baseline_family)}: "
            f"{json.dumps(diff, sort_keys=True)}"
        )
    return {
        "status": "validated",
        "expected": expected,
        "actual": actual,
        "config_to_resolved_objective": diff,
    }


def _tree_leaf_pressure_ablation_diagnostics(
    *,
    config_snapshot: Mapping[str, Any],
    objective_snapshot: Mapping[str, Any],
) -> Dict[str, Any]:
    local_total = (
        _safe_float(objective_snapshot.get("local_law_c1_weight"), default=0.0)
        + _safe_float(objective_snapshot.get("local_law_c2_weight"), default=0.0)
        + _safe_float(objective_snapshot.get("local_law_c3_weight"), default=0.0)
    )
    leaf_active = bool(config_snapshot.get("stage2_leaf_supervision_active", False))
    local_active = bool(config_snapshot.get("stage2_local_supervision_active", False))
    c1_active = _safe_float(objective_snapshot.get("local_law_c1_weight"), default=0.0) > 1e-12
    return {
        "leaf_pressure_ablation_vacuous": bool(c1_active and not leaf_active),
        "local_pressure_ablation_vacuous": bool(local_total > 1e-12 and not local_active),
    }


def _tree_neural_family_effective_config(
    config: OPSCountConfig,
    *,
    family: str,
) -> OPSCountConfig:
    normalized_family = _normalize_baseline_family(family)
    if normalized_family not in TREE_NEURAL_BASELINE_FAMILIES:
        return config
    collapse_mode = str(getattr(config, "tree_exact_collapse_mode", "") or "").strip()
    if collapse_mode in TREE_NEURAL_EXACT_LOCKED_MODES:
        # Exact-collapse rows are intentionally locked to root-only parity.
        return OPSCountConfig(
            **{
                **asdict(config),
                "law_package": "",
                "leaf_weight": 0.0,
                "c2_weight": 0.0,
                "c3_weight": 0.0,
                "local_law_weight": None,
                "task_objective_weight": 1.0,
                "c1_relative_weight": 0.0,
                "c2_relative_weight": 0.0,
                "c3_relative_weight": 0.0,
                "leaf_supervision_kind": "count_only",
                "leaf_label_rate": 0.0,
                "leaf_exact_supervision": False,
                "internal_supervision_kind": "none",
                "internal_label_rate": 0.0,
            }
        )
    lambda_local = (
        float(config.local_law_weight)
        if config.local_law_weight is not None
        else float(DEFAULT_NORMALIZED_LOCAL_LAW_WEIGHT)
    )
    effective_mapping = {
        **asdict(config),
        "law_package": "",
        "leaf_weight": 0.0,
        "c2_weight": 0.0,
        "c3_weight": 0.0,
        "local_law_weight": float(lambda_local),
    }
    if normalized_family in TREE_NEURAL_LEGACY_PROFILE_FAMILIES:
        # Legacy family profiles intentionally pin the relative-weight split.
        profile = dict(TREE_NEURAL_FAMILY_PROFILES[normalized_family])
        effective_mapping.update(
            {
                "c1_relative_weight": float(profile["c1_relative_weight"]),
                "c2_relative_weight": float(profile["c2_relative_weight"]),
                "c3_relative_weight": float(profile["c3_relative_weight"]),
            }
        )
    return OPSCountConfig(**effective_mapping)


def _tree_stage1_artifact_exists(artifact_dir: str | Path) -> bool:
    artifact_root = Path(str(artifact_dir)).expanduser()
    return bool(
        artifact_root.joinpath("metadata.json").exists()
        and artifact_root.joinpath("model_state.pt").exists()
    )


def _tree_stage1_expected_layout_metadata(
    config: OPSCountConfig,
) -> Dict[str, Any]:
    summary_spec_name = str(getattr(config, "summary_spec_name", "") or "").strip()
    state_dim = int(getattr(config, "state_dim", 0) or 0)
    slot_count = int(getattr(config, "slot_count", 0) or 0)
    raw_count_dim = int(getattr(config, "tree_theorem_count_dim", 0) or 0)
    raw_first_dim = int(getattr(config, "tree_theorem_first_dim", 0) or 0)
    raw_last_dim = int(getattr(config, "tree_theorem_last_dim", 0) or 0)
    count_dim = 0
    first_dim = 0
    last_dim = 0
    residual_dim = 0
    if summary_spec_name and state_dim > 0 and slot_count >= 4:
        if raw_count_dim > 0 and raw_first_dim > 0 and raw_last_dim > 0:
            count_dim = int(raw_count_dim)
            first_dim = int(raw_first_dim)
            last_dim = int(raw_last_dim)
            residual_dim = max(
                0,
                int(state_dim) - int(count_dim) - int(first_dim) - int(last_dim),
            )
        elif state_dim % slot_count == 0:
            slot_dim = int(state_dim) // int(slot_count)
            count_dim = int(slot_dim)
            first_dim = int(slot_dim)
            last_dim = int(slot_dim)
            residual_dim = max(0, int(slot_count) - 3) * int(slot_dim)
    theorem_surface_mode = str(
        getattr(config, "tree_theorem_surface_mode", "") or ""
    ).strip()
    shared_surface_modes = {
        "shared_bottleneck",
        "shared_feature",
        "shared_feature_adapters",
        "factorized_score_fiber",
    }
    carrier_state_dim = 0
    carrier_state_merger_in_features = 0
    if summary_spec_name and theorem_surface_mode == "opaque_carrier_exact_sketch":
        carrier_state_dim = int(state_dim)
        residual_dim = int(state_dim)
        count_dim = 1
        first_dim = int(getattr(config, "n_regimes", 0) or 0)
        last_dim = int(getattr(config, "n_regimes", 0) or 0)
        summary_state_merger_in_features = 0
        carrier_state_merger_in_features = 2 * int(carrier_state_dim)
    elif summary_spec_name and theorem_surface_mode in shared_surface_modes:
        summary_state_merger_in_features = 2 * int(state_dim)
    elif summary_spec_name:
        summary_state_merger_in_features = (
            2 * int(count_dim) + int(first_dim) + int(last_dim)
        )
    else:
        summary_state_merger_in_features = 0
    return {
        "state_dim": int(state_dim),
        "summary_spec_name": str(summary_spec_name),
        "slot_count": int(slot_count),
        "theorem_surface_mode": str(theorem_surface_mode),
        "theorem_feature_dim": int(
            getattr(config, "tree_theorem_feature_dim", 0) or 0
        ),
        "theorem_feature_hidden_dim": int(
            getattr(config, "tree_theorem_feature_hidden_dim", 0) or 0
        ),
        "theorem_score_dim": int(
            getattr(config, "tree_theorem_score_dim", 0) or 0
        ),
        "theorem_fiber_dim": int(
            getattr(config, "tree_theorem_fiber_dim", 0) or 0
        ),
        "theorem_aux_dim": int(
            getattr(config, "tree_theorem_aux_dim", 0) or 0
        ),
        "carrier_state_dim": int(carrier_state_dim),
        "merge_hidden_dim": int(
            getattr(config, "tree_merge_hidden_dim", 0) or 0
        ),
        "count_theorem_dim": int(count_dim),
        "first_theorem_dim": int(first_dim),
        "last_theorem_dim": int(last_dim),
        "residual_dim": int(residual_dim),
        "summary_state_merger_in_features": int(summary_state_merger_in_features),
        "carrier_state_merger_in_features": int(carrier_state_merger_in_features),
        "task_head_mode": str(getattr(config, "tree_task_head_mode", "") or ""),
        "summary_spec_root_mode": str(
            getattr(config, "tree_summary_spec_root_mode", "") or ""
        ),
        "theorem_count_head_mode": str(
            getattr(config, "tree_theorem_count_head_mode", "") or ""
        ),
        "c2_mode": str(getattr(config, "tree_c2_mode", "") or ""),
        "tree_model_version": str(getattr(config, "tree_model_version", "") or ""),
    }


def _tree_stage1_artifact_resume_compatible(
    artifact_dir: str | Path,
    *,
    config: OPSCountConfig,
) -> bool:
    artifact_root = Path(str(artifact_dir)).expanduser()
    metadata_path = artifact_root / "metadata.json"
    model_state_path = artifact_root / "model_state.pt"
    if not metadata_path.exists() or not model_state_path.exists():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected = {
        "n_regimes": int(getattr(config, "n_regimes", 0)),
        "vocab_size": int(getattr(config, "vocab_size", 0)),
        "fixed_leaf_tokens": int(getattr(config, "fixed_leaf_tokens", 0)),
    }
    expected.update(_tree_stage1_expected_layout_metadata(config))
    for key, expected_value in expected.items():
        actual_value = payload.get(key, None)
        if actual_value is None:
            return False
        if isinstance(expected_value, str):
            if str(actual_value) != str(expected_value):
                return False
            continue
        try:
            if int(actual_value) != int(expected_value):
                return False
        except Exception:
            return False
    return True


def _tree_stage1_artifact_cache_key(
    *,
    benchmark: FullDocDiagnosticBenchmarkSpec,
    baseline_family: str,
    train_doc_count: int,
    config: OPSCountConfig,
) -> str:
    # Only include fields that actually affect stage1 training.
    # Stage1 overrides leaf_label_rate→1.0, internal_label_rate→varies,
    # root_weight→stage1_root_weight, leaf_supervision_kind→"full_sketch",
    # c1/c2→max(val,1.0), etc.  Stage2-only fields like budget,
    # mass_target, package_semantics must NOT affect the cache key,
    # otherwise packages with identical stage1 behavior (e.g. full100 vs
    # superset+10%) get different artifacts and diverge due to randomness.
    full = asdict(config)
    # Keys that stage1 overrides or that only matter for stage2/budget:
    _STAGE2_ONLY_KEYS = {
        "local_law_weight",
        "c1_relative_weight",
        "c2_relative_weight",
        "c3_relative_weight",
        "leaf_label_rate",
        "internal_label_rate",
        "leaf_supervision_kind",
        "internal_supervision_kind",
        "budget_total_calls",
        "budget_total_calls_per_doc",
        "full_doc_budget_share",
        "mass_target_per_doc",
        "doc_consumption_mode",
        "local_split_mode",
        "local_allocation_policy",
        "package_semantics",
        "root_query_rate",
        "c2_weight",
        "c3_weight",
        "tree_stage1_artifact_dir",
        "tree_stage1_artifact_root",
        "comparison_mode",
        "comparison_surface_diff",
        "comparison_surface_snapshot",
        "computed_assumed_doc_tokens",
        "computed_assumed_internal_nodes",
        "computed_assumed_leaves",
        "computed_doc_review_mass_per_doc",
        "computed_internal_mass_full_per_doc",
        "computed_internal_mass_per_doc",
        "computed_leaf_mass_full_per_doc",
        "computed_leaf_mass_per_doc",
        "computed_local_mass_per_doc",
        "computed_total_mass_per_doc",
        "artifact_dir",
    }
    stage1_config = {
        k: v for k, v in full.items() if k not in _STAGE2_ONLY_KEYS
    }
    payload = {
        "benchmark": str(benchmark.name),
        "cell_id": str(benchmark.cell_id or ""),
        "baseline_family": str(baseline_family),
        "train_doc_count": int(train_doc_count),
        "config": stage1_config,
        "_cache_version": 2,  # bump to invalidate old artifacts
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _default_tree_stage1_artifact_root() -> Path:
    return REPO_ROOT / "outputs" / "_stage1_artifacts" / "markov_full_doc_anchor_diagnostics"


def _tree_stage1_artifact_dir_for_run(
    *,
    benchmark: FullDocDiagnosticBenchmarkSpec,
    baseline_family: str,
    train_doc_count: int,
    config: OPSCountConfig,
) -> Path:
    artifact_root_text = str(
        getattr(config, "tree_stage1_artifact_root", "") or ""
    ).strip()
    artifact_root = (
        Path(artifact_root_text).expanduser()
        if artifact_root_text
        else _default_tree_stage1_artifact_root()
    )
    stem_parts: List[str] = []
    if str(benchmark.cell_id or "").strip():
        stem_parts.append(str(benchmark.cell_id))
    else:
        stem_parts.append(str(benchmark.name))
    stem_parts.extend(
        [
            str(baseline_family),
            f"train_{int(train_doc_count)}",
            f"seed_{int(config.seed)}",
            _tree_stage1_artifact_cache_key(
                benchmark=benchmark,
                baseline_family=baseline_family,
                train_doc_count=int(train_doc_count),
                config=config,
            ),
        ]
    )
    return artifact_root / "__".join(stem_parts)


def _effective_train_config_for_full_doc_run(
    *,
    benchmark: FullDocDiagnosticBenchmarkSpec,
    baseline_family: str,
    train_doc_count: int,
    config: OPSCountConfig,
) -> OPSCountConfig:
    normalized_family = _normalize_baseline_family(baseline_family)
    if normalized_family not in TREE_NEURAL_BASELINE_FAMILIES:
        return config
    effective_config = config
    artifact_dir_text = str(
        getattr(config, "tree_stage1_artifact_dir", "") or ""
    ).strip()
    if not artifact_dir_text:
        artifact_dir = _tree_stage1_artifact_dir_for_run(
            benchmark=benchmark,
            baseline_family=normalized_family,
            train_doc_count=int(train_doc_count),
            config=config,
        )
        artifact_dir_text = str(artifact_dir)
        effective_config = replace(
            effective_config,
            tree_stage1_artifact_dir=str(artifact_dir),
        )
    if bool(getattr(effective_config, "tree_stage1_resume_if_available", False)) and _tree_stage1_artifact_resume_compatible(
        artifact_dir_text,
        config=effective_config,
    ):
        effective_config = replace(
            effective_config,
            tree_stage1_artifact_dir=str(Path(artifact_dir_text).expanduser()),
            tree_stage1_epochs=0,
        )
    return effective_config


def _resolved_objective_metadata_for_run(
    config: OPSCountConfig,
    *,
    baseline_family: str,
) -> Dict[str, Any]:
    normalized_family = _normalize_baseline_family(baseline_family)
    if normalized_family not in TREE_NEURAL_BASELINE_FAMILIES:
        return {
            "parameterization": "inactive_for_family",
            "weighting_scheme": "inactive_for_family",
            "optimization_root_weight": 0.0,
            "local_law_c1_weight": 0.0,
            "local_law_c2_weight": 0.0,
            "local_law_c3_weight": 0.0,
            "task_objective_weight_source": "inactive_for_family",
            "proxy_schedule_consistency_weight": 0.0,
            "theorem_terms": [],
            "proxy_terms": [
                {
                    "name": "schedule_consistency",
                    "weight": 0.0,
                    "active": False,
                    "evidence_status": "PROXY_ONLY",
                    "notes": "Associativity proxy over schedule spread; not a Lean local law.",
                }
            ],
            "formal_notes": (
                "This baseline has no active theorem-facing local-law objective."
            ),
            "theorem_local_law_total_weight": 0.0,
            "proxy_schedule_term_total_weight": 0.0,
            "objective_surface_name": MARKOV_FULL_DOC_OBJECTIVE_SURFACE,
            "objective_surface_distinct_from": [TREEPO_REGULARIZED_OBJECTIVE_SURFACE],
            "objective_surface_distinct_note": (
                "The Markov full-doc normalized task/local-law weighting surface is "
                "not the same objective family as the separate TreePO Regularized "
                "Objective note."
            ),
            "paper_to_lean_local_law_mapping": dict(PAPER_TO_LEAN_LOCAL_LAW_MAPPING),
            "objective_weights_active": False,
        }
    objective = _build_objective_summary(config)
    metadata = {
        "parameterization": str(objective.get("parameterization", "")),
        "weighting_scheme": str(objective.get("weighting_scheme", "")),
        "optimization_root_weight": float(
            objective.get("optimization_root_weight", float("nan"))
        ),
        "local_law_c1_weight": float(
            objective.get("local_law_c1_weight", float("nan"))
        ),
        "local_law_c2_weight": float(
            objective.get("local_law_c2_weight", float("nan"))
        ),
        "local_law_c3_weight": float(
            objective.get("local_law_c3_weight", float("nan"))
        ),
        "task_objective_weight_source": str(
            objective.get("task_objective_weight_source", "")
        ),
        "proxy_schedule_consistency_weight": float(
            objective.get("proxy_schedule_consistency_weight", 0.0)
        ),
        "theorem_terms": list(objective.get("theorem_terms") or []),
        "proxy_terms": list(objective.get("proxy_terms") or []),
        "formal_notes": str(objective.get("formal_notes", "")),
        "theorem_local_law_total_weight": float(
            float(objective.get("local_law_c1_weight", 0.0))
            + float(objective.get("local_law_c2_weight", 0.0))
            + float(objective.get("local_law_c3_weight", 0.0))
        ),
        "proxy_schedule_term_total_weight": float(
            objective.get("proxy_schedule_consistency_weight", 0.0)
        ),
        "objective_surface_name": MARKOV_FULL_DOC_OBJECTIVE_SURFACE,
        "objective_surface_distinct_from": [TREEPO_REGULARIZED_OBJECTIVE_SURFACE],
        "objective_surface_distinct_note": (
            "The Markov full-doc normalized task/local-law weighting surface is "
            "not the same objective family as the separate TreePO Regularized "
            "Objective note."
        ),
        "paper_to_lean_local_law_mapping": dict(PAPER_TO_LEAN_LOCAL_LAW_MAPPING),
        "objective_weights_active": True,
    }
    metadata["semantics_version"] = CURRENT_TREE_NEURAL_SEMANTICS_VERSION
    return metadata


def _run_semantics_metadata(
    run: Mapping[str, Any],
) -> Dict[str, Any]:
    family = _normalize_baseline_family(str(run.get("baseline_family", "")))
    parameterization = str(run.get("parameterization", "")).strip()
    c2_metric_kind = str(run.get("c2_metric_kind", "")).strip()
    optimization_root_weight = float(
        run.get("optimization_root_weight", float("nan"))
    )
    local_law_c1_weight = float(run.get("local_law_c1_weight", float("nan")))
    local_law_c2_weight = float(run.get("local_law_c2_weight", float("nan")))
    local_law_c3_weight = float(run.get("local_law_c3_weight", float("nan")))
    task_objective_weight_source = str(
        run.get("task_objective_weight_source", "")
    ).strip()
    semantics_version = str(run.get("semantics_version", "")).strip()
    run_intent_validation_status = str(
        run.get("run_intent_validation_status", "")
    ).strip()
    has_complete_run_intent = all(
        intent_is_complete(cast(Mapping[str, Any] | None, run.get(field_name)))
        for field_name in (
            "requested_run_intent",
            "effective_run_intent",
            "reported_run_intent",
        )
    )
    objective_weights_active_raw = run.get("objective_weights_active", None)
    if objective_weights_active_raw is None and family in TREE_NEURAL_BASELINE_FAMILIES:
        objective_weights_active = parameterization not in {"", "inactive_for_family"}
    else:
        objective_weights_active = bool(objective_weights_active_raw)

    legacy_reasons: List[str] = []
    if family in TREE_NEURAL_BASELINE_FAMILIES:
        if semantics_version and semantics_version != CURRENT_TREE_NEURAL_SEMANTICS_VERSION:
            legacy_reasons.append(f"semantics_version={semantics_version}")
        if parameterization == "legacy_term_weights":
            legacy_reasons.append("direct_weight_tree_baseline")
        if c2_metric_kind and c2_metric_kind != FNO_TREE_C2_METRIC_KIND:
            legacy_reasons.append(f"c2_metric_kind={c2_metric_kind}")
        if not has_complete_run_intent:
            legacy_reasons.append("missing_run_intent_metadata")
        if not semantics_version and not c2_metric_kind:
            legacy_reasons.append("missing_tree_neural_semantics_metadata")
        if not c2_metric_kind:
            c2_metric_kind = "legacy_or_unspecified"
    if legacy_reasons:
        comparison_semantics = "legacy_quarantined"
    elif (
        family in TREE_NEURAL_BASELINE_FAMILIES
        and run_intent_validation_status == "locked_comparator"
    ):
        comparison_semantics = "locked_comparator"
    else:
        comparison_semantics = "current"
    comparison_semantics_label = (
        "legacy_quarantined_tree_neural_semantics"
        if legacy_reasons
        else (
            "locked_comparator"
            if comparison_semantics == "locked_comparator"
            else (
                CURRENT_TREE_NEURAL_SEMANTICS_VERSION
                if family in TREE_NEURAL_BASELINE_FAMILIES
                else "default"
            )
        )
    )
    return {
        "parameterization": parameterization,
        "optimization_root_weight": optimization_root_weight,
        "local_law_c1_weight": local_law_c1_weight,
        "local_law_c2_weight": local_law_c2_weight,
        "local_law_c3_weight": local_law_c3_weight,
        "task_objective_weight_source": task_objective_weight_source,
        "c2_metric_kind": c2_metric_kind,
        "comparison_semantics": comparison_semantics,
        "comparison_semantics_label": comparison_semantics_label,
        "legacy_semantics": bool(legacy_reasons),
        "legacy_semantics_reason": ";".join(legacy_reasons),
        "semantics_version": semantics_version,
        "objective_weights_active": objective_weights_active,
    }


def _is_headline_comparison_semantics(value: str) -> bool:
    normalized = str(value or "").strip()
    return normalized in {"current", "locked_comparator"}


def _normalize_loaded_run_semantics(run: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(run)
    normalized_family = _normalize_baseline_family(str(normalized.get("baseline_family", "")))
    if normalized_family not in TREE_NEURAL_BASELINE_FAMILIES:
        inactive_defaults = {
            "parameterization": "inactive_for_family",
            "weighting_scheme": "inactive_for_family",
            "optimization_root_weight": 0.0,
            "local_law_c1_weight": 0.0,
            "local_law_c2_weight": 0.0,
            "local_law_c3_weight": 0.0,
            "task_objective_weight_source": "inactive_for_family",
            "proxy_schedule_consistency_weight": 0.0,
            "theorem_local_law_total_weight": 0.0,
            "proxy_schedule_term_total_weight": 0.0,
            "objective_weights_active": False,
        }
        for key, value in inactive_defaults.items():
            current_value = normalized.get(key)
            if (
                key not in normalized
                or current_value in {"", None}
                or (
                    isinstance(current_value, float)
                    and not np.isfinite(float(current_value))
                )
            ):
                normalized[key] = value
    requested_run_intent = cast(
        Mapping[str, Any] | None,
        normalized.get("requested_run_intent"),
    )
    effective_run_intent = cast(
        Mapping[str, Any] | None,
        normalized.get("effective_run_intent"),
    )
    reported_run_intent = cast(
        Mapping[str, Any] | None,
        normalized.get("reported_run_intent"),
    )
    if intent_is_complete(reported_run_intent):
        normalized.setdefault(
            "run_intent_hash",
            intent_hash(reported_run_intent),
        )
    if normalized_family in TREE_NEURAL_BASELINE_FAMILIES:
        if (
            not str(normalized.get("run_intent_validation_status", "")).strip()
            and intent_is_complete(requested_run_intent)
            and intent_is_complete(effective_run_intent)
        ):
            drift = intent_diff(requested_run_intent, effective_run_intent)
            normalized["run_intent_validation_status"] = (
                "locked_comparator"
                if drift
                and str(
                    dict(requested_run_intent or {}).get("tree_exact_collapse_mode", "")
                ).strip()
                else ("validated" if not drift else "unexpected_drift")
            )
    semantics = _run_semantics_metadata(normalized)
    for key, value in semantics.items():
        normalized[key] = value
    provenance = full_doc_baseline_provenance(
        str(normalized.get("baseline_family", "")),
        objective_weights_active=bool(
            normalized.get("objective_weights_active", semantics.get("objective_weights_active", False))
        ),
        config_like=normalized,
        c2_metric_kind=str(normalized.get("c2_metric_kind", "")),
        c2_proxy_metric_kind=str(normalized.get("c2_proxy_metric_kind", "")),
        c2_exact_witness_kind=str(normalized.get("c2_exact_witness_kind", "")),
        mean_leaves_per_doc=_safe_float(
            normalized.get("test_mean_leaves_per_doc", float("nan")),
            default=float("nan"),
        ),
    )
    for key, value in provenance.items():
        normalized.setdefault(key, value)
    return _backfill_loaded_run_fields(normalized)


def _fit_ridge_control_with_predictions(
    *,
    config: OPSCountConfig,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
) -> Dict[str, Any]:
    if not train_docs:
        zero = _eval_root_predictions([], [], tau=float(config.violation_tau))
        empty = np.zeros((0,), dtype=np.float64)
        return {
            "train_metrics": zero,
            "val_metrics": zero,
            "test_metrics": zero,
            "fit_diag": TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
            "train_preds": empty,
            "val_preds": empty,
            "test_preds": empty,
            "train_truths": empty,
            "val_truths": empty,
            "test_truths": empty,
            "train_docs_used": 0,
        }
    orders = (1, 2)
    train_x = _doc_token_ngram_feature_matrix(
        train_docs,
        vocab_size=int(config.vocab_size),
        orders=orders,
    )
    train_y = _doc_root_targets(train_docs)
    target_scale = _resolved_markov_target_scale(
        config,
        observed_targets=train_y,
    )
    train_supervision = _dense_doc_matrix_supervision_dataset(
        train_x,
        train_y,
        split="train",
        target_scale=target_scale,
        input_view="full_document_token_unigram_bigram_counts",
        metadata={"ngram_orders": [1, 2]},
    )
    model, fit_result = fit_dense_scalar_ridge_regressor(
        train_supervision,
        config=DenseScalarRidgeTrainingConfig(
            model=DenseScalarRidgeModelConfig(
                ridge_alpha=float(config.doc_level_ridge_alpha)
            )
        ),
    )

    def _predict(docs: Sequence[ChangepointMarkovDoc]) -> np.ndarray:
        if not docs:
            return np.zeros((0,), dtype=np.float64)
        features = _doc_token_ngram_feature_matrix(
            docs,
            vocab_size=int(config.vocab_size),
            orders=orders,
        )
        preds = predict_dense_scalar_ridge_regressor(model, features)
        return np.asarray(preds, dtype=np.float64).reshape(-1)

    train_preds = _predict(train_docs)
    val_preds = _predict(val_docs)
    test_preds = _predict(test_docs)
    val_truths = _doc_root_targets(val_docs)
    test_truths = _doc_root_targets(test_docs)
    fit_diag = TrainFitDiagnostics(
        train_loss_final=float("nan"),
        train_loss_curve=tuple(),
        epochs_completed=0,
        selection_metric_curve=tuple(),
        selection_mode=str(fit_result.selection_mode),
        selection_split=str(fit_result.selection_split),
        selection_metric_name=str(fit_result.selection_metric_name),
        selection_metric_value=float(fit_result.selection_metric_value),
        best_epoch=int(fit_result.best_epoch),
        train_exact_match_rate=float(_exact_match_rate(train_preds, train_y.tolist())),
        val_exact_match_rate=float(_exact_match_rate(val_preds, val_truths.tolist())),
        test_exact_match_rate=float(_exact_match_rate(test_preds, test_truths.tolist())),
    )
    return {
        "train_metrics": _eval_root_predictions(
            train_preds,
            train_y.tolist(),
            tau=float(config.violation_tau),
        ),
        "val_metrics": _eval_root_predictions(
            val_preds,
            val_truths.tolist(),
            tau=float(config.violation_tau),
        ),
        "test_metrics": _eval_root_predictions(
            test_preds,
            test_truths.tolist(),
            tau=float(config.violation_tau),
        ),
        "fit_diag": fit_diag,
        "train_preds": np.asarray(train_preds, dtype=np.float64),
        "val_preds": np.asarray(val_preds, dtype=np.float64),
        "test_preds": np.asarray(test_preds, dtype=np.float64),
        "train_truths": np.asarray(train_y, dtype=np.float64),
        "val_truths": np.asarray(val_truths, dtype=np.float64),
        "test_truths": np.asarray(test_truths, dtype=np.float64),
        "train_docs_used": int(len(train_docs)),
    }


def _tree_ridge_root_predictions(
    model: AdditiveCountSketch,
    count_docs: Sequence[_CountDoc],
    *,
    device: torch.device,
) -> np.ndarray:
    """Predict root count for each doc by encoding leaves and merging up."""
    model.eval()
    preds: List[float] = []
    with torch.no_grad():
        for doc in count_docs:
            if not doc.leaf_features:
                preds.append(0.0)
                continue
            leaf_feats = [f.to(device=device) for f in doc.leaf_features]
            states = [model.encode_leaf(x) for x in leaf_feats]
            root_state, _ = model._merge_states(
                states, schedule="balanced", collect_merge_states=False,
            )
            root_pred = float(
                model.predict_count_from_state(root_state).detach().cpu().item()
            )
            preds.append(root_pred)
    return np.asarray(preds, dtype=np.float64)


def _fit_tree_ridge_baseline_with_predictions(
    *,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
) -> Dict[str, Any]:
    """Tree-based ridge baseline: fit ridge on leaf features, merge up to root."""
    device = torch.device("cpu")
    if not train_docs:
        zero = _eval_root_predictions([], [], tau=float(config.violation_tau))
        empty = np.zeros((0,), dtype=np.float64)
        return {
            "train_metrics": zero,
            "val_metrics": zero,
            "test_metrics": zero,
            "fit_diag": TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
            "train_preds": empty,
            "val_preds": empty,
            "test_preds": empty,
            "train_truths": empty,
            "val_truths": empty,
            "test_truths": empty,
            "train_docs_used": 0,
        }

    n_regimes = int(config.n_regimes)
    vocab_size = int(config.vocab_size)
    leaf_tokens = int(config.fixed_leaf_tokens)
    feature_mode = str(config.feature_mode)
    use_endpoints = True
    ridge_alpha = float(config.doc_level_ridge_alpha)
    leaf_query_rate = float(config.leaf_query_rate)

    # Convert ChangepointMarkovDoc → _CountDoc (tree-structured).
    train_count_docs = _prepare_count_docs(
        train_docs,
        leaf_tokens=leaf_tokens,
        n_regimes=n_regimes,
        vocab_size=vocab_size,
        feature_mode=feature_mode,
    )
    val_count_docs = _prepare_count_docs(
        val_docs,
        leaf_tokens=leaf_tokens,
        n_regimes=n_regimes,
        vocab_size=vocab_size,
        feature_mode=feature_mode,
    )
    test_count_docs = _prepare_count_docs(
        test_docs,
        leaf_tokens=leaf_tokens,
        n_regimes=n_regimes,
        vocab_size=vocab_size,
        feature_mode=feature_mode,
    )

    # Build leaf-level supervision from training docs.
    train_y = _doc_root_targets(train_docs)
    target_scale = _resolved_markov_target_scale(
        config,
        observed_targets=train_y,
    )
    supervision_seed = int(seeds.get("effective_model_seed", 0)) + 71_003
    train_supervision = _leaf_ridge_tree_supervision_dataset(
        train_count_docs,
        split="train",
        target_scale=target_scale,
        n_regimes=n_regimes,
        use_endpoints=use_endpoints,
        leaf_query_rate=leaf_query_rate,
        seed=supervision_seed,
    )

    # Fit ridge on leaf features.
    ridge_model, fit_result = fit_dense_scalar_ridge_regressor(
        train_supervision,
        config=DenseScalarRidgeTrainingConfig(
            model=DenseScalarRidgeModelConfig(ridge_alpha=float(ridge_alpha))
        ),
    )

    # Build AdditiveCountSketch and load ridge weights.
    if not train_count_docs or not train_count_docs[0].leaf_features:
        raise ValueError("tree_ridge requires at least one doc with leaf features")
    feature_dim = int(train_count_docs[0].leaf_features[0].numel())
    model = AdditiveCountSketch(
        feature_dim=feature_dim,
        hidden_dim=1,
        target_scale=target_scale,
        n_regimes=n_regimes,
        use_endpoints=use_endpoints,
    ).to(device=device)
    with torch.no_grad():
        model.encoder.weight.copy_(
            torch.tensor(
                np.asarray(ridge_model.weights, dtype=np.float64).reshape(1, -1),
                device=device,
                dtype=torch.float32,
            )
        )
        model.encoder.bias.copy_(
            torch.tensor([float(ridge_model.bias)], device=device, dtype=torch.float32)
        )

    # Predict root counts via merge-up for all splits.
    train_preds = _tree_ridge_root_predictions(model, train_count_docs, device=device)
    val_preds = _tree_ridge_root_predictions(model, val_count_docs, device=device)
    test_preds = _tree_ridge_root_predictions(model, test_count_docs, device=device)

    val_truths = _doc_root_targets(val_docs)
    test_truths = _doc_root_targets(test_docs)

    tau = float(config.violation_tau)
    fit_diag = TrainFitDiagnostics(
        train_loss_final=float("nan"),
        train_loss_curve=tuple(),
        epochs_completed=0,
        selection_metric_curve=tuple(),
        selection_mode=str(fit_result.selection_mode),
        selection_split=str(fit_result.selection_split),
        selection_metric_name=str(fit_result.selection_metric_name),
        selection_metric_value=float(fit_result.selection_metric_value),
        best_epoch=int(fit_result.best_epoch),
        train_exact_match_rate=float(_exact_match_rate(train_preds, train_y.tolist())),
        val_exact_match_rate=float(_exact_match_rate(val_preds, val_truths.tolist())),
        test_exact_match_rate=float(_exact_match_rate(test_preds, test_truths.tolist())),
    )
    return {
        "train_metrics": _eval_root_predictions(
            train_preds, train_y.tolist(), tau=tau,
        ),
        "val_metrics": _eval_root_predictions(
            val_preds, val_truths.tolist(), tau=tau,
        ),
        "test_metrics": _eval_root_predictions(
            test_preds, test_truths.tolist(), tau=tau,
        ),
        "fit_diag": fit_diag,
        "train_preds": np.asarray(train_preds, dtype=np.float64),
        "val_preds": np.asarray(val_preds, dtype=np.float64),
        "test_preds": np.asarray(test_preds, dtype=np.float64),
        "train_truths": np.asarray(train_y, dtype=np.float64),
        "val_truths": np.asarray(val_truths, dtype=np.float64),
        "test_truths": np.asarray(test_truths, dtype=np.float64),
        "train_docs_used": int(len(train_docs)),
    }


def _fit_tree_doc_ridge_baseline_with_predictions(
    *,
    config: OPSCountConfig,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
) -> Dict[str, Any]:
    """Tree doc-level ridge: tree-structured features on full document, trained on root target."""
    if not train_docs:
        zero = _eval_root_predictions([], [], tau=float(config.violation_tau))
        empty = np.zeros((0,), dtype=np.float64)
        return {
            "train_metrics": zero,
            "val_metrics": zero,
            "test_metrics": zero,
            "fit_diag": TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
            "train_preds": empty,
            "val_preds": empty,
            "test_preds": empty,
            "train_truths": empty,
            "val_truths": empty,
            "test_truths": empty,
            "train_docs_used": 0,
        }

    n_regimes = int(config.n_regimes)
    vocab_size = int(config.vocab_size)
    feature_mode = str(config.feature_mode)

    # Prepare doc-level count docs (entire document = one "leaf", root count as target).
    train_count_docs = _prepare_doc_level_count_docs(
        train_docs, n_regimes=n_regimes, vocab_size=vocab_size, feature_mode=feature_mode,
    )
    val_count_docs = _prepare_doc_level_count_docs(
        val_docs, n_regimes=n_regimes, vocab_size=vocab_size, feature_mode=feature_mode,
    )
    test_count_docs = _prepare_doc_level_count_docs(
        test_docs, n_regimes=n_regimes, vocab_size=vocab_size, feature_mode=feature_mode,
    )

    # Build doc-level supervision (root count as target).
    train_y = _doc_root_targets(train_docs)
    target_scale = _resolved_markov_target_scale(
        config,
        observed_targets=train_y,
    )
    train_supervision = _doc_level_supervision_dataset(
        train_count_docs, split="train", target_scale=target_scale,
    )

    # Fit ridge on doc-level features → root count.
    model, fit_result = fit_dense_scalar_ridge_regressor(
        train_supervision,
        config=DenseScalarRidgeTrainingConfig(
            model=DenseScalarRidgeModelConfig(
                ridge_alpha=float(config.doc_level_ridge_alpha),
            ),
        ),
    )

    def _predict(count_docs: Sequence[_CountDoc]) -> np.ndarray:
        if not count_docs:
            return np.zeros((0,), dtype=np.float64)
        features = _doc_level_feature_matrix(count_docs)
        preds = predict_dense_scalar_ridge_regressor(model, features)
        return np.asarray(preds, dtype=np.float64).reshape(-1)

    train_preds = _predict(train_count_docs)
    val_preds = _predict(val_count_docs)
    test_preds = _predict(test_count_docs)
    val_truths = _doc_root_targets(val_docs)
    test_truths = _doc_root_targets(test_docs)

    tau = float(config.violation_tau)
    fit_diag = TrainFitDiagnostics(
        train_loss_final=float("nan"),
        train_loss_curve=tuple(),
        epochs_completed=0,
        selection_metric_curve=tuple(),
        selection_mode=str(fit_result.selection_mode),
        selection_split=str(fit_result.selection_split),
        selection_metric_name=str(fit_result.selection_metric_name),
        selection_metric_value=float(fit_result.selection_metric_value),
        best_epoch=int(fit_result.best_epoch),
        train_exact_match_rate=float(_exact_match_rate(train_preds, train_y.tolist())),
        val_exact_match_rate=float(_exact_match_rate(val_preds, val_truths.tolist())),
        test_exact_match_rate=float(_exact_match_rate(test_preds, test_truths.tolist())),
    )
    return {
        "train_metrics": _eval_root_predictions(
            train_preds, train_y.tolist(), tau=tau,
        ),
        "val_metrics": _eval_root_predictions(
            val_preds, val_truths.tolist(), tau=tau,
        ),
        "test_metrics": _eval_root_predictions(
            test_preds, test_truths.tolist(), tau=tau,
        ),
        "fit_diag": fit_diag,
        "train_preds": np.asarray(train_preds, dtype=np.float64),
        "val_preds": np.asarray(val_preds, dtype=np.float64),
        "test_preds": np.asarray(test_preds, dtype=np.float64),
        "train_truths": np.asarray(train_y, dtype=np.float64),
        "val_truths": np.asarray(val_truths, dtype=np.float64),
        "test_truths": np.asarray(test_truths, dtype=np.float64),
        "train_docs_used": int(len(train_docs)),
    }


def _fno_tree_root_predictions(
    model: "FNOCountSketch",
    fno_docs: Sequence["_FNOCountDoc"],
    *,
    device: torch.device,
) -> np.ndarray:
    """Predict root count for each doc using FNOCountSketch tree merge-up."""
    model.eval()
    preds: List[float] = []
    with torch.no_grad():
        for doc in fno_docs:
            if not doc.leaf_token_ids:
                preds.append(0.0)
                continue
            out = model.forward_doc(
                doc.leaf_token_ids,
                doc.leaf_counts,
                doc.merge_counts_balanced,
                schedule="balanced",
                collect_leaf=False,
                collect_c3=False,
                collect_c2=False,
                device=device,
                leaf_first_regimes=doc.leaf_first_regimes,
                leaf_last_regimes=doc.leaf_last_regimes,
            )
            root_pred = float(
                model.predict_canonical_count_from_state(out["root_state"])
                .detach()
                .cpu()
                .item()
            )
            preds.append(root_pred)
    return np.asarray(preds, dtype=np.float64)


def _balanced_exact_state_tree(
    doc: ChangepointMarkovDoc,
    *,
    leaf_tokens: int,
) -> Dict[str, List[_ExactState]]:
    spans = _leaf_spans(
        int(len(doc.token_regimes)),
        leaf_tokens=int(leaf_tokens),
    )
    leaf_states = [_exact_from_span(doc, span) for span in spans]
    merge_states: List[_ExactState] = []
    cur_states = list(leaf_states)
    while len(cur_states) > 1:
        nxt_states: List[_ExactState] = []
        i = 0
        while i < len(cur_states):
            if i + 1 >= len(cur_states):
                nxt_states.append(cur_states[i])
                i += 1
                continue
            merged = _exact_merge(cur_states[i], cur_states[i + 1])
            merge_states.append(merged)
            nxt_states.append(merged)
            i += 2
        cur_states = nxt_states
    root_states = [cur_states[0]] if cur_states else []
    return {
        "leaf": list(leaf_states),
        "merge": list(merge_states),
        "root": list(root_states),
    }


def _exact_state_feature_vector(
    state: _ExactState,
    *,
    n_regimes: int,
) -> np.ndarray:
    n = int(max(1, n_regimes))
    feat = np.zeros((1 + 2 * n,), dtype=np.float64)
    feat[0] = float(state.count)
    if 0 <= int(state.first) < n:
        feat[1 + int(state.first)] = 1.0
    if 0 <= int(state.last) < n:
        feat[1 + n + int(state.last)] = 1.0
    return feat


def _summary_component_metrics(
    *,
    count_preds: Sequence[float],
    first_preds: Sequence[int],
    last_preds: Sequence[int],
    count_targets: Sequence[int],
    first_targets: Sequence[int],
    last_targets: Sequence[int],
) -> Dict[str, Any]:
    n_examples = int(len(count_targets))
    if n_examples <= 0:
        return {
            "count_mae": float("nan"),
            "count_match_rate": float("nan"),
            "first_accuracy": float("nan"),
            "last_accuracy": float("nan"),
            "exact_summary_match_rate": float("nan"),
            "n_examples": 0,
        }
    count_pred_arr = np.asarray(count_preds, dtype=np.float64)
    first_pred_arr = np.asarray(first_preds, dtype=np.int64)
    last_pred_arr = np.asarray(last_preds, dtype=np.int64)
    count_target_arr = np.asarray(count_targets, dtype=np.int64)
    first_target_arr = np.asarray(first_targets, dtype=np.int64)
    last_target_arr = np.asarray(last_targets, dtype=np.int64)
    count_match = np.asarray(
        np.rint(count_pred_arr),
        dtype=np.int64,
    ) == count_target_arr
    first_match = first_pred_arr == first_target_arr
    last_match = last_pred_arr == last_target_arr
    exact_match = count_match & first_match & last_match
    return {
        "count_mae": float(np.mean(np.abs(count_pred_arr - count_target_arr))),
        "count_match_rate": float(np.mean(count_match.astype(np.float64))),
        "first_accuracy": float(np.mean(first_match.astype(np.float64))),
        "last_accuracy": float(np.mean(last_match.astype(np.float64))),
        "exact_summary_match_rate": float(np.mean(exact_match.astype(np.float64))),
        "n_examples": int(n_examples),
    }


def _identity_exact_summary_metrics(
    states: Sequence[_ExactState],
) -> Dict[str, Any]:
    counts = [int(state.count) for state in states]
    firsts = [int(state.first) for state in states]
    lasts = [int(state.last) for state in states]
    return _summary_component_metrics(
        count_preds=[float(value) for value in counts],
        first_preds=firsts,
        last_preds=lasts,
        count_targets=counts,
        first_targets=firsts,
        last_targets=lasts,
    )


def _augmented_probe_design(x: np.ndarray) -> np.ndarray:
    features = np.asarray(x, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("probe design matrix must be rank-2")
    ones = np.ones((features.shape[0], 1), dtype=np.float64)
    return np.concatenate([features, ones], axis=1)


def _fit_linear_regression_probe(
    x: np.ndarray,
    y: np.ndarray,
) -> Optional[np.ndarray]:
    if x.size == 0 or y.size == 0:
        return None
    design = _augmented_probe_design(x)
    coeffs, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=np.float64), rcond=None)
    return np.asarray(coeffs, dtype=np.float64)


def _predict_linear_regression_probe(
    coeffs: Optional[np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    if coeffs is None or x.size == 0:
        return np.zeros((int(x.shape[0]),), dtype=np.float64)
    design = _augmented_probe_design(x)
    return np.asarray(design @ coeffs, dtype=np.float64)


def _fit_linear_classifier_probe(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
) -> Optional[np.ndarray]:
    if x.size == 0 or y.size == 0:
        return None
    design = _augmented_probe_design(x)
    targets = np.zeros((design.shape[0], int(n_classes)), dtype=np.float64)
    targets[np.arange(design.shape[0]), np.asarray(y, dtype=np.int64)] = 1.0
    coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
    return np.asarray(coeffs, dtype=np.float64)


def _predict_linear_classifier_probe(
    coeffs: Optional[np.ndarray],
    x: np.ndarray,
    *,
    n_classes: int,
) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if coeffs is None:
        return np.zeros((int(x.shape[0]),), dtype=np.int64)
    design = _augmented_probe_design(x)
    logits = np.asarray(design @ coeffs, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != int(n_classes):
        raise ValueError("classifier probe logits have unexpected shape")
    return np.asarray(np.argmax(logits, axis=1), dtype=np.int64)


def _direct_state_prediction_arrays(
    model: FNOCountSketch,
    states: Sequence[torch.Tensor],
    *,
    use_canonical_count: bool,
    use_task_count: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not states:
        empty = np.zeros((0,), dtype=np.float64)
        empty_int = np.zeros((0,), dtype=np.int64)
        return empty, empty_int, empty_int
    state_tensor = torch.stack(list(states), dim=0)
    if bool(use_task_count):
        count_preds_tensor = model.predict_task_count_from_state(state_tensor)
    else:
        count_preds_tensor = (
            model.predict_canonical_count_from_state(state_tensor)
            if bool(use_canonical_count)
            else model.predict_count_from_state(state_tensor)
        )
    _h, first_logits, last_logits = model._split_state(state_tensor)
    return (
        np.asarray(count_preds_tensor.detach().cpu(), dtype=np.float64),
        np.asarray(torch.argmax(first_logits, dim=-1).detach().cpu(), dtype=np.int64),
        np.asarray(torch.argmax(last_logits, dim=-1).detach().cpu(), dtype=np.int64),
    )


def _collect_tree_exact_state_records(
    *,
    model: FNOCountSketch,
    docs: Sequence[ChangepointMarkovDoc],
    fno_docs: Sequence[_FNOCountDoc],
    device: torch.device,
    leaf_tokens: int,
    n_regimes: int,
    retain_phi_features: bool = True,
    retain_exact_features: bool = True,
) -> Dict[str, Dict[str, Any]]:
    levels = ("leaf", "merge", "root")
    records: Dict[str, Dict[str, Any]] = {
        level: {
            "state_features": [],
            "phi_features": [],
            "exact_features": [],
            "doc_index": [],
            "local_index": [],
            "depth": [],
            "count_targets": [],
            "task_targets": [],
            "first_targets": [],
            "last_targets": [],
            "direct_count_preds": [],
            "task_count_preds": [],
            "direct_first_preds": [],
            "direct_last_preds": [],
            "direct_count_entropy": [],
            "direct_count_margin": [],
            "direct_first_entropy": [],
            "direct_first_margin": [],
            "direct_last_entropy": [],
            "direct_last_margin": [],
            "is_first_leaf": [],
            "is_last_leaf": [],
            "merge_join_bit_correct": [],
            "phi_merge_alignment": [],
            "c2_on_range_exact_match": [],
            "merge_consistency_count_abs": [],
            "merge_consistency_first_correct": [],
            "merge_consistency_last_correct": [],
            "phi_label_moments": {},
        }
        for level in levels
    }
    model.eval()
    with torch.no_grad():
        for doc_index, (doc, fno_doc) in enumerate(zip(docs, fno_docs)):
            exact_tree = _balanced_exact_state_tree(
                doc,
                leaf_tokens=int(leaf_tokens),
            )
            leaf_states = [
                model.encode_leaf_tokens(token_ids, device=device)
                for token_ids in list(fno_doc.leaf_token_ids)
            ]
            root_state, merge_states = model._merge_states(
                leaf_states,
                schedule="balanced",
                collect_merge_states=True,
            )
            learned_tree = {
                "leaf": list(leaf_states),
                "merge": list(merge_states),
                "root": [root_state],
            }
            learned_layout = model._balanced_tree_layout(len(leaf_states))
            all_learned_states = list(leaf_states) + list(merge_states)
            all_exact_states = list(exact_tree["leaf"]) + list(exact_tree["merge"])
            children_map = model._balanced_merge_children_map(len(leaf_states))
            for level in levels:
                learned_states = list(learned_tree[level])
                exact_states = list(exact_tree[level])
                if len(learned_states) != len(exact_states):
                    raise ValueError(
                        f"exact/tree state alignment mismatch at level={level!r}: "
                        f"{len(learned_states)} learned vs {len(exact_states)} exact"
                    )
                # One-leaf/full-document regimes have no merge-level states.
                if not learned_states:
                    continue
                state_tensor = torch.stack(list(learned_states), dim=0)
                count_preds_tensor = model.predict_count_from_state(state_tensor)
                task_count_preds_tensor = (
                    model.predict_task_count_from_state(state_tensor)
                    if level == "root"
                    else count_preds_tensor
                )
                _h, first_logits_tensor, last_logits_tensor = model._split_state(
                    state_tensor
                )
                first_preds = np.asarray(
                    torch.argmax(first_logits_tensor, dim=-1).detach().cpu(),
                    dtype=np.int64,
                )
                last_preds = np.asarray(
                    torch.argmax(last_logits_tensor, dim=-1).detach().cpu(),
                    dtype=np.int64,
                )
                count_preds = np.asarray(
                    count_preds_tensor.detach().cpu(),
                    dtype=np.float64,
                )
                task_count_preds = np.asarray(
                    task_count_preds_tensor.detach().cpu(),
                    dtype=np.float64,
                )
                if model.use_markov_summary_spec:
                    count_entropy = np.asarray(
                        model.predict_count_support_entropy_from_state(state_tensor)
                        .detach()
                        .cpu(),
                        dtype=np.float64,
                    )
                    count_margin = np.asarray(
                        model.predict_count_support_margin_from_state(state_tensor)
                        .detach()
                        .cpu(),
                        dtype=np.float64,
                    )
                else:
                    count_entropy = np.full((len(learned_states),), np.nan, dtype=np.float64)
                    count_margin = np.full((len(learned_states),), np.nan, dtype=np.float64)
                first_entropy = np.asarray(
                    PrototypeClassifier.logits_entropy(first_logits_tensor)
                    .detach()
                    .cpu(),
                    dtype=np.float64,
                )
                first_margin = np.asarray(
                    PrototypeClassifier.logits_margin(first_logits_tensor)
                    .detach()
                    .cpu(),
                    dtype=np.float64,
                )
                last_entropy = np.asarray(
                    PrototypeClassifier.logits_entropy(last_logits_tensor)
                    .detach()
                    .cpu(),
                    dtype=np.float64,
                )
                last_margin = np.asarray(
                    PrototypeClassifier.logits_margin(last_logits_tensor)
                    .detach()
                    .cpu(),
                    dtype=np.float64,
                )
                level_records = records[level]
                for local_idx, (
                    state,
                    exact_state,
                    count_pred,
                    task_count_pred,
                    first_pred,
                    last_pred,
                    count_entropy_value,
                    count_margin_value,
                    first_entropy_value,
                    first_margin_value,
                    last_entropy_value,
                    last_margin_value,
                ) in enumerate(
                    zip(
                        learned_states,
                        exact_states,
                        count_preds.tolist(),
                        task_count_preds.tolist(),
                        first_preds.tolist(),
                        last_preds.tolist(),
                        count_entropy.tolist(),
                        count_margin.tolist(),
                        first_entropy.tolist(),
                        first_margin.tolist(),
                        last_entropy.tolist(),
                        last_margin.tolist(),
                    )
                ):
                    level_records["state_features"].append(
                        np.asarray(
                            state.detach().cpu(),
                            dtype=np.float64,
                        )
                    )
                    phi_feature: np.ndarray | None = None
                    if getattr(model, "use_shared_theorem_surface", False):
                        phi_feature = np.asarray(
                            model.predict_phi_from_state(state).detach().cpu(),
                            dtype=np.float64,
                        )
                        if bool(retain_phi_features):
                            level_records["phi_features"].append(phi_feature)
                    if bool(retain_exact_features):
                        level_records["exact_features"].append(
                            _exact_state_feature_vector(
                                exact_state,
                                n_regimes=int(n_regimes),
                            )
                        )
                    level_records["count_targets"].append(int(exact_state.count))
                    task_target_value = float(exact_state.count)
                    if getattr(model, "use_shared_theorem_surface", False):
                        task_target_value = float(
                            model.task_target_from_label(
                                model.theorem_feature_adapter.oracle_label(
                                    count=float(exact_state.count),
                                    first=int(exact_state.first),
                                    last=int(exact_state.last),
                                    metadata=None,
                                )
                            )
                        )
                    level_records["task_targets"].append(task_target_value)
                    level_records["first_targets"].append(int(exact_state.first))
                    level_records["last_targets"].append(int(exact_state.last))
                    if level == "leaf":
                        node_global_idx = int(local_idx)
                    elif level == "merge":
                        node_global_idx = int(len(leaf_states) + int(local_idx))
                    else:
                        node_global_idx = int(learned_layout["root_global_idx"])
                    level_records["doc_index"].append(int(doc_index))
                    level_records["local_index"].append(int(local_idx))
                    level_records["depth"].append(
                        int(
                            learned_layout["depth_by_global_idx"].get(
                                int(node_global_idx),
                                0,
                            )
                        )
                    )
                    if phi_feature is not None:
                        label_key = (
                            int(exact_state.count),
                            int(exact_state.first),
                            int(exact_state.last),
                        )
                        phi_label_moments = cast(
                            Dict[Tuple[int, int, int], Dict[str, Any]],
                            level_records["phi_label_moments"],
                        )
                        stats = phi_label_moments.get(label_key)
                        if stats is None:
                            stats = {
                                "count": 0,
                                "sum": np.zeros_like(phi_feature, dtype=np.float64),
                                "sq_norm_sum": 0.0,
                            }
                            phi_label_moments[label_key] = stats
                        stats["count"] = int(stats["count"]) + 1
                        stats["sum"] = np.asarray(
                            np.asarray(stats["sum"], dtype=np.float64) + phi_feature,
                            dtype=np.float64,
                        )
                        stats["sq_norm_sum"] = float(stats["sq_norm_sum"]) + float(
                            np.dot(phi_feature, phi_feature)
                        )
                    level_records["direct_count_preds"].append(float(count_pred))
                    level_records["task_count_preds"].append(float(task_count_pred))
                    level_records["direct_first_preds"].append(int(first_pred))
                    level_records["direct_last_preds"].append(int(last_pred))
                    level_records["direct_count_entropy"].append(float(count_entropy_value))
                    level_records["direct_count_margin"].append(float(count_margin_value))
                    level_records["direct_first_entropy"].append(float(first_entropy_value))
                    level_records["direct_first_margin"].append(float(first_margin_value))
                    level_records["direct_last_entropy"].append(float(last_entropy_value))
                    level_records["direct_last_margin"].append(float(last_margin_value))
                    level_records["is_first_leaf"].append(
                        bool(level == "leaf" and local_idx == 0)
                    )
                    level_records["is_last_leaf"].append(
                        bool(level == "leaf" and local_idx == (len(learned_states) - 1))
                    )
                    if model.use_markov_summary_spec and model.codec_contract is not None:
                        decoded = model.codec_contract.decode(state)
                        replay_state = model.codec_contract.reencode(decoded)
                        replay_decoded = model.codec_contract.decode(replay_state)
                        level_records["c2_on_range_exact_match"].append(
                            float(
                                int(
                                    int(torch.round(replay_decoded.count).detach().cpu().item())
                                    == int(torch.round(decoded.count).detach().cpu().item())
                                    and int(replay_decoded.first.detach().cpu().item())
                                    == int(decoded.first.detach().cpu().item())
                                    and int(replay_decoded.last.detach().cpu().item())
                                    == int(decoded.last.detach().cpu().item())
                                )
                            )
                        )
            merge_level_records = records["merge"]
            for merge_idx, merge_state in enumerate(merge_states):
                child_indices = children_map.get(int(merge_idx))
                if child_indices is None:
                    continue
                left_idx, right_idx = child_indices
                left_state = all_learned_states[int(left_idx)]
                right_state = all_learned_states[int(right_idx)]
                left_exact = all_exact_states[int(left_idx)]
                right_exact = all_exact_states[int(right_idx)]
                parent_count = float(
                    model.predict_count_from_state(merge_state).detach().cpu().item()
                )
                left_count = float(
                    model.predict_count_from_state(left_state).detach().cpu().item()
                )
                right_count = float(
                    model.predict_count_from_state(right_state).detach().cpu().item()
                )
                _lh, _left_first, left_last = model._split_state(left_state)
                _rh, right_first, _right_last = model._split_state(right_state)
                _ph, parent_first, parent_last = model._split_state(merge_state)
                if model.use_summary_spec:
                    join_prob = float(
                        model.predict_join_prob_from_states(left_state, right_state)
                        .detach()
                        .cpu()
                        .item()
                    )
                    pred_join = int(join_prob >= 0.5)
                else:
                    left_last_probs = torch.softmax(left_last, dim=-1)
                    right_first_probs = torch.softmax(right_first, dim=-1)
                    join_prob = float(
                        1.0
                        - torch.sum(left_last_probs * right_first_probs).detach().cpu().item()
                    )
                    pred_left_last = int(
                        torch.argmax(left_last, dim=-1).detach().cpu().item()
                    )
                    pred_right_first = int(
                        torch.argmax(right_first, dim=-1).detach().cpu().item()
                    )
                    pred_join = int(pred_left_last != pred_right_first)
                pred_parent_first = int(
                    torch.argmax(parent_first, dim=-1).detach().cpu().item()
                )
                pred_parent_last = int(
                    torch.argmax(parent_last, dim=-1).detach().cpu().item()
                )
                pred_left_first = int(
                    torch.argmax(model._split_state(left_state)[1], dim=-1).detach().cpu().item()
                )
                pred_right_last = int(
                    torch.argmax(model._split_state(right_state)[2], dim=-1).detach().cpu().item()
                )
                truth_join = 0 if int(left_exact.last) == int(right_exact.first) else 1
                merge_level_records["merge_join_bit_correct"].append(
                    float(int(pred_join == truth_join))
                )
                if getattr(model, "use_shared_theorem_surface", False):
                    pred_phi = model.predict_phi_parent_from_children(
                        left_state,
                        right_state,
                    ).detach()
                    target_phi = model.predict_phi_from_state(merge_state).detach()
                    merge_level_records["phi_merge_alignment"].append(
                        float(
                            F.cosine_similarity(
                                pred_phi.unsqueeze(0),
                                target_phi.unsqueeze(0),
                                dim=-1,
                            )
                            .cpu()
                            .item()
                        )
                    )
                merge_level_records["merge_consistency_count_abs"].append(
                    abs(float(parent_count) - float(left_count + right_count + join_prob))
                )
                merge_level_records["merge_consistency_first_correct"].append(
                    float(int(pred_parent_first == pred_left_first))
                )
                merge_level_records["merge_consistency_last_correct"].append(
                    float(int(pred_parent_last == pred_right_last))
                )
    return records


def _finalize_tree_exact_state_records(
    records: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    finalized: Dict[str, Dict[str, Any]] = {}
    for level, values in records.items():
        state_features = list(values.get("state_features") or [])
        phi_features = list(values.get("phi_features") or [])
        exact_features = list(values.get("exact_features") or [])
        finalized[str(level)] = {
            "state_features": (
                np.stack(state_features).astype(np.float64)
                if state_features
                else np.zeros((0, 0), dtype=np.float64)
            ),
            "phi_features": (
                np.stack(phi_features).astype(np.float64)
                if phi_features
                else np.zeros((0, 0), dtype=np.float64)
            ),
            "exact_features": (
                np.stack(exact_features).astype(np.float64)
                if exact_features
                else np.zeros((0, 0), dtype=np.float64)
            ),
            "doc_index": np.asarray(
                list(values.get("doc_index") or []),
                dtype=np.int64,
            ),
            "local_index": np.asarray(
                list(values.get("local_index") or []),
                dtype=np.int64,
            ),
            "depth": np.asarray(
                list(values.get("depth") or []),
                dtype=np.int64,
            ),
            "count_targets": np.asarray(
                list(values.get("count_targets") or []),
                dtype=np.int64,
            ),
            "task_targets": np.asarray(
                list(values.get("task_targets") or []),
                dtype=np.float64,
            ),
            "first_targets": np.asarray(
                list(values.get("first_targets") or []),
                dtype=np.int64,
            ),
            "last_targets": np.asarray(
                list(values.get("last_targets") or []),
                dtype=np.int64,
            ),
            "direct_count_preds": np.asarray(
                list(values.get("direct_count_preds") or []),
                dtype=np.float64,
            ),
            "task_count_preds": np.asarray(
                list(values.get("task_count_preds") or []),
                dtype=np.float64,
            ),
            "direct_first_preds": np.asarray(
                list(values.get("direct_first_preds") or []),
                dtype=np.int64,
            ),
            "direct_last_preds": np.asarray(
                list(values.get("direct_last_preds") or []),
                dtype=np.int64,
            ),
            "direct_count_entropy": np.asarray(
                list(values.get("direct_count_entropy") or []),
                dtype=np.float64,
            ),
            "direct_count_margin": np.asarray(
                list(values.get("direct_count_margin") or []),
                dtype=np.float64,
            ),
            "direct_first_entropy": np.asarray(
                list(values.get("direct_first_entropy") or []),
                dtype=np.float64,
            ),
            "direct_first_margin": np.asarray(
                list(values.get("direct_first_margin") or []),
                dtype=np.float64,
            ),
            "direct_last_entropy": np.asarray(
                list(values.get("direct_last_entropy") or []),
                dtype=np.float64,
            ),
            "direct_last_margin": np.asarray(
                list(values.get("direct_last_margin") or []),
                dtype=np.float64,
            ),
            "is_first_leaf": np.asarray(
                list(values.get("is_first_leaf") or []),
                dtype=bool,
            ),
            "is_last_leaf": np.asarray(
                list(values.get("is_last_leaf") or []),
                dtype=bool,
            ),
            "merge_join_bit_correct": np.asarray(
                list(values.get("merge_join_bit_correct") or []),
                dtype=np.float64,
            ),
            "phi_merge_alignment": np.asarray(
                list(values.get("phi_merge_alignment") or []),
                dtype=np.float64,
            ),
            "c2_on_range_exact_match": np.asarray(
                list(values.get("c2_on_range_exact_match") or []),
                dtype=np.float64,
            ),
            "merge_consistency_count_abs": np.asarray(
                list(values.get("merge_consistency_count_abs") or []),
                dtype=np.float64,
            ),
            "merge_consistency_first_correct": np.asarray(
                list(values.get("merge_consistency_first_correct") or []),
                dtype=np.float64,
            ),
            "merge_consistency_last_correct": np.asarray(
                list(values.get("merge_consistency_last_correct") or []),
                dtype=np.float64,
            ),
            "phi_label_moments": {
                tuple(label_key): {
                    "count": int(dict(stats).get("count", 0)),
                    "sum": np.asarray(
                        dict(stats).get("sum", []),
                        dtype=np.float64,
                    ),
                    "sq_norm_sum": float(dict(stats).get("sq_norm_sum", 0.0)),
                }
                for label_key, stats in cast(
                    Mapping[Tuple[int, int, int], Mapping[str, Any]],
                    values.get("phi_label_moments") or {},
                ).items()
            },
        }
    return finalized


def _resolved_diagnostic_detail_mode(config: OPSCountConfig) -> str:
    mode = str(getattr(config, "diagnostic_detail_mode", "summary") or "").strip().lower()
    return "debug_raw" if mode == "debug_raw" else "summary"


def _resolved_raw_diagnostic_artifact_root(
    config: OPSCountConfig,
    *,
    namespace: str,
) -> Path:
    raw_root_text = str(getattr(config, "raw_diagnostic_artifact_dir", "") or "").strip()
    if raw_root_text:
        root = Path(raw_root_text).expanduser()
    else:
        signature_payload = {
            "namespace": str(namespace),
            "prepared_data_signature": str(getattr(config, "prepared_data_signature", "")),
            "seed": int(getattr(config, "seed", 0)),
            "train_docs": int(getattr(config, "train_docs", 0)),
            "fixed_leaf_tokens": int(getattr(config, "fixed_leaf_tokens", 0)),
            "tree_model_version": str(getattr(config, "tree_model_version", "")),
            "tree_task_head_mode": str(getattr(config, "tree_task_head_mode", "")),
        }
        digest = hashlib.sha1(
            json.dumps(signature_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        root = REPO_ROOT / "outputs" / "_raw_diagnostics" / f"{namespace}_{digest}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_tree_exact_split_raw_artifacts(
    *,
    config: OPSCountConfig,
    split: str,
    records: Mapping[str, Mapping[str, Any]],
) -> Dict[str, str]:
    split_root = _resolved_raw_diagnostic_artifact_root(
        config,
        namespace="exact_sketch",
    ) / str(split)
    split_root.mkdir(parents=True, exist_ok=True)
    artifact_paths: Dict[str, str] = {}
    metadata: Dict[str, Any] = {"split": str(split), "levels": {}}
    for level, payload in records.items():
        arrays = {
            str(key): np.asarray(value)
            for key, value in dict(payload).items()
            if not isinstance(value, Mapping)
        }
        npz_path = split_root / f"{level}.npz"
        np.savez_compressed(npz_path, **arrays)
        artifact_paths[str(level)] = str(npz_path)
        metadata["levels"][str(level)] = {
            key: list(np.asarray(value).shape)
            for key, value in arrays.items()
        }
    metadata_path = split_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    artifact_paths["metadata_json"] = str(metadata_path)
    return artifact_paths


def _fit_tree_exact_probe_models(
    *,
    train_records: Mapping[str, Mapping[str, Any]],
    n_regimes: int,
) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
    levels = ("leaf", "merge", "root")
    probe_models: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    for level in levels:
        train_level = dict(train_records.get(level) or {})
        state_features = np.asarray(train_level.get("state_features", []), dtype=np.float64)
        probe_models[level] = {
            "count": _fit_linear_regression_probe(
                state_features,
                np.asarray(train_level.get("count_targets", []), dtype=np.float64),
            ),
            "first": _fit_linear_classifier_probe(
                state_features,
                np.asarray(train_level.get("first_targets", []), dtype=np.int64),
                n_classes=int(n_regimes),
            ),
            "last": _fit_linear_classifier_probe(
                state_features,
                np.asarray(train_level.get("last_targets", []), dtype=np.int64),
                n_classes=int(n_regimes),
            ),
        }
    return probe_models


def _summarize_tree_exact_split(
    *,
    split: str,
    docs: Sequence[ChangepointMarkovDoc],
    split_records_payload: Mapping[str, Mapping[str, Any]],
    probe_models: Mapping[str, Mapping[str, Optional[np.ndarray]]],
    model: FNOCountSketch,
    config: OPSCountConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
    levels = ("leaf", "merge", "root")
    law_metrics = _metrics_payload(
        _eval_exact_family(
            docs,
            leaf_tokens=int(config.fixed_leaf_tokens),
            tau=float(config.violation_tau),
        )
    )
    exact_split: Dict[str, Any] = {"law_metrics": law_metrics}
    tree_split: Dict[str, Any] = {}
    for level in levels:
        level_records = dict(split_records_payload.get(level) or {})
        count_targets = np.asarray(level_records.get("count_targets", []), dtype=np.int64)
        first_targets = np.asarray(level_records.get("first_targets", []), dtype=np.int64)
        last_targets = np.asarray(level_records.get("last_targets", []), dtype=np.int64)
        exact_states = [
            _ExactState(
                count=int(count),
                first=int(first),
                last=int(last),
            )
            for count, first, last in zip(
                count_targets.tolist(),
                first_targets.tolist(),
                last_targets.tolist(),
            )
        ]
        witness_metrics = _identity_exact_summary_metrics(exact_states)
        exact_split[level] = {
            "direct": dict(witness_metrics),
            "probe_control": dict(witness_metrics),
        }
        state_features = np.asarray(level_records.get("state_features", []), dtype=np.float64)
        probe_count_preds = _predict_linear_regression_probe(
            probe_models[level]["count"],
            state_features,
        )
        probe_first_preds = _predict_linear_classifier_probe(
            probe_models[level]["first"],
            state_features,
            n_classes=int(config.n_regimes),
        )
        probe_last_preds = _predict_linear_classifier_probe(
            probe_models[level]["last"],
            state_features,
            n_classes=int(config.n_regimes),
        )
        tree_split[level] = {
            "direct": _summary_component_metrics(
                count_preds=np.asarray(level_records.get("direct_count_preds", []), dtype=np.float64),
                first_preds=np.asarray(level_records.get("direct_first_preds", []), dtype=np.int64),
                last_preds=np.asarray(level_records.get("direct_last_preds", []), dtype=np.int64),
                count_targets=count_targets,
                first_targets=first_targets,
                last_targets=last_targets,
            ),
            "probe": _summary_component_metrics(
                count_preds=probe_count_preds,
                first_preds=probe_first_preds,
                last_preds=probe_last_preds,
                count_targets=count_targets,
                first_targets=first_targets,
                last_targets=last_targets,
            ),
        }
        if level == "merge":
            def _mean(values: Sequence[float]) -> float:
                arr = np.asarray(values, dtype=np.float64)
                return float(np.mean(arr)) if arr.size else float("nan")

            merge_depth = np.asarray(level_records.get("depth", []), dtype=np.int64)
            merge_direct_by_depth: Dict[str, Dict[str, float]] = {}
            if merge_depth.size:
                direct_component_metrics = tree_split[level]["direct"]
                direct_count_preds = np.asarray(
                    level_records.get("direct_count_preds", []),
                    dtype=np.float64,
                )
                direct_first_preds = np.asarray(
                    level_records.get("direct_first_preds", []),
                    dtype=np.int64,
                )
                direct_last_preds = np.asarray(
                    level_records.get("direct_last_preds", []),
                    dtype=np.int64,
                )
                for depth_value in sorted({int(value) for value in merge_depth.tolist()}):
                    depth_mask = merge_depth == int(depth_value)
                    if not np.any(depth_mask):
                        continue
                    merge_direct_by_depth[str(int(depth_value))] = {
                        "exact_summary_match_rate": float(
                            np.mean(
                                (
                                    np.rint(direct_count_preds[depth_mask]).astype(np.int64)
                                    == count_targets[depth_mask]
                                )
                                & (direct_first_preds[depth_mask] == first_targets[depth_mask])
                                & (direct_last_preds[depth_mask] == last_targets[depth_mask])
                            )
                        ),
                        "count_mae": float(
                            np.mean(
                                np.abs(
                                    direct_count_preds[depth_mask]
                                    - count_targets[depth_mask].astype(np.float64)
                                )
                            )
                        ),
                        "first_accuracy": float(
                            np.mean(
                                direct_first_preds[depth_mask] == first_targets[depth_mask]
                            )
                        ),
                        "last_accuracy": float(
                            np.mean(
                                direct_last_preds[depth_mask] == last_targets[depth_mask]
                            )
                        ),
                    }
                if merge_direct_by_depth:
                    tree_split[level]["direct_by_depth"] = merge_direct_by_depth

            tree_split[level]["decoded_consistency"] = {
                "merge_join_bit_accuracy": _mean(level_records.get("merge_join_bit_correct", [])),
                "merge_decoded_consistency_count_mae": _mean(
                    level_records.get("merge_consistency_count_abs", [])
                ),
                "merge_decoded_consistency_first_accuracy": _mean(
                    level_records.get("merge_consistency_first_correct", [])
                ),
                "merge_decoded_consistency_last_accuracy": _mean(
                    level_records.get("merge_consistency_last_correct", [])
                ),
            }

    split_leaf_direct = dict((tree_split.get("leaf") or {}).get("direct") or {})
    split_leaf_probe = dict((tree_split.get("leaf") or {}).get("probe") or {})
    split_merge_direct = dict((tree_split.get("merge") or {}).get("direct") or {})
    split_merge_probe = dict((tree_split.get("merge") or {}).get("probe") or {})
    split_root_direct = dict((tree_split.get("root") or {}).get("direct") or {})
    split_merge_consistency = dict(
        ((tree_split.get("merge") or {}).get("decoded_consistency") or {})
    )
    split_root_records = dict(split_records_payload.get("root") or {})
    split_leaf_records = dict(split_records_payload.get("leaf") or {})
    split_merge_records = dict(split_records_payload.get("merge") or {})
    leaf_state_features = np.asarray(
        split_leaf_records.get("state_features", []),
        dtype=np.float64,
    )
    leaf_doc_indices = np.asarray(
        split_leaf_records.get("doc_index", []),
        dtype=np.int64,
    )
    leaf_local_indices = np.asarray(
        split_leaf_records.get("local_index", []),
        dtype=np.int64,
    )
    leaf_count_targets_arr = np.asarray(
        split_leaf_records.get("count_targets", []),
        dtype=np.int64,
    )
    leaf_first_targets_arr = np.asarray(
        split_leaf_records.get("first_targets", []),
        dtype=np.int64,
    )
    leaf_last_targets_arr = np.asarray(
        split_leaf_records.get("last_targets", []),
        dtype=np.int64,
    )
    leaf_direct_count_preds_arr = np.asarray(
        split_leaf_records.get("direct_count_preds", []),
        dtype=np.float64,
    )
    leaf_direct_first_preds_arr = np.asarray(
        split_leaf_records.get("direct_first_preds", []),
        dtype=np.int64,
    )
    leaf_direct_last_preds_arr = np.asarray(
        split_leaf_records.get("direct_last_preds", []),
        dtype=np.int64,
    )
    merge_depth_arr = np.asarray(
        split_merge_records.get("depth", []),
        dtype=np.int64,
    )
    merge_count_targets_arr = np.asarray(
        split_merge_records.get("count_targets", []),
        dtype=np.int64,
    )
    merge_first_targets_arr = np.asarray(
        split_merge_records.get("first_targets", []),
        dtype=np.int64,
    )
    merge_last_targets_arr = np.asarray(
        split_merge_records.get("last_targets", []),
        dtype=np.int64,
    )
    merge_direct_count_preds_arr = np.asarray(
        split_merge_records.get("direct_count_preds", []),
        dtype=np.float64,
    )
    merge_direct_first_preds_arr = np.asarray(
        split_merge_records.get("direct_first_preds", []),
        dtype=np.int64,
    )
    merge_direct_last_preds_arr = np.asarray(
        split_merge_records.get("direct_last_preds", []),
        dtype=np.int64,
    )
    root_task_preds = np.asarray(split_root_records.get("task_count_preds", []), dtype=np.float64)
    root_targets = np.asarray(
        split_root_records.get("task_targets", split_root_records.get("count_targets", [])),
        dtype=np.float64,
    )
    root_direct_count_mae = float(split_root_direct.get("count_mae", float("nan")))
    task_root_mae = (
        float(np.mean(np.abs(root_task_preds - root_targets)))
        if root_targets.size
        else float("nan")
    )
    model_param = next(model.parameters(), None)
    model_dtype = model_param.dtype if model_param is not None else torch.float32
    model_device = model_param.device if model_param is not None else torch.device("cpu")
    exact_projected_root_abs_errors: List[float] = []
    root_mae_oracle_counts_predicted_endpoints_errors: List[float] = []
    root_mae_predicted_counts_oracle_endpoints_errors: List[float] = []
    use_leaf_doc_grouping = (
        leaf_state_features.ndim == 2
        and leaf_doc_indices.shape[0] == leaf_state_features.shape[0]
        and leaf_local_indices.shape[0] == leaf_state_features.shape[0]
    )
    leaf_state_offset = 0
    for doc_index, doc in enumerate(docs):
        doc_root_count = float(
            _oracle_count(
                doc,
                start=0,
                end=int(len(doc.token_regimes)),
            )
        )
        if use_leaf_doc_grouping:
            doc_leaf_mask = leaf_doc_indices == int(doc_index)
            if not np.any(doc_leaf_mask):
                exact_projected_root_abs_errors.append(abs(doc_root_count))
                continue
            order = np.argsort(leaf_local_indices[doc_leaf_mask], kind="stable")
            doc_leaf_features = leaf_state_features[doc_leaf_mask][order]
            doc_leaf_states = [
                torch.as_tensor(
                    doc_leaf_features[idx],
                    dtype=model_dtype,
                    device=model_device,
                )
                for idx in range(int(doc_leaf_features.shape[0]))
            ]
            doc_pred_counts = leaf_direct_count_preds_arr[doc_leaf_mask][order]
            doc_pred_first = leaf_direct_first_preds_arr[doc_leaf_mask][order]
            doc_pred_last = leaf_direct_last_preds_arr[doc_leaf_mask][order]
            doc_truth_counts = leaf_count_targets_arr[doc_leaf_mask][order]
            doc_truth_first = leaf_first_targets_arr[doc_leaf_mask][order]
            doc_truth_last = leaf_last_targets_arr[doc_leaf_mask][order]
        else:
            doc_leaf_count = len(
                _leaf_spans(
                    int(len(doc.token_regimes)),
                    leaf_tokens=int(config.fixed_leaf_tokens),
                )
            )
            if doc_leaf_count <= 0:
                exact_projected_root_abs_errors.append(abs(doc_root_count))
                continue
            next_offset = int(leaf_state_offset + doc_leaf_count)
            if next_offset > int(leaf_state_features.shape[0]):
                exact_projected_root_abs_errors = []
                root_mae_oracle_counts_predicted_endpoints_errors = []
                root_mae_predicted_counts_oracle_endpoints_errors = []
                break
            doc_leaf_states = [
                torch.as_tensor(
                    leaf_state_features[idx],
                    dtype=model_dtype,
                    device=model_device,
                )
                for idx in range(leaf_state_offset, next_offset)
            ]
            doc_pred_counts = leaf_direct_count_preds_arr[leaf_state_offset:next_offset]
            doc_pred_first = leaf_direct_first_preds_arr[leaf_state_offset:next_offset]
            doc_pred_last = leaf_direct_last_preds_arr[leaf_state_offset:next_offset]
            doc_truth_counts = leaf_count_targets_arr[leaf_state_offset:next_offset]
            doc_truth_first = leaf_first_targets_arr[leaf_state_offset:next_offset]
            doc_truth_last = leaf_last_targets_arr[leaf_state_offset:next_offset]
            leaf_state_offset = next_offset
        exact_projected_root = _exact_projected_root_count_from_states(
            model,
            doc_leaf_states,
            schedule="balanced",
        )
        exact_projected_root_abs_errors.append(
            abs(float(exact_projected_root) - float(doc_root_count))
        )
        if (
            doc_pred_counts.size
            and doc_pred_first.size
            and doc_pred_last.size
            and doc_truth_counts.size
            and doc_truth_first.size
            and doc_truth_last.size
        ):
            root_decomposition = _exact_markov_root_error_decomposition(
                truth_root_count=float(doc_root_count),
                predicted_counts=doc_pred_counts,
                predicted_first=doc_pred_first,
                predicted_last=doc_pred_last,
                truth_counts=doc_truth_counts,
                truth_first=doc_truth_first,
                truth_last=doc_truth_last,
                schedule="balanced",
            )
            root_mae_oracle_counts_predicted_endpoints_errors.append(
                float(root_decomposition["root_mae_oracle_counts_predicted_endpoints"])
            )
            root_mae_predicted_counts_oracle_endpoints_errors.append(
                float(root_decomposition["root_mae_predicted_counts_oracle_endpoints"])
            )
    if (not use_leaf_doc_grouping) and leaf_state_offset != int(leaf_state_features.shape[0]):
        exact_projected_root_abs_errors = []
    exact_projected_root_mae = (
        float(np.mean(np.asarray(exact_projected_root_abs_errors, dtype=np.float64)))
        if exact_projected_root_abs_errors
        else float("nan")
    )
    root_mae_oracle_counts_predicted_endpoints = (
        float(
            np.mean(
                np.asarray(
                    root_mae_oracle_counts_predicted_endpoints_errors,
                    dtype=np.float64,
                )
            )
        )
        if root_mae_oracle_counts_predicted_endpoints_errors
        else float("nan")
    )
    root_mae_predicted_counts_oracle_endpoints = (
        float(
            np.mean(
                np.asarray(
                    root_mae_predicted_counts_oracle_endpoints_errors,
                    dtype=np.float64,
                )
            )
        )
        if root_mae_predicted_counts_oracle_endpoints_errors
        else float("nan")
    )
    task_root_mae_ablation = float(task_root_mae)
    leaf_direct_count_mae = float(split_leaf_direct.get("count_mae", float("nan")))
    leaf_direct_exact_match = float(
        split_leaf_direct.get("exact_summary_match_rate", float("nan"))
    )
    leaf_first_accuracy = float(split_leaf_direct.get("first_accuracy", float("nan")))
    leaf_last_accuracy = float(split_leaf_direct.get("last_accuracy", float("nan")))
    leaf_probe_exact_match = float(
        split_leaf_probe.get("exact_summary_match_rate", float("nan"))
    )
    merge_direct_exact_match = float(
        split_merge_direct.get("exact_summary_match_rate", float("nan"))
    )
    merge_first_accuracy = float(split_merge_direct.get("first_accuracy", float("nan")))
    merge_last_accuracy = float(split_merge_direct.get("last_accuracy", float("nan")))
    merge_probe_exact_match = float(
        split_merge_probe.get("exact_summary_match_rate", float("nan"))
    )
    leaf_count_off_by_k_histogram: Dict[str, float] = {}
    if leaf_direct_count_preds_arr.size and leaf_count_targets_arr.size:
        total_leaf_count = max(1, int(leaf_direct_count_preds_arr.shape[0]))
        leaf_count_diffs = np.abs(
            np.rint(leaf_direct_count_preds_arr).astype(np.int64) - leaf_count_targets_arr
        )
        for diff_value in leaf_count_diffs.tolist():
            key = str(int(diff_value))
            leaf_count_off_by_k_histogram[key] = float(
                leaf_count_off_by_k_histogram.get(key, 0.0) + 1.0
            )
        leaf_count_off_by_k_histogram = {
            str(key): float(float(value) / float(total_leaf_count))
            for key, value in sorted(
                leaf_count_off_by_k_histogram.items(),
                key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]),
            )
        }
    merge_exact_summary_match_rate_by_depth = {
        str(depth): float(metrics.get("exact_summary_match_rate", float("nan")))
        for depth, metrics in dict(
            (tree_split.get("merge") or {}).get("direct_by_depth") or {}
        ).items()
    }
    merge_join_bit_accuracy = float(
        split_merge_consistency.get("merge_join_bit_accuracy", float("nan"))
    )
    c2_on_range_values = np.concatenate(
        [
            np.asarray(
                dict(split_records_payload.get(level) or {}).get(
                    "c2_on_range_exact_match",
                    [],
                ),
                dtype=np.float64,
            )
            for level in levels
        ]
    )
    c2_on_range_exact_match = (
        float(np.mean(c2_on_range_values))
        if c2_on_range_values.size
        else float("nan")
    )
    leaf_first_boundary_mask = np.asarray(
        split_leaf_records.get("is_first_leaf", []),
        dtype=bool,
    )
    leaf_last_boundary_mask = np.asarray(
        split_leaf_records.get("is_last_leaf", []),
        dtype=bool,
    )
    leaf_first_boundary_targets = np.asarray(
        split_leaf_records.get("first_targets", []),
        dtype=np.int64,
    )
    leaf_last_boundary_targets = np.asarray(
        split_leaf_records.get("last_targets", []),
        dtype=np.int64,
    )
    leaf_direct_first_preds = np.asarray(
        split_leaf_records.get("direct_first_preds", []),
        dtype=np.int64,
    )
    leaf_direct_last_preds = np.asarray(
        split_leaf_records.get("direct_last_preds", []),
        dtype=np.int64,
    )
    first_leaf_direct_accuracy = (
        float(
            np.mean(
                (
                    leaf_direct_first_preds[leaf_first_boundary_mask]
                    == leaf_first_boundary_targets[leaf_first_boundary_mask]
                ).astype(np.float64)
            )
        )
        if leaf_first_boundary_mask.size and np.any(leaf_first_boundary_mask)
        else float("nan")
    )
    last_leaf_direct_accuracy = (
        float(
            np.mean(
                (
                    leaf_direct_last_preds[leaf_last_boundary_mask]
                    == leaf_last_boundary_targets[leaf_last_boundary_mask]
                ).astype(np.float64)
            )
        )
        if leaf_last_boundary_mask.size and np.any(leaf_last_boundary_mask)
        else float("nan")
    )
    split_phi_merge_alignment_values = np.asarray(
        split_merge_records.get("phi_merge_alignment", []),
        dtype=np.float64,
    )
    split_phi_merge_alignment = (
        float(np.mean(split_phi_merge_alignment_values))
        if split_phi_merge_alignment_values.size
        else float("nan")
    )
    split_phi_moments: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for level in levels:
        level_payload = dict(split_records_payload.get(level) or {})
        level_phi_moments = cast(
            Mapping[Tuple[int, int, int], Mapping[str, Any]],
            level_payload.get("phi_label_moments") or {},
        )
        if level_phi_moments:
            for label_key, stats in level_phi_moments.items():
                label_count = int(dict(stats).get("count", 0))
                if label_count <= 0:
                    continue
                label_sum = np.asarray(dict(stats).get("sum", []), dtype=np.float64)
                label_sq_norm_sum = float(dict(stats).get("sq_norm_sum", 0.0))
                existing = split_phi_moments.get(tuple(label_key))
                if existing is None:
                    split_phi_moments[tuple(label_key)] = {
                        "count": label_count,
                        "sum": label_sum,
                        "sq_norm_sum": label_sq_norm_sum,
                    }
                else:
                    existing["count"] = int(existing["count"]) + label_count
                    existing["sum"] = np.asarray(
                        np.asarray(existing["sum"], dtype=np.float64) + label_sum,
                        dtype=np.float64,
                    )
                    existing["sq_norm_sum"] = float(existing["sq_norm_sum"]) + label_sq_norm_sum
            continue
        phi_features = np.asarray(level_payload.get("phi_features", []), dtype=np.float64)
        count_targets_arr = np.asarray(level_payload.get("count_targets", []), dtype=np.int64)
        first_targets_arr = np.asarray(level_payload.get("first_targets", []), dtype=np.int64)
        last_targets_arr = np.asarray(level_payload.get("last_targets", []), dtype=np.int64)
        if phi_features.ndim != 2 or not phi_features.size:
            continue
        for feature_vec, count_value, first_value, last_value in zip(
            phi_features,
            count_targets_arr.tolist(),
            first_targets_arr.tolist(),
            last_targets_arr.tolist(),
        ):
            key = (int(count_value), int(first_value), int(last_value))
            feature_arr = np.asarray(feature_vec, dtype=np.float64)
            existing = split_phi_moments.get(key)
            if existing is None:
                split_phi_moments[key] = {
                    "count": 1,
                    "sum": feature_arr,
                    "sq_norm_sum": float(np.dot(feature_arr, feature_arr)),
                }
            else:
                existing["count"] = int(existing["count"]) + 1
                existing["sum"] = np.asarray(
                    np.asarray(existing["sum"], dtype=np.float64) + feature_arr,
                    dtype=np.float64,
                )
                existing["sq_norm_sum"] = float(existing["sq_norm_sum"]) + float(
                    np.dot(feature_arr, feature_arr)
                )
    split_phi_centroids: List[np.ndarray] = []
    split_phi_within_terms: List[float] = []
    for stats in split_phi_moments.values():
        label_count = int(dict(stats).get("count", 0))
        if label_count <= 0:
            continue
        centroid = np.asarray(dict(stats).get("sum", []), dtype=np.float64) / float(label_count)
        sq_norm_sum = float(dict(stats).get("sq_norm_sum", 0.0))
        split_phi_centroids.append(centroid)
        split_phi_within_terms.append(
            float(
                max(
                    0.0,
                    (sq_norm_sum / float(label_count)) - float(np.dot(centroid, centroid)),
                )
            )
        )
    split_phi_within_class_variance = (
        float(np.mean(np.asarray(split_phi_within_terms, dtype=np.float64)))
        if split_phi_within_terms
        else float("nan")
    )
    split_phi_between_terms: List[float] = []
    for i in range(len(split_phi_centroids)):
        for j in range(i + 1, len(split_phi_centroids)):
            split_phi_between_terms.append(
                float(np.linalg.norm(split_phi_centroids[i] - split_phi_centroids[j]))
            )
    split_phi_between_class_margin = (
        float(np.mean(np.asarray(split_phi_between_terms, dtype=np.float64)))
        if split_phi_between_terms
        else float("nan")
    )

    def _mean_field(level_payload: Mapping[str, Any], field: str) -> float:
        values = np.asarray(level_payload.get(field, []), dtype=np.float64)
        return float(np.mean(values)) if values.size else float("nan")

    leaf_codec_selection_value = (
        float(leaf_direct_count_mae)
        + float(max(0.0, 1.0 - leaf_direct_exact_match))
        + float(max(0.0, 1.0 - merge_join_bit_accuracy))
        + float(max(0.0, 1.0 - c2_on_range_exact_match))
        if np.isfinite(leaf_direct_count_mae)
        and np.isfinite(leaf_direct_exact_match)
        and np.isfinite(merge_join_bit_accuracy)
        and np.isfinite(c2_on_range_exact_match)
        else float("nan")
    )
    theorem_bootstrap_selection_value = (
        float(leaf_direct_count_mae)
        + float(max(0.0, 1.0 - leaf_direct_exact_match))
        + float(max(0.0, 1.0 - merge_direct_exact_match))
        + float(max(0.0, 1.0 - merge_join_bit_accuracy))
        + float(max(0.0, 1.0 - c2_on_range_exact_match))
        if np.isfinite(leaf_direct_count_mae)
        and np.isfinite(leaf_direct_exact_match)
        and np.isfinite(merge_direct_exact_match)
        and np.isfinite(merge_join_bit_accuracy)
        and np.isfinite(c2_on_range_exact_match)
        else float("nan")
    )
    selection_root_mae = (
        float(exact_projected_root_mae)
        if bool(getattr(model, "exact_projected_merge_is_runtime_merge", False))
        and np.isfinite(exact_projected_root_mae)
        else float(root_direct_count_mae)
    )
    direct_selection_value = (
        float(selection_root_mae)
        + float(max(0.0, 1.0 - leaf_direct_exact_match))
        + float(max(0.0, 1.0 - merge_direct_exact_match))
        + float(max(0.0, 1.0 - merge_join_bit_accuracy))
        if np.isfinite(selection_root_mae)
        and np.isfinite(leaf_direct_exact_match)
        and np.isfinite(merge_direct_exact_match)
        and np.isfinite(merge_join_bit_accuracy)
        else float("nan")
    )
    count_support_size = (
        int(model.theorem_count_support_size())
        if getattr(model, "use_markov_summary_spec", False)
        else 0
    )
    direct_selection = {
        "root_direct_count_mae": root_direct_count_mae,
        "exact_projected_root_mae": float(exact_projected_root_mae),
        "certified_projected_root_mae": float(exact_projected_root_mae),
        "root_mae_predicted_counts_predicted_endpoints": float(
            exact_projected_root_mae
        ),
        "root_mae_oracle_counts_predicted_endpoints": float(
            root_mae_oracle_counts_predicted_endpoints
        ),
        "root_mae_predicted_counts_oracle_endpoints": float(
            root_mae_predicted_counts_oracle_endpoints
        ),
        "learned_merger_gap": (
            float(root_direct_count_mae - exact_projected_root_mae)
            if np.isfinite(root_direct_count_mae)
            and np.isfinite(exact_projected_root_mae)
            else float("nan")
        ),
        "task_root_mae": task_root_mae,
        "task_root_mae_ablation": task_root_mae_ablation,
        "leaf_direct_count_mae": leaf_direct_count_mae,
        "leaf_direct_exact_match": leaf_direct_exact_match,
        "leaf_first_accuracy": leaf_first_accuracy,
        "leaf_last_accuracy": leaf_last_accuracy,
        "leaf_probe_exact_match": leaf_probe_exact_match,
        "leaf_count_off_by_k_histogram": {
            str(key): float(value)
            for key, value in leaf_count_off_by_k_histogram.items()
        },
        "leaf_direct_probe_exact_gap": (
            float(max(0.0, leaf_probe_exact_match - leaf_direct_exact_match))
            if np.isfinite(leaf_probe_exact_match)
            and np.isfinite(leaf_direct_exact_match)
            else float("nan")
        ),
        "phi_direct_probe_leaf_gap": (
            float(max(0.0, leaf_probe_exact_match - leaf_direct_exact_match))
            if np.isfinite(leaf_probe_exact_match)
            and np.isfinite(leaf_direct_exact_match)
            else float("nan")
        ),
        "merge_direct_exact_match": merge_direct_exact_match,
        "merge_first_accuracy": merge_first_accuracy,
        "merge_last_accuracy": merge_last_accuracy,
        "merge_exact_summary_match_rate_by_depth": {
            str(key): float(value)
            for key, value in merge_exact_summary_match_rate_by_depth.items()
        },
        "merge_probe_exact_match": merge_probe_exact_match,
        "merge_direct_probe_exact_gap": (
            float(max(0.0, merge_probe_exact_match - merge_direct_exact_match))
            if np.isfinite(merge_probe_exact_match)
            and np.isfinite(merge_direct_exact_match)
            else float("nan")
        ),
        "phi_direct_probe_merge_gap": (
            float(max(0.0, merge_probe_exact_match - merge_direct_exact_match))
            if np.isfinite(merge_probe_exact_match)
            and np.isfinite(merge_direct_exact_match)
            else float("nan")
        ),
        "merge_join_bit_accuracy": merge_join_bit_accuracy,
        "c2_on_range_exact_match": c2_on_range_exact_match,
        "first_leaf_direct_accuracy": first_leaf_direct_accuracy,
        "last_leaf_direct_accuracy": last_leaf_direct_accuracy,
        "leaf_count_head_entropy_mean": _mean_field(
            split_leaf_records, "direct_count_entropy"
        ),
        "leaf_count_head_margin_mean": _mean_field(
            split_leaf_records, "direct_count_margin"
        ),
        "leaf_first_head_entropy_mean": _mean_field(
            split_leaf_records, "direct_first_entropy"
        ),
        "leaf_first_head_margin_mean": _mean_field(
            split_leaf_records, "direct_first_margin"
        ),
        "leaf_last_head_entropy_mean": _mean_field(
            split_leaf_records, "direct_last_entropy"
        ),
        "leaf_last_head_margin_mean": _mean_field(
            split_leaf_records, "direct_last_margin"
        ),
        "merge_count_head_entropy_mean": _mean_field(
            split_merge_records, "direct_count_entropy"
        ),
        "merge_count_head_margin_mean": _mean_field(
            split_merge_records, "direct_count_margin"
        ),
        "merge_first_head_entropy_mean": _mean_field(
            split_merge_records, "direct_first_entropy"
        ),
        "merge_first_head_margin_mean": _mean_field(
            split_merge_records, "direct_first_margin"
        ),
        "merge_last_head_entropy_mean": _mean_field(
            split_merge_records, "direct_last_entropy"
        ),
        "merge_last_head_margin_mean": _mean_field(
            split_merge_records, "direct_last_margin"
        ),
        "phi_merge_alignment": split_phi_merge_alignment,
        "phi_within_class_variance": split_phi_within_class_variance,
        "phi_between_class_margin": split_phi_between_class_margin,
        "phi_pair_same_accuracy": float("nan"),
        "phi_pair_diff_accuracy": float("nan"),
        "phi_pair_auc": float("nan"),
        "phi_replay_same_class_rate": float("nan"),
        "task_factorization_gap": float("nan"),
        "count_head_support_size": float(count_support_size),
        "val_leaf_codec_direct": float(leaf_codec_selection_value),
        "val_theorem_bootstrap_direct": float(theorem_bootstrap_selection_value),
        "val_exact_sketch_direct": float(direct_selection_value),
        "exact_sketch_direct_selection_value": float(direct_selection_value),
        "tree_model_version": str(getattr(model, "tree_model_version", "")),
        "tree_runtime_merge_kind": str(getattr(model, "runtime_merge_kind", "")),
        "tree_exact_projected_merge_is_runtime_merge": bool(
            getattr(model, "exact_projected_merge_is_runtime_merge", False)
        ),
        "uses_unified_g_learned_merge": bool(
            getattr(model, "uses_unified_g_learned_merge", False)
        ),
    }
    return exact_split, tree_split, direct_selection


def _tree_exact_sketch_diagnostics(
    *,
    model: FNOCountSketch,
    config: OPSCountConfig,
    device: torch.device,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
    train_fno_docs: Sequence[_FNOCountDoc],
    val_fno_docs: Sequence[_FNOCountDoc],
    test_fno_docs: Sequence[_FNOCountDoc],
) -> Dict[str, Any]:
    split_docs = {
        "train": tuple(train_docs),
        "val": tuple(val_docs),
        "test": tuple(test_docs),
    }
    split_fno_docs = {
        "train": tuple(train_fno_docs),
        "val": tuple(val_fno_docs),
        "test": tuple(test_fno_docs),
    }
    detail_mode = _resolved_diagnostic_detail_mode(config)
    raw_artifacts: Dict[str, Any] = {}
    train_records = _finalize_tree_exact_state_records(
        _collect_tree_exact_state_records(
            model=model,
            docs=split_docs["train"],
            fno_docs=split_fno_docs["train"],
            device=device,
            leaf_tokens=int(config.fixed_leaf_tokens),
            n_regimes=int(config.n_regimes),
            retain_phi_features=bool(detail_mode == "debug_raw"),
            retain_exact_features=bool(detail_mode == "debug_raw"),
        )
    )
    if detail_mode == "debug_raw":
        raw_artifacts["train"] = _write_tree_exact_split_raw_artifacts(
            config=config,
            split="train",
            records=train_records,
        )
    probe_models = _fit_tree_exact_probe_models(
        train_records=train_records,
        n_regimes=int(config.n_regimes),
    )
    exact_witness: Dict[str, Any] = {}
    tree_neural: Dict[str, Any] = {}
    direct_selection_metrics: Dict[str, Dict[str, float]] = {}
    exact_witness["train"], tree_neural["train"], direct_selection_metrics["train"] = (
        _summarize_tree_exact_split(
            split="train",
            docs=split_docs["train"],
            split_records_payload=train_records,
            probe_models=probe_models,
            model=model,
            config=config,
        )
    )
    del train_records
    _trim_host_allocator()

    for split in ("val", "test"):
        split_records = _finalize_tree_exact_state_records(
            _collect_tree_exact_state_records(
                model=model,
                docs=split_docs[split],
                fno_docs=split_fno_docs[split],
                device=device,
                leaf_tokens=int(config.fixed_leaf_tokens),
                n_regimes=int(config.n_regimes),
                retain_phi_features=bool(detail_mode == "debug_raw"),
                retain_exact_features=bool(detail_mode == "debug_raw"),
            )
        )
        if detail_mode == "debug_raw":
            raw_artifacts[split] = _write_tree_exact_split_raw_artifacts(
                config=config,
                split=split,
                records=split_records,
            )
        exact_witness[split], tree_neural[split], direct_selection_metrics[split] = (
            _summarize_tree_exact_split(
                split=split,
                docs=split_docs[split],
                split_records_payload=split_records,
                probe_models=probe_models,
                model=model,
                config=config,
            )
        )
        del split_records
        _trim_host_allocator()

    test_tree = dict(tree_neural.get("test") or {})
    leaf_probe_match = float(
        ((test_tree.get("leaf") or {}).get("probe") or {}).get(
            "exact_summary_match_rate",
            float("nan"),
        )
    )
    merge_probe_match = float(
        ((test_tree.get("merge") or {}).get("probe") or {}).get(
            "exact_summary_match_rate",
            float("nan"),
        )
    )
    leaf_direct_match = float(
        ((test_tree.get("leaf") or {}).get("direct") or {}).get(
            "exact_summary_match_rate",
            float("nan"),
        )
    )
    merge_direct_match = float(
        ((test_tree.get("merge") or {}).get("direct") or {}).get(
            "exact_summary_match_rate",
            float("nan"),
        )
    )
    leaf_direct_first_accuracy = float(
        ((test_tree.get("leaf") or {}).get("direct") or {}).get(
            "first_accuracy",
            float("nan"),
        )
    )
    leaf_direct_last_accuracy = float(
        ((test_tree.get("leaf") or {}).get("direct") or {}).get(
            "last_accuracy",
            float("nan"),
        )
    )
    root_probe_count_mae = float(
        ((test_tree.get("root") or {}).get("probe") or {}).get(
            "count_mae",
            float("nan"),
        )
    )
    root_direct_count_mae = float(
        ((test_tree.get("root") or {}).get("direct") or {}).get(
            "count_mae",
            float("nan"),
        )
    )
    leaf_direct_count_mae = float(
        ((test_tree.get("leaf") or {}).get("direct") or {}).get(
            "count_mae",
            float("nan"),
        )
    )
    merge_direct_count_mae = float(
        ((test_tree.get("merge") or {}).get("direct") or {}).get(
            "count_mae",
            float("nan"),
        )
    )
    merge_direct_first_accuracy = float(
        ((test_tree.get("merge") or {}).get("direct") or {}).get(
            "first_accuracy",
            float("nan"),
        )
    )
    merge_direct_last_accuracy = float(
        ((test_tree.get("merge") or {}).get("direct") or {}).get(
            "last_accuracy",
            float("nan"),
        )
    )
    test_direct_metrics = dict(direct_selection_metrics.get("test") or {})
    leaf_count_head_entropy_mean = float(
        test_direct_metrics.get("leaf_count_head_entropy_mean", float("nan"))
    )
    merge_count_head_entropy_mean = float(
        test_direct_metrics.get("merge_count_head_entropy_mean", float("nan"))
    )
    phi_merge_alignment = float(
        test_direct_metrics.get("phi_merge_alignment", float("nan"))
    )
    phi_within_class_variance = float(
        test_direct_metrics.get("phi_within_class_variance", float("nan"))
    )
    phi_between_class_margin = float(
        test_direct_metrics.get("phi_between_class_margin", float("nan"))
    )
    count_support_size = int(
        round(float(test_direct_metrics.get("count_head_support_size", 0.0)))
    )
    entropy_normalizer = (
        float(np.log(float(count_support_size)))
        if int(count_support_size) > 1
        else float("nan")
    )
    normalized_leaf_count_entropy = (
        float(leaf_count_head_entropy_mean / entropy_normalizer)
        if np.isfinite(leaf_count_head_entropy_mean) and np.isfinite(entropy_normalizer)
        else float("nan")
    )
    normalized_merge_count_entropy = (
        float(merge_count_head_entropy_mean / entropy_normalizer)
        if np.isfinite(merge_count_head_entropy_mean) and np.isfinite(entropy_normalizer)
        else float("nan")
    )
    leaf_count_entropy_penalty = (
        float(max(0.0, normalized_leaf_count_entropy))
        if np.isfinite(normalized_leaf_count_entropy)
        else 0.0
    )
    merge_count_entropy_penalty = (
        float(max(0.0, normalized_merge_count_entropy))
        if np.isfinite(normalized_merge_count_entropy)
        else 0.0
    )
    leaf_boundary_gap_score = (
        float(max(0.0, leaf_probe_match - leaf_direct_match))
        + 0.5 * float(max(0.0, 1.0 - leaf_direct_first_accuracy))
        + 0.5 * float(max(0.0, 1.0 - leaf_direct_last_accuracy))
        if np.isfinite(leaf_probe_match)
        and np.isfinite(leaf_direct_match)
        and np.isfinite(leaf_direct_first_accuracy)
        and np.isfinite(leaf_direct_last_accuracy)
        else float("nan")
    )
    count_composition_gap_score = (
        float(max(0.0, leaf_direct_match - merge_direct_match))
        if np.isfinite(leaf_direct_match) and np.isfinite(merge_direct_match)
        else float("nan")
    )
    readout_gap_score = (
        float(max(0.0, root_direct_count_mae - root_probe_count_mae))
        if np.isfinite(root_direct_count_mae) and np.isfinite(root_probe_count_mae)
        else float("nan")
    )
    subtree_label_value_gap_score = (
        float(max(0.0, count_composition_gap_score - readout_gap_score))
        if np.isfinite(count_composition_gap_score) and np.isfinite(readout_gap_score)
        else float("nan")
    )
    theorem_count_decode_gap_score = (
        float(max(0.0, leaf_probe_match - leaf_direct_match))
        + float(max(0.0, merge_probe_match - merge_direct_match))
        + float(max(0.0, root_direct_count_mae))
        + float(leaf_count_entropy_penalty)
        + float(merge_count_entropy_penalty)
        if np.isfinite(leaf_probe_match)
        and np.isfinite(leaf_direct_match)
        and np.isfinite(merge_probe_match)
        and np.isfinite(merge_direct_match)
        and np.isfinite(root_direct_count_mae)
        else float("nan")
    )
    phi_not_sufficient_score = (
        float(max(0.0, leaf_probe_match - leaf_direct_match))
        + float(max(0.0, merge_probe_match - merge_direct_match))
        + float(max(0.0, phi_within_class_variance))
        + float(max(0.0, 1.0 - phi_between_class_margin))
        if np.isfinite(leaf_probe_match)
        and np.isfinite(leaf_direct_match)
        and np.isfinite(merge_probe_match)
        and np.isfinite(merge_direct_match)
        and np.isfinite(phi_within_class_variance)
        and np.isfinite(phi_between_class_margin)
        else float("nan")
    )
    phi_not_compositional_score = (
        float(max(0.0, 1.0 - phi_merge_alignment))
        + float(max(0.0, count_composition_gap_score))
        if np.isfinite(phi_merge_alignment) and np.isfinite(count_composition_gap_score)
        else float("nan")
    )
    high_boundary_accuracy = bool(
        np.isfinite(leaf_direct_first_accuracy)
        and np.isfinite(leaf_direct_last_accuracy)
        and np.isfinite(merge_direct_first_accuracy)
        and np.isfinite(merge_direct_last_accuracy)
        and float(leaf_direct_first_accuracy) >= 0.95
        and float(leaf_direct_last_accuracy) >= 0.95
        and float(merge_direct_first_accuracy) >= 0.95
        and float(merge_direct_last_accuracy) >= 0.95
    )
    count_decode_is_bad = bool(
        np.isfinite(root_direct_count_mae)
        and np.isfinite(leaf_direct_count_mae)
        and np.isfinite(merge_direct_count_mae)
        and (
            float(root_direct_count_mae) > 0.25
            or float(leaf_direct_count_mae) > 0.5
            or float(merge_direct_count_mae) > 0.5
        )
    )
    count_head_is_near_uniform = bool(
        np.isfinite(normalized_leaf_count_entropy)
        and np.isfinite(normalized_merge_count_entropy)
        and max(
            float(normalized_leaf_count_entropy),
            float(normalized_merge_count_entropy),
        )
        >= 0.9
    )
    score_pairs = [
        ("phi_not_sufficient", phi_not_sufficient_score),
        ("phi_not_compositional", phi_not_compositional_score),
        ("leaf_boundary_encoding_gap", leaf_boundary_gap_score),
        ("theorem_count_decode_gap", theorem_count_decode_gap_score),
        ("count_composition_gap", count_composition_gap_score),
        ("subtree_label_value_gap", subtree_label_value_gap_score),
        ("legacy_readout_gap", readout_gap_score),
    ]
    finite_pairs = [
        (name, float(score))
        for name, score in score_pairs
        if np.isfinite(float(score))
    ]
    if high_boundary_accuracy and count_decode_is_bad and count_head_is_near_uniform:
        failure_bucket = "theorem_count_decode_gap"
    else:
        failure_bucket = (
            max(finite_pairs, key=lambda item: item[1])[0]
            if finite_pairs
            else "insufficient_data"
        )
    payload = {
        "summary_fields": ["count", "first", "last"],
        "codec_fields": ["count", "first", "last", "join"],
        "paper_to_lean_local_law_mapping": dict(PAPER_TO_LEAN_LOCAL_LAW_MAPPING),
        "theorem_contract": {
            "summary_ref": LEAN_MARKOV_COUNT_SKETCH_REF,
            "codec_ref": LEAN_SKETCH_CODEC_EXACT_ASSUMPTIONS_REF,
            "bundle_ref": LEAN_APPROX_BUNDLE_OF_NODEWISE_REF,
            "observed_token_recoverability": _markov_observed_token_recoverability_contract(
                config=config,
            ),
            "runtime_markov_c2_note": (
                "Markov C2 count-drift diagnostics live in the runtime_markov_c2 payload "
                "section; C2_to_L3 here is reserved for the stricter on-range reencode "
                "exactness witness."
            ),
            "simulation_validation_refs": {
                "exact_state_support_ref": LEAN_MARKOV_PATH_SUPPORT_EXACT_REF,
                "changepoint_count_support_ref": LEAN_MARKOV_PATH_COUNT_SUPPORT_EXACT_REF,
                "count_only_invalid_ref": LEAN_MARKOV_COUNT_ONLY_INVALID_REF,
                "sufficiency_decoder_ref": LEAN_MARKOV_SUFFICIENCY_DECODER_REF,
                "runtime_audit_stochastic_approx_ref": (
                    LEAN_MARKOV_RUNTIME_AUDIT_STOCHASTIC_APPROX_REF
                ),
                "representation_exact_pass_ref": (
                    LEAN_MARKOV_REPRESENTATION_EXACT_PASS_REF
                ),
                "representation_zero_root_count_error_ref": (
                    LEAN_MARKOV_REPRESENTATION_ZERO_ROOT_COUNT_ERROR_REF
                ),
                "representation_count_transport_ref": (
                    LEAN_MARKOV_REPRESENTATION_COUNT_TRANSPORT_REF
                ),
            },
            "markov_sufficiency_note": (
                "For the Markov changepoint task, recoverability of the exact sketch "
                "`(count, first, last)` is treated as a sufficiency witness: a summary "
                "that is sufficient for all two-sided changepoint-count queries admits "
                "a decoder back to an equivalent exact sketch."
            ),
            "representation_learnability_note": (
                "When the learned representation exactly recovers the theorem-domain "
                "Markov sketch, Lean reduces the study target to exact query "
                "sufficiency plus zero root count error; approximate exact-sketch "
                "error upper-bounds count error through a discrete transport bound."
            ),
            "named_sections": {
                "C1_to_L1": "leaf exact sketch",
                "C3_to_L2": "merge exact sketch",
                "C2_to_L3": "on-range reencode exactness",
            },
        },
        "schedule_proxy_status": "PROXY_ONLY",
        "schedule_proxy_note": (
            "schedule_consistency / schedule spread remain proxy-only diagnostics and are "
            "excluded from theorem-facing exact-sketch totals."
        ),
        "diagnostic_detail_mode": str(detail_mode),
        "exact_witness": exact_witness,
        "tree_neural": tree_neural,
        "aligned_sketch_surface": str(config.aligned_sketch_surface),
        "internal_supervision_kind": str(config.internal_supervision_kind),
        "internal_label_rate": float(config.internal_label_rate),
        "leaf_exact_supervision": bool(config.leaf_exact_supervision),
        "summary_spec_name": str(getattr(config, "summary_spec_name", "")),
        "slot_count": int(getattr(config, "slot_count", 0)),
        "leaf_label_rate": float(getattr(config, "leaf_label_rate", 1.0)),
        "tree_document_loss_normalization_mode": str(
            getattr(config, "tree_document_loss_normalization_mode", "auto")
        ),
        "tree_checkpoint_metric": str(getattr(config, "tree_checkpoint_metric", "")),
        "tree_stage1_checkpoint_metric": str(
            getattr(config, "tree_stage1_checkpoint_metric", "")
        ),
        "tree_stage1_eval_mode": str(getattr(config, "tree_stage1_eval_mode", "")),
        "tree_stage1_screen_doc_limit": int(
            getattr(config, "tree_stage1_screen_doc_limit", 0)
        ),
        "tree_stage1_final_exact_doc_limit": int(
            getattr(config, "tree_stage1_final_exact_doc_limit", 0)
        ),
        "exact_metric_selection_doc_limit": int(
            getattr(config, "exact_metric_selection_doc_limit", 0)
        ),
        "exact_metric_selection_interval": int(
            getattr(config, "exact_metric_selection_interval", 1)
        ),
        "exact_metric_final_doc_limit": int(
            getattr(config, "exact_metric_final_doc_limit", 0)
        ),
        "tree_posttrain_train_doc_limit": int(
            getattr(config, "tree_posttrain_train_doc_limit", 0)
        ),
        "posttrain_diagnostics_mode": str(
            getattr(config, "posttrain_diagnostics_mode", "")
        ),
        "tree_batch_pack_mode": str(getattr(config, "tree_batch_pack_mode", "")),
        "tree_batch_token_budget": int(
            getattr(config, "tree_batch_token_budget", 0)
        ),
        "tree_batch_node_budget": int(
            getattr(config, "tree_batch_node_budget", 0)
        ),
        "tree_batch_autotune": bool(
            getattr(config, "tree_batch_autotune", False)
        ),
        "tree_eval_workers_per_mig": int(
            getattr(config, "tree_eval_workers_per_mig", 0)
        ),
        "tree_stage1_artifact_dir": str(
            getattr(config, "tree_stage1_artifact_dir", "")
        ),
        "tree_stage1_root_weight": float(
            getattr(config, "tree_stage1_root_weight", 0.0)
        ),
        "tree_join_bit_weight": float(getattr(config, "tree_join_bit_weight", 0.0)),
        "tree_training_schedule": str(getattr(config, "tree_training_schedule", "")),
        "tree_stage1_epochs": int(getattr(config, "tree_stage1_epochs", 0)),
        "tree_stage2_epochs": int(getattr(config, "tree_stage2_epochs", 0)),
        "tree_task_head_mode": str(getattr(config, "tree_task_head_mode", "")),
        "tree_theorem_surface_mode": str(
            getattr(config, "tree_theorem_surface_mode", "")
        ),
        "tree_theorem_count_head_mode": str(
            getattr(config, "tree_theorem_count_head_mode", "")
        ),
        "tree_theorem_feature_dim": int(
            getattr(config, "tree_theorem_feature_dim", 0)
        ),
        "tree_theorem_feature_hidden_dim": int(
            getattr(config, "tree_theorem_feature_hidden_dim", 0)
        ),
        "tree_phi_compose_weight": float(
            getattr(config, "tree_phi_compose_weight", 0.0)
        ),
        "tree_phi_contrastive_weight": float(
            getattr(config, "tree_phi_contrastive_weight", 0.0)
        ),
        "tree_phi_alignment_loss": str(
            getattr(config, "tree_phi_alignment_loss", "")
        ),
        "tree_c2_mode": str(getattr(config, "tree_c2_mode", "") or ""),
        "tree_theorem_count_ordinal_weight": float(
            getattr(config, "tree_theorem_count_ordinal_weight", 1.0)
        ),
        "tree_theorem_count_scalar_aux_weight": float(
            getattr(config, "tree_theorem_count_scalar_aux_weight", 0.25)
        ),
        "tree_theorem_count_threshold_balance": bool(
            getattr(config, "tree_theorem_count_threshold_balance", True)
        ),
        "tree_summary_spec_root_mode": str(
            getattr(config, "tree_summary_spec_root_mode", "")
        ),
        "phi_merge_alignment": float(phi_merge_alignment),
        "phi_within_class_variance": float(phi_within_class_variance),
        "phi_between_class_margin": float(phi_between_class_margin),
        "phi_not_sufficient_score": float(phi_not_sufficient_score),
        "phi_not_compositional_score": float(phi_not_compositional_score),
        "tree_theorem_count_dim": int(
            getattr(config, "tree_theorem_count_dim", 0)
        ),
        "tree_theorem_first_dim": int(
            getattr(config, "tree_theorem_first_dim", 0)
        ),
        "tree_theorem_last_dim": int(
            getattr(config, "tree_theorem_last_dim", 0)
        ),
        "leaf_supervision_kind": str(getattr(config, "leaf_supervision_kind", "")),
        "direct_selection_metrics": direct_selection_metrics,
        "runtime_markov_c2": {
            "primary_metric_kind": FNO_TREE_C2_METRIC_KIND,
            "proxy_metric_kind": FNO_TREE_C2_PROXY_METRIC_KIND,
            "exact_witness_kind": FNO_TREE_C2_EXACT_WITNESS_KIND,
        },
        "theorem_sections": {
            "C1_to_L1": {
                split: dict(direct_selection_metrics.get(split) or {})
                for split in REPORTED_SPLITS
            },
            "C3_to_L2": {
                split: {
                    "merge_direct_exact_match": float(
                        dict(direct_selection_metrics.get(split) or {}).get(
                            "merge_direct_exact_match",
                            float("nan"),
                        )
                    ),
                    "merge_join_bit_accuracy": float(
                        dict(direct_selection_metrics.get(split) or {}).get(
                            "merge_join_bit_accuracy",
                            float("nan"),
                        )
                    ),
                }
                for split in REPORTED_SPLITS
            },
            "C2_to_L3": {
                split: {
                    "c2_on_range_exact_match": float(
                        dict(direct_selection_metrics.get(split) or {}).get(
                            "c2_on_range_exact_match",
                            float("nan"),
                        )
                    )
                }
                for split in REPORTED_SPLITS
            },
        },
        "failure_attribution": {
            "bucket": str(failure_bucket),
            "leaf_boundary_encoding_gap_score": float(leaf_boundary_gap_score),
            "theorem_count_decode_gap_score": float(theorem_count_decode_gap_score),
            "markov_sufficiency_gap_score": float(theorem_count_decode_gap_score),
            "root_mae_predicted_counts_predicted_endpoints": float(
                dict(direct_selection_metrics.get("test") or {}).get(
                    "root_mae_predicted_counts_predicted_endpoints",
                    float("nan"),
                )
            ),
            "root_mae_oracle_counts_predicted_endpoints": float(
                dict(direct_selection_metrics.get("test") or {}).get(
                    "root_mae_oracle_counts_predicted_endpoints",
                    float("nan"),
                )
            ),
            "root_mae_predicted_counts_oracle_endpoints": float(
                dict(direct_selection_metrics.get("test") or {}).get(
                    "root_mae_predicted_counts_oracle_endpoints",
                    float("nan"),
                )
            ),
            "phi_not_sufficient_score": float(phi_not_sufficient_score),
            "phi_not_compositional_score": float(phi_not_compositional_score),
            "count_composition_gap_score": float(count_composition_gap_score),
            "leaf_gap_score": float(leaf_boundary_gap_score),
            "merge_gap_score": float(count_composition_gap_score),
            "subtree_label_value_gap_score": float(subtree_label_value_gap_score),
            "internal_label_value_gap_score": float(subtree_label_value_gap_score),
            "readout_gap_score": float(readout_gap_score),
        },
    }
    if detail_mode == "debug_raw" and raw_artifacts:
        payload["raw_diagnostic_artifacts"] = {"exact_sketch": raw_artifacts}
    return payload


def _fit_tree_neural_baseline_with_predictions(
    *,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    device: torch.device,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
    root_weight: float = 1.0,
    c1_weight: float = 0.0,
    c2_weight: float = 0.0,
    c3_weight: float = 0.0,
    objective_summary: Mapping[str, Any] | None = None,
    budget_manifest: BudgetedTrainSupervisionManifest | None = None,
    prepared_train_fno_docs: Sequence[_FNOCountDoc] | None = None,
    prepared_val_fno_docs: Sequence[_FNOCountDoc] | None = None,
    prepared_test_fno_docs: Sequence[_FNOCountDoc] | None = None,
    prepared_leaf_sample_ordering_by_doc: Mapping[int, Sequence[int]] | None = None,
    prepared_internal_sample_ordering_by_doc: Mapping[int, Sequence[int]] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    memory_probe: Callable[[str, Mapping[str, Any]], None] | None = None,
    posttrain_diagnostics_mode: str = "full",
) -> Dict[str, Any]:
    """FNO tree-merge baseline with configurable law weights.

    Uses the official FNOCountSketch model trained via train_fno_tree.
    Evaluation uses _eval_fno_model to report full law-level metrics
    (root MAE, leaf MAE / C1, C2 count drift, merge MAE / C3)
    alongside prediction arrays.
    """
    if not HAS_NEURAL_OPERATOR:
        raise ImportError(
            "tree_neural baseline requires neuraloperator; install with: "
            "uv add neuraloperator"
        )
    if not train_docs:
        zero = _eval_root_predictions([], [], tau=float(config.violation_tau))
        empty = np.zeros((0,), dtype=np.float64)
        return {
            "train_metrics": zero,
            "val_metrics": zero,
            "test_metrics": zero,
            "fit_diag": TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=tuple(),
                epochs_completed=0,
                selection_metric_curve=tuple(),
                selection_mode="not_trained",
                selection_split="config",
                selection_metric_name="not_trained",
                selection_metric_value=float("nan"),
                best_epoch=0,
            ),
            "train_preds": empty,
            "val_preds": empty,
            "test_preds": empty,
            "train_truths": empty,
            "val_truths": empty,
            "test_truths": empty,
            "train_docs_used": 0,
            "objective_summary": dict(objective_summary or {}),
            "c2_metric_kind": FNO_TREE_C2_METRIC_KIND,
            "c2_proxy_metric_kind": FNO_TREE_C2_PROXY_METRIC_KIND,
            "c2_exact_witness_kind": FNO_TREE_C2_EXACT_WITNESS_KIND,
        }

    def _emit_memory_probe(event: str, **payload: Any) -> None:
        if memory_probe is None:
            return
        memory_probe(
            str(event),
            {str(key): value for key, value in dict(payload).items()},
        )

    n_regimes = int(config.n_regimes)
    vocab_size = int(config.vocab_size)
    leaf_tokens = int(config.fixed_leaf_tokens)

    # Prepare FNO count docs (raw token IDs per leaf span).
    train_fno_docs = tuple(prepared_train_fno_docs or ())
    if len(train_fno_docs) != len(train_docs):
        train_fno_docs = _prepare_fno_count_docs(train_docs, leaf_tokens=leaf_tokens)
    val_fno_docs = tuple(prepared_val_fno_docs or ())
    if len(val_fno_docs) != len(val_docs):
        val_fno_docs = _prepare_fno_count_docs(val_docs, leaf_tokens=leaf_tokens)
    test_fno_docs = tuple(prepared_test_fno_docs or ())
    if len(test_fno_docs) != len(test_docs):
        test_fno_docs = _prepare_fno_count_docs(test_docs, leaf_tokens=leaf_tokens)

    train_y = _doc_root_targets(train_docs)
    val_y = _doc_root_targets(val_docs)
    test_y = _doc_root_targets(test_docs)
    class_target_max, root_class_values, root_class_index, _class_values_arr = _class_setup(
        train_y,
        val_y,
        test_y,
    )
    target_scale = (
        float(class_target_max)
        if str(config.tree_root_supervision_kind) == "count_ce"
        else _resolved_markov_target_scale(
            config,
            observed_targets=train_y,
        )
    )
    tree_leaf_fno_hparams = _resolved_tree_leaf_fno_hyperparameters(config)

    # Build FNOCountSketch model.
    model = FNOCountSketch(
        vocab_size=vocab_size,
        leaf_tokens=leaf_tokens,
        state_dim=int(config.state_dim),
        hidden_dim=int(config.hidden_dim),
        target_scale=target_scale,
        n_regimes=n_regimes,
        doc_sequence_class_values=root_class_values,
        fno_width=int(tree_leaf_fno_hparams["tree_leaf_fno_width"]),
        fno_n_modes=int(tree_leaf_fno_hparams["tree_leaf_fno_n_modes"]),
        fno_n_layers=int(tree_leaf_fno_hparams["tree_leaf_fno_n_layers"]),
        leaf_fno_pooling=str(tree_leaf_fno_hparams["tree_leaf_fno_pooling"]),
        root_supervision_kind=str(config.tree_root_supervision_kind),
        root_count_class_values=root_class_values,
        aligned_sketch_surface=str(config.aligned_sketch_surface),
        summary_spec_name=str(getattr(config, "summary_spec_name", "")),
        slot_count=int(getattr(config, "slot_count", 0)),
        join_bit_weight=float(getattr(config, "tree_join_bit_weight", 0.0)),
        endpoint_loss_scale=float(getattr(config, "endpoint_loss_scale", 1.0)),
        task_head_mode=str(getattr(config, "tree_task_head_mode", "full_state_scalar")),
        theorem_surface_mode=str(
            getattr(config, "tree_theorem_surface_mode", "shared_bottleneck")
        ),
        theorem_count_head_mode=str(
            getattr(config, "tree_theorem_count_head_mode", "scalar_mse")
        ),
        theorem_count_ordinal_weight=float(
            getattr(config, "tree_theorem_count_ordinal_weight", 1.0)
        ),
        theorem_count_scalar_aux_weight=float(
            getattr(config, "tree_theorem_count_scalar_aux_weight", 0.25)
        ),
        theorem_count_threshold_balance=bool(
            getattr(config, "tree_theorem_count_threshold_balance", True)
        ),
        theorem_feature_dim=int(getattr(config, "tree_theorem_feature_dim", 48)),
        theorem_feature_hidden_dim=int(
            getattr(config, "tree_theorem_feature_hidden_dim", 256)
        ),
        theorem_score_dim=int(getattr(config, "tree_theorem_score_dim", 0)),
        theorem_fiber_dim=int(getattr(config, "tree_theorem_fiber_dim", 0)),
        theorem_aux_dim=int(getattr(config, "tree_theorem_aux_dim", 0)),
        score_merge_mode="gated_affine",
        phi_alignment_loss=str(
            getattr(config, "tree_phi_alignment_loss", "cosine_mse")
        ),
        c2_mode=str(getattr(config, "tree_c2_mode", "reconstruction")),
        theorem_feature_adapter=str(
            getattr(config, "theorem_feature_adapter", "markov_count_sketch")
        ),
        theorem_pair_same_threshold=getattr(
            config, "theorem_pair_same_threshold", None
        ),
        theorem_pair_diff_threshold=getattr(
            config, "theorem_pair_diff_threshold", None
        ),
        summary_spec_root_mode=str(
            getattr(config, "tree_summary_spec_root_mode", "factored_theorem_readout")
        ),
        theorem_count_dim=int(getattr(config, "tree_theorem_count_dim", 0)),
        theorem_first_dim=int(getattr(config, "tree_theorem_first_dim", 0)),
        theorem_last_dim=int(getattr(config, "tree_theorem_last_dim", 0)),
        tree_model_version=str(getattr(config, "tree_model_version", "legacy")),
    ).to(device=device)

    model_seed = int(seeds.get("effective_model_seed", 0))
    tree_supervision_source = str(
        getattr(config, "tree_supervision_source", "rate") or "rate"
    ).strip().lower() or "rate"
    budget_manifest = _resolved_tree_supervision_manifest(
        docs=train_fno_docs,
        config=config,
        budget_manifest=budget_manifest,
        leaf_sample_ordering_by_doc=prepared_leaf_sample_ordering_by_doc,
        internal_sample_ordering_by_doc=prepared_internal_sample_ordering_by_doc,
    )
    document_mode_by_doc, leaf_indices_by_doc, internal_indices_by_doc = budgeted_manifest_plan_maps(
        budget_manifest
    )
    posttrain_train_doc_limit = int(
        getattr(config, "tree_posttrain_train_doc_limit", 0)
    )
    if posttrain_train_doc_limit > 0:
        train_eval_docs = tuple(train_docs[:posttrain_train_doc_limit])
        train_eval_fno_docs = tuple(train_fno_docs[:posttrain_train_doc_limit])
        train_eval_y = np.asarray(train_y[:posttrain_train_doc_limit], dtype=np.float64)
    else:
        train_eval_docs = tuple(train_docs)
        train_eval_fno_docs = tuple(train_fno_docs)
        train_eval_y = np.asarray(train_y, dtype=np.float64)

    # Train using the official train_fno_tree loop.
    _prev_tree_runtime_mode = os.environ.get("TT_TREE_BATCH_RUNTIME_MODE")
    os.environ["TT_TREE_BATCH_RUNTIME_MODE"] = str(
        getattr(config, "tree_batch_runtime_mode", "legacy") or "legacy"
    )
    try:
        progress_snapshot_dir = ""
        artifact_dir_text = str(getattr(config, "artifact_dir", "") or "").strip()
        if artifact_dir_text:
            progress_snapshot_dir = str(
                Path(artifact_dir_text).expanduser() / "training_progress"
            )
        train_result = train_fno_tree(
            model=model,
            train_docs=train_fno_docs,
            val_docs=val_fno_docs,
            device=device,
            n_epochs=int(config.n_epochs),
            batch_size=int(config.batch_size),
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
            root_weight=float(root_weight),
            c1_weight=float(c1_weight),
            c2_weight=float(c2_weight),
            c3_weight=float(c3_weight),
            root_class_index=root_class_index,
            doc_sequence_class_index=root_class_index,
            document_supervision_mode_by_doc=(
                document_mode_by_doc
                if tree_supervision_source == "manifest"
                else (document_mode_by_doc or None)
            ),
            tree_document_loss_normalization_mode=str(
                getattr(config, "tree_document_loss_normalization_mode", "auto")
            ),
            leaf_audit_indices_by_doc=(
                leaf_indices_by_doc
                if tree_supervision_source == "manifest"
                else (leaf_indices_by_doc or None)
            ),
            c3_audit_indices_by_doc=(
                internal_indices_by_doc
                if tree_supervision_source == "manifest"
                else (internal_indices_by_doc or None)
            ),
            internal_supervision_kind=str(config.internal_supervision_kind),
            internal_label_rate=float(config.internal_label_rate),
            max_internal_depth=int(getattr(config, "max_internal_depth", 0)),
            leaf_exact_supervision=bool(config.leaf_exact_supervision),
            leaf_supervision_kind=str(getattr(config, "leaf_supervision_kind", "full_sketch")),
            leaf_label_rate=float(getattr(config, "leaf_label_rate", 1.0)),
            tree_local_weighting_mode=str(
                getattr(config, "tree_local_weighting_mode", "fixed_k_hajek")
            ),
            tree_supervision_source=str(tree_supervision_source),
            phi_compose_weight=float(getattr(config, "tree_phi_compose_weight", 1.0)),
            phi_contrastive_weight=float(
                getattr(config, "tree_phi_contrastive_weight", 0.25)
            ),
            checkpoint_metric=str(getattr(config, "tree_checkpoint_metric", "val_root_mae")),
            tree_training_schedule=str(
                getattr(config, "tree_training_schedule", "single_stage")
            ),
            tree_stage1_epochs=int(getattr(config, "tree_stage1_epochs", 0)),
            tree_stage2_epochs=int(getattr(config, "tree_stage2_epochs", 0)),
            tree_stage1_checkpoint_metric=str(
                getattr(config, "tree_stage1_checkpoint_metric", "val_root_mae")
            ),
            tree_stage1_eval_mode=str(
                getattr(config, "tree_stage1_eval_mode", "per_epoch")
            ),
            tree_stage1_screen_doc_limit=int(
                getattr(config, "tree_stage1_screen_doc_limit", 0)
            ),
            tree_stage1_final_exact_doc_limit=int(
                getattr(config, "tree_stage1_final_exact_doc_limit", 0)
            ),
            exact_metric_selection_doc_limit=int(
                getattr(config, "exact_metric_selection_doc_limit", 0)
            ),
            exact_metric_selection_interval=int(
                getattr(config, "exact_metric_selection_interval", 1)
            ),
            exact_metric_final_doc_limit=int(
                getattr(config, "exact_metric_final_doc_limit", 0)
            ),
            tree_exact_eval_max_docs=int(
                getattr(config, "tree_exact_eval_max_docs", 0)
            ),
            tree_batch_pack_mode=str(
                getattr(config, "tree_batch_pack_mode", "structure_bucket")
            ),
            tree_batch_token_budget=int(
                getattr(config, "tree_batch_token_budget", 0)
            ),
            tree_batch_node_budget=int(
                getattr(config, "tree_batch_node_budget", 0)
            ),
            tree_batch_autotune=bool(
                getattr(config, "tree_batch_autotune", True)
            ),
            tree_eval_workers_per_mig=int(
                getattr(config, "tree_eval_workers_per_mig", 0)
            ),
            tree_stage1_artifact_dir=str(
                getattr(config, "tree_stage1_artifact_dir", "")
            ),
            tree_stage1_resume_if_available=bool(
                getattr(config, "tree_stage1_resume_if_available", True)
            ),
            leaf_sample_ordering_by_doc=prepared_leaf_sample_ordering_by_doc,
            internal_sample_ordering_by_doc=prepared_internal_sample_ordering_by_doc,
            tree_stage1_root_weight=float(
                getattr(config, "tree_stage1_root_weight", 0.0)
            ),
            runtime_config=_gpu_runtime_config_from_ops_config(
                config,
                device=device,
            ),
            progress_callback=progress_callback,
            progress_snapshot_interval=int(
                getattr(config, "tree_progress_snapshot_interval", 10)
            ),
            progress_snapshot_dir=progress_snapshot_dir,
            memory_probe=memory_probe,
            depth_discount_gamma=float(
                getattr(config, "depth_discount_gamma", 1.0)
            ),
            seed=model_seed,
        )
    finally:
        if _prev_tree_runtime_mode is None:
            os.environ.pop("TT_TREE_BATCH_RUNTIME_MODE", None)
        else:
            os.environ["TT_TREE_BATCH_RUNTIME_MODE"] = _prev_tree_runtime_mode
    train_result.pop("best_model_state", None)
    _trim_host_allocator()
    _emit_memory_probe(
        "post_train_result_cleanup",
        posttrain_diagnostics_mode=str(posttrain_diagnostics_mode),
    )

    # Evaluate using _eval_fno_model for full law-level metrics (root, C1, C2, C3).
    tau = float(config.violation_tau)
    _emit_memory_probe("pre_posttrain_eval_fno_model", split="train")
    train_metrics = _eval_fno_model(model, train_eval_fno_docs, device=device, tau=tau)
    _emit_memory_probe("post_posttrain_eval_fno_model", split="train")
    _trim_host_allocator()
    _emit_memory_probe("post_posttrain_eval_fno_model_trim", split="train")
    _emit_memory_probe("pre_posttrain_eval_fno_model", split="val")
    val_metrics = _eval_fno_model(model, val_fno_docs, device=device, tau=tau)
    _emit_memory_probe("post_posttrain_eval_fno_model", split="val")
    _trim_host_allocator()
    _emit_memory_probe("post_posttrain_eval_fno_model_trim", split="val")
    _emit_memory_probe("pre_posttrain_eval_fno_model", split="test")
    test_metrics = _eval_fno_model(model, test_fno_docs, device=device, tau=tau)
    _emit_memory_probe("post_posttrain_eval_fno_model", split="test")
    _trim_host_allocator()
    _emit_memory_probe("post_posttrain_eval_fno_model_trim", split="test")

    # Extract prediction arrays via tree merge-up.
    _emit_memory_probe("pre_posttrain_root_predictions", split="train")
    train_preds = _fno_tree_root_predictions(model, train_eval_fno_docs, device=device)
    _emit_memory_probe("post_posttrain_root_predictions", split="train")
    _trim_host_allocator()
    _emit_memory_probe("post_posttrain_root_predictions_trim", split="train")
    _emit_memory_probe("pre_posttrain_root_predictions", split="val")
    val_preds = _fno_tree_root_predictions(model, val_fno_docs, device=device)
    _emit_memory_probe("post_posttrain_root_predictions", split="val")
    _trim_host_allocator()
    _emit_memory_probe("post_posttrain_root_predictions_trim", split="val")
    _emit_memory_probe("pre_posttrain_root_predictions", split="test")
    test_preds = _fno_tree_root_predictions(model, test_fno_docs, device=device)
    _emit_memory_probe("post_posttrain_root_predictions", split="test")
    _trim_host_allocator()
    _emit_memory_probe("post_posttrain_root_predictions_trim", split="test")

    val_truths = val_y
    test_truths = test_y

    fit_diag_raw = train_result.get("fit_diag")
    if isinstance(fit_diag_raw, TrainFitDiagnostics):
        fit_diag = TrainFitDiagnostics(
            train_loss_final=float(fit_diag_raw.train_loss_final),
            train_loss_curve=fit_diag_raw.train_loss_curve,
            epochs_completed=int(fit_diag_raw.epochs_completed),
            selection_metric_curve=fit_diag_raw.selection_metric_curve,
            selection_mode=str(fit_diag_raw.selection_mode),
            selection_split=str(fit_diag_raw.selection_split),
            selection_metric_name=str(fit_diag_raw.selection_metric_name),
            selection_metric_value=float(fit_diag_raw.selection_metric_value),
            best_epoch=int(fit_diag_raw.best_epoch),
            train_exact_match_rate=float(_exact_match_rate(train_preds, train_eval_y.tolist())),
            val_exact_match_rate=float(_exact_match_rate(val_preds, val_truths.tolist())),
            test_exact_match_rate=float(_exact_match_rate(test_preds, test_truths.tolist())),
            stage1_selection_metric_curve=fit_diag_raw.stage1_selection_metric_curve,
            stage2_selection_metric_curve=fit_diag_raw.stage2_selection_metric_curve,
            stage1_selection_metric_name=str(fit_diag_raw.stage1_selection_metric_name),
            stage2_selection_metric_name=str(fit_diag_raw.stage2_selection_metric_name),
            training_schedule=str(fit_diag_raw.training_schedule),
        )
    else:
        fit_diag = TrainFitDiagnostics(
            train_loss_final=float(train_result.get("train_loss_final", float("nan"))),
            train_loss_curve=tuple(train_result.get("loss_curve", ())),
            epochs_completed=int(train_result.get("epochs_completed", config.n_epochs)),
            selection_metric_curve=tuple(
                train_result.get("selection_metric_curve", ())
            ),
            selection_mode=str(train_result.get("selection_mode", "val_root_mae")),
            selection_split=str(train_result.get("selection_split", "val")),
            selection_metric_name=str(train_result.get("selection_metric_name", "root_mae")),
            selection_metric_value=float(train_result.get("best_val_mae", float("nan"))),
            best_epoch=int(train_result.get("best_epoch", 0)),
            train_exact_match_rate=float(_exact_match_rate(train_preds, train_eval_y.tolist())),
            val_exact_match_rate=float(_exact_match_rate(val_preds, val_truths.tolist())),
            test_exact_match_rate=float(_exact_match_rate(test_preds, test_truths.tolist())),
            stage1_selection_metric_curve=tuple(
                train_result.get("stage1_selection_metric_curve", ())
            ),
            stage2_selection_metric_curve=tuple(
                train_result.get("stage2_selection_metric_curve", ())
            ),
            stage1_selection_metric_name=str(
                train_result.get("stage1_selection_metric_name", "")
            ),
            stage2_selection_metric_name=str(
                train_result.get("stage2_selection_metric_name", "")
            ),
            training_schedule=str(train_result.get("training_schedule", "")),
        )

    # Store law weights used for training in the result for later analysis.
    law_weights = {
        "root_weight": float(root_weight),
        "c1_weight": float(c1_weight),
        "c2_weight": float(c2_weight),
        "c3_weight": float(c3_weight),
    }
    posttrain_diagnostics_t0 = time.perf_counter()
    exact_sketch_diagnostics: Dict[str, Any] = {}
    teacher_first_decomposition: Dict[str, Dict[str, float]] = {}
    if str(posttrain_diagnostics_mode).strip().lower() != "minimal":
        _emit_memory_probe("pre_posttrain_exact_sketch_diagnostics")
        exact_sketch_diagnostics = _tree_exact_sketch_diagnostics(
            model=model,
            config=config,
            device=device,
            train_docs=train_eval_docs,
            val_docs=val_docs,
            test_docs=test_docs,
            train_fno_docs=train_eval_fno_docs,
            val_fno_docs=val_fno_docs,
            test_fno_docs=test_fno_docs,
        )
        _emit_memory_probe("post_posttrain_exact_sketch_diagnostics")
        _trim_host_allocator()
        _emit_memory_probe("post_posttrain_exact_sketch_diagnostics_trim")
        if bool(getattr(model, "use_factorized_score_fiber_surface", False)):
            _emit_memory_probe("pre_posttrain_scorefiber_root_probe")
            exact_sketch_diagnostics["scorefiber_root_probe"] = {
                "train": eval_scorefiber_root_probe_metrics(
                    model,
                    train_eval_fno_docs,
                    train_eval_fno_docs,
                    device=device,
                ),
                "val": eval_scorefiber_root_probe_metrics(
                    model,
                    train_eval_fno_docs,
                    val_fno_docs,
                    device=device,
                ),
                "test": eval_scorefiber_root_probe_metrics(
                    model,
                    train_eval_fno_docs,
                    test_fno_docs,
                    device=device,
                ),
            }
            _emit_memory_probe("post_posttrain_scorefiber_root_probe")
            _trim_host_allocator()
            _emit_memory_probe("post_posttrain_scorefiber_root_probe_trim")
        stage1_best_model_state = train_result.get("stage1_best_model_state")
        if isinstance(stage1_best_model_state, Mapping):
            _emit_memory_probe("pre_posttrain_teacher_first_decomposition")
            teacher_first_detail_mode = _resolved_diagnostic_detail_mode(config)
            teacher_first_raw_root = (
                _resolved_raw_diagnostic_artifact_root(
                    config,
                    namespace="teacher_first",
                )
                if teacher_first_detail_mode == "debug_raw"
                else None
            )
            teacher_first_decomposition = {
                "train": _eval_fno_teacher_first_decomposition_metrics(
                    model,
                    train_eval_fno_docs,
                    device=device,
                    stage1_model_state=stage1_best_model_state,
                    diagnostic_detail_mode=teacher_first_detail_mode,
                    raw_artifact_dir=(
                        None if teacher_first_raw_root is None else teacher_first_raw_root / "train"
                    ),
                ),
                "val": _eval_fno_teacher_first_decomposition_metrics(
                    model,
                    val_fno_docs,
                    device=device,
                    stage1_model_state=stage1_best_model_state,
                    diagnostic_detail_mode=teacher_first_detail_mode,
                    raw_artifact_dir=(
                        None if teacher_first_raw_root is None else teacher_first_raw_root / "val"
                    ),
                ),
                "test": _eval_fno_teacher_first_decomposition_metrics(
                    model,
                    test_fno_docs,
                    device=device,
                    stage1_model_state=stage1_best_model_state,
                    diagnostic_detail_mode=teacher_first_detail_mode,
                    raw_artifact_dir=(
                        None if teacher_first_raw_root is None else teacher_first_raw_root / "test"
                    ),
                ),
            }
            _emit_memory_probe("post_posttrain_teacher_first_decomposition")
            _trim_host_allocator()
            _emit_memory_probe("post_posttrain_teacher_first_decomposition_trim")
        train_result.pop("stage1_best_model_state", None)
    else:
        _emit_memory_probe(
            "skip_posttrain_heavy_diagnostics",
            posttrain_diagnostics_mode=str(posttrain_diagnostics_mode),
        )
    failure_attribution = dict(exact_sketch_diagnostics.get("failure_attribution") or {})
    elapsed_s_posttrain_diag = float(time.perf_counter() - posttrain_diagnostics_t0)

    return {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "fit_diag": fit_diag,
        "train_preds": np.asarray(train_preds, dtype=np.float64),
        "val_preds": np.asarray(val_preds, dtype=np.float64),
        "test_preds": np.asarray(test_preds, dtype=np.float64),
        "train_truths": np.asarray(train_eval_y, dtype=np.float64),
        "val_truths": np.asarray(val_truths, dtype=np.float64),
        "test_truths": np.asarray(test_truths, dtype=np.float64),
        "train_docs_used": int(
            budget_manifest.touched_docs_total
            if budget_manifest is not None
            else len(train_docs)
        ),
        "law_weights": law_weights,
        "objective_summary": dict(objective_summary or {}),
        "c2_metric_kind": FNO_TREE_C2_METRIC_KIND,
        "c2_proxy_metric_kind": FNO_TREE_C2_PROXY_METRIC_KIND,
        "c2_exact_witness_kind": FNO_TREE_C2_EXACT_WITNESS_KIND,
        "budget_manifest": _budget_manifest_metadata(budget_manifest),
        "selection_metric_curve": tuple(train_result.get("selection_metric_curve", ())),
        "stage1_selection_metric_curve": tuple(
            train_result.get("stage1_selection_metric_curve", ())
        ),
        "stage2_selection_metric_curve": tuple(
            train_result.get("stage2_selection_metric_curve", ())
        ),
        "timing_breakdown": dict(train_result.get("timing_breakdown", {}) or {}),
        "batching_metrics": dict(train_result.get("batching_metrics", {}) or {}),
        "runtime_efficiency": dict(
            train_result.get("runtime_efficiency", {}) or {}
        ),
        "tree_document_loss_normalization_mode": str(
            train_result.get(
                "tree_document_loss_normalization_mode",
                getattr(config, "tree_document_loss_normalization_mode", "auto"),
            )
            or "auto"
        ),
        "effective_tree_document_loss_normalization_mode": str(
            train_result.get(
                "effective_tree_document_loss_normalization_mode",
                getattr(config, "tree_document_loss_normalization_mode", "auto"),
            )
            or "auto"
        ),
        "document_supervision_docs_total": int(
            train_result.get("document_supervision_docs_total", 0) or 0
        ),
        "root_supervision_docs_total": int(
            train_result.get("root_supervision_docs_total", 0) or 0
        ),
        "doc_sequence_supervision_docs_total": int(
            train_result.get("doc_sequence_supervision_docs_total", 0) or 0
        ),
        "document_supervision_coverage_rate": float(
            train_result.get("document_supervision_coverage_rate", 0.0) or 0.0
        ),
        "document_loss_mean_batch_scale": float(
            train_result.get("document_loss_mean_batch_scale", 1.0) or 1.0
        ),
        "normalized_root_contribution_final": float(
            train_result.get("normalized_root_contribution_final", float("nan"))
        ),
        "autotuned_batch_budgets": dict(
            train_result.get("autotuned_batch_budgets", {}) or {}
        ),
        "elapsed_s_train_loop": float(
            train_result.get("elapsed_s_train_loop", float("nan"))
        ),
        "elapsed_s_screen_eval": float(
            train_result.get("elapsed_s_screen_eval", float("nan"))
        ),
        "elapsed_s_exact_metric_eval": float(
            train_result.get("elapsed_s_exact_metric_eval", float("nan"))
        ),
        "elapsed_s_split_eval": float(
            train_result.get("elapsed_s_split_eval", float("nan"))
        ),
        "elapsed_s_posttrain_diag": float(elapsed_s_posttrain_diag),
        "elapsed_s_state_clone": float(
            train_result.get("elapsed_s_state_clone", float("nan"))
        ),
        "posttrain_diagnostics_mode": str(posttrain_diagnostics_mode),
        "training_component_loss_curves": dict(
            train_result.get("training_component_loss_curves", {}) or {}
        ),
        "training_component_loss_finals": dict(
            train_result.get("training_component_loss_finals", {}) or {}
        ),
        "training_schedule": str(train_result.get("training_schedule", "")),
        "training_progress_snapshot_interval": int(
            train_result.get("progress_snapshot_interval", 0) or 0
        ),
        "training_progress_snapshot_dir": str(
            train_result.get("progress_snapshot_dir", "") or ""
        ),
        "latest_training_progress_snapshot_path": str(
            train_result.get("latest_progress_snapshot_path", "") or ""
        ),
        "tree_local_weighting_mode": str(
            train_result.get(
                "tree_local_weighting_mode",
                getattr(config, "tree_local_weighting_mode", "fixed_k_hajek"),
            )
        ),
        "local_loss_kind": str(train_result.get("local_loss_kind", "")),
        "local_sampling_design_name": str(
            train_result.get("local_sampling_design_name", "")
        ),
        "c2_pair_weighting_mode": str(
            train_result.get("c2_pair_weighting_mode", "")
        ),
        "c2_same_pair_count": float(
            train_result.get("c2_same_pair_count", float("nan"))
        ),
        "c2_different_pair_count": float(
            train_result.get("c2_different_pair_count", float("nan"))
        ),
        "c2_pair_weight_ess": float(
            train_result.get("c2_pair_weight_ess", float("nan"))
        ),
        "c2_pair_weight_max": float(
            train_result.get("c2_pair_weight_max", float("nan"))
        ),
        "leaf_population_size": float(
            train_result.get("leaf_population_size", float("nan"))
        ),
        "leaf_sample_size": float(
            train_result.get("leaf_sample_size", float("nan"))
        ),
        "leaf_effective_propensity": float(
            train_result.get("leaf_effective_propensity", float("nan"))
        ),
        "merge_population_size": float(
            train_result.get("merge_population_size", float("nan"))
        ),
        "merge_sample_size": float(
            train_result.get("merge_sample_size", float("nan"))
        ),
        "merge_effective_propensity": float(
            train_result.get("merge_effective_propensity", float("nan"))
        ),
        "local_objective_audit": dict(
            train_result.get("local_objective_audit", {}) or {}
        ),
        "exact_sketch_diagnostics": exact_sketch_diagnostics,
        "exact_sketch_failure_bucket": str(failure_attribution.get("bucket", "")),
        "exact_sketch_leaf_gap_score": float(
            failure_attribution.get("leaf_gap_score", float("nan"))
        ),
        "exact_sketch_merge_gap_score": float(
            failure_attribution.get("merge_gap_score", float("nan"))
        ),
        "exact_sketch_theorem_count_decode_gap_score": float(
            failure_attribution.get("theorem_count_decode_gap_score", float("nan"))
        ),
        "exact_sketch_markov_sufficiency_gap_score": float(
            failure_attribution.get(
                "markov_sufficiency_gap_score",
                failure_attribution.get("theorem_count_decode_gap_score", float("nan")),
            )
        ),
        "exact_sketch_phi_not_sufficient_score": float(
            failure_attribution.get("phi_not_sufficient_score", float("nan"))
        ),
        "exact_sketch_phi_not_compositional_score": float(
            failure_attribution.get("phi_not_compositional_score", float("nan"))
        ),
        "exact_sketch_subtree_label_value_gap_score": float(
            failure_attribution.get("subtree_label_value_gap_score", float("nan"))
        ),
        "exact_sketch_internal_label_value_gap_score": float(
            failure_attribution.get("internal_label_value_gap_score", float("nan"))
        ),
        "exact_sketch_readout_gap_score": float(
            failure_attribution.get("readout_gap_score", float("nan"))
        ),
        "teacher_first_decomposition": teacher_first_decomposition,
        "stage1_artifact": dict(train_result.get("stage1_artifact", {}) or {}),
        "root_summary_probe_audit": dict(
            train_result.get("root_summary_probe_audit") or {}
        ),
        "training_component_loss_curves": dict(
            train_result.get("training_component_loss_curves") or {}
        ),
        "training_component_loss_finals": dict(
            train_result.get("training_component_loss_finals") or {}
        ),
    }


def _fit_tree_neural_c2_baseline_with_predictions(
    *,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    device: torch.device,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
    budget_manifest: BudgetedTrainSupervisionManifest | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    memory_probe: Callable[[str, Mapping[str, Any]], None] | None = None,
    posttrain_diagnostics_mode: str = "full",
) -> Dict[str, Any]:
    """FNO tree-merge baseline with root + C2 count-drift objective."""
    objective_summary = _build_objective_summary(config)
    return _fit_tree_neural_baseline_with_predictions(
        config=config,
        seeds=seeds,
        device=device,
        train_docs=train_docs,
        val_docs=val_docs,
        test_docs=test_docs,
        root_weight=float(objective_summary["optimization_root_weight"]),
        c1_weight=float(objective_summary["local_law_c1_weight"]),
        c2_weight=float(objective_summary["local_law_c2_weight"]),
        c3_weight=float(objective_summary["local_law_c3_weight"]),
        objective_summary=objective_summary,
        budget_manifest=budget_manifest,
        progress_callback=progress_callback,
        memory_probe=memory_probe,
        posttrain_diagnostics_mode=posttrain_diagnostics_mode,
    )


def _palette_block_ids_from_tokens(
    tokens: Sequence[int],
    *,
    vocab_size: int,
    n_regimes: int,
) -> np.ndarray:
    v = int(vocab_size)
    n = int(n_regimes)
    if v <= 0 or n <= 0:
        raise ValueError("vocab_size and n_regimes must be positive")
    block_by_token = np.empty((v,), dtype=np.int64)
    for regime_id, block in enumerate(np.array_split(np.arange(v, dtype=np.int64), n)):
        block_by_token[np.asarray(block, dtype=np.int64)] = int(regime_id)
    toks = np.asarray(tokens, dtype=np.int64)
    if toks.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if int(np.min(toks)) < 0 or int(np.max(toks)) >= v:
        raise ValueError(
            "palette_block_exact encountered token ids outside [0, vocab_size)"
        )
    return block_by_token[toks]


def _palette_block_exact_predictions(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    vocab_size: int,
    n_regimes: int,
) -> np.ndarray:
    preds: List[float] = []
    for doc in docs:
        block_ids = _palette_block_ids_from_tokens(
            doc.tokens,
            vocab_size=int(vocab_size),
            n_regimes=int(n_regimes),
        )
        if int(block_ids.size) < 2:
            preds.append(0.0)
            continue
        preds.append(float(np.sum(block_ids[:-1] != block_ids[1:])))
    return np.asarray(preds, dtype=np.float64)


def _fit_palette_block_exact_with_predictions(
    *,
    config: OPSCountConfig,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
) -> Dict[str, Any]:
    if str(config.generator_profile).strip().lower() not in {
        "piecewise_disjoint_palette",
        "hazard_topic",
    }:
        raise ValueError(
            "palette_block_exact requires a disjoint-palette generator profile "
            "(piecewise_disjoint_palette or hazard_topic)"
        )
    train_truths = _doc_root_targets(train_docs)
    val_truths = _doc_root_targets(val_docs)
    test_truths = _doc_root_targets(test_docs)
    train_preds = _palette_block_exact_predictions(
        train_docs,
        vocab_size=int(config.vocab_size),
        n_regimes=int(config.n_regimes),
    )
    val_preds = _palette_block_exact_predictions(
        val_docs,
        vocab_size=int(config.vocab_size),
        n_regimes=int(config.n_regimes),
    )
    test_preds = _palette_block_exact_predictions(
        test_docs,
        vocab_size=int(config.vocab_size),
        n_regimes=int(config.n_regimes),
    )
    fit_diag = TrainFitDiagnostics(
        train_loss_final=0.0,
        train_loss_curve=(0.0,),
        epochs_completed=0,
        selection_metric_curve=(0.0,),
        selection_mode="deterministic_exact_rule",
        selection_split="oracle_rule",
        selection_metric_name="train_root_mae",
        selection_metric_value=0.0,
        best_epoch=0,
        train_exact_match_rate=_exact_match_rate(train_preds, train_truths),
        val_exact_match_rate=_exact_match_rate(val_preds, val_truths),
        test_exact_match_rate=_exact_match_rate(test_preds, test_truths),
    )
    return {
        "train_metrics": _eval_root_predictions(
            train_preds,
            train_truths,
            tau=float(config.violation_tau),
        ),
        "val_metrics": _eval_root_predictions(
            val_preds,
            val_truths,
            tau=float(config.violation_tau),
        ),
        "test_metrics": _eval_root_predictions(
            test_preds,
            test_truths,
            tau=float(config.violation_tau),
        ),
        "fit_diag": fit_diag,
        "train_preds": train_preds,
        "val_preds": val_preds,
        "test_preds": test_preds,
        "train_truths": train_truths,
        "val_truths": val_truths,
        "test_truths": test_truths,
        "train_docs_used": int(len(train_docs)),
    }


def _normalize_baseline_family(family: str) -> str:
    key = str(family).strip().lower()
    if key == "ridge_control":
        return "raw_token_ngram_ridge"
    if key == "tree_ridge":
        return "tree_ridge_leaf"
    return key


def default_baseline_families_for_mode(
    *,
    hardness_grid: str = "",
) -> tuple[str, ...]:
    if str(hardness_grid or "").strip().lower().startswith("structural_core_"):
        return DEFAULT_STRUCTURAL_CORE_BASELINE_FAMILIES
    return DEFAULT_DIAGNOSTIC_BASELINE_FAMILIES


def _prediction_histogram(
    values: np.ndarray,
    *,
    support_values: Sequence[int],
) -> Dict[str, int]:
    if values.size <= 0:
        return {str(int(value)): 0 for value in support_values}
    rounded = np.asarray(np.rint(values), dtype=np.int64)
    observed_values = sorted(set(int(x) for x in rounded.tolist()) | {int(v) for v in support_values})
    return {
        str(int(value)): int(np.sum(rounded == int(value)))
        for value in observed_values
    }


def _confusion_payload(
    truths: np.ndarray,
    preds: np.ndarray,
    *,
    support_values: Sequence[int],
) -> Dict[str, Any]:
    classes = [int(value) for value in support_values]
    truth_int = np.asarray(np.rint(truths), dtype=np.int64)
    pred_int = np.asarray(np.rint(preds), dtype=np.int64)
    matrix: List[List[int]] = []
    per_class: Dict[str, Dict[str, Any]] = {}
    for truth_class in classes:
        row: List[int] = []
        for pred_class in classes:
            row.append(
                int(
                    np.sum(
                        (truth_int == int(truth_class))
                        & (pred_int == int(pred_class))
                    )
                )
            )
        matrix.append(row)
    for cls in classes:
        tp = int(np.sum((truth_int == cls) & (pred_int == cls)))
        support = int(np.sum(truth_int == cls))
        predicted = int(np.sum(pred_int == cls))
        recall = float(tp / support) if support > 0 else float("nan")
        precision = float(tp / predicted) if predicted > 0 else float("nan")
        per_class[str(int(cls))] = {
            "support": int(support),
            "predicted": int(predicted),
            "true_positives": int(tp),
            "recall": float(recall),
            "precision": float(precision),
        }
    return {
        "labels": [int(value) for value in classes],
        "matrix": matrix,
        "per_class": per_class,
    }


def _run_family_with_predictions(
    *,
    baseline_family: str,
    config: OPSCountConfig,
    benchmark: FullDocDiagnosticBenchmarkSpec | None = None,
    seeds: Mapping[str, int],
    device: torch.device,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
    budget_manifest: BudgetedTrainSupervisionManifest | None = None,
    prepared_train_fno_docs: Sequence[_FNOCountDoc] | None = None,
    prepared_val_fno_docs: Sequence[_FNOCountDoc] | None = None,
    prepared_test_fno_docs: Sequence[_FNOCountDoc] | None = None,
    prepared_leaf_sample_ordering_by_doc: Mapping[int, Sequence[int]] | None = None,
    prepared_internal_sample_ordering_by_doc: Mapping[int, Sequence[int]] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    memory_probe: Callable[[str, Mapping[str, Any]], None] | None = None,
    posttrain_diagnostics_mode: str = "full",
) -> Dict[str, Any]:
    family = _normalize_baseline_family(baseline_family)
    collapse_runtime_mode = ""
    if family == "tree_neural":
        collapse_runtime_mode = str(
            getattr(config, "tree_exact_collapse_mode", "") or ""
        ).strip()
    train_docs_effective = tuple(train_docs)
    if budget_manifest is not None and (
        family in FULL_DOC_ONLY_BUDGET_FAMILIES
        or (
            family == "tree_neural"
            and collapse_runtime_mode in TREE_NEURAL_EXACT_LOCKED_MODES
        )
    ):
        train_docs_effective = _subset_docs_by_indices(
            train_docs,
            _selected_document_indices_from_budget_manifest(
                budget_manifest,
                n_items=len(train_docs),
                seed=int(getattr(config, "seed", 0)),
            ),
        )
    _set_global_seed(int(seeds["effective_model_seed"]))
    if family in {"official_fno", "official_fno_sumlen"}:
        if not HAS_NEURAL_OPERATOR:
            raise ImportError(
                f"{family} baseline requested but neuraloperator is not installed"
            )
        fno_config = (
            _effective_config_for_family(
                benchmark=benchmark,
                baseline_family=family,
                config=config,
            )
            if benchmark is not None
            else config
        )
        if family == "official_fno_sumlen":
            fno_config = replace(
                fno_config,
                doc_sequence_fno_pooling="sum",
                doc_sequence_fno_concat_length_feature=True,
                doc_sequence_fno_include_transition_channel=False,
            )
        result = _fit_fno_baseline_with_predictions(
            config=fno_config,
            seeds=seeds,
            device=device,
            train_docs=train_docs_effective,
            val_docs=val_docs,
            test_docs=test_docs,
        )
        result["effective_config"] = fno_config
        if budget_manifest is not None:
            result["train_docs_used"] = int(len(train_docs_effective))
            result["budget_manifest"] = _budget_manifest_metadata(budget_manifest)
        return result
    if family == "cnn1d":
        result = _fit_cnn1d_baseline_with_predictions(
            config=config,
            seeds=seeds,
            device=device,
            train_docs=train_docs_effective,
            val_docs=val_docs,
            test_docs=test_docs,
        )
        result["effective_config"] = config
        if budget_manifest is not None:
            result["train_docs_used"] = int(len(train_docs_effective))
            result["budget_manifest"] = _budget_manifest_metadata(budget_manifest)
        return result
    if family == "mlp_bigram":
        result = _fit_mlp_bigram_baseline_with_predictions(
            config=config,
            seeds=seeds,
            device=device,
            train_docs=train_docs_effective,
            val_docs=val_docs,
            test_docs=test_docs,
        )
        result["effective_config"] = config
        if budget_manifest is not None:
            result["train_docs_used"] = int(len(train_docs_effective))
            result["budget_manifest"] = _budget_manifest_metadata(budget_manifest)
        return result
    if family == "palette_block_exact":
        result = _fit_palette_block_exact_with_predictions(
            config=config,
            train_docs=train_docs_effective,
            val_docs=val_docs,
            test_docs=test_docs,
        )
        result["effective_config"] = config
        if budget_manifest is not None:
            result["train_docs_used"] = int(len(train_docs_effective))
            result["budget_manifest"] = _budget_manifest_metadata(budget_manifest)
        return result
    if family == "raw_token_ngram_ridge":
        result = _fit_ridge_control_with_predictions(
            config=config,
            train_docs=train_docs_effective,
            val_docs=val_docs,
            test_docs=test_docs,
        )
        result["effective_config"] = config
        if budget_manifest is not None:
            result["train_docs_used"] = int(len(train_docs_effective))
            result["budget_manifest"] = _budget_manifest_metadata(budget_manifest)
        return result
    if family == "tree_ridge_leaf":
        result = _fit_tree_ridge_baseline_with_predictions(
            config=config,
            seeds=seeds,
            train_docs=train_docs,
            val_docs=val_docs,
            test_docs=test_docs,
        )
        result["effective_config"] = config
        return result
    if family == "tree_doc_ridge":
        result = _fit_tree_doc_ridge_baseline_with_predictions(
            config=config,
            train_docs=train_docs_effective,
            val_docs=val_docs,
            test_docs=test_docs,
        )
        result["effective_config"] = config
        if budget_manifest is not None:
            result["train_docs_used"] = int(len(train_docs_effective))
            result["budget_manifest"] = _budget_manifest_metadata(budget_manifest)
        return result
    if family == "tree_neural_c2":
        effective_config = _tree_neural_family_effective_config(
            config,
            family=family,
        )
        result = _fit_tree_neural_c2_baseline_with_predictions(
            config=effective_config,
            seeds=seeds,
            device=device,
            train_docs=train_docs,
            val_docs=val_docs,
            test_docs=test_docs,
            budget_manifest=budget_manifest,
            progress_callback=progress_callback,
            memory_probe=memory_probe,
            posttrain_diagnostics_mode=posttrain_diagnostics_mode,
        )
        result["effective_config"] = effective_config
        return result
    if family == "tree_neural_c2c3":
        effective_config = _tree_neural_family_effective_config(
            config,
            family=family,
        )
        objective_summary = _build_objective_summary(effective_config)
        result = _fit_tree_neural_baseline_with_predictions(
            config=effective_config,
            seeds=seeds,
            device=device,
            train_docs=train_docs,
            val_docs=val_docs,
            test_docs=test_docs,
            root_weight=float(objective_summary["optimization_root_weight"]),
            c1_weight=float(objective_summary["local_law_c1_weight"]),
            c2_weight=float(objective_summary["local_law_c2_weight"]),
            c3_weight=float(objective_summary["local_law_c3_weight"]),
            objective_summary=objective_summary,
            budget_manifest=budget_manifest,
            prepared_train_fno_docs=prepared_train_fno_docs,
            prepared_val_fno_docs=prepared_val_fno_docs,
            prepared_test_fno_docs=prepared_test_fno_docs,
            prepared_leaf_sample_ordering_by_doc=prepared_leaf_sample_ordering_by_doc,
            prepared_internal_sample_ordering_by_doc=prepared_internal_sample_ordering_by_doc,
            progress_callback=progress_callback,
            memory_probe=memory_probe,
            posttrain_diagnostics_mode=posttrain_diagnostics_mode,
        )
        result["effective_config"] = effective_config
        return result
    if family == "tree_neural":
        if collapse_runtime_mode in {
            "official_fno_one_tree_identity",
            "official_fno_runtime_identity",
        }:
            if benchmark is None:
                raise ValueError(
                    "tree_exact_collapse_mode requires a resolved benchmark"
                )
            fno_config = _effective_config_for_family(
                benchmark=benchmark,
                baseline_family="official_fno",
                config=config,
            )
            tree_effective_config = _tree_neural_family_effective_config(
                config,
                family=family,
            )
            result = _fit_fno_baseline_with_predictions(
                config=fno_config,
                seeds=seeds,
                device=device,
                train_docs=train_docs_effective,
                val_docs=val_docs,
                test_docs=test_docs,
            )
            result["c2_metric_kind"] = FNO_TREE_C2_METRIC_KIND
            result["c2_proxy_metric_kind"] = FNO_TREE_C2_PROXY_METRIC_KIND
            result["c2_exact_witness_kind"] = FNO_TREE_C2_EXACT_WITNESS_KIND
            result["effective_config"] = (
                tree_effective_config
                if collapse_runtime_mode == "official_fno_one_tree_identity"
                else fno_config
            )
            result["collapse_runtime_delegate_family"] = "official_fno"
            result["collapse_runtime_mode"] = collapse_runtime_mode
            if budget_manifest is not None:
                result["train_docs_used"] = int(len(train_docs_effective))
                result["budget_manifest"] = _budget_manifest_metadata(
                    budget_manifest
                )
            return result
        effective_config = _tree_neural_family_effective_config(
            config,
            family=family,
        )
        objective_summary = _build_objective_summary(effective_config)
        result = _fit_tree_neural_baseline_with_predictions(
            config=effective_config,
            seeds=seeds,
            device=device,
            train_docs=train_docs,
            val_docs=val_docs,
            test_docs=test_docs,
            root_weight=float(objective_summary["optimization_root_weight"]),
            c1_weight=float(objective_summary["local_law_c1_weight"]),
            c2_weight=float(objective_summary["local_law_c2_weight"]),
            c3_weight=float(objective_summary["local_law_c3_weight"]),
            objective_summary=objective_summary,
            budget_manifest=budget_manifest,
            prepared_train_fno_docs=prepared_train_fno_docs,
            prepared_val_fno_docs=prepared_val_fno_docs,
            prepared_test_fno_docs=prepared_test_fno_docs,
            prepared_leaf_sample_ordering_by_doc=prepared_leaf_sample_ordering_by_doc,
            prepared_internal_sample_ordering_by_doc=prepared_internal_sample_ordering_by_doc,
            progress_callback=progress_callback,
            memory_probe=memory_probe,
            posttrain_diagnostics_mode=posttrain_diagnostics_mode,
        )
        result["effective_config"] = effective_config
        return result
    raise ValueError(
        f"unsupported baseline_family={baseline_family!r}; expected one of {VALID_BASELINE_FAMILIES}"
    )


def _run_payload(
    *,
    benchmark: FullDocDiagnosticBenchmarkSpec,
    baseline_family: str,
    train_doc_count: int,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    device: torch.device,
    bundle: MarkovOPSDataBundle,
    bundle_source: str,
    emit_confusion: bool,
    prepared_tree_data: _PreparedMarkovTreeData | None = None,
    output_dir: Path | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    memory_probe: Callable[[str, Mapping[str, Any]], None] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    normalized_family = _normalize_baseline_family(baseline_family)
    run_config = config
    if prepared_tree_data is not None and str(prepared_tree_data.signature).strip():
        run_config = replace(
            run_config,
            prepared_data_signature=str(prepared_tree_data.signature),
    )
    effective_train_config = _effective_train_config_for_full_doc_run(
        benchmark=benchmark,
        baseline_family=normalized_family,
        train_doc_count=int(train_doc_count),
        config=run_config,
    )
    budget_manifest: BudgetedTrainSupervisionManifest | None = None
    if _requires_explicit_budget_manifest_for_run(
        baseline_family=normalized_family,
        config=effective_train_config,
    ):
        budget_manifest = build_budgeted_train_supervision_manifest(
            docs=tuple(bundle.train_docs),
            config=effective_train_config,
            baseline_family=normalized_family,
            seed=int(effective_train_config.seed),
        )
    tuning_stage = str((run_metadata or {}).get("tuning_stage", "") or "").strip().lower()
    configured_posttrain_diagnostics_mode = str(
        getattr(effective_train_config, "posttrain_diagnostics_mode", "") or ""
    ).strip().lower()
    metadata_posttrain_diagnostics_mode = str(
        (run_metadata or {}).get("posttrain_diagnostics_mode", "") or ""
    ).strip().lower()
    posttrain_diagnostics_mode = (
        configured_posttrain_diagnostics_mode
        or metadata_posttrain_diagnostics_mode
        or (
            "minimal"
            if tuning_stage in {"capacity_screen", "capacity_locked", "screen", "study_screen"}
            else "full"
        )
    )
    fit = _run_family_with_predictions(
        baseline_family=normalized_family,
        config=effective_train_config,
        benchmark=benchmark,
        seeds=seeds,
        device=device,
        train_docs=tuple(bundle.train_docs),
        val_docs=tuple(bundle.val_docs),
        test_docs=tuple(bundle.test_docs),
        budget_manifest=budget_manifest,
        prepared_train_fno_docs=(
            tuple(prepared_tree_data.train_fno_docs[: int(train_doc_count)])
            if prepared_tree_data is not None
            else None
        ),
        prepared_val_fno_docs=(
            tuple(prepared_tree_data.val_fno_docs)
            if prepared_tree_data is not None
            else None
        ),
        prepared_test_fno_docs=(
            tuple(prepared_tree_data.test_fno_docs)
            if prepared_tree_data is not None
            else None
        ),
        prepared_leaf_sample_ordering_by_doc=(
            {
                int(doc_idx): tuple(ordering)
                for doc_idx, ordering in enumerate(
                    tuple(
                        prepared_tree_data.leaf_orderings_by_seed.get(
                            int(effective_train_config.seed),
                            tuple(),
                        )
                    )[: int(train_doc_count)]
                )
            }
            if prepared_tree_data is not None
            else None
        ),
        prepared_internal_sample_ordering_by_doc=(
            {
                int(doc_idx): tuple(ordering)
                for doc_idx, ordering in enumerate(
                    tuple(
                        prepared_tree_data.internal_orderings_by_seed.get(
                            int(effective_train_config.seed),
                            tuple(),
                        )
                    )[: int(train_doc_count)]
                )
            }
            if prepared_tree_data is not None
            else None
        ),
        progress_callback=progress_callback,
        memory_probe=memory_probe,
        posttrain_diagnostics_mode=posttrain_diagnostics_mode,
    )
    effective_config = fit.get("effective_config", effective_train_config)
    if not isinstance(effective_config, OPSCountConfig):
        effective_config = effective_train_config
    comparison_mode = normalize_markov_comparison_mode(
        str(getattr(effective_config, "comparison_mode", "legacy") or "legacy")
    )
    comparison_surface = resolve_markov_comparable_surface(
        benchmark=benchmark,
        config=effective_config,
        comparison_mode=comparison_mode,
    )
    comparison_surface_snapshot = comparison_surface.to_dict()
    comparison_surface_diff_payload = comparison_surface_diff(
        expected_surface=comparison_surface,
        actual_config=effective_config,
    )
    requested_intent_config = effective_train_config
    if normalized_family in TREE_NEURAL_BASELINE_FAMILIES:
        requested_intent_config = _tree_neural_family_effective_config(
            effective_train_config,
            family=normalized_family,
        )
    requested_run_intent = materialize_tree_run_intent(
        requested_intent_config,
        fixed_leaf_tokens_override=int(
            getattr(requested_intent_config, "fixed_leaf_tokens", 0) or 0
        ),
        baseline_family_override=normalized_family,
    )
    effective_run_intent = materialize_tree_run_intent(
        effective_config,
        fixed_leaf_tokens_override=int(
            getattr(effective_config, "fixed_leaf_tokens", 0) or 0
        ),
        baseline_family_override=normalized_family,
    )
    requested_effective_run_intent_diff = intent_diff(
        requested_run_intent,
        effective_run_intent,
    )
    allowed_locked_comparator_drift = bool(
        requested_effective_run_intent_diff
        and str(
            requested_run_intent.get("tree_exact_collapse_mode", "")
        ).strip()
    )
    if requested_effective_run_intent_diff and not allowed_locked_comparator_drift:
        raise ValueError(
            "unexpected run intent drift: "
            f"{json.dumps(requested_effective_run_intent_diff, sort_keys=True)}"
        )
    run_intent_validation_status = (
        "locked_comparator"
        if requested_effective_run_intent_diff
        else "validated"
    )
    objective_summary = fit.get("objective_summary")
    if isinstance(objective_summary, Mapping):
        resolved_objective = {
            **_resolved_objective_metadata_for_run(
                effective_config,
                baseline_family=normalized_family,
            ),
            **dict(objective_summary),
        }
    else:
        resolved_objective = _resolved_objective_metadata_for_run(
            effective_config,
            baseline_family=normalized_family,
        )
    requested_tree_semantics = _tree_semantic_config_snapshot(config)
    pre_family_tree_semantics = _tree_semantic_config_snapshot(
        effective_train_config
    )
    post_family_tree_semantics = _tree_semantic_config_snapshot(effective_config)
    resolved_objective_semantics = _tree_semantic_objective_snapshot(
        resolved_objective
    )
    semantic_drift = _validate_tree_semantic_preservation(
        baseline_family=normalized_family,
        requested_config=config,
        pre_family_config=effective_train_config,
        post_family_config=effective_config,
    )
    objective_semantic_validation = _validate_tree_objective_snapshot(
        baseline_family=normalized_family,
        effective_config=effective_config,
        resolved_objective=resolved_objective,
    )
    leaf_pressure_ablation_diagnostics = _tree_leaf_pressure_ablation_diagnostics(
        config_snapshot=post_family_tree_semantics,
        objective_snapshot=resolved_objective_semantics,
    )
    fit_diag = fit["fit_diag"]
    support_values = sorted(
        {
            int(round(float(value)))
            for value in np.asarray(fit["train_truths"], dtype=np.float64).tolist()
        }
        | {
            int(round(float(value)))
            for value in np.asarray(fit["val_truths"], dtype=np.float64).tolist()
        }
        | {
            int(round(float(value)))
            for value in np.asarray(fit["test_truths"], dtype=np.float64).tolist()
        }
    )
    if not support_values:
        support_values = [0]
    tree_leaf_fno_hparams = _resolved_tree_leaf_fno_hyperparameters(effective_config)
    budget_metadata = _budget_metadata_for_payload(
        config=effective_config,
        fit_budget_manifest=cast(
            Mapping[str, Any] | None,
            fit.get("budget_manifest"),
        ),
        fallback_manifest=budget_manifest,
        train_doc_count=len(bundle.train_docs),
    )
    supervision_contract = _tree_supervision_contract_summary(
        config=effective_config,
        budget_metadata=budget_metadata,
    )
    if bool(supervision_contract["required"]) and not bool(supervision_contract["passed"]):
        raise ValueError(
            "tree supervision contract failed: "
            f"{json.dumps(supervision_contract, sort_keys=True)}"
        )
    requested_fixed_leaf_tokens = int(getattr(config, "fixed_leaf_tokens", 0) or 0)
    leaf_tokens = int(effective_config.fixed_leaf_tokens)
    train_leaf_counts = _leaf_count_stats(
        tuple(bundle.train_docs),
        leaf_tokens=leaf_tokens,
    )
    val_leaf_counts = _leaf_count_stats(
        tuple(bundle.val_docs),
        leaf_tokens=leaf_tokens,
    )
    test_leaf_counts = _leaf_count_stats(
        tuple(bundle.test_docs),
        leaf_tokens=leaf_tokens,
    )
    executed_leaves_per_doc = int(
        max(0, round(float(test_leaf_counts["mean_leaves_per_doc"])))
    )
    executed_internal_nodes_per_doc = max(0, int(executed_leaves_per_doc) - 1)
    exact_parity_row = bool(
        normalized_family == "tree_neural"
        and str(getattr(effective_config, "tree_exact_collapse_mode", "") or "").strip()
        == "official_fno_one_tree_identity"
        and int(executed_leaves_per_doc) == 1
        and str(getattr(effective_config, "doc_consumption_mode", "") or "").strip().lower()
        == "root_only"
        and abs(float(getattr(effective_config, "leaf_label_rate", 0.0) or 0.0)) <= 1e-12
        and abs(float(getattr(effective_config, "internal_label_rate", 0.0) or 0.0)) <= 1e-12
    )
    parity_mode = "exact_full_doc" if exact_parity_row else ""
    payload = {
        "benchmark": str(benchmark.name),
        "cell_id": str(benchmark.cell_id or ""),
        "hardness_grid": str(benchmark.grid_name or ""),
        "benchmark_description": str(benchmark.description),
        "degenerate_benchmark": bool(benchmark.degenerate),
        "baseline_family": str(normalized_family),
        "seed": int(effective_config.seed),
        "train_doc_count": int(train_doc_count),
        "n_regimes": int(config.n_regimes),
        "segment_density_band": str(benchmark.segment_density_band or ""),
        "segment_min": int(benchmark.segment_min or config.min_segments),
        "segment_max": int(benchmark.segment_max or config.max_segments),
        "train_size_multiplier": float(
            float(train_doc_count)
            / float(
                max(
                    1,
                    resolve_markov_observed_token_policy(
                        profile_name=str(benchmark.observed_token_profile)
                    ).train_docs,
                )
            )
        ),
        "bundle_source": str(bundle_source),
        "train_corpus_signature": str(bundle.train_corpus_signature),
        "val_corpus_signature": str(bundle.val_corpus_signature),
        "test_corpus_signature": str(bundle.test_corpus_signature),
        "objective_variant": str(effective_config.doc_sequence_objective),
        "device_requested": "cuda" if bool(effective_config.use_cuda) else "cpu",
        "device_resolved": str(device),
        "posttrain_diagnostics_mode": str(posttrain_diagnostics_mode),
        "config": {
            "baseline_family": str(normalized_family),
            "state_dim": int(effective_config.state_dim),
            "hidden_dim": int(effective_config.hidden_dim),
            "n_epochs": int(effective_config.n_epochs),
            "batch_size": int(effective_config.batch_size),
            "lr": float(effective_config.lr),
            "weight_decay": float(effective_config.weight_decay),
            "doc_sequence_objective": str(effective_config.doc_sequence_objective),
            "doc_sequence_fno_pooling": str(effective_config.doc_sequence_fno_pooling),
            "doc_sequence_fno_concat_length_feature": bool(
                effective_config.doc_sequence_fno_concat_length_feature
            ),
            "doc_sequence_fno_include_transition_channel": bool(
                effective_config.doc_sequence_fno_include_transition_channel
            ),
            "tree_root_supervision_kind": str(
                effective_config.tree_root_supervision_kind
            ),
            "tree_checkpoint_metric": str(effective_config.tree_checkpoint_metric),
            "tree_stage1_checkpoint_metric": str(
                effective_config.tree_stage1_checkpoint_metric
            ),
            "tree_stage1_artifact_dir": str(
                getattr(effective_config, "tree_stage1_artifact_dir", "")
            ),
            "tree_stage1_artifact_root": str(
                getattr(effective_config, "tree_stage1_artifact_root", "")
            ),
            "tree_stage1_resume_if_available": bool(
                getattr(effective_config, "tree_stage1_resume_if_available", True)
            ),
            "prepared_data_root": str(
                getattr(effective_config, "prepared_data_root", "")
            ),
            "prepared_data_allow_create": bool(
                getattr(effective_config, "prepared_data_allow_create", True)
            ),
            "prepared_data_signature": str(
                getattr(effective_config, "prepared_data_signature", "")
            ),
            "tree_document_loss_normalization_mode": str(
                getattr(
                    effective_config,
                    "tree_document_loss_normalization_mode",
                    "auto",
                )
            ),
            "tree_stage1_root_weight": float(
                getattr(effective_config, "tree_stage1_root_weight", 0.0)
            ),
            "tree_training_schedule": str(effective_config.tree_training_schedule),
            "tree_stage1_epochs": int(effective_config.tree_stage1_epochs),
            "tree_stage2_epochs": int(effective_config.tree_stage2_epochs),
            "tree_batch_structural_pad_limit": float(
                getattr(effective_config, "tree_batch_structural_pad_limit", 0.5)
            ),
            "tree_batch_auto_queue_min_docs": int(
                getattr(effective_config, "tree_batch_auto_queue_min_docs", 8)
            ),
            "tree_batch_auto_queue_min_fill_ratio": float(
                getattr(effective_config, "tree_batch_auto_queue_min_fill_ratio", 0.5)
            ),
            "tree_task_head_mode": str(effective_config.tree_task_head_mode),
            "tree_theorem_count_head_mode": str(
                getattr(effective_config, "tree_theorem_count_head_mode", "")
            ),
            "tree_summary_spec_root_mode": str(
                getattr(effective_config, "tree_summary_spec_root_mode", "")
            ),
            "tree_join_bit_weight": float(effective_config.tree_join_bit_weight),
            "aligned_sketch_surface": str(effective_config.aligned_sketch_surface),
            "internal_supervision_kind": str(
                effective_config.internal_supervision_kind
            ),
            "internal_label_rate": float(effective_config.internal_label_rate),
            "leaf_exact_supervision": bool(effective_config.leaf_exact_supervision),
            "leaf_supervision_kind": str(effective_config.leaf_supervision_kind),
            "summary_spec_name": str(getattr(effective_config, "summary_spec_name", "")),
            "slot_count": int(getattr(effective_config, "slot_count", 0)),
            "tree_theorem_count_dim": int(
                getattr(effective_config, "tree_theorem_count_dim", 0)
            ),
            "tree_theorem_first_dim": int(
                getattr(effective_config, "tree_theorem_first_dim", 0)
            ),
            "tree_theorem_last_dim": int(
                getattr(effective_config, "tree_theorem_last_dim", 0)
            ),
            "tree_local_weighting_mode": str(
                getattr(effective_config, "tree_local_weighting_mode", "fixed_k_hajek")
            ),
            "tree_supervision_source": str(
                getattr(effective_config, "tree_supervision_source", "rate")
            ),
            "tree_c2_mode": str(getattr(effective_config, "tree_c2_mode", "")),
            "depth_discount_gamma": float(
                getattr(effective_config, "depth_discount_gamma", 1.0)
            ),
            "tree_exact_collapse_mode": str(
                getattr(effective_config, "tree_exact_collapse_mode", "")
            ),
            "leaf_label_rate": float(getattr(effective_config, "leaf_label_rate", 1.0)),
            "tree_leaf_fno_width": int(tree_leaf_fno_hparams["tree_leaf_fno_width"]),
            "tree_leaf_fno_n_modes": int(
                tree_leaf_fno_hparams["tree_leaf_fno_n_modes"]
            ),
            "tree_leaf_fno_n_layers": int(
                tree_leaf_fno_hparams["tree_leaf_fno_n_layers"]
            ),
            "tree_aux_doc_sequence_fraction": float(
                effective_config.doc_sequence_train_fraction
            ),
            "local_law_weight": (
                None
                if effective_config.local_law_weight is None
                else float(effective_config.local_law_weight)
            ),
            "task_objective_weight": (
                None
                if effective_config.task_objective_weight is None
                else float(effective_config.task_objective_weight)
            ),
            "c1_relative_weight": float(effective_config.c1_relative_weight),
            "c2_relative_weight": float(effective_config.c2_relative_weight),
            "c3_relative_weight": float(effective_config.c3_relative_weight),
            "schedule_consistency_weight": float(
                effective_config.schedule_consistency_weight
            ),
            "law_package": str(effective_config.law_package),
            "root_weight": float(effective_config.root_weight),
            "leaf_weight": float(effective_config.leaf_weight),
            "c2_weight": float(effective_config.c2_weight),
            "c3_weight": float(effective_config.c3_weight),
            "fixed_leaf_tokens": int(effective_config.fixed_leaf_tokens),
            "max_internal_depth": int(
                getattr(effective_config, "max_internal_depth", 0)
            ),
            "requested_fixed_leaf_tokens": int(requested_fixed_leaf_tokens),
            "executed_fixed_leaf_tokens": int(effective_config.fixed_leaf_tokens),
            "budget_total_calls": int(getattr(effective_config, "budget_total_calls", 0)),
            "budget_total_calls_per_doc": float(
                getattr(effective_config, "budget_total_calls_per_doc", 0.0)
            ),
            "mass_target_per_doc": float(
                getattr(effective_config, "mass_target_per_doc", float("nan"))
            ),
            "full_doc_budget_share": float(
                getattr(effective_config, "full_doc_budget_share", 1.0)
            ),
            "doc_consumption_mode": str(
                getattr(effective_config, "doc_consumption_mode", "")
            ),
            "package_semantics": str(
                getattr(effective_config, "package_semantics", "")
            ),
            "local_split_mode": str(getattr(effective_config, "local_split_mode", "")),
            "local_allocation_policy": str(
                getattr(effective_config, "local_allocation_policy", "")
            ),
            "vocab_size": int(effective_config.vocab_size),
            "n_regimes": int(effective_config.n_regimes),
            "min_segments": int(effective_config.min_segments),
            "max_segments": int(effective_config.max_segments),
            "min_distinct_regimes_per_doc": (
                None
                if effective_config.min_distinct_regimes_per_doc is None
                else int(effective_config.min_distinct_regimes_per_doc)
            ),
            "max_distinct_regimes_per_doc": (
                None
                if effective_config.max_distinct_regimes_per_doc is None
                else int(effective_config.max_distinct_regimes_per_doc)
            ),
            "resolved_objective": {
                str(key): value
                for key, value in resolved_objective.items()
            },
            "comparison_mode": str(comparison_mode),
            "comparison_surface_snapshot": dict(comparison_surface_snapshot),
            "comparison_surface_diff": dict(comparison_surface_diff_payload),
        },
        "target_support": {
            "train": _root_count_support(bundle.train_docs),
            "val": _root_count_support(bundle.val_docs),
            "test": _root_count_support(bundle.test_docs),
        },
        "distinct_regime_support": {
            "train": _distinct_regime_support(bundle.train_docs),
            "val": _distinct_regime_support(bundle.val_docs),
            "test": _distinct_regime_support(bundle.test_docs),
        },
        "train_metrics": _metrics_payload(fit["train_metrics"]),
        "val_metrics": _metrics_payload(fit["val_metrics"]),
        "test_metrics": _metrics_payload(fit["test_metrics"]),
        "fit_diagnostics": asdict(fit_diag),
        "training_selection_metric_curve": tuple(
            fit.get("selection_metric_curve", ())
        ),
        "stage1_selection_metric_curve": tuple(
            fit.get("stage1_selection_metric_curve", ())
        ),
        "stage2_selection_metric_curve": tuple(
            fit.get("stage2_selection_metric_curve", ())
        ),
        "training_component_loss_curves": dict(
            fit.get("training_component_loss_curves", {}) or {}
        ),
        "training_component_loss_finals": dict(
            fit.get("training_component_loss_finals", {}) or {}
        ),
        "timing_breakdown": dict(fit.get("timing_breakdown", {}) or {}),
        "batching_metrics": dict(fit.get("batching_metrics", {}) or {}),
        "runtime_efficiency": dict(fit.get("runtime_efficiency", {}) or {}),
        "autotuned_batch_budgets": dict(
            fit.get("autotuned_batch_budgets", {}) or {}
        ),
        "elapsed_s_train_loop": float(
            fit.get("elapsed_s_train_loop", float("nan"))
        ),
        "elapsed_s_screen_eval": float(
            fit.get("elapsed_s_screen_eval", float("nan"))
        ),
        "elapsed_s_exact_metric_eval": float(
            fit.get("elapsed_s_exact_metric_eval", float("nan"))
        ),
        "elapsed_s_split_eval": float(
            fit.get("elapsed_s_split_eval", float("nan"))
        ),
        "elapsed_s_state_clone": float(
            fit.get("elapsed_s_state_clone", float("nan"))
        ),
        "training_schedule": str(fit.get("training_schedule", "")),
        "tree_supervision_source": str(
            fit.get(
                "tree_supervision_source",
                getattr(effective_config, "tree_supervision_source", "rate"),
            )
        ),
        "local_estimand_mode": str(
            fit.get(
                "local_estimand_mode",
                getattr(effective_config, "tree_local_weighting_mode", ""),
            )
        ),
        "package_semantics": str(
            fit.get(
                "package_semantics",
                getattr(effective_config, "package_semantics", ""),
            )
            or ""
        ),
        "depth_discount_gamma": float(
            fit.get(
                "depth_discount_gamma",
                getattr(effective_config, "depth_discount_gamma", 1.0),
            )
        ),
        "c2_pair_weighting_mode": str(
            fit.get("c2_pair_weighting_mode", "") or ""
        ),
        "c2_same_pair_count": float(
            fit.get("c2_same_pair_count", float("nan"))
        ),
        "c2_different_pair_count": float(
            fit.get("c2_different_pair_count", float("nan"))
        ),
        "c2_pair_weight_ess": float(
            fit.get("c2_pair_weight_ess", float("nan"))
        ),
        "c2_pair_weight_max": float(
            fit.get("c2_pair_weight_max", float("nan"))
        ),
        "collapse_runtime_delegate_family": str(
            fit.get("collapse_runtime_delegate_family", "") or ""
        ),
        "collapse_runtime_mode": str(fit.get("collapse_runtime_mode", "") or ""),
        "comparison_mode": str(comparison_mode),
        "comparison_surface_snapshot": dict(comparison_surface_snapshot),
        "comparison_surface_diff": dict(comparison_surface_diff_payload),
        "requested_run_intent": dict(requested_run_intent),
        "effective_run_intent": dict(effective_run_intent),
        "requested_effective_run_intent_diff": dict(
            requested_effective_run_intent_diff
        ),
        "run_intent_validation_status": str(run_intent_validation_status),
        "requested_fixed_leaf_tokens": int(requested_fixed_leaf_tokens),
        "executed_fixed_leaf_tokens": int(effective_config.fixed_leaf_tokens),
        "executed_leaves_per_doc": int(executed_leaves_per_doc),
        "executed_internal_nodes_per_doc": int(executed_internal_nodes_per_doc),
        "parity_mode": str(parity_mode),
        "is_exact_full_doc_parity_row": bool(exact_parity_row),
        "tree_root_supervision_kind": str(effective_config.tree_root_supervision_kind),
        "aligned_sketch_surface": str(effective_config.aligned_sketch_surface),
        "internal_supervision_kind": str(effective_config.internal_supervision_kind),
        "internal_label_rate": float(effective_config.internal_label_rate),
        "leaf_exact_supervision": bool(effective_config.leaf_exact_supervision),
        "tree_supervision_contract": dict(supervision_contract),
        "summary_spec_name": str(getattr(effective_config, "summary_spec_name", "")),
        "slot_count": int(getattr(effective_config, "slot_count", 0)),
        "leaf_label_rate": float(getattr(effective_config, "leaf_label_rate", 1.0)),
        "tree_document_loss_normalization_mode": str(
            getattr(effective_config, "tree_document_loss_normalization_mode", "auto")
        ),
        "effective_tree_document_loss_normalization_mode": str(
            fit.get(
                "effective_tree_document_loss_normalization_mode",
                getattr(effective_config, "tree_document_loss_normalization_mode", "auto"),
            )
            or "auto"
        ),
        "document_supervision_docs_total": int(
            fit.get("document_supervision_docs_total", 0) or 0
        ),
        "root_supervision_docs_total": int(
            fit.get("root_supervision_docs_total", 0) or 0
        ),
        "doc_sequence_supervision_docs_total": int(
            fit.get("doc_sequence_supervision_docs_total", 0) or 0
        ),
        "document_supervision_coverage_rate": float(
            fit.get("document_supervision_coverage_rate", 0.0) or 0.0
        ),
        "document_loss_mean_batch_scale": float(
            fit.get("document_loss_mean_batch_scale", 1.0) or 1.0
        ),
        "normalized_root_contribution_final": float(
            fit.get("normalized_root_contribution_final", float("nan"))
        ),
        "tree_checkpoint_metric": str(
            getattr(effective_config, "tree_checkpoint_metric", "")
        ),
        "tree_stage1_checkpoint_metric": str(
            getattr(effective_config, "tree_stage1_checkpoint_metric", "")
        ),
        "tree_stage1_eval_mode": str(
            getattr(effective_config, "tree_stage1_eval_mode", "")
        ),
        "tree_stage1_screen_doc_limit": int(
            getattr(effective_config, "tree_stage1_screen_doc_limit", 0)
        ),
        "tree_stage1_final_exact_doc_limit": int(
            getattr(effective_config, "tree_stage1_final_exact_doc_limit", 0)
        ),
        "exact_metric_selection_doc_limit": int(
            getattr(effective_config, "exact_metric_selection_doc_limit", 0)
        ),
        "exact_metric_selection_interval": int(
            getattr(effective_config, "exact_metric_selection_interval", 1)
        ),
        "exact_metric_final_doc_limit": int(
            getattr(effective_config, "exact_metric_final_doc_limit", 0)
        ),
        "tree_exact_eval_max_docs": int(
            getattr(effective_config, "tree_exact_eval_max_docs", 0)
        ),
        "tree_posttrain_train_doc_limit": int(
            getattr(effective_config, "tree_posttrain_train_doc_limit", 0)
        ),
        "tree_batch_pack_mode": str(
            getattr(effective_config, "tree_batch_pack_mode", "")
        ),
        "tree_batch_token_budget": int(
            getattr(effective_config, "tree_batch_token_budget", 0)
        ),
        "tree_batch_node_budget": int(
            getattr(effective_config, "tree_batch_node_budget", 0)
        ),
        "tree_batch_autotune": bool(
            getattr(effective_config, "tree_batch_autotune", False)
        ),
        "tree_batch_structural_pad_limit": float(
            getattr(effective_config, "tree_batch_structural_pad_limit", 0.5)
        ),
        "tree_batch_auto_queue_min_docs": int(
            getattr(effective_config, "tree_batch_auto_queue_min_docs", 8)
        ),
        "tree_batch_auto_queue_min_fill_ratio": float(
            getattr(effective_config, "tree_batch_auto_queue_min_fill_ratio", 0.5)
        ),
        "tree_eval_workers_per_mig": int(
            getattr(effective_config, "tree_eval_workers_per_mig", 0)
        ),
        "tree_stage1_artifact_dir": str(
            getattr(effective_config, "tree_stage1_artifact_dir", "")
        ),
        "prepared_data_root": str(getattr(effective_config, "prepared_data_root", "")),
        "prepared_data_allow_create": bool(
            getattr(effective_config, "prepared_data_allow_create", True)
        ),
        "prepared_data_signature": str(
            getattr(effective_config, "prepared_data_signature", "")
        ),
        "tree_stage1_root_weight": float(
            getattr(effective_config, "tree_stage1_root_weight", 0.0)
        ),
        "tree_join_bit_weight": float(
            getattr(effective_config, "tree_join_bit_weight", 0.0)
        ),
        "tree_training_schedule": str(
            getattr(effective_config, "tree_training_schedule", "")
        ),
        "tree_stage1_epochs": int(getattr(effective_config, "tree_stage1_epochs", 0)),
        "tree_stage2_epochs": int(getattr(effective_config, "tree_stage2_epochs", 0)),
        "tree_task_head_mode": str(
            getattr(effective_config, "tree_task_head_mode", "")
        ),
        "tree_theorem_count_head_mode": str(
            getattr(effective_config, "tree_theorem_count_head_mode", "")
        ),
        "tree_summary_spec_root_mode": str(
            getattr(effective_config, "tree_summary_spec_root_mode", "")
        ),
        "tree_theorem_count_dim": int(
            getattr(effective_config, "tree_theorem_count_dim", 0)
        ),
        "tree_theorem_first_dim": int(
            getattr(effective_config, "tree_theorem_first_dim", 0)
        ),
        "tree_theorem_last_dim": int(
            getattr(effective_config, "tree_theorem_last_dim", 0)
        ),
        "leaf_supervision_kind": str(
            getattr(effective_config, "leaf_supervision_kind", "")
        ),
        "tree_leaf_fno_width": int(tree_leaf_fno_hparams["tree_leaf_fno_width"]),
        "tree_leaf_fno_n_modes": int(tree_leaf_fno_hparams["tree_leaf_fno_n_modes"]),
        "tree_leaf_fno_n_layers": int(tree_leaf_fno_hparams["tree_leaf_fno_n_layers"]),
        "tree_aux_doc_sequence_fraction": float(
            effective_config.doc_sequence_train_fraction
        ),
        "fixed_leaf_tokens": int(effective_config.fixed_leaf_tokens),
        "train_mean_leaves_per_doc": float(train_leaf_counts["mean_leaves_per_doc"]),
        "val_mean_leaves_per_doc": float(val_leaf_counts["mean_leaves_per_doc"]),
        "test_mean_leaves_per_doc": float(test_leaf_counts["mean_leaves_per_doc"]),
        "train_max_leaves_per_doc": float(train_leaf_counts["max_leaves_per_doc"]),
        "val_max_leaves_per_doc": float(val_leaf_counts["max_leaves_per_doc"]),
        "test_max_leaves_per_doc": float(test_leaf_counts["max_leaves_per_doc"]),
        "train_docs_used": int(fit.get("train_docs_used", len(bundle.train_docs))),
        "budget_total_calls": int(budget_metadata["budget_total_calls"]),
        "budget_total_calls_per_doc": float(
            budget_metadata["budget_total_calls_per_doc"]
        ),
        "mass_target_per_doc": float(budget_metadata["mass_target_per_doc"]),
        "requested_root_mass_per_doc": float(
            budget_metadata["requested_root_mass_per_doc"]
        ),
        "budget_total_calls_used": int(budget_metadata["budget_total_calls_used"]),
        "budget_utilization": float(budget_metadata["budget_utilization"]),
        "full_doc_budget_share": float(budget_metadata["full_doc_budget_share"]),
        "full_doc_calls_requested": int(budget_metadata["full_doc_calls_requested"]),
        "full_doc_calls_total": int(budget_metadata["full_doc_calls_total"]),
        "local_calls_requested": int(budget_metadata["local_calls_requested"]),
        "local_calls_total": int(budget_metadata["local_calls_total"]),
        "doc_consumption_mode": str(budget_metadata["doc_consumption_mode"]),
        "local_split_mode": str(budget_metadata["local_split_mode"]),
        "local_allocation_policy": str(
            budget_metadata["local_allocation_policy"]
        ),
        "sampling_scheme": str(budget_metadata["sampling_scheme"]),
        "effective_full_doc_mass_total": float(
            budget_metadata["effective_full_doc_mass_total"]
        ),
        "effective_full_doc_mass_per_doc": float(
            budget_metadata["effective_full_doc_mass_per_doc"]
        ),
        "document_mass_share": float(budget_metadata["document_mass_share"]),
        "leaf_mass_share": float(budget_metadata["leaf_mass_share"]),
        "internal_mass_share": float(budget_metadata["internal_mass_share"]),
        "document_call_share": float(budget_metadata["document_call_share"]),
        "leaf_call_share": float(budget_metadata["leaf_call_share"]),
        "internal_call_share": float(budget_metadata["internal_call_share"]),
        "doc_touch_rate": float(budget_metadata["doc_touch_rate"]),
        "mean_labels_per_touched_doc": float(
            budget_metadata["mean_labels_per_touched_doc"]
        ),
        "touched_docs_total": int(budget_metadata["touched_docs_total"]),
        "budget_manifest": dict(budget_metadata),
        "prediction_histograms": {
            "train": _prediction_histogram(
                np.asarray(fit["train_preds"], dtype=np.float64),
                support_values=support_values,
            ),
            "val": _prediction_histogram(
                np.asarray(fit["val_preds"], dtype=np.float64),
                support_values=support_values,
            ),
            "test": _prediction_histogram(
                np.asarray(fit["test_preds"], dtype=np.float64),
                support_values=support_values,
            ),
        },
        **_split_final_metric_fields(fit=fit, fit_diag=fit_diag),
        "test_root_mae": float(fit["test_metrics"].root_mae),
        "test_exact_match_rate": float(fit_diag.test_exact_match_rate),
        "test_leaf_mae": float(getattr(fit["test_metrics"], "leaf_mae", float("nan"))),
        "test_c2_count_drift_r1_mae": float(
            getattr(fit["test_metrics"], "c2_count_drift_r1_mae", float("nan"))
        ),
        "test_c2_count_drift_r2_mae": float(
            getattr(fit["test_metrics"], "c2_count_drift_r2_mae", float("nan"))
        ),
        "test_c2_count_drift_r4_mae": float(
            getattr(fit["test_metrics"], "c2_count_drift_r4_mae", float("nan"))
        ),
        "test_c2_root_count_drift_r1_mae": float(
            getattr(fit["test_metrics"], "c2_root_count_drift_r1_mae", float("nan"))
        ),
        "test_c2_root_count_drift_r2_mae": float(
            getattr(fit["test_metrics"], "c2_root_count_drift_r2_mae", float("nan"))
        ),
        "test_c2_root_count_drift_r4_mae": float(
            getattr(fit["test_metrics"], "c2_root_count_drift_r4_mae", float("nan"))
        ),
        "test_c2_idempotence_mae": float(getattr(fit["test_metrics"], "c2_idempotence_mae", float("nan"))),
        "test_c2_state_replay_mse": float(
            getattr(fit["test_metrics"], "c2_state_replay_mse", float("nan"))
        ),
        "test_merge_mae": float(getattr(fit["test_metrics"], "merge_mae", float("nan"))),
        "test_schedule_spread_mean": float(getattr(fit["test_metrics"], "schedule_spread_mean", float("nan"))),
        "parameterization": str(resolved_objective.get("parameterization", "")),
        "weighting_scheme": str(resolved_objective.get("weighting_scheme", "")),
        "optimization_root_weight": float(
            resolved_objective.get("optimization_root_weight", float("nan"))
        ),
        "local_law_c1_weight": float(
            resolved_objective.get("local_law_c1_weight", float("nan"))
        ),
        "local_law_c2_weight": float(
            resolved_objective.get("local_law_c2_weight", float("nan"))
        ),
        "local_law_c3_weight": float(
            resolved_objective.get("local_law_c3_weight", float("nan"))
        ),
        "task_objective_weight_source": str(
            resolved_objective.get("task_objective_weight_source", "")
        ),
        "proxy_schedule_consistency_weight": float(
            resolved_objective.get("proxy_schedule_consistency_weight", 0.0)
        ),
        "theorem_terms": list(resolved_objective.get("theorem_terms") or []),
        "proxy_terms": list(resolved_objective.get("proxy_terms") or []),
        "formal_notes": str(resolved_objective.get("formal_notes", "")),
        "theorem_local_law_total_weight": float(
            resolved_objective.get("theorem_local_law_total_weight", 0.0)
        ),
        "proxy_schedule_term_total_weight": float(
            resolved_objective.get("proxy_schedule_term_total_weight", 0.0)
        ),
        "objective_surface_name": str(
            resolved_objective.get("objective_surface_name", MARKOV_FULL_DOC_OBJECTIVE_SURFACE)
        ),
        "objective_surface_distinct_from": list(
            resolved_objective.get("objective_surface_distinct_from")
            or [TREEPO_REGULARIZED_OBJECTIVE_SURFACE]
        ),
        "objective_surface_distinct_note": str(
            resolved_objective.get(
                "objective_surface_distinct_note",
                "",
            )
        ),
        "paper_to_lean_local_law_mapping": dict(
            resolved_objective.get("paper_to_lean_local_law_mapping")
            or PAPER_TO_LEAN_LOCAL_LAW_MAPPING
        ),
        "objective_weights_active": bool(
            resolved_objective.get("objective_weights_active", False)
        ),
        "c2_metric_kind": str(fit.get("c2_metric_kind", "")),
        "c2_proxy_metric_kind": str(fit.get("c2_proxy_metric_kind", "")),
        "c2_exact_witness_kind": str(fit.get("c2_exact_witness_kind", "")),
        "semantics_version": str(
            resolved_objective.get("semantics_version", "")
        ),
        "runtime_markov_c2": {
            "primary_metric_kind": str(fit.get("c2_metric_kind", "")),
            "proxy_metric_kind": str(fit.get("c2_proxy_metric_kind", "")),
            "exact_witness_kind": str(fit.get("c2_exact_witness_kind", "")),
            **{
                split: {
                    "c2_count_drift_r1_mae": float(
                        getattr(fit[f"{split}_metrics"], "c2_count_drift_r1_mae", float("nan"))
                    ),
                    "c2_count_drift_r2_mae": float(
                        getattr(fit[f"{split}_metrics"], "c2_count_drift_r2_mae", float("nan"))
                    ),
                    "c2_count_drift_r4_mae": float(
                        getattr(fit[f"{split}_metrics"], "c2_count_drift_r4_mae", float("nan"))
                    ),
                    "c2_root_count_drift_r1_mae": float(
                        getattr(
                            fit[f"{split}_metrics"],
                            "c2_root_count_drift_r1_mae",
                            float("nan"),
                        )
                    ),
                    "c2_root_count_drift_r2_mae": float(
                        getattr(
                            fit[f"{split}_metrics"],
                            "c2_root_count_drift_r2_mae",
                            float("nan"),
                        )
                    ),
                    "c2_root_count_drift_r4_mae": float(
                        getattr(
                            fit[f"{split}_metrics"],
                            "c2_root_count_drift_r4_mae",
                            float("nan"),
                        )
                    ),
                    "c2_state_replay_mse": float(
                        getattr(fit[f"{split}_metrics"], "c2_state_replay_mse", float("nan"))
                    ),
                }
                for split in REPORTED_SPLITS
            },
        },
        "semantic_config_views": {
            "requested": dict(requested_tree_semantics),
            "effective_pre_family_normalization": dict(pre_family_tree_semantics),
            "effective_post_family_normalization": dict(post_family_tree_semantics),
            "resolved_objective": dict(resolved_objective_semantics),
        },
        "semantic_config_drift": {
            "requested_to_pre_family_normalization": dict(
                semantic_drift["requested_to_pre_family_normalization"]
            ),
            "pre_to_post_family_normalization": dict(
                semantic_drift["pre_to_post_family_normalization"]
            ),
            "config_to_resolved_objective": dict(
                objective_semantic_validation["config_to_resolved_objective"]
            ),
        },
        "semantic_config_validation_status": str(semantic_drift["status"]),
        "semantic_config_validation_note": str(semantic_drift["note"]),
        "leaf_pressure_ablation_vacuous": bool(
            leaf_pressure_ablation_diagnostics["leaf_pressure_ablation_vacuous"]
        ),
        "local_pressure_ablation_vacuous": bool(
            leaf_pressure_ablation_diagnostics["local_pressure_ablation_vacuous"]
        ),
    }
    reported_run_intent = materialize_tree_run_intent(
        dict(payload.get("config") or {}),
        fixed_leaf_tokens_override=int(
            payload.get(
                "executed_fixed_leaf_tokens",
                getattr(effective_config, "fixed_leaf_tokens", 0),
            )
            or 0
        ),
        baseline_family_override=normalized_family,
    )
    effective_reported_run_intent_diff = intent_diff(
        effective_run_intent,
        reported_run_intent,
    )
    if effective_reported_run_intent_diff:
        raise ValueError(
            "unexpected reported run intent drift: "
            f"{json.dumps(effective_reported_run_intent_diff, sort_keys=True)}"
        )
    payload["reported_run_intent"] = dict(reported_run_intent)
    payload["effective_reported_run_intent_diff"] = dict(
        effective_reported_run_intent_diff
    )
    payload["run_intent_hash"] = str(intent_hash(reported_run_intent))
    payload.update(_run_semantics_metadata(payload))
    payload.update(
        full_doc_baseline_provenance(
            normalized_family,
            objective_weights_active=bool(payload.get("objective_weights_active", False)),
            config_like=payload,
            c2_metric_kind=str(payload.get("c2_metric_kind", "")),
            c2_proxy_metric_kind=str(payload.get("c2_proxy_metric_kind", "")),
            c2_exact_witness_kind=str(payload.get("c2_exact_witness_kind", "")),
            mean_leaves_per_doc=_safe_float(
                payload.get("test_mean_leaves_per_doc", float("nan")),
                default=float("nan"),
            ),
        )
    )
    payload.update(_derived_report_objective_fields(payload))
    exact_sketch_diagnostics = fit.get("exact_sketch_diagnostics")
    if isinstance(exact_sketch_diagnostics, Mapping):
        payload["exact_sketch_diagnostics"] = dict(exact_sketch_diagnostics)
        failure_attribution = dict(
            exact_sketch_diagnostics.get("failure_attribution") or {}
        )
        test_direct_metrics = dict(
            (exact_sketch_diagnostics.get("direct_selection_metrics") or {}).get(
                "test",
                {},
            )
            or {}
        )
        payload["exact_sketch_failure_bucket"] = str(
            failure_attribution.get("bucket", "")
        )
        for key in (
            "tree_model_version",
            "tree_runtime_merge_kind",
            "tree_exact_projected_merge_is_runtime_merge",
            "uses_unified_g_learned_merge",
        ):
            if key in test_direct_metrics:
                payload[key] = test_direct_metrics[key]
        payload["exact_sketch_leaf_gap_score"] = float(
            failure_attribution.get("leaf_gap_score", float("nan"))
        )
        payload["exact_sketch_merge_gap_score"] = float(
            failure_attribution.get("merge_gap_score", float("nan"))
        )
        payload["exact_sketch_theorem_count_decode_gap_score"] = float(
            failure_attribution.get("theorem_count_decode_gap_score", float("nan"))
        )
        payload["exact_sketch_markov_sufficiency_gap_score"] = float(
            failure_attribution.get(
                "markov_sufficiency_gap_score",
                failure_attribution.get("theorem_count_decode_gap_score", float("nan")),
            )
        )
        payload["exact_sketch_phi_not_sufficient_score"] = float(
            failure_attribution.get("phi_not_sufficient_score", float("nan"))
        )
        payload["exact_sketch_phi_not_compositional_score"] = float(
            failure_attribution.get("phi_not_compositional_score", float("nan"))
        )
        payload["exact_sketch_subtree_label_value_gap_score"] = float(
            failure_attribution.get("subtree_label_value_gap_score", float("nan"))
        )
        payload["exact_sketch_internal_label_value_gap_score"] = float(
            failure_attribution.get("internal_label_value_gap_score", float("nan"))
        )
        payload["exact_sketch_readout_gap_score"] = float(
            failure_attribution.get("readout_gap_score", float("nan"))
        )
        payload["root_direct_count_mae"] = float(
            test_direct_metrics.get("root_direct_count_mae", float("nan"))
        )
        payload["test_root_direct_count_mae"] = float(
            test_direct_metrics.get("root_direct_count_mae", float("nan"))
        )
        payload["exact_projected_root_mae"] = float(
            test_direct_metrics.get("exact_projected_root_mae", float("nan"))
        )
        payload["test_exact_projected_root_mae"] = float(
            test_direct_metrics.get("exact_projected_root_mae", float("nan"))
        )
        payload["certified_projected_root_mae"] = float(
            test_direct_metrics.get("certified_projected_root_mae", float("nan"))
        )
        payload["test_certified_projected_root_mae"] = float(
            test_direct_metrics.get("certified_projected_root_mae", float("nan"))
        )
        payload["root_mae_predicted_counts_predicted_endpoints"] = float(
            test_direct_metrics.get(
                "root_mae_predicted_counts_predicted_endpoints",
                float("nan"),
            )
        )
        payload["test_root_mae_predicted_counts_predicted_endpoints"] = float(
            test_direct_metrics.get(
                "root_mae_predicted_counts_predicted_endpoints",
                float("nan"),
            )
        )
        payload["root_mae_oracle_counts_predicted_endpoints"] = float(
            test_direct_metrics.get(
                "root_mae_oracle_counts_predicted_endpoints",
                float("nan"),
            )
        )
        payload["test_root_mae_oracle_counts_predicted_endpoints"] = float(
            test_direct_metrics.get(
                "root_mae_oracle_counts_predicted_endpoints",
                float("nan"),
            )
        )
        payload["root_mae_predicted_counts_oracle_endpoints"] = float(
            test_direct_metrics.get(
                "root_mae_predicted_counts_oracle_endpoints",
                float("nan"),
            )
        )
        payload["test_root_mae_predicted_counts_oracle_endpoints"] = float(
            test_direct_metrics.get(
                "root_mae_predicted_counts_oracle_endpoints",
                float("nan"),
            )
        )
        payload["learned_merger_gap"] = float(
            test_direct_metrics.get("learned_merger_gap", float("nan"))
        )
        payload["test_learned_merger_gap"] = float(
            test_direct_metrics.get("learned_merger_gap", float("nan"))
        )
        payload["leaf_direct_exact_summary_match_rate"] = float(
            test_direct_metrics.get("leaf_direct_exact_match", float("nan"))
        )
        payload["test_leaf_direct_exact_summary_match_rate"] = float(
            test_direct_metrics.get("leaf_direct_exact_match", float("nan"))
        )
        payload["merge_direct_exact_summary_match_rate"] = float(
            test_direct_metrics.get("merge_direct_exact_match", float("nan"))
        )
        payload["test_merge_direct_exact_summary_match_rate"] = float(
            test_direct_metrics.get("merge_direct_exact_match", float("nan"))
        )
        payload["merge_join_bit_accuracy"] = float(
            test_direct_metrics.get("merge_join_bit_accuracy", float("nan"))
        )
        payload["test_merge_join_bit_accuracy"] = float(
            test_direct_metrics.get("merge_join_bit_accuracy", float("nan"))
        )
        payload["leaf_first_accuracy"] = float(
            test_direct_metrics.get("leaf_first_accuracy", float("nan"))
        )
        payload["test_leaf_first_accuracy"] = float(
            test_direct_metrics.get("leaf_first_accuracy", float("nan"))
        )
        payload["leaf_last_accuracy"] = float(
            test_direct_metrics.get("leaf_last_accuracy", float("nan"))
        )
        payload["test_leaf_last_accuracy"] = float(
            test_direct_metrics.get("leaf_last_accuracy", float("nan"))
        )
        payload["merge_first_accuracy"] = float(
            test_direct_metrics.get("merge_first_accuracy", float("nan"))
        )
        payload["test_merge_first_accuracy"] = float(
            test_direct_metrics.get("merge_first_accuracy", float("nan"))
        )
        payload["merge_last_accuracy"] = float(
            test_direct_metrics.get("merge_last_accuracy", float("nan"))
        )
        payload["test_merge_last_accuracy"] = float(
            test_direct_metrics.get("merge_last_accuracy", float("nan"))
        )
        payload["leaf_count_off_by_k_histogram"] = dict(
            test_direct_metrics.get("leaf_count_off_by_k_histogram", {}) or {}
        )
        payload["merge_exact_summary_match_rate_by_depth"] = dict(
            test_direct_metrics.get("merge_exact_summary_match_rate_by_depth", {}) or {}
        )
        payload["phi_merge_alignment"] = float(
            test_direct_metrics.get("phi_merge_alignment", float("nan"))
        )
        payload["phi_within_class_variance"] = float(
            test_direct_metrics.get("phi_within_class_variance", float("nan"))
        )
        payload["phi_between_class_margin"] = float(
            test_direct_metrics.get("phi_between_class_margin", float("nan"))
        )
        payload["phi_pair_same_accuracy"] = float(
            test_direct_metrics.get("phi_pair_same_accuracy", float("nan"))
        )
        payload["phi_pair_diff_accuracy"] = float(
            test_direct_metrics.get("phi_pair_diff_accuracy", float("nan"))
        )
        payload["phi_pair_auc"] = float(
            test_direct_metrics.get("phi_pair_auc", float("nan"))
        )
        payload["phi_replay_same_class_rate"] = float(
            test_direct_metrics.get("phi_replay_same_class_rate", float("nan"))
        )
        payload["task_factorization_gap"] = float(
            test_direct_metrics.get("task_factorization_gap", float("nan"))
        )
        payload["leaf_count_head_entropy_mean"] = float(
            test_direct_metrics.get("leaf_count_head_entropy_mean", float("nan"))
        )
        payload["merge_count_head_entropy_mean"] = float(
            test_direct_metrics.get("merge_count_head_entropy_mean", float("nan"))
        )
        payload["leaf_count_head_margin_mean"] = float(
            test_direct_metrics.get("leaf_count_head_margin_mean", float("nan"))
        )
        payload["merge_count_head_margin_mean"] = float(
            test_direct_metrics.get("merge_count_head_margin_mean", float("nan"))
        )
    if isinstance(fit.get("root_summary_probe_audit"), Mapping):
        payload["root_summary_probe_audit"] = dict(fit.get("root_summary_probe_audit") or {})
    if isinstance(fit.get("stage1_artifact"), Mapping):
        payload["stage1_artifact"] = dict(fit.get("stage1_artifact") or {})
    if isinstance(fit.get("teacher_first_decomposition"), Mapping):
        payload["teacher_first_decomposition"] = dict(
            fit.get("teacher_first_decomposition") or {}
        )
        teacher_first_test = dict(
            (fit.get("teacher_first_decomposition") or {}).get("test") or {}
        )
        payload["stage2_transport_budget"] = float(
            teacher_first_test.get("stage2_transport_budget", float("nan"))
        )
        payload["stage2_leaf_transport_mae"] = float(
            teacher_first_test.get("stage2_leaf_transport_mae", float("nan"))
        )
        payload["stage2_merge_transport_mae"] = float(
            teacher_first_test.get("stage2_merge_transport_mae", float("nan"))
        )
        payload["stage2_fiber_error"] = float(
            teacher_first_test.get("stage2_fiber_error", float("nan"))
        )
        payload["stage2_fiber_pair_same_accuracy"] = float(
            teacher_first_test.get("stage2_fiber_pair_same_accuracy", float("nan"))
        )
        payload["stage2_fiber_pair_diff_accuracy"] = float(
            teacher_first_test.get("stage2_fiber_pair_diff_accuracy", float("nan"))
        )
        payload["stage2_fiber_pair_auc"] = float(
            teacher_first_test.get("stage2_fiber_pair_auc", float("nan"))
        )
        payload["root_measurement_error"] = float(
            teacher_first_test.get("root_measurement_error", float("nan"))
        )
        payload["stage1_substitution_cost"] = float(
            teacher_first_test.get("stage1_substitution_cost", float("nan"))
        )
        payload["teacher_first_total_bound"] = float(
            teacher_first_test.get("teacher_first_total_bound", float("nan"))
        )
    if isinstance(fit.get("training_component_loss_curves"), Mapping):
        payload["training_component_loss_curves"] = {
            str(name): list(values)
            for name, values in dict(fit.get("training_component_loss_curves") or {}).items()
        }
    if isinstance(fit.get("training_component_loss_finals"), Mapping):
        payload["training_component_loss_finals"] = {
            str(name): float(value)
            for name, value in dict(fit.get("training_component_loss_finals") or {}).items()
        }
    if emit_confusion:
        payload["confusion"] = {
            "train": _confusion_payload(
                np.asarray(fit["train_truths"], dtype=np.float64),
                np.asarray(fit["train_preds"], dtype=np.float64),
                support_values=support_values,
            ),
            "val": _confusion_payload(
                np.asarray(fit["val_truths"], dtype=np.float64),
                np.asarray(fit["val_preds"], dtype=np.float64),
                support_values=support_values,
            ),
            "test": _confusion_payload(
                np.asarray(fit["test_truths"], dtype=np.float64),
                np.asarray(fit["test_preds"], dtype=np.float64),
                support_values=support_values,
            ),
        }
    return payload


def _aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def _group_key_float(value: Any) -> float | str:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return "__nan__"
        return float(parsed) if math.isfinite(parsed) else "__nan__"

    grouped: Dict[tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for run in runs:
        normalized_run = _normalize_loaded_run_semantics(run)
        key = (
            str(normalized_run.get("benchmark", "")),
            str(normalized_run.get("cell_id", "")),
            str(normalized_run.get("baseline_family", "")),
            int(normalized_run.get("train_doc_count", 0)),
            int(normalized_run.get("fixed_leaf_tokens", 0)),
            int(normalized_run.get("n_regimes", 0)),
            str(normalized_run.get("segment_density_band", "")),
            str(normalized_run.get("comparison_mode", "")),
            str(normalized_run.get("comparison_semantics_label", "")),
            str(normalized_run.get("run_intent_hash", "")),
            str(normalized_run.get("run_intent_validation_status", "")),
            str(normalized_run.get("parameterization", "")),
            str(normalized_run.get("weighting_scheme", "")),
            float(normalized_run.get("optimization_root_weight", float("nan"))),
            float(normalized_run.get("local_law_c1_weight", float("nan"))),
            float(normalized_run.get("local_law_c2_weight", float("nan"))),
            float(normalized_run.get("local_law_c3_weight", float("nan"))),
            str(normalized_run.get("task_objective_weight_source", "")),
            float(normalized_run.get("proxy_schedule_consistency_weight", 0.0)),
            str(normalized_run.get("c2_metric_kind", "")),
            str(normalized_run.get("backend_name", "")),
            str(normalized_run.get("backend_package", "")),
            str(normalized_run.get("backend_version", "")),
            str(normalized_run.get("operator_class", "")),
            str(normalized_run.get("operator_evidence_status", "")),
            bool(normalized_run.get("theorem_relevance", False)),
            bool(normalized_run.get("objective_weights_active", False)),
            str(normalized_run.get("tree_root_supervision_kind", "")),
            str(normalized_run.get("aligned_sketch_surface", "")),
            str(normalized_run.get("internal_supervision_kind", "")),
            float(normalized_run.get("internal_label_rate", 0.0)),
            bool(normalized_run.get("leaf_exact_supervision", False)),
            str(normalized_run.get("summary_spec_name", "")),
            int(normalized_run.get("slot_count", 0)),
            int(normalized_run.get("tree_theorem_count_dim", 0)),
            int(normalized_run.get("tree_theorem_first_dim", 0)),
            int(normalized_run.get("tree_theorem_last_dim", 0)),
            str(normalized_run.get("tree_theorem_count_head_mode", "")),
            float(normalized_run.get("leaf_label_rate", 1.0)),
            int(normalized_run.get("tree_leaf_fno_width", 0)),
            int(normalized_run.get("tree_leaf_fno_n_modes", 0)),
            int(normalized_run.get("tree_leaf_fno_n_layers", 0)),
            float(normalized_run.get("tree_aux_doc_sequence_fraction", 0.0)),
            str(normalized_run.get("config_label", "")),
            str(normalized_run.get("tuning_stage", "")),
            bool(normalized_run.get("test_metrics_hidden_during_selection", False)),
            str(normalized_run.get("study_name", "")),
            str(normalized_run.get("study_axis", "")),
            normalized_run.get("axis_value", ""),
            str(normalized_run.get("locked_tree_neural_config_label", "")),
            str(normalized_run.get("selection_metric", "")),
            int(normalized_run.get("budget_total_calls", 0)),
            float(normalized_run.get("budget_total_calls_per_doc", 0.0)),
            _group_key_float(normalized_run.get("mass_target_per_doc", float("nan"))),
            float(normalized_run.get("full_doc_budget_share", 1.0)),
            str(normalized_run.get("doc_consumption_mode", "")),
            str(normalized_run.get("local_split_mode", "")),
            str(normalized_run.get("local_allocation_policy", "")),
            str(normalized_run.get("package_semantics", "")),
            float(normalized_run.get("depth_discount_gamma", 1.0)),
        )
        grouped.setdefault(key, []).append(normalized_run)

    aggregate_rows: List[Dict[str, Any]] = []
    for (
        benchmark,
        cell_id,
        baseline_family,
        train_doc_count,
        fixed_leaf_tokens,
        n_regimes,
        segment_density_band,
        comparison_mode,
        comparison_semantics_label,
        run_intent_hash,
        run_intent_validation_status,
        parameterization,
        weighting_scheme,
        optimization_root_weight,
        local_law_c1_weight,
        local_law_c2_weight,
        local_law_c3_weight,
        task_objective_weight_source,
        proxy_schedule_consistency_weight,
        c2_metric_kind,
        backend_name,
        backend_package,
        backend_version,
        operator_class,
        operator_evidence_status,
        theorem_relevance,
        objective_weights_active,
        tree_root_supervision_kind,
        aligned_sketch_surface,
        internal_supervision_kind,
        internal_label_rate,
        leaf_exact_supervision,
        summary_spec_name,
        slot_count,
        tree_theorem_count_dim,
        tree_theorem_first_dim,
        tree_theorem_last_dim,
        tree_theorem_count_head_mode,
        leaf_label_rate,
        tree_leaf_fno_width,
        tree_leaf_fno_n_modes,
        tree_leaf_fno_n_layers,
        tree_aux_doc_sequence_fraction,
        config_label,
        tuning_stage,
        test_metrics_hidden_during_selection,
        study_name,
        study_axis,
        axis_value,
        locked_tree_neural_config_label,
        selection_metric,
        budget_total_calls,
        budget_total_calls_per_doc,
        mass_target_per_doc,
        full_doc_budget_share,
        doc_consumption_mode,
        local_split_mode,
        local_allocation_policy,
        package_semantics,
        depth_discount_gamma,
    ), group in sorted(
        grouped.items()
    ):
        train_root_mae = np.asarray(
            [float(item.get("train_root_mae", float("nan"))) for item in group],
            dtype=np.float64,
        )
        val_root_mae = np.asarray(
            [float(item.get("val_root_mae", float("nan"))) for item in group],
            dtype=np.float64,
        )
        test_root_mae = np.asarray(
            [float(item.get("test_root_mae", float("nan"))) for item in group],
            dtype=np.float64,
        )
        train_exact_match = np.asarray(
            [float(item.get("train_exact_match_rate", float("nan"))) for item in group],
            dtype=np.float64,
        )
        val_exact_match = np.asarray(
            [float(item.get("val_exact_match_rate", float("nan"))) for item in group],
            dtype=np.float64,
        )
        test_exact_match = np.asarray(
            [float(item.get("test_exact_match_rate", float("nan"))) for item in group],
            dtype=np.float64,
        )
        test_leaf_mae = np.asarray(
            [float(item.get("test_leaf_mae", float("nan"))) for item in group],
            dtype=np.float64,
        )
        c2_count_drift_r1 = np.asarray(
            [
                float(
                    item.get(
                        "test_c2_count_drift_r1_mae",
                        item.get("test_c2_idempotence_mae", float("nan")),
                    )
                )
                for item in group
            ],
            dtype=np.float64,
        )
        c2_count_drift_r2 = np.asarray(
            [float(item.get("test_c2_count_drift_r2_mae", float("nan"))) for item in group],
            dtype=np.float64,
        )
        c2_count_drift_r4 = np.asarray(
            [float(item.get("test_c2_count_drift_r4_mae", float("nan"))) for item in group],
            dtype=np.float64,
        )
        c2_root_count_drift_r1 = np.asarray(
            [
                float(item.get("test_c2_root_count_drift_r1_mae", float("nan")))
                for item in group
            ],
            dtype=np.float64,
        )
        c2_root_count_drift_r2 = np.asarray(
            [
                float(item.get("test_c2_root_count_drift_r2_mae", float("nan")))
                for item in group
            ],
            dtype=np.float64,
        )
        c2_root_count_drift_r4 = np.asarray(
            [
                float(item.get("test_c2_root_count_drift_r4_mae", float("nan")))
                for item in group
            ],
            dtype=np.float64,
        )
        c2_state_replay = np.asarray(
            [float(item.get("test_c2_state_replay_mse", float("nan"))) for item in group],
            dtype=np.float64,
        )
        test_merge_mae = np.asarray(
            [float(item.get("test_merge_mae", float("nan"))) for item in group],
            dtype=np.float64,
        )
        test_schedule_spread_mean = np.asarray(
            [float(item.get("test_schedule_spread_mean", float("nan"))) for item in group],
            dtype=np.float64,
        )
        elapsed_s = np.asarray(
            [float(item.get("elapsed_s", float("nan"))) for item in group],
            dtype=np.float64,
        )
        val_unweighted_full_law_objective = np.asarray(
            [
                float(item.get("val_unweighted_full_law_objective", float("nan")))
                for item in group
            ],
            dtype=np.float64,
        )
        val_unweighted_active_objective = np.asarray(
            [
                float(item.get("val_unweighted_active_objective", float("nan")))
                for item in group
            ],
            dtype=np.float64,
        )
        test_unweighted_full_law_objective = np.asarray(
            [
                float(item.get("test_unweighted_full_law_objective", float("nan")))
                for item in group
            ],
            dtype=np.float64,
        )
        test_unweighted_active_objective = np.asarray(
            [
                float(item.get("test_unweighted_active_objective", float("nan")))
                for item in group
            ],
            dtype=np.float64,
        )
        selection_metric_values = np.asarray(
            [
                _safe_float(
                    (item.get("fit_diagnostics") or {}).get("selection_metric_value", float("nan"))
                )
                for item in group
            ],
            dtype=np.float64,
        )
        best_epochs = np.asarray(
            [
                _safe_float((item.get("fit_diagnostics") or {}).get("best_epoch", float("nan")))
                for item in group
            ],
            dtype=np.float64,
        )
        c2_count_drift_r1_has_finite = bool(np.isfinite(c2_count_drift_r1).any())
        c2_count_drift_r2_has_finite = bool(np.isfinite(c2_count_drift_r2).any())
        c2_count_drift_r4_has_finite = bool(np.isfinite(c2_count_drift_r4).any())
        c2_root_count_drift_r1_has_finite = bool(np.isfinite(c2_root_count_drift_r1).any())
        c2_root_count_drift_r2_has_finite = bool(np.isfinite(c2_root_count_drift_r2).any())
        c2_root_count_drift_r4_has_finite = bool(np.isfinite(c2_root_count_drift_r4).any())
        c2_state_replay_has_finite = bool(np.isfinite(c2_state_replay).any())
        test_leaf_mae_has_finite = bool(np.isfinite(test_leaf_mae).any())
        test_merge_mae_has_finite = bool(np.isfinite(test_merge_mae).any())
        test_schedule_spread_mean_has_finite = bool(
            np.isfinite(test_schedule_spread_mean).any()
        )
        elapsed_s_has_finite = bool(np.isfinite(elapsed_s).any())
        budget_total_calls_used = np.asarray(
            [float(item.get("budget_total_calls_used", float("nan"))) for item in group],
            dtype=np.float64,
        )
        budget_utilization = np.asarray(
            [float(item.get("budget_utilization", float("nan"))) for item in group],
            dtype=np.float64,
        )
        effective_full_doc_mass_total = np.asarray(
            [
                float(item.get("effective_full_doc_mass_total", float("nan")))
                for item in group
            ],
            dtype=np.float64,
        )
        effective_full_doc_mass_per_doc = np.asarray(
            [
                float(item.get("effective_full_doc_mass_per_doc", float("nan")))
                for item in group
            ],
            dtype=np.float64,
        )
        document_mass_share = np.asarray(
            [float(item.get("document_mass_share", float("nan"))) for item in group],
            dtype=np.float64,
        )
        leaf_mass_share = np.asarray(
            [float(item.get("leaf_mass_share", float("nan"))) for item in group],
            dtype=np.float64,
        )
        internal_mass_share = np.asarray(
            [float(item.get("internal_mass_share", float("nan"))) for item in group],
            dtype=np.float64,
        )
        doc_touch_rate = np.asarray(
            [float(item.get("doc_touch_rate", float("nan"))) for item in group],
            dtype=np.float64,
        )
        mean_labels_per_touched_doc = np.asarray(
            [
                float(item.get("mean_labels_per_touched_doc", float("nan")))
                for item in group
            ],
            dtype=np.float64,
        )
        budget_total_calls_used_has_finite = bool(np.isfinite(budget_total_calls_used).any())
        budget_utilization_has_finite = bool(np.isfinite(budget_utilization).any())
        effective_full_doc_mass_total_has_finite = bool(
            np.isfinite(effective_full_doc_mass_total).any()
        )
        effective_full_doc_mass_per_doc_has_finite = bool(
            np.isfinite(effective_full_doc_mass_per_doc).any()
        )
        document_mass_share_has_finite = bool(np.isfinite(document_mass_share).any())
        leaf_mass_share_has_finite = bool(np.isfinite(leaf_mass_share).any())
        internal_mass_share_has_finite = bool(np.isfinite(internal_mass_share).any())
        doc_touch_rate_has_finite = bool(np.isfinite(doc_touch_rate).any())
        mean_labels_per_touched_doc_has_finite = bool(
            np.isfinite(mean_labels_per_touched_doc).any()
        )
        val_unweighted_full_law_objective_has_finite = bool(
            np.isfinite(val_unweighted_full_law_objective).any()
        )
        val_unweighted_active_objective_has_finite = bool(
            np.isfinite(val_unweighted_active_objective).any()
        )
        test_unweighted_full_law_objective_has_finite = bool(
            np.isfinite(test_unweighted_full_law_objective).any()
        )
        test_unweighted_active_objective_has_finite = bool(
            np.isfinite(test_unweighted_active_objective).any()
        )
        selection_metric_values_has_finite = bool(np.isfinite(selection_metric_values).any())
        best_epochs_has_finite = bool(np.isfinite(best_epochs).any())
        segment_min = int(group[0].get("segment_min", 0))
        segment_max = int(group[0].get("segment_max", 0))
        row = {
            "benchmark": str(benchmark),
            "cell_id": str(cell_id),
            "baseline_family": str(baseline_family),
            "train_doc_count": int(train_doc_count),
            "fixed_leaf_tokens": int(fixed_leaf_tokens),
            "n_regimes": int(n_regimes),
            "segment_density_band": str(segment_density_band),
            "segment_min": int(segment_min),
            "segment_max": int(segment_max),
            "train_mean_leaves_per_doc": float(
                group[0].get("train_mean_leaves_per_doc", float("nan"))
            ),
            "val_mean_leaves_per_doc": float(
                group[0].get("val_mean_leaves_per_doc", float("nan"))
            ),
            "test_mean_leaves_per_doc": float(
                group[0].get("test_mean_leaves_per_doc", float("nan"))
            ),
            "n_runs": int(len(group)),
            **_summary_stats(train_root_mae, prefix="train_root_mae"),
            **_summary_stats(val_root_mae, prefix="val_root_mae"),
            **_summary_stats(test_root_mae, prefix="test_root_mae"),
            **_summary_stats(train_exact_match, prefix="train_exact_match_rate"),
            **_summary_stats(val_exact_match, prefix="val_exact_match_rate"),
            **_summary_stats(test_exact_match, prefix="test_exact_match_rate"),
            **_summary_stats(elapsed_s, prefix="elapsed_s"),
            "val_unweighted_full_law_objective_mean": (
                float(np.nanmean(val_unweighted_full_law_objective))
                if val_unweighted_full_law_objective_has_finite
                else float("nan")
            ),
            "val_unweighted_full_law_objective_std": (
                float(np.nanstd(val_unweighted_full_law_objective))
                if val_unweighted_full_law_objective_has_finite
                else float("nan")
            ),
            "val_unweighted_active_objective_mean": (
                float(np.nanmean(val_unweighted_active_objective))
                if val_unweighted_active_objective_has_finite
                else float("nan")
            ),
            "val_unweighted_active_objective_std": (
                float(np.nanstd(val_unweighted_active_objective))
                if val_unweighted_active_objective_has_finite
                else float("nan")
            ),
            "test_unweighted_full_law_objective_mean": (
                float(np.nanmean(test_unweighted_full_law_objective))
                if test_unweighted_full_law_objective_has_finite
                else float("nan")
            ),
            "test_unweighted_full_law_objective_std": (
                float(np.nanstd(test_unweighted_full_law_objective))
                if test_unweighted_full_law_objective_has_finite
                else float("nan")
            ),
            "test_unweighted_active_objective_mean": (
                float(np.nanmean(test_unweighted_active_objective))
                if test_unweighted_active_objective_has_finite
                else float("nan")
            ),
            "test_unweighted_active_objective_std": (
                float(np.nanstd(test_unweighted_active_objective))
                if test_unweighted_active_objective_has_finite
                else float("nan")
            ),
            "test_leaf_mae_mean": (
                float(np.nanmean(test_leaf_mae))
                if test_leaf_mae_has_finite
                else float("nan")
            ),
            "test_leaf_mae_std": (
                float(np.nanstd(test_leaf_mae))
                if test_leaf_mae_has_finite
                else float("nan")
            ),
            "test_c2_count_drift_r1_mae_mean": (
                float(np.nanmean(c2_count_drift_r1))
                if c2_count_drift_r1_has_finite
                else float("nan")
            ),
            "test_c2_count_drift_r1_mae_std": (
                float(np.nanstd(c2_count_drift_r1))
                if c2_count_drift_r1_has_finite
                else float("nan")
            ),
            "test_c2_count_drift_r2_mae_mean": (
                float(np.nanmean(c2_count_drift_r2))
                if c2_count_drift_r2_has_finite
                else float("nan")
            ),
            "test_c2_count_drift_r2_mae_std": (
                float(np.nanstd(c2_count_drift_r2))
                if c2_count_drift_r2_has_finite
                else float("nan")
            ),
            "test_c2_count_drift_r4_mae_mean": (
                float(np.nanmean(c2_count_drift_r4))
                if c2_count_drift_r4_has_finite
                else float("nan")
            ),
            "test_c2_count_drift_r4_mae_std": (
                float(np.nanstd(c2_count_drift_r4))
                if c2_count_drift_r4_has_finite
                else float("nan")
            ),
            "test_c2_root_count_drift_r1_mae_mean": (
                float(np.nanmean(c2_root_count_drift_r1))
                if c2_root_count_drift_r1_has_finite
                else float("nan")
            ),
            "test_c2_root_count_drift_r1_mae_std": (
                float(np.nanstd(c2_root_count_drift_r1))
                if c2_root_count_drift_r1_has_finite
                else float("nan")
            ),
            "test_c2_root_count_drift_r2_mae_mean": (
                float(np.nanmean(c2_root_count_drift_r2))
                if c2_root_count_drift_r2_has_finite
                else float("nan")
            ),
            "test_c2_root_count_drift_r2_mae_std": (
                float(np.nanstd(c2_root_count_drift_r2))
                if c2_root_count_drift_r2_has_finite
                else float("nan")
            ),
            "test_c2_root_count_drift_r4_mae_mean": (
                float(np.nanmean(c2_root_count_drift_r4))
                if c2_root_count_drift_r4_has_finite
                else float("nan")
            ),
            "test_c2_root_count_drift_r4_mae_std": (
                float(np.nanstd(c2_root_count_drift_r4))
                if c2_root_count_drift_r4_has_finite
                else float("nan")
            ),
            "test_c2_idempotence_mae_mean": (
                float(np.nanmean(c2_count_drift_r1))
                if c2_count_drift_r1_has_finite
                else float("nan")
            ),
            "test_c2_idempotence_mae_std": (
                float(np.nanstd(c2_count_drift_r1))
                if c2_count_drift_r1_has_finite
                else float("nan")
            ),
            "test_c2_state_replay_mse_mean": (
                float(np.nanmean(c2_state_replay))
                if c2_state_replay_has_finite
                else float("nan")
            ),
            "test_c2_state_replay_mse_std": (
                float(np.nanstd(c2_state_replay))
                if c2_state_replay_has_finite
                else float("nan")
            ),
            "test_merge_mae_mean": (
                float(np.nanmean(test_merge_mae))
                if test_merge_mae_has_finite
                else float("nan")
            ),
            "test_merge_mae_std": (
                float(np.nanstd(test_merge_mae))
                if test_merge_mae_has_finite
                else float("nan")
            ),
            "test_schedule_spread_mean_mean": (
                float(np.nanmean(test_schedule_spread_mean))
                if test_schedule_spread_mean_has_finite
                else float("nan")
            ),
            "test_schedule_spread_mean_std": (
                float(np.nanstd(test_schedule_spread_mean))
                if test_schedule_spread_mean_has_finite
                else float("nan")
            ),
            "elapsed_s_mean": (
                float(np.nanmean(elapsed_s))
                if elapsed_s_has_finite
                else float("nan")
            ),
            "elapsed_s_std": (
                float(np.nanstd(elapsed_s))
                if elapsed_s_has_finite
                else float("nan")
            ),
            "parameterization": str(parameterization),
            "weighting_scheme": str(weighting_scheme),
            "optimization_root_weight": float(optimization_root_weight),
            "local_law_c1_weight": float(local_law_c1_weight),
            "local_law_c2_weight": float(local_law_c2_weight),
            "local_law_c3_weight": float(local_law_c3_weight),
            "task_objective_weight_source": str(task_objective_weight_source),
            "proxy_schedule_consistency_weight": float(
                proxy_schedule_consistency_weight
            ),
            "c2_metric_kind": str(c2_metric_kind),
            "c2_proxy_metric_kind": str(group[0].get("c2_proxy_metric_kind", "")),
            "c2_exact_witness_kind": str(group[0].get("c2_exact_witness_kind", "")),
            "backend_name": str(backend_name),
            "backend_package": str(backend_package),
            "backend_version": str(backend_version),
            "operator_class": str(operator_class),
            "operator_evidence_status": str(operator_evidence_status),
            "theorem_relevance": bool(theorem_relevance),
            "objective_weights_active": bool(objective_weights_active),
            "family_api_group": str(group[0].get("family_api_group", "")),
            "family_runner_kind": str(group[0].get("family_runner_kind", "")),
            "shared_framework_group": str(group[0].get("shared_framework_group", "")),
            "law_contract_version": str(group[0].get("law_contract_version", "")),
            "law_alignment_status": str(group[0].get("law_alignment_status", "")),
            "law_contract_gap_count": int(group[0].get("law_contract_gap_count", 0)),
            "c2_nontriviality_status": str(
                group[0].get("c2_nontriviality_status", "")
            ),
            "tree_root_supervision_kind": str(tree_root_supervision_kind),
            "aligned_sketch_surface": str(aligned_sketch_surface),
            "internal_supervision_kind": str(internal_supervision_kind),
            "internal_label_rate": float(internal_label_rate),
            "leaf_exact_supervision": bool(leaf_exact_supervision),
            "summary_spec_name": str(summary_spec_name),
            "slot_count": int(slot_count),
            "tree_theorem_count_dim": int(tree_theorem_count_dim),
            "tree_theorem_first_dim": int(tree_theorem_first_dim),
            "tree_theorem_last_dim": int(tree_theorem_last_dim),
            "tree_theorem_count_head_mode": str(tree_theorem_count_head_mode),
            "tree_c2_mode": str(group[0].get("tree_c2_mode", "")),
            "leaf_label_rate": float(leaf_label_rate),
            "tree_leaf_fno_width": int(tree_leaf_fno_width),
            "tree_leaf_fno_n_modes": int(tree_leaf_fno_n_modes),
            "tree_leaf_fno_n_layers": int(tree_leaf_fno_n_layers),
            "tree_aux_doc_sequence_fraction": float(tree_aux_doc_sequence_fraction),
            "comparison_mode": str(comparison_mode),
            "comparison_semantics": str(group[0].get("comparison_semantics", "")),
            "comparison_semantics_label": str(comparison_semantics_label),
            "run_intent_hash": str(run_intent_hash),
            "run_intent_validation_status": str(run_intent_validation_status),
            "comparison_surface_snapshot": dict(
                group[0].get("comparison_surface_snapshot") or {}
            ),
            "comparison_surface_diff": dict(
                group[0].get("comparison_surface_diff") or {}
            ),
            "legacy_semantics": any(
                bool(item.get("legacy_semantics", False)) for item in group
            ),
            "config_label": str(config_label),
            "tuning_stage": str(tuning_stage),
            "test_metrics_hidden_during_selection": bool(
                test_metrics_hidden_during_selection
            ),
            "study_name": str(study_name),
            "study_axis": str(study_axis),
            "axis_value": axis_value,
            "locked_tree_neural_config_label": str(locked_tree_neural_config_label),
            "selection_metric": str(selection_metric),
            "selection_metric_value_mean": (
                float(np.nanmean(selection_metric_values))
                if selection_metric_values_has_finite
                else float("nan")
            ),
            "best_epoch_mean": (
                float(np.nanmean(best_epochs))
                if best_epochs_has_finite
                else float("nan")
            ),
            "budget_total_calls": int(budget_total_calls),
            "budget_total_calls_per_doc": float(budget_total_calls_per_doc),
            "mass_target_per_doc": (
                float(mass_target_per_doc)
                if mass_target_per_doc != "__nan__"
                else float("nan")
            ),
            "full_doc_budget_share": float(full_doc_budget_share),
            "full_doc_calls_requested": int(group[0].get("full_doc_calls_requested", 0)),
            "full_doc_calls_total": int(group[0].get("full_doc_calls_total", 0)),
            "local_calls_requested": int(group[0].get("local_calls_requested", 0)),
            "local_calls_total": int(group[0].get("local_calls_total", 0)),
            "doc_consumption_mode": str(doc_consumption_mode),
            "local_split_mode": str(local_split_mode),
            "local_allocation_policy": str(local_allocation_policy),
            "package_semantics": str(package_semantics),
            "depth_discount_gamma": float(depth_discount_gamma),
            "budget_total_calls_used_mean": (
                float(np.nanmean(budget_total_calls_used))
                if budget_total_calls_used_has_finite
                else float("nan")
            ),
            "budget_total_calls_used_std": (
                float(np.nanstd(budget_total_calls_used))
                if budget_total_calls_used_has_finite
                else float("nan")
            ),
            "budget_utilization_mean": (
                float(np.nanmean(budget_utilization))
                if budget_utilization_has_finite
                else float("nan")
            ),
            "budget_utilization_std": (
                float(np.nanstd(budget_utilization))
                if budget_utilization_has_finite
                else float("nan")
            ),
            "effective_full_doc_mass_total_mean": (
                float(np.nanmean(effective_full_doc_mass_total))
                if effective_full_doc_mass_total_has_finite
                else float("nan")
            ),
            "effective_full_doc_mass_total_std": (
                float(np.nanstd(effective_full_doc_mass_total))
                if effective_full_doc_mass_total_has_finite
                else float("nan")
            ),
            "effective_full_doc_mass_per_doc_mean": (
                float(np.nanmean(effective_full_doc_mass_per_doc))
                if effective_full_doc_mass_per_doc_has_finite
                else float("nan")
            ),
            "effective_full_doc_mass_per_doc_std": (
                float(np.nanstd(effective_full_doc_mass_per_doc))
                if effective_full_doc_mass_per_doc_has_finite
                else float("nan")
            ),
            "document_mass_share_mean": (
                float(np.nanmean(document_mass_share))
                if document_mass_share_has_finite
                else float("nan")
            ),
            "leaf_mass_share_mean": (
                float(np.nanmean(leaf_mass_share))
                if leaf_mass_share_has_finite
                else float("nan")
            ),
            "internal_mass_share_mean": (
                float(np.nanmean(internal_mass_share))
                if internal_mass_share_has_finite
                else float("nan")
            ),
            "doc_touch_rate_mean": (
                float(np.nanmean(doc_touch_rate))
                if doc_touch_rate_has_finite
                else float("nan")
            ),
            "mean_labels_per_touched_doc_mean": (
                float(np.nanmean(mean_labels_per_touched_doc))
                if mean_labels_per_touched_doc_has_finite
                else float("nan")
            ),
            "legacy_semantics_reason": ";".join(
                sorted(
                    {
                        str(item.get("legacy_semantics_reason", "")).strip()
                        for item in group
                        if str(item.get("legacy_semantics_reason", "")).strip()
                    }
                )
            ),
            "semantics_version": str(group[0].get("semantics_version", "")),
            "objective_variant": str(group[0].get("objective_variant", "")),
            "device_requested": str(group[0].get("device_requested", "")),
            "device_resolved": str(group[0].get("device_resolved", "")),
        }
        aggregate_rows.append(row)

    diagnostic_readout = {"status": "insufficient_data"}
    unique_cells = {str(row.get("cell_id", "")) for row in aggregate_rows}
    if aggregate_rows and len(unique_cells) <= 1:
        family_rows: Dict[str, List[Dict[str, Any]]] = {}
        for row in aggregate_rows:
            family_rows.setdefault(str(row["baseline_family"]), []).append(row)
        for rows in family_rows.values():
            rows.sort(key=lambda item: int(item["train_doc_count"]))
        fno_rows = family_rows.get("official_fno_sumlen", []) or family_rows.get("official_fno", [])
        control_rows = [
            row
            for row in aggregate_rows
            if str(row["baseline_family"])
            in {
                "cnn1d",
                "mlp_bigram",
                "palette_block_exact",
                "raw_token_ngram_ridge",
            }
        ]
        if fno_rows and control_rows:
            fno_base = fno_rows[0]
            fno_best = min(
                fno_rows,
                key=lambda item: float(item["test_root_mae_mean"]),
            )
            control_best = min(
                control_rows,
                key=lambda item: float(item["test_root_mae_mean"]),
            )
            gap_to_control = float(fno_best["test_root_mae_mean"]) - float(
                control_best["test_root_mae_mean"]
            )
            data_scale_gain = float(fno_base["test_root_mae_mean"]) - float(
                fno_best["test_root_mae_mean"]
            )
            fno_seed_std = float(fno_best["test_root_mae_std"])
            diagnosis = "mixed_or_unclear"
            if float(control_best["test_exact_match_rate_mean"]) >= 0.98 and float(
                fno_best["test_exact_match_rate_mean"]
            ) < 0.95:
                diagnosis = "fno_readout_or_architecture_bottleneck"
            elif data_scale_gain >= 0.02 and gap_to_control <= 0.02:
                diagnosis = "primarily_data_scale"
            elif gap_to_control > 0.0 and fno_seed_std >= 0.5 * gap_to_control:
                diagnosis = "optimization_stability"
            diagnostic_readout = {
                "status": str(diagnosis),
                "fno_best_train_doc_count": int(fno_best["train_doc_count"]),
                "fno_best_root_mae_mean": float(fno_best["test_root_mae_mean"]),
                "fno_base_root_mae_mean": float(fno_base["test_root_mae_mean"]),
                "best_control_family": str(control_best["baseline_family"]),
                "best_control_root_mae_mean": float(control_best["test_root_mae_mean"]),
                "best_control_exact_match_rate_mean": float(
                    control_best["test_exact_match_rate_mean"]
                ),
                "fno_seed_std_at_best": float(fno_seed_std),
                "fno_data_scale_gain": float(data_scale_gain),
                "gap_to_best_control": float(gap_to_control),
            }
    elif len(unique_cells) > 1:
        diagnostic_readout = {"status": "grid_mode_use_grid_diagnostic_summary"}

    return {
        "aggregate_rows": aggregate_rows,
        "diagnostic_readout": diagnostic_readout,
    }


def _attach_markov_witness_gap_fields(
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in aggregate_rows]
    exact_lookup: Dict[tuple[str, int], Dict[str, Any]] = {}
    ridge_lookup: Dict[tuple[str, int], Dict[str, Any]] = {}
    by_family: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    by_family_objective: Dict[tuple[str, str, int, str], Dict[str, Any]] = {}

    for row in rows:
        key = (str(row.get("cell_id", "")), int(row.get("train_doc_count", 0)))
        family = str(row.get("baseline_family", ""))
        if family == "palette_block_exact":
            exact_lookup[key] = row
        if family == "ridge_control":
            ridge_lookup[key] = row
        by_family.setdefault((str(row.get("cell_id", "")), family), []).append(row)
        by_family_objective[
            (
                str(row.get("cell_id", "")),
                family,
                int(row.get("train_doc_count", 0)),
                str(row.get("objective_variant", "")),
            )
        ] = row

    for family_rows in by_family.values():
        family_rows.sort(key=lambda row: int(row.get("train_doc_count", 0)))

    for row in rows:
        key = (str(row.get("cell_id", "")), int(row.get("train_doc_count", 0)))
        exact_row = exact_lookup.get(key)
        ridge_row = ridge_lookup.get(key)
        row["gap_to_exact_witness"] = (
            float(row.get("test_root_mae_mean", float("nan")))
            - float(exact_row.get("test_root_mae_mean", float("nan")))
            if exact_row is not None
            else float("nan")
        )
        row["gap_to_ridge_control"] = (
            float(row.get("test_root_mae_mean", float("nan")))
            - float(ridge_row.get("test_root_mae_mean", float("nan")))
            if ridge_row is not None
            else float("nan")
        )
        row["train_val_gap"] = float(
            float(row.get("val_root_mae_mean", float("nan")))
            - float(row.get("train_root_mae_mean", float("nan")))
        )
        row["val_test_gap"] = float(
            float(row.get("test_root_mae_mean", float("nan")))
            - float(row.get("val_root_mae_mean", float("nan")))
        )

        family_key = (str(row.get("cell_id", "")), str(row.get("baseline_family", "")))
        family_rows = by_family.get(family_key, [])
        larger_rows = [
            candidate
            for candidate in family_rows
            if int(candidate.get("train_doc_count", 0)) > int(row.get("train_doc_count", 0))
        ]
        scaling_gain = float("nan")
        if larger_rows:
            best_larger = min(
                larger_rows,
                key=lambda candidate: float(candidate.get("test_root_mae_mean", float("inf"))),
            )
            scaling_gain = float(row.get("test_root_mae_mean", float("nan"))) - float(
                best_larger.get("test_root_mae_mean", float("nan"))
            )
        row["scaling_gain_available"] = float(scaling_gain)

        family = str(row.get("baseline_family", ""))
        if family in {"palette_block_exact", "ridge_control"}:
            row["cause_code"] = "witness_control"
            continue

        exact_failed = (
            exact_row is not None
            and float(exact_row.get("test_root_mae_mean", float("nan"))) > 1e-3
        )
        ridge_failed = (
            ridge_row is not None
            and float(ridge_row.get("test_root_mae_mean", float("nan"))) > 1e-3
        )
        if exact_failed and ridge_failed:
            row["cause_code"] = "information_barrier"
            continue

        simpler_row = by_family_objective.get(
            (
                str(row.get("cell_id", "")),
                family,
                int(row.get("train_doc_count", 0)),
                "count_ce_only",
            )
        )
        if (
            simpler_row is not None
            and str(row.get("objective_variant", "")) != "count_ce_only"
            and float(row.get("test_root_mae_mean", float("nan")))
                > float(simpler_row.get("test_root_mae_mean", float("nan"))) + 1e-9
        ):
            row["cause_code"] = "objective_mismatch"
            continue

        requested_cuda = str(row.get("device_requested", "")).strip().lower() == "cuda"
        resolved_cpu = "cpu" in str(row.get("device_resolved", "")).strip().lower()
        if requested_cuda and resolved_cpu:
            row["cause_code"] = "implementation_path_issue"
            continue

        gap_to_ridge = float(row.get("gap_to_ridge_control", float("nan")))
        train_root = float(row.get("train_root_mae_mean", float("nan")))
        if scaling_gain == scaling_gain and scaling_gain > 0.02 and (
            (gap_to_ridge == gap_to_ridge and gap_to_ridge > 0.02)
            or float(row.get("train_val_gap", float("nan"))) > 0.0
            or float(row.get("val_test_gap", float("nan"))) > 0.0
        ):
            row["cause_code"] = "optimization_limit"
            continue

        if gap_to_ridge == gap_to_ridge and gap_to_ridge > 0.05 and train_root > 0.05:
            row["cause_code"] = "representation_limit"
            continue

        row["cause_code"] = "mixed_or_unclear"

    return rows


def _selection_metric_curve_summary(
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "baseline_family": str(row.get("baseline_family", "")),
            "train_doc_count": int(row.get("train_doc_count", 0)),
            "objective_variant": str(row.get("objective_variant", "")),
            "selection_metric": str(row.get("selection_metric", "")),
            "selection_metric_value_mean": float(
                row.get("selection_metric_value_mean", float("nan"))
            ),
            "best_epoch_mean": float(row.get("best_epoch_mean", float("nan"))),
        }
        for row in aggregate_rows
    ]


def _backend_device_summary(
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "baseline_family": str(row.get("baseline_family", "")),
            "train_doc_count": int(row.get("train_doc_count", 0)),
            "backend_name": str(row.get("backend_name", "")),
            "operator_class": str(row.get("operator_class", "")),
            "device_requested": str(row.get("device_requested", "")),
            "device_resolved": str(row.get("device_resolved", "")),
            "objective_variant": str(row.get("objective_variant", "")),
            "cause_code": str(row.get("cause_code", "")),
        }
        for row in aggregate_rows
    ]


def _flatten_run_rows(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        normalized_run = _normalize_loaded_run_semantics(run)
        fit_diag = dict(run.get("fit_diagnostics") or {})
        row = {
            "benchmark": str(normalized_run.get("benchmark", "")),
            "cell_id": str(normalized_run.get("cell_id", "")),
            "hardness_grid": str(normalized_run.get("hardness_grid", "")),
            "degenerate_benchmark": bool(normalized_run.get("degenerate_benchmark", False)),
            "baseline_family": str(normalized_run.get("baseline_family", "")),
            "seed": int(normalized_run.get("seed", 0)),
            "train_doc_count": int(normalized_run.get("train_doc_count", 0)),
            "train_size_multiplier": float(normalized_run.get("train_size_multiplier", 0.0)),
            "fixed_leaf_tokens": int(normalized_run.get("fixed_leaf_tokens", 0)),
            "train_mean_leaves_per_doc": float(
                normalized_run.get("train_mean_leaves_per_doc", float("nan"))
            ),
            "val_mean_leaves_per_doc": float(
                normalized_run.get("val_mean_leaves_per_doc", float("nan"))
            ),
            "test_mean_leaves_per_doc": float(
                normalized_run.get("test_mean_leaves_per_doc", float("nan"))
            ),
            "n_regimes": int(normalized_run.get("n_regimes", 0)),
            "segment_density_band": str(normalized_run.get("segment_density_band", "")),
            "segment_min": int(normalized_run.get("segment_min", 0)),
            "segment_max": int(normalized_run.get("segment_max", 0)),
            "train_root_mae": float(normalized_run.get("train_root_mae", float("nan"))),
            "val_root_mae": float(normalized_run.get("val_root_mae", float("nan"))),
            "test_root_mae": float(normalized_run.get("test_root_mae", float("nan"))),
            "train_exact_match_rate": float(
                normalized_run.get("train_exact_match_rate", float("nan"))
            ),
            "val_exact_match_rate": float(
                normalized_run.get("val_exact_match_rate", float("nan"))
            ),
            "test_exact_match_rate": float(
                normalized_run.get("test_exact_match_rate", float("nan"))
            ),
            "val_unweighted_full_law_objective": float(
                normalized_run.get("val_unweighted_full_law_objective", float("nan"))
            ),
            "val_unweighted_active_objective": float(
                normalized_run.get("val_unweighted_active_objective", float("nan"))
            ),
            "test_unweighted_full_law_objective": float(
                normalized_run.get("test_unweighted_full_law_objective", float("nan"))
            ),
            "test_unweighted_active_objective": float(
                normalized_run.get("test_unweighted_active_objective", float("nan"))
            ),
            "test_leaf_mae": float(normalized_run.get("test_leaf_mae", float("nan"))),
            "test_c2_count_drift_r1_mae": float(
                normalized_run.get("test_c2_count_drift_r1_mae", float("nan"))
            ),
            "test_c2_count_drift_r2_mae": float(
                normalized_run.get("test_c2_count_drift_r2_mae", float("nan"))
            ),
            "test_c2_count_drift_r4_mae": float(
                normalized_run.get("test_c2_count_drift_r4_mae", float("nan"))
            ),
            "test_c2_root_count_drift_r1_mae": float(
                normalized_run.get("test_c2_root_count_drift_r1_mae", float("nan"))
            ),
            "test_c2_root_count_drift_r2_mae": float(
                normalized_run.get("test_c2_root_count_drift_r2_mae", float("nan"))
            ),
            "test_c2_root_count_drift_r4_mae": float(
                normalized_run.get("test_c2_root_count_drift_r4_mae", float("nan"))
            ),
            "test_c2_idempotence_mae": float(
                normalized_run.get("test_c2_idempotence_mae", float("nan"))
            ),
            "test_c2_state_replay_mse": float(
                normalized_run.get("test_c2_state_replay_mse", float("nan"))
            ),
            "test_merge_mae": float(normalized_run.get("test_merge_mae", float("nan"))),
            "test_schedule_spread_mean": float(
                normalized_run.get("test_schedule_spread_mean", float("nan"))
            ),
            "parameterization": str(normalized_run.get("parameterization", "")),
            "weighting_scheme": str(normalized_run.get("weighting_scheme", "")),
            "optimization_root_weight": float(
                normalized_run.get("optimization_root_weight", float("nan"))
            ),
            "local_law_c1_weight": float(
                normalized_run.get("local_law_c1_weight", float("nan"))
            ),
            "local_law_c2_weight": float(
                normalized_run.get("local_law_c2_weight", float("nan"))
            ),
            "local_law_c3_weight": float(
                normalized_run.get("local_law_c3_weight", float("nan"))
            ),
            "task_objective_weight_source": str(
                normalized_run.get("task_objective_weight_source", "")
            ),
            "proxy_schedule_consistency_weight": float(
                normalized_run.get("proxy_schedule_consistency_weight", 0.0)
            ),
            "c2_metric_kind": str(normalized_run.get("c2_metric_kind", "")),
            "c2_proxy_metric_kind": str(
                normalized_run.get("c2_proxy_metric_kind", "")
            ),
            "c2_exact_witness_kind": str(
                normalized_run.get("c2_exact_witness_kind", "")
            ),
            "backend_name": str(normalized_run.get("backend_name", "")),
            "backend_package": str(normalized_run.get("backend_package", "")),
            "backend_version": str(normalized_run.get("backend_version", "")),
            "operator_class": str(normalized_run.get("operator_class", "")),
            "operator_evidence_status": str(
                normalized_run.get("operator_evidence_status", "")
            ),
            "theorem_relevance": bool(normalized_run.get("theorem_relevance", False)),
            "objective_weights_active": bool(
                normalized_run.get("objective_weights_active", False)
            ),
            "tree_root_supervision_kind": str(
                normalized_run.get("tree_root_supervision_kind", "")
            ),
            "aligned_sketch_surface": str(
                normalized_run.get("aligned_sketch_surface", "")
            ),
            "internal_supervision_kind": str(
                normalized_run.get("internal_supervision_kind", "")
            ),
            "internal_label_rate": float(
                normalized_run.get("internal_label_rate", 0.0)
            ),
            "leaf_exact_supervision": bool(
                normalized_run.get("leaf_exact_supervision", False)
            ),
            "tree_leaf_fno_width": int(
                normalized_run.get("tree_leaf_fno_width", 0)
            ),
            "tree_leaf_fno_n_modes": int(
                normalized_run.get("tree_leaf_fno_n_modes", 0)
            ),
            "tree_leaf_fno_n_layers": int(
                normalized_run.get("tree_leaf_fno_n_layers", 0)
            ),
            "tree_aux_doc_sequence_fraction": float(
                normalized_run.get("tree_aux_doc_sequence_fraction", 0.0)
            ),
            "comparison_semantics": str(
                normalized_run.get("comparison_semantics", "")
            ),
            "comparison_semantics_label": str(
                normalized_run.get("comparison_semantics_label", "")
            ),
            "run_intent_hash": str(normalized_run.get("run_intent_hash", "")),
            "run_intent_validation_status": str(
                normalized_run.get("run_intent_validation_status", "")
            ),
            "legacy_semantics": bool(normalized_run.get("legacy_semantics", False)),
            "legacy_semantics_reason": str(
                normalized_run.get("legacy_semantics_reason", "")
            ),
            "config_label": str(normalized_run.get("config_label", "")),
            "tuning_stage": str(normalized_run.get("tuning_stage", "")),
            "test_metrics_hidden_during_selection": bool(
                normalized_run.get("test_metrics_hidden_during_selection", False)
            ),
            "study_name": str(normalized_run.get("study_name", "")),
            "study_axis": str(normalized_run.get("study_axis", "")),
            "axis_value": normalized_run.get("axis_value", ""),
            "locked_tree_neural_config_label": str(
                normalized_run.get("locked_tree_neural_config_label", "")
            ),
            "selection_metric": str(normalized_run.get("selection_metric", "")),
            "budget_total_calls": int(normalized_run.get("budget_total_calls", 0)),
            "budget_total_calls_per_doc": float(
                normalized_run.get("budget_total_calls_per_doc", 0.0)
            ),
            "mass_target_per_doc": float(
                normalized_run.get("mass_target_per_doc", float("nan"))
            ),
            "budget_total_calls_used": float(
                normalized_run.get("budget_total_calls_used", float("nan"))
            ),
            "budget_utilization": float(
                normalized_run.get("budget_utilization", float("nan"))
            ),
            "full_doc_budget_share": float(
                normalized_run.get("full_doc_budget_share", 1.0)
            ),
            "full_doc_calls_requested": int(
                normalized_run.get("full_doc_calls_requested", 0)
            ),
            "full_doc_calls_total": int(
                normalized_run.get("full_doc_calls_total", 0)
            ),
            "local_calls_requested": int(
                normalized_run.get("local_calls_requested", 0)
            ),
            "local_calls_total": int(normalized_run.get("local_calls_total", 0)),
            "doc_consumption_mode": str(
                normalized_run.get("doc_consumption_mode", "")
            ),
            "local_split_mode": str(normalized_run.get("local_split_mode", "")),
            "local_allocation_policy": str(
                normalized_run.get("local_allocation_policy", "")
            ),
            "package_semantics": str(
                normalized_run.get("package_semantics", "")
            ),
            "depth_discount_gamma": float(
                normalized_run.get("depth_discount_gamma", 1.0)
            ),
            "effective_full_doc_mass_total": float(
                normalized_run.get("effective_full_doc_mass_total", float("nan"))
            ),
            "effective_full_doc_mass_per_doc": float(
                normalized_run.get("effective_full_doc_mass_per_doc", float("nan"))
            ),
            "document_mass_share": float(
                normalized_run.get("document_mass_share", float("nan"))
            ),
            "leaf_mass_share": float(
                normalized_run.get("leaf_mass_share", float("nan"))
            ),
            "internal_mass_share": float(
                normalized_run.get("internal_mass_share", float("nan"))
            ),
            "doc_touch_rate": float(
                normalized_run.get("doc_touch_rate", float("nan"))
            ),
            "mean_labels_per_touched_doc": float(
                normalized_run.get("mean_labels_per_touched_doc", float("nan"))
            ),
            "best_epoch": int(fit_diag.get("best_epoch", 0)),
            "epochs_completed": int(fit_diag.get("epochs_completed", 0)),
            "selection_metric_name": str(fit_diag.get("selection_metric_name", "")),
            "selection_metric_value": float(
                fit_diag.get("selection_metric_value", float("nan"))
            ),
            "test_root_direct_count_mae": float(
                normalized_run.get("test_root_direct_count_mae", float("nan"))
            ),
            "test_leaf_direct_exact_summary_match_rate": float(
                normalized_run.get(
                    "test_leaf_direct_exact_summary_match_rate",
                    float("nan"),
                )
            ),
            "test_merge_direct_exact_summary_match_rate": float(
                normalized_run.get(
                    "test_merge_direct_exact_summary_match_rate",
                    float("nan"),
                )
            ),
            "test_merge_join_bit_accuracy": float(
                normalized_run.get("test_merge_join_bit_accuracy", float("nan"))
            ),
            "leaf_count_head_entropy_mean": float(
                normalized_run.get("leaf_count_head_entropy_mean", float("nan"))
            ),
            "merge_count_head_entropy_mean": float(
                normalized_run.get("merge_count_head_entropy_mean", float("nan"))
            ),
            "leaf_count_head_margin_mean": float(
                normalized_run.get("leaf_count_head_margin_mean", float("nan"))
            ),
            "merge_count_head_margin_mean": float(
                normalized_run.get("merge_count_head_margin_mean", float("nan"))
            ),
            "exact_sketch_failure_bucket": str(
                normalized_run.get("exact_sketch_failure_bucket", "")
            ),
            "exact_sketch_leaf_gap_score": float(
                normalized_run.get("exact_sketch_leaf_gap_score", float("nan"))
            ),
            "exact_sketch_merge_gap_score": float(
                normalized_run.get("exact_sketch_merge_gap_score", float("nan"))
            ),
            "exact_sketch_theorem_count_decode_gap_score": float(
                normalized_run.get(
                    "exact_sketch_theorem_count_decode_gap_score",
                    float("nan"),
                )
            ),
            "exact_sketch_markov_sufficiency_gap_score": float(
                normalized_run.get(
                    "exact_sketch_markov_sufficiency_gap_score",
                    normalized_run.get(
                        "exact_sketch_theorem_count_decode_gap_score",
                        float("nan"),
                    ),
                )
            ),
            "exact_sketch_internal_label_value_gap_score": float(
                normalized_run.get(
                    "exact_sketch_internal_label_value_gap_score",
                    float("nan"),
                )
            ),
            "exact_sketch_readout_gap_score": float(
                normalized_run.get("exact_sketch_readout_gap_score", float("nan"))
            ),
            "train_loss_final": float(fit_diag.get("train_loss_final", float("nan"))),
            "elapsed_s": float(normalized_run.get("elapsed_s", float("nan"))),
            "bundle_source": str(normalized_run.get("bundle_source", "")),
            "train_corpus_signature": str(normalized_run.get("train_corpus_signature", "")),
            "val_corpus_signature": str(normalized_run.get("val_corpus_signature", "")),
            "test_corpus_signature": str(normalized_run.get("test_corpus_signature", "")),
        }
        rows.append(row)
    return rows


def _build_heatmap_rows(
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in aggregate_rows:
        rows.append(
            {
                "benchmark": str(row.get("benchmark", "")),
                "cell_id": str(row.get("cell_id", "")),
                "baseline_family": str(row.get("baseline_family", "")),
                "train_doc_count": int(row.get("train_doc_count", 0)),
                "fixed_leaf_tokens": int(row.get("fixed_leaf_tokens", 0)),
                "train_mean_leaves_per_doc": float(
                    row.get("train_mean_leaves_per_doc", float("nan"))
                ),
                "val_mean_leaves_per_doc": float(
                    row.get("val_mean_leaves_per_doc", float("nan"))
                ),
                "test_mean_leaves_per_doc": float(
                    row.get("test_mean_leaves_per_doc", float("nan"))
                ),
                "n_regimes": int(row.get("n_regimes", 0)),
                "segment_density_band": str(row.get("segment_density_band", "")),
                "segment_min": int(row.get("segment_min", 0)),
                "segment_max": int(row.get("segment_max", 0)),
                "train_root_mae_mean": float(row.get("train_root_mae_mean", float("nan"))),
                "val_root_mae_mean": float(row.get("val_root_mae_mean", float("nan"))),
                "test_root_mae_mean": float(row.get("test_root_mae_mean", float("nan"))),
                "train_exact_match_rate_mean": float(
                    row.get("train_exact_match_rate_mean", float("nan"))
                ),
                "val_exact_match_rate_mean": float(
                    row.get("val_exact_match_rate_mean", float("nan"))
                ),
                "test_exact_match_rate_mean": float(
                    row.get("test_exact_match_rate_mean", float("nan"))
                ),
                "val_unweighted_full_law_objective_mean": float(
                    row.get("val_unweighted_full_law_objective_mean", float("nan"))
                ),
                "val_unweighted_active_objective_mean": float(
                    row.get("val_unweighted_active_objective_mean", float("nan"))
                ),
                "test_unweighted_full_law_objective_mean": float(
                    row.get("test_unweighted_full_law_objective_mean", float("nan"))
                ),
                "test_unweighted_active_objective_mean": float(
                    row.get("test_unweighted_active_objective_mean", float("nan"))
                ),
                "test_leaf_mae_mean": float(row.get("test_leaf_mae_mean", float("nan"))),
                "test_merge_mae_mean": float(
                    row.get("test_merge_mae_mean", float("nan"))
                ),
            "test_schedule_spread_mean_mean": float(
                row.get("test_schedule_spread_mean_mean", float("nan"))
            ),
            "elapsed_s_mean": float(row.get("elapsed_s_mean", float("nan"))),
            "parameterization": str(row.get("parameterization", "")),
            "weighting_scheme": str(row.get("weighting_scheme", "")),
            "optimization_root_weight": float(
                row.get("optimization_root_weight", float("nan"))
            ),
                "local_law_c1_weight": float(
                    row.get("local_law_c1_weight", float("nan"))
                ),
                "local_law_c2_weight": float(
                    row.get("local_law_c2_weight", float("nan"))
                ),
                "local_law_c3_weight": float(
                    row.get("local_law_c3_weight", float("nan"))
                ),
            "task_objective_weight_source": str(
                row.get("task_objective_weight_source", "")
            ),
            "proxy_schedule_consistency_weight": float(
                row.get("proxy_schedule_consistency_weight", 0.0)
            ),
            "c2_metric_kind": str(row.get("c2_metric_kind", "")),
                "backend_name": str(row.get("backend_name", "")),
                "backend_package": str(row.get("backend_package", "")),
                "backend_version": str(row.get("backend_version", "")),
                "operator_class": str(row.get("operator_class", "")),
                "operator_evidence_status": str(
                    row.get("operator_evidence_status", "")
                ),
                "theorem_relevance": bool(row.get("theorem_relevance", False)),
                "objective_weights_active": bool(
                    row.get("objective_weights_active", False)
                ),
                "comparison_semantics": str(row.get("comparison_semantics", "")),
                "comparison_semantics_label": str(
                    row.get("comparison_semantics_label", "")
                ),
                "run_intent_hash": str(row.get("run_intent_hash", "")),
                "run_intent_validation_status": str(
                    row.get("run_intent_validation_status", "")
                ),
                "legacy_semantics": bool(row.get("legacy_semantics", False)),
                "config_label": str(row.get("config_label", "")),
                "tuning_stage": str(row.get("tuning_stage", "")),
                "study_name": str(row.get("study_name", "")),
                "study_axis": str(row.get("study_axis", "")),
                "axis_value": row.get("axis_value", ""),
                "locked_tree_neural_config_label": str(
                    row.get("locked_tree_neural_config_label", "")
                ),
                "selection_metric": str(row.get("selection_metric", "")),
                "test_metrics_hidden_during_selection": bool(
                    row.get("test_metrics_hidden_during_selection", False)
                ),
                "budget_total_calls": int(row.get("budget_total_calls", 0)),
                "budget_total_calls_per_doc": float(
                    row.get("budget_total_calls_per_doc", 0.0)
                ),
                "mass_target_per_doc": float(
                    row.get("mass_target_per_doc", float("nan"))
                ),
                "full_doc_budget_share": float(
                    row.get("full_doc_budget_share", 1.0)
                ),
                "full_doc_calls_requested": int(row.get("full_doc_calls_requested", 0)),
                "full_doc_calls_total": int(row.get("full_doc_calls_total", 0)),
                "local_calls_requested": int(row.get("local_calls_requested", 0)),
                "local_calls_total": int(row.get("local_calls_total", 0)),
                "doc_consumption_mode": str(row.get("doc_consumption_mode", "")),
                "local_split_mode": str(row.get("local_split_mode", "")),
                "local_allocation_policy": str(
                    row.get("local_allocation_policy", "")
                ),
                "package_semantics": str(row.get("package_semantics", "")),
                "depth_discount_gamma": float(
                    row.get("depth_discount_gamma", 1.0)
                ),
                "effective_full_doc_mass_total_mean": float(
                    row.get("effective_full_doc_mass_total_mean", float("nan"))
                ),
                "effective_full_doc_mass_per_doc_mean": float(
                    row.get("effective_full_doc_mass_per_doc_mean", float("nan"))
                ),
                "document_mass_share_mean": float(
                    row.get("document_mass_share_mean", float("nan"))
                ),
                "leaf_mass_share_mean": float(
                    row.get("leaf_mass_share_mean", float("nan"))
                ),
                "internal_mass_share_mean": float(
                    row.get("internal_mass_share_mean", float("nan"))
                ),
                "doc_touch_rate_mean": float(
                    row.get("doc_touch_rate_mean", float("nan"))
                ),
                "mean_labels_per_touched_doc_mean": float(
                    row.get("mean_labels_per_touched_doc_mean", float("nan"))
                ),
            }
        )
    return rows


def _tree_neural_validation_summary(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    if str(payload.get("benchmark", "")).strip() != "recoverable_v4":
        return {}
    if len(list(payload.get("study_names") or [])) > 0:
        return {}
    fixed_leaf_tokens = {
        int(row.get("fixed_leaf_tokens", 0))
        for row in list(payload.get("aggregate_rows") or [])
        if int(row.get("fixed_leaf_tokens", 0)) > 0
    }
    if len(fixed_leaf_tokens) > 1:
        return {}
    aggregate_rows = [
        row
        for row in list(payload.get("aggregate_rows") or [])
        if str(row.get("baseline_family", "")) in TREE_NEURAL_BASELINE_FAMILIES
        and _is_headline_comparison_semantics(
            str(row.get("comparison_semantics", ""))
        )
    ]
    if not aggregate_rows:
        return {}

    by_family_and_count: Dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in aggregate_rows:
        key = (
            str(row.get("baseline_family", "")),
            int(row.get("train_doc_count", 0)),
        )
        by_family_and_count[key] = row

    comparisons: List[Dict[str, Any]] = []
    train_counts = sorted(
        {
            int(row.get("train_doc_count", 0))
            for row in aggregate_rows
            if int(row.get("train_doc_count", 0)) > 0
        }
    )
    for train_doc_count in train_counts:
        c2_row = by_family_and_count.get(("tree_neural_c2", int(train_doc_count)))
        all_row = by_family_and_count.get(("tree_neural", int(train_doc_count)))
        if c2_row is None or all_row is None:
            continue
        c2_root_mae = float(c2_row.get("test_root_mae_mean", float("nan")))
        all_root_mae = float(all_row.get("test_root_mae_mean", float("nan")))
        comparisons.append(
            {
                "train_doc_count": int(train_doc_count),
                "tree_neural_c2_root_mae_mean": c2_root_mae,
                "tree_neural_root_mae_mean": all_root_mae,
                "all_laws_worse_than_c2_only": bool(all_root_mae > c2_root_mae),
                "tree_neural_c2_resolved_weights": {
                    "parameterization": str(c2_row.get("parameterization", "")),
                    "optimization_root_weight": float(
                        c2_row.get("optimization_root_weight", float("nan"))
                    ),
                    "local_law_c1_weight": float(
                        c2_row.get("local_law_c1_weight", float("nan"))
                    ),
                    "local_law_c2_weight": float(
                        c2_row.get("local_law_c2_weight", float("nan"))
                    ),
                    "local_law_c3_weight": float(
                        c2_row.get("local_law_c3_weight", float("nan"))
                    ),
                },
                "tree_neural_resolved_weights": {
                    "parameterization": str(all_row.get("parameterization", "")),
                    "optimization_root_weight": float(
                        all_row.get("optimization_root_weight", float("nan"))
                    ),
                    "local_law_c1_weight": float(
                        all_row.get("local_law_c1_weight", float("nan"))
                    ),
                    "local_law_c2_weight": float(
                        all_row.get("local_law_c2_weight", float("nan"))
                    ),
                    "local_law_c3_weight": float(
                        all_row.get("local_law_c3_weight", float("nan"))
                    ),
                },
                "c2_metric_kind": str(all_row.get("c2_metric_kind", "")),
                "c2_proxy_metric_kind": str(all_row.get("c2_proxy_metric_kind", "")),
                "c2_exact_witness_kind": str(all_row.get("c2_exact_witness_kind", "")),
            }
        )

    if not comparisons:
        return {}

    return {
        "benchmark": "recoverable_v4",
        "comparison_basis": CURRENT_TREE_NEURAL_SEMANTICS_VERSION,
        "c2_metric_kind": FNO_TREE_C2_METRIC_KIND,
        "c2_proxy_metric_kind": FNO_TREE_C2_PROXY_METRIC_KIND,
        "c2_exact_witness_kind": FNO_TREE_C2_EXACT_WITNESS_KIND,
        "current_rows_only": True,
        "comparisons": comparisons,
        "all_laws_worse_than_c2_only_still_holds": bool(
            all(item["all_laws_worse_than_c2_only"] for item in comparisons)
        ),
    }


def _render_tree_neural_validation_markdown(
    summary: Mapping[str, Any],
) -> str:
    comparisons = list(summary.get("comparisons") or [])
    if not comparisons:
        return ""
    lines = [
        "# Tree-Neural Semantic Alignment Validation",
        "",
        f"- benchmark: `{str(summary.get('benchmark', ''))}`",
        f"- comparison basis: `{str(summary.get('comparison_basis', ''))}`",
        f"- primary Markov C2 metric: `{str(summary.get('c2_metric_kind', ''))}`",
        f"- C2 replay proxy: `{str(summary.get('c2_proxy_metric_kind', ''))}`",
        f"- C2 exact witness: `{str(summary.get('c2_exact_witness_kind', ''))}`",
        (
            "- all laws worse than C2-only after semantic alignment: "
            f"`{bool(summary.get('all_laws_worse_than_c2_only_still_holds', False))}`"
        ),
        "",
        "| train_docs | tree_neural_c2 root_mae | tree_neural root_mae | all_laws_worse_than_c2_only |",
        "|---|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            "| "
            f"{int(item.get('train_doc_count', 0))} | "
            f"{float(item.get('tree_neural_c2_root_mae_mean', float('nan'))):.6g} | "
            f"{float(item.get('tree_neural_root_mae_mean', float('nan'))):.6g} | "
            f"{bool(item.get('all_laws_worse_than_c2_only', False))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _tree_fno_fair_parity_summary(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    if str(payload.get("benchmark", "")).strip() != "recoverable_v4":
        return {}
    if str(payload.get("hardness_grid", "")).strip():
        return {}
    aggregate_rows = list(payload.get("aggregate_rows") or [])
    if not aggregate_rows:
        return {}

    tree_rows = [
        dict(row)
        for row in aggregate_rows
        if str(row.get("baseline_family", "")).startswith("tree_neural")
        and str(row.get("config_label", "")).strip() == FAIR_FNO_PARITY_CONFIG_LABEL
    ]
    fno_rows = [
        dict(row)
        for row in aggregate_rows
        if str(row.get("baseline_family", "")) in {"official_fno", "official_fno_sumlen"}
    ]
    if not tree_rows or not fno_rows:
        return {}

    threshold_ratio = 0.10
    train_doc_counts = sorted(
        {
            int(row.get("train_doc_count", 0))
            for row in tree_rows
            if int(row.get("train_doc_count", 0)) > 0
        }
    )
    comparisons: List[Dict[str, Any]] = []
    for train_doc_count in train_doc_counts:
        tree_subset = [
            row
            for row in tree_rows
            if int(row.get("train_doc_count", 0)) == int(train_doc_count)
        ]
        fno_subset = [
            row
            for row in fno_rows
            if int(row.get("train_doc_count", 0)) == int(train_doc_count)
        ]
        if not tree_subset or not fno_subset:
            continue
        best_fno = min(
            fno_subset,
            key=lambda row: float(row.get("test_root_mae_mean", float("inf"))),
        )
        best_tree = min(
            tree_subset,
            key=lambda row: float(row.get("test_root_mae_mean", float("inf"))),
        )
        tree_family_map = {
            str(row.get("baseline_family", "")): row for row in tree_subset
        }
        tree_neural_row = tree_family_map.get("tree_neural")
        best_fno_mae = float(best_fno.get("test_root_mae_mean", float("nan")))
        best_tree_mae = float(best_tree.get("test_root_mae_mean", float("nan")))
        tree_neural_mae = (
            float(tree_neural_row.get("test_root_mae_mean", float("nan")))
            if tree_neural_row is not None
            else float("nan")
        )
        primary_gap_ratio = (
            float((tree_neural_mae - best_fno_mae) / best_fno_mae)
            if np.isfinite(tree_neural_mae)
            and np.isfinite(best_fno_mae)
            and best_fno_mae > 0.0
            else float("inf")
        )
        secondary_gap_ratio = (
            float((best_tree_mae - best_fno_mae) / best_fno_mae)
            if np.isfinite(best_tree_mae)
            and np.isfinite(best_fno_mae)
            and best_fno_mae > 0.0
            else float("inf")
        )
        comparisons.append(
            {
                "train_doc_count": int(train_doc_count),
                "best_full_doc_fno_family": str(best_fno.get("baseline_family", "")),
                "best_full_doc_fno_test_root_mae_mean": float(best_fno_mae),
                "tree_neural_test_root_mae_mean": float(tree_neural_mae),
                "tree_neural_c2_test_root_mae_mean": float(
                    tree_family_map.get("tree_neural_c2", {}).get(
                        "test_root_mae_mean", float("nan")
                    )
                ),
                "tree_neural_c2c3_test_root_mae_mean": float(
                    tree_family_map.get("tree_neural_c2c3", {}).get(
                        "test_root_mae_mean", float("nan")
                    )
                ),
                "best_parity_tree_family": str(best_tree.get("baseline_family", "")),
                "best_parity_tree_test_root_mae_mean": float(best_tree_mae),
                "tree_neural_gap_ratio_vs_best_fno": float(primary_gap_ratio),
                "best_parity_tree_gap_ratio_vs_best_fno": float(secondary_gap_ratio),
                "primary_success_within_10pct": bool(primary_gap_ratio <= threshold_ratio),
                "secondary_success_within_10pct": bool(
                    secondary_gap_ratio <= threshold_ratio
                ),
            }
        )
    if not comparisons:
        return {}

    gate_count = 10240
    gate = next(
        (
            item
            for item in comparisons
            if int(item.get("train_doc_count", 0)) == int(gate_count)
        ),
        None,
    )
    widths = sorted(
        {
            int(row.get("tree_leaf_fno_width", 0))
            for row in tree_rows
            if int(row.get("tree_leaf_fno_width", 0)) > 0
        }
    )
    modes = sorted(
        {
            int(row.get("tree_leaf_fno_n_modes", 0))
            for row in tree_rows
            if int(row.get("tree_leaf_fno_n_modes", 0)) > 0
        }
    )
    layers = sorted(
        {
            int(row.get("tree_leaf_fno_n_layers", 0))
            for row in tree_rows
            if int(row.get("tree_leaf_fno_n_layers", 0)) > 0
        }
    )
    root_kinds = sorted(
        {
            str(row.get("tree_root_supervision_kind", "")).strip()
            for row in tree_rows
            if str(row.get("tree_root_supervision_kind", "")).strip()
        }
    )
    aux_fracs = sorted(
        {
            float(row.get("tree_aux_doc_sequence_fraction", 0.0))
            for row in tree_rows
        }
    )
    return {
        "benchmark": "recoverable_v4",
        "parity_config_label": FAIR_FNO_PARITY_CONFIG_LABEL,
        "gate_train_doc_count": int(gate_count),
        "success_threshold_ratio": float(threshold_ratio),
        "tree_root_supervision_kind": root_kinds[0] if len(root_kinds) == 1 else "",
        "tree_root_supervision_kinds": root_kinds,
        "tree_leaf_fno_width": widths[0] if len(widths) == 1 else None,
        "tree_leaf_fno_width_values": widths,
        "tree_leaf_fno_n_modes": modes[0] if len(modes) == 1 else None,
        "tree_leaf_fno_n_modes_values": modes,
        "tree_leaf_fno_n_layers": layers[0] if len(layers) == 1 else None,
        "tree_leaf_fno_n_layers_values": layers,
        "tree_aux_doc_sequence_fraction": (
            aux_fracs[0] if len(aux_fracs) == 1 else float("nan")
        ),
        "tree_aux_doc_sequence_fraction_values": aux_fracs,
        "comparisons": comparisons,
        "primary_success_met": bool(
            gate is not None and gate.get("primary_success_within_10pct", False)
        ),
        "secondary_success_met": bool(
            gate is not None and gate.get("secondary_success_within_10pct", False)
        ),
        "best_full_doc_fno_family_at_gate": (
            str(gate.get("best_full_doc_fno_family", "")) if gate is not None else ""
        ),
        "best_parity_tree_family_at_gate": (
            str(gate.get("best_parity_tree_family", "")) if gate is not None else ""
        ),
        "tree_neural_gap_ratio_vs_best_fno_at_gate": (
            float(gate.get("tree_neural_gap_ratio_vs_best_fno", float("nan")))
            if gate is not None
            else float("nan")
        ),
        "best_parity_tree_gap_ratio_vs_best_fno_at_gate": (
            float(gate.get("best_parity_tree_gap_ratio_vs_best_fno", float("nan")))
            if gate is not None
            else float("nan")
        ),
        "comparison_interpretation": (
            "old_tree_gap_was_confounded_by_capacity_or_objective_mismatch"
            if gate is not None
            and (
                gate.get("primary_success_within_10pct", False)
                or gate.get("secondary_success_within_10pct", False)
            )
            else "residual_gap_looks_like_a_true_recursive_operator_gap"
        ),
    }


def _render_tree_fno_fair_parity_markdown(
    summary: Mapping[str, Any],
) -> str:
    comparisons = list(summary.get("comparisons") or [])
    if not comparisons:
        return ""
    lines = [
        "# Tree/FNO Fair-Parity Summary",
        "",
        f"- benchmark: `{str(summary.get('benchmark', ''))}`",
        f"- parity config label: `{str(summary.get('parity_config_label', ''))}`",
        f"- gate train docs: `{int(summary.get('gate_train_doc_count', 0))}`",
        f"- success threshold ratio: `{float(summary.get('success_threshold_ratio', 0.0)):.2f}`",
        f"- tree root supervision kind: `{str(summary.get('tree_root_supervision_kind', ''))}`",
        f"- tree aux doc-sequence fraction: `{float(summary.get('tree_aux_doc_sequence_fraction', float('nan'))):.6g}`",
        f"- primary success met: `{bool(summary.get('primary_success_met', False))}`",
        f"- secondary success met: `{bool(summary.get('secondary_success_met', False))}`",
        f"- interpretation: `{str(summary.get('comparison_interpretation', ''))}`",
        "",
        "| train_docs | best_fno | best_fno_mae | tree_neural_mae | best_parity_tree | best_parity_tree_mae | primary<=10% | secondary<=10% |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            "| "
            f"{int(item.get('train_doc_count', 0))} | "
            f"{str(item.get('best_full_doc_fno_family', ''))} | "
            f"{float(item.get('best_full_doc_fno_test_root_mae_mean', float('nan'))):.6g} | "
            f"{float(item.get('tree_neural_test_root_mae_mean', float('nan'))):.6g} | "
            f"{str(item.get('best_parity_tree_family', ''))} | "
            f"{float(item.get('best_parity_tree_test_root_mae_mean', float('nan'))):.6g} | "
            f"{bool(item.get('primary_success_within_10pct', False))} | "
            f"{bool(item.get('secondary_success_within_10pct', False))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _tree_fno_upper_bound_summary(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    if str(payload.get("benchmark", "")).strip() != "recoverable_v4":
        return {}
    if str(payload.get("hardness_grid", "")).strip():
        return {}
    aggregate_rows = list(payload.get("aggregate_rows") or [])
    if not aggregate_rows:
        return {}

    fno_rows = [
        dict(row)
        for row in aggregate_rows
        if str(row.get("baseline_family", "")) in {"official_fno", "official_fno_sumlen"}
    ]
    upper_rows = [
        dict(row)
        for row in aggregate_rows
        if str(row.get("baseline_family", "")).startswith("tree_neural")
        and float(row.get("tree_aux_doc_sequence_fraction", 0.0)) > 0.0
        and str(row.get("config_label", "")).startswith(FAIR_FNO_PARITY_CONFIG_LABEL)
    ]
    if not fno_rows or not upper_rows:
        return {}

    aux_fractions = sorted(
        {
            float(row.get("tree_aux_doc_sequence_fraction", 0.0))
            for row in upper_rows
        }
    )
    train_doc_counts = sorted(
        {
            int(row.get("train_doc_count", 0))
            for row in upper_rows
            if int(row.get("train_doc_count", 0)) > 0
        }
    )
    comparisons: List[Dict[str, Any]] = []
    for train_doc_count in train_doc_counts:
        fno_subset = [
            row
            for row in fno_rows
            if int(row.get("train_doc_count", 0)) == int(train_doc_count)
        ]
        if not fno_subset:
            continue
        best_fno = min(
            fno_subset,
            key=lambda row: float(row.get("test_root_mae_mean", float("inf"))),
        )
        best_fno_mae = float(best_fno.get("test_root_mae_mean", float("nan")))
        for aux_fraction in aux_fractions:
            tree_subset = [
                row
                for row in upper_rows
                if int(row.get("train_doc_count", 0)) == int(train_doc_count)
                and abs(
                    float(row.get("tree_aux_doc_sequence_fraction", 0.0))
                    - float(aux_fraction)
                )
                <= 1e-12
            ]
            if not tree_subset:
                continue
            tree_family_map = {
                str(row.get("baseline_family", "")): row for row in tree_subset
            }
            best_tree = min(
                tree_subset,
                key=lambda row: float(row.get("test_root_mae_mean", float("inf"))),
            )
            best_tree_mae = float(best_tree.get("test_root_mae_mean", float("nan")))
            best_tree_gap_ratio = (
                float((best_tree_mae - best_fno_mae) / best_fno_mae)
                if np.isfinite(best_tree_mae)
                and np.isfinite(best_fno_mae)
                and best_fno_mae > 0.0
                else float("inf")
            )
            comparisons.append(
                {
                    "train_doc_count": int(train_doc_count),
                    "tree_aux_doc_sequence_fraction": float(aux_fraction),
                    "best_full_doc_fno_family": str(best_fno.get("baseline_family", "")),
                    "best_full_doc_fno_test_root_mae_mean": float(best_fno_mae),
                    "tree_neural_test_root_mae_mean": float(
                        tree_family_map.get("tree_neural", {}).get(
                            "test_root_mae_mean", float("nan")
                        )
                    ),
                    "tree_neural_c2_test_root_mae_mean": float(
                        tree_family_map.get("tree_neural_c2", {}).get(
                            "test_root_mae_mean", float("nan")
                        )
                    ),
                    "tree_neural_c2c3_test_root_mae_mean": float(
                        tree_family_map.get("tree_neural_c2c3", {}).get(
                            "test_root_mae_mean", float("nan")
                        )
                    ),
                    "best_upper_bound_tree_family": str(
                        best_tree.get("baseline_family", "")
                    ),
                    "best_upper_bound_tree_test_root_mae_mean": float(best_tree_mae),
                    "best_upper_bound_tree_gap_ratio_vs_best_fno": float(
                        best_tree_gap_ratio
                    ),
                }
            )
    if not comparisons:
        return {}
    gate_count = 10240
    gate_rows = [
        row
        for row in comparisons
        if int(row.get("train_doc_count", 0)) == int(gate_count)
    ]
    best_gate = min(
        gate_rows,
        key=lambda row: float(
            row.get("best_upper_bound_tree_gap_ratio_vs_best_fno", float("inf"))
        ),
    ) if gate_rows else None
    return {
        "benchmark": "recoverable_v4",
        "gate_train_doc_count": int(gate_count),
        "aux_fractions": [float(value) for value in aux_fractions],
        "comparisons": comparisons,
        "best_gate_aux_fraction": (
            float(best_gate.get("tree_aux_doc_sequence_fraction", float("nan")))
            if best_gate is not None
            else float("nan")
        ),
        "best_gate_upper_bound_family": (
            str(best_gate.get("best_upper_bound_tree_family", ""))
            if best_gate is not None
            else ""
        ),
        "best_gate_upper_bound_gap_ratio_vs_best_fno": (
            float(best_gate.get("best_upper_bound_tree_gap_ratio_vs_best_fno", float("nan")))
            if best_gate is not None
            else float("nan")
        ),
    }


def _render_tree_fno_upper_bound_markdown(
    summary: Mapping[str, Any],
) -> str:
    comparisons = list(summary.get("comparisons") or [])
    if not comparisons:
        return ""
    lines = [
        "# Tree/FNO Upper-Bound Summary",
        "",
        f"- benchmark: `{str(summary.get('benchmark', ''))}`",
        f"- gate train docs: `{int(summary.get('gate_train_doc_count', 0))}`",
        f"- aux fractions: `{list(summary.get('aux_fractions') or [])}`",
        f"- best gate aux fraction: `{float(summary.get('best_gate_aux_fraction', float('nan'))):.6g}`",
        f"- best gate upper-bound family: `{str(summary.get('best_gate_upper_bound_family', ''))}`",
        "",
        "| train_docs | aux_frac | best_fno | best_fno_mae | tree_neural_mae | best_upper_bound_tree | best_upper_bound_tree_mae | best_upper_bound_gap_vs_fno |",
        "|---|---:|---|---:|---:|---|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            "| "
            f"{int(item.get('train_doc_count', 0))} | "
            f"{float(item.get('tree_aux_doc_sequence_fraction', float('nan'))):.6g} | "
            f"{str(item.get('best_full_doc_fno_family', ''))} | "
            f"{float(item.get('best_full_doc_fno_test_root_mae_mean', float('nan'))):.6g} | "
            f"{float(item.get('tree_neural_test_root_mae_mean', float('nan'))):.6g} | "
            f"{str(item.get('best_upper_bound_tree_family', ''))} | "
            f"{float(item.get('best_upper_bound_tree_test_root_mae_mean', float('nan'))):.6g} | "
            f"{100.0 * float(item.get('best_upper_bound_tree_gap_ratio_vs_best_fno', float('nan'))):.3g}% |"
        )
    lines.append("")
    return "\n".join(lines)


def _tree_oracle_budget_frontier_summary(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    aggregate_rows = [
        dict(row)
        for row in list(payload.get("aggregate_rows") or [])
        if (
            int(row.get("budget_total_calls", 0)) > 0
            or float(row.get("budget_total_calls_per_doc", 0.0)) > 0.0
            or str(row.get("study_name", "")) == ORACLE_BUDGET_STUDY_NAME
        )
    ]
    if not aggregate_rows:
        return {}

    tree_rows = [
        row
        for row in aggregate_rows
        if str(row.get("baseline_family", "")) in TREE_NEURAL_BASELINE_FAMILIES
    ]
    reference_rows = [
        row
        for row in aggregate_rows
        if str(row.get("baseline_family", "")) in FULL_DOC_ONLY_BUDGET_FAMILIES
    ]
    budget_levels = sorted(
        {
            float(row.get("budget_total_calls_per_doc", 0.0))
            for row in aggregate_rows
            if float(row.get("budget_total_calls_per_doc", 0.0)) > 0.0
        }
    )
    full_doc_shares = sorted(
        {
            float(row.get("full_doc_budget_share", 1.0))
            for row in aggregate_rows
        }
    )

    best_tree_by_budget: List[Dict[str, Any]] = []
    for budget_per_doc in budget_levels:
        subset = [
            row
            for row in tree_rows
            if abs(float(row.get("budget_total_calls_per_doc", 0.0)) - float(budget_per_doc))
            <= 1e-12
        ]
        if not subset:
            continue
        best = min(
            subset,
            key=lambda row: float(row.get("test_root_mae_mean", float("inf"))),
        )
        best_tree_by_budget.append(
            {
                "budget_total_calls_per_doc": float(budget_per_doc),
                "baseline_family": str(best.get("baseline_family", "")),
                "test_root_mae_mean": float(best.get("test_root_mae_mean", float("nan"))),
                "full_doc_budget_share": float(best.get("full_doc_budget_share", 1.0)),
                "doc_consumption_mode": str(best.get("doc_consumption_mode", "")),
                "local_split_mode": str(best.get("local_split_mode", "")),
                "effective_full_doc_mass_per_doc_mean": float(
                    best.get("effective_full_doc_mass_per_doc_mean", float("nan"))
                ),
            }
        )

    best_reference_by_budget: List[Dict[str, Any]] = []
    for budget_per_doc in budget_levels:
        subset = [
            row
            for row in reference_rows
            if abs(float(row.get("budget_total_calls_per_doc", 0.0)) - float(budget_per_doc))
            <= 1e-12
            and abs(float(row.get("full_doc_budget_share", 1.0)) - 1.0) <= 1e-12
        ]
        if not subset:
            continue
        best = min(
            subset,
            key=lambda row: float(row.get("test_root_mae_mean", float("inf"))),
        )
        best_reference_by_budget.append(
            {
                "budget_total_calls_per_doc": float(budget_per_doc),
                "baseline_family": str(best.get("baseline_family", "")),
                "test_root_mae_mean": float(best.get("test_root_mae_mean", float("nan"))),
                "effective_full_doc_mass_per_doc_mean": float(
                    best.get("effective_full_doc_mass_per_doc_mean", float("nan"))
                ),
            }
        )

    return {
        "benchmark": str(payload.get("benchmark", "")),
        "study_name": ORACLE_BUDGET_STUDY_NAME,
        "selection_metric": str(payload.get("selection_metric", "") or DEV_SELECTION_METRIC),
        "budget_levels_per_doc": [float(value) for value in budget_levels],
        "full_doc_budget_shares": [float(value) for value in full_doc_shares],
        "tree_rows": tree_rows,
        "reference_rows": reference_rows,
        "best_tree_by_budget": best_tree_by_budget,
        "best_reference_by_budget": best_reference_by_budget,
    }


def _render_tree_oracle_budget_frontier_markdown(
    summary: Mapping[str, Any],
) -> str:
    best_tree_rows = list(summary.get("best_tree_by_budget") or [])
    best_reference_rows = list(summary.get("best_reference_by_budget") or [])
    if not best_tree_rows and not best_reference_rows:
        return ""
    lines = [
        "# Oracle Attention Budget Share Frontier",
        "",
        f"- benchmark: `{str(summary.get('benchmark', ''))}`",
        f"- study_name: `{str(summary.get('study_name', ''))}`",
        f"- budget levels per doc: `{list(summary.get('budget_levels_per_doc') or [])}`",
        f"- full-doc budget shares: `{list(summary.get('full_doc_budget_shares') or [])}`",
        "",
        "## Best Tree Policy By Raw-Call Budget",
        "",
        "| calls/doc | best tree family | test root_mae | full-doc share | doc mode | local split | eff full-doc mass/doc |",
        "|---:|---|---:|---:|---|---|---:|",
    ]
    for row in best_tree_rows:
        lines.append(
            "| "
            f"{float(row.get('budget_total_calls_per_doc', float('nan'))):.6g} | "
            f"{str(row.get('baseline_family', ''))} | "
            f"{float(row.get('test_root_mae_mean', float('nan'))):.6g} | "
            f"{float(row.get('full_doc_budget_share', float('nan'))):.6g} | "
            f"{str(row.get('doc_consumption_mode', ''))} | "
            f"{str(row.get('local_split_mode', ''))} | "
            f"{float(row.get('effective_full_doc_mass_per_doc_mean', float('nan'))):.6g} |"
        )
    if best_reference_rows:
        lines.extend(
            [
                "",
                "## Best Document-Only Reference By Raw-Call Budget",
                "",
                "| calls/doc | best reference family | test root_mae | eff full-doc mass/doc |",
                "|---:|---|---:|---:|",
            ]
        )
        for row in best_reference_rows:
            lines.append(
                "| "
                f"{float(row.get('budget_total_calls_per_doc', float('nan'))):.6g} | "
                f"{str(row.get('baseline_family', ''))} | "
                f"{float(row.get('test_root_mae_mean', float('nan'))):.6g} | "
                f"{float(row.get('effective_full_doc_mass_per_doc_mean', float('nan'))):.6g} |"
            )
    lines.append("")
    return "\n".join(lines)


def _learning_efficiency_summary(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    if str(payload.get("hardness_grid", "")).strip():
        return {}
    aggregate_rows = [
        dict(row)
        for row in list(payload.get("aggregate_rows") or [])
        if np.isfinite(float(row.get("test_root_mae_mean", float("nan"))))
    ]
    learned_families = [
        family
        for family in (
            "tree_neural_c2",
            "tree_neural_c2c3",
            "tree_neural",
            "official_fno",
            "official_fno_sumlen",
        )
        if any(str(row.get("baseline_family", "")) == family for row in aggregate_rows)
    ]
    family_summaries: List[Dict[str, Any]] = []
    for family in learned_families:
        rows = sorted(
            [
                row
                for row in aggregate_rows
                if str(row.get("baseline_family", "")) == family
            ],
            key=lambda row: int(row.get("train_doc_count", 0)),
        )
        if len(rows) < 2:
            continue
        best = min(rows, key=lambda row: float(row.get("test_root_mae_mean", float("inf"))))
        best_mae = float(best.get("test_root_mae_mean", float("nan")))
        if not np.isfinite(best_mae):
            continue

        def _first_within(multiplier: float) -> Mapping[str, Any] | None:
            threshold = float(best_mae * float(multiplier))
            for row in rows:
                value = float(row.get("test_root_mae_mean", float("nan")))
                if np.isfinite(value) and value <= threshold:
                    return row
            return None

        within_10 = _first_within(1.10)
        within_25 = _first_within(1.25)
        family_summaries.append(
            {
                "baseline_family": str(family),
                "smallest_train_doc_count": int(rows[0].get("train_doc_count", 0)),
                "smallest_test_root_mae_mean": float(
                    rows[0].get("test_root_mae_mean", float("nan"))
                ),
                "best_train_doc_count": int(best.get("train_doc_count", 0)),
                "best_test_root_mae_mean": best_mae,
                "first_within_10pct_train_doc_count": (
                    int(within_10.get("train_doc_count", 0)) if within_10 else 0
                ),
                "first_within_10pct_test_root_mae_mean": (
                    float(within_10.get("test_root_mae_mean", float("nan")))
                    if within_10
                    else float("nan")
                ),
                "first_within_25pct_train_doc_count": (
                    int(within_25.get("train_doc_count", 0)) if within_25 else 0
                ),
                "first_within_25pct_test_root_mae_mean": (
                    float(within_25.get("test_root_mae_mean", float("nan")))
                    if within_25
                    else float("nan")
                ),
            }
        )
    if not family_summaries:
        return {}
    cheapest_within_10 = sorted(
        [
            row
            for row in family_summaries
            if int(row.get("first_within_10pct_train_doc_count", 0)) > 0
        ],
        key=lambda row: (
            int(row.get("first_within_10pct_train_doc_count", 0)),
            float(row.get("first_within_10pct_test_root_mae_mean", float("inf"))),
        ),
    )
    return {
        "families": family_summaries,
        "cheapest_within_10pct": cheapest_within_10[0] if cheapest_within_10 else {},
    }


def _grid_diagnostic_summary(
    heatmap_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not heatmap_rows:
        return {"status": "insufficient_data"}

    by_family: Dict[str, List[Mapping[str, Any]]] = {}
    for row in heatmap_rows:
        by_family.setdefault(str(row.get("baseline_family", "")), []).append(row)

    sumlen_rows = by_family.get("official_fno_sumlen", [])
    default_rows = by_family.get("official_fno", [])
    sumlen_dominates = False
    if sumlen_rows and default_rows:
        default_by_cell = {
            (
                int(row.get("train_doc_count", 0)),
                int(row.get("n_regimes", 0)),
                str(row.get("segment_density_band", "")),
            ): float(row.get("test_root_mae_mean", float("nan")))
            for row in default_rows
        }
        paired_deltas: List[float] = []
        for row in sumlen_rows:
            key = (
                int(row.get("train_doc_count", 0)),
                int(row.get("n_regimes", 0)),
                str(row.get("segment_density_band", "")),
            )
            if key not in default_by_cell:
                continue
            paired_deltas.append(
                float(default_by_cell[key]) - float(row.get("test_root_mae_mean", float("nan")))
            )
        sumlen_dominates = bool(paired_deltas) and all(delta >= -1e-9 for delta in paired_deltas)
    else:
        paired_deltas = []

    control_families = (
        "palette_block_exact",
        "cnn1d",
        "raw_token_ngram_ridge",
    )
    control_exactness: Dict[str, Dict[str, Any]] = {}
    for family in control_families:
        family_rows = by_family.get(family, [])
        if not family_rows:
            continue
        max_mae = max(float(row.get("test_root_mae_mean", float("nan"))) for row in family_rows)
        min_exact = min(
            float(row.get("test_exact_match_rate_mean", float("nan"))) for row in family_rows
        )
        control_exactness[family] = {
            "max_root_mae_mean": float(max_mae),
            "min_exact_match_rate_mean": float(min_exact),
            "remains_exact_like": bool(max_mae <= 0.01 and min_exact >= 0.98),
        }

    target_family = "official_fno_sumlen" if sumlen_rows else "official_fno"
    family_rows = by_family.get(target_family, [])
    axis_effect = {"regime_span": float("nan"), "segment_span": float("nan")}
    main_failure_axis = "insufficient_data"
    if family_rows:
        regime_means: Dict[int, List[float]] = {}
        segment_means: Dict[str, List[float]] = {}
        for row in family_rows:
            regime_means.setdefault(int(row.get("n_regimes", 0)), []).append(
                float(row.get("test_root_mae_mean", float("nan")))
            )
            segment_means.setdefault(str(row.get("segment_density_band", "")), []).append(
                float(row.get("test_root_mae_mean", float("nan")))
            )
        regime_summary = {
            key: float(np.nanmean(np.asarray(values, dtype=np.float64)))
            for key, values in regime_means.items()
        }
        segment_summary = {
            key: float(np.nanmean(np.asarray(values, dtype=np.float64)))
            for key, values in segment_means.items()
        }
        regime_span = (
            max(regime_summary.values()) - min(regime_summary.values())
            if regime_summary
            else float("nan")
        )
        segment_span = (
            max(segment_summary.values()) - min(segment_summary.values())
            if segment_summary
            else float("nan")
        )
        axis_effect = {
            "regime_span": float(regime_span),
            "segment_span": float(segment_span),
            "regime_means": {str(int(k)): float(v) for k, v in regime_summary.items()},
            "segment_means": {str(k): float(v) for k, v in segment_summary.items()},
        }
        if not np.isfinite(regime_span) or not np.isfinite(segment_span):
            main_failure_axis = "insufficient_data"
        elif abs(float(regime_span) - float(segment_span)) <= 0.01:
            main_failure_axis = "both"
        elif float(regime_span) > float(segment_span):
            main_failure_axis = "regime_axis"
        else:
            main_failure_axis = "segment_axis"

    return {
        "status": "ok",
        "target_family": str(target_family),
        "official_fno_sumlen_dominates_official_fno": bool(sumlen_dominates),
        "official_fno_sumlen_minus_official_fno_mean_delta": (
            float(np.nanmean(np.asarray(paired_deltas, dtype=np.float64)))
            if paired_deltas
            else float("nan")
        ),
        "control_exactness": control_exactness,
        "main_failure_axis": str(main_failure_axis),
        "axis_effect": axis_effect,
    }


def write_full_doc_anchor_diagnostic_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(str(key))
            fieldnames.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _format_int_set(values: Sequence[Any]) -> str:
    normalized = sorted({int(value) for value in values})
    return "{" + ", ".join(str(int(value)) for value in normalized) + "}" if normalized else "{}"


def _format_histogram(hist: Mapping[str, Any]) -> str:
    items = sorted((int(key), int(value)) for key, value in dict(hist).items())
    return ", ".join(f"{int(key)}:{int(value)}" for key, value in items) if items else "NA"


def _ordered_train_doc_counts_from_payload(payload: Mapping[str, Any]) -> List[int]:
    train_doc_counts = payload.get("train_doc_counts")
    if isinstance(train_doc_counts, Mapping):
        values = {
            int(item)
            for seq in train_doc_counts.values()
            for item in list(seq or [])
        }
        return sorted(values)
    return sorted(int(item) for item in list(train_doc_counts or []))


def _representative_runs_by_unit(
    payload: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    representatives: Dict[str, Dict[str, Any]] = {}
    runs = sorted(
        list(payload.get("runs") or []),
        key=lambda run: (
            str(run.get("cell_id") or run.get("benchmark") or ""),
            int(run.get("train_doc_count", 0)),
            int(run.get("seed", 0)),
            str(run.get("baseline_family", "")),
        ),
    )
    for run in runs:
        key = str(run.get("cell_id") or run.get("benchmark") or "").strip()
        if not key or key in representatives:
            continue
        representatives[key] = dict(run)
    return representatives


def _fixed_eval_split_check(payload: Mapping[str, Any]) -> bool:
    runs = list(payload.get("runs") or [])
    if not runs:
        return False
    signatures: Dict[str, set[tuple[str, str]]] = {}
    for run in runs:
        key = str(run.get("cell_id") or run.get("benchmark") or "").strip()
        signatures.setdefault(key, set()).add(
            (
                str(run.get("val_corpus_signature", "")),
                str(run.get("test_corpus_signature", "")),
            )
        )
    return all(len(items) == 1 for items in signatures.values())


def _infer_report_mode_from_payload(payload: Mapping[str, Any]) -> str:
    hardness_grid = str(payload.get("hardness_grid", "")).strip()
    if not hardness_grid:
        return "recoverable_scale"
    train_doc_counts = _ordered_train_doc_counts_from_payload(payload)
    seeds = [int(seed) for seed in list(payload.get("seeds") or [])]
    if len(train_doc_counts) > 1:
        return "structural_grid"
    if len(seeds) > 1:
        return "structural_stability"
    return "structural_grid"


def _report_title_from_payload(payload: Mapping[str, Any], *, mode: str) -> str:
    if mode == "recoverable_scale":
        return "# Markov Recoverable Scale Report"
    if mode == "structural_stability":
        return "# Markov Structural Stability Report"
    return "# Markov Structural Grid Report"


def _narrative_baseline_families(payload: Mapping[str, Any]) -> List[str]:
    observed = [
        str(item).strip()
        for item in list(payload.get("baseline_families") or [])
        if str(item).strip()
    ]
    preferred = [
        family
        for family in (
            "official_fno",
            "official_fno_sumlen",
            "cnn1d",
            "palette_block_exact",
            "raw_token_ngram_ridge",
        )
        if family in observed
    ]
    return preferred or observed


def _control_exactness_for_family(
    payload: Mapping[str, Any],
    family: str,
) -> Dict[str, Any]:
    grid_summary = dict(payload.get("grid_diagnostic_summary") or {})
    control_exactness = dict(grid_summary.get("control_exactness") or {})
    if family in control_exactness:
        return dict(control_exactness[family])
    rows = [
        row
        for row in list(payload.get("aggregate_rows") or [])
        if str(row.get("baseline_family", "")) == str(family)
    ]
    if not rows:
        return {}
    max_root_mae = float(
        np.nanmax(
            np.asarray(
                [float(row.get("test_root_mae_mean", float("nan"))) for row in rows],
                dtype=np.float64,
            )
        )
    )
    min_exact_match = float(
        np.nanmin(
            np.asarray(
                [
                    float(row.get("test_exact_match_rate_mean", float("nan")))
                    for row in rows
                ],
                dtype=np.float64,
            )
        )
    )
    return {
        "remains_exact_like": bool(max_root_mae <= 1e-9 and min_exact_match >= 1.0 - 1e-9),
        "max_root_mae_mean": max_root_mae,
        "min_exact_match_rate_mean": min_exact_match,
    }


def _check_status(pass_value: bool | None) -> str:
    if pass_value is None:
        return "n/a"
    return "pass" if bool(pass_value) else "fail"


def _what_this_report_is_trying_to_show_lines(
    payload: Mapping[str, Any],
    *,
    mode: str,
) -> List[str]:
    lines = ["## What This Report Is Trying To Show", ""]
    benchmark_name = str(payload.get("benchmark", "")).strip().lower()
    hardness_grid = str(payload.get("hardness_grid", "")).strip().lower()
    sticky_surface = ("recoverable_v5" in benchmark_name) or ("structural_core_v2" in hardness_grid)
    if mode == "recoverable_scale":
        lines.extend(
            [
                (
                    "- Whether the fixed sticky recoverable full-document benchmark is now basically solved by "
                    "the learned full-doc lane once train scale increases, while the exact witness stays exact."
                    if sticky_surface
                    else "- Whether the fixed `recoverable_v4` full-document benchmark is now basically solved by the learned full-doc lane once train scale increases, while the exact witness stays exact."
                ),
                "- Whether the remaining gap is mostly a data-scale issue or an architecture/readout issue for the FNO family.",
            ]
        )
    elif mode == "structural_grid":
        lines.extend(
            [
                (
                    "- Whether the harder sticky stay/switch grid is still recoverable in principle once we increase context complexity along two axes: more regimes and higher boundary density."
                    if sticky_surface
                    else "- Whether the harder `piecewise_disjoint_palette` grid is still recoverable in principle once we increase context complexity along two axes: more regimes and denser segment schedules."
                ),
                "- Whether moving from `1024` to `10240` train documents rescues the learned full-doc models across the grid, and which hardness axis dominates at low scale.",
            ]
        )
    else:
        lines.extend(
            [
                "- Whether the remaining error on the publication-relevant structural anchor cells at `10240` train documents is mostly seed instability or a systematic residual bias.",
                "- Whether the same learned family wins consistently across the easiest, topics-hard, segments-hard, and hardest-both anchor cells.",
            ]
        )
    lines.append("")
    return lines


def _experimental_contract_markdown_lines(
    payload: Mapping[str, Any],
    *,
    mode: str,
) -> List[str]:
    lines = ["## Experimental Contract", ""]
    train_doc_counts = _ordered_train_doc_counts_from_payload(payload)
    seed_values = [int(seed) for seed in list(payload.get("seeds") or [])]
    baselines = _narrative_baseline_families(payload)
    benchmark = str(payload.get("benchmark", "")).strip()
    hardness_grid = str(payload.get("hardness_grid", "")).strip()
    bundle_manifest = dict(payload.get("bundle_manifest") or {})
    representative = next(iter(_representative_runs_by_unit(payload).values()), {})
    if mode == "recoverable_scale":
        lines.extend(
            [
                f"- comparison unit: one fixed saved benchmark bundle `{benchmark}` reused across train-doc sweeps `{_format_int_set(train_doc_counts)}`",
                f"- fixed seed sweep: `{_format_int_set(seed_values)}`",
                f"- fixed evaluation contract: same validation/test split reused across all train sizes = `{_fixed_eval_split_check(payload)}`",
                f"- main reported families: `{', '.join(baselines)}`",
                (
                    f"- primary paper score: `{str(payload.get('primary_report_metric', PRIMARY_REPORT_METRIC))}` "
                    f"on the `{str(payload.get('primary_report_split', PRIMARY_REPORT_SPLIT))}` split "
                    f"for `{str(payload.get('primary_report_target', PRIMARY_REPORT_TARGET))}` "
                    f"using `{str(payload.get('primary_report_weighting', PRIMARY_REPORT_WEIGHTING))}`"
                ),
                (
                    f"- dev/model-selection metric: `{str(payload.get('dev_selection_metric', DEV_SELECTION_METRIC))}`; "
                    "test metrics are for final reporting, not config selection"
                ),
                "- diagnostic-only views: train/val root MAE, train/val/test exact match, weighted training objective curves, unweighted val/test objective views, and local-law metrics.",
                "- common unweighted split objective: `root_mae + leaf_mae + c2_count_drift_r1_mae + merge_mae`, reported on validation and test for diagnosis only.",
                "- active-term unweighted split objective: `root_mae` plus the unit-weight sum of only the law terms active for that family, again reported on validation and test for diagnosis only.",
                "- merge-order spread (`schedule spread`) is not part of either objective; it is the root-count range across balanced, left-to-right, and right-to-left merge schedules.",
                "- exact witness interpretation: if `palette_block_exact` stays exact, any learned-model gap is not an information barrier in the DGP.",
            ]
        )
        if bundle_manifest:
            lines.append(f"- canonical saved bundle: `{next(iter(bundle_manifest.values()))}`")
    else:
        lines.extend(
            [
                f"- comparison unit: structural grid `{hardness_grid}` over saved per-cell bundles",
                f"- train-doc sweeps: `{_format_int_set(train_doc_counts)}`; seed sweep: `{_format_int_set(seed_values)}`",
                f"- fixed evaluation contract: same validation/test split reused within each cell = `{_fixed_eval_split_check(payload)}`",
                f"- main reported families: `{', '.join(baselines)}`",
                (
                    f"- primary paper score: `{str(payload.get('primary_report_metric', PRIMARY_REPORT_METRIC))}` "
                    "with train/val/objective/law quantities retained only as diagnostics"
                ),
                "- exact witness interpretation: if `palette_block_exact` stays exact, failures by learned baselines are failures to exploit visible local boundary evidence, not failures of recoverability.",
            ]
        )
        if representative:
            lines.append(
                f"- fixed document geometry: `{int(representative.get('min_tokens', 0) or 96)}` tokens/doc with segment lengths "
                f"`{int(representative.get('min_seg_len', 0) or 8)}-{int(representative.get('max_seg_len', 0) or 24)}`"
            )
        if bundle_manifest:
            lines.append(f"- per-cell bundle cache entries: `{len(bundle_manifest)}`")
    lines.append("")
    return lines


def _checks_markdown_lines(payload: Mapping[str, Any], *, mode: str) -> List[str]:
    representatives = _representative_runs_by_unit(payload)
    bundle_manifest = dict(payload.get("bundle_manifest") or {})
    exact_family = (
        "palette_block_exact"
        if "palette_block_exact" in list(payload.get("baseline_families") or [])
        else "raw_token_ngram_ridge"
        if "raw_token_ngram_ridge" in list(payload.get("baseline_families") or [])
        else ""
    )
    exact_stats = _control_exactness_for_family(payload, exact_family) if exact_family else {}
    if mode == "recoverable_scale":
        bundle_reuse = len(bundle_manifest) == 1
    else:
        bundle_reuse = bool(representatives) and len(bundle_manifest) == len(representatives)
    lines = ["## Checks", ""]
    lines.extend(
        [
            f"- `fixed_eval_splits_reused`: `{_check_status(_fixed_eval_split_check(payload))}`",
            f"- `bundle_reuse_within_comparison_units`: `{_check_status(bundle_reuse)}`",
            f"- `nondegenerate_target`: `{_check_status(not bool(payload.get('degenerate_benchmark', False)))}`",
        ]
    )
    if exact_family:
        lines.append(
            f"- `exact_witness_exact` ({exact_family}): `{_check_status(bool(exact_stats.get('remains_exact_like')) if exact_stats else None)}`"
        )
    lines.append("")
    return lines


def _setup_markdown_lines(payload: Mapping[str, Any]) -> List[str]:
    hardness_grid = str(payload.get("hardness_grid", ""))
    lines = ["## Setup", ""]
    runs = list(payload.get("runs") or [])
    if not runs:
        lines.extend(["- no runs available", ""])
        return lines
    sample_run = dict(runs[0])
    config = dict(sample_run.get("config") or {})
    train_doc_counts = _ordered_train_doc_counts_from_payload(payload)
    seed_values = [int(seed) for seed in list(payload.get("seeds") or [])]
    baselines = [str(item) for item in list(payload.get("baseline_families") or [])]
    bundle_manifest = dict(payload.get("bundle_manifest") or {})
    if hardness_grid:
        grid_cells = list(payload.get("grid_cells") or [])
        first_cell = dict(grid_cells[0]) if grid_cells else {}
        first_overrides = dict(first_cell.get("config_overrides") or {})
        regime_values = sorted(
            {
                int(cell.get("regime_count", 0))
                for cell in grid_cells
                if int(cell.get("regime_count", 0)) > 0
            }
        )
        segment_bands = sorted(
            {
                f"{str(cell.get('segment_density_band', ''))}={int(cell.get('segment_min', 0))}-{int(cell.get('segment_max', 0))}"
                for cell in grid_cells
                if str(cell.get("segment_density_band", "")).strip()
            }
        )
        lines.extend(
            [
                (
                    "- generator: `hazard_topic` with disjoint token blocks per regime and a simple stay/switch hazard process"
                    if str(first_overrides.get("generator_profile", "")).strip().lower() == "hazard_topic"
                    else "- generator: `piecewise_disjoint_palette` with disjoint token blocks per regime"
                ),
                (
                    f"- fixed token budget: `{int(first_overrides.get('min_tokens', 0))}` tokens per document; "
                    f"segment lengths `{int(first_overrides.get('min_seg_len', 0))}-{int(first_overrides.get('max_seg_len', 0))}`"
                ),
                (
                    f"- regime axis: `{_format_int_set(regime_values)}`; vocab scales as "
                    f"`4 * n_regimes`, so tokens-per-regime stay fixed"
                ),
                f"- segment-density axis: `{', '.join(segment_bands)}`",
                (
                    "- per-cell density calibration: each grid cell keeps its regime count and target density band, "
                    "but the actual number of boundaries is stochastic under the stay/switch process"
                    if str(first_overrides.get("generator_profile", "")).strip().lower() == "hazard_topic"
                    else "- distinct-regime constraint per cell: "
                    "`min(n_regimes, min_segments)` through `min(n_regimes, max_segments)`"
                ),
                f"- train-doc sweeps: `{_format_int_set(train_doc_counts)}`; seeds: `{_format_int_set(seed_values)}`",
                (
                    f"- fixed validation/test docs per cell: "
                    f"`{int(first_overrides.get('val_docs', 0))}/{int(first_overrides.get('test_docs', 0))}`"
                ),
                f"- baseline families: `{', '.join(baselines)}`",
                f"- fixed evaluation splits reused within each cell: `{_fixed_eval_split_check(payload)}`",
            ]
        )
        if bundle_manifest:
            lines.append(
                f"- per-cell bundles cached under: `{next(iter(bundle_manifest.values()))}`"
            )
    else:
        benchmark_spec = dict(payload.get("benchmark_spec") or {})
        observed_profile = str(benchmark_spec.get("observed_token_profile", "")).strip()
        policy = (
            resolve_markov_observed_token_policy(profile_name=observed_profile)
            if observed_profile
            else None
        )
        target_support = dict(sample_run.get("target_support") or {})
        distinct_support = dict(sample_run.get("distinct_regime_support") or {})
        lines.extend(
            [
                f"- generator: `{str(getattr(policy, 'generator_profile', 'unknown'))}`",
                (
                    f"- regimes / vocab: `{int(getattr(policy, 'n_regimes', config.get('n_regimes', 0)))}` regimes, "
                    f"`{int(getattr(policy, 'vocab_size', config.get('vocab_size', 0)))}` observed tokens"
                ),
                (
                    f"- document length: `{int(getattr(policy, 'min_tokens', 0))}-{int(getattr(policy, 'max_tokens', 0))}` tokens; "
                    f"segments `{int(getattr(policy, 'min_segments', config.get('min_segments', 0)))}-{int(getattr(policy, 'max_segments', config.get('max_segments', 0)))}`"
                ),
                (
                    f"- fixed validation/test docs: "
                    f"`{int(getattr(policy, 'val_docs', 0))}/{int(getattr(policy, 'test_docs', 0))}`"
                ),
                f"- train-doc sweeps: `{_format_int_set(train_doc_counts)}`; seeds: `{_format_int_set(seed_values)}`",
                f"- baseline families: `{', '.join(baselines)}`",
                f"- fixed evaluation split reused across train-doc sweeps: `{_fixed_eval_split_check(payload)}`",
            ]
        )
        if target_support:
            lines.append(
                f"- root-count support on test split: `{_format_int_set(dict(target_support.get('test') or {}).get('values', []))}`"
            )
        if distinct_support:
            lines.append(
                f"- distinct-regime support on test split: `{_format_int_set(dict(distinct_support.get('test') or {}).get('values', []))}`"
            )
        if bundle_manifest:
            lines.append(f"- canonical bundle: `{next(iter(bundle_manifest.values()))}`")
    lines.append("")
    return lines


def _dgp_intuition_markdown_lines(payload: Mapping[str, Any]) -> List[str]:
    benchmark_name = str(payload.get("benchmark", "")).strip().lower()
    hardness_grid = str(payload.get("hardness_grid", "")).strip().lower()
    sticky_surface = ("recoverable_v5" in benchmark_name) or ("structural_core_v2" in hardness_grid)
    lines = [
        "## DGP And Why It Is Recoverable",
        "",
        (
            "- Generator: `hazard_topic`. Each token either stays in the current hidden regime or switches to a different regime, and each hidden regime owns its own disjoint token palette, so adjacent token pairs still carry direct evidence about whether a regime boundary occurred."
            if sticky_surface
            else "- Generator: `piecewise_disjoint_palette`. Each hidden regime owns its own disjoint token palette, so adjacent token pairs carry direct evidence about whether a regime boundary occurred."
        ),
        "- The root/document target is the number of changepoints, i.e. the number of adjacent regime switches across the full token sequence.",
        "- `palette_block_exact` is the recoverability witness: it maps each token to its palette block and counts adjacent block changes. If it stays exact, the benchmark is still recoverable and model error is not an information barrier.",
    ]
    if str(payload.get("hardness_grid", "")):
        lines.extend(
            [
                "- Increasing `n_regimes` makes the identity problem harder because the model must distinguish more topic palettes while keeping only four tokens per regime.",
                (
                    "- Increasing the hazard-calibrated density band makes the counting problem harder because the document contains more expected boundaries and shorter average runs, so missing one local transition is enough to change the global count."
                    if sticky_surface
                    else "- Increasing segment density makes the counting problem harder because the document contains more boundaries and more short segments, so missing one local transition is enough to change the global count."
                ),
                (
                    "- The sticky version broadens support by allowing the actual number of changepoints to vary around the target density band instead of enforcing a hard segment-count constraint."
                    if sticky_surface
                    else "- Forcing more distinct regimes per document removes the easy failure mode where a document reuses only a tiny subset of the available topics."
                ),
            ]
        )
    else:
        lines.extend(
            [
                "- The recoverable benchmark is intentionally nondegenerate but simple: the full document contains enough signal to solve the count exactly, so any residual gap is about optimization or inductive bias rather than missing information.",
                "- The useful comparison is therefore not only FNO versus zero, but learned full-document models versus exact controls and local-transition baselines on the same fixed bundle.",
            ]
        )
    lines.append("")
    return lines


def _key_findings_markdown_lines(payload: Mapping[str, Any]) -> List[str]:
    lines = ["## Headline Findings", ""]
    aggregate_rows = list(payload.get("aggregate_rows") or [])
    heatmap_rows = list(payload.get("heatmap_rows") or [])
    grid_summary = dict(payload.get("grid_diagnostic_summary") or {})
    diagnostic = dict(payload.get("diagnostic_readout") or {})
    if str(payload.get("hardness_grid", "")):
        target_family = str(grid_summary.get("target_family", "official_fno"))
        lines.extend(
            [
                f"- target learned family for interpretation: `{target_family}`",
                f"- sum+length dominates default FNO everywhere: `{bool(grid_summary.get('official_fno_sumlen_dominates_official_fno', False))}`",
                f"- main failure axis at low scale: `{str(grid_summary.get('main_failure_axis', 'unknown'))}`",
            ]
        )
        control_exactness = dict(grid_summary.get("control_exactness") or {})
        if "palette_block_exact" in control_exactness:
            stats = dict(control_exactness["palette_block_exact"])
            lines.append(
                f"- exact witness remains exact across the grid: `{bool(stats.get('remains_exact_like', False))}` "
                f"(max mean root MAE `{float(stats.get('max_root_mae_mean', float('nan'))):.6g}`)"
            )
        if "cnn1d" in control_exactness:
            stats = dict(control_exactness["cnn1d"])
            lines.append(
                f"- learned local-transition control remains exact-like across all cells: `{bool(stats.get('remains_exact_like', False))}`"
            )
        target_rows = [
            row for row in heatmap_rows if str(row.get("baseline_family", "")) == target_family
        ]
        train_doc_counts = sorted(
            {int(row.get("train_doc_count", 0)) for row in target_rows}
        )
        for train_doc_count in train_doc_counts:
            subset = [
                row
                for row in target_rows
                if int(row.get("train_doc_count", 0)) == int(train_doc_count)
            ]
            if not subset:
                continue
            hardest = max(subset, key=lambda row: float(row.get("test_root_mae_mean", float("nan"))))
            easiest = min(subset, key=lambda row: float(row.get("test_root_mae_mean", float("nan"))))
            mean_mae = float(
                np.nanmean(
                    np.asarray(
                        [float(row.get("test_root_mae_mean", float("nan"))) for row in subset],
                        dtype=np.float64,
                    )
                )
            )
            lines.append(
                f"- at train_docs=`{int(train_doc_count)}`, mean `{target_family}` root MAE is `{mean_mae:.6g}`; "
                f"easiest cell is `{str(easiest.get('cell_id', ''))}` at `{float(easiest.get('test_root_mae_mean', float('nan'))):.6g}`, "
                f"hardest is `{str(hardest.get('cell_id', ''))}` at `{float(hardest.get('test_root_mae_mean', float('nan'))):.6g}`"
            )
        if len(train_doc_counts) >= 2:
            base_count = int(train_doc_counts[0])
            final_count = int(train_doc_counts[-1])
            by_cell = {}
            for row in target_rows:
                by_cell.setdefault(str(row.get("cell_id", "")), {})[
                    int(row.get("train_doc_count", 0))
                ] = float(row.get("test_root_mae_mean", float("nan")))
            deltas = []
            for cell_id, per_count in by_cell.items():
                if base_count not in per_count or final_count not in per_count:
                    continue
                deltas.append((cell_id, float(per_count[base_count] - per_count[final_count])))
            if deltas:
                mean_gain = float(
                    np.nanmean(np.asarray([delta for _, delta in deltas], dtype=np.float64))
                )
                best_gain_cell, best_gain = max(deltas, key=lambda item: float(item[1]))
                lines.append(
                    f"- scaling from `{base_count}` to `{final_count}` train docs cuts mean `{target_family}` root MAE by `{mean_gain:.6g}` on average; "
                    f"largest cell-wise gain is `{best_gain_cell}` with delta `{best_gain:.6g}`"
                )
    else:
        status = str(diagnostic.get("status", "unknown"))
        lines.append(f"- diagnosis status: `{status}`")
        if "fno_best_root_mae_mean" in diagnostic:
            lines.append(
                f"- best official FNO root MAE is `{float(diagnostic['fno_best_root_mae_mean']):.6g}` at train_docs=`{int(diagnostic['fno_best_train_doc_count'])}`"
            )
            lines.append(
                f"- best control is `{str(diagnostic.get('best_control_family', ''))}` with root MAE `{float(diagnostic.get('best_control_root_mae_mean', float('nan'))):.6g}`"
            )
            lines.append(
                f"- FNO data-scale gain from smallest to best train size is `{float(diagnostic.get('fno_data_scale_gain', float('nan'))):.6g}`"
            )
            lines.append(
                f"- residual gap to the best control at the best FNO point is `{float(diagnostic.get('gap_to_best_control', float('nan'))):.6g}`"
            )
        learned_rows = [
            row
            for row in aggregate_rows
            if str(row.get("baseline_family", ""))
            not in {"palette_block_exact", "raw_token_ngram_ridge"}
        ]
        if learned_rows:
            best_learned = min(
                learned_rows,
                key=lambda row: float(row.get("test_root_mae_mean", float("nan"))),
            )
            lines.append(
                f"- best learned family overall is `{str(best_learned.get('baseline_family', ''))}` at train_docs=`{int(best_learned.get('train_doc_count', 0))}` "
                f"with mean root MAE `{float(best_learned.get('test_root_mae_mean', float('nan'))):.6g}`"
            )
        parity = dict(payload.get("tree_fno_fair_parity_summary") or {})
        if parity:
            gate_count = int(parity.get("gate_train_doc_count", 0))
            lines.append(
                f"- fair-parity gate at train_docs=`{gate_count}`: primary success "
                f"`{bool(parity.get('primary_success_met', False))}`, secondary success "
                f"`{bool(parity.get('secondary_success_met', False))}`"
            )
            if str(parity.get("best_full_doc_fno_family_at_gate", "")).strip():
                lines.append(
                    f"- best full-doc FNO at the gate is `{str(parity.get('best_full_doc_fno_family_at_gate', ''))}`, "
                    f"best parity-tree is `{str(parity.get('best_parity_tree_family_at_gate', ''))}`, "
                    f"and `tree_neural` gap vs best FNO is "
                    f"`{100.0 * float(parity.get('tree_neural_gap_ratio_vs_best_fno_at_gate', float('nan'))):.3g}%`"
                )
        budget_frontier = dict(payload.get("tree_oracle_budget_frontier_summary") or {})
        best_tree_by_budget = list(budget_frontier.get("best_tree_by_budget") or [])
        if best_tree_by_budget:
            cheapest = min(
                best_tree_by_budget,
                key=lambda row: (
                    float(row.get("budget_total_calls_per_doc", float("inf"))),
                    float(row.get("test_root_mae_mean", float("inf"))),
                ),
            )
            lines.append(
                "- oracle-attention frontier active: cheapest tree point recorded is "
                f"`{str(cheapest.get('baseline_family', ''))}` at "
                f"`{float(cheapest.get('budget_total_calls_per_doc', float('nan'))):.6g}` "
                "calls/doc with full-doc share "
                f"`{float(cheapest.get('full_doc_budget_share', float('nan'))):.6g}` "
                f"and test root MAE `{float(cheapest.get('test_root_mae_mean', float('nan'))):.6g}`"
            )
    lines.append("")
    return lines


def _figure_first_interpretation_markdown_lines(
    payload: Mapping[str, Any],
    *,
    mode: str,
) -> List[str]:
    lines = ["## Figure-First Interpretation", ""]
    if mode == "recoverable_scale":
        lines.extend(
            [
                "- Read the test root-MAE ranking first: it is the only paper-facing score used to compare families.",
                "- If multiple train scales are present, read the efficiency frontier next: it identifies the earliest train size that is already near a learned family's best point.",
                "- Then read the split-diagnostic pages: train/val root MAE and exact-match tell you whether the model is fitting cleanly or merely making nearby count errors.",
                "- Then read the unweighted validation/test objective page: it summarizes theorem-facing errors on a common scale, but it is still diagnostic-only.",
                "- Read weighted objective and local-law pages last: they are optimization and merge-order diagnostics, not replacement ranking metrics.",
            ]
        )
    elif mode == "structural_grid":
        train_doc_counts = _ordered_train_doc_counts_from_payload(payload)
        if len(train_doc_counts) >= 2:
            lines.append(
                f"- Read the `{int(train_doc_counts[0])}`-doc heatmaps first, then the `{int(train_doc_counts[-1])}`-doc heatmaps, then the delta page. That order isolates low-scale failure structure before showing what extra data rescues."
            )
        else:
            lines.append(
                "- Read the heatmaps first, then the line slices. The heatmaps show where the benchmark is hard; the slices tell you whether regimes or segment density dominate."
            )
        lines.extend(
            [
                "- The regime-slice plots ask whether adding more topics hurts mainly by palette identification; the segment-slice plots ask whether denser local transition schedules are the real long pole.",
                "- The exact witness staying on the floor matters more than any one learned curve: it means the grid is still a recoverable benchmark rather than a broken one.",
            ]
        )
    else:
        lines.extend(
            [
                "- Read the point-range plots first. If the bars are small relative to the absolute error, the residual problem is systematic bias, not seed noise.",
                "- Then read the mean/std table page to see whether instability concentrates on one anchor cell or stays broad across all four anchor cells.",
                "- The exact witness remaining pinned at zero is the anchor sanity check; the question is how much of the remaining learned gap is variance versus misspecification.",
            ]
        )
    lines.append("")
    return lines


def _best_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    reverse: bool = False,
) -> Mapping[str, Any] | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: float(row.get(metric_key, float("nan"))),
        reverse=bool(reverse),
    )[0]


def _compact_key_tables_markdown_lines(
    payload: Mapping[str, Any],
    *,
    mode: str,
) -> List[str]:
    aggregate_rows = list(payload.get("aggregate_rows") or [])
    if not aggregate_rows:
        return []
    highlight = set(_narrative_baseline_families(payload))
    lines = ["## Compact Key Tables", ""]
    if mode == "recoverable_scale":
        leaf_tokens_vary = len(
            {int(row.get("fixed_leaf_tokens", 0)) for row in aggregate_rows if int(row.get("fixed_leaf_tokens", 0)) > 0}
        ) > 1
        lines.extend(
            [
                "### Learned Frontier And Controls",
                "",
                "_Ranking below is by mean **test** root-count MAE. Exact-match is included only as a secondary diagnostic column._",
                "",
                (
                    "| family | train_docs | leaf_tokens | mean root_mae | std root_mae | mean exact-match |"
                    if leaf_tokens_vary
                    else "| family | train_docs | mean root_mae | std root_mae | mean exact-match |"
                ),
                (
                    "|---|---:|---:|---:|---:|---:|"
                    if leaf_tokens_vary
                    else "|---|---:|---:|---:|---:|"
                ),
            ]
        )
        selected = [
            row
            for row in aggregate_rows
            if str(row.get("baseline_family", "")) in highlight
        ]
        for row in sorted(
            selected,
            key=lambda item: (
                str(item.get("baseline_family", "")),
                int(item.get("train_doc_count", 0)),
                int(item.get("fixed_leaf_tokens", 0)),
            ),
        ):
            leaf_token_prefix = ""
            if leaf_tokens_vary:
                leaf_token_prefix = f"{int(row.get('fixed_leaf_tokens', 0))} | "
            lines.append(
                "| "
                f"{str(row.get('baseline_family', ''))} | "
                f"{int(row.get('train_doc_count', 0))} | "
                f"{leaf_token_prefix}"
                f"{float(row.get('test_root_mae_mean', float('nan'))):.6g} | "
                f"{float(row.get('test_root_mae_std', float('nan'))):.6g} | "
                f"{float(row.get('test_exact_match_rate_mean', float('nan'))):.6g} |"
            )
        budget_frontier = dict(payload.get("tree_oracle_budget_frontier_summary") or {})
        best_tree_rows = list(budget_frontier.get("best_tree_by_budget") or [])
        if best_tree_rows:
            lines.extend(
                [
                    "",
                    "### Oracle Attention Budget Share",
                    "",
                    "| calls/doc | best tree family | test root_mae | full-doc share | doc mode | local split | eff full-doc mass/doc |",
                    "|---:|---|---:|---:|---|---|---:|",
                ]
            )
            for row in best_tree_rows:
                lines.append(
                    "| "
                    f"{float(row.get('budget_total_calls_per_doc', float('nan'))):.6g} | "
                    f"{str(row.get('baseline_family', ''))} | "
                    f"{float(row.get('test_root_mae_mean', float('nan'))):.6g} | "
                    f"{float(row.get('full_doc_budget_share', float('nan'))):.6g} | "
                    f"{str(row.get('doc_consumption_mode', ''))} | "
                    f"{str(row.get('local_split_mode', ''))} | "
                    f"{float(row.get('effective_full_doc_mass_per_doc_mean', float('nan'))):.6g} |"
                )
    elif mode == "structural_grid":
        lines.extend(
            [
                "### Grid Headline Table",
                "",
                "| family | train_docs | mean grid root_mae | best cell | worst cell |",
                "|---|---:|---:|---|---|",
            ]
        )
        grouped: Dict[tuple[str, int], List[Mapping[str, Any]]] = {}
        for row in aggregate_rows:
            family = str(row.get("baseline_family", ""))
            if family not in highlight:
                continue
            grouped.setdefault(
                (family, int(row.get("train_doc_count", 0))),
                [],
            ).append(row)
        for (family, train_doc_count), rows in sorted(grouped.items()):
            mean_mae = float(
                np.nanmean(
                    np.asarray(
                        [float(row.get("test_root_mae_mean", float("nan"))) for row in rows],
                        dtype=np.float64,
                    )
                )
            )
            best = _best_row(rows, metric_key="test_root_mae_mean", reverse=False)
            worst = _best_row(rows, metric_key="test_root_mae_mean", reverse=True)
            lines.append(
                "| "
                f"{family} | "
                f"{int(train_doc_count)} | "
                f"{mean_mae:.6g} | "
                f"{str(best.get('cell_id', '')) if best else 'NA'} ({float(best.get('test_root_mae_mean', float('nan'))):.6g}) | "
                f"{str(worst.get('cell_id', '')) if worst else 'NA'} ({float(worst.get('test_root_mae_mean', float('nan'))):.6g}) |"
            )
    else:
        lines.extend(
            [
                "### Anchor Stability Table",
                "",
                "| cell | family | mean root_mae | std | min | max | mean exact-match |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        selected = [
            row
            for row in aggregate_rows
            if str(row.get("baseline_family", "")) in highlight
        ]
        for row in sorted(
            selected,
            key=lambda item: (
                int(item.get("n_regimes", 0)),
                int(item.get("segment_min", 0)),
                str(item.get("baseline_family", "")),
            ),
        ):
            lines.append(
                "| "
                f"{str(row.get('cell_id', ''))} | "
                f"{str(row.get('baseline_family', ''))} | "
                f"{float(row.get('test_root_mae_mean', float('nan'))):.6g} | "
                f"{float(row.get('test_root_mae_std', float('nan'))):.6g} | "
                f"{float(row.get('test_root_mae_min', float('nan'))):.6g} | "
                f"{float(row.get('test_root_mae_max', float('nan'))):.6g} | "
                f"{float(row.get('test_exact_match_rate_mean', float('nan'))):.6g} |"
            )
    lines.append("")
    return lines


def _what_reader_should_conclude_lines(
    payload: Mapping[str, Any],
    *,
    mode: str,
) -> List[str]:
    aggregate_rows = list(payload.get("aggregate_rows") or [])
    highlight = set(_narrative_baseline_families(payload))
    lines = ["## What The Reader Should Conclude", ""]
    if mode == "recoverable_scale":
        learned_rows = [
            row
            for row in aggregate_rows
            if str(row.get("baseline_family", "")) in {"official_fno", "official_fno_sumlen"}
        ]
        best_learned = _best_row(learned_rows, metric_key="test_root_mae_mean")
        exact_stats = _control_exactness_for_family(payload, "palette_block_exact")
        efficiency = dict(payload.get("learning_efficiency_summary") or {})
        cheapest = dict(efficiency.get("cheapest_within_10pct") or {})
        if best_learned:
            lines.append(
                f"- On the fixed recoverable bundle, the learned full-doc lane is close to solved at its best point: `{str(best_learned.get('baseline_family', ''))}` reaches mean root MAE `{float(best_learned.get('test_root_mae_mean', float('nan'))):.6g}` at train_docs=`{int(best_learned.get('train_doc_count', 0))}`."
            )
        elif cheapest:
            lines.append(
                f"- The cheapest near-best learned point currently visible is `{str(cheapest.get('baseline_family', ''))}` at train_docs=`{int(cheapest.get('first_within_10pct_train_doc_count', 0))}`, already within 10% of that family's best test root MAE."
            )
        if exact_stats:
            lines.append(
                f"- The exact witness remains exact (`{bool(exact_stats.get('remains_exact_like', False))}`), so the residual gap is a learned-model issue rather than a recoverability issue."
            )
        lines.append("- Scaling data is part of the story, but architecture/readout choices still matter because `official_fno_sumlen` can outperform the default FNO at the same train scale.")
        parity = dict(payload.get("tree_fno_fair_parity_summary") or {})
        if parity:
            interpretation = str(parity.get("comparison_interpretation", "")).strip()
            if interpretation:
                lines.append(
                    f"- Under the explicit fair-parity tree preset, the current reading is `{interpretation}`."
                )
            if bool(parity.get("primary_success_met", False)):
                lines.append(
                    "- The full-law tree model reaches the parity threshold against the best full-doc FNO at the 10k gate, so the older gap was at least partly a training/objective mismatch."
                )
            elif bool(parity.get("secondary_success_met", False)):
                lines.append(
                    "- The full-law tree model misses the parity threshold, but another parity-tagged tree family reaches it; that shifts the question from tree structure to which local-law mix is still helping at equalized capacity."
                )
            else:
                lines.append(
                    "- Even after equalizing root supervision and leaf-FNO capacity, the parity-tagged tree families still miss the 10% gate; that makes the remaining gap look like a real recursive-operator limitation rather than a reporting artifact."
                )
    elif mode == "structural_grid":
        grid_summary = dict(payload.get("grid_diagnostic_summary") or {})
        target_family = str(grid_summary.get("target_family", "official_fno_sumlen"))
        lines.append(
            f"- The structural grid is recoverable by construction because `palette_block_exact` stays exact everywhere; the interesting failures are learned-model failures, not DGP failures."
        )
        lines.append(
            f"- At low scale, the main failure axis is `{str(grid_summary.get('main_failure_axis', 'unknown'))}` for the target learned family `{target_family}`."
        )
        lines.append("- Moving from `1024` to `10240` train documents is the decisive comparison: it shows whether added data rescues structural context rather than just helping on the easiest cell.")
        if "official_fno_sumlen" in highlight:
            lines.append("- `official_fno_sumlen` is the current publication-facing learned candidate, but the grid should still show where it improves and where it does not uniformly dominate the default FNO.")
    else:
        target_family = str(
            dict(payload.get("grid_diagnostic_summary") or {}).get("target_family", "official_fno_sumlen")
        )
        target_rows = [
            row
            for row in aggregate_rows
            if str(row.get("baseline_family", "")) == target_family
        ]
        mean_std = float(
            np.nanmean(
                np.asarray(
                    [float(row.get("test_root_mae_std", float("nan"))) for row in target_rows],
                    dtype=np.float64,
                )
            )
        ) if target_rows else float("nan")
        mean_mae = float(
            np.nanmean(
                np.asarray(
                    [float(row.get("test_root_mae_mean", float("nan"))) for row in target_rows],
                    dtype=np.float64,
                )
            )
        ) if target_rows else float("nan")
        lines.append(
            f"- The stability report should be read as a bias-vs-variance check on `{target_family}` at the final train scale, not as a new benchmark sweep."
        )
        lines.append(
            f"- Mean seed std for `{target_family}` is `{mean_std:.6g}` against mean root MAE `{mean_mae:.6g}`, so the remaining gap is {'mostly instability' if np.isfinite(mean_std) and np.isfinite(mean_mae) and mean_std >= 0.5 * mean_mae else 'mostly systematic bias'}."
        )
        lines.append("- The exact witness staying at zero across all anchor cells is the contract check that keeps the interpretation honest.")
    lines.append("")
    return lines


def _full_appendix_markdown_lines(payload: Mapping[str, Any]) -> List[str]:
    lines = ["## Full Aggregates / Diagnostics Appendix", ""]
    lines.extend(_support_markdown_lines(payload))
    aggregate_rows = list(payload.get("aggregate_rows") or [])
    if aggregate_rows:
        lines.extend(
            [
                "### Split Diagnostics",
                "",
                "| family | train_docs | train root_mae | val root_mae | test root_mae | train exact-match | val exact-match | test exact-match |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in aggregate_rows:
            lines.append(
                "| "
                f"{str(row.get('baseline_family', ''))} | "
                f"{int(row.get('train_doc_count', 0))} | "
                f"{float(row.get('train_root_mae_mean', float('nan'))):.6g} | "
                f"{float(row.get('val_root_mae_mean', float('nan'))):.6g} | "
                f"{float(row.get('test_root_mae_mean', float('nan'))):.6g} | "
                f"{float(row.get('train_exact_match_rate_mean', float('nan'))):.6g} | "
                f"{float(row.get('val_exact_match_rate_mean', float('nan'))):.6g} | "
                f"{float(row.get('test_exact_match_rate_mean', float('nan'))):.6g} |"
            )
        lines.extend(
            [
                "",
                "### Unweighted Validation/Test Objectives",
                "",
                "| family | train_docs | full-law val objective | active-term val objective | full-law test objective | active-term test objective | merge-order spread |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in aggregate_rows:
            lines.append(
                "| "
                f"{str(row.get('baseline_family', ''))} | "
                f"{int(row.get('train_doc_count', 0))} | "
                f"{float(row.get('val_unweighted_full_law_objective_mean', float('nan'))):.6g} | "
                f"{float(row.get('val_unweighted_active_objective_mean', float('nan'))):.6g} | "
                f"{float(row.get('test_unweighted_full_law_objective_mean', float('nan'))):.6g} | "
                f"{float(row.get('test_unweighted_active_objective_mean', float('nan'))):.6g} | "
                f"{float(row.get('test_schedule_spread_mean_mean', float('nan'))):.6g} |"
            )
        lines.extend(
            [
                "",
                "### Diagnostic-Only Metric Roles",
                "",
                f"- primary report metric: `{str(payload.get('primary_report_metric', PRIMARY_REPORT_METRIC))}`",
                f"- dev selection metric: `{str(payload.get('dev_selection_metric', DEV_SELECTION_METRIC))}`",
                "- common unweighted split objective: `root_mae + leaf_mae + c2_count_drift_r1_mae + merge_mae`, reported on validation and test for diagnosis only.",
                "- active-term unweighted split objective: `root_mae` plus the unit-weight sum of the law terms active for that family, reported on validation and test for diagnosis only.",
                "- merge-order spread (`schedule spread`) is `max(pred_root) - min(pred_root)` across balanced, left-to-right, and right-to-left merge schedules; it is a stability diagnostic, not an objective term.",
                "- weighted training objective curves and local-law metrics are retained for diagnosis only.",
                "",
            ]
        )
        efficiency = dict(payload.get("learning_efficiency_summary") or {})
        family_rows = list(efficiency.get("families") or [])
        if family_rows:
            lines.extend(
                [
                    "### Learning Efficiency Frontier",
                    "",
                    "| family | best train_docs | best test root_mae | first within 10% of best | first within 25% of best |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for row in family_rows:
                lines.append(
                    "| "
                    f"{str(row.get('baseline_family', ''))} | "
                    f"{int(row.get('best_train_doc_count', 0))} | "
                    f"{float(row.get('best_test_root_mae_mean', float('nan'))):.6g} | "
                    f"{int(row.get('first_within_10pct_train_doc_count', 0))} | "
                    f"{int(row.get('first_within_25pct_train_doc_count', 0))} |"
                )
            lines.append("")
        parity = dict(payload.get("tree_fno_fair_parity_summary") or {})
        parity_rows = list(parity.get("comparisons") or [])
        if parity_rows:
            lines.extend(
                [
                    "### FNO vs Tree Fair-Parity",
                    "",
                    f"- parity config label: `{str(parity.get('parity_config_label', ''))}`",
                    f"- tree root supervision kind: `{str(parity.get('tree_root_supervision_kind', ''))}`",
                    f"- tree leaf FNO: width=`{parity.get('tree_leaf_fno_width')}`, modes=`{parity.get('tree_leaf_fno_n_modes')}`, layers=`{parity.get('tree_leaf_fno_n_layers')}`",
                    f"- tree aux doc-sequence fraction: `{float(parity.get('tree_aux_doc_sequence_fraction', float('nan'))):.6g}`",
                    f"- primary success met: `{bool(parity.get('primary_success_met', False))}`",
                    f"- secondary success met: `{bool(parity.get('secondary_success_met', False))}`",
                    "",
                    "| train_docs | best_fno | best_fno_mae | tree_neural_mae | best_parity_tree | best_parity_tree_mae | tree_neural gap vs best_fno |",
                    "|---|---|---:|---:|---|---:|---:|",
                ]
            )
            for row in parity_rows:
                lines.append(
                    "| "
                    f"{int(row.get('train_doc_count', 0))} | "
                    f"{str(row.get('best_full_doc_fno_family', ''))} | "
                    f"{float(row.get('best_full_doc_fno_test_root_mae_mean', float('nan'))):.6g} | "
                    f"{float(row.get('tree_neural_test_root_mae_mean', float('nan'))):.6g} | "
                    f"{str(row.get('best_parity_tree_family', ''))} | "
                    f"{float(row.get('best_parity_tree_test_root_mae_mean', float('nan'))):.6g} | "
                    f"{100.0 * float(row.get('tree_neural_gap_ratio_vs_best_fno', float('nan'))):.3g}% |"
                )
            lines.append("")
        upper_bound = dict(payload.get("tree_fno_upper_bound_summary") or {})
        upper_rows = list(upper_bound.get("comparisons") or [])
        if upper_rows:
            lines.extend(
                [
                    "### Tree+Aux Upper Bound",
                    "",
                    "| train_docs | aux_frac | best_fno | best_fno_mae | tree_neural_mae | best_upper_bound_tree | best_upper_bound_tree_mae |",
                    "|---|---:|---|---:|---:|---|---:|",
                ]
            )
            for row in upper_rows:
                lines.append(
                    "| "
                    f"{int(row.get('train_doc_count', 0))} | "
                    f"{float(row.get('tree_aux_doc_sequence_fraction', float('nan'))):.6g} | "
                    f"{str(row.get('best_full_doc_fno_family', ''))} | "
                    f"{float(row.get('best_full_doc_fno_test_root_mae_mean', float('nan'))):.6g} | "
                    f"{float(row.get('tree_neural_test_root_mae_mean', float('nan'))):.6g} | "
                    f"{str(row.get('best_upper_bound_tree_family', ''))} | "
                    f"{float(row.get('best_upper_bound_tree_test_root_mae_mean', float('nan'))):.6g} |"
                )
            lines.append("")
        lines.extend(
            [
                "### Separate Tuning Report",
                "",
                "- Width/layers/modes sweeps, runtime frontiers, and tree+aux appendix comparisons belong in the dedicated tree-FNO tuning PDF rather than replacing the main recoverable ranking pages.",
                "",
            ]
        )
    lines.extend(
        [
            "### Aggregate Rows",
            "",
            "| family | train_docs | cell | n_runs | mean root_mae | std root_mae | mean exact-match |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| "
            f"{str(row.get('baseline_family', ''))} | "
            f"{int(row.get('train_doc_count', 0))} | "
            f"{str(row.get('cell_id') or row.get('benchmark') or '')} | "
            f"{int(row.get('n_runs', 0))} | "
            f"{float(row.get('test_root_mae_mean', float('nan'))):.6g} | "
            f"{float(row.get('test_root_mae_std', float('nan'))):.6g} | "
            f"{float(row.get('test_exact_match_rate_mean', float('nan'))):.6g} |"
        )
    heatmap_rows = list(payload.get("heatmap_rows") or [])
    if heatmap_rows:
        lines.extend(
            [
                "",
                "### Heatmap Rows",
                "",
                "| family | regimes | segment band | train_docs | mean root_mae | mean exact-match |",
                "|---|---:|---|---:|---:|---:|",
            ]
        )
        for row in heatmap_rows:
            lines.append(
                "| "
                f"{str(row.get('baseline_family', ''))} | "
                f"{int(row.get('n_regimes', 0))} | "
                f"{str(row.get('segment_density_band', ''))} | "
                f"{int(row.get('train_doc_count', 0))} | "
                f"{float(row.get('test_root_mae_mean', float('nan'))):.6g} | "
                f"{float(row.get('test_exact_match_rate_mean', float('nan'))):.6g} |"
            )
    lines.extend(
        [
            "",
            "### File Artifacts",
            "",
            f"- runs CSV: `{str(payload.get('runs_csv', ''))}`",
            f"- aggregate CSV: `{str(payload.get('aggregate_csv', ''))}`",
            f"- heatmap CSV: `{str(payload.get('heatmap_csv', ''))}`",
            "",
        ]
    )
    return lines


def _scale_effects_markdown_lines(payload: Mapping[str, Any]) -> List[str]:
    aggregate_rows = list(payload.get("aggregate_rows") or [])
    if not aggregate_rows:
        return []
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in aggregate_rows:
        grouped.setdefault(str(row.get("baseline_family", "")), []).append(row)
    lines = [
        "## Scale Effects",
        "",
        "| family | train_docs | mean root_mae | mean exact-match |",
        "|---|---:|---:|---:|",
    ]
    for family, rows in sorted(grouped.items(), key=lambda item: item[0]):
        for row in sorted(rows, key=lambda item: (int(item.get("train_doc_count", 0)), str(item.get("cell_id", "")))):
            if str(payload.get("hardness_grid", "")):
                cell_label = str(row.get("cell_id", "")).strip()
                prefix = f"{family} / {cell_label}"
            else:
                prefix = str(family)
            lines.append(
                "| "
                f"{prefix} | "
                f"{int(row.get('train_doc_count', 0))} | "
                f"{float(row.get('test_root_mae_mean', float('nan'))):.6g} | "
                f"{float(row.get('test_exact_match_rate_mean', float('nan'))):.6g} |"
            )
    lines.append("")
    return lines


def _support_markdown_lines(payload: Mapping[str, Any]) -> List[str]:
    representatives = _representative_runs_by_unit(payload)
    if not representatives:
        return []
    lines = ["### Realized Supports", ""]
    if str(payload.get("hardness_grid", "")):
        lines.extend(
            [
                "| cell | regimes | segment band | test root counts | test distinct regimes |",
                "|---|---:|---|---|---|",
            ]
        )
        ordered_items = sorted(
            representatives.items(),
            key=lambda item: (
                int(item[1].get("n_regimes", 0)),
                int(item[1].get("segment_min", 0)),
                str(item[0]),
            ),
        )
        for cell_id, run in ordered_items:
            target_support = dict(run.get("target_support") or {})
            distinct_support = dict(run.get("distinct_regime_support") or {})
            test_target = dict(target_support.get("test") or {})
            test_distinct = dict(distinct_support.get("test") or {})
            lines.append(
                "| "
                f"{cell_id} | "
                f"{int(run.get('n_regimes', 0))} | "
                f"{str(run.get('segment_density_band', ''))} | "
                f"{_format_histogram(dict(test_target.get('histogram') or {}))} | "
                f"{_format_histogram(dict(test_distinct.get('histogram') or {}))} |"
            )
    else:
        sample_run = next(iter(representatives.values()))
        target_support = dict(sample_run.get("target_support") or {})
        distinct_support = dict(sample_run.get("distinct_regime_support") or {})
        lines.extend(
            [
                "| split | root counts | distinct regimes |",
                "|---|---|---|",
            ]
        )
        for split in ("train", "val", "test"):
            split_target = dict(target_support.get(split) or {})
            split_distinct = dict(distinct_support.get(split) or {})
            lines.append(
                "| "
                f"{split} | "
                f"{_format_histogram(dict(split_target.get('histogram') or {}))} | "
                f"{_format_histogram(dict(split_distinct.get('histogram') or {}))} |"
            )
    lines.append("")
    return lines


def render_full_doc_anchor_diagnostic_markdown(payload: Mapping[str, Any]) -> str:
    mode = _infer_report_mode_from_payload(payload)
    hardness_grid = str(payload.get("hardness_grid", ""))
    title = _report_title_from_payload(payload, mode=mode)
    lines = [
        title,
        "",
        f"- benchmark: `{str(payload.get('benchmark', ''))}`"
        if not hardness_grid
        else f"- hardness grid: `{hardness_grid}`",
        f"- description: {str(payload.get('benchmark_description', ''))}"
        if not hardness_grid
        else f"- grid cells: `{len(list(payload.get('grid_cells') or []))}`",
        f"- report mode: `{mode}`",
        f"- degenerate benchmark: `{bool(payload.get('degenerate_benchmark', False))}`",
        "",
    ]
    study_name = str(payload.get("study_name", "")).strip()
    if study_name:
        lines.insert(-1, f"- study name: `{study_name}`")
        lines.insert(
            -1,
            f"- study axis: `{str(payload.get('study_axis', '')).strip()}`"
            f" values={list(payload.get('axis_values') or [])}",
        )
        if str(payload.get("locked_tree_neural_config_label", "")).strip():
            lines.insert(
                -1,
                f"- locked tree_neural config: `{str(payload.get('locked_tree_neural_config_label', '')).strip()}`",
            )
    lines.extend(_what_this_report_is_trying_to_show_lines(payload, mode=mode))
    lines.extend(_dgp_intuition_markdown_lines(payload))
    lines.extend(_experimental_contract_markdown_lines(payload, mode=mode))
    lines.extend(_checks_markdown_lines(payload, mode=mode))
    lines.extend(_key_findings_markdown_lines(payload))
    lines.extend(_figure_first_interpretation_markdown_lines(payload, mode=mode))
    lines.extend(_what_reader_should_conclude_lines(payload, mode=mode))
    lines.extend(_compact_key_tables_markdown_lines(payload, mode=mode))
    lines.extend(_full_appendix_markdown_lines(payload))
    return "\n".join(lines) + "\n"


def _payload_from_saved_runs(
    *,
    runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    normalized_runs = [_normalize_loaded_run_semantics(run) for run in runs]
    aggregate = _aggregate_runs(normalized_runs)
    aggregate_rows = _attach_markov_witness_gap_fields(list(aggregate.get("aggregate_rows") or []))
    heatmap_rows = _build_heatmap_rows(aggregate_rows)
    grid_names = sorted(
        {
            str(run.get("hardness_grid", "")).strip()
            for run in normalized_runs
            if str(run.get("hardness_grid", "")).strip()
        }
    )
    normalized_grid = grid_names[0] if len(grid_names) == 1 else ""
    grid_cells: List[Dict[str, Any]] = []
    if normalized_grid:
        cell_ids = {
            str(run.get("cell_id", "")).strip()
            for run in normalized_runs
            if str(run.get("cell_id", "")).strip()
        }
        benchmarks = resolve_full_doc_diagnostic_grid(normalized_grid)
        grid_cells = [
            asdict(benchmark)
            for benchmark in benchmarks
            if str(benchmark.cell_id) in cell_ids
        ]
    benchmark_names = sorted(
        {
            str(run.get("benchmark", "")).strip()
            for run in normalized_runs
            if str(run.get("benchmark", "")).strip()
        }
    )
    benchmark = benchmark_names[0] if len(benchmark_names) == 1 else ""
    benchmark_description = ""
    degenerate_benchmark = False
    benchmark_spec: Dict[str, Any] = {}
    if benchmark and not normalized_grid:
        benchmark_payload = resolve_full_doc_diagnostic_benchmark(benchmark)
        benchmark_description = str(benchmark_payload.description)
        degenerate_benchmark = bool(benchmark_payload.degenerate)
        benchmark_spec = asdict(benchmark_payload)
    seeds = sorted(
        {int(run.get("seed", 0)) for run in normalized_runs if "seed" in run}
    )
    baseline_families = sorted(
        {
            str(run.get("baseline_family", "")).strip()
            for run in normalized_runs
            if str(run.get("baseline_family", "")).strip()
        }
    )
    config_labels = sorted(
        {
            str(run.get("config_label", "")).strip()
            for run in normalized_runs
            if str(run.get("config_label", "")).strip()
        }
    )
    tuning_stages = sorted(
        {
            str(run.get("tuning_stage", "")).strip()
            for run in normalized_runs
            if str(run.get("tuning_stage", "")).strip()
        }
    )
    study_names = sorted(
        {
            str(run.get("study_name", "")).strip()
            for run in normalized_runs
            if str(run.get("study_name", "")).strip()
        }
    )
    study_axes = sorted(
        {
            str(run.get("study_axis", "")).strip()
            for run in normalized_runs
            if str(run.get("study_axis", "")).strip()
        }
    )
    axis_values_raw = [
        _coerce_axis_value(run.get("axis_value", ""))
        for run in normalized_runs
        if _coerce_axis_value(run.get("axis_value", "")) != ""
    ]
    axis_values = sorted(
        axis_values_raw,
        key=lambda value: (
            0 if isinstance(value, (int, float)) else 1,
            float(value) if isinstance(value, (int, float)) else str(value),
        ),
    )
    locked_tree_neural_config_labels = sorted(
        {
            str(run.get("locked_tree_neural_config_label", "")).strip()
            for run in normalized_runs
            if str(run.get("locked_tree_neural_config_label", "")).strip()
        }
    )
    selection_metrics = sorted(
        {
            str(run.get("selection_metric", "")).strip()
            for run in normalized_runs
            if str(run.get("selection_metric", "")).strip()
        }
    )
    bundle_manifest = {
        str(run.get("cell_id") or run.get("benchmark") or ""): str(
            run.get("bundle_source", "")
        )
        for run in normalized_runs
        if str(run.get("bundle_source", "")).strip()
    }
    train_doc_counts_by_key: Dict[str, List[int]] = {}
    for run in normalized_runs:
        key = str(run.get("cell_id") or run.get("benchmark") or "").strip()
        train_doc_counts_by_key.setdefault(key, [])
        train_doc_counts_by_key[key].append(int(run.get("train_doc_count", 0)))
    normalized_train_counts_payload: Dict[str, List[int]] = {
        key: sorted({int(value) for value in values})
        for key, values in train_doc_counts_by_key.items()
    }
    payload = {
        "simulation": "markov_full_doc_anchor_diagnostics",
        **_report_metric_contract_payload(),
        "benchmark": str(benchmark),
        "benchmark_description": str(benchmark_description),
        "degenerate_benchmark": bool(degenerate_benchmark),
        "benchmark_spec": benchmark_spec,
        "hardness_grid": str(normalized_grid),
        "grid_cells": grid_cells,
        "seeds": [int(seed) for seed in seeds],
        "train_doc_counts": (
            [int(value) for value in next(iter(normalized_train_counts_payload.values()), [])]
            if len(normalized_train_counts_payload) == 1
            else normalized_train_counts_payload
        ),
        "baseline_families": [str(family) for family in baseline_families],
        "config_labels": [str(label) for label in config_labels],
        "tuning_stages": [str(stage) for stage in tuning_stages],
        "study_name": study_names[0] if len(study_names) == 1 else "",
        "study_axis": study_axes[0] if len(study_axes) == 1 else "",
        "axis_values": axis_values,
        "study_names": [str(name) for name in study_names],
        "study_axes": [str(axis) for axis in study_axes],
        "locked_tree_neural_config_label": (
            locked_tree_neural_config_labels[0]
            if len(locked_tree_neural_config_labels) == 1
            else ""
        ),
        "locked_tree_neural_config_labels": [
            str(label) for label in locked_tree_neural_config_labels
        ],
        "selection_metric": selection_metrics[0] if len(selection_metrics) == 1 else "",
        "selection_metrics": [str(metric) for metric in selection_metrics],
        "emit_confusion": any("confusion" in run for run in normalized_runs),
        "bundle_manifest": bundle_manifest,
        "runs": list(normalized_runs),
        "aggregate_rows": aggregate_rows,
        "diagnostic_readout": dict(aggregate.get("diagnostic_readout") or {}),
        "heatmap_rows": heatmap_rows,
        "grid_diagnostic_summary": (
            _grid_diagnostic_summary(heatmap_rows) if normalized_grid else {}
        ),
        "selection_metric_curve_summary": _selection_metric_curve_summary(aggregate_rows),
        "backend_device_summary": _backend_device_summary(aggregate_rows),
        "witness_gap_table": [
            {
                "baseline_family": str(row.get("baseline_family", "")),
                "train_doc_count": int(row.get("train_doc_count", 0)),
                "test_root_mae_mean": float(row.get("test_root_mae_mean", float("nan"))),
                "gap_to_ridge_control": float(row.get("gap_to_ridge_control", float("nan"))),
                "gap_to_exact_witness": float(row.get("gap_to_exact_witness", float("nan"))),
                "train_val_gap": float(row.get("train_val_gap", float("nan"))),
                "val_test_gap": float(row.get("val_test_gap", float("nan"))),
                "objective_variant": str(row.get("objective_variant", "")),
                "cause_code": str(row.get("cause_code", "")),
            }
            for row in aggregate_rows
        ],
    }
    payload["tree_neural_validation_summary"] = _tree_neural_validation_summary(payload)
    payload["tree_fno_fair_parity_summary"] = _tree_fno_fair_parity_summary(payload)
    payload["tree_fno_upper_bound_summary"] = _tree_fno_upper_bound_summary(payload)
    payload["learning_efficiency_summary"] = _learning_efficiency_summary(payload)
    payload["tree_oracle_budget_frontier_summary"] = _tree_oracle_budget_frontier_summary(payload)
    return payload


def load_markov_full_doc_anchor_diagnostics_from_output_dir(
    output_dir: Path,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    run_paths = sorted(output_path.glob("runs/*.json"))
    if not run_paths:
        run_paths = sorted(output_path.glob("**/runs/*.json"))
    if not run_paths:
        raise FileNotFoundError(
            f"no saved diagnostic run JSONs found under {output_path}"
        )
    runs = [
        dict(json.loads(path.read_text(encoding="utf-8")))
        for path in run_paths
    ]
    payload = _payload_from_saved_runs(runs=runs)
    runs_csv = output_path / "runs.csv"
    write_full_doc_anchor_diagnostic_csv(runs_csv, _flatten_run_rows(runs))
    aggregate_csv = output_path / "aggregate.csv"
    write_full_doc_anchor_diagnostic_csv(
        aggregate_csv,
        list(payload.get("aggregate_rows") or []),
    )
    heatmap_csv = output_path / "heatmap.csv"
    write_full_doc_anchor_diagnostic_csv(
        heatmap_csv,
        list(payload.get("heatmap_rows") or []),
    )
    validation_summary = dict(payload.get("tree_neural_validation_summary") or {})
    if validation_summary:
        validation_json = output_path / "tree_neural_validation_summary.json"
        validation_json.write_text(
            json.dumps(validation_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        validation_md = output_path / "tree_neural_validation_summary.md"
        validation_md.write_text(
            _render_tree_neural_validation_markdown(validation_summary),
            encoding="utf-8",
        )
        payload["tree_neural_validation_summary_json"] = str(validation_json)
        payload["tree_neural_validation_summary_md"] = str(validation_md)
    parity_summary = dict(payload.get("tree_fno_fair_parity_summary") or {})
    if parity_summary:
        parity_json = output_path / "tree_fno_fair_parity_summary.json"
        parity_json.write_text(
            json.dumps(parity_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        parity_md = output_path / "tree_fno_fair_parity_summary.md"
        parity_md.write_text(
            _render_tree_fno_fair_parity_markdown(parity_summary),
            encoding="utf-8",
        )
        payload["tree_fno_fair_parity_summary_json"] = str(parity_json)
        payload["tree_fno_fair_parity_summary_md"] = str(parity_md)
    upper_bound_summary = dict(payload.get("tree_fno_upper_bound_summary") or {})
    if upper_bound_summary:
        upper_json = output_path / "tree_fno_upper_bound_summary.json"
        upper_json.write_text(
            json.dumps(upper_bound_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        upper_md = output_path / "tree_fno_upper_bound_summary.md"
        upper_md.write_text(
            _render_tree_fno_upper_bound_markdown(upper_bound_summary),
            encoding="utf-8",
        )
        payload["tree_fno_upper_bound_summary_json"] = str(upper_json)
        payload["tree_fno_upper_bound_summary_md"] = str(upper_md)
    budget_frontier_summary = dict(payload.get("tree_oracle_budget_frontier_summary") or {})
    if budget_frontier_summary:
        frontier_json = output_path / "tree_oracle_budget_frontier_summary.json"
        frontier_json.write_text(
            json.dumps(budget_frontier_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        frontier_md = output_path / "tree_oracle_budget_frontier_summary.md"
        frontier_md.write_text(
            _render_tree_oracle_budget_frontier_markdown(budget_frontier_summary),
            encoding="utf-8",
        )
        payload["tree_oracle_budget_frontier_summary_json"] = str(frontier_json)
        payload["tree_oracle_budget_frontier_summary_md"] = str(frontier_md)
    summary_json = output_path / "summary.json"
    summary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload["summary_json"] = str(summary_json)
    payload["runs_csv"] = str(runs_csv)
    payload["aggregate_csv"] = str(aggregate_csv)
    payload["heatmap_csv"] = str(heatmap_csv)
    return payload


def run_markov_full_doc_anchor_diagnostics(
    *,
    benchmark_name: str = "recoverable_v4",
    hardness_grid: str = "",
    grid_cell_ids: Sequence[str] = tuple(),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    train_doc_counts: Sequence[int] = tuple(),
    baseline_families: Sequence[str] | None = None,
    emit_confusion: bool = False,
    output_dir: Path | None = None,
    use_cuda: bool = False,
    cuda_device: int | None = None,
    torch_threads: int = 1,
    config_overrides: Mapping[str, Any] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    base_bundle_path: str = "",
    memory_probe: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    normalized_grid = str(hardness_grid or "").strip().lower()
    benchmarks: tuple[FullDocDiagnosticBenchmarkSpec, ...]
    if normalized_grid:
        benchmarks = resolve_full_doc_diagnostic_grid(normalized_grid)
        if grid_cell_ids:
            selected_ids = _normalized_selected_grid_cell_ids(
                normalized_grid,
                grid_cell_ids,
            )
            benchmarks = tuple(
                benchmark
                for benchmark in benchmarks
                if str(benchmark.cell_id) in selected_ids
            )
            if not benchmarks:
                raise ValueError(
                    f"grid_cell_ids={tuple(grid_cell_ids)!r} did not match any cells in {normalized_grid}"
                )
    else:
        benchmarks = (resolve_full_doc_diagnostic_benchmark(str(benchmark_name)),)
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds:
        raise ValueError("seeds must be non-empty")
    requested_families = (
        tuple(baseline_families)
        if baseline_families is not None and tuple(baseline_families)
        else default_baseline_families_for_mode(hardness_grid=normalized_grid)
    )
    normalized_families = tuple(
        dict.fromkeys(_normalize_baseline_family(str(family)) for family in requested_families)
    )
    if not normalized_families:
        raise ValueError("baseline_families must be non-empty")
    for family in normalized_families:
        if family not in VALID_BASELINE_FAMILIES:
            raise ValueError(
                f"unsupported baseline family {family!r}; expected one of {VALID_BASELINE_FAMILIES}"
            )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "runs").mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    bundle_manifest: Dict[str, str] = {}
    normalized_train_counts_payload: Dict[str, List[int]] = {}
    planned_runs_total = 0
    per_benchmark_train_counts: Dict[str, tuple[int, ...]] = {}
    prepared_tree_cache: Dict[Tuple[str, int, int, str, bool, Tuple[int, ...]], _PreparedMarkovTreeData] = {}
    for benchmark in benchmarks:
        benchmark_train_counts = tuple(
            int(value)
            for value in (
                train_doc_counts
                if train_doc_counts
                else default_train_doc_counts_for_benchmark(benchmark)
            )
        )
        if not benchmark_train_counts:
            raise ValueError("train_doc_counts must be non-empty")
        per_benchmark_train_counts[str(benchmark.cell_id or benchmark.name)] = benchmark_train_counts
        planned_runs_total += (
            len(benchmark_train_counts) * len(normalized_seeds) * len(normalized_families)
        )
        normalized_train_counts_payload[str(benchmark.cell_id or benchmark.name)] = [
            int(value) for value in benchmark_train_counts
        ]
    completed_runs = 0
    for benchmark in benchmarks:
        benchmark_train_counts = per_benchmark_train_counts[str(benchmark.cell_id or benchmark.name)]
        required_train_docs = int(max(benchmark_train_counts))
        if progress_callback is not None:
            progress_callback(
                {
                    "run_index": int(completed_runs) + 1,
                    "runs_total": int(planned_runs_total),
                    "baseline_family": "",
                    "benchmark": str(benchmark.name),
                    "cell_id": str(benchmark.cell_id or ""),
                    "hardness_grid": str(benchmark.grid_name or ""),
                    "train_doc_count": int(required_train_docs),
                    "seed": int(normalized_seeds[0]),
                    "state": "running",
                    "stage": "materializing_bundle",
                    "epoch_completed": 0,
                    "epochs_total": 0,
                }
            )
        base_bundle, base_source = _materialize_base_bundle(
            benchmark=benchmark,
            required_train_docs=int(required_train_docs),
            output_dir=output_dir,
            base_bundle_path=str(base_bundle_path or ""),
        )
        bundle_manifest[str(benchmark.cell_id or benchmark.name)] = str(base_source)
        for train_doc_count in benchmark_train_counts:
            bundle, bundle_source = _bundle_with_fixed_eval_splits(
                base_bundle=base_bundle,
                base_source=base_source,
                train_doc_count=int(train_doc_count),
            )
            for seed in normalized_seeds:
                config = _base_config_for_benchmark(
                    benchmark=benchmark,
                    train_docs=int(train_doc_count),
                    use_cuda=bool(use_cuda),
                    cuda_device=cuda_device,
                    torch_threads=int(torch_threads),
                    seed=int(seed),
                    config_overrides=config_overrides,
                    baseline_families=normalized_families,
                )
                runtime_seeds, device = _resolve_device(config)
                for family in normalized_families:
                    current_run_index = int(completed_runs) + 1
                    effective_family_config = _effective_config_for_family(
                        benchmark=benchmark,
                        baseline_family=family,
                        config=config,
                    )
                    effective_train_config = _effective_train_config_for_full_doc_run(
                        benchmark=benchmark,
                        baseline_family=family,
                        train_doc_count=int(train_doc_count),
                        config=effective_family_config,
                    )
                    prepared_tree_data: _PreparedMarkovTreeData | None = None
                    if family in TREE_NEURAL_BASELINE_FAMILIES:
                        prepared_cache_key = (
                            str(benchmark.cell_id or benchmark.name),
                            int(effective_train_config.fixed_leaf_tokens),
                            int(getattr(effective_train_config, "max_internal_depth", 0)),
                            str(getattr(effective_train_config, "prepared_data_root", "")),
                            bool(
                                getattr(
                                    effective_train_config,
                                    "prepared_data_allow_create",
                                    True,
                                )
                            ),
                            tuple(int(value) for value in benchmark_train_counts),
                        )
                        prepared_tree_data = prepared_tree_cache.get(prepared_cache_key)
                        if prepared_tree_data is None:
                            prepared_tree_data = _ensure_prepared_markov_tree_data(
                                benchmark=benchmark,
                                base_bundle=base_bundle,
                                required_train_docs=int(required_train_docs),
                                train_prefix_counts=benchmark_train_counts,
                                fixed_leaf_tokens=int(effective_train_config.fixed_leaf_tokens),
                                max_internal_depth=int(
                                    getattr(effective_train_config, "max_internal_depth", 0)
                                ),
                                seeds=normalized_seeds,
                                prepared_data_root=str(
                                    getattr(effective_train_config, "prepared_data_root", "")
                                ),
                                allow_create=bool(
                                    getattr(
                                        effective_train_config,
                                        "prepared_data_allow_create",
                                        True,
                                    )
                                ),
                            )
                            prepared_tree_cache[prepared_cache_key] = prepared_tree_data
                    expected_epochs = _expected_epochs_for_config(
                        effective_train_config
                    )

                    def _emit_progress(update: Mapping[str, Any] | None = None) -> None:
                        if progress_callback is None:
                            return
                        payload = dict(update or {})
                        payload.update(
                            {
                                "run_index": int(current_run_index),
                                "runs_total": int(planned_runs_total),
                                "baseline_family": str(family),
                                "benchmark": str(benchmark.name),
                                "cell_id": str(benchmark.cell_id or ""),
                                "hardness_grid": str(benchmark.grid_name or ""),
                                "train_doc_count": int(train_doc_count),
                                "seed": int(seed),
                            }
                        )
                        progress_callback(payload)

                    _emit_progress(
                        {
                            "state": "running",
                            "stage": "run_start",
                            "epoch_completed": 0,
                            "epochs_total": int(expected_epochs),
                        }
                    )
                    run = _run_payload(
                        benchmark=benchmark,
                        baseline_family=family,
                        train_doc_count=int(train_doc_count),
                        config=effective_train_config,
                        seeds=runtime_seeds,
                        device=device,
                        bundle=bundle,
                        bundle_source=bundle_source,
                        emit_confusion=bool(emit_confusion),
                        prepared_tree_data=prepared_tree_data,
                        output_dir=output_dir,
                        progress_callback=_emit_progress,
                        memory_probe=memory_probe,
                        run_metadata=run_metadata,
                    )
                    _emit_progress(
                        {
                            "state": "running",
                            "stage": "run_complete",
                            "epoch_completed": int(expected_epochs),
                            "epochs_total": int(expected_epochs),
                        }
                    )
                    completed_runs += 1
                    if run_metadata:
                        run.update(dict(run_metadata))
                    runs.append(run)
                    if output_dir is not None:
                        stem_parts = []
                        if str(benchmark.cell_id or "").strip():
                            stem_parts.append(str(benchmark.cell_id))
                        stem_parts.extend(
                            [str(family), f"train_{int(train_doc_count)}", f"seed_{int(seed)}"]
                        )
                        run_path = output_dir / "runs" / f"{'__'.join(stem_parts)}.json"
                        run_path.write_text(
                            json.dumps(run, indent=2, sort_keys=True),
                            encoding="utf-8",
                        )

    payload = _payload_from_saved_runs(runs=runs)
    benchmark_payload = benchmarks[0] if len(benchmarks) == 1 else None
    payload["benchmark_description"] = (
        str(benchmark_payload.description) if benchmark_payload is not None else ""
    )
    payload["degenerate_benchmark"] = (
        bool(benchmark_payload.degenerate) if benchmark_payload is not None else False
    )
    payload["benchmark_spec"] = (
        asdict(benchmark_payload) if benchmark_payload is not None else {}
    )
    payload["hardness_grid"] = str(normalized_grid)
    payload["grid_cells"] = [asdict(benchmark) for benchmark in benchmarks] if normalized_grid else []
    payload["seeds"] = [int(seed) for seed in normalized_seeds]
    payload["train_doc_counts"] = (
        [int(value) for value in next(iter(normalized_train_counts_payload.values()), [])]
        if len(normalized_train_counts_payload) == 1
        else normalized_train_counts_payload
    )
    payload["baseline_families"] = [str(family) for family in normalized_families]
    payload["emit_confusion"] = bool(emit_confusion)
    payload["bundle_manifest"] = bundle_manifest
    if output_dir is not None:
        runs_csv = output_dir / "runs.csv"
        write_full_doc_anchor_diagnostic_csv(
            runs_csv,
            _flatten_run_rows(runs),
        )
        aggregate_csv = output_dir / "aggregate.csv"
        write_full_doc_anchor_diagnostic_csv(
            aggregate_csv,
            list(payload.get("aggregate_rows") or []),
        )
        heatmap_csv = output_dir / "heatmap.csv"
        write_full_doc_anchor_diagnostic_csv(
            heatmap_csv,
            list(payload.get("heatmap_rows") or []),
        )
        validation_summary = dict(payload.get("tree_neural_validation_summary") or {})
        if validation_summary:
            validation_json = output_dir / "tree_neural_validation_summary.json"
            validation_json.write_text(
                json.dumps(validation_summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            validation_md = output_dir / "tree_neural_validation_summary.md"
            validation_md.write_text(
                _render_tree_neural_validation_markdown(validation_summary),
                encoding="utf-8",
            )
            payload["tree_neural_validation_summary_json"] = str(validation_json)
            payload["tree_neural_validation_summary_md"] = str(validation_md)
        parity_summary = dict(payload.get("tree_fno_fair_parity_summary") or {})
        if parity_summary:
            parity_json = output_dir / "tree_fno_fair_parity_summary.json"
            parity_json.write_text(
                json.dumps(parity_summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            parity_md = output_dir / "tree_fno_fair_parity_summary.md"
            parity_md.write_text(
                _render_tree_fno_fair_parity_markdown(parity_summary),
                encoding="utf-8",
            )
            payload["tree_fno_fair_parity_summary_json"] = str(parity_json)
            payload["tree_fno_fair_parity_summary_md"] = str(parity_md)
        upper_bound_summary = dict(payload.get("tree_fno_upper_bound_summary") or {})
        if upper_bound_summary:
            upper_json = output_dir / "tree_fno_upper_bound_summary.json"
            upper_json.write_text(
                json.dumps(upper_bound_summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            upper_md = output_dir / "tree_fno_upper_bound_summary.md"
            upper_md.write_text(
                _render_tree_fno_upper_bound_markdown(upper_bound_summary),
                encoding="utf-8",
            )
            payload["tree_fno_upper_bound_summary_json"] = str(upper_json)
            payload["tree_fno_upper_bound_summary_md"] = str(upper_md)
        budget_frontier_summary = dict(payload.get("tree_oracle_budget_frontier_summary") or {})
        if budget_frontier_summary:
            frontier_json = output_dir / "tree_oracle_budget_frontier_summary.json"
            frontier_json.write_text(
                json.dumps(budget_frontier_summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            frontier_md = output_dir / "tree_oracle_budget_frontier_summary.md"
            frontier_md.write_text(
                _render_tree_oracle_budget_frontier_markdown(budget_frontier_summary),
                encoding="utf-8",
            )
            payload["tree_oracle_budget_frontier_summary_json"] = str(frontier_json)
            payload["tree_oracle_budget_frontier_summary_md"] = str(frontier_md)
        summary_json = output_dir / "summary.json"
        summary_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        payload["summary_json"] = str(summary_json)
        payload["runs_csv"] = str(runs_csv)
        payload["aggregate_csv"] = str(aggregate_csv)
        payload["heatmap_csv"] = str(heatmap_csv)
    return payload


def prepare_markov_full_doc_anchor_diagnostics_data(
    *,
    benchmark_name: str = "recoverable_v4",
    hardness_grid: str = "",
    grid_cell_ids: Sequence[str] = tuple(),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    train_doc_counts: Sequence[int] = tuple(),
    use_cuda: bool = False,
    cuda_device: int | None = None,
    torch_threads: int = 1,
    config_overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    normalized_grid = str(hardness_grid or "").strip().lower()
    if normalized_grid:
        benchmarks = resolve_full_doc_diagnostic_grid(normalized_grid)
        if grid_cell_ids:
            selected_ids = _normalized_selected_grid_cell_ids(
                normalized_grid,
                grid_cell_ids,
            )
            benchmarks = tuple(
                benchmark
                for benchmark in benchmarks
                if str(benchmark.cell_id) in selected_ids
            )
            if not benchmarks:
                raise ValueError(
                    f"grid_cell_ids={tuple(grid_cell_ids)!r} did not match any cells in {normalized_grid}"
                )
    else:
        benchmarks = (resolve_full_doc_diagnostic_benchmark(str(benchmark_name)),)
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds:
        raise ValueError("seeds must be non-empty")
    prepared_payloads: List[Dict[str, Any]] = []
    for benchmark in benchmarks:
        benchmark_train_counts = tuple(
            int(value)
            for value in (
                train_doc_counts
                if train_doc_counts
                else default_train_doc_counts_for_benchmark(benchmark)
            )
        )
        if not benchmark_train_counts:
            raise ValueError("train_doc_counts must be non-empty")
        required_train_docs = int(max(benchmark_train_counts))
        config = _base_config_for_benchmark(
            benchmark=benchmark,
            train_docs=int(required_train_docs),
            use_cuda=bool(use_cuda),
            cuda_device=cuda_device,
            torch_threads=int(torch_threads),
            seed=int(normalized_seeds[0]),
            config_overrides=config_overrides,
        )
        prepared = _ensure_prepared_markov_tree_data(
            benchmark=benchmark,
            base_bundle=_materialize_base_bundle(
                benchmark=benchmark,
                required_train_docs=int(required_train_docs),
                output_dir=None,
            )[0],
            required_train_docs=int(required_train_docs),
            train_prefix_counts=benchmark_train_counts,
            fixed_leaf_tokens=int(config.fixed_leaf_tokens),
            max_internal_depth=int(getattr(config, "max_internal_depth", 0)),
            seeds=normalized_seeds,
            prepared_data_root=str(getattr(config, "prepared_data_root", "")),
            allow_create=bool(getattr(config, "prepared_data_allow_create", True)),
        )
        prepared_payloads.append(
            {
                "benchmark": str(benchmark.name),
                "cell_id": str(benchmark.cell_id or ""),
                "prepared_data_root": str(prepared.root),
                "prepared_data_signature": str(prepared.signature),
                "required_train_docs": int(prepared.required_train_docs),
                "fixed_leaf_tokens": int(prepared.fixed_leaf_tokens),
                "max_internal_depth": int(prepared.max_internal_depth),
                "train_prefix_counts": [int(value) for value in prepared.train_prefix_counts],
                "train_prefix_signatures": {
                    str(int(key)): str(value)
                    for key, value in dict(prepared.train_prefix_signatures).items()
                },
                "train_fno_docs_json": str(prepared.root / "train_fno_docs.json"),
                "val_fno_docs_json": str(prepared.root / "val_fno_docs.json"),
                "test_fno_docs_json": str(prepared.root / "test_fno_docs.json"),
                "leaf_orderings_json": str(prepared.root / "leaf_orderings.json"),
                "internal_orderings_json": str(prepared.root / "internal_orderings.json"),
                "metadata_json": str(prepared.root / "metadata.json"),
            }
        )
    return {
        "simulation": "markov_full_doc_anchor_diagnostics_prepare_data",
        "hardness_grid": str(normalized_grid),
        "prepared": prepared_payloads,
    }


__all__ = [
    "CANONICAL_DIAGNOSTIC_BUNDLES",
    "DEFAULT_STICKY_STRUCTURAL_V2_CELL_ID",
    "DEFAULT_DIAGNOSTIC_BASELINE_FAMILIES",
    "DEFAULT_STRUCTURAL_CORE_BASELINE_FAMILIES",
    "FullDocDiagnosticBenchmarkSpec",
    "STICKY_STRUCTURAL_V2_CELL_SPECS",
    "STICKY_STRUCTURAL_V2_LEGACY_ALIAS_MAP",
    "VALID_BASELINE_FAMILIES",
    "VALID_HARDNESS_GRIDS",
    "default_baseline_families_for_mode",
    "default_train_doc_counts_for_benchmark",
    "load_markov_full_doc_anchor_diagnostics_from_output_dir",
    "prepare_markov_full_doc_anchor_diagnostics_data",
    "render_full_doc_anchor_diagnostic_markdown",
    "resolve_full_doc_diagnostic_benchmark",
    "resolve_full_doc_diagnostic_grid",
    "run_markov_full_doc_anchor_diagnostics",
    "write_full_doc_anchor_diagnostic_csv",
]
