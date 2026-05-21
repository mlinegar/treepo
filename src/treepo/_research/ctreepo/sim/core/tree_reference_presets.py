"""Tree reference preset configurations.

Extracted from run_markov_optimization_tradeoff_pipeline.py so that
presets can be imported without pulling in the full 11K-line pipeline script.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


# ---------------------------------------------------------------------------
# Preset name constants
# ---------------------------------------------------------------------------

SUPERVISION_RECOVERY_COMMON_TREE_REFERENCE_PRESET = "common_factorized_sketch_v1"
UNIFIED_G_FULL_LOCAL_LAWS_PRESET = "unified_g_full_local_laws_v1"
COMPARISON_GRID_V3_PRESET = "comparison_grid_v3"
UNIFIED_G_HALF_LEAF_LAW_PRESET = "unified_g_half_leaf_law_v1"
UNIFIED_G_FNO_PARITY_CANARY_PRESET = "unified_g_fno_parity_canary_v1"
# Ablation steps from canary → standard (one change at a time):
UNIFIED_G_ABLATION_MSE_PRESET = "unified_g_ablation_mse_v1"             # canary + MSE
UNIFIED_G_ABLATION_TWO_STAGE_PRESET = "unified_g_ablation_two_stage_v1"  # canary + MSE + two_stage
UNIFIED_G_ABLATION_LOCAL_LAWS_PRESET = "unified_g_ablation_local_laws_v1"  # = standard
ROOT_ONLY_PARITY_HISTORICAL_REPLAY_PRESET = (
    "recoverable_root_only_parity_historical_replay_v1"
)
ROOT_ONLY_PARITY_OPTIMIZATION_FIX_PRESET = (
    "recoverable_root_only_parity_optimization_fix_v1"
)
ROOT_ONLY_PARITY_CAPACITY_FIX_PRESET = (
    "recoverable_root_only_parity_capacity_fix_v1"
)
ROOT_ONLY_PARITY_MATCHED_ROOT_V1_PRESET = (
    "recoverable_root_only_parity_matched_root_v1"
)
ROOT_ONLY_PARITY_MATCHED_ROOT_V2_PRESET = (
    "recoverable_root_only_parity_matched_root_v2"
)
ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET = (
    "recoverable_root_only_parity_matched_root_v3"
)
STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_V1_PRESET = (
    "structural_root_only_parity_matched_root_v1"
)
STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_V2_PRESET = (
    "structural_root_only_parity_matched_root_v2"
)
STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET = (
    "structural_root_only_parity_matched_root_v3"
)

# Stable public names can point at an underlying recipe id. This lets checked-in
# configs and reports use short human-readable names while preserving the exact
# recipe provenance separately.
TREE_REFERENCE_PRESET_ALIASES: dict[str, str] = {
    "common_tree": SUPERVISION_RECOVERY_COMMON_TREE_REFERENCE_PRESET,
    "standard_tree": UNIFIED_G_FULL_LOCAL_LAWS_PRESET,
    COMPARISON_GRID_V3_PRESET: UNIFIED_G_FULL_LOCAL_LAWS_PRESET,
    "half_c1": UNIFIED_G_HALF_LEAF_LAW_PRESET,
    "fno_parity_canary": UNIFIED_G_FNO_PARITY_CANARY_PRESET,
    "mse_only": UNIFIED_G_ABLATION_MSE_PRESET,
    "two_stage_no_laws": UNIFIED_G_ABLATION_TWO_STAGE_PRESET,
    "full_laws": UNIFIED_G_ABLATION_LOCAL_LAWS_PRESET,
    "multileaf_root_only": "unified_g_multi_leaf_root_only_v1",
    "root_only_replay": ROOT_ONLY_PARITY_HISTORICAL_REPLAY_PRESET,
    "root_only_opt_fix": ROOT_ONLY_PARITY_OPTIMIZATION_FIX_PRESET,
    "root_only_capacity_fix": ROOT_ONLY_PARITY_CAPACITY_FIX_PRESET,
    "root_only_matched": ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET,
    "root_only_matched_v1": ROOT_ONLY_PARITY_MATCHED_ROOT_V1_PRESET,
    "root_only_matched_v2": ROOT_ONLY_PARITY_MATCHED_ROOT_V2_PRESET,
    "structural_root_only_matched": STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET,
    "structural_root_only_matched_v1": STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_V1_PRESET,
    "structural_root_only_matched_v2": STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_V2_PRESET,
}


# ---------------------------------------------------------------------------
# Override keys: fields that tree reference presets can set on a config.
# ---------------------------------------------------------------------------

TREE_REFERENCE_OVERRIDE_KEYS: tuple[str, ...] = (
    "n_epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "state_dim",
    "hidden_dim",
    "fixed_leaf_tokens",
    "task_objective_weight",
    "local_law_weight",
    "c1_relative_weight",
    "c2_relative_weight",
    "c3_relative_weight",
    "tree_model_version",
    "tree_batch_runtime_mode",
    "tree_batch_pack_mode",
    "tree_batch_autotune",
    "tree_batch_structural_pad_limit",
    "tree_batch_auto_queue_min_docs",
    "tree_batch_auto_queue_min_fill_ratio",
    "tree_training_schedule",
    "tree_task_head_mode",
    "tree_theorem_surface_mode",
    "tree_summary_spec_root_mode",
    "tree_theorem_feature_dim",
    "tree_theorem_feature_hidden_dim",
    "tree_theorem_score_dim",
    "tree_theorem_fiber_dim",
    "tree_theorem_aux_dim",
    "tree_leaf_fno_width",
    "tree_leaf_fno_n_modes",
    "tree_leaf_fno_n_layers",
    "tree_root_supervision_kind",
    "tree_checkpoint_metric",
    "tree_stage1_checkpoint_metric",
    "tree_stage1_eval_mode",
    "tree_stage1_screen_doc_limit",
    "tree_stage1_final_exact_doc_limit",
    "tree_stage1_epochs",
    "tree_stage2_epochs",
    "tree_stage1_root_weight",
    "tree_join_bit_weight",
    "tree_theorem_count_head_mode",
    "tree_theorem_count_dim",
    "tree_theorem_first_dim",
    "tree_theorem_last_dim",
    "tree_theorem_count_ordinal_weight",
    "tree_theorem_count_scalar_aux_weight",
    "tree_theorem_count_threshold_balance",
    "tree_phi_compose_weight",
    "tree_phi_contrastive_weight",
    "tree_phi_alignment_loss",
    "aligned_sketch_surface",
    "summary_spec_name",
    "slot_count",
    "tree_eval_workers_per_mig",
    "leaf_supervision_kind",
    "leaf_label_rate",
    "leaf_exact_supervision",
    "internal_supervision_kind",
    "internal_label_rate",
    "root_weight",
    "depth_discount_gamma",
    "schedule_consistency_weight",
    "tree_aux_doc_sequence_fraction",
    "gpu_runtime_data_mode",
    "gpu_runtime_bucket_mode",
    "gpu_runtime_preload_splits",
    "gpu_runtime_preload_targets",
    "gpu_runtime_workers_per_mig",
    "gpu_runtime_allow_multi_worker_screen",
    "gpu_runtime_capacity_workers_per_mig",
)


# Keys that presets should NEVER override (runtime, data gen, environment).
# Used by the deny-list approach: any OPSCountConfig field NOT in this set
# can be overridden by a preset, without needing to add it to the allowlist.
TREE_REFERENCE_DENY_KEYS: frozenset[str] = frozenset({
    # Data generation / environment
    "n_regimes", "vocab_size", "generator_profile",
    "min_tokens", "max_tokens", "min_segments", "max_segments",
    "min_seg_len", "max_seg_len",
    "min_distinct_regimes_per_doc", "max_distinct_regimes_per_doc",
    "sinkhorn_iters", "transition_log_std",
    "train_docs", "val_docs", "test_docs",
    "data_seed", "model_seed", "seed",
    "val_seed_offset", "test_seed_offset",
    # Runtime / device
    "use_cuda", "cuda_device", "torch_threads",
    "artifact_dir", "raw_diagnostic_artifact_dir",
    "prepared_data_root", "prepared_data_allow_create", "prepared_data_signature",
    "tree_stage1_artifact_dir", "tree_stage1_artifact_root", "tree_stage1_resume_if_available",
    "save_logged_observations", "suite_role",
    # Comparison / eval control
    "comparison_mode", "preserve_requested_leaf_tokens",
    "official_fno_preserve_requested_leaf_tokens",
    # Baseline inclusion flags
    "include_rf_root_baseline", "include_doc_level_baseline",
    "include_doc_sequence_baseline", "include_doc_transformer_baseline",
    "include_fno_baseline", "include_deeponet_baseline",
    "include_mlp_bigram_baseline", "include_cnn1d_baseline",
    "include_doc_level_ridge_baseline",
    "include_leaf_ridge_tree_baseline", "include_leaf_knn_tree_baseline",
    "include_leaf_endpoint_table_tree_baseline",
    "include_leaf_dt_tree_baseline", "include_leaf_rf_tree_baseline",
    "include_sampled_leaf_pool_ridge_baseline", "include_sampled_leaf_pool_rf_baseline",
    # Budget (set by package spec, not preset)
    "budget_total_calls", "budget_total_calls_per_doc",
    "full_doc_budget_share", "doc_consumption_mode", "local_split_mode",
    # Legacy weights (superseded by local_law_weight parameterization)
    "c2_weight", "c3_weight", "leaf_weight", "law_package",
})


# ---------------------------------------------------------------------------
# Preset configs
# ---------------------------------------------------------------------------

_COMMON_FACTORIZED_SKETCH: dict[str, Any] = {
    "n_epochs": 52,
    "batch_size": 128,
    "lr": 5e-4,
    "weight_decay": 0.0,
    # Conservative defaults (reverted 2026-05-03 after the wide-heads
    # experiment was empirically refuted). The hypothesis was that head
    # capacity was the bottleneck for the ~2.14 zero-merge root_mae floor on
    # `recoverable_v5_t2048`. Tested by bumping to (state=2048, hid=2048,
    # merge=4096): the floor moved 2.14 -> 2.13 (noise) and SEVERAL
    # composition cells got worse (full100 @ leaf=256 went 1.06 -> 3.72,
    # converging to a bad local min by best_epoch=10). So head capacity is
    # NOT the bottleneck for that floor; the limit is somewhere else
    # (FNO encoder width, leaf-pooling info loss, or DGP irreducible noise).
    # See `outputs/markov_t2048_full_grid_wide_heads_20260503_003820/` for
    # the experiment that refuted it.
    "state_dim": 128,
    "hidden_dim": 512,
    "fixed_leaf_tokens": 16,
    "local_law_weight": 0.8,
    "c1_relative_weight": 1.0,
    "c2_relative_weight": 1.0,
    "c3_relative_weight": 1.0,
    "tree_model_version": "v2",
    "tree_batch_runtime_mode": "unified_v2",
    "tree_batch_pack_mode": "fixed_fused",
    "tree_training_schedule": "two_stage",
    "tree_task_head_mode": "theorem_feature_scalar",
    "tree_theorem_surface_mode": "factorized_score_fiber",
    "tree_summary_spec_root_mode": "factored_theorem_readout",
    "tree_theorem_feature_dim": 48,
    "tree_theorem_feature_hidden_dim": 256,
    "tree_theorem_score_dim": 1,
    "tree_theorem_fiber_dim": 47,
    "tree_theorem_aux_dim": 0,
    "tree_leaf_fno_width": 128,
    "tree_leaf_fno_n_modes": 8,
    "tree_leaf_fno_n_layers": 4,
    "tree_root_supervision_kind": "mse",
    "tree_checkpoint_metric": "val_root_mae",
    "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
    "tree_stage1_eval_mode": "per_epoch",
    "tree_stage1_epochs": 12,
    "tree_stage2_epochs": 40,
    "tree_stage1_root_weight": 0.0,
    "tree_join_bit_weight": 1.0,
    "tree_theorem_count_head_mode": "scalar_mse",
    "tree_theorem_count_dim": 8,
    "tree_theorem_first_dim": 8,
    "tree_theorem_last_dim": 8,
    "tree_phi_compose_weight": 0.0,
    "tree_phi_contrastive_weight": 0.0,
    "tree_phi_alignment_loss": "cosine_mse",
    "aligned_sketch_surface": "",
    "summary_spec_name": "markov_count_sketch",
    "slot_count": 4,
    "leaf_supervision_kind": "full_sketch",
    "leaf_label_rate": 1.0,
    "leaf_exact_supervision": False,
    "internal_supervision_kind": "full_sketch",
    "internal_label_rate": 1.0,
    "root_weight": 1.0,
    "depth_discount_gamma": 1.0,
    "schedule_consistency_weight": 0.0,
    "tree_aux_doc_sequence_fraction": 0.0,
}

_SLOTWISE_DENSE: dict[str, Any] = {
    **_COMMON_FACTORIZED_SKETCH,
    "tree_theorem_surface_mode": "slotwise",
    "tree_theorem_score_dim": 0,
    "tree_theorem_fiber_dim": 0,
    "tree_checkpoint_metric": "val_exact_sketch_direct",
}

_ROOT_ONLY_FROM_SLOTWISE = {
    "leaf_supervision_kind": "count_only",
    "leaf_label_rate": 0.0,
    "internal_supervision_kind": "none",
    "internal_label_rate": 0.0,
}


TREE_REFERENCE_PRESET_CONFIGS: dict[str, dict[str, Any]] = {
    SUPERVISION_RECOVERY_COMMON_TREE_REFERENCE_PRESET: dict(_COMMON_FACTORIZED_SKETCH),
    "recoverable_slotwise_dense_v1": dict(_SLOTWISE_DENSE),
    "structural_factorized_fiber_v1": {
        "n_epochs": 10,
        "batch_size": 256,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "state_dim": 32,
        "hidden_dim": 64,
        "fixed_leaf_tokens": 8,
        "local_law_weight": 0.5,
        "c1_relative_weight": 1.0,
        "c2_relative_weight": 1.0,
        "c3_relative_weight": 1.0,
        "tree_training_schedule": "single_stage",
        "tree_task_head_mode": "theorem_feature_scalar",
        "tree_theorem_surface_mode": "factorized_score_fiber",
        "tree_summary_spec_root_mode": "factored_theorem_readout",
        "tree_theorem_feature_dim": 16,
        "tree_theorem_feature_hidden_dim": 32,
        "tree_theorem_score_dim": 1,
        "tree_theorem_fiber_dim": 15,
        "tree_theorem_aux_dim": 0,
        "tree_root_supervision_kind": "mse",
        "tree_checkpoint_metric": "val_root_mae",
        "tree_stage1_checkpoint_metric": "val_root_mae",
        "tree_stage1_eval_mode": "per_epoch",
        "tree_stage1_epochs": 0,
        "tree_stage2_epochs": 0,
        "tree_stage1_root_weight": 1.0,
        "tree_join_bit_weight": 0.0,
        "tree_phi_compose_weight": 0.0,
        "tree_phi_contrastive_weight": 0.0,
        "tree_phi_alignment_loss": "cosine_mse",
        "aligned_sketch_surface": "",
        "leaf_supervision_kind": "count_only",
        "leaf_label_rate": 0.0,
        "internal_supervision_kind": "none",
        "internal_label_rate": 0.0,
        "root_weight": 1.0,
        "schedule_consistency_weight": 0.0,
        "tree_aux_doc_sequence_fraction": 0.0,
    },
    "structural_factorized_fiber_v2": {
        "n_epochs": 32,
        "batch_size": 256,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "state_dim": 128,
        "hidden_dim": 512,
        "fixed_leaf_tokens": 16,
        "local_law_weight": 0.5,
        "c1_relative_weight": 1.0,
        "c2_relative_weight": 1.0,
        "c3_relative_weight": 1.0,
        "tree_model_version": "v2",
        "tree_batch_runtime_mode": "unified_v2",
        "tree_batch_pack_mode": "fixed_fused",
        "tree_training_schedule": "two_stage",
        "tree_task_head_mode": "full_state_scalar",
        "tree_theorem_surface_mode": "factorized_score_fiber",
        "tree_summary_spec_root_mode": "task_split_ablation",
        "tree_theorem_feature_dim": 16,
        "tree_theorem_feature_hidden_dim": 32,
        "tree_theorem_score_dim": 1,
        "tree_theorem_fiber_dim": 15,
        "tree_theorem_aux_dim": 0,
        "tree_leaf_fno_width": 128,
        "tree_leaf_fno_n_modes": 4,
        "tree_leaf_fno_n_layers": 4,
        "tree_root_supervision_kind": "count_ce",
        "tree_checkpoint_metric": "val_root_mae",
        "tree_stage1_checkpoint_metric": "val_root_mae",
        "tree_stage1_eval_mode": "per_epoch",
        "tree_stage1_epochs": 12,
        "tree_stage2_epochs": 20,
        "tree_stage1_root_weight": 0.0,
        "tree_join_bit_weight": 0.0,
        "tree_theorem_count_head_mode": "scalar_mse",
        "tree_theorem_count_dim": 0,
        "tree_theorem_first_dim": 0,
        "tree_theorem_last_dim": 0,
        "tree_phi_compose_weight": 0.0,
        "tree_phi_contrastive_weight": 0.0,
        "tree_phi_alignment_loss": "cosine_mse",
        "aligned_sketch_surface": "",
        "leaf_supervision_kind": "full_sketch",
        "leaf_label_rate": 1.0,
        "leaf_exact_supervision": False,
        "internal_supervision_kind": "full_sketch",
        "internal_label_rate": 1.0,
        "root_weight": 1.0,
        "schedule_consistency_weight": 0.0,
        "tree_aux_doc_sequence_fraction": 0.0,
    },
    "structural_factorized_sketch_v3": {
        **_COMMON_FACTORIZED_SKETCH,
        "tree_checkpoint_metric": "val_exact_sketch_direct",
    },
}

# Derived presets (extend base presets with overrides)
TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_FULL_LOCAL_LAWS_PRESET] = {
    **TREE_REFERENCE_PRESET_CONFIGS[SUPERVISION_RECOVERY_COMMON_TREE_REFERENCE_PRESET],
    "n_epochs": 40,
    "tree_model_version": "unified_g",
    "tree_stage1_epochs": 10,
    "tree_stage2_epochs": 30,
}

# Small-model variant of unified_g_full_local_laws_v1: half state/hidden/fno
# dimensions to test whether per-node forward compute scales linearly with
# model size in the long-merge-chain regime (recoverable_v5_t2048 leaf=16).
UNIFIED_G_FULL_LOCAL_LAWS_SMALL_PRESET = "unified_g_full_local_laws_small_v1"
TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_FULL_LOCAL_LAWS_SMALL_PRESET] = {
    **TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_FULL_LOCAL_LAWS_PRESET],
    "state_dim": 64,
    "hidden_dim": 256,
    "tree_leaf_fno_width": 64,
    "tree_leaf_fno_n_modes": 4,
    "tree_leaf_fno_n_layers": 2,
}

# Single-stage variant of unified_g_full_local_laws_v1: skip stage1 entirely
# (stage1 was burning ~47 min/cell at leaf=16 in the t2048 sweep, vs ~14 min
# for stage2). Tests whether stage1 theorem-feature alignment is necessary
# for composition recovery or whether stage2's end-to-end training suffices.
UNIFIED_G_FULL_LOCAL_LAWS_SINGLE_STAGE_PRESET = "unified_g_full_local_laws_single_stage_v1"
TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_FULL_LOCAL_LAWS_SINGLE_STAGE_PRESET] = {
    **TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_FULL_LOCAL_LAWS_PRESET],
    "tree_training_schedule": "single_stage",
    "tree_stage1_epochs": 0,
    "tree_stage2_epochs": 0,
    "n_epochs": 40,
}

# Wide-head variant of unified_g_full_local_laws_v1: 2048 hidden dim for the
# count head (f), 4096 hidden dim for the merge head (g). Tested 2026-05-03
# and refuted - did not crack the ~2.14 root_mae floor at zero merges and
# regressed several composition cells. Kept for explicit ablations only.
# See `feedback_head_capacity_was_not_the_bottleneck.md`.
UNIFIED_G_FULL_LOCAL_LAWS_WIDE_HEADS_PRESET = "unified_g_full_local_laws_wide_heads_v1"
TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_FULL_LOCAL_LAWS_WIDE_HEADS_PRESET] = {
    **TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_FULL_LOCAL_LAWS_PRESET],
    "hidden_dim": 2048,
    "tree_merge_hidden_dim": 4096,
}

# FNO modes ablation presets (2026-05-03). The leaf encoder uses an FNO
# with `tree_leaf_fno_n_modes` Fourier modes. On `recoverable_v5_t2048`
# with mean segment length ~128 tokens, the boundary signal has period
# ~128 -> Nyquist mode index = 16. The default n_modes=8 is sub-Nyquist.
# These presets test whether bumping n_modes cracks the ~2.14 zero-merge
# root_mae floor on t2048.
for _modes in (16, 32, 64, 128, 256, 512, 1024, 2048):
    _name = f"unified_g_full_local_laws_modes{_modes}_v1"
    TREE_REFERENCE_PRESET_CONFIGS[_name] = {
        **TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_FULL_LOCAL_LAWS_PRESET],
        "tree_leaf_fno_n_modes": _modes,
    }
del _modes, _name

# Sum-pool variants of the unified-g and modes-ablation presets (2026-05-03).
# Theory: for boundary counting, sum-pool preserves count-additivity (each
# per-token boundary signal contributes additively), while mean-pool dilutes
# by L. The ~1.08 floor at modes=1024 (full Nyquist) suggests the bottleneck
# is downstream of the FNO; sum-pool tests the pooling step directly.
TREE_REFERENCE_PRESET_CONFIGS["unified_g_full_local_laws_sumpool_v1"] = {
    **TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_FULL_LOCAL_LAWS_PRESET],
    "tree_leaf_fno_pooling": "sum",
}
for _modes in (32, 128, 512, 1024):
    _name = f"unified_g_full_local_laws_modes{_modes}_sumpool_v1"
    TREE_REFERENCE_PRESET_CONFIGS[_name] = {
        **TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_FULL_LOCAL_LAWS_PRESET],
        "tree_leaf_fno_n_modes": _modes,
        "tree_leaf_fno_pooling": "sum",
    }
del _modes, _name

# Stable alias for the current comparison-grid default tree surface.
# This decouples the public comparison-grid profile from the underlying
# implementation preset name so future tree revisions can move behind the alias
# without changing every checked-in grid config.
TREE_REFERENCE_PRESET_CONFIGS[COMPARISON_GRID_V3_PRESET] = {
    **TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_FULL_LOCAL_LAWS_PRESET],
}

# Standard unified-g, but with the effective C1 weight cut in half while
# keeping the effective C2/C3 weights unchanged. The standard preset has
# effective weights (0.2666..., 0.2666..., 0.2666...). Choosing
# local_law_weight=2/3 and relative weights (0.5, 1.0, 1.0) yields
# (0.1333..., 0.2666..., 0.2666...).
TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_HALF_LEAF_LAW_PRESET] = {
    **TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_FULL_LOCAL_LAWS_PRESET],
    "local_law_weight": 2.0 / 3.0,
    "c1_relative_weight": 0.5,
    "c2_relative_weight": 1.0,
    "c3_relative_weight": 1.0,
}

# FNO-parity canary: matches standalone FNO architecture at 1-leaf geometry.
# CE classification, single-stage, no local laws → exact FNO parity at 1 leaf.
TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_FNO_PARITY_CANARY_PRESET] = {
    **TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_FULL_LOCAL_LAWS_PRESET],
    "tree_leaf_fno_n_modes": 16,
    "tree_root_supervision_kind": "count_ce",
    "tree_training_schedule": "single_stage",
    "n_epochs": 40,
    "tree_stage1_epochs": 0,
    "tree_stage2_epochs": 0,
    "tree_checkpoint_metric": "val_root_mae",
    "tree_stage1_checkpoint_metric": "val_root_mae",
    "local_law_weight": 0.0,
    "c1_relative_weight": 0.0,
    "c2_relative_weight": 0.0,
    "c3_relative_weight": 0.0,
}

# ---------------------------------------------------------------------------
# Ablation ladder: canary → standard, one change at a time.
# Each step adds back one aspect of the standard preset to isolate its impact.
# ---------------------------------------------------------------------------

# Step 1: canary + MSE regression (swap CE → MSE, keep single-stage + no laws)
TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_ABLATION_MSE_PRESET] = {
    **TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_FNO_PARITY_CANARY_PRESET],
    "tree_root_supervision_kind": "mse",
}

# Step 2: canary + MSE + two-stage schedule (add stage1 local-law-only training)
TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_ABLATION_TWO_STAGE_PRESET] = {
    **TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_ABLATION_MSE_PRESET],
    "tree_training_schedule": "two_stage",
    "tree_stage1_epochs": 10,
    "tree_stage2_epochs": 30,
    "tree_checkpoint_metric": "val_root_mae",
    "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
}

# Step 3: canary + MSE + two-stage + local laws = standard unified_g preset
TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_ABLATION_LOCAL_LAWS_PRESET] = {
    **TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_ABLATION_TWO_STAGE_PRESET],
    "local_law_weight": 0.8,
    "c1_relative_weight": 1.0,
    "c2_relative_weight": 1.0,
    "c3_relative_weight": 1.0,
}

# Multi-leaf root-only baseline: same as standard unified_g but with
# single-stage training and no local laws.  Isolates the merge architecture
# from the training protocol when compared against the standard preset.
UNIFIED_G_MULTI_LEAF_ROOT_ONLY_PRESET = "unified_g_multi_leaf_root_only_v1"
TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_MULTI_LEAF_ROOT_ONLY_PRESET] = {
    **TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_FULL_LOCAL_LAWS_PRESET],
    "tree_training_schedule": "single_stage",
    "n_epochs": 40,
    "tree_stage1_epochs": 0,
    "tree_stage2_epochs": 0,
    "tree_checkpoint_metric": "val_root_mae",
    "tree_stage1_checkpoint_metric": "val_root_mae",
    "local_law_weight": 0.0,
    "c1_relative_weight": 0.0,
    "c2_relative_weight": 0.0,
    "c3_relative_weight": 0.0,
}

TREE_REFERENCE_PRESET_CONFIGS[ROOT_ONLY_PARITY_HISTORICAL_REPLAY_PRESET] = {
    **_SLOTWISE_DENSE,
    **_ROOT_ONLY_FROM_SLOTWISE,
}

TREE_REFERENCE_PRESET_CONFIGS[ROOT_ONLY_PARITY_OPTIMIZATION_FIX_PRESET] = {
    **_SLOTWISE_DENSE,
    **_ROOT_ONLY_FROM_SLOTWISE,
    "n_epochs": 128,
    "tree_training_schedule": "single_stage",
    "tree_checkpoint_metric": "val_root_mae",
    "tree_stage1_checkpoint_metric": "val_root_mae",
    "tree_stage1_root_weight": 1.0,
}

TREE_REFERENCE_PRESET_CONFIGS[ROOT_ONLY_PARITY_CAPACITY_FIX_PRESET] = {
    **_SLOTWISE_DENSE,
    **_ROOT_ONLY_FROM_SLOTWISE,
    "state_dim": 256,
    "hidden_dim": 1024,
}

TREE_REFERENCE_PRESET_CONFIGS[ROOT_ONLY_PARITY_MATCHED_ROOT_V1_PRESET] = {
    **_SLOTWISE_DENSE,
    **_ROOT_ONLY_FROM_SLOTWISE,
    "state_dim": 256,
    "hidden_dim": 1024,
    "n_epochs": 128,
    "tree_training_schedule": "single_stage",
    "tree_checkpoint_metric": "val_root_mae",
    "tree_stage1_checkpoint_metric": "val_root_mae",
    "tree_stage1_root_weight": 1.0,
}

TREE_REFERENCE_PRESET_CONFIGS[STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_V1_PRESET] = {
    **TREE_REFERENCE_PRESET_CONFIGS["structural_factorized_sketch_v3"],
    "leaf_supervision_kind": "count_only",
    "leaf_label_rate": 0.0,
    "internal_supervision_kind": "none",
    "internal_label_rate": 0.0,
    "state_dim": 256,
    "hidden_dim": 1024,
    "n_epochs": 128,
    "tree_training_schedule": "single_stage",
    "tree_checkpoint_metric": "val_root_mae",
    "tree_stage1_checkpoint_metric": "val_root_mae",
    "tree_stage1_root_weight": 1.0,
}

# Corrected one-leaf matched-root parity surface: keep the exact FNO-parity
# canary recipe and vary only the root-supervision budget.
TREE_REFERENCE_PRESET_CONFIGS[ROOT_ONLY_PARITY_MATCHED_ROOT_V2_PRESET] = {
    **TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_FNO_PARITY_CANARY_PRESET],
    "leaf_supervision_kind": "count_only",
    "leaf_label_rate": 0.0,
    "internal_supervision_kind": "none",
    "internal_label_rate": 0.0,
    "vocab_size": 16,
    "tree_stage1_root_weight": 0.0,
}

TREE_REFERENCE_PRESET_CONFIGS[STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_V2_PRESET] = {
    **TREE_REFERENCE_PRESET_CONFIGS[UNIFIED_G_FNO_PARITY_CANARY_PRESET],
    "leaf_supervision_kind": "count_only",
    "leaf_label_rate": 0.0,
    "internal_supervision_kind": "none",
    "internal_label_rate": 0.0,
    "vocab_size": 16,
    "tree_stage1_root_weight": 0.0,
}

TREE_REFERENCE_PRESET_CONFIGS[ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET] = {
    **TREE_REFERENCE_PRESET_CONFIGS[ROOT_ONLY_PARITY_MATCHED_ROOT_V2_PRESET],
    "state_dim": 256,
    "hidden_dim": 1024,
    "n_epochs": 128,
    "tree_leaf_fno_width": 256,
}

TREE_REFERENCE_PRESET_CONFIGS[STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_PRESET] = {
    **TREE_REFERENCE_PRESET_CONFIGS[STRUCTURAL_ROOT_ONLY_PARITY_MATCHED_ROOT_V2_PRESET],
    "state_dim": 256,
    "hidden_dim": 1024,
    "n_epochs": 128,
    "tree_leaf_fno_width": 256,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tree_reference_preset_names(*, public_only: bool = False) -> tuple[str, ...]:
    """Return the registered tree-reference preset names."""
    if public_only:
        return tuple(sorted(TREE_REFERENCE_PRESET_ALIASES))
    return tuple(sorted(TREE_REFERENCE_PRESET_CONFIGS))


def resolve_tree_reference_preset_recipe(name: str) -> str:
    """Resolve a preset name to the underlying recipe id."""
    preset_name = str(name or "").strip()
    if not preset_name:
        raise ValueError("tree reference preset name must be non-empty")
    if preset_name in TREE_REFERENCE_PRESET_ALIASES:
        return str(TREE_REFERENCE_PRESET_ALIASES[preset_name])
    if preset_name in TREE_REFERENCE_PRESET_CONFIGS:
        return preset_name
    raise ValueError(
        "unsupported tree reference preset "
        f"{preset_name!r}; valid recipe ids are {sorted(TREE_REFERENCE_PRESET_CONFIGS)} "
        f"and public aliases are {sorted(TREE_REFERENCE_PRESET_ALIASES)}"
    )


def resolve_tree_reference_preset_config(name: str) -> dict[str, Any]:
    """Return the resolved config for a preset name or public alias."""
    recipe_name = resolve_tree_reference_preset_recipe(name)
    return dict(TREE_REFERENCE_PRESET_CONFIGS[recipe_name])


def resolve_tree_reference_preset(name: str) -> dict[str, Any]:
    """Return the full resolved preset record for a preset name or alias."""
    preset_name = str(name or "").strip()
    recipe_name = resolve_tree_reference_preset_recipe(preset_name)
    return {
        "requested_name": preset_name,
        "public_name": preset_name,
        "recipe_name": recipe_name,
        "is_public_alias": preset_name in TREE_REFERENCE_PRESET_ALIASES,
        "config": resolve_tree_reference_preset_config(preset_name),
    }


def tree_reference_label(tree_reference: Mapping[str, Any]) -> str:
    """Extract a human-readable label from a tree_reference dict."""
    return str(
        tree_reference.get("preset_display_name")
        or tree_reference.get("preset_public_name")
        or tree_reference.get("preset_requested_name")
        or tree_reference.get("preset_recipe")
        or tree_reference.get("preset")
        or tree_reference.get("winning_config_label")
        or tree_reference.get("capacity_root")
        or ""
    ).strip()
