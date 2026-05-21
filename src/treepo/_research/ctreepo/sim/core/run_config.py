"""Shared run configuration types for full-doc tree/FNO experiments.

RunConfigSpec is the canonical scientific config surface. All launch scripts
should construct configs through this type (or through run_config_from_mapping)
rather than building raw dicts.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping

from treepo._research.ctreepo.sim.core.full_doc_config_codec import (
    runtime_config_overrides_from_config_like,
    write_tree_run_config_json,
)
from treepo._research.ctreepo.sim.core.run_intent import (
    TOPOLOGY_FULL_DOC,
    TOPOLOGY_TREE,
    VALID_TOPOLOGIES,
    resolve_package_semantics,
)


# ---------------------------------------------------------------------------
# Batch pack mode helpers
# ---------------------------------------------------------------------------

def default_tree_batch_pack_mode(benchmark: str) -> str:
    return (
        "fixed_fused"
        if str(benchmark).strip().lower() == "recoverable_v4"
        else "structure_bucket"
    )


def resolved_tree_batch_pack_mode(*, benchmark: str, raw_value: str | None) -> str:
    raw = str(raw_value or "").strip()
    if raw:
        return raw
    return default_tree_batch_pack_mode(benchmark)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def sanitize_label(value: str) -> str:
    return (
        str(value or "")
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def format_float_label(value: float, *, precision: int = 4) -> str:
    formatted = f"{float(value):.{int(precision)}f}".rstrip("0").rstrip(".")
    return formatted or "0"


# ---------------------------------------------------------------------------
# RunConfigSpec — canonical scientific config surface
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfigSpec:
    label: str
    state_dim: int
    hidden_dim: int
    n_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    baseline_family: str = ""
    topology: str = ""  # "tree", "full_doc", or "" (inferred from fixed_leaf_tokens)
    fixed_leaf_tokens: int | None = None
    tree_local_law_weight: float | None = None
    tree_task_objective_weight: float | None = None
    tree_local_weighting_mode: str = "fixed_k_hajek"
    tree_exact_collapse_mode: str = ""
    tree_c1_relative_weight: float = 1.0
    tree_c2_relative_weight: float = 1.0
    tree_c3_relative_weight: float = 1.0
    official_fno_preserve_requested_leaf_tokens: bool = False
    preserve_requested_leaf_tokens: bool = False
    comparison_mode: str = "legacy"
    tree_leaf_fno_width: int | None = None
    tree_leaf_fno_n_modes: int | None = None
    tree_leaf_fno_n_layers: int | None = None
    tree_leaf_fno_pooling: str | None = None
    tree_model_version: str = ""
    tree_batch_runtime_mode: str = ""
    tree_root_supervision_kind: str = "mse"
    tree_document_loss_normalization_mode: str = "auto"
    tree_supervision_source: str = "rate"
    tree_checkpoint_metric: str = "val_root_mae"
    tree_stage1_checkpoint_metric: str = "val_root_mae"
    tree_stage1_eval_mode: str = "per_epoch"
    tree_stage1_screen_doc_limit: int = 0
    tree_stage1_final_exact_doc_limit: int = 0
    exact_metric_selection_doc_limit: int = 0
    exact_metric_selection_interval: int = 1
    tree_exact_eval_max_docs: int = 0
    tree_posttrain_train_doc_limit: int = 0
    tree_batch_pack_mode: str = "structure_bucket"
    tree_batch_token_budget: int = 0
    tree_batch_node_budget: int = 0
    tree_batch_autotune: bool = True
    tree_batch_structural_pad_limit: float = 0.5
    tree_batch_auto_queue_min_docs: int = 8
    tree_batch_auto_queue_min_fill_ratio: float = 0.5
    tree_eval_workers_per_mig: int = 0
    gpu_runtime_data_mode: str = "resident"
    gpu_runtime_bucket_mode: str = "exact_then_bucketed"
    gpu_runtime_preload_splits: tuple[str, ...] = ("train", "val", "test")
    gpu_runtime_preload_targets: bool = True
    gpu_runtime_workers_per_mig: int = 1
    gpu_runtime_allow_multi_worker_screen: bool = True
    gpu_runtime_capacity_workers_per_mig: int = 2
    tree_stage1_artifact_dir: str = ""
    prepared_data_root: str = ""
    prepared_data_allow_create: bool = True
    base_bundle_path: str = ""
    diagnostic_detail_mode: str = "summary"
    posttrain_diagnostics_mode: str = ""
    raw_diagnostic_artifact_dir: str = ""
    tree_stage1_root_weight: float = 0.0
    tree_join_bit_weight: float = 0.0
    tree_training_schedule: str = "two_stage"
    tree_stage1_epochs: int = 12
    tree_stage2_epochs: int = 20
    tree_task_head_mode: str = "full_state_scalar"
    tree_theorem_surface_mode: str = "slotwise"
    tree_theorem_count_head_mode: str = "scalar_mse"
    tree_theorem_count_ordinal_weight: float = 1.0
    tree_theorem_count_scalar_aux_weight: float = 0.25
    tree_theorem_count_threshold_balance: bool = True
    tree_theorem_feature_dim: int = 48
    tree_theorem_feature_hidden_dim: int = 256
    tree_merge_hidden_dim: int = 0
    tree_phi_compose_weight: float = 1.0
    tree_phi_contrastive_weight: float = 0.25
    tree_phi_alignment_loss: str = "cosine_mse"
    tree_c2_mode: str = "reconstruction"
    oracle_metric_name: str = ""
    oracle_same_threshold: float = 0.0
    oracle_diff_threshold: float = 0.0
    theorem_feature_adapter: str = "markov_count_sketch"
    theorem_pair_same_threshold: float | None = None
    theorem_pair_diff_threshold: float | None = None
    tree_summary_spec_root_mode: str = "task_split_ablation"
    doc_sequence_train_fraction: float = 0.0
    aligned_sketch_surface: str = ""
    summary_spec_name: str = ""
    slot_count: int = 0
    tree_theorem_score_dim: int = 0
    tree_theorem_fiber_dim: int = 0
    tree_theorem_aux_dim: int = 0
    tree_score_merge_mode: str = "gated_affine"
    tree_theorem_count_dim: int = 0
    tree_theorem_first_dim: int = 0
    tree_theorem_last_dim: int = 0
    leaf_supervision_kind: str = "full_sketch"
    internal_supervision_kind: str = "none"
    internal_label_rate: float = 0.0
    max_internal_depth: int = 0
    leaf_exact_supervision: bool = False
    leaf_label_rate: float = 1.0
    root_weight: float = 1.0
    schedule_consistency_weight: float = 0.0
    endpoint_loss_scale: float = 1.0
    budget_total_calls: int = 0
    budget_total_calls_per_doc: float = 0.0
    mass_target_per_doc: float = float("nan")
    full_doc_budget_share: float = 1.0
    doc_consumption_mode: str = ""
    local_split_mode: str = ""
    local_allocation_policy: str = ""
    package_semantics: str = ""
    depth_discount_gamma: float = 1.0

    def __post_init__(self) -> None:
        topology = str(self.topology or "").strip()
        if topology not in VALID_TOPOLOGIES:
            raise ValueError(
                f"topology must be one of {sorted(VALID_TOPOLOGIES)}, got {topology!r}"
            )
        if self.tree_local_law_weight is not None:
            local = float(self.tree_local_law_weight)
            if local < 0.0 or local > 1.0:
                raise ValueError("tree_local_law_weight must be in [0, 1]")
        if self.tree_task_objective_weight is not None and float(self.tree_task_objective_weight) < 0.0:
            raise ValueError("tree_task_objective_weight must be non-negative")
        if self.tree_local_law_weight is not None and self.tree_task_objective_weight is not None:
            raise ValueError(
                "tree_local_law_weight is mutually exclusive with tree_task_objective_weight"
            )


# ---------------------------------------------------------------------------
# JobSpec — job identity + config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JobSpec:
    family: str
    train_doc_count: int
    benchmark: str
    hardness_grid: str
    grid_cell_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    config: RunConfigSpec
    tuning_stage: str = ""
    test_metrics_hidden_during_selection: bool = False
    study_name: str = ""
    study_axis: str = ""
    axis_value: str = ""
    locked_tree_neural_config_label: str = ""
    selection_metric: str = ""

    def __post_init__(self) -> None:
        family = str(self.family or "").strip()
        if not family:
            raise ValueError("JobSpec.family must be non-empty")
        config_family = str(getattr(self.config, "baseline_family", "") or "").strip()
        if config_family and config_family != family:
            raise ValueError(
                f"JobSpec family/config mismatch: job.family={family!r} "
                f"config.baseline_family={config_family!r}"
            )
        if not config_family:
            object.__setattr__(
                self,
                "config",
                replace(self.config, baseline_family=family),
            )

    @property
    def job_name(self) -> str:
        scope = self.hardness_grid or self.benchmark
        cell_suffix = ""
        if self.grid_cell_ids:
            cell_suffix = "__" + "_".join(str(cell) for cell in self.grid_cell_ids)
        leaf_suffix = ""
        if self.config.fixed_leaf_tokens is not None:
            leaf_suffix = f"__leaf_{int(self.config.fixed_leaf_tokens)}"
        seed_suffix = ""
        if len(self.seeds) == 1:
            seed_suffix = f"__seed_{int(self.seeds[0])}"
        stage_suffix = ""
        if str(self.tuning_stage).strip():
            stage_suffix = f"__stage_{str(self.tuning_stage)}"
        config_suffix = ""
        if str(self.config.label).strip():
            config_suffix = f"__cfg_{str(self.config.label)}"
        study_suffix = ""
        study_axis = sanitize_label(str(self.study_axis))
        axis_value = sanitize_label(str(self.axis_value))
        if study_axis and axis_value:
            study_suffix = f"__{study_axis}_{axis_value}"
        return (
            f"{scope}__{self.family}__train_{int(self.train_doc_count)}"
            f"{cell_suffix}{leaf_suffix}{stage_suffix}{config_suffix}{study_suffix}{seed_suffix}"
        )

    @property
    def budget_total_calls(self) -> int:
        return int(self.config.budget_total_calls)

    @property
    def budget_total_calls_per_doc(self) -> float:
        return float(self.config.budget_total_calls_per_doc)

    @property
    def mass_target_per_doc(self) -> float:
        return float(self.config.mass_target_per_doc)

    @property
    def full_doc_budget_share(self) -> float:
        return float(self.config.full_doc_budget_share)

    @property
    def doc_consumption_mode(self) -> str:
        return str(self.config.doc_consumption_mode)

    @property
    def local_split_mode(self) -> str:
        return str(self.config.local_split_mode)

    @property
    def local_allocation_policy(self) -> str:
        return str(self.config.local_allocation_policy)

    @property
    def package_semantics(self) -> str:
        return str(self.config.package_semantics)


# ---------------------------------------------------------------------------
# Config manipulation helpers
# ---------------------------------------------------------------------------

def with_run_intent_overrides(
    config: RunConfigSpec,
    *,
    budget_total_calls: int | None = None,
    budget_total_calls_per_doc: float | None = None,
    mass_target_per_doc: float | None = None,
    full_doc_budget_share: float | None = None,
    doc_consumption_mode: str | None = None,
    local_split_mode: str | None = None,
    local_allocation_policy: str | None = None,
    package_semantics: str | None = None,
    depth_discount_gamma: float | None = None,
) -> RunConfigSpec:
    updated = replace(
        config,
        budget_total_calls=(
            int(budget_total_calls)
            if budget_total_calls is not None
            else int(config.budget_total_calls)
        ),
        budget_total_calls_per_doc=(
            float(budget_total_calls_per_doc)
            if budget_total_calls_per_doc is not None
            else float(config.budget_total_calls_per_doc)
        ),
        mass_target_per_doc=(
            float(mass_target_per_doc)
            if mass_target_per_doc is not None
            else float(config.mass_target_per_doc)
        ),
        full_doc_budget_share=(
            float(full_doc_budget_share)
            if full_doc_budget_share is not None
            else float(config.full_doc_budget_share)
        ),
        doc_consumption_mode=(
            str(doc_consumption_mode)
            if doc_consumption_mode is not None
            else str(config.doc_consumption_mode)
        ),
        local_split_mode=(
            str(local_split_mode)
            if local_split_mode is not None
            else str(config.local_split_mode)
        ),
        local_allocation_policy=(
            str(local_allocation_policy)
            if local_allocation_policy is not None
            else str(config.local_allocation_policy)
        ),
        package_semantics=(
            str(package_semantics)
            if package_semantics is not None
            else str(config.package_semantics)
        ),
        depth_discount_gamma=(
            float(depth_discount_gamma)
            if depth_discount_gamma is not None
            else float(config.depth_discount_gamma)
        ),
    )
    recompute_package_semantics = package_semantics is None and any(
        value is not None
        for value in (
            budget_total_calls,
            budget_total_calls_per_doc,
            mass_target_per_doc,
            full_doc_budget_share,
            doc_consumption_mode,
            local_split_mode,
            local_allocation_policy,
        )
    )
    if recompute_package_semantics or not str(updated.package_semantics).strip():
        package_semantics_mapping = asdict(updated)
        if recompute_package_semantics:
            package_semantics_mapping["package_semantics"] = ""
        updated = replace(
            updated,
            package_semantics=str(resolve_package_semantics(package_semantics_mapping)),
        )
    return updated


def config_mapping_for_run_config(config: RunConfigSpec) -> Dict[str, Any]:
    return runtime_config_overrides_from_config_like(config)


def write_run_config_spec(path: Path, config: RunConfigSpec) -> None:
    write_tree_run_config_json(path, config)


def run_config_from_mapping(mapping: Mapping[str, Any]) -> RunConfigSpec:
    """Create a RunConfigSpec from a flat dict, applying defaults for missing fields."""
    legacy_objective_fields = sorted(
        key
        for key in (
            "task_objective_weight",
            "tree_local_law_weight",
            "tree_task_objective_weight",
        )
        if key in mapping and mapping.get(key) not in {"", None}
    )
    if legacy_objective_fields:
        raise ValueError(
            "legacy public objective config fields are not supported: "
            + ", ".join(legacy_objective_fields)
            + ". Use local_law_weight and root_share."
        )

    def _first_present(*keys: str) -> Any:
        for key in keys:
            value = mapping.get(key)
            if value not in {"", None}:
                return value
        return None

    requested_fixed_leaf_tokens = (
        None
        if mapping.get("fixed_leaf_tokens") in {"", None}
        else int(mapping.get("fixed_leaf_tokens"))
    )
    preserve_by_default = bool(
        requested_fixed_leaf_tokens is not None
        and int(requested_fixed_leaf_tokens) > 0
    )
    return with_run_intent_overrides(
        RunConfigSpec(
        label=sanitize_label(str(mapping.get("label", "default"))),
        state_dim=int(mapping.get("state_dim", 128)),
        hidden_dim=int(mapping.get("hidden_dim", 512)),
        n_epochs=int(mapping.get("n_epochs", 32)),
        batch_size=int(mapping.get("batch_size", 64)),
        lr=float(mapping.get("lr", 5e-4)),
        weight_decay=float(mapping.get("weight_decay", 0.0)),
        baseline_family=str(mapping.get("baseline_family", "") or ""),
        topology=str(mapping.get("topology", "") or ""),
        fixed_leaf_tokens=(
            None
            if mapping.get("fixed_leaf_tokens") in {"", None}
            else int(mapping.get("fixed_leaf_tokens"))
        ),
        tree_local_law_weight=(
            None
            if _first_present("local_law_weight") is None
            else float(_first_present("local_law_weight"))
        ),
        tree_task_objective_weight=(
            None
            if _first_present("root_share")
            is None
            else float(_first_present("root_share"))
        ),
        tree_local_weighting_mode=str(
            mapping.get("tree_local_weighting_mode", "fixed_k_hajek")
            or "fixed_k_hajek"
        ),
        tree_exact_collapse_mode=str(
            mapping.get("tree_exact_collapse_mode", "") or ""
        ),
        tree_c1_relative_weight=float(
            1.0
            if _first_present("tree_c1_relative_weight", "c1_relative_weight")
            is None
            else _first_present("tree_c1_relative_weight", "c1_relative_weight")
        ),
        tree_c2_relative_weight=float(
            1.0
            if _first_present("tree_c2_relative_weight", "c2_relative_weight")
            is None
            else _first_present("tree_c2_relative_weight", "c2_relative_weight")
        ),
        tree_c3_relative_weight=float(
            1.0
            if _first_present("tree_c3_relative_weight", "c3_relative_weight")
            is None
            else _first_present("tree_c3_relative_weight", "c3_relative_weight")
        ),
        official_fno_preserve_requested_leaf_tokens=bool(
            mapping.get(
                "official_fno_preserve_requested_leaf_tokens",
                preserve_by_default,
            )
        ),
        preserve_requested_leaf_tokens=bool(
            mapping.get("preserve_requested_leaf_tokens", preserve_by_default)
            or mapping.get(
                "official_fno_preserve_requested_leaf_tokens",
                preserve_by_default,
            )
        ),
        comparison_mode=str(mapping.get("comparison_mode", "legacy") or "legacy"),
        tree_leaf_fno_width=(
            None
            if mapping.get("tree_leaf_fno_width") in {"", None}
            else int(mapping.get("tree_leaf_fno_width"))
        ),
        tree_leaf_fno_n_modes=(
            None
            if mapping.get("tree_leaf_fno_n_modes") in {"", None}
            else int(mapping.get("tree_leaf_fno_n_modes"))
        ),
        tree_leaf_fno_n_layers=(
            None
            if mapping.get("tree_leaf_fno_n_layers") in {"", None}
            else int(mapping.get("tree_leaf_fno_n_layers"))
        ),
        tree_model_version=str(mapping.get("tree_model_version", "")),
        tree_batch_runtime_mode=str(mapping.get("tree_batch_runtime_mode", "")),
        tree_root_supervision_kind=str(
            mapping.get("tree_root_supervision_kind", "mse")
        ),
        tree_document_loss_normalization_mode=str(
            mapping.get("tree_document_loss_normalization_mode", "auto") or "auto"
        ),
        tree_supervision_source=str(
            mapping.get("tree_supervision_source", "rate") or "rate"
        ),
        tree_checkpoint_metric=str(
            mapping.get("tree_checkpoint_metric", "val_root_mae")
        ),
        tree_stage1_checkpoint_metric=str(
            mapping.get("tree_stage1_checkpoint_metric", "val_root_mae")
        ),
        tree_stage1_eval_mode=str(
            mapping.get("tree_stage1_eval_mode", "per_epoch")
        ),
        tree_stage1_screen_doc_limit=int(
            mapping.get("tree_stage1_screen_doc_limit", 0)
        ),
        tree_stage1_final_exact_doc_limit=int(
            mapping.get("tree_stage1_final_exact_doc_limit", 0)
        ),
        exact_metric_selection_doc_limit=int(
            mapping.get("exact_metric_selection_doc_limit", 0)
        ),
        exact_metric_selection_interval=int(
            mapping.get("exact_metric_selection_interval", 1)
        ),
        tree_exact_eval_max_docs=int(mapping.get("tree_exact_eval_max_docs", 0)),
        tree_posttrain_train_doc_limit=int(
            mapping.get("tree_posttrain_train_doc_limit", 0)
        ),
        tree_batch_pack_mode=str(
            resolved_tree_batch_pack_mode(
                benchmark=str(mapping.get("benchmark", "")),
                raw_value=str(mapping.get("tree_batch_pack_mode", "")),
            )
        ),
        tree_batch_token_budget=int(mapping.get("tree_batch_token_budget", 0)),
        tree_batch_node_budget=int(mapping.get("tree_batch_node_budget", 0)),
        tree_batch_autotune=bool(mapping.get("tree_batch_autotune", True)),
        tree_batch_structural_pad_limit=float(
            mapping.get("tree_batch_structural_pad_limit", 0.5)
        ),
        tree_batch_auto_queue_min_docs=int(
            mapping.get("tree_batch_auto_queue_min_docs", 8)
        ),
        tree_batch_auto_queue_min_fill_ratio=float(
            mapping.get("tree_batch_auto_queue_min_fill_ratio", 0.5)
        ),
        tree_eval_workers_per_mig=int(mapping.get("tree_eval_workers_per_mig", 0)),
        gpu_runtime_data_mode=str(mapping.get("gpu_runtime_data_mode", "resident")),
        gpu_runtime_bucket_mode=str(
            mapping.get("gpu_runtime_bucket_mode", "exact_then_bucketed")
        ),
        gpu_runtime_preload_splits=tuple(
            str(item)
            for item in (
                mapping.get("gpu_runtime_preload_splits", ("train", "val", "test"))
                if not isinstance(mapping.get("gpu_runtime_preload_splits", ("train", "val", "test")), str)
                else str(mapping.get("gpu_runtime_preload_splits", "train val test")).replace(",", " ").split()
            )
            if str(item).strip()
        ),
        gpu_runtime_preload_targets=bool(
            mapping.get("gpu_runtime_preload_targets", True)
        ),
        gpu_runtime_workers_per_mig=int(
            mapping.get("gpu_runtime_workers_per_mig", 1)
        ),
        gpu_runtime_allow_multi_worker_screen=bool(
            mapping.get("gpu_runtime_allow_multi_worker_screen", True)
        ),
        gpu_runtime_capacity_workers_per_mig=int(
            mapping.get("gpu_runtime_capacity_workers_per_mig", 2)
        ),
        tree_stage1_artifact_dir=str(mapping.get("tree_stage1_artifact_dir", "")),
        prepared_data_root=str(mapping.get("prepared_data_root", "")),
        prepared_data_allow_create=bool(
            mapping.get("prepared_data_allow_create", True)
        ),
        diagnostic_detail_mode=str(mapping.get("diagnostic_detail_mode", "summary")),
        posttrain_diagnostics_mode=str(
            mapping.get("posttrain_diagnostics_mode", "")
        ),
        raw_diagnostic_artifact_dir=str(
            mapping.get("raw_diagnostic_artifact_dir", "")
        ),
        tree_stage1_root_weight=float(mapping.get("tree_stage1_root_weight", 0.0)),
        tree_join_bit_weight=float(mapping.get("tree_join_bit_weight", 0.0)),
        tree_training_schedule=str(mapping.get("tree_training_schedule", "two_stage")),
        tree_stage1_epochs=int(mapping.get("tree_stage1_epochs", 0)),
        tree_stage2_epochs=int(mapping.get("tree_stage2_epochs", 0)),
        tree_task_head_mode=str(mapping.get("tree_task_head_mode", "full_state_scalar")),
        tree_theorem_surface_mode=str(
            mapping.get("tree_theorem_surface_mode", "slotwise")
        ),
        tree_theorem_count_head_mode=str(
            mapping.get("tree_theorem_count_head_mode", "scalar_mse")
        ),
        tree_theorem_feature_dim=int(mapping.get("tree_theorem_feature_dim", 48)),
        tree_theorem_feature_hidden_dim=int(
            mapping.get("tree_theorem_feature_hidden_dim", 256)
        ),
        tree_merge_hidden_dim=int(mapping.get("tree_merge_hidden_dim", 0)),
        tree_theorem_score_dim=int(mapping.get("tree_theorem_score_dim", 0)),
        tree_theorem_fiber_dim=int(mapping.get("tree_theorem_fiber_dim", 0)),
        tree_theorem_aux_dim=int(mapping.get("tree_theorem_aux_dim", 0)),
        tree_score_merge_mode=str(mapping.get("tree_score_merge_mode", "gated_affine")),
        tree_phi_compose_weight=float(mapping.get("tree_phi_compose_weight", 1.0)),
        tree_phi_contrastive_weight=float(
            mapping.get("tree_phi_contrastive_weight", 0.25)
        ),
        tree_phi_alignment_loss=str(
            mapping.get("tree_phi_alignment_loss", "cosine_mse")
        ),
        tree_theorem_count_ordinal_weight=float(
            mapping.get("tree_theorem_count_ordinal_weight", 1.0)
        ),
        tree_theorem_count_scalar_aux_weight=float(
            mapping.get("tree_theorem_count_scalar_aux_weight", 0.25)
        ),
        tree_theorem_count_threshold_balance=bool(
            mapping.get("tree_theorem_count_threshold_balance", True)
        ),
        tree_summary_spec_root_mode=str(
            mapping.get("tree_summary_spec_root_mode", "task_split_ablation")
        ),
        tree_c2_mode=str(mapping.get("tree_c2_mode", "reconstruction")),
        oracle_metric_name=str(mapping.get("oracle_metric_name", "")),
        oracle_same_threshold=float(mapping.get("oracle_same_threshold", 0.0)),
        oracle_diff_threshold=float(mapping.get("oracle_diff_threshold", 0.0)),
        theorem_feature_adapter=str(
            mapping.get("theorem_feature_adapter", "markov_count_sketch")
        ),
        theorem_pair_same_threshold=(
            None
            if mapping.get("theorem_pair_same_threshold", None) is None
            else float(mapping.get("theorem_pair_same_threshold"))
        ),
        theorem_pair_diff_threshold=(
            None
            if mapping.get("theorem_pair_diff_threshold", None) is None
            else float(mapping.get("theorem_pair_diff_threshold"))
        ),
        doc_sequence_train_fraction=float(
            mapping.get("doc_sequence_train_fraction", 0.0)
        ),
        aligned_sketch_surface=str(mapping.get("aligned_sketch_surface", "")),
        summary_spec_name=str(mapping.get("summary_spec_name", "")),
        slot_count=int(mapping.get("slot_count", 0)),
        tree_theorem_count_dim=int(mapping.get("tree_theorem_count_dim", 0)),
        tree_theorem_first_dim=int(mapping.get("tree_theorem_first_dim", 0)),
        tree_theorem_last_dim=int(mapping.get("tree_theorem_last_dim", 0)),
        internal_supervision_kind=str(
            mapping.get("internal_supervision_kind", "none")
        ),
        internal_label_rate=float(mapping.get("internal_label_rate", 0.0)),
        max_internal_depth=int(mapping.get("max_internal_depth", 0)),
        leaf_exact_supervision=bool(mapping.get("leaf_exact_supervision", False)),
        leaf_supervision_kind=str(mapping.get("leaf_supervision_kind", "full_sketch")),
        leaf_label_rate=float(mapping.get("leaf_label_rate", 1.0)),
        root_weight=float(mapping.get("root_weight", 1.0)),
        schedule_consistency_weight=float(
            mapping.get("schedule_consistency_weight", 0.0)
        ),
        endpoint_loss_scale=float(mapping.get("endpoint_loss_scale", 1.0)),
        ),
        budget_total_calls=int(mapping.get("budget_total_calls", 0)),
        budget_total_calls_per_doc=float(
            mapping.get("budget_total_calls_per_doc", 0.0)
        ),
        mass_target_per_doc=(
            float("nan")
            if mapping.get("mass_target_per_doc") in {"", None}
            else float(mapping.get("mass_target_per_doc"))
        ),
        full_doc_budget_share=float(
            mapping.get("full_doc_budget_share", 1.0)
        ),
        doc_consumption_mode=str(mapping.get("doc_consumption_mode", "") or ""),
        local_split_mode=str(mapping.get("local_split_mode", "") or ""),
        local_allocation_policy=str(
            mapping.get("local_allocation_policy", "") or ""
        ),
        package_semantics=str(mapping.get("package_semantics", "") or ""),
        depth_discount_gamma=float(mapping.get("depth_discount_gamma", 1.0)),
    )
