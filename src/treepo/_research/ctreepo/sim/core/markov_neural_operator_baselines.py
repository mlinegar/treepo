"""Tree-specific FNO neural operator code for the Markov changepoint task.

Standalone baseline models (FNOCountPredictor, DeepONetCountPredictor, etc.)
have been extracted to fno_doc_baselines.py. This module contains the tree-merge
operator (FNOCountSketch) and its training/evaluation infrastructure.

All names are re-exported for backward compatibility — existing import paths
from this module continue to work unchanged.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-export facade: standalone baselines (extracted to fno_doc_baselines.py)
# ---------------------------------------------------------------------------
from treepo._research.ctreepo.sim.core.fno_doc_baselines import (  # noqa: F401
    HAS_NEURAL_OPERATOR,
    FNOTokenEncoder,
    apply_fno_token_encoder,
    FNOCountPredictor,
    DeepONetCountPredictor,
    MLPBigramCountPredictor,
    CNN1DCountPredictor,
    FNO_TREE_C2_METRIC_KIND,
    FNO_TREE_C2_PROXY_METRIC_KIND,
    FNO_TREE_C2_EXACT_WITNESS_KIND,
    DECODED_MARKOV_SKETCH_SURFACE,
    MARKOV_COUNT_SKETCH_SUMMARY_SPEC,
    VALID_MARKOV_MERGE_OBJECTIVE_MODES,
    VALID_MARKOV_MERGE_WEIGHTING_MODES,
    _bigram_features_from_tokens,
    _class_setup,
    _train_loop_with_predictions,
    _train_loop,
    _fit_fno_baseline,
    _fit_fno_baseline_with_predictions,
    _fit_deeponet_baseline,
    _fit_mlp_bigram_baseline,
    _fit_mlp_bigram_baseline_with_predictions,
    _fit_cnn1d_baseline,
    _fit_cnn1d_baseline_with_predictions,
    _NeuralOpFNO,
    _INSTALL_MSG,
)

# ---------------------------------------------------------------------------
# Tree-specific imports (needed by the code below)
# ---------------------------------------------------------------------------
import ctypes
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import lru_cache
import gc
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from treepo._research.core.autotune_probe_cache import (
    AUTOTUNE_PROBE_CACHE_VERSION,
    ProbeCacheEntry,
    ProbeCacheStore,
    ProbeCandidateProfile,
    ProbeRunProfile,
    build_probe_cache_key,
    classify_device_signature,
)
from treepo._research.core.unified_runtime import (
    BatchTelemetry,
    GPU_RUNTIME_BUCKET_MODE_EXACT_THEN_BUCKETED,
    GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE,
    GPU_RUNTIME_DATA_MODE_CPU_DEBUG,
    RUNTIME_MODE_UNIFIED_V2,
    GpuBatchStore,
    GpuBatchStoreKey,
    GpuBatchView,
    GpuRuntimeConfig,
    GpuRuntimeTelemetry,
    WorkItem,
    build_leaf_count_auto_queue_targets,
    gpu_runtime_config_from_mapping,
    get_named_plan_cache,
    normalize_gpu_runtime_bucket_mode,
    plan_work_batches,
    resolve_runtime_mode,
)

from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (
    ChangepointMarkovDoc,
    ObjectiveMetrics,
    OPSCountConfig,
    SketchMetrics,
    TrainFitDiagnostics,
    VALID_INTERNAL_SUPERVISION_KINDS,
    VALID_LEAF_SUPERVISION_KINDS,
    VALID_TREE_CHECKPOINT_METRICS,
    VALID_TREE_DOCUMENT_LOSS_NORMALIZATION_MODES,
    VALID_TREE_LOCAL_WEIGHTING_MODES,
    VALID_TREE_ROOT_SUPERVISION_KINDS,
    VALID_TREE_SUMMARY_SPEC_ROOT_MODES,
    VALID_TREE_SUPERVISION_SOURCES,
    VALID_TREE_THEOREM_COUNT_HEAD_MODES,
    VALID_TREE_THEOREM_SURFACE_MODES,
    VALID_TREE_SCORE_MERGE_MODES,
    VALID_TREE_TRAINING_SCHEDULES,
    VALID_SCHEDULES,
    _condition_error_diagnostics,
    _eval_root_predictions,
    _exact_match_rate,
    _predict_count_from_summary,
    _resummary_summary_sequence,
    _set_global_seed,
    _token_sequence_arrays,
    _zero_sketch_metrics,
)
from treepo._research.ctreepo.sim.core.theorem_feature_route import (
    DEFAULT_THEOREM_FEATURE_ADAPTER,
    TheoremFeatureAdapter,
    build_theorem_feature_pair_sets,
    load_theorem_feature_stage1_artifact,
    resolve_theorem_feature_adapter,
    theorem_feature_pair_metrics_from_scores,
    theorem_feature_targets_from_markov_exact_targets,
    write_theorem_feature_stage1_artifact,
)
from treepo._research.ctreepo.sim.core.oracle_metric import (
    OracleMetricSpace,
    build_contrastive_pairs,
    contrastive_fiber_loss,
)
from treepo._research.ctreepo.sim.core.markov_theorem_feature_adapter import (
    COARSENED_THEOREM_FEATURE_ADAPTER,
    SCOREFIBER_MARKOV_ENDPOINTS_ADAPTER,
    MarkovTheoremFeatureLabel,
    ScoreFiberTheoremFeatureLabel,
)
from treepo._research.ctreepo.sim.core.training_selection import (
    TrainingSelectionMetadata,
    clone_module_state,
    improved_metric,
    restore_module_state,
)
from treepo._research.tree.full_tree_ipw import (
    DocumentLevelPredictionRecord,
    FullTreeIPWSummaryAccumulator,
    FullTreeNodeRecord,
    summarize_full_tree_ipw,
)
from treepo._research.tree.ipw import NodeType
from treepo._research.tree.state_tree import StateNode, StateTree
from treepo._research.tree.tree_model_v2 import (
    TreeModelV2View,
    normalize_tree_model_version,
)
from treepo._research.training.supervision.local_law_torch import (
    local_law_objective_target_mse,
)
from treepo._research.core.local_law_adjustment import (
    LOCAL_LAW_OBJECTIVE_CORRECTED,
    normalize_local_law_objective_mode,
)

_CUDA_FAST_MATH_CONFIGURED = False


# ---------------------------------------------------------------------------
# FNOCountSketch: FNO-backed tree-merge operator with local law interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FNOCountDoc:
    """Like _CountDoc but also carries raw token IDs per leaf span."""

    n_tokens: int
    leaf_token_ids: Tuple[Tuple[int, ...], ...]  # raw tokens per leaf
    leaf_counts: Tuple[float, ...]
    leaf_first_regimes: Tuple[int, ...]  # regime id of first token in each leaf
    leaf_last_regimes: Tuple[int, ...]  # regime id of last token in each leaf
    leaf_token_lengths: Tuple[int, ...]
    merge_counts_balanced: Tuple[float, ...]
    merge_sizes_balanced: Tuple[int, ...]
    merge_token_lengths: Tuple[int, ...]
    root_count: float
    proxy_leaf_counts: Tuple[float, ...] = tuple()
    proxy_merge_counts_balanced: Tuple[float, ...] = tuple()


@dataclass(frozen=True)
class _PrecomputedDocStateView:
    state_batch: torch.Tensor
    root_state: torch.Tensor
    merge_states: Tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _PrecomputedBatchTreeLevels:
    leaf_states: torch.Tensor
    merge_levels: Tuple[torch.Tensor, ...]
    root_states: torch.Tensor
    leaf_valid_mask: torch.Tensor = field(
        default_factory=lambda: torch.zeros((0, 0), dtype=torch.bool)
    )
    merge_valid_levels: Tuple[torch.Tensor, ...] = tuple()
    node_valid_mask: torch.Tensor = field(
        default_factory=lambda: torch.zeros((0, 0), dtype=torch.bool)
    )


@dataclass(frozen=True)
class _CachedFusedDocTargets:
    leaf_count_targets_cpu: torch.Tensor
    leaf_first_targets_cpu: torch.Tensor
    leaf_last_targets_cpu: torch.Tensor
    leaf_valid_mask_cpu: torch.Tensor
    merge_count_targets_cpu: torch.Tensor
    merge_first_targets_cpu: torch.Tensor
    merge_last_targets_cpu: torch.Tensor
    merge_valid_mask_cpu: torch.Tensor
    node_count_targets_cpu: torch.Tensor
    node_count_keys_cpu: torch.Tensor
    node_first_targets_cpu: torch.Tensor
    node_last_targets_cpu: torch.Tensor
    node_valid_mask_cpu: torch.Tensor


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _trim_host_allocator() -> None:
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        pass


def _merge_gpu_runtime_telemetry(
    dest: GpuRuntimeTelemetry,
    src: GpuRuntimeTelemetry | None,
) -> GpuRuntimeTelemetry:
    if src is None:
        return dest
    dest.resident_store_build_time_s += float(src.resident_store_build_time_s)
    dest.steady_state_h2d_bytes += int(src.steady_state_h2d_bytes)
    dest.steady_state_h2d_events += int(src.steady_state_h2d_events)
    dest.steady_state_h2d_time_s += float(src.steady_state_h2d_time_s)
    dest.resident_store_hits += int(src.resident_store_hits)
    dest.resident_store_misses += int(src.resident_store_misses)
    for key, value in dict(src.cpu_fallback_reason_counts).items():
        dest.cpu_fallback_reason_counts[str(key)] = (
            int(dest.cpu_fallback_reason_counts.get(str(key), 0)) + int(value)
        )
    for key, value in dict(src.extra_counters).items():
        dest.extra_counters[str(key)] = float(dest.extra_counters.get(str(key), 0.0)) + float(value)
    return dest


def _merge_gpu_runtime_telemetry_delta(
    dest: GpuRuntimeTelemetry | None,
    src: GpuRuntimeTelemetry,
    *,
    resident_store_hits_before: int,
    resident_store_misses_before: int,
    cpu_fallback_reason_counts_before: Mapping[str, int],
    extra_counters_before: Mapping[str, float],
) -> None:
    if dest is None:
        return
    hit_delta = int(src.resident_store_hits) - int(resident_store_hits_before)
    miss_delta = int(src.resident_store_misses) - int(resident_store_misses_before)
    if hit_delta > 0:
        dest.resident_store_hits += int(hit_delta)
    if miss_delta > 0:
        dest.resident_store_misses += int(miss_delta)
    for key, value in dict(src.cpu_fallback_reason_counts).items():
        delta = int(value) - int(cpu_fallback_reason_counts_before.get(str(key), 0))
        if delta > 0:
            dest.cpu_fallback_reason_counts[str(key)] = (
                int(dest.cpu_fallback_reason_counts.get(str(key), 0)) + int(delta)
            )
    for key, value in dict(src.extra_counters).items():
        delta = float(value) - float(extra_counters_before.get(str(key), 0.0))
        if delta > 0.0:
            dest.extra_counters[str(key)] = (
                float(dest.extra_counters.get(str(key), 0.0)) + float(delta)
            )


def _gpu_peak_memory_gb(device: torch.device | None) -> Tuple[float, float]:
    reserved = float("nan")
    allocated = float("nan")
    if device is None or str(device.type) != "cuda" or not torch.cuda.is_available():
        return reserved, allocated
    try:
        reserved = float(torch.cuda.max_memory_reserved(device=device) / float(1024 ** 3))
    except Exception:
        reserved = float("nan")
    try:
        allocated = float(torch.cuda.max_memory_allocated(device=device) / float(1024 ** 3))
    except Exception:
        allocated = float("nan")
    return reserved, allocated


def _docs_match_prefix(
    candidate_docs: Sequence[_FNOCountDoc],
    reference_docs: Sequence[_FNOCountDoc],
) -> bool:
    if len(candidate_docs) > len(reference_docs):
        return False
    return all(candidate_docs[idx] is reference_docs[idx] for idx in range(len(candidate_docs)))


def _ops_gpu_runtime_config(
    *,
    device: torch.device,
    mapping: Mapping[str, Any] | None = None,
    workers_per_mig_override: int | None = None,
) -> GpuRuntimeConfig:
    merged = dict(mapping or {})
    if workers_per_mig_override is not None and int(workers_per_mig_override) > 0:
        merged["workers_per_mig"] = int(workers_per_mig_override)
    return gpu_runtime_config_from_mapping(
        merged,
        device_type=str(device.type),
    )


def _gpu_runtime_config_from_ops_config(
    config: OPSCountConfig,
    *,
    device: torch.device,
) -> GpuRuntimeConfig:
    return _ops_gpu_runtime_config(
        device=device,
        mapping={
            "data_mode": str(getattr(config, "gpu_runtime_data_mode", "resident")),
            "bucket_mode": str(
                getattr(config, "gpu_runtime_bucket_mode", "exact_then_bucketed")
            ),
            "preload_splits": tuple(
                str(value)
                for value in getattr(
                    config,
                    "gpu_runtime_preload_splits",
                    ("train", "val", "test"),
                )
            ),
            "preload_targets": bool(
                getattr(config, "gpu_runtime_preload_targets", True)
            ),
            "workers_per_mig": int(
                getattr(config, "gpu_runtime_workers_per_mig", 1)
            ),
            "allow_multi_worker_screen": bool(
                getattr(config, "gpu_runtime_allow_multi_worker_screen", True)
            ),
            "capacity_workers_per_mig": int(
                getattr(config, "gpu_runtime_capacity_workers_per_mig", 2)
            ),
        },
    )


def _tree_topology_signature(doc: _FNOCountDoc) -> str:
    return (
        f"n{int(len(doc.leaf_token_ids))}:"
        f"leaf{tuple(int(v) for v in doc.leaf_token_lengths)}:"
        f"merge{tuple(int(v) for v in doc.merge_token_lengths)}"
    )


def _tree_leaf_count_auto_queue_enabled(bucket_mode: str) -> bool:
    return (
        str(normalize_gpu_runtime_bucket_mode(bucket_mode))
        == GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE
    )


def _effective_tree_bucket_mode(
    *,
    pack_mode: str,
    bucket_mode: str,
) -> str:
    normalized_bucket_mode = normalize_gpu_runtime_bucket_mode(bucket_mode)
    if (
        str(pack_mode or "").strip().lower() == "fixed_fused"
        and normalized_bucket_mode == GPU_RUNTIME_BUCKET_MODE_EXACT_THEN_BUCKETED
    ):
        return GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE
    return str(normalized_bucket_mode)


def _leaf_count_auto_queue_targets_for_docs(
    docs: Sequence[_FNOCountDoc],
    *,
    structural_pad_limit: float,
    min_docs: int,
) -> Dict[int, int]:
    counts_by_leaf: Dict[int, int] = {}
    for doc in docs:
        n_leaves = int(len(doc.leaf_token_ids))
        if n_leaves <= 0:
            continue
        counts_by_leaf[int(n_leaves)] = int(counts_by_leaf.get(int(n_leaves), 0)) + 1
    return build_leaf_count_auto_queue_targets(
        counts_by_leaf,
        structural_pad_limit=float(structural_pad_limit),
        min_docs=int(min_docs),
    )


def _tree_store_key_for_docs(
    docs: Sequence[_FNOCountDoc],
    *,
    backend_family: str,
    work_kind: str = "",
    supervision_mask: str = "",
    target_n_leaves: int | None = None,
    auto_queue_enabled: bool = False,
) -> GpuBatchStoreKey:
    reference_doc = docs[0]
    resolved_target_n_leaves = int(target_n_leaves or len(reference_doc.leaf_token_ids))
    exact_layout_signature = (
        f"auto_queue_target_n{int(resolved_target_n_leaves)}"
        if bool(auto_queue_enabled)
        else _tree_topology_signature(reference_doc)
    )
    max_leaf_tokens = max(
        (int(length) for doc in docs for length in doc.leaf_token_lengths),
        default=1,
    )
    return GpuBatchStoreKey(
        backend_family=str(backend_family),
        topology_signature=str(exact_layout_signature),
        leaf_count_band=int(resolved_target_n_leaves),
        max_leaf_tokens_band=int(_length_band(max_leaf_tokens)),
        work_kind=str(work_kind),
        supervision_mask=str(supervision_mask),
        exact_layout_signature=str(exact_layout_signature),
    )


def _full_doc_store_key(
    *,
    backend_family: str,
    seq_len: int,
) -> GpuBatchStoreKey:
    return GpuBatchStoreKey(
        backend_family=str(backend_family),
        topology_signature=f"full_doc_len_{int(seq_len)}",
        leaf_count_band=1,
        max_leaf_tokens_band=int(_length_band(max(1, int(seq_len)))),
        work_kind="full_doc",
        supervision_mask="root",
        exact_layout_signature=f"full_doc_len_{int(seq_len)}",
    )


def _group_tree_docs_for_store(
    docs: Sequence[_FNOCountDoc],
    *,
    bucket_mode: str,
    structural_pad_limit: float,
    auto_queue_min_docs: int,
) -> List[Tuple[str, int, bool, List[Tuple[int, _FNOCountDoc]]]]:
    grouped: Dict[Tuple[str, int, bool], List[Tuple[int, _FNOCountDoc]]] = {}
    auto_queue_targets = (
        _leaf_count_auto_queue_targets_for_docs(
            docs,
            structural_pad_limit=float(structural_pad_limit),
            min_docs=int(auto_queue_min_docs),
        )
        if _tree_leaf_count_auto_queue_enabled(bucket_mode)
        else {}
    )
    for doc_index, doc in enumerate(docs):
        n_leaves = int(len(doc.leaf_token_ids))
        if auto_queue_targets:
            target_n_leaves = int(auto_queue_targets.get(int(n_leaves), int(n_leaves)))
            group_key = (f"auto_queue_target_n{int(target_n_leaves)}", int(target_n_leaves), True)
        else:
            signature = _tree_topology_signature(doc)
            group_key = (str(signature), int(n_leaves), False)
        grouped.setdefault(group_key, []).append((int(doc_index), doc))
    return [
        (signature, int(target_n_leaves), bool(auto_queue_enabled), items)
        for (signature, target_n_leaves, auto_queue_enabled), items in sorted(
            grouped.items(),
            key=lambda pair: (pair[0][1], pair[0][0]),
        )
    ]


def _build_tree_gpu_batch_store(
    *,
    docs: Sequence[_FNOCountDoc],
    model: FNOCountSketch,
    device: torch.device,
    split_name: str,
    runtime_config: GpuRuntimeConfig,
    structural_pad_limit: float = 0.5,
    auto_queue_min_docs: int = 8,
) -> Tuple[GpuBatchStore | None, GpuRuntimeTelemetry]:
    telemetry = GpuRuntimeTelemetry(
        data_mode=str(runtime_config.data_mode),
        bucket_mode=str(runtime_config.bucket_mode),
        workers_per_mig=int(runtime_config.workers_per_mig),
    )
    if not docs:
        telemetry.add_store_miss(reason="empty_split")
        return None, telemetry
    if str(device.type) != "cuda" or not torch.cuda.is_available():
        telemetry.add_store_miss(reason="non_cuda_device")
        return None, telemetry
    if not bool(runtime_config.is_resident and runtime_config.should_preload_split(split_name)):
        telemetry.add_store_miss(reason="runtime_split_not_preloaded")
        return None, telemetry

    started = time.perf_counter()
    store = GpuBatchStore(
        backend_family="neural_tree",
        split_name=str(split_name),
        config=runtime_config,
        device=str(device),
        telemetry=telemetry,
    )
    total_bytes = 0
    auto_queue_target_leaf_counts: List[int] = []
    for _signature, target_n_leaves, auto_queue_enabled, entries in _group_tree_docs_for_store(
        docs,
        bucket_mode=str(runtime_config.bucket_mode),
        structural_pad_limit=float(structural_pad_limit),
        auto_queue_min_docs=int(auto_queue_min_docs),
    ):
        bucket_docs = [doc for _doc_index, doc in entries]
        bucket_doc_indices = [int(doc_index) for doc_index, _doc in entries]
        reference = bucket_docs[0]
        resolved_target_n_leaves = int(target_n_leaves or len(reference.leaf_token_ids))
        leaf_tokens_cpu, leaf_mask_cpu, leaf_valid_mask_cpu = _build_dense_leaf_token_tensors_from_docs(
            bucket_docs,
            pad_id=int(model.pad_id),
            target_n_leaves=int(resolved_target_n_leaves),
        )
        max_leaf_len = int(leaf_tokens_cpu.shape[-1])
        if device.type == "cuda":
            leaf_tokens_cpu = leaf_tokens_cpu.pin_memory()
            leaf_mask_cpu = leaf_mask_cpu.pin_memory()
            leaf_valid_mask_cpu = leaf_valid_mask_cpu.pin_memory()
        tensors: Dict[str, torch.Tensor] = {
            "leaf_tokens": leaf_tokens_cpu.to(device=device, non_blocking=True),
            "leaf_mask": leaf_mask_cpu.to(device=device, non_blocking=True),
            "leaf_valid_mask": leaf_valid_mask_cpu.to(device=device, non_blocking=True),
            "root_targets": torch.as_tensor(
                [float(doc.root_count) for doc in bucket_docs],
                dtype=torch.float32,
                device=device,
            ),
        }
        if bool(runtime_config.preload_targets):
            cached_targets = tuple(
                _cached_fused_doc_targets_for_target_leaves(
                    doc,
                    int(resolved_target_n_leaves),
                )
                for doc in bucket_docs
            )

            def _stack_cached(attr_name: str, *, dtype: torch.dtype) -> torch.Tensor:
                stacked = torch.stack(
                    [getattr(targets, attr_name) for targets in cached_targets],
                    dim=0,
                )
                if device.type == "cuda":
                    stacked = stacked.pin_memory()
                return stacked.to(device=device, dtype=dtype, non_blocking=bool(device.type == "cuda"))

            tensors.update(
                {
                    "leaf_count_targets": _stack_cached(
                        "leaf_count_targets_cpu",
                        dtype=torch.float32,
                    ),
                    "leaf_first_targets": _stack_cached(
                        "leaf_first_targets_cpu",
                        dtype=torch.long,
                    ),
                    "leaf_last_targets": _stack_cached(
                        "leaf_last_targets_cpu",
                        dtype=torch.long,
                    ),
                    "leaf_valid_mask_targets": _stack_cached(
                        "leaf_valid_mask_cpu",
                        dtype=torch.bool,
                    ),
                    "merge_count_targets": _stack_cached(
                        "merge_count_targets_cpu",
                        dtype=torch.float32,
                    ),
                    "merge_first_targets": _stack_cached(
                        "merge_first_targets_cpu",
                        dtype=torch.long,
                    ),
                    "merge_last_targets": _stack_cached(
                        "merge_last_targets_cpu",
                        dtype=torch.long,
                    ),
                    "merge_valid_mask": _stack_cached(
                        "merge_valid_mask_cpu",
                        dtype=torch.bool,
                    ),
                    "node_count_targets": _stack_cached(
                        "node_count_targets_cpu",
                        dtype=torch.float32,
                    ),
                    "node_count_keys": _stack_cached(
                        "node_count_keys_cpu",
                        dtype=torch.long,
                    ),
                    "node_first_targets": _stack_cached(
                        "node_first_targets_cpu",
                        dtype=torch.long,
                    ),
                    "node_last_targets": _stack_cached(
                        "node_last_targets_cpu",
                        dtype=torch.long,
                    ),
                    "node_valid_mask": _stack_cached(
                        "node_valid_mask_cpu",
                        dtype=torch.bool,
                    ),
                }
            )
        bucket_bytes = 0
        for tensor in tensors.values():
            tensor_bytes = _tensor_nbytes(tensor)
            bucket_bytes += int(tensor_bytes)
            total_bytes += int(tensor_bytes)
        telemetry.add_extra_counter("fixed_shape_bucket_store_count", 1.0)
        telemetry.add_extra_counter("fixed_shape_dense_bucket_store_count", 1.0)
        telemetry.add_extra_counter("fixed_shape_dense_bucket_store_bytes", float(bucket_bytes))
        if bool(auto_queue_enabled):
            telemetry.add_extra_counter("auto_queue_family_count", 1.0)
            auto_queue_target_leaf_counts.append(int(resolved_target_n_leaves))
        store.add_bucket(
            key=_tree_store_key_for_docs(
                bucket_docs,
                backend_family="neural_tree",
                target_n_leaves=int(resolved_target_n_leaves),
                auto_queue_enabled=bool(auto_queue_enabled),
            ),
            doc_indices=bucket_doc_indices,
            tensors=tensors,
            metadata={
                "bucket_store_mode": "dense_resident",
                "n_leaves": int(resolved_target_n_leaves),
                "actual_n_leaves_min": int(min(len(doc.leaf_token_ids) for doc in bucket_docs)),
                "actual_n_leaves_max": int(max(len(doc.leaf_token_ids) for doc in bucket_docs)),
                "max_leaf_tokens": int(max_leaf_len),
                "resident_bucket_bytes": int(bucket_bytes),
                "resident_layout_mode": (
                    "dense_fixed_shape_auto_queue"
                    if bool(auto_queue_enabled)
                    else "dense_fixed_shape"
                ),
                "topology_signature": (
                    f"auto_queue_target_n{int(resolved_target_n_leaves)}"
                    if bool(auto_queue_enabled)
                    else str(_tree_topology_signature(reference))
                ),
                "auto_queue_enabled": bool(auto_queue_enabled),
                "auto_queue_target_n_leaves": int(resolved_target_n_leaves),
            },
        )
    telemetry.add_store_build(wall_time_s=time.perf_counter() - started)
    telemetry.add_extra_counter("resident_store_bytes", float(total_bytes))
    if auto_queue_target_leaf_counts:
        telemetry.add_extra_counter(
            "auto_queue_target_leaf_counts_count",
            float(len(set(auto_queue_target_leaf_counts))),
        )
    return store, telemetry


def estimate_tree_gpu_batch_store_bytes(
    *,
    docs: Sequence[_FNOCountDoc],
    runtime_config: GpuRuntimeConfig,
    split_name: str,
    structural_pad_limit: float = 0.5,
    auto_queue_min_docs: int = 8,
    pad_id: int = 0,
) -> Dict[str, Any]:
    split = str(split_name)
    summary: Dict[str, Any] = {
        "split_name": split,
        "preloaded": False,
        "preload_targets": bool(runtime_config.preload_targets),
        "resident_store_bytes": 0,
        "bucket_count": 0,
        "bucket_layouts": [],
    }
    if not docs:
        summary["reason"] = "empty_split"
        return summary
    if not bool(runtime_config.is_resident and runtime_config.should_preload_split(split)):
        summary["reason"] = "runtime_split_not_preloaded"
        return summary

    bucket_layouts: List[Dict[str, Any]] = []
    total_bytes = 0
    for signature, target_n_leaves, auto_queue_enabled, entries in _group_tree_docs_for_store(
        docs,
        bucket_mode=str(runtime_config.bucket_mode),
        structural_pad_limit=float(structural_pad_limit),
        auto_queue_min_docs=int(auto_queue_min_docs),
    ):
        bucket_docs = [doc for _doc_index, doc in entries]
        resolved_target_n_leaves = int(
            target_n_leaves or len(bucket_docs[0].leaf_token_ids)
        )
        leaf_tokens_cpu, leaf_mask_cpu, leaf_valid_mask_cpu = _build_dense_leaf_token_tensors_from_docs(
            bucket_docs,
            pad_id=int(pad_id),
            target_n_leaves=int(resolved_target_n_leaves),
        )
        tensors: Dict[str, torch.Tensor] = {
            "leaf_tokens": leaf_tokens_cpu,
            "leaf_mask": leaf_mask_cpu,
            "leaf_valid_mask": leaf_valid_mask_cpu,
            "root_targets": torch.as_tensor(
                [float(doc.root_count) for doc in bucket_docs],
                dtype=torch.float32,
            ),
        }
        if bool(runtime_config.preload_targets):
            cached_targets = tuple(
                _cached_fused_doc_targets_for_target_leaves(
                    doc,
                    int(resolved_target_n_leaves),
                )
                for doc in bucket_docs
            )
            tensors.update(
                {
                    "leaf_count_targets": torch.stack(
                        [targets.leaf_count_targets_cpu for targets in cached_targets],
                        dim=0,
                    ),
                    "leaf_first_targets": torch.stack(
                        [targets.leaf_first_targets_cpu for targets in cached_targets],
                        dim=0,
                    ),
                    "leaf_last_targets": torch.stack(
                        [targets.leaf_last_targets_cpu for targets in cached_targets],
                        dim=0,
                    ),
                    "leaf_valid_mask_targets": torch.stack(
                        [targets.leaf_valid_mask_cpu for targets in cached_targets],
                        dim=0,
                    ),
                    "merge_count_targets": torch.stack(
                        [targets.merge_count_targets_cpu for targets in cached_targets],
                        dim=0,
                    ),
                    "merge_first_targets": torch.stack(
                        [targets.merge_first_targets_cpu for targets in cached_targets],
                        dim=0,
                    ),
                    "merge_last_targets": torch.stack(
                        [targets.merge_last_targets_cpu for targets in cached_targets],
                        dim=0,
                    ),
                    "merge_valid_mask": torch.stack(
                        [targets.merge_valid_mask_cpu for targets in cached_targets],
                        dim=0,
                    ),
                    "node_count_targets": torch.stack(
                        [targets.node_count_targets_cpu for targets in cached_targets],
                        dim=0,
                    ),
                    "node_count_keys": torch.stack(
                        [targets.node_count_keys_cpu for targets in cached_targets],
                        dim=0,
                    ),
                    "node_first_targets": torch.stack(
                        [targets.node_first_targets_cpu for targets in cached_targets],
                        dim=0,
                    ),
                    "node_last_targets": torch.stack(
                        [targets.node_last_targets_cpu for targets in cached_targets],
                        dim=0,
                    ),
                    "node_valid_mask": torch.stack(
                        [targets.node_valid_mask_cpu for targets in cached_targets],
                        dim=0,
                    ),
                }
            )
        bucket_bytes = sum(_tensor_nbytes(tensor) for tensor in tensors.values())
        total_bytes += int(bucket_bytes)
        bucket_layouts.append(
            {
                "topology_signature": str(signature),
                "doc_count": int(len(bucket_docs)),
                "target_n_leaves": int(resolved_target_n_leaves),
                "auto_queue_enabled": bool(auto_queue_enabled),
                "resident_bucket_bytes": int(bucket_bytes),
                "max_leaf_tokens": int(leaf_tokens_cpu.shape[-1]),
                "tensor_names": sorted(str(name) for name in tensors),
            }
        )
    summary["preloaded"] = True
    summary["resident_store_bytes"] = int(total_bytes)
    summary["bucket_count"] = int(len(bucket_layouts))
    summary["bucket_layouts"] = list(bucket_layouts)
    return summary


def _build_full_doc_gpu_batch_store(
    *,
    tokens: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    split_name: str,
    runtime_config: GpuRuntimeConfig,
) -> Tuple[GpuBatchStore | None, GpuRuntimeTelemetry]:
    telemetry = GpuRuntimeTelemetry(
        data_mode=str(runtime_config.data_mode),
        bucket_mode=str(runtime_config.bucket_mode),
        workers_per_mig=int(runtime_config.workers_per_mig),
    )
    if int(tokens.shape[0]) <= 0:
        telemetry.add_store_miss(reason="empty_split")
        return None, telemetry
    if str(device.type) != "cuda" or not torch.cuda.is_available():
        telemetry.add_store_miss(reason="non_cuda_device")
        return None, telemetry
    if not bool(runtime_config.is_resident and runtime_config.should_preload_split(split_name)):
        telemetry.add_store_miss(reason="runtime_split_not_preloaded")
        return None, telemetry

    started = time.perf_counter()
    store = GpuBatchStore(
        backend_family="full_doc_fno",
        split_name=str(split_name),
        config=runtime_config,
        device=str(device),
        telemetry=telemetry,
    )
    tensors = {
        "tokens": torch.as_tensor(tokens, dtype=torch.long, device=device),
        "mask": torch.as_tensor(mask, dtype=torch.float32, device=device),
        "targets": torch.as_tensor(targets, dtype=torch.float32, device=device),
    }
    total_bytes = sum(_tensor_nbytes(tensor) for tensor in tensors.values())
    store.add_bucket(
        key=_full_doc_store_key(
            backend_family="full_doc_fno",
            seq_len=int(tokens.shape[1]) if tokens.ndim >= 2 else 1,
        ),
        doc_indices=list(range(int(tokens.shape[0]))),
        tensors=tensors,
        metadata={
            "seq_len": int(tokens.shape[1]) if tokens.ndim >= 2 else 1,
        },
    )
    telemetry.add_store_build(wall_time_s=time.perf_counter() - started)
    telemetry.add_extra_counter("resident_store_bytes", float(total_bytes))
    return store, telemetry


def _tree_store_view_for_items(
    store: GpuBatchStore | None,
    items: Sequence[_TreeWorkItem],
    *,
    model: FNOCountSketch,
    runtime_telemetry: GpuRuntimeTelemetry | None = None,
) -> GpuBatchView | None:
    if store is None:
        return None
    resident_store_hits_before = int(store.telemetry.resident_store_hits)
    resident_store_misses_before = int(store.telemetry.resident_store_misses)
    cpu_fallback_reason_counts_before = dict(store.telemetry.cpu_fallback_reason_counts)
    extra_counters_before = dict(store.telemetry.extra_counters)
    view = store.view_for_doc_indices(
        [int(item.doc_index) for item in items],
        pad_values={"leaf_tokens": float(int(model.pad_id))},
    )
    if view is not None and str(view.metadata.get("bucket_store_mode", "")).strip() == "dense_resident":
        store.telemetry.add_extra_counter("fixed_shape_dense_bucket_store_hits", 1.0)
        if bool(view.metadata.get("auto_queue_enabled", False)):
            store.telemetry.add_extra_counter("auto_queue_fused_batches", 1.0)
    _merge_gpu_runtime_telemetry_delta(
        runtime_telemetry,
        store.telemetry,
        resident_store_hits_before=resident_store_hits_before,
        resident_store_misses_before=resident_store_misses_before,
        cpu_fallback_reason_counts_before=cpu_fallback_reason_counts_before,
        extra_counters_before=extra_counters_before,
    )
    return view


@dataclass(frozen=True)
class _TreeWorkItem:
    doc_index: int
    doc: _FNOCountDoc
    work_kind: str
    collect_leaf: bool
    collect_c2: bool
    collect_c3: bool
    root_only_supervision: bool
    doc_sequence_supervision: bool
    leaf_audit_indices: Optional[Tuple[int, ...]] = None
    c3_audit_indices: Optional[Tuple[int, ...]] = None
    document_mode: str = ""

    @property
    def n_leaves(self) -> int:
        return int(len(self.doc.leaf_token_ids))

    @property
    def total_leaf_tokens(self) -> int:
        return int(sum(int(length) for length in self.doc.leaf_token_lengths))

    @property
    def total_nodes(self) -> int:
        n_leaves = int(len(self.doc.leaf_token_ids))
        return int(n_leaves + max(0, n_leaves - 1))

    @property
    def total_merge_ops(self) -> int:
        return int(max(0, len(self.doc.merge_counts_balanced)))


@dataclass(frozen=True)
class _BatchBucketKey:
    n_leaves: int
    work_kind: str
    collect_leaf: bool
    collect_c2: bool
    collect_c3: bool
    max_leaf_tokens_band: int
    max_merge_tokens_band: int
    irregular_leaf_layout: bool
    auto_queue_enabled: bool = False


@dataclass(frozen=True)
class _PackedTreeBatch:
    bucket_key: _BatchBucketKey
    items: Tuple[_TreeWorkItem, ...]
    actual_leaf_tokens: int
    padded_leaf_tokens: int
    total_nodes: int
    total_merge_ops: int
    actual_leaf_slots: int
    padded_leaf_slots: int


@dataclass
class _BatchingMetricsAccumulator:
    runtime_mode: str = field(
        default_factory=lambda: resolve_runtime_mode(None, env_var="TT_TREE_BATCH_RUNTIME_MODE")
    )
    n_batches: int = 0
    total_docs: int = 0
    total_leaf_tokens: int = 0
    total_padded_leaf_tokens: int = 0
    total_leaf_slots: int = 0
    total_padded_leaf_slots: int = 0
    total_nodes: int = 0
    total_merge_ops: int = 0
    total_bucket_utilization: float = 0.0
    train_forward_time_s: float = 0.0
    train_backward_time_s: float = 0.0
    eval_time_s: float = 0.0
    idle_wait_time_s: float = 0.0
    fallback_reason_counts: Dict[str, int] = field(default_factory=dict)
    autotune_heuristic_time_s: float = 0.0
    autotune_train_probe_time_s: float = 0.0
    autotune_eval_probe_time_s: float = 0.0
    autotune_cache_lookup_time_s: float = 0.0
    autotune_cache_write_time_s: float = 0.0
    autotune_cache_hits: int = 0
    autotune_cache_misses: int = 0
    autotune_cache_writes: int = 0
    autotune_probe_runs: int = 0
    autotune_probe_candidate_evals: int = 0

    def add_batch(
        self,
        batch: _PackedTreeBatch,
        *,
        token_budget: int,
        node_budget: int,
        max_docs_budget: int,
        fallback_reason: str = "",
    ) -> None:
        self.n_batches += 1
        self.total_docs += int(len(batch.items))
        self.total_leaf_tokens += int(batch.actual_leaf_tokens)
        self.total_padded_leaf_tokens += int(batch.padded_leaf_tokens)
        self.total_leaf_slots += int(batch.actual_leaf_slots)
        self.total_padded_leaf_slots += int(batch.padded_leaf_slots)
        self.total_nodes += int(batch.total_nodes)
        self.total_merge_ops += int(batch.total_merge_ops)
        utilization_terms: List[float] = []
        if int(token_budget) > 0:
            utilization_terms.append(
                min(1.0, float(batch.actual_leaf_tokens) / float(max(1, int(token_budget))))
            )
        if int(node_budget) > 0:
            utilization_terms.append(
                min(1.0, float(batch.total_nodes) / float(max(1, int(node_budget))))
            )
        if int(max_docs_budget) > 0:
            utilization_terms.append(
                min(1.0, float(len(batch.items)) / float(max(1, int(max_docs_budget))))
            )
        if utilization_terms:
            self.total_bucket_utilization += float(
                sum(utilization_terms) / float(len(utilization_terms))
            )
        if str(fallback_reason or "").strip():
            self.fallback_reason_counts[str(fallback_reason)] = (
                int(self.fallback_reason_counts.get(str(fallback_reason), 0)) + 1
            )

    def as_dict(
        self,
        *,
        device: torch.device | None,
        runtime_telemetry: GpuRuntimeTelemetry | None = None,
    ) -> Dict[str, float]:
        mean_docs = (
            float(self.total_docs) / float(self.n_batches)
            if self.n_batches > 0
            else 0.0
        )
        mean_leaf_tokens = (
            float(self.total_leaf_tokens) / float(self.n_batches)
            if self.n_batches > 0
            else 0.0
        )
        mean_nodes = (
            float(self.total_nodes) / float(self.n_batches)
            if self.n_batches > 0
            else 0.0
        )
        padded_total = max(1, int(self.total_padded_leaf_tokens))
        padding_waste_ratio = (
            float(max(0, self.total_padded_leaf_tokens - self.total_leaf_tokens))
            / float(padded_total)
            if self.total_padded_leaf_tokens > 0
            else 0.0
        )
        padded_leaf_slots_total = max(1, int(self.total_padded_leaf_slots))
        structural_padding_waste_ratio = (
            float(max(0, self.total_padded_leaf_slots - self.total_leaf_slots))
            / float(padded_leaf_slots_total)
            if self.total_padded_leaf_slots > 0
            else 0.0
        )
        gpu_reserved_peak, gpu_allocated_peak = _gpu_peak_memory_gb(device)
        payload: Dict[str, Any] = {
            "runtime_mode": str(self.runtime_mode),
            "mean_docs_per_batch": float(mean_docs),
            "mean_leaf_tokens_per_batch": float(mean_leaf_tokens),
            "mean_nodes_per_batch": float(mean_nodes),
            "padding_waste_ratio": float(padding_waste_ratio),
            "structural_padding_waste_ratio": float(structural_padding_waste_ratio),
            "bucket_utilization_rate": (
                float(self.total_bucket_utilization) / float(self.n_batches)
                if self.n_batches > 0
                else 0.0
            ),
            "gpu_reserved_mem_peak_gb": float(gpu_reserved_peak),
            "train_forward_time_s": float(self.train_forward_time_s),
            "train_backward_time_s": float(self.train_backward_time_s),
            "eval_time_s": float(self.eval_time_s),
            "idle_wait_time_s": float(self.idle_wait_time_s),
            "fallback_reason_counts": dict(self.fallback_reason_counts),
            "autotune_heuristic_time_s": float(self.autotune_heuristic_time_s),
            "autotune_train_probe_time_s": float(self.autotune_train_probe_time_s),
            "autotune_eval_probe_time_s": float(self.autotune_eval_probe_time_s),
            "autotune_cache_lookup_time_s": float(self.autotune_cache_lookup_time_s),
            "autotune_cache_write_time_s": float(self.autotune_cache_write_time_s),
            "autotune_cache_hits": int(self.autotune_cache_hits),
            "autotune_cache_misses": int(self.autotune_cache_misses),
            "autotune_cache_writes": int(self.autotune_cache_writes),
            "autotune_probe_runs": int(self.autotune_probe_runs),
            "autotune_probe_candidate_evals": int(self.autotune_probe_candidate_evals),
            "gpu_allocated_mem_peak_gb": float(gpu_allocated_peak),
        }
        if runtime_telemetry is not None:
            payload.update(
                runtime_telemetry.as_dict(
                    gpu_reserved_peak_gb=gpu_reserved_peak,
                    gpu_allocated_peak_gb=gpu_allocated_peak,
                )
            )
        return payload


@dataclass(frozen=True)
class _AutotuneProbeDiagnostics:
    heuristic_time_s: float = 0.0
    train_probe_time_s: float = 0.0
    eval_probe_time_s: float = 0.0
    cache_lookup_time_s: float = 0.0
    cache_write_time_s: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    probe_runs: int = 0
    probe_candidate_evals: int = 0
    probe_profiles: Tuple[ProbeRunProfile, ...] = tuple()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "profile_version": int(AUTOTUNE_PROBE_CACHE_VERSION),
            "heuristic_time_s": float(self.heuristic_time_s),
            "train_probe_time_s": float(self.train_probe_time_s),
            "eval_probe_time_s": float(self.eval_probe_time_s),
            "cache_lookup_time_s": float(self.cache_lookup_time_s),
            "cache_write_time_s": float(self.cache_write_time_s),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "cache_writes": int(self.cache_writes),
            "probe_run_count": int(self.probe_runs),
            "probe_candidate_count": int(self.probe_candidate_evals),
            "runs": [profile.as_dict() for profile in self.probe_profiles],
        }


@dataclass(frozen=True)
class _ProbeRunOutcome:
    selected_docs_cap: int
    run_profile: ProbeRunProfile
    cache_hit: bool = False
    cache_lookup_time_s: float = 0.0
    cache_write_time_s: float = 0.0
    candidate_evaluations: int = 0


@dataclass(frozen=True)
class _AutotunedTreeBatchBudgets:
    train_leaf_token_budget: int
    train_node_budget: int
    eval_leaf_token_budget: int
    eval_node_budget: int
    eval_workers_per_mig: int
    train_bucket_max_docs_by_n_leaves: Tuple[Tuple[int, int], ...] = tuple()
    eval_bucket_max_docs_by_n_leaves: Tuple[Tuple[int, int], ...] = tuple()
    probe_diagnostics: _AutotuneProbeDiagnostics = _AutotuneProbeDiagnostics()

    def train_docs_cap_for_leaves(self, n_leaves: int) -> int:
        for key_n_leaves, docs_cap in self.train_bucket_max_docs_by_n_leaves:
            if int(key_n_leaves) == int(n_leaves):
                return int(docs_cap)
        return 0

    def eval_docs_cap_for_leaves(self, n_leaves: int) -> int:
        for key_n_leaves, docs_cap in self.eval_bucket_max_docs_by_n_leaves:
            if int(key_n_leaves) == int(n_leaves):
                return int(docs_cap)
        return 0


def _merge_autotune_probe_diagnostics(
    *diagnostics: _AutotuneProbeDiagnostics,
) -> _AutotuneProbeDiagnostics:
    heuristic_time_s = 0.0
    train_probe_time_s = 0.0
    eval_probe_time_s = 0.0
    cache_lookup_time_s = 0.0
    cache_write_time_s = 0.0
    cache_hits = 0
    cache_misses = 0
    cache_writes = 0
    probe_runs = 0
    probe_candidate_evals = 0
    probe_profiles: List[ProbeRunProfile] = []
    for item in diagnostics:
        heuristic_time_s += float(item.heuristic_time_s)
        train_probe_time_s += float(item.train_probe_time_s)
        eval_probe_time_s += float(item.eval_probe_time_s)
        cache_lookup_time_s += float(item.cache_lookup_time_s)
        cache_write_time_s += float(item.cache_write_time_s)
        cache_hits += int(item.cache_hits)
        cache_misses += int(item.cache_misses)
        cache_writes += int(item.cache_writes)
        probe_runs += int(item.probe_runs)
        probe_candidate_evals += int(item.probe_candidate_evals)
        probe_profiles.extend(list(item.probe_profiles))
    return _AutotuneProbeDiagnostics(
        heuristic_time_s=float(heuristic_time_s),
        train_probe_time_s=float(train_probe_time_s),
        eval_probe_time_s=float(eval_probe_time_s),
        cache_lookup_time_s=float(cache_lookup_time_s),
        cache_write_time_s=float(cache_write_time_s),
        cache_hits=int(cache_hits),
        cache_misses=int(cache_misses),
        cache_writes=int(cache_writes),
        probe_runs=int(probe_runs),
        probe_candidate_evals=int(probe_candidate_evals),
        probe_profiles=tuple(probe_profiles),
    )


def _apply_autotune_probe_diagnostics(
    batching_metrics: _BatchingMetricsAccumulator,
    diagnostics: _AutotuneProbeDiagnostics,
) -> None:
    batching_metrics.autotune_heuristic_time_s += float(diagnostics.heuristic_time_s)
    batching_metrics.autotune_train_probe_time_s += float(diagnostics.train_probe_time_s)
    batching_metrics.autotune_eval_probe_time_s += float(diagnostics.eval_probe_time_s)
    batching_metrics.autotune_cache_lookup_time_s += float(diagnostics.cache_lookup_time_s)
    batching_metrics.autotune_cache_write_time_s += float(diagnostics.cache_write_time_s)
    batching_metrics.autotune_cache_hits += int(diagnostics.cache_hits)
    batching_metrics.autotune_cache_misses += int(diagnostics.cache_misses)
    batching_metrics.autotune_cache_writes += int(diagnostics.cache_writes)
    batching_metrics.autotune_probe_runs += int(diagnostics.probe_runs)
    batching_metrics.autotune_probe_candidate_evals += int(
        diagnostics.probe_candidate_evals
    )


def _merge_autotune_probe_profile_dicts(
    *profiles: Mapping[str, Any],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "profile_version": int(AUTOTUNE_PROBE_CACHE_VERSION),
        "heuristic_time_s": 0.0,
        "train_probe_time_s": 0.0,
        "eval_probe_time_s": 0.0,
        "cache_lookup_time_s": 0.0,
        "cache_write_time_s": 0.0,
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_writes": 0,
        "probe_run_count": 0,
        "probe_candidate_count": 0,
        "runs": [],
    }
    for profile in profiles:
        payload = dict(profile or {})
        if not payload:
            continue
        merged["heuristic_time_s"] += float(payload.get("heuristic_time_s", 0.0) or 0.0)
        merged["train_probe_time_s"] += float(
            payload.get("train_probe_time_s", 0.0) or 0.0
        )
        merged["eval_probe_time_s"] += float(
            payload.get("eval_probe_time_s", 0.0) or 0.0
        )
        merged["cache_lookup_time_s"] += float(
            payload.get("cache_lookup_time_s", 0.0) or 0.0
        )
        merged["cache_write_time_s"] += float(
            payload.get("cache_write_time_s", 0.0) or 0.0
        )
        merged["cache_hits"] += int(payload.get("cache_hits", 0) or 0)
        merged["cache_misses"] += int(payload.get("cache_misses", 0) or 0)
        merged["cache_writes"] += int(payload.get("cache_writes", 0) or 0)
        merged["probe_run_count"] += int(payload.get("probe_run_count", 0) or 0)
        merged["probe_candidate_count"] += int(
            payload.get("probe_candidate_count", 0) or 0
        )
        merged["runs"].extend(list(payload.get("runs", ()) or ()))
    return merged


def _tree_batch_probe_topology_signature(doc: _FNOCountDoc) -> str:
    return (
        f"n{int(len(doc.leaf_token_ids))}:"
        f"leaf{tuple(int(v) for v in doc.leaf_token_lengths)}:"
        f"merge{tuple(int(v) for v in doc.merge_token_lengths)}"
    )


def _tree_batch_probe_model_signature(model: FNOCountSketch) -> Dict[str, Any]:
    return {
        "model_class": type(model).__name__,
        "state_dim": int(getattr(model, "state_dim", 0) or 0),
        "hidden_dim": int(getattr(model, "hidden_dim", 0) or 0),
        "leaf_tokens": int(getattr(model, "leaf_tokens", 0) or 0),
        "fno_width": int(getattr(model, "fno_width", 0) or 0),
        "fno_n_modes": int(getattr(model, "fno_n_modes", 0) or 0),
        "fno_n_layers": int(getattr(model, "fno_n_layers", 0) or 0),
        "summary_spec_name": str(getattr(model, "summary_spec_name", "") or ""),
        "slot_count": int(getattr(model, "slot_count", 0) or 0),
        "task_head_mode": str(getattr(model, "task_head_mode", "") or ""),
        "theorem_surface_mode": str(
            getattr(model, "theorem_surface_mode", "") or ""
        ),
        "summary_spec_root_mode": str(
            getattr(model, "summary_spec_root_mode", "") or ""
        ),
        "tree_model_version": str(
            getattr(model, "tree_model_version", "") or ""
        ),
        "theorem_feature_dim": int(getattr(model, "theorem_feature_dim", 0) or 0),
        "theorem_feature_hidden_dim": int(
            getattr(model, "theorem_feature_hidden_dim", 0) or 0
        ),
        "theorem_score_dim": int(getattr(model, "theorem_score_dim", 0) or 0),
        "theorem_fiber_dim": int(getattr(model, "theorem_fiber_dim", 0) or 0),
        "use_shared_theorem_surface": bool(
            getattr(model, "use_shared_theorem_surface", False)
        ),
        "use_factorized_score_fiber_surface": bool(
            getattr(model, "use_factorized_score_fiber_surface", False)
        ),
        "use_summary_spec": bool(getattr(model, "use_summary_spec", False)),
    }


def _tree_batch_probe_device_signature(device: torch.device) -> Dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "device_name": str(device.type),
            "total_memory_bytes": 0,
            "compute_capability": tuple(),
            "is_mig": False,
            "mig_profile": "cpu",
        }
    props = torch.cuda.get_device_properties(device)
    capability = (
        int(getattr(props, "major", 0) or 0),
        int(getattr(props, "minor", 0) or 0),
    )
    return classify_device_signature(
        device_name=str(getattr(props, "name", "") or ""),
        total_memory_bytes=int(getattr(props, "total_memory", 0) or 0),
        capability=capability,
    )


def _prepare_fno_count_docs(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    leaf_tokens: int,
) -> Tuple[_FNOCountDoc, ...]:
    """Prepare docs with raw token spans for FNO leaf encoding."""
    from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (
        _leaf_spans,
        _oracle_count,
    )

    out: List[_FNOCountDoc] = []
    for doc in docs:
        n_tok = int(len(doc.tokens))
        spans = _leaf_spans(n_tok, leaf_tokens=int(leaf_tokens))
        leaf_token_ids = tuple(
            tuple(int(doc.tokens[t]) for t in range(sp[0], sp[1]))
            for sp in spans
        )
        leaf_token_lengths = tuple(int(sp[1] - sp[0]) for sp in spans)
        leaf_counts = tuple(
            float(_oracle_count(doc, start=sp[0], end=sp[1])) for sp in spans
        )
        leaf_first_regimes = tuple(
            int(doc.token_regimes[sp[0]]) for sp in spans
        )
        leaf_last_regimes = tuple(
            int(doc.token_regimes[sp[1] - 1]) for sp in spans
        )
        # Balanced merge labels
        cur_spans = list(spans)
        cur_sizes = [1 for _ in spans]
        merge_counts: List[float] = []
        merge_sizes: List[int] = []
        merge_token_lengths: List[int] = []
        while len(cur_spans) > 1:
            nxt_spans: List[Tuple[int, int]] = []
            nxt_sizes: List[int] = []
            i = 0
            while i < len(cur_spans):
                if i + 1 >= len(cur_spans):
                    nxt_spans.append(cur_spans[i])
                    nxt_sizes.append(int(cur_sizes[i]))
                    i += 1
                    continue
                parent = (int(cur_spans[i][0]), int(cur_spans[i + 1][1]))
                parent_size = int(cur_sizes[i]) + int(cur_sizes[i + 1])
                merge_counts.append(float(_oracle_count(doc, start=parent[0], end=parent[1])))
                merge_sizes.append(int(parent_size))
                merge_token_lengths.append(int(parent[1] - parent[0]))
                nxt_spans.append(parent)
                nxt_sizes.append(int(parent_size))
                i += 2
            cur_spans = nxt_spans
            cur_sizes = nxt_sizes
        root_count = float(_oracle_count(doc, start=0, end=n_tok))
        out.append(_FNOCountDoc(
            n_tokens=n_tok,
            leaf_token_ids=leaf_token_ids,
            leaf_counts=leaf_counts,
            leaf_first_regimes=leaf_first_regimes,
            leaf_last_regimes=leaf_last_regimes,
            leaf_token_lengths=leaf_token_lengths,
            merge_counts_balanced=tuple(merge_counts),
            merge_sizes_balanced=tuple(merge_sizes),
            merge_token_lengths=tuple(merge_token_lengths),
            root_count=root_count,
        ))
    return tuple(out)


def _balanced_merge_token_lengths_from_leaf_lengths(
    leaf_token_lengths: Sequence[int],
) -> Tuple[int, ...]:
    cur = [int(length) for length in leaf_token_lengths]
    merge_token_lengths: List[int] = []
    while len(cur) > 1:
        nxt: List[int] = []
        i = 0
        while i < len(cur):
            if i + 1 >= len(cur):
                nxt.append(int(cur[i]))
                i += 1
                continue
            merged = int(cur[i]) + int(cur[i + 1])
            merge_token_lengths.append(int(merged))
            nxt.append(int(merged))
            i += 2
        cur = nxt
    return tuple(int(value) for value in merge_token_lengths)


def _aggregate_fno_doc_from_leaf_range(
    doc: _FNOCountDoc,
    *,
    start_leaf_idx: int,
    end_leaf_idx: int,
) -> _FNOCountDoc:
    start = max(0, int(start_leaf_idx))
    end = min(int(len(doc.leaf_token_ids)), int(end_leaf_idx))
    if end <= start:
        raise ValueError("aggregate leaf range must contain at least one leaf")
    leaf_token_ids = tuple(doc.leaf_token_ids[start:end])
    leaf_counts = tuple(float(value) for value in doc.leaf_counts[start:end])
    leaf_first_regimes = tuple(int(value) for value in doc.leaf_first_regimes[start:end])
    leaf_last_regimes = tuple(int(value) for value in doc.leaf_last_regimes[start:end])
    leaf_token_lengths = tuple(int(value) for value in doc.leaf_token_lengths[start:end])
    exact_targets = _balanced_exact_sketch_targets(
        leaf_counts=leaf_counts,
        leaf_first_regimes=leaf_first_regimes,
        leaf_last_regimes=leaf_last_regimes,
    )
    root_target = exact_targets["root"][0]
    return _FNOCountDoc(
        n_tokens=int(sum(int(length) for length in leaf_token_lengths)),
        leaf_token_ids=leaf_token_ids,
        leaf_counts=leaf_counts,
        leaf_first_regimes=leaf_first_regimes,
        leaf_last_regimes=leaf_last_regimes,
        leaf_token_lengths=leaf_token_lengths,
        merge_counts_balanced=tuple(float(target[0]) for target in exact_targets["merge"]),
        merge_sizes_balanced=tuple(0 for _ in exact_targets["merge"]),
        merge_token_lengths=_balanced_merge_token_lengths_from_leaf_lengths(
            leaf_token_lengths
        ),
        root_count=float(root_target[0]),
    )


def _length_band(value: int) -> int:
    target = max(1, int(value))
    band = 1
    while int(band) < int(target):
        band *= 2
    return int(band)


def _tree_work_item_from_doc(
    doc: _FNOCountDoc,
    *,
    doc_index: int = -1,
    work_kind: str = "full_tree",
    collect_leaf: bool = False,
    collect_c2: bool = False,
    collect_c3: bool = False,
    root_only_supervision: bool = False,
    doc_sequence_supervision: bool = False,
    leaf_audit_indices: Optional[set[int] | Tuple[int, ...]] = None,
    c3_audit_indices: Optional[set[int] | Tuple[int, ...]] = None,
    document_mode: str = "",
) -> _TreeWorkItem:
    normalized_leaf_audits = None
    if leaf_audit_indices is not None:
        normalized_leaf_audits = tuple(
            int(value) for value in sorted(int(idx) for idx in leaf_audit_indices)
        )
    normalized_c3_audits = None
    if c3_audit_indices is not None:
        normalized_c3_audits = tuple(
            int(value) for value in sorted(int(idx) for idx in c3_audit_indices)
        )
    return _TreeWorkItem(
        doc_index=int(doc_index),
        doc=doc,
        work_kind=str(work_kind),
        collect_leaf=bool(collect_leaf),
        collect_c2=bool(collect_c2),
        collect_c3=bool(collect_c3),
        root_only_supervision=bool(root_only_supervision),
        doc_sequence_supervision=bool(doc_sequence_supervision),
        leaf_audit_indices=normalized_leaf_audits,
        c3_audit_indices=normalized_c3_audits,
        document_mode=str(document_mode),
    )


def _tree_work_item_bucket_key(
    item: _TreeWorkItem,
    *,
    target_n_leaves: int | None = None,
    auto_queue_enabled: bool = False,
) -> _BatchBucketKey:
    leaf_lengths = [int(length) for length in item.doc.leaf_token_lengths]
    merge_lengths = [int(length) for length in item.doc.merge_token_lengths]
    resolved_target_n_leaves = int(target_n_leaves or item.n_leaves)
    return _BatchBucketKey(
        n_leaves=int(resolved_target_n_leaves),
        work_kind=str(item.work_kind),
        collect_leaf=bool(item.collect_leaf),
        collect_c2=bool(item.collect_c2),
        collect_c3=bool(item.collect_c3),
        max_leaf_tokens_band=int(_length_band(max(leaf_lengths) if leaf_lengths else 1)),
        max_merge_tokens_band=int(_length_band(max(merge_lengths) if merge_lengths else 1)),
        irregular_leaf_layout=bool(len(set(leaf_lengths)) > 1),
        auto_queue_enabled=bool(auto_queue_enabled),
    )


def _tail_repack_group_key(
    batch: _PackedTreeBatch,
    item: _TreeWorkItem,
) -> Tuple[str, int, str, bool, bool, bool, int, int]:
    if bool(batch.bucket_key.auto_queue_enabled) and int(batch.bucket_key.n_leaves) > 0:
        return (
            "auto_queue",
            int(batch.bucket_key.n_leaves),
            str(batch.bucket_key.work_kind),
            bool(batch.bucket_key.collect_leaf),
            bool(batch.bucket_key.collect_c2),
            bool(batch.bucket_key.collect_c3),
            int(batch.bucket_key.max_leaf_tokens_band),
            int(batch.bucket_key.max_merge_tokens_band),
        )
    return (
        "generic",
        0,
        str(item.work_kind),
        bool(item.collect_leaf),
        bool(item.collect_c2),
        bool(item.collect_c3),
        int(batch.bucket_key.max_leaf_tokens_band),
        int(batch.bucket_key.max_merge_tokens_band),
    )


def _tree_work_item_auto_queue_group_key(item: _TreeWorkItem) -> str:
    leaf_lengths = [int(length) for length in item.doc.leaf_token_lengths]
    merge_lengths = [int(length) for length in item.doc.merge_token_lengths]
    return (
        f"{str(item.work_kind)}|leaf={int(bool(item.collect_leaf))}|"
        f"c2={int(bool(item.collect_c2))}|c3={int(bool(item.collect_c3))}|"
        f"leaf_band={int(_length_band(max(leaf_lengths) if leaf_lengths else 1))}|"
        f"merge_band={int(_length_band(max(merge_lengths) if merge_lengths else 1))}"
    )


def _packed_batch_docs_cap(
    batch: _PackedTreeBatch,
    *,
    max_docs: int,
    bucket_docs_cap_by_n_leaves: Mapping[int, int] | None,
) -> int:
    docs_cap = int(max_docs)
    if bucket_docs_cap_by_n_leaves is not None:
        docs_cap = int(
            bucket_docs_cap_by_n_leaves.get(int(batch.bucket_key.n_leaves), docs_cap)
        )
    return max(1, docs_cap)


def _build_generic_tail_packed_batch(
    items: Sequence[_TreeWorkItem],
) -> _PackedTreeBatch:
    if not items:
        raise ValueError("generic tail batch requires at least one item")
    actual_leaf_tokens = int(sum(int(item.total_leaf_tokens) for item in items))
    actual_leaf_slots = int(sum(int(len(item.doc.leaf_token_ids)) for item in items))
    total_nodes = int(sum(int(item.total_nodes) for item in items))
    total_merge_ops = int(sum(int(item.total_merge_ops) for item in items))
    max_leaf_len = max(
        (
            max((int(length) for length in item.doc.leaf_token_lengths), default=1)
            for item in items
        ),
        default=1,
    )
    max_merge_len = max(
        (
            max((int(length) for length in item.doc.merge_token_lengths), default=1)
            for item in items
        ),
        default=1,
    )
    padded_leaf_tokens = int(
        max(
            actual_leaf_tokens,
            sum(int(len(item.doc.leaf_token_ids)) for item in items) * int(max_leaf_len),
        )
    )
    first = items[0]
    return _PackedTreeBatch(
        bucket_key=_BatchBucketKey(
            n_leaves=0,
            work_kind=str(first.work_kind),
            collect_leaf=bool(first.collect_leaf),
            collect_c2=bool(first.collect_c2),
            collect_c3=bool(first.collect_c3),
            max_leaf_tokens_band=int(_length_band(max_leaf_len)),
            max_merge_tokens_band=int(_length_band(max_merge_len)),
            irregular_leaf_layout=True,
        ),
        items=tuple(items),
        actual_leaf_tokens=int(actual_leaf_tokens),
        padded_leaf_tokens=int(padded_leaf_tokens),
        total_nodes=int(total_nodes),
        total_merge_ops=int(total_merge_ops),
        actual_leaf_slots=int(actual_leaf_slots),
        padded_leaf_slots=int(actual_leaf_slots),
    )


def _build_auto_queue_tail_packed_batch(
    items: Sequence[_TreeWorkItem],
    *,
    bucket_key: _BatchBucketKey,
) -> _PackedTreeBatch:
    if not items:
        raise ValueError("auto-queue tail batch requires at least one item")
    target_n_leaves = int(bucket_key.n_leaves)
    if not bool(bucket_key.auto_queue_enabled) or target_n_leaves <= 0:
        raise ValueError("auto-queue tail batch requires an enabled target leaf count")
    actual_leaf_tokens = int(sum(int(item.total_leaf_tokens) for item in items))
    actual_leaf_slots = int(sum(int(len(item.doc.leaf_token_ids)) for item in items))
    padded_leaf_slots = int(len(items)) * int(target_n_leaves)
    max_leaf_len = max(
        (
            max((int(length) for length in item.doc.leaf_token_lengths), default=1)
            for item in items
        ),
        default=1,
    )
    padded_leaf_tokens = int(
        max(
            actual_leaf_tokens,
            int(padded_leaf_slots) * int(max_leaf_len),
        )
    )
    total_nodes = int(len(items)) * int((2 * target_n_leaves) - 1)
    total_merge_ops = int(len(items)) * int(max(0, target_n_leaves - 1))
    return _PackedTreeBatch(
        bucket_key=bucket_key,
        items=tuple(items),
        actual_leaf_tokens=int(actual_leaf_tokens),
        padded_leaf_tokens=int(padded_leaf_tokens),
        total_nodes=int(total_nodes),
        total_merge_ops=int(total_merge_ops),
        actual_leaf_slots=int(actual_leaf_slots),
        padded_leaf_slots=int(padded_leaf_slots),
    )


def _repack_small_tail_tree_batches(
    packed_batches: Sequence[_PackedTreeBatch],
    *,
    max_docs: int,
    max_total_leaf_tokens: int,
    max_total_nodes: int,
    max_total_merge_ops: int,
    bucket_docs_cap_by_n_leaves: Mapping[int, int] | None,
    tail_repack_fill_ratio: float,
    tail_repack_min_docs: int,
) -> List[_PackedTreeBatch]:
    fill_ratio = max(0.0, float(tail_repack_fill_ratio))
    min_docs = max(0, int(tail_repack_min_docs))
    if not packed_batches or (fill_ratio <= 0.0 and min_docs <= 0):
        return list(packed_batches)

    retained: List[_PackedTreeBatch] = []
    tail_groups: Dict[Tuple[str, int, str, bool, bool, bool, int, int], List[_TreeWorkItem]] = {}
    tail_bucket_keys: Dict[Tuple[str, int, str, bool, bool, bool, int, int], _BatchBucketKey] = {}
    for batch in packed_batches:
        docs_cap = _packed_batch_docs_cap(
            batch,
            max_docs=int(max_docs),
            bucket_docs_cap_by_n_leaves=bucket_docs_cap_by_n_leaves,
        )
        batch_docs = int(len(batch.items))
        should_repack = False
        if batch_docs < docs_cap:
            if min_docs > 0 and batch_docs <= min_docs:
                should_repack = True
            elif fill_ratio > 0.0 and (float(batch_docs) / float(max(1, docs_cap))) < fill_ratio:
                should_repack = True
        if not should_repack:
            retained.append(batch)
            continue
        for item in batch.items:
            group_key = _tail_repack_group_key(batch, item)
            tail_groups.setdefault(group_key, []).append(item)
            tail_bucket_keys.setdefault(group_key, batch.bucket_key)

    if not tail_groups:
        return list(packed_batches)

    repacked: List[_PackedTreeBatch] = []
    token_budget = int(max_total_leaf_tokens)
    node_budget = int(max_total_nodes)
    merge_budget = int(max_total_merge_ops)
    for group_key, group_items in tail_groups.items():
        bucket_key = tail_bucket_keys[group_key]
        docs_cap = max(1, int(max_docs))
        if bool(bucket_key.auto_queue_enabled) and int(bucket_key.n_leaves) > 0:
            if bucket_docs_cap_by_n_leaves is not None:
                docs_cap = max(
                    1,
                    int(
                        bucket_docs_cap_by_n_leaves.get(
                            int(bucket_key.n_leaves),
                            int(max_docs),
                        )
                    ),
                )
        ordered_items = sorted(
            group_items,
            key=lambda item: (
                -int(item.total_leaf_tokens),
                -int(item.total_nodes),
                int(item.doc_index),
            ),
        )
        active: List[_TreeWorkItem] = []
        actual_leaf_tokens = 0
        total_nodes = 0
        total_merge_ops = 0
        max_leaf_len = 0

        def _flush() -> None:
            nonlocal active, actual_leaf_tokens, total_nodes, total_merge_ops, max_leaf_len
            if not active:
                return
            if bool(bucket_key.auto_queue_enabled) and int(bucket_key.n_leaves) > 0:
                repacked.append(
                    _build_auto_queue_tail_packed_batch(
                        active,
                        bucket_key=bucket_key,
                    )
                )
            else:
                repacked.append(_build_generic_tail_packed_batch(active))
            active = []
            actual_leaf_tokens = 0
            total_nodes = 0
            total_merge_ops = 0
            max_leaf_len = 0

        for item in ordered_items:
            proposed_max_leaf_len = max(
                int(max_leaf_len),
                max((int(length) for length in item.doc.leaf_token_lengths), default=0),
            )
            proposed_leaf_tokens = int(actual_leaf_tokens + item.total_leaf_tokens)
            if bool(bucket_key.auto_queue_enabled) and int(bucket_key.n_leaves) > 0:
                target_n_leaves = int(bucket_key.n_leaves)
                proposed_batch_docs = int(len(active) + 1)
                proposed_nodes = int(proposed_batch_docs) * int((2 * target_n_leaves) - 1)
                proposed_merge_ops = int(proposed_batch_docs) * int(max(0, target_n_leaves - 1))
                padded_if_added = (
                    int(proposed_batch_docs)
                    * int(target_n_leaves)
                    * int(proposed_max_leaf_len)
                )
            else:
                proposed_nodes = int(total_nodes + item.total_nodes)
                proposed_merge_ops = int(total_merge_ops + item.total_merge_ops)
                padded_if_added = (
                    sum(
                        int(len(existing.doc.leaf_token_ids)) * int(proposed_max_leaf_len)
                        for existing in active
                    )
                    + int(len(item.doc.leaf_token_ids)) * int(proposed_max_leaf_len)
                )
            if active and (
                int(len(active) + 1) > int(docs_cap)
                or (token_budget > 0 and int(padded_if_added) > int(token_budget))
                or (node_budget > 0 and int(proposed_nodes) > int(node_budget))
                or (merge_budget > 0 and int(proposed_merge_ops) > int(merge_budget))
            ):
                _flush()
                proposed_max_leaf_len = max(
                    (int(length) for length in item.doc.leaf_token_lengths),
                    default=0,
                )
                proposed_leaf_tokens = int(item.total_leaf_tokens)
                if bool(bucket_key.auto_queue_enabled) and int(bucket_key.n_leaves) > 0:
                    target_n_leaves = int(bucket_key.n_leaves)
                    proposed_nodes = int((2 * target_n_leaves) - 1)
                    proposed_merge_ops = int(max(0, target_n_leaves - 1))
                else:
                    proposed_nodes = int(item.total_nodes)
                    proposed_merge_ops = int(item.total_merge_ops)
            active.append(item)
            actual_leaf_tokens = int(proposed_leaf_tokens)
            total_nodes = int(proposed_nodes)
            total_merge_ops = int(proposed_merge_ops)
            max_leaf_len = int(proposed_max_leaf_len)
        _flush()
    return list(retained) + list(repacked)


def _pack_tree_work_items(
    items: Sequence[_TreeWorkItem],
    *,
    max_docs: int,
    max_total_leaf_tokens: int,
    max_total_nodes: int,
    max_total_merge_ops: int,
    bucket_docs_cap_by_n_leaves: Mapping[int, int] | None = None,
    bucket_mode: str = GPU_RUNTIME_BUCKET_MODE_EXACT_THEN_BUCKETED,
    structural_pad_limit: float = 0.5,
    auto_queue_min_docs: int = 8,
    auto_queue_target_by_n_leaves: Mapping[int, int] | None = None,
    runtime_mode: str | None = None,
    tail_repack_fill_ratio: float = 0.0,
    tail_repack_min_docs: int = 0,
) -> List[_PackedTreeBatch]:
    resolved_runtime_mode = resolve_runtime_mode(
        runtime_mode,
        env_var="TT_TREE_BATCH_RUNTIME_MODE",
    )
    normalized_bucket_mode = normalize_gpu_runtime_bucket_mode(bucket_mode)
    use_shared_planner = bool(
        resolved_runtime_mode == RUNTIME_MODE_UNIFIED_V2
        or normalized_bucket_mode == GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE
    )
    resolved_auto_queue_targets = {
        int(key): int(value)
        for key, value in dict(auto_queue_target_by_n_leaves or {}).items()
        if int(key) > 0 and int(value) >= int(key)
    }
    if normalized_bucket_mode == GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE and not resolved_auto_queue_targets:
        resolved_auto_queue_targets = _leaf_count_auto_queue_targets_for_docs(
            [item.doc for item in items],
            structural_pad_limit=float(structural_pad_limit),
            min_docs=int(auto_queue_min_docs),
        )
    if use_shared_planner:
        work_items: List[WorkItem] = []
        docs_cap_by_signature: Dict[str, int] = {}
        source_by_id: Dict[str, _TreeWorkItem] = {}
        for item in items:
            actual_n_leaves = int(len(item.doc.leaf_token_ids))
            target_n_leaves = int(
                resolved_auto_queue_targets.get(int(actual_n_leaves), int(actual_n_leaves))
            )
            max_leaf_tokens = max(
                (int(length) for length in item.doc.leaf_token_lengths),
                default=1,
            )
            item_id = (
                f"{int(item.doc_index)}:{int(item.n_leaves)}:{str(item.work_kind)}:"
                f"{int(item.collect_leaf)}:{int(item.collect_c2)}:{int(item.collect_c3)}"
            )
            topology_signature = (
                f"n{int(item.n_leaves)}:"
                f"leaf{tuple(int(v) for v in item.doc.leaf_token_lengths)}:"
                f"merge{tuple(int(v) for v in item.doc.merge_token_lengths)}"
            )
            work_item = WorkItem(
                item_id=item_id,
                backend_family="neural_tree",
                op_kind=str(item.work_kind),
                topology_signature=topology_signature,
                supervision_mask=(
                    f"leaf={int(bool(item.collect_leaf))}|"
                    f"c2={int(bool(item.collect_c2))}|"
                    f"c3={int(bool(item.collect_c3))}|"
                    f"root={int(bool(item.root_only_supervision))}|"
                    f"docseq={int(bool(item.doc_sequence_supervision))}"
                ),
                doc_id=str(item.doc_index),
                payload=item,
                estimated_tokens=int(item.total_leaf_tokens),
                estimated_nodes=int((2 * target_n_leaves) - 1),
                estimated_merge_ops=int(max(0, target_n_leaves - 1)),
                padding_multiple=max(1, int(target_n_leaves)),
                padding_length=max(1, int(max_leaf_tokens)),
                metadata={
                    "leaf_count": int(actual_n_leaves),
                    "auto_queue_group_key": _tree_work_item_auto_queue_group_key(item),
                    "auto_queue_target_leaf_count": (
                        int(target_n_leaves)
                        if normalized_bucket_mode == GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE
                        else 0
                    ),
                    "auto_queue_docs_cap_key": (
                        f"tree_auto_queue_target_n{int(target_n_leaves)}"
                        if normalized_bucket_mode == GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE
                        else ""
                    ),
                },
            )
            work_items.append(work_item)
            source_by_id[work_item.item_id] = item
            if bucket_docs_cap_by_n_leaves is not None:
                docs_cap_key = (
                    f"tree_auto_queue_target_n{int(target_n_leaves)}"
                    if normalized_bucket_mode == GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE
                    else str(work_item.shape_key)
                )
                docs_cap_by_signature[str(docs_cap_key)] = int(
                    bucket_docs_cap_by_n_leaves.get(
                        int(target_n_leaves),
                        int(max_docs),
                    )
                )
        planned = plan_work_batches(
            work_items,
            max_docs=int(max_docs),
            max_total_tokens=int(max_total_leaf_tokens),
            max_total_nodes=int(max_total_nodes),
            max_total_merge_ops=int(max_total_merge_ops),
            docs_cap_by_signature=docs_cap_by_signature or None,
            plan_cache=get_named_plan_cache("ctreepo_neural_tree_batches"),
            bucket_mode=str(normalized_bucket_mode),
            structural_pad_limit=float(structural_pad_limit),
            auto_queue_min_docs=int(auto_queue_min_docs),
        )
        packed_batches: List[_PackedTreeBatch] = []
        for batch in planned:
            packed_items = tuple(
                source_by_id[str(batch_item.item_id)] for batch_item in batch.items
            )
            first_work_item = batch.items[0]
            target_n_leaves = int(
                first_work_item.metadata.get("auto_queue_target_leaf_count", 0) or len(packed_items[0].doc.leaf_token_ids)
            )
            actual_leaf_slots = int(sum(int(len(item.doc.leaf_token_ids)) for item in packed_items))
            padded_leaf_slots = int(len(packed_items)) * int(target_n_leaves)
            packed_batches.append(
                _PackedTreeBatch(
                    bucket_key=_tree_work_item_bucket_key(
                        packed_items[0],
                        target_n_leaves=int(target_n_leaves),
                        auto_queue_enabled=bool(
                            normalized_bucket_mode == GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE
                        ),
                    ),
                    items=packed_items,
                    actual_leaf_tokens=int(batch.actual_tokens),
                    padded_leaf_tokens=int(batch.padded_tokens),
                    total_nodes=int(batch.total_nodes),
                    total_merge_ops=int(batch.total_merge_ops),
                    actual_leaf_slots=int(actual_leaf_slots),
                    padded_leaf_slots=int(padded_leaf_slots),
                )
            )
        packed_batches = _repack_small_tail_tree_batches(
            packed_batches,
            max_docs=int(max_docs),
            max_total_leaf_tokens=int(max_total_leaf_tokens),
            max_total_nodes=int(max_total_nodes),
            max_total_merge_ops=int(max_total_merge_ops),
            bucket_docs_cap_by_n_leaves=bucket_docs_cap_by_n_leaves,
            tail_repack_fill_ratio=float(tail_repack_fill_ratio),
            tail_repack_min_docs=int(tail_repack_min_docs),
        )
        packed_batches.sort(
            key=lambda batch: (
                int(batch.bucket_key.n_leaves),
                int(batch.bucket_key.max_leaf_tokens_band),
                str(batch.bucket_key.work_kind),
                -int(len(batch.items)),
            )
        )
        return packed_batches

    grouped: Dict[_BatchBucketKey, List[_TreeWorkItem]] = {}
    for item in items:
        grouped.setdefault(_tree_work_item_bucket_key(item), []).append(item)
    packed_batches: List[_PackedTreeBatch] = []
    for bucket_key, bucket_items in grouped.items():
        ordered_items = sorted(
            bucket_items,
            key=lambda item: (
                -int(item.total_leaf_tokens),
                -int(item.total_nodes),
                int(item.doc_index),
            ),
        )
        docs_cap = int(max_docs)
        if bucket_docs_cap_by_n_leaves is not None:
            docs_cap = int(
                bucket_docs_cap_by_n_leaves.get(
                    int(bucket_key.n_leaves),
                    docs_cap,
                )
            )
        if docs_cap <= 0:
            docs_cap = 10**9
        active: List[_TreeWorkItem] = []
        actual_leaf_tokens = 0
        total_nodes = 0
        total_merge_ops = 0
        max_leaf_len = 0

        def _flush() -> None:
            nonlocal active, actual_leaf_tokens, total_nodes, total_merge_ops, max_leaf_len
            if not active:
                return
            padded_leaf_tokens = sum(
                int(len(item.doc.leaf_token_ids)) * int(max_leaf_len)
                for item in active
            )
            packed_batches.append(
                _PackedTreeBatch(
                    bucket_key=bucket_key,
                    items=tuple(active),
                    actual_leaf_tokens=int(actual_leaf_tokens),
                    padded_leaf_tokens=int(max(int(actual_leaf_tokens), int(padded_leaf_tokens))),
                    total_nodes=int(total_nodes),
                    total_merge_ops=int(total_merge_ops),
                    actual_leaf_slots=int(sum(int(len(item.doc.leaf_token_ids)) for item in active)),
                    padded_leaf_slots=int(sum(int(len(item.doc.leaf_token_ids)) for item in active)),
                )
            )
            active = []
            actual_leaf_tokens = 0
            total_nodes = 0
            total_merge_ops = 0
            max_leaf_len = 0

        for item in ordered_items:
            proposed_max_leaf_len = max(
                int(max_leaf_len),
                max((int(length) for length in item.doc.leaf_token_lengths), default=0),
            )
            proposed_leaf_tokens = int(actual_leaf_tokens + item.total_leaf_tokens)
            proposed_nodes = int(total_nodes + item.total_nodes)
            proposed_merge_ops = int(total_merge_ops + item.total_merge_ops)
            padded_if_added = (
                sum(
                    int(len(existing.doc.leaf_token_ids)) * int(proposed_max_leaf_len)
                    for existing in active
                )
                + int(len(item.doc.leaf_token_ids)) * int(proposed_max_leaf_len)
            )
            if active and (
                int(len(active) + 1) > int(docs_cap)
                or (
                    int(max_total_leaf_tokens) > 0
                    and int(padded_if_added) > int(max_total_leaf_tokens)
                )
                or (
                    int(max_total_nodes) > 0
                    and int(proposed_nodes) > int(max_total_nodes)
                )
                or (
                    int(max_total_merge_ops) > 0
                    and int(proposed_merge_ops) > int(max_total_merge_ops)
                )
            ):
                _flush()
                proposed_max_leaf_len = max(
                    (int(length) for length in item.doc.leaf_token_lengths),
                    default=0,
                )
                proposed_leaf_tokens = int(item.total_leaf_tokens)
                proposed_nodes = int(item.total_nodes)
                proposed_merge_ops = int(item.total_merge_ops)
            active.append(item)
            actual_leaf_tokens = int(proposed_leaf_tokens)
            total_nodes = int(proposed_nodes)
            total_merge_ops = int(proposed_merge_ops)
            max_leaf_len = int(proposed_max_leaf_len)
        _flush()
    packed_batches = _repack_small_tail_tree_batches(
        packed_batches,
        max_docs=int(max_docs),
        max_total_leaf_tokens=int(max_total_leaf_tokens),
        max_total_nodes=int(max_total_nodes),
        max_total_merge_ops=int(max_total_merge_ops),
        bucket_docs_cap_by_n_leaves=bucket_docs_cap_by_n_leaves,
        tail_repack_fill_ratio=float(tail_repack_fill_ratio),
        tail_repack_min_docs=int(tail_repack_min_docs),
    )
    packed_batches.sort(
        key=lambda batch: (
            int(batch.bucket_key.n_leaves),
            int(batch.bucket_key.max_leaf_tokens_band),
            str(batch.bucket_key.work_kind),
            -int(len(batch.items)),
        )
    )
    return packed_batches


def _docs_share_fixed_leaf_shape(docs: Sequence[_FNOCountDoc]) -> bool:
    if not docs:
        return False
    reference_leaf_lengths = tuple(int(v) for v in docs[0].leaf_token_lengths)
    reference_merge_lengths = tuple(int(v) for v in docs[0].merge_token_lengths)
    if not reference_leaf_lengths:
        return False
    if len(set(reference_leaf_lengths)) != 1:
        return False
    for doc in docs[1:]:
        if tuple(int(v) for v in doc.leaf_token_lengths) != reference_leaf_lengths:
            return False
        if tuple(int(v) for v in doc.merge_token_lengths) != reference_merge_lengths:
            return False
    return True


def _docs_support_fixed_leaf_auto_queue(
    docs: Sequence[_FNOCountDoc],
) -> bool:
    if not docs:
        return False
    reference_leaf_width = max(
        (int(length) for length in docs[0].leaf_token_lengths),
        default=0,
    )
    if reference_leaf_width <= 0:
        return False
    for doc in docs:
        if int(len(doc.leaf_token_ids)) <= 0:
            return False
        leaf_lengths = tuple(int(v) for v in doc.leaf_token_lengths)
        if not leaf_lengths:
            return False
        if max(leaf_lengths, default=0) != int(reference_leaf_width):
            return False
        if any(int(length) <= 0 or int(length) > int(reference_leaf_width) for length in leaf_lengths):
            return False
        if any(int(length) != int(reference_leaf_width) for length in leaf_lengths[:-1]):
            return False
    return True


def _build_dense_leaf_token_tensors_from_docs(
    docs: Sequence[_FNOCountDoc],
    *,
    pad_id: int,
    target_n_leaves: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not docs:
        raise ValueError("docs must be non-empty")
    if not _docs_support_fixed_leaf_auto_queue(docs):
        raise ValueError("docs must support fixed-leaf auto-queue batching")
    resolved_target_n_leaves = int(
        target_n_leaves or max((int(len(doc.leaf_token_ids)) for doc in docs), default=0)
    )
    if resolved_target_n_leaves <= 0:
        raise ValueError("target_n_leaves must be positive")
    max_leaf_tokens = max(
        (int(length) for doc in docs for length in doc.leaf_token_lengths),
        default=1,
    )
    batch_size = int(len(docs))
    leaf_tokens = torch.full(
        (batch_size, int(resolved_target_n_leaves), int(max_leaf_tokens)),
        int(pad_id),
        dtype=torch.long,
    )
    token_mask = torch.zeros(
        (batch_size, int(resolved_target_n_leaves), int(max_leaf_tokens)),
        dtype=torch.float32,
    )
    leaf_valid_mask = torch.zeros(
        (batch_size, int(resolved_target_n_leaves)),
        dtype=torch.bool,
    )
    for row_idx, doc in enumerate(docs):
        for leaf_idx, span_tokens in enumerate(doc.leaf_token_ids):
            valid = int(len(span_tokens))
            if valid <= 0:
                continue
            leaf_tokens[int(row_idx), int(leaf_idx), :valid] = torch.as_tensor(
                list(span_tokens),
                dtype=torch.long,
            )
            token_mask[int(row_idx), int(leaf_idx), :valid] = 1.0
            leaf_valid_mask[int(row_idx), int(leaf_idx)] = True
    return leaf_tokens, token_mask, leaf_valid_mask


def _encode_dense_leaf_token_tensor_batch(
    model: FNOCountSketch,
    token_tensor: torch.Tensor,
    *,
    device: torch.device,
    token_mask_tensor: torch.Tensor | None = None,
    runtime_telemetry: GpuRuntimeTelemetry | None = None,
) -> torch.Tensor:
    if token_tensor.ndim != 3:
        raise ValueError("token_tensor must have shape (batch, leaves, leaf_tokens)")
    batch_size, n_leaves, leaf_tokens = [int(v) for v in token_tensor.shape]
    use_pinned_staging = bool(device.type == "cuda" and token_tensor.device.type == "cpu")
    if use_pinned_staging and not token_tensor.is_pinned():
        token_tensor = token_tensor.pin_memory()
    flat_tokens = token_tensor.reshape(batch_size * n_leaves, leaf_tokens)
    if token_mask_tensor is None:
        flat_mask = torch.ones(
            (batch_size * n_leaves, leaf_tokens),
            dtype=torch.float32,
            device=torch.device("cpu") if use_pinned_staging else token_tensor.device,
            pin_memory=use_pinned_staging,
        )
    else:
        flat_mask = token_mask_tensor.reshape(batch_size * n_leaves, leaf_tokens)
        if use_pinned_staging and flat_mask.device.type == "cpu" and not flat_mask.is_pinned():
            flat_mask = flat_mask.pin_memory()
        elif flat_mask.device != token_tensor.device and flat_mask.device.type == "cpu":
            flat_mask = flat_mask.to(device=token_tensor.device)
    if use_pinned_staging:
        copy_start_s = time.perf_counter()
        flat_tokens = flat_tokens.to(device=device, non_blocking=True)
        flat_mask = flat_mask.to(device=device, non_blocking=True)
        if runtime_telemetry is not None:
            runtime_telemetry.add_h2d(
                bytes_transferred=_tensor_nbytes(flat_tokens) + _tensor_nbytes(flat_mask),
                wall_time_s=time.perf_counter() - copy_start_s,
            )
    elif flat_tokens.device != device:
        copy_start_s = time.perf_counter()
        flat_tokens = flat_tokens.to(device=device, non_blocking=True)
        flat_mask = flat_mask.to(device=device, non_blocking=True)
        if runtime_telemetry is not None:
            runtime_telemetry.add_h2d(
                bytes_transferred=_tensor_nbytes(flat_tokens) + _tensor_nbytes(flat_mask),
                wall_time_s=time.perf_counter() - copy_start_s,
            )

    with _autocast_context(device):
        x, pooled = model._encode_token_batch(flat_tokens, token_mask=flat_mask)
        h = model.leaf_proj(pooled)
        if model.use_summary_spec:
            if model.use_opaque_carrier_exact_sketch_surface:
                packed = model._opaque_carrier_leaf_state_from_carrier(h)
                return packed.reshape(batch_size, n_leaves, -1)
            if model.use_carrier_projection_surface:
                x_seq = x.permute(0, 2, 1)
                if token_mask_tensor is not None:
                    last_idx = flat_mask.sum(dim=-1).long().clamp(min=1) - 1
                else:
                    last_idx = torch.full(
                        (int(x_seq.shape[0]),),
                        int(x_seq.shape[1]) - 1,
                        device=x_seq.device,
                        dtype=torch.long,
                    )
                packed = model._carrier_leaf_state_from_leaf_features(
                    pooled=pooled,
                    first_features=x_seq[:, 0, :],
                    last_features=x_seq[
                        torch.arange(int(x_seq.shape[0]), device=x_seq.device),
                        last_idx,
                    ],
                )
                return packed.reshape(batch_size, n_leaves, -1)
            if model.use_shared_theorem_surface:
                return h.reshape(batch_size, n_leaves, -1)
            if (
                model.summary_count_leaf_proj is None
                or model.summary_first_leaf_proj is None
                or model.summary_last_leaf_proj is None
            ):
                raise RuntimeError("summary-spec leaf projectors are not initialized")
            x_seq = x.permute(0, 2, 1)
            count_slot = model.summary_count_leaf_proj(pooled)
            first_slot = model.summary_first_leaf_proj(x_seq[:, 0, :])
            last_slot = model.summary_last_leaf_proj(x_seq[:, -1, :])
            if model.summary_residual_leaf_proj is None or int(model.residual_dim) <= 0:
                residual = count_slot.new_zeros((batch_size * n_leaves, int(model.residual_dim)))
            else:
                residual = model.summary_residual_leaf_proj(pooled)
            packed = model._pack_summary_spec_state(
                count_slot,
                first_slot,
                last_slot,
                residual,
            )
            return packed.reshape(batch_size, n_leaves, -1)
        if model.use_decoded_markov_sketch:
            return h.reshape(batch_size, n_leaves, -1)
        x_seq = x.permute(0, 2, 1)
        first_reg = model.first_endpoint_proj(x_seq[:, 0, :])
        last_reg = model.last_endpoint_proj(x_seq[:, -1, :])
        packed = model._pack_state(h, first_reg, last_reg)
        return packed.reshape(batch_size, n_leaves, -1)


def _flatten_merge_state_batch(merge_levels: Sequence[torch.Tensor]) -> torch.Tensor | None:
    flat_levels = [level for level in list(merge_levels) if int(level.shape[1]) > 0]
    if not flat_levels:
        return None
    return torch.cat(flat_levels, dim=1)


def _precompute_balanced_doc_state_levels(
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
    collect_merge_states: bool,
    prefer_fixed_fused: bool = False,
    target_n_leaves: int | None = None,
    resident_view: GpuBatchView | None = None,
    runtime_telemetry: GpuRuntimeTelemetry | None = None,
) -> _PrecomputedBatchTreeLevels:
    if not docs:
        raise ValueError("docs must be non-empty")
    n_docs = int(len(docs))
    resident_leaf_tokens = (
        resident_view.tensors.get("leaf_tokens")
        if resident_view is not None
        else None
    )
    resident_leaf_mask = (
        resident_view.tensors.get("leaf_mask")
        if resident_view is not None
        else None
    )
    resident_leaf_valid_mask = (
        resident_view.tensors.get("leaf_valid_mask")
        if resident_view is not None
        else None
    )
    actual_leaf_counts = [int(len(doc.leaf_token_ids)) for doc in docs]
    reference_leaf_count = int(actual_leaf_counts[0])
    same_leaf_count = bool(
        all(int(count) == int(reference_leaf_count) for count in actual_leaf_counts[1:])
    )
    effective_target_n_leaves = int(target_n_leaves or 0)
    if isinstance(resident_leaf_tokens, torch.Tensor):
        effective_target_n_leaves = int(resident_leaf_tokens.shape[1])
    elif effective_target_n_leaves <= 0:
        effective_target_n_leaves = int(max(actual_leaf_counts, default=0))
    if effective_target_n_leaves <= 0:
        raise ValueError("effective target leaf count must be positive")
    if any(int(count) > int(effective_target_n_leaves) for count in actual_leaf_counts):
        raise ValueError("target_n_leaves must be >= each document leaf count")
    if not same_leaf_count and not (
        bool(prefer_fixed_fused) and _docs_support_fixed_leaf_auto_queue(docs)
    ):
        raise ValueError("all docs in a precomputed batch must share the same leaf count")

    leaf_valid_mask: torch.Tensor
    if isinstance(resident_leaf_tokens, torch.Tensor):
        leaf_states = _encode_dense_leaf_token_tensor_batch(
            model,
            resident_leaf_tokens,
            device=device,
            token_mask_tensor=resident_leaf_mask if isinstance(resident_leaf_mask, torch.Tensor) else None,
            runtime_telemetry=runtime_telemetry,
        )
        if isinstance(resident_leaf_valid_mask, torch.Tensor):
            leaf_valid_mask = resident_leaf_valid_mask.to(
                device=device,
                dtype=torch.bool,
                non_blocking=bool(
                    resident_leaf_valid_mask.device.type == "cpu"
                    and resident_leaf_valid_mask.is_pinned()
                ),
            )
        else:
            leaf_valid_mask = torch.ones(
                (n_docs, int(effective_target_n_leaves)),
                device=device,
                dtype=torch.bool,
            )
    elif bool(prefer_fixed_fused) and (
        _docs_share_fixed_leaf_shape(docs) or _docs_support_fixed_leaf_auto_queue(docs)
    ):
        dense_leaf_tokens, dense_leaf_mask, leaf_valid_mask_cpu = _build_dense_leaf_token_tensors_from_docs(
            docs,
            pad_id=int(model.pad_id),
            target_n_leaves=int(effective_target_n_leaves),
        )
        leaf_states = _encode_dense_leaf_token_tensor_batch(
            model,
            dense_leaf_tokens,
            device=device,
            token_mask_tensor=dense_leaf_mask,
            runtime_telemetry=runtime_telemetry,
        )
        leaf_valid_mask = leaf_valid_mask_cpu.to(device=device, dtype=torch.bool)
    else:
        if not same_leaf_count:
            raise ValueError("non-fixed-fused batches must share the same leaf count")
        flat_leaf_tokens: List[Sequence[int]] = []
        for doc in docs:
            flat_leaf_tokens.extend(doc.leaf_token_ids)
        with _autocast_context(device):
            flat_state_batch = model.encode_leaf_tokens_batch(
                flat_leaf_tokens,
                device=device,
                runtime_telemetry=runtime_telemetry,
            )
            state_dim = int(flat_state_batch.shape[-1])
            leaf_states = flat_state_batch.reshape(n_docs, reference_leaf_count, state_dim)
        leaf_valid_mask = torch.ones(
            (n_docs, int(reference_leaf_count)),
            device=device,
            dtype=torch.bool,
        )

    if int(leaf_states.shape[1]) != int(leaf_valid_mask.shape[1]):
        raise ValueError("leaf_states and leaf_valid_mask must align")
    leaf_states = leaf_states * leaf_valid_mask.unsqueeze(-1).to(dtype=leaf_states.dtype)

    cur = leaf_states
    cur_active = leaf_valid_mask
    merge_levels: List[torch.Tensor] = []
    merge_valid_levels: List[torch.Tensor] = []
    while int(cur.shape[1]) > 1:
        pair_count = int(cur.shape[1] // 2)
        if pair_count > 0:
            left_states = cur[:, 0 : 2 * pair_count : 2, :]
            right_states = cur[:, 1 : 2 * pair_count : 2, :]
            left_active = cur_active[:, 0 : 2 * pair_count : 2]
            right_active = cur_active[:, 1 : 2 * pair_count : 2]
            with _autocast_context(device):
                merged_raw = model._merge_state_pairs(
                    left_states,
                    right_states,
                )
            if merged_raw.ndim == 2:
                merged_raw = merged_raw.unsqueeze(1)
            merge_valid = left_active & right_active
            carry_left = left_active & (~right_active)
            carry_right = (~left_active) & right_active
            zero_state = merged_raw.new_zeros(merged_raw.shape)
            merged = torch.where(
                merge_valid.unsqueeze(-1),
                merged_raw,
                torch.where(
                    carry_left.unsqueeze(-1),
                    left_states,
                    torch.where(carry_right.unsqueeze(-1), right_states, zero_state),
                ),
            )
            merged = merged * (merge_valid | carry_left | carry_right).unsqueeze(-1).to(dtype=merged.dtype)
            if collect_merge_states:
                merge_levels.append(merged)
                merge_valid_levels.append(merge_valid)
        else:
            merged = cur.new_zeros((n_docs, 0, int(cur.shape[-1])))
            merge_valid = cur_active.new_zeros((n_docs, 0))
        if int(cur.shape[1]) % 2 == 1:
            cur = torch.cat([merged, cur[:, -1:, :]], dim=1)
            cur_active = torch.cat(
                [merge_valid | carry_left | carry_right, cur_active[:, -1:]],
                dim=1,
            )
        else:
            cur = merged
            cur_active = merge_valid | carry_left | carry_right
    root_states = cur[:, 0, :]
    merge_valid_flat = (
        torch.cat([mask for mask in merge_valid_levels if int(mask.shape[1]) > 0], dim=1)
        if merge_valid_levels
        else leaf_valid_mask.new_zeros((n_docs, 0))
    )
    node_valid_mask = torch.cat(
        (
            leaf_valid_mask,
            merge_valid_flat,
            torch.ones((n_docs, 1), device=device, dtype=torch.bool),
        ),
        dim=1,
    )
    return _PrecomputedBatchTreeLevels(
        leaf_states=leaf_states,
        merge_levels=tuple(merge_levels),
        root_states=root_states,
        leaf_valid_mask=leaf_valid_mask,
        merge_valid_levels=tuple(merge_valid_levels),
        node_valid_mask=node_valid_mask,
    )


def _flatten_fno_doc_tokens(doc: _FNOCountDoc) -> Tuple[int, ...]:
    tokens: List[int] = []
    for span_tokens in doc.leaf_token_ids:
        tokens.extend(int(token) for token in span_tokens)
    return tuple(tokens)


def _collapse_fno_doc_to_root_only(doc: _FNOCountDoc) -> _FNOCountDoc:
    full_tokens = list(_flatten_fno_doc_tokens(doc))
    if not full_tokens:
        raise ValueError("cannot collapse an empty FNO doc to root-only view")
    first_regime = int(doc.leaf_first_regimes[0]) if doc.leaf_first_regimes else 0
    last_regime = int(doc.leaf_last_regimes[-1]) if doc.leaf_last_regimes else first_regime
    return _FNOCountDoc(
        n_tokens=int(doc.n_tokens),
        leaf_token_ids=(tuple(full_tokens),),
        leaf_counts=(float(doc.root_count),),
        leaf_first_regimes=(int(first_regime),),
        leaf_last_regimes=(int(last_regime),),
        leaf_token_lengths=(int(len(full_tokens)),),
        merge_counts_balanced=tuple(),
        merge_sizes_balanced=tuple(),
        merge_token_lengths=tuple(),
        root_count=float(doc.root_count),
    )


def _apply_root_only_fraction_to_fno_docs(
    docs: Sequence[_FNOCountDoc],
    *,
    root_only_fraction: float,
    seed: int,
) -> Tuple[_FNOCountDoc, ...]:
    fraction = min(1.0, max(0.0, float(root_only_fraction)))
    if fraction <= 0.0 or not docs:
        return tuple(docs)
    if fraction >= 1.0:
        return tuple(_collapse_fno_doc_to_root_only(doc) for doc in docs)

    import random as _random

    rng = _random.Random(int(seed))
    n_docs = int(len(docs))
    n_root_only = min(n_docs, max(0, int(round(fraction * float(n_docs)))))
    selected = set(rng.sample(range(n_docs), k=n_root_only)) if n_root_only > 0 else set()
    out: List[_FNOCountDoc] = []
    for idx, doc in enumerate(docs):
        if idx in selected:
            out.append(_collapse_fno_doc_to_root_only(doc))
        else:
            out.append(doc)
    return tuple(out)


def _root_only_view_fno_docs(
    docs: Sequence[_FNOCountDoc],
) -> Tuple[_FNOCountDoc, ...]:
    return tuple(_collapse_fno_doc_to_root_only(doc) for doc in docs)


def _sample_doc_index_subset(
    *,
    n_docs: int,
    fraction: float,
    seed: int,
) -> set[int]:
    subset_fraction = min(1.0, max(0.0, float(fraction)))
    if int(n_docs) <= 0 or subset_fraction <= 0.0:
        return set()
    if subset_fraction >= 1.0:
        return set(range(int(n_docs)))

    import random as _random

    rng = _random.Random(int(seed))
    n_selected = max(1, int(round(subset_fraction * float(n_docs))))
    n_selected = min(int(n_docs), int(n_selected))
    return set(rng.sample(range(int(n_docs)), k=int(n_selected)))


ScheduleName = str


def _markov_equiv_class(target: Tuple[float, ...]) -> Tuple[int, ...]:
    """Default Markov DGP equivalence class: (rounded_count, first_regime, last_regime)."""
    return (int(np.rint(float(target[0]))), int(target[1]), int(target[2]))


def _fno_doc_structure_key(doc: _FNOCountDoc) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    return (
        tuple(int(length) for length in doc.leaf_token_lengths),
        tuple(int(length) for length in doc.merge_token_lengths),
    )


# Default equiv class function for backward compatibility.
EquivClassFn = Callable[[Tuple[float, ...]], Hashable]


def _balanced_exact_sketch_targets(
    *,
    leaf_counts: Sequence[float],
    leaf_first_regimes: Sequence[int],
    leaf_last_regimes: Sequence[int],
) -> Dict[str, Tuple[Tuple[float, int, int], ...]]:
    if len(leaf_counts) != len(leaf_first_regimes) or len(leaf_counts) != len(leaf_last_regimes):
        raise ValueError("leaf count/first/last targets must align")
    leaf_targets = tuple(
        (float(count), int(first), int(last))
        for count, first, last in zip(
            list(leaf_counts),
            list(leaf_first_regimes),
            list(leaf_last_regimes),
        )
    )
    merge_targets: List[Tuple[float, int, int]] = []
    merge_join_bits: List[int] = []
    cur = list(leaf_targets)
    while len(cur) > 1:
        nxt: List[Tuple[float, int, int]] = []
        i = 0
        while i < len(cur):
            if i + 1 >= len(cur):
                nxt.append(cur[i])
                i += 1
                continue
            left_count, left_first, left_last = cur[i]
            right_count, right_first, right_last = cur[i + 1]
            join_bit = 0 if int(left_last) == int(right_first) else 1
            merged = (
                float(left_count) + float(right_count) + float(join_bit),
                int(left_first),
                int(right_last),
            )
            merge_targets.append(merged)
            merge_join_bits.append(int(join_bit))
            nxt.append(merged)
            i += 2
        cur = nxt
    return {
        "leaf": tuple(leaf_targets),
        "merge": tuple(merge_targets),
        "merge_join_bits": tuple(int(value) for value in merge_join_bits),
        "root": tuple(cur),
    }


@lru_cache(maxsize=65_536)
def _cached_fused_doc_targets(doc: _FNOCountDoc) -> _CachedFusedDocTargets:
    target_n_leaves = int(len(doc.leaf_token_ids))
    leaf_count_targets = torch.tensor(
        tuple(float(value) for value in doc.leaf_counts),
        dtype=torch.float32,
    )
    leaf_first_targets = torch.tensor(
        tuple(int(value) for value in doc.leaf_first_regimes),
        dtype=torch.long,
    )
    leaf_last_targets = torch.tensor(
        tuple(int(value) for value in doc.leaf_last_regimes),
        dtype=torch.long,
    )
    leaf_valid_mask = torch.ones((int(target_n_leaves),), dtype=torch.bool)

    merge_count_values = tuple(float(value) for value in doc.merge_counts_balanced)
    merge_first_values: List[int] = []
    merge_last_values: List[int] = []
    cur_first = [int(value) for value in doc.leaf_first_regimes]
    cur_last = [int(value) for value in doc.leaf_last_regimes]
    while len(cur_first) > 1:
        next_first: List[int] = []
        next_last: List[int] = []
        idx = 0
        while idx < len(cur_first):
            if idx + 1 >= len(cur_first):
                next_first.append(int(cur_first[idx]))
                next_last.append(int(cur_last[idx]))
                idx += 1
                continue
            merge_first_values.append(int(cur_first[idx]))
            merge_last_values.append(int(cur_last[idx + 1]))
            next_first.append(int(cur_first[idx]))
            next_last.append(int(cur_last[idx + 1]))
            idx += 2
        cur_first = next_first
        cur_last = next_last

    merge_count_targets = torch.tensor(merge_count_values, dtype=torch.float32)
    merge_first_targets = torch.tensor(tuple(merge_first_values), dtype=torch.long)
    merge_last_targets = torch.tensor(tuple(merge_last_values), dtype=torch.long)
    merge_valid_mask = torch.ones((int(merge_count_targets.shape[0]),), dtype=torch.bool)
    root_count_targets = torch.tensor((float(doc.root_count),), dtype=torch.float32)
    root_first_targets = torch.tensor(
        (int(doc.leaf_first_regimes[0]) if doc.leaf_first_regimes else 0,),
        dtype=torch.long,
    )
    root_last_targets = torch.tensor(
        (int(doc.leaf_last_regimes[-1]) if doc.leaf_last_regimes else 0,),
        dtype=torch.long,
    )
    node_count_targets = torch.cat(
        (leaf_count_targets, merge_count_targets, root_count_targets),
        dim=0,
    )
    node_count_keys = torch.round(node_count_targets).to(dtype=torch.long)
    node_first_targets = torch.cat(
        (leaf_first_targets, merge_first_targets, root_first_targets),
        dim=0,
    )
    node_last_targets = torch.cat(
        (leaf_last_targets, merge_last_targets, root_last_targets),
        dim=0,
    )
    node_valid_mask = torch.cat(
        (
            leaf_valid_mask,
            merge_valid_mask,
            torch.ones((1,), dtype=torch.bool),
        ),
        dim=0,
    )
    return _CachedFusedDocTargets(
        leaf_count_targets_cpu=leaf_count_targets,
        leaf_first_targets_cpu=leaf_first_targets,
        leaf_last_targets_cpu=leaf_last_targets,
        leaf_valid_mask_cpu=leaf_valid_mask,
        merge_count_targets_cpu=merge_count_targets,
        merge_first_targets_cpu=merge_first_targets,
        merge_last_targets_cpu=merge_last_targets,
        merge_valid_mask_cpu=merge_valid_mask,
        node_count_targets_cpu=node_count_targets,
        node_count_keys_cpu=node_count_keys,
        node_first_targets_cpu=node_first_targets,
        node_last_targets_cpu=node_last_targets,
        node_valid_mask_cpu=node_valid_mask,
    )


@lru_cache(maxsize=262_144)
def _cached_fused_doc_targets_for_target_leaves(
    doc: _FNOCountDoc,
    target_n_leaves: int,
) -> _CachedFusedDocTargets:
    actual_n_leaves = int(len(doc.leaf_token_ids))
    resolved_target_n_leaves = int(target_n_leaves)
    if resolved_target_n_leaves < actual_n_leaves:
        raise ValueError(
            f"target_n_leaves must be >= actual leaf count ({resolved_target_n_leaves} < {actual_n_leaves})"
        )
    if resolved_target_n_leaves == actual_n_leaves:
        return _cached_fused_doc_targets(doc)

    leaf_count_targets = torch.zeros((int(resolved_target_n_leaves),), dtype=torch.float32)
    leaf_first_targets = torch.zeros((int(resolved_target_n_leaves),), dtype=torch.long)
    leaf_last_targets = torch.zeros((int(resolved_target_n_leaves),), dtype=torch.long)
    leaf_valid_mask = torch.zeros((int(resolved_target_n_leaves),), dtype=torch.bool)
    if actual_n_leaves > 0:
        leaf_count_targets[:actual_n_leaves] = torch.tensor(
            tuple(float(value) for value in doc.leaf_counts),
            dtype=torch.float32,
        )
        leaf_first_targets[:actual_n_leaves] = torch.tensor(
            tuple(int(value) for value in doc.leaf_first_regimes),
            dtype=torch.long,
        )
        leaf_last_targets[:actual_n_leaves] = torch.tensor(
            tuple(int(value) for value in doc.leaf_last_regimes),
            dtype=torch.long,
        )
        leaf_valid_mask[:actual_n_leaves] = True

    current_targets: List[Tuple[float, int, int]] = [
        (
            float(doc.leaf_counts[idx]),
            int(doc.leaf_first_regimes[idx]),
            int(doc.leaf_last_regimes[idx]),
        )
        for idx in range(actual_n_leaves)
    ] + [(0.0, 0, 0) for _ in range(int(resolved_target_n_leaves - actual_n_leaves))]
    current_valid: List[bool] = [True for _ in range(actual_n_leaves)] + [
        False for _ in range(int(resolved_target_n_leaves - actual_n_leaves))
    ]
    merge_count_values: List[float] = []
    merge_first_values: List[int] = []
    merge_last_values: List[int] = []
    merge_valid_values: List[bool] = []
    while len(current_targets) > 1:
        next_targets: List[Tuple[float, int, int]] = []
        next_valid: List[bool] = []
        idx = 0
        while idx < len(current_targets):
            if idx + 1 >= len(current_targets):
                next_targets.append(current_targets[idx])
                next_valid.append(bool(current_valid[idx]))
                idx += 1
                continue
            left_count, left_first, left_last = current_targets[idx]
            right_count, right_first, right_last = current_targets[idx + 1]
            left_valid = bool(current_valid[idx])
            right_valid = bool(current_valid[idx + 1])
            if left_valid and right_valid:
                join_bit = 0 if int(left_last) == int(right_first) else 1
                merged = (
                    float(left_count) + float(right_count) + float(join_bit),
                    int(left_first),
                    int(right_last),
                )
                merge_count_values.append(float(merged[0]))
                merge_first_values.append(int(merged[1]))
                merge_last_values.append(int(merged[2]))
                merge_valid_values.append(True)
                next_targets.append(merged)
                next_valid.append(True)
            elif left_valid or right_valid:
                carried = current_targets[idx] if left_valid else current_targets[idx + 1]
                merge_count_values.append(float(carried[0]))
                merge_first_values.append(int(carried[1]))
                merge_last_values.append(int(carried[2]))
                merge_valid_values.append(False)
                next_targets.append(carried)
                next_valid.append(True)
            else:
                merge_count_values.append(0.0)
                merge_first_values.append(0)
                merge_last_values.append(0)
                merge_valid_values.append(False)
                next_targets.append((0.0, 0, 0))
                next_valid.append(False)
            idx += 2
        current_targets = next_targets
        current_valid = next_valid

    merge_count_targets = torch.tensor(tuple(merge_count_values), dtype=torch.float32)
    merge_first_targets = torch.tensor(tuple(merge_first_values), dtype=torch.long)
    merge_last_targets = torch.tensor(tuple(merge_last_values), dtype=torch.long)
    merge_valid_mask = torch.tensor(tuple(bool(value) for value in merge_valid_values), dtype=torch.bool)
    root_target = current_targets[0] if current_targets else (0.0, 0, 0)
    root_valid = bool(current_valid[0]) if current_valid else False
    root_count_targets = torch.tensor((float(root_target[0]),), dtype=torch.float32)
    root_first_targets = torch.tensor((int(root_target[1]),), dtype=torch.long)
    root_last_targets = torch.tensor((int(root_target[2]),), dtype=torch.long)
    node_count_targets = torch.cat(
        (leaf_count_targets, merge_count_targets, root_count_targets),
        dim=0,
    )
    node_count_keys = torch.round(node_count_targets).to(dtype=torch.long)
    node_first_targets = torch.cat(
        (leaf_first_targets, merge_first_targets, root_first_targets),
        dim=0,
    )
    node_last_targets = torch.cat(
        (leaf_last_targets, merge_last_targets, root_last_targets),
        dim=0,
    )
    node_valid_mask = torch.cat(
        (
            leaf_valid_mask,
            merge_valid_mask,
            torch.tensor((bool(root_valid),), dtype=torch.bool),
        ),
        dim=0,
    )
    return _CachedFusedDocTargets(
        leaf_count_targets_cpu=leaf_count_targets,
        leaf_first_targets_cpu=leaf_first_targets,
        leaf_last_targets_cpu=leaf_last_targets,
        leaf_valid_mask_cpu=leaf_valid_mask,
        merge_count_targets_cpu=merge_count_targets,
        merge_first_targets_cpu=merge_first_targets,
        merge_last_targets_cpu=merge_last_targets,
        merge_valid_mask_cpu=merge_valid_mask,
        node_count_targets_cpu=node_count_targets,
        node_count_keys_cpu=node_count_keys,
        node_first_targets_cpu=node_first_targets,
        node_last_targets_cpu=node_last_targets,
        node_valid_mask_cpu=node_valid_mask,
    )


def _padded_merge_feature_targets_for_valid_mask(
    merge_targets: Sequence[Any],
    merge_valid_mask: torch.Tensor,
) -> Tuple[Any | None, ...]:
    valid_values = merge_valid_mask.detach().cpu().reshape(-1).tolist()
    padded_targets: List[Any | None] = []
    merge_target_idx = 0
    for is_valid in valid_values:
        if bool(is_valid) and merge_target_idx < len(merge_targets):
            padded_targets.append(merge_targets[int(merge_target_idx)])
            merge_target_idx += 1
        else:
            padded_targets.append(None)
    return tuple(padded_targets)


def _all_or_sampled_indices(
    total: int,
    sampled_indices: Optional[set],
) -> Tuple[int, ...]:
    if int(total) <= 0:
        return tuple()
    if sampled_indices is None:
        return tuple(range(int(total)))
    return tuple(
        int(idx)
        for idx in sorted(int(value) for value in sampled_indices)
        if 0 <= int(idx) < int(total)
    )


def _theorem_feature_metadata_sequences_from_fno_doc(
    doc: _FNOCountDoc,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    leaf_metadata = tuple(
        {
            "span_length": int(length),
            "leaf_span_count": 1,
            "root_token_length": int(doc.n_tokens),
        }
        for length in list(doc.leaf_token_lengths)
    )
    merge_metadata = tuple(
        {
            "span_length": int(token_length),
            "leaf_span_count": int(leaf_span_count),
            "root_token_length": int(doc.n_tokens),
        }
        for token_length, leaf_span_count in zip(
            list(doc.merge_token_lengths),
            list(doc.merge_sizes_balanced),
        )
    )
    root_metadata = (
        {
            "span_length": int(doc.n_tokens),
            "leaf_span_count": int(len(doc.leaf_token_ids)),
            "root_token_length": int(doc.n_tokens),
        },
    )
    return leaf_metadata, merge_metadata, root_metadata


def _fit_linear_regression_probe_local(
    features: np.ndarray,
    targets: np.ndarray,
) -> Optional[np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).reshape(-1, 1)
    if x.ndim != 2 or y.shape[0] != x.shape[0] or x.shape[0] <= 0:
        return None
    design = np.concatenate(
        [x, np.ones((x.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    weights, *_ = np.linalg.lstsq(design, y, rcond=None)
    return np.asarray(weights, dtype=np.float64)


def _predict_linear_regression_probe_local(
    weights: Optional[np.ndarray],
    features: np.ndarray,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if weights is None or x.ndim != 2 or x.shape[0] <= 0:
        return np.zeros((int(max(0, x.shape[0] if x.ndim == 2 else 0)),), dtype=np.float64)
    design = np.concatenate(
        [x, np.ones((x.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    preds = design @ np.asarray(weights, dtype=np.float64)
    return np.asarray(preds.reshape(-1), dtype=np.float64)


def _fit_linear_classifier_probe_local(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    n_classes: int,
) -> Optional[np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or y.shape[0] != x.shape[0] or x.shape[0] <= 0:
        return None
    if int(n_classes) <= 0:
        return None
    design = np.concatenate(
        [x, np.ones((x.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    one_hot = np.zeros((x.shape[0], int(n_classes)), dtype=np.float64)
    valid = (y >= 0) & (y < int(n_classes))
    one_hot[np.arange(x.shape[0])[valid], y[valid]] = 1.0
    weights, *_ = np.linalg.lstsq(design, one_hot, rcond=None)
    return np.asarray(weights, dtype=np.float64)


def _predict_linear_classifier_probe_local(
    weights: Optional[np.ndarray],
    features: np.ndarray,
    *,
    n_classes: int,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if (
        weights is None
        or x.ndim != 2
        or x.shape[0] <= 0
        or int(n_classes) <= 0
    ):
        return np.zeros((int(max(0, x.shape[0] if x.ndim == 2 else 0)),), dtype=np.int64)
    design = np.concatenate(
        [x, np.ones((x.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    logits = design @ np.asarray(weights, dtype=np.float64)
    return np.asarray(np.argmax(logits, axis=1), dtype=np.int64)


@torch.inference_mode()
def eval_scorefiber_root_probe_metrics(
    model: "FNOCountSketch",
    train_docs: Sequence[_FNOCountDoc],
    eval_docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
) -> Dict[str, float]:
    if not bool(getattr(model, "use_factorized_score_fiber_surface", False)):
        return {
            "root_summary_probe_accuracy_full_state": float("nan"),
            "root_summary_probe_accuracy_score_slice": float("nan"),
            "root_summary_probe_n_classes": 0.0,
            "root_summary_probe_n_train": 0.0,
            "root_summary_probe_n_eval": 0.0,
        }

    def _collect(
        docs: Sequence[_FNOCountDoc],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feature_rows: List[np.ndarray] = []
        score_rows: List[np.ndarray] = []
        label_keys: List[Hashable] = []
        for doc in docs:
            leaf_states = _encode_leaf_state_batch(model, doc.leaf_token_ids, device=device)
            root_state, _merge_states = model._merge_states(
                leaf_states,
                schedule="balanced",
                collect_merge_states=False,
            )
            exact_targets = _balanced_exact_sketch_targets(
                leaf_counts=doc.leaf_counts,
                leaf_first_regimes=doc.leaf_first_regimes,
                leaf_last_regimes=doc.leaf_last_regimes,
            )
            leaf_metadata, merge_metadata, root_metadata = (
                _theorem_feature_metadata_sequences_from_fno_doc(doc)
            )
            feature_targets = theorem_feature_targets_from_markov_exact_targets(
                adapter=model.theorem_feature_adapter,
                exact_targets=exact_targets,
                leaf_metadata=leaf_metadata,
                merge_metadata=merge_metadata,
                root_metadata=root_metadata,
            )
            if not feature_targets.root:
                continue
            theorem_feature = model.theorem_feature_from_state(root_state).detach()
            score_feature = model._score_feature_from_theorem_feature(theorem_feature).detach()
            feature_rows.append(
                np.asarray(theorem_feature.cpu().reshape(-1).numpy(), dtype=np.float64)
            )
            score_rows.append(
                np.asarray(score_feature.cpu().reshape(-1).numpy(), dtype=np.float64)
            )
            label_keys.append(model.theorem_feature_adapter.diagnostic_key(feature_targets.root[0]))
        if not feature_rows:
            return (
                np.zeros((0, 0), dtype=np.float64),
                np.zeros((0, 0), dtype=np.float64),
                np.zeros((0,), dtype=np.int64),
            )
        key_index: Dict[Hashable, int] = {}
        labels = np.asarray(
            [int(key_index.setdefault(key, len(key_index))) for key in label_keys],
            dtype=np.int64,
        )
        return (
            np.asarray(feature_rows, dtype=np.float64),
            np.asarray(score_rows, dtype=np.float64),
            labels,
        )

    train_features, train_score_features, train_labels = _collect(train_docs)
    eval_features, eval_score_features, eval_labels = _collect(eval_docs)
    n_classes = int(len(set(train_labels.tolist())))
    if (
        train_features.shape[0] <= 0
        or eval_features.shape[0] <= 0
        or int(n_classes) <= 1
    ):
        return {
            "root_summary_probe_accuracy_full_state": float("nan"),
            "root_summary_probe_accuracy_score_slice": float("nan"),
            "root_summary_probe_n_classes": float(n_classes),
            "root_summary_probe_n_train": float(train_features.shape[0]),
            "root_summary_probe_n_eval": float(eval_features.shape[0]),
        }
    full_weights = _fit_linear_classifier_probe_local(
        train_features,
        train_labels,
        n_classes=n_classes,
    )
    score_weights = _fit_linear_classifier_probe_local(
        train_score_features,
        train_labels,
        n_classes=n_classes,
    )
    full_preds = _predict_linear_classifier_probe_local(
        full_weights,
        eval_features,
        n_classes=n_classes,
    )
    score_preds = _predict_linear_classifier_probe_local(
        score_weights,
        eval_score_features,
        n_classes=n_classes,
    )
    return {
        "root_summary_probe_accuracy_full_state": float(
            np.mean((full_preds == eval_labels).astype(np.float64))
        ),
        "root_summary_probe_accuracy_score_slice": float(
            np.mean((score_preds == eval_labels).astype(np.float64))
        ),
        "root_summary_probe_n_classes": float(n_classes),
        "root_summary_probe_n_train": float(train_features.shape[0]),
        "root_summary_probe_n_eval": float(eval_features.shape[0]),
    }


def _summary_component_metrics_local(
    *,
    count_preds: np.ndarray,
    first_preds: np.ndarray,
    last_preds: np.ndarray,
    count_targets: np.ndarray,
    first_targets: np.ndarray,
    last_targets: np.ndarray,
) -> Dict[str, float]:
    cp = np.asarray(count_preds, dtype=np.float64)
    fp = np.asarray(first_preds, dtype=np.int64)
    lp = np.asarray(last_preds, dtype=np.int64)
    ct = np.asarray(count_targets, dtype=np.int64)
    ft = np.asarray(first_targets, dtype=np.int64)
    lt = np.asarray(last_targets, dtype=np.int64)
    if ct.size <= 0:
        return {
            "count_mae": float("nan"),
            "first_accuracy": float("nan"),
            "last_accuracy": float("nan"),
            "exact_summary_match_rate": float("nan"),
        }
    rounded_counts = np.rint(cp).astype(np.int64)
    return {
        "count_mae": float(np.mean(np.abs(cp - ct.astype(np.float64)))),
        "first_accuracy": float(np.mean((fp == ft).astype(np.float64))),
        "last_accuracy": float(np.mean((lp == lt).astype(np.float64))),
        "exact_summary_match_rate": float(
            np.mean(
                (
                    (rounded_counts == ct)
                    & (fp == ft)
                    & (lp == lt)
                ).astype(np.float64)
            )
        ),
    }


@dataclass(frozen=True)
class SummaryFieldSpec:
    name: str
    loss_kind: str


@dataclass(frozen=True)
class SummarySpec:
    name: str
    fields: Tuple[SummaryFieldSpec, ...]


LEAN_MARKOV_COUNT_SKETCH_REF = "MarkovCountSketch"
LEAN_SKETCH_CODEC_EXACT_ASSUMPTIONS_REF = "SketchCodecExactAssumptions"
LEAN_APPROX_BUNDLE_OF_NODEWISE_REF = "approx_bundle_of_nodewise"
LEAN_MARKOV_PATH_SUPPORT_EXACT_REF = "markov_path_state_exact_on_support_of_contract"
LEAN_MARKOV_PATH_COUNT_SUPPORT_EXACT_REF = (
    "markov_path_changepoint_count_exact_on_support_of_contract"
)
LEAN_MARKOV_COUNT_ONLY_INVALID_REF = "markov_countOnly_not_exact_on_all_trees"
LEAN_MARKOV_SUFFICIENCY_DECODER_REF = "markov_count_query_sufficient_has_decoder"
LEAN_MARKOV_OBSERVED_TOKEN_RECOVERABILITY_REF = (
    "piecewise_disjoint_palette_observed_tokens_recover_latent_path"
)
LEAN_MARKOV_OBSERVED_TOKEN_EXACT_SKETCH_REF = (
    "piecewise_disjoint_palette_observed_tokens_recover_exact_sketch"
)
LEAN_MARKOV_ZERO_BAYES_ERROR_REF = "piecewise_disjoint_palette_zero_bayes_error"
LEAN_MARKOV_REPRESENTATION_EXACT_PASS_REF = (
    "markov_representation_exact_recovery_implies_query_sufficient"
)
LEAN_MARKOV_REPRESENTATION_ZERO_ROOT_COUNT_ERROR_REF = (
    "markov_representation_exact_recovery_zero_root_count_error"
)
LEAN_MARKOV_REPRESENTATION_COUNT_TRANSPORT_REF = (
    "markov_count_error_le_exact_sketch_error"
)
LEAN_MARKOV_RUNTIME_AUDIT_STOCHASTIC_APPROX_REF = (
    "runtime_audited_markov_path_stochastic_approx_local_laws"
)


@dataclass(frozen=True)
class DecodedMarkovSketch:
    count: torch.Tensor
    first: torch.Tensor
    last: torch.Tensor


@dataclass(frozen=True)
class MarkovSketchCodecContract:
    model: "FNOCountSketch"
    lean_summary_ref: str = LEAN_MARKOV_COUNT_SKETCH_REF
    lean_codec_ref: str = LEAN_SKETCH_CODEC_EXACT_ASSUMPTIONS_REF
    lean_bundle_ref: str = LEAN_APPROX_BUNDLE_OF_NODEWISE_REF

    def decode(self, latent: torch.Tensor) -> DecodedMarkovSketch:
        return self.model.decode_markov_codec(latent)

    def join(self, left_last: torch.Tensor, right_first: torch.Tensor) -> torch.Tensor:
        left = torch.as_tensor(left_last, device=self.model.root_count_class_values.device)
        right = torch.as_tensor(right_first, device=left.device)
        return (left.to(torch.long) != right.to(torch.long)).to(torch.float32)

    def compose(
        self,
        left: DecodedMarkovSketch,
        right: DecodedMarkovSketch,
    ) -> DecodedMarkovSketch:
        join = self.join(left.last, right.first).to(left.count.dtype)
        return DecodedMarkovSketch(
            count=left.count + right.count + join,
            first=left.first.to(torch.long),
            last=right.last.to(torch.long),
        )

    def reencode(self, decoded: DecodedMarkovSketch) -> torch.Tensor:
        count = decoded.count
        if count.ndim == 0:
            count = count.unsqueeze(0)
        else:
            count = count.unsqueeze(-1)
        first = F.one_hot(
            decoded.first.to(torch.long),
            num_classes=int(self.model.n_regimes),
        ).to(dtype=count.dtype, device=count.device)
        last = F.one_hot(
            decoded.last.to(torch.long),
            num_classes=int(self.model.n_regimes),
        ).to(dtype=count.dtype, device=count.device)
        summary = torch.cat([count / float(self.model.target_scale), first, last], dim=-1)
        encoded = self.model.encode_summary(summary)
        if decoded.count.ndim == 0:
            return encoded.squeeze(0)
        return encoded


MARKOV_COUNT_SKETCH_SPEC = SummarySpec(
    name=MARKOV_COUNT_SKETCH_SUMMARY_SPEC,
    fields=(
        SummaryFieldSpec(name="count", loss_kind="mse"),
        SummaryFieldSpec(name="first", loss_kind="cross_entropy"),
        SummaryFieldSpec(name="last", loss_kind="cross_entropy"),
    ),
)


class PrototypeClassifier(nn.Module):
    """Cosine-similarity classifier with learned class prototypes."""

    def __init__(
        self,
        *,
        input_dim: int,
        n_classes: int,
        init_temperature: float = 0.7,
    ) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(
            torch.randn(int(n_classes), int(input_dim), dtype=torch.float32)
        )
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(float(init_temperature), dtype=torch.float32))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False
        x_unit = F.normalize(x, dim=-1, eps=1e-6)
        proto_unit = F.normalize(self.prototypes, dim=-1, eps=1e-6)
        temperature = torch.exp(self.log_temperature).clamp(min=1e-3)
        logits = torch.matmul(x_unit, proto_unit.transpose(0, 1)) / temperature
        return logits.squeeze(0) if squeezed else logits

    @staticmethod
    def logits_entropy(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return -torch.sum(probs * torch.log(probs.clamp(min=1e-12)), dim=-1)

    @staticmethod
    def logits_margin(logits: torch.Tensor) -> torch.Tensor:
        top2 = torch.topk(logits, k=min(2, int(logits.shape[-1])), dim=-1).values
        if int(top2.shape[-1]) == 1:
            return top2[..., 0]
        return top2[..., 0] - top2[..., 1]

    @staticmethod
    def distribution_entropy(probs: torch.Tensor) -> torch.Tensor:
        return -torch.sum(probs * torch.log(probs.clamp(min=1e-12)), dim=-1)

    @staticmethod
    def distribution_margin(probs: torch.Tensor) -> torch.Tensor:
        top2 = torch.topk(probs, k=min(2, int(probs.shape[-1])), dim=-1).values
        if int(top2.shape[-1]) == 1:
            return top2[..., 0]
        return top2[..., 0] - top2[..., 1]


def _resolve_summary_spec(summary_spec_name: str) -> SummarySpec | None:
    normalized = str(summary_spec_name or "").strip()
    if not normalized:
        return None
    if normalized == MARKOV_COUNT_SKETCH_SUMMARY_SPEC:
        return MARKOV_COUNT_SKETCH_SPEC
    raise ValueError(
        f"unsupported summary_spec_name={summary_spec_name!r}; "
        f"expected {MARKOV_COUNT_SKETCH_SUMMARY_SPEC!r}"
    )


class FNOCountSketch(nn.Module):
    """FNO-backed tree-merge operator with local law interface.

    Each leaf span's raw tokens are embedded and processed by a 1D FNO,
    then pooled to produce a state vector. States are merged bottom-up
    using a learned merger (same protocol as AdditiveCountSketch).

    Satisfies the same ``forward_doc``-style interface for local law
    supervision (C1 leaf, C2 count drift, C3 merge smoothness).
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        leaf_tokens: int,
        state_dim: int,
        hidden_dim: int,
        target_scale: float,
        n_regimes: int,
        doc_sequence_class_values: Sequence[int] = (),
        fno_width: int = 64,
        fno_n_modes: int = 8,
        fno_n_layers: int = 2,
        leaf_fno_pooling: str = "mean",
        root_supervision_kind: str = "mse",
        root_count_class_values: Sequence[int] = (),
        aligned_sketch_surface: str = "",
        summary_spec_name: str = "",
        slot_count: int = 0,
        join_bit_weight: float = 0.0,
        endpoint_loss_scale: float = 1.0,
        task_head_mode: str = "full_state_scalar",
        summary_spec_root_mode: str = "task_split_ablation",
        theorem_surface_mode: str = "slotwise",
        theorem_count_head_mode: str = "scalar_mse",
        theorem_count_ordinal_weight: float = 1.0,
        theorem_count_scalar_aux_weight: float = 0.25,
        theorem_count_threshold_balance: bool = True,
        theorem_feature_dim: int = 48,
        theorem_feature_hidden_dim: int = 256,
        theorem_score_dim: int = 0,
        theorem_fiber_dim: int = 0,
        theorem_aux_dim: int = 0,
        merge_hidden_dim: int = 0,
        score_merge_mode: str = "gated_affine",
        phi_alignment_loss: str = "cosine_mse",
        theorem_feature_adapter: str = DEFAULT_THEOREM_FEATURE_ADAPTER,
        theorem_pair_same_threshold: float | None = None,
        theorem_pair_diff_threshold: float | None = None,
        theorem_count_dim: int = 0,
        theorem_first_dim: int = 0,
        theorem_last_dim: int = 0,
        c2_mode: str = "reconstruction",
        equiv_class_fn: Optional[EquivClassFn] = None,
        oracle_metric: Optional[Any] = None,
        oracle_same_threshold: float = 0.0,
        oracle_diff_threshold: float = 0.0,
        tree_model_version: str = "legacy",
        runtime_count_discretization: str = "continuous",
    ) -> None:
        super().__init__()
        if _NeuralOpFNO is None:
            raise ImportError(_INSTALL_MSG)
        normalized_root_supervision = (
            str(root_supervision_kind or "mse").strip().lower() or "mse"
        )
        if normalized_root_supervision not in VALID_TREE_ROOT_SUPERVISION_KINDS:
            raise ValueError(
                "unsupported root_supervision_kind="
                f"{root_supervision_kind!r}; expected one of "
                f"{VALID_TREE_ROOT_SUPERVISION_KINDS}"
        )
        self.target_scale = float(target_scale)
        self.n_regimes = int(n_regimes)
        self.requested_state_dim = int(state_dim)
        self.state_dim = int(self.requested_state_dim)
        self.default_head = "count"
        self.tree_model_version = normalize_tree_model_version(tree_model_version)
        self.runtime_count_discretization = (
            str(runtime_count_discretization or "continuous").strip().lower()
            or "continuous"
        )
        if self.runtime_count_discretization not in {
            "continuous",
            "none",
            "st_round",
        }:
            raise ValueError(
                "unsupported runtime_count_discretization="
                f"{runtime_count_discretization!r}; expected one of "
                "('continuous', 'none', 'st_round')"
            )
        if self.runtime_count_discretization == "none":
            self.runtime_count_discretization = "continuous"
        self.leaf_tokens = int(leaf_tokens)
        self.pad_id = int(vocab_size)
        self.root_supervision_kind = normalized_root_supervision
        self.aligned_sketch_surface = str(aligned_sketch_surface or "").strip()
        self.summary_spec_name = str(summary_spec_name or "").strip()
        self.slot_count = int(slot_count)
        self.join_bit_weight = float(join_bit_weight)
        self.endpoint_loss_scale = float(endpoint_loss_scale)
        self.task_head_mode = str(task_head_mode or "full_state_scalar").strip().lower()
        self.summary_spec_root_mode = (
            str(summary_spec_root_mode or "task_split_ablation").strip().lower()
            or "task_split_ablation"
        )
        self.theorem_surface_mode = (
            str(theorem_surface_mode or "slotwise").strip().lower() or "slotwise"
        )
        self.theorem_count_head_mode = (
            str(theorem_count_head_mode or "scalar_mse").strip().lower()
            or "scalar_mse"
        )
        self.theorem_count_ordinal_weight = float(theorem_count_ordinal_weight)
        self.theorem_count_scalar_aux_weight = float(theorem_count_scalar_aux_weight)
        self.theorem_count_threshold_balance = bool(theorem_count_threshold_balance)
        self.shared_theorem_feature_dim = int(theorem_feature_dim)
        self.shared_theorem_feature_hidden_dim = int(theorem_feature_hidden_dim)
        self.factorized_score_dim = int(theorem_score_dim)
        self.factorized_fiber_dim = int(theorem_fiber_dim)
        self.factorized_aux_dim = int(theorem_aux_dim)
        self.score_merge_mode = (
            str(score_merge_mode or "gated_affine").strip().lower() or "gated_affine"
        )
        self.phi_alignment_loss = (
            str(phi_alignment_loss or "cosine_mse").strip().lower() or "cosine_mse"
        )
        self.theorem_feature_adapter_name = (
            str(
                theorem_feature_adapter
                or self.summary_spec_name
                or DEFAULT_THEOREM_FEATURE_ADAPTER
            )
            .strip()
            .lower()
            or DEFAULT_THEOREM_FEATURE_ADAPTER
        )
        self.theorem_pair_same_threshold = (
            None
            if theorem_pair_same_threshold is None
            else float(theorem_pair_same_threshold)
        )
        self.theorem_pair_diff_threshold = (
            None
            if theorem_pair_diff_threshold is None
            else float(theorem_pair_diff_threshold)
        )
        self.theorem_feature_adapter: TheoremFeatureAdapter = resolve_theorem_feature_adapter(
            self.theorem_feature_adapter_name
        )
        if self.summary_spec_root_mode not in VALID_TREE_SUMMARY_SPEC_ROOT_MODES:
            raise ValueError(
                "unsupported summary_spec_root_mode="
                f"{summary_spec_root_mode!r}; expected one of "
                f"{VALID_TREE_SUMMARY_SPEC_ROOT_MODES}"
            )
        if self.theorem_surface_mode not in VALID_TREE_THEOREM_SURFACE_MODES:
            raise ValueError(
                "unsupported theorem_surface_mode="
                f"{theorem_surface_mode!r}; expected one of "
                f"{VALID_TREE_THEOREM_SURFACE_MODES}"
            )
        if self.theorem_count_head_mode not in VALID_TREE_THEOREM_COUNT_HEAD_MODES:
            raise ValueError(
                "unsupported theorem_count_head_mode="
                f"{theorem_count_head_mode!r}; expected one of "
                f"{VALID_TREE_THEOREM_COUNT_HEAD_MODES}"
            )
        if float(self.theorem_count_ordinal_weight) < 0.0:
            raise ValueError("theorem_count_ordinal_weight must be non-negative")
        if float(self.theorem_count_scalar_aux_weight) < 0.0:
            raise ValueError("theorem_count_scalar_aux_weight must be non-negative")
        if int(self.shared_theorem_feature_dim) <= 0:
            raise ValueError("theorem_feature_dim must be positive")
        if int(self.shared_theorem_feature_hidden_dim) <= 0:
            raise ValueError("theorem_feature_hidden_dim must be positive")
        if self.score_merge_mode not in VALID_TREE_SCORE_MERGE_MODES:
            raise ValueError(
                "unsupported score_merge_mode="
                f"{score_merge_mode!r}; expected one of {VALID_TREE_SCORE_MERGE_MODES}"
            )
        if self.phi_alignment_loss not in {"cosine_mse"}:
            raise ValueError(
                "unsupported phi_alignment_loss="
                f"{phi_alignment_loss!r}; expected 'cosine_mse'"
            )
        _valid_c2_modes = {"reconstruction", "fiber"}
        _c2_mode = str(c2_mode or "reconstruction").strip().lower() or "reconstruction"
        if _c2_mode not in _valid_c2_modes:
            raise ValueError(
                f"c2_mode must be one of {_valid_c2_modes}; got {c2_mode!r}"
            )
        self.c2_mode = _c2_mode
        self.equiv_class_fn: EquivClassFn = equiv_class_fn or self._theorem_equiv_class
        self.oracle_metric = oracle_metric  # OracleMetricSpace | None
        self.oracle_same_threshold = float(oracle_same_threshold)
        self.oracle_diff_threshold = float(oracle_diff_threshold)
        self.summary_spec = _resolve_summary_spec(self.summary_spec_name)
        self.use_decoded_markov_sketch = (
            self.aligned_sketch_surface == DECODED_MARKOV_SKETCH_SURFACE
        )
        self.use_summary_spec = self.summary_spec is not None
        self.use_markov_summary_spec = bool(
            self.summary_spec is not None
            and self.summary_spec.name == MARKOV_COUNT_SKETCH_SUMMARY_SPEC
        )
        self.use_carrier_projection_surface = bool(
            self.theorem_surface_mode == "carrier_projection"
        )
        self.use_opaque_carrier_exact_sketch_surface = bool(
            self.theorem_surface_mode == "opaque_carrier_exact_sketch"
        )
        self.use_direct_markov_sketch_slots = bool(
            self.use_carrier_projection_surface
            or self.use_opaque_carrier_exact_sketch_surface
        )
        self.use_shared_theorem_surface = bool(
            self.theorem_surface_mode
            in (
                "shared_bottleneck",
                "shared_feature",
                "shared_feature_adapters",
                "factorized_score_fiber",
            )
        )
        self.use_shared_bottleneck_surface = bool(
            self.theorem_surface_mode == "shared_bottleneck"
        )
        self.use_shared_feature_surface = bool(
            self.theorem_surface_mode == "shared_feature"
        )
        self.use_shared_feature_adapters_surface = bool(
            self.theorem_surface_mode == "shared_feature_adapters"
        )
        self.use_factorized_score_fiber_surface = bool(
            self.theorem_surface_mode == "factorized_score_fiber"
        )
        self.use_exact_projected_sketch_merge = bool(
            self.score_merge_mode == "exact_projected_sketch"
        )
        if self.use_carrier_projection_surface and not self.use_markov_summary_spec:
            raise ValueError(
                "theorem_surface_mode='carrier_projection' requires "
                "summary_spec_name='markov_count_sketch'"
            )
        if self.use_exact_projected_sketch_merge and not self.use_markov_summary_spec:
            raise ValueError(
                "score_merge_mode='exact_projected_sketch' requires "
                "summary_spec_name='markov_count_sketch'"
            )
        if self.use_opaque_carrier_exact_sketch_surface and not self.use_markov_summary_spec:
            raise ValueError(
                "theorem_surface_mode='opaque_carrier_exact_sketch' requires "
                "summary_spec_name='markov_count_sketch'"
            )
        if (
            self.use_opaque_carrier_exact_sketch_surface
            and not self.use_exact_projected_sketch_merge
        ):
            raise ValueError(
                "theorem_surface_mode='opaque_carrier_exact_sketch' requires "
                "score_merge_mode='exact_projected_sketch'"
            )
        if self.use_opaque_carrier_exact_sketch_surface:
            self.state_dim = int(self.requested_state_dim) + 1 + 2 * int(self.n_regimes)
            self.carrier_state_dim = int(self.requested_state_dim)
        else:
            self.state_dim = int(self.requested_state_dim)
            self.carrier_state_dim = 0
        requested_merge_hidden_dim = int(merge_hidden_dim)
        default_merge_hidden_dim = (
            max(32, 4 * int(self.carrier_state_dim))
            if self.use_opaque_carrier_exact_sketch_surface
            else int(hidden_dim)
        )
        self.merge_hidden_dim = int(
            requested_merge_hidden_dim
            if requested_merge_hidden_dim > 0
            else default_merge_hidden_dim
        )
        if self.use_factorized_score_fiber_surface:
            total_requested = (
                int(self.factorized_score_dim)
                + int(self.factorized_fiber_dim)
                + int(self.factorized_aux_dim)
            )
            if total_requested <= 0:
                self.factorized_score_dim = 1
                self.factorized_aux_dim = 0
                self.factorized_fiber_dim = max(
                    1,
                    int(self.shared_theorem_feature_dim) - int(self.factorized_score_dim),
                )
                total_requested = (
                    int(self.factorized_score_dim)
                    + int(self.factorized_fiber_dim)
                    + int(self.factorized_aux_dim)
                )
            if int(self.factorized_score_dim) <= 0:
                raise ValueError(
                    "factorized_score_fiber requires theorem_score_dim > 0"
                )
            if int(total_requested) != int(self.shared_theorem_feature_dim):
                raise ValueError(
                    "factorized score/fiber/aux dims must sum to theorem_feature_dim"
                )
        else:
            self.factorized_score_dim = 0
            self.factorized_fiber_dim = 0
            self.factorized_aux_dim = 0
        self.shared_feature_adapter_dim = max(
            1,
            int(self.shared_theorem_feature_dim) // 2,
        )
        self._score_feature_slice = slice(0, int(self.factorized_score_dim))
        self._fiber_feature_slice = slice(
            int(self.factorized_score_dim),
            int(self.factorized_score_dim) + int(self.factorized_fiber_dim),
        )
        self._aux_feature_slice = slice(
            int(self.factorized_score_dim) + int(self.factorized_fiber_dim),
            int(self.shared_theorem_feature_dim),
        )
        if (
            self.task_head_mode == "theorem_feature_scalar"
            and not (self.use_markov_summary_spec or self.use_shared_theorem_surface)
        ):
            raise ValueError(
                "task_head_mode='theorem_feature_scalar' requires "
                "summary_spec_name='markov_count_sketch' or a shared theorem surface"
            )
        if (
            self.summary_spec_root_mode == "factored_theorem_readout"
            and self.task_head_mode != "theorem_feature_scalar"
        ):
            raise ValueError(
                "summary_spec_root_mode='factored_theorem_readout' requires "
                "task_head_mode='theorem_feature_scalar'"
            )
        if (
            self.theorem_surface_mode == "learned_projection"
            and not self.use_markov_summary_spec
        ):
            raise ValueError(
                "theorem_surface_mode='learned_projection' requires "
                "summary_spec_name='markov_count_sketch'"
            )
        if (
            self.summary_spec_root_mode in ("theorem_primary", "unified_f")
            and not self.use_markov_summary_spec
        ):
            raise ValueError(
                "summary_spec_root_mode='theorem_primary' and 'unified_f' require "
                "summary_spec_name='markov_count_sketch'"
            )
        if (
            self.summary_spec_root_mode == "factored_theorem_readout"
            and self.theorem_surface_mode == "learned_projection"
        ):
            raise ValueError(
                "summary_spec_root_mode='factored_theorem_readout' does not allow "
                "theorem_surface_mode='learned_projection'; theory-aligned readouts "
                "must use phi or theorem surfaces derived from phi"
            )
        self.use_explicit_theorem_subspace = False
        if self.use_summary_spec:
            if int(self.slot_count) <= 0:
                raise ValueError("slot_count must be positive when summary_spec_name is set")
            if int(self.slot_count) < 4:
                raise ValueError("slot_count must be at least 4 when summary_spec_name is set")
            requested_dims = (
                int(theorem_count_dim),
                int(theorem_first_dim),
                int(theorem_last_dim),
            )
            if self.use_direct_markov_sketch_slots and not any(
                dim > 0 for dim in requested_dims
            ):
                requested_dims = (
                    1,
                    int(self.n_regimes),
                    int(self.n_regimes),
                )
            if any(dim > 0 for dim in requested_dims):
                if any(dim <= 0 for dim in requested_dims):
                    raise ValueError(
                        "theorem_count_dim, theorem_first_dim, and theorem_last_dim "
                        "must all be positive when any are set"
                    )
                if int(self.slot_count) != 4:
                    raise ValueError("explicit theorem dims require slot_count == 4")
                if self.use_direct_markov_sketch_slots and requested_dims != (
                    1,
                    int(self.n_regimes),
                    int(self.n_regimes),
                ):
                    raise ValueError(
                        "direct Markov sketch slots require theorem_count_dim=1 "
                        "and theorem_first_dim=theorem_last_dim=n_regimes"
                    )
                if sum(requested_dims) > int(self.state_dim):
                    raise ValueError(
                        "sum of explicit theorem dims must not exceed state_dim"
                    )
                self.use_explicit_theorem_subspace = True
                self.count_theorem_dim = int(requested_dims[0])
                self.first_theorem_dim = int(requested_dims[1])
                self.last_theorem_dim = int(requested_dims[2])
                self.residual_dim = int(self.state_dim) - (
                    int(self.count_theorem_dim)
                    + int(self.first_theorem_dim)
                    + int(self.last_theorem_dim)
                )
                self.slot_dim = int(self.count_theorem_dim)
                self.residual_slot_count = 1 if int(self.residual_dim) > 0 else 0
            else:
                if int(self.state_dim) % int(self.slot_count) != 0:
                    raise ValueError(
                        "state_dim must be divisible by slot_count when summary_spec_name is set"
                    )
                self.slot_dim = int(self.state_dim) // int(self.slot_count)
                self.count_theorem_dim = int(self.slot_dim)
                self.first_theorem_dim = int(self.slot_dim)
                self.last_theorem_dim = int(self.slot_dim)
                self.residual_slot_count = int(self.slot_count) - 3
                self.residual_dim = int(self.residual_slot_count) * int(self.slot_dim)
        else:
            self.slot_dim = 0
            self.count_theorem_dim = 0
            self.first_theorem_dim = 0
            self.last_theorem_dim = 0
            self.residual_slot_count = 0
            self.residual_dim = 0
        if self.use_summary_spec:
            count_start = 0
            first_start = count_start + int(self.count_theorem_dim)
            last_start = first_start + int(self.first_theorem_dim)
            residual_start = last_start + int(self.last_theorem_dim)
            self._count_slice = slice(count_start, first_start)
            self._first_slice = slice(first_start, last_start)
            self._last_slice = slice(last_start, residual_start)
            self._residual_slice = slice(
                residual_start,
                residual_start + int(self.residual_dim),
            )
        else:
            self._count_slice = slice(0, 0)
            self._first_slice = slice(0, 0)
            self._last_slice = slice(0, 0)
            self._residual_slice = slice(0, 0)
        class_values = [int(v) for v in doc_sequence_class_values]
        if not class_values:
            class_values = [0]
        self.register_buffer(
            "doc_sequence_class_values",
            torch.tensor(class_values, dtype=torch.float32),
            persistent=True,
        )
        root_class_values = [int(v) for v in root_count_class_values]
        if not root_class_values:
            root_class_values = [0]
        self.register_buffer(
            "root_count_class_values",
            torch.tensor(root_class_values, dtype=torch.float32),
            persistent=True,
        )
        theorem_count_values = (
            list(range(0, max(int(round(float(self.target_scale))), 0) + 1))
            if self.use_markov_summary_spec
            else [0]
        )
        if not theorem_count_values:
            theorem_count_values = [0]
        self.register_buffer(
            "theorem_count_class_values",
            torch.tensor(theorem_count_values, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "theorem_count_threshold_values",
            torch.tensor(theorem_count_values[1:], dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "theorem_count_threshold_pos_weight",
            torch.ones((max(0, len(theorem_count_values) - 1),), dtype=torch.float32),
            persistent=True,
        )

        _resolved_pooling = str(leaf_fno_pooling or "mean").strip().lower() or "mean"
        if _resolved_pooling not in {"mean", "sum"}:
            raise ValueError(
                f"unsupported leaf_fno_pooling={leaf_fno_pooling!r}; expected 'mean' or 'sum'"
            )
        self.leaf_fno_pooling = _resolved_pooling
        # Leaf encoder: shared FNOTokenEncoder (token embedding -> FNO -> pool)
        # Note: we keep token_embedding and fno_encoder as direct attributes
        # so existing checkpoint state_dict keys remain valid. Pooling logic
        # is applied inside _encode_token_batch (see self.leaf_fno_pooling)
        # so the FNOTokenEncoder pooling_mode kwarg is unused here.
        _leaf_encoder = FNOTokenEncoder(
            vocab_size=int(vocab_size),
            width=int(fno_width),
            n_modes=int(fno_n_modes),
            n_layers=int(fno_n_layers),
            pooling_mode=self.leaf_fno_pooling,
        )
        self.token_embedding = _leaf_encoder.token_embedding
        self.fno_encoder = _leaf_encoder.fno
        # Doc-sequence FNO: separate encoder with deeper layers for full-doc processing.
        # Shares vocab embedding with leaf encoder but has its own input proj and FNO.
        _doc_seq_encoder = FNOTokenEncoder(
            vocab_size=int(vocab_size),
            width=int(fno_width),
            n_modes=int(fno_n_modes),
            n_layers=max(4, int(fno_n_layers)),
            pooling_mode=self.leaf_fno_pooling,
        )
        self.doc_sequence_input_proj = nn.Linear(int(fno_width), int(fno_width))
        self.doc_sequence_fno = _doc_seq_encoder.fno
        leaf_proj_output_dim = (
            int(self.carrier_state_dim)
            if self.use_opaque_carrier_exact_sketch_surface
            else int(self.state_dim)
        )
        self.leaf_proj = nn.Sequential(
            nn.Linear(int(fno_width), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(leaf_proj_output_dim)),
        )
        # Unified-g architecture: ONE function g = encode_summary applied at
        # every tree level.  The summary surface is WIDE (3 * fno_width) —
        # no compression bottleneck.  For leaves the FNO features (pooled +
        # first token + last token) ARE the summary; for merges, a learned
        # projector maps (left_state, right_state) to the same width.
        # Then g maps summary → state at both levels.
        self.unified_g_leaf_summary_proj: nn.Module | None = None
        self.unified_g_merge_summary_proj: nn.Module | None = None
        self.unified_g_summary_dim: int = 0
        if self.tree_model_version == "unified_g":
            # Summary = 3 * fno_width (no compression for leaves).
            self.unified_g_summary_dim = 3 * int(fno_width)
            # Leaf path: NO projection — cat(pooled, first, last) IS the summary.
            # leaf_summary_proj is identity (None signals direct pass-through).
            self.unified_g_leaf_summary_proj = None
            # Merge path: (left_state, right_state) → summary of same width.
            self.unified_g_merge_summary_proj = nn.Sequential(
                nn.Linear(2 * int(state_dim), self.unified_g_summary_dim),
                nn.GELU(),
                nn.LayerNorm(self.unified_g_summary_dim),
                nn.Linear(self.unified_g_summary_dim, self.unified_g_summary_dim),
            )
        self.summary_count_leaf_proj: nn.Module | None = None
        self.summary_first_leaf_proj: nn.Module | None = None
        self.summary_last_leaf_proj: nn.Module | None = None
        self.summary_residual_leaf_proj: nn.Module | None = None
        self.summary_surface_count_proj: nn.Module | None = None
        self.summary_surface_first_proj: nn.Module | None = None
        self.summary_surface_last_proj: nn.Module | None = None
        self.phi_projector: nn.Module | None = None
        self.phi_merge_predictor: nn.Module | None = None
        self.score_merge_predictor: nn.Module | None = None
        self.fiber_merge_predictor: nn.Module | None = None
        self.count_phi_adapter: nn.Module | None = None
        self.first_phi_adapter: nn.Module | None = None
        self.last_phi_adapter: nn.Module | None = None
        self.root_phi_adapter: nn.Module | None = None
        self.join_phi_adapter: nn.Module | None = None
        if self.use_summary_spec:
            if not self.use_shared_theorem_surface:
                leaf_head_input_dim = (
                    int(self.carrier_state_dim)
                    if self.use_opaque_carrier_exact_sketch_surface
                    else int(fno_width)
                )
                self.summary_count_leaf_proj = nn.Sequential(
                    nn.Linear(int(leaf_head_input_dim), int(hidden_dim)),
                    nn.GELU(),
                    nn.Linear(int(hidden_dim), int(self.count_theorem_dim)),
                )
                self.summary_first_leaf_proj = nn.Sequential(
                    nn.Linear(int(leaf_head_input_dim), int(hidden_dim)),
                    nn.GELU(),
                    nn.Linear(int(hidden_dim), int(self.first_theorem_dim)),
                )
                self.summary_last_leaf_proj = nn.Sequential(
                    nn.Linear(int(leaf_head_input_dim), int(hidden_dim)),
                    nn.GELU(),
                    nn.Linear(int(hidden_dim), int(self.last_theorem_dim)),
                )
                if int(self.residual_dim) > 0:
                    self.summary_residual_leaf_proj = nn.Sequential(
                        nn.Linear(int(leaf_head_input_dim), int(hidden_dim)),
                        nn.GELU(),
                        nn.Linear(int(hidden_dim), int(self.residual_dim)),
                    )
            if (
                self.use_markov_summary_spec
                and self.theorem_surface_mode == "learned_projection"
            ):
                self.summary_surface_count_proj = nn.Sequential(
                    nn.Linear(int(state_dim), int(hidden_dim)),
                    nn.GELU(),
                )
                self.summary_surface_first_proj = nn.Sequential(
                    nn.Linear(int(state_dim), int(hidden_dim)),
                    nn.GELU(),
                    nn.Linear(int(hidden_dim), int(self.first_theorem_dim)),
                )
                self.summary_surface_last_proj = nn.Sequential(
                    nn.Linear(int(state_dim), int(hidden_dim)),
                    nn.GELU(),
                    nn.Linear(int(hidden_dim), int(self.last_theorem_dim)),
                )
            if self.use_shared_theorem_surface:
                self.phi_projector = nn.Sequential(
                    nn.LayerNorm(int(state_dim)),
                    nn.Linear(
                        int(state_dim),
                        int(self.shared_theorem_feature_hidden_dim),
                    ),
                    nn.SiLU(),
                    nn.Linear(
                        int(self.shared_theorem_feature_hidden_dim),
                        int(self.shared_theorem_feature_dim),
                    ),
                )
                if self.use_factorized_score_fiber_surface:
                    self.score_merge_predictor = nn.Sequential(
                        nn.Linear(
                            2 * int(self.shared_theorem_feature_dim),
                            int(self.shared_theorem_feature_hidden_dim),
                        ),
                        nn.GELU(),
                        nn.Linear(int(self.shared_theorem_feature_hidden_dim), 3),
                    )
                    fiber_aux_dim = int(self.factorized_fiber_dim) + int(self.factorized_aux_dim)
                    if int(fiber_aux_dim) > 0:
                        self.fiber_merge_predictor = nn.Sequential(
                            nn.Linear(
                                2 * int(self.shared_theorem_feature_dim),
                                int(self.shared_theorem_feature_hidden_dim),
                            ),
                            nn.GELU(),
                            nn.Linear(
                                int(self.shared_theorem_feature_hidden_dim),
                                int(fiber_aux_dim),
                            ),
                        )
                else:
                    self.phi_merge_predictor = nn.Sequential(
                        nn.Linear(
                            2 * int(self.shared_theorem_feature_dim),
                            int(self.shared_theorem_feature_hidden_dim),
                        ),
                        nn.GELU(),
                        nn.Linear(
                            int(self.shared_theorem_feature_hidden_dim),
                            int(self.shared_theorem_feature_dim),
                        ),
                    )
                if self.use_shared_feature_adapters_surface:
                    adapter_hidden = max(32, int(self.shared_theorem_feature_dim))

                    def _make_phi_adapter() -> nn.Sequential:
                        return nn.Sequential(
                            nn.Linear(
                                int(self.shared_theorem_feature_dim),
                                int(adapter_hidden),
                            ),
                            nn.GELU(),
                            nn.Linear(
                                int(adapter_hidden),
                                int(self.shared_feature_adapter_dim),
                            ),
                        )

                    self.count_phi_adapter = _make_phi_adapter()
                    self.first_phi_adapter = _make_phi_adapter()
                    self.last_phi_adapter = _make_phi_adapter()
                    self.root_phi_adapter = _make_phi_adapter()
                    self.join_phi_adapter = _make_phi_adapter()
        if self.use_shared_theorem_surface and self.phi_projector is None:
            phi_input_dim = int(self.state_dim) if self.use_summary_spec else int(self.summary_dim)
            self.phi_projector = nn.Sequential(
                nn.LayerNorm(int(phi_input_dim)),
                nn.Linear(
                    int(phi_input_dim),
                    int(self.shared_theorem_feature_hidden_dim),
                ),
                nn.SiLU(),
                nn.Linear(
                    int(self.shared_theorem_feature_hidden_dim),
                    int(self.shared_theorem_feature_dim),
                ),
            )
            if self.use_factorized_score_fiber_surface:
                self.score_merge_predictor = nn.Sequential(
                    nn.Linear(
                        2 * int(self.shared_theorem_feature_dim),
                        int(self.shared_theorem_feature_hidden_dim),
                    ),
                    nn.GELU(),
                    nn.Linear(int(self.shared_theorem_feature_hidden_dim), 3),
                )
                fiber_aux_dim = int(self.factorized_fiber_dim) + int(self.factorized_aux_dim)
                if int(fiber_aux_dim) > 0:
                    self.fiber_merge_predictor = nn.Sequential(
                        nn.Linear(
                            2 * int(self.shared_theorem_feature_dim),
                            int(self.shared_theorem_feature_hidden_dim),
                        ),
                        nn.GELU(),
                        nn.Linear(
                            int(self.shared_theorem_feature_hidden_dim),
                            int(fiber_aux_dim),
                        ),
                    )
            else:
                self.phi_merge_predictor = nn.Sequential(
                    nn.Linear(
                        2 * int(self.shared_theorem_feature_dim),
                        int(self.shared_theorem_feature_hidden_dim),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        int(self.shared_theorem_feature_hidden_dim),
                        int(self.shared_theorem_feature_dim),
                    ),
                )
            if self.use_shared_feature_adapters_surface:
                adapter_hidden = max(32, int(self.shared_theorem_feature_dim))

                def _make_phi_adapter() -> nn.Sequential:
                    return nn.Sequential(
                        nn.Linear(
                            int(self.shared_theorem_feature_dim),
                            int(adapter_hidden),
                        ),
                        nn.GELU(),
                        nn.Linear(
                            int(adapter_hidden),
                            int(self.shared_feature_adapter_dim),
                        ),
                    )

                self.count_phi_adapter = _make_phi_adapter()
                self.first_phi_adapter = _make_phi_adapter()
                self.last_phi_adapter = _make_phi_adapter()
                self.root_phi_adapter = _make_phi_adapter()
                self.join_phi_adapter = _make_phi_adapter()

        # Merger: slot-structured for summary-spec mode, legacy dense otherwise.
        self.merger: nn.Module | None = None
        self.summary_state_merger: nn.Module | None = None
        self.carrier_state_merger: nn.Module | None = None
        self.count_slot_merger: nn.Module | None = None
        self.residual_slot_merger: nn.Module | None = None
        self.join_bit_head: nn.Module | None = None
        if self.use_summary_spec:
            if self.use_opaque_carrier_exact_sketch_surface:
                self.carrier_state_merger = nn.Sequential(
                    nn.Linear(
                        2 * int(self.carrier_state_dim),
                        int(self.merge_hidden_dim),
                    ),
                    nn.GELU(),
                    nn.LayerNorm(int(self.merge_hidden_dim)),
                    nn.Linear(
                        int(self.merge_hidden_dim),
                        int(self.carrier_state_dim),
                    ),
                )
            elif self.use_shared_theorem_surface:
                self.summary_state_merger = nn.Sequential(
                    nn.Linear(2 * int(self.state_dim), int(hidden_dim)),
                    nn.GELU(),
                    nn.LayerNorm(int(hidden_dim)),
                    nn.Linear(int(hidden_dim), int(self.state_dim)),
                )
            else:
                self.count_slot_merger = nn.Sequential(
                    nn.Linear(
                        2 * int(self.count_theorem_dim)
                        + int(self.first_theorem_dim)
                        + int(self.last_theorem_dim),
                        int(hidden_dim),
                    ),
                    nn.GELU(),
                    nn.LayerNorm(int(hidden_dim)),
                    nn.Linear(int(hidden_dim), int(self.count_theorem_dim)),
                )
                if int(self.residual_dim) > 0:
                    self.residual_slot_merger = nn.Sequential(
                        nn.Linear(2 * int(self.residual_dim), int(hidden_dim)),
                        nn.GELU(),
                        nn.LayerNorm(int(hidden_dim)),
                        nn.Linear(int(hidden_dim), int(self.residual_dim)),
                    )
            join_hidden = max(32, int(hidden_dim) // 2)
            join_input_dim = (
                2
                * (
                    int(self.shared_feature_adapter_dim)
                    if self.use_shared_feature_adapters_surface
                    else int(self.shared_theorem_feature_dim)
                )
                if self.use_shared_theorem_surface
                else int(self.last_theorem_dim) + int(self.first_theorem_dim)
            )
            self.join_bit_head = nn.Sequential(
                nn.Linear(int(join_input_dim), int(join_hidden)),
                nn.GELU(),
                nn.Linear(int(join_hidden), 1),
            )
        else:
            self.merger = nn.Sequential(
                nn.Linear(2 * int(self.state_dim) + 2 * int(n_regimes), int(hidden_dim)),
                nn.GELU(),
                nn.LayerNorm(int(hidden_dim)),
                nn.Linear(int(hidden_dim), int(self.state_dim)),
            )

        # Readout
        self.readout = nn.Sequential(
            nn.Linear(int(self.state_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, 1),
        )
        self.theorem_feature_readout: nn.Module | None = None
        self.theorem_feature_dim = 0
        self.theorem_root_feature_dim = 0
        if self.use_shared_theorem_surface:
            self.theorem_feature_dim = int(self.shared_theorem_feature_dim)
            self.theorem_root_feature_dim = (
                int(self.shared_feature_adapter_dim)
                if self.use_shared_feature_adapters_surface
                else int(self.shared_theorem_feature_dim)
            )
        elif self.use_markov_summary_spec:
            self.theorem_feature_dim = (
                int(hidden_dim)
                + int(self.first_theorem_dim)
                + int(self.last_theorem_dim)
            )
            self.theorem_root_feature_dim = int(self.theorem_feature_dim)
        if (
            self.task_head_mode == "theorem_feature_scalar"
            and not self.use_factorized_score_fiber_surface
        ):
            self.theorem_feature_readout = nn.Sequential(
                nn.Linear(int(self.theorem_root_feature_dim), int(hidden_dim) // 2),
                nn.GELU(),
                nn.Linear(int(hidden_dim) // 2, 1),
            )
        doc_head_hidden = max(32, int(hidden_dim) // 2)
        self.doc_sequence_classifier = nn.Sequential(
            nn.Linear(int(fno_width), int(doc_head_hidden)),
            nn.GELU(),
            nn.Linear(int(doc_head_hidden), int(len(class_values))),
        )
        self.root_count_classifier: nn.Module | None = None
        if self.root_supervision_kind == "count_ce":
            root_head_hidden = max(32, int(hidden_dim) // 2)
            self.root_count_classifier = nn.Sequential(
                nn.Linear(int(self.summary_dim), int(root_head_hidden)),
                nn.GELU(),
                nn.Linear(int(root_head_hidden), int(len(root_class_values))),
            )

        # Endpoint projections: extract boundary features from FNO output
        # at first/last token positions. Supervised indirectly via L2 merge gradient.
        self.first_endpoint_proj: nn.Module
        self.last_endpoint_proj: nn.Module
        self.summary_count_classifier: nn.Module | None = None
        if self.use_decoded_markov_sketch or self.use_summary_spec:
            if self.use_summary_spec:
                if self.use_direct_markov_sketch_slots:
                    self.first_endpoint_proj = nn.Identity()
                    self.last_endpoint_proj = nn.Identity()
                elif self.use_shared_theorem_surface:
                    endpoint_hidden = max(32, int(hidden_dim) // 2)
                    self.first_endpoint_proj = nn.Sequential(
                        nn.Linear(
                            int(self.theorem_root_feature_dim)
                            if self.use_shared_feature_adapters_surface
                            else int(self.shared_theorem_feature_dim),
                            int(endpoint_hidden),
                        ),
                        nn.GELU(),
                        nn.Linear(int(endpoint_hidden), int(n_regimes)),
                    )
                    self.last_endpoint_proj = nn.Sequential(
                        nn.Linear(
                            int(self.theorem_root_feature_dim)
                            if self.use_shared_feature_adapters_surface
                            else int(self.shared_theorem_feature_dim),
                            int(endpoint_hidden),
                        ),
                        nn.GELU(),
                        nn.Linear(int(endpoint_hidden), int(n_regimes)),
                    )
                else:
                    self.first_endpoint_proj = PrototypeClassifier(
                        input_dim=int(self.first_theorem_dim),
                        n_classes=int(n_regimes),
                    )
                    self.last_endpoint_proj = PrototypeClassifier(
                        input_dim=int(self.last_theorem_dim),
                        n_classes=int(n_regimes),
                    )
            else:
                self.first_endpoint_proj = nn.Linear(int(state_dim), int(n_regimes))
                self.last_endpoint_proj = nn.Linear(int(state_dim), int(n_regimes))
        else:
            self.first_endpoint_proj = nn.Linear(int(fno_width), int(n_regimes))
            self.last_endpoint_proj = nn.Linear(int(fno_width), int(n_regimes))

        # Summary encoder = g: the SINGLE function applied at every tree level.
        # Maps summary surface → state.  For unified_g, this is deeper/wider
        # since it's the model's core representational bottleneck.
        summary_encoder_output_dim = (
            int(self.carrier_state_dim)
            if self.use_opaque_carrier_exact_sketch_surface
            else int(self.state_dim)
        )
        if self.use_decoded_markov_sketch or self.use_summary_spec:
            if self.tree_model_version == "unified_g":
                # g = summary_encoder: the SINGLE function applied at every
                # tree level.  Takes wide summary (3*fno_width) → state.
                # This is the core of the model — make it deep enough.
                self.summary_encoder = nn.Sequential(
                    nn.Linear(int(self.unified_g_summary_dim), int(hidden_dim)),
                    nn.GELU(),
                    nn.LayerNorm(int(hidden_dim)),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.GELU(),
                    nn.Linear(int(hidden_dim), int(summary_encoder_output_dim)),
                )
            else:
                self.summary_encoder = nn.Sequential(
                    nn.Linear(1 + 2 * int(n_regimes), int(hidden_dim)),
                    nn.GELU(),
                    nn.Linear(int(hidden_dim), int(summary_encoder_output_dim)),
                )
        else:
            self.summary_encoder = nn.Sequential(
                nn.Linear(1, int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), int(self.state_dim)),
            )
        self.summary_decode_trunk: nn.Module | None = None
        self.summary_count_head: nn.Module | None = None
        self.summary_count_ordinal_head: nn.Module | None = None
        self.summary_count_scalar_aux_head: nn.Module | None = None
        self.theorem_reencoder: nn.Module | None = None
        self.codec_contract: MarkovSketchCodecContract | None = None
        if self.use_summary_spec:
            self.summary_decode_trunk = nn.Sequential(
                nn.Linear(
                    (
                        int(self.shared_feature_adapter_dim)
                        if self.use_shared_feature_adapters_surface
                        else int(self.shared_theorem_feature_dim)
                    )
                    if self.use_shared_theorem_surface
                    else int(self.count_theorem_dim),
                    int(hidden_dim),
                ),
                nn.GELU(),
            )
            if self.theorem_count_head_mode == "support_classifier":
                if self.use_shared_theorem_surface:
                    self.summary_count_classifier = nn.Linear(
                        int(hidden_dim), int(len(theorem_count_values))
                    )
                else:
                    self.summary_count_classifier = PrototypeClassifier(
                        input_dim=int(hidden_dim),
                        n_classes=int(len(theorem_count_values)),
                    )
            elif self.theorem_count_head_mode == "hybrid_ordinal":
                threshold_count = int(max(0, len(theorem_count_values) - 1))
                self.summary_count_ordinal_head = nn.Linear(
                    int(hidden_dim), int(threshold_count)
                )
                self.summary_count_scalar_aux_head = nn.Linear(int(hidden_dim), 1)
            else:
                self.summary_count_head = nn.Linear(int(hidden_dim), 1)
            if (
                not self.use_shared_theorem_surface
                and not self.use_opaque_carrier_exact_sketch_surface
            ):
                self.theorem_reencoder = nn.Sequential(
                    nn.Linear(int(self.summary_dim), int(hidden_dim)),
                    nn.GELU(),
                    nn.Linear(
                        int(hidden_dim),
                        int(self.count_theorem_dim)
                        + int(self.first_theorem_dim)
                        + int(self.last_theorem_dim),
                    ),
                )
            self.codec_contract = MarkovSketchCodecContract(model=self)

    @property
    def summary_dim(self) -> int:
        if self.use_decoded_markov_sketch or self.use_summary_spec:
            return 1 + 2 * int(self.n_regimes)
        return int(self.state_dim) + 2 * int(self.n_regimes)

    @property
    def carrier_residual_dim(self) -> int:
        if not self.use_direct_markov_sketch_slots:
            return 0
        return int(self.residual_dim)

    @property
    def uses_unified_g_learned_merge(self) -> bool:
        """Whether runtime merge is learned as ``compose -> g``.

        Unified-g merges through the learned ``unified_g_merge_summary_proj``
        and the shared ``g`` encoder.  Exact Markov sketch composition is only
        a diagnostic/oracle path unless the runtime merge kind says otherwise.
        """
        return bool(
            self.tree_model_version == "unified_g"
            and self.unified_g_merge_summary_proj is not None
        )

    @property
    def exact_projected_merge_is_runtime_merge(self) -> bool:
        return bool(
            self.use_exact_projected_sketch_merge
            and not self.uses_unified_g_learned_merge
        )

    @property
    def runtime_merge_kind(self) -> str:
        if self.uses_unified_g_learned_merge:
            return "learned_unified_g"
        if self.exact_projected_merge_is_runtime_merge:
            return "exact_projected_sketch"
        if self.use_summary_spec:
            return "summary_spec_learned"
        return "legacy_learned"

    def _carrier_count_slot(self, state: torch.Tensor) -> torch.Tensor:
        if not self.use_direct_markov_sketch_slots:
            raise RuntimeError("carrier count slot requested outside direct-sketch mode")
        return self._count_slot(state)

    def _carrier_first_logits(self, state: torch.Tensor) -> torch.Tensor:
        if not self.use_direct_markov_sketch_slots:
            raise RuntimeError("carrier first logits requested outside direct-sketch mode")
        return self._first_slot(state)

    def _carrier_last_logits(self, state: torch.Tensor) -> torch.Tensor:
        if not self.use_direct_markov_sketch_slots:
            raise RuntimeError("carrier last logits requested outside direct-sketch mode")
        return self._last_slot(state)

    def _canonical_direct_endpoint_slot(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.use_direct_markov_sketch_slots:
            return logits
        return _straight_through_one_hot_from_logits(logits)

    def _runtime_count_slot(self, count_norm: torch.Tensor) -> torch.Tensor:
        mode = str(getattr(self, "runtime_count_discretization", "continuous"))
        if mode != "st_round":
            return count_norm
        count = count_norm * float(self.target_scale)
        rounded_norm = torch.round(count).clamp(
            min=0.0,
            max=float(self.target_scale),
        ) / float(self.target_scale)
        return count_norm + (rounded_norm - count_norm).detach()

    def _opaque_carrier_from_state(self, state: torch.Tensor) -> torch.Tensor:
        if not self.use_opaque_carrier_exact_sketch_surface:
            raise RuntimeError(
                "opaque carrier requested outside opaque_carrier_exact_sketch mode"
            )
        return self._residual_slots_flat(state)

    def _canonical_summary_state_from_components(
        self,
        *,
        count_norm: torch.Tensor,
        first_logits: torch.Tensor,
        last_logits: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_direct_markov_sketch_slots:
            raise RuntimeError(
                "canonical summary state requested outside direct-sketch mode"
            )
        if count_norm.ndim == 0:
            count_slot = count_norm.reshape(1)
        else:
            count_slot = count_norm.unsqueeze(-1)
        if residual is None:
            residual = count_slot.new_zeros((*count_slot.shape[:-1], int(self.residual_dim)))
        return self._pack_summary_spec_state(
            count_slot,
            first_logits,
            last_logits,
            residual,
        )

    def _count_slot(self, state: torch.Tensor) -> torch.Tensor:
        if not self.use_summary_spec:
            raise RuntimeError("count slot requested without summary spec")
        return state[..., self._count_slice]

    def _first_slot(self, state: torch.Tensor) -> torch.Tensor:
        if not self.use_summary_spec:
            raise RuntimeError("first slot requested without summary spec")
        return state[..., self._first_slice]

    def _last_slot(self, state: torch.Tensor) -> torch.Tensor:
        if not self.use_summary_spec:
            raise RuntimeError("last slot requested without summary spec")
        return state[..., self._last_slice]

    def _residual_slots_flat(self, state: torch.Tensor) -> torch.Tensor:
        if not self.use_summary_spec:
            raise RuntimeError("residual slot requested without summary spec")
        if int(self.residual_dim) <= 0:
            return torch.zeros(
                (*state.shape[:-1], 0),
                device=state.device,
                dtype=state.dtype,
            )
        return state[..., self._residual_slice]

    def _pack_summary_spec_state(
        self,
        count_slot: torch.Tensor,
        first_slot: torch.Tensor,
        last_slot: torch.Tensor,
        residual_slots_flat: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_summary_spec:
            raise RuntimeError("summary spec packing requested without summary spec")
        if int(self.residual_dim) <= 0:
            residual = count_slot.new_zeros((*count_slot.shape[:-1], 0))
        else:
            residual = residual_slots_flat
        return torch.cat([count_slot, first_slot, last_slot, residual], dim=-1)

    def _count_hidden_from_state(self, state: torch.Tensor) -> torch.Tensor:
        if not self.use_markov_summary_spec:
            raise RuntimeError("count hidden requested without markov summary spec")
        if self.use_direct_markov_sketch_slots:
            if self.summary_decode_trunk is None:
                raise RuntimeError("count hidden requested without summary decode trunk")
            return self.summary_decode_trunk(self._count_slot(state))
        if self.use_shared_theorem_surface:
            if self.summary_decode_trunk is None:
                raise RuntimeError("count hidden requested without summary decode trunk")
            surface = self.theorem_feature_from_state(state)
            if self.count_phi_adapter is not None:
                surface = self.count_phi_adapter(surface)
            return self.summary_decode_trunk(surface)
        if self.summary_surface_count_proj is not None:
            return self.summary_surface_count_proj(state)
        if self.summary_decode_trunk is None:
            raise RuntimeError("count hidden requested without markov summary spec")
        return self.summary_decode_trunk(self._count_slot(state))

    def _count_hidden_from_theorem_feature(
        self,
        theorem_feature: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_markov_summary_spec or not self.use_shared_theorem_surface:
            raise RuntimeError(
                "count hidden from theorem feature requested without shared theorem summary spec"
            )
        if self.summary_decode_trunk is None:
            raise RuntimeError("count hidden requested without summary decode trunk")
        surface = theorem_feature
        if self.count_phi_adapter is not None:
            surface = self.count_phi_adapter(surface)
        return self.summary_decode_trunk(surface)

    def _first_surface_from_state(self, state: torch.Tensor) -> torch.Tensor:
        if not self.use_markov_summary_spec:
            raise RuntimeError("first surface requested without markov summary spec")
        if self.use_direct_markov_sketch_slots:
            return self._first_slot(state)
        if self.use_shared_theorem_surface:
            surface = self.theorem_feature_from_state(state)
            if self.first_phi_adapter is not None:
                surface = self.first_phi_adapter(surface)
            return surface
        if self.summary_surface_first_proj is not None:
            return self.summary_surface_first_proj(state)
        return self._first_slot(state)

    def _first_surface_from_theorem_feature(
        self,
        theorem_feature: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_markov_summary_spec or not self.use_shared_theorem_surface:
            raise RuntimeError(
                "first surface from theorem feature requested without shared theorem summary spec"
            )
        surface = theorem_feature
        if self.first_phi_adapter is not None:
            surface = self.first_phi_adapter(surface)
        return surface

    def _last_surface_from_state(self, state: torch.Tensor) -> torch.Tensor:
        if not self.use_markov_summary_spec:
            raise RuntimeError("last surface requested without markov summary spec")
        if self.use_direct_markov_sketch_slots:
            return self._last_slot(state)
        if self.use_shared_theorem_surface:
            surface = self.theorem_feature_from_state(state)
            if self.last_phi_adapter is not None:
                surface = self.last_phi_adapter(surface)
            return surface
        if self.summary_surface_last_proj is not None:
            return self.summary_surface_last_proj(state)
        return self._last_slot(state)

    def _last_surface_from_theorem_feature(
        self,
        theorem_feature: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_markov_summary_spec or not self.use_shared_theorem_surface:
            raise RuntimeError(
                "last surface from theorem feature requested without shared theorem summary spec"
            )
        surface = theorem_feature
        if self.last_phi_adapter is not None:
            surface = self.last_phi_adapter(surface)
        return surface

    def _root_surface_from_state(self, state: torch.Tensor) -> torch.Tensor:
        theorem_feature = self.theorem_feature_from_state(state)
        return self._root_surface_from_theorem_feature(theorem_feature)

    def _root_surface_from_theorem_feature(
        self,
        theorem_feature: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_factorized_score_fiber_surface:
            return self._score_feature_from_theorem_feature(theorem_feature)
        if self.root_phi_adapter is not None:
            return self.root_phi_adapter(theorem_feature)
        return theorem_feature

    def _join_surface_from_state(self, state: torch.Tensor) -> torch.Tensor:
        theorem_feature = self.theorem_feature_from_state(state)
        if self.join_phi_adapter is not None:
            return self.join_phi_adapter(theorem_feature)
        return theorem_feature

    def _theorem_equiv_class(self, target: Tuple[float, ...]) -> Hashable:
        if len(target) >= 3:
            label = self.theorem_feature_adapter.oracle_label(
                count=float(target[0]),
                first=int(target[1]),
                last=int(target[2]),
                metadata=None,
            )
            return self.theorem_feature_adapter.diagnostic_key(label)
        return _markov_equiv_class(target)

    def task_target_from_label(self, label: Any) -> float:
        return float(self.theorem_feature_adapter.task_readout_target(label))

    def _score_feature_from_theorem_feature(
        self,
        theorem_feature: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_factorized_score_fiber_surface:
            return theorem_feature
        return theorem_feature[..., self._score_feature_slice]

    def _fiber_feature_from_theorem_feature(
        self,
        theorem_feature: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_factorized_score_fiber_surface:
            return theorem_feature
        fiber = theorem_feature[..., self._fiber_feature_slice]
        aux = theorem_feature[..., self._aux_feature_slice]
        if int(self.factorized_aux_dim) <= 0:
            return fiber
        return torch.cat([fiber, aux], dim=-1)

    def theorem_feature_from_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.use_shared_theorem_surface:
            if self.phi_projector is None:
                raise RuntimeError("shared theorem projector is not initialized")
            return self.phi_projector(state)
        if not self.use_markov_summary_spec:
            raise RuntimeError(
                "theorem feature requested without theorem surface support"
            )
        return torch.cat(
            [
                self._count_hidden_from_state(state),
                self._first_surface_from_state(state),
                self._last_surface_from_state(state),
            ],
            dim=-1,
        )

    def _carrier_leaf_state_from_leaf_features(
        self,
        *,
        pooled: torch.Tensor,
        first_features: torch.Tensor,
        last_features: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_carrier_projection_surface:
            raise RuntimeError(
                "carrier leaf state requested outside carrier projection mode"
            )
        if (
            self.summary_count_leaf_proj is None
            or self.summary_first_leaf_proj is None
            or self.summary_last_leaf_proj is None
        ):
            raise RuntimeError("carrier projection leaf heads are not initialized")
        count_norm = torch.sigmoid(self.summary_count_leaf_proj(pooled)).squeeze(-1)
        first_logits = self.summary_first_leaf_proj(first_features)
        last_logits = self.summary_last_leaf_proj(last_features)
        if self.summary_residual_leaf_proj is None or int(self.residual_dim) <= 0:
            residual = pooled.new_zeros((*pooled.shape[:-1], int(self.residual_dim)))
        else:
            residual = self.summary_residual_leaf_proj(pooled)
        return self._canonical_summary_state_from_components(
            count_norm=count_norm,
            first_logits=first_logits,
            last_logits=last_logits,
            residual=residual,
        )

    def _opaque_carrier_leaf_state_from_carrier(
        self,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_opaque_carrier_exact_sketch_surface:
            raise RuntimeError(
                "opaque carrier leaf state requested outside opaque_carrier_exact_sketch mode"
            )
        if (
            self.summary_count_leaf_proj is None
            or self.summary_first_leaf_proj is None
            or self.summary_last_leaf_proj is None
        ):
            raise RuntimeError("opaque carrier leaf heads are not initialized")
        count_norm = torch.sigmoid(self.summary_count_leaf_proj(carrier)).squeeze(-1)
        first_logits = self.summary_first_leaf_proj(carrier)
        last_logits = self.summary_last_leaf_proj(carrier)
        return self._canonical_summary_state_from_components(
            count_norm=count_norm,
            first_logits=first_logits,
            last_logits=last_logits,
            residual=carrier,
        )

    @property
    def leaf_state_dim(self) -> int:
        return int(self.state_dim)

    @property
    def has_phi(self) -> bool:
        return bool(self.use_shared_theorem_surface or self.use_markov_summary_spec)

    def encode_leaves(
        self,
        *,
        embeddings: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[Sequence[int]]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if embeddings is not None:
            raise ValueError("FNOCountSketch uses token-id leaves, not embedding leaves")
        if token_ids is None:
            raise ValueError("FNOCountSketch requires token_ids")
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        return self.encode_leaf_tokens_batch(token_ids, device=device)

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self._merge_state_pairs(left, right)

    def merge_batch(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self._merge_state_pairs(left, right)

    def phi(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.has_phi:
            return None
        return self.theorem_feature_from_state(state)

    def phi_batch(self, states: torch.Tensor) -> Optional[torch.Tensor]:
        return self.phi(states)

    def phi_score(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        phi = self.phi(state)
        if phi is None:
            return None
        return self._score_feature_from_theorem_feature(phi)

    def phi_fiber(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        phi = self.phi(state)
        if phi is None:
            return None
        return self._fiber_feature_from_theorem_feature(phi)

    def predict_phi_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.predict_phi_from_theorem_feature(
            self.theorem_feature_from_state(state)
        )

    def predict_phi_from_theorem_feature(
        self,
        theorem_feature: torch.Tensor,
    ) -> torch.Tensor:
        return self._fiber_feature_from_theorem_feature(theorem_feature)

    def predict_score_parent_from_theorem_features(
        self,
        left_feature: torch.Tensor,
        right_feature: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_factorized_score_fiber_surface or self.score_merge_predictor is None:
            raise RuntimeError(
                "score parent prediction requested without factorized score-fiber surface"
            )
        merge_input = torch.cat([left_feature, right_feature], dim=-1)
        score_params = self.score_merge_predictor(merge_input)
        left_score = self._score_feature_from_theorem_feature(left_feature)
        right_score = self._score_feature_from_theorem_feature(right_feature)
        left_gate = F.softplus(score_params[..., :1])
        right_gate = F.softplus(score_params[..., 1:2])
        bias = score_params[..., 2:3]
        return left_gate * left_score + right_gate * right_score + bias

    def predict_phi_parent_from_theorem_features(
        self,
        left_feature: torch.Tensor,
        right_feature: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_shared_theorem_surface:
            raise RuntimeError(
                "phi parent prediction requested without shared theorem surface"
            )
        if self.use_factorized_score_fiber_surface:
            parent_score = self.predict_score_parent_from_theorem_features(
                left_feature,
                right_feature,
            )
            fiber_aux_dim = int(self.factorized_fiber_dim) + int(self.factorized_aux_dim)
            if int(fiber_aux_dim) <= 0:
                return parent_score[..., 0] if parent_score.ndim == 1 else parent_score
            if self.fiber_merge_predictor is None:
                raise RuntimeError("factorized fiber merge predictor is not initialized")
            merge_input = torch.cat([left_feature, right_feature], dim=-1)
            parent_fiber = self.fiber_merge_predictor(merge_input)
            parent_full = torch.cat([parent_score, parent_fiber], dim=-1)
            return self.predict_phi_from_theorem_feature(parent_full)
        if self.phi_merge_predictor is None:
            raise RuntimeError(
                "phi parent prediction requested without shared theorem surface"
            )
        return self.phi_merge_predictor(
            torch.cat([left_feature, right_feature], dim=-1)
        )

    def predict_phi_parent_from_children(
        self,
        left_state: torch.Tensor,
        right_state: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_shared_theorem_surface:
            raise RuntimeError(
                "phi parent prediction requested without shared theorem surface"
            )
        left_feature = self.theorem_feature_from_state(left_state)
        right_feature = self.theorem_feature_from_state(right_state)
        return self.predict_phi_parent_from_theorem_features(
            left_feature,
            right_feature,
        )

    def predict_score_parent_from_children(
        self,
        left_state: torch.Tensor,
        right_state: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_factorized_score_fiber_surface or self.score_merge_predictor is None:
            raise RuntimeError(
                "score parent prediction requested without factorized score-fiber surface"
            )
        left_feature = self.theorem_feature_from_state(left_state)
        right_feature = self.theorem_feature_from_state(right_state)
        return self.predict_score_parent_from_theorem_features(
            left_feature,
            right_feature,
        )

    def uses_hybrid_ordinal_count_head(self) -> bool:
        return bool(
            self.use_markov_summary_spec
            and self.theorem_count_head_mode == "hybrid_ordinal"
        )

    def theorem_count_support_size(self) -> int:
        return int(self.theorem_count_class_values.numel())

    def theorem_count_threshold_count(self) -> int:
        return int(self.theorem_count_threshold_values.numel())

    def set_theorem_count_threshold_pos_weight(
        self,
        values: Sequence[float] | torch.Tensor,
    ) -> None:
        if not self.uses_hybrid_ordinal_count_head():
            return
        weight = torch.as_tensor(
            values,
            dtype=self.theorem_count_threshold_pos_weight.dtype,
            device=self.theorem_count_threshold_pos_weight.device,
        )
        if weight.shape != self.theorem_count_threshold_pos_weight.shape:
            raise ValueError(
                "threshold pos weight shape mismatch: "
                f"expected {tuple(self.theorem_count_threshold_pos_weight.shape)}, "
                f"got {tuple(weight.shape)}"
            )
        self.theorem_count_threshold_pos_weight.copy_(weight)

    def predict_count_logits_from_state(self, state: torch.Tensor) -> torch.Tensor:
        if not self.use_markov_summary_spec or self.summary_count_classifier is None:
            raise RuntimeError(
                "count logits requested without support-classifier theorem head"
            )
        hidden = self._count_hidden_from_state(state)
        return self.summary_count_classifier(hidden)

    def predict_count_ordinal_logits_from_state(self, state: torch.Tensor) -> torch.Tensor:
        if not self.use_markov_summary_spec or self.summary_count_ordinal_head is None:
            raise RuntimeError(
                "ordinal count logits requested without hybrid-ordinal theorem head"
            )
        hidden = self._count_hidden_from_state(state)
        return self.summary_count_ordinal_head(hidden)

    def predict_count_scalar_aux_from_state(self, state: torch.Tensor) -> torch.Tensor:
        if not self.use_markov_summary_spec:
            raise RuntimeError("scalar count requested without markov summary spec")
        hidden = self._count_hidden_from_state(state)
        if self.summary_count_scalar_aux_head is not None:
            count_norm = torch.sigmoid(self.summary_count_scalar_aux_head(hidden)).squeeze(-1)
            return count_norm * float(self.target_scale)
        if self.summary_count_head is None:
            raise RuntimeError("scalar auxiliary count head is not initialized")
        count_norm = torch.sigmoid(self.summary_count_head(hidden)).squeeze(-1)
        return count_norm * float(self.target_scale)

    def _ordinal_threshold_probs_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if int(probs.shape[-1]) <= 1:
            return probs
        return torch.cummin(probs, dim=-1).values

    def _class_probs_from_ordinal_logits(self, logits: torch.Tensor) -> torch.Tensor:
        threshold_probs = self._ordinal_threshold_probs_from_logits(logits)
        if int(threshold_probs.shape[-1]) == 0:
            return torch.ones(
                (*threshold_probs.shape[:-1], 1),
                device=threshold_probs.device,
                dtype=threshold_probs.dtype,
            )
        first = 1.0 - threshold_probs[..., :1]
        if int(threshold_probs.shape[-1]) > 1:
            middle = threshold_probs[..., :-1] - threshold_probs[..., 1:]
            probs = torch.cat([first, middle, threshold_probs[..., -1:]], dim=-1)
        else:
            probs = torch.cat([first, threshold_probs], dim=-1)
        return probs.clamp(min=0.0)

    def predict_count_distribution_from_state(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.use_markov_summary_spec:
            return None
        if self.summary_count_classifier is not None:
            logits = self.predict_count_logits_from_state(state)
            return torch.softmax(logits, dim=-1)
        if self.summary_count_ordinal_head is not None:
            logits = self.predict_count_ordinal_logits_from_state(state)
            return self._class_probs_from_ordinal_logits(logits)
        return None

    def predict_count_support_entropy_from_state(self, state: torch.Tensor) -> torch.Tensor:
        probs = self.predict_count_distribution_from_state(state)
        if probs is None:
            base = self.predict_count_from_state(state)
            return torch.full_like(base, float("nan"))
        return PrototypeClassifier.distribution_entropy(probs)

    def predict_count_support_margin_from_state(self, state: torch.Tensor) -> torch.Tensor:
        probs = self.predict_count_distribution_from_state(state)
        if probs is None:
            base = self.predict_count_from_state(state)
            return torch.full_like(base, float("nan"))
        return PrototypeClassifier.distribution_margin(probs)

    def _count_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        support = self.theorem_count_class_values.to(
            device=logits.device,
            dtype=logits.dtype,
        )
        probs = torch.softmax(logits, dim=-1)
        return torch.sum(probs * support, dim=-1)

    def _count_from_ordinal_logits(self, logits: torch.Tensor) -> torch.Tensor:
        threshold_probs = self._ordinal_threshold_probs_from_logits(logits)
        return torch.sum(threshold_probs, dim=-1)

    def count_target_index(self, truth_count: float, *, device: torch.device) -> torch.Tensor:
        support = self.theorem_count_class_values.to(device=device)
        target = torch.tensor(float(truth_count), device=device, dtype=support.dtype)
        return torch.argmin(torch.abs(support - target)).to(torch.long)

    def count_threshold_targets(
        self,
        truth_count: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        threshold_values = self.theorem_count_threshold_values.to(
            device=device,
            dtype=dtype,
        )
        target = torch.tensor(float(truth_count), device=device, dtype=dtype)
        return (target >= threshold_values).to(dtype=dtype)

    def _decode_markov_summary_components(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_markov_summary_spec:
            raise RuntimeError("markov summary components requested without summary spec")
        if self.use_direct_markov_sketch_slots:
            count_norm = self._carrier_count_slot(state).squeeze(-1)
            first_logits = self._carrier_first_logits(state)
            last_logits = self._carrier_last_logits(state)
            return count_norm, first_logits, last_logits
        if self.summary_decode_trunk is None:
            raise RuntimeError("summary decode heads are not initialized")
        hidden = self._count_hidden_from_state(state)
        if self.summary_count_classifier is not None:
            count = self._count_from_logits(self.summary_count_classifier(hidden))
            count_norm = count / float(self.target_scale)
        elif self.summary_count_ordinal_head is not None:
            count = self._count_from_ordinal_logits(self.summary_count_ordinal_head(hidden))
            count_norm = count / float(self.target_scale)
        else:
            if self.summary_count_head is None:
                raise RuntimeError("summary count head is not initialized")
            count_norm = torch.sigmoid(self.summary_count_head(hidden)).squeeze(-1)
        first_logits = self.first_endpoint_proj(self._first_surface_from_state(state))
        last_logits = self.last_endpoint_proj(self._last_surface_from_state(state))
        return count_norm, first_logits, last_logits

    def predict_join_logit_from_states(
        self,
        left_state: torch.Tensor,
        right_state: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_markov_summary_spec or self.join_bit_head is None:
            raise RuntimeError("join-bit head requested without markov summary spec")
        if self.exact_projected_merge_is_runtime_merge:
            left_last_logits = self._split_state(left_state)[2]
            right_first_logits = self._split_state(right_state)[1]
            join_prob = 1.0 - torch.sum(
                torch.softmax(left_last_logits, dim=-1)
                * torch.softmax(right_first_logits, dim=-1),
                dim=-1,
            )
            clipped = join_prob.clamp(min=1e-6, max=1.0 - 1e-6)
            return torch.log(clipped) - torch.log1p(-clipped)
        if self.use_shared_theorem_surface:
            join_features = torch.cat(
                [
                    self._join_surface_from_state(left_state),
                    self._join_surface_from_state(right_state),
                ],
                dim=-1,
            )
        else:
            left_last = self._last_surface_from_state(left_state)
            right_first = self._first_surface_from_state(right_state)
            if self.use_direct_markov_sketch_slots:
                left_last = self._canonical_direct_endpoint_slot(left_last)
                right_first = self._canonical_direct_endpoint_slot(right_first)
            join_features = torch.cat(
                [
                    left_last,
                    right_first,
                ],
                dim=-1,
            )
        return self.join_bit_head(join_features).squeeze(-1)

    def predict_join_prob_from_states(
        self,
        left_state: torch.Tensor,
        right_state: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.predict_join_logit_from_states(left_state, right_state))

    def _split_state(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = int(self.state_dim)
        n = int(self.n_regimes)
        if self.use_summary_spec:
            h = state
            _count_norm, first_logits, last_logits = self._decode_markov_summary_components(
                h
            )
            return h, first_logits, last_logits
        if self.use_decoded_markov_sketch:
            h = state
            first = self.first_endpoint_proj(h)
            last = self.last_endpoint_proj(h)
            return h, first, last
        h = state[..., :d]
        first = state[..., d : d + n]
        last = state[..., d + n : d + 2 * n]
        return h, first, last

    def _pack_state(
        self,
        h: torch.Tensor,
        first_logits: torch.Tensor,
        last_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_decoded_markov_sketch or self.use_summary_spec:
            return h
        return torch.cat([h, first_logits, last_logits], dim=-1)

    def encode_leaf_tokens(
        self,
        token_ids: Sequence[int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode a single leaf span's raw tokens -> state vector."""
        state_batch = self.encode_leaf_tokens_batch((tuple(token_ids),), device=device)
        return state_batch[0]

    def encode_leaf_tokens_batch(
        self,
        token_id_batch: Sequence[Sequence[int]] | torch.Tensor | GpuBatchView,
        *,
        device: torch.device,
        token_mask: torch.Tensor | None = None,
        runtime_telemetry: GpuRuntimeTelemetry | None = None,
    ) -> torch.Tensor:
        """Encode a batch of leaf spans' raw tokens -> state matrix."""
        if isinstance(token_id_batch, GpuBatchView):
            view_tokens = token_id_batch.tensors.get("leaf_tokens")
            view_mask = token_id_batch.tensors.get("leaf_mask")
            if not isinstance(view_tokens, torch.Tensor):
                raise ValueError("GpuBatchView is missing leaf_tokens tensor")
            if view_tokens.ndim == 3:
                dense_states = _encode_dense_leaf_token_tensor_batch(
                    self,
                    view_tokens,
                    device=device,
                    token_mask_tensor=view_mask if isinstance(view_mask, torch.Tensor) else None,
                    runtime_telemetry=runtime_telemetry,
                )
                return dense_states.reshape(-1, int(dense_states.shape[-1]))
            token_id_batch = view_tokens
            token_mask = view_mask if isinstance(view_mask, torch.Tensor) else token_mask
        if isinstance(token_id_batch, torch.Tensor):
            if token_id_batch.ndim == 3:
                dense_states = _encode_dense_leaf_token_tensor_batch(
                    self,
                    token_id_batch,
                    device=device,
                    token_mask_tensor=token_mask,
                    runtime_telemetry=runtime_telemetry,
                )
                return dense_states.reshape(-1, int(dense_states.shape[-1]))
            if token_id_batch.ndim != 2:
                raise ValueError("token tensor must have shape (batch, tokens) or (batch, leaves, tokens)")
            tokens = token_id_batch
            if token_mask is None:
                token_mask = tokens.ne(int(self.pad_id)).to(dtype=torch.float32)
            if tokens.device != device:
                copy_start_s = time.perf_counter()
                tokens = tokens.to(device=device, non_blocking=True)
                token_mask = token_mask.to(device=device, non_blocking=True)
                if runtime_telemetry is not None:
                    runtime_telemetry.add_h2d(
                        bytes_transferred=_tensor_nbytes(tokens) + _tensor_nbytes(token_mask),
                        wall_time_s=time.perf_counter() - copy_start_s,
                    )
            x, pooled = self._encode_token_batch(tokens, token_mask=token_mask)
            # Unified-g: extract (count, first, last) then call encode_summary.
            # This is the SAME g used by merges (decode-compose-reencode).
            if self.tree_model_version == "unified_g" and self.unified_g_summary_dim > 0:
                x_seq = x.permute(0, 2, 1)
                # Gather last valid token features (not padding)
                if token_mask is not None:
                    last_idx = token_mask.sum(dim=-1).long().clamp(min=1) - 1
                else:
                    last_idx = torch.full(
                        (int(x_seq.shape[0]),), int(x_seq.shape[1]) - 1,
                        device=x_seq.device, dtype=torch.long,
                    )
                first_features = x_seq[:, 0, :]
                last_features = x_seq[
                    torch.arange(int(x_seq.shape[0]), device=x_seq.device),
                    last_idx, :,
                ]
                # FNO features ARE the summary — no compression.
                leaf_summary = torch.cat([pooled, first_features, last_features], dim=-1)
                if self.unified_g_leaf_summary_proj is not None:
                    leaf_summary = self.unified_g_leaf_summary_proj(leaf_summary)
                # Apply unified g (same function used by merges)
                return self.encode_summary(leaf_summary)
            h = self.leaf_proj(pooled)
            if self.use_summary_spec:
                if self.use_opaque_carrier_exact_sketch_surface:
                    return self._opaque_carrier_leaf_state_from_carrier(h)
                if self.use_carrier_projection_surface:
                    x_seq = x.permute(0, 2, 1)
                    if token_mask is not None:
                        last_idx = token_mask.sum(dim=-1).long().clamp(min=1) - 1
                    else:
                        last_idx = torch.full(
                            (int(x_seq.shape[0]),),
                            int(x_seq.shape[1]) - 1,
                            device=x_seq.device,
                            dtype=torch.long,
                        )
                    return self._carrier_leaf_state_from_leaf_features(
                        pooled=pooled,
                        first_features=x_seq[:, 0, :],
                        last_features=x_seq[
                            torch.arange(int(x_seq.shape[0]), device=x_seq.device),
                            last_idx,
                        ],
                    )
                if self.use_shared_theorem_surface:
                    return h
                if (
                    self.summary_count_leaf_proj is None
                    or self.summary_first_leaf_proj is None
                    or self.summary_last_leaf_proj is None
                ):
                    raise RuntimeError("summary-spec leaf projectors are not initialized")
                x_seq = x.permute(0, 2, 1)
                count_slot = self.summary_count_leaf_proj(pooled)
                first_slot = self.summary_first_leaf_proj(x_seq[:, 0, :])
                last_slot = self.summary_last_leaf_proj(x_seq[:, -1, :])
                if self.summary_residual_leaf_proj is None or int(self.residual_dim) <= 0:
                    residual = count_slot.new_zeros((int(tokens.shape[0]), int(self.residual_dim)))
                else:
                    residual = self.summary_residual_leaf_proj(pooled)
                return self._pack_summary_spec_state(
                    count_slot,
                    first_slot,
                    last_slot,
                    residual,
                )
            if self.use_decoded_markov_sketch:
                return h
            x_seq = x.permute(0, 2, 1)
            first_reg = self.first_endpoint_proj(x_seq[:, 0, :])
            last_reg = self.last_endpoint_proj(x_seq[:, -1, :])
            return self._pack_state(h, first_reg, last_reg)
        if len(token_id_batch) == 0:
            feature_dim = (
                int(self.state_dim)
                if (self.use_summary_spec or self.use_decoded_markov_sketch)
                else int(self.summary_dim)
            )
            return torch.zeros((0, feature_dim), dtype=torch.float32, device=device)
        token_lists = [list(token_ids) for token_ids in token_id_batch]
        lengths = [int(len(tokens)) for tokens in token_lists]
        effective_len = max(
            int(self.leaf_tokens),
            max(lengths) if lengths else int(self.leaf_tokens),
        )
        use_pinned_staging = bool(device.type == "cuda")
        staging_device = torch.device("cpu") if use_pinned_staging else device
        tokens = torch.full(
            (len(token_lists), int(effective_len)),
            int(self.pad_id),
            dtype=torch.long,
            device=staging_device,
            pin_memory=use_pinned_staging,
        )
        token_mask = torch.zeros(
            (len(token_lists), int(effective_len)),
            dtype=torch.float32,
            device=staging_device,
            pin_memory=use_pinned_staging,
        )
        for row_idx, toks in enumerate(token_lists):
            n_valid = int(lengths[row_idx])
            if n_valid <= 0:
                continue
            tokens[row_idx, :n_valid] = torch.as_tensor(
                toks[:n_valid],
                dtype=torch.long,
                device=staging_device,
            )
            token_mask[row_idx, :n_valid] = 1.0
        if use_pinned_staging:
            copy_start_s = time.perf_counter()
            tokens = tokens.to(device=device, non_blocking=True)
            token_mask = token_mask.to(device=device, non_blocking=True)
            if runtime_telemetry is not None:
                runtime_telemetry.add_h2d(
                    bytes_transferred=_tensor_nbytes(tokens) + _tensor_nbytes(token_mask),
                    wall_time_s=time.perf_counter() - copy_start_s,
                )

        x, pooled = self._encode_token_batch(tokens, token_mask=token_mask)
        h = self.leaf_proj(pooled)
        if self.use_summary_spec:
            if self.use_opaque_carrier_exact_sketch_surface:
                return self._opaque_carrier_leaf_state_from_carrier(h)
            if self.use_carrier_projection_surface:
                x_seq = x.permute(0, 2, 1)
                batch_indices = torch.arange(len(token_lists), device=device)
                last_indices = torch.as_tensor(
                    [
                        max(0, min(int(length), int(effective_len)) - 1)
                        for length in lengths
                    ],
                    dtype=torch.long,
                    device=device,
                )
                return self._carrier_leaf_state_from_leaf_features(
                    pooled=pooled,
                    first_features=x_seq[:, 0, :],
                    last_features=x_seq[batch_indices, last_indices, :],
                )
            if self.use_shared_theorem_surface:
                return h
            if (
                self.summary_count_leaf_proj is None
                or self.summary_first_leaf_proj is None
                or self.summary_last_leaf_proj is None
            ):
                raise RuntimeError("summary-spec leaf projectors are not initialized")
            x_seq = x.permute(0, 2, 1)  # (B, L, fno_width)
            batch_indices = torch.arange(len(token_lists), device=device)
            last_indices = torch.as_tensor(
                [
                    max(0, min(int(length), int(effective_len)) - 1)
                    for length in lengths
                ],
                dtype=torch.long,
                device=device,
            )
            count_slot = self.summary_count_leaf_proj(pooled)
            first_slot = self.summary_first_leaf_proj(x_seq[:, 0, :])
            last_slot = self.summary_last_leaf_proj(x_seq[batch_indices, last_indices, :])
            if self.summary_residual_leaf_proj is None or int(self.residual_dim) <= 0:
                residual = count_slot.new_zeros((len(token_lists), int(self.residual_dim)))
            else:
                residual = self.summary_residual_leaf_proj(pooled)
            return self._pack_summary_spec_state(
                count_slot,
                first_slot,
                last_slot,
                residual,
            )
        if self.use_decoded_markov_sketch:
            return h

        x_seq = x.permute(0, 2, 1)  # (B, L, fno_width)
        batch_indices = torch.arange(len(token_lists), device=device)
        last_indices = torch.as_tensor(
            [
                max(0, min(int(length), int(effective_len)) - 1)
                for length in lengths
            ],
            dtype=torch.long,
            device=device,
        )
        first_reg = self.first_endpoint_proj(x_seq[:, 0, :])
        last_reg = self.last_endpoint_proj(x_seq[batch_indices, last_indices, :])
        return self._pack_state(h, first_reg, last_reg)

    def _encode_token_batch(
        self,
        tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2 or token_mask.ndim != 2:
            raise ValueError("tokens and token_mask must both be rank-2")
        return apply_fno_token_encoder(
            tokens,
            token_mask=token_mask,
            token_embedding=self.token_embedding,
            fno=self.fno_encoder,
            pooling_mode=self.leaf_fno_pooling,
        )

    def predict_doc_sequence_logits(
        self,
        tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 2 or token_mask.ndim != 2:
            raise ValueError("tokens and token_mask must both be rank-2")
        _x, pooled = apply_fno_token_encoder(
            tokens,
            token_mask=token_mask,
            token_embedding=self.token_embedding,
            fno=self.doc_sequence_fno,
            pooling_mode=self.leaf_fno_pooling,
            input_proj=self.doc_sequence_input_proj,
        )
        return self.doc_sequence_classifier(pooled)

    def predict_doc_sequence_expected_normalized_from_logits(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        class_values = self.doc_sequence_class_values.to(
            device=logits.device,
            dtype=logits.dtype,
        )
        return (probs * (class_values / float(self.target_scale)).unsqueeze(0)).sum(dim=-1)

    def predict_doc_sequence_counts_from_logits(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        class_values = self.doc_sequence_class_values.to(
            device=logits.device,
            dtype=logits.dtype,
        )
        pred_idx = torch.argmax(logits, dim=-1)
        return class_values.index_select(0, pred_idx)

    def predict_task_norm_from_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.use_opaque_carrier_exact_sketch_surface:
            count_norm, _first_logits, _last_logits = self._decode_markov_summary_components(
                state
            )
            return count_norm
        if self.task_head_mode == "theorem_feature_scalar":
            theorem_feature = self.theorem_feature_from_state(state)
            return self.predict_task_norm_from_theorem_feature(theorem_feature)
        h, _, _ = self._split_state(state)
        return torch.sigmoid(self.readout(h)).squeeze(-1)

    def predict_task_norm_from_theorem_feature(
        self,
        theorem_feature: torch.Tensor,
    ) -> torch.Tensor:
        if self.task_head_mode != "theorem_feature_scalar":
            raise RuntimeError(
                "task norm from theorem feature requested without theorem_feature_scalar task head"
            )
        root_surface = self._root_surface_from_theorem_feature(theorem_feature)
        if self.use_factorized_score_fiber_surface:
            return torch.sigmoid(root_surface).squeeze(-1)
        if self.theorem_feature_readout is None:
            raise RuntimeError("theorem feature readout is not initialized")
        return torch.sigmoid(self.theorem_feature_readout(root_surface)).squeeze(-1)

    def predict_task_count_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.predict_task_norm_from_state(state) * float(self.target_scale)

    def uses_theorem_primary_root_mode(self) -> bool:
        return bool(
            self.use_markov_summary_spec
            and self.summary_spec_root_mode == "theorem_primary"
        )

    def uses_factored_theorem_readout_root_mode(self) -> bool:
        return bool(
            self.summary_spec_root_mode == "factored_theorem_readout"
            and self.task_head_mode == "theorem_feature_scalar"
        )

    def uses_theory_aligned_root_surface(self) -> bool:
        return bool(
            self.uses_factored_theorem_readout_root_mode()
            or (
                self.use_markov_summary_spec
                and self.summary_spec_root_mode in ("theorem_primary", "unified_f")
            )
        )

    def uses_unified_readout(self) -> bool:
        return bool(
            self.use_markov_summary_spec
            and self.summary_spec_root_mode in ("unified_f", "theorem_primary")
        )

    def predict_norm_from_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.use_summary_spec:
            count_norm, _first_logits, _last_logits = self._decode_markov_summary_components(
                state
            )
            return count_norm
        return self.predict_task_norm_from_state(state)

    def predict_count_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.predict_norm_from_state(state) * float(self.target_scale)

    def predict(self, state: torch.Tensor, head: str = "count") -> torch.Tensor:
        predicted = self.predict_count_from_state(state)
        if predicted.ndim == 0:
            return predicted.reshape(1)
        return predicted.reshape(-1, 1) if predicted.ndim == 1 else predicted

    def predict_batch(self, states: torch.Tensor, head: str = "count") -> torch.Tensor:
        predicted = self.predict_count_from_state(states)
        if predicted.ndim == 0:
            return predicted.reshape(1, 1)
        return predicted.reshape(-1, 1)

    def predict_normalized(self, state: torch.Tensor, head: str = "count") -> torch.Tensor:
        predicted = self.predict_norm_from_state(state)
        if predicted.ndim == 0:
            return predicted.reshape(1)
        return predicted.reshape(-1, 1) if predicted.ndim == 1 else predicted

    def predict_normalized_batch(self, states: torch.Tensor, head: str = "count") -> torch.Tensor:
        predicted = self.predict_norm_from_state(states)
        if predicted.ndim == 0:
            return predicted.reshape(1, 1)
        return predicted.reshape(-1, 1)

    def predict_confidence(self, state: torch.Tensor, head: str = "count") -> torch.Tensor:
        pred_norm = self.predict_normalized(state, head=head)
        return 1.0 - 2.0 * torch.abs(pred_norm - 0.5)

    def predict_confidence_batch(self, states: torch.Tensor, head: str = "count") -> torch.Tensor:
        pred_norm = self.predict_normalized_batch(states, head=head)
        return 1.0 - 2.0 * torch.abs(pred_norm - 0.5)

    def as_tree_model_v2(self) -> TreeModelV2View:
        return TreeModelV2View(
            self,
            leaf_input_kind="token_ids",
            tree_model_version=self.tree_model_version,
        )

    def predict_root_count_logits_from_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.root_count_classifier is None:
            raise RuntimeError(
                "root count logits requested but root_supervision_kind is not 'count_ce'"
            )
        squeezed = False
        x = self.decode_summary(state) if (self.use_decoded_markov_sketch or self.use_summary_spec) else state
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeezed = True
        logits = self.root_count_classifier(x)
        return logits.squeeze(0) if squeezed else logits

    def predict_canonical_count_from_state(self, state: torch.Tensor) -> torch.Tensor:
        # When root_supervision_kind is count_ce and the classifier exists,
        # always use CE argmax — even under summary_spec / factored surfaces.
        if self.root_supervision_kind == "count_ce" and self.root_count_classifier is not None:
            logits = self.predict_root_count_logits_from_state(state)
            class_values = self.root_count_class_values.to(
                device=logits.device,
                dtype=logits.dtype,
            )
            pred_idx = torch.argmax(logits, dim=-1)
            if pred_idx.ndim == 0:
                return class_values[int(pred_idx.item())]
            return class_values.index_select(0, pred_idx)
        if self.use_summary_spec:
            if self.uses_unified_readout():
                return self.predict_count_from_state(state)
            return self.predict_task_count_from_state(state)
        if self.use_decoded_markov_sketch:
            return self.predict_count_from_state(state)
        if self.root_supervision_kind != "count_ce" or self.root_count_classifier is None:
            return self.predict_count_from_state(state)
        logits = self.predict_root_count_logits_from_state(state)
        class_values = self.root_count_class_values.to(
            device=logits.device,
            dtype=logits.dtype,
        )
        pred_idx = torch.argmax(logits, dim=-1)
        if pred_idx.ndim == 0:
            return class_values[int(pred_idx.item())]
        return class_values.index_select(0, pred_idx)

    def decode_markov_codec(self, state: torch.Tensor) -> DecodedMarkovSketch:
        count = self.predict_count_from_state(state)
        _h, first_logits, last_logits = self._split_state(state)
        return DecodedMarkovSketch(
            count=count,
            first=torch.argmax(first_logits, dim=-1),
            last=torch.argmax(last_logits, dim=-1),
        )

    def decode_summary(self, state: torch.Tensor) -> torch.Tensor:
        if self.use_summary_spec:
            count_norm, first_logits, last_logits = self._decode_markov_summary_components(
                state
            )
            if count_norm.ndim == 0:
                count_norm = count_norm.unsqueeze(0)
            else:
                count_norm = count_norm.unsqueeze(-1)
            return torch.cat([count_norm, first_logits, last_logits], dim=-1)
        if not self.use_decoded_markov_sketch:
            return state
        pred_norm = self.predict_norm_from_state(state)
        _h, first_logits, last_logits = self._split_state(state)
        if pred_norm.ndim == 0:
            pred_norm = pred_norm.unsqueeze(0)
        else:
            pred_norm = pred_norm.unsqueeze(-1)
        return torch.cat([pred_norm, first_logits, last_logits], dim=-1)

    def encode_summary(self, summary: torch.Tensor) -> torch.Tensor:
        x = summary
        if x.ndim == 0:
            x = x.unsqueeze(0)
        if (
            self.use_summary_spec
            and self.use_direct_markov_sketch_slots
            and x.shape[-1] == int(self.summary_dim)
        ):
            count_norm = x[..., 0]
            first_logits = x[..., 1 : 1 + int(self.n_regimes)]
            last_logits = x[..., 1 + int(self.n_regimes) :]
            if self.use_opaque_carrier_exact_sketch_surface:
                carrier = self.summary_encoder(x)
                return self._canonical_summary_state_from_components(
                    count_norm=count_norm,
                    first_logits=first_logits,
                    last_logits=last_logits,
                    residual=carrier,
                )
            if self.use_carrier_projection_surface:
                return self._canonical_summary_state_from_components(
                    count_norm=count_norm,
                    first_logits=first_logits,
                    last_logits=last_logits,
                )
        # Unified-g: wide summary surface → state via g.
        if self.tree_model_version == "unified_g" and self.unified_g_summary_dim > 0:
            if x.shape[-1] == int(self.unified_g_summary_dim):
                return self.summary_encoder(x)
            # Narrow input (C2 reencode path): pad to wide format with zeros
            # so it still goes through the same g.
            if x.shape[-1] < int(self.unified_g_summary_dim):
                pad = torch.zeros(
                    (*x.shape[:-1], int(self.unified_g_summary_dim) - int(x.shape[-1])),
                    device=x.device, dtype=x.dtype,
                )
                return self.summary_encoder(torch.cat([x, pad], dim=-1))
        if self.use_summary_spec:
            if x.shape[-1] != int(self.summary_dim):
                raise ValueError("decoded summary surface must have shape (..., summary_dim)")
            if self.use_opaque_carrier_exact_sketch_surface:
                count_norm = x[..., 0]
                first_logits = x[..., 1 : 1 + int(self.n_regimes)]
                last_logits = x[..., 1 + int(self.n_regimes) :]
                carrier = self.summary_encoder(x)
                return self._canonical_summary_state_from_components(
                    count_norm=count_norm,
                    first_logits=first_logits,
                    last_logits=last_logits,
                    residual=carrier,
                )
            if self.use_carrier_projection_surface:
                count_norm = x[..., 0]
                first_logits = x[..., 1 : 1 + int(self.n_regimes)]
                last_logits = x[..., 1 + int(self.n_regimes) :]
                return self._canonical_summary_state_from_components(
                    count_norm=count_norm,
                    first_logits=first_logits,
                    last_logits=last_logits,
                )
            if self.use_shared_theorem_surface:
                return self.summary_encoder(x)
            if self.theorem_reencoder is None:
                raise RuntimeError("theorem reencoder is not initialized")
            encoded = self.theorem_reencoder(x)
            count_slot, first_slot, last_slot = torch.split(
                encoded,
                [
                    int(self.count_theorem_dim),
                    int(self.first_theorem_dim),
                    int(self.last_theorem_dim),
                ],
                dim=-1,
            )
            residual = torch.zeros(
                (*encoded.shape[:-1], int(self.residual_dim)),
                device=encoded.device,
                dtype=encoded.dtype,
            )
            return self._pack_summary_spec_state(
                count_slot,
                first_slot,
                last_slot,
                residual,
            )
        if self.use_decoded_markov_sketch or self.use_summary_spec:
            if x.shape[-1] != int(self.summary_dim):
                raise ValueError("decoded summary surface must have shape (..., summary_dim)")
            return self.summary_encoder(x)
        if x.shape[-1] == int(self.summary_dim):
            return x
        if x.shape[-1] != 1:
            x = x.unsqueeze(-1)
        h = self.summary_encoder(x)
        zeros = torch.zeros(
            (*h.shape[:-1], 2 * int(self.n_regimes)),
            device=h.device, dtype=h.dtype,
        )
        return torch.cat([h, zeros], dim=-1)

    def _exact_projected_merge_state(
        self,
        left_state: torch.Tensor,
        right_state: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_markov_summary_spec:
            raise RuntimeError(
                "exact projected merge requested without markov summary spec"
            )
        left_count = self.predict_count_from_state(left_state)
        right_count = self.predict_count_from_state(right_state)
        _left_count_norm, left_first_logits, left_last_logits = (
            self._decode_markov_summary_components(left_state)
        )
        _right_count_norm, right_first_logits, right_last_logits = (
            self._decode_markov_summary_components(right_state)
        )
        left_first = _straight_through_one_hot_from_logits(left_first_logits)
        left_last = _straight_through_one_hot_from_logits(left_last_logits)
        right_first = _straight_through_one_hot_from_logits(right_first_logits)
        right_last = _straight_through_one_hot_from_logits(right_last_logits)
        soft_join = 1.0 - torch.sum(
            torch.softmax(left_last_logits, dim=-1)
            * torch.softmax(right_first_logits, dim=-1),
            dim=-1,
        )
        hard_join = 1.0 - torch.sum(left_last * right_first, dim=-1)
        join_bit = hard_join + soft_join - soft_join.detach()
        merged_count = left_count + right_count + join_bit
        # Lean's exact Markov merge is additive on the count sketch.
        # Do not clip internal normalized counts here; clipping hides
        # leaf-sketch errors and breaks equivalence with exact re-merging
        # of the decoded leaves.
        merged_count_norm = merged_count / float(self.target_scale)
        merged_summary = torch.cat(
            [
                merged_count_norm.unsqueeze(-1),
                left_first,
                right_last,
            ],
            dim=-1,
        )
        if self.use_opaque_carrier_exact_sketch_surface:
            if self.carrier_state_merger is None:
                raise RuntimeError(
                    "opaque carrier exact merge requested without carrier_state_merger"
                )
            merged_carrier = self.carrier_state_merger(
                torch.cat(
                    [
                        self._opaque_carrier_from_state(left_state),
                        self._opaque_carrier_from_state(right_state),
                    ],
                    dim=-1,
                )
            )
            return self._canonical_summary_state_from_components(
                count_norm=merged_count_norm,
                first_logits=left_first,
                last_logits=right_last,
                residual=merged_carrier,
            )
        return self.encode_summary(merged_summary)

    def _merge_summary_spec_states(
        self,
        left_state: torch.Tensor,
        right_state: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_summary_spec:
            raise RuntimeError("summary spec merge requested without slot-structured modules")
        if self.exact_projected_merge_is_runtime_merge:
            return self._exact_projected_merge_state(left_state, right_state)
        if self.use_shared_theorem_surface:
            if self.summary_state_merger is None:
                raise RuntimeError(
                    "shared-theorem summary merge requested without merger"
                )
            return self.summary_state_merger(
                torch.cat([left_state, right_state], dim=-1)
            )
        if self.count_slot_merger is None:
            raise RuntimeError("summary spec merge requested without slot-structured modules")
        left_first = self._first_slot(left_state)
        left_last = self._last_slot(left_state)
        right_first = self._first_slot(right_state)
        right_last = self._last_slot(right_state)
        if self.use_direct_markov_sketch_slots:
            left_first = self._canonical_direct_endpoint_slot(left_first)
            left_last = self._canonical_direct_endpoint_slot(left_last)
            right_first = self._canonical_direct_endpoint_slot(right_first)
            right_last = self._canonical_direct_endpoint_slot(right_last)
        merged_count = self.count_slot_merger(
            torch.cat(
                [
                    self._count_slot(left_state),
                    self._count_slot(right_state),
                    left_last,
                    right_first,
                ],
                dim=-1,
            )
        )
        if self.use_direct_markov_sketch_slots:
            merged_count = self._runtime_count_slot(merged_count)
        if self.residual_slot_merger is None or int(self.residual_dim) <= 0:
            merged_residual = merged_count.new_zeros((*merged_count.shape[:-1], 0))
        else:
            merged_residual = self.residual_slot_merger(
                torch.cat(
                    [
                        self._residual_slots_flat(left_state),
                        self._residual_slots_flat(right_state),
                    ],
                    dim=-1,
                )
            )
        return self._pack_summary_spec_state(
            merged_count,
            left_first,
            right_last,
            merged_residual,
        )

    def _merge_state_pairs(
        self,
        left_states: torch.Tensor,
        right_states: torch.Tensor,
    ) -> torch.Tensor:
        # Unified-g: merge = compose(left, right) → wide summary → g (same g as leaves).
        if self.tree_model_version == "unified_g" and self.unified_g_merge_summary_proj is not None:
            merged_summary = self.unified_g_merge_summary_proj(
                torch.cat([left_states, right_states], dim=-1)
            )
            return self.encode_summary(merged_summary)
        if self.use_summary_spec:
            return self._merge_summary_spec_states(left_states, right_states)
        left_h, left_first, left_last = self._split_state(left_states)
        right_h, right_first, right_last = self._split_state(right_states)
        merger_input = torch.cat(
            [left_h, right_h, left_last, right_first],
            dim=-1,
        )
        if self.merger is None:
            raise RuntimeError("legacy merger is not initialized")
        merged_h = self.merger(merger_input)
        return self._pack_state(merged_h, left_first, right_last)

    def _merge_states(
        self,
        states: Sequence[torch.Tensor] | torch.Tensor,
        *,
        schedule: ScheduleName,
        collect_merge_states: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if isinstance(states, torch.Tensor):
            if states.ndim == 1:
                state_list = [states]
            elif states.ndim >= 2:
                state_list = [states[i] for i in range(int(states.shape[0]))]
            else:
                raise ValueError("states tensor must be rank-1 or rank-2")
        else:
            state_list = list(states)
        if len(state_list) == 0:
            raise ValueError("need at least one state")
        if len(state_list) == 1:
            return state_list[0], []

        merged_states: List[torch.Tensor] = []
        if str(schedule) == "balanced":
            cur = list(state_list)
            while len(cur) > 1:
                nxt: List[torch.Tensor] = []
                pair_count = int(len(cur) // 2)
                if pair_count > 0:
                    left_batch = torch.stack(cur[0 : 2 * pair_count : 2], dim=0)
                    right_batch = torch.stack(cur[1 : 2 * pair_count : 2], dim=0)
                    merged_batch = self._merge_state_pairs(left_batch, right_batch)
                    if merged_batch.ndim == 1:
                        merged_batch = merged_batch.unsqueeze(0)
                    if collect_merge_states:
                        merged_states.extend(
                            [merged_batch[idx] for idx in range(int(merged_batch.shape[0]))]
                        )
                    nxt.extend([merged_batch[idx] for idx in range(int(merged_batch.shape[0]))])
                if len(cur) % 2 == 1:
                    nxt.append(cur[-1])
                cur = nxt
            return cur[0], merged_states

        if str(schedule) in ("left_to_right", "right_to_left"):
            if str(schedule) == "left_to_right":
                acc = state_list[0]
                for st in state_list[1:]:
                    acc = self._merge_state_pairs(acc, st)
                    if collect_merge_states:
                        merged_states.append(acc)
                return acc, merged_states

            acc = state_list[-1]
            for st in reversed(state_list[:-1]):
                acc = self._merge_state_pairs(st, acc)
                if collect_merge_states:
                    merged_states.append(acc)
            return acc, merged_states

        raise ValueError(f"unsupported schedule: {schedule!r}")

    def forward_doc(
        self,
        leaf_token_ids: Sequence[Sequence[int]],
        leaf_counts: Sequence[float],
        merge_counts_balanced: Sequence[float],
        merge_token_lengths: Sequence[int] | None = None,
        *,
        schedule: ScheduleName,
        collect_leaf: bool,
        collect_c3: bool,
        collect_c2: bool,
        device: torch.device,
        leaf_audit_indices: Optional[set] = None,
        c3_audit_indices: Optional[set] = None,
        leaf_first_regimes: Optional[Sequence[int]] = None,
        leaf_last_regimes: Optional[Sequence[int]] = None,
        internal_supervision_kind: str = "count_only",
        leaf_exact_supervision: bool = False,
        leaf_supervision_kind: str = "full_sketch",
        tree_local_weighting_mode: str = "fixed_k_hajek",
        tree_supervision_source: str = "rate",
        depth_discount_gamma: float = 1.0,
        defer_contrastive: bool = False,
        leaf_supervision_population_size: int | None = None,
        internal_supervision_population_size: int | None = None,
        precomputed_state_batch: torch.Tensor | None = None,
        precomputed_root_state: torch.Tensor | None = None,
        precomputed_merge_states: Sequence[torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor | float]:
        """Forward pass with local law loss collection."""
        if len(leaf_token_ids) == 0:
            raise ValueError("leaf_token_ids must be non-empty")

        if precomputed_state_batch is None:
            state_batch = self.encode_leaf_tokens_batch(leaf_token_ids, device=device)
            states = [state_batch[idx] for idx in range(int(state_batch.shape[0]))]
            root_state, merge_states = self._merge_states(
                states, schedule=schedule,
                collect_merge_states=(collect_c3 or collect_c2) and str(schedule) == "balanced",
            )
        else:
            state_batch = precomputed_state_batch
            if state_batch.ndim == 1:
                state_batch = state_batch.unsqueeze(0)
            states = [state_batch[idx] for idx in range(int(state_batch.shape[0]))]
            if precomputed_root_state is None:
                root_state, merge_states = self._merge_states(
                    states,
                    schedule=schedule,
                    collect_merge_states=(collect_c3 or collect_c2) and str(schedule) == "balanced",
                )
            else:
                root_state = precomputed_root_state
                if precomputed_merge_states is None:
                    _unused_root_state, merge_states = self._merge_states(
                        states,
                        schedule=schedule,
                        collect_merge_states=(collect_c3 or collect_c2)
                        and str(schedule) == "balanced",
                    )
                else:
                    merge_states = list(precomputed_merge_states)
        exact_targets = (
            _balanced_exact_sketch_targets(
                leaf_counts=leaf_counts,
                leaf_first_regimes=leaf_first_regimes,
                leaf_last_regimes=leaf_last_regimes,
            )
            if (
                leaf_first_regimes is not None
                and leaf_last_regimes is not None
                and len(leaf_first_regimes) == len(states)
                and len(leaf_last_regimes) == len(states)
            )
            else None
        )
        feature_targets = (
            theorem_feature_targets_from_markov_exact_targets(
                adapter=self.theorem_feature_adapter,
                exact_targets=exact_targets,
                leaf_metadata=tuple(
                    {
                        "span_length": int(len(token_ids)),
                        "leaf_span_count": 1,
                    }
                    for token_ids in list(leaf_token_ids)
                ),
                merge_metadata=tuple(
                    {
                        "span_length": int(length),
                    }
                    for length in list(merge_token_lengths or ())
                ),
                root_metadata=(
                    {
                        "span_length": int(sum(len(token_ids) for token_ids in list(leaf_token_ids))),
                        "leaf_span_count": int(len(leaf_token_ids)),
                    },
                ),
            )
            if exact_targets is not None
            else None
        )
        pred_norm = self.predict_norm_from_state(root_state)
        normalized_weighting_mode = _normalize_tree_local_weighting_mode(
            tree_local_weighting_mode
        )
        normalized_supervision_source = _normalize_tree_supervision_source(
            tree_supervision_source
        )
        c2_pair_weighting_mode = _c2_pair_weighting_mode(
            tree_supervision_source=normalized_supervision_source,
            local_estimand_mode=normalized_weighting_mode,
        )
        n_leaves = int(len(states))
        tree_layout = self._balanced_tree_layout(n_leaves)
        depth_by_global_idx = dict(tree_layout.get("depth_by_global_idx") or {})
        leaf_depth = max(
            [int(depth_by_global_idx.get(int(idx), 0)) for idx in range(n_leaves)],
            default=0,
        )
        doc_token_count = int(
            max(1, sum(int(len(token_ids)) for token_ids in list(leaf_token_ids)))
        )
        gamma = float(depth_discount_gamma)
        local_loss_kind = _resolved_local_loss_kind(
            leaf_supervision_kind=str(leaf_supervision_kind),
            internal_supervision_kind=str(internal_supervision_kind),
        )
        out: Dict[str, torch.Tensor | float] = {
            "root_state": root_state,
            "pred_norm": pred_norm,
            "pred_count": self.predict_count_from_state(root_state),
            "pred_task_count": self.predict_task_count_from_state(root_state),
            "pred_root_count_canonical": self.predict_canonical_count_from_state(
                root_state
            ),
            "tree_local_weighting_mode": normalized_weighting_mode,
            "local_loss_kind": local_loss_kind,
            "local_sampling_design_name": (
                "manifest_explicit_deterministic_ordering"
                if normalized_supervision_source == "manifest"
                else "deterministic_fixed_k_uniform"
            ),
            "depth_discount_gamma": float(gamma),
            "c2_pair_weighting_mode": str(c2_pair_weighting_mode),
        }
        if feature_targets is not None and feature_targets.root:
            out["root_feature_target"] = feature_targets.root[0]
            out["root_task_target"] = float(
                self.task_target_from_label(feature_targets.root[0])
            )
        zero_term = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
        component_terms: Dict[str, torch.Tensor] = {
            "leaf_count_loss": zero_term.clone(),
            "leaf_first_loss": zero_term.clone(),
            "leaf_last_loss": zero_term.clone(),
            "merge_count_loss": zero_term.clone(),
            "merge_first_loss": zero_term.clone(),
            "merge_last_loss": zero_term.clone(),
            "c2_count_loss": zero_term.clone(),
            "c2_first_loss": zero_term.clone(),
            "c2_last_loss": zero_term.clone(),
            "c2_join_loss": zero_term.clone(),
            "c2_on_range_reencode_loss": zero_term.clone(),
            "phi_compose_loss": zero_term.clone(),
            "phi_contrastive_loss": zero_term.clone(),
        }
        leaf_raw_numerator = zero_term.clone()
        leaf_weight_denominator = zero_term.clone()
        merge_raw_numerator = zero_term.clone()
        merge_weight_denominator = zero_term.clone()

        # C1: Leaf supervision
        if collect_leaf:
            leaf_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            leaf_count_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            leaf_first_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            leaf_last_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            leaf_weight_sum = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            lc = 0
            leaf_population_size, leaf_sample_size, leaf_effective_propensity = (
                _effective_local_sampling_summary(
                    int(
                        len(states)
                        if leaf_supervision_population_size is None
                        else leaf_supervision_population_size
                    ),
                    leaf_audit_indices,
                )
            )
            for idx, (st, truth) in enumerate(zip(states, leaf_counts)):
                if leaf_audit_indices is not None and idx not in leaf_audit_indices:
                    continue
                if normalized_weighting_mode == "subset_mean":
                    leaf_weight_scalar = 1.0
                elif normalized_weighting_mode == "fixed_k_hajek":
                    leaf_weight_scalar = 1.0 / max(
                        float(leaf_effective_propensity),
                        1e-12,
                    )
                else:
                    leaf_span_mass = float(len(leaf_token_ids[int(idx)])) / float(
                        max(1, doc_token_count)
                    )
                    leaf_depth = int(depth_by_global_idx.get(int(idx), 0))
                    leaf_weight_scalar = (
                        float(leaf_span_mass)
                        * (float(gamma) ** int(max(0, leaf_depth)))
                        / max(float(leaf_effective_propensity), 1e-12)
                    )
                leaf_sample_weight = torch.tensor(
                    float(leaf_weight_scalar),
                    device=pred_norm.device,
                    dtype=pred_norm.dtype,
                )
                if (
                    str(leaf_supervision_kind).strip().lower() == "full_sketch"
                    and self.use_shared_theorem_surface
                    and feature_targets is not None
                ):
                    legacy_leaf_terms = _theorem_feature_task_supervision_terms(
                        self,
                        st,
                        truth_target=self.task_target_from_label(
                            feature_targets.leaf[int(idx)]
                        ),
                    )
                    if self.use_factorized_score_fiber_surface and self.use_summary_spec and exact_targets is not None:
                        _, exact_first, exact_last = exact_targets["leaf"][idx]
                        summary_terms = _local_supervision_terms(
                            self,
                            st,
                            truth_count=float(truth),
                            truth_first=int(exact_first),
                            truth_last=int(exact_last),
                            supervision_kind="full_sketch",
                        )
                        leaf_loss = (
                            leaf_loss
                            + leaf_sample_weight
                            * (
                                legacy_leaf_terms["total_loss"]
                                + summary_terms["total_loss"]
                            )
                        )
                        leaf_count_loss = (
                            leaf_count_loss
                            + leaf_sample_weight
                            * (
                                legacy_leaf_terms["task_loss"]
                                + summary_terms["count_loss"]
                            )
                        )
                        leaf_first_loss = (
                            leaf_first_loss
                            + leaf_sample_weight * summary_terms["first_loss"]
                        )
                        leaf_last_loss = (
                            leaf_last_loss
                            + leaf_sample_weight * summary_terms["last_loss"]
                        )
                    else:
                        leaf_loss = (
                            leaf_loss
                            + leaf_sample_weight * legacy_leaf_terms["total_loss"]
                        )
                        leaf_count_loss = (
                            leaf_count_loss
                            + leaf_sample_weight * legacy_leaf_terms["task_loss"]
                        )
                else:
                    exact_first: int | None = None
                    exact_last: int | None = None
                    if exact_targets is not None:
                        _, exact_first_raw, exact_last_raw = exact_targets["leaf"][idx]
                        exact_first = int(exact_first_raw)
                        exact_last = int(exact_last_raw)
                    leaf_terms = _local_supervision_terms(
                        self,
                        st,
                        truth_count=float(truth),
                        truth_first=exact_first,
                        truth_last=exact_last,
                        supervision_kind=str(leaf_supervision_kind),
                    )
                    leaf_loss = leaf_loss + leaf_sample_weight * leaf_terms["total_loss"]
                    leaf_count_loss = (
                        leaf_count_loss + leaf_sample_weight * leaf_terms["count_loss"]
                    )
                    leaf_first_loss = (
                        leaf_first_loss + leaf_sample_weight * leaf_terms["first_loss"]
                    )
                    leaf_last_loss = (
                        leaf_last_loss + leaf_sample_weight * leaf_terms["last_loss"]
                    )
                lc += 1
                leaf_weight_sum = leaf_weight_sum + leaf_sample_weight
            leaf_raw_numerator = leaf_loss
            leaf_weight_denominator = leaf_weight_sum
            if normalized_weighting_mode == "span_mass_ipw_sum":
                out["leaf_loss"] = torch.where(
                    leaf_weight_sum > 0,
                    leaf_loss,
                    torch.zeros_like(leaf_loss),
                )
            else:
                out["leaf_loss"] = torch.where(
                    leaf_weight_sum > 0,
                    leaf_loss / leaf_weight_sum.clamp_min(1e-12),
                    torch.zeros_like(leaf_loss),
                )
            out["leaf_count"] = float(lc)
            out["leaf_population_size"] = float(leaf_population_size)
            out["leaf_sample_size"] = float(leaf_sample_size)
            out["leaf_effective_propensity"] = float(leaf_effective_propensity)
            if normalized_weighting_mode == "span_mass_ipw_sum":
                component_terms["leaf_count_loss"] = torch.where(
                    leaf_weight_sum > 0,
                    leaf_count_loss,
                    torch.zeros_like(leaf_count_loss),
                )
                component_terms["leaf_first_loss"] = torch.where(
                    leaf_weight_sum > 0,
                    leaf_first_loss,
                    torch.zeros_like(leaf_first_loss),
                )
                component_terms["leaf_last_loss"] = torch.where(
                    leaf_weight_sum > 0,
                    leaf_last_loss,
                    torch.zeros_like(leaf_last_loss),
                )
            else:
                component_terms["leaf_count_loss"] = torch.where(
                    leaf_weight_sum > 0,
                    leaf_count_loss / leaf_weight_sum.clamp_min(1e-12),
                    torch.zeros_like(leaf_count_loss),
                )
                component_terms["leaf_first_loss"] = torch.where(
                    leaf_weight_sum > 0,
                    leaf_first_loss / leaf_weight_sum.clamp_min(1e-12),
                    torch.zeros_like(leaf_first_loss),
                )
                component_terms["leaf_last_loss"] = torch.where(
                    leaf_weight_sum > 0,
                    leaf_last_loss / leaf_weight_sum.clamp_min(1e-12),
                    torch.zeros_like(leaf_last_loss),
                )
        else:
            out["leaf_loss"] = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            out["leaf_count"] = 0.0
            out["leaf_population_size"] = 0.0
            out["leaf_sample_size"] = 0.0
            out["leaf_effective_propensity"] = 0.0

        # C3 = L2: Merge consistency (Lean: one_pass / nodewise_preservation).
        #
        # Two components, following the Lean's certifiedRegularizedObjective:
        #
        # 1. Oracle supervision: compare merger output directly against oracle
        #    count/first/last of the merged span.  Gives the merger a fixed,
        #    correct target (no self-reference, no encoder interference).
        #
        # 2. Algebraic consistency (secondary): self-consistent merge algebra
        #    count(P) = count(L) + count(R) + join.  Child predictions are
        #    detached so gradient flows only to the merger.
        if collect_c3 and self.use_markov_summary_spec:
            c3_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            merge_count_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            merge_first_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            merge_last_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            merge_weight_sum = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            c3c = 0
            merge_population_size, merge_sample_size, merge_effective_propensity = (
                _effective_local_sampling_summary(
                    int(
                        len(merge_states)
                        if internal_supervision_population_size is None
                        else internal_supervision_population_size
                    ),
                    c3_audit_indices,
                )
            )
            if merge_states:
                all_states_for_merge = list(states) + list(merge_states)
                children_map = self._balanced_merge_children_map(len(states))
                for idx, st in enumerate(merge_states):
                    if c3_audit_indices is not None and idx not in c3_audit_indices:
                        continue
                    if idx not in children_map:
                        continue
                    if idx >= len(merge_counts_balanced):
                        continue
                    if normalized_weighting_mode == "subset_mean":
                        merge_weight_scalar = 1.0
                    elif normalized_weighting_mode == "fixed_k_hajek":
                        merge_weight_scalar = 1.0 / max(
                            float(merge_effective_propensity),
                            1e-12,
                        )
                    else:
                        merge_span_length = int(
                            merge_token_lengths[int(idx)]
                            if merge_token_lengths is not None
                            and int(idx) < len(merge_token_lengths)
                            else 0
                        )
                        merge_span_mass = float(merge_span_length) / float(
                            max(1, doc_token_count)
                        )
                        merge_depth = int(
                            depth_by_global_idx.get(int(n_leaves) + int(idx), 0)
                        )
                        merge_weight_scalar = (
                            float(merge_span_mass)
                            * (float(gamma) ** int(max(0, merge_depth)))
                            / max(float(merge_effective_propensity), 1e-12)
                        )
                    merge_sample_weight = torch.tensor(
                        float(merge_weight_scalar),
                        device=pred_norm.device,
                        dtype=pred_norm.dtype,
                    )
                    # --- Oracle supervision (primary C3 signal) ---
                    # Direct MSE against oracle count of the merged span.
                    # Gradient flows: loss → predict_count → parent_state → merger.
                    exact_first: int | None = None
                    exact_last: int | None = None
                    if exact_targets is not None and idx < len(exact_targets["merge"]):
                        _, merge_first_val, merge_last_val = exact_targets["merge"][idx]
                        exact_first = int(merge_first_val)
                        exact_last = int(merge_last_val)
                    oracle_terms = _local_supervision_terms(
                        self,
                        st,
                        truth_count=float(merge_counts_balanced[idx]),
                        truth_first=exact_first,
                        truth_last=exact_last,
                        supervision_kind=str(internal_supervision_kind),
                    )
                    c3_loss = c3_loss + merge_sample_weight * oracle_terms["total_loss"]
                    merge_count_loss = merge_count_loss + merge_sample_weight * oracle_terms["count_loss"]
                    merge_first_loss = merge_first_loss + merge_sample_weight * oracle_terms["first_loss"]
                    merge_last_loss = merge_last_loss + merge_sample_weight * oracle_terms["last_loss"]
                    # --- Algebraic consistency + join BCE supervision ---
                    left_idx, right_idx = children_map[idx]
                    left_state = all_states_for_merge[left_idx]
                    right_state = all_states_for_merge[right_idx]
                    oracle_join_bit = None
                    if exact_targets is not None and idx < len(exact_targets["merge_join_bits"]):
                        oracle_join_bit = int(exact_targets["merge_join_bits"][idx])
                    mc_terms = _summary_spec_merge_consistency_terms(
                        self, left_state, right_state, st,
                        truth_join_bit=oracle_join_bit,
                    )
                    c3_loss = c3_loss + merge_sample_weight * mc_terms["total_loss"]
                    c3c += 1
                    merge_weight_sum = merge_weight_sum + merge_sample_weight
            merge_raw_numerator = c3_loss
            merge_weight_denominator = merge_weight_sum
            if normalized_weighting_mode == "span_mass_ipw_sum":
                out["c3_loss"] = torch.where(
                    merge_weight_sum > 0,
                    c3_loss,
                    torch.zeros_like(c3_loss),
                )
            else:
                out["c3_loss"] = torch.where(
                    merge_weight_sum > 0,
                    c3_loss / merge_weight_sum.clamp_min(1e-12),
                    torch.zeros_like(c3_loss),
                )
            out["c3_count"] = float(c3c)
            out["merge_population_size"] = float(merge_population_size)
            out["merge_sample_size"] = float(merge_sample_size)
            out["merge_effective_propensity"] = float(merge_effective_propensity)
            if normalized_weighting_mode == "span_mass_ipw_sum":
                component_terms["merge_count_loss"] = torch.where(
                    merge_weight_sum > 0,
                    merge_count_loss,
                    torch.zeros_like(merge_count_loss),
                )
                component_terms["merge_first_loss"] = torch.where(
                    merge_weight_sum > 0,
                    merge_first_loss,
                    torch.zeros_like(merge_first_loss),
                )
                component_terms["merge_last_loss"] = torch.where(
                    merge_weight_sum > 0,
                    merge_last_loss,
                    torch.zeros_like(merge_last_loss),
                )
            else:
                component_terms["merge_count_loss"] = torch.where(
                    merge_weight_sum > 0,
                    merge_count_loss / merge_weight_sum.clamp_min(1e-12),
                    torch.zeros_like(merge_count_loss),
                )
                component_terms["merge_first_loss"] = torch.where(
                    merge_weight_sum > 0,
                    merge_first_loss / merge_weight_sum.clamp_min(1e-12),
                    torch.zeros_like(merge_first_loss),
                )
                component_terms["merge_last_loss"] = torch.where(
                    merge_weight_sum > 0,
                    merge_last_loss / merge_weight_sum.clamp_min(1e-12),
                    torch.zeros_like(merge_last_loss),
                )
        else:
            out["c3_loss"] = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            out["c3_count"] = 0.0
            out["merge_population_size"] = 0.0
            out["merge_sample_size"] = 0.0
            out["merge_effective_propensity"] = 0.0

        # C2: Markov count drift under re-summary.
        if collect_c2:
            c2_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            c2_state_replay_mse = torch.zeros(
                (), device=pred_norm.device, dtype=pred_norm.dtype
            )
            c2_count_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            c2_first_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            c2_last_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            c2_join_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            c2_on_range_reencode_loss = torch.zeros(
                (), device=pred_norm.device, dtype=pred_norm.dtype
            )
            c2c = 0
            c2_same_pair_count = torch.zeros(
                (), device=pred_norm.device, dtype=pred_norm.dtype
            )
            c2_different_pair_count = torch.zeros(
                (), device=pred_norm.device, dtype=pred_norm.dtype
            )
            c2_pair_weight_ess = torch.zeros(
                (), device=pred_norm.device, dtype=pred_norm.dtype
            )
            c2_pair_weight_max = torch.zeros(
                (), device=pred_norm.device, dtype=pred_norm.dtype
            )
            # ---- Fiber C2: contrastive loss over f*-equivalence classes ----
            if (
                self.use_shared_theorem_surface
                and feature_targets is not None
                and str(schedule) == "balanced"
            ):
                _fiber_phi_features: List[torch.Tensor] = []
                _fiber_labels: List[Any] = []
                _fiber_oracle_vecs: List[Any] = []
                _fiber_kind_codes: List[int] = []
                _fiber_node_scales: List[float] = []

                def _append_c2_feature(
                    state: torch.Tensor,
                    target: Any,
                    *,
                    node_kind: int,
                    node_scale: float,
                ) -> None:
                    _fiber_phi_features.append(self.predict_phi_from_state(state))
                    _fiber_labels.append(target)
                    _fiber_kind_codes.append(int(node_kind))
                    _fiber_node_scales.append(float(node_scale))
                    if self.oracle_metric is not None:
                        _fiber_oracle_vecs.append(
                            self.oracle_metric.oracle_vector(
                                count=float(target.count),
                                first=int(target.first),
                                last=int(target.last),
                            )
                        )

                for idx in _all_or_sampled_indices(len(states), leaf_audit_indices):
                    if int(idx) < len(feature_targets.leaf):
                        _append_c2_feature(
                            states[int(idx)],
                            feature_targets.leaf[int(idx)],
                            node_kind=_C2_NODE_KIND_LEAF,
                            node_scale=float(
                                (float(len(leaf_token_ids[int(idx)])) / float(max(1, doc_token_count)))
                                * (float(gamma) ** int(max(0, leaf_depth)))
                            ),
                        )
                for idx in _all_or_sampled_indices(len(merge_states), c3_audit_indices):
                    if int(idx) < len(feature_targets.merge):
                        _append_c2_feature(
                            merge_states[int(idx)],
                            feature_targets.merge[int(idx)],
                            node_kind=_C2_NODE_KIND_MERGE,
                            node_scale=float(
                                (
                                    float(
                                        merge_token_lengths[int(idx)]
                                        if merge_token_lengths is not None
                                        and int(idx) < len(merge_token_lengths)
                                        else 0
                                    )
                                    / float(max(1, doc_token_count))
                                )
                                * (float(gamma) ** int(max(0, depth_by_global_idx.get(int(n_leaves) + int(idx), 0))))
                            ),
                        )
                if feature_targets.root:
                    _append_c2_feature(
                        root_state,
                        feature_targets.root[0],
                        node_kind=_C2_NODE_KIND_ROOT,
                        node_scale=1.0,
                    )
                if len(_fiber_phi_features) > 1:
                    if (
                        c2_pair_weighting_mode == "pair_ipw_geomean"
                        and self.oracle_metric is not None
                    ):
                        raise ValueError(
                            "authoritative C2 pair weighting does not support oracle_metric mode"
                        )
                    if self.oracle_metric is not None:
                        _fiber_pair_data = build_contrastive_pairs(
                            _fiber_oracle_vecs,
                            metric=self.oracle_metric,
                            same_threshold=self.oracle_same_threshold,
                            diff_threshold=self.oracle_diff_threshold,
                        )
                        c2_loss = contrastive_fiber_loss(
                            torch.stack(_fiber_phi_features, dim=0),
                            _fiber_pair_data,
                            margin=float(SUMMARY_SPEC_PHI_DIFFERENT_MARGIN),
                        )
                    else:
                        _fiber_pairs = build_theorem_feature_pair_sets(
                            _fiber_labels,
                            adapter=self.theorem_feature_adapter,
                            same_threshold=self.theorem_pair_same_threshold,
                            diff_threshold=self.theorem_pair_diff_threshold,
                        )
                        same_pair_weights = None
                        different_pair_weights = None
                        pair_weights = None
                        if c2_pair_weighting_mode == "pair_ipw_geomean":
                            pair_weights = _c2_pair_weight_matrix(
                                node_scales=torch.tensor(
                                    _fiber_node_scales,
                                    device=pred_norm.device,
                                    dtype=pred_norm.dtype,
                                ),
                                node_kind_codes=torch.tensor(
                                    _fiber_kind_codes,
                                    device=pred_norm.device,
                                    dtype=torch.long,
                                ),
                                valid_mask=torch.ones(
                                    (len(_fiber_phi_features),),
                                    device=pred_norm.device,
                                    dtype=torch.bool,
                                ),
                                leaf_population_size=int(
                                    len(states)
                                    if leaf_supervision_population_size is None
                                    else leaf_supervision_population_size
                                ),
                                leaf_sample_size=_effective_subset_sample_size(
                                    int(
                                        len(states)
                                        if leaf_supervision_population_size is None
                                        else leaf_supervision_population_size
                                    ),
                                    leaf_audit_indices,
                                ),
                                merge_population_size=int(
                                    len(merge_states)
                                    if internal_supervision_population_size is None
                                    else internal_supervision_population_size
                                ),
                                merge_sample_size=_effective_subset_sample_size(
                                    int(
                                        len(merge_states)
                                        if internal_supervision_population_size is None
                                        else internal_supervision_population_size
                                    ),
                                    c3_audit_indices,
                                ),
                            )
                            same_pair_weights = [
                                float(pair_weights[int(left_idx), int(right_idx)].detach().item())
                                for left_idx, right_idx in list(_fiber_pairs.same_pairs)
                            ]
                            different_pair_weights = [
                                float(pair_weights[int(left_idx), int(right_idx)].detach().item())
                                for left_idx, right_idx in list(_fiber_pairs.different_pairs)
                            ]
                        c2_loss = _pairwise_theorem_feature_contrastive_loss(
                            torch.stack(_fiber_phi_features, dim=0),
                            same_pairs=_fiber_pairs.same_pairs,
                            different_pairs=_fiber_pairs.different_pairs,
                            same_pair_weights=same_pair_weights,
                            different_pair_weights=different_pair_weights,
                        )
                        same_pair_mask = torch.zeros(
                            (len(_fiber_phi_features), len(_fiber_phi_features)),
                            device=pred_norm.device,
                            dtype=torch.bool,
                        )
                        different_pair_mask = torch.zeros_like(same_pair_mask)
                        for left_idx, right_idx in list(_fiber_pairs.same_pairs):
                            same_pair_mask[int(left_idx), int(right_idx)] = True
                            same_pair_mask[int(right_idx), int(left_idx)] = True
                        for left_idx, right_idx in list(_fiber_pairs.different_pairs):
                            different_pair_mask[int(left_idx), int(right_idx)] = True
                            different_pair_mask[int(right_idx), int(left_idx)] = True
                        pair_diag = _pairwise_weight_diagnostics_from_masks(
                            same_mask=same_pair_mask,
                            different_mask=different_pair_mask,
                            pair_weights=pair_weights,
                        )
                        c2_same_pair_count = pair_diag["same_pair_count"].to(dtype=pred_norm.dtype)
                        c2_different_pair_count = pair_diag["different_pair_count"].to(dtype=pred_norm.dtype)
                        c2_pair_weight_ess = pair_diag["pair_weight_ess"].to(dtype=pred_norm.dtype)
                        c2_pair_weight_max = pair_diag["pair_weight_max"].to(dtype=pred_norm.dtype)
                    c2c = 1
            elif (self.use_decoded_markov_sketch or self.use_summary_spec) and str(schedule) == "balanced":
                # C2 = L3 (on-range idempotence only).  Merge consistency
                # belongs to C3 = L2 and is handled in the c3_loss block.
                c2_states = list(states) + list(merge_states)
                for replay_state in c2_states:
                    if self.use_summary_spec:
                        on_range_terms = _summary_spec_on_range_reencode_terms(
                            self,
                            replay_state,
                        )
                        c2_loss = c2_loss + on_range_terms["total_loss"]
                        c2_count_loss = c2_count_loss + on_range_terms["count_loss"]
                        c2_first_loss = c2_first_loss + on_range_terms["first_loss"]
                        c2_last_loss = c2_last_loss + on_range_terms["last_loss"]
                        c2_on_range_reencode_loss = (
                            c2_on_range_reencode_loss + on_range_terms["total_loss"]
                        )
                        c2c += 1
            else:
                for st in list(states) + list(merge_states) or [root_state]:
                    base_norm, replay_norm, base_state, replay_state = _fno_summary_replay_tensors(
                        self, st
                    )
                    replay_loss = F.mse_loss(replay_norm, base_norm)
                    c2_loss = c2_loss + replay_loss
                    c2_count_loss = c2_count_loss + replay_loss
                    c2_state_replay_mse = c2_state_replay_mse + F.mse_loss(
                        replay_state, base_state
                    )
                    c2c += 1
            out["c2_loss"] = c2_loss / float(max(1, c2c))
            out["c2_state_replay_mse"] = c2_state_replay_mse / float(max(1, c2c))
            out["c2_count"] = float(c2c)
            out["c2_same_pair_count"] = float(c2_same_pair_count.detach().cpu())
            out["c2_different_pair_count"] = float(
                c2_different_pair_count.detach().cpu()
            )
            out["c2_pair_weight_ess"] = float(c2_pair_weight_ess.detach().cpu())
            out["c2_pair_weight_max"] = float(c2_pair_weight_max.detach().cpu())
            component_terms["c2_count_loss"] = c2_count_loss / float(max(1, c2c))
            component_terms["c2_first_loss"] = c2_first_loss / float(max(1, c2c))
            component_terms["c2_last_loss"] = c2_last_loss / float(max(1, c2c))
            component_terms["c2_join_loss"] = c2_join_loss / float(max(1, c2c))
            component_terms["c2_on_range_reencode_loss"] = (
                c2_on_range_reencode_loss / float(max(1, c2c))
            )
        else:
            out["c2_loss"] = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
            out["c2_state_replay_mse"] = torch.zeros(
                (), device=pred_norm.device, dtype=pred_norm.dtype
            )
            out["c2_count"] = 0.0
            out["c2_same_pair_count"] = 0.0
            out["c2_different_pair_count"] = 0.0
            out["c2_pair_weight_ess"] = 0.0
            out["c2_pair_weight_max"] = 0.0

        phi_compose_loss = torch.zeros((), device=pred_norm.device, dtype=pred_norm.dtype)
        phi_contrastive_loss = torch.zeros(
            (), device=pred_norm.device, dtype=pred_norm.dtype
        )
        if (
            self.use_shared_theorem_surface
            and str(schedule) == "balanced"
        ):
            all_states = list(states) + list(merge_states)
            children_map = self._balanced_merge_children_map(len(states))
            if merge_states:
                for idx, parent_state in enumerate(merge_states):
                    left_idx, right_idx = children_map[idx]
                    pred_phi = self.predict_phi_parent_from_children(
                        all_states[left_idx],
                        all_states[right_idx],
                    )
                    target_phi = self.predict_phi_from_state(parent_state)
                    phi_compose_loss = phi_compose_loss + _phi_alignment_loss(
                        pred_phi,
                        target_phi,
                        mode=self.phi_alignment_loss,
                    )
                phi_compose_loss = phi_compose_loss / float(max(1, len(merge_states)))
            if feature_targets is not None:
                phi_features: List[torch.Tensor] = []
                phi_labels: List[Any] = []
                phi_oracle_vecs: List[Any] = []

                def _append_phi_label(
                    state: torch.Tensor,
                    target: Any,
                ) -> None:
                    phi_features.append(self.predict_phi_from_state(state))
                    phi_labels.append(target)
                    if self.oracle_metric is not None:
                        phi_oracle_vecs.append(
                            self.oracle_metric.oracle_vector(
                                count=float(target.count),
                                first=int(target.first),
                                last=int(target.last),
                            )
                        )

                for idx in _all_or_sampled_indices(len(states), leaf_audit_indices):
                    if int(idx) < len(feature_targets.leaf):
                        _append_phi_label(states[int(idx)], feature_targets.leaf[int(idx)])
                for idx in _all_or_sampled_indices(len(merge_states), c3_audit_indices):
                    if int(idx) < len(feature_targets.merge):
                        _append_phi_label(
                            merge_states[int(idx)],
                            feature_targets.merge[int(idx)],
                        )
                if feature_targets.root:
                    _append_phi_label(root_state, feature_targets.root[0])
                if len(phi_features) > 1:
                    if defer_contrastive:
                        out["_deferred_phi_features"] = phi_features
                        if self.oracle_metric is not None:
                            out["_deferred_oracle_vectors"] = phi_oracle_vecs
                        else:
                            out["_deferred_phi_labels"] = phi_labels
                    elif self.oracle_metric is not None:
                        _phi_pair_data = build_contrastive_pairs(
                            phi_oracle_vecs,
                            metric=self.oracle_metric,
                            same_threshold=self.oracle_same_threshold,
                            diff_threshold=self.oracle_diff_threshold,
                        )
                        phi_contrastive_loss = contrastive_fiber_loss(
                            torch.stack(phi_features, dim=0),
                            _phi_pair_data,
                            margin=float(SUMMARY_SPEC_PHI_DIFFERENT_MARGIN),
                        )
                    else:
                        phi_pairs = build_theorem_feature_pair_sets(
                            phi_labels,
                            adapter=self.theorem_feature_adapter,
                            same_threshold=self.theorem_pair_same_threshold,
                            diff_threshold=self.theorem_pair_diff_threshold,
                        )
                        phi_contrastive_loss = _pairwise_theorem_feature_contrastive_loss(
                            torch.stack(phi_features, dim=0),
                            same_pairs=phi_pairs.same_pairs,
                            different_pairs=phi_pairs.different_pairs,
                        )
        out["phi_compose_loss"] = phi_compose_loss
        out["phi_contrastive_loss"] = phi_contrastive_loss
        component_terms["phi_compose_loss"] = phi_compose_loss
        component_terms["phi_contrastive_loss"] = phi_contrastive_loss

        out["local_objective_audit"] = {
            "weighting_mode": normalized_weighting_mode,
            "design_name": "deterministic_fixed_k_uniform",
            "leaf": {
                "population_size": float(out.get("leaf_population_size", 0.0)),
                "sample_size": float(out.get("leaf_sample_size", 0.0)),
                "effective_propensity": float(
                    out.get("leaf_effective_propensity", 0.0)
                ),
                "numerator": float(leaf_raw_numerator.detach().cpu()),
                "denominator": float(leaf_weight_denominator.detach().cpu()),
                "implemented_loss": float(out["leaf_loss"].detach().cpu())
                if isinstance(out.get("leaf_loss"), torch.Tensor)
                else 0.0,
            },
            "merge": {
                "population_size": float(out.get("merge_population_size", 0.0)),
                "sample_size": float(out.get("merge_sample_size", 0.0)),
                "effective_propensity": float(
                    out.get("merge_effective_propensity", 0.0)
                ),
                "numerator": float(merge_raw_numerator.detach().cpu()),
                "denominator": float(merge_weight_denominator.detach().cpu()),
                "implemented_loss": float(out["c3_loss"].detach().cpu())
                if isinstance(out.get("c3_loss"), torch.Tensor)
                else 0.0,
            },
            "c2": {
                "pair_weighting_mode": str(out.get("c2_pair_weighting_mode", "")),
                "same_pair_count": float(out.get("c2_same_pair_count", 0.0)),
                "different_pair_count": float(
                    out.get("c2_different_pair_count", 0.0)
                ),
                "pair_weight_ess": float(out.get("c2_pair_weight_ess", 0.0)),
                "pair_weight_max": float(out.get("c2_pair_weight_max", 0.0)),
            },
        }
        out["loss_components"] = {
            name: value
            for name, value in component_terms.items()
        }
        return out

    @staticmethod
    def _balanced_merge_children_map(
        n_leaves: int,
    ) -> Dict[int, Tuple[int, int]]:
        """Return {merge_local_idx: (left_child_global_idx, right_child_global_idx)}.

        Global indices: 0..n_leaves-1 are leaves, n_leaves.. are merge nodes
        in the order produced by ``_merge_states`` with balanced schedule.

        Example (4 leaves)::

            {0: (0, 1), 1: (2, 3), 2: (4, 5)}
              merge0 = merge(leaf0, leaf1)
              merge1 = merge(leaf2, leaf3)
              merge2 = merge(merge0, merge1)  # root
        """
        # Simulate the balanced merge loop to track children.
        # cur holds global indices of current-level nodes.
        cur: List[int] = list(range(n_leaves))
        merge_idx = 0  # local merge counter
        children: Dict[int, Tuple[int, int]] = {}
        while len(cur) > 1:
            nxt: List[int] = []
            i = 0
            while i < len(cur):
                if i + 1 >= len(cur):
                    nxt.append(cur[i])
                    i += 1
                    continue
                global_idx = n_leaves + merge_idx
                children[merge_idx] = (cur[i], cur[i + 1])
                nxt.append(global_idx)
                merge_idx += 1
                i += 2
            cur = nxt
        return children

    @classmethod
    def _balanced_tree_layout(
        cls,
        n_leaves: int,
    ) -> Dict[str, Any]:
        """Return node ids, depths, and child links for the realized balanced tree."""

        if int(n_leaves) <= 0:
            raise ValueError("n_leaves must be positive")

        children_map = cls._balanced_merge_children_map(int(n_leaves))
        if int(n_leaves) == 1:
            return {
                "children_map": {},
                "root_global_idx": 0,
                "depth_by_global_idx": {0: 0},
                "node_id_by_global_idx": {0: "root"},
            }

        root_global_idx = int(n_leaves) + len(children_map) - 1
        depth_by_global_idx: Dict[int, int] = {int(root_global_idx): 0}
        stack: List[int] = [int(root_global_idx)]
        while stack:
            global_idx = int(stack.pop())
            if global_idx < int(n_leaves):
                continue
            merge_local_idx = global_idx - int(n_leaves)
            if merge_local_idx not in children_map:
                continue
            left_child, right_child = children_map[merge_local_idx]
            child_depth = int(depth_by_global_idx[global_idx]) + 1
            depth_by_global_idx[int(left_child)] = int(child_depth)
            depth_by_global_idx[int(right_child)] = int(child_depth)
            stack.append(int(left_child))
            stack.append(int(right_child))

        node_id_by_global_idx: Dict[int, str] = {}
        total_nodes = int(n_leaves) + len(children_map)
        for global_idx in range(total_nodes):
            if global_idx == int(root_global_idx):
                node_id_by_global_idx[global_idx] = "root"
            elif global_idx < int(n_leaves):
                node_id_by_global_idx[global_idx] = f"leaf_{global_idx}"
            else:
                node_id_by_global_idx[global_idx] = f"merge_{global_idx - int(n_leaves)}"

        return {
            "children_map": children_map,
            "root_global_idx": int(root_global_idx),
            "depth_by_global_idx": depth_by_global_idx,
            "node_id_by_global_idx": node_id_by_global_idx,
        }

    def forward_doc_unified(
        self,
        leaf_token_ids: Sequence[Sequence[int]],
        leaf_counts: Sequence[float],
        merge_counts_balanced: Sequence[float],
        root_count: float,
        *,
        doc_id: str = "doc",
        schedule: ScheduleName,
        device: torch.device,
        sampled_leaf_indices: Optional[set] = None,
        sampled_internal_indices: Optional[set] = None,
        leaf_propensity: float = 1.0,
        internal_propensity: float = 1.0,
        proxy_leaf_counts: Optional[Sequence[float]] = None,
        proxy_merge_counts_balanced: Optional[Sequence[float]] = None,
        use_residual_decomposition: bool = True,
        collect_full_trace: bool = False,
    ) -> Dict[str, Any]:
        # collect_full_trace: when True, build per-node FullTreeNodeRecord
        # objects, the document_record, and the StateNode trace tree. Each of
        # these requires a GPU->CPU sync, so the per-node loop becomes the
        # bottleneck on long merge chains (e.g. recoverable_v5_t2048 leaf=16
        # is 255 syncs/doc). Default is False so that training/eval/anything
        # that only consumes the GPU tensors (`document_pred_norm`,
        # `all_node_preds`, etc.) stays CPU-launch-bound only by its own
        # logic. Telemetry consumers (phi-feature collection, eval reports,
        # diagnostic dumps) must pass collect_full_trace=True explicitly.
        """Unified IPW forward pass: all nodes in one pool with propensity weights.

        When *use_residual_decomposition* is True, merge-node loss terms use
        residuals: ``pred = g(merge) - g(left) - g(right)`` paired with
        ``target = (oracle_merge - oracle_left - oracle_right) / scale``.
        Leaf nodes always use direct loss.

        Returns the full realized node table, sampled-node tensors for the
        node-level Hajek objective, and the always-observed document-level
        supervision target at the top.
        """
        if len(leaf_token_ids) == 0:
            raise ValueError("leaf_token_ids must be non-empty")

        # Always encode all leaves and build full tree (needed for root).
        state_batch = self.encode_leaf_tokens_batch(leaf_token_ids, device=device)
        states = [state_batch[idx] for idx in range(int(state_batch.shape[0]))]
        root_state, merge_states = self._merge_states(
            states, schedule=schedule, collect_merge_states=True,
        )

        target_scale = float(self.target_scale)
        n_leaves = len(states)
        n_merges = len(merge_states)
        if int(n_merges) != int(len(merge_counts_balanced)):
            raise ValueError(
                "merge_counts_balanced must include one target per realized merge "
                f"(expected {n_merges}, got {len(merge_counts_balanced)})"
            )

        # Oracle counts in global order: leaves, then merges (root included last).
        all_oracle = [float(c) for c in leaf_counts] + [float(c) for c in merge_counts_balanced]
        effective_proxy_leaf_counts = (
            tuple(float(c) for c in proxy_leaf_counts)
            if proxy_leaf_counts is not None
            else tuple(float(c) for c in leaf_counts)
        )
        effective_proxy_merge_counts = (
            tuple(float(c) for c in proxy_merge_counts_balanced)
            if proxy_merge_counts_balanced is not None
            else tuple(float(c) for c in merge_counts_balanced)
        )
        if len(effective_proxy_leaf_counts) != int(n_leaves):
            raise ValueError(
                "proxy_leaf_counts must include one proxy target per leaf "
                f"(expected {n_leaves}, got {len(effective_proxy_leaf_counts)})"
            )
        if len(effective_proxy_merge_counts) != int(n_merges):
            raise ValueError(
                "proxy_merge_counts_balanced must include one proxy target per realized merge "
                f"(expected {n_merges}, got {len(effective_proxy_merge_counts)})"
            )
        all_proxy = list(effective_proxy_leaf_counts) + list(effective_proxy_merge_counts)

        # Compute g(state) for ALL nodes (needed for residuals even if unsampled).
        # Batched: stack all states into one tensor and call predict_norm once.
        # Avoids 255 sequential nn.Linear calls per doc at leaf=16.
        _all_states_stacked = torch.stack(list(states) + list(merge_states), dim=0)
        _all_preds_stacked = self.predict_norm_from_state(_all_states_stacked)
        if _all_preds_stacked.ndim == 0:
            _all_preds_stacked = _all_preds_stacked.unsqueeze(0)
        all_preds_raw: List[torch.Tensor] = [
            _all_preds_stacked[i] for i in range(int(_all_preds_stacked.shape[0]))
        ]

        layout = self._balanced_tree_layout(n_leaves)
        children_map = (
            dict(layout["children_map"]) if bool(use_residual_decomposition) else {}
        )
        trace_children_map = dict(layout["children_map"])
        root_global_idx = int(layout["root_global_idx"])
        depth_by_global_idx = dict(layout["depth_by_global_idx"])
        node_id_by_global_idx = dict(layout["node_id_by_global_idx"])

        node_preds: List[torch.Tensor] = []
        node_targets: List[float] = []
        node_weights: List[float] = []
        all_node_preds: List[torch.Tensor] = []
        all_node_proxy_targets: List[float] = []
        all_node_oracle_targets: List[float] = []
        all_node_observed: List[float] = []
        all_node_propensities: List[float] = []
        all_node_depths: List[float] = []
        node_records: List[FullTreeNodeRecord] = []
        # Telemetry deferral buffers (only populated when collect_full_trace=True).
        # We collect the GPU tensors during the per-node loop and do a single
        # batched .cpu() after the loop to avoid 255 sync points per doc.
        _record_metadata_buf: List[Dict[str, Any]] = []
        _record_objective_preds: List[torch.Tensor] = []

        for global_idx, raw_pred in enumerate(all_preds_raw):
            is_leaf = global_idx < n_leaves
            is_root = bool(global_idx == root_global_idx)
            if is_leaf:
                node_type = NodeType.LEAF
                sampled = (
                    sampled_leaf_indices is None or global_idx in sampled_leaf_indices
                )
                propensity = float(leaf_propensity)
            else:
                node_type = NodeType.MERGE
                merge_local_idx = global_idx - n_leaves
                sampled = (
                    sampled_internal_indices is None
                    or merge_local_idx in sampled_internal_indices
                )
                propensity = float(internal_propensity)

            direct_target = float(all_oracle[global_idx]) / float(target_scale)
            direct_proxy_target = float(all_proxy[global_idx]) / float(target_scale)
            objective_pred = raw_pred
            objective_target = float(direct_target)
            proxy_objective_target = float(direct_proxy_target)
            target_mode = "direct"
            metadata: Dict[str, Any] = {
                "depth": int(depth_by_global_idx.get(global_idx, 0)),
                "is_root": bool(is_root),
                "logged_propensity": float(propensity),
                "target_mode": str(target_mode),
                "direct_node_target": float(direct_target),
                "direct_proxy_node_target": float(direct_proxy_target),
            }

            if not is_leaf:
                merge_local_idx = global_idx - n_leaves
                if merge_local_idx in children_map:
                    left_gidx, right_gidx = children_map[merge_local_idx]
                    objective_pred = (
                        all_preds_raw[global_idx]
                        - all_preds_raw[left_gidx].detach()
                        - all_preds_raw[right_gidx].detach()
                    )
                    objective_target = (
                        float(all_oracle[global_idx])
                        - float(all_oracle[left_gidx])
                        - float(all_oracle[right_gidx])
                    ) / float(target_scale)
                    proxy_objective_target = (
                        float(all_proxy[global_idx])
                        - float(all_proxy[left_gidx])
                        - float(all_proxy[right_gidx])
                    ) / float(target_scale)
                    target_mode = "residual"
                    metadata["left_child_id"] = str(node_id_by_global_idx[left_gidx])
                    metadata["right_child_id"] = str(node_id_by_global_idx[right_gidx])
                    metadata["target_mode"] = str(target_mode)
                    metadata["proxy_target_mode"] = str(target_mode)

            all_node_preds.append(objective_pred)
            all_node_proxy_targets.append(float(proxy_objective_target))
            all_node_oracle_targets.append(float(objective_target))
            all_node_observed.append(1.0 if bool(sampled) and float(propensity) > 0.0 else 0.0)
            all_node_propensities.append(float(propensity))
            all_node_depths.append(float(depth_by_global_idx.get(global_idx, 0)))

            if collect_full_trace:
                # Stash the metadata needed to build a FullTreeNodeRecord after
                # the loop. The actual GPU->CPU sync for raw_pred / objective_pred
                # is deferred to a single batched .cpu() call below; per-node
                # syncs were the leaf=16 hot path bottleneck (255 syncs/doc).
                _record_metadata_buf.append({
                    "global_idx": int(global_idx),
                    "node_id": str(node_id_by_global_idx[global_idx]),
                    "depth": int(depth_by_global_idx.get(global_idx, 0)),
                    "node_type": node_type,
                    "is_root": bool(is_root),
                    "objective_target": float(objective_target),
                    "proxy_objective_target": float(proxy_objective_target),
                    "sampled": bool(sampled),
                    "propensity": float(propensity),
                    "metadata": metadata,
                    "_raw_pred_idx": int(global_idx),
                    "_objective_pred_pos": len(_record_objective_preds),
                })
                _record_objective_preds.append(objective_pred)

            if bool(sampled) and float(propensity) > 0.0:
                node_preds.append(objective_pred)
                node_targets.append(float(objective_target))
                node_weights.append(1.0 / max(float(propensity), 1e-12))

        # Batched telemetry materialization: one .cpu() call per tensor list
        # instead of one per node. Skipped entirely when collect_full_trace=False.
        if collect_full_trace and _record_metadata_buf:
            _objective_pred_cpu = (
                torch.stack(_record_objective_preds, dim=0).detach().cpu().reshape(-1).tolist()
            )
            _raw_pred_cpu = _all_preds_stacked.detach().cpu().reshape(-1).tolist()
            for _meta in _record_metadata_buf:
                _obj_pred_val = float(_objective_pred_cpu[int(_meta["_objective_pred_pos"])])
                _raw_pred_val = float(_raw_pred_cpu[int(_meta["_raw_pred_idx"])])
                _proxy_loss = (_obj_pred_val - float(_meta["proxy_objective_target"])) ** 2
                _oracle_loss = (_obj_pred_val - float(_meta["objective_target"])) ** 2
                node_records.append(
                    FullTreeNodeRecord(
                        doc_id=str(doc_id),
                        node_id=str(_meta["node_id"]),
                        depth=int(_meta["depth"]),
                        node_type=_meta["node_type"],
                        is_root=bool(_meta["is_root"]),
                        prediction=_raw_pred_val,
                        target=float(_meta["objective_target"]),
                        sampled=bool(_meta["sampled"]),
                        propensity=float(_meta["propensity"]),
                        objective_prediction=_obj_pred_val,
                        proxy_loss=float(_proxy_loss),
                        oracle_loss=float(_oracle_loss),
                        metadata=dict(_meta["metadata"]),
                    )
                )

        root_pred_norm = all_preds_raw[root_global_idx]
        if node_preds:
            pred_stack = torch.stack(node_preds, dim=0)
            target_t = torch.tensor(node_targets, device=device, dtype=pred_stack.dtype)
            weight_t = torch.tensor(node_weights, device=device, dtype=pred_stack.dtype)
        else:
            pred_stack = torch.empty((0,), device=device, dtype=root_pred_norm.dtype)
            target_t = torch.empty((0,), device=device, dtype=root_pred_norm.dtype)
            weight_t = torch.empty((0,), device=device, dtype=root_pred_norm.dtype)

        if all_node_preds:
            all_pred_stack = torch.stack(all_node_preds, dim=0)
            all_proxy_target_t = torch.tensor(
                all_node_proxy_targets, device=device, dtype=all_pred_stack.dtype,
            )
            all_oracle_target_t = torch.tensor(
                all_node_oracle_targets, device=device, dtype=all_pred_stack.dtype,
            )
            all_observed_t = torch.tensor(
                all_node_observed, device=device, dtype=all_pred_stack.dtype,
            )
            all_propensity_t = torch.tensor(
                all_node_propensities, device=device, dtype=all_pred_stack.dtype,
            )
            all_depth_t = torch.tensor(
                all_node_depths, device=device, dtype=all_pred_stack.dtype,
            )
        else:
            all_pred_stack = torch.empty((0,), device=device, dtype=root_pred_norm.dtype)
            all_proxy_target_t = torch.empty((0,), device=device, dtype=root_pred_norm.dtype)
            all_oracle_target_t = torch.empty((0,), device=device, dtype=root_pred_norm.dtype)
            all_observed_t = torch.empty((0,), device=device, dtype=root_pred_norm.dtype)
            all_propensity_t = torch.empty((0,), device=device, dtype=root_pred_norm.dtype)
            all_depth_t = torch.empty((0,), device=device, dtype=root_pred_norm.dtype)

        document_target_norm = float(root_count) / float(target_scale)
        document_record: Optional[DocumentLevelPredictionRecord] = None
        state_tree: Optional[StateTree] = None
        if collect_full_trace:
            # One batched .cpu() for the document scalar (was 3 separate syncs).
            _root_pred_norm_val = float(root_pred_norm.detach().cpu())
            document_record = DocumentLevelPredictionRecord(
                doc_id=str(doc_id),
                prediction=_root_pred_norm_val,
                target=float(document_target_norm),
                metadata={
                    "raw_prediction": float(_root_pred_norm_val * float(target_scale)),
                    "raw_target": float(root_count),
                    "final_summary_prediction": _root_pred_norm_val,
                },
            )
            state_by_global_idx = list(states) + list(merge_states)
            max_depth = max((int(v) for v in depth_by_global_idx.values()), default=0)
            record_by_node_id = {str(record.node_id): record for record in node_records}
            # One batched state .cpu() for all global indices, instead of
            # 255 sequential syncs.
            _all_states_cpu = (
                torch.stack(state_by_global_idx, dim=0).detach().cpu()
                if state_by_global_idx else None
            )
            trace_nodes: Dict[int, StateNode[Any, Any]] = {}
            for global_idx in range(len(state_by_global_idx)):
                node_id = str(node_id_by_global_idx.get(global_idx, f"node_{global_idx}"))
                record = record_by_node_id.get(node_id)
                metadata = dict(record.metadata if record is not None else {})
                is_root = bool(global_idx == root_global_idx)
                is_leaf = bool(global_idx < n_leaves)
                if record is not None:
                    metadata.update(
                        {
                            "doc_id": str(doc_id),
                            "source_node_id": node_id,
                            "node_type": "root" if is_root else str(record.node_type.value),
                            "is_root": bool(is_root),
                            "is_leaf": bool(is_leaf),
                            "depth": int(record.depth),
                            "prediction": float(record.loss_prediction),
                            "readout_prediction": float(record.loss_prediction),
                            "target": float(record.target),
                            "proxy_loss": record.proxy_loss,
                            "oracle_loss": record.oracle_loss,
                            "observed": bool(record.sampled),
                            "sampled": bool(record.sampled),
                            "propensity": float(record.propensity),
                            "node_weight": 1.0,
                            "law_channel": "root" if is_root else ("leaf" if is_leaf else "merge"),
                            "state_kind": "markov_fno_state",
                        }
                    )
                _state_cpu_slice = (
                    _all_states_cpu[global_idx]
                    if _all_states_cpu is not None
                    else state_by_global_idx[global_idx].detach().cpu()
                )
                trace_nodes[global_idx] = StateNode[Any, Any](
                    id=node_id,
                    level=max(0, int(max_depth - int(depth_by_global_idx.get(global_idx, 0)))),
                    span={
                        "global_index": int(global_idx),
                        "token_ids": list(leaf_token_ids[global_idx]) if is_leaf else [],
                    },
                    state=_state_cpu_slice,
                    rendered=str(metadata.get("target_mode", "")),
                    metadata=metadata,
                )
            for merge_local_idx, (left_gidx, right_gidx) in trace_children_map.items():
                parent_gidx = int(n_leaves) + int(merge_local_idx)
                parent = trace_nodes.get(parent_gidx)
                if parent is None:
                    continue
                parent.left_child = trace_nodes.get(int(left_gidx))
                parent.right_child = trace_nodes.get(int(right_gidx))
                if parent.left_child is not None:
                    parent.left_child.parent = parent
                if parent.right_child is not None:
                    parent.right_child.parent = parent
            state_tree = StateTree(
                root=trace_nodes[int(root_global_idx)],
                metadata={
                    "doc_id": str(doc_id),
                    "method_family": "markov_fno",
                    "state_kind": "markov_fno_state",
                    "trace_schema": "state_tree_full_trace_v1",
                    "schedule": str(schedule),
                },
            )

        return {
            "node_records": tuple(node_records),
            "document_record": document_record,
            "state_tree": state_tree,
            "document_pred_norm": root_pred_norm,
            "document_target_norm": torch.tensor(
                float(document_target_norm), device=device, dtype=root_pred_norm.dtype
            ),
            "node_preds": pred_stack,
            "node_targets": target_t,
            "node_weights": weight_t,
            "all_node_preds": all_pred_stack,
            "all_node_proxy_targets": all_proxy_target_t,
            "all_node_oracle_targets": all_oracle_target_t,
            "all_node_observed": all_observed_t,
            "all_node_propensities": all_propensity_t,
            "all_node_depths": all_depth_t,
            "root_pred_count": self.predict_count_from_state(root_state),
            "n_nodes": len(node_records),
            "n_sampled_nodes": len(node_preds),
        }


def _fno_summary_replay_tensors(
    model: FNOCountSketch,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return base/replay normalized scores and encoded states for one C2 step."""

    base_summary = model.decode_summary(state)
    base_state = model.encode_summary(base_summary)
    replay_state = model.encode_summary(model.decode_summary(base_state))
    base_norm = model.predict_norm_from_state(base_state)
    replay_norm = model.predict_norm_from_state(replay_state)
    return base_norm, replay_norm, base_state, replay_state


def _softmax_1d(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return torch.softmax(logits, dim=0)
    return torch.softmax(logits, dim=-1)


def _soft_endpoint_consistency_loss(
    pred_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> torch.Tensor:
    target_probs = _softmax_1d(target_logits).detach()
    pred_log_probs = torch.log_softmax(
        pred_logits,
        dim=0 if pred_logits.ndim == 1 else -1,
    )
    return torch.sum(target_probs * (torch.log(target_probs.clamp(min=1e-12)) - pred_log_probs))


SUMMARY_SPEC_ENDPOINT_AUX_SCALE = 0.1
SUMMARY_SPEC_PHI_CONTRASTIVE_TEMPERATURE = 0.5
SUMMARY_SPEC_PHI_DIFFERENT_MARGIN = 0.5


def _phi_alignment_loss(
    pred_phi: torch.Tensor,
    target_phi: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    normalized_mode = str(mode or "cosine_mse").strip().lower() or "cosine_mse"
    mse = F.mse_loss(pred_phi, target_phi)
    if normalized_mode == "cosine_mse":
        cosine = 1.0 - F.cosine_similarity(pred_phi, target_phi, dim=-1).mean()
        return mse + cosine
    raise ValueError(f"unsupported phi alignment loss mode: {mode!r}")


def _phi_alignment_loss_per_item(
    pred_phi: torch.Tensor,
    target_phi: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    normalized_mode = str(mode or "cosine_mse").strip().lower() or "cosine_mse"
    mse = F.mse_loss(pred_phi, target_phi, reduction="none").mean(dim=-1)
    if normalized_mode == "cosine_mse":
        cosine = 1.0 - F.cosine_similarity(pred_phi, target_phi, dim=-1)
        return mse + cosine
    raise ValueError(f"unsupported phi alignment loss mode: {mode!r}")


def _supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = SUMMARY_SPEC_PHI_CONTRASTIVE_TEMPERATURE,
) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be rank-2")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must be rank-1 aligned with embeddings")
    if int(embeddings.shape[0]) <= 1:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    z = F.normalize(embeddings, dim=-1)
    sim = torch.matmul(z, z.transpose(0, 1)) / float(max(temperature, 1e-6))
    diag_mask = torch.eye(
        int(sim.shape[0]), device=sim.device, dtype=torch.bool
    )
    sim = sim.masked_fill(diag_mask, float("-inf"))
    label_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~diag_mask)
    positive_counts = label_mask.sum(dim=1)
    valid = positive_counts > 0
    if not bool(valid.any()):
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positive_log_prob = torch.where(
        label_mask,
        log_prob,
        torch.zeros_like(log_prob),
    ).sum(dim=1)
    per_anchor = -positive_log_prob[valid] / positive_counts[valid].to(
        dtype=log_prob.dtype
    )
    return per_anchor.mean()


def _pairwise_theorem_feature_contrastive_loss(
    embeddings: torch.Tensor,
    *,
    same_pairs: Sequence[tuple[int, int]],
    different_pairs: Sequence[tuple[int, int]],
    same_pair_weights: Sequence[float] | None = None,
    different_pair_weights: Sequence[float] | None = None,
    different_margin: float = SUMMARY_SPEC_PHI_DIFFERENT_MARGIN,
) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be rank-2")
    if not same_pairs and not different_pairs:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    normalized = F.normalize(embeddings, dim=-1)
    same_loss = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    diff_loss = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    active_terms = 0
    if same_pairs:
        same_index = torch.as_tensor(
            list(same_pairs),
            device=embeddings.device,
            dtype=torch.long,
        )
        same_sim = torch.sum(
            normalized.index_select(0, same_index[:, 0])
            * normalized.index_select(0, same_index[:, 1]),
            dim=-1,
        )
        same_terms = 1.0 - same_sim
        if same_pair_weights is None:
            same_loss = same_terms.mean()
            active_terms += 1
        else:
            same_weights = torch.as_tensor(
                list(same_pair_weights),
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
            same_weight_sum = same_weights.sum()
            if float(same_weight_sum.detach().item()) > 0.0:
                same_loss = (same_terms * same_weights).sum() / same_weight_sum.clamp_min(1e-12)
                active_terms += 1
    if different_pairs:
        diff_index = torch.as_tensor(
            list(different_pairs),
            device=embeddings.device,
            dtype=torch.long,
        )
        diff_sim = torch.sum(
            normalized.index_select(0, diff_index[:, 0])
            * normalized.index_select(0, diff_index[:, 1]),
            dim=-1,
        )
        diff_terms = F.relu(diff_sim - float(different_margin))
        if different_pair_weights is None:
            diff_loss = diff_terms.mean()
            active_terms += 1
        else:
            diff_weights = torch.as_tensor(
                list(different_pair_weights),
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
            diff_weight_sum = diff_weights.sum()
            if float(diff_weight_sum.detach().item()) > 0.0:
                diff_loss = (diff_terms * diff_weights).sum() / diff_weight_sum.clamp_min(1e-12)
                active_terms += 1
    if active_terms <= 0:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    return (same_loss + diff_loss) / float(active_terms)


def _pairwise_theorem_feature_contrastive_loss_from_masks(
    embeddings: torch.Tensor,
    *,
    same_mask: torch.Tensor,
    different_mask: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
    different_margin: float = SUMMARY_SPEC_PHI_DIFFERENT_MARGIN,
) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be rank-2")
    if same_mask.ndim != 2 or different_mask.ndim != 2:
        raise ValueError("same_mask and different_mask must be rank-2")
    if same_mask.shape != different_mask.shape:
        raise ValueError("same_mask and different_mask must align")
    if int(embeddings.shape[0]) != int(same_mask.shape[0]):
        raise ValueError("mask size must match embedding batch size")
    normalized = F.normalize(embeddings, dim=-1)
    sim = torch.matmul(normalized, normalized.transpose(0, 1))
    upper = torch.triu(
        torch.ones_like(same_mask, dtype=torch.bool),
        diagonal=1,
    )
    same_mask = same_mask.to(dtype=torch.bool) & upper
    different_mask = different_mask.to(dtype=torch.bool) & upper
    if pair_weights is not None:
        if pair_weights.ndim != 2 or pair_weights.shape != same_mask.shape:
            raise ValueError("pair_weights must align with same_mask and different_mask")
        weight_matrix = pair_weights.to(device=embeddings.device, dtype=embeddings.dtype)
    else:
        weight_matrix = None
    same_loss = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    diff_loss = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    active_terms = 0
    if bool(same_mask.any()):
        same_terms = 1.0 - sim[same_mask]
        if weight_matrix is None:
            same_loss = same_terms.mean()
            active_terms += 1
        else:
            same_weights = weight_matrix[same_mask]
            same_weight_sum = same_weights.sum()
            if float(same_weight_sum.detach().item()) > 0.0:
                same_loss = (same_terms * same_weights).sum() / same_weight_sum.clamp_min(1e-12)
                active_terms += 1
    if bool(different_mask.any()):
        diff_terms = F.relu(sim[different_mask] - float(different_margin))
        if weight_matrix is None:
            diff_loss = diff_terms.mean()
            active_terms += 1
        else:
            diff_weights = weight_matrix[different_mask]
            diff_weight_sum = diff_weights.sum()
            if float(diff_weight_sum.detach().item()) > 0.0:
                diff_loss = (diff_terms * diff_weights).sum() / diff_weight_sum.clamp_min(1e-12)
                active_terms += 1
    if active_terms <= 0:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    return (same_loss + diff_loss) / float(active_terms)


def _straight_through_one_hot_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    indices = torch.argmax(probs, dim=-1)
    hard = F.one_hot(indices, num_classes=int(logits.shape[-1])).to(dtype=logits.dtype)
    return hard + probs - probs.detach()


def _batched_pairwise_theorem_feature_contrastive_loss_from_masks(
    embeddings: torch.Tensor,
    *,
    same_mask: torch.Tensor,
    different_mask: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
    different_margin: float = SUMMARY_SPEC_PHI_DIFFERENT_MARGIN,
    return_diagnostics: bool = False,
) -> torch.Tensor | Dict[str, torch.Tensor]:
    if embeddings.ndim != 3:
        raise ValueError("embeddings must be rank-3")
    if same_mask.ndim != 3 or different_mask.ndim != 3:
        raise ValueError("same_mask and different_mask must be rank-3")
    if same_mask.shape != different_mask.shape:
        raise ValueError("same_mask and different_mask must align")
    if int(embeddings.shape[0]) != int(same_mask.shape[0]) or int(embeddings.shape[1]) != int(
        same_mask.shape[1]
    ):
        raise ValueError("mask size must match embedding batch size")
    normalized = F.normalize(embeddings, dim=-1)
    sim = torch.matmul(normalized, normalized.transpose(1, 2))
    upper = torch.triu(
        torch.ones(
            (int(sim.shape[1]), int(sim.shape[2])),
            device=sim.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    ).unsqueeze(0)
    same_mask = same_mask.to(dtype=torch.bool) & upper
    different_mask = different_mask.to(dtype=torch.bool) & upper
    if pair_weights is not None:
        if pair_weights.ndim != 3 or pair_weights.shape != same_mask.shape:
            raise ValueError("pair_weights must align with same_mask and different_mask")
        weight_matrix = pair_weights.to(device=embeddings.device, dtype=embeddings.dtype)
    else:
        weight_matrix = None
    same_count = same_mask.sum(dim=(1, 2))
    diff_count = different_mask.sum(dim=(1, 2))
    if weight_matrix is None:
        same_weights = same_mask.to(dtype=sim.dtype)
        diff_weights = different_mask.to(dtype=sim.dtype)
    else:
        same_weights = torch.where(same_mask, weight_matrix, torch.zeros_like(weight_matrix))
        diff_weights = torch.where(different_mask, weight_matrix, torch.zeros_like(weight_matrix))
    same_weight_sum = same_weights.sum(dim=(1, 2))
    diff_weight_sum = diff_weights.sum(dim=(1, 2))
    same_sum = ((1.0 - sim) * same_weights).sum(dim=(1, 2))
    diff_sum = (
        F.relu(sim - float(different_margin)) * diff_weights
    ).sum(dim=(1, 2))
    same_loss = same_sum / same_weight_sum.clamp_min(1e-12)
    diff_loss = diff_sum / diff_weight_sum.clamp_min(1e-12)
    active_terms = (
        (same_weight_sum > 0).to(dtype=sim.dtype)
        + (diff_weight_sum > 0).to(dtype=sim.dtype)
    )
    loss = torch.where(
        active_terms > 0,
        (same_loss * (same_weight_sum > 0).to(dtype=sim.dtype)
         + diff_loss * (diff_weight_sum > 0).to(dtype=sim.dtype))
        / active_terms.clamp_min(1.0),
        torch.zeros_like(same_loss),
    )
    if not bool(return_diagnostics):
        return loss
    weight_sum = same_weights.sum(dim=(1, 2)) + diff_weights.sum(dim=(1, 2))
    weight_sq_sum = (same_weights * same_weights).sum(dim=(1, 2)) + (
        diff_weights * diff_weights
    ).sum(dim=(1, 2))
    ess = torch.where(
        weight_sq_sum > 0,
        (weight_sum * weight_sum) / weight_sq_sum.clamp_min(1e-12),
        torch.zeros_like(weight_sum),
    )
    max_weight = torch.maximum(
        same_weights.amax(dim=(1, 2)),
        diff_weights.amax(dim=(1, 2)),
    )
    return {
        "loss": loss,
        "same_pair_count": same_count.to(dtype=torch.float32),
        "different_pair_count": diff_count.to(dtype=torch.float32),
        "pair_weight_ess": ess.to(dtype=torch.float32),
        "pair_weight_max": max_weight.to(dtype=torch.float32),
    }


def _pairwise_weight_diagnostics_from_masks(
    *,
    same_mask: torch.Tensor,
    different_mask: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    squeeze = False
    if same_mask.ndim == 2:
        same_mask = same_mask.unsqueeze(0)
        different_mask = different_mask.unsqueeze(0)
        if pair_weights is not None:
            pair_weights = pair_weights.unsqueeze(0)
        squeeze = True
    if same_mask.ndim != 3 or different_mask.ndim != 3:
        raise ValueError("same_mask and different_mask must be rank-2 or rank-3")
    if same_mask.shape != different_mask.shape:
        raise ValueError("same_mask and different_mask must align")
    if pair_weights is not None and pair_weights.shape != same_mask.shape:
        raise ValueError("pair_weights must align with masks")
    upper = torch.triu(
        torch.ones(
            (int(same_mask.shape[-2]), int(same_mask.shape[-1])),
            device=same_mask.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    ).unsqueeze(0)
    same_bool = same_mask.to(dtype=torch.bool) & upper
    diff_bool = different_mask.to(dtype=torch.bool) & upper
    if pair_weights is None:
        weights = torch.where(
            same_bool | diff_bool,
            torch.ones(
                same_mask.shape,
                device=same_mask.device,
                dtype=torch.float32,
            ),
            torch.zeros(
                same_mask.shape,
                device=same_mask.device,
                dtype=torch.float32,
            ),
        )
    else:
        weights = torch.where(
            same_bool | diff_bool,
            pair_weights.to(dtype=torch.float32),
            torch.zeros_like(pair_weights, dtype=torch.float32),
        )
    weight_sum = weights.sum(dim=(1, 2))
    weight_sq_sum = (weights * weights).sum(dim=(1, 2))
    ess = torch.where(
        weight_sq_sum > 0,
        (weight_sum * weight_sum) / weight_sq_sum.clamp_min(1e-12),
        torch.zeros_like(weight_sum),
    )
    max_weight = weights.amax(dim=(1, 2))
    out = {
        "same_pair_count": same_bool.sum(dim=(1, 2)).to(dtype=torch.float32),
        "different_pair_count": diff_bool.sum(dim=(1, 2)).to(dtype=torch.float32),
        "pair_weight_ess": ess.to(dtype=torch.float32),
        "pair_weight_max": max_weight.to(dtype=torch.float32),
    }
    if squeeze:
        return {key: value.squeeze(0) for key, value in out.items()}
    return out


def _supports_fast_markov_pair_masks(model: FNOCountSketch) -> bool:
    if model.oracle_metric is not None:
        return False
    adapter_name = str(getattr(model, "theorem_feature_adapter_name", "") or "").strip().lower()
    return adapter_name in {
        DEFAULT_THEOREM_FEATURE_ADAPTER,
        SCOREFIBER_MARKOV_ENDPOINTS_ADAPTER,
        COARSENED_THEOREM_FEATURE_ADAPTER,
    }


def _fast_markov_pair_masks_from_tensors(
    model: FNOCountSketch,
    *,
    count_keys: torch.Tensor,
    first_targets: torch.Tensor,
    last_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    adapter_name = str(getattr(model, "theorem_feature_adapter_name", "") or "").strip().lower()
    if count_keys.shape != first_targets.shape or count_keys.shape != last_targets.shape:
        raise ValueError("count_keys, first_targets, and last_targets must align")
    if count_keys.ndim not in {1, 2}:
        raise ValueError("fast pair masks only support rank-1 or rank-2 tensors")
    count_left = count_keys.unsqueeze(-1)
    count_right = count_keys.unsqueeze(-2)
    first_left = first_targets.unsqueeze(-1)
    first_right = first_targets.unsqueeze(-2)
    last_left = last_targets.unsqueeze(-1)
    last_right = last_targets.unsqueeze(-2)
    if adapter_name == DEFAULT_THEOREM_FEATURE_ADAPTER:
        same_mask = (
            count_left.eq(count_right)
            & first_left.eq(first_right)
            & last_left.eq(last_right)
        )
        different_mask = ~same_mask
        return same_mask, different_mask
    if adapter_name == SCOREFIBER_MARKOV_ENDPOINTS_ADAPTER:
        same_mask = first_left.eq(first_right) & last_left.eq(last_right)
        different_mask = ~same_mask
        return same_mask, different_mask
    if adapter_name == COARSENED_THEOREM_FEATURE_ADAPTER:
        adapter = getattr(model, "theorem_feature_adapter", None)
        bin_width = max(1, int(getattr(adapter, "count_bin_width", 3)))
        ignore_endpoints = bool(getattr(adapter, "ignore_endpoints", False))
        binned = torch.div(count_keys, int(bin_width), rounding_mode="floor")
        same_mask = binned.unsqueeze(-1).eq(binned.unsqueeze(-2))
        if ignore_endpoints:
            different_mask = (binned.unsqueeze(-1) - binned.unsqueeze(-2)).abs() >= 2
        else:
            same_mask = same_mask & first_left.eq(first_right) & last_left.eq(last_right)
            different_mask = ~same_mask
        return same_mask, different_mask
    raise ValueError(
        f"fast markov pair masks do not support adapter {adapter_name!r}"
    )


def _materialize_fast_markov_labels(
    model: FNOCountSketch,
    *,
    count_values: torch.Tensor,
    first_targets: torch.Tensor,
    last_targets: torch.Tensor,
) -> list[Any]:
    adapter_name = str(getattr(model, "theorem_feature_adapter_name", "") or "").strip().lower()
    count_list = [float(value) for value in count_values.detach().cpu().tolist()]
    first_list = [int(value) for value in first_targets.detach().cpu().tolist()]
    last_list = [int(value) for value in last_targets.detach().cpu().tolist()]
    if adapter_name == SCOREFIBER_MARKOV_ENDPOINTS_ADAPTER:
        return [
            ScoreFiberTheoremFeatureLabel(
                score=float(count),
                fiber_key=(int(first), int(last)),
            )
            for count, first, last in zip(count_list, first_list, last_list)
        ]
    return [
        MarkovTheoremFeatureLabel(
            count=float(count),
            first=int(first),
            last=int(last),
        )
        for count, first, last in zip(count_list, first_list, last_list)
    ]


def _summary_spec_count_supervision_terms(
    model: FNOCountSketch,
    state: torch.Tensor,
    *,
    truth_count: float,
) -> Dict[str, torch.Tensor]:
    if model.summary_count_classifier is not None:
        count_logits = model.predict_count_logits_from_state(state)
        target_index = model.count_target_index(
            float(truth_count),
            device=state.device,
        )
        primary_loss = F.cross_entropy(
            count_logits.unsqueeze(0),
            target_index.unsqueeze(0),
        )
        aux_loss = torch.zeros((), device=state.device, dtype=primary_loss.dtype)
        total_loss = primary_loss
    elif model.uses_hybrid_ordinal_count_head():
        ordinal_logits = model.predict_count_ordinal_logits_from_state(state)
        threshold_targets = model.count_threshold_targets(
            float(truth_count),
            device=state.device,
            dtype=ordinal_logits.dtype,
        )
        threshold_pos_weight = model.theorem_count_threshold_pos_weight.to(
            device=ordinal_logits.device,
            dtype=ordinal_logits.dtype,
        )
        primary_loss = F.binary_cross_entropy_with_logits(
            ordinal_logits,
            threshold_targets,
            pos_weight=threshold_pos_weight,
        )
        pred_count_aux = model.predict_count_scalar_aux_from_state(state)
        target_count = torch.tensor(
            float(truth_count),
            device=state.device,
            dtype=pred_count_aux.dtype,
        )
        aux_loss = F.mse_loss(pred_count_aux, target_count)
        total_loss = (
            float(model.theorem_count_ordinal_weight) * primary_loss
            + float(model.theorem_count_scalar_aux_weight) * aux_loss
        )
    else:
        pred_count = model.predict_count_from_state(state)
        target_count = torch.tensor(
            float(truth_count),
            device=state.device,
            dtype=pred_count.dtype,
        )
        primary_loss = F.mse_loss(pred_count, target_count)
        aux_loss = torch.zeros((), device=state.device, dtype=primary_loss.dtype)
        total_loss = primary_loss
    return {
        "primary_loss": primary_loss,
        "aux_loss": aux_loss,
        "total_loss": total_loss,
    }


def _normalize_tree_local_weighting_mode(value: str) -> str:
    mode = str(value or "fixed_k_hajek").strip().lower() or "fixed_k_hajek"
    if mode not in VALID_TREE_LOCAL_WEIGHTING_MODES:
        raise ValueError(
            "tree_local_weighting_mode must be one of "
            f"{VALID_TREE_LOCAL_WEIGHTING_MODES}; got {value!r}"
        )
    return mode


def _normalize_tree_supervision_source(value: str | None) -> str:
    source = str(value or "rate").strip().lower() or "rate"
    if source not in VALID_TREE_SUPERVISION_SOURCES:
        raise ValueError(
            "tree_supervision_source must be one of "
            f"{VALID_TREE_SUPERVISION_SOURCES}; got {value!r}"
        )
    return source


def _resolved_local_loss_kind(
    *,
    leaf_supervision_kind: str,
    internal_supervision_kind: str,
) -> str:
    active = {
        str(leaf_supervision_kind or "").strip().lower(),
        str(internal_supervision_kind or "").strip().lower(),
    }
    active.discard("")
    active.discard("none")
    if "bounded_full_sketch" in active:
        return "bounded_full_sketch"
    if "full_sketch" in active:
        return "legacy_empirical_full_sketch"
    if "count_only" in active:
        return "count_only"
    return "none"


def _effective_local_sampling_summary(
    population_size: int,
    sampled_indices: Sequence[int] | set[int] | None,
) -> tuple[int, int, float]:
    n = int(max(0, int(population_size)))
    if n <= 0:
        return 0, 0, 0.0
    if sampled_indices is None:
        return n, n, 1.0
    k = int(max(0, len(list(sampled_indices))))
    if k <= 0:
        return n, 0, 0.0
    return n, k, float(k) / float(n)


def _bounded_endpoint_surprise_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    squeeze_result = bool(logits.ndim == 1)
    if squeeze_result:
        logits = logits.unsqueeze(0)
    target_tensor = targets.to(device=logits.device, dtype=torch.long)
    if target_tensor.ndim == 0:
        target_tensor = target_tensor.unsqueeze(0)
    if int(target_tensor.shape[0]) != int(logits.shape[0]):
        raise ValueError("targets must align with logits batch dimension")
    probs = torch.softmax(logits, dim=-1)
    gathered = probs.gather(
        dim=-1,
        index=target_tensor.unsqueeze(-1),
    ).squeeze(-1)
    loss = 1.0 - gathered
    if squeeze_result:
        return loss.reshape(())
    return loss


def _summary_spec_bounded_supervision_terms(
    model: FNOCountSketch,
    state: torch.Tensor,
    *,
    truth_count: float,
    truth_first: int | None = None,
    truth_last: int | None = None,
    supervise_endpoints: bool = False,
) -> Dict[str, torch.Tensor]:
    if not model.use_markov_summary_spec:
        raise RuntimeError("bounded summary supervision requested without markov summary spec")
    pred_norm = model.predict_norm_from_state(state)
    target_norm = torch.tensor(
        float(truth_count) / float(model.target_scale),
        device=state.device,
        dtype=pred_norm.dtype,
    )
    count_loss = (pred_norm - target_norm) ** 2
    first_loss = torch.zeros((), device=state.device, dtype=pred_norm.dtype)
    last_loss = torch.zeros((), device=state.device, dtype=pred_norm.dtype)
    active_terms = 1.0
    if bool(supervise_endpoints):
        if truth_first is None or truth_last is None:
            raise ValueError("truth_first and truth_last are required for endpoint supervision")
        _h, first_logits, last_logits = model._split_state(state)
        first_loss = _bounded_endpoint_surprise_loss(
            first_logits,
            torch.tensor([int(truth_first)], device=state.device),
        ).reshape(())
        last_loss = _bounded_endpoint_surprise_loss(
            last_logits,
            torch.tensor([int(truth_last)], device=state.device),
        ).reshape(())
        active_terms += 2.0
    total_loss = (count_loss + first_loss + last_loss) / float(active_terms)
    return {
        "count_loss": count_loss,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "total_loss": total_loss,
    }


def _summary_spec_supervision_terms(
    model: FNOCountSketch,
    state: torch.Tensor,
    *,
    truth_count: float,
    truth_first: int | None = None,
    truth_last: int | None = None,
    supervise_count: bool = True,
    supervise_endpoints: bool = False,
) -> torch.Tensor:
    if not model.use_markov_summary_spec:
        raise RuntimeError("summary spec supervision requested without markov summary spec")
    count_loss = torch.zeros((), device=state.device, dtype=state.dtype)
    first_loss = torch.zeros((), device=state.device, dtype=state.dtype)
    last_loss = torch.zeros((), device=state.device, dtype=state.dtype)
    if bool(supervise_count):
        count_loss = _summary_spec_count_supervision_terms(
            model,
            state,
            truth_count=float(truth_count),
        )["total_loss"]
    if bool(supervise_endpoints):
        _h, first_logits, last_logits = model._split_state(state)
        if truth_first is None or truth_last is None:
            raise ValueError("truth_first and truth_last are required for endpoint supervision")
        ep_scale = float(model.endpoint_loss_scale)
        first_loss = ep_scale * F.cross_entropy(
            first_logits.unsqueeze(0),
            torch.tensor([int(truth_first)], dtype=torch.long, device=state.device),
        )
        last_loss = ep_scale * F.cross_entropy(
            last_logits.unsqueeze(0),
            torch.tensor([int(truth_last)], dtype=torch.long, device=state.device),
        )
    return {
        "count_loss": count_loss,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "total_loss": count_loss + first_loss + last_loss,
    }


def _theorem_feature_task_supervision_terms(
    model: FNOCountSketch,
    state: torch.Tensor,
    *,
    truth_target: float,
) -> Dict[str, torch.Tensor]:
    pred_target = model.predict_task_count_from_state(state)
    target_value = torch.tensor(
        float(truth_target),
        device=state.device,
        dtype=pred_target.dtype,
    )
    task_loss = F.mse_loss(pred_target, target_value)
    return {
        "task_loss": task_loss,
        "total_loss": task_loss,
    }


def _local_supervision_terms(
    model: FNOCountSketch,
    state: torch.Tensor,
    *,
    truth_count: float,
    truth_first: int | None = None,
    truth_last: int | None = None,
    supervision_kind: str,
) -> Dict[str, torch.Tensor]:
    kind = str(supervision_kind or "count_only").strip().lower() or "count_only"
    if kind == "full_sketch":
        if model.use_summary_spec:
            return _summary_spec_supervision_terms(
                model,
                state,
                truth_count=float(truth_count),
                truth_first=truth_first,
                truth_last=truth_last,
                supervise_count=True,
                supervise_endpoints=(
                    truth_first is not None and truth_last is not None
                ),
            )
        pred_norm = model.predict_norm_from_state(state)
        target_norm = torch.tensor(
            float(truth_count) / float(model.target_scale),
            device=state.device,
            dtype=pred_norm.dtype,
        )
        count_loss = (pred_norm - target_norm) ** 2
        first_loss = torch.zeros((), device=state.device, dtype=pred_norm.dtype)
        last_loss = torch.zeros((), device=state.device, dtype=pred_norm.dtype)
        if truth_first is not None and truth_last is not None:
            _h, first_logits, last_logits = model._split_state(state)
            first_loss = F.cross_entropy(
                first_logits.unsqueeze(0),
                torch.tensor([int(truth_first)], dtype=torch.long, device=state.device),
            )
            last_loss = F.cross_entropy(
                last_logits.unsqueeze(0),
                torch.tensor([int(truth_last)], dtype=torch.long, device=state.device),
            )
        return {
            "count_loss": count_loss,
            "first_loss": first_loss,
            "last_loss": last_loss,
            "total_loss": count_loss + first_loss + last_loss,
        }
    if kind == "bounded_full_sketch":
        if model.use_summary_spec:
            return _summary_spec_bounded_supervision_terms(
                model,
                state,
                truth_count=float(truth_count),
                truth_first=truth_first,
                truth_last=truth_last,
                supervise_endpoints=(
                    truth_first is not None and truth_last is not None
                ),
            )
        pred_norm = model.predict_norm_from_state(state)
        target_norm = torch.tensor(
            float(truth_count) / float(model.target_scale),
            device=state.device,
            dtype=pred_norm.dtype,
        )
        count_loss = (pred_norm - target_norm) ** 2
        first_loss = torch.zeros((), device=state.device, dtype=pred_norm.dtype)
        last_loss = torch.zeros((), device=state.device, dtype=pred_norm.dtype)
        active_terms = 1.0
        if truth_first is not None and truth_last is not None:
            _h, first_logits, last_logits = model._split_state(state)
            first_loss = _bounded_endpoint_surprise_loss(
                first_logits,
                torch.tensor([int(truth_first)], device=state.device),
            ).reshape(())
            last_loss = _bounded_endpoint_surprise_loss(
                last_logits,
                torch.tensor([int(truth_last)], device=state.device),
            ).reshape(())
            active_terms += 2.0
        return {
            "count_loss": count_loss,
            "first_loss": first_loss,
            "last_loss": last_loss,
            "total_loss": (count_loss + first_loss + last_loss) / float(active_terms),
        }
    if kind != "count_only":
        raise ValueError(
            "supervision_kind must be one of {'count_only','bounded_full_sketch','full_sketch'}"
        )
    if model.use_summary_spec:
        return _summary_spec_bounded_supervision_terms(
            model,
            state,
            truth_count=float(truth_count),
            supervise_endpoints=False,
        )
    pred_norm = model.predict_norm_from_state(state)
    target_norm = torch.tensor(
        float(truth_count) / float(model.target_scale),
        device=state.device,
        dtype=pred_norm.dtype,
    )
    count_loss = (pred_norm - target_norm) ** 2
    zero = torch.zeros((), device=state.device, dtype=pred_norm.dtype)
    return {
        "count_loss": count_loss,
        "first_loss": zero,
        "last_loss": zero,
        "total_loss": count_loss,
    }


def _summary_spec_merge_consistency_terms(
    model: FNOCountSketch,
    left_state: torch.Tensor,
    right_state: torch.Tensor,
    parent_state: torch.Tensor,
    *,
    truth_join_bit: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Algebraic merge consistency (C3 = L2).

    Tests the sufficient-statistic merge formula:
      count(parent) = count(left) + count(right) + join(left, right)
      first(parent) = first(left)
      last(parent)  = last(right)

    When ``truth_join_bit`` is provided, the join head receives direct
    BCE supervision (the strongest signal for learning boundary detection).
    The join probability is NOT detached in the count-additivity target,
    so the C3 count gradient also flows through the join head — but the
    BCE anchor prevents it from learning arbitrary values.

    Child counts (left, right) are detached so C3 gradient flows only
    to the merger and join head, not back through the encoder (C1's job).
    """
    if not model.use_markov_summary_spec:
        raise RuntimeError("summary spec merge consistency requested without markov summary spec")
    if bool(getattr(model, "exact_projected_merge_is_runtime_merge", False)):
        zero = torch.zeros((), device=parent_state.device, dtype=parent_state.dtype)
        join_prob = model.predict_join_prob_from_states(left_state, right_state)
        return {
            "count_loss": zero,
            "first_loss": zero,
            "last_loss": zero,
            "join_loss": zero,
            "join_prob": join_prob,
            "total_loss": zero,
        }
    # Detach child counts (trained by C1), but NOT join_prob (trained here).
    left_count = model.predict_count_from_state(left_state).detach()
    right_count = model.predict_count_from_state(right_state).detach()
    parent_count = model.predict_count_from_state(parent_state)
    _lh, left_first, _left_last = model._split_state(left_state)
    _rh, _right_first, right_last = model._split_state(right_state)
    _ph, parent_first, parent_last = model._split_state(parent_state)
    # Join bit prediction — gradient flows through for count additivity.
    join_logit = model.predict_join_logit_from_states(left_state, right_state)
    join_prob = torch.sigmoid(join_logit)
    # Count additivity: count(P) = count(L) + count(R) + join
    # join_prob is NOT detached — C3 count gradient teaches the join head.
    count_target = left_count + right_count + join_prob.to(dtype=parent_count.dtype)
    count_loss = F.mse_loss(
        parent_count / float(model.target_scale),
        count_target / float(model.target_scale),
    )
    # Endpoint propagation — equal weight with count.
    first_loss = _soft_endpoint_consistency_loss(parent_first, left_first)
    last_loss = _soft_endpoint_consistency_loss(parent_last, right_last)
    # Direct BCE supervision on join head when oracle is available.
    join_loss = torch.zeros((), device=parent_state.device, dtype=parent_state.dtype)
    if truth_join_bit is not None:
        join_loss = F.binary_cross_entropy_with_logits(
            join_logit,
            torch.full_like(
                join_logit,
                float(int(truth_join_bit)),
            ),
        )
    return {
        "count_loss": count_loss,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "join_loss": join_loss,
        "join_prob": join_prob,
        "total_loss": (
            count_loss
            + first_loss
            + last_loss
            + float(model.join_bit_weight) * join_loss
        ),
    }


def _markov_merge_objective_terms(
    model: FNOCountSketch,
    left_state: torch.Tensor,
    right_state: torch.Tensor,
    parent_state: torch.Tensor,
    *,
    truth_count: float,
    truth_first: int | None = None,
    truth_last: int | None = None,
    objective_mode: str = "strict_c3",
) -> Dict[str, torch.Tensor]:
    """Return a reusable Markov merge-training objective.

    ``strict_c3`` is the pure algebraic C3/L2 loss on learned child/parent
    states. The teacher-guided modes supervise the realized parent sketch
    directly while keeping the latent carrier opaque:

    - ``teacher_parent_count``: parent count target only
    - ``teacher_parent_full_sketch``: full parent `(count, first, last)` target

    These modes are intended for small feasibility studies and targeted signal
    attribution. The main tree trainer can continue to combine objectives at a
    higher level without changing its default behavior.
    """

    normalized_mode = (
        str(objective_mode or "strict_c3").strip().lower() or "strict_c3"
    )
    if normalized_mode not in VALID_MARKOV_MERGE_OBJECTIVE_MODES:
        raise ValueError(
            "objective_mode must be one of "
            f"{VALID_MARKOV_MERGE_OBJECTIVE_MODES}; got {objective_mode!r}"
        )
    if normalized_mode == "strict_c3":
        terms = _summary_spec_merge_consistency_terms(
            model,
            left_state,
            right_state,
            parent_state,
            truth_join_bit=None,
        )
        return {
            "mode": normalized_mode,
            "count_loss": terms["count_loss"],
            "first_loss": terms["first_loss"],
            "last_loss": terms["last_loss"],
            "join_loss": terms["join_loss"],
            "join_prob": terms["join_prob"],
            "total_loss": terms["total_loss"],
        }
    supervision_kind = (
        "count_only" if normalized_mode == "teacher_parent_count" else "full_sketch"
    )
    if normalized_mode == "teacher_parent_count":
        parent_terms = _summary_spec_supervision_terms(
            model,
            parent_state,
            truth_count=float(truth_count),
            supervise_count=True,
            supervise_endpoints=False,
        )
    else:
        parent_terms = _local_supervision_terms(
            model,
            parent_state,
            truth_count=float(truth_count),
            truth_first=truth_first,
            truth_last=truth_last,
            supervision_kind=supervision_kind,
        )
    zero = torch.zeros((), device=parent_state.device, dtype=parent_state.dtype)
    return {
        "mode": normalized_mode,
        "count_loss": parent_terms["count_loss"],
        "first_loss": parent_terms["first_loss"],
        "last_loss": parent_terms["last_loss"],
        "join_loss": zero,
        "join_prob": model.predict_join_prob_from_states(left_state, right_state),
        "total_loss": parent_terms["total_loss"],
    }


def _markov_merge_objective_terms_batched(
    model: FNOCountSketch,
    left_state_batch: torch.Tensor,
    right_state_batch: torch.Tensor,
    parent_state_batch: torch.Tensor,
    *,
    truth_counts: torch.Tensor,
    truth_first: torch.Tensor | None = None,
    truth_last: torch.Tensor | None = None,
    objective_mode: str = "strict_c3",
) -> Dict[str, torch.Tensor]:
    """Vectorized variant of `_markov_merge_objective_terms`."""

    normalized_mode = (
        str(objective_mode or "strict_c3").strip().lower() or "strict_c3"
    )
    if normalized_mode not in VALID_MARKOV_MERGE_OBJECTIVE_MODES:
        raise ValueError(
            "objective_mode must be one of "
            f"{VALID_MARKOV_MERGE_OBJECTIVE_MODES}; got {objective_mode!r}"
        )
    if left_state_batch.ndim != 2 or right_state_batch.ndim != 2 or parent_state_batch.ndim != 2:
        raise ValueError("merge objective batches must be rank-2 tensors")
    if int(left_state_batch.shape[0]) != int(parent_state_batch.shape[0]) or int(
        right_state_batch.shape[0]
    ) != int(parent_state_batch.shape[0]):
        raise ValueError("left/right/parent state batches must align")
    if truth_counts.ndim != 1 or int(truth_counts.shape[0]) != int(parent_state_batch.shape[0]):
        raise ValueError("truth_counts must be rank-1 and align with parent_state_batch")
    if normalized_mode == "strict_c3":
        terms = _summary_spec_merge_consistency_terms(
            model,
            left_state_batch,
            right_state_batch,
            parent_state_batch,
            truth_join_bit=None,
        )
        node_count = int(parent_state_batch.shape[0])
        zero_vec = torch.zeros(
            (node_count,),
            device=parent_state_batch.device,
            dtype=parent_state_batch.dtype,
        )
        mean_total = terms["total_loss"]
        mean_count = terms["count_loss"]
        mean_first = terms["first_loss"]
        mean_last = terms["last_loss"]
        mean_join = terms["join_loss"]
        return {
            "mode": normalized_mode,
            "count_loss": mean_count.expand(node_count),
            "first_loss": mean_first.expand(node_count),
            "last_loss": mean_last.expand(node_count),
            "join_loss": mean_join.expand(node_count),
            "join_prob": terms["join_prob"],
            "total_loss": mean_total.expand(node_count),
            "mean_count_loss": mean_count,
            "mean_first_loss": mean_first,
            "mean_last_loss": mean_last,
            "mean_join_loss": mean_join,
            "mean_total_loss": mean_total,
            "node_mask": torch.ones_like(zero_vec, dtype=torch.bool),
        }
    if normalized_mode == "teacher_parent_count":
        parent_terms = _summary_spec_supervision_terms_batched(
            model,
            parent_state_batch,
            truth_counts=truth_counts,
            supervise_count=True,
            supervise_endpoints=False,
        )
    else:
        if truth_first is None or truth_last is None:
            raise ValueError(
                "truth_first and truth_last are required for teacher_parent_full_sketch"
            )
        parent_terms = _summary_spec_supervision_terms_batched(
            model,
            parent_state_batch,
            truth_counts=truth_counts,
            truth_first=truth_first,
            truth_last=truth_last,
            supervise_count=True,
            supervise_endpoints=True,
        )
    zero = torch.zeros_like(parent_terms["count_loss"])
    return {
        "mode": normalized_mode,
        "count_loss": parent_terms["count_loss"],
        "first_loss": parent_terms["first_loss"],
        "last_loss": parent_terms["last_loss"],
        "join_loss": zero,
        "join_prob": model.predict_join_prob_from_states(left_state_batch, right_state_batch),
        "total_loss": parent_terms["total_loss"],
        "mean_count_loss": parent_terms["count_loss"].mean(),
        "mean_first_loss": parent_terms["first_loss"].mean(),
        "mean_last_loss": parent_terms["last_loss"].mean(),
        "mean_join_loss": zero.mean(),
        "mean_total_loss": parent_terms["total_loss"].mean(),
        "node_mask": torch.ones_like(parent_terms["count_loss"], dtype=torch.bool),
    }


def _summary_spec_on_range_reencode_terms(
    model: FNOCountSketch,
    state: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if not model.use_markov_summary_spec or model.codec_contract is None:
        raise RuntimeError("on-range reencode requested without markov summary spec")
    decoded = model.codec_contract.decode(state)
    replay_state = model.codec_contract.reencode(
        DecodedMarkovSketch(
            count=decoded.count.detach(),
            first=decoded.first.detach(),
            last=decoded.last.detach(),
        )
    )
    # Compare replayed predictions to the original decoded values.
    # All three sufficient-stat components are weighted equally (Lean L3
    # treats distortion D as a single pseudo-metric over the full oracle).
    replay_count = model.predict_count_from_state(replay_state)
    original_count = decoded.count.detach()
    count_loss = F.mse_loss(
        replay_count / float(model.target_scale),
        original_count / float(model.target_scale),
    )
    _h, replay_first_logits, replay_last_logits = model._split_state(replay_state)
    first_target = decoded.first.detach().to(dtype=torch.long, device=replay_state.device)
    last_target = decoded.last.detach().to(dtype=torch.long, device=replay_state.device)
    if first_target.ndim == 0:
        first_target = first_target.unsqueeze(0)
    if last_target.ndim == 0:
        last_target = last_target.unsqueeze(0)
    if replay_first_logits.ndim == 1:
        replay_first_logits = replay_first_logits.unsqueeze(0)
    if replay_last_logits.ndim == 1:
        replay_last_logits = replay_last_logits.unsqueeze(0)
    first_loss = F.cross_entropy(replay_first_logits, first_target)
    last_loss = F.cross_entropy(replay_last_logits, last_target)
    return {
        "count_loss": count_loss,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "total_loss": count_loss + first_loss + last_loss,
    }


def _deterministic_sample_ordering(
    *,
    n_items: int,
    seed: int,
) -> Tuple[int, ...]:
    n = int(max(0, n_items))
    if n <= 0:
        return tuple()
    import random as _random

    rng = _random.Random(int(seed))
    ordering = list(range(n))
    rng.shuffle(ordering)
    return tuple(int(index) for index in ordering)


def _deterministic_sample_indices_from_ordering(
    *,
    ordering: Sequence[int],
    rate: float,
    n_items: int | None = None,
) -> Tuple[int, ...] | None:
    n = int(max(0, n_items if n_items is not None else len(ordering)))
    sample_rate = float(rate)
    if n <= 0 or sample_rate <= 0.0:
        return tuple()
    if sample_rate >= 1.0:
        return None
    sample_count = max(1, int(round(sample_rate * float(n))))
    sample_count = min(n, int(sample_count))
    if sample_count >= n:
        return None
    normalized = [int(index) for index in list(ordering)[:n]]
    if len(normalized) < n:
        raise ValueError(
            f"ordering shorter than n_items ({len(normalized)} < {n})"
        )
    # Build one deterministic ordering per doc so higher supervision rates
    # strictly extend lower-rate selections for the same seed.
    return tuple(sorted(int(index) for index in normalized[:sample_count]))


def _deterministic_sample_indices(
    *,
    n_items: int,
    rate: float,
    seed: int,
) -> Tuple[int, ...] | None:
    ordering = _deterministic_sample_ordering(n_items=n_items, seed=seed)
    return _deterministic_sample_indices_from_ordering(
        ordering=ordering,
        rate=rate,
        n_items=n_items,
    )


def _theorem_count_threshold_pos_weights_from_docs(
    docs: Sequence[_FNOCountDoc],
    *,
    max_count: int,
) -> np.ndarray:
    threshold_count = max(0, int(max_count))
    if threshold_count <= 0:
        return np.zeros((0,), dtype=np.float32)
    counts: List[int] = []
    for doc in docs:
        counts.extend(int(round(float(value))) for value in list(doc.leaf_counts))
        counts.extend(
            int(round(float(value))) for value in list(doc.merge_counts_balanced)
        )
        counts.append(int(round(float(doc.root_count))))
    if not counts:
        return np.ones((threshold_count,), dtype=np.float32)
    count_arr = np.asarray(counts, dtype=np.int64)
    weights: List[float] = []
    total = int(count_arr.size)
    for threshold in range(1, threshold_count + 1):
        positives = int(np.sum(count_arr >= int(threshold)))
        negatives = int(total - positives)
        if positives <= 0 or negatives <= 0:
            weights.append(1.0)
        else:
            weights.append(float(np.clip(float(negatives) / float(positives), 1e-3, 1e3)))
    return np.asarray(weights, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training loop for FNO tree-merge with local laws
# ---------------------------------------------------------------------------


def _curriculum_selection_value(
    *,
    tree_root_mae: float,
    doc_sequence_root_mae: float,
    doc_sequence_fraction: float,
) -> float:
    fraction = min(1.0, max(0.0, float(doc_sequence_fraction)))
    if fraction <= 0.0:
        return float(tree_root_mae)
    if fraction >= 1.0:
        return float(doc_sequence_root_mae)
    return float(
        (1.0 - fraction) * float(tree_root_mae)
        + fraction * float(doc_sequence_root_mae)
    )


def _empty_exact_sketch_direct_metrics() -> Dict[str, Any]:
    return {
        "root_direct_count_mae": 0.0,
        "exact_projected_root_mae": 0.0,
        "certified_projected_root_mae": 0.0,
        "root_mae_predicted_counts_predicted_endpoints": 0.0,
        "root_mae_oracle_counts_predicted_endpoints": 0.0,
        "root_mae_predicted_counts_oracle_endpoints": 0.0,
        "learned_merger_gap": 0.0,
        "task_root_mae": 0.0,
        "task_root_mae_ablation": 0.0,
        "leaf_direct_count_mae": 0.0,
        "leaf_direct_exact_match": 1.0,
        "leaf_first_accuracy": 1.0,
        "leaf_last_accuracy": 1.0,
        "merge_first_accuracy": 1.0,
        "merge_last_accuracy": 1.0,
        "leaf_count_off_by_k_histogram": {},
        "merge_exact_summary_match_rate_by_depth": {},
        "merge_direct_exact_match": 1.0,
        "merge_join_bit_accuracy": 1.0,
        "c2_on_range_exact_match": 1.0,
        "phi_merge_alignment": 1.0,
        "phi_within_class_variance": 0.0,
        "phi_between_class_margin": 0.0,
        "phi_pair_same_accuracy": float("nan"),
        "phi_pair_diff_accuracy": float("nan"),
        "phi_pair_auc": float("nan"),
        "phi_replay_same_class_rate": float("nan"),
        "task_factorization_gap": 0.0,
        "val_leaf_codec_direct": 0.0,
        "val_theorem_bootstrap_direct": 0.0,
        "val_exact_sketch_direct": 0.0,
        "val_task_root_exact_sketch_direct": 0.0,
        "n_docs": 0.0,
        "n_leaf_nodes": 0.0,
        "n_merge_nodes": 0.0,
    }


def _selection_uses_exact_projected_merge(model: Any) -> bool:
    """Return whether theorem-facing selection should treat a model as exact-merge.

    Selection follows runtime semantics.  Exact Markov sketch composition can be
    used as a diagnostic/oracle reference without being the learned tree merge.
    """
    runtime_merge_kind = str(getattr(model, "runtime_merge_kind", "")).strip().lower()
    if runtime_merge_kind:
        return runtime_merge_kind == "exact_projected_sketch"
    if bool(getattr(model, "uses_unified_g_learned_merge", False)):
        return False
    tree_model_version = str(
        getattr(model, "tree_model_version", "")
        or getattr(model, "tree_tree_model_version", "")
    ).strip().lower()
    if tree_model_version == "unified_g":
        return False
    if hasattr(model, "exact_projected_merge_is_runtime_merge"):
        return bool(getattr(model, "exact_projected_merge_is_runtime_merge"))
    if bool(getattr(model, "use_exact_projected_sketch_merge", False)):
        return True
    score_merge_mode = str(getattr(model, "score_merge_mode", "")).strip().lower()
    if score_merge_mode == "exact_projected_sketch":
        return True
    theorem_surface_mode = str(
        getattr(model, "theorem_surface_mode", "")
    ).strip().lower()
    if not theorem_surface_mode:
        theorem_surface_mode = str(
            getattr(model, "tree_theorem_surface_mode", "")
        ).strip().lower()
    if theorem_surface_mode == "opaque_carrier_exact_sketch":
        return True
    return False


def _exact_sketch_selection_root_mae(
    model: FNOCountSketch,
    *,
    root_direct_count_mae: float,
    exact_projected_root_mae: float,
) -> float:
    """Return the root metric used for exact-sketch checkpoint selection.

    When the runtime merge is itself the exact projected Markov merge, the
    theorem-facing checkpoint metric should track the certified projected root
    error rather than any auxiliary internal root path.
    """
    if _selection_uses_exact_projected_merge(model) and np.isfinite(
        float(exact_projected_root_mae)
    ):
        return float(exact_projected_root_mae)
    return float(root_direct_count_mae)


def _exact_sketch_selection_split_penalty(
    model: FNOCountSketch,
    *,
    root_mae_oracle_counts_predicted_endpoints: float,
    root_mae_predicted_counts_oracle_endpoints: float,
) -> float:
    """Return an exact-lane penalty for certified count/endpoint disagreement.

    The exact projected root MAE can be spuriously low when count and endpoint
    errors cancel at the root. For theorem-facing exact-merge runs, selection
    should penalize that split failure mode directly instead of rewarding the
    cancellation. Non-exact lanes keep the legacy zero penalty.
    """
    if not _selection_uses_exact_projected_merge(model):
        return 0.0
    endpoint_only = float(root_mae_oracle_counts_predicted_endpoints)
    count_only = float(root_mae_predicted_counts_oracle_endpoints)
    if not (np.isfinite(endpoint_only) and np.isfinite(count_only)):
        return 0.0
    return 0.5 * (endpoint_only + count_only)


@dataclass
class _VectorMomentAccumulator:
    count: int = 0
    sum_vec: np.ndarray | None = None
    sum_sq_norm: float = 0.0

    def add(self, vec: np.ndarray) -> None:
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        if self.sum_vec is None:
            self.sum_vec = np.zeros_like(arr, dtype=np.float64)
        self.sum_vec = self.sum_vec + arr
        self.sum_sq_norm += float(np.dot(arr, arr))
        self.count += 1

    def centroid(self) -> np.ndarray | None:
        if self.count <= 0 or self.sum_vec is None:
            return None
        return self.sum_vec / float(self.count)

    def mean_sq_distance(self) -> float:
        centroid = self.centroid()
        if centroid is None:
            return 0.0
        return float(
            max(
                0.0,
                float(self.sum_sq_norm / float(self.count))
                - float(np.dot(centroid, centroid)),
            )
        )


def _merge_decoded_markov_sketches_exact(
    left: DecodedMarkovSketch,
    right: DecodedMarkovSketch,
) -> DecodedMarkovSketch:
    left_count = left.count.to(dtype=torch.float32)
    right_count = right.count.to(dtype=torch.float32)
    join = left.last.to(dtype=torch.long).ne(right.first.to(dtype=torch.long)).to(
        dtype=left_count.dtype
    )
    return DecodedMarkovSketch(
        count=left_count + right_count + join,
        first=left.first.to(dtype=torch.long),
        last=right.last.to(dtype=torch.long),
    )


def _merge_markov_sketch_values_exact(
    left: Tuple[float, int, int],
    right: Tuple[float, int, int],
) -> Tuple[float, int, int]:
    left_count, left_first, left_last = left
    right_count, right_first, right_last = right
    join = 0.0 if int(left_last) == int(right_first) else 1.0
    return (
        float(left_count) + float(right_count) + float(join),
        int(left_first),
        int(right_last),
    )


def _exact_projected_root_count_from_leaf_components(
    counts: Sequence[float] | np.ndarray | torch.Tensor,
    firsts: Sequence[int] | np.ndarray | torch.Tensor,
    lasts: Sequence[int] | np.ndarray | torch.Tensor,
    *,
    schedule: ScheduleName = "balanced",
) -> float:
    count_values = np.asarray(counts, dtype=np.float64).reshape(-1)
    first_values = np.asarray(firsts, dtype=np.int64).reshape(-1)
    last_values = np.asarray(lasts, dtype=np.int64).reshape(-1)
    if (
        int(count_values.shape[0]) != int(first_values.shape[0])
        or int(count_values.shape[0]) != int(last_values.shape[0])
    ):
        raise ValueError("count, first, and last sequences must align")
    cur: List[Tuple[float, int, int]] = [
        (
            float(count_values[idx]),
            int(first_values[idx]),
            int(last_values[idx]),
        )
        for idx in range(int(count_values.shape[0]))
    ]
    if not cur:
        return 0.0
    normalized_schedule = str(schedule or "balanced")
    if normalized_schedule == "balanced":
        while len(cur) > 1:
            nxt: List[Tuple[float, int, int]] = []
            pair_count = int(len(cur) // 2)
            for idx in range(pair_count):
                nxt.append(
                    _merge_markov_sketch_values_exact(
                        cur[2 * idx],
                        cur[2 * idx + 1],
                    )
                )
            if len(cur) % 2 == 1:
                nxt.append(cur[-1])
            cur = nxt
        return float(cur[0][0])
    if normalized_schedule == "left_to_right":
        acc = cur[0]
        for item in cur[1:]:
            acc = _merge_markov_sketch_values_exact(acc, item)
        return float(acc[0])
    if normalized_schedule == "right_to_left":
        acc = cur[-1]
        for item in reversed(cur[:-1]):
            acc = _merge_markov_sketch_values_exact(item, acc)
        return float(acc[0])
    raise ValueError(f"unsupported exact-projected schedule: {schedule!r}")


def _exact_markov_root_error_decomposition(
    *,
    truth_root_count: float,
    predicted_counts: Sequence[float] | np.ndarray | torch.Tensor,
    predicted_first: Sequence[int] | np.ndarray | torch.Tensor,
    predicted_last: Sequence[int] | np.ndarray | torch.Tensor,
    truth_counts: Sequence[float] | np.ndarray | torch.Tensor,
    truth_first: Sequence[int] | np.ndarray | torch.Tensor,
    truth_last: Sequence[int] | np.ndarray | torch.Tensor,
    schedule: ScheduleName = "balanced",
) -> Dict[str, float]:
    predicted_counts_predicted_endpoints = _exact_projected_root_count_from_leaf_components(
        predicted_counts,
        predicted_first,
        predicted_last,
        schedule=schedule,
    )
    oracle_counts_predicted_endpoints = _exact_projected_root_count_from_leaf_components(
        truth_counts,
        predicted_first,
        predicted_last,
        schedule=schedule,
    )
    predicted_counts_oracle_endpoints = _exact_projected_root_count_from_leaf_components(
        predicted_counts,
        truth_first,
        truth_last,
        schedule=schedule,
    )
    return {
        "root_mae_predicted_counts_predicted_endpoints": abs(
            float(predicted_counts_predicted_endpoints) - float(truth_root_count)
        ),
        "root_mae_oracle_counts_predicted_endpoints": abs(
            float(oracle_counts_predicted_endpoints) - float(truth_root_count)
        ),
        "root_mae_predicted_counts_oracle_endpoints": abs(
            float(predicted_counts_oracle_endpoints) - float(truth_root_count)
        ),
    }


def _leaf_count_off_by_k_count_histogram(
    predicted_counts: Sequence[float] | np.ndarray | torch.Tensor,
    truth_counts: Sequence[float] | np.ndarray | torch.Tensor,
) -> Dict[str, float]:
    pred = np.rint(np.asarray(predicted_counts, dtype=np.float64).reshape(-1)).astype(np.int64)
    truth = np.rint(np.asarray(truth_counts, dtype=np.float64).reshape(-1)).astype(np.int64)
    if int(pred.shape[0]) != int(truth.shape[0]):
        raise ValueError("predicted and truth count sequences must align")
    histogram: Dict[str, float] = {}
    for abs_diff in np.abs(pred - truth).tolist():
        key = str(int(abs_diff))
        histogram[key] = float(histogram.get(key, 0.0) + 1.0)
    return histogram


def _normalize_count_histogram(
    histogram: Mapping[str, float],
) -> Dict[str, float]:
    total = float(sum(float(value) for value in histogram.values()))
    if total <= 0.0:
        return {}
    ordered_keys = sorted(
        histogram.keys(),
        key=lambda key: (int(key) if str(key).isdigit() else str(key)),
    )
    return {
        str(key): float(float(histogram[key]) / total)
        for key in ordered_keys
    }


def _merge_rate_dict_from_sums(
    sum_by_key: Mapping[str, float],
    count_by_key: Mapping[str, int],
) -> Dict[str, float]:
    keys = sorted(
        {str(key) for key in sum_by_key.keys()} | {str(key) for key in count_by_key.keys()},
        key=lambda key: (int(key) if str(key).isdigit() else str(key)),
    )
    rates: Dict[str, float] = {}
    for key in keys:
        count = int(count_by_key.get(str(key), 0))
        if count <= 0:
            continue
        rates[str(key)] = float(float(sum_by_key.get(str(key), 0.0)) / float(count))
    return rates


def _exact_projected_root_count_from_states(
    model: FNOCountSketch,
    states: Sequence[torch.Tensor],
    *,
    schedule: ScheduleName = "balanced",
) -> float:
    decoded_states = [model.decode_markov_codec(state) for state in list(states)]
    if not decoded_states:
        return 0.0
    normalized_schedule = str(schedule or "balanced")
    if normalized_schedule == "balanced":
        cur = list(decoded_states)
        while len(cur) > 1:
            nxt: List[DecodedMarkovSketch] = []
            pair_count = int(len(cur) // 2)
            for idx in range(pair_count):
                nxt.append(
                    _merge_decoded_markov_sketches_exact(
                        cur[2 * idx],
                        cur[2 * idx + 1],
                    )
                )
            if len(cur) % 2 == 1:
                nxt.append(cur[-1])
            cur = nxt
        return float(cur[0].count.detach().cpu().item())
    if normalized_schedule == "left_to_right":
        acc = decoded_states[0]
        for decoded in decoded_states[1:]:
            acc = _merge_decoded_markov_sketches_exact(acc, decoded)
        return float(acc.count.detach().cpu().item())
    if normalized_schedule == "right_to_left":
        acc = decoded_states[-1]
        for decoded in reversed(decoded_states[:-1]):
            acc = _merge_decoded_markov_sketches_exact(decoded, acc)
        return float(acc.count.detach().cpu().item())
    raise ValueError(f"unsupported exact-projected schedule: {schedule!r}")


def _limit_eval_docs(
    docs: Sequence[_FNOCountDoc],
    *,
    doc_limit: int | None = None,
) -> Sequence[_FNOCountDoc]:
    if doc_limit is None:
        return docs
    limit = int(doc_limit)
    if limit <= 0:
        return docs
    return tuple(docs[:limit])


def _mean_or_default(
    *,
    total: float,
    count: int,
    default: float,
) -> float:
    if int(count) <= 0:
        return float(default)
    return float(total / float(count))


def _masked_doc_means(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2 or mask.ndim != 2:
        raise ValueError("values and mask must both be rank-2")
    mask_bool = mask.to(dtype=torch.bool)
    denom = mask_bool.sum(dim=1).clamp_min(1).to(dtype=values.dtype)
    means = (values * mask_bool.to(dtype=values.dtype)).sum(dim=1) / denom
    active = mask_bool.any(dim=1)
    means = torch.where(active, means, torch.zeros_like(means))
    return means, active


def _uniform_subset_inclusion_propensity(
    population_size: int,
    sampled_indices: Sequence[int] | set[int] | None,
) -> float:
    n = int(max(0, int(population_size)))
    if n <= 0:
        return 1.0
    if sampled_indices is None:
        return 1.0
    k = int(len(list(sampled_indices)))
    if k <= 0:
        return 1.0
    return float(min(1.0, max(1.0 / float(n), float(k) / float(n))))


_C2_NODE_KIND_LEAF = 0
_C2_NODE_KIND_MERGE = 1
_C2_NODE_KIND_ROOT = 2


def _effective_subset_sample_size(
    population_size: int,
    sampled_indices: Sequence[int] | set[int] | None,
) -> int:
    n = int(max(0, int(population_size)))
    if n <= 0:
        return 0
    if sampled_indices is None:
        return int(n)
    return int(min(n, max(0, len(list(sampled_indices)))))


def _uniform_subset_pair_inclusion_propensity(
    population_size: int,
    sample_size: int,
) -> float:
    n = int(max(0, int(population_size)))
    k = int(max(0, int(sample_size)))
    if n <= 1:
        return 1.0
    if k >= n:
        return 1.0
    if k <= 1:
        return 0.0
    return float(
        (float(k) * float(k - 1))
        / max(1.0, float(n) * float(n - 1))
    )


def _uniform_subset_single_inclusion_propensity(
    population_size: int,
    sample_size: int,
) -> float:
    n = int(max(0, int(population_size)))
    k = int(max(0, int(sample_size)))
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    if k <= 0:
        return 0.0
    return float(k) / float(n)


def _c2_pair_weighting_mode(
    *,
    tree_supervision_source: str,
    local_estimand_mode: str,
) -> str:
    normalized_source = _normalize_tree_supervision_source(tree_supervision_source)
    normalized_mode = _normalize_tree_local_weighting_mode(local_estimand_mode)
    if (
        normalized_source == "manifest"
        and normalized_mode == "span_mass_ipw_sum"
    ):
        return "pair_ipw_geomean"
    return "legacy_unweighted"


def _c2_pair_inclusion_propensity(
    *,
    kind_left: int,
    kind_right: int,
    leaf_population_size: int,
    leaf_sample_size: int,
    merge_population_size: int,
    merge_sample_size: int,
) -> float:
    left_kind = int(kind_left)
    right_kind = int(kind_right)
    pair_kinds = frozenset((left_kind, right_kind))
    if left_kind == _C2_NODE_KIND_ROOT and right_kind == _C2_NODE_KIND_ROOT:
        return 1.0
    if pair_kinds == frozenset((_C2_NODE_KIND_ROOT, _C2_NODE_KIND_LEAF)):
        return _uniform_subset_single_inclusion_propensity(
            leaf_population_size,
            leaf_sample_size,
        )
    if pair_kinds == frozenset((_C2_NODE_KIND_ROOT, _C2_NODE_KIND_MERGE)):
        return _uniform_subset_single_inclusion_propensity(
            merge_population_size,
            merge_sample_size,
        )
    if pair_kinds == frozenset((_C2_NODE_KIND_LEAF,)):
        return _uniform_subset_pair_inclusion_propensity(
            leaf_population_size,
            leaf_sample_size,
        )
    if pair_kinds == frozenset((_C2_NODE_KIND_MERGE,)):
        return _uniform_subset_pair_inclusion_propensity(
            merge_population_size,
            merge_sample_size,
        )
    if pair_kinds == frozenset((_C2_NODE_KIND_LEAF, _C2_NODE_KIND_MERGE)):
        return (
            _uniform_subset_single_inclusion_propensity(
                leaf_population_size,
                leaf_sample_size,
            )
            * _uniform_subset_single_inclusion_propensity(
                merge_population_size,
                merge_sample_size,
            )
        )
    return 0.0


def _c2_pair_weight_matrix(
    *,
    node_scales: torch.Tensor,
    node_kind_codes: torch.Tensor,
    valid_mask: torch.Tensor,
    leaf_population_size: int,
    leaf_sample_size: int,
    merge_population_size: int,
    merge_sample_size: int,
) -> torch.Tensor:
    if node_scales.ndim != 1 or node_kind_codes.ndim != 1 or valid_mask.ndim != 1:
        raise ValueError("node_scales, node_kind_codes, and valid_mask must be rank-1")
    if (
        node_scales.shape != node_kind_codes.shape
        or node_scales.shape != valid_mask.shape
    ):
        raise ValueError("node_scales, node_kind_codes, and valid_mask must align")
    valid_bool = valid_mask.to(dtype=torch.bool)
    active_nodes = valid_bool & (node_scales > 0)
    n_nodes = int(node_scales.shape[0])
    if int(n_nodes) <= 0:
        return node_scales.new_zeros((0, 0))
    kind_left = node_kind_codes.unsqueeze(1)
    kind_right = node_kind_codes.unsqueeze(0)
    pair_propensity = node_scales.new_zeros((n_nodes, n_nodes))

    def _assign_pair_propensity(
        mask: torch.Tensor,
        propensity: float,
    ) -> None:
        if float(propensity) <= 0.0:
            return
        pair_propensity.masked_fill_(mask, float(propensity))

    leaf_single = _uniform_subset_single_inclusion_propensity(
        leaf_population_size,
        leaf_sample_size,
    )
    merge_single = _uniform_subset_single_inclusion_propensity(
        merge_population_size,
        merge_sample_size,
    )
    leaf_pair = _uniform_subset_pair_inclusion_propensity(
        leaf_population_size,
        leaf_sample_size,
    )
    merge_pair = _uniform_subset_pair_inclusion_propensity(
        merge_population_size,
        merge_sample_size,
    )

    root_root_mask = (
        (kind_left == _C2_NODE_KIND_ROOT)
        & (kind_right == _C2_NODE_KIND_ROOT)
    )
    root_leaf_mask = (
        ((kind_left == _C2_NODE_KIND_ROOT) & (kind_right == _C2_NODE_KIND_LEAF))
        | ((kind_left == _C2_NODE_KIND_LEAF) & (kind_right == _C2_NODE_KIND_ROOT))
    )
    root_merge_mask = (
        ((kind_left == _C2_NODE_KIND_ROOT) & (kind_right == _C2_NODE_KIND_MERGE))
        | ((kind_left == _C2_NODE_KIND_MERGE) & (kind_right == _C2_NODE_KIND_ROOT))
    )
    leaf_leaf_mask = (
        (kind_left == _C2_NODE_KIND_LEAF)
        & (kind_right == _C2_NODE_KIND_LEAF)
    )
    merge_merge_mask = (
        (kind_left == _C2_NODE_KIND_MERGE)
        & (kind_right == _C2_NODE_KIND_MERGE)
    )
    leaf_merge_mask = (
        ((kind_left == _C2_NODE_KIND_LEAF) & (kind_right == _C2_NODE_KIND_MERGE))
        | ((kind_left == _C2_NODE_KIND_MERGE) & (kind_right == _C2_NODE_KIND_LEAF))
    )

    _assign_pair_propensity(root_root_mask, 1.0)
    _assign_pair_propensity(root_leaf_mask, leaf_single)
    _assign_pair_propensity(root_merge_mask, merge_single)
    _assign_pair_propensity(leaf_leaf_mask, leaf_pair)
    _assign_pair_propensity(merge_merge_mask, merge_pair)
    _assign_pair_propensity(leaf_merge_mask, leaf_single * merge_single)

    pair_scale = torch.sqrt(
        (node_scales.clamp_min(0.0).unsqueeze(1) * node_scales.clamp_min(0.0).unsqueeze(0))
    )
    active_pairs = active_nodes.unsqueeze(1) & active_nodes.unsqueeze(0)
    upper = torch.triu(
        torch.ones((n_nodes, n_nodes), device=node_scales.device, dtype=torch.bool),
        diagonal=1,
    )
    active_pairs = active_pairs & upper
    safe_propensity = pair_propensity.clamp_min(1e-12)
    pair_weights = torch.where(
        active_pairs & (pair_propensity > 0.0),
        pair_scale / safe_propensity,
        torch.zeros_like(pair_scale),
    )
    pair_weights = pair_weights + pair_weights.transpose(0, 1)
    return pair_weights


def _masked_doc_hajek_means(
    values: torch.Tensor,
    mask: torch.Tensor,
    propensities: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2 or mask.ndim != 2 or propensities.ndim != 2:
        raise ValueError("values, mask, and propensities must all be rank-2")
    if values.shape != mask.shape or values.shape != propensities.shape:
        raise ValueError("values, mask, and propensities must align")
    mask_bool = mask.to(dtype=torch.bool)
    masked_propensities = torch.where(
        mask_bool,
        propensities.to(device=values.device, dtype=values.dtype).clamp_min(1e-12),
        torch.ones_like(values),
    )
    weights = torch.where(
        mask_bool,
        torch.ones_like(values) / masked_propensities,
        torch.zeros_like(values),
    )
    denom = weights.sum(dim=1).clamp_min(1e-12)
    means = (values * weights).sum(dim=1) / denom
    active = mask_bool.any(dim=1)
    means = torch.where(active, means, torch.zeros_like(means))
    return means, active


def _masked_doc_local_means(
    values: torch.Tensor,
    mask: torch.Tensor,
    propensities: torch.Tensor,
    *,
    weighting_mode: str,
    node_scales: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if values.ndim != 2 or mask.ndim != 2 or propensities.ndim != 2:
        raise ValueError("values, mask, and propensities must all be rank-2")
    if values.shape != mask.shape or values.shape != propensities.shape:
        raise ValueError("values, mask, and propensities must align")
    mode = _normalize_tree_local_weighting_mode(weighting_mode)
    mask_bool = mask.to(dtype=torch.bool)
    masked_propensities = torch.where(
        mask_bool,
        propensities.to(device=values.device, dtype=values.dtype).clamp_min(1e-12),
        torch.ones_like(values),
    )
    if mode == "subset_mean":
        weights = mask_bool.to(dtype=values.dtype)
    elif mode == "fixed_k_hajek":
        weights = torch.where(
            mask_bool,
            torch.ones_like(values) / masked_propensities,
            torch.zeros_like(values),
        )
    else:
        if node_scales is None:
            raise ValueError("node_scales are required for span_mass_ipw_sum")
        if node_scales.shape != values.shape:
            raise ValueError("node_scales must align with values")
        weights = torch.where(
            mask_bool,
            node_scales.to(device=values.device, dtype=values.dtype)
            / masked_propensities,
            torch.zeros_like(values),
        )
    numerators = (values * weights).sum(dim=1)
    denominators = weights.sum(dim=1)
    active = mask_bool.any(dim=1)
    if mode == "span_mass_ipw_sum":
        means = numerators
    else:
        safe_denominators = denominators.clamp_min(1e-12)
        means = numerators / safe_denominators
    means = torch.where(active, means, torch.zeros_like(means))
    numerators = torch.where(active, numerators, torch.zeros_like(numerators))
    denominators = torch.where(active, denominators, torch.zeros_like(denominators))
    return means, active, numerators, denominators


def _masked_vector(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 1 or mask.ndim != 1:
        raise ValueError("values and mask must both be rank-1")
    mask_bool = mask.to(dtype=torch.bool)
    masked = torch.where(mask_bool, values, torch.zeros_like(values))
    return masked, mask_bool


def _summary_spec_count_supervision_terms_batched(
    model: FNOCountSketch,
    state_batch: torch.Tensor,
    *,
    truth_counts: torch.Tensor,
    hidden_batch: torch.Tensor | None = None,
    theorem_feature_batch: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    if state_batch.ndim != 2 or truth_counts.ndim != 1:
        raise ValueError("state_batch must be rank-2 and truth_counts rank-1")
    if int(state_batch.shape[0]) != int(truth_counts.shape[0]):
        raise ValueError("state_batch and truth_counts must align")
    if model.summary_count_classifier is not None:
        if hidden_batch is None:
            if theorem_feature_batch is not None:
                hidden_batch = model._count_hidden_from_theorem_feature(
                    theorem_feature_batch
                )
            else:
                hidden_batch = model._count_hidden_from_state(state_batch)
        count_logits = model.summary_count_classifier(hidden_batch)
        support = model.theorem_count_class_values.to(
            device=state_batch.device,
            dtype=truth_counts.dtype,
        )
        target_indices = torch.argmin(
            torch.abs(support.unsqueeze(0) - truth_counts.unsqueeze(1)),
            dim=1,
        ).to(torch.long)
        primary_loss = F.cross_entropy(
            count_logits,
            target_indices,
            reduction="none",
        )
        aux_loss = torch.zeros_like(primary_loss)
        total_loss = primary_loss
    elif model.uses_hybrid_ordinal_count_head():
        if hidden_batch is None:
            if theorem_feature_batch is not None:
                hidden_batch = model._count_hidden_from_theorem_feature(
                    theorem_feature_batch
                )
            else:
                hidden_batch = model._count_hidden_from_state(state_batch)
        if model.summary_count_ordinal_head is None:
            raise RuntimeError("ordinal count logits requested without hybrid-ordinal theorem head")
        ordinal_logits = model.summary_count_ordinal_head(hidden_batch)
        threshold_values = model.theorem_count_threshold_values.to(
            device=ordinal_logits.device,
            dtype=ordinal_logits.dtype,
        )
        threshold_targets = (
            truth_counts.to(dtype=ordinal_logits.dtype).unsqueeze(1)
            >= threshold_values.unsqueeze(0)
        ).to(dtype=ordinal_logits.dtype)
        threshold_pos_weight = model.theorem_count_threshold_pos_weight.to(
            device=ordinal_logits.device,
            dtype=ordinal_logits.dtype,
        )
        primary_loss = F.binary_cross_entropy_with_logits(
            ordinal_logits,
            threshold_targets,
            pos_weight=threshold_pos_weight,
            reduction="none",
        ).mean(dim=-1)
        if model.summary_count_scalar_aux_head is not None:
            count_norm = torch.sigmoid(model.summary_count_scalar_aux_head(hidden_batch)).squeeze(-1)
        else:
            if model.summary_count_head is None:
                raise RuntimeError("scalar auxiliary count head is not initialized")
            count_norm = torch.sigmoid(model.summary_count_head(hidden_batch)).squeeze(-1)
        pred_count_aux = count_norm * float(model.target_scale)
        aux_loss = (pred_count_aux - truth_counts.to(dtype=pred_count_aux.dtype)) ** 2
        total_loss = (
            float(model.theorem_count_ordinal_weight) * primary_loss
            + float(model.theorem_count_scalar_aux_weight) * aux_loss
        )
    else:
        if model.use_direct_markov_sketch_slots:
            count_norm = model._carrier_count_slot(state_batch).squeeze(-1)
        else:
            if hidden_batch is None:
                if theorem_feature_batch is not None:
                    hidden_batch = model._count_hidden_from_theorem_feature(
                        theorem_feature_batch
                    )
                else:
                    hidden_batch = model._count_hidden_from_state(state_batch)
            if model.summary_count_head is None:
                raise RuntimeError("summary count head is not initialized")
            count_norm = torch.sigmoid(model.summary_count_head(hidden_batch)).squeeze(-1)
        pred_count = count_norm * float(model.target_scale)
        target = truth_counts.to(device=pred_count.device, dtype=pred_count.dtype)
        primary_loss = (pred_count - target) ** 2
        aux_loss = torch.zeros_like(primary_loss)
        total_loss = primary_loss
    return {
        "primary_loss": primary_loss,
        "aux_loss": aux_loss,
        "total_loss": total_loss,
    }


def _summary_spec_endpoint_logits_batched(
    model: FNOCountSketch,
    state_batch: torch.Tensor,
    *,
    theorem_feature_batch: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not model.use_markov_summary_spec:
        raise RuntimeError("summary spec endpoint logits requested without markov summary spec")
    if model.use_direct_markov_sketch_slots:
        return (
            model._carrier_first_logits(state_batch),
            model._carrier_last_logits(state_batch),
        )
    return (
        model.first_endpoint_proj(
            model._first_surface_from_theorem_feature(theorem_feature_batch)
            if theorem_feature_batch is not None
            else model._first_surface_from_state(state_batch)
        ),
        model.last_endpoint_proj(
            model._last_surface_from_theorem_feature(theorem_feature_batch)
            if theorem_feature_batch is not None
            else model._last_surface_from_state(state_batch)
        ),
    )


def _summary_spec_count_norm_batched(
    model: FNOCountSketch,
    state_batch: torch.Tensor,
    *,
    hidden_batch: torch.Tensor | None = None,
    theorem_feature_batch: torch.Tensor | None = None,
) -> torch.Tensor:
    if not model.use_markov_summary_spec:
        raise RuntimeError("summary spec count norm requested without markov summary spec")
    if model.use_direct_markov_sketch_slots:
        return model._carrier_count_slot(state_batch).squeeze(-1)
    if hidden_batch is None:
        if theorem_feature_batch is not None:
            hidden_batch = model._count_hidden_from_theorem_feature(
                theorem_feature_batch
            )
        else:
            hidden_batch = model._count_hidden_from_state(state_batch)
    if model.summary_count_classifier is not None:
        count = model._count_from_logits(model.summary_count_classifier(hidden_batch))
        return count / float(model.target_scale)
    if model.summary_count_ordinal_head is not None:
        count = model._count_from_ordinal_logits(model.summary_count_ordinal_head(hidden_batch))
        return count / float(model.target_scale)
    if model.summary_count_head is None:
        raise RuntimeError("summary count head is not initialized")
    return torch.sigmoid(model.summary_count_head(hidden_batch)).squeeze(-1)


def _summary_spec_supervision_terms_batched(
    model: FNOCountSketch,
    state_batch: torch.Tensor,
    *,
    truth_counts: torch.Tensor,
    truth_first: torch.Tensor | None = None,
    truth_last: torch.Tensor | None = None,
    supervise_count: bool = True,
    supervise_endpoints: bool = False,
    theorem_feature_batch: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    if not model.use_markov_summary_spec:
        raise RuntimeError("summary spec supervision requested without markov summary spec")
    if state_batch.ndim != 2 or truth_counts.ndim != 1:
        raise ValueError("state_batch must be rank-2 and truth_counts rank-1")
    count_loss = torch.zeros(
        (int(state_batch.shape[0]),),
        device=state_batch.device,
        dtype=state_batch.dtype,
    )
    first_loss = torch.zeros_like(count_loss)
    last_loss = torch.zeros_like(count_loss)
    hidden_batch: torch.Tensor | None = None
    shared_theorem_batch = theorem_feature_batch
    if (
        shared_theorem_batch is None
        and model.use_markov_summary_spec
        and model.use_shared_theorem_surface
        and not model.use_direct_markov_sketch_slots
    ):
        shared_theorem_batch = model.theorem_feature_from_state(state_batch)
    if bool(supervise_count) and not model.use_direct_markov_sketch_slots:
        if shared_theorem_batch is not None:
            hidden_batch = model._count_hidden_from_theorem_feature(
                shared_theorem_batch
            )
        else:
            hidden_batch = model._count_hidden_from_state(state_batch)
    if bool(supervise_count):
        count_loss = _summary_spec_count_supervision_terms_batched(
            model,
            state_batch,
            truth_counts=truth_counts,
            hidden_batch=hidden_batch,
            theorem_feature_batch=shared_theorem_batch,
        )["total_loss"]
    if bool(supervise_endpoints):
        if truth_first is None or truth_last is None:
            raise ValueError("truth_first and truth_last are required for endpoint supervision")
        first_logits, last_logits = _summary_spec_endpoint_logits_batched(
            model,
            state_batch,
            theorem_feature_batch=shared_theorem_batch,
        )
        ep_scale = float(model.endpoint_loss_scale)
        first_loss = ep_scale * F.cross_entropy(
            first_logits,
            truth_first.to(device=state_batch.device, dtype=torch.long),
            reduction="none",
        )
        last_loss = ep_scale * F.cross_entropy(
            last_logits,
            truth_last.to(device=state_batch.device, dtype=torch.long),
            reduction="none",
        )
    return {
        "count_loss": count_loss,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "total_loss": count_loss + first_loss + last_loss,
    }


def _summary_spec_bounded_supervision_terms_batched(
    model: FNOCountSketch,
    state_batch: torch.Tensor,
    *,
    truth_counts: torch.Tensor,
    truth_first: torch.Tensor | None = None,
    truth_last: torch.Tensor | None = None,
    supervise_endpoints: bool = False,
    theorem_feature_batch: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    if not model.use_markov_summary_spec:
        raise RuntimeError("bounded summary supervision requested without markov summary spec")
    if state_batch.ndim != 2 or truth_counts.ndim != 1:
        raise ValueError("state_batch must be rank-2 and truth_counts rank-1")
    hidden_batch: torch.Tensor | None = None
    shared_theorem_batch = theorem_feature_batch
    if (
        shared_theorem_batch is None
        and model.use_markov_summary_spec
        and model.use_shared_theorem_surface
        and not model.use_direct_markov_sketch_slots
    ):
        shared_theorem_batch = model.theorem_feature_from_state(state_batch)
    if not model.use_direct_markov_sketch_slots:
        if shared_theorem_batch is not None:
            hidden_batch = model._count_hidden_from_theorem_feature(
                shared_theorem_batch
            )
        else:
            hidden_batch = model._count_hidden_from_state(state_batch)
    pred_norm = _summary_spec_count_norm_batched(
        model,
        state_batch,
        hidden_batch=hidden_batch,
        theorem_feature_batch=shared_theorem_batch,
    )
    target_norm = truth_counts.to(device=pred_norm.device, dtype=pred_norm.dtype) / float(
        model.target_scale
    )
    count_loss = (pred_norm - target_norm) ** 2
    first_loss = torch.zeros_like(count_loss)
    last_loss = torch.zeros_like(count_loss)
    active_terms = torch.ones_like(count_loss)
    if bool(supervise_endpoints):
        if truth_first is None or truth_last is None:
            raise ValueError("truth_first and truth_last are required for endpoint supervision")
        first_logits, last_logits = _summary_spec_endpoint_logits_batched(
            model,
            state_batch,
            theorem_feature_batch=shared_theorem_batch,
        )
        first_loss = _bounded_endpoint_surprise_loss(
            first_logits,
            truth_first.to(device=state_batch.device, dtype=torch.long),
        )
        last_loss = _bounded_endpoint_surprise_loss(
            last_logits,
            truth_last.to(device=state_batch.device, dtype=torch.long),
        )
        active_terms = active_terms + 2.0
    return {
        "count_loss": count_loss,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "total_loss": (count_loss + first_loss + last_loss) / active_terms,
    }


def _theorem_feature_task_supervision_terms_batched(
    model: FNOCountSketch,
    state_batch: torch.Tensor,
    *,
    truth_targets: torch.Tensor,
    theorem_feature_batch: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    # When count_ce is active and the classifier exists, use CE loss.
    if (
        str(getattr(model, "root_supervision_kind", "mse")) == "count_ce"
        and getattr(model, "root_count_classifier", None) is not None
    ):
        logits = model.predict_root_count_logits_from_state(state_batch)
        class_values = model.root_count_class_values
        # Build class index from the model's registered class values.
        _class_index = {int(v.item()): idx for idx, v in enumerate(class_values)}
        # The theorem-feature training path supplies raw count targets here,
        # not normalized count / target_scale values.
        raw_target_classes: List[int] = []
        for target in truth_targets:
            raw_target = int(round(float(target)))
            if raw_target not in _class_index:
                available = sorted(int(v.item()) for v in class_values)
                raise ValueError(
                    "raw theorem-feature CE target is missing from root_count_class_values: "
                    f"target={raw_target} available={available[:12]}"
                    f"{'...' if len(available) > 12 else ''}"
                )
            raw_target_classes.append(int(_class_index[raw_target]))
        target_classes = torch.tensor(
            raw_target_classes,
            dtype=torch.long,
            device=logits.device,
        )
        per_doc_loss = F.cross_entropy(logits, target_classes, reduction="none")
        return {"task_loss": per_doc_loss, "total_loss": per_doc_loss}
    if theorem_feature_batch is not None and model.task_head_mode == "theorem_feature_scalar":
        pred_target = (
            model.predict_task_norm_from_theorem_feature(theorem_feature_batch)
            * float(model.target_scale)
        )
    else:
        pred_target = model.predict_task_count_from_state(state_batch)
    target_value = truth_targets.to(device=pred_target.device, dtype=pred_target.dtype)
    task_loss = (pred_target - target_value) ** 2
    return {
        "task_loss": task_loss,
        "total_loss": task_loss,
    }


def _local_supervision_terms_batched(
    model: FNOCountSketch,
    state_batch: torch.Tensor,
    *,
    truth_counts: torch.Tensor,
    truth_first: torch.Tensor | None = None,
    truth_last: torch.Tensor | None = None,
    supervision_kind: str,
    theorem_feature_batch: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    kind = str(supervision_kind or "count_only").strip().lower() or "count_only"
    shared_theorem_batch = theorem_feature_batch
    if (
        shared_theorem_batch is None
        and model.use_summary_spec
        and model.use_shared_theorem_surface
        and not model.use_direct_markov_sketch_slots
    ):
        shared_theorem_batch = model.theorem_feature_from_state(state_batch)
    if kind == "full_sketch":
        if model.use_summary_spec:
            return _summary_spec_supervision_terms_batched(
                model,
                state_batch,
                truth_counts=truth_counts,
                truth_first=truth_first,
                truth_last=truth_last,
                supervise_count=True,
                supervise_endpoints=(
                    truth_first is not None and truth_last is not None
                ),
                theorem_feature_batch=shared_theorem_batch,
            )
        pred_norm = model.predict_norm_from_state(state_batch)
        target_norm = truth_counts.to(device=pred_norm.device, dtype=pred_norm.dtype) / float(
            model.target_scale
        )
        count_loss = (pred_norm - target_norm) ** 2
        first_loss = torch.zeros_like(count_loss)
        last_loss = torch.zeros_like(count_loss)
        if truth_first is not None and truth_last is not None:
            _h, first_logits, last_logits = model._split_state(state_batch)
            first_loss = F.cross_entropy(
                first_logits,
                truth_first.to(device=state_batch.device, dtype=torch.long),
                reduction="none",
            )
            last_loss = F.cross_entropy(
                last_logits,
                truth_last.to(device=state_batch.device, dtype=torch.long),
                reduction="none",
            )
        return {
            "count_loss": count_loss,
            "first_loss": first_loss,
            "last_loss": last_loss,
            "total_loss": count_loss + first_loss + last_loss,
        }
    if kind == "bounded_full_sketch":
        if model.use_summary_spec:
            return _summary_spec_bounded_supervision_terms_batched(
                model,
                state_batch,
                truth_counts=truth_counts,
                truth_first=truth_first,
                truth_last=truth_last,
                supervise_endpoints=(
                    truth_first is not None and truth_last is not None
                ),
                theorem_feature_batch=shared_theorem_batch,
            )
        pred_norm = model.predict_norm_from_state(state_batch)
        target_norm = truth_counts.to(device=pred_norm.device, dtype=pred_norm.dtype) / float(
            model.target_scale
        )
        count_loss = (pred_norm - target_norm) ** 2
        first_loss = torch.zeros_like(count_loss)
        last_loss = torch.zeros_like(count_loss)
        active_terms = torch.ones_like(count_loss)
        if truth_first is not None and truth_last is not None:
            _h, first_logits, last_logits = model._split_state(state_batch)
            first_loss = _bounded_endpoint_surprise_loss(
                first_logits,
                truth_first.to(device=state_batch.device, dtype=torch.long),
            )
            last_loss = _bounded_endpoint_surprise_loss(
                last_logits,
                truth_last.to(device=state_batch.device, dtype=torch.long),
            )
            active_terms = active_terms + 2.0
        return {
            "count_loss": count_loss,
            "first_loss": first_loss,
            "last_loss": last_loss,
            "total_loss": (count_loss + first_loss + last_loss) / active_terms,
        }
    if kind != "count_only":
        raise ValueError(
            "supervision_kind must be one of {'count_only','bounded_full_sketch','full_sketch'}"
        )
    if model.use_summary_spec:
        return _summary_spec_bounded_supervision_terms_batched(
            model,
            state_batch,
            truth_counts=truth_counts,
            supervise_endpoints=False,
            theorem_feature_batch=shared_theorem_batch,
        )
    pred_norm = model.predict_norm_from_state(state_batch)
    target_norm = truth_counts.to(device=pred_norm.device, dtype=pred_norm.dtype) / float(
        model.target_scale
    )
    count_loss = (pred_norm - target_norm) ** 2
    zero = torch.zeros_like(count_loss)
    return {
        "count_loss": count_loss,
        "first_loss": zero,
        "last_loss": zero,
        "total_loss": count_loss,
    }


def _supports_fixed_fused_batch(
    model: FNOCountSketch,
    items: Sequence[_TreeWorkItem],
) -> bool:
    if not items:
        return False
    docs = [item.doc for item in items]
    if not (_docs_share_fixed_leaf_shape(docs) or _docs_support_fixed_leaf_auto_queue(docs)):
        return False
    if any(bool(item.doc_sequence_supervision) for item in items):
        return False
    if not bool(getattr(model, "use_shared_theorem_surface", False)):
        return False
    return True


def _fixed_fused_fast_markov_c2_tensors(
    *,
    leaf_phi_batch: torch.Tensor,
    merge_phi_batch: torch.Tensor | None,
    root_phi_batch: torch.Tensor,
    leaf_valid_mask: torch.Tensor,
    merge_valid_mask: torch.Tensor | None,
    node_count_targets: torch.Tensor,
    node_count_keys: torch.Tensor,
    node_first_targets: torch.Tensor,
    node_last_targets: torch.Tensor,
    leaf_node_scales: torch.Tensor,
    merge_node_scales: torch.Tensor,
    include_merge: bool,
) -> Dict[str, torch.Tensor]:
    if leaf_phi_batch.ndim != 3 or root_phi_batch.ndim != 2:
        raise ValueError("leaf_phi_batch must be rank-3 and root_phi_batch rank-2")
    parts_phi = [leaf_phi_batch]
    parts_valid = [leaf_valid_mask.to(dtype=torch.bool)]
    parts_count_values = [node_count_targets[:, : int(leaf_phi_batch.shape[1])]]
    parts_count_keys = [node_count_keys[:, : int(leaf_phi_batch.shape[1])]]
    parts_first_targets = [node_first_targets[:, : int(leaf_phi_batch.shape[1])]]
    parts_last_targets = [node_last_targets[:, : int(leaf_phi_batch.shape[1])]]
    parts_kind_codes = [
        torch.full(
            (int(leaf_phi_batch.shape[0]), int(leaf_phi_batch.shape[1])),
            _C2_NODE_KIND_LEAF,
            device=leaf_phi_batch.device,
            dtype=torch.long,
        )
    ]
    parts_node_scales = [leaf_node_scales]
    if bool(include_merge):
        if merge_phi_batch is None or merge_phi_batch.ndim != 3 or merge_valid_mask is None:
            raise ValueError("merge_phi_batch is required when include_merge=True")
        merge_start = int(leaf_phi_batch.shape[1])
        merge_end = merge_start + int(merge_phi_batch.shape[1])
        parts_phi.append(merge_phi_batch)
        parts_valid.append(merge_valid_mask.to(dtype=torch.bool))
        parts_count_values.append(node_count_targets[:, merge_start:merge_end])
        parts_count_keys.append(node_count_keys[:, merge_start:merge_end])
        parts_first_targets.append(node_first_targets[:, merge_start:merge_end])
        parts_last_targets.append(node_last_targets[:, merge_start:merge_end])
        parts_kind_codes.append(
            torch.full(
                (int(merge_phi_batch.shape[0]), int(merge_phi_batch.shape[1])),
                _C2_NODE_KIND_MERGE,
                device=merge_phi_batch.device,
                dtype=torch.long,
            )
        )
        parts_node_scales.append(merge_node_scales)
    parts_phi.append(root_phi_batch.unsqueeze(1))
    parts_valid.append(torch.ones((int(root_phi_batch.shape[0]), 1), device=root_phi_batch.device, dtype=torch.bool))
    parts_count_values.append(node_count_targets[:, -1:])
    parts_count_keys.append(node_count_keys[:, -1:])
    parts_first_targets.append(node_first_targets[:, -1:])
    parts_last_targets.append(node_last_targets[:, -1:])
    parts_kind_codes.append(
        torch.full(
            (int(root_phi_batch.shape[0]), 1),
            _C2_NODE_KIND_ROOT,
            device=root_phi_batch.device,
            dtype=torch.long,
        )
    )
    parts_node_scales.append(
        torch.ones(
            (int(root_phi_batch.shape[0]), 1),
            device=root_phi_batch.device,
            dtype=root_phi_batch.dtype,
        )
    )
    return {
        "phi_batch": torch.cat(parts_phi, dim=1),
        "valid_mask": torch.cat(parts_valid, dim=1),
        "count_values": torch.cat(parts_count_values, dim=1),
        "count_keys": torch.cat(parts_count_keys, dim=1),
        "first_targets": torch.cat(parts_first_targets, dim=1),
        "last_targets": torch.cat(parts_last_targets, dim=1),
        "node_kind_codes": torch.cat(parts_kind_codes, dim=1),
        "node_scales": torch.cat(parts_node_scales, dim=1),
    }


def _fixed_fused_training_batch_forward(
    model: FNOCountSketch,
    packed_tree_batch: _PackedTreeBatch,
    *,
    work_lookup: Mapping[int, Mapping[str, Any]],
    device: torch.device,
    resident_store: GpuBatchStore | None = None,
    runtime_telemetry: GpuRuntimeTelemetry | None = None,
    root_weight: float,
    c1_weight: float,
    c2_weight: float,
    c3_weight: float,
    phi_compose_weight: float,
    leaf_supervision_kind: str,
    internal_supervision_kind: str,
    tree_local_weighting_mode: str = "fixed_k_hajek",
    tree_supervision_source: str = "rate",
    defer_contrastive: bool = False,
    depth_discount_gamma: float = 1.0,
) -> Dict[str, Any]:
    items = list(packed_tree_batch.items)
    if not _supports_fixed_fused_batch(model, items):
        raise ValueError("packed_tree_batch is not eligible for fixed_fused execution")
    docs = [item.doc for item in items]
    batch_size = int(len(docs))
    resident_view = _tree_store_view_for_items(
        resident_store,
        items,
        model=model,
        runtime_telemetry=runtime_telemetry,
    )
    levels = _precompute_balanced_doc_state_levels(
        model,
        docs,
        device=device,
        collect_merge_states=True,
        prefer_fixed_fused=True,
        target_n_leaves=int(packed_tree_batch.bucket_key.n_leaves),
        resident_view=resident_view,
        runtime_telemetry=runtime_telemetry,
    )
    leaf_state_batch = levels.leaf_states
    root_state_batch = levels.root_states
    leaf_valid_mask = levels.leaf_valid_mask
    merge_state_batch = _flatten_merge_state_batch(levels.merge_levels)
    if merge_state_batch is None:
        merge_state_batch = leaf_state_batch.new_zeros(
            (int(leaf_state_batch.shape[0]), 0, int(leaf_state_batch.shape[-1]))
        )
    merge_valid_mask = (
        torch.cat(
            [mask for mask in levels.merge_valid_levels if int(mask.shape[1]) > 0],
            dim=1,
        )
        if levels.merge_valid_levels
        else torch.zeros(
            (int(leaf_state_batch.shape[0]), 0),
            device=device,
            dtype=torch.bool,
        )
    )
    node_valid_mask = levels.node_valid_mask
    n_leaf = int(leaf_state_batch.shape[1])
    n_merge = int(merge_state_batch.shape[1])

    # -- Depth discount factors (Lean: DiscountedTreeMetaObjective) --
    # merge_levels is bottom-up: [0]=deepest (n_leaf/2 nodes), [-1]=root merge.
    # Depth 0 = root, depth d gets gamma^d.
    _gamma = float(depth_discount_gamma)
    _n_merge_levels = len(levels.merge_levels)
    # Leaf depth = number of merge levels (leaves are one level below the
    # deepest merge layer).
    _leaf_depth = _n_merge_levels
    _leaf_depth_discount = _gamma ** _leaf_depth if _leaf_depth > 0 else 1.0
    # Per-merge-node discount, aligned with the flattened merge tensor.
    if _n_merge_levels > 0 and _gamma < 1.0:
        _merge_depth_discounts_list: list[torch.Tensor] = []
        for _level_idx, _level_states in enumerate(levels.merge_levels):
            _level_n = int(_level_states.shape[1])
            if _level_n <= 0:
                continue
            # level_idx=0 is deepest merge; depth = n_merge_levels - 1 - level_idx... no:
            # merge_levels[0] merges leaf pairs → depth = n_merge_levels - 1
            # merge_levels[-1] = root merge → depth = 0
            _depth = _n_merge_levels - 1 - _level_idx
            _discount = _gamma ** _depth
            _merge_depth_discounts_list.append(
                torch.full((_level_n,), _discount, device=device, dtype=root_state_batch.dtype)
            )
        _merge_depth_discount = (
            torch.cat(_merge_depth_discounts_list, dim=0)
            if _merge_depth_discounts_list
            else torch.ones((n_merge,), device=device, dtype=root_state_batch.dtype)
        )
    else:
        _merge_depth_discount = torch.ones((n_merge,), device=device, dtype=root_state_batch.dtype)

    zero = torch.zeros((), device=device, dtype=root_state_batch.dtype)
    works = [work_lookup[int(item.doc_index)] for item in items]
    resident_tensors = dict(resident_view.tensors) if resident_view is not None else {}
    normalized_local_weighting_mode = _normalize_tree_local_weighting_mode(
        tree_local_weighting_mode
    )
    normalized_supervision_source = _normalize_tree_supervision_source(
        tree_supervision_source
    )
    local_loss_kind = _resolved_local_loss_kind(
        leaf_supervision_kind=str(leaf_supervision_kind),
        internal_supervision_kind=str(internal_supervision_kind),
    )

    def _stack_cached_tensor(
        attr_name: str,
        *,
        dtype: torch.dtype,
        resident_name: str,
    ) -> torch.Tensor:
        resident_tensor = resident_tensors.get(str(resident_name))
        if isinstance(resident_tensor, torch.Tensor):
            return resident_tensor.to(
                device=device,
                dtype=dtype,
                non_blocking=bool(resident_tensor.device.type == "cpu" and resident_tensor.is_pinned()),
            )
        cached_targets = tuple(
            _cached_fused_doc_targets_for_target_leaves(
                doc,
                int(packed_tree_batch.bucket_key.n_leaves),
            )
            for doc in docs
        )
        if runtime_telemetry is not None and resident_store is not None:
            runtime_telemetry.add_store_miss(reason=f"missing_{resident_name}")
        stacked = torch.stack(
            [getattr(targets, attr_name) for targets in cached_targets],
            dim=0,
        )
        if device.type == "cuda":
            stacked = stacked.pin_memory()
        return stacked.to(
            device=device,
            dtype=dtype,
            non_blocking=bool(device.type == "cuda"),
        )

    root_mask = torch.as_tensor(
        [bool(work["root_only_supervision"]) for work in works],
        device=device,
        dtype=torch.bool,
    )
    c2_collect_mask = torch.as_tensor(
        [bool(work["collect_c2"]) for work in works],
        device=device,
        dtype=torch.bool,
    )
    leaf_truth_counts = _stack_cached_tensor(
        "leaf_count_targets_cpu",
        dtype=root_state_batch.dtype,
        resident_name="leaf_count_targets",
    )
    leaf_truth_first = _stack_cached_tensor(
        "leaf_first_targets_cpu",
        dtype=torch.long,
        resident_name="leaf_first_targets",
    )
    leaf_truth_last = _stack_cached_tensor(
        "leaf_last_targets_cpu",
        dtype=torch.long,
        resident_name="leaf_last_targets",
    )
    merge_truth_counts = _stack_cached_tensor(
        "merge_count_targets_cpu",
        dtype=root_state_batch.dtype,
        resident_name="merge_count_targets",
    )
    merge_truth_first = _stack_cached_tensor(
        "merge_first_targets_cpu",
        dtype=torch.long,
        resident_name="merge_first_targets",
    )
    merge_truth_last = _stack_cached_tensor(
        "merge_last_targets_cpu",
        dtype=torch.long,
        resident_name="merge_last_targets",
    )
    root_task_targets = (
        resident_tensors["root_targets"].to(
            device=device,
            dtype=root_state_batch.dtype,
            non_blocking=bool(
                resident_tensors["root_targets"].device.type == "cpu"
                and resident_tensors["root_targets"].is_pinned()
            ),
        )
        if isinstance(resident_tensors.get("root_targets"), torch.Tensor)
        else torch.as_tensor(
            [float(doc.root_count) for doc in docs],
            device=device,
            dtype=root_state_batch.dtype,
        )
    )
    node_count_targets = _stack_cached_tensor(
        "node_count_targets_cpu",
        dtype=root_state_batch.dtype,
        resident_name="node_count_targets",
    )
    node_count_keys = _stack_cached_tensor(
        "node_count_keys_cpu",
        dtype=torch.long,
        resident_name="node_count_keys",
    )
    node_first_targets = _stack_cached_tensor(
        "node_first_targets_cpu",
        dtype=torch.long,
        resident_name="node_first_targets",
    )
    node_last_targets = _stack_cached_tensor(
        "node_last_targets_cpu",
        dtype=torch.long,
        resident_name="node_last_targets",
    )

    dense_leaf_fast = bool(
        all(work.get("leaf_audit_indices") is None for work in works)
    )
    dense_merge_fast = bool(
        all(work.get("c3_audit_indices") is None for work in works)
    )
    all_collect_leaf = bool(all(bool(work.get("collect_leaf")) for work in works))
    all_collect_c2 = bool(all(bool(work.get("collect_c2")) for work in works))
    all_collect_c3 = bool(all(bool(work.get("collect_c3")) for work in works))
    any_collect_leaf = bool(any(bool(work.get("collect_leaf")) for work in works))
    any_collect_c2 = bool(any(bool(work.get("collect_c2")) for work in works))
    any_collect_c3 = bool(any(bool(work.get("collect_c3")) for work in works))
    dense_full_fast = bool(
        _supports_fast_markov_pair_masks(model)
        and dense_leaf_fast
        and dense_merge_fast
        and all_collect_leaf
        and (int(n_merge) <= 0 or all_collect_c3)
        and all_collect_c2
    )
    dense_leaf_root_fast = bool(
        _supports_fast_markov_pair_masks(model)
        and dense_leaf_fast
        and all_collect_leaf
        and all_collect_c2
        and not any_collect_c3
    )
    dense_markov_c2_fast = bool(dense_full_fast or dense_leaf_root_fast)
    leaf_supervision_active = bool(int(n_leaf) > 0 and float(c1_weight) > 0.0 and any_collect_leaf)
    merge_supervision_active = bool(int(n_merge) > 0 and float(c3_weight) > 0.0 and any_collect_c3)

    leaf_mask = torch.zeros((batch_size, n_leaf), device=device, dtype=torch.bool)
    merge_mask = torch.zeros((batch_size, n_merge), device=device, dtype=torch.bool)
    if int(n_leaf) > 0:
        leaf_propensity = torch.as_tensor(
            [
                _uniform_subset_inclusion_propensity(
                    int(
                        work.get(
                            "leaf_supervision_population_size",
                            len(work["doc"].leaf_token_ids),
                        )
                    ),
                    work.get("leaf_audit_indices"),
                )
                for work in works
            ],
            device=device,
            dtype=root_state_batch.dtype,
        ).unsqueeze(1).expand(-1, n_leaf)
    else:
        leaf_propensity = torch.zeros(
            (batch_size, 0),
            device=device,
            dtype=root_state_batch.dtype,
        )
    if int(n_merge) > 0:
        merge_propensity = torch.as_tensor(
            [
                _uniform_subset_inclusion_propensity(
                    int(work.get("internal_supervision_population_size", 0)),
                    work.get("c3_audit_indices"),
                )
                for work in works
            ],
            device=device,
            dtype=root_state_batch.dtype,
        ).unsqueeze(1).expand(-1, n_merge)
    else:
        merge_propensity = torch.zeros(
            (batch_size, 0),
            device=device,
            dtype=root_state_batch.dtype,
        )
    leaf_node_scales = torch.zeros(
        (batch_size, n_leaf),
        device=device,
        dtype=root_state_batch.dtype,
    )
    merge_node_scales = torch.zeros(
        (batch_size, n_merge),
        device=device,
        dtype=root_state_batch.dtype,
    )
    for batch_idx, doc in enumerate(docs):
        doc_tokens = int(max(1, int(doc.n_tokens)))
        for leaf_idx in range(min(int(n_leaf), len(doc.leaf_token_lengths))):
            leaf_node_scales[int(batch_idx), int(leaf_idx)] = float(
                doc.leaf_token_lengths[int(leaf_idx)]
            ) / float(doc_tokens)
        for merge_idx in range(min(int(n_merge), len(doc.merge_token_lengths))):
            merge_node_scales[int(batch_idx), int(merge_idx)] = float(
                doc.merge_token_lengths[int(merge_idx)]
            ) / float(doc_tokens)
    if normalized_local_weighting_mode == "span_mass_ipw_sum":
        leaf_node_scales = leaf_node_scales * float(_leaf_depth_discount)
        if int(n_merge) > 0:
            merge_node_scales = merge_node_scales * _merge_depth_discount.unsqueeze(0)
    feature_targets_by_doc: List[Any] = []
    if dense_leaf_fast:
        leaf_mask = leaf_valid_mask & torch.as_tensor(
            [[bool(work.get("collect_leaf"))] * int(n_leaf) for work in works],
            device=device,
            dtype=torch.bool,
        )
    if dense_merge_fast and int(n_merge) > 0:
        merge_mask = merge_valid_mask & torch.as_tensor(
            [[bool(work.get("collect_c3"))] * int(n_merge) for work in works],
            device=device,
            dtype=torch.bool,
        )
    if not dense_markov_c2_fast:
        need_feature_targets = bool(c2_collect_mask.any()) or bool(defer_contrastive)
        for batch_idx, item in enumerate(items):
            work = works[int(batch_idx)]
            if not dense_leaf_fast and bool(work["collect_leaf"]):
                if work["leaf_audit_indices"] is None:
                    leaf_mask[int(batch_idx)] = leaf_valid_mask[int(batch_idx)]
                else:
                    for leaf_idx in _all_or_sampled_indices(int(n_leaf), work["leaf_audit_indices"]):
                        if bool(leaf_valid_mask[int(batch_idx), int(leaf_idx)].item()):
                            leaf_mask[int(batch_idx), int(leaf_idx)] = True
            if not dense_merge_fast and bool(work["collect_c3"]) and int(n_merge) > 0:
                if work["c3_audit_indices"] is None:
                    merge_mask[int(batch_idx)] = merge_valid_mask[int(batch_idx)]
                else:
                    for merge_idx in _all_or_sampled_indices(int(n_merge), work["c3_audit_indices"]):
                        if bool(merge_valid_mask[int(batch_idx), int(merge_idx)].item()):
                            merge_mask[int(batch_idx), int(merge_idx)] = True
            if need_feature_targets:
                exact_targets = _balanced_exact_sketch_targets(
                    leaf_counts=item.doc.leaf_counts,
                    leaf_first_regimes=item.doc.leaf_first_regimes,
                    leaf_last_regimes=item.doc.leaf_last_regimes,
                )
                leaf_metadata, merge_metadata, root_metadata = (
                    _theorem_feature_metadata_sequences_from_fno_doc(item.doc)
                )
                feature_targets_by_doc.append(
                    theorem_feature_targets_from_markov_exact_targets(
                        adapter=model.theorem_feature_adapter,
                        exact_targets=exact_targets,
                        leaf_metadata=leaf_metadata,
                        merge_metadata=merge_metadata,
                        root_metadata=root_metadata,
                    )
                )
            else:
                feature_targets_by_doc.append(None)

    flat_leaf_states = leaf_state_batch.reshape(int(batch_size * n_leaf), -1)
    flat_merge_states = merge_state_batch.reshape(int(batch_size * max(1, n_merge)), -1)

    document_loss_sum = zero.clone()
    non_document_loss_sum = zero.clone()
    batch_loss = zero.clone()
    component_sums: Dict[str, torch.Tensor] = {
        "root_count_loss": zero.clone(),
        "leaf_count_loss": zero.clone(),
        "leaf_first_loss": zero.clone(),
        "leaf_last_loss": zero.clone(),
        "merge_count_loss": zero.clone(),
        "merge_first_loss": zero.clone(),
        "merge_last_loss": zero.clone(),
        "c2_count_loss": zero.clone(),
        "c2_first_loss": zero.clone(),
        "c2_last_loss": zero.clone(),
        "c2_join_loss": zero.clone(),
        "c2_on_range_reencode_loss": zero.clone(),
        "phi_compose_loss": zero.clone(),
        "phi_contrastive_loss": zero.clone(),
    }
    component_counts: Dict[str, int] = {name: 0 for name in component_sums}
    deferred_phi_features: List[torch.Tensor] = []
    deferred_phi_labels: List[Any] = []
    deferred_oracle_vecs: List[Any] = []
    deferred_phi_feature_batch: torch.Tensor | None = None
    deferred_phi_fast_keys: Dict[str, torch.Tensor] | None = None
    c2_pair_weighting_mode = _c2_pair_weighting_mode(
        tree_supervision_source=normalized_supervision_source,
        local_estimand_mode=normalized_local_weighting_mode,
    )
    c2_pair_same_count = torch.zeros(
        (batch_size,),
        device=device,
        dtype=root_state_batch.dtype,
    )
    c2_pair_different_count = torch.zeros_like(c2_pair_same_count)
    c2_pair_weight_ess = torch.zeros_like(c2_pair_same_count)
    c2_pair_weight_max = torch.zeros_like(c2_pair_same_count)
    normalized_leaf_supervision_kind = str(leaf_supervision_kind).strip().lower()
    normalized_internal_supervision_kind = str(internal_supervision_kind).strip().lower()
    needs_node_phi = bool(any_collect_c2) or bool(defer_contrastive)
    need_phi_compose = bool(float(phi_compose_weight) > 0.0 and levels.merge_levels)
    root_task_uses_theorem = bool(model.task_head_mode == "theorem_feature_scalar")
    leaf_task_uses_theorem = bool(
        leaf_supervision_active
        and normalized_leaf_supervision_kind == "full_sketch"
        and root_task_uses_theorem
    )
    merge_task_uses_theorem = bool(
        merge_supervision_active
        and normalized_internal_supervision_kind == "full_sketch"
        and root_task_uses_theorem
    )
    leaf_local_uses_theorem = bool(
        leaf_supervision_active
        and model.use_summary_spec
        and model.use_shared_theorem_surface
        and not model.use_direct_markov_sketch_slots
    )
    merge_local_uses_theorem = bool(
        merge_supervision_active
        and model.use_summary_spec
        and model.use_shared_theorem_surface
        and not model.use_direct_markov_sketch_slots
    )
    need_leaf_theorem_batch = bool(
        int(n_leaf) > 0
        and (leaf_task_uses_theorem or leaf_local_uses_theorem or needs_node_phi or need_phi_compose)
    )
    need_merge_theorem_batch = bool(
        int(n_merge) > 0
        and (merge_task_uses_theorem or merge_local_uses_theorem or needs_node_phi or need_phi_compose)
    )
    need_root_theorem_batch = bool(root_task_uses_theorem or needs_node_phi)
    leaf_theorem_batch: torch.Tensor | None = None
    merge_theorem_batch: torch.Tensor | None = None
    root_theorem_batch: torch.Tensor | None = None
    if need_leaf_theorem_batch:
        leaf_theorem_batch = model.theorem_feature_from_state(flat_leaf_states).reshape(
            batch_size,
            n_leaf,
            -1,
        )
    if need_merge_theorem_batch:
        merge_theorem_batch = model.theorem_feature_from_state(
            merge_state_batch.reshape(batch_size * n_merge, -1)
        ).reshape(batch_size, n_merge, -1)
    if need_root_theorem_batch:
        root_theorem_batch = model.theorem_feature_from_state(root_state_batch)

    root_loss_per_doc = _theorem_feature_task_supervision_terms_batched(
        model,
        root_state_batch,
        truth_targets=root_task_targets,
        theorem_feature_batch=root_theorem_batch,
    )["task_loss"]
    if bool(root_mask.any()):
        component_sums["root_count_loss"] = root_loss_per_doc[root_mask].sum().to(dtype=torch.float32)
        component_counts["root_count_loss"] = int(root_mask.shape[0]) if bool(root_mask.all()) else int(root_mask.sum().item())
        document_loss_sum = (
            document_loss_sum + float(root_weight) * root_loss_per_doc[root_mask].sum()
        )

    if bool(leaf_supervision_active):
        flat_leaf_theorem_batch = (
            leaf_theorem_batch.reshape(batch_size * n_leaf, -1)
            if leaf_theorem_batch is not None
            else None
        )
        if normalized_leaf_supervision_kind == "full_sketch":
            leaf_task_terms = _theorem_feature_task_supervision_terms_batched(
                model,
                flat_leaf_states,
                truth_targets=leaf_truth_counts.reshape(-1),
                theorem_feature_batch=flat_leaf_theorem_batch,
            )
            leaf_task_loss = leaf_task_terms["task_loss"].reshape(batch_size, n_leaf)
            leaf_total_loss = leaf_task_loss
            leaf_count_loss = leaf_task_loss
            leaf_first_loss = torch.zeros_like(leaf_task_loss)
            leaf_last_loss = torch.zeros_like(leaf_task_loss)
            if bool(model.use_summary_spec):
                leaf_summary_terms = _local_supervision_terms_batched(
                    model,
                    flat_leaf_states,
                    truth_counts=leaf_truth_counts.reshape(-1),
                    truth_first=leaf_truth_first.reshape(-1),
                    truth_last=leaf_truth_last.reshape(-1),
                    supervision_kind="full_sketch",
                    theorem_feature_batch=flat_leaf_theorem_batch,
                )
                leaf_total_loss = (
                    leaf_total_loss
                    + leaf_summary_terms["total_loss"].reshape(batch_size, n_leaf)
                )
                leaf_count_loss = (
                    leaf_count_loss
                    + leaf_summary_terms["count_loss"].reshape(batch_size, n_leaf)
                )
                leaf_first_loss = leaf_summary_terms["first_loss"].reshape(batch_size, n_leaf)
                leaf_last_loss = leaf_summary_terms["last_loss"].reshape(batch_size, n_leaf)
        else:
            leaf_terms = _local_supervision_terms_batched(
                model,
                flat_leaf_states,
                truth_counts=leaf_truth_counts.reshape(-1),
                truth_first=leaf_truth_first.reshape(-1),
                truth_last=leaf_truth_last.reshape(-1),
                supervision_kind=str(leaf_supervision_kind),
                theorem_feature_batch=flat_leaf_theorem_batch,
            )
            leaf_total_loss = leaf_terms["total_loss"].reshape(batch_size, n_leaf)
            leaf_count_loss = leaf_terms["count_loss"].reshape(batch_size, n_leaf)
            leaf_first_loss = leaf_terms["first_loss"].reshape(batch_size, n_leaf)
            leaf_last_loss = leaf_terms["last_loss"].reshape(batch_size, n_leaf)
        leaf_doc_loss, leaf_active, leaf_doc_numerators, leaf_doc_denominators = _masked_doc_local_means(
            leaf_total_loss,
            leaf_mask,
            leaf_propensity,
            weighting_mode=normalized_local_weighting_mode,
            node_scales=leaf_node_scales,
        )
        leaf_count_doc_loss, _, _, _ = _masked_doc_local_means(
            leaf_count_loss,
            leaf_mask,
            leaf_propensity,
            weighting_mode=normalized_local_weighting_mode,
            node_scales=leaf_node_scales,
        )
        leaf_first_doc_loss, _, _, _ = _masked_doc_local_means(
            leaf_first_loss,
            leaf_mask,
            leaf_propensity,
            weighting_mode=normalized_local_weighting_mode,
            node_scales=leaf_node_scales,
        )
        leaf_last_doc_loss, _, _, _ = _masked_doc_local_means(
            leaf_last_loss,
            leaf_mask,
            leaf_propensity,
            weighting_mode=normalized_local_weighting_mode,
            node_scales=leaf_node_scales,
        )
        if bool(leaf_active.any()):
            leaf_loss_weight = float(c1_weight)
            if normalized_local_weighting_mode != "span_mass_ipw_sum":
                leaf_loss_weight *= float(_leaf_depth_discount)
            non_document_loss_sum = (
                non_document_loss_sum
                + float(leaf_loss_weight) * leaf_doc_loss[leaf_active].sum()
            )
            component_sums["leaf_count_loss"] = leaf_count_doc_loss[leaf_active].sum().to(dtype=torch.float32)
            component_sums["leaf_first_loss"] = leaf_first_doc_loss[leaf_active].sum().to(dtype=torch.float32)
            component_sums["leaf_last_loss"] = leaf_last_doc_loss[leaf_active].sum().to(dtype=torch.float32)
            active_count = int(leaf_active.sum().item())
            component_counts["leaf_count_loss"] = active_count
            component_counts["leaf_first_loss"] = active_count
            component_counts["leaf_last_loss"] = active_count
    else:
        leaf_doc_numerators = torch.zeros((batch_size,), device=device, dtype=root_state_batch.dtype)
        leaf_doc_denominators = torch.zeros((batch_size,), device=device, dtype=root_state_batch.dtype)

    if bool(merge_supervision_active):
        flat_merge_states_real = merge_state_batch.reshape(int(batch_size * n_merge), -1)
        flat_merge_theorem_batch = (
            merge_theorem_batch.reshape(batch_size * n_merge, -1)
            if merge_theorem_batch is not None
            else None
        )
        if normalized_internal_supervision_kind == "full_sketch":
            merge_task_terms = _theorem_feature_task_supervision_terms_batched(
                model,
                flat_merge_states_real,
                truth_targets=merge_truth_counts.reshape(-1),
                theorem_feature_batch=flat_merge_theorem_batch,
            )
            merge_task_loss = merge_task_terms["task_loss"].reshape(batch_size, n_merge)
            merge_total_loss = merge_task_loss
            merge_count_loss = merge_task_loss
            merge_first_loss = torch.zeros_like(merge_task_loss)
            merge_last_loss = torch.zeros_like(merge_task_loss)
            if bool(model.use_summary_spec):
                merge_summary_terms = _local_supervision_terms_batched(
                    model,
                    flat_merge_states_real,
                    truth_counts=merge_truth_counts.reshape(-1),
                    truth_first=merge_truth_first.reshape(-1),
                    truth_last=merge_truth_last.reshape(-1),
                    supervision_kind="full_sketch",
                    theorem_feature_batch=flat_merge_theorem_batch,
                )
                merge_total_loss = (
                    merge_total_loss
                    + merge_summary_terms["total_loss"].reshape(batch_size, n_merge)
                )
                merge_count_loss = (
                    merge_count_loss
                    + merge_summary_terms["count_loss"].reshape(batch_size, n_merge)
                )
                merge_first_loss = merge_summary_terms["first_loss"].reshape(batch_size, n_merge)
                merge_last_loss = merge_summary_terms["last_loss"].reshape(batch_size, n_merge)
        else:
            merge_terms = _local_supervision_terms_batched(
                model,
                flat_merge_states_real,
                truth_counts=merge_truth_counts.reshape(-1),
                truth_first=merge_truth_first.reshape(-1),
                truth_last=merge_truth_last.reshape(-1),
                supervision_kind=str(internal_supervision_kind),
                theorem_feature_batch=flat_merge_theorem_batch,
            )
            merge_total_loss = merge_terms["total_loss"].reshape(batch_size, n_merge)
            merge_count_loss = merge_terms["count_loss"].reshape(batch_size, n_merge)
            merge_first_loss = merge_terms["first_loss"].reshape(batch_size, n_merge)
            merge_last_loss = merge_terms["last_loss"].reshape(batch_size, n_merge)
        if normalized_local_weighting_mode != "span_mass_ipw_sum":
            # Legacy modes keep the historical loss-side depth discount.
            _mdd = _merge_depth_discount.unsqueeze(0)
            merge_total_loss = merge_total_loss * _mdd
            merge_count_loss = merge_count_loss * _mdd
            merge_first_loss = merge_first_loss * _mdd
            merge_last_loss = merge_last_loss * _mdd
        merge_doc_loss, merge_active, merge_doc_numerators, merge_doc_denominators = _masked_doc_local_means(
            merge_total_loss,
            merge_mask,
            merge_propensity,
            weighting_mode=normalized_local_weighting_mode,
            node_scales=merge_node_scales,
        )
        merge_count_doc_loss, _, _, _ = _masked_doc_local_means(
            merge_count_loss,
            merge_mask,
            merge_propensity,
            weighting_mode=normalized_local_weighting_mode,
            node_scales=merge_node_scales,
        )
        merge_first_doc_loss, _, _, _ = _masked_doc_local_means(
            merge_first_loss,
            merge_mask,
            merge_propensity,
            weighting_mode=normalized_local_weighting_mode,
            node_scales=merge_node_scales,
        )
        merge_last_doc_loss, _, _, _ = _masked_doc_local_means(
            merge_last_loss,
            merge_mask,
            merge_propensity,
            weighting_mode=normalized_local_weighting_mode,
            node_scales=merge_node_scales,
        )
        if bool(merge_active.any()):
            non_document_loss_sum = (
                non_document_loss_sum
                + float(c3_weight) * merge_doc_loss[merge_active].sum()
            )
            component_sums["merge_count_loss"] = merge_count_doc_loss[merge_active].sum().to(dtype=torch.float32)
            component_sums["merge_first_loss"] = merge_first_doc_loss[merge_active].sum().to(dtype=torch.float32)
            component_sums["merge_last_loss"] = merge_last_doc_loss[merge_active].sum().to(dtype=torch.float32)
            active_count = int(merge_active.sum().item())
            component_counts["merge_count_loss"] = active_count
            component_counts["merge_first_loss"] = active_count
            component_counts["merge_last_loss"] = active_count
    else:
        merge_doc_numerators = torch.zeros((batch_size,), device=device, dtype=root_state_batch.dtype)
        merge_doc_denominators = torch.zeros((batch_size,), device=device, dtype=root_state_batch.dtype)

    leaf_phi_batch: torch.Tensor | None = None
    merge_phi_batch: torch.Tensor | None = None
    root_phi_batch: torch.Tensor | None = None
    if needs_node_phi:
        if leaf_theorem_batch is None:
            raise RuntimeError("leaf theorem features are required for node-phi batching")
        leaf_phi_batch = model.predict_phi_from_theorem_feature(leaf_theorem_batch)
        need_merge_phi = bool(int(n_merge) > 0 and (dense_full_fast or not dense_leaf_root_fast))
        merge_phi_batch = (
            model.predict_phi_from_theorem_feature(merge_theorem_batch)
            if need_merge_phi and merge_theorem_batch is not None
            else leaf_phi_batch.new_zeros((batch_size, 0, int(leaf_phi_batch.shape[-1])))
        )
        if root_theorem_batch is None:
            raise RuntimeError("root theorem features are required for node-phi batching")
        root_phi_batch = model.predict_phi_from_theorem_feature(root_theorem_batch)

        if dense_markov_c2_fast:
            fast_c2_tensors = _fixed_fused_fast_markov_c2_tensors(
                leaf_phi_batch=leaf_phi_batch,
                merge_phi_batch=merge_phi_batch,
                root_phi_batch=root_phi_batch,
                leaf_valid_mask=leaf_valid_mask,
                merge_valid_mask=merge_valid_mask,
                node_count_targets=node_count_targets,
                node_count_keys=node_count_keys,
                node_first_targets=node_first_targets,
                node_last_targets=node_last_targets,
                leaf_node_scales=leaf_node_scales,
                merge_node_scales=merge_node_scales,
                include_merge=bool(dense_full_fast and int(n_merge) > 0),
            )
            node_phi_batch = fast_c2_tensors["phi_batch"]
            same_mask, different_mask = _fast_markov_pair_masks_from_tensors(
                model,
                count_keys=fast_c2_tensors["count_keys"],
                first_targets=fast_c2_tensors["first_targets"],
                last_targets=fast_c2_tensors["last_targets"],
            )
            valid_pairs = fast_c2_tensors["valid_mask"].unsqueeze(-1) & fast_c2_tensors["valid_mask"].unsqueeze(-2)
            same_mask = same_mask & valid_pairs
            different_mask = different_mask & valid_pairs
            pair_weights = None
            if c2_pair_weighting_mode == "pair_ipw_geomean":
                pair_weight_rows: List[torch.Tensor] = []
                for batch_idx, work in enumerate(works):
                    pair_weight_rows.append(
                        _c2_pair_weight_matrix(
                            node_scales=fast_c2_tensors["node_scales"][int(batch_idx)],
                            node_kind_codes=fast_c2_tensors["node_kind_codes"][int(batch_idx)],
                            valid_mask=fast_c2_tensors["valid_mask"][int(batch_idx)],
                            leaf_population_size=int(
                                work.get(
                                    "leaf_supervision_population_size",
                                    len(work["doc"].leaf_token_ids),
                                )
                            ),
                            leaf_sample_size=_effective_subset_sample_size(
                                int(
                                    work.get(
                                        "leaf_supervision_population_size",
                                        len(work["doc"].leaf_token_ids),
                                    )
                                ),
                                work.get("leaf_audit_indices"),
                            ),
                            merge_population_size=int(
                                work.get("internal_supervision_population_size", 0)
                            ),
                            merge_sample_size=_effective_subset_sample_size(
                                int(work.get("internal_supervision_population_size", 0)),
                                work.get("c3_audit_indices"),
                            ),
                        )
                    )
                pair_weights = torch.stack(pair_weight_rows, dim=0)
            c2_batched_out = _batched_pairwise_theorem_feature_contrastive_loss_from_masks(
                node_phi_batch,
                same_mask=same_mask,
                different_mask=different_mask,
                pair_weights=pair_weights,
                return_diagnostics=True,
            )
            if not isinstance(c2_batched_out, Mapping):
                raise RuntimeError("expected batched C2 diagnostics mapping")
            c2_doc_loss = c2_batched_out["loss"]
            c2_pair_same_count = c2_batched_out["same_pair_count"].to(dtype=root_state_batch.dtype)
            c2_pair_different_count = c2_batched_out["different_pair_count"].to(dtype=root_state_batch.dtype)
            c2_pair_weight_ess = c2_batched_out["pair_weight_ess"].to(dtype=root_state_batch.dtype)
            c2_pair_weight_max = c2_batched_out["pair_weight_max"].to(dtype=root_state_batch.dtype)
            if bool(c2_collect_mask.any()):
                non_document_loss_sum = (
                    non_document_loss_sum
                    + float(c2_weight) * c2_doc_loss[c2_collect_mask].sum()
                )
                component_sums["c2_count_loss"] = c2_doc_loss[c2_collect_mask].sum().to(dtype=torch.float32)
                component_counts["c2_count_loss"] = int(c2_collect_mask.sum().item())
            if bool(defer_contrastive):
                flat_valid_mask = fast_c2_tensors["valid_mask"].reshape(-1)
                deferred_phi_feature_batch = node_phi_batch.reshape(
                    int(batch_size * node_phi_batch.shape[1]),
                    int(node_phi_batch.shape[-1]),
                )[flat_valid_mask]
                deferred_phi_fast_keys = {
                    "count_values": fast_c2_tensors["count_values"].reshape(-1)[flat_valid_mask],
                    "count_keys": fast_c2_tensors["count_keys"].reshape(-1)[flat_valid_mask],
                    "first_targets": fast_c2_tensors["first_targets"].reshape(-1)[flat_valid_mask],
                    "last_targets": fast_c2_tensors["last_targets"].reshape(-1)[flat_valid_mask],
                }
        else:
            c2_losses: List[torch.Tensor] = []
            for batch_idx, item in enumerate(items):
                work = work_lookup[int(item.doc_index)]
                feature_targets = feature_targets_by_doc[int(batch_idx)]
                padded_merge_targets = _padded_merge_feature_targets_for_valid_mask(
                    feature_targets.merge,
                    merge_valid_mask[int(batch_idx)],
                )
                phi_features: List[torch.Tensor] = []
                phi_labels: List[Any] = []
                phi_oracle_vecs: List[Any] = []
                phi_kind_codes: List[int] = []
                phi_node_scales: List[float] = []

                def _append_phi_feature(
                    phi_value: torch.Tensor,
                    target: Any,
                    *,
                    node_kind: int,
                    node_scale: float,
                ) -> None:
                    phi_features.append(phi_value)
                    phi_labels.append(target)
                    phi_kind_codes.append(int(node_kind))
                    phi_node_scales.append(float(node_scale))
                    if model.oracle_metric is not None:
                        phi_oracle_vecs.append(
                            model.oracle_metric.oracle_vector(
                                count=float(target.count),
                                first=int(target.first),
                                last=int(target.last),
                            )
                        )

                for leaf_idx in _all_or_sampled_indices(int(n_leaf), work["leaf_audit_indices"]):
                    if int(leaf_idx) < len(feature_targets.leaf):
                        _append_phi_feature(
                            leaf_phi_batch[int(batch_idx), int(leaf_idx)],
                            feature_targets.leaf[int(leaf_idx)],
                            node_kind=_C2_NODE_KIND_LEAF,
                            node_scale=float(
                                leaf_node_scales[int(batch_idx), int(leaf_idx)].detach().item()
                            ),
                        )
                for merge_idx in _all_or_sampled_indices(int(n_merge), work["c3_audit_indices"]):
                    if int(merge_idx) < len(padded_merge_targets):
                        merge_target = padded_merge_targets[int(merge_idx)]
                        if merge_target is None:
                            continue
                        _append_phi_feature(
                            merge_phi_batch[int(batch_idx), int(merge_idx)],
                            merge_target,
                            node_kind=_C2_NODE_KIND_MERGE,
                            node_scale=float(
                                merge_node_scales[int(batch_idx), int(merge_idx)].detach().item()
                            ),
                        )
                if feature_targets.root:
                    _append_phi_feature(
                        root_phi_batch[int(batch_idx)],
                        feature_targets.root[0],
                        node_kind=_C2_NODE_KIND_ROOT,
                        node_scale=1.0,
                    )

                if bool(work["collect_c2"]) and len(phi_features) > 1:
                    if (
                        c2_pair_weighting_mode == "pair_ipw_geomean"
                        and model.oracle_metric is not None
                    ):
                        raise ValueError(
                            "authoritative C2 pair weighting does not support oracle_metric mode"
                        )
                    if model.oracle_metric is not None:
                        pair_data = build_contrastive_pairs(
                            phi_oracle_vecs,
                            metric=model.oracle_metric,
                            same_threshold=model.oracle_same_threshold,
                            diff_threshold=model.oracle_diff_threshold,
                        )
                        c2_doc_loss = contrastive_fiber_loss(
                            torch.stack(phi_features, dim=0),
                            pair_data,
                            margin=float(SUMMARY_SPEC_PHI_DIFFERENT_MARGIN),
                        )
                    else:
                        pair_sets = build_theorem_feature_pair_sets(
                            phi_labels,
                            adapter=model.theorem_feature_adapter,
                            same_threshold=model.theorem_pair_same_threshold,
                            diff_threshold=model.theorem_pair_diff_threshold,
                        )
                        same_pair_weights = None
                        different_pair_weights = None
                        pair_weights = None
                        if c2_pair_weighting_mode == "pair_ipw_geomean":
                            pair_weights = _c2_pair_weight_matrix(
                                node_scales=torch.tensor(
                                    phi_node_scales,
                                    device=device,
                                    dtype=root_state_batch.dtype,
                                ),
                                node_kind_codes=torch.tensor(
                                    phi_kind_codes,
                                    device=device,
                                    dtype=torch.long,
                                ),
                                valid_mask=torch.ones(
                                    (len(phi_features),),
                                    device=device,
                                    dtype=torch.bool,
                                ),
                                leaf_population_size=int(
                                    work.get(
                                        "leaf_supervision_population_size",
                                        len(work["doc"].leaf_token_ids),
                                    )
                                ),
                                leaf_sample_size=_effective_subset_sample_size(
                                    int(
                                        work.get(
                                            "leaf_supervision_population_size",
                                            len(work["doc"].leaf_token_ids),
                                        )
                                    ),
                                    work.get("leaf_audit_indices"),
                                ),
                                merge_population_size=int(
                                    work.get("internal_supervision_population_size", 0)
                                ),
                                merge_sample_size=_effective_subset_sample_size(
                                    int(work.get("internal_supervision_population_size", 0)),
                                    work.get("c3_audit_indices"),
                                ),
                            )
                            same_pair_weights = [
                                float(pair_weights[int(left_idx), int(right_idx)].detach().item())
                                for left_idx, right_idx in list(pair_sets.same_pairs)
                            ]
                            different_pair_weights = [
                                float(pair_weights[int(left_idx), int(right_idx)].detach().item())
                                for left_idx, right_idx in list(pair_sets.different_pairs)
                            ]
                        c2_doc_loss = _pairwise_theorem_feature_contrastive_loss(
                            torch.stack(phi_features, dim=0),
                            same_pairs=pair_sets.same_pairs,
                            different_pairs=pair_sets.different_pairs,
                            same_pair_weights=same_pair_weights,
                            different_pair_weights=different_pair_weights,
                        )
                        same_pair_mask = torch.zeros(
                            (len(phi_features), len(phi_features)),
                            device=device,
                            dtype=torch.bool,
                        )
                        different_pair_mask = torch.zeros_like(same_pair_mask)
                        for left_idx, right_idx in list(pair_sets.same_pairs):
                            same_pair_mask[int(left_idx), int(right_idx)] = True
                            same_pair_mask[int(right_idx), int(left_idx)] = True
                        for left_idx, right_idx in list(pair_sets.different_pairs):
                            different_pair_mask[int(left_idx), int(right_idx)] = True
                            different_pair_mask[int(right_idx), int(left_idx)] = True
                        pair_diag = _pairwise_weight_diagnostics_from_masks(
                            same_mask=same_pair_mask,
                            different_mask=different_pair_mask,
                            pair_weights=pair_weights,
                        )
                        c2_pair_same_count[int(batch_idx)] = pair_diag["same_pair_count"].to(dtype=root_state_batch.dtype)
                        c2_pair_different_count[int(batch_idx)] = pair_diag["different_pair_count"].to(dtype=root_state_batch.dtype)
                        c2_pair_weight_ess[int(batch_idx)] = pair_diag["pair_weight_ess"].to(dtype=root_state_batch.dtype)
                        c2_pair_weight_max[int(batch_idx)] = pair_diag["pair_weight_max"].to(dtype=root_state_batch.dtype)
                    c2_losses.append(c2_doc_loss)

                if len(phi_features) > 1 and bool(defer_contrastive):
                    deferred_phi_features.extend(phi_features)
                    if model.oracle_metric is not None:
                        deferred_oracle_vecs.extend(phi_oracle_vecs)
                    else:
                        deferred_phi_labels.extend(phi_labels)

            if c2_losses:
                c2_stack = torch.stack(c2_losses, dim=0)
                non_document_loss_sum = (
                    non_document_loss_sum + float(c2_weight) * c2_stack.sum()
                )
                component_sums["c2_count_loss"] = c2_stack.sum().to(dtype=torch.float32)
                component_counts["c2_count_loss"] = int(c2_stack.shape[0])

    phi_compose_doc_loss = torch.zeros((batch_size,), device=device, dtype=root_state_batch.dtype)
    if bool(need_phi_compose):
        if leaf_theorem_batch is None:
            raise RuntimeError("leaf theorem features are required for phi compose batching")
        cur_theorem = leaf_theorem_batch
        merge_feature_offset = 0
        phi_level_losses: List[torch.Tensor] = []
        phi_level_masks: List[torch.Tensor] = []
        for merge_level_valid_mask in levels.merge_valid_levels:
            pair_count = int(merge_level_valid_mask.shape[1])
            if pair_count <= 0:
                continue
            if merge_theorem_batch is None:
                raise RuntimeError("merge theorem features are required for phi compose batching")
            merged_theorem_level = merge_theorem_batch[
                :,
                merge_feature_offset : merge_feature_offset + pair_count,
                :,
            ]
            merge_feature_offset += pair_count
            pred_phi = model.predict_phi_parent_from_theorem_features(
                cur_theorem[:, 0 : 2 * pair_count : 2, :],
                cur_theorem[:, 1 : 2 * pair_count : 2, :],
            )
            target_phi = model.predict_phi_from_theorem_feature(merged_theorem_level)
            phi_level_losses.append(
                _phi_alignment_loss_per_item(
                    pred_phi,
                    target_phi,
                    mode=model.phi_alignment_loss,
                )
            )
            phi_level_masks.append(merge_level_valid_mask)
            if int(cur_theorem.shape[1]) % 2 == 1:
                cur_theorem = torch.cat(
                    [merged_theorem_level, cur_theorem[:, -1:, :]],
                    dim=1,
                )
            else:
                cur_theorem = merged_theorem_level
        if phi_level_losses:
            phi_compose_doc_loss, _ = _masked_doc_means(
                torch.cat(phi_level_losses, dim=1),
                torch.cat(phi_level_masks, dim=1),
            )
    non_document_loss_sum = (
        non_document_loss_sum + float(phi_compose_weight) * phi_compose_doc_loss.sum()
    )
    component_sums["phi_compose_loss"] = phi_compose_doc_loss.sum().to(dtype=torch.float32)
    component_counts["phi_compose_loss"] = int(batch_size)
    component_counts["phi_contrastive_loss"] = int(batch_size)
    batch_loss = document_loss_sum + non_document_loss_sum
    local_objective_audit_rows: List[Dict[str, Any]] = []
    for batch_idx, item in enumerate(items[: min(3, len(items))]):
        leaf_population_size, leaf_sample_size, leaf_effective_propensity = (
            _effective_local_sampling_summary(
                int(
                    works[int(batch_idx)].get(
                        "leaf_supervision_population_size",
                        len(item.doc.leaf_token_ids),
                    )
                ),
                works[int(batch_idx)].get("leaf_audit_indices"),
            )
        )
        merge_population_size, merge_sample_size, merge_effective_propensity = (
            _effective_local_sampling_summary(
                int(
                    works[int(batch_idx)].get(
                        "internal_supervision_population_size",
                        max(0, len(item.doc.leaf_token_ids) - 1),
                    )
                ),
                works[int(batch_idx)].get("c3_audit_indices"),
            )
        )
        local_objective_audit_rows.append(
            {
                "doc_index": int(item.doc_index),
                "leaf": {
                    "population_size": int(leaf_population_size),
                    "sample_size": int(leaf_sample_size),
                    "effective_propensity": float(leaf_effective_propensity),
                    "numerator": float(leaf_doc_numerators[int(batch_idx)].detach().cpu()),
                    "denominator": float(
                        leaf_doc_denominators[int(batch_idx)].detach().cpu()
                    ),
                    "implemented_loss": float(leaf_doc_loss[int(batch_idx)].detach().cpu())
                    if bool(leaf_supervision_active)
                    else 0.0,
                },
                "merge": {
                    "population_size": int(merge_population_size),
                    "sample_size": int(merge_sample_size),
                    "effective_propensity": float(merge_effective_propensity),
                    "numerator": float(merge_doc_numerators[int(batch_idx)].detach().cpu()),
                    "denominator": float(
                        merge_doc_denominators[int(batch_idx)].detach().cpu()
                    ),
                    "implemented_loss": float(merge_doc_loss[int(batch_idx)].detach().cpu())
                    if bool(merge_supervision_active)
                    else 0.0,
                },
                "c2": {
                    "pair_weighting_mode": str(c2_pair_weighting_mode),
                    "same_pair_count": float(
                        c2_pair_same_count[int(batch_idx)].detach().cpu()
                    ),
                    "different_pair_count": float(
                        c2_pair_different_count[int(batch_idx)].detach().cpu()
                    ),
                    "pair_weight_ess": float(
                        c2_pair_weight_ess[int(batch_idx)].detach().cpu()
                    ),
                    "pair_weight_max": float(
                        c2_pair_weight_max[int(batch_idx)].detach().cpu()
                    ),
                },
            }
        )

    return {
        "batch_loss": batch_loss,
        "document_loss_sum": document_loss_sum,
        "non_document_loss_sum": non_document_loss_sum,
        "component_sums": component_sums,
        "component_counts": component_counts,
        "deferred_phi_features": deferred_phi_features,
        "deferred_phi_labels": deferred_phi_labels,
        "deferred_oracle_vecs": deferred_oracle_vecs,
        "deferred_phi_feature_batch": deferred_phi_feature_batch,
        "deferred_phi_fast_keys": deferred_phi_fast_keys,
        "tree_local_weighting_mode": normalized_local_weighting_mode,
        "tree_supervision_source": normalized_supervision_source,
        "local_estimand_mode": normalized_local_weighting_mode,
        "local_loss_kind": local_loss_kind,
        "c2_pair_weighting_mode": str(c2_pair_weighting_mode),
        "c2_same_pair_count": float(c2_pair_same_count.mean().detach().cpu()),
        "c2_different_pair_count": float(c2_pair_different_count.mean().detach().cpu()),
        "c2_pair_weight_ess": float(c2_pair_weight_ess.mean().detach().cpu()),
        "c2_pair_weight_max": float(c2_pair_weight_max.max().detach().cpu()),
        "local_sampling_design_name": (
            "manifest_explicit_deterministic_ordering"
            if normalized_supervision_source == "manifest"
            else "deterministic_fixed_k_uniform"
        ),
        "local_objective_audit_rows": local_objective_audit_rows,
    }


def _maybe_enable_cuda_fast_math(device: torch.device) -> None:
    global _CUDA_FAST_MATH_CONFIGURED
    if str(device.type) != "cuda" or _CUDA_FAST_MATH_CONFIGURED:
        return
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    _CUDA_FAST_MATH_CONFIGURED = True


def _use_cuda_bf16_autocast(device: torch.device) -> bool:
    # neuraloperator's FNO stack currently reaches torch.fft.rfftn on the hot path,
    # and this environment rejects bf16 there. Keep the helper so we can re-enable
    # selective mixed precision later without changing call sites.
    return False


def _autocast_context(device: torch.device):
    if not _use_cuda_bf16_autocast(device):
        if str(device.type) == "cuda":
            _maybe_enable_cuda_fast_math(device)
        return nullcontext()
    _maybe_enable_cuda_fast_math(device)
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.cuda.amp.autocast(dtype=torch.bfloat16)


@torch.inference_mode()
def _encode_leaf_state_batch(
    model: FNOCountSketch,
    token_batches: Sequence[Sequence[int]],
    *,
    device: torch.device,
) -> List[torch.Tensor]:
    token_lists = [list(token_ids) for token_ids in list(token_batches)]
    if not token_lists:
        return []
    with _autocast_context(device):
        state_batch = model.encode_leaf_tokens_batch(token_lists, device=device)
    batch_size = int(state_batch.shape[0])
    return [state_batch[idx] for idx in range(batch_size)]


def _precompute_balanced_doc_state_views(
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
    collect_merge_states: bool,
    prefer_fixed_fused: bool = False,
    target_n_leaves: int | None = None,
    resident_view: GpuBatchView | None = None,
    runtime_telemetry: GpuRuntimeTelemetry | None = None,
) -> List[_PrecomputedDocStateView]:
    if not docs:
        return []
    if len({int(len(doc.leaf_token_ids)) for doc in docs}) > 1:
        views: List[_PrecomputedDocStateView] = []
        for doc in docs:
            views.extend(
                _precompute_balanced_doc_state_views(
                    model,
                    [doc],
                    device=device,
                    collect_merge_states=collect_merge_states,
                    prefer_fixed_fused=False,
                    target_n_leaves=None,
                    resident_view=None,
                    runtime_telemetry=runtime_telemetry,
                )
            )
        return views
    levels = _precompute_balanced_doc_state_levels(
        model,
        docs,
        device=device,
        collect_merge_states=collect_merge_states,
        prefer_fixed_fused=bool(prefer_fixed_fused),
        target_n_leaves=target_n_leaves,
        resident_view=resident_view,
        runtime_telemetry=runtime_telemetry,
    )
    leaf_states = levels.leaf_states
    merge_levels = list(levels.merge_levels)
    root_states = levels.root_states

    views: List[_PrecomputedDocStateView] = []
    for doc_idx in range(int(leaf_states.shape[0])):
        merge_states: List[torch.Tensor] = []
        if collect_merge_states:
            for level, valid_mask in zip(merge_levels, levels.merge_valid_levels):
                for merge_idx in range(int(level.shape[1])):
                    if bool(valid_mask[doc_idx, merge_idx].item()):
                        merge_states.append(level[doc_idx, merge_idx])
        views.append(
            _PrecomputedDocStateView(
                state_batch=leaf_states[doc_idx],
                root_state=root_states[doc_idx],
                merge_states=tuple(merge_states),
            )
        )
    return views


def _probe_tree_docs_cap_for_representative(
    model: FNOCountSketch,
    doc: _FNOCountDoc,
    *,
    device: torch.device,
    training: bool,
    max_candidate_docs: int,
    pack_mode: str = "structure_bucket",
    heuristic_docs_cap: int = 0,
    probe_cache: ProbeCacheStore | None = None,
    model_signature: Mapping[str, Any] | None = None,
    device_class_signature: Mapping[str, Any] | None = None,
    topology_signature: str = "",
) -> _ProbeRunOutcome:
    probe_mode = "train" if bool(training) else "eval"
    normalized_pack_mode = str(pack_mode or "structure_bucket").strip().lower()
    topology_sig = str(topology_signature or _tree_batch_probe_topology_signature(doc))
    model_sig = dict(model_signature or _tree_batch_probe_model_signature(model))
    device_sig = dict(device_class_signature or _tree_batch_probe_device_signature(device))
    cache_key = build_probe_cache_key(
        model_signature=model_sig,
        pack_mode=normalized_pack_mode,
        topology_signature=topology_sig,
        probe_mode=probe_mode,
        device_class_signature=device_sig,
    )
    if device.type != "cuda" or not torch.cuda.is_available():
        run_profile = ProbeRunProfile(
            probe_mode=probe_mode,
            topology_signature=topology_sig,
            selected_docs_cap=int(max(1, max_candidate_docs)),
            heuristic_docs_cap=int(max(1, heuristic_docs_cap or max_candidate_docs)),
            max_candidate_docs=int(max(1, max_candidate_docs)),
            target_fraction=0.0,
            cache_key=str(cache_key),
            cache_hit=False,
            total_wall_time_s=0.0,
            stop_reason="cuda_unavailable",
            candidate_profiles=tuple(),
        )
        return _ProbeRunOutcome(
            selected_docs_cap=int(max(1, max_candidate_docs)),
            run_profile=run_profile,
        )
    try:
        total_mem = float(torch.cuda.get_device_properties(device).total_memory)
    except Exception:
        total_mem = 0.0
    if total_mem <= 0.0:
        run_profile = ProbeRunProfile(
            probe_mode=probe_mode,
            topology_signature=topology_sig,
            selected_docs_cap=int(max(1, max_candidate_docs)),
            heuristic_docs_cap=int(max(1, heuristic_docs_cap or max_candidate_docs)),
            max_candidate_docs=int(max(1, max_candidate_docs)),
            target_fraction=0.0,
            cache_key=str(cache_key),
            cache_hit=False,
            total_wall_time_s=0.0,
            stop_reason="missing_total_memory",
            candidate_profiles=tuple(),
        )
        return _ProbeRunOutcome(
            selected_docs_cap=int(max(1, max_candidate_docs)),
            run_profile=run_profile,
        )
    fixed_fused_preferred = bool(
        normalized_pack_mode == "fixed_fused"
        and bool(getattr(model, "use_shared_theorem_surface", False))
    )
    cache_lookup_time_s = 0.0
    if probe_cache is not None:
        lookup_start_s = time.perf_counter()
        cached_entry = probe_cache.get(cache_key)
        cache_lookup_time_s = time.perf_counter() - lookup_start_s
        if cached_entry is not None and int(cached_entry.selected_docs_cap) > 0:
            cached_profile = replace(
                cached_entry.run_profile,
                cache_hit=True,
                total_wall_time_s=float(cache_lookup_time_s),
                cached_source_wall_time_s=float(
                    cached_entry.run_profile.total_wall_time_s
                ),
                stop_reason=f"cache_hit:{str(cached_entry.run_profile.stop_reason or 'selected')}",
            )
            return _ProbeRunOutcome(
                selected_docs_cap=int(cached_entry.selected_docs_cap),
                run_profile=cached_profile,
                cache_hit=True,
                cache_lookup_time_s=float(cache_lookup_time_s),
                cache_write_time_s=0.0,
                candidate_evaluations=0,
            )
    candidates = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256]
    if fixed_fused_preferred:
        candidates.extend([384, 512, 768, 1024, 1536, 2048])
    candidates = [
        int(value)
        for value in candidates
        if int(value) <= max(1, int(max_candidate_docs))
    ] or [1]
    best = 1
    target_fraction = (
        0.78
        if bool(training) and fixed_fused_preferred
        else (0.55 if fixed_fused_preferred else (0.66 if bool(training) else 0.42))
    )
    total_start_s = time.perf_counter()
    candidate_profiles: List[ProbeCandidateProfile] = []
    run_stop_reason = "candidate_limit_reached"
    for candidate_docs in candidates:
        repeated_docs = [doc] * int(candidate_docs)
        prefer_fixed_fused = bool(
            fixed_fused_preferred
            and _docs_share_fixed_leaf_shape(repeated_docs)
        )
        pack_time_s = 0.0
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model.zero_grad(set_to_none=True)
            probe_start_s = time.perf_counter()
            if training:
                model.train()
                if prefer_fixed_fused:
                    pack_start_s = time.perf_counter()
                    packed_items = [
                        _tree_work_item_from_doc(
                            doc,
                            doc_index=int(idx),
                            work_kind="full_tree",
                            collect_leaf=True,
                            collect_c2=True,
                            collect_c3=True,
                            root_only_supervision=True,
                        )
                        for idx in range(int(candidate_docs))
                    ]
                    packed_batch = _pack_tree_work_items(
                        packed_items,
                        max_docs=int(candidate_docs),
                        max_total_leaf_tokens=0,
                        max_total_nodes=0,
                        max_total_merge_ops=0,
                        bucket_docs_cap_by_n_leaves=None,
                    )[0]
                    work_lookup = {
                        int(idx): {
                            "doc": doc,
                            "root_only_supervision": True,
                            "doc_sequence_supervision": False,
                            "doc_sequence_loss": torch.zeros(
                                (),
                                device=device,
                                dtype=torch.float32,
                            ),
                            "collect_leaf": True,
                            "collect_c2": True,
                            "collect_c3": True,
                            "leaf_audit_indices": None,
                            "c3_audit_indices": None,
                        }
                        for idx in range(int(candidate_docs))
                    }
                    pack_time_s = time.perf_counter() - pack_start_s
                    with _autocast_context(device):
                        probe_loss = _fixed_fused_training_batch_forward(
                            model,
                            packed_batch,
                            work_lookup=work_lookup,
                            device=device,
                            root_weight=1.0,
                            c1_weight=1.0,
                            c2_weight=1.0,
                            c3_weight=1.0,
                            phi_compose_weight=1.0,
                            leaf_supervision_kind="full_sketch",
                            internal_supervision_kind=(
                                "full_sketch"
                                if bool(getattr(model, "use_summary_spec", False))
                                else "none"
                            ),
                            tree_local_weighting_mode="fixed_k_hajek",
                            defer_contrastive=bool(
                                getattr(model, "use_shared_theorem_surface", False)
                            ),
                        )["batch_loss"]
                else:
                    views = _precompute_balanced_doc_state_views(
                        model,
                        repeated_docs,
                        device=device,
                        collect_merge_states=True,
                        prefer_fixed_fused=False,
                    )
                    root_states = torch.stack([view.root_state for view in views], dim=0)
                    with _autocast_context(device):
                        pred = model.predict_task_count_from_state(root_states)
                        probe_loss = pred.reshape(-1).sum()
                if bool(getattr(probe_loss, "requires_grad", False)):
                    probe_loss.backward()
                model.zero_grad(set_to_none=True)
            else:
                model.eval()
                with torch.inference_mode():
                    if prefer_fixed_fused:
                        root_states = _precompute_balanced_doc_state_levels(
                            model,
                            repeated_docs,
                            device=device,
                            collect_merge_states=False,
                            prefer_fixed_fused=True,
                        ).root_states
                    else:
                        views = _precompute_balanced_doc_state_views(
                            model,
                            repeated_docs,
                            device=device,
                            collect_merge_states=False,
                            prefer_fixed_fused=False,
                        )
                        root_states = torch.stack(
                            [view.root_state for view in views],
                            dim=0,
                        )
                    with _autocast_context(device):
                        _ = model.predict_canonical_count_from_state(root_states).reshape(-1).sum()
            torch.cuda.synchronize(device)
            probe_wall_time_s = time.perf_counter() - probe_start_s
            peak_reserved = float(torch.cuda.max_memory_reserved(device=device))
            peak_allocated = float(torch.cuda.max_memory_allocated(device=device))
            if peak_reserved / total_mem > float(target_fraction):
                candidate_profiles.append(
                    ProbeCandidateProfile(
                        candidate_docs=int(candidate_docs),
                        pack_time_s=float(pack_time_s),
                        forward_backward_time_s=float(probe_wall_time_s),
                        peak_reserved_gb=float(peak_reserved / float(1024 ** 3)),
                        peak_allocated_gb=float(peak_allocated / float(1024 ** 3)),
                        stop_reason="target_fraction_exceeded",
                    )
                )
                run_stop_reason = "target_fraction_exceeded"
                break
            best = int(candidate_docs)
            candidate_profiles.append(
                ProbeCandidateProfile(
                    candidate_docs=int(candidate_docs),
                    pack_time_s=float(pack_time_s),
                    forward_backward_time_s=float(probe_wall_time_s),
                    peak_reserved_gb=float(peak_reserved / float(1024 ** 3)),
                    peak_allocated_gb=float(peak_allocated / float(1024 ** 3)),
                    stop_reason="accepted",
                )
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            peak_reserved = 0.0
            peak_allocated = 0.0
            try:
                peak_reserved = float(torch.cuda.max_memory_reserved(device=device))
                peak_allocated = float(torch.cuda.max_memory_allocated(device=device))
            except Exception:
                pass
            try:
                model.zero_grad(set_to_none=True)
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            candidate_profiles.append(
                ProbeCandidateProfile(
                    candidate_docs=int(candidate_docs),
                    pack_time_s=float(pack_time_s),
                    forward_backward_time_s=float(
                        max(0.0, time.perf_counter() - probe_start_s)
                    ),
                    peak_reserved_gb=float(peak_reserved / float(1024 ** 3)),
                    peak_allocated_gb=float(peak_allocated / float(1024 ** 3)),
                    stop_reason="oom",
                )
            )
            run_stop_reason = "oom"
            break
    if training:
        best = (
            max(1, int(best * 0.85))
            if fixed_fused_preferred
            else max(1, int(best // 2))
        )
    total_wall_time_s = time.perf_counter() - total_start_s
    run_profile = ProbeRunProfile(
        probe_mode=probe_mode,
        topology_signature=topology_sig,
        selected_docs_cap=int(max(1, best)),
        heuristic_docs_cap=int(max(1, heuristic_docs_cap or best)),
        max_candidate_docs=int(max(1, max_candidate_docs)),
        target_fraction=float(target_fraction),
        cache_key=str(cache_key),
        cache_hit=False,
        total_wall_time_s=float(total_wall_time_s),
        stop_reason=str(run_stop_reason),
        candidate_profiles=tuple(candidate_profiles),
    )
    cache_write_time_s = 0.0
    if probe_cache is not None:
        write_start_s = time.perf_counter()
        probe_cache.put(
            ProbeCacheEntry(
                cache_key=str(cache_key),
                cache_version=int(AUTOTUNE_PROBE_CACHE_VERSION),
                created_at_utc=datetime.now(UTC).isoformat(),
                model_signature=model_sig,
                pack_mode=str(normalized_pack_mode),
                topology_signature=str(topology_sig),
                probe_mode=str(probe_mode),
                device_class_signature=device_sig,
                selected_docs_cap=int(max(1, best)),
                run_profile=run_profile,
            )
        )
        cache_write_time_s = time.perf_counter() - write_start_s
    return _ProbeRunOutcome(
        selected_docs_cap=int(max(1, best)),
        run_profile=run_profile,
        cache_hit=False,
        cache_lookup_time_s=float(cache_lookup_time_s),
        cache_write_time_s=float(cache_write_time_s),
        candidate_evaluations=int(len(candidate_profiles)),
    )


def _heuristic_docs_cap_for_representative(
    model: FNOCountSketch,
    doc: _FNOCountDoc,
    *,
    base_max_docs: int,
    training: bool,
) -> int:
    leaf_tokens = max(1, int(sum(int(length) for length in doc.leaf_token_lengths)))
    total_nodes = max(1, int(len(doc.leaf_token_ids) + len(doc.merge_counts_balanced)))
    state_width = max(1, int(getattr(model, "state_dim", 64)))
    complexity = max(1.0, float(leaf_tokens) * float(total_nodes) * float(state_width))
    if bool(getattr(model, "use_shared_theorem_surface", False)):
        complexity *= 1.5
    if bool(getattr(model, "use_factorized_score_fiber_surface", False)):
        complexity *= 1.15
    target = 180_000.0 if bool(training) else 320_000.0
    estimate = max(1, int(target / complexity))
    return int(max(1, min(int(base_max_docs), int(estimate))))


def _autotune_tree_batch_budgets(
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
    legacy_batch_size: int,
    pack_mode: str = "structure_bucket",
    bucket_mode: str = GPU_RUNTIME_BUCKET_MODE_EXACT_THEN_BUCKETED,
    structural_pad_limit: float = 0.5,
    auto_queue_min_docs: int = 8,
) -> _AutotunedTreeBatchBudgets:
    if not docs:
        fallback_docs = max(1, int(legacy_batch_size) if int(legacy_batch_size) > 0 else 64)
        return _AutotunedTreeBatchBudgets(
            train_leaf_token_budget=int(fallback_docs),
            train_node_budget=int(fallback_docs),
            eval_leaf_token_budget=int(fallback_docs),
            eval_node_budget=int(fallback_docs),
            eval_workers_per_mig=1,
            train_bucket_max_docs_by_n_leaves=tuple(),
            eval_bucket_max_docs_by_n_leaves=tuple(),
        )
    normalized_pack_mode = str(pack_mode or "structure_bucket").strip().lower()
    effective_bucket_mode = _effective_tree_bucket_mode(
        pack_mode=str(normalized_pack_mode),
        bucket_mode=str(bucket_mode),
    )
    fixed_fused_preferred = bool(normalized_pack_mode == "fixed_fused")
    auto_queue_targets = (
        _leaf_count_auto_queue_targets_for_docs(
            docs,
            structural_pad_limit=float(structural_pad_limit),
            min_docs=int(auto_queue_min_docs),
        )
        if _tree_leaf_count_auto_queue_enabled(effective_bucket_mode)
        else {}
    )
    representatives: Dict[int, _FNOCountDoc] = {}
    for doc in docs:
        n_leaves = int(len(doc.leaf_token_ids))
        target_n_leaves = int(auto_queue_targets.get(int(n_leaves), int(n_leaves)))
        existing = representatives.get(int(target_n_leaves))
        if existing is None or int(len(doc.leaf_token_ids)) > int(len(existing.leaf_token_ids)):
            representatives[int(target_n_leaves)] = doc
    base_max_docs = max(
        1,
        int(legacy_batch_size) if int(legacy_batch_size) > 0 else (512 if fixed_fused_preferred else 256),
    )
    if fixed_fused_preferred:
        base_max_docs = max(int(base_max_docs), 512)
    probe_cache = (
        ProbeCacheStore()
        if device.type == "cuda" and torch.cuda.is_available()
        else None
    )
    model_signature = _tree_batch_probe_model_signature(model)
    device_signature = _tree_batch_probe_device_signature(device)
    heuristic_time_s = 0.0
    train_probe_time_s = 0.0
    eval_probe_time_s = 0.0
    cache_lookup_time_s = 0.0
    cache_write_time_s = 0.0
    cache_hits = 0
    cache_misses = 0
    cache_writes = 0
    probe_runs = 0
    probe_candidate_evals = 0
    probe_profiles: List[ProbeRunProfile] = []
    train_caps: List[Tuple[int, int]] = []
    eval_caps: List[Tuple[int, int]] = []
    for n_leaves, rep_doc in sorted(representatives.items()):
        heuristic_start_s = time.perf_counter()
        train_cap = _heuristic_docs_cap_for_representative(
            model,
            rep_doc,
            base_max_docs=base_max_docs,
            training=True,
        )
        eval_cap = _heuristic_docs_cap_for_representative(
            model,
            rep_doc,
            base_max_docs=max(int(base_max_docs), 128),
            training=False,
        )
        heuristic_time_s += time.perf_counter() - heuristic_start_s
        topology_signature = _tree_batch_probe_topology_signature(rep_doc)
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                train_outcome = _probe_tree_docs_cap_for_representative(
                    model,
                    rep_doc,
                    device=device,
                    training=True,
                    max_candidate_docs=max(
                        1,
                        int(max(base_max_docs, train_cap, 2048 if fixed_fused_preferred else base_max_docs)),
                    ),
                    pack_mode=normalized_pack_mode,
                    heuristic_docs_cap=int(train_cap),
                    probe_cache=probe_cache,
                    model_signature=model_signature,
                    device_class_signature=device_signature,
                    topology_signature=topology_signature,
                )
                train_cap = int(train_outcome.selected_docs_cap)
                if not train_outcome.cache_hit:
                    train_probe_time_s += float(
                        train_outcome.run_profile.total_wall_time_s
                    )
                cache_lookup_time_s += float(train_outcome.cache_lookup_time_s)
                cache_write_time_s += float(train_outcome.cache_write_time_s)
                cache_hits += int(1 if train_outcome.cache_hit else 0)
                cache_misses += int(0 if train_outcome.cache_hit else 1)
                cache_writes += int(1 if train_outcome.cache_write_time_s > 0.0 else 0)
                probe_runs += 1
                probe_candidate_evals += int(train_outcome.candidate_evaluations)
                probe_profiles.append(train_outcome.run_profile)
            except Exception:
                pass
            try:
                eval_outcome = _probe_tree_docs_cap_for_representative(
                    model,
                    rep_doc,
                    device=device,
                    training=False,
                    max_candidate_docs=max(
                        1,
                        int(max(base_max_docs * 2, eval_cap, 2048 if fixed_fused_preferred else base_max_docs * 2)),
                    ),
                    pack_mode=normalized_pack_mode,
                    heuristic_docs_cap=int(eval_cap),
                    probe_cache=probe_cache,
                    model_signature=model_signature,
                    device_class_signature=device_signature,
                    topology_signature=topology_signature,
                )
                eval_cap = int(eval_outcome.selected_docs_cap)
                if not eval_outcome.cache_hit:
                    eval_probe_time_s += float(
                        eval_outcome.run_profile.total_wall_time_s
                    )
                cache_lookup_time_s += float(eval_outcome.cache_lookup_time_s)
                cache_write_time_s += float(eval_outcome.cache_write_time_s)
                cache_hits += int(1 if eval_outcome.cache_hit else 0)
                cache_misses += int(0 if eval_outcome.cache_hit else 1)
                cache_writes += int(1 if eval_outcome.cache_write_time_s > 0.0 else 0)
                probe_runs += 1
                probe_candidate_evals += int(eval_outcome.candidate_evaluations)
                probe_profiles.append(eval_outcome.run_profile)
            except Exception:
                pass
        train_caps.append((int(n_leaves), int(max(1, train_cap))))
        eval_caps.append((int(n_leaves), int(max(1, eval_cap))))
    train_leaf_budget = 0
    train_node_budget = 0
    eval_leaf_budget = 0
    eval_node_budget = 0
    for n_leaves, docs_cap in train_caps:
        rep_doc = representatives[int(n_leaves)]
        train_leaf_budget = max(
            int(train_leaf_budget),
            int(docs_cap) * int(sum(int(length) for length in rep_doc.leaf_token_lengths)),
        )
        train_node_budget = max(
            int(train_node_budget),
            int(docs_cap) * int(len(rep_doc.leaf_token_ids) + len(rep_doc.merge_counts_balanced)),
        )
    for n_leaves, docs_cap in eval_caps:
        rep_doc = representatives[int(n_leaves)]
        eval_leaf_budget = max(
            int(eval_leaf_budget),
            int(docs_cap) * int(sum(int(length) for length in rep_doc.leaf_token_lengths)),
        )
        eval_node_budget = max(
            int(eval_node_budget),
            int(docs_cap) * int(len(rep_doc.leaf_token_ids) + len(rep_doc.merge_counts_balanced)),
        )
    max_eval_cap = max((int(cap) for _n_leaves, cap in eval_caps), default=1)
    eval_workers_per_mig = 2 if device.type == "cuda" and max_eval_cap <= 8 else 1
    return _AutotunedTreeBatchBudgets(
        train_leaf_token_budget=int(max(1, train_leaf_budget)),
        train_node_budget=int(max(1, train_node_budget)),
        eval_leaf_token_budget=int(max(1, eval_leaf_budget)),
        eval_node_budget=int(max(1, eval_node_budget)),
        eval_workers_per_mig=int(max(1, eval_workers_per_mig)),
        train_bucket_max_docs_by_n_leaves=tuple(train_caps),
        eval_bucket_max_docs_by_n_leaves=tuple(eval_caps),
        probe_diagnostics=_AutotuneProbeDiagnostics(
            heuristic_time_s=float(heuristic_time_s),
            train_probe_time_s=float(train_probe_time_s),
            eval_probe_time_s=float(eval_probe_time_s),
            cache_lookup_time_s=float(cache_lookup_time_s),
            cache_write_time_s=float(cache_write_time_s),
            cache_hits=int(cache_hits),
            cache_misses=int(cache_misses),
            cache_writes=int(cache_writes),
            probe_runs=int(probe_runs),
            probe_candidate_evals=int(probe_candidate_evals),
            probe_profiles=tuple(probe_profiles),
        ),
    )


def _batched_root_predictions(
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
    resident_store: GpuBatchStore | None = None,
    runtime_telemetry: GpuRuntimeTelemetry | None = None,
    pack_mode: str = "structure_bucket",
    runtime_bucket_mode: str = GPU_RUNTIME_BUCKET_MODE_EXACT_THEN_BUCKETED,
    max_docs: int = 0,
    token_budget: int = 0,
    node_budget: int = 0,
    bucket_docs_cap_by_n_leaves: Mapping[int, int] | None = None,
    structural_pad_limit: float = 0.5,
    auto_queue_min_docs: int = 8,
    auto_queue_min_fill_ratio: float = 0.5,
    auto_queue_target_by_n_leaves: Mapping[int, int] | None = None,
    batching_metrics: _BatchingMetricsAccumulator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(docs) == 0:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    preds = np.zeros((len(docs),), dtype=np.float64)
    truths = np.asarray([float(doc.root_count) for doc in docs], dtype=np.float64)
    work_items = [
        _tree_work_item_from_doc(
            doc,
            doc_index=int(idx),
            work_kind="full_tree",
        )
        for idx, doc in enumerate(docs)
    ]
    packed_batches = _pack_tree_work_items(
        work_items,
        max_docs=int(max_docs) if int(max_docs) > 0 else len(work_items),
        max_total_leaf_tokens=int(token_budget),
        max_total_nodes=int(node_budget),
        max_total_merge_ops=0,
        bucket_docs_cap_by_n_leaves=bucket_docs_cap_by_n_leaves,
        bucket_mode=_effective_tree_bucket_mode(
            pack_mode=str(pack_mode),
            bucket_mode=str(runtime_bucket_mode),
        ),
        structural_pad_limit=float(structural_pad_limit),
        auto_queue_min_docs=int(auto_queue_min_docs),
        auto_queue_target_by_n_leaves=auto_queue_target_by_n_leaves,
        tail_repack_fill_ratio=(
            float(auto_queue_min_fill_ratio)
            if str(pack_mode or "structure_bucket").strip().lower() == "fixed_fused"
            else 0.0
        ),
        tail_repack_min_docs=int(auto_queue_min_docs),
    )

    model.eval()
    with torch.inference_mode():
        for packed_batch in packed_batches:
            group_indices = [int(item.doc_index) for item in packed_batch.items]
            group_docs = [item.doc for item in packed_batch.items]
            _eval_start_s = time.perf_counter()
            if (
                str(pack_mode or "structure_bucket").strip().lower() == "fixed_fused"
                and (_docs_share_fixed_leaf_shape(group_docs) or _docs_support_fixed_leaf_auto_queue(group_docs))
            ):
                resident_view = _tree_store_view_for_items(
                    resident_store,
                    packed_batch.items,
                    model=model,
                    runtime_telemetry=runtime_telemetry,
                )
                root_states = _precompute_balanced_doc_state_levels(
                    model,
                    group_docs,
                    device=device,
                    collect_merge_states=False,
                    prefer_fixed_fused=True,
                    target_n_leaves=int(packed_batch.bucket_key.n_leaves),
                    resident_view=resident_view,
                    runtime_telemetry=runtime_telemetry,
                ).root_states
            else:
                resident_view = _tree_store_view_for_items(
                    resident_store,
                    packed_batch.items,
                    model=model,
                    runtime_telemetry=runtime_telemetry,
                )
                views = _precompute_balanced_doc_state_views(
                    model,
                    group_docs,
                    device=device,
                    collect_merge_states=False,
                    prefer_fixed_fused=False,
                    resident_view=resident_view,
                    runtime_telemetry=runtime_telemetry,
                )
                root_states = torch.stack(
                    [view.root_state for view in views],
                    dim=0,
                )
            with _autocast_context(device):
                pred_tensor = model.predict_canonical_count_from_state(root_states)
            pred_arr = (
                np.asarray(pred_tensor.detach().cpu().reshape(-1).tolist(), dtype=np.float64)
                if pred_tensor.numel() > 0
                else np.zeros((0,), dtype=np.float64)
            )
            for local_idx, doc_idx in enumerate(group_indices):
                preds[int(doc_idx)] = float(pred_arr[int(local_idx)])
            if batching_metrics is not None:
                batching_metrics.eval_time_s += time.perf_counter() - _eval_start_s
                batching_metrics.add_batch(
                    packed_batch,
                    token_budget=int(token_budget),
                    node_budget=int(node_budget),
                    max_docs_budget=(
                        int(max_docs)
                        if int(max_docs) > 0
                        else max(1, int(len(group_docs)))
                    ),
                )
    return preds, truths


def _record_phi_feature(
    *,
    adapter: TheoremFeatureAdapter,
    label: Any,
    phi_tensor: torch.Tensor,
    moments_by_label: Dict[Hashable, _VectorMomentAccumulator],
    calibration_embeddings: List[np.ndarray],
    calibration_labels: List[Any],
    phi_pair_calibration_max_nodes: int | None,
) -> None:
    phi_vec = np.asarray(
        phi_tensor.detach().cpu().reshape(-1).numpy(),
        dtype=np.float64,
    )
    key = adapter.diagnostic_key(label)
    moments_by_label.setdefault(key, _VectorMomentAccumulator()).add(phi_vec)
    max_nodes = (
        None
        if phi_pair_calibration_max_nodes is None
        else int(phi_pair_calibration_max_nodes)
    )
    if max_nodes is not None and max_nodes > 0 and len(calibration_embeddings) >= max_nodes:
        return
    calibration_embeddings.append(phi_vec)
    calibration_labels.append(label)


def _theorem_feature_same_class_match_sum(
    left_labels: Sequence[Any],
    right_labels: Sequence[Any],
    *,
    adapter: TheoremFeatureAdapter,
    same_threshold: float | None = None,
    diff_threshold: float | None = None,
) -> tuple[float, int]:
    left_values = list(left_labels)
    right_values = list(right_labels)
    if len(left_values) != len(right_values) or not left_values:
        return 0.0, 0
    same_sum = 0.0
    for left, right in zip(left_values, right_values):
        same_sum += float(
            int(
                adapter.same_pair(
                    left,
                    right,
                    same_threshold=same_threshold,
                    diff_threshold=diff_threshold,
                )
            )
        )
    return float(same_sum), int(len(left_values))


def _eval_fno_exact_sketch_direct_metrics_fixed_fused_batch(
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
    resident_view: GpuBatchView | None = None,
    runtime_telemetry: GpuRuntimeTelemetry | None = None,
    phi_pair_calibration_max_nodes: int | None,
    moments_by_label: Dict[Hashable, _VectorMomentAccumulator],
    calibration_embeddings: List[np.ndarray],
    calibration_labels: List[Any],
) -> Dict[str, Any]:
    if not docs:
        return {
            "root_abs_sum": 0.0,
            "exact_projected_root_abs_sum": 0.0,
            "root_mae_oracle_counts_predicted_endpoints_abs_sum": 0.0,
            "root_mae_predicted_counts_oracle_endpoints_abs_sum": 0.0,
            "task_root_abs_sum": 0.0,
            "leaf_count_abs_sum": 0.0,
            "leaf_exact_match_sum": 0.0,
            "leaf_first_correct_sum": 0.0,
            "leaf_last_correct_sum": 0.0,
            "merge_first_correct_sum": 0.0,
            "merge_last_correct_sum": 0.0,
            "merge_exact_match_sum": 0.0,
            "leaf_count_off_by_k_histogram": {},
            "merge_exact_match_sum_by_depth": {},
            "merge_exact_match_count_by_depth": {},
            "merge_join_correct_sum": 0.0,
            "c2_on_range_exact_sum": 0.0,
            "phi_merge_alignment_sum": 0.0,
            "task_factorization_gap_sum": 0.0,
            "n_root": 0,
            "n_leaf": 0,
            "n_merge": 0,
            "n_join": 0,
            "n_c2": 0,
            "n_phi_merge_alignment": 0,
            "n_task_factorization_gap": 0,
            "n_replay": 0,
            "replay_same_class_sum": 0.0,
        }

    levels = _precompute_balanced_doc_state_levels(
        model,
        docs,
        device=device,
        collect_merge_states=True,
        prefer_fixed_fused=True,
        target_n_leaves=(
            int(resident_view.metadata.get("auto_queue_target_n_leaves", 0))
            if resident_view is not None
            else max(int(len(doc.leaf_token_ids)) for doc in docs)
        ),
        resident_view=resident_view,
        runtime_telemetry=runtime_telemetry,
    )
    leaf_state_batch = levels.leaf_states
    root_state_batch = levels.root_states
    leaf_valid_mask = levels.leaf_valid_mask
    merge_state_batch = _flatten_merge_state_batch(levels.merge_levels)
    if merge_state_batch is None:
        merge_state_batch = leaf_state_batch.new_zeros(
            (int(leaf_state_batch.shape[0]), 0, int(leaf_state_batch.shape[-1]))
        )
    merge_valid_mask = (
        torch.cat(
            [mask for mask in levels.merge_valid_levels if int(mask.shape[1]) > 0],
            dim=1,
        )
        if levels.merge_valid_levels
        else torch.zeros((int(leaf_state_batch.shape[0]), 0), device=device, dtype=torch.bool)
    )
    node_valid_mask = levels.node_valid_mask

    batch_size = int(len(docs))
    n_leaf = int(leaf_state_batch.shape[1])
    n_merge = int(merge_state_batch.shape[1])
    resident_tensors = dict(resident_view.tensors) if resident_view is not None else {}
    cached_targets = tuple(
        _cached_fused_doc_targets_for_target_leaves(
            doc,
            int(leaf_state_batch.shape[1]),
        )
        for doc in docs
    )

    def _stack_cached(attr_name: str, resident_name: str) -> torch.Tensor:
        resident_tensor = resident_tensors.get(str(resident_name))
        if isinstance(resident_tensor, torch.Tensor):
            return resident_tensor.detach()
        if runtime_telemetry is not None and resident_view is not None:
            runtime_telemetry.add_store_miss(reason=f"missing_{resident_name}")
        return torch.stack(
            [getattr(targets, attr_name) for targets in cached_targets],
            dim=0,
        )

    leaf_count_target_source = _stack_cached("leaf_count_targets_cpu", "leaf_count_targets")
    leaf_first_target_source = _stack_cached("leaf_first_targets_cpu", "leaf_first_targets")
    leaf_last_target_source = _stack_cached("leaf_last_targets_cpu", "leaf_last_targets")
    merge_count_target_source = _stack_cached("merge_count_targets_cpu", "merge_count_targets")
    merge_first_target_source = _stack_cached("merge_first_targets_cpu", "merge_first_targets")
    merge_last_target_source = _stack_cached("merge_last_targets_cpu", "merge_last_targets")
    node_count_target_source = _stack_cached("node_count_targets_cpu", "node_count_targets")
    node_first_target_source = _stack_cached("node_first_targets_cpu", "node_first_targets")
    node_last_target_source = _stack_cached("node_last_targets_cpu", "node_last_targets")

    pred_dtype = root_state_batch.dtype
    leaf_count_targets = leaf_count_target_source.to(device=device, dtype=pred_dtype)
    leaf_first_targets = leaf_first_target_source.to(device=device, dtype=torch.long)
    leaf_last_targets = leaf_last_target_source.to(device=device, dtype=torch.long)
    merge_count_targets = merge_count_target_source.to(device=device, dtype=pred_dtype)
    merge_first_targets = merge_first_target_source.to(device=device, dtype=torch.long)
    merge_last_targets = merge_last_target_source.to(device=device, dtype=torch.long)
    node_count_targets = node_count_target_source.to(device=device, dtype=pred_dtype)
    node_first_targets = node_first_target_source.to(device=device, dtype=torch.long)
    node_last_targets = node_last_target_source.to(device=device, dtype=torch.long)
    root_count_targets = node_count_targets[:, -1]

    pred_root_count = model.predict_count_from_state(root_state_batch)
    task_root_pred = model.predict_task_count_from_state(root_state_batch)
    canonical_root_pred = model.predict_canonical_count_from_state(root_state_batch)

    root_labels: List[Any] = []
    if model.use_shared_theorem_surface:
        root_labels = _materialize_fast_markov_labels(
            model,
            count_values=node_count_targets[:, -1].detach().cpu(),
            first_targets=node_first_targets[:, -1].detach().cpu(),
            last_targets=node_last_targets[:, -1].detach().cpu(),
        )
        task_root_targets = torch.as_tensor(
            [float(model.task_target_from_label(label)) for label in root_labels],
            device=device,
            dtype=task_root_pred.dtype,
        )
    else:
        task_root_targets = root_count_targets.to(dtype=task_root_pred.dtype)

    root_abs_sum = float(torch.abs(pred_root_count - root_count_targets).sum().detach().cpu())
    exact_projected_root_abs_sum = 0.0
    root_mae_oracle_counts_predicted_endpoints_abs_sum = 0.0
    root_mae_predicted_counts_oracle_endpoints_abs_sum = 0.0
    for batch_idx, doc in enumerate(docs):
        valid_leaf_indices = torch.nonzero(
            leaf_valid_mask[int(batch_idx)],
            as_tuple=False,
        ).reshape(-1)
        leaf_states = [
            leaf_state_batch[int(batch_idx), int(leaf_idx)]
            for leaf_idx in valid_leaf_indices.detach().cpu().tolist()
        ]
        exact_projected_root = _exact_projected_root_count_from_states(
            model,
            leaf_states,
            schedule="balanced",
        )
        exact_projected_root_abs_sum += abs(
            float(exact_projected_root) - float(doc.root_count)
        )
    task_root_abs_sum = float(
        torch.abs(task_root_pred - task_root_targets).sum().detach().cpu()
    )
    task_factorization_gap_sum = float(
        torch.abs(task_root_pred - canonical_root_pred).sum().detach().cpu()
    )

    flat_leaf_states = leaf_state_batch.reshape(int(batch_size * n_leaf), -1)
    leaf_pred_count = model.predict_count_from_state(flat_leaf_states).reshape(
        batch_size,
        n_leaf,
    )
    _leaf_h, leaf_first_logits, leaf_last_logits = model._split_state(flat_leaf_states)
    leaf_first_pred = torch.argmax(leaf_first_logits, dim=-1).reshape(batch_size, n_leaf)
    leaf_last_pred = torch.argmax(leaf_last_logits, dim=-1).reshape(batch_size, n_leaf)
    leaf_exact = (
        torch.round(leaf_pred_count).to(dtype=torch.long)
        == torch.round(leaf_count_targets).to(dtype=torch.long)
    ) & leaf_first_pred.eq(leaf_first_targets) & leaf_last_pred.eq(leaf_last_targets)
    leaf_exact = leaf_exact & leaf_valid_mask
    leaf_count_abs_sum = float(
        (
            torch.abs(leaf_pred_count - leaf_count_targets)
            * leaf_valid_mask.to(dtype=pred_dtype)
        )
        .sum()
        .detach()
        .cpu()
    )
    leaf_exact_match_sum = float(leaf_exact.to(dtype=pred_dtype).sum().detach().cpu())
    leaf_first_correct_sum = float(
        (
            leaf_first_pred.eq(leaf_first_targets).to(dtype=pred_dtype)
            * leaf_valid_mask.to(dtype=pred_dtype)
        )
        .sum()
        .detach()
        .cpu()
    )
    leaf_last_correct_sum = float(
        (
            leaf_last_pred.eq(leaf_last_targets).to(dtype=pred_dtype)
            * leaf_valid_mask.to(dtype=pred_dtype)
        )
        .sum()
        .detach()
        .cpu()
    )
    leaf_count_off_by_k_histogram: Dict[str, float] = {}
    if int(leaf_valid_mask.sum().item()) > 0:
        leaf_pred_count_cpu = leaf_pred_count.detach().cpu()
        leaf_count_targets_cpu = leaf_count_targets.detach().cpu()
        leaf_first_pred_cpu = leaf_first_pred.detach().cpu()
        leaf_last_pred_cpu = leaf_last_pred.detach().cpu()
        leaf_first_targets_cpu = leaf_first_targets.detach().cpu()
        leaf_last_targets_cpu = leaf_last_targets.detach().cpu()
        leaf_valid_mask_cpu = leaf_valid_mask.detach().cpu()
        for batch_idx in range(batch_size):
            valid_idx = torch.nonzero(
                leaf_valid_mask_cpu[int(batch_idx)],
                as_tuple=False,
            ).reshape(-1)
            if int(valid_idx.numel()) <= 0:
                continue
            doc_hist = _leaf_count_off_by_k_count_histogram(
                leaf_pred_count_cpu[int(batch_idx)].index_select(0, valid_idx).numpy(),
                leaf_count_targets_cpu[int(batch_idx)].index_select(0, valid_idx).numpy(),
            )
            for key, value in doc_hist.items():
                leaf_count_off_by_k_histogram[str(key)] = float(
                    leaf_count_off_by_k_histogram.get(str(key), 0.0) + float(value)
                )
            root_decomposition = _exact_markov_root_error_decomposition(
                truth_root_count=float(docs[int(batch_idx)].root_count),
                predicted_counts=leaf_pred_count_cpu[int(batch_idx)].index_select(0, valid_idx).numpy(),
                predicted_first=leaf_first_pred_cpu[int(batch_idx)].index_select(0, valid_idx).numpy(),
                predicted_last=leaf_last_pred_cpu[int(batch_idx)].index_select(0, valid_idx).numpy(),
                truth_counts=leaf_count_targets_cpu[int(batch_idx)].index_select(0, valid_idx).numpy(),
                truth_first=leaf_first_targets_cpu[int(batch_idx)].index_select(0, valid_idx).numpy(),
                truth_last=leaf_last_targets_cpu[int(batch_idx)].index_select(0, valid_idx).numpy(),
                schedule="balanced",
            )
            root_mae_oracle_counts_predicted_endpoints_abs_sum += float(
                root_decomposition["root_mae_oracle_counts_predicted_endpoints"]
            )
            root_mae_predicted_counts_oracle_endpoints_abs_sum += float(
                root_decomposition["root_mae_predicted_counts_oracle_endpoints"]
            )

    merge_exact_match_sum = 0.0
    merge_first_correct_sum = 0.0
    merge_last_correct_sum = 0.0
    merge_exact_match_sum_by_depth: Dict[str, float] = {}
    merge_exact_match_count_by_depth: Dict[str, int] = {}
    merge_join_correct_sum = 0.0
    phi_merge_alignment_sum = 0.0
    n_join = 0
    n_phi_merge_alignment = 0
    merge_phi_batch: torch.Tensor | None = None
    if int(n_merge) > 0:
        flat_merge_states = merge_state_batch.reshape(int(batch_size * n_merge), -1)
        merge_pred_count = model.predict_count_from_state(flat_merge_states).reshape(
            batch_size,
            n_merge,
        )
        _merge_h, merge_first_logits, merge_last_logits = model._split_state(flat_merge_states)
        merge_first_pred = torch.argmax(merge_first_logits, dim=-1).reshape(batch_size, n_merge)
        merge_last_pred = torch.argmax(merge_last_logits, dim=-1).reshape(batch_size, n_merge)
        merge_exact = (
            torch.round(merge_pred_count).to(dtype=torch.long)
            == torch.round(merge_count_targets).to(dtype=torch.long)
        ) & merge_first_pred.eq(merge_first_targets) & merge_last_pred.eq(merge_last_targets)
        merge_exact = merge_exact & merge_valid_mask
        merge_exact_match_sum = float(
            merge_exact.to(dtype=pred_dtype).sum().detach().cpu()
        )
        merge_first_correct_sum = float(
            (
                merge_first_pred.eq(merge_first_targets).to(dtype=pred_dtype)
                * merge_valid_mask.to(dtype=pred_dtype)
            )
            .sum()
            .detach()
            .cpu()
        )
        merge_last_correct_sum = float(
            (
                merge_last_pred.eq(merge_last_targets).to(dtype=pred_dtype)
                * merge_valid_mask.to(dtype=pred_dtype)
            )
            .sum()
            .detach()
            .cpu()
        )

        children_map = model._balanced_merge_children_map(int(n_leaf))
        depth_layout = model._balanced_tree_layout(int(n_leaf))
        merge_depth_by_local_idx = {
            int(local_idx): int(
                depth_layout["depth_by_global_idx"].get(int(n_leaf) + int(local_idx), 0)
            )
            for local_idx in range(int(n_merge))
        }
        left_index = torch.as_tensor(
            [int(children_map[idx][0]) for idx in range(int(n_merge))],
            device=device,
            dtype=torch.long,
        )
        right_index = torch.as_tensor(
            [int(children_map[idx][1]) for idx in range(int(n_merge))],
            device=device,
            dtype=torch.long,
        )
        all_state_batch = torch.cat((leaf_state_batch, merge_state_batch), dim=1)
        left_states = all_state_batch.index_select(1, left_index)
        right_states = all_state_batch.index_select(1, right_index)
        if model.use_markov_summary_spec:
            join_prob = model.predict_join_prob_from_states(
                left_states.reshape(int(batch_size * n_merge), -1),
                right_states.reshape(int(batch_size * n_merge), -1),
            ).reshape(batch_size, n_merge)
            pred_join = join_prob >= 0.5
        else:
            _lh, _left_first, left_last_logits = model._split_state(
                left_states.reshape(int(batch_size * n_merge), -1)
            )
            _rh, right_first_logits, _right_last = model._split_state(
                right_states.reshape(int(batch_size * n_merge), -1)
            )
            pred_join = torch.argmax(left_last_logits, dim=-1).reshape(
                batch_size,
                n_merge,
            ).ne(
                torch.argmax(right_first_logits, dim=-1).reshape(batch_size, n_merge)
            )
        truth_join = node_last_targets.index_select(1, left_index).ne(
            node_first_targets.index_select(1, right_index)
        )
        merge_join_correct_sum = float(
            (
                pred_join.eq(truth_join).to(dtype=pred_dtype)
                * merge_valid_mask.to(dtype=pred_dtype)
            )
            .sum()
            .detach()
            .cpu()
        )
        n_join = int(merge_valid_mask.sum().item())
        merge_exact_cpu = merge_exact.detach().cpu()
        merge_valid_mask_cpu = merge_valid_mask.detach().cpu()
        for merge_idx in range(int(n_merge)):
            depth_key = str(merge_depth_by_local_idx.get(int(merge_idx), 0))
            valid_column = merge_valid_mask_cpu[:, int(merge_idx)]
            valid_count = int(valid_column.to(dtype=torch.int64).sum().item())
            if valid_count <= 0:
                continue
            exact_sum = float(
                merge_exact_cpu[:, int(merge_idx)]
                .to(dtype=torch.float32)
                .masked_select(valid_column)
                .sum()
                .item()
            )
            merge_exact_match_sum_by_depth[depth_key] = float(
                merge_exact_match_sum_by_depth.get(depth_key, 0.0) + exact_sum
            )
            merge_exact_match_count_by_depth[depth_key] = int(
                merge_exact_match_count_by_depth.get(depth_key, 0) + valid_count
            )

        if model.use_shared_theorem_surface:
            merge_phi_batch = model.predict_phi_from_state(flat_merge_states).reshape(
                batch_size,
                n_merge,
                -1,
            )
            pred_parent_phi = model.predict_phi_parent_from_children(
                left_states.reshape(int(batch_size * n_merge), -1),
                right_states.reshape(int(batch_size * n_merge), -1),
            ).reshape(batch_size, n_merge, -1)
            phi_merge_alignment_sum = float(
                (
                    F.cosine_similarity(
                        pred_parent_phi.reshape(int(batch_size * n_merge), -1),
                        merge_phi_batch.reshape(int(batch_size * n_merge), -1),
                        dim=-1,
                    ).reshape(batch_size, n_merge)
                    * merge_valid_mask.to(dtype=pred_dtype)
                )
                .sum()
                .detach()
                .cpu()
            )
            n_phi_merge_alignment = int(merge_valid_mask.sum().item())
    root_phi_batch: torch.Tensor | None = None
    leaf_phi_batch: torch.Tensor | None = None
    if model.use_shared_theorem_surface:
        root_phi_batch = model.predict_phi_from_state(root_state_batch)
        leaf_phi_batch = (
            model.predict_phi_from_state(flat_leaf_states)
            .reshape(batch_size, n_leaf, -1)
        )
        for batch_idx, label in enumerate(root_labels):
            _record_phi_feature(
                adapter=model.theorem_feature_adapter,
                label=label,
                phi_tensor=root_phi_batch[int(batch_idx)],
                moments_by_label=moments_by_label,
                calibration_embeddings=calibration_embeddings,
                calibration_labels=calibration_labels,
                phi_pair_calibration_max_nodes=phi_pair_calibration_max_nodes,
            )
        for batch_idx in range(batch_size):
            leaf_labels = _materialize_fast_markov_labels(
                model,
                count_values=leaf_count_target_source[int(batch_idx)].detach().cpu(),
                first_targets=leaf_first_target_source[int(batch_idx)].detach().cpu(),
                last_targets=leaf_last_target_source[int(batch_idx)].detach().cpu(),
            )
            for leaf_idx, label in enumerate(leaf_labels):
                if not bool(leaf_valid_mask[int(batch_idx), int(leaf_idx)].item()):
                    continue
                _record_phi_feature(
                    adapter=model.theorem_feature_adapter,
                    label=label,
                    phi_tensor=leaf_phi_batch[int(batch_idx), int(leaf_idx)],
                    moments_by_label=moments_by_label,
                    calibration_embeddings=calibration_embeddings,
                    calibration_labels=calibration_labels,
                    phi_pair_calibration_max_nodes=phi_pair_calibration_max_nodes,
                )
            if int(n_merge) > 0 and merge_phi_batch is not None:
                merge_labels = _materialize_fast_markov_labels(
                    model,
                    count_values=merge_count_target_source[int(batch_idx)].detach().cpu(),
                    first_targets=merge_first_target_source[int(batch_idx)].detach().cpu(),
                    last_targets=merge_last_target_source[int(batch_idx)].detach().cpu(),
                )
                for merge_idx, label in enumerate(merge_labels):
                    if not bool(merge_valid_mask[int(batch_idx), int(merge_idx)].item()):
                        continue
                    _record_phi_feature(
                        adapter=model.theorem_feature_adapter,
                        label=label,
                        phi_tensor=merge_phi_batch[int(batch_idx), int(merge_idx)],
                        moments_by_label=moments_by_label,
                        calibration_embeddings=calibration_embeddings,
                        calibration_labels=calibration_labels,
                        phi_pair_calibration_max_nodes=phi_pair_calibration_max_nodes,
                    )

    c2_on_range_exact_sum = 0.0
    replay_same_class_sum = 0.0
    n_c2 = 0
    n_replay = 0
    if model.use_markov_summary_spec and model.codec_contract is not None:
        replay_state_batch = torch.cat(
            (leaf_state_batch, merge_state_batch, root_state_batch.unsqueeze(1)),
            dim=1,
        )
        replay_state_flat = replay_state_batch.reshape(
            int(replay_state_batch.shape[0] * replay_state_batch.shape[1]),
            -1,
        )
        decoded = model.codec_contract.decode(replay_state_flat)
        replay_encoded = model.codec_contract.reencode(decoded)
        replay_decoded = model.codec_contract.decode(replay_encoded)
        decoded_count = torch.round(decoded.count.detach()).reshape(-1)
        decoded_first = decoded.first.detach().to(dtype=torch.long).reshape(-1)
        decoded_last = decoded.last.detach().to(dtype=torch.long).reshape(-1)
        replay_count = torch.round(replay_decoded.count.detach()).reshape(-1)
        replay_first = replay_decoded.first.detach().to(dtype=torch.long).reshape(-1)
        replay_last = replay_decoded.last.detach().to(dtype=torch.long).reshape(-1)
        replay_valid = node_valid_mask.reshape(-1)
        c2_on_range_exact_sum = float(
            (
                decoded_count.eq(replay_count)
                & decoded_first.eq(replay_first)
                & decoded_last.eq(replay_last)
                & replay_valid
            )
            .to(dtype=torch.float32)
            .sum()
            .item()
        )
        n_c2 = int(replay_valid.sum().item())
        if n_c2 > 0:
            valid_indices = torch.nonzero(replay_valid, as_tuple=False).reshape(-1)
            replay_original_labels = _materialize_fast_markov_labels(
                model,
                count_values=decoded_count.index_select(0, valid_indices).detach().cpu(),
                first_targets=decoded_first.index_select(0, valid_indices).detach().cpu(),
                last_targets=decoded_last.index_select(0, valid_indices).detach().cpu(),
            )
            replay_reencoded_labels = _materialize_fast_markov_labels(
                model,
                count_values=replay_count.index_select(0, valid_indices).detach().cpu(),
                first_targets=replay_first.index_select(0, valid_indices).detach().cpu(),
                last_targets=replay_last.index_select(0, valid_indices).detach().cpu(),
            )
            replay_same_class_sum, n_replay = _theorem_feature_same_class_match_sum(
                replay_original_labels,
                replay_reencoded_labels,
                adapter=model.theorem_feature_adapter,
                same_threshold=model.theorem_pair_same_threshold,
                diff_threshold=model.theorem_pair_diff_threshold,
            )

    return {
        "root_abs_sum": float(root_abs_sum),
        "exact_projected_root_abs_sum": float(exact_projected_root_abs_sum),
        "root_mae_oracle_counts_predicted_endpoints_abs_sum": float(
            root_mae_oracle_counts_predicted_endpoints_abs_sum
        ),
        "root_mae_predicted_counts_oracle_endpoints_abs_sum": float(
            root_mae_predicted_counts_oracle_endpoints_abs_sum
        ),
        "task_root_abs_sum": float(task_root_abs_sum),
        "leaf_count_abs_sum": float(leaf_count_abs_sum),
        "leaf_exact_match_sum": float(leaf_exact_match_sum),
        "leaf_first_correct_sum": float(leaf_first_correct_sum),
        "leaf_last_correct_sum": float(leaf_last_correct_sum),
        "merge_first_correct_sum": float(merge_first_correct_sum),
        "merge_last_correct_sum": float(merge_last_correct_sum),
        "leaf_count_off_by_k_histogram": {
            str(key): float(value)
            for key, value in leaf_count_off_by_k_histogram.items()
        },
        "merge_exact_match_sum": float(merge_exact_match_sum),
        "merge_exact_match_sum_by_depth": {
            str(key): float(value)
            for key, value in merge_exact_match_sum_by_depth.items()
        },
        "merge_exact_match_count_by_depth": {
            str(key): int(value)
            for key, value in merge_exact_match_count_by_depth.items()
        },
        "merge_join_correct_sum": float(merge_join_correct_sum),
        "c2_on_range_exact_sum": float(c2_on_range_exact_sum),
        "phi_merge_alignment_sum": float(phi_merge_alignment_sum),
        "task_factorization_gap_sum": float(task_factorization_gap_sum),
        "n_root": int(batch_size),
        "n_leaf": int(leaf_valid_mask.sum().item()),
        "n_merge": int(merge_valid_mask.sum().item()),
        "n_join": int(n_join),
        "n_c2": int(n_c2),
        "n_phi_merge_alignment": int(n_phi_merge_alignment),
        "n_task_factorization_gap": int(batch_size),
        "n_replay": int(n_replay),
        "replay_same_class_sum": float(replay_same_class_sum),
    }


@torch.no_grad()
def _eval_fno_exact_sketch_direct_metrics_legacy(
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
    doc_limit: int | None = None,
    phi_pair_calibration_max_nodes: int | None = None,
) -> Dict[str, Any]:
    docs = _limit_eval_docs(docs, doc_limit=doc_limit)
    if len(docs) == 0:
        return _empty_exact_sketch_direct_metrics()

    model.eval()
    root_abs_sum = 0.0
    exact_projected_root_abs_sum = 0.0
    root_mae_oracle_counts_predicted_endpoints_abs_sum = 0.0
    root_mae_predicted_counts_oracle_endpoints_abs_sum = 0.0
    task_root_abs_sum = 0.0
    leaf_count_abs_sum = 0.0
    leaf_exact_match_sum = 0.0
    leaf_first_correct_sum = 0.0
    leaf_last_correct_sum = 0.0
    merge_first_correct_sum = 0.0
    merge_last_correct_sum = 0.0
    leaf_count_off_by_k_histogram: Dict[str, float] = {}
    merge_exact_match_sum_by_depth: Dict[str, float] = {}
    merge_exact_match_count_by_depth: Dict[str, int] = {}
    merge_exact_match_sum = 0.0
    merge_join_correct_sum = 0.0
    c2_on_range_exact_sum = 0.0
    phi_merge_alignment_sum = 0.0
    replay_same_class_sum = 0.0
    task_factorization_gap_sum = 0.0
    n_root = 0
    n_leaf = 0
    n_merge = 0
    n_join = 0
    n_c2 = 0
    n_phi_merge_alignment = 0
    n_replay = 0
    n_task_factorization_gap = 0
    phi_moments_by_label: Dict[Hashable, _VectorMomentAccumulator] = {}
    phi_calibration_embeddings: List[np.ndarray] = []
    phi_calibration_labels: List[Any] = []

    for doc in docs:
        leaf_state_batch = model.encode_leaf_tokens_batch(doc.leaf_token_ids, device=device)
        leaf_states = [leaf_state_batch[idx] for idx in range(int(leaf_state_batch.shape[0]))]
        root_state, merge_states = model._merge_states(
            leaf_states,
            schedule="balanced",
            collect_merge_states=True,
        )
        exact_targets = _balanced_exact_sketch_targets(
            leaf_counts=doc.leaf_counts,
            leaf_first_regimes=doc.leaf_first_regimes,
            leaf_last_regimes=doc.leaf_last_regimes,
        )
        leaf_metadata, merge_metadata, root_metadata = (
            _theorem_feature_metadata_sequences_from_fno_doc(doc)
        )
        feature_targets = theorem_feature_targets_from_markov_exact_targets(
            adapter=model.theorem_feature_adapter,
            exact_targets=exact_targets,
            leaf_metadata=leaf_metadata,
            merge_metadata=merge_metadata,
            root_metadata=root_metadata,
        )

        pred_root_count = float(model.predict_count_from_state(root_state).detach().cpu().item())
        root_abs_sum += abs(pred_root_count - float(doc.root_count))
        exact_projected_root = _exact_projected_root_count_from_states(
            model,
            leaf_states,
            schedule="balanced",
        )
        exact_projected_root_abs_sum += abs(
            float(exact_projected_root) - float(doc.root_count)
        )
        root_decomposition = _exact_markov_root_error_decomposition(
            truth_root_count=float(doc.root_count),
            predicted_counts=[
                float(model.predict_count_from_state(state).detach().cpu().item())
                for state in leaf_states
            ],
            predicted_first=[
                int(torch.argmax(model._split_state(state)[1], dim=-1).detach().cpu().item())
                for state in leaf_states
            ],
            predicted_last=[
                int(torch.argmax(model._split_state(state)[2], dim=-1).detach().cpu().item())
                for state in leaf_states
            ],
            truth_counts=[float(count) for count, _first, _last in exact_targets["leaf"]],
            truth_first=[int(first) for _count, first, _last in exact_targets["leaf"]],
            truth_last=[int(last) for _count, _first, last in exact_targets["leaf"]],
            schedule="balanced",
        )
        root_mae_oracle_counts_predicted_endpoints_abs_sum += float(
            root_decomposition["root_mae_oracle_counts_predicted_endpoints"]
        )
        root_mae_predicted_counts_oracle_endpoints_abs_sum += float(
            root_decomposition["root_mae_predicted_counts_oracle_endpoints"]
        )
        task_root_pred = float(
            model.predict_task_count_from_state(root_state).detach().cpu().item()
        )
        task_root_target = float(doc.root_count)
        if model.use_shared_theorem_surface and feature_targets.root:
            task_root_target = float(
                model.task_target_from_label(feature_targets.root[0])
            )
        task_root_abs_sum += abs(task_root_pred - task_root_target)
        task_factorization_gap_sum += (
            abs(
                float(model.predict_task_count_from_state(root_state).detach().cpu().item())
                - float(model.predict_canonical_count_from_state(root_state).detach().cpu().item())
            )
        )
        n_root += 1
        n_task_factorization_gap += 1
        if model.use_shared_theorem_surface and feature_targets.root:
            _record_phi_feature(
                adapter=model.theorem_feature_adapter,
                label=feature_targets.root[0],
                phi_tensor=model.predict_phi_from_state(root_state).detach(),
                moments_by_label=phi_moments_by_label,
                calibration_embeddings=phi_calibration_embeddings,
                calibration_labels=phi_calibration_labels,
                phi_pair_calibration_max_nodes=phi_pair_calibration_max_nodes,
            )

        for idx, (state, (truth_count, truth_first, truth_last)) in enumerate(zip(
            leaf_states,
            exact_targets["leaf"],
        )):
            pred_count = float(model.predict_count_from_state(state).detach().cpu().item())
            leaf_count_abs_sum += abs(pred_count - float(truth_count))
            _h, first_logits, last_logits = model._split_state(state)
            pred_first = int(torch.argmax(first_logits, dim=-1).detach().cpu().item())
            pred_last = int(torch.argmax(last_logits, dim=-1).detach().cpu().item())
            leaf_first_correct_sum += float(int(pred_first == int(truth_first)))
            leaf_last_correct_sum += float(int(pred_last == int(truth_last)))
            abs_diff_key = str(
                int(abs(int(np.rint(pred_count)) - int(np.rint(float(truth_count)))))
            )
            leaf_count_off_by_k_histogram[abs_diff_key] = float(
                leaf_count_off_by_k_histogram.get(abs_diff_key, 0.0) + 1.0
            )
            leaf_exact_match_sum += float(
                float(
                    int(
                        int(np.rint(pred_count)) == int(truth_count)
                        and pred_first == int(truth_first)
                        and pred_last == int(truth_last)
                    )
                )
            )
            n_leaf += 1
            if model.use_shared_theorem_surface:
                _record_phi_feature(
                    adapter=model.theorem_feature_adapter,
                    label=feature_targets.leaf[int(idx)],
                    phi_tensor=model.predict_phi_from_state(state).detach(),
                    moments_by_label=phi_moments_by_label,
                    calibration_embeddings=phi_calibration_embeddings,
                    calibration_labels=phi_calibration_labels,
                    phi_pair_calibration_max_nodes=phi_pair_calibration_max_nodes,
                )

        all_states = list(leaf_states) + list(merge_states)
        all_exact = list(exact_targets["leaf"]) + list(exact_targets["merge"])
        children_map = model._balanced_merge_children_map(len(leaf_states))
        depth_layout = model._balanced_tree_layout(len(leaf_states))
        for merge_idx, (state, (truth_count, truth_first, truth_last)) in enumerate(
            zip(merge_states, exact_targets["merge"])
        ):
            pred_count = float(model.predict_count_from_state(state).detach().cpu().item())
            _h, first_logits, last_logits = model._split_state(state)
            pred_first = int(torch.argmax(first_logits, dim=-1).detach().cpu().item())
            pred_last = int(torch.argmax(last_logits, dim=-1).detach().cpu().item())
            merge_first_correct_sum += float(int(pred_first == int(truth_first)))
            merge_last_correct_sum += float(int(pred_last == int(truth_last)))
            merge_exact_value = float(
                int(
                    int(np.rint(pred_count)) == int(truth_count)
                    and pred_first == int(truth_first)
                    and pred_last == int(truth_last)
                )
            )
            merge_exact_match_sum += float(merge_exact_value)
            depth_key = str(
                int(
                    depth_layout["depth_by_global_idx"].get(
                        int(len(leaf_states)) + int(merge_idx),
                        0,
                    )
                )
            )
            merge_exact_match_sum_by_depth[depth_key] = float(
                merge_exact_match_sum_by_depth.get(depth_key, 0.0)
                + float(merge_exact_value)
            )
            merge_exact_match_count_by_depth[depth_key] = int(
                merge_exact_match_count_by_depth.get(depth_key, 0) + 1
            )
            n_merge += 1
            if model.use_shared_theorem_surface:
                _record_phi_feature(
                    adapter=model.theorem_feature_adapter,
                    label=feature_targets.merge[int(merge_idx)],
                    phi_tensor=model.predict_phi_from_state(state).detach(),
                    moments_by_label=phi_moments_by_label,
                    calibration_embeddings=phi_calibration_embeddings,
                    calibration_labels=phi_calibration_labels,
                    phi_pair_calibration_max_nodes=phi_pair_calibration_max_nodes,
                )

            child_indices = children_map.get(int(merge_idx))
            if child_indices is None:
                continue
            left_idx, right_idx = child_indices
            left_state = all_states[int(left_idx)]
            right_state = all_states[int(right_idx)]
            left_exact = all_exact[int(left_idx)]
            right_exact = all_exact[int(right_idx)]
            truth_join = 0 if int(left_exact[2]) == int(right_exact[1]) else 1
            if model.use_markov_summary_spec:
                join_prob = float(
                    model.predict_join_prob_from_states(left_state, right_state)
                    .detach()
                    .cpu()
                    .item()
                )
                pred_join = int(join_prob >= 0.5)
            else:
                _lh, _left_first, left_last = model._split_state(left_state)
                _rh, right_first, _right_last = model._split_state(right_state)
                pred_join = int(
                    int(torch.argmax(left_last, dim=-1).detach().cpu().item())
                    != int(torch.argmax(right_first, dim=-1).detach().cpu().item())
                )
            merge_join_correct_sum += float(int(pred_join == truth_join))
            n_join += 1
            if model.use_shared_theorem_surface:
                pred_phi = model.predict_phi_parent_from_children(
                    left_state,
                    right_state,
                ).detach()
                target_phi = model.predict_phi_from_state(state).detach()
                phi_merge_alignment_sum += (
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
                n_phi_merge_alignment += 1

        if model.use_markov_summary_spec and model.codec_contract is not None:
            for state in list(leaf_states) + list(merge_states) + [root_state]:
                decoded = model.codec_contract.decode(state)
                replay_state = model.codec_contract.reencode(decoded)
                replay_decoded = model.codec_contract.decode(replay_state)
                original_label = model.theorem_feature_adapter.oracle_label(
                    count=float(torch.round(decoded.count).detach().cpu().item()),
                    first=int(decoded.first.detach().cpu().item()),
                    last=int(decoded.last.detach().cpu().item()),
                    metadata=None,
                )
                replay_label = model.theorem_feature_adapter.oracle_label(
                    count=float(torch.round(replay_decoded.count).detach().cpu().item()),
                    first=int(replay_decoded.first.detach().cpu().item()),
                    last=int(replay_decoded.last.detach().cpu().item()),
                    metadata=None,
                )
                replay_same_class_sum += float(
                    int(
                        model.theorem_feature_adapter.same_pair(
                            original_label,
                            replay_label,
                            same_threshold=model.theorem_pair_same_threshold,
                            diff_threshold=model.theorem_pair_diff_threshold,
                        )
                    )
                )
                c2_on_range_exact_sum += (
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
                n_replay += 1
                n_c2 += 1

    root_direct_count_mae = _mean_or_default(total=root_abs_sum, count=n_root, default=0.0)
    exact_projected_root_mae = _mean_or_default(
        total=exact_projected_root_abs_sum,
        count=n_root,
        default=0.0,
    )
    root_mae_oracle_counts_predicted_endpoints = _mean_or_default(
        total=root_mae_oracle_counts_predicted_endpoints_abs_sum,
        count=n_root,
        default=0.0,
    )
    root_mae_predicted_counts_oracle_endpoints = _mean_or_default(
        total=root_mae_predicted_counts_oracle_endpoints_abs_sum,
        count=n_root,
        default=0.0,
    )
    task_root_mae = _mean_or_default(total=task_root_abs_sum, count=n_root, default=0.0)
    leaf_direct_count_mae = _mean_or_default(
        total=leaf_count_abs_sum,
        count=n_leaf,
        default=0.0,
    )
    leaf_direct_exact_match = _mean_or_default(
        total=leaf_exact_match_sum,
        count=n_leaf,
        default=1.0,
    )
    leaf_first_accuracy = _mean_or_default(
        total=leaf_first_correct_sum,
        count=n_leaf,
        default=1.0,
    )
    leaf_last_accuracy = _mean_or_default(
        total=leaf_last_correct_sum,
        count=n_leaf,
        default=1.0,
    )
    merge_direct_exact_match = _mean_or_default(
        total=merge_exact_match_sum,
        count=n_merge,
        default=1.0,
    )
    merge_first_accuracy = _mean_or_default(
        total=merge_first_correct_sum,
        count=n_merge,
        default=1.0,
    )
    merge_last_accuracy = _mean_or_default(
        total=merge_last_correct_sum,
        count=n_merge,
        default=1.0,
    )
    merge_join_bit_accuracy = _mean_or_default(
        total=merge_join_correct_sum,
        count=n_join,
        default=1.0,
    )
    c2_on_range_exact_match = _mean_or_default(
        total=c2_on_range_exact_sum,
        count=n_c2,
        default=1.0,
    )
    phi_merge_alignment = _mean_or_default(
        total=phi_merge_alignment_sum,
        count=n_phi_merge_alignment,
        default=1.0,
    )
    class_centroids: List[np.ndarray] = []
    within_class_terms: List[float] = []
    for moments in phi_moments_by_label.values():
        centroid = moments.centroid()
        if centroid is None:
            continue
        class_centroids.append(centroid)
        within_class_terms.append(moments.mean_sq_distance())
    phi_within_class_variance = (
        float(np.mean(np.asarray(within_class_terms, dtype=np.float64)))
        if within_class_terms
        else 0.0
    )
    between_terms: List[float] = []
    for i in range(len(class_centroids)):
        for j in range(i + 1, len(class_centroids)):
            between_terms.append(
                float(np.linalg.norm(class_centroids[i] - class_centroids[j]))
            )
    phi_between_class_margin = (
        float(np.mean(np.asarray(between_terms, dtype=np.float64)))
        if between_terms
        else 0.0
    )
    pair_same_scores: List[float] = []
    pair_diff_scores: List[float] = []
    if model.use_shared_theorem_surface and len(phi_calibration_embeddings) > 1:
        phi_pairs = build_theorem_feature_pair_sets(
            phi_calibration_labels,
            adapter=model.theorem_feature_adapter,
            same_threshold=model.theorem_pair_same_threshold,
            diff_threshold=model.theorem_pair_diff_threshold,
        )
        if phi_pairs.total_pairs > 0:
            phi_array = np.asarray(phi_calibration_embeddings, dtype=np.float64)
            phi_norm = np.linalg.norm(phi_array, axis=1, keepdims=True)
            phi_norm = np.maximum(phi_norm, 1e-12)
            phi_unit = phi_array / phi_norm
            for left_idx, right_idx in list(phi_pairs.same_pairs):
                pair_same_scores.append(
                    float(np.dot(phi_unit[int(left_idx)], phi_unit[int(right_idx)]))
                )
            for left_idx, right_idx in list(phi_pairs.different_pairs):
                pair_diff_scores.append(
                    float(np.dot(phi_unit[int(left_idx)], phi_unit[int(right_idx)]))
                )
    pair_metrics = theorem_feature_pair_metrics_from_scores(
        same_scores=pair_same_scores,
        different_scores=pair_diff_scores,
    ).as_dict()
    phi_replay_same_class_rate = _mean_or_default(
        total=replay_same_class_sum,
        count=n_replay,
        default=float("nan"),
    )
    task_factorization_gap = _mean_or_default(
        total=task_factorization_gap_sum,
        count=n_task_factorization_gap,
        default=0.0,
    )
    leaf_codec_selection_value = float(
        leaf_direct_count_mae
        + (1.0 - leaf_direct_exact_match)
        + (1.0 - merge_join_bit_accuracy)
        + (1.0 - c2_on_range_exact_match)
    )
    theorem_bootstrap_selection_value = float(
        leaf_direct_count_mae
        + (1.0 - leaf_direct_exact_match)
        + (1.0 - merge_direct_exact_match)
        + (1.0 - merge_join_bit_accuracy)
        + (1.0 - c2_on_range_exact_match)
    )
    selection_root_mae = _exact_sketch_selection_root_mae(
        model,
        root_direct_count_mae=float(root_direct_count_mae),
        exact_projected_root_mae=float(exact_projected_root_mae),
    )
    selection_split_penalty = _exact_sketch_selection_split_penalty(
        model,
        root_mae_oracle_counts_predicted_endpoints=float(
            root_mae_oracle_counts_predicted_endpoints
        ),
        root_mae_predicted_counts_oracle_endpoints=float(
            root_mae_predicted_counts_oracle_endpoints
        ),
    )
    selection_value = float(
        selection_root_mae
        + selection_split_penalty
        + (1.0 - leaf_direct_exact_match)
        + (1.0 - merge_direct_exact_match)
        + (1.0 - merge_join_bit_accuracy)
    )
    task_root_selection_value = float(
        task_root_mae
        + (1.0 - leaf_direct_exact_match)
        + (1.0 - merge_direct_exact_match)
        + (1.0 - merge_join_bit_accuracy)
    )
    return {
        "root_direct_count_mae": root_direct_count_mae,
        "exact_projected_root_mae": exact_projected_root_mae,
        "certified_projected_root_mae": exact_projected_root_mae,
        "root_mae_predicted_counts_predicted_endpoints": float(
            exact_projected_root_mae
        ),
        "root_mae_oracle_counts_predicted_endpoints": float(
            root_mae_oracle_counts_predicted_endpoints
        ),
        "root_mae_predicted_counts_oracle_endpoints": float(
            root_mae_predicted_counts_oracle_endpoints
        ),
        "learned_merger_gap": float(root_direct_count_mae - exact_projected_root_mae),
        "task_root_mae": task_root_mae,
        "task_root_mae_ablation": task_root_mae,
        "leaf_direct_count_mae": leaf_direct_count_mae,
        "leaf_direct_exact_match": leaf_direct_exact_match,
        "leaf_first_accuracy": float(leaf_first_accuracy),
        "leaf_last_accuracy": float(leaf_last_accuracy),
        "merge_first_accuracy": float(merge_first_accuracy),
        "merge_last_accuracy": float(merge_last_accuracy),
        "leaf_count_off_by_k_histogram": _normalize_count_histogram(
            leaf_count_off_by_k_histogram
        ),
        "merge_exact_summary_match_rate_by_depth": _merge_rate_dict_from_sums(
            merge_exact_match_sum_by_depth,
            merge_exact_match_count_by_depth,
        ),
        "merge_direct_exact_match": merge_direct_exact_match,
        "merge_join_bit_accuracy": merge_join_bit_accuracy,
        "c2_on_range_exact_match": c2_on_range_exact_match,
        "phi_merge_alignment": phi_merge_alignment,
        "phi_within_class_variance": phi_within_class_variance,
        "phi_between_class_margin": phi_between_class_margin,
        "phi_pair_same_accuracy": float(pair_metrics["phi_pair_same_accuracy"]),
        "phi_pair_diff_accuracy": float(pair_metrics["phi_pair_diff_accuracy"]),
        "phi_pair_auc": float(pair_metrics["phi_pair_auc"]),
        "phi_replay_same_class_rate": float(phi_replay_same_class_rate),
        "task_factorization_gap": float(task_factorization_gap),
        "val_leaf_codec_direct": leaf_codec_selection_value,
        "val_theorem_bootstrap_direct": theorem_bootstrap_selection_value,
        "val_exact_sketch_direct": selection_value,
        "val_task_root_exact_sketch_direct": task_root_selection_value,
        "n_docs": float(len(docs)),
        "n_leaf_nodes": float(n_leaf),
        "n_merge_nodes": float(n_merge),
    }


@torch.inference_mode()
def _eval_fno_exact_sketch_direct_metrics(
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
    resident_store: GpuBatchStore | None = None,
    runtime_telemetry: GpuRuntimeTelemetry | None = None,
    pack_mode: str = "structure_bucket",
    runtime_bucket_mode: str = GPU_RUNTIME_BUCKET_MODE_EXACT_THEN_BUCKETED,
    doc_limit: int | None = None,
    phi_pair_calibration_max_nodes: int | None = 512,
    max_docs: int = 0,
    token_budget: int = 0,
    node_budget: int = 0,
    bucket_docs_cap_by_n_leaves: Mapping[int, int] | None = None,
    structural_pad_limit: float = 0.5,
    auto_queue_min_docs: int = 8,
    auto_queue_min_fill_ratio: float = 0.5,
    auto_queue_target_by_n_leaves: Mapping[int, int] | None = None,
    batching_metrics: _BatchingMetricsAccumulator | None = None,
    memory_probe: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    docs = _limit_eval_docs(docs, doc_limit=doc_limit)
    if len(docs) == 0:
        return _empty_exact_sketch_direct_metrics()

    def _emit_memory_probe(event: str, **payload: Any) -> None:
        if memory_probe is None:
            return
        memory_probe(str(event), {str(key): value for key, value in payload.items()})

    model.eval()
    root_abs_sum = 0.0
    exact_projected_root_abs_sum = 0.0
    root_mae_oracle_counts_predicted_endpoints_abs_sum = 0.0
    root_mae_predicted_counts_oracle_endpoints_abs_sum = 0.0
    task_root_abs_sum = 0.0
    leaf_count_abs_sum = 0.0
    leaf_exact_match_sum = 0.0
    leaf_first_correct_sum = 0.0
    leaf_last_correct_sum = 0.0
    merge_first_correct_sum = 0.0
    merge_last_correct_sum = 0.0
    leaf_count_off_by_k_histogram: Dict[str, float] = {}
    merge_exact_match_sum_by_depth: Dict[str, float] = {}
    merge_exact_match_count_by_depth: Dict[str, int] = {}
    merge_exact_match_sum = 0.0
    merge_join_correct_sum = 0.0
    c2_on_range_exact_sum = 0.0
    phi_merge_alignment_sum = 0.0
    replay_same_class_sum = 0.0
    task_factorization_gap_sum = 0.0
    n_root = 0
    n_leaf = 0
    n_merge = 0
    n_join = 0
    n_c2 = 0
    n_phi_merge_alignment = 0
    n_replay = 0
    n_task_factorization_gap = 0
    phi_moments_by_label: Dict[Hashable, _VectorMomentAccumulator] = {}
    phi_calibration_embeddings: List[np.ndarray] = []
    phi_calibration_labels: List[Any] = []

    work_items = [
        _tree_work_item_from_doc(
            doc,
            doc_index=int(idx),
            work_kind="full_tree",
            collect_leaf=True,
            collect_c2=True,
            collect_c3=True,
        )
        for idx, doc in enumerate(docs)
    ]
    packed_max_docs = int(max_docs) if int(max_docs) > 0 else len(work_items)
    packed_batches = _pack_tree_work_items(
        work_items,
        max_docs=int(packed_max_docs),
        max_total_leaf_tokens=int(token_budget),
        max_total_nodes=int(node_budget),
        max_total_merge_ops=0,
        bucket_docs_cap_by_n_leaves=bucket_docs_cap_by_n_leaves,
        bucket_mode=_effective_tree_bucket_mode(
            pack_mode=str(pack_mode),
            bucket_mode=str(runtime_bucket_mode),
        ),
        structural_pad_limit=float(structural_pad_limit),
        auto_queue_min_docs=int(auto_queue_min_docs),
        auto_queue_target_by_n_leaves=auto_queue_target_by_n_leaves,
        tail_repack_fill_ratio=(
            float(auto_queue_min_fill_ratio)
            if str(pack_mode or "structure_bucket").strip().lower() == "fixed_fused"
            else 0.0
        ),
        tail_repack_min_docs=int(auto_queue_min_docs),
    )

    for batch_index, packed_batch in enumerate(packed_batches):
        eval_start_s = time.perf_counter()
        batch_docs = [item.doc for item in packed_batch.items]
        batch_doc_count = int(len(packed_batch.items))
        _emit_memory_probe(
            "pre_exact_eval_batch",
            batch_index=int(batch_index),
            batch_docs=int(batch_doc_count),
            max_docs=int(packed_max_docs),
            pack_mode=str(pack_mode),
            runtime_bucket_mode=str(runtime_bucket_mode),
            padded_leaf_tokens=int(packed_batch.padded_leaf_tokens),
            total_nodes=int(packed_batch.total_nodes),
        )
        use_fixed_fused = (
            str(pack_mode or "structure_bucket").strip().lower() == "fixed_fused"
            and (_docs_share_fixed_leaf_shape(batch_docs) or _docs_support_fixed_leaf_auto_queue(batch_docs))
            and _supports_fast_markov_pair_masks(model)
        )
        if use_fixed_fused:
            resident_view = _tree_store_view_for_items(
                resident_store,
                packed_batch.items,
                model=model,
                runtime_telemetry=runtime_telemetry,
            )
            batch_metrics = _eval_fno_exact_sketch_direct_metrics_fixed_fused_batch(
                model,
                batch_docs,
                device=device,
                resident_view=resident_view,
                runtime_telemetry=runtime_telemetry,
                phi_pair_calibration_max_nodes=phi_pair_calibration_max_nodes,
                moments_by_label=phi_moments_by_label,
                calibration_embeddings=phi_calibration_embeddings,
                calibration_labels=phi_calibration_labels,
            )
            root_abs_sum += float(batch_metrics["root_abs_sum"])
            exact_projected_root_abs_sum += float(
                batch_metrics.get("exact_projected_root_abs_sum", 0.0)
            )
            root_mae_oracle_counts_predicted_endpoints_abs_sum += float(
                batch_metrics.get(
                    "root_mae_oracle_counts_predicted_endpoints_abs_sum",
                    0.0,
                )
            )
            root_mae_predicted_counts_oracle_endpoints_abs_sum += float(
                batch_metrics.get(
                    "root_mae_predicted_counts_oracle_endpoints_abs_sum",
                    0.0,
                )
            )
            task_root_abs_sum += float(batch_metrics["task_root_abs_sum"])
            leaf_count_abs_sum += float(batch_metrics["leaf_count_abs_sum"])
            leaf_exact_match_sum += float(batch_metrics["leaf_exact_match_sum"])
            leaf_first_correct_sum += float(batch_metrics.get("leaf_first_correct_sum", 0.0))
            leaf_last_correct_sum += float(batch_metrics.get("leaf_last_correct_sum", 0.0))
            merge_first_correct_sum += float(
                batch_metrics.get("merge_first_correct_sum", 0.0)
            )
            merge_last_correct_sum += float(
                batch_metrics.get("merge_last_correct_sum", 0.0)
            )
            for key, value in dict(
                batch_metrics.get("leaf_count_off_by_k_histogram", {}) or {}
            ).items():
                leaf_count_off_by_k_histogram[str(key)] = float(
                    leaf_count_off_by_k_histogram.get(str(key), 0.0) + float(value)
                )
            merge_exact_match_sum += float(batch_metrics["merge_exact_match_sum"])
            for key, value in dict(
                batch_metrics.get("merge_exact_match_sum_by_depth", {}) or {}
            ).items():
                merge_exact_match_sum_by_depth[str(key)] = float(
                    merge_exact_match_sum_by_depth.get(str(key), 0.0) + float(value)
                )
            for key, value in dict(
                batch_metrics.get("merge_exact_match_count_by_depth", {}) or {}
            ).items():
                merge_exact_match_count_by_depth[str(key)] = int(
                    merge_exact_match_count_by_depth.get(str(key), 0) + int(value)
                )
            merge_join_correct_sum += float(batch_metrics["merge_join_correct_sum"])
            c2_on_range_exact_sum += float(batch_metrics["c2_on_range_exact_sum"])
            phi_merge_alignment_sum += float(batch_metrics["phi_merge_alignment_sum"])
            task_factorization_gap_sum += float(batch_metrics["task_factorization_gap_sum"])
            n_root += int(batch_metrics["n_root"])
            n_leaf += int(batch_metrics["n_leaf"])
            n_merge += int(batch_metrics["n_merge"])
            n_join += int(batch_metrics["n_join"])
            n_c2 += int(batch_metrics["n_c2"])
            n_phi_merge_alignment += int(batch_metrics["n_phi_merge_alignment"])
            n_task_factorization_gap += int(batch_metrics["n_task_factorization_gap"])
            n_replay += int(batch_metrics["n_replay"])
            replay_same_class_sum += float(batch_metrics["replay_same_class_sum"])
            if batching_metrics is not None:
                batching_metrics.eval_time_s += time.perf_counter() - eval_start_s
                batching_metrics.add_batch(
                    packed_batch,
                    token_budget=int(token_budget),
                    node_budget=int(node_budget),
                    max_docs_budget=(
                        int(max_docs) if int(max_docs) > 0 else max(1, int(len(packed_batch.items)))
                    ),
                )
        else:
            resident_view = _tree_store_view_for_items(
                resident_store,
                packed_batch.items,
                model=model,
                runtime_telemetry=runtime_telemetry,
            )
            views = _precompute_balanced_doc_state_views(
                model,
                batch_docs,
                device=device,
                collect_merge_states=True,
                prefer_fixed_fused=False,
                resident_view=resident_view,
                runtime_telemetry=runtime_telemetry,
            )
            for batch_idx, doc in enumerate(batch_docs):
                precomputed_view = views[int(batch_idx)]
                leaf_states = [
                    precomputed_view.state_batch[idx]
                    for idx in range(int(precomputed_view.state_batch.shape[0]))
                ]
                root_state = precomputed_view.root_state
                merge_states = list(precomputed_view.merge_states)
                exact_targets = _balanced_exact_sketch_targets(
                    leaf_counts=doc.leaf_counts,
                    leaf_first_regimes=doc.leaf_first_regimes,
                    leaf_last_regimes=doc.leaf_last_regimes,
                )
                leaf_metadata, merge_metadata, root_metadata = (
                    _theorem_feature_metadata_sequences_from_fno_doc(doc)
                )
                feature_targets = theorem_feature_targets_from_markov_exact_targets(
                    adapter=model.theorem_feature_adapter,
                    exact_targets=exact_targets,
                    leaf_metadata=leaf_metadata,
                    merge_metadata=merge_metadata,
                    root_metadata=root_metadata,
                )

                pred_root_count = float(
                    model.predict_count_from_state(root_state).detach().cpu().item()
                )
                root_abs_sum += abs(pred_root_count - float(doc.root_count))
                exact_projected_root = _exact_projected_root_count_from_states(
                    model,
                    leaf_states,
                    schedule="balanced",
                )
                exact_projected_root_abs_sum += abs(
                    float(exact_projected_root) - float(doc.root_count)
                )
                root_decomposition = _exact_markov_root_error_decomposition(
                    truth_root_count=float(doc.root_count),
                    predicted_counts=[
                        float(model.predict_count_from_state(state).detach().cpu().item())
                        for state in leaf_states
                    ],
                    predicted_first=[
                        int(
                            torch.argmax(
                                model._split_state(state)[1],
                                dim=-1,
                            )
                            .detach()
                            .cpu()
                            .item()
                        )
                        for state in leaf_states
                    ],
                    predicted_last=[
                        int(
                            torch.argmax(
                                model._split_state(state)[2],
                                dim=-1,
                            )
                            .detach()
                            .cpu()
                            .item()
                        )
                        for state in leaf_states
                    ],
                    truth_counts=[float(count) for count, _first, _last in exact_targets["leaf"]],
                    truth_first=[int(first) for _count, first, _last in exact_targets["leaf"]],
                    truth_last=[int(last) for _count, _first, last in exact_targets["leaf"]],
                    schedule="balanced",
                )
                root_mae_oracle_counts_predicted_endpoints_abs_sum += float(
                    root_decomposition["root_mae_oracle_counts_predicted_endpoints"]
                )
                root_mae_predicted_counts_oracle_endpoints_abs_sum += float(
                    root_decomposition["root_mae_predicted_counts_oracle_endpoints"]
                )
                task_root_pred = float(
                    model.predict_task_count_from_state(root_state).detach().cpu().item()
                )
                task_root_target = float(doc.root_count)
                if model.use_shared_theorem_surface and feature_targets.root:
                    task_root_target = float(
                        model.task_target_from_label(feature_targets.root[0])
                    )
                task_root_abs_sum += abs(task_root_pred - task_root_target)
                task_factorization_gap_sum += abs(
                    task_root_pred
                    - float(
                        model.predict_canonical_count_from_state(root_state)
                        .detach()
                        .cpu()
                        .item()
                    )
                )
                n_root += 1
                n_task_factorization_gap += 1
                if model.use_shared_theorem_surface and feature_targets.root:
                    _record_phi_feature(
                        adapter=model.theorem_feature_adapter,
                        label=feature_targets.root[0],
                        phi_tensor=model.predict_phi_from_state(root_state).detach(),
                        moments_by_label=phi_moments_by_label,
                        calibration_embeddings=phi_calibration_embeddings,
                        calibration_labels=phi_calibration_labels,
                        phi_pair_calibration_max_nodes=phi_pair_calibration_max_nodes,
                    )

                for leaf_idx, (state, (truth_count, truth_first, truth_last)) in enumerate(
                    zip(leaf_states, exact_targets["leaf"])
                ):
                    pred_count = float(
                        model.predict_count_from_state(state).detach().cpu().item()
                    )
                    leaf_count_abs_sum += abs(pred_count - float(truth_count))
                    _h, first_logits, last_logits = model._split_state(state)
                    pred_first = int(torch.argmax(first_logits, dim=-1).detach().cpu().item())
                    pred_last = int(torch.argmax(last_logits, dim=-1).detach().cpu().item())
                    leaf_first_correct_sum += float(int(pred_first == int(truth_first)))
                    leaf_last_correct_sum += float(int(pred_last == int(truth_last)))
                    abs_diff_key = str(
                        int(abs(int(np.rint(pred_count)) - int(np.rint(float(truth_count)))))
                    )
                    leaf_count_off_by_k_histogram[abs_diff_key] = float(
                        leaf_count_off_by_k_histogram.get(abs_diff_key, 0.0) + 1.0
                    )
                    leaf_exact_match_sum += float(
                        int(
                            int(np.rint(pred_count)) == int(truth_count)
                            and pred_first == int(truth_first)
                            and pred_last == int(truth_last)
                        )
                    )
                    n_leaf += 1
                    if model.use_shared_theorem_surface:
                        _record_phi_feature(
                            adapter=model.theorem_feature_adapter,
                            label=feature_targets.leaf[int(leaf_idx)],
                            phi_tensor=model.predict_phi_from_state(state).detach(),
                            moments_by_label=phi_moments_by_label,
                            calibration_embeddings=phi_calibration_embeddings,
                            calibration_labels=phi_calibration_labels,
                            phi_pair_calibration_max_nodes=phi_pair_calibration_max_nodes,
                        )

                all_states = list(leaf_states) + list(merge_states)
                all_exact = list(exact_targets["leaf"]) + list(exact_targets["merge"])
                children_map = model._balanced_merge_children_map(len(leaf_states))
                depth_layout = model._balanced_tree_layout(len(leaf_states))
                for merge_idx, (state, (truth_count, truth_first, truth_last)) in enumerate(
                    zip(merge_states, exact_targets["merge"])
                ):
                    pred_count = float(
                        model.predict_count_from_state(state).detach().cpu().item()
                    )
                    _h, first_logits, last_logits = model._split_state(state)
                    pred_first = int(torch.argmax(first_logits, dim=-1).detach().cpu().item())
                    pred_last = int(torch.argmax(last_logits, dim=-1).detach().cpu().item())
                    merge_first_correct_sum += float(int(pred_first == int(truth_first)))
                    merge_last_correct_sum += float(int(pred_last == int(truth_last)))
                    merge_exact_value = float(
                        int(
                            int(np.rint(pred_count)) == int(truth_count)
                            and pred_first == int(truth_first)
                            and pred_last == int(truth_last)
                        )
                    )
                    merge_exact_match_sum += float(merge_exact_value)
                    depth_key = str(
                        int(
                            depth_layout["depth_by_global_idx"].get(
                                int(len(leaf_states)) + int(merge_idx),
                                0,
                            )
                        )
                    )
                    merge_exact_match_sum_by_depth[depth_key] = float(
                        merge_exact_match_sum_by_depth.get(depth_key, 0.0)
                        + float(merge_exact_value)
                    )
                    merge_exact_match_count_by_depth[depth_key] = int(
                        merge_exact_match_count_by_depth.get(depth_key, 0) + 1
                    )
                    n_merge += 1
                    if model.use_shared_theorem_surface:
                        _record_phi_feature(
                            adapter=model.theorem_feature_adapter,
                            label=feature_targets.merge[int(merge_idx)],
                            phi_tensor=model.predict_phi_from_state(state).detach(),
                            moments_by_label=phi_moments_by_label,
                            calibration_embeddings=phi_calibration_embeddings,
                            calibration_labels=phi_calibration_labels,
                            phi_pair_calibration_max_nodes=phi_pair_calibration_max_nodes,
                        )

                    child_indices = children_map.get(int(merge_idx))
                    if child_indices is None:
                        continue
                    left_idx, right_idx = child_indices
                    left_state = all_states[int(left_idx)]
                    right_state = all_states[int(right_idx)]
                    left_exact = all_exact[int(left_idx)]
                    right_exact = all_exact[int(right_idx)]
                    truth_join = 0 if int(left_exact[2]) == int(right_exact[1]) else 1
                    if model.use_markov_summary_spec:
                        join_prob = float(
                            model.predict_join_prob_from_states(left_state, right_state)
                            .detach()
                            .cpu()
                            .item()
                        )
                        pred_join = int(join_prob >= 0.5)
                    else:
                        _lh, _left_first, left_last = model._split_state(left_state)
                        _rh, right_first, _right_last = model._split_state(right_state)
                        pred_join = int(
                            int(torch.argmax(left_last, dim=-1).detach().cpu().item())
                            != int(torch.argmax(right_first, dim=-1).detach().cpu().item())
                        )
                    merge_join_correct_sum += float(int(pred_join == truth_join))
                    n_join += 1
                    if model.use_shared_theorem_surface:
                        pred_phi = model.predict_phi_parent_from_children(
                            left_state,
                            right_state,
                        ).detach()
                        target_phi = model.predict_phi_from_state(state).detach()
                        phi_merge_alignment_sum += float(
                            F.cosine_similarity(
                                pred_phi.unsqueeze(0),
                                target_phi.unsqueeze(0),
                                dim=-1,
                            )
                            .cpu()
                            .item()
                        )
                        n_phi_merge_alignment += 1

                if model.use_markov_summary_spec and model.codec_contract is not None:
                    for state in list(leaf_states) + list(merge_states) + [root_state]:
                        decoded = model.codec_contract.decode(state)
                        replay_state = model.codec_contract.reencode(decoded)
                        replay_decoded = model.codec_contract.decode(replay_state)
                        original_label = model.theorem_feature_adapter.oracle_label(
                            count=float(torch.round(decoded.count).detach().cpu().item()),
                            first=int(decoded.first.detach().cpu().item()),
                            last=int(decoded.last.detach().cpu().item()),
                            metadata=None,
                        )
                        replay_label = model.theorem_feature_adapter.oracle_label(
                            count=float(torch.round(replay_decoded.count).detach().cpu().item()),
                            first=int(replay_decoded.first.detach().cpu().item()),
                            last=int(replay_decoded.last.detach().cpu().item()),
                            metadata=None,
                        )
                        replay_same_class_sum += float(
                            int(
                                model.theorem_feature_adapter.same_pair(
                                    original_label,
                                    replay_label,
                                    same_threshold=model.theorem_pair_same_threshold,
                                    diff_threshold=model.theorem_pair_diff_threshold,
                                )
                            )
                        )
                        c2_on_range_exact_sum += float(
                            int(
                                int(torch.round(replay_decoded.count).detach().cpu().item())
                                == int(torch.round(decoded.count).detach().cpu().item())
                                and int(replay_decoded.first.detach().cpu().item())
                                == int(decoded.first.detach().cpu().item())
                                and int(replay_decoded.last.detach().cpu().item())
                                == int(decoded.last.detach().cpu().item())
                            )
                        )
                        n_replay += 1
                        n_c2 += 1
        if batching_metrics is not None:
            batching_metrics.eval_time_s += time.perf_counter() - eval_start_s
            batching_metrics.add_batch(
                packed_batch,
                token_budget=int(token_budget),
                node_budget=int(node_budget),
                max_docs_budget=(
                    int(max_docs) if int(max_docs) > 0 else max(1, int(len(packed_batch.items)))
                ),
            )
        _emit_memory_probe(
            "post_exact_eval_batch",
            batch_index=int(batch_index),
            batch_docs=int(batch_doc_count),
            max_docs=int(packed_max_docs),
            pack_mode=str(pack_mode),
            runtime_bucket_mode=str(runtime_bucket_mode),
            padded_leaf_tokens=int(packed_batch.padded_leaf_tokens),
            total_nodes=int(packed_batch.total_nodes),
        )
        _trim_host_allocator()
        _emit_memory_probe(
            "post_exact_eval_batch_trim",
            batch_index=int(batch_index),
            batch_docs=int(batch_doc_count),
            max_docs=int(packed_max_docs),
            pack_mode=str(pack_mode),
            runtime_bucket_mode=str(runtime_bucket_mode),
            padded_leaf_tokens=int(packed_batch.padded_leaf_tokens),
            total_nodes=int(packed_batch.total_nodes),
        )

    phi_within_terms: List[float] = []
    class_centroids: List[np.ndarray] = []
    for moments in phi_moments_by_label.values():
        centroid = moments.centroid()
        if centroid is None:
            continue
        class_centroids.append(centroid)
        phi_within_terms.append(moments.mean_sq_distance())
    phi_between_terms: List[float] = []
    for left_idx in range(len(class_centroids)):
        for right_idx in range(left_idx + 1, len(class_centroids)):
            phi_between_terms.append(
                float(
                    np.linalg.norm(
                        class_centroids[int(left_idx)] - class_centroids[int(right_idx)]
                    )
                )
            )
    pair_same_scores: List[float] = []
    pair_diff_scores: List[float] = []
    if model.use_shared_theorem_surface and len(phi_calibration_embeddings) > 1:
        phi_pairs = build_theorem_feature_pair_sets(
            phi_calibration_labels,
            adapter=model.theorem_feature_adapter,
            same_threshold=model.theorem_pair_same_threshold,
            diff_threshold=model.theorem_pair_diff_threshold,
        )
        if phi_pairs.total_pairs > 0:
            phi_array = np.asarray(phi_calibration_embeddings, dtype=np.float64)
            phi_norm = np.linalg.norm(phi_array, axis=1, keepdims=True)
            phi_norm = np.maximum(phi_norm, 1e-12)
            phi_unit = phi_array / phi_norm
            for left_idx, right_idx in phi_pairs.same_pairs:
                pair_same_scores.append(
                    float(np.dot(phi_unit[int(left_idx)], phi_unit[int(right_idx)]))
                )
            for left_idx, right_idx in phi_pairs.different_pairs:
                pair_diff_scores.append(
                    float(np.dot(phi_unit[int(left_idx)], phi_unit[int(right_idx)]))
                )
    pair_metrics = theorem_feature_pair_metrics_from_scores(
        same_scores=pair_same_scores,
        different_scores=pair_diff_scores,
    ).as_dict()

    root_direct_count_mae = _mean_or_default(total=root_abs_sum, count=n_root, default=0.0)
    exact_projected_root_mae = _mean_or_default(
        total=exact_projected_root_abs_sum,
        count=n_root,
        default=0.0,
    )
    root_mae_oracle_counts_predicted_endpoints = _mean_or_default(
        total=root_mae_oracle_counts_predicted_endpoints_abs_sum,
        count=n_root,
        default=0.0,
    )
    root_mae_predicted_counts_oracle_endpoints = _mean_or_default(
        total=root_mae_predicted_counts_oracle_endpoints_abs_sum,
        count=n_root,
        default=0.0,
    )
    task_root_mae = _mean_or_default(total=task_root_abs_sum, count=n_root, default=0.0)
    leaf_direct_count_mae = _mean_or_default(
        total=leaf_count_abs_sum,
        count=n_leaf,
        default=0.0,
    )
    leaf_direct_exact_match = _mean_or_default(
        total=leaf_exact_match_sum,
        count=n_leaf,
        default=1.0,
    )
    leaf_first_accuracy = _mean_or_default(
        total=leaf_first_correct_sum,
        count=n_leaf,
        default=1.0,
    )
    leaf_last_accuracy = _mean_or_default(
        total=leaf_last_correct_sum,
        count=n_leaf,
        default=1.0,
    )
    merge_direct_exact_match = _mean_or_default(
        total=merge_exact_match_sum,
        count=n_merge,
        default=1.0,
    )
    merge_first_accuracy = _mean_or_default(
        total=merge_first_correct_sum,
        count=n_merge,
        default=1.0,
    )
    merge_last_accuracy = _mean_or_default(
        total=merge_last_correct_sum,
        count=n_merge,
        default=1.0,
    )
    merge_join_bit_accuracy = _mean_or_default(
        total=merge_join_correct_sum,
        count=n_join,
        default=1.0,
    )
    c2_on_range_exact_match = _mean_or_default(
        total=c2_on_range_exact_sum,
        count=n_c2,
        default=1.0,
    )
    phi_merge_alignment = _mean_or_default(
        total=phi_merge_alignment_sum,
        count=n_phi_merge_alignment,
        default=1.0,
    )
    phi_within_class_variance = (
        float(np.mean(np.asarray(phi_within_terms, dtype=np.float64)))
        if phi_within_terms
        else 0.0
    )
    phi_between_class_margin = (
        float(np.mean(np.asarray(phi_between_terms, dtype=np.float64)))
        if phi_between_terms
        else 0.0
    )
    phi_replay_same_class_rate = _mean_or_default(
        total=replay_same_class_sum,
        count=n_replay,
        default=float("nan"),
    )
    task_factorization_gap = _mean_or_default(
        total=task_factorization_gap_sum,
        count=n_task_factorization_gap,
        default=0.0,
    )
    leaf_codec_selection_value = float(
        leaf_direct_count_mae
        + (1.0 - leaf_direct_exact_match)
        + (1.0 - merge_join_bit_accuracy)
        + (1.0 - c2_on_range_exact_match)
    )
    theorem_bootstrap_selection_value = float(
        leaf_direct_count_mae
        + (1.0 - leaf_direct_exact_match)
        + (1.0 - merge_direct_exact_match)
        + (1.0 - merge_join_bit_accuracy)
        + (1.0 - c2_on_range_exact_match)
    )
    selection_root_mae = _exact_sketch_selection_root_mae(
        model,
        root_direct_count_mae=float(root_direct_count_mae),
        exact_projected_root_mae=float(exact_projected_root_mae),
    )
    selection_split_penalty = _exact_sketch_selection_split_penalty(
        model,
        root_mae_oracle_counts_predicted_endpoints=float(
            root_mae_oracle_counts_predicted_endpoints
        ),
        root_mae_predicted_counts_oracle_endpoints=float(
            root_mae_predicted_counts_oracle_endpoints
        ),
    )
    selection_value = float(
        selection_root_mae
        + selection_split_penalty
        + (1.0 - leaf_direct_exact_match)
        + (1.0 - merge_direct_exact_match)
        + (1.0 - merge_join_bit_accuracy)
    )
    task_root_selection_value = float(
        task_root_mae
        + (1.0 - leaf_direct_exact_match)
        + (1.0 - merge_direct_exact_match)
        + (1.0 - merge_join_bit_accuracy)
    )
    return {
        "root_direct_count_mae": float(root_direct_count_mae),
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
        "learned_merger_gap": float(root_direct_count_mae - exact_projected_root_mae),
        "task_root_mae": float(task_root_mae),
        "task_root_mae_ablation": float(task_root_mae),
        "leaf_direct_count_mae": float(leaf_direct_count_mae),
        "leaf_direct_exact_match": float(leaf_direct_exact_match),
        "leaf_first_accuracy": float(leaf_first_accuracy),
        "leaf_last_accuracy": float(leaf_last_accuracy),
        "merge_first_accuracy": float(merge_first_accuracy),
        "merge_last_accuracy": float(merge_last_accuracy),
        "leaf_count_off_by_k_histogram": _normalize_count_histogram(
            leaf_count_off_by_k_histogram
        ),
        "merge_exact_summary_match_rate_by_depth": _merge_rate_dict_from_sums(
            merge_exact_match_sum_by_depth,
            merge_exact_match_count_by_depth,
        ),
        "merge_direct_exact_match": float(merge_direct_exact_match),
        "merge_join_bit_accuracy": float(merge_join_bit_accuracy),
        "c2_on_range_exact_match": float(c2_on_range_exact_match),
        "phi_merge_alignment": float(phi_merge_alignment),
        "phi_within_class_variance": float(phi_within_class_variance),
        "phi_between_class_margin": float(phi_between_class_margin),
        "phi_pair_same_accuracy": float(pair_metrics["phi_pair_same_accuracy"]),
        "phi_pair_diff_accuracy": float(pair_metrics["phi_pair_diff_accuracy"]),
        "phi_pair_auc": float(pair_metrics["phi_pair_auc"]),
        "phi_replay_same_class_rate": float(phi_replay_same_class_rate),
        "task_factorization_gap": float(task_factorization_gap),
        "val_leaf_codec_direct": float(leaf_codec_selection_value),
        "val_theorem_bootstrap_direct": float(theorem_bootstrap_selection_value),
        "val_exact_sketch_direct": float(selection_value),
        "val_task_root_exact_sketch_direct": float(task_root_selection_value),
        "n_docs": float(len(docs)),
        "n_leaf_nodes": float(n_leaf),
        "n_merge_nodes": float(n_merge),
    }


@torch.no_grad()
def _collect_teacher_first_node_view(
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
    pack_mode: str = "structure_bucket",
    runtime_bucket_mode: str = GPU_RUNTIME_BUCKET_MODE_EXACT_THEN_BUCKETED,
    max_docs: int = 0,
    token_budget: int = 0,
    node_budget: int = 0,
    bucket_docs_cap_by_n_leaves: Mapping[int, int] | None = None,
    structural_pad_limit: float = 0.5,
    auto_queue_min_docs: int = 8,
    auto_queue_min_fill_ratio: float = 0.5,
    auto_queue_target_by_n_leaves: Mapping[int, int] | None = None,
    batching_metrics: _BatchingMetricsAccumulator | None = None,
) -> Dict[str, Tuple[float, ...] | Tuple[np.ndarray, ...]]:
    root_task_predictions: List[float] = []
    root_true_targets: List[float] = []
    leaf_task_predictions: List[float] = []
    merge_task_predictions: List[float] = []
    node_task_predictions: List[float] = []
    phi_embeddings: List[np.ndarray] = []

    work_items = [
        _tree_work_item_from_doc(
            doc,
            doc_index=int(idx),
            work_kind="full_tree",
            collect_leaf=True,
            collect_c2=True,
            collect_c3=True,
        )
        for idx, doc in enumerate(docs)
    ]
    packed_batches = _pack_tree_work_items(
        work_items,
        max_docs=int(max_docs) if int(max_docs) > 0 else len(work_items),
        max_total_leaf_tokens=int(token_budget),
        max_total_nodes=int(node_budget),
        max_total_merge_ops=0,
        bucket_docs_cap_by_n_leaves=bucket_docs_cap_by_n_leaves,
        bucket_mode=_effective_tree_bucket_mode(
            pack_mode=str(pack_mode),
            bucket_mode=str(runtime_bucket_mode),
        ),
        structural_pad_limit=float(structural_pad_limit),
        auto_queue_min_docs=int(auto_queue_min_docs),
        auto_queue_target_by_n_leaves=auto_queue_target_by_n_leaves,
        tail_repack_fill_ratio=(
            float(auto_queue_min_fill_ratio)
            if str(pack_mode or "structure_bucket").strip().lower() == "fixed_fused"
            else 0.0
        ),
        tail_repack_min_docs=int(auto_queue_min_docs),
    )

    with _autocast_context(device):
        for packed_batch in packed_batches:
            eval_start_s = time.perf_counter()
            batch_docs = [item.doc for item in packed_batch.items]
            use_fixed_fused = (
                str(pack_mode or "structure_bucket").strip().lower() == "fixed_fused"
                and (_docs_share_fixed_leaf_shape(batch_docs) or _docs_support_fixed_leaf_auto_queue(batch_docs))
            )
            if use_fixed_fused:
                levels = _precompute_balanced_doc_state_levels(
                    model,
                    batch_docs,
                    device=device,
                    collect_merge_states=True,
                    prefer_fixed_fused=True,
                    target_n_leaves=int(packed_batch.bucket_key.n_leaves),
                )
                merge_state_batch = _flatten_merge_state_batch(levels.merge_levels)
                if merge_state_batch is None:
                    merge_state_batch = levels.leaf_states.new_zeros(
                        (
                            int(levels.leaf_states.shape[0]),
                            0,
                            int(levels.leaf_states.shape[-1]),
                        )
                    )
                merge_valid_mask = (
                    torch.cat(
                        [mask for mask in levels.merge_valid_levels if int(mask.shape[1]) > 0],
                        dim=1,
                    )
                    if levels.merge_valid_levels
                    else torch.zeros(
                        (int(levels.leaf_states.shape[0]), 0),
                        device=device,
                        dtype=torch.bool,
                    )
                )
                root_task_batch = model.predict_task_count_from_state(levels.root_states)
                leaf_task_batch = model.predict_task_count_from_state(
                    levels.leaf_states.reshape(-1, int(levels.leaf_states.shape[-1]))
                ).reshape(int(levels.leaf_states.shape[0]), int(levels.leaf_states.shape[1]))
                merge_task_batch = (
                    model.predict_task_count_from_state(
                        merge_state_batch.reshape(-1, int(merge_state_batch.shape[-1]))
                    ).reshape(int(merge_state_batch.shape[0]), int(merge_state_batch.shape[1]))
                    if int(merge_state_batch.shape[1]) > 0
                    else merge_state_batch.new_zeros(
                        (int(merge_state_batch.shape[0]), 0)
                    )
                )
                leaf_phi_batch = (
                    model.predict_phi_from_state(
                        levels.leaf_states.reshape(-1, int(levels.leaf_states.shape[-1]))
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(int(levels.leaf_states.shape[0]), int(levels.leaf_states.shape[1]), -1)
                ) if model.use_shared_theorem_surface else None
                merge_phi_batch = (
                    model.predict_phi_from_state(
                        merge_state_batch.reshape(-1, int(merge_state_batch.shape[-1]))
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(int(merge_state_batch.shape[0]), int(merge_state_batch.shape[1]), -1)
                    if model.use_shared_theorem_surface and int(merge_state_batch.shape[1]) > 0
                    else None
                )
                root_phi_batch = (
                    model.predict_phi_from_state(levels.root_states).detach().cpu().numpy()
                    if model.use_shared_theorem_surface
                    else None
                )
            else:
                views = _precompute_balanced_doc_state_views(
                    model,
                    batch_docs,
                    device=device,
                    collect_merge_states=True,
                    prefer_fixed_fused=False,
                )
            for batch_idx, doc in enumerate(batch_docs):
                if use_fixed_fused:
                    leaf_state_batch = levels.leaf_states[int(batch_idx)]
                    leaf_states = [
                        leaf_state_batch[idx]
                        for idx in range(int(leaf_state_batch.shape[0]))
                        if bool(levels.leaf_valid_mask[int(batch_idx), int(idx)].item())
                    ]
                    root_state = levels.root_states[int(batch_idx)]
                    merge_states = [
                        merge_state_batch[int(batch_idx), idx]
                        for idx in range(int(merge_state_batch.shape[1]))
                        if bool(merge_valid_mask[int(batch_idx), int(idx)].item())
                    ]
                else:
                    precomputed_view = views[int(batch_idx)]
                    leaf_state_batch = precomputed_view.state_batch
                    leaf_states = [
                        leaf_state_batch[idx]
                        for idx in range(int(leaf_state_batch.shape[0]))
                    ]
                    root_state = precomputed_view.root_state
                    merge_states = list(precomputed_view.merge_states)
                exact_targets = _balanced_exact_sketch_targets(
                    leaf_counts=doc.leaf_counts,
                    leaf_first_regimes=doc.leaf_first_regimes,
                    leaf_last_regimes=doc.leaf_last_regimes,
                )
                leaf_metadata, merge_metadata, root_metadata = (
                    _theorem_feature_metadata_sequences_from_fno_doc(doc)
                )
                feature_targets = theorem_feature_targets_from_markov_exact_targets(
                    adapter=model.theorem_feature_adapter,
                    exact_targets=exact_targets,
                    leaf_metadata=leaf_metadata,
                    merge_metadata=merge_metadata,
                    root_metadata=root_metadata,
                )
                root_true_target = float(doc.root_count)
                if model.use_shared_theorem_surface and feature_targets.root:
                    root_true_target = float(
                        model.task_target_from_label(feature_targets.root[0])
                    )
                if use_fixed_fused:
                    root_state_task_prediction = float(
                        root_task_batch[int(batch_idx)].detach().cpu().item()
                    )
                else:
                    root_state_task_prediction = float(
                        model.predict_task_count_from_state(root_state).detach().cpu().item()
                    )
                root_task_predictions.append(root_state_task_prediction)
                root_true_targets.append(root_true_target)

                if len(leaf_states) > 0:
                    if use_fixed_fused:
                        leaf_task_prediction_batch = leaf_task_batch[int(batch_idx)][
                            levels.leaf_valid_mask[int(batch_idx)]
                        ]
                    else:
                        leaf_task_prediction_batch = model.predict_task_count_from_state(
                            leaf_state_batch
                        )
                    leaf_task_prediction_list = [
                        float(value)
                        for value in leaf_task_prediction_batch.detach().cpu().reshape(-1).tolist()
                    ]
                    leaf_task_predictions.extend(leaf_task_prediction_list)
                    node_task_predictions.extend(leaf_task_prediction_list)
                    if model.use_shared_theorem_surface:
                        phi_batch = (
                            leaf_phi_batch[int(batch_idx)][
                                levels.leaf_valid_mask[int(batch_idx)].detach().cpu().numpy().astype(bool)
                            ]
                            if use_fixed_fused and leaf_phi_batch is not None
                            else model.predict_phi_from_state(leaf_state_batch)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        phi_embeddings.extend(list(phi_batch))

                if len(merge_states) > 0:
                    if use_fixed_fused:
                        merge_local_batch = merge_state_batch[int(batch_idx)][
                            merge_valid_mask[int(batch_idx)]
                        ]
                        merge_task_prediction_batch = merge_task_batch[int(batch_idx)][
                            merge_valid_mask[int(batch_idx)]
                        ]
                    else:
                        merge_local_batch = torch.stack(list(merge_states), dim=0)
                        merge_task_prediction_batch = model.predict_task_count_from_state(
                            merge_local_batch
                        )
                    merge_task_prediction_list = [
                        float(value)
                        for value in merge_task_prediction_batch.detach().cpu().reshape(-1).tolist()
                    ]
                    merge_task_predictions.extend(merge_task_prediction_list)
                    node_task_predictions.extend(merge_task_prediction_list)
                    if model.use_shared_theorem_surface:
                        phi_batch = (
                            merge_phi_batch[int(batch_idx)][
                                merge_valid_mask[int(batch_idx)].detach().cpu().numpy().astype(bool)
                            ]
                            if use_fixed_fused and merge_phi_batch is not None
                            else model.predict_phi_from_state(merge_local_batch)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        phi_embeddings.extend(list(phi_batch))

                node_task_predictions.append(root_state_task_prediction)
                if model.use_shared_theorem_surface:
                    phi_embeddings.append(
                        (
                            root_phi_batch[int(batch_idx)]
                            if use_fixed_fused and root_phi_batch is not None
                            else model.predict_phi_from_state(root_state).detach().cpu().numpy()
                        )
                    )
            if batching_metrics is not None:
                batching_metrics.eval_time_s += time.perf_counter() - eval_start_s
                batching_metrics.add_batch(
                    packed_batch,
                    token_budget=int(token_budget),
                    node_budget=int(node_budget),
                    max_docs_budget=(
                        int(max_docs)
                        if int(max_docs) > 0
                        else max(1, int(len(packed_batch.items)))
                    ),
                )

    return {
        "root_task_predictions": tuple(float(x) for x in root_task_predictions),
        "root_true_targets": tuple(float(x) for x in root_true_targets),
        "leaf_task_predictions": tuple(float(x) for x in leaf_task_predictions),
        "merge_task_predictions": tuple(float(x) for x in merge_task_predictions),
        "node_task_predictions": tuple(float(x) for x in node_task_predictions),
        "phi_embeddings": tuple(phi_embeddings),
    }


DEFAULT_TEACHER_FIRST_PAIR_METRIC_MAX_NODES = 2048
DEFAULT_TEACHER_FIRST_PAIR_METRIC_SAMPLE_SEED = 0


def _teacher_first_pair_metric_sample_indices(
    n_items: int,
    *,
    max_nodes: int,
) -> Tuple[int, ...] | None:
    n_total = int(max(0, n_items))
    n_cap = int(max_nodes)
    if n_total <= 1 or n_cap <= 0 or n_total <= n_cap:
        return None
    sampled = _deterministic_sample_indices(
        n_items=int(n_total),
        rate=float(n_cap) / float(n_total),
        seed=DEFAULT_TEACHER_FIRST_PAIR_METRIC_SAMPLE_SEED,
    )
    if not sampled:
        return None
    trimmed = tuple(sorted(int(idx) for idx in tuple(sampled)[: int(n_cap)]))
    return trimmed if len(trimmed) > 1 else None


def _teacher_first_pair_metrics_from_node_view(
    *,
    stage1_node_scores: Sequence[float],
    final_phi_embeddings: Sequence[np.ndarray],
    same_threshold: float | None,
    diff_threshold: float | None,
    max_nodes: int = DEFAULT_TEACHER_FIRST_PAIR_METRIC_MAX_NODES,
) -> Dict[str, float]:
    total_node_count = int(len(stage1_node_scores))
    empty_payload = {
        "stage2_fiber_error": float("nan"),
        "stage2_fiber_pair_same_accuracy": float("nan"),
        "stage2_fiber_pair_diff_accuracy": float("nan"),
        "stage2_fiber_pair_auc": float("nan"),
        "stage2_fiber_pair_count_same": 0.0,
        "stage2_fiber_pair_count_diff": 0.0,
        "stage2_fiber_pair_sampled_node_count": float(min(total_node_count, max(0, int(max_nodes)))),
        "stage2_fiber_pair_total_node_count": float(total_node_count),
        "stage2_fiber_pair_sampled_pair_count": 0.0,
    }
    if total_node_count <= 1 or len(final_phi_embeddings) != total_node_count:
        return empty_payload

    sampled_indices = _teacher_first_pair_metric_sample_indices(
        int(total_node_count),
        max_nodes=int(max_nodes),
    )
    if sampled_indices is None:
        score_arr = np.asarray(stage1_node_scores, dtype=np.float64)
        phi_array = np.asarray(final_phi_embeddings, dtype=np.float64)
    else:
        score_arr = np.asarray(
            [float(stage1_node_scores[int(idx)]) for idx in sampled_indices],
            dtype=np.float64,
        )
        phi_array = np.asarray(
            [final_phi_embeddings[int(idx)] for idx in sampled_indices],
            dtype=np.float64,
        )

    sampled_node_count = int(score_arr.shape[0])
    empty_payload["stage2_fiber_pair_sampled_node_count"] = float(sampled_node_count)
    if sampled_node_count <= 1 or phi_array.ndim != 2 or int(phi_array.shape[0]) != sampled_node_count:
        return empty_payload

    tri_i, tri_j = np.triu_indices(sampled_node_count, k=1)
    if tri_i.size <= 0:
        return empty_payload

    pair_score_gaps = np.abs(score_arr[tri_i] - score_arr[tri_j])
    resolved_same, resolved_diff = _default_teacher_first_pair_thresholds_from_pair_gaps(
        pair_score_gaps,
        same_threshold=same_threshold,
        diff_threshold=diff_threshold,
    )
    phi_norm = np.linalg.norm(phi_array, axis=1, keepdims=True)
    phi_norm = np.maximum(phi_norm, 1e-12)
    phi_unit = phi_array / phi_norm
    pair_cosine_scores = np.einsum(
        "ij,ij->i",
        phi_unit[tri_i],
        phi_unit[tri_j],
        optimize=True,
    )
    same_scores = pair_cosine_scores[pair_score_gaps <= float(resolved_same)]
    diff_scores = pair_cosine_scores[pair_score_gaps >= float(resolved_diff)]
    pair_metrics = theorem_feature_pair_metrics_from_scores(
        same_scores=same_scores.tolist(),
        different_scores=diff_scores.tolist(),
    ).as_dict()
    stage2_fiber_pair_auc = float(pair_metrics["phi_pair_auc"])
    stage2_fiber_error = (
        float(max(0.0, 1.0 - stage2_fiber_pair_auc))
        if np.isfinite(stage2_fiber_pair_auc)
        else float("nan")
    )
    return {
        "stage2_fiber_error": float(stage2_fiber_error),
        "stage2_fiber_pair_same_accuracy": float(pair_metrics["phi_pair_same_accuracy"]),
        "stage2_fiber_pair_diff_accuracy": float(pair_metrics["phi_pair_diff_accuracy"]),
        "stage2_fiber_pair_auc": float(stage2_fiber_pair_auc),
        "stage2_fiber_pair_count_same": float(same_scores.size),
        "stage2_fiber_pair_count_diff": float(diff_scores.size),
        "stage2_fiber_pair_sampled_node_count": float(sampled_node_count),
        "stage2_fiber_pair_total_node_count": float(total_node_count),
        "stage2_fiber_pair_sampled_pair_count": float(pair_score_gaps.size),
    }


def _default_teacher_first_pair_thresholds_from_pair_gaps(
    pair_gaps: Sequence[float] | np.ndarray,
    *,
    same_threshold: float | None,
    diff_threshold: float | None,
) -> Tuple[float, float]:
    pair_distance_arr = np.asarray(pair_gaps, dtype=np.float64).reshape(-1)
    if pair_distance_arr.size > 0:
        inferred_same = float(np.quantile(pair_distance_arr, 0.2))
        inferred_diff = float(np.quantile(pair_distance_arr, 0.8))
    else:
        inferred_same = 0.0
        inferred_diff = 0.0
    resolved_same = (
        float(same_threshold) if same_threshold is not None else float(inferred_same)
    )
    resolved_diff = (
        float(diff_threshold) if diff_threshold is not None else float(inferred_diff)
    )
    if resolved_diff < resolved_same:
        resolved_diff = float(resolved_same)
    return float(max(0.0, resolved_same)), float(max(resolved_same, resolved_diff))


def _default_teacher_first_pair_thresholds(
    values: Sequence[float],
    *,
    same_threshold: float | None,
    diff_threshold: float | None,
) -> Tuple[float, float]:
    pair_distances: List[float] = []
    data = [float(x) for x in values]
    for left_idx in range(len(data)):
        for right_idx in range(left_idx + 1, len(data)):
            pair_distances.append(abs(float(data[left_idx]) - float(data[right_idx])))
    if pair_distances:
        pair_distance_arr = np.asarray(pair_distances, dtype=np.float64)
        inferred_same = float(np.quantile(pair_distance_arr, 0.2))
        inferred_diff = float(np.quantile(pair_distance_arr, 0.8))
    else:
        inferred_same = 0.0
        inferred_diff = 0.0
    resolved_same = (
        float(same_threshold) if same_threshold is not None else float(inferred_same)
    )
    resolved_diff = (
        float(diff_threshold) if diff_threshold is not None else float(inferred_diff)
    )
    if resolved_diff < resolved_same:
        resolved_diff = float(resolved_same)
    return float(max(0.0, resolved_same)), float(max(resolved_same, resolved_diff))


def _mae_between_sequences(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.asarray(list(left), dtype=np.float64)
    right_arr = np.asarray(list(right), dtype=np.float64)
    if left_arr.size <= 0 or right_arr.size <= 0 or left_arr.shape != right_arr.shape:
        return float("nan")
    return float(np.mean(np.abs(left_arr - right_arr)))


def _write_teacher_first_view_artifacts(
    *,
    artifact_dir: str | Path,
    final_view: Mapping[str, Sequence[float] | Sequence[np.ndarray]],
    stage1_view: Mapping[str, Sequence[float] | Sequence[np.ndarray]],
) -> Dict[str, str]:
    root = Path(str(artifact_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    def _serialize_view(view: Mapping[str, Sequence[float] | Sequence[np.ndarray]]) -> Dict[str, np.ndarray]:
        payload: Dict[str, np.ndarray] = {}
        for key, values in dict(view).items():
            if str(key) == "phi_embeddings":
                phi_values = list(values)  # type: ignore[arg-type]
                payload[str(key)] = (
                    np.asarray(phi_values, dtype=np.float64)
                    if phi_values
                    else np.zeros((0, 0), dtype=np.float64)
                )
            else:
                payload[str(key)] = np.asarray(list(values), dtype=np.float64)
        return payload

    final_npz = root / "final_view.npz"
    stage1_npz = root / "stage1_view.npz"
    np.savez_compressed(final_npz, **_serialize_view(final_view))
    np.savez_compressed(stage1_npz, **_serialize_view(stage1_view))
    metadata_path = root / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "final_npz": str(final_npz),
                "stage1_npz": str(stage1_npz),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "final_npz": str(final_npz),
        "stage1_npz": str(stage1_npz),
        "metadata_json": str(metadata_path),
    }


@torch.no_grad()
def _eval_fno_teacher_first_decomposition_metrics(
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
    stage1_model_state: Mapping[str, Any],
    diagnostic_detail_mode: str = "summary",
    raw_artifact_dir: str | Path | None = None,
) -> Dict[str, Any]:
    if len(docs) <= 0:
        return {
            "stage2_transport_budget": 0.0,
            "stage2_leaf_transport_mae": 0.0,
            "stage2_merge_transport_mae": 0.0,
            "stage2_fiber_error": 0.0,
            "stage2_fiber_pair_same_accuracy": float("nan"),
            "stage2_fiber_pair_diff_accuracy": float("nan"),
            "stage2_fiber_pair_auc": float("nan"),
            "root_measurement_error": 0.0,
            "stage1_substitution_cost": 0.0,
            "teacher_first_total_bound": 0.0,
            "stage2_fiber_pair_count_same": 0.0,
            "stage2_fiber_pair_count_diff": 0.0,
        }

    final_state = clone_module_state(model)
    final_view = _collect_teacher_first_node_view(model, docs, device=device)
    restore_module_state(model, stage1_model_state)
    stage1_view = _collect_teacher_first_node_view(model, docs, device=device)
    restore_module_state(model, final_state)

    stage2_leaf_transport_mae = _mae_between_sequences(
        final_view["leaf_task_predictions"],
        stage1_view["leaf_task_predictions"],
    )
    stage2_merge_transport_mae = _mae_between_sequences(
        final_view["merge_task_predictions"],
        stage1_view["merge_task_predictions"],
    )
    root_measurement_error = _mae_between_sequences(
        final_view["root_task_predictions"],
        stage1_view["root_task_predictions"],
    )
    stage1_substitution_cost = _mae_between_sequences(
        stage1_view["root_task_predictions"],
        stage1_view["root_true_targets"],
    )
    stage2_transport_budget = float(
        np.nansum(
            np.asarray(
                [stage2_leaf_transport_mae, stage2_merge_transport_mae],
                dtype=np.float64,
            )
        )
    )

    pair_metric_payload = _teacher_first_pair_metrics_from_node_view(
        stage1_node_scores=stage1_view["node_task_predictions"],
        final_phi_embeddings=final_view["phi_embeddings"],
        same_threshold=getattr(model, "theorem_pair_same_threshold", None),
        diff_threshold=getattr(model, "theorem_pair_diff_threshold", None),
    )
    stage2_fiber_error = float(pair_metric_payload["stage2_fiber_error"])
    stage2_fiber_pair_auc = float(pair_metric_payload["stage2_fiber_pair_auc"])
    teacher_first_total_bound = (
        float(
            stage2_transport_budget
            + stage2_fiber_error
            + root_measurement_error
            + stage1_substitution_cost
        )
        if np.all(
            np.isfinite(
                np.asarray(
                    [
                        stage2_transport_budget,
                        stage2_fiber_error,
                        root_measurement_error,
                        stage1_substitution_cost,
                    ],
                    dtype=np.float64,
                )
            )
        )
        else float("nan")
    )
    payload = {
        "stage2_transport_budget": float(stage2_transport_budget),
        "stage2_leaf_transport_mae": float(stage2_leaf_transport_mae),
        "stage2_merge_transport_mae": float(stage2_merge_transport_mae),
        "stage2_fiber_error": float(stage2_fiber_error),
        "stage2_fiber_pair_same_accuracy": float(
            pair_metric_payload["stage2_fiber_pair_same_accuracy"]
        ),
        "stage2_fiber_pair_diff_accuracy": float(
            pair_metric_payload["stage2_fiber_pair_diff_accuracy"]
        ),
        "stage2_fiber_pair_auc": float(stage2_fiber_pair_auc),
        "root_measurement_error": float(root_measurement_error),
        "stage1_substitution_cost": float(stage1_substitution_cost),
        "teacher_first_total_bound": float(teacher_first_total_bound),
        "stage2_fiber_pair_count_same": float(
            pair_metric_payload["stage2_fiber_pair_count_same"]
        ),
        "stage2_fiber_pair_count_diff": float(
            pair_metric_payload["stage2_fiber_pair_count_diff"]
        ),
        "stage2_fiber_pair_sampled_node_count": float(
            pair_metric_payload["stage2_fiber_pair_sampled_node_count"]
        ),
        "stage2_fiber_pair_total_node_count": float(
            pair_metric_payload["stage2_fiber_pair_total_node_count"]
        ),
        "stage2_fiber_pair_sampled_pair_count": float(
            pair_metric_payload["stage2_fiber_pair_sampled_pair_count"]
        ),
    }
    if (
        str(diagnostic_detail_mode).strip().lower() == "debug_raw"
        and raw_artifact_dir is not None
    ):
        payload["raw_diagnostic_artifacts"] = _write_teacher_first_view_artifacts(
            artifact_dir=raw_artifact_dir,
            final_view=final_view,
            stage1_view=stage1_view,
        )
    del final_view
    del stage1_view
    _trim_host_allocator()
    return payload


@torch.inference_mode()
def _eval_fno_root_only_metrics(
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
    resident_store: GpuBatchStore | None = None,
    runtime_telemetry: GpuRuntimeTelemetry | None = None,
    pack_mode: str = "structure_bucket",
    runtime_bucket_mode: str = GPU_RUNTIME_BUCKET_MODE_EXACT_THEN_BUCKETED,
    max_docs: int = 0,
    token_budget: int = 0,
    node_budget: int = 0,
    bucket_docs_cap_by_n_leaves: Mapping[int, int] | None = None,
    structural_pad_limit: float = 0.5,
    auto_queue_min_docs: int = 8,
    auto_queue_min_fill_ratio: float = 0.5,
    auto_queue_target_by_n_leaves: Mapping[int, int] | None = None,
    batching_metrics: _BatchingMetricsAccumulator | None = None,
) -> SketchMetrics:
    """Cheap root-only evaluator for stage-1 screening and final MAE checks."""
    if len(docs) == 0:
        return _zero_sketch_metrics(n_docs=0)

    preds, truths = _batched_root_predictions(
        model,
        docs,
        device=device,
        resident_store=resident_store,
        runtime_telemetry=runtime_telemetry,
        pack_mode=pack_mode,
        runtime_bucket_mode=runtime_bucket_mode,
        max_docs=max_docs,
        token_budget=token_budget,
        node_budget=node_budget,
        bucket_docs_cap_by_n_leaves=bucket_docs_cap_by_n_leaves,
        structural_pad_limit=structural_pad_limit,
        auto_queue_min_docs=auto_queue_min_docs,
        auto_queue_min_fill_ratio=auto_queue_min_fill_ratio,
        auto_queue_target_by_n_leaves=auto_queue_target_by_n_leaves,
        batching_metrics=batching_metrics,
    )
    root_abs_arr = np.abs(preds - truths)
    root_sq_arr = (preds - truths) ** 2
    _nan = float("nan")
    return SketchMetrics(
        root_mae=float(np.mean(root_abs_arr)),
        root_mse=float(np.mean(root_sq_arr)),
        root_median_abs_error=float(np.median(root_abs_arr)),
        root_p95_abs_error=float(np.percentile(root_abs_arr, 95.0)),
        schedule_spread_mean=_nan,
        schedule_spread_p95=_nan,
        # Local-law metrics not applicable for root-only evaluation.
        leaf_mae=_nan,
        leaf_violation_rate=_nan,
        c2_idempotence_mae=_nan,
        c2_r1_mae=_nan,
        c2_r2_mae=_nan,
        c2_r4_mae=_nan,
        resummary_root_drift_r1=_nan,
        resummary_root_drift_r2=_nan,
        resummary_root_drift_r4=_nan,
        merge_mae=_nan,
        merge_violation_rate=_nan,
        n_docs=int(len(docs)),
        c2_state_replay_mse=0.0,
    )


def _normalize_tree_document_loss_normalization_mode(mode: str | None) -> str:
    normalized = str(mode or "auto").strip().lower() or "auto"
    if normalized not in VALID_TREE_DOCUMENT_LOSS_NORMALIZATION_MODES:
        raise ValueError(
            "tree_document_loss_normalization_mode must be one of "
            f"{VALID_TREE_DOCUMENT_LOSS_NORMALIZATION_MODES}; got {mode!r}"
        )
    return normalized


def _effective_tree_document_loss_normalization_mode(
    mode: str | None,
    *,
    explicit_doc_modes: Mapping[int, str] | None,
) -> str:
    normalized = _normalize_tree_document_loss_normalization_mode(mode)
    if normalized != "auto":
        return normalized
    return "supervised_docs" if explicit_doc_modes is not None else "batch_docs"


def _tree_document_loss_batch_scale(
    *,
    normalization_mode: str,
    batch_docs: int,
    supervised_docs: int,
) -> float:
    effective_mode = _normalize_tree_document_loss_normalization_mode(
        normalization_mode
    )
    if effective_mode == "batch_docs":
        return 1.0
    if int(supervised_docs) <= 0:
        return 1.0
    return float(max(1, int(batch_docs))) / float(max(1, int(supervised_docs)))


def _train_fno_tree_single_stage(
    *,
    model: FNOCountSketch,
    train_docs: Sequence[_FNOCountDoc],
    val_docs: Sequence[_FNOCountDoc],
    device: torch.device,
    n_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 1e-5,
    c1_weight: float = 0.0,
    c2_weight: float = 0.0,
    c3_weight: float = 0.0,
    root_weight: float = 1.0,
    leaf_query_rate: float = 1.0,
    leaf_label_rate: float | None = None,
    audit_fraction: float = 0.2,
    doc_sequence_train_fraction: float = 0.0,
    doc_sequence_objective: str = "count_ce_only",
    doc_sequence_class_index: Mapping[int, int] | None = None,
    root_class_index: Mapping[int, int] | None = None,
    document_supervision_mode_by_doc: Mapping[int, str] | None = None,
    tree_document_loss_normalization_mode: str = "auto",
    leaf_audit_indices_by_doc: Mapping[int, Sequence[int]] | None = None,
    c3_audit_indices_by_doc: Mapping[int, Sequence[int]] | None = None,
    internal_supervision_kind: str = "count_only",
    internal_label_rate: float = 0.0,
    max_internal_depth: int = 0,
    leaf_exact_supervision: bool = False,
    leaf_supervision_kind: str = "full_sketch",
    tree_local_weighting_mode: str = "fixed_k_hajek",
    tree_supervision_source: str = "rate",
    phi_compose_weight: float = 1.0,
    phi_contrastive_weight: float = 0.25,
    checkpoint_metric: str = "val_root_mae",
    eval_mode: str = "per_epoch",
    screen_doc_limit: int = 0,
    final_exact_doc_limit_override: int = 0,
    tree_batch_pack_mode: str = "structure_bucket",
    tree_batch_token_budget: int = 0,
    tree_batch_node_budget: int = 0,
    tree_batch_autotune: bool = True,
    tree_batch_structural_pad_limit: float = 0.5,
    tree_batch_auto_queue_min_docs: int = 8,
    tree_batch_auto_queue_min_fill_ratio: float = 0.5,
    tree_eval_workers_per_mig: int = 0,
    exact_metric_evaluator: Callable[..., Dict[str, float]] | None = None,
    exact_metric_selection_doc_limit: int = 0,
    exact_metric_selection_interval: int = 1,
    exact_metric_phi_pair_calibration_max_nodes: int | None = 512,
    exact_metric_final_doc_limit: int = 0,
    tree_exact_eval_max_docs: int = 0,
    runtime_config: GpuRuntimeConfig | None = None,
    leaf_sample_ordering_by_doc: Mapping[int, Sequence[int]] | None = None,
    internal_sample_ordering_by_doc: Mapping[int, Sequence[int]] | None = None,
    memory_probe: Callable[[str, Mapping[str, Any]], None] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    progress_stage_name: str = "single_stage",
    progress_epoch_offset: int = 0,
    progress_epochs_total: int = 0,
    progress_snapshot_interval: int = 10,
    progress_snapshot_dir: str | Path = "",
    grad_clip_norm: float = 1.0,
    depth_discount_gamma: float = 1.0,
    seed: int = 42,
) -> Dict[str, object]:
    """Train FNO tree-merge operator with local law supervision."""
    import random as _random

    normalized_checkpoint_metric = (
        str(checkpoint_metric or "val_root_mae").strip().lower() or "val_root_mae"
    )
    normalized_eval_mode = str(eval_mode or "per_epoch").strip().lower() or "per_epoch"
    normalized_pack_mode = (
        str(tree_batch_pack_mode or "structure_bucket").strip().lower()
        or "structure_bucket"
    )
    if normalized_checkpoint_metric not in VALID_TREE_CHECKPOINT_METRICS:
        raise ValueError(
            "checkpoint_metric must be one of "
            f"{VALID_TREE_CHECKPOINT_METRICS}; got {checkpoint_metric!r}"
        )
    if normalized_eval_mode not in {"per_epoch", "end_only"}:
        raise ValueError(
            "eval_mode must be one of {'per_epoch','end_only'}; "
            f"got {eval_mode!r}"
        )
    if normalized_pack_mode not in {"structure_bucket", "fixed_fused"}:
        raise ValueError(
            "tree_batch_pack_mode must be one of {'structure_bucket','fixed_fused'}; "
            f"got {tree_batch_pack_mode!r}"
        )
    effective_runtime_config = (
        runtime_config
        if runtime_config is not None
        else _ops_gpu_runtime_config(device=device)
    )
    effective_runtime_config = replace(
        effective_runtime_config,
        bucket_mode=_effective_tree_bucket_mode(
            pack_mode=str(normalized_pack_mode),
            bucket_mode=str(effective_runtime_config.bucket_mode),
        ),
    )
    runtime_telemetry = GpuRuntimeTelemetry(
        data_mode=str(effective_runtime_config.data_mode),
        bucket_mode=str(effective_runtime_config.bucket_mode),
        workers_per_mig=int(effective_runtime_config.workers_per_mig),
    )
    train_auto_queue_target_by_n_leaves = (
        _leaf_count_auto_queue_targets_for_docs(
            train_docs,
            structural_pad_limit=float(tree_batch_structural_pad_limit),
            min_docs=int(tree_batch_auto_queue_min_docs),
        )
        if _tree_leaf_count_auto_queue_enabled(str(effective_runtime_config.bucket_mode))
        else {}
    )
    val_auto_queue_target_by_n_leaves = (
        _leaf_count_auto_queue_targets_for_docs(
            val_docs,
            structural_pad_limit=float(tree_batch_structural_pad_limit),
            min_docs=int(tree_batch_auto_queue_min_docs),
        )
        if _tree_leaf_count_auto_queue_enabled(str(effective_runtime_config.bucket_mode))
        else {}
    )
    auto_queue_target_leaf_counts = sorted(
        {
            int(value)
            for value in list(train_auto_queue_target_by_n_leaves.values())
            + list(val_auto_queue_target_by_n_leaves.values())
            if int(value) > 0
        }
    )
    if auto_queue_target_leaf_counts:
        runtime_telemetry.add_extra_counter(
            "auto_queue_family_count",
            float(len(auto_queue_target_leaf_counts)),
        )

    def _emit_memory_probe(event: str, **payload: Any) -> None:
        if memory_probe is None:
            return
        memory_probe(str(event), {str(key): value for key, value in payload.items()})

    eval_screen_docs = _limit_eval_docs(
        val_docs,
        doc_limit=(
            None
            if int(screen_doc_limit) <= 0
            else int(screen_doc_limit)
        ),
    )
    train_store, train_store_telemetry = _build_tree_gpu_batch_store(
        docs=train_docs,
        model=model,
        device=device,
        split_name="train",
        runtime_config=effective_runtime_config,
        structural_pad_limit=float(tree_batch_structural_pad_limit),
        auto_queue_min_docs=int(tree_batch_auto_queue_min_docs),
    )
    if (
        train_store is None
        and str(device.type) == "cuda"
        and torch.cuda.is_available()
        and str(effective_runtime_config.data_mode) == "resident"
    ):
        raise RuntimeError(
            "GPU resident store failed to build for training data — "
            f"data_mode={effective_runtime_config.data_mode}, "
            f"bucket_mode={effective_runtime_config.bucket_mode}, "
            f"preload_splits={effective_runtime_config.preload_splits}"
        )
    _merge_gpu_runtime_telemetry(runtime_telemetry, train_store_telemetry)
    val_store, val_store_telemetry = _build_tree_gpu_batch_store(
        docs=val_docs,
        model=model,
        device=device,
        split_name="val",
        runtime_config=effective_runtime_config,
        structural_pad_limit=float(tree_batch_structural_pad_limit),
        auto_queue_min_docs=int(tree_batch_auto_queue_min_docs),
    )
    _merge_gpu_runtime_telemetry(runtime_telemetry, val_store_telemetry)

    def _resident_store_for_docs(
        eval_docs: Sequence[_FNOCountDoc],
    ) -> GpuBatchStore | None:
        if _docs_match_prefix(eval_docs, train_docs):
            return train_store
        if _docs_match_prefix(eval_docs, val_docs):
            return val_store
        return None

    def _auto_queue_targets_for_docs(
        eval_docs: Sequence[_FNOCountDoc],
    ) -> Mapping[int, int]:
        if _docs_match_prefix(eval_docs, train_docs):
            return train_auto_queue_target_by_n_leaves
        if _docs_match_prefix(eval_docs, val_docs):
            return val_auto_queue_target_by_n_leaves
        if _tree_leaf_count_auto_queue_enabled(str(effective_runtime_config.bucket_mode)):
            return _leaf_count_auto_queue_targets_for_docs(
                eval_docs,
                structural_pad_limit=float(tree_batch_structural_pad_limit),
                min_docs=int(tree_batch_auto_queue_min_docs),
            )
        return {}

    def _run_exact_metric_eval(
        eval_docs: Sequence[_FNOCountDoc],
        *,
        doc_limit: int | None,
    ) -> Dict[str, float]:
        import inspect

        exact_metric_fn = exact_metric_evaluator or _eval_fno_exact_sketch_direct_metrics
        call_kwargs: Dict[str, Any] = {"device": device}
        signature = inspect.signature(exact_metric_fn)
        if "doc_limit" in signature.parameters:
            call_kwargs["doc_limit"] = doc_limit
        if "phi_pair_calibration_max_nodes" in signature.parameters:
            call_kwargs["phi_pair_calibration_max_nodes"] = (
                exact_metric_phi_pair_calibration_max_nodes
            )
        if "max_docs" in signature.parameters:
            call_kwargs["max_docs"] = int(effective_eval_max_docs)
        if "token_budget" in signature.parameters:
            call_kwargs["token_budget"] = int(effective_eval_token_budget)
        if "node_budget" in signature.parameters:
            call_kwargs["node_budget"] = int(effective_eval_node_budget)
        if "bucket_docs_cap_by_n_leaves" in signature.parameters:
            call_kwargs["bucket_docs_cap_by_n_leaves"] = eval_bucket_docs_cap_by_n_leaves
        if "runtime_bucket_mode" in signature.parameters:
            call_kwargs["runtime_bucket_mode"] = str(effective_runtime_config.bucket_mode)
        if "structural_pad_limit" in signature.parameters:
            call_kwargs["structural_pad_limit"] = float(tree_batch_structural_pad_limit)
        if "auto_queue_min_docs" in signature.parameters:
            call_kwargs["auto_queue_min_docs"] = int(tree_batch_auto_queue_min_docs)
        if "auto_queue_min_fill_ratio" in signature.parameters:
            call_kwargs["auto_queue_min_fill_ratio"] = float(tree_batch_auto_queue_min_fill_ratio)
        if "auto_queue_target_by_n_leaves" in signature.parameters:
            call_kwargs["auto_queue_target_by_n_leaves"] = _auto_queue_targets_for_docs(eval_docs)
        if "batching_metrics" in signature.parameters:
            call_kwargs["batching_metrics"] = batching_metrics
        if "memory_probe" in signature.parameters:
            call_kwargs["memory_probe"] = memory_probe
        if "resident_store" in signature.parameters:
            call_kwargs["resident_store"] = _resident_store_for_docs(eval_docs)
        if "runtime_telemetry" in signature.parameters:
            call_kwargs["runtime_telemetry"] = runtime_telemetry
        if "pack_mode" in signature.parameters:
            call_kwargs["pack_mode"] = str(normalized_pack_mode)
        return exact_metric_fn(model, eval_docs, **call_kwargs)

    rng = _random.Random(int(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)
    batching_metrics = _BatchingMetricsAccumulator()
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
    autotuned_budgets = _autotune_tree_batch_budgets(
        model,
        train_docs if train_docs else val_docs,
        device=device,
        legacy_batch_size=int(batch_size),
        pack_mode=str(normalized_pack_mode),
        bucket_mode=str(effective_runtime_config.bucket_mode),
        structural_pad_limit=float(tree_batch_structural_pad_limit),
        auto_queue_min_docs=int(tree_batch_auto_queue_min_docs),
    ) if bool(tree_batch_autotune) else _AutotunedTreeBatchBudgets(
        train_leaf_token_budget=0,
        train_node_budget=0,
        eval_leaf_token_budget=0,
        eval_node_budget=0,
        eval_workers_per_mig=max(1, int(tree_eval_workers_per_mig) if int(tree_eval_workers_per_mig) > 0 else 1),
    )
    autotune_probe_diagnostics = autotuned_budgets.probe_diagnostics
    _apply_autotune_probe_diagnostics(
        batching_metrics,
        autotune_probe_diagnostics,
    )
    effective_train_token_budget = int(tree_batch_token_budget)
    if effective_train_token_budget <= 0:
        effective_train_token_budget = int(autotuned_budgets.train_leaf_token_budget)
    effective_train_node_budget = int(tree_batch_node_budget)
    if effective_train_node_budget <= 0:
        effective_train_node_budget = int(autotuned_budgets.train_node_budget)
    effective_eval_token_budget = int(tree_batch_token_budget)
    if effective_eval_token_budget <= 0:
        effective_eval_token_budget = int(autotuned_budgets.eval_leaf_token_budget)
    effective_eval_node_budget = int(tree_batch_node_budget)
    if effective_eval_node_budget <= 0:
        effective_eval_node_budget = int(autotuned_budgets.eval_node_budget)
    raw_train_bucket_docs_cap_by_n_leaves = {
        int(n_leaves): int(cap)
        for n_leaves, cap in autotuned_budgets.train_bucket_max_docs_by_n_leaves
    }
    configured_train_max_docs = max(
        1,
        int(batch_size) if int(batch_size) > 0 else 1,
    )
    train_bucket_docs_cap_by_n_leaves = {
        int(n_leaves): min(int(configured_train_max_docs), int(cap))
        for n_leaves, cap in raw_train_bucket_docs_cap_by_n_leaves.items()
    }
    eval_bucket_docs_cap_by_n_leaves = {
        int(n_leaves): int(cap)
        for n_leaves, cap in autotuned_budgets.eval_bucket_max_docs_by_n_leaves
    }
    effective_train_max_docs = int(configured_train_max_docs)
    configured_exact_eval_max_docs = int(tree_exact_eval_max_docs)
    if configured_exact_eval_max_docs <= 0:
        configured_exact_eval_max_docs = max(
            1,
            int(batch_size) if int(batch_size) > 0 else 1,
        )
    effective_eval_max_docs = max(1, int(configured_exact_eval_max_docs))
    eval_bucket_docs_cap_by_n_leaves = {
        int(n_leaves): max(1, min(int(cap), int(effective_eval_max_docs)))
        for n_leaves, cap in eval_bucket_docs_cap_by_n_leaves.items()
    }
    effective_eval_workers_per_mig = (
        max(1, int(tree_eval_workers_per_mig))
        if int(tree_eval_workers_per_mig) > 0
        else int(autotuned_budgets.eval_workers_per_mig)
    )
    target_scale = float(model.target_scale)
    doc_sequence_fraction = min(1.0, max(0.0, float(doc_sequence_train_fraction)))
    doc_sequence_class_index = dict(doc_sequence_class_index or {})
    root_class_index = dict(root_class_index or {})
    normalized_supervision_source = _normalize_tree_supervision_source(
        tree_supervision_source
    )
    explicit_doc_modes = (
        {
            int(doc_idx): str(mode).strip().lower()
            for doc_idx, mode in dict(document_supervision_mode_by_doc or {}).items()
            if str(mode).strip()
        }
        if document_supervision_mode_by_doc is not None
        else ({} if normalized_supervision_source == "manifest" else None)
    )
    effective_document_loss_normalization_mode = (
        _effective_tree_document_loss_normalization_mode(
            tree_document_loss_normalization_mode,
            explicit_doc_modes=explicit_doc_modes,
        )
    )
    explicit_leaf_audits = (
        {
            int(doc_idx): tuple(
                sorted({int(index) for index in list(indices or [])})
            )
            for doc_idx, indices in dict(leaf_audit_indices_by_doc or {}).items()
        }
        if leaf_audit_indices_by_doc is not None
        else ({} if normalized_supervision_source == "manifest" else None)
    )
    explicit_c3_audits = (
        {
            int(doc_idx): tuple(
                sorted({int(index) for index in list(indices or [])})
            )
            for doc_idx, indices in dict(c3_audit_indices_by_doc or {}).items()
        }
        if c3_audit_indices_by_doc is not None
        else ({} if normalized_supervision_source == "manifest" else None)
    )
    explicit_supervision = any(
        item is not None
        for item in (
            explicit_doc_modes,
            explicit_leaf_audits,
            explicit_c3_audits,
        )
    )
    effective_leaf_label_rate = (
        float(leaf_query_rate)
        if leaf_label_rate is None
        else float(leaf_label_rate)
    )
    max_internal_depth_value = int(max_internal_depth)

    def _internal_index_limit(n_leaves: int) -> int:
        total = int(max(0, int(n_leaves) - 1))
        if total <= 0:
            return 0
        if max_internal_depth_value <= 0:
            return total
        n = int(max(0, n_leaves))
        merges = 0
        depth = 0
        while n > 1 and depth < max_internal_depth_value:
            depth += 1
            merges += int(n // 2)
            n = int((n + 1) // 2)
        return int(min(total, merges))
    doc_sequence_train_indices = _sample_doc_index_subset(
        n_docs=len(train_docs),
        fraction=float(doc_sequence_fraction),
        seed=int(seed) + 70_001,
    )
    if explicit_doc_modes is not None:
        doc_sequence_train_indices = {
            int(doc_idx)
            for doc_idx, mode in explicit_doc_modes.items()
            if str(mode) == "doc_sequence"
        }
    if explicit_doc_modes is None:
        doc_sequence_supervision_docs_total = int(len(doc_sequence_train_indices))
        root_supervision_docs_total = max(
            0,
            int(len(train_docs)) - int(doc_sequence_supervision_docs_total),
        )
    else:
        doc_sequence_supervision_docs_total = sum(
            1 for mode in explicit_doc_modes.values() if str(mode) == "doc_sequence"
        )
        root_supervision_docs_total = sum(
            1
            for mode in explicit_doc_modes.values()
            if str(mode) in {"root_only", "full_doc_only"}
        )
    document_supervision_docs_total = int(
        root_supervision_docs_total + doc_sequence_supervision_docs_total
    )
    document_supervision_coverage_rate = (
        float(document_supervision_docs_total) / float(len(train_docs))
        if train_docs
        else 0.0
    )
    if doc_sequence_fraction > 0.0 and not doc_sequence_class_index:
        raise ValueError("doc_sequence_class_index is required when doc_sequence_train_fraction > 0")
    if (
        explicit_doc_modes is not None
        and any(str(mode) == "doc_sequence" for mode in explicit_doc_modes.values())
        and not doc_sequence_class_index
    ):
        raise ValueError(
            "doc_sequence_class_index is required when explicit doc supervision uses doc_sequence"
        )
    if str(model.root_supervision_kind) == "count_ce" and not root_class_index:
        raise ValueError(
            "root_class_index is required when tree root supervision uses count_ce"
        )
    normalized_internal_supervision = (
        str(internal_supervision_kind or "none").strip().lower() or "none"
    )
    if normalized_internal_supervision not in VALID_INTERNAL_SUPERVISION_KINDS:
        raise ValueError(
            "internal_supervision_kind must be one of "
            f"{VALID_INTERNAL_SUPERVISION_KINDS}"
        )
    normalized_leaf_supervision = (
        str(leaf_supervision_kind or "full_sketch").strip().lower() or "full_sketch"
    )
    if normalized_leaf_supervision not in VALID_LEAF_SUPERVISION_KINDS:
        raise ValueError(
            "leaf_supervision_kind must be one of "
            f"{VALID_LEAF_SUPERVISION_KINDS}"
        )
    normalized_local_weighting_mode = _normalize_tree_local_weighting_mode(
        tree_local_weighting_mode
    )

    idxs = list(range(len(train_docs)))
    fixed_leaf_indices_by_doc: Dict[int, Tuple[int, ...] | None] = {}
    fixed_internal_indices_by_doc: Dict[int, Tuple[int, ...] | None] = {}
    if model.use_summary_spec and explicit_leaf_audits is None:
        for doc_idx, doc in enumerate(train_docs):
            ordering = None
            if leaf_sample_ordering_by_doc is not None:
                ordering = leaf_sample_ordering_by_doc.get(int(doc_idx))
            if ordering is None:
                fixed_leaf_indices_by_doc[int(doc_idx)] = _deterministic_sample_indices(
                    n_items=len(doc.leaf_token_ids),
                    rate=float(effective_leaf_label_rate),
                    seed=int(seed) + 81_000 + int(doc_idx),
                )
            else:
                fixed_leaf_indices_by_doc[int(doc_idx)] = _deterministic_sample_indices_from_ordering(
                    ordering=ordering,
                    rate=float(effective_leaf_label_rate),
                    n_items=len(doc.leaf_token_ids),
                )
    if model.use_summary_spec and explicit_c3_audits is None:
        internal_rate = (
            float(internal_label_rate)
            if str(internal_supervision_kind) != "none"
            else 0.0
        )
        for doc_idx, doc in enumerate(train_docs):
            internal_n_items = _internal_index_limit(len(doc.leaf_token_ids))
            ordering = None
            if internal_sample_ordering_by_doc is not None:
                ordering = internal_sample_ordering_by_doc.get(int(doc_idx))
            if ordering is None:
                fixed_internal_indices_by_doc[int(doc_idx)] = _deterministic_sample_indices(
                    n_items=internal_n_items,
                    rate=float(internal_rate),
                    seed=int(seed) + 91_000 + int(doc_idx),
                )
            else:
                fixed_internal_indices_by_doc[int(doc_idx)] = _deterministic_sample_indices_from_ordering(
                    ordering=ordering,
                    rate=float(internal_rate),
                    n_items=internal_n_items,
                )

    def _mean_sampling_summary(
        *,
        population_sizes: Sequence[int],
        sampled_by_doc: Mapping[int, Tuple[int, ...] | None] | None,
    ) -> tuple[float, float, float]:
        populations: List[int] = []
        sample_sizes: List[int] = []
        propensities: List[float] = []
        mapping = dict(sampled_by_doc or {})
        for doc_idx, population_size in enumerate(population_sizes):
            population, sample_size, propensity = _effective_local_sampling_summary(
                int(population_size),
                mapping.get(int(doc_idx)),
            )
            populations.append(int(population))
            sample_sizes.append(int(sample_size))
            propensities.append(float(propensity))
        if not populations:
            return 0.0, 0.0, 0.0
        return (
            float(sum(populations)) / float(len(populations)),
            float(sum(sample_sizes)) / float(len(sample_sizes)),
            float(sum(propensities)) / float(len(propensities)),
        )

    leaf_population_size_mean, leaf_sample_size_mean, leaf_effective_propensity_mean = (
        _mean_sampling_summary(
            population_sizes=[len(doc.leaf_token_ids) for doc in train_docs],
            sampled_by_doc=(
                explicit_leaf_audits
                if explicit_leaf_audits is not None
                else fixed_leaf_indices_by_doc
            ),
        )
    )
    merge_population_size_mean, merge_sample_size_mean, merge_effective_propensity_mean = (
        _mean_sampling_summary(
            population_sizes=[
                _internal_index_limit(len(doc.leaf_token_ids)) for doc in train_docs
            ],
            sampled_by_doc=(
                explicit_c3_audits
                if explicit_c3_audits is not None
                else fixed_internal_indices_by_doc
            ),
        )
    )
    c2_pair_weighting_mode = _c2_pair_weighting_mode(
        tree_supervision_source=normalized_supervision_source,
        local_estimand_mode=normalized_local_weighting_mode,
    )
    c2_same_pair_count_sum = 0.0
    c2_different_pair_count_sum = 0.0
    c2_pair_weight_ess_sum = 0.0
    c2_pair_weight_max_value = 0.0
    c2_pair_diag_count = 0
    first_batch_local_objective_audit: Dict[str, Any] = {}
    all_grad_diagnostics: List[Dict[str, object]] = []
    loss_curve: List[float] = []
    selection_curve: List[float] = []
    component_curve_names = (
        "root_count_loss",
        "leaf_count_loss",
        "leaf_first_loss",
        "leaf_last_loss",
        "merge_count_loss",
        "merge_first_loss",
        "merge_last_loss",
        "c2_count_loss",
        "c2_first_loss",
        "c2_last_loss",
        "c2_join_loss",
        "c2_on_range_reencode_loss",
        "phi_compose_loss",
        "phi_contrastive_loss",
    )
    component_loss_curves: Dict[str, List[float]] = {
        name: [] for name in component_curve_names
    }
    document_loss_batch_scale_sum = 0.0
    document_loss_batch_scale_count = 0
    last_document_loss_batch_scale = 1.0
    elapsed_train_loop_s = 0.0
    elapsed_screen_eval_s = 0.0
    elapsed_exact_metric_eval_s = 0.0
    elapsed_split_eval_s = 0.0
    elapsed_state_clone_s = 0.0
    _emit_memory_probe("pre_clone_module_state", location="initial_best_state")
    _clone_start_s = time.perf_counter()
    best_state = clone_module_state(model)
    elapsed_state_clone_s += time.perf_counter() - _clone_start_s
    _emit_memory_probe("post_clone_module_state", location="initial_best_state")
    root_selection_metric_name = (
        "val_root_mae"
        if explicit_supervision
        else (
            "val_doc_sequence_root_mae"
            if doc_sequence_fraction >= 1.0
            else (
                "val_tree_doc_sequence_curriculum_mae"
                if doc_sequence_fraction > 0.0
                else "val_root_mae"
            )
        )
    )
    root_selection_mode = (
        "best_val_root_mae"
        if explicit_supervision
        else (
            "best_val_doc_sequence_root_mae"
            if doc_sequence_fraction >= 1.0
            else (
                "best_val_tree_doc_sequence_curriculum_mae"
                if doc_sequence_fraction > 0.0
                else "best_val_root_mae"
            )
        )
    )
    best_selection = TrainingSelectionMetadata(
        mode=(
            "best_val_leaf_codec_direct"
            if normalized_checkpoint_metric == "val_leaf_codec_direct"
            else (
                "best_val_theorem_bootstrap_direct"
                if normalized_checkpoint_metric == "val_theorem_bootstrap_direct"
                else (
                    "best_val_exact_sketch_direct"
                    if normalized_checkpoint_metric == "val_exact_sketch_direct"
                    else (
                        "best_val_task_root_exact_sketch_direct"
                        if normalized_checkpoint_metric == "val_task_root_exact_sketch_direct"
                        else root_selection_mode
                    )
                )
            )
        )
        if val_docs
        else "final_epoch_no_validation",
        split="val" if val_docs else "config",
        metric_name=(
            "val_leaf_codec_direct"
            if normalized_checkpoint_metric == "val_leaf_codec_direct"
            else (
                "val_theorem_bootstrap_direct"
                if normalized_checkpoint_metric == "val_theorem_bootstrap_direct"
                else (
                    "val_task_root_exact_sketch_direct"
                    if normalized_checkpoint_metric == "val_task_root_exact_sketch_direct"
                    else root_selection_metric_name
                )
            )
        )
        if normalized_checkpoint_metric not in {
            "val_exact_sketch_direct",
            "val_task_root_exact_sketch_direct",
        }
        else str(normalized_checkpoint_metric)
        if val_docs
        else (
            "train_leaf_codec_direct"
            if normalized_checkpoint_metric == "val_leaf_codec_direct"
            else (
                "train_theorem_bootstrap_direct"
                if normalized_checkpoint_metric == "val_theorem_bootstrap_direct"
                else (
                    "train_exact_sketch_direct"
                    if normalized_checkpoint_metric == "val_exact_sketch_direct"
                    else (
                        "train_task_root_exact_sketch_direct"
                        if normalized_checkpoint_metric == "val_task_root_exact_sketch_direct"
                        else "train_root_mae"
                    )
                )
            )
        ),
        metric_value=float("inf"),
        best_epoch=0,
    )
    best_tree_val_mae = float("inf")
    best_doc_sequence_val_mae = float("inf")
    final_exact_doc_limit = (
        int(final_exact_doc_limit_override)
        if int(final_exact_doc_limit_override) > 0
        else int(exact_metric_final_doc_limit)
    )

    normalized_progress_stage = (
        str(progress_stage_name or "single_stage").strip().lower() or "single_stage"
    )
    overall_epochs_total = int(progress_epochs_total)
    if overall_epochs_total <= 0:
        overall_epochs_total = int(progress_epoch_offset) + int(n_epochs)
    effective_progress_snapshot_interval = max(0, int(progress_snapshot_interval))
    normalized_progress_snapshot_dir = str(progress_snapshot_dir or "").strip()
    progress_snapshot_root = (
        Path(normalized_progress_snapshot_dir).expanduser()
        if normalized_progress_snapshot_dir
        else None
    )
    if progress_snapshot_root is not None:
        progress_snapshot_root.mkdir(parents=True, exist_ok=True)
    latest_progress_snapshot_path = ""
    progress_snapshot_paths: List[str] = []
    latest_tree_val_mae = float("nan")
    latest_tree_val_exact_match = float("nan")
    latest_doc_sequence_val_mae = float("nan")
    latest_doc_sequence_val_exact_match = float("nan")

    def _safe_progress_float(value: Any) -> float | None:
        try:
            numeric = float(value)
        except Exception:
            return None
        if not bool(np.isfinite(numeric)):
            return None
        return numeric

    def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def _maybe_write_progress_snapshot(
        stage_suffix: str,
        *,
        epoch_completed: int,
    ) -> str:
        nonlocal latest_progress_snapshot_path
        if progress_snapshot_root is None or effective_progress_snapshot_interval <= 0:
            return latest_progress_snapshot_path
        overall_epoch_completed = int(progress_epoch_offset) + int(epoch_completed)
        if overall_epoch_completed <= 0:
            return latest_progress_snapshot_path
        should_persist = (
            overall_epoch_completed % int(effective_progress_snapshot_interval) == 0
            or int(epoch_completed) >= int(n_epochs)
        )
        if not should_persist:
            return latest_progress_snapshot_path
        snapshot_payload: Dict[str, Any] = {
            "schema_version": 1,
            "saved_at": datetime.now(UTC).isoformat(),
            "stage": str(stage_suffix),
            "epoch_completed": int(overall_epoch_completed),
            "epochs_total": int(overall_epochs_total),
            "stage_epoch_completed": int(epoch_completed),
            "stage_epochs_total": int(n_epochs),
            "best_epoch": int(progress_epoch_offset) + int(best_selection.best_epoch),
            "selection_metric_name": str(best_selection.metric_name),
            "progress_snapshot_interval": int(effective_progress_snapshot_interval),
        }
        best_metric_value = _safe_progress_float(best_selection.metric_value)
        if best_metric_value is not None:
            snapshot_payload["selection_metric_value"] = best_metric_value
        if loss_curve:
            latest_loss = _safe_progress_float(loss_curve[-1])
            if latest_loss is not None:
                snapshot_payload["train_loss"] = latest_loss
        for key, value in (
            ("val_root_mae", latest_tree_val_mae),
            ("val_exact_match", latest_tree_val_exact_match),
            ("doc_sequence_val_root_mae", latest_doc_sequence_val_mae),
            ("doc_sequence_val_exact_match", latest_doc_sequence_val_exact_match),
        ):
            numeric = _safe_progress_float(value)
            if numeric is not None:
                snapshot_payload[key] = numeric
        component_snapshot: Dict[str, float] = {}
        for name, values in component_loss_curves.items():
            if not values:
                continue
            numeric = _safe_progress_float(values[-1])
            if numeric is not None:
                component_snapshot[str(name)] = numeric
        if component_snapshot:
            snapshot_payload["training_component_loss_finals"] = component_snapshot
        snapshot_path = progress_snapshot_root / (
            f"{str(stage_suffix)}__epoch_{int(overall_epoch_completed):04d}.json"
        )
        _write_json_atomic(snapshot_path, snapshot_payload)
        _write_json_atomic(progress_snapshot_root / "latest.json", snapshot_payload)
        latest_progress_snapshot_path = str(snapshot_path)
        if str(snapshot_path) not in progress_snapshot_paths:
            progress_snapshot_paths.append(str(snapshot_path))
        return latest_progress_snapshot_path

    def _emit_progress_snapshot(stage_suffix: str, *, epoch_completed: int) -> None:
        latest_snapshot_path = _maybe_write_progress_snapshot(
            str(stage_suffix),
            epoch_completed=int(epoch_completed),
        )
        if progress_callback is None:
            return
        payload: Dict[str, Any] = {
            "state": "running",
            "stage": str(stage_suffix),
            "epoch_completed": int(progress_epoch_offset) + int(epoch_completed),
            "epochs_total": int(overall_epochs_total),
            "stage_epoch_completed": int(epoch_completed),
            "stage_epochs_total": int(n_epochs),
            "best_epoch": int(progress_epoch_offset) + int(best_selection.best_epoch),
            "selection_metric_name": str(best_selection.metric_name),
        }
        best_metric_value = float(best_selection.metric_value)
        if bool(np.isfinite(best_metric_value)):
            payload["selection_metric_value"] = best_metric_value
        if loss_curve:
            latest_loss = float(loss_curve[-1])
            if bool(np.isfinite(latest_loss)):
                payload["train_loss"] = latest_loss
        for key, value in (
            ("val_root_mae", latest_tree_val_mae),
            ("val_exact_match", latest_tree_val_exact_match),
            ("doc_sequence_val_root_mae", latest_doc_sequence_val_mae),
            ("doc_sequence_val_exact_match", latest_doc_sequence_val_exact_match),
        ):
            numeric = _safe_progress_float(value)
            if numeric is not None:
                payload[key] = numeric
        if effective_progress_snapshot_interval > 0:
            payload["progress_snapshot_interval"] = int(
                effective_progress_snapshot_interval
            )
        if progress_snapshot_root is not None:
            payload["progress_snapshot_dir"] = str(progress_snapshot_root)
        if latest_snapshot_path:
            payload["latest_progress_snapshot_path"] = str(latest_snapshot_path)
        progress_callback(payload)

    _emit_memory_probe(
        "start_train_loop",
        n_epochs=int(n_epochs),
        n_train_docs=int(len(train_docs)),
        n_val_docs=int(len(val_docs)),
        n_eval_screen_docs=int(len(eval_screen_docs)),
        checkpoint_metric=str(normalized_checkpoint_metric),
    )
    _maybe_enable_cuda_fast_math(device)
    _emit_progress_snapshot(f"{normalized_progress_stage}_train", epoch_completed=0)

    for epoch in range(int(n_epochs)):
        rng.shuffle(idxs)
        model.train()
        batch_loss_total = torch.zeros((), device=device, dtype=torch.float32)
        batch_loss_count = 0
        epoch_component_sums = {
            name: torch.zeros((), device=device, dtype=torch.float32)
            for name in component_curve_names
        }
        epoch_component_counts = {name: 0 for name in component_curve_names}
        epoch_grad_diagnostics: list[dict[str, object]] = []

        for b0 in range(0, len(idxs), int(max(1, effective_train_max_docs))):
            _batch_start_s = time.perf_counter()
            batch_idx = idxs[b0 : b0 + int(max(1, effective_train_max_docs))]
            opt.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=device, dtype=torch.float32)
            batch_document_loss_sum = torch.zeros(
                (), device=device, dtype=torch.float32
            )
            batch_non_document_loss_sum = torch.zeros(
                (), device=device, dtype=torch.float32
            )
            batch_document_supervision_count = 0
            # Cross-document phi accumulation for batch-level contrastive loss
            _batch_phi_feature_chunks: List[torch.Tensor] = []
            _batch_phi_labels: List[Any] = []
            _batch_oracle_vecs: List[Any] = []
            _batch_phi_fast_key_chunks: Dict[str, List[torch.Tensor]] = {
                "count_values": [],
                "count_keys": [],
                "first_targets": [],
                "last_targets": [],
            }
            _batch_all_deferred_chunks_are_fast_markov = True
            _use_deferred_contrastive = bool(
                phi_contrastive_weight > 0
                and bool(getattr(model, "use_shared_theorem_surface", False))
            )

            with _autocast_context(device):
                tree_forward_work: List[Dict[str, Any]] = []
                for i in batch_idx:
                    doc = train_docs[i]
                    if explicit_doc_modes is not None:
                        document_mode = str(explicit_doc_modes.get(int(i), "")).strip().lower()
                    else:
                        document_mode = "doc_sequence" if i in doc_sequence_train_indices else "root_only"
                    root_only_supervision = document_mode in {"root_only", "full_doc_only"}
                    doc_sequence_supervision = document_mode == "doc_sequence"
                    if bool(root_only_supervision or doc_sequence_supervision):
                        batch_document_supervision_count += 1
                    n_leaf = len(doc.leaf_token_ids)
                    n_internal_total = max(0, int(n_leaf) - 1)
                    n_internal = _internal_index_limit(int(n_leaf))

                    if explicit_leaf_audits is not None:
                        explicit_leaf_indices = tuple(
                            int(index)
                            for index in explicit_leaf_audits.get(int(i), tuple())
                            if 0 <= int(index) < int(n_leaf)
                        )
                        leaf_audit_indices = set(explicit_leaf_indices)
                        collect_leaf = bool(
                            (c1_weight > 0 or model.use_summary_spec) and leaf_audit_indices
                        )
                    elif model.use_summary_spec:
                        fixed_leaf_indices = fixed_leaf_indices_by_doc.get(int(i), tuple())
                        if fixed_leaf_indices is None:
                            leaf_audit_indices = None
                            collect_leaf = True
                        else:
                            leaf_audit_indices = set(fixed_leaf_indices)
                            collect_leaf = bool(leaf_audit_indices)
                    else:
                        leaf_sample_rate = float(effective_leaf_label_rate)
                        if bool(leaf_exact_supervision) or leaf_sample_rate >= 1.0:
                            n_leaf_audit = int(n_leaf)
                        elif leaf_sample_rate <= 0.0:
                            n_leaf_audit = 0
                        else:
                            n_leaf_audit = max(
                                1, int(round(float(leaf_sample_rate) * float(n_leaf)))
                            )
                            n_leaf_audit = min(int(n_leaf), int(n_leaf_audit))
                        if n_leaf_audit >= n_leaf:
                            leaf_audit_indices = None
                            collect_leaf = bool(c1_weight > 0 or leaf_exact_supervision)
                        elif n_leaf_audit > 0 and c1_weight > 0:
                            leaf_audit_indices = set(rng.sample(range(n_leaf), k=n_leaf_audit))
                            collect_leaf = True
                        else:
                            leaf_audit_indices = set()
                            collect_leaf = False

                    if explicit_c3_audits is not None:
                        explicit_internal_indices = tuple(
                            int(index)
                            for index in explicit_c3_audits.get(int(i), tuple())
                            if 0 <= int(index) < int(n_internal)
                        )
                        c3_audit_indices = set(explicit_internal_indices)
                        collect_c3 = bool(
                            ((c3_weight > 0) or str(internal_supervision_kind) != "none")
                            and c3_audit_indices
                        )
                    elif model.use_summary_spec:
                        fixed_internal_indices = fixed_internal_indices_by_doc.get(
                            int(i), tuple()
                        )
                        if fixed_internal_indices is None:
                            if int(n_internal) <= 0 or str(internal_supervision_kind) == "none":
                                c3_audit_indices = set()
                                collect_c3 = False
                            elif int(n_internal) < int(n_internal_total):
                                c3_audit_indices = set(range(int(n_internal)))
                                collect_c3 = True
                            else:
                                c3_audit_indices = None
                                collect_c3 = True
                        else:
                            c3_audit_indices = set(
                                int(index)
                                for index in fixed_internal_indices
                                if 0 <= int(index) < int(n_internal)
                            )
                            collect_c3 = bool(
                                str(internal_supervision_kind) != "none"
                                and c3_audit_indices
                            )
                    else:
                        if int(n_internal) <= 0 or str(internal_supervision_kind) == "none":
                            c3_audit_indices = set()
                            collect_c3 = False
                        else:
                            internal_sample_rate = float(internal_label_rate)
                            if internal_sample_rate >= 1.0:
                                if int(n_internal) < int(n_internal_total):
                                    c3_audit_indices = set(range(int(n_internal)))
                                else:
                                    c3_audit_indices = None
                                collect_c3 = True
                            elif internal_sample_rate <= 0.0:
                                c3_audit_indices = set()
                                collect_c3 = False
                            else:
                                n_audit = max(
                                    1,
                                    int(
                                        round(
                                            float(internal_sample_rate) * float(n_internal)
                                        )
                                    ),
                                )
                                n_audit = min(int(n_internal), int(n_audit))
                                if n_audit >= int(n_internal):
                                    if int(n_internal) < int(n_internal_total):
                                        c3_audit_indices = set(range(int(n_internal)))
                                    else:
                                        c3_audit_indices = None
                                    collect_c3 = True
                                else:
                                    c3_audit_indices = set(
                                        rng.sample(range(int(n_internal)), k=int(n_audit))
                                    )
                                    collect_c3 = True

                    collect_c2 = bool(
                        c2_weight > 0 or model.use_decoded_markov_sketch or model.use_summary_spec
                    )
                    if not any((root_only_supervision, doc_sequence_supervision, collect_leaf, collect_c2, collect_c3)):
                        continue

                    doc_sequence_loss = torch.zeros((), device=device, dtype=torch.float32)
                    if doc_sequence_supervision:
                        doc_sequence_loss = _doc_sequence_loss_for_doc(
                            model,
                            doc,
                            device=device,
                            class_index=doc_sequence_class_index,
                            objective_name=str(doc_sequence_objective),
                        )

                    if not any((root_only_supervision, collect_leaf, collect_c2, collect_c3)):
                        doc_loss = (
                            float(root_weight) * doc_sequence_loss
                            if explicit_supervision
                            else doc_sequence_loss
                        )
                        batch_document_loss_sum = batch_document_loss_sum + doc_loss
                        continue

                    tree_forward_work.append(
                        {
                            "doc_index": int(i),
                            "doc": doc,
                            "root_only_supervision": bool(root_only_supervision),
                            "doc_sequence_supervision": bool(doc_sequence_supervision),
                            "doc_sequence_loss": doc_sequence_loss,
                            "collect_leaf": bool(collect_leaf),
                            "collect_c2": bool(collect_c2),
                            "collect_c3": bool(collect_c3),
                            "leaf_audit_indices": leaf_audit_indices,
                            "c3_audit_indices": c3_audit_indices,
                            "leaf_supervision_population_size": int(n_leaf),
                            "internal_supervision_population_size": int(n_internal),
                        }
                    )

                tree_items_with_payload: List[Tuple[_TreeWorkItem, Dict[str, Any]]] = []
                for work in tree_forward_work:
                    document_mode = "root_only" if bool(work["root_only_supervision"]) else "full_tree"
                    tree_items_with_payload.append(
                        (
                            _tree_work_item_from_doc(
                                work["doc"],
                                doc_index=int(work.get("doc_index", -1)),
                                work_kind=document_mode,
                                collect_leaf=bool(work["collect_leaf"]),
                                collect_c2=bool(work["collect_c2"]),
                                collect_c3=bool(work["collect_c3"]),
                                root_only_supervision=bool(work["root_only_supervision"]),
                                doc_sequence_supervision=bool(work["doc_sequence_supervision"]),
                                leaf_audit_indices=work["leaf_audit_indices"],
                                c3_audit_indices=work["c3_audit_indices"],
                                document_mode=str(document_mode),
                            ),
                            work,
                        )
                    )

                packed_tree_batches = _pack_tree_work_items(
                    [item for item, _work in tree_items_with_payload],
                    max_docs=int(effective_train_max_docs),
                    max_total_leaf_tokens=int(effective_train_token_budget),
                    max_total_nodes=int(effective_train_node_budget),
                    max_total_merge_ops=0,
                    bucket_docs_cap_by_n_leaves=train_bucket_docs_cap_by_n_leaves,
                    bucket_mode=str(effective_runtime_config.bucket_mode),
                    structural_pad_limit=float(tree_batch_structural_pad_limit),
                    auto_queue_min_docs=int(tree_batch_auto_queue_min_docs),
                    auto_queue_target_by_n_leaves=train_auto_queue_target_by_n_leaves,
                    tail_repack_fill_ratio=(
                        float(tree_batch_auto_queue_min_fill_ratio)
                        if normalized_pack_mode == "fixed_fused"
                        else 0.0
                    ),
                    tail_repack_min_docs=int(tree_batch_auto_queue_min_docs),
                )
                work_lookup = {
                    int(item.doc_index): work for item, work in tree_items_with_payload
                }

                for packed_tree_batch in packed_tree_batches:
                    forward_start_s = time.perf_counter()
                    if (
                        normalized_pack_mode == "fixed_fused"
                        and _supports_fixed_fused_batch(model, packed_tree_batch.items)
                    ):
                        if bool(packed_tree_batch.bucket_key.auto_queue_enabled):
                            runtime_telemetry.add_extra_counter("auto_queue_fused_batches", 1.0)
                        fused = _fixed_fused_training_batch_forward(
                            model,
                            packed_tree_batch,
                            work_lookup=work_lookup,
                            device=device,
                            resident_store=train_store,
                            runtime_telemetry=runtime_telemetry,
                            root_weight=float(root_weight),
                            c1_weight=float(c1_weight),
                            c2_weight=float(c2_weight),
                            c3_weight=float(c3_weight),
                            phi_compose_weight=float(phi_compose_weight),
                            leaf_supervision_kind=str(leaf_supervision_kind),
                            internal_supervision_kind=str(internal_supervision_kind),
                            tree_local_weighting_mode=normalized_local_weighting_mode,
                            tree_supervision_source=normalized_supervision_source,
                            defer_contrastive=bool(_use_deferred_contrastive),
                            depth_discount_gamma=float(depth_discount_gamma),
                        )
                        if (
                            not first_batch_local_objective_audit
                            and list(fused.get("local_objective_audit_rows") or [])
                        ):
                            first_batch_local_objective_audit = {
                                "weighting_mode": str(
                                    fused.get(
                                        "tree_local_weighting_mode",
                                        normalized_local_weighting_mode,
                                    )
                                ),
                                "local_loss_kind": str(
                                    fused.get("local_loss_kind", "")
                                ),
                                "design_name": str(
                                    fused.get("local_sampling_design_name", "")
                                ),
                                "docs": list(
                                    fused.get("local_objective_audit_rows") or []
                                )[:3],
                            }
                        _fused_c2_diag_docs = int(
                            dict(fused.get("component_counts") or {}).get(
                                "c2_count_loss",
                                0,
                            )
                        )
                        if _fused_c2_diag_docs > 0:
                            c2_same_pair_count_sum += float(
                                fused.get("c2_same_pair_count", 0.0)
                            ) * float(_fused_c2_diag_docs)
                            c2_different_pair_count_sum += float(
                                fused.get("c2_different_pair_count", 0.0)
                            ) * float(_fused_c2_diag_docs)
                            c2_pair_weight_ess_sum += float(
                                fused.get("c2_pair_weight_ess", 0.0)
                            ) * float(_fused_c2_diag_docs)
                            c2_pair_weight_max_value = max(
                                c2_pair_weight_max_value,
                                float(fused.get("c2_pair_weight_max", 0.0)),
                            )
                            c2_pair_diag_count += int(_fused_c2_diag_docs)
                        batch_document_loss_sum = (
                            batch_document_loss_sum
                            + fused.get("document_loss_sum", fused["batch_loss"])
                        )
                        batch_non_document_loss_sum = (
                            batch_non_document_loss_sum
                            + fused.get(
                                "non_document_loss_sum",
                                torch.zeros((), device=device, dtype=torch.float32),
                            )
                        )
                        for name, value in dict(fused["component_sums"]).items():
                            epoch_component_sums[name] = (
                                epoch_component_sums[name]
                                + value.detach().to(dtype=torch.float32)
                            )
                            epoch_component_counts[name] += int(
                                dict(fused["component_counts"]).get(name, 0)
                            )
                        deferred_phi_batch = fused.get("deferred_phi_feature_batch")
                        if isinstance(deferred_phi_batch, torch.Tensor) and int(deferred_phi_batch.shape[0]) > 0:
                            _batch_phi_feature_chunks.append(deferred_phi_batch)
                            deferred_fast_keys = fused.get("deferred_phi_fast_keys")
                            if isinstance(deferred_fast_keys, Mapping):
                                fast_count_values = deferred_fast_keys.get("count_values")
                                fast_first_targets = deferred_fast_keys.get("first_targets")
                                fast_last_targets = deferred_fast_keys.get("last_targets")
                                for key in tuple(_batch_phi_fast_key_chunks.keys()):
                                    value = deferred_fast_keys.get(key)
                                    if isinstance(value, torch.Tensor) and int(value.shape[0]) > 0:
                                        _batch_phi_fast_key_chunks[key].append(value)
                                if (
                                    isinstance(fast_count_values, torch.Tensor)
                                    and isinstance(fast_first_targets, torch.Tensor)
                                    and isinstance(fast_last_targets, torch.Tensor)
                                ):
                                    _batch_phi_labels.extend(
                                        _materialize_fast_markov_labels(
                                            model,
                                            count_values=fast_count_values,
                                            first_targets=fast_first_targets,
                                            last_targets=fast_last_targets,
                                        )
                                    )
                            elif fused.get("deferred_phi_features"):
                                _batch_all_deferred_chunks_are_fast_markov = False
                        elif fused.get("deferred_phi_features"):
                            _batch_phi_feature_chunks.append(
                                torch.stack(list(fused["deferred_phi_features"]), dim=0)
                            )
                            _batch_all_deferred_chunks_are_fast_markov = False
                        _batch_phi_labels.extend(list(fused["deferred_phi_labels"]))
                        _batch_oracle_vecs.extend(list(fused["deferred_oracle_vecs"]))
                    else:
                        if normalized_pack_mode == "fixed_fused" and bool(packed_tree_batch.bucket_key.auto_queue_enabled):
                            runtime_telemetry.add_extra_counter("auto_queue_generic_fallback_batches", 1.0)
                        batch_docs = [item.doc for item in packed_tree_batch.items]
                        precomputed_views: List[_PrecomputedDocStateView] | None = None
                        if len(batch_docs) > 1:
                            resident_view = _tree_store_view_for_items(
                                train_store,
                                packed_tree_batch.items,
                                model=model,
                                runtime_telemetry=runtime_telemetry,
                            )
                            precomputed_views = _precompute_balanced_doc_state_views(
                                model,
                                batch_docs,
                                device=device,
                                collect_merge_states=any(
                                    bool(item.collect_c2) or bool(item.collect_c3)
                                    for item in packed_tree_batch.items
                                ),
                                prefer_fixed_fused=False,
                                resident_view=resident_view,
                                runtime_telemetry=runtime_telemetry,
                            )
                        for group_idx, item in enumerate(packed_tree_batch.items):
                            work = work_lookup[int(item.doc_index)]
                            doc = work["doc"]
                            precomputed_view = (
                                None if precomputed_views is None else precomputed_views[group_idx]
                            )
                            out = model.forward_doc(
                                doc.leaf_token_ids,
                                doc.leaf_counts,
                                doc.merge_counts_balanced,
                                doc.merge_token_lengths,
                                schedule="balanced",
                                collect_leaf=bool(work["collect_leaf"]),
                                collect_c3=bool(work["collect_c3"]),
                                collect_c2=bool(work["collect_c2"]),
                                device=device,
                                leaf_audit_indices=work["leaf_audit_indices"],
                                c3_audit_indices=work["c3_audit_indices"],
                                leaf_first_regimes=doc.leaf_first_regimes,
                                leaf_last_regimes=doc.leaf_last_regimes,
                                internal_supervision_kind=str(internal_supervision_kind),
                                leaf_exact_supervision=bool(leaf_exact_supervision),
                                leaf_supervision_kind=str(leaf_supervision_kind),
                                tree_local_weighting_mode=normalized_local_weighting_mode,
                                tree_supervision_source=normalized_supervision_source,
                                depth_discount_gamma=float(depth_discount_gamma),
                                defer_contrastive=_use_deferred_contrastive,
                                leaf_supervision_population_size=int(
                                    work.get(
                                        "leaf_supervision_population_size",
                                        len(doc.leaf_token_ids),
                                    )
                                ),
                                internal_supervision_population_size=int(
                                    work.get(
                                        "internal_supervision_population_size",
                                        max(0, len(doc.leaf_token_ids) - 1),
                                    )
                                ),
                                precomputed_state_batch=(
                                    None
                                    if precomputed_view is None
                                    else precomputed_view.state_batch
                                ),
                                precomputed_root_state=(
                                    None
                                    if precomputed_view is None
                                    else precomputed_view.root_state
                                ),
                                precomputed_merge_states=(
                                    None
                                    if precomputed_view is None
                                    else precomputed_view.merge_states
                                ),
                            )
                            if (
                                not first_batch_local_objective_audit
                                and isinstance(out.get("local_objective_audit"), Mapping)
                            ):
                                first_batch_local_objective_audit = {
                                    "weighting_mode": str(
                                        out.get(
                                            "tree_local_weighting_mode",
                                            normalized_local_weighting_mode,
                                        )
                                    ),
                                    "local_loss_kind": str(out.get("local_loss_kind", "")),
                                    "design_name": str(
                                        out.get("local_sampling_design_name", "")
                                    ),
                                    "docs": [
                                        {
                                            "doc_index": int(item.doc_index),
                                            **dict(out.get("local_objective_audit") or {}),
                                        }
                                    ],
                                }
                            if float(out.get("c2_count", 0.0)) > 0.0:
                                c2_same_pair_count_sum += float(
                                    out.get("c2_same_pair_count", 0.0)
                                )
                                c2_different_pair_count_sum += float(
                                    out.get("c2_different_pair_count", 0.0)
                                )
                                c2_pair_weight_ess_sum += float(
                                    out.get("c2_pair_weight_ess", 0.0)
                                )
                                c2_pair_weight_max_value = max(
                                    c2_pair_weight_max_value,
                                    float(out.get("c2_pair_weight_max", 0.0)),
                                )
                                c2_pair_diag_count += 1

                            if bool(work["root_only_supervision"]):
                                pred_norm = out["pred_norm"]
                                if (
                                    str(getattr(model, "root_supervision_kind", "mse")) == "count_ce"
                                    and root_class_index
                                    and getattr(model, "root_count_classifier", None) is not None
                                ):
                                    root_logits = model.predict_root_count_logits_from_state(
                                        out["root_state"]
                                    )
                                    if root_logits.ndim == 1:
                                        root_logits = root_logits.unsqueeze(0)
                                    target_class = torch.tensor(
                                        [int(root_class_index[int(round(float(doc.root_count)))])],
                                        dtype=torch.long,
                                        device=device,
                                    )
                                    root_loss = F.cross_entropy(root_logits, target_class)
                                elif (
                                    bool(getattr(model, "use_shared_theorem_surface", False))
                                    and "root_task_target" in out
                                ):
                                    root_loss = _theorem_feature_task_supervision_terms(
                                        model,
                                        out["root_state"],
                                        truth_target=float(out["root_task_target"]),
                                    )["task_loss"]
                                elif model.use_summary_spec:
                                    if model.uses_theorem_primary_root_mode():
                                        root_loss = _summary_spec_supervision_terms(
                                            model,
                                            out["root_state"],
                                            truth_count=float(doc.root_count),
                                            supervise_count=True,
                                            supervise_endpoints=False,
                                        )["count_loss"]
                                    elif str(model.root_supervision_kind) == "count_ce" and root_class_index:
                                        root_logits = model.predict_root_count_logits_from_state(
                                            out["root_state"]
                                        )
                                        if root_logits.ndim == 1:
                                            root_logits = root_logits.unsqueeze(0)
                                        target_class = torch.tensor(
                                            [int(root_class_index[int(round(float(doc.root_count)))])],
                                            dtype=torch.long,
                                            device=device,
                                        )
                                        root_loss = F.cross_entropy(root_logits, target_class)
                                    else:
                                        pred_count = model.predict_canonical_count_from_state(
                                            out["root_state"]
                                        )
                                        target_count = torch.tensor(
                                            float(doc.root_count),
                                            device=device,
                                            dtype=pred_count.dtype,
                                        )
                                        root_loss = F.mse_loss(pred_count, target_count)
                                elif model.use_decoded_markov_sketch:
                                    pred_count = model.predict_count_from_state(out["root_state"])
                                    target_count = torch.tensor(
                                        float(doc.root_count),
                                        device=device,
                                        dtype=pred_count.dtype,
                                    )
                                    root_loss = F.mse_loss(pred_count, target_count)
                                elif str(model.root_supervision_kind) == "count_ce":
                                    root_logits = model.predict_root_count_logits_from_state(
                                        out["root_state"]
                                    )
                                    if root_logits.ndim == 1:
                                        root_logits = root_logits.unsqueeze(0)
                                    target_class = torch.tensor(
                                        [int(root_class_index[int(round(float(doc.root_count)))])],
                                        dtype=torch.long,
                                        device=device,
                                    )
                                    root_loss = F.cross_entropy(root_logits, target_class)
                                else:
                                    true_norm = torch.tensor(
                                        float(doc.root_count) / target_scale,
                                        device=device,
                                        dtype=pred_norm.dtype,
                                    )
                                    root_loss = F.mse_loss(pred_norm, true_norm)
                            else:
                                root_loss = torch.zeros(
                                    (), device=device, dtype=torch.float32
                                )
                            if bool(work["root_only_supervision"]):
                                epoch_component_sums["root_count_loss"] = (
                                    epoch_component_sums["root_count_loss"]
                                    + root_loss.detach().to(dtype=torch.float32)
                                )
                                epoch_component_counts["root_count_loss"] += 1
                            document_loss = float(root_weight) * root_loss
                            if bool(work["doc_sequence_supervision"]):
                                document_loss = document_loss + (
                                    float(root_weight) * work["doc_sequence_loss"]
                                    if explicit_supervision
                                    else work["doc_sequence_loss"]
                                )
                            # Depth discount for non-fused path.
                            # Leaf loss is a Hajek mean over all leaf nodes
                            # (all at the same depth), so a single scalar
                            # discount suffices.  C2/C3 losses mix depths —
                            # we approximate with the mean merge depth
                            # discount, which is exact when the internal
                            # Hajek weights are uniform.
                            _nf_n_leaves = max(1, len(doc.leaf_token_ids))
                            _nf_n_levels = max(0, int(math.log2(max(1, _nf_n_leaves))))
                            _nf_gamma = float(depth_discount_gamma)
                            _nf_leaf_dd = _nf_gamma ** _nf_n_levels if _nf_n_levels > 0 else 1.0
                            if _nf_n_levels > 0 and _nf_gamma < 1.0:
                                # Average gamma^d over merge depths 0..n_levels-1
                                _nf_merge_dd = sum(
                                    _nf_gamma ** d for d in range(_nf_n_levels)
                                ) / max(1, _nf_n_levels)
                            else:
                                _nf_merge_dd = 1.0
                            _leaf_loss_weight = float(c1_weight)
                            _merge_loss_weight = float(c3_weight)
                            if normalized_local_weighting_mode != "span_mass_ipw_sum":
                                _leaf_loss_weight *= float(_nf_leaf_dd)
                                _merge_loss_weight *= float(_nf_merge_dd)
                            local_loss = (
                                float(_leaf_loss_weight) * out["leaf_loss"]
                                + float(c2_weight) * out["c2_loss"]
                                + float(_merge_loss_weight) * out["c3_loss"]
                                + float(phi_compose_weight)
                                * out.get(
                                    "phi_compose_loss",
                                    torch.zeros((), device=device, dtype=document_loss.dtype),
                                )
                                + float(phi_contrastive_weight)
                                * out.get(
                                    "phi_contrastive_loss",
                                    torch.zeros((), device=device, dtype=document_loss.dtype),
                                )
                            )
                            loss_components = out.get("loss_components")
                            if isinstance(loss_components, Mapping):
                                if float(out.get("leaf_count", 0.0)) > 0.0:
                                    for name in ("leaf_count_loss", "leaf_first_loss", "leaf_last_loss"):
                                        epoch_component_sums[name] = (
                                            epoch_component_sums[name]
                                            + loss_components[name].detach().to(dtype=torch.float32)
                                        )
                                        epoch_component_counts[name] += 1
                                if float(out.get("c3_count", 0.0)) > 0.0:
                                    for name in ("merge_count_loss", "merge_first_loss", "merge_last_loss"):
                                        epoch_component_sums[name] = (
                                            epoch_component_sums[name]
                                            + loss_components[name].detach().to(dtype=torch.float32)
                                        )
                                        epoch_component_counts[name] += 1
                                if float(out.get("c2_count", 0.0)) > 0.0:
                                    for name in (
                                        "c2_count_loss",
                                        "c2_first_loss",
                                        "c2_last_loss",
                                        "c2_join_loss",
                                        "c2_on_range_reencode_loss",
                                    ):
                                        epoch_component_sums[name] = (
                                            epoch_component_sums[name]
                                            + loss_components[name].detach().to(dtype=torch.float32)
                                        )
                                        epoch_component_counts[name] += 1
                                for name in ("phi_compose_loss", "phi_contrastive_loss"):
                                    epoch_component_sums[name] = (
                                        epoch_component_sums[name]
                                        + loss_components.get(
                                            name,
                                            torch.zeros((), device=device, dtype=batch_loss.dtype),
                                        ).detach().to(dtype=torch.float32)
                                    )
                                    epoch_component_counts[name] += 1
                            batch_document_loss_sum = (
                                batch_document_loss_sum + document_loss
                            )
                            batch_non_document_loss_sum = (
                                batch_non_document_loss_sum + local_loss
                            )
                            if _use_deferred_contrastive:
                                _doc_phi = out.get("_deferred_phi_features")
                                if _doc_phi is not None:
                                    _batch_phi_feature_chunks.append(
                                        torch.stack(list(_doc_phi), dim=0)
                                    )
                                    _batch_all_deferred_chunks_are_fast_markov = False
                                    _doc_oracle = out.get("_deferred_oracle_vectors")
                                    if _doc_oracle is not None:
                                        _batch_oracle_vecs.extend(_doc_oracle)
                                    else:
                                        _doc_labels = out.get("_deferred_phi_labels")
                                        if _doc_labels is not None:
                                            _batch_phi_labels.extend(_doc_labels)
                    batching_metrics.train_forward_time_s += time.perf_counter() - forward_start_s
                    batching_metrics.add_batch(
                        packed_tree_batch,
                        token_budget=int(effective_train_token_budget),
                        node_budget=int(effective_train_node_budget),
                        max_docs_budget=int(effective_train_max_docs),
                    )

                # Batch-level contrastive loss over all accumulated phi features
                deferred_feature_count = int(
                    sum(int(chunk.shape[0]) for chunk in _batch_phi_feature_chunks)
                )
                if _use_deferred_contrastive and deferred_feature_count > 1:
                    phi_feature_batch = torch.cat(_batch_phi_feature_chunks, dim=0)
                    if _batch_oracle_vecs:
                        # Theory-aligned path: continuous oracle distances
                        _batch_pair_data = build_contrastive_pairs(
                            _batch_oracle_vecs,
                            metric=model.oracle_metric,
                            same_threshold=model.oracle_same_threshold,
                            diff_threshold=model.oracle_diff_threshold,
                        )
                        _batch_contrastive = contrastive_fiber_loss(
                            phi_feature_batch,
                            _batch_pair_data,
                            margin=float(SUMMARY_SPEC_PHI_DIFFERENT_MARGIN),
                        )
                    elif (
                        _batch_all_deferred_chunks_are_fast_markov
                        and all(_batch_phi_fast_key_chunks[key] for key in _batch_phi_fast_key_chunks)
                    ):
                        flat_count_values = torch.cat(_batch_phi_fast_key_chunks["count_values"], dim=0)
                        flat_count_keys = torch.cat(_batch_phi_fast_key_chunks["count_keys"], dim=0)
                        flat_first_targets = torch.cat(_batch_phi_fast_key_chunks["first_targets"], dim=0)
                        flat_last_targets = torch.cat(_batch_phi_fast_key_chunks["last_targets"], dim=0)
                        same_mask, different_mask = _fast_markov_pair_masks_from_tensors(
                            model,
                            count_keys=flat_count_keys,
                            first_targets=flat_first_targets,
                            last_targets=flat_last_targets,
                        )
                        _batch_contrastive = _pairwise_theorem_feature_contrastive_loss_from_masks(
                            phi_feature_batch,
                            same_mask=same_mask,
                            different_mask=different_mask,
                        )
                    else:
                        # Legacy adapter path: discrete pair sets
                        _batch_pairs = build_theorem_feature_pair_sets(
                            _batch_phi_labels,
                            adapter=model.theorem_feature_adapter,
                            same_threshold=model.theorem_pair_same_threshold,
                            diff_threshold=model.theorem_pair_diff_threshold,
                        )
                        _batch_contrastive = _pairwise_theorem_feature_contrastive_loss(
                            phi_feature_batch,
                            same_pairs=_batch_pairs.same_pairs,
                            different_pairs=_batch_pairs.different_pairs,
                        )
                    batch_non_document_loss_sum = (
                        batch_non_document_loss_sum
                        + float(phi_contrastive_weight) * _batch_contrastive
                    )
                    epoch_component_sums["phi_contrastive_loss"] = (
                        epoch_component_sums["phi_contrastive_loss"]
                        + _batch_contrastive.detach().to(dtype=torch.float32)
                    )

            last_document_loss_batch_scale = _tree_document_loss_batch_scale(
                normalization_mode=effective_document_loss_normalization_mode,
                batch_docs=int(len(batch_idx)),
                supervised_docs=int(batch_document_supervision_count),
            )
            document_loss_batch_scale_sum += float(last_document_loss_batch_scale)
            document_loss_batch_scale_count += 1
            batch_loss = (
                batch_non_document_loss_sum
                + batch_document_loss_sum * float(last_document_loss_batch_scale)
            ) / float(len(batch_idx))
            if bool(getattr(batch_loss, "requires_grad", False)):
                backward_start_s = time.perf_counter()
                _emit_memory_probe(
                    "pre_batch_backward",
                    epoch=int(epoch),
                    batch_start=int(b0),
                    batch_size=int(len(batch_idx)),
                )
                batch_loss.backward()
                _emit_memory_probe(
                    "post_batch_backward",
                    epoch=int(epoch),
                    batch_start=int(b0),
                    batch_size=int(len(batch_idx)),
                )
                # -- Gradient-norm diagnostics (sampled every 50 batches) --
                _grad_diag_step = int(epoch) * 10000 + int(batch_loss_count)
                if _grad_diag_step % 50 == 0:
                    _gn_leaf_enc = 0.0
                    _gn_merge = 0.0
                    _gn_root_head = 0.0
                    _gn_other = 0.0
                    for _pname, _param in model.named_parameters():
                        if _param.grad is None:
                            continue
                        _gnorm = float(torch.norm(_param.grad).item())
                        _pn = str(_pname).lower()
                        if "leaf_fno" in _pn or "encoder" in _pn or "leaf_token" in _pn:
                            _gn_leaf_enc += _gnorm
                        elif "merge" in _pn:
                            _gn_merge += _gnorm
                        elif "task_head" in _pn or "theorem" in _pn or "root_count" in _pn:
                            _gn_root_head += _gnorm
                        else:
                            _gn_other += _gnorm
                    epoch_grad_diagnostics.append({
                        "epoch": int(epoch),
                        "step": int(_grad_diag_step),
                        "leaf_encoder_grad_norm": _gn_leaf_enc,
                        "merge_grad_norm": _gn_merge,
                        "root_head_grad_norm": _gn_root_head,
                        "other_grad_norm": _gn_other,
                    })
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                opt.step()
                batching_metrics.train_backward_time_s += time.perf_counter() - backward_start_s
            batch_loss_total = batch_loss_total + batch_loss.detach().to(dtype=torch.float32)
            batch_loss_count += 1
            elapsed_train_loop_s += time.perf_counter() - _batch_start_s

        scheduler.step()
        loss_curve.append(
            float(
                (
                    batch_loss_total
                    / float(max(1, int(batch_loss_count)))
                ).detach().cpu()
            )
        )
        for name in component_curve_names:
            if int(epoch_component_counts[name]) > 0:
                component_loss_curves[name].append(
                    float(
                        (
                            epoch_component_sums[name]
                            / float(epoch_component_counts[name])
                        ).detach().cpu()
                    )
                )
            else:
                component_loss_curves[name].append(float("nan"))
        all_grad_diagnostics.extend(epoch_grad_diagnostics)

        if val_docs:
            should_run_screen_eval = bool(eval_screen_docs) and (
                normalized_eval_mode == "per_epoch"
                or int(epoch) == int(max(0, n_epochs) - 1)
            )
            if should_run_screen_eval:
                _emit_memory_probe(
                    "pre_eval_fno_model",
                    epoch=int(epoch),
                    split="val",
                    doc_limit=int(screen_doc_limit),
                    eval_mode=str(normalized_eval_mode),
                )
                _screen_eval_start_s = time.perf_counter()
                tree_val_metrics = _eval_fno_root_only_metrics(
                    model,
                    eval_screen_docs,
                    device=device,
                    resident_store=val_store,
                    runtime_telemetry=runtime_telemetry,
                    pack_mode=str(normalized_pack_mode),
                    runtime_bucket_mode=str(effective_runtime_config.bucket_mode),
                    max_docs=int(effective_eval_max_docs),
                    token_budget=int(effective_eval_token_budget),
                    node_budget=int(effective_eval_node_budget),
                    bucket_docs_cap_by_n_leaves=eval_bucket_docs_cap_by_n_leaves,
                    structural_pad_limit=float(tree_batch_structural_pad_limit),
                    auto_queue_min_docs=int(tree_batch_auto_queue_min_docs),
                    auto_queue_min_fill_ratio=float(tree_batch_auto_queue_min_fill_ratio),
                    auto_queue_target_by_n_leaves=val_auto_queue_target_by_n_leaves,
                    batching_metrics=batching_metrics,
                )
                _emit_memory_probe(
                    "post_eval_fno_model",
                    epoch=int(epoch),
                    split="val",
                    doc_limit=int(screen_doc_limit),
                    eval_mode=str(normalized_eval_mode),
                )
                doc_sequence_val_metrics = (
                    _eval_fno_doc_sequence_view(
                        model,
                        eval_screen_docs,
                        device=device,
                        tau=0.0,
                    )
                    if doc_sequence_fraction > 0.0 and not explicit_supervision
                    else tree_val_metrics
                )
                latest_tree_val_mae = float(
                    getattr(tree_val_metrics, "root_mae", float("nan"))
                )
                latest_tree_val_exact_match = float(
                    getattr(tree_val_metrics, "exact_match", float("nan"))
                )
                latest_doc_sequence_val_mae = float(
                    getattr(doc_sequence_val_metrics, "root_mae", float("nan"))
                )
                latest_doc_sequence_val_exact_match = float(
                    getattr(doc_sequence_val_metrics, "exact_match", float("nan"))
                )
                elapsed_screen_eval_s += time.perf_counter() - _screen_eval_start_s
                if normalized_checkpoint_metric in {
                    "val_leaf_codec_direct",
                    "val_theorem_bootstrap_direct",
                    "val_exact_sketch_direct",
                    "val_task_root_exact_sketch_direct",
                }:
                    should_run_exact_eval = (
                        int(epoch) == int(max(0, n_epochs) - 1)
                        or (
                            normalized_eval_mode == "per_epoch"
                            and (
                                int(epoch)
                                % max(1, int(exact_metric_selection_interval))
                                == 0
                            )
                        )
                    )
                    selection_value = float("inf")
                    if should_run_exact_eval:
                        _emit_memory_probe(
                            "pre_eval_fno_exact_sketch_direct_metrics",
                            epoch=int(epoch),
                            split="val",
                            doc_limit=int(exact_metric_selection_doc_limit),
                        )
                        _exact_eval_start_s = time.perf_counter()
                        exact_sketch_val = _run_exact_metric_eval(
                            eval_screen_docs,
                            doc_limit=(
                                None
                                if int(exact_metric_selection_doc_limit) <= 0
                                else int(exact_metric_selection_doc_limit)
                            ),
                        )
                        elapsed_exact_metric_eval_s += (
                            time.perf_counter() - _exact_eval_start_s
                        )
                        _emit_memory_probe(
                            "post_eval_fno_exact_sketch_direct_metrics",
                            epoch=int(epoch),
                            split="val",
                            doc_limit=int(exact_metric_selection_doc_limit),
                        )
                        _trim_host_allocator()
                        selection_value = float(
                            exact_sketch_val[normalized_checkpoint_metric]
                        )
                    selection_metadata = TrainingSelectionMetadata(
                        mode=(
                            "best_val_leaf_codec_direct"
                            if normalized_checkpoint_metric == "val_leaf_codec_direct"
                            else (
                                "best_val_theorem_bootstrap_direct"
                                if normalized_checkpoint_metric
                                == "val_theorem_bootstrap_direct"
                                else (
                                    "best_val_exact_sketch_direct"
                                    if normalized_checkpoint_metric
                                    == "val_exact_sketch_direct"
                                    else "best_val_task_root_exact_sketch_direct"
                                )
                            )
                        ),
                        split="val",
                        metric_name=str(normalized_checkpoint_metric),
                        metric_value=float(selection_value),
                        best_epoch=int(epoch),
                    )
                else:
                    if explicit_supervision:
                        selection_value = float(tree_val_metrics.root_mae)
                    else:
                        selection_value = _curriculum_selection_value(
                            tree_root_mae=float(tree_val_metrics.root_mae),
                            doc_sequence_root_mae=float(
                                doc_sequence_val_metrics.root_mae
                            ),
                            doc_sequence_fraction=float(doc_sequence_fraction),
                        )
                    selection_metadata = TrainingSelectionMetadata(
                        mode=str(root_selection_mode),
                        split="val",
                        metric_name=str(root_selection_metric_name),
                        metric_value=float(selection_value),
                        best_epoch=int(epoch),
                    )
                selection_curve.append(float(selection_value))
                if improved_metric(
                    float(selection_value), float(best_selection.metric_value)
                ):
                    best_selection = selection_metadata
                    best_tree_val_mae = float(tree_val_metrics.root_mae)
                    best_doc_sequence_val_mae = float(
                        doc_sequence_val_metrics.root_mae
                    )
                    _emit_memory_probe(
                        "pre_clone_module_state",
                        location="best_selection_update",
                        epoch=int(epoch),
                    )
                    _clone_start_s = time.perf_counter()
                    best_state = clone_module_state(model)
                    elapsed_state_clone_s += time.perf_counter() - _clone_start_s
                    _emit_memory_probe(
                        "post_clone_module_state",
                        location="best_selection_update",
                        epoch=int(epoch),
                    )
            else:
                selection_curve.append(float("nan"))

        _emit_progress_snapshot(
            f"{normalized_progress_stage}_train",
            epoch_completed=int(epoch) + 1,
        )

    if val_docs:
        restore_module_state(model, best_state)
    else:
        _emit_memory_probe("pre_clone_module_state", location="final_no_validation")
        _clone_start_s = time.perf_counter()
        best_state = clone_module_state(model)
        elapsed_state_clone_s += time.perf_counter() - _clone_start_s
        _emit_memory_probe("post_clone_module_state", location="final_no_validation")
        best_epoch = max(0, int(n_epochs) - 1)
        if normalized_checkpoint_metric in {
            "val_leaf_codec_direct",
            "val_theorem_bootstrap_direct",
            "val_exact_sketch_direct",
            "val_task_root_exact_sketch_direct",
        }:
            _emit_memory_probe(
                "pre_eval_fno_exact_sketch_direct_metrics",
                epoch=int(best_epoch),
                split="train",
                doc_limit=int(exact_metric_selection_doc_limit),
            )
            _exact_eval_start_s = time.perf_counter()
            train_exact = _run_exact_metric_eval(
                train_docs,
                doc_limit=(
                    None
                    if int(exact_metric_selection_doc_limit) <= 0
                    else int(exact_metric_selection_doc_limit)
                ),
            )
            elapsed_exact_metric_eval_s += time.perf_counter() - _exact_eval_start_s
            _emit_memory_probe(
                "post_eval_fno_exact_sketch_direct_metrics",
                epoch=int(best_epoch),
                split="train",
                doc_limit=int(exact_metric_selection_doc_limit),
            )
            _trim_host_allocator()
            best_selection = TrainingSelectionMetadata(
                mode="final_epoch_no_validation",
                split="config",
                metric_name=(
                    "train_leaf_codec_direct"
                    if normalized_checkpoint_metric == "val_leaf_codec_direct"
                    else (
                        "train_theorem_bootstrap_direct"
                        if normalized_checkpoint_metric == "val_theorem_bootstrap_direct"
                        else (
                            "train_exact_sketch_direct"
                            if normalized_checkpoint_metric == "val_exact_sketch_direct"
                            else "train_task_root_exact_sketch_direct"
                        )
                    )
                ),
                metric_value=float(train_exact[normalized_checkpoint_metric]),
                best_epoch=int(best_epoch),
            )

    _emit_progress_snapshot(
        f"{normalized_progress_stage}_final_eval",
        epoch_completed=int(len(loss_curve)),
    )

    exact_metrics_split = "val" if val_docs else "train"
    exact_metrics_docs = val_docs if val_docs else train_docs
    _emit_memory_probe(
        "pre_eval_fno_exact_sketch_direct_metrics",
        epoch=int(best_selection.best_epoch),
        split=str(exact_metrics_split),
        doc_limit=int(final_exact_doc_limit),
        phase="final_exact_metrics",
    )
    _exact_eval_start_s = time.perf_counter()
    final_exact_metrics = _run_exact_metric_eval(
        exact_metrics_docs,
        doc_limit=(
            None
            if int(final_exact_doc_limit) <= 0
            else int(final_exact_doc_limit)
        ),
    )
    elapsed_exact_metric_eval_s += time.perf_counter() - _exact_eval_start_s
    _emit_memory_probe(
        "post_eval_fno_exact_sketch_direct_metrics",
        epoch=int(best_selection.best_epoch),
        split=str(exact_metrics_split),
        doc_limit=int(final_exact_doc_limit),
        phase="final_exact_metrics",
    )
    _trim_host_allocator()
    gpu_reserved_peak_gb, gpu_allocated_peak_gb = _gpu_peak_memory_gb(device)

    def _eval_split(docs):
        if hasattr(model, "encode_leaf_tokens_batch") and hasattr(model, "_merge_states"):
            p, t = _batched_root_predictions(
                model,
                docs,
                device=device,
                resident_store=_resident_store_for_docs(docs),
                runtime_telemetry=runtime_telemetry,
                pack_mode=str(normalized_pack_mode),
                runtime_bucket_mode=str(effective_runtime_config.bucket_mode),
                max_docs=int(effective_eval_max_docs),
                token_budget=int(effective_eval_token_budget),
                node_budget=int(effective_eval_node_budget),
                bucket_docs_cap_by_n_leaves=eval_bucket_docs_cap_by_n_leaves,
                structural_pad_limit=float(tree_batch_structural_pad_limit),
                auto_queue_min_docs=int(tree_batch_auto_queue_min_docs),
                auto_queue_min_fill_ratio=float(tree_batch_auto_queue_min_fill_ratio),
                auto_queue_target_by_n_leaves=_auto_queue_targets_for_docs(docs),
                batching_metrics=batching_metrics,
            )
        else:
            model.eval()
            preds, truths = [], []
            with torch.no_grad():
                for doc in docs:
                    out = model.forward_doc(
                        doc.leaf_token_ids,
                        doc.leaf_counts,
                        doc.merge_counts_balanced,
                        doc.merge_token_lengths,
                        schedule="balanced", collect_leaf=False, collect_c3=False,
                        collect_c2=False, device=device,
                    )
                    preds.append(
                        float(
                            model.predict_canonical_count_from_state(out["root_state"])
                            .detach()
                            .cpu()
                        )
                    )
                    truths.append(float(doc.root_count))
            p, t = np.array(preds), np.array(truths)
        return {
            "root_mae": float(np.mean(np.abs(p - t))),
            "exact_match": float(np.mean((np.rint(p) == np.rint(t)).astype(np.float64))),
        }

    _emit_memory_probe("pre_eval_split", split="train")
    _split_eval_start_s = time.perf_counter()
    train_eval = _eval_split(train_docs)
    elapsed_split_eval_s += time.perf_counter() - _split_eval_start_s
    _emit_memory_probe("post_eval_split", split="train")
    if val_docs:
        _emit_memory_probe("pre_eval_split", split="val")
        _split_eval_start_s = time.perf_counter()
        val_eval = _eval_split(val_docs)
        elapsed_split_eval_s += time.perf_counter() - _split_eval_start_s
        _emit_memory_probe("post_eval_split", split="val")
    else:
        val_eval = {"root_mae": float("nan"), "exact_match": float("nan")}
    if not val_docs and normalized_checkpoint_metric not in {
        "val_leaf_codec_direct",
        "val_theorem_bootstrap_direct",
        "val_exact_sketch_direct",
        "val_task_root_exact_sketch_direct",
    }:
        best_selection = TrainingSelectionMetadata(
            mode="final_epoch_no_validation",
            split="config",
            metric_name="train_root_mae",
            metric_value=float(train_eval["root_mae"]),
            best_epoch=int(max(0, int(len(loss_curve) - 1))),
        )
    fit_diag = TrainFitDiagnostics(
        train_loss_final=float(loss_curve[-1]) if loss_curve else float("nan"),
        train_loss_curve=tuple(float(x) for x in loss_curve),
        epochs_completed=int(len(loss_curve)),
        selection_metric_curve=tuple(float(x) for x in selection_curve),
        selection_mode=str(best_selection.mode),
        selection_split=str(best_selection.split),
        selection_metric_name=str(best_selection.metric_name),
        selection_metric_value=float(best_selection.metric_value),
        best_epoch=int(best_selection.best_epoch),
        train_exact_match_rate=float(train_eval["exact_match"]),
        val_exact_match_rate=float(val_eval["exact_match"]),
        test_exact_match_rate=float("nan"),
        stage1_selection_metric_curve=tuple(),
        stage2_selection_metric_curve=tuple(),
        stage1_selection_metric_name="",
        stage2_selection_metric_name="",
        training_schedule="single_stage",
    )
    training_component_loss_finals = {
        name: (
            float(values[-1])
            if values
            else float("nan")
        )
        for name, values in component_loss_curves.items()
    }
    normalized_root_contribution_final = (
        float(root_weight)
        * float(training_component_loss_finals.get("root_count_loss", float("nan")))
    )
    mean_document_loss_batch_scale = (
        float(document_loss_batch_scale_sum) / float(max(1, document_loss_batch_scale_count))
    )
    runtime_efficiency = runtime_telemetry.as_dict(
        gpu_reserved_peak_gb=gpu_reserved_peak_gb,
        gpu_allocated_peak_gb=gpu_allocated_peak_gb,
    )
    runtime_efficiency.update(
        {
            "tree_document_loss_normalization_mode": str(
                _normalize_tree_document_loss_normalization_mode(
                    tree_document_loss_normalization_mode
                )
            ),
            "effective_tree_document_loss_normalization_mode": str(
                effective_document_loss_normalization_mode
            ),
            "document_supervision_docs_total": int(
                document_supervision_docs_total
            ),
            "root_supervision_docs_total": int(root_supervision_docs_total),
            "doc_sequence_supervision_docs_total": int(
                doc_sequence_supervision_docs_total
            ),
            "document_supervision_coverage_rate": float(
                document_supervision_coverage_rate
            ),
            "document_loss_mean_batch_scale": float(
                mean_document_loss_batch_scale
            ),
            "document_loss_last_batch_scale": float(
                last_document_loss_batch_scale
            ),
            "normalized_root_contribution_final": float(
                normalized_root_contribution_final
            ),
        }
    )
    return {
        "train": train_eval,
        "val": val_eval,
        "fit_diag": fit_diag,
        "best_epoch": int(best_selection.best_epoch),
        "best_val_mae": float(best_selection.metric_value),
        "best_tree_val_mae": best_tree_val_mae,
        "best_doc_sequence_val_mae": best_doc_sequence_val_mae,
        "selection_mode": str(best_selection.mode),
        "selection_split": str(best_selection.split),
        "selection_metric_name": str(best_selection.metric_name),
        "selection_metric_curve": tuple(float(x) for x in selection_curve),
        "loss_curve": tuple(float(x) for x in loss_curve),
        "grad_diagnostics": list(all_grad_diagnostics),
        "epochs_completed": int(len(loss_curve)),
        "training_component_loss_curves": {
            name: tuple(float(x) for x in values)
            for name, values in component_loss_curves.items()
        },
        "training_component_loss_finals": training_component_loss_finals,
        "doc_sequence_train_docs_used": int(
            len(doc_sequence_train_indices)
            if explicit_doc_modes is None
            else sum(1 for mode in explicit_doc_modes.values() if str(mode) == "doc_sequence")
        ),
        "tree_document_loss_normalization_mode": str(
            _normalize_tree_document_loss_normalization_mode(
                tree_document_loss_normalization_mode
            )
        ),
        "effective_tree_document_loss_normalization_mode": str(
            effective_document_loss_normalization_mode
        ),
        "document_supervision_docs_total": int(document_supervision_docs_total),
        "root_supervision_docs_total": int(root_supervision_docs_total),
        "doc_sequence_supervision_docs_total": int(
            doc_sequence_supervision_docs_total
        ),
        "document_supervision_coverage_rate": float(
            document_supervision_coverage_rate
        ),
        "document_loss_mean_batch_scale": float(mean_document_loss_batch_scale),
        "document_loss_last_batch_scale": float(last_document_loss_batch_scale),
        "normalized_root_contribution_final": float(
            normalized_root_contribution_final
        ),
        "tree_local_weighting_mode": str(normalized_local_weighting_mode),
        "tree_supervision_source": str(normalized_supervision_source),
        "local_estimand_mode": str(normalized_local_weighting_mode),
        "depth_discount_gamma": float(depth_discount_gamma),
        "c2_pair_weighting_mode": str(c2_pair_weighting_mode),
        "c2_same_pair_count": float(
            _mean_or_default(
                total=c2_same_pair_count_sum,
                count=c2_pair_diag_count,
                default=0.0,
            )
        ),
        "c2_different_pair_count": float(
            _mean_or_default(
                total=c2_different_pair_count_sum,
                count=c2_pair_diag_count,
                default=0.0,
            )
        ),
        "c2_pair_weight_ess": float(
            _mean_or_default(
                total=c2_pair_weight_ess_sum,
                count=c2_pair_diag_count,
                default=0.0,
            )
        ),
        "c2_pair_weight_max": float(c2_pair_weight_max_value),
        "local_loss_kind": _resolved_local_loss_kind(
            leaf_supervision_kind=str(leaf_supervision_kind),
            internal_supervision_kind=str(internal_supervision_kind),
        ),
        "local_sampling_design_name": (
            "manifest_explicit_deterministic_ordering"
            if normalized_supervision_source == "manifest"
            else "deterministic_fixed_k_uniform"
        ),
        "leaf_population_size": float(leaf_population_size_mean),
        "leaf_sample_size": float(leaf_sample_size_mean),
        "leaf_effective_propensity": float(leaf_effective_propensity_mean),
        "merge_population_size": float(merge_population_size_mean),
        "merge_sample_size": float(merge_sample_size_mean),
        "merge_effective_propensity": float(merge_effective_propensity_mean),
        "local_objective_audit": (
            {
                **dict(first_batch_local_objective_audit or {}),
                "design_name": "manifest_explicit_deterministic_ordering",
            }
            if normalized_supervision_source == "manifest"
            and dict(first_batch_local_objective_audit or {})
            else dict(first_batch_local_objective_audit or {})
        ),
        "n_params": sum(p.numel() for p in model.parameters()),
        "best_model_state": best_state,
        "best_exact_metrics": dict(final_exact_metrics),
        "best_exact_metrics_split": str(exact_metrics_split),
        "elapsed_s_train_loop": float(elapsed_train_loop_s),
        "elapsed_s_screen_eval": float(elapsed_screen_eval_s),
        "elapsed_s_exact_metric_eval": float(elapsed_exact_metric_eval_s),
        "elapsed_s_split_eval": float(elapsed_split_eval_s),
        "elapsed_s_state_clone": float(elapsed_state_clone_s),
        "timing_breakdown": {
            "train_loop_s": float(elapsed_train_loop_s),
            "screen_eval_s": float(elapsed_screen_eval_s),
            "exact_metric_eval_s": float(elapsed_exact_metric_eval_s),
            "split_eval_s": float(elapsed_split_eval_s),
            "state_clone_s": float(elapsed_state_clone_s),
            "autotune_heuristic_s": float(autotune_probe_diagnostics.heuristic_time_s),
            "autotune_train_probe_s": float(
                autotune_probe_diagnostics.train_probe_time_s
            ),
            "autotune_eval_probe_s": float(
                autotune_probe_diagnostics.eval_probe_time_s
            ),
            "autotune_cache_lookup_s": float(
                autotune_probe_diagnostics.cache_lookup_time_s
            ),
            "autotune_cache_write_s": float(
                autotune_probe_diagnostics.cache_write_time_s
            ),
            "autotune_total_s": float(
                autotune_probe_diagnostics.heuristic_time_s
                + autotune_probe_diagnostics.train_probe_time_s
                + autotune_probe_diagnostics.eval_probe_time_s
                + autotune_probe_diagnostics.cache_lookup_time_s
                + autotune_probe_diagnostics.cache_write_time_s
            ),
            "eval_total_s": float(
                elapsed_screen_eval_s
                + elapsed_exact_metric_eval_s
                + elapsed_split_eval_s
            ),
        },
        "runtime_efficiency": runtime_efficiency,
        "progress_snapshot_interval": int(effective_progress_snapshot_interval),
        "progress_snapshot_dir": (
            str(progress_snapshot_root) if progress_snapshot_root is not None else ""
        ),
        "latest_progress_snapshot_path": str(latest_progress_snapshot_path),
        "progress_snapshot_paths": tuple(
            str(path) for path in progress_snapshot_paths
        ),
        "batching_metrics": batching_metrics.as_dict(
            device=device,
            runtime_telemetry=runtime_telemetry,
        ),
        "autotuned_batch_budgets": {
            "train_leaf_token_budget": int(effective_train_token_budget),
            "train_node_budget": int(effective_train_node_budget),
            "eval_leaf_token_budget": int(effective_eval_token_budget),
            "eval_node_budget": int(effective_eval_node_budget),
            "eval_workers_per_mig": int(effective_eval_workers_per_mig),
            "tree_batch_pack_mode": str(normalized_pack_mode),
            "tree_batch_autotune": bool(tree_batch_autotune),
            "gpu_runtime_config": dict(effective_runtime_config.as_dict()),
            "configured_train_batch_size_docs": int(configured_train_max_docs),
            "effective_train_max_docs": int(effective_train_max_docs),
            "configured_exact_eval_max_docs": int(tree_exact_eval_max_docs),
            "effective_exact_eval_max_docs": int(effective_eval_max_docs),
            "tree_batch_structural_pad_limit": float(tree_batch_structural_pad_limit),
            "tree_batch_auto_queue_min_docs": int(tree_batch_auto_queue_min_docs),
            "tree_batch_auto_queue_min_fill_ratio": float(tree_batch_auto_queue_min_fill_ratio),
            "auto_queue_target_leaf_counts": [int(value) for value in auto_queue_target_leaf_counts],
            "train_auto_queue_target_by_n_leaves": {
                str(int(key)): int(value)
                for key, value in train_auto_queue_target_by_n_leaves.items()
            },
            "val_auto_queue_target_by_n_leaves": {
                str(int(key)): int(value)
                for key, value in val_auto_queue_target_by_n_leaves.items()
            },
            "train_bucket_max_docs_by_n_leaves": {
                str(int(key)): int(value)
                for key, value in train_bucket_docs_cap_by_n_leaves.items()
            },
            "train_bucket_max_docs_by_n_leaves_raw": {
                str(int(key)): int(value)
                for key, value in raw_train_bucket_docs_cap_by_n_leaves.items()
            },
            "eval_bucket_max_docs_by_n_leaves": {
                str(int(key)): int(value)
                for key, value in eval_bucket_docs_cap_by_n_leaves.items()
            },
            "probe_cache_version": int(AUTOTUNE_PROBE_CACHE_VERSION),
            "probe_cache_hits": int(autotune_probe_diagnostics.cache_hits),
            "probe_cache_misses": int(autotune_probe_diagnostics.cache_misses),
            "probe_cache_writes": int(autotune_probe_diagnostics.cache_writes),
            "probe_run_count": int(autotune_probe_diagnostics.probe_runs),
            "probe_candidate_count": int(
                autotune_probe_diagnostics.probe_candidate_evals
            ),
        },
        "autotune_probe_profile": autotune_probe_diagnostics.as_dict(),
    }


def train_fno_tree(
    *,
    model: FNOCountSketch,
    train_docs: Sequence[_FNOCountDoc],
    val_docs: Sequence[_FNOCountDoc],
    device: torch.device,
    n_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 1e-5,
    c1_weight: float = 0.0,
    c2_weight: float = 0.0,
    c3_weight: float = 0.0,
    root_weight: float = 1.0,
    leaf_query_rate: float = 1.0,
    leaf_label_rate: float | None = None,
    audit_fraction: float = 0.2,
    doc_sequence_train_fraction: float = 0.0,
    doc_sequence_objective: str = "count_ce_only",
    doc_sequence_class_index: Mapping[int, int] | None = None,
    root_class_index: Mapping[int, int] | None = None,
    document_supervision_mode_by_doc: Mapping[int, str] | None = None,
    tree_document_loss_normalization_mode: str = "auto",
    leaf_audit_indices_by_doc: Mapping[int, Sequence[int]] | None = None,
    c3_audit_indices_by_doc: Mapping[int, Sequence[int]] | None = None,
    internal_supervision_kind: str = "count_only",
    internal_label_rate: float = 0.0,
    max_internal_depth: int = 0,
    leaf_exact_supervision: bool = False,
    leaf_supervision_kind: str = "full_sketch",
    tree_local_weighting_mode: str = "fixed_k_hajek",
    tree_supervision_source: str = "rate",
    phi_compose_weight: float = 1.0,
    phi_contrastive_weight: float = 0.25,
    checkpoint_metric: str = "val_root_mae",
    tree_training_schedule: str = "single_stage",
    tree_stage1_epochs: int = 0,
    tree_stage2_epochs: int = 0,
    tree_stage1_checkpoint_metric: str = "val_root_mae",
    tree_stage1_eval_mode: str = "per_epoch",
    tree_stage1_screen_doc_limit: int = 0,
    tree_stage1_final_exact_doc_limit: int = 0,
    tree_batch_pack_mode: str = "structure_bucket",
    tree_batch_token_budget: int = 0,
    tree_batch_node_budget: int = 0,
    tree_batch_autotune: bool = True,
    tree_batch_structural_pad_limit: float = 0.5,
    tree_batch_auto_queue_min_docs: int = 8,
    tree_batch_auto_queue_min_fill_ratio: float = 0.5,
    tree_eval_workers_per_mig: int = 0,
    tree_stage1_artifact_dir: str = "",
    tree_stage1_resume_if_available: bool = False,
    tree_stage1_root_weight: float = 0.0,
    tree_summary_spec_root_mode: str = "task_split_ablation",
    exact_metric_evaluator: Callable[..., Dict[str, float]] | None = None,
    exact_metric_selection_doc_limit: int = 0,
    exact_metric_selection_interval: int = 1,
    exact_metric_phi_pair_calibration_max_nodes: int | None = 512,
    exact_metric_final_doc_limit: int = 0,
    tree_exact_eval_max_docs: int = 0,
    runtime_config: GpuRuntimeConfig | None = None,
    leaf_sample_ordering_by_doc: Mapping[int, Sequence[int]] | None = None,
    internal_sample_ordering_by_doc: Mapping[int, Sequence[int]] | None = None,
    memory_probe: Callable[[str, Mapping[str, Any]], None] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    progress_snapshot_interval: int = 10,
    progress_snapshot_dir: str | Path = "",
    grad_clip_norm: float = 1.0,
    depth_discount_gamma: float = 1.0,
    seed: int = 42,
) -> Dict[str, object]:
    normalized_leaf_supervision = (
        str(leaf_supervision_kind or "full_sketch").strip().lower() or "full_sketch"
    )
    normalized_local_weighting_mode = _normalize_tree_local_weighting_mode(
        tree_local_weighting_mode
    )
    normalized_supervision_source = _normalize_tree_supervision_source(
        tree_supervision_source
    )
    if normalized_leaf_supervision not in VALID_LEAF_SUPERVISION_KINDS:
        raise ValueError(
            "leaf_supervision_kind must be one of "
            f"{VALID_LEAF_SUPERVISION_KINDS}; got {leaf_supervision_kind!r}"
        )
    normalized_schedule = (
        str(tree_training_schedule or "single_stage").strip().lower() or "single_stage"
    )
    if normalized_schedule not in VALID_TREE_TRAINING_SCHEDULES:
        raise ValueError(
            "tree_training_schedule must be one of "
            f"{VALID_TREE_TRAINING_SCHEDULES}; got {tree_training_schedule!r}"
        )
    normalized_root_mode = (
        str(tree_summary_spec_root_mode or "task_split_ablation").strip().lower()
        or "task_split_ablation"
    )
    if normalized_root_mode not in VALID_TREE_SUMMARY_SPEC_ROOT_MODES:
        raise ValueError(
            "tree_summary_spec_root_mode must be one of "
            f"{VALID_TREE_SUMMARY_SPEC_ROOT_MODES}; got {tree_summary_spec_root_mode!r}"
        )
    if bool(getattr(model, "uses_hybrid_ordinal_count_head", lambda: False)()):
        threshold_weights = _theorem_count_threshold_pos_weights_from_docs(
            train_docs,
            max_count=int(getattr(model, "theorem_count_threshold_count")()),
        )
        if not bool(getattr(model, "theorem_count_threshold_balance", True)):
            threshold_weights = np.ones_like(threshold_weights, dtype=np.float32)
        getattr(model, "set_theorem_count_threshold_pos_weight")(threshold_weights)
    if normalized_schedule == "single_stage":
        return _train_fno_tree_single_stage(
            model=model,
            train_docs=train_docs,
            val_docs=val_docs,
            device=device,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            c1_weight=c1_weight,
            c2_weight=c2_weight,
            c3_weight=c3_weight,
            root_weight=root_weight,
            leaf_query_rate=leaf_query_rate,
            leaf_label_rate=leaf_label_rate,
            audit_fraction=audit_fraction,
            doc_sequence_train_fraction=doc_sequence_train_fraction,
            doc_sequence_objective=doc_sequence_objective,
            doc_sequence_class_index=doc_sequence_class_index,
            root_class_index=root_class_index,
            document_supervision_mode_by_doc=document_supervision_mode_by_doc,
            tree_document_loss_normalization_mode=tree_document_loss_normalization_mode,
            leaf_audit_indices_by_doc=leaf_audit_indices_by_doc,
            c3_audit_indices_by_doc=c3_audit_indices_by_doc,
            internal_supervision_kind=internal_supervision_kind,
            internal_label_rate=internal_label_rate,
            max_internal_depth=max_internal_depth,
            leaf_exact_supervision=leaf_exact_supervision,
            leaf_supervision_kind=normalized_leaf_supervision,
            tree_local_weighting_mode=normalized_local_weighting_mode,
            tree_supervision_source=normalized_supervision_source,
            phi_compose_weight=phi_compose_weight,
            phi_contrastive_weight=phi_contrastive_weight,
            checkpoint_metric=checkpoint_metric,
            eval_mode=tree_stage1_eval_mode,
            screen_doc_limit=tree_stage1_screen_doc_limit,
            final_exact_doc_limit_override=tree_stage1_final_exact_doc_limit,
            tree_batch_pack_mode=tree_batch_pack_mode,
            tree_batch_token_budget=tree_batch_token_budget,
            tree_batch_node_budget=tree_batch_node_budget,
            tree_batch_autotune=tree_batch_autotune,
            tree_batch_structural_pad_limit=tree_batch_structural_pad_limit,
            tree_batch_auto_queue_min_docs=tree_batch_auto_queue_min_docs,
            tree_batch_auto_queue_min_fill_ratio=tree_batch_auto_queue_min_fill_ratio,
            tree_eval_workers_per_mig=tree_eval_workers_per_mig,
            exact_metric_evaluator=exact_metric_evaluator,
            exact_metric_selection_doc_limit=exact_metric_selection_doc_limit,
            exact_metric_selection_interval=exact_metric_selection_interval,
            exact_metric_phi_pair_calibration_max_nodes=exact_metric_phi_pair_calibration_max_nodes,
            exact_metric_final_doc_limit=exact_metric_final_doc_limit,
            tree_exact_eval_max_docs=tree_exact_eval_max_docs,
            runtime_config=runtime_config,
            leaf_sample_ordering_by_doc=leaf_sample_ordering_by_doc,
            internal_sample_ordering_by_doc=internal_sample_ordering_by_doc,
            memory_probe=memory_probe,
            progress_callback=progress_callback,
            progress_stage_name="single_stage",
            progress_epoch_offset=0,
            progress_epochs_total=int(n_epochs),
            progress_snapshot_interval=int(progress_snapshot_interval),
            progress_snapshot_dir=progress_snapshot_dir,
            grad_clip_norm=grad_clip_norm,
            depth_discount_gamma=float(depth_discount_gamma),
            seed=seed,
        )

    normalized_stage1_artifact_dir = str(tree_stage1_artifact_dir or "").strip()
    requested_stage1_epochs = int(tree_stage1_epochs)
    stage1_epochs = int(tree_stage1_epochs)
    stage2_epochs = int(tree_stage2_epochs)
    resume_from_saved_stage1 = bool(
        normalized_stage1_artifact_dir
        and bool(tree_stage1_resume_if_available)
        and Path(normalized_stage1_artifact_dir).expanduser().joinpath("metadata.json").exists()
        and Path(normalized_stage1_artifact_dir).expanduser().joinpath("model_state.pt").exists()
    )
    if resume_from_saved_stage1:
        stage1_epochs = 0
    explicit_stage2_only = bool(
        normalized_stage1_artifact_dir
        and int(stage1_epochs) <= 0
        and int(tree_stage2_epochs) > 0
    )
    if explicit_stage2_only:
        stage1_epochs = 0
        stage2_epochs = max(1, int(stage2_epochs))
    elif stage1_epochs <= 0 and stage2_epochs <= 0:
        stage1_epochs = max(1, int(n_epochs) // 3)
        stage2_epochs = max(0, int(n_epochs) - int(stage1_epochs))
    elif stage1_epochs <= 0:
        stage1_epochs = max(1, int(max(1, n_epochs) - max(0, stage2_epochs)))
    elif stage2_epochs <= 0:
        stage2_epochs = max(0, int(max(1, n_epochs) - max(1, stage1_epochs)))

    aligned_theorem_primary = bool(
        model.use_markov_summary_spec and model.uses_theory_aligned_root_surface()
    )
    effective_stage1_checkpoint_metric = str(
        tree_stage1_checkpoint_metric or "val_root_mae"
    ).strip() or "val_root_mae"
    if (
        float(tree_stage1_root_weight) <= 0.0
        and effective_stage1_checkpoint_metric == "val_root_mae"
    ):
        if aligned_theorem_primary:
            effective_stage1_checkpoint_metric = "val_theorem_bootstrap_direct"
        else:
            effective_stage1_checkpoint_metric = "val_exact_sketch_direct"
    stage1_c3_weight = (
        max(float(c3_weight), 1.0) if aligned_theorem_primary else 0.0
    )
    stage1_internal_supervision_kind = (
        "full_sketch" if aligned_theorem_primary else "none"
    )
    stage1_internal_label_rate = 1.0 if aligned_theorem_primary else 0.0
    stage1_artifact: Dict[str, Any] | None = None
    stage1_result: Dict[str, Any] | None = None
    if normalized_stage1_artifact_dir and stage1_epochs <= 0:
        try:
            artifact, stage1_state = load_theorem_feature_stage1_artifact(
                normalized_stage1_artifact_dir
            )
            restore_module_state(model, stage1_state)
            artifact_payload = artifact.as_dict()
            artifact_payload["artifact_source"] = "loaded"
            stage1_artifact = artifact_payload
            stage1_result = {
                "train": {"root_mae": float("nan"), "exact_match": float("nan")},
                "val": {"root_mae": float("nan"), "exact_match": float("nan")},
                "fit_diag": TrainFitDiagnostics(
                    train_loss_final=float("nan"),
                    train_loss_curve=tuple(),
                    epochs_completed=int(artifact.epochs_completed),
                    selection_metric_curve=tuple(),
                    selection_mode="loaded_stage1_artifact",
                    selection_split="artifact",
                    selection_metric_name=str(artifact.selection_metric_name),
                    selection_metric_value=float(artifact.selection_metric_value),
                    best_epoch=int(artifact.best_epoch),
                    train_exact_match_rate=float("nan"),
                    val_exact_match_rate=float("nan"),
                    test_exact_match_rate=float("nan"),
                ),
                "best_epoch": int(artifact.best_epoch),
                "best_val_mae": float(artifact.selection_metric_value),
                "selection_mode": "loaded_stage1_artifact",
                "selection_split": "artifact",
                "selection_metric_name": str(artifact.selection_metric_name),
                "selection_metric_curve": tuple(),
                "loss_curve": tuple(),
                "epochs_completed": int(artifact.epochs_completed),
                "training_component_loss_curves": {},
                "training_component_loss_finals": {},
                "best_model_state": dict(stage1_state),
                "best_exact_metrics": dict(artifact_payload.get("best_exact_metrics", {}) or {}),
                "best_exact_metrics_split": str(
                    artifact_payload.get("best_exact_metrics_split", "")
                ),
            }
        except RuntimeError:
            if resume_from_saved_stage1 and requested_stage1_epochs > 0:
                stage1_epochs = max(1, int(requested_stage1_epochs))
                stage1_result = None
            else:
                raise
    if stage1_result is None:
        stage1_result = _train_fno_tree_single_stage(
            model=model,
            train_docs=train_docs,
            val_docs=val_docs,
            device=device,
            n_epochs=int(stage1_epochs),
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            c1_weight=max(float(c1_weight), 1.0),
            c2_weight=max(float(c2_weight), 1.0),
            c3_weight=stage1_c3_weight,
            root_weight=float(tree_stage1_root_weight),
            leaf_query_rate=1.0,
            leaf_label_rate=1.0,
            audit_fraction=audit_fraction,
            doc_sequence_train_fraction=0.0,
            doc_sequence_objective=doc_sequence_objective,
            doc_sequence_class_index=doc_sequence_class_index,
            root_class_index=root_class_index,
            document_supervision_mode_by_doc=None,
            tree_document_loss_normalization_mode=tree_document_loss_normalization_mode,
            leaf_audit_indices_by_doc=None,
            c3_audit_indices_by_doc=None,
            internal_supervision_kind=stage1_internal_supervision_kind,
            internal_label_rate=stage1_internal_label_rate,
            max_internal_depth=max_internal_depth,
            leaf_exact_supervision=leaf_exact_supervision,
            leaf_supervision_kind="full_sketch",
            tree_local_weighting_mode=normalized_local_weighting_mode,
            tree_supervision_source="rate",
            phi_compose_weight=phi_compose_weight,
            phi_contrastive_weight=phi_contrastive_weight,
            checkpoint_metric=effective_stage1_checkpoint_metric,
            eval_mode=tree_stage1_eval_mode,
            screen_doc_limit=tree_stage1_screen_doc_limit,
            final_exact_doc_limit_override=tree_stage1_final_exact_doc_limit,
            tree_batch_pack_mode=tree_batch_pack_mode,
            tree_batch_token_budget=tree_batch_token_budget,
            tree_batch_node_budget=tree_batch_node_budget,
            tree_batch_autotune=tree_batch_autotune,
            tree_batch_structural_pad_limit=tree_batch_structural_pad_limit,
            tree_batch_auto_queue_min_docs=tree_batch_auto_queue_min_docs,
            tree_batch_auto_queue_min_fill_ratio=tree_batch_auto_queue_min_fill_ratio,
            tree_eval_workers_per_mig=tree_eval_workers_per_mig,
            exact_metric_evaluator=exact_metric_evaluator,
            exact_metric_selection_doc_limit=exact_metric_selection_doc_limit,
            exact_metric_selection_interval=exact_metric_selection_interval,
            exact_metric_phi_pair_calibration_max_nodes=exact_metric_phi_pair_calibration_max_nodes,
            exact_metric_final_doc_limit=exact_metric_final_doc_limit,
            tree_exact_eval_max_docs=tree_exact_eval_max_docs,
            runtime_config=runtime_config,
            leaf_sample_ordering_by_doc=leaf_sample_ordering_by_doc,
            internal_sample_ordering_by_doc=internal_sample_ordering_by_doc,
            memory_probe=memory_probe,
            progress_callback=progress_callback,
            progress_stage_name="stage1",
            progress_epoch_offset=0,
            progress_epochs_total=int(stage1_epochs + stage2_epochs),
            progress_snapshot_interval=int(progress_snapshot_interval),
            progress_snapshot_dir=progress_snapshot_dir,
            grad_clip_norm=grad_clip_norm,
            depth_discount_gamma=float(depth_discount_gamma),
            seed=seed,
        )
        if normalized_stage1_artifact_dir:
            summary_state_merger_in_features = 0
            if getattr(model, "summary_state_merger", None) is not None:
                try:
                    summary_state_merger_in_features = int(
                        getattr(model.summary_state_merger[0], "in_features", 0)
                    )
                except Exception:
                    summary_state_merger_in_features = 0
            carrier_state_merger_in_features = 0
            if getattr(model, "carrier_state_merger", None) is not None:
                try:
                    carrier_state_merger_in_features = int(
                        getattr(model.carrier_state_merger[0], "in_features", 0)
                    )
                except Exception:
                    carrier_state_merger_in_features = 0
            artifact = write_theorem_feature_stage1_artifact(
                normalized_stage1_artifact_dir,
                model_state=dict(stage1_result.get("best_model_state", {})),
                metadata={
                    "selection_metric_name": str(
                        stage1_result.get("selection_metric_name", "")
                    ),
                    "selection_metric_value": float(
                        stage1_result.get("best_val_mae", float("nan"))
                    ),
                    "best_epoch": int(stage1_result.get("best_epoch", 0)),
                    "epochs_completed": int(stage1_result.get("epochs_completed", 0)),
                    "training_schedule": "two_stage",
                    "stage1_root_weight": float(tree_stage1_root_weight),
                    "best_exact_metrics_split": str(
                        stage1_result.get("best_exact_metrics_split", "")
                    ),
                    "best_exact_metrics": dict(
                        stage1_result.get("best_exact_metrics", {}) or {}
                    ),
                    "artifact_source": "trained",
                    "theorem_surface_mode": str(
                        getattr(model, "theorem_surface_mode", "")
                    ),
                    "theorem_feature_dim": int(
                        getattr(model, "shared_theorem_feature_dim", 0)
                    ),
                    "theorem_score_dim": int(
                        getattr(model, "factorized_score_dim", 0)
                    ),
                    "theorem_fiber_dim": int(
                        getattr(model, "factorized_fiber_dim", 0)
                    ),
                    "theorem_aux_dim": int(
                        getattr(model, "factorized_aux_dim", 0)
                    ),
                    "runtime_merge_kind": str(
                        getattr(model, "runtime_merge_kind", "")
                    ),
                    "exact_projected_merge_is_runtime_merge": bool(
                        getattr(
                            model,
                            "exact_projected_merge_is_runtime_merge",
                            False,
                        )
                    ),
                    "uses_unified_g_learned_merge": bool(
                        getattr(model, "uses_unified_g_learned_merge", False)
                    ),
                    "n_regimes": int(getattr(model, "n_regimes", 0)),
                    "vocab_size": int(getattr(model, "pad_id", 0)),
                    "fixed_leaf_tokens": int(getattr(model, "leaf_tokens", 0)),
                    "state_dim": int(getattr(model, "requested_state_dim", getattr(model, "state_dim", 0))),
                    "model_state_dim": int(getattr(model, "state_dim", 0)),
                    "carrier_state_dim": int(
                        getattr(model, "carrier_state_dim", 0)
                    ),
                    "merge_hidden_dim": int(
                        getattr(model, "merge_hidden_dim", 0)
                    ),
                    "summary_spec_name": str(
                        getattr(model, "summary_spec_name", "")
                    ),
                    "slot_count": int(getattr(model, "slot_count", 0)),
                    "count_theorem_dim": int(
                        getattr(model, "count_theorem_dim", 0)
                    ),
                    "first_theorem_dim": int(
                        getattr(model, "first_theorem_dim", 0)
                    ),
                    "last_theorem_dim": int(
                        getattr(model, "last_theorem_dim", 0)
                    ),
                    "residual_dim": int(getattr(model, "residual_dim", 0)),
                    "summary_state_merger_in_features": int(
                        summary_state_merger_in_features
                    ),
                    "carrier_state_merger_in_features": int(
                        carrier_state_merger_in_features
                    ),
                    "task_head_mode": str(getattr(model, "task_head_mode", "")),
                    "summary_spec_root_mode": str(
                        getattr(model, "summary_spec_root_mode", "")
                    ),
                    "theorem_count_head_mode": str(
                        getattr(model, "theorem_count_head_mode", "")
                    ),
                    "c2_mode": str(getattr(model, "c2_mode", "")),
                    "tree_model_version": str(
                        getattr(model, "tree_model_version", "")
                    ),
                },
            )
            stage1_artifact = artifact.as_dict()

    stage2_result = _train_fno_tree_single_stage(
        model=model,
        train_docs=train_docs,
        val_docs=val_docs,
        device=device,
        n_epochs=int(stage2_epochs),
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        c1_weight=c1_weight,
        c2_weight=c2_weight,
        c3_weight=c3_weight,
        root_weight=root_weight,
        leaf_query_rate=leaf_query_rate,
        leaf_label_rate=leaf_label_rate,
        audit_fraction=audit_fraction,
        doc_sequence_train_fraction=doc_sequence_train_fraction,
        doc_sequence_objective=doc_sequence_objective,
        doc_sequence_class_index=doc_sequence_class_index,
        root_class_index=root_class_index,
        document_supervision_mode_by_doc=document_supervision_mode_by_doc,
        tree_document_loss_normalization_mode=tree_document_loss_normalization_mode,
        leaf_audit_indices_by_doc=leaf_audit_indices_by_doc,
        c3_audit_indices_by_doc=c3_audit_indices_by_doc,
        internal_supervision_kind=internal_supervision_kind,
        internal_label_rate=internal_label_rate,
        max_internal_depth=max_internal_depth,
        leaf_exact_supervision=leaf_exact_supervision,
        leaf_supervision_kind=normalized_leaf_supervision,
        tree_local_weighting_mode=normalized_local_weighting_mode,
        tree_supervision_source=normalized_supervision_source,
        phi_compose_weight=phi_compose_weight,
        phi_contrastive_weight=phi_contrastive_weight,
        checkpoint_metric=checkpoint_metric,
        eval_mode="per_epoch",
        screen_doc_limit=0,
        final_exact_doc_limit_override=0,
        tree_batch_pack_mode=tree_batch_pack_mode,
        tree_batch_token_budget=tree_batch_token_budget,
        tree_batch_node_budget=tree_batch_node_budget,
        tree_batch_autotune=tree_batch_autotune,
        tree_batch_structural_pad_limit=tree_batch_structural_pad_limit,
        tree_batch_auto_queue_min_docs=tree_batch_auto_queue_min_docs,
        tree_batch_auto_queue_min_fill_ratio=tree_batch_auto_queue_min_fill_ratio,
        tree_eval_workers_per_mig=tree_eval_workers_per_mig,
        exact_metric_evaluator=exact_metric_evaluator,
        exact_metric_selection_doc_limit=exact_metric_selection_doc_limit,
        exact_metric_selection_interval=exact_metric_selection_interval,
        exact_metric_phi_pair_calibration_max_nodes=exact_metric_phi_pair_calibration_max_nodes,
        exact_metric_final_doc_limit=exact_metric_final_doc_limit,
        tree_exact_eval_max_docs=tree_exact_eval_max_docs,
        runtime_config=runtime_config,
        leaf_sample_ordering_by_doc=leaf_sample_ordering_by_doc,
        internal_sample_ordering_by_doc=internal_sample_ordering_by_doc,
        memory_probe=memory_probe,
        progress_callback=progress_callback,
        progress_stage_name="stage2",
        progress_epoch_offset=int(stage1_epochs),
        progress_epochs_total=int(stage1_epochs + stage2_epochs),
        progress_snapshot_interval=int(progress_snapshot_interval),
        progress_snapshot_dir=progress_snapshot_dir,
        grad_clip_norm=grad_clip_norm,
        depth_discount_gamma=float(depth_discount_gamma),
        seed=seed + 17,
    )

    combined_selection_curve = tuple(
        float(x)
        for x in (
            list(stage1_result.get("selection_metric_curve", ()))
            + list(stage2_result.get("selection_metric_curve", ()))
        )
    )
    combined_loss_curve = tuple(
        float(x)
        for x in (
            list(stage1_result.get("loss_curve", ()))
            + list(stage2_result.get("loss_curve", ()))
        )
    )
    component_names = sorted(
        set(stage1_result.get("training_component_loss_curves", {}).keys())
        | set(stage2_result.get("training_component_loss_curves", {}).keys())
    )
    combined_component_curves: Dict[str, Tuple[float, ...]] = {}
    combined_component_finals: Dict[str, float] = {}
    for name in component_names:
        stage1_values = list(
            (stage1_result.get("training_component_loss_curves", {}) or {}).get(name, ())
        )
        stage2_values = list(
            (stage2_result.get("training_component_loss_curves", {}) or {}).get(name, ())
        )
        combined_values = tuple(float(x) for x in (stage1_values + stage2_values))
        combined_component_curves[name] = combined_values
        combined_component_finals[name] = (
            float(combined_values[-1]) if combined_values else float("nan")
        )

    fit_diag = TrainFitDiagnostics(
        train_loss_final=float(stage2_result.get("fit_diag").train_loss_final),
        train_loss_curve=combined_loss_curve,
        epochs_completed=int(stage1_epochs + stage2_epochs),
        selection_metric_curve=combined_selection_curve,
        selection_mode=str(stage2_result.get("selection_mode", "")),
        selection_split=str(stage2_result.get("selection_split", "")),
        selection_metric_name=str(stage2_result.get("selection_metric_name", "")),
        selection_metric_value=float(stage2_result.get("best_val_mae", float("nan"))),
        best_epoch=int(stage1_epochs + int(stage2_result.get("best_epoch", 0))),
        train_exact_match_rate=float(stage2_result.get("fit_diag").train_exact_match_rate),
        val_exact_match_rate=float(stage2_result.get("fit_diag").val_exact_match_rate),
        test_exact_match_rate=float("nan"),
        stage1_selection_metric_curve=tuple(
            float(x) for x in stage1_result.get("selection_metric_curve", ())
        ),
        stage2_selection_metric_curve=tuple(
            float(x) for x in stage2_result.get("selection_metric_curve", ())
        ),
        stage1_selection_metric_name=str(stage1_result.get("selection_metric_name", "")),
        stage2_selection_metric_name=str(stage2_result.get("selection_metric_name", "")),
        training_schedule="two_stage",
    )

    out = dict(stage2_result)
    out["fit_diag"] = fit_diag
    out["loss_curve"] = combined_loss_curve
    out["selection_metric_curve"] = combined_selection_curve
    out["epochs_completed"] = int(stage1_epochs + stage2_epochs)
    out["best_epoch"] = int(stage1_epochs + int(stage2_result.get("best_epoch", 0)))
    out["training_component_loss_curves"] = combined_component_curves
    out["training_component_loss_finals"] = combined_component_finals
    out["training_schedule"] = "two_stage"
    out["stage1_selection_metric_curve"] = tuple(
        float(x) for x in stage1_result.get("selection_metric_curve", ())
    )
    out["stage2_selection_metric_curve"] = tuple(
        float(x) for x in stage2_result.get("selection_metric_curve", ())
    )
    out["stage1_selection_metric_name"] = str(stage1_result.get("selection_metric_name", ""))
    out["stage2_selection_metric_name"] = str(stage2_result.get("selection_metric_name", ""))
    out["stage1_best_epoch"] = int(stage1_result.get("best_epoch", 0))
    out["stage2_best_epoch"] = int(stage2_result.get("best_epoch", 0))
    out["progress_snapshot_interval"] = int(
        stage2_result.get(
            "progress_snapshot_interval",
            stage1_result.get("progress_snapshot_interval", int(progress_snapshot_interval)),
        )
    )
    out["progress_snapshot_dir"] = str(
        stage2_result.get(
            "progress_snapshot_dir",
            stage1_result.get("progress_snapshot_dir", str(progress_snapshot_dir)),
        )
        or ""
    )
    out["progress_snapshot_paths"] = tuple(
        list(stage1_result.get("progress_snapshot_paths", ()) or ())
        + list(stage2_result.get("progress_snapshot_paths", ()) or ())
    )
    out["latest_progress_snapshot_path"] = str(
        stage2_result.get(
            "latest_progress_snapshot_path",
            stage1_result.get("latest_progress_snapshot_path", ""),
        )
        or ""
    )
    out["elapsed_s_train_loop"] = float(
        stage1_result.get("elapsed_s_train_loop", 0.0)
    ) + float(stage2_result.get("elapsed_s_train_loop", 0.0))
    out["elapsed_s_screen_eval"] = float(
        stage1_result.get("elapsed_s_screen_eval", 0.0)
    ) + float(stage2_result.get("elapsed_s_screen_eval", 0.0))
    out["elapsed_s_exact_metric_eval"] = float(
        stage1_result.get("elapsed_s_exact_metric_eval", 0.0)
    ) + float(stage2_result.get("elapsed_s_exact_metric_eval", 0.0))
    out["elapsed_s_split_eval"] = float(
        stage1_result.get("elapsed_s_split_eval", 0.0)
    ) + float(stage2_result.get("elapsed_s_split_eval", 0.0))
    out["elapsed_s_state_clone"] = float(
        stage1_result.get("elapsed_s_state_clone", 0.0)
    ) + float(stage2_result.get("elapsed_s_state_clone", 0.0))
    stage1_timing = dict(stage1_result.get("timing_breakdown", {}) or {})
    stage2_timing = dict(stage2_result.get("timing_breakdown", {}) or {})
    out["timing_breakdown"] = {
        "train_loop_s": float(out["elapsed_s_train_loop"]),
        "screen_eval_s": float(out["elapsed_s_screen_eval"]),
        "exact_metric_eval_s": float(out["elapsed_s_exact_metric_eval"]),
        "split_eval_s": float(out["elapsed_s_split_eval"]),
        "state_clone_s": float(out["elapsed_s_state_clone"]),
        "autotune_heuristic_s": float(stage1_timing.get("autotune_heuristic_s", 0.0))
        + float(stage2_timing.get("autotune_heuristic_s", 0.0)),
        "autotune_train_probe_s": float(
            stage1_timing.get("autotune_train_probe_s", 0.0)
        )
        + float(stage2_timing.get("autotune_train_probe_s", 0.0)),
        "autotune_eval_probe_s": float(
            stage1_timing.get("autotune_eval_probe_s", 0.0)
        )
        + float(stage2_timing.get("autotune_eval_probe_s", 0.0)),
        "autotune_cache_lookup_s": float(
            stage1_timing.get("autotune_cache_lookup_s", 0.0)
        )
        + float(stage2_timing.get("autotune_cache_lookup_s", 0.0)),
        "autotune_cache_write_s": float(
            stage1_timing.get("autotune_cache_write_s", 0.0)
        )
        + float(stage2_timing.get("autotune_cache_write_s", 0.0)),
        "autotune_total_s": float(stage1_timing.get("autotune_total_s", 0.0))
        + float(stage2_timing.get("autotune_total_s", 0.0)),
        "eval_total_s": float(
            out["elapsed_s_screen_eval"]
            + out["elapsed_s_exact_metric_eval"]
            + out["elapsed_s_split_eval"]
        ),
        "stage1_train_loop_s": float(stage1_result.get("elapsed_s_train_loop", 0.0)),
        "stage1_screen_eval_s": float(stage1_result.get("elapsed_s_screen_eval", 0.0)),
        "stage1_exact_metric_eval_s": float(
            stage1_result.get("elapsed_s_exact_metric_eval", 0.0)
        ),
        "stage1_split_eval_s": float(stage1_result.get("elapsed_s_split_eval", 0.0)),
        "stage2_train_loop_s": float(stage2_result.get("elapsed_s_train_loop", 0.0)),
        "stage2_screen_eval_s": float(stage2_result.get("elapsed_s_screen_eval", 0.0)),
        "stage2_exact_metric_eval_s": float(
            stage2_result.get("elapsed_s_exact_metric_eval", 0.0)
        ),
        "stage2_split_eval_s": float(stage2_result.get("elapsed_s_split_eval", 0.0)),
    }
    out["stage1_result_summary"] = {
        "selection_metric_name": str(stage1_result.get("selection_metric_name", "")),
        "best_epoch": int(stage1_result.get("best_epoch", 0)),
        "best_metric_value": float(stage1_result.get("best_val_mae", float("nan"))),
    }
    out["stage2_result_summary"] = {
        "selection_metric_name": str(stage2_result.get("selection_metric_name", "")),
        "best_epoch": int(stage2_result.get("best_epoch", 0)),
        "best_metric_value": float(stage2_result.get("best_val_mae", float("nan"))),
    }
    stage1_batching = dict(stage1_result.get("batching_metrics", {}) or {})
    stage2_batching = dict(stage2_result.get("batching_metrics", {}) or {})
    out["batching_metrics"] = {
        "mean_docs_per_batch": float(stage2_batching.get("mean_docs_per_batch", float("nan"))),
        "mean_leaf_tokens_per_batch": float(
            stage2_batching.get("mean_leaf_tokens_per_batch", float("nan"))
        ),
        "mean_nodes_per_batch": float(stage2_batching.get("mean_nodes_per_batch", float("nan"))),
        "padding_waste_ratio": float(stage2_batching.get("padding_waste_ratio", float("nan"))),
        "bucket_utilization_rate": float(
            stage2_batching.get("bucket_utilization_rate", float("nan"))
        ),
        "gpu_reserved_mem_peak_gb": float(
            max(
                float(stage1_batching.get("gpu_reserved_mem_peak_gb", float("nan"))),
                float(stage2_batching.get("gpu_reserved_mem_peak_gb", float("nan"))),
            )
        ),
        "gpu_allocated_mem_peak_gb": float(
            max(
                float(stage1_batching.get("gpu_allocated_mem_peak_gb", float("nan"))),
                float(stage2_batching.get("gpu_allocated_mem_peak_gb", float("nan"))),
            )
        ),
        "train_forward_time_s": float(
            stage1_batching.get("train_forward_time_s", 0.0)
        ) + float(stage2_batching.get("train_forward_time_s", 0.0)),
        "train_backward_time_s": float(
            stage1_batching.get("train_backward_time_s", 0.0)
        ) + float(stage2_batching.get("train_backward_time_s", 0.0)),
        "eval_time_s": float(stage1_batching.get("eval_time_s", 0.0))
        + float(stage2_batching.get("eval_time_s", 0.0)),
        "idle_wait_time_s": float(stage1_batching.get("idle_wait_time_s", 0.0))
        + float(stage2_batching.get("idle_wait_time_s", 0.0)),
        "autotune_heuristic_time_s": float(
            stage1_batching.get("autotune_heuristic_time_s", 0.0)
        )
        + float(stage2_batching.get("autotune_heuristic_time_s", 0.0)),
        "autotune_train_probe_time_s": float(
            stage1_batching.get("autotune_train_probe_time_s", 0.0)
        )
        + float(stage2_batching.get("autotune_train_probe_time_s", 0.0)),
        "autotune_eval_probe_time_s": float(
            stage1_batching.get("autotune_eval_probe_time_s", 0.0)
        )
        + float(stage2_batching.get("autotune_eval_probe_time_s", 0.0)),
        "autotune_cache_lookup_time_s": float(
            stage1_batching.get("autotune_cache_lookup_time_s", 0.0)
        )
        + float(stage2_batching.get("autotune_cache_lookup_time_s", 0.0)),
        "autotune_cache_write_time_s": float(
            stage1_batching.get("autotune_cache_write_time_s", 0.0)
        )
        + float(stage2_batching.get("autotune_cache_write_time_s", 0.0)),
        "autotune_cache_hits": int(stage1_batching.get("autotune_cache_hits", 0) or 0)
        + int(stage2_batching.get("autotune_cache_hits", 0) or 0),
        "autotune_cache_misses": int(
            stage1_batching.get("autotune_cache_misses", 0) or 0
        )
        + int(stage2_batching.get("autotune_cache_misses", 0) or 0),
        "autotune_cache_writes": int(
            stage1_batching.get("autotune_cache_writes", 0) or 0
        )
        + int(stage2_batching.get("autotune_cache_writes", 0) or 0),
        "autotune_probe_runs": int(stage1_batching.get("autotune_probe_runs", 0) or 0)
        + int(stage2_batching.get("autotune_probe_runs", 0) or 0),
        "autotune_probe_candidate_evals": int(
            stage1_batching.get("autotune_probe_candidate_evals", 0) or 0
        )
        + int(stage2_batching.get("autotune_probe_candidate_evals", 0) or 0),
        "runtime_data_mode": str(
            stage2_batching.get(
                "runtime_data_mode",
                stage1_batching.get("runtime_data_mode", ""),
            )
        ),
        "runtime_bucket_mode": str(
            stage2_batching.get(
                "runtime_bucket_mode",
                stage1_batching.get("runtime_bucket_mode", ""),
            )
        ),
        "runtime_workers_per_mig": int(
            stage2_batching.get(
                "runtime_workers_per_mig",
                stage1_batching.get("runtime_workers_per_mig", 1),
            )
            or 1
        ),
        "resident_store_build_time_s": float(
            stage1_batching.get("resident_store_build_time_s", 0.0)
        )
        + float(stage2_batching.get("resident_store_build_time_s", 0.0)),
        "steady_state_h2d_bytes": int(stage1_batching.get("steady_state_h2d_bytes", 0) or 0)
        + int(stage2_batching.get("steady_state_h2d_bytes", 0) or 0),
        "steady_state_h2d_time_s": float(
            stage1_batching.get("steady_state_h2d_time_s", 0.0)
        )
        + float(stage2_batching.get("steady_state_h2d_time_s", 0.0)),
        "resident_store_hits": int(stage1_batching.get("resident_store_hits", 0) or 0)
        + int(stage2_batching.get("resident_store_hits", 0) or 0),
        "resident_store_misses": int(stage1_batching.get("resident_store_misses", 0) or 0)
        + int(stage2_batching.get("resident_store_misses", 0) or 0),
        "stage1_batching_metrics": stage1_batching,
        "stage2_batching_metrics": stage2_batching,
    }
    out["runtime_efficiency"] = dict(stage2_result.get("runtime_efficiency", {}) or {})
    out["tree_local_weighting_mode"] = str(normalized_local_weighting_mode)
    out["tree_supervision_source"] = str(normalized_supervision_source)
    out["local_estimand_mode"] = str(normalized_local_weighting_mode)
    out["c2_pair_weighting_mode"] = str(
        stage2_result.get(
            "c2_pair_weighting_mode",
            stage1_result.get("c2_pair_weighting_mode", "legacy_unweighted"),
        )
    )
    out["c2_same_pair_count"] = float(
        stage2_result.get(
            "c2_same_pair_count",
            stage1_result.get("c2_same_pair_count", 0.0),
        )
    )
    out["c2_different_pair_count"] = float(
        stage2_result.get(
            "c2_different_pair_count",
            stage1_result.get("c2_different_pair_count", 0.0),
        )
    )
    out["c2_pair_weight_ess"] = float(
        stage2_result.get(
            "c2_pair_weight_ess",
            stage1_result.get("c2_pair_weight_ess", 0.0),
        )
    )
    out["c2_pair_weight_max"] = float(
        stage2_result.get(
            "c2_pair_weight_max",
            stage1_result.get("c2_pair_weight_max", 0.0),
        )
    )
    out["local_loss_kind"] = str(stage2_result.get("local_loss_kind", ""))
    out["local_sampling_design_name"] = str(
        stage2_result.get("local_sampling_design_name", "")
    )
    out["leaf_population_size"] = float(
        stage2_result.get("leaf_population_size", float("nan"))
    )
    out["leaf_sample_size"] = float(
        stage2_result.get("leaf_sample_size", float("nan"))
    )
    out["leaf_effective_propensity"] = float(
        stage2_result.get("leaf_effective_propensity", float("nan"))
    )
    out["merge_population_size"] = float(
        stage2_result.get("merge_population_size", float("nan"))
    )
    out["merge_sample_size"] = float(
        stage2_result.get("merge_sample_size", float("nan"))
    )
    out["merge_effective_propensity"] = float(
        stage2_result.get("merge_effective_propensity", float("nan"))
    )
    out["local_objective_audit"] = dict(
        stage2_result.get("local_objective_audit", {})
        or stage1_result.get("local_objective_audit", {})
        or {}
    )
    out["autotuned_batch_budgets"] = dict(
        stage2_result.get("autotuned_batch_budgets", {})
        or stage1_result.get("autotuned_batch_budgets", {})
        or {}
    )
    stage1_probe_profile = dict(stage1_result.get("autotune_probe_profile", {}) or {})
    stage2_probe_profile = dict(stage2_result.get("autotune_probe_profile", {}) or {})
    merged_probe_profile = _merge_autotune_probe_profile_dicts(
        stage1_probe_profile,
        stage2_probe_profile,
    )
    out["autotune_probe_profile"] = merged_probe_profile
    out["autotuned_batch_budgets"].update(
        {
            "probe_cache_version": int(AUTOTUNE_PROBE_CACHE_VERSION),
            "probe_cache_hits": int(merged_probe_profile.get("cache_hits", 0) or 0),
            "probe_cache_misses": int(
                merged_probe_profile.get("cache_misses", 0) or 0
            ),
            "probe_cache_writes": int(
                merged_probe_profile.get("cache_writes", 0) or 0
            ),
            "probe_run_count": int(
                merged_probe_profile.get("probe_run_count", 0) or 0
            ),
            "probe_candidate_count": int(
                merged_probe_profile.get("probe_candidate_count", 0) or 0
            ),
        }
    )
    out["stage1_best_model_state"] = stage1_result.get("best_model_state")
    out["stage1_artifact"] = dict(stage1_artifact or {})
    return out


def _sample_realized_node_indices(
    rng: "random.Random",
    *,
    n_nodes: int,
    sample_rate: float,
) -> Tuple[Optional[set], float]:
    """Sample realized node indices with a logged first-order propensity."""

    n = int(max(0, n_nodes))
    if n <= 0:
        return set(), 0.0

    rate = float(sample_rate)
    if rate <= 0.0:
        return set(), 0.0
    if rate >= 1.0:
        return None, 1.0

    sample_count = max(1, int(round(rate * float(n))))
    sample_count = min(n, int(sample_count))
    if sample_count >= n:
        return None, 1.0

    indices = set(rng.sample(range(n), k=int(sample_count)))
    return indices, float(sample_count) / float(n)


def _doc_sequence_inputs_for_doc(
    doc: _FNOCountDoc,
    *,
    pad_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    full_tokens = list(_flatten_fno_doc_tokens(doc))
    if not full_tokens:
        raise ValueError("doc_sequence supervision requires at least one token")
    tok_t = torch.tensor(full_tokens, dtype=torch.long, device=device).unsqueeze(0)
    mask_t = torch.ones((1, len(full_tokens)), dtype=torch.float32, device=device)
    return tok_t, mask_t


def _doc_sequence_loss_for_doc(
    model: FNOCountSketch,
    doc: _FNOCountDoc,
    *,
    device: torch.device,
    class_index: Mapping[int, int],
    objective_name: str,
) -> torch.Tensor:
    tokens_t, mask_t = _doc_sequence_inputs_for_doc(doc, pad_id=model.pad_id, device=device)
    logits = model.predict_doc_sequence_logits(tokens_t, token_mask=mask_t)
    target_class = torch.tensor(
        [int(class_index[int(round(float(doc.root_count)))])],
        dtype=torch.long,
        device=device,
    )
    loss = F.cross_entropy(logits, target_class)
    if str(objective_name) == "count_ce_plus_scalar_mse":
        pred_norm = model.predict_doc_sequence_expected_normalized_from_logits(logits)
        target_norm = torch.tensor(
            [float(doc.root_count) / float(model.target_scale)],
            dtype=pred_norm.dtype,
            device=device,
        )
        loss = loss + 0.25 * F.mse_loss(pred_norm, target_norm, reduction="mean")
    return loss


def _validate_fno_objective_shares(
    *,
    root_objective_share: float,
    local_law_objective_share: float,
) -> Tuple[float, float]:
    """Validate resolved convex root/local-law shares for unified FNO training."""

    root_share = float(root_objective_share)
    local_share = float(local_law_objective_share)
    if not math.isfinite(root_share) or not math.isfinite(local_share):
        raise ValueError("FNO objective shares must be finite")
    if root_share < 0.0 or local_share < 0.0:
        raise ValueError("FNO objective shares must be non-negative")
    if not math.isclose(root_share + local_share, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            "FNO objective shares must form a convex root/local-law objective; "
            f"got root={root_share:g}, local_law={local_share:g}"
        )
    return root_share, local_share


def _fno_single_lambda_objective_loss(
    *,
    root_loss: torch.Tensor,
    local_law_loss: torch.Tensor,
    root_objective_share: float,
    local_law_objective_share: float,
) -> torch.Tensor:
    """Return ``(1-lambda) * L_root + lambda * L_corrected`` for FNO."""

    root_share, local_share = _validate_fno_objective_shares(
        root_objective_share=float(root_objective_share),
        local_law_objective_share=float(local_law_objective_share),
    )
    return float(root_share) * root_loss + float(local_share) * local_law_loss


@torch.no_grad()
def _eval_fno_doc_sequence_view(
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
    tau: float,
) -> SketchMetrics:
    if len(docs) == 0:
        return _zero_sketch_metrics(n_docs=0)

    preds: List[float] = []
    truths: List[float] = []
    model.eval()
    for doc in docs:
        tokens_t, mask_t = _doc_sequence_inputs_for_doc(doc, pad_id=model.pad_id, device=device)
        logits = model.predict_doc_sequence_logits(tokens_t, token_mask=mask_t)
        pred = float(model.predict_doc_sequence_counts_from_logits(logits).squeeze(0).detach().cpu())
        preds.append(pred)
        truths.append(float(doc.root_count))
    return _eval_root_predictions(preds, truths, tau=float(tau))


def train_fno_tree_local_law(
    *,
    model: FNOCountSketch,
    train_docs: Sequence[_FNOCountDoc],
    val_docs: Sequence[_FNOCountDoc],
    device: torch.device,
    n_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 1e-5,
    leaf_sample_rate: float = 1.0,
    internal_sample_rate: float = 1.0,
    root_objective_share: float,
    local_law_objective_share: float,
    local_law_objective_mode: str = LOCAL_LAW_OBJECTIVE_CORRECTED,
    depth_discount_gamma: float = 1.0,
    use_residual_decomposition: bool = True,
    doc_sequence_train_fraction: float = 0.0,
    doc_sequence_objective: str = "count_ce_only",
    doc_sequence_class_index: Mapping[int, int] | None = None,
    grad_clip_norm: float = 1.0,
    seed: int = 42,
) -> Dict[str, object]:
    """Train FNO tree-merge operator with the single-lambda local-law objective.

    The shared readout ``g`` is applied at every realized node. The objective is
    ``root_objective_share * L_root + local_law_objective_share * L_local_law``.
    In paper-facing runs ``L_local_law`` is the bundled corrected full-tree loss:
    a dense proxy population plus sparse oracle residuals. ``sampled_ipw`` is
    retained as an ablation mode and uses observed oracle rows only.

    When *use_residual_decomposition* is True, merge-node losses use
    residuals (g(merge) - g(left) - g(right)) vs boundary correction
    targets, making it easier for the merger to learn.
    """
    import random as _random

    normalized_local_law_objective_mode = normalize_local_law_objective_mode(
        str(local_law_objective_mode)
    )
    root_share, local_share = _validate_fno_objective_shares(
        root_objective_share=float(root_objective_share),
        local_law_objective_share=float(local_law_objective_share),
    )
    rng = _random.Random(int(seed))
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(n_epochs), eta_min=float(lr) * 0.01,
    )
    doc_sequence_fraction = min(1.0, max(0.0, float(doc_sequence_train_fraction)))
    doc_sequence_class_index = dict(doc_sequence_class_index or {})
    doc_sequence_train_indices = _sample_doc_index_subset(
        n_docs=len(train_docs),
        fraction=float(doc_sequence_fraction),
        seed=int(seed) + 71_001,
    )
    if doc_sequence_fraction > 0.0 and not doc_sequence_class_index:
        raise ValueError("doc_sequence_class_index is required when doc_sequence_train_fraction > 0")

    idxs = list(range(len(train_docs)))
    loss_curve: List[float] = []
    best_state = clone_module_state(model)
    best_val_mae = float("inf")
    best_epoch = 0
    best_tree_val_mae = float("inf")
    best_doc_sequence_val_mae = float("inf")

    for epoch in range(int(n_epochs)):
        rng.shuffle(idxs)
        model.train()
        batch_losses: List[float] = []

        for b0 in range(0, len(idxs), int(max(1, batch_size))):
            batch_idx = idxs[b0 : b0 + int(max(1, batch_size))]
            opt.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=device, dtype=torch.float32)

            for i in batch_idx:
                doc = train_docs[i]
                if i in doc_sequence_train_indices:
                    batch_loss = batch_loss + _doc_sequence_loss_for_doc(
                        model,
                        doc,
                        device=device,
                        class_index=doc_sequence_class_index,
                        objective_name=str(doc_sequence_objective),
                    )
                    continue
                n_leaf = len(doc.leaf_token_ids)
                n_internal = int(len(doc.merge_counts_balanced))

                sampled_leaf_indices, leaf_propensity = _sample_realized_node_indices(
                    rng,
                    n_nodes=int(n_leaf),
                    sample_rate=float(leaf_sample_rate),
                )
                sampled_internal_indices, internal_propensity = _sample_realized_node_indices(
                    rng,
                    n_nodes=int(n_internal),
                    sample_rate=float(internal_sample_rate),
                )

                out = model.forward_doc_unified(
                    doc.leaf_token_ids, doc.leaf_counts,
                    doc.merge_counts_balanced, doc.root_count,
                    doc_id=f"train_doc_{i}",
                    schedule="balanced", device=device,
                    sampled_leaf_indices=sampled_leaf_indices,
                    sampled_internal_indices=sampled_internal_indices,
                    leaf_propensity=leaf_propensity,
                    internal_propensity=internal_propensity,
                    proxy_leaf_counts=(
                        doc.proxy_leaf_counts if doc.proxy_leaf_counts else None
                    ),
                    proxy_merge_counts_balanced=(
                        doc.proxy_merge_counts_balanced
                        if doc.proxy_merge_counts_balanced
                        else None
                    ),
                    use_residual_decomposition=use_residual_decomposition,
                    collect_full_trace=False,
                )

                node_loss = local_law_objective_target_mse(
                    predictions=out["all_node_preds"],
                    proxy_targets=out["all_node_proxy_targets"],
                    oracle_targets=out["all_node_oracle_targets"],
                    observed=out["all_node_observed"],
                    propensity=out["all_node_propensities"],
                    depths=out["all_node_depths"],
                    gamma_depth=float(depth_discount_gamma),
                    objective_mode=str(normalized_local_law_objective_mode),
                )
                document_loss = F.mse_loss(
                    out["document_pred_norm"],
                    out["document_target_norm"],
                )
                doc_loss = _fno_single_lambda_objective_loss(
                    root_loss=document_loss,
                    local_law_loss=node_loss,
                    root_objective_share=float(root_share),
                    local_law_objective_share=float(local_share),
                )
                batch_loss = batch_loss + doc_loss

            batch_loss = batch_loss / float(len(batch_idx))
            if bool(getattr(batch_loss, "requires_grad", False)):
                batch_loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                opt.step()
            batch_losses.append(float(batch_loss.detach().cpu()))

        scheduler.step()
        loss_curve.append(float(np.mean(np.asarray(batch_losses, dtype=np.float64))))

        # Validation (root MAE for model selection).
        if val_docs:
            tree_val_metrics = _eval_fno_model(
                model,
                val_docs,
                device=device,
                tau=0.0,
            )
            doc_sequence_val_metrics = (
                _eval_fno_doc_sequence_view(
                    model,
                    val_docs,
                    device=device,
                    tau=0.0,
                )
                if doc_sequence_fraction > 0.0
                else tree_val_metrics
            )
            selection_value = _curriculum_selection_value(
                tree_root_mae=float(tree_val_metrics.root_mae),
                doc_sequence_root_mae=float(doc_sequence_val_metrics.root_mae),
                doc_sequence_fraction=float(doc_sequence_fraction),
            )
            if selection_value < best_val_mae - 1e-9:
                best_val_mae = selection_value
                best_epoch = epoch
                best_tree_val_mae = float(tree_val_metrics.root_mae)
                best_doc_sequence_val_mae = float(doc_sequence_val_metrics.root_mae)
                best_state = clone_module_state(model)

    if val_docs:
        restore_module_state(model, best_state)
    else:
        best_epoch = max(0, int(n_epochs) - 1)

    def _eval_split(docs):
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for doc in docs:
                out = model.forward_doc_unified(
                    doc.leaf_token_ids, doc.leaf_counts,
                    doc.merge_counts_balanced, doc.root_count,
                    doc_id="eval_doc",
                    schedule="balanced", device=device,
                    proxy_leaf_counts=(
                        doc.proxy_leaf_counts if doc.proxy_leaf_counts else None
                    ),
                    proxy_merge_counts_balanced=(
                        doc.proxy_merge_counts_balanced
                        if doc.proxy_merge_counts_balanced
                        else None
                    ),
                    use_residual_decomposition=use_residual_decomposition,
                    collect_full_trace=False,
                )
                preds.append(float(out["root_pred_count"].detach().cpu()))
                truths.append(float(doc.root_count))
        p, t = np.array(preds), np.array(truths)
        return {
            "root_mae": float(np.mean(np.abs(p - t))),
            "exact_match": float(np.mean((np.rint(p) == np.rint(t)).astype(np.float64))),
        }

    return {
        "train": _eval_split(train_docs),
        "val": _eval_split(val_docs) if val_docs else {"root_mae": float("nan"), "exact_match": float("nan")},
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "best_tree_val_mae": best_tree_val_mae,
        "best_doc_sequence_val_mae": best_doc_sequence_val_mae,
        "selection_metric_name": (
            "val_doc_sequence_root_mae"
            if doc_sequence_fraction >= 1.0
            else (
                "val_tree_doc_sequence_curriculum_mae"
                if doc_sequence_fraction > 0.0
                else "val_root_mae"
            )
        ),
        "loss_curve": loss_curve,
        "doc_sequence_train_docs_used": int(len(doc_sequence_train_indices)),
        "root_objective_share": float(root_share),
        "local_law_objective_share": float(local_share),
        "local_law_objective_mode": str(normalized_local_law_objective_mode),
        "depth_discount_gamma": float(depth_discount_gamma),
        "n_params": sum(p.numel() for p in model.parameters()),
    }


# ---------------------------------------------------------------------------
# Evaluation helpers for FNOCountSketch (mirrors _eval_learned_model /
# _eval_objective_terms from markov_changepoint_ops_count.py, adapted for
# token-based leaf encoding)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _eval_fno_model_legacy(
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
    tau: float,
) -> SketchMetrics:
    """Evaluate FNOCountSketch on a set of FNO docs, returning SketchMetrics."""
    if len(docs) == 0:
        return _zero_sketch_metrics(n_docs=0)

    model.eval()
    root_abs: List[float] = []
    spreads: List[float] = []
    leaf_abs: List[float] = []
    merge_abs: List[float] = []
    c2_r1_abs: List[float] = []
    c2_r2_abs: List[float] = []
    c2_r4_abs: List[float] = []
    c2_state_replay_mse: List[float] = []
    root_drift_r1: List[float] = []
    root_drift_r2: List[float] = []
    root_drift_r4: List[float] = []

    for doc in docs:
        # Encode leaves from raw token ids.
        state_batch = model.encode_leaf_tokens_batch(doc.leaf_token_ids, device=device)
        states = [state_batch[idx] for idx in range(int(state_batch.shape[0]))]

        # Leaf C1.
        for st, truth in zip(states, doc.leaf_counts):
            pred = float(model.predict_count_from_state(st).detach().cpu())
            leaf_abs.append(abs(pred - float(truth)))

        # Merge C3 (balanced schedule, all internal nodes).
        _root_state, merge_states = model._merge_states(
            states, schedule="balanced", collect_merge_states=True,
        )
        for pred_st, truth in zip(merge_states, doc.merge_counts_balanced):
            pred = float(model.predict_count_from_state(pred_st).detach().cpu())
            merge_abs.append(abs(pred - float(truth)))

        # C2 count drift after replaying the decoded summary surface.
        c2_states = list(states)
        c2_states.extend(list(merge_states))
        if not c2_states:
            c2_states = [_root_state]
        for st in c2_states:
            base_summary = model.decode_summary(st)
            base_value = float(_predict_count_from_summary(model, base_summary).detach().cpu())
            _base_norm, _replay_norm, base_state, replay_state = _fno_summary_replay_tensors(
                model, st
            )
            c2_state_replay_mse.append(
                float(F.mse_loss(replay_state, base_state).detach().cpu())
            )
            replay = _resummary_summary_sequence(model, st, depths=(1, 2, 4))
            c2_r1_abs.append(
                abs(float(_predict_count_from_summary(model, replay[1]).detach().cpu()) - base_value)
            )
            c2_r2_abs.append(
                abs(float(_predict_count_from_summary(model, replay[2]).detach().cpu()) - base_value)
            )
            c2_r4_abs.append(
                abs(float(_predict_count_from_summary(model, replay[4]).detach().cpu()) - base_value)
            )

        # Root distortion + schedule spread.
        roots: Dict[str, float] = {}
        for sched in VALID_SCHEDULES:
            root_state, _ = model._merge_states(states, schedule=sched, collect_merge_states=False)
            roots[str(sched)] = float(model.predict_count_from_state(root_state).detach().cpu())
        pred_root = float(
            model.predict_canonical_count_from_state(_root_state).detach().cpu()
        )
        root_abs.append(abs(pred_root - float(doc.root_count)))
        spreads.append(max(roots.values()) - min(roots.values()))
        root_replay = _resummary_summary_sequence(model, _root_state, depths=(1, 2, 4))
        root_base = float(
            _predict_count_from_summary(model, model.decode_summary(_root_state)).detach().cpu()
        )
        root_drift_r1.append(
            abs(float(_predict_count_from_summary(model, root_replay[1]).detach().cpu()) - root_base)
        )
        root_drift_r2.append(
            abs(float(_predict_count_from_summary(model, root_replay[2]).detach().cpu()) - root_base)
        )
        root_drift_r4.append(
            abs(float(_predict_count_from_summary(model, root_replay[4]).detach().cpu()) - root_base)
        )

    tau = float(tau)
    leaf_abs_arr = np.asarray(leaf_abs, dtype=np.float64)
    merge_abs_arr = np.asarray(merge_abs, dtype=np.float64)
    root_abs_arr = np.asarray(root_abs, dtype=np.float64)
    spreads_arr = np.asarray(spreads, dtype=np.float64)
    c2_r1_arr = np.asarray(c2_r1_abs, dtype=np.float64)
    c2_r2_arr = np.asarray(c2_r2_abs, dtype=np.float64)
    c2_r4_arr = np.asarray(c2_r4_abs, dtype=np.float64)
    c2_state_replay_arr = np.asarray(c2_state_replay_mse, dtype=np.float64)
    root_drift_r1_arr = np.asarray(root_drift_r1, dtype=np.float64)
    root_drift_r2_arr = np.asarray(root_drift_r2, dtype=np.float64)
    root_drift_r4_arr = np.asarray(root_drift_r4, dtype=np.float64)

    return SketchMetrics(
        root_mae=float(np.mean(root_abs_arr)),
        root_median_abs_error=float(np.median(root_abs_arr)),
        root_p95_abs_error=float(np.percentile(root_abs_arr, 95.0)),
        schedule_spread_mean=float(np.mean(spreads_arr)),
        schedule_spread_p95=float(np.percentile(spreads_arr, 95.0)),
        leaf_mae=float(np.mean(leaf_abs_arr)) if leaf_abs_arr.size else 0.0,
        leaf_violation_rate=(
            float(np.mean((leaf_abs_arr > tau).astype(np.float64))) if leaf_abs_arr.size else 0.0
        ),
        c2_idempotence_mae=float(np.mean(c2_r1_arr)) if c2_r1_arr.size else 0.0,
        c2_r1_mae=float(np.mean(c2_r1_arr)) if c2_r1_arr.size else 0.0,
        c2_r2_mae=float(np.mean(c2_r2_arr)) if c2_r2_arr.size else 0.0,
        c2_r4_mae=float(np.mean(c2_r4_arr)) if c2_r4_arr.size else 0.0,
        resummary_root_drift_r1=(
            float(np.mean(root_drift_r1_arr)) if root_drift_r1_arr.size else 0.0
        ),
        resummary_root_drift_r2=(
            float(np.mean(root_drift_r2_arr)) if root_drift_r2_arr.size else 0.0
        ),
        resummary_root_drift_r4=(
            float(np.mean(root_drift_r4_arr)) if root_drift_r4_arr.size else 0.0
        ),
        merge_mae=float(np.mean(merge_abs_arr)) if merge_abs_arr.size else 0.0,
        merge_violation_rate=(
            float(np.mean((merge_abs_arr > tau).astype(np.float64))) if merge_abs_arr.size else 0.0
        ),
        n_docs=int(len(docs)),
        c2_state_replay_mse=(
            float(np.mean(c2_state_replay_arr)) if c2_state_replay_arr.size else 0.0
        ),
    )


@torch.inference_mode()
def _eval_fno_model(
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
    tau: float,
    condition_ids: Sequence[str] | None = None,
) -> SketchMetrics:
    """Evaluate FNOCountSketch on a set of FNO docs, returning SketchMetrics."""
    if len(docs) == 0:
        return _zero_sketch_metrics(n_docs=0)

    model.eval()
    root_abs: List[float] = []
    spreads: List[float] = []
    leaf_abs_total = 0.0
    leaf_abs_count = 0
    leaf_violation_count = 0
    merge_abs_total = 0.0
    merge_abs_count = 0
    merge_violation_count = 0
    c2_r1_total = 0.0
    c2_r2_total = 0.0
    c2_r4_total = 0.0
    c2_count = 0
    c2_state_replay_total = 0.0
    root_drift_r1_total = 0.0
    root_drift_r2_total = 0.0
    root_drift_r4_total = 0.0
    root_drift_count = 0

    with _autocast_context(device):
        for doc in docs:
            states = _encode_leaf_state_batch(
                model,
                doc.leaf_token_ids,
                device=device,
            )
            if not states:
                continue

            state_batch = torch.stack(states, dim=0)
            leaf_truth = torch.as_tensor(
                list(doc.leaf_counts),
                dtype=state_batch.dtype,
                device=device,
            )
            leaf_pred = model.predict_count_from_state(state_batch)
            leaf_abs_tensor = torch.abs(leaf_pred - leaf_truth)
            leaf_abs_total += float(leaf_abs_tensor.sum().cpu())
            leaf_abs_count += int(leaf_abs_tensor.numel())
            leaf_violation_count += int((leaf_abs_tensor > float(tau)).sum().cpu())

            root_state, merge_states = model._merge_states(
                states,
                schedule="balanced",
                collect_merge_states=True,
            )
            if merge_states:
                merge_batch = torch.stack(list(merge_states), dim=0)
                merge_truth = torch.as_tensor(
                    list(doc.merge_counts_balanced),
                    dtype=merge_batch.dtype,
                    device=device,
                )
                merge_pred = model.predict_count_from_state(merge_batch)
                merge_abs_tensor = torch.abs(merge_pred - merge_truth)
                merge_abs_total += float(merge_abs_tensor.sum().cpu())
                merge_abs_count += int(merge_abs_tensor.numel())
                merge_violation_count += int((merge_abs_tensor > float(tau)).sum().cpu())

            c2_states = list(states)
            c2_states.extend(list(merge_states))
            if not c2_states:
                c2_states = [root_state]
            for st in c2_states:
                base_summary = model.decode_summary(st)
                base_value = float(
                    _predict_count_from_summary(model, base_summary).detach().cpu()
                )
                _base_norm, _replay_norm, base_state, replay_state = _fno_summary_replay_tensors(
                    model, st
                )
                c2_state_replay_total += float(
                    F.mse_loss(replay_state, base_state).detach().cpu()
                )
                replay = _resummary_summary_sequence(model, st, depths=(1, 2, 4))
                c2_r1_total += abs(
                    float(_predict_count_from_summary(model, replay[1]).detach().cpu())
                    - base_value
                )
                c2_r2_total += abs(
                    float(_predict_count_from_summary(model, replay[2]).detach().cpu())
                    - base_value
                )
                c2_r4_total += abs(
                    float(_predict_count_from_summary(model, replay[4]).detach().cpu())
                    - base_value
                )
                c2_count += 1

            roots: Dict[str, float] = {}
            for sched in VALID_SCHEDULES:
                root_state_sched, _ = model._merge_states(
                    states,
                    schedule=sched,
                    collect_merge_states=False,
                )
                roots[str(sched)] = float(
                    model.predict_count_from_state(root_state_sched).detach().cpu()
                )
            pred_root = float(
                model.predict_canonical_count_from_state(root_state).detach().cpu()
            )
            root_abs.append(abs(pred_root - float(doc.root_count)))
            spreads.append(max(roots.values()) - min(roots.values()))
            root_replay = _resummary_summary_sequence(model, root_state, depths=(1, 2, 4))
            root_base = float(
                _predict_count_from_summary(model, model.decode_summary(root_state))
                .detach()
                .cpu()
            )
            root_drift_r1_total += abs(
                float(_predict_count_from_summary(model, root_replay[1]).detach().cpu())
                - root_base
            )
            root_drift_r2_total += abs(
                float(_predict_count_from_summary(model, root_replay[2]).detach().cpu())
                - root_base
            )
            root_drift_r4_total += abs(
                float(_predict_count_from_summary(model, root_replay[4]).detach().cpu())
                - root_base
            )
            root_drift_count += 1

    tau = float(tau)
    root_abs_arr = np.asarray(root_abs, dtype=np.float64)
    spreads_arr = np.asarray(spreads, dtype=np.float64)
    condition_metrics = _condition_error_diagnostics(root_abs_arr, condition_ids)

    root_sq_arr = (root_abs_arr) ** 2  # abs_arr is already |pred - truth|, so sq = abs^2
    return SketchMetrics(
        root_mae=float(np.mean(root_abs_arr)),
        root_mse=float(np.mean(root_sq_arr)),
        root_median_abs_error=float(np.median(root_abs_arr)),
        root_p95_abs_error=float(np.percentile(root_abs_arr, 95.0)),
        schedule_spread_mean=float(np.mean(spreads_arr)),
        schedule_spread_p95=float(np.percentile(spreads_arr, 95.0)),
        leaf_mae=_mean_or_default(
            total=leaf_abs_total,
            count=leaf_abs_count,
            default=0.0,
        ),
        leaf_violation_rate=_mean_or_default(
            total=float(leaf_violation_count),
            count=leaf_abs_count,
            default=0.0,
        ),
        c2_idempotence_mae=_mean_or_default(
            total=c2_r1_total,
            count=c2_count,
            default=0.0,
        ),
        c2_r1_mae=_mean_or_default(
            total=c2_r1_total,
            count=c2_count,
            default=0.0,
        ),
        c2_r2_mae=_mean_or_default(
            total=c2_r2_total,
            count=c2_count,
            default=0.0,
        ),
        c2_r4_mae=_mean_or_default(
            total=c2_r4_total,
            count=c2_count,
            default=0.0,
        ),
        resummary_root_drift_r1=_mean_or_default(
            total=root_drift_r1_total,
            count=root_drift_count,
            default=0.0,
        ),
        resummary_root_drift_r2=_mean_or_default(
            total=root_drift_r2_total,
            count=root_drift_count,
            default=0.0,
        ),
        resummary_root_drift_r4=_mean_or_default(
            total=root_drift_r4_total,
            count=root_drift_count,
            default=0.0,
        ),
        merge_mae=_mean_or_default(
            total=merge_abs_total,
            count=merge_abs_count,
            default=0.0,
        ),
        merge_violation_rate=_mean_or_default(
            total=float(merge_violation_count),
            count=merge_abs_count,
            default=0.0,
        ),
        n_docs=int(len(docs)),
        c2_state_replay_mse=_mean_or_default(
            total=c2_state_replay_total,
            count=c2_count,
            default=0.0,
        ),
        condition_root_mae=dict(condition_metrics["condition_root_mae"]),
        condition_root_n_docs=dict(condition_metrics["condition_root_n_docs"]),
        condition_root_macro_mae=float(condition_metrics["condition_root_macro_mae"]),
        condition_root_worst_mae=float(condition_metrics["condition_root_worst_mae"]),
    )


@torch.no_grad()
def _eval_fno_full_tree_ipw_metrics(
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
    leaf_sample_rate: float,
    internal_sample_rate: float,
    use_residual_decomposition: bool,
    seed: int,
) -> Dict[str, Any]:
    """Evaluate the full-tree node estimand and separate document-level top loss."""

    if len(docs) == 0:
        return summarize_full_tree_ipw([], [])

    import random as _random

    rng = _random.Random(int(seed))
    model.eval()
    accumulator = FullTreeIPWSummaryAccumulator()

    for index, doc in enumerate(docs):
        sampled_leaf_indices, leaf_propensity = _sample_realized_node_indices(
            rng,
            n_nodes=int(len(doc.leaf_token_ids)),
            sample_rate=float(leaf_sample_rate),
        )
        sampled_internal_indices, internal_propensity = _sample_realized_node_indices(
            rng,
            n_nodes=int(len(doc.merge_counts_balanced)),
            sample_rate=float(internal_sample_rate),
        )
        out = model.forward_doc_unified(
            doc.leaf_token_ids,
            doc.leaf_counts,
            doc.merge_counts_balanced,
            doc.root_count,
            doc_id=f"eval_doc_{index}",
            schedule="balanced",
            device=device,
            sampled_leaf_indices=sampled_leaf_indices,
            sampled_internal_indices=sampled_internal_indices,
            leaf_propensity=float(leaf_propensity),
            internal_propensity=float(internal_propensity),
            proxy_leaf_counts=(doc.proxy_leaf_counts if doc.proxy_leaf_counts else None),
            proxy_merge_counts_balanced=(
                doc.proxy_merge_counts_balanced
                if doc.proxy_merge_counts_balanced
                else None
            ),
            use_residual_decomposition=bool(use_residual_decomposition),
            collect_full_trace=True,  # consumer reads node_records/document_record
        )
        for record in list(out["node_records"]):
            accumulator.update_node_record(record)
        accumulator.update_document_record(out["document_record"])

    return accumulator.finalize()


@torch.no_grad()
def _eval_fno_objective_terms(
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    *,
    device: torch.device,
    leaf_weight: float,
    c2_weight: float,
    c3_weight: float,
    root_weight: float,
    schedule_consistency_weight: float,
    include_root_query: bool,
) -> ObjectiveMetrics:
    """Evaluate weighted objective terms for FNOCountSketch."""
    if len(docs) == 0:
        return ObjectiveMetrics(
            optimization_total_loss=0.0,
            optimization_root_loss=0.0,
            optimization_leaf_loss=0.0,
            optimization_c2_loss=0.0,
            optimization_merge_loss=0.0,
            optimization_schedule_consistency_loss=0.0,
            raw_total_loss=0.0,
            raw_root_loss=0.0,
            raw_leaf_loss=0.0,
            raw_c2_loss=0.0,
            raw_merge_loss=0.0,
            raw_schedule_consistency_loss=0.0,
            n_docs=0,
        )

    model.eval()
    optimization_total_terms: List[float] = []
    optimization_root_terms: List[float] = []
    optimization_leaf_terms: List[float] = []
    optimization_c2_terms: List[float] = []
    optimization_merge_terms: List[float] = []
    optimization_consistency_terms: List[float] = []
    raw_total_terms: List[float] = []
    raw_root_terms: List[float] = []
    raw_leaf_terms: List[float] = []
    raw_c2_terms: List[float] = []
    raw_merge_terms: List[float] = []
    raw_consistency_terms: List[float] = []

    for doc in docs:
        out = model.forward_doc(
            doc.leaf_token_ids,
            doc.leaf_counts,
            doc.merge_counts_balanced,
            doc.merge_token_lengths,
            schedule="balanced",
            collect_leaf=True,
            collect_c3=True,
            collect_c2=True,
            device=device,
            leaf_audit_indices=None,
            c3_audit_indices=None,
        )
        pred_norm = out["pred_norm"]
        leaf_loss_tensor = out["leaf_loss"]
        c2_loss_tensor = out["c2_loss"]
        c3_loss_tensor = out["c3_loss"]

        true_norm = torch.tensor(
            float(doc.root_count) / float(model.target_scale),
            device=device,
            dtype=pred_norm.dtype,
        )
        raw_root_term = (
            float(F.mse_loss(pred_norm, true_norm, reduction="mean").detach().cpu())
            if bool(include_root_query)
            else 0.0
        )
        raw_leaf_term = float(leaf_loss_tensor.detach().cpu())
        raw_c2_term = float(c2_loss_tensor.detach().cpu())
        raw_merge_term = float(c3_loss_tensor.detach().cpu())

        if float(schedule_consistency_weight) > 0.0 and len(doc.leaf_token_ids) > 1:
            state_batch_sched = model.encode_leaf_tokens_batch(doc.leaf_token_ids, device=device)
            states_sched = [
                state_batch_sched[idx] for idx in range(int(state_batch_sched.shape[0]))
            ]
            sched_preds = []
            for sched in VALID_SCHEDULES:
                root_state_sched, _ = model._merge_states(
                    states_sched, schedule=sched, collect_merge_states=False,
                )
                sched_preds.append(model.predict_norm_from_state(root_state_sched))
            pred_stack = torch.stack(sched_preds, dim=0)
            consistency_raw = torch.mean((pred_stack - torch.mean(pred_stack)) ** 2)
            raw_consistency_term = float(consistency_raw.detach().cpu())
        else:
            raw_consistency_term = 0.0

        optimization_root_term = float(root_weight) * raw_root_term
        optimization_leaf_term = float(leaf_weight) * raw_leaf_term
        optimization_c2_term = float(c2_weight) * raw_c2_term
        optimization_merge_term = float(c3_weight) * raw_merge_term
        optimization_consistency_term = float(schedule_consistency_weight) * raw_consistency_term

        optimization_total_terms.append(
            float(
                optimization_root_term
                + optimization_leaf_term
                + optimization_c2_term
                + optimization_merge_term
                + optimization_consistency_term
            )
        )
        optimization_root_terms.append(float(optimization_root_term))
        optimization_leaf_terms.append(float(optimization_leaf_term))
        optimization_c2_terms.append(float(optimization_c2_term))
        optimization_merge_terms.append(float(optimization_merge_term))
        optimization_consistency_terms.append(float(optimization_consistency_term))
        raw_total_terms.append(
            float(raw_root_term + raw_leaf_term + raw_c2_term + raw_merge_term + raw_consistency_term)
        )
        raw_root_terms.append(float(raw_root_term))
        raw_leaf_terms.append(float(raw_leaf_term))
        raw_c2_terms.append(float(raw_c2_term))
        raw_merge_terms.append(float(raw_merge_term))
        raw_consistency_terms.append(float(raw_consistency_term))

    return ObjectiveMetrics(
        optimization_total_loss=float(np.mean(np.asarray(optimization_total_terms, dtype=np.float64))),
        optimization_root_loss=float(np.mean(np.asarray(optimization_root_terms, dtype=np.float64))),
        optimization_leaf_loss=float(np.mean(np.asarray(optimization_leaf_terms, dtype=np.float64))),
        optimization_c2_loss=float(np.mean(np.asarray(optimization_c2_terms, dtype=np.float64))),
        optimization_merge_loss=float(np.mean(np.asarray(optimization_merge_terms, dtype=np.float64))),
        optimization_schedule_consistency_loss=float(np.mean(np.asarray(optimization_consistency_terms, dtype=np.float64))),
        raw_total_loss=float(np.mean(np.asarray(raw_total_terms, dtype=np.float64))),
        raw_root_loss=float(np.mean(np.asarray(raw_root_terms, dtype=np.float64))),
        raw_leaf_loss=float(np.mean(np.asarray(raw_leaf_terms, dtype=np.float64))),
        raw_c2_loss=float(np.mean(np.asarray(raw_c2_terms, dtype=np.float64))),
        raw_merge_loss=float(np.mean(np.asarray(raw_merge_terms, dtype=np.float64))),
        raw_schedule_consistency_loss=float(np.mean(np.asarray(raw_consistency_terms, dtype=np.float64))),
        n_docs=int(len(docs)),
    )
