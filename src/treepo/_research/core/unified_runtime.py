"""
Unified batching runtime primitives shared by LLM and neural-tree execution.

The goal of this module is not to force every backend into one physical batch,
but to give all batching paths a shared IR, planner, plan-cache, and telemetry
surface so higher-level orchestration can reason about work uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple

import torch


logger = logging.getLogger(__name__)

RUNTIME_MODE_LEGACY = "legacy"
RUNTIME_MODE_UNIFIED_V2 = "unified_v2"
VALID_RUNTIME_MODES = {RUNTIME_MODE_LEGACY, RUNTIME_MODE_UNIFIED_V2}
GPU_RUNTIME_DATA_MODE_RESIDENT = "resident"
GPU_RUNTIME_DATA_MODE_CPU_DEBUG = "cpu_debug"
VALID_GPU_RUNTIME_DATA_MODES = {
    GPU_RUNTIME_DATA_MODE_RESIDENT,
    GPU_RUNTIME_DATA_MODE_CPU_DEBUG,
}
GPU_RUNTIME_BUCKET_MODE_EXACT_THEN_BUCKETED = "exact_then_bucketed"
GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE = "leaf_count_auto_queue"
VALID_GPU_RUNTIME_BUCKET_MODES = {
    GPU_RUNTIME_BUCKET_MODE_EXACT_THEN_BUCKETED,
    GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE,
}


def normalize_runtime_mode(value: Optional[str]) -> str:
    rendered = str(value or "").strip().lower()
    if rendered in VALID_RUNTIME_MODES:
        return rendered
    return RUNTIME_MODE_LEGACY


def resolve_runtime_mode(
    value: Optional[str],
    *,
    env_var: str = "TT_UNIFIED_BATCH_RUNTIME_MODE",
) -> str:
    if value is not None and str(value).strip():
        return normalize_runtime_mode(value)
    env_value = os.getenv(env_var)
    if env_value is not None and str(env_value).strip():
        return normalize_runtime_mode(env_value)
    return RUNTIME_MODE_LEGACY


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(encoded.encode("utf-8", errors="ignore")).hexdigest()


def _length_band(value: int) -> int:
    target = max(1, int(value))
    band = 1
    while int(band) < int(target):
        band *= 2
    return int(band)


def _normalize_preload_splits(value: Sequence[str] | str | None) -> Tuple[str, ...]:
    if value is None:
        return ("train", "val", "test")
    if isinstance(value, str):
        tokens = [
            token.strip().lower()
            for token in str(value).replace(",", " ").split()
            if token.strip()
        ]
    else:
        tokens = [str(item).strip().lower() for item in value if str(item).strip()]
    deduped: List[str] = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return tuple(deduped or ("train", "val", "test"))


def normalize_gpu_runtime_data_mode(value: Optional[str]) -> str:
    rendered = str(value or "").strip().lower()
    if rendered in VALID_GPU_RUNTIME_DATA_MODES:
        return rendered
    return GPU_RUNTIME_DATA_MODE_RESIDENT


def normalize_gpu_runtime_bucket_mode(value: Optional[str]) -> str:
    rendered = str(value or "").strip().lower()
    if rendered in VALID_GPU_RUNTIME_BUCKET_MODES:
        return rendered
    return GPU_RUNTIME_BUCKET_MODE_EXACT_THEN_BUCKETED


@dataclass(frozen=True)
class GpuRuntimeConfig:
    data_mode: str = GPU_RUNTIME_DATA_MODE_RESIDENT
    bucket_mode: str = GPU_RUNTIME_BUCKET_MODE_EXACT_THEN_BUCKETED
    preload_splits: Tuple[str, ...] = ("train", "val", "test")
    preload_targets: bool = True
    workers_per_mig: int = 1
    allow_multi_worker_screen: bool = True
    capacity_workers_per_mig: int = 2

    @property
    def is_resident(self) -> bool:
        return str(self.data_mode) == GPU_RUNTIME_DATA_MODE_RESIDENT

    def should_preload_split(self, split_name: str) -> bool:
        return str(split_name or "").strip().lower() in set(self.preload_splits)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "data_mode": str(self.data_mode),
            "bucket_mode": str(self.bucket_mode),
            "preload_splits": list(self.preload_splits),
            "preload_targets": bool(self.preload_targets),
            "workers_per_mig": int(self.workers_per_mig),
            "allow_multi_worker_screen": bool(self.allow_multi_worker_screen),
            "capacity_workers_per_mig": int(self.capacity_workers_per_mig),
        }


def gpu_runtime_config_from_mapping(
    payload: Mapping[str, Any] | None,
    *,
    device_type: str | None = None,
) -> GpuRuntimeConfig:
    mapping = dict(payload or {})
    default_data_mode = (
        GPU_RUNTIME_DATA_MODE_RESIDENT
        if str(device_type or "").strip().lower() == "cuda"
        else GPU_RUNTIME_DATA_MODE_CPU_DEBUG
    )
    return GpuRuntimeConfig(
        data_mode=normalize_gpu_runtime_data_mode(
            mapping.get("data_mode", default_data_mode)
        ),
        bucket_mode=normalize_gpu_runtime_bucket_mode(
            mapping.get("bucket_mode", GPU_RUNTIME_BUCKET_MODE_EXACT_THEN_BUCKETED)
        ),
        preload_splits=_normalize_preload_splits(mapping.get("preload_splits")),
        preload_targets=bool(mapping.get("preload_targets", True)),
        workers_per_mig=max(1, int(mapping.get("workers_per_mig", 1) or 1)),
        allow_multi_worker_screen=bool(
            mapping.get("allow_multi_worker_screen", True)
        ),
        capacity_workers_per_mig=max(
            1,
            int(mapping.get("capacity_workers_per_mig", 2) or 2),
        ),
    )


@dataclass(frozen=True)
class GpuBatchStoreKey:
    backend_family: str
    topology_signature: str
    leaf_count_band: int
    max_leaf_tokens_band: int
    work_kind: str = ""
    supervision_mask: str = ""
    exact_layout_signature: str = ""

    @property
    def stable_key(self) -> str:
        return _stable_digest(
            {
                "backend_family": str(self.backend_family or ""),
                "topology_signature": str(self.topology_signature or ""),
                "leaf_count_band": int(self.leaf_count_band),
                "max_leaf_tokens_band": int(self.max_leaf_tokens_band),
                "work_kind": str(self.work_kind or ""),
                "supervision_mask": str(self.supervision_mask or ""),
                "exact_layout_signature": str(self.exact_layout_signature or ""),
            }
        )


@dataclass(frozen=True)
class GpuBatchView:
    store_key: GpuBatchStoreKey
    doc_indices: Tuple[int, ...]
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GpuRuntimeTelemetry:
    data_mode: str = GPU_RUNTIME_DATA_MODE_RESIDENT
    bucket_mode: str = GPU_RUNTIME_BUCKET_MODE_EXACT_THEN_BUCKETED
    workers_per_mig: int = 1
    resident_store_build_time_s: float = 0.0
    steady_state_h2d_bytes: int = 0
    steady_state_h2d_events: int = 0
    steady_state_h2d_time_s: float = 0.0
    resident_store_hits: int = 0
    resident_store_misses: int = 0
    cpu_fallback_reason_counts: Dict[str, int] = field(default_factory=dict)
    extra_counters: Dict[str, float] = field(default_factory=dict)

    def add_store_build(self, *, wall_time_s: float) -> None:
        self.resident_store_build_time_s += float(max(0.0, wall_time_s))

    def add_h2d(self, *, bytes_transferred: int, wall_time_s: float) -> None:
        self.steady_state_h2d_bytes += int(max(0, bytes_transferred))
        if int(max(0, bytes_transferred)) > 0 or float(max(0.0, wall_time_s)) > 0.0:
            self.steady_state_h2d_events += 1
        self.steady_state_h2d_time_s += float(max(0.0, wall_time_s))

    def add_store_hit(self) -> None:
        self.resident_store_hits += 1

    def add_store_miss(self, *, reason: str = "") -> None:
        self.resident_store_misses += 1
        if str(reason or "").strip():
            key = str(reason)
            self.cpu_fallback_reason_counts[key] = (
                int(self.cpu_fallback_reason_counts.get(key, 0)) + 1
            )

    def add_extra_counter(self, name: str, value: float = 1.0) -> None:
        self.extra_counters[str(name)] = float(
            self.extra_counters.get(str(name), 0.0)
        ) + float(value)

    def as_dict(
        self,
        *,
        gpu_reserved_peak_gb: Optional[float] = None,
        gpu_allocated_peak_gb: Optional[float] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "runtime_data_mode": str(self.data_mode),
            "runtime_bucket_mode": str(self.bucket_mode),
            "runtime_workers_per_mig": int(self.workers_per_mig),
            "resident_store_build_time_s": float(self.resident_store_build_time_s),
            "steady_state_h2d_bytes": int(self.steady_state_h2d_bytes),
            "steady_state_h2d_events": int(self.steady_state_h2d_events),
            "steady_state_h2d_time_s": float(self.steady_state_h2d_time_s),
            "host_to_device_bytes": int(self.steady_state_h2d_bytes),
            "host_to_device_events": int(self.steady_state_h2d_events),
            "resident_store_hits": int(self.resident_store_hits),
            "resident_store_misses": int(self.resident_store_misses),
            "cpu_fallback_reason_counts": dict(self.cpu_fallback_reason_counts),
            "gpu_reserved_mem_peak_gb": (
                float(gpu_reserved_peak_gb)
                if gpu_reserved_peak_gb is not None
                else float("nan")
            ),
            "gpu_allocated_mem_peak_gb": (
                float(gpu_allocated_peak_gb)
                if gpu_allocated_peak_gb is not None
                else float("nan")
            ),
        }
        payload.update({str(key): float(value) for key, value in self.extra_counters.items()})
        return payload


def _pad_tensor_like(
    tensor: torch.Tensor,
    *,
    target_shape: Sequence[int],
    pad_value: float,
) -> torch.Tensor:
    if tuple(int(v) for v in tensor.shape) == tuple(int(v) for v in target_shape):
        return tensor
    padded = tensor.new_full(tuple(int(v) for v in target_shape), pad_value)
    slices = tuple(slice(0, int(dim)) for dim in tensor.shape)
    padded[slices] = tensor
    return padded


@dataclass
class GpuBatchStore:
    backend_family: str
    split_name: str
    config: GpuRuntimeConfig
    device: str
    telemetry: GpuRuntimeTelemetry = field(default_factory=GpuRuntimeTelemetry)
    buckets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    doc_locations: Dict[int, Tuple[str, int]] = field(default_factory=dict)

    def add_bucket(
        self,
        *,
        key: GpuBatchStoreKey,
        doc_indices: Sequence[int],
        tensors: Mapping[str, torch.Tensor],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        bucket_id = str(key.stable_key)
        self.buckets[bucket_id] = {
            "key": key,
            "doc_indices": tuple(int(index) for index in doc_indices),
            "tensors": {str(name): tensor for name, tensor in tensors.items()},
            "metadata": dict(metadata or {}),
        }
        for local_idx, doc_index in enumerate(doc_indices):
            self.doc_locations[int(doc_index)] = (bucket_id, int(local_idx))

    def view_for_doc_indices(
        self,
        doc_indices: Sequence[int],
        *,
        pad_values: Mapping[str, float] | None = None,
    ) -> Optional[GpuBatchView]:
        if not doc_indices:
            self.telemetry.add_store_miss(reason="empty_doc_indices")
            return None
        grouped: Dict[str, List[Tuple[int, int]]] = {}
        ordered_doc_indices: List[int] = []
        for doc_index in doc_indices:
            location = self.doc_locations.get(int(doc_index))
            if location is None:
                self.telemetry.add_store_miss(reason="doc_not_resident")
                return None
            bucket_id, local_idx = location
            grouped.setdefault(str(bucket_id), []).append((int(local_idx), int(doc_index)))
            ordered_doc_indices.append(int(doc_index))
        bucket_keys = [self.buckets[bucket_id]["key"] for bucket_id in grouped]
        primary_key = bucket_keys[0]
        if any(
            (
                int(key.leaf_count_band) != int(primary_key.leaf_count_band)
                or str(key.backend_family) != str(primary_key.backend_family)
            )
            for key in bucket_keys[1:]
        ):
            self.telemetry.add_store_miss(reason="incompatible_bucket_mix")
            return None

        tensor_chunks: Dict[str, List[torch.Tensor]] = {}
        metadata_chunks: List[Mapping[str, Any]] = []
        target_shapes: Dict[str, List[int]] = {}
        for bucket_id, entries in grouped.items():
            bucket = self.buckets[str(bucket_id)]
            metadata_chunks.append(dict(bucket.get("metadata", {}) or {}))
            local_positions = torch.as_tensor(
                [int(local_idx) for local_idx, _doc_idx in entries],
                dtype=torch.long,
                device=next(iter(bucket["tensors"].values())).device,
            )
            for name, tensor in dict(bucket["tensors"]).items():
                selected = tensor.index_select(0, local_positions)
                tensor_chunks.setdefault(str(name), []).append(selected)
                shape = target_shapes.setdefault(str(name), list(selected.shape))
                for dim_idx, dim_size in enumerate(selected.shape):
                    if dim_idx >= len(shape):
                        shape.append(int(dim_size))
                    else:
                        shape[dim_idx] = max(int(shape[dim_idx]), int(dim_size))

        merged_tensors: Dict[str, torch.Tensor] = {}
        for name, chunks in tensor_chunks.items():
            if len(chunks) == 1:
                merged_tensors[str(name)] = chunks[0]
                continue
            target_shape = tuple(int(v) for v in target_shapes[str(name)])
            pad_value = float((pad_values or {}).get(str(name), 0.0))
            merged_tensors[str(name)] = torch.cat(
                [
                    _pad_tensor_like(
                        chunk,
                        target_shape=(int(chunk.shape[0]), *target_shape[1:]),
                        pad_value=pad_value,
                    )
                    for chunk in chunks
                ],
                dim=0,
            )

        self.telemetry.add_store_hit()
        common_metadata: Dict[str, Any] = {}
        for name in (
            "bucket_store_mode",
            "resident_layout_mode",
            "auto_queue_enabled",
            "auto_queue_target_n_leaves",
        ):
            values = [
                metadata.get(str(name))
                for metadata in metadata_chunks
                if str(metadata.get(str(name), "")).strip()
            ]
            if values and all(value == values[0] for value in values[1:]):
                common_metadata[str(name)] = values[0]
        resident_bucket_bytes = sum(
            int(metadata.get("resident_bucket_bytes", 0) or 0)
            for metadata in metadata_chunks
        )
        if resident_bucket_bytes > 0:
            common_metadata["resident_bucket_bytes"] = int(resident_bucket_bytes)
        return GpuBatchView(
            store_key=primary_key,
            doc_indices=tuple(int(index) for index in ordered_doc_indices),
            tensors=merged_tensors,
            metadata={
                "split_name": str(self.split_name),
                "bucket_count": int(len(grouped)),
                "bucket_metadata": [dict(meta) for meta in metadata_chunks],
                **common_metadata,
            },
        )


@dataclass(frozen=True)
class TopologyRef:
    index: int
    is_internal: bool


@dataclass(frozen=True)
class TopologyNode:
    node_id: str
    level: int
    left: Optional[TopologyRef] = None
    right: Optional[TopologyRef] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass
class TopologyPlan:
    """Common dependency graph for fixed, adaptive, unified, and neural trees."""

    doc_id: str
    doc_index: int
    topology_signature: str
    internal_nodes: Dict[int, TopologyNode]
    dependents_by_leaf: Dict[int, List[int]]
    dependents_by_internal: Dict[int, List[int]]
    remaining_deps: Dict[int, int]
    final_ref: TopologyRef
    max_level: int
    leaf_count: int
    internal_count: int
    plan_summary: Dict[str, Any]

    def copy_remaining_deps(self) -> Dict[int, int]:
        return {int(key): int(value) for key, value in self.remaining_deps.items()}


def build_balanced_topology_plan(
    *,
    doc_index: int,
    doc_id: str,
    leaf_metadata: Sequence[Mapping[str, Any]],
) -> TopologyPlan:
    leaf_ids = [f"d{doc_index}_leaf_{idx}" for idx in range(len(leaf_metadata))]
    leaf_token_counts = [
        int(item.get("token_count", 0) or max(1, int(item.get("char_count", 0) or 0) // 4))
        for item in leaf_metadata
    ]
    levels: List[List[str]] = [leaf_ids.copy()] if leaf_ids else []
    edges: List[Dict[str, str]] = []
    internal_nodes: Dict[int, TopologyNode] = {}
    dependents_by_leaf: Dict[int, List[int]] = {}
    dependents_by_internal: Dict[int, List[int]] = {}
    remaining_deps: Dict[int, int] = {}

    current_refs: List[TopologyRef] = [
        TopologyRef(index=int(idx), is_internal=False) for idx in range(len(leaf_metadata))
    ]
    internal_id = 0
    level = 0

    while len(current_refs) > 1:
        level += 1
        if len(levels) <= level:
            levels.append([])
        next_refs: List[TopologyRef] = []
        for idx in range(0, len(current_refs), 2):
            if idx + 1 >= len(current_refs):
                next_refs.append(current_refs[idx])
                continue
            left = current_refs[idx]
            right = current_refs[idx + 1]
            left_input_tokens = (
                int(internal_nodes[left.index].metadata.get("estimated_output_tokens", 0))
                if left.is_internal
                else int(leaf_token_counts[left.index])
            )
            right_input_tokens = (
                int(internal_nodes[right.index].metadata.get("estimated_output_tokens", 0))
                if right.is_internal
                else int(leaf_token_counts[right.index])
            )
            estimated_input_tokens = int(left_input_tokens + right_input_tokens)
            node_id = f"d{doc_index}_L{level}_{internal_id}"
            internal_nodes[internal_id] = TopologyNode(
                node_id=node_id,
                level=int(level),
                left=left,
                right=right,
                metadata={
                    "estimated_input_tokens": int(estimated_input_tokens),
                    "estimated_output_tokens": max(1, int(estimated_input_tokens // 2)),
                },
            )
            remaining_deps[internal_id] = 2
            if left.is_internal:
                dependents_by_internal.setdefault(int(left.index), []).append(int(internal_id))
            else:
                dependents_by_leaf.setdefault(int(left.index), []).append(int(internal_id))
            if right.is_internal:
                dependents_by_internal.setdefault(int(right.index), []).append(int(internal_id))
            else:
                dependents_by_leaf.setdefault(int(right.index), []).append(int(internal_id))

            levels[level].append(node_id)
            edges.append(
                {
                    "parent": node_id,
                    "left": internal_nodes[left.index].node_id if left.is_internal else leaf_ids[left.index],
                    "right": internal_nodes[right.index].node_id if right.is_internal else leaf_ids[right.index],
                }
            )
            next_refs.append(TopologyRef(index=int(internal_id), is_internal=True))
            internal_id += 1
        current_refs = next_refs

    if current_refs:
        final_ref = current_refs[0]
        root_id = (
            internal_nodes[final_ref.index].node_id
            if final_ref.is_internal
            else leaf_ids[final_ref.index]
        )
    else:
        final_ref = TopologyRef(index=0, is_internal=False)
        root_id = None

    summary = {
        "doc_id": str(doc_id),
        "doc_index": int(doc_index),
        "leaf_count": int(len(leaf_metadata)),
        "merge_count": int(len(internal_nodes)),
        "levels": levels,
        "edges": edges,
        "root_id": root_id,
        "leaf_nodes": [dict(item) for item in leaf_metadata],
        "topology_kind": "balanced_binary",
    }
    topology_signature = _stable_digest(
        {
            "kind": "balanced_binary",
            "leaf_count": int(len(leaf_metadata)),
            "levels": tuple(len(level_ids) for level_ids in levels),
        }
    )
    return TopologyPlan(
        doc_id=str(doc_id),
        doc_index=int(doc_index),
        topology_signature=str(topology_signature),
        internal_nodes=internal_nodes,
        dependents_by_leaf=dependents_by_leaf,
        dependents_by_internal=dependents_by_internal,
        remaining_deps=remaining_deps,
        final_ref=final_ref,
        max_level=int(level),
        leaf_count=int(len(leaf_metadata)),
        internal_count=int(len(internal_nodes)),
        plan_summary=summary,
    )


def build_unified_topology_plan(
    *,
    doc_index: int,
    doc_id: str,
    nodes: Sequence[Any],
) -> TopologyPlan:
    """Lower an ``EmbeddingTreeNode`` list to the shared topology contract."""
    leaf_ids: List[str] = []
    leaf_nodes: List[Dict[str, Any]] = []
    internal_nodes: Dict[int, TopologyNode] = {}
    dependents_by_leaf: Dict[int, List[int]] = {}
    dependents_by_internal: Dict[int, List[int]] = {}
    remaining_deps: Dict[int, int] = {}
    node_ref_by_original_index: Dict[int, TopologyRef] = {}
    node_id_by_original_index: Dict[int, str] = {}
    levels_map: Dict[int, List[str]] = {}

    leaf_count = 0
    internal_count = 0
    max_level = 0
    output_token_budget_by_original_index: Dict[int, int] = {}

    for original_index, raw_node in enumerate(nodes):
        level = int(getattr(raw_node, "level", 0) or 0)
        max_level = max(max_level, level)
        if bool(getattr(raw_node, "is_leaf", False)):
            leaf_id = f"d{doc_index}_leaf_{leaf_count}"
            node_ref_by_original_index[original_index] = TopologyRef(
                index=int(leaf_count),
                is_internal=False,
            )
            node_id_by_original_index[original_index] = leaf_id
            leaf_ids.append(leaf_id)
            levels_map.setdefault(level, []).append(leaf_id)
            text_span = str(getattr(raw_node, "text_span", "") or "")
            leaf_nodes.append(
                {
                    "id": leaf_id,
                    "chunk_index": int(leaf_count),
                    "start_char": int(getattr(raw_node, "char_start", 0) or 0),
                    "end_char": int(getattr(raw_node, "char_end", len(text_span)) or len(text_span)),
                    "char_count": len(text_span),
                    "token_count": max(1, len(text_span) // 4) if text_span else 0,
                    "embedding_tree_index": int(original_index),
                }
            )
            output_token_budget_by_original_index[original_index] = max(1, len(text_span) // 4) if text_span else 0
            leaf_count += 1
            continue

        children = getattr(raw_node, "children", None)
        if not isinstance(children, tuple) or len(children) != 2:
            raise ValueError(
                f"Unified tree internal node at index {original_index} is missing binary children"
            )
        left = node_ref_by_original_index[int(children[0])]
        right = node_ref_by_original_index[int(children[1])]
        left_input_tokens = int(output_token_budget_by_original_index.get(int(children[0]), 0))
        right_input_tokens = int(output_token_budget_by_original_index.get(int(children[1]), 0))
        estimated_input_tokens = int(left_input_tokens + right_input_tokens)
        node_id = f"d{doc_index}_U{level}_{internal_count}"
        node_ref_by_original_index[original_index] = TopologyRef(
            index=int(internal_count),
            is_internal=True,
        )
        node_id_by_original_index[original_index] = node_id
        levels_map.setdefault(level, []).append(node_id)
        internal_nodes[internal_count] = TopologyNode(
            node_id=node_id,
            level=int(level),
            left=left,
            right=right,
            metadata={
                "embedding_tree_index": int(original_index),
                "char_start": int(getattr(raw_node, "char_start", 0) or 0),
                "char_end": int(getattr(raw_node, "char_end", 0) or 0),
                "estimated_input_tokens": int(estimated_input_tokens),
                "estimated_output_tokens": max(1, int(estimated_input_tokens // 2)),
            },
        )
        output_token_budget_by_original_index[original_index] = max(1, int(estimated_input_tokens // 2))
        remaining_deps[internal_count] = 2
        if left.is_internal:
            dependents_by_internal.setdefault(int(left.index), []).append(int(internal_count))
        else:
            dependents_by_leaf.setdefault(int(left.index), []).append(int(internal_count))
        if right.is_internal:
            dependents_by_internal.setdefault(int(right.index), []).append(int(internal_count))
        else:
            dependents_by_leaf.setdefault(int(right.index), []).append(int(internal_count))
        internal_count += 1

    if not nodes:
        final_ref = TopologyRef(index=0, is_internal=False)
        root_id = None
    else:
        final_ref = node_ref_by_original_index[len(nodes) - 1]
        root_id = node_id_by_original_index[len(nodes) - 1]

    levels = [levels_map[level] for level in sorted(levels_map.keys())]
    edges: List[Dict[str, str]] = []
    for internal_idx, node in internal_nodes.items():
        assert node.left is not None and node.right is not None
        edges.append(
            {
                "parent": node.node_id,
                "left": internal_nodes[node.left.index].node_id if node.left.is_internal else leaf_ids[node.left.index],
                "right": internal_nodes[node.right.index].node_id if node.right.is_internal else leaf_ids[node.right.index],
            }
        )
    topology_signature = _stable_digest(
        {
            "kind": "embedding_tree",
            "shape": tuple(
                (
                    int(getattr(raw_node, "level", 0) or 0),
                    tuple(int(value) for value in getattr(raw_node, "children", tuple()) or tuple()),
                )
                for raw_node in nodes
            ),
        }
    )
    summary = {
        "doc_id": str(doc_id),
        "doc_index": int(doc_index),
        "leaf_count": int(leaf_count),
        "merge_count": int(internal_count),
        "levels": levels,
        "edges": edges,
        "root_id": root_id,
        "leaf_nodes": leaf_nodes,
        "topology_kind": "embedding_tree",
        "embedding_tree_node_count": int(len(nodes)),
    }
    return TopologyPlan(
        doc_id=str(doc_id),
        doc_index=int(doc_index),
        topology_signature=str(topology_signature),
        internal_nodes=internal_nodes,
        dependents_by_leaf=dependents_by_leaf,
        dependents_by_internal=dependents_by_internal,
        remaining_deps=remaining_deps,
        final_ref=final_ref,
        max_level=int(max_level),
        leaf_count=int(leaf_count),
        internal_count=int(internal_count),
        plan_summary=summary,
    )


@dataclass(frozen=True)
class WorkItem:
    item_id: str
    backend_family: str
    op_kind: str
    topology_signature: str
    supervision_mask: str = ""
    doc_id: str = ""
    payload: Any = None
    priority: int = 0
    estimated_tokens: int = 0
    estimated_nodes: int = 0
    estimated_merge_ops: int = 0
    padding_multiple: int = 1
    padding_length: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape_key(self) -> str:
        payload = {
            "backend_family": str(self.backend_family or ""),
            "op_kind": str(self.op_kind or ""),
            "topology_signature": str(self.topology_signature or ""),
            "supervision_mask": str(self.supervision_mask or ""),
            "token_band": int(_length_band(self.padding_length)),
            "node_band": int(_length_band(max(1, self.estimated_nodes))),
        }
        return _stable_digest(payload)


@dataclass(frozen=True)
class WorkBatch:
    backend_family: str
    op_kind: str
    shape_key: str
    items: Tuple[WorkItem, ...]
    actual_tokens: int
    padded_tokens: int
    total_nodes: int
    total_merge_ops: int
    fallback_reason: str = ""


class BatchExecutor(Protocol):
    async def execute(self, batch: WorkBatch) -> Any:
        ...


@dataclass
class BatchTelemetry:
    runtime_mode: str = RUNTIME_MODE_LEGACY
    n_batches: int = 0
    total_docs: int = 0
    total_leaf_tokens: int = 0
    total_padded_leaf_tokens: int = 0
    total_nodes: int = 0
    total_merge_ops: int = 0
    total_bucket_utilization: float = 0.0
    train_forward_time_s: float = 0.0
    train_backward_time_s: float = 0.0
    eval_time_s: float = 0.0
    idle_wait_time_s: float = 0.0
    fallback_reason_counts: Dict[str, int] = field(default_factory=dict)
    extra_counters: Dict[str, float] = field(default_factory=dict)

    def add_batch(
        self,
        batch: WorkBatch,
        *,
        token_budget: int,
        node_budget: int,
        max_docs_budget: int,
        fallback_reason: Optional[str] = None,
    ) -> None:
        self.n_batches += 1
        self.total_docs += int(len(batch.items))
        self.total_leaf_tokens += int(batch.actual_tokens)
        self.total_padded_leaf_tokens += int(batch.padded_tokens)
        self.total_nodes += int(batch.total_nodes)
        self.total_merge_ops += int(batch.total_merge_ops)
        if fallback_reason:
            self.fallback_reason_counts[str(fallback_reason)] = (
                int(self.fallback_reason_counts.get(str(fallback_reason), 0)) + 1
            )

        utilization_terms: List[float] = []
        if int(token_budget) > 0:
            utilization_terms.append(
                min(1.0, float(batch.actual_tokens) / float(max(1, int(token_budget))))
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
            self.total_bucket_utilization += (
                float(sum(utilization_terms)) / float(len(utilization_terms))
            )

    def add_extra_counter(self, name: str, value: float = 1.0) -> None:
        self.extra_counters[str(name)] = float(self.extra_counters.get(str(name), 0.0)) + float(value)

    def as_dict(self, *, gpu_reserved_peak_gb: Optional[float] = None) -> Dict[str, Any]:
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
        payload: Dict[str, Any] = {
            "runtime_mode": str(self.runtime_mode),
            "mean_docs_per_batch": float(mean_docs),
            "mean_leaf_tokens_per_batch": float(mean_leaf_tokens),
            "mean_nodes_per_batch": float(mean_nodes),
            "padding_waste_ratio": float(padding_waste_ratio),
            "bucket_utilization_rate": (
                float(self.total_bucket_utilization) / float(self.n_batches)
                if self.n_batches > 0
                else 0.0
            ),
            "train_forward_time_s": float(self.train_forward_time_s),
            "train_backward_time_s": float(self.train_backward_time_s),
            "eval_time_s": float(self.eval_time_s),
            "idle_wait_time_s": float(self.idle_wait_time_s),
            "gpu_reserved_mem_peak_gb": float(gpu_reserved_peak_gb)
            if gpu_reserved_peak_gb is not None
            else float("nan"),
            "fallback_reason_counts": dict(self.fallback_reason_counts),
        }
        payload.update({str(key): float(value) for key, value in self.extra_counters.items()})
        return payload


@dataclass
class BatchPlanCache:
    cache: MutableMapping[str, Tuple[int, ...]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, key: str) -> Optional[Tuple[int, ...]]:
        plan = self.cache.get(str(key))
        if plan is None:
            self.misses += 1
            return None
        self.hits += 1
        return tuple(int(value) for value in plan)

    def put(self, key: str, group_sizes: Sequence[int]) -> None:
        self.cache[str(key)] = tuple(int(max(1, size)) for size in group_sizes if int(size) > 0)

    def as_dict(self) -> Dict[str, int]:
        return {
            "hits": int(self.hits),
            "misses": int(self.misses),
            "entries": int(len(self.cache)),
        }


_NAMED_PLAN_CACHES: Dict[str, BatchPlanCache] = {}


def get_named_plan_cache(name: str) -> BatchPlanCache:
    key = str(name or "default").strip() or "default"
    cache = _NAMED_PLAN_CACHES.get(key)
    if cache is None:
        cache = BatchPlanCache()
        _NAMED_PLAN_CACHES[key] = cache
    return cache


def _make_bucket_plan_key(
    *,
    shape_key: str,
    docs_cap: int,
    token_budget: int,
    node_budget: int,
    merge_budget: int,
    ordered_items: Sequence[WorkItem],
) -> str:
    return _stable_digest(
        {
            "shape_key": str(shape_key),
            "docs_cap": int(docs_cap),
            "token_budget": int(token_budget),
            "node_budget": int(node_budget),
            "merge_budget": int(merge_budget),
            "items": tuple(
                (
                    int(item.estimated_tokens),
                    int(item.estimated_nodes),
                    int(item.estimated_merge_ops),
                    int(item.padding_multiple),
                    int(item.padding_length),
                    int(item.priority),
                )
                for item in ordered_items
            ),
        }
    )


def _build_work_batch(
    *,
    items: Sequence[WorkItem],
) -> WorkBatch:
    actual_tokens = int(sum(int(item.estimated_tokens) for item in items))
    total_nodes = int(sum(int(item.estimated_nodes) for item in items))
    total_merge_ops = int(sum(int(item.estimated_merge_ops) for item in items))
    max_padding_length = max((int(item.padding_length) for item in items), default=0)
    padded_tokens = int(
        max(
            actual_tokens,
            sum(int(item.padding_multiple) for item in items) * int(max_padding_length),
        )
    )
    first = items[0]
    return WorkBatch(
        backend_family=str(first.backend_family),
        op_kind=str(first.op_kind),
        shape_key=str(first.shape_key),
        items=tuple(items),
        actual_tokens=int(actual_tokens),
        padded_tokens=int(padded_tokens),
        total_nodes=int(total_nodes),
        total_merge_ops=int(total_merge_ops),
        fallback_reason=str(first.metadata.get("fallback_reason", "") or ""),
    )


def _auto_queue_group_key(item: WorkItem) -> str:
    override = str(item.metadata.get("auto_queue_group_key", "") or "").strip()
    if override:
        return override
    payload = {
        "backend_family": str(item.backend_family or ""),
        "op_kind": str(item.op_kind or ""),
        "supervision_mask": str(item.supervision_mask or ""),
        "token_band": int(_length_band(max(1, int(item.padding_length or 1)))),
    }
    return _stable_digest(payload)


def _auto_queue_leaf_count(item: WorkItem) -> int:
    raw = item.metadata.get("leaf_count")
    if raw is None:
        return 0
    try:
        value = int(raw)
    except Exception:
        return 0
    return max(0, value)


def _auto_queue_target_leaf_count(item: WorkItem) -> int:
    raw = item.metadata.get("auto_queue_target_leaf_count")
    if raw is None:
        return 0
    try:
        value = int(raw)
    except Exception:
        return 0
    return max(0, value)


def _auto_queue_target_docs_cap_key(
    *,
    item: WorkItem,
    target_leaf_count: int,
) -> str:
    override = str(item.metadata.get("auto_queue_docs_cap_key", "") or "").strip()
    if override:
        return override
    return f"leaf_count_auto_queue:{_auto_queue_group_key(item)}:n{int(target_leaf_count)}"


def _leaf_count_padding_ratio(
    *,
    actual_leaf_count: int,
    target_leaf_count: int,
) -> float:
    actual = max(1, int(actual_leaf_count))
    target = max(actual, int(target_leaf_count))
    return float(max(0, target - actual)) / float(actual)


def build_leaf_count_auto_queue_targets(
    counts_by_leaf: Mapping[int, int],
    *,
    structural_pad_limit: float = 0.5,
    min_docs: int = 8,
) -> Dict[int, int]:
    normalized_counts = {
        int(key): max(0, int(value))
        for key, value in dict(counts_by_leaf).items()
        if int(key) > 0 and int(value) > 0
    }
    if not normalized_counts:
        return {}
    pad_limit = max(0.0, float(structural_pad_limit))
    min_docs = max(0, int(min_docs))
    sorted_leaf_counts = sorted(int(value) for value in normalized_counts.keys())
    families: List[Dict[str, Any]] = []
    current_counts: List[int] = []
    current_docs = 0
    for leaf_count in sorted_leaf_counts:
        proposed_counts = list(current_counts) + [int(leaf_count)]
        proposed_target = int(max(proposed_counts))
        if proposed_counts and all(
            _leaf_count_padding_ratio(
                actual_leaf_count=int(candidate),
                target_leaf_count=int(proposed_target),
            ) <= pad_limit
            for candidate in proposed_counts
        ):
            current_counts = proposed_counts
            current_docs += int(normalized_counts[int(leaf_count)])
            continue
        if current_counts:
            families.append(
                {
                    "leaf_counts": tuple(int(value) for value in current_counts),
                    "target_leaf_count": int(max(current_counts)),
                    "doc_count": int(current_docs),
                }
            )
        current_counts = [int(leaf_count)]
        current_docs = int(normalized_counts[int(leaf_count)])
    if current_counts:
        families.append(
            {
                "leaf_counts": tuple(int(value) for value in current_counts),
                "target_leaf_count": int(max(current_counts)),
                "doc_count": int(current_docs),
            }
        )

    if min_docs > 0 and len(families) > 1:
        merged = True
        while merged and len(families) > 1:
            merged = False
            for family_idx, family in enumerate(list(families)):
                if int(family["doc_count"]) >= min_docs:
                    continue
                merge_candidates: List[Tuple[float, int, int, Dict[str, Any]]] = []
                for offset in (-1, 1):
                    neighbor_idx = int(family_idx + offset)
                    if neighbor_idx < 0 or neighbor_idx >= len(families):
                        continue
                    neighbor = families[neighbor_idx]
                    merged_counts = sorted(
                        {
                            int(value)
                            for value in list(family["leaf_counts"]) + list(neighbor["leaf_counts"])
                        }
                    )
                    merged_target = int(max(merged_counts))
                    if any(
                        _leaf_count_padding_ratio(
                            actual_leaf_count=int(candidate),
                            target_leaf_count=int(merged_target),
                        ) > pad_limit
                        for candidate in merged_counts
                    ):
                        continue
                    merged_family = {
                        "leaf_counts": tuple(int(value) for value in merged_counts),
                        "target_leaf_count": int(merged_target),
                        "doc_count": int(family["doc_count"]) + int(neighbor["doc_count"]),
                    }
                    mean_padding = float(
                        sum(
                            _leaf_count_padding_ratio(
                                actual_leaf_count=int(candidate),
                                target_leaf_count=int(merged_target),
                            )
                            * float(normalized_counts[int(candidate)])
                            for candidate in merged_counts
                        )
                        / float(max(1, merged_family["doc_count"]))
                    )
                    merge_candidates.append(
                        (
                            float(mean_padding),
                            -int(merged_family["doc_count"]),
                            int(neighbor_idx),
                            merged_family,
                        )
                    )
                if not merge_candidates:
                    continue
                merge_candidates.sort(key=lambda value: (value[0], value[1]))
                chosen_neighbor_idx = int(merge_candidates[0][2])
                replacement = dict(merge_candidates[0][3])
                left_idx = min(int(family_idx), int(chosen_neighbor_idx))
                right_idx = max(int(family_idx), int(chosen_neighbor_idx))
                keep: List[Dict[str, Any]] = []
                for idx, existing in enumerate(families):
                    if left_idx <= idx <= right_idx:
                        continue
                    keep.append(existing)
                insert_at = max(0, min(left_idx, len(keep)))
                keep.insert(insert_at, replacement)
                families = keep
                merged = True
                break

    targets: Dict[int, int] = {}
    for family in families:
        target_leaf_count = int(family["target_leaf_count"])
        for leaf_count in family["leaf_counts"]:
            targets[int(leaf_count)] = int(target_leaf_count)
    return targets


def _infer_leaf_count_auto_queue_targets(
    items: Sequence[WorkItem],
    *,
    structural_pad_limit: float,
    min_docs: int,
) -> Dict[Tuple[str, int], int]:
    weighted_counts: Dict[str, Dict[int, int]] = {}
    for item in items:
        leaf_count = _auto_queue_leaf_count(item)
        if leaf_count <= 0:
            continue
        group_key = _auto_queue_group_key(item)
        weighted_counts.setdefault(group_key, {})
        weighted_counts[group_key][leaf_count] = (
            int(weighted_counts[group_key].get(leaf_count, 0)) + 1
        )

    pad_limit = max(0.0, float(structural_pad_limit))
    min_docs = max(0, int(min_docs))
    targets: Dict[Tuple[str, int], int] = {}
    for group_key, counts_by_leaf in weighted_counts.items():
        family_targets = build_leaf_count_auto_queue_targets(
            counts_by_leaf,
            structural_pad_limit=pad_limit,
            min_docs=min_docs,
        )
        for leaf_count, target_leaf_count in family_targets.items():
            targets[(str(group_key), int(leaf_count))] = int(target_leaf_count)
    return targets


def plan_work_batches(
    items: Sequence[WorkItem],
    *,
    max_docs: int,
    max_total_tokens: int,
    max_total_nodes: int,
    max_total_merge_ops: int,
    docs_cap_by_signature: Optional[Mapping[str, int]] = None,
    plan_cache: Optional[BatchPlanCache] = None,
    bucket_mode: str = GPU_RUNTIME_BUCKET_MODE_EXACT_THEN_BUCKETED,
    structural_pad_limit: float = 0.5,
    auto_queue_min_docs: int = 8,
) -> List[WorkBatch]:
    normalized_bucket_mode = normalize_gpu_runtime_bucket_mode(bucket_mode)
    grouped: Dict[str, List[WorkItem]] = {}
    docs_cap_key_by_group: Dict[str, str] = {}
    if normalized_bucket_mode == GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE:
        inferred_targets = _infer_leaf_count_auto_queue_targets(
            items,
            structural_pad_limit=float(structural_pad_limit),
            min_docs=int(auto_queue_min_docs),
        )
        for item in items:
            group_key = _auto_queue_group_key(item)
            leaf_count = _auto_queue_leaf_count(item)
            target_leaf_count = _auto_queue_target_leaf_count(item)
            if target_leaf_count <= 0 and leaf_count > 0:
                target_leaf_count = int(
                    inferred_targets.get((str(group_key), int(leaf_count)), int(leaf_count))
                )
            if target_leaf_count <= 0:
                grouped.setdefault(str(item.shape_key), []).append(item)
                docs_cap_key_by_group.setdefault(str(item.shape_key), str(item.shape_key))
                continue
            family_key = f"{str(group_key)}:target_leaf_count={int(target_leaf_count)}"
            grouped.setdefault(str(family_key), []).append(item)
            docs_cap_key_by_group.setdefault(
                str(family_key),
                _auto_queue_target_docs_cap_key(
                    item=item,
                    target_leaf_count=int(target_leaf_count),
                ),
            )
    else:
        for item in items:
            grouped.setdefault(str(item.shape_key), []).append(item)
            docs_cap_key_by_group.setdefault(str(item.shape_key), str(item.shape_key))

    batches: List[WorkBatch] = []
    for shape_key, bucket_items in grouped.items():
        ordered_items = sorted(
            bucket_items,
            key=lambda item: (
                -int(item.priority),
                -int(item.estimated_tokens),
                -int(item.estimated_nodes),
                str(item.item_id),
            ),
        )
        docs_cap = int(max_docs)
        if docs_cap_by_signature is not None:
            docs_cap = int(
                docs_cap_by_signature.get(
                    str(docs_cap_key_by_group.get(str(shape_key), str(shape_key))),
                    docs_cap,
                )
            )
        if docs_cap <= 0:
            docs_cap = 10**9

        plan_key = None
        cached_group_sizes: Optional[Tuple[int, ...]] = None
        if plan_cache is not None:
            plan_key = _make_bucket_plan_key(
                shape_key=str(shape_key),
                docs_cap=int(docs_cap),
                token_budget=int(max_total_tokens),
                node_budget=int(max_total_nodes),
                merge_budget=int(max_total_merge_ops),
                ordered_items=ordered_items,
            )
            cached_group_sizes = plan_cache.get(plan_key)
        if cached_group_sizes:
            cursor = 0
            for group_size in cached_group_sizes:
                group = ordered_items[cursor : cursor + int(group_size)]
                if group:
                    batches.append(_build_work_batch(items=group))
                cursor += int(group_size)
            if cursor >= len(ordered_items):
                continue

        active: List[WorkItem] = []
        group_sizes: List[int] = []
        actual_tokens = 0
        total_nodes = 0
        total_merge_ops = 0
        max_padding_length = 0
        total_padding_multiple = 0

        def _flush() -> None:
            nonlocal active, actual_tokens, total_nodes, total_merge_ops, max_padding_length, total_padding_multiple
            if not active:
                return
            batches.append(_build_work_batch(items=active))
            group_sizes.append(int(len(active)))
            active = []
            actual_tokens = 0
            total_nodes = 0
            total_merge_ops = 0
            max_padding_length = 0
            total_padding_multiple = 0

        for item in ordered_items:
            proposed_max_padding_length = max(int(max_padding_length), int(item.padding_length))
            proposed_padding_multiple = int(total_padding_multiple + item.padding_multiple)
            padded_if_added = int(proposed_max_padding_length * proposed_padding_multiple)
            proposed_tokens = int(actual_tokens + item.estimated_tokens)
            proposed_nodes = int(total_nodes + item.estimated_nodes)
            proposed_merge_ops = int(total_merge_ops + item.estimated_merge_ops)

            if active and (
                int(len(active) + 1) > int(docs_cap)
                or (int(max_total_tokens) > 0 and int(padded_if_added) > int(max_total_tokens))
                or (int(max_total_nodes) > 0 and int(proposed_nodes) > int(max_total_nodes))
                or (
                    int(max_total_merge_ops) > 0
                    and int(proposed_merge_ops) > int(max_total_merge_ops)
                )
            ):
                _flush()
                proposed_max_padding_length = int(item.padding_length)
                proposed_padding_multiple = int(item.padding_multiple)
                proposed_tokens = int(item.estimated_tokens)
                proposed_nodes = int(item.estimated_nodes)
                proposed_merge_ops = int(item.estimated_merge_ops)

            active.append(item)
            actual_tokens = int(proposed_tokens)
            total_nodes = int(proposed_nodes)
            total_merge_ops = int(proposed_merge_ops)
            max_padding_length = int(proposed_max_padding_length)
            total_padding_multiple = int(proposed_padding_multiple)
        _flush()

        if plan_key is not None and group_sizes:
            plan_cache.put(plan_key, group_sizes)

    batches.sort(
        key=lambda batch: (
            str(batch.backend_family),
            str(batch.op_kind),
            -int(len(batch.items)),
            -int(batch.actual_tokens),
        )
    )
    return batches


async def execute_batches(
    *,
    batches: Sequence[WorkBatch],
    executor: BatchExecutor,
    telemetry: Optional[BatchTelemetry] = None,
    token_budget: int = 0,
    node_budget: int = 0,
    max_docs_budget: int = 0,
) -> List[Any]:
    results: List[Any] = []
    for batch in batches:
        start = time.perf_counter()
        result = await executor.execute(batch)
        elapsed = time.perf_counter() - start
        results.append(result)
        if telemetry is not None:
            telemetry.eval_time_s += float(elapsed)
            telemetry.add_batch(
                batch,
                token_budget=int(token_budget),
                node_budget=int(node_budget),
                max_docs_budget=int(max_docs_budget) if int(max_docs_budget) > 0 else len(batch.items),
                fallback_reason=str(batch.fallback_reason or ""),
            )
    return results


__all__ = [
    "BatchExecutor",
    "BatchPlanCache",
    "BatchTelemetry",
    "GPU_RUNTIME_BUCKET_MODE_EXACT_THEN_BUCKETED",
    "GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE",
    "GPU_RUNTIME_DATA_MODE_CPU_DEBUG",
    "GPU_RUNTIME_DATA_MODE_RESIDENT",
    "GpuBatchStore",
    "GpuBatchStoreKey",
    "GpuBatchView",
    "GpuRuntimeConfig",
    "GpuRuntimeTelemetry",
    "RUNTIME_MODE_LEGACY",
    "RUNTIME_MODE_UNIFIED_V2",
    "TopologyNode",
    "TopologyPlan",
    "TopologyRef",
    "VALID_RUNTIME_MODES",
    "VALID_GPU_RUNTIME_BUCKET_MODES",
    "VALID_GPU_RUNTIME_DATA_MODES",
    "WorkBatch",
    "WorkItem",
    "build_leaf_count_auto_queue_targets",
    "build_balanced_topology_plan",
    "build_unified_topology_plan",
    "execute_batches",
    "gpu_runtime_config_from_mapping",
    "get_named_plan_cache",
    "normalize_runtime_mode",
    "normalize_gpu_runtime_bucket_mode",
    "normalize_gpu_runtime_data_mode",
    "plan_work_batches",
    "resolve_runtime_mode",
]
