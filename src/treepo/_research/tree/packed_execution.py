"""Packed tensor execution helpers for embedding-backed CTreePO trees."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from treepo._research.tree.ctreepo_model import CTreePOModel

if TYPE_CHECKING:
    from treepo._research.tree.embedding_tree import EmbeddingTreeNode


RESIDENT_VRAM_FRACTION: float = 0.70
DENSE_BUCKET_VRAM_FRACTION: float = 0.70


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    if not isinstance(tensor, torch.Tensor):
        return 0
    return int(tensor.numel()) * int(tensor.element_size())


def _devices_match(lhs: torch.device, rhs: torch.device) -> bool:
    if lhs.type != rhs.type:
        return False
    if lhs.type != "cuda":
        return lhs == rhs
    if lhs.index is None or rhs.index is None:
        return True
    return int(lhs.index) == int(rhs.index)


def canonicalize_leaf_embedding(embedding: Any) -> torch.Tensor:
    """Return a contiguous float32 CPU tensor for a leaf embedding."""
    if isinstance(embedding, torch.Tensor):
        tensor = embedding.detach()
        if tensor.dtype != torch.float32:
            tensor = tensor.to(dtype=torch.float32)
        if tensor.device.type != "cpu":
            tensor = tensor.to(device="cpu")
        return tensor.contiguous()
    return torch.as_tensor(embedding, dtype=torch.float32).contiguous()


@dataclass
class PackedTreeLevel:
    level: int
    parent_indices_cpu: torch.Tensor
    left_indices_cpu: torch.Tensor
    right_indices_cpu: torch.Tensor
    left_weights_cpu: torch.Tensor


@dataclass
class PackedEmbeddingTree:
    nodes: Sequence["EmbeddingTreeNode"]
    node_count: int
    leaf_count: int
    root_index: int
    leaf_embeddings_cpu: torch.Tensor
    levels: Tuple[PackedTreeLevel, ...]
    level_widths: Tuple[int, ...]
    merge_level_slices: Tuple[Tuple[int, int], ...]
    merge_edge_indices_cpu: torch.Tensor
    merge_left_weights_cpu: torch.Tensor
    oracle_score_values_cpu: torch.Tensor
    oracle_score_mask_cpu: torch.Tensor
    leaf_mask_cpu: torch.Tensor
    runtime_data_mode: str = "staged"
    leaf_embeddings_staged: Optional[torch.Tensor] = None
    leaf_embeddings_resident: Optional[torch.Tensor] = None
    merge_edge_indices_staged: Optional[torch.Tensor] = None
    merge_left_weights_staged: Optional[torch.Tensor] = None
    merge_edge_indices_resident: Optional[torch.Tensor] = None
    merge_left_weights_resident: Optional[torch.Tensor] = None
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def resident_bytes(self) -> int:
        total = _tensor_nbytes(self.leaf_embeddings_cpu)
        total += _tensor_nbytes(self.merge_edge_indices_cpu)
        total += _tensor_nbytes(self.merge_left_weights_cpu)
        return int(total)


@dataclass
class PackedTreeBatchLevel:
    level: int
    parent_indices: torch.Tensor
    left_indices: torch.Tensor
    right_indices: torch.Tensor
    left_weights: torch.Tensor


@dataclass
class PackedTreeBatch:
    trees: Tuple[PackedEmbeddingTree, ...]
    node_offsets: Tuple[int, ...]
    root_indices: torch.Tensor
    leaf_node_indices: torch.Tensor
    leaf_embeddings: torch.Tensor
    levels: Tuple[PackedTreeBatchLevel, ...]
    total_nodes: int
    total_leaves: int
    shared_leaf_count: int
    fixed_fused_eligible: bool
    runtime_stats: Dict[str, Any]


@dataclass
class PackedForwardResult:
    packed_batch: PackedTreeBatch
    state_batch: torch.Tensor
    root_indices: torch.Tensor
    node_offsets: Tuple[int, ...]
    runtime_stats: Dict[str, Any]

    def global_index(self, doc_index: int, node_index: int) -> int:
        return int(self.node_offsets[int(doc_index)]) + int(node_index)


@dataclass
class PackedTreeBucketStore:
    trees: Tuple[PackedEmbeddingTree, ...]
    row_by_nodes_id: Mapping[int, int]
    leaf_rows: Tuple[torch.Tensor, ...]
    leaf_dense_resident: Optional[torch.Tensor]
    leaf_count: int
    node_count: int
    root_index: int
    level_widths: Tuple[int, ...]
    merge_level_slices: Tuple[Tuple[int, int], ...]
    local_merge_edge_indices: torch.Tensor
    local_merge_left_weights: torch.Tensor
    runtime_data_mode: str
    bucket_store_mode: str
    resident_dense_bytes: int = 0
    structure_cache: Dict[Tuple[str, int], Dict[str, Any]] = field(default_factory=dict)


def _move_runtime_tensor_to_device(
    tensor: torch.Tensor,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, int, int]:
    if _devices_match(tensor.device, device):
        return tensor, 0, 0
    pinned = bool(tensor.device.type == "cpu" and tensor.is_pinned())
    moved = tensor.to(device=device, non_blocking=pinned)
    bytes_transferred = int(_tensor_nbytes(tensor))
    return moved, bytes_transferred, (1 if bytes_transferred > 0 else 0)


def build_packed_embedding_tree(
    nodes: Sequence["EmbeddingTreeNode"],
) -> PackedEmbeddingTree:
    """Compile a node list into a packed tensor representation."""
    leaf_embeddings: List[torch.Tensor] = []
    levels: Dict[int, List[int]] = {}
    oracle_values: List[float] = []
    oracle_mask: List[bool] = []
    leaf_mask: List[bool] = []

    for node_index, node in enumerate(nodes):
        is_leaf = bool(getattr(node, "is_leaf", False))
        leaf_mask.append(is_leaf)
        score_map = dict(getattr(node, "oracle_scores", {}) or {})
        if "rile" in score_map:
            oracle_values.append(float(score_map["rile"]))
            oracle_mask.append(True)
        else:
            oracle_values.append(float("nan"))
            oracle_mask.append(False)

        if is_leaf:
            if getattr(node, "embedding", None) is None:
                raise ValueError(f"leaf node {node_index} missing embedding")
            canonical = canonicalize_leaf_embedding(getattr(node, "embedding"))
            setattr(node, "embedding", canonical)
            leaf_embeddings.append(canonical)
        else:
            levels.setdefault(int(getattr(node, "level", 0)), []).append(node_index)

    if leaf_embeddings:
        leaf_embeddings_cpu = torch.stack(leaf_embeddings, dim=0).contiguous()
    else:
        leaf_embeddings_cpu = torch.empty((0, 0), dtype=torch.float32)

    packed_levels: List[PackedTreeLevel] = []
    level_widths: List[int] = []
    merge_level_slices: List[Tuple[int, int]] = []
    merge_edge_parts: List[torch.Tensor] = []
    merge_weight_parts: List[torch.Tensor] = []
    merge_cursor = 0
    for level in sorted(levels):
        parent_indices: List[int] = []
        left_indices: List[int] = []
        right_indices: List[int] = []
        left_weights: List[float] = []
        for node_index in levels[level]:
            left_idx, right_idx = getattr(nodes[node_index], "children")
            left_node = nodes[left_idx]
            right_node = nodes[right_idx]
            left_len = int(getattr(left_node, "text_len"))
            right_len = int(getattr(right_node, "text_len"))
            denom = max(left_len + right_len, 1)
            parent_indices.append(int(node_index))
            left_indices.append(int(left_idx))
            right_indices.append(int(right_idx))
            left_weights.append(float(left_len) / float(denom))
        packed_levels.append(
            PackedTreeLevel(
                level=int(level),
                parent_indices_cpu=torch.tensor(parent_indices, dtype=torch.long),
                left_indices_cpu=torch.tensor(left_indices, dtype=torch.long),
                right_indices_cpu=torch.tensor(right_indices, dtype=torch.long),
                left_weights_cpu=torch.tensor(left_weights, dtype=torch.float32),
            )
        )
        level_widths.append(int(len(parent_indices)))
        width = int(len(parent_indices))
        merge_level_slices.append((int(merge_cursor), int(merge_cursor + width)))
        merge_cursor += width
        if width > 0:
            merge_edge_parts.append(
                torch.stack(
                    (
                        packed_levels[-1].parent_indices_cpu,
                        packed_levels[-1].left_indices_cpu,
                        packed_levels[-1].right_indices_cpu,
                    ),
                    dim=0,
                ).contiguous()
            )
            merge_weight_parts.append(packed_levels[-1].left_weights_cpu.contiguous())

    staged_leaf = leaf_embeddings_cpu.contiguous()
    merge_edge_indices_cpu = (
        torch.cat(merge_edge_parts, dim=1).contiguous()
        if merge_edge_parts
        else torch.empty((3, 0), dtype=torch.long)
    )
    merge_left_weights_cpu = (
        torch.cat(merge_weight_parts, dim=0).contiguous()
        if merge_weight_parts
        else torch.empty((0,), dtype=torch.float32)
    )
    return PackedEmbeddingTree(
        nodes=tuple(nodes),
        node_count=int(len(nodes)),
        leaf_count=int(leaf_embeddings_cpu.shape[0]),
        root_index=int(len(nodes) - 1 if nodes else -1),
        leaf_embeddings_cpu=leaf_embeddings_cpu,
        levels=tuple(packed_levels),
        level_widths=tuple(level_widths),
        merge_level_slices=tuple(merge_level_slices),
        merge_edge_indices_cpu=merge_edge_indices_cpu,
        merge_left_weights_cpu=merge_left_weights_cpu,
        oracle_score_values_cpu=torch.tensor(oracle_values, dtype=torch.float32),
        oracle_score_mask_cpu=torch.tensor(oracle_mask, dtype=torch.bool),
        leaf_mask_cpu=torch.tensor(leaf_mask, dtype=torch.bool),
        leaf_embeddings_staged=staged_leaf,
        merge_edge_indices_staged=merge_edge_indices_cpu.contiguous(),
        merge_left_weights_staged=merge_left_weights_cpu.contiguous(),
    )


def apply_runtime_mode_to_packed_trees(
    packed_trees: Sequence[PackedEmbeddingTree],
    *,
    device: torch.device,
    resident_fraction: float = RESIDENT_VRAM_FRACTION,
) -> Dict[str, Any]:
    """Populate staged or resident tensors for a split."""
    summary: Dict[str, Any] = {
        "runtime_data_mode": "staged",
        "resident_store_hits": 0,
        "resident_store_misses": int(len(packed_trees)),
        "resident_store_bytes": 0,
        "resident_store_build_time_s": 0.0,
    }
    for tree in packed_trees:
        tree.runtime_data_mode = "staged"
        tree.leaf_embeddings_resident = None
        tree.merge_edge_indices_resident = None
        tree.merge_left_weights_resident = None
        tree.runtime_metadata = {"runtime_data_mode": "staged"}
        if device.type == "cuda" and torch.cuda.is_available():
            staged = tree.leaf_embeddings_cpu
            if not staged.is_pinned():
                try:
                    staged = staged.pin_memory()
                except RuntimeError:
                    staged = staged.contiguous()
            tree.leaf_embeddings_staged = staged
            staged_edges = tree.merge_edge_indices_cpu
            if not staged_edges.is_pinned():
                try:
                    staged_edges = staged_edges.pin_memory()
                except RuntimeError:
                    staged_edges = staged_edges.contiguous()
            tree.merge_edge_indices_staged = staged_edges
            staged_weights = tree.merge_left_weights_cpu
            if not staged_weights.is_pinned():
                try:
                    staged_weights = staged_weights.pin_memory()
                except RuntimeError:
                    staged_weights = staged_weights.contiguous()
            tree.merge_left_weights_staged = staged_weights
        else:
            tree.leaf_embeddings_staged = tree.leaf_embeddings_cpu.contiguous()
            tree.merge_edge_indices_staged = tree.merge_edge_indices_cpu.contiguous()
            tree.merge_left_weights_staged = tree.merge_left_weights_cpu.contiguous()

    if not packed_trees or device.type != "cuda" or not torch.cuda.is_available():
        return summary

    device_index = device.index
    if device_index is None:
        device_index = int(torch.cuda.current_device())
    total_bytes = sum(int(tree.resident_bytes) for tree in packed_trees)
    try:
        with torch.cuda.device(device_index):
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
    except RuntimeError:
        return summary

    threshold = int(float(max(0.0, resident_fraction)) * float(free_bytes))
    if total_bytes > threshold:
        summary["resident_store_bytes"] = int(total_bytes)
        return summary

    started = time.perf_counter()
    try:
        for tree in packed_trees:
            source = tree.leaf_embeddings_staged
            if source is None:
                source = tree.leaf_embeddings_cpu
            tree.leaf_embeddings_resident = source.to(
                device=device,
                non_blocking=bool(source.is_pinned()),
            )
            merge_edges = tree.merge_edge_indices_staged
            if merge_edges is None:
                merge_edges = tree.merge_edge_indices_cpu
            tree.merge_edge_indices_resident = merge_edges.to(
                device=device,
                non_blocking=bool(merge_edges.is_pinned()),
            )
            merge_weights = tree.merge_left_weights_staged
            if merge_weights is None:
                merge_weights = tree.merge_left_weights_cpu
            tree.merge_left_weights_resident = merge_weights.to(
                device=device,
                non_blocking=bool(merge_weights.is_pinned()),
            )
            tree.runtime_data_mode = "resident"
            tree.runtime_metadata = {"runtime_data_mode": "resident"}
    except RuntimeError:
        for tree in packed_trees:
            tree.runtime_data_mode = "staged"
            tree.leaf_embeddings_resident = None
            tree.merge_edge_indices_resident = None
            tree.merge_left_weights_resident = None
            tree.runtime_metadata = {"runtime_data_mode": "staged"}
        return summary

    summary.update(
        {
            "runtime_data_mode": "resident",
            "resident_store_hits": int(len(packed_trees)),
            "resident_store_misses": 0,
            "resident_store_bytes": int(total_bytes),
            "resident_store_build_time_s": float(time.perf_counter() - started),
        }
    )
    return summary


def build_packed_tree_bucket_stores(
    packed_trees: Sequence[PackedEmbeddingTree],
    *,
    device: torch.device,
) -> Tuple[PackedTreeBucketStore, ...]:
    grouped: Dict[Tuple[int, int, int, Tuple[int, ...]], List[PackedEmbeddingTree]] = {}
    for tree in packed_trees:
        signature = (
            int(tree.leaf_count),
            int(tree.node_count),
            int(tree.root_index),
            tuple(int(width) for width in tree.level_widths),
        )
        grouped.setdefault(signature, []).append(tree)

    stores: List[PackedTreeBucketStore] = []
    dense_free_bytes: Optional[int] = None
    if device.type == "cuda" and torch.cuda.is_available():
        device_index = device.index
        if device_index is None:
            device_index = int(torch.cuda.current_device())
        try:
            with torch.cuda.device(device_index):
                dense_free_bytes, _total_bytes = torch.cuda.mem_get_info()
        except RuntimeError:
            dense_free_bytes = None
    for _signature, group in sorted(grouped.items(), key=lambda item: item[0]):
        if not group:
            continue
        reference = group[0]
        can_use_resident = bool(
            device.type == "cuda"
            and all(
                str(tree.runtime_data_mode) == "resident"
                and tree.leaf_embeddings_resident is not None
                for tree in group
            )
        )
        leaf_dense_resident: Optional[torch.Tensor] = None
        resident_dense_bytes = 0
        if can_use_resident:
            leaf_rows = tuple(tree.leaf_embeddings_resident for tree in group if tree.leaf_embeddings_resident is not None)
            local_merge_edge_indices = (
                reference.merge_edge_indices_resident
                if reference.merge_edge_indices_resident is not None
                else reference.merge_edge_indices_cpu
            )
            local_merge_left_weights = (
                reference.merge_left_weights_resident
                if reference.merge_left_weights_resident is not None
                else reference.merge_left_weights_cpu
            )
            runtime_data_mode = "resident"
            bucket_store_mode = "resident_rows"
            dense_candidate_bytes = int(len(group)) * int(reference.leaf_count) * int(reference.leaf_embeddings_cpu.shape[-1]) * int(reference.leaf_embeddings_cpu.element_size())
            dense_threshold = (
                int(float(max(0.0, DENSE_BUCKET_VRAM_FRACTION)) * float(dense_free_bytes))
                if dense_free_bytes is not None
                else -1
            )
            if dense_free_bytes is not None and dense_candidate_bytes <= max(0, dense_threshold):
                try:
                    leaf_dense_resident = torch.stack(list(leaf_rows), dim=0).contiguous()
                    resident_dense_bytes = int(_tensor_nbytes(leaf_dense_resident))
                    dense_free_bytes = max(0, int(dense_free_bytes) - resident_dense_bytes)
                    bucket_store_mode = "dense_resident"
                except RuntimeError:
                    leaf_dense_resident = None
                    resident_dense_bytes = 0
        else:
            leaf_rows = tuple(
                tree.leaf_embeddings_staged
                if tree.leaf_embeddings_staged is not None
                else tree.leaf_embeddings_cpu
                for tree in group
            )
            local_merge_edge_indices = (
                reference.merge_edge_indices_staged
                if reference.merge_edge_indices_staged is not None
                else reference.merge_edge_indices_cpu
            )
            local_merge_left_weights = (
                reference.merge_left_weights_staged
                if reference.merge_left_weights_staged is not None
                else reference.merge_left_weights_cpu
            )
            runtime_data_mode = "staged"
            bucket_store_mode = "staged_rows"

        stores.append(
            PackedTreeBucketStore(
                trees=tuple(group),
                row_by_nodes_id={
                    int(id(tree.nodes)): int(row_index)
                    for row_index, tree in enumerate(group)
                },
                leaf_rows=leaf_rows,
                leaf_dense_resident=leaf_dense_resident,
                leaf_count=int(reference.leaf_count),
                node_count=int(reference.node_count),
                root_index=int(reference.root_index),
                level_widths=tuple(int(width) for width in reference.level_widths),
                merge_level_slices=tuple(reference.merge_level_slices),
                local_merge_edge_indices=local_merge_edge_indices,
                local_merge_left_weights=local_merge_left_weights,
                runtime_data_mode=str(runtime_data_mode),
                bucket_store_mode=str(bucket_store_mode),
                resident_dense_bytes=int(resident_dense_bytes),
            )
        )
    return tuple(stores)


def _fixed_fused_structure_for_bucket_store(
    store: PackedTreeBucketStore,
    *,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    cache_key = (str(device), int(batch_size))
    cached = store.structure_cache.get(cache_key)
    if cached is not None:
        return cached

    root_indices = (
        torch.arange(int(batch_size), dtype=torch.long, device=device) * int(store.node_count)
        + int(store.root_index)
    )
    leaf_offsets = torch.arange(
        int(batch_size),
        dtype=torch.long,
        device=device,
    ) * int(store.node_count)
    leaf_node_indices = (
        torch.arange(
            int(store.leaf_count),
            dtype=torch.long,
            device=device,
        )
        .unsqueeze(0)
        .repeat(int(batch_size), 1)
        + leaf_offsets.unsqueeze(1)
    ).reshape(int(batch_size) * int(store.leaf_count))
    local_edges = store.local_merge_edge_indices
    local_weights = store.local_merge_left_weights
    if not _devices_match(local_edges.device, device):
        local_edges, _bytes, _events = _move_runtime_tensor_to_device(local_edges, device=device)
    if not _devices_match(local_weights.device, device):
        local_weights, _bytes, _events = _move_runtime_tensor_to_device(local_weights, device=device)

    levels: List[PackedTreeBatchLevel] = []
    offsets = torch.arange(int(batch_size), dtype=torch.long, device=device) * int(store.node_count)
    for level_index, (start, stop) in enumerate(store.merge_level_slices):
        width = int(stop - start)
        if width <= 0:
            continue
        local_level_edges = local_edges[:, int(start):int(stop)]
        repeated_edges = (
            local_level_edges.unsqueeze(0).repeat(int(batch_size), 1, 1)
            + offsets.view(int(batch_size), 1, 1)
        )
        global_edges = repeated_edges.permute(1, 0, 2).reshape(3, int(batch_size) * width).contiguous()
        global_weights = (
            local_weights[int(start):int(stop)]
            .unsqueeze(0)
            .repeat(int(batch_size), 1)
            .reshape(int(batch_size) * width)
            .contiguous()
        )
        levels.append(
            PackedTreeBatchLevel(
                level=int(level_index + 1),
                parent_indices=global_edges[0],
                left_indices=global_edges[1],
                right_indices=global_edges[2],
                left_weights=global_weights,
            )
        )

    structure = {
        "root_indices": root_indices,
        "leaf_node_indices": leaf_node_indices,
        "levels": tuple(levels),
    }
    store.structure_cache[cache_key] = structure
    return structure


def build_packed_tree_batch_from_bucket_store(
    store: PackedTreeBucketStore,
    selected_trees: Sequence[PackedEmbeddingTree],
    *,
    device: torch.device,
) -> PackedTreeBatch:
    tree_list = tuple(selected_trees)
    if not tree_list:
        return build_packed_tree_batch(tree_list, device=device)

    row_indices = [
        int(store.row_by_nodes_id[int(id(tree.nodes))])
        for tree in tree_list
    ]
    row_index_tensor = torch.tensor(row_indices, dtype=torch.long, device=device)
    if store.leaf_dense_resident is not None and _devices_match(store.leaf_dense_resident.device, device):
        leaf_dense = store.leaf_dense_resident.index_select(0, row_index_tensor)
    else:
        leaf_rows = [store.leaf_rows[row_index] for row_index in row_indices]
        if len(leaf_rows) == 1:
            leaf_dense = leaf_rows[0].unsqueeze(0)
        else:
            leaf_dense = torch.stack(list(leaf_rows), dim=0).contiguous()

    host_to_device_bytes = 0
    host_to_device_events = 0
    runtime_data_mode = str(store.runtime_data_mode)
    if leaf_dense.device.type == "cpu" and device.type == "cuda":
        if not leaf_dense.is_pinned():
            try:
                leaf_dense = leaf_dense.pin_memory()
            except RuntimeError:
                leaf_dense = leaf_dense.contiguous()
        leaf_dense, host_to_device_bytes, host_to_device_events = _move_runtime_tensor_to_device(
            leaf_dense,
            device=device,
        )
    elif not _devices_match(leaf_dense.device, device):
        leaf_dense, host_to_device_bytes, host_to_device_events = _move_runtime_tensor_to_device(
            leaf_dense,
            device=device,
        )

    structure = _fixed_fused_structure_for_bucket_store(
        store,
        batch_size=int(len(tree_list)),
        device=device,
    )
    node_offsets = tuple(int(idx * store.node_count) for idx in range(len(tree_list)))
    return PackedTreeBatch(
        trees=tree_list,
        node_offsets=node_offsets,
        root_indices=structure["root_indices"],
        leaf_node_indices=structure["leaf_node_indices"],
        leaf_embeddings=leaf_dense.reshape(
            int(len(tree_list)) * int(store.leaf_count),
            int(leaf_dense.shape[-1]),
        ),
        levels=structure["levels"],
        total_nodes=int(len(tree_list)) * int(store.node_count),
        total_leaves=int(len(tree_list)) * int(store.leaf_count),
        shared_leaf_count=int(store.leaf_count),
        fixed_fused_eligible=bool(store.level_widths),
        runtime_stats={
            "runtime_data_mode": str(runtime_data_mode),
            "host_to_device_bytes": int(host_to_device_bytes),
            "host_to_device_events": int(host_to_device_events),
            "packed_executor_mode": "fixed_fused",
            "resident_store_hits": int(len(tree_list)) if runtime_data_mode == "resident" else 0,
            "resident_store_misses": 0 if runtime_data_mode == "resident" else int(len(tree_list)),
            "materialized_node_sketch_count": 0,
            "packed_bucket_store_hit": True,
            "packed_bucket_store_mode": str(store.bucket_store_mode),
        },
    )


def build_packed_tree_batch(
    packed_trees: Sequence[PackedEmbeddingTree],
    *,
    device: torch.device,
) -> PackedTreeBatch:
    """Assemble a batch of packed trees on the target device."""
    tree_list = tuple(packed_trees)
    if not tree_list:
        empty_leaf = torch.empty((0, 0), dtype=torch.float32, device=device)
        empty_index = torch.empty((0,), dtype=torch.long, device=device)
        return PackedTreeBatch(
            trees=tuple(),
            node_offsets=tuple(),
            root_indices=empty_index,
            leaf_node_indices=empty_index,
            leaf_embeddings=empty_leaf,
            levels=tuple(),
            total_nodes=0,
            total_leaves=0,
            shared_leaf_count=0,
            fixed_fused_eligible=False,
            runtime_stats={
                "runtime_data_mode": "staged",
                "host_to_device_bytes": 0,
                "host_to_device_events": 0,
                "packed_executor_mode": "generic_packed",
                "resident_store_hits": 0,
                "resident_store_misses": 0,
                "materialized_node_sketch_count": 0,
            },
        )

    node_offsets: List[int] = []
    root_indices_list: List[int] = []
    leaf_node_indices_parts: List[torch.Tensor] = []
    total_nodes = 0
    total_leaves = 0
    shared_leaf_count = int(tree_list[0].leaf_count)
    shared_level_widths = tuple(tree_list[0].level_widths)
    resident_hits = 0

    for tree in tree_list:
        node_offsets.append(int(total_nodes))
        root_indices_list.append(int(total_nodes + tree.root_index))
        leaf_node_indices_parts.append(
            torch.arange(
                int(total_nodes),
                int(total_nodes + tree.leaf_count),
                dtype=torch.long,
                device=device,
            )
        )
        total_nodes += int(tree.node_count)
        total_leaves += int(tree.leaf_count)
        if (
            int(tree.leaf_count) != int(shared_leaf_count)
            or tuple(tree.level_widths) != shared_level_widths
        ):
            shared_leaf_count = 0
            shared_level_widths = tuple()
        if (
            str(tree.runtime_data_mode) == "resident"
            and tree.leaf_embeddings_resident is not None
            and tree.merge_edge_indices_resident is not None
            and tree.merge_left_weights_resident is not None
        ):
            resident_hits += 1

    all_resident = resident_hits == int(len(tree_list)) and device.type == "cuda"
    host_to_device_bytes = 0
    host_to_device_events = 0
    if all_resident:
        leaf_parts = [tree.leaf_embeddings_resident for tree in tree_list]
        runtime_data_mode = "resident"
    else:
        leaf_parts = [
            tree.leaf_embeddings_staged
            if tree.leaf_embeddings_staged is not None
            else tree.leaf_embeddings_cpu
            for tree in tree_list
        ]
        runtime_data_mode = "staged"

    if len(leaf_parts) == 1:
        leaf_embeddings = leaf_parts[0]
    else:
        leaf_embeddings = torch.cat(list(leaf_parts), dim=0).contiguous()
    if leaf_embeddings.device.type == "cpu" and device.type == "cuda" and not leaf_embeddings.is_pinned():
        try:
            leaf_embeddings = leaf_embeddings.pin_memory()
        except RuntimeError:
            leaf_embeddings = leaf_embeddings.contiguous()
    leaf_embeddings, moved_bytes, moved_events = _move_runtime_tensor_to_device(
        leaf_embeddings,
        device=device,
    )
    host_to_device_bytes += int(moved_bytes)
    host_to_device_events += int(moved_events)

    levels: List[PackedTreeBatchLevel] = []
    max_levels = max((len(tree.levels) for tree in tree_list), default=0)
    level_edge_parts: List[torch.Tensor] = []
    level_weight_parts: List[torch.Tensor] = []
    level_sizes: List[int] = []
    for level_index in range(max_levels):
        edge_parts: List[torch.Tensor] = []
        weight_parts: List[torch.Tensor] = []
        for tree_offset, tree in zip(node_offsets, tree_list):
            if level_index >= len(tree.merge_level_slices):
                continue
            edge_source = (
                tree.merge_edge_indices_resident
                if all_resident and tree.merge_edge_indices_resident is not None
                else tree.merge_edge_indices_staged
                if tree.merge_edge_indices_staged is not None
                else tree.merge_edge_indices_cpu
            )
            weight_source = (
                tree.merge_left_weights_resident
                if all_resident and tree.merge_left_weights_resident is not None
                else tree.merge_left_weights_staged
                if tree.merge_left_weights_staged is not None
                else tree.merge_left_weights_cpu
            )
            start, stop = tree.merge_level_slices[level_index]
            if int(stop - start) <= 0:
                continue
            level_edges = edge_source[:, int(start):int(stop)]
            if int(tree_offset) != 0:
                offset = level_edges.new_full((1, int(level_edges.shape[1])), int(tree_offset))
                level_edges = level_edges + offset
            edge_parts.append(level_edges)
            weight_parts.append(weight_source[int(start):int(stop)])
        if not edge_parts:
            continue
        level_edges = (
            edge_parts[0] if len(edge_parts) == 1 else torch.cat(edge_parts, dim=1).contiguous()
        )
        level_weights = (
            weight_parts[0] if len(weight_parts) == 1 else torch.cat(weight_parts, dim=0).contiguous()
        )
        level_edge_parts.append(level_edges)
        level_weight_parts.append(level_weights)
        level_sizes.append(int(level_weights.shape[0]))

    if level_edge_parts:
        merge_edge_indices = (
            level_edge_parts[0]
            if len(level_edge_parts) == 1
            else torch.cat(level_edge_parts, dim=1).contiguous()
        )
        merge_left_weights = (
            level_weight_parts[0]
            if len(level_weight_parts) == 1
            else torch.cat(level_weight_parts, dim=0).contiguous()
        )
        if merge_edge_indices.device.type == "cpu" and device.type == "cuda" and not merge_edge_indices.is_pinned():
            try:
                merge_edge_indices = merge_edge_indices.pin_memory()
            except RuntimeError:
                merge_edge_indices = merge_edge_indices.contiguous()
        if merge_left_weights.device.type == "cpu" and device.type == "cuda" and not merge_left_weights.is_pinned():
            try:
                merge_left_weights = merge_left_weights.pin_memory()
            except RuntimeError:
                merge_left_weights = merge_left_weights.contiguous()
        merge_edge_indices, moved_bytes, moved_events = _move_runtime_tensor_to_device(
            merge_edge_indices,
            device=device,
        )
        host_to_device_bytes += int(moved_bytes)
        host_to_device_events += int(moved_events)
        merge_left_weights, moved_bytes, moved_events = _move_runtime_tensor_to_device(
            merge_left_weights,
            device=device,
        )
        host_to_device_bytes += int(moved_bytes)
        host_to_device_events += int(moved_events)

        cursor = 0
        for level_index, level_size in enumerate(level_sizes):
            next_cursor = int(cursor + level_size)
            levels.append(
                PackedTreeBatchLevel(
                    level=int(level_index + 1),
                    parent_indices=merge_edge_indices[0, int(cursor):int(next_cursor)],
                    left_indices=merge_edge_indices[1, int(cursor):int(next_cursor)],
                    right_indices=merge_edge_indices[2, int(cursor):int(next_cursor)],
                    left_weights=merge_left_weights[int(cursor):int(next_cursor)],
                )
            )
            cursor = next_cursor

    fixed_fused_eligible = bool(shared_leaf_count > 0 and shared_level_widths)
    return PackedTreeBatch(
        trees=tree_list,
        node_offsets=tuple(node_offsets),
        root_indices=torch.tensor(root_indices_list, dtype=torch.long, device=device),
        leaf_node_indices=(
            torch.cat(leaf_node_indices_parts, dim=0)
            if leaf_node_indices_parts
            else torch.empty((0,), dtype=torch.long, device=device)
        ),
        leaf_embeddings=leaf_embeddings,
        levels=tuple(levels),
        total_nodes=int(total_nodes),
        total_leaves=int(total_leaves),
        shared_leaf_count=int(shared_leaf_count),
        fixed_fused_eligible=bool(fixed_fused_eligible),
        runtime_stats={
            "runtime_data_mode": str(runtime_data_mode),
            "host_to_device_bytes": int(host_to_device_bytes),
            "host_to_device_events": int(host_to_device_events),
            "packed_executor_mode": "fixed_fused"
            if fixed_fused_eligible
            else "generic_packed",
            "resident_store_hits": int(resident_hits),
            "resident_store_misses": int(len(tree_list) - resident_hits),
            "materialized_node_sketch_count": 0,
        },
    )


def _encode_leaf_embeddings(
    model: CTreePOModel,
    leaf_embeddings: torch.Tensor,
    *,
    max_batch_leaves: int,
) -> torch.Tensor:
    if int(leaf_embeddings.shape[0]) <= 0:
        return leaf_embeddings.new_empty(
            (0, int(getattr(model, "state_dim", 0))),
            dtype=leaf_embeddings.dtype,
        )
    chunk = max(1, int(max_batch_leaves))
    outputs: List[torch.Tensor] = []
    for start in range(0, int(leaf_embeddings.shape[0]), chunk):
        stop = min(int(leaf_embeddings.shape[0]), int(start + chunk))
        outputs.append(model.encode_leaf_batch(leaf_embeddings[start:stop]))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


def materialize_packed_forward_result(
    result: PackedForwardResult,
) -> int:
    """Write packed node states back onto node objects."""
    count = 0
    state_batch = result.state_batch
    for tree_index, tree in enumerate(result.packed_batch.trees):
        node_offset = int(result.node_offsets[tree_index])
        for node_index, node in enumerate(tree.nodes):
            node.sketch = state_batch[int(node_offset + node_index)]
            count += 1
    result.runtime_stats["materialized_node_sketch_count"] = int(count)
    result.packed_batch.runtime_stats["materialized_node_sketch_count"] = int(count)
    return int(count)


def forward_packed_tree_batch(
    model: CTreePOModel,
    packed_batch: PackedTreeBatch,
    *,
    max_batch_leaves: int = 8192,
    materialize_nodes: bool = False,
) -> PackedForwardResult:
    """Run a packed batch through the embedding-backed CTreePO model."""
    if packed_batch.total_nodes <= 0:
        result = PackedForwardResult(
            packed_batch=packed_batch,
            state_batch=packed_batch.leaf_embeddings.new_empty(
                (0, int(getattr(model, "state_dim", 0)))
            ),
            root_indices=packed_batch.root_indices,
            node_offsets=packed_batch.node_offsets,
            runtime_stats=dict(packed_batch.runtime_stats),
        )
        if materialize_nodes:
            materialize_packed_forward_result(result)
        return result

    flat_leaf_states = _encode_leaf_embeddings(
        model,
        packed_batch.leaf_embeddings,
        max_batch_leaves=max_batch_leaves,
    )
    state_dim = int(flat_leaf_states.shape[-1]) if flat_leaf_states.ndim == 2 else 0
    state_batch = flat_leaf_states.new_empty((int(packed_batch.total_nodes), state_dim))
    if int(flat_leaf_states.shape[0]) > 0:
        state_batch.index_copy_(0, packed_batch.leaf_node_indices, flat_leaf_states)

    if packed_batch.fixed_fused_eligible:
        batch_size = int(len(packed_batch.trees))
        current = flat_leaf_states.reshape(
            batch_size,
            int(packed_batch.shared_leaf_count),
            state_dim,
        )
        for level in packed_batch.levels:
            if int(current.shape[1]) % 2 == 1:
                current = torch.cat([current, current[:, -1:, :]], dim=1)
            left = current[:, 0::2, :]
            right = current[:, 1::2, :]
            merged = model.merge_batch(
                left.reshape(-1, state_dim),
                right.reshape(-1, state_dim),
            ).reshape(batch_size, -1, state_dim)
            state_batch.index_copy_(0, level.parent_indices, merged.reshape(-1, state_dim))
            current = merged
        executor_mode = "fixed_fused"
    else:
        for level in packed_batch.levels:
            merged = model.merge_batch(
                state_batch.index_select(0, level.left_indices),
                state_batch.index_select(0, level.right_indices),
            )
            state_batch.index_copy_(0, level.parent_indices, merged)
        executor_mode = "generic_packed"

    runtime_stats = dict(packed_batch.runtime_stats)
    runtime_stats["packed_executor_mode"] = str(executor_mode)
    result = PackedForwardResult(
        packed_batch=packed_batch,
        state_batch=state_batch,
        root_indices=packed_batch.root_indices,
        node_offsets=packed_batch.node_offsets,
        runtime_stats=runtime_stats,
    )
    if materialize_nodes:
        materialize_packed_forward_result(result)
    return result
