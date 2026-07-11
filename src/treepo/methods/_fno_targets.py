"""Scalar and vector target extraction for neural-operator families.

Data-prep: read the supervision target(s) off each training tree, following the
config's ``target_key`` / ``target_vector_key`` preferences with sensible
fallbacks. ``_target_rows`` also enforces a uniform target width
across the batch. ``_node_supervision_targets`` reads optional per-node targets
(leaf and merge labels) aligned to the model's canonical trace order via leaf
spans, so labeled bundles supervise every node the balanced schedule produces.
No torch here — this is pure metadata shaping.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from treepo.methods._coerce import safe_float as _safe_float
from treepo.methods._fno_config import NeuralOperatorFamilyConfig
from treepo.methods._grid_axes import tree_node_units
from treepo.schedule import fold_with_trace
from treepo.tree import TreeRecord, tree_leaves, tree_root_target


def _target_rows(
    traces: Sequence[Any],
    config: NeuralOperatorFamilyConfig,
) -> tuple[list[Any], list[list[float]]]:
    rows: list[tuple[Any, list[float]]] = []
    for tree in traces:
        target = _target_vector(tree, config)
        if target is not None:
            rows.append((tree, target))
    if not rows:
        raise ValueError(
            "neural-operator families need training trees with scalar or vector "
            "target metadata ('teacher_score_native', backend_config['target_key'], "
            "or backend_config['target_vector_key'])."
        )
    width = len(rows[0][1])
    if width <= 0:
        raise ValueError("target vectors must be non-empty")
    for _tree, target in rows:
        if len(target) != width:
            raise ValueError("all target vectors must have the same length")
    return [tree for tree, _target in rows], [target for _tree, target in rows]


def _target_vector(tree: Any, config: NeuralOperatorFamilyConfig) -> list[float] | None:
    if config.target_vector_key:
        values = _vector_by_key(tree, config.target_vector_key)
        if values is None:
            return None
        if config.target_dim is not None and len(values) != int(config.target_dim):
            raise ValueError(
                f"target_vector_key={config.target_vector_key!r} produced {len(values)} values; "
                f"expected target_dim={int(config.target_dim)}"
            )
        return values
    if config.target_dim and int(config.target_dim) > 1:
        values = _vector_by_key(tree, "topic_proportions")
        if values is not None and len(values) == int(config.target_dim):
            return values
    score = _target_score(tree, config.target_key)
    return None if score is None else [float(score)]


def _vector_by_key(tree: Any, key: str | None) -> list[float] | None:
    if not key:
        return None
    meta = getattr(tree, "metadata", None)
    meta = meta if isinstance(meta, Mapping) else {}
    value = meta.get(key) if key in meta else getattr(tree, str(key), None)
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        out = [float(item) for item in value]
    except TypeError:
        return None
    return out if out else None


def _target_score(tree: Any, target_key: str | None) -> float | None:
    return tree_root_target(tree, target_key=target_key)


def _node_supervision_targets(
    trees: Sequence[Any],
    config: NeuralOperatorFamilyConfig,
    *,
    width: int,
    include_leaves: bool = True,
    include_merges: bool = True,
) -> list[tuple[list[list[float]], list[bool]]] | None:
    """Per-tree ``(targets, observed)`` node supervision in canonical trace order.

    Each tree contributes ``2L - 1`` rows following :mod:`treepo.schedule`
    (leaves ``0..L-1`` in encoder order, merges in merge order, root last).
    Leaf targets are read off the leaf objects; merge targets come from a
    tree's stored internal nodes, matched to trace positions by the contiguous
    leaf span they cover. Nodes without a consumable target are unobserved.
    The root row is always unobserved here — the root term supervises it.
    Returns ``None`` when no tree carries any observed node target.
    """

    allowed_units = _allowed_unit_set(config)
    out: list[tuple[list[list[float]], list[bool]]] = []
    any_observed = False
    for index, tree in enumerate(trees):
        rows, observed = _tree_node_targets(
            tree,
            config,
            width=int(width),
            index=index,
            allowed_units=allowed_units,
            include_leaves=bool(include_leaves),
            include_merges=bool(include_merges),
        )
        out.append((rows, observed))
        any_observed = any_observed or any(observed)
    return out if any_observed else None


def _tree_node_targets(
    tree: Any,
    config: NeuralOperatorFamilyConfig,
    *,
    width: int,
    index: int,
    allowed_units: frozenset[str] | None,
    include_leaves: bool = True,
    include_merges: bool = True,
) -> tuple[list[list[float]], list[bool]]:
    leaves = tuple(tree_leaves(tree) or ())
    leaf_count = max(1, len(leaves))
    n_nodes = 2 * leaf_count - 1
    targets = [[0.0] * int(width) for _ in range(n_nodes)]
    observed = [False] * n_nodes
    root_position = n_nodes - 1

    unit_ids = (
        tree_node_units(tree, index)
        if include_leaves and allowed_units is not None
        else None
    )
    for leaf_idx, leaf in enumerate(leaves):
        if not include_leaves:
            break
        if leaf_idx == root_position:
            # Single-leaf trees: the lone leaf IS the root; the root term owns it.
            continue
        if unit_ids is not None and (
            leaf_idx >= len(unit_ids) or unit_ids[leaf_idx] not in allowed_units
        ):
            continue
        value = _node_target_value(leaf, config, width=width)
        if value is not None:
            targets[leaf_idx] = value
            observed[leaf_idx] = True

    if include_merges:
        for position, value in _merge_targets_by_position(
            tree, config, width=width, leaves=leaves, leaf_count=leaf_count
        ):
            if position != root_position:
                targets[position] = value
                observed[position] = True
    return targets, observed


def _merge_targets_by_position(
    tree: Any,
    config: NeuralOperatorFamilyConfig,
    *,
    width: int,
    leaves: tuple[Any, ...],
    leaf_count: int,
) -> list[tuple[int, list[float]]]:
    """Match stored internal nodes to trace positions by contiguous leaf span."""

    if leaf_count < 2:
        return []
    has_nodes = bool(getattr(tree, "nodes", None)) or (
        isinstance(tree, Mapping) and tree.get("nodes")
    )
    if not has_nodes:
        return []
    record = tree if isinstance(tree, TreeRecord) else TreeRecord.from_value(tree)
    record_leaves = record.leaves()
    if len(record_leaves) != len(leaves):
        return []
    leaf_index = {str(node.node_id): idx for idx, node in enumerate(record_leaves)}
    by_id = {str(node.node_id): node for node in record.nodes}

    span_cache: dict[str, tuple[int, int] | None] = {}

    def span_of(node_id: str) -> tuple[int, int] | None:
        if node_id in span_cache:
            return span_cache[node_id]
        span_cache[node_id] = None  # cycle guard
        if node_id in leaf_index:
            idx = leaf_index[node_id]
            span_cache[node_id] = (idx, idx)
            return span_cache[node_id]
        node = by_id.get(node_id)
        if node is None or not node.has_children():
            return None
        child_spans = [
            span_of(str(child))
            for child in (node.left_child_id, node.right_child_id)
            if child is not None
        ]
        if not child_spans or any(span is None for span in child_spans):
            return None
        lo = min(span[0] for span in child_spans)  # type: ignore[index]
        hi = max(span[1] for span in child_spans)  # type: ignore[index]
        covered = sum(span[1] - span[0] + 1 for span in child_spans)  # type: ignore[index]
        if covered != hi - lo + 1:
            return None  # children overlap or leave gaps: not a contiguous span
        span_cache[node_id] = (lo, hi)
        return span_cache[node_id]

    trace_spans = fold_with_trace(
        [(idx, idx) for idx in range(leaf_count)],
        lambda left, right: (left[0], right[1]),
        schedule="balanced",
    )
    position_of_span = {
        span: position
        for position, span in enumerate(trace_spans)
        if position >= leaf_count
    }

    out: list[tuple[int, list[float]]] = []
    for node in record.nodes:
        if not node.has_children():
            continue
        span = span_of(str(node.node_id))
        if span is None:
            continue
        position = position_of_span.get(span)
        if position is None:
            continue
        value = _node_target_value(node, config, width=width)
        if value is not None:
            out.append((position, value))
    return out


def _node_target_value(
    node: Any,
    config: NeuralOperatorFamilyConfig,
    *,
    width: int,
) -> list[float] | None:
    meta = getattr(node, "metadata", None)
    meta = meta if isinstance(meta, Mapping) else {}
    candidates: list[Any] = []
    if config.node_target_key:
        candidates.append(meta.get(str(config.node_target_key)))
        if isinstance(node, Mapping):
            candidates.append(node.get(str(config.node_target_key)))
    candidates.append(getattr(node, "label", None))
    if isinstance(node, Mapping):
        candidates.extend([node.get("label"), node.get("score")])
    candidates.extend([meta.get("score"), meta.get("oracle_score")])
    for candidate in candidates:
        value = _coerce_node_target(candidate, width=width)
        if value is not None:
            return value
    return None


def _coerce_node_target(value: Any, *, width: int) -> list[float] | None:
    if value is None or isinstance(value, (str, bytes, bool, Mapping)):
        return None
    scalar = _safe_float(value)
    if scalar is not None:
        return [scalar] if int(width) == 1 else None
    try:
        vector = [_safe_float(item) for item in value]
    except TypeError:
        return None
    if len(vector) != int(width) or any(item is None for item in vector):
        return None
    return [float(item) for item in vector]  # type: ignore[arg-type]


def _allowed_unit_set(config: NeuralOperatorFamilyConfig) -> frozenset[str] | None:
    units = getattr(config, "supervised_node_units", None)
    if units is None:
        return None
    return frozenset(str(unit) for unit in units)


__all__ = [
    "_node_supervision_targets",
    "_node_target_value",
    "_target_rows",
    "_target_score",
    "_target_vector",
    "_vector_by_key",
]
