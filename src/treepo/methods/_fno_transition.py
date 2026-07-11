"""Numeric transition-state supervision for neural-operator families.

Data-prep for the optional local-law signal: derive per-node changepoint-count
transition states from numeric leaf tokens, build the pairwise merge trace, and
turn a trained model's node trace into canonical per-node law rows. Active when
a law-bearing ``ObjectiveSpec`` is configured or the legacy
``numeric_transition_state_weight`` knob is positive; otherwise inert.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from treepo.methods._fno_config import NeuralOperatorFamilyConfig
from treepo.methods._fno_encoding import _leaf_token_groups
from treepo.schedule import fold_with_trace, merge_children, merge_depths
from treepo.tree import tree_row_id


def _numeric_transition_state_targets(
    trees: Sequence[Any] | None,
    config: NeuralOperatorFamilyConfig,
    *,
    torch: Any,
    device: Any,
) -> list[Any] | None:
    if trees is None:
        return None
    out = []
    for tree in trees:
        spec = _numeric_transition_spec(tree, config)
        groups = _leaf_token_groups(tree)
        if spec is None or groups is None:
            return None
        n_states, bucket, count_scale = spec
        rows = _numeric_transition_rows(
            groups,
            n_states=n_states,
            bucket=bucket,
            count_scale=count_scale,
        )
        out.append(torch.tensor(rows, dtype=torch.float32, device=device))
    return out


def _numeric_transition_law_rows(
    traces: Sequence[Any],
    targets: Sequence[Any],
    *,
    torch: Any,
    device: Any,
    dtype: Any,
) -> tuple[Any, Any, Any] | None:
    """Return per-node ``(proxy_loss, depths, is_leaf)`` law rows for a batch.

    The model trace and the target rows are built over the same pairwise
    merge topology, so their node counts must agree exactly; a mismatch
    means the two topology definitions have drifted. The state dimension may
    legitimately differ: the first target-width dims of the learned state
    carry the supervised transition vector.

    Each tree contributes ``2L - 1`` rows in trace order (leaves first, then
    merges level by level). ``proxy_loss`` is the per-node mean squared error
    over the supervised dims, ``depths`` follows the shared merge schedule
    with the root at depth 0, and ``is_leaf`` marks the C1 channel (merge
    rows are the C3 channel) — the same channel split the audit statistic
    uses. Rows from all trees are concatenated.
    """

    if len(traces) != len(targets):
        raise ValueError(
            f"transition-state supervision got {len(traces)} traces for {len(targets)} targets"
        )
    for pred, target in zip(traces, targets):
        if int(pred.shape[0]) != int(target.shape[0]):
            raise ValueError(
                "transition-state node count mismatch: model trace has "
                f"{int(pred.shape[0])} nodes, targets have {int(target.shape[0])}"
            )

    def _node_meta(n_nodes: int) -> tuple[Any, Any]:
        # The schedule always yields 2L-1 nodes for L leaves.
        leaf_count = (int(n_nodes) + 1) // 2
        depths = torch.tensor(
            _pairwise_merge_depths(leaf_count)[: int(n_nodes)],
            dtype=torch.long,
            device=device,
        )
        is_leaf = torch.arange(int(n_nodes), device=device) < leaf_count
        return depths, is_leaf

    if traces and targets:
        first_trace_shape = tuple(int(x) for x in traces[0].shape)
        first_target_shape = tuple(int(x) for x in targets[0].shape)
        if (
            all(tuple(int(x) for x in trace.shape) == first_trace_shape for trace in traces)
            and all(tuple(int(x) for x in target.shape) == first_target_shape for target in targets)
        ):
            pred = torch.stack(list(traces), dim=0)
            target = torch.stack(
                [target.to(device=device, dtype=dtype) for target in targets],
                dim=0,
            )
            d = min(int(pred.shape[-1]), int(target.shape[-1]))
            if d <= 0:
                return None
            losses = ((pred[:, :, :d] - target[:, :, :d]) ** 2).mean(dim=-1).reshape(-1)
            n_nodes = int(pred.shape[1])
            depths, is_leaf = _node_meta(n_nodes)
            batch = int(pred.shape[0])
            return losses, depths.repeat(batch), is_leaf.repeat(batch)

    loss_chunks = []
    depth_chunks = []
    leaf_chunks = []
    for pred, target in zip(traces, targets):
        target = target.to(device=device, dtype=dtype)
        n = int(pred.shape[0])
        d = min(int(pred.shape[1]), int(target.shape[1]))
        if n <= 0 or d <= 0:
            continue
        loss_chunks.append(((pred[:n, :d] - target[:n, :d]) ** 2).mean(dim=-1))
        depths, is_leaf = _node_meta(n)
        depth_chunks.append(depths)
        leaf_chunks.append(is_leaf)
    if not loss_chunks:
        return None
    return (
        torch.cat(loss_chunks),
        torch.cat(depth_chunks),
        torch.cat(leaf_chunks),
    )


def _numeric_transition_spec(
    tree: Any,
    config: NeuralOperatorFamilyConfig,
) -> tuple[int, int, float] | None:
    meta = getattr(tree, "metadata", None) or {}
    if not isinstance(meta, Mapping):
        return None
    n_states = _optional_positive_int(meta.get("n_states"))
    vocabulary_size = _optional_positive_int(meta.get("vocabulary_size"))
    if n_states is None or vocabulary_size is None:
        return None
    bucket = max(1, int(vocabulary_size) // int(n_states))
    raw_scale = config.numeric_transition_count_scale
    if raw_scale is None:
        raw_scale = meta.get("doc_tokens") or 1.0
    count_scale = max(1.0, float(raw_scale))
    return int(n_states), int(bucket), float(count_scale)


def _numeric_transition_rows(
    groups: Sequence[Sequence[int]],
    *,
    n_states: int,
    bucket: int,
    count_scale: float,
) -> list[list[float]]:
    node_states = fold_with_trace(
        [
            _numeric_transition_leaf_state(tokens, n_states=n_states, bucket=bucket)
            for tokens in groups
        ],
        _numeric_transition_merge_state,
        schedule="balanced",
    )
    return [
        _numeric_transition_vector(state, n_states=n_states, count_scale=count_scale)
        for state in node_states
    ]


def _pairwise_merge_children(leaf_count: int) -> dict[int, tuple[int, int]]:
    """The balanced schedule's children map (see :mod:`treepo.schedule`)."""

    return merge_children(int(leaf_count), schedule="balanced")


def _pairwise_merge_depths(leaf_count: int) -> list[int]:
    """The balanced schedule's node depths (see :mod:`treepo.schedule`)."""

    return merge_depths(int(leaf_count), schedule="balanced")


def _numeric_transition_leaf_state(
    tokens: Sequence[int],
    *,
    n_states: int,
    bucket: int,
) -> tuple[float, int, int]:
    states = [
        min(max(0, int(token) // max(1, int(bucket))), int(n_states) - 1)
        for token in tokens
    ]
    if not states:
        return 0.0, 0, 0
    count = sum(1 for left, right in zip(states, states[1:]) if int(left) != int(right))
    return float(count), int(states[0]), int(states[-1])


def _numeric_transition_merge_state(
    left: tuple[float, int, int],
    right: tuple[float, int, int],
) -> tuple[float, int, int]:
    left_count, left_first, left_last = left
    right_count, right_first, right_last = right
    join = 1.0 if int(left_last) != int(right_first) else 0.0
    return float(left_count + right_count + join), int(left_first), int(right_last)


def _numeric_transition_vector(
    state: tuple[float, int, int],
    *,
    n_states: int,
    count_scale: float,
) -> list[float]:
    count, first, last = state
    vec = [float(count) / max(1.0, float(count_scale))]
    first_oh = [0.0] * int(n_states)
    last_oh = [0.0] * int(n_states)
    if 0 <= int(first) < int(n_states):
        first_oh[int(first)] = 1.0
    if 0 <= int(last) < int(n_states):
        last_oh[int(last)] = 1.0
    return vec + first_oh + last_oh


def _optional_positive_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _tree_row_id(tree: Any, idx: int) -> str:
    return tree_row_id(tree, idx, fallback_prefix="tree")


__all__ = [
    "_numeric_transition_leaf_state",
    "_pairwise_merge_children",
    "_pairwise_merge_depths",
    "_numeric_transition_merge_state",
    "_numeric_transition_rows",
    "_numeric_transition_spec",
    "_numeric_transition_law_rows",
    "_numeric_transition_state_targets",
    "_numeric_transition_vector",
    "_optional_positive_int",
    "_tree_row_id",
]
