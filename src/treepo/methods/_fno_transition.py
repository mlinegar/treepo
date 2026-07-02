"""Numeric transition-state supervision for neural-operator families.

Data-prep for the optional local-law signal: derive per-node changepoint-count
transition states from numeric leaf tokens, build the pairwise merge trace, and
score a trained model's node trace against those targets. Used only when
``numeric_transition_state_weight > 0``; otherwise inert.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from treepo.methods._fno_config import NeuralOperatorFamilyConfig
from treepo.methods._fno_encoding import _leaf_token_groups


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


def _numeric_transition_state_loss(
    traces: Sequence[Any],
    targets: Sequence[Any],
    *,
    torch: Any,
    device: Any,
    dtype: Any,
) -> Any | None:
    if traces and targets and len(traces) == len(targets):
        first_trace_shape = tuple(int(x) for x in traces[0].shape)
        first_target_shape = tuple(int(x) for x in targets[0].shape)
        if (
            first_trace_shape[0] == first_target_shape[0]
            and all(tuple(int(x) for x in trace.shape) == first_trace_shape for trace in traces)
            and all(tuple(int(x) for x in target.shape) == first_target_shape for target in targets)
        ):
            pred = torch.stack(list(traces), dim=0)
            target = torch.stack(
                [target.to(device=device, dtype=dtype) for target in targets],
                dim=0,
            )
            d = min(int(pred.shape[-1]), int(target.shape[-1]))
            if d > 0:
                return torch.nn.functional.mse_loss(pred[:, :, :d], target[:, :, :d])
    losses = []
    for pred, target in zip(traces, targets):
        target = target.to(device=device, dtype=dtype)
        n = min(int(pred.shape[0]), int(target.shape[0]))
        d = min(int(pred.shape[1]), int(target.shape[1]))
        if n <= 0 or d <= 0:
            continue
        losses.append(torch.nn.functional.mse_loss(pred[:n, :d], target[:n, :d]))
    if not losses:
        return None
    return torch.stack(losses).mean()


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
    cur = [_numeric_transition_leaf_state(tokens, n_states=n_states, bucket=bucket) for tokens in groups]
    rows = [_numeric_transition_vector(state, n_states=n_states, count_scale=count_scale) for state in cur]
    while len(cur) > 1:
        next_level = []
        for idx in range(0, len(cur) - 1, 2):
            merged = _numeric_transition_merge_state(cur[idx], cur[idx + 1])
            next_level.append(merged)
            rows.append(_numeric_transition_vector(merged, n_states=n_states, count_scale=count_scale))
        if len(cur) % 2:
            next_level.append(cur[-1])
        cur = next_level
    return rows


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
    metadata = getattr(tree, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("tree_id", "doc_id", "unit_id"):
            if metadata.get(key) is not None:
                return str(metadata[key])
    return f"tree_{idx}"


__all__ = [
    "_numeric_transition_leaf_state",
    "_numeric_transition_merge_state",
    "_numeric_transition_rows",
    "_numeric_transition_spec",
    "_numeric_transition_state_loss",
    "_numeric_transition_state_targets",
    "_numeric_transition_vector",
    "_optional_positive_int",
    "_tree_row_id",
]
