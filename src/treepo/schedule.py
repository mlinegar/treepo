"""The canonical merge schedule: one definition of tree topology.

Every family folds leaves into a root through a merge schedule. This module
is the single source of truth for what each schedule *means*: which children
every merge consumes, in what order merges happen, and how deep each node
sits. Consumers either fold values directly (:func:`fold`,
:func:`fold_with_trace`) or read the index bookkeeping
(:func:`merge_order`, :func:`merge_children`, :func:`merge_depths`) to label
externally-produced traces.

Node indexing convention (trace order): leaves are ``0..L-1`` in position
order; merge ``k`` creates node ``L + k``; the final node ``2L - 2`` is the
root. Depths hang off real parent edges with the root at depth 0, so a node
carried past a level sits one level below the merge that finally consumes it.

Tensor fast paths (e.g. the FNO model's level-vectorized compose) may keep
their own batched loops for throughput, but must produce traces in exactly
this order — pin that with a parity test against :func:`merge_children`
rather than re-deriving the topology.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from treepo.common import VALID_SCHEDULES, ScheduleName


def _validate_schedule(schedule: str) -> str:
    name = str(schedule)
    if name not in VALID_SCHEDULES:
        raise ValueError(
            f"unsupported schedule: {schedule!r}; expected one of {VALID_SCHEDULES}"
        )
    return name


def merge_order(
    leaf_count: int,
    *,
    schedule: ScheduleName = "balanced",
) -> list[tuple[int, int]]:
    """Return each merge's ``(left, right)`` children in merge order.

    Merge ``k`` of the returned list creates node ``leaf_count + k``.
    ``balanced`` merges adjacent pairs level by level, carrying an odd
    leftover to the end of the next level. ``left_to_right`` and
    ``right_to_left`` accumulate sequentially with the running state as the
    left child.
    """

    name = _validate_schedule(schedule)
    n = int(leaf_count)
    if n <= 0:
        return []
    order: list[tuple[int, int]] = []
    next_index = n
    if name == "balanced":
        current = list(range(n))
        while len(current) > 1:
            next_level: list[int] = []
            for idx in range(0, len(current) - 1, 2):
                order.append((current[idx], current[idx + 1]))
                next_level.append(next_index)
                next_index += 1
            if len(current) % 2:
                next_level.append(current[-1])
            current = next_level
        return order
    leaves = list(range(n)) if name == "left_to_right" else list(range(n - 1, -1, -1))
    acc = leaves[0]
    for leaf in leaves[1:]:
        order.append((acc, leaf))
        acc = next_index
        next_index += 1
    return order


def merge_children(
    leaf_count: int,
    *,
    schedule: ScheduleName = "balanced",
) -> dict[int, tuple[int, int]]:
    """Return each merge node's ``(left, right)`` children by trace index."""

    n = int(leaf_count)
    return {
        n + k: pair
        for k, pair in enumerate(merge_order(n, schedule=schedule))
    }


def merge_depths(
    leaf_count: int,
    *,
    schedule: ScheduleName = "balanced",
) -> list[int]:
    """Return the depth of every node in trace order, root at depth 0."""

    n = int(leaf_count)
    if n <= 0:
        return []
    children = merge_children(n, schedule=schedule)
    total = n + len(children)
    parents: list[int | None] = [None] * total
    for node, (left, right) in children.items():
        parents[left] = node
        parents[right] = node
    depths = [0] * total
    for node in range(total - 2, -1, -1):
        parent = parents[node]
        depths[node] = 0 if parent is None else depths[parent] + 1
    return depths


def fold_with_trace(
    leaf_values: Sequence[Any] | Iterable[Any],
    merge: Callable[[Any, Any], Any],
    *,
    schedule: ScheduleName = "balanced",
) -> list[Any]:
    """Fold leaf values through the schedule; return all node values in trace order.

    The returned list has ``2L - 1`` entries (leaves first, merges in merge
    order, root last). For a single leaf the list is just that leaf.
    """

    states = list(leaf_values)
    if not states:
        raise ValueError("fold requires at least one leaf value")
    for left, right in merge_order(len(states), schedule=schedule):
        states.append(merge(states[left], states[right]))
    return states


def fold(
    leaf_values: Sequence[Any] | Iterable[Any],
    merge: Callable[[Any, Any], Any],
    *,
    schedule: ScheduleName = "balanced",
) -> Any:
    """Fold leaf values through the schedule and return the root value."""

    return fold_with_trace(leaf_values, merge, schedule=schedule)[-1]


__all__ = [
    "fold",
    "fold_with_trace",
    "merge_children",
    "merge_depths",
    "merge_order",
]
