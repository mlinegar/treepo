"""Sketch-agnostic tree reduction.

`treepo_reduce(items_per_leaf, adapter, schedule)` encodes each leaf's
items via the adapter, then folds the resulting states according to `schedule`.

Schedules (matching `treepo.common.VALID_SCHEDULES`):
- "balanced": pairwise merge, level by level.
- "left_to_right": accumulate left to right.
- "right_to_left": accumulate right to left.

For adapters where `is_associative` and `is_commutative` are both True (HLL,
CMS, Bloom, Theta), all three schedules produce states that compare equal via
`adapter.state_equal`. For adapters that are associative but not commutative
(e.g. sequence-sensitive), only `balanced` ≡ `left_to_right` as bracketings of
the same sequence.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from treepo.common import ScheduleName
from treepo.bench.sketches.protocol import SketchAdapter
from treepo.schedule import fold


def treepo_reduce(
    items_per_leaf: Sequence[Iterable],
    adapter: SketchAdapter,
    *,
    schedule: ScheduleName = "balanced",
):
    """Encode each leaf's items and fold via the adapter's `merge`.

    Returns the adapter's state type. Raises `ValueError` on empty leaves or
    unknown schedule.
    """
    if len(items_per_leaf) == 0:
        raise ValueError("items_per_leaf must be non-empty")

    leaf_states = [adapter.encode(list(items)) for items in items_per_leaf]
    return _fold(leaf_states, adapter, schedule=schedule)


def fold_states(
    leaf_states: Sequence,
    adapter: SketchAdapter,
    *,
    schedule: ScheduleName = "balanced",
):
    """Fold a pre-encoded sequence of leaf states (skip the encode step).

    Useful when the same leaf states are folded under multiple schedules in a
    schedule-invariance test.
    """
    if len(leaf_states) == 0:
        raise ValueError("leaf_states must be non-empty")
    return _fold(leaf_states, adapter, schedule=schedule)


def _fold(states: Sequence, adapter: SketchAdapter, *, schedule: ScheduleName):
    return fold(states, adapter.merge, schedule=schedule)
