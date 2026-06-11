"""Trees are pre-allocated, never built live inside ``fit()``.

The paper-grade workflow builds trees ONCE (synthetic samplers or
labeled-tree JSONL on disk), passes them as ``eval_data`` / ``train_data``,
and reuses them across many cells in a grid. ``fit()`` must therefore:

1. Pass the supplied trees *by reference* to the family — no deep copy,
   no rebuild, no LLM summarizer firing at score time.
2. Not call the fixture builders during the alternating loop.
3. Reuse the same tree *objects* across grid cells with matching args
   (the ``lru_cache`` on the fixture builders is the mechanism).

These tests assert all three. A failure here means a future change
broke the pre-allocation contract — either ``fit()`` started copying
trees, or a family started rebuilding from raw inputs, or the fixture
cache regressed.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, List, Optional, Sequence
from unittest.mock import patch

import pytest

import treepo.methods
from treepo._research.methods.lda_fixtures import make_leaf_local_mixture_trees
from treepo.methods.fixtures import make_hll_token_trees


@pytest.fixture(autouse=True)
def _clear_fixture_caches():
    make_hll_token_trees.cache_clear()
    make_leaf_local_mixture_trees.cache_clear()
    yield
    make_hll_token_trees.cache_clear()
    make_leaf_local_mixture_trees.cache_clear()


class _IdentityRecordingFamily:
    """Minimum FamilyRuntime that records the ``id()`` of every tree it
    is asked to score. We use object identity (not equality) so the test
    fails loud if ``fit()`` ever inserts a copy/clone/serialize step.
    """

    name = "identity_recorder"

    def __init__(self) -> None:
        self.scored_ids: List[int] = []
        self.train_call_count: int = 0

    def train_f(self, *, f_init, g, traces, output_dir, iteration):
        self.train_call_count += 1
        return f_init

    def train_g(self, *, g_init, f, traces, output_dir, iteration):
        self.train_call_count += 1
        return g_init

    def score_roots_with_f(
        self, *, f: Any, g: Any, trees: Sequence[Any]
    ) -> List[Optional[float]]:
        self.scored_ids.extend(id(t) for t in trees)
        return [None] * len(trees)

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        return None


def _fit_with(family, eval_data, tmp_path: Path, **overrides) -> None:
    treepo.methods.run(
        "fit",
        {
            "family": "identity_check",
            "eval_data": eval_data,
            "backend_config": {
                "family_runtime": family,
                "output_dir": str(tmp_path),
            },
            **overrides,
        },
    )


# --------------------------------------------------------------------------- #
# 1. Trees pass through fit() by reference — same object ids in and out.
# --------------------------------------------------------------------------- #


def test_hll_trees_pass_to_family_by_reference(tmp_path: Path) -> None:
    trees = make_hll_token_trees(n_trees=6, seed=0)
    expected_ids = {id(t) for t in trees}

    family = _IdentityRecordingFamily()
    _fit_with(family, trees, tmp_path)

    assert set(family.scored_ids) == expected_ids, (
        "fit() must pass trees to the family by reference; got different "
        f"object ids in than out (expected={expected_ids}, got={set(family.scored_ids)})"
    )


def test_lda_trees_pass_to_family_by_reference(tmp_path: Path) -> None:
    trees, _cfg = make_leaf_local_mixture_trees(seed=0)
    expected_ids = {id(t) for t in trees}

    family = _IdentityRecordingFamily()
    _fit_with(family, trees, tmp_path)

    assert set(family.scored_ids) == expected_ids


def test_trees_passed_as_iterator_still_arrive_by_reference(tmp_path: Path) -> None:
    """``_as_sequence`` accepts a non-Sequence iterable. Each tree object
    must still arrive at the family unchanged.
    """
    trees = make_hll_token_trees(n_trees=4, seed=7)
    expected_ids = {id(t) for t in trees}

    family = _IdentityRecordingFamily()
    _fit_with(family, iter(trees), tmp_path)
    assert set(family.scored_ids) == expected_ids


# --------------------------------------------------------------------------- #
# 2. fit() does not call the fixture builders mid-flight.
# --------------------------------------------------------------------------- #


def test_fit_does_not_invoke_hll_builder_during_loop(tmp_path: Path) -> None:
    """Build trees up front. Patch the builder to fail loud if it's
    called again. fit() must succeed without re-entering the builder.
    """
    trees = make_hll_token_trees(n_trees=4, seed=11)
    # Snapshot the cache state — fit() must NOT increment misses.
    before = make_hll_token_trees.cache_info()

    family = _IdentityRecordingFamily()
    with patch(
        "treepo.methods.fixtures.make_hll_token_trees",
        side_effect=AssertionError("fit() must not rebuild trees mid-loop"),
    ):
        _fit_with(family, trees, tmp_path)

    after = make_hll_token_trees.cache_info()
    assert after.misses == before.misses
    assert len(family.scored_ids) == len(trees)


def test_fit_does_not_invoke_lda_builder_during_loop(tmp_path: Path) -> None:
    trees, _cfg = make_leaf_local_mixture_trees(seed=13)
    before = make_leaf_local_mixture_trees.cache_info()

    family = _IdentityRecordingFamily()
    with patch(
        "treepo._research.methods.lda_fixtures.make_leaf_local_mixture_trees",
        side_effect=AssertionError("fit() must not rebuild LDA trees mid-loop"),
    ):
        _fit_with(family, trees, tmp_path)

    after = make_leaf_local_mixture_trees.cache_info()
    assert after.misses == before.misses


# --------------------------------------------------------------------------- #
# 3. Grid runs share tree objects across cells with matching fixture args.
# --------------------------------------------------------------------------- #


def test_grid_shares_tree_objects_across_cells_with_matching_args(
    tmp_path: Path,
) -> None:
    """A grid that varies a downstream knob (here: ``output_dir`` per
    cell) but holds the fixture args constant must reuse the same tree
    objects via ``lru_cache``. The recorder's id sequences match across
    cells.
    """
    family = _IdentityRecordingFamily()
    id_sequences: List[List[int]] = []
    for cell_idx in range(5):
        trees = make_hll_token_trees(n_trees=4, seed=17)
        family.scored_ids = []  # reset per cell
        _fit_with(
            family,
            trees,
            tmp_path / f"cell_{cell_idx}",
        )
        id_sequences.append(list(family.scored_ids))

    # Every cell saw the same five tree objects (same ids).
    assert all(seq == id_sequences[0] for seq in id_sequences), (
        f"grid cells saw different tree objects: {id_sequences}"
    )
    # And the fixture was sampled exactly once.
    assert make_hll_token_trees.cache_info().misses == 1


def test_two_axis_grid_caches_one_world_per_axis_combo(tmp_path: Path) -> None:
    """A 2-axis fixture grid (seed × n_trees) over 20 cells (5 seeds × 4
    n_trees × 1 dummy outer) builds at most 5*4 = 20 unique trees, then
    every repeat is a cache hit — same object identities.
    """
    seeds = [0, 1, 2, 3, 4]
    n_trees_values = [2, 4, 8, 16]

    ids_first_pass: dict[tuple[int, int], list[int]] = {}
    ids_second_pass: dict[tuple[int, int], list[int]] = {}

    for pass_num, target in enumerate((ids_first_pass, ids_second_pass)):
        for seed, n_trees in product(seeds, n_trees_values):
            trees = make_hll_token_trees(n_trees=n_trees, seed=seed)
            family = _IdentityRecordingFamily()
            _fit_with(
                family,
                trees,
                tmp_path / f"pass{pass_num}_s{seed}_n{n_trees}",
            )
            target[(seed, n_trees)] = list(family.scored_ids)

    # Second pass over identical args produces identical object ids.
    for key in ids_first_pass:
        assert ids_first_pass[key] == ids_second_pass[key], (
            f"cell {key} saw different objects across passes — cache regressed"
        )
    # And only 20 unique fixtures were sampled across 40 cells.
    info = make_hll_token_trees.cache_info()
    assert info.misses == len(seeds) * len(n_trees_values)
    assert info.hits == info.misses  # each unique sampled once, hit once


# --------------------------------------------------------------------------- #
# 4. Documented expectation: fit() does not lazily construct trees from
#    a "builder spec". The caller hands fit() the trees; fit() runs them.
# --------------------------------------------------------------------------- #


def test_fit_with_zero_eval_trees_does_not_construct_anything(
    tmp_path: Path,
) -> None:
    """Empty eval_data means zero score calls — the family is never
    asked to score anything, and no fixture is sampled.
    """
    before_hll = make_hll_token_trees.cache_info()
    before_lda = make_leaf_local_mixture_trees.cache_info()

    family = _IdentityRecordingFamily()
    with patch(
        "treepo.methods.fixtures.make_hll_token_trees",
        side_effect=AssertionError("fit() must not sample HLL trees lazily"),
    ), patch(
        "treepo._research.methods.lda_fixtures.make_leaf_local_mixture_trees",
        side_effect=AssertionError("fit() must not sample LDA trees lazily"),
    ):
        _fit_with(family, [], tmp_path)

    assert make_hll_token_trees.cache_info() == before_hll
    assert make_leaf_local_mixture_trees.cache_info() == before_lda
    # No trees → no score calls. (The evaluator at k=0 still runs with
    # an empty list; that's the alternating-loop's responsibility, but
    # no calls into the family's score function happen because there
    # are no trees to score.)
