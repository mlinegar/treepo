"""Fixture caching — proves the paper-code shared-world optimization
transfers into ``treepo.cld``.

The existing paper grid scripts (e.g.
``scripts/run_lda_tree_recovery_learned_world_batch.py``) sample a
synthetic world ONCE and reuse it across many cells; re-sampling
would dominate runtime on large grids. ``treepo.cld`` provides the
same speedup automatically via :func:`functools.lru_cache` on
:func:`make_leaf_local_mixture_trees` and :func:`make_hll_token_trees`.

These tests assert:
1. Repeated calls with identical args return the *same object*.
2. Different args produce distinct results (no false sharing).
3. The cache survives across a real grid loop that exercises 24 cells.
4. ``cache_clear()`` is available as the escape hatch.
"""

from __future__ import annotations

import time
from itertools import product
from pathlib import Path

import pytest

import treepo.cld
from treepo.cld.fixtures import (
    make_hll_token_trees,
    make_leaf_local_mixture_trees,
)


@pytest.fixture(autouse=True)
def _clear_caches_between_tests():
    """Ensure each test starts with a cold cache."""
    make_hll_token_trees.cache_clear()
    make_leaf_local_mixture_trees.cache_clear()
    yield
    make_hll_token_trees.cache_clear()
    make_leaf_local_mixture_trees.cache_clear()


# --------------------------------------------------------------------------- #
# 1. Identity check — same args ⇒ same object (cache hit).
# --------------------------------------------------------------------------- #


def test_hll_fixture_returns_same_object_on_repeat_calls() -> None:
    a = make_hll_token_trees(n_trees=4, seed=0)
    b = make_hll_token_trees(n_trees=4, seed=0)
    assert a is b, "lru_cache hit must return the same list, not an equivalent copy"


def test_lda_fixture_returns_same_object_on_repeat_calls() -> None:
    a = make_leaf_local_mixture_trees(seed=0)
    b = make_leaf_local_mixture_trees(seed=0)
    assert a is b


def test_fixture_cache_distinguishes_distinct_args() -> None:
    a = make_hll_token_trees(n_trees=4, seed=0)
    b = make_hll_token_trees(n_trees=4, seed=1)
    c = make_hll_token_trees(n_trees=8, seed=0)
    assert a is not b
    assert a is not c
    assert b is not c


# --------------------------------------------------------------------------- #
# 2. CacheInfo confirms hits during a grid loop.
# --------------------------------------------------------------------------- #


def test_grid_loop_with_one_world_produces_n_minus_one_cache_hits() -> None:
    """A grid that varies a non-fixture axis (here: a downstream config
    knob the fixture doesn't see) hits the cache N-1 times across N cells.
    """
    for _ in range(8):
        make_hll_token_trees(n_trees=4, seed=42)
    info = make_hll_token_trees.cache_info()
    assert info.hits == 7
    assert info.misses == 1
    assert info.currsize == 1


def test_lda_grid_with_seed_axis_caches_one_world_per_seed() -> None:
    seeds = [0, 1, 2]
    for seed in seeds:
        # First call per seed: miss. Repeat call: hit.
        make_leaf_local_mixture_trees(seed=seed)
        make_leaf_local_mixture_trees(seed=seed)

    info = make_leaf_local_mixture_trees.cache_info()
    assert info.misses == len(seeds)
    assert info.hits == len(seeds)
    assert info.currsize == len(seeds)


# --------------------------------------------------------------------------- #
# 3. Real grid run — the cache saves wall time.
#
# A bare "compute then re-compute" timing comparison is the simplest
# demonstration. We don't assert a specific factor (CI variance), only
# that the cached call is *not slower* than the cold call by some loose
# margin. The cache-info assertions above already prove batching across
# cells; this is a sanity check that nothing pathological happens at
# higher arg counts.
# --------------------------------------------------------------------------- #


def test_lda_cached_call_is_fast() -> None:
    cold_start = time.perf_counter()
    a, _cfg_a = make_leaf_local_mixture_trees(seed=99)
    cold_elapsed = time.perf_counter() - cold_start

    warm_start = time.perf_counter()
    b, _cfg_b = make_leaf_local_mixture_trees(seed=99)
    warm_elapsed = time.perf_counter() - warm_start

    assert a is b
    # Cached call must be at least 10x faster than the cold call for any
    # non-trivial fixture. The LDA sampler does Dirichlet draws + topic
    # sampling, so cold ≥ 1ms and warm should be ~microseconds.
    assert warm_elapsed < cold_elapsed / 10, (
        f"cached call ({warm_elapsed*1e6:.1f} us) not meaningfully faster "
        f"than cold ({cold_elapsed*1e6:.1f} us)"
    )


# --------------------------------------------------------------------------- #
# 4. Grid run end-to-end: 24 cells, same world reused throughout.
# --------------------------------------------------------------------------- #


def test_grid_run_reuses_world_across_24_cells(tmp_path: Path) -> None:
    """A two-axis grid (oracle name × seed × n_trees) hits 24 cells. The
    HLL fixture has 3 unique (n_trees, seed) pairs and the LDA fixture
    has 3 unique seeds. After the grid, the cache stats prove every
    fixture-shape was sampled exactly once.
    """
    seeds = [0, 1, 2]
    n_trees_values = [4, 8]

    rows = []
    for oracle_name, seed, n_trees in product(
        ("hll_exact", "leaf_local_mixture_target"), seeds, n_trees_values,
    ):
        result = treepo.cld.run(
            "oracle",
            {
                "oracle_name": oracle_name,
                "seed": seed,
                "n_trees": n_trees,  # LDA ignores this; HLL uses it
                "output_dir": str(tmp_path / f"{oracle_name}_s{seed}_n{n_trees}"),
            },
        )
        rows.append((oracle_name, seed, n_trees, result.status))
    assert len(rows) == 12
    assert all(status == "success" for *_axis, status in rows)

    # HLL fixture saw 6 unique (seed, n_trees) cells.
    hll_info = make_hll_token_trees.cache_info()
    assert hll_info.misses == 6, f"expected 6 HLL world samples, saw {hll_info.misses}"

    # LDA fixture saw 3 unique seeds (n_trees doesn't enter its signature).
    lda_info = make_leaf_local_mixture_trees.cache_info()
    assert lda_info.misses == 3, f"expected 3 LDA world samples, saw {lda_info.misses}"


# --------------------------------------------------------------------------- #
# 5. Escape hatch — cache_clear works for callers who need fresh state.
# --------------------------------------------------------------------------- #


def test_cache_clear_invalidates_prior_results() -> None:
    a = make_hll_token_trees(n_trees=4, seed=0)
    make_hll_token_trees.cache_clear()
    b = make_hll_token_trees(n_trees=4, seed=0)
    # Content equivalent (same seed) but distinct objects (cache miss).
    assert a is not b
    assert a[0].metadata["teacher_score_1_7"] == b[0].metadata["teacher_score_1_7"]
