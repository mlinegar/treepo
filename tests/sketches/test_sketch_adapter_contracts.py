"""Adapter-level contract tests (S1/S2/S3/S4) — fast CI gate.

These bypass `fit()` for quick feedback; the `fit()`-routed equivalents live
in `parallel/unified_g_v1/tests/test_classical_hll_parity_fit.py`.

- S1 single-leaf identity: L=1 TreePO state byte-identical to flat state.
- S2 schedule invariance: balanced / L→R / R→L produce byte-equal states.
- S3 permutation invariance: permuting leaf order preserves state.
- S4 reference agreement: TreePO-reduced state equals flat-reduced state
  for associative+commutative adapters, for L ∈ {2,4,8,16}.
"""

from __future__ import annotations

import random

import pytest

from treepo.bench.sketches import make_hll_adapter, treepo_reduce
from treepo.bench.sketches.tree_reducer import fold_states

pytest.importorskip("datasketches")

BACKENDS = ["datasketches"]
SCHEDULES = ("balanced", "left_to_right", "right_to_left")


def _seeded_tokens(n: int, universe: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(universe) for _ in range(n)]


def _chunk(xs: list[int], n_chunks: int) -> list[list[int]]:
    k = max(1, (len(xs) + n_chunks - 1) // n_chunks)
    return [xs[i : i + k] for i in range(0, len(xs), k)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_s1_single_leaf_identity(backend: str) -> None:
    adapter = make_hll_adapter(backend=backend, precision=10)
    items = _seeded_tokens(500, universe=10_000, seed=0)

    tree_state = treepo_reduce([items], adapter, schedule="balanced")
    flat_state = adapter.encode(items)

    assert adapter.state_equal(tree_state, flat_state)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_leaves", [2, 4, 8, 16])
def test_s2_schedule_invariance(backend: str, n_leaves: int) -> None:
    adapter = make_hll_adapter(backend=backend, precision=10)
    items = _seeded_tokens(2000, universe=20_000, seed=1)
    leaves = _chunk(items, n_leaves)

    leaf_states = [adapter.encode(chunk) for chunk in leaves]
    roots = [fold_states(leaf_states, adapter, schedule=sched) for sched in SCHEDULES]

    for other in roots[1:]:
        assert adapter.state_equal(roots[0], other)


@pytest.mark.parametrize("backend", BACKENDS)
def test_s3_permutation_invariance(backend: str) -> None:
    adapter = make_hll_adapter(backend=backend, precision=10)
    items = _seeded_tokens(2000, universe=20_000, seed=2)
    leaves = _chunk(items, 8)

    leaf_states = [adapter.encode(chunk) for chunk in leaves]
    permuted = list(leaf_states)
    random.Random(42).shuffle(permuted)

    original = fold_states(leaf_states, adapter, schedule="balanced")
    shuffled = fold_states(permuted, adapter, schedule="balanced")

    assert adapter.state_equal(original, shuffled)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_leaves", [2, 4, 8, 16])
def test_s4_reference_agreement(backend: str, n_leaves: int) -> None:
    adapter = make_hll_adapter(backend=backend, precision=11)
    items = _seeded_tokens(3000, universe=50_000, seed=3)
    leaves = _chunk(items, n_leaves)

    tree_state = treepo_reduce(leaves, adapter, schedule="balanced")
    flat_state = adapter.encode(items)

    assert adapter.state_equal(tree_state, flat_state)
    # Byte-deterministic adapters hold the stronger claim: bytes match exactly.
    if adapter.is_byte_deterministic:
        assert adapter.serialize(tree_state) == adapter.serialize(flat_state)



@pytest.mark.parametrize("backend", BACKENDS)
def test_theoretical_accuracy_floor(backend: str) -> None:
    precision = 12
    adapter = make_hll_adapter(backend=backend, precision=precision)
    m = 1 << precision

    rel_errors = []
    for seed, n in enumerate((256, 1024, 4096, 16384)):
        items = _seeded_tokens(n * 2, universe=n * 4, seed=200 + seed)
        est = adapter.query(adapter.encode(items))
        truth = float(len(set(items)))
        rel_errors.append(abs(est - truth) / truth)

    rmse_rel = (sum(e * e for e in rel_errors) / len(rel_errors)) ** 0.5
    theoretical_floor = 1.04 / (m ** 0.5)
    assert rmse_rel < 2.5 * theoretical_floor, (
        f"{backend} rmse_rel={rmse_rel:.4f} exceeds 2.5x floor {theoretical_floor:.4f}"
    )
