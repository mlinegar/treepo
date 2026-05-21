from __future__ import annotations

import math

import numpy as np
import pytest

from treepo import (
    HLLConfig,
    HyperLogLogSketch,
    hll_relative_standard_error,
    reduce_hll_sketches,
)


def test_hll_streaming_update_matches_batch_update() -> None:
    tokens = [1, 5, 5, 7, 11, 13, 13, 21, 42]
    cfg = HLLConfig(precision=8, hash_bits=64)

    a = HyperLogLogSketch(cfg)
    for tok in tokens:
        a.add(tok)

    b = HyperLogLogSketch(cfg)
    b.update(tokens)

    assert np.array_equal(a.registers, b.registers)


def test_hll_merge_is_associative_commutative_and_idempotent() -> None:
    cfg = HLLConfig(precision=8, hash_bits=64)
    left = HyperLogLogSketch(cfg).update([1, 2, 3, 4])
    right = HyperLogLogSketch(cfg).update([3, 4, 5, 6])
    extra = HyperLogLogSketch(cfg).update([6, 7, 8, 9])

    a = left.copy().merge(right.copy()).merge(extra.copy())
    b = left.copy().merge(right.copy().merge(extra.copy()))
    c = right.copy().merge(left.copy()).merge(extra.copy())
    d = left.copy().merge(left.copy())

    assert np.array_equal(a.registers, b.registers)
    assert np.array_equal(a.registers, c.registers)
    assert np.array_equal(d.registers, left.registers)


def test_hll_schedule_reduction_is_register_invariant() -> None:
    cfg = HLLConfig(precision=8, hash_bits=64)
    leaves = [
        HyperLogLogSketch(cfg).update([0, 1, 2]),
        HyperLogLogSketch(cfg).update([2, 3, 4]),
        HyperLogLogSketch(cfg).update([4, 5, 6]),
        HyperLogLogSketch(cfg).update([6, 7, 8]),
    ]
    roots = [
        reduce_hll_sketches(leaves, schedule=sched)
        for sched in ("balanced", "left_to_right", "right_to_left")
    ]
    for root in roots[1:]:
        assert np.array_equal(roots[0].registers, root.registers)


def test_hll_accuracy_sanity_on_fixed_cardinalities() -> None:
    cfg = HLLConfig(precision=10, hash_bits=64)
    rel_errors = []
    for n in (64, 128, 256, 512):
        sk = HyperLogLogSketch(cfg).update(range(n))
        rel_errors.append(abs(sk.estimate() - float(n)) / float(n))

    assert float(np.mean(np.asarray(rel_errors, dtype=np.float64))) < 0.15
    assert hll_relative_standard_error(cfg.precision) == pytest.approx(
        1.04 / math.sqrt(2**10)
    )
