"""Concrete `SketchAdapter` implementations.

Factory functions keep optional Apache DataSketches imports out of callers.
"""

from __future__ import annotations

from typing import Literal

from treepo.bench.sketches.protocol import CardinalitySketch


def make_hll_adapter(
    *,
    backend: Literal["native", "datasketches"],
    precision: int,
    hash_bits: int = 64,
) -> CardinalitySketch:
    """Return an HLL `CardinalitySketch` adapter.

    - `backend="native"` wraps `treepo.hll.HyperLogLogSketch` (no extra deps).
    - `backend="datasketches"` wraps `datasketches.hll_sketch`. Requires
      `uv sync --extra sketches`; raises a clear `ImportError` otherwise.
    """
    if backend == "native":
        from treepo.bench.sketches.adapters.hll_native import HLLNativeAdapter
        return HLLNativeAdapter(precision=int(precision), hash_bits=int(hash_bits))
    if backend == "datasketches":
        from treepo.bench.sketches.adapters.hll_datasketches import HLLDatasketchesAdapter
        return HLLDatasketchesAdapter(precision=int(precision))
    raise ValueError(f"unknown HLL backend: {backend!r}; expected 'native' or 'datasketches'")


def make_cpc_adapter(*, lg_k: int = 10) -> CardinalitySketch:
    from treepo.bench.sketches.adapters.datasketches_cardinality import CPCDatasketchesAdapter

    return CPCDatasketchesAdapter(lg_k=int(lg_k))


def make_theta_adapter(*, lg_k: int = 12) -> CardinalitySketch:
    from treepo.bench.sketches.adapters.datasketches_cardinality import ThetaDatasketchesAdapter

    return ThetaDatasketchesAdapter(lg_k=int(lg_k))


def make_count_min_adapter(*, num_hashes: int = 5, num_buckets: int = 256):
    from treepo.bench.sketches.adapters.datasketches_frequency import CountMinDatasketchesAdapter

    return CountMinDatasketchesAdapter(num_hashes=int(num_hashes), num_buckets=int(num_buckets))


def make_frequent_strings_adapter(*, lg_max_map_size: int = 8):
    from treepo.bench.sketches.adapters.datasketches_frequency import FrequentStringsDatasketchesAdapter

    return FrequentStringsDatasketchesAdapter(lg_max_map_size=int(lg_max_map_size))


def make_kll_floats_adapter(*, k: int = 200):
    from treepo.bench.sketches.adapters.datasketches_quantiles import KLLFloatsDatasketchesAdapter

    return KLLFloatsDatasketchesAdapter(k=int(k))


def make_quantiles_floats_adapter(*, k: int = 128):
    from treepo.bench.sketches.adapters.datasketches_quantiles import QuantilesFloatsDatasketchesAdapter

    return QuantilesFloatsDatasketchesAdapter(k=int(k))


def make_req_floats_adapter(*, k: int = 12, high_rank_accuracy: bool = True):
    from treepo.bench.sketches.adapters.datasketches_quantiles import REQFloatsDatasketchesAdapter

    return REQFloatsDatasketchesAdapter(k=int(k), high_rank_accuracy=bool(high_rank_accuracy))


def make_tdigest_double_adapter(*, k: int = 200):
    from treepo.bench.sketches.adapters.datasketches_quantiles import TDigestDoubleDatasketchesAdapter

    return TDigestDoubleDatasketchesAdapter(k=int(k))


def make_tuple_accumulator_adapter(*, lg_k: int = 12):
    from treepo.bench.sketches.adapters.datasketches_tuple_sampling import (
        TupleAccumulatorDatasketchesAdapter,
    )

    return TupleAccumulatorDatasketchesAdapter(lg_k=int(lg_k))


def make_varopt_strings_adapter(*, k: int = 64):
    from treepo.bench.sketches.adapters.datasketches_tuple_sampling import (
        VarOptStringsDatasketchesAdapter,
    )

    return VarOptStringsDatasketchesAdapter(k=int(k))


__all__ = [
    "make_hll_adapter",
    "make_cpc_adapter",
    "make_theta_adapter",
    "make_count_min_adapter",
    "make_frequent_strings_adapter",
    "make_kll_floats_adapter",
    "make_quantiles_floats_adapter",
    "make_req_floats_adapter",
    "make_tdigest_double_adapter",
    "make_tuple_accumulator_adapter",
    "make_varopt_strings_adapter",
]
