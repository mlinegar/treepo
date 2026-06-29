"""Mergeable-sketch adapters and tree-reduction for classical-sketch benchmarks.

The `SketchAdapter` Protocol lets classical sketches plug into TreePO's tree
reduction with a uniform `update / encode / merge / query / state_equal /
serialize / serialized_size_bytes / memory_bytes` surface.

`treepo_reduce(items_per_leaf, adapter, schedule)` is the sketch-agnostic
generalization of tree-style sketch reduction.
"""

from treepo.bench.sketches.adapters import (
    make_count_min_adapter,
    make_cpc_adapter,
    make_frequent_strings_adapter,
    make_hll_adapter,
    make_kll_floats_adapter,
    make_quantiles_floats_adapter,
    make_req_floats_adapter,
    make_tdigest_double_adapter,
    make_theta_adapter,
    make_tuple_accumulator_adapter,
    make_varopt_strings_adapter,
)
from treepo.bench.sketches.protocol import CardinalitySketch, SketchAdapter
from treepo.bench.sketches.tree_reducer import treepo_reduce

__all__ = [
    "CardinalitySketch",
    "SketchAdapter",
    "treepo_reduce",
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
