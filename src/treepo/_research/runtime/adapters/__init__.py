"""Benchmark adapters for the runtime backbone."""

from treepo._research.runtime.adapters.longbench import LongBenchV2Adapter, LongBenchV2Spec
from treepo._research.runtime.adapters.registry import build_benchmark_adapter
from treepo._research.runtime.adapters.ruler import RulerDatasetSpec, RulerSyntheticAdapter

__all__ = [
    "LongBenchV2Adapter",
    "LongBenchV2Spec",
    "RulerDatasetSpec",
    "RulerSyntheticAdapter",
    "build_benchmark_adapter",
]
