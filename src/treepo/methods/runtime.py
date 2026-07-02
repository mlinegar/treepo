"""Compatibility facade for the methods alternating runtime."""

from __future__ import annotations

from treepo.methods._runtime_evaluation import evaluate_splits
from treepo.methods._runtime_loop import run_alternating_family
from treepo.methods._runtime_types import IterationRecord, SplitMetrics

__all__ = [
    "IterationRecord",
    "SplitMetrics",
    "evaluate_splits",
    "run_alternating_family",
]
