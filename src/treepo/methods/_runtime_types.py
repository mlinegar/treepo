"""Dataclasses shared by the methods alternating runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SplitMetrics:
    n: int
    internal_f_pearson: float | None = None
    internal_f_mae: float | None = None
    external_expert_pearson: float | None = None
    external_expert_mae: float | None = None
    f_star_gap: float | None = None
    mean_prediction: float | None = None
    mean_teacher: float | None = None
    mean_expert: float | None = None
    metrics_scale: str | None = None
    per_dimension: dict[str, dict[str, float | None]] = field(default_factory=dict)


@dataclass
class IterationRecord:
    iteration: int
    stage_name: str
    family: str
    trained: str
    stage_label: str | None = None
    f_degree: int | None = None
    g_degree: int | None = None
    axis_kind: str = "leaf_count"
    axis_value: int = 0
    leaf_count: int | None = None
    f_artifact: Any = None
    g_artifact: Any = None
    split_metrics: dict[str, SplitMetrics] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


__all__ = ["IterationRecord", "SplitMetrics"]
