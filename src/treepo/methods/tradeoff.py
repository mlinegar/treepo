"""The compression-tradeoff curve record.

``TradeoffCurve`` is the canonical shape for the package's central empirical
object: task error as a function of leaf grouping size (or any other single
axis). Grid runs build it from their row dicts with ``from_rows``; it writes
one JSON payload plus a flat CSV, and its ``to_dict`` payload renders as a
chart panel in ``treepo.viz``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from treepo.common import jsonable
from treepo.methods._coerce import safe_float


@dataclass(frozen=True)
class TradeoffCurve:
    """Metric values across one ordered axis, sorted by axis value."""

    axis_kind: str
    metric_keys: tuple[str, ...]
    points: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.axis_kind):
            raise ValueError("axis_kind is required")
        metric_keys = tuple(str(key) for key in self.metric_keys)
        if not metric_keys:
            raise ValueError("metric_keys must be non-empty")
        normalized = []
        for point in self.points:
            axis_value = safe_float(point.get("axis_value"))
            if axis_value is None:
                raise ValueError(f"tradeoff point requires a numeric axis_value: {point!r}")
            metrics_in = point.get("metrics")
            metrics_in = metrics_in if isinstance(metrics_in, Mapping) else {}
            normalized.append(
                {
                    "axis_value": axis_value,
                    "metrics": {key: safe_float(metrics_in.get(key)) for key in metric_keys},
                    "metadata": jsonable(dict(point.get("metadata") or {})),
                }
            )
        normalized.sort(key=lambda point: point["axis_value"])
        axis_values = [point["axis_value"] for point in normalized]
        if len(set(axis_values)) != len(axis_values):
            raise ValueError(f"duplicate axis values in tradeoff curve: {axis_values!r}")
        object.__setattr__(self, "axis_kind", str(self.axis_kind))
        object.__setattr__(self, "metric_keys", metric_keys)
        object.__setattr__(self, "points", tuple(normalized))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
        *,
        metric_keys: Sequence[str],
        axis_key: str = "leaf_unit_count",
        axis_kind: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "TradeoffCurve":
        """Build a curve from flat row dicts (one row per grid cell)."""

        points = [
            {
                "axis_value": row.get(axis_key),
                "metrics": {str(key): row.get(key) for key in metric_keys},
            }
            for row in rows
        ]
        return cls(
            axis_kind=str(axis_kind or axis_key),
            metric_keys=tuple(str(key) for key in metric_keys),
            points=tuple(points),
            metadata=dict(metadata or {}),
        )

    def rows(self) -> list[dict[str, Any]]:
        """Return flat rows (axis value plus one column per metric)."""

        return [
            {self.axis_kind: point["axis_value"], **point["metrics"]}
            for point in self.points
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_kind": self.axis_kind,
            "metric_keys": list(self.metric_keys),
            "points": [jsonable(point) for point in self.points],
            "metadata": jsonable(dict(self.metadata)),
        }

    def write(self, *, json_out: str | Path, csv_out: str | Path) -> None:
        """Write the curve payload as JSON plus a flat CSV."""

        from treepo.methods.grid import write_grid_outputs

        write_grid_outputs(
            json_out=json_out,
            csv_out=csv_out,
            payload=self.to_dict(),
            rows=self.rows(),
        )


__all__ = ["TradeoffCurve"]
