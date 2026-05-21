"""Broad classical-sketch grid routed through ``fit()``.

This is the unified-g wrapper for ``treepo.bench.classical_sketches``. The
actual sketch protocol and DataSketches adapters stay in ``treepo``; this file
only makes the broad grid a first-class ``TrainerConfig`` task so official
baselines and learned companions share the same orchestration entry point.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from treepo.bench.classical_sketches import (
    ClassicalSketchComparisonConfig,
    run_classical_sketch_comparison,
)
from treepo.bench.io import (
    atomic_write_text,
    dump_json,
    summary_to_csv_rows_classical_sketches,
    write_csv_rows,
)

from treepo._research.unified_g_v1.training.tree_task import TrainerConfig


def _coerce_config(raw: Any) -> ClassicalSketchComparisonConfig:
    if isinstance(raw, ClassicalSketchComparisonConfig):
        return raw
    if raw is None:
        return ClassicalSketchComparisonConfig()
    if isinstance(raw, Mapping):
        return ClassicalSketchComparisonConfig(**dict(raw))
    return ClassicalSketchComparisonConfig(**dict(raw))


def _aggregate_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    numeric_keys = (
        "relative_rmse",
        "mean_abs_rel_error",
        "schedule_spread_mean",
        "schedule_spread_p95",
        "memory_bytes_mean",
        "distance_to_official_floor",
    )
    metrics: dict[str, float] = {"row_count": float(len(rows))}
    for key in numeric_keys:
        vals: list[float] = []
        for row in rows:
            try:
                val = float(row.get(key, float("nan")))
            except (TypeError, ValueError):
                continue
            if val == val:
                vals.append(val)
        if vals:
            metrics[f"mean_{key}"] = float(sum(vals) / len(vals))
            metrics[f"max_{key}"] = float(max(vals))
    return metrics


def _attach_official_floors(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    floors: dict[tuple[str, str], float] = {}
    for row in rows:
        if str(row.get("implementation_status")) not in {"official_empirical", "lean_backed"}:
            continue
        key = (str(row.get("family")), str(row.get("query")))
        try:
            value = float(row.get("relative_rmse", float("nan")))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        floors[key] = min(value, floors.get(key, value))
    out: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        key = (str(item.get("family")), str(item.get("query")))
        floor = floors.get(key, float("nan"))
        item["official_floor_rel_rmse"] = float(floor)
        try:
            value = float(item.get("relative_rmse", float("nan")))
        except (TypeError, ValueError):
            value = float("nan")
        item["distance_to_official_floor"] = (
            float(value - floor) if math.isfinite(value) and math.isfinite(floor) else float("nan")
        )
        out.append(item)
    return out


def run_classical_sketch_grid_baseline(
    cfg: TrainerConfig,
    output_dir: str | Path,
    dataset: Any = None,
):
    """Zero-optimization trainer for the broad official-sketch grid."""
    del dataset
    from treepo._research.unified_g_v1.training.fit import FitResult

    sketch_cfg = _coerce_config(dict(cfg.extra or {}).get("classical_sketch_config"))
    t0 = time.perf_counter()
    summary = run_classical_sketch_comparison(sketch_cfg)
    rows = summary_to_csv_rows_classical_sketches(summary)
    if bool(sketch_cfg.include_learned):
        from treepo._research.unified_g_v1.sketch.learned_sketch_grid import run_learned_sketch_grid

        learned_rows = run_learned_sketch_grid(
            sketch_cfg,
            output_dir=Path(output_dir) / "learned",
        )
        for row in learned_rows:
            row["experiment"] = "classical_sketches"
        rows = _attach_official_floors([dict(row) for row in rows] + learned_rows)
    metrics = _aggregate_metrics([dict(row) for row in rows])
    metrics["total_wall_seconds"] = float(time.perf_counter() - t0)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = json.loads(summary.to_json())
    summary_payload["rows"] = [dict(row) for row in rows]
    fit_payload = {
        "config": asdict(sketch_cfg),
        "backend": "classical_sketch_grid",
        "metrics": metrics,
        "summary": summary_payload,
    }
    atomic_write_text(out_dir / "fit_result.json", dump_json(fit_payload))
    write_csv_rows(out_dir / "summary.csv", rows)

    return FitResult(
        backend="classical_sketch_grid",
        summary=fit_payload,
        status="completed",
        metrics=metrics,
        artifacts={
            "fit_result_json": str(out_dir / "fit_result.json"),
            "summary_csv": str(out_dir / "summary.csv"),
        },
        history=[{"epoch": 0, **metrics}],
    )


def classical_sketch_grid_task(
    config: ClassicalSketchComparisonConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> TrainerConfig:
    """Build a ``TrainerConfig`` for the broad classical-sketch comparison."""
    base = _coerce_config(config)
    if overrides:
        raw = asdict(base)
        raw.update(overrides)
        base = ClassicalSketchComparisonConfig(**raw)
    return TrainerConfig(
        trainer=run_classical_sketch_grid_baseline,
        n_epochs=0,
        seed=int(base.seed),
        best_metric_key="mean_relative_rmse",
        use_cuda=False,
        extra={"classical_sketch_config": base},
    )


__all__ = [
    "classical_sketch_grid_task",
    "run_classical_sketch_grid_baseline",
]
