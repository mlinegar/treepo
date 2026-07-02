"""Alternating f/g runtime orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from treepo.methods._runtime_evaluation import evaluate_splits
from treepo.methods._runtime_schedule import (
    stage_label_for_iteration,
    stage_name_for_iteration,
    stage_powers_for_iteration,
    trains_f_at_iteration,
    trains_g_at_iteration,
)
from treepo.methods._runtime_statistics import statistic_payload
from treepo.methods._runtime_types import IterationRecord
from treepo.methods.contracts import FamilyRuntime


def run_alternating_family(
    *,
    family: FamilyRuntime,
    f_init: Any,
    g_init: Any,
    traces: Sequence[Any],
    f_traces: Sequence[Any] | None = None,
    g_traces: Sequence[Any] | None = None,
    eval_trees: Sequence[Any],
    max_iterations: int,
    axis_value: int,
    output_dir: Path,
    axis_kind: str = "leaf_count",
    leaf_count: int | None = None,
) -> list[IterationRecord]:
    """Run a compact alternating loop over a public ``FamilyRuntime``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    f_artifact = f_init
    g_artifact = g_init
    trace_list = list(traces or ())
    f_trace_list = list(f_traces) if f_traces is not None else trace_list
    g_trace_list = list(g_traces) if g_traces is not None else trace_list
    records: list[IterationRecord] = []
    max_k = max(0, int(max_iterations))
    for k in range(max_k + 1):
        trained = "none"
        iteration_dir = output_dir / f"iter_{k:02d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        if trains_f_at_iteration(k):
            f_artifact = family.train_f(
                f_init=f_artifact,
                g=g_artifact,
                traces=f_trace_list,
                output_dir=iteration_dir,
                iteration=k,
            )
            family.validate_artifact(kind="f", artifact=f_artifact)
            trained = "f"
        elif trains_g_at_iteration(k):
            g_artifact = family.train_g(
                g_init=g_artifact,
                f=f_artifact,
                traces=g_trace_list,
                output_dir=iteration_dir,
                iteration=k,
            )
            family.validate_artifact(kind="g", artifact=g_artifact)
            trained = "g"
        f_degree, g_degree = stage_powers_for_iteration(k)
        prediction_rows: list[dict[str, Any]] = []
        split_metrics = evaluate_splits(
            family,
            f_artifact,
            g_artifact,
            eval_trees,
            prediction_rows=prediction_rows,
        )
        extra: dict[str, Any] = {}
        if prediction_rows:
            extra["prediction_rows"] = prediction_rows
        stats = statistic_payload(
            family=family,
            f_artifact=f_artifact,
            g_artifact=g_artifact,
            eval_trees=eval_trees,
        )
        if stats:
            extra["statistic"] = stats
        records.append(
            IterationRecord(
                iteration=k,
                stage_name=stage_name_for_iteration(k),
                stage_label=stage_label_for_iteration(k),
                family=str(getattr(family, "name", type(family).__name__)),
                trained=trained,
                f_degree=f_degree,
                g_degree=g_degree,
                axis_kind=str(axis_kind),
                axis_value=int(axis_value),
                leaf_count=leaf_count,
                f_artifact=f_artifact,
                g_artifact=g_artifact,
                split_metrics=split_metrics,
                extra=extra,
            )
        )
    return records


__all__ = ["run_alternating_family"]
