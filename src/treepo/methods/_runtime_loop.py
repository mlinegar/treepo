"""Alternating f/g runtime orchestration, schedule, and statistic payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from treepo.local_law import local_law_objective_summary
from treepo.methods._runtime_evaluation import evaluate_splits
from treepo.methods._runtime_types import IterationRecord
from treepo.methods.contracts import FamilyRuntime
from treepo.statistic import family_statistic


def stage_powers_for_iteration(k: int) -> tuple[int, int]:
    if k < 0:
        raise ValueError(f"iteration must be >= 0, got {k}")
    f_degree = 1
    g_degree = 1
    side = "f"
    for _ in range(int(k)):
        if side == "f":
            f_degree += 1
            side = "g"
        else:
            g_degree += 1
            side = "f"
    return f_degree, g_degree


def stage_name_for_iteration(k: int) -> str:
    if k == 0:
        return "fg"
    return "fg" + "".join("f" if i % 2 == 0 else "g" for i in range(k))


def stage_label_for_iteration(k: int) -> str:
    f_degree, g_degree = stage_powers_for_iteration(k)
    return f"f^{f_degree} g^{g_degree}"


def trains_f_at_iteration(k: int) -> bool:
    return k >= 1 and k % 2 == 1


def trains_g_at_iteration(k: int) -> bool:
    return k >= 1 and k % 2 == 0


def statistic_payload(
    *,
    family: FamilyRuntime,
    f_artifact: Any,
    g_artifact: Any,
    eval_trees: Sequence[Any],
    objective: Any = None,
) -> dict[str, Any]:
    statistic = family_statistic(family, f=f_artifact, g=g_artifact)
    if statistic is None:
        return {}
    payload: dict[str, Any] = {"info": statistic.info.to_dict()}
    # A broken statistic must fail the run, not degrade into a metadata
    # string; local_law_rows errors propagate.
    rows = list(statistic.local_law_rows(list(eval_trees or ())))
    if rows:
        gamma_depth = float(objective.gamma_depth) if objective is not None else 1.0
        payload["local_law_summary"] = local_law_objective_summary(
            rows,
            gamma_depth=gamma_depth,
        ).to_dict()
        payload["local_law_row_count"] = int(len(rows))
        payload["local_law_gamma_depth"] = gamma_depth
    return payload


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
    objective: Any = None,
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
            objective=objective,
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


__all__ = [
    "run_alternating_family",
    "stage_label_for_iteration",
    "stage_name_for_iteration",
    "stage_powers_for_iteration",
    "statistic_payload",
    "trains_f_at_iteration",
    "trains_g_at_iteration",
]
