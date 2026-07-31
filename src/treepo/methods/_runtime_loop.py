"""Alternating f/g runtime orchestration, schedule, and statistic payloads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from treepo.local_law import local_law_objective_summary
from treepo.methods._runtime_evaluation import evaluate_splits
from treepo.methods._runtime_types import IterationRecord
from treepo.methods.contracts import (
    G_MODE_FIXED,
    G_MODE_IDENTITY,
    G_MODE_LEARNED,
    FamilyRuntime,
    normalize_g_mode,
)
from treepo.statistic import family_statistic


@dataclass(frozen=True)
class GTrainOutcome:
    """Explicit outcome of one ``train_g`` call.

    Families whose artifact identity does not reveal whether a parameter update
    occurred should return this wrapper. Legacy families may continue to
    return the artifact directly; the runtime then treats an unchanged artifact
    as a no-op and a changed artifact as an update for backward compatibility.
    """

    artifact: Any
    update_performed: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.update_performed, bool):
            raise TypeError("GTrainOutcome.update_performed must be boolean")


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


def canonical_g_artifact(g_mode: str) -> dict[str, Any]:
    """Materialize the sole package-defined implicit operator: identity g."""

    mode = normalize_g_mode(g_mode)
    if mode != G_MODE_IDENTITY:
        raise ValueError(
            f"g_mode={mode!r} has no implicit canonical artifact; supply "
            "initial_artifacts['g'] explicitly"
        )
    return {
        "kind": "treepo_identity_g",
        "g_mode": G_MODE_IDENTITY,
        "operator": "identity",
        "trainable": False,
        "train_g_enabled": False,
        "merge_call_count": 0,
        "same_g_across_node_roles": True,
        "reduce_g_is_derived": True,
    }


def resolve_g_artifact(g_mode: str, g_init: Any) -> Any:
    """Resolve g, requiring identity correctness or an explicit fixed artifact."""

    mode = normalize_g_mode(g_mode)
    if isinstance(g_init, GTrainOutcome):
        g_init = g_init.artifact
    if mode == G_MODE_LEARNED:
        return g_init
    if mode == G_MODE_IDENTITY:
        if g_init is not None:
            raise ValueError(
                "g_mode='identity' rejects initial_artifacts['g']; omit it so the "
                "package can materialize and own the canonical identity operator"
            )
        return canonical_g_artifact(mode)
    if g_init is None:
        raise ValueError(
            "g_mode='fixed' requires an explicit initial_artifacts['g']; "
            "a fixed label without a concrete operator fails closed"
        )
    if not isinstance(g_init, Mapping):
        raise TypeError("g_mode='fixed' initial_artifacts['g'] must be a mapping")
    required = (
        "kind",
        "g_mode",
        "operator",
        "trainable",
        "same_g_across_node_roles",
        "reduce_g_is_derived",
    )
    missing = [key for key in required if key not in g_init]
    if missing:
        raise ValueError(
            f"g_mode='fixed' artifact is missing required fixed-operator fields {missing!r}"
        )
    if normalize_g_mode(g_init.get("g_mode")) != G_MODE_FIXED:
        raise ValueError("g_mode='fixed' artifact must declare g_mode='fixed'")
    if not str(g_init.get("kind") or "").strip():
        raise ValueError("g_mode='fixed' artifact kind must be non-empty")
    if not str(g_init.get("operator") or "").strip():
        raise ValueError("g_mode='fixed' artifact operator must be non-empty")
    if g_init.get("trainable") is not False:
        raise ValueError("g_mode='fixed' artifact must declare trainable=false")
    if g_init.get("same_g_across_node_roles") is not True:
        raise ValueError("g_mode='fixed' artifact must declare one shared g")
    if g_init.get("reduce_g_is_derived") is not True:
        raise ValueError("g_mode='fixed' artifact must declare reduce_g as a derived fold")
    if g_init.get("train_g_enabled") not in (None, False):
        raise ValueError("g_mode='fixed' artifact cannot enable train_g")
    return g_init


def resolve_g_train_outcome(
    value: Any,
    *,
    previous_artifact: Any,
) -> tuple[Any, bool, str]:
    """Normalize explicit and legacy ``train_g`` return values.

    ``GTrainOutcome`` is authoritative. For legacy direct-artifact returns,
    an unchanged value is conservatively a no-op; a changed value retains the
    historical update interpretation. A family that mutates an artifact
    in-place should return ``GTrainOutcome(..., update_performed=True)``.
    """

    if isinstance(value, GTrainOutcome):
        return (
            value.artifact,
            value.update_performed,
            str(value.reason or "explicit_family_outcome"),
        )
    if _artifacts_equivalent(previous_artifact, value):
        return value, False, "legacy_unchanged_artifact"
    return value, True, "legacy_changed_artifact"


def _artifacts_equivalent(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        comparison = left == right
    except Exception:
        return False
    if isinstance(comparison, bool):
        return comparison
    try:
        return bool(comparison)
    except (TypeError, ValueError):
        return False


def stage_metadata_for_iteration(k: int, g_mode: str) -> tuple[str, str, int, int]:
    mode = normalize_g_mode(g_mode)
    if mode == G_MODE_LEARNED:
        f_degree, g_degree = stage_powers_for_iteration(k)
        return (
            stage_name_for_iteration(k),
            stage_label_for_iteration(k),
            f_degree,
            g_degree,
        )
    f_updates = sum(1 for index in range(k + 1) if trains_f_at_iteration(index))
    f_degree = 1 + f_updates
    stage_name = f"{'f' * f_degree}_g_{mode}"
    return stage_name, f"f^{f_degree} g={mode}", f_degree, 0


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
    reporting_oracle_targets: Sequence[Any] = (),
    g_mode: str = G_MODE_LEARNED,
    topology_contract: Mapping[str, Any] | None = None,
) -> list[IterationRecord]:
    """Run a compact alternating loop over a public ``FamilyRuntime``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = normalize_g_mode(g_mode)
    f_artifact = f_init
    g_artifact = resolve_g_artifact(mode, g_init)
    if mode == G_MODE_FIXED:
        family.validate_artifact(kind="g", artifact=g_artifact)
    trace_list = list(traces or ())
    f_trace_list = list(f_traces) if f_traces is not None else trace_list
    g_trace_list = list(g_traces) if g_traces is not None else trace_list
    if mode != G_MODE_LEARNED and g_traces:
        raise ValueError(f"g_mode={mode!r} cannot consume target='g' preference traces")
    records: list[IterationRecord] = []
    max_k = max(0, int(max_iterations))
    for k in range(max_k + 1):
        if mode != G_MODE_LEARNED and trains_g_at_iteration(k):
            continue
        trained = "none"
        g_update_performed: bool | None = None
        g_update_signal: str | None = None
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
        elif mode == G_MODE_LEARNED and trains_g_at_iteration(k):
            previous_g_artifact = g_artifact
            raw_g_outcome = family.train_g(
                g_init=g_artifact,
                f=f_artifact,
                traces=g_trace_list,
                output_dir=iteration_dir,
                iteration=k,
            )
            g_artifact, g_update_performed, g_update_signal = resolve_g_train_outcome(
                raw_g_outcome,
                previous_artifact=previous_g_artifact,
            )
            family.validate_artifact(kind="g", artifact=g_artifact)
            trained = "g"
        stage_name, stage_label, f_degree, g_degree = stage_metadata_for_iteration(k, mode)
        prediction_rows: list[dict[str, Any]] = []
        split_metrics = evaluate_splits(
            family,
            f_artifact,
            g_artifact,
            eval_trees,
            prediction_rows=prediction_rows,
            reporting_oracle_targets=reporting_oracle_targets,
        )
        extra: dict[str, Any] = {
            "g_mode": mode,
            "topology_contract": dict(topology_contract or {}),
        }
        if trained == "g":
            extra["g_update_performed"] = bool(g_update_performed)
            extra["g_update_signal"] = str(g_update_signal)
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
                stage_name=stage_name,
                stage_label=stage_label,
                family=str(getattr(family, "name", type(family).__name__)),
                trained=trained,
                g_mode=mode,
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
    "GTrainOutcome",
    "canonical_g_artifact",
    "resolve_g_artifact",
    "resolve_g_train_outcome",
    "run_alternating_family",
    "stage_label_for_iteration",
    "stage_metadata_for_iteration",
    "stage_name_for_iteration",
    "stage_powers_for_iteration",
    "statistic_payload",
    "trains_f_at_iteration",
    "trains_g_at_iteration",
]
