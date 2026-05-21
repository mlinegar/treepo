"""``fit()`` — the single unified entry point for treepo.cld.

Thin orchestrator over :func:`src.ctreepo.alternating.run_alternating_family`.
Backend dispatch is by :class:`FamilyRuntime` protocol; there is no
second signature, no parallel audit/certificate surface, no validator
layer. The audit lives at :func:`treepo.cld.methods.run` with
``method="audit"`` — not as a side-channel on ``fit()``.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo._research.ctreepo.alternating import (
    FamilyRuntime,
    IterationRecord,
    run_alternating_family,
)
from treepo._research.ctreepo.contracts import (
    CTreePOFitResult,
    CTreePOLearningSpec,
    ObjectiveSpec,
)
from treepo.cld.families import resolve_family


_MANIFEST_NAME = "treepo_cld_run_manifest.json"


def fit(spec: CTreePOLearningSpec) -> CTreePOFitResult:
    """Run one alternating f/g ladder described by ``spec``.

    The :class:`FamilyRuntime` is resolved in two steps:

    1. If ``spec.backend_config['family_runtime']`` is set, use it
       directly (the testing / ad-hoc injection path).
    2. Otherwise, ``spec.family`` is looked up in
       :func:`treepo.cld.families.resolve_family` (``"oracle"`` /
       ``"fno"`` / ``"dspy"`` / ``"trl"`` / ``"sketch"`` /
       ``"learnable_constant"``).
    """
    backend_config = dict(spec.backend_config or {})
    axis = dict(spec.axis or {})
    initial = spec.initial_artifacts or {}

    family = _resolve_family(spec, backend_config)
    output_dir = (
        Path(backend_config["output_dir"])
        if backend_config.get("output_dir")
        else Path(tempfile.mkdtemp(prefix="treepo_cld_fit_"))
    )

    records = run_alternating_family(
        family=family,
        f_init=initial.get("f"),
        g_init=initial.get("g"),
        traces=_as_sequence(spec.train_data),
        eval_trees=_as_sequence(spec.eval_data),
        max_iterations=int(axis.get("max_iterations", 0)),
        axis_value=int(axis.get("axis_value", 0)),
        output_dir=output_dir,
        axis_kind=str(axis.get("axis_kind", "leaf_count")),
        leaf_count=_optional_int(axis.get("leaf_count")),
        leaf_size_tokens=_optional_int(axis.get("leaf_size_tokens")),
        first_train_side=str(backend_config.get("first_train_side", "f")),
        initial_f_degree=int(backend_config.get("initial_f_degree", 1)),
        initial_g_degree=int(backend_config.get("initial_g_degree", 1)),
        stage_naming=str(backend_config.get("stage_naming", "legacy")),
        artifact_namer=backend_config.get("artifact_namer"),
    )

    return _build_result(
        spec=spec,
        records=records,
        output_dir=output_dir,
        objective=_resolve_objective(backend_config),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _resolve_family(
    spec: CTreePOLearningSpec, backend_config: Mapping[str, Any]
) -> FamilyRuntime:
    injected = backend_config.get("family_runtime")
    if injected is not None:
        if not isinstance(injected, FamilyRuntime):
            raise TypeError(
                "spec.backend_config['family_runtime'] must implement "
                f"FamilyRuntime; got {type(injected).__name__}"
            )
        return injected
    if not spec.family:
        raise ValueError(
            "spec.family is empty and no family_runtime was injected via "
            "spec.backend_config['family_runtime']. Set one or the other."
        )
    return resolve_family(spec.family, backend_config)


def _as_sequence(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return value
    return tuple(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _resolve_objective(backend_config: Mapping[str, Any]) -> ObjectiveSpec | None:
    """Accept an optional ``ObjectiveSpec`` (or mapping) and record it in
    the run manifest. v1 does NOT fan the spec out to ``family.train_f``
    / ``train_g`` — existing families consume objective knobs via their
    own typed configs (FNOFamilyConfig, etc.).
    """
    raw = backend_config.get("objective")
    if raw is None:
        return None
    if isinstance(raw, ObjectiveSpec):
        return raw
    if isinstance(raw, Mapping):
        return ObjectiveSpec(**dict(raw))
    raise TypeError(
        "backend_config['objective'] must be an ObjectiveSpec or mapping; "
        f"got {type(raw).__name__}"
    )


def _build_result(
    *,
    spec: CTreePOLearningSpec,
    records: Sequence[IterationRecord],
    output_dir: Path,
    objective: ObjectiveSpec | None,
) -> CTreePOFitResult:
    last = records[-1] if records else None
    error = (last.extra or {}).get("error") if last is not None else None
    status = "failed" if error else "success"

    metrics = _final_metrics(last)
    artifacts: dict[str, Any] = (
        {"f": last.f_artifact, "g": last.g_artifact} if last is not None else {}
    )
    prediction_records = _collect_prediction_records(output_dir)
    if prediction_records:
        artifacts["prediction_records"] = prediction_records
    history = [dataclasses.asdict(r) for r in records]

    summary: dict[str, Any] = {
        "family": str(spec.family),
        "schedule": str(spec.schedule),
        "n_iterations": len(records),
        "output_dir": str(output_dir),
    }
    split_metrics = _split_metrics_payload(last)
    if split_metrics:
        summary["split_metrics"] = split_metrics
    if last is not None:
        summary["final_stage"] = last.stage_name
        summary["final_stage_label"] = last.stage_label
    if objective is not None:
        summary["objective"] = (
            objective.to_dict()
            if hasattr(objective, "to_dict")
            else dataclasses.asdict(objective)
        )

    manifest_path = _write_manifest(
        spec=spec,
        records=records,
        output_dir=output_dir,
        objective=objective,
        status=status,
        metrics=metrics,
        summary=summary,
    )
    return CTreePOFitResult(
        status=status,
        metrics=metrics,
        artifacts=artifacts,
        history=history,
        summary=summary,
        manifest_path=str(manifest_path) if manifest_path is not None else None,
    )


def _write_manifest(
    *,
    spec: CTreePOLearningSpec,
    records: Sequence[IterationRecord],
    output_dir: Path,
    objective: ObjectiveSpec | None,
    status: str,
    metrics: Mapping[str, float],
    summary: Mapping[str, Any],
) -> Path | None:
    """JSON sidecar at ``output_dir/treepo_cld_run_manifest.json`` —
    same shape as paper-script manifests.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    manifest_path = output_dir / _MANIFEST_NAME
    payload: dict[str, Any] = {
        "status": str(status),
        "spec": {
            "space_kind": str(spec.space_kind),
            "family": str(spec.family),
            "schedule": str(spec.schedule),
            "initial_artifacts": dict(spec.initial_artifacts or {}),
            "axis": dict(spec.axis or {}),
            # backend_config may carry non-JSON-serializable instances.
            "backend_config_keys": sorted((spec.backend_config or {}).keys()),
        },
        "objective": (
            objective.to_dict() if (objective is not None and hasattr(objective, "to_dict"))
            else (dataclasses.asdict(objective) if objective is not None else None)
        ),
        "summary": dict(summary),
        "metrics": dict(metrics),
        "n_iterations": len(records),
    }
    try:
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    except OSError:
        return None
    return manifest_path


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            pass
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return str(value)


_METRIC_FIELDS: tuple[str, ...] = (
    "internal_f_pearson",
    "internal_f_mae",
    "internal_f_mae_1_7",
    "external_expert_pearson",
    "external_expert_mae",
    "external_expert_mae_1_7",
    "f_star_gap",
    "mean_prediction",
    "mean_teacher",
    "mean_expert",
)

_SPLIT_ORDER: tuple[str, ...] = ("all", "train", "val", "test")


def _final_metrics(record: IterationRecord | None) -> Mapping[str, float]:
    """Flatten every available split into a single ``{prefix}{field}`` dict.

    ``"all"`` keys stay unprefixed (so existing callers / grid CSVs keep
    working). Per-split keys are prefixed (``test_internal_f_mae``,
    ``train_f_star_gap``, ...) so a grid loop can extract train/test
    generalization gaps with one CSV column per cell.
    """
    if record is None or not record.split_metrics:
        return {}
    out: dict[str, float] = {}
    for split_name in _SPLIT_ORDER:
        sm = record.split_metrics.get(split_name)
        if sm is None:
            continue
        prefix = "" if split_name == "all" else f"{split_name}_"
        for field_name in _METRIC_FIELDS:
            value = getattr(sm, field_name, None)
            if value is not None:
                out[f"{prefix}{field_name}"] = float(value)
        out[f"{prefix}n"] = float(getattr(sm, "n", 0))
    return out


def _split_metrics_payload(record: IterationRecord | None) -> Mapping[str, Any]:
    """Structured ``summary['split_metrics']`` carrying every per-split
    field including ``per_dimension`` breakdowns (vector families) and
    raw ``mean_*`` calibration anchors.
    """
    if record is None or not record.split_metrics:
        return {}
    out: dict[str, Any] = {}
    for split_name, sm in record.split_metrics.items():
        entry: dict[str, Any] = {"n": int(getattr(sm, "n", 0))}
        for field_name in _METRIC_FIELDS:
            value = getattr(sm, field_name, None)
            if value is not None:
                entry[field_name] = float(value)
        per_dimension = getattr(sm, "per_dimension", None) or {}
        if per_dimension:
            entry["per_dimension"] = {
                str(dim): {str(k): (float(v) if v is not None else None) for k, v in dim_metrics.items()}
                for dim, dim_metrics in per_dimension.items()
            }
        out[str(split_name)] = entry
    return out


def _collect_prediction_records(output_dir: Path) -> list[str]:
    """Return paths to per-iteration prediction-record JSONL files
    written by :func:`evaluate_iteration`. One per iteration.
    """
    pred_dir = output_dir / "prediction_records"
    if not pred_dir.exists():
        return []
    return sorted(str(p) for p in pred_dir.glob("iter_*_post_eval.jsonl"))


__all__ = ["fit"]
