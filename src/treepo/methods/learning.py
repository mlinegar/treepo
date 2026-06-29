"""``fit()`` — the single unified entry point for treepo.methods."""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo.methods.contracts import CTreePOFitResult, ObjectiveSpec
from treepo.methods.families import resolve_family
from treepo.methods.estimators import EstimatorDescriptor, resolve_estimator
from treepo.methods.runtime import run_alternating_family


_MANIFEST_NAME = "treepo_methods_run_manifest.json"


def fit(spec: Any) -> Any:
    """Run one alternating f/g ladder described by ``spec``.

    The :class:`FamilyRuntime` is resolved in two steps:

    1. If ``spec.backend_config['family_runtime']`` is set, use it
       directly (the testing / ad-hoc injection path).
    2. Otherwise, ``spec.family`` is looked up in
       :func:`treepo.methods.families.resolve_family` (``"oracle"`` /
       ``"fno"`` / ``"dspy"`` / ``"trl"`` / ``"learnable_constant"``).
    """
    backend_config = dict(spec.backend_config or {})
    axis = dict(spec.axis or {})
    initial = spec.initial_artifacts or {}
    estimator = _resolve_estimator_descriptor(spec, backend_config)
    if estimator is not None:
        backend_config = estimator.apply_to_backend_config(backend_config)

    family = _resolve_family(spec, backend_config, estimator=estimator)

    output_dir = (
        Path(backend_config["output_dir"])
        if backend_config.get("output_dir")
        else Path(tempfile.mkdtemp(prefix="treepo_methods_fit_"))
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
    )

    return _build_result(
        spec=spec,
        records=records,
        output_dir=output_dir,
        objective=_resolve_objective(backend_config),
        estimator=estimator,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _resolve_estimator_descriptor(
    spec: Any, backend_config: Mapping[str, Any]
) -> EstimatorDescriptor | None:
    raw = getattr(spec, "estimator", None)
    if raw is None:
        raw = getattr(spec, "g_estimator", None)
    if raw is None:
        raw = backend_config.get("estimator")
    if raw is None:
        raw = backend_config.get("g_estimator")
    if raw is None:
        return None
    return resolve_estimator(raw, backend_config)


def _resolve_family(
    spec: Any,
    backend_config: Mapping[str, Any],
    *,
    estimator: EstimatorDescriptor | None = None,
) -> Any:
    injected = backend_config.get("family_runtime")
    if injected is not None:
        if not _implements_family_runtime(injected):
            raise TypeError(
                "spec.backend_config['family_runtime'] must implement "
                f"FamilyRuntime; got {type(injected).__name__}"
            )
        return injected
    family_name = str(getattr(spec, "family", "") or "")
    if not family_name and estimator is not None:
        family_name = str(estimator.family)
    if not family_name:
        raise ValueError(
            "spec.family is empty and no family_runtime or estimator was supplied. "
            "Set spec.family, spec.backend_config['family_runtime'], spec.estimator, or spec.g_estimator."
        )
    return resolve_family(family_name, backend_config)


def _implements_family_runtime(value: Any) -> bool:
    required = (
        "train_f",
        "train_g",
        "score_roots_with_f",
        "validate_artifact",
    )
    return all(callable(getattr(value, name, None)) for name in required)


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


def _resolve_objective(backend_config: Mapping[str, Any]) -> Any | None:
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
    spec: Any,
    records: Sequence[Any],
    output_dir: Path,
    objective: Any | None,
    estimator: EstimatorDescriptor | None,
) -> Any:
    last = records[-1] if records else None
    error = (last.extra or {}).get("error") if last is not None else None
    status = "failed" if error else "success"

    metrics = _final_metrics(last)
    artifacts: dict[str, Any] = (
        {"f": last.f_artifact, "g": last.g_artifact} if last is not None else {}
    )
    _write_prediction_records(output_dir, records)
    prediction_records = _collect_prediction_records(output_dir)
    if prediction_records:
        artifacts["prediction_records"] = prediction_records
    history = [dataclasses.asdict(r) for r in records]

    summary: dict[str, Any] = {
        "family": str(spec.family or (estimator.family if estimator is not None else "")),
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
    if estimator is not None:
        estimator_payload = estimator.to_dict()
        summary["estimator"] = estimator_payload
        if estimator_payload.get("target") == "g":
            summary["g_estimator"] = estimator_payload
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
        estimator=estimator,
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
    spec: Any,
    records: Sequence[Any],
    output_dir: Path,
    objective: Any | None,
    estimator: EstimatorDescriptor | None,
    status: str,
    metrics: Mapping[str, float],
    summary: Mapping[str, Any],
) -> Path | None:
    """Write the methods JSON sidecar for a run."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    manifest_path = output_dir / _MANIFEST_NAME
    payload: dict[str, Any] = {
        "status": str(status),
        "spec": {
            "space_kind": str(spec.space_kind),
            "family": str(spec.family or (estimator.family if estimator is not None else "")),
            "schedule": str(spec.schedule),
            "initial_artifacts": dict(spec.initial_artifacts or {}),
            "axis": dict(spec.axis or {}),
            "estimator": estimator.to_dict() if estimator is not None else getattr(spec, "estimator", None),
            "g_estimator": (
                estimator.to_dict()
                if estimator is not None and estimator.to_dict().get("target") == "g"
                else getattr(spec, "g_estimator", None)
            ),
            # backend_config may carry non-JSON-serializable instances.
            "backend_config_keys": sorted((spec.backend_config or {}).keys()),
        },
        "estimator": estimator.to_dict() if estimator is not None else None,
        "g_estimator": (
            estimator.to_dict()
            if estimator is not None and estimator.to_dict().get("target") == "g"
            else None
        ),
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


def _final_metrics(record: Any | None) -> Mapping[str, float]:
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
        for dim_name, dim_metrics in (getattr(sm, "per_dimension", None) or {}).items():
            for field_name, value in dict(dim_metrics or {}).items():
                if value is not None:
                    out[f"{prefix}{dim_name}_{field_name}"] = float(value)
        out[f"{prefix}n"] = float(getattr(sm, "n", 0))
    return out


def _split_metrics_payload(record: Any | None) -> Mapping[str, Any]:
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


def _write_prediction_records(output_dir: Path, records: Sequence[Any]) -> None:
    pred_dir = output_dir / "prediction_records"
    wrote_any = False
    for record in records:
        rows = (getattr(record, "extra", None) or {}).get("prediction_rows") or []
        if not rows:
            continue
        pred_dir.mkdir(parents=True, exist_ok=True)
        path = pred_dir / f"iter_{int(record.iteration):02d}_post_eval.jsonl"
        enriched = []
        for row in rows:
            payload = dict(row)
            payload.setdefault("iteration", int(record.iteration))
            payload.setdefault("stage_name", str(record.stage_name))
            payload.setdefault("family", str(record.family))
            enriched.append(payload)
        path.write_text(
            "\n".join(json.dumps(row, sort_keys=True, default=_json_default) for row in enriched)
            + "\n",
            encoding="utf-8",
        )
        wrote_any = True
    if not wrote_any:
        return


__all__ = ["fit"]
