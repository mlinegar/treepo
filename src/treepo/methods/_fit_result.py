"""FitResult assembly and metric payload helpers."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo.evidence import build_evidence
from treepo.methods._run_manifest import json_default, write_manifest
from treepo.methods.contracts import FitResult
from treepo.methods.preference import PreferenceDataset, export_preference_records


def build_result(
    *,
    spec: Any,
    records: Sequence[Any],
    output_dir: Path,
    objective: Any | None,
    preference_dataset: PreferenceDataset,
    grid_axes: Mapping[str, Any] | None = None,
    supervision: Mapping[str, Any] | None = None,
) -> Any:
    last = records[-1] if records else None
    error = (last.extra or {}).get("error") if last is not None else None
    status = "failed" if error else "success"

    metrics = final_metrics(last)
    artifacts: dict[str, Any] = (
        {"f": last.f_artifact, "g": last.g_artifact} if last is not None else {}
    )
    write_prediction_records(output_dir, records)
    prediction_records = collect_prediction_records(output_dir)
    if prediction_records:
        artifacts["prediction_records"] = prediction_records
    statistic_artifact = (last.extra or {}).get("statistic") if last is not None else None
    if statistic_artifact:
        artifacts["statistic"] = statistic_artifact
    preference_artifacts = (
        export_preference_records(preference_dataset, output_dir / "preference")
        if len(preference_dataset) > 0
        else {}
    )
    if preference_artifacts:
        artifacts["preference_data"] = preference_artifacts
    history = [dataclasses.asdict(r) for r in records]

    summary: dict[str, Any] = {
        "family": str(spec.family or ""),
        "schedule": str(spec.schedule),
        "n_iterations": len(records),
        "output_dir": str(output_dir),
    }
    if grid_axes:
        summary["grid_axes"] = dict(grid_axes)
        artifacts["grid_axes"] = dict(grid_axes)
    if supervision:
        summary["supervision"] = dict(supervision)
    split_metrics = split_metrics_payload(last)
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
    if preference_artifacts:
        summary["preference_data"] = preference_artifacts["summary"]
    if statistic_artifact:
        summary["statistic"] = dict(statistic_artifact.get("info") or {})
        if statistic_artifact.get("local_law_summary"):
            summary["statistic_local_law"] = dict(statistic_artifact["local_law_summary"])
    artifacts["evidence"] = build_evidence(
        status=status,
        metrics=metrics,
        summary=summary,
        artifacts=artifacts,
    )

    manifest_path = write_manifest(
        spec=spec,
        records=records,
        output_dir=output_dir,
        objective=objective,
        status=status,
        metrics=metrics,
        summary=summary,
        preference_artifacts=preference_artifacts,
    )
    return FitResult(
        status=status,
        metrics=metrics,
        artifacts=artifacts,
        history=history,
        summary=summary,
        manifest_path=str(manifest_path) if manifest_path is not None else None,
    )


METRIC_FIELDS: tuple[str, ...] = (
    "internal_f_pearson",
    "internal_f_mae",
    "external_expert_pearson",
    "external_expert_mae",
    "f_star_gap",
    "mean_prediction",
    "mean_teacher",
    "mean_expert",
)

SPLIT_ORDER: tuple[str, ...] = ("all", "train", "val", "test")


def final_metrics(record: Any | None) -> Mapping[str, float]:
    """Flatten every available split into a single metric dict."""
    if record is None or not record.split_metrics:
        return {}
    out: dict[str, float] = {}
    for split_name in SPLIT_ORDER:
        sm = record.split_metrics.get(split_name)
        if sm is None:
            continue
        prefix = "" if split_name == "all" else f"{split_name}_"
        for field_name in METRIC_FIELDS:
            value = getattr(sm, field_name, None)
            if value is not None:
                out[f"{prefix}{field_name}"] = float(value)
        for dim_name, dim_metrics in (getattr(sm, "per_dimension", None) or {}).items():
            for field_name, value in dict(dim_metrics or {}).items():
                if value is not None:
                    out[f"{prefix}{dim_name}_{field_name}"] = float(value)
        out[f"{prefix}n"] = float(getattr(sm, "n", 0))
    return out


def split_metrics_payload(record: Any | None) -> Mapping[str, Any]:
    """Structured ``summary['split_metrics']`` carrying every per-split field."""
    if record is None or not record.split_metrics:
        return {}
    out: dict[str, Any] = {}
    for split_name, sm in record.split_metrics.items():
        entry: dict[str, Any] = {"n": int(getattr(sm, "n", 0))}
        for field_name in METRIC_FIELDS:
            value = getattr(sm, field_name, None)
            if value is not None:
                entry[field_name] = float(value)
        per_dimension = getattr(sm, "per_dimension", None) or {}
        if per_dimension:
            entry["per_dimension"] = {
                str(dim): {
                    str(k): (float(v) if v is not None else None)
                    for k, v in dim_metrics.items()
                }
                for dim, dim_metrics in per_dimension.items()
            }
        out[str(split_name)] = entry
    return out


def collect_prediction_records(output_dir: Path) -> list[str]:
    """Return per-iteration prediction-record JSONL paths."""
    pred_dir = output_dir / "prediction_records"
    if not pred_dir.exists():
        return []
    return sorted(str(p) for p in pred_dir.glob("iter_*_post_eval.jsonl"))


def write_prediction_records(output_dir: Path, records: Sequence[Any]) -> None:
    pred_dir = output_dir / "prediction_records"
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
            "\n".join(json.dumps(row, sort_keys=True, default=json_default) for row in enriched)
            + "\n",
            encoding="utf-8",
        )


__all__ = [
    "build_result",
    "collect_prediction_records",
    "final_metrics",
    "split_metrics_payload",
    "write_prediction_records",
]
