"""Unified evidence artifact assembly for treepo fit runs.

The evidence artifact is a compact JSONable view over data that a run already
produces: root metrics, preference exports, statistic metadata, local-law
summaries, and prediction files. It reuses objects the run already produced.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo.local_law import LocalLawAuditRow, audit_local_laws


EVIDENCE_VERSION = "0.1"


def build_evidence(
    *,
    status: str,
    metrics: Mapping[str, Any] | None = None,
    summary: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    local_law_rows: Sequence[LocalLawAuditRow | Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the package-standard evidence view for one run."""

    metric_payload = dict(metrics or {})
    summary_payload = dict(summary or {})
    artifact_payload = dict(artifacts or {})
    preference_artifacts = _mapping(artifact_payload.get("preference_data"))
    statistic_artifact = _mapping(artifact_payload.get("statistic"))
    prediction_files = [str(path) for path in list(artifact_payload.get("prediction_records") or [])]
    local_law_payload = _local_law_payload(
        rows=local_law_rows,
        statistic_artifact=statistic_artifact,
        artifact_payload=artifact_payload,
    )

    evidence = {
        "version": EVIDENCE_VERSION,
        "run": {
            "family": str(summary_payload.get("family") or ""),
            "schedule": str(summary_payload.get("schedule") or ""),
            "status": str(status),
            "n_iterations": int(summary_payload.get("n_iterations") or 0),
            "output_dir": str(summary_payload.get("output_dir") or ""),
        },
        "root": {
            "present": bool(metric_payload or summary_payload.get("split_metrics")),
            "metrics": _jsonable(metric_payload),
            "split_metrics": _jsonable(_mapping(summary_payload.get("split_metrics"))),
        },
        "preferences": {
            "present": bool(preference_artifacts),
            "summary": _jsonable(_mapping(preference_artifacts.get("summary"))),
            "counts": _jsonable(_mapping(preference_artifacts.get("counts"))),
            "files": _jsonable(_mapping(preference_artifacts.get("files"))),
        },
        "statistic": {
            "present": bool(statistic_artifact),
            "info": _jsonable(_mapping(statistic_artifact.get("info"))),
            "local_law_summary": _jsonable(_mapping(statistic_artifact.get("local_law_summary"))),
            "local_law_row_count": int(statistic_artifact.get("local_law_row_count") or 0),
        },
        "local_laws": local_law_payload,
        "predictions": {
            "present": bool(prediction_files),
            "files": prediction_files,
            "file_count": len(prediction_files),
        },
    }
    return _jsonable(evidence)


def _local_law_payload(
    *,
    rows: Sequence[LocalLawAuditRow | Mapping[str, Any]] | None,
    statistic_artifact: Mapping[str, Any],
    artifact_payload: Mapping[str, Any],
) -> dict[str, Any]:
    explicit = _mapping(artifact_payload.get("local_laws"))
    if explicit:
        return {
            "present": True,
            "summary": _jsonable(_mapping(explicit.get("summary", explicit.get("local_law_objective")))),
            "by_law_kind": _jsonable(_mapping(explicit.get("by_law_kind"))),
            "source": str(explicit.get("source") or "artifact"),
        }
    if rows:
        audit = audit_local_laws(rows)
        return {
            "present": True,
            "summary": _jsonable(_mapping(audit.get("local_law_objective"))),
            "by_law_kind": _jsonable(_mapping(audit.get("by_law_kind"))),
            "source": "rows",
        }
    statistic_summary = _mapping(statistic_artifact.get("local_law_summary"))
    if statistic_summary:
        return {
            "present": True,
            "summary": _jsonable(statistic_summary),
            "by_law_kind": {},
            "source": "statistic",
        }
    return {"present": False, "summary": {}, "by_law_kind": {}, "source": "none"}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _jsonable(value: Any) -> Any:
    # Kept local rather than delegating to ``treepo.common.jsonable``: this
    # helper preserves Enum objects as-is and expands dataclasses before
    # ``to_dict``, so it differs from ``treepo.common.jsonable`` on the
    # arbitrary external metrics/artifacts payloads passed to ``build_evidence``.
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    return value


__all__ = ["EVIDENCE_VERSION", "build_evidence"]
