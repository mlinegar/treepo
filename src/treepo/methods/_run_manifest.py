"""Methods run manifest writer."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo.state import state_to_dict

MANIFEST_NAME = "treepo_methods_run_manifest.json"


def write_manifest(
    *,
    spec: Any,
    records: Sequence[Any],
    output_dir: Path,
    objective: Any | None,
    status: str,
    metrics: Mapping[str, float],
    summary: Mapping[str, Any],
    preference_artifacts: Mapping[str, Any],
) -> Path | None:
    """Write the methods JSON sidecar for a run."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    manifest_path = output_dir / MANIFEST_NAME
    payload: dict[str, Any] = {
        "status": str(status),
        "spec": {
            "space_kind": str(spec.space_kind),
            "family": str(spec.family or ""),
            "schedule": str(spec.schedule),
            "initial_artifacts": dict(spec.initial_artifacts or {}),
            "axis": dict(spec.axis or {}),
            "has_preference_data": bool(getattr(spec, "preference_data", None)),
            # backend_config may carry non-JSON-serializable instances.
            "backend_config_keys": sorted((spec.backend_config or {}).keys()),
        },
        "objective": (
            objective.to_dict()
            if (objective is not None and hasattr(objective, "to_dict"))
            else (dataclasses.asdict(objective) if objective is not None else None)
        ),
        "summary": dict(summary),
        "metrics": dict(metrics),
        "preference_data": dict(preference_artifacts or {}),
        "n_iterations": len(records),
    }
    try:
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=json_default)
        )
    except OSError:
        return None
    return manifest_path


def json_default(value: Any) -> Any:
    state_value = state_to_dict(value)
    if state_value is not value:
        return state_value
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            pass
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return str(value)


__all__ = ["MANIFEST_NAME", "json_default", "write_manifest"]
