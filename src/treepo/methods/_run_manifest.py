"""Methods run manifest writer."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo.common import stable_digest
from treepo.state import state_to_dict

MANIFEST_NAME = "treepo_methods_run_manifest.json"
JOINT_TARGET_DEFINITION = "single_shared_g_joint_vector_f_star"


def joint_target_schema(spec: Any) -> dict[str, Any] | None:
    """Return the canonical provenance block for a named joint-vector fit."""

    targets = tuple(getattr(spec, "oracle_targets", ()) or ())
    if not targets:
        return None
    target_schema = [
        (
            dict(target.to_dict())
            if hasattr(target, "to_dict")
            else {
                "target_name": str(getattr(target, "target_name", "")),
                "oracle_id": str(getattr(target, "oracle_id", "")),
                "metadata": dict(getattr(target, "metadata", {}) or {}),
            }
        )
        for target in targets
    ]
    payload: dict[str, Any] = {
        "definition": JOINT_TARGET_DEFINITION,
        "target_order": [str(target["target_name"]) for target in target_schema],
        "oracle_ids_by_target": {
            str(target["target_name"]): str(target["oracle_id"]) for target in target_schema
        },
        "target_schema": target_schema,
    }
    payload["target_schema_digest"] = stable_digest(payload)
    return payload


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
    joint_schema = joint_target_schema(spec)
    spec_payload: dict[str, Any] = {
        "space_kind": str(spec.space_kind),
        "family": str(spec.family or ""),
        "schedule": str(spec.schedule),
        "g_mode": str(getattr(spec, "g_mode", "undeclared")),
        "initial_artifacts": dict(spec.initial_artifacts or {}),
        "axis": dict(spec.axis or {}),
        "has_preference_data": bool(getattr(spec, "preference_data", None)),
        # backend_config may carry non-JSON-serializable instances.
        "backend_config_keys": sorted((spec.backend_config or {}).keys()),
        "doc_gold_n": getattr(spec, "doc_gold_n", None),
        "root_observed_doc_ids": (
            None
            if getattr(spec, "root_observed_doc_ids", None) is None
            else [str(value) for value in getattr(spec, "root_observed_doc_ids", ())]
        ),
        "local_label_mix": str(getattr(spec, "local_label_mix", "none")),
        "gold_fraction_p": float(getattr(spec, "gold_fraction_p", 1.0)),
        "distilled_labels_path": getattr(spec, "distilled_labels_path", None),
        "seed": int(getattr(spec, "seed", 0) or 0),
    }
    if joint_schema is not None:
        spec_payload["oracle_targets"] = list(joint_schema["target_schema"])

    payload: dict[str, Any] = {
        "status": str(status),
        "spec": spec_payload,
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
    if joint_schema is not None:
        payload.update(joint_schema)
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


__all__ = [
    "JOINT_TARGET_DEFINITION",
    "MANIFEST_NAME",
    "joint_target_schema",
    "json_default",
    "write_manifest",
]
