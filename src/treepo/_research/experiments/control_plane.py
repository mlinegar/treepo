from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from treepo._research.experiments.contracts import ArtifactRef, ExperimentSpec, ProgressSnapshot, ResultRow


EXPERIMENT_MANIFEST_NAME = "experiment_manifest.json"
EXPERIMENT_STATUS_NAME = "experiment_status.json"
EVENT_LOG_NAME = "event_log.jsonl"
ARTIFACTS_NAME = "artifacts.json"
RESULTS_NAME = "results.jsonl"


def experiment_paths(output_root: str | Path) -> Dict[str, Path]:
    root = Path(output_root).expanduser().resolve()
    return {
        "output_root": root,
        "manifest": root / EXPERIMENT_MANIFEST_NAME,
        "status": root / EXPERIMENT_STATUS_NAME,
        "event_log": root / EVENT_LOG_NAME,
        "artifacts": root / ARTIFACTS_NAME,
        "results": root / RESULTS_NAME,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payloads: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(dict(payload), sort_keys=False) + "\n")
        handle.flush()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_experiment_manifest(
    output_root: str | Path,
    spec: ExperimentSpec,
) -> Path:
    paths = experiment_paths(output_root)
    _write_json(paths["manifest"], spec.to_dict())
    return paths["manifest"]


def write_experiment_status(
    output_root: str | Path,
    snapshot: ProgressSnapshot,
    *,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    paths = experiment_paths(output_root)
    payload = snapshot.to_dict()
    if extra:
        payload.update(dict(extra))
    _write_json(paths["status"], payload)
    return paths["status"]


def merge_artifacts(
    output_root: str | Path,
    artifacts: Sequence[ArtifactRef] | Mapping[str, Any],
) -> Path:
    paths = experiment_paths(output_root)
    existing = load_json(paths["artifacts"])
    current_entries = dict(existing.get("artifacts", {}) or {})
    if isinstance(artifacts, Mapping):
        for key, value in dict(artifacts).items():
            current_entries[str(key)] = value
    else:
        for artifact in artifacts:
            current_entries[str(artifact.artifact_id)] = artifact.to_dict()
    payload = {
        "output_root": str(paths["output_root"]),
        "artifacts": current_entries,
    }
    _write_json(paths["artifacts"], payload)
    return paths["artifacts"]


def append_result_rows(
    output_root: str | Path,
    rows: Sequence[ResultRow] | Sequence[Mapping[str, Any]],
) -> Path:
    paths = experiment_paths(output_root)
    payloads = [
        row.to_dict() if isinstance(row, ResultRow) else dict(row)
        for row in rows
    ]
    if payloads:
        _append_jsonl(paths["results"], payloads)
    else:
        paths["results"].parent.mkdir(parents=True, exist_ok=True)
        paths["results"].touch(exist_ok=True)
    return paths["results"]


def progress_snapshot_from_scheduler_payload(
    payload: Mapping[str, Any],
    *,
    experiment_id: str,
    artifact_targets: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ProgressSnapshot:
    phase_progress = dict(payload.get("phase_progress") or {})
    active_phase = ""
    for phase_name, phase_payload in phase_progress.items():
        phase_state = dict(phase_payload or {})
        if int(phase_state.get("active", 0) or 0) > 0:
            active_phase = str(phase_name)
            break
    if not active_phase and phase_progress:
        first_key = next(iter(phase_progress))
        active_phase = str(first_key)
    return ProgressSnapshot(
        experiment_id=str(experiment_id),
        state=str(payload.get("state", "") or ""),
        active_phase=active_phase,
        items_total=int(payload.get("items_total", 0) or 0),
        completed_items=int(payload.get("completed_items", 0) or 0),
        failed_items=int(payload.get("failed_items", 0) or 0),
        active_items=int(payload.get("active_items", 0) or 0),
        pending_items=int(payload.get("pending_items", 0) or 0),
        percent_complete=float(payload.get("percent_complete", 0.0) or 0.0),
        artifact_targets=tuple(str(item) for item in artifact_targets),
        live_child_status_path=str(payload.get("status_path", "") or ""),
        metadata=dict(metadata or {}),
    )


def canonical_artifact_refs_from_paths(
    path_map: Mapping[str, Any],
    *,
    phase_id: str = "",
    required: bool = False,
) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    for key, value in dict(path_map).items():
        text = str(value or "").strip()
        if not text:
            continue
        refs.append(
            ArtifactRef(
                artifact_id=str(key),
                artifact_type=str(key),
                path=text,
                phase_id=str(phase_id),
                required=bool(required),
            )
        )
    return refs
