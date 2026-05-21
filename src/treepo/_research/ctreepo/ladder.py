"""Generic artifact-threading ladder for C-TreePO component training.

The ladder runner is intentionally backend-agnostic: a stage trains exactly
one named component, returns that component's new artifact, and may update
shared-interface artifacts such as a leaf adapter.  The runner records every
stage's input artifacts and latest output artifacts in `ladder_manifest.json`
so a later call can continue from the exact previous state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from treepo._research.ctreepo.contracts import jsonable

SCHEMA_VERSION = 1
DEFAULT_MANIFEST_NAME = "ladder_manifest.json"


@dataclass(frozen=True)
class LadderStageContext:
    """Inputs supplied to one component-training stage."""

    index: int
    component: str
    stage_dir: Path
    component_artifacts: Mapping[str, Any]
    shared_artifacts: Mapping[str, Any]


@dataclass(frozen=True)
class LadderStageOutput:
    """Outputs returned by one component-training stage."""

    component_artifact: Any
    shared_artifacts: Mapping[str, Any] = field(default_factory=dict)
    result: Any = None
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LadderStageRecord:
    """Recorded input/output state for one ladder stage."""

    index: int
    component: str
    stage_dir: Path
    input_component_artifacts: Mapping[str, Any]
    input_shared_artifacts: Mapping[str, Any]
    output_component_artifact: Any
    output_shared_artifacts: Mapping[str, Any]
    result: Any = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "component": str(self.component),
            "stage_dir": str(self.stage_dir),
            "input_component_artifacts": jsonable(
                dict(self.input_component_artifacts or {})
            ),
            "input_shared_artifacts": jsonable(dict(self.input_shared_artifacts or {})),
            "output_component_artifact": jsonable(self.output_component_artifact),
            "output_shared_artifacts": jsonable(
                dict(self.output_shared_artifacts or {})
            ),
            "result": jsonable(self.result),
            "metrics": jsonable(dict(self.metrics or {})),
        }


@dataclass(frozen=True)
class LadderResult:
    """Final artifacts and per-stage records from a ladder run."""

    schedule: tuple[str, ...]
    component_artifacts: Mapping[str, Any]
    shared_artifacts: Mapping[str, Any]
    stages: tuple[LadderStageRecord, ...]
    manifest_path: Path
    status: str = "completed"

    @property
    def artifacts(self) -> Mapping[str, Any]:
        return self.component_artifacts

    @property
    def final_artifact(self) -> Any:
        if not self.stages:
            return None
        return self.stages[-1].output_component_artifact

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "schedule": list(self.schedule),
            "component_artifacts": jsonable(dict(self.component_artifacts or {})),
            "shared_artifacts": jsonable(dict(self.shared_artifacts or {})),
            "stages": [stage.to_dict() for stage in self.stages],
            "manifest_path": str(self.manifest_path),
        }


StageTrainFn = Callable[[LadderStageContext], LadderStageOutput]


def _normalize_schedule(schedule: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(schedule, str):
        components = tuple(str(ch) for ch in schedule)
    else:
        components = tuple(str(ch) for ch in schedule)
    if not components or any(not str(ch) for ch in components):
        raise ValueError("ladder schedule must be a non-empty sequence")
    return components


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(jsonable(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _manifest_payload(
    *,
    output_dir: Path,
    manifest_path: Path,
    schedule: Sequence[str],
    component_artifacts: Mapping[str, Any],
    shared_artifacts: Mapping[str, Any],
    stages: Sequence[LadderStageRecord],
    status: str,
    previous_manifest: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": str(status),
        "schedule": list(schedule),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "previous_manifest": str(previous_manifest) if previous_manifest else None,
        "component_artifacts": jsonable(dict(component_artifacts or {})),
        "shared_artifacts": jsonable(dict(shared_artifacts or {})),
        "stages": [stage.to_dict() for stage in stages],
        "metadata": jsonable(dict(metadata or {})),
    }


def write_ladder_manifest(
    path: Path,
    *,
    output_dir: Path,
    schedule: Sequence[str],
    component_artifacts: Mapping[str, Any],
    shared_artifacts: Mapping[str, Any],
    stages: Sequence[LadderStageRecord],
    status: str = "completed",
    previous_manifest: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    payload = _manifest_payload(
        output_dir=Path(output_dir),
        manifest_path=Path(path),
        schedule=schedule,
        component_artifacts=component_artifacts,
        shared_artifacts=shared_artifacts,
        stages=stages,
        status=status,
        previous_manifest=previous_manifest,
        metadata=metadata,
    )
    _atomic_write_json(Path(path), payload)
    return Path(path)


def load_ladder_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported ladder manifest schema_version={payload.get('schema_version')!r}"
        )
    return payload


def run_component_ladder(
    *,
    schedule: str | Sequence[str],
    output_dir: str | Path,
    train_stage: StageTrainFn,
    initial_component_artifacts: Mapping[str, Any] | None = None,
    initial_shared_artifacts: Mapping[str, Any] | None = None,
    allowed_components: set[str] | frozenset[str] | None = None,
    stage_dir_name: Callable[[int, str], str] | None = None,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
    previous_manifest: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LadderResult:
    """Run a backend-agnostic component ladder and persist a manifest."""

    components = _normalize_schedule(schedule)
    allowed = (
        set(str(item) for item in allowed_components)
        if allowed_components is not None
        else None
    )
    if allowed is not None:
        invalid = sorted(set(components) - allowed)
        if invalid:
            raise ValueError(
                f"ladder schedule contains unsupported components {invalid}; "
                f"allowed={sorted(allowed)}"
            )

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / str(manifest_name)
    component_artifacts: dict[str, Any] = dict(initial_component_artifacts or {})
    shared_artifacts: dict[str, Any] = dict(initial_shared_artifacts or {})
    records: list[LadderStageRecord] = []
    namer = stage_dir_name or (lambda i, comp: f"stage_{i}_{comp}")

    write_ladder_manifest(
        manifest_path,
        output_dir=root,
        schedule=components,
        component_artifacts=component_artifacts,
        shared_artifacts=shared_artifacts,
        stages=records,
        status="running",
        previous_manifest=previous_manifest,
        metadata=metadata,
    )

    for index, component in enumerate(components):
        stage_dir = root / str(namer(int(index), str(component)))
        stage_dir.mkdir(parents=True, exist_ok=True)
        input_components = dict(component_artifacts)
        input_shared = dict(shared_artifacts)
        context = LadderStageContext(
            index=int(index),
            component=str(component),
            stage_dir=stage_dir,
            component_artifacts=input_components,
            shared_artifacts=input_shared,
        )
        output = train_stage(context)
        component_artifacts[str(component)] = output.component_artifact
        shared_artifacts.update(dict(output.shared_artifacts or {}))
        records.append(
            LadderStageRecord(
                index=int(index),
                component=str(component),
                stage_dir=stage_dir,
                input_component_artifacts=input_components,
                input_shared_artifacts=input_shared,
                output_component_artifact=output.component_artifact,
                output_shared_artifacts=dict(output.shared_artifacts or {}),
                result=output.result,
                metrics=dict(output.metrics or {}),
            )
        )
        write_ladder_manifest(
            manifest_path,
            output_dir=root,
            schedule=components,
            component_artifacts=component_artifacts,
            shared_artifacts=shared_artifacts,
            stages=records,
            status="running",
            previous_manifest=previous_manifest,
            metadata=metadata,
        )

    write_ladder_manifest(
        manifest_path,
        output_dir=root,
        schedule=components,
        component_artifacts=component_artifacts,
        shared_artifacts=shared_artifacts,
        stages=records,
        status="completed",
        previous_manifest=previous_manifest,
        metadata=metadata,
    )
    return LadderResult(
        schedule=components,
        component_artifacts=component_artifacts,
        shared_artifacts=shared_artifacts,
        stages=tuple(records),
        manifest_path=manifest_path,
        status="completed",
    )


def continue_ladder(
    *,
    previous_manifest: str | Path,
    schedule: str | Sequence[str],
    output_dir: str | Path,
    train_stage: StageTrainFn,
    allowed_components: set[str] | frozenset[str] | None = None,
    stage_dir_name: Callable[[int, str], str] | None = None,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
    metadata: Mapping[str, Any] | None = None,
) -> LadderResult:
    """Continue a ladder from a previous `ladder_manifest.json`."""

    manifest = load_ladder_manifest(previous_manifest)
    merged_metadata = dict(manifest.get("metadata") or {})
    merged_metadata.update(dict(metadata or {}))
    return run_component_ladder(
        schedule=schedule,
        output_dir=output_dir,
        train_stage=train_stage,
        initial_component_artifacts=dict(manifest.get("component_artifacts") or {}),
        initial_shared_artifacts=dict(manifest.get("shared_artifacts") or {}),
        allowed_components=allowed_components,
        stage_dir_name=stage_dir_name,
        manifest_name=manifest_name,
        previous_manifest=previous_manifest,
        metadata=merged_metadata,
    )


__all__ = [
    "DEFAULT_MANIFEST_NAME",
    "LadderResult",
    "LadderStageContext",
    "LadderStageOutput",
    "LadderStageRecord",
    "StageTrainFn",
    "continue_ladder",
    "load_ladder_manifest",
    "run_component_ladder",
    "write_ladder_manifest",
]
