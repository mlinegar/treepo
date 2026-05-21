"""Generic alternating component ladder.

This module is deliberately artifact-agnostic.  A ladder stage trains one
component while all other components are held fixed; callers own the concrete
model construction, optimizer, artifact format, and validation semantics.
The runner only threads component artifacts and shared-interface artifacts
through a schedule such as ``"fgfg"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class ComponentLadderStageContext:
    """Inputs supplied to one alternating stage."""

    index: int
    component: str
    stage_dir: Path
    component_artifacts: Mapping[str, Any]
    shared_artifacts: Mapping[str, Any]


@dataclass(frozen=True)
class ComponentLadderStageOutput:
    """Outputs returned by one alternating stage."""

    component_artifact: Any
    shared_artifacts: Mapping[str, Any] = field(default_factory=dict)
    result: Any = None


@dataclass(frozen=True)
class ComponentLadderStageRecord:
    """Recorded stage input/output for diagnostics and summaries."""

    index: int
    component: str
    stage_dir: Path
    input_component_artifacts: Mapping[str, Any]
    input_shared_artifacts: Mapping[str, Any]
    output_component_artifact: Any
    output_shared_artifacts: Mapping[str, Any]
    result: Any = None


@dataclass(frozen=True)
class ComponentLadderResult:
    """Final artifacts and per-stage records from an alternating ladder."""

    schedule: tuple[str, ...]
    component_artifacts: Mapping[str, Any]
    shared_artifacts: Mapping[str, Any]
    stages: tuple[ComponentLadderStageRecord, ...]

    @property
    def final_artifact(self) -> Any:
        if not self.stages:
            return None
        return self.stages[-1].output_component_artifact


StageTrainFn = Callable[[ComponentLadderStageContext], ComponentLadderStageOutput]


def _normalize_schedule(schedule: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(schedule, str):
        components = tuple(str(ch) for ch in schedule)
    else:
        components = tuple(str(ch) for ch in schedule)
    if not components or any(not str(ch) for ch in components):
        raise ValueError("component ladder schedule must be a non-empty sequence")
    return components


def run_component_ladder(
    *,
    schedule: str | Sequence[str],
    output_dir: str | Path,
    train_stage: StageTrainFn,
    initial_component_artifacts: Mapping[str, Any] | None = None,
    initial_shared_artifacts: Mapping[str, Any] | None = None,
    allowed_components: set[str] | frozenset[str] | None = None,
    stage_dir_name: Callable[[int, str], str] | None = None,
) -> ComponentLadderResult:
    """Run an alternating component schedule.

    Parameters
    ----------
    schedule:
        Component names in training order, e.g. ``"fgfg"``.  Components may
        be any non-empty strings when passed as a sequence.
    output_dir:
        Parent directory for per-stage outputs.
    train_stage:
        Callback that trains exactly one component and returns its new artifact
        plus any shared-interface artifacts updated by that stage.
    initial_component_artifacts:
        Current artifact per component before the first stage.
    initial_shared_artifacts:
        Artifacts for modules/interfaces shared across components.  Callers
        decide what these names mean; the runner just threads them forward.
    allowed_components:
        Optional validation set for schedule entries.
    stage_dir_name:
        Optional naming callback.  Defaults to ``stage_<i>_<component>``.
    """

    components = _normalize_schedule(schedule)
    allowed = set(str(x) for x in allowed_components) if allowed_components is not None else None
    if allowed is not None:
        invalid = sorted(set(components) - allowed)
        if invalid:
            raise ValueError(
                f"component ladder schedule contains unsupported components {invalid}; "
                f"allowed={sorted(allowed)}"
            )

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    component_artifacts: dict[str, Any] = dict(initial_component_artifacts or {})
    shared_artifacts: dict[str, Any] = dict(initial_shared_artifacts or {})
    records: list[ComponentLadderStageRecord] = []
    namer = stage_dir_name or (lambda i, comp: f"stage_{i}_{comp}")

    for index, component in enumerate(components):
        stage_dir = root / str(namer(int(index), str(component)))
        stage_dir.mkdir(parents=True, exist_ok=True)
        input_components = dict(component_artifacts)
        input_shared = dict(shared_artifacts)
        context = ComponentLadderStageContext(
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
            ComponentLadderStageRecord(
                index=int(index),
                component=str(component),
                stage_dir=stage_dir,
                input_component_artifacts=input_components,
                input_shared_artifacts=input_shared,
                output_component_artifact=output.component_artifact,
                output_shared_artifacts=dict(output.shared_artifacts or {}),
                result=output.result,
            )
        )

    return ComponentLadderResult(
        schedule=components,
        component_artifacts=component_artifacts,
        shared_artifacts=shared_artifacts,
        stages=tuple(records),
    )


__all__ = [
    "ComponentLadderResult",
    "ComponentLadderStageContext",
    "ComponentLadderStageOutput",
    "ComponentLadderStageRecord",
    "run_component_ladder",
]
