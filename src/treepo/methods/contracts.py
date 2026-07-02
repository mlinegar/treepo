"""Lightweight public contracts for :mod:`treepo.methods`.

These are the small stable records needed by the publishable package. Larger
contract modules live outside the public treepo runtime and should adapt to
these records at the package boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from treepo.objective import ObjectiveSpec

JsonDict = dict[str, Any]


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return value


@runtime_checkable
class FamilyRuntime(Protocol):
    """Minimal contract for a family used by the alternating methods loop."""

    name: str

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[Any],
        output_dir: Any,
        iteration: int,
    ) -> Any:
        ...

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Any,
        iteration: int,
    ) -> Any:
        ...

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[Any],
    ) -> list[float | None]:
        ...

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        ...


@dataclass(frozen=True)
class CTreePOLearningSpec:
    """Learning job description for a C-TreePO f/g ladder."""

    space_kind: str
    family: str
    schedule: str
    initial_artifacts: Mapping[str, Any] = field(default_factory=dict)
    train_data: Any = None
    preference_data: Any = None
    eval_data: Any = None
    backend_config: Mapping[str, Any] = field(default_factory=dict)
    axis: Mapping[str, Any] = field(default_factory=dict)

    def with_schedule(self, schedule: str) -> "CTreePOLearningSpec":
        return replace(self, schedule=str(schedule))

    def with_initial_artifacts(
        self,
        artifacts: Mapping[str, Any],
    ) -> "CTreePOLearningSpec":
        return replace(self, initial_artifacts=dict(artifacts))

    def to_dict(self) -> JsonDict:
        return {
            "space_kind": str(self.space_kind),
            "family": str(self.family),
            "schedule": str(self.schedule),
            "initial_artifacts": jsonable(dict(self.initial_artifacts or {})),
            "backend_config": jsonable(dict(self.backend_config or {})),
            "axis": jsonable(dict(self.axis or {})),
            "preference_data": jsonable(self.preference_data),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CTreePOLearningSpec":
        return cls(
            space_kind=str(payload.get("space_kind") or ""),
            family=str(payload.get("family") or ""),
            schedule=str(payload.get("schedule") or ""),
            initial_artifacts=dict(payload.get("initial_artifacts") or {}),
            train_data=payload.get("train_data"),
            preference_data=payload.get("preference_data"),
            eval_data=payload.get("eval_data"),
            backend_config=dict(payload.get("backend_config") or {}),
            axis=dict(payload.get("axis") or {}),
        )


@dataclass(frozen=True)
class FitResult:
    """Uniform result returned by treepo.fit and internal fit helpers."""

    status: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    history: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    summary: Mapping[str, Any] = field(default_factory=dict)
    manifest_path: str | None = None
    mode: str = "learning"

    def to_dict(self) -> JsonDict:
        return {
            "status": str(self.status),
            "mode": str(self.mode),
            "metrics": jsonable(dict(self.metrics or {})),
            "artifacts": jsonable(dict(self.artifacts or {})),
            "history": jsonable(list(self.history or ())),
            "summary": jsonable(dict(self.summary or {})),
            "manifest_path": self.manifest_path,
        }


CTreePOFitResult = FitResult


__all__ = [
    "CTreePOFitResult",
    "FitResult",
    "CTreePOLearningSpec",
    "FamilyRuntime",
    "JsonDict",
    "ObjectiveSpec",
    "jsonable",
]
