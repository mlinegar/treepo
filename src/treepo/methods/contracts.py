"""Lightweight public contracts for :mod:`treepo.methods`.

These are the small stable records needed by the publishable package. Larger
contract modules live outside the public treepo runtime and should adapt to
these records at the package boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from treepo.common import jsonable
from treepo.objective import ObjectiveSpec

JsonDict = dict[str, Any]


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
    # First-class supervision-grid axes (see treepo.methods._grid_axes and
    # docs/treepo_fit_grid_upgrade_plan_2026_07_10.md Phase 3). Defaults keep
    # today's behavior: all documents, root-only labels, seed 0.
    doc_gold_n: int | None = None
    local_label_mix: str = "none"
    gold_fraction_p: float = 1.0
    distilled_labels_path: str | None = None
    seed: int = 0
    # Per-node supervision (fit-grid plan Phase 1). ``supervision_level`` names
    # a cell with the TT ladder's vocabulary (default/root/leaf/node/mix); the
    # explicit weights apply only at the ``default`` level. ``local_law_weight``
    # builds the canonical convex ObjectiveSpec when no explicit objective is
    # configured. Defaults preserve today's behavior exactly.
    supervision_level: str = "default"
    root_weight: float | None = None
    leaf_weight: float | None = None
    merge_weight: float | None = None
    local_law_weight: float | None = None

    def to_dict(self) -> JsonDict:
        return {
            "space_kind": str(self.space_kind),
            "family": str(self.family),
            "schedule": str(self.schedule),
            "initial_artifacts": jsonable(dict(self.initial_artifacts or {})),
            "backend_config": jsonable(dict(self.backend_config or {})),
            "axis": jsonable(dict(self.axis or {})),
            "preference_data": jsonable(self.preference_data),
            "doc_gold_n": (None if self.doc_gold_n is None else int(self.doc_gold_n)),
            "local_label_mix": str(self.local_label_mix),
            "gold_fraction_p": float(self.gold_fraction_p),
            "distilled_labels_path": (
                None if self.distilled_labels_path is None else str(self.distilled_labels_path)
            ),
            "seed": int(self.seed),
            "supervision_level": str(self.supervision_level),
            "root_weight": (None if self.root_weight is None else float(self.root_weight)),
            "leaf_weight": (None if self.leaf_weight is None else float(self.leaf_weight)),
            "merge_weight": (None if self.merge_weight is None else float(self.merge_weight)),
            "local_law_weight": (
                None if self.local_law_weight is None else float(self.local_law_weight)
            ),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CTreePOLearningSpec":
        doc_gold_n = payload.get("doc_gold_n")
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
            doc_gold_n=(None if doc_gold_n is None else int(doc_gold_n)),
            local_label_mix=str(payload.get("local_label_mix") or "none"),
            gold_fraction_p=float(
                payload["gold_fraction_p"] if payload.get("gold_fraction_p") is not None else 1.0
            ),
            distilled_labels_path=(
                None
                if payload.get("distilled_labels_path") is None
                else str(payload["distilled_labels_path"])
            ),
            seed=int(payload.get("seed") or 0),
            supervision_level=str(payload.get("supervision_level") or "default"),
            root_weight=(
                None if payload.get("root_weight") is None else float(payload["root_weight"])
            ),
            leaf_weight=(
                None if payload.get("leaf_weight") is None else float(payload["leaf_weight"])
            ),
            merge_weight=(
                None if payload.get("merge_weight") is None else float(payload["merge_weight"])
            ),
            local_law_weight=(
                None
                if payload.get("local_law_weight") is None
                else float(payload["local_law_weight"])
            ),
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


__all__ = [
    "FitResult",
    "CTreePOLearningSpec",
    "FamilyRuntime",
    "JsonDict",
    "ObjectiveSpec",
    "jsonable",
]
