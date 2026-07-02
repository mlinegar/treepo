"""JSONable task-state and tree-unit records.

``TaskState`` is the lightweight value shape used in preference supervision
and artifacts when a task has an explicit composable state. Executable states
may still be tensors, sketch objects, or model artifacts behind
``ComposableStatistic``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from treepo.common import jsonable


@dataclass(frozen=True)
class TaskState:
    """A compact JSONable task state produced by ``g`` and read by ``f``."""

    kind: str
    items: Sequence[Any] = field(default_factory=tuple)
    counts: Mapping[str, float] = field(default_factory=dict)
    measures: Mapping[str, Any] = field(default_factory=dict)
    text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "items": [jsonable(item) for item in self.items],
            "counts": {str(k): float(v) for k, v in dict(self.counts or {}).items()},
            "measures": {str(k): jsonable(v) for k, v in dict(self.measures or {}).items()},
            "text": self.text,
            "metadata": {str(k): jsonable(v) for k, v in dict(self.metadata or {}).items()},
        }


def make_unit_id(tree_id: Any, node_id: Any) -> str:
    """Return the package-wide deterministic unit id spelling."""

    return f"{str(tree_id)}:{str(node_id)}"


def split_unit_id(unit_id: str) -> tuple[str, str]:
    """Split a ``make_unit_id`` spelling back into ``(tree_id, node_id)``."""

    tree_id, _, node_id = str(unit_id).partition(":")
    return tree_id, node_id


def state_from_value(value: Any, *, default_kind: str | None = None) -> TaskState | Any:
    """Coerce a TaskState-like mapping into ``TaskState`` when possible."""

    if isinstance(value, TaskState):
        return value
    if isinstance(value, Mapping) and "kind" in value:
        payload = dict(value)
        return TaskState(
            kind=str(payload.get("kind") or default_kind or "state"),
            items=tuple(payload.get("items") or ()),
            counts=dict(payload.get("counts") or {}),
            measures=dict(payload.get("measures") or {}),
            text=None if payload.get("text") is None else str(payload.get("text")),
            metadata=dict(payload.get("metadata") or {}),
        )
    return value


def state_to_dict(value: Any) -> Any:
    """Return a JSONable representation, preserving non-state values."""

    coerced = state_from_value(value)
    if isinstance(coerced, TaskState):
        return coerced.to_dict()
    return jsonable(coerced)


__all__ = [
    "TaskState",
    "make_unit_id",
    "split_unit_id",
    "state_from_value",
    "state_to_dict",
]
