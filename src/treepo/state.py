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


@dataclass(frozen=True)
class TreeUnitRef:
    """Stable identity for a tree unit without imposing a concrete tree class."""

    tree_id: str
    node_id: str
    unit_id: str
    unit_type: str
    level: int | None = None
    position: int | None = None
    parent_id: str | None = None
    left_child_id: str | None = None
    right_child_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree_id": str(self.tree_id),
            "node_id": str(self.node_id),
            "unit_id": str(self.unit_id),
            "unit_type": str(self.unit_type),
            "level": self.level,
            "position": self.position,
            "parent_id": self.parent_id,
            "left_child_id": self.left_child_id,
            "right_child_id": self.right_child_id,
            "metadata": {str(k): jsonable(v) for k, v in dict(self.metadata or {}).items()},
        }


def make_unit_id(tree_id: Any, node_id: Any) -> str:
    """Return the package-wide deterministic unit id spelling."""

    return f"{str(tree_id)}:{str(node_id)}"


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


def unit_ref_from(
    value: Any,
    *,
    tree_id: Any | None = None,
    node_id: Any | None = None,
    unit_id: Any | None = None,
    unit_type: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TreeUnitRef:
    """Build a ``TreeUnitRef`` from an object or mapping plus overrides."""

    row = dict(value) if isinstance(value, Mapping) else {}
    meta = dict(row.get("metadata") or getattr(value, "metadata", {}) or {})
    if metadata:
        meta.update(dict(metadata))
    resolved_tree = str(
        tree_id
        or row.get("tree_id")
        or row.get("doc_id")
        or meta.get("tree_id")
        or meta.get("doc_id")
        or getattr(value, "tree_id", "")
        or getattr(value, "doc_id", "")
        or "tree"
    )
    resolved_node = str(
        node_id
        or row.get("node_id")
        or getattr(value, "node_id", "")
        or ("root" if str(unit_type or row.get("unit_type") or "") == "root" else "")
        or "unit"
    )
    resolved_unit = str(unit_id or row.get("unit_id") or make_unit_id(resolved_tree, resolved_node))
    return TreeUnitRef(
        tree_id=resolved_tree,
        node_id=resolved_node,
        unit_id=resolved_unit,
        unit_type=str(unit_type or row.get("unit_type") or row.get("kind") or "unit"),
        level=_optional_int(row.get("level", getattr(value, "level", None))),
        position=_optional_int(row.get("position", getattr(value, "position", None))),
        parent_id=_optional_str(row.get("parent_id", getattr(value, "parent_id", None))),
        left_child_id=_optional_str(row.get("left_child_id", getattr(value, "left_child_id", None))),
        right_child_id=_optional_str(row.get("right_child_id", getattr(value, "right_child_id", None))),
        metadata=meta,
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "TaskState",
    "TreeUnitRef",
    "jsonable",
    "make_unit_id",
    "state_from_value",
    "state_to_dict",
    "unit_ref_from",
]
