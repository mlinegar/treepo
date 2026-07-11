"""Minimal labeled tree records for treepo examples and adapters.

``TreeRecord`` is the package-owned tree artifact shape. It is intentionally
small: it stores stable topology, optional labels/states, and JSONable metadata
without imposing a training runtime.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import asdict, dataclass, field, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo.common import jsonable
from treepo.state import TaskState, make_unit_id, state_from_value, state_to_dict


@dataclass(frozen=True)
class TreeNode:
    """One labeled unit in a composable tree."""

    node_id: str
    unit_type: str = "node"
    text: str = ""
    level: int | None = None
    position: int | None = None
    parent_id: str | None = None
    left_child_id: str | None = None
    right_child_id: str | None = None
    label: Any = None
    state: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "TreeNode":
        if isinstance(value, TreeNode):
            return value
        if isinstance(value, MappingABC):
            row = dict(value)
            return cls(
                node_id=str(row.get("node_id") or row.get("id") or ""),
                unit_type=str(row.get("unit_type") or row.get("kind") or "node"),
                text=str(row.get("text") or row.get("content") or ""),
                level=_optional_int(row.get("level")),
                position=_optional_int(row.get("position")),
                parent_id=_optional_str(row.get("parent_id")),
                left_child_id=_optional_str(row.get("left_child_id")),
                right_child_id=_optional_str(row.get("right_child_id")),
                label=_maybe_json(row.get("label_json", row.get("label", row.get("score")))),
                state=state_from_value(_maybe_json(row.get("state_json", row.get("state")))),
                metadata=dict(_maybe_json(row.get("metadata_json", row.get("metadata") or {})) or {}),
            )
        return cls(
            node_id=str(getattr(value, "node_id", getattr(value, "id", ""))),
            unit_type=str(getattr(value, "unit_type", getattr(value, "kind", "node"))),
            text=str(getattr(value, "text", getattr(value, "content", "")) or ""),
            level=_optional_int(getattr(value, "level", None)),
            position=_optional_int(getattr(value, "position", None)),
            parent_id=_optional_str(getattr(value, "parent_id", None)),
            left_child_id=_optional_str(getattr(value, "left_child_id", None)),
            right_child_id=_optional_str(getattr(value, "right_child_id", None)),
            label=getattr(value, "label", getattr(value, "score", None)),
            state=state_from_value(getattr(value, "state", None)),
            metadata=dict(getattr(value, "metadata", None) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": str(self.node_id),
            "unit_type": str(self.unit_type),
            "text": str(self.text or ""),
            "level": self.level,
            "position": self.position,
            "parent_id": self.parent_id,
            "left_child_id": self.left_child_id,
            "right_child_id": self.right_child_id,
            "label": jsonable(self.label),
            "state": state_to_dict(self.state),
            "metadata": {str(k): jsonable(v) for k, v in dict(self.metadata or {}).items()},
        }

    def has_children(self) -> bool:
        return bool(self.left_child_id or self.right_child_id)

    def supervised_value(self) -> Any:
        if self.state is not None:
            return self.state
        return self.label


@dataclass(frozen=True)
class TreeRecord:
    """A small JSONable labeled tree artifact."""

    tree_id: str
    doc_id: str | None = None
    text: str = ""
    root_label: Any = None
    nodes: Sequence[TreeNode | Mapping[str, Any] | Any] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = tuple(TreeNode.from_value(node) for node in (self.nodes or ()))
        object.__setattr__(self, "nodes", normalized)
        if self.doc_id is None:
            object.__setattr__(self, "doc_id", str(self.tree_id))

    @classmethod
    def from_value(cls, value: Any) -> "TreeRecord":
        if isinstance(value, TreeRecord):
            return value
        if isinstance(value, MappingABC):
            row = dict(value)
            nodes = _normalize_nodes(row.get("nodes", ()), levels=row.get("levels"))
            return cls(
                tree_id=str(row.get("tree_id") or row.get("doc_id") or row.get("id") or "tree"),
                doc_id=_optional_str(row.get("doc_id")),
                text=str(row.get("text") or row.get("document_text") or ""),
                root_label=_maybe_json(row.get("root_label_json", row.get("root_label", row.get("document_score")))),
                nodes=nodes,
                metadata=dict(_maybe_json(row.get("metadata_json", row.get("metadata") or {})) or {}),
            )
        meta = dict(getattr(value, "metadata", None) or {})
        raw_nodes = _normalize_nodes(getattr(value, "nodes", ()), levels=getattr(value, "levels", None))
        return cls(
            tree_id=str(
                getattr(value, "tree_id", None)
                or getattr(value, "doc_id", None)
                or meta.get("tree_id")
                or meta.get("doc_id")
                or "tree"
            ),
            doc_id=_optional_str(getattr(value, "doc_id", meta.get("doc_id", None))),
            text=str(getattr(value, "text", getattr(value, "document_text", "")) or ""),
            root_label=getattr(value, "root_label", getattr(value, "document_score", meta.get("root_label", None))),
            nodes=raw_nodes,
            metadata=meta,
        )

    @classmethod
    def load(cls, path: Path | str) -> "TreeRecord":
        return cls.from_value(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree_id": str(self.tree_id),
            "doc_id": self.doc_id,
            "text": str(self.text or ""),
            "root_label": jsonable(self.root_label),
            "nodes": [node.to_dict() for node in self.nodes],
            "levels": self.levels(),
            "metadata": {str(k): jsonable(v) for k, v in dict(self.metadata or {}).items()},
        }

    def get_node(self, node_id: Any) -> TreeNode | None:
        needle = str(node_id)
        for node in self.nodes:
            if str(node.node_id) == needle:
                return node
        return None

    def levels(self) -> list[list[str]]:
        buckets: dict[int, list[tuple[int, int, str]]] = {}
        for idx, node in enumerate(self.nodes):
            level = 0 if node.level is None else int(node.level)
            position = idx if node.position is None else int(node.position)
            buckets.setdefault(level, []).append((position, idx, str(node.node_id)))
        return [[node_id for _, _, node_id in sorted(rows)] for _, rows in sorted(buckets.items())]

    def root(self) -> TreeNode | None:
        roots = [node for node in self.nodes if str(node.unit_type) == "root"]
        if roots:
            return roots[-1]
        child_ids = {
            child
            for node in self.nodes
            for child in (_optional_str(node.left_child_id), _optional_str(node.right_child_id))
            if child is not None
        }
        candidates = [
            node
            for node in self.nodes
            if str(node.node_id) not in child_ids and node.parent_id is None
        ]
        if not candidates:
            candidates = [node for node in self.nodes if str(node.node_id) not in child_ids]
        if candidates:
            return sorted(candidates, key=lambda node: (-1 if node.level is None else int(node.level), node.position or 0))[-1]
        return None

    def leaves(self) -> tuple[TreeNode, ...]:
        # A node is internal if it points at children or if any node names it
        # as parent; parent_id edges carry topology even when a wide node has
        # more children than the two child-id slots.
        parent_ids = {
            str(node.parent_id) for node in self.nodes if node.parent_id is not None
        }
        return tuple(
            node
            for node in self.nodes
            if not node.has_children() and str(node.node_id) not in parent_ids
        )

    def unit_id(self, node_id: Any) -> str:
        return make_unit_id(self.tree_id, node_id)


def _normalize_nodes(value: Any, *, levels: Any = None) -> tuple[TreeNode, ...]:
    raw_nodes = value
    if isinstance(raw_nodes, MappingABC):
        by_id = {str(key): TreeNode.from_value(node) for key, node in raw_nodes.items()}
    elif isinstance(raw_nodes, SequenceABC) and not isinstance(raw_nodes, (str, bytes)):
        converted = [TreeNode.from_value(node) for node in raw_nodes]
        by_id = {str(node.node_id): node for node in converted}
    else:
        by_id = {}
    ordered: list[TreeNode] = []
    seen: set[str] = set()
    if isinstance(levels, SequenceABC) and not isinstance(levels, (str, bytes)):
        for level_idx, level_ids in enumerate(levels):
            if not isinstance(level_ids, SequenceABC) or isinstance(level_ids, (str, bytes)):
                continue
            for position, raw_id in enumerate(level_ids):
                node_id = str(raw_id)
                node = by_id.get(node_id)
                if node is None:
                    continue
                ordered.append(_with_tree_position(node, level=level_idx, position=position))
                seen.add(node_id)
    for node in by_id.values():
        if str(node.node_id) not in seen:
            ordered.append(node)
    return tuple(ordered)


def _with_tree_position(node: TreeNode, *, level: int, position: int) -> TreeNode:
    return TreeNode(
        node_id=node.node_id,
        unit_type=node.unit_type,
        text=node.text,
        level=node.level if node.level is not None else int(level),
        position=node.position if node.position is not None else int(position),
        parent_id=node.parent_id,
        left_child_id=node.left_child_id,
        right_child_id=node.right_child_id,
        label=node.label,
        state=node.state,
        metadata=node.metadata,
    )


def load_tree_records(path: Path | str) -> list[TreeRecord]:
    """Load tree records from a JSON file, JSONL file, or directory of both."""

    root = Path(path)
    if root.is_dir():
        out: list[TreeRecord] = []
        for child in sorted(root.glob("*.json")):
            out.append(TreeRecord.load(child))
        for child in sorted(root.glob("*.jsonl")):
            out.extend(load_tree_records(child))
        return out
    if root.suffix.lower() == ".jsonl":
        records: list[TreeRecord] = []
        with root.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    records.append(TreeRecord.from_value(json.loads(text)))
        return records
    return [TreeRecord.load(root)]


def write_tree_records_jsonl(path: Path | str, trees: Sequence[Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for tree in trees:
            handle.write(json.dumps(TreeRecord.from_value(tree).to_dict(), sort_keys=True) + "\n")
    return out


def iter_tree_units(tree: Any, *, order: str = "levels") -> tuple[TreeNode, ...]:
    """Return nodes from a tree in a stable order.

    Inputs are normalized through ``TreeRecord``, so any tree-like value
    works. ``order="levels"`` walks level by level from the leaves;
    ``order="root_first"`` puts the root ahead of the remaining nodes.
    """

    record = TreeRecord.from_value(tree)
    if order == "root_first":
        root = record.root()
        return tuple(
            ([] if root is None else [root])
            + [node for node in record.nodes if root is None or node.node_id != root.node_id]
        )
    if order == "levels":
        by_id = {str(node.node_id): node for node in record.nodes}
        return tuple(
            by_id[node_id]
            for level in record.levels()
            for node_id in level
            if node_id in by_id
        )
    raise ValueError("order must be 'levels' or 'root_first'")


def validate_tree_record(tree: Any) -> tuple[str, ...]:
    """Return structural validation errors for a ``TreeRecord``.

    The function is intentionally non-throwing so examples can include validation
    summaries in artifacts while still deciding how strict to be.
    """

    record = TreeRecord.from_value(tree)
    errors: list[str] = []
    if not str(record.tree_id or ""):
        errors.append("tree_id is required")
    node_ids = [str(node.node_id) for node in record.nodes]
    if any(not node_id for node_id in node_ids):
        errors.append("all nodes require node_id")
    seen: set[str] = set()
    for node_id in node_ids:
        if node_id in seen:
            errors.append(f"duplicate node_id: {node_id}")
        seen.add(node_id)
    node_set = set(node_ids)
    by_id = {str(node.node_id): node for node in record.nodes}
    for node in record.nodes:
        node_id = str(node.node_id)
        if node.parent_id is not None and str(node.parent_id) not in node_set:
            errors.append(f"node {node_id} parent_id does not exist: {node.parent_id}")
        for child_key, child_id in (("left_child_id", node.left_child_id), ("right_child_id", node.right_child_id)):
            if child_id is not None and str(child_id) not in node_set:
                errors.append(f"node {node_id} {child_key} does not exist: {child_id}")
            child = by_id.get(str(child_id)) if child_id is not None else None
            if child is not None and child.parent_id is not None and str(child.parent_id) != node_id:
                errors.append(f"node {child.node_id} parent_id disagrees with parent {node_id}")
    if record.nodes and record.root() is None:
        errors.append("tree has nodes but no root candidate")
    return tuple(errors)


def tree_summary(tree: Any) -> dict[str, Any]:
    """Return a compact JSONable topology and supervision summary."""

    record = TreeRecord.from_value(tree)
    leaves = record.leaves()
    root = record.root()
    return {
        "tree_id": str(record.tree_id),
        "doc_id": record.doc_id,
        "n_nodes": len(record.nodes),
        "n_leaves": len(leaves),
        "n_internal": len(record.nodes) - len(leaves),
        "root_node_id": None if root is None else str(root.node_id),
        "levels": record.levels(),
        "has_root_label": record.root_label is not None,
        "n_labeled_nodes": sum(1 for node in record.nodes if node.label is not None),
        "n_state_nodes": sum(1 for node in record.nodes if node.state is not None),
        "validation_errors": list(validate_tree_record(record)),
    }


def local_law_rows_from_tree_records(
    trees: Sequence[Any],
    *,
    default_observed: bool | None = None,
    default_propensity: float = 1.0,
    default_node_weight: float = 1.0,
) -> tuple[Any, ...]:
    """Build local-law rows from explicit node-level loss annotations.

    Nodes opt in by placing ``proxy_loss`` or ``local_law_proxy_loss`` in their
    metadata. Optional metadata keys include ``oracle_loss``, ``observed``,
    ``propensity``, ``node_weight``, ``law_kind``, and ``row_id``. This helper
    standardizes row identity and tree metadata; losses come only from explicit
    node metadata.
    """

    from treepo.local_law import LawKind, LocalLawAuditRow

    rows: list[LocalLawAuditRow] = []
    for tree_idx, raw_tree in enumerate(trees):
        record = TreeRecord.from_value(raw_tree)
        # TreeRecord levels count up from the leaves (leaf level 0, root
        # highest); the local-law depth convention puts the root at depth 0
        # with weight gamma^depth. Convert here so gamma discounts deep nodes.
        max_level = max(
            (int(node.level) for node in record.nodes if node.level is not None),
            default=0,
        )
        for node in iter_tree_units(record, order="levels"):
            node_meta = dict(node.metadata or {})
            proxy_loss = _first_present(node_meta, "proxy_loss", "local_law_proxy_loss")
            if proxy_loss is None:
                continue
            oracle_loss = _first_present(node_meta, "oracle_loss", "local_law_oracle_loss")
            observed_raw = _first_present(node_meta, "observed", "local_law_observed")
            observed = _bool_value(
                observed_raw,
                default=(oracle_loss is not None if default_observed is None else bool(default_observed)),
            )
            if observed and oracle_loss is None:
                raise ValueError(f"observed local-law node {record.tree_id}:{node.node_id} requires oracle_loss")
            law_kind = _first_present(node_meta, "law_kind", "local_law_kind") or _infer_law_kind(node, record)
            law = LawKind.from_value(law_kind)
            row_id = str(
                _first_present(node_meta, "row_id", "local_law_row_id")
                or f"{record.tree_id}:{node.node_id}:{law.value}"
            )
            propensity = _float_or_default(
                _first_present(node_meta, "propensity", "local_law_propensity"),
                default_propensity,
            )
            node_weight = _float_or_default(
                _first_present(node_meta, "node_weight", "local_law_node_weight"),
                default_node_weight,
            )
            rows.append(
                LocalLawAuditRow(
                    row_id=row_id,
                    law_kind=law,
                    proxy_loss=float(proxy_loss),
                    oracle_loss=None if oracle_loss is None else float(oracle_loss),
                    observed=observed,
                    propensity=float(propensity),
                    node_weight=float(node_weight),
                    depth=0 if node.level is None else max_level - int(node.level),
                    metadata={
                        "tree_index": tree_idx,
                        "tree_id": str(record.tree_id),
                        "doc_id": record.doc_id,
                        "node_id": str(node.node_id),
                        "unit_id": record.unit_id(node.node_id),
                        "unit_type": str(node.unit_type),
                        "level": node.level,
                        "position": node.position,
                        "parent_id": node.parent_id,
                        "left_child_id": node.left_child_id,
                        "right_child_id": node.right_child_id,
                        "tree_metadata": jsonable(record.metadata),
                        "node_metadata": jsonable(node_meta),
                    },
                )
            )
    return tuple(rows)


def _first_present(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _bool_value(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "observed"}
    return bool(value)


def _float_or_default(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _infer_law_kind(node: TreeNode, tree: TreeRecord) -> str:
    root = tree.root()
    if root is not None and str(root.node_id) == str(node.node_id) and not node.has_children():
        return "on_range_idempotence"
    if node.has_children():
        return "merge_preservation"
    return "leaf_preservation"


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in ("[", "{", '"'):
            try:
                return json.loads(stripped)
            except Exception:
                return value
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def tree_leaves(tree: Any) -> tuple[Any, ...] | None:
    """Return the leaf sequence of any tree-like value.

    The single leaf-extraction convention for every family: a ``leaves``
    attribute holding a sequence (fixture trees), a callable ``leaves()``
    method (``TreeRecord``), or a ``get_leaves()`` method. Leaf order is the
    composition order the family merges in. Returns ``None`` when the value
    exposes no leaves.
    """

    leaves = getattr(tree, "leaves", None)
    if callable(leaves):
        leaves = leaves()
    if leaves is None and callable(getattr(tree, "get_leaves", None)):
        leaves = tree.get_leaves()
    if leaves is None:
        return None
    out = tuple(leaves)
    return out if out else None


_TREE_ID_KEYS = ("tree_id", "doc_id", "unit_id")


def tree_row_id(tree: Any, index: int, *, fallback_prefix: str | None = "tree") -> str:
    """Return the canonical row identifier for a tree-like value.

    The single id-resolution convention for audit rows, prediction rows, and
    exports: attributes ``tree_id``/``doc_id``/``unit_id`` first, then the
    same keys in ``metadata``. Anonymous trees fall back to
    ``{fallback_prefix}_{index}`` (or bare ``str(index)`` when
    ``fallback_prefix`` is ``None``).
    """

    for key in _TREE_ID_KEYS:
        value = getattr(tree, key, None)
        if value is not None and not callable(value):
            return str(value)
    metadata = getattr(tree, "metadata", None)
    if isinstance(metadata, MappingABC):
        for key in _TREE_ID_KEYS:
            if metadata.get(key) is not None:
                return str(metadata[key])
    if fallback_prefix is None:
        return str(int(index))
    return f"{fallback_prefix}_{int(index)}"


__all__ = [
    "TreeNode",
    "TreeRecord",
    "iter_tree_units",
    "load_tree_records",
    "local_law_rows_from_tree_records",
    "tree_leaves",
    "tree_row_id",
    "tree_summary",
    "validate_tree_record",
    "write_tree_records_jsonl",
]
