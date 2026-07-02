"""TreeRecord-derived preference helpers.

Build a ``PreferenceDataset`` of node-level units from tree records (one unit
per node, gold supervision as the sole candidate) and filter an existing
dataset down to one tree. Depends on the data model; the ``TreeRecord`` import
is deferred so importing the preference boundary stays light.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from typing import Any, Sequence

from treepo.methods._preference_dataset import (
    Candidate,
    PreferenceDataset,
    PreferenceRecord,
    PreferenceTarget,
)
from treepo.methods._preference_normalize import _optional_str
from treepo.state import make_unit_id as _make_unit_id


def make_unit_id(tree_id: Any, node_id: Any) -> str:
    return _make_unit_id(tree_id, node_id)


def preference_units_from_trees(
    trees: Sequence[Any],
    *,
    target: PreferenceTarget = "g",
    unit_type: str = "node",
) -> PreferenceDataset:
    from treepo.tree import TreeRecord

    records: list[PreferenceRecord] = []
    for tree_idx, raw_tree in enumerate(trees):
        tree = TreeRecord.from_value(raw_tree)
        tree_id = str(tree.tree_id or f"tree_{tree_idx}")
        doc_id = str(tree.doc_id or tree_id)
        tree_meta = dict(tree.metadata or {})
        nodes = list(tree.nodes or ())
        if not nodes:
            candidates = _supervised_candidates(tree.root_label)
            records.append(
                PreferenceRecord(
                    unit_id=make_unit_id(tree_id, "root"),
                    unit_type="root",
                    target=target,
                    context=tree.text,
                    candidates=candidates,
                    tree_id=tree_id,
                    doc_id=doc_id,
                    node_id="root",
                    metadata=tree_meta,
                )
            )
            continue
        root = tree.root()
        parent_ids = _parent_ids_by_node(nodes)
        for pos, node in enumerate(nodes):
            node_id = str(node.node_id or pos)
            node_meta = dict(node.metadata or {})
            record_target = str(node_meta.get("target") or target)
            if record_target not in {"f", "g", "both"}:
                record_target = target
            resolved_unit_type = str(node.unit_type or unit_type)
            if root is not None and str(root.node_id) == node_id:
                resolved_unit_type = "root"
            elif resolved_unit_type == "node":
                resolved_unit_type = unit_type
            records.append(
                PreferenceRecord(
                    unit_id=make_unit_id(tree_id, node_id),
                    unit_type=resolved_unit_type,
                    target=record_target,  # type: ignore[arg-type]
                    context=node.text,
                    candidates=_supervised_candidates(node.supervised_value()),
                    tree_id=tree_id,
                    doc_id=doc_id,
                    node_id=node_id,
                    level=node.level,
                    position=node.position if node.position is not None else pos,
                    parent_id=node.parent_id or parent_ids.get(node_id),
                    left_child_id=node.left_child_id,
                    right_child_id=node.right_child_id,
                    metadata=node_meta,
                )
            )
    return PreferenceDataset.from_records(records)


def _supervised_candidates(value: Any) -> tuple[Candidate, ...]:
    if value is None:
        return ()
    return (Candidate(id="gold", value=value, score=1.0, preferred=True),)


def _parent_ids_by_node(nodes: Sequence[Any]) -> dict[str, str]:
    parents: dict[str, str] = {}
    for node in nodes:
        parent_id = str(getattr(node, "node_id", ""))
        if not parent_id:
            continue
        for child_id in (getattr(node, "left_child_id", None), getattr(node, "right_child_id", None)):
            child = _optional_str(child_id)
            if child is not None:
                parents.setdefault(child, parent_id)
    return parents


def filter_units_for_tree(dataset: Any, tree_id: Any) -> PreferenceDataset:
    pref = PreferenceDataset.from_value(dataset)
    tree = str(tree_id)
    units = [
        row
        for row in pref.units
        if str(row.get("tree_id") or row.get("doc_id") or "").startswith(tree)
        or str(row.get("tree_id") or row.get("doc_id") or "") == tree
    ]
    unit_ids = {str(row["unit_id"]) for row in units}
    candidates = [row for row in pref.candidates if str(row.get("unit_id")) in unit_ids]
    return PreferenceDataset(units=units, candidates=candidates)


def _tree_nodes(tree: Any) -> list[Any]:
    raw = getattr(tree, "nodes", None)
    if isinstance(raw, MappingABC):
        return list(raw.values())
    if isinstance(raw, SequenceABC) and not isinstance(raw, (str, bytes)):
        return list(raw)
    return []


def _is_root_node(tree: Any, node: Any) -> bool:
    levels = getattr(tree, "levels", None) or []
    if levels:
        root_id = str(levels[-1][-1])
        return str(getattr(node, "node_id", "")) == root_id
    nodes = _tree_nodes(tree)
    return bool(nodes and node is nodes[-1])


__all__ = [
    "_is_root_node",
    "_parent_ids_by_node",
    "_supervised_candidates",
    "_tree_nodes",
    "filter_units_for_tree",
    "make_unit_id",
    "preference_units_from_trees",
]
