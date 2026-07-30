"""TreeRecord-derived preference helpers.

Build a ``PreferenceDataset`` of node-level units from tree records (one unit
per node, gold supervision as the sole candidate). Depends on the data model;
the ``TreeRecord`` import is deferred so importing the preference boundary
stays light.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from treepo.methods._preference_dataset import (
    Candidate,
    PreferenceDataset,
    PreferenceRecord,
    PreferenceTarget,
)
from treepo.methods._preference_normalize import _optional_str
from treepo.state import TaskState, state_from_value, state_to_dict
from treepo.state import make_unit_id as _make_unit_id


def make_unit_id(tree_id: Any, node_id: Any) -> str:
    return _make_unit_id(tree_id, node_id)


def preference_units_from_trees(
    trees: Sequence[Any],
    *,
    target: PreferenceTarget = "g",
    unit_type: str = "node",
) -> PreferenceDataset:
    from treepo.methods._dspy_records import binary_topology
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
            root_meta = dict(tree_meta)
            if target in {"g", "both"}:
                root_meta["call_role"] = _validated_tree_g_call_role(
                    root_meta,
                    child_ids=(),
                    where=f"tree {tree_id!r} root",
                )
                root_meta.update(supervision_role="root", g_prompt=_leaf_g_prompt(tree.text))
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
                    metadata=root_meta,
                )
            )
            continue
        # Preference records intentionally discard their source TreeRecord
        # shape. Validate the exact supplied topology before that happens, and
        # derive every exported edge from the same strict topology used by the
        # DSPy f/g training adapter. This prevents a malformed DAG or an
        # inconsistent parent/child pair from becoming an apparently valid
        # flat preference dataset.
        topology = binary_topology(tree)
        root = topology.root
        child_ids_by_node = {
            str(node.node_id): tuple(
                str(child.node_id) for child in topology.children(node)
            )
            for node in nodes
        }
        parent_ids = {
            str(child.node_id): str(node.node_id)
            for node in nodes
            for child in topology.children(node)
        }
        nodes_by_id = {str(node.node_id): node for node in nodes}
        for pos, node in enumerate(nodes):
            node_id = str(node.node_id or pos)
            node_meta = dict(node.metadata or {})
            record_target = str(node_meta.get("target") or target)
            if record_target not in {"f", "g", "both"}:
                record_target = target
            child_ids = child_ids_by_node.get(node_id, ())
            if record_target in {"g", "both"}:
                call_role = _validated_tree_g_call_role(
                    node_meta,
                    child_ids=child_ids,
                    where=f"tree {tree_id!r} node {node_id!r}",
                )
                node_meta["call_role"] = call_role
                node_meta["supervision_role"] = (
                    "root"
                    if root is not None and str(root.node_id) == node_id
                    else "leaf"
                    if not child_ids
                    else "merge"
                )
                node_meta["g_prompt"] = _tree_g_prompt(
                    node,
                    child_ids=child_ids,
                    nodes_by_id=nodes_by_id,
                    where=f"tree {tree_id!r} node {node_id!r}",
                )
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
                    parent_id=parent_ids.get(node_id),
                    left_child_id=child_ids[0] if child_ids else None,
                    right_child_id=child_ids[1] if len(child_ids) == 2 else None,
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
        for child_id in (
            getattr(node, "left_child_id", None),
            getattr(node, "right_child_id", None),
        ):
            child = _optional_str(child_id)
            if child is not None:
                parents.setdefault(child, parent_id)
    return parents


def _child_ids_by_node(nodes: Sequence[Any]) -> dict[str, tuple[str, ...]]:
    """Collect only child edges explicitly supplied on either edge endpoint."""

    children: dict[str, list[str]] = {}
    for node in nodes:
        node_id = str(getattr(node, "node_id", ""))
        if not node_id:
            continue
        for child_id in (
            getattr(node, "left_child_id", None),
            getattr(node, "right_child_id", None),
        ):
            child = _optional_str(child_id)
            if child is not None and child not in children.setdefault(node_id, []):
                children[node_id].append(child)
    for node in nodes:
        parent = _optional_str(getattr(node, "parent_id", None))
        child = _optional_str(getattr(node, "node_id", None))
        if parent is None or child is None:
            continue
        if child not in children.setdefault(parent, []):
            children[parent].append(child)
    return {node_id: tuple(child_ids) for node_id, child_ids in children.items()}


def _validated_tree_g_call_role(
    metadata: Mapping[str, Any],
    *,
    child_ids: Sequence[str],
    where: str,
) -> str:
    count = len(tuple(child_ids))
    if count > 2:
        raise ValueError(
            f"{where} supplies {count} children; g preference supervision "
            "requires exact leaf/unary/binary C-Tree topology"
        )
    derived = "leaf" if count == 0 else "recompression" if count == 1 else "merge"
    for key in ("call_role", "g_call_role"):
        value = metadata.get(key)
        if value is None or not str(value).strip():
            continue
        declared = str(value).strip().lower()
        if declared not in {"leaf", "recompression", "merge"}:
            raise ValueError(
                f"{where} has invalid {key}={value!r}; expected 'leaf', 'recompression', or 'merge'"
            )
        if declared != derived:
            raise ValueError(
                f"{where} declares {key}={declared!r}, but its supplied "
                f"topology requires {derived!r}"
            )
    return derived


def _tree_g_prompt(
    node: Any,
    *,
    child_ids: Sequence[str],
    nodes_by_id: Mapping[str, Any],
    where: str,
) -> str:
    if not child_ids:
        return _leaf_g_prompt(str(getattr(node, "text", "") or ""))
    child_states: list[str] = []
    for child_id in child_ids:
        child = nodes_by_id.get(str(child_id))
        if child is None:
            raise ValueError(f"{where} names missing child {child_id!r}")
        state = _node_reference_state(child)
        if state is None:
            raise ValueError(
                f"{where} cannot export {len(child_ids)}-child g supervision: "
                f"child {child_id!r} has no supplied reference state"
            )
        child_states.append(state)
    if len(child_states) == 1:
        return _recompression_g_prompt(child_states[0])
    return _merge_g_prompt(child_states[0], child_states[1])


def _node_reference_state(node: Any) -> str | None:
    value = state_from_value(getattr(node, "state", None))
    if value is None:
        value = state_from_value(getattr(node, "label", None))
    if isinstance(value, TaskState):
        if str(value.text or "").strip():
            return str(value.text).strip()
        return json.dumps(state_to_dict(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, str) and value.strip():
        return value.strip()
    metadata = dict(getattr(node, "metadata", None) or {})
    for key in (
        "summary",
        "teacher_summary",
        "target_summary",
        "reference_summary",
        "completion",
        "response",
    ):
        candidate = metadata.get(key)
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return None


def _leaf_g_prompt(text: str) -> str:
    return (
        "[TREEPO_G_CALL=leaf]\n"
        "Apply the shared C-Tree state operator g to this leaf. Return only the state.\n"
        f"LEAF_TEXT:\n{text}"
    )


def _recompression_g_prompt(child_state: str) -> str:
    return (
        "[TREEPO_G_CALL=recompression]\n"
        "Promote this odd carry with the same shared C-Tree state operator g. "
        "Do not fabricate a second child. Return only the parent state.\n"
        f"CHILD_STATE:\n{child_state}"
    )


def _merge_g_prompt(left_state: str, right_state: str) -> str:
    return (
        "[TREEPO_G_CALL=merge]\n"
        "Apply the same shared C-Tree state operator g to these two child states. "
        "Return only the merged state.\n"
        f"LEFT_STATE:\n{left_state}\n\nRIGHT_STATE:\n{right_state}"
    )


__all__ = [
    "_parent_ids_by_node",
    "_supervised_candidates",
    "make_unit_id",
    "preference_units_from_trees",
]
