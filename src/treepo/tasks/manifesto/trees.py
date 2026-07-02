"""Tree and artifact conversion helpers for Manifesto/RILE examples."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from treepo.state import make_unit_id
from treepo.tasks.manifesto.common import root_label
from treepo.tasks.manifesto.documents import (
    DEFAULT_MANIFESTO_REPLICATIONS,
    ManifestoDocument,
    ManifestoLeaf,
    ManifestoQSentence,
    ManifestoReplicationTree,
)
from treepo.tasks.manifesto.rile import clamp_rile
from treepo.tasks.manifesto.state import manifesto_policy_state_from_leaf
from treepo.tree import TreeNode, TreeRecord


def make_manifesto_replication_trees(
    documents: Sequence[ManifestoDocument] | None = None,
    *,
    split: str = "test",
    leaf_unit_count: int = 1,
    doc_unit_kind: str = "qsentence",
) -> list[ManifestoReplicationTree]:
    docs = tuple(documents or DEFAULT_MANIFESTO_REPLICATIONS)
    trees: list[ManifestoReplicationTree] = []
    for doc in docs:
        label = clamp_rile(float(doc.rile_label))
        unit_kind = str(doc_unit_kind or "qsentence")
        leaves = tuple(
            _group_doc_units_into_leaves(
                doc.qsentences,
                leaf_unit_count=leaf_unit_count,
                doc_unit_kind=unit_kind,
            )
        )
        metadata = {
            "split": split,
            "doc_id": doc.doc_id,
            "country": doc.country,
            "party": doc.party,
            "year": int(doc.year),
            "replication": doc.replication,
            "text": doc.text,
            "teacher_score_1_7": label,
            "teacher_score_native": label,
            "expert_score_1_7": label,
            "expert_score_native": label,
            "expert_target_scale": "rile",
            "expert_score_for_objective": label,
            "root_label": label,
            "root_label_name": "rile",
            "leaf_unit_count": int(max(1, leaf_unit_count)),
            "doc_unit_kind": unit_kind,
        }
        metadata.update(dict(doc.metadata or {}))
        trees.append(
            ManifestoReplicationTree(
                doc_id=str(doc.doc_id),
                text=str(doc.text),
                leaves=leaves,
                metadata=metadata,
            )
        )
    return trees


def manifesto_tree_records(trees: Sequence[ManifestoReplicationTree]) -> list[TreeRecord]:
    """Convert Manifesto fixtures into canonical ``TreeRecord`` artifacts."""

    records: list[TreeRecord] = []
    for tree in trees or ():
        leaves: list[TreeNode] = []
        for idx, leaf in enumerate(tree.leaves or ()):  # document-unit leaves
            leaf_meta = dict(leaf.metadata or {})
            unit_kind = str(leaf_meta.get("doc_unit_kind") or "unit")
            state = manifesto_policy_state_from_leaf(leaf) if leaf.score is not None else None
            leaves.append(
                TreeNode(
                    node_id=str(leaf.qid),
                    unit_type=unit_kind,
                    text=str(leaf.text),
                    level=0,
                    position=idx,
                    parent_id="root",
                    label=None if leaf.score is None else float(leaf.score),
                    state=state,
                    metadata={
                        **leaf_meta,
                        "doc_id": str(tree.doc_id),
                        "unit_id": make_unit_id(tree.doc_id, leaf.qid),
                        "score": leaf.score,
                        "weight": float(leaf.weight or 1.0),
                    },
                )
            )
        root = TreeNode(
            node_id="root",
            unit_type="root",
            text=str(tree.text),
            level=1 if leaves else 0,
            position=0,
            left_child_id=(leaves[0].node_id if len(leaves) >= 1 else None),
            right_child_id=(leaves[1].node_id if len(leaves) >= 2 else None),
            label=root_label(tree),
            metadata={
                "target": "f",
                "doc_id": str(tree.doc_id),
                "unit_id": make_unit_id(tree.doc_id, "root"),
                "root_label_name": "rile",
            },
        )
        records.append(
            TreeRecord(
                tree_id=str(tree.doc_id),
                doc_id=str(tree.doc_id),
                text=str(tree.text),
                root_label=root_label(tree),
                nodes=tuple([*leaves, root]),
                metadata=dict(tree.metadata or {}),
            )
        )
    return records


def replication_payload(trees: Sequence[ManifestoReplicationTree]) -> list[Mapping[str, Any]]:
    return [
        {
            "doc_id": tree.doc_id,
            "metadata": dict(tree.metadata),
            "leaves": [asdict(leaf) for leaf in tree.leaves],
        }
        for tree in trees
    ]


def _group_doc_units_into_leaves(
    qsentences: Sequence[ManifestoQSentence],
    *,
    leaf_unit_count: int,
    doc_unit_kind: str,
) -> tuple[ManifestoLeaf, ...]:
    q_rows = tuple(qsentences or ())
    unit_kind = str(doc_unit_kind or "qsentence")
    width = max(1, int(leaf_unit_count or 1))
    leaves: list[ManifestoLeaf] = []
    for start in range(0, len(q_rows), width):
        group = q_rows[start : start + width]
        if not group:
            continue
        if len(group) == 1:
            q = group[0]
            leaves.append(
                ManifestoLeaf(
                    text=str(q.text),
                    qid=str(q.qid),
                    code=str(q.code),
                    score=(None if q.score is None else float(q.score)),
                    weight=float(q.weight),
                    metadata={
                        **dict(q.metadata or {}),
                        "source_unit_ids": [str(q.qid)],
                        "doc_unit_kind": unit_kind,
                        "source_qids": [str(q.qid)],
                        "leaf_unit_count": 1,
                    },
                )
            )
            continue
        total_weight = sum(float(q.weight or 1.0) for q in group)
        scored_weight = sum(float(q.weight or 1.0) for q in group if q.score is not None)
        score = (
            sum(float(q.score) * float(q.weight or 1.0) for q in group if q.score is not None)
            / scored_weight
            if scored_weight > 0.0
            else None
        )
        qids = [str(q.qid) for q in group]
        leaves.append(
            ManifestoLeaf(
                text=" ".join(str(q.text) for q in group),
                qid="+".join(qids),
                code="",
                score=score,
                weight=total_weight if total_weight > 0.0 else float(len(group)),
                metadata={
                    "source_unit_ids": qids,
                    "doc_unit_kind": unit_kind,
                    "source_qids": qids,
                    "source_codes": [str(q.code) for q in group],
                    "leaf_unit_count": len(group),
                    "grouped_leaf": True,
                },
            )
        )
    return tuple(leaves)


__all__ = [
    "make_manifesto_replication_trees",
    "manifesto_tree_records",
    "replication_payload",
]
