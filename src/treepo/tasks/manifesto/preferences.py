"""Preference records for Manifesto/RILE examples."""

from __future__ import annotations

from typing import Sequence

from treepo.methods.preference import Candidate, PreferenceDataset, PreferenceRecord
from treepo.sampling import ObservationUnitKind, SamplingMetadata
from treepo.state import make_unit_id
from treepo.tasks.manifesto.common import (
    document_propensity,
    root_label,
    sample_indices,
)
from treepo.tasks.manifesto.documents import ManifestoLeaf, ManifestoReplicationTree
from treepo.tasks.manifesto.rile import RILE_RANGE
from treepo.tasks.manifesto.state import manifesto_policy_state_from_leaf


def make_manifesto_preferences(
    trees: Sequence[ManifestoReplicationTree],
    *,
    mode: str = "scores",
    scope: str = "both",
    sample_size: int | None = None,
    sample_rate: float | None = None,
    seed: int = 0,
    min_sampled: int = 1,
) -> PreferenceDataset:
    """Build compact root and document-unit candidate preferences for examples."""

    if mode not in {"scores", "pairwise", "ranked"}:
        raise ValueError("mode must be 'scores', 'pairwise', or 'ranked'")
    if sample_size is not None and sample_rate is not None:
        raise ValueError("pass sample_size or sample_rate, not both")
    include_roots, include_units = _preference_scope_flags(scope)
    dataset = PreferenceDataset()
    for tree_idx, tree in enumerate(trees):
        tree_root_label = root_label(tree)
        if include_roots:
            dataset.append(
                _root_preference_record(tree, root_label=tree_root_label, mode=mode)
            )
        if include_units:
            for leaf, propensity in _sample_document_unit_leaves(
                tree,
                tree_index=tree_idx,
                sample_size=sample_size,
                sample_rate=sample_rate,
                seed=seed,
                min_sampled=min_sampled,
            ):
                dataset.append(
                    _document_unit_preference_record(
                        tree,
                        leaf=leaf,
                        propensity=propensity,
                        mode=mode,
                    )
                )
    return dataset


def _preference_scope_flags(scope: str) -> tuple[bool, bool]:
    normalized = str(scope or "both").strip().lower().replace("-", "_")
    aliases = {
        "all": "both",
        "both": "both",
        "root": "roots",
        "roots": "roots",
        "root_only": "roots",
        "f": "roots",
        "qsentence": "qsentences",
        "qsentences": "qsentences",
        "document_unit": "qsentences",
        "document_units": "qsentences",
        "unit": "qsentences",
        "units": "qsentences",
        "leaf": "qsentences",
        "leaves": "qsentences",
        "g": "qsentences",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        raise ValueError("scope must be one of: both, roots, qsentences")
    return resolved in {"both", "roots"}, resolved in {"both", "qsentences"}


def _sample_document_unit_leaves(
    tree: ManifestoReplicationTree,
    *,
    tree_index: int,
    sample_size: int | None,
    sample_rate: float | None,
    seed: int,
    min_sampled: int,
) -> list[tuple[ManifestoLeaf, float]]:
    leaves = list(tree.leaves or ())
    indices, propensity = sample_indices(
        len(leaves),
        sample_size=sample_size,
        sample_rate=sample_rate,
        seed=int(seed) + int(tree_index),
        min_sampled=min_sampled,
    )
    return [(leaves[idx], propensity) for idx in indices]


def _root_preference_record(
    tree: ManifestoReplicationTree,
    *,
    root_label: float,
    mode: str,
) -> PreferenceRecord:
    neutral = 0.0
    doc_propensity = document_propensity(tree)
    sampling = SamplingMetadata(
        document_propensity=doc_propensity,
        unit_propensity=1.0,
        label_propensity=1.0,
        joint_propensity=doc_propensity,
        sampling_scheme="uniform_without_replacement",
        policy_name=str(
            dict(tree.metadata or {}).get("document_sampling_policy")
            or "manifesto_document_uniform"
        ),
        unit_kind=ObservationUnitKind.DOCUMENT,
        metadata={"unit_type": "root"},
    )
    candidate_rows = [
        Candidate(
            id="teacher",
            value=f"RILE score: {root_label:g}",
            score=1.0,
            rank=1 if mode == "ranked" else None,
            preferred=mode == "pairwise",
            metadata={"oracle_target": root_label, "score": root_label},
        ),
        Candidate(
            id="neutral",
            value=f"RILE score: {neutral:g}",
            score=max(0.0, 1.0 - abs(root_label - neutral) / RILE_RANGE),
            rank=2 if mode == "ranked" else None,
            metadata={"oracle_target": neutral, "score": neutral},
        ),
    ]
    if mode == "ranked":
        candidate_rows.append(
            Candidate(
                id="wrong_sign",
                value=f"RILE score: {-root_label:g}",
                score=max(0.0, 1.0 - abs(root_label + root_label) / RILE_RANGE),
                rank=3,
                metadata={"oracle_target": -root_label, "score": -root_label},
            )
        )
    return PreferenceRecord(
        record_id=f"{tree.doc_id}:root:{mode}",
        unit_id=make_unit_id(tree.doc_id, "root"),
        unit_type="root",
        target="f",
        context=f"Estimate the document-level RILE score.\n\nDocument:\n{tree.text}",
        candidates=tuple(candidate_rows),
        weight=1.0,
        propensity=doc_propensity,
        tree_id=tree.doc_id,
        doc_id=tree.doc_id,
        node_id="root",
        level=None,
        metadata={
            "doc_id": tree.doc_id,
            "source_doc_id": tree.doc_id,
            "law_type": "root_label",
            "preference_mode": mode,
            "document_propensity": doc_propensity,
            "unit_propensity": 1.0,
            "label_propensity": 1.0,
            "joint_propensity": doc_propensity,
            "ipw_weight": sampling.ipw_weight(),
            "sampling": sampling.to_dict(),
        },
    )


def _document_unit_preference_record(
    tree: ManifestoReplicationTree,
    *,
    leaf: ManifestoLeaf,
    propensity: float,
    mode: str,
) -> PreferenceRecord:
    unit_id = str(leaf.qid)
    unit_kind = str((dict(leaf.metadata or {}).get("doc_unit_kind") or "unit"))
    text = str(leaf.text)
    score = leaf.score
    score_text = "unknown" if score is None else f"{float(score):g}"
    gold_state = manifesto_policy_state_from_leaf(leaf)
    doc_propensity = document_propensity(tree)
    unit_propensity = float(propensity or 1.0)
    joint_propensity = float(doc_propensity * unit_propensity)
    sampling = SamplingMetadata(
        document_propensity=doc_propensity,
        unit_propensity=unit_propensity,
        label_propensity=1.0,
        joint_propensity=joint_propensity,
        sampling_scheme="uniform_without_replacement",
        policy_name="manifesto_qsentence_uniform",
        unit_kind=ObservationUnitKind.LEAF,
        metadata={"doc_unit_kind": unit_kind},
    )
    candidates = [
        Candidate(
            id="specific",
            value=gold_state,
            score=1.0,
            rank=1 if mode == "ranked" else None,
            preferred=mode == "pairwise",
            metadata={
                "oracle_target": gold_state.to_dict(),
                "score": score,
                "state_kind": gold_state.kind,
            },
        ),
        Candidate(
            id="generic",
            value="This unit contains general campaign language.",
            score=0.25,
            rank=2 if mode == "ranked" else None,
        ),
    ]
    if mode == "ranked":
        candidates.append(
            Candidate(
                id="empty",
                value="No useful evidence.",
                score=0.0,
                rank=3,
            )
        )
    return PreferenceRecord(
        record_id=f"{tree.doc_id}:{unit_id}:{mode}",
        unit_id=make_unit_id(tree.doc_id, unit_id),
        unit_type=unit_kind,
        target="g",
        context=(
            f"Summarize this {unit_kind} for downstream RILE scoring.\n\n"
            f"Document unit:\n{text}"
        ),
        candidates=tuple(candidates),
        weight=float(leaf.weight or 1.0),
        propensity=joint_propensity,
        tree_id=tree.doc_id,
        doc_id=tree.doc_id,
        node_id=unit_id,
        level=0,
        metadata={
            "doc_id": tree.doc_id,
            "source_doc_id": tree.doc_id,
            "unit_id": unit_id,
            "unit_type": unit_kind,
            "qid": unit_id if unit_kind == "qsentence" else None,
            "law_type": "c1_leaf",
            "state_kind": gold_state.kind,
            "preference_mode": mode,
            "local_rile_evidence": score_text,
            "document_propensity": doc_propensity,
            "unit_propensity": unit_propensity,
            "label_propensity": 1.0,
            "joint_propensity": joint_propensity,
            "ipw_weight": sampling.ipw_weight(),
            "sampling": sampling.to_dict(),
        },
    )


__all__ = ["make_manifesto_preferences"]
