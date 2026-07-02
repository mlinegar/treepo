"""Document and qsentence sampling helpers for Manifesto/RILE fixtures."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from treepo.sampling import DocumentSamplingRow, ObservationUnitKind, SamplingMetadata
from treepo.state import make_unit_id
from treepo.tasks.manifesto.common import (
    document_propensity,
    root_label,
    sample_indices,
)
from treepo.tasks.manifesto.documents import ManifestoReplicationTree


def sample_manifesto_replication_trees(
    trees: Sequence[ManifestoReplicationTree],
    *,
    sample_size: int | None = None,
    sample_rate: float | None = None,
    seed: int = 0,
    min_sampled: int = 1,
    policy_name: str = "manifesto_document_uniform",
) -> tuple[list[ManifestoReplicationTree], list[dict[str, Any]]]:
    """Sample documents and attach logged design propensities.

    Returned sampling rows cover the full document population. Returned trees
    are only observed documents and carry ``document_propensity`` in metadata
    for downstream root/qsentence supervision.
    """

    population = list(trees or ())
    indices, propensity = sample_indices(
        len(population),
        sample_size=sample_size,
        sample_rate=sample_rate,
        seed=seed,
        min_sampled=min_sampled,
    )
    observed_indices = set(indices)
    selected: list[ManifestoReplicationTree] = []
    rows: list[dict[str, Any]] = []
    for tree_idx, tree in enumerate(population):
        meta = dict(tree.metadata or {})
        observed = tree_idx in observed_indices
        rows.append(
            DocumentSamplingRow(
                top_level_unit_id=str(tree.doc_id),
                observed=observed,
                inclusion_probability=float(propensity),
                truth=root_label(tree),
                split=str(meta.get("split") or ""),
                metadata={
                    "doc_id": str(tree.doc_id),
                    "tree_index": tree_idx,
                    "policy_name": policy_name,
                    "sampling_scheme": "uniform_without_replacement",
                    "population_size": len(population),
                    "sample_size": len(indices),
                },
            ).to_dict()
        )
        if not observed:
            continue
        sampling = SamplingMetadata(
            document_propensity=float(propensity),
            unit_propensity=1.0,
            label_propensity=1.0,
            sampling_scheme="uniform_without_replacement",
            policy_name=policy_name,
            unit_kind=ObservationUnitKind.DOCUMENT,
            metadata={"population_size": len(population), "sample_size": len(indices)},
        )
        meta.update(
            {
                "document_observed": True,
                "document_propensity": float(propensity),
                "document_sampling_index": tree_idx,
                "document_sampling_policy": policy_name,
                "document_sampling": sampling.to_dict(),
            }
        )
        selected.append(replace(tree, metadata=meta))
    return selected, rows


def manifesto_document_unit_sampling_rows(
    trees: Sequence[ManifestoReplicationTree],
    *,
    sample_size: int | None = None,
    sample_rate: float | None = None,
    seed: int = 0,
    min_sampled: int = 1,
    policy_name: str = "manifesto_qsentence_uniform",
) -> list[dict[str, Any]]:
    """Return observed/unobserved qsentence sampling rows for the DSL view."""

    rows: list[dict[str, Any]] = []
    for tree_idx, tree in enumerate(trees or ()):
        leaves = list(tree.leaves or ())
        indices, unit_propensity = sample_indices(
            len(leaves),
            sample_size=sample_size,
            sample_rate=sample_rate,
            seed=int(seed) + int(tree_idx),
            min_sampled=min_sampled,
        )
        selected = set(indices)
        doc_propensity = document_propensity(tree)
        joint_propensity = float(doc_propensity * unit_propensity)
        for leaf_idx, leaf in enumerate(leaves):
            leaf_meta = dict(leaf.metadata or {})
            unit_kind = str(leaf_meta.get("doc_unit_kind") or "unit")
            unit_id = str(leaf.qid)
            row: dict[str, Any] = {
                "tree_id": str(tree.doc_id),
                "doc_id": str(tree.doc_id),
                "node_id": unit_id,
                "unit_id": make_unit_id(tree.doc_id, unit_id),
                "unit_type": unit_kind,
                "observed": leaf_idx in selected,
                "document_propensity": float(doc_propensity),
                "unit_propensity": float(unit_propensity),
                "label_propensity": 1.0,
                "joint_propensity": joint_propensity,
                "inclusion_probability": joint_propensity,
                "ipw_weight": (None if joint_propensity <= 0.0 else float(1.0 / joint_propensity)),
                "sampling_scheme": "uniform_without_replacement",
                "policy_name": policy_name,
                "tree_index": tree_idx,
                "unit_index": leaf_idx,
                "population_size": len(leaves),
                "sample_size": len(indices),
                "metadata": {
                    "source_qids": list(leaf_meta.get("source_qids") or [unit_id]),
                    "source_codes": list(
                        leaf_meta.get("source_codes") or ([leaf.code] if leaf.code else [])
                    ),
                    "score": leaf.score,
                },
            }
            if joint_propensity > 0.0:
                row["sampling"] = SamplingMetadata(
                    document_propensity=float(doc_propensity),
                    unit_propensity=float(unit_propensity),
                    label_propensity=1.0,
                    joint_propensity=joint_propensity,
                    sampling_scheme="uniform_without_replacement",
                    policy_name=policy_name,
                    unit_kind=ObservationUnitKind.LEAF,
                    metadata={
                        "doc_unit_kind": unit_kind,
                        "population_size": len(leaves),
                        "sample_size": len(indices),
                    },
                ).to_dict()
            rows.append(row)
    return rows


__all__ = [
    "manifesto_document_unit_sampling_rows",
    "sample_manifesto_replication_trees",
]
