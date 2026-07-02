#!/usr/bin/env python3
"""Render trees as standalone expandable HTML files.

Two views, both openable in any browser:

1. Manifesto/RILE qsentence trees with sampled-leaf markers, propensities/IPW
   weights, and gold qsentence and root labels.
2. Markov changepoint trees scored by a trained neural-operator family, with
   per-node local-law proxy losses shown on the synthesized pairwise merge
   tree the family actually computes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def markov_tree_record(tree: Any) -> "Any":
    from treepo.tree import TreeRecord

    metadata = dict(tree.metadata or {})
    nodes: list[dict[str, Any]] = [
        {
            "node_id": f"leaf_{idx}",
            "unit_type": "leaf",
            "text": " ".join(str(token) for token in leaf.tokens),
            "parent_id": "root",
            "level": 0,
            "position": idx,
        }
        for idx, leaf in enumerate(tree.leaves)
    ]
    nodes.append(
        {
            "node_id": "root",
            "unit_type": "root",
            "level": 1,
            "label": metadata.get("teacher_score_native"),
        }
    )
    return TreeRecord(
        tree_id=str(metadata.get("tree_id") or "markov"),
        nodes=nodes,
        root_label=metadata.get("teacher_score_native"),
        metadata=metadata,
    )


def write_markov_law_view(output_dir: Path) -> Path:
    from treepo.methods.families import resolve_family
    from treepo.methods.fixtures import make_markov_changepoint_trees
    from treepo.viz import write_tree_visualization_html

    trees = make_markov_changepoint_trees(
        n_trees=4,
        doc_tokens=48,
        leaf_unit_count=16,
        vocabulary_size=64,
        seed=11,
        split="train",
    )
    family = resolve_family(
        "neural_operator",
        {
            "operator_kind": "conv1d",
            "embedding_dim": 8,
            "hidden_channels": 4,
            "n_layers": 1,
            "head_hidden_dim": 8,
            "epochs_per_iteration": 2,
            "batch_size": 4,
            "numeric_transition_state_weight": 0.05,
            "device": "cpu",
            "seed": 3,
        },
    )
    f_artifact = family.train_f(
        f_init=None, g=None, traces=trees, output_dir=output_dir / "f", iteration=1
    )
    family.train_g(
        g_init=None, f=f_artifact, traces=trees, output_dir=output_dir / "g", iteration=2
    )
    law_rows = family.as_statistic().local_law_rows(trees)
    return write_tree_visualization_html(
        [markov_tree_record(tree) for tree in trees],
        output_dir / "markov_law_trees.html",
        law_rows=law_rows,
        title="Markov trees: local-law losses on the merge tree",
    )


def main() -> int:
    from treepo.tasks.manifesto import (
        make_manifesto_replication_trees,
        manifesto_document_unit_sampling_rows,
        manifesto_tree_records,
        sample_manifesto_replication_trees,
    )
    from treepo.viz import write_tree_visualization_html

    trees = make_manifesto_replication_trees(split="test", leaf_unit_count=1)
    observed, _doc_rows = sample_manifesto_replication_trees(
        trees, sample_rate=0.75, seed=0
    )
    unit_rows = manifesto_document_unit_sampling_rows(
        observed, sample_rate=0.5, seed=0
    )
    records = manifesto_tree_records(observed)
    output_dir = Path("outputs/tree_visualization")
    out = write_tree_visualization_html(
        records,
        output_dir / "manifesto_trees.html",
        sampling_rows=unit_rows,
        title="Manifesto qsentence trees: sampling and gold labels",
    )
    print(f"wrote {out}")
    print(f"wrote {write_markov_law_view(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
