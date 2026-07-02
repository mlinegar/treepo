#!/usr/bin/env python3
"""Render trees as standalone expandable HTML files.

Three views, one per supported input style, all openable in any browser:

1. Manifesto/RILE qsentence trees with sampled-leaf markers, propensities/IPW
   weights, gold qsentence and root labels, and per-leaf policy-state
   summaries.
2. Markov changepoint trees scored by a trained neural-operator family, with
   local-law proxy losses shown on the synthesized pairwise merge tree the
   family actually computes, next to exact per-leaf changepoint labels.
3. Hand-built ``TreeRecord``s showing the general surface: gold labels next
   to prediction metadata, LLM summaries, sampling rows, and node-keyed
   local-law rows from node metadata.
"""

from __future__ import annotations

from pathlib import Path


def write_manifesto_view(output_dir: Path) -> Path:
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
    return write_tree_visualization_html(
        manifesto_tree_records(observed),
        output_dir / "manifesto_trees.html",
        sampling_rows=unit_rows,
        title="Manifesto qsentence trees: sampling and gold labels",
    )


def write_markov_law_view(output_dir: Path) -> Path:
    from treepo.methods.families import resolve_family
    from treepo.methods.fixtures import (
        make_markov_changepoint_trees,
        markov_tree_records,
    )
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
        markov_tree_records(trees),
        output_dir / "markov_law_trees.html",
        law_rows=law_rows,
        title="Markov trees: local-law losses on the merge tree",
    )


def write_generic_view(output_dir: Path) -> Path:
    from treepo.tree import TreeRecord, local_law_rows_from_tree_records
    from treepo.viz import write_tree_visualization_html

    def record(idx: int) -> TreeRecord:
        return TreeRecord(
            tree_id=f"doc_{idx}",
            root_label=0.4 + 0.1 * idx,
            nodes=[
                {
                    "node_id": "para_0",
                    "text": "The committee recommends expanding the pilot program.",
                    "parent_id": "root",
                    "level": 0,
                    "position": 0,
                    "label": 0.6,
                    "state": {
                        "kind": "note",
                        "text": "Supports expansion of the pilot.",
                    },
                    "metadata": {"llm_score": 0.55, "proxy_loss": 0.02},
                },
                {
                    "node_id": "para_1",
                    "text": "Budget constraints require a phased rollout.",
                    "parent_id": "root",
                    "level": 0,
                    "position": 1,
                    "label": 0.2,
                    "metadata": {
                        "llm_score": 0.3,
                        "llm_summary": "Flags budget limits; prefers phasing.",
                        "proxy_loss": 0.08,
                    },
                },
                {
                    "node_id": "root",
                    "unit_type": "root",
                    "level": 1,
                    "label": 0.4 + 0.1 * idx,
                    "metadata": {"proxy_loss": 0.05},
                },
            ],
        )

    records = [record(idx) for idx in range(2)]
    sampling_rows = [
        {
            "tree_id": f"doc_{idx}",
            "node_id": f"para_{para}",
            "observed": (idx + para) % 2 == 0,
            "joint_propensity": 0.5,
            "ipw_weight": 2.0,
        }
        for idx in range(2)
        for para in range(2)
    ]
    return write_tree_visualization_html(
        records,
        output_dir / "generic_trees.html",
        sampling_rows=sampling_rows,
        law_rows=local_law_rows_from_tree_records(records),
        title="Hand-built tree records: labels, summaries, sampling, laws",
    )


def main() -> int:
    output_dir = Path("outputs/tree_visualization")
    for view in (write_manifesto_view, write_markov_law_view, write_generic_view):
        print(f"wrote {view(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
