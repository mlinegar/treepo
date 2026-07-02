#!/usr/bin/env python3
"""Render trees as standalone expandable HTML files.

Five views, one per supported input style, all openable in any browser:

1. Manifesto/RILE qsentence trees with sampled-leaf markers, propensities/IPW
   weights, gold qsentence and root labels, and per-leaf policy-state
   summaries.
2. Markov changepoint trees scored by a trained neural-operator family:
   per-node ``f`` readouts and audited local-law losses on the synthesized
   merge tree (a sqrt node-audit design picks which nodes get oracle labels),
   next to exact per-leaf changepoint labels, with the AIPW audit summary as
   a panel.
3. LDA topic trees scored by a trained family: per-node readouts converging
   toward the exact target-topic proportion.
4. HLL item trees: exact distinct-count gold labels at every leaf and root.
5. Hand-built ``TreeRecord``s showing the general surface: gold labels next
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


def _train_tiny_family(trees: list, output_dir: Path, **config_overrides: object) -> "object":
    from treepo.methods.families import resolve_family

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
            **config_overrides,
        },
    )
    f_artifact = family.train_f(
        f_init=None, g=None, traces=trees, output_dir=output_dir / "f", iteration=1
    )
    family.train_g(
        g_init=None, f=f_artifact, traces=trees, output_dir=output_dir / "g", iteration=2
    )
    return family


def write_markov_law_view(output_dir: Path) -> Path:
    from treepo.local_law import audit_local_laws
    from treepo.methods.fixtures import (
        make_markov_changepoint_trees,
        markov_tree_records,
    )
    from treepo.sampling import apply_node_audit, sample_node_audit
    from treepo.viz import write_tree_visualization_html

    trees = make_markov_changepoint_trees(
        n_trees=4,
        doc_tokens=48,
        leaf_unit_count=16,
        vocabulary_size=64,
        seed=11,
        split="train",
    )
    family = _train_tiny_family(trees, output_dir / "markov")
    statistic = family.as_statistic()
    # The statistic labels every node; the audit design chooses which nodes
    # keep their oracle labels (sqrt policy, logged propensity q/N per tree).
    audited_rows = []
    for tree_idx, tree in enumerate(trees):
        prefix = f"{tree.metadata['tree_id']}:state:"
        tree_rows = [
            row for row in statistic.local_law_rows([tree]) if row.row_id.startswith(prefix)
        ]
        design = sample_node_audit(len(tree_rows), policy="sqrt", seed=tree_idx)
        audited_rows.extend(apply_node_audit(tree_rows, design))
    audit = audit_local_laws(audited_rows, gamma_depth=0.9)
    curve = _markov_tradeoff_curve(output_dir / "markov_tradeoff")
    curve.write(
        json_out=output_dir / "markov_tradeoff_curve.json",
        csv_out=output_dir / "markov_tradeoff_curve.csv",
    )
    return write_tree_visualization_html(
        markov_tree_records(trees),
        output_dir / "markov_law_trees.html",
        law_rows=audited_rows,
        readout_rows=statistic.node_readouts(trees),
        audit=audit,
        tradeoff=curve.to_dict(),
        title="Markov trees: audited local-law losses, node readouts, tradeoff",
    )


def _markov_tradeoff_curve(output_dir: Path) -> "object":
    """Sweep leaf grouping sizes and record held-out root error per size."""

    from treepo.methods.fixtures import make_markov_changepoint_trees
    from treepo.methods.tradeoff import TradeoffCurve

    rows = []
    for leaf_unit_count in (8, 16, 48):
        train = make_markov_changepoint_trees(
            n_trees=6,
            doc_tokens=48,
            leaf_unit_count=leaf_unit_count,
            vocabulary_size=64,
            seed=11,
            split="train",
        )
        eval_trees = make_markov_changepoint_trees(
            n_trees=6,
            doc_tokens=48,
            leaf_unit_count=leaf_unit_count,
            vocabulary_size=64,
            seed=12,
            split="test",
        )
        family = _train_tiny_family(train, output_dir / f"leaf_{leaf_unit_count:03d}")
        scores = family.score_roots_with_f(f=None, g=None, trees=eval_trees)
        golds = [float(tree.metadata["teacher_score_native"]) for tree in eval_trees]
        mae = sum(abs(score - gold) for score, gold in zip(scores, golds)) / len(golds)
        rows.append({"leaf_unit_count": leaf_unit_count, "root_mae": mae})
    return TradeoffCurve.from_rows(
        rows,
        metric_keys=("root_mae",),
        metadata={"task": "markov_changepoint", "family": "neural_operator/conv1d"},
    )


def write_lda_readout_view(output_dir: Path) -> Path:
    from treepo.methods.fixtures import lda_tree_records, make_lda_topic_trees
    from treepo.viz import write_tree_visualization_html

    trees = make_lda_topic_trees(
        n_trees=4,
        n_topics=3,
        doc_tokens=48,
        leaf_unit_count=16,
        vocabulary_size=30,
        seed=7,
        split="train",
    )
    # target_dim=3 trains against the full topic-proportion vector, so gold
    # labels and node readouts are both 3-vectors.
    family = _train_tiny_family(trees, output_dir / "lda", target_dim=3)
    return write_tree_visualization_html(
        lda_tree_records(trees, vector_labels=True),
        output_dir / "lda_readout_trees.html",
        readout_rows=family.as_statistic().node_readouts(trees),
        title="LDA trees: vector node readouts vs exact topic proportions",
    )


def write_hll_view(output_dir: Path) -> Path:
    from treepo.methods.fixtures import hll_tree_records, make_hll_item_trees
    from treepo.viz import write_tree_visualization_html

    trees = make_hll_item_trees(
        n_trees=4, leaves_per_tree=4, leaf_unit_count=16, vocabulary_size=64, seed=2
    )
    return write_tree_visualization_html(
        hll_tree_records(trees),
        output_dir / "hll_trees.html",
        title="HLL item trees: exact distinct counts per leaf and root",
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
    views = (
        write_manifesto_view,
        write_markov_law_view,
        write_lda_readout_view,
        write_hll_view,
        write_generic_view,
    )
    for view in views:
        print(f"wrote {view(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
