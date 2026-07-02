"""Fixture-to-TreeRecord converters share one contract across DGPs."""

from __future__ import annotations

import pytest

from treepo.methods.fixtures import (
    hll_tree_records,
    lda_tree_records,
    make_hll_item_trees,
    make_lda_topic_trees,
    make_markov_changepoint_trees,
    markov_tree_records,
)
from treepo.tree import validate_tree_record


def test_hll_records_carry_exact_distinct_counts() -> None:
    trees = make_hll_item_trees(
        n_trees=2, leaves_per_tree=3, leaf_unit_count=8, vocabulary_size=16, seed=4
    )
    records = hll_tree_records(trees)
    assert len(records) == 2
    for tree, record in zip(trees, records):
        assert not validate_tree_record(record)
        leaves = record.leaves()
        assert len(leaves) == 3
        for leaf, fixture_leaf in zip(leaves, tree.leaves):
            assert leaf.label == len(set(int(t) for t in fixture_leaf.tokens))
        assert record.root_label == tree.metadata["teacher_score_native"]


def test_lda_records_carry_target_topic_proportions() -> None:
    trees = make_lda_topic_trees(
        n_trees=2,
        n_topics=3,
        doc_tokens=32,
        leaf_unit_count=8,
        vocabulary_size=24,
        seed=6,
    )
    records = lda_tree_records(trees)
    for tree, record in zip(trees, records):
        assert not validate_tree_record(record)
        target_topic = int(tree.metadata["target_topic"])
        for leaf, fixture_leaf in zip(record.leaves(), tree.leaves):
            topics = [int(t) for t in fixture_leaf.topics]
            expected = topics.count(target_topic) / max(1, len(topics))
            assert leaf.label == pytest.approx(expected)
            proportions = leaf.metadata["leaf_topic_proportions"]
            assert sum(proportions) == pytest.approx(1.0)
        assert record.root_label == pytest.approx(
            float(tree.topic_proportions[target_topic])
        )


def test_lda_vector_labels_and_vector_readouts(tmp_path) -> None:
    from treepo.methods.families import resolve_family

    trees = make_lda_topic_trees(
        n_trees=2,
        n_topics=3,
        doc_tokens=32,
        leaf_unit_count=8,
        vocabulary_size=24,
        seed=6,
    )
    records = lda_tree_records(trees, vector_labels=True)
    for tree, record in zip(trees, records):
        assert record.root_label == pytest.approx(list(tree.topic_proportions))
        for leaf in record.leaves():
            assert len(leaf.label) == 3
            assert sum(leaf.label) == pytest.approx(1.0)

    family = resolve_family(
        "neural_operator",
        {
            "operator_kind": "conv1d",
            "embedding_dim": 8,
            "hidden_channels": 4,
            "n_layers": 1,
            "head_hidden_dim": 8,
            "epochs_per_iteration": 1,
            "batch_size": 4,
            "target_dim": 3,
            "device": "cpu",
            "seed": 3,
        },
    )
    family.train_f(
        f_init=None, g=None, traces=trees, output_dir=tmp_path / "f", iteration=1
    )
    readouts = family.as_statistic().node_readouts(trees)
    leaf_count = len(trees[0].leaves)
    assert len(readouts) == len(trees) * (2 * leaf_count - 1)
    assert all(
        isinstance(row["value"], list) and len(row["value"]) == 3 for row in readouts
    )


def test_converters_share_the_record_shape() -> None:
    markov = markov_tree_records(
        make_markov_changepoint_trees(
            n_trees=1, doc_tokens=32, leaf_unit_count=8, vocabulary_size=32, seed=1
        )
    )
    lda = lda_tree_records(
        make_lda_topic_trees(
            n_trees=1, doc_tokens=32, leaf_unit_count=8, vocabulary_size=24, seed=1
        )
    )
    hll = hll_tree_records(
        make_hll_item_trees(
            n_trees=1, leaves_per_tree=4, leaf_unit_count=8, vocabulary_size=32, seed=1
        )
    )
    for record in (*markov, *lda, *hll):
        root = record.root()
        assert root is not None and root.unit_type == "root"
        leaves = record.leaves()
        assert leaves and all(leaf.parent_id == "root" for leaf in leaves)
        assert all(leaf.label is not None for leaf in leaves)
        assert record.root_label is not None
        # tree_id comes from fixture metadata, so records line up with the
        # statistic's "<tree_id>:state:<index>" law-row and readout ids.
        assert record.tree_id == record.metadata["tree_id"]
