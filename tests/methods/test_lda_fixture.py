from __future__ import annotations

import math
from pathlib import Path

from treepo.methods import run
from treepo.methods.fixtures import make_lda_topic_trees
from treepo.methods.lda import fit_sklearn_lda_baseline


def _tiny_neural_operator_config() -> dict[str, object]:
    return {
        "embedding_dim": 8,
        "hidden_channels": 4,
        "n_modes": 2,
        "n_layers": 1,
        "head_hidden_dim": 8,
        "epochs_per_iteration": 1,
        "batch_size": 4,
        "learning_rate": 0.01,
        "device": "cpu",
        "seed": 5,
    }


def test_lda_fixture_has_exact_topic_proportion_targets() -> None:
    trees = make_lda_topic_trees(
        n_trees=4,
        n_topics=3,
        doc_tokens=48,
        leaf_token_count=12,
        vocabulary_size=30,
        seed=7,
        split="train",
    )

    assert len(trees) == 4
    for tree in trees:
        assert len(tree.tokens) == 48
        assert len(tree.topics) == 48
        assert len(tree.leaves) == 4
        assert math.isclose(sum(tree.topic_proportions), 1.0)
        assert math.isclose(
            tree.metadata["topic_0_proportion"],
            tree.topic_proportions[0],
        )
        assert tree.metadata["teacher_score_native"] == tree.topic_proportions[0]
        topic_words = tree.metadata["topic_word_distributions"]
        assert len(topic_words) == 3
        assert all(len(row) == 30 for row in topic_words)
        for row in topic_words:
            assert math.isclose(sum(row), 1.0)
            assert all(float(value) > 0.0 for value in row)
        overlap_mass = sum(min(topic_words[0][idx], topic_words[1][idx]) for idx in range(30))
        assert overlap_mass > 0.0


def test_lda_leaf_sizes_keep_documents_fixed() -> None:
    small_leaves = make_lda_topic_trees(
        n_trees=3,
        n_topics=3,
        doc_tokens=48,
        leaf_token_count=8,
        vocabulary_size=30,
        topic_seed=11,
        seed=12,
        split="train",
    )
    large_leaves = make_lda_topic_trees(
        n_trees=3,
        n_topics=3,
        doc_tokens=48,
        leaf_token_count=24,
        vocabulary_size=30,
        topic_seed=11,
        seed=12,
        split="train",
    )

    assert small_leaves[0].tokens == large_leaves[0].tokens
    assert small_leaves[0].topics == large_leaves[0].topics
    assert small_leaves[0].topic_proportions == large_leaves[0].topic_proportions
    assert len(small_leaves[0].leaves) == 6
    assert len(large_leaves[0].leaves) == 2


def test_sklearn_lda_baseline_runs_on_overlapping_topics() -> None:
    train = make_lda_topic_trees(
        n_trees=80,
        n_topics=3,
        doc_tokens=160,
        leaf_token_count=20,
        vocabulary_size=45,
        target_topic=0,
        topic_word_concentration=0.05,
        seed=27,
        split="train",
    )
    eval_trees = make_lda_topic_trees(
        n_trees=20,
        n_topics=3,
        doc_tokens=160,
        leaf_token_count=20,
        vocabulary_size=45,
        target_topic=0,
        topic_word_concentration=0.05,
        seed=28,
        split="test",
    )

    baseline = fit_sklearn_lda_baseline(
        train,
        eval_trees,
        n_topics=3,
        vocabulary_size=45,
        doc_topic_prior=0.7,
        topic_word_prior=0.05,
        max_iter=50,
        random_state=0,
        target_topic=0,
    )

    assert baseline.target_key == "topic_0_proportion"
    assert len(baseline.topic_order) == 3
    assert baseline.mean_mae < 0.2
    assert baseline.target_mae < 0.25
    assert 0.0 <= baseline.average_guess_target <= 1.0
    assert baseline.average_guess_target_mae > baseline.target_mae


def test_neural_operator_runs_on_lda_topic_proportions(tmp_path: Path) -> None:
    train = make_lda_topic_trees(
        n_trees=8,
        n_topics=3,
        doc_tokens=48,
        leaf_token_count=12,
        vocabulary_size=30,
        target_topic=0,
        seed=17,
        split="train",
    )
    eval_trees = make_lda_topic_trees(
        n_trees=5,
        n_topics=3,
        doc_tokens=48,
        leaf_token_count=12,
        vocabulary_size=30,
        target_topic=0,
        seed=18,
        split="test",
    )

    result = run(
        "fit",
        {
            "family": "neural_operator",
            "train_data": train,
            "eval_data": eval_trees,
            "backend_config": {
                **_tiny_neural_operator_config(),
                "operator_kind": "fno",
                "target_key": "topic_0_proportion",
                "target_vector_key": "topic_proportions",
                "target_dim": 3,
                "target_min": 0.0,
                "target_max": 1.0,
                "output_dir": str(tmp_path),
            },
            "axis": {"max_iterations": 3, "axis_value": 0},
        },
    )

    assert result.status == "success"
    assert result.summary["family"] == "neural_operator"
    assert result.artifacts["f"]["operator_kind"] == "fno"
    assert result.artifacts["f"]["output_dim"] == 3
    assert result.artifacts["g"]["trained"] == "g"
    assert result.metrics["n"] == 5.0
    assert math.isfinite(float(result.metrics["internal_f_mae"]))
    for topic_idx in range(3):
        assert math.isfinite(float(result.metrics[f"topic_{topic_idx}_internal_f_mae"]))
    assert result.artifacts["prediction_records"]
