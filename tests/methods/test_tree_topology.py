"""Merge-topology and depth-convention contracts for the built-in families.

The forward pass (``_fno_models``), the supervision-target builder, and the
depth helper (``_fno_transition``) implement one pairwise-with-carry merge
schedule; these tests pin their alignment, including odd leaf counts where the
carried node changes both the row order and the depths.
"""

from __future__ import annotations

from pathlib import Path

from treepo.methods.families import resolve_family
from treepo.methods.fixtures import make_markov_changepoint_trees
from treepo.methods._fno_transition import (
    _numeric_transition_rows,
    _pairwise_merge_depths,
)
from treepo.tree import TreeRecord, local_law_rows_from_tree_records


def _tiny_config() -> dict[str, object]:
    return {
        "operator_kind": "conv1d",
        "embedding_dim": 8,
        "hidden_channels": 4,
        "n_layers": 1,
        "head_hidden_dim": 8,
        "epochs_per_iteration": 1,
        "batch_size": 4,
        "learning_rate": 0.01,
        "device": "cpu",
        "seed": 3,
    }


def test_transition_rows_cover_each_node_once() -> None:
    for leaf_count in (1, 2, 3, 5, 8):
        groups = [[i % 3 for i in range(4)] for _ in range(leaf_count)]
        rows = _numeric_transition_rows(groups, n_states=3, bucket=1, count_scale=4.0)
        assert len(rows) == 2 * leaf_count - 1
        depths = _pairwise_merge_depths(leaf_count)
        assert len(depths) == len(rows)
        assert depths[-1] == 0  # root is last and at depth 0


def test_pairwise_merge_depths_follow_carry_parent() -> None:
    # Five leaves: pairs (0,1) and (2,3) merge first, leaf 4 carries up and
    # merges directly under the root, so it sits at depth 1.
    assert _pairwise_merge_depths(5) == [3, 3, 3, 3, 1, 2, 2, 1, 0]


def test_model_trace_matches_targets_on_odd_leaf_count(tmp_path: Path) -> None:
    family = resolve_family(
        "neural_operator",
        {**_tiny_config(), "numeric_transition_state_weight": 0.05},
    )
    # doc_tokens=24, leaf_unit_count=8 -> 3 leaves per tree (odd, carry path).
    train = make_markov_changepoint_trees(
        n_trees=4,
        doc_tokens=24,
        leaf_unit_count=8,
        vocabulary_size=64,
        seed=7,
        split="train",
    )
    f_artifact = family.train_f(
        f_init=None, g=None, traces=train, output_dir=tmp_path / "f", iteration=1
    )
    family.train_g(
        g_init=None, f=f_artifact, traces=train, output_dir=tmp_path / "g", iteration=2
    )
    statistic = family.as_statistic()
    rows = statistic.local_law_rows(train)
    per_tree = len(rows) // len(train)
    assert per_tree == 2 * 3 - 1
    root_rows = [row for row in rows if row.metadata["node_index"] == per_tree - 1]
    assert all(row.depth == 0 for row in root_rows)


def test_tree_record_flows_through_family(tmp_path: Path) -> None:
    def record(idx: int, score: float) -> TreeRecord:
        return TreeRecord(
            tree_id=f"doc_{idx}",
            text="alpha beta gamma delta epsilon zeta",
            nodes=[
                {"node_id": "a", "text": "alpha beta", "parent_id": "root", "level": 0},
                {"node_id": "b", "text": "gamma delta", "parent_id": "root", "level": 0},
                {"node_id": "c", "text": "epsilon zeta", "parent_id": "root", "level": 0},
                {"node_id": "root", "unit_type": "root", "level": 1},
            ],
            metadata={"teacher_score_native": score},
        )

    trees = [record(idx, float(idx % 3)) for idx in range(4)]
    family = resolve_family("neural_operator", _tiny_config())
    f_artifact = family.train_f(
        f_init=None, g=None, traces=trees, output_dir=tmp_path / "f", iteration=1
    )
    scores = family.score_roots_with_f(f=f_artifact, g=None, trees=trees)
    assert len(scores) == len(trees)
    assert all(isinstance(score, float) for score in scores)


def test_local_law_rows_use_root_depth_zero() -> None:
    tree = TreeRecord(
        tree_id="t",
        nodes=[
            {
                "node_id": "leaf",
                "text": "leaf text",
                "parent_id": "root",
                "level": 0,
                "metadata": {"proxy_loss": 0.5},
            },
            {
                "node_id": "root",
                "unit_type": "root",
                "level": 1,
                "left_child_id": "leaf",
                "metadata": {"proxy_loss": 0.25},
            },
        ],
    )
    rows = local_law_rows_from_tree_records([tree])
    by_node = {row.metadata["node_id"]: row for row in rows}
    assert by_node["root"].depth == 0
    assert by_node["leaf"].depth == 1
