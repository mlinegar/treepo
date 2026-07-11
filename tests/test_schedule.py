"""treepo.schedule is the single merge-schedule definition.

Index bookkeeping, generic folds, and the FNO model's level-vectorized
compose must all agree on the same topology: leaves ``0..L-1``, merges in
schedule order, root last, depths off real parent edges with root at 0.
"""

from __future__ import annotations

import pytest

from treepo.schedule import (
    fold,
    fold_with_trace,
    merge_children,
    merge_depths,
    merge_order,
)


def test_balanced_merge_order_carries_odd_leftover() -> None:
    # 5 leaves: level 1 merges (0,1),(2,3) -> nodes 5,6 with leaf 4 carried;
    # level 2 merges (5,6) -> node 7; level 3 merges (7,4) -> root 8.
    assert merge_order(5, schedule="balanced") == [(0, 1), (2, 3), (5, 6), (7, 4)]
    assert merge_children(5, schedule="balanced") == {
        5: (0, 1),
        6: (2, 3),
        7: (5, 6),
        8: (7, 4),
    }


def test_balanced_depths_match_the_pinned_convention() -> None:
    # Same expectation the FNO topology test pins via _pairwise_merge_depths.
    assert merge_depths(5, schedule="balanced") == [3, 3, 3, 3, 1, 2, 2, 1, 0]
    assert merge_depths(1, schedule="balanced") == [0]


def test_sequential_schedules_accumulate_with_running_left_child() -> None:
    assert merge_order(4, schedule="left_to_right") == [(0, 1), (4, 2), (5, 3)]
    assert merge_order(4, schedule="right_to_left") == [(3, 2), (4, 1), (5, 0)]
    # Root depth 0; the first-consumed leaf sits deepest.
    assert merge_depths(4, schedule="left_to_right") == [3, 3, 2, 1, 2, 1, 0]


def test_unknown_schedule_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported schedule"):
        merge_order(4, schedule="fourier")


def test_fold_with_trace_visits_every_node_in_trace_order() -> None:
    trace = fold_with_trace(
        [("a",), ("b",), ("c",)],
        lambda left, right: left + right,
        schedule="balanced",
    )
    assert trace == [("a",), ("b",), ("c",), ("a", "b"), ("a", "b", "c")]
    assert fold([("a",)], lambda l, r: l + r) == ("a",)


def test_fold_requires_at_least_one_leaf() -> None:
    with pytest.raises(ValueError, match="at least one leaf"):
        fold([], lambda l, r: l + r)


def test_fno_compose_trace_matches_the_canonical_topology() -> None:
    """The model's vectorized compose must produce trace rows in schedule order.

    The fast path keeps its own batched level loop for throughput; this test
    is the contract that its topology equals ``treepo.schedule``: every merge
    row is the merge of exactly the children the canonical schedule names.
    """

    from treepo.methods.families import resolve_family
    from treepo.methods.fixtures import make_markov_changepoint_trees

    family = resolve_family(
        "fno",
        {
            "embedding_dim": 8,
            "hidden_channels": 4,
            "n_modes": 2,
            "n_layers": 1,
            "head_hidden_dim": 8,
            "device": "cpu",
            "seed": 3,
        },
    )
    torch = family._torch
    # 40 tokens at 8 per leaf -> 5 leaves: odd count exercises the carry.
    tree = make_markov_changepoint_trees(
        n_trees=1,
        doc_tokens=40,
        leaf_unit_count=8,
        vocabulary_size=64,
        seed=7,
        split="test",
    )[0]
    family._ensure_model(output_dim=1)
    x, lengths = family._encode_trees([tree])
    family._model.eval()
    with torch.no_grad():
        _raw, traces = family._model.forward_with_trace(x, lengths)
        trace = traces[0]
        leaf_count = 5
        assert int(trace.shape[0]) == 2 * leaf_count - 1
        for node, (left, right) in merge_children(leaf_count, schedule="balanced").items():
            expected = family._model.merge(
                torch.cat([trace[left], trace[right]], dim=-1).unsqueeze(0)
            ).squeeze(0)
            assert torch.allclose(trace[node], expected, atol=1.0e-5), (
                f"trace node {node} is not merge({left}, {right})"
            )
