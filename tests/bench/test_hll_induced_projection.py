from __future__ import annotations

import math

import pytest


def _torch_or_skip():
    try:
        import torch
    except Exception:
        pytest.skip("torch not available")
    return torch


def test_induced_projection_uses_same_g_for_leaves_and_merge() -> None:
    torch = _torch_or_skip()
    from treepo.bench.hll_merge_learning import InducedProjectionHLLMerger, merge_leaf_states

    class AddOneProjection(InducedProjectionHLLMerger):
        def __init__(self):
            super().__init__(precision=4, hidden_dim=4)
            self.project_calls = 0

        def project(self, state):
            self.project_calls += 1
            return state + 1.0

    model = AddOneProjection()
    left = torch.ones(16)
    right = 2.0 * torch.ones(16)

    root = merge_leaf_states(model, [left, right], schedule="balanced")

    assert model.project_calls == 3
    assert torch.allclose(root, 6.0 * torch.ones(16))


def test_rollout_depths_match_carried_balanced_tree() -> None:
    torch = _torch_or_skip()
    from treepo.bench.hll_merge_learning import ExactMaxMerger, _rollout_hll_node_rows

    leaves = [torch.full((16,), float(i)) for i in range(3)]
    _, _, depths, n_internal = _rollout_hll_node_rows(ExactMaxMerger(), leaves)

    assert n_internal == 2
    assert depths.tolist() == [2, 2, 1, 1, 0]


def test_hll_merge_learning_induced_projection_reports_adjusted_loss_fields() -> None:
    _torch_or_skip()
    from treepo.bench.hll_merge_learning import (
        HLLMergeLearningConfig,
        experiment_rows,
        run_hll_merge_learning_experiment,
    )

    runs = run_hll_merge_learning_experiment(
        HLLMergeLearningConfig(
            model_kind="induced_projection",
            precisions=(4,),
            train_docs_grid=(4,),
            audit_policies=("fraction",),
            min_tokens=32,
            max_tokens=32,
            leaf_size=16,
            n_test=4,
            n_epochs=1,
            batch_docs=2,
            hidden_dim=4,
            use_cuda=False,
            seed=0,
        )
    )
    row = experiment_rows(runs)[0]

    assert row["model_kind"] == "induced_projection"
    assert row["objective_mode"] == "corrected_local_law"
    assert row["proxy_mode"] == "frozen_rollout"
    assert "proxy + R / pi" in row["lean_adjusted_loss"]
    assert "g_theta(a+b)" in row["lean_merge_adapter"]
    assert "f*(x+y)" in row["lean_projection_target"]
    assert row["observed_rows_mean"] > 0
    assert math.isfinite(row["corrected_local_law_loss_mean"])


def test_hll_merge_learning_direct_state_mlp_remains_legacy_ablation() -> None:
    _torch_or_skip()
    from treepo.bench.hll_merge_learning import (
        HLLMergeLearningConfig,
        experiment_rows,
        run_hll_merge_learning_experiment,
    )

    runs = run_hll_merge_learning_experiment(
        HLLMergeLearningConfig(
            model_kind="direct_state_mlp",
            precisions=(4,),
            train_docs_grid=(4,),
            audit_policies=("all",),
            min_tokens=32,
            max_tokens=32,
            leaf_size=16,
            n_test=4,
            n_epochs=1,
            batch_docs=2,
            hidden_dim=4,
            use_cuda=False,
            seed=1,
        )
    )
    row = experiment_rows(runs)[0]

    assert row["model_kind"] == "direct_state_mlp"
    assert row["objective_mode"] == "direct_state_mse"
    assert row["proxy_mode"] == "none"
    assert math.isnan(row["corrected_local_law_loss_mean"])
