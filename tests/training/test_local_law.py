from __future__ import annotations

import pytest

from treepo.training.local_law import (
    corrected_local_law_loss,
    corrected_local_law_loss_tensor,
    local_law_objective_target_mse,
    observed_uniform_node_ipw_mean_loss,
    sampled_uniform_node_ipw_mean_loss,
)


def _torch_or_skip():
    try:
        import torch
    except Exception:
        pytest.skip("torch not available")
    return torch


def test_corrected_local_law_loss_keeps_proxy_when_unsampled() -> None:
    assert corrected_local_law_loss(
        proxy_loss=0.4,
        oracle_loss=None,
        observed=False,
        propensity=0.0,
    ) == pytest.approx(0.4)


def test_corrected_local_law_loss_applies_sampled_ipw_residual() -> None:
    assert corrected_local_law_loss(
        proxy_loss=0.4,
        oracle_loss=0.1,
        observed=True,
        propensity=0.5,
    ) == pytest.approx(-0.2)


def test_corrected_local_law_loss_rejects_invalid_observed_propensity() -> None:
    with pytest.raises(ValueError, match="propensity"):
        corrected_local_law_loss(
            proxy_loss=0.4,
            oracle_loss=0.1,
            observed=True,
            propensity=0.0,
        )


def test_corrected_local_law_loss_tensor_matches_adjusted_formula() -> None:
    torch = _torch_or_skip()
    rows = corrected_local_law_loss_tensor(
        proxy_loss=torch.tensor([0.4, 0.4]),
        oracle_loss=torch.tensor([0.1, 0.2]),
        observed=torch.tensor([False, True]),
        propensity=torch.tensor([0.0, 0.5]),
    )
    assert rows.tolist() == pytest.approx([0.4, 0.0])


def test_local_law_objective_target_mse_uses_proxy_plus_ipw_correction() -> None:
    torch = _torch_or_skip()
    objective = local_law_objective_target_mse(
        predictions=torch.tensor([1.0, 2.0]),
        proxy_targets=torch.tensor([1.5, 3.0]),
        oracle_targets=torch.tensor([1.0, 2.5]),
        observed=torch.tensor([False, True]),
        propensity=torch.tensor([0.0, 0.5]),
        depths=torch.tensor([0, 1]),
        gamma_depth=0.5,
    )
    assert float(objective) == pytest.approx((0.25 + 0.5 * -0.5) / 1.5)


def test_local_law_objective_rejects_gamma_above_one() -> None:
    torch = _torch_or_skip()
    with pytest.raises(ValueError, match="gamma_depth"):
        local_law_objective_target_mse(
            predictions=torch.tensor([1.0]),
            proxy_targets=torch.tensor([1.0]),
            oracle_targets=torch.tensor([1.0]),
            observed=torch.tensor([True]),
            propensity=torch.tensor([1.0]),
            depths=torch.tensor([0]),
            gamma_depth=1.5,
        )


def test_sampled_uniform_node_ipw_mean_loss_full_rate_uses_node_weights() -> None:
    torch = _torch_or_skip()
    objective = sampled_uniform_node_ipw_mean_loss(
        torch.tensor([[1.0, 9.0, 16.0]]),
        rate=1.0,
        node_weights=torch.tensor([[1.0, 2.0, 1.0]]),
    )
    assert float(objective) == pytest.approx((1.0 + 2.0 * 9.0 + 16.0) / 4.0)


def test_sampled_uniform_node_ipw_mean_loss_zero_rate_returns_zero() -> None:
    torch = _torch_or_skip()
    objective = sampled_uniform_node_ipw_mean_loss(
        torch.tensor([[1.0, 9.0, 16.0]]),
        rate=0.0,
    )
    assert float(objective) == pytest.approx(0.0)


def test_observed_uniform_node_ipw_mean_loss_allows_empty_sample() -> None:
    torch = _torch_or_skip()
    objective = observed_uniform_node_ipw_mean_loss(
        torch.tensor([[1.0, 9.0, 16.0]]),
        observed=torch.tensor([[False, False, False]]),
        propensity=0.1,
    )
    assert float(objective) == pytest.approx(0.0)


def test_persistent_uniform_node_mask_has_no_doc_minimum() -> None:
    torch = _torch_or_skip()
    np = pytest.importorskip("numpy")
    sampled = pytest.importorskip("treepo._research.unified_g_v1.sketch.sampled_supervision")
    tree_task = pytest.importorskip("treepo._research.unified_g_v1.training.tree_task")

    example = tree_task.TreeExample(
        leaves=[(1,)],
        target=0.0,
        extra={
            sampled.PERSISTENT_UNIFORM_NODE_SCORES_KEY: np.asarray(
                [0.9, 0.95],
                dtype=np.float32,
            ),
        },
    )
    mask = sampled.persistent_uniform_node_mask(
        [example],
        width=2,
        rate=0.1,
        device=torch.device("cpu"),
    )
    assert mask.tolist() == [[False, False]]
