from __future__ import annotations

import pytest

from treepo.training.local_law import (
    LocalLawTrainingRow,
    aggregate_local_law_training_rows,
    corrected_local_law_loss,
    corrected_local_law_loss_tensor,
    local_law_objective_from_losses,
    local_law_objective_target_mse,
    local_law_training_objective_mean,
)
from treepo.local_law import LocalLawAuditRow


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


def test_training_rows_allow_unobserved_zero_propensity() -> None:
    row = LocalLawTrainingRow(proxy_loss=0.4, observed=False, propensity=0.0)

    assert row.corrected_loss() == pytest.approx(0.4)
    assert row.propensity == pytest.approx(0.0)


def test_audit_rows_reject_zero_propensity() -> None:
    with pytest.raises(ValueError, match="propensity"):
        LocalLawAuditRow(
            row_id="audit-0",
            law_kind="c1",
            proxy_loss=0.4,
            observed=False,
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


def test_scalar_training_rows_match_tensor_objective() -> None:
    torch = _torch_or_skip()
    rows = [
        LocalLawTrainingRow(proxy_loss=0.25, observed=False, propensity=0.0, depth=0),
        LocalLawTrainingRow(
            proxy_loss=1.0,
            oracle_loss=0.25,
            observed=True,
            propensity=0.5,
            depth=1,
            node_weight=2.0,
        ),
    ]
    scalar = local_law_training_objective_mean(rows, gamma_depth=0.5)
    tensor = local_law_objective_from_losses(
        proxy_loss=torch.tensor([0.25, 1.0]),
        oracle_loss=torch.tensor([0.0, 0.25]),
        observed=torch.tensor([False, True]),
        propensity=torch.tensor([0.0, 0.5]),
        depths=torch.tensor([0, 1]),
        node_weights=torch.tensor([1.0, 2.0]),
        gamma_depth=0.5,
    )

    assert scalar == pytest.approx(float(tensor))


def test_scalar_training_objective_rejects_gamma_above_one() -> None:
    with pytest.raises(ValueError, match="gamma_depth"):
        local_law_training_objective_mean(
            [LocalLawTrainingRow(proxy_loss=0.0, observed=False, propensity=0.0)],
            gamma_depth=1.5,
        )


def test_training_aggregate_rejects_invalid_local_law_weight() -> None:
    with pytest.raises(ValueError, match="local_law_weight"):
        aggregate_local_law_training_rows(
            [LocalLawTrainingRow(proxy_loss=0.0, observed=False, propensity=0.0)],
            local_law_weight=1.5,
        )
