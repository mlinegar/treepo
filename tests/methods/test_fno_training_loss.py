"""FNO/neural-operator target losses follow the joint sum-L1 contract."""

from __future__ import annotations

import pytest

from treepo.methods._fno_config import _target_schema_payload
from treepo.methods._fno_transition import _numeric_transition_law_rows
from treepo.methods.fno import NeuralOperatorFamily, NeuralOperatorFamilyConfig


def _family(*, training_loss: str = "sum_l1") -> NeuralOperatorFamily:
    return NeuralOperatorFamily(
        NeuralOperatorFamilyConfig(
            operator_kind="conv1d",
            embedding_dim=4,
            hidden_channels=2,
            training_loss=training_loss,
        )
    )


def test_training_loss_defaults_to_sum_l1_and_rejects_unknown_values() -> None:
    assert NeuralOperatorFamilyConfig().training_loss == "sum_l1"
    assert (
        NeuralOperatorFamilyConfig(training_loss="coordinate_mean_mse").training_loss
        == "coordinate_mean_mse"
    )
    with pytest.raises(ValueError, match="training_loss must be one of"):
        NeuralOperatorFamilyConfig(training_loss="mse")


def test_k1_root_sum_l1_reduces_exactly_to_absolute_error() -> None:
    torch = pytest.importorskip("torch")
    family = _family()

    loss, count = family._masked_root_loss(
        torch.tensor([[3.0]]),
        torch.tensor([[1.0]]),
        torch.tensor([True]),
    )

    assert count == 1
    assert float(loss) == pytest.approx(abs(3.0 - 1.0))


def test_k3_root_sum_l1_is_a_coordinate_sum_not_a_mean() -> None:
    torch = pytest.importorskip("torch")
    family = _family()

    loss, count = family._masked_root_loss(
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.zeros((1, 3)),
        torch.tensor([True]),
    )

    assert count == 1
    assert float(loss) == pytest.approx(6.0)


def test_root_loss_keeps_explicit_legacy_coordinate_mean_mse() -> None:
    torch = pytest.importorskip("torch")
    family = _family(training_loss="coordinate_mean_mse")

    loss, count = family._masked_root_loss(
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.zeros((1, 3)),
        torch.tensor([True]),
    )

    assert count == 1
    assert float(loss) == pytest.approx(14.0 / 3.0)


def test_k3_node_f_supervision_uses_the_same_sum_l1_distance() -> None:
    torch = pytest.importorskip("torch")
    family = _family()

    class _IdentityReadout:
        @staticmethod
        def _read(states):
            return states

    family._model = _IdentityReadout()
    trace = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.1, 0.1, 0.1],
        ]
    )
    target = torch.zeros((3, 3))
    rows = family._node_supervision_rows(
        [trace],
        [(target, torch.tensor([True, True, True]))],
        dtype=trace.dtype,
    )

    assert rows is not None
    losses, _depths, _is_leaf = rows
    assert losses.tolist() == pytest.approx([0.6, 1.5, 0.3])


def test_numeric_state_rows_use_sum_l1_and_keep_explicit_legacy_mse() -> None:
    torch = pytest.importorskip("torch")
    prediction = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.zeros((1, 3))

    l1_rows = _numeric_transition_law_rows(
        [prediction],
        [target],
        training_loss="sum_l1",
        torch=torch,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    mse_rows = _numeric_transition_law_rows(
        [prediction],
        [target],
        training_loss="coordinate_mean_mse",
        torch=torch,
        device=prediction.device,
        dtype=prediction.dtype,
    )

    assert l1_rows is not None
    assert mse_rows is not None
    assert float(l1_rows[0][0]) == pytest.approx(6.0)
    assert float(mse_rows[0][0]) == pytest.approx(14.0 / 3.0)


def test_named_vector_schema_records_executed_training_loss() -> None:
    config = NeuralOperatorFamilyConfig(
        target_names=("left", "neutral", "right"),
        target_oracle_ids=("oracle:left", "oracle:neutral", "oracle:right"),
    )

    schema = _target_schema_payload(config)

    assert schema is not None
    assert schema["training_loss"] == "sum_l1"
    assert schema["training_loss_definition"] == "sum_absolute_coordinate_error"
    assert schema["coordinate_reduction"] == "sum_not_mean"


def test_fno_artifact_records_executed_loss_provenance() -> None:
    family = NeuralOperatorFamily(
        NeuralOperatorFamilyConfig(
            operator_kind="conv1d",
            embedding_dim=4,
            hidden_channels=2,
            target_names=("left", "neutral", "right"),
            target_oracle_ids=("oracle:left", "oracle:neutral", "oracle:right"),
        )
    )
    family._ensure_model(output_dim=3)

    artifact = family._artifact_payload(
        kind="f",
        iteration=1,
        n_train=0,
        loss=None,
    )

    assert artifact["training_loss"] == "sum_l1"
    assert artifact["training_loss_definition"] == "sum_absolute_coordinate_error"
    assert artifact["target_schema"]["training_loss"] == "sum_l1"
