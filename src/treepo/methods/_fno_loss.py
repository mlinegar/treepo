"""Coordinate losses shared by the neural-operator training paths.

The public target metric is one point distance per example.  ``sum_l1`` keeps
that distance invariant across the DSPy and neural-operator families:
``sum_j |prediction_j - target_j|``.  The historical coordinate-mean MSE
remains available only as an explicit compatibility mode.
"""

from __future__ import annotations

from typing import Any

SUM_L1 = "sum_l1"
COORDINATE_MEAN_MSE = "coordinate_mean_mse"
TRAINING_LOSSES = frozenset({SUM_L1, COORDINATE_MEAN_MSE})


def normalize_training_loss(value: Any) -> str:
    """Return one validated neural-operator coordinate-loss identifier."""

    normalized = str(value or "").strip().lower()
    if normalized not in TRAINING_LOSSES:
        choices = ", ".join(sorted(TRAINING_LOSSES))
        raise ValueError(f"training_loss must be one of {{{choices}}}, got {value!r}")
    return normalized


def per_row_training_loss(prediction: Any, target: Any, *, training_loss: str) -> Any:
    """Reduce coordinate residuals to one loss per leading row."""

    mode = normalize_training_loss(training_loss)
    residual = prediction - target
    if mode == SUM_L1:
        return residual.abs().sum(dim=-1)
    return residual.square().mean(dim=-1)


def training_loss_definition(training_loss: str) -> str:
    """Return a serialization-friendly mathematical definition."""

    mode = normalize_training_loss(training_loss)
    if mode == SUM_L1:
        return "sum_absolute_coordinate_error"
    return "coordinate_mean_squared_error"


__all__ = [
    "COORDINATE_MEAN_MSE",
    "SUM_L1",
    "TRAINING_LOSSES",
    "normalize_training_loss",
    "per_row_training_loss",
    "training_loss_definition",
]
