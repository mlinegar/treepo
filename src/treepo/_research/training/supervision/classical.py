"""Shared classical learners over the canonical dense supervision surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from treepo._research.training.config_sections import RunConfig
from treepo._research.training.supervision.numeric_rows import (
    DenseScalarSupervision,
    DenseVectorSupervision,
    dense_scalar_rows,
    dense_scalar_rows_to_numpy,
    dense_vector_rows,
    dense_vector_rows_to_numpy,
)


def _normalize_simplex_rows(x: np.ndarray) -> np.ndarray:
    rows = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    sums = np.sum(rows, axis=1, keepdims=True)
    sums = np.maximum(sums, 1e-12)
    return np.asarray(rows / sums, dtype=np.float64)


@dataclass(frozen=True, kw_only=True)
class DenseSimplexForestModelConfig:
    n_estimators: int = 200
    max_depth: Optional[int] = 16
    min_samples_leaf: int = 5


@dataclass(frozen=True, kw_only=True)
class DenseSimplexForestTrainingConfig:
    model: DenseSimplexForestModelConfig = field(
        default_factory=DenseSimplexForestModelConfig
    )
    run: RunConfig = field(default_factory=lambda: RunConfig(seed=0))


@dataclass(frozen=True)
class DenseSimplexForestTrainingResult:
    input_dim: int
    output_dim: int
    n_train_rows: int
    selection_mode: str
    selection_split: str
    selection_metric_name: str
    selection_metric_value: float
    best_epoch: int


class DenseSimplexForestRegressor:
    """Random-forest simplex regressor with canonical normalization on predict."""

    def __init__(self, estimator: object) -> None:
        self.estimator = estimator

    def predict(self, features: np.ndarray) -> np.ndarray:
        pred = np.asarray(
            getattr(self.estimator, "predict")(np.asarray(features, dtype=np.float32)),
            dtype=np.float64,
        )
        return _normalize_simplex_rows(pred)


@dataclass(frozen=True, kw_only=True)
class DenseScalarRidgeModelConfig:
    ridge_alpha: float = 1.0


@dataclass(frozen=True, kw_only=True)
class DenseScalarRidgeTrainingConfig:
    model: DenseScalarRidgeModelConfig = field(
        default_factory=DenseScalarRidgeModelConfig
    )


@dataclass(frozen=True)
class DenseScalarRidgeTrainingResult:
    input_dim: int
    n_train_rows: int
    selection_mode: str
    selection_split: str
    selection_metric_name: str
    selection_metric_value: float
    best_epoch: int


@dataclass(frozen=True)
class DenseScalarRidgeRegressor:
    weights: np.ndarray
    bias: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("dense scalar ridge regressor expects a 2D feature matrix")
        return np.asarray(
            x @ np.asarray(self.weights, dtype=np.float64) + float(self.bias),
            dtype=np.float64,
        )


def fit_dense_scalar_ridge_regressor(
    supervision: DenseScalarSupervision,
    *,
    config: Optional[DenseScalarRidgeTrainingConfig] = None,
    sample_weights: Optional[np.ndarray] = None,
) -> tuple[DenseScalarRidgeRegressor, DenseScalarRidgeTrainingResult]:
    """Fit a weighted closed-form ridge regressor over dense scalar supervision."""

    cfg = config or DenseScalarRidgeTrainingConfig()
    rows = dense_scalar_rows(supervision)
    if not rows:
        raise ValueError("no dense scalar supervision rows available for ridge training")
    x_train, y_train, w_train = dense_scalar_rows_to_numpy(rows)
    weights = (
        np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if sample_weights is not None
        else np.asarray(w_train, dtype=np.float64).reshape(-1)
    )
    if weights.size != x_train.shape[0]:
        raise ValueError("sample_weights must match the number of dense scalar rows")
    sqrt_w = np.sqrt(np.clip(weights, 0.0, None))
    x1 = np.concatenate(
        [np.asarray(x_train, dtype=np.float64), np.ones((int(x_train.shape[0]), 1), dtype=np.float64)],
        axis=1,
    )
    x1w = x1 * sqrt_w[:, np.newaxis]
    yw = np.asarray(y_train, dtype=np.float64) * sqrt_w
    gram = x1w.T @ x1w
    reg = float(max(0.0, cfg.model.ridge_alpha)) * np.eye(int(x1.shape[1]), dtype=np.float64)
    reg[-1, -1] = 0.0
    rhs = x1w.T @ yw
    try:
        beta = np.linalg.solve(gram + reg, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(gram + reg, rhs, rcond=None)[0]
    model = DenseScalarRidgeRegressor(
        weights=np.asarray(beta[:-1], dtype=np.float64),
        bias=float(beta[-1]),
    )
    return model, DenseScalarRidgeTrainingResult(
        input_dim=int(x_train.shape[1]),
        n_train_rows=int(len(rows)),
        selection_mode="closed_form_ridge",
        selection_split="config",
        selection_metric_name="closed_form_fit_no_validation",
        selection_metric_value=float("nan"),
        best_epoch=0,
    )


def predict_dense_scalar_ridge_regressor(
    model: DenseScalarRidgeRegressor,
    features: np.ndarray,
) -> np.ndarray:
    """Predict scalar targets from dense feature rows."""

    return model.predict(np.asarray(features, dtype=np.float64))


def fit_dense_simplex_forest_regressor(
    supervision: DenseVectorSupervision,
    *,
    config: Optional[DenseSimplexForestTrainingConfig] = None,
) -> tuple[DenseSimplexForestRegressor, DenseSimplexForestTrainingResult]:
    """Fit a weighted random-forest regressor over dense simplex supervision."""

    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for dense simplex random-forest training. "
            "Install with: pip install scikit-learn>=1.4.2"
        ) from e

    cfg = config or DenseSimplexForestTrainingConfig()
    rows = dense_vector_rows(supervision)
    if not rows:
        raise ValueError("no dense vector supervision rows available for random-forest training")
    x_train, y_train, w_train = dense_vector_rows_to_numpy(
        rows,
        normalize_targets_to_simplex=True,
    )
    model = RandomForestRegressor(
        n_estimators=int(cfg.model.n_estimators),
        max_depth=(None if cfg.model.max_depth is None else int(cfg.model.max_depth)),
        min_samples_leaf=int(cfg.model.min_samples_leaf),
        random_state=int(cfg.run.seed),
        n_jobs=1,
    )
    model.fit(
        np.asarray(x_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),
        sample_weight=np.asarray(w_train, dtype=np.float32),
    )
    return DenseSimplexForestRegressor(model), DenseSimplexForestTrainingResult(
        input_dim=int(x_train.shape[1]),
        output_dim=int(y_train.shape[1]),
        n_train_rows=int(len(rows)),
        selection_mode="rf_fit_no_validation",
        selection_split="config",
        selection_metric_name="rf_training_objective_untracked",
        selection_metric_value=float("nan"),
        best_epoch=0,
    )


def predict_dense_simplex_forest_regressor(
    model: DenseSimplexForestRegressor,
    features: np.ndarray,
) -> np.ndarray:
    """Predict simplex-valued targets from dense feature rows."""

    return model.predict(np.asarray(features, dtype=np.float64))


@dataclass(frozen=True)
class AffineSimplexCalibrationConfig:
    ridge: float = 1e-4
    use_sample_weights: bool = True


@dataclass(frozen=True)
class AffineSimplexCalibrationResult:
    input_dim: int
    output_dim: int
    n_train_rows: int
    selection_mode: str
    selection_split: str
    selection_metric_name: str
    selection_metric_value: float
    best_epoch: int
    uses_sample_weights: bool


@dataclass(frozen=True)
class DenseAffineSimplexCalibrator:
    weights: np.ndarray
    bias: np.ndarray

    @classmethod
    def identity(cls, dim: int) -> "DenseAffineSimplexCalibrator":
        return cls(
            weights=np.eye(int(dim), dtype=np.float64),
            bias=np.zeros((int(dim),), dtype=np.float64),
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("affine simplex calibrator expects a 2D feature matrix")
        mapped = x @ np.asarray(self.weights, dtype=np.float64) + np.asarray(
            self.bias,
            dtype=np.float64,
        )
        return _normalize_simplex_rows(mapped)


def fit_dense_affine_simplex_calibrator(
    supervision: DenseVectorSupervision,
    *,
    config: Optional[AffineSimplexCalibrationConfig] = None,
) -> tuple[DenseAffineSimplexCalibrator, AffineSimplexCalibrationResult]:
    """Fit a weighted affine simplex calibrator from dense supervision rows."""

    cfg = config or AffineSimplexCalibrationConfig()
    rows = dense_vector_rows(supervision)
    if not rows:
        raise ValueError("no dense vector supervision rows available for affine calibration")
    x_train, y_train, w_train = dense_vector_rows_to_numpy(
        rows,
        normalize_targets_to_simplex=True,
    )
    n_rows = int(x_train.shape[0])
    input_dim = int(x_train.shape[1])
    output_dim = int(y_train.shape[1])
    x1 = np.concatenate([x_train.astype(np.float64), np.ones((n_rows, 1), dtype=np.float64)], axis=1)
    if bool(cfg.use_sample_weights):
        sqrt_w = np.sqrt(np.maximum(np.asarray(w_train, dtype=np.float64), 1e-12))
        x1w = x1 * sqrt_w[:, np.newaxis]
        yw = np.asarray(y_train, dtype=np.float64) * sqrt_w[:, np.newaxis]
    else:
        x1w = x1
        yw = np.asarray(y_train, dtype=np.float64)
    gram = x1w.T @ x1w
    ridge = float(max(0.0, cfg.ridge))
    if ridge > 0.0:
        reg = ridge * np.eye(input_dim + 1, dtype=np.float64)
        reg[-1, -1] = 0.0
        gram = gram + reg
    rhs = x1w.T @ yw
    coef, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
    calibrator = DenseAffineSimplexCalibrator(
        weights=np.asarray(coef[:input_dim, :], dtype=np.float64),
        bias=np.asarray(coef[input_dim, :], dtype=np.float64),
    )
    return calibrator, AffineSimplexCalibrationResult(
        input_dim=input_dim,
        output_dim=output_dim,
        n_train_rows=n_rows,
        selection_mode="closed_form_affine_ridge",
        selection_split="config",
        selection_metric_name="closed_form_fit_no_validation",
        selection_metric_value=float("nan"),
        best_epoch=0,
        uses_sample_weights=bool(cfg.use_sample_weights),
    )


def apply_dense_affine_simplex_calibrator(
    calibrator: DenseAffineSimplexCalibrator,
    features: np.ndarray,
) -> np.ndarray:
    """Apply a fitted affine simplex calibrator to dense features."""

    return calibrator.predict(np.asarray(features, dtype=np.float64))


__all__ = [
    "AffineSimplexCalibrationConfig",
    "AffineSimplexCalibrationResult",
    "DenseAffineSimplexCalibrator",
    "DenseScalarRidgeModelConfig",
    "DenseScalarRidgeRegressor",
    "DenseScalarRidgeTrainingConfig",
    "DenseScalarRidgeTrainingResult",
    "DenseSimplexForestModelConfig",
    "DenseSimplexForestRegressor",
    "DenseSimplexForestTrainingConfig",
    "DenseSimplexForestTrainingResult",
    "apply_dense_affine_simplex_calibrator",
    "fit_dense_scalar_ridge_regressor",
    "fit_dense_affine_simplex_calibrator",
    "fit_dense_simplex_forest_regressor",
    "predict_dense_scalar_ridge_regressor",
    "predict_dense_simplex_forest_regressor",
]
