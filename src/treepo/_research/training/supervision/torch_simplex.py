"""Lightweight torch regression for dense simplex-valued supervision."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for dense simplex supervision training. "
        "Install with: pip install torch>=2.0.0"
    ) from e

from treepo._research.training.supervision.numeric_rows import (
    DenseVectorSupervision,
    dense_vector_rows,
    dense_vector_rows_to_numpy,
)
from treepo._research.training.config_sections import (
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
    TrainConfig,
)
DenseSimplexSupervision = DenseVectorSupervision


@dataclass(frozen=True, kw_only=True)
class DenseSimplexModelConfig:
    hidden_dims: Tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True, kw_only=True)
class DenseSimplexTrainingConfig:
    model: DenseSimplexModelConfig = field(default_factory=DenseSimplexModelConfig)
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(batch_size=32, epochs=10)
    )
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(learning_rate=1e-3)
    )
    runtime: RuntimeConfig = field(
        default_factory=lambda: RuntimeConfig(
            device="cpu",
            bf16=False,
            gradient_checkpointing=False,
        )
    )
    run: RunConfig = field(default_factory=lambda: RunConfig(seed=0))


@dataclass(frozen=True)
class DenseSimplexTrainingResult:
    train_loss_final: float
    train_loss_curve: Tuple[float, ...]
    epochs_completed: int
    selection_metric_curve: Tuple[float, ...]
    selection_mode: str
    selection_split: str
    selection_metric_name: str
    selection_metric_value: float
    best_epoch: int
    input_dim: int
    output_dim: int
    n_train_rows: int
    n_val_rows: int


class DenseSimplexRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        dims = [int(input_dim), *(int(dim) for dim in hidden_dims if int(dim) > 0), int(output_dim)]
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward_logits(features), dim=-1)


def _weighted_epoch_loss(
    model: DenseSimplexRegressor,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    grad_clip_norm: float,
    rng: Optional[random.Random] = None,
) -> float:
    indices = list(range(len(x)))
    if rng is not None:
        rng.shuffle(indices)
    losses: List[float] = []
    for start in range(0, len(indices), int(max(1, batch_size))):
        batch_idx = indices[start : start + int(max(1, batch_size))]
        xb = torch.tensor(x[batch_idx], dtype=torch.float32, device=device)
        yb = torch.tensor(y[batch_idx], dtype=torch.float32, device=device)
        wb = torch.tensor(w[batch_idx], dtype=torch.float32, device=device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        pred = torch.clamp(model(xb), min=1e-8)
        loss_vec = -torch.sum(yb * torch.log(pred), dim=1)
        weight_sum = torch.clamp(wb.sum(), min=1e-12)
        loss = torch.sum(loss_vec * wb) / weight_sum
        if optimizer is not None and bool(getattr(loss, "requires_grad", False)):
            loss.backward()
            if float(grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(np.asarray(losses, dtype=np.float64))) if losses else 0.0


def fit_dense_simplex_regressor(
    supervision: DenseSimplexSupervision,
    *,
    val_supervision: Optional[DenseSimplexSupervision] = None,
    config: Optional[DenseSimplexTrainingConfig] = None,
) -> tuple[DenseSimplexRegressor, DenseSimplexTrainingResult]:
    cfg = config or DenseSimplexTrainingConfig()
    train_rows = dense_vector_rows(supervision)
    if not train_rows:
        raise ValueError("no dense simplex supervision rows available for training")
    val_rows = dense_vector_rows(val_supervision) if val_supervision else []

    x_train, y_train, w_train = dense_vector_rows_to_numpy(
        train_rows,
        normalize_targets_to_simplex=True,
    )
    if val_rows:
        x_val, y_val, w_val = dense_vector_rows_to_numpy(
            val_rows,
            normalize_targets_to_simplex=True,
        )
        if x_val.shape[1] != x_train.shape[1] or y_val.shape[1] != y_train.shape[1]:
            raise ValueError("train/val dense simplex dimensions must match")
    else:
        x_val = y_val = w_val = None

    random.seed(int(cfg.run.seed))
    np.random.seed(int(cfg.run.seed))
    torch.manual_seed(int(cfg.run.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.run.seed))

    device = torch.device(str(cfg.runtime.device))
    model = DenseSimplexRegressor(
        int(x_train.shape[1]),
        int(y_train.shape[1]),
        hidden_dims=tuple(cfg.model.hidden_dims),
    ).to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optimizer.learning_rate),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    rng = random.Random(int(cfg.run.seed))
    train_curve: List[float] = []
    selection_curve: List[float] = []
    best_epoch = 0
    best_value = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    selection_mode = "best_val_dense_simplex_loss" if val_rows else "final_epoch_no_validation"
    selection_split = "val" if val_rows else "config"
    selection_metric_name = "val_dense_simplex_loss" if val_rows else "train_loss_final"

    for epoch in range(int(max(1, cfg.train.epochs))):
        model.train()
        train_loss = _weighted_epoch_loss(
            model,
            x_train,
            y_train,
            w_train,
            batch_size=int(cfg.train.batch_size),
            device=device,
            optimizer=optimizer,
            grad_clip_norm=float(cfg.optimizer.grad_clip_norm),
            rng=rng,
        )
        train_curve.append(float(train_loss))
        if val_rows:
            model.eval()
            with torch.no_grad():
                selection_value = _weighted_epoch_loss(
                    model,
                    x_val,
                    y_val,
                    w_val,
                    batch_size=int(cfg.train.batch_size),
                    device=device,
                    optimizer=None,
                    grad_clip_norm=0.0,
                    rng=None,
                )
            selection_curve.append(float(selection_value))
            if not math.isfinite(best_value) or float(selection_value) < float(best_value):
                best_value = float(selection_value)
                best_epoch = int(epoch)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        else:
            selection_curve.append(float(train_loss))

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_value = float(train_curve[-1])
        best_epoch = max(0, len(train_curve) - 1)

    result = DenseSimplexTrainingResult(
        train_loss_final=float(train_curve[-1]),
        train_loss_curve=tuple(float(value) for value in train_curve),
        epochs_completed=int(len(train_curve)),
        selection_metric_curve=tuple(float(value) for value in selection_curve),
        selection_mode=str(selection_mode),
        selection_split=str(selection_split),
        selection_metric_name=str(selection_metric_name),
        selection_metric_value=float(best_value),
        best_epoch=int(best_epoch),
        input_dim=int(x_train.shape[1]),
        output_dim=int(y_train.shape[1]),
        n_train_rows=int(len(train_rows)),
        n_val_rows=int(len(val_rows)),
    )
    return model, result


@torch.no_grad()
def predict_dense_simplex_regressor(
    model: DenseSimplexRegressor,
    *,
    supervision: Optional[DenseSimplexSupervision] = None,
    rows: Optional[Sequence[Dict[str, Any]]] = None,
    device: str = "cpu",
) -> np.ndarray:
    if rows is None:
        if supervision is None:
            raise ValueError("provide supervision or rows for dense simplex prediction")
        rows = dense_vector_rows(supervision)
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    x, _y, _w = dense_vector_rows_to_numpy(
        list(rows),
        normalize_targets_to_simplex=True,
    )
    dev = torch.device(str(device))
    model = model.to(device=dev)
    model.eval()
    xb = torch.tensor(x, dtype=torch.float32, device=dev)
    pred = model(xb).detach().cpu().numpy()
    return np.asarray(pred, dtype=np.float64)


__all__ = [
    "DenseSimplexModelConfig",
    "DenseSimplexRegressor",
    "DenseSimplexTrainingConfig",
    "DenseSimplexTrainingResult",
    "fit_dense_simplex_regressor",
    "predict_dense_simplex_regressor",
]
