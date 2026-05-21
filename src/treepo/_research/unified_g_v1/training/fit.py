"""`fit()` — one call, one config, one dispatch.

Everything flows through `TrainerConfig` (from `training.tree_task`). The
trainer is a plain callable `(cfg, output_dir, dataset) -> FitResult` and is
picked by `resolve_trainer(cfg)` unless the user sets `cfg.trainer` directly.

There are no specialized config subclasses. If your paradigm needs fields
the base config does not have, either (a) stash them in `cfg.extra`, or
(b) write a custom trainer callable that closes over them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from treepo._research.unified_g_v1.training.backends.pytorch_loop import (
    EvaluateFn,
    SupervisionAdapter,
)
from treepo._research.unified_g_v1.training.prepared_dataset import PreparedDataset
from treepo._research.unified_g_v1.training.tree_task import TrainerConfig, TreeTaskConfig, run_tree_task


# ---------------------------------------------------------------------------
# FitResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitResult:
    """Uniform training result.

    The top-level fields (`status`, `metrics`, `artifacts`, `history`,
    `backend`) are a shared schema every trainer populates. `summary` is the
    kitchen sink — a trainer-specific dict with raw backend output kept for
    backward compatibility and debugging.
    """

    backend: str
    summary: Mapping[str, Any] = field(default_factory=dict)
    status: str = "completed"  # "completed" | "failed" | "planned"
    metrics: Mapping[str, float] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    history: Sequence[Mapping[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PyTorch runtime helper — a small record some tests / custom trainers use.
# Not a config; just a convenient return type for a "build me a runtime" fn.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PyTorchRuntime:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    train_items: Sequence[Any]
    val_items: Sequence[Any]
    supervision_adapter: SupervisionAdapter
    evaluate_fn: EvaluateFn
    checkpoint_extra: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# fit() — single entry point.
# ---------------------------------------------------------------------------


def fit(
    *,
    trainer_config: TrainerConfig,
    output_dir: str | Path,
    dataset: PreparedDataset | None = None,
) -> FitResult:
    """Resolve the trainer for `trainer_config` and run it.

    - `trainer_config` is always a `TrainerConfig` (fill in only the fields
      your trainer needs).
    - `dataset` is optional and threaded through to the trainer.
    """
    if not isinstance(trainer_config, TrainerConfig):
        raise TypeError(
            f"trainer_config must be a TrainerConfig, got {type(trainer_config).__name__}"
        )
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_tree_task(trainer_config, output_dir=output_dir, dataset=dataset)


__all__ = [
    "FitResult",
    "PyTorchRuntime",
    "TrainerConfig",
    "TreeTaskConfig",
    "fit",
]
