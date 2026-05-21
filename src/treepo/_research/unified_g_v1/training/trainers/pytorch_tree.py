"""PyTorch generic tree trainer.

The body of what `run_tree_task` used to do, lifted to a plain function so it
can be swapped, replaced, or called directly. Default trainer for any
`TreeTaskConfig` whose oracle declares a non-text `space_kind`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from treepo._research.unified_g_v1.training.backends.pytorch_loop import (
    PyTorchLoopConfig,
    run_pytorch_training,
)
from treepo._research.unified_g_v1.training.trainers import register_trainer


def _default_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )


class _TreeTaskSupervisionAdapter:
    def __init__(self, *, model: nn.Module, objective) -> None:
        self.model = model
        # Wrap plain callables as a TreeObjective automatically.
        from treepo._research.unified_g_v1.training.objectives import as_objective

        self.objective = as_objective(objective)
        # Detect at init time whether the objective accepts `forward_aux`.
        # User-written objectives that pre-date the aux channel expose the
        # older `compute_loss(*, root_state, prediction, batch)` signature
        # and should not be passed the new kwarg. This keeps the adapter
        # back-compatible while still supporting rich model returns.
        import inspect

        try:
            sig = inspect.signature(self.objective.compute_loss)
            self._objective_accepts_forward_aux = (
                "forward_aux" in sig.parameters
                or any(
                    p.kind is inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
            )
        except (TypeError, ValueError):
            self._objective_accepts_forward_aux = False

    def prepare_batch(self, batch):
        return batch

    def compute_supervision_loss(self, prepared):
        result = self.model.forward_tree(prepared)
        # Models may return either (root_state, prediction) or
        # (root_state, prediction, forward_aux). `forward_aux` is a dict the
        # model fills with per-forward intermediate tensors (e.g. Markov's
        # per-node C1/C2/C3 losses) that the objective wants access to but
        # the generic loop doesn't need to know about.
        if isinstance(result, tuple) and len(result) == 3:
            root_state, prediction, forward_aux = result
        else:
            root_state, prediction = result
            forward_aux = None
        # Only forward `forward_aux` when both the model produced one AND
        # the objective declares it in its signature. This lets newer
        # objectives enforce per-node local laws while older objectives keep
        # working with the (root_state, prediction, batch)-only contract.
        if forward_aux is None or not self._objective_accepts_forward_aux:
            return self.objective.compute_loss(
                root_state=root_state,
                prediction=prediction,
                batch=prepared,
            )
        return self.objective.compute_loss(
            root_state=root_state,
            prediction=prediction,
            batch=prepared,
            forward_aux=forward_aux,
        )


def pytorch_tree_trainer(cfg, output_dir: Path, dataset=None):
    """Run a `TreeTaskConfig` through the generic PyTorch tree-training loop.

    `dataset` is unused: the oracle produces its own examples. The argument
    exists to match the `Trainer` signature.
    """
    del dataset
    from treepo._research.unified_g_v1.training.fit import FitResult  # avoid cycle at import time

    if cfg.model is None or cfg.objective is None:
        raise ValueError(
            "pytorch_tree_trainer requires `model` and `objective` on the config"
        )

    torch.manual_seed(int(cfg.seed))
    model = cfg.model
    device = torch.device("cpu")
    if bool(getattr(cfg, "use_cuda", False)) and torch.cuda.is_available():
        cuda_device = getattr(cfg, "cuda_device", None)
        device = torch.device(f"cuda:{int(cuda_device)}" if cuda_device is not None else "cuda")
    model.to(device)
    optimizer_builder = cfg.optimizer_builder or _default_optimizer
    optimizer = optimizer_builder(model, cfg)
    adapter = _TreeTaskSupervisionAdapter(model=model, objective=cfg.objective)
    cfg_extra = dict(getattr(cfg, "extra", {}) or {})
    batch_key = str(cfg_extra.get("batch_key") or "") or None

    def _evaluate(*, model: nn.Module, items: Sequence[Any], batch_size: int):
        return adapter.objective.evaluate(model=model, items=items, batch_size=batch_size)

    loop_result = run_pytorch_training(
        model=model,
        optimizer=optimizer,
        train_items=list(cfg.oracle.train_examples()),
        val_items=list(cfg.oracle.val_examples()),
        supervision_adapter=adapter,
        evaluate_fn=_evaluate,
        config=PyTorchLoopConfig(
            n_epochs=int(cfg.n_epochs),
            train_batch_size=int(cfg.train_batch_size),
            grad_clip_norm=float(cfg.grad_clip_norm),
            seed=int(cfg.seed),
            save_every_epoch=bool(cfg.save_every_epoch),
            best_metric_key=str(cfg.best_metric_key),
            batch_key=batch_key,
            eval_every_n_epochs=int(cfg_extra.get("eval_every_n_epochs", 1)),
            evaluate_train_on_eval=bool(cfg_extra.get("evaluate_train_on_eval", True)),
        ),
        output_dir=Path(output_dir),
        checkpoint_extra={"oracle_metadata": dict(cfg.oracle.metadata())},
    )
    summary = {
        "backend": "tree_task",
        "oracle_metadata": dict(cfg.oracle.metadata()),
        "history": list(loop_result["history"]),
        "best_epoch": int(loop_result["best_epoch"]),
        "best_metric_key": str(loop_result["best_metric_key"]),
        "best_metric_value": float(loop_result["best_metric_value"]),
        "best_checkpoint_path": str(loop_result["best_checkpoint_path"]),
    }
    # Surface law-stress gain_frac metrics when the objective populated them.
    # The generic loop passes through scalar eval keys into each epoch's
    # history entry; we read the last epoch for the final reported gain.
    metrics: dict[str, float] = {
        "best_metric_value": float(loop_result["best_metric_value"]),
        "best_epoch": float(loop_result["best_epoch"]),
    }
    history_list = list(loop_result["history"])
    if history_list:
        last = history_list[-1]
        for key in (
            "val_mae_raw",
            "val_mae_normalized",
            "val_merge_recon_mse",
            "baseline_val_mae",
            "val_mae_gain_frac",
            "val_mae_pass",
        ):
            if key in last:
                try:
                    metrics[key] = float(last[key])
                except (TypeError, ValueError):
                    continue
    return FitResult(
        backend="tree_task",
        summary=summary,
        status="completed",
        metrics=metrics,
        artifacts={"best_checkpoint_path": summary["best_checkpoint_path"]},
        history=history_list,
    )


register_trainer("pytorch_tree", pytorch_tree_trainer)
