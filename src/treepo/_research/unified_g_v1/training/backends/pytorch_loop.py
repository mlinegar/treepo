from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
import torch.nn as nn


class SupervisionAdapter(Protocol):
    """Minimal surface the PyTorch loop needs to compute a per-batch loss.

    TreeModelV2Trainer already satisfies this: its `prepare_batch` and
    `compute_supervision_loss` methods have the shape below.
    """

    def prepare_batch(self, batch: Sequence[Any]) -> Any: ...

    def compute_supervision_loss(
        self, prepared: Any
    ) -> tuple[torch.Tensor, int, Mapping[str, Any]]: ...


EvaluateFn = Callable[..., Mapping[str, Any]]
"""Called as evaluate_fn(model=model, items=items, batch_size=batch_size).

Must return a mapping with keys 'mae_raw', 'mae_normalized', 'predictions',
'count' so the loop's best-checkpoint logic and the FitResult summary can
read them uniformly.
"""


@dataclass(frozen=True)
class PyTorchLoopConfig:
    n_epochs: int
    train_batch_size: int
    grad_clip_norm: float
    seed: int
    save_every_epoch: bool = False
    best_metric_key: str = "mae_raw"
    # Periodic training-state snapshot: saves every N epochs and at end.
    # Includes model weights, optimizer state, rng state, epoch counter, history.
    # 0 disables periodic snapshots (best-only + final remain).
    checkpoint_every_n_epochs: int = 10
    # If set, path to a train_state.pt from a prior run; training resumes at
    # saved epoch+1. `output_dir/train_state.pt` is also checked automatically.
    resume_from: Path | None = None
    # Optional homogeneous-batch constraint. Leaf-size sketch runs have
    # variable observed leaf counts, but the tensorized local-law losses need a
    # rectangular [batch, leaves, ...] shape. Bucketing keeps large minibatches
    # without falling back to one document per step.
    batch_key: str | None = None
    # Full train/validation evaluation can dominate tiny synthetic models.
    # Evaluate every N epochs, always including the final epoch. Default 1
    # preserves historical behavior.
    eval_every_n_epochs: int = 1
    evaluate_train_on_eval: bool = True


def _item_batch_key(item: Any, batch_key: str | None) -> Any:
    if not batch_key:
        return None
    if str(batch_key) == "leaf_count":
        leaves = getattr(item, "leaves", None)
        if leaves is not None:
            return int(len(leaves))
    return None


def _batched(items: Sequence[Any], batch_size: int, batch_key: str | None = None):
    size = max(1, int(batch_size))
    key_name = str(batch_key or "")
    if not key_name:
        for index in range(0, len(items), size):
            yield list(items[index:index + size])
        return

    buckets: dict[Any, list[Any]] = {}
    order: list[Any] = []
    for item in items:
        key = _item_batch_key(item, key_name)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(item)
    for key in order:
        bucket = buckets[key]
        for index in range(0, len(bucket), size):
            yield list(bucket[index:index + size])


def _save_train_state(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list,
    best_metric_value: float,
    best_epoch: int,
    metric_key: str,
    extra: Mapping[str, Any],
) -> None:
    """Atomic save of a resumable training state (weights + optimizer + rng)."""
    try:
        rng_state = torch.get_rng_state()
    except Exception:
        rng_state = None
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "history": list(history),
        "best_metric_value": float(best_metric_value),
        "best_epoch": int(best_epoch),
        "best_metric_key": str(metric_key),
        "rng_state": rng_state,
        "extra": dict(extra or {}),
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def run_pytorch_training(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_items: Sequence[Any],
    val_items: Sequence[Any],
    supervision_adapter: SupervisionAdapter,
    evaluate_fn: EvaluateFn,
    config: PyTorchLoopConfig,
    output_dir: Path,
    checkpoint_extra: Mapping[str, Any] | None = None,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, Any]:
    """Run a standard PyTorch AdamW-style training loop.

    This is a modality-agnostic generalization of the inner loop that used to
    live inside `run_embedding_fno_training`. Callers are responsible for
    constructing the model, optimizer, data items, supervision adapter, and
    per-modality evaluation function.

    Returns a dict with keys: history, best_epoch, best_metric_value,
    best_checkpoint_path, final_train, final_val.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / "best_model.pt"
    latest_checkpoint_path = output_dir / "latest_model.pt"
    train_state_path = output_dir / "train_state.pt"
    epoch_checkpoint_dir = output_dir / "checkpoints"
    epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_metric_value = math.inf
    best_epoch = 0
    extra = dict(checkpoint_extra or {})
    metric_key = str(config.best_metric_key)
    start_epoch = 1

    # ---- Resume from prior training state, if any -----------------------
    resume_candidate = config.resume_from
    if resume_candidate is None and train_state_path.exists():
        resume_candidate = train_state_path
    if resume_candidate is not None and Path(resume_candidate).exists():
        saved = torch.load(resume_candidate, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(saved["model_state_dict"])
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            history = list(saved.get("history") or [])
            best_metric_value = float(saved.get("best_metric_value", math.inf))
            best_epoch = int(saved.get("best_epoch", 0))
            start_epoch = int(saved.get("epoch", 0)) + 1
            rng_state = saved.get("rng_state")
            if rng_state is not None:
                try:
                    torch.set_rng_state(rng_state)
                except Exception:
                    pass
        except KeyError:
            # Treat as incompatible snapshot; start from scratch.
            start_epoch = 1
    checkpoint_every = max(0, int(config.checkpoint_every_n_epochs))
    eval_every = max(1, int(config.eval_every_n_epochs))

    train_order = list(train_items)
    for epoch in range(start_epoch, int(config.n_epochs) + 1):
        model.train()
        random.Random(int(config.seed) + int(epoch)).shuffle(train_order)
        train_loss_sum = 0.0
        train_term_count = 0
        batch_count = 0
        for batch in _batched(
            train_order,
            int(config.train_batch_size),
            str(config.batch_key or "") or None,
        ):
            optimizer.zero_grad(set_to_none=True)
            prepared = supervision_adapter.prepare_batch(batch)
            loss, n_terms, _stats = supervision_adapter.compute_supervision_loss(prepared)
            if int(n_terms) <= 0:
                continue
            denom = max(1, int(n_terms))
            normalized_loss = loss / float(denom)
            normalized_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
            optimizer.step()
            train_loss_sum += float(normalized_loss.detach().cpu().item())
            train_term_count += int(n_terms)
            batch_count += 1

        should_eval = (
            epoch == int(config.n_epochs)
            or epoch == start_epoch
            or (epoch % eval_every == 0)
        )
        train_eval = {}
        val_eval = {}
        if should_eval and bool(config.evaluate_train_on_eval):
            train_eval = evaluate_fn(
                model=model,
                items=train_items,
                batch_size=int(config.train_batch_size),
            )
        if should_eval:
            val_eval = evaluate_fn(
                model=model,
                items=val_items,
                batch_size=int(config.train_batch_size),
            )
        epoch_payload = {
            "epoch": int(epoch),
            "train_batches": int(batch_count),
            "train_loss_mean": float(train_loss_sum / max(1, batch_count)),
            "train_loss_terms": int(train_term_count),
            "evaluated": bool(should_eval),
        }
        if train_eval:
            epoch_payload["train_mae_normalized"] = float(train_eval.get("mae_normalized", 0.0))
            epoch_payload["train_mae_raw"] = float(train_eval.get("mae_raw", 0.0))
        if val_eval:
            epoch_payload["val_mae_normalized"] = float(val_eval.get("mae_normalized", 0.0))
            epoch_payload["val_mae_raw"] = float(val_eval.get("mae_raw", 0.0))
        # Pass through any extra scalar metrics the evaluator returns so
        # objective-specific signals (e.g. val_merge_recon_mse) reach history.
        _passthrough_keys = {"count", "predictions"}
        for key, value in val_eval.items():
            if key in _passthrough_keys or key in epoch_payload:
                continue
            try:
                epoch_payload[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        history.append(epoch_payload)
        val_metric = float(val_eval.get(metric_key, math.inf)) if val_eval else math.inf
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "best_metric_key": metric_key,
            "best_metric_value": float(best_metric_value),
            "best_epoch": int(best_epoch),
            "current_metric_value": float(val_metric),
            "evaluated": bool(should_eval),
        }
        checkpoint_payload.update(extra)
        if val_eval and val_metric < float(best_metric_value):
            best_metric_value = val_metric
            best_epoch = int(epoch)
            checkpoint_payload["best_metric_value"] = float(best_metric_value)
            checkpoint_payload["best_epoch"] = int(best_epoch)
            torch.save(checkpoint_payload, best_checkpoint_path)
        torch.save(checkpoint_payload, latest_checkpoint_path)
        if bool(config.save_every_epoch):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": int(epoch),
                },
                epoch_checkpoint_dir / f"epoch_{epoch:03d}.pt",
            )

        if lr_scheduler is not None:
            try:
                lr_scheduler.step()
            except Exception:
                pass
        # Periodic training-state snapshot for resume.
        if checkpoint_every > 0 and (epoch % checkpoint_every == 0):
            _save_train_state(
                train_state_path,
                model=model, optimizer=optimizer, epoch=int(epoch),
                history=history, best_metric_value=best_metric_value,
                best_epoch=best_epoch, metric_key=metric_key, extra=extra,
            )

    # End-of-run snapshot: always save, regardless of periodic interval,
    # so the training can be resumed or inspected after completion.
    _save_train_state(
        train_state_path,
        model=model, optimizer=optimizer, epoch=int(config.n_epochs),
        history=history, best_metric_value=best_metric_value,
        best_epoch=best_epoch, metric_key=metric_key, extra=extra,
    )

    return {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_metric_value": float(best_metric_value),
        "best_metric_key": metric_key,
        "best_checkpoint_path": str(best_checkpoint_path),
        "latest_checkpoint_path": str(latest_checkpoint_path),
    }
