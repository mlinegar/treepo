"""`SimpleObjective` — wrap a plain loss callable as a TreeObjective.

The full `TreeObjective` protocol wants `compute_loss` + `evaluate` methods.
Most simple cases (MSE on a scalar target, cross-entropy on a class label)
only need a loss callable. `SimpleObjective` accepts such a callable and
synthesizes a reasonable default evaluator.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn


LossFn = Callable[[torch.Tensor, torch.Tensor, Sequence[Any]], Any]
"""Signature: loss_fn(root_state, prediction, batch) -> Tensor | (Tensor, n_terms, stats)."""


EvaluateFn = Callable[..., Mapping[str, Any]]


class SimpleObjective:
    """Adapt a plain callable into a full TreeObjective.

    `loss_fn(root_state, prediction, batch)` returns either a loss tensor or
    a `(loss, n_terms, stats)` triple. If `evaluate_fn` is omitted, a default
    evaluator runs the model on the items and reports MAE against each
    example's `target` attribute.
    """

    def __init__(
        self,
        loss_fn: LossFn,
        evaluate_fn: EvaluateFn | None = None,
        *,
        best_metric_key: str = "val_mae",
    ) -> None:
        self.loss_fn = loss_fn
        self._evaluate_fn = evaluate_fn
        self.best_metric_key = str(best_metric_key)

    def compute_loss(
        self,
        *,
        root_state: torch.Tensor,
        prediction: torch.Tensor,
        batch: Sequence[Any],
        forward_aux: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, int, Mapping[str, Any]]:
        del forward_aux  # plain loss_fn wrapper ignores the aux channel
        result = self.loss_fn(root_state, prediction, batch)
        if isinstance(result, tuple) and len(result) == 3:
            loss, n_terms, stats = result
            return loss, int(n_terms), dict(stats)
        return result, int(len(batch)), {}

    def evaluate(
        self,
        *,
        model: nn.Module,
        items: Sequence[Any],
        batch_size: int,
    ) -> Mapping[str, Any]:
        if self._evaluate_fn is not None:
            return self._evaluate_fn(model=model, items=items, batch_size=batch_size)
        # Default: compute MAE over scalar targets via model.forward_tree.
        # `forward_tree` may return either a 2-tuple `(root_state, prediction)`
        # or a 3-tuple `(root_state, prediction, forward_aux)` — accept both.
        model.eval()
        errs: list[float] = []
        with torch.no_grad():
            for start in range(0, len(items), max(1, int(batch_size))):
                batch = list(items[start : start + int(batch_size)])
                result = model.forward_tree(batch)
                if isinstance(result, tuple) and len(result) == 3:
                    _, prediction, _ = result
                else:
                    _, prediction = result
                targets = torch.tensor(
                    [float(ex.target) for ex in batch], dtype=torch.float32
                )
                errs.extend((prediction.detach().cpu() - targets).abs().tolist())
        mae = float(sum(errs) / max(1, len(errs)))
        return {"count": int(len(items)), "val_mae": mae, "mae_raw": mae}


def as_objective(obj_or_callable: Any) -> Any:
    """Return a TreeObjective: identity if already one, wrap if callable."""
    if obj_or_callable is None:
        return None
    if callable(obj_or_callable) and not hasattr(obj_or_callable, "compute_loss"):
        return SimpleObjective(obj_or_callable)
    return obj_or_callable
