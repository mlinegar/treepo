"""Mergeable-sketch objective: root supervision + per-node C1/C2/C3 local laws.

Losses enforced at every node of the tree:

* **Root supervision** — `MSE(prediction, target)` where `prediction` is the
  head applied to the learned root state and `target` is the oracle
  target-bigram count.
* **C1 (per-leaf)** — `MSE(leaf_scalars, leaf_targets)` forces the head
  applied to each leaf state to predict the leaf's own target-bigram count.
* **C2 (merge reconstruction)** — `MSE(root_state, analytic_root)` plus
  per-internal-node `MSE(merge_states[i], analytic_merges[i])` forces the
  learned merge operator to reconstruct the analytic mergeable sketch at
  every level, not just at the root.
* **C3 (per-merge)** — `MSE(merge_scalars, merge_targets)` forces the head
  applied to each intermediate merge state to predict the cumulative
  target-bigram count at that node.

Without C1/C3 the merge operator can learn arbitrary intermediate
representations as long as the root happens to match — so local feedback at
every node is required for the merges to actually mean something.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from treepo._research.unified_g_v1.training.tree_task import TreeExample


def _stack_targets(batch: Sequence[TreeExample]) -> torch.Tensor:
    return torch.tensor([float(item.target) for item in batch], dtype=torch.float32)


def _stack_analytic_roots(
    batch: Sequence[TreeExample], *, vocab_size: int
) -> torch.Tensor:
    return torch.stack(
        [item.extra["analytic_root"].as_flat_tensor(vocab_size=vocab_size) for item in batch]
    )


def _stack_leaf_scalar_targets(batch: Sequence[TreeExample]) -> torch.Tensor:
    return torch.tensor(
        [list(item.extra["leaf_scalar_targets"]) for item in batch],
        dtype=torch.float32,
    )


def _stack_merge_scalar_targets(batch: Sequence[TreeExample]) -> torch.Tensor:
    return torch.tensor(
        [list(item.extra["merge_scalar_targets"]) for item in batch],
        dtype=torch.float32,
    )


def _stack_analytic_merge_states(
    batch: Sequence[TreeExample], *, vocab_size: int
) -> torch.Tensor:
    flat = [
        [sketch.as_flat_tensor(vocab_size=vocab_size) for sketch in item.extra["analytic_merge_sketches"]]
        for item in batch
    ]
    if not flat or not flat[0]:
        return torch.zeros(len(batch), 0, 0)
    return torch.stack([torch.stack(row) for row in flat], dim=0)


class MergeableSketchObjective:
    def __init__(
        self,
        *,
        vocab_size: int,
        target_bigram: tuple[int, int],
        local_law_weight: float = 0.3,
        c1_relative_weight: float = 1.0,
        c2_relative_weight: float = 1.0,
        c3_relative_weight: float = 1.0,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.target_bigram = target_bigram
        self.local_law_weight = float(local_law_weight)
        self.c1_relative_weight = float(c1_relative_weight)
        self.c2_relative_weight = float(c2_relative_weight)
        self.c3_relative_weight = float(c3_relative_weight)

    def _compose(
        self,
        *,
        root_loss: torch.Tensor,
        c1_loss: torch.Tensor | None,
        c2_loss: torch.Tensor | None,
        c3_loss: torch.Tensor | None,
    ) -> torch.Tensor:
        """Canonical `(1-λ)·root + λ · Σ ρᵢ·C_i / Σ ρᵢ` with drop-out for missing laws."""
        lam = max(0.0, min(1.0, float(self.local_law_weight)))
        active: list[tuple[float, torch.Tensor]] = []
        if c1_loss is not None:
            active.append((max(0.0, self.c1_relative_weight), c1_loss))
        if c2_loss is not None:
            active.append((max(0.0, self.c2_relative_weight), c2_loss))
        if c3_loss is not None:
            active.append((max(0.0, self.c3_relative_weight), c3_loss))
        rho_total = sum(rho for rho, _l in active)
        if not active or rho_total <= 0.0:
            return root_loss
        local_block = sum((rho / rho_total) * loss for rho, loss in active)
        return (1.0 - lam) * root_loss + lam * local_block

    def compute_loss(self, *, root_state, prediction, batch, forward_aux=None):
        targets = _stack_targets(batch)
        root_scalar_loss = nn.functional.mse_loss(prediction, targets)
        roots_analytic = _stack_analytic_roots(batch, vocab_size=self.vocab_size)
        root_c2 = nn.functional.mse_loss(root_state, roots_analytic)

        if forward_aux is None:
            # Fallback: no per-node states → only C2-at-root is available.
            loss = self._compose(
                root_loss=root_scalar_loss,
                c1_loss=None,
                c2_loss=root_c2,
                c3_loss=None,
            )
            return loss, int(len(batch)), {
                "root_scalar_loss": float(root_scalar_loss.detach()),
                "c2_loss_root": float(root_c2.detach()),
                "local_law_weight": float(self.local_law_weight),
            }

        leaf_scalars = forward_aux["leaf_scalars"]
        leaf_targets = _stack_leaf_scalar_targets(batch)
        c1_loss = nn.functional.mse_loss(leaf_scalars, leaf_targets)

        merge_states = forward_aux["merge_states"]
        merges_analytic = _stack_analytic_merge_states(batch, vocab_size=self.vocab_size)
        if merge_states.numel() > 0 and merges_analytic.numel() > 0:
            c2_merge_loss = nn.functional.mse_loss(merge_states, merges_analytic)
        else:
            c2_merge_loss = torch.zeros((), dtype=root_state.dtype)
        c2_loss = root_c2 + c2_merge_loss

        merge_scalars = forward_aux["merge_scalars"]
        merge_targets = _stack_merge_scalar_targets(batch)
        if merge_scalars.numel() > 0:
            c3_loss = nn.functional.mse_loss(merge_scalars, merge_targets)
        else:
            c3_loss = None

        loss = self._compose(
            root_loss=root_scalar_loss,
            c1_loss=c1_loss,
            c2_loss=c2_loss,
            c3_loss=c3_loss,
        )
        stats = {
            "root_scalar_loss": float(root_scalar_loss.detach()),
            "c1_loss": float(c1_loss.detach()),
            "c2_loss": float(c2_loss.detach()),
            "c2_loss_root": float(root_c2.detach()),
            "c2_loss_merges": float(c2_merge_loss.detach()),
            "c3_loss": float(c3_loss.detach()) if c3_loss is not None else 0.0,
            "local_law_weight": float(self.local_law_weight),
        }
        return loss, int(len(batch)), stats

    def evaluate(self, *, model, items, batch_size):
        model.eval()
        scalar_preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        merge_errs: list[torch.Tensor] = []
        leaf_errs: list[torch.Tensor] = []
        merge_scalar_errs: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(items), max(1, int(batch_size))):
                batch = list(items[start : start + int(batch_size)])
                result = model.forward_tree(batch)
                if isinstance(result, tuple) and len(result) == 3:
                    root_state, prediction, forward_aux = result
                else:
                    root_state, prediction = result
                    forward_aux = None
                scalar_preds.append(prediction)
                targets.append(_stack_targets(batch))
                roots_analytic = _stack_analytic_roots(batch, vocab_size=self.vocab_size)
                merge_errs.append(
                    nn.functional.mse_loss(
                        root_state, roots_analytic, reduction="none"
                    ).mean(dim=-1)
                )
                if forward_aux is not None:
                    leaf_preds = forward_aux["leaf_scalars"]
                    leaf_tgt = _stack_leaf_scalar_targets(batch)
                    leaf_errs.append((leaf_preds - leaf_tgt).abs().mean(dim=-1))
                    merge_preds = forward_aux["merge_scalars"]
                    if merge_preds.numel() > 0:
                        merge_tgt = _stack_merge_scalar_targets(batch)
                        merge_scalar_errs.append(
                            (merge_preds - merge_tgt).abs().mean(dim=-1)
                        )
        preds_vec = torch.cat(scalar_preds)
        targets_vec = torch.cat(targets)
        val_mae = float((preds_vec - targets_vec).abs().mean().item())
        merge_mse = float(torch.cat(merge_errs).mean().item())
        out: dict[str, Any] = {
            "count": int(len(items)),
            "val_mae": val_mae,
            "mae_raw": val_mae,
            "val_merge_recon_mse": merge_mse,
        }
        if leaf_errs:
            out["val_leaf_mae"] = float(torch.cat(leaf_errs).mean().item())
        if merge_scalar_errs:
            out["val_merge_scalar_mae"] = float(
                torch.cat(merge_scalar_errs).mean().item()
            )
        return out
