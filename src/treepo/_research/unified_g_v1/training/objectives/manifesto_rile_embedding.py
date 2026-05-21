"""Manifesto RILE embedding-FNO objective.

Predictions are normalized RILE in [0, 1] via the model's
`predict_normalized_batch`. The objective enforces root supervision plus
per-node local laws (C1/C2/C3) so the merge operator is actually
constrained, not just the root prediction.

Local laws enforced when the model supplies `forward_aux["per_doc"]`:

* **C1 — per-leaf RILE**: strict span-level RILE targets when the document
  comes from the Manifesto Project quasi-sentence codings; otherwise a soft
  pull-toward-root fallback.
* **C2 — merge commutativity**: `merge(a, b)` and `merge(b, a)` should give
  the same state at every internal node. RILE is permutation-invariant, so
  a well-calibrated merge operator must be commutative.
* **C3 — per-merge RILE**: strict span-level RILE targets when available;
  otherwise a soft pull-toward-root fallback.

The three local laws are combined with the root loss under the canonical
`(1-λ)·root + λ·Σρᵢ·Cᵢ/Σρ` composition over active laws.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from treepo._research.unified_g_v1.eval.law_stress import (
    DEFAULT_PRIMARY_GAIN_THRESHOLD,
    report as law_stress_report,
)


class ManifestoRileEmbeddingObjective:
    def __init__(
        self,
        *,
        rile_scale,
        baseline_val_mae: float | None = None,
        primary_gain_threshold: float = DEFAULT_PRIMARY_GAIN_THRESHOLD,
        local_law_weight: float = 0.3,
        c1_relative_weight: float = 1.0,
        c2_relative_weight: float = 1.0,
        c3_relative_weight: float = 1.0,
    ) -> None:
        self.rile_scale = rile_scale
        self.baseline_val_mae = (
            None if baseline_val_mae is None else float(baseline_val_mae)
        )
        self.primary_gain_threshold = float(primary_gain_threshold)
        self.local_law_weight = float(local_law_weight)
        self.c1_relative_weight = float(c1_relative_weight)
        self.c2_relative_weight = float(c2_relative_weight)
        self.c3_relative_weight = float(c3_relative_weight)

    def _normalize(self, raw: float) -> float:
        return float(
            max(0.0, min(1.0, self.rile_scale.normalize(float(raw))))
        )

    def _denormalize(self, normalized: float) -> float:
        return float(self.rile_scale.denormalize(float(normalized)))

    def _root_loss(self, prediction: torch.Tensor, batch: Sequence[Any]) -> torch.Tensor:
        device = prediction.device if hasattr(prediction, "device") else None
        targets_normalized = torch.tensor(
            [self._normalize(float(ex.target)) for ex in batch],
            dtype=torch.float32,
            device=device,
        )
        return nn.functional.mse_loss(prediction.reshape(-1), targets_normalized)

    def _per_node_targets(
        self,
        *,
        batch: Sequence[Any],
        doc_idx: int,
        key: str,
        device,
        n_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return `(targets, mask)` normalized to [0, 1] for the doc's nodes.

        `key` is either `"leaf_rile_targets"` or `"internal_rile_targets"`
        on `batch[doc_idx].extra`. Any None entries mark quasi-sentences
        without granular coding and are masked out of the loss.
        """
        ex = batch[doc_idx]
        raw = list((ex.extra or {}).get(key) or [])
        if not raw:
            return None
        # Pad/truncate to n_nodes so shape matches the predictions.
        while len(raw) < n_nodes:
            raw.append(None)
        raw = raw[:n_nodes]
        normalized: list[float] = []
        mask: list[float] = []
        for value in raw:
            if value is None:
                normalized.append(0.0)
                mask.append(0.0)
            else:
                normalized.append(self._normalize(float(value)))
                mask.append(1.0)
        if sum(mask) == 0:
            return None
        return (
            torch.tensor(normalized, dtype=torch.float32, device=device),
            torch.tensor(mask, dtype=torch.float32, device=device),
        )

    def compute_loss(self, *, root_state, prediction, batch, forward_aux=None):
        del root_state
        root_loss = self._root_loss(prediction, batch)
        if forward_aux is None:
            # No aux channel → pure root supervision. λ doesn't matter here
            # because there's no local-law component to weight against.
            return root_loss, int(len(batch)), {
                "root_loss": float(root_loss.detach().cpu().item()),
                "local_law_weight": float(self.local_law_weight),
            }

        device = prediction.device if hasattr(prediction, "device") else None
        per_doc = list(forward_aux.get("per_doc", ()))
        root_targets = torch.tensor(
            [self._normalize(float(ex.target)) for ex in batch],
            dtype=torch.float32,
            device=device,
        )

        c1_terms: list[torch.Tensor] = []
        c3_terms: list[torch.Tensor] = []
        c2_terms: list[torch.Tensor] = []
        strict_c1_count = 0
        strict_c3_count = 0
        for doc_idx, doc_aux in enumerate(per_doc):
            if doc_idx >= len(root_targets):
                break
            doc_target = root_targets[doc_idx]

            # --- C1 on leaves: strict targets from Manifesto quasi-sentence
            # codings when present; soft pull-to-root otherwise.
            leaf_preds = doc_aux.get("leaf_predictions")
            if leaf_preds is not None and leaf_preds.numel() > 0:
                preds_flat = leaf_preds.reshape(-1)
                strict = self._per_node_targets(
                    batch=batch,
                    doc_idx=doc_idx,
                    key="leaf_rile_targets",
                    device=device,
                    n_nodes=int(preds_flat.shape[0]),
                )
                if strict is not None:
                    tgts, mask = strict
                    squared = (preds_flat - tgts) ** 2
                    denom = float(mask.sum().item()) or 1.0
                    c1_terms.append((squared * mask).sum() / denom)
                    strict_c1_count += 1
                else:
                    c1_terms.append(
                        nn.functional.mse_loss(
                            preds_flat,
                            doc_target.expand(preds_flat.shape),
                        )
                    )

            # --- C3 on internal nodes: strict per-span RILE when present,
            # soft pull-to-root otherwise.
            internal_preds = doc_aux.get("internal_predictions")
            if internal_preds is not None and internal_preds.numel() > 0:
                preds_flat = internal_preds.reshape(-1)
                strict = self._per_node_targets(
                    batch=batch,
                    doc_idx=doc_idx,
                    key="internal_rile_targets",
                    device=device,
                    n_nodes=int(preds_flat.shape[0]),
                )
                if strict is not None:
                    tgts, mask = strict
                    squared = (preds_flat - tgts) ** 2
                    denom = float(mask.sum().item()) or 1.0
                    c3_terms.append((squared * mask).sum() / denom)
                    strict_c3_count += 1
                else:
                    c3_terms.append(
                        nn.functional.mse_loss(
                            preds_flat,
                            doc_target.expand(preds_flat.shape),
                        )
                    )

            # --- C2 merge commutativity: always strict (no target needed).
            orig_states = doc_aux.get("internal_states")
            swap_states = doc_aux.get("commutativity_states")
            if (
                orig_states is not None
                and swap_states is not None
                and orig_states.numel() > 0
                and swap_states.numel() > 0
            ):
                c2_terms.append(nn.functional.mse_loss(orig_states, swap_states))

        def _mean_or_zero(terms: list[torch.Tensor]) -> torch.Tensor:
            if not terms:
                return torch.zeros((), device=device)
            return torch.stack(terms, dim=0).mean()

        c1_loss = _mean_or_zero(c1_terms) if c1_terms else None
        c2_loss = _mean_or_zero(c2_terms) if c2_terms else None
        c3_loss = _mean_or_zero(c3_terms) if c3_terms else None

        # Canonical (1-λ)·root + λ · Σ ρᵢ·Cᵢ / Σ ρᵢ over ACTIVE laws.
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
            loss = root_loss
        else:
            local_block = sum((rho / rho_total) * l for rho, l in active)
            loss = (1.0 - lam) * root_loss + lam * local_block

        def _scalar(value) -> float:
            if value is None:
                return 0.0
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu().item())
            return float(value)

        stats = {
            "root_loss": _scalar(root_loss),
            "c1_loss": _scalar(c1_loss),
            "c2_loss": _scalar(c2_loss),
            "c3_loss": _scalar(c3_loss),
            "c1_strict_docs": float(strict_c1_count),
            "c3_strict_docs": float(strict_c3_count),
            "local_law_weight": float(lam),
        }
        return loss, int(len(batch)), stats

    def evaluate(self, *, model, items, batch_size):
        model.eval()
        errs_norm: list[float] = []
        errs_raw: list[float] = []
        with torch.no_grad():
            for start in range(0, len(items), max(1, int(batch_size))):
                batch = list(items[start : start + int(batch_size)])
                result = model.forward_tree(batch)
                if isinstance(result, tuple) and len(result) == 3:
                    _, prediction, _forward_aux = result
                else:
                    _, prediction = result
                pred_list = prediction.detach().cpu().tolist()
                for ex, pred_norm in zip(batch, pred_list):
                    target_raw = float(ex.target)
                    target_norm = self._normalize(target_raw)
                    pred_raw = self._denormalize(float(pred_norm))
                    errs_norm.append(abs(float(pred_norm) - target_norm))
                    errs_raw.append(abs(pred_raw - target_raw))
        mae_norm = float(sum(errs_norm) / max(1, len(errs_norm)))
        mae_raw = float(sum(errs_raw) / max(1, len(errs_raw)))
        out: dict[str, Any] = {
            "count": int(len(items)),
            "mae_normalized": mae_norm,
            "mae_raw": mae_raw,
            "val_mae": mae_raw,
        }
        if self.baseline_val_mae is not None:
            rpt = law_stress_report(
                model_mae=mae_raw,
                baseline_mae=self.baseline_val_mae,
                primary_gain_threshold=self.primary_gain_threshold,
            )
            out["baseline_val_mae"] = float(self.baseline_val_mae)
            out["val_mae_gain_frac"] = float(rpt.gain_frac)
            out["val_mae_pass"] = 1.0 if rpt.passed else 0.0
        return out
