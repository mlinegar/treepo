"""Markov changepoint workload expressed through the unified fit() framework.

Provides a `TreeOracle` + `TreeModel` + `TreeObjective` trio that wraps the
legacy `FNOCountSketch` so the Markov synthetic workload runs through
`fit() -> pytorch_tree_trainer -> run_pytorch_training` — the same loop
that drives sketch and embedding-FNO training. The loss terms (C1 leaf,
C2 re-summarization, C3 merge) are computed inside `FNOCountSketch.forward_doc`
and pushed to the objective via the `forward_aux` channel on `TreeObjective`.

This is the V1 "fit()-native" path. The legacy
`run_markov_changepoint_ops_count_experiment` still exists for papers that
need multi-stage schedules or adaptive audit sampling — features V1 doesn't
yet express.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from treepo._research.unified_g_v1.training.tree_task import TreeExample, TrainerConfig

from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (
    MarkovOPSDataBundle,
    OPSCountConfig,
)
from treepo._research.ctreepo.sim.core.markov_neural_operator_baselines import (
    DEFAULT_THEOREM_FEATURE_ADAPTER,
    FNOCountSketch,
    _FNOCountDoc,
    _prepare_fno_count_docs,
)


# ---------------------------------------------------------------------------
# Oracle: emits one TreeExample per doc with pre-computed FNO fields in extra.
# ---------------------------------------------------------------------------


@dataclass
class MarkovChangepointOracle:
    """Yields one `TreeExample` per document, with `_FNOCountDoc` in `extra`.

    Built from a `MarkovOPSDataBundle` (already materialized — see
    `markov/benchmarks.materialize_scope_bundle`) plus an `OPSCountConfig`
    that supplies `fixed_leaf_tokens` and downstream model hyperparameters.
    """

    config: OPSCountConfig
    bundle: MarkovOPSDataBundle
    schedule: str = "balanced"
    collect_leaf: bool = True
    collect_c2: bool = True
    collect_c3: bool = True

    def __post_init__(self) -> None:
        leaf_tokens = int(self.config.fixed_leaf_tokens)
        self._train_fno = _prepare_fno_count_docs(
            self.bundle.train_docs, leaf_tokens=leaf_tokens
        )
        self._val_fno = _prepare_fno_count_docs(
            self.bundle.val_docs, leaf_tokens=leaf_tokens
        )

    def _to_tree_example(self, fno_doc: _FNOCountDoc) -> TreeExample:
        extra = {
            "fno_doc": fno_doc,
            "schedule": str(self.schedule),
            "collect_leaf": bool(self.collect_leaf),
            "collect_c2": bool(self.collect_c2),
            "collect_c3": bool(self.collect_c3),
        }
        return TreeExample(
            leaves=list(fno_doc.leaf_token_ids),
            target=float(fno_doc.root_count),
            extra=extra,
        )

    def train_examples(self) -> Sequence[TreeExample]:
        return [self._to_tree_example(doc) for doc in self._train_fno]

    def val_examples(self) -> Sequence[TreeExample]:
        return [self._to_tree_example(doc) for doc in self._val_fno]

    def metadata(self) -> Mapping[str, Any]:
        return {
            "oracle": "markov_changepoint",
            "space_kind": "token_sequence",
            "vocab_size": int(self.config.vocab_size),
            "leaf_tokens": int(self.config.fixed_leaf_tokens),
            "n_regimes": int(self.config.n_regimes),
            "schedule": str(self.schedule),
            "collect_leaf": bool(self.collect_leaf),
            "collect_c2": bool(self.collect_c2),
            "collect_c3": bool(self.collect_c3),
        }


# ---------------------------------------------------------------------------
# Model: nn.Module wrapping FNOCountSketch with config-driven construction.
# ---------------------------------------------------------------------------


def _build_fno_count_sketch(config: OPSCountConfig, *, target_scale: float) -> FNOCountSketch:
    """Mirror the legacy construction block in `markov_changepoint_ops_count.py:7637`."""
    from treepo._research.ctreepo.sim.core.fno_arch_config import resolve_fno_arch

    arch = resolve_fno_arch(config)
    return FNOCountSketch(
        vocab_size=int(config.vocab_size),
        leaf_tokens=int(config.fixed_leaf_tokens),
        state_dim=int(config.state_dim),
        hidden_dim=int(config.hidden_dim),
        target_scale=float(target_scale),
        n_regimes=int(config.n_regimes),
        doc_sequence_class_values=(),
        fno_width=int(arch.width),
        fno_n_modes=int(arch.n_modes),
        fno_n_layers=int(arch.n_layers),
        root_supervision_kind=str(config.tree_root_supervision_kind),
        root_count_class_values=(),
        endpoint_loss_scale=float(config.endpoint_loss_scale),
        aligned_sketch_surface=str(config.aligned_sketch_surface),
        summary_spec_name=str(config.summary_spec_name),
        slot_count=int(config.slot_count),
        join_bit_weight=float(config.tree_join_bit_weight),
        task_head_mode=str(config.tree_task_head_mode),
        theorem_surface_mode=str(config.tree_theorem_surface_mode),
        theorem_count_head_mode=str(config.tree_theorem_count_head_mode),
        theorem_count_ordinal_weight=float(config.tree_theorem_count_ordinal_weight),
        theorem_count_scalar_aux_weight=float(config.tree_theorem_count_scalar_aux_weight),
        theorem_count_threshold_balance=bool(config.tree_theorem_count_threshold_balance),
        theorem_feature_dim=int(config.tree_theorem_feature_dim),
        theorem_feature_hidden_dim=int(config.tree_theorem_feature_hidden_dim),
        merge_hidden_dim=int(getattr(config, "tree_merge_hidden_dim", 0)),
        theorem_score_dim=int(getattr(config, "tree_theorem_score_dim", 0)),
        theorem_fiber_dim=int(getattr(config, "tree_theorem_fiber_dim", 0)),
        theorem_aux_dim=int(getattr(config, "tree_theorem_aux_dim", 0)),
        score_merge_mode=str(getattr(config, "tree_score_merge_mode", "gated_affine")),
        phi_alignment_loss=str(config.tree_phi_alignment_loss),
        c2_mode=str(getattr(config, "tree_c2_mode", "reconstruction")),
        theorem_feature_adapter=str(
            getattr(config, "theorem_feature_adapter", DEFAULT_THEOREM_FEATURE_ADAPTER)
        ),
        theorem_pair_same_threshold=getattr(config, "theorem_pair_same_threshold", None),
        theorem_pair_diff_threshold=getattr(config, "theorem_pair_diff_threshold", None),
        summary_spec_root_mode=str(config.tree_summary_spec_root_mode),
        theorem_count_dim=int(config.tree_theorem_count_dim),
        theorem_first_dim=int(config.tree_theorem_first_dim),
        theorem_last_dim=int(config.tree_theorem_last_dim),
        oracle_metric=None,
        oracle_same_threshold=float(getattr(config, "oracle_same_threshold", 0.0)),
        oracle_diff_threshold=float(getattr(config, "oracle_diff_threshold", 0.0)),
        tree_model_version=str(getattr(config, "tree_model_version", "legacy")),
    )


class MarkovFNOModel(nn.Module):
    """`TreeModel` over FNOCountSketch.

    `forward_tree(batch)` loops over the batch (forward_doc is per-doc) and
    returns `(root_state, prediction, forward_aux)`:
      * `root_state` is `(B, state_dim)` — stacked root states
      * `prediction` is `(B,)` — `pred_norm` values in normalized frame
      * `forward_aux["per_doc"]` is a list of dicts with `leaf_loss`,
        `c2_loss`, `c3_loss` tensors the objective weights and sums.
    """

    def __init__(
        self,
        *,
        config: OPSCountConfig,
        target_scale: float | None = None,
        depth_discount_gamma: float | None = None,
        tree_local_weighting_mode: str | None = None,
        tree_supervision_source: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        resolved_scale = (
            float(target_scale)
            if target_scale is not None
            else float(max(1.0, int(config.state_dim)))
        )
        self.target_scale = resolved_scale
        self.depth_discount_gamma = float(
            depth_discount_gamma
            if depth_discount_gamma is not None
            else getattr(config, "depth_discount_gamma", 1.0)
        )
        self.tree_local_weighting_mode = str(
            tree_local_weighting_mode
            if tree_local_weighting_mode is not None
            else getattr(config, "tree_local_weighting_mode", "fixed_k_hajek")
        )
        self.tree_supervision_source = str(
            tree_supervision_source
            if tree_supervision_source is not None
            else getattr(config, "tree_supervision_source", "rate")
        )
        self._fno = _build_fno_count_sketch(config, target_scale=resolved_scale)

    def _device(self) -> torch.device:
        try:
            return next(self._fno.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def forward_tree(
        self, batch: Sequence[TreeExample]
    ) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
        device = self._device()
        per_doc: list[dict[str, Any]] = []
        root_states: list[torch.Tensor] = []
        pred_norms: list[torch.Tensor] = []
        for example in batch:
            fno_doc: _FNOCountDoc = example.extra["fno_doc"]
            out = self._fno.forward_doc(
                list(fno_doc.leaf_token_ids),
                list(fno_doc.leaf_counts),
                list(fno_doc.merge_counts_balanced),
                list(fno_doc.merge_token_lengths),
                schedule=str(example.extra.get("schedule", "balanced")),
                collect_leaf=bool(example.extra.get("collect_leaf", True)),
                collect_c3=bool(example.extra.get("collect_c3", True)),
                collect_c2=bool(example.extra.get("collect_c2", True)),
                device=device,
                leaf_first_regimes=list(fno_doc.leaf_first_regimes),
                leaf_last_regimes=list(fno_doc.leaf_last_regimes),
                tree_local_weighting_mode=self.tree_local_weighting_mode,
                tree_supervision_source=self.tree_supervision_source,
                depth_discount_gamma=self.depth_discount_gamma,
            )
            root_states.append(out["root_state"].reshape(-1))
            pred_norms.append(out["pred_norm"].reshape(-1))
            per_doc.append(
                {
                    "leaf_loss": out.get("leaf_loss"),
                    "c2_loss": out.get("c2_loss"),
                    "c3_loss": out.get("c3_loss"),
                    "pred_count": out.get("pred_count"),
                    "root_count": float(fno_doc.root_count),
                }
            )
        root_state = torch.stack(root_states, dim=0)
        prediction = torch.cat(pred_norms, dim=0)
        forward_aux = {"per_doc": per_doc, "target_scale": float(self.target_scale)}
        return root_state, prediction, forward_aux


# ---------------------------------------------------------------------------
# Objective: sums weighted root MSE + C1 + C2 + C3 terms from forward_aux.
# ---------------------------------------------------------------------------


@dataclass
class MarkovChangepointObjective:
    """`TreeObjective` for the Markov workload.

    Reads per-doc `leaf_loss`/`c2_loss`/`c3_loss` tensors from `forward_aux`
    (populated by `MarkovFNOModel.forward_tree`) and combines them with
    the root-prediction MSE under the canonical formula:

        loss = (1 - λ) · root_mse
             + λ · [ρ_C1 · leaf_loss + ρ_C2 · c2_loss + ρ_C3 · c3_loss]
                   / (ρ_C1 + ρ_C2 + ρ_C3)

    with `λ = local_law_weight` (default 0.3, matching
    `scripts/run_markov_publication_bundle.py`) and the three ρ's equal
    by default (uniform local-law mass).
    """

    local_law_weight: float = 0.3
    c1_relative_weight: float = 1.0
    c2_relative_weight: float = 1.0
    c3_relative_weight: float = 1.0
    primary_metric_key: str = "mae_raw"

    def compute_loss(
        self,
        *,
        root_state: torch.Tensor,
        prediction: torch.Tensor,
        batch: Sequence[TreeExample],
        forward_aux: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, int, Mapping[str, Any]]:
        del root_state
        if forward_aux is None:
            raise RuntimeError(
                "MarkovChangepointObjective requires `forward_aux` from "
                "MarkovFNOModel.forward_tree"
            )
        per_doc = list(forward_aux.get("per_doc", ()))
        target_scale = float(forward_aux.get("target_scale", 1.0)) or 1.0
        if not per_doc:
            raise ValueError("MarkovChangepointObjective: empty batch in forward_aux")

        device = prediction.device
        targets_norm = torch.tensor(
            [float(item.target) / target_scale for item in batch],
            dtype=prediction.dtype,
            device=device,
        )
        root_mse = F.mse_loss(prediction.reshape(-1), targets_norm.reshape(-1))

        def _sum(key: str) -> torch.Tensor:
            acc: torch.Tensor | None = None
            for item in per_doc:
                tensor = item.get(key)
                if tensor is None:
                    continue
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.as_tensor(float(tensor), device=device)
                acc = tensor if acc is None else acc + tensor
            if acc is None:
                return torch.zeros((), device=device)
            return acc

        n = max(1, len(per_doc))
        leaf_mean = _sum("leaf_loss") / float(n)
        c2_mean = _sum("c2_loss") / float(n)
        c3_mean = _sum("c3_loss") / float(n)

        # Canonical λ balance: (1-λ)·root + λ · Σ ρᵢ·C_i / Σ ρᵢ.
        rho = {
            "c1": max(0.0, float(self.c1_relative_weight)),
            "c2": max(0.0, float(self.c2_relative_weight)),
            "c3": max(0.0, float(self.c3_relative_weight)),
        }
        rho_total = rho["c1"] + rho["c2"] + rho["c3"]
        lam = max(0.0, min(1.0, float(self.local_law_weight)))
        if rho_total <= 0.0:
            # All ρ's zeroed — degenerate to pure root.
            loss = root_mse
        else:
            local_block = (
                rho["c1"] * leaf_mean
                + rho["c2"] * c2_mean
                + rho["c3"] * c3_mean
            ) / rho_total
            loss = (1.0 - lam) * root_mse + lam * local_block

        def _scalar(t: torch.Tensor) -> float:
            return float(t.detach().cpu().item()) if isinstance(t, torch.Tensor) else float(t)

        stats = {
            "root_mse": _scalar(root_mse),
            "leaf_loss": _scalar(leaf_mean),
            "c2_loss": _scalar(c2_mean),
            "c3_loss": _scalar(c3_mean),
            "local_law_weight": float(lam),
        }
        return loss, int(len(batch)), stats

    def evaluate(
        self,
        *,
        model: nn.Module,
        items: Sequence[TreeExample],
        batch_size: int,
    ) -> Mapping[str, Any]:
        model.eval()
        preds_norm: list[float] = []
        targets_raw: list[float] = []
        leaf_losses: list[float] = []
        c2_losses: list[float] = []
        c3_losses: list[float] = []
        target_scale = 1.0
        with torch.no_grad():
            step = max(1, int(batch_size))
            for start in range(0, len(items), step):
                batch = list(items[start : start + step])
                _root_state, prediction, forward_aux = model.forward_tree(batch)
                target_scale = float(forward_aux.get("target_scale", 1.0)) or 1.0
                preds_norm.extend(prediction.detach().cpu().tolist())
                for item in batch:
                    targets_raw.append(float(item.target))
                for doc_aux in forward_aux.get("per_doc", ()):
                    for key, bucket in (
                        ("leaf_loss", leaf_losses),
                        ("c2_loss", c2_losses),
                        ("c3_loss", c3_losses),
                    ):
                        tensor = doc_aux.get(key)
                        if tensor is None:
                            continue
                        if isinstance(tensor, torch.Tensor):
                            bucket.append(float(tensor.detach().cpu().item()))
                        else:
                            bucket.append(float(tensor))
        pred_raw = [float(p) * target_scale for p in preds_norm]
        errs_raw = [abs(p - t) for p, t in zip(pred_raw, targets_raw)]
        errs_norm = [
            abs(float(p) - float(t) / target_scale)
            for p, t in zip(preds_norm, targets_raw)
        ]
        n = max(1, len(errs_raw))
        out: dict[str, Any] = {
            "count": int(len(items)),
            "mae_raw": float(sum(errs_raw) / n),
            "mae_normalized": float(sum(errs_norm) / n),
            "predictions": pred_raw,
        }

        def _mean(values: list[float]) -> float | None:
            if not values:
                return None
            return float(sum(values) / max(1, len(values)))

        for key, values in (
            ("val_leaf_loss", leaf_losses),
            ("val_c2_loss", c2_losses),
            ("val_c3_loss", c3_losses),
        ):
            mean = _mean(values)
            if mean is not None:
                out[key] = mean
        return out


# ---------------------------------------------------------------------------
# Preset constructor.
# ---------------------------------------------------------------------------


def markov_changepoint_task(
    *,
    config: OPSCountConfig,
    bundle: MarkovOPSDataBundle,
    n_epochs: int | None = None,
    train_batch_size: int | None = None,
    learning_rate: float | None = None,
    seed: int | None = None,
    schedule: str = "balanced",
    collect_leaf: bool = True,
    collect_c2: bool = True,
    collect_c3: bool = True,
    local_law_weight: float = 0.3,
    c1_relative_weight: float = 1.0,
    c2_relative_weight: float = 1.0,
    c3_relative_weight: float = 1.0,
    target_scale: float | None = None,
    best_metric_key: str = "mae_raw",
) -> TrainerConfig:
    """Build a `TrainerConfig` that runs Markov through `fit()`.

    `local_law_weight` (λ, default 0.3) balances root vs. local laws;
    `*_relative_weight` knobs balance C1/C2/C3 against each other (default
    equal).
    """
    oracle = MarkovChangepointOracle(
        config=config,
        bundle=bundle,
        schedule=schedule,
        collect_leaf=collect_leaf,
        collect_c2=collect_c2,
        collect_c3=collect_c3,
    )
    model = MarkovFNOModel(config=config, target_scale=target_scale)
    objective = MarkovChangepointObjective(
        local_law_weight=float(local_law_weight),
        c1_relative_weight=float(c1_relative_weight),
        c2_relative_weight=float(c2_relative_weight),
        c3_relative_weight=float(c3_relative_weight),
        primary_metric_key=str(best_metric_key),
    )
    return TrainerConfig(
        oracle=oracle,
        model=model,
        objective=objective,
        n_epochs=int(n_epochs if n_epochs is not None else config.n_epochs),
        train_batch_size=int(
            train_batch_size if train_batch_size is not None else config.batch_size
        ),
        learning_rate=float(
            learning_rate if learning_rate is not None else config.lr
        ),
        seed=int(seed if seed is not None else config.seed),
        best_metric_key=str(best_metric_key),
    )
