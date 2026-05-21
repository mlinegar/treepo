"""Learned-g and learned-joint companions to `classical_hll_parity`.

Three variants, all trained against per-node classical-HLL targets emitted by
`ClassicalHLLParityOracle(oracle_kind="hll_reference")`:

1. **learn_readout=False ("learned g, classical f")** — the learned leaf and merge
   operator produces an HLL-register-space state (dim = 2^precision), and
   `f` is the classical HLL estimator formula applied differentiably to
   that state. Tests whether a learned g can reproduce the classical
   mergeable-sketch's register semantics.

2. **learn_readout=True ("learned g, learned f")** — the merge operator
   and the scalar readout are both learned MLPs. Arbitrary state_dim
   respecting the repo-wide FNO-head 2× invariant. Tests the fully
   end-to-end learned pipeline.

3. **learned_g_oracle_state** — leaf states are fixed native HLL registers,
   `f` is fixed to the classical HLL estimator over those registers, and only
   the merge operator is learned. At one leaf per document, no learned operator
   is on the path, so the supplied oracle is recovered exactly.

Both share the synthetic Zipfian document generator from `classical_parity`
so rows land on the same axes in the paper figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from treepo.hll import HLLConfig, HyperLogLogSketch
from treepo.training.local_law import (
    observed_uniform_node_ipw_mean_loss,
    sampled_uniform_node_ipw_mean_loss,
)

from treepo._research.unified_g_v1.sketch.classical_parity import (
    BackendName,
    ClassicalHLLParityConfig,
    ClassicalHLLParityOracle,
    OracleKind,
    ScheduleName,
    TargetFn,
    generate_documents,
)
from treepo._research.unified_g_v1.dimension_guards import promote_dim
from treepo._research.unified_g_v1.sketch.sampled_supervision import (
    attach_persistent_uniform_node_scores,
    persistent_uniform_node_mask,
    sampled_axis_mse,
    sampled_batch_mse,
    sampled_tree_node_mse,
)
from treepo._research.unified_g_v1.training.tree_task import TreeExample, TrainerConfig


MethodKind = Literal[
    "learned_g",
    "learned_joint",
    "learned_fg",
    "learned_g_oracle_state",
    "learned_f_oracle_state",
]


def _leaf_width_floor(
    *,
    max_tokens: int,
    n_leaves: int | None,
    leaf_size: int | None,
) -> int:
    """Return the 2x no-compression width for the largest leaf in a cell."""

    if n_leaves is not None:
        max_leaf_tokens = int(
            math.ceil(float(max(1, int(max_tokens))) / float(max(1, int(n_leaves))))
        )
    elif leaf_size is not None:
        max_leaf_tokens = min(max(1, int(max_tokens)), max(1, int(leaf_size)))
    else:
        max_leaf_tokens = max(1, int(max_tokens))
    return 2 * int(max_leaf_tokens)


# ---------------------------------------------------------------------------
# Differentiable HLL estimator (for the classical-f variant).
# ---------------------------------------------------------------------------


def _hll_alpha(m: int) -> float:
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / float(m))


def hll_estimate_differentiable(registers: torch.Tensor) -> torch.Tensor:
    """Classical HLL estimator applied to a float register vector.

    Shape: `registers` is `(..., m)`. Returns `(...,)`. Uses the raw HLL
    formula plus the small-range linear-counting correction; the correction
    is gated through `torch.where`, which is piecewise-differentiable away
    from the branch boundary.
    """
    m = registers.shape[-1]
    alpha = float(_hll_alpha(int(m)))
    # Match the native HLL estimator in double precision. Clamp to prevent
    # numerically huge contributions from rogue negative register values during
    # early training.
    clamped = torch.clamp(registers.double(), min=0.0, max=64.0)
    z = torch.exp2(-clamped).sum(dim=-1)  # (...,)
    raw = (alpha * float(m) * float(m)) / torch.clamp(z, min=1e-9)
    # Linear counting: the count of ~zero registers is soft-approximated by
    # relu(1 - register) so gradients keep flowing.
    n_zeros = torch.clamp(1.0 - clamped, min=0.0).sum(dim=-1)
    linear = float(m) * torch.log(float(m) / torch.clamp(n_zeros, min=1e-3))
    use_linear = ((raw <= 2.5 * float(m)) & (n_zeros > 0.5)).detach()
    small_range = torch.where(use_linear, linear, raw)
    hash_space = float(2.0 ** 64)
    clipped = torch.clamp(raw / hash_space, max=1.0 - 1e-12)
    large_range = -hash_space * torch.log1p(-clipped)
    use_large = (raw > hash_space / 30.0).detach()
    return torch.where(use_large, large_range, small_range)


def _native_hll_registers(
    tokens: Sequence[int],
    *,
    precision: int,
) -> np.ndarray:
    cfg = HLLConfig(precision=int(precision), hash_bits=64)
    return HyperLogLogSketch.from_tokens(cfg, list(tokens)).registers.astype(np.float64, copy=True)


def _native_hll_registers_uint8(
    tokens: Sequence[int],
    *,
    precision: int,
) -> np.ndarray:
    cfg = HLLConfig(precision=int(precision), hash_bits=64)
    return HyperLogLogSketch.from_tokens(cfg, list(tokens)).registers.astype(np.uint8, copy=True)


def _hll_estimate_np(registers: np.ndarray, *, precision: int) -> float:
    cfg = HLLConfig(precision=int(precision), hash_bits=64)
    return float(HyperLogLogSketch.from_registers(cfg, np.asarray(registers, dtype=np.uint8)).estimate())


def _native_register_tensor(
    tokens: Sequence[int],
    *,
    precision: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        _native_hll_registers(tokens, precision=int(precision)),
        dtype=torch.float64,
        device=device,
    )


# ---------------------------------------------------------------------------
# Learned model.
# ---------------------------------------------------------------------------


class LearnedHLLMergeModel(nn.Module):
    """Learned-g model, optionally with learned-f readout.

    When `learn_readout=False`, `state_dim` is forced to `2 ** precision` so
    the classical HLL estimator can read the state as an HLL register vector.
    Otherwise `state_dim >= 2 * summary_dim` (repo-wide FNO head invariant).
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        precision: int,
        embedding_dim: int,
        summary_dim: int,
        hidden_dim: int,
        state_dim: int | None = None,
        target_scale: float = 10_000.0,
        learn_readout: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.precision = int(precision)
        self.m = 1 << self.precision
        self.embedding_dim = int(embedding_dim)
        self.summary_dim = int(summary_dim)
        self.hidden_dim = max(int(hidden_dim), self.m)
        self.target_scale = float(target_scale)
        self.learn_readout = bool(learn_readout)

        if self.learn_readout:
            self.state_dim = int(state_dim or 2 * self.summary_dim)
            if self.state_dim < 2 * self.summary_dim:
                raise ValueError(
                    f"learned_hll: state_dim={self.state_dim} violates the "
                    f"FNO head 2x invariant (summary_dim={self.summary_dim})"
                )
        else:
            # Classical f reads state as an HLL register vector → state_dim=m.
            self.state_dim = self.m

        # Embedding (with a pad index at `vocab_size`).
        self.embedding = nn.Embedding(
            self.vocab_size + 1, self.embedding_dim, padding_idx=self.vocab_size
        )

        # Leaf adapter: pooled embedding -> summary_dim. HLL leaf states are
        # register-wise maxima over token hash contributions, so expose both
        # mean and max token pools to avoid forcing distinct-count structure
        # through an averaging bottleneck.
        self.leaf_adapter = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.summary_dim),
        )

        # Shared g: summary_dim → state_dim.
        self.g = nn.Sequential(
            nn.Linear(self.summary_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )

        # Merge adapter: (state, state) → summary_dim, then g maps to state_dim.
        self.merge_adapter = nn.Sequential(
            nn.Linear(2 * self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.summary_dim),
        )

        if self.learn_readout:
            self.f_head = nn.Sequential(
                nn.Linear(self.state_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )
        else:
            self.f_head = None  # classical f applied to state below

    # -----------------------------------------------------------------------
    # Per-example forward; the caller batches over a list of TreeExamples.
    # -----------------------------------------------------------------------

    def _pad_token_ids(self, leaves: Sequence[Sequence[int]]) -> torch.Tensor:
        max_tokens = max(1, max(len(leaf) for leaf in leaves))
        pad = self.vocab_size
        device = next(self.parameters()).device
        out = torch.full((len(leaves), max_tokens), pad, dtype=torch.long, device=device)
        for i, leaf in enumerate(leaves):
            if len(leaf) > 0:
                out[i, : len(leaf)] = torch.tensor(
                    [int(t) % self.vocab_size for t in leaf],
                    dtype=torch.long,
                    device=device,
                )
        return out

    def _encode_leaves(self, leaves: Sequence[Sequence[int]]) -> torch.Tensor:
        token_ids = self._pad_token_ids(leaves)  # (L, T)
        mask_bool = (token_ids != self.vocab_size).unsqueeze(-1)  # (L, T, 1)
        mask = mask_bool.float()
        embeds = self.embedding(token_ids)  # (L, T, E)
        summed = (embeds * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        mean_pool = summed / counts  # (L, E)
        masked = embeds.masked_fill(~mask_bool, -1.0e9)
        max_pool = masked.max(dim=1).values
        has_tokens = mask_bool.squeeze(-1).any(dim=1)
        max_pool = torch.where(has_tokens.unsqueeze(-1), max_pool, torch.zeros_like(max_pool))
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # (L, 2E)
        summary = self.leaf_adapter(pooled)  # (L, summary_dim)
        return self.g(summary)  # (L, state_dim)

    def _predict_scalar(self, state: torch.Tensor) -> torch.Tensor:
        if self.learn_readout:
            logit = self.f_head(state).reshape(-1)
            return torch.sigmoid(logit) * self.target_scale
        # Classical f: HLL estimator formula over the state interpreted as
        # register values. Soften with a (relu + small constant) so the state
        # behaves more like non-negative uint8 register counts during training.
        register_like = F.softplus(state)
        return hll_estimate_differentiable(register_like)

    def forward_tree(
        self, batch: Sequence[TreeExample]
    ) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
        device = next(self.parameters()).device
        roots: list[torch.Tensor] = []
        leaf_scalars_list: list[torch.Tensor] = []
        merge_scalars_list: list[torch.Tensor] = []
        for item in batch:
            leaf_states = self._encode_leaves(list(item.leaves))  # (L, state_dim)
            leaf_scalars = self._predict_scalar(leaf_states)  # (L,)
            state = leaf_states[0]
            merge_scalars: list[torch.Tensor] = []
            for idx in range(1, leaf_states.shape[0]):
                pair = torch.cat([state, leaf_states[idx]], dim=-1)
                summary = self.merge_adapter(pair.unsqueeze(0)).squeeze(0)
                state = self.g(summary.unsqueeze(0)).squeeze(0)
                merge_scalars.append(self._predict_scalar(state.unsqueeze(0)).squeeze(0))
            roots.append(state.unsqueeze(0))
            leaf_scalars_list.append(leaf_scalars.unsqueeze(0))
            if merge_scalars:
                merge_scalars_list.append(torch.stack(merge_scalars).unsqueeze(0))
            else:
                merge_scalars_list.append(
                    torch.zeros(1, 0, device=device, dtype=leaf_states.dtype)
                )
        root_state = torch.cat(roots, dim=0)  # (B, state_dim)
        root_scalar = self._predict_scalar(root_state)  # (B,)
        leaf_scalars_tensor = torch.cat(leaf_scalars_list, dim=0)
        merge_scalars_tensor = torch.cat(merge_scalars_list, dim=0)
        forward_aux = {
            "leaf_scalars": leaf_scalars_tensor,
            "merge_scalars": merge_scalars_tensor,
        }
        return root_state, root_scalar, forward_aux


class OracleStateHLLMergeModel(nn.Module):
    """Learn only HLL register merge while leaf states and readout are fixed.

    Leaves are encoded as native `treepo.hll.HyperLogLogSketch` register
    vectors. The scalar readout is the classical HLL estimator over those
    registers. The only trainable map is a shared per-register pair network.
    """

    def __init__(
        self,
        *,
        precision: int,
        hidden_dim: int = 128,
        variant: Literal["f", "g"] = "g",
        readout_arch: Literal["structured", "mlp"] = "structured",
        use_learned_readout: bool | None = None,
        init_from: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.precision = int(precision)
        self.m = 1 << self.precision
        self.state_dim = self.m
        self.hidden_dim = int(hidden_dim)
        if variant not in {"f", "g"}:
            raise ValueError(f"HLL oracle-state variant must be 'f' or 'g', got {variant!r}")
        self.variant = str(variant)
        self.learn_readout = (
            self.variant == "f"
            if use_learned_readout is None
            else bool(use_learned_readout)
        )
        if readout_arch not in {"structured", "mlp"}:
            raise ValueError(f"HLL oracle-state readout_arch must be 'structured' or 'mlp', got {readout_arch!r}")
        self.readout_arch = str(readout_arch)
        # HLL's exposed register-state merge is elementwise max. Keep that law
        # exact in the oracle-state lane so this path is a recoverability
        # anchor, not a numerical test of whether a smooth approximation to max
        # is sharp enough.
        self.log_sharpness = nn.Parameter(torch.tensor(40.0, dtype=torch.float64))
        self.readout_scale = nn.Parameter(torch.ones((), dtype=torch.float64))
        self.readout_bias = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.f_head = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        ).double()
        if init_from is not None:
            self._load_from_checkpoint(Path(init_from))
        if self.readout_arch == "structured":
            final = self.f_head[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        if self.variant == "f":
            self.log_sharpness.requires_grad = False
            if self.readout_arch == "structured":
                for p in self.f_head.parameters():
                    p.requires_grad = False
        else:
            self.readout_scale.requires_grad = False
            self.readout_bias.requires_grad = False
            for p in self.f_head.parameters():
                p.requires_grad = False

    def _load_from_checkpoint(self, ckpt_path: Path) -> None:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload)
        own = self.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in own and tuple(value.shape) == tuple(own[key].shape)
        }
        if not compatible:
            raise ValueError(f"HLL oracle-state: no compatible weights in {ckpt_path}")
        own.update(compatible)
        self.load_state_dict(own, strict=False)

    def _parameter_anchor(self) -> torch.Tensor:
        return next(self.parameters()).sum() * 0.0

    def _encode_leaves(self, leaves: Sequence[Sequence[int]], device: torch.device) -> torch.Tensor:
        return torch.stack(
            [
                _native_register_tensor(leaf, precision=self.precision, device=device)
                for leaf in leaves
            ],
            dim=0,
        )

    def _encode_item_leaves(self, item: TreeExample, device: torch.device) -> torch.Tensor:
        cached = item.extra.get("hll_leaf_registers") if hasattr(item, "extra") else None
        if cached is not None:
            return torch.as_tensor(
                np.asarray(cached),
                dtype=torch.float64,
                device=device,
            )
        return self._encode_leaves(list(item.leaves), device)

    def _is_rectangular_batch(self, batch: Sequence[TreeExample]) -> bool:
        if not batch:
            return False
        width = len(batch[0].leaves)
        return all(len(item.leaves) == width for item in batch)

    def _encode_batch_leaf_states(
        self,
        batch: Sequence[TreeExample],
        device: torch.device,
    ) -> torch.Tensor:
        """Encode a rectangular minibatch as ``[batch, leaves, registers]``."""
        if not self._is_rectangular_batch(batch):
            raise ValueError("HLL oracle-state batch must have equal leaf counts")
        cached_rows: list[Any] = []
        all_cached = True
        for item in batch:
            cached = item.extra.get("hll_leaf_registers") if hasattr(item, "extra") else None
            if cached is None:
                all_cached = False
                break
            cached_rows.append(cached)
        if all_cached:
            return torch.as_tensor(
                np.asarray(cached_rows),
                dtype=torch.float64,
                device=device,
            )
        flat_leaves = [leaf for item in batch for leaf in item.leaves]
        n_leaves = int(len(batch[0].leaves))
        return self._encode_leaves(flat_leaves, device).reshape(
            int(len(batch)),
            n_leaves,
            self.state_dim,
        )

    def _cached_batch_cumulative_targets(
        self,
        batch: Sequence[TreeExample],
        device: torch.device,
    ) -> torch.Tensor | None:
        rows: list[Any] = []
        for item in batch:
            cached = (
                item.extra.get("hll_cumulative_registers") if hasattr(item, "extra") else None
            )
            if cached is None:
                return None
            rows.append(cached)
        return torch.as_tensor(
            np.asarray(rows),
            dtype=torch.float64,
            device=device,
        )

    def _predict_scalar(self, state: torch.Tensor) -> torch.Tensor:
        if self.learn_readout and self.readout_arch == "mlp":
            return self.f_head(state.double()).reshape(-1)
        value = hll_estimate_differentiable(state)
        if self.learn_readout:
            if self.readout_arch == "structured":
                return value + self.readout_scale.to(value.dtype) * 0.0
            residual = self.f_head(state.double()).reshape(-1).to(value.dtype)
            return value + residual
        return value

    def _merge_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.maximum(left, right) + self.log_sharpness.to(left.dtype) * 0.0

    def forward_tree(
        self, batch: Sequence[TreeExample]
    ) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
        device = next(self.parameters()).device
        if self._is_rectangular_batch(batch):
            leaf_states = self._encode_batch_leaf_states(batch, device)  # (B, L, m)
            batch_size, n_leaves, _state_dim = leaf_states.shape
            leaf_scalars = self._predict_scalar(
                leaf_states.reshape(batch_size * n_leaves, self.state_dim)
            ).reshape(batch_size, n_leaves)
            state = leaf_states[:, 0, :]
            oracle_state = leaf_states[:, 0, :]
            cumulative_targets = self._cached_batch_cumulative_targets(batch, device)
            merge_scalars: list[torch.Tensor] = []
            merge_states: list[torch.Tensor] = []
            merge_state_targets: list[torch.Tensor] = []
            for idx in range(1, n_leaves):
                state = self._merge_pair(state, leaf_states[:, idx, :])
                oracle_state = torch.maximum(oracle_state, leaf_states[:, idx, :])
                target_state = (
                    cumulative_targets[:, idx - 1, :]
                    if cumulative_targets is not None
                    else oracle_state
                )
                merge_scalars.append(self._predict_scalar(state).unsqueeze(1))
                merge_states.append(state.unsqueeze(1))
                merge_state_targets.append(target_state.unsqueeze(1))
            if merge_scalars:
                merge_scalars_tensor = torch.cat(merge_scalars, dim=1)
                merge_states_tensor = torch.cat(merge_states, dim=1)
                merge_state_targets_tensor = torch.cat(merge_state_targets, dim=1)
            else:
                merge_scalars_tensor = torch.zeros(
                    batch_size,
                    0,
                    device=device,
                    dtype=torch.float64,
                )
                merge_states_tensor = torch.zeros(
                    batch_size,
                    0,
                    self.state_dim,
                    device=device,
                    dtype=torch.float64,
                )
                merge_state_targets_tensor = torch.zeros(
                    batch_size,
                    0,
                    self.state_dim,
                    device=device,
                    dtype=torch.float64,
                )
            root_state = state
            root_scalar = self._predict_scalar(root_state) + self._parameter_anchor()
            return root_state, root_scalar, {
                "leaf_scalars": leaf_scalars,
                "merge_scalars": merge_scalars_tensor,
                "merge_states": merge_states_tensor,
                "merge_state_targets": merge_state_targets_tensor,
            }

        roots: list[torch.Tensor] = []
        leaf_scalars_list: list[torch.Tensor] = []
        merge_scalars_list: list[torch.Tensor] = []
        merge_states_list: list[torch.Tensor] = []
        merge_state_targets_list: list[torch.Tensor] = []
        for item in batch:
            leaves = list(item.leaves)
            leaf_states = self._encode_item_leaves(item, device)  # (L, m)
            cached_cumulative = (
                item.extra.get("hll_cumulative_registers") if hasattr(item, "extra") else None
            )
            cumulative_targets = (
                torch.as_tensor(
                    np.asarray(cached_cumulative),
                    dtype=torch.float64,
                    device=device,
                )
                if cached_cumulative is not None
                else None
            )
            leaf_scalars = self._predict_scalar(leaf_states)  # (L,)
            state = leaf_states[0]
            oracle_state = leaf_states[0]
            merge_scalars: list[torch.Tensor] = []
            merge_states: list[torch.Tensor] = []
            merge_state_targets: list[torch.Tensor] = []
            for idx in range(1, leaf_states.shape[0]):
                state = self._merge_pair(state, leaf_states[idx])
                oracle_state = torch.maximum(oracle_state, leaf_states[idx])
                target_state = (
                    cumulative_targets[idx - 1]
                    if cumulative_targets is not None
                    else oracle_state
                )
                merge_scalars.append(self._predict_scalar(state.unsqueeze(0)).squeeze(0))
                merge_states.append(state)
                merge_state_targets.append(target_state)
            roots.append(state.unsqueeze(0))
            leaf_scalars_list.append(leaf_scalars.unsqueeze(0))
            if merge_scalars:
                merge_scalars_list.append(torch.stack(merge_scalars).unsqueeze(0))
                merge_states_list.append(torch.stack(merge_states).unsqueeze(0))
                merge_state_targets_list.append(torch.stack(merge_state_targets).unsqueeze(0))
            else:
                merge_scalars_list.append(torch.zeros(1, 0, device=device, dtype=torch.float64))
                merge_states_list.append(torch.zeros(1, 0, self.state_dim, device=device, dtype=torch.float64))
                merge_state_targets_list.append(
                    torch.zeros(1, 0, self.state_dim, device=device, dtype=torch.float64)
                )
        root_state = torch.cat(roots, dim=0)  # (B, m)
        # Add a zero-valued parameter dependency so one-leaf exact-anchor cells
        # still support backward() through the generic training loop.
        root_scalar = self._predict_scalar(root_state) + self._parameter_anchor()
        forward_aux = {
            "leaf_scalars": torch.cat(leaf_scalars_list, dim=0),
            "merge_scalars": torch.cat(merge_scalars_list, dim=0),
            "merge_states": torch.cat(merge_states_list, dim=0),
            "merge_state_targets": torch.cat(merge_state_targets_list, dim=0),
        }
        return root_state, root_scalar, forward_aux


def _leaf_count_batches(
    items: Sequence[TreeExample],
    batch_size: int,
) -> list[list[TreeExample]]:
    """Return minibatches with equal observed leaf counts.

    HLL leaf-size runs have variable document lengths, so a fixed token budget
    can still yield different leaf counts. Local-law tensors concatenate along
    the leaf axis; bucketing keeps those tensors rectangular while preserving
    large minibatches.
    """
    size = max(1, int(batch_size))
    buckets: dict[int, list[TreeExample]] = {}
    order: list[int] = []
    for item in items:
        key = int(len(item.leaves))
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(item)
    out: list[list[TreeExample]] = []
    for key in order:
        bucket = buckets[key]
        for start in range(0, len(bucket), size):
            out.append(list(bucket[start : start + size]))
    return out


# ---------------------------------------------------------------------------
# Objective.
# ---------------------------------------------------------------------------


@dataclass
class LearnedHLLParityObjective:
    """`TreeObjective` for the learned-HLL parity workload.

    Mirrors the canonical formula used by `MarkovChangepointObjective` and
    `MergeableSketchObjective`:

        loss = (1 - λ) · root_mse
             + λ · [ρ_C1 · leaf_loss + ρ_C2 · c2_loss + ρ_C3 · c3_loss]
                   / (ρ_C1 + ρ_C2 + ρ_C3)

    with `λ = local_law_weight` (default 0.3) and the three ρ's equal by
    default (uniform local-law mass).

    **C2 is structurally absent in this workload**: the oracle emits scalar
    per-node HLL-reference targets, not analytic merge-state vectors, so
    there is no reconstruction term to fold in. `c2_relative_weight` is
    accepted for API compatibility with Markov / mergeable-sketch objectives
    but contributes zero to the local-law block.
    """

    local_law_weight: float = 0.3
    c1_relative_weight: float = 1.0
    c2_relative_weight: float = 1.0  # unused in this workload; kept for parity
    c3_relative_weight: float = 1.0
    merge_state_relative_weight: float = 1.0
    primary_metric_key: str = "val_mae"
    root_query_rate: float = 1.0
    leaf_query_rate: float = 1.0
    internal_query_rate: float = 1.0
    supervision_sampling_policy: str = "separate_axes"

    def _stack_1d(self, items: Sequence[TreeExample], key: str) -> torch.Tensor:
        return torch.tensor(
            [list(item.extra[key]) for item in items],
            dtype=torch.float64,
        )

    def _stack_targets(self, items: Sequence[TreeExample]) -> torch.Tensor:
        return torch.tensor(
            [float(item.target) for item in items],
            dtype=torch.float64,
        )

    def compute_loss(
        self,
        *,
        root_state: torch.Tensor,
        prediction: torch.Tensor,
        batch: Sequence[TreeExample],
        forward_aux: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, int, Mapping[str, Any]]:
        device = prediction.device
        targets = self._stack_targets(batch).to(device=device, dtype=prediction.dtype)
        root_loss = sampled_batch_mse(
            prediction,
            targets,
            rate=float(self.root_query_rate),
        )

        if forward_aux is None:
            return root_loss, int(len(batch)), {
                "root_loss": float(root_loss.detach()),
                "local_law_weight": self.local_law_weight,
                "root_query_rate": float(self.root_query_rate),
            }

        leaf_targets = self._stack_1d(batch, "leaf_cardinalities").to(
            device=device,
            dtype=forward_aux["leaf_scalars"].dtype,
        )
        c1_loss = sampled_axis_mse(
            forward_aux["leaf_scalars"],
            leaf_targets,
            rate=float(self.leaf_query_rate),
        )

        merge_scalars = forward_aux["merge_scalars"]
        merge_state_loss = torch.zeros((), dtype=root_loss.dtype, device=device)
        if merge_scalars.numel() > 0:
            merge_targets = self._stack_1d(batch, "cumulative_cardinalities").to(
                device=device,
                dtype=merge_scalars.dtype,
            )
            c3_scalar_loss = sampled_axis_mse(
                merge_scalars,
                merge_targets,
                rate=float(self.internal_query_rate),
            )
            if "merge_states" in forward_aux and "merge_state_targets" in forward_aux:
                merge_state_rows = (
                    forward_aux["merge_states"] - forward_aux["merge_state_targets"].detach()
                ) ** 2
                merge_state_rows = merge_state_rows.reshape(
                    int(merge_state_rows.shape[0]),
                    int(merge_state_rows.shape[1]),
                    -1,
                ).mean(dim=-1)
                merge_state_loss = sampled_uniform_node_ipw_mean_loss(
                    merge_state_rows,
                    rate=float(self.internal_query_rate),
                ).to(device)
            c3_loss = c3_scalar_loss + float(self.merge_state_relative_weight) * merge_state_loss
        else:
            c3_loss = torch.zeros((), dtype=root_loss.dtype, device=device)

        if str(self.supervision_sampling_policy) == "uniform_all_nodes":
            node_rate = (
                float(self.root_query_rate)
                if abs(float(self.root_query_rate) - float(self.leaf_query_rate)) <= 1e-12
                and abs(float(self.root_query_rate) - float(self.internal_query_rate)) <= 1e-12
                else None
            )
            if node_rate is None:
                raise ValueError(
                    "uniform_all_nodes requires equal root, leaf, and internal query rates"
                )
            # ``cumulative_cardinalities`` includes the final root merge. The
            # uniform node pool adds root explicitly, so keep only non-root internals.
            internal_pred = merge_scalars[:, :-1] if merge_scalars.ndim >= 2 else merge_scalars[:0]
            internal_target = (
                merge_targets[:, :-1]
                if merge_scalars.numel() > 0 and merge_targets.ndim >= 2
                else None
            )
            node_width = 1 + int(leaf_targets.shape[1]) + int(internal_pred.shape[1])
            node_mask = persistent_uniform_node_mask(
                list(batch),
                width=int(node_width),
                rate=float(node_rate),
                device=device,
            )
            scalar_node_loss = sampled_tree_node_mse(
                root_pred=prediction,
                root_target=targets,
                leaf_pred=forward_aux["leaf_scalars"],
                leaf_target=leaf_targets,
                internal_pred=internal_pred,
                internal_target=internal_target,
                rate=float(node_rate),
                node_mask=node_mask,
                node_propensity=float(node_rate),
            )
            persistent_merge_state_loss = torch.zeros((), dtype=root_loss.dtype, device=device)
            if (
                "merge_states" in forward_aux
                and "merge_state_targets" in forward_aux
                and internal_pred.numel() > 0
            ):
                merge_state_rows = (
                    forward_aux["merge_states"][:, :-1, :]
                    - forward_aux["merge_state_targets"][:, :-1, :].detach()
                ) ** 2
                merge_state_rows = merge_state_rows.reshape(
                    int(merge_state_rows.shape[0]),
                    int(merge_state_rows.shape[1]),
                    -1,
                ).mean(dim=-1)
                internal_mask = node_mask[:, 1 + int(leaf_targets.shape[1]) :]
                persistent_merge_state_loss = observed_uniform_node_ipw_mean_loss(
                    merge_state_rows,
                    observed=internal_mask,
                    propensity=float(node_rate),
                ).to(device)
            total = scalar_node_loss + float(self.merge_state_relative_weight) * persistent_merge_state_loss
            return total, int(len(batch)), {
                "root_loss": float(root_loss.detach()),
                "leaf_loss": float(c1_loss.detach()),
                "c1_loss": float(c1_loss.detach()),
                "c2_loss": 0.0,
                "c3_loss": float(c3_loss.detach()),
                "merge_state_loss": float(persistent_merge_state_loss.detach()),
                "local_block": float(total.detach()),
                "uniform_node_loss": float(scalar_node_loss.detach()),
                "local_law_weight": 1.0,
                "root_query_rate": float(self.root_query_rate),
                "leaf_query_rate": float(self.leaf_query_rate),
                "internal_query_rate": float(self.internal_query_rate),
                "node_query_rate": float(node_rate),
                "supervision_sampling_policy": str(self.supervision_sampling_policy),
            }

        # Canonical λ balance — matches MarkovChangepointObjective. C2 is
        # absent in this workload; ρ_C2 sums in but its loss term is zero.
        rho = {
            "c1": max(0.0, float(self.c1_relative_weight)),
            "c2": max(0.0, float(self.c2_relative_weight)),
            "c3": max(0.0, float(self.c3_relative_weight)),
        }
        rho_total = rho["c1"] + rho["c2"] + rho["c3"]
        lam = max(0.0, min(1.0, float(self.local_law_weight)))
        has_root_supervision = float(self.root_query_rate) > 0.0
        has_local_supervision = (
            float(self.leaf_query_rate) > 0.0 or float(self.internal_query_rate) > 0.0
        )
        if not has_root_supervision and has_local_supervision:
            lam = 1.0
        elif has_root_supervision and not has_local_supervision:
            lam = 0.0
        if rho_total <= 0.0 or not has_local_supervision:
            total = root_loss
            local_scalar = 0.0
        else:
            c2_loss = torch.zeros((), dtype=root_loss.dtype, device=device)
            local = (
                rho["c1"] * c1_loss + rho["c2"] * c2_loss + rho["c3"] * c3_loss
            ) / rho_total
            total = (1.0 - lam) * root_loss + lam * local
            local_scalar = float(local.detach())
        return total, int(len(batch)), {
            "root_loss": float(root_loss.detach()),
            "leaf_loss": float(c1_loss.detach()),
            "c1_loss": float(c1_loss.detach()),
            "c2_loss": 0.0,
            "c3_loss": float(c3_loss.detach()),
            "merge_state_loss": float(merge_state_loss.detach()),
            "local_block": float(local_scalar),
            "local_law_weight": float(lam),
            "root_query_rate": float(self.root_query_rate),
            "leaf_query_rate": float(self.leaf_query_rate),
            "internal_query_rate": float(self.internal_query_rate),
        }

    def evaluate(
        self,
        *,
        model: nn.Module,
        items: Sequence[TreeExample],
        batch_size: int,
    ) -> Mapping[str, Any]:
        model.eval()
        preds_chunks: list[torch.Tensor] = []
        targets_chunks: list[torch.Tensor] = []
        c1_chunks: list[torch.Tensor] = []
        c3_chunks: list[torch.Tensor] = []
        merge_state_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in _leaf_count_batches(list(items), int(batch_size)):
                _root_state, prediction, forward_aux = model.forward_tree(batch)
                preds_chunks.append(prediction.detach().cpu())
                targets_chunks.append(self._stack_targets(batch))
                if forward_aux is not None:
                    leaf_targets = self._stack_1d(batch, "leaf_cardinalities")
                    c1_chunks.append(
                        (forward_aux["leaf_scalars"].detach().cpu() - leaf_targets)
                        .abs()
                        .mean(dim=-1)
                    )
                    if forward_aux["merge_scalars"].numel() > 0:
                        cm = self._stack_1d(batch, "cumulative_cardinalities")
                        c3_chunks.append(
                            (forward_aux["merge_scalars"].detach().cpu() - cm)
                            .abs()
                            .mean(dim=-1)
                        )
                    if (
                        "merge_states" in forward_aux
                        and "merge_state_targets" in forward_aux
                        and forward_aux["merge_states"].numel() > 0
                    ):
                        merge_state_chunks.append(
                            (
                                forward_aux["merge_states"].detach().cpu()
                                - forward_aux["merge_state_targets"].detach().cpu()
                            )
                            .abs()
                            .mean(dim=(-1, -2))
                        )
        preds = torch.cat(preds_chunks) if preds_chunks else torch.zeros(0)
        tgts = torch.cat(targets_chunks) if targets_chunks else torch.zeros(0)
        mae = float((preds - tgts).abs().mean()) if preds.numel() > 0 else 0.0
        rmse = (
            float(torch.sqrt(((preds - tgts) ** 2).mean()))
            if preds.numel() > 0
            else 0.0
        )
        tgt_scale = float(tgts.abs().clamp_min(1.0).mean()) if tgts.numel() > 0 else 1.0
        rel_mae = mae / max(1.0, tgt_scale)
        # Canonical keys consumed by `pytorch_tree_trainer` / pytorch_loop:
        # `mae_raw` is the primary scalar (best_metric_key default); the loop
        # prefixes it with "val_" before surfacing as `val_mae_raw`.
        # `mae_normalized` lives in the same schema. Other keys are passed
        # through into the per-epoch history block.
        out: dict[str, Any] = {
            "count": int(len(items)),
            "mae_raw": mae,
            "mae_normalized": rel_mae,
            "val_mae": mae,  # convenience alias for non-pytorch-loop consumers
            "root_mae": mae,
            "root_rmse": rmse,
            "root_rel_mae": rel_mae,
        }
        if c1_chunks:
            c1_mae = float(torch.cat(c1_chunks).mean())
            out["c1_mae"] = c1_mae
            out["val_leaf_loss"] = c1_mae  # Markov-canonical name
        if c3_chunks:
            c3_mae = float(torch.cat(c3_chunks).mean())
            out["c3_mae"] = c3_mae
            out["val_c3_loss"] = c3_mae
        if merge_state_chunks:
            merge_state_mae = float(torch.cat(merge_state_chunks).mean())
            out["merge_state_mae"] = merge_state_mae
            out["val_merge_state_loss"] = merge_state_mae
        return out


# ---------------------------------------------------------------------------
# Train/val split oracle (shares data generator with the classical oracle).
# ---------------------------------------------------------------------------


class LearnedHLLParityOracle(ClassicalHLLParityOracle):
    """Extends the classical oracle to expose both `train_examples` and
    `val_examples` by splitting the generated Zipfian documents."""

    def __init__(
        self,
        *,
        config: ClassicalHLLParityConfig,
        n_train: int,
        n_val: int,
        target_fn: TargetFn | None = None,
        cache_registers: bool = False,
    ) -> None:
        # Oracle generates n_val = n_train + n_val docs total (we ignore the
        # config's own n_val and split below). Store the split sizes.
        super().__init__(config=config, target_fn=target_fn)
        self._n_train = int(n_train)
        self._n_val = int(n_val)
        self._cache_registers = bool(cache_registers)
        self._cached: list[TreeExample] | None = None

    def _to_tree_example(
        self,
        leaves: tuple[tuple[int, ...], ...],
        _analytic_truth: float,
        flat_tokens: list[int],
    ) -> TreeExample:
        if not self._cache_registers:
            return super()._to_tree_example(leaves, _analytic_truth, flat_tokens)
        leaf_registers = tuple(
            _native_hll_registers_uint8(leaf, precision=int(self.config.precision))
            for leaf in leaves
        )
        leaf_values = [
            _hll_estimate_np(reg, precision=int(self.config.precision))
            for reg in leaf_registers
        ]
        cumulative_registers: list[np.ndarray] = []
        cumulative_values: list[float] = []
        root_registers = np.zeros(1 << int(self.config.precision), dtype=np.uint8)
        if leaf_registers:
            state = leaf_registers[0].copy()
            root_registers = state.copy()
            for reg in leaf_registers[1:]:
                state = np.maximum(state, reg).astype(np.uint8, copy=False)
                cumulative_registers.append(state.copy())
                cumulative_values.append(_hll_estimate_np(state, precision=int(self.config.precision)))
            root_registers = state
        root_target = _hll_estimate_np(root_registers, precision=int(self.config.precision))
        analytic_root = float(len(set(flat_tokens)))
        extra = {
            "flat_tokens": list(flat_tokens),
            "leaf_cardinalities": leaf_values,
            "cumulative_cardinalities": cumulative_values,
            "analytic_root_cardinality": analytic_root,
            "oracle_kind": str(self.config.oracle_kind),
        }
        extra["hll_leaf_registers"] = leaf_registers
        extra["hll_cumulative_registers"] = tuple(cumulative_registers)
        return TreeExample(leaves=leaves, target=root_target, extra=extra)

    def _all_examples(self) -> list[TreeExample]:
        if self._cached is None:
            raw = generate_documents(self.config)
            self._cached = [self._to_tree_example(*item) for item in raw]
            self._cached = attach_persistent_uniform_node_scores(
                self._cached,
                seed=int(self.config.seed),
            )
        return self._cached

    def train_examples(self) -> Sequence[TreeExample]:
        return self._all_examples()[: self._n_train]

    def val_examples(self) -> Sequence[TreeExample]:
        return self._all_examples()[self._n_train : self._n_train + self._n_val]

    def metadata(self) -> Mapping[str, Any]:
        base = dict(super().metadata())
        base["oracle"] = "learned_hll_parity"
        base["n_train"] = int(self._n_train)
        base["n_val_learned"] = int(self._n_val)
        return base


# ---------------------------------------------------------------------------
# Preset.
# ---------------------------------------------------------------------------


def learned_hll_parity_task(
    *,
    method: MethodKind = "learned_joint",
    precision: int = 11,
    n_leaves: int | None = 4,
    leaf_size: int | None = None,
    schedule: ScheduleName = "balanced",
    backend: BackendName = "native",
    oracle_kind: OracleKind = "hll_reference",
    n_train: int = 128,
    n_val: int = 48,
    seed: int = 0,
    universe_size: int = 10_000,
    min_tokens: int = 128,
    max_tokens: int = 512,
    zipf_alphas: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4),
    embedding_dim: int | None = None,
    summary_dim: int | None = None,
    state_dim: int | None = None,
    hidden_dim: int | None = None,
    readout_arch: Literal["structured", "mlp"] = "structured",
    use_learned_readout: bool | None = None,
    n_epochs: int = 20,
    train_batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    local_law_weight: float = 0.3,
    c1_relative_weight: float = 1.0,
    c2_relative_weight: float = 1.0,
    c3_relative_weight: float = 1.0,
    merge_state_relative_weight: float = 1.0,
    root_query_rate: float = 1.0,
    leaf_query_rate: float = 1.0,
    internal_query_rate: float = 1.0,
    supervision_sampling_policy: str = "separate_axes",
    best_metric_key: str = "mae_raw",
    use_cuda: bool = False,
    cuda_device: int | None = None,
    eval_every_n_epochs: int = 1,
    evaluate_train_on_eval: bool = True,
    init_from: str | Path | None = None,
) -> TrainerConfig:
    """Build a `TrainerConfig` that trains a learned merge against per-node HLL targets.

    Mirrors `markov_changepoint_task` / `mergeable_sketch_task` / the
    embedding-FNO presets: it returns a fully-populated `TrainerConfig`
    consumable by `fit()`, with an `oracle` / `model` / `objective` triple that
    implements the `TreeOracle` / `TreeModel` / `TreeObjective` Protocols.

    `method` selects between the learned variants:
    - `"learned_g"`: merge operator is learned; the readout is the classical HLL
      estimator formula applied to the state (which has dim = 2^precision,
      interpreted as an HLL register vector).
    - `"learned_joint"`: both the merge operator and the readout are learned MLPs.
      Uses arbitrary state_dim subject to the repo-wide 2× FNO-head invariant.
      `"learned_fg"` remains accepted as a legacy alias.
    - `"learned_g_oracle_state"`: leaf states are fixed native-HLL registers,
      the readout is the classical HLL estimator, and only the pairwise merge
      map is learned. This method requires `backend="native"`.

    Default `oracle_kind="hll_reference"` points training at the classical HLL's
    own scoring head so the learned g can recover classical behavior.

    `local_law_weight` (λ, default 0.3) + `c{1,2,3}_relative_weight` (ρ's,
    default 1.0) match the canonical λ/ρ knobs used across the framework.
    C2 is structurally absent in this workload (see `LearnedHLLParityObjective`).
    """
    canonical_method = "learned_joint" if method == "learned_fg" else str(method)
    if canonical_method not in {"learned_g", "learned_joint", "learned_g_oracle_state", "learned_f_oracle_state"}:
        raise ValueError(
            "method must be one of 'learned_g', 'learned_joint', "
            f"'learned_fg', 'learned_g_oracle_state', or 'learned_f_oracle_state'; got {method!r}"
        )
    if canonical_method in {"learned_g_oracle_state", "learned_f_oracle_state"} and backend != "native":
        raise ValueError(
            f"{canonical_method} requires backend='native' because "
            "DataSketches does not expose HLL registers"
        )
    if canonical_method in {"learned_g_oracle_state", "learned_f_oracle_state"} and oracle_kind != "hll_reference":
        raise ValueError(
            f"{canonical_method} uses supplied HLL register state, so it "
            "requires oracle_kind='hll_reference'"
        )
    learn_readout = (canonical_method == "learned_joint")
    leaf_width_floor = _leaf_width_floor(
        max_tokens=int(max_tokens),
        n_leaves=int(n_leaves) if n_leaves is not None else None,
        leaf_size=int(leaf_size) if leaf_size is not None else None,
    )
    context = "learned_hll_parity"
    if canonical_method in {"learned_g_oracle_state", "learned_f_oracle_state"}:
        resolved_embedding_dim = 0
        resolved_summary_dim = 0
        oracle_state_dim = int(1 << int(precision))
        resolved_hidden_dim = promote_dim(
            name="hidden_dim",
            requested=hidden_dim,
            default=max(128, 2 * oracle_state_dim),
            minimum=2 * oracle_state_dim,
            context=context,
            reason="oracle-state decoder width must cover 2*state_dim",
        )
    else:
        resolved_embedding_dim = promote_dim(
            name="embedding_dim",
            requested=embedding_dim,
            default=int(leaf_width_floor),
            minimum=int(leaf_width_floor),
            context=context,
            reason="token embedding width must cover the 2x leaf-token floor",
        )
        resolved_summary_dim = promote_dim(
            name="summary_dim",
            requested=summary_dim,
            default=int(resolved_embedding_dim),
            minimum=int(leaf_width_floor),
            context=context,
            reason="summary width must cover the 2x leaf-token floor",
        )
        resolved_state_dim = promote_dim(
            name="state_dim",
            requested=state_dim,
            default=2 * int(resolved_summary_dim),
            minimum=2 * int(resolved_summary_dim),
            context=context,
            reason="state width must satisfy the 2x FNO head invariant relative to summary_dim",
        )
        hidden_floor = max(128, int(resolved_state_dim), 2 * int(resolved_state_dim))
        resolved_hidden_dim = promote_dim(
            name="hidden_dim",
            requested=hidden_dim,
            default=int(hidden_floor),
            minimum=int(hidden_floor),
            context=context,
            reason="f hidden width must cover state_dim and g hidden width must cover 2*state_dim",
        )
    total = int(n_train + n_val)
    cfg = ClassicalHLLParityConfig(
        precision=int(precision),
        n_leaves=int(n_leaves) if n_leaves is not None else None,
        leaf_size=int(leaf_size) if leaf_size is not None else None,
        schedule=schedule,
        backend=backend,
        n_val=total,  # oracle generates train+val combined; we split below
        seed=int(seed),
        universe_size=int(universe_size),
        min_tokens=int(min_tokens),
        max_tokens=int(max_tokens),
        zipf_alphas=tuple(float(a) for a in zipf_alphas),
        oracle_kind=oracle_kind,
    )
    oracle = LearnedHLLParityOracle(
        config=cfg,
        n_train=int(n_train),
        n_val=int(n_val),
        cache_registers=canonical_method in {"learned_g_oracle_state", "learned_f_oracle_state"},
    )
    target_scale = float(max_tokens) * 1.5  # soft upper bound on root cardinality
    if canonical_method in {"learned_g_oracle_state", "learned_f_oracle_state"}:
        exact_state_learn_readout = (
            canonical_method == "learned_f_oracle_state"
            if use_learned_readout is None
            else bool(use_learned_readout)
        )
        model = OracleStateHLLMergeModel(
            precision=int(precision),
            hidden_dim=int(resolved_hidden_dim),
            variant="f" if canonical_method == "learned_f_oracle_state" else "g",
            readout_arch=readout_arch,
            use_learned_readout=exact_state_learn_readout,
            init_from=init_from,
        )
    else:
        model = LearnedHLLMergeModel(
            vocab_size=int(universe_size),
            precision=int(precision),
            embedding_dim=int(resolved_embedding_dim),
            summary_dim=int(resolved_summary_dim),
            hidden_dim=int(resolved_hidden_dim),
            state_dim=int(resolved_state_dim),
            target_scale=target_scale,
            learn_readout=bool(learn_readout),
        )
    objective = LearnedHLLParityObjective(
        local_law_weight=float(local_law_weight),
        c1_relative_weight=float(c1_relative_weight),
        c2_relative_weight=float(c2_relative_weight),
        c3_relative_weight=float(c3_relative_weight),
        merge_state_relative_weight=float(merge_state_relative_weight),
        primary_metric_key=str(best_metric_key),
        root_query_rate=float(root_query_rate),
        leaf_query_rate=float(leaf_query_rate),
        internal_query_rate=float(internal_query_rate),
        supervision_sampling_policy=str(supervision_sampling_policy),
    )
    return TrainerConfig(
        oracle=oracle,
        model=model,
        objective=objective,
        n_epochs=int(n_epochs),
        train_batch_size=int(train_batch_size),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        seed=int(seed),
        best_metric_key=str(best_metric_key),
        use_cuda=bool(use_cuda),
        cuda_device=int(cuda_device) if cuda_device is not None else None,
        extra={
            "method": canonical_method,
            "requested_method": str(method),
            "learn_readout": bool(
                getattr(model, "learn_readout", False)
                if canonical_method in {"learned_g_oracle_state", "learned_f_oracle_state"}
                else learn_readout
            ),
            "fixed_oracle_state": canonical_method in {"learned_g_oracle_state", "learned_f_oracle_state"},
            "fixed_oracle_readout": not bool(getattr(model, "learn_readout", False)),
            "init_from": str(init_from) if init_from is not None else None,
            "readout_arch": (
                str(readout_arch)
                if canonical_method in {"learned_g_oracle_state", "learned_f_oracle_state"}
                else ("mlp" if bool(learn_readout) else "structured")
            ),
            "embedding_dim": int(resolved_embedding_dim),
            "summary_dim": int(resolved_summary_dim),
            "state_dim": int(getattr(model, "state_dim", 0)),
            "hidden_dim": int(resolved_hidden_dim),
            "leaf_width_floor": int(leaf_width_floor),
            "leaf_pooling": (
                "oracle_state"
                if canonical_method in {"learned_g_oracle_state", "learned_f_oracle_state"}
                else "mean_max"
            ),
            "batch_key": "leaf_count" if n_leaves is None else "",
            "eval_every_n_epochs": int(eval_every_n_epochs),
            "evaluate_train_on_eval": bool(evaluate_train_on_eval),
            "leaf_query_rate": float(leaf_query_rate),
            "internal_query_rate": float(internal_query_rate),
        },
    )


__all__ = [
    "LearnedHLLMergeModel",
    "LearnedHLLParityObjective",
    "LearnedHLLParityOracle",
    "MethodKind",
    "OracleStateHLLMergeModel",
    "hll_estimate_differentiable",
    "learned_hll_parity_task",
]
