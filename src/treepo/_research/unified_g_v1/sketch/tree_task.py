"""Mergeable-sketch as a `TreeTask`: reference model + preset.

The oracle and objective live in `training/oracles/` and `training/objectives/`
respectively. This module keeps the sketch-specific model + the
`mergeable_sketch_task` preset constructor.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from treepo._research.unified_g_v1.sketch.baselines import (
    HyperLogLogSketch,
    bigrams_of,
    true_distinct_bigrams,
)
from treepo._research.unified_g_v1.sketch.sketch_data import BigramSketch, SketchSyntheticConfig, flat_sketch_dim
from treepo._research.unified_g_v1.training.objectives import MergeableSketchObjective
from treepo._research.unified_g_v1.training.oracles import MergeableSketchOracle
from treepo._research.unified_g_v1.training.tree_task import TreeExample, TrainerConfig


# ---------------------------------------------------------------------------
# Model: MLP merge + linear head over flat sketch tensors.
# ---------------------------------------------------------------------------


class MergeableSketchModel(nn.Module):
    def __init__(self, *, sketch_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.sketch_dim = int(sketch_dim)
        self.merge = nn.Sequential(
            nn.Linear(2 * sketch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sketch_dim),
        )
        self.head = nn.Linear(sketch_dim, 1)

    def _stack_leaves(self, batch: Sequence[TreeExample]) -> torch.Tensor:
        stacked = [torch.stack(list(item.leaves)) for item in batch]
        return torch.stack(stacked, dim=0)

    def forward_tree(self, batch: Sequence[TreeExample]):
        """Fold leaves left-to-right and expose every intermediate state.

        Returns `(root_state, root_scalar, forward_aux)` where `forward_aux`
        carries per-leaf and per-merge predictions + the intermediate merge
        states so the objective can enforce C1 (per-leaf head), C2
        (merge-reconstruction MSE at every level), and C3 (per-merge head)
        on top of the root supervision.
        """
        leaves = self._stack_leaves(batch)
        batch_size, n_leaves, _ = leaves.shape
        # C1: apply the head to every leaf state BEFORE merging.
        leaf_scalars = self.head(leaves).reshape(batch_size, n_leaves)
        # Fold left-to-right; remember every intermediate merge state so we
        # can apply the head at each internal node (C3) and compare the
        # learned merge to the analytic merge (C2 per-merge).
        state = leaves[:, 0, :]
        merge_states: list[torch.Tensor] = []
        for idx in range(1, n_leaves):
            paired = torch.cat([state, leaves[:, idx, :]], dim=-1)
            state = self.merge(paired)
            merge_states.append(state)
        # C3: scalar head on every merge state. Stack -> (B, n_leaves-1).
        if merge_states:
            merge_states_stack = torch.stack(merge_states, dim=1)
            merge_scalars = self.head(merge_states_stack).reshape(
                batch_size, n_leaves - 1
            )
        else:
            merge_states_stack = torch.zeros(batch_size, 0, self.sketch_dim)
            merge_scalars = torch.zeros(batch_size, 0)
        scalar = self.head(state).reshape(batch_size)
        forward_aux = {
            "leaf_states": leaves,  # (B, n_leaves, sketch_dim)
            "leaf_scalars": leaf_scalars,  # (B, n_leaves)
            "merge_states": merge_states_stack,  # (B, n_leaves-1, sketch_dim)
            "merge_scalars": merge_scalars,  # (B, n_leaves-1)
        }
        return state, scalar, forward_aux


# ---------------------------------------------------------------------------
# Preset constructor: fit(trainer_config=mergeable_sketch_task(...))
# ---------------------------------------------------------------------------


def mergeable_sketch_task(
    *,
    vocab_size: int = 8,
    seq_length: int = 32,
    n_leaves: int = 4,
    train_docs: int = 256,
    val_docs: int = 64,
    target_bigram: tuple[int, int] = (0, 1),
    hidden_dim: int = 64,
    n_epochs: int = 4,
    train_batch_size: int = 32,
    learning_rate: float = 1e-2,
    seed: int = 0,
    local_law_weight: float = 0.3,
    c1_relative_weight: float = 1.0,
    c2_relative_weight: float = 1.0,
    c3_relative_weight: float = 1.0,
) -> TrainerConfig:
    """Build a `TrainerConfig` for the mergeable-bigram-sketch synthetic task.

    `local_law_weight` (λ, default 0.3) balances root vs. local laws;
    `*_relative_weight` knobs balance C1/C2/C3 against each other (default
    equal).
    """
    syn_cfg = SketchSyntheticConfig(
        vocab_size=int(vocab_size),
        seq_length=int(seq_length),
        n_leaves=int(n_leaves),
        train_docs=int(train_docs),
        val_docs=int(val_docs),
        seed=int(seed),
        target_bigram=target_bigram,
    )
    oracle = MergeableSketchOracle(
        config=syn_cfg,
        train_docs=int(train_docs),
        val_docs=int(val_docs),
        seed=int(seed),
    )
    model = MergeableSketchModel(
        sketch_dim=flat_sketch_dim(vocab_size),
        hidden_dim=int(hidden_dim),
    )
    objective = MergeableSketchObjective(
        vocab_size=int(vocab_size),
        target_bigram=target_bigram,
        local_law_weight=float(local_law_weight),
        c1_relative_weight=float(c1_relative_weight),
        c2_relative_weight=float(c2_relative_weight),
        c3_relative_weight=float(c3_relative_weight),
    )
    return TrainerConfig(
        oracle=oracle,
        model=model,
        objective=objective,
        n_epochs=int(n_epochs),
        train_batch_size=int(train_batch_size),
        learning_rate=float(learning_rate),
        seed=int(seed),
        best_metric_key="val_mae",
    )


def evaluate_baselines(oracle: MergeableSketchOracle) -> dict[str, Any]:
    """Run the standard mergeable-sketch baselines on the oracle's val set."""
    items = oracle.val_examples()
    a, b = int(oracle.config.target_bigram[0]), int(oracle.config.target_bigram[1])
    analytic_err: list[float] = []
    hll_err: list[float] = []
    for ex in items:
        analytic_root: BigramSketch = ex.extra["analytic_root"]
        analytic_pred = float(analytic_root.bigram_counts[a, b].item())
        analytic_err.append(abs(analytic_pred - float(ex.target)))

        tokens = list(ex.extra["tokens"])
        hll = HyperLogLogSketch.from_bigrams(bigrams_of(tokens), p=6)
        hll_err.append(abs(hll.estimate() - true_distinct_bigrams(tokens)))
    n = max(1, len(items))
    return {
        "analytic_bigram_sketch": {
            "task": "count_fixed_bigram",
            "mean_absolute_error": sum(analytic_err) / n,
            "note": "exact; zero error by construction",
        },
        "hyperloglog_p6": {
            "task": "distinct_bigram_cardinality",
            "mean_absolute_error": sum(hll_err) / n,
            "registers": 64,
            "note": "standard mergeable baseline",
        },
    }
