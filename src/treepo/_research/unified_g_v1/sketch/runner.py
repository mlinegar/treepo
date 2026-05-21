"""Synthetic mergeable-sketch training runner.

Mirrors `markov/runner.py` at a minimal scale. Trains a learned MLP merge
operator to approximate the analytic bigram-sketch merge, with a linear
head that regresses a scalar target (count of a fixed bigram) from the
root sketch.

Training loop is delegated to the shared `run_pytorch_training` helper so
sketch inherits best-checkpoint tracking, periodic train_state snapshots,
and resume support for free. The sketch-specific bits — batch packing,
dual merge+scalar loss, and merge-reconstruction MSE — live in
`_SketchSupervisionAdapter` and `_sketch_evaluate` below.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from treepo._research.unified_g_v1.core.manifest import now_iso, write_json
from treepo._research.unified_g_v1.sketch.baselines import (
    HyperLogLogSketch,
    bigrams_of,
    true_distinct_bigrams,
)
from treepo._research.unified_g_v1.sketch.sketch_data import (
    BigramSketch,
    SketchSyntheticConfig,
    SketchTreeExample,
    flat_sketch_dim,
    generate_sketch_dataset,
)
from treepo._research.unified_g_v1.training.backends.pytorch_loop import (
    PyTorchLoopConfig,
    run_pytorch_training,
)


@dataclass(frozen=True)
class SketchRunSpec:
    """Minimal descriptor for a mergeable-sketch training run.

    Parallel to `MarkovRunSpec`. Keep small; add fields as the setting grows.
    """

    vocab_size: int = 8
    seq_length: int = 32
    n_leaves: int = 4
    train_docs: int = 256
    val_docs: int = 64
    n_epochs: int = 4
    train_batch_size: int = 32
    learning_rate: float = 1e-2
    seed: int = 0
    target_bigram: tuple[int, int] = (0, 1)

    @property
    def run_key(self) -> str:
        a, b = self.target_bigram
        return (
            f"sketch_v{self.vocab_size}_L{self.seq_length}_k{self.n_leaves}"
            f"_n{self.train_docs}_e{self.n_epochs}_s{self.seed}_tb{a}{b}"
        )

    def to_synthetic_config(self) -> SketchSyntheticConfig:
        return SketchSyntheticConfig(
            vocab_size=self.vocab_size,
            seq_length=self.seq_length,
            n_leaves=self.n_leaves,
            train_docs=self.train_docs,
            val_docs=self.val_docs,
            seed=self.seed,
            target_bigram=self.target_bigram,
        )


@dataclass(frozen=True)
class SketchRunRecord:
    spec: SketchRunSpec
    run_dir: Path
    summary_path: Path
    history: list[dict[str, Any]]
    final_train_mae: float
    final_val_mae: float
    merge_recon_mse: float
    baselines: Mapping[str, Any] = field(default_factory=dict)
    program_contract: Mapping[str, Any] = field(default_factory=dict)


class _LearnedMergeModel(nn.Module):
    """Takes two flat leaf sketches, produces a parent sketch and scalar prediction.

    `merge(a, b) -> parent` is an MLP over the concatenated sketch tensors.
    `predict(root) -> scalar` is a linear head from the root sketch.
    The parent sketch has the same flat dimension as the leaf sketch, so we
    can cascade the merge up a balanced binary tree.
    """

    def __init__(self, *, sketch_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.sketch_dim = int(sketch_dim)
        self.merge = nn.Sequential(
            nn.Linear(2 * sketch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sketch_dim),
        )
        self.head = nn.Linear(sketch_dim, 1)

    def forward_tree(self, leaves: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold leaves left-to-right; return (root_sketch, scalar_prediction).

        leaves shape: (batch, n_leaves, sketch_dim).
        """
        batch, n_leaves, _ = leaves.shape
        state = leaves[:, 0, :]
        for idx in range(1, n_leaves):
            paired = torch.cat([state, leaves[:, idx, :]], dim=-1)
            state = self.merge(paired)
        scalar = self.head(state).reshape(batch)
        return state, scalar


def _stack_leaves(items: list[SketchTreeExample], *, vocab_size: int) -> torch.Tensor:
    tensors = [
        torch.stack([leaf.as_flat_tensor(vocab_size=vocab_size) for leaf in item.leaves])
        for item in items
    ]
    return torch.stack(tensors, dim=0)


def _stack_roots(items: list[SketchTreeExample], *, vocab_size: int) -> torch.Tensor:
    return torch.stack(
        [item.root.as_flat_tensor(vocab_size=vocab_size) for item in items]
    )


def _stack_targets(items: list[SketchTreeExample]) -> torch.Tensor:
    return torch.tensor([item.target for item in items], dtype=torch.float32)


def _batched(items, batch_size: int):
    size = max(1, int(batch_size))
    for idx in range(0, len(items), size):
        yield list(items[idx:idx + size])


class _SketchSupervisionAdapter:
    """SupervisionAdapter for `run_pytorch_training`.

    Packs a batch of `SketchTreeExample`s into (leaves, analytic_roots, targets)
    tensors, forwards the learned merge model, and returns the dual
    merge-reconstruction + scalar-head MSE loss. Both sub-losses use mean
    reduction; `n_terms=1` is returned so the loop's `loss / n_terms`
    normalization is a no-op and the loss is exactly what the pre-refactor
    sketch loop computed.
    """

    def __init__(self, *, model: "_LearnedMergeModel", vocab_size: int) -> None:
        self.model = model
        self.vocab_size = int(vocab_size)

    def prepare_batch(self, batch):
        leaves = _stack_leaves(list(batch), vocab_size=self.vocab_size)
        roots_analytic = _stack_roots(list(batch), vocab_size=self.vocab_size)
        targets = _stack_targets(list(batch))
        return leaves, roots_analytic, targets

    def compute_supervision_loss(self, prepared):
        leaves, roots_analytic, targets = prepared
        root_pred, scalar_pred = self.model.forward_tree(leaves)
        loss_merge = nn.functional.mse_loss(root_pred, roots_analytic)
        loss_scalar = nn.functional.mse_loss(scalar_pred, targets)
        loss = loss_merge + loss_scalar
        return loss, 1, {
            "merge_loss": float(loss_merge.detach().cpu().item()),
            "scalar_loss": float(loss_scalar.detach().cpu().item()),
        }


def _sketch_evaluate(
    *,
    model: "_LearnedMergeModel",
    items: Sequence[SketchTreeExample],
    batch_size: int,
    vocab_size: int,
) -> dict[str, Any]:
    """Eval callback: scalar MAE + merge-reconstruction MSE over `items`.

    Returns keys expected by `run_pytorch_training`: `mae_raw`, `mae_normalized`
    (both the scalar-head MAE), `count`, `predictions`, plus a pass-through
    `val_merge_recon_mse` so the merge-reconstruction signal flows into history.
    """
    model.eval()
    preds: list[torch.Tensor] = []
    tgts: list[torch.Tensor] = []
    merge_errs: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in _batched(list(items), int(batch_size)):
            leaves = _stack_leaves(batch, vocab_size=int(vocab_size))
            roots_analytic = _stack_roots(batch, vocab_size=int(vocab_size))
            root_pred, scalar_pred = model.forward_tree(leaves)
            preds.append(scalar_pred)
            tgts.append(_stack_targets(batch))
            merge_errs.append(
                nn.functional.mse_loss(root_pred, roots_analytic, reduction="none").mean(dim=-1)
            )
    if not preds:
        return {
            "mae_raw": 0.0,
            "mae_normalized": 0.0,
            "count": 0,
            "predictions": [],
            "val_merge_recon_mse": 0.0,
        }
    preds_vec = torch.cat(preds)
    tgts_vec = torch.cat(tgts)
    mae = float((preds_vec - tgts_vec).abs().mean().item())
    merge_mse = float(torch.cat(merge_errs).mean().item())
    return {
        "mae_raw": mae,
        "mae_normalized": mae,
        "count": int(preds_vec.shape[0]),
        "predictions": preds_vec.tolist(),
        "val_merge_recon_mse": merge_mse,
    }


def run_sketch_spec(
    spec: SketchRunSpec,
    *,
    output_root: str | Path,
    reuse_existing: bool = True,
) -> SketchRunRecord:
    output_root = Path(output_root).expanduser()
    run_dir = output_root / "runs" / spec.run_key
    summary_path = run_dir / "summary.json"
    if reuse_existing and summary_path.exists():
        payload = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
        return SketchRunRecord(
            spec=spec,
            run_dir=run_dir,
            summary_path=summary_path,
            history=list(payload.get("history") or []),
            final_train_mae=float(payload.get("final_train_mae", 0.0)),
            final_val_mae=float(payload.get("final_val_mae", 0.0)),
            merge_recon_mse=float(payload.get("merge_recon_mse", 0.0)),
            baselines=dict(payload.get("baselines") or {}),
            program_contract=dict(payload.get("program_contract") or {}),
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(spec.seed))
    syn_cfg = spec.to_synthetic_config()
    train_items = generate_sketch_dataset(syn_cfg, n_docs=spec.train_docs, seed=spec.seed)
    val_items = generate_sketch_dataset(
        syn_cfg, n_docs=spec.val_docs, seed=spec.seed + 1000
    )

    vocab_size = spec.vocab_size
    sketch_dim = flat_sketch_dim(vocab_size)
    model = _LearnedMergeModel(sketch_dim=sketch_dim, hidden_dim=64)
    optimizer = optim.AdamW(model.parameters(), lr=float(spec.learning_rate))

    adapter = _SketchSupervisionAdapter(model=model, vocab_size=vocab_size)
    evaluate_fn = partial(_sketch_evaluate, vocab_size=vocab_size)

    loop_result = run_pytorch_training(
        model=model,
        optimizer=optimizer,
        train_items=list(train_items),
        val_items=list(val_items),
        supervision_adapter=adapter,
        evaluate_fn=evaluate_fn,
        config=PyTorchLoopConfig(
            n_epochs=int(spec.n_epochs),
            train_batch_size=int(spec.train_batch_size),
            # Sketch loop historically ran without gradient clipping; pick a
            # large cap so behavior is unchanged unless gradients blow up.
            grad_clip_norm=1e9,
            seed=int(spec.seed),
        ),
        output_dir=run_dir,
    )
    history = list(loop_result["history"])

    program_contract = {
        "space_kind": "numeric_sequence",
        "leaf_adapter": "mergeable_window_matrix_adapter",
        "merge_adapter": "mergeable_numeric_merge_adapter",
        "task_head": "mergeable_scalar_head",
        "sketch_dim": int(sketch_dim),
        "vocab_size": int(vocab_size),
        "target_bigram": list(spec.target_bigram),
    }

    baselines = _evaluate_baselines(val_items, vocab_size=vocab_size, target_bigram=spec.target_bigram)

    summary_payload = {
        "generated_at": now_iso(),
        "spec": asdict(spec),
        "history": history,
        # `run_pytorch_training` emits {train,val}_mae_raw; the
        # `val_merge_recon_mse` pass-through key comes from `_sketch_evaluate`.
        "final_train_mae": float(history[-1]["train_mae_raw"]) if history else 0.0,
        "final_val_mae": float(history[-1]["val_mae_raw"]) if history else 0.0,
        "merge_recon_mse": float(history[-1].get("val_merge_recon_mse", 0.0)) if history else 0.0,
        "baselines": baselines,
        "program_contract": program_contract,
    }
    write_json(summary_path, summary_payload)
    return SketchRunRecord(
        spec=spec,
        run_dir=run_dir,
        summary_path=summary_path,
        history=history,
        final_train_mae=float(summary_payload["final_train_mae"]),
        final_val_mae=float(summary_payload["final_val_mae"]),
        merge_recon_mse=float(summary_payload["merge_recon_mse"]),
        baselines=baselines,
        program_contract=program_contract,
    )


def _evaluate_baselines(
    items: list[SketchTreeExample],
    *,
    vocab_size: int,
    target_bigram: tuple[int, int],
) -> dict[str, Any]:
    """Compare the learned sketch against the canonical mergeable baselines.

    * Analytic `BigramSketch`: the exact mergeable sketch; zero error on the
      count-of-fixed-bigram task by construction.
    * HyperLogLog (p=6, 64 registers): field-standard mergeable sketch for
      cardinality; we evaluate it on distinct-bigram count.
    """
    a, b = int(target_bigram[0]), int(target_bigram[1])
    analytic_count_abs_errors: list[float] = []
    hll_card_abs_errors: list[float] = []
    true_card_values: list[float] = []
    for item in items:
        # Analytic baseline on the primary training target.
        analytic_prediction = float(item.root.bigram_counts[a, b].item())
        analytic_count_abs_errors.append(abs(analytic_prediction - item.target))

        # HyperLogLog baseline on a different (but natural) mergeable task:
        # distinct-bigram cardinality. Compute it by merging per-leaf HLLs.
        leaves_hll: list[HyperLogLogSketch] = []
        leaf_len = len(item.tokens) // max(1, len(item.leaves))
        for idx in range(len(item.leaves)):
            leaf_tokens = item.tokens[idx * leaf_len : (idx + 1) * leaf_len]
            leaves_hll.append(
                HyperLogLogSketch.from_bigrams(bigrams_of(leaf_tokens), p=6)
            )
        merged = leaves_hll[0]
        for nxt in leaves_hll[1:]:
            merged = merged.merge(nxt)
        # HLL over leaf-internal bigrams misses cross-boundary bigrams; to
        # compare against a true per-sequence HLL we also add boundary bigrams.
        for idx in range(len(item.leaves) - 1):
            left_tokens = item.tokens[idx * leaf_len : (idx + 1) * leaf_len]
            right_tokens = item.tokens[(idx + 1) * leaf_len : (idx + 2) * leaf_len]
            if left_tokens and right_tokens:
                merged.add(left_tokens[-1], right_tokens[0])
        hll_estimate = merged.estimate()
        true_card = float(true_distinct_bigrams(item.tokens))
        hll_card_abs_errors.append(abs(hll_estimate - true_card))
        true_card_values.append(true_card)

    n = max(1, len(items))
    return {
        "analytic_bigram_sketch": {
            "task": "count_fixed_bigram",
            "mean_absolute_error": sum(analytic_count_abs_errors) / n,
            "note": "exact; zero error by construction (sanity check)",
        },
        "hyperloglog_p6": {
            "task": "distinct_bigram_cardinality",
            "mean_absolute_error": sum(hll_card_abs_errors) / n,
            "mean_true_cardinality": sum(true_card_values) / n,
            "registers": 64,
            "note": "standard mergeable baseline; ~13% theoretical stderr at p=6",
        },
    }
