#!/usr/bin/env python3
"""Tiny standalone probe: train ``CleanUnifiedNO`` on recoverable_v5_t2048.

This is the smallest possible end-to-end test of the minimal unified-g
neural-operator contract: token embedding as a trivial input adapter,
shared FNO ``g`` at leaves and merges, and FNO ``f`` as the state readout.

Usage:
    ./venv/bin/python scripts/research/probe_clean_unified_no.py \\
        --leaf-tokens 2048 --train-docs 1024 --epochs 30 --batch-size 16

Defaults are intentionally small and regularized. The 34M matched probe
(channels=128, g_n_modes=1024, g_n_layers=4) overfit/oscillated and should
not be the next tuning target unless passed explicitly for diagnosis. The
output JSON + log lands in outputs/clean_unified_no_<ts>/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from treepo._research.ctreepo.sim.core.clean_unified_fg import (  # noqa: E402
    CleanUnifiedNO,
    TreeForwardOutputNO,
    leaf_mse_loss,
    merge_mse_loss,
    root_mse_loss,
)
from treepo._research.ctreepo.sim.core.fno_doc_baselines import (  # noqa: E402
    HAS_NEURAL_OPERATOR,
    _fit_cnn1d_baseline_with_predictions,
    _fit_fno_baseline_with_predictions,
)
from treepo._research.ctreepo.sim.core.contextual_sbijax import (  # noqa: E402
    ContextualQueryProblem,
    MarkovTwoSidedContextProblem,
    build_contextual_query_dataset,
    contextual_sbijax_provenance,
)
from treepo._research.ctreepo.sim.core.full_doc_anchor_diagnostics import (  # noqa: E402
    _load_fno_docs,
    prepare_markov_full_doc_anchor_diagnostics_data,
)
from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (  # noqa: E402
    OPSCountConfig,
    build_markov_changepoint_ops_count_data_bundle,
)
from treepo._research.ctreepo.sim.core.markov_neural_operator_baselines import (  # noqa: E402
    _prepare_fno_count_docs,
)


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _load_split_docs(
    *,
    benchmark: str,
    train_docs: int,
    fixed_leaf_tokens: int,
    seed: int = 0,
):
    """Use the existing data prep machinery to materialize train/val/test."""
    payload = prepare_markov_full_doc_anchor_diagnostics_data(
        benchmark_name=benchmark,
        seeds=(int(seed),),
        train_doc_counts=(int(train_docs),),
        use_cuda=False,
        torch_threads=1,
        config_overrides={"fixed_leaf_tokens": int(fixed_leaf_tokens)},
    )
    prepared = dict(payload["prepared"][0])
    train = list(_load_fno_docs(Path(str(prepared["train_fno_docs_json"]))))[:train_docs]
    val = list(_load_fno_docs(Path(str(prepared["val_fno_docs_json"]))))
    test = list(_load_fno_docs(Path(str(prepared["test_fno_docs_json"]))))
    return train, val, test


def _sticky_mean_segment_length(
    *,
    doc_tokens: int,
    min_segments: int,
    max_segments: int,
) -> int:
    target_segments = 0.5 * (float(min_segments) + float(max_segments))
    target_boundaries = max(1.0, target_segments - 1.0)
    return max(1, int(round(float(max(1, int(doc_tokens) - 1)) / target_boundaries)))


def _load_split_docs_direct(
    *,
    doc_tokens: int,
    train_docs: int,
    eval_docs: int,
    fixed_leaf_tokens: int,
    expected_boundaries: float | None,
    seed: int = 0,
):
    """Generate a sticky recoverable long-doc corpus without a named preset."""
    scale = math.sqrt(float(int(doc_tokens)) / 128.0)
    resolved_expected_boundaries = (
        float(expected_boundaries)
        if expected_boundaries is not None
        else 5.0 * scale
    )
    min_segments = max(2, int(round(2.0 * scale)))
    max_segments = max(min_segments, int(round(6.0 * scale)))
    mean_seg_len = _sticky_mean_segment_length(
        doc_tokens=int(doc_tokens),
        min_segments=int(min_segments),
        max_segments=int(max_segments),
    )
    switch_prob = float(
        max(0.0, resolved_expected_boundaries)
        / max(1.0, float(int(doc_tokens) - 1))
    )
    cfg = OPSCountConfig(
        generator_profile="hazard_topic",
        n_regimes=4,
        vocab_size=16,
        min_tokens=int(doc_tokens),
        max_tokens=int(doc_tokens),
        min_segments=int(min_segments),
        max_segments=int(max_segments),
        min_seg_len=int(mean_seg_len),
        max_seg_len=int(mean_seg_len),
        hazard_switch_prob=float(switch_prob),
        fixed_leaf_tokens=int(fixed_leaf_tokens),
        train_docs=int(train_docs),
        val_docs=int(eval_docs),
        test_docs=int(eval_docs),
        seed=int(seed),
        data_seed=int(seed),
        model_seed=int(seed),
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(cfg)
    train = list(_prepare_fno_count_docs(bundle.train_docs, leaf_tokens=int(fixed_leaf_tokens)))
    val = list(_prepare_fno_count_docs(bundle.val_docs, leaf_tokens=int(fixed_leaf_tokens)))
    test = list(_prepare_fno_count_docs(bundle.test_docs, leaf_tokens=int(fixed_leaf_tokens)))
    return train, val, test, {
        "doc_tokens": int(doc_tokens),
        "expected_boundaries": float(resolved_expected_boundaries),
        "min_segments": int(min_segments),
        "max_segments": int(max_segments),
        "mean_seg_len": int(mean_seg_len),
        "hazard_switch_prob": float(switch_prob),
    }


def _load_split_docs_from_bundle(
    *,
    bundle_path: Path,
    train_docs: int,
    eval_docs: int,
    fixed_leaf_tokens: int,
):
    """Load a saved MarkovOPSDataBundle and adapt it to FNO tree docs."""

    from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import MarkovOPSDataBundle

    def _infer_block_by_token(docs) -> list[int]:
        mapping: dict[int, int] = {}
        for doc in docs:
            for token, regime in zip(doc.tokens, doc.token_regimes, strict=True):
                tok = int(token)
                reg = int(regime)
                prev = mapping.get(tok)
                if prev is not None and int(prev) != reg:
                    raise ValueError(
                        f"token {tok} maps to multiple regimes: {prev} and {reg}"
                    )
                mapping[tok] = reg
        if not mapping:
            return []
        max_token = max(mapping)
        return [int(mapping.get(tok, 0)) for tok in range(max_token + 1)]

    bundle = MarkovOPSDataBundle.load(Path(bundle_path))
    n_train = min(int(train_docs), len(bundle.train_docs))
    n_val = min(int(eval_docs), len(bundle.val_docs))
    n_test = min(int(eval_docs), len(bundle.test_docs))
    train = list(
        _prepare_fno_count_docs(
            bundle.train_docs[:n_train],
            leaf_tokens=int(fixed_leaf_tokens),
        )
    )
    val = list(
        _prepare_fno_count_docs(
            bundle.val_docs[:n_val],
            leaf_tokens=int(fixed_leaf_tokens),
        )
    )
    test = list(
        _prepare_fno_count_docs(
            bundle.test_docs[:n_test],
            leaf_tokens=int(fixed_leaf_tokens),
        )
    )
    metadata = dict(getattr(bundle, "metadata", {}) or {})
    doc_token_values = {
        int(len(getattr(doc, "tokens", ()) or ()))
        for doc in tuple(bundle.train_docs[:1]) + tuple(bundle.val_docs[:1]) + tuple(bundle.test_docs[:1])
    }
    block_by_token = _infer_block_by_token(
        tuple(bundle.train_docs[:n_train])
        + tuple(bundle.val_docs[:n_val])
        + tuple(bundle.test_docs[:n_test])
    )
    n_regimes = (max(block_by_token) + 1) if block_by_token else 0
    metadata.update(
        {
            "data_source": "saved_ops_count_bundle",
            "bundle_path": str(bundle_path),
            "fixed_leaf_tokens": int(fixed_leaf_tokens),
            "train_docs": int(n_train),
            "val_docs": int(n_val),
            "test_docs": int(n_test),
            "doc_tokens": (
                int(next(iter(doc_token_values)))
                if len(doc_token_values) == 1
                else None
            ),
            "block_by_token": block_by_token,
            "n_regimes": int(n_regimes),
            "train_corpus_signature": str(getattr(bundle, "train_corpus_signature", "")),
            "val_corpus_signature": str(getattr(bundle, "val_corpus_signature", "")),
            "test_corpus_signature": str(getattr(bundle, "test_corpus_signature", "")),
        }
    )
    return train, val, test, metadata


def _doc_arrays(docs):
    """Return raw per-doc lists for training (we batch later)."""
    leaf_tokens = [
        [list(map(int, leaf)) for leaf in d.leaf_token_ids] for d in docs
    ]
    leaf_counts = [[float(c) for c in d.leaf_counts] for d in docs]
    merge_counts = [[float(c) for c in d.merge_counts_balanced] for d in docs]
    root_counts = [float(d.root_count) for d in docs]
    return leaf_tokens, leaf_counts, merge_counts, root_counts


def _palette_block_map(*, vocab_size: int, n_regimes: int) -> list[int]:
    """Map token id -> disjoint palette block, matching numpy.array_split."""
    v = int(vocab_size)
    n = int(n_regimes)
    if v <= 0 or n <= 0:
        raise ValueError("vocab_size and n_regimes must be positive")
    block_by_token = [0 for _ in range(v)]
    start = 0
    base, extra = divmod(v, n)
    for regime_id in range(n):
        size = base + (1 if regime_id < extra else 0)
        for token_id in range(start, start + size):
            block_by_token[token_id] = regime_id
        start += size
    return block_by_token


def _palette_block_exact_root_mae(
    docs,
    *,
    vocab_size: int,
    n_regimes: int,
    block_by_token: list[int] | None = None,
) -> dict[str, float | int]:
    """Exact recoverable-DGP witness: count adjacent disjoint-palette changes."""
    preds = _palette_block_exact_predictions(
        docs,
        vocab_size=int(vocab_size),
        n_regimes=int(n_regimes),
        block_by_token=block_by_token,
    )
    truths = _root_truths(docs)
    if truths.size == 0:
        return {"n": 0, "mae": float("nan"), "max_abs_error": float("nan")}
    abs_errors = np.abs(preds - truths)
    return {
        "n": int(truths.size),
        "mae": float(np.mean(abs_errors)),
        "max_abs_error": float(np.max(abs_errors)),
    }


def _palette_block_exact_predictions(
    docs,
    *,
    vocab_size: int,
    n_regimes: int,
    block_by_token: list[int] | None = None,
) -> np.ndarray:
    """Exact recoverable-DGP witness predictions for root counts."""
    if block_by_token is None:
        block_by_token = _palette_block_map(
            vocab_size=int(vocab_size),
            n_regimes=int(n_regimes),
        )
    preds: list[float] = []
    for doc in docs:
        tokens = [int(tok) for leaf in doc.leaf_token_ids for tok in leaf]
        if any(tok < 0 or tok >= int(vocab_size) for tok in tokens):
            raise ValueError("token id outside [0, vocab_size) in exact witness")
        blocks = [block_by_token[tok] for tok in tokens]
        pred = float(sum(1 for left, right in zip(blocks, blocks[1:]) if left != right))
        preds.append(pred)
    return np.asarray(preds, dtype=np.float64)


def _flat_doc_tokens(doc) -> list[int]:
    return [int(tok) for leaf in doc.leaf_token_ids for tok in leaf]


def _exact_count_for_tokens(
    tokens: list[int] | tuple[int, ...],
    *,
    block_by_token: list[int],
) -> float:
    if len(tokens) <= 1:
        return 0.0
    blocks = [int(block_by_token[int(tok)]) for tok in tokens]
    return float(sum(1 for left, right in zip(blocks, blocks[1:]) if left != right))


def _sample_token_fragment(
    tokens: list[int],
    *,
    fragment_len: int,
    rng: np.random.Generator,
) -> list[int]:
    if not tokens:
        raise ValueError("cannot sample a fragment from an empty token sequence")
    length = max(1, min(int(fragment_len), len(tokens)))
    if len(tokens) == length:
        return list(tokens)
    start = int(rng.integers(0, len(tokens) - length + 1))
    return list(tokens[start : start + length])


def _root_truths(docs) -> np.ndarray:
    return np.asarray([float(doc.root_count) for doc in docs], dtype=np.float64)


def _safe_corrcoef(preds: np.ndarray, truths: np.ndarray) -> float:
    if preds.size < 2 or truths.size < 2:
        return float("nan")
    if float(np.std(preds)) <= 0.0 or float(np.std(truths)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(preds.astype(np.float64), truths.astype(np.float64))[0, 1])


def _prediction_diagnostics(
    preds,
    truths,
    *,
    tau: float = 0.5,
) -> dict[str, float | int]:
    pred_arr = np.asarray(preds, dtype=np.float64)
    truth_arr = np.asarray(truths, dtype=np.float64)
    if pred_arr.shape != truth_arr.shape:
        raise ValueError(
            f"prediction/truth shape mismatch: {pred_arr.shape} vs {truth_arr.shape}"
        )
    if truth_arr.size == 0:
        return {
            "n": 0,
            "root_mae": float("nan"),
            "root_mse": float("nan"),
            "root_median_abs_error": float("nan"),
            "root_p95_abs_error": float("nan"),
            "max_abs_error": float("nan"),
            "exact_match_rate": float("nan"),
            "pred_mean": float("nan"),
            "pred_std": float("nan"),
            "pred_min": float("nan"),
            "pred_max": float("nan"),
            "truth_mean": float("nan"),
            "truth_std": float("nan"),
            "truth_min": float("nan"),
            "truth_max": float("nan"),
            "pred_truth_corr": float("nan"),
            "violation_rate": float("nan"),
        }
    abs_err = np.abs(pred_arr - truth_arr)
    sq_err = (pred_arr - truth_arr) ** 2
    return {
        "n": int(truth_arr.size),
        "root_mae": float(np.mean(abs_err)),
        "root_mse": float(np.mean(sq_err)),
        "root_median_abs_error": float(np.median(abs_err)),
        "root_p95_abs_error": float(np.percentile(abs_err, 95.0)),
        "max_abs_error": float(np.max(abs_err)),
        "exact_match_rate": float(
            np.mean((np.rint(pred_arr) == np.rint(truth_arr)).astype(np.float64))
        ),
        "pred_mean": float(np.mean(pred_arr)),
        "pred_std": float(np.std(pred_arr)),
        "pred_min": float(np.min(pred_arr)),
        "pred_max": float(np.max(pred_arr)),
        "truth_mean": float(np.mean(truth_arr)),
        "truth_std": float(np.std(truth_arr)),
        "truth_min": float(np.min(truth_arr)),
        "truth_max": float(np.max(truth_arr)),
        "pred_truth_corr": _safe_corrcoef(pred_arr, truth_arr),
        "violation_rate": float(np.mean((abs_err > float(tau)).astype(np.float64))),
    }


def _constant_baseline_diagnostics(
    *,
    train_docs,
    val_docs,
    test_docs,
) -> dict[str, object]:
    split_docs = {"train": train_docs, "val": val_docs, "test": test_docs}
    split_truths = {name: _root_truths(docs) for name, docs in split_docs.items()}
    train_truth = split_truths["train"]
    train_mean = float(np.mean(train_truth)) if train_truth.size else float("nan")
    train_median = float(np.median(train_truth)) if train_truth.size else float("nan")
    out: dict[str, object] = {
        "train_mean_count": train_mean,
        "train_median_count": train_median,
        "splits": {},
    }
    split_out: dict[str, object] = {}
    for name, truths in split_truths.items():
        split_mean = float(np.mean(truths)) if truths.size else float("nan")
        split_median = float(np.median(truths)) if truths.size else float("nan")
        split_out[name] = {
            "root_count": {
                "n": int(truths.size),
                "mean": split_mean,
                "median": split_median,
                "std": float(np.std(truths)) if truths.size else float("nan"),
                "min": float(np.min(truths)) if truths.size else float("nan"),
                "max": float(np.max(truths)) if truths.size else float("nan"),
            },
            "train_mean_predictor": _prediction_diagnostics(
                np.full_like(truths, fill_value=train_mean, dtype=np.float64),
                truths,
            ),
            "train_median_predictor": _prediction_diagnostics(
                np.full_like(truths, fill_value=train_median, dtype=np.float64),
                truths,
            ),
            "split_mean_predictor": _prediction_diagnostics(
                np.full_like(truths, fill_value=split_mean, dtype=np.float64),
                truths,
            ),
            "split_median_predictor": _prediction_diagnostics(
                np.full_like(truths, fill_value=split_median, dtype=np.float64),
                truths,
            ),
        }
    out["splits"] = split_out
    return out


def _to_tensor(rows, *, dtype, device):
    return torch.tensor(rows, dtype=dtype, device=device)


class _PaletteBlockTokenEmbedding(nn.Module):
    """Deterministic token -> palette-block one-hot function adapter."""

    def __init__(self, *, vocab_size: int, n_regimes: int) -> None:
        super().__init__()
        block_by_token = _palette_block_map(
            vocab_size=int(vocab_size),
            n_regimes=int(n_regimes),
        )
        weight = torch.zeros((int(vocab_size) + 1, int(n_regimes)), dtype=torch.float32)
        for token_id, block_id in enumerate(block_by_token):
            weight[int(token_id), int(block_id)] = 1.0
        self.pad_id = int(vocab_size)
        self.channels = int(n_regimes)
        self.register_buffer("weight", weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be (B, L); got shape {tuple(tokens.shape)}")
        emb = F.embedding(tokens, self.weight)
        return emb.permute(0, 2, 1).contiguous()


class _PaletteBlockIdentityG(nn.Module):
    """Exact surface g: leaf call returns the content state unchanged."""

    def __init__(self, *, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)

    def forward(
        self,
        left_or_pair: torch.Tensor,
        right_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if right_state is not None:
            raise NotImplementedError(
                "exact palette-block surface contract currently checks one-leaf data only"
            )
        if left_or_pair.ndim != 3:
            raise ValueError(f"g input must be (B, C, L); got {tuple(left_or_pair.shape)}")
        n_channels = int(left_or_pair.shape[-2])
        if n_channels == self.channels:
            return left_or_pair
        if n_channels == 2 * self.channels:
            return left_or_pair[:, : self.channels, :]
        raise ValueError(
            f"g input channels must be {self.channels} or {2 * self.channels}; "
            f"got {n_channels}"
        )


class _PaletteBlockTransitionF(nn.Module):
    """Exact f: count adjacent palette-block changes in a function state."""

    def __init__(self, *, target_scale: float) -> None:
        super().__init__()
        self.target_scale = float(target_scale)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3:
            raise ValueError(f"state must be (B, C, L); got {tuple(state.shape)}")
        block_ids = state.argmax(dim=-2)
        if int(block_ids.shape[-1]) <= 1:
            count = block_ids.new_zeros((int(block_ids.shape[0]),), dtype=torch.float32)
        else:
            count = block_ids[:, 1:].ne(block_ids[:, :-1]).to(dtype=torch.float32).sum(dim=-1)
        return count / float(self.target_scale)


class _PaletteBlockExactSurface(nn.Module):
    """Deterministic one-leaf model using the same token_embedding -> g -> f surface."""

    def __init__(
        self,
        *,
        vocab_size: int,
        n_regimes: int,
        target_scale: float,
    ) -> None:
        super().__init__()
        self.target_scale = float(target_scale)
        self.channels = int(n_regimes)
        self.token_embedding = _PaletteBlockTokenEmbedding(
            vocab_size=int(vocab_size),
            n_regimes=int(n_regimes),
        )
        self.g = _PaletteBlockIdentityG(channels=int(n_regimes))
        self.f = _PaletteBlockTransitionF(target_scale=float(target_scale))

    def forward_doc(self, leaf_tokens: torch.Tensor) -> TreeForwardOutputNO:
        if leaf_tokens.ndim != 2:
            raise ValueError(
                f"leaf_tokens must be (n_leaves, L); got {tuple(leaf_tokens.shape)}"
            )
        if int(leaf_tokens.shape[0]) != 1:
            raise ValueError("exact palette-block surface contract is one-leaf only")
        embedded = self.token_embedding(leaf_tokens)
        leaf_states_batch = self.g(embedded)
        leaf_counts_norm = self.f(leaf_states_batch)
        return TreeForwardOutputNO(
            leaf_states=[leaf_states_batch[0]],
            merge_states=[],
            leaf_counts_norm=leaf_counts_norm,
            merge_counts_norm=leaf_counts_norm.new_empty((0,)),
            root_state=leaf_states_batch[0],
            root_count_norm=leaf_counts_norm[0],
        )

    def forward(self, leaf_tokens: torch.Tensor) -> TreeForwardOutputNO:
        return self.forward_doc(leaf_tokens)


def _clone_checkpoint_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone trainable checkpoint tensors without NeuralOp metadata entries."""
    return {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if isinstance(v, torch.Tensor) and k != "_metadata"
    }


def _eval_root_mae(model, docs, *, batch_size: int, device: torch.device) -> float:
    preds, truths = _eval_model_root_predictions(
        model,
        docs,
        batch_size=int(batch_size),
        device=device,
    )
    return float(_prediction_diagnostics(preds, truths)["root_mae"])


def _eval_model_root_predictions(
    model,
    docs,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    leaf_tokens, _, _, root_counts = _doc_arrays(docs)
    n = len(docs)
    preds: list[float] = []
    truths: list[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_tokens = leaf_tokens[i : i + batch_size]
            batch_root = root_counts[i : i + batch_size]
            for j, leaves in enumerate(batch_tokens):
                tok = _to_tensor(leaves, dtype=torch.long, device=device)
                out = model(tok)
                pred_unnorm = out.root_count_norm * model.target_scale
                preds.append(float(pred_unnorm.detach().cpu()))
                truths.append(float(batch_root[j]))
    return (
        np.asarray(preds, dtype=np.float64),
        np.asarray(truths, dtype=np.float64),
    )


def _eval_model_root_diagnostics(
    model,
    docs,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    preds, truths = _eval_model_root_predictions(
        model,
        docs,
        batch_size=int(batch_size),
        device=device,
    )
    return _prediction_diagnostics(preds, truths)


def _encode_flat_fragment_via_g(
    model: CleanUnifiedNO,
    tokens: list[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    tok = torch.tensor([tokens], dtype=torch.long, device=device)
    return model._encode_leaf_states_via_g(tok)


def _compose_three_context_states(
    model: CleanUnifiedNO,
    left_state: torch.Tensor,
    item_state: torch.Tensor,
    right_state: torch.Tensor,
) -> torch.Tensor:
    left_item = model._merge_state_batch_via_g(left_state, item_state)
    return model._merge_state_batch_via_g(left_item, right_state)


def _standardize_batch_columns(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"expected a 2D tensor; got {tuple(x.shape)}")
    return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True).clamp(min=1e-6)


def _random_unit_slices(
    *,
    n_contexts: int,
    n_slices: int,
    device: torch.device,
) -> torch.Tensor:
    raw = torch.randn((int(n_contexts), int(n_slices)), dtype=torch.float32, device=device)
    return raw / raw.norm(dim=0, keepdim=True).clamp(min=1e-6)


class _MeanPoolResponseRegressor(nn.Module):
    def __init__(self, *, channels: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(channels), int(out_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 3:
            state = state.mean(dim=-1)
        return self.linear(state)


class _ConvPoolResponseRegressor(nn.Module):
    def __init__(self, *, channels: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(int(channels), int(hidden_dim), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(int(hidden_dim), int(hidden_dim), kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.linear = nn.Linear(int(hidden_dim), int(out_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3:
            raise ValueError(f"conv_pool response regressor expects (B, C, L); got {tuple(state.shape)}")
        pooled = self.net(state).mean(dim=-1)
        return self.linear(pooled)


class _FlattenResponseRegressor(nn.Module):
    def __init__(self, *, channels: int, length: int, out_dim: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.length = int(length)
        self.linear = nn.Linear(int(channels) * int(length), int(out_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3:
            raise ValueError(f"flatten response regressor expects (B, C, L); got {tuple(state.shape)}")
        if int(state.shape[-2]) != self.channels or int(state.shape[-1]) != self.length:
            raise ValueError(
                "flatten response regressor received incompatible state shape; "
                f"got {tuple(state.shape)}, expected (B, {self.channels}, {self.length})"
            )
        return self.linear(state.reshape(int(state.shape[0]), -1))


def _make_response_regressor(
    *,
    kind: str,
    channels: int,
    length: int,
    out_dim: int,
) -> nn.Module:
    if kind == "mean_linear":
        return _MeanPoolResponseRegressor(channels=int(channels), out_dim=int(out_dim))
    if kind == "conv_pool":
        return _ConvPoolResponseRegressor(
            channels=int(channels),
            hidden_dim=max(int(channels), 16),
            out_dim=int(out_dim),
        )
    if kind == "flatten":
        return _FlattenResponseRegressor(
            channels=int(channels),
            length=int(length),
            out_dim=int(out_dim),
        )
    raise ValueError(f"unknown contextual response regressor: {kind!r}")


def _distance_correlation_loss(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Critic-free dependence loss, adapted from the NASS distance-correlation option."""
    if int(z.shape[0]) < 2:
        return z.new_zeros(())
    z = _standardize_batch_columns(z)
    y = _standardize_batch_columns(y)
    a = torch.cdist(z, z, p=2)
    b = torch.cdist(y, y, p=2)
    a = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
    b = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
    dcov2 = (a * b).mean().clamp(min=0.0)
    dvar_z = (a * a).mean().clamp(min=1e-12)
    dvar_y = (b * b).mean().clamp(min=1e-12)
    dcor2 = dcov2 / torch.sqrt(dvar_z * dvar_y)
    return 1.0 - dcor2.clamp(min=0.0, max=1.0).sqrt()


def _contextual_dependence_loss(
    *,
    item_features: torch.Tensor,
    response_signatures: torch.Tensor,
    objective: str,
    signature_projector: nn.Module | None,
    response_regressor: nn.Module | None,
    response_signature_slices: int,
    response_slice_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Auxiliary loss between learned item states and contextual responses.

    The menu mirrors the useful parts of the NASS/NASSS implementations:
    regression and sliced regression for the 2023 low-dimensional target idea,
    distance correlation as a critic-free dependence objective, and JSD/DV/WD/
    InfoNCE as contrastive infomax variants.
    """
    if int(item_features.shape[0]) < 2:
        return item_features.new_zeros(())
    objective = str(objective)
    z = item_features
    sig = response_signatures

    if objective == "regression":
        if response_regressor is None:
            raise ValueError("response_regressor is required for regression objective")
        n_slices = int(response_signature_slices)
        if n_slices > 0:
            if response_slice_matrix is None:
                phi = _random_unit_slices(
                    n_contexts=int(sig.shape[1]),
                    n_slices=n_slices,
                    device=sig.device,
                )
            else:
                phi = response_slice_matrix.to(device=sig.device, dtype=sig.dtype)
            target = sig @ phi
        else:
            target = sig
        regressor_input = z
        if isinstance(response_regressor, nn.Linear) and z.ndim == 3:
            regressor_input = z.mean(dim=-1)
        pred = response_regressor(regressor_input)
        return F.mse_loss(pred, target)

    if z.ndim == 3:
        z = z.mean(dim=-1)

    if objective == "dcorr":
        return _distance_correlation_loss(z, sig)

    if signature_projector is None:
        raise ValueError(f"signature_projector is required for {objective!r} objective")
    z_emb = F.normalize(z, dim=-1)
    sig_emb = F.normalize(signature_projector(sig), dim=-1)
    m = int(z_emb.shape[0])
    idx_neg = torch.randperm(m, device=z_emb.device)

    if objective == "infonce":
        logits = (z_emb @ sig_emb.T) / 0.1
        labels = torch.arange(int(logits.shape[0]), dtype=torch.long, device=z_emb.device)
        return 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        )

    f_pos = (z_emb * sig_emb).sum(dim=-1) / 0.1
    f_neg = (z_emb * sig_emb[idx_neg]).sum(dim=-1) / 0.1
    if objective == "jsd":
        return F.softplus(-f_pos).mean() + F.softplus(f_neg).mean()
    if objective == "dv":
        return -(f_pos.mean() - torch.logsumexp(f_neg, dim=0) + math.log(float(m)))
    if objective == "wasserstein":
        return -(f_pos.mean() - f_neg.mean())
    raise ValueError(f"unknown contextual dependence objective: {objective!r}")


def _contextual_sufficiency_batch_losses(
    *,
    model: CleanUnifiedNO,
    flat_train_docs: list[list[int]],
    batch_indices: list[int],
    block_by_token: list[int],
    target_scale: float,
    samples_per_doc: int,
    fragment_len: int,
    rng: np.random.Generator,
    device: torch.device,
    signature_projector: nn.Module | None = None,
    response_regressor: nn.Module | None = None,
    response_slice_matrix: torch.Tensor | None = None,
    response_signature_contexts: int = 0,
    response_signature_slices: int = 0,
    dependence_objective: str = "infonce",
    contextual_problem: ContextualQueryProblem | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Losses for sampled finite-context query signatures.

    Generic response-signature preservation always trains the item state
    ``z_x = g(embed(x), null)`` against sampled ``R_K(x)`` signatures when a
    dependence objective is enabled.  If the problem adapter exposes
    ``predict_contextual_response``, the same finite contexts are also enacted
    through the model for contextual MSE.
    """
    zero = torch.zeros((), dtype=torch.float32, device=device)
    n_docs = len(flat_train_docs)
    if n_docs <= 0 or int(samples_per_doc) <= 0 or not batch_indices:
        return zero, zero, 0

    problem = contextual_problem or MarkovTwoSidedContextProblem(
        block_by_token=block_by_token,
        vocab_size=len(block_by_token),
        target_scale=float(target_scale),
    )
    n_contexts = max(1, int(response_signature_contexts))
    selected_sources = [flat_train_docs[int(idx)] for idx in batch_indices]
    dataset = build_contextual_query_dataset(
        selected_sources,
        problem=problem,
        samples_per_source=int(samples_per_doc),
        item_len=int(fragment_len),
        n_contexts=int(n_contexts),
        rng=rng,
        context_sources=flat_train_docs,
    )

    contextual_mse = zero
    n_contextual_queries = 0
    predict_contextual_response = getattr(problem, "predict_contextual_response", None)
    if callable(predict_contextual_response):
        n_items = int(dataset.item_tokens.shape[0])
        k = int(dataset.response_signatures.shape[1])
        item_batch = np.repeat(dataset.item_tokens, k, axis=0)
        context_batch = {
            str(name): np.tile(np.asarray(values), (n_items, 1))
            for name, values in dict(dataset.context_tensors).items()
        }
        preds = predict_contextual_response(model, item_batch, context_batch, device)
        target_array = dataset.response_signatures.reshape(n_items * k, -1)
        if int(target_array.shape[1]) == 1:
            target_array = target_array.reshape(-1)
        target = torch.as_tensor(
            target_array,
            dtype=torch.float32,
            device=device,
        )
        contextual_mse = F.mse_loss(preds.reshape(target.shape), target)
        n_contextual_queries = int(target.numel())

    dependence_loss = zero
    if (
        str(dependence_objective) != "none"
        and int(dataset.item_tokens.shape[0]) >= 2
        and int(dataset.response_signatures.shape[1]) > 0
    ):
        item_tokens = torch.as_tensor(
            dataset.item_tokens,
            dtype=torch.long,
            device=device,
        )
        z_states = model._encode_leaf_states_via_g(item_tokens)
        signature_array = dataset.response_signatures.reshape(
            dataset.response_signatures.shape[0],
            -1,
        )
        sig_tensor = torch.as_tensor(
            signature_array,
            dtype=torch.float32,
            device=device,
        )
        dependence_loss = _contextual_dependence_loss(
            item_features=z_states,
            response_signatures=sig_tensor,
            objective=str(dependence_objective),
            signature_projector=signature_projector,
            response_regressor=response_regressor,
            response_signature_slices=int(response_signature_slices),
            response_slice_matrix=response_slice_matrix,
        )
    return contextual_mse, dependence_loss, int(n_contextual_queries)


def _eval_contextual_sufficiency_diagnostics(
    model: CleanUnifiedNO,
    docs,
    *,
    n_queries: int,
    block_by_token: list[int],
    target_scale: float,
    fragment_len: int,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    if int(n_queries) <= 0:
        return {"status": "skipped", "reason": "no contextual eval queries requested"}
    if not docs:
        return {"status": "skipped", "reason": "no docs"}
    flat_docs = [_flat_doc_tokens(doc) for doc in docs]
    rng = np.random.default_rng(int(seed))
    preds: list[float] = []
    truths: list[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(int(n_queries)):
            left_idx = int(rng.integers(0, len(flat_docs)))
            item_idx = int(rng.integers(0, len(flat_docs)))
            right_idx = int(rng.integers(0, len(flat_docs)))
            left_tokens = _sample_token_fragment(
                flat_docs[left_idx],
                fragment_len=int(fragment_len),
                rng=rng,
            )
            item_tokens = _sample_token_fragment(
                flat_docs[item_idx],
                fragment_len=int(fragment_len),
                rng=rng,
            )
            right_tokens = _sample_token_fragment(
                flat_docs[right_idx],
                fragment_len=int(fragment_len),
                rng=rng,
            )
            z_left = _encode_flat_fragment_via_g(
                model,
                left_tokens,
                device=device,
            )
            z_item = _encode_flat_fragment_via_g(
                model,
                item_tokens,
                device=device,
            )
            z_right = _encode_flat_fragment_via_g(
                model,
                right_tokens,
                device=device,
            )
            z_full = _compose_three_context_states(model, z_left, z_item, z_right)
            pred = model._score_states_via_f(z_full)[0] * float(target_scale)
            truth = _exact_count_for_tokens(
                list(left_tokens) + list(item_tokens) + list(right_tokens),
                block_by_token=block_by_token,
            )
            preds.append(float(pred.detach().cpu()))
            truths.append(float(truth))
    diagnostics = _prediction_diagnostics(preds, truths)
    return {
        "status": "ok",
        "n_queries": int(n_queries),
        "fragment_len": int(fragment_len),
        "problem_id": "markov_changepoint_count",
        "context_kind": "markov_two_sided",
        "query": "f(g(g(g(embed(left),null), g(embed(item),null)), g(embed(right),null)))",
        "diagnostics": diagnostics,
    }


def _eval_exact_surface_contract(
    *,
    docs,
    vocab_size: int,
    n_regimes: int,
    target_scale: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    if not docs:
        return {"status": "skipped", "reason": "no docs"}
    n_leaves = len(docs[0].leaf_token_ids)
    if int(n_leaves) != 1:
        return {
            "status": "skipped",
            "reason": f"requires one leaf per doc; got {int(n_leaves)}",
        }
    model = _PaletteBlockExactSurface(
        vocab_size=int(vocab_size),
        n_regimes=int(n_regimes),
        target_scale=float(target_scale),
    ).to(device=device)
    diagnostics = _eval_model_root_diagnostics(
        model,
        docs,
        batch_size=int(batch_size),
        device=device,
    )
    return {
        "status": "ok",
        "minimal_surface": "token_embedding(tokens) -> g(content,null) -> f(state)",
        "diagnostics": diagnostics,
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _palette_block_bigram_features(
    docs,
    *,
    vocab_size: int,
    n_regimes: int,
) -> np.ndarray:
    block_by_token = _palette_block_map(
        vocab_size=int(vocab_size),
        n_regimes=int(n_regimes),
    )
    n = len(docs)
    k = int(n_regimes)
    features = np.zeros((n, k * k), dtype=np.float64)
    for row_idx, doc in enumerate(docs):
        tokens = [int(tok) for leaf in doc.leaf_token_ids for tok in leaf]
        if any(tok < 0 or tok >= int(vocab_size) for tok in tokens):
            raise ValueError("token id outside [0, vocab_size) in palette bigram ridge")
        blocks = [block_by_token[tok] for tok in tokens]
        for left, right in zip(blocks, blocks[1:]):
            features[row_idx, int(left) * k + int(right)] += 1.0
    return features


def _fit_palette_block_bigram_ridge(
    *,
    train_docs,
    val_docs,
    test_docs,
    vocab_size: int,
    n_regimes: int,
    alpha: float,
) -> dict[str, object]:
    """Closed-form ridge/least-squares on palette-block bigram counts."""
    train_x = _palette_block_bigram_features(
        train_docs,
        vocab_size=int(vocab_size),
        n_regimes=int(n_regimes),
    )
    val_x = _palette_block_bigram_features(
        val_docs,
        vocab_size=int(vocab_size),
        n_regimes=int(n_regimes),
    )
    test_x = _palette_block_bigram_features(
        test_docs,
        vocab_size=int(vocab_size),
        n_regimes=int(n_regimes),
    )
    train_y = _root_truths(train_docs)
    val_y = _root_truths(val_docs)
    test_y = _root_truths(test_docs)
    train_aug = np.concatenate(
        [train_x, np.ones((int(train_x.shape[0]), 1), dtype=np.float64)],
        axis=1,
    )
    alpha_v = float(alpha)
    if alpha_v > 0.0:
        gram = train_aug.T @ train_aug
        penalty = np.eye(int(gram.shape[0]), dtype=np.float64) * alpha_v
        penalty[-1, -1] = 0.0
        coef = np.linalg.solve(gram + penalty, train_aug.T @ train_y)
    else:
        coef, *_ = np.linalg.lstsq(train_aug, train_y, rcond=None)

    def _predict(x: np.ndarray) -> np.ndarray:
        aug = np.concatenate(
            [x, np.ones((int(x.shape[0]), 1), dtype=np.float64)],
            axis=1,
        )
        return aug @ coef

    train_pred = _predict(train_x)
    val_pred = _predict(val_x)
    test_pred = _predict(test_x)
    return {
        "status": "ok",
        "feature": "palette_block_bigram_counts",
        "alpha": alpha_v,
        "n_features": int(train_x.shape[1]),
        "train": _prediction_diagnostics(train_pred, train_y),
        "val": _prediction_diagnostics(val_pred, val_y),
        "test": _prediction_diagnostics(test_pred, test_y),
        "coef_norm": float(np.linalg.norm(coef[:-1])),
        "intercept": float(coef[-1]),
    }


def _sequence_baseline_docs_from_fno_docs(
    docs,
    *,
    vocab_size: int,
    n_regimes: int,
) -> list[SimpleNamespace]:
    """Adapt prepared FNO docs to the flat-token contract used by baselines."""
    block_by_token = _palette_block_map(
        vocab_size=int(vocab_size),
        n_regimes=int(n_regimes),
    )
    out: list[SimpleNamespace] = []
    for doc in docs:
        tokens = tuple(int(tok) for leaf in doc.leaf_token_ids for tok in leaf)
        if any(tok < 0 or tok >= int(vocab_size) for tok in tokens):
            raise ValueError("token id outside [0, vocab_size) in sequence baseline adapter")
        regimes = tuple(int(block_by_token[int(tok)]) for tok in tokens)
        boundaries = tuple(
            idx
            for idx, (left, right) in enumerate(zip(regimes, regimes[1:]), start=1)
            if int(left) != int(right)
        )
        out.append(
            SimpleNamespace(
                tokens=tokens,
                token_regimes=regimes,
                transition_regimes=regimes,
                true_boundaries=boundaries,
                root_count=float(doc.root_count),
            )
        )
    return out


def _baseline_fit_to_summary(fit: dict[str, object]) -> dict[str, object]:
    train_preds = np.asarray(fit.get("train_preds", []), dtype=np.float64)
    val_preds = np.asarray(fit.get("val_preds", []), dtype=np.float64)
    test_preds = np.asarray(fit.get("test_preds", []), dtype=np.float64)
    train_truths = np.asarray(fit.get("train_truths", []), dtype=np.float64)
    val_truths = np.asarray(fit.get("val_truths", []), dtype=np.float64)
    test_truths = np.asarray(fit.get("test_truths", []), dtype=np.float64)
    return {
        "status": "ok",
        "train": _prediction_diagnostics(train_preds, train_truths),
        "val": _prediction_diagnostics(val_preds, val_truths),
        "test": _prediction_diagnostics(test_preds, test_truths),
        "train_metrics": _jsonable(fit.get("train_metrics")),
        "val_metrics": _jsonable(fit.get("val_metrics")),
        "test_metrics": _jsonable(fit.get("test_metrics")),
        "fit_diag": _jsonable(fit.get("fit_diag")),
        "train_docs_used": int(fit.get("train_docs_used", 0) or 0),
    }


def _parse_diagnostic_baselines(raw: str) -> set[str]:
    aliases = {
        "palette": "palette_ridge",
        "palette_bigram": "palette_ridge",
        "palette_block_bigram_ridge": "palette_ridge",
        "ridge": "palette_ridge",
        "fno_transition_channel": "fno_transition",
        "transition_fno": "fno_transition",
        "vanilla_fno": "fno_vanilla",
    }
    parts = {
        str(part).strip().lower()
        for part in str(raw or "").replace(";", ",").split(",")
        if str(part).strip()
    }
    if not parts or parts == {"none"}:
        return set()
    if "all" in parts:
        return {"palette_ridge", "cnn1d", "fno_vanilla", "fno_transition"}
    return {aliases.get(part, part) for part in parts if part != "none"}


def _run_controlled_diagnostic_baselines(
    *,
    requested: set[str],
    train_docs,
    val_docs,
    test_docs,
    args,
    vocab_size: int,
    n_regimes: int,
    device: torch.device,
    log,
) -> dict[str, object]:
    out: dict[str, object] = {}
    if "palette_ridge" in requested:
        log("running diagnostic baseline: palette_block_bigram_ridge")
        try:
            out["palette_block_bigram_ridge"] = _fit_palette_block_bigram_ridge(
                train_docs=train_docs,
                val_docs=val_docs,
                test_docs=test_docs,
                vocab_size=int(vocab_size),
                n_regimes=int(n_regimes),
                alpha=float(args.palette_ridge_alpha),
            )
        except Exception as exc:  # pragma: no cover - diagnostic should not hide main run
            out["palette_block_bigram_ridge"] = {
                "status": "error",
                "error": repr(exc),
            }

    neural_requested = requested.intersection({"cnn1d", "fno_vanilla", "fno_transition"})
    if neural_requested:
        sequence_train_docs = _sequence_baseline_docs_from_fno_docs(
            train_docs,
            vocab_size=int(vocab_size),
            n_regimes=int(n_regimes),
        )
        sequence_val_docs = _sequence_baseline_docs_from_fno_docs(
            val_docs,
            vocab_size=int(vocab_size),
            n_regimes=int(n_regimes),
        )
        sequence_test_docs = _sequence_baseline_docs_from_fno_docs(
            test_docs,
            vocab_size=int(vocab_size),
            n_regimes=int(n_regimes),
        )
        baseline_epochs = (
            int(args.diagnostic_baseline_epochs)
            if int(args.diagnostic_baseline_epochs) > 0
            else int(args.epochs)
        )
        baseline_batch_size = (
            int(args.diagnostic_baseline_batch_size)
            if int(args.diagnostic_baseline_batch_size) > 0
            else int(args.batch_size)
        )
        base_cfg = OPSCountConfig(
            vocab_size=int(vocab_size),
            n_epochs=int(baseline_epochs),
            state_dim=int(args.channels),
            hidden_dim=max(64, int(args.channels)),
            batch_size=int(baseline_batch_size),
            lr=float(args.diagnostic_baseline_lr or args.lr),
            weight_decay=float(args.weight_decay),
            grad_clip_norm=float(args.grad_clip),
            fno_width=int(args.channels),
            fno_n_modes=int(args.g_n_modes),
            fno_n_layers=int(args.g_n_layers),
            doc_sequence_fno_pooling=str(args.leaf_pool),
        )
        seeds = {"effective_model_seed": int(args.seed)}
        if "cnn1d" in requested:
            log("running diagnostic baseline: cnn1d")
            try:
                fit = _fit_cnn1d_baseline_with_predictions(
                    config=base_cfg,
                    seeds=seeds,
                    device=device,
                    train_docs=sequence_train_docs,
                    val_docs=sequence_val_docs,
                    test_docs=sequence_test_docs,
                )
                out["cnn1d"] = _baseline_fit_to_summary(fit)
            except Exception as exc:  # pragma: no cover
                out["cnn1d"] = {"status": "error", "error": repr(exc)}
        if "fno_vanilla" in requested:
            log("running diagnostic baseline: fno_vanilla")
            try:
                fit = _fit_fno_baseline_with_predictions(
                    config=base_cfg,
                    seeds=seeds,
                    device=device,
                    train_docs=sequence_train_docs,
                    val_docs=sequence_val_docs,
                    test_docs=sequence_test_docs,
                )
                out["fno_vanilla"] = _baseline_fit_to_summary(fit)
            except Exception as exc:  # pragma: no cover
                out["fno_vanilla"] = {"status": "error", "error": repr(exc)}
        if "fno_transition" in requested:
            log("running diagnostic baseline: fno_transition_channel")
            try:
                transition_cfg = OPSCountConfig(
                    vocab_size=int(vocab_size),
                    n_epochs=int(baseline_epochs),
                    state_dim=int(args.channels),
                    hidden_dim=max(64, int(args.channels)),
                    batch_size=int(baseline_batch_size),
                    lr=float(args.diagnostic_baseline_lr or args.lr),
                    weight_decay=float(args.weight_decay),
                    grad_clip_norm=float(args.grad_clip),
                    fno_width=int(args.channels),
                    fno_n_modes=int(args.g_n_modes),
                    fno_n_layers=int(args.g_n_layers),
                    doc_sequence_fno_pooling=str(args.leaf_pool),
                    doc_sequence_fno_include_transition_channel=True,
                )
                fit = _fit_fno_baseline_with_predictions(
                    config=transition_cfg,
                    seeds=seeds,
                    device=device,
                    train_docs=sequence_train_docs,
                    val_docs=sequence_val_docs,
                    test_docs=sequence_test_docs,
                )
                out["fno_transition_channel"] = _baseline_fit_to_summary(fit)
            except Exception as exc:  # pragma: no cover
                out["fno_transition_channel"] = {"status": "error", "error": repr(exc)}
    return out


def _boundary_labels_for_leaves(
    leaves,
    *,
    block_by_token: list[int],
    device: torch.device,
) -> torch.Tensor:
    if len(leaves) != 1:
        raise ValueError("boundary-supervision ablation currently requires one leaf")
    tokens = [int(tok) for tok in leaves[0]]
    blocks = [int(block_by_token[int(tok)]) for tok in tokens]
    labels = [
        1.0 if int(left) != int(right) else 0.0
        for left, right in zip(blocks, blocks[1:])
    ]
    return torch.tensor(labels, dtype=torch.float32, device=device)


def _eval_boundary_head(
    *,
    model: CleanUnifiedNO,
    boundary_head: nn.Module,
    docs,
    block_by_token: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    leaf_tokens, _, _, _ = _doc_arrays(docs)
    total_loss = 0.0
    total_items = 0
    total_correct = 0
    total_positive = 0
    total_pred_positive = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    model.eval()
    boundary_head.eval()
    with torch.no_grad():
        for i in range(0, len(docs), int(batch_size)):
            for leaves in leaf_tokens[i : i + int(batch_size)]:
                tok = _to_tensor(leaves, dtype=torch.long, device=device)
                out = model(tok)
                labels = _boundary_labels_for_leaves(
                    leaves,
                    block_by_token=block_by_token,
                    device=device,
                )
                logits = boundary_head(out.root_state.unsqueeze(0)).squeeze(0).squeeze(0)
                if int(labels.numel()) == 0:
                    continue
                loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
                pred = logits.sigmoid().ge(0.5)
                truth = labels.ge(0.5)
                total_loss += float(loss.detach().cpu())
                total_items += int(labels.numel())
                total_correct += int(pred.eq(truth).sum().detach().cpu())
                total_positive += int(truth.sum().detach().cpu())
                total_pred_positive += int(pred.sum().detach().cpu())
                total_tp += int((pred & truth).sum().detach().cpu())
                total_fp += int((pred & ~truth).sum().detach().cpu())
                total_fn += int((~pred & truth).sum().detach().cpu())
    if total_items <= 0:
        return {
            "n_boundary_positions": 0,
            "bce": float("nan"),
            "accuracy": float("nan"),
            "positive_rate": float("nan"),
            "pred_positive_rate": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
    precision = (
        float(total_tp) / float(total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0.0
    )
    recall = (
        float(total_tp) / float(total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    return {
        "n_boundary_positions": int(total_items),
        "bce": float(total_loss / float(total_items)),
        "accuracy": float(total_correct / float(total_items)),
        "positive_rate": float(total_positive / float(total_items)),
        "pred_positive_rate": float(total_pred_positive / float(total_items)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
    }


class _MarkovWitnessReadout(nn.Module):
    """Decode the exact Markov sufficient statistic from a function-valued state."""

    def __init__(
        self,
        *,
        channels: int,
        length: int,
        n_regimes: int,
        kind: str,
    ) -> None:
        super().__init__()
        self.kind = str(kind)
        self.channels = int(channels)
        self.length = int(length)
        self.n_regimes = int(n_regimes)
        if self.kind == "flatten":
            feature_dim = int(channels) * int(length)
            self.net = nn.Identity()
        elif self.kind == "conv_pool":
            hidden = max(int(channels), 16)
            self.net = nn.Sequential(
                nn.Conv1d(int(channels), hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.GELU(),
            )
            feature_dim = 3 * hidden
        else:
            raise ValueError(f"unknown Markov witness readout kind: {kind!r}")
        self.count_head = nn.Linear(feature_dim, 1)
        self.first_head = nn.Linear(feature_dim, int(n_regimes))
        self.last_head = nn.Linear(feature_dim, int(n_regimes))

    def _features(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3:
            raise ValueError(f"witness readout expects (B, C, L); got {tuple(state.shape)}")
        if int(state.shape[-2]) != self.channels or int(state.shape[-1]) != self.length:
            raise ValueError(
                "witness readout received incompatible state shape; "
                f"got {tuple(state.shape)}, expected (B, {self.channels}, {self.length})"
            )
        if self.kind == "flatten":
            return state.reshape(int(state.shape[0]), -1)
        h = self.net(state)
        pooled = h.mean(dim=-1)
        first = h[..., 0]
        last = h[..., -1]
        return torch.cat([pooled, first, last], dim=-1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._features(state)
        count_norm = self.count_head(features).squeeze(-1)
        return count_norm, self.first_head(features), self.last_head(features)


def _markov_witness_targets_for_leaves(
    leaves,
    *,
    block_by_token: list[int],
    target_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(leaves) != 1:
        raise ValueError("Markov witness supervision currently requires one leaf")
    tokens = [int(tok) for tok in leaves[0]]
    if not tokens:
        raise ValueError("Markov witness supervision requires non-empty leaves")
    blocks = [int(block_by_token[int(tok)]) for tok in tokens]
    count = float(sum(1 for left, right in zip(blocks, blocks[1:]) if left != right))
    count_norm = torch.tensor(count / float(target_scale), dtype=torch.float32, device=device)
    first = torch.tensor(int(blocks[0]), dtype=torch.long, device=device)
    last = torch.tensor(int(blocks[-1]), dtype=torch.long, device=device)
    return count_norm, first, last


def _markov_witness_targets_for_tokens(
    tokens,
    *,
    block_by_token: list[int],
    target_scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_list = [int(tok) for tok in tokens]
    if not token_list:
        raise ValueError("Markov witness supervision requires non-empty token spans")
    blocks = [int(block_by_token[int(tok)]) for tok in token_list]
    count = float(sum(1 for left, right in zip(blocks, blocks[1:]) if left != right))
    count_norm = torch.tensor(count / float(target_scale), dtype=torch.float32, device=device)
    first = torch.tensor(int(blocks[0]), dtype=torch.long, device=device)
    last = torch.tensor(int(blocks[-1]), dtype=torch.long, device=device)
    return count_norm, first, last


def _markov_witness_targets_for_spans(
    spans,
    *,
    block_by_token: list[int],
    target_scale: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    count_targets: list[torch.Tensor] = []
    first_targets: list[torch.Tensor] = []
    last_targets: list[torch.Tensor] = []
    for span in spans:
        count, first, last = _markov_witness_targets_for_tokens(
            span,
            block_by_token=block_by_token,
            target_scale=float(target_scale),
            device=device,
        )
        count_targets.append(count)
        first_targets.append(first)
        last_targets.append(last)
    if not count_targets:
        return {
            "count_norm": torch.empty((0,), dtype=torch.float32, device=device),
            "first": torch.empty((0,), dtype=torch.long, device=device),
            "last": torch.empty((0,), dtype=torch.long, device=device),
        }
    return {
        "count_norm": torch.stack(count_targets, dim=0),
        "first": torch.stack(first_targets, dim=0),
        "last": torch.stack(last_targets, dim=0),
    }


def _markov_node_witness_targets_for_leaves(
    leaves,
    *,
    block_by_token: list[int],
    target_scale: float,
    device: torch.device,
) -> dict[str, object]:
    """Exact `(count, first, last)` targets in CleanUnifiedNO merge order.

    `CleanUnifiedNO.forward_doc` appends merge states level by level, carrying
    odd leftovers forward unchanged. This helper mirrors that order with token
    spans so decoded witness losses line up with `output.merge_states`.
    """

    leaf_spans = [[int(tok) for tok in leaf] for leaf in leaves]
    if not leaf_spans:
        raise ValueError("Markov node witness supervision requires at least one leaf")
    if any(len(span) == 0 for span in leaf_spans):
        raise ValueError("Markov node witness supervision requires non-empty leaves")
    merge_spans: list[list[int]] = []
    cur = [list(span) for span in leaf_spans]
    while len(cur) > 1:
        nxt: list[list[int]] = []
        pair_count = len(cur) // 2
        for idx in range(pair_count):
            merged = list(cur[2 * idx]) + list(cur[2 * idx + 1])
            merge_spans.append(merged)
            nxt.append(merged)
        if len(cur) % 2 == 1:
            nxt.append(cur[-1])
        cur = nxt
    root_span = cur[0]
    return {
        "leaf_spans": leaf_spans,
        "merge_spans": merge_spans,
        "root_span": root_span,
        "leaf": _markov_witness_targets_for_spans(
            leaf_spans,
            block_by_token=block_by_token,
            target_scale=float(target_scale),
            device=device,
        ),
        "merge": _markov_witness_targets_for_spans(
            merge_spans,
            block_by_token=block_by_token,
            target_scale=float(target_scale),
            device=device,
        ),
        "root": _markov_witness_targets_for_spans(
            [root_span],
            block_by_token=block_by_token,
            target_scale=float(target_scale),
            device=device,
        ),
    }


def _markov_witness_loss(
    *,
    witness_head: _MarkovWitnessReadout,
    state: torch.Tensor,
    leaves,
    block_by_token: list[int],
    target_scale: float,
    count_weight: float,
    edge_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    count_target, first_target, last_target = _markov_witness_targets_for_leaves(
        leaves,
        block_by_token=block_by_token,
        target_scale=float(target_scale),
        device=device,
    )
    count_pred, first_logits, last_logits = witness_head(state.unsqueeze(0))
    count_loss = F.mse_loss(count_pred.reshape(()), count_target.reshape(()))
    first_loss = F.cross_entropy(first_logits, first_target.reshape(1))
    last_loss = F.cross_entropy(last_logits, last_target.reshape(1))
    edge_loss = 0.5 * (first_loss + last_loss)
    total = float(count_weight) * count_loss + float(edge_weight) * edge_loss
    return total, count_loss, first_loss, last_loss


def _markov_node_witness_loss(
    *,
    witness_head: _MarkovWitnessReadout,
    output: TreeForwardOutputNO,
    leaves,
    block_by_token: list[int],
    target_scale: float,
    count_weight: float,
    edge_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    targets = _markov_node_witness_targets_for_leaves(
        leaves,
        block_by_token=block_by_token,
        target_scale=float(target_scale),
        device=device,
    )

    def _component(states: list[torch.Tensor], target: dict[str, torch.Tensor]):
        if not states:
            zero = output.root_count_norm.new_zeros(())
            return zero, zero, zero, zero, zero
        state_batch = torch.stack(states, dim=0)
        count_pred, first_logits, last_logits = witness_head(state_batch)
        count_target = target["count_norm"].to(device=count_pred.device)
        first_target = target["first"].to(device=count_pred.device)
        last_target = target["last"].to(device=count_pred.device)
        count_loss = F.mse_loss(count_pred, count_target)
        first_loss = F.cross_entropy(first_logits, first_target)
        last_loss = F.cross_entropy(last_logits, last_target)
        edge_loss = 0.5 * (first_loss + last_loss)
        total = float(count_weight) * count_loss + float(edge_weight) * edge_loss
        return total, count_loss, first_loss, last_loss, edge_loss

    leaf_total, leaf_count, leaf_first, leaf_last, leaf_edge = _component(
        list(output.leaf_states),
        targets["leaf"],  # type: ignore[arg-type]
    )
    merge_total, merge_count, merge_first, merge_last, merge_edge = _component(
        list(output.merge_states),
        targets["merge"],  # type: ignore[arg-type]
    )
    total = leaf_total + merge_total
    return total, {
        "leaf_total": leaf_total,
        "merge_total": merge_total,
        "leaf_count": leaf_count,
        "merge_count": merge_count,
        "leaf_first": leaf_first,
        "leaf_last": leaf_last,
        "merge_first": merge_first,
        "merge_last": merge_last,
        "leaf_edge": leaf_edge,
        "merge_edge": merge_edge,
        "count": leaf_count + merge_count,
        "first": leaf_first + merge_first,
        "last": leaf_last + merge_last,
    }


def _balanced_merge_state_triples(
    output: TreeForwardOutputNO,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return `(left, right, parent)` triples in `forward_doc` merge order."""

    triples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    cur = list(output.leaf_states)
    merge_idx = 0
    while len(cur) > 1:
        nxt: list[torch.Tensor] = []
        pair_count = len(cur) // 2
        for idx in range(pair_count):
            if merge_idx >= len(output.merge_states):
                raise ValueError("merge state trace ended before balanced replay finished")
            left = cur[2 * idx]
            right = cur[2 * idx + 1]
            parent = output.merge_states[merge_idx]
            merge_idx += 1
            triples.append((left, right, parent))
            nxt.append(parent)
        if len(cur) % 2 == 1:
            nxt.append(cur[-1])
        cur = nxt
    if merge_idx != len(output.merge_states):
        raise ValueError(
            "balanced replay did not consume every merge state: "
            f"used {merge_idx}, have {len(output.merge_states)}"
        )
    return triples


def _soft_cross_entropy_from_probs(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def _endpoint_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1).mean()


def _markov_local_law_fno_loss(
    *,
    law_head: _MarkovWitnessReadout,
    output: TreeForwardOutputNO,
    leaves,
    block_by_token: list[int],
    target_scale: float,
    leaf_weight: float,
    merge_weight: float,
    idempotence_weight: float,
    count_weight: float,
    edge_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Local-law-style FNO objective without exact merge-node targets.

    C1 supervises decoded leaf states against exact local leaf sketches. C2 is
    relational: decoded parent sketches must agree with the decoded child
    sketches under the Markov boundary rule. It does not regress parent/merge
    states to exact `(count, first, last)` labels. C3 is a differentiable range
    proxy encouraging rounded counts and sharp endpoint distributions.
    """

    targets = _markov_node_witness_targets_for_leaves(
        leaves,
        block_by_token=block_by_token,
        target_scale=float(target_scale),
        device=device,
    )

    if output.leaf_states:
        leaf_state_batch = torch.stack(list(output.leaf_states), dim=0)
        leaf_count_pred, leaf_first_logits, leaf_last_logits = law_head(leaf_state_batch)
        leaf_target = targets["leaf"]  # type: ignore[assignment]
        leaf_count_target = leaf_target["count_norm"].to(device=leaf_count_pred.device)  # type: ignore[index]
        leaf_first_target = leaf_target["first"].to(device=leaf_count_pred.device)  # type: ignore[index]
        leaf_last_target = leaf_target["last"].to(device=leaf_count_pred.device)  # type: ignore[index]
        leaf_count_loss = F.mse_loss(leaf_count_pred, leaf_count_target)
        leaf_first_loss = F.cross_entropy(leaf_first_logits, leaf_first_target)
        leaf_last_loss = F.cross_entropy(leaf_last_logits, leaf_last_target)
        leaf_edge_loss = 0.5 * (leaf_first_loss + leaf_last_loss)
        leaf_loss = float(count_weight) * leaf_count_loss + float(edge_weight) * leaf_edge_loss
    else:
        zero = output.root_count_norm.new_zeros(())
        leaf_loss = zero
        leaf_count_loss = zero
        leaf_first_loss = zero
        leaf_last_loss = zero
        leaf_edge_loss = zero

    triples = _balanced_merge_state_triples(output)
    if triples:
        left_states = torch.stack([triple[0] for triple in triples], dim=0)
        right_states = torch.stack([triple[1] for triple in triples], dim=0)
        parent_states = torch.stack([triple[2] for triple in triples], dim=0)
        left_count, left_first_logits, left_last_logits = law_head(left_states)
        right_count, right_first_logits, right_last_logits = law_head(right_states)
        parent_count, parent_first_logits, parent_last_logits = law_head(parent_states)

        left_first_probs = F.softmax(left_first_logits.detach(), dim=-1)
        left_last_probs = F.softmax(left_last_logits.detach(), dim=-1)
        right_first_probs = F.softmax(right_first_logits.detach(), dim=-1)
        right_last_probs = F.softmax(right_last_logits.detach(), dim=-1)
        same_boundary = (left_last_probs * right_first_probs).sum(dim=-1)
        join = 1.0 - same_boundary
        parent_count_target = (
            left_count.detach()
            + right_count.detach()
            + join / float(target_scale)
        )
        merge_count_loss = F.mse_loss(parent_count, parent_count_target)
        merge_first_loss = _soft_cross_entropy_from_probs(
            parent_first_logits,
            left_first_probs,
        )
        merge_last_loss = _soft_cross_entropy_from_probs(
            parent_last_logits,
            right_last_probs,
        )
        merge_edge_loss = 0.5 * (merge_first_loss + merge_last_loss)
        merge_loss = (
            float(count_weight) * merge_count_loss
            + float(edge_weight) * merge_edge_loss
        )
    else:
        zero = output.root_count_norm.new_zeros(())
        merge_loss = zero
        merge_count_loss = zero
        merge_first_loss = zero
        merge_last_loss = zero
        merge_edge_loss = zero

    all_states = list(output.leaf_states) + list(output.merge_states)
    if all_states:
        all_state_batch = torch.stack(all_states, dim=0)
        count_pred, first_logits, last_logits = law_head(all_state_batch)
        rounded = torch.round(count_pred * float(target_scale)).detach() / float(target_scale)
        count_idemp_loss = F.mse_loss(count_pred, rounded)
        endpoint_idemp_loss = 0.5 * (
            _endpoint_entropy(first_logits) + _endpoint_entropy(last_logits)
        )
        idempotence_loss = count_idemp_loss + endpoint_idemp_loss
    else:
        zero = output.root_count_norm.new_zeros(())
        count_idemp_loss = zero
        endpoint_idemp_loss = zero
        idempotence_loss = zero

    total = (
        float(leaf_weight) * leaf_loss
        + float(merge_weight) * merge_loss
        + float(idempotence_weight) * idempotence_loss
    )
    return total, {
        "leaf_total": leaf_loss,
        "merge_total": merge_loss,
        "idempotence": idempotence_loss,
        "leaf_count": leaf_count_loss,
        "leaf_first": leaf_first_loss,
        "leaf_last": leaf_last_loss,
        "leaf_edge": leaf_edge_loss,
        "merge_count": merge_count_loss,
        "merge_first": merge_first_loss,
        "merge_last": merge_last_loss,
        "merge_edge": merge_edge_loss,
        "count_idempotence": count_idemp_loss,
        "endpoint_idempotence": endpoint_idemp_loss,
        "count": leaf_count_loss + merge_count_loss,
        "first": leaf_first_loss + merge_first_loss,
        "last": leaf_last_loss + merge_last_loss,
    }


def _markov_witness_decode_metrics(
    *,
    witness_head: _MarkovWitnessReadout,
    states: list[torch.Tensor],
    targets: dict[str, torch.Tensor],
    target_scale: float,
) -> dict[str, object]:
    if not states:
        return {
            "status": "skipped",
            "reason": "no states",
            "n": 0,
            "count_diagnostics": _prediction_diagnostics([], []),
        }
    state_batch = torch.stack(states, dim=0)
    count_pred_norm, first_logits, last_logits = witness_head(state_batch)
    count_target_norm = targets["count_norm"].to(device=count_pred_norm.device)
    first_target = targets["first"].to(device=count_pred_norm.device)
    last_target = targets["last"].to(device=count_pred_norm.device)
    pred_counts = (count_pred_norm.detach().cpu().numpy() * float(target_scale)).tolist()
    truth_counts = (count_target_norm.detach().cpu().numpy() * float(target_scale)).tolist()
    first_pred = torch.argmax(first_logits, dim=-1)
    last_pred = torch.argmax(last_logits, dim=-1)
    first_correct = first_pred.eq(first_target)
    last_correct = last_pred.eq(last_target)
    pred_round = torch.round(count_pred_norm * float(target_scale))
    truth_round = torch.round(count_target_norm * float(target_scale))
    count_correct = pred_round.eq(truth_round)
    total = max(1, int(count_pred_norm.numel()))
    count_mse = F.mse_loss(count_pred_norm, count_target_norm)
    first_ce = F.cross_entropy(first_logits, first_target)
    last_ce = F.cross_entropy(last_logits, last_target)
    first_probs = F.softmax(first_logits, dim=-1)
    last_probs = F.softmax(last_logits, dim=-1)
    first_target_oh = F.one_hot(
        first_target,
        num_classes=int(first_logits.shape[-1]),
    ).to(dtype=first_probs.dtype)
    last_target_oh = F.one_hot(
        last_target,
        num_classes=int(last_logits.shape[-1]),
    ).to(dtype=last_probs.dtype)
    decoded_sketch = torch.cat(
        [count_pred_norm.unsqueeze(-1), first_probs, last_probs],
        dim=-1,
    )
    target_sketch = torch.cat(
        [count_target_norm.unsqueeze(-1), first_target_oh, last_target_oh],
        dim=-1,
    )
    canonical_sketch = torch.cat(
        [
            (torch.round(count_pred_norm * float(target_scale)) / float(target_scale)).unsqueeze(-1),
            F.one_hot(first_pred, num_classes=int(first_logits.shape[-1])).to(dtype=first_probs.dtype),
            F.one_hot(last_pred, num_classes=int(last_logits.shape[-1])).to(dtype=last_probs.dtype),
        ],
        dim=-1,
    )
    theta_mae = torch.mean(torch.abs(decoded_sketch - target_sketch))
    eps_idemp_range = torch.mean(torch.abs(decoded_sketch - canonical_sketch))
    eps_idemp_to_exact = torch.mean(torch.abs(canonical_sketch - target_sketch))
    return {
        "status": "ok",
        "n": int(total),
        "count_diagnostics": _prediction_diagnostics(pred_counts, truth_counts),
        "theta_mae": float(theta_mae.detach().cpu()),
        "theta_first_regime_accuracy": float(first_correct.float().mean().detach().cpu()),
        "theta_last_regime_accuracy": float(last_correct.float().mean().detach().cpu()),
        "rounded_count_exact_rate": float(count_correct.float().mean().detach().cpu()),
        "first_accuracy": float(first_correct.float().mean().detach().cpu()),
        "last_accuracy": float(last_correct.float().mean().detach().cpu()),
        "edge_accuracy": float(
            0.5
            * (
                first_correct.float().mean().detach().cpu()
                + last_correct.float().mean().detach().cpu()
            )
        ),
        "full_witness_exact_rate": float(
            (count_correct & first_correct & last_correct).float().mean().detach().cpu()
        ),
        "eps_idemp_range": float(eps_idemp_range.detach().cpu()),
        "eps_idemp_to_exact": float(eps_idemp_to_exact.detach().cpu()),
        "count_mse_norm": float(count_mse.detach().cpu()),
        "first_ce": float(first_ce.detach().cpu()),
        "last_ce": float(last_ce.detach().cpu()),
    }


def _eval_markov_node_witness_head(
    *,
    model: CleanUnifiedNO,
    witness_head: _MarkovWitnessReadout,
    docs,
    block_by_token: list[int],
    target_scale: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    leaf_tokens, _, _, _ = _doc_arrays(docs)
    leaf_states: list[torch.Tensor] = []
    merge_states: list[torch.Tensor] = []
    root_states: list[torch.Tensor] = []
    leaf_targets_count: list[torch.Tensor] = []
    leaf_targets_first: list[torch.Tensor] = []
    leaf_targets_last: list[torch.Tensor] = []
    merge_targets_count: list[torch.Tensor] = []
    merge_targets_first: list[torch.Tensor] = []
    merge_targets_last: list[torch.Tensor] = []
    root_targets_count: list[torch.Tensor] = []
    root_targets_first: list[torch.Tensor] = []
    root_targets_last: list[torch.Tensor] = []
    model.eval()
    witness_head.eval()
    with torch.no_grad():
        for i in range(0, len(docs), int(batch_size)):
            for leaves in leaf_tokens[i : i + int(batch_size)]:
                tok = _to_tensor(leaves, dtype=torch.long, device=device)
                out = model(tok)
                targets = _markov_node_witness_targets_for_leaves(
                    leaves,
                    block_by_token=block_by_token,
                    target_scale=float(target_scale),
                    device=device,
                )
                leaf_states.extend([state.detach() for state in out.leaf_states])
                merge_states.extend([state.detach() for state in out.merge_states])
                root_states.append(out.root_state.detach())

                leaf_t = targets["leaf"]  # type: ignore[assignment]
                merge_t = targets["merge"]  # type: ignore[assignment]
                root_t = targets["root"]  # type: ignore[assignment]
                leaf_targets_count.append(leaf_t["count_norm"])  # type: ignore[index]
                leaf_targets_first.append(leaf_t["first"])  # type: ignore[index]
                leaf_targets_last.append(leaf_t["last"])  # type: ignore[index]
                if int(merge_t["count_norm"].numel()) > 0:  # type: ignore[index]
                    merge_targets_count.append(merge_t["count_norm"])  # type: ignore[index]
                    merge_targets_first.append(merge_t["first"])  # type: ignore[index]
                    merge_targets_last.append(merge_t["last"])  # type: ignore[index]
                root_targets_count.append(root_t["count_norm"])  # type: ignore[index]
                root_targets_first.append(root_t["first"])  # type: ignore[index]
                root_targets_last.append(root_t["last"])  # type: ignore[index]

    def _cat(parts: list[torch.Tensor], *, dtype: torch.dtype) -> torch.Tensor:
        if not parts:
            return torch.empty((0,), dtype=dtype, device=device)
        return torch.cat(parts, dim=0).to(device=device)

    leaf_target = {
        "count_norm": _cat(leaf_targets_count, dtype=torch.float32),
        "first": _cat(leaf_targets_first, dtype=torch.long),
        "last": _cat(leaf_targets_last, dtype=torch.long),
    }
    merge_target = {
        "count_norm": _cat(merge_targets_count, dtype=torch.float32),
        "first": _cat(merge_targets_first, dtype=torch.long),
        "last": _cat(merge_targets_last, dtype=torch.long),
    }
    root_target = {
        "count_norm": _cat(root_targets_count, dtype=torch.float32),
        "first": _cat(root_targets_first, dtype=torch.long),
        "last": _cat(root_targets_last, dtype=torch.long),
    }
    all_states = leaf_states + merge_states
    all_target = {
        "count_norm": torch.cat(
            [leaf_target["count_norm"], merge_target["count_norm"]],
            dim=0,
        ),
        "first": torch.cat([leaf_target["first"], merge_target["first"]], dim=0),
        "last": torch.cat([leaf_target["last"], merge_target["last"]], dim=0),
    }
    all_metrics = _markov_witness_decode_metrics(
        witness_head=witness_head,
        states=all_states,
        targets=all_target,
        target_scale=float(target_scale),
    )
    return {
        "status": "ok" if docs else "skipped",
        "n_docs": int(len(docs)),
        "leaf": _markov_witness_decode_metrics(
            witness_head=witness_head,
            states=leaf_states,
            targets=leaf_target,
            target_scale=float(target_scale),
        ),
        "merge": _markov_witness_decode_metrics(
            witness_head=witness_head,
            states=merge_states,
            targets=merge_target,
            target_scale=float(target_scale),
        ),
        "root": _markov_witness_decode_metrics(
            witness_head=witness_head,
            states=root_states,
            targets=root_target,
            target_scale=float(target_scale),
        ),
        "all_nonroot_nodes": all_metrics,
        "range_diagnostics": {
            "status": all_metrics.get("status"),
            "note": (
                "Decoded witness range/idempotence proxy only: rounded count "
                "and first/last exact rates after decoding states."
            ),
            "rounded_count_exact_rate": all_metrics.get("rounded_count_exact_rate"),
            "first_accuracy": all_metrics.get("first_accuracy"),
            "last_accuracy": all_metrics.get("last_accuracy"),
            "full_witness_exact_rate": all_metrics.get("full_witness_exact_rate"),
        },
    }


def _eval_markov_witness_head(
    *,
    model: CleanUnifiedNO,
    witness_head: _MarkovWitnessReadout,
    docs,
    block_by_token: list[int],
    target_scale: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    leaf_tokens, _, _, _ = _doc_arrays(docs)
    preds: list[float] = []
    truths: list[float] = []
    first_correct = 0
    last_correct = 0
    full_correct = 0
    total = 0
    count_loss_sum = 0.0
    first_loss_sum = 0.0
    last_loss_sum = 0.0
    model.eval()
    witness_head.eval()
    with torch.no_grad():
        for i in range(0, len(docs), int(batch_size)):
            for leaves in leaf_tokens[i : i + int(batch_size)]:
                tok = _to_tensor(leaves, dtype=torch.long, device=device)
                out = model(tok)
                count_target, first_target, last_target = _markov_witness_targets_for_leaves(
                    leaves,
                    block_by_token=block_by_token,
                    target_scale=float(target_scale),
                    device=device,
                )
                count_pred_norm, first_logits, last_logits = witness_head(
                    out.root_state.unsqueeze(0)
                )
                count_loss_sum += float(
                    F.mse_loss(count_pred_norm.reshape(()), count_target.reshape(())).detach().cpu()
                )
                first_loss_sum += float(
                    F.cross_entropy(first_logits, first_target.reshape(1)).detach().cpu()
                )
                last_loss_sum += float(
                    F.cross_entropy(last_logits, last_target.reshape(1)).detach().cpu()
                )
                pred_count = float(count_pred_norm.reshape(()).detach().cpu()) * float(target_scale)
                truth_count = float(count_target.detach().cpu()) * float(target_scale)
                first_pred = int(first_logits.argmax(dim=-1).reshape(()).detach().cpu())
                last_pred = int(last_logits.argmax(dim=-1).reshape(()).detach().cpu())
                first_truth = int(first_target.detach().cpu())
                last_truth = int(last_target.detach().cpu())
                preds.append(pred_count)
                truths.append(truth_count)
                first_ok = int(first_pred == first_truth)
                last_ok = int(last_pred == last_truth)
                count_ok = int(round(pred_count) == round(truth_count))
                first_correct += first_ok
                last_correct += last_ok
                full_correct += int(first_ok and last_ok and count_ok)
                total += 1
    if total <= 0:
        return {
            "status": "skipped",
            "reason": "no docs",
            "count_diagnostics": _prediction_diagnostics([], []),
        }
    return {
        "status": "ok",
        "n": int(total),
        "count_diagnostics": _prediction_diagnostics(preds, truths),
        "first_accuracy": float(first_correct / total),
        "last_accuracy": float(last_correct / total),
        "edge_accuracy": float(0.5 * (first_correct + last_correct) / total),
        "full_witness_exact_rate": float(full_correct / total),
        "count_mse_norm": float(count_loss_sum / total),
        "first_ce": float(first_loss_sum / total),
        "last_ce": float(last_loss_sum / total),
    }


def _run_boundary_supervision_ablation(
    *,
    train_docs,
    val_docs,
    test_docs,
    args,
    vocab_size: int,
    target_scale: float,
    n_regimes: int,
    device: torch.device,
    log,
) -> dict[str, object]:
    if not train_docs:
        return {"status": "skipped", "reason": "no train docs"}
    n_leaves = len(train_docs[0].leaf_token_ids)
    if int(n_leaves) != 1:
        return {
            "status": "skipped",
            "reason": f"requires one leaf per doc; got {int(n_leaves)}",
        }
    if not HAS_NEURAL_OPERATOR:
        return {
            "status": "skipped",
            "reason": "neuraloperator is not installed",
        }
    log("running stronger-supervision ablation: CleanUnifiedNO + boundary BCE")
    torch.manual_seed(int(args.seed) + 90_000)
    model = CleanUnifiedNO(
        vocab_size=int(vocab_size),
        target_scale=float(target_scale),
        channels=int(args.channels),
        g_n_modes=int(args.g_n_modes),
        g_n_layers=int(args.g_n_layers),
        scorer_n_modes=int(args.scorer_n_modes),
        scorer_n_layers=int(args.scorer_n_layers),
        pooling_mode=str(args.leaf_pool),
    ).to(device)
    boundary_head = nn.Conv1d(int(model.channels), 1, kernel_size=2).to(device)
    lr = float(args.boundary_supervision_lr or args.lr)
    if str(args.optimizer) == "adamw":
        optimizer = optim.AdamW(
            list(model.parameters()) + list(boundary_head.parameters()),
            lr=lr,
            weight_decay=float(args.weight_decay),
        )
    else:
        optimizer = optim.Adam(
            list(model.parameters()) + list(boundary_head.parameters()),
            lr=lr,
        )
    scheduler = None
    n_epochs = int(args.boundary_supervision_epochs)
    if str(args.lr_schedule) == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, n_epochs),
        )
    block_by_token = _palette_block_map(vocab_size=int(vocab_size), n_regimes=int(n_regimes))
    train_arrays = _doc_arrays(train_docs)
    best_val = float("inf")
    best_epoch = 0
    best_model_state = _clone_checkpoint_state(model)
    best_boundary_state = _clone_checkpoint_state(boundary_head)
    history: list[dict[str, float | int]] = []
    batch_size = int(max(1, args.batch_size))
    for epoch in range(1, n_epochs + 1):
        model.train()
        boundary_head.train()
        order = torch.randperm(len(train_docs)).tolist()
        running_loss = 0.0
        running_root = 0.0
        running_boundary = 0.0
        n_batches = 0
        for bi in range(0, len(train_docs), batch_size):
            optimizer.zero_grad()
            losses = []
            root_terms = []
            boundary_terms = []
            for di in order[bi : bi + batch_size]:
                leaves = train_arrays[0][di]
                tok = _to_tensor(leaves, dtype=torch.long, device=device)
                out = model(tok)
                root_t = _to_tensor(train_arrays[3][di], dtype=torch.float32, device=device)
                root_loss = root_mse_loss(
                    out,
                    root_count=root_t,
                    target_scale=model.target_scale,
                )
                labels = _boundary_labels_for_leaves(
                    leaves,
                    block_by_token=block_by_token,
                    device=device,
                )
                logits = boundary_head(out.root_state.unsqueeze(0)).squeeze(0).squeeze(0)
                if int(labels.numel()) == 0:
                    boundary_loss = root_loss.new_zeros(())
                else:
                    boundary_loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss = root_loss + float(args.boundary_supervision_weight) * boundary_loss
                losses.append(loss)
                root_terms.append(root_loss.detach())
                boundary_terms.append(boundary_loss.detach())
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            if float(args.grad_clip) > 0:
                nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(boundary_head.parameters()),
                    float(args.grad_clip),
                )
            optimizer.step()
            running_loss += float(batch_loss.detach().cpu())
            running_root += float(torch.stack(root_terms).mean().cpu())
            running_boundary += float(torch.stack(boundary_terms).mean().cpu())
            n_batches += 1
        if scheduler is not None:
            scheduler.step()
        val_root_mae = _eval_root_mae(
            model,
            val_docs,
            batch_size=batch_size,
            device=device,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": int(epoch),
            "train_loss": float(running_loss / max(1, n_batches)),
            "train_root_loss": float(running_root / max(1, n_batches)),
            "train_boundary_loss": float(running_boundary / max(1, n_batches)),
            "val_root_mae": float(val_root_mae),
            "lr": current_lr,
        }
        history.append(row)
        if val_root_mae < best_val:
            best_val = float(val_root_mae)
            best_epoch = int(epoch)
            best_model_state = _clone_checkpoint_state(model)
            best_boundary_state = _clone_checkpoint_state(boundary_head)
        log(
            "boundary ablation epoch "
            f"{epoch:2d}: train_loss={row['train_loss']:.4f} "
            f"val_root_mae={val_root_mae:.4f} best={best_val:.4f}@ep{best_epoch}"
        )
    model.load_state_dict(best_model_state)
    boundary_head.load_state_dict(best_boundary_state)
    return {
        "status": "ok",
        "supervision": "root_mse_plus_boundary_bce",
        "boundary_supervision_weight": float(args.boundary_supervision_weight),
        "best_val_root_mae": float(best_val),
        "best_val_epoch": int(best_epoch),
        "history": history,
        "prediction_diagnostics": {
            "train": _eval_model_root_diagnostics(
                model,
                train_docs,
                batch_size=batch_size,
                device=device,
            ),
            "val": _eval_model_root_diagnostics(
                model,
                val_docs,
                batch_size=batch_size,
                device=device,
            ),
            "test": _eval_model_root_diagnostics(
                model,
                test_docs,
                batch_size=batch_size,
                device=device,
            ),
        },
        "boundary_diagnostics": {
            "train": _eval_boundary_head(
                model=model,
                boundary_head=boundary_head,
                docs=train_docs,
                block_by_token=block_by_token,
                batch_size=batch_size,
                device=device,
            ),
            "val": _eval_boundary_head(
                model=model,
                boundary_head=boundary_head,
                docs=val_docs,
                block_by_token=block_by_token,
                batch_size=batch_size,
                device=device,
            ),
            "test": _eval_boundary_head(
                model=model,
                boundary_head=boundary_head,
                docs=test_docs,
                block_by_token=block_by_token,
                batch_size=batch_size,
                device=device,
            ),
        },
    }


def _run_markov_witness_supervision_ablation(
    *,
    train_docs,
    val_docs,
    test_docs,
    args,
    vocab_size: int,
    target_scale: float,
    n_regimes: int,
    device: torch.device,
    log,
) -> dict[str, object]:
    if not train_docs:
        return {"status": "skipped", "reason": "no train docs"}
    n_leaves = len(train_docs[0].leaf_token_ids)
    if int(n_leaves) != 1:
        return {
            "status": "skipped",
            "reason": f"requires one leaf per doc; got {int(n_leaves)}",
        }
    if not HAS_NEURAL_OPERATOR:
        return {
            "status": "skipped",
            "reason": "neuraloperator is not installed",
        }
    log("running theorem-sketch ablation: CleanUnifiedNO + Markov (count, first, last) witness")
    torch.manual_seed(int(args.seed) + 91_000)
    model = CleanUnifiedNO(
        vocab_size=int(vocab_size),
        target_scale=float(target_scale),
        channels=int(args.channels),
        g_n_modes=int(args.g_n_modes),
        g_n_layers=int(args.g_n_layers),
        scorer_n_modes=int(args.scorer_n_modes),
        scorer_n_layers=int(args.scorer_n_layers),
        pooling_mode=str(args.leaf_pool),
    ).to(device)
    witness_head = _MarkovWitnessReadout(
        channels=int(model.channels),
        length=int(len(train_docs[0].leaf_token_ids[0])),
        n_regimes=int(n_regimes),
        kind=str(args.markov_witness_readout),
    ).to(device)
    trainable_params = list(model.parameters()) + list(witness_head.parameters())
    lr = float(args.markov_witness_lr or args.lr)
    if str(args.optimizer) == "adamw":
        optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=float(args.weight_decay),
        )
    else:
        optimizer = optim.Adam(trainable_params, lr=lr)
    scheduler = None
    n_epochs = int(args.markov_witness_epochs)
    if str(args.lr_schedule) == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, n_epochs),
        )
    block_by_token = _palette_block_map(vocab_size=int(vocab_size), n_regimes=int(n_regimes))
    train_arrays = _doc_arrays(train_docs)
    best_val = float("inf")
    best_epoch = 0
    best_model_state = _clone_checkpoint_state(model)
    best_witness_state = _clone_checkpoint_state(witness_head)
    history: list[dict[str, float | int]] = []
    batch_size = int(max(1, args.batch_size))
    for epoch in range(1, n_epochs + 1):
        model.train()
        witness_head.train()
        order = torch.randperm(len(train_docs)).tolist()
        running_loss = 0.0
        running_root = 0.0
        running_witness = 0.0
        running_witness_count = 0.0
        running_first = 0.0
        running_last = 0.0
        n_batches = 0
        for bi in range(0, len(train_docs), batch_size):
            optimizer.zero_grad()
            losses = []
            root_terms = []
            witness_terms = []
            witness_count_terms = []
            first_terms = []
            last_terms = []
            for di in order[bi : bi + batch_size]:
                leaves = train_arrays[0][di]
                tok = _to_tensor(leaves, dtype=torch.long, device=device)
                out = model(tok)
                root_t = _to_tensor(train_arrays[3][di], dtype=torch.float32, device=device)
                root_loss = root_mse_loss(
                    out,
                    root_count=root_t,
                    target_scale=model.target_scale,
                )
                witness_loss, count_loss, first_loss, last_loss = _markov_witness_loss(
                    witness_head=witness_head,
                    state=out.root_state,
                    leaves=leaves,
                    block_by_token=block_by_token,
                    target_scale=float(model.target_scale),
                    count_weight=float(args.markov_witness_count_weight),
                    edge_weight=float(args.markov_witness_edge_weight),
                    device=device,
                )
                loss = root_loss + float(args.markov_witness_weight) * witness_loss
                losses.append(loss)
                root_terms.append(root_loss.detach())
                witness_terms.append(witness_loss.detach())
                witness_count_terms.append(count_loss.detach())
                first_terms.append(first_loss.detach())
                last_terms.append(last_loss.detach())
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            if float(args.grad_clip) > 0:
                nn.utils.clip_grad_norm_(trainable_params, float(args.grad_clip))
            optimizer.step()
            running_loss += float(batch_loss.detach().cpu())
            running_root += float(torch.stack(root_terms).mean().cpu())
            running_witness += float(torch.stack(witness_terms).mean().cpu())
            running_witness_count += float(torch.stack(witness_count_terms).mean().cpu())
            running_first += float(torch.stack(first_terms).mean().cpu())
            running_last += float(torch.stack(last_terms).mean().cpu())
            n_batches += 1
        if scheduler is not None:
            scheduler.step()
        val_root_mae = _eval_root_mae(
            model,
            val_docs,
            batch_size=batch_size,
            device=device,
        )
        val_witness = _eval_markov_witness_head(
            model=model,
            witness_head=witness_head,
            docs=val_docs,
            block_by_token=block_by_token,
            target_scale=float(model.target_scale),
            batch_size=batch_size,
            device=device,
        )
        val_witness_count_mae = float(
            dict(val_witness.get("count_diagnostics", {})).get("root_mae", float("nan"))
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": int(epoch),
            "train_loss": float(running_loss / max(1, n_batches)),
            "train_root_loss": float(running_root / max(1, n_batches)),
            "train_witness_loss": float(running_witness / max(1, n_batches)),
            "train_witness_count_loss": float(running_witness_count / max(1, n_batches)),
            "train_first_ce": float(running_first / max(1, n_batches)),
            "train_last_ce": float(running_last / max(1, n_batches)),
            "val_root_mae": float(val_root_mae),
            "val_witness_count_mae": float(val_witness_count_mae),
            "val_witness_full_exact_rate": float(
                val_witness.get("full_witness_exact_rate", float("nan"))
            ),
            "lr": current_lr,
        }
        history.append(row)
        if val_root_mae < best_val:
            best_val = float(val_root_mae)
            best_epoch = int(epoch)
            best_model_state = _clone_checkpoint_state(model)
            best_witness_state = _clone_checkpoint_state(witness_head)
        log(
            "markov witness ablation epoch "
            f"{epoch:2d}: train_loss={row['train_loss']:.4f} "
            f"val_root_mae={val_root_mae:.4f} "
            f"val_witness_count_mae={val_witness_count_mae:.4f} "
            f"val_witness_exact={row['val_witness_full_exact_rate']:.3f} "
            f"best={best_val:.4f}@ep{best_epoch}"
        )
    model.load_state_dict(best_model_state)
    witness_head.load_state_dict(best_witness_state)
    return {
        "status": "ok",
        "supervision": "root_mse_plus_markov_witness",
        "markov_witness_weight": float(args.markov_witness_weight),
        "markov_witness_count_weight": float(args.markov_witness_count_weight),
        "markov_witness_edge_weight": float(args.markov_witness_edge_weight),
        "markov_witness_readout": str(args.markov_witness_readout),
        "best_val_root_mae": float(best_val),
        "best_val_epoch": int(best_epoch),
        "history": history,
        "prediction_diagnostics": {
            "train": _eval_model_root_diagnostics(
                model,
                train_docs,
                batch_size=batch_size,
                device=device,
            ),
            "val": _eval_model_root_diagnostics(
                model,
                val_docs,
                batch_size=batch_size,
                device=device,
            ),
            "test": _eval_model_root_diagnostics(
                model,
                test_docs,
                batch_size=batch_size,
                device=device,
            ),
        },
        "witness_diagnostics": {
            "train": _eval_markov_witness_head(
                model=model,
                witness_head=witness_head,
                docs=train_docs,
                block_by_token=block_by_token,
                target_scale=float(model.target_scale),
                batch_size=batch_size,
                device=device,
            ),
            "val": _eval_markov_witness_head(
                model=model,
                witness_head=witness_head,
                docs=val_docs,
                block_by_token=block_by_token,
                target_scale=float(model.target_scale),
                batch_size=batch_size,
                device=device,
            ),
            "test": _eval_markov_witness_head(
                model=model,
                witness_head=witness_head,
                docs=test_docs,
                block_by_token=block_by_token,
                target_scale=float(model.target_scale),
                batch_size=batch_size,
                device=device,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="recoverable_v5_t2048")
    parser.add_argument(
        "--load-data-bundle",
        default=None,
        help="Load an existing MarkovOPSDataBundle JSON/PKL for train/val/test docs.",
    )
    parser.add_argument(
        "--doc-tokens",
        type=int,
        default=0,
        help=(
            "If >0, generate a sticky recoverable corpus with this document "
            "length instead of using a named prepared benchmark."
        ),
    )
    parser.add_argument(
        "--expected-boundaries",
        type=float,
        default=None,
        help="Expected boundaries/doc for --doc-tokens. Defaults to 5*sqrt(doc_tokens/128).",
    )
    parser.add_argument("--leaf-tokens", type=int, default=2048,
                        help="2048 = 1 leaf/doc (zero merges); 256 = 8 leaves/doc; 16 = 128 leaves/doc")
    parser.add_argument("--train-docs", type=int, default=1024)
    parser.add_argument(
        "--eval-docs",
        type=int,
        default=None,
        help="Optional cap for val/test docs during quick architecture probes.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16, help="Docs per batch")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--g-n-modes", type=int, default=32,
                        help="Modes for the shared g (used at leaves and merges)")
    parser.add_argument("--g-n-layers", type=int, default=2)
    parser.add_argument("--scorer-n-modes", type=int, default=16)
    parser.add_argument("--scorer-n-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lr-schedule", default="cosine", choices=["none", "cosine"])
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm. Set to 0 to disable.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--leaf-pool", default="sum", choices=["mean", "sum"])
    parser.add_argument("--root-only", action="store_true",
                        help="Train only on root MSE (skip leaf/merge labels).")
    parser.add_argument(
        "--training-objective",
        default="root",
        choices=[
            "root",
            "contextual_sufficiency",
            "markov_node_witness",
            "markov_local_laws_fno",
        ],
        help=(
            "root keeps the existing objective. contextual_sufficiency adds "
            "generic finite-context query losses; the current Markov adapter "
            "can also enact two-sided contexts through the shared-g surface. "
            "markov_node_witness adds direct decoded (count, first, last) "
            "supervision on every leaf and merge state. markov_local_laws_fno "
            "uses leaf calibration plus relational merge/range laws without "
            "exact merge-node targets. Use --enable-contextual-sufficiency to "
            "stack the contextual NASS/NASSS-style loss on top of any "
            "training-objective for laws+contextual hybrid runs."
        ),
    )
    parser.add_argument(
        "--enable-contextual-sufficiency",
        action="store_true",
        help=(
            "Add the contextual-sufficiency (NASS/NASSS-style) auxiliary loss "
            "on top of whichever --training-objective is selected. Lets us run "
            "markov_local_laws_fno + contextual together (the JAX-style "
            "framework: laws as identifiability tie-breaker on top of the "
            "contextual sufficient-summary objective)."
        ),
    )
    parser.add_argument(
        "--context-samples-per-doc",
        type=int,
        default=0,
        help=(
            "Contextual queries per batch document when contextual sufficiency "
            "is active (--training-objective contextual_sufficiency or "
            "--enable-contextual-sufficiency). 0 keeps root-only behavior."
        ),
    )
    parser.add_argument(
        "--contextual-loss-weight",
        type=float,
        default=1.0,
        help="Weight on sampled contextual oracle MSE.",
    )
    parser.add_argument(
        "--infomax-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on the optional dependence loss between item states and "
            "empirical contextual response signatures. Kept under the older "
            "infomax flag name for CLI compatibility."
        ),
    )
    parser.add_argument(
        "--contextual-dependence-objective",
        default="infonce",
        choices=["infonce", "regression", "dcorr", "jsd", "dv", "wasserstein", "none"],
        help=(
            "Auxiliary contextual-response dependence objective. regression "
            "uses the 2023 sliced-summary pattern; dcorr is critic-free; "
            "jsd/dv/wasserstein/infonce mirror NASS-style infomax variants."
        ),
    )
    parser.add_argument(
        "--response-signature-contexts",
        type=int,
        default=0,
        help=(
            "Number of sampled contexts used to build each empirical response "
            "signature for the optional infomax loss."
        ),
    )
    parser.add_argument(
        "--response-signature-slices",
        type=int,
        default=0,
        help=(
            "Random target slices for --contextual-dependence-objective "
            "regression. 0 predicts the full empirical response signature."
        ),
    )
    parser.add_argument(
        "--contextual-response-regressor",
        default="mean_linear",
        choices=["mean_linear", "conv_pool", "flatten"],
        help=(
            "Readout used by --contextual-dependence-objective=regression. "
            "mean_linear preserves the previous pooled-state behavior; "
            "conv_pool and flatten let the auxiliary see the full function-valued state."
        ),
    )
    parser.add_argument(
        "--exact-witness-n-regimes",
        type=int,
        default=4,
        help=(
            "If >0, report the disjoint-palette exact root-count witness using "
            "this many palette blocks. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--require-exact-contract-zero",
        action="store_true",
        help="Fail if the deterministic one-leaf g/f surface check is nonzero.",
    )
    parser.add_argument(
        "--diagnostic-baselines",
        default="palette_ridge",
        help=(
            "Comma-separated optional baselines: none, palette_ridge, cnn1d, "
            "fno_vanilla, fno_transition, all. Heavy neural baselines are opt-in."
        ),
    )
    parser.add_argument(
        "--diagnostic-baseline-epochs",
        type=int,
        default=0,
        help="Epochs for optional neural diagnostic baselines. 0 uses --epochs.",
    )
    parser.add_argument(
        "--diagnostic-baseline-batch-size",
        type=int,
        default=0,
        help="Batch size for optional neural diagnostic baselines. 0 uses --batch-size.",
    )
    parser.add_argument(
        "--diagnostic-baseline-lr",
        type=float,
        default=0.0,
        help="LR for optional neural diagnostic baselines. 0 uses --lr.",
    )
    parser.add_argument(
        "--palette-ridge-alpha",
        type=float,
        default=0.0,
        help="Ridge alpha for palette-block bigram baseline. 0 uses least squares.",
    )
    parser.add_argument(
        "--run-boundary-supervision-ablation",
        action="store_true",
        help="Run a second one-leaf CleanUnifiedNO trained with root MSE + boundary BCE.",
    )
    parser.add_argument(
        "--boundary-supervision-epochs",
        type=int,
        default=0,
        help="Epochs for boundary-supervision ablation. 0 uses --epochs.",
    )
    parser.add_argument(
        "--boundary-supervision-weight",
        type=float,
        default=1.0,
        help="Weight on boundary BCE in the stronger-supervision ablation.",
    )
    parser.add_argument(
        "--boundary-supervision-lr",
        type=float,
        default=0.0,
        help="LR for boundary-supervision ablation. 0 uses --lr.",
    )
    parser.add_argument(
        "--run-markov-witness-supervision-ablation",
        action="store_true",
        help=(
            "Run a second one-leaf CleanUnifiedNO trained with root MSE plus "
            "direct Markov witness decoding: count, first regime, last regime."
        ),
    )
    parser.add_argument(
        "--markov-witness-epochs",
        type=int,
        default=0,
        help="Epochs for Markov witness ablation. 0 uses --epochs.",
    )
    parser.add_argument(
        "--markov-witness-weight",
        type=float,
        default=1.0,
        help="Overall weight on Markov witness loss in the theorem-sketch ablation.",
    )
    parser.add_argument(
        "--markov-witness-count-weight",
        type=float,
        default=1.0,
        help="Weight on normalized count MSE inside the Markov witness loss.",
    )
    parser.add_argument(
        "--markov-witness-edge-weight",
        type=float,
        default=1.0,
        help="Weight on first/last regime cross-entropy inside the Markov witness loss.",
    )
    parser.add_argument(
        "--markov-witness-lr",
        type=float,
        default=0.0,
        help="LR for Markov witness ablation. 0 uses --lr.",
    )
    parser.add_argument(
        "--markov-witness-readout",
        default="flatten",
        choices=["flatten", "conv_pool"],
        help="Readout architecture used to decode the Markov witness from z_x.",
    )
    parser.add_argument(
        "--markov-law-weight",
        type=float,
        default=1.0,
        help="Overall weight on the FNO local-law objective.",
    )
    parser.add_argument(
        "--markov-law-leaf-weight",
        type=float,
        default=1.0,
        help="Weight on C1-style decoded leaf calibration.",
    )
    parser.add_argument(
        "--markov-law-merge-weight",
        type=float,
        default=1.0,
        help="Weight on C2-style decoded relational merge consistency.",
    )
    parser.add_argument(
        "--markov-law-idempotence-weight",
        type=float,
        default=0.1,
        help="Weight on decoded range/idempotence proxy.",
    )
    parser.add_argument(
        "--markov-law-count-weight",
        type=float,
        default=1.0,
        help="Weight on count terms inside FNO local-law losses.",
    )
    parser.add_argument(
        "--markov-law-edge-weight",
        type=float,
        default=1.0,
        help="Weight on first/last endpoint terms inside FNO local-law losses.",
    )
    parser.add_argument(
        "--markov-law-readout",
        default="flatten",
        choices=["flatten", "conv_pool"],
        help="Readout architecture used by --training-objective=markov_local_laws_fno.",
    )
    args = parser.parse_args()
    if int(args.boundary_supervision_epochs) <= 0:
        args.boundary_supervision_epochs = int(args.epochs)
    if int(args.markov_witness_epochs) <= 0:
        args.markov_witness_epochs = int(args.epochs)
    contextual_sufficiency_active = (
        str(args.training_objective) == "contextual_sufficiency"
        or bool(args.enable_contextual_sufficiency)
    )
    if contextual_sufficiency_active and int(args.context_samples_per_doc) <= 0:
        args.context_samples_per_doc = 1
    if (
        float(args.infomax_loss_weight) > 0.0
        and str(args.contextual_dependence_objective) == "none"
    ):
        raise ValueError("--contextual-dependence-objective=none requires --infomax-loss-weight=0")
    if float(args.infomax_loss_weight) > 0.0 and int(args.response_signature_contexts) <= 0:
        args.response_signature_contexts = 4

    torch.manual_seed(int(args.seed))
    if args.device == "cuda" and args.gpu is not None:
        torch.cuda.set_device(int(args.gpu))
    device = torch.device(args.device)

    out_root = Path(
        args.output_root or str(REPO / f"outputs/clean_unified_no_{_ts()}")
    )
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "run.log"

    def log(msg: str) -> None:
        line = f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    log(f"args: {vars(args)}")
    log(f"device: {device}")

    log("loading data ...")
    t0 = time.time()
    generation_meta = {}
    if args.load_data_bundle:
        bundle_eval_docs = int(args.eval_docs) if args.eval_docs is not None else 128
        train_docs, val_docs, test_docs, generation_meta = _load_split_docs_from_bundle(
            bundle_path=Path(str(args.load_data_bundle)),
            train_docs=int(args.train_docs),
            eval_docs=int(bundle_eval_docs),
            fixed_leaf_tokens=int(args.leaf_tokens),
        )
    elif int(args.doc_tokens) > 0:
        direct_eval_docs = int(args.eval_docs) if args.eval_docs is not None else 128
        train_docs, val_docs, test_docs, generation_meta = _load_split_docs_direct(
            doc_tokens=int(args.doc_tokens),
            train_docs=int(args.train_docs),
            eval_docs=int(direct_eval_docs),
            fixed_leaf_tokens=int(args.leaf_tokens),
            expected_boundaries=args.expected_boundaries,
            seed=int(args.seed),
        )
    else:
        train_docs, val_docs, test_docs = _load_split_docs(
            benchmark=str(args.benchmark),
            train_docs=int(args.train_docs),
            fixed_leaf_tokens=int(args.leaf_tokens),
            seed=int(args.seed),
        )
    if args.eval_docs is not None:
        eval_docs = max(1, int(args.eval_docs))
        val_docs = val_docs[:eval_docs]
        test_docs = test_docs[:eval_docs]
    log(f"  loaded in {time.time() - t0:.1f}s. "
        f"train={len(train_docs)}, val={len(val_docs)}, test={len(test_docs)}")
    n_leaves_per_doc = len(train_docs[0].leaf_token_ids)
    leaf_len = len(train_docs[0].leaf_token_ids[0])
    all_loaded_docs = list(train_docs) + list(val_docs) + list(test_docs)
    vocab_max = 1 + max(
        int(t) for d in all_loaded_docs for leaf in d.leaf_token_ids for t in leaf
    )
    target_scale = float(max(d.root_count for d in train_docs)) or 1.0
    log(f"  shape: n_leaves={n_leaves_per_doc}, leaf_len={leaf_len}, "
        f"vocab_max={vocab_max}, target_scale={target_scale:.2f}")
    metadata_block_by_token = list(generation_meta.get("block_by_token") or [])
    if metadata_block_by_token:
        if len(metadata_block_by_token) < int(vocab_max):
            raise ValueError(
                "loaded bundle metadata block_by_token is shorter than observed vocab: "
                f"{len(metadata_block_by_token)} < {vocab_max}"
            )
        contextual_block_by_token = [int(x) for x in metadata_block_by_token[: int(vocab_max)]]
        contextual_n_regimes = int(max(contextual_block_by_token)) + 1
    else:
        contextual_n_regimes = (
            int(args.exact_witness_n_regimes)
            if int(args.exact_witness_n_regimes) > 0
            else 4
        )
        contextual_block_by_token = _palette_block_map(
            vocab_size=int(vocab_max),
            n_regimes=max(1, int(contextual_n_regimes)),
        )
    contextual_problem = MarkovTwoSidedContextProblem(
        block_by_token=contextual_block_by_token,
        vocab_size=int(vocab_max),
        target_scale=float(target_scale),
    )
    use_contextual_sufficiency = contextual_sufficiency_active
    use_markov_node_witness = str(args.training_objective) == "markov_node_witness"
    use_markov_local_laws_fno = str(args.training_objective) == "markov_local_laws_fno"
    contextual_via_enable_flag = (
        bool(args.enable_contextual_sufficiency)
        and str(args.training_objective) != "contextual_sufficiency"
    )
    if use_contextual_sufficiency:
        suffix = " (additive auxiliary)" if contextual_via_enable_flag else ""
        log(
            "  contextual sufficiency objective enabled" + suffix + ": "
            f"problem={contextual_problem.problem_id} "
            f"context_kind={contextual_problem.context_kind} "
            f"samples_per_doc={int(args.context_samples_per_doc)} "
            f"contextual_weight={float(args.contextual_loss_weight):.3g} "
            f"infomax_weight={float(args.infomax_loss_weight):.3g} "
            f"dependence_objective={str(args.contextual_dependence_objective)} "
            f"signature_contexts={int(args.response_signature_contexts)} "
            f"signature_slices={int(args.response_signature_slices)}"
        )
    if use_markov_node_witness:
        log(
            "  Markov node witness objective enabled: "
            f"count_weight={float(args.markov_witness_count_weight):.3g} "
            f"edge_weight={float(args.markov_witness_edge_weight):.3g} "
            f"overall_weight={float(args.markov_witness_weight):.3g} "
            f"readout={str(args.markov_witness_readout)}"
        )
    if use_markov_local_laws_fno:
        log(
            "  Markov local-law FNO objective enabled: "
            f"leaf_weight={float(args.markov_law_leaf_weight):.3g} "
            f"merge_weight={float(args.markov_law_merge_weight):.3g} "
            f"idemp_weight={float(args.markov_law_idempotence_weight):.3g} "
            f"count_weight={float(args.markov_law_count_weight):.3g} "
            f"edge_weight={float(args.markov_law_edge_weight):.3g} "
            f"overall_weight={float(args.markov_law_weight):.3g} "
            f"readout={str(args.markov_law_readout)} "
            "c2_target=decoded_child_relational"
        )
    exact_witness: dict[str, dict[str, float | int]] = {}
    if int(args.exact_witness_n_regimes) > 0:
        exact_witness = {
            "train": _palette_block_exact_root_mae(
                train_docs,
                vocab_size=int(vocab_max),
                n_regimes=int(contextual_n_regimes),
                block_by_token=contextual_block_by_token,
            ),
            "val": _palette_block_exact_root_mae(
                val_docs,
                vocab_size=int(vocab_max),
                n_regimes=int(contextual_n_regimes),
                block_by_token=contextual_block_by_token,
            ),
            "test": _palette_block_exact_root_mae(
                test_docs,
                vocab_size=int(vocab_max),
                n_regimes=int(contextual_n_regimes),
                block_by_token=contextual_block_by_token,
            ),
        }
        log(
            "  exact palette-block witness root_mae: "
            f"train={exact_witness['train']['mae']:.6g} "
            f"val={exact_witness['val']['mae']:.6g} "
            f"test={exact_witness['test']['mae']:.6g}"
        )
    constant_baselines = _constant_baseline_diagnostics(
        train_docs=train_docs,
        val_docs=val_docs,
        test_docs=test_docs,
    )
    val_constant = dict(
        (dict(constant_baselines["splits"])["val"])  # type: ignore[index]
    )
    test_constant = dict(
        (dict(constant_baselines["splits"])["test"])  # type: ignore[index]
    )
    log(
        "  constant baselines: "
        f"val split-median MAE={val_constant['split_median_predictor']['root_mae']:.6g} "
        f"test split-median MAE={test_constant['split_median_predictor']['root_mae']:.6g}"
    )
    exact_surface_contract: dict[str, object] = {}
    if int(args.exact_witness_n_regimes) > 0:
        exact_surface_splits = {
            "train": _eval_exact_surface_contract(
                docs=train_docs,
                vocab_size=int(vocab_max),
                n_regimes=int(contextual_n_regimes),
                target_scale=float(target_scale),
                batch_size=int(args.batch_size),
                device=device,
            ),
            "val": _eval_exact_surface_contract(
                docs=val_docs,
                vocab_size=int(vocab_max),
                n_regimes=int(contextual_n_regimes),
                target_scale=float(target_scale),
                batch_size=int(args.batch_size),
                device=device,
            ),
            "test": _eval_exact_surface_contract(
                docs=test_docs,
                vocab_size=int(vocab_max),
                n_regimes=int(contextual_n_regimes),
                target_scale=float(target_scale),
                batch_size=int(args.batch_size),
                device=device,
            ),
        }
        exact_surface_contract = {
            "status": (
                "ok"
                if all(split.get("status") == "ok" for split in exact_surface_splits.values())
                else "partial_or_skipped"
            ),
            "splits": exact_surface_splits,
            "diagnostics": dict(exact_surface_splits["test"].get("diagnostics") or {}),
        }
        if exact_surface_splits["test"].get("status") == "ok":
            diag = dict(exact_surface_splits["test"].get("diagnostics") or {})
            log(
                "  exact g/f surface contract test_root_mae="
                f"{float(diag.get('root_mae', float('nan'))):.6g}"
            )
            exact_contract_root_mae = float(diag.get("root_mae", float("nan")))
            if (
                bool(args.require_exact_contract_zero)
                and exact_contract_root_mae > 1e-6
            ):
                raise RuntimeError(
                    "exact one-leaf g/f surface contract failed: "
                    f"root_mae={diag.get('root_mae')}"
                )
        else:
            log(
                "  exact g/f surface contract skipped: "
                f"{exact_surface_splits['test'].get('reason')}"
            )

    model = CleanUnifiedNO(
        vocab_size=int(vocab_max),
        target_scale=target_scale,
        channels=int(args.channels),
        g_n_modes=int(args.g_n_modes),
        g_n_layers=int(args.g_n_layers),
        scorer_n_modes=int(args.scorer_n_modes),
        scorer_n_layers=int(args.scorer_n_layers),
        pooling_mode=str(args.leaf_pool),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_emb_params = sum(p.numel() for p in model.token_embedding.parameters())
    n_g_params = sum(p.numel() for p in model.g.parameters())
    n_f_params = sum(p.numel() for p in model.f.parameters())
    log(f"  model params: total={n_params/1e6:.2f}M  "
        f"emb={n_emb_params/1e6:.2f}M  g(shared)={n_g_params/1e6:.2f}M  f={n_f_params/1e6:.2f}M")
    if n_g_params >= 10_000_000:
        log("  NOTE: shared g has >=10M params; treat this as a capacity diagnostic, "
            "not the default minimal-contract probe.")

    node_witness_head: _MarkovWitnessReadout | None = None
    if use_markov_node_witness or use_markov_local_laws_fno:
        readout_kind = (
            str(args.markov_law_readout)
            if use_markov_local_laws_fno
            else str(args.markov_witness_readout)
        )
        node_witness_head = _MarkovWitnessReadout(
            channels=int(model.channels),
            length=int(leaf_len),
            n_regimes=max(1, int(contextual_n_regimes)),
            kind=readout_kind,
        ).to(device)
        log(
            "  Markov decoded-law readout params="
            f"{sum(p.numel() for p in node_witness_head.parameters())} "
            f"type={readout_kind}"
        )

    signature_projector: nn.Module | None = None
    response_regressor: nn.Module | None = None
    response_slice_matrix: torch.Tensor | None = None
    if (
        float(args.infomax_loss_weight) > 0.0
        and str(args.contextual_dependence_objective)
        in {"infonce", "jsd", "dv", "wasserstein"}
    ):
        signature_projector = nn.Linear(
            int(args.response_signature_contexts),
            int(model.channels),
        ).to(device)
        log(
            "  dependence signature projector params="
            f"{sum(p.numel() for p in signature_projector.parameters())}"
        )
    if (
        float(args.infomax_loss_weight) > 0.0
        and str(args.contextual_dependence_objective) == "regression"
    ):
        regression_targets = (
            int(args.response_signature_slices)
            if int(args.response_signature_slices) > 0
            else int(args.response_signature_contexts)
        )
        response_regressor = _make_response_regressor(
            kind=str(args.contextual_response_regressor),
            channels=int(model.channels),
            length=int(leaf_len),
            out_dim=int(regression_targets),
        ).to(device)
        if int(args.response_signature_slices) > 0:
            response_slice_matrix = _random_unit_slices(
                n_contexts=int(args.response_signature_contexts),
                n_slices=int(args.response_signature_slices),
                device=device,
            ).detach()
        log(
            "  dependence response regressor params="
            f"{sum(p.numel() for p in response_regressor.parameters())} "
            f"type={str(args.contextual_response_regressor)}"
        )
        if response_slice_matrix is not None:
            log(
                "  dependence response slice matrix shape="
                f"{tuple(response_slice_matrix.shape)}"
            )
    trainable_params = list(model.parameters())
    if node_witness_head is not None:
        trainable_params += list(node_witness_head.parameters())
    if signature_projector is not None:
        trainable_params += list(signature_projector.parameters())
    if response_regressor is not None:
        trainable_params += list(response_regressor.parameters())

    if str(args.optimizer) == "adamw":
        optimizer = optim.AdamW(
            trainable_params,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
    else:
        optimizer = optim.Adam(trainable_params, lr=float(args.lr))
    scheduler = None
    if str(args.lr_schedule) == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(args.epochs)),
        )

    train_arrays = _doc_arrays(train_docs)
    flat_train_docs = [_flat_doc_tokens(doc) for doc in train_docs]
    contextual_rng = np.random.default_rng(int(args.seed) + 70_001)
    val_root_mae = _eval_root_mae(model, val_docs, batch_size=int(args.batch_size), device=device)
    log(f"epoch  0: val_root_mae = {val_root_mae:.4f} (untrained baseline)")

    best_val = float(val_root_mae)
    best_epoch = 0
    best_state = _clone_checkpoint_state(model)
    best_node_witness_state = (
        _clone_checkpoint_state(node_witness_head)
        if node_witness_head is not None
        else None
    )
    history = [{"epoch": 0, "val_root_mae": val_root_mae}]
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        if signature_projector is not None:
            signature_projector.train()
        if response_regressor is not None:
            response_regressor.train()
        t_epoch = time.time()
        n_train = len(train_docs)
        order = torch.randperm(n_train).tolist()
        running_loss = 0.0
        running_root_loss = 0.0
        running_contextual_loss = 0.0
        running_infomax_loss = 0.0
        running_node_witness_loss = 0.0
        running_node_leaf_witness_loss = 0.0
        running_node_merge_witness_loss = 0.0
        running_node_witness_count_loss = 0.0
        running_node_witness_first_ce = 0.0
        running_node_witness_last_ce = 0.0
        running_markov_law_loss = 0.0
        running_markov_law_leaf_loss = 0.0
        running_markov_law_merge_loss = 0.0
        running_markov_law_idempotence_loss = 0.0
        running_markov_law_count_loss = 0.0
        running_markov_law_first_ce = 0.0
        running_markov_law_last_ce = 0.0
        running_contextual_queries = 0
        n_batches = 0
        for bi in range(0, n_train, int(args.batch_size)):
            optimizer.zero_grad()
            batch_indices = order[bi : bi + int(args.batch_size)]
            losses = []
            root_terms = []
            node_witness_terms = []
            node_leaf_witness_terms = []
            node_merge_witness_terms = []
            node_count_terms = []
            node_first_terms = []
            node_last_terms = []
            law_terms = []
            law_leaf_terms = []
            law_merge_terms = []
            law_idempotence_terms = []
            law_count_terms = []
            law_first_terms = []
            law_last_terms = []
            for di in batch_indices:
                tok = _to_tensor(train_arrays[0][di], dtype=torch.long, device=device)
                out = model(tok)
                root_t = _to_tensor(train_arrays[3][di], dtype=torch.float32, device=device)
                root_loss = root_mse_loss(out, root_count=root_t, target_scale=model.target_scale)
                root_terms.append(root_loss.detach())
                loss = root_loss
                if (
                    not args.root_only
                    and not use_markov_node_witness
                    and not use_markov_local_laws_fno
                    and len(train_arrays[1][di]) > 0
                ):
                    leaf_t = _to_tensor(train_arrays[1][di], dtype=torch.float32, device=device)
                    loss = loss + leaf_mse_loss(
                        out, leaf_counts=leaf_t, target_scale=model.target_scale
                    )
                if (
                    not args.root_only
                    and not use_markov_node_witness
                    and not use_markov_local_laws_fno
                    and len(train_arrays[2][di]) > 0
                ):
                    mg_t = _to_tensor(train_arrays[2][di], dtype=torch.float32, device=device)
                    loss = loss + merge_mse_loss(
                        out, merge_counts=mg_t, target_scale=model.target_scale
                    )
                if use_markov_node_witness:
                    assert node_witness_head is not None
                    witness_loss, witness_parts = _markov_node_witness_loss(
                        witness_head=node_witness_head,
                        output=out,
                        leaves=train_arrays[0][di],
                        block_by_token=contextual_block_by_token,
                        target_scale=float(model.target_scale),
                        count_weight=float(args.markov_witness_count_weight),
                        edge_weight=float(args.markov_witness_edge_weight),
                        device=device,
                    )
                    loss = loss + float(args.markov_witness_weight) * witness_loss
                    node_witness_terms.append(witness_loss.detach())
                    node_leaf_witness_terms.append(witness_parts["leaf_total"].detach())
                    node_merge_witness_terms.append(witness_parts["merge_total"].detach())
                    node_count_terms.append(witness_parts["count"].detach())
                    node_first_terms.append(witness_parts["first"].detach())
                    node_last_terms.append(witness_parts["last"].detach())
                if use_markov_local_laws_fno:
                    assert node_witness_head is not None
                    law_loss, law_parts = _markov_local_law_fno_loss(
                        law_head=node_witness_head,
                        output=out,
                        leaves=train_arrays[0][di],
                        block_by_token=contextual_block_by_token,
                        target_scale=float(model.target_scale),
                        leaf_weight=float(args.markov_law_leaf_weight),
                        merge_weight=float(args.markov_law_merge_weight),
                        idempotence_weight=float(args.markov_law_idempotence_weight),
                        count_weight=float(args.markov_law_count_weight),
                        edge_weight=float(args.markov_law_edge_weight),
                        device=device,
                    )
                    loss = loss + float(args.markov_law_weight) * law_loss
                    law_terms.append(law_loss.detach())
                    law_leaf_terms.append(law_parts["leaf_total"].detach())
                    law_merge_terms.append(law_parts["merge_total"].detach())
                    law_idempotence_terms.append(law_parts["idempotence"].detach())
                    law_count_terms.append(law_parts["count"].detach())
                    law_first_terms.append(law_parts["first"].detach())
                    law_last_terms.append(law_parts["last"].detach())
                losses.append(loss)
            supervised_batch_loss = torch.stack(losses).mean()
            root_batch_loss = torch.stack(root_terms).mean()
            contextual_loss = torch.zeros((), dtype=torch.float32, device=device)
            infomax_loss = torch.zeros((), dtype=torch.float32, device=device)
            n_context_queries = 0
            if use_contextual_sufficiency:
                contextual_loss, infomax_loss, n_context_queries = (
                    _contextual_sufficiency_batch_losses(
                        model=model,
                        flat_train_docs=flat_train_docs,
                        batch_indices=batch_indices,
                        block_by_token=contextual_block_by_token,
                        target_scale=float(model.target_scale),
                        samples_per_doc=int(args.context_samples_per_doc),
                        fragment_len=int(leaf_len),
                        rng=contextual_rng,
                        device=device,
                        signature_projector=signature_projector,
                        response_regressor=response_regressor,
                        response_slice_matrix=response_slice_matrix,
                        response_signature_contexts=int(args.response_signature_contexts),
                        response_signature_slices=int(args.response_signature_slices),
                        dependence_objective=str(args.contextual_dependence_objective),
                        contextual_problem=contextual_problem,
                    )
                )
            batch_loss = (
                supervised_batch_loss
                + float(args.contextual_loss_weight) * contextual_loss
                + float(args.infomax_loss_weight) * infomax_loss
            )
            batch_loss.backward()
            if float(args.grad_clip) > 0:
                nn.utils.clip_grad_norm_(trainable_params, float(args.grad_clip))
            optimizer.step()
            running_loss += float(batch_loss.detach().cpu())
            running_root_loss += float(root_batch_loss.detach().cpu())
            running_contextual_loss += float(contextual_loss.detach().cpu())
            running_infomax_loss += float(infomax_loss.detach().cpu())
            if node_witness_terms:
                running_node_witness_loss += float(torch.stack(node_witness_terms).mean().cpu())
                running_node_leaf_witness_loss += float(
                    torch.stack(node_leaf_witness_terms).mean().cpu()
                )
                running_node_merge_witness_loss += float(
                    torch.stack(node_merge_witness_terms).mean().cpu()
                )
                running_node_witness_count_loss += float(torch.stack(node_count_terms).mean().cpu())
                running_node_witness_first_ce += float(torch.stack(node_first_terms).mean().cpu())
                running_node_witness_last_ce += float(torch.stack(node_last_terms).mean().cpu())
            if law_terms:
                running_markov_law_loss += float(torch.stack(law_terms).mean().cpu())
                running_markov_law_leaf_loss += float(torch.stack(law_leaf_terms).mean().cpu())
                running_markov_law_merge_loss += float(torch.stack(law_merge_terms).mean().cpu())
                running_markov_law_idempotence_loss += float(
                    torch.stack(law_idempotence_terms).mean().cpu()
                )
                running_markov_law_count_loss += float(torch.stack(law_count_terms).mean().cpu())
                running_markov_law_first_ce += float(torch.stack(law_first_terms).mean().cpu())
                running_markov_law_last_ce += float(torch.stack(law_last_terms).mean().cpu())
            running_contextual_queries += int(n_context_queries)
            n_batches += 1
        if scheduler is not None:
            scheduler.step()
        train_loss = running_loss / max(1, n_batches)
        train_root_loss = running_root_loss / max(1, n_batches)
        train_contextual_loss = running_contextual_loss / max(1, n_batches)
        train_infomax_loss = running_infomax_loss / max(1, n_batches)
        train_node_witness_loss = running_node_witness_loss / max(1, n_batches)
        train_node_leaf_witness_loss = running_node_leaf_witness_loss / max(1, n_batches)
        train_node_merge_witness_loss = running_node_merge_witness_loss / max(1, n_batches)
        train_node_witness_count_loss = running_node_witness_count_loss / max(1, n_batches)
        train_node_witness_first_ce = running_node_witness_first_ce / max(1, n_batches)
        train_node_witness_last_ce = running_node_witness_last_ce / max(1, n_batches)
        train_markov_law_loss = running_markov_law_loss / max(1, n_batches)
        train_markov_law_leaf_loss = running_markov_law_leaf_loss / max(1, n_batches)
        train_markov_law_merge_loss = running_markov_law_merge_loss / max(1, n_batches)
        train_markov_law_idempotence_loss = (
            running_markov_law_idempotence_loss / max(1, n_batches)
        )
        train_markov_law_count_loss = running_markov_law_count_loss / max(1, n_batches)
        train_markov_law_first_ce = running_markov_law_first_ce / max(1, n_batches)
        train_markov_law_last_ce = running_markov_law_last_ce / max(1, n_batches)
        val_root_mae = _eval_root_mae(model, val_docs, batch_size=int(args.batch_size), device=device)
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_root_loss": train_root_loss,
            "train_contextual_loss": train_contextual_loss,
            "train_infomax_loss": train_infomax_loss,
            "train_markov_node_witness_loss": train_node_witness_loss,
            "train_markov_node_leaf_witness_loss": train_node_leaf_witness_loss,
            "train_markov_node_merge_witness_loss": train_node_merge_witness_loss,
            "train_markov_node_witness_count_loss": train_node_witness_count_loss,
            "train_markov_node_witness_first_ce": train_node_witness_first_ce,
            "train_markov_node_witness_last_ce": train_node_witness_last_ce,
            "train_markov_local_law_loss": train_markov_law_loss,
            "train_markov_local_law_leaf_loss": train_markov_law_leaf_loss,
            "train_markov_local_law_merge_loss": train_markov_law_merge_loss,
            "train_markov_local_law_idempotence_loss": train_markov_law_idempotence_loss,
            "train_markov_local_law_count_loss": train_markov_law_count_loss,
            "train_markov_local_law_first_ce": train_markov_law_first_ce,
            "train_markov_local_law_last_ce": train_markov_law_last_ce,
            "train_contextual_queries": int(running_contextual_queries),
            "val_root_mae": val_root_mae,
            "lr": current_lr,
        }
        if use_contextual_sufficiency:
            n_val_context_queries = max(
                1,
                min(32, int(args.context_samples_per_doc) * min(32, len(val_docs))),
            )
            val_context_diag = _eval_contextual_sufficiency_diagnostics(
                model,
                val_docs,
                n_queries=int(n_val_context_queries),
                block_by_token=contextual_block_by_token,
                target_scale=float(model.target_scale),
                fragment_len=int(leaf_len),
                device=device,
                seed=int(args.seed) + 80_000 + int(epoch),
            )
            val_context_summary = dict(val_context_diag.get("diagnostics") or {})
            row["val_contextual_mae"] = float(
                val_context_summary.get("root_mae", float("nan"))
            )
            row["val_contextual_corr"] = float(
                val_context_summary.get("pred_truth_corr", float("nan"))
            )
        history.append(row)
        if val_root_mae < best_val:
            best_val = val_root_mae
            best_epoch = epoch
            best_state = _clone_checkpoint_state(model)
            if node_witness_head is not None:
                best_node_witness_state = _clone_checkpoint_state(node_witness_head)
        msg = (
            f"epoch {epoch:2d}: train_loss={train_loss:.4f}  "
            f"val_root_mae={val_root_mae:.4f}  best={best_val:.4f}@ep{best_epoch}  "
            f"lr={current_lr:.2e}"
        )
        if use_contextual_sufficiency:
            msg += (
                f"  ctx_loss={train_contextual_loss:.4f}"
                f"  ctx_val_mae={row.get('val_contextual_mae', float('nan')):.4f}"
            )
            if float(args.infomax_loss_weight) > 0.0:
                msg += f"  infomax={train_infomax_loss:.4f}"
        if use_markov_node_witness:
            msg += (
                f"  node_witness={train_node_witness_loss:.4f}"
                f"  leaf={train_node_leaf_witness_loss:.4f}"
                f"  merge={train_node_merge_witness_loss:.4f}"
            )
        if use_markov_local_laws_fno:
            msg += (
                f"  local_law={train_markov_law_loss:.4f}"
                f"  leaf={train_markov_law_leaf_loss:.4f}"
                f"  merge_rel={train_markov_law_merge_loss:.4f}"
                f"  idemp={train_markov_law_idempotence_loss:.4f}"
            )
        log(msg + f"  ({time.time() - t_epoch:.1f}s)")

    log(f"loading best checkpoint (epoch {best_epoch}, val_root_mae={best_val:.4f}) for test eval ...")
    model.load_state_dict(best_state)
    if node_witness_head is not None and best_node_witness_state is not None:
        node_witness_head.load_state_dict(best_node_witness_state)
    learned_prediction_diagnostics = {
        "train": _eval_model_root_diagnostics(
            model,
            train_docs,
            batch_size=int(args.batch_size),
            device=device,
        ),
        "val": _eval_model_root_diagnostics(
            model,
            val_docs,
            batch_size=int(args.batch_size),
            device=device,
        ),
        "test": _eval_model_root_diagnostics(
            model,
            test_docs,
            batch_size=int(args.batch_size),
            device=device,
        ),
    }
    test_root_mae = float(learned_prediction_diagnostics["test"]["root_mae"])
    log(f"FINAL: test_root_mae = {test_root_mae:.4f}  "
        f"best_val_root_mae = {best_val:.4f} @ epoch {best_epoch}")
    log(
        "FINAL diagnostics: "
        f"test_pred_std={learned_prediction_diagnostics['test']['pred_std']:.6g} "
        f"test_corr={learned_prediction_diagnostics['test']['pred_truth_corr']:.6g}"
    )
    contextual_sufficiency_diagnostics: dict[str, object] = {
        "status": "not_requested",
        "reason": "training_objective=root and context_samples_per_doc=0",
    }
    if use_contextual_sufficiency or int(args.context_samples_per_doc) > 0:
        def _context_eval_queries(n_docs: int) -> int:
            samples = max(1, int(args.context_samples_per_doc))
            return max(1, min(128, samples * min(64, int(n_docs))))

        contextual_sufficiency_diagnostics = {
            "status": "ok",
            "n_regimes_for_exact_oracle": int(contextual_n_regimes),
            "splits": {
                "train": _eval_contextual_sufficiency_diagnostics(
                    model,
                    train_docs,
                    n_queries=_context_eval_queries(len(train_docs)),
                    block_by_token=contextual_block_by_token,
                    target_scale=float(model.target_scale),
                    fragment_len=int(leaf_len),
                    device=device,
                    seed=int(args.seed) + 81_000,
                ),
                "val": _eval_contextual_sufficiency_diagnostics(
                    model,
                    val_docs,
                    n_queries=_context_eval_queries(len(val_docs)),
                    block_by_token=contextual_block_by_token,
                    target_scale=float(model.target_scale),
                    fragment_len=int(leaf_len),
                    device=device,
                    seed=int(args.seed) + 82_000,
                ),
                "test": _eval_contextual_sufficiency_diagnostics(
                    model,
                    test_docs,
                    n_queries=_context_eval_queries(len(test_docs)),
                    block_by_token=contextual_block_by_token,
                    target_scale=float(model.target_scale),
                    fragment_len=int(leaf_len),
                    device=device,
                    seed=int(args.seed) + 83_000,
                ),
            },
        }
        test_context_diag = dict(
            dict(contextual_sufficiency_diagnostics["splits"])["test"].get(
                "diagnostics",
                {},
            )
        )
        log(
            "FINAL contextual diagnostics: "
            f"test_context_mae={float(test_context_diag.get('root_mae', float('nan'))):.6g} "
            f"test_context_corr={float(test_context_diag.get('pred_truth_corr', float('nan'))):.6g}"
        )

    markov_node_witness_diagnostics: dict[str, object] = {"status": "not_requested"}
    markov_local_law_fno_diagnostics: dict[str, object] = {"status": "not_requested"}
    if node_witness_head is not None:
        decoded_splits = {
            "train": _eval_markov_node_witness_head(
                model=model,
                witness_head=node_witness_head,
                docs=train_docs,
                block_by_token=contextual_block_by_token,
                target_scale=float(model.target_scale),
                batch_size=int(args.batch_size),
                device=device,
            ),
            "val": _eval_markov_node_witness_head(
                model=model,
                witness_head=node_witness_head,
                docs=val_docs,
                block_by_token=contextual_block_by_token,
                target_scale=float(model.target_scale),
                batch_size=int(args.batch_size),
                device=device,
            ),
            "test": _eval_markov_node_witness_head(
                model=model,
                witness_head=node_witness_head,
                docs=test_docs,
                block_by_token=contextual_block_by_token,
                target_scale=float(model.target_scale),
                batch_size=int(args.batch_size),
                device=device,
            ),
        }
    if node_witness_head is not None and use_markov_node_witness:
        markov_node_witness_diagnostics = {
            "status": "ok",
            "supervision": "root_mse_plus_leaf_and_merge_markov_witness",
            "markov_witness_weight": float(args.markov_witness_weight),
            "markov_witness_count_weight": float(args.markov_witness_count_weight),
            "markov_witness_edge_weight": float(args.markov_witness_edge_weight),
            "markov_witness_readout": str(args.markov_witness_readout),
            "splits": decoded_splits,
        }
        test_node_diag = dict(
            dict(markov_node_witness_diagnostics["splits"])["test"]
        )
        test_leaf_diag = dict(test_node_diag.get("leaf") or {})
        test_merge_diag = dict(test_node_diag.get("merge") or {})
        log(
            "FINAL Markov node witness diagnostics: "
            f"leaf_full={float(test_leaf_diag.get('full_witness_exact_rate', float('nan'))):.4f} "
            f"merge_full={float(test_merge_diag.get('full_witness_exact_rate', float('nan'))):.4f}"
        )
    if node_witness_head is not None and use_markov_local_laws_fno:
        markov_local_law_fno_diagnostics = {
            "status": "ok",
            "supervision": "root_mse_plus_decoded_leaf_law_and_relational_merge_law",
            "c2_merge_target": "decoded_child_relational",
            "note": (
                "Leaf sketches are supervised; merge diagnostics are exact-target "
                "eval only. The training loss does not regress merge states to "
                "exact merge-node labels."
            ),
            "markov_law_weight": float(args.markov_law_weight),
            "markov_law_leaf_weight": float(args.markov_law_leaf_weight),
            "markov_law_merge_weight": float(args.markov_law_merge_weight),
            "markov_law_idempotence_weight": float(args.markov_law_idempotence_weight),
            "markov_law_count_weight": float(args.markov_law_count_weight),
            "markov_law_edge_weight": float(args.markov_law_edge_weight),
            "markov_law_readout": str(args.markov_law_readout),
            "splits": decoded_splits,
        }
        test_law_diag = dict(decoded_splits["test"])
        test_leaf_diag = dict(test_law_diag.get("leaf") or {})
        test_merge_diag = dict(test_law_diag.get("merge") or {})
        log(
            "FINAL Markov local-law FNO diagnostics: "
            f"leaf_theta_mae={float(test_leaf_diag.get('theta_mae', float('nan'))):.6g} "
            f"merge_theta_mae={float(test_merge_diag.get('theta_mae', float('nan'))):.6g} "
            f"leaf_first={float(test_leaf_diag.get('theta_first_regime_accuracy', float('nan'))):.4f} "
            f"merge_first={float(test_merge_diag.get('theta_first_regime_accuracy', float('nan'))):.4f}"
        )

    diagnostic_baselines = _run_controlled_diagnostic_baselines(
        requested=_parse_diagnostic_baselines(str(args.diagnostic_baselines)),
        train_docs=train_docs,
        val_docs=val_docs,
        test_docs=test_docs,
        args=args,
        vocab_size=int(vocab_max),
        n_regimes=max(1, int(contextual_n_regimes)),
        device=device,
        log=log,
    )
    boundary_supervision_ablation: dict[str, object] = {"status": "not_requested"}
    if bool(args.run_boundary_supervision_ablation):
        boundary_supervision_ablation = _run_boundary_supervision_ablation(
            train_docs=train_docs,
            val_docs=val_docs,
            test_docs=test_docs,
            args=args,
            vocab_size=int(vocab_max),
            target_scale=float(target_scale),
            n_regimes=max(1, int(contextual_n_regimes)),
            device=device,
            log=log,
        )
    markov_witness_supervision_ablation: dict[str, object] = {"status": "not_requested"}
    if bool(args.run_markov_witness_supervision_ablation):
        markov_witness_supervision_ablation = _run_markov_witness_supervision_ablation(
            train_docs=train_docs,
            val_docs=val_docs,
            test_docs=test_docs,
            args=args,
            vocab_size=int(vocab_max),
            target_scale=float(target_scale),
            n_regimes=max(1, int(contextual_n_regimes)),
            device=device,
            log=log,
        )

    summary = {
        "args": vars(args),
        "n_leaves_per_doc": n_leaves_per_doc,
        "leaf_len": leaf_len,
        "vocab_max": vocab_max,
        "target_scale": target_scale,
        "generation_meta": generation_meta,
        "exact_palette_block_witness": exact_witness,
        "exact_surface_contract": exact_surface_contract,
        "constant_baselines": constant_baselines,
        "learned_prediction_diagnostics": learned_prediction_diagnostics,
        "contextual_sufficiency_diagnostics": contextual_sufficiency_diagnostics,
        "contextual_query_problem": {
            "problem_id": str(contextual_problem.problem_id),
            "context_kind": str(contextual_problem.context_kind),
            "theorem_object": "query: Ctx -> X -> Y",
        },
        "sbijax_objective_provenance": contextual_sbijax_provenance(
            method=(
                "nasss"
                if str(args.contextual_dependence_objective) == "regression"
                else "nass"
            ),
            response_signature_contexts=int(max(0, args.response_signature_contexts)),
            response_signature_slices=int(max(0, args.response_signature_slices)),
        ),
        "diagnostic_baselines": diagnostic_baselines,
        "markov_node_witness_diagnostics": markov_node_witness_diagnostics,
        "markov_local_law_fno_diagnostics": markov_local_law_fno_diagnostics,
        "boundary_supervision_ablation": boundary_supervision_ablation,
        "markov_witness_supervision_ablation": markov_witness_supervision_ablation,
        "n_params_total": n_params,
        "n_params_token_embedding": n_emb_params,
        "n_params_g": n_g_params,
        "n_params_f": n_f_params,
        "history": history,
        "test_root_mae": test_root_mae,
        "best_val_root_mae": best_val,
        "best_val_epoch": best_epoch,
    }
    with open(out_root / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"summary written to {out_root}/summary.json")


if __name__ == "__main__":
    main()
