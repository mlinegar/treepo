"""
Neural operator baselines for the Markov changepoint-count task.

Provides four baseline models for diagnosing why the custom CTreePO tree-merge
operator underperforms simple ridge regression on bigram features:

1. FNOCountPredictor       -- Official neuraloperator package FNO (1D)
2. DeepONetCountPredictor  -- Branch-net style DeepONet for scalar output
3. MLPBigramCountPredictor -- MLP on the same bigram features ridge uses
4. CNN1DCountPredictor     -- 1D CNN with kernel_size=2 (transition detector)

Each model has a corresponding ``_fit_*_baseline`` function that follows the
same signature and return convention as ``_fit_doc_sequence_baseline`` in
``markov_changepoint_ops_count.py``.
"""

from __future__ import annotations

import ctypes
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import lru_cache
import gc
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from treepo._research.core.autotune_probe_cache import (
    AUTOTUNE_PROBE_CACHE_VERSION,
    ProbeCacheEntry,
    ProbeCacheStore,
    ProbeCandidateProfile,
    ProbeRunProfile,
    build_probe_cache_key,
    classify_device_signature,
)
from treepo._research.core.unified_runtime import (
    BatchTelemetry,
    GPU_RUNTIME_BUCKET_MODE_EXACT_THEN_BUCKETED,
    GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE,
    GPU_RUNTIME_DATA_MODE_CPU_DEBUG,
    RUNTIME_MODE_UNIFIED_V2,
    GpuBatchStore,
    GpuBatchStoreKey,
    GpuBatchView,
    GpuRuntimeConfig,
    GpuRuntimeTelemetry,
    WorkItem,
    build_leaf_count_auto_queue_targets,
    gpu_runtime_config_from_mapping,
    get_named_plan_cache,
    normalize_gpu_runtime_bucket_mode,
    plan_work_batches,
    resolve_runtime_mode,
)

from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (
    ChangepointMarkovDoc,
    ObjectiveMetrics,
    OPSCountConfig,
    SketchMetrics,
    TrainFitDiagnostics,
    VALID_INTERNAL_SUPERVISION_KINDS,
    VALID_LEAF_SUPERVISION_KINDS,
    VALID_TREE_CHECKPOINT_METRICS,
    VALID_TREE_DOCUMENT_LOSS_NORMALIZATION_MODES,
    VALID_TREE_LOCAL_WEIGHTING_MODES,
    VALID_TREE_ROOT_SUPERVISION_KINDS,
    VALID_TREE_SUMMARY_SPEC_ROOT_MODES,
    VALID_TREE_THEOREM_COUNT_HEAD_MODES,
    VALID_TREE_THEOREM_SURFACE_MODES,
    VALID_TREE_SCORE_MERGE_MODES,
    VALID_TREE_TRAINING_SCHEDULES,
    VALID_SCHEDULES,
    _eval_root_predictions,
    _exact_match_rate,
    _predict_count_from_summary,
    _resummary_summary_sequence,
    _set_global_seed,
    _token_sequence_arrays,
    _zero_sketch_metrics,
)
from treepo._research.ctreepo.sim.core.theorem_feature_route import (
    DEFAULT_THEOREM_FEATURE_ADAPTER,
    TheoremFeatureAdapter,
    build_theorem_feature_pair_sets,
    load_theorem_feature_stage1_artifact,
    resolve_theorem_feature_adapter,
    theorem_feature_pair_metrics_from_scores,
    theorem_feature_targets_from_markov_exact_targets,
    write_theorem_feature_stage1_artifact,
)
from treepo._research.ctreepo.sim.core.oracle_metric import (
    OracleMetricSpace,
    build_contrastive_pairs,
    contrastive_fiber_loss,
)
from treepo._research.ctreepo.sim.core.markov_theorem_feature_adapter import (
    COARSENED_THEOREM_FEATURE_ADAPTER,
    SCOREFIBER_MARKOV_ENDPOINTS_ADAPTER,
    MarkovTheoremFeatureLabel,
    ScoreFiberTheoremFeatureLabel,
)
from treepo._research.ctreepo.sim.core.training_selection import (
    TrainingSelectionMetadata,
    clone_module_state,
    improved_metric,
    restore_module_state,
)
from treepo._research.tree.full_tree_ipw import (
    DocumentLevelPredictionRecord,
    FullTreeIPWSummaryAccumulator,
    FullTreeNodeRecord,
    summarize_full_tree_ipw,
)
from treepo._research.tree.ipw import NodeType
from treepo._research.tree.tree_model_v2 import (
    TreeModelV2View,
    normalize_tree_model_version,
)

# ---------------------------------------------------------------------------
# Optional neuraloperator import
# ---------------------------------------------------------------------------

try:
    from neuralop.models import FNO as _NeuralOpFNO

    HAS_NEURAL_OPERATOR = True
except ImportError:
    _NeuralOpFNO = None  # type: ignore[assignment,misc]
    HAS_NEURAL_OPERATOR = False

_INSTALL_MSG = (
    "The neuraloperator package is required for FNO/DeepONet baselines. "
    "Install it with: uv add neuraloperator"
)
_CUDA_FAST_MATH_CONFIGURED = False
FNO_TREE_C2_METRIC_KIND = "count_drift"
FNO_TREE_C2_PROXY_METRIC_KIND = "state_replay_mse"
FNO_TREE_C2_EXACT_WITNESS_KIND = "on_range_exact_match"
DECODED_MARKOV_SKETCH_SURFACE = "decoded_markov_sketch"
MARKOV_COUNT_SKETCH_SUMMARY_SPEC = "markov_count_sketch"
VALID_MARKOV_MERGE_OBJECTIVE_MODES = (
    "strict_c3",
    "teacher_parent_count",
    "teacher_parent_full_sketch",
)
VALID_MARKOV_MERGE_WEIGHTING_MODES = (
    "flat_mean",
    "depth_balanced",
)

# ---------------------------------------------------------------------------
# Shared FNO token encoder (used by both FNOCountPredictor and FNOCountSketch)
# ---------------------------------------------------------------------------


def apply_fno_token_encoder(
    tokens: torch.Tensor,
    *,
    token_mask: torch.Tensor,
    token_embedding: nn.Embedding,
    fno: nn.Module,
    pooling_mode: str = "mean",
    input_proj: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode tokens through embedding -> optional projection -> FNO -> masked pool.

    Single source of truth for the embed/FNO/pool pipeline. Both
    ``FNOTokenEncoder.forward`` and ``FNOCountSketch._encode_token_batch``
    call this with their own (token_embedding, fno) modules so the encode
    sequence stays a thin wrapper around the official neuralop FNO.

    Returns ``(fno_output, pooled)`` where ``fno_output`` is (B, width, L)
    channel-first FNO output and ``pooled`` is (B, width).
    """
    emb = token_embedding(tokens)  # (B, L, embed_dim)
    if input_proj is not None:
        emb = input_proj(emb)  # (B, L, width)
    x = emb.permute(0, 2, 1)  # (B, width, L)
    x = fno(x)  # (B, width, L)
    mask = token_mask.unsqueeze(1)  # (B, 1, L)
    pooled = (x * mask).sum(dim=-1)
    if pooling_mode == "mean":
        pooled = pooled / mask.sum(dim=-1).clamp(min=1)
    elif pooling_mode != "sum":
        raise ValueError(
            f"unsupported FNO pooling_mode={pooling_mode!r}; expected 'mean' or 'sum'"
        )
    return x, pooled


class FNOTokenEncoder(nn.Module):
    """Shared: embed tokens -> optional projection -> 1D FNO -> masked pool.

    Both FNOCountPredictor (standalone) and FNOCountSketch (tree leaf encoder)
    compose this module so FNO parity is structural, not a numerical coincidence.

    When ``embed_dim == width`` and ``input_proj`` is None (the default),
    tokens are embedded directly at FNO width — matching FNOCountSketch's
    original behavior. When ``embed_dim != width``, a linear projection is
    inserted — matching FNOCountPredictor's original behavior.

    Returns:
        ``(fno_output, pooled)`` where ``fno_output`` is (B, width, L)
        channel-first FNO output and ``pooled`` is (B, width) mean-pooled.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        width: int,
        n_modes: int,
        n_layers: int,
        embed_dim: int | None = None,
        pooling_mode: str = "mean",
    ) -> None:
        super().__init__()
        if _NeuralOpFNO is None:
            raise ImportError(_INSTALL_MSG)
        resolved_embed_dim = int(embed_dim) if embed_dim is not None else int(width)
        self.pad_id = int(vocab_size)
        self.width = int(width)
        normalized_pooling = str(pooling_mode or "mean").strip().lower() or "mean"
        if normalized_pooling not in {"mean", "sum"}:
            raise ValueError(
                f"unsupported FNO pooling_mode={pooling_mode!r}; expected 'mean' or 'sum'"
            )
        self.pooling_mode = normalized_pooling
        self.token_embedding = nn.Embedding(
            int(vocab_size) + 1, resolved_embed_dim, padding_idx=self.pad_id
        )
        self.input_proj: nn.Module | None = None
        if resolved_embed_dim != int(width):
            self.input_proj = nn.Linear(resolved_embed_dim, int(width))
        self.fno = _NeuralOpFNO(
            n_modes=(int(n_modes),),
            in_channels=int(width),
            out_channels=int(width),
            hidden_channels=int(width),
            n_layers=int(n_layers),
        )

    def forward(
        self, tokens: torch.Tensor, *, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_fno_token_encoder(
            tokens,
            token_mask=token_mask,
            token_embedding=self.token_embedding,
            fno=self.fno,
            pooling_mode=self.pooling_mode,
            input_proj=self.input_proj,
        )


# ---------------------------------------------------------------------------
# Model 1: FNO baseline
# ---------------------------------------------------------------------------


class FNOCountPredictor(nn.Module):
    """FNO-based changepoint count predictor.

    Embeds discrete tokens, projects to FNO channel width, applies 1D FNO
    layers, then does masked mean pooling followed by a classification head.

    Note: FNO's Fourier-mode inductive bias is designed for smooth/periodic
    continuous functions, not sharp discrete changepoints.  This is intentional
    — the comparison shows whether the general-purpose spectral architecture
    offers any advantage on this task.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int,
        n_modes: int,
        width: int,
        n_layers: int,
        n_count_classes: int,
        pooling_mode: str = "mean",
        concat_normalized_length: bool = False,
        include_transition_channel: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = FNOTokenEncoder(
            vocab_size=int(vocab_size),
            width=int(width),
            n_modes=int(n_modes),
            n_layers=int(n_layers),
            embed_dim=int(embed_dim),
            pooling_mode=str(pooling_mode),
        )
        self.pad_id = self.encoder.pad_id
        self.pooling_mode = self.encoder.pooling_mode
        self.concat_normalized_length = bool(concat_normalized_length)
        self.include_transition_channel = bool(include_transition_channel)
        self.pair_vocab_size = int(vocab_size) + 1
        self.transition_embedding: nn.Embedding | None = None
        if self.include_transition_channel:
            self.transition_embedding = nn.Embedding(
                int(self.pair_vocab_size) * int(self.pair_vocab_size),
                int(width),
            )
        head_hidden = max(32, int(width) // 2)
        classifier_in = int(width) + (1 if self.concat_normalized_length else 0)
        self.count_classifier = nn.Sequential(
            nn.Linear(int(classifier_in), head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, int(n_count_classes)),
        )

    def forward(
        self, tokens: torch.Tensor, *, token_mask: torch.Tensor
    ) -> torch.Tensor:
        pooled = self.encode_representation(tokens, token_mask=token_mask)
        return self.count_classifier(pooled)  # (B, n_count_classes)

    def encode_representation(
        self,
        tokens: torch.Tensor,
        *,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Transition channel is applied pre-FNO if configured.
        if self.transition_embedding is not None:
            emb = self.encoder.token_embedding(tokens)
            if self.encoder.input_proj is not None:
                emb = self.encoder.input_proj(emb)
            next_tokens = torch.full_like(tokens, fill_value=self.pad_id)
            if int(tokens.shape[1]) > 1:
                next_tokens[:, :-1] = tokens[:, 1:]
            pair_ids = tokens * int(self.pair_vocab_size) + next_tokens
            pair_mask = token_mask * F.pad(token_mask[:, 1:], (0, 1), value=0.0)
            emb = emb + self.transition_embedding(pair_ids) * pair_mask.unsqueeze(-1)
            x = emb.permute(0, 2, 1)
            x = self.encoder.fno(x)
            mask = token_mask.unsqueeze(1)
            pooled = (x * mask).sum(dim=-1)
            valid_lengths = mask.sum(dim=-1).clamp(min=1)
            if self.pooling_mode == "mean":
                pooled = pooled / valid_lengths
        else:
            _, pooled = self.encoder(tokens, token_mask=token_mask)
            if self.pooling_mode == "mean":
                pass  # encoder already mean-pooled
            mask = token_mask.unsqueeze(1)
            valid_lengths = mask.sum(dim=-1).clamp(min=1)
        if self.concat_normalized_length:
            normalized_length = valid_lengths / float(max(1, int(tokens.shape[1])))
            pooled = torch.cat([pooled, normalized_length], dim=-1)
        return pooled


# ---------------------------------------------------------------------------
# Model 2: DeepONet-style baseline (branch net only)
# ---------------------------------------------------------------------------


class DeepONetCountPredictor(nn.Module):
    """Branch-net DeepONet for scalar output.

    For a function → scalar mapping the trunk net degenerates to a fixed
    query point, making this effectively a function encoder + MLP.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        max_len: int,
        n_count_classes: int,
    ) -> None:
        super().__init__()
        self.pad_id = int(vocab_size)
        self.max_len = int(max_len)
        self.embed_dim = int(embed_dim)
        self.token_embedding = nn.Embedding(
            int(vocab_size) + 1, int(embed_dim), padding_idx=self.pad_id
        )
        branch_in = int(max_len) * int(embed_dim)
        self.branch = nn.Sequential(
            nn.Linear(branch_in, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
        )
        head_hidden = max(32, int(hidden_dim) // 4)
        self.count_classifier = nn.Sequential(
            nn.Linear(int(hidden_dim) // 2, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, int(n_count_classes)),
        )

    def forward(
        self, tokens: torch.Tensor, *, token_mask: torch.Tensor
    ) -> torch.Tensor:
        emb = self.token_embedding(tokens)  # (B, L, embed_dim)
        emb = emb * token_mask.unsqueeze(-1)  # zero out padding
        # Pad/truncate to fixed max_len for the branch net
        B, L, D = emb.shape
        if L < self.max_len:
            pad = torch.zeros(
                B, self.max_len - L, D, device=emb.device, dtype=emb.dtype
            )
            emb = torch.cat([emb, pad], dim=1)
        elif L > self.max_len:
            emb = emb[:, : self.max_len, :]
        flat = emb.reshape(B, -1)  # (B, max_len * embed_dim)
        h = self.branch(flat)  # (B, hidden_dim // 2)
        return self.count_classifier(h)  # (B, n_count_classes)


# ---------------------------------------------------------------------------
# Model 3: MLP on bigram features (critical diagnostic)
# ---------------------------------------------------------------------------


def _bigram_features_from_tokens(
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    vocab_size: int,
) -> np.ndarray:
    """Construct the same bigram feature vector that ridge regression uses.

    For each document: first_tok one-hot (V), last_tok one-hot (V),
    normalized unigram (V), normalized bigram (V^2), length (1).
    Total dim = 2*V + V + V^2 + 1.
    """
    V = int(vocab_size)
    feat_dim = 2 * V + V + V * V + 1
    N = int(tokens.shape[0])
    out = np.zeros((N, feat_dim), dtype=np.float32)
    for i in range(N):
        valid = np.flatnonzero(mask[i] > 0.0)
        if valid.size == 0:
            continue
        toks = tokens[i, valid].astype(np.int64)
        offset = 0
        # first_tok one-hot
        out[i, int(toks[0])] = 1.0
        offset += V
        # last_tok one-hot
        out[i, offset + int(toks[-1])] = 1.0
        offset += V
        # normalized unigram
        unigram = np.bincount(toks, minlength=V).astype(np.float32)
        unigram /= float(max(1, toks.size))
        out[i, offset : offset + V] = unigram
        offset += V
        # normalized bigram
        if toks.size >= 2:
            pair_idx = toks[:-1] * V + toks[1:]
            bigram = np.bincount(pair_idx, minlength=V * V).astype(np.float32)
            bigram /= float(max(1, toks.size - 1))
            out[i, offset : offset + V * V] = bigram
        offset += V * V
        # length
        out[i, offset] = float(toks.size)
    return out


class MLPBigramCountPredictor(nn.Module):
    """MLP on the same bigram features that ridge regression uses.

    This is the critical diagnostic: if this model achieves near-zero MAE,
    the CTreePO tree-merge architecture (not neural training) is the bottleneck.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        n_count_classes: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(n_count_classes)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)  # (B, n_count_classes)


# ---------------------------------------------------------------------------
# Model 4: 1D CNN (kernel_size=2 transition detector)
# ---------------------------------------------------------------------------


class CNN1DCountPredictor(nn.Module):
    """1D CNN with kernel_size=2 for direct transition detection.

    A kernel_size=2 convolution directly detects adjacent-token changes,
    which is exactly the sufficient statistic for changepoint counting.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int,
        n_filters: int,
        n_count_classes: int,
    ) -> None:
        super().__init__()
        self.pad_id = int(vocab_size)
        self.token_embedding = nn.Embedding(
            int(vocab_size) + 1, int(embed_dim), padding_idx=self.pad_id
        )
        self.conv1 = nn.Conv1d(int(embed_dim), int(n_filters), kernel_size=2, padding=0)
        self.conv2 = nn.Conv1d(int(n_filters), int(n_filters), kernel_size=1, padding=0)
        head_hidden = max(32, int(n_filters) // 2)
        self.count_classifier = nn.Sequential(
            nn.Linear(int(n_filters), head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, int(n_count_classes)),
        )

    def forward(
        self, tokens: torch.Tensor, *, token_mask: torch.Tensor
    ) -> torch.Tensor:
        emb = self.token_embedding(tokens)  # (B, L, embed_dim)
        x = emb.permute(0, 2, 1)  # (B, embed_dim, L)
        x = F.gelu(self.conv1(x))  # (B, n_filters, L-1)
        x = F.gelu(self.conv2(x))  # (B, n_filters, L-1)
        # Mask for valid token pairs
        pair_mask = (token_mask[:, :-1] * token_mask[:, 1:]).unsqueeze(1)  # (B, 1, L-1)
        x = (x * pair_mask).sum(dim=-1) / pair_mask.sum(dim=-1).clamp(min=1)  # (B, n_filters)
        return self.count_classifier(x)  # (B, n_count_classes)


# ---------------------------------------------------------------------------
# Common training infrastructure
# ---------------------------------------------------------------------------


def _class_setup(
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
) -> Tuple[float, List[int], Dict[int, int], np.ndarray]:
    """Shared class-value bookkeeping for all baselines."""
    target_max = float(
        max(
            1.0,
            float(np.max(train_y)) if train_y.size > 0 else 0.0,
            float(np.max(val_y)) if val_y.size > 0 else 0.0,
            float(np.max(test_y)) if test_y.size > 0 else 0.0,
        )
    )
    class_limit = int(max(0, round(float(target_max))))
    class_values = list(range(class_limit + 1))
    if not class_values:
        class_values = [0]
    class_index = {int(v): idx for idx, v in enumerate(class_values)}
    class_values_arr = np.asarray([float(v) for v in class_values], dtype=np.float64)
    return target_max, class_values, class_index, class_values_arr


def _train_loop_with_predictions(
    *,
    model: nn.Module,
    device: torch.device,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    seed_offset: int,
    baseline_label: str,
    train_inputs: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    val_inputs: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    test_inputs: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
    class_index: Dict[int, int],
    class_values_arr: np.ndarray,
    needs_mask: bool,
    train_eval_inputs: torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | None = None,
    train_eval_y: np.ndarray | None = None,
    train_docs_used: int | None = None,
    runtime_config: GpuRuntimeConfig | None = None,
    runtime_telemetry: GpuRuntimeTelemetry | None = None,
) -> Dict[str, object]:
    """Generic training loop shared by all baselines.

    ``train_inputs`` is either a single tensor (for MLP bigram) or a
    (tokens, mask) tuple (for FNO/DeepONet/CNN).
    """
    from treepo._research.ctreepo.sim.core.markov_neural_operator_baselines import (
        _tensor_nbytes, _ops_gpu_runtime_config,
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    rng = np.random.default_rng(int(seeds["effective_model_seed"]) + int(seed_offset))
    effective_runtime_config = (
        runtime_config
        if runtime_config is not None
        else _ops_gpu_runtime_config(device=device)
    )

    # Unpack inputs
    if needs_mask:
        assert isinstance(train_inputs, tuple)
        train_tok, train_mask_t = train_inputs
        val_tok, val_mask_t = val_inputs  # type: ignore[misc]
        test_tok, test_mask_t = test_inputs  # type: ignore[misc]
        n_train = int(train_tok.shape[0])
    else:
        assert isinstance(train_inputs, torch.Tensor)
        train_feat = train_inputs
        val_feat = val_inputs  # type: ignore[assignment]
        test_feat = test_inputs  # type: ignore[assignment]
        n_train = int(train_feat.shape[0])

    effective_train_eval_inputs = train_inputs if train_eval_inputs is None else train_eval_inputs
    effective_train_eval_y = (
        train_y if train_eval_y is None else np.asarray(train_eval_y, dtype=np.float64)
    )

    train_target_class = torch.tensor(
        [int(class_index[int(round(float(v)))]) for v in train_y.tolist()],
        dtype=torch.long,
        device=device,
    )
    batch_size = int(max(1, min(int(config.batch_size), n_train)))

    best_state = clone_module_state(model)
    has_val = val_y.size > 0
    best_selection = TrainingSelectionMetadata(
        mode=f"best_val_root_mae_{baseline_label}" if has_val else "final_epoch_no_validation",
        split="val" if has_val else "config",
        metric_name=f"val_root_mae_{baseline_label}" if has_val else f"train_root_mae_{baseline_label}",
        metric_value=float("inf"),
        best_epoch=0,
    )
    loss_curve: List[float] = []
    selection_curve: List[float] = []

    def _move_tensor_batch_to_device(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device == device:
            return tensor
        started = time.perf_counter()
        moved = tensor.to(device=device, non_blocking=True)
        if runtime_telemetry is not None:
            runtime_telemetry.add_h2d(
                bytes_transferred=_tensor_nbytes(tensor),
                wall_time_s=time.perf_counter() - started,
            )
        return moved

    def _predict(inputs: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            if needs_mask:
                tok, msk = inputs  # type: ignore[misc]
                if int(tok.shape[0]) == 0:
                    return np.zeros((0,), dtype=np.float64)
                logits_chunks: List[torch.Tensor] = []
                for start in range(0, int(tok.shape[0]), batch_size):
                    stop = min(int(tok.shape[0]), int(start + batch_size))
                    batch_idx = torch.arange(
                        start,
                        stop,
                        dtype=torch.long,
                        device=tok.device,
                    )
                    b_tok = tok.index_select(0, batch_idx)
                    b_msk = msk.index_select(0, batch_idx.to(device=msk.device))
                    if b_tok.device != device:
                        b_tok = _move_tensor_batch_to_device(b_tok)
                    if b_msk.device != device:
                        b_msk = _move_tensor_batch_to_device(b_msk)
                    logits_chunks.append(model(b_tok, token_mask=b_msk))
                logits = torch.cat(logits_chunks, dim=0)
            else:
                feat = inputs  # type: ignore[assignment]
                if int(feat.shape[0]) == 0:
                    return np.zeros((0,), dtype=np.float64)
                logits_chunks = []
                for start in range(0, int(feat.shape[0]), batch_size):
                    stop = min(int(feat.shape[0]), int(start + batch_size))
                    batch_idx = torch.arange(
                        start,
                        stop,
                        dtype=torch.long,
                        device=feat.device,
                    )
                    b_feat = feat.index_select(0, batch_idx)
                    if b_feat.device != device:
                        b_feat = _move_tensor_batch_to_device(b_feat)
                    logits_chunks.append(model(b_feat))
                logits = torch.cat(logits_chunks, dim=0)
            pred_idx = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
        return class_values_arr[pred_idx]

    for epoch_idx in range(int(config.n_epochs)):
        model.train()
        perm = rng.permutation(n_train)
        batch_losses: List[float] = []
        for start in range(0, n_train, batch_size):
            batch_idx_np = perm[start : start + batch_size]
            batch_idx_cpu = torch.tensor(batch_idx_np, dtype=torch.long, device="cpu")
            batch_idx_device = batch_idx_cpu.to(device=device)
            opt.zero_grad(set_to_none=True)

            if needs_mask:
                b_tok = train_tok.index_select(0, batch_idx_cpu.to(device=train_tok.device))
                b_mask = train_mask_t.index_select(
                    0,
                    batch_idx_cpu.to(device=train_mask_t.device),
                )
                if b_tok.device != device:
                    b_tok = _move_tensor_batch_to_device(b_tok)
                if b_mask.device != device:
                    b_mask = _move_tensor_batch_to_device(b_mask)
                logits = model(b_tok, token_mask=b_mask)
            else:
                b_feat = train_feat.index_select(
                    0,
                    batch_idx_cpu.to(device=train_feat.device),
                )
                if b_feat.device != device:
                    b_feat = _move_tensor_batch_to_device(b_feat)
                logits = model(b_feat)

            b_target = train_target_class.index_select(0, batch_idx_device)
            loss = F.cross_entropy(logits, b_target)
            loss.backward()
            if float(config.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
            opt.step()
            batch_losses.append(float(loss.detach().cpu()))

        epoch_loss = float(np.mean(np.asarray(batch_losses, dtype=np.float64)))
        loss_curve.append(epoch_loss)

        if has_val:
            val_pred = _predict(val_inputs)
            sel_val = float(np.mean(np.abs(val_pred - val_y.astype(np.float64))))
            selection_curve.append(sel_val)
            if improved_metric(sel_val, best_selection.metric_value):
                best_selection = TrainingSelectionMetadata(
                    mode=f"best_val_root_mae_{baseline_label}",
                    split="val",
                    metric_name=f"val_root_mae_{baseline_label}",
                    metric_value=float(sel_val),
                    best_epoch=int(epoch_idx),
                )
                best_state = clone_module_state(model)
        else:
            train_pred_epoch = _predict(effective_train_eval_inputs)
            sel_val = float(
                np.mean(
                    np.abs(
                        train_pred_epoch
                        - effective_train_eval_y.astype(np.float64, copy=False)
                    )
                )
            )
            selection_curve.append(sel_val)

    if has_val:
        restore_module_state(model, best_state)
    else:
        final_pred = _predict(effective_train_eval_inputs)
        final_mae = float(
            np.mean(
                np.abs(
                    final_pred
                    - effective_train_eval_y.astype(np.float64, copy=False)
                )
            )
        )
        best_selection = TrainingSelectionMetadata(
            mode="final_epoch_no_validation",
            split="config",
            metric_name=f"train_root_mae_{baseline_label}",
            metric_value=float(final_mae),
            best_epoch=max(0, int(len(loss_curve) - 1)),
        )

    train_pred = _predict(effective_train_eval_inputs)
    val_pred = _predict(val_inputs)
    test_pred = _predict(test_inputs)

    tau = float(config.violation_tau)
    fit_diag = TrainFitDiagnostics(
        train_loss_final=float(loss_curve[-1]) if loss_curve else float("nan"),
        train_loss_curve=tuple(float(x) for x in loss_curve),
        epochs_completed=int(len(loss_curve)),
        selection_metric_curve=tuple(float(x) for x in selection_curve),
        selection_mode=str(best_selection.mode),
        selection_split=str(best_selection.split),
        selection_metric_name=str(best_selection.metric_name),
        selection_metric_value=float(best_selection.metric_value),
        best_epoch=int(best_selection.best_epoch),
        train_exact_match_rate=float(
            _exact_match_rate(
                train_pred,
                effective_train_eval_y.astype(np.float64, copy=False).tolist(),
            )
        ),
        val_exact_match_rate=float(_exact_match_rate(val_pred, val_y.tolist())),
        test_exact_match_rate=float(_exact_match_rate(test_pred, test_y.tolist())),
    )
    return {
        "train_metrics": _eval_root_predictions(
            train_pred,
            effective_train_eval_y.astype(np.float64, copy=False).tolist(),
            tau=tau,
        ),
        "val_metrics": _eval_root_predictions(val_pred, val_y.tolist(), tau=tau),
        "test_metrics": _eval_root_predictions(test_pred, test_y.tolist(), tau=tau),
        "fit_diag": fit_diag,
        "train_preds": np.asarray(train_pred, dtype=np.float64),
        "val_preds": np.asarray(val_pred, dtype=np.float64),
        "test_preds": np.asarray(test_pred, dtype=np.float64),
        "train_truths": np.asarray(effective_train_eval_y, dtype=np.float64),
        "val_truths": np.asarray(val_y, dtype=np.float64),
        "test_truths": np.asarray(test_y, dtype=np.float64),
        "train_docs_used": int(n_train if train_docs_used is None else train_docs_used),
        "runtime_config": dict(effective_runtime_config.as_dict()),
    }


def _train_loop(
    *,
    model: nn.Module,
    device: torch.device,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    seed_offset: int,
    baseline_label: str,
    train_inputs: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    val_inputs: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    test_inputs: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
    class_index: Dict[int, int],
    class_values_arr: np.ndarray,
    needs_mask: bool,
) -> Tuple[SketchMetrics, SketchMetrics, SketchMetrics, TrainFitDiagnostics]:
    fit = _train_loop_with_predictions(
        model=model,
        device=device,
        config=config,
        seeds=seeds,
        seed_offset=seed_offset,
        baseline_label=baseline_label,
        train_inputs=train_inputs,
        val_inputs=val_inputs,
        test_inputs=test_inputs,
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
        class_index=class_index,
        class_values_arr=class_values_arr,
        needs_mask=needs_mask,
    )
    return (
        fit["train_metrics"],
        fit["val_metrics"],
        fit["test_metrics"],
        fit["fit_diag"],
    )


# ---------------------------------------------------------------------------
# Fit functions (public API for each baseline)
# ---------------------------------------------------------------------------


def _fit_fno_baseline(
    *,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    device: torch.device,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
) -> Tuple[SketchMetrics, SketchMetrics, SketchMetrics, TrainFitDiagnostics]:
    """Train an FNO-based count predictor (from the official neuraloperator package)."""
    fit = _fit_fno_baseline_with_predictions(
        config=config,
        seeds=seeds,
        device=device,
        train_docs=train_docs,
        val_docs=val_docs,
        test_docs=test_docs,
    )
    return (
        fit["train_metrics"],
        fit["val_metrics"],
        fit["test_metrics"],
        fit["fit_diag"],
    )


def _fit_fno_baseline_with_predictions(
    *,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    device: torch.device,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
    train_eval_docs: Sequence[ChangepointMarkovDoc] | None = None,
    runtime_config: GpuRuntimeConfig | None = None,
) -> Dict[str, object]:
    """Train the full-document FNO baseline and return prediction arrays."""
    from treepo._research.ctreepo.sim.core.markov_neural_operator_baselines import (
        _build_full_doc_gpu_batch_store, _merge_gpu_runtime_telemetry,
        _gpu_peak_memory_gb, _gpu_runtime_config_from_ops_config,
        _tensor_nbytes,
        _fit_linear_regression_probe_local, _fit_linear_classifier_probe_local,
        _predict_linear_regression_probe_local, _predict_linear_classifier_probe_local,
        _summary_component_metrics_local,
    )
    if not HAS_NEURAL_OPERATOR:
        raise ImportError(_INSTALL_MSG)
    if not train_docs:
        z = _zero_sketch_metrics(n_docs=0)
        empty = np.zeros((0,), dtype=np.float64)
        return {
            "train_metrics": z,
            "val_metrics": z,
            "test_metrics": z,
            "fit_diag": TrainFitDiagnostics(
                train_loss_final=float("nan"),
                train_loss_curve=(),
                epochs_completed=0,
            ),
            "train_preds": empty,
            "val_preds": empty,
            "test_preds": empty,
            "train_truths": empty,
            "val_truths": empty,
            "test_truths": empty,
            "train_docs_used": 0,
        }

    # Reset global seed immediately before FNO training so that results are
    # reproducible regardless of how much random state was consumed upstream
    # (e.g. the public API path runs tree training before this baseline).
    _set_global_seed(int(seeds["effective_model_seed"]) + 60_000)

    train_eval_docs = tuple(train_docs if train_eval_docs is None else train_eval_docs)
    effective_runtime_config = (
        runtime_config
        if runtime_config is not None
        else _gpu_runtime_config_from_ops_config(config, device=device)
    )
    runtime_telemetry = GpuRuntimeTelemetry(
        data_mode=str(effective_runtime_config.data_mode),
        bucket_mode=str(effective_runtime_config.bucket_mode),
        workers_per_mig=int(effective_runtime_config.workers_per_mig),
    )
    pad_id = int(config.vocab_size)
    train_tokens, train_mask, train_y = _token_sequence_arrays(train_docs, pad_id=pad_id)
    train_eval_tokens, train_eval_mask, train_eval_y = _token_sequence_arrays(
        train_eval_docs,
        pad_id=pad_id,
    )
    val_tokens, val_mask, val_y = _token_sequence_arrays(val_docs, pad_id=pad_id)
    test_tokens, test_mask, test_y = _token_sequence_arrays(test_docs, pad_id=pad_id)
    target_max, class_values, class_index, class_values_arr = _class_setup(
        train_eval_y, val_y, test_y
    )

    # Resolve FNO architecture via the single canonical path.
    # When tree_leaf_fno_* is set (e.g. canary/comparable presets), those
    # values are used; otherwise falls back to fno_* on OPSCountConfig.
    from treepo._research.ctreepo.sim.core.fno_arch_config import resolve_fno_arch as _resolve_fno_arch
    _fno_arch = _resolve_fno_arch(config)
    width = _fno_arch.width
    embed_dim = width
    n_modes = _fno_arch.n_modes
    n_layers = _fno_arch.n_layers

    model = FNOCountPredictor(
        vocab_size=int(config.vocab_size),
        embed_dim=embed_dim,
        n_modes=n_modes,
        width=width,
        n_layers=n_layers,
        n_count_classes=len(class_values),
        pooling_mode=str(getattr(config, "doc_sequence_fno_pooling", "mean")),
        concat_normalized_length=bool(
            getattr(config, "doc_sequence_fno_concat_length_feature", False)
        ),
        include_transition_channel=bool(
            getattr(config, "doc_sequence_fno_include_transition_channel", False)
        ),
    ).to(device=device)
    train_store, train_store_telemetry = _build_full_doc_gpu_batch_store(
        tokens=train_tokens,
        mask=train_mask,
        targets=train_y,
        device=device,
        split_name="train",
        runtime_config=effective_runtime_config,
    )
    _merge_gpu_runtime_telemetry(runtime_telemetry, train_store_telemetry)
    train_eval_store, train_eval_store_telemetry = _build_full_doc_gpu_batch_store(
        tokens=train_eval_tokens,
        mask=train_eval_mask,
        targets=train_eval_y,
        device=device,
        split_name="train_eval",
        runtime_config=effective_runtime_config,
    )
    _merge_gpu_runtime_telemetry(runtime_telemetry, train_eval_store_telemetry)
    val_store, val_store_telemetry = _build_full_doc_gpu_batch_store(
        tokens=val_tokens,
        mask=val_mask,
        targets=val_y,
        device=device,
        split_name="val",
        runtime_config=effective_runtime_config,
    )
    _merge_gpu_runtime_telemetry(runtime_telemetry, val_store_telemetry)
    test_store, test_store_telemetry = _build_full_doc_gpu_batch_store(
        tokens=test_tokens,
        mask=test_mask,
        targets=test_y,
        device=device,
        split_name="test",
        runtime_config=effective_runtime_config,
    )
    _merge_gpu_runtime_telemetry(runtime_telemetry, test_store_telemetry)

    def _full_doc_inputs(
        *,
        tokens: np.ndarray,
        mask: np.ndarray,
        store: GpuBatchStore | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if store is not None:
            view = store.view_for_doc_indices(list(range(int(tokens.shape[0]))))
            if view is not None:
                view_tokens = view.tensors.get("tokens")
                view_mask = view.tensors.get("mask")
                if isinstance(view_tokens, torch.Tensor) and isinstance(view_mask, torch.Tensor):
                    return (
                        view_tokens.to(device=device, dtype=torch.long),
                        view_mask.to(device=device, dtype=torch.float32),
                    )
        tensor_device = (
            torch.device("cpu")
            if str(device.type) == "cuda" and not bool(effective_runtime_config.is_resident)
            else device
        )
        return (
            torch.tensor(tokens, dtype=torch.long, device=tensor_device),
            torch.tensor(mask, dtype=torch.float32, device=tensor_device),
        )

    train_tok, train_mask_t = _full_doc_inputs(
        tokens=train_tokens,
        mask=train_mask,
        store=train_store,
    )
    train_eval_tok, train_eval_mask_t = _full_doc_inputs(
        tokens=train_eval_tokens,
        mask=train_eval_mask,
        store=train_eval_store,
    )
    val_tok, val_mask_t = _full_doc_inputs(
        tokens=val_tokens,
        mask=val_mask,
        store=val_store,
    )
    test_tok, test_mask_t = _full_doc_inputs(
        tokens=test_tokens,
        mask=test_mask,
        store=test_store,
    )
    fit = _train_loop_with_predictions(
        model=model,
        device=device,
        config=config,
        seeds=seeds,
        seed_offset=60_001,
        baseline_label="fno",
        train_inputs=(train_tok, train_mask_t),
        val_inputs=(val_tok, val_mask_t),
        test_inputs=(test_tok, test_mask_t),
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
        class_index=class_index,
        class_values_arr=class_values_arr,
        needs_mask=True,
        train_eval_inputs=(train_eval_tok, train_eval_mask_t),
        train_eval_y=train_eval_y,
        train_docs_used=int(len(train_docs)),
        runtime_config=effective_runtime_config,
        runtime_telemetry=runtime_telemetry,
    )

    def _repr_inputs_on_device(
        tok: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tok.device != device:
            started = time.perf_counter()
            moved_tok = tok.to(device=device, non_blocking=True)
            if runtime_telemetry is not None:
                runtime_telemetry.add_h2d(
                    bytes_transferred=_tensor_nbytes(tok),
                    wall_time_s=time.perf_counter() - started,
                )
            tok = moved_tok
        if mask.device != device:
            started = time.perf_counter()
            moved_mask = mask.to(device=device, non_blocking=True)
            if runtime_telemetry is not None:
                runtime_telemetry.add_h2d(
                    bytes_transferred=_tensor_nbytes(mask),
                    wall_time_s=time.perf_counter() - started,
                )
            mask = moved_mask
        return tok, mask

    def _encode_representations_batched(
        tok: torch.Tensor,
        mask: torch.Tensor,
    ) -> np.ndarray:
        n_items = int(tok.shape[0])
        if n_items <= 0:
            return np.zeros((0, int(config.state_dim)), dtype=np.float64)
        batch_size = int(max(1, min(int(config.batch_size), n_items)))
        chunks: List[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, n_items, batch_size):
                stop = min(n_items, int(start + batch_size))
                batch_tok = tok[start:stop]
                batch_mask = mask[start:stop]
                batch_tok, batch_mask = _repr_inputs_on_device(batch_tok, batch_mask)
                batch_repr = (
                    model.encode_representation(batch_tok, token_mask=batch_mask)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
                chunks.append(batch_repr)
        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks, axis=0)

    def _root_summary_probe_metrics(
        tok: torch.Tensor,
        mask: torch.Tensor,
        docs: Sequence[ChangepointMarkovDoc],
        count_probe: Optional[np.ndarray],
        first_probe: Optional[np.ndarray],
        last_probe: Optional[np.ndarray],
    ) -> Dict[str, float]:
        if int(tok.shape[0]) <= 0 or not docs:
            return {
                "count_mae": float("nan"),
                "first_accuracy": float("nan"),
                "last_accuracy": float("nan"),
                "exact_summary_match_rate": float("nan"),
            }
        reps = _encode_representations_batched(tok, mask)
        count_targets = np.asarray(
            [int(len(doc.true_boundaries)) + 1 for doc in docs],
            dtype=np.int64,
        )
        first_targets = np.asarray(
            [int(doc.token_regimes[0]) for doc in docs],
            dtype=np.int64,
        )
        last_targets = np.asarray(
            [int(doc.token_regimes[-1]) for doc in docs],
            dtype=np.int64,
        )
        return _summary_component_metrics_local(
            count_preds=_predict_linear_regression_probe_local(count_probe, reps),
            first_preds=_predict_linear_classifier_probe_local(
                first_probe,
                reps,
                n_classes=int(config.n_regimes),
            ),
            last_preds=_predict_linear_classifier_probe_local(
                last_probe,
                reps,
                n_classes=int(config.n_regimes),
            ),
            count_targets=count_targets,
            first_targets=first_targets,
            last_targets=last_targets,
        )

    train_repr = _encode_representations_batched(train_eval_tok, train_eval_mask_t)
    train_count_targets = np.asarray(
        [int(len(doc.true_boundaries)) + 1 for doc in train_eval_docs],
        dtype=np.int64,
    )
    train_first_targets = np.asarray(
        [int(doc.token_regimes[0]) for doc in train_eval_docs],
        dtype=np.int64,
    )
    train_last_targets = np.asarray(
        [int(doc.token_regimes[-1]) for doc in train_eval_docs],
        dtype=np.int64,
    )
    count_probe = _fit_linear_regression_probe_local(train_repr, train_count_targets)
    first_probe = _fit_linear_classifier_probe_local(
        train_repr,
        train_first_targets,
        n_classes=int(config.n_regimes),
    )
    last_probe = _fit_linear_classifier_probe_local(
        train_repr,
        train_last_targets,
        n_classes=int(config.n_regimes),
    )
    fit["root_summary_probe_audit"] = {
            "train": _root_summary_probe_metrics(
                train_eval_tok,
                train_eval_mask_t,
                train_eval_docs,
                count_probe,
                first_probe,
                last_probe,
            ),
            "val": _root_summary_probe_metrics(
                val_tok,
                val_mask_t,
                val_docs,
                count_probe,
                first_probe,
                last_probe,
            ),
            "test": _root_summary_probe_metrics(
                test_tok,
                test_mask_t,
                test_docs,
                count_probe,
                first_probe,
                last_probe,
            ),
    }
    fit["baseline_fno_actual_config"] = {
        "n_layers": 4,
        "n_modes": int(n_modes),
        "width": int(width),
        "embed_dim": int(embed_dim),
        "pooling_mode": str(getattr(config, "doc_sequence_fno_pooling", "mean")),
        "concat_normalized_length": bool(
            getattr(config, "doc_sequence_fno_concat_length_feature", False)
        ),
        "include_transition_channel": bool(
            getattr(config, "doc_sequence_fno_include_transition_channel", False)
        ),
        "training_path": "shared_flat_baseline_trainer",
        "runtime_config": dict(effective_runtime_config.as_dict()),
    }
    reserved_peak_gb, allocated_peak_gb = _gpu_peak_memory_gb(device)
    fit["runtime_efficiency"] = runtime_telemetry.as_dict(
        gpu_reserved_peak_gb=reserved_peak_gb,
        gpu_allocated_peak_gb=allocated_peak_gb,
    )
    return fit


def _fit_deeponet_baseline(
    *,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    device: torch.device,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
) -> Tuple[SketchMetrics, SketchMetrics, SketchMetrics, TrainFitDiagnostics]:
    """Train a DeepONet-style branch-net count predictor."""
    if not train_docs:
        z = _zero_sketch_metrics(n_docs=0)
        return z, z, z, TrainFitDiagnostics(
            train_loss_final=float("nan"), train_loss_curve=(), epochs_completed=0
        )

    pad_id = int(config.vocab_size)
    train_tokens, train_mask, train_y = _token_sequence_arrays(train_docs, pad_id=pad_id)
    val_tokens, val_mask, val_y = _token_sequence_arrays(val_docs, pad_id=pad_id)
    test_tokens, test_mask, test_y = _token_sequence_arrays(test_docs, pad_id=pad_id)
    target_max, class_values, class_index, class_values_arr = _class_setup(
        train_y, val_y, test_y
    )

    embed_dim = max(64, int(config.state_dim))
    max_len = int(train_tokens.shape[1])

    model = DeepONetCountPredictor(
        vocab_size=int(config.vocab_size),
        embed_dim=embed_dim,
        hidden_dim=max(256, int(config.hidden_dim)),
        max_len=max_len,
        n_count_classes=len(class_values),
    ).to(device=device)

    train_t = (
        torch.tensor(train_tokens, dtype=torch.long, device=device),
        torch.tensor(train_mask, dtype=torch.float32, device=device),
    )
    val_t = (
        torch.tensor(val_tokens, dtype=torch.long, device=device),
        torch.tensor(val_mask, dtype=torch.float32, device=device),
    )
    test_t = (
        torch.tensor(test_tokens, dtype=torch.long, device=device),
        torch.tensor(test_mask, dtype=torch.float32, device=device),
    )

    return _train_loop(
        model=model,
        device=device,
        config=config,
        seeds=seeds,
        seed_offset=60_002,
        baseline_label="deeponet",
        train_inputs=train_t,
        val_inputs=val_t,
        test_inputs=test_t,
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
        class_index=class_index,
        class_values_arr=class_values_arr,
        needs_mask=True,
    )




def _fit_mlp_bigram_baseline(
    *,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    device: torch.device,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
) -> Tuple[SketchMetrics, SketchMetrics, SketchMetrics, TrainFitDiagnostics]:
    fit = _fit_mlp_bigram_baseline_with_predictions(
        config=config,
        seeds=seeds,
        device=device,
        train_docs=train_docs,
        val_docs=val_docs,
        test_docs=test_docs,
    )
    return (
        fit["train_metrics"],
        fit["val_metrics"],
        fit["test_metrics"],
        fit["fit_diag"],
    )


def _fit_mlp_bigram_baseline_with_predictions(
    *,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    device: torch.device,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
) -> Dict[str, object]:
    """Train an MLP on the same bigram features that ridge regression uses.

    This is the critical diagnostic baseline.  If it achieves near-zero MAE,
    the CTreePO tree-merge architecture is the bottleneck, not neural training.
    """
    if not train_docs:
        z = _zero_sketch_metrics(n_docs=0)
        empty = np.zeros((0,), dtype=np.float64)
        return {
            "train_metrics": z,
            "val_metrics": z,
            "test_metrics": z,
            "fit_diag": TrainFitDiagnostics(
                train_loss_final=float("nan"), train_loss_curve=(), epochs_completed=0
            ),
            "train_preds": empty,
            "val_preds": empty,
            "test_preds": empty,
            "train_truths": empty,
            "val_truths": empty,
            "test_truths": empty,
            "train_docs_used": 0,
        }

    # Reset global seed so model init is reproducible regardless of upstream state.
    _set_global_seed(int(seeds["effective_model_seed"]) + 60_003)

    pad_id = int(config.vocab_size)
    train_tokens, train_mask, train_y = _token_sequence_arrays(train_docs, pad_id=pad_id)
    val_tokens, val_mask, val_y = _token_sequence_arrays(val_docs, pad_id=pad_id)
    test_tokens, test_mask, test_y = _token_sequence_arrays(test_docs, pad_id=pad_id)
    target_max, class_values, class_index, class_values_arr = _class_setup(
        train_y, val_y, test_y
    )

    V = int(config.vocab_size)
    train_feat = _bigram_features_from_tokens(train_tokens, train_mask, vocab_size=V)
    val_feat = _bigram_features_from_tokens(val_tokens, val_mask, vocab_size=V)
    test_feat = _bigram_features_from_tokens(test_tokens, test_mask, vocab_size=V)
    feat_dim = int(train_feat.shape[1])

    model = MLPBigramCountPredictor(
        input_dim=feat_dim,
        hidden_dim=max(256, int(config.hidden_dim)),
        n_count_classes=len(class_values),
    ).to(device=device)

    train_t = torch.tensor(train_feat, dtype=torch.float32, device=device)
    val_t = torch.tensor(val_feat, dtype=torch.float32, device=device)
    test_t = torch.tensor(test_feat, dtype=torch.float32, device=device)

    return _train_loop_with_predictions(
        model=model,
        device=device,
        config=config,
        seeds=seeds,
        seed_offset=60_003,
        baseline_label="mlp_bigram",
        train_inputs=train_t,
        val_inputs=val_t,
        test_inputs=test_t,
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
        class_index=class_index,
        class_values_arr=class_values_arr,
        needs_mask=False,
    )


def _fit_cnn1d_baseline(
    *,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    device: torch.device,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
) -> Tuple[SketchMetrics, SketchMetrics, SketchMetrics, TrainFitDiagnostics]:
    fit = _fit_cnn1d_baseline_with_predictions(
        config=config,
        seeds=seeds,
        device=device,
        train_docs=train_docs,
        val_docs=val_docs,
        test_docs=test_docs,
    )
    return (
        fit["train_metrics"],
        fit["val_metrics"],
        fit["test_metrics"],
        fit["fit_diag"],
    )


def _fit_cnn1d_baseline_with_predictions(
    *,
    config: OPSCountConfig,
    seeds: Mapping[str, int],
    device: torch.device,
    train_docs: Sequence[ChangepointMarkovDoc],
    val_docs: Sequence[ChangepointMarkovDoc],
    test_docs: Sequence[ChangepointMarkovDoc],
) -> Dict[str, object]:
    """Train a 1D CNN with kernel_size=2 for direct transition detection."""
    if not train_docs:
        z = _zero_sketch_metrics(n_docs=0)
        empty = np.zeros((0,), dtype=np.float64)
        return {
            "train_metrics": z,
            "val_metrics": z,
            "test_metrics": z,
            "fit_diag": TrainFitDiagnostics(
                train_loss_final=float("nan"), train_loss_curve=(), epochs_completed=0
            ),
            "train_preds": empty,
            "val_preds": empty,
            "test_preds": empty,
            "train_truths": empty,
            "val_truths": empty,
            "test_truths": empty,
            "train_docs_used": 0,
        }

    # Reset global seed so model init is reproducible regardless of upstream state.
    _set_global_seed(int(seeds["effective_model_seed"]) + 60_002)

    pad_id = int(config.vocab_size)
    train_tokens, train_mask, train_y = _token_sequence_arrays(train_docs, pad_id=pad_id)
    val_tokens, val_mask, val_y = _token_sequence_arrays(val_docs, pad_id=pad_id)
    test_tokens, test_mask, test_y = _token_sequence_arrays(test_docs, pad_id=pad_id)
    target_max, class_values, class_index, class_values_arr = _class_setup(
        train_y, val_y, test_y
    )

    embed_dim = max(64, int(config.state_dim))
    n_filters = max(64, int(config.hidden_dim))

    model = CNN1DCountPredictor(
        vocab_size=int(config.vocab_size),
        embed_dim=embed_dim,
        n_filters=n_filters,
        n_count_classes=len(class_values),
    ).to(device=device)

    train_t = (
        torch.tensor(train_tokens, dtype=torch.long, device=device),
        torch.tensor(train_mask, dtype=torch.float32, device=device),
    )
    val_t = (
        torch.tensor(val_tokens, dtype=torch.long, device=device),
        torch.tensor(val_mask, dtype=torch.float32, device=device),
    )
    test_t = (
        torch.tensor(test_tokens, dtype=torch.long, device=device),
        torch.tensor(test_mask, dtype=torch.float32, device=device),
    )

    return _train_loop_with_predictions(
        model=model,
        device=device,
        config=config,
        seeds=seeds,
        seed_offset=60_004,
        baseline_label="cnn1d",
        train_inputs=train_t,
        val_inputs=val_t,
        test_inputs=test_t,
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
        class_index=class_index,
        class_values_arr=class_values_arr,
        needs_mask=True,
    )
