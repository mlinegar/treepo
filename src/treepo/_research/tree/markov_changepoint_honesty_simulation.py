"""
Markov changepoint boundary-detection honesty simulation.

This simulation extends the Markov boundary-cost toy setup with explicit
changepoint labels and boundary-detection metrics:
- Precision/recall/F1 with token-tolerance matching
- Mean localization error for matched boundaries
- Boundary-count ratio (predicted vs true)

It keeps the real adaptive chunker + honest role split:
- `chunker_honest`: chunking consumes predicted boundary-role signals
- `chunker_leaky`: chunking consumes oracle evaluation-role signals
- `fixed`: no adaptive feedback
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for Markov changepoint honesty simulations. "
        "Install with: uv sync --extra torch"
    ) from e

from treepo._research.preprocessing.chunker import (
    AdaptiveChunkMemory,
    AdaptiveChunkingConfig,
    ChunkFeedbackSignal,
    HonestChunkingPolicy,
    chunk_for_ops,
)
from treepo._research.tree.markov_boundary_honesty_simulation import (
    _boundary_costs,
    _chunked_posterior,
    _kl_divergence,
    _l1_discrepancy,
    _loglik,
    _make_transition_matrices,
    _set_global_seed,
    _window_features,
)


@dataclass(frozen=True)
class ChangepointMarkovDoc:
    tokens: Tuple[int, ...]
    token_regimes: Tuple[int, ...]
    transition_regimes: Tuple[int, ...]
    true_boundaries: Tuple[int, ...]


@dataclass(frozen=True)
class MarkovChangepointConfig:
    n_regimes: int = 4
    vocab_size: int = 96
    min_tokens: int = 96
    max_tokens: int = 96
    min_segments: int = 2
    max_segments: int = 5
    min_seg_len: int = 8
    max_seg_len: int = 32

    min_leaf_tokens: int = 1
    max_leaf_tokens: int = 8
    fixed_leaf_tokens: int = 2
    token_char_width: int = 300
    boundary_tolerance_tokens: int = 2

    train_docs: int = 120
    test_docs: int = 60

    sinkhorn_iters: int = 30
    transition_log_std: float = 1.25

    window_size: int = 2
    boundary_emb_dim: int = 24
    boundary_hidden_dim: int = 48
    boundary_batch_size: int = 256
    boundary_max_train_samples: int = 60000
    balance_training: bool = True
    positive_class_weight: Optional[float] = None
    calibrate_prior: bool = True
    calibrate_pos_weight: bool = True

    n_epochs: int = 6
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0

    seed: int = 0
    use_cuda: bool = True
    cuda_device: Optional[int] = None
    torch_threads: int = 0


@dataclass(frozen=True)
class ChangepointPolicyMetrics:
    mean_boundary_cost: float
    mean_num_boundaries: float
    mean_l1: float
    mean_kl: float
    mean_loglik_drop: float

    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    mean_localization_error: float
    predicted_to_true_ratio: float

    total_predicted_boundaries: int
    total_true_boundaries: int
    total_matched_boundaries: int
    n_docs: int


@dataclass(frozen=True)
class MarkovChangepointHonestySummary:
    config: Dict[str, object]
    boundary_model_train_loss_final: float
    boundary_train_positive_rate: float
    metrics: Dict[str, ChangepointPolicyMetrics]

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "boundary_model_train_loss_final": self.boundary_model_train_loss_final,
            "boundary_train_positive_rate": self.boundary_train_positive_rate,
            "metrics": {k: asdict(v) for k, v in self.metrics.items()},
        }
        return json.dumps(payload, indent=2, sort_keys=True)


class BoundaryChangepointPredictor(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        window_size: int,
        emb_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.window_size = int(window_size)
        self.pad_id = int(vocab_size)
        self.embedding = nn.Embedding(int(vocab_size) + 1, int(emb_dim))
        self.net = nn.Sequential(
            nn.Linear(2 * int(window_size) * int(emb_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        flat = emb.reshape(emb.shape[0], -1)
        return self.net(flat).squeeze(-1)


@dataclass(frozen=True)
class _BoundaryMatchStats:
    tp: int
    fp: int
    fn: int
    localization_error_sum: float


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _logit(p: float) -> float:
    p = float(p)
    p = max(1e-8, min(1.0 - 1e-8, p))
    return float(math.log(p / (1.0 - p)))


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - float(np.max(x))
    exp = np.exp(x)
    return exp / float(np.sum(exp))


def _validate_config(config: MarkovChangepointConfig) -> None:
    if int(config.n_regimes) < 1:
        raise ValueError("n_regimes must be >= 1")
    if int(config.vocab_size) < 2:
        raise ValueError("vocab_size must be >= 2")
    if int(config.min_tokens) < 2 or int(config.max_tokens) < int(config.min_tokens):
        raise ValueError("require 2 <= min_tokens <= max_tokens")
    if int(config.min_segments) < 2 or int(config.max_segments) < int(config.min_segments):
        raise ValueError("require 2 <= min_segments <= max_segments")
    if int(config.min_seg_len) < 2 or int(config.max_seg_len) < int(config.min_seg_len):
        raise ValueError("require 2 <= min_seg_len <= max_seg_len")
    if int(config.window_size) < 1:
        raise ValueError("window_size must be >= 1")
    if int(config.token_char_width) < 2:
        raise ValueError("token_char_width must be >= 2")


def _sample_num_segments(
    *,
    n_tokens: int,
    config: MarkovChangepointConfig,
    rng: np.random.Generator,
) -> int:
    feasible_min = int(max(int(config.min_segments), math.ceil(float(n_tokens) / float(config.max_seg_len))))
    feasible_max = int(min(int(config.max_segments), int(n_tokens) // int(config.min_seg_len)))
    if feasible_min > feasible_max:
        raise ValueError(
            "No feasible segment count for token length "
            f"{n_tokens} with min/max segments {config.min_segments}/{config.max_segments} "
            f"and min/max seg len {config.min_seg_len}/{config.max_seg_len}."
        )
    return int(rng.integers(feasible_min, feasible_max + 1))


def _sample_segment_lengths(
    *,
    n_tokens: int,
    n_segments: int,
    min_seg_len: int,
    max_seg_len: int,
    rng: np.random.Generator,
) -> Tuple[int, ...]:
    lengths: List[int] = []
    remaining_tokens = int(n_tokens)
    remaining_segments = int(n_segments)

    for _ in range(int(n_segments) - 1):
        min_allowed = max(
            int(min_seg_len),
            remaining_tokens - (remaining_segments - 1) * int(max_seg_len),
        )
        max_allowed = min(
            int(max_seg_len),
            remaining_tokens - (remaining_segments - 1) * int(min_seg_len),
        )
        if min_allowed > max_allowed:
            raise ValueError("Infeasible segment-length draw under current constraints.")
        seg_len = int(rng.integers(min_allowed, max_allowed + 1))
        lengths.append(seg_len)
        remaining_tokens -= seg_len
        remaining_segments -= 1

    lengths.append(int(remaining_tokens))
    return tuple(lengths)


def _sample_segment_regimes(
    *,
    n_segments: int,
    n_regimes: int,
    rng: np.random.Generator,
) -> Tuple[int, ...]:
    regimes: List[int] = []
    for i in range(int(n_segments)):
        if i == 0 or int(n_regimes) == 1:
            regimes.append(int(rng.integers(0, int(n_regimes))))
            continue
        prev = regimes[-1]
        offset = int(rng.integers(1, int(n_regimes)))
        regimes.append(int((prev + offset) % int(n_regimes)))
    return tuple(regimes)


def generate_changepoint_docs(
    config: MarkovChangepointConfig,
    *,
    transitions: np.ndarray,
) -> Tuple[ChangepointMarkovDoc, ...]:
    rng = np.random.default_rng(int(config.seed))
    n_docs = int(config.train_docs) + int(config.test_docs)

    cdfs = np.cumsum(transitions, axis=2)
    cdfs[:, :, -1] = 1.0

    docs: List[ChangepointMarkovDoc] = []
    for _ in range(n_docs):
        n_tokens = int(rng.integers(int(config.min_tokens), int(config.max_tokens) + 1))
        n_segments = _sample_num_segments(n_tokens=n_tokens, config=config, rng=rng)
        seg_lengths = _sample_segment_lengths(
            n_tokens=n_tokens,
            n_segments=n_segments,
            min_seg_len=int(config.min_seg_len),
            max_seg_len=int(config.max_seg_len),
            rng=rng,
        )
        seg_regimes = _sample_segment_regimes(
            n_segments=n_segments,
            n_regimes=int(config.n_regimes),
            rng=rng,
        )

        token_regimes = np.empty(int(n_tokens), dtype=np.int64)
        idx = 0
        for seg_len, regime in zip(seg_lengths, seg_regimes):
            token_regimes[idx : idx + int(seg_len)] = int(regime)
            idx += int(seg_len)

        tokens = np.empty(int(n_tokens), dtype=np.int64)
        tokens[0] = int(rng.integers(0, int(config.vocab_size)))

        # Transition t -> t+1 uses the regime active at token t+1.
        transition_regimes = token_regimes[1:].copy()
        for t in range(int(n_tokens) - 1):
            regime = int(transition_regimes[t])
            row = cdfs[regime, int(tokens[t])]
            u = float(rng.random())
            nxt = int(np.searchsorted(row, u, side="right"))
            if nxt >= int(config.vocab_size):
                nxt = int(config.vocab_size) - 1
            tokens[t + 1] = nxt

        boundaries = np.nonzero(token_regimes[:-1] != token_regimes[1:])[0].astype(np.int64)

        docs.append(
            ChangepointMarkovDoc(
                tokens=tuple(int(x) for x in tokens.tolist()),
                token_regimes=tuple(int(x) for x in token_regimes.tolist()),
                transition_regimes=tuple(int(x) for x in transition_regimes.tolist()),
                true_boundaries=tuple(int(x) for x in boundaries.tolist()),
            )
        )

    return tuple(docs)


def _boundary_labels_for_doc(doc: ChangepointMarkovDoc) -> np.ndarray:
    n_tokens = int(len(doc.tokens))
    if n_tokens < 2:
        return np.zeros((0,), dtype=np.float32)
    y = np.zeros((n_tokens - 1,), dtype=np.float32)
    if len(doc.true_boundaries) > 0:
        idx = np.asarray(doc.true_boundaries, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n_tokens - 1)]
        y[idx] = 1.0
    return y


def _oracle_boundary_probabilities(
    doc: ChangepointMarkovDoc,
    *,
    tolerance_tokens: int,
) -> np.ndarray:
    """
    Build tolerance-aware oracle probabilities from exact changepoints.

    The adaptive chunker operates over coarse axis segments, so strict one-hot
    labels can under-segment when boundaries are slightly misaligned in char
    space. This widens oracle supervision within the same matching tolerance
    used for evaluation.
    """
    labels = _boundary_labels_for_doc(doc).astype(np.float64, copy=False)
    if len(labels) == 0:
        return labels
    radius = int(max(0, tolerance_tokens))
    if radius <= 0:
        return labels

    idx = np.where(labels > 0.5)[0]
    if len(idx) == 0:
        return labels

    out = np.zeros_like(labels)
    for b in idx.tolist():
        lo = max(0, int(b) - radius)
        hi = min(len(labels) - 1, int(b) + radius)
        for t in range(lo, hi + 1):
            dist = abs(int(t) - int(b))
            # Linear taper: boundary center=1, edges>0.
            score = 1.0 - (float(dist) / float(radius + 1))
            if score > out[t]:
                out[t] = score
    return np.clip(out, 0.0, 1.0)


def _collect_boundary_training_data(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    config: MarkovChangepointConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    pad_id = int(config.vocab_size)
    window = int(config.window_size)

    xs: List[np.ndarray] = []
    ys: List[float] = []

    for doc in docs:
        toks = np.asarray(doc.tokens, dtype=np.int64)
        if len(toks) < 2:
            continue
        labels = _boundary_labels_for_doc(doc)
        for t, y in enumerate(labels.tolist()):
            xs.append(_window_features(toks, boundary_idx=int(t), window=window, pad_id=pad_id))
            ys.append(float(y))

    if len(xs) == 0:
        return np.zeros((0, 2 * window), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    x_arr = np.stack(xs, axis=0)
    y_arr = np.asarray(ys, dtype=np.float32)

    max_samples = int(max(1, config.boundary_max_train_samples))
    pos_idx = np.where(y_arr > 0.5)[0]
    neg_idx = np.where(y_arr <= 0.5)[0]

    if bool(config.balance_training) and len(pos_idx) > 0 and len(neg_idx) > 0 and max_samples >= 2:
        n_pos_take = max(1, max_samples // 2)
        n_neg_take = max(1, max_samples - n_pos_take)
        pos_keep = rng.choice(pos_idx, size=n_pos_take, replace=len(pos_idx) < n_pos_take)
        neg_keep = rng.choice(neg_idx, size=n_neg_take, replace=len(neg_idx) < n_neg_take)
        keep = np.concatenate([pos_keep, neg_keep], axis=0)
    else:
        all_idx = np.arange(len(y_arr), dtype=np.int64)
        if len(all_idx) > max_samples:
            keep = rng.choice(all_idx, size=max_samples, replace=False)
        else:
            keep = all_idx

    keep = np.asarray(keep, dtype=np.int64)
    rng.shuffle(keep)
    return x_arr[keep], y_arr[keep]


def _resolve_positive_weight(
    *,
    y_train: np.ndarray,
    config: MarkovChangepointConfig,
) -> float:
    if config.positive_class_weight is not None:
        return float(max(1e-6, float(config.positive_class_weight)))

    n_pos = float(np.sum(y_train > 0.5))
    n_neg = float(np.sum(y_train <= 0.5))
    if n_pos <= 0:
        return 1.0
    return float(max(1.0, n_neg / n_pos))


def _train_boundary_model(
    model: BoundaryChangepointPredictor,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: MarkovChangepointConfig,
    device: torch.device,
) -> float:
    model.to(device)
    model.train()

    if len(x_train) == 0:
        return float("nan")

    x_t = torch.tensor(x_train, dtype=torch.long, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    positive_weight = _resolve_positive_weight(y_train=y_train, config=config)
    pos_weight_t = torch.tensor(float(positive_weight), dtype=torch.float32, device=device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )

    batch = int(max(1, config.boundary_batch_size))
    idxs = np.arange(len(x_train), dtype=np.int64)
    rng = np.random.default_rng(int(config.seed) + 101)

    loss_final = float("nan")
    for _ in range(int(config.n_epochs)):
        rng.shuffle(idxs)
        losses: List[float] = []
        for i0 in range(0, len(idxs), batch):
            batch_idxs = idxs[i0 : i0 + batch]
            opt.zero_grad(set_to_none=True)
            logits = model(x_t[batch_idxs])
            loss = F.binary_cross_entropy_with_logits(
                logits,
                y_t[batch_idxs],
                pos_weight=pos_weight_t,
                reduction="mean",
            )
            loss.backward()
            if float(config.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        loss_final = float(np.mean(np.asarray(losses, dtype=np.float64)))

    return loss_final


@torch.no_grad()
def _predict_boundary_probabilities(
    model: BoundaryChangepointPredictor,
    tokens: np.ndarray,
    *,
    config: MarkovChangepointConfig,
    device: torch.device,
    logit_offset: float = 0.0,
) -> np.ndarray:
    model.to(device)
    model.eval()

    if len(tokens) < 2:
        return np.zeros((0,), dtype=np.float64)

    pad_id = int(config.vocab_size)
    window = int(config.window_size)

    feats = np.stack(
        [
            _window_features(tokens, boundary_idx=t, window=window, pad_id=pad_id)
            for t in range(len(tokens) - 1)
        ],
        axis=0,
    )

    x_t = torch.tensor(feats, dtype=torch.long, device=device)
    batch = int(max(1, config.boundary_batch_size))

    preds: List[np.ndarray] = []
    offset_t = torch.tensor(float(logit_offset), dtype=torch.float32, device=device)
    for i0 in range(0, len(feats), batch):
        logits = model(x_t[i0 : i0 + batch]) + offset_t
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds.append(probs.astype(np.float64, copy=False))

    out = np.concatenate(preds, axis=0)
    return np.clip(out, 0.0, 1.0)


def _encode_tokens_fixed_width(tokens: Sequence[int], *, width: int) -> str:
    width = int(width)
    if width < 2:
        raise ValueError("token_char_width must be >= 2")
    fmt_width = width - 1
    return "".join(f"{int(tok):0{fmt_width}d} " for tok in tokens)


def _signals_from_boundary_probabilities(
    probabilities: np.ndarray,
    *,
    token_char_width: int,
    source: str,
) -> List[ChunkFeedbackSignal]:
    width = int(token_char_width)
    signals: List[ChunkFeedbackSignal] = []
    for t, prob in enumerate(probabilities.tolist()):
        p_boundary = _clamp01(float(prob))
        low_info_prob = _clamp01(1.0 - p_boundary)
        start = int((t + 1) * width)
        end = int((t + 2) * width)
        signals.append(
            ChunkFeedbackSignal(
                start_char=start,
                end_char=end,
                low_info_probability=low_info_prob,
                noise_probability=0.0,
                confidence=1.0,
                source=str(source),
                metadata={
                    "boundary_index": int(t),
                    "p_boundary": p_boundary,
                },
            )
        )
    return signals


def _chunk_ends_from_chunks(
    *,
    chunks: Sequence[object],
    token_char_width: int,
    n_tokens: int,
) -> List[int]:
    width = int(token_char_width)
    ends: List[int] = []
    for chunk in chunks:
        end_char = int(getattr(chunk, "end_char"))
        end_excl = int(end_char // width)
        end_excl = max(1, min(int(n_tokens), end_excl))
        ends.append(int(end_excl - 1))
    if not ends:
        return [int(n_tokens - 1)]
    if ends[-1] != int(n_tokens - 1):
        ends[-1] = int(n_tokens - 1)

    deduped: List[int] = []
    for end in ends:
        if not deduped or end > deduped[-1]:
            deduped.append(int(end))

    if not deduped:
        deduped = [int(n_tokens - 1)]
    if deduped[-1] != int(n_tokens - 1):
        deduped.append(int(n_tokens - 1))
    return deduped


def _match_boundaries(
    *,
    predicted_boundaries: Sequence[int],
    true_boundaries: Sequence[int],
    tolerance: int,
) -> _BoundaryMatchStats:
    pred = sorted({int(x) for x in predicted_boundaries})
    truth = sorted({int(x) for x in true_boundaries})
    tol = int(max(0, tolerance))

    matched_truth: set[int] = set()
    tp = 0
    localization_sum = 0.0

    for p in pred:
        best_i: Optional[int] = None
        best_dist: Optional[int] = None
        for i, t in enumerate(truth):
            if i in matched_truth:
                continue
            dist = abs(int(p) - int(t))
            if dist > tol:
                continue
            if best_dist is None or dist < best_dist:
                best_i = i
                best_dist = dist
        if best_i is not None and best_dist is not None:
            matched_truth.add(int(best_i))
            tp += 1
            localization_sum += float(best_dist)

    fp = int(len(pred) - tp)
    fn = int(len(truth) - tp)
    return _BoundaryMatchStats(tp=tp, fp=fp, fn=fn, localization_error_sum=localization_sum)


def _boundary_cost_sum_from_positions(costs: np.ndarray, boundaries: Sequence[int]) -> float:
    if len(boundaries) == 0:
        return 0.0
    idx = np.asarray(boundaries, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(costs))]
    if len(idx) == 0:
        return 0.0
    return float(np.sum(costs[idx]))


def _loglik_drop_from_positions(
    *,
    doc: ChangepointMarkovDoc,
    tokens: np.ndarray,
    boundaries: Sequence[int],
    log_transitions: np.ndarray,
) -> float:
    if len(tokens) < 2 or len(boundaries) == 0:
        return 0.0

    transition_regimes = np.asarray(doc.transition_regimes, dtype=np.int64)
    total_drop = 0.0
    for t in boundaries:
        t_int = int(t)
        if t_int < 0 or t_int >= len(tokens) - 1:
            continue
        regime = int(transition_regimes[t_int])
        a = int(tokens[t_int])
        b = int(tokens[t_int + 1])
        total_drop += -float(log_transitions[regime, a, b])
    return float(total_drop)


def _evaluate_chunker_policy(
    docs: Sequence[Tuple[str, ChangepointMarkovDoc, str]],
    *,
    config: MarkovChangepointConfig,
    log_transitions: np.ndarray,
    adaptive_config: Optional[AdaptiveChunkingConfig],
    feedback_memory: AdaptiveChunkMemory,
    honest_policy: HonestChunkingPolicy,
    signal_role: str,
    strategy: str,
    base_max_chars: int,
) -> ChangepointPolicyMetrics:
    total_boundary_cost = 0.0
    total_boundaries = 0
    total_l1 = 0.0
    total_kl = 0.0
    total_loglik_drop = 0.0

    total_tp = 0
    total_fp = 0
    total_fn = 0
    localization_error_sum = 0.0

    total_true = 0
    total_pred = 0
    n_docs = 0

    for doc_id, doc, text in docs:
        toks = np.asarray(doc.tokens, dtype=np.int64)
        if len(toks) < 2:
            continue

        if signal_role == "none":
            signals: List[ChunkFeedbackSignal] = []
        elif signal_role == "chunking":
            signals = feedback_memory.get_signals_for_chunking(doc_id, honest_policy=honest_policy)
        elif signal_role == "evaluation":
            signals = feedback_memory.get_signals_for_evaluation(doc_id, honest_policy=honest_policy)
        else:
            raise ValueError(f"unknown signal_role: {signal_role!r}")

        chunks = chunk_for_ops(
            text,
            max_chars=int(base_max_chars),
            strategy=str(strategy),
            adaptive_config=adaptive_config,
            feedback_signals=signals,
        )
        ends = _chunk_ends_from_chunks(
            chunks=chunks,
            token_char_width=int(config.token_char_width),
            n_tokens=len(toks),
        )

        predicted_boundaries = [int(t) for t in ends[:-1] if int(t) < len(toks) - 1]
        true_boundaries = [int(t) for t in doc.true_boundaries if int(t) < len(toks) - 1]

        match = _match_boundaries(
            predicted_boundaries=predicted_boundaries,
            true_boundaries=true_boundaries,
            tolerance=int(config.boundary_tolerance_tokens),
        )

        total_tp += int(match.tp)
        total_fp += int(match.fp)
        total_fn += int(match.fn)
        localization_error_sum += float(match.localization_error_sum)

        total_true += len(true_boundaries)
        total_pred += len(predicted_boundaries)

        costs = _boundary_costs(toks, log_transitions=log_transitions)
        total_boundary_cost += _boundary_cost_sum_from_positions(costs, predicted_boundaries)
        total_boundaries += int(len(predicted_boundaries))

        ll = _loglik(toks, log_transitions)
        p_oracle = _softmax_np(ll)
        p_chunk = _chunked_posterior(
            toks,
            loglik_full=ll,
            segment_ends=ends,
            log_transitions=log_transitions,
        )
        total_l1 += _l1_discrepancy(p_oracle, p_chunk)
        total_kl += _kl_divergence(p_oracle, p_chunk)

        total_loglik_drop += _loglik_drop_from_positions(
            doc=doc,
            tokens=toks,
            boundaries=predicted_boundaries,
            log_transitions=log_transitions,
        )

        n_docs += 1

    denom = float(max(1, n_docs))
    precision = float(total_tp) / float(max(1, total_tp + total_fp))
    recall = float(total_tp) / float(max(1, total_tp + total_fn))
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )

    return ChangepointPolicyMetrics(
        mean_boundary_cost=float(total_boundary_cost / denom),
        mean_num_boundaries=float(total_boundaries / denom),
        mean_l1=float(total_l1 / denom),
        mean_kl=float(total_kl / denom),
        mean_loglik_drop=float(total_loglik_drop / denom),
        boundary_precision=float(precision),
        boundary_recall=float(recall),
        boundary_f1=float(f1),
        mean_localization_error=float(localization_error_sum / float(max(1, total_tp))),
        predicted_to_true_ratio=float(total_pred) / float(max(1, total_true)),
        total_predicted_boundaries=int(total_pred),
        total_true_boundaries=int(total_true),
        total_matched_boundaries=int(total_tp),
        n_docs=int(n_docs),
    )


def run_markov_changepoint_honesty_experiment(
    config: MarkovChangepointConfig,
    *,
    honest_policy: Optional[HonestChunkingPolicy] = None,
    adaptive_config: Optional[AdaptiveChunkingConfig] = None,
    strategy: str = "axis",
) -> MarkovChangepointHonestySummary:
    _validate_config(config)
    _set_global_seed(int(config.seed))

    if int(config.torch_threads) > 0:
        try:
            torch.set_num_threads(int(config.torch_threads))
        except RuntimeError:
            pass
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(int(config.torch_threads))
            except RuntimeError:
                pass

    if config.use_cuda and torch.cuda.is_available():
        if config.cuda_device is not None:
            cuda_idx = int(config.cuda_device)
            n_cuda = int(torch.cuda.device_count())
            if cuda_idx < 0 or cuda_idx >= n_cuda:
                raise ValueError(
                    f"cuda_device={cuda_idx} out of range; available devices: 0..{n_cuda - 1}"
                )
            torch.cuda.set_device(cuda_idx)
            device = torch.device(f"cuda:{cuda_idx}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rng = np.random.default_rng(int(config.seed))
    transitions = _make_transition_matrices(
        n_classes=int(config.n_regimes),
        vocab_size=int(config.vocab_size),
        log_std=float(config.transition_log_std),
        sinkhorn_iters=int(config.sinkhorn_iters),
        rng=rng,
    )
    log_transitions = np.log(transitions)

    docs = generate_changepoint_docs(config, transitions=transitions)
    train_docs = docs[: int(config.train_docs)]
    test_docs = docs[int(config.train_docs) :]

    x_train, y_train = _collect_boundary_training_data(train_docs, config=config, rng=rng)

    # Calibration: align training-time logits to deployment-time probabilities.
    total_true = int(sum(len(d.true_boundaries) for d in train_docs))
    total_possible = int(sum(max(0, len(d.tokens) - 1) for d in train_docs))
    pi_true = float(total_true) / float(max(1, total_possible))
    pi_train = float(np.mean(np.asarray(y_train, dtype=np.float64))) if len(y_train) > 0 else float(pi_true)
    pos_weight = _resolve_positive_weight(y_train=y_train, config=config)
    prior_offset = (_logit(pi_true) - _logit(pi_train)) if bool(config.calibrate_prior) else 0.0
    pos_weight_offset = -float(math.log(float(pos_weight))) if bool(config.calibrate_pos_weight) else 0.0
    logit_offset = float(prior_offset + pos_weight_offset)

    model = BoundaryChangepointPredictor(
        vocab_size=int(config.vocab_size),
        window_size=int(config.window_size),
        emb_dim=int(config.boundary_emb_dim),
        hidden_dim=int(config.boundary_hidden_dim),
    )
    train_loss = _train_boundary_model(
        model,
        x_train=x_train,
        y_train=y_train,
        config=config,
        device=device,
    )

    honest = honest_policy or HonestChunkingPolicy(enabled=True)
    token_char_width = int(config.token_char_width)

    if adaptive_config is None:
        adaptive_config = AdaptiveChunkingConfig(
            enabled=True,
            min_chars=int(config.min_leaf_tokens) * token_char_width,
            max_chars=int(config.max_leaf_tokens) * token_char_width,
            low_info_expansion_weight=1.0,
            noise_expansion_weight=0.0,
            high_info_compression_weight=1.0,
            proxy_blend=0.0,
        )

    base_max_chars = int(config.fixed_leaf_tokens) * token_char_width
    if base_max_chars < 256:
        raise ValueError("token_char_width/fixed_leaf_tokens combination makes base_max_chars < 256")

    memory = AdaptiveChunkMemory()
    encoded: List[Tuple[str, ChangepointMarkovDoc, str]] = []
    for i, doc in enumerate(docs):
        doc_id = f"changepoint_doc_{i}"
        toks = np.asarray(doc.tokens, dtype=np.int64)
        text = _encode_tokens_fixed_width(toks, width=token_char_width)

        oracle_probs = _oracle_boundary_probabilities(
            doc,
            tolerance_tokens=int(config.boundary_tolerance_tokens),
        )
        pred_probs = _predict_boundary_probabilities(
            model,
            toks,
            config=config,
            device=device,
            logit_offset=logit_offset,
        )

        pred_signals = _signals_from_boundary_probabilities(
            pred_probs,
            token_char_width=token_char_width,
            source="predicted_boundary_probability",
        )
        oracle_signals = _signals_from_boundary_probabilities(
            oracle_probs,
            token_char_width=token_char_width,
            source="oracle_boundary_probability",
        )

        memory.update_signals(
            doc_id,
            pred_signals,
            honest_role=honest.boundary_role,
            replace_existing=True,
        )
        memory.update_signals(
            doc_id,
            oracle_signals,
            honest_role=honest.evaluation_role,
            replace_existing=False,
        )

        encoded.append((doc_id, doc, text))

    test_encoded = encoded[int(config.train_docs) :]

    metrics: Dict[str, ChangepointPolicyMetrics] = {}
    metrics["fixed"] = _evaluate_chunker_policy(
        test_encoded,
        config=config,
        log_transitions=log_transitions,
        adaptive_config=None,
        feedback_memory=memory,
        honest_policy=honest,
        signal_role="none",
        strategy=str(strategy),
        base_max_chars=int(base_max_chars),
    )
    metrics["chunker_honest"] = _evaluate_chunker_policy(
        test_encoded,
        config=config,
        log_transitions=log_transitions,
        adaptive_config=adaptive_config,
        feedback_memory=memory,
        honest_policy=honest,
        signal_role="chunking",
        strategy=str(strategy),
        base_max_chars=int(base_max_chars),
    )
    metrics["chunker_leaky"] = _evaluate_chunker_policy(
        test_encoded,
        config=config,
        log_transitions=log_transitions,
        adaptive_config=adaptive_config,
        feedback_memory=memory,
        honest_policy=honest,
        signal_role="evaluation",
        strategy=str(strategy),
        base_max_chars=int(base_max_chars),
    )

    cfg_dict = asdict(config)
    cfg_dict["device_used"] = str(device)
    cfg_dict["adaptive_config"] = asdict(adaptive_config)
    cfg_dict["honest_policy"] = asdict(honest)
    cfg_dict["chunk_strategy"] = str(strategy)
    cfg_dict["calibration_pi_true"] = float(pi_true)
    cfg_dict["calibration_pi_train"] = float(pi_train)
    cfg_dict["boundary_pos_weight_used"] = float(pos_weight)
    cfg_dict["calibration_prior_offset"] = float(prior_offset)
    cfg_dict["calibration_pos_weight_offset"] = float(pos_weight_offset)
    cfg_dict["calibration_logit_offset"] = float(logit_offset)

    return MarkovChangepointHonestySummary(
        config=cfg_dict,
        boundary_model_train_loss_final=float(train_loss),
        boundary_train_positive_rate=float(np.mean(y_train)) if len(y_train) > 0 else 0.0,
        metrics=metrics,
    )


__all__ = [
    "ChangepointMarkovDoc",
    "BoundaryChangepointPredictor",
    "ChangepointPolicyMetrics",
    "MarkovChangepointConfig",
    "MarkovChangepointHonestySummary",
    "generate_changepoint_docs",
    "run_markov_changepoint_honesty_experiment",
]
