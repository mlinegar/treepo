"""
Markov changepoint honesty simulation with an end-to-end preference objective.

This module is a "v2" simulation intended to replace leaky-relative reporting.
It keeps the real adaptive chunker + honest role split, but evaluates:

1. Boundary detection quality (precision/recall/F1 with tolerance matching)
2. A toy preference-learning *gap to oracle optimum* induced by chunking.

The preference objective is intentionally simple and oracle-measurable:
the oracle value for a document is the (integer) number of true changepoints
in that document. A preference sample prefers the correct count over an
incorrect count. When chunking preserves the oracle value, the preference gap
is zero; otherwise it is positive and scales with distortion.
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
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for Markov changepoint preference simulations. "
        "Install with: pip install torch>=2.0.0"
    ) from e

from treepo._research.preprocessing.chunker import (
    AdaptiveChunkMemory,
    AdaptiveChunkingConfig,
    ChunkFeedbackSignal,
    HonestChunkingPolicy,
    chunk_for_ops,
)
from treepo._research.tree.markov_boundary_honesty_simulation import (
    _make_transition_matrices,
    _set_global_seed,
    _window_features,
)
from treepo._research.tree.markov_changepoint_honesty_simulation import (
    BoundaryChangepointPredictor,
    ChangepointMarkovDoc,
    MarkovChangepointConfig,
    generate_changepoint_docs,
)


@dataclass(frozen=True)
class MarkovChangepointPreferenceConfig(MarkovChangepointConfig):
    dpo_beta: float = 1.0
    dpo_policy_sharpness: float = 2.5
    dpo_negatives_per_doc: int = 8


@dataclass(frozen=True)
class _BoundaryMatchStats:
    tp: int
    fp: int
    fn: int


@dataclass(frozen=True)
class ChangepointPreferencePolicyMetrics:
    # Boundary-set evaluation
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    predicted_to_true_ratio: float

    total_predicted_boundaries: int
    total_true_boundaries: int
    total_matched_boundaries: int
    n_docs: int

    # Count distortion
    mean_predicted_boundary_count: float
    mean_true_boundary_count: float
    mean_signed_count_error: float
    mean_abs_count_error: float

    # Preference gap to oracle optimum (true environment)
    mean_dpo_loss_true_env: float
    mean_dpo_loss_true_opt: float
    mean_dpo_loss_gap_to_opt: float


@dataclass(frozen=True)
class MarkovChangepointPreferenceSummary:
    config: Dict[str, object]
    boundary_model_train_loss_final: float
    boundary_train_positive_rate: float
    action_max_count: int
    metrics: Dict[str, ChangepointPreferencePolicyMetrics]

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "boundary_model_train_loss_final": self.boundary_model_train_loss_final,
            "boundary_train_positive_rate": self.boundary_train_positive_rate,
            "action_max_count": int(self.action_max_count),
            "metrics": {k: asdict(v) for k, v in self.metrics.items()},
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def _validate_config(config: MarkovChangepointPreferenceConfig) -> None:
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
    if float(config.dpo_beta) <= 0:
        raise ValueError("dpo_beta must be > 0")
    if float(config.dpo_policy_sharpness) <= 0:
        raise ValueError("dpo_policy_sharpness must be > 0")
    if int(config.dpo_negatives_per_doc) < 1:
        raise ValueError("dpo_negatives_per_doc must be >= 1")


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


def _oracle_boundary_probabilities(doc: ChangepointMarkovDoc, *, tolerance_tokens: int) -> np.ndarray:
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
            score = 1.0 - (float(dist) / float(radius + 1))
            if score > out[t]:
                out[t] = score
    return np.clip(out, 0.0, 1.0)


def _collect_boundary_training_data(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    config: MarkovChangepointPreferenceConfig,
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
        keep = (
            rng.choice(all_idx, size=max_samples, replace=False)
            if len(all_idx) > max_samples
            else all_idx
        )

    keep = np.asarray(keep, dtype=np.int64)
    rng.shuffle(keep)
    return x_arr[keep], y_arr[keep]


def _resolve_positive_weight(y_train: np.ndarray, *, config: MarkovChangepointPreferenceConfig) -> float:
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
    config: MarkovChangepointPreferenceConfig,
    device: torch.device,
) -> float:
    model.to(device)
    model.train()

    if len(x_train) == 0:
        return float("nan")

    x_t = torch.tensor(x_train, dtype=torch.long, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    positive_weight = _resolve_positive_weight(y_train, config=config)
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
    config: MarkovChangepointPreferenceConfig,
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


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _logit(p: float) -> float:
    p = float(p)
    p = max(1e-8, min(1.0 - 1e-8, p))
    return float(math.log(p / (1.0 - p)))


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
        if best_i is not None:
            matched_truth.add(int(best_i))
            tp += 1

    fp = int(len(pred) - tp)
    fn = int(len(truth) - tp)
    return _BoundaryMatchStats(tp=tp, fp=fp, fn=fn)


def _dp_segment_ends_maximize_cost(
    costs: np.ndarray,
    *,
    min_leaf: int,
    max_leaf: int,
) -> List[int]:
    """
    DP segmenter that returns token end indices, maximizing sum(cost[t]) over cuts.

    Args:
        costs: length (n_tokens-1). costs[t] is earned if we cut after token t.
        min_leaf/max_leaf: segment-length constraints in tokens.
    """
    t_minus_1 = int(costs.shape[0])
    n_tokens = t_minus_1 + 1
    if n_tokens <= 0:
        return []
    if n_tokens < int(min_leaf):
        return [int(n_tokens - 1)]

    neg_inf = float("-inf")
    dp = np.full(n_tokens + 1, neg_inf, dtype=np.float64)
    prev = np.full(n_tokens + 1, -1, dtype=np.int64)
    dp[0] = 0.0

    for end_excl in range(1, n_tokens + 1):
        lo = max(0, int(end_excl) - int(max_leaf))
        hi = int(end_excl) - int(min_leaf)
        if hi < lo:
            continue
        best = neg_inf
        best_start = -1
        for start in range(lo, hi + 1):
            base = float(dp[int(start)])
            if not math.isfinite(base):
                continue
            add = 0.0
            if int(end_excl) < int(n_tokens):
                t = int(end_excl - 1)
                if 0 <= t < int(costs.shape[0]):
                    add = float(costs[t])
            score = base + add
            if score > best:
                best = score
                best_start = int(start)
        if best_start >= 0:
            dp[int(end_excl)] = best
            prev[int(end_excl)] = best_start

    if prev[int(n_tokens)] < 0:
        return [int(n_tokens - 1)]

    ends: List[int] = []
    end = int(n_tokens)
    while end > 0:
        start = int(prev[end])
        if start < 0:
            break
        ends.append(int(end - 1))
        end = start
    ends.reverse()
    if len(ends) == 0:
        ends = [int(n_tokens - 1)]
    ends[-1] = int(n_tokens - 1)
    return ends


def _oracle_cut_ends(
    doc: ChangepointMarkovDoc,
    *,
    min_leaf_tokens: int,
    max_leaf_tokens: int,
) -> List[int]:
    n_tokens = int(len(doc.tokens))
    if n_tokens <= 0:
        return []
    ends = [int(t) for t in doc.true_boundaries if int(t) < n_tokens - 1]
    ends.append(int(n_tokens - 1))

    prev = -1
    feasible = True
    for end in ends:
        length = int(end - prev)
        if length < int(min_leaf_tokens) or length > int(max_leaf_tokens):
            feasible = False
            break
        prev = int(end)
    if feasible:
        return ends

    costs = np.zeros((max(0, n_tokens - 1),), dtype=np.float64)
    for t in doc.true_boundaries:
        ti = int(t)
        if 0 <= ti < n_tokens - 1:
            costs[ti] = 1.0
    return _dp_segment_ends_maximize_cost(
        costs,
        min_leaf=int(min_leaf_tokens),
        max_leaf=int(max_leaf_tokens),
    )


def _log_softmax_from_center(*, center: int, action_max: int, sharpness: float) -> np.ndarray:
    center = int(max(0, min(int(action_max), int(center))))
    a = np.arange(int(action_max) + 1, dtype=np.int64)
    logits = -float(sharpness) * np.abs(a - center).astype(np.float64)
    logits = logits - float(np.max(logits))
    log_z = float(np.log(np.sum(np.exp(logits))))
    return logits - log_z


def _neg_log_sigmoid(x: float) -> float:
    return float(np.logaddexp(0.0, -float(x)))


def _dpo_loss_for_doc(*, winner: int, losers: Sequence[int], log_probs: np.ndarray, beta: float) -> float:
    w = int(winner)
    if w < 0 or w >= int(log_probs.shape[0]):
        raise ValueError("winner out of range for action space")
    losses: List[float] = []
    for l in losers:
        li = int(l)
        if li == w:
            continue
        if li < 0 or li >= int(log_probs.shape[0]):
            continue
        logit = float(beta) * float(log_probs[w] - log_probs[li])
        losses.append(_neg_log_sigmoid(logit))
    if len(losses) == 0:
        return 0.0
    return float(np.mean(np.asarray(losses, dtype=np.float64)))


def _sample_losers(*, winner: int, action_max: int, n: int, rng: np.random.Generator) -> List[int]:
    winner = int(winner)
    action_max = int(action_max)
    n = int(n)
    if action_max < 1:
        return []
    choices = [a for a in range(action_max + 1) if a != winner]
    if not choices:
        return []
    return [int(rng.choice(choices)) for _ in range(n)]


def run_markov_changepoint_preference_experiment(
    config: MarkovChangepointPreferenceConfig,
    *,
    honest_policy: Optional[HonestChunkingPolicy] = None,
    adaptive_config: Optional[AdaptiveChunkingConfig] = None,
    strategy: str = "axis",
) -> MarkovChangepointPreferenceSummary:
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

    docs = generate_changepoint_docs(config, transitions=transitions)
    train_docs = docs[: int(config.train_docs)]
    test_docs = docs[int(config.train_docs) :]

    x_train, y_train = _collect_boundary_training_data(train_docs, config=config, rng=rng)

    total_true = int(sum(len(d.true_boundaries) for d in train_docs))
    total_possible = int(sum(max(0, len(d.tokens) - 1) for d in train_docs))
    pi_true = float(total_true) / float(max(1, total_possible))
    pi_train = (
        float(np.mean(np.asarray(y_train, dtype=np.float64))) if len(y_train) > 0 else float(pi_true)
    )
    pos_weight = _resolve_positive_weight(y_train, config=config)
    prior_offset = (_logit(pi_true) - _logit(pi_train)) if bool(config.calibrate_prior) else 0.0
    pos_weight_offset = (
        -float(math.log(float(pos_weight))) if bool(config.calibrate_pos_weight) else 0.0
    )
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
            doc, tolerance_tokens=int(config.boundary_tolerance_tokens)
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

    policies = ("fixed", "chunker_honest", "chunker_leaky", "oracle_cut")

    per_policy_y_true: Dict[str, List[int]] = {p: [] for p in policies}
    per_policy_y_pred: Dict[str, List[int]] = {p: [] for p in policies}
    per_policy_match: Dict[str, List[_BoundaryMatchStats]] = {p: [] for p in policies}

    for doc_id, doc, text in test_encoded:
        toks = np.asarray(doc.tokens, dtype=np.int64)
        if len(toks) < 2:
            continue

        true_boundaries = [int(t) for t in doc.true_boundaries if int(t) < len(toks) - 1]
        y_true = int(len(true_boundaries))

        chunks_fixed = chunk_for_ops(
            text,
            max_chars=int(base_max_chars),
            strategy=str(strategy),
            adaptive_config=None,
            feedback_signals=[],
        )
        ends_fixed = _chunk_ends_from_chunks(
            chunks=chunks_fixed,
            token_char_width=token_char_width,
            n_tokens=len(toks),
        )
        pred_fixed = [int(t) for t in ends_fixed[:-1] if int(t) < len(toks) - 1]

        signals_honest = memory.get_signals_for_chunking(doc_id, honest_policy=honest)
        chunks_honest = chunk_for_ops(
            text,
            max_chars=int(base_max_chars),
            strategy=str(strategy),
            adaptive_config=adaptive_config,
            feedback_signals=signals_honest,
        )
        ends_honest = _chunk_ends_from_chunks(
            chunks=chunks_honest,
            token_char_width=token_char_width,
            n_tokens=len(toks),
        )
        pred_honest = [int(t) for t in ends_honest[:-1] if int(t) < len(toks) - 1]

        signals_leaky = memory.get_signals_for_evaluation(doc_id, honest_policy=honest)
        chunks_leaky = chunk_for_ops(
            text,
            max_chars=int(base_max_chars),
            strategy=str(strategy),
            adaptive_config=adaptive_config,
            feedback_signals=signals_leaky,
        )
        ends_leaky = _chunk_ends_from_chunks(
            chunks=chunks_leaky,
            token_char_width=token_char_width,
            n_tokens=len(toks),
        )
        pred_leaky = [int(t) for t in ends_leaky[:-1] if int(t) < len(toks) - 1]

        ends_oracle = _oracle_cut_ends(
            doc,
            min_leaf_tokens=int(config.min_leaf_tokens),
            max_leaf_tokens=int(config.max_leaf_tokens),
        )
        pred_oracle = [int(t) for t in ends_oracle[:-1] if int(t) < len(toks) - 1]

        for name, pred in (
            ("fixed", pred_fixed),
            ("chunker_honest", pred_honest),
            ("chunker_leaky", pred_leaky),
            ("oracle_cut", pred_oracle),
        ):
            per_policy_y_true[name].append(int(y_true))
            per_policy_y_pred[name].append(int(len(pred)))
            per_policy_match[name].append(
                _match_boundaries(
                    predicted_boundaries=pred,
                    true_boundaries=true_boundaries,
                    tolerance=int(config.boundary_tolerance_tokens),
                )
            )

    action_max = 1
    for p in policies:
        if per_policy_y_true[p]:
            action_max = max(action_max, int(max(per_policy_y_true[p])))
        if per_policy_y_pred[p]:
            action_max = max(action_max, int(max(per_policy_y_pred[p])))

    rng_dpo = np.random.default_rng(int(config.seed) + 202)

    y_true_ref = per_policy_y_true["fixed"]
    losers_true_env: List[List[int]] = [
        _sample_losers(
            winner=int(y_true),
            action_max=int(action_max),
            n=int(config.dpo_negatives_per_doc),
            rng=rng_dpo,
        )
        for y_true in y_true_ref
    ]

    opt_true_losses: List[float] = []
    for y_true, losers in zip(y_true_ref, losers_true_env):
        logp_opt = _log_softmax_from_center(
            center=int(y_true),
            action_max=int(action_max),
            sharpness=float(config.dpo_policy_sharpness),
        )
        opt_true_losses.append(
            _dpo_loss_for_doc(
                winner=int(y_true),
                losers=losers,
                log_probs=logp_opt,
                beta=float(config.dpo_beta),
            )
        )
    mean_opt_true = (
        float(np.mean(np.asarray(opt_true_losses, dtype=np.float64)))
        if opt_true_losses
        else float("nan")
    )

    metrics: Dict[str, ChangepointPreferencePolicyMetrics] = {}
    for policy in policies:
        y_true_list = per_policy_y_true[policy]
        y_pred_list = per_policy_y_pred[policy]
        matches = per_policy_match[policy]
        if not y_true_list:
            continue

        total_tp = int(sum(m.tp for m in matches))
        total_fp = int(sum(m.fp for m in matches))
        total_fn = int(sum(m.fn for m in matches))
        total_true = int(sum(int(x) for x in y_true_list))
        total_pred = int(sum(int(x) for x in y_pred_list))
        n_docs = int(len(y_true_list))

        precision = float(total_tp) / float(max(1, total_tp + total_fp))
        recall = float(total_tp) / float(max(1, total_tp + total_fn))
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )

        y_true_arr = np.asarray(y_true_list, dtype=np.float64)
        y_pred_arr = np.asarray(y_pred_list, dtype=np.float64)
        diff = y_pred_arr - y_true_arr

        true_env_losses: List[float] = []
        for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
            logp_pred = _log_softmax_from_center(
                center=int(y_pred),
                action_max=int(action_max),
                sharpness=float(config.dpo_policy_sharpness),
            )
            losers = losers_true_env[i] if i < len(losers_true_env) else []
            true_env_losses.append(
                _dpo_loss_for_doc(
                    winner=int(y_true),
                    losers=losers,
                    log_probs=logp_pred,
                    beta=float(config.dpo_beta),
                )
            )

        mean_true_env = float(np.mean(np.asarray(true_env_losses, dtype=np.float64)))
        metrics[policy] = ChangepointPreferencePolicyMetrics(
            boundary_precision=float(precision),
            boundary_recall=float(recall),
            boundary_f1=float(f1),
            predicted_to_true_ratio=float(total_pred) / float(max(1, total_true)),
            total_predicted_boundaries=int(total_pred),
            total_true_boundaries=int(total_true),
            total_matched_boundaries=int(total_tp),
            n_docs=int(n_docs),
            mean_predicted_boundary_count=float(np.mean(y_pred_arr)),
            mean_true_boundary_count=float(np.mean(y_true_arr)),
            mean_signed_count_error=float(np.mean(diff)),
            mean_abs_count_error=float(np.mean(np.abs(diff))),
            mean_dpo_loss_true_env=float(mean_true_env),
            mean_dpo_loss_true_opt=float(mean_opt_true),
            mean_dpo_loss_gap_to_opt=float(mean_true_env - mean_opt_true),
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

    return MarkovChangepointPreferenceSummary(
        config=cfg_dict,
        boundary_model_train_loss_final=float(train_loss),
        boundary_train_positive_rate=float(np.mean(y_train)) if len(y_train) > 0 else 0.0,
        action_max_count=int(action_max),
        metrics=metrics,
    )


__all__ = [
    "MarkovChangepointPreferenceConfig",
    "ChangepointPreferencePolicyMetrics",
    "MarkovChangepointPreferenceSummary",
    "run_markov_changepoint_preference_experiment",
]
