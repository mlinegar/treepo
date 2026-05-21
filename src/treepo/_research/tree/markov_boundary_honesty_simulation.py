"""
Markov-posterior boundary honesty simulation (adaptive chunking toy model).

Goal
----
Stress *boundary placement* as the binding constraint in a tree decomposition.
We classify token sequences under class-conditional Markov chains (order-1),
but we only allow the tree state to preserve *within-leaf* transitions.
Cross-leaf boundary transitions are dropped, so where we cut determines what
information is lost.

This is a controlled analogue of "adaptive tracker learns chunk boundaries":
learn a boundary cost predictor from local windows, then segment documents
to avoid cutting *class-informative* bigrams.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for Markov boundary honesty simulations. "
        "Install with: pip install torch>=2.0.0"
    ) from e


@dataclass(frozen=True)
class MarkovDoc:
    label: int
    tokens: Tuple[int, ...]


@dataclass(frozen=True)
class MarkovBoundaryConfig:
    n_classes: int = 5
    vocab_size: int = 128
    min_tokens: int = 128
    max_tokens: int = 256
    min_leaf_tokens: int = 16
    max_leaf_tokens: int = 64
    fixed_leaf_tokens: int = 64
    train_docs: int = 200
    test_docs: int = 80
    sinkhorn_iters: int = 40
    transition_log_std: float = 1.25
    window_size: int = 1
    boundary_emb_dim: int = 32
    boundary_hidden_dim: int = 64
    boundary_batch_size: int = 256
    boundary_max_train_samples: int = 120000
    n_epochs: int = 6
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    seed: int = 0
    use_cuda: bool = True
    cuda_device: Optional[int] = None
    torch_threads: int = 0


@dataclass(frozen=True)
class PolicyMetrics:
    mean_boundary_cost: float
    mean_num_boundaries: float
    mean_l1: float
    mean_kl: float
    accuracy: float
    n_docs: int


@dataclass(frozen=True)
class MarkovBoundarySummary:
    config: Dict[str, object]
    boundary_model_train_loss_final: float
    metrics: Dict[str, PolicyMetrics]

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "boundary_model_train_loss_final": self.boundary_model_train_loss_final,
            "metrics": {k: asdict(v) for k, v in self.metrics.items()},
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x = x - float(np.max(x))
    exp = np.exp(x)
    return exp / float(np.sum(exp))


def _softmax_cols(x: np.ndarray) -> np.ndarray:
    """Column-wise softmax for array of shape (C, N)."""
    x = x.astype(np.float64, copy=False)
    x = x - np.max(x, axis=0, keepdims=True)
    exp = np.exp(x)
    denom = np.sum(exp, axis=0, keepdims=True)
    return exp / denom


def _l1_discrepancy(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.mean(np.abs(p - q)))


def _kl_divergence(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


def _sinkhorn_doubly_stochastic(x: np.ndarray, *, iters: int) -> np.ndarray:
    mat = np.asarray(x, dtype=np.float64).copy()
    mat = np.maximum(mat, 1e-12)
    for _ in range(int(max(1, iters))):
        mat /= np.sum(mat, axis=1, keepdims=True)
        mat /= np.sum(mat, axis=0, keepdims=True)
    mat /= np.sum(mat, axis=1, keepdims=True)
    return mat


def _make_transition_matrices(
    *,
    n_classes: int,
    vocab_size: int,
    log_std: float,
    sinkhorn_iters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.empty((int(n_classes), int(vocab_size), int(vocab_size)), dtype=np.float64)
    for c in range(int(n_classes)):
        raw = rng.normal(loc=0.0, scale=float(log_std), size=(int(vocab_size), int(vocab_size)))
        weights = np.exp(raw)
        out[c] = _sinkhorn_doubly_stochastic(weights, iters=int(sinkhorn_iters))
    out = np.maximum(out, 1e-12)
    out /= np.sum(out, axis=2, keepdims=True)
    return out


def _sample_markov_tokens(
    cdf: np.ndarray,
    *,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    vocab_size = int(cdf.shape[0])
    toks = np.empty(int(length), dtype=np.int64)
    toks[0] = int(rng.integers(0, vocab_size))
    for t in range(int(length) - 1):
        row = cdf[int(toks[t])]
        u = float(rng.random())
        toks[t + 1] = int(np.searchsorted(row, u, side="right"))
        if toks[t + 1] >= vocab_size:
            toks[t + 1] = vocab_size - 1
    return toks


def generate_docs(
    config: MarkovBoundaryConfig,
    *,
    transitions: np.ndarray,
) -> Tuple[MarkovDoc, ...]:
    rng = np.random.default_rng(int(config.seed))
    n_docs = int(config.train_docs) + int(config.test_docs)
    cdfs = np.cumsum(transitions, axis=2)
    cdfs[:, :, -1] = 1.0

    docs: List[MarkovDoc] = []
    for _ in range(n_docs):
        label = int(rng.integers(0, int(config.n_classes)))
        length = int(rng.integers(int(config.min_tokens), int(config.max_tokens) + 1))
        toks = _sample_markov_tokens(cdfs[label], length=length, rng=rng)
        docs.append(MarkovDoc(label=label, tokens=tuple(int(x) for x in toks)))
    return tuple(docs)


def _loglik(tokens: np.ndarray, log_transitions: np.ndarray) -> np.ndarray:
    a = tokens[:-1]
    b = tokens[1:]
    return log_transitions[:, a, b].sum(axis=1)


def _boundary_costs(
    tokens: np.ndarray,
    *,
    log_transitions: np.ndarray,
) -> np.ndarray:
    a = tokens[:-1]
    b = tokens[1:]
    boundary_ll = log_transitions[:, a, b]  # (C, T-1)

    # Make boundary costs *locally learnable* from a small token window by
    # scoring how class-informative the boundary bigram is. This avoids a
    # degenerate regime where full-sequence posteriors become near one-hot and
    # "remove one bigram" sensitivity collapses to ~0 for all boundaries.
    #
    # We use 1 - normalized entropy of p(class | bigram).
    posterior_bigram = _softmax_cols(boundary_ll)
    entropy = -np.sum(posterior_bigram * np.log(posterior_bigram + 1e-12), axis=0)
    max_entropy = float(math.log(float(posterior_bigram.shape[0])))
    if max_entropy <= 0:
        return np.zeros((int(boundary_ll.shape[1]),), dtype=np.float64)
    costs = 1.0 - (entropy / max_entropy)
    return costs.astype(np.float64, copy=False)


def _fixed_segment_ends(
    n_tokens: int,
    *,
    min_leaf: int,
    max_leaf: int,
    target_leaf: int,
) -> List[int]:
    if n_tokens <= 0:
        return []
    target = int(max(min_leaf, min(max_leaf, target_leaf)))
    ends: List[int] = []
    start = 0
    while start < n_tokens:
        end_excl = min(n_tokens, start + target)
        remaining = n_tokens - end_excl
        if remaining != 0 and remaining < int(min_leaf):
            end_excl = max(start + int(min_leaf), n_tokens - int(min_leaf))
        end_excl = min(n_tokens, max(end_excl, start + int(min_leaf)))
        ends.append(int(end_excl - 1))
        start = int(end_excl)
    ends[-1] = int(n_tokens - 1)
    return ends


def _dp_segment_ends(
    costs: np.ndarray,
    *,
    min_leaf: int,
    max_leaf: int,
    maximize: bool,
) -> List[int]:
    """
    Segment tokens into contiguous leaves with length constraints.

    Args:
        costs: length T-1 array. costs[t] is incurred if we cut after token t.
        min_leaf/max_leaf: leaf length constraints in tokens.
        maximize: if True, maximize total cost; else minimize.
    """
    t_minus_1 = int(costs.shape[0])
    n_tokens = t_minus_1 + 1
    if n_tokens <= 0:
        return []
    if n_tokens < int(min_leaf):
        return [n_tokens - 1]

    inf = float("inf")
    dp = np.full(n_tokens + 1, inf, dtype=np.float64)
    prev = np.full(n_tokens + 1, -1, dtype=np.int64)
    dp[0] = 0.0
    if maximize:
        dp[:] = -inf
        dp[0] = 0.0

    for start in range(n_tokens):
        base = float(dp[start])
        if (not maximize and not math.isfinite(base)) or (maximize and not math.isfinite(base)):
            continue
        for length in range(int(min_leaf), int(max_leaf) + 1):
            end = start + length
            if end > n_tokens:
                break
            add = 0.0 if end == n_tokens else float(costs[end - 1])
            cand = base + add
            if maximize:
                if cand > float(dp[end]):
                    dp[end] = cand
                    prev[end] = start
            else:
                if cand < float(dp[end]):
                    dp[end] = cand
                    prev[end] = start

    if prev[n_tokens] < 0:
        return _fixed_segment_ends(
            n_tokens,
            min_leaf=int(min_leaf),
            max_leaf=int(max_leaf),
            target_leaf=int(max_leaf),
        )

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
        ends = [n_tokens - 1]
    ends[-1] = n_tokens - 1
    return ends


def _chunked_posterior(
    tokens: np.ndarray,
    *,
    loglik_full: np.ndarray,
    segment_ends: Sequence[int],
    log_transitions: np.ndarray,
) -> np.ndarray:
    if len(tokens) <= 1 or len(segment_ends) <= 1:
        return _softmax_np(loglik_full)
    cut_positions = [int(x) for x in segment_ends[:-1] if int(x) < len(tokens) - 1]
    if len(cut_positions) == 0:
        return _softmax_np(loglik_full)
    a = tokens[np.array(cut_positions, dtype=np.int64)]
    b = tokens[np.array(cut_positions, dtype=np.int64) + 1]
    delta = log_transitions[:, a, b].sum(axis=1)
    return _softmax_np(loglik_full - delta)


def _boundary_cost_sum(costs: np.ndarray, segment_ends: Sequence[int]) -> float:
    if len(segment_ends) <= 1:
        return 0.0
    total = 0.0
    for t in segment_ends[:-1]:
        total += float(costs[int(t)])
    return total


def _window_features(
    tokens: np.ndarray,
    *,
    boundary_idx: int,
    window: int,
    pad_id: int,
) -> np.ndarray:
    t = int(boundary_idx)
    w = int(window)
    left = tokens[max(0, t - w + 1) : t + 1]
    right = tokens[t + 1 : min(len(tokens), t + 1 + w)]
    if len(left) < w:
        left = np.concatenate([np.full(w - len(left), int(pad_id), dtype=np.int64), left])
    if len(right) < w:
        right = np.concatenate([right, np.full(w - len(right), int(pad_id), dtype=np.int64)])
    return np.concatenate([left, right]).astype(np.int64, copy=False)


class BoundaryHarmPredictor(nn.Module):
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
        emb = self.embedding(x)  # (B, 2W, D)
        flat = emb.reshape(emb.shape[0], -1)
        out = self.net(flat).squeeze(-1)
        return torch.sigmoid(out)


def _collect_boundary_training_data(
    docs: Sequence[MarkovDoc],
    *,
    config: MarkovBoundaryConfig,
    log_transitions: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    pad_id = int(config.vocab_size)
    window = int(config.window_size)
    max_samples = int(config.boundary_max_train_samples)
    per_doc = max(1, int(math.ceil(float(max_samples) / float(max(1, len(docs))))))

    xs: List[np.ndarray] = []
    ys: List[float] = []
    for doc in docs:
        toks = np.asarray(doc.tokens, dtype=np.int64)
        if len(toks) < 2:
            continue
        costs = _boundary_costs(toks, log_transitions=log_transitions)
        idxs = np.arange(len(costs), dtype=np.int64)
        if len(idxs) > per_doc:
            idxs = rng.choice(idxs, size=int(per_doc), replace=False)
        for t in idxs:
            xs.append(
                _window_features(toks, boundary_idx=int(t), window=window, pad_id=pad_id)
            )
            ys.append(float(costs[int(t)]))
        if len(xs) >= max_samples:
            break
    if len(xs) == 0:
        return np.zeros((0, 2 * window), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    x_arr = np.stack(xs, axis=0)
    y_arr = np.asarray(ys, dtype=np.float32)
    return x_arr, y_arr


def _train_boundary_model(
    model: BoundaryHarmPredictor,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: MarkovBoundaryConfig,
    device: torch.device,
) -> float:
    model.to(device)
    model.train()

    if len(x_train) == 0:
        return float("nan")

    x_t = torch.tensor(x_train, dtype=torch.long, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )

    batch = int(max(1, config.boundary_batch_size))
    idxs = np.arange(len(x_train), dtype=np.int64)
    rng = np.random.default_rng(int(config.seed) + 11)

    loss_final = float("nan")
    for _ in range(int(config.n_epochs)):
        rng.shuffle(idxs)
        losses: List[float] = []
        for i0 in range(0, len(idxs), batch):
            batch_idxs = idxs[i0 : i0 + batch]
            opt.zero_grad(set_to_none=True)
            pred = model(x_t[batch_idxs])
            loss = F.mse_loss(pred, y_t[batch_idxs], reduction="mean")
            loss.backward()
            if float(config.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        loss_final = float(np.mean(np.array(losses, dtype=np.float64)))
    return loss_final


@torch.no_grad()
def _predict_boundary_costs(
    model: BoundaryHarmPredictor,
    tokens: np.ndarray,
    *,
    config: MarkovBoundaryConfig,
    device: torch.device,
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
    for i0 in range(0, len(feats), batch):
        out = model(x_t[i0 : i0 + batch]).detach().cpu().numpy()
        preds.append(out.astype(np.float64, copy=False))
    return np.concatenate(preds, axis=0)


def _evaluate_policy(
    docs: Sequence[MarkovDoc],
    *,
    config: MarkovBoundaryConfig,
    log_transitions: np.ndarray,
    boundary_model: Optional[BoundaryHarmPredictor],
    policy: str,
    device: torch.device,
) -> PolicyMetrics:
    total_cost = 0.0
    total_boundaries = 0.0
    total_l1 = 0.0
    total_kl = 0.0
    correct = 0
    n_docs = 0

    for doc in docs:
        toks = np.asarray(doc.tokens, dtype=np.int64)
        if len(toks) < 2:
            continue
        ll = _loglik(toks, log_transitions)
        p_oracle = _softmax_np(ll)
        costs = _boundary_costs(toks, log_transitions=log_transitions)

        if policy == "fixed":
            ends = _fixed_segment_ends(
                len(toks),
                min_leaf=int(config.min_leaf_tokens),
                max_leaf=int(config.max_leaf_tokens),
                target_leaf=int(config.fixed_leaf_tokens),
            )
        elif policy == "oracle":
            ends = _dp_segment_ends(
                costs,
                min_leaf=int(config.min_leaf_tokens),
                max_leaf=int(config.max_leaf_tokens),
                maximize=False,
            )
        elif policy == "worst":
            ends = _dp_segment_ends(
                costs,
                min_leaf=int(config.min_leaf_tokens),
                max_leaf=int(config.max_leaf_tokens),
                maximize=True,
            )
        elif policy == "learned":
            if boundary_model is None:
                raise ValueError("learned policy requested but no boundary model provided")
            pred_costs = _predict_boundary_costs(boundary_model, toks, config=config, device=device)
            ends = _dp_segment_ends(
                pred_costs,
                min_leaf=int(config.min_leaf_tokens),
                max_leaf=int(config.max_leaf_tokens),
                maximize=False,
            )
        else:
            raise ValueError(f"unknown policy: {policy!r}")

        p_chunk = _chunked_posterior(
            toks, loglik_full=ll, segment_ends=ends, log_transitions=log_transitions
        )
        total_cost += _boundary_cost_sum(costs, ends)
        total_boundaries += float(max(0, len(ends) - 1))
        total_l1 += _l1_discrepancy(p_oracle, p_chunk)
        total_kl += _kl_divergence(p_oracle, p_chunk)
        correct += int(int(np.argmax(p_chunk)) == int(doc.label))
        n_docs += 1

    denom = float(max(1, n_docs))
    return PolicyMetrics(
        mean_boundary_cost=float(total_cost / denom),
        mean_num_boundaries=float(total_boundaries / denom),
        mean_l1=float(total_l1 / denom),
        mean_kl=float(total_kl / denom),
        accuracy=float(correct) / denom,
        n_docs=int(n_docs),
    )


def run_markov_boundary_experiment(config: MarkovBoundaryConfig) -> MarkovBoundarySummary:
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
        n_classes=int(config.n_classes),
        vocab_size=int(config.vocab_size),
        log_std=float(config.transition_log_std),
        sinkhorn_iters=int(config.sinkhorn_iters),
        rng=rng,
    )
    log_transitions = np.log(transitions)

    docs = generate_docs(config, transitions=transitions)
    train_docs = docs[: int(config.train_docs)]
    test_docs = docs[int(config.train_docs) :]

    x_train, y_train = _collect_boundary_training_data(
        train_docs, config=config, log_transitions=log_transitions, rng=rng
    )
    boundary_model = BoundaryHarmPredictor(
        vocab_size=int(config.vocab_size),
        window_size=int(config.window_size),
        emb_dim=int(config.boundary_emb_dim),
        hidden_dim=int(config.boundary_hidden_dim),
    )
    train_loss = _train_boundary_model(
        boundary_model, x_train=x_train, y_train=y_train, config=config, device=device
    )

    metrics: Dict[str, PolicyMetrics] = {}
    for policy in ("fixed", "oracle", "learned", "worst"):
        metrics[policy] = _evaluate_policy(
            test_docs,
            config=config,
            log_transitions=log_transitions,
            boundary_model=boundary_model,
            policy=policy,
            device=device,
        )

    cfg_dict = asdict(config)
    cfg_dict["device_used"] = str(device)
    if device.type == "cuda":
        cfg_dict["cuda_current_device"] = int(torch.cuda.current_device())
        cfg_dict["cuda_device_name"] = str(torch.cuda.get_device_name(torch.cuda.current_device()))

    return MarkovBoundarySummary(
        config=cfg_dict,
        boundary_model_train_loss_final=float(train_loss),
        metrics=metrics,
    )


__all__ = [
    "MarkovBoundaryConfig",
    "MarkovBoundarySummary",
    "MarkovDoc",
    "run_markov_boundary_experiment",
]
