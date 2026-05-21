"""
Cut-budgeted Markov changepoint simulation (theory-first).

This simulation reframes "adaptive chunking" as a *constrained segmentation*
problem under a compute budget:

- A document of `n_tokens` admits cut positions t = 0..n_tokens-2 (cut after token t).
- A segmentation corresponds to a set of cuts C plus segment-length constraints
  (`min_leaf_tokens <= segment_len <= max_leaf_tokens`).
- Compute budget is expressed as a *maximum number of cuts* `K_max`.
  By default, we set `K_max` to the number of cuts produced by the fixed
  chunking baseline for that document (same max_chars, same chunk strategy).

Objective (Lean-friendly)
-------------------------
We optimize a boundary-set Hamming objective (fp + fn) over exact cut locations:

  loss(C) = |C △ B| = fp + fn

where B is the true boundary set. Using |C △ B| = |C| + |B| - 2|C ∩ B| and
|B| being constant for a document, minimizing loss is equivalent to maximizing

  reward(C) = Σ_{t in C} (2·1{t in B} - 1),

which is *additive over cuts*. Therefore, the oracle optimum under (K_max,
min/max leaf length) is solvable by a standard DP segmenter.

For a learned boundary model with calibrated probabilities p̂_t ≈ P(t in B | x),
we use the plug-in expected reward per cut:

  r̂_t = E[2·1{t in B} - 1] = 2 p̂_t - 1.

This creates a clean decomposition:
- approximation error: the gap between fixed chunking and the DP optimum;
- estimation error: the gap between learned DP and oracle DP.
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
        "PyTorch is required for Markov changepoint simulations. Install with: pip install torch>=2.0.0"
    ) from e

from treepo._research.preprocessing.chunker import AdaptiveChunkingConfig, HonestChunkingPolicy, chunk_for_ops
from treepo._research.tree.markov_boundary_honesty_simulation import _make_transition_matrices, _set_global_seed, _window_features
from treepo._research.tree.markov_changepoint_honesty_simulation import (
    BoundaryChangepointPredictor,
    ChangepointMarkovDoc,
    MarkovChangepointConfig,
    generate_changepoint_docs,
)


@dataclass(frozen=True)
class MarkovChangepointCutBudgetConfig(MarkovChangepointConfig):
    max_cuts: Optional[int] = None
    guidance_multipliers: Tuple[float, ...] = ()
    guidance_per_leaf: Tuple[float, ...] = ()
    guidance_strategies: Tuple[str, ...] = ()
    guidance_interface: str = "position"
    guidance_rounds: int = 3
    include_greedy_chunker: bool = False


@dataclass(frozen=True)
class _BoundaryMatchStats:
    tp: int
    fp: int
    fn: int


@dataclass(frozen=True)
class ChangepointCutBudgetPolicyMetrics:
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    predicted_to_true_ratio: float

    total_predicted_boundaries: int
    total_true_boundaries: int
    total_matched_boundaries: int
    n_docs: int

    mean_predicted_boundary_count: float
    mean_true_boundary_count: float
    mean_oracle_queries_used: float
    total_oracle_queries_used: int
    mean_hamming_loss: float
    mean_hamming_gap_to_oracle: float
    mean_theory_gap_upper_bound: float


@dataclass(frozen=True)
class MarkovChangepointCutBudgetSummary:
    config: Dict[str, object]
    boundary_model_train_loss_final: float
    boundary_true_positive_rate: float
    mean_fixed_cut_budget: float
    metrics: Dict[str, ChangepointCutBudgetPolicyMetrics]

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "boundary_model_train_loss_final": self.boundary_model_train_loss_final,
            "boundary_true_positive_rate": float(self.boundary_true_positive_rate),
            "mean_fixed_cut_budget": float(self.mean_fixed_cut_budget),
            "metrics": {k: asdict(v) for k, v in self.metrics.items()},
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def _validate_config(config: MarkovChangepointCutBudgetConfig) -> None:
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
    if int(config.min_leaf_tokens) < 1 or int(config.max_leaf_tokens) < int(config.min_leaf_tokens):
        raise ValueError("require 1 <= min_leaf_tokens <= max_leaf_tokens")
    if int(config.fixed_leaf_tokens) < 1:
        raise ValueError("fixed_leaf_tokens must be >= 1")
    if int(config.fixed_leaf_tokens) < int(config.min_leaf_tokens) or int(config.fixed_leaf_tokens) > int(
        config.max_leaf_tokens
    ):
        raise ValueError(
            "require min_leaf_tokens <= fixed_leaf_tokens <= max_leaf_tokens (so fixed chunking is feasible under the same constraints)"
        )
    if int(config.window_size) < 1:
        raise ValueError("window_size must be >= 1")
    if int(config.token_char_width) < 2:
        raise ValueError("token_char_width must be >= 2")
    if config.max_cuts is not None and int(config.max_cuts) < 0:
        raise ValueError("max_cuts must be >= 0 when provided")
    if int(config.guidance_rounds) < 1:
        raise ValueError("guidance_rounds must be >= 1")
    for mult in tuple(config.guidance_multipliers):
        if not math.isfinite(float(mult)) or float(mult) < 0.0:
            raise ValueError("guidance_multipliers must be finite and >= 0")
    for per_leaf in tuple(config.guidance_per_leaf):
        if not math.isfinite(float(per_leaf)) or float(per_leaf) < 0.0:
            raise ValueError("guidance_per_leaf must be finite and >= 0")
    interface = str(getattr(config, "guidance_interface", "position"))
    allowed_ifaces = {"position", "tree"}
    if interface not in allowed_ifaces:
        raise ValueError(f"unknown guidance_interface: {interface} (allowed: {sorted(allowed_ifaces)})")
    allowed = {"random", "uncertainty", "active"}
    for strat in tuple(config.guidance_strategies):
        if str(strat) not in allowed:
            raise ValueError(f"unknown guidance strategy: {strat} (allowed: {sorted(allowed)})")


def _encode_tokens_fixed_width(tokens: Sequence[int], *, width: int) -> str:
    width = int(width)
    if width < 2:
        raise ValueError("token_char_width must be >= 2")
    fmt_width = width - 1
    return "".join(f"{int(tok):0{fmt_width}d} " for tok in tokens)


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


def _collect_boundary_training_data(
    docs: Sequence[ChangepointMarkovDoc],
    *,
    config: MarkovChangepointCutBudgetConfig,
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
        labels = np.zeros((len(toks) - 1,), dtype=np.float32)
        if len(doc.true_boundaries) > 0:
            idx = np.asarray(doc.true_boundaries, dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < len(toks) - 1)]
            labels[idx] = 1.0
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


def _resolve_positive_weight(y_train: np.ndarray, *, config: MarkovChangepointCutBudgetConfig) -> float:
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
    config: MarkovChangepointCutBudgetConfig,
    device: torch.device,
) -> float:
    model.to(device)
    model.train()

    if len(x_train) == 0:
        return float("nan")

    x_t = torch.tensor(x_train, dtype=torch.long, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    pos_weight = _resolve_positive_weight(y_train, config=config)
    pos_weight_t = torch.tensor(float(pos_weight), dtype=torch.float32, device=device)

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
    config: MarkovChangepointCutBudgetConfig,
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

    out = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float64)
    return np.clip(out, 0.0, 1.0)


def _logit(p: float) -> float:
    p = float(p)
    p = max(1e-8, min(1.0 - 1e-8, p))
    return float(math.log(p / (1.0 - p)))


def _dp_segment_ends_maximize_cut_reward(
    cut_rewards: np.ndarray,
    *,
    min_leaf: int,
    max_leaf: int,
    max_cuts: int,
) -> List[int]:
    """
    DP segmenter that returns token end indices, maximizing sum(cut_rewards[t]) over cuts.

    Args:
        cut_rewards: length (n_tokens-1). cut_rewards[t] earned if we cut after token t.
        min_leaf/max_leaf: segment-length constraints in tokens.
        max_cuts: maximum number of cuts allowed (i.e., at most max_cuts+1 segments).
    """
    t_minus_1 = int(cut_rewards.shape[0])
    n_tokens = t_minus_1 + 1
    if n_tokens <= 0:
        return []
    if n_tokens < int(min_leaf):
        return [int(n_tokens - 1)]

    min_leaf = int(min_leaf)
    max_leaf = int(max_leaf)
    max_cuts = int(max(0, max_cuts))

    min_segments = int(math.ceil(float(n_tokens) / float(max_leaf)))
    max_segments = int(max_cuts + 1)
    if max_segments < min_segments:
        raise ValueError(
            f"Infeasible cut budget: need at least {min_segments - 1} cuts to satisfy max_leaf, "
            f"but max_cuts={max_cuts}."
        )

    neg_inf = float("-inf")
    dp = np.full((n_tokens + 1, max_segments + 1), neg_inf, dtype=np.float64)
    prev_start = np.full((n_tokens + 1, max_segments + 1), -1, dtype=np.int64)

    dp[0, 0] = 0.0

    for end_excl in range(1, n_tokens + 1):
        add = 0.0 if end_excl == n_tokens else float(cut_rewards[int(end_excl - 1)])
        lo = max(0, int(end_excl) - int(max_leaf))
        hi = int(end_excl) - int(min_leaf)
        if hi < lo:
            continue
        for segs in range(1, max_segments + 1):
            best = neg_inf
            best_start = -1
            for start in range(lo, hi + 1):
                base = float(dp[int(start), int(segs - 1)])
                if not math.isfinite(base):
                    continue
                score = base + add
                if score > best:
                    best = score
                    best_start = int(start)
            if best_start >= 0:
                dp[int(end_excl), int(segs)] = best
                prev_start[int(end_excl), int(segs)] = best_start

    end = int(n_tokens)
    best_segs = -1
    best_val = neg_inf
    for segs in range(1, max_segments + 1):
        val = float(dp[end, int(segs)])
        if not math.isfinite(val):
            continue
        if val > best_val + 1e-12 or (abs(val - best_val) <= 1e-12 and (best_segs < 0 or segs < best_segs)):
            best_val = val
            best_segs = int(segs)

    if best_segs < 0 or prev_start[end, best_segs] < 0:
        return [int(n_tokens - 1)]

    ends: List[int] = []
    segs = int(best_segs)
    while end > 0 and segs > 0:
        start = int(prev_start[end, segs])
        if start < 0:
            break
        ends.append(int(end - 1))
        end = start
        segs -= 1

    ends.reverse()
    if not ends:
        ends = [int(n_tokens - 1)]
    ends[-1] = int(n_tokens - 1)
    return ends


def _true_cut_rewards(doc: ChangepointMarkovDoc) -> np.ndarray:
    n_tokens = int(len(doc.tokens))
    rewards = -np.ones((max(0, n_tokens - 1),), dtype=np.float64)
    for t in doc.true_boundaries:
        ti = int(t)
        if 0 <= ti < n_tokens - 1:
            rewards[ti] = 1.0
    return rewards


def _expected_cut_rewards(boundary_probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(boundary_probabilities, dtype=np.float64)
    if probs.ndim != 1:
        probs = probs.reshape(-1)
    return np.clip(2.0 * probs - 1.0, -1.0, 1.0)


def _format_guidance_multiplier(mult: float) -> str:
    pct = int(round(100.0 * float(mult)))
    return f"q{pct}"


def _format_guidance_per_leaf(per_leaf: float) -> str:
    units = int(round(100.0 * float(per_leaf)))
    return f"l{units}"


def _guidance_budget(
    *,
    unit: str,
    value: float,
    max_cuts: int,
    n_positions: int,
) -> int:
    unit = str(unit)
    max_cuts = int(max(0, max_cuts))
    n_positions = int(max(0, n_positions))
    if unit == "cut_budget_multiplier":
        budget = int(round(float(value) * float(max_cuts)))
    elif unit == "per_leaf":
        budget = int(round(float(value) * float(max_cuts + 1)))
    else:
        raise ValueError(f"unknown guidance budget unit: {unit}")
    return int(min(max(0, budget), n_positions))


def _oracle_label_for_pos(true_set: set[int], t: int) -> float:
    return 1.0 if int(t) in true_set else 0.0


def _override_probs_with_oracle(
    base_probs: np.ndarray,
    *,
    true_set: set[int],
    query_positions: Sequence[int],
) -> np.ndarray:
    probs = np.asarray(base_probs, dtype=np.float64).copy()
    for t in query_positions:
        ti = int(t)
        if 0 <= ti < int(probs.shape[0]):
            probs[ti] = _oracle_label_for_pos(true_set, ti)
    return np.clip(probs, 0.0, 1.0)


def _query_order_random(n_positions: int, *, rng: np.random.Generator) -> List[int]:
    n_positions = int(max(0, n_positions))
    if n_positions <= 0:
        return []
    order = rng.permutation(np.arange(n_positions, dtype=np.int64)).tolist()
    return [int(x) for x in order]


def _query_order_uncertainty(probs: np.ndarray) -> List[int]:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    n = int(p.shape[0])
    if n <= 0:
        return []
    scores = np.abs(p - 0.5)
    order = np.argsort(scores, kind="mergesort").tolist()
    return [int(i) for i in order]


def _tree_guidance_partition(
    *,
    leaf_ends: Sequence[int],
    n_positions: int,
) -> Tuple[List[np.ndarray], List[int], np.ndarray, np.ndarray]:
    """Build disjoint oracle-query units from a fixed leaf partition.

    Units:
      - leaf units: all cut positions strictly inside each leaf
      - split units: the cut position between adjacent leaves (one per internal node)

    Returns:
      leaf_internal_positions: list of int arrays (positions) per leaf unit
      split_positions: list of int positions between leaves (len = n_leaves-1)
      pos_unit_kind: int8 array of length n_positions (0=leaf, 1=split)
      pos_unit_idx: int32 array of length n_positions (leaf index or split index)
    """
    ends = [int(x) for x in leaf_ends]
    n_positions = int(max(0, n_positions))
    if not ends:
        return [], [], np.zeros((n_positions,), dtype=np.int8), np.zeros((n_positions,), dtype=np.int32)

    leaf_internal_positions: List[np.ndarray] = []
    split_positions: List[int] = []

    pos_unit_kind = np.full((n_positions,), -1, dtype=np.int8)
    pos_unit_idx = np.full((n_positions,), -1, dtype=np.int32)

    start_tok = 0
    for leaf_i, end_tok in enumerate(ends):
        end_tok = int(end_tok)
        # Internal cut positions are t where start_tok <= t < end_tok.
        end_pos = int(min(max(0, end_tok), n_positions))
        start_pos = int(min(max(0, start_tok), n_positions))
        if end_pos > start_pos:
            internal = np.arange(start_pos, end_pos, dtype=np.int64)
        else:
            internal = np.zeros((0,), dtype=np.int64)
        leaf_internal_positions.append(internal)
        if internal.size > 0:
            pos_unit_kind[internal] = 0
            pos_unit_idx[internal] = int(leaf_i)

        # Split position between this leaf and the next is t=end_tok (if valid).
        if leaf_i < len(ends) - 1 and 0 <= end_tok < n_positions:
            split_idx = int(len(split_positions))
            split_positions.append(int(end_tok))
            pos_unit_kind[int(end_tok)] = 1
            pos_unit_idx[int(end_tok)] = int(split_idx)

        start_tok = int(end_tok) + 1

    covered = int(sum(int(x.size) for x in leaf_internal_positions) + len(split_positions))
    if covered != n_positions or np.any(pos_unit_kind < 0):
        raise ValueError(
            f"invalid fixed leaf partition: expected to cover n_positions={n_positions} cut positions, got covered={covered}"
        )

    return leaf_internal_positions, split_positions, pos_unit_kind, pos_unit_idx


def _tree_unit_uncertainty(
    unit: Tuple[str, int],
    *,
    var: np.ndarray,
    leaf_internal_positions: Sequence[np.ndarray],
    split_positions: Sequence[int],
) -> float:
    kind, idx = unit
    if kind == "leaf":
        internal = leaf_internal_positions[int(idx)]
        return float(np.sum(var[internal])) if internal.size > 0 else 0.0
    if kind == "split":
        pos = int(split_positions[int(idx)])
        return float(var[pos]) if 0 <= pos < int(var.shape[0]) else 0.0
    raise ValueError(f"unknown tree guidance unit: {unit}")


def _query_order_uncertainty_tree_units(
    probs: np.ndarray,
    *,
    leaf_internal_positions: Sequence[np.ndarray],
    split_positions: Sequence[int],
) -> List[Tuple[str, int]]:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    var = p * (1.0 - p)
    units: List[Tuple[str, int]] = [("leaf", int(i)) for i in range(len(leaf_internal_positions))] + [
        ("split", int(i)) for i in range(len(split_positions))
    ]
    units.sort(
        key=lambda u: _tree_unit_uncertainty(
            u,
            var=var,
            leaf_internal_positions=leaf_internal_positions,
            split_positions=split_positions,
        ),
        reverse=True,
    )
    return units


def _query_order_random_tree_units(
    *,
    n_leaves: int,
    rng: np.random.Generator,
) -> List[Tuple[str, int]]:
    n_leaves = int(max(0, n_leaves))
    units: List[Tuple[str, int]] = [("leaf", int(i)) for i in range(n_leaves)] + [
        ("split", int(i)) for i in range(max(0, n_leaves - 1))
    ]
    if not units:
        return []
    order = rng.permutation(np.arange(len(units), dtype=np.int64)).tolist()
    return [units[int(i)] for i in order]


def _guided_query_order_active_tree_units(
    base_probs: np.ndarray,
    *,
    true_set: set[int],
    budget: int,
    rounds: int,
    max_cuts: int,
    min_leaf: int,
    max_leaf: int,
    leaf_internal_positions: Sequence[np.ndarray],
    split_positions: Sequence[int],
    pos_unit_kind: np.ndarray,
    pos_unit_idx: np.ndarray,
) -> List[Tuple[str, int]]:
    p = np.asarray(base_probs, dtype=np.float64).reshape(-1).copy()
    n_positions = int(p.shape[0])

    n_leaves = int(len(leaf_internal_positions))
    n_units = int(n_leaves + len(split_positions))
    budget = int(min(max(0, budget), n_units))
    rounds = int(max(1, rounds))

    remaining = int(budget)
    queried: set[Tuple[str, int]] = set()
    query_order: List[Tuple[str, int]] = []

    def _unit_for_pos(t: int) -> Tuple[str, int]:
        kind = int(pos_unit_kind[int(t)])
        idx = int(pos_unit_idx[int(t)])
        return ("leaf", int(idx)) if kind == 0 else ("split", int(idx))

    for _ in range(rounds):
        if remaining <= 0:
            break
        rewards = _expected_cut_rewards(p)
        ends = _dp_segment_ends_maximize_cut_reward(
            rewards,
            min_leaf=int(min_leaf),
            max_leaf=int(max_leaf),
            max_cuts=int(max_cuts),
        )
        cuts = [int(t) for t in ends[:-1] if 0 <= int(t) < n_positions]
        candidates = [_unit_for_pos(t) for t in cuts]
        candidates = [c for c in candidates if c not in queried]
        if not candidates:
            break

        var = p * (1.0 - p)
        uniq_candidates = list(dict.fromkeys(candidates))
        uniq_candidates.sort(
            key=lambda u: _tree_unit_uncertainty(
                u,
                var=var,
                leaf_internal_positions=leaf_internal_positions,
                split_positions=split_positions,
            ),
            reverse=True,
        )
        take = uniq_candidates[: int(min(remaining, len(uniq_candidates)))]
        for unit in take:
            queried.add(unit)
            query_order.append(unit)
            kind, idx = unit
            if kind == "leaf":
                positions = leaf_internal_positions[int(idx)].tolist()
            else:
                positions = [int(split_positions[int(idx)])]
            for pos in positions:
                if 0 <= int(pos) < n_positions:
                    p[int(pos)] = _oracle_label_for_pos(true_set, int(pos))
        remaining -= int(len(take))

    if remaining > 0:
        var = p * (1.0 - p)
        all_units: List[Tuple[str, int]] = [("leaf", int(i)) for i in range(n_leaves)] + [
            ("split", int(i)) for i in range(len(split_positions))
        ]
        rest = [u for u in all_units if u not in queried]
        rest.sort(
            key=lambda u: _tree_unit_uncertainty(
                u,
                var=var,
                leaf_internal_positions=leaf_internal_positions,
                split_positions=split_positions,
            ),
            reverse=True,
        )
        query_order.extend(rest[: int(remaining)])

    return query_order


def _guided_query_order_active(
    base_probs: np.ndarray,
    *,
    true_set: set[int],
    budget: int,
    rounds: int,
    max_cuts: int,
    min_leaf: int,
    max_leaf: int,
    rng: np.random.Generator,
) -> List[int]:
    p = np.asarray(base_probs, dtype=np.float64).reshape(-1).copy()
    n_positions = int(p.shape[0])
    budget = int(min(max(0, budget), n_positions))
    rounds = int(max(1, rounds))
    remaining = int(budget)
    queried: set[int] = set()
    query_order: List[int] = []

    for _ in range(rounds):
        if remaining <= 0:
            break
        rewards = _expected_cut_rewards(p)
        ends = _dp_segment_ends_maximize_cut_reward(
            rewards,
            min_leaf=int(min_leaf),
            max_leaf=int(max_leaf),
            max_cuts=int(max_cuts),
        )
        cuts = [int(t) for t in ends[:-1] if 0 <= int(t) < n_positions]
        candidates = [c for c in cuts if c not in queried]
        if not candidates:
            break
        # Prefer querying the most uncertain among the currently selected cuts.
        cand_probs = p[np.asarray(candidates, dtype=np.int64)]
        idx_order = np.argsort(np.abs(cand_probs - 0.5), kind="mergesort")
        take = [int(candidates[i]) for i in idx_order[: int(min(remaining, len(candidates)))]]
        for t in take:
            queried.add(int(t))
            query_order.append(int(t))
            p[int(t)] = _oracle_label_for_pos(true_set, int(t))
        remaining -= int(len(take))

    # Use any leftover budget on globally-uncertain positions.
    if remaining > 0:
        full = _query_order_uncertainty(p)
        extra: List[int] = []
        for t in full:
            if int(t) in queried:
                continue
            extra.append(int(t))
            if len(extra) >= int(remaining):
                break
        for t in extra:
            queried.add(int(t))
            query_order.append(int(t))
            p[int(t)] = _oracle_label_for_pos(true_set, int(t))

    return query_order


def _theory_gap_bound_from_cut_reward_errors(
    *,
    oracle_rewards: np.ndarray,
    estimated_rewards: np.ndarray,
    oracle_cuts: Sequence[int],
    predicted_cuts: Sequence[int],
) -> float:
    oracle = np.asarray(oracle_rewards, dtype=np.float64).reshape(-1)
    est = np.asarray(estimated_rewards, dtype=np.float64).reshape(-1)
    if oracle.shape != est.shape:
        raise ValueError("oracle_rewards and estimated_rewards must have the same shape")
    delta = est - oracle
    n = int(delta.shape[0])

    def _sum_abs(cuts: Sequence[int]) -> float:
        s = 0.0
        for t in cuts:
            ti = int(t)
            if 0 <= ti < n:
                s += float(abs(delta[ti]))
        return float(s)

    return float(_sum_abs(oracle_cuts) + _sum_abs(predicted_cuts))


def run_markov_changepoint_cut_budget_experiment(
    config: MarkovChangepointCutBudgetConfig,
    *,
    strategy: str = "axis",
    honest_policy: Optional[HonestChunkingPolicy] = None,
    adaptive_config: Optional[AdaptiveChunkingConfig] = None,
) -> MarkovChangepointCutBudgetSummary:
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

    # Prior correction: adjust logits by Δ = logit(pi_true) - logit(pi_train).
    # This reduces the common failure mode where training uses a rebalanced dataset
    # (pi_train≈0.5) but deployment prior pi_true is sparse.
    total_true = int(sum(len(d.true_boundaries) for d in train_docs))
    total_possible = int(sum(max(0, len(d.tokens) - 1) for d in train_docs))
    pi_true = float(total_true) / float(max(1, total_possible))
    pi_train = float(np.mean(np.asarray(y_train, dtype=np.float64))) if len(y_train) > 0 else float(pi_true)
    pos_weight = _resolve_positive_weight(y_train, config=config)
    prior_offset = (_logit(pi_true) - _logit(pi_train)) if bool(config.calibrate_prior) else 0.0
    # PyTorch's BCE pos_weight shifts the optimum logits by +log(pos_weight).
    # Undo that shift so `sigmoid(logits + logit_offset)` is interpretable as p̂(y=1|x).
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

    # Optional: keep the existing adaptive chunker policy around for comparison.
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

    guidance_multipliers = tuple(float(x) for x in tuple(config.guidance_multipliers) if float(x) > 0.0)
    guidance_per_leaf = tuple(float(x) for x in tuple(config.guidance_per_leaf) if float(x) > 0.0)
    guidance_strategies = tuple(str(x) for x in tuple(config.guidance_strategies))
    if (guidance_multipliers or guidance_per_leaf) and not guidance_strategies:
        guidance_strategies = ("active",)
    guidance_specs: List[Tuple[str, str, str, float]] = []
    for strat in guidance_strategies:
        for mult in guidance_multipliers:
            policy_name = f"dp_guided_{strat}_{_format_guidance_multiplier(mult)}"
            guidance_specs.append((policy_name, str(strat), "cut_budget_multiplier", float(mult)))
        for per_leaf in guidance_per_leaf:
            policy_name = f"dp_guided_{strat}_{_format_guidance_per_leaf(per_leaf)}"
            guidance_specs.append((policy_name, str(strat), "per_leaf", float(per_leaf)))
    # Preserve insertion order while deduplicating.
    seen = set()
    guidance_specs = [s for s in guidance_specs if not (s[0] in seen or seen.add(s[0]))]

    guidance_by_strategy: Dict[str, List[Tuple[str, str, float]]] = {}
    for policy_name, strat, unit, value in guidance_specs:
        key = str(strat)
        guidance_by_strategy.setdefault(key, []).append((str(policy_name), str(unit), float(value)))

    policies: Tuple[str, ...] = ("fixed", "dp_honest") + tuple(s[0] for s in guidance_specs) + ("oracle_opt",)
    include_greedy = bool(config.include_greedy_chunker)
    if include_greedy:
        policies = policies + ("chunker_honest",)

    per_policy_true: Dict[str, List[int]] = {p: [] for p in policies}
    per_policy_pred: Dict[str, List[int]] = {p: [] for p in policies}
    per_policy_match: Dict[str, List[_BoundaryMatchStats]] = {p: [] for p in policies}
    per_policy_hamming: Dict[str, List[int]] = {p: [] for p in policies}
    per_policy_gap: Dict[str, List[int]] = {p: [] for p in policies}
    per_policy_queries: Dict[str, List[int]] = {p: [] for p in policies}
    per_policy_theory_bound: Dict[str, List[float]] = {p: [] for p in policies}

    fixed_budgets: List[int] = []

    for doc_i, doc in enumerate(test_docs):
        toks = np.asarray(doc.tokens, dtype=np.int64)
        if len(toks) < 2:
            continue

        text = _encode_tokens_fixed_width(toks, width=token_char_width)
        true_boundaries = [int(t) for t in doc.true_boundaries if int(t) < len(toks) - 1]
        true_set = set(true_boundaries)

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
        k_fixed = int(len(pred_fixed))
        fixed_budgets.append(int(k_fixed))

        max_cuts = int(config.max_cuts) if config.max_cuts is not None else int(k_fixed)
        max_cuts = max(0, max_cuts)

        # DP policy on calibrated boundary probabilities.
        probs = _predict_boundary_probabilities(
            model,
            toks,
            config=config,
            device=device,
            logit_offset=logit_offset,
        )
        dp_rewards = _expected_cut_rewards(probs)
        ends_dp = _dp_segment_ends_maximize_cut_reward(
            dp_rewards,
            min_leaf=int(config.min_leaf_tokens),
            max_leaf=int(config.max_leaf_tokens),
            max_cuts=max_cuts,
        )
        pred_dp = [int(t) for t in ends_dp[:-1] if int(t) < len(toks) - 1]

        # Oracle optimum for the exact-Hamming objective under the same constraints.
        oracle_rewards = _true_cut_rewards(doc)
        ends_oracle = _dp_segment_ends_maximize_cut_reward(
            oracle_rewards,
            min_leaf=int(config.min_leaf_tokens),
            max_leaf=int(config.max_leaf_tokens),
            max_cuts=max_cuts,
        )
        pred_oracle = [int(t) for t in ends_oracle[:-1] if int(t) < len(toks) - 1]

        # Optional: current greedy adaptive chunker path (can oversplit without a cut budget).
        per_doc: Dict[str, List[int]] = {
            "fixed": pred_fixed,
            "dp_honest": pred_dp,
            "oracle_opt": pred_oracle,
        }
        per_doc_rewards: Dict[str, Optional[np.ndarray]] = {
            "fixed": None,
            "dp_honest": dp_rewards,
            "oracle_opt": oracle_rewards,
        }
        per_doc_queries: Dict[str, int] = {
            "fixed": 0,
            "dp_honest": 0,
            "oracle_opt": 0,
        }

        # Optional: guided DP policies using a limited oracle-query budget.
        if guidance_specs:
            n_positions = int(max(0, len(toks) - 1))
            interface = str(getattr(config, "guidance_interface", "position"))
            if interface == "position":
                # Position-level guidance: oracle queries label individual cut positions.
                # Precompute a nested query order per strategy so increasing guidance
                # levels are genuine supersets.
                strategy_orders: Dict[str, List[int]] = {}
                for strat_key, policy_mults in guidance_by_strategy.items():
                    strat_id = {"random": 0, "uncertainty": 1, "active": 2}.get(strat_key, 99)
                    seed_val = (int(config.seed) * 1000003 + int(doc_i) * 9176 + int(strat_id) * 100000) % (2**32)
                    local_rng = np.random.default_rng(int(seed_val))
                    max_budget = 0
                    for _policy_name, unit, value in policy_mults:
                        b = _guidance_budget(
                            unit=str(unit),
                            value=float(value),
                            max_cuts=int(max_cuts),
                            n_positions=int(n_positions),
                        )
                        max_budget = max(int(max_budget), int(b))

                    if max_budget <= 0:
                        strategy_orders[strat_key] = []
                        continue

                    if strat_key == "random":
                        strategy_orders[strat_key] = _query_order_random(n_positions, rng=local_rng)
                    elif strat_key == "uncertainty":
                        strategy_orders[strat_key] = _query_order_uncertainty(probs)
                    elif strat_key == "active":
                        strategy_orders[strat_key] = _guided_query_order_active(
                            probs,
                            true_set=true_set,
                            budget=int(max_budget),
                            rounds=int(config.guidance_rounds),
                            max_cuts=int(max_cuts),
                            min_leaf=int(config.min_leaf_tokens),
                            max_leaf=int(config.max_leaf_tokens),
                            rng=local_rng,
                        )
                    else:
                        raise ValueError(f"unknown guidance strategy: {strat_key}")

                for strat_key, policy_mults in guidance_by_strategy.items():
                    order = strategy_orders.get(strat_key, [])
                    for policy_name, unit, value in policy_mults:
                        budget = _guidance_budget(
                            unit=str(unit),
                            value=float(value),
                            max_cuts=int(max_cuts),
                            n_positions=int(n_positions),
                        )
                        if budget <= 0:
                            per_doc[policy_name] = pred_dp
                            per_doc_rewards[policy_name] = dp_rewards
                            per_doc_queries[policy_name] = 0
                            continue
                        query_positions = order[: int(budget)]
                        guided_probs = _override_probs_with_oracle(
                            probs,
                            true_set=true_set,
                            query_positions=query_positions,
                        )
                        guided_rewards = _expected_cut_rewards(guided_probs)
                        ends_guided = _dp_segment_ends_maximize_cut_reward(
                            guided_rewards,
                            min_leaf=int(config.min_leaf_tokens),
                            max_leaf=int(config.max_leaf_tokens),
                            max_cuts=max_cuts,
                        )
                        pred_guided = [int(t) for t in ends_guided[:-1] if int(t) < len(toks) - 1]
                        per_doc[policy_name] = pred_guided
                        per_doc_rewards[policy_name] = guided_rewards
                        per_doc_queries[policy_name] = int(len(query_positions))
            else:
                # Tree-level guidance: oracle queries label fixed leaves and internal-node split boundaries.
                leaf_internal_positions, split_positions, pos_unit_kind, pos_unit_idx = _tree_guidance_partition(
                    leaf_ends=ends_fixed,
                    n_positions=int(n_positions),
                )
                n_leaves = int(len(leaf_internal_positions))
                n_units = int(n_leaves + len(split_positions))
                guidance_base_cuts = int(k_fixed)

                # Precompute a nested query order per strategy so increasing guidance
                # levels are genuine supersets.
                strategy_orders_units: Dict[str, List[Tuple[str, int]]] = {}
                for strat_key, policy_mults in guidance_by_strategy.items():
                    strat_id = {"random": 0, "uncertainty": 1, "active": 2}.get(strat_key, 99)
                    seed_val = (int(config.seed) * 1000003 + int(doc_i) * 9176 + int(strat_id) * 100000) % (2**32)
                    local_rng = np.random.default_rng(int(seed_val))
                    max_budget = 0
                    for _policy_name, unit, value in policy_mults:
                        b = _guidance_budget(
                            unit=str(unit),
                            value=float(value),
                            max_cuts=int(guidance_base_cuts),
                            n_positions=int(n_units),
                        )
                        max_budget = max(int(max_budget), int(b))

                    if max_budget <= 0:
                        strategy_orders_units[strat_key] = []
                        continue

                    if strat_key == "random":
                        strategy_orders_units[strat_key] = _query_order_random_tree_units(
                            n_leaves=int(n_leaves),
                            rng=local_rng,
                        )
                    elif strat_key == "uncertainty":
                        strategy_orders_units[strat_key] = _query_order_uncertainty_tree_units(
                            probs,
                            leaf_internal_positions=leaf_internal_positions,
                            split_positions=split_positions,
                        )
                    elif strat_key == "active":
                        strategy_orders_units[strat_key] = _guided_query_order_active_tree_units(
                            probs,
                            true_set=true_set,
                            budget=int(max_budget),
                            rounds=int(config.guidance_rounds),
                            max_cuts=int(max_cuts),
                            min_leaf=int(config.min_leaf_tokens),
                            max_leaf=int(config.max_leaf_tokens),
                            leaf_internal_positions=leaf_internal_positions,
                            split_positions=split_positions,
                            pos_unit_kind=pos_unit_kind,
                            pos_unit_idx=pos_unit_idx,
                        )
                    else:
                        raise ValueError(f"unknown guidance strategy: {strat_key}")

                for strat_key, policy_mults in guidance_by_strategy.items():
                    order_units = strategy_orders_units.get(strat_key, [])
                    for policy_name, unit, value in policy_mults:
                        budget_units = _guidance_budget(
                            unit=str(unit),
                            value=float(value),
                            max_cuts=int(guidance_base_cuts),
                            n_positions=int(n_units),
                        )
                        if budget_units <= 0:
                            per_doc[policy_name] = pred_dp
                            per_doc_rewards[policy_name] = dp_rewards
                            per_doc_queries[policy_name] = 0
                            continue
                        queried_units = order_units[: int(budget_units)]
                        query_positions: List[int] = []
                        for kind, idx in queried_units:
                            if str(kind) == "leaf":
                                query_positions.extend(leaf_internal_positions[int(idx)].tolist())
                            else:
                                query_positions.append(int(split_positions[int(idx)]))

                        guided_probs = _override_probs_with_oracle(
                            probs,
                            true_set=true_set,
                            query_positions=query_positions,
                        )
                        guided_rewards = _expected_cut_rewards(guided_probs)
                        ends_guided = _dp_segment_ends_maximize_cut_reward(
                            guided_rewards,
                            min_leaf=int(config.min_leaf_tokens),
                            max_leaf=int(config.max_leaf_tokens),
                            max_cuts=max_cuts,
                        )
                        pred_guided = [int(t) for t in ends_guided[:-1] if int(t) < len(toks) - 1]
                        per_doc[policy_name] = pred_guided
                        per_doc_rewards[policy_name] = guided_rewards
                        per_doc_queries[policy_name] = int(len(queried_units))
        if include_greedy:
            pred_chunker: List[int] = []
            try:
                # Convert probabilities into feedback signals via the existing adaptive chunker path.
                # We use the same convention as other changepoint sims: high p(boundary) => compress.
                from treepo._research.tree.markov_changepoint_honesty_simulation import (
                    _signals_from_boundary_probabilities,
                )

                signals = _signals_from_boundary_probabilities(
                    probs,
                    token_char_width=token_char_width,
                    source="predicted_boundary_probability",
                )
                # Tag all signals into the boundary role if honesty is enabled.
                for s in signals:
                    s.metadata["honest_role"] = honest.boundary_role
                chunks_adaptive = chunk_for_ops(
                    text,
                    max_chars=int(base_max_chars),
                    strategy=str(strategy),
                    adaptive_config=adaptive_config,
                    feedback_signals=signals
                    if not honest.enabled
                    else [
                        s
                        for s in signals
                        if s.metadata.get("honest_role") == honest.boundary_role
                    ],
                )
                ends_chunker = _chunk_ends_from_chunks(
                    chunks=chunks_adaptive,
                    token_char_width=token_char_width,
                    n_tokens=len(toks),
                )
                pred_chunker = [int(t) for t in ends_chunker[:-1] if int(t) < len(toks) - 1]
            except Exception:
                pred_chunker = []
            per_doc["chunker_honest"] = pred_chunker
            per_doc_rewards["chunker_honest"] = None
            per_doc_queries["chunker_honest"] = 0

        # Exact-Hamming oracle loss for this doc (used to compute gaps).
        oracle_set = set(pred_oracle)
        oracle_loss = int(len(oracle_set.symmetric_difference(true_set)))

        for name, pred in per_doc.items():
            pred_set = set(int(t) for t in pred)
            per_policy_true[name].append(int(len(true_boundaries)))
            per_policy_pred[name].append(int(len(pred_set)))
            per_policy_queries[name].append(int(per_doc_queries.get(name, 0)))
            per_policy_match[name].append(
                _match_boundaries(
                    predicted_boundaries=sorted(pred_set),
                    true_boundaries=true_boundaries,
                    tolerance=int(config.boundary_tolerance_tokens),
                )
            )
            loss = int(len(pred_set.symmetric_difference(true_set)))
            per_policy_hamming[name].append(int(loss))
            per_policy_gap[name].append(int(loss - oracle_loss))
            rewards = per_doc_rewards.get(name)
            if rewards is None:
                per_policy_theory_bound[name].append(float("nan"))
            else:
                per_policy_theory_bound[name].append(
                    _theory_gap_bound_from_cut_reward_errors(
                        oracle_rewards=oracle_rewards,
                        estimated_rewards=rewards,
                        oracle_cuts=pred_oracle,
                        predicted_cuts=sorted(pred_set),
                    )
                )

    metrics: Dict[str, ChangepointCutBudgetPolicyMetrics] = {}
    for policy in policies:
        if not per_policy_true[policy]:
            continue

        y_true = np.asarray(per_policy_true[policy], dtype=np.float64)
        y_pred = np.asarray(per_policy_pred[policy], dtype=np.float64)
        matches = per_policy_match[policy]

        total_tp = int(sum(m.tp for m in matches))
        total_fp = int(sum(m.fp for m in matches))
        total_fn = int(sum(m.fn for m in matches))
        total_true = int(sum(int(x) for x in per_policy_true[policy]))
        total_pred = int(sum(int(x) for x in per_policy_pred[policy]))
        total_queries = int(sum(int(x) for x in per_policy_queries[policy]))
        n_docs = int(len(per_policy_true[policy]))

        precision = float(total_tp) / float(max(1, total_tp + total_fp))
        recall = float(total_tp) / float(max(1, total_tp + total_fn))
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )

        metrics[policy] = ChangepointCutBudgetPolicyMetrics(
            boundary_precision=float(precision),
            boundary_recall=float(recall),
            boundary_f1=float(f1),
            predicted_to_true_ratio=float(total_pred) / float(max(1, total_true)),
            total_predicted_boundaries=int(total_pred),
            total_true_boundaries=int(total_true),
            total_matched_boundaries=int(total_tp),
            n_docs=int(n_docs),
            mean_predicted_boundary_count=float(np.mean(y_pred)),
            mean_true_boundary_count=float(np.mean(y_true)),
            mean_oracle_queries_used=float(total_queries) / float(max(1, n_docs)),
            total_oracle_queries_used=int(total_queries),
            mean_hamming_loss=float(np.mean(np.asarray(per_policy_hamming[policy], dtype=np.float64))),
            mean_hamming_gap_to_oracle=float(np.mean(np.asarray(per_policy_gap[policy], dtype=np.float64))),
            mean_theory_gap_upper_bound=float(
                np.mean(np.asarray(per_policy_theory_bound[policy], dtype=np.float64))
            ),
        )

    cfg_dict = asdict(config)
    cfg_dict["device_used"] = str(device)
    cfg_dict["chunk_strategy"] = str(strategy)
    cfg_dict["honest_policy"] = asdict(honest)
    cfg_dict["adaptive_config"] = asdict(adaptive_config)
    cfg_dict["calibration_pi_true"] = float(pi_true)
    cfg_dict["calibration_pi_train"] = float(pi_train)
    cfg_dict["boundary_pos_weight_used"] = float(pos_weight)
    cfg_dict["calibration_prior_offset"] = float(prior_offset)
    cfg_dict["calibration_pos_weight_offset"] = float(pos_weight_offset)
    cfg_dict["calibration_logit_offset"] = float(logit_offset)

    return MarkovChangepointCutBudgetSummary(
        config=cfg_dict,
        boundary_model_train_loss_final=float(train_loss),
        boundary_true_positive_rate=float(pi_true),
        mean_fixed_cut_budget=float(np.mean(np.asarray(fixed_budgets, dtype=np.float64))) if fixed_budgets else float("nan"),
        metrics=metrics,
    )


__all__ = [
    "MarkovChangepointCutBudgetConfig",
    "ChangepointCutBudgetPolicyMetrics",
    "MarkovChangepointCutBudgetSummary",
    "run_markov_changepoint_cut_budget_experiment",
]
