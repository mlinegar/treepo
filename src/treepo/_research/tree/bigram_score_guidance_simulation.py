"""
Mergeable bigram-score guidance simulation (CPU-first).

This simulation targets the "oracle doesn't care about splits; splits are for efficiency" story:

- There is a split-invariant oracle score f*(x) for any token sequence x:
    f*(x) = sum_t w[token_t, token_{t+1}]
  where w is an unknown bigram-weight matrix.

- The score is mergeable (BigramSketch.lean):
    f*(u ++ v) = f*(u) + f*(v) + w[last(u), first(v)].

- Training data comes from *oracle queries on spans* (leaves/internal nodes of a fixed tree).
  Each query returns f*(span). Queries on longer spans are assumed more costly.

- From queried spans, we estimate w via ridge regression on bigram-count features, then
  evaluate prediction error on held-out documents.

Key qualitative behaviors this can show:
1) Querying only leaves cannot identify cross-leaf boundary weights (bias floor).
2) Adding internal-node queries resolves boundary weights and can drive error to 0.
3) For intermediate guidance, increasing train docs improves estimation.
4) If topic changepoints follow a global position profile, a globally learned guidance policy can
   prioritize the most informative internal queries.

Guidance interface (intended to be intuitive):
- Every training document always receives one oracle score per *leaf* (one per chunk).
- `guidance_per_leaf` controls *additional* internal-node oracle queries per leaf.

Oracle feature modes (for interpretability):
- `token_bigrams`: weights are on token-to-token bigrams (dimension `vocab_size^2`).
- `topic_bigrams`: tokens are generated from disjoint topic vocab blocks, so each token maps
  to a topic id; weights are on topic-to-topic transitions (dimension `n_topics^2`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Span = Tuple[int, int]  # [start, end) token indices


def _set_global_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _pair_index(a: int, b: int, vocab_size: int) -> int:
    return int(a) * int(vocab_size) + int(b)


def bigram_count_indices(tokens: Sequence[int], *, vocab_size: int) -> np.ndarray:
    toks = np.asarray(tokens, dtype=np.int64)
    if toks.size < 2:
        return np.zeros((0,), dtype=np.int64)
    a = toks[:-1]
    b = toks[1:]
    return (a * int(vocab_size) + b).astype(np.int64, copy=False)


def bigram_counts_dense(tokens: Sequence[int], *, vocab_size: int) -> np.ndarray:
    d = int(vocab_size) * int(vocab_size)
    idx = bigram_count_indices(tokens, vocab_size=vocab_size)
    if idx.size == 0:
        return np.zeros((d,), dtype=np.float64)
    counts = np.bincount(idx, minlength=d).astype(np.float64, copy=False)
    if counts.size != d:
        counts = np.pad(counts, (0, d - counts.size), mode="constant")
    return counts


def bigram_counts_sparse(tokens: Sequence[int], *, vocab_size: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = bigram_count_indices(tokens, vocab_size=vocab_size)
    if idx.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    uniq, counts = np.unique(idx, return_counts=True)
    return uniq.astype(np.int64, copy=False), counts.astype(np.float64, copy=False)


def oracle_bigram_score(tokens: Sequence[int], *, w_true: np.ndarray, vocab_size: int) -> float:
    idx, vals = bigram_counts_sparse(tokens, vocab_size=vocab_size)
    if idx.size == 0:
        return 0.0
    return float(np.dot(w_true[idx], vals))


def oracle_merge_score(
    left_tokens: Sequence[int],
    right_tokens: Sequence[int],
    *,
    w_true: np.ndarray,
    vocab_size: int,
) -> float:
    if len(left_tokens) == 0:
        return oracle_bigram_score(right_tokens, w_true=w_true, vocab_size=vocab_size)
    if len(right_tokens) == 0:
        return oracle_bigram_score(left_tokens, w_true=w_true, vocab_size=vocab_size)
    cross_idx = _pair_index(int(left_tokens[-1]), int(right_tokens[0]), vocab_size)
    return (
        oracle_bigram_score(left_tokens, w_true=w_true, vocab_size=vocab_size)
        + oracle_bigram_score(right_tokens, w_true=w_true, vocab_size=vocab_size)
        + float(w_true[cross_idx])
    )


@dataclass(frozen=True)
class NodeSpan:
    leaf_start: int
    leaf_end: int
    token_start: int
    token_end: int

    @property
    def n_leaves(self) -> int:
        return int(self.leaf_end - self.leaf_start)

    @property
    def token_len(self) -> int:
        return int(self.token_end - self.token_start)

    @property
    def is_leaf(self) -> bool:
        return self.n_leaves == 1


def build_leaf_spans(n_tokens: int, *, leaf_tokens: int) -> List[Span]:
    leaf = int(max(1, leaf_tokens))
    spans: List[Span] = []
    start = 0
    while start < int(n_tokens):
        end = min(int(n_tokens), start + leaf)
        spans.append((start, end))
        start = end
    return spans


def build_balanced_tree_nodes(n_tokens: int, *, leaf_tokens: int) -> List[NodeSpan]:
    leaves = build_leaf_spans(n_tokens, leaf_tokens=leaf_tokens)
    n_leaves = len(leaves)
    if n_leaves == 0:
        return []

    def rec(lo: int, hi: int) -> List[NodeSpan]:
        if hi - lo == 1:
            token_start, token_end = leaves[lo]
            return [NodeSpan(lo, hi, token_start, token_end)]
        mid = lo + (hi - lo) // 2
        left = rec(lo, mid)
        right = rec(mid, hi)
        token_start = leaves[lo][0]
        token_end = leaves[hi - 1][1]
        return left + right + [NodeSpan(lo, hi, token_start, token_end)]

    return rec(0, n_leaves)


def _splitmix64(x: int) -> int:
    z = (int(x) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z & 0xFFFFFFFFFFFFFFFF


def _normalize_oracle_feature_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode in {"token", "token_bigrams", "bigrams"}:
        return "token_bigrams"
    if mode in {"topic", "topic_bigrams"}:
        return "topic_bigrams"
    raise ValueError(f"Unknown oracle_feature_mode: {mode!r}")


def _token_to_topic_block(vocab_size: int, n_topics: int) -> np.ndarray:
    v = int(vocab_size)
    k = int(n_topics)
    if v <= 0:
        raise ValueError("vocab_size must be positive")
    if k <= 0:
        raise ValueError("n_topics must be positive")
    block_size = max(1, v // k)
    return np.minimum(np.arange(v, dtype=np.int64) // int(block_size), k - 1)


def make_boundary_profile(
    profile: str,
    *,
    strength: float,
    rng: np.random.Generator,
) -> Callable[[float], float]:
    """Return a positive-valued boundary weight function w(t) over normalized position t in (0,1).

    This is used to bias *segment boundary locations* in `generate_leaf_aligned_topic_docs`, enabling
    globally learnable structure such as "changepoints tend to happen near the middle".
    """
    prof = str(profile).strip().lower()
    p = float(strength)
    if not np.isfinite(p) or p < 0:
        raise ValueError("boundary_profile_strength must be finite and >= 0")
    if p == 0.0:
        return lambda t: 1.0

    if prof in {"uniform", "flat"}:
        return lambda t: 1.0

    if prof in {"start", "begin"}:
        return lambda t: float(max(0.0, 1.0 - float(t))) ** max(1e-8, p)

    if prof in {"end", "finish"}:
        return lambda t: float(max(0.0, float(t))) ** max(1e-8, p)

    if prof in {"middle", "center", "centre"}:
        return lambda t: float(max(0.0, float(t) * (1.0 - float(t)))) ** max(1e-8, p)

    if prof in {"bimodal", "ends"}:
        return lambda t: (float(max(0.0, float(t))) ** max(1e-8, p)) + (
            float(max(0.0, 1.0 - float(t))) ** max(1e-8, p)
        )

    if prof in {"random", "rand"}:
        # A smooth random profile via a small sinusoidal basis; p controls amplitude.
        m = 3
        amps = rng.normal(loc=0.0, scale=max(1e-8, p), size=(m,)).astype(np.float64, copy=False)
        phases = rng.uniform(0.0, 2.0 * math.pi, size=(m,)).astype(np.float64, copy=False)
        freqs = np.arange(1, m + 1, dtype=np.float64)

        def _w(t: float) -> float:
            tt = float(min(1.0, max(0.0, t)))
            x = 0.0
            for a, ph, f in zip(amps.tolist(), phases.tolist(), freqs.tolist()):
                x += float(a) * math.sin(2.0 * math.pi * float(f) * tt + float(ph))
            return float(math.exp(x))

        return _w

    raise ValueError(f"Unknown boundary_profile: {profile!r}")


def generate_topic_bigram_docs(
    n_docs: int,
    *,
    vocab_size: int,
    n_topics: int,
    min_tokens: int,
    max_tokens: int,
    min_segments: int,
    max_segments: int,
    min_seg_len: int,
    max_seg_len: int,
    topic_concentration: float,
    seed: int,
) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[np.ndarray, ...]]:
    rng = _set_global_seed(seed)
    v = int(vocab_size)
    k = int(n_topics)
    if v < 2:
        raise ValueError("vocab_size must be >= 2")
    if k < 1:
        raise ValueError("n_topics must be >= 1")
    if int(min_tokens) < 2 or int(max_tokens) < int(min_tokens):
        raise ValueError("require 2 <= min_tokens <= max_tokens")

    alpha = float(topic_concentration)
    if alpha <= 0:
        raise ValueError("topic_concentration must be > 0")

    topics: List[np.ndarray] = []
    for _ in range(k):
        probs = rng.dirichlet(np.full((v,), alpha, dtype=np.float64))
        topics.append(probs.astype(np.float64, copy=False))

    docs: List[Tuple[int, ...]] = []
    for _ in range(int(n_docs)):
        n = int(rng.integers(int(min_tokens), int(max_tokens) + 1))
        s = int(rng.integers(int(min_segments), int(max_segments) + 1))
        s = max(1, s)
        remaining = n
        seg_lens: List[int] = []
        for seg_idx in range(s):
            if seg_idx == s - 1:
                seg_lens.append(remaining)
                break
            min_needed = int(min_seg_len) * (s - seg_idx - 1)
            hi = min(int(max_seg_len), remaining - min_needed)
            lo = int(min_seg_len)
            hi = max(lo, hi)
            seg_len = int(rng.integers(lo, hi + 1))
            seg_lens.append(seg_len)
            remaining -= seg_len
        # If constraints forced an invalid split, fall back to a single segment.
        if sum(seg_lens) != n or any(x <= 0 for x in seg_lens):
            seg_lens = [n]

        tokens: List[int] = []
        for seg_len in seg_lens:
            topic_id = int(rng.integers(0, k))
            probs = topics[topic_id]
            draws = rng.choice(v, size=int(seg_len), replace=True, p=probs)
            tokens.extend(int(x) for x in draws.tolist())
        docs.append(tuple(tokens))

    return (tuple(docs), tuple(topics))


def generate_leaf_aligned_topic_docs(
    n_docs: int,
    *,
    vocab_size: int,
    n_topics: int,
    min_tokens: int,
    max_tokens: int,
    min_segments: int,
    max_segments: int,
    min_seg_len: int,
    max_seg_len: int,
    leaf_tokens: int,
    topic_concentration: float,
    disjoint_topic_vocab: bool,
    boundary_profile: str,
    boundary_profile_strength: float,
    boundary_profile_seed: int,
    seed: int,
) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[np.ndarray, ...]]:
    """Generate documents where topic changes occur only at *leaf boundaries*.

    This is a CPU-friendly setting meant to make boundary information matter:
    if `disjoint_topic_vocab=True`, cross-topic bigrams occur only at segment boundaries,
    which are placed between leaves. Leaf-only supervision will never observe those
    cross-leaf bigrams.
    """
    rng = _set_global_seed(seed)
    profile_rng = _set_global_seed(int(boundary_profile_seed))
    profile_fn = make_boundary_profile(
        str(boundary_profile),
        strength=float(boundary_profile_strength),
        rng=profile_rng,
    )
    v = int(vocab_size)
    k = int(n_topics)
    if v < 2:
        raise ValueError("vocab_size must be >= 2")
    if k < 1:
        raise ValueError("n_topics must be >= 1")
    if int(min_tokens) < 2 or int(max_tokens) < int(min_tokens):
        raise ValueError("require 2 <= min_tokens <= max_tokens")

    alpha = float(topic_concentration)
    if alpha <= 0:
        raise ValueError("topic_concentration must be > 0")

    topics: List[np.ndarray] = []
    if disjoint_topic_vocab:
        # Partition vocab into k contiguous blocks.
        block_size = max(1, v // k)
        for t in range(k):
            start = int(t * block_size)
            end = int((t + 1) * block_size) if t < k - 1 else v
            support = np.arange(start, end, dtype=np.int64)
            probs = np.zeros((v,), dtype=np.float64)
            if support.size == 0:
                support = np.arange(0, v, dtype=np.int64)
            sub = rng.dirichlet(np.full((support.size,), alpha, dtype=np.float64))
            probs[support] = sub
            topics.append(probs)
    else:
        for _ in range(k):
            probs = rng.dirichlet(np.full((v,), alpha, dtype=np.float64))
            topics.append(probs.astype(np.float64, copy=False))

    leaf = int(max(1, leaf_tokens))
    min_seg_leaves = int(max(1, math.ceil(float(min_seg_len) / float(leaf))))
    max_seg_leaves = int(max(1, math.floor(float(max_seg_len) / float(leaf))))
    max_seg_leaves = max(min_seg_leaves, max_seg_leaves)

    docs: List[Tuple[int, ...]] = []
    for _ in range(int(n_docs)):
        n = int(rng.integers(int(min_tokens), int(max_tokens) + 1))
        leaves = build_leaf_spans(n, leaf_tokens=leaf)
        n_leaves = len(leaves)
        if n_leaves == 0:
            docs.append(tuple())
            continue

        s = int(rng.integers(int(min_segments), int(max_segments) + 1))
        s = max(1, min(int(n_leaves), s))
        remaining = int(n_leaves)
        seg_lens_leaves: List[int] = []
        for seg_idx in range(s):
            if seg_idx == s - 1:
                seg_lens_leaves.append(remaining)
                break
            min_needed = min_seg_leaves * (s - seg_idx - 1)
            hi = min(max_seg_leaves, remaining - min_needed)
            lo = min_seg_leaves
            hi = max(lo, hi)
            # Bias segment boundaries by a global profile over boundary position.
            # This introduces cross-document structure that an algorithm can learn globally.
            cand = np.arange(int(lo), int(hi) + 1, dtype=np.int64)
            boundary_idx = (int(n_leaves) - int(remaining)) + cand
            t = boundary_idx.astype(np.float64) / float(max(1, int(n_leaves)))
            w = np.asarray([float(profile_fn(float(tt))) for tt in t.tolist()], dtype=np.float64)
            w = np.maximum(w, 1e-12)
            total = float(np.sum(w))
            probs = (w / total) if total > 0 else None
            seg_len = int(rng.choice(cand, p=probs))
            seg_lens_leaves.append(seg_len)
            remaining -= seg_len
        if sum(seg_lens_leaves) != n_leaves or any(x <= 0 for x in seg_lens_leaves):
            seg_lens_leaves = [n_leaves]

        tokens: List[int] = []
        leaf_i = 0
        for seg_leaves in seg_lens_leaves:
            topic_id = int(rng.integers(0, k))
            probs = topics[topic_id]
            for _leaf_j in range(int(seg_leaves)):
                start, end = leaves[leaf_i]
                leaf_i += 1
                leaf_len = int(end - start)
                draws = rng.choice(v, size=leaf_len, replace=True, p=probs)
                tokens.extend(int(x) for x in draws.tolist())
        docs.append(tuple(tokens))

    return (tuple(docs), tuple(topics))


def sample_oracle_weight_vector(
    *,
    vocab_size: int,
    sparsity: float,
    scale: float,
    seed: int,
) -> np.ndarray:
    rng = _set_global_seed(seed)
    d = int(vocab_size) * int(vocab_size)
    w = rng.normal(loc=0.0, scale=float(scale), size=(d,)).astype(np.float64, copy=False)
    keep = float(sparsity)
    if keep < 1.0:
        keep = max(0.0, keep)
        mask = rng.random((d,)) < keep
        w = w * mask.astype(np.float64, copy=False)
    return w


def oracle_query_cost(
    span_len: int,
    *,
    power: float = 1.0,
    per_query: float = 0.0,
) -> float:
    n = float(max(0, int(span_len)))
    return float(per_query) + (n ** float(power))


def fit_ridge_from_span_queries(
    span_queries: Iterable[Tuple[Sequence[int], float]],
    *,
    vocab_size: int,
    ridge_lambda: float,
) -> np.ndarray:
    d = int(vocab_size) * int(vocab_size)
    lam = float(ridge_lambda)
    if lam < 0:
        raise ValueError("ridge_lambda must be >= 0")
    A = lam * np.eye(d, dtype=np.float64)
    b = np.zeros((d,), dtype=np.float64)

    for tokens, y in span_queries:
        idx, vals = bigram_counts_sparse(tokens, vocab_size=vocab_size)
        if idx.size == 0:
            continue
        A[np.ix_(idx, idx)] += np.outer(vals, vals)
        b[idx] += vals * float(y)

    if d == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.solve(A, b)


def select_guidance_nodes(
    nodes: Sequence[NodeSpan],
    *,
    n_queries: int,
    strategy: str,
    rng: np.random.Generator,
    boundary_rate_by_index: Optional[np.ndarray] = None,
) -> List[int]:
    q = int(max(0, n_queries))
    if q <= 0 or len(nodes) == 0:
        return []
    if q >= len(nodes):
        return list(range(len(nodes)))

    strat = str(strategy).lower().strip()
    if strat == "random":
        idx = rng.choice(len(nodes), size=q, replace=False)
        return [int(x) for x in idx.tolist()]

    if strat == "active":
        # Heuristic: prioritize 2-leaf internal nodes (captures cross-leaf boundary bigrams),
        # then larger internal nodes.
        pair_nodes = [i for i, n in enumerate(nodes) if n.n_leaves == 2]
        other_internal = [i for i, n in enumerate(nodes) if n.n_leaves != 2]

        rng.shuffle(pair_nodes)
        rng.shuffle(other_internal)

        ordered = pair_nodes + other_internal
        return ordered[:q]

    if strat == "profile":
        # Content-derived, globally learnable prioritization: query 2-leaf nodes at boundary indices
        # that are empirically more likely to be changepoints in the training data.
        scored_pairs: List[Tuple[float, float, int]] = []
        other_internal: List[int] = []
        for i, n in enumerate(nodes):
            if n.n_leaves == 2:
                b = int(n.leaf_start)
                score = (
                    float(boundary_rate_by_index[b])
                    if boundary_rate_by_index is not None and 0 <= b < int(boundary_rate_by_index.size)
                    else 0.0
                )
                scored_pairs.append((score, float(rng.random()), int(i)))
            else:
                other_internal.append(int(i))

        scored_pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)
        rng.shuffle(other_internal)
        ordered = [i for _score, _tie, i in scored_pairs] + other_internal
        return ordered[:q]

    if strat == "leaves":
        leaf_nodes = [i for i, n in enumerate(nodes) if n.is_leaf]
        if q >= len(leaf_nodes):
            return leaf_nodes
        idx = rng.choice(len(leaf_nodes), size=q, replace=False)
        return [leaf_nodes[int(x)] for x in idx.tolist()]

    if strat == "root":
        # Nodes are returned with the root last.
        return [len(nodes) - 1]

    raise ValueError(f"Unknown guidance strategy: {strategy!r}")


@dataclass(frozen=True)
class BigramScoreGuidanceConfig:
    vocab_size: int = 16
    n_topics: int = 4
    topic_concentration: float = 0.4
    oracle_feature_mode: str = "topic_bigrams"
    align_segments_to_leaves: bool = True
    disjoint_topic_vocab: bool = True
    cross_topic_weight_multiplier: float = 5.0
    min_tokens: int = 256
    max_tokens: int = 256
    min_segments: int = 3
    max_segments: int = 8
    min_seg_len: int = 16
    max_seg_len: int = 96
    leaf_tokens: int = 32
    boundary_profile: str = "uniform"
    boundary_profile_strength: float = 1.0
    boundary_profile_seed: int = -1
    w_scale: float = 1.0
    w_sparsity: float = 0.25
    ridge_lambda: float = 1e-3
    oracle_cost_power: float = 1.25
    oracle_cost_per_query: float = 0.0
    guidance_per_leaf: Tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)
    guidance_strategies: Tuple[str, ...] = ("random", "active")
    train_docs: int = 200
    test_docs: int = 1000
    seed: int = 0


@dataclass(frozen=True)
class BigramScoreGuidancePolicyMetrics:
    guidance_strategy: str
    guidance_per_leaf: float
    oracle_queries_leaf_total: int
    oracle_queries_extra_total: int
    oracle_queries_total: int
    oracle_cost_leaf_total: float
    oracle_cost_extra_total: float
    oracle_cost_total: float
    mean_abs_error: float
    rmse: float
    weight_rmse: float
    weight_cosine: float


@dataclass(frozen=True)
class BigramScoreGuidanceSummary:
    config: Dict[str, object]
    vocab_size: int
    leaf_tokens: int
    oracle_cost_power: float
    oracle_cost_per_query: float
    mean_leaf_count_train: float
    mean_leaf_count_test: float
    train_full_doc_cost_total: float
    test_oracle_score_rmse: float
    test_boundary_term_rmse: float
    test_boundary_term_fraction: float
    metrics: Dict[str, BigramScoreGuidancePolicyMetrics]

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "vocab_size": int(self.vocab_size),
            "leaf_tokens": int(self.leaf_tokens),
            "oracle_cost_power": float(self.oracle_cost_power),
            "oracle_cost_per_query": float(self.oracle_cost_per_query),
            "mean_leaf_count_train": float(self.mean_leaf_count_train),
            "mean_leaf_count_test": float(self.mean_leaf_count_test),
            "train_full_doc_cost_total": float(self.train_full_doc_cost_total),
            "test_oracle_score_rmse": float(self.test_oracle_score_rmse),
            "test_boundary_term_rmse": float(self.test_boundary_term_rmse),
            "test_boundary_term_fraction": float(self.test_boundary_term_fraction),
            "metrics": {k: asdict(v) for k, v in self.metrics.items()},
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def run_bigram_score_guidance_experiment(config: BigramScoreGuidanceConfig) -> BigramScoreGuidanceSummary:
    rng = _set_global_seed(config.seed)
    vocab_size = int(config.vocab_size)
    leaf_tokens = int(config.leaf_tokens)
    oracle_mode = _normalize_oracle_feature_mode(config.oracle_feature_mode)
    if oracle_mode == "topic_bigrams" and not bool(config.disjoint_topic_vocab):
        raise ValueError("oracle_feature_mode=topic_bigrams requires disjoint_topic_vocab=True")

    token_to_block = _token_to_topic_block(vocab_size, int(config.n_topics))
    oracle_vocab_size = vocab_size if oracle_mode == "token_bigrams" else int(config.n_topics)
    w_true = sample_oracle_weight_vector(
        vocab_size=oracle_vocab_size,
        sparsity=float(config.w_sparsity),
        scale=float(config.w_scale),
        seed=int(_splitmix64(config.seed) & 0xFFFFFFFF),
    )
    if float(config.cross_topic_weight_multiplier) != 1.0:
        if oracle_mode == "topic_bigrams":
            block_id = np.arange(oracle_vocab_size, dtype=np.int64)
        else:
            block_id = token_to_block if bool(config.disjoint_topic_vocab) else None
        if block_id is not None:
            w_mat = w_true.reshape((oracle_vocab_size, oracle_vocab_size))
            mask = block_id[:, None] != block_id[None, :]
            w_mat[mask] *= float(config.cross_topic_weight_multiplier)

    if bool(config.align_segments_to_leaves):
        boundary_profile_seed = int(config.boundary_profile_seed)
        if boundary_profile_seed < 0:
            boundary_profile_seed = int(_splitmix64(config.seed + 19) & 0xFFFFFFFF)
        train_docs, _topics = generate_leaf_aligned_topic_docs(
            int(config.train_docs),
            vocab_size=vocab_size,
            n_topics=int(config.n_topics),
            min_tokens=int(config.min_tokens),
            max_tokens=int(config.max_tokens),
            min_segments=int(config.min_segments),
            max_segments=int(config.max_segments),
            min_seg_len=int(config.min_seg_len),
            max_seg_len=int(config.max_seg_len),
            leaf_tokens=leaf_tokens,
            topic_concentration=float(config.topic_concentration),
            disjoint_topic_vocab=bool(config.disjoint_topic_vocab),
            boundary_profile=str(config.boundary_profile),
            boundary_profile_strength=float(config.boundary_profile_strength),
            boundary_profile_seed=boundary_profile_seed,
            seed=int(config.seed),
        )
        test_docs, _topics2 = generate_leaf_aligned_topic_docs(
            int(config.test_docs),
            vocab_size=vocab_size,
            n_topics=int(config.n_topics),
            min_tokens=int(config.min_tokens),
            max_tokens=int(config.max_tokens),
            min_segments=int(config.min_segments),
            max_segments=int(config.max_segments),
            min_seg_len=int(config.min_seg_len),
            max_seg_len=int(config.max_seg_len),
            leaf_tokens=leaf_tokens,
            topic_concentration=float(config.topic_concentration),
            disjoint_topic_vocab=bool(config.disjoint_topic_vocab),
            boundary_profile=str(config.boundary_profile),
            boundary_profile_strength=float(config.boundary_profile_strength),
            boundary_profile_seed=boundary_profile_seed,
            seed=int(_splitmix64(config.seed + 11) & 0xFFFFFFFF),
        )
    else:
        train_docs, _topics = generate_topic_bigram_docs(
            int(config.train_docs),
            vocab_size=vocab_size,
            n_topics=int(config.n_topics),
            min_tokens=int(config.min_tokens),
            max_tokens=int(config.max_tokens),
            min_segments=int(config.min_segments),
            max_segments=int(config.max_segments),
            min_seg_len=int(config.min_seg_len),
            max_seg_len=int(config.max_seg_len),
            topic_concentration=float(config.topic_concentration),
            seed=int(config.seed),
        )
        test_docs, _topics2 = generate_topic_bigram_docs(
            int(config.test_docs),
            vocab_size=vocab_size,
            n_topics=int(config.n_topics),
            min_tokens=int(config.min_tokens),
            max_tokens=int(config.max_tokens),
            min_segments=int(config.min_segments),
            max_segments=int(config.max_segments),
            min_seg_len=int(config.min_seg_len),
            max_seg_len=int(config.max_seg_len),
            topic_concentration=float(config.topic_concentration),
            seed=int(_splitmix64(config.seed + 11) & 0xFFFFFFFF),
        )

    if oracle_mode == "topic_bigrams":
        train_docs_features = [tuple(int(token_to_block[int(t)]) for t in doc) for doc in train_docs]
        test_docs_features = [tuple(int(token_to_block[int(t)]) for t in doc) for doc in test_docs]
    else:
        train_docs_features = list(train_docs)
        test_docs_features = list(test_docs)

    train_leaf_counts: List[int] = []
    test_leaf_counts: List[int] = []
    train_full_doc_cost_total = 0.0

    train_nodes: List[List[NodeSpan]] = []
    for tokens in train_docs:
        nodes = build_balanced_tree_nodes(len(tokens), leaf_tokens=leaf_tokens)
        train_nodes.append(nodes)
        train_leaf_counts.append(sum(1 for n in nodes if n.is_leaf))
        train_full_doc_cost_total += oracle_query_cost(
            len(tokens),
            power=float(config.oracle_cost_power),
            per_query=float(config.oracle_cost_per_query),
        )

    for tokens in test_docs:
        nodes = build_balanced_tree_nodes(len(tokens), leaf_tokens=leaf_tokens)
        test_leaf_counts.append(sum(1 for n in nodes if n.is_leaf))

    metrics: Dict[str, BigramScoreGuidancePolicyMetrics] = {}

    # A simple, globally learnable "where are changepoints?" signal from observed tokens:
    # estimate how often adjacent leaves change topic id at each leaf-boundary index.
    boundary_rate_by_index: Optional[np.ndarray] = None
    if any(str(s).strip().lower() == "profile" for s in config.guidance_strategies):
        max_leaves = int(max(train_leaf_counts)) if train_leaf_counts else 0
        if max_leaves >= 2:
            hits = np.zeros((max_leaves - 1,), dtype=np.float64)
            cnt = np.zeros((max_leaves - 1,), dtype=np.float64)
            for doc_feat in train_docs_features:
                leaf_spans = build_leaf_spans(len(doc_feat), leaf_tokens=leaf_tokens)
                if len(leaf_spans) < 2:
                    continue
                leaf_ids: List[int] = []
                for start, end in leaf_spans:
                    seg = np.asarray(doc_feat[int(start) : int(end)], dtype=np.int64)
                    if seg.size == 0:
                        leaf_ids.append(-1)
                        continue
                    uniq, counts = np.unique(seg, return_counts=True)
                    leaf_ids.append(int(uniq[int(np.argmax(counts))]))
                for i in range(min(len(leaf_ids) - 1, hits.size)):
                    if leaf_ids[i] < 0 or leaf_ids[i + 1] < 0:
                        continue
                    hits[i] += 1.0 if int(leaf_ids[i]) != int(leaf_ids[i + 1]) else 0.0
                    cnt[i] += 1.0
            with np.errstate(divide="ignore", invalid="ignore"):
                boundary_rate_by_index = np.where(cnt > 0, hits / cnt, 0.0)

    # Measure how much of the oracle depends on cross-leaf boundary bigrams under the chosen leaf size.
    # This provides an interpretable "need for correction" scalar in [0,1] at the simulation level:
    # - 0.0 means the oracle is leaf-additive (no missing boundary term).
    # - larger values mean the missing boundary term is a nontrivial fraction of the oracle's scale.
    test_score_sq: List[float] = []
    test_boundary_sq: List[float] = []
    for doc_feat in test_docs_features:
        y_true = oracle_bigram_score(doc_feat, w_true=w_true, vocab_size=oracle_vocab_size)
        leaf_spans = build_leaf_spans(len(doc_feat), leaf_tokens=leaf_tokens)
        leaf_sum = 0.0
        for start, end in leaf_spans:
            leaf_sum += oracle_bigram_score(
                doc_feat[int(start) : int(end)],
                w_true=w_true,
                vocab_size=oracle_vocab_size,
            )
        boundary_term = float(y_true) - float(leaf_sum)
        test_score_sq.append(float(y_true) * float(y_true))
        test_boundary_sq.append(float(boundary_term) * float(boundary_term))
    test_oracle_score_rmse = float(math.sqrt(float(np.mean(test_score_sq)))) if test_score_sq else 0.0
    test_boundary_term_rmse = (
        float(math.sqrt(float(np.mean(test_boundary_sq)))) if test_boundary_sq else 0.0
    )
    test_boundary_term_fraction = (
        float(test_boundary_term_rmse / test_oracle_score_rmse)
        if test_oracle_score_rmse > 0
        else 0.0
    )

    for strategy in config.guidance_strategies:
        for q_leaf in config.guidance_per_leaf:
            span_queries: List[Tuple[Sequence[int], float]] = []
            oracle_queries_leaf_total = 0
            oracle_queries_extra_total = 0
            oracle_cost_leaf_total = 0.0
            oracle_cost_extra_total = 0.0

            for doc_feat, nodes in zip(train_docs_features, train_nodes):
                # Base supervision: one oracle score per leaf.
                leaf_nodes = [n for n in nodes if n.is_leaf]
                n_leaves = len(leaf_nodes)
                for node in leaf_nodes:
                    span = doc_feat[node.token_start : node.token_end]
                    y = oracle_bigram_score(span, w_true=w_true, vocab_size=oracle_vocab_size)
                    span_queries.append((span, y))
                    oracle_queries_leaf_total += 1
                    oracle_cost_leaf_total += oracle_query_cost(
                        len(span),
                        power=float(config.oracle_cost_power),
                        per_query=float(config.oracle_cost_per_query),
                    )

                # Extra guidance: internal node queries (above leaves).
                extra = float(q_leaf)
                q_extra = int(round(extra * float(max(1, n_leaves))))
                internal_nodes = [n for n in nodes if not n.is_leaf]
                selected_internal = select_guidance_nodes(
                    internal_nodes,
                    n_queries=q_extra,
                    strategy=strategy,
                    rng=rng,
                    boundary_rate_by_index=boundary_rate_by_index,
                )

                for rel_idx in selected_internal:
                    node = internal_nodes[int(rel_idx)]
                    span = doc_feat[node.token_start : node.token_end]
                    y = oracle_bigram_score(span, w_true=w_true, vocab_size=oracle_vocab_size)
                    span_queries.append((span, y))
                    oracle_queries_extra_total += 1
                    oracle_cost_extra_total += oracle_query_cost(
                        len(span),
                        power=float(config.oracle_cost_power),
                        per_query=float(config.oracle_cost_per_query),
                    )

            w_hat = fit_ridge_from_span_queries(
                span_queries,
                vocab_size=oracle_vocab_size,
                ridge_lambda=float(config.ridge_lambda),
            )

            # Evaluate on held-out documents.
            abs_errs: List[float] = []
            sq_errs: List[float] = []
            for doc_feat in test_docs_features:
                y_true = oracle_bigram_score(doc_feat, w_true=w_true, vocab_size=oracle_vocab_size)
                idx, vals = bigram_counts_sparse(doc_feat, vocab_size=oracle_vocab_size)
                y_pred = float(np.dot(w_hat[idx], vals)) if idx.size else 0.0
                err = y_pred - y_true
                abs_errs.append(abs(err))
                sq_errs.append(err * err)

            mean_abs_error = float(np.mean(abs_errs)) if abs_errs else 0.0
            rmse = float(math.sqrt(float(np.mean(sq_errs)))) if sq_errs else 0.0

            w_diff = w_hat - w_true
            weight_rmse = float(math.sqrt(float(np.mean(w_diff * w_diff))))
            denom = float(np.linalg.norm(w_hat) * np.linalg.norm(w_true))
            weight_cosine = float(np.dot(w_hat, w_true) / denom) if denom > 0 else 0.0

            key = f"{strategy}_extra{q_leaf:g}"
            oracle_queries_total = int(oracle_queries_leaf_total + oracle_queries_extra_total)
            oracle_cost_total = float(oracle_cost_leaf_total + oracle_cost_extra_total)
            metrics[key] = BigramScoreGuidancePolicyMetrics(
                guidance_strategy=strategy,
                guidance_per_leaf=float(q_leaf),
                oracle_queries_leaf_total=int(oracle_queries_leaf_total),
                oracle_queries_extra_total=int(oracle_queries_extra_total),
                oracle_queries_total=int(oracle_queries_total),
                oracle_cost_leaf_total=float(oracle_cost_leaf_total),
                oracle_cost_extra_total=float(oracle_cost_extra_total),
                oracle_cost_total=float(oracle_cost_total),
                mean_abs_error=float(mean_abs_error),
                rmse=float(rmse),
                weight_rmse=float(weight_rmse),
                weight_cosine=float(weight_cosine),
            )

    return BigramScoreGuidanceSummary(
        config=asdict(config),
        vocab_size=vocab_size,
        leaf_tokens=leaf_tokens,
        oracle_cost_power=float(config.oracle_cost_power),
        oracle_cost_per_query=float(config.oracle_cost_per_query),
        mean_leaf_count_train=float(np.mean(train_leaf_counts)) if train_leaf_counts else 0.0,
        mean_leaf_count_test=float(np.mean(test_leaf_counts)) if test_leaf_counts else 0.0,
        train_full_doc_cost_total=float(train_full_doc_cost_total),
        test_oracle_score_rmse=float(test_oracle_score_rmse),
        test_boundary_term_rmse=float(test_boundary_term_rmse),
        test_boundary_term_fraction=float(test_boundary_term_fraction),
        metrics=metrics,
    )
