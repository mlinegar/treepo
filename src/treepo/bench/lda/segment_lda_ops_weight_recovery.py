"""
OPS-style weight-recovery simulation with a Segment-LDA document generator.

High-level idea
---------------
We generate documents from either:
- **bag-of-words LDA** (topic per token i.i.d. from a per-document Dirichlet mixture), or
- **segmented LDA** (piecewise-constant topic segments, optionally with a global boundary-location profile),
then emit *words* from Dirichlet topic-word distributions (LDA-style emissions).

The oracle target is a mergeable linear functional of the *latent* topic sequence:

    f⋆(span) = <θ, topic_counts(span)> + λ * <W, topic_bigrams(span)>

where:
- θ ∈ R^K is sparse (only a few "relevant topics" have nonzero weights),
- W ∈ R^{K×K} is sparse (only transitions involving relevant topics matter),
- λ ≥ 0 is a scalar multiplier for the bigram term (we include λ in the grid).

OPS connection / Lean alignment
-------------------------------
This targets the same mergeability structure as:
- `lean3/FormalProofs/OPT/BigramSketch.lean` (boundary metadata needed for bigram mergeability)
- `lean3/FormalProofs/OPT/MarkovCountSketchExample.lean` (exact sketch vs undersupported; L3 failure)

The minimal exact mergeable sketch state for the oracle is:
- topic unigram counts (K)
- topic bigram counts (K^2)
- first / last topic ids (to add the cross-boundary bigram on merges)

We fit (θ, λW) via ridge regression from span-level oracle queries on leaves + a sampled set of
internal nodes (OPS-style C1/C3 supervision).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import itertools
import json
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


Span = Tuple[int, int]  # [start, end) token indices

ScheduleName = str
VALID_SCHEDULES: Tuple[ScheduleName, ...] = ("balanced", "left_to_right", "right_to_left")

AuditPolicyName = str
VALID_AUDIT_POLICIES: Tuple[AuditPolicyName, ...] = ("all", "fixed", "fraction", "sqrt", "log2")

AuditStrategyName = str
VALID_AUDIT_STRATEGIES: Tuple[AuditStrategyName, ...] = ("random", "active_small", "profile")

TopicSourceName = str
VALID_TOPIC_SOURCES: Tuple[TopicSourceName, ...] = ("true", "infer")

TopicPhiEstimatorName = str
VALID_TOPIC_PHI_ESTIMATORS: Tuple[TopicPhiEstimatorName, ...] = (
    "true",
    "noisy_theory",
    "tensor_lda",
    "online_tensor_lda",
    "sklearn_lda",
    "embedding_spectral",
    "neural_ctreepo",
    "neural_mergeable_sketch",
    "neural_hybrid",
    "neural_embedding_hybrid",
)

NEURAL_TOPIC_PHI_ESTIMATORS: Tuple[TopicPhiEstimatorName, ...] = (
    "neural_ctreepo",
    "neural_mergeable_sketch",
    "neural_hybrid",
)

BoundaryProfileName = str
VALID_BOUNDARY_PROFILES: Tuple[BoundaryProfileName, ...] = (
    "uniform",
    "start",
    "middle",
    "end",
    "bimodal",
    "random",
)

TopicProcessName = str
VALID_TOPIC_PROCESSES: Tuple[TopicProcessName, ...] = ("segments", "bag_of_words")

FeatureInferenceName = str
VALID_FEATURE_INFERENCE: Tuple[FeatureInferenceName, ...] = ("hard", "soft")


def _splitmix64(x: int) -> int:
    z = (int(x) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z & 0xFFFFFFFFFFFFFFFF


def make_boundary_profile(
    profile: BoundaryProfileName,
    *,
    strength: float,
    rng: np.random.Generator,
) -> Callable[[float], float]:
    """Return a positive-valued boundary weight function w(t) over normalized position t in (0,1).

    Used to bias segment boundary locations in `generate_segment_lda_docs`, creating globally
    learnable structure like "changepoints concentrate in the middle" or "bimodal at the ends".
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

    raise ValueError(f"unsupported boundary_profile: {profile!r} (expected one of {VALID_BOUNDARY_PROFILES})")


def oracle_query_cost(span_len: int, *, power: float = 1.0, per_query: float = 0.0) -> float:
    n = float(max(0, int(span_len)))
    return float(per_query) + (n ** float(power))


def audit_sample_count(
    internal_nodes: int,
    *,
    policy: AuditPolicyName,
    fixed_nodes: int = 0,
    fraction: float = 1.0,
    scale: float = 1.0,
) -> int:
    """
    How many realized internal nodes to label (per doc), mirroring ops-count semantics.
    """

    n = int(max(0, internal_nodes))
    if n <= 0:
        return 0

    pol = str(policy)
    if pol == "all":
        q = n
    elif pol == "fixed":
        q = int(max(0, fixed_nodes))
    elif pol == "fraction":
        q = int(math.ceil(float(fraction) * float(n)))
    elif pol == "sqrt":
        q = int(math.ceil(float(scale) * math.sqrt(float(n))))
    elif pol == "log2":
        q = int(math.ceil(float(scale) * math.log2(float(n) + 1.0)))
    else:
        raise ValueError(f"unsupported audit policy: {policy!r}; expected one of {VALID_AUDIT_POLICIES}")
    return int(max(0, min(n, q)))


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


def select_audit_nodes(
    internal_nodes: Sequence[NodeSpan],
    *,
    n_queries: int,
    strategy: AuditStrategyName,
    rng: np.random.Generator,
    boundary_rate_by_index: Optional[np.ndarray] = None,
) -> List[int]:
    q = int(max(0, n_queries))
    if q <= 0 or len(internal_nodes) == 0:
        return []
    if q >= len(internal_nodes):
        return list(range(len(internal_nodes)))

    strat = str(strategy).lower().strip()
    if strat == "random":
        idx = rng.choice(len(internal_nodes), size=q, replace=False)
        return [int(x) for x in idx.tolist()]

    if strat == "active_small":
        # Prioritize 2-leaf internal nodes (captures cross-leaf boundary bigrams cleanly),
        # then larger nodes.
        pair_nodes = [i for i, n in enumerate(internal_nodes) if n.n_leaves == 2]
        other_nodes = [i for i, n in enumerate(internal_nodes) if n.n_leaves != 2]
        rng.shuffle(pair_nodes)
        rng.shuffle(other_nodes)
        ordered = pair_nodes + other_nodes
        return ordered[:q]

    if strat == "profile":
        # Content-derived, globally learnable prioritization: query 2-leaf nodes at boundary indices
        # that are empirically more likely to be changepoints in the training data.
        scored_pairs: List[Tuple[float, float, int]] = []
        other_nodes: List[int] = []
        for i, n in enumerate(internal_nodes):
            if n.n_leaves == 2:
                b = int(n.leaf_start)
                score = (
                    float(boundary_rate_by_index[b])
                    if boundary_rate_by_index is not None and 0 <= b < int(boundary_rate_by_index.size)
                    else 0.0
                )
                scored_pairs.append((score, float(rng.random()), int(i)))
            else:
                other_nodes.append(int(i))

        scored_pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)
        rng.shuffle(other_nodes)
        ordered = [i for _score, _tie, i in scored_pairs] + other_nodes
        return ordered[:q]

    raise ValueError(f"unsupported audit strategy: {strategy!r}; expected one of {VALID_AUDIT_STRATEGIES}")


@dataclass(frozen=True)
class SegmentLDADoc:
    tokens: Tuple[int, ...]
    topics: Tuple[int, ...]  # true topic id per token


def _dirichlet_on_support(
    rng: np.random.Generator,
    *,
    vocab_size: int,
    support: Sequence[int],
    concentration: float,
) -> np.ndarray:
    v = int(vocab_size)
    alpha = float(concentration)
    if alpha <= 0:
        raise ValueError("concentration must be positive")
    probs = np.zeros((v,), dtype=np.float64)
    supp = np.asarray(list(support), dtype=np.int64)
    if supp.size == 0:
        raise ValueError("support must be non-empty")
    sub = rng.dirichlet(np.full((supp.size,), alpha, dtype=np.float64))
    probs[supp] = sub
    return probs


def sample_topic_distributions(
    *,
    vocab_size: int,
    n_topics: int,
    topic_concentration: float,
    emission_mode: str,
    anchor_words_per_topic: int,
    anchor_multiplier: float,
    seed: int,
) -> Tuple[Tuple[np.ndarray, ...], Dict[str, object]]:
    """
    Sample topic-word distributions φ_k.

    Modes:
    - "disjoint": each topic's support is a disjoint vocab block.
    - "anchored": each topic has disjoint anchor words + shared background words.
    """

    rng = np.random.default_rng(int(seed))
    v = int(vocab_size)
    k = int(n_topics)
    if v < 2:
        raise ValueError("vocab_size must be >= 2")
    if k < 2:
        raise ValueError("n_topics must be >= 2")

    mode = str(emission_mode).strip().lower()
    alpha = float(topic_concentration)
    if alpha <= 0:
        raise ValueError("topic_concentration must be > 0")

    if mode == "disjoint":
        block = max(1, v // k)
        topics: List[np.ndarray] = []
        for topic_id in range(k):
            lo = int(topic_id * block)
            hi = int(v if topic_id == k - 1 else min(v, (topic_id + 1) * block))
            support = list(range(lo, hi))
            topics.append(_dirichlet_on_support(rng, vocab_size=v, support=support, concentration=alpha))
        meta = {"mode": mode, "block_size": int(block)}
        return (tuple(topics), meta)

    if mode == "anchored":
        a = int(max(1, anchor_words_per_topic))
        if a * k >= v:
            raise ValueError("anchor_words_per_topic too large: need K * anchors < vocab_size")
        mult = float(anchor_multiplier)
        if mult <= 1.0:
            raise ValueError("anchor_multiplier must be > 1.0")

        anchors: List[List[int]] = []
        start = 0
        for _topic_id in range(k):
            anchors.append(list(range(start, start + a)))
            start += a
        shared = list(range(start, v))

        topics = []
        for topic_id in range(k):
            alpha_vec = np.full((v,), alpha, dtype=np.float64)
            alpha_vec[anchors[topic_id]] *= mult
            probs = rng.dirichlet(alpha_vec)
            topics.append(probs.astype(np.float64, copy=False))
        meta = {
            "mode": mode,
            "anchor_words_per_topic": int(a),
            "anchor_multiplier": float(mult),
            "shared_words": int(len(shared)),
            "anchors": anchors,
        }
        return (tuple(topics), meta)

    raise ValueError(f"unsupported emission_mode: {emission_mode!r} (expected 'disjoint' or 'anchored')")


def _theorem51_l2_error_bound(
    *,
    c: float,
    alpha0: float,
    pmin: float,
    sigmaK: float,
    k: int,
    delta: float,
    N: int,
) -> float:
    """
    Mirror of the Lean-side rate expression `theorem51L2ErrorBound` (Anandkumar et al. 2013, Thm 5.1).

    This is a *scaling model* used for simulation; we do not attempt to re-derive the theorem
    conditions here.
    """

    cc = float(c)
    a0 = float(alpha0)
    p = float(pmin)
    s = float(sigmaK)
    kk = int(k)
    d = float(delta)
    n = int(N)
    if n <= 0:
        return float("inf")
    if not (0.0 < d < 1.0):
        raise ValueError("delta must be in (0,1)")
    if p <= 0.0:
        raise ValueError("pmin must be positive")
    if s <= 0.0:
        raise ValueError("sigmaK must be positive")
    scale = cc * (((a0 + 1.0) ** 2) * (float(kk) ** 3) / ((p**2) * (s**3)))
    return float(scale) * ((1.0 + math.sqrt(math.log(1.0 / d))) / math.sqrt(float(n)))


def _best_topic_permutation_l2(
    topics_est: Sequence[np.ndarray],
    topics_true: Sequence[np.ndarray],
) -> Tuple[Tuple[int, ...], np.ndarray]:
    """
    Find σ : est_index ↦ true_index minimizing Σ_i ||φ̂_i - φ_{σ(i)}||₂.

    Returns (perm, cost_matrix) where perm[i]=σ(i).
    """

    k = int(len(topics_true))
    if int(len(topics_est)) != k:
        raise ValueError("topics_est and topics_true must have same length")
    if k <= 0:
        return (tuple(), np.zeros((0, 0), dtype=np.float64))

    est = np.stack([np.asarray(t, dtype=np.float64).reshape(-1) for t in topics_est], axis=0)
    tru = np.stack([np.asarray(t, dtype=np.float64).reshape(-1) for t in topics_true], axis=0)
    if est.shape != tru.shape:
        raise ValueError("topics_est and topics_true must have aligned shapes")

    cost = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        diff = tru[None, :, :] - est[i : i + 1, None, :]
        cost[i] = np.linalg.norm(diff.reshape(k, -1), axis=1)

    # Exact assignment via brute force for small K.
    if k <= 9:
        best_perm: Tuple[int, ...] = tuple(range(k))
        best = float("inf")
        for perm in itertools.permutations(range(k)):
            total = 0.0
            for i, j in enumerate(perm):
                total += float(cost[i, j])
                if total >= best:
                    break
            if total < best:
                best = total
                best_perm = tuple(int(x) for x in perm)
        return best_perm, cost

    # Greedy fallback for large K (kept simple; this is a simulation helper).
    remaining = set(range(k))
    perm_out: List[int] = [-1 for _ in range(k)]
    for i in range(k):
        j = min(remaining, key=lambda jj: float(cost[i, jj]))
        perm_out[i] = int(j)
        remaining.remove(j)
    return (tuple(perm_out), cost)


def _invert_perm(perm: Sequence[int]) -> Tuple[int, ...]:
    k = int(len(perm))
    inv = [-1 for _ in range(k)]
    for i, j in enumerate(perm):
        jj = int(j)
        if jj < 0 or jj >= k:
            raise ValueError("perm must be a bijection on [0,K)")
        inv[jj] = int(i)
    if any(x < 0 for x in inv):
        raise ValueError("perm must be a bijection on [0,K)")
    return tuple(int(x) for x in inv)


def _permute_theta_to_true_basis(theta_est: np.ndarray, *, perm_est_to_true: Sequence[int]) -> np.ndarray:
    inv = _invert_perm(tuple(int(x) for x in perm_est_to_true))
    th = np.asarray(theta_est, dtype=np.float64).reshape(-1)
    if th.size != len(inv):
        raise ValueError("theta size mismatch for permutation")
    return th[np.asarray(inv, dtype=np.int64)].astype(np.float64, copy=False)


def _permute_W_to_true_basis(W_est: np.ndarray, *, perm_est_to_true: Sequence[int]) -> np.ndarray:
    inv = _invert_perm(tuple(int(x) for x in perm_est_to_true))
    W = np.asarray(W_est, dtype=np.float64)
    k = int(len(inv))
    if W.shape != (k, k):
        raise ValueError("W shape mismatch for permutation")
    idx = np.asarray(inv, dtype=np.int64)
    return W[np.ix_(idx, idx)].astype(np.float64, copy=False)


def _docs_to_frequency_matrix(
    docs_tokens: Sequence[Sequence[int]],
    *,
    vocab_size: int,
) -> np.ndarray:
    """
    Convert tokenized documents to a D×V matrix of empirical word frequencies (rows sum to 1).

    This corresponds to the "document vector" viewpoint used in the Tensor-LDA paper for
    centered-moment computation.
    """

    v = int(vocab_size)
    if v <= 0:
        raise ValueError("vocab_size must be positive")
    D = int(len(docs_tokens))
    F = np.zeros((D, v), dtype=np.float64)
    for i, toks in enumerate(docs_tokens):
        x = np.asarray(list(toks), dtype=np.int64)
        if x.size == 0:
            continue
        if int(np.max(x)) >= v or int(np.min(x)) < 0:
            raise ValueError("tokens contain ids outside vocab_size")
        cnt = np.bincount(x, minlength=v).astype(np.float64, copy=False)
        F[int(i)] = cnt / float(x.size)
    return F


def _docs_to_count_matrix(
    docs_tokens: Sequence[Sequence[int]],
    *,
    vocab_size: int,
) -> np.ndarray:
    """
    Convert tokenized documents to a D×V integer count matrix.

    This is the standard "document-term matrix" (DTM) representation consumed by most
    classical LDA implementations (e.g. scikit-learn).
    """

    v = int(vocab_size)
    if v <= 0:
        raise ValueError("vocab_size must be positive")
    D = int(len(docs_tokens))
    X = np.zeros((D, v), dtype=np.int64)
    for i, toks in enumerate(docs_tokens):
        x = np.asarray(list(toks), dtype=np.int64)
        if x.size == 0:
            continue
        if int(np.max(x)) >= v or int(np.min(x)) < 0:
            raise ValueError("tokens contain ids outside vocab_size")
        X[int(i)] = np.bincount(x, minlength=v).astype(np.int64, copy=False)
    return X


def _symmetrize_tensor3(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.ndim != 3 or T.shape[0] != T.shape[1] or T.shape[1] != T.shape[2]:
        raise ValueError("T must be K×K×K")
    return (
        T
        + np.transpose(T, (1, 0, 2))
        + np.transpose(T, (2, 1, 0))
        + np.transpose(T, (0, 2, 1))
        + np.transpose(T, (1, 2, 0))
        + np.transpose(T, (2, 0, 1))
    ) / 6.0


def _tensor_power_decompose(
    T: np.ndarray,
    *,
    n_components: int,
    n_iters: int,
    n_restarts: int,
    tol: float,
    seed: int,
) -> Tuple[Tuple[np.ndarray, ...], Tuple[float, ...]]:
    """
    Symmetric tensor power method with deflation (orthogonal-ish components).

    For an (approximately) orthogonal decomposition T ≈ Σ λ_i v_i^{⊗3}, returns (v_i, λ_i).
    """

    rng = np.random.default_rng(int(seed))
    Twork = np.asarray(T, dtype=np.float64).copy()
    if Twork.ndim != 3 or Twork.shape[0] != Twork.shape[1] or Twork.shape[1] != Twork.shape[2]:
        raise ValueError("T must be K×K×K")
    K = int(Twork.shape[0])
    m = int(max(0, n_components))
    if m == 0:
        return (tuple(), tuple())

    vecs: List[np.ndarray] = []
    vals: List[float] = []
    for _comp in range(m):
        best_v: Optional[np.ndarray] = None
        best_val = 0.0

        for _restart in range(int(max(1, n_restarts))):
            v = rng.normal(size=(K,)).astype(np.float64, copy=False)
            nrm = float(np.linalg.norm(v))
            if nrm <= 0:
                continue
            v /= nrm

            for _it in range(int(max(1, n_iters))):
                v_new = np.einsum("ijk,j,k->i", Twork, v, v).astype(np.float64, copy=False)
                for u in vecs:
                    v_new = v_new - float(np.dot(u, v_new)) * u
                nrm_new = float(np.linalg.norm(v_new))
                if nrm_new <= 1e-12:
                    break
                v_new /= nrm_new
                if float(np.linalg.norm(v_new - v)) < float(tol) or float(np.linalg.norm(v_new + v)) < float(tol):
                    v = v_new
                    break
                v = v_new

            lam = float(np.einsum("ijk,i,j,k", Twork, v, v, v))
            if abs(lam) > abs(best_val):
                best_val = float(lam)
                best_v = v.copy()

        if best_v is None:
            break

        if best_val < 0:
            best_val = -best_val
            best_v = -best_v

        vecs.append(best_v)
        vals.append(float(best_val))
        Twork = Twork - float(best_val) * np.einsum("i,j,k->ijk", best_v, best_v, best_v).astype(
            np.float64, copy=False
        )

    return (tuple(vecs), tuple(vals))


def _estimate_topics_tensor_lda_from_docs(
    docs_tokens: Sequence[Sequence[int]],
    *,
    vocab_size: int,
    n_topics: int,
    doc_topic_concentration: float,
    seed: int,
    m2_ridge: float = 1e-8,
    power_iters: int = 200,
    power_restarts: int = 16,
    power_tol: float = 1e-7,
) -> Tuple[Tuple[np.ndarray, ...], Dict[str, float]]:
    """
    Batch "Tensor-LDA" style topic estimation:
      1) build centered second/third moments from document frequency vectors,
      2) whiten M2 and form a K×K×K tensor,
      3) decompose the whitened tensor by the power method,
      4) unwhiten and re-center to recover φ̂_k.

    This is intended as a computational baseline consistent with the paper's centered-moment layer,
    not a production-grade implementation.
    """

    k = int(n_topics)
    v = int(vocab_size)
    if k < 2:
        raise ValueError("n_topics must be >= 2")
    if v < 2:
        raise ValueError("vocab_size must be >= 2")
    if float(doc_topic_concentration) <= 0:
        raise ValueError("doc_topic_concentration must be positive")

    F = _docs_to_frequency_matrix(docs_tokens, vocab_size=v)
    D = int(F.shape[0])
    if D <= 0:
        raise ValueError("need at least one document for tensor_lda estimation")

    m1 = np.mean(F, axis=0).astype(np.float64, copy=False)
    Xc = (F - m1[None, :]).astype(np.float64, copy=False)

    alpha0 = float(k) * float(doc_topic_concentration)
    scale2 = float(alpha0 + 1.0) / float(D)
    M2 = scale2 * (Xc.T @ Xc)
    M2 = 0.5 * (M2 + M2.T)
    ridge = float(m2_ridge)
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("m2_ridge must be finite and >= 0")
    if ridge > 0:
        M2 = M2 + ridge * np.eye(v, dtype=np.float64)

    evals, evecs = np.linalg.eigh(M2)
    order = np.argsort(evals)[::-1]
    top = order[:k]
    s = np.maximum(evals[top].astype(np.float64, copy=False), 1e-12)
    U = evecs[:, top].astype(np.float64, copy=False)

    W = U @ (np.diag(1.0 / np.sqrt(s)))
    B = U @ (np.diag(np.sqrt(s)))
    whiten_err = float(np.linalg.norm(W.T @ M2 @ W - np.eye(k, dtype=np.float64), ord="fro"))

    Y = (Xc @ W).astype(np.float64, copy=False)  # D×K
    scale3 = float((alpha0 + 1.0) * (alpha0 + 2.0)) / float(2.0 * D)
    T = scale3 * np.einsum("ik,il,im->klm", Y, Y, Y).astype(np.float64, copy=False)
    T = _symmetrize_tensor3(T)

    vecs, lambdas = _tensor_power_decompose(
        T,
        n_components=k,
        n_iters=int(power_iters),
        n_restarts=int(power_restarts),
        tol=float(power_tol),
        seed=int(seed),
    )
    if len(vecs) != k:
        raise RuntimeError(f"tensor power method recovered {len(vecs)}/{k} components")

    topics_est: List[np.ndarray] = []
    for v_i, lam in zip(vecs, lambdas):
        nu = float(lam) * (B @ np.asarray(v_i, dtype=np.float64).reshape(-1))
        mu = nu + m1
        mu = np.clip(mu, 1e-12, None)
        mu = mu / float(np.sum(mu))
        topics_est.append(mu.astype(np.float64, copy=False))

    meta = {
        "topic_phi_estimator": "tensor_lda",
        "topic_phi_docs_effective": float(D),
        "tensor_lda_alpha0": float(alpha0),
        "tensor_lda_m2_ridge": float(ridge),
        "tensor_lda_m2_top_eigs_min": float(np.min(s)) if s.size else float("nan"),
        "tensor_lda_m2_top_eigs_max": float(np.max(s)) if s.size else float("nan"),
        "tensor_lda_whiten_fro_error": float(whiten_err),
        "tensor_lda_power_iters": float(power_iters),
        "tensor_lda_power_restarts": float(power_restarts),
        "tensor_lda_lambdas_min": float(np.min(np.asarray(lambdas, dtype=np.float64))) if lambdas else float("nan"),
        "tensor_lda_lambdas_max": float(np.max(np.asarray(lambdas, dtype=np.float64))) if lambdas else float("nan"),
    }
    return (tuple(topics_est), meta)


def _estimate_topics_online_tensor_lda_from_docs(
    docs_tokens: Sequence[Sequence[int]],
    *,
    vocab_size: int,
    n_topics: int,
    doc_topic_concentration: float,
    seed: int,
    burn_in_docs: int,
    batch_docs: int,
    passes: int,
    lr: float,
    grad_clip_norm: float,
    m2_ridge: float = 1e-8,
    power_iters: int = 200,
    power_restarts: int = 16,
    power_tol: float = 1e-7,
) -> Tuple[Tuple[np.ndarray, ...], Dict[str, float]]:
    """
    Online/STGD-flavored Tensor-LDA baseline:
      - estimate `m1`, `M2` and whitening from an initial burn-in,
      - initialize factors by tensor power method on the burn-in whitened third moment,
      - stream mini-batches and update tensor factors by stochastic gradient steps (Eq. (13)-style).

    Notes:
      - Whitening is frozen after burn-in (keeps the implementation simple and stable).
      - This is intended for comparative simulation, not as a full reproduction of the paper.
    """

    k = int(n_topics)
    v = int(vocab_size)
    if k < 2:
        raise ValueError("n_topics must be >= 2")
    if v < 2:
        raise ValueError("vocab_size must be >= 2")
    if float(doc_topic_concentration) <= 0:
        raise ValueError("doc_topic_concentration must be positive")
    D = int(len(docs_tokens))
    if D <= 0:
        raise ValueError("need at least one document for online_tensor_lda estimation")

    burn_cfg = int(max(0, burn_in_docs))
    if burn_cfg == 0:
        burn_cfg = int(min(D, max(200, 20 * k)))
    burn = int(max(k, min(D, burn_cfg)))

    bsz = int(max(1, batch_docs))
    n_passes = int(max(1, passes))
    step = float(lr)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("lr must be finite and > 0")
    clip = float(grad_clip_norm)
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("grad_clip_norm must be finite and > 0")

    # Burn-in moments and whitening.
    F0 = _docs_to_frequency_matrix(docs_tokens[:burn], vocab_size=v)
    m1 = np.mean(F0, axis=0).astype(np.float64, copy=False)
    Xc0 = (F0 - m1[None, :]).astype(np.float64, copy=False)

    alpha0 = float(k) * float(doc_topic_concentration)
    scale2 = float(alpha0 + 1.0) / float(burn)
    M2 = scale2 * (Xc0.T @ Xc0)
    M2 = 0.5 * (M2 + M2.T)
    ridge = float(m2_ridge)
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("m2_ridge must be finite and >= 0")
    if ridge > 0:
        M2 = M2 + ridge * np.eye(v, dtype=np.float64)

    evals, evecs = np.linalg.eigh(M2)
    order = np.argsort(evals)[::-1]
    top = order[:k]
    s = np.maximum(evals[top].astype(np.float64, copy=False), 1e-12)
    U = evecs[:, top].astype(np.float64, copy=False)
    W = U @ (np.diag(1.0 / np.sqrt(s)))
    B = U @ (np.diag(np.sqrt(s)))
    whiten_err = float(np.linalg.norm(W.T @ M2 @ W - np.eye(k, dtype=np.float64), ord="fro"))

    # Initialize factors by a power decomposition on the burn-in whitened third moment.
    Y0 = (Xc0 @ W).astype(np.float64, copy=False)
    scale3 = float((alpha0 + 1.0) * (alpha0 + 2.0)) / float(2.0 * burn)
    T0 = scale3 * np.einsum("ik,il,im->klm", Y0, Y0, Y0).astype(np.float64, copy=False)
    T0 = _symmetrize_tensor3(T0)
    vecs0, lambdas0 = _tensor_power_decompose(
        T0,
        n_components=k,
        n_iters=int(power_iters),
        n_restarts=int(power_restarts),
        tol=float(power_tol),
        seed=int(seed),
    )
    if len(vecs0) != k:
        raise RuntimeError(f"online_tensor_lda init recovered {len(vecs0)}/{k} components")

    V = np.stack([np.asarray(x, dtype=np.float64).reshape(-1) for x in vecs0], axis=1)  # K×K
    lam = np.asarray(lambdas0, dtype=np.float64).reshape(-1)
    if V.shape != (k, k) or lam.size != k:
        raise RuntimeError("online_tensor_lda init shape mismatch")

    rng = np.random.default_rng(int(seed))
    n_steps = 0
    last_loss = float("nan")
    for _ep in range(n_passes):
        order_idx = np.arange(D, dtype=np.int64)
        rng.shuffle(order_idx)
        for start in range(0, D, bsz):
            idx = order_idx[int(start) : int(min(D, start + bsz))]
            batch = [docs_tokens[int(i)] for i in idx.tolist()]
            Fb = _docs_to_frequency_matrix(batch, vocab_size=v)
            Xcb = (Fb - m1[None, :]).astype(np.float64, copy=False)
            Yb = (Xcb @ W).astype(np.float64, copy=False)
            scale3b = float((alpha0 + 1.0) * (alpha0 + 2.0)) / float(2.0 * max(1, int(Yb.shape[0])))
            Tb = scale3b * np.einsum("ik,il,im->klm", Yb, Yb, Yb).astype(np.float64, copy=False)
            Tb = _symmetrize_tensor3(Tb)

            # Current model tensor and residual.
            Tpred = np.zeros((k, k, k), dtype=np.float64)
            for j in range(k):
                vj = V[:, j]
                Tpred += float(lam[j]) * np.einsum("i,j,k->ijk", vj, vj, vj).astype(np.float64, copy=False)
            R = (Tpred - Tb).astype(np.float64, copy=False)
            last_loss = 0.5 * float(np.sum(R * R))

            # STGD updates (coordinate-wise).
            for j in range(k):
                vj = V[:, j].copy()
                lamj = float(lam[j])

                gv = (3.0 * lamj) * np.einsum("abc,b,c->a", R, vj, vj).astype(np.float64, copy=False)
                gl = float(np.einsum("abc,a,b,c", R, vj, vj, vj))

                gvn = float(np.linalg.norm(gv))
                if gvn > clip:
                    gv = (clip / gvn) * gv

                vj = vj - step * gv
                lamj = lamj - step * gl

                vn = float(np.linalg.norm(vj))
                if vn > 1e-12:
                    vj = vj / vn
                    lamj = lamj * (vn**3)

                if lamj < 0:
                    lamj = -lamj
                    vj = -vj

                V[:, j] = vj
                lam[j] = float(lamj)

            n_steps += 1

    topics_est: List[np.ndarray] = []
    for j in range(k):
        nu = float(lam[j]) * (B @ V[:, j]).astype(np.float64, copy=False)
        mu = nu + m1
        mu = np.clip(mu, 1e-12, None)
        mu = mu / float(np.sum(mu))
        topics_est.append(mu.astype(np.float64, copy=False))

    meta = {
        "topic_phi_estimator": "online_tensor_lda",
        "topic_phi_docs_effective": float(D),
        "online_tensor_lda_alpha0": float(alpha0),
        "online_tensor_lda_burn_in_docs": float(burn),
        "online_tensor_lda_batch_docs": float(bsz),
        "online_tensor_lda_passes": float(n_passes),
        "online_tensor_lda_lr": float(step),
        "online_tensor_lda_grad_clip_norm": float(clip),
        "online_tensor_lda_m2_ridge": float(ridge),
        "online_tensor_lda_whiten_fro_error": float(whiten_err),
        "online_tensor_lda_steps": float(n_steps),
        "online_tensor_lda_last_batch_loss": float(last_loss),
        "online_tensor_lda_lambdas_min": float(np.min(lam)) if lam.size else float("nan"),
        "online_tensor_lda_lambdas_max": float(np.max(lam)) if lam.size else float("nan"),
    }
    return (tuple(topics_est), meta)


def _estimate_topics_sklearn_lda_from_docs(
    docs_tokens: Sequence[Sequence[int]],
    *,
    vocab_size: int,
    n_topics: int,
    doc_topic_concentration: float,
    topic_word_concentration: float,
    seed: int,
    max_iter: int = 60,
) -> Tuple[Tuple[np.ndarray, ...], Dict[str, float]]:
    """
    Classical LDA baseline via scikit-learn (variational Bayes).

    This is intentionally a simple reference implementation for "DTM -> LDA" comparisons.
    """

    try:
        from sklearn.decomposition import LatentDirichletAllocation  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for topic_phi_estimator='sklearn_lda'. "
            "Install with: pip install 'treepo[sklearn]'"
        ) from e

    k = int(n_topics)
    v = int(vocab_size)
    if k < 2:
        raise ValueError("n_topics must be >= 2")
    if v < 2:
        raise ValueError("vocab_size must be >= 2")
    if float(doc_topic_concentration) <= 0:
        raise ValueError("doc_topic_concentration must be positive")
    if float(topic_word_concentration) <= 0:
        raise ValueError("topic_word_concentration must be positive")
    iters = int(max_iter)
    if iters < 1:
        raise ValueError("max_iter must be >= 1")

    X = _docs_to_count_matrix(docs_tokens, vocab_size=v)
    D = int(X.shape[0])
    if D <= 0:
        raise ValueError("need at least one document for sklearn_lda estimation")

    lda = LatentDirichletAllocation(
        n_components=int(k),
        max_iter=int(iters),
        learning_method="batch",
        evaluate_every=-1,
        random_state=int(seed),
        n_jobs=1,
        doc_topic_prior=float(doc_topic_concentration),
        topic_word_prior=float(topic_word_concentration),
    )
    lda.fit(X)

    comps = np.asarray(getattr(lda, "components_"), dtype=np.float64)
    if comps.shape != (k, v):
        raise RuntimeError("sklearn_lda components_ shape mismatch")
    comps = np.clip(comps, 1e-12, None)
    topic_word = comps / np.sum(comps, axis=1, keepdims=True)
    topics_est = tuple(np.asarray(topic_word[i], dtype=np.float64).reshape(-1) for i in range(k))

    meta = {
        "topic_phi_estimator": "sklearn_lda",
        "topic_phi_docs_effective": float(D),
        "sklearn_lda_max_iter": float(iters),
        "sklearn_lda_n_iter": float(getattr(lda, "n_iter_", float("nan"))),
        "sklearn_lda_doc_topic_prior": float(doc_topic_concentration),
        "sklearn_lda_topic_word_prior": float(topic_word_concentration),
    }
    return topics_est, meta


def _kmeans_lloyd_rows(
    x: np.ndarray,
    *,
    k: int,
    n_init: int,
    max_iter: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if n == 0:
        return np.zeros((int(k), int(d)), dtype=np.float64), np.zeros((0,), dtype=np.int64), float("nan")

    kk = int(max(1, k))
    best_inertia = float("inf")
    best_centers = np.zeros((kk, int(d)), dtype=np.float64)
    best_labels = np.zeros((n,), dtype=np.int64)

    for _ in range(int(max(1, n_init))):
        if n >= kk:
            init_ids = rng.choice(np.arange(n, dtype=np.int64), size=kk, replace=False)
        else:
            init_ids = rng.choice(np.arange(n, dtype=np.int64), size=kk, replace=True)
        centers = np.asarray(x[init_ids], dtype=np.float64).copy()
        labels_prev: Optional[np.ndarray] = None

        for _it in range(int(max(1, max_iter))):
            dist2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dist2, axis=1).astype(np.int64, copy=False)
            if labels_prev is not None and np.array_equal(labels, labels_prev):
                break
            labels_prev = labels

            for j in range(kk):
                idx = np.where(labels == j)[0]
                if idx.size == 0:
                    centers[j] = x[int(rng.integers(0, n))]
                else:
                    centers[j] = np.mean(x[idx], axis=0)

        final_dist2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        inertia = float(np.sum(np.min(final_dist2, axis=1)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = np.asarray(centers, dtype=np.float64).copy()
            best_labels = np.argmin(final_dist2, axis=1).astype(np.int64, copy=False)

    return best_centers, best_labels, float(best_inertia)


def _estimate_topics_embedding_spectral_from_docs(
    docs_tokens: Sequence[Sequence[int]],
    *,
    vocab_size: int,
    n_topics: int,
    seed: int,
    svd_dim_extra: int = 4,
    kmeans_inits: int = 8,
    kmeans_max_iter: int = 80,
    assignment_temperature: float = 0.35,
    ppmi_shift: float = 1.0,
) -> Tuple[Tuple[np.ndarray, ...], Dict[str, float]]:
    """
    Embedding-based topic estimator from unlabeled docs.

    Pipeline:
      1) build token co-occurrence from document frequency vectors,
      2) compute shifted PPMI and project words into spectral embedding space,
      3) k-means cluster words and convert cluster affinities into topic-word rows.
    """

    k = int(n_topics)
    v = int(vocab_size)
    if k < 2:
        raise ValueError("n_topics must be >= 2")
    if v < 2:
        raise ValueError("vocab_size must be >= 2")
    if int(svd_dim_extra) < 0:
        raise ValueError("svd_dim_extra must be >= 0")
    if int(kmeans_inits) < 1:
        raise ValueError("kmeans_inits must be >= 1")
    if int(kmeans_max_iter) < 1:
        raise ValueError("kmeans_max_iter must be >= 1")
    temp = float(assignment_temperature)
    if not np.isfinite(temp) or temp <= 0:
        raise ValueError("assignment_temperature must be finite and > 0")
    shift = float(ppmi_shift)
    if not np.isfinite(shift) or shift <= 0:
        raise ValueError("ppmi_shift must be finite and > 0")

    F = _docs_to_frequency_matrix(docs_tokens, vocab_size=v)
    d_docs = int(F.shape[0])
    if d_docs <= 0:
        raise ValueError("need at least one document for embedding_spectral estimation")

    # Word-word co-occurrence from document frequencies.
    C = np.asarray(F.T @ F, dtype=np.float64)
    C = np.maximum(C, 0.0)
    mass = float(np.sum(C))
    if not np.isfinite(mass) or mass <= 0:
        raise RuntimeError("degenerate co-occurrence matrix in embedding_spectral estimator")
    P = C / mass
    p = np.maximum(np.sum(P, axis=1), 1e-12)
    denom = np.maximum(np.outer(p, p), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(np.maximum(P, 1e-12) / denom) - math.log(float(shift))
    ppmi = np.maximum(np.asarray(pmi, dtype=np.float64), 0.0)
    if not np.any(ppmi > 0):
        ppmi = np.asarray(P, dtype=np.float64)

    xc = ppmi - np.mean(ppmi, axis=0, keepdims=True)
    max_rank = int(min(xc.shape[0], xc.shape[1]))
    d = int(max(k, min(max_rank, k + int(svd_dim_extra))))
    if d <= 0:
        raise RuntimeError("invalid embedding_spectral projection dimension")

    u, s, _vt = np.linalg.svd(xc, full_matrices=False)
    if s.size == 0:
        raise RuntimeError("embedding_spectral SVD returned empty spectrum")
    s_use = np.maximum(np.asarray(s[:d], dtype=np.float64), 1e-12)
    x_embed = np.asarray(u[:, :d], dtype=np.float64) * np.sqrt(s_use[None, :])

    rng = np.random.default_rng(int(seed))
    centers, labels, inertia = _kmeans_lloyd_rows(
        x_embed,
        k=int(k),
        n_init=int(kmeans_inits),
        max_iter=int(kmeans_max_iter),
        rng=rng,
    )
    dist2 = np.sum((x_embed[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    logits = -dist2 / float(max(1e-8, temp))
    logits = logits - np.max(logits, axis=1, keepdims=True)
    assign = np.exp(logits)
    assign = assign / np.maximum(np.sum(assign, axis=1, keepdims=True), 1e-12)  # V×K

    # Build topic-word rows by weighting assignment with empirical word mass.
    word_mass = np.maximum(np.sum(F, axis=0), 1e-12)
    topic_word = np.asarray((assign * word_mass[:, None]).T, dtype=np.float64)
    topic_word = _normalize_topic_rows(topic_word)
    topics_est = tuple(np.asarray(topic_word[i], dtype=np.float64).reshape(-1) for i in range(k))

    # Diagnostics.
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=k).astype(np.float64, copy=False)
    active = int(np.sum(counts > 0))
    entropy = -np.sum(assign * np.log(np.maximum(assign, 1e-12)), axis=1)
    meta = {
        "topic_phi_estimator": "embedding_spectral",
        "topic_phi_docs_effective": float(d_docs),
        "embedding_spectral_svd_dim": float(d),
        "embedding_spectral_svd_dim_extra": float(svd_dim_extra),
        "embedding_spectral_kmeans_inits": float(kmeans_inits),
        "embedding_spectral_kmeans_max_iter": float(kmeans_max_iter),
        "embedding_spectral_assignment_temperature": float(temp),
        "embedding_spectral_ppmi_shift": float(shift),
        "embedding_spectral_kmeans_inertia": float(inertia),
        "embedding_spectral_active_clusters": float(active),
        "embedding_spectral_cluster_balance_min": float(np.min(counts)) if counts.size else float("nan"),
        "embedding_spectral_cluster_balance_max": float(np.max(counts)) if counts.size else float("nan"),
        "embedding_spectral_assignment_entropy_mean": float(np.mean(entropy)) if entropy.size else float("nan"),
    }
    return topics_est, meta


def _normalize_topic_rows(x: np.ndarray) -> np.ndarray:
    rows = np.maximum(np.asarray(x, dtype=np.float64), 1e-12)
    s = np.sum(rows, axis=1, keepdims=True)
    s = np.maximum(s, 1e-12)
    return (rows / s).astype(np.float64, copy=False)


def _align_est_topics_to_true_order(
    topics_est: Sequence[np.ndarray],
    *,
    perm_est_to_true: Sequence[int],
) -> np.ndarray:
    """Return estimated topics re-indexed into true-topic order."""
    est = np.stack([np.asarray(t, dtype=np.float64).reshape(-1) for t in topics_est], axis=0)
    inv = _invert_perm(tuple(int(x) for x in perm_est_to_true))
    return np.asarray(est[np.asarray(inv, dtype=np.int64)], dtype=np.float64)


def _softmax_rows(x: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / float(max(1e-8, temperature))
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    d = np.maximum(np.sum(e, axis=1, keepdims=True), 1e-12)
    return np.asarray(e / d, dtype=np.float64)


def _topic_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    an = np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bn = np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return np.asarray((a / an) @ (b / bn).T, dtype=np.float64)


def _kernel_ridge_topic_map(
    x_seed: np.ndarray,
    y_seed: np.ndarray,
    x_all: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    """Linear-kernel ridge map from seed estimated topics to seed true topics."""
    xs = np.asarray(x_seed, dtype=np.float64)
    ys = np.asarray(y_seed, dtype=np.float64)
    xa = np.asarray(x_all, dtype=np.float64)
    n_seed = int(xs.shape[0])
    if n_seed <= 0:
        return np.asarray(xa, dtype=np.float64)
    K = xs @ xs.T
    A = K + float(max(1e-12, ridge)) * np.eye(n_seed, dtype=np.float64)
    alpha = np.linalg.solve(A, ys)  # [n_seed, V]
    K_all = xa @ xs.T  # [K, n_seed]
    return np.asarray(K_all @ alpha, dtype=np.float64)


def _train_topic_operator_network(
    *,
    mode: str,
    x_seed: np.ndarray,
    y_seed: np.ndarray,
    x_all: np.ndarray,
    hidden_dim: int,
    steps: int,
    lr: float,
    weight_decay: float,
    mix_samples: int,
    mix_temperature: float,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Fit a lightweight CPU neural operator that maps estimated topic vectors to refined vectors.

    Falls back to kernel ridge when torch is unavailable.
    """
    xs = np.asarray(x_seed, dtype=np.float64)
    ys = np.asarray(y_seed, dtype=np.float64)
    xa = np.asarray(x_all, dtype=np.float64)
    n_seed = int(xs.shape[0])
    if n_seed <= 0:
        return np.asarray(xa, dtype=np.float64), {"backend": "identity", "train_loss": float("nan")}

    try:
        import torch
        import torch.nn as nn
    except Exception:
        pred = _kernel_ridge_topic_map(xs, ys, xa, ridge=1e-3)
        return pred, {"backend": "kernel_ridge_fallback", "train_loss": float("nan")}

    rng = np.random.default_rng(int(seed))
    x_rows = [xs]
    y_rows = [ys]

    n_mix = int(max(0, mix_samples))
    if n_seed >= 2 and n_mix > 0:
        alpha = np.full((n_seed,), float(max(1e-3, mix_temperature)), dtype=np.float64)
        mix_w = rng.dirichlet(alpha, size=n_mix).astype(np.float64, copy=False)
        x_mix = mix_w @ xs
        y_mix = mix_w @ ys
        x_mix = _normalize_topic_rows(np.maximum(x_mix + rng.normal(0.0, 0.002, size=x_mix.shape), 1e-12))
        y_mix = _normalize_topic_rows(np.maximum(y_mix + rng.normal(0.0, 0.001, size=y_mix.shape), 1e-12))
        x_rows.append(x_mix)
        y_rows.append(y_mix)

    x_train = _normalize_topic_rows(np.vstack(x_rows))
    y_train = _normalize_topic_rows(np.vstack(y_rows))

    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    x_all_t = torch.tensor(_normalize_topic_rows(xa), dtype=torch.float32)

    v = int(x_train.shape[1])
    h = int(max(8, hidden_dim))
    mode_l = str(mode).strip().lower()
    if mode_l == "ctreepo":
        class _CTreePONet(nn.Module):
            def __init__(self, dim: int, hid: int):
                super().__init__()
                self.gate = nn.Linear(dim, dim)
                self.res = nn.Sequential(
                    nn.Linear(dim, hid),
                    nn.ReLU(),
                    nn.Linear(hid, dim),
                )

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                g = torch.sigmoid(self.gate(z))
                out = g * z + (1.0 - g) * self.res(z)
                return torch.softmax(out, dim=-1)

        model: nn.Module = _CTreePONet(v, h)
    else:
        class _MergeableNet(nn.Module):
            def __init__(self, dim: int, hid: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, hid),
                    nn.ReLU(),
                    nn.Linear(hid, hid),
                    nn.ReLU(),
                    nn.Linear(hid, dim),
                )

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                return torch.softmax(self.net(z), dim=-1)

        model = _MergeableNet(v, h)

    torch.manual_seed(int(seed))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(max(1e-5, lr)),
        weight_decay=float(max(0.0, weight_decay)),
    )
    last_loss = float("nan")
    n_steps = int(max(1, steps))
    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        pred = model(x_t)
        loss = torch.mean((pred - y_t) ** 2)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu().item())

    with torch.no_grad():
        out = model(x_all_t).detach().cpu().numpy().astype(np.float64, copy=False)

    return out, {"backend": "torch_cpu", "train_loss": float(last_loss)}


def _neural_refine_topics(
    topics_est: Sequence[np.ndarray],
    topics_true: Sequence[np.ndarray],
    *,
    perm_est_to_true: Sequence[int],
    mode: str,
    seed: int,
    seed_fraction: float,
    hidden_dim: int,
    steps: int,
    lr: float,
    weight_decay: float,
    mix_samples: int,
    mix_temperature: float,
    operator_boost: float,
    seed_min_weight: float,
    seed_max_weight: float,
    similarity_temperature: float,
    ridge: float,
) -> Tuple[Tuple[np.ndarray, ...], Dict[str, Any]]:
    """
    Refine topic-word estimates using lightweight CPU neural operators.

    Modes:
      - ctreepo: residual gated refiner.
      - mergeable_sketch: MLP refiner.
      - hybrid: oracle-seeded ensemble of base + ctreepo + mergeable refiners.
    """
    true_rows = _normalize_topic_rows(np.stack([np.asarray(t, dtype=np.float64).reshape(-1) for t in topics_true], axis=0))
    est_aligned = _normalize_topic_rows(
        _align_est_topics_to_true_order(topics_est, perm_est_to_true=tuple(int(x) for x in perm_est_to_true))
    )
    k = int(true_rows.shape[0])
    rng = np.random.default_rng(int(seed))

    frac = float(seed_fraction)
    frac = min(1.0, max(1.0 / float(max(1, k)), frac))
    n_seed = int(max(1, min(k, round(frac * k))))
    seed_idx = np.asarray(sorted(int(x) for x in rng.choice(np.arange(k, dtype=np.int64), size=n_seed, replace=False).tolist()), dtype=np.int64)

    seed_est = est_aligned[seed_idx]
    seed_true = true_rows[seed_idx]

    # CTreePO-style local correction from seed anchors.
    sim = _topic_cosine_similarity(est_aligned, seed_est)
    w_anchor = _softmax_rows(sim, temperature=float(max(1e-5, similarity_temperature)))
    est_anchor = w_anchor @ seed_est
    true_anchor = w_anchor @ seed_true
    pred_ct = est_aligned + float(max(0.0, operator_boost)) * (true_anchor - est_anchor)
    pred_ct[seed_idx] = seed_true
    pred_ct = _normalize_topic_rows(pred_ct)

    # Mergeable-sketch-style learned map from seeds.
    pred_ms_nn, ms_meta = _train_topic_operator_network(
        mode="mergeable_sketch",
        x_seed=seed_est,
        y_seed=seed_true,
        x_all=est_aligned,
        hidden_dim=int(hidden_dim),
        steps=int(steps),
        lr=float(lr),
        weight_decay=float(weight_decay),
        mix_samples=int(mix_samples),
        mix_temperature=float(mix_temperature),
        seed=int(_splitmix64(int(seed) + 991) & 0xFFFFFFFF),
    )
    pred_ms = _normalize_topic_rows(pred_ms_nn)
    if not np.all(np.isfinite(pred_ms)):
        pred_ms = _normalize_topic_rows(_kernel_ridge_topic_map(seed_est, seed_true, est_aligned, ridge=float(ridge)))
        ms_meta = {"backend": "kernel_ridge_numeric_fallback", "train_loss": float("nan")}
    pred_ms[seed_idx] = seed_true
    pred_ms = _normalize_topic_rows(pred_ms)

    mode_l = str(mode).strip().lower()
    if mode_l == "ctreepo":
        pred = pred_ct
    elif mode_l == "mergeable_sketch":
        pred = pred_ms
    elif mode_l == "hybrid":
        coverage = float(n_seed) / float(max(1, k))
        w_base = float(np.clip(
            float(seed_max_weight) - (float(seed_max_weight) - float(seed_min_weight)) * coverage,
            float(seed_min_weight),
            float(seed_max_weight),
        ))
        op_boost = float(max(1e-8, operator_boost))
        raw_ct = 0.5 * op_boost
        raw_ms = 0.5 * op_boost
        scale = (1.0 - w_base) / max(1e-8, raw_ct + raw_ms)
        w_ct = raw_ct * scale
        w_ms = raw_ms * scale
        pred = (w_base * est_aligned) + (w_ct * pred_ct) + (w_ms * pred_ms)
        pred[seed_idx] = seed_true
        pred = _normalize_topic_rows(pred)
    else:
        raise ValueError(f"unsupported neural refine mode: {mode!r}")

    meta: Dict[str, Any] = {
        "topic_phi_estimator": f"neural_{mode_l}",
        "topic_phi_neural_seed_fraction": float(frac),
        "topic_phi_neural_seed_count": int(n_seed),
        "topic_phi_neural_seed_indices": [int(x) for x in seed_idx.tolist()],
        "topic_phi_neural_operator_boost": float(operator_boost),
        "topic_phi_neural_similarity_temperature": float(similarity_temperature),
        "topic_phi_neural_hidden_dim": int(hidden_dim),
        "topic_phi_neural_steps": int(steps),
        "topic_phi_neural_lr": float(lr),
        "topic_phi_neural_weight_decay": float(weight_decay),
        "topic_phi_neural_mix_samples": int(mix_samples),
        "topic_phi_neural_mix_temperature": float(mix_temperature),
        "topic_phi_neural_ridge": float(ridge),
        "topic_phi_neural_mergeable_backend": str(ms_meta.get("backend", "unknown")),
        "topic_phi_neural_mergeable_train_loss": float(ms_meta.get("train_loss", float("nan"))),
    }
    return (tuple(np.asarray(row, dtype=np.float64).reshape(-1) for row in pred), meta)


def estimate_topic_distributions(
    topics_true: Sequence[np.ndarray],
    *,
    estimator: TopicPhiEstimatorName,
    n_docs: int,
    doc_topic_concentration: float,
    tlda_delta: float,
    tlda_rate_constant: float,
    sigmaK_floor: float,
    permute: bool,
    seed: int,
    topic_word_concentration: float = 0.1,
    docs_tokens: Optional[Sequence[Sequence[int]]] = None,
    sklearn_lda_max_iter: int = 60,
    online_burn_in_docs: int = 0,
    online_batch_docs: int = 32,
    online_passes: int = 1,
    online_lr: float = 0.1,
    online_grad_clip_norm: float = 1.0,
    neural_base_estimator: str = "tensor_lda",
    neural_seed_fraction: float = 0.35,
    neural_hidden_dim: int = 48,
    neural_steps: int = 60,
    neural_lr: float = 3e-3,
    neural_weight_decay: float = 1e-4,
    neural_mix_samples: int = 128,
    neural_mix_temperature: float = 1.0,
    neural_operator_boost: float = 1.4,
    neural_seed_min_weight: float = 0.2,
    neural_seed_max_weight: float = 0.55,
    neural_similarity_temperature: float = 0.15,
    neural_ridge: float = 1e-3,
    embedding_svd_dim_extra: int = 4,
    embedding_kmeans_inits: int = 8,
    embedding_kmeans_max_iter: int = 80,
    embedding_assignment_temperature: float = 0.35,
    embedding_ppmi_shift: float = 1.0,
) -> Tuple[Tuple[np.ndarray, ...], Dict[str, object], Tuple[int, ...]]:
    """
    Return (topics_est, meta, perm_est_to_true).

    - estimator="true": φ̂ = φ (perm is identity).
    - estimator="noisy_theory": add bounded noise at the Theorem-5.1-like rate O(1/sqrt(n_docs)).
    - estimator="tensor_lda": estimate φ̂ from docs via centered moments + whitening + tensor power method.
    - estimator="online_tensor_lda": same moments, but update tensor factors online via STGD-style steps.
    - estimator="sklearn_lda": fit scikit-learn's variational Bayes LDA on the DTM.
    - estimator="embedding_spectral": estimate φ̂ from word embeddings induced by shifted PPMI + SVD + k-means.
    - estimator in {"neural_ctreepo","neural_mergeable_sketch","neural_hybrid"}:
      run a base estimator, then apply a lightweight CPU neural-operator refiner
      using oracle-seeded topics and return the refined φ̂.
    - estimator="neural_embedding_hybrid": alias for neural_hybrid with embedding_spectral base estimator.
    """

    mode = str(estimator).strip().lower()
    if mode not in VALID_TOPIC_PHI_ESTIMATORS:
        raise ValueError(f"unsupported topic_phi_estimator: {estimator!r}; expected {VALID_TOPIC_PHI_ESTIMATORS}")

    k = int(len(topics_true))
    if k <= 0:
        raise ValueError("need at least one topic")
    v = int(np.asarray(topics_true[0]).reshape(-1).size)

    topics = tuple(np.asarray(t, dtype=np.float64).reshape(-1) for t in topics_true)
    if any(int(np.asarray(t).reshape(-1).size) != v for t in topics):
        raise ValueError("topic vectors must share vocabulary size")

    if mode == "neural_embedding_hybrid":
        aliased_topics, aliased_meta_raw, aliased_perm = estimate_topic_distributions(
            topics,
            estimator="neural_hybrid",
            n_docs=int(n_docs),
            doc_topic_concentration=float(doc_topic_concentration),
            tlda_delta=float(tlda_delta),
            tlda_rate_constant=float(tlda_rate_constant),
            sigmaK_floor=float(sigmaK_floor),
            permute=bool(permute),
            seed=int(seed),
            topic_word_concentration=float(topic_word_concentration),
            docs_tokens=docs_tokens,
            sklearn_lda_max_iter=int(sklearn_lda_max_iter),
            online_burn_in_docs=int(online_burn_in_docs),
            online_batch_docs=int(online_batch_docs),
            online_passes=int(online_passes),
            online_lr=float(online_lr),
            online_grad_clip_norm=float(online_grad_clip_norm),
            neural_base_estimator="embedding_spectral",
            neural_seed_fraction=float(neural_seed_fraction),
            neural_hidden_dim=int(neural_hidden_dim),
            neural_steps=int(neural_steps),
            neural_lr=float(neural_lr),
            neural_weight_decay=float(neural_weight_decay),
            neural_mix_samples=int(neural_mix_samples),
            neural_mix_temperature=float(neural_mix_temperature),
            neural_operator_boost=float(neural_operator_boost),
            neural_seed_min_weight=float(neural_seed_min_weight),
            neural_seed_max_weight=float(neural_seed_max_weight),
            neural_similarity_temperature=float(neural_similarity_temperature),
            neural_ridge=float(neural_ridge),
            embedding_svd_dim_extra=int(embedding_svd_dim_extra),
            embedding_kmeans_inits=int(embedding_kmeans_inits),
            embedding_kmeans_max_iter=int(embedding_kmeans_max_iter),
            embedding_assignment_temperature=float(embedding_assignment_temperature),
            embedding_ppmi_shift=float(embedding_ppmi_shift),
        )
        aliased_meta: Dict[str, object] = dict(aliased_meta_raw)
        aliased_meta["topic_phi_estimator"] = "neural_embedding_hybrid"
        aliased_meta["topic_phi_neural_base_estimator"] = "embedding_spectral"
        return aliased_topics, aliased_meta, aliased_perm

    if mode in NEURAL_TOPIC_PHI_ESTIMATORS:
        base_mode = str(neural_base_estimator).strip().lower()
        if base_mode.startswith("neural_"):
            raise ValueError(
                f"neural_base_estimator={neural_base_estimator!r} cannot itself be neural; "
                f"expected one of {[m for m in VALID_TOPIC_PHI_ESTIMATORS if not str(m).startswith('neural_')]}"
            )
        if base_mode not in VALID_TOPIC_PHI_ESTIMATORS:
            raise ValueError(
                f"unsupported neural_base_estimator={neural_base_estimator!r}; expected {VALID_TOPIC_PHI_ESTIMATORS}"
            )
        base_topics, base_meta, base_perm = estimate_topic_distributions(
            topics,
            estimator=base_mode,
            n_docs=int(n_docs),
            doc_topic_concentration=float(doc_topic_concentration),
            tlda_delta=float(tlda_delta),
            tlda_rate_constant=float(tlda_rate_constant),
            sigmaK_floor=float(sigmaK_floor),
            permute=bool(permute),
            seed=int(seed),
            topic_word_concentration=float(topic_word_concentration),
            docs_tokens=docs_tokens,
            sklearn_lda_max_iter=int(sklearn_lda_max_iter),
            online_burn_in_docs=int(online_burn_in_docs),
            online_batch_docs=int(online_batch_docs),
            online_passes=int(online_passes),
            online_lr=float(online_lr),
            online_grad_clip_norm=float(online_grad_clip_norm),
            embedding_svd_dim_extra=int(embedding_svd_dim_extra),
            embedding_kmeans_inits=int(embedding_kmeans_inits),
            embedding_kmeans_max_iter=int(embedding_kmeans_max_iter),
            embedding_assignment_temperature=float(embedding_assignment_temperature),
            embedding_ppmi_shift=float(embedding_ppmi_shift),
        )
        neural_mode = mode.replace("neural_", "", 1)
        refined_topics, refine_meta = _neural_refine_topics(
            base_topics,
            topics,
            perm_est_to_true=base_perm,
            mode=str(neural_mode),
            seed=int(seed),
            seed_fraction=float(neural_seed_fraction),
            hidden_dim=int(neural_hidden_dim),
            steps=int(neural_steps),
            lr=float(neural_lr),
            weight_decay=float(neural_weight_decay),
            mix_samples=int(neural_mix_samples),
            mix_temperature=float(neural_mix_temperature),
            operator_boost=float(neural_operator_boost),
            seed_min_weight=float(neural_seed_min_weight),
            seed_max_weight=float(neural_seed_max_weight),
            similarity_temperature=float(neural_similarity_temperature),
            ridge=float(neural_ridge),
        )

        O = np.stack(list(topics), axis=0).T  # V × K
        svals = np.linalg.svd(O, full_matrices=False, compute_uv=False).astype(np.float64, copy=False)
        sigmaK = float(np.min(svals)) if svals.size else 0.0
        sigmaK = max(float(sigmaK_floor), float(sigmaK))
        alpha0 = float(k) * float(doc_topic_concentration)
        pmin = 1.0 / float(k)

        perm_est_to_true, cost = _best_topic_permutation_l2(refined_topics, topics)
        aligned_err = np.asarray([float(cost[i, perm_est_to_true[i]]) for i in range(k)], dtype=np.float64)

        meta: Dict[str, object] = {
            "topic_phi_estimator": str(mode),
            "topic_phi_docs_effective": float(max(0, int(n_docs))),
            "topic_phi_sigmaK": float(sigmaK),
            "topic_phi_pmin": float(pmin),
            "topic_phi_alpha0": float(alpha0),
            "topic_phi_delta": float(tlda_delta),
            "topic_phi_l2_error_mean": float(np.mean(aligned_err)) if aligned_err.size else 0.0,
            "topic_phi_l2_error_p95": float(np.percentile(aligned_err, 95.0)) if aligned_err.size else 0.0,
            "topic_phi_l2_error_max": float(np.max(aligned_err)) if aligned_err.size else 0.0,
            "topic_phi_neural_base_estimator": str(base_mode),
        }
        for key, value in base_meta.items():
            meta[f"topic_phi_neural_base_{key}"] = value
        for key, value in refine_meta.items():
            meta[str(key)] = value
        return (tuple(refined_topics), meta, perm_est_to_true)

    # Identity baseline.
    if mode == "true":
        perm = tuple(range(k))
        meta = {
            "topic_phi_estimator": "true",
            "topic_phi_docs_effective": float(max(0, int(n_docs))),
            "topic_phi_eps_bound": 0.0,
            "topic_phi_sigmaK": float("nan"),
            "topic_phi_pmin": float("nan"),
            "topic_phi_alpha0": float("nan"),
            "topic_phi_delta": float(tlda_delta),
        }
        return topics, meta, perm

    if mode == "sklearn_lda":
        if docs_tokens is None:
            raise ValueError("sklearn_lda requires docs_tokens")
        docs_list = list(docs_tokens)
        if int(n_docs) > 0:
            docs_list = docs_list[: int(n_docs)]
        if not docs_list:
            raise ValueError("sklearn_lda requires at least one document")
        topics_est, algo_meta = _estimate_topics_sklearn_lda_from_docs(
            docs_list,
            vocab_size=int(v),
            n_topics=int(k),
            doc_topic_concentration=float(doc_topic_concentration),
            topic_word_concentration=float(topic_word_concentration),
            seed=int(seed),
            max_iter=int(sklearn_lda_max_iter),
        )
        if bool(permute):
            rng = np.random.default_rng(int(seed))
            perm_idx = rng.permutation(k)
            topics_est = tuple(topics_est[int(i)] for i in perm_idx.tolist())

        O = np.stack(list(topics), axis=0).T  # V × K
        svals = np.linalg.svd(O, full_matrices=False, compute_uv=False).astype(np.float64, copy=False)
        sigmaK = float(np.min(svals)) if svals.size else 0.0
        sigmaK = max(float(sigmaK_floor), float(sigmaK))
        alpha0 = float(k) * float(doc_topic_concentration)
        pmin = 1.0 / float(k)

        perm_est_to_true, cost = _best_topic_permutation_l2(topics_est, topics)
        aligned_err = np.asarray([float(cost[i, perm_est_to_true[i]]) for i in range(k)], dtype=np.float64)

        meta = {
            **algo_meta,
            "topic_phi_sigmaK": float(sigmaK),
            "topic_phi_pmin": float(pmin),
            "topic_phi_alpha0": float(alpha0),
            "topic_phi_delta": float(tlda_delta),
            "topic_phi_l2_error_mean": float(np.mean(aligned_err)) if aligned_err.size else 0.0,
            "topic_phi_l2_error_p95": float(np.percentile(aligned_err, 95.0)) if aligned_err.size else 0.0,
            "topic_phi_l2_error_max": float(np.max(aligned_err)) if aligned_err.size else 0.0,
        }
        return (tuple(topics_est), meta, perm_est_to_true)

    # Spectral/Tensor-LDA estimator from a corpus.
    if mode == "tensor_lda":
        if docs_tokens is None:
            raise ValueError("tensor_lda requires docs_tokens")
        docs_list = list(docs_tokens)
        if int(n_docs) > 0:
            docs_list = docs_list[: int(n_docs)]
        if not docs_list:
            raise ValueError("tensor_lda requires at least one document")
        topics_est, algo_meta = _estimate_topics_tensor_lda_from_docs(
            docs_list,
            vocab_size=int(v),
            n_topics=int(k),
            doc_topic_concentration=float(doc_topic_concentration),
            seed=int(seed),
        )
        if bool(permute):
            rng = np.random.default_rng(int(seed))
            perm_idx = rng.permutation(k)
            topics_est = tuple(topics_est[int(i)] for i in perm_idx.tolist())

        O = np.stack(list(topics), axis=0).T  # V × K
        svals = np.linalg.svd(O, full_matrices=False, compute_uv=False).astype(np.float64, copy=False)
        sigmaK = float(np.min(svals)) if svals.size else 0.0
        sigmaK = max(float(sigmaK_floor), float(sigmaK))
        alpha0 = float(k) * float(doc_topic_concentration)
        pmin = 1.0 / float(k)

        perm_est_to_true, cost = _best_topic_permutation_l2(topics_est, topics)
        aligned_err = np.asarray([float(cost[i, perm_est_to_true[i]]) for i in range(k)], dtype=np.float64)

        meta = {
            **algo_meta,
            "topic_phi_sigmaK": float(sigmaK),
            "topic_phi_pmin": float(pmin),
            "topic_phi_alpha0": float(alpha0),
            "topic_phi_delta": float(tlda_delta),
            "topic_phi_l2_error_mean": float(np.mean(aligned_err)) if aligned_err.size else 0.0,
            "topic_phi_l2_error_p95": float(np.percentile(aligned_err, 95.0)) if aligned_err.size else 0.0,
            "topic_phi_l2_error_max": float(np.max(aligned_err)) if aligned_err.size else 0.0,
        }
        return (tuple(topics_est), meta, perm_est_to_true)

    if mode == "online_tensor_lda":
        if docs_tokens is None:
            raise ValueError("online_tensor_lda requires docs_tokens")
        docs_list = list(docs_tokens)
        if int(n_docs) > 0:
            docs_list = docs_list[: int(n_docs)]
        if not docs_list:
            raise ValueError("online_tensor_lda requires at least one document")

        topics_est, algo_meta = _estimate_topics_online_tensor_lda_from_docs(
            docs_list,
            vocab_size=int(v),
            n_topics=int(k),
            doc_topic_concentration=float(doc_topic_concentration),
            seed=int(seed),
            burn_in_docs=int(online_burn_in_docs),
            batch_docs=int(online_batch_docs),
            passes=int(online_passes),
            lr=float(online_lr),
            grad_clip_norm=float(online_grad_clip_norm),
        )
        if bool(permute):
            rng = np.random.default_rng(int(seed))
            perm_idx = rng.permutation(k)
            topics_est = tuple(topics_est[int(i)] for i in perm_idx.tolist())

        O = np.stack(list(topics), axis=0).T  # V × K
        svals = np.linalg.svd(O, full_matrices=False, compute_uv=False).astype(np.float64, copy=False)
        sigmaK = float(np.min(svals)) if svals.size else 0.0
        sigmaK = max(float(sigmaK_floor), float(sigmaK))
        alpha0 = float(k) * float(doc_topic_concentration)
        pmin = 1.0 / float(k)

        perm_est_to_true, cost = _best_topic_permutation_l2(topics_est, topics)
        aligned_err = np.asarray([float(cost[i, perm_est_to_true[i]]) for i in range(k)], dtype=np.float64)

        meta = {
            **algo_meta,
            "topic_phi_sigmaK": float(sigmaK),
            "topic_phi_pmin": float(pmin),
            "topic_phi_alpha0": float(alpha0),
            "topic_phi_delta": float(tlda_delta),
            "topic_phi_l2_error_mean": float(np.mean(aligned_err)) if aligned_err.size else 0.0,
            "topic_phi_l2_error_p95": float(np.percentile(aligned_err, 95.0)) if aligned_err.size else 0.0,
            "topic_phi_l2_error_max": float(np.max(aligned_err)) if aligned_err.size else 0.0,
        }
        return (tuple(topics_est), meta, perm_est_to_true)

    if mode == "embedding_spectral":
        if docs_tokens is None:
            raise ValueError("embedding_spectral requires docs_tokens")
        docs_list = list(docs_tokens)
        if int(n_docs) > 0:
            docs_list = docs_list[: int(n_docs)]
        if not docs_list:
            raise ValueError("embedding_spectral requires at least one document")

        topics_est, algo_meta = _estimate_topics_embedding_spectral_from_docs(
            docs_list,
            vocab_size=int(v),
            n_topics=int(k),
            seed=int(seed),
            svd_dim_extra=int(embedding_svd_dim_extra),
            kmeans_inits=int(embedding_kmeans_inits),
            kmeans_max_iter=int(embedding_kmeans_max_iter),
            assignment_temperature=float(embedding_assignment_temperature),
            ppmi_shift=float(embedding_ppmi_shift),
        )
        if bool(permute):
            rng = np.random.default_rng(int(seed))
            perm_idx = rng.permutation(k)
            topics_est = tuple(topics_est[int(i)] for i in perm_idx.tolist())

        O = np.stack(list(topics), axis=0).T  # V × K
        svals = np.linalg.svd(O, full_matrices=False, compute_uv=False).astype(np.float64, copy=False)
        sigmaK = float(np.min(svals)) if svals.size else 0.0
        sigmaK = max(float(sigmaK_floor), float(sigmaK))
        alpha0 = float(k) * float(doc_topic_concentration)
        pmin = 1.0 / float(k)

        perm_est_to_true, cost = _best_topic_permutation_l2(topics_est, topics)
        aligned_err = np.asarray([float(cost[i, perm_est_to_true[i]]) for i in range(k)], dtype=np.float64)

        meta = {
            **algo_meta,
            "topic_phi_sigmaK": float(sigmaK),
            "topic_phi_pmin": float(pmin),
            "topic_phi_alpha0": float(alpha0),
            "topic_phi_delta": float(tlda_delta),
            "topic_phi_l2_error_mean": float(np.mean(aligned_err)) if aligned_err.size else 0.0,
            "topic_phi_l2_error_p95": float(np.percentile(aligned_err, 95.0)) if aligned_err.size else 0.0,
            "topic_phi_l2_error_max": float(np.max(aligned_err)) if aligned_err.size else 0.0,
        }
        return (tuple(topics_est), meta, perm_est_to_true)

    # Noisy-theory estimator: calibrate eps via a Theorem-5.1-like expression.
    N = int(n_docs)
    if N <= 0:
        raise ValueError("topic_phi_docs_effective must be positive for noisy topic estimation")

    rng = np.random.default_rng(int(seed))
    O = np.stack(list(topics), axis=0).T  # V × K
    svals = np.linalg.svd(O, full_matrices=False, compute_uv=False).astype(np.float64, copy=False)
    sigmaK = float(np.min(svals)) if svals.size else 0.0
    sigmaK = max(float(sigmaK_floor), float(sigmaK))

    alpha0 = float(k) * float(doc_topic_concentration)
    pmin = 1.0 / float(k)
    eps = _theorem51_l2_error_bound(
        c=float(tlda_rate_constant),
        alpha0=float(alpha0),
        pmin=float(pmin),
        sigmaK=float(sigmaK),
        k=int(k),
        delta=float(tlda_delta),
        N=int(N),
    )

    noisy: List[np.ndarray] = []
    for t in topics:
        g = rng.normal(loc=0.0, scale=1.0, size=(v,)).astype(np.float64, copy=False)
        gn = float(np.linalg.norm(g))
        if gn > 0 and np.isfinite(eps):
            g = (float(eps) / gn) * g
        out = np.clip(t + g, 1e-12, None)
        out = out / float(np.sum(out))
        noisy.append(out.astype(np.float64, copy=False))

    # Optional random permutation to simulate unidentifiability.
    if bool(permute):
        perm_idx = rng.permutation(k)
        noisy = [noisy[int(i)] for i in perm_idx.tolist()]

    # Compute best alignment back to truth for metrics.
    perm_est_to_true, cost = _best_topic_permutation_l2(noisy, topics)
    aligned_err = np.asarray([float(cost[i, perm_est_to_true[i]]) for i in range(k)], dtype=np.float64)

    meta = {
        "topic_phi_estimator": "noisy_theory",
        "topic_phi_docs_effective": float(N),
        "topic_phi_eps_bound": float(eps),
        "topic_phi_sigmaK": float(sigmaK),
        "topic_phi_pmin": float(pmin),
        "topic_phi_alpha0": float(alpha0),
        "topic_phi_delta": float(tlda_delta),
        "topic_phi_l2_error_mean": float(np.mean(aligned_err)) if aligned_err.size else 0.0,
        "topic_phi_l2_error_p95": float(np.percentile(aligned_err, 95.0)) if aligned_err.size else 0.0,
        "topic_phi_l2_error_max": float(np.max(aligned_err)) if aligned_err.size else 0.0,
    }
    return (tuple(noisy), meta, perm_est_to_true)


def _sample_segment_lengths_in_leaves(
    *,
    n_leaves: int,
    n_segments: int,
    min_seg_leaves: int,
    max_seg_leaves: int,
    boundary_profile_fn: Optional[Callable[[float], float]],
    length_power: float,
    rng: np.random.Generator,
) -> List[int]:
    lp = float(length_power)
    if not np.isfinite(lp):
        raise ValueError("segment_length_power must be finite")
    s = int(max(1, n_segments))
    s = max(1, min(int(n_leaves), s))
    remaining = int(n_leaves)
    seg_lens: List[int] = []
    for seg_idx in range(s):
        if seg_idx == s - 1:
            seg_lens.append(remaining)
            break
        min_needed = int(min_seg_leaves) * (s - seg_idx - 1)
        hi = min(int(max_seg_leaves), remaining - min_needed)
        lo = int(min_seg_leaves)
        hi = max(lo, hi)
        cand = np.arange(int(lo), int(hi) + 1, dtype=np.int64)
        w = np.ones_like(cand, dtype=np.float64)
        if lp != 0.0:
            w *= cand.astype(np.float64) ** lp
        if boundary_profile_fn is not None:
            boundary_idx = (int(n_leaves) - int(remaining)) + cand
            # Normalize boundary position to ~[0,1] over *internal* leaf boundaries.
            t = boundary_idx.astype(np.float64) / float(max(1, int(n_leaves) - 1))
            w_pos = np.asarray([float(boundary_profile_fn(float(tt))) for tt in t.tolist()], dtype=np.float64)
            w *= np.maximum(w_pos, 1e-12)
        w = np.maximum(w, 1e-12)
        total = float(np.sum(w))
        probs = (w / total) if total > 0 else None
        seg_len = int(rng.choice(cand, p=probs))
        seg_lens.append(seg_len)
        remaining -= seg_len

    if sum(seg_lens) != n_leaves or any(x <= 0 for x in seg_lens):
        return [n_leaves]
    return seg_lens


def generate_segment_lda_docs(
    n_docs: int,
    *,
    topics: Sequence[np.ndarray],
    min_tokens: int,
    max_tokens: int,
    min_segments: int,
    max_segments: int,
    min_seg_len: int,
    max_seg_len: int,
    leaf_tokens: int,
    align_segments_to_leaves: bool,
    doc_topic_concentration: float,
    topic_process: TopicProcessName,
    boundary_profile: BoundaryProfileName,
    boundary_profile_strength: float,
    boundary_profile_seed: int,
    segment_length_power: float = 0.0,
    seed: int,
) -> Tuple[Tuple[SegmentLDADoc, ...], Dict[str, float]]:
    """
    Generate docs with a piecewise-constant topic id, and LDA-style word emissions.

    The returned `doc.topics` is the *true* topic id per token (latent in principle).
    """

    rng = np.random.default_rng(int(seed))
    profile_rng = np.random.default_rng(int(boundary_profile_seed))
    bstrength = float(boundary_profile_strength)
    boundary_profile_fn = (
        make_boundary_profile(
            str(boundary_profile),
            strength=bstrength,
            rng=profile_rng,
        )
        if bstrength > 0.0
        else None
    )
    if int(min_tokens) < 2 or int(max_tokens) < int(min_tokens):
        raise ValueError("require 2 <= min_tokens <= max_tokens")
    if int(min_segments) < 1 or int(max_segments) < int(min_segments):
        raise ValueError("require 1 <= min_segments <= max_segments")
    if int(min_seg_len) < 1 or int(max_seg_len) < int(min_seg_len):
        raise ValueError("require 1 <= min_seg_len <= max_seg_len")
    if int(leaf_tokens) <= 0:
        raise ValueError("leaf_tokens must be positive")
    if float(doc_topic_concentration) <= 0:
        raise ValueError("doc_topic_concentration must be positive")
    if len(topics) < 2:
        raise ValueError("need at least 2 topics")

    k = int(len(topics))
    v = int(topics[0].shape[0])

    leaf = int(max(1, leaf_tokens))
    min_seg_leaves = int(max(1, math.ceil(float(min_seg_len) / float(leaf))))
    max_seg_leaves = int(max(1, math.floor(float(max_seg_len) / float(leaf))))
    max_seg_leaves = max(min_seg_leaves, max_seg_leaves)

    process = str(topic_process).strip().lower()
    if process not in VALID_TOPIC_PROCESSES:
        raise ValueError(f"unsupported topic_process: {topic_process!r}; expected one of {VALID_TOPIC_PROCESSES}")

    docs: List[SegmentLDADoc] = []
    segment_stats: List[int] = []
    segment_len_leaves_all: List[int] = []
    changepoints: List[int] = []
    changepoint_rates: List[float] = []
    leaf_purities: List[float] = []
    for _ in range(int(n_docs)):
        n = int(rng.integers(int(min_tokens), int(max_tokens) + 1))
        leaves = build_leaf_spans(n, leaf_tokens=leaf)
        n_leaves = len(leaves)
        if n_leaves == 0:
            docs.append(SegmentLDADoc(tokens=tuple(), topics=tuple()))
            segment_stats.append(0)
            continue

        pi = rng.dirichlet(np.full((k,), float(doc_topic_concentration), dtype=np.float64))

        tokens_out: List[int] = []
        topics_out: List[int] = []
        if process == "bag_of_words":
            for _tok in range(int(n)):
                topic_id = int(rng.choice(k, p=pi))
                w = int(rng.choice(v, p=topics[topic_id]))
                tokens_out.append(w)
                topics_out.append(topic_id)
            segment_stats.append(0)
        else:
            if bool(align_segments_to_leaves):
                s = int(rng.integers(int(min_segments), int(max_segments) + 1))
                s = max(1, min(int(n_leaves), s))
                seg_lens_leaves = _sample_segment_lengths_in_leaves(
                    n_leaves=n_leaves,
                    n_segments=s,
                    min_seg_leaves=min_seg_leaves,
                    max_seg_leaves=max_seg_leaves,
                    boundary_profile_fn=boundary_profile_fn,
                    length_power=float(segment_length_power),
                    rng=rng,
                )
            else:
                # If segments may split leaves, approximate by sampling token-length segments directly
                # and then projecting to leaves (kept simple for now).
                s = int(rng.integers(int(min_segments), int(max_segments) + 1))
                s = max(1, min(int(n_leaves), s))
                seg_lens_leaves = _sample_segment_lengths_in_leaves(
                    n_leaves=n_leaves,
                    n_segments=s,
                    min_seg_leaves=1,
                    max_seg_leaves=max(1, n_leaves // s),
                    boundary_profile_fn=None,
                    length_power=float(segment_length_power),
                    rng=rng,
                )

            leaf_i = 0
            prev_topic: Optional[int] = None
            for seg_leaves in seg_lens_leaves:
                if prev_topic is None:
                    topic_id = int(rng.choice(k, p=pi))
                else:
                    # Ensure segment boundaries correspond to actual topic changes.
                    p = np.asarray(pi, dtype=np.float64).copy()
                    p[int(prev_topic)] = 0.0
                    total = float(np.sum(p))
                    if total > 0:
                        p = p / total
                        topic_id = int(rng.choice(k, p=p))
                    else:
                        cands = np.arange(k, dtype=np.int64)
                        cands = cands[cands != int(prev_topic)]
                        topic_id = int(rng.choice(cands))
                prev_topic = int(topic_id)
                phi = topics[topic_id]
                for _leaf_j in range(int(seg_leaves)):
                    start, end = leaves[leaf_i]
                    leaf_i += 1
                    leaf_len = int(end - start)
                    draws = rng.choice(v, size=leaf_len, replace=True, p=phi)
                    tokens_out.extend(int(x) for x in draws.tolist())
                    topics_out.extend([topic_id] * leaf_len)
            segment_stats.append(int(len(seg_lens_leaves)))
            segment_len_leaves_all.extend(int(x) for x in seg_lens_leaves)

        docs.append(SegmentLDADoc(tokens=tuple(tokens_out), topics=tuple(topics_out)))

        # Diagnostics for interpretability: leaf-level changepoints + leaf topic purity.
        if n_leaves <= 1:
            changepoints.append(0)
            changepoint_rates.append(0.0)
        else:
            leaf_topics: List[int] = []
            purities: List[float] = []
            z = np.asarray(topics_out, dtype=np.int64)
            for start, end in leaves:
                seg = z[int(start) : int(end)]
                if seg.size == 0:
                    continue
                uniq, counts = np.unique(seg, return_counts=True)
                m = int(np.max(counts)) if counts.size else 0
                leaf_topics.append(int(uniq[int(np.argmax(counts))]))
                purities.append(float(m) / float(seg.size) if seg.size else 0.0)
            cp = sum(int(a != b) for a, b in zip(leaf_topics[:-1], leaf_topics[1:])) if len(leaf_topics) > 1 else 0
            changepoints.append(int(cp))
            changepoint_rates.append(float(cp) / float(max(1, len(leaf_topics) - 1)))
            if purities:
                leaf_purities.append(float(np.mean(np.asarray(purities, dtype=np.float64))))

    stats = {
        "mean_segments": float(np.mean(np.asarray(segment_stats, dtype=np.float64)))
        if segment_stats
        else 0.0
        ,
        "mean_changepoints": float(np.mean(np.asarray(changepoints, dtype=np.float64))) if changepoints else 0.0,
        "mean_changepoint_rate": float(np.mean(np.asarray(changepoint_rates, dtype=np.float64)))
        if changepoint_rates
        else 0.0,
        "mean_leaf_topic_purity": float(np.mean(np.asarray(leaf_purities, dtype=np.float64)))
        if leaf_purities
        else 0.0,
    }
    if segment_len_leaves_all:
        seg_lens = np.asarray(segment_len_leaves_all, dtype=np.float64)
        stats.update(
            {
                "mean_segment_len_leaves": float(np.mean(seg_lens)),
                "p50_segment_len_leaves": float(np.percentile(seg_lens, 50.0)),
                "p90_segment_len_leaves": float(np.percentile(seg_lens, 90.0)),
            }
        )
    return (tuple(docs), stats)


def infer_leaf_topics_from_words(
    tokens: Sequence[int],
    *,
    log_topics: np.ndarray,
    leaf_tokens: int,
) -> Tuple[int, ...]:
    """
    Infer a single topic id per leaf by maximum log-likelihood under φ_k.

    Returns a token-level topic sequence by repeating the inferred leaf topic across tokens.
    """

    toks = np.asarray(tokens, dtype=np.int64)
    if toks.size == 0:
        return tuple()
    k, v = log_topics.shape
    if int(v) <= int(np.max(toks)):
        raise ValueError("tokens contain ids outside vocab_size for provided topics")

    spans = build_leaf_spans(int(toks.size), leaf_tokens=int(leaf_tokens))
    inferred_leaf_topics: List[int] = []
    for start, end in spans:
        x = toks[int(start) : int(end)]
        # Score each topic by sum log p(word | topic).
        scores = np.sum(log_topics[:, x], axis=1)
        inferred_leaf_topics.append(int(np.argmax(scores)))
    # Repeat per-token.
    out: List[int] = []
    for (start, end), z in zip(spans, inferred_leaf_topics):
        out.extend([int(z)] * int(end - start))
    return tuple(out)


def _prefix_counts(topics: Sequence[int], *, n_topics: int) -> Tuple[np.ndarray, np.ndarray]:
    z = np.asarray(topics, dtype=np.int64)
    if z.size == 0:
        return (
            np.zeros((1, int(n_topics)), dtype=np.int32),
            np.zeros((1, int(n_topics) * int(n_topics)), dtype=np.int32),
        )

    k = int(n_topics)
    if int(np.max(z)) >= k or int(np.min(z)) < 0:
        raise ValueError("topic ids out of range")

    onehot = np.eye(k, dtype=np.int32)[z]
    topic_prefix = np.concatenate([np.zeros((1, k), dtype=np.int32), np.cumsum(onehot, axis=0)], axis=0)

    if z.size >= 2:
        idx = (z[:-1] * k + z[1:]).astype(np.int64, copy=False)
        onehot_b = np.eye(k * k, dtype=np.int32)[idx]
        bigram_prefix = np.concatenate(
            [np.zeros((1, k * k), dtype=np.int32), np.cumsum(onehot_b, axis=0)], axis=0
        )
    else:
        bigram_prefix = np.zeros((1, k * k), dtype=np.int32)

    return topic_prefix, bigram_prefix


def _span_features_from_prefix(
    topic_prefix: np.ndarray,
    bigram_prefix: np.ndarray,
    topics: Sequence[int],
    span: Span,
    *,
    n_topics: int,
) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
    start, end = span
    i = int(start)
    j = int(end)
    if i < 0 or j < i or j > len(topics):
        raise ValueError("span out of range")
    k = int(n_topics)
    u = (topic_prefix[j] - topic_prefix[i]).astype(np.float64, copy=False)
    if j - i >= 2:
        b = (bigram_prefix[j - 1] - bigram_prefix[i]).astype(np.float64, copy=False)
    else:
        b = np.zeros((k * k,), dtype=np.float64)
    first = int(topics[i]) if j - i > 0 else None
    last = int(topics[j - 1]) if j - i > 0 else None
    feat = np.concatenate([u, b], axis=0)
    return feat, first, last


def _oracle_from_prefix(
    theta: np.ndarray,
    w_big: np.ndarray,
    topic_prefix: np.ndarray,
    bigram_prefix: np.ndarray,
    topics: Sequence[int],
    span: Span,
) -> float:
    feat, _first, _last = _span_features_from_prefix(
        topic_prefix,
        bigram_prefix,
        topics,
        span,
        n_topics=int(theta.size),
    )
    return float(np.dot(np.concatenate([theta, w_big], axis=0), feat))


def sample_sparse_oracle_weights(
    *,
    n_topics: int,
    relevant_topics: int,
    theta_scale: float,
    zero_diagonal: bool,
    seed: int,
) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray]:
    """
    Sample sparse (θ, W_base) with a small relevant topic set R.

    W_base is normalized to Frobenius norm 1 (unless it is identically zero).
    """

    rng = np.random.default_rng(int(seed))
    k = int(n_topics)
    r = int(max(1, relevant_topics))
    if r > k:
        raise ValueError("relevant_topics must be <= n_topics")
    R = tuple(int(x) for x in rng.choice(k, size=r, replace=False).tolist())

    theta = np.zeros((k,), dtype=np.float64)
    theta[list(R)] = rng.normal(loc=0.0, scale=float(theta_scale), size=(r,)).astype(np.float64, copy=False)

    W = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            if bool(zero_diagonal) and i == j:
                continue
            if (i in R) or (j in R):
                W[i, j] = float(rng.normal(loc=0.0, scale=1.0))

    frob = float(np.linalg.norm(W.reshape(-1), ord=2))
    if frob > 0:
        W = W / frob
    return (R, theta, W)


@dataclass(frozen=True)
class SketchMetrics:
    root_mae: float
    root_median_abs_error: float
    root_p95_abs_error: float
    schedule_spread_mean: float
    schedule_spread_p95: float
    leaf_mae: float
    leaf_violation_rate: float
    merge_mae: float
    merge_violation_rate: float
    n_docs: int


@dataclass(frozen=True)
class WeightRecoveryMetrics:
    beta_rmse: float
    beta_cosine: float
    theta_rmse: float
    theta_cosine: float
    bigram_rmse: float
    bigram_cosine: float
    lambda_hat: float
    lambda_abs_error: float
    W_direction_cosine: float


@dataclass(frozen=True)
class TopicInferenceMetrics:
    """How well inferred leaf topics match the true latent topics (after topic alignment)."""

    leaf_accuracy_train: float
    leaf_accuracy_test: float


@dataclass(frozen=True)
class TrainingGeometry:
    mean_tokens: float
    mean_leaves: float
    mean_internal_nodes: float
    mean_leaf_labels: float
    mean_internal_labels: float
    mean_queries_per_doc: float
    leaf_labels_total: int
    internal_labels_total: int
    total_labels_total: int
    train_full_doc_cost_total: float


@dataclass(frozen=True)
class OracleBudget:
    oracle_queries_leaf_total: int
    oracle_queries_internal_total: int
    oracle_queries_total: int
    oracle_cost_leaf_total: float
    oracle_cost_internal_total: float
    oracle_cost_total: float
    oracle_cost_ratio: float


@dataclass(frozen=True)
class DesignMetrics:
    """Diagnostics for the ridge design matrix X (train labels)."""

    n_labels: int
    d: int
    rank: int
    xtx_trace: float
    xtx_min_eig: float
    xtx_max_eig: float
    xtx_condition: float
    a_min_eig: float
    a_max_eig: float
    a_condition: float
    row_norm_mean: float
    row_norm_p95: float
    train_rmse: float


@dataclass(frozen=True)
class RidgeRunMetrics:
    sketch: SketchMetrics
    weights: WeightRecoveryMetrics
    topic_inference: TopicInferenceMetrics
    budget: OracleBudget
    design: DesignMetrics


@dataclass(frozen=True)
class SegmentLDAOpsWeightRecoveryConfig:
    # Generator.
    n_topics: int = 8
    vocab_size: int = 512
    min_tokens: int = 384
    max_tokens: int = 384
    min_segments: int = 2
    max_segments: int = 6
    min_seg_len: int = 48
    max_seg_len: int = 256
    leaf_tokens: int = 16
    align_segments_to_leaves: bool = True
    doc_topic_concentration: float = 0.6
    topic_process: TopicProcessName = "segments"
    boundary_profile: BoundaryProfileName = "uniform"
    boundary_profile_strength: float = 0.0
    boundary_profile_seed: int = -1
    segment_length_power: float = 1.0

    # Topic-word distributions.
    topic_concentration: float = 0.2
    emission_mode: str = "anchored"  # "anchored" or "disjoint"
    anchor_words_per_topic: int = 20
    anchor_multiplier: float = 25.0

    # Oracle weights.
    relevant_topics: int = 2
    theta_scale: float = 1.0
    zero_diagonal: bool = True
    lambda_multiplier: float = 1.0

    # Oracle observation noise (added to training labels only).
    oracle_noise_std: float = 0.0

    # Supervision / budgets.
    audit_policy: AuditPolicyName = "fraction"
    audit_fixed_nodes: int = 0
    audit_fraction: float = 0.2
    audit_scale: float = 1.0
    audit_strategy: AuditStrategyName = "random"
    oracle_cost_power: float = 1.25
    oracle_cost_per_query: float = 0.0

    # Estimation.
    ridge_lambda: float = 1e-3
    topic_source: TopicSourceName = "infer"  # features from "true" topics or inferred from words
    feature_inference: FeatureInferenceName = "hard"  # hard leaf MAP vs soft posteriors (infer-only)

    # Topic-word estimation (Tensor-LDA-inspired upstream error model).
    topic_phi_estimator: TopicPhiEstimatorName = "true"  # See VALID_TOPIC_PHI_ESTIMATORS.
    topic_phi_docs: int = 0  # if <=0, defaults to train_docs for the estimator's effective N
    sklearn_lda_max_iter: int = 60
    tlda_delta: float = 0.10
    tlda_rate_constant: float = 1.0
    tlda_sigmaK_floor: float = 1e-6
    topic_phi_permute: bool = True  # simulate topic unidentifiability (up to permutation)

    # Online Tensor-LDA knobs (used only when topic_phi_estimator="online_tensor_lda").
    online_tensor_lda_burn_in_docs: int = 0  # 0 => auto
    online_tensor_lda_batch_docs: int = 32
    online_tensor_lda_passes: int = 1
    online_tensor_lda_lr: float = 0.1
    online_tensor_lda_grad_clip_norm: float = 1.0

    # Embedding topic estimator knobs (used by embedding_spectral and neural_embedding_hybrid).
    embedding_topic_svd_dim_extra: int = 4
    embedding_topic_kmeans_inits: int = 8
    embedding_topic_kmeans_max_iter: int = 80
    embedding_topic_assignment_temperature: float = 0.35
    embedding_topic_ppmi_shift: float = 1.0

    # Neural topic refiner knobs (used when topic_phi_estimator starts with "neural_").
    neural_topic_base_estimator: str = "tensor_lda"
    neural_topic_seed_fraction: float = 0.35
    neural_topic_hidden_dim: int = 48
    neural_topic_steps: int = 60
    neural_topic_lr: float = 3e-3
    neural_topic_weight_decay: float = 1e-4
    neural_topic_mix_samples: int = 128
    neural_topic_mix_temperature: float = 1.0
    neural_topic_operator_boost: float = 1.4
    neural_topic_seed_llm_min_weight: float = 0.2
    neural_topic_seed_llm_max_weight: float = 0.55
    neural_topic_similarity_temperature: float = 0.15
    neural_topic_ridge: float = 1e-3

    # Optional decomposition mode: report ridge metrics under multiple feature sources in one run.
    run_all_feature_modes: bool = False

    # Evaluation.
    violation_tau: float = 0.0
    train_docs: int = 1000
    test_docs: int = 1000
    seed: int = 0


@dataclass(frozen=True)
class SegmentLDAOpsWeightRecoverySummary:
    config: Dict[str, object]
    topic_meta: Dict[str, object]
    weight_truth: Dict[str, object]
    training_geometry: Dict[str, object]
    metrics: Dict[str, object]

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "topic_meta": self.topic_meta,
            "weight_truth": self.weight_truth,
            "training_geometry": self.training_geometry,
            "metrics": self.metrics,
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def _training_geometry(
    docs: Sequence[SegmentLDADoc],
    *,
    leaf_tokens: int,
    audit_policy: AuditPolicyName,
    audit_fixed_nodes: int,
    audit_fraction: float,
    audit_scale: float,
    oracle_cost_power: float,
    oracle_cost_per_query: float,
) -> TrainingGeometry:
    if len(docs) == 0:
        return TrainingGeometry(
            mean_tokens=0.0,
            mean_leaves=0.0,
            mean_internal_nodes=0.0,
            mean_leaf_labels=0.0,
            mean_internal_labels=0.0,
            mean_queries_per_doc=0.0,
            leaf_labels_total=0,
            internal_labels_total=0,
            total_labels_total=0,
            train_full_doc_cost_total=0.0,
        )

    toks: List[float] = []
    leaves: List[float] = []
    internals: List[float] = []
    internal_labels: List[float] = []
    leaf_labels_total = 0
    internal_labels_total = 0
    full_cost_total = 0.0

    for doc in docs:
        n_tok = int(len(doc.tokens))
        nodes = build_balanced_tree_nodes(n_tok, leaf_tokens=int(leaf_tokens))
        n_leaves = sum(1 for n in nodes if n.is_leaf)
        n_internal = max(0, n_leaves - 1)
        q_internal = audit_sample_count(
            n_internal,
            policy=str(audit_policy),
            fixed_nodes=int(audit_fixed_nodes),
            fraction=float(audit_fraction),
            scale=float(audit_scale),
        )
        toks.append(float(n_tok))
        leaves.append(float(n_leaves))
        internals.append(float(n_internal))
        internal_labels.append(float(q_internal))
        leaf_labels_total += int(n_leaves)
        internal_labels_total += int(q_internal)
        full_cost_total += oracle_query_cost(
            n_tok,
            power=float(oracle_cost_power),
            per_query=float(oracle_cost_per_query),
        )

    mean_leaves = float(np.mean(np.asarray(leaves, dtype=np.float64)))
    mean_internal_labels = float(np.mean(np.asarray(internal_labels, dtype=np.float64)))
    return TrainingGeometry(
        mean_tokens=float(np.mean(np.asarray(toks, dtype=np.float64))),
        mean_leaves=float(mean_leaves),
        mean_internal_nodes=float(np.mean(np.asarray(internals, dtype=np.float64))),
        mean_leaf_labels=float(mean_leaves),
        mean_internal_labels=float(mean_internal_labels),
        mean_queries_per_doc=float(mean_leaves + mean_internal_labels),
        leaf_labels_total=int(leaf_labels_total),
        internal_labels_total=int(internal_labels_total),
        total_labels_total=int(leaf_labels_total + internal_labels_total),
        train_full_doc_cost_total=float(full_cost_total),
    )


def _fit_ridge(X: np.ndarray, y: np.ndarray, *, ridge_lambda: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be a 1D vector aligned with X rows")
    lam = float(ridge_lambda)
    if lam < 0:
        raise ValueError("ridge_lambda must be >= 0")
    d = int(X.shape[1])
    XtX = X.T @ X
    Xty = X.T @ y
    A = XtX + lam * np.eye(d, dtype=np.float64)
    return np.linalg.solve(A, Xty)


def _design_metrics(
    X: Optional[np.ndarray],
    y: Optional[np.ndarray],
    beta_hat: np.ndarray,
    *,
    ridge_lambda: float,
) -> DesignMetrics:
    """Compute lightweight diagnostics for the ridge design matrix."""

    d = int(np.asarray(beta_hat, dtype=np.float64).reshape(-1).size)
    if X is None or y is None:
        return DesignMetrics(
            n_labels=0,
            d=int(d),
            rank=0,
            xtx_trace=0.0,
            xtx_min_eig=float("nan"),
            xtx_max_eig=float("nan"),
            xtx_condition=float("nan"),
            a_min_eig=float("nan"),
            a_max_eig=float("nan"),
            a_condition=float("nan"),
            row_norm_mean=float("nan"),
            row_norm_p95=float("nan"),
            train_rmse=float("nan"),
        )

    X = np.asarray(X, dtype=np.float64)
    y_vec = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or y_vec.ndim != 1 or int(X.shape[0]) != int(y_vec.shape[0]):
        raise ValueError("design_metrics: X and y must be aligned")

    n_labels = int(X.shape[0])
    d = int(X.shape[1])
    rank = int(np.linalg.matrix_rank(X)) if n_labels > 0 and d > 0 else 0

    XtX = X.T @ X
    XtX_sym = 0.5 * (XtX + XtX.T)
    if d > 0:
        eig = np.linalg.eigvalsh(XtX_sym).astype(np.float64, copy=False)
        xtx_min = float(np.min(eig)) if eig.size else 0.0
        xtx_max = float(np.max(eig)) if eig.size else 0.0
    else:
        xtx_min = 0.0
        xtx_max = 0.0

    xtx_trace = float(np.trace(XtX_sym)) if d > 0 else 0.0
    xtx_cond = float(xtx_max / xtx_min) if xtx_min > 0 else float("inf") if d > 0 else float("nan")

    lam = float(ridge_lambda)
    a_min = float(xtx_min + lam)
    a_max = float(xtx_max + lam)
    a_cond = float(a_max / a_min) if a_min > 0 else float("inf") if d > 0 else float("nan")

    row_norms = np.linalg.norm(X, axis=1) if n_labels > 0 else np.asarray([], dtype=np.float64)
    row_norm_mean = float(np.mean(row_norms)) if row_norms.size else float("nan")
    row_norm_p95 = float(np.percentile(row_norms, 95.0)) if row_norms.size else float("nan")

    resid = (X @ np.asarray(beta_hat, dtype=np.float64).reshape(-1)) - y_vec
    train_rmse = float(math.sqrt(float(np.mean(resid * resid)))) if resid.size else float("nan")

    return DesignMetrics(
        n_labels=int(n_labels),
        d=int(d),
        rank=int(rank),
        xtx_trace=float(xtx_trace),
        xtx_min_eig=float(xtx_min),
        xtx_max_eig=float(xtx_max),
        xtx_condition=float(xtx_cond),
        a_min_eig=float(a_min),
        a_max_eig=float(a_max),
        a_condition=float(a_cond),
        row_norm_mean=float(row_norm_mean),
        row_norm_p95=float(row_norm_p95),
        train_rmse=float(train_rmse),
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(math.sqrt(float(np.mean(diff * diff)))) if diff.size else 0.0


def _eval_schedule_spread(
    leaf_values: Sequence[float],
    *,
    leaf_topics: Sequence[int],
    theta: np.ndarray,
    w_big: np.ndarray,
    n_topics: int,
    schedule: ScheduleName,
) -> float:
    """
    Reduce leaf-level topic segments under a schedule using the *exact* sketch merge.

    This returns the root prediction (not the spread); we use it to compute spread over schedules.
    """

    # Each leaf is treated as a constant-topic block; its sketch is:
    # - unigram counts: leaf_len topic occurrences (implicit via leaf_values computation)
    # Here we already have leaf_values, but to keep this function small, reconstruct via topics.
    # This is only used for schedule-spread diagnostics (should be 0 under exact mergeability).
    del theta, w_big, n_topics
    # Since the exact merge is associative and linear, root value is the sum over leaves regardless.
    return float(np.sum(np.asarray(leaf_values, dtype=np.float64)))


def _eval_sketch_family(
    docs: Sequence[SegmentLDADoc],
    *,
    theta_true: np.ndarray,
    w_big_true: np.ndarray,
    leaf_tokens: int,
    tau: float,
    kind: str,
    rounds: int = 1,
    flip_bonus: float = 1.0,
) -> SketchMetrics:
    """
    Evaluate baseline sketch families on true topics with true weights:
    - kind="exact": exact mergeable sketch (0 distortion)
    - kind="undersupported": drops boundary bigram contributions on merges
    - kind="flip": exact one-pass, but resummary toggles a +bonus at R>=2 (L3 failure)
    """

    if len(docs) == 0:
        return SketchMetrics(
            root_mae=0.0,
            root_median_abs_error=0.0,
            root_p95_abs_error=0.0,
            schedule_spread_mean=0.0,
            schedule_spread_p95=0.0,
            leaf_mae=0.0,
            leaf_violation_rate=0.0,
            merge_mae=0.0,
            merge_violation_rate=0.0,
            n_docs=0,
        )

    if str(kind) not in {"exact", "undersupported", "flip"}:
        raise ValueError("kind must be one of: exact, undersupported, flip")

    R = int(max(1, rounds))
    tau = float(tau)

    root_abs: List[float] = []
    spreads: List[float] = []
    leaf_abs: List[float] = []
    merge_abs: List[float] = []

    for doc in docs:
        n_tok = int(len(doc.topics))
        nodes = build_balanced_tree_nodes(n_tok, leaf_tokens=int(leaf_tokens))
        spans = [(n.token_start, n.token_end) for n in nodes if n.is_leaf]
        if not spans:
            continue

        topic_prefix, bigram_prefix = _prefix_counts(doc.topics, n_topics=int(theta_true.size))

        # Leaf values (C1).
        leaf_vals: List[float] = []
        for sp in spans:
            y_true = _oracle_from_prefix(theta_true, w_big_true, topic_prefix, bigram_prefix, doc.topics, sp)
            leaf_vals.append(float(y_true))
            leaf_abs.append(0.0)  # exact by construction at leaves for all kinds we consider

        # Root prediction under schedules.
        roots: Dict[str, float] = {}
        for sched in VALID_SCHEDULES:
            if str(sched) == "balanced":
                # For these baselines, schedule spread is only meaningful if merge is not associative.
                roots[str(sched)] = float(np.sum(np.asarray(leaf_vals, dtype=np.float64)))
            elif str(sched) in ("left_to_right", "right_to_left"):
                roots[str(sched)] = float(np.sum(np.asarray(leaf_vals, dtype=np.float64)))
            else:
                raise ValueError(f"unsupported schedule: {sched!r}")

        # True root oracle.
        root_span = (0, n_tok)
        y_root_true = _oracle_from_prefix(
            theta_true, w_big_true, topic_prefix, bigram_prefix, doc.topics, root_span
        )

        if str(kind) == "exact":
            pred_root = float(y_root_true)
        elif str(kind) == "undersupported":
            # Drop cross-leaf boundary bigrams: subtract contributions for every leaf boundary.
            # Under align-to-leaves generation, topic changes only occur at leaf boundaries, but
            # even when topic stays the same, that boundary bigram still exists in the oracle.
            # We remove *all* boundary bigram contributions to simulate a sketch without endpoints.
            pred_root = float(y_root_true)
            # Remove boundary contributions between consecutive leaves.
            for (s0, e0), (s1, e1) in zip(spans[:-1], spans[1:]):
                last = int(doc.topics[int(e0) - 1])
                first = int(doc.topics[int(s1)])
                idx = last * int(theta_true.size) + first
                pred_root -= float(w_big_true[idx])
        else:  # flip
            pred_root = float(y_root_true)
            if R >= 2:
                pred_root += float(flip_bonus)

        root_abs.append(abs(pred_root - float(y_root_true)))
        spreads.append(max(roots.values()) - min(roots.values()))

        # Merge C3 discrepancies (balanced schedule only), first-pass only.
        # We compute discrepancies on all internal nodes in the balanced tree.
        internal_nodes = [n for n in nodes if not n.is_leaf]
        for node in internal_nodes:
            sp = (node.token_start, node.token_end)
            y_true = _oracle_from_prefix(theta_true, w_big_true, topic_prefix, bigram_prefix, doc.topics, sp)
            if str(kind) == "exact" or str(kind) == "flip":
                y_pred = float(y_true)
            else:
                # Undersupported: remove boundary contributions *within* this span, i.e. at leaf
                # boundaries that lie inside [start,end).
                y_pred = float(y_true)
                # Identify leaf boundaries inside the node span by using leaf spans.
                # Since `nodes` is balanced over these leaves, the node spans contiguous leaves.
                # Here we approximate by scanning leaf boundary indices.
                start, end = sp
                for (s0, e0), (s1, e1) in zip(spans[:-1], spans[1:]):
                    if int(e0) <= int(start) or int(s1) >= int(end):
                        continue
                    last = int(doc.topics[int(e0) - 1])
                    first = int(doc.topics[int(s1)])
                    idx = last * int(theta_true.size) + first
                    y_pred -= float(w_big_true[idx])
            merge_abs.append(abs(float(y_pred) - float(y_true)))

    root_abs_arr = np.asarray(root_abs, dtype=np.float64)
    spreads_arr = np.asarray(spreads, dtype=np.float64)
    merge_abs_arr = np.asarray(merge_abs, dtype=np.float64)

    leaf_abs_arr = np.asarray(leaf_abs, dtype=np.float64)

    return SketchMetrics(
        root_mae=float(np.mean(root_abs_arr)) if root_abs_arr.size else 0.0,
        root_median_abs_error=float(np.median(root_abs_arr)) if root_abs_arr.size else 0.0,
        root_p95_abs_error=float(np.percentile(root_abs_arr, 95.0)) if root_abs_arr.size else 0.0,
        schedule_spread_mean=float(np.mean(spreads_arr)) if spreads_arr.size else 0.0,
        schedule_spread_p95=float(np.percentile(spreads_arr, 95.0)) if spreads_arr.size else 0.0,
        leaf_mae=float(np.mean(leaf_abs_arr)) if leaf_abs_arr.size else 0.0,
        leaf_violation_rate=float(np.mean((leaf_abs_arr > tau).astype(np.float64)))
        if leaf_abs_arr.size
        else 0.0,
        merge_mae=float(np.mean(merge_abs_arr)) if merge_abs_arr.size else 0.0,
        merge_violation_rate=float(np.mean((merge_abs_arr > tau).astype(np.float64)))
        if merge_abs_arr.size
        else 0.0,
        n_docs=int(len(root_abs)),
    )


def _compute_ridge_metrics(
    *,
    docs_train: Sequence[SegmentLDADoc],
    docs_test: Sequence[SegmentLDADoc],
    topics_phi: Sequence[np.ndarray],
    perm_est_to_true: Optional[Sequence[int]] = None,
    theta_true: np.ndarray,
    W_base: np.ndarray,
    lambda_multiplier: float,
    config: SegmentLDAOpsWeightRecoveryConfig,
) -> RidgeRunMetrics:
    k = int(config.n_topics)
    d = int(k + k * k)

    w_big_true = float(lambda_multiplier) * W_base.reshape(-1).astype(np.float64, copy=False)
    beta_true = np.concatenate([theta_true, w_big_true], axis=0)

    log_topics = np.log(np.clip(np.stack(list(topics_phi), axis=0), 1e-12, 1.0))

    rng = np.random.default_rng(int(_splitmix64(int(config.seed) + 333) & 0xFFFFFFFF))

    X_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    oracle_noise_std = float(getattr(config, "oracle_noise_std", 0.0))

    oracle_q_leaf = 0
    oracle_q_internal = 0
    oracle_cost_leaf = 0.0
    oracle_cost_internal = 0.0

    if str(config.feature_inference) != "hard":
        raise ValueError(
            f"feature_inference={config.feature_inference!r} unsupported; expected {VALID_FEATURE_INFERENCE}"
        )

    perm = tuple(int(x) for x in perm_est_to_true) if perm_est_to_true is not None else None
    if perm is not None and len(perm) != k:
        raise ValueError("perm_est_to_true must have length n_topics")

    def _map_topic_to_true(topic_id: int) -> int:
        if perm is None:
            return int(topic_id)
        tid = int(topic_id)
        if tid < 0 or tid >= k:
            return int(tid)
        return int(perm[tid])

    def _leaf_accuracy(docs: Sequence[SegmentLDADoc], *, topics_inferred: Optional[Sequence[Tuple[int, ...]]]) -> float:
        if topics_inferred is None:
            return float("nan")
        correct = 0
        total = 0
        for doc, inferred in zip(docs, topics_inferred):
            spans = build_leaf_spans(len(doc.tokens), leaf_tokens=int(config.leaf_tokens))
            if not spans:
                continue
            z_true = np.asarray(doc.topics, dtype=np.int64)
            z_inf = np.asarray(inferred, dtype=np.int64)
            for start, end in spans:
                seg_true = z_true[int(start) : int(end)]
                seg_inf = z_inf[int(start) : int(end)]
                if seg_true.size == 0 or seg_inf.size == 0:
                    continue
                uniq_t, cnt_t = np.unique(seg_true, return_counts=True)
                true_leaf = int(uniq_t[int(np.argmax(cnt_t))])
                uniq_i, cnt_i = np.unique(seg_inf, return_counts=True)
                inf_leaf = int(uniq_i[int(np.argmax(cnt_i))])
                inf_leaf_true = _map_topic_to_true(inf_leaf)
                correct += 1 if int(inf_leaf_true) == int(true_leaf) else 0
                total += 1
        return float(correct) / float(total) if total > 0 else float("nan")

    inferred_train: Optional[List[Tuple[int, ...]]] = [] if str(config.topic_source) == "infer" else None
    inferred_test: Optional[List[Tuple[int, ...]]] = [] if str(config.topic_source) == "infer" else None

    boundary_rate_by_index: Optional[np.ndarray] = None
    if str(config.audit_strategy).strip().lower() == "profile":
        max_leaves = 0
        for doc in docs_train:
            spans = build_leaf_spans(len(doc.tokens), leaf_tokens=int(config.leaf_tokens))
            max_leaves = max(max_leaves, int(len(spans)))
        if max_leaves >= 2:
            hits = np.zeros((max_leaves - 1,), dtype=np.float64)
            cnt = np.zeros((max_leaves - 1,), dtype=np.float64)
            for doc in docs_train:
                spans = build_leaf_spans(len(doc.tokens), leaf_tokens=int(config.leaf_tokens))
                if len(spans) < 2:
                    continue
                if str(config.topic_source) == "true":
                    topics_feat = doc.topics
                else:
                    topics_feat = infer_leaf_topics_from_words(
                        doc.tokens,
                        log_topics=log_topics,
                        leaf_tokens=int(config.leaf_tokens),
                    )

                leaf_ids: List[int] = []
                for start, end in spans:
                    seg = np.asarray(topics_feat[int(start) : int(end)], dtype=np.int64)
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

    for doc in docs_train:
        n_tok = int(len(doc.tokens))
        nodes = build_balanced_tree_nodes(n_tok, leaf_tokens=int(config.leaf_tokens))
        leaf_nodes = [n for n in nodes if n.is_leaf]
        internal_nodes = [n for n in nodes if not n.is_leaf]
        n_leaves = int(len(leaf_nodes))
        n_internal = int(len(internal_nodes))

        if str(config.topic_source) == "true":
            topics_feat = doc.topics
        elif str(config.topic_source) == "infer":
            topics_feat = infer_leaf_topics_from_words(
                doc.tokens,
                log_topics=log_topics,
                leaf_tokens=int(config.leaf_tokens),
            )
            if inferred_train is not None:
                inferred_train.append(tuple(int(x) for x in topics_feat))
        else:
            raise ValueError(
                f"unsupported topic_source: {config.topic_source!r}; expected one of {VALID_TOPIC_SOURCES}"
            )

        feat_topic_prefix, feat_bigram_prefix = _prefix_counts(topics_feat, n_topics=k)
        true_topic_prefix, true_bigram_prefix = _prefix_counts(doc.topics, n_topics=k)

        # Leaf queries: always include.
        for node in leaf_nodes:
            sp = (node.token_start, node.token_end)
            feat, _first, _last = _span_features_from_prefix(
                feat_topic_prefix,
                feat_bigram_prefix,
                topics_feat,
                sp,
                n_topics=k,
            )
            y = _oracle_from_prefix(theta_true, w_big_true, true_topic_prefix, true_bigram_prefix, doc.topics, sp)
            if oracle_noise_std > 0.0:
                y = float(y) + float(rng.normal(loc=0.0, scale=oracle_noise_std))
            X_rows.append(feat)
            y_rows.append(float(y))
            oracle_q_leaf += 1
            oracle_cost_leaf += oracle_query_cost(
                node.token_len,
                power=float(config.oracle_cost_power),
                per_query=float(config.oracle_cost_per_query),
            )

        # Internal queries: budget + strategy.
        q_internal = audit_sample_count(
            n_internal,
            policy=str(config.audit_policy),
            fixed_nodes=int(config.audit_fixed_nodes),
            fraction=float(config.audit_fraction),
            scale=float(config.audit_scale),
        )
        selected = select_audit_nodes(
            internal_nodes,
            n_queries=q_internal,
            strategy=str(config.audit_strategy),
            rng=rng,
            boundary_rate_by_index=boundary_rate_by_index,
        )
        for rel_idx in selected:
            node = internal_nodes[int(rel_idx)]
            sp = (node.token_start, node.token_end)
            feat, _first, _last = _span_features_from_prefix(
                feat_topic_prefix,
                feat_bigram_prefix,
                topics_feat,
                sp,
                n_topics=k,
            )
            y = _oracle_from_prefix(theta_true, w_big_true, true_topic_prefix, true_bigram_prefix, doc.topics, sp)
            if oracle_noise_std > 0.0:
                y = float(y) + float(rng.normal(loc=0.0, scale=oracle_noise_std))
            X_rows.append(feat)
            y_rows.append(float(y))
            oracle_q_internal += 1
            oracle_cost_internal += oracle_query_cost(
                node.token_len,
                power=float(config.oracle_cost_power),
                per_query=float(config.oracle_cost_per_query),
            )

    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    if not X_rows:
        beta_hat = np.zeros((d,), dtype=np.float64)
    else:
        X = np.vstack(X_rows).astype(np.float64, copy=False)
        y = np.asarray(y_rows, dtype=np.float64)
        beta_hat = _fit_ridge(X, y, ridge_lambda=float(config.ridge_lambda))
    design = _design_metrics(X, y, beta_hat, ridge_lambda=float(config.ridge_lambda))

    theta_hat = beta_hat[:k]
    w_big_hat = beta_hat[k:]

    # Weight recovery metrics.
    if perm is not None:
        theta_hat_cmp = _permute_theta_to_true_basis(theta_hat, perm_est_to_true=perm)
        W_hat_cmp = _permute_W_to_true_basis(w_big_hat.reshape(k, k), perm_est_to_true=perm)
        w_big_hat_cmp = W_hat_cmp.reshape(-1).astype(np.float64, copy=False)
        beta_hat_cmp = np.concatenate([theta_hat_cmp, w_big_hat_cmp], axis=0)
    else:
        theta_hat_cmp = theta_hat
        w_big_hat_cmp = w_big_hat
        beta_hat_cmp = beta_hat

    beta_rmse = _rmse(beta_hat_cmp, beta_true)
    beta_cos = _cosine(beta_hat_cmp, beta_true)
    theta_rmse = _rmse(theta_hat_cmp, theta_true)
    theta_cos = _cosine(theta_hat_cmp, theta_true)
    bigram_rmse = _rmse(w_big_hat_cmp, w_big_true)
    bigram_cos = _cosine(w_big_hat_cmp, w_big_true)

    lambda_hat = float(np.linalg.norm(w_big_hat_cmp))
    lambda_abs_error = abs(lambda_hat - float(lambda_multiplier))
    if float(lambda_multiplier) > 0 and lambda_hat > 0:
        W_dir_cos = _cosine(w_big_hat_cmp / lambda_hat, W_base.reshape(-1))
    else:
        W_dir_cos = float("nan")

    weights = WeightRecoveryMetrics(
        beta_rmse=float(beta_rmse),
        beta_cosine=float(beta_cos),
        theta_rmse=float(theta_rmse),
        theta_cosine=float(theta_cos),
        bigram_rmse=float(bigram_rmse),
        bigram_cosine=float(bigram_cos),
        lambda_hat=float(lambda_hat),
        lambda_abs_error=float(lambda_abs_error),
        W_direction_cosine=float(W_dir_cos),
    )

    # Budget summary.
    geom = _training_geometry(
        docs_train,
        leaf_tokens=int(config.leaf_tokens),
        audit_policy=str(config.audit_policy),
        audit_fixed_nodes=int(config.audit_fixed_nodes),
        audit_fraction=float(config.audit_fraction),
        audit_scale=float(config.audit_scale),
        oracle_cost_power=float(config.oracle_cost_power),
        oracle_cost_per_query=float(config.oracle_cost_per_query),
    )
    full_cost = float(geom.train_full_doc_cost_total)
    oracle_cost_total = float(oracle_cost_leaf + oracle_cost_internal)
    budget = OracleBudget(
        oracle_queries_leaf_total=int(oracle_q_leaf),
        oracle_queries_internal_total=int(oracle_q_internal),
        oracle_queries_total=int(oracle_q_leaf + oracle_q_internal),
        oracle_cost_leaf_total=float(oracle_cost_leaf),
        oracle_cost_internal_total=float(oracle_cost_internal),
        oracle_cost_total=float(oracle_cost_total),
        oracle_cost_ratio=float(oracle_cost_total / full_cost) if full_cost > 0 else float("nan"),
    )

    # Evaluate predictions on test docs (C1/C3/root).
    tau = float(config.violation_tau)
    root_abs: List[float] = []
    spreads: List[float] = []
    leaf_abs: List[float] = []
    merge_abs: List[float] = []

    for doc in docs_test:
        n_tok = int(len(doc.tokens))
        nodes = build_balanced_tree_nodes(n_tok, leaf_tokens=int(config.leaf_tokens))
        leaf_nodes = [n for n in nodes if n.is_leaf]
        internal_nodes = [n for n in nodes if not n.is_leaf]
        if not leaf_nodes:
            continue

        if str(config.topic_source) == "true":
            topics_feat = doc.topics
        else:
            topics_feat = infer_leaf_topics_from_words(
                doc.tokens,
                log_topics=log_topics,
                leaf_tokens=int(config.leaf_tokens),
            )
            if inferred_test is not None:
                inferred_test.append(tuple(int(x) for x in topics_feat))

        feat_topic_prefix, feat_bigram_prefix = _prefix_counts(topics_feat, n_topics=k)
        true_topic_prefix, true_bigram_prefix = _prefix_counts(doc.topics, n_topics=k)

        # Leaf errors.
        for node in leaf_nodes:
            sp = (node.token_start, node.token_end)
            feat, _first, _last = _span_features_from_prefix(
                feat_topic_prefix,
                feat_bigram_prefix,
                topics_feat,
                sp,
                n_topics=k,
            )
            pred = float(np.dot(beta_hat, feat))
            truth = _oracle_from_prefix(theta_true, w_big_true, true_topic_prefix, true_bigram_prefix, doc.topics, sp)
            leaf_abs.append(abs(pred - float(truth)))

        # Merge errors (all internal nodes in balanced tree).
        for node in internal_nodes:
            sp = (node.token_start, node.token_end)
            feat, _first, _last = _span_features_from_prefix(
                feat_topic_prefix,
                feat_bigram_prefix,
                topics_feat,
                sp,
                n_topics=k,
            )
            pred = float(np.dot(beta_hat, feat))
            truth = _oracle_from_prefix(theta_true, w_big_true, true_topic_prefix, true_bigram_prefix, doc.topics, sp)
            merge_abs.append(abs(pred - float(truth)))

        # Root predictions for schedule spread.
        roots: Dict[str, float] = {}
        root_span = (0, n_tok)
        root_feat, _first, _last = _span_features_from_prefix(
            feat_topic_prefix,
            feat_bigram_prefix,
            topics_feat,
            root_span,
            n_topics=k,
        )
        root_pred = float(np.dot(beta_hat, root_feat))
        for sched in VALID_SCHEDULES:
            roots[str(sched)] = float(root_pred)
        spreads.append(max(roots.values()) - min(roots.values()))

        root_truth = _oracle_from_prefix(
            theta_true, w_big_true, true_topic_prefix, true_bigram_prefix, doc.topics, root_span
        )
        root_abs.append(abs(root_pred - float(root_truth)))

    root_abs_arr = np.asarray(root_abs, dtype=np.float64)
    spreads_arr = np.asarray(spreads, dtype=np.float64)
    leaf_abs_arr = np.asarray(leaf_abs, dtype=np.float64)
    merge_abs_arr = np.asarray(merge_abs, dtype=np.float64)

    sketch = SketchMetrics(
        root_mae=float(np.mean(root_abs_arr)) if root_abs_arr.size else 0.0,
        root_median_abs_error=float(np.median(root_abs_arr)) if root_abs_arr.size else 0.0,
        root_p95_abs_error=float(np.percentile(root_abs_arr, 95.0)) if root_abs_arr.size else 0.0,
        schedule_spread_mean=float(np.mean(spreads_arr)) if spreads_arr.size else 0.0,
        schedule_spread_p95=float(np.percentile(spreads_arr, 95.0)) if spreads_arr.size else 0.0,
        leaf_mae=float(np.mean(leaf_abs_arr)) if leaf_abs_arr.size else 0.0,
        leaf_violation_rate=float(np.mean((leaf_abs_arr > tau).astype(np.float64)))
        if leaf_abs_arr.size
        else 0.0,
        merge_mae=float(np.mean(merge_abs_arr)) if merge_abs_arr.size else 0.0,
        merge_violation_rate=float(np.mean((merge_abs_arr > tau).astype(np.float64)))
        if merge_abs_arr.size
        else 0.0,
        n_docs=int(len(root_abs)),
    )

    topic_inf = TopicInferenceMetrics(
        leaf_accuracy_train=float(_leaf_accuracy(docs_train, topics_inferred=inferred_train)),
        leaf_accuracy_test=float(_leaf_accuracy(docs_test, topics_inferred=inferred_test)),
    )

    return RidgeRunMetrics(sketch=sketch, weights=weights, topic_inference=topic_inf, budget=budget, design=design)


def run_segment_lda_ops_weight_recovery_experiment(
    config: SegmentLDAOpsWeightRecoveryConfig,
) -> SegmentLDAOpsWeightRecoverySummary:
    # Validate config.
    if int(config.n_topics) < 2:
        raise ValueError("n_topics must be >= 2")
    if int(config.vocab_size) < 2:
        raise ValueError("vocab_size must be >= 2")
    if int(config.leaf_tokens) <= 0:
        raise ValueError("leaf_tokens must be positive")
    if int(config.train_docs) < 0 or int(config.test_docs) < 0:
        raise ValueError("train_docs/test_docs must be non-negative")
    if str(config.audit_policy) not in VALID_AUDIT_POLICIES:
        raise ValueError(f"audit_policy={config.audit_policy!r} unsupported; expected {VALID_AUDIT_POLICIES}")
    if str(config.audit_strategy) not in VALID_AUDIT_STRATEGIES:
        raise ValueError(
            f"audit_strategy={config.audit_strategy!r} unsupported; expected {VALID_AUDIT_STRATEGIES}"
        )
    if str(config.topic_source) not in VALID_TOPIC_SOURCES:
        raise ValueError(f"topic_source={config.topic_source!r} unsupported; expected {VALID_TOPIC_SOURCES}")
    if str(config.feature_inference) not in VALID_FEATURE_INFERENCE:
        raise ValueError(f"feature_inference={config.feature_inference!r} unsupported; expected {VALID_FEATURE_INFERENCE}")
    if str(config.topic_process) not in VALID_TOPIC_PROCESSES:
        raise ValueError(f"topic_process={config.topic_process!r} unsupported; expected {VALID_TOPIC_PROCESSES}")
    if str(config.boundary_profile) not in VALID_BOUNDARY_PROFILES:
        raise ValueError(
            f"boundary_profile={config.boundary_profile!r} unsupported; expected {VALID_BOUNDARY_PROFILES}"
        )
    if float(config.boundary_profile_strength) < 0.0:
        raise ValueError("boundary_profile_strength must be >= 0")
    if not math.isfinite(float(getattr(config, "segment_length_power", 0.0))):
        raise ValueError("segment_length_power must be finite")
    if float(config.audit_fraction) <= 0.0:
        raise ValueError("audit_fraction must be positive")
    if float(config.audit_scale) <= 0.0:
        raise ValueError("audit_scale must be positive")
    if int(config.audit_fixed_nodes) < 0:
        raise ValueError("audit_fixed_nodes must be non-negative")
    if float(config.lambda_multiplier) < 0.0:
        raise ValueError("lambda_multiplier must be >= 0")
    if float(getattr(config, "oracle_noise_std", 0.0)) < 0.0:
        raise ValueError("oracle_noise_std must be >= 0")
    est = str(config.topic_phi_estimator).strip().lower()
    if est not in VALID_TOPIC_PHI_ESTIMATORS:
        raise ValueError(
            f"topic_phi_estimator={config.topic_phi_estimator!r} unsupported; expected {VALID_TOPIC_PHI_ESTIMATORS}"
        )
    if not (0.0 < float(config.tlda_delta) and float(config.tlda_delta) < 1.0):
        raise ValueError("tlda_delta must be in (0,1)")
    if float(config.tlda_rate_constant) < 0.0 or not math.isfinite(float(config.tlda_rate_constant)):
        raise ValueError("tlda_rate_constant must be finite and >= 0")
    if float(config.tlda_sigmaK_floor) <= 0.0 or not math.isfinite(float(config.tlda_sigmaK_floor)):
        raise ValueError("tlda_sigmaK_floor must be finite and > 0")
    if int(getattr(config, "online_tensor_lda_burn_in_docs", 0)) < 0:
        raise ValueError("online_tensor_lda_burn_in_docs must be >= 0")
    if int(getattr(config, "online_tensor_lda_batch_docs", 1)) <= 0:
        raise ValueError("online_tensor_lda_batch_docs must be positive")
    if int(getattr(config, "online_tensor_lda_passes", 1)) <= 0:
        raise ValueError("online_tensor_lda_passes must be positive")
    if float(getattr(config, "online_tensor_lda_lr", 0.0)) <= 0.0 or not math.isfinite(
        float(getattr(config, "online_tensor_lda_lr", 0.0))
    ):
        raise ValueError("online_tensor_lda_lr must be finite and > 0")
    if float(getattr(config, "online_tensor_lda_grad_clip_norm", 0.0)) <= 0.0 or not math.isfinite(
        float(getattr(config, "online_tensor_lda_grad_clip_norm", 0.0))
    ):
        raise ValueError("online_tensor_lda_grad_clip_norm must be finite and > 0")
    if int(getattr(config, "embedding_topic_svd_dim_extra", 0)) < 0:
        raise ValueError("embedding_topic_svd_dim_extra must be >= 0")
    if int(getattr(config, "embedding_topic_kmeans_inits", 0)) <= 0:
        raise ValueError("embedding_topic_kmeans_inits must be positive")
    if int(getattr(config, "embedding_topic_kmeans_max_iter", 0)) <= 0:
        raise ValueError("embedding_topic_kmeans_max_iter must be positive")
    if float(getattr(config, "embedding_topic_assignment_temperature", 0.0)) <= 0.0:
        raise ValueError("embedding_topic_assignment_temperature must be > 0")
    if float(getattr(config, "embedding_topic_ppmi_shift", 0.0)) <= 0.0:
        raise ValueError("embedding_topic_ppmi_shift must be > 0")
    base_est = str(getattr(config, "neural_topic_base_estimator", "tensor_lda")).strip().lower()
    if base_est not in VALID_TOPIC_PHI_ESTIMATORS:
        raise ValueError(
            f"neural_topic_base_estimator={base_est!r} unsupported; expected {VALID_TOPIC_PHI_ESTIMATORS}"
        )
    if base_est.startswith("neural_"):
        raise ValueError("neural_topic_base_estimator must be a non-neural estimator")
    seed_frac = float(getattr(config, "neural_topic_seed_fraction", 0.35))
    if not math.isfinite(seed_frac) or not (0.0 < seed_frac <= 1.0):
        raise ValueError("neural_topic_seed_fraction must be in (0,1]")
    if int(getattr(config, "neural_topic_hidden_dim", 0)) <= 0:
        raise ValueError("neural_topic_hidden_dim must be positive")
    if int(getattr(config, "neural_topic_steps", 0)) <= 0:
        raise ValueError("neural_topic_steps must be positive")
    if float(getattr(config, "neural_topic_lr", 0.0)) <= 0.0:
        raise ValueError("neural_topic_lr must be > 0")
    if float(getattr(config, "neural_topic_weight_decay", 0.0)) < 0.0:
        raise ValueError("neural_topic_weight_decay must be >= 0")
    if int(getattr(config, "neural_topic_mix_samples", 0)) < 0:
        raise ValueError("neural_topic_mix_samples must be >= 0")
    if float(getattr(config, "neural_topic_mix_temperature", 0.0)) <= 0.0:
        raise ValueError("neural_topic_mix_temperature must be > 0")
    if float(getattr(config, "neural_topic_operator_boost", 0.0)) <= 0.0:
        raise ValueError("neural_topic_operator_boost must be > 0")
    w_min = float(getattr(config, "neural_topic_seed_llm_min_weight", 0.2))
    w_max = float(getattr(config, "neural_topic_seed_llm_max_weight", 0.55))
    if not (0.0 <= w_min <= 1.0 and 0.0 <= w_max <= 1.0 and w_min <= w_max):
        raise ValueError("neural_topic_seed_llm_min_weight/max_weight must satisfy 0<=min<=max<=1")
    if float(getattr(config, "neural_topic_similarity_temperature", 0.0)) <= 0.0:
        raise ValueError("neural_topic_similarity_temperature must be > 0")
    if float(getattr(config, "neural_topic_ridge", 0.0)) <= 0.0:
        raise ValueError("neural_topic_ridge must be > 0")

    # Sample topic distributions + weights.
    topics_phi, topic_meta = sample_topic_distributions(
        vocab_size=int(config.vocab_size),
        n_topics=int(config.n_topics),
        topic_concentration=float(config.topic_concentration),
        emission_mode=str(config.emission_mode),
        anchor_words_per_topic=int(config.anchor_words_per_topic),
        anchor_multiplier=float(config.anchor_multiplier),
        seed=int(_splitmix64(int(config.seed) + 101) & 0xFFFFFFFF),
    )
    R, theta_true, W_base = sample_sparse_oracle_weights(
        n_topics=int(config.n_topics),
        relevant_topics=int(config.relevant_topics),
        theta_scale=float(config.theta_scale),
        zero_diagonal=bool(config.zero_diagonal),
        seed=int(_splitmix64(int(config.seed) + 202) & 0xFFFFFFFF),
    )
    lambda_multiplier = float(config.lambda_multiplier)
    w_big_true = lambda_multiplier * W_base.reshape(-1).astype(np.float64, copy=False)

    # Generate train/test docs.
    boundary_profile_seed = int(config.boundary_profile_seed)
    if boundary_profile_seed < 0:
        boundary_profile_seed = int(_splitmix64(int(config.seed) + 19) & 0xFFFFFFFF)
    docs_train, train_stats = generate_segment_lda_docs(
        int(config.train_docs),
        topics=topics_phi,
        min_tokens=int(config.min_tokens),
        max_tokens=int(config.max_tokens),
        min_segments=int(config.min_segments),
        max_segments=int(config.max_segments),
        min_seg_len=int(config.min_seg_len),
        max_seg_len=int(config.max_seg_len),
        leaf_tokens=int(config.leaf_tokens),
        align_segments_to_leaves=bool(config.align_segments_to_leaves),
        doc_topic_concentration=float(config.doc_topic_concentration),
        topic_process=str(config.topic_process),
        boundary_profile=str(config.boundary_profile),
        boundary_profile_strength=float(config.boundary_profile_strength),
        boundary_profile_seed=int(boundary_profile_seed),
        segment_length_power=float(getattr(config, "segment_length_power", 0.0)),
        seed=int(_splitmix64(int(config.seed) + 7) & 0xFFFFFFFF),
    )
    docs_test, test_stats = generate_segment_lda_docs(
        int(config.test_docs),
        topics=topics_phi,
        min_tokens=int(config.min_tokens),
        max_tokens=int(config.max_tokens),
        min_segments=int(config.min_segments),
        max_segments=int(config.max_segments),
        min_seg_len=int(config.min_seg_len),
        max_seg_len=int(config.max_seg_len),
        leaf_tokens=int(config.leaf_tokens),
        align_segments_to_leaves=bool(config.align_segments_to_leaves),
        doc_topic_concentration=float(config.doc_topic_concentration),
        topic_process=str(config.topic_process),
        boundary_profile=str(config.boundary_profile),
        boundary_profile_strength=float(config.boundary_profile_strength),
        boundary_profile_seed=int(boundary_profile_seed),
        segment_length_power=float(getattr(config, "segment_length_power", 0.0)),
        seed=int(_splitmix64(int(config.seed) + 11) & 0xFFFFFFFF),
    )

    # Estimate topic-word distributions (Tensor-LDA-like upstream step).
    phi_docs_effective = int(config.topic_phi_docs) if int(config.topic_phi_docs) > 0 else int(config.train_docs)
    phi_docs_effective = int(max(0, phi_docs_effective))
    docs_phi: List[SegmentLDADoc] = list(docs_train)
    phi_from_train = int(min(len(docs_phi), phi_docs_effective))
    phi_extra = int(max(0, phi_docs_effective - len(docs_phi)))
    phi_extra_stats: Dict[str, float] = {}
    if phi_extra > 0:
        docs_extra, phi_extra_stats = generate_segment_lda_docs(
            phi_extra,
            topics=topics_phi,
            min_tokens=int(config.min_tokens),
            max_tokens=int(config.max_tokens),
            min_segments=int(config.min_segments),
            max_segments=int(config.max_segments),
            min_seg_len=int(config.min_seg_len),
            max_seg_len=int(config.max_seg_len),
            leaf_tokens=int(config.leaf_tokens),
            align_segments_to_leaves=bool(config.align_segments_to_leaves),
            doc_topic_concentration=float(config.doc_topic_concentration),
            topic_process=str(config.topic_process),
            boundary_profile=str(config.boundary_profile),
            boundary_profile_strength=float(config.boundary_profile_strength),
            boundary_profile_seed=int(boundary_profile_seed),
            segment_length_power=float(getattr(config, "segment_length_power", 0.0)),
            seed=int(_splitmix64(int(config.seed) + 23) & 0xFFFFFFFF),
        )
        docs_phi.extend(list(docs_extra))
    docs_phi = docs_phi[:phi_docs_effective]
    docs_phi_tokens = [d.tokens for d in docs_phi]

    topics_phi_est, topic_est_meta, perm_est_to_true = estimate_topic_distributions(
        topics_phi,
        estimator=str(config.topic_phi_estimator),
        n_docs=int(phi_docs_effective),
        doc_topic_concentration=float(config.doc_topic_concentration),
        tlda_delta=float(config.tlda_delta),
        tlda_rate_constant=float(config.tlda_rate_constant),
        sigmaK_floor=float(config.tlda_sigmaK_floor),
        permute=bool(config.topic_phi_permute),
        seed=int(_splitmix64(int(config.seed) + 404) & 0xFFFFFFFF),
        topic_word_concentration=float(config.topic_concentration),
        docs_tokens=docs_phi_tokens,
        sklearn_lda_max_iter=int(getattr(config, "sklearn_lda_max_iter", 60)),
        online_burn_in_docs=int(getattr(config, "online_tensor_lda_burn_in_docs", 0)),
        online_batch_docs=int(getattr(config, "online_tensor_lda_batch_docs", 32)),
        online_passes=int(getattr(config, "online_tensor_lda_passes", 1)),
        online_lr=float(getattr(config, "online_tensor_lda_lr", 0.1)),
        online_grad_clip_norm=float(getattr(config, "online_tensor_lda_grad_clip_norm", 1.0)),
        embedding_svd_dim_extra=int(getattr(config, "embedding_topic_svd_dim_extra", 4)),
        embedding_kmeans_inits=int(getattr(config, "embedding_topic_kmeans_inits", 8)),
        embedding_kmeans_max_iter=int(getattr(config, "embedding_topic_kmeans_max_iter", 80)),
        embedding_assignment_temperature=float(getattr(config, "embedding_topic_assignment_temperature", 0.35)),
        embedding_ppmi_shift=float(getattr(config, "embedding_topic_ppmi_shift", 1.0)),
        neural_base_estimator=str(getattr(config, "neural_topic_base_estimator", "tensor_lda")),
        neural_seed_fraction=float(getattr(config, "neural_topic_seed_fraction", 0.35)),
        neural_hidden_dim=int(getattr(config, "neural_topic_hidden_dim", 48)),
        neural_steps=int(getattr(config, "neural_topic_steps", 60)),
        neural_lr=float(getattr(config, "neural_topic_lr", 3e-3)),
        neural_weight_decay=float(getattr(config, "neural_topic_weight_decay", 1e-4)),
        neural_mix_samples=int(getattr(config, "neural_topic_mix_samples", 128)),
        neural_mix_temperature=float(getattr(config, "neural_topic_mix_temperature", 1.0)),
        neural_operator_boost=float(getattr(config, "neural_topic_operator_boost", 1.4)),
        neural_seed_min_weight=float(getattr(config, "neural_topic_seed_llm_min_weight", 0.2)),
        neural_seed_max_weight=float(getattr(config, "neural_topic_seed_llm_max_weight", 0.55)),
        neural_similarity_temperature=float(getattr(config, "neural_topic_similarity_temperature", 0.15)),
        neural_ridge=float(getattr(config, "neural_topic_ridge", 1e-3)),
    )

    # Baseline sketch families on true topics (Lean-aligned checks).
    exact = _eval_sketch_family(
        docs_test,
        theta_true=theta_true,
        w_big_true=w_big_true,
        leaf_tokens=int(config.leaf_tokens),
        tau=float(config.violation_tau),
        kind="exact",
    )
    undersupported = _eval_sketch_family(
        docs_test,
        theta_true=theta_true,
        w_big_true=w_big_true,
        leaf_tokens=int(config.leaf_tokens),
        tau=float(config.violation_tau),
        kind="undersupported",
    )
    flip_r1 = _eval_sketch_family(
        docs_test,
        theta_true=theta_true,
        w_big_true=w_big_true,
        leaf_tokens=int(config.leaf_tokens),
        tau=float(config.violation_tau),
        kind="flip",
        rounds=1,
    )
    flip_r2 = _eval_sketch_family(
        docs_test,
        theta_true=theta_true,
        w_big_true=w_big_true,
        leaf_tokens=int(config.leaf_tokens),
        tau=float(config.violation_tau),
        kind="flip",
        rounds=2,
    )

    # Main ridge run matches the configured "feature source" and (if needed) uses estimated topics.
    topics_phi_for_ridge = topics_phi_est if str(config.topic_source) == "infer" else topics_phi
    perm_for_ridge = perm_est_to_true if str(config.topic_source) == "infer" else None
    ridge = _compute_ridge_metrics(
        docs_train=docs_train,
        docs_test=docs_test,
        topics_phi=topics_phi_for_ridge,
        perm_est_to_true=perm_for_ridge,
        theta_true=theta_true,
        W_base=W_base,
        lambda_multiplier=lambda_multiplier,
        config=config,
    )

    ridge_true_topics: Optional[RidgeRunMetrics] = None
    ridge_infer_true_phi: Optional[RidgeRunMetrics] = None
    ridge_infer_est_phi: Optional[RidgeRunMetrics] = None
    if bool(getattr(config, "run_all_feature_modes", False)):
        cfg_true = replace(config, topic_source="true")
        ridge_true_topics = _compute_ridge_metrics(
            docs_train=docs_train,
            docs_test=docs_test,
            topics_phi=topics_phi,
            perm_est_to_true=None,
            theta_true=theta_true,
            W_base=W_base,
            lambda_multiplier=lambda_multiplier,
            config=cfg_true,
        )

        cfg_infer = replace(config, topic_source="infer")
        ridge_infer_true_phi = _compute_ridge_metrics(
            docs_train=docs_train,
            docs_test=docs_test,
            topics_phi=topics_phi,
            perm_est_to_true=None,
            theta_true=theta_true,
            W_base=W_base,
            lambda_multiplier=lambda_multiplier,
            config=cfg_infer,
        )
        ridge_infer_est_phi = _compute_ridge_metrics(
            docs_train=docs_train,
            docs_test=docs_test,
            topics_phi=topics_phi_est,
            perm_est_to_true=perm_est_to_true,
            theta_true=theta_true,
            W_base=W_base,
            lambda_multiplier=lambda_multiplier,
            config=cfg_infer,
        )

    geom = _training_geometry(
        docs_train,
        leaf_tokens=int(config.leaf_tokens),
        audit_policy=str(config.audit_policy),
        audit_fixed_nodes=int(config.audit_fixed_nodes),
        audit_fraction=float(config.audit_fraction),
        audit_scale=float(config.audit_scale),
        oracle_cost_power=float(config.oracle_cost_power),
        oracle_cost_per_query=float(config.oracle_cost_per_query),
    )

    weight_truth = {
        "relevant_topics": list(R),
        "theta_true": theta_true.tolist(),
        "W_base_fro_norm": float(np.linalg.norm(W_base.reshape(-1))),
        "W_base": W_base.reshape(-1).tolist(),
        "lambda_multiplier": float(lambda_multiplier),
    }
    metrics = {
        "exact": asdict(exact),
        "undersupported": asdict(undersupported),
        "flip_R1": asdict(flip_r1),
        "flip_R2": asdict(flip_r2),
        "ridge": {
            **asdict(ridge.sketch),
            **asdict(ridge.weights),
            **asdict(ridge.topic_inference),
            **asdict(ridge.budget),
            **asdict(ridge.design),
        },
    }
    if ridge_true_topics is not None:
        metrics["ridge_true_topics"] = {
            **asdict(ridge_true_topics.sketch),
            **asdict(ridge_true_topics.weights),
            **asdict(ridge_true_topics.topic_inference),
            **asdict(ridge_true_topics.budget),
            **asdict(ridge_true_topics.design),
        }
    if ridge_infer_true_phi is not None:
        metrics["ridge_infer_true_phi"] = {
            **asdict(ridge_infer_true_phi.sketch),
            **asdict(ridge_infer_true_phi.weights),
            **asdict(ridge_infer_true_phi.topic_inference),
            **asdict(ridge_infer_true_phi.budget),
            **asdict(ridge_infer_true_phi.design),
        }
    if ridge_infer_est_phi is not None:
        metrics["ridge_infer_est_phi"] = {
            **asdict(ridge_infer_est_phi.sketch),
            **asdict(ridge_infer_est_phi.weights),
            **asdict(ridge_infer_est_phi.topic_inference),
            **asdict(ridge_infer_est_phi.budget),
            **asdict(ridge_infer_est_phi.design),
        }

    return SegmentLDAOpsWeightRecoverySummary(
        config=asdict(config),
        topic_meta={
            **topic_meta,
            **topic_est_meta,
            "topic_phi_perm_est_to_true": list(int(x) for x in perm_est_to_true),
            "topic_phi_docs_target": float(phi_docs_effective),
            "topic_phi_docs_from_train": float(phi_from_train),
            "topic_phi_docs_extra": float(phi_extra),
            **{f"topic_phi_extra_{k}": v for k, v in phi_extra_stats.items()},
            **train_stats,
            **{f"test_{k}": v for k, v in test_stats.items()},
        },
        weight_truth=weight_truth,
        training_geometry=asdict(geom),
        metrics=metrics,
    )


__all__ = [
    "SegmentLDAOpsWeightRecoveryConfig",
    "SegmentLDAOpsWeightRecoverySummary",
    "VALID_AUDIT_POLICIES",
    "VALID_AUDIT_STRATEGIES",
    "VALID_BOUNDARY_PROFILES",
    "VALID_FEATURE_INFERENCE",
    "NEURAL_TOPIC_PHI_ESTIMATORS",
    "VALID_TOPIC_PHI_ESTIMATORS",
    "VALID_TOPIC_SOURCES",
    "VALID_TOPIC_PROCESSES",
    "VALID_SCHEDULES",
    "audit_sample_count",
    "estimate_topic_distributions",
    "run_segment_lda_ops_weight_recovery_experiment",
]
