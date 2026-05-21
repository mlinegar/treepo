"""
Exact bag-of-words LDA recovery through mergeable tree count sketches.

This module implements the simulation family described in:

    docs/lda_tree_recovery_simulation_spec.md

The clean base case is:

1. sample ordinary bag-of-words LDA documents,
2. represent each leaf by an exact word-count vector,
3. merge leaf counts by addition,
4. run the same document-level operator on the merged root counts as on the full document.

That gives an exact positive control for the tree architecture. The same module also reports
two deliberately weaker leafwise baselines:

- `leaf_average`: infer a topic mixture independently in each leaf and average the leaf posteriors;
- `leaf_utility_only`: estimate utility separately inside each leaf and average utilities.

Those leafwise baselines are useful negative controls under ordinary bag-of-words LDA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from treepo._research.ctreepo.sim.objective_semantics import latent_quadratic_utility_objective_semantics

from treepo._research.ctreepo.sim.core.segment_lda_ops_weight_recovery import (
    _splitmix64,
    build_leaf_spans,
    sample_sparse_oracle_weights,
    sample_topic_distributions,
)


VALID_EMISSION_MODES: Tuple[str, ...] = ("anchored", "disjoint")
VALID_SCHEDULES: Tuple[str, ...] = ("balanced", "left_to_right", "right_to_left")


@dataclass(frozen=True)
class LDATreeRecoveryConfig:
    # LDA DGP.
    n_topics: int = 8
    vocab_size: int = 512
    min_tokens: int = 384
    max_tokens: int = 384
    doc_topic_concentration: float = 0.6

    # Topic-word distributions.
    topic_concentration: float = 0.2
    emission_mode: str = "anchored"
    anchor_words_per_topic: int = 20
    anchor_multiplier: float = 25.0

    # Document-level utility on inferred topic mixtures.
    # `lambda_multiplier` is a legacy internal name for the quadratic utility
    # weight. It is not the paper-facing local-law lambda.
    relevant_topics: int = 2
    theta_scale: float = 1.0
    zero_diagonal: bool = False
    lambda_multiplier: float = 1.0

    # Tree geometry.
    leaf_tokens: int = 16

    # Fixed-world sizes.
    train_docs: int = 0
    test_docs: int = 1024

    # Document-level mixture inference with known topics.
    inference_prior_mass: float = 0.25
    inference_max_iter: int = 200
    inference_tol: float = 1e-9

    seed: int = 0


@dataclass(frozen=True)
class LDATreeRecoveryDoc:
    tokens: Tuple[int, ...]
    topics: Tuple[int, ...]
    topic_weights: Tuple[float, ...]


@dataclass(frozen=True)
class ExactRecoveryMetrics:
    n_docs: int
    root_count_l1_mean: float
    root_count_l2_mean: float
    root_pi_l1_mean: float
    root_pi_l2_mean: float
    root_utility_abs_mean: float
    root_loglik_abs_mean: float
    schedule_count_l1_spread_mean: float
    schedule_pi_l1_spread_mean: float
    schedule_utility_spread_mean: float
    schedule_loglik_spread_mean: float


@dataclass(frozen=True)
class MethodMetrics:
    n_docs: int
    pi_l1_to_true_mean: float
    pi_l1_to_true_median: float
    pi_l1_to_true_p95: float
    pi_l1_to_full_mean: float
    pi_l1_to_full_median: float
    pi_l1_to_full_p95: float
    utility_abs_to_true_mean: float
    utility_abs_to_true_median: float
    utility_abs_to_true_p95: float
    utility_abs_to_full_mean: float
    utility_abs_to_full_median: float
    utility_abs_to_full_p95: float
    log_likelihood_mean: float
    log_likelihood_abs_to_full_mean: float
    log_likelihood_abs_to_full_median: float
    log_likelihood_abs_to_full_p95: float


@dataclass(frozen=True)
class LDATreeRecoveryWorld:
    signature: Dict[str, object]
    topic_meta: Dict[str, object]
    topics_phi: Tuple[np.ndarray, ...]
    relevant_topics: Tuple[int, ...]
    theta_true: np.ndarray
    W_base: np.ndarray
    docs_train: Tuple[LDATreeRecoveryDoc, ...]
    docs_test: Tuple[LDATreeRecoveryDoc, ...]


@dataclass(frozen=True)
class LDATreeRecoverySummary:
    config: Dict[str, object]
    topic_meta: Dict[str, object]
    utility_truth: Dict[str, object]
    world_stats: Dict[str, object]
    exact_recovery: Dict[str, object]
    methods: Dict[str, object]
    objective: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "topic_meta": self.topic_meta,
            "utility_truth": self.utility_truth,
            "world_stats": self.world_stats,
            "exact_recovery": self.exact_recovery,
            "methods": self.methods,
            "objective": self.objective,
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def _validate_config(config: LDATreeRecoveryConfig) -> None:
    if int(config.n_topics) < 2:
        raise ValueError("n_topics must be >= 2")
    if int(config.vocab_size) < 2:
        raise ValueError("vocab_size must be >= 2")
    if int(config.min_tokens) < 2 or int(config.max_tokens) < int(config.min_tokens):
        raise ValueError("require 2 <= min_tokens <= max_tokens")
    if float(config.doc_topic_concentration) <= 0.0:
        raise ValueError("doc_topic_concentration must be > 0")
    if float(config.topic_concentration) <= 0.0:
        raise ValueError("topic_concentration must be > 0")
    if str(config.emission_mode).strip().lower() not in VALID_EMISSION_MODES:
        raise ValueError(f"emission_mode must be one of {VALID_EMISSION_MODES}")
    if int(config.anchor_words_per_topic) <= 0:
        raise ValueError("anchor_words_per_topic must be positive")
    if float(config.anchor_multiplier) <= 1.0:
        raise ValueError("anchor_multiplier must be > 1")
    if int(config.leaf_tokens) <= 0:
        raise ValueError("leaf_tokens must be positive")
    if int(config.train_docs) < 0 or int(config.test_docs) < 0:
        raise ValueError("train_docs/test_docs must be non-negative")
    if float(config.inference_prior_mass) < 0.0:
        raise ValueError("inference_prior_mass must be >= 0")
    if int(config.inference_max_iter) <= 0:
        raise ValueError("inference_max_iter must be positive")
    if float(config.inference_tol) < 0.0:
        raise ValueError("inference_tol must be >= 0")


def _world_signature(config: LDATreeRecoveryConfig) -> Dict[str, object]:
    return {
        "problem_id": "lda_tree_recovery",
        "n_topics": int(config.n_topics),
        "vocab_size": int(config.vocab_size),
        "min_tokens": int(config.min_tokens),
        "max_tokens": int(config.max_tokens),
        "doc_topic_concentration": float(config.doc_topic_concentration),
        "topic_concentration": float(config.topic_concentration),
        "emission_mode": str(config.emission_mode),
        "anchor_words_per_topic": int(config.anchor_words_per_topic),
        "anchor_multiplier": float(config.anchor_multiplier),
        "relevant_topics": int(config.relevant_topics),
        "theta_scale": float(config.theta_scale),
        "zero_diagonal": bool(config.zero_diagonal),
        "seed": int(config.seed),
    }


def lda_tree_recovery_world_cache_signature(
    config: LDATreeRecoveryConfig,
    *,
    train_docs_capacity: int,
    test_docs_capacity: int,
) -> Dict[str, object]:
    return {
        **_world_signature(config),
        "train_docs_capacity": int(train_docs_capacity),
        "test_docs_capacity": int(test_docs_capacity),
    }


def _safe_stat(xs: Sequence[float], *, kind: str) -> float:
    arr = np.asarray([float(x) for x in xs if math.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if kind == "mean":
        return float(np.mean(arr))
    if kind == "median":
        return float(np.median(arr))
    if kind == "p95":
        return float(np.percentile(arr, 95.0))
    raise ValueError(f"unsupported stat kind: {kind!r}")


def _counts_from_tokens(tokens: Sequence[int], *, vocab_size: int) -> np.ndarray:
    toks = np.asarray(tokens, dtype=np.int64)
    if toks.size == 0:
        return np.zeros((int(vocab_size),), dtype=np.float64)
    return np.bincount(toks, minlength=int(vocab_size)).astype(np.float64, copy=False)


def _generate_bag_of_words_docs(
    n_docs: int,
    *,
    topics_phi: Sequence[np.ndarray],
    min_tokens: int,
    max_tokens: int,
    doc_topic_concentration: float,
    seed: int,
) -> Tuple[Tuple[LDATreeRecoveryDoc, ...], Dict[str, float]]:
    rng = np.random.default_rng(int(seed))
    k = int(len(topics_phi))
    v = int(np.asarray(topics_phi[0], dtype=np.float64).size)

    docs: List[LDATreeRecoveryDoc] = []
    tok_lens: List[int] = []
    topic_entropies: List[float] = []
    for _ in range(int(n_docs)):
        n = int(rng.integers(int(min_tokens), int(max_tokens) + 1))
        pi = rng.dirichlet(np.full((k,), float(doc_topic_concentration), dtype=np.float64))
        z = rng.choice(k, size=n, replace=True, p=pi).astype(np.int64, copy=False)
        x = np.zeros((n,), dtype=np.int64)
        for topic_id in range(k):
            idx = np.flatnonzero(z == int(topic_id))
            if idx.size == 0:
                continue
            x[idx] = rng.choice(v, size=int(idx.size), replace=True, p=np.asarray(topics_phi[topic_id], dtype=np.float64))
        docs.append(
            LDATreeRecoveryDoc(
                tokens=tuple(int(t) for t in x.tolist()),
                topics=tuple(int(t) for t in z.tolist()),
                topic_weights=tuple(float(t) for t in pi.tolist()),
            )
        )
        tok_lens.append(int(n))
        topic_entropies.append(float(-np.sum(pi * np.log(np.clip(pi, 1e-12, 1.0)))))

    stats = {
        "mean_tokens": _safe_stat(tok_lens, kind="mean") if tok_lens else 0.0,
        "mean_topic_entropy": _safe_stat(topic_entropies, kind="mean") if topic_entropies else 0.0,
    }
    return (tuple(docs), stats)


def _summarize_docs(
    docs: Sequence[LDATreeRecoveryDoc],
    *,
    leaf_tokens: int,
) -> Dict[str, float]:
    n_tokens: List[float] = []
    n_leaves: List[float] = []
    leaf_purities: List[float] = []
    for doc in docs:
        n_tok = int(len(doc.tokens))
        leaves = build_leaf_spans(n_tok, leaf_tokens=int(leaf_tokens))
        z = np.asarray(doc.topics, dtype=np.int64)
        n_tokens.append(float(n_tok))
        n_leaves.append(float(len(leaves)))
        for start, end in leaves:
            seg = z[int(start) : int(end)]
            if seg.size == 0:
                continue
            _uniq, counts = np.unique(seg, return_counts=True)
            leaf_purities.append(float(np.max(counts)) / float(seg.size))

    return {
        "mean_tokens": _safe_stat(n_tokens, kind="mean") if n_tokens else 0.0,
        "mean_leaves": _safe_stat(n_leaves, kind="mean") if n_leaves else 0.0,
        "mean_leaf_topic_purity": _safe_stat(leaf_purities, kind="mean") if leaf_purities else 0.0,
    }


def _utility_from_pi(
    pi: np.ndarray,
    *,
    theta: np.ndarray,
    W_base: np.ndarray,
    lambda_multiplier: float,
) -> float:
    x = np.asarray(pi, dtype=np.float64).reshape(-1)
    return float(np.dot(theta, x) + float(lambda_multiplier) * float(x @ W_base @ x))


def _doc_log_likelihood(
    counts: np.ndarray,
    *,
    pi: np.ndarray,
    topics_phi: Sequence[np.ndarray],
) -> float:
    c = np.asarray(counts, dtype=np.float64).reshape(-1)
    if c.size == 0 or float(np.sum(c)) <= 0.0:
        return 0.0
    phi = np.stack([np.asarray(t, dtype=np.float64).reshape(-1) for t in topics_phi], axis=0)
    probs = np.clip(np.asarray(pi, dtype=np.float64) @ phi, 1e-12, 1.0)
    return float(np.dot(c, np.log(probs)))


def _infer_topic_mixture_from_counts(
    counts: np.ndarray,
    *,
    topics_phi: Sequence[np.ndarray],
    prior_mass: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    c = np.asarray(counts, dtype=np.float64).reshape(-1)
    k = int(len(topics_phi))
    total = float(np.sum(c))
    if c.size == 0 or total <= 0.0:
        return np.full((k,), 1.0 / float(k), dtype=np.float64)

    phi = np.stack([np.asarray(t, dtype=np.float64).reshape(-1) for t in topics_phi], axis=0)
    obs = np.flatnonzero(c > 0.0)
    if obs.size == 0:
        return np.full((k,), 1.0 / float(k), dtype=np.float64)
    phi_obs = phi[:, obs]
    counts_obs = c[obs]

    pi = np.full((k,), 1.0 / float(k), dtype=np.float64)
    pmass = float(max(0.0, prior_mass))
    for _ in range(int(max_iter)):
        weighted = np.clip(pi[:, None] * phi_obs, 1e-18, None)
        denom = np.clip(np.sum(weighted, axis=0, keepdims=True), 1e-18, None)
        resp = weighted / denom
        expected_topics = resp @ counts_obs
        pi_new = expected_topics + pmass
        z = float(np.sum(pi_new))
        if not math.isfinite(z) or z <= 0.0:
            pi_new = np.full((k,), 1.0 / float(k), dtype=np.float64)
        else:
            pi_new = pi_new / z
        if float(np.max(np.abs(pi_new - pi))) <= float(tol):
            pi = pi_new
            break
        pi = pi_new
    return pi.astype(np.float64, copy=False)


def _reduce_counts(leaf_counts: Sequence[np.ndarray], *, schedule: str, vocab_size: int) -> np.ndarray:
    if len(leaf_counts) == 0:
        return np.zeros((int(vocab_size),), dtype=np.float64)

    sch = str(schedule).strip().lower()
    if sch == "left_to_right":
        out = np.zeros((int(vocab_size),), dtype=np.float64)
        for x in leaf_counts:
            out = out + np.asarray(x, dtype=np.float64)
        return out
    if sch == "right_to_left":
        out = np.zeros((int(vocab_size),), dtype=np.float64)
        for x in reversed(list(leaf_counts)):
            out = out + np.asarray(x, dtype=np.float64)
        return out
    if sch == "balanced":
        cur = [np.asarray(x, dtype=np.float64) for x in leaf_counts]
        while len(cur) > 1:
            nxt: List[np.ndarray] = []
            i = 0
            while i < len(cur):
                if i + 1 < len(cur):
                    nxt.append(cur[i] + cur[i + 1])
                    i += 2
                else:
                    nxt.append(cur[i])
                    i += 1
            cur = nxt
        return cur[0]
    raise ValueError(f"unsupported schedule: {schedule!r}; expected one of {VALID_SCHEDULES}")


def _pairwise_spread(xs: Sequence[np.ndarray], *, metric: str) -> float:
    if len(xs) <= 1:
        return 0.0
    vals: List[float] = []
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if metric == "l1":
                vals.append(float(np.sum(np.abs(np.asarray(xs[i]) - np.asarray(xs[j])))))
            elif metric == "abs":
                vals.append(abs(float(xs[i]) - float(xs[j])))
            else:
                raise ValueError(f"unsupported spread metric: {metric!r}")
    return max(vals) if vals else 0.0


def sample_lda_tree_recovery_world(
    config: LDATreeRecoveryConfig,
    *,
    train_docs_capacity: Optional[int] = None,
    test_docs_capacity: Optional[int] = None,
) -> LDATreeRecoveryWorld:
    _validate_config(config)
    train_cap = int(config.train_docs if train_docs_capacity is None else train_docs_capacity)
    test_cap = int(config.test_docs if test_docs_capacity is None else test_docs_capacity)
    if train_cap < 0 or test_cap < 0:
        raise ValueError("train_docs_capacity/test_docs_capacity must be non-negative")

    topics_phi, topic_meta = sample_topic_distributions(
        vocab_size=int(config.vocab_size),
        n_topics=int(config.n_topics),
        topic_concentration=float(config.topic_concentration),
        emission_mode=str(config.emission_mode),
        anchor_words_per_topic=int(config.anchor_words_per_topic),
        anchor_multiplier=float(config.anchor_multiplier),
        seed=int(_splitmix64(int(config.seed) + 101) & 0xFFFFFFFF),
    )
    relevant_topics, theta_true, W_base = sample_sparse_oracle_weights(
        n_topics=int(config.n_topics),
        relevant_topics=int(config.relevant_topics),
        theta_scale=float(config.theta_scale),
        zero_diagonal=bool(config.zero_diagonal),
        seed=int(_splitmix64(int(config.seed) + 202) & 0xFFFFFFFF),
    )

    docs_train, _train_stats = _generate_bag_of_words_docs(
        train_cap,
        topics_phi=topics_phi,
        min_tokens=int(config.min_tokens),
        max_tokens=int(config.max_tokens),
        doc_topic_concentration=float(config.doc_topic_concentration),
        seed=int(_splitmix64(int(config.seed) + 7) & 0xFFFFFFFF),
    )
    docs_test, _test_stats = _generate_bag_of_words_docs(
        test_cap,
        topics_phi=topics_phi,
        min_tokens=int(config.min_tokens),
        max_tokens=int(config.max_tokens),
        doc_topic_concentration=float(config.doc_topic_concentration),
        seed=int(_splitmix64(int(config.seed) + 11) & 0xFFFFFFFF),
    )

    return LDATreeRecoveryWorld(
        signature=_world_signature(config),
        topic_meta=dict(topic_meta),
        topics_phi=tuple(np.asarray(t, dtype=np.float64).copy() for t in topics_phi),
        relevant_topics=tuple(int(x) for x in relevant_topics),
        theta_true=np.asarray(theta_true, dtype=np.float64).copy(),
        W_base=np.asarray(W_base, dtype=np.float64).copy(),
        docs_train=tuple(docs_train),
        docs_test=tuple(docs_test),
    )


def _method_metrics(
    *,
    pi_to_true: Sequence[float],
    pi_to_full: Sequence[float],
    utility_to_true: Sequence[float],
    utility_to_full: Sequence[float],
    log_likelihoods: Sequence[float],
    loglik_to_full: Sequence[float],
    n_docs: int,
) -> MethodMetrics:
    return MethodMetrics(
        n_docs=int(n_docs),
        pi_l1_to_true_mean=_safe_stat(pi_to_true, kind="mean"),
        pi_l1_to_true_median=_safe_stat(pi_to_true, kind="median"),
        pi_l1_to_true_p95=_safe_stat(pi_to_true, kind="p95"),
        pi_l1_to_full_mean=_safe_stat(pi_to_full, kind="mean"),
        pi_l1_to_full_median=_safe_stat(pi_to_full, kind="median"),
        pi_l1_to_full_p95=_safe_stat(pi_to_full, kind="p95"),
        utility_abs_to_true_mean=_safe_stat(utility_to_true, kind="mean"),
        utility_abs_to_true_median=_safe_stat(utility_to_true, kind="median"),
        utility_abs_to_true_p95=_safe_stat(utility_to_true, kind="p95"),
        utility_abs_to_full_mean=_safe_stat(utility_to_full, kind="mean"),
        utility_abs_to_full_median=_safe_stat(utility_to_full, kind="median"),
        utility_abs_to_full_p95=_safe_stat(utility_to_full, kind="p95"),
        log_likelihood_mean=_safe_stat(log_likelihoods, kind="mean"),
        log_likelihood_abs_to_full_mean=_safe_stat(loglik_to_full, kind="mean"),
        log_likelihood_abs_to_full_median=_safe_stat(loglik_to_full, kind="median"),
        log_likelihood_abs_to_full_p95=_safe_stat(loglik_to_full, kind="p95"),
    )


def run_lda_tree_recovery_experiment_from_world(
    config: LDATreeRecoveryConfig,
    world: LDATreeRecoveryWorld,
) -> LDATreeRecoverySummary:
    _validate_config(config)
    if dict(world.signature) != _world_signature(config):
        raise ValueError("config is incompatible with the provided fixed world")
    if int(config.train_docs) > len(world.docs_train):
        raise ValueError("config.train_docs exceeds fixed world train_docs capacity")
    if int(config.test_docs) > len(world.docs_test):
        raise ValueError("config.test_docs exceeds fixed world test_docs capacity")

    docs_test = tuple(world.docs_test[: int(config.test_docs)])
    topics_phi = tuple(np.asarray(t, dtype=np.float64) for t in world.topics_phi)
    theta_true = np.asarray(world.theta_true, dtype=np.float64)
    W_base = np.asarray(world.W_base, dtype=np.float64)

    exact_count_l1: List[float] = []
    exact_count_l2: List[float] = []
    exact_pi_l1: List[float] = []
    exact_pi_l2: List[float] = []
    exact_utility_abs: List[float] = []
    exact_loglik_abs: List[float] = []
    sched_count_spread: List[float] = []
    sched_pi_spread: List[float] = []
    sched_util_spread: List[float] = []
    sched_loglik_spread: List[float] = []

    full_pi_true: List[float] = []
    full_pi_full: List[float] = []
    full_util_true: List[float] = []
    full_util_full: List[float] = []
    full_logliks: List[float] = []
    full_loglik_full: List[float] = []

    tree_pi_true: List[float] = []
    tree_pi_full: List[float] = []
    tree_util_true: List[float] = []
    tree_util_full: List[float] = []
    tree_logliks: List[float] = []
    tree_loglik_full: List[float] = []

    leaf_avg_pi_true: List[float] = []
    leaf_avg_pi_full: List[float] = []
    leaf_avg_util_true: List[float] = []
    leaf_avg_util_full: List[float] = []
    leaf_avg_logliks: List[float] = []
    leaf_avg_loglik_full: List[float] = []

    leaf_u_pi_true: List[float] = []
    leaf_u_pi_full: List[float] = []
    leaf_u_util_true: List[float] = []
    leaf_u_util_full: List[float] = []
    leaf_u_logliks: List[float] = []
    leaf_u_loglik_full: List[float] = []

    for doc in docs_test:
        counts_full = _counts_from_tokens(doc.tokens, vocab_size=int(config.vocab_size))
        pi_true = np.asarray(doc.topic_weights, dtype=np.float64)
        utility_true = _utility_from_pi(
            pi_true,
            theta=theta_true,
            W_base=W_base,
            lambda_multiplier=float(config.lambda_multiplier),
        )

        pi_full = _infer_topic_mixture_from_counts(
            counts_full,
            topics_phi=topics_phi,
            prior_mass=float(config.inference_prior_mass),
            max_iter=int(config.inference_max_iter),
            tol=float(config.inference_tol),
        )
        utility_full = _utility_from_pi(
            pi_full,
            theta=theta_true,
            W_base=W_base,
            lambda_multiplier=float(config.lambda_multiplier),
        )
        loglik_full = _doc_log_likelihood(counts_full, pi=pi_full, topics_phi=topics_phi)

        leaves = build_leaf_spans(len(doc.tokens), leaf_tokens=int(config.leaf_tokens))
        leaf_counts = [
            _counts_from_tokens(doc.tokens[int(start) : int(end)], vocab_size=int(config.vocab_size))
            for start, end in leaves
        ]
        leaf_lens = np.asarray([max(0, int(end) - int(start)) for start, end in leaves], dtype=np.float64)
        total_leaf_len = float(np.sum(leaf_lens))
        leaf_weights = leaf_lens / total_leaf_len if total_leaf_len > 0.0 else np.zeros_like(leaf_lens)

        root_counts_by_schedule: Dict[str, np.ndarray] = {}
        root_pi_by_schedule: Dict[str, np.ndarray] = {}
        root_utility_by_schedule: Dict[str, float] = {}
        root_loglik_by_schedule: Dict[str, float] = {}
        for schedule in VALID_SCHEDULES:
            root_counts = _reduce_counts(leaf_counts, schedule=schedule, vocab_size=int(config.vocab_size))
            root_counts_by_schedule[str(schedule)] = root_counts
            pi_sched = _infer_topic_mixture_from_counts(
                root_counts,
                topics_phi=topics_phi,
                prior_mass=float(config.inference_prior_mass),
                max_iter=int(config.inference_max_iter),
                tol=float(config.inference_tol),
            )
            root_pi_by_schedule[str(schedule)] = pi_sched
            root_utility_by_schedule[str(schedule)] = _utility_from_pi(
                pi_sched,
                theta=theta_true,
                W_base=W_base,
                lambda_multiplier=float(config.lambda_multiplier),
            )
            root_loglik_by_schedule[str(schedule)] = _doc_log_likelihood(counts_full, pi=pi_sched, topics_phi=topics_phi)

        counts_tree = root_counts_by_schedule["balanced"]
        pi_tree = root_pi_by_schedule["balanced"]
        utility_tree = root_utility_by_schedule["balanced"]
        loglik_tree = root_loglik_by_schedule["balanced"]

        leaf_posteriors = [
            _infer_topic_mixture_from_counts(
                c,
                topics_phi=topics_phi,
                prior_mass=float(config.inference_prior_mass),
                max_iter=int(config.inference_max_iter),
                tol=float(config.inference_tol),
            )
            for c in leaf_counts
        ]
        if leaf_posteriors:
            leaf_avg = np.sum(
                np.stack(leaf_posteriors, axis=0) * leaf_weights[:, None],
                axis=0,
            )
            s = float(np.sum(leaf_avg))
            if s > 0.0:
                leaf_avg = leaf_avg / s
        else:
            leaf_avg = np.full((int(config.n_topics),), 1.0 / float(int(config.n_topics)), dtype=np.float64)
        leaf_avg_utility = _utility_from_pi(
            leaf_avg,
            theta=theta_true,
            W_base=W_base,
            lambda_multiplier=float(config.lambda_multiplier),
        )
        leaf_avg_loglik = _doc_log_likelihood(counts_full, pi=leaf_avg, topics_phi=topics_phi)
        leaf_utility_only = float(
            np.sum(
                leaf_weights
                * np.asarray(
                    [
                        _utility_from_pi(
                            p,
                            theta=theta_true,
                            W_base=W_base,
                            lambda_multiplier=float(config.lambda_multiplier),
                        )
                        for p in leaf_posteriors
                    ],
                    dtype=np.float64,
                )
            )
        ) if leaf_posteriors else 0.0

        exact_count_l1.append(float(np.sum(np.abs(counts_tree - counts_full))))
        exact_count_l2.append(float(np.sqrt(np.sum((counts_tree - counts_full) ** 2))))
        exact_pi_l1.append(float(np.sum(np.abs(pi_tree - pi_full))))
        exact_pi_l2.append(float(np.sqrt(np.sum((pi_tree - pi_full) ** 2))))
        exact_utility_abs.append(abs(float(utility_tree) - float(utility_full)))
        exact_loglik_abs.append(abs(float(loglik_tree) - float(loglik_full)))

        sched_count_spread.append(_pairwise_spread(list(root_counts_by_schedule.values()), metric="l1"))
        sched_pi_spread.append(_pairwise_spread(list(root_pi_by_schedule.values()), metric="l1"))
        sched_util_spread.append(_pairwise_spread(list(root_utility_by_schedule.values()), metric="abs"))
        sched_loglik_spread.append(_pairwise_spread(list(root_loglik_by_schedule.values()), metric="abs"))

        full_pi_true.append(float(np.sum(np.abs(pi_full - pi_true))))
        full_pi_full.append(0.0)
        full_util_true.append(abs(float(utility_full) - float(utility_true)))
        full_util_full.append(0.0)
        full_logliks.append(float(loglik_full))
        full_loglik_full.append(0.0)

        tree_pi_true.append(float(np.sum(np.abs(pi_tree - pi_true))))
        tree_pi_full.append(float(np.sum(np.abs(pi_tree - pi_full))))
        tree_util_true.append(abs(float(utility_tree) - float(utility_true)))
        tree_util_full.append(abs(float(utility_tree) - float(utility_full)))
        tree_logliks.append(float(loglik_tree))
        tree_loglik_full.append(abs(float(loglik_tree) - float(loglik_full)))

        leaf_avg_pi_true.append(float(np.sum(np.abs(leaf_avg - pi_true))))
        leaf_avg_pi_full.append(float(np.sum(np.abs(leaf_avg - pi_full))))
        leaf_avg_util_true.append(abs(float(leaf_avg_utility) - float(utility_true)))
        leaf_avg_util_full.append(abs(float(leaf_avg_utility) - float(utility_full)))
        leaf_avg_logliks.append(float(leaf_avg_loglik))
        leaf_avg_loglik_full.append(abs(float(leaf_avg_loglik) - float(loglik_full)))

        leaf_u_pi_true.append(float("nan"))
        leaf_u_pi_full.append(float("nan"))
        leaf_u_util_true.append(abs(float(leaf_utility_only) - float(utility_true)))
        leaf_u_util_full.append(abs(float(leaf_utility_only) - float(utility_full)))
        leaf_u_logliks.append(float("nan"))
        leaf_u_loglik_full.append(float("nan"))

    exact = ExactRecoveryMetrics(
        n_docs=int(len(docs_test)),
        root_count_l1_mean=_safe_stat(exact_count_l1, kind="mean"),
        root_count_l2_mean=_safe_stat(exact_count_l2, kind="mean"),
        root_pi_l1_mean=_safe_stat(exact_pi_l1, kind="mean"),
        root_pi_l2_mean=_safe_stat(exact_pi_l2, kind="mean"),
        root_utility_abs_mean=_safe_stat(exact_utility_abs, kind="mean"),
        root_loglik_abs_mean=_safe_stat(exact_loglik_abs, kind="mean"),
        schedule_count_l1_spread_mean=_safe_stat(sched_count_spread, kind="mean"),
        schedule_pi_l1_spread_mean=_safe_stat(sched_pi_spread, kind="mean"),
        schedule_utility_spread_mean=_safe_stat(sched_util_spread, kind="mean"),
        schedule_loglik_spread_mean=_safe_stat(sched_loglik_spread, kind="mean"),
    )

    methods = {
        "full_doc": asdict(
            _method_metrics(
                pi_to_true=full_pi_true,
                pi_to_full=full_pi_full,
                utility_to_true=full_util_true,
                utility_to_full=full_util_full,
                log_likelihoods=full_logliks,
                loglik_to_full=full_loglik_full,
                n_docs=len(docs_test),
            )
        ),
        "exact_tree": asdict(
            _method_metrics(
                pi_to_true=tree_pi_true,
                pi_to_full=tree_pi_full,
                utility_to_true=tree_util_true,
                utility_to_full=tree_util_full,
                log_likelihoods=tree_logliks,
                loglik_to_full=tree_loglik_full,
                n_docs=len(docs_test),
            )
        ),
        "leaf_average": asdict(
            _method_metrics(
                pi_to_true=leaf_avg_pi_true,
                pi_to_full=leaf_avg_pi_full,
                utility_to_true=leaf_avg_util_true,
                utility_to_full=leaf_avg_util_full,
                log_likelihoods=leaf_avg_logliks,
                loglik_to_full=leaf_avg_loglik_full,
                n_docs=len(docs_test),
            )
        ),
        "leaf_utility_only": asdict(
            _method_metrics(
                pi_to_true=leaf_u_pi_true,
                pi_to_full=leaf_u_pi_full,
                utility_to_true=leaf_u_util_true,
                utility_to_full=leaf_u_util_full,
                log_likelihoods=leaf_u_logliks,
                loglik_to_full=leaf_u_loglik_full,
                n_docs=len(docs_test),
            )
        ),
    }

    world_stats = {
        "train_docs_reserved": int(config.train_docs),
        "test_docs_evaluated": int(config.test_docs),
        **{f"test_{k}": v for k, v in _summarize_docs(docs_test, leaf_tokens=int(config.leaf_tokens)).items()},
    }

    utility_truth = {
        "relevant_topics": list(int(x) for x in world.relevant_topics),
        "theta_true": [float(x) for x in theta_true.tolist()],
        "W_base": [[float(x) for x in row] for row in W_base.tolist()],
        "quadratic_utility_weight": float(config.lambda_multiplier),
    }
    public_config = {**asdict(config), "quadratic_utility_weight": float(config.lambda_multiplier)}
    public_config.pop("lambda_multiplier", None)

    return LDATreeRecoverySummary(
        config=public_config,
        topic_meta=dict(world.topic_meta),
        utility_truth=utility_truth,
        world_stats=world_stats,
        exact_recovery=asdict(exact),
        methods=methods,
        objective=latent_quadratic_utility_objective_semantics(
            name="lda_document_utility_target",
            optimized_against="document_level_latent_utility",
            quadratic_utility_weight=float(config.lambda_multiplier),
            linear_component_name="topic_mixture_linear_term",
            interaction_component_name="topic_mixture_quadratic_term",
            weighting_scheme="linear_plus_quadratic_utility",
            metadata={"problem_id": "lda_tree_recovery"},
        ),
    )


def run_lda_tree_recovery_experiment(config: LDATreeRecoveryConfig) -> LDATreeRecoverySummary:
    world = sample_lda_tree_recovery_world(config)
    return run_lda_tree_recovery_experiment_from_world(config, world)


__all__ = [
    "ExactRecoveryMetrics",
    "LDATreeRecoveryConfig",
    "LDATreeRecoveryDoc",
    "LDATreeRecoverySummary",
    "LDATreeRecoveryWorld",
    "MethodMetrics",
    "VALID_EMISSION_MODES",
    "VALID_SCHEDULES",
    "lda_tree_recovery_world_cache_signature",
    "run_lda_tree_recovery_experiment",
    "run_lda_tree_recovery_experiment_from_world",
    "sample_lda_tree_recovery_world",
]
