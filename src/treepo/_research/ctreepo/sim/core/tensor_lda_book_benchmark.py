"""
Tensor-LDA DGP benchmark for ThinkingTrees / C-TreePO-style comparisons.

This module implements a synthetic "books" benchmark grounded in the same traditional LDA
data generation process (DGP) used in the Tensor-LDA paper's simulation section:

* topic-word distributions:      mu_k ~ Dirichlet(beta)
* book-level topic weights:      w_b  ~ Dirichlet(alpha)
* chapter-level topic mixtures:  theta_bc ~ Dirichlet(concentration * w_b)
* tokens:                        z ~ Mult(theta_bc), word ~ Mult(mu_z)

Benchmark goal
--------------
Compare:
1) Tensor-LDA-style book-level estimation (moment/projection baseline),
2) C-TreePO-style tree aggregation with local summaries, calibration labels, and query budgets,
3) oracle upper bound.

The benchmark includes:
* root error metrics (L1/L2),
* local-law proxies (C1 leaf discrepancy, C3 merge discrepancy),
* query accounting,
* optional selection-bias audit (naive vs IPW vs DSL-style estimators).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from statistics import fmean
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from treepo._research.ctreepo.sim.objective_semantics import discrepancy_benchmark_objective_semantics


VALID_CALIBRATION_POLICIES: Tuple[str, ...] = ("uniform", "entropy")
VALID_INTERNAL_QUERY_DESIGNS: Tuple[str, ...] = ("none", "uniform", "risk")


@dataclass(frozen=True)
class TensorLDABookBenchmarkConfig:
    # LDA DGP.
    n_topics: int = 5
    vocab_size: int = 600
    chapters_per_book: int = 16
    tokens_per_chapter: int = 128
    alpha_topic: float = 0.20
    beta_word: float = 0.10
    chapter_concentration: float = 40.0

    # Dataset sizes.
    n_books_train: int = 256
    n_books_test: int = 256

    # Proxy estimator built from anchor words.
    anchor_words_per_topic: int = 20
    proxy_temperature: float = 0.50
    proxy_noise_std: float = 0.05

    # Calibration labels (queried leaves on training books).
    calibration_leaf_query_rate: float = 0.10
    calibration_policy: str = "uniform"  # uniform|entropy
    calibration_ridge: float = 1e-4
    calibration_pi_min: float = 0.01

    # Optional evaluation-time guidance budget on test books.
    eval_leaf_query_rate: float = 0.0
    eval_internal_query_rate: float = 0.0
    eval_internal_query_design: str = "none"  # none|uniform|risk

    # Thresholds for violation-rate reporting.
    c1_threshold: float = 0.20
    c3_threshold: float = 0.20

    # Optional selection-bias audit over internal-node discrepancy population.
    selection_audit_trials: int = 0
    selection_audit_sample_rate: float = 0.10
    selection_audit_pi_min: float = 0.01

    seed: int = 0


@dataclass(frozen=True)
class SyntheticBookCorpus:
    topic_word: np.ndarray  # [K, V]
    book_topic_weights: np.ndarray  # [B, K]
    chapter_topic_weights: np.ndarray  # [B, C, K]
    chapter_word_counts: np.ndarray  # [B, C, V]


@dataclass(frozen=True)
class PolicyMetrics:
    n_books: int
    root_l1_mean: float
    root_l1_median: float
    root_l1_p95: float
    root_l2_mean: float
    latent_root_l1_mean: float
    c1_violation_rate: float
    c3_violation_rate: float
    mean_leaf_queries: float
    mean_internal_queries: float
    mean_total_queries: float


@dataclass(frozen=True)
class EstimatorStats:
    mean: float
    bias: float
    variance: float
    rmse: float


@dataclass(frozen=True)
class SelectionAuditSummary:
    n_units: int
    true_mean_discrepancy: float
    true_violation_rate: float
    trials: int
    target_sample_rate: float
    pi_min: float
    mean_sample_size: float
    mean_effective_sample_size: float
    naive_mean_discrepancy: EstimatorStats
    ipw_mean_discrepancy: EstimatorStats
    dsl0_mean_discrepancy: EstimatorStats
    dsl_oracle_mean_discrepancy: EstimatorStats
    naive_violation_rate: EstimatorStats
    ipw_violation_rate: EstimatorStats
    dsl0_violation_rate: EstimatorStats
    dsl_oracle_violation_rate: EstimatorStats
    ipw_violation_ci_coverage: float
    ipw_violation_ci_mean_radius: float


@dataclass(frozen=True)
class TensorLDABookBenchmarkSummary:
    config: Dict[str, object]
    calibration_samples: int
    metrics: Dict[str, PolicyMetrics]
    selection_audit: Optional[SelectionAuditSummary]
    objective: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "calibration_samples": int(self.calibration_samples),
            "metrics": {k: asdict(v) for k, v in self.metrics.items()},
            "selection_audit": asdict(self.selection_audit) if self.selection_audit is not None else None,
            "objective": self.objective,
        }
        return json.dumps(payload, indent=2, sort_keys=True)


@dataclass
class _TreeNode:
    est: np.ndarray
    truth: np.ndarray
    mass: float


def _validate_config(config: TensorLDABookBenchmarkConfig) -> None:
    if config.n_topics < 2:
        raise ValueError("n_topics must be >= 2")
    if config.vocab_size < config.n_topics:
        raise ValueError("vocab_size must be >= n_topics")
    if config.chapters_per_book < 2:
        raise ValueError("chapters_per_book must be >= 2")
    if config.tokens_per_chapter < 2:
        raise ValueError("tokens_per_chapter must be >= 2")
    if config.n_books_train < 1 or config.n_books_test < 1:
        raise ValueError("n_books_train and n_books_test must be >= 1")
    if config.alpha_topic <= 0 or config.beta_word <= 0 or config.chapter_concentration <= 0:
        raise ValueError("alpha_topic, beta_word, chapter_concentration must be > 0")
    if config.anchor_words_per_topic < 1 or config.anchor_words_per_topic > config.vocab_size:
        raise ValueError("anchor_words_per_topic must be in [1, vocab_size]")
    if config.proxy_temperature <= 0:
        raise ValueError("proxy_temperature must be > 0")
    if config.proxy_noise_std < 0:
        raise ValueError("proxy_noise_std must be >= 0")
    if not (0.0 <= config.calibration_leaf_query_rate <= 1.0):
        raise ValueError("calibration_leaf_query_rate must be in [0, 1]")
    if not (0.0 <= config.eval_leaf_query_rate <= 1.0):
        raise ValueError("eval_leaf_query_rate must be in [0, 1]")
    if not (0.0 <= config.eval_internal_query_rate <= 1.0):
        raise ValueError("eval_internal_query_rate must be in [0, 1]")
    if config.calibration_policy not in VALID_CALIBRATION_POLICIES:
        raise ValueError(f"calibration_policy must be one of {VALID_CALIBRATION_POLICIES}")
    if config.eval_internal_query_design not in VALID_INTERNAL_QUERY_DESIGNS:
        raise ValueError(f"eval_internal_query_design must be one of {VALID_INTERNAL_QUERY_DESIGNS}")
    if config.calibration_ridge < 0:
        raise ValueError("calibration_ridge must be >= 0")
    if not (0.0 < config.calibration_pi_min <= 1.0):
        raise ValueError("calibration_pi_min must be in (0, 1]")
    if config.c1_threshold < 0 or config.c3_threshold < 0:
        raise ValueError("c1_threshold and c3_threshold must be >= 0")
    if config.selection_audit_trials < 0:
        raise ValueError("selection_audit_trials must be >= 0")
    if not (0.0 <= config.selection_audit_sample_rate <= 1.0):
        raise ValueError("selection_audit_sample_rate must be in [0, 1]")
    if not (0.0 < config.selection_audit_pi_min <= 1.0):
        raise ValueError("selection_audit_pi_min must be in (0, 1]")


def _safe_mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(fmean(vals))


def _median(xs: Sequence[float]) -> float:
    vals = np.asarray([float(x) for x in xs if math.isfinite(float(x))], dtype=np.float64)
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def _p95(xs: Sequence[float]) -> float:
    vals = np.asarray([float(x) for x in xs if math.isfinite(float(x))], dtype=np.float64)
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, 0.95))


def _l1(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(u, dtype=np.float64) - np.asarray(v, dtype=np.float64))))


def _l2(u: np.ndarray, v: np.ndarray) -> float:
    d = np.asarray(u, dtype=np.float64) - np.asarray(v, dtype=np.float64)
    return float(np.sqrt(np.sum(d * d)))


from treepo._research.ctreepo.sim.util import normalize_simplex_vec, normalize_simplex_rows

_normalize_simplex_vec = normalize_simplex_vec
_normalize_simplex_rows = normalize_simplex_rows


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    denom = np.sum(exp_z, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return exp_z / denom


def _inclusion_probs_from_scores(scores: np.ndarray, *, target_rate: float, pi_min: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    target_rate = float(max(pi_min, min(1.0, target_rate)))
    pi_min = float(max(1e-9, min(1.0, pi_min)))
    if scores.size == 0:
        return np.zeros((0,), dtype=np.float64)
    s = scores.copy()
    s -= np.min(s)
    if float(np.max(s)) > 0.0:
        s /= float(np.max(s))
    c = float(max(0.0, target_rate - pi_min))
    pi = pi_min + c * s
    if float(np.mean(pi)) > target_rate:
        lo = 0.0
        hi = max(c, 1.0)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            cur = np.clip(pi_min + mid * s, pi_min, 1.0)
            if float(np.mean(cur)) > target_rate:
                hi = mid
            else:
                lo = mid
        pi = np.clip(pi_min + lo * s, pi_min, 1.0)
    return np.asarray(pi, dtype=np.float64)


def _bernoulli_sample(pi: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    pi = np.asarray(pi, dtype=np.float64)
    return rng.random(size=pi.shape) < pi


def _effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    if s2 <= 0.0:
        return 0.0
    return float((s1 * s1) / s2)


def _estimator_stats(estimates: Sequence[float], *, truth: float) -> EstimatorStats:
    vals = np.asarray([float(x) for x in estimates if math.isfinite(float(x))], dtype=np.float64)
    if vals.size == 0:
        return EstimatorStats(mean=float("nan"), bias=float("nan"), variance=float("nan"), rmse=float("nan"))
    mean = float(np.mean(vals))
    bias = float(mean - truth)
    var = float(np.var(vals))
    rmse = float(np.sqrt(np.mean((vals - truth) ** 2)))
    return EstimatorStats(mean=mean, bias=bias, variance=var, rmse=rmse)


def sample_topic_word_matrix(config: TensorLDABookBenchmarkConfig, *, rng: np.random.Generator) -> np.ndarray:
    beta = np.full((int(config.vocab_size),), float(config.beta_word), dtype=np.float64)
    return np.asarray(rng.dirichlet(beta, size=int(config.n_topics)), dtype=np.float64)


def generate_synthetic_books(
    config: TensorLDABookBenchmarkConfig,
    *,
    topic_word: np.ndarray,
    n_books: int,
    rng: np.random.Generator,
) -> SyntheticBookCorpus:
    k = int(config.n_topics)
    v = int(config.vocab_size)
    c = int(config.chapters_per_book)
    tpc = int(config.tokens_per_chapter)
    n_books = int(n_books)

    alpha = np.full((k,), float(config.alpha_topic), dtype=np.float64)
    book_topic_weights = np.asarray(rng.dirichlet(alpha, size=n_books), dtype=np.float64)
    chapter_topic_weights = np.zeros((n_books, c, k), dtype=np.float64)
    chapter_word_counts = np.zeros((n_books, c, v), dtype=np.int64)

    for b in range(n_books):
        base = np.maximum(float(config.chapter_concentration) * book_topic_weights[b], 1e-9)
        for j in range(c):
            theta = np.asarray(rng.dirichlet(base), dtype=np.float64)
            chapter_topic_weights[b, j] = theta

            topic_counts = np.asarray(rng.multinomial(tpc, theta), dtype=np.int64)
            word_counts = np.zeros((v,), dtype=np.int64)
            for topic_idx, n_t in enumerate(topic_counts):
                if int(n_t) <= 0:
                    continue
                word_counts += np.asarray(rng.multinomial(int(n_t), topic_word[topic_idx]), dtype=np.int64)
            chapter_word_counts[b, j] = word_counts

    return SyntheticBookCorpus(
        topic_word=np.asarray(topic_word, dtype=np.float64),
        book_topic_weights=book_topic_weights,
        chapter_topic_weights=chapter_topic_weights,
        chapter_word_counts=chapter_word_counts,
    )


def _anchor_indices(topic_word: np.ndarray, *, n_anchor_words: int) -> np.ndarray:
    k, v = topic_word.shape
    m = int(max(1, min(n_anchor_words, v)))
    idx = np.argsort(topic_word, axis=1)[:, -m:]
    return np.asarray(idx, dtype=np.int64)


def _estimate_proxy_leaf_thetas(
    chapter_word_counts: np.ndarray,
    *,
    anchors: np.ndarray,
    temperature: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = np.asarray(chapter_word_counts, dtype=np.float64)
    sums = np.sum(counts, axis=2, keepdims=True)
    sums = np.maximum(sums, 1.0)
    freqs = counts / sums

    k = anchors.shape[0]
    v = counts.shape[2]
    mask = np.zeros((k, v), dtype=np.float64)
    for i in range(k):
        mask[i, anchors[i]] = 1.0
    scores = np.einsum("bcv,kv->bck", freqs, mask) / float(max(1, anchors.shape[1]))
    logits = scores / float(max(1e-9, temperature))
    if noise_std > 0.0:
        logits = logits + rng.normal(0.0, float(noise_std), size=logits.shape)
    flat = logits.reshape(-1, logits.shape[2])
    out = _softmax_rows(flat).reshape(logits.shape)
    return np.asarray(out, dtype=np.float64)


def _estimate_projection_from_counts(counts: np.ndarray, *, topic_word: np.ndarray) -> np.ndarray:
    x = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(x))
    if total <= 0.0:
        return np.full((topic_word.shape[0],), 1.0 / float(topic_word.shape[0]), dtype=np.float64)
    freq = x / total
    raw, *_ = np.linalg.lstsq(topic_word.T, freq, rcond=None)
    return _normalize_simplex_vec(np.asarray(raw, dtype=np.float64))


def _sample_leaf_query_mask(
    proxy_leaf_thetas: np.ndarray,
    *,
    rate: float,
    policy: str,
    pi_min: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    b, c, _ = proxy_leaf_thetas.shape
    rate = float(max(0.0, min(1.0, rate)))
    pi_min = float(max(1e-9, min(1.0, pi_min)))
    if rate <= 0.0:
        pi = np.zeros((b, c), dtype=np.float64)
        return np.zeros((b, c), dtype=bool), pi
    if policy == "uniform":
        pi = np.full((b, c), float(max(pi_min, rate)), dtype=np.float64)
    elif policy == "entropy":
        p = np.clip(np.asarray(proxy_leaf_thetas, dtype=np.float64), 1e-12, 1.0)
        entropy = -np.sum(p * np.log(p), axis=2)
        pi = _inclusion_probs_from_scores(
            entropy.reshape(-1), target_rate=float(rate), pi_min=float(pi_min)
        ).reshape(b, c)
    else:
        raise ValueError(f"unknown leaf query policy: {policy}")
    mask = _bernoulli_sample(pi, rng=rng)
    return np.asarray(mask, dtype=bool), np.asarray(pi, dtype=np.float64)


def _fit_affine_calibration(
    proxy_leaf_thetas: np.ndarray,
    true_leaf_thetas: np.ndarray,
    queried_mask: np.ndarray,
    *,
    ridge: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    x = np.asarray(proxy_leaf_thetas, dtype=np.float64)[queried_mask]
    y = np.asarray(true_leaf_thetas, dtype=np.float64)[queried_mask]
    k = proxy_leaf_thetas.shape[2]
    n = int(x.shape[0])
    if n <= 0:
        return np.eye(k, dtype=np.float64), np.zeros((k,), dtype=np.float64), 0

    x1 = np.concatenate([x, np.ones((n, 1), dtype=np.float64)], axis=1)  # [n, k+1]
    reg = float(max(0.0, ridge))
    gram = x1.T @ x1
    if reg > 0.0:
        r = reg * np.eye(k + 1, dtype=np.float64)
        r[-1, -1] = 0.0
        gram = gram + r
    rhs = x1.T @ y
    coef, *_ = np.linalg.lstsq(gram, rhs, rcond=None)  # [k+1, k]
    w = np.asarray(coef[:k, :], dtype=np.float64)
    b = np.asarray(coef[k, :], dtype=np.float64)
    return w, b, n


def _apply_affine_calibration(theta: np.ndarray, *, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    z = np.asarray(theta, dtype=np.float64)
    flat = z.reshape(-1, z.shape[2])
    mapped = flat @ np.asarray(w, dtype=np.float64) + np.asarray(b, dtype=np.float64)
    mapped = _normalize_simplex_rows(mapped)
    return mapped.reshape(z.shape)


def _reduce_balanced_tree_with_guidance(
    leaf_est: np.ndarray,
    leaf_truth: np.ndarray,
    *,
    leaf_query_rate: float,
    internal_query_rate: float,
    internal_query_design: str,
    rng: np.random.Generator,
    leaf_masses: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[float], List[float], int, int, List[float], List[float]]:
    c, k = leaf_est.shape
    leaf_query_rate = float(max(0.0, min(1.0, leaf_query_rate)))
    internal_query_rate = float(max(0.0, min(1.0, internal_query_rate)))
    if leaf_masses is None:
        masses = np.ones((c,), dtype=np.float64)
    else:
        masses = np.asarray(leaf_masses, dtype=np.float64).reshape(-1)
        if masses.shape[0] != c:
            raise ValueError("leaf_masses must align with leaf_est/leaf_truth rows")
        masses = np.clip(masses, 1e-12, None)

    leaf_mask = rng.random(size=(c,)) < leaf_query_rate
    est = np.asarray(leaf_est, dtype=np.float64).copy()
    est[leaf_mask] = np.asarray(leaf_truth, dtype=np.float64)[leaf_mask]

    c1_errors = [_l1(est[i], leaf_truth[i]) for i in range(c)]
    leaf_queries = int(np.sum(leaf_mask))
    internal_queries = 0
    c3_errors: List[float] = []
    internal_population_errors: List[float] = []
    internal_population_scores: List[float] = []

    nodes: List[_TreeNode] = [
        _TreeNode(
            est=est[i].copy(),
            truth=np.asarray(leaf_truth[i], dtype=np.float64).copy(),
            mass=float(masses[i]),
        )
        for i in range(c)
    ]

    while len(nodes) > 1:
        next_nodes: List[_TreeNode] = []
        merged_flags: List[bool] = []
        merge_scores: List[float] = []

        i = 0
        while i < len(nodes):
            if i + 1 >= len(nodes):
                next_nodes.append(nodes[i])
                merged_flags.append(False)
                merge_scores.append(float("-inf"))
                i += 1
                continue

            left = nodes[i]
            right = nodes[i + 1]
            n = float(left.mass + right.mass)
            est_merge = (left.est * float(left.mass) + right.est * float(right.mass)) / float(n)
            truth_merge = (left.truth * float(left.mass) + right.truth * float(right.mass)) / float(n)

            pre_err = _l1(est_merge, truth_merge)
            score = _l1(left.est, right.est)
            internal_population_errors.append(pre_err)
            internal_population_scores.append(score)

            next_nodes.append(_TreeNode(est=est_merge, truth=truth_merge, mass=float(n)))
            merged_flags.append(True)
            merge_scores.append(score)
            i += 2

        candidate_ids = [idx for idx, flag in enumerate(merged_flags) if flag]
        n_candidates = len(candidate_ids)
        selected: set[int] = set()
        if n_candidates > 0 and internal_query_rate > 0.0 and internal_query_design != "none":
            q = int(round(internal_query_rate * float(n_candidates)))
            q = max(0, min(n_candidates, q))
            if q > 0:
                if internal_query_design == "uniform":
                    chosen = rng.choice(np.asarray(candidate_ids, dtype=np.int64), size=q, replace=False)
                    selected = {int(x) for x in np.asarray(chosen, dtype=np.int64)}
                elif internal_query_design == "risk":
                    ranked = sorted(candidate_ids, key=lambda idx: float(merge_scores[idx]), reverse=True)
                    selected = set(ranked[:q])
                else:
                    raise ValueError(f"unknown internal_query_design: {internal_query_design}")

        for idx in candidate_ids:
            node = next_nodes[idx]
            if idx in selected:
                node.est = node.truth.copy()
                internal_queries += 1
            c3_errors.append(_l1(node.est, node.truth))

        nodes = next_nodes

    root_est = _normalize_simplex_vec(nodes[0].est)
    return (
        root_est,
        c1_errors,
        c3_errors,
        leaf_queries,
        internal_queries,
        internal_population_errors,
        internal_population_scores,
    )


def _violation_rate(errs: Sequence[float], *, threshold: float) -> float:
    vals = [float(x) for x in errs if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=np.float64) > float(threshold)))


def _run_selection_bias_audit(
    *,
    discrepancies: np.ndarray,
    violations: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    trials: int,
    sample_rate: float,
    pi_min: float,
    seed: int,
) -> SelectionAuditSummary:
    rng = np.random.default_rng(int(seed))
    disc = np.asarray(discrepancies, dtype=np.float64)
    viol = np.asarray(violations, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n = int(disc.size)
    if n == 0:
        nan_stats = EstimatorStats(mean=float("nan"), bias=float("nan"), variance=float("nan"), rmse=float("nan"))
        return SelectionAuditSummary(
            n_units=0,
            true_mean_discrepancy=float("nan"),
            true_violation_rate=float("nan"),
            trials=int(trials),
            target_sample_rate=float(sample_rate),
            pi_min=float(pi_min),
            mean_sample_size=float("nan"),
            mean_effective_sample_size=float("nan"),
            naive_mean_discrepancy=nan_stats,
            ipw_mean_discrepancy=nan_stats,
            dsl0_mean_discrepancy=nan_stats,
            dsl_oracle_mean_discrepancy=nan_stats,
            naive_violation_rate=nan_stats,
            ipw_violation_rate=nan_stats,
            dsl0_violation_rate=nan_stats,
            dsl_oracle_violation_rate=nan_stats,
            ipw_violation_ci_coverage=float("nan"),
            ipw_violation_ci_mean_radius=float("nan"),
        )

    pi = _inclusion_probs_from_scores(scores, target_rate=float(sample_rate), pi_min=float(pi_min))
    truth_mu = float(np.mean(disc))
    truth_p = float(np.mean(viol > float(threshold)))
    viol01 = (viol > float(threshold)).astype(np.float64)

    naive_mu: List[float] = []
    ipw_mu: List[float] = []
    dsl0_mu: List[float] = []
    dsl_oracle_mu: List[float] = []

    naive_p: List[float] = []
    ipw_p: List[float] = []
    dsl0_p: List[float] = []
    dsl_oracle_p: List[float] = []

    sample_sizes: List[float] = []
    ess_vals: List[float] = []
    ci_covered: List[float] = []
    ci_radius: List[float] = []

    pred0_mu = np.zeros_like(disc)
    pred0_p = np.zeros_like(viol01)
    pred_oracle_mu = disc.copy()
    pred_oracle_p = viol01.copy()

    for _ in range(int(trials)):
        idx = _bernoulli_sample(pi, rng=rng)
        m = int(np.sum(idx))
        if m <= 0:
            naive_mu.append(float("nan"))
            ipw_mu.append(float("nan"))
            dsl0_mu.append(float("nan"))
            dsl_oracle_mu.append(float("nan"))
            naive_p.append(float("nan"))
            ipw_p.append(float("nan"))
            dsl0_p.append(float("nan"))
            dsl_oracle_p.append(float("nan"))
            sample_sizes.append(0.0)
            ess_vals.append(0.0)
            ci_covered.append(float("nan"))
            ci_radius.append(float("nan"))
            continue

        w = idx.astype(np.float64) / pi
        ess = _effective_sample_size(w)
        sample_sizes.append(float(m))
        ess_vals.append(float(ess))

        naive_mu_t = float(np.mean(disc[idx]))
        naive_p_t = float(np.mean(viol01[idx]))
        ipw_mu_t = float(np.sum(w * disc) / float(n))
        ipw_p_t = float(np.sum(w * viol01) / float(n))
        dsl0_mu_t = float(np.mean(pred0_mu) + np.sum(w * (disc - pred0_mu)) / float(n))
        dsl0_p_t = float(np.mean(pred0_p) + np.sum(w * (viol01 - pred0_p)) / float(n))
        dsl_oracle_mu_t = float(np.mean(pred_oracle_mu) + np.sum(w * (disc - pred_oracle_mu)) / float(n))
        dsl_oracle_p_t = float(np.mean(pred_oracle_p) + np.sum(w * (viol01 - pred_oracle_p)) / float(n))

        naive_mu.append(naive_mu_t)
        ipw_mu.append(ipw_mu_t)
        dsl0_mu.append(dsl0_mu_t)
        dsl_oracle_mu.append(dsl_oracle_mu_t)

        naive_p.append(naive_p_t)
        ipw_p.append(ipw_p_t)
        dsl0_p.append(dsl0_p_t)
        dsl_oracle_p.append(dsl_oracle_p_t)

        rad = float(1.96 * math.sqrt(max(ipw_p_t * (1.0 - ipw_p_t), 1e-9) / max(ess, 1e-9)))
        ci_radius.append(rad)
        ci_covered.append(float(abs(ipw_p_t - truth_p) <= rad))

    return SelectionAuditSummary(
        n_units=int(n),
        true_mean_discrepancy=truth_mu,
        true_violation_rate=truth_p,
        trials=int(trials),
        target_sample_rate=float(sample_rate),
        pi_min=float(pi_min),
        mean_sample_size=_safe_mean(sample_sizes),
        mean_effective_sample_size=_safe_mean(ess_vals),
        naive_mean_discrepancy=_estimator_stats(naive_mu, truth=truth_mu),
        ipw_mean_discrepancy=_estimator_stats(ipw_mu, truth=truth_mu),
        dsl0_mean_discrepancy=_estimator_stats(dsl0_mu, truth=truth_mu),
        dsl_oracle_mean_discrepancy=_estimator_stats(dsl_oracle_mu, truth=truth_mu),
        naive_violation_rate=_estimator_stats(naive_p, truth=truth_p),
        ipw_violation_rate=_estimator_stats(ipw_p, truth=truth_p),
        dsl0_violation_rate=_estimator_stats(dsl0_p, truth=truth_p),
        dsl_oracle_violation_rate=_estimator_stats(dsl_oracle_p, truth=truth_p),
        ipw_violation_ci_coverage=_safe_mean(ci_covered),
        ipw_violation_ci_mean_radius=_safe_mean(ci_radius),
    )


def run_tensor_lda_book_weight_benchmark(
    config: TensorLDABookBenchmarkConfig,
) -> TensorLDABookBenchmarkSummary:
    _validate_config(config)
    rng = np.random.default_rng(int(config.seed))

    topic_word = sample_topic_word_matrix(config, rng=rng)
    train = generate_synthetic_books(config, topic_word=topic_word, n_books=int(config.n_books_train), rng=rng)
    test = generate_synthetic_books(config, topic_word=topic_word, n_books=int(config.n_books_test), rng=rng)

    anchors = _anchor_indices(topic_word, n_anchor_words=int(config.anchor_words_per_topic))
    train_proxy = _estimate_proxy_leaf_thetas(
        train.chapter_word_counts,
        anchors=anchors,
        temperature=float(config.proxy_temperature),
        noise_std=float(config.proxy_noise_std),
        rng=rng,
    )
    test_proxy = _estimate_proxy_leaf_thetas(
        test.chapter_word_counts,
        anchors=anchors,
        temperature=float(config.proxy_temperature),
        noise_std=float(config.proxy_noise_std),
        rng=rng,
    )

    calib_mask, _calib_pi = _sample_leaf_query_mask(
        train_proxy,
        rate=float(config.calibration_leaf_query_rate),
        policy=str(config.calibration_policy),
        pi_min=float(config.calibration_pi_min),
        rng=rng,
    )
    w_cal, b_cal, n_calib = _fit_affine_calibration(
        train_proxy, train.chapter_topic_weights, calib_mask, ridge=float(config.calibration_ridge)
    )
    test_cal = _apply_affine_calibration(test_proxy, w=w_cal, b=b_cal)

    policy_names = (
        "tlda_projection",
        "ctree_proxy",
        "ctree_calibrated",
        "ctree_calibrated_budgeted",
        "ctree_oracle",
    )
    root_l1: Dict[str, List[float]] = {k: [] for k in policy_names}
    root_l2: Dict[str, List[float]] = {k: [] for k in policy_names}
    latent_l1: Dict[str, List[float]] = {k: [] for k in policy_names}
    c1_err: Dict[str, List[float]] = {k: [] for k in policy_names}
    c3_err: Dict[str, List[float]] = {k: [] for k in policy_names}
    q_leaf: Dict[str, List[float]] = {k: [] for k in policy_names}
    q_internal: Dict[str, List[float]] = {k: [] for k in policy_names}

    audit_disc_population: List[float] = []
    audit_score_population: List[float] = []

    for b in range(int(config.n_books_test)):
        truth_leaf = np.asarray(test.chapter_topic_weights[b], dtype=np.float64)
        leaf_masses = np.asarray(np.sum(test.chapter_word_counts[b], axis=1), dtype=np.float64).reshape(-1)
        leaf_masses = np.clip(leaf_masses, 1e-12, None)
        truth_root = _normalize_simplex_vec(
            np.sum(truth_leaf * leaf_masses[:, None], axis=0) / float(np.sum(leaf_masses))
        )
        latent_root = _normalize_simplex_vec(test.book_topic_weights[b])
        counts_book = np.asarray(np.sum(test.chapter_word_counts[b], axis=0), dtype=np.float64)

        # Policy 1: Tensor-LDA-style projection baseline.
        est_tlda = _estimate_projection_from_counts(counts_book, topic_word=topic_word)
        root_l1["tlda_projection"].append(_l1(est_tlda, truth_root))
        root_l2["tlda_projection"].append(_l2(est_tlda, truth_root))
        latent_l1["tlda_projection"].append(_l1(est_tlda, latent_root))
        q_leaf["tlda_projection"].append(0.0)
        q_internal["tlda_projection"].append(0.0)

        # Policy 2: raw proxy tree.
        est_proxy, c1p, c3p, lqp, iqp, _pop_err_proxy, _pop_score_proxy = _reduce_balanced_tree_with_guidance(
            np.asarray(test_proxy[b], dtype=np.float64),
            truth_leaf,
            leaf_query_rate=0.0,
            internal_query_rate=0.0,
            internal_query_design="none",
            rng=rng,
            leaf_masses=leaf_masses,
        )
        root_l1["ctree_proxy"].append(_l1(est_proxy, truth_root))
        root_l2["ctree_proxy"].append(_l2(est_proxy, truth_root))
        latent_l1["ctree_proxy"].append(_l1(est_proxy, latent_root))
        c1_err["ctree_proxy"].extend(c1p)
        c3_err["ctree_proxy"].extend(c3p)
        q_leaf["ctree_proxy"].append(float(lqp))
        q_internal["ctree_proxy"].append(float(iqp))

        # Policy 3: calibrated tree without eval-time guidance.
        est_cal, c1c, c3c, lqc, iqc, pop_err, pop_score = _reduce_balanced_tree_with_guidance(
            np.asarray(test_cal[b], dtype=np.float64),
            truth_leaf,
            leaf_query_rate=0.0,
            internal_query_rate=0.0,
            internal_query_design="none",
            rng=rng,
            leaf_masses=leaf_masses,
        )
        root_l1["ctree_calibrated"].append(_l1(est_cal, truth_root))
        root_l2["ctree_calibrated"].append(_l2(est_cal, truth_root))
        latent_l1["ctree_calibrated"].append(_l1(est_cal, latent_root))
        c1_err["ctree_calibrated"].extend(c1c)
        c3_err["ctree_calibrated"].extend(c3c)
        q_leaf["ctree_calibrated"].append(float(lqc))
        q_internal["ctree_calibrated"].append(float(iqc))
        audit_disc_population.extend(float(x) for x in pop_err)
        audit_score_population.extend(float(x) for x in pop_score)

        # Policy 4: calibrated + eval-time leaf/internal oracle budget.
        est_budget, c1b, c3b, lqb, iqb, _e, _s = _reduce_balanced_tree_with_guidance(
            np.asarray(test_cal[b], dtype=np.float64),
            truth_leaf,
            leaf_query_rate=float(config.eval_leaf_query_rate),
            internal_query_rate=float(config.eval_internal_query_rate),
            internal_query_design=str(config.eval_internal_query_design),
            rng=rng,
            leaf_masses=leaf_masses,
        )
        root_l1["ctree_calibrated_budgeted"].append(_l1(est_budget, truth_root))
        root_l2["ctree_calibrated_budgeted"].append(_l2(est_budget, truth_root))
        latent_l1["ctree_calibrated_budgeted"].append(_l1(est_budget, latent_root))
        c1_err["ctree_calibrated_budgeted"].extend(c1b)
        c3_err["ctree_calibrated_budgeted"].extend(c3b)
        q_leaf["ctree_calibrated_budgeted"].append(float(lqb))
        q_internal["ctree_calibrated_budgeted"].append(float(iqb))

        # Policy 5: oracle tree upper bound (same reducer path; full guidance).
        est_oracle, c1o, c3o, lqo, iqo, _e3, _s3 = _reduce_balanced_tree_with_guidance(
            truth_leaf,
            truth_leaf,
            leaf_query_rate=1.0,
            internal_query_rate=1.0,
            internal_query_design="risk",
            rng=rng,
            leaf_masses=leaf_masses,
        )
        root_l1["ctree_oracle"].append(_l1(est_oracle, truth_root))
        root_l2["ctree_oracle"].append(_l2(est_oracle, truth_root))
        latent_l1["ctree_oracle"].append(_l1(truth_root, latent_root))
        c1_err["ctree_oracle"].extend(c1o)
        c3_err["ctree_oracle"].extend(c3o)
        q_leaf["ctree_oracle"].append(float(lqo))
        q_internal["ctree_oracle"].append(float(iqo))

    metrics: Dict[str, PolicyMetrics] = {}
    for key in policy_names:
        total_q = [float(a + b) for a, b in zip(q_leaf[key], q_internal[key])]
        metrics[key] = PolicyMetrics(
            n_books=int(config.n_books_test),
            root_l1_mean=_safe_mean(root_l1[key]),
            root_l1_median=_median(root_l1[key]),
            root_l1_p95=_p95(root_l1[key]),
            root_l2_mean=_safe_mean(root_l2[key]),
            latent_root_l1_mean=_safe_mean(latent_l1[key]),
            c1_violation_rate=_violation_rate(c1_err[key], threshold=float(config.c1_threshold)),
            c3_violation_rate=_violation_rate(c3_err[key], threshold=float(config.c3_threshold)),
            mean_leaf_queries=_safe_mean(q_leaf[key]),
            mean_internal_queries=_safe_mean(q_internal[key]),
            mean_total_queries=_safe_mean(total_q),
        )

    selection_audit: Optional[SelectionAuditSummary] = None
    if int(config.selection_audit_trials) > 0 and len(audit_disc_population) > 0:
        disc = np.asarray(audit_disc_population, dtype=np.float64)
        viol = (disc > float(config.c3_threshold)).astype(np.float64)
        scores = np.asarray(audit_score_population, dtype=np.float64)
        selection_audit = _run_selection_bias_audit(
            discrepancies=disc,
            violations=viol,
            scores=scores,
            threshold=float(config.c3_threshold),
            trials=int(config.selection_audit_trials),
            sample_rate=float(config.selection_audit_sample_rate),
            pi_min=float(config.selection_audit_pi_min),
            seed=int(config.seed),
        )

    return TensorLDABookBenchmarkSummary(
        config=asdict(config),
        calibration_samples=int(n_calib),
        metrics=metrics,
        selection_audit=selection_audit,
        objective=discrepancy_benchmark_objective_semantics(
            name="tensor_lda_book_benchmark",
            optimized_against="ridge_calibration_on_queried_chapters",
            benchmark_metric_name="root_l1_mean",
            metadata={
                "family": "tensor_lda_book_benchmark",
                "calibration_leaf_query_rate": float(config.calibration_leaf_query_rate),
                "eval_leaf_query_rate": float(config.eval_leaf_query_rate),
                "eval_internal_query_rate": float(config.eval_internal_query_rate),
            },
        ),
    )


__all__ = [
    "TensorLDABookBenchmarkConfig",
    "SyntheticBookCorpus",
    "PolicyMetrics",
    "EstimatorStats",
    "SelectionAuditSummary",
    "TensorLDABookBenchmarkSummary",
    "sample_topic_word_matrix",
    "generate_synthetic_books",
    "run_tensor_lda_book_weight_benchmark",
    "_run_selection_bias_audit",
]
