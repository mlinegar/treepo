"""
SFM-style privacy-constrained sketch comparison harness.

This module sets up a direct comparison under *matched constraints*:
1) fixed sketch memory (B x P bits for PCSA-family methods), and
2) fixed privacy budget epsilon.

It includes:
- PCSA + composite-likelihood estimator (non-private baseline),
- SFM(sym)-style local privatization + randomized merge g + MLE,
- SFM(xor)-style local privatization + deterministic xor merge + MLE,
- deterministic counterfactual merges under Msym (or/xor) for theorem alignment,
- optional HLL non-private baseline,
- a lightweight learned decoder over local-Msym + deterministic-or merged sketches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from treepo._research.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from treepo._research.tree.learned_sketch_simulation import (
    HLLConfig,
    HyperLogLogSketch,
    match_hll_precision_for_bits,
)
from treepo._research.tree.ipw import (
    NodeType,
    TreeSample,
    effective_sample_size,
    ipw_preference_empirical_bernstein_ci,
    ipw_preference_loss,
    ipw_violation_empirical_bernstein_ci,
    ipw_violation_rate,
)


MethodName = str

METHOD_PCSA_NON_PRIVATE: MethodName = "pcsa_non_private_mle"
METHOD_SFM_SYM: MethodName = "sfm_sym_randmerge_mle"
METHOD_SFM_XOR: MethodName = "sfm_xor_detxor_mle"
METHOD_SYM_LOCAL_OR: MethodName = "sym_local_detor_mle"
METHOD_SYM_LOCAL_XOR: MethodName = "sym_local_detxor_mle"
METHOD_HLL_NON_PRIVATE: MethodName = "hll_non_private"
METHOD_OURS_RIDGE_SYM: MethodName = "ours_ridge_sym_local_detor"


@dataclass(frozen=True)
class SFMComparisonConfig:
    # Distinct-count workload
    n_values: Tuple[int, ...] = (100, 300, 1000, 3000, 10_000, 30_000, 100_000)
    n_trials: int = 200
    merge_counts: Tuple[int, ...] = (1, 2, 8, 32)
    universe_size: int = 20_000_000
    # Privacy budgets for private methods
    epsilons: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    # Sketch memory for PCSA-family methods
    buckets: int = 4096
    levels: int = 24
    # MLE search range
    n_min_est: int = 1
    n_max_est: int = 2_000_000
    # Learned decoder options
    include_hll_non_private: bool = True
    include_ours_ridge_sym: bool = True
    ridge_train_samples: int = 4000
    ridge_l2: float = 1e-3
    # Theory floor for "distance to optimum" reporting
    include_theory_floor: bool = True
    # IPW audit simulation (optional)
    enable_ipw: bool = False
    ipw_audit_rates: Tuple[float, ...] = (0.10, 0.25, 0.50, 1.00)
    ipw_delta: float = 0.05
    ipw_sampling_scheme: str = "prediction_stratified"  # {"uniform", "prediction_stratified"}
    ipw_propensity_floor: float = 0.01
    ipw_violation_abs_rel_threshold: float = 0.10
    # RNG
    seed: int = 0


@dataclass(frozen=True)
class ComparisonRow:
    method: str
    n: int
    merge_count: int
    epsilon: Optional[float]
    epsilon_effective: Optional[float]
    memory_bits: int
    n_trials: int
    rrmse: float
    mean_rel_error: float
    mean_abs_rel_error: float
    median_abs_rel_error: float
    mse: float
    rel_eff_vs_sfm_sym: Optional[float]
    channel_p_hat: Optional[float]
    channel_q_hat: Optional[float]
    channel_p_target: Optional[float]
    channel_q_target: Optional[float]
    channel_calibration_l1: Optional[float]
    theory_rrmse_floor: Optional[float]
    rrmse_gap_to_theory_floor: Optional[float]
    ipw_audit_rate: Optional[float]
    ipw_sample_count: Optional[int]
    ipw_effective_sample_size: Optional[float]
    true_preference_loss: Optional[float]
    true_violation_rate: Optional[float]
    ipw_preference_loss: Optional[float]
    ipw_preference_ci_low: Optional[float]
    ipw_preference_ci_high: Optional[float]
    ipw_violation_rate: Optional[float]
    ipw_violation_ci_low: Optional[float]
    ipw_violation_ci_high: Optional[float]


@dataclass(frozen=True)
class ComparisonSummary:
    config: Dict[str, object]
    rows: Tuple[ComparisonRow, ...]

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "rows": [asdict(r) for r in self.rows],
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def _set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _splitmix64(x: int) -> int:
    z = (int(x) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return int(z & 0xFFFFFFFFFFFFFFFF)


class PCSASketch:
    """
    PCSA-style occupancy sketch with B buckets and P levels.

    Each element maps to one (bucket, level) bit:
    - bucket index in [0, B-1]
    - level has geometric tail, truncated to P
    """

    def __init__(self, buckets: int, levels: int):
        if int(buckets) <= 0:
            raise ValueError("buckets must be positive")
        if int(levels) <= 1:
            raise ValueError("levels must be >= 2")
        self.buckets = int(buckets)
        self.levels = int(levels)
        self.bits = np.zeros((self.buckets, self.levels), dtype=np.bool_)

    def add(self, token_id: int) -> None:
        h = _splitmix64(int(token_id))
        b = int(h % self.buckets)
        w = int(h // self.buckets)
        # Geometric(1/2)-style level from trailing-zero count (+1), truncated at P.
        lvl = 1
        while lvl < self.levels and (w & 1) == 0:
            lvl += 1
            w >>= 1
        self.bits[b, lvl - 1] = True

    def merge_or(self, other: "PCSASketch") -> None:
        if self.buckets != other.buckets or self.levels != other.levels:
            raise ValueError("cannot merge incompatible PCSA sketches")
        np.logical_or(self.bits, other.bits, out=self.bits)

    @staticmethod
    def from_tokens(tokens: Sequence[int], *, buckets: int, levels: int) -> "PCSASketch":
        sk = PCSASketch(buckets=int(buckets), levels=int(levels))
        for tok in tokens:
            sk.add(int(tok))
        return sk


def _pcsa_gamma(level_1_based: int, buckets: int, levels: int) -> float:
    j = int(level_1_based)
    rho = (2.0 ** (-float(min(j, int(levels) - 1)))) / float(buckets)
    return float(max(1e-15, 1.0 - rho))


def _priv_params_sym(epsilon: float) -> Tuple[float, float]:
    eps = float(epsilon)
    p = math.exp(eps) / (math.exp(eps) + 1.0)
    q = 1.0 - p
    return float(p), float(q)


def _priv_params_xor(epsilon: float) -> Tuple[float, float]:
    eps = float(epsilon)
    p = 0.5
    q = 1.0 / (2.0 * math.exp(eps))
    return float(p), float(q)


def _epsilon_star_pairwise_merge(epsilon: float, merge_count: int) -> float:
    """
    Effective epsilon* after merging k summaries, each with epsilon.
    Remark 4.9 for equal epsilons:
      epsilon* = -log(1 - (1 - exp(-epsilon))^k)
    """
    eps = float(epsilon)
    k = int(max(1, merge_count))
    a = 1.0 - math.exp(-eps)
    one_minus = 1.0 - (a ** k)
    if one_minus <= 0.0:
        return 60.0
    return float(-math.log(one_minus))


def _epsilon_star_two(epsilon_left: float, epsilon_right: float) -> float:
    e1 = float(epsilon_left)
    e2 = float(epsilon_right)
    x = math.exp(-e1) + math.exp(-e2) - math.exp(-(e1 + e2))
    x = min(max(x, 1e-15), 1.0 - 1e-15)
    return float(-math.log(x))


def _sfm_sym_pair_probs(epsilon_left: float, epsilon_right: float) -> np.ndarray:
    """
    Theorem 4.8 transition probabilities for randomized merge g under Msym.

    Returns probs [t00, t01, t10, t11], where tab = P(g(a,b)=1).
    """
    e1 = float(epsilon_left)
    e2 = float(epsilon_right)
    e_star = _epsilon_star_two(e1, e2)
    q1 = 1.0 / (math.exp(e1) + 1.0)
    q2 = 1.0 / (math.exp(e2) + 1.0)
    q_star = 1.0 / (math.exp(e_star) + 1.0)
    k1 = np.array([[1.0 - q1, q1], [q1, 1.0 - q1]], dtype=np.float64)
    k2 = np.array([[1.0 - q2, q2], [q2, 1.0 - q2]], dtype=np.float64)
    v_star = np.array([q_star, 1.0 - q_star, 1.0 - q_star, 1.0 - q_star], dtype=np.float64)
    t = np.kron(np.linalg.inv(k1), np.linalg.inv(k2)) @ v_star
    return np.clip(t, 0.0, 1.0)


def _merge_sfm_sym_pair(
    bits_left: np.ndarray,
    epsilon_left: float,
    bits_right: np.ndarray,
    epsilon_right: float,
    *,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    if bits_left.shape != bits_right.shape:
        raise ValueError("left/right shapes must match")
    probs = _sfm_sym_pair_probs(float(epsilon_left), float(epsilon_right))
    idx = bits_left.astype(np.int8) * 2 + bits_right.astype(np.int8)
    p_one = probs[idx]
    merged = (rng.random(bits_left.shape) < p_one).astype(np.bool_)
    e_star = _epsilon_star_two(float(epsilon_left), float(epsilon_right))
    return merged, float(e_star)


def _reduce_local_msym_randomized(
    local_raw_bits: Sequence[np.ndarray],
    *,
    epsilon_local: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    p, q = _priv_params_sym(float(epsilon_local))
    states: List[Tuple[np.ndarray, float]] = [
        (_apply_rr(b, p=float(p), q=float(q), rng=rng), float(epsilon_local))
        for b in local_raw_bits
    ]
    while len(states) > 1:
        nxt: List[Tuple[np.ndarray, float]] = []
        i = 0
        while i < len(states):
            if i + 1 >= len(states):
                nxt.append(states[i])
                i += 1
                continue
            merged_bits, merged_eps = _merge_sfm_sym_pair(
                states[i][0],
                states[i][1],
                states[i + 1][0],
                states[i + 1][1],
                rng=rng,
            )
            nxt.append((merged_bits, merged_eps))
            i += 2
        states = nxt
    return states[0][0], float(states[0][1])


def _reduce_local_rr_deterministic(
    local_raw_bits: Sequence[np.ndarray],
    *,
    p: float,
    q: float,
    op: str,
    epsilon_local: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    noisy = [_apply_rr(b, p=float(p), q=float(q), rng=rng) for b in local_raw_bits]
    if len(noisy) == 0:
        raise ValueError("need non-empty local bits")
    out = np.array(noisy[0], dtype=np.bool_, copy=True)
    for nxt in noisy[1:]:
        if op == "or":
            np.logical_or(out, nxt, out=out)
        elif op == "xor":
            np.logical_xor(out, nxt, out=out)
        else:
            raise ValueError(f"unsupported op: {op!r}")
    e_star = _epsilon_star_pairwise_merge(float(epsilon_local), len(noisy))
    return out, float(e_star)


def _apply_rr(bits: np.ndarray, *, p: float, q: float, rng: np.random.Generator) -> np.ndarray:
    if bits.dtype != np.bool_:
        raise ValueError("bits must be bool ndarray")
    u = rng.random(bits.shape)
    return np.where(bits, u < float(p), u < float(q)).astype(np.bool_)


def _level_ones(bits: np.ndarray) -> np.ndarray:
    # shape: (levels,)
    return bits.astype(np.float64).sum(axis=0)


def _composite_ll_and_dll(
    n: float,
    *,
    ones_by_level: np.ndarray,
    buckets: int,
    levels: int,
    p: float,
    q: float,
) -> Tuple[float, float]:
    n_val = float(max(1e-9, n))
    b = float(buckets)
    ll = 0.0
    d_ll = 0.0
    pq = float(p) - float(q)
    for j in range(1, int(levels) + 1):
        c = float(ones_by_level[j - 1])
        g = _pcsa_gamma(j, buckets=int(buckets), levels=int(levels))
        logg = math.log(g)
        gn = math.exp(n_val * logg)
        pi = float(p) - pq * gn
        pi = min(max(pi, 1e-12), 1.0 - 1e-12)
        ll += c * math.log(pi) + (b - c) * math.log(1.0 - pi)
        dpi = -pq * gn * logg
        d_ll += (c / pi - (b - c) / (1.0 - pi)) * dpi
    return float(ll), float(d_ll)


def estimate_n_composite_mle(
    bits: np.ndarray,
    *,
    p: float,
    q: float,
    n_min: int,
    n_max: int,
) -> float:
    if bits.ndim != 2:
        raise ValueError("bits must be 2D [buckets, levels]")
    buckets, levels = int(bits.shape[0]), int(bits.shape[1])
    if int(n_min) < 1 or int(n_max) <= int(n_min):
        raise ValueError("require 1 <= n_min < n_max")

    ones = _level_ones(bits)
    lo = float(n_min)
    hi = float(n_max)
    _, d_lo = _composite_ll_and_dll(
        lo,
        ones_by_level=ones,
        buckets=buckets,
        levels=levels,
        p=float(p),
        q=float(q),
    )
    _, d_hi = _composite_ll_and_dll(
        hi,
        ones_by_level=ones,
        buckets=buckets,
        levels=levels,
        p=float(p),
        q=float(q),
    )
    if not (math.isfinite(d_lo) and math.isfinite(d_hi)):
        return _estimate_n_grid(
            ones,
            buckets=buckets,
            levels=levels,
            p=float(p),
            q=float(q),
            n_min=int(n_min),
            n_max=int(n_max),
        )
    if d_lo <= 0.0:
        return float(lo)
    if d_hi >= 0.0:
        return float(hi)

    for _ in range(70):
        mid = 0.5 * (lo + hi)
        _, d_mid = _composite_ll_and_dll(
            mid,
            ones_by_level=ones,
            buckets=buckets,
            levels=levels,
            p=float(p),
            q=float(q),
        )
        if not math.isfinite(d_mid):
            return _estimate_n_grid(
                ones,
                buckets=buckets,
                levels=levels,
                p=float(p),
                q=float(q),
                n_min=int(n_min),
                n_max=int(n_max),
            )
        if d_mid > 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _estimate_n_grid(
    ones_by_level: np.ndarray,
    *,
    buckets: int,
    levels: int,
    p: float,
    q: float,
    n_min: int,
    n_max: int,
) -> float:
    n0 = int(max(1, n_min))
    n1 = int(max(n0 + 1, n_max))
    xs = np.geomspace(float(n0), float(n1), num=512, dtype=np.float64)
    best_n = float(n0)
    best_ll = -float("inf")
    for n in xs:
        ll, _ = _composite_ll_and_dll(
            float(n),
            ones_by_level=ones_by_level,
            buckets=int(buckets),
            levels=int(levels),
            p=float(p),
            q=float(q),
        )
        if ll > best_ll:
            best_ll = float(ll)
            best_n = float(n)
    return float(best_n)


def _sample_partitioned_sets(
    n: int,
    *,
    merge_count: int,
    universe_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    n_int = int(n)
    if n_int <= 0:
        raise ValueError("n must be positive")
    if n_int > int(universe_size):
        raise ValueError("n cannot exceed universe_size for unique sampling")
    m = int(max(1, merge_count))
    values = rng.choice(int(universe_size), size=n_int, replace=False).astype(np.int64)
    rng.shuffle(values)
    chunks = np.array_split(values, m)
    return values, tuple(np.array(x, dtype=np.int64, copy=False) for x in chunks)


def _pcsa_union_bits(
    partitions: Sequence[np.ndarray],
    *,
    buckets: int,
    levels: int,
) -> np.ndarray:
    acc = PCSASketch(buckets=int(buckets), levels=int(levels))
    for part in partitions:
        sk = PCSASketch.from_tokens(part.tolist(), buckets=int(buckets), levels=int(levels))
        acc.merge_or(sk)
    return np.array(acc.bits, dtype=np.bool_, copy=True)


def _pcsa_local_bits(
    partitions: Sequence[np.ndarray],
    *,
    buckets: int,
    levels: int,
) -> Tuple[np.ndarray, ...]:
    out: List[np.ndarray] = []
    for part in partitions:
        sk = PCSASketch.from_tokens(part.tolist(), buckets=int(buckets), levels=int(levels))
        out.append(np.array(sk.bits, dtype=np.bool_, copy=True))
    return tuple(out)


def _rrmse(preds: Sequence[float], truth: float) -> float:
    t = float(max(1.0, truth))
    arr = np.array(preds, dtype=np.float64)
    return float(np.sqrt(np.mean((arr - float(truth)) ** 2)) / t)


def _rel_errors(preds: Sequence[float], truth: float) -> np.ndarray:
    t = float(max(1.0, truth))
    arr = np.array(preds, dtype=np.float64)
    return (arr - float(truth)) / t


def _hll_theory_rrmse_floor(memory_bits: int) -> float:
    """
    Asymptotic HLL relative standard error floor under matched memory:
      RSE ~= 1.04 / sqrt(m), with m = 2^p registers.
    """
    p_hll = match_hll_precision_for_bits(
        int(memory_bits),
        hash_bits=64,
        p_min=4,
        p_max=16,
    )
    m = float(2 ** int(p_hll))
    return float(1.04 / math.sqrt(m))


@dataclass(frozen=True)
class _IPWAuditResult:
    audit_rate: float
    sample_count: int
    effective_sample_size: float
    true_preference_loss: float
    true_violation_rate: float
    ipw_preference_loss: float
    ipw_preference_ci_low: float
    ipw_preference_ci_high: float
    ipw_violation_rate: float
    ipw_violation_ci_low: float
    ipw_violation_ci_high: float


def _build_inclusion_probs(
    preds: np.ndarray,
    *,
    audit_rate: float,
    n_max_est: int,
    floor: float,
    scheme: str,
) -> np.ndarray:
    n = int(preds.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    rate = float(max(0.0, min(1.0, audit_rate)))
    floor_clamped = float(max(1e-6, min(1.0, floor)))
    if rate >= 1.0:
        return np.ones((n,), dtype=np.float64)
    if rate <= 0.0:
        return np.full((n,), floor_clamped, dtype=np.float64)
    if scheme == "uniform":
        return np.full((n,), max(floor_clamped, rate), dtype=np.float64)
    if scheme == "prediction_stratified":
        # Prefer auditing trials with larger predicted counts. This uses a
        # query-time observable, so it is valid for design-based IPW.
        denom = math.log1p(float(max(2, n_max_est)))
        scaled = np.log1p(np.maximum(0.0, preds)) / denom
        base = np.clip(0.5 + scaled, 0.5, 1.5)
        base_mean = float(np.mean(base))
        if base_mean <= 0.0:
            return np.full((n,), max(floor_clamped, rate), dtype=np.float64)
        probs = rate * (base / base_mean)
        probs = np.clip(probs, floor_clamped, 1.0)
        return probs.astype(np.float64, copy=False)
    raise ValueError(f"unsupported IPW sampling scheme: {scheme!r}")


def _run_ipw_audit(
    *,
    preds: Sequence[float],
    truth_n: int,
    method_name: str,
    merge_count: int,
    epsilon: Optional[float],
    config: SFMComparisonConfig,
    rng: np.random.Generator,
) -> Tuple[_IPWAuditResult, ...]:
    if not config.enable_ipw:
        return tuple()
    pred_arr = np.array(preds, dtype=np.float64, copy=False)
    if pred_arr.size == 0:
        return tuple()

    rel = _rel_errors(pred_arr, float(truth_n))
    abs_rel = np.abs(rel)
    pref_loss = np.clip(abs_rel, 0.0, 1.0)
    violation = (abs_rel > float(config.ipw_violation_abs_rel_threshold)).astype(np.int8)
    true_pref = float(np.mean(pref_loss))
    true_viol = float(np.mean(violation))

    results: List[_IPWAuditResult] = []
    for audit_rate in (float(x) for x in config.ipw_audit_rates):
        pi = _build_inclusion_probs(
            pred_arr,
            audit_rate=float(audit_rate),
            n_max_est=int(config.n_max_est),
            floor=float(config.ipw_propensity_floor),
            scheme=str(config.ipw_sampling_scheme),
        )
        include = rng.random(pi.shape[0]) < pi
        samples: List[TreeSample] = []
        for idx, is_in in enumerate(include.tolist()):
            if not bool(is_in):
                continue
            samples.append(
                TreeSample(
                    doc_id=f"sfm-m{int(merge_count)}-n{int(truth_n)}",
                    node_id=f"{method_name}-trial-{idx}",
                    node_type=NodeType.MERGE,
                    violation=int(violation[idx]),
                    preference_loss=float(pref_loss[idx]),
                    sampling=SamplingMetadata(
                        document_propensity=1.0,
                        unit_propensity=float(pi[idx]),
                        label_propensity=1.0,
                        unit_kind=ObservationUnitKind.MERGE,
                    ),
                    metadata={
                        "method": str(method_name),
                        "epsilon": None if epsilon is None else float(epsilon),
                        "prediction": float(pred_arr[idx]),
                        "truth_n": int(truth_n),
                    },
                )
            )

        pref_hat = float(ipw_preference_loss(samples))
        pref_ci = ipw_preference_empirical_bernstein_ci(
            samples,
            delta=float(config.ipw_delta),
        )
        viol_hat = float(ipw_violation_rate(samples))
        viol_ci = ipw_violation_empirical_bernstein_ci(
            samples,
            delta=float(config.ipw_delta),
        )
        results.append(
            _IPWAuditResult(
                audit_rate=float(audit_rate),
                sample_count=int(len(samples)),
                effective_sample_size=float(effective_sample_size(samples)),
                true_preference_loss=float(true_pref),
                true_violation_rate=float(true_viol),
                ipw_preference_loss=pref_hat,
                ipw_preference_ci_low=float(pref_ci[0]),
                ipw_preference_ci_high=float(pref_ci[1]),
                ipw_violation_rate=viol_hat,
                ipw_violation_ci_low=float(viol_ci[0]),
                ipw_violation_ci_high=float(viol_ci[1]),
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class _RidgeDecoder:
    w: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray

    def predict(self, features: np.ndarray) -> float:
        x = np.array(features, dtype=np.float64, copy=False)
        z = (x - self.x_mean) / self.x_std
        y = float(np.dot(self.w, z))
        return float(math.exp(y))


def _sketch_features(bits: np.ndarray) -> np.ndarray:
    ones = _level_ones(bits)
    b = float(bits.shape[0])
    c = ones / b
    return np.concatenate([c, c * c], axis=0).astype(np.float64)


def _fit_ridge_decoder(
    *,
    epsilon: float,
    merge_count: int,
    config: SFMComparisonConfig,
    rng: np.random.Generator,
) -> _RidgeDecoder:
    x_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    n_min = int(min(config.n_values))
    n_max = int(max(config.n_values))
    log_min = math.log(float(max(1, n_min)))
    log_max = math.log(float(max(n_min + 1, n_max)))
    for _ in range(int(config.ridge_train_samples)):
        n = int(round(math.exp(float(rng.uniform(log_min, log_max)))))
        _, parts = _sample_partitioned_sets(
            n,
            merge_count=int(merge_count),
            universe_size=int(config.universe_size),
            rng=rng,
        )
        local_raw_bits = _pcsa_local_bits(
            parts,
            buckets=int(config.buckets),
            levels=int(config.levels),
        )
        p_local, q_local = _priv_params_sym(float(epsilon))
        priv_bits, _ = _reduce_local_rr_deterministic(
            local_raw_bits,
            p=float(p_local),
            q=float(q_local),
            op="or",
            epsilon_local=float(epsilon),
            rng=rng,
        )
        x_rows.append(_sketch_features(priv_bits))
        y_rows.append(math.log(float(n)))

    x = np.stack(x_rows, axis=0).astype(np.float64)
    y = np.array(y_rows, dtype=np.float64)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    z = (x - x_mean) / x_std
    # Linear ridge in standardized space (includes intercept as first feature via manual augment).
    ones = np.ones((z.shape[0], 1), dtype=np.float64)
    z_aug = np.concatenate([ones, z], axis=1)
    l2 = float(max(0.0, config.ridge_l2))
    reg = l2 * np.eye(z_aug.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    w = np.linalg.solve(z_aug.T @ z_aug + reg, z_aug.T @ y)
    return _RidgeDecoder(
        w=w,
        x_mean=np.concatenate([np.array([0.0], dtype=np.float64), x_mean]),
        x_std=np.concatenate([np.array([1.0], dtype=np.float64), x_std]),
    )


def _predict_ridge(decoder: _RidgeDecoder, bits: np.ndarray) -> float:
    feat = _sketch_features(bits)
    feat_aug = np.concatenate([np.array([1.0], dtype=np.float64), feat], axis=0)
    return float(decoder.predict(feat_aug))


def run_sfm_style_comparison(config: SFMComparisonConfig) -> ComparisonSummary:
    if len(config.n_values) == 0:
        raise ValueError("n_values must be non-empty")
    if len(config.epsilons) == 0:
        raise ValueError("epsilons must be non-empty")
    if len(config.merge_counts) == 0:
        raise ValueError("merge_counts must be non-empty")
    if config.enable_ipw and len(config.ipw_audit_rates) == 0:
        raise ValueError("ipw_audit_rates must be non-empty when IPW is enabled")
    if str(config.ipw_sampling_scheme) not in {"uniform", "prediction_stratified"}:
        raise ValueError("ipw_sampling_scheme must be one of {'uniform', 'prediction_stratified'}")

    rng = _set_seed(int(config.seed))
    rows: List[ComparisonRow] = []

    pcsa_bits = int(config.buckets * config.levels)
    theory_floor = _hll_theory_rrmse_floor(pcsa_bits) if config.include_theory_floor else None

    def _emit_rows(
        *,
        method_name: str,
        preds: Sequence[float],
        n_truth: int,
        merge_count_value: int,
        epsilon_value: Optional[float],
        epsilon_effective_value: Optional[float],
        rel_eff_value: Optional[float],
        p_hat: Optional[float],
        q_hat: Optional[float],
        p_target: Optional[float],
        q_target: Optional[float],
        channel_l1: Optional[float],
    ) -> None:
        arr = np.array(preds, dtype=np.float64)
        rel = _rel_errors(arr, float(n_truth))
        mse = float(np.mean((arr - float(n_truth)) ** 2))
        rrmse_val = float(_rrmse(arr, float(n_truth)))
        floor_val = None if theory_floor is None else float(theory_floor)
        floor_gap = None if floor_val is None else float(max(0.0, rrmse_val - floor_val))

        common = dict(
            method=str(method_name),
            n=int(n_truth),
            merge_count=int(merge_count_value),
            epsilon=None if epsilon_value is None else float(epsilon_value),
            epsilon_effective=None if epsilon_effective_value is None else float(epsilon_effective_value),
            memory_bits=int(pcsa_bits),
            n_trials=int(config.n_trials),
            rrmse=rrmse_val,
            mean_rel_error=float(np.mean(rel)),
            mean_abs_rel_error=float(np.mean(np.abs(rel))),
            median_abs_rel_error=float(np.median(np.abs(rel))),
            mse=float(mse),
            rel_eff_vs_sfm_sym=None if rel_eff_value is None else float(rel_eff_value),
            channel_p_hat=p_hat,
            channel_q_hat=q_hat,
            channel_p_target=p_target,
            channel_q_target=q_target,
            channel_calibration_l1=channel_l1,
            theory_rrmse_floor=floor_val,
            rrmse_gap_to_theory_floor=floor_gap,
        )

        ipw_runs = _run_ipw_audit(
            preds=arr,
            truth_n=int(n_truth),
            method_name=str(method_name),
            merge_count=int(merge_count_value),
            epsilon=None if epsilon_value is None else float(epsilon_value),
            config=config,
            rng=rng,
        )
        if not ipw_runs:
            rows.append(
                ComparisonRow(
                    **common,
                    ipw_audit_rate=None,
                    ipw_sample_count=None,
                    ipw_effective_sample_size=None,
                    true_preference_loss=None,
                    true_violation_rate=None,
                    ipw_preference_loss=None,
                    ipw_preference_ci_low=None,
                    ipw_preference_ci_high=None,
                    ipw_violation_rate=None,
                    ipw_violation_ci_low=None,
                    ipw_violation_ci_high=None,
                )
            )
            return

        for run in ipw_runs:
            rows.append(
                ComparisonRow(
                    **common,
                    ipw_audit_rate=float(run.audit_rate),
                    ipw_sample_count=int(run.sample_count),
                    ipw_effective_sample_size=float(run.effective_sample_size),
                    true_preference_loss=float(run.true_preference_loss),
                    true_violation_rate=float(run.true_violation_rate),
                    ipw_preference_loss=float(run.ipw_preference_loss),
                    ipw_preference_ci_low=float(run.ipw_preference_ci_low),
                    ipw_preference_ci_high=float(run.ipw_preference_ci_high),
                    ipw_violation_rate=float(run.ipw_violation_rate),
                    ipw_violation_ci_low=float(run.ipw_violation_ci_low),
                    ipw_violation_ci_high=float(run.ipw_violation_ci_high),
                )
            )

    # Per (epsilon, merge_count) optional learned decoder.
    ridge_cache: Dict[Tuple[float, int], _RidgeDecoder] = {}

    for merge_count in (int(x) for x in config.merge_counts):
        for eps in (float(x) for x in config.epsilons):
            if config.include_ours_ridge_sym:
                ridge_cache[(eps, merge_count)] = _fit_ridge_decoder(
                    epsilon=float(eps),
                    merge_count=int(merge_count),
                    config=config,
                    rng=rng,
                )

        for n in (int(x) for x in config.n_values):
            trial_union_bits: List[np.ndarray] = []
            trial_local_raw_bits: List[Tuple[np.ndarray, ...]] = []
            trial_full_vals: List[np.ndarray] = []
            for _ in range(int(config.n_trials)):
                full_vals, parts = _sample_partitioned_sets(
                    int(n),
                    merge_count=int(merge_count),
                    universe_size=int(config.universe_size),
                    rng=rng,
                )
                local_raw = _pcsa_local_bits(
                    parts,
                    buckets=int(config.buckets),
                    levels=int(config.levels),
                )
                union_bits = np.array(local_raw[0], dtype=np.bool_, copy=True)
                for nxt in local_raw[1:]:
                    np.logical_or(union_bits, nxt, out=union_bits)
                trial_union_bits.append(union_bits)
                trial_local_raw_bits.append(local_raw)
                trial_full_vals.append(full_vals)

            # Non-private methods.
            preds_np: Dict[str, List[float]] = {METHOD_PCSA_NON_PRIVATE: []}
            if config.include_hll_non_private:
                preds_np[METHOD_HLL_NON_PRIVATE] = []
            p_hll = match_hll_precision_for_bits(
                int(pcsa_bits),
                hash_bits=64,
                p_min=4,
                p_max=16,
            )
            for union_bits, full_vals in zip(trial_union_bits, trial_full_vals):
                n_hat_pcsa = estimate_n_composite_mle(
                    union_bits,
                    p=1.0,
                    q=0.0,
                    n_min=int(config.n_min_est),
                    n_max=int(config.n_max_est),
                )
                preds_np[METHOD_PCSA_NON_PRIVATE].append(float(n_hat_pcsa))
                if config.include_hll_non_private:
                    hll = HyperLogLogSketch(HLLConfig(precision=int(p_hll), hash_bits=64))
                    for tok in full_vals.tolist():
                        hll.add(int(tok))
                    preds_np[METHOD_HLL_NON_PRIVATE].append(float(hll.estimate()))

            for method_name, method_preds in preds_np.items():
                _emit_rows(
                    method_name=str(method_name),
                    preds=method_preds,
                    n_truth=int(n),
                    merge_count_value=int(merge_count),
                    epsilon_value=None,
                    epsilon_effective_value=None,
                    rel_eff_value=None,
                    p_hat=None,
                    q_hat=None,
                    p_target=None,
                    q_target=None,
                    channel_l1=None,
                )

            # Private methods by epsilon, explicit local-noise + explicit merge.
            for eps in (float(x) for x in config.epsilons):
                eps_star_formula = float(_epsilon_star_pairwise_merge(eps, merge_count))
                p_sym_assumed, q_sym_assumed = _priv_params_sym(eps_star_formula)

                preds_priv: Dict[str, List[float]] = {
                    METHOD_SFM_SYM: [],
                    METHOD_SFM_XOR: [],
                    METHOD_SYM_LOCAL_OR: [],
                    METHOD_SYM_LOCAL_XOR: [],
                }
                if config.include_ours_ridge_sym:
                    preds_priv[METHOD_OURS_RIDGE_SYM] = []

                # Channel calibration accumulators.
                ch_ones_sum: Dict[str, float] = {k: 0.0 for k in preds_priv.keys()}
                ch_zeros_sum: Dict[str, float] = {k: 0.0 for k in preds_priv.keys()}
                ch_ones_cnt: Dict[str, float] = {k: 0.0 for k in preds_priv.keys()}
                ch_zeros_cnt: Dict[str, float] = {k: 0.0 for k in preds_priv.keys()}
                ch_p_target_sum: Dict[str, float] = {k: 0.0 for k in preds_priv.keys()}
                ch_q_target_sum: Dict[str, float] = {k: 0.0 for k in preds_priv.keys()}
                ch_target_n: Dict[str, int] = {k: 0 for k in preds_priv.keys()}

                for union_bits, local_raw in zip(trial_union_bits, trial_local_raw_bits):
                    # 1) Proper SFM(sym): local Msym + randomized merge g.
                    merged_sym, eps_eff_sym = _reduce_local_msym_randomized(
                        local_raw,
                        epsilon_local=float(eps),
                        rng=rng,
                    )
                    p_eff_sym, q_eff_sym = _priv_params_sym(float(eps_eff_sym))
                    n_hat_sym = estimate_n_composite_mle(
                        merged_sym,
                        p=float(p_eff_sym),
                        q=float(q_eff_sym),
                        n_min=int(config.n_min_est),
                        n_max=int(config.n_max_est),
                    )
                    preds_priv[METHOD_SFM_SYM].append(float(n_hat_sym))

                    # 2) Proper xor construction: local Mxor + deterministic xor merge.
                    p_local_xor, q_local_xor = _priv_params_xor(float(eps))
                    merged_xor, eps_eff_xor = _reduce_local_rr_deterministic(
                        local_raw,
                        p=float(p_local_xor),
                        q=float(q_local_xor),
                        op="xor",
                        epsilon_local=float(eps),
                        rng=rng,
                    )
                    p_eff_xor, q_eff_xor = _priv_params_xor(float(eps_eff_xor))
                    n_hat_xor = estimate_n_composite_mle(
                        merged_xor,
                        p=float(p_eff_xor),
                        q=float(q_eff_xor),
                        n_min=int(config.n_min_est),
                        n_max=int(config.n_max_est),
                    )
                    preds_priv[METHOD_SFM_XOR].append(float(n_hat_xor))

                    # 3) Counterfactual deterministic merges under local Msym (theorem-aligned negatives).
                    p_local_sym, q_local_sym = _priv_params_sym(float(eps))
                    merged_det_or, _ = _reduce_local_rr_deterministic(
                        local_raw,
                        p=float(p_local_sym),
                        q=float(q_local_sym),
                        op="or",
                        epsilon_local=float(eps),
                        rng=rng,
                    )
                    merged_det_xor, _ = _reduce_local_rr_deterministic(
                        local_raw,
                        p=float(p_local_sym),
                        q=float(q_local_sym),
                        op="xor",
                        epsilon_local=float(eps),
                        rng=rng,
                    )
                    # Use the commutative-target channel MLE (intentionally exposes merge mismatch).
                    n_hat_det_or = estimate_n_composite_mle(
                        merged_det_or,
                        p=float(p_sym_assumed),
                        q=float(q_sym_assumed),
                        n_min=int(config.n_min_est),
                        n_max=int(config.n_max_est),
                    )
                    n_hat_det_xor = estimate_n_composite_mle(
                        merged_det_xor,
                        p=float(p_sym_assumed),
                        q=float(q_sym_assumed),
                        n_min=int(config.n_min_est),
                        n_max=int(config.n_max_est),
                    )
                    preds_priv[METHOD_SYM_LOCAL_OR].append(float(n_hat_det_or))
                    preds_priv[METHOD_SYM_LOCAL_XOR].append(float(n_hat_det_xor))

                    if config.include_ours_ridge_sym:
                        dec = ridge_cache[(eps, merge_count)]
                        preds_priv[METHOD_OURS_RIDGE_SYM].append(float(_predict_ridge(dec, merged_det_or)))

                    # Channel calibration wrt true union occupancy.
                    mask1 = union_bits
                    mask0 = np.logical_not(union_bits)
                    c1 = float(mask1.sum())
                    c0 = float(mask0.sum())
                    obs_bits = {
                        METHOD_SFM_SYM: (merged_sym, float(p_eff_sym), float(q_eff_sym)),
                        METHOD_SFM_XOR: (merged_xor, float(p_eff_xor), float(q_eff_xor)),
                        METHOD_SYM_LOCAL_OR: (merged_det_or, float(p_sym_assumed), float(q_sym_assumed)),
                        METHOD_SYM_LOCAL_XOR: (merged_det_xor, float(p_sym_assumed), float(q_sym_assumed)),
                    }
                    if config.include_ours_ridge_sym:
                        obs_bits[METHOD_OURS_RIDGE_SYM] = (
                            merged_det_or,
                            float(p_sym_assumed),
                            float(q_sym_assumed),
                        )
                    for mname, (mbits, p_tar, q_tar) in obs_bits.items():
                        if c1 > 0.0:
                            ch_ones_sum[mname] += float(mbits[mask1].sum())
                            ch_ones_cnt[mname] += c1
                        if c0 > 0.0:
                            ch_zeros_sum[mname] += float(mbits[mask0].sum())
                            ch_zeros_cnt[mname] += c0
                        ch_p_target_sum[mname] += float(p_tar)
                        ch_q_target_sum[mname] += float(q_tar)
                        ch_target_n[mname] += 1

                sym_arr = np.array(preds_priv[METHOD_SFM_SYM], dtype=np.float64)
                sym_mse = float(np.mean((sym_arr - float(n)) ** 2))
                sym_mse = max(sym_mse, 1e-12)

                for method_name, method_preds in preds_priv.items():
                    arr = np.array(method_preds, dtype=np.float64)
                    mse = float(np.mean((arr - float(n)) ** 2))
                    p_hat = (
                        float(ch_ones_sum[method_name] / ch_ones_cnt[method_name])
                        if ch_ones_cnt[method_name] > 0.0
                        else None
                    )
                    q_hat = (
                        float(ch_zeros_sum[method_name] / ch_zeros_cnt[method_name])
                        if ch_zeros_cnt[method_name] > 0.0
                        else None
                    )
                    p_tar = (
                        float(ch_p_target_sum[method_name] / max(1, ch_target_n[method_name]))
                        if ch_target_n[method_name] > 0
                        else None
                    )
                    q_tar = (
                        float(ch_q_target_sum[method_name] / max(1, ch_target_n[method_name]))
                        if ch_target_n[method_name] > 0
                        else None
                    )
                    ch_l1 = (
                        float(abs(float(p_hat) - float(p_tar)) + abs(float(q_hat) - float(q_tar)))
                        if (p_hat is not None and q_hat is not None and p_tar is not None and q_tar is not None)
                        else None
                    )
                    _emit_rows(
                        method_name=str(method_name),
                        preds=method_preds,
                        n_truth=int(n),
                        merge_count_value=int(merge_count),
                        epsilon_value=float(eps),
                        epsilon_effective_value=float(eps_star_formula),
                        rel_eff_value=float(mse / sym_mse),
                        p_hat=p_hat,
                        q_hat=q_hat,
                        p_target=p_tar,
                        q_target=q_tar,
                        channel_l1=ch_l1,
                    )

    cfg_dict = asdict(config)
    cfg_dict["pcsa_memory_bits"] = int(pcsa_bits)
    return ComparisonSummary(config=cfg_dict, rows=tuple(rows))


__all__ = [
    "ComparisonRow",
    "ComparisonSummary",
    "METHOD_HLL_NON_PRIVATE",
    "METHOD_OURS_RIDGE_SYM",
    "METHOD_PCSA_NON_PRIVATE",
    "METHOD_SFM_SYM",
    "METHOD_SFM_XOR",
    "METHOD_SYM_LOCAL_OR",
    "METHOD_SYM_LOCAL_XOR",
    "SFMComparisonConfig",
    "estimate_n_composite_mle",
    "run_sfm_style_comparison",
]
