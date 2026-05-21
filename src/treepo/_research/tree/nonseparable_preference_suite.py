"""Non-separable preference DGP separation suite with explicit optimization gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import statistics
from typing import Dict, List, Sequence, Tuple


ARM_ORACLE = "oracle"
ARM_SUPPORTED = "supported_merge_safe"
ARM_UNDERSUPPORTED = "undersupported_sketch"
ARM_WRONG_CHUNKER = "right_rule_wrong_chunker"
ARM_NAIVE = "naive_non_merge_safe"
ARM_ORDER: Tuple[str, ...] = (
    ARM_ORACLE,
    ARM_SUPPORTED,
    ARM_UNDERSUPPORTED,
    ARM_WRONG_CHUNKER,
    ARM_NAIVE,
)

DGP1_AND = "dgp1_complementarity_and"
DGP2_BOUNDARY = "dgp2_boundary_interaction"
DGP_ORDER: Tuple[str, ...] = (DGP1_AND, DGP2_BOUNDARY)


@dataclass(frozen=True)
class NonseparableSuiteConfig:
    n_replicates: int = 80
    n_pairs_per_replicate: int = 300
    seed: int = 0
    beta: float = 3.0
    and_left_threshold: int = 3
    and_right_threshold: int = 3
    and_count_max: int = 7
    dgp2_vocab_size: int = 6
    dgp2_seq_len: int = 24
    dgp2_lambda: float = 2.0
    hard_regime: bool = False
    effect_gate: float = 0.05
    strong_effect_gate: float = 0.10
    bound_tolerance: float = 0.02


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    se: float
    ci95_low: float
    ci95_high: float
    n: int


@dataclass(frozen=True)
class ArmSummary:
    arm: str
    mean_gap_to_oracle_loss: float
    mean_gap_to_oracle_loss_ci95_low: float
    mean_gap_to_oracle_loss_ci95_high: float
    mean_utility_regret: float
    mean_utility_regret_ci95_low: float
    mean_utility_regret_ci95_high: float
    mean_conditional_event_bias: float
    mean_conditional_event_bias_ci95_low: float
    mean_conditional_event_bias_ci95_high: float
    empirical_coverage: float
    mean_bound_envelope: float
    mean_bound_envelope_ci95_low: float
    mean_bound_envelope_ci95_high: float
    bound_consistent: bool
    bound_gap_excess: float


@dataclass(frozen=True)
class SeparationCheck:
    arm: str
    mean_delta_supported_vs_arm: float
    ci95_low: float
    ci95_high: float
    passes_gate: bool


@dataclass(frozen=True)
class DgpResult:
    name: str
    utility_range: Tuple[float, float]
    arms: Tuple[ArmSummary, ...]
    separation_checks: Tuple[SeparationCheck, ...]
    strong_separation_pass: bool
    flagged_cells: Tuple[str, ...]


@dataclass(frozen=True)
class NonseparableSuiteResult:
    config: NonseparableSuiteConfig
    dgps: Tuple[DgpResult, ...]

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "dgps": [asdict(x) for x in self.dgps],
        }


def _normal_ci95(xs: Sequence[float]) -> MetricSummary:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    n = len(vals)
    if n == 0:
        return MetricSummary(
            mean=float("nan"),
            se=float("nan"),
            ci95_low=float("nan"),
            ci95_high=float("nan"),
            n=0,
        )
    mu = float(sum(vals) / float(n))
    if n <= 1:
        return MetricSummary(mean=mu, se=0.0, ci95_low=mu, ci95_high=mu, n=n)
    var = float(sum((v - mu) ** 2 for v in vals) / float(n - 1))
    se = math.sqrt(max(0.0, var) / float(n))
    z = 1.96
    return MetricSummary(mean=mu, se=se, ci95_low=mu - z * se, ci95_high=mu + z * se, n=n)


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    nn = int(max(0, n))
    if nn <= 0:
        return (0.0, 1.0)
    ss = int(max(0, min(successes, nn)))
    p = float(ss) / float(nn)
    z2 = z * z
    denom = 1.0 + z2 / float(nn)
    center = (p + z2 / (2.0 * float(nn))) / denom
    half = (
        z
        * math.sqrt((p * (1.0 - p) + z2 / (4.0 * float(nn))) / float(nn))
        / denom
    )
    return (max(0.0, center - half), min(1.0, center + half))


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _dpo_like_gap(true_a: float, true_b: float, pred_a: float, pred_b: float, beta: float) -> float:
    y = 1.0 if true_a >= true_b else -1.0
    pred_margin = float(pred_a) - float(pred_b)
    true_margin = float(true_a) - float(true_b)
    loss = math.log1p(math.exp(-float(beta) * y * pred_margin))
    oracle_loss = math.log1p(math.exp(-float(beta) * y * true_margin))
    return float(loss - oracle_loss)


def _sample_dgp1_candidate(rng: random.Random, cfg: NonseparableSuiteConfig) -> dict:
    left = rng.randint(0, int(cfg.and_count_max))
    right = rng.randint(0, int(cfg.and_count_max))
    if cfg.hard_regime:
        left = max(0, min(int(cfg.and_count_max), left + rng.choice((-1, 0, 1))))
        right = max(0, min(int(cfg.and_count_max), right + rng.choice((-1, 0, 1))))
    proxy_l = left + rng.uniform(-1.2, 1.2)
    proxy_r = right + rng.uniform(-1.2, 1.2)
    return {
        "left": int(left),
        "right": int(right),
        "proxy_left": float(proxy_l),
        "proxy_right": float(proxy_r),
    }


def _dgp1_true_utility(cand: dict, cfg: NonseparableSuiteConfig) -> float:
    left_ok = int(cand["left"]) >= int(cfg.and_left_threshold)
    right_ok = int(cand["right"]) >= int(cfg.and_right_threshold)
    return 1.0 if (left_ok and right_ok) else 0.0


def _dgp1_estimate_for_arm(cand: dict, arm: str, cfg: NonseparableSuiteConfig, rng: random.Random) -> Tuple[float, dict]:
    left = int(cand["left"])
    right = int(cand["right"])
    true_u = _dgp1_true_utility(cand, cfg)
    left_thr = int(cfg.and_left_threshold)
    right_thr = int(cfg.and_right_threshold)
    support_insufficient = 0.0
    dropped_side = 0.0

    if arm in (ARM_ORACLE, ARM_SUPPORTED):
        est = true_u
    elif arm == ARM_UNDERSUPPORTED:
        support_insufficient = 1.0
        est = 1.0 if max(left, right) >= max(left_thr, right_thr) else 0.0
    elif arm == ARM_WRONG_CHUNKER:
        dropped_side = 1.0
        keep_left = bool(float(cand["proxy_left"]) >= float(cand["proxy_right"]))
        obs_left = left if keep_left else 0
        obs_right = right if (not keep_left) else 0
        est = 1.0 if (obs_left >= left_thr and obs_right >= right_thr) else 0.0
    elif arm == ARM_NAIVE:
        support_insufficient = 1.0
        scale = float(max(1, left_thr + right_thr))
        est = max(0.0, min(1.0, float(left + right) / scale))
    else:
        raise ValueError(f"unsupported arm: {arm}")

    miss_required = 1.0 if (true_u >= 0.5 and est < 0.5) else 0.0
    envelope = min(1.0, 0.60 * miss_required + 0.30 * support_insufficient + 0.30 * dropped_side)
    components = {
        "support_insufficient": float(support_insufficient),
        "dropped_side": float(dropped_side),
        "miss_required_event": float(miss_required),
        "bound_envelope": float(envelope),
    }
    if cfg.hard_regime and arm == ARM_SUPPORTED:
        est = max(0.0, min(1.0, est + rng.uniform(-0.03, 0.03)))
    return (float(est), components)


def _dgp2_weights(cfg: NonseparableSuiteConfig) -> Tuple[List[float], List[List[float]]]:
    v = int(cfg.dgp2_vocab_size)
    theta = [0.0 for _ in range(v)]
    for i in range(v):
        theta[i] = (float(i) - 0.5 * float(v - 1)) / float(max(1, v - 1))
    w = [[0.0 for _ in range(v)] for _ in range(v)]
    if v >= 4:
        w[0][1] = 6.0
        w[1][0] = 3.0
        w[2][3] = -6.0
        w[3][2] = -3.0
    return (theta, w)


def _sample_dgp2_candidate(rng: random.Random, cfg: NonseparableSuiteConfig) -> dict:
    v = int(cfg.dgp2_vocab_size)
    n = int(cfg.dgp2_seq_len)
    split = n // 2
    left = [rng.randrange(v) for _ in range(split)]
    right = [rng.randrange(v) for _ in range(n - split)]
    if split > 0 and n - split > 0 and v >= 4:
        trigger = 0.70 if cfg.hard_regime else 0.45
        if rng.random() < trigger:
            if rng.random() < 0.5:
                left[-1] = 0
                right[0] = 1
            else:
                left[-1] = 2
                right[0] = 3
    seq = left + right
    proxy_left = sum(1 for x in left if x in (0, 1))
    proxy_right = sum(1 for x in right if x in (0, 1))
    return {
        "seq": tuple(int(x) for x in seq),
        "split": int(split),
        "proxy_left": float(proxy_left) + rng.uniform(-1.0, 1.0),
        "proxy_right": float(proxy_right) + rng.uniform(-1.0, 1.0),
    }


def _dgp2_components(cand: dict, cfg: NonseparableSuiteConfig) -> Tuple[float, float, float]:
    theta, w = _dgp2_weights(cfg)
    seq = list(cand["seq"])
    n = len(seq)
    if n == 0:
        return (0.0, 0.0, 0.0)
    v = int(cfg.dgp2_vocab_size)
    uni = [0.0 for _ in range(v)]
    for x in seq:
        uni[int(x)] += 1.0
    uni = [u / float(n) for u in uni]
    uni_term = sum(theta[i] * uni[i] for i in range(v))

    if n <= 1:
        return (float(uni_term), 0.0, 0.0)
    denom = float(n - 1)
    big_term = 0.0
    cross_term = 0.0
    split = int(cand["split"])
    for i in range(n - 1):
        a = int(seq[i])
        b = int(seq[i + 1])
        contrib = w[a][b] / denom
        big_term += contrib
        if i == split - 1:
            cross_term += contrib
    return (float(uni_term), float(big_term), float(cross_term))


def _dgp2_true_utility(cand: dict, cfg: NonseparableSuiteConfig) -> float:
    uni, big, _ = _dgp2_components(cand, cfg)
    raw = float(uni) + float(cfg.dgp2_lambda) * float(big)
    return float(_sigmoid(2.5 * raw))


def _dgp2_estimate_for_arm(cand: dict, arm: str, cfg: NonseparableSuiteConfig, rng: random.Random) -> Tuple[float, dict]:
    uni, big, cross = _dgp2_components(cand, cfg)
    lam = float(cfg.dgp2_lambda)
    true_raw = uni + lam * big
    support_insufficient = 0.0
    missing_boundary = 0.0
    dropped_mass = 0.0
    if arm in (ARM_ORACLE, ARM_SUPPORTED):
        raw = true_raw
    elif arm == ARM_UNDERSUPPORTED:
        support_insufficient = 1.0
        missing_boundary = abs(lam * cross)
        raw = uni + lam * (big - cross)
    elif arm == ARM_WRONG_CHUNKER:
        split = int(cand["split"])
        seq = list(cand["seq"])
        keep_left = bool(float(cand["proxy_left"]) < float(cand["proxy_right"]))
        kept = seq[:split] if keep_left else seq[split:]
        dropped_mass = 1.0 - (float(len(kept)) / float(max(1, len(seq))))
        if len(kept) == 0:
            raw = 0.0
        else:
            pseudo = {"seq": tuple(int(x) for x in kept), "split": len(kept) // 2}
            uni_k, big_k, _ = _dgp2_components(pseudo, cfg)
            raw = 0.5 * (uni_k + lam * big_k) - 0.20
        missing_boundary = abs(lam * cross)
    elif arm == ARM_NAIVE:
        support_insufficient = 1.0
        missing_boundary = abs(lam * big)
        raw = uni
    else:
        raise ValueError(f"unsupported arm: {arm}")

    est = float(_sigmoid(2.5 * raw))
    if cfg.hard_regime and arm == ARM_SUPPORTED:
        est = max(0.0, min(1.0, est + rng.uniform(-0.02, 0.02)))

    bound = min(1.0, missing_boundary + dropped_mass + 0.25 * support_insufficient)
    components = {
        "support_insufficient": float(support_insufficient),
        "missing_boundary_term": float(missing_boundary),
        "dropped_mass": float(dropped_mass),
        "bound_envelope": float(bound),
    }
    return (float(est), components)


def _simulate_dgp(
    dgp_name: str,
    cfg: NonseparableSuiteConfig,
    rng: random.Random,
) -> DgpResult:
    by_arm_gap: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    by_arm_regret: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    by_arm_event_bias: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    by_arm_coverage_hit: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    by_arm_bound: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}

    for rep in range(int(cfg.n_replicates)):
        rep_rng = random.Random(int(cfg.seed) + (1_000_000 * (1 + DGP_ORDER.index(dgp_name))) + rep)
        pair_gap_sum = {a: 0.0 for a in ARM_ORDER}
        pair_regret_sum = {a: 0.0 for a in ARM_ORDER}
        pair_bound_sum = {a: 0.0 for a in ARM_ORDER}
        pred_event_count = {a: 0 for a in ARM_ORDER}
        true_event_count = 0
        n_events = 0

        for _ in range(int(cfg.n_pairs_per_replicate)):
            if dgp_name == DGP1_AND:
                cand_a = _sample_dgp1_candidate(rep_rng, cfg)
                cand_b = _sample_dgp1_candidate(rep_rng, cfg)
                true_a = _dgp1_true_utility(cand_a, cfg)
                true_b = _dgp1_true_utility(cand_b, cfg)
                est_by_arm = {}
                bound_by_arm = {}
                for arm in ARM_ORDER:
                    est_a, comp_a = _dgp1_estimate_for_arm(cand_a, arm, cfg, rep_rng)
                    est_b, comp_b = _dgp1_estimate_for_arm(cand_b, arm, cfg, rep_rng)
                    est_by_arm[arm] = (est_a, est_b)
                    bound_by_arm[arm] = max(
                        float(comp_a["bound_envelope"]),
                        float(comp_b["bound_envelope"]),
                    )
            elif dgp_name == DGP2_BOUNDARY:
                cand_a = _sample_dgp2_candidate(rep_rng, cfg)
                cand_b = _sample_dgp2_candidate(rep_rng, cfg)
                true_a = _dgp2_true_utility(cand_a, cfg)
                true_b = _dgp2_true_utility(cand_b, cfg)
                est_by_arm = {}
                bound_by_arm = {}
                for arm in ARM_ORDER:
                    est_a, comp_a = _dgp2_estimate_for_arm(cand_a, arm, cfg, rep_rng)
                    est_b, comp_b = _dgp2_estimate_for_arm(cand_b, arm, cfg, rep_rng)
                    est_by_arm[arm] = (est_a, est_b)
                    bound_by_arm[arm] = max(
                        float(comp_a["bound_envelope"]),
                        float(comp_b["bound_envelope"]),
                    )
            else:
                raise ValueError(f"unsupported dgp_name={dgp_name!r}")

            true_best = max(float(true_a), float(true_b))
            true_event_count += int(true_a >= 0.5) + int(true_b >= 0.5)
            n_events += 2

            for arm in ARM_ORDER:
                pred_a, pred_b = est_by_arm[arm]
                chosen_true = float(true_a) if pred_a >= pred_b else float(true_b)
                regret = float(true_best - chosen_true)
                gap = _dpo_like_gap(true_a, true_b, pred_a, pred_b, beta=float(cfg.beta))
                pair_gap_sum[arm] += float(gap)
                pair_regret_sum[arm] += float(regret)
                pair_bound_sum[arm] += float(bound_by_arm[arm])
                pred_event_count[arm] += int(pred_a >= 0.5) + int(pred_b >= 0.5)

        n_pairs = float(max(1, int(cfg.n_pairs_per_replicate)))
        true_event_rate = float(true_event_count) / float(max(1, n_events))
        for arm in ARM_ORDER:
            gap_rep = pair_gap_sum[arm] / n_pairs
            reg_rep = pair_regret_sum[arm] / n_pairs
            bnd_rep = pair_bound_sum[arm] / n_pairs
            pred_rate = float(pred_event_count[arm]) / float(max(1, n_events))
            bias_rep = pred_rate - true_event_rate
            lo, hi = _wilson_interval(pred_event_count[arm], max(1, n_events))
            cov_hit = 1.0 if (lo <= true_event_rate <= hi) else 0.0

            by_arm_gap[arm].append(float(gap_rep))
            by_arm_regret[arm].append(float(reg_rep))
            by_arm_event_bias[arm].append(float(bias_rep))
            by_arm_coverage_hit[arm].append(float(cov_hit))
            by_arm_bound[arm].append(float(bnd_rep))

    arm_summaries: List[ArmSummary] = []
    flagged: List[str] = []
    for arm in ARM_ORDER:
        gap_s = _normal_ci95(by_arm_gap[arm])
        reg_s = _normal_ci95(by_arm_regret[arm])
        bias_s = _normal_ci95(by_arm_event_bias[arm])
        bnd_s = _normal_ci95(by_arm_bound[arm])
        coverage = _normal_ci95(by_arm_coverage_hit[arm]).mean
        bound_excess = reg_s.mean - bnd_s.mean
        bound_consistent = bool(bound_excess <= float(cfg.bound_tolerance))
        if not bound_consistent:
            flagged.append(f"{dgp_name}:{arm}:bound_excess={bound_excess:.4f}")
        arm_summaries.append(
            ArmSummary(
                arm=arm,
                mean_gap_to_oracle_loss=gap_s.mean,
                mean_gap_to_oracle_loss_ci95_low=gap_s.ci95_low,
                mean_gap_to_oracle_loss_ci95_high=gap_s.ci95_high,
                mean_utility_regret=reg_s.mean,
                mean_utility_regret_ci95_low=reg_s.ci95_low,
                mean_utility_regret_ci95_high=reg_s.ci95_high,
                mean_conditional_event_bias=bias_s.mean,
                mean_conditional_event_bias_ci95_low=bias_s.ci95_low,
                mean_conditional_event_bias_ci95_high=bias_s.ci95_high,
                empirical_coverage=float(coverage),
                mean_bound_envelope=bnd_s.mean,
                mean_bound_envelope_ci95_low=bnd_s.ci95_low,
                mean_bound_envelope_ci95_high=bnd_s.ci95_high,
                bound_consistent=bool(bound_consistent),
                bound_gap_excess=float(bound_excess),
            )
        )

    supported_gaps = by_arm_gap[ARM_SUPPORTED]
    separation_rows: List[SeparationCheck] = []
    strong_pass = False
    for arm in (ARM_UNDERSUPPORTED, ARM_WRONG_CHUNKER, ARM_NAIVE):
        delta = [float(a - b) for a, b in zip(by_arm_gap[arm], supported_gaps)]
        d = _normal_ci95(delta)
        passes = bool(d.mean >= float(cfg.effect_gate) and d.ci95_low > 0.0)
        strong_here = bool(d.mean >= float(cfg.strong_effect_gate) and d.ci95_low > 0.0)
        strong_pass = bool(strong_pass or strong_here)
        if not passes:
            flagged.append(f"{dgp_name}:{arm}:separation_gate_failed")
        separation_rows.append(
            SeparationCheck(
                arm=arm,
                mean_delta_supported_vs_arm=d.mean,
                ci95_low=d.ci95_low,
                ci95_high=d.ci95_high,
                passes_gate=passes,
            )
        )

    return DgpResult(
        name=dgp_name,
        utility_range=(0.0, 1.0),
        arms=tuple(arm_summaries),
        separation_checks=tuple(separation_rows),
        strong_separation_pass=bool(strong_pass),
        flagged_cells=tuple(sorted(flagged)),
    )


def run_nonseparable_preference_suite(config: NonseparableSuiteConfig) -> NonseparableSuiteResult:
    if int(config.n_replicates) <= 0:
        raise ValueError("n_replicates must be >= 1")
    if int(config.n_pairs_per_replicate) <= 0:
        raise ValueError("n_pairs_per_replicate must be >= 1")
    if float(config.beta) <= 0.0:
        raise ValueError("beta must be > 0")
    rng = random.Random(int(config.seed))
    _ = rng.random()  # deterministic burn-in, keeps interface parity if extended later.
    out = [_simulate_dgp(dgp_name, config, rng) for dgp_name in DGP_ORDER]
    return NonseparableSuiteResult(config=config, dgps=tuple(out))


__all__ = [
    "ARM_ORACLE",
    "ARM_SUPPORTED",
    "ARM_UNDERSUPPORTED",
    "ARM_WRONG_CHUNKER",
    "ARM_NAIVE",
    "ARM_ORDER",
    "DGP1_AND",
    "DGP2_BOUNDARY",
    "DGP_ORDER",
    "NonseparableSuiteConfig",
    "MetricSummary",
    "ArmSummary",
    "SeparationCheck",
    "DgpResult",
    "NonseparableSuiteResult",
    "run_nonseparable_preference_suite",
]
