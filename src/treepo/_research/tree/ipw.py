"""
IPW utilities for tree-level audit and preference estimation.

This module mirrors the main executable objects from Lean's TreeIPW layer:
- TreeSample with joint propensity and inverse-probability weight
- Hajek/IPW estimators for violation rates and preference losses
- Honest sample-splitting and K-fold helpers
- Empirical Bernstein confidence intervals for bounded targets

The implementation is deliberately lightweight (pure Python) so it can be used
both in offline analysis and in runtime audit reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from treepo._research.core.logged_supervision import (
    LoggedLabelObservation,
    ObservationUnitKind,
    SamplingMetadata,
)

MIN_PROPENSITY = 1e-12
MAX_PROPENSITY = 1.0


class NodeType(Enum):
    """Node type for logged TreePO audit/training samples."""

    LEAF = "leaf"
    MERGE = "merge"
    RESUMMARY = "resummary"
    SUBSTITUTION = "substitution"


def node_type_to_observation_unit_kind(node_type: NodeType) -> ObservationUnitKind:
    mapping = {
        NodeType.LEAF: ObservationUnitKind.LEAF,
        NodeType.MERGE: ObservationUnitKind.MERGE,
        NodeType.RESUMMARY: ObservationUnitKind.RESUMMARY,
        NodeType.SUBSTITUTION: ObservationUnitKind.SUBSTITUTION,
    }
    return mapping[node_type]


def observation_unit_kind_to_node_type(unit_kind: ObservationUnitKind) -> NodeType:
    mapping = {
        ObservationUnitKind.LEAF: NodeType.LEAF,
        ObservationUnitKind.INTERNAL: NodeType.MERGE,
        ObservationUnitKind.MERGE: NodeType.MERGE,
        ObservationUnitKind.RESUMMARY: NodeType.RESUMMARY,
        ObservationUnitKind.SUBSTITUTION: NodeType.SUBSTITUTION,
    }
    return mapping[unit_kind]


@dataclass(init=False)
class TreeSample:
    """
    Logged tree-level sample for IPW estimation.

    `violation` should be a binary indicator (0/1). `preference_loss` is
    bounded to [0, 1] in OPS usage (e.g., discrepancy-style losses).
    """

    doc_id: str
    node_id: str
    node_type: NodeType
    violation: int
    preference_loss: float = 0.0
    sampling: SamplingMetadata = field(default_factory=SamplingMetadata)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        doc_id: str,
        node_id: str,
        node_type: NodeType,
        violation: int,
        preference_loss: float = 0.0,
        sampling: Optional[SamplingMetadata] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.doc_id = doc_id
        self.node_id = node_id
        self.node_type = node_type
        self.violation = violation
        self.preference_loss = preference_loss
        self.sampling = sampling if sampling is not None else SamplingMetadata()
        self.metadata = dict(metadata or {})
        self.__post_init__()

    def __post_init__(self) -> None:
        if isinstance(self.violation, bool):
            self.violation = int(self.violation)
        if self.violation not in (0, 1):
            raise ValueError(f"violation must be 0/1 (got {self.violation!r})")
        if not math.isfinite(self.preference_loss):
            raise ValueError(f"preference_loss must be finite (got {self.preference_loss!r})")
        if not isinstance(self.node_type, NodeType):
            self.node_type = NodeType(str(self.node_type))
        if not isinstance(self.sampling, SamplingMetadata):
            self.sampling = SamplingMetadata.from_dict(self.sampling)
        if self.sampling.unit_kind is None:
            self.sampling = self.sampling.with_updates(
                unit_kind=node_type_to_observation_unit_kind(self.node_type)
            )
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)

    @property
    def joint_propensity(self) -> float:
        """Joint inclusion probability."""
        return self.sampling.effective_joint_propensity(min_propensity=0.0)

    @property
    def weight(self) -> float:
        """Inverse-probability weight with floor for numerical stability."""
        return self.sampling.ipw_weight(min_propensity=MIN_PROPENSITY)

    @classmethod
    def from_logged_observation(
        cls,
        observation: LoggedLabelObservation[Any],
        *,
        violation: int,
        preference_loss: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "TreeSample":
        return cls(
            doc_id=str(observation.document_id),
            node_id=str(observation.unit_id),
            node_type=observation_unit_kind_to_node_type(observation.unit_kind),
            violation=int(violation),
            preference_loss=float(preference_loss),
            sampling=observation.sampling,
            metadata=dict(metadata or observation.context or {}),
        )


@dataclass(frozen=True)
class SampleSplit:
    """Honest train/eval split keyed by document IDs."""

    train_doc_ids: Set[str]
    eval_doc_ids: Set[str]

    @classmethod
    def from_eval_doc_ids(cls, all_doc_ids: Iterable[str], eval_doc_ids: Iterable[str]) -> "SampleSplit":
        eval_ids = set(eval_doc_ids)
        all_ids = set(all_doc_ids)
        return cls(train_doc_ids=all_ids - eval_ids, eval_doc_ids=eval_ids)


@dataclass(frozen=True)
class KFoldSplit:
    """K-fold split represented as disjoint doc-ID folds."""

    folds: Tuple[Set[str], ...]

    @property
    def K(self) -> int:
        return len(self.folds)

    @classmethod
    def from_doc_ids(
        cls,
        doc_ids: Sequence[str],
        k: int,
        *,
        seed: int = 42,
        shuffle: bool = True,
    ) -> "KFoldSplit":
        if k <= 0:
            raise ValueError("k must be >= 1")
        ids = list(dict.fromkeys(doc_ids))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(ids)
        folds: List[Set[str]] = [set() for _ in range(k)]
        for idx, doc_id in enumerate(ids):
            folds[idx % k].add(doc_id)
        return cls(folds=tuple(folds))


@dataclass
class IPWAnalysisSummary:
    """Compact diagnostics summary for TreeIPW sample quality."""

    n_samples: int
    n_docs: int
    violation_rate: float
    preference_loss: float
    union_bound: float
    effective_sample_size: float
    effective_sample_ratio: float
    max_weight: float
    has_adequate_neff: bool
    has_adequate_weight_bound: bool


def _sample_weight(sample: TreeSample, min_propensity: float = MIN_PROPENSITY) -> float:
    return 1.0 / max(sample.joint_propensity, min_propensity)


def _filter_samples(
    samples: Iterable[TreeSample],
    node_type: Optional[NodeType] = None,
) -> List[TreeSample]:
    filtered = list(samples)
    if node_type is not None:
        filtered = [sample for sample in filtered if sample.node_type == node_type]
    return filtered


def filter_by_type(samples: Iterable[TreeSample], node_type: NodeType) -> List[TreeSample]:
    return [s for s in samples if s.node_type == node_type]


def leaf_samples(samples: Iterable[TreeSample]) -> List[TreeSample]:
    return filter_by_type(samples, NodeType.LEAF)


def merge_samples(samples: Iterable[TreeSample]) -> List[TreeSample]:
    return filter_by_type(samples, NodeType.MERGE)


def resummary_samples(samples: Iterable[TreeSample]) -> List[TreeSample]:
    return filter_by_type(samples, NodeType.RESUMMARY)


def substitution_samples(samples: Iterable[TreeSample]) -> List[TreeSample]:
    return filter_by_type(samples, NodeType.SUBSTITUTION)


def train_samples(split: SampleSplit, samples: Iterable[TreeSample]) -> List[TreeSample]:
    return [s for s in samples if s.doc_id in split.train_doc_ids]


def eval_samples(split: SampleSplit, samples: Iterable[TreeSample]) -> List[TreeSample]:
    return [s for s in samples if s.doc_id in split.eval_doc_ids]


def eval_samples_fold(split: KFoldSplit, k: int, samples: Iterable[TreeSample]) -> List[TreeSample]:
    if k < 0 or k >= split.K:
        raise IndexError(f"fold index out of range: {k} for K={split.K}")
    fold_ids = split.folds[k]
    return [s for s in samples if s.doc_id in fold_ids]


def train_samples_fold(split: KFoldSplit, k: int, samples: Iterable[TreeSample]) -> List[TreeSample]:
    if k < 0 or k >= split.K:
        raise IndexError(f"fold index out of range: {k} for K={split.K}")
    holdout = split.folds[k]
    return [s for s in samples if s.doc_id not in holdout]


def hajek_estimate(
    samples: Iterable[TreeSample],
    value_fn: Callable[[TreeSample], float],
    *,
    min_propensity: float = MIN_PROPENSITY,
) -> float:
    """Self-normalized (Hajek) weighted estimator."""
    num = 0.0
    den = 0.0
    for sample in samples:
        w = _sample_weight(sample, min_propensity=min_propensity)
        num += w * float(value_fn(sample))
        den += w
    if den <= 0:
        return 0.0
    return num / den


def horvitz_thompson_total(
    samples: Iterable[TreeSample],
    value_fn: Callable[[TreeSample], float],
    *,
    min_propensity: float = MIN_PROPENSITY,
) -> float:
    """Horvitz-Thompson total estimator: Σ (I_i / π_i) y_i."""
    total = 0.0
    for sample in samples:
        total += _sample_weight(sample, min_propensity=min_propensity) * float(value_fn(sample))
    return total


def horvitz_thompson_mean(
    samples: Iterable[TreeSample],
    value_fn: Callable[[TreeSample], float],
    population_size: float,
    *,
    min_propensity: float = MIN_PROPENSITY,
) -> float:
    """
    Horvitz-Thompson mean estimator: (1 / N) Σ (I_i / π_i) y_i.

    `population_size` should be the target finite-population size N for the
    estimand. If unknown, use Hajek or pass a diagnostic fallback explicitly.
    """
    if not math.isfinite(population_size) or population_size <= 0:
        return 0.0
    return horvitz_thompson_total(
        samples,
        value_fn,
        min_propensity=min_propensity,
    ) / float(population_size)


def hajek_ht_comparison(
    samples: Iterable[TreeSample],
    value_fn: Callable[[TreeSample], float],
    *,
    population_size: Optional[float] = None,
    min_propensity: float = MIN_PROPENSITY,
) -> Dict[str, float]:
    """
    Compare self-normalized Hajek and HT-mean estimators on the same sample set.

    If `population_size` is omitted, this uses `len(samples)` as a fallback
    denominator for a purely diagnostic HT-style comparison.
    """
    filtered = list(samples)
    pop_size = float(population_size) if population_size is not None else float(len(filtered))
    ht_total = horvitz_thompson_total(filtered, value_fn, min_propensity=min_propensity)
    ht_mean = ht_total / pop_size if pop_size > 0 else 0.0
    hajek = hajek_estimate(filtered, value_fn, min_propensity=min_propensity)
    return {
        "sample_count": float(len(filtered)),
        "population_size": pop_size,
        "weight_sum": sum(_sample_weight(s, min_propensity=min_propensity) for s in filtered),
        "ht_total": ht_total,
        "ht_mean": ht_mean,
        "hajek": hajek,
        "abs_diff": abs(hajek - ht_mean),
    }


def _validate_clip_max_weight(max_weight: float) -> float:
    if not math.isfinite(max_weight) or max_weight <= 0:
        raise ValueError(f"clip max_weight must be finite and > 0 (got {max_weight!r})")
    return float(max_weight)


def clipped_weight_sum(
    samples: Iterable[TreeSample],
    max_weight: float,
    *,
    min_propensity: float = MIN_PROPENSITY,
) -> float:
    """Σ min(w_i, w_max)."""
    w_max = _validate_clip_max_weight(max_weight)
    return sum(min(_sample_weight(sample, min_propensity=min_propensity), w_max) for sample in samples)


def total_clipping_excess(
    samples: Iterable[TreeSample],
    max_weight: float,
    *,
    min_propensity: float = MIN_PROPENSITY,
) -> float:
    """Σ (w_i - min(w_i, w_max))."""
    w_max = _validate_clip_max_weight(max_weight)
    return sum(
        max(0.0, _sample_weight(sample, min_propensity=min_propensity) - w_max)
        for sample in samples
    )


def clipped_hajek_estimate(
    samples: Iterable[TreeSample],
    value_fn: Callable[[TreeSample], float],
    max_weight: float,
    *,
    min_propensity: float = MIN_PROPENSITY,
) -> float:
    """Clipped Hajek estimator using min(w_i, w_max)."""
    w_max = _validate_clip_max_weight(max_weight)
    num = 0.0
    den = 0.0
    for sample in samples:
        w = min(_sample_weight(sample, min_propensity=min_propensity), w_max)
        num += w * float(value_fn(sample))
        den += w
    if den <= 0:
        return 0.0
    return num / den


def clipped_hajek_abs_diff_bound(
    samples: Iterable[TreeSample],
    max_weight: float,
    value_bound: float = 1.0,
    *,
    min_propensity: float = MIN_PROPENSITY,
) -> float:
    """
    Deterministic bound:
      |μ_clip - μ_raw| ≤ 2 M * Σ(w - w_clip) / Σ w_clip

    where |y| ≤ M (`value_bound`).
    """
    w_max = _validate_clip_max_weight(max_weight)
    M = abs(float(value_bound))
    clipped_sum = clipped_weight_sum(samples, w_max, min_propensity=min_propensity)
    if clipped_sum <= 0:
        return float("inf")
    excess = total_clipping_excess(samples, w_max, min_propensity=min_propensity)
    return (2.0 * M * excess) / clipped_sum


def clipped_hajek_diagnostics(
    samples: Iterable[TreeSample],
    value_fn: Callable[[TreeSample], float],
    max_weight: float,
    *,
    value_min: float = 0.0,
    value_max: float = 1.0,
    min_propensity: float = MIN_PROPENSITY,
) -> Dict[str, float]:
    """
    Runtime clipping diagnostics aligned with Lean clipping bounds.

    Returns raw/clipped estimators, observed gap, deterministic bound, and
    clipping-mass diagnostics.
    """
    filtered = list(samples)
    w_max = _validate_clip_max_weight(max_weight)
    raw = hajek_estimate(filtered, value_fn, min_propensity=min_propensity)
    clipped = clipped_hajek_estimate(filtered, value_fn, w_max, min_propensity=min_propensity)
    clipped_sum = clipped_weight_sum(filtered, w_max, min_propensity=min_propensity)
    excess = total_clipping_excess(filtered, w_max, min_propensity=min_propensity)
    value_bound = max(abs(float(value_min)), abs(float(value_max)))
    bound = clipped_hajek_abs_diff_bound(
        filtered,
        w_max,
        value_bound=value_bound,
        min_propensity=min_propensity,
    )
    rel_excess = (excess / clipped_sum) if clipped_sum > 0 else float("inf")
    abs_diff = abs(clipped - raw)
    return {
        "sample_count": float(len(filtered)),
        "clip_max_weight": w_max,
        "raw_hajek": raw,
        "clipped_hajek": clipped,
        "abs_diff": abs_diff,
        "abs_diff_bound": bound,
        "bound_holds": 1.0 if abs_diff <= (bound + 1e-12) else 0.0,
        "total_clipping_excess": excess,
        "clipped_weight_sum": clipped_sum,
        "relative_excess": rel_excess,
    }


def ipw_violation_rate_ht(
    samples: Iterable[TreeSample],
    node_type: Optional[NodeType] = None,
    *,
    population_size: Optional[float] = None,
) -> float:
    """HT-mean estimate for violation probability."""
    filtered = _filter_samples(samples, node_type=node_type)
    pop = float(population_size) if population_size is not None else float(len(filtered))
    return horvitz_thompson_mean(filtered, lambda s: float(s.violation), pop)


def ipw_preference_loss_ht(
    samples: Iterable[TreeSample],
    node_type: Optional[NodeType] = None,
    *,
    population_size: Optional[float] = None,
) -> float:
    """HT-mean estimate for bounded preference loss."""
    filtered = _filter_samples(samples, node_type=node_type)
    pop = float(population_size) if population_size is not None else float(len(filtered))
    return horvitz_thompson_mean(filtered, lambda s: float(s.preference_loss), pop)


def ipw_violation_rate(
    samples: Iterable[TreeSample],
    node_type: Optional[NodeType] = None,
) -> float:
    """IPW/Hajek estimate for violation probability."""
    filtered = _filter_samples(samples, node_type=node_type)
    return hajek_estimate(filtered, lambda s: float(s.violation))


def ipw_preference_loss(
    samples: Iterable[TreeSample],
    node_type: Optional[NodeType] = None,
) -> float:
    """IPW/Hajek estimate for bounded preference loss."""
    filtered = _filter_samples(samples, node_type=node_type)
    return hajek_estimate(filtered, lambda s: float(s.preference_loss))


def weighted_variance(
    samples: Iterable[TreeSample],
    value_fn: Callable[[TreeSample], float],
    *,
    min_propensity: float = MIN_PROPENSITY,
) -> float:
    """
    Weighted variance with normalized IPW weights.

    This is the finite-sample ingredient used by empirical-Bernstein style
    confidence radii.
    """
    filtered = list(samples)
    if not filtered:
        return 0.0
    weights = [_sample_weight(s, min_propensity=min_propensity) for s in filtered]
    values = [float(value_fn(s)) for s in filtered]
    sum_w = sum(weights)
    if sum_w <= 0:
        return 0.0
    mean = sum(w * v for w, v in zip(weights, values)) / sum_w
    probs = [w / sum_w for w in weights]
    var = sum(p * (v - mean) ** 2 for p, v in zip(probs, values))
    return max(0.0, var)


def effective_sample_size(
    samples: Iterable[TreeSample],
    *,
    min_propensity: float = MIN_PROPENSITY,
) -> float:
    """Kish effective sample size under IPW weights."""
    weights = [_sample_weight(s, min_propensity=min_propensity) for s in samples]
    if not weights:
        return 0.0
    sum_w = sum(weights)
    sum_w_sq = sum(w * w for w in weights)
    if sum_w_sq <= 0:
        return 0.0
    return (sum_w * sum_w) / sum_w_sq


def max_weight(
    samples: Iterable[TreeSample],
    *,
    min_propensity: float = MIN_PROPENSITY,
) -> float:
    """Maximum IPW weight in the sample set."""
    weights = [_sample_weight(s, min_propensity=min_propensity) for s in samples]
    return max(weights) if weights else 0.0


def empirical_bernstein_radius(
    samples: Iterable[TreeSample],
    value_fn: Callable[[TreeSample], float],
    delta: float,
    *,
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> float:
    """
    Empirical-Bernstein style confidence radius for bounded targets.

    This follows the Lean/FormalProbability convention:
      radius = sqrt(2 * var * log(2/delta) / n_eff)
             + (7/3) * range * log(2/delta) / (n_eff - 1)

    Returns 0.0 when n_eff <= 1 to match the formal definition.
    """
    if delta <= 0 or delta >= 1:
        return float("inf")
    filtered = list(samples)
    if not filtered:
        return float("inf")
    n_eff = effective_sample_size(filtered)
    if n_eff <= 1.0:
        return 0.0
    var = weighted_variance(filtered, value_fn)
    width = max(0.0, value_max - value_min)
    log_term = math.log(2.0 / delta)
    radius = math.sqrt((2.0 * var * log_term) / n_eff) + ((7.0 / 3.0) * width * log_term) / (n_eff - 1.0)
    return max(0.0, radius)


def empirical_bernstein_ci(
    samples: Iterable[TreeSample],
    value_fn: Callable[[TreeSample], float],
    delta: float,
    *,
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> Tuple[float, float]:
    """Two-sided empirical-Bernstein interval for bounded means."""
    filtered = list(samples)
    if not filtered:
        return (value_min, value_max)
    estimate = hajek_estimate(filtered, value_fn)
    radius = empirical_bernstein_radius(
        filtered,
        value_fn,
        delta,
        value_min=value_min,
        value_max=value_max,
    )
    return (max(value_min, estimate - radius), min(value_max, estimate + radius))


def ipw_violation_empirical_bernstein_ci(
    samples: Iterable[TreeSample],
    delta: float = 0.05,
    node_type: Optional[NodeType] = None,
) -> Tuple[float, float]:
    filtered = list(samples)
    if node_type is not None:
        filtered = [s for s in filtered if s.node_type == node_type]
    return empirical_bernstein_ci(filtered, lambda s: float(s.violation), delta, value_min=0.0, value_max=1.0)


def ipw_preference_empirical_bernstein_ci(
    samples: Iterable[TreeSample],
    delta: float = 0.05,
    node_type: Optional[NodeType] = None,
) -> Tuple[float, float]:
    filtered = list(samples)
    if node_type is not None:
        filtered = [s for s in filtered if s.node_type == node_type]
    return empirical_bernstein_ci(filtered, lambda s: float(s.preference_loss), delta, value_min=0.0, value_max=1.0)


def honest_ipw_violation_rate(
    split: SampleSplit,
    samples: Iterable[TreeSample],
    node_type: Optional[NodeType] = None,
) -> float:
    return ipw_violation_rate(eval_samples(split, samples), node_type=node_type)


def honest_ipw_preference_loss(
    split: SampleSplit,
    samples: Iterable[TreeSample],
    node_type: Optional[NodeType] = None,
) -> float:
    return ipw_preference_loss(eval_samples(split, samples), node_type=node_type)


def kfold_ipw_violation_rate(
    split: KFoldSplit,
    samples: Iterable[TreeSample],
    node_type: Optional[NodeType] = None,
) -> float:
    if split.K == 0:
        return 0.0
    sample_list = list(samples)
    fold_rates: List[float] = []
    for idx in range(split.K):
        fold_eval = eval_samples_fold(split, idx, sample_list)
        if fold_eval:
            fold_rates.append(ipw_violation_rate(fold_eval, node_type=node_type))
    if not fold_rates:
        return 0.0
    return sum(fold_rates) / len(fold_rates)


def kfold_ipw_preference_loss(
    split: KFoldSplit,
    samples: Iterable[TreeSample],
    node_type: Optional[NodeType] = None,
) -> float:
    if split.K == 0:
        return 0.0
    sample_list = list(samples)
    fold_losses: List[float] = []
    for idx in range(split.K):
        fold_eval = eval_samples_fold(split, idx, sample_list)
        if fold_eval:
            fold_losses.append(ipw_preference_loss(fold_eval, node_type=node_type))
    if not fold_losses:
        return 0.0
    return sum(fold_losses) / len(fold_losses)


def kfold_ipw_violation_empirical_bernstein_ci(
    split: KFoldSplit,
    samples: Iterable[TreeSample],
    delta: float = 0.05,
    node_type: Optional[NodeType] = None,
) -> Tuple[float, float]:
    return _kfold_ipw_empirical_bernstein_ci(
        split,
        samples,
        delta,
        node_type=node_type,
        value_fn=lambda s: float(s.violation),
        estimate_fn=lambda fold: ipw_violation_rate(fold),
        value_min=0.0,
        value_max=1.0,
    )


def kfold_ipw_preference_empirical_bernstein_ci(
    split: KFoldSplit,
    samples: Iterable[TreeSample],
    delta: float = 0.05,
    node_type: Optional[NodeType] = None,
) -> Tuple[float, float]:
    return _kfold_ipw_empirical_bernstein_ci(
        split,
        samples,
        delta,
        node_type=node_type,
        value_fn=lambda s: float(s.preference_loss),
        estimate_fn=lambda fold: ipw_preference_loss(fold),
        value_min=0.0,
        value_max=1.0,
    )


def _kfold_ipw_empirical_bernstein_ci(
    split: KFoldSplit,
    samples: Iterable[TreeSample],
    delta: float,
    *,
    node_type: Optional[NodeType],
    value_fn: Callable[[TreeSample], float],
    estimate_fn: Callable[[List[TreeSample]], float],
    value_min: float,
    value_max: float,
) -> Tuple[float, float]:
    """
    K-fold empirical-Bernstein CI in the same style as the Lean wrappers:
    - estimate = average fold estimate
    - radius   = average fold empirical-Bernstein radius
    - per-fold deltas split uniformly so total failure budget is `delta`
    """
    if split.K == 0:
        return (value_min, value_max)
    if delta <= 0 or delta >= 1:
        return (value_min, value_max)

    sample_list = list(samples)
    nonempty_folds: List[List[TreeSample]] = []
    for idx in range(split.K):
        fold_eval = eval_samples_fold(split, idx, sample_list)
        if node_type is not None:
            fold_eval = [sample for sample in fold_eval if sample.node_type == node_type]
        if fold_eval:
            nonempty_folds.append(fold_eval)

    if not nonempty_folds:
        return (value_min, value_max)

    mean_est = sum(estimate_fn(fold) for fold in nonempty_folds) / len(nonempty_folds)
    per_fold_delta = delta / float(len(nonempty_folds))
    if per_fold_delta <= 0 or per_fold_delta >= 1:
        return (value_min, value_max)

    mean_radius = (
        sum(
            empirical_bernstein_radius(
                fold,
                value_fn,
                per_fold_delta,
                value_min=value_min,
                value_max=value_max,
            )
            for fold in nonempty_folds
        )
        / len(nonempty_folds)
    )
    return (max(value_min, mean_est - mean_radius), min(value_max, mean_est + mean_radius))


def ipw_union_bound(
    samples: Iterable[TreeSample],
    num_leaves: int,
    num_merges: Optional[int] = None,
    num_rounds: int = 1,
) -> float:
    """
    Tree-level union bound with IPW-estimated rates.

    Bound:
      N * p_suff + M * p_assoc + (R - 1) * p_idem
    """
    sample_list = list(samples)
    if num_merges is None:
        num_merges = max(0, num_leaves - 1)

    p_suff = ipw_violation_rate(sample_list, node_type=NodeType.LEAF)
    p_merge = ipw_violation_rate(sample_list, node_type=NodeType.MERGE)
    p_sub = ipw_violation_rate(sample_list, node_type=NodeType.SUBSTITUTION)
    p_idem = ipw_violation_rate(sample_list, node_type=NodeType.RESUMMARY)

    n_merge = len(merge_samples(sample_list))
    n_sub = len(substitution_samples(sample_list))
    assoc_total = n_merge + n_sub
    if assoc_total > 0:
        lambda_sub = n_sub / assoc_total
        p_assoc = lambda_sub * p_sub + (1.0 - lambda_sub) * p_merge
    else:
        p_assoc = 0.0

    bound = (
        (num_leaves * p_suff)
        + (num_merges * p_assoc)
        + (max(0, num_rounds - 1) * p_idem)
    )
    return min(1.0, max(0.0, bound))


def analyze_tree_samples(
    samples: Iterable[TreeSample],
    num_leaves: int,
    num_merges: Optional[int] = None,
    num_rounds: int = 1,
    *,
    neff_ratio_threshold: float = 0.5,
    max_weight_multiplier: float = 10.0,
) -> IPWAnalysisSummary:
    """Compute IPW summary diagnostics used for audit quality checks."""
    sample_list = list(samples)
    n_samples = len(sample_list)
    n_docs = len({s.doc_id for s in sample_list})

    violation_rate = ipw_violation_rate(sample_list)
    pref_loss = ipw_preference_loss(sample_list)
    union_bound = ipw_union_bound(sample_list, num_leaves=num_leaves, num_merges=num_merges, num_rounds=num_rounds)

    neff = effective_sample_size(sample_list)
    neff_ratio = (neff / n_samples) if n_samples > 0 else 0.0
    max_w = max_weight(sample_list)
    avg_w = sum(s.weight for s in sample_list) / n_samples if n_samples > 0 else 0.0

    return IPWAnalysisSummary(
        n_samples=n_samples,
        n_docs=n_docs,
        violation_rate=violation_rate,
        preference_loss=pref_loss,
        union_bound=union_bound,
        effective_sample_size=neff,
        effective_sample_ratio=neff_ratio,
        max_weight=max_w,
        has_adequate_neff=(neff_ratio >= neff_ratio_threshold),
        has_adequate_weight_bound=(max_w <= (avg_w * max_weight_multiplier) if avg_w > 0 else True),
    )


# ---------------------------------------------------------------------------
# Certificate envelope: three-component gap bound (Lean: treepo_gap_with_
# calibration_estimation_clipping in DSL/TreeIPW.lean).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CertificateEnvelope:
    """Three-envelope certificate mirroring Lean's DSLBound structure.

    |G*| <= |gap_clip| + b_cal + b_est + b_clip

    where:
      b_cal  = judge-vs-oracle calibration mismatch
      b_est  = sampling uncertainty (empirical-Bernstein radius)
      b_clip = clipping-induced bias bound
    """

    gap_clip: float
    b_cal: float
    b_est: float
    b_clip: float
    lean_theorem: str = "treepo_gap_with_calibration_estimation_clipping"

    @property
    def total_margin(self) -> float:
        return abs(self.gap_clip) + self.b_cal + self.b_est + self.b_clip


def compute_certificate(
    samples: Sequence[TreeSample],
    value_fn: Callable[[TreeSample], float],
    *,
    delta: float = 0.05,
    w_max: float = 20.0,
    b_cal: float = 0.0,
    value_min: float = 0.0,
    value_max: float = 1.0,
    min_propensity: float = MIN_PROPENSITY,
) -> CertificateEnvelope:
    """Assemble a three-envelope certificate from existing IPW primitives.

    Args:
        samples: Logged tree-level audit samples.
        value_fn: Extracts the distortion value Y_i from a sample.
        delta: Confidence level for the EB radius (estimation envelope).
        w_max: Clipping threshold for inverse-propensity weights.
        b_cal: Calibration bound (user-supplied; 0 when using exact oracle).
        value_min / value_max: Bounds on the distortion values |Y_i|.
        min_propensity: Floor on logged propensities.

    Returns:
        CertificateEnvelope with all three envelopes and total margin.
    """
    filtered = list(samples)

    # G^C: clipped Hajek point estimate of distortion.
    gap_clip = clipped_hajek_estimate(
        filtered, value_fn, w_max, min_propensity=min_propensity,
    )

    # B_est: empirical-Bernstein radius for sampling uncertainty.
    b_est = empirical_bernstein_radius(
        filtered, value_fn, delta,
        value_min=value_min, value_max=value_max,
    )

    # B_clip: deterministic clipping bias bound.
    value_bound = max(abs(float(value_min)), abs(float(value_max)))
    b_clip = clipped_hajek_abs_diff_bound(
        filtered, w_max, value_bound=value_bound,
        min_propensity=min_propensity,
    )

    return CertificateEnvelope(
        gap_clip=gap_clip,
        b_cal=float(b_cal),
        b_est=b_est,
        b_clip=b_clip,
    )


def compute_calibration_bound(
    gold_judge_pairs: Sequence[Tuple[float, float]],
) -> float:
    """Compute B_cal from held-out gold-vs-judge distortion pairs.

    Returns the mean absolute difference |gold_i - judge_i| as an estimate
    of the calibration mismatch.  For an exact oracle, this is 0.
    """
    if not gold_judge_pairs:
        return 0.0
    total = sum(abs(g - j) for g, j in gold_judge_pairs)
    return total / len(gold_judge_pairs)


__all__ = [
    "NodeType",
    "TreeSample",
    "SampleSplit",
    "KFoldSplit",
    "IPWAnalysisSummary",
    "filter_by_type",
    "leaf_samples",
    "merge_samples",
    "resummary_samples",
    "substitution_samples",
    "train_samples",
    "eval_samples",
    "eval_samples_fold",
    "train_samples_fold",
    "hajek_estimate",
    "horvitz_thompson_total",
    "horvitz_thompson_mean",
    "hajek_ht_comparison",
    "clipped_weight_sum",
    "total_clipping_excess",
    "clipped_hajek_estimate",
    "clipped_hajek_abs_diff_bound",
    "clipped_hajek_diagnostics",
    "ipw_violation_rate_ht",
    "ipw_preference_loss_ht",
    "ipw_violation_rate",
    "ipw_preference_loss",
    "weighted_variance",
    "effective_sample_size",
    "max_weight",
    "empirical_bernstein_radius",
    "empirical_bernstein_ci",
    "ipw_violation_empirical_bernstein_ci",
    "ipw_preference_empirical_bernstein_ci",
    "honest_ipw_violation_rate",
    "honest_ipw_preference_loss",
    "kfold_ipw_violation_rate",
    "kfold_ipw_preference_loss",
    "kfold_ipw_violation_empirical_bernstein_ci",
    "kfold_ipw_preference_empirical_bernstein_ci",
    "ipw_union_bound",
    "analyze_tree_samples",
    "CertificateEnvelope",
    "compute_certificate",
    "compute_calibration_bound",
]
