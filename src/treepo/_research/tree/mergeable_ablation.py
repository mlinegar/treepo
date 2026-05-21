"""
Simple ablations for repeated tree aggregation under chunking errors.

This module is intentionally minimal and numeric. It is designed to show:

1) naive repeated aggregation failures (e.g. majority vote, mean-of-means), and
2) "right merge rule, wrong adaptive chunker" failures under chunk-budgeting.

The target objective is non-additive by default:
  "spike-exists" = 1 if any token score crosses a threshold, else 0.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import random
import statistics
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from treepo._research.tree.weighting import (
    DEFAULT_WEIGHTING_MODES,
    WeightingMode,
    build_weighting_views_from_replicates,
    parse_weighting_modes,
    validate_legacy_weighting_mode,
)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


class ObjectiveProfile(str, Enum):
    """Document-level ground-truth objective."""

    SPIKE_EXISTS = "spike-exists"
    TOP2_MEAN = "top2-mean"


class ChunkerPolicy(str, Enum):
    """Chunk-boundary policy used before tree aggregation."""

    FIXED = "fixed"
    ADAPTIVE_ALIGNED = "adaptive-aligned"
    ADAPTIVE_MISSPECIFIED = "adaptive-misspecified"


class SelectorPolicy(str, Enum):
    """Chunk retention policy under finite chunk budget."""

    ALL = "all"
    TOP_PROXY = "top-proxy"
    TOP_TRUE = "top-true"
    BOTTOM_PROXY = "bottom-proxy"
    RANDOM = "random"


class AggregatorPolicy(str, Enum):
    """Repeated tree aggregation operator."""

    MERGE_SAFE_MAX = "merge-safe-max"
    MERGE_SAFE_SECOND_MAX = "merge-safe-second-max"
    MERGE_SAFE_THIRD_MAX = "merge-safe-third-max"
    MERGE_SAFE_BOUNDARY_MAX = "merge-safe-boundary-max"
    MERGE_SAFE_WEIGHTED_MEAN = "merge-safe-weighted-mean"
    NAIVE_MAJORITY = "naive-majority"
    NAIVE_MEAN_OF_MEANS = "naive-mean-of-means"


class MergeOrder(str, Enum):
    """Pairwise reduction order for repeated aggregation."""

    LEFT_TO_RIGHT = "left-to-right"
    RIGHT_TO_LEFT = "right-to-left"
    RANDOM = "random"


class TokenPattern(str, Enum):
    """Toy document token score patterns."""

    BOUNDARY_SPIKE = "boundary-spike"
    INTERIOR_SPIKE = "interior-spike"
    TWO_SPIKES = "two-spikes"
    MULTI_SPIKES = "multi-spikes"
    DIFFUSE = "diffuse"


VALID_WEIGHTING_MODES: Tuple[str, ...] = tuple(m.value for m in WeightingMode)


def _resolve_weighting(
    *,
    weighting_modes: Optional[Sequence[str]],
    legacy_weighting_mode: str,
) -> Tuple[Tuple[WeightingMode, ...], WeightingMode]:
    modes = parse_weighting_modes(weighting_modes if weighting_modes is not None else DEFAULT_WEIGHTING_MODES)
    legacy = validate_legacy_weighting_mode(legacy_weighting_mode, weighting_modes=modes)
    return modes, legacy


@dataclass(frozen=True)
class ToyTokenDocument:
    """Per-token ground-truth and proxy values."""

    token_scores: Tuple[float, ...]
    proxy_scores: Tuple[float, ...]

    @property
    def n_tokens(self) -> int:
        return len(self.token_scores)


@dataclass(frozen=True)
class Chunk:
    """A chunk with precomputed summary stats."""

    start: int
    end: int
    values: Tuple[float, ...]
    proxy_values: Tuple[float, ...]

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / float(len(self.values))

    @property
    def vmax(self) -> float:
        if not self.values:
            return 0.0
        return max(self.values)

    @property
    def proxy_max(self) -> float:
        if not self.proxy_values:
            return 0.0
        return max(self.proxy_values)


@dataclass(frozen=True)
class AblationSpec:
    """One ablation method configuration."""

    name: str
    description: str
    chunker: ChunkerPolicy
    selector: SelectorPolicy
    aggregator: AggregatorPolicy
    chunk_budget: Optional[int] = None
    merge_order: MergeOrder = MergeOrder.LEFT_TO_RIGHT
    two_spike_aggregator: Optional[AggregatorPolicy] = None
    three_spike_aggregator: Optional[AggregatorPolicy] = None
    boundary_spike_aggregator: Optional[AggregatorPolicy] = None
    fixed_chunk_size: int = 8
    min_chunk_size: int = 1
    max_chunk_size: int = 10
    boundary_span_tokens: int = 4


@dataclass(frozen=True)
class DocumentEvaluation:
    """Single-document evaluation result for one ablation spec."""

    spec_name: str
    true_score: float
    estimated_score: float
    true_label: int
    estimated_label: int
    abs_error: float
    n_chunks_total: int
    n_chunks_kept: int


@dataclass(frozen=True)
class AblationSummary:
    """Aggregate result across many documents for one ablation spec."""

    name: str
    description: str
    n_docs: int
    mean_true_score: float
    mean_estimated_score: float
    mean_abs_error: float
    label_error_rate: float
    mean_chunks_total: float
    mean_chunks_kept: float
    order_spread_mean: float
    order_flip_rate: float


@dataclass(frozen=True)
class SpikeMixtureDistributionSpec:
    """Known DGP for parameter-recovery experiments."""

    p_spike_doc: float = 0.50
    p_boundary_given_spike: float = 0.50
    p_two_spikes_given_spike: float = 0.25
    p_multi_given_two_spikes: float = 0.0
    n_tokens: int = 32
    proxy_noise: float = 0.08
    boundary_span_tokens: int = 4
    token_length_support: Optional[Tuple[int, ...]] = None
    token_length_probs: Optional[Tuple[float, ...]] = None


@dataclass(frozen=True)
class ParameterRecoverySummary:
    """Method-level parameter-recovery summary over repeated simulations."""

    method_name: str
    description: str
    true_param: float
    n_replicates: int
    docs_per_replicate: int
    mean_estimate: float
    mean_bias: float
    mean_abs_bias: float
    sample_target_bias: float
    std_estimate: float
    rmse: float
    legacy_weighting_mode: str = "doc"
    weighting_views: Optional[Dict[str, Dict[str, float]]] = None


@dataclass(frozen=True)
class TwoParameterRecoverySummary:
    """Joint recovery summary for (p_spike, p_two_spikes_given_spike)."""

    method_name: str
    description: str
    supports_two_spike: bool
    true_p_spike: float
    true_p_two_given_spike: float
    true_p_two_doc: float
    n_replicates: int
    docs_per_replicate: int
    mean_hat_p_spike: float
    bias_p_spike: float
    mean_abs_bias_p_spike: float
    rmse_p_spike: float
    mean_hat_p_two_given_spike: float
    bias_p_two_given_spike: float
    mean_abs_bias_p_two_given_spike: float
    rmse_p_two_given_spike: float
    mean_hat_p_two_doc: float
    bias_p_two_doc: float
    mean_abs_bias_p_two_doc: float
    rmse_p_two_doc: float
    sample_target_bias_p_spike: float
    sample_target_bias_p_two_given_spike: float
    sample_target_bias_p_two_doc: float
    legacy_weighting_mode: str = "doc"
    weighting_views: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None


@dataclass(frozen=True)
class ThreeParameterRecoverySummary:
    """Joint recovery summary for (p_spike, p_two|spike, p_boundary|spike)."""

    method_name: str
    description: str
    supports_two_spike: bool
    supports_boundary_spike: bool
    true_p_spike: float
    true_p_two_given_spike: float
    true_p_boundary_given_spike: float
    true_p_two_doc: float
    true_p_boundary_doc: float
    n_replicates: int
    docs_per_replicate: int
    mean_hat_p_spike: float
    bias_p_spike: float
    mean_abs_bias_p_spike: float
    rmse_p_spike: float
    mean_hat_p_two_given_spike: float
    bias_p_two_given_spike: float
    mean_abs_bias_p_two_given_spike: float
    rmse_p_two_given_spike: float
    mean_hat_p_boundary_given_spike: float
    bias_p_boundary_given_spike: float
    mean_abs_bias_p_boundary_given_spike: float
    rmse_p_boundary_given_spike: float
    mean_hat_p_two_doc: float
    bias_p_two_doc: float
    mean_abs_bias_p_two_doc: float
    rmse_p_two_doc: float
    mean_hat_p_boundary_doc: float
    bias_p_boundary_doc: float
    mean_abs_bias_p_boundary_doc: float
    rmse_p_boundary_doc: float
    sample_target_bias_p_spike: float
    sample_target_bias_p_two_given_spike: float
    sample_target_bias_p_boundary_given_spike: float
    sample_target_bias_p_two_doc: float
    sample_target_bias_p_boundary_doc: float
    legacy_weighting_mode: str = "doc"
    weighting_views: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None


@dataclass(frozen=True)
class FourParameterRecoverySummary:
    """Joint recovery summary for (p_spike, p_two|spike, p_three_plus|spike, p_boundary|spike)."""

    method_name: str
    description: str
    supports_two_spike: bool
    supports_three_spike: bool
    supports_boundary_spike: bool
    true_p_spike: float
    true_p_two_given_spike: float
    true_p_three_given_spike: float
    true_p_boundary_given_spike: float
    true_p_two_doc: float
    true_p_three_doc: float
    true_p_boundary_doc: float
    n_replicates: int
    docs_per_replicate: int
    mean_hat_p_spike: float
    bias_p_spike: float
    mean_abs_bias_p_spike: float
    rmse_p_spike: float
    mean_hat_p_two_given_spike: float
    bias_p_two_given_spike: float
    mean_abs_bias_p_two_given_spike: float
    rmse_p_two_given_spike: float
    mean_hat_p_three_given_spike: float
    bias_p_three_given_spike: float
    mean_abs_bias_p_three_given_spike: float
    rmse_p_three_given_spike: float
    mean_hat_p_boundary_given_spike: float
    bias_p_boundary_given_spike: float
    mean_abs_bias_p_boundary_given_spike: float
    rmse_p_boundary_given_spike: float
    mean_hat_p_two_doc: float
    bias_p_two_doc: float
    mean_abs_bias_p_two_doc: float
    rmse_p_two_doc: float
    mean_hat_p_three_doc: float
    bias_p_three_doc: float
    mean_abs_bias_p_three_doc: float
    rmse_p_three_doc: float
    mean_hat_p_boundary_doc: float
    bias_p_boundary_doc: float
    mean_abs_bias_p_boundary_doc: float
    rmse_p_boundary_doc: float
    sample_target_bias_p_spike: float
    sample_target_bias_p_two_given_spike: float
    sample_target_bias_p_three_given_spike: float
    sample_target_bias_p_boundary_given_spike: float
    sample_target_bias_p_two_doc: float
    sample_target_bias_p_three_doc: float
    sample_target_bias_p_boundary_doc: float
    legacy_weighting_mode: str = "doc"
    weighting_views: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None


@dataclass(frozen=True)
class GeneralizationStressScenario:
    """One stress scenario for robustness and generalization checks."""

    name: str
    description: str
    distribution: SpikeMixtureDistributionSpec


@dataclass(frozen=True)
class GeneralizationStressSummary:
    """Method-level summary within one stress scenario."""

    scenario_name: str
    scenario_description: str
    method_name: str
    method_description: str
    supports_two_spike: bool
    supports_boundary_spike: bool
    true_p_spike: float
    true_p_two_given_spike: float
    true_p_boundary_given_spike: float
    mean_hat_p_spike: float
    mean_hat_p_two_given_spike: float
    mean_hat_p_boundary_given_spike: float
    mean_abs_bias_p_spike: float
    mean_abs_bias_p_two_given_spike: float
    mean_abs_bias_p_boundary_given_spike: float
    aggregate_mean_abs_bias: float
    generalization_gap_vs_baseline: float


@dataclass(frozen=True)
class NonLanguageScenario:
    """Domain-agnostic scenario with an intuition label and spike-count DGP."""

    name: str
    intuition: str
    distribution: SpikeCountMixtureDistributionSpec


class KSketchEstimator(str, Enum):
    """Estimator family for generic-k recovery."""

    MERGE_SAFE_TOPK = "merge-safe-topk"
    NAIVE_MAJORITY = "naive-majority"
    NAIVE_MEAN_OF_MEANS = "naive-mean-of-means"


@dataclass(frozen=True)
class SpikeCountMixtureDistributionSpec:
    """Known DGP over exact spike counts for generic-k recovery."""

    p_spike_doc: float = 0.62
    spike_count_support: Tuple[int, ...] = (1, 2, 3, 4, 5)
    spike_count_probs_given_spike: Tuple[float, ...] = (0.28, 0.27, 0.20, 0.15, 0.10)
    p_boundary_given_spike: float = 0.35
    n_tokens: int = 32
    proxy_noise: float = 0.12
    boundary_span_tokens: int = 4
    token_length_support: Optional[Tuple[int, ...]] = None
    token_length_probs: Optional[Tuple[float, ...]] = None


@dataclass(frozen=True)
class KSketchMethodSpec:
    """Method configuration for generic-k recovery."""

    name: str
    description: str
    estimator: KSketchEstimator
    chunker: ChunkerPolicy
    selector: SelectorPolicy
    sketch_order: int
    chunk_budget: Optional[int] = None
    merge_order: MergeOrder = MergeOrder.LEFT_TO_RIGHT
    fixed_chunk_size: int = 8
    min_chunk_size: int = 1
    max_chunk_size: int = 10


@dataclass(frozen=True)
class KTargetRecoverySummary:
    """Method-level summary for one target k in generic-k recovery."""

    method_name: str
    method_description: str
    estimator: KSketchEstimator
    sketch_order: int
    target_k: int
    supports_target: bool
    true_p_at_least_k_given_spike: float
    n_replicates: int
    docs_per_replicate: int
    mean_hat_p_at_least_k_given_spike: float
    bias: float
    mean_abs_bias: float
    rmse: float
    sample_target_bias: float
    legacy_weighting_mode: str = "doc"
    weighting_views: Optional[Dict[str, Dict[str, float]]] = None


@dataclass(frozen=True)
class ChunkQualitySweepSummary:
    """Method-level summary for chunk-quality and leaf-granularity sweeps."""

    method_name: str
    method_description: str
    chunker: ChunkerPolicy
    selector: SelectorPolicy
    sketch_order: int
    target_k: int
    chunk_budget: Optional[int]
    fixed_chunk_size: int
    min_chunk_size: int
    max_chunk_size: int
    supports_target: bool
    true_p_at_least_k_given_spike: float
    n_replicates: int
    docs_per_replicate: int
    mean_hat_p_at_least_k_given_spike: float
    bias: float
    mean_abs_bias: float
    rmse: float
    sample_target_bias: float
    mean_target_capture_rate: float
    mean_spike_capture_rate: float
    mean_spike_token_recall: float
    mean_spike_token_isolation: float
    mean_chunks_total: float
    mean_chunks_kept: float
    legacy_weighting_mode: str = "doc"
    weighting_views: Optional[Dict[str, Dict[str, float]]] = None


@dataclass(frozen=True)
class ChunkQualityCoverageSummary:
    """Coverage summary for chunk-quality sweeps with per-replicate CIs."""

    method_name: str
    method_description: str
    chunker: ChunkerPolicy
    selector: SelectorPolicy
    sketch_order: int
    target_k: int
    chunk_budget: Optional[int]
    fixed_chunk_size: int
    min_chunk_size: int
    max_chunk_size: int
    supports_target: bool
    true_p_at_least_k_given_spike: float
    n_replicates: int
    docs_per_replicate: int
    ci_level: float
    mean_hat_p_at_least_k_given_spike: float
    bias: float
    mean_abs_bias: float
    rmse: float
    empirical_coverage: float
    mean_ci_width: float
    mean_ci_low: float
    mean_ci_high: float
    legacy_weighting_mode: str = "doc"
    weighting_views: Optional[Dict[str, Dict[str, float]]] = None


def true_objective_score(
    token_scores: Sequence[float],
    *,
    objective: ObjectiveProfile = ObjectiveProfile.SPIKE_EXISTS,
    spike_threshold: float = 0.90,
) -> float:
    """Compute the exact document-level objective from token scores."""
    if not token_scores:
        return 0.0
    values = [float(v) for v in token_scores]
    if objective == ObjectiveProfile.SPIKE_EXISTS:
        return 1.0 if max(values) >= spike_threshold else 0.0
    if objective == ObjectiveProfile.TOP2_MEAN:
        top = sorted(values, reverse=True)[:2]
        return sum(top) / float(len(top))
    raise ValueError(f"Unsupported objective: {objective!r}")


def true_two_spike_event(
    token_scores: Sequence[float],
    *,
    spike_threshold: float = 0.90,
) -> int:
    """Indicator that at least two tokens cross spike threshold."""
    if not token_scores:
        return 0
    count = sum(1 for v in token_scores if float(v) >= spike_threshold)
    return 1 if count >= 2 else 0


def true_three_plus_spike_event(
    token_scores: Sequence[float],
    *,
    spike_threshold: float = 0.90,
) -> int:
    """Indicator that at least three tokens cross spike threshold."""
    if not token_scores:
        return 0
    count = sum(1 for v in token_scores if float(v) >= spike_threshold)
    return 1 if count >= 3 else 0


def true_boundary_spike_event(
    token_scores: Sequence[float],
    *,
    spike_threshold: float = 0.90,
    boundary_span_tokens: int = 4,
) -> int:
    """Indicator that any spike appears within the boundary token windows."""
    if not token_scores:
        return 0
    n = len(token_scores)
    span = max(1, min(int(boundary_span_tokens), n))
    for i, value in enumerate(token_scores):
        if float(value) >= spike_threshold and (i < span or i >= n - span):
            return 1
    return 0


def true_spike_count(
    token_scores: Sequence[float],
    *,
    spike_threshold: float = 0.90,
) -> int:
    """Count tokens crossing the spike threshold."""
    return sum(1 for v in token_scores if float(v) >= spike_threshold)


def generate_exact_spike_count_document(
    *,
    n_spikes: int,
    n_tokens: int = 32,
    proxy_noise: float = 0.08,
    boundary_span_tokens: int = 4,
    force_boundary_spike: bool = False,
    seed: int = 0,
) -> ToyTokenDocument:
    """Generate a document with (approximately) exact number of spikes."""
    if n_tokens <= 0:
        raise ValueError("n_tokens must be >= 1")
    rng = random.Random(seed)
    scores = [0.12 + rng.uniform(-0.05, 0.05) for _ in range(n_tokens)]

    count = max(0, min(int(n_spikes), n_tokens))
    if count > 0:
        span = max(1, min(int(boundary_span_tokens), n_tokens))
        boundary_idxs = list(range(span)) + list(range(max(0, n_tokens - span), n_tokens))
        boundary_set = set(boundary_idxs)
        interior_idxs = [i for i in range(n_tokens) if i not in boundary_set]

        chosen: List[int] = []
        if force_boundary_spike and boundary_idxs:
            chosen.append(rng.choice(boundary_idxs))

        remaining = count - len(chosen)
        pool = [i for i in interior_idxs if i not in chosen]
        if len(pool) < remaining:
            pool = [i for i in range(n_tokens) if i not in chosen]
        if remaining > 0:
            chosen.extend(rng.sample(pool, k=min(remaining, len(pool))))

        high_vals = [0.99, 0.97, 0.95, 0.93, 0.92, 0.91, 0.905, 0.901]
        for j, idx in enumerate(chosen):
            scores[idx] = high_vals[min(j, len(high_vals) - 1)]

    proxy = []
    for v in scores:
        proxy.append(_clip01(v + rng.uniform(-proxy_noise, proxy_noise)))
    return ToyTokenDocument(token_scores=tuple(scores), proxy_scores=tuple(proxy))


def generate_toy_token_document(
    *,
    pattern: TokenPattern = TokenPattern.BOUNDARY_SPIKE,
    n_tokens: int = 32,
    proxy_noise: float = 0.08,
    boundary_span_tokens: Optional[int] = None,
    seed: int = 0,
) -> ToyTokenDocument:
    """Generate a simple token-level toy document with known structure."""
    if n_tokens <= 0:
        raise ValueError("n_tokens must be >= 1")

    rng = random.Random(seed)
    scores = [0.12 + rng.uniform(-0.05, 0.05) for _ in range(n_tokens)]

    if pattern == TokenPattern.BOUNDARY_SPIKE:
        span = boundary_span_tokens
        if span is None:
            span = max(1, n_tokens // 8)
        span = max(1, min(int(span), n_tokens))
        if rng.random() < 0.5:
            idx = rng.randrange(0, span)
        else:
            idx = rng.randrange(max(0, n_tokens - span), n_tokens)
        scores[idx] = 0.99
    elif pattern == TokenPattern.INTERIOR_SPIKE:
        idx = max(1, min(n_tokens - 2, n_tokens // 2))
        scores[idx] = 0.99
    elif pattern == TokenPattern.TWO_SPIKES:
        idx1 = max(1, min(n_tokens - 2, n_tokens // 3))
        idx2 = max(idx1 + 1, min(n_tokens - 1, (2 * n_tokens) // 3))
        scores[idx1] = 0.95
        scores[idx2] = 0.93
    elif pattern == TokenPattern.MULTI_SPIKES:
        # Explicit "multiple spikes" case with >=3 spikes when feasible.
        if n_tokens < 3:
            idx1 = 0
            idx2 = min(1, n_tokens - 1)
            scores[idx1] = 0.95
            scores[idx2] = 0.93
        else:
            span = boundary_span_tokens
            if span is None:
                span = max(1, n_tokens // 8)
            span = max(1, min(int(span), n_tokens // 2))
            interior_pool = list(range(span, max(span, n_tokens - span)))
            if len(interior_pool) < 3:
                interior_pool = list(range(n_tokens))
            n_spikes = min(len(interior_pool), rng.randint(3, 6))
            indices = sorted(rng.sample(interior_pool, k=n_spikes))
            high_vals = [0.99, 0.97, 0.95, 0.93, 0.92, 0.91]
            for j, idx in enumerate(indices):
                scores[idx] = high_vals[min(j, len(high_vals) - 1)]
    elif pattern == TokenPattern.DIFFUSE:
        for i in range(n_tokens):
            scores[i] = 0.45 + rng.uniform(-0.15, 0.15)
    else:
        raise ValueError(f"Unsupported token pattern: {pattern!r}")

    proxy = []
    for v in scores:
        noisy = _clip01(v + rng.uniform(-proxy_noise, proxy_noise))
        proxy.append(noisy)
    return ToyTokenDocument(token_scores=tuple(scores), proxy_scores=tuple(proxy))


def _chunk_boundaries_fixed(n_tokens: int, chunk_size: int) -> List[Tuple[int, int]]:
    boundaries: List[Tuple[int, int]] = []
    i = 0
    size = max(1, int(chunk_size))
    while i < n_tokens:
        j = min(n_tokens, i + size)
        boundaries.append((i, j))
        i = j
    return boundaries


def _chunk_boundaries_adaptive(
    proxy_scores: Sequence[float],
    *,
    aligned: bool,
    min_chunk_size: int,
    max_chunk_size: int,
) -> List[Tuple[int, int]]:
    n = len(proxy_scores)
    boundaries: List[Tuple[int, int]] = []
    i = 0
    lo = max(1, int(min_chunk_size))
    hi = max(lo, int(max_chunk_size))
    span = float(max(1, hi - lo))
    while i < n:
        p = _clip01(float(proxy_scores[i]))
        effective = p if aligned else (1.0 - p)
        size = int(round(hi - effective * span))
        size = max(lo, min(hi, size))
        j = min(n, i + size)
        boundaries.append((i, j))
        i = j
    return boundaries


def chunk_document(
    document: ToyTokenDocument,
    *,
    policy: ChunkerPolicy,
    fixed_chunk_size: int = 8,
    min_chunk_size: int = 1,
    max_chunk_size: int = 10,
) -> List[Chunk]:
    """Create chunks from token/proxy sequences under a chosen chunker policy."""
    n = document.n_tokens
    if policy == ChunkerPolicy.FIXED:
        boundaries = _chunk_boundaries_fixed(n, fixed_chunk_size)
    elif policy == ChunkerPolicy.ADAPTIVE_ALIGNED:
        boundaries = _chunk_boundaries_adaptive(
            document.proxy_scores,
            aligned=True,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )
    elif policy == ChunkerPolicy.ADAPTIVE_MISSPECIFIED:
        boundaries = _chunk_boundaries_adaptive(
            document.proxy_scores,
            aligned=False,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )
    else:
        raise ValueError(f"Unsupported chunker policy: {policy!r}")

    return [
        Chunk(
            start=a,
            end=b,
            values=tuple(document.token_scores[a:b]),
            proxy_values=tuple(document.proxy_scores[a:b]),
        )
        for (a, b) in boundaries
        if b > a
    ]


def select_chunks(
    chunks: Sequence[Chunk],
    *,
    selector: SelectorPolicy,
    chunk_budget: Optional[int],
    rng: Optional[random.Random] = None,
) -> List[Chunk]:
    """Apply chunk-budget selection policy (kept chunks preserve original order)."""
    if chunk_budget is None or chunk_budget >= len(chunks):
        return list(chunks)
    if chunk_budget <= 0:
        return []

    indexed = list(enumerate(chunks))
    if selector == SelectorPolicy.ALL:
        selected = indexed[:chunk_budget]
    elif selector == SelectorPolicy.TOP_PROXY:
        selected = sorted(indexed, key=lambda t: t[1].proxy_max, reverse=True)[:chunk_budget]
    elif selector == SelectorPolicy.TOP_TRUE:
        selected = sorted(indexed, key=lambda t: t[1].vmax, reverse=True)[:chunk_budget]
    elif selector == SelectorPolicy.BOTTOM_PROXY:
        selected = sorted(indexed, key=lambda t: t[1].proxy_max)[:chunk_budget]
    elif selector == SelectorPolicy.RANDOM:
        rr = rng if rng is not None else random.Random(0)
        selected = rr.sample(indexed, k=chunk_budget)
    else:
        raise ValueError(f"Unsupported selector: {selector!r}")

    return [chunk for _, chunk in sorted(selected, key=lambda t: t[0])]


def _doc_weight(
    *,
    doc: ToyTokenDocument,
    all_chunks: Sequence[Chunk],
    mode: WeightingMode,
) -> float:
    if mode == WeightingMode.DOC:
        return 1.0
    if mode == WeightingMode.LEAF:
        return float(max(1, len(all_chunks)))
    if mode == WeightingMode.TOKEN:
        return float(max(1, doc.n_tokens))
    raise ValueError(f"unsupported weighting mode: {mode!r}")


def _pair_groups(n: int, order: MergeOrder, rng: random.Random) -> List[List[int]]:
    idxs = list(range(n))
    if order == MergeOrder.RIGHT_TO_LEFT:
        idxs = list(reversed(idxs))
    elif order == MergeOrder.RANDOM:
        rng.shuffle(idxs)
    groups: List[List[int]] = []
    i = 0
    while i < len(idxs):
        if i + 1 < len(idxs):
            groups.append([idxs[i], idxs[i + 1]])
            i += 2
        else:
            groups.append([idxs[i]])
            i += 1
    return groups


def _reduce_tree_states(
    states: Sequence[object],
    *,
    merge_order: MergeOrder,
    merge_fn,
    seed: int = 0,
) -> object:
    rng = random.Random(seed)
    current = list(states)
    while len(current) > 1:
        groups = _pair_groups(len(current), merge_order, rng)
        next_states: List[object] = []
        for group in groups:
            vals = [current[i] for i in group]
            next_states.append(merge_fn(vals))
        current = next_states
    return current[0]


def aggregate_chunks(
    chunks: Sequence[Chunk],
    *,
    aggregator: AggregatorPolicy,
    merge_order: MergeOrder = MergeOrder.LEFT_TO_RIGHT,
    vote_threshold: float = 0.50,
    spike_threshold: float = 0.90,
    boundary_span_tokens: int = 4,
    seed: int = 0,
) -> float:
    """Aggregate chunk summaries by repeated pairwise merging."""
    if not chunks:
        return 0.0

    if aggregator == AggregatorPolicy.MERGE_SAFE_MAX:
        leaves = [chunk.vmax for chunk in chunks]
        root = _reduce_tree_states(
            leaves,
            merge_order=merge_order,
            merge_fn=lambda xs: max(float(v) for v in xs),
            seed=seed,
        )
        return 1.0 if float(root) >= spike_threshold else 0.0

    if aggregator == AggregatorPolicy.MERGE_SAFE_SECOND_MAX:
        leaves = []
        for chunk in chunks:
            vals = sorted((float(v) for v in chunk.values), reverse=True)
            top1 = vals[0] if vals else -float("inf")
            top2 = vals[1] if len(vals) > 1 else -float("inf")
            leaves.append((top1, top2))

        def _merge_top2(xs: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
            candidates: List[float] = []
            for a, b in xs:
                candidates.append(float(a))
                candidates.append(float(b))
            top = sorted(candidates, reverse=True)[:2]
            if len(top) == 1:
                top.append(-float("inf"))
            return (top[0], top[1])

        _, second = _reduce_tree_states(
            leaves,
            merge_order=merge_order,
            merge_fn=_merge_top2,
            seed=seed,
        )
        return 1.0 if float(second) >= spike_threshold else 0.0

    if aggregator == AggregatorPolicy.MERGE_SAFE_THIRD_MAX:
        leaves = []
        for chunk in chunks:
            vals = sorted((float(v) for v in chunk.values), reverse=True)
            top1 = vals[0] if vals else -float("inf")
            top2 = vals[1] if len(vals) > 1 else -float("inf")
            top3 = vals[2] if len(vals) > 2 else -float("inf")
            leaves.append((top1, top2, top3))

        def _merge_top3(xs: Sequence[Tuple[float, float, float]]) -> Tuple[float, float, float]:
            candidates: List[float] = []
            for a, b, c in xs:
                candidates.append(float(a))
                candidates.append(float(b))
                candidates.append(float(c))
            top = sorted(candidates, reverse=True)[:3]
            while len(top) < 3:
                top.append(-float("inf"))
            return (top[0], top[1], top[2])

        _, _, third = _reduce_tree_states(
            leaves,
            merge_order=merge_order,
            merge_fn=_merge_top3,
            seed=seed,
        )
        return 1.0 if float(third) >= spike_threshold else 0.0

    if aggregator == AggregatorPolicy.MERGE_SAFE_BOUNDARY_MAX:
        n_tokens = max(chunk.end for chunk in chunks)
        span = max(1, min(int(boundary_span_tokens), n_tokens))
        leaves: List[float] = []
        for chunk in chunks:
            local = -float("inf")
            for offset, value in enumerate(chunk.values):
                idx = chunk.start + offset
                if idx < span or idx >= n_tokens - span:
                    local = max(local, float(value))
            leaves.append(local)
        root = _reduce_tree_states(
            leaves,
            merge_order=merge_order,
            merge_fn=lambda xs: max(float(v) for v in xs),
            seed=seed,
        )
        return 1.0 if float(root) >= spike_threshold else 0.0

    if aggregator == AggregatorPolicy.MERGE_SAFE_WEIGHTED_MEAN:
        leaves = [(sum(chunk.values), float(chunk.count)) for chunk in chunks]

        def _merge_weighted(xs: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
            return (sum(x[0] for x in xs), sum(x[1] for x in xs))

        total_sum, total_count = _reduce_tree_states(
            leaves,
            merge_order=merge_order,
            merge_fn=_merge_weighted,
            seed=seed,
        )
        if total_count <= 0:
            return 0.0
        return float(total_sum) / float(total_count)

    if aggregator == AggregatorPolicy.NAIVE_MAJORITY:
        leaves = [1.0 if chunk.mean >= vote_threshold else 0.0 for chunk in chunks]

        def _merge_majority(xs: Sequence[float]) -> float:
            yes = sum(1 for x in xs if float(x) >= 0.5)
            return 1.0 if yes * 2 >= len(xs) else 0.0

        return float(
            _reduce_tree_states(
                leaves,
                merge_order=merge_order,
                merge_fn=_merge_majority,
                seed=seed,
            )
        )

    if aggregator == AggregatorPolicy.NAIVE_MEAN_OF_MEANS:
        leaves = [chunk.mean for chunk in chunks]
        root = _reduce_tree_states(
            leaves,
            merge_order=merge_order,
            merge_fn=lambda xs: sum(float(x) for x in xs) / float(len(xs)),
            seed=seed,
        )
        return float(root)

    raise ValueError(f"Unsupported aggregator policy: {aggregator!r}")


def aggregate_chunks_topk_event(
    chunks: Sequence[Chunk],
    *,
    target_k: int,
    sketch_order: int,
    merge_order: MergeOrder = MergeOrder.LEFT_TO_RIGHT,
    spike_threshold: float = 0.90,
    seed: int = 0,
) -> float:
    """
    Generic top-k merge-safe event estimator.

    If `target_k <= sketch_order`, this is exact under no chunk loss.
    If `target_k > sketch_order`, estimator is necessarily lossy and reuses
    the highest available order statistic as a proxy.
    """
    if target_k <= 0:
        raise ValueError("target_k must be >= 1")
    m = max(1, int(sketch_order))
    if not chunks:
        return 0.0

    leaves: List[Tuple[float, ...]] = []
    for chunk in chunks:
        vals = sorted((float(v) for v in chunk.values), reverse=True)
        top = vals[:m]
        while len(top) < m:
            top.append(-float("inf"))
        leaves.append(tuple(top))

    def _merge_topm(xs: Sequence[Tuple[float, ...]]) -> Tuple[float, ...]:
        candidates: List[float] = []
        for tup in xs:
            candidates.extend(float(v) for v in tup)
        top = sorted(candidates, reverse=True)[:m]
        while len(top) < m:
            top.append(-float("inf"))
        return tuple(top)

    root = _reduce_tree_states(
        leaves,
        merge_order=merge_order,
        merge_fn=_merge_topm,
        seed=seed,
    )
    if not isinstance(root, tuple):
        root = tuple(root)

    idx = min(target_k, m) - 1
    return 1.0 if float(root[idx]) >= spike_threshold else 0.0


def evaluate_document(
    document: ToyTokenDocument,
    *,
    spec: AblationSpec,
    objective: ObjectiveProfile = ObjectiveProfile.SPIKE_EXISTS,
    spike_threshold: float = 0.90,
    vote_threshold: float = 0.50,
    decision_threshold: float = 0.50,
    seed: int = 0,
) -> DocumentEvaluation:
    """Evaluate one ablation spec on one document."""
    chunks_all = chunk_document(
        document,
        policy=spec.chunker,
        fixed_chunk_size=spec.fixed_chunk_size,
        min_chunk_size=spec.min_chunk_size,
        max_chunk_size=spec.max_chunk_size,
    )
    chunks_kept = select_chunks(
        chunks_all,
        selector=spec.selector,
        chunk_budget=spec.chunk_budget,
        rng=random.Random(seed),
    )
    estimated_score = aggregate_chunks(
        chunks_kept,
        aggregator=spec.aggregator,
        merge_order=spec.merge_order,
        vote_threshold=vote_threshold,
        spike_threshold=spike_threshold,
        boundary_span_tokens=spec.boundary_span_tokens,
        seed=seed,
    )
    true_score = true_objective_score(
        document.token_scores,
        objective=objective,
        spike_threshold=spike_threshold,
    )
    true_label = 1 if true_score >= decision_threshold else 0
    estimated_label = 1 if estimated_score >= decision_threshold else 0
    return DocumentEvaluation(
        spec_name=spec.name,
        true_score=true_score,
        estimated_score=estimated_score,
        true_label=true_label,
        estimated_label=estimated_label,
        abs_error=abs(true_score - estimated_score),
        n_chunks_total=len(chunks_all),
        n_chunks_kept=len(chunks_kept),
    )


def _order_spread_for_doc(
    document: ToyTokenDocument,
    *,
    spec: AblationSpec,
    spike_threshold: float,
    vote_threshold: float,
    seed: int,
) -> Tuple[float, int]:
    chunks_all = chunk_document(
        document,
        policy=spec.chunker,
        fixed_chunk_size=spec.fixed_chunk_size,
        min_chunk_size=spec.min_chunk_size,
        max_chunk_size=spec.max_chunk_size,
    )
    chunks_kept = select_chunks(
        chunks_all,
        selector=spec.selector,
        chunk_budget=spec.chunk_budget,
        rng=random.Random(seed),
    )
    vals = [
        aggregate_chunks(
            chunks_kept,
            aggregator=spec.aggregator,
            merge_order=order,
            spike_threshold=spike_threshold,
            vote_threshold=vote_threshold,
            boundary_span_tokens=spec.boundary_span_tokens,
            seed=seed,
        )
        for order in (MergeOrder.LEFT_TO_RIGHT, MergeOrder.RIGHT_TO_LEFT, MergeOrder.RANDOM)
    ]
    spread = max(vals) - min(vals)
    flip = 1 if ((vals[0] >= 0.5) != (vals[1] >= 0.5)) or ((vals[1] >= 0.5) != (vals[2] >= 0.5)) else 0
    return (spread, flip)


def summarize_ablation(
    docs: Sequence[ToyTokenDocument],
    *,
    spec: AblationSpec,
    objective: ObjectiveProfile = ObjectiveProfile.SPIKE_EXISTS,
    spike_threshold: float = 0.90,
    vote_threshold: float = 0.50,
    decision_threshold: float = 0.50,
    seed: int = 0,
) -> AblationSummary:
    """Compute summary metrics for one ablation method across documents."""
    if not docs:
        raise ValueError("docs must be non-empty")

    evals = [
        evaluate_document(
            doc,
            spec=spec,
            objective=objective,
            spike_threshold=spike_threshold,
            vote_threshold=vote_threshold,
            decision_threshold=decision_threshold,
            seed=seed + i,
        )
        for i, doc in enumerate(docs)
    ]
    order_stats = [
        _order_spread_for_doc(
            doc,
            spec=spec,
            spike_threshold=spike_threshold,
            vote_threshold=vote_threshold,
            seed=seed + i,
        )
        for i, doc in enumerate(docs)
    ]

    n = float(len(evals))
    return AblationSummary(
        name=spec.name,
        description=spec.description,
        n_docs=len(evals),
        mean_true_score=sum(ev.true_score for ev in evals) / n,
        mean_estimated_score=sum(ev.estimated_score for ev in evals) / n,
        mean_abs_error=sum(ev.abs_error for ev in evals) / n,
        label_error_rate=sum(1 for ev in evals if ev.true_label != ev.estimated_label) / n,
        mean_chunks_total=sum(ev.n_chunks_total for ev in evals) / n,
        mean_chunks_kept=sum(ev.n_chunks_kept for ev in evals) / n,
        order_spread_mean=sum(s for s, _ in order_stats) / n,
        order_flip_rate=sum(f for _, f in order_stats) / n,
    )


def default_ablation_specs() -> Tuple[AblationSpec, ...]:
    """Curated side-by-side ablations for repeated aggregation failure modes."""
    return (
        AblationSpec(
            name="merge_safe_oracle_aligned",
            description="Reference: merge-safe max with aligned adaptive chunking and proxy-top selection.",
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
            chunk_budget=6,
            merge_order=MergeOrder.LEFT_TO_RIGHT,
        ),
        AblationSpec(
            name="naive_majority_same_chunker",
            description="Ablation: same chunker/selection, but naive majority tree aggregator.",
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            aggregator=AggregatorPolicy.NAIVE_MAJORITY,
            chunk_budget=6,
            merge_order=MergeOrder.LEFT_TO_RIGHT,
        ),
        AblationSpec(
            name="naive_mean_of_means_same_chunker",
            description="Ablation: same chunker/selection, but unweighted mean-of-means tree aggregator.",
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            aggregator=AggregatorPolicy.NAIVE_MEAN_OF_MEANS,
            chunk_budget=6,
            merge_order=MergeOrder.LEFT_TO_RIGHT,
        ),
        AblationSpec(
            name="right_rule_wrong_chunker",
            description="Ablation: merge-safe max kept, but misspecified adaptive chunking + bottom-proxy selection.",
            chunker=ChunkerPolicy.ADAPTIVE_MISSPECIFIED,
            selector=SelectorPolicy.BOTTOM_PROXY,
            aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
            chunk_budget=6,
            merge_order=MergeOrder.LEFT_TO_RIGHT,
        ),
    )


def run_default_ablation_suite(
    *,
    n_docs: int = 200,
    n_tokens: int = 32,
    seed: int = 0,
) -> List[AblationSummary]:
    """Generate mixed toy documents and run default ablation comparisons."""
    if n_docs <= 0:
        raise ValueError("n_docs must be >= 1")
    patterns = [
        TokenPattern.BOUNDARY_SPIKE,
        TokenPattern.INTERIOR_SPIKE,
        TokenPattern.TWO_SPIKES,
        TokenPattern.DIFFUSE,
    ]
    docs = [
        generate_toy_token_document(
            pattern=patterns[i % len(patterns)],
            n_tokens=n_tokens,
            seed=seed + i,
        )
        for i in range(n_docs)
    ]
    return [summarize_ablation(docs, spec=spec, seed=seed) for spec in default_ablation_specs()]


def worked_failure_examples() -> List[Tuple[str, List[float], List[Tuple[str, float]]]]:
    """
    Deterministic tiny examples to inspect by hand.

    Returns:
      [(name, token_scores, [(method_label, estimated_score), ...]), ...]
    """
    examples: List[Tuple[str, List[float], List[Tuple[str, float]]]] = []

    doc = ToyTokenDocument(
        token_scores=(0.10, 0.11, 0.10, 0.99, 0.10, 0.09, 0.11, 0.10),
        proxy_scores=(0.10, 0.12, 0.11, 0.95, 0.12, 0.10, 0.12, 0.11),
    )
    methods = [
        ("merge-safe-max", AblationSpec("a", "", ChunkerPolicy.FIXED, SelectorPolicy.ALL, AggregatorPolicy.MERGE_SAFE_MAX)),
        ("naive-majority", AblationSpec("b", "", ChunkerPolicy.FIXED, SelectorPolicy.ALL, AggregatorPolicy.NAIVE_MAJORITY)),
        ("naive-mean-of-means", AblationSpec("c", "", ChunkerPolicy.FIXED, SelectorPolicy.ALL, AggregatorPolicy.NAIVE_MEAN_OF_MEANS)),
    ]
    outputs: List[Tuple[str, float]] = []
    for label, spec in methods:
        res = evaluate_document(doc, spec=spec, decision_threshold=0.5, seed=7)
        outputs.append((label, res.estimated_score))
    examples.append(("single-spike-naive-vs-merge-safe", list(doc.token_scores), outputs))

    doc2 = ToyTokenDocument(
        token_scores=(0.10, 0.10, 0.11, 0.98, 0.09, 0.10, 0.11, 0.10, 0.09, 0.10, 0.11, 0.10),
        proxy_scores=(0.10, 0.10, 0.12, 0.97, 0.09, 0.10, 0.11, 0.10, 0.09, 0.10, 0.11, 0.10),
    )
    right_rule = AblationSpec(
        "d",
        "",
        ChunkerPolicy.ADAPTIVE_MISSPECIFIED,
        SelectorPolicy.BOTTOM_PROXY,
        AggregatorPolicy.MERGE_SAFE_MAX,
        chunk_budget=2,
    )
    fail_res = evaluate_document(doc2, spec=right_rule, decision_threshold=0.5, seed=11)
    good_res = evaluate_document(
        doc2,
        spec=AblationSpec(
            "e",
            "",
            ChunkerPolicy.ADAPTIVE_ALIGNED,
            SelectorPolicy.TOP_PROXY,
            AggregatorPolicy.MERGE_SAFE_MAX,
            chunk_budget=2,
        ),
        decision_threshold=0.5,
        seed=11,
    )
    examples.append(
        (
            "right-rule-wrong-chunker",
            list(doc2.token_scores),
            [("aligned-keeps-spike", good_res.estimated_score), ("mispecified-drops-spike", fail_res.estimated_score)],
        )
    )

    return examples


def _sample_pattern_from_spike_mixture(spec: SpikeMixtureDistributionSpec, rng: random.Random) -> TokenPattern:
    if rng.random() >= _clip01(spec.p_spike_doc):
        return TokenPattern.DIFFUSE
    draw = rng.random()
    p_two = _clip01(spec.p_two_spikes_given_spike)
    p_boundary = _clip01(spec.p_boundary_given_spike)
    p_multi_given_two = _clip01(spec.p_multi_given_two_spikes)
    if p_two + p_boundary > 1.0:
        raise ValueError(
            "Invalid spike-mixture specification: require "
            "p_two_spikes_given_spike + p_boundary_given_spike <= 1 "
            "for disjoint toy categories."
        )
    if draw < p_two:
        return TokenPattern.MULTI_SPIKES if rng.random() < p_multi_given_two else TokenPattern.TWO_SPIKES
    if draw < p_two + p_boundary:
        return TokenPattern.BOUNDARY_SPIKE
    return TokenPattern.INTERIOR_SPIKE


def _draw_doc_length(spec: SpikeMixtureDistributionSpec, rng: random.Random) -> int:
    return _draw_doc_length_from_support(
        n_tokens_default=spec.n_tokens,
        support=spec.token_length_support,
        probs=spec.token_length_probs,
        rng=rng,
    )


def _draw_doc_length_from_support(
    *,
    n_tokens_default: int,
    support: Optional[Tuple[int, ...]],
    probs: Optional[Tuple[float, ...]],
    rng: random.Random,
) -> int:
    if support is None:
        return int(n_tokens_default)
    if len(support) == 0:
        raise ValueError("token_length_support must be non-empty when provided")
    if probs is not None and len(probs) != len(support):
        raise ValueError("token_length_probs must match token_length_support length")
    if any(int(n) <= 0 for n in support):
        raise ValueError("All token_length_support values must be >= 1")
    if probs is None:
        weights = [1.0] * len(support)
    else:
        weights = [float(p) for p in probs]
    if any(p < 0 for p in weights):
        raise ValueError("token_length_probs must be non-negative")
    if sum(weights) <= 0:
        raise ValueError("token_length_probs must sum to > 0")
    return int(rng.choices(list(support), weights=weights, k=1)[0])


def _validate_spike_count_distribution(spec: SpikeCountMixtureDistributionSpec) -> None:
    if spec.n_tokens <= 0:
        raise ValueError("spec.n_tokens must be >= 1")
    support = tuple(int(x) for x in spec.spike_count_support)
    probs = tuple(float(x) for x in spec.spike_count_probs_given_spike)
    if len(support) == 0:
        raise ValueError("spike_count_support must be non-empty")
    if len(support) != len(probs):
        raise ValueError("spike_count_support and spike_count_probs_given_spike must have equal length")
    if any(k <= 0 for k in support):
        raise ValueError("spike_count_support values must be >= 1")
    if any(p < 0 for p in probs):
        raise ValueError("spike_count_probs_given_spike must be non-negative")
    if sum(probs) <= 0:
        raise ValueError("spike_count_probs_given_spike must sum to > 0")


def _draw_exact_spike_count_given_spike(spec: SpikeCountMixtureDistributionSpec, rng: random.Random) -> int:
    support = [int(x) for x in spec.spike_count_support]
    probs = [float(x) for x in spec.spike_count_probs_given_spike]
    return int(rng.choices(support, weights=probs, k=1)[0])


def sample_spike_count_mixture_documents(
    *,
    spec: SpikeCountMixtureDistributionSpec,
    n_docs: int,
    seed: int = 0,
) -> List[ToyTokenDocument]:
    """Draw documents from an exact-spike-count mixture DGP."""
    if n_docs <= 0:
        raise ValueError("n_docs must be >= 1")
    _validate_spike_count_distribution(spec)

    rng = random.Random(seed)
    docs: List[ToyTokenDocument] = []
    for _ in range(n_docs):
        n_tokens_doc = _draw_doc_length_from_support(
            n_tokens_default=spec.n_tokens,
            support=spec.token_length_support,
            probs=spec.token_length_probs,
            rng=rng,
        )
        if rng.random() >= _clip01(spec.p_spike_doc):
            n_spikes = 0
            force_boundary = False
        else:
            n_spikes = _draw_exact_spike_count_given_spike(spec, rng)
            force_boundary = rng.random() < _clip01(spec.p_boundary_given_spike)
        doc_seed = rng.randrange(0, 2**31 - 1)
        docs.append(
            generate_exact_spike_count_document(
                n_spikes=n_spikes,
                n_tokens=n_tokens_doc,
                proxy_noise=spec.proxy_noise,
                boundary_span_tokens=spec.boundary_span_tokens,
                force_boundary_spike=force_boundary,
                seed=doc_seed,
            )
        )
    return docs


def sample_spike_mixture_documents(
    *,
    spec: SpikeMixtureDistributionSpec,
    n_docs: int,
    seed: int = 0,
) -> List[ToyTokenDocument]:
    """Draw toy documents from a known spike-mixture distribution."""
    if n_docs <= 0:
        raise ValueError("n_docs must be >= 1")
    if spec.n_tokens <= 0:
        raise ValueError("spec.n_tokens must be >= 1")
    rng = random.Random(seed)
    docs: List[ToyTokenDocument] = []
    for _ in range(n_docs):
        pattern = _sample_pattern_from_spike_mixture(spec, rng)
        doc_seed = rng.randrange(0, 2**31 - 1)
        n_tokens_doc = _draw_doc_length(spec, rng)
        docs.append(
            generate_toy_token_document(
                pattern=pattern,
                n_tokens=n_tokens_doc,
                proxy_noise=spec.proxy_noise,
                boundary_span_tokens=spec.boundary_span_tokens,
                seed=doc_seed,
            )
        )
    return docs


def run_spike_prevalence_recovery_study(
    *,
    distribution: SpikeMixtureDistributionSpec,
    methods: Optional[Sequence[AblationSpec]] = None,
    n_replicates: int = 200,
    docs_per_replicate: int = 200,
    seed: int = 0,
    decision_threshold: float = 0.50,
    weighting_modes: Optional[Sequence[str]] = None,
    legacy_weighting_mode: str = "doc",
) -> List[ParameterRecoverySummary]:
    """
    Estimate spike prevalence parameter under different aggregation pipelines.

    Parameter of interest:
      theta = P(doc has spike) = distribution.p_spike_doc.
    """
    if n_replicates <= 0:
        raise ValueError("n_replicates must be >= 1")
    if docs_per_replicate <= 0:
        raise ValueError("docs_per_replicate must be >= 1")
    modes, legacy_mode = _resolve_weighting(
        weighting_modes=weighting_modes,
        legacy_weighting_mode=legacy_weighting_mode,
    )
    active_methods = list(methods) if methods is not None else list(default_ablation_specs())

    summaries: List[ParameterRecoverySummary] = []
    for m_idx, method in enumerate(active_methods):
        estimates: List[float] = []
        sample_truths: List[float] = []
        estimates_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        sample_truths_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        for rep in range(n_replicates):
            rep_seed = seed + (10_000 * m_idx) + rep
            docs = sample_spike_mixture_documents(
                spec=distribution,
                n_docs=docs_per_replicate,
                seed=rep_seed,
            )
            evals = [
                evaluate_document(
                    doc,
                    spec=method,
                    objective=ObjectiveProfile.SPIKE_EXISTS,
                    decision_threshold=decision_threshold,
                    seed=rep_seed + i,
                )
                for i, doc in enumerate(docs)
            ]
            n = float(len(evals))
            theta_hat = sum(ev.estimated_label for ev in evals) / n
            theta_sample = sum(ev.true_label for ev in evals) / n
            estimates.append(theta_hat)
            sample_truths.append(theta_sample)
            for mode in modes:
                ws = []
                est_lab = []
                tru_lab = []
                for doc, ev in zip(docs, evals):
                    if mode == WeightingMode.DOC:
                        w = 1.0
                    elif mode == WeightingMode.LEAF:
                        w = float(max(1, int(ev.n_chunks_total)))
                    elif mode == WeightingMode.TOKEN:
                        w = float(max(1, int(doc.n_tokens)))
                    else:  # pragma: no cover
                        raise ValueError(f"unsupported weighting mode: {mode!r}")
                    ws.append(w)
                    est_lab.append(float(ev.estimated_label))
                    tru_lab.append(float(ev.true_label))
                wsum = float(sum(ws))
                if wsum <= 0:
                    est_rate = 0.0
                    tru_rate = 0.0
                else:
                    est_rate = float(sum(w * y for w, y in zip(ws, est_lab)) / wsum)
                    tru_rate = float(sum(w * y for w, y in zip(ws, tru_lab)) / wsum)
                estimates_by_mode[mode.value].append(est_rate)
                sample_truths_by_mode[mode.value].append(tru_rate)

        n_rep = float(len(estimates))
        mean_est = sum(estimates) / n_rep
        mean_sample = sum(sample_truths) / n_rep
        mean_bias = mean_est - distribution.p_spike_doc
        mean_abs_bias = sum(abs(x - distribution.p_spike_doc) for x in estimates) / n_rep
        sample_target_bias = sum((x - y) for x, y in zip(estimates, sample_truths)) / n_rep
        var_est = sum((x - mean_est) * (x - mean_est) for x in estimates) / n_rep
        rmse = math.sqrt(sum((x - distribution.p_spike_doc) ** 2 for x in estimates) / n_rep)

        summaries.append(
            ParameterRecoverySummary(
                method_name=method.name,
                description=method.description,
                true_param=distribution.p_spike_doc,
                n_replicates=n_replicates,
                docs_per_replicate=docs_per_replicate,
                mean_estimate=mean_est,
                mean_bias=mean_bias,
                mean_abs_bias=mean_abs_bias,
                sample_target_bias=sample_target_bias,
                std_estimate=math.sqrt(var_est),
                rmse=rmse,
                legacy_weighting_mode=legacy_mode.value,
                weighting_views=build_weighting_views_from_replicates(
                    estimates_by_mode=estimates_by_mode,
                    sample_targets_by_mode=sample_truths_by_mode,
                    true_target=float(distribution.p_spike_doc),
                ),
            )
        )

    return summaries


def default_two_parameter_method_specs() -> Tuple[AblationSpec, ...]:
    """Curated method set for joint recovery of spike prevalence and two-spike prevalence."""
    return (
        AblationSpec(
            name="one_pass_oracle",
            description="Oracle baseline: single chunk over full document (no chunk loss).",
            chunker=ChunkerPolicy.FIXED,
            selector=SelectorPolicy.ALL,
            aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
            two_spike_aggregator=AggregatorPolicy.MERGE_SAFE_SECOND_MAX,
            chunk_budget=None,
            fixed_chunk_size=10**9,
        ),
        AblationSpec(
            name="full_model_aligned",
            description="Full model: aligned adaptive chunking + top-proxy selection + merge-safe operators.",
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
            two_spike_aggregator=AggregatorPolicy.MERGE_SAFE_SECOND_MAX,
            chunk_budget=6,
        ),
        AblationSpec(
            name="naive_majority_same_chunker",
            description="Ablation: same chunker, naive majority aggregation used for both targets.",
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            aggregator=AggregatorPolicy.NAIVE_MAJORITY,
            two_spike_aggregator=AggregatorPolicy.NAIVE_MAJORITY,
            chunk_budget=6,
        ),
        AblationSpec(
            name="naive_mean_of_means_same_chunker",
            description="Ablation: same chunker, unweighted mean-of-means used for both targets.",
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            aggregator=AggregatorPolicy.NAIVE_MEAN_OF_MEANS,
            two_spike_aggregator=AggregatorPolicy.NAIVE_MEAN_OF_MEANS,
            chunk_budget=6,
        ),
        AblationSpec(
            name="right_rule_wrong_chunker",
            description="Ablation: merge-safe operators but misspecified adaptive chunking + bottom-proxy selection.",
            chunker=ChunkerPolicy.ADAPTIVE_MISSPECIFIED,
            selector=SelectorPolicy.BOTTOM_PROXY,
            aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
            two_spike_aggregator=AggregatorPolicy.MERGE_SAFE_SECOND_MAX,
            chunk_budget=6,
        ),
    )


def run_two_parameter_recovery_study(
    *,
    distribution: SpikeMixtureDistributionSpec,
    methods: Optional[Sequence[AblationSpec]] = None,
    n_replicates: int = 200,
    docs_per_replicate: int = 200,
    seed: int = 0,
    spike_threshold: float = 0.90,
    decision_threshold: float = 0.50,
    weighting_modes: Optional[Sequence[str]] = None,
    legacy_weighting_mode: str = "doc",
) -> List[TwoParameterRecoverySummary]:
    """
    Joint recovery for:
      p_spike = P(doc has >=1 spike),
      p_two_given_spike = P(doc has >=2 spikes | doc has >=1 spike).
    """
    if n_replicates <= 0:
        raise ValueError("n_replicates must be >= 1")
    if docs_per_replicate <= 0:
        raise ValueError("docs_per_replicate must be >= 1")
    modes, legacy_mode = _resolve_weighting(
        weighting_modes=weighting_modes,
        legacy_weighting_mode=legacy_weighting_mode,
    )

    true_p_spike = _clip01(distribution.p_spike_doc)
    true_p_two_given = _clip01(distribution.p_two_spikes_given_spike)
    true_p_two_doc = true_p_spike * true_p_two_given

    active_methods = list(methods) if methods is not None else list(default_two_parameter_method_specs())
    summaries: List[TwoParameterRecoverySummary] = []

    for m_idx, method in enumerate(active_methods):
        two_agg = method.two_spike_aggregator
        if two_agg is None:
            two_agg = method.aggregator

        est_spike_rates: List[float] = []
        est_two_doc_rates: List[float] = []
        est_two_given_rates: List[float] = []

        true_spike_rates: List[float] = []
        true_two_doc_rates: List[float] = []
        true_two_given_rates: List[float] = []

        est_spike_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        est_two_doc_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        est_two_given_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        true_spike_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        true_two_doc_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        true_two_given_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}

        for rep in range(n_replicates):
            rep_seed = seed + (20_000 * m_idx) + rep
            docs = sample_spike_mixture_documents(
                spec=distribution,
                n_docs=docs_per_replicate,
                seed=rep_seed,
            )

            est_spike = 0.0
            est_two = 0.0
            true_spike = 0.0
            true_two = 0.0
            mode_tot_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_spike_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_two_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_two_inter_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_spike_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_two_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_two_inter_weight: Dict[str, float] = {m.value: 0.0 for m in modes}

            for i, doc in enumerate(docs):
                doc_seed = rep_seed + i
                chunks_all = chunk_document(
                    doc,
                    policy=method.chunker,
                    fixed_chunk_size=method.fixed_chunk_size,
                    min_chunk_size=method.min_chunk_size,
                    max_chunk_size=method.max_chunk_size,
                )
                chunks_kept = select_chunks(
                    chunks_all,
                    selector=method.selector,
                    chunk_budget=method.chunk_budget,
                    rng=random.Random(doc_seed),
                )
                est_spike_score = aggregate_chunks(
                    chunks_kept,
                    aggregator=method.aggregator,
                    merge_order=method.merge_order,
                    spike_threshold=spike_threshold,
                    seed=doc_seed,
                )
                est_two_score = aggregate_chunks(
                    chunks_kept,
                    aggregator=two_agg,
                    merge_order=method.merge_order,
                    spike_threshold=spike_threshold,
                    seed=doc_seed,
                )
                est_spike += 1.0 if est_spike_score >= decision_threshold else 0.0
                est_two += 1.0 if est_two_score >= decision_threshold else 0.0
                t_spike = true_objective_score(
                    doc.token_scores,
                    objective=ObjectiveProfile.SPIKE_EXISTS,
                    spike_threshold=spike_threshold,
                )
                t_two = float(true_two_spike_event(doc.token_scores, spike_threshold=spike_threshold))
                true_spike += t_spike
                true_two += t_two

                est_spike_event = 1.0 if est_spike_score >= decision_threshold else 0.0
                est_two_event = 1.0 if est_two_score >= decision_threshold else 0.0
                true_spike_event = float(t_spike >= decision_threshold)
                true_two_event = float(t_two >= decision_threshold)
                for mode in modes:
                    w = _doc_weight(doc=doc, all_chunks=chunks_all, mode=mode)
                    mode_tot_weight[mode.value] += w
                    mode_est_spike_weight[mode.value] += w * est_spike_event
                    mode_est_two_weight[mode.value] += w * est_two_event
                    mode_est_two_inter_weight[mode.value] += w * (1.0 if (est_spike_event >= 0.5 and est_two_event >= 0.5) else 0.0)
                    mode_true_spike_weight[mode.value] += w * true_spike_event
                    mode_true_two_weight[mode.value] += w * true_two_event
                    mode_true_two_inter_weight[mode.value] += w * (1.0 if (true_spike_event >= 0.5 and true_two_event >= 0.5) else 0.0)

            n = float(len(docs))
            est_spike_rate = est_spike / n
            est_two_doc_rate = est_two / n
            true_spike_rate = true_spike / n
            true_two_doc_rate = true_two / n

            est_two_given = est_two_doc_rate / est_spike_rate if est_spike_rate > 0 else 0.0
            true_two_given = true_two_doc_rate / true_spike_rate if true_spike_rate > 0 else 0.0

            est_spike_rates.append(est_spike_rate)
            est_two_doc_rates.append(est_two_doc_rate)
            est_two_given_rates.append(est_two_given)
            true_spike_rates.append(true_spike_rate)
            true_two_doc_rates.append(true_two_doc_rate)
            true_two_given_rates.append(true_two_given)
            for mode in modes:
                mode_key = mode.value
                w_tot = mode_tot_weight[mode_key]
                est_spike_w = mode_est_spike_weight[mode_key] / w_tot if w_tot > 0 else 0.0
                est_two_doc_w = mode_est_two_weight[mode_key] / w_tot if w_tot > 0 else 0.0
                true_spike_w = mode_true_spike_weight[mode_key] / w_tot if w_tot > 0 else 0.0
                true_two_doc_w = mode_true_two_weight[mode_key] / w_tot if w_tot > 0 else 0.0
                est_two_given_w = (
                    mode_est_two_inter_weight[mode_key] / mode_est_spike_weight[mode_key]
                    if mode_est_spike_weight[mode_key] > 0
                    else 0.0
                )
                true_two_given_w = (
                    mode_true_two_inter_weight[mode_key] / mode_true_spike_weight[mode_key]
                    if mode_true_spike_weight[mode_key] > 0
                    else 0.0
                )
                est_spike_by_mode[mode_key].append(float(est_spike_w))
                est_two_doc_by_mode[mode_key].append(float(est_two_doc_w))
                est_two_given_by_mode[mode_key].append(float(est_two_given_w))
                true_spike_by_mode[mode_key].append(float(true_spike_w))
                true_two_doc_by_mode[mode_key].append(float(true_two_doc_w))
                true_two_given_by_mode[mode_key].append(float(true_two_given_w))

        n_rep = float(n_replicates)
        mean_hat_spike = sum(est_spike_rates) / n_rep
        mean_hat_two_doc = sum(est_two_doc_rates) / n_rep
        mean_hat_two_given = sum(est_two_given_rates) / n_rep

        mean_true_spike_sample = sum(true_spike_rates) / n_rep
        mean_true_two_doc_sample = sum(true_two_doc_rates) / n_rep
        mean_true_two_given_sample = sum(true_two_given_rates) / n_rep

        supports_two = two_agg == AggregatorPolicy.MERGE_SAFE_SECOND_MAX

        summaries.append(
            TwoParameterRecoverySummary(
                method_name=method.name,
                description=method.description,
                supports_two_spike=supports_two,
                true_p_spike=true_p_spike,
                true_p_two_given_spike=true_p_two_given,
                true_p_two_doc=true_p_two_doc,
                n_replicates=n_replicates,
                docs_per_replicate=docs_per_replicate,
                mean_hat_p_spike=mean_hat_spike,
                bias_p_spike=mean_hat_spike - true_p_spike,
                mean_abs_bias_p_spike=sum(abs(x - true_p_spike) for x in est_spike_rates) / n_rep,
                rmse_p_spike=math.sqrt(sum((x - true_p_spike) ** 2 for x in est_spike_rates) / n_rep),
                mean_hat_p_two_given_spike=mean_hat_two_given,
                bias_p_two_given_spike=mean_hat_two_given - true_p_two_given,
                mean_abs_bias_p_two_given_spike=sum(abs(x - true_p_two_given) for x in est_two_given_rates) / n_rep,
                rmse_p_two_given_spike=math.sqrt(sum((x - true_p_two_given) ** 2 for x in est_two_given_rates) / n_rep),
                mean_hat_p_two_doc=mean_hat_two_doc,
                bias_p_two_doc=mean_hat_two_doc - true_p_two_doc,
                mean_abs_bias_p_two_doc=sum(abs(x - true_p_two_doc) for x in est_two_doc_rates) / n_rep,
                rmse_p_two_doc=math.sqrt(sum((x - true_p_two_doc) ** 2 for x in est_two_doc_rates) / n_rep),
                sample_target_bias_p_spike=sum((x - y) for x, y in zip(est_spike_rates, true_spike_rates)) / n_rep,
                sample_target_bias_p_two_given_spike=sum((x - y) for x, y in zip(est_two_given_rates, true_two_given_rates)) / n_rep,
                sample_target_bias_p_two_doc=sum((x - y) for x, y in zip(est_two_doc_rates, true_two_doc_rates)) / n_rep,
                legacy_weighting_mode=legacy_mode.value,
                weighting_views={
                    mode.value: {
                        "p_spike": build_weighting_views_from_replicates(
                            estimates_by_mode=est_spike_by_mode,
                            sample_targets_by_mode=true_spike_by_mode,
                            true_target=true_p_spike,
                        )[mode.value],
                        "p_two_doc": build_weighting_views_from_replicates(
                            estimates_by_mode=est_two_doc_by_mode,
                            sample_targets_by_mode=true_two_doc_by_mode,
                            true_target=true_p_two_doc,
                        )[mode.value],
                        "p_two_given_spike": build_weighting_views_from_replicates(
                            estimates_by_mode=est_two_given_by_mode,
                            sample_targets_by_mode=true_two_given_by_mode,
                            true_target=true_p_two_given,
                        )[mode.value],
                    }
                    for mode in modes
                },
            )
        )

    return summaries


def default_three_parameter_method_specs() -> Tuple[AblationSpec, ...]:
    """Curated method set for (p_spike, p_two|spike, p_boundary|spike) recovery."""
    return (
        AblationSpec(
            name="one_pass_oracle",
            description="Oracle baseline: one-pass full document with merge-safe sufficient statistics.",
            chunker=ChunkerPolicy.FIXED,
            selector=SelectorPolicy.ALL,
            aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
            two_spike_aggregator=AggregatorPolicy.MERGE_SAFE_SECOND_MAX,
            boundary_spike_aggregator=AggregatorPolicy.MERGE_SAFE_BOUNDARY_MAX,
            chunk_budget=None,
            fixed_chunk_size=10**9,
        ),
        AblationSpec(
            name="full_model_aligned",
            description="Full model: aligned adaptive chunking + top-proxy selection + merge-safe operators.",
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
            two_spike_aggregator=AggregatorPolicy.MERGE_SAFE_SECOND_MAX,
            boundary_spike_aggregator=AggregatorPolicy.MERGE_SAFE_BOUNDARY_MAX,
            chunk_budget=6,
        ),
        AblationSpec(
            name="full_model_missing_boundary_stat",
            description="Ablation: full model chunking/selection but boundary target uses generic spike statistic.",
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
            two_spike_aggregator=AggregatorPolicy.MERGE_SAFE_SECOND_MAX,
            boundary_spike_aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
            chunk_budget=6,
        ),
        AblationSpec(
            name="naive_majority_same_chunker",
            description="Ablation: same chunker, naive majority aggregation for all targets.",
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            aggregator=AggregatorPolicy.NAIVE_MAJORITY,
            two_spike_aggregator=AggregatorPolicy.NAIVE_MAJORITY,
            boundary_spike_aggregator=AggregatorPolicy.NAIVE_MAJORITY,
            chunk_budget=6,
        ),
        AblationSpec(
            name="naive_mean_of_means_same_chunker",
            description="Ablation: same chunker, unweighted mean-of-means for all targets.",
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            aggregator=AggregatorPolicy.NAIVE_MEAN_OF_MEANS,
            two_spike_aggregator=AggregatorPolicy.NAIVE_MEAN_OF_MEANS,
            boundary_spike_aggregator=AggregatorPolicy.NAIVE_MEAN_OF_MEANS,
            chunk_budget=6,
        ),
        AblationSpec(
            name="right_rule_wrong_chunker",
            description="Ablation: merge-safe operators but misspecified chunking + bottom-proxy selection.",
            chunker=ChunkerPolicy.ADAPTIVE_MISSPECIFIED,
            selector=SelectorPolicy.BOTTOM_PROXY,
            aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
            two_spike_aggregator=AggregatorPolicy.MERGE_SAFE_SECOND_MAX,
            boundary_spike_aggregator=AggregatorPolicy.MERGE_SAFE_BOUNDARY_MAX,
            chunk_budget=6,
        ),
    )


def run_three_parameter_recovery_study(
    *,
    distribution: SpikeMixtureDistributionSpec,
    methods: Optional[Sequence[AblationSpec]] = None,
    n_replicates: int = 200,
    docs_per_replicate: int = 200,
    seed: int = 0,
    spike_threshold: float = 0.90,
    decision_threshold: float = 0.50,
    weighting_modes: Optional[Sequence[str]] = None,
    legacy_weighting_mode: str = "doc",
) -> List[ThreeParameterRecoverySummary]:
    """
    Joint recovery for:
      p_spike = P(doc has >=1 spike),
      p_two_given_spike = P(doc has >=2 spikes | doc has >=1 spike),
      p_boundary_given_spike = P(doc has boundary spike | doc has >=1 spike).
    """
    if n_replicates <= 0:
        raise ValueError("n_replicates must be >= 1")
    if docs_per_replicate <= 0:
        raise ValueError("docs_per_replicate must be >= 1")
    modes, legacy_mode = _resolve_weighting(
        weighting_modes=weighting_modes,
        legacy_weighting_mode=legacy_weighting_mode,
    )

    true_p_spike = _clip01(distribution.p_spike_doc)
    true_p_two_given = _clip01(distribution.p_two_spikes_given_spike)
    true_p_boundary_given = _clip01(distribution.p_boundary_given_spike)
    true_p_two_doc = true_p_spike * true_p_two_given
    true_p_boundary_doc = true_p_spike * true_p_boundary_given

    active_methods = list(methods) if methods is not None else list(default_three_parameter_method_specs())
    summaries: List[ThreeParameterRecoverySummary] = []

    for m_idx, method in enumerate(active_methods):
        two_agg = method.two_spike_aggregator if method.two_spike_aggregator is not None else method.aggregator
        boundary_agg = (
            method.boundary_spike_aggregator if method.boundary_spike_aggregator is not None else method.aggregator
        )

        est_spike_rates: List[float] = []
        est_two_doc_rates: List[float] = []
        est_boundary_doc_rates: List[float] = []
        est_two_given_rates: List[float] = []
        est_boundary_given_rates: List[float] = []

        true_spike_rates: List[float] = []
        true_two_doc_rates: List[float] = []
        true_boundary_doc_rates: List[float] = []
        true_two_given_rates: List[float] = []
        true_boundary_given_rates: List[float] = []
        est_spike_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        est_two_doc_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        est_boundary_doc_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        est_two_given_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        est_boundary_given_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        true_spike_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        true_two_doc_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        true_boundary_doc_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        true_two_given_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        true_boundary_given_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}

        for rep in range(n_replicates):
            rep_seed = seed + (30_000 * m_idx) + rep
            docs = sample_spike_mixture_documents(
                spec=distribution,
                n_docs=docs_per_replicate,
                seed=rep_seed,
            )

            est_spike = 0.0
            est_two = 0.0
            est_boundary = 0.0
            true_spike = 0.0
            true_two = 0.0
            true_boundary = 0.0
            mode_tot_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_spike_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_two_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_boundary_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_two_inter_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_boundary_inter_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_spike_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_two_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_boundary_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_two_inter_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_boundary_inter_weight: Dict[str, float] = {m.value: 0.0 for m in modes}

            for i, doc in enumerate(docs):
                doc_seed = rep_seed + i
                chunks_all = chunk_document(
                    doc,
                    policy=method.chunker,
                    fixed_chunk_size=method.fixed_chunk_size,
                    min_chunk_size=method.min_chunk_size,
                    max_chunk_size=method.max_chunk_size,
                )
                chunks_kept = select_chunks(
                    chunks_all,
                    selector=method.selector,
                    chunk_budget=method.chunk_budget,
                    rng=random.Random(doc_seed),
                )
                est_spike_score = aggregate_chunks(
                    chunks_kept,
                    aggregator=method.aggregator,
                    merge_order=method.merge_order,
                    spike_threshold=spike_threshold,
                    boundary_span_tokens=method.boundary_span_tokens,
                    seed=doc_seed,
                )
                est_two_score = aggregate_chunks(
                    chunks_kept,
                    aggregator=two_agg,
                    merge_order=method.merge_order,
                    spike_threshold=spike_threshold,
                    boundary_span_tokens=method.boundary_span_tokens,
                    seed=doc_seed,
                )
                est_boundary_score = aggregate_chunks(
                    chunks_kept,
                    aggregator=boundary_agg,
                    merge_order=method.merge_order,
                    spike_threshold=spike_threshold,
                    boundary_span_tokens=method.boundary_span_tokens,
                    seed=doc_seed,
                )
                est_spike_event = 1.0 if est_spike_score >= decision_threshold else 0.0
                est_two_event = 1.0 if est_two_score >= decision_threshold else 0.0
                est_boundary_event = 1.0 if est_boundary_score >= decision_threshold else 0.0
                est_spike += est_spike_event
                est_two += est_two_event
                est_boundary += est_boundary_event

                t_spike = true_objective_score(
                    doc.token_scores,
                    objective=ObjectiveProfile.SPIKE_EXISTS,
                    spike_threshold=spike_threshold,
                )
                t_two = float(true_two_spike_event(doc.token_scores, spike_threshold=spike_threshold))
                t_boundary = float(
                    true_boundary_spike_event(
                        doc.token_scores,
                        spike_threshold=spike_threshold,
                        boundary_span_tokens=distribution.boundary_span_tokens,
                    )
                )
                true_spike_event = float(t_spike >= decision_threshold)
                true_two_event = float(t_two >= decision_threshold)
                true_boundary_event = float(t_boundary >= decision_threshold)
                true_spike += true_spike_event
                true_two += true_two_event
                true_boundary += true_boundary_event
                for mode in modes:
                    w = _doc_weight(doc=doc, all_chunks=chunks_all, mode=mode)
                    mode_key = mode.value
                    mode_tot_weight[mode_key] += w
                    mode_est_spike_weight[mode_key] += w * est_spike_event
                    mode_est_two_weight[mode_key] += w * est_two_event
                    mode_est_boundary_weight[mode_key] += w * est_boundary_event
                    mode_est_two_inter_weight[mode_key] += w * (
                        1.0 if (est_spike_event >= 0.5 and est_two_event >= 0.5) else 0.0
                    )
                    mode_est_boundary_inter_weight[mode_key] += w * (
                        1.0 if (est_spike_event >= 0.5 and est_boundary_event >= 0.5) else 0.0
                    )
                    mode_true_spike_weight[mode_key] += w * true_spike_event
                    mode_true_two_weight[mode_key] += w * true_two_event
                    mode_true_boundary_weight[mode_key] += w * true_boundary_event
                    mode_true_two_inter_weight[mode_key] += w * (
                        1.0 if (true_spike_event >= 0.5 and true_two_event >= 0.5) else 0.0
                    )
                    mode_true_boundary_inter_weight[mode_key] += w * (
                        1.0 if (true_spike_event >= 0.5 and true_boundary_event >= 0.5) else 0.0
                    )

            n = float(len(docs))
            est_spike_rate = est_spike / n
            est_two_doc_rate = est_two / n
            est_boundary_doc_rate = est_boundary / n
            true_spike_rate = true_spike / n
            true_two_doc_rate = true_two / n
            true_boundary_doc_rate = true_boundary / n

            est_two_given = est_two_doc_rate / est_spike_rate if est_spike_rate > 0 else 0.0
            est_boundary_given = est_boundary_doc_rate / est_spike_rate if est_spike_rate > 0 else 0.0
            true_two_given = true_two_doc_rate / true_spike_rate if true_spike_rate > 0 else 0.0
            true_boundary_given = true_boundary_doc_rate / true_spike_rate if true_spike_rate > 0 else 0.0

            est_spike_rates.append(est_spike_rate)
            est_two_doc_rates.append(est_two_doc_rate)
            est_boundary_doc_rates.append(est_boundary_doc_rate)
            est_two_given_rates.append(est_two_given)
            est_boundary_given_rates.append(est_boundary_given)

            true_spike_rates.append(true_spike_rate)
            true_two_doc_rates.append(true_two_doc_rate)
            true_boundary_doc_rates.append(true_boundary_doc_rate)
            true_two_given_rates.append(true_two_given)
            true_boundary_given_rates.append(true_boundary_given)
            for mode in modes:
                mode_key = mode.value
                w_tot = mode_tot_weight[mode_key]
                est_spike_w = mode_est_spike_weight[mode_key] / w_tot if w_tot > 0 else 0.0
                est_two_doc_w = mode_est_two_weight[mode_key] / w_tot if w_tot > 0 else 0.0
                est_boundary_doc_w = (
                    mode_est_boundary_weight[mode_key] / w_tot if w_tot > 0 else 0.0
                )
                true_spike_w = mode_true_spike_weight[mode_key] / w_tot if w_tot > 0 else 0.0
                true_two_doc_w = mode_true_two_weight[mode_key] / w_tot if w_tot > 0 else 0.0
                true_boundary_doc_w = (
                    mode_true_boundary_weight[mode_key] / w_tot if w_tot > 0 else 0.0
                )
                est_two_given_w = (
                    mode_est_two_inter_weight[mode_key] / mode_est_spike_weight[mode_key]
                    if mode_est_spike_weight[mode_key] > 0
                    else 0.0
                )
                est_boundary_given_w = (
                    mode_est_boundary_inter_weight[mode_key] / mode_est_spike_weight[mode_key]
                    if mode_est_spike_weight[mode_key] > 0
                    else 0.0
                )
                true_two_given_w = (
                    mode_true_two_inter_weight[mode_key] / mode_true_spike_weight[mode_key]
                    if mode_true_spike_weight[mode_key] > 0
                    else 0.0
                )
                true_boundary_given_w = (
                    mode_true_boundary_inter_weight[mode_key] / mode_true_spike_weight[mode_key]
                    if mode_true_spike_weight[mode_key] > 0
                    else 0.0
                )
                est_spike_by_mode[mode_key].append(float(est_spike_w))
                est_two_doc_by_mode[mode_key].append(float(est_two_doc_w))
                est_boundary_doc_by_mode[mode_key].append(float(est_boundary_doc_w))
                est_two_given_by_mode[mode_key].append(float(est_two_given_w))
                est_boundary_given_by_mode[mode_key].append(float(est_boundary_given_w))
                true_spike_by_mode[mode_key].append(float(true_spike_w))
                true_two_doc_by_mode[mode_key].append(float(true_two_doc_w))
                true_boundary_doc_by_mode[mode_key].append(float(true_boundary_doc_w))
                true_two_given_by_mode[mode_key].append(float(true_two_given_w))
                true_boundary_given_by_mode[mode_key].append(float(true_boundary_given_w))

        n_rep = float(n_replicates)
        mean_hat_spike = sum(est_spike_rates) / n_rep
        mean_hat_two_doc = sum(est_two_doc_rates) / n_rep
        mean_hat_boundary_doc = sum(est_boundary_doc_rates) / n_rep
        mean_hat_two_given = sum(est_two_given_rates) / n_rep
        mean_hat_boundary_given = sum(est_boundary_given_rates) / n_rep

        supports_two = two_agg == AggregatorPolicy.MERGE_SAFE_SECOND_MAX
        supports_boundary = boundary_agg == AggregatorPolicy.MERGE_SAFE_BOUNDARY_MAX
        spike_views = build_weighting_views_from_replicates(
            estimates_by_mode=est_spike_by_mode,
            sample_targets_by_mode=true_spike_by_mode,
            true_target=true_p_spike,
        )
        two_doc_views = build_weighting_views_from_replicates(
            estimates_by_mode=est_two_doc_by_mode,
            sample_targets_by_mode=true_two_doc_by_mode,
            true_target=true_p_two_doc,
        )
        boundary_doc_views = build_weighting_views_from_replicates(
            estimates_by_mode=est_boundary_doc_by_mode,
            sample_targets_by_mode=true_boundary_doc_by_mode,
            true_target=true_p_boundary_doc,
        )
        two_given_views = build_weighting_views_from_replicates(
            estimates_by_mode=est_two_given_by_mode,
            sample_targets_by_mode=true_two_given_by_mode,
            true_target=true_p_two_given,
        )
        boundary_given_views = build_weighting_views_from_replicates(
            estimates_by_mode=est_boundary_given_by_mode,
            sample_targets_by_mode=true_boundary_given_by_mode,
            true_target=true_p_boundary_given,
        )

        summaries.append(
            ThreeParameterRecoverySummary(
                method_name=method.name,
                description=method.description,
                supports_two_spike=supports_two,
                supports_boundary_spike=supports_boundary,
                true_p_spike=true_p_spike,
                true_p_two_given_spike=true_p_two_given,
                true_p_boundary_given_spike=true_p_boundary_given,
                true_p_two_doc=true_p_two_doc,
                true_p_boundary_doc=true_p_boundary_doc,
                n_replicates=n_replicates,
                docs_per_replicate=docs_per_replicate,
                mean_hat_p_spike=mean_hat_spike,
                bias_p_spike=mean_hat_spike - true_p_spike,
                mean_abs_bias_p_spike=sum(abs(x - true_p_spike) for x in est_spike_rates) / n_rep,
                rmse_p_spike=math.sqrt(sum((x - true_p_spike) ** 2 for x in est_spike_rates) / n_rep),
                mean_hat_p_two_given_spike=mean_hat_two_given,
                bias_p_two_given_spike=mean_hat_two_given - true_p_two_given,
                mean_abs_bias_p_two_given_spike=sum(abs(x - true_p_two_given) for x in est_two_given_rates) / n_rep,
                rmse_p_two_given_spike=math.sqrt(sum((x - true_p_two_given) ** 2 for x in est_two_given_rates) / n_rep),
                mean_hat_p_boundary_given_spike=mean_hat_boundary_given,
                bias_p_boundary_given_spike=mean_hat_boundary_given - true_p_boundary_given,
                mean_abs_bias_p_boundary_given_spike=sum(abs(x - true_p_boundary_given) for x in est_boundary_given_rates) / n_rep,
                rmse_p_boundary_given_spike=math.sqrt(
                    sum((x - true_p_boundary_given) ** 2 for x in est_boundary_given_rates) / n_rep
                ),
                mean_hat_p_two_doc=mean_hat_two_doc,
                bias_p_two_doc=mean_hat_two_doc - true_p_two_doc,
                mean_abs_bias_p_two_doc=sum(abs(x - true_p_two_doc) for x in est_two_doc_rates) / n_rep,
                rmse_p_two_doc=math.sqrt(sum((x - true_p_two_doc) ** 2 for x in est_two_doc_rates) / n_rep),
                mean_hat_p_boundary_doc=mean_hat_boundary_doc,
                bias_p_boundary_doc=mean_hat_boundary_doc - true_p_boundary_doc,
                mean_abs_bias_p_boundary_doc=sum(abs(x - true_p_boundary_doc) for x in est_boundary_doc_rates) / n_rep,
                rmse_p_boundary_doc=math.sqrt(
                    sum((x - true_p_boundary_doc) ** 2 for x in est_boundary_doc_rates) / n_rep
                ),
                sample_target_bias_p_spike=sum((x - y) for x, y in zip(est_spike_rates, true_spike_rates)) / n_rep,
                sample_target_bias_p_two_given_spike=sum((x - y) for x, y in zip(est_two_given_rates, true_two_given_rates)) / n_rep,
                sample_target_bias_p_boundary_given_spike=sum(
                    (x - y) for x, y in zip(est_boundary_given_rates, true_boundary_given_rates)
                ) / n_rep,
                sample_target_bias_p_two_doc=sum((x - y) for x, y in zip(est_two_doc_rates, true_two_doc_rates)) / n_rep,
                sample_target_bias_p_boundary_doc=sum(
                    (x - y) for x, y in zip(est_boundary_doc_rates, true_boundary_doc_rates)
                ) / n_rep,
                legacy_weighting_mode=legacy_mode.value,
                weighting_views={
                    mode.value: {
                        "p_spike": spike_views[mode.value],
                        "p_two_doc": two_doc_views[mode.value],
                        "p_boundary_doc": boundary_doc_views[mode.value],
                        "p_two_given_spike": two_given_views[mode.value],
                        "p_boundary_given_spike": boundary_given_views[mode.value],
                    }
                    for mode in modes
                },
            )
        )

    return summaries


def default_four_parameter_method_specs() -> Tuple[AblationSpec, ...]:
    """Curated method set for (p_spike, p_two|spike, p_three+|spike, p_boundary|spike) recovery."""
    return (
        AblationSpec(
            name="one_pass_oracle",
            description="Oracle baseline: one-pass full document with merge-safe sufficient statistics.",
            chunker=ChunkerPolicy.FIXED,
            selector=SelectorPolicy.ALL,
            aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
            two_spike_aggregator=AggregatorPolicy.MERGE_SAFE_SECOND_MAX,
            three_spike_aggregator=AggregatorPolicy.MERGE_SAFE_THIRD_MAX,
            boundary_spike_aggregator=AggregatorPolicy.MERGE_SAFE_BOUNDARY_MAX,
            chunk_budget=None,
            fixed_chunk_size=10**9,
        ),
        AblationSpec(
            name="full_model_aligned",
            description="Full model: aligned adaptive chunking + top-proxy selection + merge-safe operators.",
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
            two_spike_aggregator=AggregatorPolicy.MERGE_SAFE_SECOND_MAX,
            three_spike_aggregator=AggregatorPolicy.MERGE_SAFE_THIRD_MAX,
            boundary_spike_aggregator=AggregatorPolicy.MERGE_SAFE_BOUNDARY_MAX,
            chunk_budget=6,
        ),
        AblationSpec(
            name="full_model_missing_three_stat",
            description="Ablation: full model chunking/selection but no merge-safe third-order statistic.",
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
            two_spike_aggregator=AggregatorPolicy.MERGE_SAFE_SECOND_MAX,
            three_spike_aggregator=AggregatorPolicy.MERGE_SAFE_SECOND_MAX,
            boundary_spike_aggregator=AggregatorPolicy.MERGE_SAFE_BOUNDARY_MAX,
            chunk_budget=6,
        ),
        AblationSpec(
            name="full_model_missing_boundary_stat",
            description="Ablation: full model chunking/selection but boundary target uses generic spike statistic.",
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
            two_spike_aggregator=AggregatorPolicy.MERGE_SAFE_SECOND_MAX,
            three_spike_aggregator=AggregatorPolicy.MERGE_SAFE_THIRD_MAX,
            boundary_spike_aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
            chunk_budget=6,
        ),
        AblationSpec(
            name="naive_majority_same_chunker",
            description="Ablation: same chunker, naive majority aggregation for all targets.",
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            aggregator=AggregatorPolicy.NAIVE_MAJORITY,
            two_spike_aggregator=AggregatorPolicy.NAIVE_MAJORITY,
            three_spike_aggregator=AggregatorPolicy.NAIVE_MAJORITY,
            boundary_spike_aggregator=AggregatorPolicy.NAIVE_MAJORITY,
            chunk_budget=6,
        ),
        AblationSpec(
            name="naive_mean_of_means_same_chunker",
            description="Ablation: same chunker, unweighted mean-of-means for all targets.",
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            aggregator=AggregatorPolicy.NAIVE_MEAN_OF_MEANS,
            two_spike_aggregator=AggregatorPolicy.NAIVE_MEAN_OF_MEANS,
            three_spike_aggregator=AggregatorPolicy.NAIVE_MEAN_OF_MEANS,
            boundary_spike_aggregator=AggregatorPolicy.NAIVE_MEAN_OF_MEANS,
            chunk_budget=6,
        ),
        AblationSpec(
            name="right_rule_wrong_chunker",
            description="Ablation: merge-safe operators but misspecified chunking + bottom-proxy selection.",
            chunker=ChunkerPolicy.ADAPTIVE_MISSPECIFIED,
            selector=SelectorPolicy.BOTTOM_PROXY,
            aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
            two_spike_aggregator=AggregatorPolicy.MERGE_SAFE_SECOND_MAX,
            three_spike_aggregator=AggregatorPolicy.MERGE_SAFE_THIRD_MAX,
            boundary_spike_aggregator=AggregatorPolicy.MERGE_SAFE_BOUNDARY_MAX,
            chunk_budget=6,
        ),
    )


def run_four_parameter_recovery_study(
    *,
    distribution: SpikeMixtureDistributionSpec,
    methods: Optional[Sequence[AblationSpec]] = None,
    n_replicates: int = 200,
    docs_per_replicate: int = 200,
    seed: int = 0,
    spike_threshold: float = 0.90,
    decision_threshold: float = 0.50,
    weighting_modes: Optional[Sequence[str]] = None,
    legacy_weighting_mode: str = "doc",
) -> List[FourParameterRecoverySummary]:
    """
    Joint recovery for:
      p_spike = P(doc has >=1 spike),
      p_two_given_spike = P(doc has >=2 spikes | doc has >=1 spike),
      p_three_given_spike = P(doc has >=3 spikes | doc has >=1 spike),
      p_boundary_given_spike = P(doc has boundary spike | doc has >=1 spike).
    """
    if n_replicates <= 0:
        raise ValueError("n_replicates must be >= 1")
    if docs_per_replicate <= 0:
        raise ValueError("docs_per_replicate must be >= 1")
    modes, legacy_mode = _resolve_weighting(
        weighting_modes=weighting_modes,
        legacy_weighting_mode=legacy_weighting_mode,
    )

    true_p_spike = _clip01(distribution.p_spike_doc)
    true_p_two_given = _clip01(distribution.p_two_spikes_given_spike)
    true_p_three_given = true_p_two_given * _clip01(distribution.p_multi_given_two_spikes)
    true_p_boundary_given = _clip01(distribution.p_boundary_given_spike)
    true_p_two_doc = true_p_spike * true_p_two_given
    true_p_three_doc = true_p_spike * true_p_three_given
    true_p_boundary_doc = true_p_spike * true_p_boundary_given

    active_methods = list(methods) if methods is not None else list(default_four_parameter_method_specs())
    summaries: List[FourParameterRecoverySummary] = []

    for m_idx, method in enumerate(active_methods):
        two_agg = method.two_spike_aggregator if method.two_spike_aggregator is not None else method.aggregator
        three_agg = (
            method.three_spike_aggregator if method.three_spike_aggregator is not None else method.aggregator
        )
        boundary_agg = (
            method.boundary_spike_aggregator if method.boundary_spike_aggregator is not None else method.aggregator
        )

        est_spike_rates: List[float] = []
        est_two_doc_rates: List[float] = []
        est_three_doc_rates: List[float] = []
        est_boundary_doc_rates: List[float] = []
        est_two_given_rates: List[float] = []
        est_three_given_rates: List[float] = []
        est_boundary_given_rates: List[float] = []

        true_spike_rates: List[float] = []
        true_two_doc_rates: List[float] = []
        true_three_doc_rates: List[float] = []
        true_boundary_doc_rates: List[float] = []
        true_two_given_rates: List[float] = []
        true_three_given_rates: List[float] = []
        true_boundary_given_rates: List[float] = []
        est_spike_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        est_two_doc_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        est_three_doc_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        est_boundary_doc_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        est_two_given_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        est_three_given_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        est_boundary_given_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        true_spike_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        true_two_doc_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        true_three_doc_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        true_boundary_doc_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        true_two_given_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        true_three_given_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        true_boundary_given_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}

        for rep in range(n_replicates):
            rep_seed = seed + (40_000 * m_idx) + rep
            docs = sample_spike_mixture_documents(
                spec=distribution,
                n_docs=docs_per_replicate,
                seed=rep_seed,
            )

            est_spike = 0.0
            est_two = 0.0
            est_three = 0.0
            est_boundary = 0.0
            true_spike = 0.0
            true_two = 0.0
            true_three = 0.0
            true_boundary = 0.0
            mode_tot_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_spike_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_two_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_three_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_boundary_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_two_inter_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_three_inter_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_boundary_inter_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_spike_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_two_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_three_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_boundary_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_two_inter_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_three_inter_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_boundary_inter_weight: Dict[str, float] = {m.value: 0.0 for m in modes}

            for i, doc in enumerate(docs):
                doc_seed = rep_seed + i
                chunks_all = chunk_document(
                    doc,
                    policy=method.chunker,
                    fixed_chunk_size=method.fixed_chunk_size,
                    min_chunk_size=method.min_chunk_size,
                    max_chunk_size=method.max_chunk_size,
                )
                chunks_kept = select_chunks(
                    chunks_all,
                    selector=method.selector,
                    chunk_budget=method.chunk_budget,
                    rng=random.Random(doc_seed),
                )
                est_spike_score = aggregate_chunks(
                    chunks_kept,
                    aggregator=method.aggregator,
                    merge_order=method.merge_order,
                    spike_threshold=spike_threshold,
                    boundary_span_tokens=method.boundary_span_tokens,
                    seed=doc_seed,
                )
                est_two_score = aggregate_chunks(
                    chunks_kept,
                    aggregator=two_agg,
                    merge_order=method.merge_order,
                    spike_threshold=spike_threshold,
                    boundary_span_tokens=method.boundary_span_tokens,
                    seed=doc_seed,
                )
                est_three_score = aggregate_chunks(
                    chunks_kept,
                    aggregator=three_agg,
                    merge_order=method.merge_order,
                    spike_threshold=spike_threshold,
                    boundary_span_tokens=method.boundary_span_tokens,
                    seed=doc_seed,
                )
                est_boundary_score = aggregate_chunks(
                    chunks_kept,
                    aggregator=boundary_agg,
                    merge_order=method.merge_order,
                    spike_threshold=spike_threshold,
                    boundary_span_tokens=method.boundary_span_tokens,
                    seed=doc_seed,
                )
                est_spike_event = 1.0 if est_spike_score >= decision_threshold else 0.0
                est_two_event = 1.0 if est_two_score >= decision_threshold else 0.0
                est_three_event = 1.0 if est_three_score >= decision_threshold else 0.0
                est_boundary_event = 1.0 if est_boundary_score >= decision_threshold else 0.0
                est_spike += est_spike_event
                est_two += est_two_event
                est_three += est_three_event
                est_boundary += est_boundary_event

                t_spike = true_objective_score(
                    doc.token_scores,
                    objective=ObjectiveProfile.SPIKE_EXISTS,
                    spike_threshold=spike_threshold,
                )
                t_two = float(true_two_spike_event(doc.token_scores, spike_threshold=spike_threshold))
                t_three = float(true_three_plus_spike_event(doc.token_scores, spike_threshold=spike_threshold))
                t_boundary = float(
                    true_boundary_spike_event(
                        doc.token_scores,
                        spike_threshold=spike_threshold,
                        boundary_span_tokens=distribution.boundary_span_tokens,
                    )
                )
                true_spike_event = float(t_spike >= decision_threshold)
                true_two_event = float(t_two >= decision_threshold)
                true_three_event = float(t_three >= decision_threshold)
                true_boundary_event = float(t_boundary >= decision_threshold)
                true_spike += true_spike_event
                true_two += true_two_event
                true_three += true_three_event
                true_boundary += true_boundary_event
                for mode in modes:
                    w = _doc_weight(doc=doc, all_chunks=chunks_all, mode=mode)
                    mode_key = mode.value
                    mode_tot_weight[mode_key] += w
                    mode_est_spike_weight[mode_key] += w * est_spike_event
                    mode_est_two_weight[mode_key] += w * est_two_event
                    mode_est_three_weight[mode_key] += w * est_three_event
                    mode_est_boundary_weight[mode_key] += w * est_boundary_event
                    mode_est_two_inter_weight[mode_key] += w * (
                        1.0 if (est_spike_event >= 0.5 and est_two_event >= 0.5) else 0.0
                    )
                    mode_est_three_inter_weight[mode_key] += w * (
                        1.0 if (est_spike_event >= 0.5 and est_three_event >= 0.5) else 0.0
                    )
                    mode_est_boundary_inter_weight[mode_key] += w * (
                        1.0 if (est_spike_event >= 0.5 and est_boundary_event >= 0.5) else 0.0
                    )
                    mode_true_spike_weight[mode_key] += w * true_spike_event
                    mode_true_two_weight[mode_key] += w * true_two_event
                    mode_true_three_weight[mode_key] += w * true_three_event
                    mode_true_boundary_weight[mode_key] += w * true_boundary_event
                    mode_true_two_inter_weight[mode_key] += w * (
                        1.0 if (true_spike_event >= 0.5 and true_two_event >= 0.5) else 0.0
                    )
                    mode_true_three_inter_weight[mode_key] += w * (
                        1.0 if (true_spike_event >= 0.5 and true_three_event >= 0.5) else 0.0
                    )
                    mode_true_boundary_inter_weight[mode_key] += w * (
                        1.0 if (true_spike_event >= 0.5 and true_boundary_event >= 0.5) else 0.0
                    )

            n = float(len(docs))
            est_spike_rate = est_spike / n
            est_two_doc_rate = est_two / n
            est_three_doc_rate = est_three / n
            est_boundary_doc_rate = est_boundary / n
            true_spike_rate = true_spike / n
            true_two_doc_rate = true_two / n
            true_three_doc_rate = true_three / n
            true_boundary_doc_rate = true_boundary / n

            est_two_given = est_two_doc_rate / est_spike_rate if est_spike_rate > 0 else 0.0
            est_three_given = est_three_doc_rate / est_spike_rate if est_spike_rate > 0 else 0.0
            est_boundary_given = est_boundary_doc_rate / est_spike_rate if est_spike_rate > 0 else 0.0
            true_two_given = true_two_doc_rate / true_spike_rate if true_spike_rate > 0 else 0.0
            true_three_given = true_three_doc_rate / true_spike_rate if true_spike_rate > 0 else 0.0
            true_boundary_given = true_boundary_doc_rate / true_spike_rate if true_spike_rate > 0 else 0.0

            est_spike_rates.append(est_spike_rate)
            est_two_doc_rates.append(est_two_doc_rate)
            est_three_doc_rates.append(est_three_doc_rate)
            est_boundary_doc_rates.append(est_boundary_doc_rate)
            est_two_given_rates.append(est_two_given)
            est_three_given_rates.append(est_three_given)
            est_boundary_given_rates.append(est_boundary_given)

            true_spike_rates.append(true_spike_rate)
            true_two_doc_rates.append(true_two_doc_rate)
            true_three_doc_rates.append(true_three_doc_rate)
            true_boundary_doc_rates.append(true_boundary_doc_rate)
            true_two_given_rates.append(true_two_given)
            true_three_given_rates.append(true_three_given)
            true_boundary_given_rates.append(true_boundary_given)
            for mode in modes:
                mode_key = mode.value
                w_tot = mode_tot_weight[mode_key]
                est_spike_w = mode_est_spike_weight[mode_key] / w_tot if w_tot > 0 else 0.0
                est_two_doc_w = mode_est_two_weight[mode_key] / w_tot if w_tot > 0 else 0.0
                est_three_doc_w = mode_est_three_weight[mode_key] / w_tot if w_tot > 0 else 0.0
                est_boundary_doc_w = (
                    mode_est_boundary_weight[mode_key] / w_tot if w_tot > 0 else 0.0
                )
                true_spike_w = mode_true_spike_weight[mode_key] / w_tot if w_tot > 0 else 0.0
                true_two_doc_w = mode_true_two_weight[mode_key] / w_tot if w_tot > 0 else 0.0
                true_three_doc_w = mode_true_three_weight[mode_key] / w_tot if w_tot > 0 else 0.0
                true_boundary_doc_w = (
                    mode_true_boundary_weight[mode_key] / w_tot if w_tot > 0 else 0.0
                )
                est_two_given_w = (
                    mode_est_two_inter_weight[mode_key] / mode_est_spike_weight[mode_key]
                    if mode_est_spike_weight[mode_key] > 0
                    else 0.0
                )
                est_three_given_w = (
                    mode_est_three_inter_weight[mode_key] / mode_est_spike_weight[mode_key]
                    if mode_est_spike_weight[mode_key] > 0
                    else 0.0
                )
                est_boundary_given_w = (
                    mode_est_boundary_inter_weight[mode_key] / mode_est_spike_weight[mode_key]
                    if mode_est_spike_weight[mode_key] > 0
                    else 0.0
                )
                true_two_given_w = (
                    mode_true_two_inter_weight[mode_key] / mode_true_spike_weight[mode_key]
                    if mode_true_spike_weight[mode_key] > 0
                    else 0.0
                )
                true_three_given_w = (
                    mode_true_three_inter_weight[mode_key] / mode_true_spike_weight[mode_key]
                    if mode_true_spike_weight[mode_key] > 0
                    else 0.0
                )
                true_boundary_given_w = (
                    mode_true_boundary_inter_weight[mode_key] / mode_true_spike_weight[mode_key]
                    if mode_true_spike_weight[mode_key] > 0
                    else 0.0
                )
                est_spike_by_mode[mode_key].append(float(est_spike_w))
                est_two_doc_by_mode[mode_key].append(float(est_two_doc_w))
                est_three_doc_by_mode[mode_key].append(float(est_three_doc_w))
                est_boundary_doc_by_mode[mode_key].append(float(est_boundary_doc_w))
                est_two_given_by_mode[mode_key].append(float(est_two_given_w))
                est_three_given_by_mode[mode_key].append(float(est_three_given_w))
                est_boundary_given_by_mode[mode_key].append(float(est_boundary_given_w))
                true_spike_by_mode[mode_key].append(float(true_spike_w))
                true_two_doc_by_mode[mode_key].append(float(true_two_doc_w))
                true_three_doc_by_mode[mode_key].append(float(true_three_doc_w))
                true_boundary_doc_by_mode[mode_key].append(float(true_boundary_doc_w))
                true_two_given_by_mode[mode_key].append(float(true_two_given_w))
                true_three_given_by_mode[mode_key].append(float(true_three_given_w))
                true_boundary_given_by_mode[mode_key].append(float(true_boundary_given_w))

        n_rep = float(n_replicates)
        mean_hat_spike = sum(est_spike_rates) / n_rep
        mean_hat_two_doc = sum(est_two_doc_rates) / n_rep
        mean_hat_three_doc = sum(est_three_doc_rates) / n_rep
        mean_hat_boundary_doc = sum(est_boundary_doc_rates) / n_rep
        mean_hat_two_given = sum(est_two_given_rates) / n_rep
        mean_hat_three_given = sum(est_three_given_rates) / n_rep
        mean_hat_boundary_given = sum(est_boundary_given_rates) / n_rep

        supports_two = two_agg == AggregatorPolicy.MERGE_SAFE_SECOND_MAX
        supports_three = three_agg == AggregatorPolicy.MERGE_SAFE_THIRD_MAX
        supports_boundary = boundary_agg == AggregatorPolicy.MERGE_SAFE_BOUNDARY_MAX
        spike_views = build_weighting_views_from_replicates(
            estimates_by_mode=est_spike_by_mode,
            sample_targets_by_mode=true_spike_by_mode,
            true_target=true_p_spike,
        )
        two_doc_views = build_weighting_views_from_replicates(
            estimates_by_mode=est_two_doc_by_mode,
            sample_targets_by_mode=true_two_doc_by_mode,
            true_target=true_p_two_doc,
        )
        three_doc_views = build_weighting_views_from_replicates(
            estimates_by_mode=est_three_doc_by_mode,
            sample_targets_by_mode=true_three_doc_by_mode,
            true_target=true_p_three_doc,
        )
        boundary_doc_views = build_weighting_views_from_replicates(
            estimates_by_mode=est_boundary_doc_by_mode,
            sample_targets_by_mode=true_boundary_doc_by_mode,
            true_target=true_p_boundary_doc,
        )
        two_given_views = build_weighting_views_from_replicates(
            estimates_by_mode=est_two_given_by_mode,
            sample_targets_by_mode=true_two_given_by_mode,
            true_target=true_p_two_given,
        )
        three_given_views = build_weighting_views_from_replicates(
            estimates_by_mode=est_three_given_by_mode,
            sample_targets_by_mode=true_three_given_by_mode,
            true_target=true_p_three_given,
        )
        boundary_given_views = build_weighting_views_from_replicates(
            estimates_by_mode=est_boundary_given_by_mode,
            sample_targets_by_mode=true_boundary_given_by_mode,
            true_target=true_p_boundary_given,
        )

        summaries.append(
            FourParameterRecoverySummary(
                method_name=method.name,
                description=method.description,
                supports_two_spike=supports_two,
                supports_three_spike=supports_three,
                supports_boundary_spike=supports_boundary,
                true_p_spike=true_p_spike,
                true_p_two_given_spike=true_p_two_given,
                true_p_three_given_spike=true_p_three_given,
                true_p_boundary_given_spike=true_p_boundary_given,
                true_p_two_doc=true_p_two_doc,
                true_p_three_doc=true_p_three_doc,
                true_p_boundary_doc=true_p_boundary_doc,
                n_replicates=n_replicates,
                docs_per_replicate=docs_per_replicate,
                mean_hat_p_spike=mean_hat_spike,
                bias_p_spike=mean_hat_spike - true_p_spike,
                mean_abs_bias_p_spike=sum(abs(x - true_p_spike) for x in est_spike_rates) / n_rep,
                rmse_p_spike=math.sqrt(sum((x - true_p_spike) ** 2 for x in est_spike_rates) / n_rep),
                mean_hat_p_two_given_spike=mean_hat_two_given,
                bias_p_two_given_spike=mean_hat_two_given - true_p_two_given,
                mean_abs_bias_p_two_given_spike=sum(abs(x - true_p_two_given) for x in est_two_given_rates) / n_rep,
                rmse_p_two_given_spike=math.sqrt(sum((x - true_p_two_given) ** 2 for x in est_two_given_rates) / n_rep),
                mean_hat_p_three_given_spike=mean_hat_three_given,
                bias_p_three_given_spike=mean_hat_three_given - true_p_three_given,
                mean_abs_bias_p_three_given_spike=sum(abs(x - true_p_three_given) for x in est_three_given_rates) / n_rep,
                rmse_p_three_given_spike=math.sqrt(
                    sum((x - true_p_three_given) ** 2 for x in est_three_given_rates) / n_rep
                ),
                mean_hat_p_boundary_given_spike=mean_hat_boundary_given,
                bias_p_boundary_given_spike=mean_hat_boundary_given - true_p_boundary_given,
                mean_abs_bias_p_boundary_given_spike=sum(abs(x - true_p_boundary_given) for x in est_boundary_given_rates) / n_rep,
                rmse_p_boundary_given_spike=math.sqrt(
                    sum((x - true_p_boundary_given) ** 2 for x in est_boundary_given_rates) / n_rep
                ),
                mean_hat_p_two_doc=mean_hat_two_doc,
                bias_p_two_doc=mean_hat_two_doc - true_p_two_doc,
                mean_abs_bias_p_two_doc=sum(abs(x - true_p_two_doc) for x in est_two_doc_rates) / n_rep,
                rmse_p_two_doc=math.sqrt(sum((x - true_p_two_doc) ** 2 for x in est_two_doc_rates) / n_rep),
                mean_hat_p_three_doc=mean_hat_three_doc,
                bias_p_three_doc=mean_hat_three_doc - true_p_three_doc,
                mean_abs_bias_p_three_doc=sum(abs(x - true_p_three_doc) for x in est_three_doc_rates) / n_rep,
                rmse_p_three_doc=math.sqrt(sum((x - true_p_three_doc) ** 2 for x in est_three_doc_rates) / n_rep),
                mean_hat_p_boundary_doc=mean_hat_boundary_doc,
                bias_p_boundary_doc=mean_hat_boundary_doc - true_p_boundary_doc,
                mean_abs_bias_p_boundary_doc=sum(abs(x - true_p_boundary_doc) for x in est_boundary_doc_rates) / n_rep,
                rmse_p_boundary_doc=math.sqrt(
                    sum((x - true_p_boundary_doc) ** 2 for x in est_boundary_doc_rates) / n_rep
                ),
                sample_target_bias_p_spike=sum((x - y) for x, y in zip(est_spike_rates, true_spike_rates)) / n_rep,
                sample_target_bias_p_two_given_spike=sum((x - y) for x, y in zip(est_two_given_rates, true_two_given_rates)) / n_rep,
                sample_target_bias_p_three_given_spike=sum(
                    (x - y) for x, y in zip(est_three_given_rates, true_three_given_rates)
                ) / n_rep,
                sample_target_bias_p_boundary_given_spike=sum(
                    (x - y) for x, y in zip(est_boundary_given_rates, true_boundary_given_rates)
                ) / n_rep,
                sample_target_bias_p_two_doc=sum((x - y) for x, y in zip(est_two_doc_rates, true_two_doc_rates)) / n_rep,
                sample_target_bias_p_three_doc=sum((x - y) for x, y in zip(est_three_doc_rates, true_three_doc_rates)) / n_rep,
                sample_target_bias_p_boundary_doc=sum(
                    (x - y) for x, y in zip(est_boundary_doc_rates, true_boundary_doc_rates)
                ) / n_rep,
                legacy_weighting_mode=legacy_mode.value,
                weighting_views={
                    mode.value: {
                        "p_spike": spike_views[mode.value],
                        "p_two_doc": two_doc_views[mode.value],
                        "p_three_doc": three_doc_views[mode.value],
                        "p_boundary_doc": boundary_doc_views[mode.value],
                        "p_two_given_spike": two_given_views[mode.value],
                        "p_three_given_spike": three_given_views[mode.value],
                        "p_boundary_given_spike": boundary_given_views[mode.value],
                    }
                    for mode in modes
                },
            )
        )

    return summaries


def default_k_sketch_method_specs(target_max_k: int) -> Tuple[KSketchMethodSpec, ...]:
    """Curated methods for generic-k recovery experiments."""
    kmax = max(2, int(target_max_k))
    return (
        KSketchMethodSpec(
            name="one_pass_oracle",
            description="Oracle baseline: one-pass full document with top-kmax merge-safe sketch.",
            estimator=KSketchEstimator.MERGE_SAFE_TOPK,
            chunker=ChunkerPolicy.FIXED,
            selector=SelectorPolicy.ALL,
            sketch_order=kmax,
            chunk_budget=None,
            fixed_chunk_size=10**9,
        ),
        KSketchMethodSpec(
            name="full_model_aligned",
            description="Full model: aligned adaptive chunking + top-proxy selection + top-kmax merge-safe sketch.",
            estimator=KSketchEstimator.MERGE_SAFE_TOPK,
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            sketch_order=kmax,
            chunk_budget=6,
        ),
        KSketchMethodSpec(
            name="full_model_limited_sketch",
            description="Ablation: same chunking/selection but sketch stores too few order statistics.",
            estimator=KSketchEstimator.MERGE_SAFE_TOPK,
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            sketch_order=max(2, min(3, kmax - 1)),
            chunk_budget=6,
        ),
        KSketchMethodSpec(
            name="naive_majority_same_chunker",
            description="Ablation: naive majority used for all k targets.",
            estimator=KSketchEstimator.NAIVE_MAJORITY,
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            sketch_order=1,
            chunk_budget=6,
        ),
        KSketchMethodSpec(
            name="naive_mean_of_means_same_chunker",
            description="Ablation: unweighted mean-of-means used for all k targets.",
            estimator=KSketchEstimator.NAIVE_MEAN_OF_MEANS,
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            sketch_order=1,
            chunk_budget=6,
        ),
        KSketchMethodSpec(
            name="right_rule_wrong_chunker",
            description="Ablation: merge-safe top-kmax sketch but misspecified chunking + bottom-proxy selection.",
            estimator=KSketchEstimator.MERGE_SAFE_TOPK,
            chunker=ChunkerPolicy.ADAPTIVE_MISSPECIFIED,
            selector=SelectorPolicy.BOTTOM_PROXY,
            sketch_order=kmax,
            chunk_budget=6,
        ),
    )


def run_k_target_recovery_study(
    *,
    distribution: SpikeCountMixtureDistributionSpec,
    target_ks: Sequence[int],
    methods: Optional[Sequence[KSketchMethodSpec]] = None,
    n_replicates: int = 200,
    docs_per_replicate: int = 200,
    seed: int = 0,
    spike_threshold: float = 0.90,
    decision_threshold: float = 0.50,
    weighting_modes: Optional[Sequence[str]] = None,
    legacy_weighting_mode: str = "doc",
) -> List[KTargetRecoverySummary]:
    """
    Recover generic conditional targets P(count>=k | spike) for user-specified `k`.
    """
    if n_replicates <= 0:
        raise ValueError("n_replicates must be >= 1")
    if docs_per_replicate <= 0:
        raise ValueError("docs_per_replicate must be >= 1")
    if len(target_ks) == 0:
        raise ValueError("target_ks must be non-empty")
    if any(int(k) < 2 for k in target_ks):
        raise ValueError("target_ks must be >= 2; k=1 is the conditioning event")
    _validate_spike_count_distribution(distribution)
    modes, legacy_mode = _resolve_weighting(
        weighting_modes=weighting_modes,
        legacy_weighting_mode=legacy_weighting_mode,
    )

    ks = tuple(sorted({int(k) for k in target_ks}))
    max_k = max(ks)
    active_methods = (
        list(methods) if methods is not None else list(default_k_sketch_method_specs(target_max_k=max_k))
    )

    support = [int(x) for x in distribution.spike_count_support]
    weights = [float(x) for x in distribution.spike_count_probs_given_spike]
    denom = float(sum(weights))
    true_p_by_k = {}
    for k in ks:
        numer = sum(w for s, w in zip(support, weights) if s >= k)
        true_p_by_k[k] = numer / denom if denom > 0 else 0.0

    out: List[KTargetRecoverySummary] = []
    for m_idx, method in enumerate(active_methods):
        est_cond_by_k = {k: [] for k in ks}
        true_cond_by_k = {k: [] for k in ks}
        est_cond_by_mode_k: Dict[str, Dict[int, List[float]]] = {
            m.value: {k: [] for k in ks} for m in modes
        }
        true_cond_by_mode_k: Dict[str, Dict[int, List[float]]] = {
            m.value: {k: [] for k in ks} for m in modes
        }

        for rep in range(n_replicates):
            rep_seed = seed + (50_000 * m_idx) + rep
            docs = sample_spike_count_mixture_documents(
                spec=distribution,
                n_docs=docs_per_replicate,
                seed=rep_seed,
            )
            est_counts_by_k = {k: 0.0 for k in ks}
            true_counts_by_k = {k: 0.0 for k in ks}
            est_spike = 0.0
            true_spike = 0.0
            mode_tot_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_spike_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_spike_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_k_weight: Dict[str, Dict[int, float]] = {
                m.value: {k: 0.0 for k in ks} for m in modes
            }
            mode_true_k_weight: Dict[str, Dict[int, float]] = {
                m.value: {k: 0.0 for k in ks} for m in modes
            }
            mode_est_inter_weight: Dict[str, Dict[int, float]] = {
                m.value: {k: 0.0 for k in ks} for m in modes
            }
            mode_true_inter_weight: Dict[str, Dict[int, float]] = {
                m.value: {k: 0.0 for k in ks} for m in modes
            }

            for i, doc in enumerate(docs):
                doc_seed = rep_seed + i
                chunks_all = chunk_document(
                    doc,
                    policy=method.chunker,
                    fixed_chunk_size=method.fixed_chunk_size,
                    min_chunk_size=method.min_chunk_size,
                    max_chunk_size=method.max_chunk_size,
                )
                chunks_kept = select_chunks(
                    chunks_all,
                    selector=method.selector,
                    chunk_budget=method.chunk_budget,
                    rng=random.Random(doc_seed),
                )

                if method.estimator == KSketchEstimator.MERGE_SAFE_TOPK:
                    est_spike_event = aggregate_chunks_topk_event(
                        chunks_kept,
                        target_k=1,
                        sketch_order=method.sketch_order,
                        merge_order=method.merge_order,
                        spike_threshold=spike_threshold,
                        seed=doc_seed,
                    )
                elif method.estimator == KSketchEstimator.NAIVE_MAJORITY:
                    est_spike_event = aggregate_chunks(
                        chunks_kept,
                        aggregator=AggregatorPolicy.NAIVE_MAJORITY,
                        merge_order=method.merge_order,
                        spike_threshold=spike_threshold,
                        seed=doc_seed,
                    )
                elif method.estimator == KSketchEstimator.NAIVE_MEAN_OF_MEANS:
                    est_spike_event = aggregate_chunks(
                        chunks_kept,
                        aggregator=AggregatorPolicy.NAIVE_MEAN_OF_MEANS,
                        merge_order=method.merge_order,
                        spike_threshold=spike_threshold,
                        seed=doc_seed,
                    )
                else:
                    raise ValueError(f"Unsupported estimator: {method.estimator!r}")

                est_spike += 1.0 if est_spike_event >= decision_threshold else 0.0
                est_spike_bin = 1.0 if est_spike_event >= decision_threshold else 0.0

                count_true = true_spike_count(doc.token_scores, spike_threshold=spike_threshold)
                true_spike += 1.0 if count_true >= 1 else 0.0
                true_spike_bin = 1.0 if count_true >= 1 else 0.0
                for mode in modes:
                    w = _doc_weight(doc=doc, all_chunks=chunks_all, mode=mode)
                    mode_tot_weight[mode.value] += w
                    mode_est_spike_weight[mode.value] += w * est_spike_bin
                    mode_true_spike_weight[mode.value] += w * true_spike_bin

                for k in ks:
                    if method.estimator == KSketchEstimator.MERGE_SAFE_TOPK:
                        est_event = aggregate_chunks_topk_event(
                            chunks_kept,
                            target_k=k,
                            sketch_order=method.sketch_order,
                            merge_order=method.merge_order,
                            spike_threshold=spike_threshold,
                            seed=doc_seed,
                        )
                    elif method.estimator == KSketchEstimator.NAIVE_MAJORITY:
                        est_event = aggregate_chunks(
                            chunks_kept,
                            aggregator=AggregatorPolicy.NAIVE_MAJORITY,
                            merge_order=method.merge_order,
                            spike_threshold=spike_threshold,
                            seed=doc_seed,
                        )
                    elif method.estimator == KSketchEstimator.NAIVE_MEAN_OF_MEANS:
                        est_event = aggregate_chunks(
                            chunks_kept,
                            aggregator=AggregatorPolicy.NAIVE_MEAN_OF_MEANS,
                            merge_order=method.merge_order,
                            spike_threshold=spike_threshold,
                            seed=doc_seed,
                        )
                    else:
                        raise ValueError(f"Unsupported estimator: {method.estimator!r}")
                    est_counts_by_k[k] += 1.0 if est_event >= decision_threshold else 0.0
                    true_counts_by_k[k] += 1.0 if count_true >= k else 0.0
                    est_event_bin = 1.0 if est_event >= decision_threshold else 0.0
                    true_event_bin = 1.0 if count_true >= k else 0.0
                    for mode in modes:
                        w = _doc_weight(doc=doc, all_chunks=chunks_all, mode=mode)
                        mode_est_k_weight[mode.value][k] += w * est_event_bin
                        mode_true_k_weight[mode.value][k] += w * true_event_bin
                        mode_est_inter_weight[mode.value][k] += w * (
                            1.0 if (est_event_bin >= 0.5 and est_spike_bin >= 0.5) else 0.0
                        )
                        mode_true_inter_weight[mode.value][k] += w * (
                            1.0 if (true_event_bin >= 0.5 and true_spike_bin >= 0.5) else 0.0
                        )

            n = float(len(docs))
            est_spike_rate = est_spike / n
            true_spike_rate = true_spike / n
            for k in ks:
                est_doc_rate = est_counts_by_k[k] / n
                true_doc_rate = true_counts_by_k[k] / n
                est_cond = est_doc_rate / est_spike_rate if est_spike_rate > 0 else 0.0
                true_cond = true_doc_rate / true_spike_rate if true_spike_rate > 0 else 0.0
                est_cond_by_k[k].append(est_cond)
                true_cond_by_k[k].append(true_cond)
                for mode in modes:
                    mode_key = mode.value
                    est_cond_w = (
                        mode_est_inter_weight[mode_key][k] / mode_est_spike_weight[mode_key]
                        if mode_est_spike_weight[mode_key] > 0
                        else 0.0
                    )
                    true_cond_w = (
                        mode_true_inter_weight[mode_key][k] / mode_true_spike_weight[mode_key]
                        if mode_true_spike_weight[mode_key] > 0
                        else 0.0
                    )
                    est_cond_by_mode_k[mode_key][k].append(float(est_cond_w))
                    true_cond_by_mode_k[mode_key][k].append(float(true_cond_w))

        n_rep = float(n_replicates)
        for k in ks:
            estimates = est_cond_by_k[k]
            true_target = true_p_by_k[k]
            mean_hat = sum(estimates) / n_rep
            out.append(
                KTargetRecoverySummary(
                    method_name=method.name,
                    method_description=method.description,
                    estimator=method.estimator,
                    sketch_order=method.sketch_order,
                    target_k=k,
                    supports_target=method.sketch_order >= k,
                    true_p_at_least_k_given_spike=true_target,
                    n_replicates=n_replicates,
                    docs_per_replicate=docs_per_replicate,
                    mean_hat_p_at_least_k_given_spike=mean_hat,
                    bias=mean_hat - true_target,
                    mean_abs_bias=sum(abs(x - true_target) for x in estimates) / n_rep,
                    rmse=math.sqrt(sum((x - true_target) ** 2 for x in estimates) / n_rep),
                    sample_target_bias=sum((x - y) for x, y in zip(estimates, true_cond_by_k[k])) / n_rep,
                    legacy_weighting_mode=legacy_mode.value,
                    weighting_views=build_weighting_views_from_replicates(
                        estimates_by_mode={mode.value: est_cond_by_mode_k[mode.value][k] for mode in modes},
                        sample_targets_by_mode={mode.value: true_cond_by_mode_k[mode.value][k] for mode in modes},
                        true_target=true_target,
                    ),
                )
            )
    return out


def _true_p_at_least_k_given_spike(
    distribution: SpikeCountMixtureDistributionSpec,
    *,
    target_k: int,
) -> float:
    support = [int(x) for x in distribution.spike_count_support]
    weights = [float(x) for x in distribution.spike_count_probs_given_spike]
    denom = float(sum(weights))
    if denom <= 0:
        return 0.0
    k = int(target_k)
    return sum(w for s, w in zip(support, weights) if s >= k) / denom


def _build_chunk_quality_methods(
    *,
    target_k: int,
    sketch_order: int,
    chunk_sizes: Sequence[int],
    chunk_budgets: Sequence[int],
    chunker: ChunkerPolicy,
    selector: SelectorPolicy,
    include_references: bool,
) -> List[KSketchMethodSpec]:
    methods: List[KSketchMethodSpec] = []
    m = int(sketch_order)
    sizes = tuple(sorted({int(s) for s in chunk_sizes}))
    budgets = tuple(sorted({int(b) for b in chunk_budgets}))

    if include_references:
        methods.append(
            KSketchMethodSpec(
                name="one_pass_reference",
                description=f"one-pass oracle top-{m}",
                estimator=KSketchEstimator.MERGE_SAFE_TOPK,
                chunker=ChunkerPolicy.FIXED,
                selector=SelectorPolicy.ALL,
                sketch_order=m,
                chunk_budget=None,
                fixed_chunk_size=10**9,
                min_chunk_size=1,
                max_chunk_size=1,
            )
        )
        methods.append(
            KSketchMethodSpec(
                name="perfect_token_leaves_all",
                description=f"perfect token leaves (size=1, keep all), top-{m}",
                estimator=KSketchEstimator.MERGE_SAFE_TOPK,
                chunker=ChunkerPolicy.FIXED,
                selector=SelectorPolicy.ALL,
                sketch_order=m,
                chunk_budget=None,
                fixed_chunk_size=1,
                min_chunk_size=1,
                max_chunk_size=1,
            )
        )

    for b in budgets:
        for s in sizes:
            if chunker == ChunkerPolicy.FIXED:
                methods.append(
                    KSketchMethodSpec(
                        name=f"grid_fixed_s{s}_b{b}",
                        description=f"fixed leaves size={s}, selector={selector.value}, budget={b}, top-{m}",
                        estimator=KSketchEstimator.MERGE_SAFE_TOPK,
                        chunker=ChunkerPolicy.FIXED,
                        selector=selector,
                        sketch_order=m,
                        chunk_budget=b,
                        fixed_chunk_size=s,
                        min_chunk_size=1,
                        max_chunk_size=max(1, s),
                    )
                )
            elif chunker in (ChunkerPolicy.ADAPTIVE_ALIGNED, ChunkerPolicy.ADAPTIVE_MISSPECIFIED):
                lo = max(1, s // 2)
                hi = max(lo, s)
                prefix = "aligned" if chunker == ChunkerPolicy.ADAPTIVE_ALIGNED else "misspecified"
                methods.append(
                    KSketchMethodSpec(
                        name=f"grid_{prefix}_s{s}_b{b}",
                        description=(
                            f"{chunker.value} leaves range=[{lo},{hi}], "
                            f"selector={selector.value}, budget={b}, top-{m}"
                        ),
                        estimator=KSketchEstimator.MERGE_SAFE_TOPK,
                        chunker=chunker,
                        selector=selector,
                        sketch_order=m,
                        chunk_budget=b,
                        fixed_chunk_size=hi,
                        min_chunk_size=lo,
                        max_chunk_size=hi,
                    )
                )
            else:
                raise ValueError(f"Unsupported chunker for quality sweep: {chunker!r}")
    return methods


def run_chunk_quality_sweep(
    *,
    distribution: SpikeCountMixtureDistributionSpec,
    target_k: int,
    sketch_order: Optional[int] = None,
    chunk_sizes: Sequence[int] = (1, 2, 4, 8, 16),
    chunk_budgets: Sequence[int] = (1, 2, 3, 4, 6, 8),
    chunker: ChunkerPolicy = ChunkerPolicy.FIXED,
    selector: SelectorPolicy = SelectorPolicy.TOP_PROXY,
    n_replicates: int = 120,
    docs_per_replicate: int = 160,
    seed: int = 0,
    spike_threshold: float = 0.90,
    decision_threshold: float = 0.50,
    include_references: bool = True,
    weighting_modes: Optional[Sequence[str]] = None,
    legacy_weighting_mode: str = "doc",
) -> List[ChunkQualitySweepSummary]:
    """
    Sweep leaf granularity / chunk budget and report both bias and chunk-quality diagnostics.

    Main target:
      P(count>=target_k | spike)

    Additional diagnostics quantify chunking quality directly:
      - target-capture rate: among docs with >=k spikes, kept chunks still contain >=k spikes
      - spike-capture rate: among docs with >=1 spike, kept chunks contain >=1 spike
      - spike-token recall: retained spike tokens / true spike tokens
      - spike-token isolation: retained spike tokens that are in singleton leaves / true spike tokens
    """
    if n_replicates <= 0:
        raise ValueError("n_replicates must be >= 1")
    if docs_per_replicate <= 0:
        raise ValueError("docs_per_replicate must be >= 1")
    _validate_spike_count_distribution(distribution)
    modes, legacy_mode = _resolve_weighting(
        weighting_modes=weighting_modes,
        legacy_weighting_mode=legacy_weighting_mode,
    )

    k = int(target_k)
    if k < 2:
        raise ValueError("target_k must be >= 2")
    m = int(sketch_order) if sketch_order is not None else k
    if m < 1:
        raise ValueError("sketch_order must be >= 1")

    sizes = tuple(sorted({int(s) for s in chunk_sizes}))
    if len(sizes) == 0:
        raise ValueError("chunk_sizes must be non-empty")
    if any(s <= 0 for s in sizes):
        raise ValueError("chunk_sizes must be >= 1")

    budgets = tuple(sorted({int(b) for b in chunk_budgets}))
    if len(budgets) == 0:
        raise ValueError("chunk_budgets must be non-empty")
    if any(b <= 0 for b in budgets):
        raise ValueError("chunk_budgets must be >= 1")

    true_p_k = _true_p_at_least_k_given_spike(distribution, target_k=k)
    methods = _build_chunk_quality_methods(
        target_k=k,
        sketch_order=m,
        chunk_sizes=sizes,
        chunk_budgets=budgets,
        chunker=chunker,
        selector=selector,
        include_references=include_references,
    )

    out: List[ChunkQualitySweepSummary] = []
    for m_idx, method in enumerate(methods):
        est_cond: List[float] = []
        true_cond: List[float] = []
        est_cond_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        true_cond_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        target_capture: List[float] = []
        spike_capture: List[float] = []
        spike_recall: List[float] = []
        spike_isolation: List[float] = []
        chunks_total: List[float] = []
        chunks_kept: List[float] = []

        for rep in range(n_replicates):
            rep_seed = seed + (70_000 * m_idx) + rep
            docs = sample_spike_count_mixture_documents(
                spec=distribution,
                n_docs=docs_per_replicate,
                seed=rep_seed,
            )

            est_k_count = 0.0
            true_k_count = 0.0
            est_spike_count = 0.0
            true_spike_count_docs = 0.0
            mode_est_spike_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_k_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_inter_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_spike_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_k_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_inter_weight: Dict[str, float] = {m.value: 0.0 for m in modes}

            true_target_docs = 0
            captured_target_docs = 0
            true_spike_docs = 0
            captured_spike_docs = 0
            recall_sum = 0.0
            isolation_sum = 0.0
            total_chunks_sum = 0.0
            kept_chunks_sum = 0.0

            for i, doc in enumerate(docs):
                doc_seed = rep_seed + i
                all_chunks = chunk_document(
                    doc,
                    policy=method.chunker,
                    fixed_chunk_size=method.fixed_chunk_size,
                    min_chunk_size=method.min_chunk_size,
                    max_chunk_size=method.max_chunk_size,
                )
                kept_chunks = select_chunks(
                    all_chunks,
                    selector=method.selector,
                    chunk_budget=method.chunk_budget,
                    rng=random.Random(doc_seed),
                )
                total_chunks_sum += float(len(all_chunks))
                kept_chunks_sum += float(len(kept_chunks))

                est_spike_event = aggregate_chunks_topk_event(
                    kept_chunks,
                    target_k=1,
                    sketch_order=method.sketch_order,
                    merge_order=method.merge_order,
                    spike_threshold=spike_threshold,
                    seed=doc_seed,
                )
                est_k_event = aggregate_chunks_topk_event(
                    kept_chunks,
                    target_k=k,
                    sketch_order=method.sketch_order,
                    merge_order=method.merge_order,
                    spike_threshold=spike_threshold,
                    seed=doc_seed,
                )
                est_spike_count += 1.0 if est_spike_event >= decision_threshold else 0.0
                est_k_count += 1.0 if est_k_event >= decision_threshold else 0.0
                est_spike_bin = 1.0 if est_spike_event >= decision_threshold else 0.0
                est_k_bin = 1.0 if est_k_event >= decision_threshold else 0.0

                n_spikes_true = true_spike_count(doc.token_scores, spike_threshold=spike_threshold)
                true_spike_count_docs += 1.0 if n_spikes_true >= 1 else 0.0
                true_k_count += 1.0 if n_spikes_true >= k else 0.0
                true_spike_bin = 1.0 if n_spikes_true >= 1 else 0.0
                true_k_bin = 1.0 if n_spikes_true >= k else 0.0
                for mode in modes:
                    w = _doc_weight(doc=doc, all_chunks=all_chunks, mode=mode)
                    mode_est_spike_weight[mode.value] += w * est_spike_bin
                    mode_est_k_weight[mode.value] += w * est_k_bin
                    mode_est_inter_weight[mode.value] += w * (
                        1.0 if (est_spike_bin >= 0.5 and est_k_bin >= 0.5) else 0.0
                    )
                    mode_true_spike_weight[mode.value] += w * true_spike_bin
                    mode_true_k_weight[mode.value] += w * true_k_bin
                    mode_true_inter_weight[mode.value] += w * (
                        1.0 if (true_spike_bin >= 0.5 and true_k_bin >= 0.5) else 0.0
                    )

                n_spikes_kept = 0
                n_spikes_isolated = 0
                for chunk in kept_chunks:
                    chunk_spikes = sum(1 for v in chunk.values if float(v) >= spike_threshold)
                    n_spikes_kept += chunk_spikes
                    if chunk.count == 1:
                        n_spikes_isolated += chunk_spikes

                if n_spikes_true >= 1:
                    true_spike_docs += 1
                    captured_spike_docs += 1 if n_spikes_kept >= 1 else 0
                    recall_sum += min(1.0, float(n_spikes_kept) / float(n_spikes_true))
                    isolation_sum += min(1.0, float(n_spikes_isolated) / float(n_spikes_true))
                if n_spikes_true >= k:
                    true_target_docs += 1
                    captured_target_docs += 1 if n_spikes_kept >= k else 0

            n_docs = float(len(docs))
            est_spike_rate = est_spike_count / n_docs
            true_spike_rate = true_spike_count_docs / n_docs
            est_k_rate = est_k_count / n_docs
            true_k_rate = true_k_count / n_docs
            est_cond.append(est_k_rate / est_spike_rate if est_spike_rate > 0 else 0.0)
            true_cond.append(true_k_rate / true_spike_rate if true_spike_rate > 0 else 0.0)
            for mode in modes:
                mode_key = mode.value
                est_cond_mode = (
                    mode_est_inter_weight[mode_key] / mode_est_spike_weight[mode_key]
                    if mode_est_spike_weight[mode_key] > 0
                    else 0.0
                )
                true_cond_mode = (
                    mode_true_inter_weight[mode_key] / mode_true_spike_weight[mode_key]
                    if mode_true_spike_weight[mode_key] > 0
                    else 0.0
                )
                est_cond_by_mode[mode_key].append(float(est_cond_mode))
                true_cond_by_mode[mode_key].append(float(true_cond_mode))

            target_capture.append(
                float(captured_target_docs) / float(true_target_docs) if true_target_docs > 0 else 0.0
            )
            spike_capture.append(
                float(captured_spike_docs) / float(true_spike_docs) if true_spike_docs > 0 else 0.0
            )
            spike_recall.append(
                recall_sum / float(true_spike_docs) if true_spike_docs > 0 else 0.0
            )
            spike_isolation.append(
                isolation_sum / float(true_spike_docs) if true_spike_docs > 0 else 0.0
            )
            chunks_total.append(total_chunks_sum / n_docs)
            chunks_kept.append(kept_chunks_sum / n_docs)

        n_rep = float(n_replicates)
        mean_hat = sum(est_cond) / n_rep
        out.append(
            ChunkQualitySweepSummary(
                method_name=method.name,
                method_description=method.description,
                chunker=method.chunker,
                selector=method.selector,
                sketch_order=method.sketch_order,
                target_k=k,
                chunk_budget=method.chunk_budget,
                fixed_chunk_size=method.fixed_chunk_size,
                min_chunk_size=method.min_chunk_size,
                max_chunk_size=method.max_chunk_size,
                supports_target=method.sketch_order >= k,
                true_p_at_least_k_given_spike=true_p_k,
                n_replicates=n_replicates,
                docs_per_replicate=docs_per_replicate,
                mean_hat_p_at_least_k_given_spike=mean_hat,
                bias=mean_hat - true_p_k,
                mean_abs_bias=sum(abs(x - true_p_k) for x in est_cond) / n_rep,
                rmse=math.sqrt(sum((x - true_p_k) ** 2 for x in est_cond) / n_rep),
                sample_target_bias=sum((x - y) for x, y in zip(est_cond, true_cond)) / n_rep,
                mean_target_capture_rate=sum(target_capture) / n_rep,
                mean_spike_capture_rate=sum(spike_capture) / n_rep,
                mean_spike_token_recall=sum(spike_recall) / n_rep,
                mean_spike_token_isolation=sum(spike_isolation) / n_rep,
                mean_chunks_total=sum(chunks_total) / n_rep,
                mean_chunks_kept=sum(chunks_kept) / n_rep,
                legacy_weighting_mode=legacy_mode.value,
                weighting_views=build_weighting_views_from_replicates(
                    estimates_by_mode=est_cond_by_mode,
                    sample_targets_by_mode=true_cond_by_mode,
                    true_target=true_p_k,
                ),
            )
        )
    return out


def _wilson_interval(successes: int, n: int, z: float) -> Tuple[float, float]:
    """Wilson score interval for a Bernoulli proportion."""
    nn = int(n)
    if nn <= 0:
        return (0.0, 1.0)
    ss = max(0, min(int(successes), nn))
    p = float(ss) / float(nn)
    z2 = z * z
    denom = 1.0 + (z2 / float(nn))
    center = (p + z2 / (2.0 * float(nn))) / denom
    half = (
        z
        * math.sqrt((p * (1.0 - p) + z2 / (4.0 * float(nn))) / float(nn))
        / denom
    )
    return (max(0.0, center - half), min(1.0, center + half))


def run_chunk_quality_coverage_sweep(
    *,
    distribution: SpikeCountMixtureDistributionSpec,
    target_k: int,
    sketch_order: Optional[int] = None,
    chunk_sizes: Sequence[int] = (1, 2, 4, 8, 16),
    chunk_budgets: Sequence[int] = (1, 2, 3, 4, 6, 8),
    chunker: ChunkerPolicy = ChunkerPolicy.FIXED,
    selector: SelectorPolicy = SelectorPolicy.TOP_PROXY,
    n_replicates: int = 120,
    docs_per_replicate: int = 160,
    seed: int = 0,
    spike_threshold: float = 0.90,
    decision_threshold: float = 0.50,
    include_references: bool = True,
    ci_level: float = 0.95,
    weighting_modes: Optional[Sequence[str]] = None,
    legacy_weighting_mode: str = "doc",
) -> List[ChunkQualityCoverageSummary]:
    """
    Coverage-focused variant of chunk-quality sweep.

    For each replicate, builds a CI for p_hat = P_hat(count>=k | spike) using a
    Wilson interval on the conditional binomial among estimated spike docs.
    """
    if n_replicates <= 0:
        raise ValueError("n_replicates must be >= 1")
    if docs_per_replicate <= 0:
        raise ValueError("docs_per_replicate must be >= 1")
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be in (0, 1)")
    _validate_spike_count_distribution(distribution)
    modes, legacy_mode = _resolve_weighting(
        weighting_modes=weighting_modes,
        legacy_weighting_mode=legacy_weighting_mode,
    )

    k = int(target_k)
    if k < 2:
        raise ValueError("target_k must be >= 2")
    m = int(sketch_order) if sketch_order is not None else k
    if m < 1:
        raise ValueError("sketch_order must be >= 1")

    sizes = tuple(sorted({int(s) for s in chunk_sizes}))
    if len(sizes) == 0:
        raise ValueError("chunk_sizes must be non-empty")
    if any(s <= 0 for s in sizes):
        raise ValueError("chunk_sizes must be >= 1")

    budgets = tuple(sorted({int(b) for b in chunk_budgets}))
    if len(budgets) == 0:
        raise ValueError("chunk_budgets must be non-empty")
    if any(b <= 0 for b in budgets):
        raise ValueError("chunk_budgets must be >= 1")

    alpha = 1.0 - float(ci_level)
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)

    true_p_k = _true_p_at_least_k_given_spike(distribution, target_k=k)
    methods = _build_chunk_quality_methods(
        target_k=k,
        sketch_order=m,
        chunk_sizes=sizes,
        chunk_budgets=budgets,
        chunker=chunker,
        selector=selector,
        include_references=include_references,
    )

    out: List[ChunkQualityCoverageSummary] = []
    for m_idx, method in enumerate(methods):
        estimates: List[float] = []
        ci_low: List[float] = []
        ci_high: List[float] = []
        covered: List[float] = []
        estimates_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}
        true_by_mode: Dict[str, List[float]] = {m.value: [] for m in modes}

        for rep in range(n_replicates):
            rep_seed = seed + (90_000 * m_idx) + rep
            docs = sample_spike_count_mixture_documents(
                spec=distribution,
                n_docs=docs_per_replicate,
                seed=rep_seed,
            )

            est_spike_docs = 0
            est_target_docs = 0
            mode_est_spike_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_est_inter_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_spike_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            mode_true_inter_weight: Dict[str, float] = {m.value: 0.0 for m in modes}
            for i, doc in enumerate(docs):
                doc_seed = rep_seed + i
                all_chunks = chunk_document(
                    doc,
                    policy=method.chunker,
                    fixed_chunk_size=method.fixed_chunk_size,
                    min_chunk_size=method.min_chunk_size,
                    max_chunk_size=method.max_chunk_size,
                )
                kept_chunks = select_chunks(
                    all_chunks,
                    selector=method.selector,
                    chunk_budget=method.chunk_budget,
                    rng=random.Random(doc_seed),
                )

                est_spike_event = aggregate_chunks_topk_event(
                    kept_chunks,
                    target_k=1,
                    sketch_order=method.sketch_order,
                    merge_order=method.merge_order,
                    spike_threshold=spike_threshold,
                    seed=doc_seed,
                )
                est_k_event = aggregate_chunks_topk_event(
                    kept_chunks,
                    target_k=k,
                    sketch_order=method.sketch_order,
                    merge_order=method.merge_order,
                    spike_threshold=spike_threshold,
                    seed=doc_seed,
                )
                is_spike = int(est_spike_event >= decision_threshold)
                is_target = int(est_k_event >= decision_threshold)
                est_spike_docs += is_spike
                est_target_docs += is_target
                n_spikes_true = true_spike_count(doc.token_scores, spike_threshold=spike_threshold)
                true_spike_event = 1 if n_spikes_true >= 1 else 0
                true_target_event = 1 if n_spikes_true >= k else 0
                for mode in modes:
                    w = _doc_weight(doc=doc, all_chunks=all_chunks, mode=mode)
                    mode_est_spike_weight[mode.value] += w * float(is_spike)
                    mode_est_inter_weight[mode.value] += w * float(1 if (is_spike and is_target) else 0)
                    mode_true_spike_weight[mode.value] += w * float(true_spike_event)
                    mode_true_inter_weight[mode.value] += w * float(1 if (true_spike_event and true_target_event) else 0)

            if est_spike_docs <= 0:
                p_hat = 0.0
                lo, hi = (0.0, 1.0)
            else:
                successes = min(est_target_docs, est_spike_docs)
                p_hat = float(successes) / float(est_spike_docs)
                lo, hi = _wilson_interval(successes, est_spike_docs, z)

            estimates.append(p_hat)
            ci_low.append(lo)
            ci_high.append(hi)
            covered.append(1.0 if (lo <= true_p_k <= hi) else 0.0)
            for mode in modes:
                mode_key = mode.value
                est_mode = (
                    mode_est_inter_weight[mode_key] / mode_est_spike_weight[mode_key]
                    if mode_est_spike_weight[mode_key] > 0
                    else 0.0
                )
                true_mode = (
                    mode_true_inter_weight[mode_key] / mode_true_spike_weight[mode_key]
                    if mode_true_spike_weight[mode_key] > 0
                    else 0.0
                )
                estimates_by_mode[mode_key].append(float(est_mode))
                true_by_mode[mode_key].append(float(true_mode))

        n_rep = float(n_replicates)
        mean_hat = sum(estimates) / n_rep
        out.append(
            ChunkQualityCoverageSummary(
                method_name=method.name,
                method_description=method.description,
                chunker=method.chunker,
                selector=method.selector,
                sketch_order=method.sketch_order,
                target_k=k,
                chunk_budget=method.chunk_budget,
                fixed_chunk_size=method.fixed_chunk_size,
                min_chunk_size=method.min_chunk_size,
                max_chunk_size=method.max_chunk_size,
                supports_target=method.sketch_order >= k,
                true_p_at_least_k_given_spike=true_p_k,
                n_replicates=n_replicates,
                docs_per_replicate=docs_per_replicate,
                ci_level=float(ci_level),
                mean_hat_p_at_least_k_given_spike=mean_hat,
                bias=mean_hat - true_p_k,
                mean_abs_bias=sum(abs(x - true_p_k) for x in estimates) / n_rep,
                rmse=math.sqrt(sum((x - true_p_k) ** 2 for x in estimates) / n_rep),
                empirical_coverage=sum(covered) / n_rep,
                mean_ci_width=sum((hi - lo) for lo, hi in zip(ci_low, ci_high)) / n_rep,
                mean_ci_low=sum(ci_low) / n_rep,
                mean_ci_high=sum(ci_high) / n_rep,
                legacy_weighting_mode=legacy_mode.value,
                weighting_views=build_weighting_views_from_replicates(
                    estimates_by_mode=estimates_by_mode,
                    sample_targets_by_mode=true_by_mode,
                    true_target=true_p_k,
                ),
            )
        )
    return out


def sketch_insufficiency_counterexample(
    *,
    sketch_order: int,
    target_k: int,
    n_tokens: int = 32,
    boundary_span_tokens: int = 4,
    spike_threshold: float = 0.90,
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
    """
    Construct two documents with identical top-m sketch but different target event.

    Returns `(doc_a_scores, doc_b_scores, shared_topm_signature)`.
    Preconditions: `target_k = sketch_order + 1`.
    """
    m = int(sketch_order)
    k = int(target_k)
    if m < 1:
        raise ValueError("sketch_order must be >= 1")
    if k != m + 1:
        raise ValueError("counterexample requires target_k = sketch_order + 1")
    if n_tokens < k:
        raise ValueError("n_tokens must be >= target_k")

    doc_a = generate_exact_spike_count_document(
        n_spikes=m,
        n_tokens=n_tokens,
        boundary_span_tokens=boundary_span_tokens,
        force_boundary_spike=False,
        seed=101,
    )
    doc_b = generate_exact_spike_count_document(
        n_spikes=k,
        n_tokens=n_tokens,
        boundary_span_tokens=boundary_span_tokens,
        force_boundary_spike=False,
        seed=202,
    )
    topm = tuple(sorted(doc_a.token_scores, reverse=True)[:m])
    # force identical top-m signature while differing at order m+1
    b_vals = list(doc_b.token_scores)
    top_idx = sorted(range(len(b_vals)), key=lambda i: b_vals[i], reverse=True)[:m]
    for i, idx in enumerate(top_idx):
        b_vals[idx] = topm[i]
    doc_b_fixed = tuple(float(v) for v in b_vals)
    # ensure target differs
    assert true_spike_count(doc_a.token_scores, spike_threshold=spike_threshold) == m
    assert true_spike_count(doc_b_fixed, spike_threshold=spike_threshold) >= k
    return (tuple(doc_a.token_scores), doc_b_fixed, topm)


def default_generalization_stress_scenarios() -> Tuple[GeneralizationStressScenario, ...]:
    """Curated stress scenarios for variable-length and adversarial robustness."""
    return (
        GeneralizationStressScenario(
            name="baseline_balanced_fixed",
            description="Reference fixed-length setting used for baseline calibration.",
            distribution=SpikeMixtureDistributionSpec(
                p_spike_doc=0.55,
                p_boundary_given_spike=0.50,
                p_two_spikes_given_spike=0.20,
                n_tokens=32,
                proxy_noise=0.08,
                boundary_span_tokens=4,
            ),
        ),
        GeneralizationStressScenario(
            name="variable_length_balanced",
            description="Balanced spike mix with high length variability to stress adaptive chunking.",
            distribution=SpikeMixtureDistributionSpec(
                p_spike_doc=0.55,
                p_boundary_given_spike=0.50,
                p_two_spikes_given_spike=0.20,
                n_tokens=32,
                proxy_noise=0.08,
                boundary_span_tokens=2,
                token_length_support=(4, 8, 16, 32, 64, 128),
                token_length_probs=(0.25, 0.25, 0.20, 0.15, 0.10, 0.05),
            ),
        ),
        GeneralizationStressScenario(
            name="boundary_adversarial_concentrated",
            description="Adversarial concentrated DGP: short docs + high boundary mass + noisy proxy.",
            distribution=SpikeMixtureDistributionSpec(
                p_spike_doc=0.70,
                p_boundary_given_spike=0.90,
                p_two_spikes_given_spike=0.05,
                n_tokens=16,
                proxy_noise=0.18,
                boundary_span_tokens=1,
                token_length_support=(4, 5, 6, 7, 8, 64),
                token_length_probs=(0.22, 0.22, 0.22, 0.16, 0.13, 0.05),
            ),
        ),
        GeneralizationStressScenario(
            name="hard_noncorner_adversarial",
            description=(
                "Adversarial but non-corner DGP: variable length + noisy proxy + "
                "moderate spike/boundary/two-spike rates."
            ),
            distribution=SpikeMixtureDistributionSpec(
                p_spike_doc=0.62,
                p_boundary_given_spike=0.42,
                p_two_spikes_given_spike=0.28,
                n_tokens=33,
                proxy_noise=0.20,
                boundary_span_tokens=2,
                token_length_support=(5, 9, 17, 33, 65, 129),
                token_length_probs=(0.24, 0.22, 0.18, 0.16, 0.12, 0.08),
            ),
        ),
        GeneralizationStressScenario(
            name="multi_spike_noncorner",
            description=(
                "Non-corner multi-spike-heavy DGP: large mass on >=2 spikes with many "
                "documents containing 3+ spikes."
            ),
            distribution=SpikeMixtureDistributionSpec(
                p_spike_doc=0.68,
                p_boundary_given_spike=0.20,
                p_two_spikes_given_spike=0.65,
                p_multi_given_two_spikes=0.75,
                n_tokens=40,
                proxy_noise=0.16,
                boundary_span_tokens=3,
                token_length_support=(8, 16, 24, 40, 64, 96),
                token_length_probs=(0.20, 0.24, 0.20, 0.16, 0.12, 0.08),
            ),
        ),
        GeneralizationStressScenario(
            name="long_tail_interior_shift",
            description="Long-document-heavy setting with mostly interior spikes and moderate noise.",
            distribution=SpikeMixtureDistributionSpec(
                p_spike_doc=0.55,
                p_boundary_given_spike=0.05,
                p_two_spikes_given_spike=0.35,
                n_tokens=64,
                proxy_noise=0.12,
                boundary_span_tokens=4,
                token_length_support=(16, 32, 64, 128, 256),
                token_length_probs=(0.10, 0.20, 0.30, 0.25, 0.15),
            ),
        ),
    )


def default_nonlanguage_chunk_quality_scenarios() -> Tuple[NonLanguageScenario, ...]:
    """
    Curated non-language analogs for chunk-quality experiments.

    These scenarios reuse the same mergeable-sketch math in domains where
    tokens are generic time/space bins rather than words.
    """
    return (
        NonLanguageScenario(
            name="icu_alarm_stream",
            intuition=(
                "ICU vital-sign timeline: spikes are dangerous instability windows; "
                "proxy is a noisy triage score."
            ),
            distribution=SpikeCountMixtureDistributionSpec(
                p_spike_doc=0.56,
                p_boundary_given_spike=0.48,
                spike_count_support=(1, 2, 3, 4, 5),
                spike_count_probs_given_spike=(0.30, 0.25, 0.20, 0.15, 0.10),
                n_tokens=40,
                proxy_noise=0.16,
                boundary_span_tokens=6,
                token_length_support=(24, 32, 48, 64),
                token_length_probs=(0.25, 0.35, 0.25, 0.15),
            ),
        ),
        NonLanguageScenario(
            name="network_intrusion_bursts",
            intuition=(
                "Network-flow windows: spikes are attack bursts; proxy is anomaly score "
                "with heavy noise and variable sequence length."
            ),
            distribution=SpikeCountMixtureDistributionSpec(
                p_spike_doc=0.46,
                p_boundary_given_spike=0.28,
                spike_count_support=(1, 2, 3, 4, 5),
                spike_count_probs_given_spike=(0.12, 0.18, 0.20, 0.25, 0.25),
                n_tokens=72,
                proxy_noise=0.22,
                boundary_span_tokens=4,
                token_length_support=(48, 64, 96, 128),
                token_length_probs=(0.20, 0.35, 0.30, 0.15),
            ),
        ),
        NonLanguageScenario(
            name="manufacturing_defect_line",
            intuition=(
                "Manufacturing sensor trace: spikes are defect events; proxy is an inline "
                "quality heuristic with moderate noise."
            ),
            distribution=SpikeCountMixtureDistributionSpec(
                p_spike_doc=0.68,
                p_boundary_given_spike=0.18,
                spike_count_support=(1, 2, 3, 4, 5),
                spike_count_probs_given_spike=(0.45, 0.30, 0.15, 0.07, 0.03),
                n_tokens=36,
                proxy_noise=0.10,
                boundary_span_tokens=3,
                token_length_support=(24, 36, 48),
                token_length_probs=(0.30, 0.50, 0.20),
            ),
        ),
        NonLanguageScenario(
            name="ecg_arrhythmia_monitor",
            intuition=(
                "ECG rhythm windows: spikes are arrhythmia beats; proxy is a lightweight "
                "beat-level risk score."
            ),
            distribution=SpikeCountMixtureDistributionSpec(
                p_spike_doc=0.42,
                p_boundary_given_spike=0.52,
                spike_count_support=(1, 2, 3, 4, 5),
                spike_count_probs_given_spike=(0.20, 0.25, 0.25, 0.20, 0.10),
                n_tokens=32,
                proxy_noise=0.14,
                boundary_span_tokens=5,
                token_length_support=(20, 28, 32, 44),
                token_length_probs=(0.20, 0.30, 0.35, 0.15),
            ),
        ),
    )


def run_three_parameter_generalization_sweep(
    *,
    scenarios: Optional[Sequence[GeneralizationStressScenario]] = None,
    methods: Optional[Sequence[AblationSpec]] = None,
    n_replicates: int = 120,
    docs_per_replicate: int = 160,
    seed: int = 0,
    baseline_scenario_name: Optional[str] = None,
    align_boundary_span_to_distribution: bool = False,
) -> List[GeneralizationStressSummary]:
    """
    Evaluate method robustness across DGP shifts.

    Returns one summary row per (scenario, method), including:
      - per-parameter mean absolute biases,
      - aggregate bias (mean across the three parameters),
      - generalization gap vs the baseline scenario for the same method.
    """
    active_scenarios = (
        list(scenarios) if scenarios is not None else list(default_generalization_stress_scenarios())
    )
    if len(active_scenarios) == 0:
        raise ValueError("scenarios must be non-empty")
    active_methods = (
        list(methods) if methods is not None else list(default_three_parameter_method_specs())
    )

    baseline_name = baseline_scenario_name
    if baseline_name is None:
        baseline_name = active_scenarios[0].name

    scenario_results: dict[str, List[ThreeParameterRecoverySummary]] = {}
    for s_idx, scenario in enumerate(active_scenarios):
        scenario_seed = seed + (200_000 * s_idx)
        methods_for_scenario = active_methods
        if align_boundary_span_to_distribution:
            methods_for_scenario = [
                replace(m, boundary_span_tokens=scenario.distribution.boundary_span_tokens) for m in active_methods
            ]
        scenario_results[scenario.name] = run_three_parameter_recovery_study(
            distribution=scenario.distribution,
            methods=methods_for_scenario,
            n_replicates=n_replicates,
            docs_per_replicate=docs_per_replicate,
            seed=scenario_seed,
        )

    if baseline_name not in scenario_results:
        raise ValueError(f"baseline_scenario_name={baseline_name!r} not found in scenarios")

    baseline_agg_by_method = {}
    for row in scenario_results[baseline_name]:
        baseline_agg_by_method[row.method_name] = (
            row.mean_abs_bias_p_spike
            + row.mean_abs_bias_p_two_given_spike
            + row.mean_abs_bias_p_boundary_given_spike
        ) / 3.0

    out: List[GeneralizationStressSummary] = []
    for scenario in active_scenarios:
        rows = scenario_results[scenario.name]
        for row in rows:
            agg = (
                row.mean_abs_bias_p_spike
                + row.mean_abs_bias_p_two_given_spike
                + row.mean_abs_bias_p_boundary_given_spike
            ) / 3.0
            base = baseline_agg_by_method.get(row.method_name, agg)
            out.append(
                GeneralizationStressSummary(
                    scenario_name=scenario.name,
                    scenario_description=scenario.description,
                    method_name=row.method_name,
                    method_description=row.description,
                    supports_two_spike=row.supports_two_spike,
                    supports_boundary_spike=row.supports_boundary_spike,
                    true_p_spike=row.true_p_spike,
                    true_p_two_given_spike=row.true_p_two_given_spike,
                    true_p_boundary_given_spike=row.true_p_boundary_given_spike,
                    mean_hat_p_spike=row.mean_hat_p_spike,
                    mean_hat_p_two_given_spike=row.mean_hat_p_two_given_spike,
                    mean_hat_p_boundary_given_spike=row.mean_hat_p_boundary_given_spike,
                    mean_abs_bias_p_spike=row.mean_abs_bias_p_spike,
                    mean_abs_bias_p_two_given_spike=row.mean_abs_bias_p_two_given_spike,
                    mean_abs_bias_p_boundary_given_spike=row.mean_abs_bias_p_boundary_given_spike,
                    aggregate_mean_abs_bias=agg,
                    generalization_gap_vs_baseline=agg - base,
                )
            )
    return out


__all__ = [
    "ObjectiveProfile",
    "ChunkerPolicy",
    "SelectorPolicy",
    "AggregatorPolicy",
    "MergeOrder",
    "TokenPattern",
    "ToyTokenDocument",
    "Chunk",
    "AblationSpec",
    "DocumentEvaluation",
    "AblationSummary",
    "SpikeMixtureDistributionSpec",
    "ParameterRecoverySummary",
    "TwoParameterRecoverySummary",
    "ThreeParameterRecoverySummary",
    "FourParameterRecoverySummary",
    "GeneralizationStressScenario",
    "GeneralizationStressSummary",
    "NonLanguageScenario",
    "KSketchEstimator",
    "SpikeCountMixtureDistributionSpec",
    "KSketchMethodSpec",
    "KTargetRecoverySummary",
    "ChunkQualitySweepSummary",
    "ChunkQualityCoverageSummary",
    "true_objective_score",
    "true_two_spike_event",
    "true_three_plus_spike_event",
    "true_boundary_spike_event",
    "true_spike_count",
    "generate_exact_spike_count_document",
    "generate_toy_token_document",
    "chunk_document",
    "select_chunks",
    "aggregate_chunks",
    "aggregate_chunks_topk_event",
    "evaluate_document",
    "summarize_ablation",
    "default_ablation_specs",
    "run_default_ablation_suite",
    "worked_failure_examples",
    "default_two_parameter_method_specs",
    "default_three_parameter_method_specs",
    "sample_spike_mixture_documents",
    "run_spike_prevalence_recovery_study",
    "run_two_parameter_recovery_study",
    "run_three_parameter_recovery_study",
    "default_four_parameter_method_specs",
    "run_four_parameter_recovery_study",
    "default_k_sketch_method_specs",
    "sample_spike_count_mixture_documents",
    "run_k_target_recovery_study",
    "run_chunk_quality_sweep",
    "run_chunk_quality_coverage_sweep",
    "sketch_insufficiency_counterexample",
    "default_generalization_stress_scenarios",
    "default_nonlanguage_chunk_quality_scenarios",
    "run_three_parameter_generalization_sweep",
]
