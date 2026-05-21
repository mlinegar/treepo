"""
Toy chunking problems for CI stress-testing.

These generators deliberately create small, interpretable chunk structures:
- one-word-at-a-time documents
- one-character-at-a-time documents

with controlled chunk-importance patterns and propensity imbalance profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from treepo._research.tree.ipw_simulation import (
    ChunkScenario,
    SamplingDesign,
    SimulatedChunk,
    SimulatedPopulation,
    compute_chunk_targets,
    compute_doc_policy_outcome,
    evaluate_empirical_bernstein_coverage,
)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _clip_probability(p: float, lower: float, upper: float) -> float:
    lo = _clip01(lower)
    hi = _clip01(upper)
    if hi < lo:
        lo, hi = hi, lo
    return max(lo, min(hi, p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    clipped_q = _clip01(q)
    sorted_vals = sorted(float(v) for v in values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = clipped_q * float(len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    w = pos - float(lo)
    return (1.0 - w) * sorted_vals[lo] + w * sorted_vals[hi]


class ChunkGranularity(str, Enum):
    """Tokenization scale for toy chunking."""

    WORD = "word"
    CHAR = "char"


class ChunkPattern(str, Enum):
    """Where informative content lives across chunk positions."""

    UNIFORM = "uniform"
    FRONT_LOADED = "front-loaded"
    BACK_LOADED = "back-loaded"
    ALTERNATING = "alternating"
    SPIKE = "spike"
    BOUNDARY = "boundary"


class ImbalanceProfile(str, Enum):
    """Propensity imbalance level and orientation."""

    BALANCED = "balanced"
    MODERATE = "moderate"
    SEVERE = "severe"
    ADVERSARIAL = "adversarial"


class LengthProfile(str, Enum):
    """Document chunk-count distribution."""

    FIXED = "fixed"
    UNIFORM = "uniform"
    BIMODAL = "bimodal"
    LONG_TAIL = "long-tail"


class OraclePreferenceProfile(str, Enum):
    """
    Doc-level oracle preference functional computed from a mergeable sketch.

    `legacy-smooth` preserves the historical policy-loss generator.
    `additive-mean` is the additive control case.
    The remaining modes are non-additive mergeable-sketch outcomes.
    """

    LEGACY_SMOOTH = "legacy-smooth"
    ADDITIVE_MEAN = "additive-mean"
    TOPK_SPIKE = "topk-spike"
    QUORUM_GATE = "quorum-gate"
    HYBRID_EXTREME = "hybrid-extreme"


class ExampleExpectation(str, Enum):
    """Expected simulation behavior label for curated examples."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass(frozen=True)
class ToyPopulationDiagnostics:
    """Diagnostics for understanding propensity imbalance severity."""

    min_joint_propensity: float
    p10_joint_propensity: float
    median_joint_propensity: float
    max_joint_weight: float
    high_signal_low_propensity_overlap: float
    min_doc_length: int
    p50_doc_length: float
    p90_doc_length: float
    max_doc_length: int


@dataclass(frozen=True)
class ToyCoverageRun:
    """One coverage run labeled by toy spec dimensions."""

    scenario: ChunkScenario
    sampling_design: SamplingDesign
    granularity: ChunkGranularity
    pattern: ChunkPattern
    imbalance: ImbalanceProfile
    length_profile: LengthProfile
    oracle_preference: OraclePreferenceProfile
    n_docs: int
    chunks_per_doc: int
    diagnostics: ToyPopulationDiagnostics
    coverage: Dict[str, float]


@dataclass(frozen=True)
class MergeableSketchExampleSpec:
    """Curated toy configuration for mergeable-sketch preference studies."""

    name: str
    expectation: ExampleExpectation
    description: str
    scenario: ChunkScenario
    granularity: ChunkGranularity
    pattern: ChunkPattern
    imbalance: ImbalanceProfile
    length_profile: LengthProfile
    oracle_preference: OraclePreferenceProfile


@dataclass(frozen=True)
class MergeableSketchCoverageRun:
    """Coverage result tagged by curated mergeable-sketch example metadata."""

    example_name: str
    expectation: ExampleExpectation
    description: str
    scenario: ChunkScenario
    sampling_design: SamplingDesign
    granularity: ChunkGranularity
    pattern: ChunkPattern
    imbalance: ImbalanceProfile
    length_profile: LengthProfile
    oracle_preference: OraclePreferenceProfile
    n_docs: int
    chunks_per_doc: int
    diagnostics: ToyPopulationDiagnostics
    coverage: Dict[str, float]


@dataclass(frozen=True)
class _ImbalanceParams:
    doc_min: float
    doc_max: float
    node_min: float
    node_max: float
    slope: float
    inverse: bool


@dataclass(frozen=True)
class _SignalSketch:
    """Mergeable sketch summary of local chunk signals for one document."""

    count: int
    sum_signal: float
    sum_abs_signal: float
    high_count: int
    top1: float
    top2: float


def _imbalance_params(profile: ImbalanceProfile) -> _ImbalanceParams:
    if profile == ImbalanceProfile.BALANCED:
        return _ImbalanceParams(0.75, 0.98, 0.55, 0.98, 1.0, False)
    if profile == ImbalanceProfile.MODERATE:
        return _ImbalanceParams(0.45, 0.95, 0.20, 0.95, 1.6, False)
    if profile == ImbalanceProfile.SEVERE:
        return _ImbalanceParams(0.20, 0.85, 0.03, 0.80, 2.4, False)
    if profile == ImbalanceProfile.ADVERSARIAL:
        return _ImbalanceParams(0.12, 0.80, 0.005, 0.55, 3.0, True)
    raise ValueError(f"Unsupported imbalance profile: {profile!r}")


def _empty_signal_sketch() -> _SignalSketch:
    return _SignalSketch(
        count=0,
        sum_signal=0.0,
        sum_abs_signal=0.0,
        high_count=0,
        top1=-float("inf"),
        top2=-float("inf"),
    )


def _singleton_signal_sketch(signal: float, *, high_threshold: float = 0.60) -> _SignalSketch:
    value = float(signal)
    return _SignalSketch(
        count=1,
        sum_signal=value,
        sum_abs_signal=abs(value),
        high_count=1 if value >= high_threshold else 0,
        top1=value,
        top2=-float("inf"),
    )


def _merge_signal_sketch(left: _SignalSketch, right: _SignalSketch) -> _SignalSketch:
    top_values = sorted([left.top1, left.top2, right.top1, right.top2], reverse=True)
    return _SignalSketch(
        count=left.count + right.count,
        sum_signal=left.sum_signal + right.sum_signal,
        sum_abs_signal=left.sum_abs_signal + right.sum_abs_signal,
        high_count=left.high_count + right.high_count,
        top1=top_values[0],
        top2=top_values[1],
    )


def _signal_sketch_from_values(local_signals: Iterable[float]) -> _SignalSketch:
    sketch = _empty_signal_sketch()
    for value in local_signals:
        sketch = _merge_signal_sketch(sketch, _singleton_signal_sketch(float(value)))
    return sketch


def _policy_from_signal_sketch(
    sketch: _SignalSketch,
    *,
    oracle_preference: OraclePreferenceProfile,
) -> float:
    if sketch.count <= 0:
        return 0.0

    mean = sketch.sum_signal / float(sketch.count)
    high_rate = sketch.high_count / float(sketch.count)
    top1 = sketch.top1 if math.isfinite(sketch.top1) else -1.0
    top2 = sketch.top2 if math.isfinite(sketch.top2) else -1.0
    top_gap = max(0.0, top1 - top2)
    concentration = top1 - mean

    if oracle_preference == OraclePreferenceProfile.ADDITIVE_MEAN:
        return _clip01(_sigmoid(-0.05 + (1.35 * mean)))
    if oracle_preference == OraclePreferenceProfile.TOPK_SPIKE:
        return _clip01(_sigmoid(-0.95 + (2.45 * top1) + (2.10 * top_gap)))
    if oracle_preference == OraclePreferenceProfile.QUORUM_GATE:
        gate = 1.0 if high_rate >= 0.24 else 0.0
        return _clip01(0.05 + (0.80 * gate) + (0.15 * _sigmoid(0.80 * mean)))
    if oracle_preference == OraclePreferenceProfile.HYBRID_EXTREME:
        return _clip01(
            _sigmoid(
                -1.25
                + (2.35 * top1)
                + (1.40 * high_rate)
                + (0.90 * concentration)
                - (0.65 * abs(mean))
            )
        )
    if oracle_preference == OraclePreferenceProfile.LEGACY_SMOOTH:
        return _clip01(_sigmoid(-0.20 + (0.90 * concentration) + (0.80 * high_rate) + (0.35 * mean * mean)))
    raise ValueError(f"Unsupported oracle preference profile: {oracle_preference!r}")


def compute_doc_policy_outcome_from_mergeable_sketch(
    local_signals: Iterable[float],
    *,
    oracle_preference: OraclePreferenceProfile,
) -> float:
    """
    Compute doc-level policy loss from mergeable sketch summaries.

    This provides additive and non-additive oracle preference families.
    """
    values = [float(x) for x in local_signals]
    if not values:
        return 0.0
    if oracle_preference == OraclePreferenceProfile.LEGACY_SMOOTH:
        return compute_doc_policy_outcome(values)
    sketch = _signal_sketch_from_values(values)
    return _policy_from_signal_sketch(sketch, oracle_preference=oracle_preference)


def _word_tokens(chunks_per_doc: int, rng: random.Random) -> List[str]:
    base = [
        "we",
        "test",
        "a",
        "simple",
        "sentence",
        "one",
        "word",
        "at",
        "a",
        "time",
        "for",
        "coverage",
    ]
    if chunks_per_doc <= 0:
        return []
    offset = rng.randrange(len(base))
    return [base[(offset + i) % len(base)] for i in range(chunks_per_doc)]


def _char_tokens(chunks_per_doc: int, rng: random.Random) -> List[str]:
    base = list("we test a simple sentence, one character at a time.")
    if chunks_per_doc <= 0:
        return []
    offset = rng.randrange(len(base))
    return [base[(offset + i) % len(base)] for i in range(chunks_per_doc)]


def _importance_profile(
    pattern: ChunkPattern,
    chunks_per_doc: int,
    rng: random.Random,
) -> List[float]:
    if chunks_per_doc <= 0:
        return []
    if chunks_per_doc == 1:
        return [1.0]

    idxs = list(range(chunks_per_doc))
    denom = float(chunks_per_doc - 1)

    if pattern == ChunkPattern.UNIFORM:
        return [0.5 for _ in idxs]
    if pattern == ChunkPattern.FRONT_LOADED:
        return [math.exp(-3.0 * (float(i) / denom)) for i in idxs]
    if pattern == ChunkPattern.BACK_LOADED:
        return [math.exp(-3.0 * ((denom - float(i)) / denom)) for i in idxs]
    if pattern == ChunkPattern.ALTERNATING:
        return [1.0 if (i % 2 == 0) else 0.05 for i in idxs]
    if pattern == ChunkPattern.SPIKE:
        spike_idx = rng.randrange(chunks_per_doc)
        return [1.0 if i == spike_idx else 0.02 for i in idxs]
    if pattern == ChunkPattern.BOUNDARY:
        return [1.0 if (i == 0 or i == chunks_per_doc - 1) else 0.08 for i in idxs]

    raise ValueError(f"Unsupported pattern: {pattern!r}")


def _toy_doc_propensity(
    *,
    doc_policy_loss: float,
    profile: ImbalanceProfile,
    rng: random.Random,
) -> float:
    params = _imbalance_params(profile)
    score = doc_policy_loss
    if params.inverse:
        score = 1.0 - score
    noisy = (score - 0.5) + rng.uniform(-0.15, 0.15)
    prob = _sigmoid(-0.3 + (2.2 * noisy))
    return _clip_probability(prob, params.doc_min, params.doc_max)


def _toy_node_propensity(
    *,
    importance: float,
    position: int,
    n_chunks: int,
    pattern: ChunkPattern,
    profile: ImbalanceProfile,
    rng: random.Random,
) -> float:
    params = _imbalance_params(profile)
    score = importance
    if params.inverse:
        score = 1.0 - score

    pos = (float(position) / float(max(1, n_chunks - 1))) if n_chunks > 1 else 0.0
    center_bonus = 0.5 - abs(pos - 0.5)
    noisy = (score - 0.5) + 0.4 * center_bonus + rng.uniform(-0.12, 0.12)
    prob = _sigmoid(-0.9 + (params.slope * noisy))
    prop = _clip_probability(prob, params.node_min, params.node_max)

    if profile == ImbalanceProfile.ADVERSARIAL and importance > 0.85:
        prop = max(params.node_min, prop * 0.25)
    if profile == ImbalanceProfile.ADVERSARIAL and pattern in (
        ChunkPattern.FRONT_LOADED,
        ChunkPattern.BACK_LOADED,
        ChunkPattern.BOUNDARY,
    ):
        if position == 0 or position == (n_chunks - 1):
            prop = max(params.node_min, prop * 0.35)
    return prop


def _sample_doc_length(
    *,
    base_chunks_per_doc: int,
    min_chunks_per_doc: Optional[int],
    max_chunks_per_doc: Optional[int],
    length_profile: LengthProfile,
    rng: random.Random,
) -> int:
    base = max(1, int(base_chunks_per_doc))
    lo = max(1, int(min_chunks_per_doc)) if min_chunks_per_doc is not None else max(1, base // 2)
    hi = max(lo, int(max_chunks_per_doc)) if max_chunks_per_doc is not None else max(lo, int(round(base * 2.5)))

    if length_profile == LengthProfile.FIXED:
        return max(lo, min(hi, base))
    if length_profile == LengthProfile.UNIFORM:
        return rng.randint(lo, hi)
    if length_profile == LengthProfile.BIMODAL:
        short_hi = max(lo, min(hi, lo + max(1, (base - lo) // 3)))
        long_lo = max(lo, min(hi, hi - max(1, (hi - base) // 3)))
        if rng.random() < 0.5:
            return rng.randint(lo, short_hi)
        return rng.randint(long_lo, hi)
    if length_profile == LengthProfile.LONG_TAIL:
        span = max(0, hi - lo)
        if span == 0:
            return lo
        k = 0
        while k < span and rng.random() > 0.28:
            k += 1
        return lo + k

    raise ValueError(f"Unsupported length profile: {length_profile!r}")


def toy_population_diagnostics(population: SimulatedPopulation) -> ToyPopulationDiagnostics:
    chunks = list(population.chunks)
    if not chunks:
        return ToyPopulationDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0)

    joints = [max(1e-12, chunk.joint_propensity) for chunk in chunks]
    high_signal = [chunk for chunk in chunks if chunk.local_signal >= 0.6]
    low_prop_threshold = _quantile(joints, 0.35)
    overlap = 0.0
    if high_signal:
        overlap_count = sum(1 for chunk in high_signal if chunk.joint_propensity <= low_prop_threshold)
        overlap = overlap_count / float(len(high_signal))

    doc_sizes: Dict[str, int] = {}
    for chunk in chunks:
        doc_sizes[chunk.doc_id] = doc_sizes.get(chunk.doc_id, 0) + 1
    lengths = [int(v) for v in doc_sizes.values()]

    return ToyPopulationDiagnostics(
        min_joint_propensity=min(joints),
        p10_joint_propensity=_quantile(joints, 0.10),
        median_joint_propensity=_quantile(joints, 0.50),
        max_joint_weight=max(1.0 / p for p in joints),
        high_signal_low_propensity_overlap=overlap,
        min_doc_length=min(lengths),
        p50_doc_length=_quantile(lengths, 0.50),
        p90_doc_length=_quantile(lengths, 0.90),
        max_doc_length=max(lengths),
    )


def generate_toy_chunk_population(
    *,
    n_docs: int = 60,
    chunks_per_doc: int = 16,
    scenario: ChunkScenario = ChunkScenario.NONSEPARABLE,
    granularity: ChunkGranularity = ChunkGranularity.WORD,
    pattern: ChunkPattern = ChunkPattern.FRONT_LOADED,
    imbalance: ImbalanceProfile = ImbalanceProfile.MODERATE,
    length_profile: LengthProfile = LengthProfile.FIXED,
    oracle_preference: OraclePreferenceProfile = OraclePreferenceProfile.LEGACY_SMOOTH,
    min_chunks_per_doc: Optional[int] = None,
    max_chunks_per_doc: Optional[int] = None,
    seed: int = 0,
) -> SimulatedPopulation:
    """Build an interpretable toy population at word or character chunk scale."""
    if n_docs <= 0:
        raise ValueError("n_docs must be >= 1")
    if chunks_per_doc <= 0:
        raise ValueError("chunks_per_doc must be >= 1")

    rng = random.Random(seed)
    chunks: List[SimulatedChunk] = []

    for doc_idx in range(n_docs):
        doc_id = f"toy-doc-{doc_idx:04d}"
        doc_chunk_count = _sample_doc_length(
            base_chunks_per_doc=chunks_per_doc,
            min_chunks_per_doc=min_chunks_per_doc,
            max_chunks_per_doc=max_chunks_per_doc,
            length_profile=length_profile,
            rng=rng,
        )
        if granularity == ChunkGranularity.WORD:
            tokens = _word_tokens(doc_chunk_count, rng)
        elif granularity == ChunkGranularity.CHAR:
            tokens = _char_tokens(doc_chunk_count, rng)
        else:
            raise ValueError(f"Unsupported granularity: {granularity!r}")

        importance = _importance_profile(pattern, doc_chunk_count, rng)
        local_signals = [2.0 * imp - 1.0 for imp in importance]
        doc_mean_signal = sum(local_signals) / len(local_signals)
        doc_signal_dispersion = sum(abs(x - doc_mean_signal) for x in local_signals) / len(local_signals)
        doc_policy_loss = compute_doc_policy_outcome_from_mergeable_sketch(
            local_signals,
            oracle_preference=oracle_preference,
        )
        doc_prop = _toy_doc_propensity(doc_policy_loss=doc_policy_loss, profile=imbalance, rng=rng)

        for i, token in enumerate(tokens):
            local_signal = local_signals[i]
            imp = importance[i]
            node_prop = _toy_node_propensity(
                importance=imp,
                position=i,
                n_chunks=doc_chunk_count,
                pattern=pattern,
                profile=imbalance,
                rng=rng,
            )
            violation_prob, preference_loss = compute_chunk_targets(
                local_signal,
                scenario=scenario,
                doc_mean_signal=doc_mean_signal,
                doc_signal_dispersion=doc_signal_dispersion,
                doc_policy_loss=doc_policy_loss,
            )
            violation = 1 if rng.random() < violation_prob else 0

            chunks.append(
                SimulatedChunk(
                    doc_id=doc_id,
                    node_id=f"{doc_id}-{granularity.value}-{i:03d}",
                    doc_propensity=doc_prop,
                    node_propensity=node_prop,
                    violation=violation,
                    preference_loss=preference_loss,
                    local_signal=local_signal,
                    doc_mean_signal=doc_mean_signal,
                    doc_signal_dispersion=doc_signal_dispersion,
                    doc_policy_loss=doc_policy_loss,
                )
            )

    true_violation_rate = sum(float(chunk.violation) for chunk in chunks) / float(len(chunks))
    true_preference_loss = sum(chunk.preference_loss for chunk in chunks) / float(len(chunks))
    return SimulatedPopulation(
        scenario=scenario,
        chunks=tuple(chunks),
        true_violation_rate=true_violation_rate,
        true_preference_loss=true_preference_loss,
    )


def run_toy_coverage_suite(
    *,
    scenarios: Iterable[ChunkScenario],
    designs: Iterable[SamplingDesign],
    granularities: Iterable[ChunkGranularity],
    patterns: Iterable[ChunkPattern],
    imbalances: Iterable[ImbalanceProfile],
    length_profiles: Iterable[LengthProfile] = (LengthProfile.FIXED,),
    oracle_preferences: Iterable[OraclePreferenceProfile] = (OraclePreferenceProfile.LEGACY_SMOOTH,),
    n_docs: int = 60,
    chunks_per_doc: int = 16,
    min_chunks_per_doc: Optional[int] = None,
    max_chunks_per_doc: Optional[int] = None,
    n_trials: int = 200,
    delta: float = 0.10,
    population_seed: int = 17,
    trial_seed: int = 23,
    wor_docs_sample: Optional[int] = None,
    wor_chunks_per_doc_sample: Optional[int] = None,
) -> List[ToyCoverageRun]:
    """
    Run a grid of toy simulations and return labeled coverage records.

    This is the main entrypoint for structured comparison studies.
    """
    runs: List[ToyCoverageRun] = []
    run_idx = 0

    for granularity in granularities:
        for pattern in patterns:
            for imbalance in imbalances:
                for length_profile in length_profiles:
                    for oracle_preference in oracle_preferences:
                        for scenario in scenarios:
                            population = generate_toy_chunk_population(
                                n_docs=n_docs,
                                chunks_per_doc=chunks_per_doc,
                                scenario=scenario,
                                granularity=granularity,
                                pattern=pattern,
                                imbalance=imbalance,
                                length_profile=length_profile,
                                oracle_preference=oracle_preference,
                                min_chunks_per_doc=min_chunks_per_doc,
                                max_chunks_per_doc=max_chunks_per_doc,
                                seed=population_seed + run_idx,
                            )
                            diag = toy_population_diagnostics(population)
                            for design in designs:
                                coverage = evaluate_empirical_bernstein_coverage(
                                    population,
                                    n_trials=n_trials,
                                    delta=delta,
                                    seed=trial_seed + run_idx,
                                    sampling_design=design,
                                    wor_docs_sample=wor_docs_sample,
                                    wor_chunks_per_doc_sample=wor_chunks_per_doc_sample,
                                )
                                runs.append(
                                    ToyCoverageRun(
                                        scenario=scenario,
                                        sampling_design=design,
                                        granularity=granularity,
                                        pattern=pattern,
                                        imbalance=imbalance,
                                        length_profile=length_profile,
                                        oracle_preference=oracle_preference,
                                        n_docs=n_docs,
                                        chunks_per_doc=chunks_per_doc,
                                        diagnostics=diag,
                                        coverage={
                                            "delta": coverage.delta,
                                            "n_trials": float(coverage.n_trials),
                                            "violation_coverage": coverage.violation_coverage,
                                            "preference_coverage": coverage.preference_coverage,
                                            "violation_mean_width": coverage.violation_mean_width,
                                            "preference_mean_width": coverage.preference_mean_width,
                                            "mean_sample_count": coverage.mean_sample_count,
                                            "mean_effective_sample_size": coverage.mean_effective_sample_size,
                                            "empty_sample_rate": coverage.empty_sample_rate,
                                            "true_violation_rate": coverage.true_violation_rate,
                                            "true_preference_loss": coverage.true_preference_loss,
                                            "ipw_violation_bias": coverage.ipw_violation_bias,
                                            "ipw_preference_bias": coverage.ipw_preference_bias,
                                            "naive_violation_coverage": coverage.naive_violation_coverage,
                                            "naive_preference_coverage": coverage.naive_preference_coverage,
                                            "naive_violation_mean_width": coverage.naive_violation_mean_width,
                                            "naive_preference_mean_width": coverage.naive_preference_mean_width,
                                            "naive_violation_bias": coverage.naive_violation_bias,
                                            "naive_preference_bias": coverage.naive_preference_bias,
                                        },
                                    )
                                )
                            run_idx += 1
    return runs


def mergeable_sketch_example_specs() -> Tuple[MergeableSketchExampleSpec, ...]:
    """Curated mergeable-sketch examples with explicit positive/negative labels."""
    return (
        MergeableSketchExampleSpec(
            name="positive_additive_uniform_balanced",
            expectation=ExampleExpectation.POSITIVE,
            description="Control case: additive oracle preference and balanced propensities.",
            scenario=ChunkScenario.DOC_NONSEPARABLE,
            granularity=ChunkGranularity.WORD,
            pattern=ChunkPattern.UNIFORM,
            imbalance=ImbalanceProfile.BALANCED,
            length_profile=LengthProfile.FIXED,
            oracle_preference=OraclePreferenceProfile.ADDITIVE_MEAN,
        ),
        MergeableSketchExampleSpec(
            name="positive_nonadditive_quorum_moderate",
            expectation=ExampleExpectation.POSITIVE,
            description="Non-additive quorum gate under moderate imbalance remains stable.",
            scenario=ChunkScenario.DOC_NONSEPARABLE,
            granularity=ChunkGranularity.WORD,
            pattern=ChunkPattern.ALTERNATING,
            imbalance=ImbalanceProfile.MODERATE,
            length_profile=LengthProfile.UNIFORM,
            oracle_preference=OraclePreferenceProfile.QUORUM_GATE,
        ),
        MergeableSketchExampleSpec(
            name="positive_nonadditive_topk_balanced",
            expectation=ExampleExpectation.POSITIVE,
            description="Top-k spike preference with balanced sampling as a non-additive sanity check.",
            scenario=ChunkScenario.DOC_NONSEPARABLE,
            granularity=ChunkGranularity.CHAR,
            pattern=ChunkPattern.SPIKE,
            imbalance=ImbalanceProfile.BALANCED,
            length_profile=LengthProfile.UNIFORM,
            oracle_preference=OraclePreferenceProfile.TOPK_SPIKE,
        ),
        MergeableSketchExampleSpec(
            name="negative_nonadditive_topk_adversarial_long_tail",
            expectation=ExampleExpectation.NEGATIVE,
            description="Critical spike chunks align with low propensity and long-tail lengths.",
            scenario=ChunkScenario.DOC_NONSEPARABLE,
            granularity=ChunkGranularity.CHAR,
            pattern=ChunkPattern.SPIKE,
            imbalance=ImbalanceProfile.ADVERSARIAL,
            length_profile=LengthProfile.LONG_TAIL,
            oracle_preference=OraclePreferenceProfile.TOPK_SPIKE,
        ),
        MergeableSketchExampleSpec(
            name="negative_nonadditive_hybrid_boundary_adversarial",
            expectation=ExampleExpectation.NEGATIVE,
            description="Boundary concentration with adversarial propensities stresses CIs.",
            scenario=ChunkScenario.DOC_NONSEPARABLE,
            granularity=ChunkGranularity.WORD,
            pattern=ChunkPattern.BOUNDARY,
            imbalance=ImbalanceProfile.ADVERSARIAL,
            length_profile=LengthProfile.BIMODAL,
            oracle_preference=OraclePreferenceProfile.HYBRID_EXTREME,
        ),
        MergeableSketchExampleSpec(
            name="negative_nonadditive_quorum_severe_long_tail",
            expectation=ExampleExpectation.NEGATIVE,
            description="Quorum gate under severe imbalance and long-tail doc lengths.",
            scenario=ChunkScenario.DOC_NONSEPARABLE,
            granularity=ChunkGranularity.CHAR,
            pattern=ChunkPattern.BACK_LOADED,
            imbalance=ImbalanceProfile.SEVERE,
            length_profile=LengthProfile.LONG_TAIL,
            oracle_preference=OraclePreferenceProfile.QUORUM_GATE,
        ),
    )


def run_mergeable_sketch_examples(
    *,
    designs: Iterable[SamplingDesign],
    n_docs: int = 60,
    chunks_per_doc: int = 16,
    min_chunks_per_doc: Optional[int] = None,
    max_chunks_per_doc: Optional[int] = None,
    n_trials: int = 200,
    delta: float = 0.10,
    population_seed: int = 17,
    trial_seed: int = 23,
    wor_docs_sample: Optional[int] = None,
    wor_chunks_per_doc_sample: Optional[int] = None,
    specs: Optional[Sequence[MergeableSketchExampleSpec]] = None,
) -> List[MergeableSketchCoverageRun]:
    """Run curated mergeable-sketch examples and collect coverage diagnostics."""
    selected_specs = list(specs) if specs is not None else list(mergeable_sketch_example_specs())

    runs: List[MergeableSketchCoverageRun] = []
    run_idx = 0
    for spec in selected_specs:
        population = generate_toy_chunk_population(
            n_docs=n_docs,
            chunks_per_doc=chunks_per_doc,
            scenario=spec.scenario,
            granularity=spec.granularity,
            pattern=spec.pattern,
            imbalance=spec.imbalance,
            length_profile=spec.length_profile,
            oracle_preference=spec.oracle_preference,
            min_chunks_per_doc=min_chunks_per_doc,
            max_chunks_per_doc=max_chunks_per_doc,
            seed=population_seed + run_idx,
        )
        diag = toy_population_diagnostics(population)
        for design in designs:
            coverage = evaluate_empirical_bernstein_coverage(
                population,
                n_trials=n_trials,
                delta=delta,
                seed=trial_seed + run_idx,
                sampling_design=design,
                wor_docs_sample=wor_docs_sample,
                wor_chunks_per_doc_sample=wor_chunks_per_doc_sample,
            )
            runs.append(
                MergeableSketchCoverageRun(
                    example_name=spec.name,
                    expectation=spec.expectation,
                    description=spec.description,
                    scenario=spec.scenario,
                    sampling_design=design,
                    granularity=spec.granularity,
                    pattern=spec.pattern,
                    imbalance=spec.imbalance,
                    length_profile=spec.length_profile,
                    oracle_preference=spec.oracle_preference,
                    n_docs=n_docs,
                    chunks_per_doc=chunks_per_doc,
                    diagnostics=diag,
                    coverage={
                        "delta": coverage.delta,
                        "n_trials": float(coverage.n_trials),
                        "violation_coverage": coverage.violation_coverage,
                        "preference_coverage": coverage.preference_coverage,
                        "violation_mean_width": coverage.violation_mean_width,
                        "preference_mean_width": coverage.preference_mean_width,
                        "mean_sample_count": coverage.mean_sample_count,
                        "mean_effective_sample_size": coverage.mean_effective_sample_size,
                        "empty_sample_rate": coverage.empty_sample_rate,
                        "true_violation_rate": coverage.true_violation_rate,
                        "true_preference_loss": coverage.true_preference_loss,
                        "ipw_violation_bias": coverage.ipw_violation_bias,
                        "ipw_preference_bias": coverage.ipw_preference_bias,
                        "naive_violation_coverage": coverage.naive_violation_coverage,
                        "naive_preference_coverage": coverage.naive_preference_coverage,
                        "naive_violation_mean_width": coverage.naive_violation_mean_width,
                        "naive_preference_mean_width": coverage.naive_preference_mean_width,
                        "naive_violation_bias": coverage.naive_violation_bias,
                        "naive_preference_bias": coverage.naive_preference_bias,
                    },
                )
            )
        run_idx += 1
    return runs


__all__ = [
    "ChunkGranularity",
    "ChunkPattern",
    "ImbalanceProfile",
    "LengthProfile",
    "OraclePreferenceProfile",
    "ExampleExpectation",
    "ToyPopulationDiagnostics",
    "ToyCoverageRun",
    "MergeableSketchExampleSpec",
    "MergeableSketchCoverageRun",
    "compute_doc_policy_outcome_from_mergeable_sketch",
    "generate_toy_chunk_population",
    "toy_population_diagnostics",
    "run_toy_coverage_suite",
    "mergeable_sketch_example_specs",
    "run_mergeable_sketch_examples",
]
