"""
Simulation harness for validating TreeIPW empirical-Bernstein confidence intervals.

This module builds finite populations of chunk-level outcomes with known ground
truth means, then repeatedly samples logged subsets under known propensities.
It supports both:

- separable chunk targets (simple mergeable-sketch style case)
- nonseparable chunk targets (chunk outcomes depend on doc-level context)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import random
from typing import Dict, Iterable, List, Optional, Tuple

from treepo._research.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from treepo._research.tree.ipw import (
    NodeType,
    TreeSample,
    effective_sample_size,
    ipw_preference_loss,
    ipw_preference_empirical_bernstein_ci,
    ipw_violation_rate,
    ipw_violation_empirical_bernstein_ci,
)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _clip_probability(p: float, lower: float, upper: float) -> float:
    lo = _clip01(lower)
    hi = _clip01(upper)
    if hi < lo:
        lo, hi = hi, lo
    if hi <= 0:
        return 0.0
    if lo >= 1:
        return 1.0
    return max(lo, min(hi, p))


class ChunkScenario(str, Enum):
    """Population generator family for chunk outcomes."""

    SEPARABLE = "separable"
    NONSEPARABLE = "nonseparable"
    DOC_NONSEPARABLE = "doc-nonseparable"


class SamplingDesign(str, Enum):
    """Logged-sample design used in simulation trials."""

    BERNOULLI = "bernoulli"
    WOR = "wor"


@dataclass(frozen=True)
class SimulatedChunk:
    """One chunk in the synthetic finite population."""

    doc_id: str
    node_id: str
    doc_propensity: float
    node_propensity: float
    violation: int
    preference_loss: float
    local_signal: float
    doc_mean_signal: float
    doc_signal_dispersion: float
    doc_policy_loss: float

    @property
    def joint_propensity(self) -> float:
        return self.doc_propensity * self.node_propensity


@dataclass(frozen=True)
class SimulatedPopulation:
    """Synthetic chunk population with exact target means."""

    scenario: ChunkScenario
    chunks: Tuple[SimulatedChunk, ...]
    true_violation_rate: float
    true_preference_loss: float

    @property
    def n_docs(self) -> int:
        return len({chunk.doc_id for chunk in self.chunks})

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    @property
    def doc_ids(self) -> Tuple[str, ...]:
        return tuple(sorted({chunk.doc_id for chunk in self.chunks}))


@dataclass(frozen=True)
class EmpiricalBernsteinCoverageResult:
    """Monte Carlo coverage summary for TreeIPW empirical-Bernstein CIs."""

    scenario: str
    sampling_design: str
    delta: float
    n_trials: int
    true_violation_rate: float
    true_preference_loss: float
    violation_coverage: float
    preference_coverage: float
    violation_mean_width: float
    preference_mean_width: float
    mean_sample_count: float
    mean_effective_sample_size: float
    empty_sample_rate: float
    ipw_violation_bias: float = float("nan")
    ipw_preference_bias: float = float("nan")
    naive_violation_coverage: float = float("nan")
    naive_preference_coverage: float = float("nan")
    naive_violation_mean_width: float = float("nan")
    naive_preference_mean_width: float = float("nan")
    naive_violation_bias: float = float("nan")
    naive_preference_bias: float = float("nan")


def compute_doc_policy_outcome(local_signals: Iterable[float]) -> float:
    """
    Nonseparable doc-level target derived from the full chunk set.

    This intentionally depends on document-level shape statistics (mean/dispersion
    and range), not on any single chunk alone.
    """
    values = [float(x) for x in local_signals]
    if not values:
        return 0.0

    mean = sum(values) / len(values)
    mad = sum(abs(x - mean) for x in values) / len(values)
    spread = max(values) - min(values)
    quadratic = mean * mean
    return _clip01(_sigmoid(-0.10 + (1.10 * mad) + (0.95 * spread) + (0.45 * quadratic) - (0.35 * mean)))


def compute_chunk_targets(
    local_signal: float,
    *,
    scenario: ChunkScenario,
    doc_mean_signal: float = 0.0,
    doc_signal_dispersion: float = 0.0,
    doc_policy_loss: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Return `(violation_prob, preference_loss)` in `[0, 1]`.

    In the separable setting, outputs are local-signal only.
    In the nonseparable setting, outputs also depend on document context.
    """
    local = float(local_signal)
    doc_mean = float(doc_mean_signal)
    doc_disp = max(0.0, float(doc_signal_dispersion))

    base_violation = -0.20 + (1.30 * local)
    base_preference = 0.05 + (1.10 * local)

    if scenario == ChunkScenario.SEPARABLE:
        violation_prob = _sigmoid(base_violation)
        preference_loss = _sigmoid(base_preference)
    elif scenario == ChunkScenario.NONSEPARABLE:
        interaction = (0.80 * doc_mean) + (1.60 * doc_disp) - (0.75 * abs(local - doc_mean))
        violation_prob = _sigmoid(base_violation + interaction)
        preference_loss = _sigmoid(base_preference - (1.00 * doc_mean) + (1.80 * doc_disp))
    elif scenario == ChunkScenario.DOC_NONSEPARABLE:
        policy = float(doc_policy_loss) if doc_policy_loss is not None else _sigmoid(
            -0.20 + (0.90 * doc_disp) + (0.70 * abs(doc_mean))
        )
        violation_prob = _sigmoid(-0.35 + (2.10 * policy) + (0.25 * (local - doc_mean)))
        preference_loss = policy
    else:
        raise ValueError(f"Unsupported scenario: {scenario!r}")

    return (_clip01(violation_prob), _clip01(preference_loss))


def generate_chunk_population(
    *,
    n_docs: int = 80,
    chunks_per_doc: int = 10,
    scenario: ChunkScenario = ChunkScenario.SEPARABLE,
    seed: int = 0,
    min_doc_propensity: float = 0.40,
    max_doc_propensity: float = 0.95,
    min_node_propensity: float = 0.10,
    max_node_propensity: float = 0.95,
) -> SimulatedPopulation:
    """Build a finite chunk population with exact true means."""
    if n_docs <= 0:
        raise ValueError("n_docs must be >= 1")
    if chunks_per_doc <= 0:
        raise ValueError("chunks_per_doc must be >= 1")

    rng = random.Random(seed)
    chunks: List[SimulatedChunk] = []

    for doc_idx in range(n_docs):
        doc_id = f"doc-{doc_idx:04d}"
        doc_signal = rng.uniform(-1.2, 1.2)
        doc_prop = _clip_probability(
            _sigmoid(-0.10 + (0.95 * doc_signal)),
            lower=min_doc_propensity,
            upper=max_doc_propensity,
        )

        local_signals = [rng.uniform(-1.0, 1.0) for _ in range(chunks_per_doc)]
        doc_mean_signal = sum(local_signals) / len(local_signals)
        doc_signal_dispersion = sum(abs(x - doc_mean_signal) for x in local_signals) / len(local_signals)
        doc_policy_loss = compute_doc_policy_outcome(local_signals)

        for chunk_idx, local_signal in enumerate(local_signals):
            violation_prob, preference_loss = compute_chunk_targets(
                local_signal,
                scenario=scenario,
                doc_mean_signal=doc_mean_signal,
                doc_signal_dispersion=doc_signal_dispersion,
                doc_policy_loss=doc_policy_loss,
            )
            violation = 1 if rng.random() < violation_prob else 0
            node_prop = _clip_probability(
                _sigmoid(-0.20 + (1.15 * local_signal) - (0.45 * doc_signal)),
                lower=min_node_propensity,
                upper=max_node_propensity,
            )
            chunks.append(
                SimulatedChunk(
                    doc_id=doc_id,
                    node_id=f"{doc_id}-chunk-{chunk_idx:03d}",
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

    if not chunks:
        raise ValueError("generated population has no chunks")

    true_violation_rate = sum(float(chunk.violation) for chunk in chunks) / len(chunks)
    true_preference_loss = sum(chunk.preference_loss for chunk in chunks) / len(chunks)

    return SimulatedPopulation(
        scenario=scenario,
        chunks=tuple(chunks),
        true_violation_rate=true_violation_rate,
        true_preference_loss=true_preference_loss,
    )


def _tree_sample_from_chunk(
    chunk: SimulatedChunk,
    *,
    scenario: ChunkScenario,
    doc_propensity: float,
    node_propensity: float,
) -> TreeSample:
    return TreeSample(
        doc_id=chunk.doc_id,
        node_id=chunk.node_id,
        node_type=NodeType.LEAF,
        violation=chunk.violation,
        preference_loss=chunk.preference_loss,
        sampling=SamplingMetadata(
            document_propensity=doc_propensity,
            unit_propensity=node_propensity,
            label_propensity=1.0,
            unit_kind=ObservationUnitKind.LEAF,
        ),
        metadata={
            "scenario": scenario.value,
            "local_signal": chunk.local_signal,
            "doc_mean_signal": chunk.doc_mean_signal,
            "doc_signal_dispersion": chunk.doc_signal_dispersion,
            "doc_policy_loss": chunk.doc_policy_loss,
        },
    )


def _as_unweighted_samples(samples: Iterable[TreeSample]) -> List[TreeSample]:
    """
    Return a copy of samples with unit inclusion propensities.

    This is used to evaluate an intentionally naive unweighted baseline under the
    same CI machinery.
    """
    out: List[TreeSample] = []
    for sample in samples:
        out.append(
            TreeSample(
                doc_id=sample.doc_id,
                node_id=sample.node_id,
                node_type=sample.node_type,
                violation=sample.violation,
                preference_loss=sample.preference_loss,
                sampling=SamplingMetadata(
                    document_propensity=1.0,
                    unit_propensity=1.0,
                    label_propensity=1.0,
                    unit_kind=sample.sampling.unit_kind,
                ),
                metadata=sample.metadata,
            )
        )
    return out


def _draw_logged_tree_samples_bernoulli(
    population: SimulatedPopulation,
    trial_rng: random.Random,
) -> List[TreeSample]:
    doc_draws: Dict[str, bool] = {}
    sampled: List[TreeSample] = []

    for chunk in population.chunks:
        include_doc = doc_draws.get(chunk.doc_id)
        if include_doc is None:
            include_doc = trial_rng.random() < chunk.doc_propensity
            doc_draws[chunk.doc_id] = include_doc
        if not include_doc:
            continue
        if trial_rng.random() >= chunk.node_propensity:
            continue
        sampled.append(
            _tree_sample_from_chunk(
                chunk,
                scenario=population.scenario,
                doc_propensity=chunk.doc_propensity,
                node_propensity=chunk.node_propensity,
            )
        )

    return sampled


def _draw_logged_tree_samples_wor(
    population: SimulatedPopulation,
    trial_rng: random.Random,
    *,
    wor_docs_sample: Optional[int],
    wor_chunks_per_doc_sample: Optional[int],
) -> List[TreeSample]:
    chunks_by_doc: Dict[str, List[SimulatedChunk]] = {}
    for chunk in population.chunks:
        chunks_by_doc.setdefault(chunk.doc_id, []).append(chunk)

    doc_ids = sorted(chunks_by_doc)
    n_docs = len(doc_ids)
    if n_docs == 0:
        return []

    if wor_docs_sample is None:
        mean_doc_prop = sum(chunks_by_doc[d][0].doc_propensity for d in doc_ids) / n_docs
        m_docs = int(round(mean_doc_prop * n_docs))
    else:
        m_docs = int(wor_docs_sample)
    m_docs = max(1, min(n_docs, m_docs))

    selected_docs = set(trial_rng.sample(doc_ids, k=m_docs))
    doc_pi = float(m_docs) / float(n_docs)

    if wor_chunks_per_doc_sample is None:
        mean_chunks = sum(len(chunks_by_doc[d]) for d in doc_ids) / float(n_docs)
        mean_node_prop = sum(chunk.node_propensity for chunk in population.chunks) / float(len(population.chunks))
        base_m_chunks = int(round(mean_node_prop * mean_chunks))
    else:
        base_m_chunks = int(wor_chunks_per_doc_sample)
    base_m_chunks = max(1, base_m_chunks)

    sampled: List[TreeSample] = []
    for doc_id in selected_docs:
        doc_chunks = chunks_by_doc[doc_id]
        n_doc_chunks = len(doc_chunks)
        m_doc_chunks = max(1, min(n_doc_chunks, base_m_chunks))
        node_pi = float(m_doc_chunks) / float(n_doc_chunks)
        chosen = trial_rng.sample(doc_chunks, k=m_doc_chunks)
        for chunk in chosen:
            sampled.append(
                _tree_sample_from_chunk(
                    chunk,
                    scenario=population.scenario,
                    doc_propensity=doc_pi,
                    node_propensity=node_pi,
                )
            )

    return sampled


def draw_logged_tree_samples(
    population: SimulatedPopulation,
    *,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
    sampling_design: SamplingDesign = SamplingDesign.BERNOULLI,
    wor_docs_sample: Optional[int] = None,
    wor_chunks_per_doc_sample: Optional[int] = None,
) -> List[TreeSample]:
    """
    Draw one logged sample set under the requested sampling design.

    - `bernoulli`: two-stage Bernoulli using per-doc and per-chunk propensities.
    - `wor`: two-stage simple-random-sampling without replacement.
    """
    if rng is not None and seed is not None:
        raise ValueError("Provide either rng or seed, not both")

    trial_rng = rng if rng is not None else random.Random(seed)
    if sampling_design == SamplingDesign.BERNOULLI:
        return _draw_logged_tree_samples_bernoulli(population, trial_rng)
    if sampling_design == SamplingDesign.WOR:
        return _draw_logged_tree_samples_wor(
            population,
            trial_rng,
            wor_docs_sample=wor_docs_sample,
            wor_chunks_per_doc_sample=wor_chunks_per_doc_sample,
        )
    raise ValueError(f"Unsupported sampling_design: {sampling_design!r}")


def evaluate_empirical_bernstein_coverage(
    population: SimulatedPopulation,
    *,
    n_trials: int = 300,
    delta: float = 0.10,
    seed: int = 0,
    sampling_design: SamplingDesign = SamplingDesign.BERNOULLI,
    wor_docs_sample: Optional[int] = None,
    wor_chunks_per_doc_sample: Optional[int] = None,
) -> EmpiricalBernsteinCoverageResult:
    """
    Estimate empirical CI coverage against exact finite-population targets.

    Returns coverage rates and basic diagnostics across Monte Carlo trials.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be >= 1")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")

    rng = random.Random(seed)
    violation_hits = 0
    preference_hits = 0
    empty_count = 0
    violation_width_sum = 0.0
    preference_width_sum = 0.0
    sample_count_sum = 0.0
    neff_sum = 0.0
    ipw_violation_bias_sum = 0.0
    ipw_preference_bias_sum = 0.0

    naive_violation_hits = 0
    naive_preference_hits = 0
    naive_violation_width_sum = 0.0
    naive_preference_width_sum = 0.0
    naive_violation_bias_sum = 0.0
    naive_preference_bias_sum = 0.0

    for _ in range(n_trials):
        sampled = draw_logged_tree_samples(
            population,
            seed=rng.randrange(0, 2**31 - 1),
            sampling_design=sampling_design,
            wor_docs_sample=wor_docs_sample,
            wor_chunks_per_doc_sample=wor_chunks_per_doc_sample,
        )
        if not sampled:
            empty_count += 1

        violation_ci = ipw_violation_empirical_bernstein_ci(sampled, delta=delta)
        preference_ci = ipw_preference_empirical_bernstein_ci(sampled, delta=delta)
        violation_hat = ipw_violation_rate(sampled)
        preference_hat = ipw_preference_loss(sampled)

        if violation_ci[0] <= population.true_violation_rate <= violation_ci[1]:
            violation_hits += 1
        if preference_ci[0] <= population.true_preference_loss <= preference_ci[1]:
            preference_hits += 1

        violation_width_sum += max(0.0, violation_ci[1] - violation_ci[0])
        preference_width_sum += max(0.0, preference_ci[1] - preference_ci[0])
        sample_count_sum += float(len(sampled))
        neff_sum += effective_sample_size(sampled)
        ipw_violation_bias_sum += float(violation_hat - population.true_violation_rate)
        ipw_preference_bias_sum += float(preference_hat - population.true_preference_loss)

        naive_samples = _as_unweighted_samples(sampled)
        naive_violation_ci = ipw_violation_empirical_bernstein_ci(naive_samples, delta=delta)
        naive_preference_ci = ipw_preference_empirical_bernstein_ci(naive_samples, delta=delta)
        naive_violation_hat = ipw_violation_rate(naive_samples)
        naive_preference_hat = ipw_preference_loss(naive_samples)

        if naive_violation_ci[0] <= population.true_violation_rate <= naive_violation_ci[1]:
            naive_violation_hits += 1
        if naive_preference_ci[0] <= population.true_preference_loss <= naive_preference_ci[1]:
            naive_preference_hits += 1

        naive_violation_width_sum += max(0.0, naive_violation_ci[1] - naive_violation_ci[0])
        naive_preference_width_sum += max(0.0, naive_preference_ci[1] - naive_preference_ci[0])
        naive_violation_bias_sum += float(naive_violation_hat - population.true_violation_rate)
        naive_preference_bias_sum += float(naive_preference_hat - population.true_preference_loss)

    inv_trials = 1.0 / float(n_trials)
    return EmpiricalBernsteinCoverageResult(
        scenario=population.scenario.value,
        sampling_design=sampling_design.value,
        delta=delta,
        n_trials=n_trials,
        true_violation_rate=population.true_violation_rate,
        true_preference_loss=population.true_preference_loss,
        violation_coverage=violation_hits * inv_trials,
        preference_coverage=preference_hits * inv_trials,
        violation_mean_width=violation_width_sum * inv_trials,
        preference_mean_width=preference_width_sum * inv_trials,
        mean_sample_count=sample_count_sum * inv_trials,
        mean_effective_sample_size=neff_sum * inv_trials,
        empty_sample_rate=empty_count * inv_trials,
        ipw_violation_bias=ipw_violation_bias_sum * inv_trials,
        ipw_preference_bias=ipw_preference_bias_sum * inv_trials,
        naive_violation_coverage=naive_violation_hits * inv_trials,
        naive_preference_coverage=naive_preference_hits * inv_trials,
        naive_violation_mean_width=naive_violation_width_sum * inv_trials,
        naive_preference_mean_width=naive_preference_width_sum * inv_trials,
        naive_violation_bias=naive_violation_bias_sum * inv_trials,
        naive_preference_bias=naive_preference_bias_sum * inv_trials,
    )


def summarize_coverage_runs(
    runs: Iterable[EmpiricalBernsteinCoverageResult],
) -> Dict[str, Dict[str, float]]:
    """Convert per-run results into a compact serializable summary."""
    summary: Dict[str, Dict[str, float]] = {}
    for run in runs:
        summary[f"{run.scenario}|{run.sampling_design}"] = {
            "delta": run.delta,
            "n_trials": float(run.n_trials),
            "true_violation_rate": run.true_violation_rate,
            "true_preference_loss": run.true_preference_loss,
            "violation_coverage": run.violation_coverage,
            "preference_coverage": run.preference_coverage,
            "violation_mean_width": run.violation_mean_width,
            "preference_mean_width": run.preference_mean_width,
            "mean_sample_count": run.mean_sample_count,
            "mean_effective_sample_size": run.mean_effective_sample_size,
            "empty_sample_rate": run.empty_sample_rate,
            "ipw_violation_bias": run.ipw_violation_bias,
            "ipw_preference_bias": run.ipw_preference_bias,
            "naive_violation_coverage": run.naive_violation_coverage,
            "naive_preference_coverage": run.naive_preference_coverage,
            "naive_violation_mean_width": run.naive_violation_mean_width,
            "naive_preference_mean_width": run.naive_preference_mean_width,
            "naive_violation_bias": run.naive_violation_bias,
            "naive_preference_bias": run.naive_preference_bias,
        }
    return summary


__all__ = [
    "ChunkScenario",
    "SamplingDesign",
    "SimulatedChunk",
    "SimulatedPopulation",
    "EmpiricalBernsteinCoverageResult",
    "compute_doc_policy_outcome",
    "compute_chunk_targets",
    "generate_chunk_population",
    "draw_logged_tree_samples",
    "evaluate_empirical_bernstein_coverage",
    "summarize_coverage_runs",
]
