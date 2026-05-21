"""
OPS Auditor - Probabilistic verification of summarization quality.

The auditor samples nodes from the OPS tree and verifies that summaries
preserve the information specified in the rubric. It uses an Oracle
(ground truth or learned approximation) to detect information loss.

Oracle Options:
- Ground Truth Oracle: Uses actual task oracle (e.g., human labels, external API)
- Oracle Approximation: Uses learned classifier trained on oracle samples
- Mini-Trees: Samples from smaller trees to generate training data for approximation

Key features:
- Probabilistic sampling with configurable budget
- Flagging system for human/oracle batch review
- Review queue for collecting items needing verification
- Support for both exact oracles and learned approximations

Usage:
    # With ground truth oracle
    auditor = TreeAuditor(oracle_judge=ground_truth_oracle, budget=10)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Protocol, Set, Tuple, Dict, Any, Union
from enum import Enum
from datetime import datetime
import random
import logging
import json
import warnings
import math

from treepo._research.core.data_models import Node, Tree, AuditStatus, AuditResult
from treepo._research.core.logged_supervision import (
    LoggedLabelObservation,
    LoggedObservationArtifact,
    ObservationUnitKind,
    SamplingMetadata,
)
from treepo._research.core.scoring import OracleScore, ScoringOracle
from treepo._research.core.ops_checks import CheckType, CheckConfig, aggregate_check_stats, LawKind
from treepo._research.core.provenance import ORACLE_SOURCE
from treepo._research.config.concurrency import ConcurrencyConfig, get_concurrency_config
from treepo._research.stats.sampling import (
    pps_inclusion_probabilities,
    systematic_pps_sample_indices,
)
from treepo._research.tree.compositional_operator import OperatorAssumptionBundle
from treepo._research.tree.compositional_learning import (
    CompositionalLearningProblemSpec,
    SupervisionDeliveryMode,
    shared_logged_substructure_observation,
    shared_protocol_problem_notes,
    shared_sampled_substructure_query_policy,
    shared_sampled_substructure_supervision_channel,
)
from treepo._research.tree.compositional_operator import make_text_compositional_operator
from treepo._research.tree.theorem_backing import TheoremAssumptionSpec


logger = logging.getLogger(__name__)


# =============================================================================
# Statistical Functions (From Audit.lean)
# =============================================================================

def confidence_margin(delta: float, n: int) -> float:
    """
    Hoeffding confidence margin: sqrt(ln(2/delta) / (2n))

    With n samples and confidence parameter delta, this is the margin epsilon
    such that P(|p_hat - p_true| >= epsilon) <= delta.

    From Audit.lean: confidence_margin

    Args:
        delta: Confidence parameter (e.g., 0.05 for 95% confidence)
        n: Number of samples

    Returns:
        Margin epsilon such that P(p_true <= p_hat + epsilon) >= 1 - delta
    """
    if n <= 0 or delta <= 0 or delta >= 2:
        return float('inf')
    return math.sqrt(math.log(2 / delta) / (2 * n))


def sample_complexity(epsilon: float, delta: float) -> int:
    """
    Minimum samples needed for (epsilon, delta)-guarantee.

    Returns ceil(ln(2/delta) / (2 * epsilon^2))

    From Audit.lean: sample_complexity

    Args:
        epsilon: Target margin (e.g., 0.05 for 5% error bound)
        delta: Confidence parameter (e.g., 0.05 for 95% confidence)

    Returns:
        Minimum number of samples n such that confidence_margin(delta, n) <= epsilon
    """
    if epsilon <= 0 or delta <= 0 or delta >= 2:
        return int(1e9)  # Return large int instead of inf for type consistency
    return math.ceil(math.log(2 / delta) / (2 * epsilon**2))


def compute_required_samples(
    epsilon: float = 0.05,
    delta: float = 0.05,
    check_types: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Compute required sample budget for target (epsilon, delta) guarantee.

    From Audit.lean: sample_complexity applied to audit checks

    Args:
        epsilon: Target margin (default 5%)
        delta: Confidence parameter (default 95% confidence)
        check_types: Which checks to include (default all)

    Returns:
        Dict mapping check type to required samples
    """
    if check_types is None:
        check_types = ["sufficiency", "merge", "idempotence", "substitution"]

    n = sample_complexity(epsilon, delta)
    return {check_type: n for check_type in check_types}


# =============================================================================
# Guarantee Levels (From PreferenceLearning.lean / Audit.lean)
# =============================================================================

class GuaranteeLevel(Enum):
    """
    Three levels of preference learning guarantees from formal proofs.

    From PreferenceLearning.lean and Audit.lean:

    These guarantees apply to ANY oracle-measurable preference learning method,
    including DPO, GRPO, PPO, RLHF, etc. The key insight is that when local laws
    hold, training on summarized data is equivalent to training on original data.

    Level 1 (EXACT): When local laws L1, L2, L3 hold exactly, preference learning gap = 0.
        This applies to all methods: DPO, GRPO, PPO, etc. (PreferenceLearning.lean:
        preference_learning_equivalence)

    Level 2 (UNION_BOUND): Quantitative bound from violation rates. The gap is bounded
        by a Lipschitz constant times the expected distortion. (PreferenceLearning.lean:
        preference_learning_gap_bound)

    Level 3 (EMPIRICAL): Probabilistic (epsilon, delta) guarantee via sampling.
        Audit statistics provide high-probability bounds on actual violation rates.
        (Audit.lean: hoeffding_bound)
    """
    EXACT = "exact"           # Level 1: Local laws hold exactly, preference learning gap = 0
    UNION_BOUND = "union"     # Level 2: Quantitative bound from violation rates
    EMPIRICAL = "empirical"   # Level 3: Probabilistic (epsilon, delta) guarantee


# =============================================================================
# Score-Centric Oracles (ScoringOracle Protocol)
# =============================================================================

class SimpleScorer:
    """
    Simple word-overlap scorer implementing ScoringOracle.

    This is the preferred API for new code. Returns OracleScore with
    score (1.0 = good) as primary output.

    Example:
        scorer = SimpleScorer()
        result = scorer.score(text_a, text_b, rubric)
        print(result.score)  # 0.85
        print(result.passes_threshold(0.8))  # True
    """

    def score(
        self,
        input_a: str,
        input_b: str,
        rubric: str,
    ) -> OracleScore:
        """
        Score similarity between two inputs using word overlap.

        Args:
            input_a: First input text
            input_b: Second input text
            rubric: Comparison criteria (unused in simple implementation)

        Returns:
            OracleScore with similarity score (1.0 = identical)
        """
        words_a = set(input_a.lower().split())
        words_b = set(input_b.lower().split())

        if not words_a or not words_b:
            return OracleScore(
                score=1.0,
                reasoning="Empty input(s)",
            )

        intersection = words_a & words_b
        union = words_a | words_b

        similarity = len(intersection) / len(union) if union else 1.0

        return OracleScore(
            score=similarity,
            reasoning=f"Word overlap: {similarity:.2%}",
        )


# =============================================================================
# Test Oracle Implementations
# =============================================================================

class AlwaysPassScorer:
    """
    Scorer that always returns a perfect score - useful for testing.

    Implements ScoringOracle protocol.
    """

    def score(
        self,
        input_a: str,
        input_b: str,
        rubric: str,
    ) -> OracleScore:
        return OracleScore(score=1.0, reasoning="Always pass scorer")


class AlwaysFailScorer:
    """
    Scorer that always returns zero score - useful for testing.

    Implements ScoringOracle protocol.
    """

    def score(
        self,
        input_a: str,
        input_b: str,
        rubric: str,
    ) -> OracleScore:
        return OracleScore(score=0.0, reasoning="Always fail scorer")


class SamplingStrategy(Enum):
    """Strategy for selecting nodes to audit."""
    RANDOM = "random"              # Uniform random sampling
    LEVEL_WEIGHTED = "level_weighted"  # Prefer higher levels (more compression)
    PRIORITY = "priority"          # Use node priority scores
    CONTENT_WEIGHTED = "content_weighted"  # VLM-derived info scores as PPS weights


# Summarizer protocol and utilities imported from shared module
from treepo._research.core.protocols import Summarizer, format_merge_input


@dataclass
class AuditConfig:
    """Configuration for the auditor."""

    # Sampling parameters
    sample_budget: int = 10
    sampling_strategy: SamplingStrategy = SamplingStrategy.RANDOM
    sampling_probability: float = 1.0  # For probabilistic sampling

    # Content-weighted sampling (VLM-derived information scores)
    content_weights: Optional[Dict[str, float]] = None  # node_id → info score
    content_weight_concentration: float = 2.0  # α exponent: weight_i = info_score_i^α
    content_weight_propensity_floor: float = 0.01  # minimum inclusion probability

    # Thresholds
    discrepancy_threshold: float = 0.1

    # Flags
    audit_leaves: bool = True
    audit_internal: bool = True
    prioritize_high_levels: bool = True

    # Idempotence and substitution checks (from paper Section 4.1)
    audit_idempotence: bool = True  # Check if re-summarizing summaries preserves oracle (C2)
    audit_substitution: bool = True  # Check leaf boundary substitution consistency (C3 Case A)
    idempotence_budget: int = 5  # Number of summaries to sample for idempotence check
    substitution_budget: int = 5  # Number of leaf boundaries to sample for substitution check

    # Seed for reproducibility
    random_seed: Optional[int] = None

    # Concurrency settings (uses centralized config)
    concurrency: Optional[ConcurrencyConfig] = None

    # Statistical guarantee parameters (optional, from Audit.lean)
    target_epsilon: Optional[float] = None  # Target margin (e.g., 0.05 for 5% error)
    target_delta: float = 0.05  # Confidence parameter (default 95% confidence)

    def get_concurrency(self) -> ConcurrencyConfig:
        """Get concurrency config, using default if not set."""
        return self.concurrency or get_concurrency_config()

    def compute_sample_budget_for_guarantee(self) -> int:
        """
        Compute sample budget needed for target (epsilon, delta) guarantee.

        From Audit.lean: sample_complexity

        Returns:
            Minimum samples for guarantee if target_epsilon set, else sample_budget
        """
        if self.target_epsilon is None:
            return self.sample_budget
        return sample_complexity(self.target_epsilon, self.target_delta)


@dataclass
class AuditCheckResult:
    """Result of a single audit check."""
    node_id: str
    check_type: str  # compatibility shim: theorem laws + explicit drift diagnostics
    passed: bool
    discrepancy_score: float
    reasoning: str
    input_a: str = ""
    input_b: str = ""
    # Skipped checks (e.g., no summarizer configured)
    skipped: bool = False
    skip_reason: Optional[str] = None
    # Inclusion probability under the sampling design that produced this check.
    inclusion_probability: Optional[float] = None
    sampling_design: Optional[str] = None

    @property
    def was_evaluated(self) -> bool:
        """True if the check was actually performed (not skipped)."""
        return not self.skipped


@dataclass(frozen=True)
class SampledUnit:
    """Sampled unit with design-time inclusion probability."""
    item: Any
    inclusion_probability: float


@dataclass
class AuditReport:
    """Complete audit report for a tree."""
    tree_id: str
    total_nodes: int
    nodes_audited: int
    nodes_passed: int
    nodes_failed: int
    failure_rate: float
    source_doc_id: Optional[str] = None
    checks: List[AuditCheckResult] = field(default_factory=list)
    failed_node_ids: List[str] = field(default_factory=list)

    # Violation rates by check type (from paper Section 4.1)
    sufficiency_violations: int = 0  # p_suff: leaf sufficiency failures
    merge_violations: int = 0  # p_merge: internal merge failures
    idempotence_violations: int = 0  # p_idem: idempotence failures (C2)
    substitution_violations: int = 0  # p_bound: leaf boundary substitution failures

    # Sample counts for computing rates
    sufficiency_samples: int = 0
    merge_samples: int = 0
    idempotence_samples: int = 0
    substitution_samples: int = 0

    # Population and sampling metadata for IPW reconstruction
    leaf_population: int = 0
    merge_population: int = 0
    idempotence_population: int = 0
    substitution_population: int = 0
    sampling_strategy: str = SamplingStrategy.RANDOM.value
    sampling_probability: float = 1.0
    operator_capabilities: Dict[str, Any] = field(default_factory=dict)
    compositional_learning_problem: Dict[str, Any] = field(default_factory=dict)
    logged_observations: List[LoggedLabelObservation[Any]] = field(default_factory=list)
    logged_observation_artifacts: Dict[str, Any] = field(default_factory=dict)

    # Design-time inclusion probabilities for ALL auditable nodes (not just sampled).
    # Populated by CONTENT_WEIGHTED and LEVEL_WEIGHTED strategies for downstream use.
    inclusion_probability_map: Dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Overall audit passed (no failures)."""
        return self.nodes_failed == 0

    @property
    def sufficiency_rate(self) -> float:
        """Empirical sufficiency violation rate (p_suff)."""
        return self.sufficiency_violations / self.sufficiency_samples if self.sufficiency_samples > 0 else 0.0

    @property
    def merge_rate(self) -> float:
        """Empirical merge violation rate (p_merge)."""
        return self.merge_violations / self.merge_samples if self.merge_samples > 0 else 0.0

    @property
    def idempotence_rate(self) -> float:
        """Empirical idempotence violation rate (p_idem)."""
        return self.idempotence_violations / self.idempotence_samples if self.idempotence_samples > 0 else 0.0

    @property
    def substitution_rate(self) -> float:
        """Empirical substitution violation rate (p_bound)."""
        return self.substitution_violations / self.substitution_samples if self.substitution_samples > 0 else 0.0

    @property
    def assoc_rate(self) -> float:
        """
        Combined audit-decomposition violation rate (p_assoc).

        Weighted average of substitution (leaf boundary) and theorem-merge rates.
        This is an audit-level aggregation, not a separate Lean local law.
        """
        total = self.substitution_samples + self.merge_samples
        if total == 0:
            return 0.0
        lambda_weight = self.substitution_samples / total
        return lambda_weight * self.substitution_rate + (1 - lambda_weight) * self.merge_rate

    def _inclusion_probability_for_check(self, check_type: str) -> Optional[float]:
        """
        Approximate inclusion probability for a check under the configured design.

        Exact inclusion probabilities are available for uniform random sampling.
        For level-weighted or custom policies, this method returns None.
        """
        if self.sampling_strategy != SamplingStrategy.RANDOM.value:
            return None

        probability_map = {
            "sufficiency": (self.sufficiency_samples, self.leaf_population),
            "merge_consistency": (self.merge_samples, self.merge_population),
            "idempotence": (self.idempotence_samples, self.idempotence_population),
            "substitution": (self.substitution_samples, self.substitution_population),
        }
        sampled, population = probability_map.get(check_type, (0, 0))
        if population <= 0 or sampled <= 0:
            return None
        base = min(1.0, sampled / population)
        return max(0.0, min(1.0, base * self.sampling_probability))

    def to_tree_samples(self) -> List["TreeSample"]:
        """
        Convert audit checks to TreeSample records for TreeIPW estimators.

        Skipped checks are excluded by construction.
        """
        from treepo._research.tree.ipw import TreeSample

        observations = list(self.logged_observations or [])
        if not observations:
            observations = self._build_logged_observations()
        doc_key = str(self.source_doc_id) if self.source_doc_id else None
        samples: List[TreeSample] = []
        for observation in observations:
            sample = TreeSample.from_logged_observation(
                observation,
                violation=int(observation.label),
                preference_loss=float(
                    max(0.0, min(1.0, observation.context.get("discrepancy_score", 0.0)))
                ),
                metadata={"check_type": observation.target_name},
            )
            if doc_key is not None:
                sample.doc_id = doc_key
            samples.append(sample)
        return samples

    def _build_logged_observations(self) -> List[LoggedLabelObservation[Any]]:
        check_to_kind = {
            "sufficiency": ObservationUnitKind.LEAF,
            "merge_consistency": ObservationUnitKind.MERGE,
            "idempotence": ObservationUnitKind.RESUMMARY,
            "substitution": ObservationUnitKind.SUBSTITUTION,
        }
        check_to_law_kind = {
            "sufficiency": LawKind.L1_LEAF,
            "merge_consistency": LawKind.L2_MERGE,
            "idempotence": LawKind.L3_IDEMPOTENCE,
        }
        doc_key = str(self.source_doc_id) if self.source_doc_id else self.tree_id
        observations: List[LoggedLabelObservation[Any]] = []
        for check in self.checks:
            if check.skipped:
                continue
            unit_kind = check_to_kind.get(check.check_type)
            if unit_kind is None:
                continue
            inclusion_prob = check.inclusion_probability
            if inclusion_prob is None:
                inclusion_prob = self._inclusion_probability_for_check(check.check_type) or 1.0
            observations.append(
                shared_logged_substructure_observation(
                    document_id=doc_key,
                    unit_id=check.node_id,
                    unit_kind=unit_kind,
                    label=int(not check.passed),
                    application_name="tree_audit_verification",
                    supervision_signal_name=check.check_type,
                    truth_label_source=ORACLE_SOURCE,
                    law_kind=check_to_law_kind.get(check.check_type),
                    sampling=SamplingMetadata(
                        document_propensity=1.0,
                        unit_propensity=float(inclusion_prob),
                        label_propensity=1.0,
                        sampling_scheme=self.sampling_strategy,
                        policy_name="sampled_substructure_query_policy",
                        unit_kind=unit_kind,
                        supports_ipw_estimation=True,
                    ),
                    context={
                        "check_type": check.check_type,
                        "passed": bool(check.passed),
                        "discrepancy_score": float(check.discrepancy_score),
                        "reasoning": check.reasoning,
                        "input_a": check.input_a,
                        "input_b": check.input_b,
                    },
                )
            )
        return observations

    def ipw_violation_rate(self, node_type: Optional[str] = None) -> float:
        """IPW/Hajek violation rate estimate from audit checks."""
        from treepo._research.tree.ipw import NodeType, ipw_violation_rate

        samples = self.to_tree_samples()
        parsed_type = None
        if node_type is not None:
            alias_map = {
                "sufficiency": NodeType.LEAF,
                "leaf": NodeType.LEAF,
                "merge_consistency": NodeType.MERGE,
                "merge": NodeType.MERGE,
                "idempotence": NodeType.RESUMMARY,
                "resummary": NodeType.RESUMMARY,
                "substitution": NodeType.SUBSTITUTION,
            }
            parsed_type = alias_map.get(str(node_type).lower())
            if parsed_type is None:
                parsed_type = NodeType(node_type)
        return ipw_violation_rate(samples, node_type=parsed_type)

    def ipw_violation_empirical_bernstein_ci(
        self,
        delta: float = 0.05,
        node_type: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Empirical-Bernstein confidence interval for IPW violation rate."""
        from treepo._research.tree.ipw import NodeType, ipw_violation_empirical_bernstein_ci

        samples = self.to_tree_samples()
        parsed_type = None
        if node_type is not None:
            alias_map = {
                "sufficiency": NodeType.LEAF,
                "leaf": NodeType.LEAF,
                "merge_consistency": NodeType.MERGE,
                "merge": NodeType.MERGE,
                "idempotence": NodeType.RESUMMARY,
                "resummary": NodeType.RESUMMARY,
                "substitution": NodeType.SUBSTITUTION,
            }
            parsed_type = alias_map.get(str(node_type).lower())
            if parsed_type is None:
                parsed_type = NodeType(node_type)
        return ipw_violation_empirical_bernstein_ci(samples, delta=delta, node_type=parsed_type)

    def ipw_preference_loss(self, node_type: Optional[str] = None) -> float:
        """IPW/Hajek preference-loss estimate from audit checks."""
        from treepo._research.tree.ipw import NodeType, ipw_preference_loss

        samples = self.to_tree_samples()
        parsed_type = None
        if node_type is not None:
            alias_map = {
                "sufficiency": NodeType.LEAF,
                "leaf": NodeType.LEAF,
                "merge_consistency": NodeType.MERGE,
                "merge": NodeType.MERGE,
                "idempotence": NodeType.RESUMMARY,
                "resummary": NodeType.RESUMMARY,
                "substitution": NodeType.SUBSTITUTION,
            }
            parsed_type = alias_map.get(str(node_type).lower())
            if parsed_type is None:
                parsed_type = NodeType(node_type)
        return ipw_preference_loss(samples, node_type=parsed_type)

    def ipw_preference_empirical_bernstein_ci(
        self,
        delta: float = 0.05,
        node_type: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Empirical-Bernstein confidence interval for IPW preference loss."""
        from treepo._research.tree.ipw import NodeType, ipw_preference_empirical_bernstein_ci

        samples = self.to_tree_samples()
        parsed_type = None
        if node_type is not None:
            alias_map = {
                "sufficiency": NodeType.LEAF,
                "leaf": NodeType.LEAF,
                "merge_consistency": NodeType.MERGE,
                "merge": NodeType.MERGE,
                "idempotence": NodeType.RESUMMARY,
                "resummary": NodeType.RESUMMARY,
                "substitution": NodeType.SUBSTITUTION,
            }
            parsed_type = alias_map.get(str(node_type).lower())
            if parsed_type is None:
                parsed_type = NodeType(node_type)
        return ipw_preference_empirical_bernstein_ci(samples, delta=delta, node_type=parsed_type)

    def ipw_union_bound(
        self,
        num_leaves: int,
        num_rounds: int = 1,
        num_merges: Optional[int] = None,
    ) -> float:
        """Union bound computed from IPW-estimated component rates."""
        from treepo._research.tree.ipw import ipw_union_bound

        return ipw_union_bound(
            self.to_tree_samples(),
            num_leaves=num_leaves,
            num_merges=num_merges,
            num_rounds=num_rounds,
        )

    def confidence_upper_bound(self, rate_type: str, delta: float = 0.05) -> float:
        """
        Upper bound on true violation rate with confidence 1-delta.

        Uses Hoeffding: p_true <= p_hat + sqrt(ln(2/delta) / (2n))

        From Audit.lean: confidence_margin combined with empirical rates

        Args:
            rate_type: One of "sufficiency", "merge", "idempotence", "substitution", "assoc"
            delta: Confidence parameter (default 0.05 for 95% confidence)

        Returns:
            Upper bound on true rate, or 1.0 if insufficient samples
        """
        rate_map = {
            "sufficiency": (self.sufficiency_rate, self.sufficiency_samples),
            "merge": (self.merge_rate, self.merge_samples),
            "idempotence": (self.idempotence_rate, self.idempotence_samples),
            "substitution": (self.substitution_rate, self.substitution_samples),
            "assoc": (self.assoc_rate, self.substitution_samples + self.merge_samples),
        }
        rate, n = rate_map.get(rate_type, (0.0, 0))
        if n == 0:
            return 1.0
        margin = confidence_margin(delta, n)
        return min(1.0, rate + margin)

    def get_probabilistic_bound(
        self,
        num_leaves: int,
        num_rounds: int = 1,
        delta: float = 0.05,
        num_merges: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Violation bound with probabilistic guarantee.

        Returns (bound, confidence) where:
        - With probability >= confidence, true root violation <= bound
        - Uses upper bounds on each component rate

        From Audit.lean: Three-level guarantee Level 3

        Args:
            num_leaves: Number of leaves (N)
            num_rounds: Re-summarization rounds (R)
            delta: Per-component confidence (0.05 = 95% per component)
            num_merges: Number of merges (defaults to N-1)

        Returns:
            Tuple of (violation_bound, overall_confidence)
        """
        if num_merges is None:
            num_merges = max(0, num_leaves - 1)

        # Use upper bounds with confidence 1-delta for each component
        p_suff_upper = self.confidence_upper_bound("sufficiency", delta)
        p_assoc_upper = self.confidence_upper_bound("assoc", delta)
        p_idem_upper = self.confidence_upper_bound("idempotence", delta)

        # Union bound over leaf, combined audit-decomposition, and idempotence rates.
        bound = (
            num_leaves * p_suff_upper +
            num_merges * p_assoc_upper +
            max(0, num_rounds - 1) * p_idem_upper
        )

        # Overall confidence: 1 - 3*delta (union bound over 3 components)
        overall_confidence = max(0.0, 1 - 3 * delta)

        return min(1.0, bound), overall_confidence

    def get_guarantee_level(self) -> GuaranteeLevel:
        """
        Determine which guarantee level applies based on audit results.

        From Audit.lean: dpo_three_level_guarantees

        Returns:
            EXACT if all violation rates are 0 and we have samples
            UNION_BOUND if we have empirical rates with some violations
            EMPIRICAL if no samples taken (weakest)
        """
        total_violations = (
            self.sufficiency_violations +
            self.merge_violations +
            self.idempotence_violations +
            self.substitution_violations
        )

        if total_violations == 0 and self.nodes_audited > 0:
            return GuaranteeLevel.EXACT
        elif self.nodes_audited > 0:
            return GuaranteeLevel.UNION_BOUND
        else:
            return GuaranteeLevel.EMPIRICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "total_nodes": int(self.total_nodes),
            "nodes_audited": int(self.nodes_audited),
            "nodes_passed": int(self.nodes_passed),
            "nodes_failed": int(self.nodes_failed),
            "failure_rate": float(self.failure_rate),
            "source_doc_id": self.source_doc_id,
            "checks": [
                {
                    "node_id": check.node_id,
                    "check_type": check.check_type,
                    "passed": bool(check.passed),
                    "discrepancy_score": float(check.discrepancy_score),
                    "reasoning": check.reasoning,
                    "input_a": check.input_a,
                    "input_b": check.input_b,
                    "skipped": bool(check.skipped),
                    "skip_reason": check.skip_reason,
                    "inclusion_probability": check.inclusion_probability,
                    "sampling_design": check.sampling_design,
                }
                for check in self.checks
            ],
            "failed_node_ids": list(self.failed_node_ids),
            "sufficiency_violations": int(self.sufficiency_violations),
            "merge_violations": int(self.merge_violations),
            "idempotence_violations": int(self.idempotence_violations),
            "substitution_violations": int(self.substitution_violations),
            "sufficiency_samples": int(self.sufficiency_samples),
            "merge_samples": int(self.merge_samples),
            "idempotence_samples": int(self.idempotence_samples),
            "substitution_samples": int(self.substitution_samples),
            "leaf_population": int(self.leaf_population),
            "merge_population": int(self.merge_population),
            "idempotence_population": int(self.idempotence_population),
            "substitution_population": int(self.substitution_population),
            "sampling_strategy": self.sampling_strategy,
            "sampling_probability": float(self.sampling_probability),
            "operator_capabilities": dict(self.operator_capabilities),
            "compositional_learning_problem": dict(self.compositional_learning_problem),
            "logged_observations": [
                {
                    **observation.to_dict(),
                    "document_id": (
                        str(self.source_doc_id)
                        if self.source_doc_id is not None
                        else observation.document_id
                    ),
                }
                for observation in self.logged_observations
            ],
            "logged_observation_artifacts": dict(self.logged_observation_artifacts),
            "inclusion_probability_map": dict(self.inclusion_probability_map),
        }


class ReviewPriority(Enum):
    """Priority levels for review items."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class FlaggedItem:
    """
    An item flagged for human or oracle review.

    Contains all information needed for batch processing of reviews.
    """
    # Identity
    item_id: str
    node_id: str
    tree_id: str

    # Check details
    check_type: str  # "sufficiency" or "merge_consistency"
    input_a: str     # Original/source content
    input_b: str     # Summary/target content
    rubric: str      # Information preservation criteria

    # Audit results from approximate oracle
    approx_discrepancy: float
    approx_reasoning: str

    # Metadata
    priority: ReviewPriority = ReviewPriority.MEDIUM
    flagged_at: str = field(default_factory=lambda: datetime.now().isoformat())
    node_level: int = 0

    # Review results (filled in after human/oracle review)
    reviewed: bool = False
    review_result: Optional[bool] = None  # True = approved, False = needs fix
    review_reasoning: Optional[str] = None
    corrected_summary: Optional[str] = None
    reviewed_at: Optional[str] = None
    review_source: str = "human"  # "human" or "oracle_func_auto" - use to filter training data

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "item_id": self.item_id,
            "node_id": self.node_id,
            "tree_id": self.tree_id,
            "check_type": self.check_type,
            "input_a": self.input_a,
            "input_b": self.input_b,
            "rubric": self.rubric,
            "approx_discrepancy": self.approx_discrepancy,
            "approx_reasoning": self.approx_reasoning,
            "priority": self.priority.name,
            "flagged_at": self.flagged_at,
            "node_level": self.node_level,
            "reviewed": self.reviewed,
            "review_result": self.review_result,
            "review_reasoning": self.review_reasoning,
            "corrected_summary": self.corrected_summary,
            "reviewed_at": self.reviewed_at,
            "review_source": self.review_source
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FlaggedItem':
        """Create from dictionary."""
        data = dict(data)
        data["priority"] = ReviewPriority[data.get("priority", "MEDIUM")]
        return cls(**data)


class ReviewQueue:
    """
    Queue for collecting flagged items for batch human/oracle review.

    Supports:
    - Adding flagged items from audit failures
    - Prioritization by level and discrepancy score
    - Batch export for processing
    - Import of review results

    Example:
        >>> queue = ReviewQueue()
        >>> auditor = Auditor(oracle, config, review_queue=queue)
        >>> auditor.audit_tree(tree)
        >>> batch = queue.get_batch(limit=10)
        >>> # Process batch with human reviewers or exact oracle
        >>> for item in batch:
        ...     item.reviewed = True
        ...     item.review_result = True  # Approved
        >>> queue.import_results(batch)
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize review queue.

        Args:
            max_size: Maximum items to hold in queue
        """
        self.max_size = max_size
        self._items: Dict[str, FlaggedItem] = {}
        self._item_counter = 0

    def add(
        self,
        node: Node,
        tree_id: str,
        check_result: AuditCheckResult,
        rubric: str,
        full_input_a: str = "",
        full_input_b: str = ""
    ) -> FlaggedItem:
        """
        Add a flagged item to the queue.

        Args:
            node: The node that failed audit
            tree_id: ID of the tree
            check_result: The audit check result
            rubric: Information preservation rubric
            full_input_a: Full (untruncated) input A
            full_input_b: Full (untruncated) input B

        Returns:
            The created FlaggedItem
        """
        self._item_counter += 1
        item_id = f"flag_{self._item_counter}"

        # Determine priority based on node level and discrepancy
        if check_result.discrepancy_score >= 0.8:
            priority = ReviewPriority.CRITICAL
        elif check_result.discrepancy_score >= 0.5:
            priority = ReviewPriority.HIGH
        elif node.level >= 2:
            priority = ReviewPriority.HIGH
        else:
            priority = ReviewPriority.MEDIUM

        item = FlaggedItem(
            item_id=item_id,
            node_id=node.id,
            tree_id=tree_id,
            check_type=check_result.check_type,
            input_a=full_input_a or check_result.input_a,
            input_b=full_input_b or check_result.input_b,
            rubric=rubric,
            approx_discrepancy=check_result.discrepancy_score,
            approx_reasoning=check_result.reasoning,
            priority=priority,
            node_level=node.level
        )

        # Enforce max size (remove lowest priority items)
        if len(self._items) >= self.max_size:
            self._evict_lowest_priority()

        self._items[item_id] = item
        return item

    def _evict_lowest_priority(self) -> None:
        """Remove lowest priority item to make room."""
        if not self._items:
            return
        # Sort by priority (ascending) and remove first
        sorted_items = sorted(
            self._items.values(),
            key=lambda x: (x.priority.value, x.approx_discrepancy)
        )
        if sorted_items:
            del self._items[sorted_items[0].item_id]

    def get_batch(
        self,
        limit: int = 10,
        priority_min: ReviewPriority = ReviewPriority.LOW,
        unreviewed_only: bool = True
    ) -> List[FlaggedItem]:
        """
        Get a batch of items for review.

        Args:
            limit: Maximum items to return
            priority_min: Minimum priority level
            unreviewed_only: Only return unreviewed items

        Returns:
            List of FlaggedItems sorted by priority (highest first)
        """
        items = list(self._items.values())

        # Filter
        if unreviewed_only:
            items = [i for i in items if not i.reviewed]
        items = [i for i in items if i.priority.value >= priority_min.value]

        # Sort by priority (descending), then discrepancy (descending)
        items.sort(key=lambda x: (-x.priority.value, -x.approx_discrepancy))

        return items[:limit]

    def get_all(self) -> List[FlaggedItem]:
        """Get all items in the queue."""
        return list(self._items.values())

    @property
    def items(self) -> List[FlaggedItem]:
        """Property to access all items (alias for get_all)."""
        return self.get_all()

    def add_item(self, item: FlaggedItem) -> None:
        """Add a pre-constructed FlaggedItem to the queue."""
        if len(self._items) >= self.max_size:
            self._evict_lowest_priority()
        self._items[item.item_id] = item

    def get_by_id(self, item_id: str) -> Optional[FlaggedItem]:
        """Get a specific item by ID."""
        return self._items.get(item_id)

    def update_item(self, item: FlaggedItem) -> None:
        """Update an item in the queue."""
        if item.item_id in self._items:
            self._items[item.item_id] = item

    def import_results(self, items: List[FlaggedItem]) -> int:
        """
        Import review results back into the queue.

        Args:
            items: List of reviewed FlaggedItems

        Returns:
            Number of items updated
        """
        updated = 0
        for item in items:
            if item.item_id in self._items:
                self._items[item.item_id] = item
                updated += 1
        return updated

    def export_to_json(self, filepath: str) -> None:
        """Export queue to JSON file for external processing."""
        data = {
            "exported_at": datetime.now().isoformat(),
            "item_count": len(self._items),
            "items": [item.to_dict() for item in self._items.values()]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def import_from_json(self, filepath: str) -> int:
        """Import items from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        items = [FlaggedItem.from_dict(d) for d in data.get("items", [])]
        return self.import_results(items)

    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        items = list(self._items.values())
        reviewed = [i for i in items if i.reviewed]
        approved = [i for i in reviewed if i.review_result]

        return {
            "total_items": len(items),
            "pending_review": len(items) - len(reviewed),
            "reviewed": len(reviewed),
            "approved": len(approved),
            "rejected": len(reviewed) - len(approved),
            "by_priority": {
                p.name: len([i for i in items if i.priority == p])
                for p in ReviewPriority
            },
            "avg_discrepancy": (
                sum(i.approx_discrepancy for i in items) / len(items)
                if items else 0.0
            )
        }

    def clear(self) -> None:
        """Clear all items from the queue."""
        self._items.clear()

    def get_reviewed_items(self) -> List[FlaggedItem]:
        """Get all reviewed items (for training data extraction)."""
        return [item for item in self._items.values() if item.reviewed]

    def get_unreviewed_items(self) -> List[FlaggedItem]:
        """Get all unreviewed items."""
        return [item for item in self._items.values() if not item.reviewed]

    def get_items_with_corrections(self) -> List[FlaggedItem]:
        """Get reviewed items that have corrected summaries."""
        return [
            item for item in self._items.values()
            if item.reviewed and item.corrected_summary
        ]

    def auto_review_with_oracle_func(
        self,
        oracle_func_engine: 'OracleFuncReviewEngine',
        auto_apply: bool = True,
        priority_min: 'ReviewPriority' = None
    ) -> List['OracleFuncReviewResult']:
        """
        Review unreviewed items using the learned oracle function approximation.

        This method integrates the oracle function approximation system with the
        review queue, allowing automated review of flagged nodes.

        Args:
            oracle_func_engine: Configured OracleFuncReviewEngine instance
            auto_apply: If True, automatically apply high-confidence decisions
            priority_min: Minimum priority level to review (default: MEDIUM)

        Returns:
            List of OracleFuncReviewResult for each reviewed item
        """
        if priority_min is None:
            priority_min = ReviewPriority.MEDIUM

        return oracle_func_engine.review_flagged_nodes(
            queue=self,
            auto_apply=auto_apply,
            priority_min=priority_min
        )

    def __len__(self) -> int:
        return len(self._items)


class Auditor:
    """
    Auditor for OPS trees.

    Performs probabilistic verification of summarization quality by
    sampling nodes and checking for information preservation.

    Two types of checks:
    1. Sufficiency Check (leaves): Does summary capture rubric info from raw text?
    2. Merge Consistency (internal): Does parent preserve info from children?

    Example:
        >>> scorer = SimpleScorer()
        >>> queue = ReviewQueue()
        >>> auditor = Auditor(scorer, config=AuditConfig(sample_budget=5), review_queue=queue)
        >>> report = auditor.audit_tree(tree)
        >>> print(f"Passed: {report.passed}, Failures: {report.nodes_failed}")
        >>> # Get flagged items for human review
        >>> batch = queue.get_batch(limit=10)
    """

    def __init__(
        self,
        oracle: ScoringOracle,
        config: Optional[AuditConfig] = None,
        review_queue: Optional[ReviewQueue] = None,
        summarizer: Optional[Callable[[str, str], str]] = None,
        theorem_operator: Optional[Any] = None,
    ):
        """
        Initialize the auditor.

        Args:
            oracle: Oracle judge for comparing inputs (ScoringOracle interface).
            config: Audit configuration
            review_queue: Optional queue for flagging failures for batch review
            summarizer: Optional summarizer function for idempotence/substitution checks.
                       Required if audit_idempotence or audit_substitution is True.
            theorem_operator: Optional theorem-facing operator with resummarize/combine support.
        """
        self._oracle = oracle
        self.config = config or AuditConfig()
        self.review_queue = review_queue
        self.summarizer = summarizer
        self.theorem_operator = theorem_operator
        if self.theorem_operator is None and self.summarizer is not None:
            self.theorem_operator = make_text_compositional_operator(
                self.summarizer,
                name="auditor_summary_operator",
            )
        self._last_inclusion_prob_map: Dict[str, float] = {}

        # Use a per-auditor RNG to avoid clobbering the global random state.
        # This is important when auditing many trees in parallel.
        self._rng = random.Random(self.config.random_seed) if self.config.random_seed is not None else random.Random()

    def _call_oracle(self, input_a: str, input_b: str, rubric: str) -> Tuple[bool, float, str]:
        """
        Call the oracle and return (is_congruent, discrepancy, reasoning).

        Requires ScoringOracle interface (has .score() method returning OracleScore).

        Returns:
            Tuple of (is_congruent, discrepancy, reasoning) where:
            - is_congruent: True if discrepancy <= threshold
            - discrepancy: 0.0-1.0 where 0.0 = perfect match (1 - score)
            - reasoning: Explanation from oracle
        """
        result = self._oracle.score(input_a, input_b, rubric)
        # Score is 0.0-1.0 where 1.0 = good; discrepancy is inverse
        discrepancy = 1.0 - result.score
        is_congruent = discrepancy <= self.config.discrepancy_threshold
        return is_congruent, discrepancy, result.reasoning

    def _combine_inputs(self, left: str, right: str, rubric: str) -> str:
        operator = self.theorem_operator
        if operator is not None and hasattr(operator, "combine"):
            return str(operator.combine(left, right, rubric=rubric))
        return format_merge_input(left, right)

    def _resummarize(self, text: str, rubric: str) -> str:
        operator = self.theorem_operator
        if operator is not None:
            if hasattr(operator, "resummarize"):
                return str(operator.resummarize(text, rubric=rubric))
            if hasattr(operator, "encode") and hasattr(operator, "decode"):
                encoded = operator.encode(text, rubric=rubric)
                return str(operator.decode(encoded, rubric=rubric))
        if self.summarizer is None:
            raise RuntimeError("No summarizer or theorem operator configured")
        return str(self.summarizer(text, rubric))

    def _resolve_operator_assumptions(
        self,
    ) -> Tuple[Optional[TheoremAssumptionSpec], Optional[OperatorAssumptionBundle]]:
        candidate = self.theorem_operator
        seen: Set[int] = set()
        while candidate is not None and id(candidate) not in seen:
            seen.add(id(candidate))
            theorem_assumptions = getattr(candidate, "theorem_assumptions", None)
            operator_assumptions = getattr(candidate, "assumptions", None)
            resolved_theorem = (
                theorem_assumptions
                if isinstance(theorem_assumptions, TheoremAssumptionSpec)
                else None
            )
            resolved_operator = (
                operator_assumptions
                if isinstance(operator_assumptions, OperatorAssumptionBundle)
                else None
            )
            if resolved_theorem is not None or resolved_operator is not None:
                return resolved_theorem, resolved_operator
            candidate = getattr(candidate, "operator", None)
        return None, None

    def _compositional_learning_problem(self) -> Dict[str, Any]:
        theorem_assumptions, operator_assumptions = self._resolve_operator_assumptions()
        capability_report = (
            self.theorem_operator.capability_report()
            if self.theorem_operator is not None and hasattr(self.theorem_operator, "capability_report")
            else None
        )
        targeted_laws = [LawKind.L1_LEAF, LawKind.L2_MERGE]
        if bool(self.config.audit_idempotence):
            targeted_laws.append(LawKind.L3_IDEMPOTENCE)
        problem = CompositionalLearningProblemSpec(
            name="tree_audit_verification",
            document_type_name="tree_structured_documents",
            theorem_domain_name="summary_objects",
            operator_name=(
                capability_report.operator_name
                if capability_report is not None
                else (
                    "auditor_summary_operator"
                    if self.theorem_operator is not None or self.summarizer is not None
                    else "no_theorem_operator"
                )
            ),
            theorem_assumptions=theorem_assumptions,
            operator_assumptions=operator_assumptions,
            operator_capabilities=capability_report,
            supervision_channels=(
                shared_sampled_substructure_supervision_channel(
                    active=bool(
                        self.config.audit_leaves
                        or self.config.audit_internal
                        or self.config.audit_idempotence
                        or self.config.audit_substitution
                    ),
                    label_source=ORACLE_SOURCE,
                    delivery_mode=SupervisionDeliveryMode.ONLINE_ORACLE_QUERY,
                    query_policy=shared_sampled_substructure_query_policy(
                        selection_strategy=str(self.config.sampling_strategy.value),
                        adaptive=bool(
                            self.config.sampling_strategy
                            != SamplingStrategy.RANDOM
                        ),
                        budget={
                            "sample_budget": int(
                                self.config.compute_sample_budget_for_guarantee()
                            ),
                            "idempotence_budget": int(self.config.idempotence_budget),
                            "substitution_budget": int(self.config.substitution_budget),
                        },
                        propensity_field_name="inclusion_probability",
                        logs_realized_propensities=True,
                        supports_ipw_estimation=True,
                        notes=(
                            "Auditor queries the oracle online on sampled nodes and logs inclusion probabilities when available.",
                        ),
                    ),
                    targeted_laws=tuple(targeted_laws),
                    requires_propensity_logging=True,
                    supports_unbiased_risk=True,
                    notes=(
                        "Auditor supervision arrives through sampled leaves, merges, and optional resummary checks.",
                        "Substitution checks are also sampled but live outside the strict L1/L2/L3 theorem-law trio.",
                    ),
                ),
            ),
            notes=shared_protocol_problem_notes(
                application_name="tree_audit_verification",
                notes=(
                "This manifest records the audit problem itself, not only the realized sampled checks.",
                "When a theorem operator is attached, the capability surface is copied into the problem spec.",
                ),
            ),
        )
        return problem.to_dict()

    def audit_tree(self, tree: Tree) -> AuditReport:
        """
        Audit an OPS tree.

        Samples nodes and checks for information preservation. Failures
        are automatically flagged to the review queue if one is configured.

        Performs four types of checks (from paper Section 4.1):
        1. Sufficiency (C1): Does leaf summary preserve oracle info from raw text?
        2. Merge Consistency (C3 Case B): Does internal merge preserve oracle info?
        3. Idempotence (C2): Does re-summarizing a summary keep oracle unchanged?
        4. Substitution (C3 Case A): Do joint vs disjoint summary paths agree?

        Args:
            tree: The tree to audit

        Returns:
            AuditReport with results
        """
        # Reset per-audit inclusion probability map for this tree
        self._last_inclusion_prob_map = {}

        tree_id = tree.root.id if tree.root else "unknown"
        rubric = tree.rubric
        source_doc_id = None
        if isinstance(tree.metadata, dict):
            raw_doc_id = tree.metadata.get("doc_id")
            if raw_doc_id is not None:
                source_doc_id = str(raw_doc_id)

        all_nodes = list(tree.traverse_preorder())
        leaves = [n for n in all_nodes if n.is_leaf]
        internal = [n for n in all_nodes if not n.is_leaf]
        substitution_population = max(0, len(leaves) - 1)

        checks = []
        audited_ids: Set[str] = set()

        # Track violation counts for computing rates
        sufficiency_violations = 0
        sufficiency_samples = 0
        merge_violations = 0
        merge_samples = 0
        idempotence_violations = 0
        idempotence_samples = 0
        substitution_violations = 0
        substitution_samples = 0

        # Determine how to split budget between leaves and internal nodes
        # Use computed budget from (epsilon, delta) if target_epsilon is set,
        # otherwise fall back to sample_budget. From Audit.lean: sample_complexity.
        effective_budget = self.config.compute_sample_budget_for_guarantee()
        if self.config.target_epsilon is not None:
            logger.info(
                f"Using computed sample budget {effective_budget} for "
                f"(epsilon={self.config.target_epsilon}, delta={self.config.target_delta}) guarantee"
            )
        leaf_budget = effective_budget // 2 if self.config.audit_internal else effective_budget
        internal_budget = effective_budget - leaf_budget

        # Audit leaves (sufficiency check - C1) - concurrent for better GPU utilization
        if self.config.audit_leaves and leaves:
            leaf_samples = self._sample_nodes(leaves, leaf_budget)
            leaf_results = self._batch_audit_nodes(leaf_samples, self._check_sufficiency, rubric)
            for result, full_a, full_b, node in leaf_results:
                checks.append(result)
                audited_ids.add(node.id)
                self._update_node_audit(node, result, tree_id, rubric, full_a, full_b)
                sufficiency_samples += 1
                if not result.passed:
                    sufficiency_violations += 1

        # Audit internal nodes using Lean L2 / paper C3 semantics - concurrent
        if self.config.audit_internal and internal:
            internal_samples = self._sample_nodes(internal, internal_budget)
            internal_results = self._batch_audit_nodes(internal_samples, self._check_merge_consistency, rubric)
            for result, full_a, full_b, node in internal_results:
                checks.append(result)
                audited_ids.add(node.id)
                self._update_node_audit(node, result, tree_id, rubric, full_a, full_b)
                merge_samples += 1
                if not result.passed:
                    merge_violations += 1

        # Idempotence check (C2) - Re-summarize summaries and check oracle stability - concurrent
        if self.config.audit_idempotence and internal and self.summarizer is not None:
            idem_samples = self._sample_nodes(internal, self.config.idempotence_budget)
            gated_idem_samples = self._apply_probability_gate(idem_samples)

            def check_idempotence(sampled: SampledUnit):
                node = sampled.item
                result = self._check_idempotence(node, rubric)
                result.inclusion_probability = sampled.inclusion_probability
                result.sampling_design = self.config.sampling_strategy.value
                return result

            audit_workers = self.config.get_concurrency().audit_max_workers
            with ThreadPoolExecutor(max_workers=audit_workers) as executor:
                idem_results = list(executor.map(check_idempotence, gated_idem_samples))

            for result in idem_results:
                checks.append(result)
                idempotence_samples += 1
                if not result.passed:
                    idempotence_violations += 1
        elif self.config.audit_idempotence and internal and self.summarizer is None and self.theorem_operator is None:
            # WARNING: Idempotence (C2/L3) is required for multi-round preservation (Theorem 2).
            # Without a summarizer, this check cannot be performed. If the tree was built
            # with R > 1 rounds, the theoretical guarantees from PreservationTheorems.lean
            # (multi_round theorem) may not hold.
            logger.warning(
                "Idempotence check (C2/L3) requested but no summarizer or theorem operator provided. "
                "This check is REQUIRED for multi-round preservation guarantees (Theorem 2 in Lean). "
                "If tree was built with R > 1 rounds, oracle preservation cannot be verified. "
                "Provide a summarizer or theorem operator to enable idempotence checking."
            )

        # Substitution check (C3 Case A) - Check leaf boundary consistency - concurrent
        if self.config.audit_substitution and len(leaves) >= 2 and (
            self.summarizer is not None or self.theorem_operator is not None
        ):
            # Get adjacent leaf pairs
            adjacent_pairs = self._get_adjacent_leaf_pairs(leaves)
            if adjacent_pairs:
                sampled_pairs = self._sample_adjacent_pairs(adjacent_pairs, self.config.substitution_budget)
                gated_pairs = self._apply_probability_gate(sampled_pairs)

                def check_substitution(sampled: SampledUnit):
                    left_node, right_node = sampled.item
                    return self._check_substitution(left_node, right_node, rubric)

                sub_workers = self.config.get_concurrency().audit_max_workers
                with ThreadPoolExecutor(max_workers=sub_workers) as executor:
                    sub_results = list(executor.map(check_substitution, gated_pairs))

                for sampled, result in zip(gated_pairs, sub_results):
                    result.inclusion_probability = sampled.inclusion_probability
                    result.sampling_design = "adjacent_uniform"
                    checks.append(result)
                    substitution_samples += 1
                    if not result.passed:
                        substitution_violations += 1
        elif self.config.audit_substitution and len(leaves) >= 2 and self.summarizer is None and self.theorem_operator is None:
            # WARNING: Substitution check (C3 Case A) requires a summarizer to compare
            # joint vs disjoint summarization paths.
            logger.warning(
                "Substitution check (C3 Case A) requested but no summarizer or theorem operator provided. "
                "This check verifies joint vs disjoint path consistency (L2 in Lean). "
                "Provide a summarizer or theorem operator to enable substitution checking."
            )

        # Compile report
        passed = sum(1 for c in checks if c.passed)
        failed = len(checks) - passed
        failed_ids = [c.node_id for c in checks if not c.passed]

        report = AuditReport(
            tree_id=tree_id,
            source_doc_id=source_doc_id,
            total_nodes=len(all_nodes),
            nodes_audited=len(checks),
            nodes_passed=passed,
            nodes_failed=failed,
            failure_rate=failed / len(checks) if checks else 0.0,
            checks=checks,
            failed_node_ids=failed_ids,
            sufficiency_violations=sufficiency_violations,
            merge_violations=merge_violations,
            idempotence_violations=idempotence_violations,
            substitution_violations=substitution_violations,
            sufficiency_samples=sufficiency_samples,
            merge_samples=merge_samples,
            idempotence_samples=idempotence_samples,
            substitution_samples=substitution_samples,
            leaf_population=len(leaves),
            merge_population=len(internal),
            idempotence_population=len(internal),
            substitution_population=substitution_population,
            sampling_strategy=self.config.sampling_strategy.value,
            sampling_probability=self.config.sampling_probability,
            operator_capabilities=(
                self.theorem_operator.capability_report().to_dict()
                if self.theorem_operator is not None and hasattr(self.theorem_operator, "capability_report")
                else {}
            ),
            compositional_learning_problem=self._compositional_learning_problem(),
            logged_observations=[],
            logged_observation_artifacts={},
            inclusion_probability_map=dict(self._last_inclusion_prob_map),
        )
        report.logged_observations = report._build_logged_observations()
        return report

    def _batch_audit_nodes(
        self,
        sampled_nodes: List[SampledUnit],
        check_fn: Callable,
        rubric: str,
        max_workers: Optional[int] = None
    ) -> List[Tuple["AuditCheckResult", str, str, Node]]:
        """
        Run audit checks on nodes concurrently for better GPU utilization.

        Args:
            sampled_nodes: Sampled nodes with inclusion probabilities
            check_fn: Check function (e.g., _check_sufficiency or _check_merge_consistency)
            rubric: Rubric for the oracle
            max_workers: Maximum concurrent workers (uses config default if None)

        Returns:
            List of (result, full_a, full_b, node) tuples for nodes that were audited
        """
        results = []
        gated_samples = self._apply_probability_gate(sampled_nodes)

        def check_node(sampled: SampledUnit):
            node = sampled.item
            result, full_a, full_b = check_fn(node, rubric)
            result.inclusion_probability = sampled.inclusion_probability
            result.sampling_design = self.config.sampling_strategy.value
            return (result, full_a, full_b, node)

        # Use concurrency config if max_workers not explicitly set
        if max_workers is None:
            max_workers = self.config.get_concurrency().audit_max_workers

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(check_node, sampled) for sampled in gated_samples]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        return results

    def _sample_nodes(self, nodes: List[Node], budget: int) -> List[SampledUnit]:
        """
        Sample nodes according to the configured strategy.

        As a side effect, populates ``self._last_inclusion_prob_map`` with
        design-time inclusion probabilities for ALL nodes (not just sampled).
        This is used to build ``AuditReport.inclusion_probability_map``.

        Args:
            nodes: Nodes to sample from
            budget: Maximum nodes to sample

        Returns:
            List of sampled nodes with pre-gate inclusion probabilities.
        """
        if not nodes or budget <= 0:
            return []

        budget = min(budget, len(nodes))

        if self.config.sampling_strategy == SamplingStrategy.RANDOM:
            sampled = self._rng.sample(nodes, budget)
            inclusion_prob = (budget / len(nodes)) if nodes else 0.0
            # Store full map for all nodes
            self._last_inclusion_prob_map.update(
                {n.id: inclusion_prob for n in nodes}
            )
            return [
                SampledUnit(item=node, inclusion_probability=inclusion_prob)
                for node in sampled
            ]

        elif self.config.sampling_strategy == SamplingStrategy.LEVEL_WEIGHTED:
            weights = self._compute_level_weights(nodes)
            inclusion_probs = self._pps_inclusion_probabilities(weights, budget)
            # Store full map for all nodes
            self._last_inclusion_prob_map.update(
                {nodes[i].id: inclusion_probs[i] for i in range(len(nodes))}
            )
            sampled_indices = self._systematic_pps_indices(inclusion_probs, budget)
            return [
                SampledUnit(
                    item=nodes[index],
                    inclusion_probability=inclusion_probs[index],
                )
                for index in sampled_indices
            ]

        elif self.config.sampling_strategy == SamplingStrategy.CONTENT_WEIGHTED:
            weights = self._compute_content_weights(nodes)
            inclusion_probs = self._pps_inclusion_probabilities(weights, budget)
            # Apply propensity floor — ensures no node has zero draw probability.
            floor = max(1e-6, self.config.content_weight_propensity_floor)
            inclusion_probs = [max(floor, p) for p in inclusion_probs]
            # Store full map for all nodes
            self._last_inclusion_prob_map.update(
                {nodes[i].id: inclusion_probs[i] for i in range(len(nodes))}
            )
            sampled_indices = self._systematic_pps_indices(inclusion_probs, budget)
            return [
                SampledUnit(
                    item=nodes[index],
                    inclusion_probability=inclusion_probs[index],
                )
                for index in sampled_indices
            ]

        else:
            # Default to random
            sampled = self._rng.sample(nodes, budget)
            inclusion_prob = (budget / len(nodes)) if nodes else 0.0
            self._last_inclusion_prob_map.update(
                {n.id: inclusion_prob for n in nodes}
            )
            return [
                SampledUnit(item=node, inclusion_probability=inclusion_prob)
                for node in sampled
            ]

    def _apply_probability_gate(self, sampled_units: List[SampledUnit]) -> List[SampledUnit]:
        """
        Apply global audit probability as an independent second-stage gate.

        The returned inclusion probabilities are first-stage inclusion
        probabilities multiplied by `sampling_probability`.
        """
        if not sampled_units:
            return []

        gate_prob = max(0.0, min(1.0, float(self.config.sampling_probability)))
        if gate_prob >= 1.0:
            return sampled_units

        gated: List[SampledUnit] = []
        for sampled in sampled_units:
            if self._rng.random() < gate_prob:
                gated.append(
                    SampledUnit(
                        item=sampled.item,
                        inclusion_probability=min(1.0, sampled.inclusion_probability * gate_prob),
                    )
                )
        return gated

    def _sample_adjacent_pairs(
        self,
        adjacent_pairs: List[Tuple[Node, Node]],
        budget: int,
    ) -> List[SampledUnit]:
        """Uniform sample of adjacent pairs with known first-stage inclusion probabilities."""
        if not adjacent_pairs or budget <= 0:
            return []
        sample_size = min(budget, len(adjacent_pairs))
        sampled = self._rng.sample(adjacent_pairs, sample_size)
        inclusion_prob = (sample_size / len(adjacent_pairs)) if adjacent_pairs else 0.0
        return [
            SampledUnit(item=pair, inclusion_probability=inclusion_prob)
            for pair in sampled
        ]

    @staticmethod
    def _compute_level_weights(nodes: List[Node]) -> List[float]:
        """Priority weights that upweight higher-level (more compressed) nodes."""
        if not nodes:
            return []
        max_level = max(node.level for node in nodes)
        raw = [(node.level + 1) / (max_level + 1) for node in nodes]
        total = sum(raw)
        if total <= 0:
            return [1.0 / len(nodes)] * len(nodes)
        return [weight / total for weight in raw]

    def _compute_content_weights(self, nodes: List[Node]) -> List[float]:
        """
        Compute PPS weights from VLM-derived information scores.

        Uses ``content_weight_concentration`` (α) as power-law exponent::

            weight_i = info_score_i ^ α

        Higher α concentrates budget on high-info segments.
        α=1 is proportional to info_score (linear).
        α=2 heavily favors high-info (quadratic).
        α=0 degenerates to uniform.
        """
        if not nodes:
            return []

        alpha = max(0.0, self.config.content_weight_concentration)
        raw: List[float] = []
        for node in nodes:
            if self.config.content_weights:
                score = self.config.content_weights.get(node.id, 0.5)
            else:
                score = 0.5
            # Power-law transform: concentrate on high-info.
            raw.append(max(1e-8, float(score) ** alpha))

        total = sum(raw)
        if total <= 0:
            return [1.0 / len(nodes)] * len(nodes)
        return [w / total for w in raw]

    @staticmethod
    def _pps_inclusion_probabilities(weights: List[float], sample_size: int) -> List[float]:
        """Fixed-size PPS first-order inclusion probabilities."""
        return pps_inclusion_probabilities(weights, sample_size)

    @staticmethod
    def _systematic_pps_indices(inclusion_probs: List[float], sample_size: int) -> List[int]:
        """Sample fixed-size without replacement using systematic PPS."""
        return systematic_pps_sample_indices(inclusion_probs, sample_size)

    def _check_sufficiency(
        self, node: Node, rubric: str
    ) -> Tuple[AuditCheckResult, str, str]:
        """
        Check if a leaf node's summary is sufficient.

        Compares raw text to summary.

        Args:
            node: Leaf node to check
            rubric: Information preservation rubric

        Returns:
            Tuple of (AuditCheckResult, full_input_a, full_input_b)
        """
        if not node.is_leaf:
            logger.warning(f"Sufficiency check called on non-leaf node {node.id}")

        input_a = node.raw_text_span or ""
        input_b = node.summary

        is_congruent, score, reasoning = self._call_oracle(input_a, input_b, rubric)
        passed = is_congruent and score <= self.config.discrepancy_threshold

        result = AuditCheckResult(
            node_id=node.id,
            check_type="sufficiency",
            passed=passed,
            discrepancy_score=score,
            reasoning=reasoning,
            input_a=input_a,
            input_b=input_b
        )
        return result, input_a, input_b

    def _check_merge_consistency(
        self, node: Node, rubric: str
    ) -> Tuple[AuditCheckResult, str, str]:
        """
        Check Lean L2 / paper C3 against the theorem-domain node span.

        Args:
            node: Internal node to check
            rubric: Information preservation rubric

        Returns:
            Tuple of (AuditCheckResult, full_input_a, full_input_b)
        """
        if node.is_leaf:
            logger.warning(f"Merge check called on leaf node {node.id}")

        if node.ops_span:
            input_a = node.ops_span
        else:
            left_span = (
                (node.left_child.ops_span if node.left_child else None)
                or (node.left_child.raw_text_span if node.left_child else None)
                or (node.left_child.summary if node.left_child else "")
            )
            right_span = (
                (node.right_child.ops_span if node.right_child else None)
                or (node.right_child.raw_text_span if node.right_child else None)
                or (node.right_child.summary if node.right_child else "")
            )
            input_a = self._combine_inputs(left_span, right_span, rubric)
        input_b = node.summary

        is_congruent, score, reasoning = self._call_oracle(input_a, input_b, rubric)
        passed = is_congruent and score <= self.config.discrepancy_threshold

        result = AuditCheckResult(
            node_id=node.id,
            check_type="merge_consistency",
            passed=passed,
            discrepancy_score=score,
            reasoning=reasoning,
            input_a=input_a,
            input_b=input_b
        )
        return result, input_a, input_b

    def _check_joint_to_disjoint_drift(
        self, node: Node, rubric: str
    ) -> Tuple[AuditCheckResult, str, str]:
        """
        Non-theorem diagnostic: compare parent summary to concatenated child summaries.
        """
        if node.is_leaf:
            logger.warning(f"Joint/disjoint drift check called on leaf node {node.id}")

        left_summary = node.left_child.summary if node.left_child else ""
        right_summary = node.right_child.summary if node.right_child else ""
        input_a = self._combine_inputs(left_summary, right_summary, rubric)
        input_b = node.summary

        is_congruent, score, reasoning = self._call_oracle(input_a, input_b, rubric)
        passed = is_congruent and score <= self.config.discrepancy_threshold
        result = AuditCheckResult(
            node_id=node.id,
            check_type="joint_to_disjoint_drift",
            passed=passed,
            discrepancy_score=score,
            reasoning=reasoning,
            input_a=input_a,
            input_b=input_b,
        )
        return result, input_a, input_b

    def _check_idempotence(
        self, node: Node, rubric: str
    ) -> AuditCheckResult:
        """
        Check if re-summarizing a summary preserves the oracle (Condition C2).

        From paper Section 4.1 (Sampling Stability):
        p_idem := (1/n_s) * sum I{d_Y(f*(g(z_k)), f*(z_k)) > τ}

        This samples summaries that have already passed through g and checks
        if re-summarization alters the oracle.

        Args:
            node: Internal node whose summary to re-summarize
            rubric: Information preservation rubric

        Returns:
            AuditCheckResult for the idempotence check
        """
        if self.summarizer is None and self.theorem_operator is None:
            logger.warning("Idempotence check requires summarizer or theorem operator to be configured")
            return AuditCheckResult(
                node_id=node.id,
                check_type="idempotence",
                passed=False,  # NOT passed - check wasn't performed
                discrepancy_score=0.0,
                reasoning="Skipped: no summarizer or theorem operator configured",
                skipped=True,
                skip_reason="no_summarizer",
            )

        # The original summary s
        original_summary = node.summary

        # Re-summarize: g(s, rubric)
        try:
            re_summarized = self._resummarize(original_summary, rubric)
        except Exception as e:
            logger.error(f"Summarizer failed during idempotence check: {e}")
            return AuditCheckResult(
                node_id=node.id,
                check_type="idempotence",
                passed=False,
                discrepancy_score=1.0,
                reasoning=f"Summarizer error: {e}"
            )

        # Compare f*(s) vs f*(g(s))
        is_congruent, score, reasoning = self._call_oracle(original_summary, re_summarized, rubric)
        passed = is_congruent and score <= self.config.discrepancy_threshold

        return AuditCheckResult(
            node_id=node.id,
            check_type="idempotence",
            passed=passed,
            discrepancy_score=score,
            reasoning=f"Idempotence: {reasoning}",
            input_a=original_summary,
            input_b=re_summarized
        )

    def _check_substitution(
        self, left_node: Node, right_node: Node, rubric: str
    ) -> AuditCheckResult:
        """
        Check leaf boundary substitution consistency (Condition C3 Case A).

        From paper Section 4.1 (Sampling Merge Consistency - Case A):
        When u, v are adjacent raw blocks, u ⊕ v fits in context.
        Compare the joint and disjoint summaries:
        I_bound := I{d_Y(f*(g(u⊕v)), f*(g(g(u)⊕g(v)))) > τ}

        This tests whether summarizing the joint raw span gives the same
        oracle result as first summarizing each part then merging.

        Args:
            left_node: Left leaf node in the adjacent pair
            right_node: Right leaf node in the adjacent pair
            rubric: Information preservation rubric

        Returns:
            AuditCheckResult for the substitution check
        """
        if self.summarizer is None and self.theorem_operator is None:
            logger.warning("Substitution check requires summarizer or theorem operator to be configured")
            return AuditCheckResult(
                node_id=f"{left_node.id}+{right_node.id}",
                check_type="substitution",
                passed=False,  # NOT passed - check wasn't performed
                discrepancy_score=0.0,
                reasoning="Skipped: no summarizer or theorem operator configured",
                skipped=True,
                skip_reason="no_summarizer",
            )

        # Get raw text spans
        raw_left = left_node.raw_text_span or ""
        raw_right = right_node.raw_text_span or ""

        # Joint path: g(u ⊕ v, rubric) - summarize the concatenated raw text directly
        joint_raw = self._combine_inputs(raw_left, raw_right, rubric)
        try:
            joint_summary = self._resummarize(joint_raw, rubric)
        except Exception as e:
            logger.error(f"Summarizer failed on joint text: {e}")
            return AuditCheckResult(
                node_id=f"{left_node.id}+{right_node.id}",
                check_type="substitution",
                passed=False,
                discrepancy_score=1.0,
                reasoning=f"Joint summarizer error: {e}"
            )

        # Disjoint path: g(g(u) ⊕ g(v)) - summarize parts first, then concatenate and summarize again
        # Use existing summaries if available, otherwise generate them
        left_summary = left_node.summary if left_node.summary else self._resummarize(raw_left, rubric)
        right_summary = right_node.summary if right_node.summary else self._resummarize(raw_right, rubric)

        # Concatenate child summaries and re-summarize
        concat_summaries = self._combine_inputs(left_summary, right_summary, rubric)
        try:
            disjoint_summary = self._resummarize(concat_summaries, rubric)
        except Exception as e:
            logger.error(f"Summarizer failed on disjoint path: {e}")
            return AuditCheckResult(
                node_id=f"{left_node.id}+{right_node.id}",
                check_type="substitution",
                passed=False,
                discrepancy_score=1.0,
                reasoning=f"Disjoint summarizer error: {e}"
            )

        # Compare f*(joint_summary) vs f*(disjoint_summary)
        is_congruent, score, reasoning = self._call_oracle(joint_summary, disjoint_summary, rubric)
        passed = is_congruent and score <= self.config.discrepancy_threshold

        return AuditCheckResult(
            node_id=f"{left_node.id}+{right_node.id}",
            check_type="substitution",
            passed=passed,
            discrepancy_score=score,
            reasoning=f"Substitution (joint vs disjoint): {reasoning}",
            input_a=joint_summary,
            input_b=disjoint_summary
        )

    def _get_adjacent_leaf_pairs(
        self, leaves: List[Node]
    ) -> List[Tuple[Node, Node]]:
        """
        Get pairs of adjacent leaf nodes for substitution checks.

        Adjacent leaves are those that appear consecutively in the
        document's original text order.

        NOTE: This method assumes leaves are already in left-to-right document
        order, which is guaranteed when leaves come from preorder/inorder
        traversal of a properly-constructed binary tree. The traversal order
        matches document order for OPS trees.

        From LocalLaws.lean (L2): The substitution check verifies that
        joint vs disjoint summarization paths produce oracle-equivalent results.

        Args:
            leaves: List of leaf nodes in document order (left-to-right)

        Returns:
            List of (left_node, right_node) tuples for adjacent pairs
        """
        if len(leaves) < 2:
            return []

        # Leaves should already be in document order from tree traversal.
        # For OPS trees, preorder traversal yields leaves in left-to-right order.
        # No sorting needed - trust the traversal order.
        pairs = []
        for i in range(len(leaves) - 1):
            pairs.append((leaves[i], leaves[i + 1]))

        return pairs

    def _update_node_audit(
        self,
        node: Node,
        result: AuditCheckResult,
        tree_id: str,
        rubric: str,
        full_input_a: str,
        full_input_b: str
    ) -> None:
        """
        Update node's audit status and flag failures to review queue.

        Args:
            node: The node being audited
            result: The audit check result
            tree_id: ID of the tree
            rubric: Information preservation rubric
            full_input_a: Full (untruncated) input A
            full_input_b: Full (untruncated) input B
        """
        if result.passed:
            node.set_audit_passed(result.discrepancy_score, result.reasoning)
        else:
            node.set_audit_failed(result.discrepancy_score, result.reasoning)

            # Flag to review queue if configured
            if self.review_queue is not None:
                self.review_queue.add(
                    node=node,
                    tree_id=tree_id,
                    check_result=result,
                    rubric=rubric,
                    full_input_a=full_input_a,
                    full_input_b=full_input_b
                )


def audit_tree(
    tree: Tree,
    scorer: Optional[ScoringOracle] = None,
    sample_budget: int = 10,
    threshold: float = 0.1
) -> AuditReport:
    """
    Convenience function to audit an OPS tree.

    Args:
        tree: Tree to audit
        scorer: ScoringOracle instance (e.g., SimpleScorer)
        sample_budget: Number of nodes to sample
        threshold: Discrepancy threshold

    Returns:
        AuditReport

    Example:
        from treepo._research.core.scoring import SimpleScorer
        report = audit_tree(tree, scorer=SimpleScorer(), threshold=0.1)
    """
    if scorer is None:
        scorer = SimpleScorer()

    config = AuditConfig(
        sample_budget=sample_budget,
        discrepancy_threshold=threshold
    )

    auditor = Auditor(scorer, config)
    return auditor.audit_tree(tree)


def get_human_review_queue(report: AuditReport) -> List[str]:
    """
    Get list of node IDs that need human review.

    Args:
        report: Audit report

    Returns:
        List of failed node IDs
    """
    return report.failed_node_ids


def compute_violation_bound(
    report: AuditReport,
    num_leaves: int,
    num_merges: Optional[int] = None,
    num_rounds: int = 1
) -> float:
    """
    Compute the global violation bound from the paper (Equation 1).

    From paper Section 4.1 (Scaling with tree size):
    Pr[root violation] <= N * p_suff + M * p_assoc + (R-1) * p_idem

    where:
    - N = number of leaves
    - M = number of merges (N-1 for binary tree)
    - R = number of re-summarization rounds
    - p_suff = sufficiency violation rate
    - p_assoc = combined audit-decomposition rate (weighted avg of p_bound and p_merge)
    - p_idem = idempotence violation rate

    This provides a transparent union bound on the probability that the
    root summary deviates from the oracle.

    Args:
        report: Audit report containing violation rates
        num_leaves: Number of leaves in the tree (N)
        num_merges: Number of internal merges (M), defaults to N-1
        num_rounds: Number of re-summarization rounds (R)

    Returns:
        Upper bound on root violation probability (capped at 1.0)
    """
    if num_merges is None:
        num_merges = max(0, num_leaves - 1)

    # Get violation rates from report
    p_suff = report.sufficiency_rate
    p_assoc = report.assoc_rate
    p_idem = report.idempotence_rate

    # Compute bound: N * p_suff + M * p_assoc + (R-1) * p_idem
    bound = (
        num_leaves * p_suff +
        num_merges * p_assoc +
        max(0, num_rounds - 1) * p_idem
    )

    return min(bound, 1.0)


def compute_expected_distortion(
    report: AuditReport,
    num_leaves: int,
    num_merges: Optional[int] = None,
    num_rounds: int = 1
) -> float:
    """
    Compute the expected task-space distortion bound.

    From paper Appendix D (Equation 2):
    Delta_1 <= N * p_suff + M * p_assoc
    Delta_R <= N * p_suff + M * p_assoc + (R-1) * p_idem  (for R >= 2)

    Since d_Y in [0,1], we have Delta_R = E[d_Y(.,.)| <= Pr(eps_R), so this
    bound also applies to expected distortion.

    Args:
        report: Audit report containing violation rates
        num_leaves: Number of leaves in the tree (N)
        num_merges: Number of internal merges (M), defaults to N-1
        num_rounds: Number of re-summarization rounds (R)

    Returns:
        Upper bound on expected distortion (capped at 1.0)
    """
    # Same computation as violation bound for bounded metrics
    return compute_violation_bound(report, num_leaves, num_merges, num_rounds)


def get_audit_statistics(report: AuditReport) -> Dict[str, Any]:
    """
    Get a summary of audit statistics for reporting.

    Returns a dictionary with all violation rates and sample counts,
    suitable for logging or display.

    Args:
        report: Audit report

    Returns:
        Dictionary of audit statistics
    """
    inclusion_probs = [
        float(check.inclusion_probability)
        for check in report.checks
        if check.inclusion_probability is not None
    ]

    return {
        "tree_id": report.tree_id,
        "source_doc_id": report.source_doc_id,
        "total_nodes": report.total_nodes,
        "nodes_audited": report.nodes_audited,
        "overall_passed": report.passed,
        "failure_rate": report.failure_rate,
        "violation_rates": {
            "p_suff": report.sufficiency_rate,
            "p_merge": report.merge_rate,
            "p_idem": report.idempotence_rate,
            "p_bound": report.substitution_rate,
            "p_assoc": report.assoc_rate,
        },
        "sample_counts": {
            "sufficiency": report.sufficiency_samples,
            "merge": report.merge_samples,
            "idempotence": report.idempotence_samples,
            "substitution": report.substitution_samples,
        },
        "violations": {
            "sufficiency": report.sufficiency_violations,
            "merge": report.merge_violations,
            "idempotence": report.idempotence_violations,
            "substitution": report.substitution_violations,
        },
        "ipw": {
            "violation_rate": report.ipw_violation_rate(),
            "violation_ci_95": report.ipw_violation_empirical_bernstein_ci(delta=0.05),
            "preference_loss": report.ipw_preference_loss(),
            "preference_ci_95": report.ipw_preference_empirical_bernstein_ci(delta=0.05),
            "sample_count": len(report.to_tree_samples()),
            "inclusion_probability_summary": {
                "count": len(inclusion_probs),
                "mean": (sum(inclusion_probs) / len(inclusion_probs)) if inclusion_probs else 0.0,
                "min": min(inclusion_probs) if inclusion_probs else 0.0,
                "max": max(inclusion_probs) if inclusion_probs else 0.0,
            },
        },
    }


def get_ipw_statistics(
    report: AuditReport,
    num_leaves: int,
    num_merges: Optional[int] = None,
    num_rounds: int = 1,
    delta: float = 0.05,
    clip_max_weight: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Get TreeIPW statistics derived from an audit report.

    This provides runtime objects that align with the Lean TreeIPW layer:
    weighted violation rates, empirical-Bernstein confidence intervals,
    union bounds, and effective sample diagnostics.
    """
    from treepo._research.tree.ipw import (
        NodeType,
        analyze_tree_samples,
        clipped_hajek_diagnostics,
        hajek_ht_comparison,
    )

    samples = report.to_tree_samples()
    analysis = analyze_tree_samples(
        samples=samples,
        num_leaves=num_leaves,
        num_merges=num_merges,
        num_rounds=num_rounds,
    )
    ht_vs_hajek = {
        "overall_violation": hajek_ht_comparison(
            samples,
            lambda sample: float(sample.violation),
            population_size=float(len(samples)) if samples else None,
        ),
        "by_check_type": {},
    }
    by_type = [
        ("sufficiency", NodeType.LEAF, report.leaf_population),
        ("merge_consistency", NodeType.MERGE, report.merge_population),
        ("idempotence", NodeType.RESUMMARY, report.idempotence_population),
        ("substitution", NodeType.SUBSTITUTION, report.substitution_population),
    ]
    for check_name, node_type, population in by_type:
        node_samples = [sample for sample in samples if sample.node_type == node_type]
        if not node_samples:
            continue
        comparison = hajek_ht_comparison(
            node_samples,
            lambda sample: float(sample.violation),
            population_size=float(population) if population > 0 else None,
        )
        comparison["population_size"] = float(population)
        ht_vs_hajek["by_check_type"][check_name] = comparison

    clipping = None
    if clip_max_weight is not None and clip_max_weight > 0:
        clipping = {
            "max_weight": float(clip_max_weight),
            "violation": clipped_hajek_diagnostics(
                samples,
                lambda sample: float(sample.violation),
                clip_max_weight,
                value_min=0.0,
                value_max=1.0,
            ),
            "preference_loss": clipped_hajek_diagnostics(
                samples,
                lambda sample: float(sample.preference_loss),
                clip_max_weight,
                value_min=0.0,
                value_max=1.0,
            ),
        }

    return {
        "tree_id": report.tree_id,
        "source_doc_id": report.source_doc_id,
        "n_samples": analysis.n_samples,
        "n_docs": analysis.n_docs,
        "violation_rate": analysis.violation_rate,
        "violation_ci": report.ipw_violation_empirical_bernstein_ci(delta=delta),
        "preference_loss": analysis.preference_loss,
        "preference_ci": report.ipw_preference_empirical_bernstein_ci(delta=delta),
        "union_bound": analysis.union_bound,
        "effective_sample_size": analysis.effective_sample_size,
        "effective_sample_ratio": analysis.effective_sample_ratio,
        "max_weight": analysis.max_weight,
        "has_adequate_neff": analysis.has_adequate_neff,
        "has_adequate_weight_bound": analysis.has_adequate_weight_bound,
        "ht_vs_hajek": ht_vs_hajek,
        "clipping": clipping,
    }
