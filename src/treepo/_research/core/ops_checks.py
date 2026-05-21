"""
Lean-aligned local-law types and audit utilities.

This module is the shared vocabulary layer between the Lean formalization and
the Python auditing / simulation code.

Important Lean mapping from ``LocalLaws.lean``:
- Paper C1 = Lean L1 = leaf preservation
- Paper C2 = Lean L3 = on-range idempotence
- Paper C3 = Lean L2 = merge preservation

The codebase still carries legacy audit names like ``merge_consistency`` for
backward compatibility, but theorem-facing code should use ``LawKind`` and
``ApproxLocalLawsBundle``.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


LEAN_LOCAL_LAW_MAP: Dict[str, str] = {
    "C1": "L1",
    "C2": "L3",
    "C3": "L2",
}


class LawKind(Enum):
    """Theorem-facing local laws from the Lean development."""

    L1_LEAF = "l1_leaf"
    L2_MERGE = "l2_merge"
    L3_IDEMPOTENCE = "l3_idempotence"

    @property
    def lean_name(self) -> str:
        return {
            LawKind.L1_LEAF: "L1",
            LawKind.L2_MERGE: "L2",
            LawKind.L3_IDEMPOTENCE: "L3",
        }[self]

    @property
    def paper_condition(self) -> str:
        return {
            LawKind.L1_LEAF: "C1",
            LawKind.L2_MERGE: "C3",
            LawKind.L3_IDEMPOTENCE: "C2",
        }[self]

    @property
    def law_id(self) -> str:
        return {
            LawKind.L1_LEAF: "leaf_preservation",
            LawKind.L2_MERGE: "merge_preservation",
            LawKind.L3_IDEMPOTENCE: "on_range_idempotence",
        }[self]


class AuditCheckKind(Enum):
    """Operational audit decompositions used by Python tooling."""

    LEAF_DIRECT = "sufficiency"
    IDEMPOTENCE_DIRECT = "idempotence"
    MERGE_RAW_TO_JOINT = "merge_consistency"
    MERGE_JOINT_TO_DISJOINT = "joint_to_disjoint_drift"
    SUBSTITUTION = "substitution"
    READOUT_AGGREGATION_DRIFT = "readout_aggregation_drift"


class EvidenceStatus(Enum):
    """How strong the evidence is for a reported operator or metric."""

    THEOREM_BACKED = "theorem_backed"
    APPROX_AUDITED = "approx_audited"
    PROXY_ONLY = "proxy_only"


class CheckType(Enum):
    """Deprecated audit check names kept as a one-release compatibility shim."""

    SUFFICIENCY = "sufficiency"      # C1: Leaf summary preserves oracle
    IDEMPOTENCE = "idempotence"      # C2: Re-summarizing is stable
    SUBSTITUTION = "substitution"    # C3A: Boundary consistency
    MERGE = "merge_consistency"      # C3B: Merge preserves oracle

    @classmethod
    def from_string(cls, s: str) -> "CheckType":
        """Convert string to CheckType, handling various formats."""
        normalized = s.lower().strip()
        if normalized in ("merge", "merge_consistency"):
            return cls.MERGE
        for check_type in cls:
            if check_type.value == normalized:
                return check_type
        raise ValueError(f"Unknown check type: {s}")

    def __str__(self) -> str:
        return self.value


@dataclass
class CheckConfig:
    """Configuration for OPS law checks.

    This provides a unified configuration interface for check parameters
    that can be used by both Auditor and Verifier implementations.
    """
    # Discrepancy threshold - scores above this are violations
    discrepancy_threshold: float = 0.1

    # Whether to treat close values as equivalent (for ordinal oracles)
    tolerance: float = 0.0

    # Which checks to enable
    check_sufficiency: bool = True
    check_idempotence: bool = True
    check_substitution: bool = True
    check_merge: bool = True


@dataclass(frozen=True)
class ApproxLocalLawsBundle:
    """Python mirror of Lean's ``ApproxLocalLawsBundle``."""

    eps_leaf: float
    eps_merge: float
    eps_idemp: float
    evidence_status: EvidenceStatus = EvidenceStatus.APPROX_AUDITED
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eps_leaf": float(self.eps_leaf),
            "eps_merge": float(self.eps_merge),
            "eps_idemp": float(self.eps_idemp),
            "evidence_status": self.evidence_status.value,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class OracleMeasurementEnvelope:
    """Objective-side oracle-measurement assumptions or certificates."""

    exact_oracle: bool = False
    uniform_error_bound: Optional[float] = None
    high_probability_error_bound: Optional[float] = None
    failure_probability: Optional[float] = None
    pointwise_error_bound_available: bool = False
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exact_oracle": bool(self.exact_oracle),
            "uniform_error_bound": (
                float(self.uniform_error_bound)
                if self.uniform_error_bound is not None
                else None
            ),
            "high_probability_error_bound": (
                float(self.high_probability_error_bound)
                if self.high_probability_error_bound is not None
                else None
            ),
            "failure_probability": (
                float(self.failure_probability)
                if self.failure_probability is not None
                else None
            ),
            "pointwise_error_bound_available": bool(self.pointwise_error_bound_available),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class LawCapabilityReport:
    """Per-law capability report for theorem-backed vs proxy operators."""

    law_kind: LawKind
    available: bool
    evidence_status: EvidenceStatus
    objective_enforced: bool = False
    exact: Optional[bool] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "law_id": self.law_kind.law_id,
            "theorem_label": self.law_kind.lean_name,
            "paper_label": self.law_kind.paper_condition,
            "available": bool(self.available),
            "evidence_status": self.evidence_status.value,
            "objective_enforced": bool(self.objective_enforced),
            "exact": self.exact,
            "notes": self.notes,
            "metadata": {
                "law_kind": self.law_kind.value,
                "lean_name": self.law_kind.lean_name,
                "paper_condition": self.law_kind.paper_condition,
            },
        }


@dataclass(frozen=True)
class OperatorCapabilityReport:
    """Unified capability surface for summary and sketch operators."""

    operator_name: str
    evidence_status: EvidenceStatus
    latent_mergeability_enforced: bool
    tree_nesting_supported: bool
    theorem_domain_decode_available: bool
    theorem_domain_reencode_available: bool
    exact_reduction_supported: bool = False
    leaf_law: LawCapabilityReport = field(
        default_factory=lambda: LawCapabilityReport(
            law_kind=LawKind.L1_LEAF,
            available=False,
            evidence_status=EvidenceStatus.PROXY_ONLY,
        )
    )
    merge_law: LawCapabilityReport = field(
        default_factory=lambda: LawCapabilityReport(
            law_kind=LawKind.L2_MERGE,
            available=False,
            evidence_status=EvidenceStatus.PROXY_ONLY,
        )
    )
    idempotence_law: LawCapabilityReport = field(
        default_factory=lambda: LawCapabilityReport(
            law_kind=LawKind.L3_IDEMPOTENCE,
            available=False,
            evidence_status=EvidenceStatus.PROXY_ONLY,
        )
    )
    approx_local_laws: Optional[ApproxLocalLawsBundle] = None
    oracle_measurement: Optional[OracleMeasurementEnvelope] = None
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def supports_resummary_idempotence(self) -> bool:
        return (
            bool(self.theorem_domain_decode_available)
            and bool(self.theorem_domain_reencode_available)
            and bool(self.idempotence_law.available)
        )

    @property
    def supports_theorem_backed_l3(self) -> bool:
        return (
            self.supports_resummary_idempotence
            and self.idempotence_law.evidence_status == EvidenceStatus.THEOREM_BACKED
        )

    @property
    def supports_oracle_measurement_bridge(self) -> bool:
        if self.oracle_measurement is None:
            return False
        return bool(
            self.oracle_measurement.exact_oracle
            or self.oracle_measurement.uniform_error_bound is not None
            or self.oracle_measurement.high_probability_error_bound is not None
            or self.oracle_measurement.pointwise_error_bound_available
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operator_name": self.operator_name,
            "evidence_status": self.evidence_status.value,
            "latent_mergeability_enforced": bool(self.latent_mergeability_enforced),
            "tree_nesting_supported": bool(self.tree_nesting_supported),
            "theorem_domain_decode_available": bool(self.theorem_domain_decode_available),
            "theorem_domain_reencode_available": bool(self.theorem_domain_reencode_available),
            "exact_reduction_supported": bool(self.exact_reduction_supported),
            "supports_resummary_idempotence": bool(self.supports_resummary_idempotence),
            "supports_theorem_backed_l3": bool(self.supports_theorem_backed_l3),
            "supports_oracle_measurement_bridge": bool(
                self.supports_oracle_measurement_bridge
            ),
            "objective_enforces_leaf_preservation": bool(self.leaf_law.objective_enforced),
            "objective_enforces_merge_preservation_against_span_oracle": bool(
                self.merge_law.objective_enforced
            ),
            "objective_enforces_idempotence": bool(self.idempotence_law.objective_enforced),
            "laws": {
                self.leaf_law.law_kind.law_id: self.leaf_law.to_dict(),
                self.merge_law.law_kind.law_id: self.merge_law.to_dict(),
                self.idempotence_law.law_kind.law_id: self.idempotence_law.to_dict(),
            },
            "approx_local_laws": (
                self.approx_local_laws.to_dict()
                if self.approx_local_laws is not None
                else None
            ),
            "oracle_measurement": (
                self.oracle_measurement.to_dict()
                if self.oracle_measurement is not None
                else None
            ),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class LawEvaluationRecord:
    """Structured record for theorem-law or audit-decomposition outputs."""

    law_kind: Optional[LawKind]
    audit_kind: Optional[AuditCheckKind]
    evidence_status: EvidenceStatus
    discrepancy: float
    passed: bool
    node_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "law_kind": self.law_kind.value if self.law_kind is not None else None,
            "audit_kind": self.audit_kind.value if self.audit_kind is not None else None,
            "evidence_status": self.evidence_status.value,
            "discrepancy": float(self.discrepancy),
            "passed": bool(self.passed),
            "node_id": self.node_id,
            "details": dict(self.details),
        }


@dataclass
class CheckResult:
    """
    Unified result format for OPS law checks.

    This is a base result type that can be used across different
    check implementations. Both AuditCheckResult and LawCheckResult
    are compatible with this interface.
    """
    check_type: CheckType
    passed: bool
    discrepancy: float
    reasoning: str = ""

    # The values that were compared (optional)
    input_a: Optional[str] = None
    input_b: Optional[str] = None

    # Node/context info
    node_id: Optional[str] = None

    # Skipped checks (e.g., no summarizer provided)
    skipped: bool = False
    skip_reason: Optional[str] = None

    @property
    def is_violation(self) -> bool:
        """True if this is a violation (failed and not skipped)."""
        return not self.passed and not self.skipped

    @property
    def was_evaluated(self) -> bool:
        """True if the check was actually performed (not skipped)."""
        return not self.skipped

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'check_type': str(self.check_type),
            'passed': self.passed,
            'discrepancy': self.discrepancy,
            'reasoning': self.reasoning,
            'input_a': self.input_a[:100] + '...' if self.input_a and len(self.input_a) > 100 else self.input_a,
            'input_b': self.input_b[:100] + '...' if self.input_b and len(self.input_b) > 100 else self.input_b,
            'node_id': self.node_id,
            'skipped': self.skipped,
            'skip_reason': self.skip_reason,
        }


def aggregate_check_stats(results: List[CheckResult]) -> Dict[str, Any]:
    """
    Aggregate statistics from multiple check results.

    Args:
        results: List of CheckResult objects

    Returns:
        Dict with violation counts and rates per check type
    """
    stats = {
        "total_checks": len(results),
        "total_passed": sum(1 for r in results if r.passed),
        "total_failed": sum(1 for r in results if not r.passed and not r.skipped),
        "total_skipped": sum(1 for r in results if r.skipped),
        "by_type": {},
    }

    for check_type in CheckType:
        type_results = [r for r in results if r.check_type == check_type]
        if type_results:
            n_passed = sum(1 for r in type_results if r.passed)
            n_failed = sum(1 for r in type_results if not r.passed and not r.skipped)
            n_skipped = sum(1 for r in type_results if r.skipped)
            n_total = len(type_results)
            n_evaluated = n_total - n_skipped
            stats["by_type"][str(check_type)] = {
                "total": n_total,
                "passed": n_passed,
                "failed": n_failed,
                "skipped": n_skipped,
                "pass_rate": n_passed / n_evaluated if n_evaluated > 0 else 1.0,
                "violation_rate": n_failed / n_evaluated if n_evaluated > 0 else 0.0,
            }

    # Overall rates
    n_evaluated = stats["total_checks"] - stats["total_skipped"]
    stats["overall_pass_rate"] = stats["total_passed"] / n_evaluated if n_evaluated > 0 else 1.0
    stats["overall_violation_rate"] = stats["total_failed"] / n_evaluated if n_evaluated > 0 else 0.0

    return stats


# =============================================================================
# Protocol for Oracle Functions
# =============================================================================

@runtime_checkable
class OracleProtocol(Protocol):
    """
    Protocol for oracle functions used in OPS law checking.

    An oracle compares two texts and determines if they are congruent
    with respect to preserving task-relevant information.
    """

    def __call__(
        self,
        input_a: str,
        input_b: str,
        rubric: str,
    ) -> tuple:
        """
        Compare two inputs and return congruence result.

        Args:
            input_a: First input text
            input_b: Second input text
            rubric: Description of what to preserve

        Returns:
            Tuple of (is_congruent: bool, discrepancy: float, reasoning: str)
        """
        ...


# Convenience re-exports
__all__ = [
    "ApproxLocalLawsBundle",
    "OracleMeasurementEnvelope",
    "AuditCheckKind",
    "CheckType",
    "CheckConfig",
    "CheckResult",
    "EvidenceStatus",
    "LawEvaluationRecord",
    "LawKind",
    "LEAN_LOCAL_LAW_MAP",
    "aggregate_check_stats",
    "OracleProtocol",
]
