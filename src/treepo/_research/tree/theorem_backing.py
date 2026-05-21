"""General theorem-backing assumption bundles for compositional operators."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from treepo._research.core.ops_checks import (
    ApproxLocalLawsBundle,
    EvidenceStatus,
    LawCapabilityReport,
    LawKind,
    OracleMeasurementEnvelope,
    OperatorCapabilityReport,
)
from treepo._research.tree.compositional_operator import OperatorAssumptionBundle


class TheoremBackingRoute(Enum):
    """Sufficient-condition route used to certify an operator."""

    DIRECT_LOCAL_LAWS = "direct_local_laws"
    DIRECT_APPROX_LOCAL_LAWS = "direct_approx_local_laws"
    AUDITED_APPROX_UPPER_BOUNDS = "audited_approx_upper_bounds"
    SKETCH_CODEC_LOCAL_LAWS = "sketch_codec_local_laws"
    SKETCH_CODEC_APPROX_LOCAL_LAWS = "sketch_codec_approx_local_laws"
    GLOBAL_PRESERVATION = "global_preservation"


@dataclass(frozen=True)
class AssumptionObligation:
    """Single formal obligation in a theorem-backing route."""

    symbol: str
    statement: str
    lean_name: Optional[str] = None
    evidence_status: EvidenceStatus = EvidenceStatus.THEOREM_BACKED
    law_kinds: Tuple[LawKind, ...] = field(default_factory=tuple)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "statement": self.statement,
            "lean_name": self.lean_name,
            "evidence_status": self.evidence_status.value,
            "law_kinds": [law.value for law in self.law_kinds],
            "notes": self.notes,
        }


@dataclass(frozen=True)
class TheoremAssumptionSpec:
    """Structured sufficient assumptions for theorem-backed exact or approximate use."""

    name: str
    route: TheoremBackingRoute
    theorem_bundle_kind: str
    operator_assumptions: OperatorAssumptionBundle
    obligations: Tuple[AssumptionObligation, ...]
    approx_local_laws: Optional[ApproxLocalLawsBundle] = None
    oracle_measurement: Optional[OracleMeasurementEnvelope] = None
    lean_bridge: Optional[str] = None
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def capability_report(
        self,
        operator_name: str,
        *,
        extra_notes: Tuple[str, ...] = (),
    ) -> OperatorCapabilityReport:
        return self.operator_assumptions.capability_report(
            operator_name,
            approx_local_laws=self.approx_local_laws,
            oracle_measurement=self.oracle_measurement,
            extra_notes=tuple(self.notes) + tuple(extra_notes),
        )

    def with_oracle_measurement(
        self,
        oracle_measurement: OracleMeasurementEnvelope,
        *,
        extra_notes: Tuple[str, ...] = (),
    ) -> "TheoremAssumptionSpec":
        merged_notes = tuple(self.notes) + tuple(extra_notes)
        return replace(
            self,
            oracle_measurement=oracle_measurement,
            notes=merged_notes,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "route": self.route.value,
            "theorem_bundle_kind": self.theorem_bundle_kind,
            "lean_bridge": self.lean_bridge,
            "notes": list(self.notes),
            "obligations": [obligation.to_dict() for obligation in self.obligations],
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
            "operator_assumptions": {
                "evidence_status": self.operator_assumptions.evidence_status.value,
                "latent_mergeability_enforced": bool(
                    self.operator_assumptions.latent_mergeability_enforced
                ),
                "tree_nesting_supported": bool(
                    self.operator_assumptions.tree_nesting_supported
                ),
                "theorem_domain_decode_available": bool(
                    self.operator_assumptions.theorem_domain_decode_available
                ),
                "theorem_domain_reencode_available": bool(
                    self.operator_assumptions.theorem_domain_reencode_available
                ),
                "exact_reduction_supported": bool(
                    self.operator_assumptions.exact_reduction_supported
                ),
            },
        }


def exact_oracle_measurement_assumption() -> OracleMeasurementEnvelope:
    """Convenience regime where the oracle target is treated as exact truth."""

    return OracleMeasurementEnvelope(
        exact_oracle=True,
        uniform_error_bound=0.0,
        high_probability_error_bound=0.0,
        failure_probability=0.0,
        pointwise_error_bound_available=True,
        notes=(
            "Convenience regime: the true target is identified exactly by the oracle, "
            "so all oracle-measurement terms collapse to zero."
        ),
    )


def uniform_oracle_measurement_assumption(
    error_bound: float,
    *,
    pointwise_error_bound_available: bool = False,
    notes: Optional[str] = None,
) -> OracleMeasurementEnvelope:
    """Deterministic objective-side oracle-measurement envelope."""

    return OracleMeasurementEnvelope(
        exact_oracle=bool(error_bound == 0.0),
        uniform_error_bound=error_bound,
        pointwise_error_bound_available=pointwise_error_bound_available,
        notes=notes
        or (
            "Uniform objective-side oracle measurement error. "
            "This is the deterministic additive term used by the optimizer perturbation theorems."
        ),
    )


def high_probability_oracle_measurement_assumption(
    error_bound: float,
    failure_probability: float,
    *,
    pointwise_error_bound_available: bool = False,
    notes: Optional[str] = None,
) -> OracleMeasurementEnvelope:
    """High-probability oracle-measurement envelope for objective perturbation."""

    return OracleMeasurementEnvelope(
        exact_oracle=bool(error_bound == 0.0 and failure_probability == 0.0),
        uniform_error_bound=error_bound,
        high_probability_error_bound=error_bound,
        failure_probability=failure_probability,
        pointwise_error_bound_available=pointwise_error_bound_available,
        notes=notes
        or (
            "High-probability oracle measurement envelope. "
            "The optimizer theorems treat failure probability as a separate additive certificate budget."
        ),
    )


def _exact_law(law_kind: LawKind, notes: str) -> LawCapabilityReport:
    return LawCapabilityReport(
        law_kind=law_kind,
        available=True,
        evidence_status=EvidenceStatus.THEOREM_BACKED,
        exact=True,
        notes=notes,
    )


def _approx_law(law_kind: LawKind, notes: str) -> LawCapabilityReport:
    return LawCapabilityReport(
        law_kind=law_kind,
        available=True,
        evidence_status=EvidenceStatus.APPROX_AUDITED,
        exact=False,
        notes=notes,
    )


def broadest_exact_theorem_assumptions() -> TheoremAssumptionSpec:
    """Broadest exact sufficient assumptions currently formalized."""

    operator_assumptions = OperatorAssumptionBundle(
        evidence_status=EvidenceStatus.THEOREM_BACKED,
        theorem_domain_decode_available=True,
        theorem_domain_reencode_available=True,
        latent_mergeability_enforced=False,
        tree_nesting_supported=True,
        exact_reduction_supported=False,
        leaf_law=_exact_law(
            LawKind.L1_LEAF,
            "Exact theorem-backedness can be discharged directly from L1.",
        ),
        merge_law=_exact_law(
            LawKind.L2_MERGE,
            "Exact theorem-backedness can be discharged directly from L2.",
        ),
        idempotence_law=_exact_law(
            LawKind.L3_IDEMPOTENCE,
            "Exact theorem-backedness can be discharged directly from L3.",
        ),
    )
    obligations = (
        AssumptionObligation(
            symbol="L1",
            statement="Leaf summaries preserve the oracle exactly on all realized leaves.",
            lean_name="LocalLawsBundle.law1",
            law_kinds=(LawKind.L1_LEAF,),
        ),
        AssumptionObligation(
            symbol="L2",
            statement="Each internal merge preserves the oracle exactly against the node span.",
            lean_name="LocalLawsBundle.law2",
            law_kinds=(LawKind.L2_MERGE,),
        ),
        AssumptionObligation(
            symbol="L3",
            statement="Re-summarizing any summary in range leaves the oracle unchanged.",
            lean_name="LocalLawsBundle.law3",
            law_kinds=(LawKind.L3_IDEMPOTENCE,),
        ),
    )
    return TheoremAssumptionSpec(
        name="broadest_exact_theorem_assumptions",
        route=TheoremBackingRoute.DIRECT_LOCAL_LAWS,
        theorem_bundle_kind="LocalLawsBundle",
        operator_assumptions=operator_assumptions,
        obligations=obligations,
        lean_bridge="ExactTheoremBacked.ofLocalLaws",
        notes=(
            "This is the broadest exact sufficient interface currently formalized.",
            "Any stronger route should compile down to a LocalLawsBundle on the induced summarizer.",
        ),
    )


def broadest_approximate_theorem_assumptions(
    *,
    eps_leaf: float,
    eps_merge: float,
    eps_idemp: float,
) -> TheoremAssumptionSpec:
    """Broadest approximate sufficient assumptions currently formalized."""

    operator_assumptions = OperatorAssumptionBundle(
        evidence_status=EvidenceStatus.APPROX_AUDITED,
        theorem_domain_decode_available=True,
        theorem_domain_reencode_available=True,
        latent_mergeability_enforced=False,
        tree_nesting_supported=True,
        exact_reduction_supported=False,
        leaf_law=_approx_law(
            LawKind.L1_LEAF,
            "Approximate theorem-backedness can be discharged directly from L1ε.",
        ),
        merge_law=_approx_law(
            LawKind.L2_MERGE,
            "Approximate theorem-backedness can be discharged directly from L2ε.",
        ),
        idempotence_law=_approx_law(
            LawKind.L3_IDEMPOTENCE,
            "Approximate theorem-backedness can be discharged directly from L3ε.",
        ),
    )
    obligations = (
        AssumptionObligation(
            symbol="L1ε",
            statement="Total leaf violation budget is bounded by epsLeaf.",
            lean_name="ApproxLocalLawsBundle.law1",
            evidence_status=EvidenceStatus.APPROX_AUDITED,
            law_kinds=(LawKind.L1_LEAF,),
        ),
        AssumptionObligation(
            symbol="L2ε",
            statement="Total merge violation budget is bounded by epsMerge.",
            lean_name="ApproxLocalLawsBundle.law2",
            evidence_status=EvidenceStatus.APPROX_AUDITED,
            law_kinds=(LawKind.L2_MERGE,),
        ),
        AssumptionObligation(
            symbol="L3ε",
            statement="Reduction-distribution idempotence budget is bounded by epsIdemp.",
            lean_name="ApproxLocalLawsBundle.law3",
            evidence_status=EvidenceStatus.APPROX_AUDITED,
            law_kinds=(LawKind.L3_IDEMPOTENCE,),
        ),
    )
    return TheoremAssumptionSpec(
        name="broadest_approximate_theorem_assumptions",
        route=TheoremBackingRoute.DIRECT_APPROX_LOCAL_LAWS,
        theorem_bundle_kind="ApproxLocalLawsBundle",
        operator_assumptions=operator_assumptions,
        obligations=obligations,
        approx_local_laws=ApproxLocalLawsBundle(
            eps_leaf=eps_leaf,
            eps_merge=eps_merge,
            eps_idemp=eps_idemp,
        ),
        lean_bridge="ApproxTheoremBacked.ofApproxLocalLaws",
        notes=(
            "This is the broadest approximate sufficient interface currently formalized.",
            "Any stronger audited or structural route should compile down to an ApproxLocalLawsBundle.",
        ),
    )


def llm_direct_exact_assumptions() -> TheoremAssumptionSpec:
    """Exact theorem-backing route for theorem-domain text summaries."""

    base = broadest_exact_theorem_assumptions()
    return TheoremAssumptionSpec(
        name="llm_direct_exact_assumptions",
        route=TheoremBackingRoute.DIRECT_LOCAL_LAWS,
        theorem_bundle_kind=base.theorem_bundle_kind,
        operator_assumptions=base.operator_assumptions,
        obligations=base.obligations,
        lean_bridge=base.lean_bridge,
        notes=(
            "The theorem domain is the summary space itself, so no sketch codec obligations are needed.",
            "What must be proved is exactly L1/L2/L3 for the summarizer acting on theorem-domain strings.",
        ),
    )


def llm_direct_approximate_assumptions(
    *,
    eps_leaf: float,
    eps_merge: float,
    eps_idemp: float,
) -> TheoremAssumptionSpec:
    """Approximate theorem-backing route for direct theorem-domain text summaries."""

    base = broadest_approximate_theorem_assumptions(
        eps_leaf=eps_leaf,
        eps_merge=eps_merge,
        eps_idemp=eps_idemp,
    )
    return TheoremAssumptionSpec(
        name="llm_direct_approximate_assumptions",
        route=TheoremBackingRoute.DIRECT_APPROX_LOCAL_LAWS,
        theorem_bundle_kind=base.theorem_bundle_kind,
        operator_assumptions=base.operator_assumptions,
        obligations=base.obligations,
        approx_local_laws=base.approx_local_laws,
        lean_bridge=base.lean_bridge,
        notes=(
            "The theorem domain is the summary space itself, so approximate theorem-backedness is just an ApproxLocalLawsBundle on the summarizer.",
            "This is the direct approximate route before introducing any audited-aggregation certificate.",
        ),
    )


def llm_audited_approximate_assumptions(
    *,
    eps_leaf: float,
    eps_merge: float,
    eps_idemp: float,
) -> TheoremAssumptionSpec:
    """Approximate theorem-backing route from audited aggregate upper bounds."""

    operator_assumptions = OperatorAssumptionBundle(
        evidence_status=EvidenceStatus.APPROX_AUDITED,
        theorem_domain_decode_available=True,
        theorem_domain_reencode_available=True,
        latent_mergeability_enforced=False,
        tree_nesting_supported=True,
        exact_reduction_supported=False,
        leaf_law=_approx_law(
            LawKind.L1_LEAF,
            "Audited total leaf violation upper bounds are sufficient.",
        ),
        merge_law=_approx_law(
            LawKind.L2_MERGE,
            "Audited total merge violation upper bounds are sufficient.",
        ),
        idempotence_law=_approx_law(
            LawKind.L3_IDEMPOTENCE,
            "Audited reduction-distribution idempotence upper bounds are sufficient.",
        ),
    )
    obligations = (
        AssumptionObligation(
            symbol="leaf_cert",
            statement="totalLeafViolation(g, fstar, T) <= epsLeaf",
            lean_name="AuditedApproxUpperBounds.leaf_cert",
            evidence_status=EvidenceStatus.APPROX_AUDITED,
            law_kinds=(LawKind.L1_LEAF,),
        ),
        AssumptionObligation(
            symbol="merge_cert",
            statement="totalMergeViolation(g, fstar, T) <= epsMerge",
            lean_name="AuditedApproxUpperBounds.merge_cert",
            evidence_status=EvidenceStatus.APPROX_AUDITED,
            law_kinds=(LawKind.L2_MERGE,),
        ),
        AssumptionObligation(
            symbol="idemp_cert",
            statement="pIdemp(g, fstar, reduce(g, T)) <= epsIdemp",
            lean_name="AuditedApproxUpperBounds.idemp_cert",
            evidence_status=EvidenceStatus.APPROX_AUDITED,
            law_kinds=(LawKind.L3_IDEMPOTENCE,),
        ),
    )
    return TheoremAssumptionSpec(
        name="llm_audited_approximate_assumptions",
        route=TheoremBackingRoute.AUDITED_APPROX_UPPER_BOUNDS,
        theorem_bundle_kind="ApproxLocalLawsBundle",
        operator_assumptions=operator_assumptions,
        obligations=obligations,
        approx_local_laws=ApproxLocalLawsBundle(
            eps_leaf=eps_leaf,
            eps_merge=eps_merge,
            eps_idemp=eps_idemp,
        ),
        lean_bridge="approx_bundle_of_audited_upper_bounds",
        notes=(
            "This is the standard audit-driven theorem-backing route for direct theorem-domain summarizers.",
            "It is weaker than exact L1/L2/L3 and stronger than a purely proxy-side regularizer.",
        ),
    )


def neural_codec_exact_assumptions() -> TheoremAssumptionSpec:
    """Exact theorem-backing route for any supplied encode/merge/decode codec."""

    operator_assumptions = OperatorAssumptionBundle(
        evidence_status=EvidenceStatus.THEOREM_BACKED,
        theorem_domain_decode_available=True,
        theorem_domain_reencode_available=True,
        latent_mergeability_enforced=True,
        tree_nesting_supported=True,
        exact_reduction_supported=False,
        leaf_law=_exact_law(
            LawKind.L1_LEAF,
            "Exact theorem-backedness follows from SketchLeafPreserving.",
        ),
        merge_law=_exact_law(
            LawKind.L2_MERGE,
            "Exact theorem-backedness follows from SketchMergeCompatible plus SketchSummaryCompatible.",
        ),
        idempotence_law=_exact_law(
            LawKind.L3_IDEMPOTENCE,
            "Exact theorem-backedness follows because SketchLeafPreserving gives pointwise preservation of decode(encode(.)) on theorem-domain spans.",
        ),
    )
    obligations = (
        AssumptionObligation(
            symbol="SketchLeafPreserving",
            statement="decode(encode(x)) preserves the oracle exactly for every theorem-domain span x.",
            lean_name="SketchLeafPreserving",
            law_kinds=(LawKind.L1_LEAF, LawKind.L3_IDEMPOTENCE),
        ),
        AssumptionObligation(
            symbol="SketchMergeCompatible",
            statement="If decoded child states match x and y at the oracle, then decode(merge(sL, sR)) matches x * y at the oracle.",
            lean_name="SketchMergeCompatible",
            law_kinds=(LawKind.L2_MERGE,),
        ),
        AssumptionObligation(
            symbol="SketchSummaryCompatible",
            statement="decode(merge(sL, sR)) = decode(encode(decode(sL) * decode(sR))).",
            lean_name="SketchSummaryCompatible",
            law_kinds=(LawKind.L2_MERGE,),
        ),
    )
    return TheoremAssumptionSpec(
        name="neural_codec_exact_assumptions",
        route=TheoremBackingRoute.SKETCH_CODEC_LOCAL_LAWS,
        theorem_bundle_kind="LocalLawsBundle",
        operator_assumptions=operator_assumptions,
        obligations=obligations,
        lean_bridge="local_laws_bundle_of_sketch",
        notes=(
            "This is the exact route for any learned or hand-specified codec with theorem-domain encode/merge/decode semantics.",
            "It does not require the base predictor itself to decode; it only requires that a codec with these properties is supplied over the same state family.",
        ),
    )


def neural_codec_approximate_assumptions(
    *,
    eps_leaf: float,
    eps_merge: float,
    eps_idemp: float,
) -> TheoremAssumptionSpec:
    """Approximate theorem-backing route for any supplied encode/merge/decode codec."""

    operator_assumptions = OperatorAssumptionBundle(
        evidence_status=EvidenceStatus.APPROX_AUDITED,
        theorem_domain_decode_available=True,
        theorem_domain_reencode_available=True,
        latent_mergeability_enforced=True,
        tree_nesting_supported=True,
        exact_reduction_supported=False,
        leaf_law=_approx_law(
            LawKind.L1_LEAF,
            "Approximate theorem-backedness follows from SketchLeafApproxPreserving.",
        ),
        merge_law=_approx_law(
            LawKind.L2_MERGE,
            "Approximate theorem-backedness follows from SketchMergeApproxCompatible.",
        ),
        idempotence_law=_approx_law(
            LawKind.L3_IDEMPOTENCE,
            "Approximate theorem-backedness follows from an L3ε budget on the induced summarizer.",
        ),
    )
    obligations = (
        AssumptionObligation(
            symbol="SketchLeafApproxPreserving",
            statement="ViolationProb(fstar, sketchSummarizer(op, b), b) <= epsLeaf(b).",
            lean_name="SketchLeafApproxPreserving",
            evidence_status=EvidenceStatus.APPROX_AUDITED,
            law_kinds=(LawKind.L1_LEAF,),
        ),
        AssumptionObligation(
            symbol="SketchMergeApproxCompatible",
            statement="ViolationProb(fstar, reduce(sketchSummarizer(op), node(TL, TR)), S(node(TL, TR))) <= epsMerge(TL, TR).",
            lean_name="SketchMergeApproxCompatible",
            evidence_status=EvidenceStatus.APPROX_AUDITED,
            law_kinds=(LawKind.L2_MERGE,),
        ),
        AssumptionObligation(
            symbol="L3ε",
            statement="pIdemp(sketchSummarizer(op), fstar, reduce(sketchSummarizer(op), T)) <= epsIdemp.",
            lean_name="L3ε",
            evidence_status=EvidenceStatus.APPROX_AUDITED,
            law_kinds=(LawKind.L3_IDEMPOTENCE,),
        ),
    )
    return TheoremAssumptionSpec(
        name="neural_codec_approximate_assumptions",
        route=TheoremBackingRoute.SKETCH_CODEC_APPROX_LOCAL_LAWS,
        theorem_bundle_kind="ApproxLocalLawsBundle",
        operator_assumptions=operator_assumptions,
        obligations=obligations,
        approx_local_laws=ApproxLocalLawsBundle(
            eps_leaf=eps_leaf,
            eps_merge=eps_merge,
            eps_idemp=eps_idemp,
        ),
        lean_bridge="approx_bundle_of_sketch",
        notes=(
            "This is the realistic approximate route for learned codecs.",
            "The theorem-domain autoencoder need not be baked into the base model class; it only needs to be supplied with these audited properties.",
        ),
    )


def global_preservation_exact_assumptions() -> TheoremAssumptionSpec:
    """Stronger exact route via global A1/A2/A3 assumptions."""

    operator_assumptions = OperatorAssumptionBundle(
        evidence_status=EvidenceStatus.THEOREM_BACKED,
        theorem_domain_decode_available=True,
        theorem_domain_reencode_available=True,
        latent_mergeability_enforced=False,
        tree_nesting_supported=True,
        exact_reduction_supported=False,
        leaf_law=_exact_law(
            LawKind.L1_LEAF,
            "A1_global implies exact L1.",
        ),
        merge_law=_exact_law(
            LawKind.L2_MERGE,
            "A1_global + A2_global + A3_global imply exact L2.",
        ),
        idempotence_law=_exact_law(
            LawKind.L3_IDEMPOTENCE,
            "A1_global implies exact L3.",
        ),
    )
    obligations = (
        AssumptionObligation(
            symbol="A1_global",
            statement="Global oracle sufficiency holds for every theorem-domain span.",
            lean_name="A1_global",
            law_kinds=(LawKind.L1_LEAF, LawKind.L3_IDEMPOTENCE),
        ),
        AssumptionObligation(
            symbol="A2_global",
            statement="Joint and disjoint summarization routes are globally oracle-equivalent.",
            lean_name="A2_global",
            law_kinds=(LawKind.L2_MERGE,),
        ),
        AssumptionObligation(
            symbol="A3_global",
            statement="An oracle-level merge function exists for the target utility.",
            lean_name="A3_global",
            law_kinds=(LawKind.L2_MERGE,),
        ),
    )
    return TheoremAssumptionSpec(
        name="global_preservation_exact_assumptions",
        route=TheoremBackingRoute.GLOBAL_PRESERVATION,
        theorem_bundle_kind="LocalLawsBundle",
        operator_assumptions=operator_assumptions,
        obligations=obligations,
        lean_bridge="GlobalPreservation.toLocalLawsBundle",
        notes=(
            "This route is stronger than the minimal LocalLawsBundle interface.",
            "It is useful when the model is easier to reason about globally than node-by-node.",
        ),
    )


__all__ = [
    "AssumptionObligation",
    "TheoremAssumptionSpec",
    "TheoremBackingRoute",
    "exact_oracle_measurement_assumption",
    "uniform_oracle_measurement_assumption",
    "high_probability_oracle_measurement_assumption",
    "broadest_exact_theorem_assumptions",
    "broadest_approximate_theorem_assumptions",
    "llm_direct_exact_assumptions",
    "llm_direct_approximate_assumptions",
    "llm_audited_approximate_assumptions",
    "neural_codec_exact_assumptions",
    "neural_codec_approximate_assumptions",
    "global_preservation_exact_assumptions",
]
