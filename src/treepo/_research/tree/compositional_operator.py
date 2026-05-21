"""
General compositional operator interfaces for theorem-backed reductions.

This module is intentionally backend-agnostic. A compositional operator is any
object with:

- a theorem-domain object type `SpanT`,
- an internal mergeable state type `StateT`,
- an `encode / merge / decode / combine / resummarize` interface, and
- an explicit capability or assumption bundle describing which local laws are
  available, exact, approximate-audited, or merely proxy-exposed.

The theorem-backed status therefore comes from the supplied assumptions and
certificates, not from whether the backend is called "LLM", "neural", or
"sketch".
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Optional, Protocol, Tuple, TypeVar

from treepo._research.core.ops_checks import (
    EvidenceStatus,
    LawCapabilityReport,
    LawKind,
    OperatorCapabilityReport,
)
from treepo._research.core.protocols import format_merge_input

SpanT = TypeVar("SpanT")
StateT = TypeVar("StateT")
ReadoutStateT = TypeVar("ReadoutStateT")


def _call_with_supported_kwargs(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a function while dropping kwargs it does not declare."""
    if not kwargs:
        return fn(*args)
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)

    accepts_var_kw = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_var_kw:
        return fn(*args, **kwargs)

    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return fn(*args, **filtered)


def _law_report_is_placeholder(report: LawCapabilityReport) -> bool:
    """Detect the default unavailable/proxy placeholder used by operator dataclasses."""
    return (
        report.available is False
        and report.evidence_status == EvidenceStatus.PROXY_ONLY
        and report.objective_enforced is False
        and report.exact is None
        and report.notes is None
    )


@dataclass(frozen=True)
class OperatorPrediction:
    """Model-agnostic scalar prediction with uncertainty metadata."""

    mean: float
    lower: float
    upper: float
    std: float
    normalized_mean: Optional[float] = None
    confidence: Optional[float] = None
    evidence_status: EvidenceStatus = EvidenceStatus.PROXY_ONLY
    aux: Dict[str, Any] = field(default_factory=dict)


class CompositionalOperator(Protocol, Generic[SpanT, StateT]):
    """Backend-agnostic theorem-facing reduction operator."""

    name: str
    evidence_status: EvidenceStatus

    def encode(self, span: SpanT, **kwargs: Any) -> StateT:
        """Encode a theorem-domain object into an internal state."""
        ...

    def merge(self, left: StateT, right: StateT, **kwargs: Any) -> StateT:
        """Merge two internal states."""
        ...

    def decode(self, state: StateT, **kwargs: Any) -> SpanT:
        """Decode an internal state back into the theorem-domain space."""
        ...

    def combine(self, left_span: SpanT, right_span: SpanT, **kwargs: Any) -> SpanT:
        """The theorem-domain monoid action used at internal nodes."""
        ...

    def resummarize(self, span: SpanT, **kwargs: Any) -> SpanT:
        """Induced theorem-domain re-summary operation."""
        ...

    def capability_report(self) -> OperatorCapabilityReport:
        """Structured law/certification report for the operator."""
        ...


class StatePredictor(Protocol, Generic[ReadoutStateT]):
    """State-to-scalar readout used by proxy models."""

    name: str
    state_dim: int
    evidence_status: EvidenceStatus

    def predict_from_state(self, state: ReadoutStateT, **kwargs: Any) -> OperatorPrediction:
        """Predict a scalar output from an internal state."""
        ...


@dataclass(frozen=True)
class OperatorAssumptionBundle:
    """Supplied structural assumptions/certificates for a compositional operator."""

    evidence_status: EvidenceStatus = EvidenceStatus.THEOREM_BACKED
    theorem_domain_decode_available: bool = True
    theorem_domain_reencode_available: bool = True
    latent_mergeability_enforced: bool = False
    tree_nesting_supported: bool = True
    exact_reduction_supported: bool = False
    leaf_law: Optional[LawCapabilityReport] = None
    merge_law: Optional[LawCapabilityReport] = None
    idempotence_law: Optional[LawCapabilityReport] = None
    oracle_measurement: Optional[Any] = None
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def _default_law(
        self,
        *,
        law_kind: LawKind,
        available: bool,
    ) -> LawCapabilityReport:
        return LawCapabilityReport(
            law_kind=law_kind,
            available=available,
            evidence_status=self.evidence_status,
            objective_enforced=False,
            exact=None,
        )

    def capability_report(
        self,
        operator_name: str,
        *,
        default_leaf_available: Optional[bool] = None,
        default_merge_available: Optional[bool] = None,
        default_idempotence_available: Optional[bool] = None,
        leaf_law: Optional[LawCapabilityReport] = None,
        merge_law: Optional[LawCapabilityReport] = None,
        idempotence_law: Optional[LawCapabilityReport] = None,
        approx_local_laws: Optional[Any] = None,
        oracle_measurement: Optional[Any] = None,
        extra_notes: Tuple[str, ...] = (),
    ) -> OperatorCapabilityReport:
        """Materialize a full capability report for a concrete operator name."""
        leaf_available = (
            self.theorem_domain_decode_available
            if default_leaf_available is None
            else bool(default_leaf_available)
        )
        merge_available = (
            self.theorem_domain_decode_available
            if default_merge_available is None
            else bool(default_merge_available)
        )
        idempotence_available = (
            self.theorem_domain_decode_available and self.theorem_domain_reencode_available
            if default_idempotence_available is None
            else bool(default_idempotence_available)
        )
        return OperatorCapabilityReport(
            operator_name=operator_name,
            evidence_status=self.evidence_status,
            latent_mergeability_enforced=self.latent_mergeability_enforced,
            tree_nesting_supported=self.tree_nesting_supported,
            theorem_domain_decode_available=self.theorem_domain_decode_available,
            theorem_domain_reencode_available=self.theorem_domain_reencode_available,
            exact_reduction_supported=self.exact_reduction_supported,
            leaf_law=leaf_law or self.leaf_law or self._default_law(
                law_kind=LawKind.L1_LEAF,
                available=leaf_available,
            ),
            merge_law=merge_law or self.merge_law or self._default_law(
                law_kind=LawKind.L2_MERGE,
                available=merge_available,
            ),
            idempotence_law=idempotence_law or self.idempotence_law or self._default_law(
                law_kind=LawKind.L3_IDEMPOTENCE,
                available=idempotence_available,
            ),
            approx_local_laws=approx_local_laws,
            oracle_measurement=oracle_measurement or self.oracle_measurement,
            notes=tuple(self.notes) + tuple(extra_notes),
        )


@dataclass
class FunctionalCompositionalOperator(Generic[SpanT, StateT]):
    """Composable operator built directly from supplied callables."""

    name: str
    encode_fn: Callable[..., StateT]
    merge_fn: Callable[..., StateT]
    decode_fn: Callable[..., SpanT]
    combine_fn: Callable[..., SpanT]
    assumptions: Optional[OperatorAssumptionBundle] = None
    evidence_status: EvidenceStatus = EvidenceStatus.THEOREM_BACKED
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
    theorem_domain_decode_available: bool = True
    theorem_domain_reencode_available: bool = True
    latent_mergeability_enforced: bool = False
    tree_nesting_supported: bool = True
    exact_reduction_supported: bool = False
    approx_local_laws: Optional[Any] = None
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def encode(self, span: SpanT, **kwargs: Any) -> StateT:
        return _call_with_supported_kwargs(self.encode_fn, span, **kwargs)

    def merge(self, left: StateT, right: StateT, **kwargs: Any) -> StateT:
        return _call_with_supported_kwargs(self.merge_fn, left, right, **kwargs)

    def decode(self, state: StateT, **kwargs: Any) -> SpanT:
        return _call_with_supported_kwargs(self.decode_fn, state, **kwargs)

    def combine(self, left_span: SpanT, right_span: SpanT, **kwargs: Any) -> SpanT:
        return _call_with_supported_kwargs(self.combine_fn, left_span, right_span, **kwargs)

    def resummarize(self, span: SpanT, **kwargs: Any) -> SpanT:
        return self.decode(self.encode(span, **kwargs), **kwargs)

    def capability_report(self) -> OperatorCapabilityReport:
        if self.assumptions is not None:
            return self.assumptions.capability_report(
                self.name,
                default_leaf_available=self.theorem_domain_decode_available,
                default_merge_available=self.theorem_domain_decode_available,
                default_idempotence_available=(
                    self.theorem_domain_decode_available and self.theorem_domain_reencode_available
                ),
                leaf_law=None if _law_report_is_placeholder(self.leaf_law) else self.leaf_law,
                merge_law=None if _law_report_is_placeholder(self.merge_law) else self.merge_law,
                idempotence_law=(
                    None
                    if _law_report_is_placeholder(self.idempotence_law)
                    else self.idempotence_law
                ),
                approx_local_laws=self.approx_local_laws,
                oracle_measurement=self.assumptions.oracle_measurement,
                extra_notes=self.notes,
            )
        return OperatorCapabilityReport(
            operator_name=self.name,
            evidence_status=self.evidence_status,
            latent_mergeability_enforced=self.latent_mergeability_enforced,
            tree_nesting_supported=self.tree_nesting_supported,
            theorem_domain_decode_available=self.theorem_domain_decode_available,
            theorem_domain_reencode_available=self.theorem_domain_reencode_available,
            exact_reduction_supported=self.exact_reduction_supported,
            leaf_law=self.leaf_law,
            merge_law=self.merge_law,
            idempotence_law=self.idempotence_law,
            approx_local_laws=self.approx_local_laws,
            oracle_measurement=None,
            notes=self.notes,
        )


def make_text_compositional_operator(
    summarize_fn: Callable[[str, str], str],
    *,
    name: str = "deterministic_summary",
    assumptions: Optional[OperatorAssumptionBundle] = None,
    evidence_status: EvidenceStatus = EvidenceStatus.PROXY_ONLY,
    law_evidence_status: Optional[EvidenceStatus] = None,
    notes: Tuple[str, ...] = (),
) -> FunctionalCompositionalOperator[str, str]:
    """Lift a deterministic text summarizer into the general operator interface."""
    law_status = law_evidence_status or evidence_status
    if assumptions is not None and law_evidence_status is None:
        law_status = assumptions.evidence_status
    default_notes = (
        "This operator exposes the theorem-domain summary space directly as strings.",
        "Certification still depends on external law audits or proofs; exposing the interface alone is not a guarantee.",
    )
    leaf_placeholder = LawCapabilityReport(
        law_kind=LawKind.L1_LEAF,
        available=False,
        evidence_status=EvidenceStatus.PROXY_ONLY,
    )
    merge_placeholder = LawCapabilityReport(
        law_kind=LawKind.L2_MERGE,
        available=False,
        evidence_status=EvidenceStatus.PROXY_ONLY,
    )
    idemp_placeholder = LawCapabilityReport(
        law_kind=LawKind.L3_IDEMPOTENCE,
        available=False,
        evidence_status=EvidenceStatus.PROXY_ONLY,
    )
    return FunctionalCompositionalOperator[str, str](
        name=name,
        encode_fn=lambda span, rubric="", **_: summarize_fn(span, rubric),
        merge_fn=lambda left, right, rubric="", **_: summarize_fn(
            format_merge_input(left, right),
            rubric,
        ),
        decode_fn=lambda sketch, **_: sketch,
        combine_fn=lambda left, right, **_: format_merge_input(str(left), str(right)),
        assumptions=assumptions,
        evidence_status=assumptions.evidence_status if assumptions is not None else evidence_status,
        theorem_domain_decode_available=True,
        theorem_domain_reencode_available=True,
        tree_nesting_supported=True,
        exact_reduction_supported=False,
        leaf_law=(
            leaf_placeholder
            if assumptions is not None
            else LawCapabilityReport(
                law_kind=LawKind.L1_LEAF,
                available=True,
                evidence_status=law_status,
                exact=False,
                notes="Leaf preservation is evaluable because the theorem-domain summary operator is explicit.",
            )
        ),
        merge_law=(
            merge_placeholder
            if assumptions is not None
            else LawCapabilityReport(
                law_kind=LawKind.L2_MERGE,
                available=True,
                evidence_status=law_status,
                exact=False,
                notes="Merge preservation is evaluable as g(format_merge_input(s_L, s_R)).",
            )
        ),
        idempotence_law=(
            idemp_placeholder
            if assumptions is not None
            else LawCapabilityReport(
                law_kind=LawKind.L3_IDEMPOTENCE,
                available=True,
                evidence_status=law_status,
                exact=False,
                notes="On-range idempotence is evaluable as repeated calls to the same summarizer g.",
            )
        ),
        notes=default_notes + tuple(notes),
    )


def make_deterministic_summary_operator(
    summarize_fn: Callable[[str, str], str],
    *,
    name: str = "deterministic_summary",
    assumptions: Optional[OperatorAssumptionBundle] = None,
    evidence_status: EvidenceStatus = EvidenceStatus.PROXY_ONLY,
    law_evidence_status: Optional[EvidenceStatus] = None,
    notes: Tuple[str, ...] = (),
) -> FunctionalCompositionalOperator[str, str]:
    """Compatibility wrapper for the historical summary-operator name."""
    return make_text_compositional_operator(
        summarize_fn,
        name=name,
        assumptions=assumptions,
        evidence_status=evidence_status,
        law_evidence_status=law_evidence_status,
        notes=notes,
    )


@dataclass
class ModelCompositionalOperatorAdapter(Generic[SpanT, StateT]):
    """Wrap any supplied encode/decode/merge codec into a compositional operator."""

    model: Any
    name: str = "model_compositional_operator"
    encode_method: str = "encode_summary"
    decode_method: str = "decode_summary"
    merge_method: str = "merge"
    combine_fn: Callable[..., SpanT] = field(
        default=lambda left, right, **_: format_merge_input(str(left), str(right))
    )
    assumptions: Optional[OperatorAssumptionBundle] = None
    evidence_status: EvidenceStatus = EvidenceStatus.PROXY_ONLY
    law_evidence_status: Optional[EvidenceStatus] = None
    latent_mergeability_enforced: bool = True
    tree_nesting_supported: bool = True
    exact_reduction_supported: bool = False
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self._law_status = self.law_evidence_status or self.evidence_status

    def _call_model(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(self.model, method_name)
        return _call_with_supported_kwargs(fn, *args, **kwargs)

    def encode(self, span: SpanT, **kwargs: Any) -> StateT:
        return self._call_model(self.encode_method, span, **kwargs)

    def merge(self, left: StateT, right: StateT, **kwargs: Any) -> StateT:
        return self._call_model(self.merge_method, left, right, **kwargs)

    def decode(self, state: StateT, **kwargs: Any) -> SpanT:
        return self._call_model(self.decode_method, state, **kwargs)

    def combine(self, left_span: SpanT, right_span: SpanT, **kwargs: Any) -> SpanT:
        return _call_with_supported_kwargs(self.combine_fn, left_span, right_span, **kwargs)

    def resummarize(self, span: SpanT, **kwargs: Any) -> SpanT:
        return self.decode(self.encode(span, **kwargs), **kwargs)

    def capability_report(self) -> OperatorCapabilityReport:
        adapter_notes = (
            "This adapter assumes a supplied encode/decode/merge codec with the declared properties.",
        ) + tuple(self.notes)
        if self.assumptions is not None:
            return self.assumptions.capability_report(
                self.name,
                default_leaf_available=True,
                default_merge_available=True,
                default_idempotence_available=True,
                oracle_measurement=self.assumptions.oracle_measurement,
                extra_notes=adapter_notes,
            )
        return OperatorCapabilityReport(
            operator_name=self.name,
            evidence_status=self.evidence_status,
            latent_mergeability_enforced=self.latent_mergeability_enforced,
            tree_nesting_supported=self.tree_nesting_supported,
            theorem_domain_decode_available=True,
            theorem_domain_reencode_available=True,
            exact_reduction_supported=self.exact_reduction_supported,
            leaf_law=LawCapabilityReport(
                law_kind=LawKind.L1_LEAF,
                available=True,
                evidence_status=self._law_status,
                exact=False,
                notes="Leaf preservation is evaluable in the theorem-domain summary space.",
            ),
            merge_law=LawCapabilityReport(
                law_kind=LawKind.L2_MERGE,
                available=True,
                evidence_status=self._law_status,
                exact=False,
                notes="Merge preservation is evaluable through decode(merge(encode(.), encode(.))).",
            ),
            idempotence_law=LawCapabilityReport(
                law_kind=LawKind.L3_IDEMPOTENCE,
                available=True,
                evidence_status=self._law_status,
                exact=False,
                notes="Idempotence is evaluable through decode(encode(summary)).",
            ),
            oracle_measurement=None,
            notes=adapter_notes,
        )


@dataclass
class CompositionalPredictorAdapter(Generic[SpanT, StateT, ReadoutStateT]):
    """Pair a supplied theorem operator with any state-based predictor."""

    operator: CompositionalOperator[SpanT, StateT]
    predictor: StatePredictor[ReadoutStateT]
    state_projector: Optional[Callable[..., ReadoutStateT]] = None
    name: Optional[str] = None
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = f"{self.predictor.name}_with_{self.operator.name}"
        self.evidence_status = getattr(
            self.operator,
            "evidence_status",
            EvidenceStatus.PROXY_ONLY,
        )
        self.prediction_evidence_status = getattr(
            self.predictor,
            "evidence_status",
            EvidenceStatus.PROXY_ONLY,
        )
        self.state_dim = int(getattr(self.predictor, "state_dim", 0))

    def encode(self, span: SpanT, **kwargs: Any) -> StateT:
        return self.operator.encode(span, **kwargs)

    def merge(self, left: StateT, right: StateT, **kwargs: Any) -> StateT:
        return self.operator.merge(left, right, **kwargs)

    def decode(self, state: StateT, **kwargs: Any) -> SpanT:
        return self.operator.decode(state, **kwargs)

    def combine(self, left_span: SpanT, right_span: SpanT, **kwargs: Any) -> SpanT:
        return self.operator.combine(left_span, right_span, **kwargs)

    def resummarize(self, span: SpanT, **kwargs: Any) -> SpanT:
        return self.operator.resummarize(span, **kwargs)

    def capability_report(self) -> OperatorCapabilityReport:
        return self.operator.capability_report()

    def _project_state(self, state: StateT, **kwargs: Any) -> ReadoutStateT:
        if self.state_projector is None:
            return state  # type: ignore[return-value]
        return _call_with_supported_kwargs(self.state_projector, state, **kwargs)

    def predict_from_state(self, state: StateT, **kwargs: Any) -> OperatorPrediction:
        projected_state = self._project_state(state, **kwargs)
        return self.predictor.predict_from_state(projected_state, **kwargs)

    def predict_from_span(self, span: SpanT, **kwargs: Any) -> OperatorPrediction:
        return self.predict_from_state(self.encode(span, **kwargs), **kwargs)

    def predictor_summary(self) -> Dict[str, Any]:
        return {
            "name": getattr(self.predictor, "name", "unknown_predictor"),
            "prediction_evidence_status": self.prediction_evidence_status.value,
            "state_projector_supplied": self.state_projector is not None,
            "notes": list(self.notes),
        }


def attach_compositional_operator(
    predictor: StatePredictor[ReadoutStateT],
    operator: CompositionalOperator[SpanT, StateT],
    *,
    state_projector: Optional[Callable[..., ReadoutStateT]] = None,
    name: Optional[str] = None,
    notes: Tuple[str, ...] = (),
) -> CompositionalPredictorAdapter[SpanT, StateT, ReadoutStateT]:
    """Bind a supplied theorem operator to a predictor over the same state family."""
    return CompositionalPredictorAdapter(
        operator=operator,
        predictor=predictor,
        state_projector=state_projector,
        name=name,
        notes=notes,
    )


def attach_theorem_operator(
    predictor: StatePredictor[ReadoutStateT],
    operator: CompositionalOperator[SpanT, StateT],
    *,
    state_projector: Optional[Callable[..., ReadoutStateT]] = None,
    name: Optional[str] = None,
    notes: Tuple[str, ...] = (),
) -> CompositionalPredictorAdapter[SpanT, StateT, ReadoutStateT]:
    """Compatibility alias emphasizing that theorem-backedness comes from the operator."""
    return attach_compositional_operator(
        predictor,
        operator,
        state_projector=state_projector,
        name=name,
        notes=notes,
    )


# Compatibility aliases for the previous theorem/sketch naming.
ReductionOperator = CompositionalOperator
FunctionalReductionOperator = FunctionalCompositionalOperator
CodecReductionOperatorAdapter = ModelCompositionalOperatorAdapter
SketchLawOperator = CompositionalOperator
FunctionalSketchLawOperator = FunctionalCompositionalOperator
SummaryAutoencoderOperatorAdapter = ModelCompositionalOperatorAdapter
ProxyOperator = StatePredictor


__all__ = [
    "CompositionalOperator",
    "ReductionOperator",
    "StatePredictor",
    "ProxyOperator",
    "OperatorAssumptionBundle",
    "OperatorPrediction",
    "FunctionalCompositionalOperator",
    "FunctionalReductionOperator",
    "ModelCompositionalOperatorAdapter",
    "CodecReductionOperatorAdapter",
    "CompositionalPredictorAdapter",
    "make_text_compositional_operator",
    "make_deterministic_summary_operator",
    "attach_compositional_operator",
    "attach_theorem_operator",
    "SketchLawOperator",
    "FunctionalSketchLawOperator",
    "SummaryAutoencoderOperatorAdapter",
]
