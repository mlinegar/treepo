"""Unified learning error certificates.

Aggregates per-component error evidence (local-law, calibration, estimation,
clipping) into a ``UnifiedLearningErrorCertificate`` with additive radii around
a reported estimate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from treepo.common import finite_float

COMPONENT_LOCAL_LAW = "local_law"
COMPONENT_CALIBRATION = "calibration"
COMPONENT_ESTIMATION = "estimation"
COMPONENT_CLIPPING = "clipping"
CERTIFICATE_COMPONENTS = (
    COMPONENT_LOCAL_LAW,
    COMPONENT_CALIBRATION,
    COMPONENT_ESTIMATION,
    COMPONENT_CLIPPING,
)


@dataclass(frozen=True)
class UnifiedLearningComponentEvidence:
    component: str
    radius: float
    estimate: float | None = None
    delta: float | None = None
    source: str = ""
    artifact_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        component = str(self.component or "").strip().lower()
        if component not in CERTIFICATE_COMPONENTS:
            raise ValueError(f"unknown certificate component: {self.component!r}")
        radius = finite_float(self.radius, name="radius")
        if radius < 0.0:
            raise ValueError("component radius must be non-negative")
        if self.delta is not None:
            delta = finite_float(self.delta, name="delta")
            if delta < 0.0 or delta > 1.0:
                raise ValueError("delta must be in [0, 1]")
            object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "artifact_ids", tuple(str(x) for x in self.artifact_ids))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnifiedLearningErrorCertificate:
    reported_estimate: float
    local_law_radius: float = 0.0
    calibration_radius: float = 0.0
    estimation_radius: float = 0.0
    clipping_radius: float = 0.0
    component_evidence: tuple[UnifiedLearningComponentEvidence, ...] = ()
    confidence_delta: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "reported_estimate",
            "local_law_radius",
            "calibration_radius",
            "estimation_radius",
            "clipping_radius",
        ):
            value = finite_float(getattr(self, name), name=name)
            if name.endswith("_radius") and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.confidence_delta is not None:
            delta = finite_float(self.confidence_delta, name="confidence_delta")
            if delta < 0.0 or delta > 1.0:
                raise ValueError("confidence_delta must be in [0, 1]")
            object.__setattr__(self, "confidence_delta", delta)
        object.__setattr__(
            self,
            "component_evidence",
            tuple(
                item
                if isinstance(item, UnifiedLearningComponentEvidence)
                else UnifiedLearningComponentEvidence(**dict(item))
                for item in self.component_evidence
            ),
        )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def radius_sum(self) -> float:
        return float(
            self.local_law_radius
            + self.calibration_radius
            + self.estimation_radius
            + self.clipping_radius
        )

    @property
    def total_bound(self) -> float:
        return float(abs(self.reported_estimate) + self.radius_sum)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["component_evidence"] = [item.to_dict() for item in self.component_evidence]
        payload["radius_sum"] = self.radius_sum
        payload["total_bound"] = self.total_bound
        return payload


@dataclass(frozen=True)
class TwoChannelResidual:
    """Leaf-up, root-down, and overidentification residual radii."""

    leaf_up_radius: float = 0.0
    root_down_radius: float = 0.0
    overidentification_radius: float = 0.0
    delta: float | None = None
    source: str = ""
    artifact_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("leaf_up_radius", "root_down_radius", "overidentification_radius"):
            value = finite_float(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "delta", _validated_delta(self.delta, name="delta"))
        object.__setattr__(self, "artifact_ids", tuple(str(x) for x in self.artifact_ids))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def total_radius(self) -> float:
        return float(self.leaf_up_radius + self.root_down_radius + self.overidentification_radius)

    def component_evidence(self) -> tuple[UnifiedLearningComponentEvidence, ...]:
        base_metadata = dict(self.metadata or {})
        return (
            UnifiedLearningComponentEvidence(
                component=COMPONENT_LOCAL_LAW,
                radius=float(self.leaf_up_radius),
                delta=self.delta,
                source=self.source,
                artifact_ids=self.artifact_ids,
                metadata={
                    **base_metadata,
                    "semantic_component": "leaf_up",
                    "observation_channel": "leaf_up",
                },
            ),
            UnifiedLearningComponentEvidence(
                component=COMPONENT_CALIBRATION,
                radius=float(self.root_down_radius),
                delta=self.delta,
                source=self.source,
                artifact_ids=self.artifact_ids,
                metadata={
                    **base_metadata,
                    "semantic_component": "root_down",
                    "observation_channel": "root_down",
                },
            ),
            UnifiedLearningComponentEvidence(
                component=COMPONENT_ESTIMATION,
                radius=float(self.overidentification_radius),
                delta=self.delta,
                source=self.source,
                artifact_ids=self.artifact_ids,
                metadata={
                    **base_metadata,
                    "semantic_component": "overidentification",
                    "observation_channel": "leaf_up_vs_root_down",
                },
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["total_radius"] = self.total_radius
        return payload


@dataclass(frozen=True)
class ConditionalAverageEnvelopeEvidence:
    """One-sided degradation envelope supplied by an external workflow.

    This is a certificate input, not a Bayesian/MRP implementation. The caller
    is responsible for fitting any multilevel model, computing the one-sided
    bound, and deciding that the diagnostics below are satisfied.
    """

    degradation_radius: float
    posterior_predictive_fit: bool
    psis_loo_stable: bool
    rank_calibrated: bool
    delta: float | None = None
    source: str = ""
    artifact_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        radius = finite_float(self.degradation_radius, name="degradation_radius")
        if radius < 0.0:
            raise ValueError("degradation_radius must be non-negative")
        object.__setattr__(self, "degradation_radius", radius)
        for name in ("posterior_predictive_fit", "psis_loo_stable", "rank_calibrated"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
        object.__setattr__(self, "delta", _validated_delta(self.delta, name="delta"))
        object.__setattr__(self, "artifact_ids", tuple(str(x) for x in self.artifact_ids))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def diagnostics_satisfied(self) -> bool:
        return bool(self.posterior_predictive_fit and self.psis_loo_stable and self.rank_calibrated)

    def component_evidence(
        self,
        *,
        require_diagnostics: bool = True,
    ) -> UnifiedLearningComponentEvidence:
        if require_diagnostics and not self.diagnostics_satisfied:
            raise ValueError("conditional-average envelope diagnostics are not all satisfied")
        diagnostics = {
            "posterior_predictive_fit": bool(self.posterior_predictive_fit),
            "psis_loo_stable": bool(self.psis_loo_stable),
            "rank_calibrated": bool(self.rank_calibrated),
            "diagnostics_satisfied": self.diagnostics_satisfied,
        }
        return UnifiedLearningComponentEvidence(
            component=COMPONENT_ESTIMATION,
            radius=float(self.degradation_radius),
            delta=self.delta,
            source=self.source,
            artifact_ids=self.artifact_ids,
            metadata={
                **dict(self.metadata or {}),
                "semantic_component": "conditional_average_envelope",
                "observation_channel": "non_additive_degradation",
                "one_sided": True,
                "envelope_source": "external_workflow",
                "model_fitted_by_treepo": False,
                "diagnostics": diagnostics,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["diagnostics_satisfied"] = self.diagnostics_satisfied
        return payload


@dataclass(frozen=True)
class CommonMechanismEnvelopeEvidence:
    """Hidden-degradation envelope induced by observed root errors.

    This is the non-additive analogue of the document-level bounding argument:
    if the same f and g are used at roots and internal nodes, and the local laws
    transport those calls into a common mechanism, an observed root-error bound
    can be amplified into a hidden-degradation bound.
    """

    observed_root_radius: float
    amplification: float = 1.0
    slack: float = 0.0
    common_f: bool = True
    common_g: bool = True
    local_law_transport: bool = True
    delta: float | None = None
    source: str = ""
    artifact_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("observed_root_radius", "amplification", "slack"):
            value = finite_float(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in ("common_f", "common_g", "local_law_transport"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
        object.__setattr__(self, "delta", _validated_delta(self.delta, name="delta"))
        object.__setattr__(self, "artifact_ids", tuple(str(x) for x in self.artifact_ids))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def assumptions_satisfied(self) -> bool:
        return bool(self.common_f and self.common_g and self.local_law_transport)

    @property
    def degradation_radius(self) -> float:
        return float(self.amplification * self.observed_root_radius + self.slack)

    def component_evidence(
        self,
        *,
        require_assumptions: bool = True,
    ) -> UnifiedLearningComponentEvidence:
        if require_assumptions and not self.assumptions_satisfied:
            raise ValueError("common-mechanism envelope assumptions are not all satisfied")
        assumptions = {
            "common_f": bool(self.common_f),
            "common_g": bool(self.common_g),
            "local_law_transport": bool(self.local_law_transport),
            "assumptions_satisfied": self.assumptions_satisfied,
        }
        return UnifiedLearningComponentEvidence(
            component=COMPONENT_ESTIMATION,
            radius=self.degradation_radius,
            delta=self.delta,
            source=self.source,
            artifact_ids=self.artifact_ids,
            metadata={
                **dict(self.metadata or {}),
                "semantic_component": "common_mechanism_envelope",
                "observation_channel": "root_error_to_hidden_degradation",
                "envelope_source": "observed_root_errors",
                "model_fitted_by_treepo": False,
                "transport_source": "merge_triangle_local_laws",
                "root_control_source": "audit_bound",
                "observed_root_radius": float(self.observed_root_radius),
                "amplification": float(self.amplification),
                "slack": float(self.slack),
                "assumptions": assumptions,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["assumptions_satisfied"] = self.assumptions_satisfied
        payload["degradation_radius"] = self.degradation_radius
        return payload


def build_error_certificate(
    *,
    reported_estimate: float,
    component_evidence: Sequence[UnifiedLearningComponentEvidence | Mapping[str, Any]],
    confidence_delta: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> UnifiedLearningErrorCertificate:
    """Sum component evidence radii into a unified error certificate."""
    evidence = tuple(
        item
        if isinstance(item, UnifiedLearningComponentEvidence)
        else UnifiedLearningComponentEvidence(**dict(item))
        for item in component_evidence
    )
    radius_by_component = {name: 0.0 for name in CERTIFICATE_COMPONENTS}
    for item in evidence:
        radius_by_component[item.component] += float(item.radius)
    return UnifiedLearningErrorCertificate(
        reported_estimate=float(reported_estimate),
        local_law_radius=radius_by_component[COMPONENT_LOCAL_LAW],
        calibration_radius=radius_by_component[COMPONENT_CALIBRATION],
        estimation_radius=radius_by_component[COMPONENT_ESTIMATION],
        clipping_radius=radius_by_component[COMPONENT_CLIPPING],
        component_evidence=evidence,
        confidence_delta=confidence_delta,
        metadata=dict(metadata or {}),
    )


def build_two_channel_error_certificate(
    *,
    reported_estimate: float,
    residual: TwoChannelResidual | Mapping[str, Any] | None = None,
    common_mechanism_envelopes: Sequence[CommonMechanismEnvelopeEvidence | Mapping[str, Any]] = (),
    conditional_envelopes: Sequence[ConditionalAverageEnvelopeEvidence | Mapping[str, Any]] = (),
    confidence_delta: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    require_common_mechanism_assumptions: bool = True,
    require_conditional_diagnostics: bool = True,
) -> UnifiedLearningErrorCertificate:
    """Build a unified certificate from two-channel residual evidence."""

    residual_evidence = (
        residual
        if isinstance(residual, TwoChannelResidual)
        else TwoChannelResidual(**dict(residual or {}))
    )
    envelopes = tuple(
        item
        if isinstance(item, ConditionalAverageEnvelopeEvidence)
        else ConditionalAverageEnvelopeEvidence(**dict(item))
        for item in (conditional_envelopes or ())
    )
    common_envelopes = tuple(
        item
        if isinstance(item, CommonMechanismEnvelopeEvidence)
        else CommonMechanismEnvelopeEvidence(**dict(item))
        for item in (common_mechanism_envelopes or ())
    )
    component_evidence = list(residual_evidence.component_evidence())
    component_evidence.extend(
        envelope.component_evidence(
            require_assumptions=require_common_mechanism_assumptions,
        )
        for envelope in common_envelopes
    )
    component_evidence.extend(
        envelope.component_evidence(require_diagnostics=require_conditional_diagnostics)
        for envelope in envelopes
    )
    certificate_metadata = {
        **dict(metadata or {}),
        "certificate_kind": "two_channel",
        "two_channel_residual": residual_evidence.to_dict(),
        "common_mechanism_envelopes": [item.to_dict() for item in common_envelopes],
        "conditional_envelopes": [item.to_dict() for item in envelopes],
    }
    return build_error_certificate(
        reported_estimate=reported_estimate,
        component_evidence=component_evidence,
        confidence_delta=confidence_delta,
        metadata=certificate_metadata,
    )


def _validated_delta(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    delta = finite_float(value, name=name)
    if delta < 0.0 or delta > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return float(delta)


__all__ = [
    "CERTIFICATE_COMPONENTS",
    "COMPONENT_CALIBRATION",
    "COMPONENT_CLIPPING",
    "COMPONENT_ESTIMATION",
    "COMPONENT_LOCAL_LAW",
    "CommonMechanismEnvelopeEvidence",
    "ConditionalAverageEnvelopeEvidence",
    "TwoChannelResidual",
    "UnifiedLearningComponentEvidence",
    "UnifiedLearningErrorCertificate",
    "build_error_certificate",
    "build_two_channel_error_certificate",
]
