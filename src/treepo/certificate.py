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
CERTIFICATE_LIMITATION = (
    "v0.1 certificates are component-radius ledgers. They do not separately "
    "instantiate the Lean Lipschitz-readout or measurement-error terms; include "
    "those constants in the supplied component radius when they are needed."
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


def build_error_certificate(
    *,
    reported_estimate: float,
    component_evidence: Sequence[UnifiedLearningComponentEvidence | Mapping[str, Any]],
    confidence_delta: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> UnifiedLearningErrorCertificate:
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


__all__ = [
    "CERTIFICATE_COMPONENTS",
    "CERTIFICATE_LIMITATION",
    "COMPONENT_CALIBRATION",
    "COMPONENT_CLIPPING",
    "COMPONENT_ESTIMATION",
    "COMPONENT_LOCAL_LAW",
    "UnifiedLearningComponentEvidence",
    "UnifiedLearningErrorCertificate",
    "build_error_certificate",
]
