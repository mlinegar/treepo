"""Sampling metadata and inverse-propensity weighting records.

Defines ``SamplingMetadata`` and ``DocumentSamplingRow`` with propensity
normalization and IPW weight computation used by preference/observation
exports.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

from treepo.common import MIN_PROPENSITY, jsonable as _jsonable


DEFAULT_PROPENSITY = 1.0


class ObservationUnitKind(str, Enum):
    DOCUMENT = "document"
    LEAF = "leaf"
    INTERNAL = "internal"
    MERGE = "merge"
    RESUMMARY = "resummary"
    SUBSTITUTION = "substitution"
    PAIR = "pair"


def _normalize_propensity(value: float | None, name: str) -> float:
    if value is None:
        return DEFAULT_PROPENSITY
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1], got {value!r}")
    return parsed


@dataclass(frozen=True)
class SamplingMetadata:
    document_propensity: float = DEFAULT_PROPENSITY
    unit_propensity: float = DEFAULT_PROPENSITY
    label_propensity: float = DEFAULT_PROPENSITY
    joint_propensity: float | None = None
    sampling_scheme: str = ""
    policy_name: str = ""
    unit_kind: ObservationUnitKind | None = None
    supports_ipw_estimation: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_propensity",
            _normalize_propensity(self.document_propensity, "document_propensity"),
        )
        object.__setattr__(
            self,
            "unit_propensity",
            _normalize_propensity(self.unit_propensity, "unit_propensity"),
        )
        object.__setattr__(
            self,
            "label_propensity",
            _normalize_propensity(self.label_propensity, "label_propensity"),
        )
        if self.joint_propensity is not None:
            object.__setattr__(
                self,
                "joint_propensity",
                _normalize_propensity(self.joint_propensity, "joint_propensity"),
            )
        if self.unit_kind is not None and not isinstance(self.unit_kind, ObservationUnitKind):
            object.__setattr__(self, "unit_kind", ObservationUnitKind(str(self.unit_kind)))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def effective_joint_propensity(self, *, min_propensity: float = MIN_PROPENSITY) -> float:
        if self.joint_propensity is not None:
            joint = float(self.joint_propensity)
        else:
            joint = (
                float(self.document_propensity)
                * float(self.unit_propensity)
                * float(self.label_propensity)
            )
        return max(float(min_propensity), float(joint))

    def ipw_weight(
        self,
        *,
        min_propensity: float = MIN_PROPENSITY,
        max_weight: float | None = None,
    ) -> float:
        weight = 1.0 / self.effective_joint_propensity(min_propensity=min_propensity)
        if max_weight is not None:
            weight = min(weight, float(max_weight))
        return float(weight)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class DocumentSamplingRow:
    top_level_unit_id: str
    observed: bool
    inclusion_probability: float
    prediction: float | None = None
    predicted_var: float | None = None
    truth: float | None = None
    split: str = ""
    fold_id: str = "0"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.top_level_unit_id):
            raise ValueError("top_level_unit_id is required")
        p = float(self.inclusion_probability)
        if bool(self.observed):
            _normalize_propensity(p, "inclusion_probability")
        elif not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"inclusion_probability must be in [0, 1], got {p!r}")
        object.__setattr__(self, "top_level_unit_id", str(self.top_level_unit_id))
        object.__setattr__(self, "inclusion_probability", p)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def sampling(self) -> SamplingMetadata:
        return SamplingMetadata(
            document_propensity=max(float(self.inclusion_probability), MIN_PROPENSITY),
            unit_propensity=1.0,
            label_propensity=1.0,
            unit_kind=ObservationUnitKind.DOCUMENT,
            sampling_scheme="document",
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


__all__ = [
    "DEFAULT_PROPENSITY",
    "MIN_PROPENSITY",
    "DocumentSamplingRow",
    "ObservationUnitKind",
    "SamplingMetadata",
]
