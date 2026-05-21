"""Canonical metadata for scalar and comparative supervision judgments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Mapping, Optional

JUDGMENT_SUPERVISION_CHANNEL_NAME = "judgment_supervision"
JUDGMENT_SUPERVISION_SIGNAL_NAME = "judgment"


def normalize_judgment_law_type(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    alias_map = {
        "l1_leaf": "sufficiency",
        "sufficiency": "sufficiency",
        "leaf": "sufficiency",
        "l2_merge": "merge",
        "merge_consistency": "merge",
        "merge": "merge",
        "l3_idempotence": "idempotence",
        "idempotence": "idempotence",
        "substitution": "substitution",
    }
    return alias_map.get(text, text)


@dataclass(frozen=True)
class JudgmentSupervisionMetadata:
    """Shared metadata carried by scalar and comparative supervision records."""

    application_name: str = "supervision_pipeline"
    supervision_channel_name: str = JUDGMENT_SUPERVISION_CHANNEL_NAME
    supervision_signal_name: str = JUDGMENT_SUPERVISION_SIGNAL_NAME
    preference_family: str = "judgment"
    law_type: Optional[str] = None
    comparison_signal_name: Optional[str] = None
    comparison_signal_min: Optional[float] = None
    comparison_signal_max: Optional[float] = None
    response_signal_name: Optional[str] = None
    response_signal_min: Optional[float] = None
    response_signal_max: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "application_name", str(self.application_name))
        object.__setattr__(
            self,
            "supervision_channel_name",
            str(self.supervision_channel_name),
        )
        object.__setattr__(
            self,
            "supervision_signal_name",
            str(self.supervision_signal_name),
        )
        object.__setattr__(self, "preference_family", str(self.preference_family))
        object.__setattr__(
            self,
            "law_type",
            normalize_judgment_law_type(self.law_type),
        )
        if self.comparison_signal_name is not None:
            object.__setattr__(
                self,
                "comparison_signal_name",
                str(self.comparison_signal_name),
            )
        if self.comparison_signal_min is not None:
            object.__setattr__(
                self,
                "comparison_signal_min",
                float(self.comparison_signal_min),
            )
        if self.comparison_signal_max is not None:
            object.__setattr__(
                self,
                "comparison_signal_max",
                float(self.comparison_signal_max),
            )
        if self.response_signal_name is not None:
            object.__setattr__(
                self,
                "response_signal_name",
                str(self.response_signal_name),
            )
        if self.response_signal_min is not None:
            object.__setattr__(
                self,
                "response_signal_min",
                float(self.response_signal_min),
            )
        if self.response_signal_max is not None:
            object.__setattr__(
                self,
                "response_signal_max",
                float(self.response_signal_max),
            )
        if not isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", dict(self.metadata))

    def with_updates(self, **kwargs: Any) -> "JudgmentSupervisionMetadata":
        return replace(self, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in (
            "law_type",
            "comparison_signal_name",
            "comparison_signal_min",
            "comparison_signal_max",
            "response_signal_name",
            "response_signal_min",
            "response_signal_max",
        ):
            if payload.get(key) is None:
                payload.pop(key, None)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JudgmentSupervisionMetadata":
        data = dict(payload or {})
        return cls(
            application_name=str(data.get("application_name", "supervision_pipeline")),
            supervision_channel_name=str(
                data.get("supervision_channel_name", JUDGMENT_SUPERVISION_CHANNEL_NAME)
            ),
            supervision_signal_name=str(
                data.get("supervision_signal_name", JUDGMENT_SUPERVISION_SIGNAL_NAME)
            ),
            preference_family=str(data.get("preference_family", "judgment")),
            law_type=data.get("law_type"),
            comparison_signal_name=data.get("comparison_signal_name"),
            comparison_signal_min=data.get("comparison_signal_min"),
            comparison_signal_max=data.get("comparison_signal_max"),
            response_signal_name=data.get("response_signal_name"),
            response_signal_min=data.get("response_signal_min"),
            response_signal_max=data.get("response_signal_max"),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def judgment_supervision_metadata(
    *,
    law_type: Optional[str] = None,
    application_name: str = "supervision_pipeline",
    supervision_channel_name: str = JUDGMENT_SUPERVISION_CHANNEL_NAME,
    supervision_signal_name: str = JUDGMENT_SUPERVISION_SIGNAL_NAME,
    preference_family: str = "judgment",
    comparison_signal_name: Optional[str] = None,
    comparison_signal_min: Optional[float] = None,
    comparison_signal_max: Optional[float] = None,
    response_signal_name: Optional[str] = None,
    response_signal_min: Optional[float] = None,
    response_signal_max: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> JudgmentSupervisionMetadata:
    return JudgmentSupervisionMetadata(
        application_name=application_name,
        supervision_channel_name=supervision_channel_name,
        supervision_signal_name=supervision_signal_name,
        preference_family=preference_family,
        law_type=law_type,
        comparison_signal_name=comparison_signal_name,
        comparison_signal_min=comparison_signal_min,
        comparison_signal_max=comparison_signal_max,
        response_signal_name=response_signal_name,
        response_signal_min=response_signal_min,
        response_signal_max=response_signal_max,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "JUDGMENT_SUPERVISION_CHANNEL_NAME",
    "JUDGMENT_SUPERVISION_SIGNAL_NAME",
    "JudgmentSupervisionMetadata",
    "judgment_supervision_metadata",
    "normalize_judgment_law_type",
]
