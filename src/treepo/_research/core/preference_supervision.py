"""Compatibility wrapper over the primary supervision metadata surface."""

from __future__ import annotations

from typing import Any, Dict, Optional

from treepo._research.core.supervision_metadata import (
    JudgmentSupervisionMetadata,
    judgment_supervision_metadata,
    normalize_judgment_law_type,
)

PREFERENCE_SUPERVISION_CHANNEL_NAME = "pairwise_preference_supervision"
PREFERENCE_SUPERVISION_SIGNAL_NAME = "pairwise_preference"

# Thin compatibility alias. New code should prefer JudgmentSupervisionMetadata.
PreferenceSupervisionMetadata = JudgmentSupervisionMetadata


def normalize_preference_law_type(value: Optional[str]) -> Optional[str]:
    return normalize_judgment_law_type(value)


def preference_supervision_metadata(
    *,
    law_type: Optional[str] = None,
    application_name: str = "preference_pipeline",
    comparison_signal_name: Optional[str] = None,
    comparison_signal_min: Optional[float] = None,
    comparison_signal_max: Optional[float] = None,
    response_signal_name: Optional[str] = None,
    response_signal_min: Optional[float] = None,
    response_signal_max: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PreferenceSupervisionMetadata:
    return judgment_supervision_metadata(
        application_name=application_name,
        supervision_channel_name=PREFERENCE_SUPERVISION_CHANNEL_NAME,
        supervision_signal_name=PREFERENCE_SUPERVISION_SIGNAL_NAME,
        preference_family="pairwise",
        law_type=law_type,
        comparison_signal_name=comparison_signal_name,
        comparison_signal_min=comparison_signal_min,
        comparison_signal_max=comparison_signal_max,
        response_signal_name=response_signal_name,
        response_signal_min=response_signal_min,
        response_signal_max=response_signal_max,
        metadata=metadata,
    )


__all__ = [
    "PREFERENCE_SUPERVISION_CHANNEL_NAME",
    "PREFERENCE_SUPERVISION_SIGNAL_NAME",
    "PreferenceSupervisionMetadata",
    "normalize_preference_law_type",
    "preference_supervision_metadata",
]
