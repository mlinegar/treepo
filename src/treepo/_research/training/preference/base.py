"""Internal compatibility wrapper over ``src.training.supervision.base``."""

from treepo._research.training.supervision.base import (
    BasePreferenceCollector,
    CandidateInfo,
    CollectionStatistics,
    PreferenceResult,
)

__all__ = [
    "BasePreferenceCollector",
    "CandidateInfo",
    "CollectionStatistics",
    "PreferenceResult",
]
