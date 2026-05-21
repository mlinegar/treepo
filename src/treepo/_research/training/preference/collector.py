"""Internal compatibility wrapper over ``src.training.supervision.collector``."""

from treepo._research.training.supervision.collector import (
    PairwiseJudge,
    PreferenceCollector,
)
from treepo._research.training.supervision.comparative_types import (
    GenerationConfig,
    PreferenceDataset,
)

__all__ = [
    "GenerationConfig",
    "PairwiseJudge",
    "PreferenceDataset",
    "PreferenceCollector",
]
