"""Judge-facing wrapper over the internal GenRM DSPy modules."""

from treepo._research.training.preference.genrm_dspy import (
    GenRMComparisonModule,
    GenRMComparisonSignature,
    GenRMPromptSignature,
)

__all__ = [
    "GenRMComparisonModule",
    "GenRMComparisonSignature",
    "GenRMPromptSignature",
]
