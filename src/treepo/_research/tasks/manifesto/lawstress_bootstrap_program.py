"""DSPy programs for LawStress local-law bootstrap optimization.

This provides:
- `UnifiedG`: a single summarizer g used for both leaf + merge inputs.
- `LawStressLocalLawProgram`: a small wrapper that produces the specific
  summaries needed to score C1/C2/C3-style local laws without extra plumbing.
"""

from __future__ import annotations

from typing import Optional

import dspy

from treepo._research.core.protocols import format_merge_input
from treepo._research.core.signatures import RecursiveSummary
from treepo._research.core.summarization import GenericSummarizer


class UnifiedG(dspy.Module):
    """Unified g: Strings -> Strings, used for both leaves and merges."""

    def __init__(self) -> None:
        super().__init__()
        # Avoid leading-underscore attribute names so DSPy serialization reliably
        # discovers nested modules/predictors.
        self.summarizer = GenericSummarizer(signature_class=RecursiveSummary, use_cot=False)

    def forward(self, content: str, rubric: str) -> str:
        return self.summarizer(content=content, rubric=rubric)


class LawStressLocalLawProgram(dspy.Module):
    """Generate summaries needed to evaluate local-law objectives on LawStress."""

    def __init__(self, g: Optional[UnifiedG] = None) -> None:
        super().__init__()
        self.g = g or UnifiedG()

    def forward(
        self,
        text: str,
        segment_a: str,
        segment_b: str,
        law_target: str,
        rubric: str,
    ) -> dspy.Prediction:
        target = str(law_target or "").strip().lower()

        if target == "c1_sufficiency":
            summary1 = self.g(content=text, rubric=rubric)
            return dspy.Prediction(summary1=summary1)

        if target == "c2_idempotence":
            summary1 = self.g(content=text, rubric=rubric)
            summary2 = self.g(content=summary1, rubric=rubric)
            return dspy.Prediction(summary1=summary1, summary2=summary2)

        # Default: c3_merge
        summary_a = self.g(content=segment_a, rubric=rubric)
        summary_b = self.g(content=segment_b, rubric=rubric)

        disjoint = self.g(content=format_merge_input(summary_a, summary_b), rubric=rubric)
        joint = self.g(content=format_merge_input(segment_a, segment_b), rubric=rubric)

        return dspy.Prediction(
            summary_a=summary_a,
            summary_b=summary_b,
            merged_summary=disjoint,
            joint_segments_summary=joint,
        )


__all__ = [
    "LawStressLocalLawProgram",
    "UnifiedG",
]
