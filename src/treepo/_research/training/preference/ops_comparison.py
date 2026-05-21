"""
OPS law comparison module for preference learning.

This module predicts pairwise preferences (A/B/tie) and confidence
for a given OPS law and rubric.
"""

import dspy


class OPSLawComparison(dspy.Signature):
    """Compare two candidates for OPS law compliance."""

    law_type: str = dspy.InputField(desc="'sufficiency', 'merge', or 'idempotence'")
    rubric: str = dspy.InputField(desc="Information preservation criteria")
    original_text: str = dspy.InputField(desc="Original/reference text")
    summary_a: str = dspy.InputField(desc="First candidate")
    summary_b: str = dspy.InputField(desc="Second candidate")
    reference_score: float = dspy.InputField(desc="Oracle score for original")

    preferred: str = dspy.OutputField(desc="'A', 'B', or 'tie'")
    confidence: float = dspy.OutputField(desc="0.0 to 1.0")
    reasoning: str = dspy.OutputField(desc="Why this candidate better satisfies the law")


class OPSComparisonModule(dspy.Module):
    """DSPy module for OPS law comparison."""

    def __init__(self, use_cot: bool = True):
        super().__init__()
        if use_cot:
            self.compare = dspy.ChainOfThought(OPSLawComparison)
        else:
            self.compare = dspy.Predict(OPSLawComparison)

    def forward(
        self,
        law_type: str,
        rubric: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        reference_score: float,
    ):
        result = self.compare(
            law_type=law_type,
            rubric=rubric,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            reference_score=reference_score,
        )

        preferred = str(result.preferred).upper().strip()
        if preferred not in ["A", "B", "TIE"]:
            if "A" in preferred and "B" not in preferred:
                preferred = "A"
            elif "B" in preferred and "A" not in preferred:
                preferred = "B"
            else:
                preferred = "TIE"

        try:
            confidence = float(result.confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5

        result.preferred = "tie" if preferred == "TIE" else preferred
        result.confidence = confidence
        result.reasoning = str(result.reasoning)
        return result
