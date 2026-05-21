"""
DSPy signatures for OPS (Oracle-Preserving Summarization).

This module defines the core DSPy signatures used in the OPS system:
- RecursiveSummary: Compresses content while preserving rubric-specified information
- OracleJudge: Compares inputs for task-equivalence per rubric
"""

import dspy
from typing import Optional


class RecursiveSummary(dspy.Signature):
    """
    Compress content into a summary that preserves information specified by the rubric.

    This signature is used for both leaf summarization (raw text -> summary)
    and internal node summarization (child summaries -> parent summary).
    """
    rubric: str = dspy.InputField(
        desc="The specific information and criteria that must be preserved in the summary"
    )
    content: str = dspy.InputField(
        desc="Raw text or concatenated child summaries to compress"
    )
    summary: str = dspy.OutputField(
        desc="The information-preserving summary that retains rubric-specified content"
    )


class OracleJudge(dspy.Signature):
    """
    Compare two inputs to determine if they yield equivalent answers per the rubric.

    This is the approximate oracle (f̂) that estimates whether information has been
    preserved through summarization. It produces both a binary congruence judgment
    and a continuous discrepancy score.
    """
    rubric: str = dspy.InputField(
        desc="The criteria defining what information must be preserved"
    )
    input_a: str = dspy.InputField(
        desc="First input (typically the more detailed/original content)"
    )
    input_b: str = dspy.InputField(
        desc="Second input (typically the summarized content)"
    )
    is_congruent: bool = dspy.OutputField(
        desc="Whether the two inputs are task-equivalent according to the rubric"
    )
    discrepancy_score: float = dspy.OutputField(
        desc="Score from 0.0 (perfect match) to 1.0 (complete information loss)"
    )
    reasoning: str = dspy.OutputField(
        desc="Explanation of any detected information drift or loss"
    )


class SufficiencyCheck(dspy.Signature):
    """
    Check if a summary sufficiently preserves the original content per rubric.

    Used for leaf-level auditing: comparing raw text against its summary.
    """
    rubric: str = dspy.InputField(
        desc="Information preservation criteria"
    )
    original: str = dspy.InputField(
        desc="Original raw text content"
    )
    summary: str = dspy.InputField(
        desc="Summary of the original content"
    )
    is_sufficient: bool = dspy.OutputField(
        desc="Whether the summary adequately preserves rubric-specified information"
    )
    missing_info: str = dspy.OutputField(
        desc="List of rubric-required information that was lost in summarization"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence in this assessment from 0.0 to 1.0"
    )


class MergeConsistencyCheck(dspy.Signature):
    """
    Check if merging child summaries preserves information.

    Compares: f*(g(left) + g(right)) vs f*(g(g(left) + g(right)))
    Used for internal node auditing.
    """
    rubric: str = dspy.InputField(
        desc="Information preservation criteria"
    )
    child_summaries: str = dspy.InputField(
        desc="Concatenated summaries from child nodes"
    )
    parent_summary: str = dspy.InputField(
        desc="Summary produced by merging the child summaries"
    )
    is_consistent: bool = dspy.OutputField(
        desc="Whether the parent summary preserves information from children"
    )
    lost_content: str = dspy.OutputField(
        desc="Content from children that was lost in the merge"
    )
    discrepancy_score: float = dspy.OutputField(
        desc="Degree of information loss from 0.0 to 1.0"
    )


# Module implementations using signatures

class Summarizer(dspy.Module):
    """DSPy module for recursive summarization."""

    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought(RecursiveSummary)

    def forward(self, content: str, rubric: str) -> str:
        """Generate a summary preserving rubric-specified information."""
        result = self.summarize(rubric=rubric, content=content)
        return result.summary


class Judge(dspy.Module):
    """DSPy module for oracle judgment."""

    def __init__(self):
        super().__init__()
        self.judge = dspy.ChainOfThought(OracleJudge)

    def forward(self, input_a: str, input_b: str, rubric: str) -> dict:
        """Compare two inputs for task-equivalence."""
        result = self.judge(rubric=rubric, input_a=input_a, input_b=input_b)
        return {
            'is_congruent': result.is_congruent,
            'discrepancy_score': result.discrepancy_score,
            'reasoning': result.reasoning
        }


class SufficiencyChecker(dspy.Module):
    """DSPy module for checking leaf sufficiency."""

    def __init__(self):
        super().__init__()
        self.check = dspy.ChainOfThought(SufficiencyCheck)

    def forward(self, original: str, summary: str, rubric: str) -> dict:
        """Check if summary preserves original content."""
        result = self.check(rubric=rubric, original=original, summary=summary)
        return {
            'is_sufficient': result.is_sufficient,
            'missing_info': result.missing_info,
            'confidence': result.confidence
        }


class MergeChecker(dspy.Module):
    """DSPy module for checking merge consistency."""

    def __init__(self):
        super().__init__()
        self.check = dspy.ChainOfThought(MergeConsistencyCheck)

    def forward(self, child_summaries: str, parent_summary: str, rubric: str) -> dict:
        """Check if merge preserves child information."""
        result = self.check(
            rubric=rubric,
            child_summaries=child_summaries,
            parent_summary=parent_summary
        )
        return {
            'is_consistent': result.is_consistent,
            'lost_content': result.lost_content,
            'discrepancy_score': result.discrepancy_score
        }


# Oracle Function Approximation signatures and modules

class OracleFuncApproximation(dspy.Signature):
    """
    Learned approximation of the oracle function for reviewing flagged nodes.

    This signature is trained on positive (true violations) and negative (false positives)
    examples from human review to predict whether a flagged audit result is a genuine
    information preservation violation or a false alarm.
    """
    rubric: str = dspy.InputField(
        desc="The information preservation criteria used for summarization"
    )
    original_content: str = dspy.InputField(
        desc="The original content (raw text for leaves, child summaries for internal nodes)"
    )
    summary: str = dspy.InputField(
        desc="The summary that was flagged during audit"
    )
    check_type: str = dspy.InputField(
        desc="Type of audit check: 'sufficiency', 'merge_consistency', 'idempotence', or 'substitution'"
    )
    approx_discrepancy: float = dspy.InputField(
        desc="The discrepancy score from the approximate oracle that flagged this item"
    )
    is_true_violation: bool = dspy.OutputField(
        desc="Whether this represents a genuine information preservation violation"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence in this judgment from 0.0 to 1.0"
    )
    corrected_summary: str = dspy.OutputField(
        desc="If is_true_violation, provide an improved summary that preserves the rubric information; otherwise empty string"
    )
    reasoning: str = dspy.OutputField(
        desc="Detailed explanation of why this is or isn't a true violation"
    )


class OracleFuncReviewer(dspy.Module):
    """
    DSPy module for reviewing flagged nodes using learned oracle function approximation.

    Trained on historical human review decisions to distinguish true violations
    from false positives and optionally provide corrected summaries.
    """

    def __init__(self):
        super().__init__()
        self.review = dspy.ChainOfThought(OracleFuncApproximation)

    def forward(
        self,
        original_content: str,
        summary: str,
        rubric: str,
        check_type: str = "sufficiency",
        approx_discrepancy: float = 0.5
    ) -> dict:
        """
        Review a flagged item to determine if it's a true violation.

        Args:
            original_content: Original content that was summarized
            summary: The flagged summary
            rubric: Information preservation criteria
            check_type: Type of audit check that flagged this item
            approx_discrepancy: Discrepancy score from initial audit

        Returns:
            Dictionary with review results
        """
        result = self.review(
            rubric=rubric,
            original_content=original_content,
            summary=summary,
            check_type=check_type,
            approx_discrepancy=approx_discrepancy
        )
        return {
            'is_true_violation': result.is_true_violation,
            'confidence': result.confidence,
            'corrected_summary': result.corrected_summary,
            'reasoning': result.reasoning
        }


# =============================================================================
# Generic Metric Scoring Signatures
# =============================================================================

class MetricScore(dspy.Signature):
    """
    Score text on a bounded numeric scale based on specified criteria.

    This is the generic foundation for domain-specific scorers. Domain modules
    (e.g., sentiment analysis, quality scoring) should extend this pattern with
    their own field names while maintaining the same structure.

    The task_context should describe:
    - What dimension/criteria to evaluate
    - The scale bounds (e.g., -100 to +100, 0 to 1)
    - What indicators suggest higher vs lower scores
    """

    task_context: str = dspy.InputField(
        desc="Explanation of the scoring task and evaluation criteria"
    )
    text: str = dspy.InputField(
        desc="Text to analyze and score"
    )
    score: float = dspy.OutputField(
        desc="Score on the specified scale. Output a single number."
    )
    high_indicators: str = dspy.OutputField(
        desc="Key indicators supporting a higher score"
    )
    low_indicators: str = dspy.OutputField(
        desc="Key indicators supporting a lower score"
    )
    reasoning: str = dspy.OutputField(
        desc="Explanation of how the score was determined"
    )


class MetricScorer(dspy.Module):
    """DSPy module for generic metric scoring."""

    def __init__(self):
        super().__init__()
        self.score = dspy.ChainOfThought(MetricScore)

    def forward(self, text: str, task_context: str) -> dict:
        """
        Score text according to the task context.

        Args:
            text: Text to score
            task_context: Scoring criteria and scale description

        Returns:
            Dictionary with score, indicators, and reasoning
        """
        result = self.score(task_context=task_context, text=text)
        return {
            'score': result.score,
            'high_indicators': result.high_indicators,
            'low_indicators': result.low_indicators,
            'reasoning': result.reasoning
        }


class PairwiseComparison(dspy.Signature):
    """
    Compare two summaries for information preservation quality.

    This signature is used to generate preference pairs for training.
    It compares which summary better preserves the information specified
    in the rubric, given the original text and its ground truth score.
    """

    rubric: str = dspy.InputField(
        desc="Criteria for what information must be preserved"
    )
    original_text: str = dspy.InputField(
        desc="Original source text"
    )
    summary_a: str = dspy.InputField(
        desc="First candidate summary"
    )
    summary_b: str = dspy.InputField(
        desc="Second candidate summary"
    )
    reference_score: float = dspy.InputField(
        desc="Ground truth score for the original text"
    )
    preferred: str = dspy.OutputField(
        desc="Which summary better preserves information: 'A', 'B', or 'tie'"
    )
    reasoning: str = dspy.OutputField(
        desc="Why this summary better preserves the target information"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence in judgment (0.0 to 1.0)"
    )


class PairwiseComparer(dspy.Module):
    """DSPy module for pairwise summary comparison."""

    def __init__(self):
        super().__init__()
        self.compare = dspy.ChainOfThought(PairwiseComparison)

    def forward(
        self,
        original_text: str,
        summary_a: str,
        summary_b: str,
        rubric: str,
        reference_score: float = 0.0
    ) -> dict:
        """
        Compare two summaries for information preservation.

        Args:
            original_text: Original source text
            summary_a: First candidate summary
            summary_b: Second candidate summary
            rubric: Preservation criteria
            reference_score: Ground truth score for original

        Returns:
            Dictionary with preference, reasoning, and confidence
        """
        result = self.compare(
            rubric=rubric,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            reference_score=reference_score
        )
        return {
            'preferred': result.preferred,
            'reasoning': result.reasoning,
            'confidence': result.confidence
        }
