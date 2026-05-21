"""
Generic DSPy summarization modules.

This module provides configurable summarization and merge modules that can
be used for any domain. Domain-specific summarizers can extend these or
use custom signatures.

Usage:
    from treepo._research.core.summarization import GenericSummarizer, GenericMerger

    # Create summarizers
    summarizer = GenericSummarizer(use_cot=True)
    merger = GenericMerger(use_cot=True)

    # Use directly
    summary = summarizer(content="...", rubric="...")
    merged = merger(left_summary="...", right_summary="...", rubric="...")

    # Or with custom signature
    summarizer = GenericSummarizer(signature_class=MySignature, use_cot=True)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Type

import dspy

from .signatures import RecursiveSummary

logger = logging.getLogger(__name__)


# =============================================================================
# Generic Summarizer Modules
# =============================================================================

class GenericSummarizer(dspy.Module):
    """
    Generic summarizer using a configurable signature.

    This provides a reusable summarization module that can work with any
    signature class. Defaults to RecursiveSummary from core.

    Args:
        signature_class: DSPy signature class with 'content', 'rubric' inputs
                        and 'summary' output. Defaults to RecursiveSummary.
        use_cot: Whether to use ChainOfThought (True) or Predict (False)

    Example:
        summarizer = GenericSummarizer()
        result = summarizer(content="Long text...", rubric="Preserve key info")
    """

    def __init__(
        self,
        signature_class: Optional[Type[dspy.Signature]] = None,
        use_cot: bool = True,
    ):
        super().__init__()
        sig_class = signature_class or RecursiveSummary

        if use_cot:
            self.summarize = dspy.ChainOfThought(sig_class)
        else:
            self.summarize = dspy.Predict(sig_class)

    def forward(self, content: str, rubric: str) -> str:
        """
        Generate summary of the content.

        Args:
            content: Text to summarize
            rubric: Information preservation criteria

        Returns:
            Summary string
        """
        result = self.summarize(content=content, rubric=rubric)
        return result.summary


class GenericMerger(dspy.Module):
    """
    Generic merge summarizer using a configurable signature.

    Combines two summaries by concatenating them and re-summarizing.
    Can use any signature that accepts content and rubric.

    Args:
        signature_class: DSPy signature class. Defaults to RecursiveSummary.
        use_cot: Whether to use ChainOfThought

    Example:
        merger = GenericMerger()
        merged = merger(
            left_summary="Summary A",
            right_summary="Summary B",
            rubric="Combine preserving key info"
        )
    """

    def __init__(
        self,
        signature_class: Optional[Type[dspy.Signature]] = None,
        use_cot: bool = True,
    ):
        super().__init__()
        sig_class = signature_class or RecursiveSummary

        if use_cot:
            self.merge = dspy.ChainOfThought(sig_class)
        else:
            self.merge = dspy.Predict(sig_class)

    def forward(self, left_summary: str, right_summary: str, rubric: str) -> str:
        """
        Merge two summaries.

        Args:
            left_summary: First summary to merge
            right_summary: Second summary to merge
            rubric: Information preservation criteria

        Returns:
            Merged summary string
        """
        combined = f"PART 1:\n{left_summary}\n\nPART 2:\n{right_summary}"
        result = self.merge(content=combined, rubric=rubric)
        return result.summary


# =============================================================================
# Result Dataclass
# =============================================================================

@dataclass
class SummarizationResult:
    """Result from a summarization operation."""
    summary: str
    input_length: int
    output_length: int
    compression_ratio: float

    @classmethod
    def from_summary(cls, original: str, summary: str) -> "SummarizationResult":
        """Create result from original text and summary."""
        return cls(
            summary=summary,
            input_length=len(original),
            output_length=len(summary),
            compression_ratio=len(summary) / max(len(original), 1),
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_summarizers(
    signature_class: Optional[Type[dspy.Signature]] = None,
    merge_signature_class: Optional[Type[dspy.Signature]] = None,
    use_cot: bool = True,
) -> tuple:
    """
    Create leaf and merge summarizer modules.

    Args:
        signature_class: Signature for leaf summarization (defaults to RecursiveSummary)
        merge_signature_class: Signature for merge (defaults to signature_class)
        use_cot: Use Chain-of-Thought reasoning

    Returns:
        Tuple of (leaf_summarizer, merge_summarizer)
    """
    merge_sig = merge_signature_class or signature_class
    return (
        GenericSummarizer(signature_class=signature_class, use_cot=use_cot),
        GenericMerger(signature_class=merge_sig, use_cot=use_cot),
    )
