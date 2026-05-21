"""Shared protocol definitions and utility functions for the codebase."""

from typing import Protocol


class Summarizer(Protocol):
    """Protocol for synchronous summarization functions.

    Used by:
    - builder.py: build() helper function with SyncSummarizerAdapter
    - auditor.py: idempotence and substitution checks
    """

    def __call__(self, text: str, rubric: str) -> str:
        """
        Summarize text according to rubric.

        Args:
            text: Input text to summarize
            rubric: Information preservation criteria

        Returns:
            Summary string
        """
        ...


def format_merge_input(left: str, right: str) -> str:
    """
    Format merge input with labeled parts.

    THEORY CORRESPONDENCE:
    This function implements the abstract concatenation operator from the paper.
    In Lean: s_L * s_R (monoid multiplication on Strings)
    In paper: s_L (concat) s_R

    The unified summarizer g handles both:
    - Leaf: g(raw_text)
    - Merge: g(format_merge_input(s_L, s_R))

    The PART labels help the LLM understand it's consolidating two summaries,
    while still using the same g function as leaf summarization.

    Used in:
    - Tree building (merging nodes)
    - Auditing (merge consistency checks)
    - Preference collection (merge candidate generation)

    Args:
        left: First summary (left child)
        right: Second summary (right child)

    Returns:
        Formatted merge input with PART labels
    """
    return f"PART 1:\n{left}\n\nPART 2:\n{right}"
