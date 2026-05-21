"""
DSPy-based Summarization Modules for RILE Preservation.

This module provides optimizable summarization modules that can be trained
using DSPy's GEPA optimizer to maximize RILE information preservation.

The key difference from the hardcoded prompts in batched_pipeline.py:
- These modules can be optimized with DSPy
- Prompts/instructions can evolve through training
- Demonstrations can be learned from training data

Usage:
    from treepo._research.tasks.manifesto import UnifiedManifestoG

    # Create modules
    g = UnifiedManifestoG()

    # Use directly
    summary = g(content="...", rubric="...")

    # Or optimize with DSPy GEPA
    optimized = gepa.compile(g, trainset=trainset)
"""

import dspy
from typing import Optional

# Import generic components from core
from treepo._research.core.summarization import (
    GenericSummarizer,
    GenericMerger,
    SummarizationResult,
)
from .pipeline import UnifiedManifestoG


# =============================================================================
# RILE-Specific Signature (extends RecursiveSummary with RILE focus)
# =============================================================================

class RILELeafSummary(dspy.Signature):
    """
    Summarize political text while preserving left-right (RILE) positioning information.

    This signature is specifically designed for leaf-level summarization of
    political manifesto chunks. The summary must preserve all information
    relevant to determining the document's position on the left-right spectrum.
    """
    rubric: str = dspy.InputField(
        desc="Information preservation criteria specifying what political indicators to preserve"
    )
    content: str = dspy.InputField(
        desc="Raw political text chunk to summarize"
    )
    summary: str = dspy.OutputField(
        desc="Concise summary that preserves all RILE-relevant political positioning information"
    )


class RILEMergeSummary(dspy.Signature):
    """
    Merge two summaries while preserving combined RILE positioning information.

    This signature is for internal node summarization - combining child summaries
    into a parent summary while ensuring no political positioning information is lost.
    """
    rubric: str = dspy.InputField(
        desc="Information preservation criteria specifying what political indicators to preserve"
    )
    left_summary: str = dspy.InputField(
        desc="First summary to merge"
    )
    right_summary: str = dspy.InputField(
        desc="Second summary to merge"
    )
    merged_summary: str = dspy.OutputField(
        desc="Combined summary preserving all RILE-relevant information from both inputs"
    )


# =============================================================================
# Optimizable Summarizer Modules
# =============================================================================

class LeafSummarizer(dspy.Module):
    """
    Optimizable leaf summarization module for RILE preservation.

    Honours the repo-wide two-tier output budget (see
    ``pipeline.ManifestoSummarizer`` docstring): prompt carries a soft
    target; API ``max_tokens`` caps at ``CONCAT_RATIO × input`` so
    concatenation remains physically possible.

    Example:
        summarizer = LeafSummarizer()
        summary = summarizer(content="The party supports...", rubric=RILE_RUBRIC)
    """

    def __init__(
        self,
        use_cot: bool = False,
        *,
        output_token_ratio: float | None = None,
        target_token_ratio: float | None = None,
    ):
        super().__init__()
        from .pipeline_config import CONCAT_RATIO, DEFAULT_TARGET_RATIO
        self.output_token_ratio = CONCAT_RATIO if output_token_ratio is None else float(output_token_ratio)
        self.target_token_ratio = DEFAULT_TARGET_RATIO if target_token_ratio is None else float(target_token_ratio)
        if use_cot:
            self.summarize = dspy.ChainOfThought(RILELeafSummary)
        else:
            self.summarize = dspy.Predict(RILELeafSummary)

    def forward(self, content: str, rubric: str) -> str:
        from .pipeline import compute_output_budget, _budget_instruction, _infer_context_window
        ctx = _infer_context_window()
        target, hmax = compute_output_budget(
            content,
            ratio=self.output_token_ratio,
            target_ratio=self.target_token_ratio,
            context_window=ctx,
        )
        effective_rubric = rubric + _budget_instruction(target, hmax)
        result = self.summarize(content=content, rubric=effective_rubric, config={"max_tokens": int(hmax)})
        return result.summary


class MergeSummarizer(dspy.Module):
    """
    Optimizable merge summarization module for RILE preservation.

    Honours the repo-wide two-tier output budget. Inputs are two summaries
    of total tokens ``T``; hard-max output ≤ ``CONCAT_RATIO × T`` so the
    merger can *concatenate* when merging would lose information, with
    the prompt nudging compression in the typical case.
    """

    def __init__(
        self,
        use_cot: bool = False,
        *,
        output_token_ratio: float | None = None,
        target_token_ratio: float | None = None,
    ):
        super().__init__()
        from .pipeline_config import CONCAT_RATIO, DEFAULT_TARGET_RATIO
        self.output_token_ratio = CONCAT_RATIO if output_token_ratio is None else float(output_token_ratio)
        self.target_token_ratio = DEFAULT_TARGET_RATIO if target_token_ratio is None else float(target_token_ratio)
        if use_cot:
            self.merge = dspy.ChainOfThought(RILEMergeSummary)
        else:
            self.merge = dspy.Predict(RILEMergeSummary)

    def forward(self, left_summary: str, right_summary: str, rubric: str) -> str:
        """Merge two summaries while preserving RILE information."""
        from .pipeline import compute_output_budget, _budget_instruction, _infer_context_window
        ctx = _infer_context_window()
        target, hmax = compute_output_budget(
            left_summary + right_summary,
            ratio=self.output_token_ratio,
            target_ratio=self.target_token_ratio,
            context_window=ctx,
        )
        effective_rubric = rubric + _budget_instruction(target, hmax)
        result = self.merge(
            left_summary=left_summary,
            right_summary=right_summary,
            rubric=effective_rubric,
            config={"max_tokens": int(hmax)},
        )
        return result.merged_summary



# =============================================================================
# Factory Functions
# =============================================================================

def create_summarizers(
    use_rile_specific: bool = True,
    use_cot: bool = False,
) -> tuple:
    """
    Create summarizer modules.

    Manifesto active paths now use a unified ``g(content, rubric)`` module for
    leaves and merges. This factory still returns a 2-tuple for compatibility;
    the second element is ``None`` for the RILE-specific path and callers should
    pass ``unified_mode=True`` to strategy wrappers.

    Args:
        use_rile_specific: Use RILE-specific signatures (recommended for manifesto work)
        use_cot: Use Chain-of-Thought reasoning

    Returns:
        Tuple of (g_summarizer, legacy_merge_summarizer_or_none)
    """
    if use_rile_specific:
        return UnifiedManifestoG(use_cot=use_cot), None
    else:
        return GenericSummarizer(use_cot=use_cot), GenericMerger(use_cot=use_cot)
