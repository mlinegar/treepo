"""
GenRM Reward Function Adapters.

This module provides adapters to convert GenRM judge outputs into:
1. TRL-compatible reward functions for GRPO training
2. DSPy-compatible metrics for BootstrapFinetune

The adapters bridge the GenRM 1-6 ranking scale with the scalar reward
signals expected by reinforcement learning training methods.

GenRM Ranking Scale:
    1-2: Summary A is much/slightly better
    3-4: Roughly equal (tie)
    5-6: Summary B is slightly/much better

These are converted to:
    - Reward function: Scalar in [0, 1] based on helpfulness
    - DSPy metric: Binary pass/fail based on threshold

Usage:
    from treepo._research.training.preference.genrm_reward import (
        create_genrm_reward_func,
        create_genrm_dspy_metric,
    )

    # For GRPO training
    reward_func = create_genrm_reward_func(genrm_judge)

    # For DSPy BootstrapFinetune
    metric = create_genrm_dspy_metric(genrm_judge, threshold=3.0)
"""

import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from treepo._research.training.preference.genrm import GenRMJudge

logger = logging.getLogger(__name__)


# =============================================================================
# TRL Reward Function
# =============================================================================

def create_genrm_reward_func(
    genrm_judge: "GenRMJudge",
    rubric: Optional[str] = None,
    use_helpfulness: bool = True,
    normalize_to_01: bool = True,
) -> Callable:
    """
    Create a TRL-compatible reward function from GenRM judge.

    The reward function can be used with GRPO trainer to score
    generated completions at runtime.

    Args:
        genrm_judge: GenRM judge instance for scoring
        rubric: Optional rubric to use for all comparisons
        use_helpfulness: If True, use helpfulness score (1-5 scale).
                        If False, use comparison vs reference.
        normalize_to_01: Normalize scores to [0, 1] range

    Returns:
        Callable that takes (completions, prompts) and returns scores
    """
    def reward_func(
        completions: List[str],
        prompts: List[str],
        **kwargs,
    ) -> List[float]:
        """
        Score completions using GenRM.

        Args:
            completions: List of generated completions
            prompts: List of prompts (same length as completions)
            **kwargs: Additional context (may include rubric, original_text)

        Returns:
            List of scalar rewards for each completion
        """
        scores = []
        context = rubric or kwargs.get("rubric", "")

        for completion, prompt in zip(completions, prompts):
            try:
                score = _score_single_completion(
                    genrm_judge=genrm_judge,
                    completion=completion,
                    prompt=prompt,
                    context=context,
                    use_helpfulness=use_helpfulness,
                    normalize_to_01=normalize_to_01,
                )
                scores.append(score)
            except Exception as e:
                logger.warning(f"GenRM scoring failed: {e}")
                # Return neutral score on failure
                scores.append(0.5 if normalize_to_01 else 3.0)

        return scores

    return reward_func


def _score_single_completion(
    genrm_judge: "GenRMJudge",
    completion: str,
    prompt: str,
    context: str,
    use_helpfulness: bool,
    normalize_to_01: bool,
) -> float:
    """
    Score a single completion using GenRM.

    For single-item scoring, we compare against a "neutral" baseline
    or use the helpfulness score directly if available.
    """
    # Try to get helpfulness score directly
    if use_helpfulness:
        # Use GenRM's single-item scoring if available
        if hasattr(genrm_judge, 'score_single'):
            result = genrm_judge.score_single(
                context=context,
                summary=completion,
            )
            helpfulness = getattr(result, 'helpfulness', 3.0)
            if normalize_to_01:
                # Convert 1-5 helpfulness to 0-1
                return (helpfulness - 1.0) / 4.0
            return helpfulness

    # Fall back to comparison-based scoring
    # Compare against an empty/minimal baseline
    baseline = "No information provided."

    result = genrm_judge.compare(
        context=context,
        original_text=prompt,
        summary_a=completion,
        summary_b=baseline,
    )

    # Convert ranking to reward
    # 1-2: A much better (completion wins) → high reward
    # 3-4: Tie → medium reward
    # 5-6: B better (baseline wins) → low reward
    ranking = getattr(result, 'ranking_score', 4)

    if normalize_to_01:
        # Map 1-6 to ~1.0-0.0 (higher ranking = worse for A)
        return (6 - ranking) / 5.0
    return float(6 - ranking)


def create_genrm_comparison_reward_func(
    genrm_judge: "GenRMJudge",
    reference_summary: str,
    rubric: str,
    original_text: Optional[str] = None,
) -> Callable:
    """
    Create a reward function that compares completions against a reference.

    This is useful when you have a known good summary to compare against.

    Args:
        genrm_judge: GenRM judge instance
        reference_summary: Reference summary to compare against
        rubric: Rubric describing what information to preserve
        original_text: Original text being summarized

    Returns:
        Callable reward function
    """
    def reward_func(
        completions: List[str],
        prompts: List[str],
        **kwargs,
    ) -> List[float]:
        """Score completions by comparing against reference."""
        scores = []
        context = rubric

        for completion, prompt in zip(completions, prompts):
            try:
                result = genrm_judge.compare(
                    context=context,
                    original_text=original_text or prompt,
                    summary_a=completion,
                    summary_b=reference_summary,
                )

                # Convert ranking (1-6) to reward
                # 1-2: completion much better → 1.0
                # 3-4: roughly equal → 0.5
                # 5-6: reference better → 0.0
                ranking = getattr(result, 'ranking_score', 4)
                score = (6 - ranking) / 5.0
                scores.append(score)

            except Exception as e:
                logger.warning(f"GenRM comparison failed: {e}")
                scores.append(0.5)

        return scores

    return reward_func


# =============================================================================
# DSPy Metric
# =============================================================================

def create_genrm_dspy_metric(
    genrm_judge: "GenRMJudge",
    threshold: float = 3.0,
    use_comparison: bool = True,
) -> Callable:
    """
    Create a DSPy-compatible metric from GenRM judge.

    The metric is used by DSPy optimizers (BootstrapFinetune, MIPROv2)
    to filter traces and guide optimization.

    Args:
        genrm_judge: GenRM judge instance
        threshold: Ranking threshold for passing (1-3 = pass, 4-6 = fail by default)
        use_comparison: If True, compare pred.summary vs gold.summary.
                       If False, score pred.summary directly.

    Returns:
        Callable metric function compatible with DSPy
    """
    def metric(
        pred: Any,
        gold: Any,
        trace: Optional[Any] = None,
    ) -> float:
        """
        DSPy metric function.

        Args:
            pred: DSPy prediction with .summary attribute
            gold: DSPy example with .summary, .rubric, .content attributes
            trace: Optional trace information

        Returns:
            1.0 if passes threshold, 0.0 otherwise
        """
        try:
            pred_summary = getattr(pred, 'summary', str(pred))
            gold_summary = getattr(gold, 'summary', '')
            rubric = getattr(gold, 'rubric', '')
            original_text = getattr(gold, 'content', '')

            if use_comparison and gold_summary:
                # Compare prediction against gold
                result = genrm_judge.compare(
                    context=rubric,
                    original_text=original_text,
                    summary_a=pred_summary,
                    summary_b=gold_summary,
                )

                ranking = getattr(result, 'ranking_score', 4)

                # Pass if prediction is at least as good as gold (ranking <= threshold)
                return 1.0 if ranking <= threshold else 0.0

            else:
                # Score prediction directly using helpfulness
                if hasattr(genrm_judge, 'score_single'):
                    result = genrm_judge.score_single(
                        context=rubric,
                        summary=pred_summary,
                    )
                    helpfulness = getattr(result, 'helpfulness', 3.0)
                    # Pass if helpfulness >= 4 (good quality)
                    return 1.0 if helpfulness >= 4.0 else 0.0
                else:
                    # Fall back to comparison with baseline
                    result = genrm_judge.compare(
                        context=rubric,
                        original_text=original_text,
                        summary_a=pred_summary,
                        summary_b="No information.",
                    )
                    ranking = getattr(result, 'ranking_score', 4)
                    return 1.0 if ranking <= threshold else 0.0

        except Exception as e:
            logger.warning(f"GenRM metric evaluation failed: {e}")
            return 0.0

    return metric


def create_genrm_quality_metric(
    genrm_judge: "GenRMJudge",
    min_helpfulness: float = 4.0,
) -> Callable:
    """
    Create a DSPy metric based on GenRM helpfulness score.

    This metric doesn't require a gold summary - it just checks if
    the prediction meets a minimum quality threshold.

    Args:
        genrm_judge: GenRM judge instance
        min_helpfulness: Minimum helpfulness score to pass (1-5 scale)

    Returns:
        Callable metric function
    """
    def metric(
        pred: Any,
        gold: Any,
        trace: Optional[Any] = None,
    ) -> float:
        """
        Score prediction quality directly.

        Returns 1.0 if helpfulness >= min_helpfulness, else 0.0
        """
        try:
            pred_summary = getattr(pred, 'summary', str(pred))
            rubric = getattr(gold, 'rubric', '')

            if hasattr(genrm_judge, 'score_single'):
                result = genrm_judge.score_single(
                    context=rubric,
                    summary=pred_summary,
                )
                helpfulness = getattr(result, 'helpfulness', 3.0)
                return 1.0 if helpfulness >= min_helpfulness else 0.0
            else:
                logger.warning("GenRM judge doesn't support score_single, using default")
                return 0.5

        except Exception as e:
            logger.warning(f"Quality metric evaluation failed: {e}")
            return 0.0

    return metric


# =============================================================================
# Utility Functions
# =============================================================================

def ranking_to_reward(ranking: int, normalize: bool = True) -> float:
    """
    Convert GenRM ranking score (1-6) to reward.

    Args:
        ranking: GenRM ranking score (1=A much better, 6=B much better)
        normalize: If True, normalize to [0, 1] range

    Returns:
        Reward value
    """
    if normalize:
        return (6 - ranking) / 5.0
    return float(6 - ranking)


def helpfulness_to_reward(helpfulness: float, normalize: bool = True) -> float:
    """
    Convert GenRM helpfulness score (1-5) to reward.

    Args:
        helpfulness: Helpfulness score from GenRM
        normalize: If True, normalize to [0, 1] range

    Returns:
        Reward value
    """
    if normalize:
        return (helpfulness - 1.0) / 4.0
    return helpfulness
