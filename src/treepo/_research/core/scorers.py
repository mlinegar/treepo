"""
Generic DSPy scorer modules.

This module provides configurable scorer modules that can be used with any
bounded scale and scoring signature. Domain-specific scorers can either
use these directly or extend them.

Usage:
    from treepo._research.core.scorers import ScaleScorer
    from treepo._research.core.signatures import MetricScore

    # Create a generic scorer
    scorer = ScaleScorer(
        signature_class=MetricScore,
        score_field="score",
    )
    result = scorer(text="...", task_context="...")

    # Or create domain-specific by passing custom signature
    from my_domain import MyScoringSignature
    scorer = ScaleScorer(
        signature_class=MyScoringSignature,
        score_field="my_score_field",
    )
"""

import logging
from typing import Any, Dict, Optional, Type

import dspy

from treepo._research.core.prompting import parse_numeric_score

logger = logging.getLogger(__name__)


class ScaleScorer(dspy.Module):
    """
    Generic DSPy scorer for any bounded scale.

    This module wraps a scoring signature and extracts the score from the
    result. It can be configured with any signature class and score field.

    Args:
        signature_class: The DSPy signature class to use for scoring
        score_field: The name of the output field containing the score
        use_cot: Whether to use ChainOfThought (True) or Predict (False)

    Example:
        scorer = ScaleScorer(MetricScore, score_field="score")
        result = scorer(text="Some text to score", task_context="Rate quality 0-10")
        print(result['score'])  # 7.5
    """

    def __init__(
        self,
        signature_class: Type[dspy.Signature],
        score_field: str = "score",
        use_cot: bool = True,
    ):
        super().__init__()
        self.score_field = score_field

        if use_cot:
            self.predict = dspy.ChainOfThought(signature_class)
        else:
            self.predict = dspy.Predict(signature_class)

    def forward(
        self,
        text: str,
        task_context: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Score the given text.

        Args:
            text: Text to score
            task_context: Context/instructions for the scoring task
            **kwargs: Additional arguments passed to the signature

        Returns:
            Dictionary with 'score' and 'raw_result' keys
        """
        result = self.predict(text=text, task_context=task_context, **kwargs)

        # Extract score, handling attribute access variations
        score = 0.0
        if hasattr(result, self.score_field):
            parsed = parse_numeric_score(
                str(getattr(result, self.score_field)),
                allow_llm_fallback=True,
            )
            if parsed is not None:
                score = parsed
            else:
                logger.warning("Could not parse %s as numeric score", self.score_field)

        return {
            'score': score,
            'raw_result': result,
        }


class PairwiseScorer(dspy.Module):
    """
    Generic pairwise comparison scorer.

    Compares two candidates and returns which is preferred.

    Args:
        signature_class: The DSPy signature class for comparison
        use_cot: Whether to use ChainOfThought
    """

    def __init__(
        self,
        signature_class: Type[dspy.Signature],
        use_cot: bool = True,
    ):
        super().__init__()

        if use_cot:
            self.predict = dspy.ChainOfThought(signature_class)
        else:
            self.predict = dspy.Predict(signature_class)

    def forward(
        self,
        candidate_a: str,
        candidate_b: str,
        context: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compare two candidates.

        Args:
            candidate_a: First candidate
            candidate_b: Second candidate
            context: Comparison context/criteria
            **kwargs: Additional arguments

        Returns:
            Dictionary with comparison results
        """
        result = self.predict(
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            context=context,
            **kwargs,
        )

        # Extract preferred choice
        preferred = getattr(result, 'preferred', 'tie')
        if isinstance(preferred, str):
            preferred = preferred.strip().upper()
            if preferred not in ('A', 'B', 'TIE'):
                preferred = 'TIE'

        return {
            'preferred': preferred,
            'raw_result': result,
        }
