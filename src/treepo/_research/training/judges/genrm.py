"""
GenRM-based pairwise comparison judge.

Wraps the existing GenRMJudge to conform to the BaseJudge protocol.
Uses NVIDIA's Qwen3-Nemotron-235B-A22B-GenRM model for comparison.
"""

import asyncio
import logging
from typing import Any, Optional

from .base import AsyncJudge, JudgeConfig, JudgeError, JudgeResult

logger = logging.getLogger(__name__)


class GenRMJudgeWrapper:
    """
    Wrapper around GenRMJudge that implements the BaseJudge protocol.

    This provides a consistent interface while delegating to the
    existing GenRMJudge implementation for the actual comparison logic.
    """

    def __init__(self, config: Optional[JudgeConfig] = None):
        """
        Initialize the GenRM judge wrapper.

        Args:
            config: Judge configuration (uses defaults if None)
        """
        self.config = config or JudgeConfig(type="genrm")

        # Lazily import and create the underlying judge
        self._judge = None

    def _ensure_judge(self):
        """Lazily initialize the underlying GenRMJudge."""
        if self._judge is None:
            from treepo._research.training.preference.genrm import GenRMJudge

            self._judge = GenRMJudge(
                base_url=self.config.base_url,
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
            )
        return self._judge

    def compare(
        self,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
        extra_context: Optional[str] = None,
        **kwargs,
    ) -> JudgeResult:
        """
        Compare two summaries using GenRM.

        Args:
            context: Description of what information to preserve (rubric)
            original_text: Original text being summarized
            summary_a: First candidate summary
            summary_b: Second candidate summary
            law_type: OPS law type ("sufficiency", "idempotence", "merge")
            extra_context: Additional context for the comparison
            **kwargs: Additional arguments passed to underlying judge

        Returns:
            JudgeResult with preference and confidence
        """
        judge = self._ensure_judge()

        # Build full context
        full_context = context
        if extra_context:
            full_context = f"{context}\n\nAdditional context: {extra_context}"

        # Call underlying judge
        result = judge.compare(
            context=full_context,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            law_type=law_type,
            **kwargs,
        )

        # Convert GenRMResult/GenRMErrorResult to JudgeResult/JudgeError
        return self._convert_result(result)

    async def compare_async(
        self,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
        extra_context: Optional[str] = None,
        **kwargs,
    ) -> JudgeResult:
        """
        Async version of compare().

        Args:
            context: Description of what information to preserve (rubric)
            original_text: Original text being summarized
            summary_a: First candidate summary
            summary_b: Second candidate summary
            law_type: OPS law type ("sufficiency", "idempotence", "merge")
            extra_context: Additional context for the comparison
            **kwargs: Additional arguments passed to underlying judge

        Returns:
            JudgeResult with preference and confidence
        """
        judge = self._ensure_judge()

        # Build full context
        full_context = context
        if extra_context:
            full_context = f"{context}\n\nAdditional context: {extra_context}"

        # Call underlying judge async
        result = await judge.compare_async(
            context=full_context,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            law_type=law_type,
            **kwargs,
        )

        return self._convert_result(result)

    def _convert_result(self, result: Any) -> JudgeResult:
        """Convert GenRM result types to unified JudgeResult."""
        from treepo._research.training.preference.genrm import (
            GenRMResult,
            GenRMErrorResult,
            is_genrm_error,
        )

        if is_genrm_error(result):
            # Convert error to a low-confidence tie
            # Callers should check JudgeError separately in real usage
            logger.warning(f"GenRM error: {result.error_message}")
            return JudgeResult(
                preferred="tie",
                confidence=0.0,
                reasoning=f"Error: {result.error_message}",
                raw_result=result,
            )

        # Convert GenRMResult to JudgeResult
        # Map ranking score (1-6) to confidence (0-1)
        # 1 or 6 = very confident, 3 or 4 = low confidence
        ranking_confidence = {
            1: 0.95,  # A much better
            2: 0.75,  # A better
            3: 0.55,  # A slightly better
            4: 0.55,  # B slightly better
            5: 0.75,  # B better
            6: 0.95,  # B much better
        }
        confidence = ranking_confidence.get(result.ranking_score, 0.5)

        return JudgeResult(
            preferred=result.preferred,
            confidence=confidence,
            reasoning=result.reasoning,
            score_estimate_a=result.helpfulness_a,
            score_estimate_b=result.helpfulness_b,
            raw_result=result,
        )

    @property
    def model_name(self) -> Optional[str]:
        """Get the model name (may require server connection)."""
        if self._judge is not None:
            return self._judge.model_name
        return self.config.model_name


def create_genrm_judge(config: Optional[JudgeConfig] = None) -> GenRMJudgeWrapper:
    """Factory function for creating GenRM judges."""
    return GenRMJudgeWrapper(config=config)


from treepo._research.training.preference.genrm import (  # noqa: E402
    GenRMErrorResult,
    GenRMJudge,
    GenRMResult,
    is_genrm_error,
)

__all__ = [
    "GenRMErrorResult",
    "GenRMJudge",
    "GenRMJudgeWrapper",
    "GenRMResult",
    "create_genrm_judge",
    "is_genrm_error",
]
