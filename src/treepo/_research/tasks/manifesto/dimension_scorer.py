"""
Dimension-aware scorer for Benoit 1-7 policy scales.

Parallels `ManifestoScorer` in pipeline.py, but parameterized by a
`DimensionSpec` rather than hardcoded to RILE's -100/+100 range. Output
is the raw value on the dimension's scale (float, typically clamped to
[min_value, max_value]), plus the reasoning string.

Intentionally does NOT normalize to [0, 1] — Pearson correlation with
expert benchmarks is affine-invariant, so downstream code can work with
raw scale values and we avoid the ambiguity of two different normalizers
fighting each other.
"""

from __future__ import annotations

import logging
from typing import Optional

import dspy

from treepo._research.core.prompting import parse_numeric_score

from .dimensions import DimensionSpec
from .scoring_contexts import get_scoring_context

logger = logging.getLogger(__name__)


class DimensionScoreSignature(dspy.Signature):
    """Score a manifesto summary on a 1-7 policy dimension."""

    task_context: str = dspy.InputField(desc="Dimension, scale, and expert framing")
    summary: str = dspy.InputField(desc="Summarized manifesto to score")

    reasoning: str = dspy.OutputField(desc="Evidence cited from the summary")
    # Kept as str so Benoit-style 'NA' is a valid literal output. We parse to
    # float in DimensionScorer.forward; strict `float` typing rejects 'NA'.
    score: str = dspy.OutputField(desc="Integer 1-7 as string, or literal 'NA' if insufficient information")


class DimensionScorer(dspy.Module):
    """
    Score a summary on a single policy dimension. Returns the raw scale
    value plus reasoning; callers handle NA semantics via `score is None`.
    """

    def __init__(
        self,
        dimension: DimensionSpec,
        *,
        use_cot: bool = False,
        max_output_tokens: Optional[int] = None,
    ):
        super().__init__()
        from .pipeline_config import DEFAULT_SCORER_MAX_TOKENS
        self.dimension = dimension
        self._scoring_context = get_scoring_context(dimension.dimension)
        self.max_output_tokens = int(max_output_tokens) if max_output_tokens is not None else DEFAULT_SCORER_MAX_TOKENS
        if use_cot:
            self.predictor = dspy.ChainOfThought(DimensionScoreSignature)
        else:
            self.predictor = dspy.Predict(DimensionScoreSignature)

    def load_state(self, state) -> None:
        """Load both current and legacy scorer states.

        Older artifacts stored the DSPy predictor under ``score``. MIPRO also
        uses ``score`` for compiled-program bookkeeping, which can overwrite
        the callable predictor on full-program saves. Current artifacts store
        the predictor under ``predictor`` to keep the output field name and
        module attribute separate.
        """
        compat_state = dict(state)
        if "predictor" not in compat_state and "score" in compat_state:
            compat_state["predictor"] = compat_state["score"]
        super().load_state(compat_state)

    def forward(self, summary: str, task_context: Optional[str] = None) -> dict:
        ctx = task_context if task_context is not None else self._scoring_context
        result = self.predictor(
            task_context=ctx,
            summary=summary,
            config={"max_tokens": self.max_output_tokens},
        )

        raw_str = str(getattr(result, "score", ""))
        if _looks_like_na(raw_str):
            return {"score": None, "reasoning": getattr(result, "reasoning", "")}

        raw = parse_numeric_score(
            raw_str,
            min_value=self.dimension.scale.min_value,
            max_value=self.dimension.scale.max_value,
            allow_llm_fallback=True,
        )
        if raw is None:
            raw = parse_numeric_score(
                str(result),
                min_value=self.dimension.scale.min_value,
                max_value=self.dimension.scale.max_value,
                allow_llm_fallback=True,
            )
        if raw is None:
            logger.warning(
                "Could not parse score for dimension=%s; returning NA", self.dimension.dimension
            )
            return {"score": None, "reasoning": getattr(result, "reasoning", "")}

        return {
            "score": self.dimension.scale.clamp(float(raw)),
            "reasoning": getattr(result, "reasoning", ""),
        }


def _looks_like_na(text: str) -> bool:
    stripped = text.strip().lower()
    return stripped in {"na", "n/a", "none", "null", "unknown", ""}
