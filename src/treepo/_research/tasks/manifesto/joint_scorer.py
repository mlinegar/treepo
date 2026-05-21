"""
Joint dimension scorer: one `dspy.Predict` shared across all 6 policy dimensions.

The key difference from `DimensionScorer` is that the underlying predictor is
held once on the Module instance — so when DSPy optimizers (BootstrapFewShot,
MIPROv2) mutate `self.score` to add demos, those demos become effective for
*every* dimension that gets scored later. This lets us pool training data
across dimensions for richer optimization.

Callers pass `dimension_spec` at forward-time; the scoring `task_context` is
looked up from the spec. The output dict and clamping behavior match
`DimensionScorer` so downstream code can reuse the same metric/eval helpers.
"""

from __future__ import annotations

import logging
from typing import Optional

import dspy

from treepo._research.core.prompting import parse_numeric_score

from .dimension_scorer import DimensionScoreSignature, _looks_like_na
from .dimensions import BENOIT_DIMENSIONS, DimensionSpec, PolicyDimension
from .pipeline_config import DEFAULT_SCORER_MAX_TOKENS
from .scoring_contexts import get_scoring_context

logger = logging.getLogger(__name__)


class JointDimensionScorer(dspy.Module):
    """One scoring predictor shared across all 6 policy dimensions.

    Parameters
    ----------
    use_cot : bool
        Whether to wrap the predictor in `dspy.ChainOfThought` rather than
        plain `dspy.Predict`.
    """

    def __init__(self, *, use_cot: bool = False, max_output_tokens: int = DEFAULT_SCORER_MAX_TOKENS):
        super().__init__()
        self.max_output_tokens = int(max_output_tokens)
        if use_cot:
            self.scorer = dspy.ChainOfThought(DimensionScoreSignature)
        else:
            self.scorer = dspy.Predict(DimensionScoreSignature)

    def forward(
        self,
        summary: str,
        dimension_spec: Optional[DimensionSpec] = None,
        dimension: Optional[PolicyDimension] = None,
        task_context: Optional[str] = None,
    ) -> dict:
        """Score a summary on one policy dimension.

        Exactly one of `dimension_spec` or `dimension` (or `task_context`)
        must effectively pin down which dimension is being scored. When only
        `dimension` is given, we look up the DimensionSpec; when only
        `task_context` is given, we score against that context directly but
        can't clamp to a scale (falls back to 1..7 as the Benoit default).
        """
        if dimension_spec is None and dimension is not None:
            dimension_spec = BENOIT_DIMENSIONS[dimension]

        if task_context is None:
            if dimension_spec is None:
                raise ValueError("Need dimension_spec, dimension, or task_context.")
            task_context = get_scoring_context(dimension_spec.dimension)

        # Keep the predictor off the name ``score``. DSPy examples and
        # predictions also use a ``score`` field, and some optimizers can leave
        # that name bound to a float on compiled modules. Legacy artifacts may
        # still contain a callable ``score`` predictor, so support it as a
        # fallback.
        predictor = getattr(self, "scorer", None)
        if not callable(predictor):
            legacy_predictor = getattr(self, "score", None)
            if callable(legacy_predictor):
                predictor = legacy_predictor
        if not callable(predictor):
            predictor = dspy.Predict(DimensionScoreSignature)
            self.scorer = predictor

        result = predictor(
            task_context=task_context,
            summary=summary,
            config={"max_tokens": self.max_output_tokens},
        )

        raw_str = str(getattr(result, "score", ""))
        if _looks_like_na(raw_str):
            return {"score": None, "reasoning": getattr(result, "reasoning", "")}

        min_v, max_v = (
            (dimension_spec.scale.min_value, dimension_spec.scale.max_value)
            if dimension_spec is not None
            else (1.0, 7.0)
        )
        raw = parse_numeric_score(raw_str, min_value=min_v, max_value=max_v, allow_llm_fallback=True)
        if raw is None:
            raw = parse_numeric_score(str(result), min_value=min_v, max_value=max_v, allow_llm_fallback=True)
        if raw is None:
            return {"score": None, "reasoning": getattr(result, "reasoning", "")}

        if dimension_spec is not None:
            raw = dimension_spec.scale.clamp(float(raw))
        return {"score": float(raw), "reasoning": getattr(result, "reasoning", "")}
