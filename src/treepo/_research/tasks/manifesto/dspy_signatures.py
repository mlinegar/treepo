"""
DSPy signatures for Manifesto Project RILE scoring.

This module provides domain-specific signatures that extend the generic
MetricScore pattern from treepo._research.core.signatures for political text scoring.

Signatures:
- RILEScore: Score political text on the left-right RILE scale
- SimpleScore: Simplified scorer for model agreement checks
- PairwiseSummaryComparison: Compare summaries for preference generation
- RILEComparison: Audit whether summarization preserves political position

See src.core.signatures.MetricScore for the generic scoring pattern.
"""

import json
import logging
import os
import re
import dspy
from typing import Any, Dict, Optional

from treepo._research.core.output_parser import NormalizedOutputAccessor
from treepo._research.core.prompting import parse_numeric_score
from treepo._research.core.engram_prompting import format_prompt_metadata_block
from .constants import RILE_MIN, RILE_MAX

logger = logging.getLogger(__name__)

_SCORER_DIAG_BUDGET_DEFAULT = 24
_SCORER_DIAG_BUDGET = _SCORER_DIAG_BUDGET_DEFAULT


def _scorer_diag_enabled() -> bool:
    raw = str(os.getenv("RILE_SCORER_DIAG", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


class RILEScore(dspy.Signature):
    """
    Score text on the RILE (Right-Left) political scale.

    Domain-specific extension of MetricScore for political manifesto scoring.
    Scale: -100 (far left) to +100 (far right).
    """
    task_context: str = dspy.InputField(
        desc="Explanation of the scoring task and dimension indicators"
    )
    text: str = dspy.InputField(
        desc="Text to score"
    )
    score: float = dspy.OutputField(
        desc="Score on the specified scale. Output a single number."
    )
    left_indicators: str = dspy.OutputField(
        desc="Key indicators for the lower end of the scale"
    )
    right_indicators: str = dspy.OutputField(
        desc="Key indicators for the higher end of the scale"
    )
    reasoning: str = dspy.OutputField(
        desc="Explanation of how the score was determined"
    )


class SimpleScore(dspy.Signature):
    """
    Score text on a bounded numeric scale with minimal output fields.

    A compact signature with a single output field to reduce format drift
    and truncation during optimization/evaluation loops.
    """
    task_context: str = dspy.InputField(
        desc="Scoring task description and criteria"
    )
    text: str = dspy.InputField(
        desc="Text to score"
    )
    score: float = dspy.OutputField(
        desc=(
            "Numeric score on the exact scale defined in task_context. "
            "Estimate as precisely as possible on the continuous scale; "
            "Output a single number only "
            "no markdown/backticks/code fences, no extra text); "
            "do not invent an alternate scale; "
            "do not output multiple numbers, ranges, or lists."
        )
    )


class PairwiseSummaryComparison(dspy.Signature):
    """
    Compare two summaries and select the one that better preserves information.

    Used by oracle models to generate preference data for training.
    See src.core.signatures.PairwiseComparison for the generic version.
    """
    rubric: str = dspy.InputField(
        desc="Information preservation criteria"
    )
    original_text: str = dspy.InputField(
        desc="Original source text being summarized"
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
        desc="Which summary is better: 'A', 'B', or 'tie'"
    )
    reasoning: str = dspy.OutputField(
        desc="Detailed explanation of why this summary better preserves the information"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence in the preference judgment (0.0 to 1.0)"
    )
    score_estimate_a: float = dspy.OutputField(
        desc="Estimated score for summary A. Output a single number."
    )
    score_estimate_b: float = dspy.OutputField(
        desc="Estimated score for summary B. Output a single number."
    )


class RILEComparison(dspy.Signature):
    """
    Compare scores between original and summarized text.

    Used for auditing whether summarization preserves target information.
    """
    task_context: str = dspy.InputField(
        desc="Explanation of the scoring task"
    )
    original_text: str = dspy.InputField(
        desc="Original text (more detailed)"
    )
    summary_text: str = dspy.InputField(
        desc="Summarized text"
    )
    original_rile: float = dspy.OutputField(
        desc="Score for original text. Output a single number."
    )
    summary_rile: float = dspy.OutputField(
        desc="Score for summary text. Output a single number."
    )
    score_difference: float = dspy.OutputField(
        desc="Absolute difference between scores. Output a single number."
    )
    is_preserved: bool = dspy.OutputField(
        desc="Whether information is adequately preserved"
    )
    drift_explanation: str = dspy.OutputField(
        desc="Explanation of any drift between original and summary"
    )


_STRICT_NUMERIC_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?$")
_LABELED_SCORE_RE = re.compile(
    r"(?i)^\s*(?:rile(?:\s+score)?|score|value|prediction)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)\s*$"
)


def _coerce_rile_range(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < RILE_MIN or parsed > RILE_MAX:
        return None
    return float(parsed)


def _strip_score_wrappers(text: str) -> str:
    rendered = str(text or "")
    cleaned = re.sub(r"<think>.*?</think>", "", rendered, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return cleaned.strip()


def _strict_parse_rile_score(value: Any) -> Optional[float]:
    cleaned = _strip_score_wrappers(str(value or ""))
    if not cleaned:
        return None

    # JSON outputs: prefer score-like keys and recurse.
    try:
        payload = json.loads(cleaned)
    except Exception:
        payload = None
    if payload is not None:
        if isinstance(payload, (int, float)):
            return _coerce_rile_range(payload)
        if isinstance(payload, str):
            return _strict_parse_rile_score(payload)
        if isinstance(payload, dict):
            for key in ("score", "rile", "rile_score", "value", "prediction"):
                if key in payload:
                    parsed = _strict_parse_rile_score(payload[key])
                    if parsed is not None:
                        return parsed
            return None
        if isinstance(payload, list) and len(payload) == 1:
            return _strict_parse_rile_score(payload[0])
        return None

    token = cleaned.strip("`\"'")
    if _STRICT_NUMERIC_RE.fullmatch(token):
        return _coerce_rile_range(token)

    candidates: list[float] = []
    for line in cleaned.splitlines():
        candidate_line = line.strip()
        if not candidate_line:
            continue
        if _STRICT_NUMERIC_RE.fullmatch(candidate_line):
            parsed = _coerce_rile_range(candidate_line)
            if parsed is not None:
                candidates.append(parsed)
            continue
        match = _LABELED_SCORE_RE.match(candidate_line)
        if match:
            parsed = _coerce_rile_range(match.group(1))
            if parsed is not None:
                candidates.append(parsed)

    if not candidates:
        return None
    return float(candidates[-1])


def _coerce_rile_score(raw_value: Any, raw_result: Any, *, strict_parse: bool = True) -> Optional[float]:
    parser = _strict_parse_rile_score if strict_parse else (
        lambda value: parse_numeric_score(
            str(value),
            min_value=RILE_MIN,
            max_value=RILE_MAX,
            allow_llm_fallback=False,
        )
    )

    if raw_value is not None:
        parsed = parser(raw_value)
        if parsed is not None:
            return parsed

    if raw_result is not None:
        parsed = parser(raw_result)
        if parsed is not None:
            return parsed

    return None


# Module implementations

class RILEScorer(dspy.Module):
    """DSPy module for RILE scoring."""

    def __init__(
        self,
        use_cot: bool = False,
        max_tokens: int = 64,
        temperature: float = 0.0,
        strict_parse: bool = True,
    ):
        super().__init__()
        # Default to the compact signature to keep scorer outputs short and
        # stable during GEPA optimization/evaluation loops.
        # Also cap completion tokens so the scorer cannot ramble.
        scorer_max_tokens = max(1, int(max_tokens))
        scorer_temperature = float(temperature)
        self._strict_parse = bool(strict_parse)
        self._use_cot = bool(use_cot)
        self._scorer_max_tokens = scorer_max_tokens
        self._scorer_temperature = scorer_temperature
        self._score_predictor = self._build_score_predictor()

    @property
    def score(self) -> Any:
        """Backwards-compatible accessor used by older optimization artifacts."""
        return self._score_predictor

    @score.setter
    def score(self, value: Any) -> None:
        self._score_predictor = value

    def _build_score_predictor(self) -> Any:
        score_signature = SimpleScore
        if self._use_cot:
            return dspy.ChainOfThought(
                score_signature,
                max_tokens=self._scorer_max_tokens,
                temperature=self._scorer_temperature,
            )
        return dspy.Predict(
            score_signature,
            max_tokens=self._scorer_max_tokens,
            temperature=self._scorer_temperature,
        )

    def _resolve_score_predictor(self) -> Any:
        predictor = getattr(self, "_score_predictor", None)
        if callable(predictor):
            return predictor
        logger.warning(
            "RILEScorer predictor is non-callable (%s); rebuilding default predictor",
            type(predictor).__name__ if predictor is not None else "None",
        )
        predictor = self._build_score_predictor()
        self._score_predictor = predictor
        return predictor

    def forward(
        self,
        text: str = None,
        task_context: str = None,
        # Training example format (alternative signature)
        summary: str = None,
        rubric: str = None,
        original_content: str = None,  # Accepted but not used for pure scoring
        metadata: Optional[Dict[str, Any]] = None,
        dspy_config: Optional[dict[str, Any]] = None,
    ) -> dict:
        """
        Score text on the RILE scale.

        Accepts either:
        - text + task_context (original format)
        - summary + rubric + original_content (training example format)

        Args:
            text: Political text to score
            task_context: Explanation of the scoring task
            summary: Alternative name for text (from training examples)
            rubric: Alternative name for task_context (from training examples)
            original_content: Ignored, accepted for compatibility
            metadata: Optional structured document metadata (year/country/party)

        Returns:
            Dictionary with score and analysis
        """
        global _SCORER_DIAG_BUDGET
        diag_enabled = _scorer_diag_enabled()

        def _diag_log(message: str, *args: Any) -> None:
            global _SCORER_DIAG_BUDGET
            if not diag_enabled:
                return
            if _SCORER_DIAG_BUDGET <= 0:
                return
            _SCORER_DIAG_BUDGET -= 1
            logger.info("RILEScorer diag: " + message, *args)

        # Support both calling conventions
        actual_text = text if text is not None else summary
        actual_context = task_context if task_context is not None else rubric

        if actual_text is None:
            raise ValueError("Either 'text' or 'summary' must be provided")
        if actual_context is None:
            raise ValueError("Either 'task_context' or 'rubric' must be provided")

        metadata_block = format_prompt_metadata_block(metadata if isinstance(metadata, dict) else None)
        if metadata_block:
            if str(actual_context).strip():
                actual_context = f"{actual_context}\n\n{metadata_block}"
            else:
                actual_context = metadata_block

        request_config: Optional[dict[str, Any]] = None
        if isinstance(dspy_config, dict) and dspy_config:
            request_config = dict(dspy_config)

        def _extract_lm_response_from_exception(exc: Exception) -> Optional[str]:
            message = str(exc or "")
            if not message:
                return None
            match = re.search(
                r"LM Response:\s*(.*?)(?:\n\nExpected to find|\Z)",
                message,
                flags=re.DOTALL,
            )
            if not match:
                return None
            candidate = match.group(1).strip()
            return candidate or None

        def _score_once(task_ctx: str) -> Optional[float]:
            predictor = self._resolve_score_predictor()
            try:
                if request_config is None:
                    result = predictor(task_context=task_ctx, text=actual_text)
                else:
                    result = predictor(
                        task_context=task_ctx,
                        text=actual_text,
                        config=request_config,
                    )
            except Exception as exc:
                # Defensive recovery for rare optimizer artifacts that overwrite
                # predictor state with a scalar.
                if "not callable" in str(exc).lower():
                    predictor = self._build_score_predictor()
                    self._score_predictor = predictor
                    try:
                        if request_config is None:
                            result = predictor(task_context=task_ctx, text=actual_text)
                        else:
                            result = predictor(
                                task_context=task_ctx,
                                text=actual_text,
                                config=request_config,
                            )
                    except Exception as retry_exc:
                        exc = retry_exc
                    else:
                        accessor = NormalizedOutputAccessor(result)
                        parsed = _coerce_rile_score(
                            accessor.get("score", None),
                            result,
                            strict_parse=self._strict_parse,
                        )
                        _diag_log(
                            "result parsed=%s score_field=%r result_snippet=%s",
                            parsed,
                            accessor.get("score", None),
                            str(result)[:240].replace("\n", " "),
                        )
                        return parsed
                lm_response = _extract_lm_response_from_exception(exc)
                if lm_response:
                    parsed = (
                        _strict_parse_rile_score(lm_response)
                        if self._strict_parse
                        else parse_numeric_score(
                            lm_response,
                            min_value=RILE_MIN,
                            max_value=RILE_MAX,
                            allow_llm_fallback=False,
                        )
                    )
                    if parsed is not None:
                        _diag_log(
                            "parsed score from exception payload parsed=%s snippet=%s",
                            parsed,
                            str(lm_response)[:240].replace("\n", " "),
                        )
                        return parsed
                logger.warning("RILEScorer prediction failed; defaulting to neutral. Error: %s", exc)
                _diag_log("exception fallback to neutral candidate error=%s", exc)
                return None

            accessor = NormalizedOutputAccessor(result)
            parsed = _coerce_rile_score(
                accessor.get("score", None),
                result,
                strict_parse=self._strict_parse,
            )
            _diag_log(
                "result parsed=%s score_field=%r result_snippet=%s",
                parsed,
                accessor.get("score", None),
                str(result)[:240].replace("\n", " "),
            )
            if parsed is None:
                logger.debug(
                    "RILEScorer parse miss (strict=%s) for output snippet: %s",
                    self._strict_parse,
                    str(result)[:240],
                )
            return parsed

        raw_score = _score_once(actual_context)

        if raw_score is None:
            retry_context = (
                f"{actual_context}\n\n"
                "IMPORTANT: Output ONLY the numeric score as plain text. "
                "No words, labels, units, punctuation (other than a leading '-' and optional decimal point)."
            )
            raw_score = _score_once(retry_context)

        if raw_score is None:
            logger.warning("RILEScorer could not parse score after retry; defaulting to neutral score 0.0")
            raw_score = 0.0
            _diag_log("final fallback raw_score=0.0 after retry")
        normalized = (raw_score - RILE_MIN) / (RILE_MAX - RILE_MIN)
        normalized = max(0.0, min(1.0, normalized))

        _diag_log("normalized score=%s from raw_score=%s", normalized, raw_score)

        return {'score': normalized}


class RILEComparator(dspy.Module):
    """DSPy module for comparing RILE scores between texts.

    Honors the repo-wide output-budget convention via
    ``DEFAULT_COMPARATOR_MAX_TOKENS`` (see
    ``src/tasks/manifesto/pipeline_config.py``). The ``drift_explanation``
    OutputField is a short-prose field whose length must stay bounded so
    the call fits comfortably in the vLLM context window even when the
    inputs are long.
    """

    def __init__(
        self,
        threshold: float = 10.0,
        use_cot: bool = False,
        *,
        max_output_tokens: Optional[int] = None,
    ):
        super().__init__()
        from .pipeline_config import DEFAULT_COMPARATOR_MAX_TOKENS
        if use_cot:
            self.compare = dspy.ChainOfThought(RILEComparison)
        else:
            self.compare = dspy.Predict(RILEComparison)
        self.threshold = threshold
        self.max_output_tokens = (
            int(max_output_tokens) if max_output_tokens is not None
            else DEFAULT_COMPARATOR_MAX_TOKENS
        )

    def forward(self, original_text: str, summary_text: str, task_context: str) -> dict:
        """Compare RILE positions between original and summary."""
        from .pipeline import _call_with_budget
        result = _call_with_budget(
            self.compare,
            max_tokens=self.max_output_tokens,
            task_context=task_context,
            original_text=original_text,
            summary_text=summary_text,
        )

        # Use normalized accessor to handle key casing variations
        accessor = NormalizedOutputAccessor(result)

        raw_original = _coerce_rile_score(accessor.get('original_rile', None), result)
        raw_summary = _coerce_rile_score(accessor.get('summary_rile', None), result)
        raw_original = 0.0 if raw_original is None else raw_original
        raw_summary = 0.0 if raw_summary is None else raw_summary

        norm_original = (raw_original - RILE_MIN) / (RILE_MAX - RILE_MIN)
        norm_summary = (raw_summary - RILE_MIN) / (RILE_MAX - RILE_MIN)
        norm_original = max(0.0, min(1.0, norm_original))
        norm_summary = max(0.0, min(1.0, norm_summary))

        return {
            'original_rile': norm_original,
            'summary_rile': norm_summary,
            'score_difference': abs(norm_original - norm_summary),
            'is_preserved': accessor.get('is_preserved', True),
            'drift_explanation': accessor.get('drift_explanation', ''),
        }
