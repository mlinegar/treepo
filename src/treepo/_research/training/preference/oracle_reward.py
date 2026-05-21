"""
Oracle-alignment reward adapters for GRPO.

This module provides reward functions that score generated completions against a
task oracle/scorer, without any GenRM dependency.
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _value_for_index(values: Any, index: int) -> Any:
    if isinstance(values, (list, tuple)):
        if index < len(values):
            return values[index]
        return None
    return values


def create_oracle_alignment_reward_func(
    oracle_predict: Callable[[str], float],
    *,
    error_scale: float = 1.0,
    neutral_reward: float = 0.5,
    min_completion_chars: int = 8,
    short_completion_penalty: float = 0.1,
    cache_size: int = 4096,
    scorer_parallelism: int = 1,
) -> Callable[..., List[float]]:
    """
    Create a TRL-compatible GRPO reward function from a task oracle/scorer.

    Reward formula:
      reward = 1 - abs(predicted_score - reference_score) / error_scale, clipped to [0, 1]

    Expected dataset columns (passed through GRPO kwargs when present):
      - reference_score: oracle score target for each prompt
      - original_text: optional fallback source to compute reference_score on the fly
    """

    bounded_error_scale = max(1e-8, float(error_scale))
    bounded_neutral = max(0.0, min(1.0, float(neutral_reward)))
    min_chars = max(0, int(min_completion_chars))
    penalty = max(0.0, min(1.0, float(short_completion_penalty)))
    max_cache = max(1, int(cache_size))
    parallelism = max(1, int(scorer_parallelism))
    score_cache: "OrderedDict[str, float]" = OrderedDict()
    cache_lock = Lock()

    def _cache_get(key: str) -> Optional[float]:
        with cache_lock:
            cached = score_cache.get(key)
            if cached is not None:
                score_cache.move_to_end(key, last=True)
            return cached

    def _cache_set(key: str, value: float) -> None:
        with cache_lock:
            score_cache[key] = value
            if len(score_cache) > max_cache:
                score_cache.popitem(last=False)

    def _score_text(text: str) -> Optional[float]:
        key = str(text)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        try:
            value = _coerce_float(oracle_predict(key))
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Oracle scorer failed in GRPO reward evaluation: %s", exc)
            return None
        if value is None:
            return None
        _cache_set(key, value)
        return value

    def _score_many(texts: Sequence[str]) -> dict[str, Optional[float]]:
        ordered: List[str] = []
        seen: set[str] = set()
        for text in texts:
            key = str(text)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)

        out: dict[str, Optional[float]] = {}
        pending: List[str] = []
        for key in ordered:
            cached = _cache_get(key)
            if cached is None:
                pending.append(key)
            else:
                out[key] = cached

        if pending:
            if parallelism <= 1 or len(pending) == 1:
                for key in pending:
                    out[key] = _score_text(key)
            else:
                with ThreadPoolExecutor(max_workers=parallelism) as executor:
                    futures = {executor.submit(_score_text, key): key for key in pending}
                    for future in as_completed(futures):
                        key = futures[future]
                        try:
                            out[key] = future.result()
                        except Exception as exc:  # pragma: no cover - defensive guard
                            logger.debug("Oracle scorer worker failed for key=%r: %s", key, exc)
                            out[key] = None
        return out

    def reward_func(
        completions: Sequence[str],
        prompts: Sequence[str],  # required by TRL GRPO signature
        **kwargs: Any,
    ) -> List[float]:
        rewards: List[float] = []
        reference_values = kwargs.get("reference_score")
        original_values = kwargs.get("original_text")

        # Keep prompts in signature for compatibility; reward is completion-centric.
        _ = prompts

        to_score: List[str] = []
        for idx, completion in enumerate(completions):
            stripped = str(completion or "").strip()
            if min_chars > 0 and len(stripped) < min_chars:
                continue
            to_score.append(stripped)
            reference = _coerce_float(_value_for_index(reference_values, idx))
            if reference is None:
                original_text = _value_for_index(original_values, idx)
                if isinstance(original_text, str) and original_text.strip():
                    to_score.append(original_text.strip())
        scored = _score_many(to_score)

        for idx, completion in enumerate(completions):
            completion_text = str(completion or "")
            stripped = completion_text.strip()
            if min_chars > 0 and len(stripped) < min_chars:
                rewards.append(max(0.0, bounded_neutral - penalty))
                continue

            predicted = scored.get(stripped)
            if predicted is None:
                predicted = _score_text(stripped)
            if predicted is None:
                rewards.append(bounded_neutral)
                continue

            reference = _coerce_float(_value_for_index(reference_values, idx))
            if reference is None:
                original_text = _value_for_index(original_values, idx)
                if isinstance(original_text, str) and original_text.strip():
                    original_key = original_text.strip()
                    reference = scored.get(original_key)
                    if reference is None:
                        reference = _score_text(original_key)

            if reference is None:
                rewards.append(bounded_neutral)
                continue

            raw_reward = 1.0 - abs(predicted - reference) / bounded_error_scale
            rewards.append(max(0.0, min(1.0, float(raw_reward))))

        return rewards

    return reward_func


def _strict_same_side_raw(pred_raw: float, true_raw: float, neutral_raw: float = 0.0) -> bool:
    pred_delta = float(pred_raw) - float(neutral_raw)
    if abs(pred_delta) <= 1e-9:
        return False
    true_delta = float(true_raw) - float(neutral_raw)
    return bool(pred_delta * true_delta > 0.0)


def create_local_law_summary_reward_func(
    oracle_predict: Callable[[str], float],
    *,
    c1_threshold_raw: float = 10.0,
    c2_threshold_raw: float = 6.0,
    neutral_raw: float = 0.0,
    c1_weight: float = 0.6,
    c2_weight: float = 0.3,
    same_side_weight: float = 0.1,
    neutral_reward: float = 0.25,
    parse_failure_reward: float = 0.0,
    min_completion_chars: int = 8,
    short_completion_penalty: float = 0.1,
    cache_size: int = 4096,
    scorer_parallelism: int = 1,
) -> Callable[..., List[float]]:
    """
    Create a local-law reward for summary GRPO.

    Uses dataset columns (when available):
      - reference_score: source score target (C1 anchor)
      - input_text: previous-hop/source text (C2 drift anchor)
      - hop: resummary hop index (increase C2 emphasis for hop>=2)
    """

    c1_scale = max(1e-8, float(c1_threshold_raw))
    c2_scale = max(1e-8, float(c2_threshold_raw))
    w_c1 = max(0.0, float(c1_weight))
    w_c2 = max(0.0, float(c2_weight))
    w_side = max(0.0, float(same_side_weight))
    bounded_neutral = max(0.0, min(1.0, float(neutral_reward)))
    bounded_parse_failure = max(0.0, min(1.0, float(parse_failure_reward)))
    min_chars = max(0, int(min_completion_chars))
    penalty = max(0.0, min(1.0, float(short_completion_penalty)))
    max_cache = max(1, int(cache_size))
    parallelism = max(1, int(scorer_parallelism))
    score_cache: "OrderedDict[str, float]" = OrderedDict()
    cache_lock = Lock()

    def _cache_get(key: str) -> Optional[float]:
        with cache_lock:
            cached = score_cache.get(key)
            if cached is not None:
                score_cache.move_to_end(key, last=True)
            return cached

    def _cache_set(key: str, value: float) -> None:
        with cache_lock:
            score_cache[key] = value
            if len(score_cache) > max_cache:
                score_cache.popitem(last=False)

    def _score_text(text: str) -> Optional[float]:
        key = str(text)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        try:
            value = _coerce_float(oracle_predict(key))
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Oracle scorer failed in local-law reward evaluation: %s", exc)
            return None
        if value is None:
            return None
        _cache_set(key, value)
        return value

    def _score_many(texts: Sequence[str]) -> dict[str, Optional[float]]:
        ordered: List[str] = []
        seen: set[str] = set()
        for text in texts:
            key = str(text)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)

        out: dict[str, Optional[float]] = {}
        pending: List[str] = []
        for key in ordered:
            cached = _cache_get(key)
            if cached is None:
                pending.append(key)
            else:
                out[key] = cached

        if pending:
            if parallelism <= 1 or len(pending) == 1:
                for key in pending:
                    out[key] = _score_text(key)
            else:
                with ThreadPoolExecutor(max_workers=parallelism) as executor:
                    futures = {executor.submit(_score_text, key): key for key in pending}
                    for future in as_completed(futures):
                        key = futures[future]
                        try:
                            out[key] = future.result()
                        except Exception as exc:  # pragma: no cover - defensive guard
                            logger.debug("Oracle scorer worker failed for key=%r: %s", key, exc)
                            out[key] = None
        return out

    def reward_func(
        completions: Sequence[str],
        prompts: Sequence[str],  # required by TRL GRPO signature
        **kwargs: Any,
    ) -> List[float]:
        rewards: List[float] = []
        reference_values = kwargs.get("reference_score")
        input_values = kwargs.get("input_text")
        hop_values = kwargs.get("hop")

        _ = prompts

        to_score: List[str] = []
        for idx, completion in enumerate(completions):
            stripped = str(completion or "").strip()
            if min_chars > 0 and len(stripped) < min_chars:
                continue
            to_score.append(stripped)
            input_text = _value_for_index(input_values, idx)
            if isinstance(input_text, str) and input_text.strip():
                to_score.append(input_text.strip())
        scored = _score_many(to_score)

        for idx, completion in enumerate(completions):
            completion_text = str(completion or "")
            stripped = completion_text.strip()
            if min_chars > 0 and len(stripped) < min_chars:
                rewards.append(max(0.0, bounded_neutral - penalty))
                continue

            completion_score = scored.get(stripped)
            if completion_score is None:
                completion_score = _score_text(stripped)
            if completion_score is None:
                rewards.append(bounded_parse_failure)
                continue

            source_score = _coerce_float(_value_for_index(reference_values, idx))
            if source_score is None:
                rewards.append(bounded_neutral)
                continue

            input_text = _value_for_index(input_values, idx)
            input_score = None
            if isinstance(input_text, str) and input_text.strip():
                input_key = input_text.strip()
                input_score = scored.get(input_key)
                if input_score is None:
                    input_score = _score_text(input_key)

            hop_value = _value_for_index(hop_values, idx)
            try:
                hop = int(hop_value) if hop_value is not None else 1
            except (TypeError, ValueError):
                hop = 1
            c2_multiplier = 1.0 if hop >= 2 else 0.5
            effective_w_c2 = w_c2 * c2_multiplier if input_score is not None else 0.0

            c1_reward = max(0.0, min(1.0, 1.0 - abs(completion_score - source_score) / c1_scale))
            c2_reward = 0.0
            if input_score is not None:
                c2_reward = max(0.0, min(1.0, 1.0 - abs(completion_score - input_score) / c2_scale))
            same_side_reward = 1.0 if _strict_same_side_raw(completion_score, source_score, neutral_raw) else 0.0

            denom = w_c1 + effective_w_c2 + w_side
            if denom <= 1e-9:
                rewards.append(bounded_neutral)
                continue

            combined = (
                w_c1 * c1_reward
                + effective_w_c2 * c2_reward
                + w_side * same_side_reward
            ) / denom
            rewards.append(max(0.0, min(1.0, float(combined))))

        return rewards

    return reward_func


__all__ = [
    "create_oracle_alignment_reward_func",
    "create_local_law_summary_reward_func",
]
