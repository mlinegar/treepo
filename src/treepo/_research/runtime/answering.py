from __future__ import annotations

import math
import re
from typing import Any, Dict, Mapping, Optional, Sequence

from treepo._research.runtime.backbone import BackboneAdapter
from treepo._research.runtime.contracts import AnswerResult, AnswerSpec, ModelResponse, RuntimeConfig
from treepo._research.runtime.memory import TokenCounter


CHOICE_RE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", re.IGNORECASE)


def render_choices(choices: Mapping[str, str]) -> str:
    return "\n".join(
        f"{letter}. {choices[letter]}"
        for letter in ("A", "B", "C", "D")
        if letter in choices
    )


def parse_multi_choice_text(text: str, *, valid_choices: Sequence[str] = ("A", "B", "C", "D")) -> str:
    valid = {str(choice).strip().upper() for choice in valid_choices}
    rendered = str(text or "").strip()
    if not rendered:
        return ""
    answer_match = re.search(r"answer\s*[:：]\s*([ABCD])", rendered, re.IGNORECASE)
    if answer_match:
        letter = answer_match.group(1).upper()
        return letter if letter in valid else ""
    matches = [match.group(1).upper() for match in CHOICE_RE.finditer(rendered)]
    for letter in reversed(matches):
        if letter in valid:
            return letter
    first = rendered[:1].upper()
    return first if first in valid else ""


def build_multi_choice_prompt(*, prompt: str, answer_spec: AnswerSpec) -> str:
    rendered = str(prompt or "").rstrip()
    choices = render_choices(answer_spec.choices)
    if choices and "Choices:" not in rendered[-2000:]:
        rendered += f"\n\nChoices:\n{choices}"
    instruction = answer_spec.instruction or "Choose the single best option. Return only A, B, C, or D."
    if instruction and instruction not in rendered[-2000:]:
        rendered += f"\n\n{instruction}"
    answer_prefix = answer_spec.answer_prefix or "Answer:"
    if answer_prefix and not rendered.rstrip().endswith(answer_prefix):
        rendered += f"\n\n{answer_prefix}"
    return rendered


def _getattr_or_key(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _token_text(token_obj: Any) -> str:
    return str(_getattr_or_key(token_obj, "token", "") or "")


def _token_logprob(token_obj: Any) -> Optional[float]:
    raw = _getattr_or_key(token_obj, "logprob", None)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def extract_choice_logprobs(raw_response: Any, *, valid_choices: Sequence[str] = ("A", "B", "C", "D")) -> Dict[str, float]:
    """Best-effort extraction from OpenAI/vLLM chat-completion logprob payloads."""
    valid = {str(choice).strip().upper() for choice in valid_choices}
    scores: Dict[str, float] = {}
    choices = _getattr_or_key(raw_response, "choices", []) or []
    if not choices:
        return scores
    first_choice = choices[0]
    logprobs = _getattr_or_key(first_choice, "logprobs", None)
    content = _getattr_or_key(logprobs, "content", None)
    if not content:
        return scores
    first_token = content[0]
    candidates = list(_getattr_or_key(first_token, "top_logprobs", None) or [])
    candidates.append(first_token)
    for candidate in candidates:
        token = _token_text(candidate).strip()
        if not token:
            continue
        letter = token[:1].upper()
        if letter not in valid:
            continue
        score = _token_logprob(candidate)
        if score is None:
            continue
        if letter not in scores or score > scores[letter]:
            scores[letter] = score
    return scores


def _response_cost(resp: ModelResponse, counter: TokenCounter, prompt: str) -> Dict[str, Any]:
    prompt_tokens = int(resp.prompt_tokens or counter.count(prompt))
    completion_tokens = int(resp.completion_tokens or counter.count(resp.text))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "n_calls": 1,
        "wall_ms": float(resp.latency_ms),
    }


def _merge_cost(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "prompt_tokens": int(left.get("prompt_tokens", 0) or 0) + int(right.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(left.get("completion_tokens", 0) or 0) + int(right.get("completion_tokens", 0) or 0),
        "n_calls": int(left.get("n_calls", 0) or 0) + int(right.get("n_calls", 0) or 0),
        "wall_ms": float(left.get("wall_ms", 0.0) or 0.0) + float(right.get("wall_ms", 0.0) or 0.0),
    }


def answer_multi_choice(
    *,
    bb: BackboneAdapter,
    prompt: str,
    answer_spec: AnswerSpec,
    runtime: RuntimeConfig,
    counter: TokenCounter,
) -> AnswerResult:
    """Answer A-D questions with logprobs when available, then text fallback."""
    valid_choices = tuple(answer_spec.choices.keys()) or ("A", "B", "C", "D")
    prompt = build_multi_choice_prompt(prompt=prompt, answer_spec=answer_spec)
    messages = [{"role": "user", "content": prompt}]
    strategy = str(runtime.choice_decode_strategy or "logprobs_then_generate")

    if strategy != "generate_only" and bool(getattr(bb, "supports_logprobs", False)):
        try:
            resp = bb.generate(
                messages,
                max_tokens=1,
                temperature=0.0,
                extra={"logprobs": True, "top_logprobs": 20},
            )
            cost = _response_cost(resp, counter, prompt)
            scores = extract_choice_logprobs(resp.raw, valid_choices=valid_choices)
            if scores:
                prediction = max(scores.items(), key=lambda item: item[1])[0]
                return AnswerResult(
                    prediction=prediction,
                    raw_text=resp.text.strip(),
                    decode_method="single_token_logprobs",
                    choice_scores=scores,
                    cost=cost,
                )
            parsed = parse_multi_choice_text(resp.text, valid_choices=valid_choices)
            if parsed:
                return AnswerResult(
                    prediction=parsed,
                    raw_text=resp.text.strip(),
                    decode_method="single_token_text",
                    choice_scores=scores,
                    cost=cost,
                )
        except Exception as exc:
            logprob_error = f"{type(exc).__name__}: {exc}"
        else:
            logprob_error = "no_choice_logprobs"
    else:
        logprob_error = "logprobs_disabled"

    resp = bb.generate(
        messages,
        max_tokens=max(1, int(runtime.max_output_tokens)),
        temperature=0.0,
    )
    cost = _response_cost(resp, counter, prompt)
    prediction = parse_multi_choice_text(resp.text, valid_choices=valid_choices)
    return AnswerResult(
        prediction=prediction or resp.text.strip(),
        raw_text=resp.text.strip(),
        decode_method="generate_parse",
        choice_scores={},
        cost=cost,
        artifacts={"logprob_fallback_reason": logprob_error},
    )


__all__ = [
    "answer_multi_choice",
    "build_multi_choice_prompt",
    "extract_choice_logprobs",
    "parse_multi_choice_text",
    "render_choices",
]

