from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


ChatMessage = Mapping[str, str]


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout_s: float = 120.0


def render_chat_payload(
    *,
    model: str,
    messages: Sequence[ChatMessage],
    temperature: float = 0.0,
    max_tokens: int = 16,
) -> dict[str, object]:
    return {
        "model": str(model),
        "messages": [dict(message) for message in messages],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }


__all__ = ["ChatMessage", "OpenAICompatibleConfig", "render_chat_payload"]
