from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional


ChatMessage = Mapping[str, str]


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


class OpenAICompatibleChatClient:
    """Small requests-based client for OpenAI-compatible chat endpoints."""

    def __init__(
        self,
        *,
        api_base: str,
        model: Optional[str] = None,
        api_key: str = "EMPTY",
        timeout_seconds: float = 120.0,
        session: Optional[Any] = None,
        verify_model: bool = True,
        default_temperature: float = 0.0,
        default_max_tokens: int = 16,
    ) -> None:
        self.api_base = str(api_base or "").rstrip("/")
        if not self.api_base:
            raise ValueError("OpenAI-compatible chat client requires api_base")
        self.model = None if model is None else str(model)
        self.api_key = api_key or "EMPTY"
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._session = session
        self.verify_model = bool(verify_model)
        self.default_temperature = float(default_temperature)
        self.default_max_tokens = int(default_max_tokens)
        self._model_verified = False

    @property
    def chat_completions_url(self) -> str:
        if self.api_base.endswith("/v1"):
            return f"{self.api_base}/chat/completions"
        return f"{self.api_base}/v1/chat/completions"

    @property
    def models_url(self) -> str:
        if self.api_base.endswith("/v1"):
            return f"{self.api_base}/models"
        return f"{self.api_base}/v1/models"

    def _requests(self) -> Any:
        if self._session is not None:
            return self._session
        import requests

        return requests

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def resolve_model(self) -> str:
        if self.model and self._model_verified:
            return self.model
        if self.model and not self.verify_model:
            return str(self.model)

        response = self._requests().get(
            self.models_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])
        if not data:
            raise RuntimeError("Chat endpoint returned no models")

        served_ids: list[str] = []
        for row in data:
            model_id = str(row.get("id", "")).strip()
            if model_id:
                served_ids.append(model_id)
        if not served_ids:
            raise RuntimeError("Chat endpoint returned empty model ids")

        if self.model:
            requested = str(self.model).strip()
            if requested in served_ids:
                self._model_verified = True
                return requested
            if len(served_ids) == 1:
                self.model = served_ids[0]
                self._model_verified = True
                return self.model
            raise RuntimeError(
                "Requested chat model id not served by endpoint. "
                f"requested={requested!r} served={served_ids!r}"
            )

        self.model = served_ids[0]
        self._model_verified = True
        return self.model

    def complete_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
    ) -> str:
        payload = render_chat_payload(
            model=str(model or self.resolve_model()),
            messages=messages,
            temperature=self.default_temperature if temperature is None else float(temperature),
            max_tokens=self.default_max_tokens if max_tokens is None else int(max_tokens),
        )
        if extra_body:
            payload.update(dict(extra_body))
        response = self._requests().post(
            self.chat_completions_url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return _chat_completion_text(response.json())

    def predict_text(self, *, messages: Sequence[ChatMessage], config: Any = None, **_: Any) -> str:
        temperature = getattr(config, "temperature", self.default_temperature)
        max_tokens = getattr(config, "max_tokens", self.default_max_tokens)
        return self.complete_chat(messages, temperature=temperature, max_tokens=max_tokens)


def build_chat_client(
    engine: str = "openai_compatible",
    *,
    api_base: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: str = "EMPTY",
    timeout_seconds: float = 120.0,
    session: Optional[Any] = None,
    verify_model: bool = True,
    default_temperature: float = 0.0,
    default_max_tokens: int = 16,
) -> OpenAICompatibleChatClient:
    """Build a chat client for an OpenAI-compatible endpoint."""

    engine_name = str(getattr(engine, "value", engine) or "openai_compatible")
    engine_name = engine_name.strip().lower().replace("-", "_")
    if engine_name not in {"openai", "openai_compatible", "vllm", "sglang"}:
        raise ValueError(f"Unsupported chat engine {engine_name!r}")
    resolved_base = str(api_base or base_url or "").strip()
    if not resolved_base:
        raise ValueError(f"Chat engine {engine_name!r} requires api_base/base_url.")
    return OpenAICompatibleChatClient(
        api_base=resolved_base,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        session=session,
        verify_model=verify_model,
        default_temperature=default_temperature,
        default_max_tokens=default_max_tokens,
    )


def _chat_completion_text(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        raise RuntimeError("Chat completion response did not contain choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise RuntimeError("Chat completion choice is not a mapping")
    message = first.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        return _content_to_text(content)
    if "text" in first:
        return _content_to_text(first.get("text"))
    raise RuntimeError("Chat completion choice did not contain message.content")


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


__all__ = [
    "ChatMessage",
    "OpenAICompatibleChatClient",
    "build_chat_client",
    "render_chat_payload",
]
