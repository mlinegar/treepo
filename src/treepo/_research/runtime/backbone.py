from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from treepo._research.core.engines import EngineSurface, default_engine_port
from treepo._research.core.inference_engine import build_inference_engine
from treepo._research.runtime.contracts import ChatInput, InferenceRequest, ModelResponse


@dataclass(frozen=True)
class BackboneConfig:
    engine: str = ""
    base_url: str = "http://localhost:8000/v1"
    model: str = "default"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    timeout: float = 120.0


class BackboneAdapter:
    """Thin wrapper around the universal chat inference engine."""

    @staticmethod
    def _infer_engine(base_url: str, explicit_engine: str = "") -> str:
        if explicit_engine:
            return str(explicit_engine)
        parsed = urlparse(str(base_url))
        host = str(parsed.netloc or "").lower()
        if "api.openai.com" in host:
            return "openai"
        port = parsed.port
        if port == default_engine_port("sglang", role="task"):
            return "sglang"
        if port == default_engine_port("vllm", role="task"):
            return "vllm"
        return "custom_http"

    def __init__(
        self,
        *,
        config: BackboneConfig,
        mock: bool = False,
        enable_cache: bool = True,
    ):
        resolved_model = config.model
        if resolved_model == "default" and "api.openai.com" not in config.base_url:
            from treepo._research.core.model_detection import detect_model_sync

            resolved_model = detect_model_sync(config.base_url, fallback="default", timeout=2.0)

        self.config = replace(config, model=resolved_model)
        self._engine_name = self._infer_engine(self.config.base_url, self.config.engine)
        self.supports_logprobs = (not bool(mock)) and self._engine_name in {"vllm", "sglang", "openai", "custom_http"}
        self.engine = build_inference_engine(
            self._engine_name,
            surface=EngineSurface.CHAT_OPENAI,
            base_url=self.config.base_url,
            model=self.config.model,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            mock=mock,
            enable_cache=enable_cache,
        )
        self.client = self.engine

    def model_id(self) -> str:
        return self.config.model

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        start = time.time()
        kwargs: Dict[str, Any] = {
            "max_tokens": int(max_tokens),
            "temperature": float(self.config.temperature if temperature is None else temperature),
        }
        if stop:
            kwargs["stop"] = stop
        if extra:
            kwargs["extra"] = dict(extra)

        response = self.engine.execute(
            InferenceRequest(
                surface=EngineSurface.CHAT_OPENAI,
                input=ChatInput(
                    messages=list(messages),
                    max_tokens=int(max_tokens),
                    temperature=float(self.config.temperature if temperature is None else temperature),
                    stop=list(stop or []),
                    extra=dict(extra or {}),
                ),
            )
        )
        model_response = response.to_model_response()
        latency_ms = (time.time() - start) * 1000.0
        return ModelResponse(
            text=model_response.text,
            model_id=model_response.model_id,
            prompt_tokens=model_response.prompt_tokens,
            completion_tokens=model_response.completion_tokens,
            latency_ms=latency_ms,
            raw=model_response.raw,
        )
