"""Backend-agnostic diffusion inference adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

import requests

from treepo._research.core.engines import (
    EngineSurface,
    EngineType,
    resolve_engine_base_url,
    resolve_engine_for_usage,
)
from treepo._research.core.url_utils import normalize_generate_base_url

logger = logging.getLogger(__name__)


@dataclass
class DiffusionGeneration:
    """One generated text returned by a diffusion request."""

    input_text: str
    output_text: str
    finish_reason: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class DiffusionBatchResponse:
    """Structured response for a batch diffusion request."""

    generations: List[DiffusionGeneration]
    latency_seconds: float
    request_payload: Dict[str, Any]
    raw_response: Any = None
    model: Optional[str] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)

    @property
    def texts(self) -> List[str]:
        """Return generated texts in order."""
        return [generation.output_text for generation in self.generations]


class DiffusionBackend(Protocol):
    """Protocol for backend-specific diffusion inference adapters."""

    backend_name: str

    def generate(
        self,
        texts: Sequence[str] | str,
        sampling_params: Optional[Mapping[str, Any]] = None,
        engine_options: Optional[Mapping[str, Any]] = None,
    ) -> DiffusionBatchResponse:
        ...


class HTTPGenerateDiffusionBackend:
    """Generic HTTP diffusion backend that talks to a `/generate`-style endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        backend_name: str,
        model: Optional[str] = None,
        timeout: float = 120.0,
        session: Optional[requests.Session] = None,
        generate_path: str = "/generate",
        default_payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.backend_name = backend_name
        self.model = model
        self.timeout = float(timeout)
        self.session = session or requests.Session()
        self.generate_path = generate_path
        self.default_payload = dict(default_payload or {})

    def generate(
        self,
        texts: Sequence[str] | str,
        sampling_params: Optional[Mapping[str, Any]] = None,
        engine_options: Optional[Mapping[str, Any]] = None,
    ) -> DiffusionBatchResponse:
        prompts = [texts] if isinstance(texts, str) else [str(text) for text in texts]
        if not prompts:
            raise ValueError(f"{self.__class__.__name__}.generate requires at least one input text.")

        payload: Dict[str, Any] = dict(self.default_payload)
        payload["text"] = prompts if len(prompts) > 1 else prompts[0]
        if self.model:
            payload["model"] = self.model
        if sampling_params:
            payload.update(dict(sampling_params))
        resolved_engine_options = dict(engine_options or {})
        if resolved_engine_options:
            payload.update(resolved_engine_options)

        started = time.time()
        response = self.session.post(
            f"{self.base_url}{self.generate_path}",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        latency_seconds = time.time() - started

        outputs = self._extract_texts(data, expected=len(prompts))
        if len(outputs) != len(prompts):
            raise ValueError(
                f"{self.backend_name} diffusion response size mismatch: expected {len(prompts)} outputs, "
                f"received {len(outputs)}."
            )

        finish_reasons = self._extract_finish_reasons(data, len(outputs))
        generations = [
            DiffusionGeneration(
                input_text=prompt,
                output_text=output,
                finish_reason=finish_reasons[index] if index < len(finish_reasons) else None,
                raw=self._extract_generation_raw(data, index),
            )
            for index, (prompt, output) in enumerate(zip(prompts, outputs))
        ]

        telemetry = {
            "backend_name": self.backend_name,
            "request_count": len(prompts),
            "engine_options": dict(resolved_engine_options),
        }
        return DiffusionBatchResponse(
            generations=generations,
            latency_seconds=latency_seconds,
            request_payload=payload,
            raw_response=data,
            model=self.model,
            telemetry=telemetry,
        )

    def _normalize_algorithm_config(
        self,
        dllm_algorithm_config: Optional[Mapping[str, Any] | str],
    ) -> Optional[Dict[str, Any]]:
        if dllm_algorithm_config is None:
            return None
        if isinstance(dllm_algorithm_config, str):
            text = dllm_algorithm_config.strip()
            if not text:
                return None
            return json.loads(text)
        return dict(dllm_algorithm_config)

    def _extract_texts(self, data: Any, *, expected: int) -> List[str]:
        if isinstance(data, str):
            return [data]
        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                return [str(item) for item in data]
            outputs: List[str] = []
            for item in data:
                nested = self._extract_texts(item, expected=1)
                if nested:
                    outputs.append(nested[0])
            return outputs
        if not isinstance(data, dict):
            return []

        for key in ("texts", "generated_texts", "outputs"):
            value = data.get(key)
            if isinstance(value, list):
                return [str(item) for item in value]

        for key in ("text", "generated_text", "output"):
            value = data.get(key)
            if isinstance(value, str):
                return [value]
            if isinstance(value, list):
                return [str(item) for item in value]

        choices = data.get("choices")
        if isinstance(choices, list):
            outputs = []
            for choice in choices:
                if isinstance(choice, dict):
                    if isinstance(choice.get("text"), str):
                        outputs.append(choice["text"])
                        continue
                    message = choice.get("message")
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        outputs.append(message["content"])
                        continue
                    if isinstance(choice.get("output_text"), str):
                        outputs.append(choice["output_text"])
                        continue
            if outputs:
                return outputs

        data_field = data.get("data")
        if data_field is not None:
            nested = self._extract_texts(data_field, expected=expected)
            if nested:
                return nested

        logger.debug("Unable to identify diffusion texts in %s response payload: %s", self.backend_name, data)
        return []

    def _extract_finish_reasons(self, data: Any, expected: int) -> List[Optional[str]]:
        if not isinstance(data, dict):
            return [None] * expected
        choices = data.get("choices")
        if isinstance(choices, list):
            reasons: List[Optional[str]] = []
            for choice in choices:
                if isinstance(choice, dict):
                    reasons.append(choice.get("finish_reason"))
            if reasons:
                return reasons
        finish_reason = data.get("finish_reason")
        if isinstance(finish_reason, str):
            return [finish_reason]
        if isinstance(finish_reason, list):
            return [str(item) if item is not None else None for item in finish_reason]
        return [None] * expected

    def _extract_generation_raw(self, data: Any, index: int) -> Optional[Dict[str, Any]]:
        if not isinstance(data, dict):
            return None
        choices = data.get("choices")
        if isinstance(choices, list) and index < len(choices) and isinstance(choices[index], dict):
            return dict(choices[index])
        return None


class SGLangDiffusionBackend(HTTPGenerateDiffusionBackend):
    """Concrete diffusion adapter for SGLang text DLLMs."""

    def __init__(
        self,
        base_url: str = "http://localhost:30000",
        *,
        model: Optional[str] = None,
        timeout: float = 120.0,
        session: Optional[requests.Session] = None,
        default_payload: Optional[Mapping[str, Any]] = None,
        generate_path: str = "/generate",
    ) -> None:
        super().__init__(
            base_url=base_url,
            backend_name="sglang",
            model=model,
            timeout=timeout,
            session=session,
            generate_path=generate_path,
            default_payload=default_payload,
        )


class VLLMOmniDiffusionBackend(HTTPGenerateDiffusionBackend):
    """Concrete diffusion adapter for a vLLM-Omni-style `/generate` surface."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        model: Optional[str] = None,
        timeout: float = 120.0,
        session: Optional[requests.Session] = None,
        default_payload: Optional[Mapping[str, Any]] = None,
        generate_path: str = "/generate",
    ) -> None:
        super().__init__(
            base_url=base_url,
            backend_name="vllm_omni",
            model=model,
            timeout=timeout,
            session=session,
            generate_path=generate_path,
            default_payload=default_payload,
        )


class InferenceDiffusionBackendAdapter:
    """Compatibility adapter that exposes the legacy diffusion backend protocol."""

    def __init__(self, inference_engine: Any, *, backend_name: str) -> None:
        self.inference_engine = inference_engine
        self.backend_name = backend_name

    def generate(
        self,
        texts: Sequence[str] | str,
        sampling_params: Optional[Mapping[str, Any]] = None,
        engine_options: Optional[Mapping[str, Any]] = None,
    ) -> DiffusionBatchResponse:
        from treepo._research.runtime.contracts import DiffusionInput, InferenceRequest, TextListOutput

        prompts = [texts] if isinstance(texts, str) else [str(text) for text in texts]
        request = InferenceRequest(
            surface=EngineSurface.DIFFUSION_GENERATE,
            input=DiffusionInput(
                texts=prompts,
                sampling_params=dict(sampling_params or {}),
            ),
            engine_options=dict(engine_options or {}),
        )
        response = self.inference_engine.execute(request)
        if not isinstance(response.output, TextListOutput):
            raise TypeError(
                "Unified diffusion inference must return TextListOutput. "
                f"Received {type(response.output).__name__}."
            )
        finish_reasons = list(response.output.finish_reasons or [])
        generations = [
            DiffusionGeneration(
                input_text=prompt,
                output_text=text,
                finish_reason=finish_reasons[index] if index < len(finish_reasons) else None,
            )
            for index, (prompt, text) in enumerate(zip(prompts, response.output.texts))
        ]
        return DiffusionBatchResponse(
            generations=generations,
            latency_seconds=float(response.latency_ms or 0.0) / 1000.0,
            request_payload=dict(response.artifacts.get("request_payload", {})),
            raw_response=response.raw,
            model=response.model_id or None,
            telemetry=dict(response.telemetry),
        )


def build_raw_diffusion_backend(
    engine: str | EngineType,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 120.0,
    session: Optional[requests.Session] = None,
    surface: EngineSurface = EngineSurface.DIFFUSION_GENERATE,
    generate_path: Optional[str] = None,
    default_payload: Optional[Mapping[str, Any]] = None,
) -> DiffusionBackend:
    """Construct a direct HTTP diffusion backend from a simple engine name."""
    spec = resolve_engine_for_usage(
        engine,
        surface=surface,
        usage="diffusion backend construction",
    )
    engine_type = spec.engine
    resolved_base_url = base_url or resolve_engine_base_url(
        engine_type,
        surface=surface,
    )
    resolved_generate_path = generate_path or spec.diffusion_generate_path
    if not resolved_base_url:
        raise ValueError(
            f"Engine '{engine_type.value}' requires an explicit base_url for diffusion generation."
        )
    resolved_base_url = normalize_generate_base_url(
        resolved_base_url,
        generate_path=resolved_generate_path,
    )
    normalized = engine_type.value.replace("_", "-")
    if engine_type is EngineType.SGLANG:
        return SGLangDiffusionBackend(
            base_url=resolved_base_url,
            model=model,
            timeout=timeout,
            session=session,
            default_payload=default_payload,
            generate_path=resolved_generate_path,
        )
    if engine_type is EngineType.VLLM_OMNI:
        return VLLMOmniDiffusionBackend(
            base_url=resolved_base_url,
            model=model,
            timeout=timeout,
            session=session,
            default_payload=default_payload,
            generate_path=resolved_generate_path,
        )
    if engine_type is EngineType.CUSTOM_HTTP:
        return HTTPGenerateDiffusionBackend(
            base_url=resolved_base_url,
            backend_name=normalized,
            model=model,
            timeout=timeout,
            session=session,
            generate_path=resolved_generate_path,
            default_payload=default_payload,
        )
    raise ValueError(f"Engine '{engine_type.value}' does not expose a diffusion backend in this pass.")


def build_diffusion_backend(
    engine: str | EngineType,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 120.0,
    session: Optional[requests.Session] = None,
    surface: EngineSurface = EngineSurface.DIFFUSION_GENERATE,
    generate_path: Optional[str] = None,
    default_payload: Optional[Mapping[str, Any]] = None,
) -> DiffusionBackend:
    """Construct a diffusion backend from a simple engine name.

    This is the default entrypoint used by the diffusion prototype tests: it
    returns concrete backend adapters like ``SGLangDiffusionBackend``.

    For a backend built via the universal inference engine, use
    ``build_inference_diffusion_backend``.
    """
    return build_raw_diffusion_backend(
        engine,
        base_url=base_url,
        model=model,
        timeout=timeout,
        session=session,
        surface=surface,
        generate_path=generate_path,
        default_payload=default_payload,
    )


def build_inference_diffusion_backend(
    engine: str | EngineType,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 120.0,
    session: Optional[requests.Session] = None,
    surface: EngineSurface = EngineSurface.DIFFUSION_GENERATE,
    generate_path: Optional[str] = None,
    default_payload: Optional[Mapping[str, Any]] = None,
) -> DiffusionBackend:
    """Construct a diffusion backend via the universal inference engine."""
    from treepo._research.core.inference_engine import build_inference_engine

    inference_engine = build_inference_engine(
        engine,
        surface=surface,
        base_url=base_url,
        model=model or "default",
        timeout=timeout,
        session=session,
        generate_path=generate_path,
        default_payload=default_payload,
    )
    return InferenceDiffusionBackendAdapter(
        inference_engine,
        backend_name=EngineType.normalize(engine).value,
    )
