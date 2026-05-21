"""Shared local OpenAI-compatible inference configuration.

This module is the small contract layer used by scripts that talk to local
vLLM/SGLang-style chat endpoints. It centralizes endpoint resolution and the
batching knobs that must stay aligned between raw batched pipelines and DSPy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from treepo._research.config.settings import get_inference_backend_config, load_settings
from treepo._research.core.engines import EngineType, LocalChatEndpoints, resolve_local_chat_endpoints


@dataclass(frozen=True)
class LocalInferenceConfig:
    """Resolved local inference contract for one OpenAI-compatible engine."""

    engine: str
    endpoints: LocalChatEndpoints
    routing_policy: str
    host: Optional[str] = None
    model: Optional[str] = None
    max_concurrent_requests: int = 200
    batch_size: int = 50
    batch_timeout: float = 0.02
    request_timeout_seconds: Optional[float] = None
    await_response_timeout_seconds: Optional[float] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    @property
    def primary_port(self) -> int:
        return self.endpoints.primary_port

    @property
    def ports(self) -> tuple[int, ...]:
        return self.endpoints.ports

    @property
    def base_urls(self) -> tuple[str, ...]:
        return self.endpoints.base_urls

    def dspy_kwargs(
        self,
        *,
        cache: bool = True,
        settings: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Keyword args for ``create_local_engine_lm(...)``."""
        kwargs: dict[str, Any] = {
            "engine": self.engine,
            "endpoints": self.endpoints,
            "cache": bool(cache),
            "batch_max_concurrent": int(self.max_concurrent_requests),
            "batch_size": int(self.batch_size),
            "batch_timeout": float(self.batch_timeout),
            "batch_await_response_timeout": self.await_response_timeout_seconds,
            "routing_policy": self.routing_policy,
        }
        if settings is not None:
            kwargs["settings"] = settings
        if self.model is not None:
            kwargs["model"] = self.model
        if self.temperature is not None:
            kwargs["temperature"] = float(self.temperature)
        if self.max_tokens is not None:
            kwargs["max_tokens"] = int(self.max_tokens)
        if self.request_timeout_seconds is not None:
            kwargs["batch_request_timeout"] = float(self.request_timeout_seconds)
        return kwargs

    def pipeline_kwargs(self, *, max_concurrent_documents: Optional[int] = None) -> dict[str, Any]:
        """Keyword args for ``BatchedPipelineConfig(...)``."""
        kwargs: dict[str, Any] = {
            "task_model_endpoints": self.endpoints,
            "routing_policy": self.routing_policy,
            "max_concurrent_requests": int(self.max_concurrent_requests),
            "batch_size": int(self.batch_size),
            "batch_timeout": float(self.batch_timeout),
            "request_timeout_seconds": self.request_timeout_seconds,
            "await_response_timeout_seconds": self.await_response_timeout_seconds,
        }
        if max_concurrent_documents is not None:
            kwargs["max_concurrent_documents"] = int(max_concurrent_documents)
        return kwargs

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe run metadata."""
        return {
            "engine": self.engine,
            "host": self.host,
            "model": self.model,
            "port": self.primary_port,
            "ports": list(self.ports),
            "base_urls": list(self.base_urls),
            "routing_policy": self.routing_policy,
            "concurrent_requests": int(self.max_concurrent_requests),
            "batch_size": int(self.batch_size),
            "batch_timeout": float(self.batch_timeout),
            "request_timeout_seconds": self.request_timeout_seconds,
            "await_response_timeout_seconds": self.await_response_timeout_seconds,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


def add_local_inference_args(
    parser: argparse.ArgumentParser,
    *,
    include_generation: bool = False,
    default_engine: Optional[str] = None,
    default_host: Optional[str] = None,
    default_port: Optional[int] = None,
    default_model: Optional[str] = None,
    default_concurrent_requests: int = 200,
    default_batch_size: int = 50,
    default_batch_timeout: float = 0.02,
    default_request_timeout_seconds: Optional[float] = None,
    default_await_response_timeout_seconds: Optional[float] = None,
    default_temperature: Optional[float] = None,
    default_max_tokens: Optional[int] = None,
) -> None:
    """Add the common local inference flags to a script parser."""
    group = parser.add_argument_group("local inference")
    group.add_argument("--engine", default=default_engine, help="Local inference engine (default: settings task_backend)")
    group.add_argument("--host", default=default_host, help="Optional local server host override")
    group.add_argument("--port", type=int, default=default_port, help="Task model port")
    group.add_argument(
        "--ports",
        type=int,
        nargs="+",
        default=None,
        help="Optional task model ports for load balancing (overrides --port)",
    )
    group.add_argument("--model", default=default_model, help="Optional served model name")
    group.add_argument(
        "--concurrent-requests",
        type=int,
        default=default_concurrent_requests,
        help="Maximum concurrent local inference requests",
    )
    group.add_argument("--batch-size", type=int, default=default_batch_size, help="Client request batch size")
    group.add_argument("--batch-timeout", type=float, default=default_batch_timeout, help="Client batch fill timeout")
    group.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=default_request_timeout_seconds,
        help="Per-request HTTP timeout for local inference",
    )
    group.add_argument(
        "--await-response-timeout-seconds",
        type=float,
        default=default_await_response_timeout_seconds,
        help="Timeout while awaiting batched local inference responses",
    )
    if include_generation:
        group.add_argument(
            "--temperature",
            type=float,
            default=default_temperature,
            help="Generation temperature for local DSPy/module calls",
        )
        group.add_argument(
            "--max-tokens",
            type=int,
            default=default_max_tokens,
            help="Generation max_tokens for local DSPy/module calls",
        )


def resolve_local_inference_config(
    args: argparse.Namespace | Mapping[str, Any],
    *,
    settings: Optional[Mapping[str, Any]] = None,
    usage: str = "local inference",
    role: str = "task",
    filter_unreachable: bool = False,
    endpoint_ready: Optional[Callable[[str], bool]] = None,
) -> LocalInferenceConfig:
    """Resolve CLI/settings into one local inference contract."""
    if settings is None:
        settings = load_settings()
    backend_cfg = get_inference_backend_config(dict(settings or {}))

    engine = _get_arg(args, "engine", None) or backend_cfg.get("task_backend") or "vllm"
    routing_policy = str(backend_cfg.get("routing_policy") or "affinity_load_aware")
    endpoints = resolve_local_chat_endpoints(
        engine,
        port=_get_arg(args, "port", None),
        ports=_get_arg(args, "ports", None),
        settings=settings,
        host=_get_arg(args, "host", None),
        role=role,
        usage=usage,
        allowed_engines=(EngineType.VLLM, EngineType.SGLANG),
        filter_unreachable=filter_unreachable,
        endpoint_ready=endpoint_ready,
    )
    return LocalInferenceConfig(
        engine=endpoints.engine.value,
        endpoints=endpoints,
        routing_policy=routing_policy,
        host=_get_arg(args, "host", None),
        model=_get_arg(args, "model", None),
        max_concurrent_requests=max(1, int(_get_arg(args, "concurrent_requests", 200) or 200)),
        batch_size=max(1, int(_get_arg(args, "batch_size", 50) or 50)),
        batch_timeout=max(0.0, float(_get_arg(args, "batch_timeout", 0.02) or 0.02)),
        request_timeout_seconds=_optional_float(_get_arg(args, "request_timeout_seconds", None)),
        await_response_timeout_seconds=_optional_float(_get_arg(args, "await_response_timeout_seconds", None)),
        temperature=_optional_float(_get_arg(args, "temperature", None)),
        max_tokens=_optional_int(_get_arg(args, "max_tokens", None)),
    )


def _get_arg(args: argparse.Namespace | Mapping[str, Any], name: str, default: Any) -> Any:
    if isinstance(args, Mapping):
        return args.get(name, default)
    return getattr(args, name, default)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


__all__ = [
    "LocalInferenceConfig",
    "add_local_inference_args",
    "resolve_local_inference_config",
]
