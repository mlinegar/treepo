"""Universal inference engine contract spanning chat, diffusion, and symbolic execution."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from treepo._research.core.async_utils import to_thread
from treepo._research.core.engines import (
    EngineSurface,
    EngineType,
    resolve_engine_base_url,
    resolve_engine_for_usage,
)
from treepo._research.core.llm_client import LLMClient, LLMConfig, MockLLMClient
from treepo._research.runtime.contracts import (
    ChatInput,
    DiffusionInput,
    EmbeddingInput,
    EmbeddingOutput,
    InferenceRequest,
    InferenceResponse,
    OperatorInput,
    OperatorOutput,
    StructuredOutput,
    SymbolicInput,
    TextListOutput,
    TextOutput,
)

logger = logging.getLogger(__name__)


class InferenceEngine(Protocol):
    """Common execution contract for chat, diffusion, and symbolic engines."""

    engine_type: EngineType
    surface: EngineSurface

    async def aexecute(self, request: InferenceRequest) -> InferenceResponse: ...

    async def asubmit(self, request: InferenceRequest) -> "InferenceHandle": ...

    async def aexecute_many(
        self, requests: Sequence[InferenceRequest]
    ) -> list[InferenceResponse]: ...

    def execute(self, request: InferenceRequest) -> InferenceResponse: ...

    async def aclose(self) -> None: ...


@dataclass
class InferenceHandle:
    """Async handle returned by unified engine submission."""

    request_id: str
    engine: EngineType
    surface: EngineSurface
    _future: asyncio.Future[InferenceResponse] | asyncio.Task[InferenceResponse]

    async def result(self, timeout: Optional[float] = None) -> InferenceResponse:
        if timeout is None:
            return await asyncio.shield(self._future)
        return await asyncio.wait_for(asyncio.shield(self._future), timeout=timeout)

    def done(self) -> bool:
        return bool(self._future.done())

    def cancel(self) -> bool:
        return bool(self._future.cancel())


class BaseInferenceEngine:
    """Convenience base class with async-first defaults."""

    engine_type: EngineType
    surface: EngineSurface

    async def aexecute(self, request: InferenceRequest) -> InferenceResponse:
        handle = await self.asubmit(request)
        return await handle.result()

    async def asubmit(self, request: InferenceRequest) -> InferenceHandle:
        request_id = request.resolved_request_id()
        task = asyncio.create_task(self.aexecute(request))
        return InferenceHandle(
            request_id=request_id,
            engine=self.engine_type,
            surface=self.surface,
            _future=task,
        )

    async def aexecute_many(self, requests: Sequence[InferenceRequest]) -> list[InferenceResponse]:
        handles = [await self.asubmit(request) for request in requests]
        return await asyncio.gather(*(handle.result() for handle in handles))

    def execute(self, request: InferenceRequest) -> InferenceResponse:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aexecute(request))
        raise RuntimeError(
            "InferenceEngine.execute() cannot be used from an active event loop; use aexecute()."
        )

    async def aclose(self) -> None:
        return None


class ChatCompatibleClient(Protocol):
    config: LLMConfig

    def chat(self, messages: list[dict[str, str]], **kwargs: Any): ...


class ChatInferenceEngine(BaseInferenceEngine):
    """Unified chat engine with sync and pooled-async execution paths."""

    def __init__(
        self,
        *,
        engine_type: EngineType,
        config: LLMConfig,
        mock: bool = False,
        enable_cache: bool = True,
        llm_client: Optional[ChatCompatibleClient] = None,
        async_batch_client: Optional[Any] = None,
        batch_client_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.engine_type = engine_type
        self.surface = EngineSurface.CHAT_OPENAI
        self.config = config
        self.mock = bool(mock)
        self._client = llm_client or (
            MockLLMClient(config) if self.mock else LLMClient(config, enable_cache=enable_cache)
        )
        self._async_batch_client = async_batch_client
        self._batch_client_factory = batch_client_factory
        self._batch_client_lock: Optional[asyncio.Lock] = None
        self._batch_client_loop: Optional[asyncio.AbstractEventLoop] = None
        self._batch_client_started = False

    async def _ensure_batch_client(self) -> Any:
        if self.mock:
            return None
        loop = asyncio.get_running_loop()
        if self._batch_client_loop is not None and self._batch_client_loop is not loop:
            # Runtime sync shims may call into the async engine through repeated
            # asyncio.run() calls. AsyncBatchLLMClient owns queues/background
            # tasks for the loop it started in, so discard it when a new loop is
            # used instead of reusing a stale worker.
            self._async_batch_client = None
            self._batch_client_lock = None
            self._batch_client_started = False
            self._batch_client_loop = None
        if self._batch_client_lock is None:
            self._batch_client_lock = asyncio.Lock()
        async with self._batch_client_lock:
            if self._async_batch_client is None:
                if self._batch_client_factory is not None:
                    self._async_batch_client = self._batch_client_factory()
                else:
                    from treepo._research.core.batch_processor import AsyncBatchLLMClient

                    model_name = None if self.config.model == "default" else self.config.model
                    self._async_batch_client = AsyncBatchLLMClient(
                        base_url=self.config.base_url,
                        model=model_name,
                        request_timeout=float(self.config.timeout),
                        api_key=self.config.api_key,
                    )
            if not self._batch_client_started:
                await self._async_batch_client.start()
                self._batch_client_started = True
                self._batch_client_loop = loop
            return self._async_batch_client

    def _validate_request(self, request: InferenceRequest) -> ChatInput:
        if request.surface is not EngineSurface.CHAT_OPENAI:
            raise ValueError(
                f"ChatInferenceEngine requires surface={EngineSurface.CHAT_OPENAI.value}, "
                f"received {request.surface.value}."
            )
        if not isinstance(request.input, ChatInput):
            raise TypeError(
                f"ChatInferenceEngine requires ChatInput, received {type(request.input).__name__}."
            )
        return request.input

    def _execute_sync(self, request: InferenceRequest) -> InferenceResponse:
        chat_input = self._validate_request(request)
        started = time.time()
        kwargs: Dict[str, Any] = {
            "max_tokens": int(chat_input.max_tokens),
            "temperature": float(chat_input.temperature),
        }
        if chat_input.stop:
            kwargs["stop"] = list(chat_input.stop)
        if chat_input.extra:
            kwargs.update(dict(chat_input.extra))
        response = self._client.chat(chat_input.messages, **kwargs)
        latency_ms = (time.time() - started) * 1000.0
        usage = {
            "prompt_tokens": int(getattr(response, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(response, "completion_tokens", 0) or 0),
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return InferenceResponse(
            surface=self.surface,
            engine=self.engine_type,
            model_id=str(getattr(response, "model", "") or self.config.model),
            output=TextOutput(text=str(getattr(response, "content", ""))),
            usage=usage,
            latency_ms=latency_ms,
            telemetry={"batched": False},
            request_id=request.request_id,
            raw=getattr(response, "raw_response", None),
        )

    async def asubmit(self, request: InferenceRequest) -> InferenceHandle:
        chat_input = self._validate_request(request)
        request_id = request.resolved_request_id()
        if self.mock:
            task = asyncio.create_task(to_thread(self._execute_sync, request))
            return InferenceHandle(
                request_id=request_id,
                engine=self.engine_type,
                surface=self.surface,
                _future=task,
            )

        batch_client = await self._ensure_batch_client()
        from treepo._research.core.batch_processor import BatchRequest

        batch_request = BatchRequest(
            request_id=request_id,
            messages=list(chat_input.messages),
            max_tokens=int(chat_input.max_tokens),
            temperature=float(chat_input.temperature),
            chat_template_kwargs=(
                dict(chat_input.extra.get("chat_template_kwargs", {}))
                if isinstance(chat_input.extra.get("chat_template_kwargs"), Mapping)
                else None
            ),
            extra_request_params={
                str(key): value
                for key, value in dict(chat_input.extra or {}).items()
                if key != "chat_template_kwargs"
            }
            or None,
            document_id=request.document_id or None,
            routing_key=request.routing_key or request.document_id or None,
            request_type=str(request.metadata.get("request_type", "chat")),
            priority=int(request.priority),
        )
        await batch_client.submit(batch_request)
        task = asyncio.create_task(self._await_batch_response(batch_client, batch_request, request))
        return InferenceHandle(
            request_id=request_id,
            engine=self.engine_type,
            surface=self.surface,
            _future=task,
        )

    async def _await_batch_response(
        self, batch_client: Any, batch_request: Any, request: InferenceRequest
    ) -> InferenceResponse:
        response = await batch_client.await_response(batch_request.request_id)
        if response.error:
            raise RuntimeError(
                f"Chat inference request {batch_request.request_id} failed on {self.engine_type.value}: {response.error}"
            )
        usage = dict(getattr(response, "usage", {}) or {})
        usage.setdefault(
            "total_tokens",
            int(usage.get("prompt_tokens", 0) or 0) + int(usage.get("completion_tokens", 0) or 0),
        )
        return InferenceResponse(
            surface=self.surface,
            engine=self.engine_type,
            model_id=str(getattr(batch_client, "model", "") or self.config.model),
            output=TextOutput(text=str(response.content)),
            usage=usage,
            latency_ms=float(getattr(response, "latency_ms", 0.0) or 0.0),
            telemetry={"batched": True},
            request_id=batch_request.request_id,
            raw=response,
        )

    async def aclose(self) -> None:
        if self._async_batch_client is not None and self._batch_client_started:
            await self._async_batch_client.stop()
            self._batch_client_started = False


class DiffusionInferenceEngine(BaseInferenceEngine):
    """Unified diffusion engine backed by the existing backend protocol."""

    def __init__(
        self,
        *,
        engine_type: EngineType,
        backend: Any,
        model_id: str = "",
    ) -> None:
        self.engine_type = engine_type
        self.surface = EngineSurface.DIFFUSION_GENERATE
        self.backend = backend
        self.model_id = model_id

    def _validate_request(self, request: InferenceRequest) -> DiffusionInput:
        if request.surface is not EngineSurface.DIFFUSION_GENERATE:
            raise ValueError(
                f"DiffusionInferenceEngine requires surface={EngineSurface.DIFFUSION_GENERATE.value}, "
                f"received {request.surface.value}."
            )
        if not isinstance(request.input, DiffusionInput):
            raise TypeError(
                f"DiffusionInferenceEngine requires DiffusionInput, received {type(request.input).__name__}."
            )
        return request.input

    def _execute_sync(self, request: InferenceRequest) -> InferenceResponse:
        diffusion_input = self._validate_request(request)
        batch = self.backend.generate(
            diffusion_input.texts,
            sampling_params=diffusion_input.sampling_params,
            engine_options=request.engine_options,
        )
        finish_reasons = [generation.finish_reason for generation in batch.generations]
        return InferenceResponse(
            surface=self.surface,
            engine=self.engine_type,
            model_id=str(batch.model or self.model_id or ""),
            output=TextListOutput(texts=list(batch.texts), finish_reasons=finish_reasons),
            usage={},
            latency_ms=float(batch.latency_seconds) * 1000.0,
            telemetry=dict(getattr(batch, "telemetry", {}) or {}),
            artifacts={"request_payload": dict(batch.request_payload)},
            request_id=request.request_id,
            raw=batch.raw_response,
        )

    async def aexecute(self, request: InferenceRequest) -> InferenceResponse:
        return await to_thread(self._execute_sync, request)


class _MockEmbeddingClient:
    def __init__(self, dim: int = 8) -> None:
        self.dim = int(max(2, dim))

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for idx, token in enumerate(str(text or "").lower().split()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                vec[int.from_bytes(digest[:8], "big") % self.dim] += 1.0 + float(idx % 5) * 0.01
            vectors.append(vec)
        return vectors


class EmbeddingInferenceEngine(BaseInferenceEngine):
    """Unified embedding engine backed by an OpenAI-compatible embeddings surface."""

    def __init__(
        self,
        *,
        engine_type: EngineType,
        base_url: str,
        model: str = "default",
        api_key: str = "EMPTY",
        timeout: float = 60.0,
        mock: bool = False,
        embedding_client: Optional[Any] = None,
    ) -> None:
        self.engine_type = engine_type
        self.surface = EngineSurface.EMBEDDING
        self.base_url = str(base_url or "").rstrip("/")
        self.model = str(model or "default")
        self.api_key = str(api_key or "EMPTY")
        self.timeout = float(timeout)
        if embedding_client is not None:
            self.client = embedding_client
        elif mock:
            self.client = _MockEmbeddingClient()
        else:
            from treepo._research.training.embedding_proxy import VLLMEmbeddingClient

            self.client = VLLMEmbeddingClient(
                api_base=self.base_url,
                model=None if self.model == "default" else self.model,
                api_key=self.api_key,
                timeout_seconds=self.timeout,
            )

    def _validate_request(self, request: InferenceRequest) -> EmbeddingInput:
        if request.surface is not EngineSurface.EMBEDDING:
            raise ValueError(
                f"EmbeddingInferenceEngine requires surface={EngineSurface.EMBEDDING.value}, "
                f"received {request.surface.value}."
            )
        if not isinstance(request.input, EmbeddingInput):
            raise TypeError(
                f"EmbeddingInferenceEngine requires EmbeddingInput, received {type(request.input).__name__}."
            )
        return request.input

    def _execute_sync(self, request: InferenceRequest) -> InferenceResponse:
        embedding_input = self._validate_request(request)
        started = time.time()
        vectors = self.client.embed_texts(list(embedding_input.texts))
        latency_ms = (time.time() - started) * 1000.0
        model_id = self.model
        if hasattr(self.client, "resolve_model"):
            try:
                model_id = str(self.client.resolve_model())
            except Exception:
                model_id = self.model
        return InferenceResponse(
            surface=self.surface,
            engine=self.engine_type,
            model_id=model_id,
            output=EmbeddingOutput(vectors=[list(map(float, vec)) for vec in vectors]),
            usage={"input_count": len(embedding_input.texts)},
            latency_ms=latency_ms,
            telemetry={"embedding_dim": len(vectors[0]) if vectors else 0},
            request_id=request.request_id,
            raw=None,
        )

    async def aexecute(self, request: InferenceRequest) -> InferenceResponse:
        return await to_thread(self._execute_sync, request)


@dataclass(frozen=True)
class NativeOperator:
    """One registered in-process runtime operator."""

    name: str
    handler: Callable[[OperatorInput], Any]
    description: str = ""


class NativeOperatorRegistry:
    """Registry for native Python/PyTorch runtime operators."""

    _operators: Dict[str, NativeOperator] = {}

    @classmethod
    def register(
        cls,
        name: str,
        handler: Callable[[OperatorInput], Any],
        *,
        description: str = "",
    ) -> None:
        cls._operators[str(name)] = NativeOperator(
            name=str(name),
            handler=handler,
            description=str(description),
        )

    @classmethod
    def resolve(cls, name: str) -> NativeOperator:
        try:
            return cls._operators[str(name)]
        except KeyError as exc:
            supported = ", ".join(sorted(cls._operators)) or "<none>"
            raise ValueError(
                f"Unknown native operator '{name}'. Registered operators: {supported}."
            ) from exc

    @classmethod
    def available(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._operators))

    @classmethod
    def clear(cls) -> None:
        cls._operators.clear()


def _coerce_operator_output(value: Any) -> OperatorOutput:
    if isinstance(value, OperatorOutput):
        return value
    if isinstance(value, Mapping):
        if "data" in value or "artifacts" in value:
            return OperatorOutput(
                data=value.get("data"),
                artifacts=dict(value.get("artifacts") or {}),
            )
    return OperatorOutput(data=value, artifacts={})


class OperatorInferenceEngine(BaseInferenceEngine):
    """Unified generic operator engine for native or served operator calls."""

    def __init__(
        self,
        *,
        engine_type: EngineType,
        model_id: str = "default",
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        operator_client: Optional[Any] = None,
    ) -> None:
        self.engine_type = engine_type
        self.surface = EngineSurface.OPERATOR
        self.model_id = str(model_id or "default")
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = str(api_key or "EMPTY")
        self.timeout = float(timeout)
        self.operator_client = operator_client

    def _validate_request(self, request: InferenceRequest) -> OperatorInput:
        if request.surface is not EngineSurface.OPERATOR:
            raise ValueError(
                f"OperatorInferenceEngine requires surface={EngineSurface.OPERATOR.value}, "
                f"received {request.surface.value}."
            )
        if not isinstance(request.input, OperatorInput):
            raise TypeError(
                f"OperatorInferenceEngine requires OperatorInput, received {type(request.input).__name__}."
            )
        return request.input

    def _execute_native(
        self, operator_input: OperatorInput
    ) -> tuple[OperatorOutput, Dict[str, Any]]:
        if self.operator_client is not None:
            if hasattr(self.operator_client, "execute"):
                value = self.operator_client.execute(operator_input)
            elif callable(self.operator_client):
                value = self.operator_client(operator_input)
            else:
                raise TypeError(
                    "Native operator backend must be callable or expose execute(operator_input)."
                )
        else:
            operator = NativeOperatorRegistry.resolve(operator_input.operation)
            value = operator.handler(operator_input)
        return _coerce_operator_output(value), {}

    def _execute_served(
        self, operator_input: OperatorInput, request: InferenceRequest
    ) -> tuple[OperatorOutput, Dict[str, Any]]:
        if self.operator_client is not None:
            if hasattr(self.operator_client, "execute"):
                value = self.operator_client.execute(operator_input)
            elif callable(self.operator_client):
                value = self.operator_client(operator_input)
            else:
                raise TypeError(
                    "Served operator client must be callable or expose execute(operator_input)."
                )
            return _coerce_operator_output(value), {}

        if not self.base_url:
            raise ValueError("Served operator execution requires base_url.")
        payload = {
            "model": self.model_id,
            "operation": operator_input.operation,
            "inputs": dict(operator_input.inputs),
            "batch": list(operator_input.batch),
            "options": dict(operator_input.options),
            "engine_options": dict(request.engine_options),
            "metadata": dict(request.metadata),
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = Request(
            self.base_url.rstrip("/") + "/operators/execute",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=max(1.0, self.timeout)) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Served operator request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, OSError, ValueError) as exc:
            raise RuntimeError(f"Served operator request failed: {exc}") from exc

        output = _coerce_operator_output(data)
        telemetry = dict(data.get("telemetry") or {}) if isinstance(data, Mapping) else {}
        if isinstance(data, Mapping) and data.get("model"):
            self.model_id = str(data.get("model"))
        return output, telemetry

    async def aexecute(self, request: InferenceRequest) -> InferenceResponse:
        operator_input = self._validate_request(request)
        started = time.time()

        def _run() -> tuple[OperatorOutput, Dict[str, Any]]:
            if self.engine_type is EngineType.NATIVE_OPERATOR:
                return self._execute_native(operator_input)
            return self._execute_served(operator_input, request)

        output, telemetry = await to_thread(_run)
        latency_ms = (time.time() - started) * 1000.0
        merged_artifacts = dict(output.artifacts)
        merged_artifacts.setdefault("operation", operator_input.operation)
        return InferenceResponse(
            surface=self.surface,
            engine=self.engine_type,
            model_id=self.model_id,
            output=OperatorOutput(data=output.data, artifacts=merged_artifacts),
            usage={
                "batch_count": len(operator_input.batch),
                "input_count": len(operator_input.inputs),
            },
            latency_ms=latency_ms,
            telemetry={"operation": operator_input.operation, **telemetry},
            artifacts=merged_artifacts,
            request_id=request.request_id,
            raw=output.data,
        )


@dataclass(frozen=True)
class SymbolicOperation:
    """One registered symbolic execution operation."""

    name: str
    handler: Callable[[Mapping[str, Any]], Any]
    description: str = ""


class SymbolicOperationRegistry:
    """Registry for namespaced symbolic-local operations."""

    _operations: Dict[str, SymbolicOperation] = {}

    @classmethod
    def register(
        cls,
        name: str,
        handler: Callable[[Mapping[str, Any]], Any],
        *,
        description: str = "",
    ) -> None:
        cls._operations[str(name)] = SymbolicOperation(
            name=str(name),
            handler=handler,
            description=str(description),
        )

    @classmethod
    def resolve(cls, name: str) -> SymbolicOperation:
        if not cls._operations:
            _register_builtin_symbolic_operations()
        try:
            return cls._operations[str(name)]
        except KeyError as exc:
            supported = ", ".join(sorted(cls._operations))
            raise ValueError(
                f"Unknown symbolic operation '{name}'. Registered operations: {supported}."
            ) from exc

    @classmethod
    def available(cls) -> tuple[str, ...]:
        if not cls._operations:
            _register_builtin_symbolic_operations()
        return tuple(sorted(cls._operations))


class SymbolicInferenceEngine(BaseInferenceEngine):
    """Unified symbolic-local engine for exact theorem-family execution."""

    def __init__(self, *, engine_type: EngineType = EngineType.SYMBOLIC_LOCAL) -> None:
        self.engine_type = engine_type
        self.surface = EngineSurface.SYMBOLIC_EXACT

    def _validate_request(self, request: InferenceRequest) -> SymbolicInput:
        if request.surface is not EngineSurface.SYMBOLIC_EXACT:
            raise ValueError(
                f"SymbolicInferenceEngine requires surface={EngineSurface.SYMBOLIC_EXACT.value}, "
                f"received {request.surface.value}."
            )
        if not isinstance(request.input, SymbolicInput):
            raise TypeError(
                f"SymbolicInferenceEngine requires SymbolicInput, received {type(request.input).__name__}."
            )
        return request.input

    async def aexecute(self, request: InferenceRequest) -> InferenceResponse:
        symbolic_input = self._validate_request(request)
        operation = SymbolicOperationRegistry.resolve(symbolic_input.operation)
        started = time.time()

        def _run() -> Any:
            payload = operation.handler(dict(symbolic_input.inputs))
            if inspect.isawaitable(payload):
                raise TypeError(
                    f"Symbolic operation '{operation.name}' must return a concrete result, not an awaitable."
                )
            return payload

        data = await to_thread(_run)
        latency_ms = (time.time() - started) * 1000.0
        return InferenceResponse(
            surface=self.surface,
            engine=self.engine_type,
            model_id=operation.name,
            output=StructuredOutput(
                data=data,
                schema_name=operation.name,
                artifacts={"operation": operation.name},
            ),
            usage={},
            latency_ms=latency_ms,
            telemetry={"operation": operation.name},
            request_id=request.request_id,
            raw=data,
        )


def _markov_exact_operation(inputs: Mapping[str, Any]) -> Any:
    from treepo._research.diffusion.markov_toy import run_markov_toy_experiment

    payload = run_markov_toy_experiment(
        list(inputs.get("states") or inputs.get("path") or []),
        chunk_size=int(inputs.get("chunk_size", 1)),
        rounds=int(inputs.get("rounds", 1)),
        eps_leaf=float(inputs.get("eps_leaf", 0.0)),
        eps_merge=float(inputs.get("eps_merge", 0.0)),
        eps_idemp=float(inputs.get("eps_idemp", 0.0)),
    )
    payload["selected_operation"] = "markov.fixed_binary_exact"
    return payload


def _markov_count_only_operation(inputs: Mapping[str, Any]) -> Any:
    payload = _markov_exact_operation(inputs)
    payload["selected_operation"] = "markov.fixed_binary_count_only"
    payload["selected_view"] = {
        "count_only_root_state": payload["count_only_root_state"],
        "count_only_full_path_value": payload["count_only_full_path_value"],
        "count_only_matches_full_path": payload["count_only_matches_full_path"],
        "count_only_gap": payload["count_only_gap"],
        "count_only_schedule": payload["count_only_schedule"],
    }
    return payload


def _register_builtin_symbolic_operations() -> None:
    if SymbolicOperationRegistry._operations:
        return
    SymbolicOperationRegistry.register(
        "markov.fixed_binary_exact",
        _markov_exact_operation,
        description="Exact fixed-binary Markov sketch checkpoint evaluation.",
    )
    SymbolicOperationRegistry.register(
        "markov.fixed_binary_count_only",
        _markov_count_only_operation,
        description="Count-only fixed-binary Markov counterexample baseline.",
    )


def build_inference_engine(
    engine: str | EngineType,
    *,
    surface: EngineSurface,
    model: str = "default",
    host: str = "localhost",
    port: Optional[int] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 120.0,
    mock: bool = False,
    enable_cache: bool = True,
    llm_config: Optional[LLMConfig] = None,
    llm_client: Optional[ChatCompatibleClient] = None,
    async_batch_client: Optional[Any] = None,
    batch_client_factory: Optional[Callable[[], Any]] = None,
    backend: Optional[Any] = None,
    session: Optional[Any] = None,
    generate_path: Optional[str] = None,
    default_payload: Optional[Mapping[str, Any]] = None,
    require_managed: bool = False,
) -> InferenceEngine:
    """Build a universal inference engine for the requested engine/surface pair."""

    allowed_by_surface = {
        EngineSurface.CHAT_OPENAI: (
            EngineType.VLLM,
            EngineType.SGLANG,
            EngineType.OPENAI,
            EngineType.CUSTOM_HTTP,
        ),
        EngineSurface.DIFFUSION_GENERATE: (
            EngineType.SGLANG,
            EngineType.VLLM_OMNI,
            EngineType.CUSTOM_HTTP,
        ),
        EngineSurface.EMBEDDING: (
            EngineType.VLLM,
            EngineType.OPENAI,
            EngineType.CUSTOM_HTTP,
        ),
        EngineSurface.OPERATOR: (
            EngineType.NATIVE_OPERATOR,
            EngineType.CUSTOM_HTTP,
        ),
        EngineSurface.SYMBOLIC_EXACT: (EngineType.SYMBOLIC_LOCAL,),
    }
    spec = resolve_engine_for_usage(
        engine,
        surface=surface,
        usage="universal inference engine construction",
        require_managed=require_managed,
        allowed_engines=allowed_by_surface.get(surface),
    )
    engine_type = spec.engine

    if surface is EngineSurface.CHAT_OPENAI:
        config = llm_config or LLMConfig.from_engine(
            engine_type,
            model=model,
            host=host,
            port=port,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        return ChatInferenceEngine(
            engine_type=engine_type,
            config=config,
            mock=mock,
            enable_cache=enable_cache,
            llm_client=llm_client,
            async_batch_client=async_batch_client,
            batch_client_factory=batch_client_factory,
        )

    if surface is EngineSurface.DIFFUSION_GENERATE:
        resolved_base_url = base_url or resolve_engine_base_url(
            engine_type,
            surface=surface,
            host=host,
            port=port,
        )
        if backend is None:
            from treepo._research.diffusion.backends import build_raw_diffusion_backend

            backend = build_raw_diffusion_backend(
                engine_type,
                base_url=resolved_base_url,
                model=model if model != "default" else None,
                timeout=timeout,
                session=session,
                surface=surface,
                generate_path=generate_path,
                default_payload=default_payload,
            )
        return DiffusionInferenceEngine(
            engine_type=engine_type,
            backend=backend,
            model_id=model if model != "default" else "",
        )

    if surface is EngineSurface.EMBEDDING:
        resolved_base_url = base_url or resolve_engine_base_url(
            engine_type,
            surface=surface,
            host=host,
            port=port,
        )
        if resolved_base_url is None:
            raise ValueError(
                f"Could not resolve embedding endpoint for engine '{engine_type.value}'."
            )
        return EmbeddingInferenceEngine(
            engine_type=engine_type,
            base_url=resolved_base_url,
            model=model,
            api_key=api_key or "EMPTY",
            timeout=timeout,
            mock=mock,
            embedding_client=backend,
        )

    if surface is EngineSurface.OPERATOR:
        resolved_base_url = base_url or resolve_engine_base_url(
            engine_type,
            surface=surface,
            host=host,
            port=port,
        )
        return OperatorInferenceEngine(
            engine_type=engine_type,
            model_id=model,
            base_url=resolved_base_url,
            api_key=api_key or "EMPTY",
            timeout=timeout,
            operator_client=backend,
        )

    if surface is EngineSurface.SYMBOLIC_EXACT:
        return SymbolicInferenceEngine(engine_type=engine_type)

    raise ValueError(f"Unsupported inference surface '{surface.value}'.")


__all__ = [
    "InferenceEngine",
    "InferenceHandle",
    "ChatInferenceEngine",
    "DiffusionInferenceEngine",
    "EmbeddingInferenceEngine",
    "NativeOperator",
    "NativeOperatorRegistry",
    "OperatorInferenceEngine",
    "SymbolicInferenceEngine",
    "SymbolicOperation",
    "SymbolicOperationRegistry",
    "build_inference_engine",
]
