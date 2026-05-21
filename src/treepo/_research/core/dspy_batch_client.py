"""DSPy LM transport backed by ThinkingTrees batch clients.

DSPy programs still own prompt construction and output parsing. This module
only swaps the LM transport so concurrent DSPy calls flow through
``AsyncBatchLLMClient`` / ``MultiServerBatchClient`` before reaching vLLM or
SGLang.
"""

from __future__ import annotations

import asyncio
import atexit
import copy
import logging
import threading
import time
import uuid
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Sequence

import dspy
from litellm import ModelResponse
from litellm.types.utils import Choices, Message, Usage

from treepo._research.core.batch_transport import (
    DEFAULT_BATCH_MAX_CONCURRENT,
    DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_BATCH_ROUTING_POLICY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_TIMEOUT_SECONDS,
    normalize_base_urls,
)
from treepo._research.core.batch_processor import (
    AsyncBatchLLMClient,
    BatchRequest,
    BatchResponse,
    MultiServerBatchClient,
    parse_routing_policy,
)

logger = logging.getLogger(__name__)

_OPENAI_CHAT_REQUEST_KEYS = {
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "min_p",
    "n",
    "parallel_tool_calls",
    "presence_penalty",
    "repetition_penalty",
    "response_format",
    "seed",
    "stop",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "user",
}


def _strip_litellm_openai_prefix(model: str) -> str:
    rendered = str(model or "").strip()
    if rendered.startswith("openai/"):
        return rendered[len("openai/") :]
    return rendered


def _response_to_litellm_model_response(
    *,
    response: BatchResponse,
    model: str,
) -> ModelResponse:
    usage = response.usage or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    return ModelResponse(
        id=f"chatcmpl-batch-{response.request_id}",
        created=int(time.time()),
        model=model,
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content=response.content or "", role="assistant"),
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
        response_ms=float(response.latency_ms or 0.0),
    )


class _BatchClientBridge:
    """Background event-loop owner for async batch clients used by sync DSPy calls."""

    def __init__(
        self,
        *,
        base_urls: Sequence[str],
        model: Optional[str],
        api_key: str,
        max_concurrent: int,
        batch_size: int,
        batch_timeout: float,
        request_timeout: float,
        await_response_timeout: Optional[float],
        routing_policy: str,
    ) -> None:
        self.base_urls = normalize_base_urls(api_bases=base_urls)
        if not self.base_urls:
            raise ValueError("Batched DSPy LM requires at least one base URL")
        self.model = _strip_litellm_openai_prefix(model or "")
        self.api_key = str(api_key or "EMPTY")
        self.max_concurrent = max(1, int(max_concurrent))
        self.batch_size = max(1, int(batch_size))
        self.batch_timeout = max(0.0, float(batch_timeout))
        self.request_timeout = max(1.0, float(request_timeout))
        self.await_response_timeout = (
            None if await_response_timeout is None else max(1.0, float(await_response_timeout))
        )
        self.routing_policy = parse_routing_policy(routing_policy).value

        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._client: Optional[AsyncBatchLLMClient | MultiServerBatchClient] = None
        self._startup_error: Optional[BaseException] = None
        self._closed = False

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Batched DSPy LM bridge is closed")
            if self._thread is not None and self._thread.is_alive():
                return
            self._ready.clear()
            self._startup_error = None
            self._thread = threading.Thread(
                target=self._thread_main,
                name="thinkingtrees_dspy_batch_client",
                daemon=True,
            )
            self._thread.start()

        self._ready.wait(timeout=30.0)
        if self._startup_error is not None:
            raise RuntimeError("Failed to start batched DSPy LM bridge") from self._startup_error
        if self._loop is None:
            raise RuntimeError("Timed out starting batched DSPy LM bridge")

    def close(self) -> None:
        with self._lock:
            self._closed = True
            loop = self._loop
            thread = self._thread
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)

    def request(
        self,
        *,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        chat_template_kwargs: Optional[Dict[str, Any]],
        extra_request_params: Optional[Dict[str, Any]],
        timeout: Optional[float],
    ) -> BatchResponse:
        future = self.submit(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            chat_template_kwargs=chat_template_kwargs,
            extra_request_params=extra_request_params,
        )
        return future.result(timeout=timeout)

    def submit(
        self,
        *,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        chat_template_kwargs: Optional[Dict[str, Any]],
        extra_request_params: Optional[Dict[str, Any]],
    ) -> Future[BatchResponse]:
        self.start()
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(
            self._request_async(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                chat_template_kwargs=chat_template_kwargs,
                extra_request_params=extra_request_params,
            ),
            self._loop,
        )

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._start_client())
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            loop.close()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                loop.run_until_complete(self._stop_client())
            finally:
                loop.close()
                self._loop = None

    async def _start_client(self) -> None:
        if len(self.base_urls) > 1:
            self._client = MultiServerBatchClient(
                servers=self.base_urls,
                max_concurrent_per_server=self.max_concurrent,
                batch_size=self.batch_size,
                batch_timeout=self.batch_timeout,
                request_timeout=self.request_timeout,
                api_key=self.api_key,
                routing_policy=self.routing_policy,
            )
        else:
            self._client = AsyncBatchLLMClient(
                base_url=self.base_urls[0],
                max_concurrent=self.max_concurrent,
                batch_size=self.batch_size,
                batch_timeout=self.batch_timeout,
                model=self.model or None,
                request_timeout=self.request_timeout,
                api_key=self.api_key,
            )
        await self._client.start()

    async def _stop_client(self) -> None:
        if self._client is not None:
            await self._client.stop()
            self._client = None

    async def _request_async(
        self,
        *,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        chat_template_kwargs: Optional[Dict[str, Any]],
        extra_request_params: Optional[Dict[str, Any]],
    ) -> BatchResponse:
        if self._client is None:
            raise RuntimeError("Batched DSPy LM bridge client is not started")
        request_id = f"dspy_batch_{uuid.uuid4().hex}"
        request = BatchRequest(
            request_id=request_id,
            messages=messages,
            max_tokens=max(1, int(max_tokens)),
            temperature=float(temperature),
            chat_template_kwargs=chat_template_kwargs,
            extra_request_params=extra_request_params,
            request_type="dspy",
        )
        await self._client.submit(request)
        response = await self._client.await_response(
            request_id,
            timeout=self.await_response_timeout,
        )
        if response.error:
            raise RuntimeError(str(response.error))
        return response


class BatchedDSPyLM(dspy.LM):
    """DSPy LM that routes chat completions through ``AsyncBatchLLMClient``.

    The class keeps DSPy's public LM behavior, including ``copy`` for
    temperature/max-token overrides, while sharing one background batch client
    across copies. It currently targets chat-style local OpenAI-compatible
    endpoints, which is the vLLM/SGLang path used by the C-TreePO examples.
    """

    def __init__(
        self,
        *,
        model: str,
        api_base: Optional[str] = None,
        api_bases: Optional[Sequence[str]] = None,
        api_key: str = "EMPTY",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        cache: bool = True,
        max_concurrent: int = DEFAULT_BATCH_MAX_CONCURRENT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_timeout: float = DEFAULT_BATCH_TIMEOUT_SECONDS,
        request_timeout: float = DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS,
        await_response_timeout: Optional[float] = None,
        routing_policy: str = DEFAULT_BATCH_ROUTING_POLICY,
        bridge: Optional[_BatchClientBridge] = None,
        **kwargs: Any,
    ) -> None:
        bases = normalize_base_urls(api_base=api_base, api_bases=api_bases)
        if not bases:
            raise ValueError("BatchedDSPyLM requires api_base or api_bases")
        base0 = str(bases[0])
        model_for_dspy = str(model)
        model_for_batch = _strip_litellm_openai_prefix(model_for_dspy)
        super().__init__(
            model=model_for_dspy,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            api_base=base0,
            api_key=api_key,
            **kwargs,
        )
        self._batch_api_bases = [str(base).rstrip("/") for base in bases]
        self._batch_api_key = str(api_key or "EMPTY")
        self._batch_model = model_for_batch
        self._batch_max_concurrent = max(1, int(max_concurrent))
        self._batch_size = max(1, int(batch_size))
        self._batch_timeout = max(0.0, float(batch_timeout))
        self._batch_request_timeout = max(1.0, float(request_timeout))
        self._batch_await_response_timeout = (
            None if await_response_timeout is None else max(1.0, float(await_response_timeout))
        )
        self._batch_routing_policy = parse_routing_policy(routing_policy).value
        self._batch_bridge = bridge or _BatchClientBridge(
            base_urls=self._batch_api_bases,
            model=model_for_batch,
            api_key=self._batch_api_key,
            max_concurrent=self._batch_max_concurrent,
            batch_size=self._batch_size,
            batch_timeout=self._batch_timeout,
            request_timeout=self._batch_request_timeout,
            await_response_timeout=self._batch_await_response_timeout,
            routing_policy=self._batch_routing_policy,
        )
        atexit.register(self.close)

    def close(self) -> None:
        bridge = getattr(self, "_batch_bridge", None)
        if bridge is not None:
            bridge.close()

    def copy(self, **kwargs: Any) -> "BatchedDSPyLM":
        new_instance = copy.copy(self)
        new_instance.history = []
        new_instance.kwargs = copy.deepcopy(self.kwargs)

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(new_instance, key, value)
            if (key in self.kwargs) or (not hasattr(self, key)):
                if value is None:
                    new_instance.kwargs.pop(key, None)
                else:
                    new_instance.kwargs[key] = value
        if hasattr(new_instance, "_warned_zero_temp_rollout"):
            new_instance._warned_zero_temp_rollout = False
        return new_instance

    def __deepcopy__(self, memo: Dict[int, Any]) -> "BatchedDSPyLM":
        copied = self.copy()
        memo[id(self)] = copied
        return copied

    def forward(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
        request_kwargs = self._prepare_batch_request(prompt=prompt, messages=messages, kwargs=kwargs)
        response = self._batch_bridge.request(**request_kwargs)
        results = _response_to_litellm_model_response(response=response, model=self.model)
        self._check_truncation(results)
        self._record_usage(results)
        return results

    async def aforward(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
        request_kwargs = self._prepare_batch_request(prompt=prompt, messages=messages, kwargs=kwargs)
        future = self._batch_bridge.submit(
            messages=request_kwargs["messages"],
            max_tokens=request_kwargs["max_tokens"],
            temperature=request_kwargs["temperature"],
            chat_template_kwargs=request_kwargs["chat_template_kwargs"],
            extra_request_params=request_kwargs["extra_request_params"],
        )
        response = await asyncio.wrap_future(future)
        results = _response_to_litellm_model_response(response=response, model=self.model)
        self._check_truncation(results)
        self._record_usage(results)
        return results

    def _prepare_batch_request(
        self,
        *,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, str]]],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        call_kwargs.pop("cache", None)
        call_kwargs.pop("rollout_id", None)
        prepared_messages = messages or [{"role": "user", "content": prompt}]
        merged_kwargs = {**self.kwargs, **call_kwargs}
        temperature = float(merged_kwargs.get("temperature", 0.0) or 0.0)
        max_tokens = int(merged_kwargs.get("max_tokens", 256) or 256)

        chat_template_kwargs = merged_kwargs.pop("chat_template_kwargs", None)
        extra_request_params: Dict[str, Any] = {}
        extra_body = merged_kwargs.pop("extra_body", None)
        if chat_template_kwargs is None and isinstance(extra_body, dict):
            maybe_chat_template = extra_body.get("chat_template_kwargs")
            if isinstance(maybe_chat_template, dict):
                chat_template_kwargs = dict(maybe_chat_template)
        if isinstance(extra_body, dict):
            extra_body_params = dict(extra_body)
            extra_body_params.pop("chat_template_kwargs", None)
            extra_request_params.update(extra_body_params)
        for key in list(merged_kwargs):
            if key in _OPENAI_CHAT_REQUEST_KEYS:
                extra_request_params[key] = merged_kwargs.pop(key)

        request_kwargs = {
            "messages": prepared_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": chat_template_kwargs,
            "extra_request_params": extra_request_params or None,
            "timeout": self._batch_await_response_timeout or (self._batch_request_timeout + 30.0),
        }
        return request_kwargs

    def _record_usage(self, results: ModelResponse) -> None:
        try:
            if dspy.settings.usage_tracker and hasattr(results, "usage"):
                dspy.settings.usage_tracker.add_usage(self.model, dict(results.usage))
        except Exception:
            logger.debug("Failed to record DSPy batched LM usage", exc_info=True)
