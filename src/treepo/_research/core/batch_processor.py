"""
Batched Request Processing for vLLM.

This module implements high-throughput batched processing that:
1. Pools requests from multiple documents/trees
2. Sends concurrent batches to vLLM (leveraging its internal batching)
3. Routes responses back to waiting coroutines

The key insight: while we can't parallelize tree levels (children before parents),
we CAN parallelize across multiple documents AND pool requests from the same
level across many trees.

Architecture:
                   ┌─────────────────────┐
                   │  AsyncBatchLLMClient │
                   │  (request pooling)  │
                   └─────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   Batch Workers     │
                   │  (N concurrent)     │
                   └─────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   vLLM Server       │
                   └─────────────────────┘
"""

import asyncio
import hashlib
import logging
import math
import os
import re as _re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Set, Union

import aiohttp

from treepo._research.config.constants import LOG_TRUNCATE_LENGTH
from treepo._research.core.async_utils import cancel_tasks, to_thread
from treepo._research.core.batch_transport import (
    DEFAULT_BATCH_MAX_CONCURRENT,
    DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_TIMEOUT_SECONDS,
)

if TYPE_CHECKING:
    from treepo._research.core.vllm_metrics import VLLMMetricsCollector

logger = logging.getLogger(__name__)

_PORT_RE = _re.compile(r":(\d+)")


def _extract_port(url: str) -> Optional[int]:
    """Extract port number from a base URL like ``http://localhost:8000/v1``."""
    m = _PORT_RE.search(url)
    return int(m.group(1)) if m else None


class RoutingPolicy(str, Enum):
    """Explicit multi-server routing policies."""

    ROUND_ROBIN = "round_robin"
    DOCUMENT_AFFINITY = "document_affinity"
    AFFINITY_LOAD_AWARE = "affinity_load_aware"


def parse_routing_policy(value: Optional[str]) -> RoutingPolicy:
    """Parse routing policy strings with a safe default."""
    rendered = str(value or "").strip().lower()
    valid = {
        RoutingPolicy.ROUND_ROBIN.value,
        RoutingPolicy.DOCUMENT_AFFINITY.value,
        RoutingPolicy.AFFINITY_LOAD_AWARE.value,
    }
    if rendered in valid:
        return RoutingPolicy(rendered)
    return RoutingPolicy.AFFINITY_LOAD_AWARE


# =============================================================================
# Request/Response Types
# =============================================================================

@dataclass
class BatchRequest:
    """A single LLM request in the batch pool."""
    request_id: str
    messages: List[Dict[str, str]]
    max_tokens: int = 8192
    temperature: float = 0.7
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    extra_request_params: Optional[Dict[str, Any]] = None

    # Tracking
    document_id: Optional[str] = None
    routing_key: Optional[str] = None
    request_type: str = "summarize"  # summarize, audit, score
    priority: int = 0  # Higher = more urgent
    call_metadata: Optional[Dict[str, Any]] = None

    # Response handling
    future: Optional[asyncio.Future] = None
    submitted_at: Optional[float] = None
    cache_key: Optional[str] = None


@dataclass
class BatchResponse:
    """Response from vLLM."""
    request_id: str
    content: str
    usage: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: float = 0.0
    call_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchStats:
    """Statistics for batch processing."""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_ms: float = 0.0
    batches_sent: int = 0
    wall_clock_start: float = 0.0
    wall_clock_end: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.completed_requests == 0:
            return 0.0
        return self.total_latency_ms / self.completed_requests

    @property
    def requests_per_second(self) -> float:
        if self.total_latency_ms == 0:
            return 0.0
        return self.completed_requests / (self.total_latency_ms / 1000)

    @property
    def wall_clock_seconds(self) -> float:
        """Wall clock time in seconds. Uses current time if not yet stopped."""
        if self.wall_clock_start == 0:
            return 0.0
        # If not stopped yet, use current time for live updates
        end_time = self.wall_clock_end if self.wall_clock_end > 0 else time.time()
        return end_time - self.wall_clock_start

    @property
    def tokens_per_second(self) -> float:
        """Total tokens per second (wall clock time)."""
        if self.wall_clock_seconds <= 0:
            return 0.0
        return self.total_tokens / self.wall_clock_seconds

    @property
    def read_tokens_per_second(self) -> float:
        """Prompt/input tokens per second."""
        if self.wall_clock_seconds <= 0:
            return 0.0
        return self.prompt_tokens / self.wall_clock_seconds

    @property
    def write_tokens_per_second(self) -> float:
        """Completion/output tokens per second."""
        if self.wall_clock_seconds <= 0:
            return 0.0
        return self.completion_tokens / self.wall_clock_seconds

    def __str__(self) -> str:
        return (
            f"BatchStats(reqs={self.completed_requests}/{self.total_requests}, "
            f"tokens={self.total_tokens:,}, "
            f"tok/s={self.tokens_per_second:.0f} "
            f"[r:{self.read_tokens_per_second:.0f}, w:{self.write_tokens_per_second:.0f}])"
        )


# =============================================================================
# Async Batch Client
# =============================================================================

class AsyncBatchLLMClient:
    """
    Async client for batched LLM requests.

    Pools requests and sends them concurrently to vLLM, which handles
    internal batching for optimal GPU utilization.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        max_concurrent: int = DEFAULT_BATCH_MAX_CONCURRENT,  # Max concurrent requests to vLLM
        batch_size: int = DEFAULT_BATCH_SIZE,       # Requests per batch
        batch_timeout: float = DEFAULT_BATCH_TIMEOUT_SECONDS,  # Max wait to fill batch (seconds)
        model: str = None,  # Auto-detect from server if None
        request_timeout: float = DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS,  # Per-request timeout
        api_key: str = "EMPTY",  # vLLM/SGLang use "EMPTY"; set real key for OpenAI
        recover_base_url_callback: Optional[Callable[[str], bool]] = None,
        recovery_cooldown_seconds: float = 120.0,
        call_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ):
        """
        Initialize async batch client.

        Args:
            base_url: vLLM server URL
            max_concurrent: Maximum concurrent HTTP requests
            batch_size: Target batch size before sending
            batch_timeout: Max time to wait for batch to fill
            model: Model name for vLLM (auto-detected if None)
            request_timeout: Per-request HTTP timeout in seconds
            api_key: API key for Authorization header (default "EMPTY" for local servers)
            recover_base_url_callback: Optional callback(base_url)->bool to auto-recover
                failed servers (e.g. orchestrator restart/wake).
            recovery_cooldown_seconds: Cooldown between recovery attempts per base_url.
        """
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self._model = model  # Will be set during start() if None
        self.request_timeout = request_timeout
        self.api_key = api_key
        self.recover_base_url_callback = recover_base_url_callback
        self.recovery_cooldown_seconds = max(0.0, float(recovery_cooldown_seconds))
        self.call_sink = call_sink
        self._last_recovery_attempt: float = 0.0
        self._recovery_lock: Optional[asyncio.Lock] = None

        # Optional disk-backed response cache (opt-in via env vars).
        # This is a pragmatic analogue of "persistent KV" for repeated
        # document reruns: skip identical LLM calls entirely.
        #
        # Env vars:
        #   TT_RESPONSE_CACHE_DIR=/path/to/cache
        #   TT_RESPONSE_CACHE_MODE=off|read|write|readwrite   (default: off)
        #   TT_RESPONSE_CACHE_REQUEST_TYPES=summarize,merge   (optional filter)
        self._response_cache = None
        self._response_cache_mode = "off"
        self._response_cache_request_types: Optional[Set[str]] = None
        cache_dir = str(os.getenv("TT_RESPONSE_CACHE_DIR", "") or "").strip()
        if cache_dir:
            mode = str(os.getenv("TT_RESPONSE_CACHE_MODE", "") or "").strip().lower()
            if mode in {"off", "read", "write", "readwrite"}:
                self._response_cache_mode = mode
            elif mode:
                self._response_cache_mode = "readwrite"
            else:
                self._response_cache_mode = "readwrite"
            try:
                from pathlib import Path

                from treepo._research.core.response_cache import FileResponseCache

                self._response_cache = FileResponseCache(Path(cache_dir))
            except Exception:
                self._response_cache = None
                self._response_cache_mode = "off"

        raw_types = str(os.getenv("TT_RESPONSE_CACHE_REQUEST_TYPES", "") or "").strip()
        if raw_types:
            selected = {part.strip() for part in raw_types.split(",") if part.strip()}
            self._response_cache_request_types = selected or None

        # Request pool
        self._request_queue: asyncio.Queue[BatchRequest] = None
        self._pending_futures: Dict[str, asyncio.Future] = {}
        self._inflight_request_tasks: Dict[str, asyncio.Task] = {}

        # Concurrency control
        self._semaphore: asyncio.Semaphore = None
        self._session: aiohttp.ClientSession = None

        # Statistics
        self.stats = BatchStats()
        self._recovery_attempts: int = 0
        self._recovery_successes: int = 0
        self._recovery_failures: int = 0
        self._recovery_skipped_cooldown: int = 0
        self._retry_attempts: int = 0
        self._retry_after_recovery: int = 0
        self._error_status_counts: Dict[str, int] = {}
        self._error_type_counts: Dict[str, int] = {}

        # State
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._active_batch_tasks: Set[asyncio.Task] = set()
        self._max_inflight_batches = max(1, math.ceil(self.max_concurrent / max(1, self.batch_size)))

    @property
    def model(self) -> str:
        """Get model name (auto-detected if not set)."""
        return self._model or "unknown"

    @property
    def pending_count(self) -> int:
        """Number of in-flight requests awaiting responses."""
        return len(self._pending_futures)

    def _response_cache_allows(self, request: BatchRequest) -> bool:
        if self._response_cache is None or self._response_cache_mode == "off":
            return False
        if self._response_cache_request_types is None:
            return True
        return str(request.request_type) in self._response_cache_request_types

    async def _detect_model(self) -> str:
        """Auto-detect model name from vLLM server."""
        from treepo._research.core.model_detection import detect_model_async
        return await detect_model_async(self.base_url, fallback="default")

    def _handle_request_error(
        self,
        request: BatchRequest,
        error_msg: str,
    ) -> None:
        """Handle request errors consistently.

        Args:
            request: The failed request
            error_msg: Error message to include in the response
        """
        logger.error(f"Request {request.request_id} failed: {error_msg}")
        self.stats.failed_requests += 1
        if request.future and not request.future.done():
            response = BatchResponse(
                request_id=request.request_id,
                content="",
                error=error_msg,
                call_metadata=dict(request.call_metadata or {}),
            )
            self._emit_call_trace(request, response)
            request.future.set_result(response)

    def _emit_call_trace(self, request: BatchRequest, response: BatchResponse) -> None:
        if self.call_sink is None:
            return
        try:
            from treepo._research.experiments.call_tracing import batch_request_call_row

            self.call_sink(
                batch_request_call_row(
                    request,
                    response,
                    base_url=str(self.base_url),
                    model=str(self.model),
                )
            )
        except Exception:
            logger.debug("Failed to emit batch call trace for %s", request.request_id, exc_info=True)

    async def start(self):
        """Start the batch processor."""
        if self._running:
            return

        # Auto-detect model if not specified
        if self._model is None:
            self._model = await self._detect_model()

        self._request_queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._recovery_lock = asyncio.Lock()
        # Set connector limit to match max_concurrent (default aiohttp limit is 100)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        # Set timeout for all requests
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        self._running = True

        # Track start time
        self.stats.wall_clock_start = time.time()

        # Start batch worker
        self._worker_task = asyncio.create_task(self._batch_worker())
        logger.debug(f"Batch client started (max_concurrent={self.max_concurrent}, model={self._model})")

    async def stop(self):
        """Stop the batch processor."""
        self._running = False
        self.stats.wall_clock_end = time.time()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        # Clean up any pending futures to prevent memory leaks
        # This handles cases where submit() was called but await_response() was not
        if self._pending_futures:
            orphaned = len(self._pending_futures)
            for request_id, future in list(self._pending_futures.items()):
                if not future.done():
                    future.set_result(BatchResponse(
                        request_id=request_id,
                        content="",
                        error="Batch client stopped",
                    ))
            self._pending_futures.clear()
            if orphaned > 0:
                logger.debug(f"Cleaned up {orphaned} orphaned futures on stop")

        if self._active_batch_tasks:
            await cancel_tasks(self._active_batch_tasks)
            self._active_batch_tasks.clear()
        if self._request_queue:
            while not self._request_queue.empty():
                try:
                    self._request_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        if self._session:
            await self._session.close()
        logger.info(f"Batch client stopped. Stats: {self.stats}")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    @property
    def diagnostics(self) -> Dict[str, Any]:
        request_types = (
            sorted(self._response_cache_request_types)
            if self._response_cache_request_types is not None
            else None
        )
        return {
            "base_url": str(self.base_url),
            "model": str(self.model),
            "pending_count": int(self.pending_count),
            "inflight_task_count": int(len(self._inflight_request_tasks)),
            "response_cache": {
                "enabled": bool(self._response_cache is not None and self._response_cache_mode != "off"),
                "mode": str(self._response_cache_mode),
                "request_types": request_types,
            },
            "recovery": {
                "attempts": int(self._recovery_attempts),
                "successes": int(self._recovery_successes),
                "failures": int(self._recovery_failures),
                "skipped_cooldown": int(self._recovery_skipped_cooldown),
                "retry_attempts": int(self._retry_attempts),
                "retry_after_recovery": int(self._retry_after_recovery),
            },
            "errors": {
                "status_counts": dict(self._error_status_counts),
                "type_counts": dict(self._error_type_counts),
            },
        }

    async def submit(self, request: BatchRequest) -> str:
        """
        Submit a request to the pool.

        Returns immediately with a request_id. Use await_response() to get result.
        """
        if not self._running:
            raise RuntimeError("Batch client not started")

        # Create future for response
        request.future = asyncio.get_running_loop().create_future()
        request.submitted_at = time.time()
        self._pending_futures[request.request_id] = request.future
        self.stats.total_requests += 1

        # Optional disk cache short-circuit.
        if self._response_cache_allows(request):
            try:
                from treepo._research.core.response_cache import make_chat_cache_key

                request.cache_key = make_chat_cache_key(
                    model=self.model,
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    extra={
                        "request_type": request.request_type,
                        "chat_template_kwargs": request.chat_template_kwargs,
                        "extra_request_params": request.extra_request_params,
                    },
                )
            except Exception:
                request.cache_key = None

            if request.cache_key and self._response_cache_mode in {"read", "readwrite"}:
                cached = self._response_cache.get(request.cache_key)
                if cached is not None:
                    self.stats.cache_hits += 1
                    response = BatchResponse(
                        request_id=request.request_id,
                        content=cached.content,
                        usage=cached.usage,
                        error=None,
                        latency_ms=0.0,
                        call_metadata=dict(request.call_metadata or {}),
                    )
                    self.stats.completed_requests += 1
                    self.stats.total_tokens += int(cached.usage.get("total_tokens", 0) or 0)
                    self.stats.prompt_tokens += int(cached.usage.get("prompt_tokens", 0) or 0)
                    self.stats.completion_tokens += int(cached.usage.get("completion_tokens", 0) or 0)
                    self._emit_call_trace(request, response)
                    if request.future and not request.future.done():
                        request.future.set_result(response)
                    return request.request_id
                self.stats.cache_misses += 1

        # Add to queue
        await self._request_queue.put(request)

        return request.request_id

    async def await_response(
        self,
        request_id: str,
        timeout: float = 600.0,  # 10 minutes default (increased for large queues)
    ) -> BatchResponse:
        """
        Wait for a submitted request to complete.

        Args:
            request_id: The request ID to wait for
            timeout: Maximum wait time in seconds (default 10 minutes)

        Returns:
            BatchResponse with the result
        """
        if request_id not in self._pending_futures:
            raise KeyError(f"Unknown request_id: {request_id}")

        future = self._pending_futures[request_id]

        try:
            # Shield future so timeout does not cancel the producer-side future.
            response = await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
            del self._pending_futures[request_id]
            return response
        except asyncio.TimeoutError:
            inflight_task = self._inflight_request_tasks.pop(request_id, None)
            cancelled_inflight = False
            if inflight_task is not None and not inflight_task.done():
                inflight_task.cancel()
                cancelled_inflight = True
            logger.error(
                "Request %s timed out after %.0fs (%d still pending, cancelled_inflight=%s)",
                request_id,
                timeout,
                len(self._pending_futures),
                cancelled_inflight,
            )
            response = BatchResponse(
                request_id=request_id,
                content="",
                error=f"Timeout after {timeout}s",
            )
            if request_id in self._pending_futures:
                del self._pending_futures[request_id]
            if not future.done():
                future.set_result(response)
            return response

    async def call(self, request: BatchRequest) -> BatchResponse:
        """Submit and await in one call (convenience method)."""
        await self.submit(request)
        return await self.await_response(request.request_id)

    async def _batch_worker(self):
        """Background worker that collects and sends batches."""
        while self._running:
            try:
                batch = []
                deadline = time.time() + self.batch_timeout

                # Collect requests until batch_size or timeout
                while len(batch) < self.batch_size:
                    timeout = max(0.001, deadline - time.time())
                    try:
                        request = await asyncio.wait_for(
                            self._request_queue.get(),
                            timeout=timeout
                        )
                        batch.append(request)
                    except asyncio.TimeoutError:
                        break

                if batch:
                    # Throttle number of in-flight batches to avoid task buildup
                    if len(self._active_batch_tasks) >= self._max_inflight_batches:
                        done, _ = await asyncio.wait(
                            self._active_batch_tasks,
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in done:
                            if task.exception():
                                logger.debug(f"Batch task error: {task.exception()}")

                    task = asyncio.create_task(self._send_batch(batch))
                    self._active_batch_tasks.add(task)
                    task.add_done_callback(self._active_batch_tasks.discard)
                    self.stats.batches_sent += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch worker error: {e}")
                await asyncio.sleep(0.1)

    async def _send_batch(self, batch: List[BatchRequest]):
        """Send a batch of requests concurrently."""
        tasks = [self._send_single(req) for req in batch]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_single(self, request: BatchRequest):
        """Send a single request with semaphore control."""
        request_task = asyncio.current_task()
        if request_task is not None:
            self._inflight_request_tasks[request.request_id] = request_task
        try:
            async with self._semaphore:
                start_time = time.time()
                payload = {
                    "model": self._model,
                    "messages": request.messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                }
                if request.extra_request_params:
                    for key, value in dict(request.extra_request_params).items():
                        if value is None or key in {"model", "messages"}:
                            continue
                        payload[str(key)] = value
                if request.chat_template_kwargs:
                    payload["chat_template_kwargs"] = dict(request.chat_template_kwargs)

                max_attempts = 2  # Initial try + one retry after recovery.
                for attempt in range(max_attempts):
                    try:
                        async with self._session.post(
                            f"{self.base_url}/chat/completions",
                            json=payload,
                            headers={"Authorization": f"Bearer {self.api_key}"}
                        ) as resp:
                            latency = (time.time() - start_time) * 1000
                            if resp.status == 200:
                                data = await resp.json()
                                content = data["choices"][0]["message"]["content"]
                                usage = data.get("usage", {})

                                response = BatchResponse(
                                    request_id=request.request_id,
                                    content=content,
                                    usage=usage,
                                    latency_ms=latency,
                                    call_metadata=dict(request.call_metadata or {}),
                                )
                                self.stats.completed_requests += 1
                                self.stats.total_latency_ms += latency
                                self.stats.total_tokens += usage.get("total_tokens", 0)
                                self.stats.prompt_tokens += usage.get("prompt_tokens", 0)
                                self.stats.completion_tokens += usage.get("completion_tokens", 0)
                                if (
                                    self._response_cache is not None
                                    and self._response_cache_mode in {"write", "readwrite"}
                                    and self._response_cache_allows(request)
                                ):
                                    try:
                                        from treepo._research.core.response_cache import (
                                            CachedChatResponse,
                                            FileResponseCache,
                                        )

                                        cache_key = request.cache_key
                                        if not cache_key:
                                            from treepo._research.core.response_cache import make_chat_cache_key

                                            cache_key = make_chat_cache_key(
                                                model=self.model,
                                                messages=request.messages,
                                                max_tokens=request.max_tokens,
                                                temperature=request.temperature,
                                                extra={
                                                    "request_type": request.request_type,
                                                    "chat_template_kwargs": request.chat_template_kwargs,
                                                    "extra_request_params": request.extra_request_params,
                                                },
                                            )
                                        if cache_key:
                                            self._response_cache.set(
                                                cache_key,
                                                CachedChatResponse(
                                                    content=content,
                                                    usage={str(k): int(v) for k, v in dict(usage or {}).items()},
                                                    model=str(data.get("model") or self.model),
                                                    created_at=FileResponseCache.now_iso(),
                                                ),
                                            )
                                            self.stats.cache_writes += 1
                                    except Exception:
                                        pass
                                self._emit_call_trace(request, response)
                                if request.future and not request.future.done():
                                    request.future.set_result(response)
                                return

                            # Non-200 response.
                            body_text = await resp.text()
                            body_text_lower = str(body_text).lower()
                            if (
                                resp.status == 400
                                and "chat_template_kwargs" in payload
                                and (
                                    "chat_template_kwargs" in body_text_lower
                                    or "enable_thinking" in body_text_lower
                                )
                            ):
                                payload.pop("chat_template_kwargs", None)
                                logger.warning(
                                    "Server %s rejected chat_template_kwargs; retrying request %s without thinking-control payload",
                                    self.base_url,
                                    request.request_id,
                                )
                                continue
                            error_msg = f"HTTP {resp.status}: {body_text[:LOG_TRUNCATE_LENGTH]}"
                            status_key = str(resp.status)
                            self._error_status_counts[status_key] = self._error_status_counts.get(status_key, 0) + 1
                            recoverable_status = resp.status in {408, 429, 500, 502, 503, 504}
                            if attempt < (max_attempts - 1) and recoverable_status:
                                self._retry_attempts += 1
                                recovered = await self._maybe_recover_server(
                                    reason=f"http_{resp.status}"
                                )
                                if recovered:
                                    self._retry_after_recovery += 1
                                    logger.warning(
                                        "Recovered %s after %s; retrying request %s",
                                        self.base_url,
                                        error_msg.split(":", 1)[0],
                                        request.request_id,
                                    )
                                    continue

                            response = BatchResponse(
                                request_id=request.request_id,
                                content="",
                                error=error_msg,
                                latency_ms=latency,
                                call_metadata=dict(request.call_metadata or {}),
                            )
                            self.stats.failed_requests += 1
                            self._emit_call_trace(request, response)
                            if request.future and not request.future.done():
                                request.future.set_result(response)
                            return

                    except aiohttp.ClientError as e:
                        err_key = type(e).__name__
                        self._error_type_counts[err_key] = self._error_type_counts.get(err_key, 0) + 1
                        if attempt < (max_attempts - 1):
                            self._retry_attempts += 1
                            recovered = await self._maybe_recover_server(reason=type(e).__name__)
                            if recovered:
                                self._retry_after_recovery += 1
                                logger.warning(
                                    "Recovered %s after %s; retrying request %s",
                                    self.base_url,
                                    type(e).__name__,
                                    request.request_id,
                                )
                                continue
                        self._handle_request_error(
                            request,
                            f"{type(e).__name__}: {str(e) or 'Connection failed'}",
                        )
                        return
                    except asyncio.TimeoutError:
                        self._error_type_counts["TimeoutError"] = self._error_type_counts.get("TimeoutError", 0) + 1
                        if attempt < (max_attempts - 1):
                            self._retry_attempts += 1
                            recovered = await self._maybe_recover_server(reason="timeout")
                            if recovered:
                                self._retry_after_recovery += 1
                                logger.warning(
                                    "Recovered %s after timeout; retrying request %s",
                                    self.base_url,
                                    request.request_id,
                                )
                                continue
                        self._handle_request_error(request, "Request timed out")
                        return
                    except Exception as e:
                        err_key = type(e).__name__
                        self._error_type_counts[err_key] = self._error_type_counts.get(err_key, 0) + 1
                        self._handle_request_error(
                            request,
                            f"{type(e).__name__}: {str(e) or 'Unknown error'}",
                        )
                        return
        except asyncio.CancelledError:
            self._error_type_counts["CancelledError"] = self._error_type_counts.get("CancelledError", 0) + 1
            if request.future and not request.future.done():
                self._handle_request_error(request, "Request cancelled")
            return
        finally:
            self._inflight_request_tasks.pop(request.request_id, None)

    async def _maybe_recover_server(self, *, reason: str) -> bool:
        """Run recovery callback at most once per cooldown window."""
        if self.recover_base_url_callback is None:
            return False

        now = time.monotonic()
        if self._recovery_lock is None:
            self._recovery_lock = asyncio.Lock()
        async with self._recovery_lock:
            if (now - self._last_recovery_attempt) < self.recovery_cooldown_seconds:
                self._recovery_skipped_cooldown += 1
                return False
            self._last_recovery_attempt = now

        self._recovery_attempts += 1
        try:
            logger.warning(
                "Attempting batch-client server recovery for %s (%s)",
                self.base_url,
                reason,
            )
            recovered = await to_thread(self.recover_base_url_callback, self.base_url)
        except Exception as exc:
            self._recovery_failures += 1
            logger.warning("Batch-client server recovery callback failed for %s: %s", self.base_url, exc)
            return False

        if recovered:
            self._recovery_successes += 1
            logger.info("Batch-client server recovery succeeded for %s", self.base_url)
            return True
        self._recovery_failures += 1
        logger.warning("Batch-client server recovery reported failure for %s", self.base_url)
        return False


# =============================================================================
# Multi-Server Load Balancer
# =============================================================================

class MultiServerBatchClient:
    """
    Load balances requests across multiple vLLM/SGLang servers.

    Uses explicit routing policies:
    - round_robin
    - document_affinity
    - affinity_load_aware

    The default policy (`affinity_load_aware`) preserves current behavior:
    stable affinity by document/request key with load-based spillover.
    Aggregates stats from all underlying clients.
    """

    def __init__(
        self,
        servers: List[str],  # List of base URLs, e.g., ["http://localhost:8000/v1", "http://localhost:8002/v1"]
        max_concurrent_per_server: int = DEFAULT_BATCH_MAX_CONCURRENT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_timeout: float = DEFAULT_BATCH_TIMEOUT_SECONDS,
        request_timeout: float = DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS,
        api_key: str = "EMPTY",
        recover_base_url_callback: Optional[Callable[[str], bool]] = None,
        recovery_cooldown_seconds: float = 120.0,
        metrics_collector: Optional["VLLMMetricsCollector"] = None,
        routing_policy: Union[RoutingPolicy, str] = RoutingPolicy.AFFINITY_LOAD_AWARE,
        load_imbalance_threshold: float = 2.0,
        min_pending_before_spillover: int = 50,
        call_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ):
        """
        Initialize multi-server client.

        Args:
            servers: List of vLLM server URLs
            max_concurrent_per_server: Max concurrent requests per server
            batch_size: Requests per batch
            batch_timeout: Max wait to fill batch
            request_timeout: Per-request HTTP timeout in seconds
            api_key: API key for Authorization header
            metrics_collector: Optional VLLMMetricsCollector for load-aware routing
        """
        self.servers = servers
        self.clients: List[AsyncBatchLLMClient] = []
        self._counter = 0  # Round-robin counter
        self._lock: Optional[asyncio.Lock] = None  # Created in start()
        self._request_client_map: Dict[str, AsyncBatchLLMClient] = {}  # request_id -> client (O(1) lookup)
        self._metrics_collector = metrics_collector
        self.routing_policy = (
            routing_policy
            if isinstance(routing_policy, RoutingPolicy)
            else parse_routing_policy(str(routing_policy))
        )
        self.load_imbalance_threshold = max(1.1, float(load_imbalance_threshold))
        self.min_pending_before_spillover = max(1, int(min_pending_before_spillover))
        self._routing_counts: Dict[str, int] = {}
        self._routing_policy_counts: Dict[str, int] = {
            RoutingPolicy.ROUND_ROBIN.value: 0,
            RoutingPolicy.DOCUMENT_AFFINITY.value: 0,
            RoutingPolicy.AFFINITY_LOAD_AWARE.value: 0,
        }
        self._affinity_total: int = 0
        self._affinity_without_key: int = 0
        self._affinity_spillovers: int = 0

        # Create a client for each server
        for server_url in servers:
            client = AsyncBatchLLMClient(
                base_url=server_url,
                max_concurrent=max_concurrent_per_server,
                batch_size=batch_size,
                batch_timeout=batch_timeout,
                request_timeout=request_timeout,
                api_key=api_key,
                recover_base_url_callback=recover_base_url_callback,
                recovery_cooldown_seconds=recovery_cooldown_seconds,
                call_sink=call_sink,
            )
            self.clients.append(client)
            self._routing_counts[str(server_url)] = 0

    @property
    def stats(self) -> BatchStats:
        """Aggregate stats from all clients."""
        combined = BatchStats()
        for client in self.clients:
            combined.total_requests += client.stats.total_requests
            combined.completed_requests += client.stats.completed_requests
            combined.failed_requests += client.stats.failed_requests
            combined.cache_hits += client.stats.cache_hits
            combined.cache_misses += client.stats.cache_misses
            combined.cache_writes += client.stats.cache_writes
            combined.total_tokens += client.stats.total_tokens
            combined.prompt_tokens += client.stats.prompt_tokens
            combined.completion_tokens += client.stats.completion_tokens
            combined.total_latency_ms += client.stats.total_latency_ms
            combined.batches_sent += client.stats.batches_sent
        starts = [c.stats.wall_clock_start for c in self.clients if c.stats.wall_clock_start > 0]
        if starts:
            combined.wall_clock_start = min(starts)
        ends = [c.stats.wall_clock_end for c in self.clients if c.stats.wall_clock_end > 0]
        if ends:
            combined.wall_clock_end = max(ends)
        return combined

    @property
    def pending_depths(self) -> Dict[str, int]:
        """Current in-flight request counts by server URL."""
        return {
            str(client.base_url): int(client.pending_count)
            for client in self.clients
        }

    async def start(self):
        """Start all underlying clients."""
        self._lock = asyncio.Lock()
        await asyncio.gather(*[c.start() for c in self.clients])
        models = [c.model for c in self.clients]
        logger.info(f"Multi-server client started with {len(self.clients)} servers: {models}")

    async def stop(self):
        """Stop all underlying clients."""
        await asyncio.gather(*[c.stop() for c in self.clients])
        logger.info(f"Multi-server client stopped. Combined stats: {self.stats}")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    def _round_robin_client(self) -> AsyncBatchLLMClient:
        client = self.clients[self._counter % len(self.clients)]
        self._counter += 1
        return client

    @staticmethod
    def _affinity_key_for_request(request: BatchRequest) -> Optional[str]:
        if request.routing_key:
            return str(request.routing_key)
        if request.document_id:
            return str(request.document_id)
        return None

    @staticmethod
    def _stable_shard_index(key: str, n: int) -> int:
        if n <= 0:
            raise ValueError(f"Invalid shard count: {n}")
        digest = hashlib.sha256(str(key).encode("utf-8", "surrogatepass")).digest()
        value = int.from_bytes(digest[:8], "big", signed=False)
        return int(value % n)

    def _client_load(self, client: AsyncBatchLLMClient) -> int:
        """Best-effort load proxy for routing.

        Prefers backend queue depth from the metrics collector when available,
        otherwise falls back to local HTTP in-flight futures.
        """
        if self._metrics_collector is not None:
            port = _extract_port(client.base_url)
            if port is not None:
                metrics = self._metrics_collector.get(port)
                if metrics.reachable:
                    return int(metrics.num_requests_waiting)
        return int(client.pending_count)

    def _affinity_client(self, request: BatchRequest, *, load_aware: bool) -> AsyncBatchLLMClient:
        if len(self.clients) <= 1:
            return self.clients[0]

        affinity_key = self._affinity_key_for_request(request)
        if affinity_key is None:
            self._affinity_without_key += 1
            return self._round_robin_client()

        self._affinity_total += 1
        preferred_idx = self._stable_shard_index(affinity_key, len(self.clients))
        preferred = self.clients[preferred_idx]
        if not load_aware:
            return preferred

        loads = [self._client_load(c) for c in self.clients]
        min_pending = min(loads) if loads else 0
        preferred_load = self._client_load(preferred)
        if (
            preferred_load >= self.min_pending_before_spillover
            and preferred_load > (min_pending * self.load_imbalance_threshold)
        ):
            self._affinity_spillovers += 1
            return self._least_loaded_client()
        return preferred

    def _get_client_for_request(self, request: BatchRequest) -> AsyncBatchLLMClient:
        """Route request using the configured routing policy."""
        policy_name = self.routing_policy.value
        self._routing_policy_counts[policy_name] = self._routing_policy_counts.get(policy_name, 0) + 1
        if self.routing_policy == RoutingPolicy.ROUND_ROBIN:
            client = self._round_robin_client()
        elif self.routing_policy == RoutingPolicy.DOCUMENT_AFFINITY:
            client = self._affinity_client(request, load_aware=False)
        elif self.routing_policy == RoutingPolicy.AFFINITY_LOAD_AWARE:
            client = self._affinity_client(request, load_aware=True)
        else:
            client = self._round_robin_client()

        base_url = str(client.base_url)
        self._routing_counts[base_url] = self._routing_counts.get(base_url, 0) + 1
        return client

    def _least_loaded_client(self) -> AsyncBatchLLMClient:
        """Return the least-loaded server.

        Uses real vLLM queue depth from metrics collector when available,
        falling back to local pending request count.
        """
        return min(self.clients, key=self._client_load)

    async def submit(self, request: BatchRequest) -> str:
        """Submit request to server selected by document-affinity routing."""
        client = self._get_client_for_request(request)
        request_id = await client.submit(request)
        # Store mapping for O(1) lookup in await_response
        self._request_client_map[request_id] = client
        return request_id

    async def await_response(
        self,
        request_id: str,
        timeout: float = 600.0,
    ) -> BatchResponse:
        """Wait for response using O(1) client lookup.

        Args:
            request_id: Request identifier returned by ``submit``.
            timeout: Maximum wait time in seconds.
        """
        # Direct lookup using stored mapping
        client = self._request_client_map.get(request_id)
        if client is None:
            raise KeyError(f"Unknown request_id: {request_id}")

        try:
            response = await client.await_response(request_id, timeout=timeout)
            return response
        finally:
            # Clean up mapping after response resolution (success/timeout/error).
            self._request_client_map.pop(request_id, None)

    async def call(self, request: BatchRequest) -> BatchResponse:
        """Submit and await in one call (no mapping needed, direct to client)."""
        client = self._get_client_for_request(request)
        # call() handles submit+await internally on the same client
        return await client.call(request)

    @property
    def routing_stats(self) -> Dict[str, Any]:
        return {
            "policy": self.routing_policy.value,
            "by_server": dict(self._routing_counts),
            "policy_counts": dict(self._routing_policy_counts),
            "affinity_total": int(self._affinity_total),
            "affinity_without_key": int(self._affinity_without_key),
            "affinity_spillovers": int(self._affinity_spillovers),
        }

    @property
    def diagnostics(self) -> Dict[str, Any]:
        return {
            "routing": self.routing_stats,
            "pending_depths": self.pending_depths,
            "servers": [c.diagnostics for c in self.clients],
        }


# =============================================================================
# Multi-Document Batch Orchestrator
# =============================================================================

# =============================================================================
# Batch Audit Checks
# =============================================================================

async def audit_nodes_batched(
    nodes: List[Dict[str, Any]],
    oracle_prompt_fn: Callable[[str, str, str], List[Dict[str, str]]],
    client: AsyncBatchLLMClient,
    rubric: str,
    document_id: str,
) -> List[Dict[str, Any]]:
    """
    Audit multiple nodes with batched oracle calls.

    Args:
        nodes: Nodes to audit
        oracle_prompt_fn: Function(original, summary, rubric) -> messages
        client: Batch LLM client
        rubric: Audit rubric
        document_id: Document identifier

    Returns:
        List of audit results
    """
    # Create requests for all nodes
    requests = []
    for i, node in enumerate(nodes):
        original = node.get("content") or ""
        summary = node.get("summary") or ""

        messages = oracle_prompt_fn(original, summary, rubric)

        request = BatchRequest(
            request_id=f"{document_id}_audit_{i}",
            messages=messages,
            document_id=document_id,
            request_type="audit",
        )
        requests.append((request, node))

    # Submit all
    for request, _ in requests:
        await client.submit(request)

    # Await all
    results = []
    for request, node in requests:
        response = await client.await_response(request.request_id)
        results.append({
            "node_id": node["id"],
            "passed": "pass" in response.content.lower() if response.content else False,
            "response": response.content,
            "error": response.error,
        })

    return results




# =============================================================================
# Convenience Functions
# =============================================================================

def run_batched(coro):
    """Run an async coroutine from sync code.

    Note: Prefer using asyncio.run() directly in new code.
    This function exists for backwards compatibility.
    """
    return asyncio.run(coro)
