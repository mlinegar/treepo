"""
Backend-neutral inference metrics collector.

Polls `/metrics` endpoints from vLLM and SGLang servers and exposes a unified
snapshot used by routing, scheduling, and benchmark telemetry.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

_QUEUE_WAITING_PATTERNS = [
    re.compile(r"^vllm:num_requests_waiting(?:\{.*?\})?\s+([\d.eE+-]+)", re.MULTILINE),
    re.compile(r"^sglang:num_requests_waiting(?:\{.*?\})?\s+([\d.eE+-]+)", re.MULTILINE),
]
_QUEUE_RUNNING_PATTERNS = [
    re.compile(r"^vllm:num_requests_running(?:\{.*?\})?\s+([\d.eE+-]+)", re.MULTILINE),
    re.compile(r"^sglang:num_requests_running(?:\{.*?\})?\s+([\d.eE+-]+)", re.MULTILINE),
]
_KV_USAGE_PATTERNS = [
    re.compile(r"^vllm:gpu_cache_usage_perc(?:\{.*?\})?\s+([\d.eE+-]+)", re.MULTILINE),
    re.compile(r"^sglang:kv_cache_usage(?:\{.*?\})?\s+([\d.eE+-]+)", re.MULTILINE),
]
_PREFIX_CACHE_HIT_PATTERNS = [
    re.compile(r"^vllm:prefix_cache_hit_rate(?:\{.*?\})?\s+([\d.eE+-]+)", re.MULTILINE),
    re.compile(r"^sglang:cache_hit_rate(?:\{.*?\})?\s+([\d.eE+-]+)", re.MULTILINE),
]


def _first_float(text: str, patterns: List[re.Pattern[str]], default: float = 0.0) -> float:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                continue
    return float(default)


@dataclass
class InferenceServerMetrics:
    server_url: str
    queue_waiting: int = 0
    queue_running: int = 0
    kv_cache_usage_pct: float = 0.0
    prefix_cache_hit_rate: float = 0.0
    reachable: bool = False
    timestamp: float = 0.0

    @property
    def queue_depth(self) -> int:
        return int(self.queue_waiting) + int(self.queue_running)


@dataclass
class InferenceMetricsSnapshot:
    servers: Dict[str, InferenceServerMetrics] = field(default_factory=dict)
    poll_count: int = 0
    last_poll: float = 0.0

    @property
    def avg_prefix_cache_hit_rate(self) -> float:
        reachable = [m for m in self.servers.values() if m.reachable]
        if not reachable:
            return 0.0
        return sum(m.prefix_cache_hit_rate for m in reachable) / len(reachable)

    @property
    def max_queue_depth(self) -> int:
        if not self.servers:
            return 0
        return max(m.queue_depth for m in self.servers.values())


class InferenceMetricsCollector:
    """Periodic metrics collector for mixed inference backends."""

    def __init__(
        self,
        server_urls: List[str],
        *,
        poll_interval_seconds: float = 2.0,
    ):
        self.server_urls = [str(url).rstrip("/") for url in server_urls]
        self.poll_interval_seconds = max(0.25, float(poll_interval_seconds))
        self._metrics: Dict[str, InferenceServerMetrics] = {
            url: InferenceServerMetrics(server_url=url) for url in self.server_urls
        }
        self._poll_count = 0
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._session = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "InferenceMetricsCollector":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def get(self, server_url: str) -> InferenceServerMetrics:
        normalized = str(server_url).rstrip("/")
        return self._metrics.get(normalized, InferenceServerMetrics(server_url=normalized))

    def snapshot(self) -> InferenceMetricsSnapshot:
        return InferenceMetricsSnapshot(
            servers=dict(self._metrics),
            poll_count=self._poll_count,
            last_poll=max((m.timestamp for m in self._metrics.values()), default=0.0),
        )

    def update_servers(self, server_urls: List[str]) -> None:
        normalized = [str(url).rstrip("/") for url in server_urls]
        next_urls = set(normalized)
        old_urls = set(self._metrics.keys())
        for removed in old_urls - next_urls:
            self._metrics.pop(removed, None)
        for added in next_urls - old_urls:
            self._metrics[added] = InferenceServerMetrics(server_url=added)
        self.server_urls = normalized

    async def _loop(self) -> None:
        while True:
            await self._poll_all()
            self._poll_count += 1
            await asyncio.sleep(self.poll_interval_seconds)

    async def _poll_all(self) -> None:
        tasks = [self._poll_one(url) for url in self.server_urls]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _poll_one(self, server_url: str) -> None:
        if self._session is None:
            return
        target = f"{server_url}/metrics"
        state = self._metrics.setdefault(server_url, InferenceServerMetrics(server_url=server_url))
        try:
            async with self._session.get(
                target,
                timeout=aiohttp.ClientTimeout(total=2.0),
            ) as resp:
                text = await resp.text()
                state.timestamp = time.time()
                state.reachable = resp.status == 200
                if resp.status != 200:
                    return
                state.queue_waiting = int(_first_float(text, _QUEUE_WAITING_PATTERNS, default=0.0))
                state.queue_running = int(_first_float(text, _QUEUE_RUNNING_PATTERNS, default=0.0))
                state.kv_cache_usage_pct = float(_first_float(text, _KV_USAGE_PATTERNS, default=0.0))
                state.prefix_cache_hit_rate = float(_first_float(text, _PREFIX_CACHE_HIT_PATTERNS, default=0.0))
        except Exception:
            state.reachable = False
            state.timestamp = time.time()
            logger.debug("Metrics poll failed for %s", server_url, exc_info=True)
