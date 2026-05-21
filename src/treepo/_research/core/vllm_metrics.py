"""
Lightweight vLLM / SGLang metrics collector.

Polls the Prometheus ``/metrics`` endpoint on each inference server and
exposes parsed values for use by the scheduler, monitoring, and logging.

Design inspired by DualPath (Section 6.1): real-time per-engine metrics
enable load-aware scheduling and cache-hit-rate visibility.

Works with both vLLM and SGLang — both expose Prometheus-compatible
``/metrics`` endpoints with similar metric names.
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

# Prometheus metric names (vLLM / SGLang compatible).
_KV_CACHE_USAGE_PATTERNS = [
    re.compile(r"^vllm:gpu_cache_usage_perc(?:\{.*?\})?\s+([\d.eE+-]+)", re.MULTILINE),
    re.compile(r"^sglang:kv_cache_usage(?:\{.*?\})?\s+([\d.eE+-]+)", re.MULTILINE),
]
_REQUESTS_WAITING_PATTERNS = [
    re.compile(r"^vllm:num_requests_waiting(?:\{.*?\})?\s+([\d.eE+-]+)", re.MULTILINE),
    re.compile(r"^sglang:num_requests_waiting(?:\{.*?\})?\s+([\d.eE+-]+)", re.MULTILINE),
]
_REQUESTS_RUNNING_PATTERNS = [
    re.compile(r"^vllm:num_requests_running(?:\{.*?\})?\s+([\d.eE+-]+)", re.MULTILINE),
    re.compile(r"^sglang:num_requests_running(?:\{.*?\})?\s+([\d.eE+-]+)", re.MULTILINE),
]
_PREFIX_CACHE_HIT_PATTERNS = [
    re.compile(r"^vllm:prefix_cache_hit_rate(?:\{.*?\})?\s+([\d.eE+-]+)", re.MULTILINE),
    re.compile(r"^sglang:cache_hit_rate(?:\{.*?\})?\s+([\d.eE+-]+)", re.MULTILINE),
]


def _first_float(text: str, patterns: List[re.Pattern[str]]) -> Optional[float]:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                continue
    return None


@dataclass
class ServerMetrics:
    """Parsed metrics snapshot from a single inference server."""

    port: int
    kv_cache_usage_pct: float = 0.0
    num_requests_waiting: int = 0
    num_requests_running: int = 0
    prefix_cache_hit_rate: float = 0.0
    timestamp: float = 0.0
    reachable: bool = False

    def __str__(self) -> str:
        return (
            f"port={self.port} kv={self.kv_cache_usage_pct:.1%} "
            f"wait={self.num_requests_waiting} run={self.num_requests_running} "
            f"prefix_hit={self.prefix_cache_hit_rate:.1%}"
        )


@dataclass
class MetricsSnapshot:
    """Aggregate snapshot across all servers."""

    servers: Dict[int, ServerMetrics] = field(default_factory=dict)
    poll_count: int = 0
    last_poll: float = 0.0

    @property
    def avg_kv_cache_usage(self) -> float:
        reachable = [s for s in self.servers.values() if s.reachable]
        if not reachable:
            return 0.0
        return sum(s.kv_cache_usage_pct for s in reachable) / len(reachable)

    @property
    def avg_prefix_cache_hit_rate(self) -> float:
        reachable = [s for s in self.servers.values() if s.reachable]
        if not reachable:
            return 0.0
        return sum(s.prefix_cache_hit_rate for s in reachable) / len(reachable)

    @property
    def max_queue_depth(self) -> int:
        if not self.servers:
            return 0
        return max(s.num_requests_waiting for s in self.servers.values())

    def summary_line(self) -> str:
        reachable = [s for s in self.servers.values() if s.reachable]
        if not reachable:
            return "no reachable servers"
        parts = []
        for s in sorted(reachable, key=lambda x: x.port):
            parts.append(
                f":{s.port}(kv={s.kv_cache_usage_pct:.0%} "
                f"pfx={s.prefix_cache_hit_rate:.0%} "
                f"q={s.num_requests_waiting})"
            )
        return " ".join(parts)


class VLLMMetricsCollector:
    """Periodically polls inference server ``/metrics`` endpoints.

    Parameters
    ----------
    ports : list of int
        Server ports to monitor.
    host : str
        Hostname (default ``localhost``).
    poll_interval : float
        Seconds between polls (default 2.0).
    """

    def __init__(
        self,
        ports: List[int],
        host: str = "localhost",
        poll_interval: float = 2.0,
    ):
        self.ports = list(ports)
        self.host = host
        self.poll_interval = poll_interval

        self._metrics: Dict[int, ServerMetrics] = {
            p: ServerMetrics(port=p) for p in ports
        }
        self._session: Optional[aiohttp.ClientSession] = None
        self._task: Optional[asyncio.Task] = None
        self._poll_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "Metrics collector started for ports %s (interval=%.1fs)",
            self.ports,
            self.poll_interval,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
        logger.info("Metrics collector stopped after %d polls", self._poll_count)

    async def __aenter__(self) -> "VLLMMetricsCollector":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, port: int) -> ServerMetrics:
        """Get latest metrics for a server (non-blocking)."""
        return self._metrics.get(port, ServerMetrics(port=port))

    def snapshot(self) -> MetricsSnapshot:
        """Get a snapshot of all server metrics."""
        return MetricsSnapshot(
            servers=dict(self._metrics),
            poll_count=self._poll_count,
            last_poll=max(
                (s.timestamp for s in self._metrics.values()), default=0.0
            ),
        )

    def update_ports(self, ports: List[int]) -> None:
        """Update the set of monitored ports (e.g., after GPU orchestration)."""
        new_ports = set(ports)
        old_ports = set(self.ports)
        for p in new_ports - old_ports:
            self._metrics[p] = ServerMetrics(port=p)
        for p in old_ports - new_ports:
            self._metrics.pop(p, None)
        self.ports = list(ports)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while True:
            await self._poll_all()
            self._poll_count += 1
            await asyncio.sleep(self.poll_interval)

    async def _poll_all(self) -> None:
        tasks = [self._poll_one(port) for port in self.ports]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _poll_one(self, port: int) -> None:
        url = f"http://{self.host}:{port}/metrics"
        m = self._metrics[port]
        try:
            timeout = aiohttp.ClientTimeout(total=2.0)
            async with self._session.get(url, timeout=timeout) as resp:
                text = await resp.text()
                m.timestamp = time.time()
                m.reachable = resp.status == 200
                if resp.status != 200:
                    return
                self._parse_prometheus(text, m)
        except Exception:
            m.reachable = False

    @staticmethod
    def _parse_prometheus(text: str, m: ServerMetrics) -> None:
        """Parse Prometheus text exposition format into ServerMetrics."""
        kv = _first_float(text, _KV_CACHE_USAGE_PATTERNS)
        if kv is not None:
            m.kv_cache_usage_pct = float(kv)

        waiting = _first_float(text, _REQUESTS_WAITING_PATTERNS)
        if waiting is not None:
            m.num_requests_waiting = int(waiting)

        running = _first_float(text, _REQUESTS_RUNNING_PATTERNS)
        if running is not None:
            m.num_requests_running = int(running)

        prefix = _first_float(text, _PREFIX_CACHE_HIT_PATTERNS)
        if prefix is not None:
            m.prefix_cache_hit_rate = float(prefix)
