"""Tree-summary batching benchmark for local vLLM-compatible endpoints.

The benchmark keeps the total synthetic input token workload fixed while
varying leaf size, output reserve, client concurrency, queue-drain batch size,
batch timeout, and the shared DSPy/offload worker cap. It sends real leaf and
merge summary requests through ``AsyncBatchLLMClient`` so the measured behavior
matches the current C-TreePO text batching path.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from treepo._research.core.async_utils import configure_to_thread_max_workers
from treepo._research.core.batch_processor import AsyncBatchLLMClient, BatchRequest, BatchResponse
from treepo._research.core.prompting import default_merge_prompt, default_summarize_prompt
from treepo._research.core.vllm_metrics import MetricsSnapshot, VLLMMetricsCollector

DEFAULT_RUBRIC = (
    "Preserve policy commitments, ideological direction, actors, tradeoffs, "
    "and evidence relevant to left-right manifesto scoring."
)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(percentile)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def parse_positive_int_grid(grid: str, *, name: str = "grid", allow_zero: bool = False) -> List[int]:
    """Parse a comma-separated positive integer grid into sorted unique values."""
    values: List[int] = []
    for part in str(grid).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 0 or (value == 0 and not allow_zero):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} values must be {qualifier}, got {value}")
        values.append(value)
    deduped = sorted(set(values))
    if not deduped:
        raise ValueError(f"{name} is empty")
    return deduped


def parse_positive_float_grid(grid: str, *, name: str = "grid") -> List[float]:
    """Parse a comma-separated positive float grid into sorted unique values."""
    values: List[float] = []
    for part in str(grid).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value <= 0.0:
            raise ValueError(f"{name} values must be positive, got {value}")
        values.append(value)
    deduped = sorted(set(values))
    if not deduped:
        raise ValueError(f"{name} is empty")
    return deduped


@dataclass(frozen=True)
class TreeBatchPointConfig:
    """One benchmark grid point."""

    leaf_tokens: int
    summary_max_tokens: int
    max_concurrent_requests: int
    batch_size: int
    batch_timeout: float
    dspy_workers: Optional[int] = None

    def label(self) -> str:
        worker_label = "default" if self.dspy_workers is None else str(self.dspy_workers)
        return (
            f"leaf={self.leaf_tokens} out={self.summary_max_tokens} "
            f"conc={self.max_concurrent_requests} batch={self.batch_size} "
            f"timeout={self.batch_timeout:g} dspy_workers={worker_label}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def expand_tree_batch_grid(
    *,
    leaf_tokens: Sequence[int],
    summary_max_tokens: Sequence[int],
    max_concurrent_requests: Sequence[int],
    batch_sizes: Sequence[int],
    batch_timeouts: Sequence[float],
    dspy_workers: Sequence[Optional[int]],
) -> List[TreeBatchPointConfig]:
    """Expand benchmark dimensions into deterministic point configs."""
    points: List[TreeBatchPointConfig] = []
    for leaf in leaf_tokens:
        for out in summary_max_tokens:
            for conc in max_concurrent_requests:
                for batch in batch_sizes:
                    for timeout in batch_timeouts:
                        for workers in dspy_workers:
                            points.append(
                                TreeBatchPointConfig(
                                    leaf_tokens=int(leaf),
                                    summary_max_tokens=int(out),
                                    max_concurrent_requests=int(conc),
                                    batch_size=int(batch),
                                    batch_timeout=float(timeout),
                                    dspy_workers=None if workers is None else int(workers),
                                )
                            )
    return points


@dataclass
class TreeBatchBudgetReport:
    """Static request-density estimate for one tree benchmark point."""

    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    prompt_overhead_tokens: int
    leaf_tokens: int
    summary_max_tokens: int
    merge_fan_in: int
    safety_fraction: float
    leaf_request_tokens: int
    merge_request_tokens: int
    leaf_context_fits: bool
    merge_context_fits: bool
    recommended_leaf_concurrency: int
    recommended_merge_concurrency: int
    recommended_max_concurrent_requests: int
    output_reserve_share: float
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_tree_batch_budget(
    *,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    prompt_overhead_tokens: int,
    leaf_tokens: int,
    summary_max_tokens: int,
    merge_fan_in: int = 2,
    safety_fraction: float = 0.90,
) -> TreeBatchBudgetReport:
    """Estimate safe active request density from vLLM and tree-shape knobs."""
    max_model_len = max(1, int(max_model_len))
    max_num_seqs = max(1, int(max_num_seqs))
    max_num_batched_tokens = max(0, int(max_num_batched_tokens))
    prompt_overhead_tokens = max(0, int(prompt_overhead_tokens))
    leaf_tokens = max(1, int(leaf_tokens))
    summary_max_tokens = max(1, int(summary_max_tokens))
    merge_fan_in = max(2, int(merge_fan_in))
    safety_fraction = min(1.0, max(0.01, float(safety_fraction)))

    leaf_request_tokens = prompt_overhead_tokens + leaf_tokens + summary_max_tokens
    merge_input_tokens = merge_fan_in * summary_max_tokens
    merge_request_tokens = prompt_overhead_tokens + merge_input_tokens + summary_max_tokens
    leaf_context_fits = leaf_request_tokens <= max_model_len
    merge_context_fits = merge_request_tokens <= max_model_len

    if max_num_batched_tokens > 0:
        active_token_budget = int(max_num_batched_tokens * safety_fraction)
    else:
        active_token_budget = int(max_model_len * max_num_seqs * safety_fraction)

    def safe_concurrency(tokens_per_request: int, fits: bool) -> int:
        if not fits:
            return 0
        by_tokens = max(1, active_token_budget // max(1, tokens_per_request))
        return max(1, min(max_num_seqs, int(by_tokens)))

    leaf_concurrency = safe_concurrency(leaf_request_tokens, leaf_context_fits)
    merge_concurrency = safe_concurrency(merge_request_tokens, merge_context_fits)
    recommended = max(1, min(v for v in (leaf_concurrency, merge_concurrency) if v > 0)) if (
        leaf_concurrency > 0 and merge_concurrency > 0
    ) else 0

    notes: List[str] = []
    if not leaf_context_fits:
        notes.append("leaf prompt+output reserve exceeds max_model_len")
    if not merge_context_fits:
        notes.append("merge prompt+output reserve exceeds max_model_len")
    if summary_max_tokens > leaf_tokens:
        notes.append("summary output reserve exceeds leaf input size")
    if not notes:
        notes.append("static budget fits context window")

    return TreeBatchBudgetReport(
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        prompt_overhead_tokens=prompt_overhead_tokens,
        leaf_tokens=leaf_tokens,
        summary_max_tokens=summary_max_tokens,
        merge_fan_in=merge_fan_in,
        safety_fraction=safety_fraction,
        leaf_request_tokens=leaf_request_tokens,
        merge_request_tokens=merge_request_tokens,
        leaf_context_fits=leaf_context_fits,
        merge_context_fits=merge_context_fits,
        recommended_leaf_concurrency=leaf_concurrency,
        recommended_merge_concurrency=merge_concurrency,
        recommended_max_concurrent_requests=recommended,
        output_reserve_share=float(summary_max_tokens) / float(max(1, leaf_request_tokens)),
        notes="; ".join(notes),
    )


@dataclass
class InferenceMetricsSummary:
    """Aggregate vLLM/SGLang metrics observed during one benchmark point."""

    samples: int = 0
    reachable_samples: int = 0
    max_kv_cache_usage_pct: Optional[float] = None
    max_requests_waiting: Optional[int] = None
    max_requests_running: Optional[int] = None
    avg_prefix_cache_hit_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_metrics_snapshots(snapshots: Sequence[Any]) -> InferenceMetricsSummary:
    """Aggregate metrics snapshots, tolerating absent or unreachable metrics."""
    kv_values: List[float] = []
    waiting_values: List[int] = []
    running_values: List[int] = []
    prefix_values: List[float] = []
    reachable_samples = 0

    for snapshot in snapshots:
        servers = getattr(snapshot, "servers", None)
        if servers is None and isinstance(snapshot, dict):
            servers = snapshot.get("servers")
        if not servers:
            continue
        sample_reachable = False
        for server in servers.values():
            reachable = bool(getattr(server, "reachable", False))
            if isinstance(server, dict):
                reachable = bool(server.get("reachable", False))
            if not reachable:
                continue
            sample_reachable = True
            if isinstance(server, dict):
                kv_values.append(float(server.get("kv_cache_usage_pct", 0.0) or 0.0))
                waiting_values.append(int(server.get("num_requests_waiting", 0) or 0))
                running_values.append(int(server.get("num_requests_running", 0) or 0))
                prefix_values.append(float(server.get("prefix_cache_hit_rate", 0.0) or 0.0))
            else:
                kv_values.append(float(getattr(server, "kv_cache_usage_pct", 0.0) or 0.0))
                waiting_values.append(int(getattr(server, "num_requests_waiting", 0) or 0))
                running_values.append(int(getattr(server, "num_requests_running", 0) or 0))
                prefix_values.append(float(getattr(server, "prefix_cache_hit_rate", 0.0) or 0.0))
        if sample_reachable:
            reachable_samples += 1

    return InferenceMetricsSummary(
        samples=len(snapshots),
        reachable_samples=reachable_samples,
        max_kv_cache_usage_pct=max(kv_values) if kv_values else None,
        max_requests_waiting=max(waiting_values) if waiting_values else None,
        max_requests_running=max(running_values) if running_values else None,
        avg_prefix_cache_hit_rate=(
            sum(prefix_values) / len(prefix_values) if prefix_values else None
        ),
    )


@dataclass
class TreeBatchPointResult:
    """Measured metrics for one benchmark point."""

    point: TreeBatchPointConfig
    total_input_tokens_target: int
    document_count: int
    leaf_requests: int
    merge_requests: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    wall_seconds: float
    docs_per_second: float
    leaves_per_second: float
    merges_per_second: float
    requests_per_second: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_per_second: float
    completion_tokens_per_second: float
    tokens_per_second: float
    latency_avg_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    metrics: InferenceMetricsSummary = field(default_factory=InferenceMetricsSummary)
    budget: Optional[TreeBatchBudgetReport] = None
    error_preview: str = ""

    @property
    def success_rate(self) -> float:
        if self.total_requests <= 0:
            return 0.0
        return float(self.successful_requests) / float(self.total_requests)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["point"] = self.point.to_dict()
        data["metrics"] = self.metrics.to_dict()
        data["budget"] = self.budget.to_dict() if self.budget is not None else None
        data["success_rate"] = self.success_rate
        return data


@dataclass
class LeafStabilityRow:
    """Throughput stability across leaf sizes for an otherwise fixed point."""

    key: Dict[str, Any]
    leaf_tokens: List[int]
    tokens_per_second: List[float]
    coefficient_of_variation: float
    max_over_min_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TreeBatchSuiteSummary:
    """High-level ranking and invariance diagnostics for a suite."""

    point_count: int
    best_tokens_point: Optional[TreeBatchPointResult]
    best_docs_point: Optional[TreeBatchPointResult]
    leaf_stability: List[LeafStabilityRow]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_count": self.point_count,
            "best_tokens_point": (
                self.best_tokens_point.to_dict() if self.best_tokens_point is not None else None
            ),
            "best_docs_point": (
                self.best_docs_point.to_dict() if self.best_docs_point is not None else None
            ),
            "leaf_stability": [row.to_dict() for row in self.leaf_stability],
        }


@dataclass
class TreeBatchSuiteResult:
    """Complete benchmark suite output."""

    config: Dict[str, Any]
    points: List[TreeBatchPointResult]
    summary: TreeBatchSuiteSummary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": dict(self.config),
            "summary": self.summary.to_dict(),
            "points": [point.to_dict() for point in self.points],
        }


def summarize_tree_batch_results(
    points: Sequence[TreeBatchPointResult],
    *,
    min_success_rate: float = 0.95,
) -> TreeBatchSuiteSummary:
    """Rank points and compute leaf-size throughput invariance diagnostics."""
    stable = [p for p in points if p.success_rate >= float(min_success_rate)]
    ranked = stable if stable else list(points)
    best_tokens = max(ranked, key=lambda p: p.tokens_per_second, default=None)
    best_docs = max(ranked, key=lambda p: p.docs_per_second, default=None)

    groups: Dict[Tuple[Any, ...], List[TreeBatchPointResult]] = {}
    for point in stable:
        key = (
            point.point.summary_max_tokens,
            point.point.max_concurrent_requests,
            point.point.batch_size,
            round(point.point.batch_timeout, 9),
            point.point.dspy_workers,
        )
        groups.setdefault(key, []).append(point)

    rows: List[LeafStabilityRow] = []
    for key, group in groups.items():
        by_leaf = {p.point.leaf_tokens: p for p in group}
        if len(by_leaf) < 2:
            continue
        ordered_leaf = sorted(by_leaf)
        values = [float(by_leaf[leaf].tokens_per_second) for leaf in ordered_leaf]
        mean = sum(values) / len(values) if values else 0.0
        cv = statistics.pstdev(values) / mean if mean > 0.0 and len(values) > 1 else 0.0
        positive = [v for v in values if v > 0.0]
        ratio = (max(positive) / min(positive)) if positive else 0.0
        rows.append(
            LeafStabilityRow(
                key={
                    "summary_max_tokens": key[0],
                    "max_concurrent_requests": key[1],
                    "batch_size": key[2],
                    "batch_timeout": key[3],
                    "dspy_workers": key[4],
                },
                leaf_tokens=ordered_leaf,
                tokens_per_second=values,
                coefficient_of_variation=float(cv),
                max_over_min_ratio=float(ratio),
            )
        )
    rows.sort(key=lambda row: row.coefficient_of_variation)

    return TreeBatchSuiteSummary(
        point_count=len(points),
        best_tokens_point=best_tokens,
        best_docs_point=best_docs,
        leaf_stability=rows,
    )


class _MetricsRunSampler:
    def __init__(self, *, base_urls: Sequence[str], poll_seconds: float):
        host, ports = _extract_host_and_ports(base_urls)
        self.host = host
        self.ports = ports
        self.poll_seconds = max(0.0, float(poll_seconds))
        self.snapshots: List[MetricsSnapshot] = []
        self._collector: Optional[VLLMMetricsCollector] = None
        self._task: Optional[asyncio.Task[None]] = None

    async def __aenter__(self) -> "_MetricsRunSampler":
        if not self.host or not self.ports or self.poll_seconds <= 0.0:
            return self
        self._collector = VLLMMetricsCollector(
            ports=self.ports,
            host=self.host,
            poll_interval=self.poll_seconds,
        )
        await self._collector.start()
        self._task = asyncio.create_task(self._sample_loop())
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._collector is not None:
            await self._collector.stop()

    async def _sample_loop(self) -> None:
        assert self._collector is not None
        while True:
            self.snapshots.append(self._collector.snapshot())
            await asyncio.sleep(self.poll_seconds)


def _extract_host_and_ports(base_urls: Sequence[str]) -> Tuple[Optional[str], List[int]]:
    host: Optional[str] = None
    ports: List[int] = []
    for raw in base_urls:
        parsed = urlparse(str(raw))
        if parsed.hostname is None or parsed.port is None:
            continue
        if host is None:
            host = parsed.hostname
        if parsed.hostname != host:
            return None, []
        ports.append(int(parsed.port))
    return host, sorted(set(ports))


def _synthetic_words(target_tokens: int, *, doc_index: int, leaf_index: int) -> str:
    fragments = [
        "labor standards wage bargaining workplace safety",
        "public investment housing transport regional infrastructure",
        "climate policy industrial transition energy resilience",
        "healthcare access primary care hospital capacity",
        "education training apprenticeships research universities",
        "tax fairness enterprise productivity fiscal responsibility",
        "trade cooperation supply chains domestic manufacturing",
        "civil liberties institutions transparency anti corruption",
    ]
    words: List[str] = []
    cursor = (doc_index * 17 + leaf_index * 31) % len(fragments)
    while len(words) < target_tokens:
        words.extend(fragments[cursor % len(fragments)].split())
        words.append(f"doc{doc_index}")
        words.append(f"leaf{leaf_index}")
        cursor += 1
    return " ".join(words[:target_tokens])


def build_synthetic_leaf_texts(
    *,
    total_input_tokens: int,
    document_count: int,
    leaf_tokens: int,
) -> Dict[str, List[str]]:
    """Build fixed-total-token synthetic documents split into leaf payloads."""
    total_input_tokens = max(1, int(total_input_tokens))
    document_count = max(1, int(document_count))
    leaf_tokens = max(1, int(leaf_tokens))
    base_tokens = total_input_tokens // document_count
    remainder = total_input_tokens % document_count

    docs: Dict[str, List[str]] = {}
    for doc_idx in range(document_count):
        doc_tokens = base_tokens + (1 if doc_idx < remainder else 0)
        leaves = max(1, int(math.ceil(doc_tokens / float(leaf_tokens))))
        leaf_texts: List[str] = []
        remaining = doc_tokens
        for leaf_idx in range(leaves):
            take = min(leaf_tokens, max(1, remaining))
            leaf_texts.append(
                _synthetic_words(take, doc_index=doc_idx, leaf_index=leaf_idx)
            )
            remaining -= take
        docs[f"synthetic_doc_{doc_idx:04d}"] = leaf_texts
    return docs


async def _submit_and_collect(
    client: AsyncBatchLLMClient,
    requests: Sequence[BatchRequest],
    *,
    await_timeout: float,
) -> List[BatchResponse]:
    for request in requests:
        await client.submit(request)
    tasks = [
        asyncio.create_task(client.await_response(request.request_id, timeout=await_timeout))
        for request in requests
    ]
    return list(await asyncio.gather(*tasks))


def _usage_total(responses: Iterable[BatchResponse], key: str) -> int:
    total = 0
    for response in responses:
        try:
            total += int(response.usage.get(key, 0) or 0)
        except Exception:
            continue
    return total


def _apply_dspy_worker_limit(workers: Optional[int]) -> None:
    if workers is None:
        os.environ.pop("TT_TO_THREAD_MAX_WORKERS", None)
        configure_to_thread_max_workers(None)
        return
    configure_to_thread_max_workers(int(workers))
    os.environ["TT_TO_THREAD_MAX_WORKERS"] = str(int(workers))


async def run_tree_batch_point(
    *,
    point: TreeBatchPointConfig,
    base_url: str,
    total_input_tokens: int,
    document_count: int,
    request_timeout_seconds: float,
    await_response_timeout_seconds: Optional[float],
    metrics_poll_seconds: float,
    api_key: str,
    temperature: float,
    rubric: str,
    budget: Optional[TreeBatchBudgetReport] = None,
    call_sink: Optional[Any] = None,
) -> TreeBatchPointResult:
    """Run one live tree-summary batching point against a vLLM-compatible server."""
    _apply_dspy_worker_limit(point.dspy_workers)
    docs = build_synthetic_leaf_texts(
        total_input_tokens=total_input_tokens,
        document_count=document_count,
        leaf_tokens=point.leaf_tokens,
    )
    all_responses: List[BatchResponse] = []
    latencies: List[float] = []
    errors: List[str] = []
    leaf_requests = 0
    merge_requests = 0
    await_timeout = float(await_response_timeout_seconds or (request_timeout_seconds + 30.0))

    async with _MetricsRunSampler(base_urls=[base_url], poll_seconds=metrics_poll_seconds) as sampler:
        start = time.perf_counter()
        async with AsyncBatchLLMClient(
            base_url=base_url,
            max_concurrent=point.max_concurrent_requests,
            batch_size=point.batch_size,
            batch_timeout=point.batch_timeout,
            request_timeout=request_timeout_seconds,
            api_key=api_key,
            call_sink=call_sink,
        ) as client:
            summaries_by_doc: Dict[str, List[str]] = {}
            leaf_batch: List[BatchRequest] = []
            run_id = uuid.uuid4().hex[:8]
            for doc_id, leaves in docs.items():
                summaries_by_doc[doc_id] = []
                for leaf_idx, text in enumerate(leaves):
                    leaf_requests += 1
                    leaf_batch.append(
                        BatchRequest(
                            request_id=f"{run_id}:{doc_id}:leaf:{leaf_idx}",
                            messages=default_summarize_prompt(text, rubric),
                            max_tokens=point.summary_max_tokens,
                            temperature=temperature,
                            document_id=doc_id,
                            request_type="leaf",
                        )
                    )
            leaf_responses = await _submit_and_collect(
                client,
                leaf_batch,
                await_timeout=await_timeout,
            )
            all_responses.extend(leaf_responses)
            for request, response in zip(leaf_batch, leaf_responses):
                if response.latency_ms > 0:
                    latencies.append(float(response.latency_ms))
                if response.error:
                    errors.append(str(response.error))
                doc_id = str(request.document_id or "")
                summaries_by_doc.setdefault(doc_id, []).append(response.content or "")

            level = 0
            while any(len(items) > 1 for items in summaries_by_doc.values()):
                merge_batch: List[BatchRequest] = []
                slots: List[Tuple[str, int]] = []
                next_by_doc: Dict[str, List[str]] = {}
                for doc_id, current in summaries_by_doc.items():
                    next_items: List[str] = []
                    idx = 0
                    while idx < len(current):
                        if idx + 1 >= len(current):
                            next_items.append(current[idx])
                            idx += 1
                            continue
                        slot_idx = len(next_items)
                        next_items.append("")
                        left = current[idx]
                        right = current[idx + 1]
                        merge_requests += 1
                        merge_batch.append(
                            BatchRequest(
                                request_id=f"{run_id}:{doc_id}:merge:{level}:{slot_idx}",
                                messages=default_merge_prompt(left, right, rubric),
                                max_tokens=point.summary_max_tokens,
                                temperature=temperature,
                                document_id=doc_id,
                                request_type="merge",
                            )
                        )
                        slots.append((doc_id, slot_idx))
                        idx += 2
                    next_by_doc[doc_id] = next_items

                if not merge_batch:
                    break
                merge_responses = await _submit_and_collect(
                    client,
                    merge_batch,
                    await_timeout=await_timeout,
                )
                all_responses.extend(merge_responses)
                for (doc_id, slot_idx), response in zip(slots, merge_responses):
                    if response.latency_ms > 0:
                        latencies.append(float(response.latency_ms))
                    if response.error:
                        errors.append(str(response.error))
                    next_by_doc[doc_id][slot_idx] = response.content or ""
                summaries_by_doc = next_by_doc
                level += 1

            stats = client.stats
            stats.wall_clock_end = time.time()
        wall = max(1e-9, time.perf_counter() - start)

    successful = sum(1 for response in all_responses if not response.error)
    failed = len(all_responses) - successful
    prompt_tokens = _usage_total(all_responses, "prompt_tokens")
    completion_tokens = _usage_total(all_responses, "completion_tokens")
    total_tokens = _usage_total(all_responses, "total_tokens")
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    return TreeBatchPointResult(
        point=point,
        total_input_tokens_target=int(total_input_tokens),
        document_count=int(document_count),
        leaf_requests=leaf_requests,
        merge_requests=merge_requests,
        total_requests=len(all_responses),
        successful_requests=successful,
        failed_requests=failed,
        wall_seconds=wall,
        docs_per_second=float(document_count) / wall,
        leaves_per_second=float(leaf_requests) / wall,
        merges_per_second=float(merge_requests) / wall,
        requests_per_second=float(successful) / wall,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_per_second=float(prompt_tokens) / wall,
        completion_tokens_per_second=float(completion_tokens) / wall,
        tokens_per_second=float(total_tokens) / wall,
        latency_avg_ms=(sum(latencies) / len(latencies)) if latencies else 0.0,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        metrics=summarize_metrics_snapshots(sampler.snapshots),
        budget=budget,
        error_preview=" | ".join(errors[:3]),
    )


async def run_fake_tree_batch_point(
    *,
    point: TreeBatchPointConfig,
    total_input_tokens: int,
    document_count: int,
    budget: Optional[TreeBatchBudgetReport] = None,
) -> TreeBatchPointResult:
    """No-server deterministic point runner for smoke tests and artifact checks."""
    docs = build_synthetic_leaf_texts(
        total_input_tokens=total_input_tokens,
        document_count=document_count,
        leaf_tokens=point.leaf_tokens,
    )
    leaf_requests = sum(len(leaves) for leaves in docs.values())
    merge_requests = sum(max(0, len(leaves) - 1) for leaves in docs.values())
    total_requests = leaf_requests + merge_requests
    effective_parallelism = max(1, min(point.max_concurrent_requests, point.batch_size * 2))
    prompt_tokens = int(total_input_tokens + merge_requests * point.summary_max_tokens * 2)
    completion_tokens = int(total_requests * min(point.summary_max_tokens, max(8, point.leaf_tokens // 8)))
    total_tokens = prompt_tokens + completion_tokens
    synthetic_capacity = 2200.0 * effective_parallelism / (1.0 + point.batch_timeout * 10.0)
    wall = max(0.001, float(total_tokens) / max(1.0, synthetic_capacity))
    latency_base = 1000.0 * wall / max(1, math.ceil(total_requests / effective_parallelism))
    latencies = [latency_base * (1.0 + (idx % 5) * 0.05) for idx in range(total_requests)]

    return TreeBatchPointResult(
        point=point,
        total_input_tokens_target=int(total_input_tokens),
        document_count=int(document_count),
        leaf_requests=leaf_requests,
        merge_requests=merge_requests,
        total_requests=total_requests,
        successful_requests=total_requests,
        failed_requests=0,
        wall_seconds=wall,
        docs_per_second=float(document_count) / wall,
        leaves_per_second=float(leaf_requests) / wall,
        merges_per_second=float(merge_requests) / wall,
        requests_per_second=float(total_requests) / wall,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_per_second=float(prompt_tokens) / wall,
        completion_tokens_per_second=float(completion_tokens) / wall,
        tokens_per_second=float(total_tokens) / wall,
        latency_avg_ms=(sum(latencies) / len(latencies)) if latencies else 0.0,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        metrics=summarize_metrics_snapshots([]),
        budget=budget,
        error_preview="",
    )


async def run_tree_batching_suite(
    *,
    points: Sequence[TreeBatchPointConfig],
    base_url: str,
    total_input_tokens: int,
    document_count: int,
    request_timeout_seconds: float,
    await_response_timeout_seconds: Optional[float],
    metrics_poll_seconds: float,
    api_key: str,
    temperature: float,
    rubric: str,
    fake: bool,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    prompt_overhead_tokens: int,
    budget_safety_fraction: float,
    min_success_rate: float = 0.95,
    call_sink: Optional[Any] = None,
) -> TreeBatchSuiteResult:
    """Run a tree batching sweep and return structured results."""
    results: List[TreeBatchPointResult] = []
    original_worker_env = os.environ.get("TT_TO_THREAD_MAX_WORKERS")
    try:
        for point in points:
            budget = compute_tree_batch_budget(
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                prompt_overhead_tokens=prompt_overhead_tokens,
                leaf_tokens=point.leaf_tokens,
                summary_max_tokens=point.summary_max_tokens,
                safety_fraction=budget_safety_fraction,
            )
            if fake:
                result = await run_fake_tree_batch_point(
                    point=point,
                    total_input_tokens=total_input_tokens,
                    document_count=document_count,
                    budget=budget,
                )
            else:
                result = await run_tree_batch_point(
                    point=point,
                    base_url=base_url,
                    total_input_tokens=total_input_tokens,
                    document_count=document_count,
                    request_timeout_seconds=request_timeout_seconds,
                    await_response_timeout_seconds=await_response_timeout_seconds,
                    metrics_poll_seconds=metrics_poll_seconds,
                    api_key=api_key,
                    temperature=temperature,
                    rubric=rubric,
                    budget=budget,
                    call_sink=call_sink,
                )
            results.append(result)
    finally:
        if original_worker_env is None:
            os.environ.pop("TT_TO_THREAD_MAX_WORKERS", None)
        else:
            os.environ["TT_TO_THREAD_MAX_WORKERS"] = original_worker_env
        configure_to_thread_max_workers(None)

    config = {
        "base_url": base_url,
        "total_input_tokens": int(total_input_tokens),
        "document_count": int(document_count),
        "request_timeout_seconds": float(request_timeout_seconds),
        "await_response_timeout_seconds": await_response_timeout_seconds,
        "metrics_poll_seconds": float(metrics_poll_seconds),
        "temperature": float(temperature),
        "fake": bool(fake),
        "max_model_len": int(max_model_len),
        "max_num_seqs": int(max_num_seqs),
        "max_num_batched_tokens": int(max_num_batched_tokens),
        "prompt_overhead_tokens": int(prompt_overhead_tokens),
        "budget_safety_fraction": float(budget_safety_fraction),
        "min_success_rate": float(min_success_rate),
    }
    return TreeBatchSuiteResult(
        config=config,
        points=results,
        summary=summarize_tree_batch_results(results, min_success_rate=min_success_rate),
    )


def render_tree_batch_markdown(suite: TreeBatchSuiteResult, *, top_n: int = 10) -> str:
    """Render a compact Markdown report for a tree batching suite."""
    lines: List[str] = []
    lines.append("# Tree Batching Throughput Sweep")
    lines.append("")
    lines.append(f"- Points: {len(suite.points)}")
    lines.append(f"- Total input token target: {suite.config.get('total_input_tokens')}")
    lines.append(f"- Documents: {suite.config.get('document_count')}")
    lines.append(f"- Fake/no-server mode: {suite.config.get('fake')}")
    lines.append("")

    if suite.summary.best_tokens_point is not None:
        best = suite.summary.best_tokens_point
        lines.append("## Best Token Throughput")
        lines.append("")
        lines.append(
            f"- {best.point.label()} -> {best.tokens_per_second:.1f} tok/s, "
            f"{best.docs_per_second:.3f} docs/s, p95={best.latency_p95_ms:.1f} ms, "
            f"success={100.0 * best.success_rate:.1f}%"
        )
        if best.budget is not None:
            lines.append(
                f"- Budget recommendation: max_concurrent_requests="
                f"{best.budget.recommended_max_concurrent_requests} "
                f"({best.budget.notes})"
            )
        lines.append("")

    ranked = sorted(suite.points, key=lambda p: p.tokens_per_second, reverse=True)
    lines.append(f"## Top {min(top_n, len(ranked))} Points")
    lines.append("")
    lines.append(
        "| rank | leaf | max_out | conc | batch | timeout | dspy_workers | tok/s | "
        "docs/s | req/s | p95 ms | kv max | wait max | success |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for rank, point in enumerate(ranked[:top_n], start=1):
        kv = "" if point.metrics.max_kv_cache_usage_pct is None else (
            f"{100.0 * point.metrics.max_kv_cache_usage_pct:.1f}%"
        )
        wait = "" if point.metrics.max_requests_waiting is None else str(point.metrics.max_requests_waiting)
        workers = "" if point.point.dspy_workers is None else str(point.point.dspy_workers)
        lines.append(
            f"| {rank} | {point.point.leaf_tokens} | {point.point.summary_max_tokens} | "
            f"{point.point.max_concurrent_requests} | {point.point.batch_size} | "
            f"{point.point.batch_timeout:g} | {workers} | {point.tokens_per_second:.1f} | "
            f"{point.docs_per_second:.3f} | {point.requests_per_second:.2f} | "
            f"{point.latency_p95_ms:.1f} | {kv} | {wait} | "
            f"{100.0 * point.success_rate:.1f}% |"
        )
    lines.append("")

    lines.append("## Leaf-Size Stability")
    lines.append("")
    if suite.summary.leaf_stability:
        lines.append("| summary_out | conc | batch | timeout | dspy_workers | leaves | tok/s | cv | max/min |")
        lines.append("|---:|---:|---:|---:|---:|---|---|---:|---:|")
        for row in suite.summary.leaf_stability[:top_n]:
            key = row.key
            workers = "" if key.get("dspy_workers") is None else str(key.get("dspy_workers"))
            leaves = ",".join(str(v) for v in row.leaf_tokens)
            tps = ",".join(f"{v:.0f}" for v in row.tokens_per_second)
            lines.append(
                f"| {key['summary_max_tokens']} | {key['max_concurrent_requests']} | "
                f"{key['batch_size']} | {key['batch_timeout']:g} | {workers} | "
                f"{leaves} | {tps} | {row.coefficient_of_variation:.3f} | "
                f"{row.max_over_min_ratio:.2f} |"
            )
    else:
        lines.append("No stable multi-leaf-size groups were available for invariance diagnostics.")
    lines.append("")
    return "\n".join(lines)


def write_tree_batch_jsonl(*, output_jsonl: Path, suite: TreeBatchSuiteResult) -> None:
    """Write one JSON object per benchmark point."""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with output_jsonl.open("w", encoding="utf-8") as f:
        for point in suite.points:
            payload = {
                "timestamp_utc": timestamp,
                "suite_config": dict(suite.config),
                "point": point.to_dict(),
            }
            f.write(json.dumps(payload, sort_keys=True) + "\n")


def write_tree_batch_markdown(*, output_markdown: Path, suite: TreeBatchSuiteResult) -> None:
    """Write the human-readable Markdown benchmark report."""
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(render_tree_batch_markdown(suite), encoding="utf-8")


def default_output_paths(
    *,
    output_dir: Path = Path("outputs"),
    prefix: str = "tree_batching",
) -> Tuple[Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = output_dir / f"{prefix}_{stamp}"
    return root.with_suffix(".jsonl"), root.with_suffix(".md")
