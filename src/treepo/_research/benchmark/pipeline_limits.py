"""
Throughput-limit sweep for ThinkingTrees pipeline components.

This module provides reusable benchmarks for the high-load stages used in the
manifesto training pipeline:
1. Task model via AsyncBatchLLMClient (single endpoint)
2. Task model merge prompts (internal tree merge behavior)
3. Task model score prompts (scorer/oracle behavior)
4. Task model via MultiServerBatchClient (DP2 / dual task endpoints)
5. GenRM via direct chat/completions requests
6. GenRM via AsyncBatchGenRMClient (pipeline path)
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import aiohttp

from treepo._research.core.batch_processor import AsyncBatchLLMClient, BatchRequest, MultiServerBatchClient
from treepo._research.core.model_detection import detect_model_async
from treepo._research.core.prompting import parse_numeric_score
from treepo._research.training.judges.genrm import is_genrm_error
from treepo._research.training.judges.genrm_batch import AsyncBatchGenRMClient, GenRMComparisonRequest

logger = logging.getLogger(__name__)

GENRM_MODE_CONFIGS: Dict[str, Dict[str, bool]] = {
    # Production-oriented: concise and stable under load.
    "fast": {
        "disable_thinking": True,
        "force_json_response": True,
    },
    # Exploration-oriented: allows chain-of-thought style outputs.
    "think": {
        "disable_thinking": False,
        "force_json_response": False,
    },
}


def parse_concurrency_grid(grid: str) -> List[int]:
    """Parse comma-separated concurrency values into sorted unique integers."""
    values: List[int] = []
    for part in str(grid).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"Concurrency values must be positive, got: {value}")
        values.append(value)
    deduped = sorted(set(values))
    if not deduped:
        raise ValueError("Concurrency grid is empty")
    return deduped


def parse_genrm_modes(mode_csv: str) -> List[str]:
    """Parse comma-separated GenRM modes."""
    values: List[str] = []
    for part in str(mode_csv).split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part not in GENRM_MODE_CONFIGS:
            raise ValueError(
                f"Unknown GenRM mode: {part}. Valid: {sorted(GENRM_MODE_CONFIGS.keys())}"
            )
        values.append(part)
    deduped = list(dict.fromkeys(values))
    if not deduped:
        raise ValueError("GenRM mode list is empty")
    return deduped


def expand_genrm_steps(steps: Sequence[str], genrm_modes: Sequence[str]) -> List[str]:
    """
    Expand generic GenRM steps into concrete mode-specific steps.

    Example:
        ["task_single", "genrm_batch"], modes=["fast","think"]
        -> ["task_single", "genrm_batch_fast", "genrm_batch_think"]
    """
    expanded: List[str] = []
    for step in steps:
        if step == "genrm_raw":
            expanded.extend([f"genrm_raw_{mode}" for mode in genrm_modes])
            continue
        if step == "genrm_batch":
            expanded.extend([f"genrm_batch_{mode}" for mode in genrm_modes])
            continue
        expanded.append(step)
    return expanded


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Compute percentile with linear interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * float(percentile)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(ordered[lo])
    weight = rank - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def _default_task_text(index: int, target_chars: int = 1200) -> str:
    """
    Build deterministic manifesto-style text for benchmarking prompts.

    Deterministic but varied strings reduce pure prefix-cache artifacts while
    keeping the test reproducible.
    """
    random.seed(index)
    fragments = [
        "We support stronger labor standards and safe workplaces.",
        "Public investment should expand affordable housing and transport.",
        "Climate policy should align industrial growth with emissions reduction.",
        "Healthcare access must improve through primary-care expansion.",
        "Tax policy should preserve growth while reducing inequality.",
        "Education funding should prioritize early learning and skills training.",
        "Trade policy should protect workers while maintaining cooperation.",
        "Energy strategy should combine grid resilience with decarbonization.",
        "Rural development requires broadband and logistics modernization.",
        "Institutions should strengthen anti-corruption enforcement and transparency.",
    ]
    pieces: List[str] = []
    while len(" ".join(pieces)) < target_chars:
        pieces.append(random.choice(fragments))
        pieces.append(f"[doc={index} policy_id={len(pieces)}]")
    return " ".join(pieces)


def _build_task_messages(index: int, target_chars: int) -> List[Dict[str, str]]:
    text = _default_task_text(index=index, target_chars=target_chars)
    return [
        {
            "role": "system",
            "content": (
                "You are a political text analyst. Summarize the text while preserving policy "
                "positions and ideological direction."
            ),
        },
        {
            "role": "user",
            "content": (
                "Summarize the following manifesto excerpt in 2-4 sentences.\n\n"
                f"{text}\n\n"
                "Summary:"
            ),
        },
    ]


def _build_task_merge_messages(index: int, target_chars: int) -> List[Dict[str, str]]:
    left = _default_task_text(index=index + 100_000, target_chars=max(220, target_chars // 2))
    right = _default_task_text(index=index + 120_000, target_chars=max(220, target_chars // 2))
    return [
        {
            "role": "system",
            "content": (
                "You merge political summaries. Preserve policy detail and ideological signals. "
                "Return only the merged summary."
            ),
        },
        {
            "role": "user",
            "content": (
                "Merge these two summaries into one coherent 2-4 sentence summary. "
                "Preserve all policy commitments and left-right cues.\n\n"
                f"SUMMARY A:\n{left}\n\n"
                f"SUMMARY B:\n{right}\n\n"
                "MERGED SUMMARY:"
            ),
        },
    ]


_RILE_SCORE_CONTEXT = """
Task: Score this political text on the left-right (RILE) scale.
Left indicators include market regulation, welfare expansion, labor support.
Right indicators include free enterprise, welfare limitation, traditional values.
Score range: -100 (far left) to +100 (far right); 0 is centrist.
Output requirement: return exactly one numeric score in [-100, +100].
"""


def _build_task_score_messages(index: int, target_chars: int) -> List[Dict[str, str]]:
    summary = _default_task_text(index=index + 220_000, target_chars=max(320, target_chars // 2))
    return [
        {
            "role": "system",
            "content": (
                "You are an expert CMP manifesto coder. Return exactly one numeric RILE score "
                "between -100 and +100."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{_RILE_SCORE_CONTEXT}\n\n"
                f"SUMMARY:\n{summary}\n\n"
                "Output only the numeric RILE score in [-100, +100]."
            ),
        },
    ]


def _is_valid_rile_score_response(content: str) -> bool:
    """Return True if response contains a parseable score in [-100, 100]."""
    return (
        parse_numeric_score(
            str(content or ""),
            min_value=-100.0,
            max_value=100.0,
            allow_llm_fallback=False,
        )
        is not None
    )


async def _run_task_single_prompt_point(
    *,
    step_name: str,
    request_type: str,
    base_url: str,
    concurrency: int,
    total_requests: int,
    request_timeout_seconds: float,
    max_tokens: int,
    batch_timeout: float,
    api_key: str,
    task_chars: int,
    message_builder: Callable[[int, int], List[Dict[str, str]]],
    validate_response: Optional[Callable[[str], bool]] = None,
) -> SweepPoint:
    """Shared implementation for single-endpoint task-model benchmark points."""
    latencies: List[float] = []
    timeout_errors = 0
    network_errors = 0
    server_errors = 0
    parse_errors = 0
    request_ids: List[str] = []

    start = time.perf_counter()
    async with AsyncBatchLLMClient(
        base_url=base_url,
        max_concurrent=concurrency,
        batch_timeout=batch_timeout,
        request_timeout=request_timeout_seconds,
        api_key=api_key,
    ) as client:
        for i in range(total_requests):
            request_id = f"{step_name}_{concurrency}_{i}_{uuid.uuid4().hex[:8]}"
            request = BatchRequest(
                request_id=request_id,
                messages=message_builder(i, task_chars),
                max_tokens=max_tokens,
                temperature=0.3,
                request_type=request_type,
            )
            await client.submit(request)
            request_ids.append(request_id)

        for rid in request_ids:
            response = await client.await_response(
                rid, timeout=max(60.0, request_timeout_seconds + 60.0)
            )
            if response.latency_ms > 0:
                latencies.append(float(response.latency_ms))
            if response.error:
                t, n, s = _classify_error(response.error)
                timeout_errors += t
                network_errors += n
                server_errors += s
                continue
            if validate_response is not None and not validate_response(response.content):
                parse_errors += 1

        client.stats.wall_clock_end = time.time()
        stats = client.stats

    wall = max(1e-9, time.perf_counter() - start)
    successful_requests = max(0, int(stats.completed_requests) - int(parse_errors))
    failed_requests = max(
        0,
        int(total_requests) - int(successful_requests),
    )
    return SweepPoint(
        step=step_name,
        concurrency=concurrency,
        total_requests=total_requests,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        timeout_errors=int(timeout_errors),
        network_errors=int(network_errors),
        server_errors=int(server_errors),
        parse_errors=int(parse_errors),
        wall_seconds=wall,
        requests_per_second=float(successful_requests) / wall,
        prompt_tokens=int(stats.prompt_tokens),
        completion_tokens=int(stats.completion_tokens),
        total_tokens=int(stats.total_tokens),
        tokens_per_second=float(stats.total_tokens) / wall,
        latency_avg_ms=(sum(latencies) / len(latencies)) if latencies else 0.0,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
    )


def _build_genrm_request(index: int) -> GenRMComparisonRequest:
    original = _default_task_text(index=index + 10_000, target_chars=700)
    summary_a = (
        "Supports labor standards, climate-aligned growth, affordable housing, and stronger "
        "public services while balancing fiscal constraints."
    )
    summary_b = (
        "Random phrase stack with weak policy grounding and missing key economic and social details."
    )
    return GenRMComparisonRequest(
        request_id=f"genrm_{index}_{uuid.uuid4().hex[:10]}",
        context="Preserve major policy commitments and ideological direction.",
        original_text=original,
        summary_a=summary_a,
        summary_b=summary_b,
        law_type="sufficiency",
    )


@dataclass
class SweepPoint:
    """Metrics for one step at one concurrency value."""

    step: str
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    timeout_errors: int
    network_errors: int
    server_errors: int
    wall_seconds: float
    requests_per_second: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_per_second: float
    latency_avg_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    parse_errors: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_requests <= 0:
            return 0.0
        return float(self.successful_requests) / float(self.total_requests)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["success_rate"] = self.success_rate
        return data


@dataclass
class StepSummary:
    """Summary stats and recommended settings for one sweep step."""

    step: str
    stable_points: int
    recommended_concurrency: Optional[int]
    recommended_req_per_s: float
    max_stable_concurrency: Optional[int]
    max_stable_req_per_s: float
    peak_req_per_s_concurrency: int
    peak_req_per_s: float
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepSweepResult:
    """Full sweep output for one step."""

    step: str
    points: List[SweepPoint]
    summary: StepSummary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "summary": self.summary.to_dict(),
            "points": [p.to_dict() for p in self.points],
        }


def summarize_step(
    step: str,
    points: Sequence[SweepPoint],
    min_success_rate: float,
    max_p95_latency_ms: float,
) -> StepSummary:
    """Choose stable/recommended operating points from sweep results."""
    if not points:
        return StepSummary(
            step=step,
            stable_points=0,
            recommended_concurrency=None,
            recommended_req_per_s=0.0,
            max_stable_concurrency=None,
            max_stable_req_per_s=0.0,
            peak_req_per_s_concurrency=0,
            peak_req_per_s=0.0,
            notes="No data points",
        )

    stable: List[SweepPoint] = []
    for p in points:
        latency_ok = True
        if max_p95_latency_ms > 0.0:
            latency_ok = p.latency_p95_ms <= max_p95_latency_ms
        if p.success_rate >= min_success_rate and latency_ok:
            stable.append(p)

    peak = max(points, key=lambda p: p.requests_per_second)

    if stable:
        recommended = max(stable, key=lambda p: p.requests_per_second)
        max_stable = max(stable, key=lambda p: p.concurrency)
        note = "Selected best req/s among stable points"
        return StepSummary(
            step=step,
            stable_points=len(stable),
            recommended_concurrency=recommended.concurrency,
            recommended_req_per_s=recommended.requests_per_second,
            max_stable_concurrency=max_stable.concurrency,
            max_stable_req_per_s=max_stable.requests_per_second,
            peak_req_per_s_concurrency=peak.concurrency,
            peak_req_per_s=peak.requests_per_second,
            notes=note,
        )

    note = "No stable point satisfied thresholds; using peak throughput point"
    return StepSummary(
        step=step,
        stable_points=0,
        recommended_concurrency=peak.concurrency,
        recommended_req_per_s=peak.requests_per_second,
        max_stable_concurrency=None,
        max_stable_req_per_s=0.0,
        peak_req_per_s_concurrency=peak.concurrency,
        peak_req_per_s=peak.requests_per_second,
        notes=note,
    )


def _classify_error(error_text: str) -> Tuple[int, int, int]:
    """Classify an error string into timeout/network/server counters."""
    text = (error_text or "").lower()
    timeout = int("timeout" in text or "timed out" in text)
    network = int(
        ("connectorerror" in text)
        or ("connection reset" in text)
        or ("connection refused" in text)
        or ("network" in text)
    )
    server = 1 if (timeout == 0 and network == 0) else 0
    return timeout, network, server


async def run_task_single_point(
    *,
    base_url: str,
    concurrency: int,
    total_requests: int,
    request_timeout_seconds: float,
    max_tokens: int,
    batch_timeout: float,
    api_key: str,
    task_chars: int,
) -> SweepPoint:
    """Benchmark task-model throughput against a single endpoint."""
    return await _run_task_single_prompt_point(
        step_name="task_single",
        request_type="summarize",
        base_url=base_url,
        concurrency=concurrency,
        total_requests=total_requests,
        request_timeout_seconds=request_timeout_seconds,
        max_tokens=max_tokens,
        batch_timeout=batch_timeout,
        api_key=api_key,
        task_chars=task_chars,
        message_builder=lambda i, chars: _build_task_messages(index=i, target_chars=chars),
    )


async def run_task_merge_point(
    *,
    base_url: str,
    concurrency: int,
    total_requests: int,
    request_timeout_seconds: float,
    max_tokens: int,
    batch_timeout: float,
    api_key: str,
    task_chars: int,
) -> SweepPoint:
    """Benchmark task-model merge-style summarization throughput."""
    return await _run_task_single_prompt_point(
        step_name="task_merge",
        request_type="merge",
        base_url=base_url,
        concurrency=concurrency,
        total_requests=total_requests,
        request_timeout_seconds=request_timeout_seconds,
        max_tokens=max_tokens,
        batch_timeout=batch_timeout,
        api_key=api_key,
        task_chars=task_chars,
        message_builder=lambda i, chars: _build_task_merge_messages(index=i, target_chars=chars),
    )


async def run_task_score_point(
    *,
    base_url: str,
    concurrency: int,
    total_requests: int,
    request_timeout_seconds: float,
    max_tokens: int,
    batch_timeout: float,
    api_key: str,
    task_chars: int,
) -> SweepPoint:
    """Benchmark task-model scorer-style numeric output throughput."""
    return await _run_task_single_prompt_point(
        step_name="task_score",
        request_type="score",
        base_url=base_url,
        concurrency=concurrency,
        total_requests=total_requests,
        request_timeout_seconds=request_timeout_seconds,
        max_tokens=max_tokens,
        batch_timeout=batch_timeout,
        api_key=api_key,
        task_chars=task_chars,
        message_builder=lambda i, chars: _build_task_score_messages(index=i, target_chars=chars),
        validate_response=_is_valid_rile_score_response,
    )


async def run_task_dp2_point(
    *,
    primary_url: str,
    replica_url: str,
    concurrency: int,
    total_requests: int,
    request_timeout_seconds: float,
    max_tokens: int,
    batch_timeout: float,
    api_key: str,
    task_chars: int,
) -> SweepPoint:
    """Benchmark task-model throughput using round-robin load balancing across 2 endpoints."""
    latencies: List[float] = []
    timeout_errors = 0
    network_errors = 0
    server_errors = 0
    request_ids: List[str] = []

    per_server_concurrency = max(1, int(math.ceil(concurrency / 2.0)))
    start = time.perf_counter()
    async with MultiServerBatchClient(
        servers=[primary_url, replica_url],
        max_concurrent_per_server=per_server_concurrency,
        batch_size=50,
        batch_timeout=batch_timeout,
        api_key=api_key,
    ) as client:
        for i in range(total_requests):
            request_id = f"task_dp2_{concurrency}_{i}_{uuid.uuid4().hex[:8]}"
            request = BatchRequest(
                request_id=request_id,
                messages=_build_task_messages(index=i + 50_000, target_chars=task_chars),
                max_tokens=max_tokens,
                temperature=0.3,
            )
            await client.submit(request)
            request_ids.append(request_id)

        for rid in request_ids:
            response = await client.await_response(rid)
            if response.latency_ms > 0:
                latencies.append(float(response.latency_ms))
            if response.error:
                t, n, s = _classify_error(response.error)
                timeout_errors += t
                network_errors += n
                server_errors += s

        stats = client.stats
        stats.wall_clock_end = time.time()

    wall = max(1e-9, time.perf_counter() - start)
    return SweepPoint(
        step="task_dp2",
        concurrency=concurrency,
        total_requests=total_requests,
        successful_requests=int(stats.completed_requests),
        failed_requests=int(stats.failed_requests),
        timeout_errors=int(timeout_errors),
        network_errors=int(network_errors),
        server_errors=int(server_errors),
        wall_seconds=wall,
        requests_per_second=float(stats.completed_requests) / wall,
        prompt_tokens=int(stats.prompt_tokens),
        completion_tokens=int(stats.completion_tokens),
        total_tokens=int(stats.total_tokens),
        tokens_per_second=float(stats.total_tokens) / wall,
        latency_avg_ms=(sum(latencies) / len(latencies)) if latencies else 0.0,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
    )


async def _run_raw_chat_completion(
    *,
    base_url: str,
    model_name: str,
    payload_factory: Callable[[int], Dict[str, Any]],
    concurrency: int,
    total_requests: int,
    timeout_seconds: float,
    api_key: str,
) -> Tuple[List[float], int, int, int, int, int, int, int, int]:
    """Send raw chat/completions requests with fixed concurrency."""
    latencies: List[float] = []
    success_count = 0
    timeout_errors = 0
    network_errors = 0
    server_errors = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    next_index = 0
    index_lock = asyncio.Lock()
    timeout_cfg = aiohttp.ClientTimeout(total=timeout_seconds)
    connector = aiohttp.TCPConnector(limit=max(1, int(concurrency)))

    async def claim_index() -> Optional[int]:
        nonlocal next_index
        async with index_lock:
            if next_index >= total_requests:
                return None
            value = next_index
            next_index += 1
            return value

    async def worker(session: aiohttp.ClientSession) -> None:
        nonlocal success_count, timeout_errors, network_errors, server_errors
        nonlocal prompt_tokens, completion_tokens, total_tokens
        while True:
            idx = await claim_index()
            if idx is None:
                return
            payload = payload_factory(idx)
            payload["model"] = model_name
            start = time.perf_counter()
            try:
                async with session.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                ) as resp:
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    if resp.status == 200:
                        data = await resp.json()
                        usage = data.get("usage", {})
                        success_count += 1
                        latencies.append(latency_ms)
                        prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                        completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                        total_tokens += int(usage.get("total_tokens", 0) or 0)
                    else:
                        _ = await resp.text()
                        server_errors += 1
            except asyncio.TimeoutError:
                timeout_errors += 1
            except aiohttp.ClientError:
                network_errors += 1
            except Exception:
                server_errors += 1

    async with aiohttp.ClientSession(timeout=timeout_cfg, connector=connector) as session:
        workers = [asyncio.create_task(worker(session)) for _ in range(max(1, concurrency))]
        await asyncio.gather(*workers)

    failed = total_requests - success_count
    return (
        latencies,
        success_count,
        failed,
        timeout_errors,
        network_errors,
        server_errors,
        prompt_tokens,
        completion_tokens,
        total_tokens,
    )


async def run_genrm_raw_point(
    *,
    base_url: str,
    concurrency: int,
    total_requests: int,
    request_timeout_seconds: float,
    max_tokens: int,
    temperature: float,
    top_p: float,
    disable_thinking: bool,
    force_json_response: bool,
    api_key: str,
) -> SweepPoint:
    """Benchmark GenRM endpoint with raw chat/completions calls."""
    model_name = await detect_model_async(base_url, fallback="default", timeout=10.0)

    def payload_factory(index: int) -> Dict[str, Any]:
        req = _build_genrm_request(index=index + 90_000)
        payload: Dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Compare candidate summaries and return JSON with keys "
                        "score_1, score_2, ranking.\n"
                        f"Context: {req.context}\n"
                        f"Original Text:\n{req.original_text}"
                    ),
                },
                {"role": "response_1", "content": req.summary_a},
                {"role": "response_2", "content": req.summary_b},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if force_json_response:
            payload["response_format"] = {"type": "json_object"}
        return payload

    start = time.perf_counter()
    (
        latencies,
        success_count,
        failed_count,
        timeout_errors,
        network_errors,
        server_errors,
        prompt_tokens,
        completion_tokens,
        total_tokens,
    ) = await _run_raw_chat_completion(
        base_url=base_url,
        model_name=model_name,
        payload_factory=payload_factory,
        concurrency=concurrency,
        total_requests=total_requests,
        timeout_seconds=request_timeout_seconds,
        api_key=api_key,
    )
    wall = max(1e-9, time.perf_counter() - start)
    return SweepPoint(
        step="genrm_raw",
        concurrency=concurrency,
        total_requests=total_requests,
        successful_requests=success_count,
        failed_requests=failed_count,
        timeout_errors=timeout_errors,
        network_errors=network_errors,
        server_errors=server_errors,
        wall_seconds=wall,
        requests_per_second=float(success_count) / wall,
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
        total_tokens=int(total_tokens),
        tokens_per_second=float(total_tokens) / wall if total_tokens > 0 else 0.0,
        latency_avg_ms=(sum(latencies) / len(latencies)) if latencies else 0.0,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
    )


async def run_genrm_batch_point(
    *,
    base_url: str,
    concurrency: int,
    total_requests: int,
    request_timeout_seconds: float,
    max_tokens: int,
    temperature: float,
    top_p: float,
    disable_thinking: bool,
    force_json_response: bool,
) -> SweepPoint:
    """Benchmark GenRM throughput using AsyncBatchGenRMClient (pipeline path)."""
    latencies: List[float] = []
    timeout_errors = 0
    network_errors = 0
    server_errors = 0
    success_count = 0
    failed_count = 0

    client = AsyncBatchGenRMClient(
        base_url=base_url,
        max_concurrent=concurrency,
        request_timeout=request_timeout_seconds,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        disable_thinking=disable_thinking,
        force_json_response=force_json_response,
    )

    async def one_call(index: int):
        request = _build_genrm_request(index=index + 120_000)
        started = time.perf_counter()
        result = await client.call(request)
        return result, (time.perf_counter() - started) * 1000.0

    start = time.perf_counter()
    async with client:
        tasks = [asyncio.create_task(one_call(i)) for i in range(total_requests)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for item in responses:
            if isinstance(item, Exception):
                failed_count += 1
                t, n, s = _classify_error(str(item))
                timeout_errors += t
                network_errors += n
                server_errors += s
                continue
            result, latency_ms = item
            latencies.append(latency_ms)
            if is_genrm_error(result):
                failed_count += 1
                t, n, s = _classify_error(getattr(result, "error_message", ""))
                timeout_errors += t
                network_errors += n
                server_errors += s
            else:
                success_count += 1
        stats = client.stats
    wall = max(1e-9, time.perf_counter() - start)

    # Cross-check failures against client stats if needed.
    if failed_count == 0 and int(stats.failed_requests) > 0:
        failed_count = int(stats.failed_requests)
        success_count = max(0, total_requests - failed_count)

    return SweepPoint(
        step="genrm_batch",
        concurrency=concurrency,
        total_requests=total_requests,
        successful_requests=success_count,
        failed_requests=failed_count,
        timeout_errors=timeout_errors,
        network_errors=network_errors,
        server_errors=server_errors,
        wall_seconds=wall,
        requests_per_second=float(success_count) / wall,
        prompt_tokens=int(stats.prompt_tokens),
        completion_tokens=int(stats.completion_tokens),
        total_tokens=int(stats.total_tokens),
        tokens_per_second=float(stats.total_tokens) / wall if stats.total_tokens > 0 else 0.0,
        latency_avg_ms=(sum(latencies) / len(latencies)) if latencies else 0.0,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
    )


async def run_step_sweep(
    *,
    step_name: str,
    concurrencies: Sequence[int],
    total_requests_for_concurrency: Callable[[int], int],
    runner: Callable[[int, int], Awaitable[SweepPoint]],
    warmup_requests: int = 0,
) -> List[SweepPoint]:
    """Run sweep across a concurrency grid for one step."""
    if warmup_requests > 0:
        warmup_concurrency = int(concurrencies[0])
        logger.info(
            "[%s] warmup: concurrency=%d requests=%d",
            step_name,
            warmup_concurrency,
            warmup_requests,
        )
        try:
            await runner(warmup_concurrency, warmup_requests)
        except Exception as exc:
            logger.warning("[%s] warmup failed (continuing): %s", step_name, exc)

    points: List[SweepPoint] = []
    for concurrency in concurrencies:
        total_requests = int(total_requests_for_concurrency(concurrency))
        logger.info(
            "[%s] sweep: concurrency=%d requests=%d",
            step_name,
            int(concurrency),
            int(total_requests),
        )
        point = await runner(int(concurrency), int(total_requests))
        points.append(point)
        logger.info(
            "[%s] c=%d ok=%d/%d req/s=%.2f tok/s=%.1f p95=%.1fms",
            step_name,
            point.concurrency,
            point.successful_requests,
            point.total_requests,
            point.requests_per_second,
            point.tokens_per_second,
            point.latency_p95_ms,
        )
    return points


async def run_pipeline_throughput_suite(
    *,
    steps: Sequence[str],
    concurrency_grid: Sequence[int],
    min_requests_per_point: int,
    requests_per_concurrency: int,
    warmup_requests: int,
    task_url: str,
    task_replica_url: Optional[str],
    genrm_url: str,
    task_timeout_seconds: float,
    genrm_timeout_seconds: float,
    task_max_tokens: int,
    genrm_max_tokens: int,
    task_batch_timeout: float,
    task_chars: int,
    api_key: str,
    genrm_disable_thinking: bool,
    genrm_force_json_response: bool,
    genrm_temperature: float,
    genrm_top_p: float,
    min_success_rate: float,
    max_p95_latency_ms: float,
) -> Dict[str, StepSweepResult]:
    """Run selected step sweeps and return structured results."""
    results: Dict[str, StepSweepResult] = {}

    def requests_for_point(concurrency: int) -> int:
        return max(int(min_requests_per_point), int(concurrency) * int(requests_per_concurrency))

    for step in steps:
        if step == "task_single":
            async def _runner(conc: int, nreq: int) -> SweepPoint:
                return await run_task_single_point(
                    base_url=task_url,
                    concurrency=conc,
                    total_requests=nreq,
                    request_timeout_seconds=task_timeout_seconds,
                    max_tokens=task_max_tokens,
                    batch_timeout=task_batch_timeout,
                    api_key=api_key,
                    task_chars=task_chars,
                )
        elif step == "task_merge":
            async def _runner(conc: int, nreq: int) -> SweepPoint:
                return await run_task_merge_point(
                    base_url=task_url,
                    concurrency=conc,
                    total_requests=nreq,
                    request_timeout_seconds=task_timeout_seconds,
                    max_tokens=task_max_tokens,
                    batch_timeout=task_batch_timeout,
                    api_key=api_key,
                    task_chars=task_chars,
                )
        elif step == "task_score":
            async def _runner(conc: int, nreq: int) -> SweepPoint:
                return await run_task_score_point(
                    base_url=task_url,
                    concurrency=conc,
                    total_requests=nreq,
                    request_timeout_seconds=task_timeout_seconds,
                    max_tokens=task_max_tokens,
                    batch_timeout=task_batch_timeout,
                    api_key=api_key,
                    task_chars=task_chars,
                )
        elif step == "task_dp2":
            if not task_replica_url:
                raise ValueError("Step 'task_dp2' requires --task-replica-url")

            async def _runner(conc: int, nreq: int) -> SweepPoint:
                return await run_task_dp2_point(
                    primary_url=task_url,
                    replica_url=task_replica_url,
                    concurrency=conc,
                    total_requests=nreq,
                    request_timeout_seconds=task_timeout_seconds,
                    max_tokens=task_max_tokens,
                    batch_timeout=task_batch_timeout,
                    api_key=api_key,
                    task_chars=task_chars,
                )
        elif step == "genrm_raw":
            async def _runner(conc: int, nreq: int) -> SweepPoint:
                return await run_genrm_raw_point(
                    base_url=genrm_url,
                    concurrency=conc,
                    total_requests=nreq,
                    request_timeout_seconds=genrm_timeout_seconds,
                    max_tokens=genrm_max_tokens,
                    temperature=genrm_temperature,
                    top_p=genrm_top_p,
                    disable_thinking=genrm_disable_thinking,
                    force_json_response=genrm_force_json_response,
                    api_key=api_key,
                )
        elif step.startswith("genrm_raw_"):
            mode = step[len("genrm_raw_") :]
            mode_cfg = GENRM_MODE_CONFIGS.get(mode)
            if mode_cfg is None:
                raise ValueError(
                    f"Unknown step variant '{step}'. Valid GenRM modes: {sorted(GENRM_MODE_CONFIGS.keys())}"
                )

            async def _runner(conc: int, nreq: int) -> SweepPoint:
                point = await run_genrm_raw_point(
                    base_url=genrm_url,
                    concurrency=conc,
                    total_requests=nreq,
                    request_timeout_seconds=genrm_timeout_seconds,
                    max_tokens=genrm_max_tokens,
                    temperature=genrm_temperature,
                    top_p=genrm_top_p,
                    disable_thinking=bool(mode_cfg["disable_thinking"]),
                    force_json_response=bool(mode_cfg["force_json_response"]),
                    api_key=api_key,
                )
                point.step = step
                return point
        elif step == "genrm_batch":
            async def _runner(conc: int, nreq: int) -> SweepPoint:
                return await run_genrm_batch_point(
                    base_url=genrm_url,
                    concurrency=conc,
                    total_requests=nreq,
                    request_timeout_seconds=genrm_timeout_seconds,
                    max_tokens=genrm_max_tokens,
                    temperature=genrm_temperature,
                    top_p=genrm_top_p,
                    disable_thinking=genrm_disable_thinking,
                    force_json_response=genrm_force_json_response,
                )
        elif step.startswith("genrm_batch_"):
            mode = step[len("genrm_batch_") :]
            mode_cfg = GENRM_MODE_CONFIGS.get(mode)
            if mode_cfg is None:
                raise ValueError(
                    f"Unknown step variant '{step}'. Valid GenRM modes: {sorted(GENRM_MODE_CONFIGS.keys())}"
                )

            async def _runner(conc: int, nreq: int) -> SweepPoint:
                point = await run_genrm_batch_point(
                    base_url=genrm_url,
                    concurrency=conc,
                    total_requests=nreq,
                    request_timeout_seconds=genrm_timeout_seconds,
                    max_tokens=genrm_max_tokens,
                    temperature=genrm_temperature,
                    top_p=genrm_top_p,
                    disable_thinking=bool(mode_cfg["disable_thinking"]),
                    force_json_response=bool(mode_cfg["force_json_response"]),
                )
                point.step = step
                return point
        else:
            raise ValueError(f"Unknown step: {step}")

        points = await run_step_sweep(
            step_name=step,
            concurrencies=concurrency_grid,
            total_requests_for_concurrency=requests_for_point,
            runner=_runner,
            warmup_requests=warmup_requests,
        )
        summary = summarize_step(
            step=step,
            points=points,
            min_success_rate=min_success_rate,
            max_p95_latency_ms=max_p95_latency_ms,
        )
        results[step] = StepSweepResult(step=step, points=points, summary=summary)

    return results


def format_human_summary(result: Dict[str, StepSweepResult]) -> str:
    """Format a readable summary table for CLI output."""
    lines: List[str] = []
    lines.append("=" * 92)
    lines.append("Pipeline Throughput Sweep Summary")
    lines.append("=" * 92)
    for step_name, sweep in result.items():
        summary = sweep.summary
        lines.append(
            f"[{step_name}] recommended_c={summary.recommended_concurrency} "
            f"recommended_req/s={summary.recommended_req_per_s:.2f} "
            f"max_stable_c={summary.max_stable_concurrency} "
            f"peak_c={summary.peak_req_per_s_concurrency} peak_req/s={summary.peak_req_per_s:.2f}"
        )
        lines.append(f"  note: {summary.notes}")
    lines.append("")
    for step_name, sweep in result.items():
        lines.append(f"-- {step_name} --")
        lines.append("c  req  ok  fail  parse  success%  req/s  tok/s  p50ms  p95ms")
        for point in sweep.points:
            lines.append(
                f"{point.concurrency:>2} "
                f"{point.total_requests:>4} "
                f"{point.successful_requests:>4} "
                f"{point.failed_requests:>4} "
                f"{point.parse_errors:>5} "
                f"{100.0 * point.success_rate:>8.2f} "
                f"{point.requests_per_second:>6.2f} "
                f"{point.tokens_per_second:>7.1f} "
                f"{point.latency_p50_ms:>6.1f} "
                f"{point.latency_p95_ms:>6.1f}"
            )
        lines.append("")
    return "\n".join(lines)


def _ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_suite_json(
    *,
    output_json: Path,
    config: Dict[str, Any],
    result: Dict[str, StepSweepResult],
) -> None:
    """Write full sweep results to JSON."""
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "steps": {name: sweep.to_dict() for name, sweep in result.items()},
    }
    _ensure_output_dir(output_json)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_suite_csv(*, output_csv: Path, result: Dict[str, StepSweepResult]) -> None:
    """Write per-point metrics to CSV for plotting or spreadsheet work."""
    _ensure_output_dir(output_csv)
    headers = [
        "step",
        "concurrency",
        "total_requests",
        "successful_requests",
        "failed_requests",
        "success_rate",
        "timeout_errors",
        "network_errors",
        "server_errors",
        "parse_errors",
        "wall_seconds",
        "requests_per_second",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "tokens_per_second",
        "latency_avg_ms",
        "latency_p50_ms",
        "latency_p95_ms",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for step_name, sweep in result.items():
            for point in sweep.points:
                row = point.to_dict()
                row["step"] = step_name
                writer.writerow(row)


def default_output_path(prefix: str = "throughput_limits") -> Path:
    """Create default output path under outputs/ with a UTC timestamp."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"{prefix}_{stamp}.json"
