"""
Component-level microbenchmarks for ThinkingTrees pipeline building blocks.

These benchmarks are pure-Python and intended for quick local performance
tracking without requiring live model servers.
"""

from __future__ import annotations

import statistics
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from treepo._research.core.conditional_memory import ConditionalMemory, ConditionalMemoryConfig
from treepo._research.core.prompting import (
    default_merge_prompt,
    default_summarize_prompt,
    default_unified_prompt,
    parse_numeric_score,
)
from treepo._research.preprocessing.chunker import chunk_for_ops
from treepo._research.tree.builder import IdentitySummarizer, build as build_tree_sync


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * float(q)
    lo = int(rank)
    hi = min(len(ordered) - 1, lo + 1)
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _synthetic_text(index: int, target_chars: int) -> str:
    blocks = [
        "Labour standards, collective bargaining, and wage growth remain central goals.",
        "Climate transition policy should align clean energy deployment with industrial strategy.",
        "Public transport and affordable housing can improve long-term productivity.",
        "Tax and welfare policy should balance fiscal discipline with social protection.",
        "Education and training investment improves labor-market resilience.",
        "Trade openness should be paired with domestic worker adjustment support.",
        "Healthcare capacity expansion reduces inequality and improves social mobility.",
        "Institutional trust depends on transparency, anti-corruption, and enforcement.",
    ]
    text: List[str] = []
    i = 0
    while len(" ".join(text)) < max(512, int(target_chars)):
        text.append(blocks[(index + i) % len(blocks)])
        text.append(f"[doc={index} part={i}]")
        i += 1
    return " ".join(text)


@dataclass(frozen=True)
class MicrobenchResult:
    name: str
    wall_seconds: float
    metadata: Dict[str, Any]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def bench_chunker(
    *,
    docs: int = 200,
    target_chars: int = 8000,
    max_chunk_chars: int = 2000,
) -> MicrobenchResult:
    latencies_ms: List[float] = []
    chunk_counts: List[int] = []
    token_counts: List[int] = []

    start = time.perf_counter()
    for i in range(max(1, int(docs))):
        text = _synthetic_text(i, target_chars=target_chars)
        t0 = time.perf_counter()
        chunks = chunk_for_ops(text, max_chars=max_chunk_chars, strategy="axis")
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        chunk_counts.append(len(chunks))
        token_counts.append(
            sum(int(c.token_count or max(1, len(c.text) // 4)) for c in chunks)
        )
    wall = max(1e-9, time.perf_counter() - start)
    docs_per_second = float(docs) / wall

    return MicrobenchResult(
        name="chunker",
        wall_seconds=wall,
        metadata={
            "docs": int(docs),
            "target_chars": int(target_chars),
            "max_chunk_chars": int(max_chunk_chars),
        },
        metrics={
            "docs_per_second": docs_per_second,
            "latency_ms_avg": float(statistics.fmean(latencies_ms)) if latencies_ms else 0.0,
            "latency_ms_p50": _percentile(latencies_ms, 0.50),
            "latency_ms_p95": _percentile(latencies_ms, 0.95),
            "chunks_per_doc_avg": float(statistics.fmean(chunk_counts)) if chunk_counts else 0.0,
            "chunks_per_doc_p95": _percentile(chunk_counts, 0.95),
            "tokens_per_doc_avg": float(statistics.fmean(token_counts)) if token_counts else 0.0,
        },
    )


def bench_conditional_memory(
    *,
    iterations: int = 1000,
    text_pool: int = 256,
) -> MicrobenchResult:
    iterations = max(1, int(iterations))
    text_pool = max(8, int(text_pool))
    pool = [_synthetic_text(i, target_chars=1200) for i in range(text_pool)]
    write_latencies_ms: List[float] = []
    lookup_latencies_ms: List[float] = []
    hit_count = 0

    with tempfile.TemporaryDirectory(prefix="tt_micro_mem_") as td:
        cfg = ConditionalMemoryConfig(
            enabled=True,
            mode="readwrite",
            root_dir=Path(td),
            l2_path=Path("microbench.db"),
            l1_capacity=max(64, text_pool // 2),
        )
        memory = ConditionalMemory(cfg)
        try:
            # Warm a subset.
            for i in range(text_pool // 2):
                memory.store(pool[i], scores={"oracle": float((i % 101) / 100.0)})

            t_start = time.perf_counter()
            for i in range(iterations):
                text = pool[i % text_pool]
                score_val = float(((i * 37) % 201) - 100) / 100.0
                t0 = time.perf_counter()
                memory.store(text, scores={"oracle": score_val}, metadata={"i": i})
                write_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

                t1 = time.perf_counter()
                rec = memory.lookup(text, score_heads=["oracle"])
                lookup_latencies_ms.append((time.perf_counter() - t1) * 1000.0)
                if rec is not None:
                    hit_count += 1
            wall = max(1e-9, time.perf_counter() - t_start)
            report = memory.report()
        finally:
            memory.close()

    return MicrobenchResult(
        name="conditional_memory",
        wall_seconds=wall,
        metadata={
            "iterations": int(iterations),
            "text_pool": int(text_pool),
        },
        metrics={
            "ops_per_second": float(iterations * 2) / wall,
            "write_latency_ms_avg": float(statistics.fmean(write_latencies_ms))
            if write_latencies_ms
            else 0.0,
            "lookup_latency_ms_avg": float(statistics.fmean(lookup_latencies_ms))
            if lookup_latencies_ms
            else 0.0,
            "lookup_latency_ms_p95": _percentile(lookup_latencies_ms, 0.95),
            "lookup_hit_rate_runtime": float(hit_count) / float(iterations),
            "report": report,
        },
    )


def bench_prompting(
    *,
    iterations: int = 2000,
) -> MicrobenchResult:
    iterations = max(1, int(iterations))
    prompt_latencies_ms: List[float] = []
    parse_latencies_ms: List[float] = []
    parse_ok = 0

    start = time.perf_counter()
    for i in range(iterations):
        text = _synthetic_text(i, target_chars=1100)
        left = _synthetic_text(i + 10_000, target_chars=650)
        right = _synthetic_text(i + 20_000, target_chars=650)
        rubric = "Preserve policy detail and left-right ideological direction."

        t0 = time.perf_counter()
        _ = default_summarize_prompt(text, rubric)
        _ = default_merge_prompt(left, right, rubric)
        _ = default_unified_prompt(text, rubric)
        prompt_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        # Mix strict and noisy outputs to exercise parser paths.
        raw = "42" if i % 3 == 0 else ("score: -12.5" if i % 3 == 1 else "not a score")
        t1 = time.perf_counter()
        parsed = parse_numeric_score(raw, min_value=-100, max_value=100, allow_llm_fallback=False)
        parse_latencies_ms.append((time.perf_counter() - t1) * 1000.0)
        if parsed is not None:
            parse_ok += 1
    wall = max(1e-9, time.perf_counter() - start)

    return MicrobenchResult(
        name="prompting",
        wall_seconds=wall,
        metadata={"iterations": int(iterations)},
        metrics={
            "iterations_per_second": float(iterations) / wall,
            "prompt_build_latency_ms_avg": float(statistics.fmean(prompt_latencies_ms))
            if prompt_latencies_ms
            else 0.0,
            "prompt_build_latency_ms_p95": _percentile(prompt_latencies_ms, 0.95),
            "parse_latency_ms_avg": float(statistics.fmean(parse_latencies_ms))
            if parse_latencies_ms
            else 0.0,
            "parse_latency_ms_p95": _percentile(parse_latencies_ms, 0.95),
            "parse_success_rate": float(parse_ok) / float(iterations),
        },
    )


def bench_tree_builder(
    *,
    docs: int = 40,
    target_chars: int = 7000,
    max_chunk_chars: int = 1800,
) -> MicrobenchResult:
    docs = max(1, int(docs))
    latencies_ms: List[float] = []
    heights: List[int] = []
    leaves: List[int] = []
    nodes: List[int] = []
    summarizer = IdentitySummarizer()

    start = time.perf_counter()
    for i in range(docs):
        text = _synthetic_text(i, target_chars=target_chars)
        t0 = time.perf_counter()
        tree = build_tree_sync(
            text=text,
            rubric="Preserve all information.",
            summarizer=summarizer,
            max_chars=max_chunk_chars,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        heights.append(int(tree.height))
        leaves.append(int(tree.leaf_count))
        nodes.append(int(tree.node_count))
    wall = max(1e-9, time.perf_counter() - start)

    return MicrobenchResult(
        name="tree_builder",
        wall_seconds=wall,
        metadata={
            "docs": int(docs),
            "target_chars": int(target_chars),
            "max_chunk_chars": int(max_chunk_chars),
            "summarizer": "IdentitySummarizer",
        },
        metrics={
            "docs_per_second": float(docs) / wall,
            "latency_ms_avg": float(statistics.fmean(latencies_ms)) if latencies_ms else 0.0,
            "latency_ms_p95": _percentile(latencies_ms, 0.95),
            "height_avg": float(statistics.fmean(heights)) if heights else 0.0,
            "leaves_avg": float(statistics.fmean(leaves)) if leaves else 0.0,
            "nodes_avg": float(statistics.fmean(nodes)) if nodes else 0.0,
        },
    )


_BENCH_REGISTRY = {
    "chunker": bench_chunker,
    "conditional_memory": bench_conditional_memory,
    "prompting": bench_prompting,
    "tree_builder": bench_tree_builder,
}


def available_benchmarks() -> List[str]:
    return sorted(_BENCH_REGISTRY.keys())


def run_selected_benchmarks(
    bench_names: Optional[Iterable[str]] = None,
    *,
    iterations: int = 1000,
) -> Dict[str, Any]:
    selected = list(bench_names or [])
    if not selected or selected == ["all"]:
        selected = available_benchmarks()

    unknown = [name for name in selected if name not in _BENCH_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown benchmark(s): {unknown}. Available: {available_benchmarks()}"
        )

    results: Dict[str, Any] = {}
    for name in selected:
        fn = _BENCH_REGISTRY[name]
        if name in {"prompting", "conditional_memory"}:
            result = fn(iterations=iterations)
        else:
            # Keep docs proportional to iterations but bounded for quick local runs.
            docs = max(8, min(400, int(max(1, iterations) // 4)))
            result = fn(docs=docs)
        results[name] = result.to_dict()

    return {
        "created_at": _utc_now_iso(),
        "benchmarks": results,
    }
