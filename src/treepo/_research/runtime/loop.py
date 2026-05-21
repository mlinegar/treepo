from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from treepo._research.runtime.adapters.base import BenchmarkAdapter
from treepo._research.runtime.backbone import BackboneAdapter
from treepo._research.runtime.contracts import ProblemSpec, RunUnit, RuntimeConfig
from treepo._research.runtime.memory import TokenCounter, chunk_text_tokens, pairwise
from treepo._research.runtime.methods import MethodResources, run_runtime_method
from treepo._research.runtime.repair import SimpleRepairPolicy
from treepo._research.runtime.trace import PredictionRecord, StepEvent, TraceWriter
from treepo._research.runtime.verifier import DeterministicVerifier


def _extract_context_and_question(problem: ProblemSpec) -> Tuple[str, str]:
    text = problem.input_text or ""
    task_type = str(problem.metadata.get("ruler_task_type", "") or "")

    if task_type == "niah":
        marker = "\nWhat "
        if marker in text:
            before, after = text.split(marker, 1)
            return before, "What " + after.strip()
        return text, ""

    if task_type in {"variable_tracking", "common_words_extraction", "freq_words_extraction"}:
        marker = "\nQuestion:"
        if marker in text:
            before, after = text.split(marker, 1)
            return before, after.strip()
        return text, ""

    if task_type == "qa":
        marker = "\n\nQuestion:"
        if marker in text:
            before, after = text.split(marker, 1)
            return before, after.strip()
        marker2 = "\nQuestion:"
        if marker2 in text:
            before, after = text.split(marker2, 1)
            return before, after.strip()
        return text, ""

    return text, problem.query or ""


def _require_backbone(backbone: Optional[BackboneAdapter], method: str) -> BackboneAdapter:
    if backbone is None:
        raise RuntimeError(f"Method {method!r} requires an LLM backbone, but none was configured.")
    return backbone


@dataclass
class UnitMetrics:
    primary_metric: str
    n_problems: int
    n_failures: int
    mean_score: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_calls: int
    wall_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def run_unit(
    *,
    unit: RunUnit,
    run_dir: Path,
    adapter: BenchmarkAdapter,
    runtime: RuntimeConfig,
    trace: TraceWriter,
    backbone: Optional[BackboneAdapter],
    resources: Optional[MethodResources] = None,
    counter: Optional[TokenCounter] = None,
    limit_problems: Optional[int] = None,
) -> UnitMetrics:
    counter = counter or TokenCounter()
    verifier = DeterministicVerifier(counter)
    repair = SimpleRepairPolicy()

    limit = int(limit_problems) if limit_problems is not None else int(unit.num_samples)
    problems = list(adapter.load_split(unit.split, limit=limit))
    primary = adapter.primary_metric()

    unit_dir = run_dir / "units" / unit.unit_id
    unit_dir.mkdir(parents=True, exist_ok=True)
    _write_json(unit_dir / "unit.json", unit.to_dict())
    _write_json(unit_dir / "runtime_config.json", runtime.to_dict())

    total_score = 0.0
    n_scored = 0
    n_failures = 0

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_calls = 0
    wall_start = __import__("time").time()
    method_resources = resources or MethodResources(backbone=backbone)
    inference_context = getattr(method_resources, "inference_context", None)

    for problem in problems:
        previous_scope: Dict[str, Any] = {}
        if inference_context is not None and hasattr(inference_context, "set_call_scope"):
            try:
                previous_scope = inference_context.set_call_scope(
                    run_id=unit.run_id,
                    experiment_id=unit.run_id,
                    unit_id=unit.unit_id,
                    method_id=unit.method,
                    problem_id=problem.problem_id,
                    metadata={
                        "benchmark": unit.benchmark,
                        "task_id": unit.task_id,
                        "phase_id": unit.phase_id,
                    },
                )
            except Exception:
                previous_scope = {}
        try:
            method_result = run_runtime_method(
                unit=unit,
                problem=problem,
                adapter=adapter,
                runtime=runtime,
                resources=method_resources,
                counter=counter,
                verifier=verifier,
                repair=repair,
            )
            pred_text = method_result.prediction
            cost = dict(method_result.cost)
            step_events = list(method_result.steps)
            # Emit steps.
            for ev in step_events:
                trace.write_step(unit.unit_id, ev)

            # Score.
            metrics = adapter.score(problem, {"prediction": pred_text})
            score = float(metrics.get(primary, 0.0))
            total_score += score
            n_scored += 1

            total_prompt_tokens += int(cost.get("prompt_tokens", 0))
            total_completion_tokens += int(cost.get("completion_tokens", 0))
            total_calls += int(cost.get("n_calls", 0))

            trace.write_prediction(
                unit.unit_id,
                PredictionRecord(
                    run_id=unit.run_id,
                    unit_id=unit.unit_id,
                    phase_id=unit.phase_id,
                    benchmark=unit.benchmark,
                    task_id=unit.task_id,
                    split=unit.split,
                    max_seq_length=unit.max_seq_length,
                    seed=unit.seed,
                    method=unit.method,
                    primary_metric=primary,
                    problem_id=problem.problem_id,
                    prediction=pred_text,
                    references=list(problem.references),
                    metrics=metrics,
                    cost=cost,
                    metadata={
                        "problem": {
                            key: value
                            for key, value in dict(problem.metadata).items()
                            if key not in {"context", "input", "prompt"}
                        },
                        "artifacts": dict(method_result.artifacts),
                    },
                    failure=None,
                ),
            )
        except Exception as e:
            n_failures += 1
            trace.write_prediction(
                unit.unit_id,
                PredictionRecord(
                    run_id=unit.run_id,
                    unit_id=unit.unit_id,
                    phase_id=unit.phase_id,
                    benchmark=unit.benchmark,
                    task_id=unit.task_id,
                    split=unit.split,
                    max_seq_length=unit.max_seq_length,
                    seed=unit.seed,
                    method=unit.method,
                    primary_metric=primary,
                    problem_id=problem.problem_id,
                    prediction="",
                    references=list(problem.references),
                    metrics={primary: 0.0},
                    cost={},
                    metadata={
                        "problem": {
                            key: value
                            for key, value in dict(problem.metadata).items()
                            if key not in {"context", "input", "prompt"}
                        }
                    },
                    failure={"error_type": type(e).__name__, "message": str(e)},
                ),
            )
        finally:
            if inference_context is not None and hasattr(inference_context, "set_call_scope"):
                try:
                    inference_context.set_call_scope(**previous_scope)
                except Exception:
                    pass

    wall_ms = (__import__("time").time() - wall_start) * 1000.0
    mean_score = total_score / max(1, n_scored)

    metrics = UnitMetrics(
        primary_metric=primary,
        n_problems=len(problems),
        n_failures=n_failures,
        mean_score=mean_score,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_calls=total_calls,
        wall_ms=wall_ms,
    )
    _write_json(unit_dir / "metrics_partial.json", metrics.to_dict())
    return metrics


def _run_problem(
    *,
    problem: ProblemSpec,
    adapter: BenchmarkAdapter,
    unit: RunUnit,
    runtime: RuntimeConfig,
    backbone: Optional[BackboneAdapter],
    counter: TokenCounter,
    verifier: DeterministicVerifier,
    repair: SimpleRepairPolicy,
) -> Tuple[str, Dict[str, Any], List[StepEvent]]:
    method = runtime.method

    answer_prefix = str(problem.metadata.get("answer_prefix", "") or "")
    full_prompt = (problem.input_text or "") + answer_prefix

    def max_input_tokens_for(output_tokens: int) -> int:
        return max(1, runtime.cap_tokens - runtime.safety_tokens - int(output_tokens))

    step_events: List[StepEvent] = []
    prompt_tokens_total = 0
    completion_tokens_total = 0
    n_calls = 0
    wall_ms = 0.0

    def log_call(
        *,
        step_idx: int,
        node_id: str,
        action_type: str,
        input_tokens: int,
        output_tokens: int,
        verifier_pass: bool,
        failures: List[str],
        repair_action: str,
        latency_ms: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        step_events.append(
            StepEvent(
                run_id=unit.run_id,
                unit_id=unit.unit_id,
                problem_id=problem.problem_id,
                step_idx=step_idx,
                node_id=node_id,
                action_type=action_type,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                verifier_pass=bool(verifier_pass),
                failure_codes=list(failures),
                repair_action=repair_action,
                latency_ms=float(latency_ms),
                extra=extra or {},
            )
        )

    if method == "ground_truth":
        pred = " ".join(problem.references)
        return (
            pred,
            {"n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "wall_ms": 0.0},
            step_events,
        )

    if method == "empty":
        return (
            "",
            {"n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "wall_ms": 0.0},
            step_events,
        )

    if method == "flat_prompt_baseline":
        bb = _require_backbone(backbone, method)
        prompt = counter.truncate_tokens(
            full_prompt, max_tokens=max_input_tokens_for(runtime.max_output_tokens), keep="tail"
        )
        messages = [{"role": "user", "content": prompt}]

        resp = bb.generate(messages, max_tokens=runtime.max_output_tokens)
        n_calls += 1
        wall_ms += resp.latency_ms
        prompt_tokens_total += resp.prompt_tokens or counter.count(prompt)
        completion_tokens_total += resp.completion_tokens or counter.count(resp.text)

        v = (
            verifier.check(
                contract=adapter.build_contract(problem),
                response=resp,
                max_output_tokens=runtime.max_output_tokens,
            )
            if runtime.verifier_enabled
            else None
        )
        passed = True if v is None else v.pass_
        failures = [] if v is None else list(v.failures)
        log_call(
            step_idx=0,
            node_id="root",
            action_type="flat_prompt",
            input_tokens=counter.count(prompt),
            output_tokens=counter.count(resp.text),
            verifier_pass=passed,
            failures=failures,
            repair_action="",
            latency_ms=resp.latency_ms,
        )

        if (not passed) and runtime.repair_enabled:
            repaired = repair.apply(
                backbone=bb,
                messages=messages,
                contract=adapter.build_contract(problem),
                verifier=v
                or verifier.check(contract=adapter.build_contract(problem), response=resp),
                max_tokens=runtime.max_output_tokens,
            )
            n_calls += 1
            wall_ms += repaired.latency_ms
            prompt_tokens_total += repaired.prompt_tokens
            completion_tokens_total += repaired.completion_tokens
            v2 = verifier.check(
                contract=adapter.build_contract(problem),
                response=repaired,
                max_output_tokens=runtime.max_output_tokens,
            )
            log_call(
                step_idx=1,
                node_id="root",
                action_type="flat_prompt_repair",
                input_tokens=counter.count(prompt),
                output_tokens=counter.count(repaired.text),
                verifier_pass=v2.pass_,
                failures=list(v2.failures),
                repair_action="retry",
                latency_ms=repaired.latency_ms,
            )
            resp = repaired

        pred = resp.text.strip()
        return (
            pred,
            {
                "n_calls": n_calls,
                "prompt_tokens": prompt_tokens_total,
                "completion_tokens": completion_tokens_total,
                "wall_ms": wall_ms,
            },
            step_events,
        )

    if method in {
        "chunk_concat_baseline",
        "runtime_no_verifier",
        "runtime_no_repair",
        "runtime_full",
    }:
        bb = _require_backbone(backbone, method)

        # Apply ablation toggles.
        verifier_enabled = runtime.verifier_enabled
        repair_enabled = runtime.repair_enabled
        if method == "runtime_no_verifier":
            verifier_enabled = False
        if method == "runtime_no_repair":
            repair_enabled = False

        context, question = _extract_context_and_question(problem)
        chunks = chunk_text_tokens(
            context,
            counter=counter,
            chunk_tokens=runtime.chunk_tokens,
            overlap_tokens=runtime.overlap_tokens,
        )

        system = (
            "You are writing a short memory to help answer a question from a long document. "
            "Do not answer the question yet. Preserve rare strings (UUIDs, numbers) exactly."
        )

        engram_cfg = None
        engram_extract = None
        engram_format = None
        if getattr(runtime, "engram_memory", False):
            system = (
                system.rstrip()
                + "\n- Preserve any STATIC MEMORY items exactly if they appear.\n"
                + "- Do not output the STATIC MEMORY list.\n"
            )
            try:
                from treepo._research.core.engram_memory import (
                    EngramMemoryConfig,
                    extract_engram_memory_items,
                    format_engram_memory_block,
                )

                engram_cfg = EngramMemoryConfig(
                    enabled=True,
                    max_items=int(getattr(runtime, "engram_memory_max_items", 32) or 32),
                    max_chars=int(getattr(runtime, "engram_memory_max_chars", 800) or 800),
                )
                engram_extract = extract_engram_memory_items
                engram_format = format_engram_memory_block
            except Exception:
                engram_cfg = None
                engram_extract = None
                engram_format = None

        def summarize_leaf(chunk_text: str, step_idx: int, node_id: str) -> str:
            nonlocal n_calls, wall_ms, prompt_tokens_total, completion_tokens_total
            memory_block = ""
            if engram_cfg is not None and engram_extract is not None and engram_format is not None:
                try:
                    memory_items = engram_extract(chunk_text, engram_cfg)
                    memory_block = engram_format(memory_items)
                except Exception:
                    memory_block = ""

            user = (
                f"Question:\n{question}\n\n"
                f"Write a concise memory (<= {runtime.leaf_memory_tokens} tokens) capturing only facts useful to answer.\n\n"
                f"Chunk:\n{chunk_text}\n"
            )
            if memory_block:
                user = user.rstrip() + "\n\n" + memory_block + "\n"
            user = counter.truncate_tokens(
                user, max_tokens=max_input_tokens_for(runtime.leaf_memory_tokens), keep="tail"
            )
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            resp = bb.generate(msgs, max_tokens=runtime.leaf_memory_tokens, temperature=0.0)
            n_calls += 1
            wall_ms += resp.latency_ms
            prompt_tokens_total += resp.prompt_tokens or counter.count(user)
            completion_tokens_total += resp.completion_tokens or counter.count(resp.text)

            v = (
                verifier.check(
                    contract=adapter.build_contract(problem),
                    response=resp,
                    max_output_tokens=runtime.leaf_memory_tokens,
                )
                if verifier_enabled
                else None
            )
            passed = True if v is None else v.pass_
            failures = [] if v is None else list(v.failures)
            log_call(
                step_idx=step_idx,
                node_id=node_id,
                action_type="leaf_summarize",
                input_tokens=counter.count(user),
                output_tokens=counter.count(resp.text),
                verifier_pass=passed,
                failures=failures,
                repair_action="",
                latency_ms=resp.latency_ms,
            )

            if (not passed) and repair_enabled:
                repaired = repair.apply(
                    backbone=bb,
                    messages=msgs,
                    contract=adapter.build_contract(problem),
                    verifier=v
                    or verifier.check(contract=adapter.build_contract(problem), response=resp),
                    max_tokens=runtime.leaf_memory_tokens,
                )
                n_calls += 1
                wall_ms += repaired.latency_ms
                prompt_tokens_total += repaired.prompt_tokens
                completion_tokens_total += repaired.completion_tokens
                v2 = verifier.check(
                    contract=adapter.build_contract(problem),
                    response=repaired,
                    max_output_tokens=runtime.leaf_memory_tokens,
                )
                log_call(
                    step_idx=step_idx + 1,
                    node_id=node_id,
                    action_type="leaf_repair",
                    input_tokens=counter.count(user),
                    output_tokens=counter.count(repaired.text),
                    verifier_pass=v2.pass_,
                    failures=list(v2.failures),
                    repair_action="retry",
                    latency_ms=repaired.latency_ms,
                )
                resp = repaired

            mem = counter.truncate_tokens(
                resp.text.strip(), max_tokens=runtime.leaf_memory_tokens, keep="head"
            )
            return mem

        def merge_memory(a: str, b: str, step_idx: int, node_id: str) -> str:
            nonlocal n_calls, wall_ms, prompt_tokens_total, completion_tokens_total
            memory_block = ""
            if engram_cfg is not None and engram_extract is not None and engram_format is not None:
                try:
                    memory_items = engram_extract(f"{a}\n\n{b}", engram_cfg)
                    memory_block = engram_format(memory_items)
                except Exception:
                    memory_block = ""

            user = (
                f"Question:\n{question}\n\n"
                f"Merge the two memories into a single concise memory (<= {runtime.merge_memory_tokens} tokens). "
                "Preserve rare strings exactly.\n\n"
                f"Memory A:\n{a}\n\nMemory B:\n{b}\n"
            )
            if memory_block:
                user = user.rstrip() + "\n\n" + memory_block + "\n"
            user = counter.truncate_tokens(
                user, max_tokens=max_input_tokens_for(runtime.merge_memory_tokens), keep="tail"
            )
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            resp = bb.generate(msgs, max_tokens=runtime.merge_memory_tokens, temperature=0.0)
            n_calls += 1
            wall_ms += resp.latency_ms
            prompt_tokens_total += resp.prompt_tokens or counter.count(user)
            completion_tokens_total += resp.completion_tokens or counter.count(resp.text)

            v = (
                verifier.check(
                    contract=adapter.build_contract(problem),
                    response=resp,
                    max_output_tokens=runtime.merge_memory_tokens,
                )
                if verifier_enabled
                else None
            )
            passed = True if v is None else v.pass_
            failures = [] if v is None else list(v.failures)
            log_call(
                step_idx=step_idx,
                node_id=node_id,
                action_type="merge_summarize",
                input_tokens=counter.count(user),
                output_tokens=counter.count(resp.text),
                verifier_pass=passed,
                failures=failures,
                repair_action="",
                latency_ms=resp.latency_ms,
            )

            if (not passed) and repair_enabled:
                repaired = repair.apply(
                    backbone=bb,
                    messages=msgs,
                    contract=adapter.build_contract(problem),
                    verifier=v
                    or verifier.check(contract=adapter.build_contract(problem), response=resp),
                    max_tokens=runtime.merge_memory_tokens,
                )
                n_calls += 1
                wall_ms += repaired.latency_ms
                prompt_tokens_total += repaired.prompt_tokens
                completion_tokens_total += repaired.completion_tokens
                v2 = verifier.check(
                    contract=adapter.build_contract(problem),
                    response=repaired,
                    max_output_tokens=runtime.merge_memory_tokens,
                )
                log_call(
                    step_idx=step_idx + 1,
                    node_id=node_id,
                    action_type="merge_repair",
                    input_tokens=counter.count(user),
                    output_tokens=counter.count(repaired.text),
                    verifier_pass=v2.pass_,
                    failures=list(v2.failures),
                    repair_action="retry",
                    latency_ms=repaired.latency_ms,
                )
                resp = repaired

            mem = counter.truncate_tokens(
                resp.text.strip(), max_tokens=runtime.merge_memory_tokens, keep="head"
            )
            return mem

        # Leaf stage.
        memories: List[str] = []
        step_idx = 0
        for i, chunk in enumerate(chunks):
            mem = summarize_leaf(chunk, step_idx=step_idx, node_id=f"leaf_{i:04d}")
            memories.append(mem)
            step_idx += 2 if (verifier_enabled and repair_enabled) else 1

        # Combine stage.
        if method == "chunk_concat_baseline":
            combined = "\n\n".join(memories)
            combined = counter.truncate_tokens(
                combined, max_tokens=runtime.merge_memory_tokens, keep="head"
            )
            root_memory = combined
        else:
            level = 0
            while len(memories) > 1:
                next_level: List[str] = []
                for j, (a, b) in enumerate(pairwise(memories)):
                    if b is None:
                        next_level.append(a)
                        continue
                    merged = merge_memory(
                        a, b, step_idx=step_idx, node_id=f"merge_L{level}_{j:04d}"
                    )
                    next_level.append(merged)
                    step_idx += 2 if (verifier_enabled and repair_enabled) else 1
                memories = next_level
                level += 1
            root_memory = memories[0] if memories else ""

        # Answer stage.
        answer_system = "You answer questions using the provided memory. Output only the answer."
        answer_user = (
            f"Question:\n{question}\n\nMemory:\n{root_memory}\n\n"
            f"{answer_prefix.strip() if answer_prefix.strip() else 'Answer:'}"
        )
        answer_user = counter.truncate_tokens(
            answer_user, max_tokens=max_input_tokens_for(runtime.max_output_tokens), keep="tail"
        )
        answer_msgs = [
            {"role": "system", "content": answer_system},
            {"role": "user", "content": answer_user},
        ]
        answer_resp = bb.generate(
            answer_msgs, max_tokens=runtime.max_output_tokens, temperature=0.0
        )
        n_calls += 1
        wall_ms += answer_resp.latency_ms
        prompt_tokens_total += answer_resp.prompt_tokens or counter.count(answer_user)
        completion_tokens_total += answer_resp.completion_tokens or counter.count(answer_resp.text)

        v_ans = (
            verifier.check(
                contract=adapter.build_contract(problem),
                response=answer_resp,
                max_output_tokens=runtime.max_output_tokens,
            )
            if verifier_enabled
            else None
        )
        passed = True if v_ans is None else v_ans.pass_
        failures = [] if v_ans is None else list(v_ans.failures)
        log_call(
            step_idx=step_idx,
            node_id="answer",
            action_type="answer",
            input_tokens=counter.count(answer_user),
            output_tokens=counter.count(answer_resp.text),
            verifier_pass=passed,
            failures=failures,
            repair_action="",
            latency_ms=answer_resp.latency_ms,
        )

        pred = answer_resp.text.strip()
        return (
            pred,
            {
                "n_calls": n_calls,
                "prompt_tokens": prompt_tokens_total,
                "completion_tokens": completion_tokens_total,
                "wall_ms": wall_ms,
            },
            step_events,
        )

    raise ValueError(f"Unknown runtime method: {method}")
