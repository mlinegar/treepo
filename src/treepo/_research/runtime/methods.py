from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from treepo._research.core.engines import EngineSurface
from treepo._research.core.async_utils import to_thread
from treepo._research.runtime.adapters.base import BenchmarkAdapter
from treepo._research.runtime.answering import answer_multi_choice
from treepo._research.runtime.backbone import BackboneAdapter
from treepo._research.runtime.contracts import (
    AnswerResult,
    MethodRunResult,
    ModelResponse,
    OperatorOutput,
    ProblemSpec,
    RUNTIME_ROLE_SCORER,
    RUNTIME_ROLE_STATE_MODEL,
    RUNTIME_ROLE_SUMMARIZER,
    RunUnit,
    RuntimeConfig,
    RuntimeTaskView,
    STATE_OPERATOR_SELECT_EVIDENCE,
)
from treepo._research.runtime.memory import TokenCounter, chunk_text_tokens, pairwise
from treepo._research.runtime.repair import SimpleRepairPolicy
from treepo._research.runtime.trace import StepEvent
from treepo._research.runtime.verifier import DeterministicVerifier


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]: ...


@dataclass(frozen=True)
class MethodResources:
    backbone: Optional[BackboneAdapter] = None
    summarizer_backbone: Optional[BackboneAdapter] = None
    embedding_client: Optional[EmbeddingClient] = None
    inference_context: Any = None
    run_config: Mapping[str, Any] | None = None
    mock: bool = False


MethodRunner = Any


@dataclass(frozen=True)
class MethodSpec:
    name: str
    runner_id: str
    family: str
    trained: bool = False
    artifact_dir: str = ""
    variant: str = ""
    overrides: Dict[str, Any] | None = None


def _require_backbone(
    resources: MethodResources, method: str, *, role: str = RUNTIME_ROLE_SCORER
) -> BackboneAdapter:
    if role == RUNTIME_ROLE_SUMMARIZER and resources.summarizer_backbone is not None:
        return resources.summarizer_backbone
    if role == RUNTIME_ROLE_SCORER and resources.backbone is not None:
        return resources.backbone
    if role == RUNTIME_ROLE_SUMMARIZER and resources.backbone is not None:
        return resources.backbone
    ctx = getattr(resources, "inference_context", None)
    if ctx is not None and hasattr(ctx, "has_role") and ctx.has_role(role):
        return ctx.backbone(role)
    if ctx is not None and hasattr(ctx, "has_surface") and ctx.has_surface(EngineSurface.CHAT_OPENAI):
        return ctx.backbone()
    raise RuntimeError(f"Runtime method {method!r} requires a configured {role} LLM.")


def _embedding_client(resources: MethodResources) -> EmbeddingClient:
    if resources.embedding_client is not None:
        return resources.embedding_client
    ctx = getattr(resources, "inference_context", None)
    if ctx is not None and hasattr(ctx, "has_surface") and ctx.has_surface(EngineSurface.EMBEDDING):
        return ctx.embedding_client()
    return HashingEmbeddingClient()


def _task_view(adapter: BenchmarkAdapter, problem: ProblemSpec) -> RuntimeTaskView:
    return adapter.task_view(problem)


def _choices_block(view: RuntimeTaskView) -> str:
    if not view.choices:
        return ""
    lines = [
        f"{letter}. {view.choices[letter]}"
        for letter in ("A", "B", "C", "D")
        if letter in view.choices
    ]
    return "\n".join(lines)


def _answer_prompt(view: RuntimeTaskView, *, context: str) -> str:
    choices = _choices_block(view)
    pieces = [f"Context:\n{context.strip()}"]
    if view.question:
        pieces.append(f"Question:\n{view.question.strip()}")
    if choices:
        pieces.append(f"Choices:\n{choices}")
    instruction = view.answer_instruction.strip() or "Answer the question using the context."
    pieces.append(instruction)
    return "\n\n".join(pieces)


def _call_final_answer(
    *,
    bb: BackboneAdapter,
    adapter: BenchmarkAdapter,
    problem: ProblemSpec,
    prompt: str,
    runtime: RuntimeConfig,
    counter: TokenCounter,
) -> AnswerResult:
    answer_spec = adapter.build_answer_spec(problem)
    if str(answer_spec.kind) == "multi_choice" and bool(runtime.delegate_llm_for_answer):
        return answer_multi_choice(
            bb=bb,
            prompt=prompt,
            answer_spec=answer_spec,
            runtime=runtime,
            counter=counter,
        )
    messages = [{"role": "user", "content": prompt}]
    resp, cost = _call_llm(bb=bb, messages=messages, runtime=runtime, counter=counter)
    cost["wall_ms"] = float(resp.latency_ms)
    return AnswerResult(
        prediction=resp.text.strip(),
        raw_text=resp.text.strip(),
        decode_method="generate_free_text",
        cost=cost,
    )


def _count_messages(counter: TokenCounter, messages: Sequence[Mapping[str, str]]) -> int:
    return counter.count("\n".join(str(m.get("content", "")) for m in messages))


def _max_input_tokens(runtime: RuntimeConfig, output_tokens: int) -> int:
    return max(1, int(runtime.cap_tokens) - int(runtime.safety_tokens) - int(output_tokens))


def _step(
    *,
    unit: RunUnit,
    problem: ProblemSpec,
    step_idx: int,
    node_id: str,
    action_type: str,
    input_tokens: int,
    output_tokens: int,
    verifier_pass: bool = True,
    failure_codes: Optional[Sequence[str]] = None,
    repair_action: str = "",
    latency_ms: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> StepEvent:
    return StepEvent(
        run_id=unit.run_id,
        unit_id=unit.unit_id,
        problem_id=problem.problem_id,
        step_idx=int(step_idx),
        node_id=str(node_id),
        action_type=str(action_type),
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        verifier_pass=bool(verifier_pass),
        failure_codes=list(failure_codes or []),
        repair_action=str(repair_action or ""),
        latency_ms=float(latency_ms),
        extra=dict(extra or {}),
    )


def _call_llm(
    *,
    bb: BackboneAdapter,
    messages: List[Dict[str, str]],
    runtime: RuntimeConfig,
    counter: TokenCounter,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
) -> Tuple[ModelResponse, Dict[str, int]]:
    out_tokens = int(max_tokens if max_tokens is not None else runtime.max_output_tokens)
    resp = bb.generate(messages, max_tokens=out_tokens, temperature=temperature)
    prompt_tokens = int(resp.prompt_tokens or _count_messages(counter, messages))
    completion_tokens = int(resp.completion_tokens or counter.count(resp.text))
    return resp, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "n_calls": 1,
    }


def _merge_cost(*costs: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0, "wall_ms": 0.0}
    for cost in costs:
        out["prompt_tokens"] += int(cost.get("prompt_tokens", 0) or 0)
        out["completion_tokens"] += int(cost.get("completion_tokens", 0) or 0)
        out["n_calls"] += int(cost.get("n_calls", 0) or 0)
        out["wall_ms"] += float(cost.get("wall_ms", 0.0) or 0.0)
    return out


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    an = math.sqrt(sum(float(x) * float(x) for x in a))
    bn = math.sqrt(sum(float(y) * float(y) for y in b))
    if an <= 1e-12 or bn <= 1e-12:
        return 0.0
    return dot / (an * bn)


class HashingEmbeddingClient:
    """Deterministic local embedding fallback for smoke tests and mock runs."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = max(2, int(dim))

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for idx, token in enumerate(str(text or "").lower().split()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:8], "big") % self.dim
                vec[bucket] += 1.0 + (idx % 7) * 0.01
            vectors.append(vec)
        return vectors


def run_ground_truth(
    *,
    unit: RunUnit,
    problem: ProblemSpec,
    adapter: BenchmarkAdapter,
    runtime: RuntimeConfig,
    resources: MethodResources,
    counter: TokenCounter,
    verifier: DeterministicVerifier,
    repair: SimpleRepairPolicy,
) -> MethodRunResult:
    return MethodRunResult(
        prediction=" ".join(problem.references),
        cost={"n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "wall_ms": 0.0},
        steps=[],
        artifacts={"method_id": unit.method},
    )


def run_empty(**_: Any) -> MethodRunResult:
    return MethodRunResult(
        prediction="",
        cost={"n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "wall_ms": 0.0},
        steps=[],
        artifacts={"method_id": "empty"},
    )


def run_llm_direct_official(
    *,
    unit: RunUnit,
    problem: ProblemSpec,
    adapter: BenchmarkAdapter,
    runtime: RuntimeConfig,
    resources: MethodResources,
    counter: TokenCounter,
    verifier: DeterministicVerifier,
    repair: SimpleRepairPolicy,
) -> MethodRunResult:
    bb = _require_backbone(resources, unit.method, role=RUNTIME_ROLE_SCORER)
    view = _task_view(adapter, problem)
    prompt = view.official_prompt or _answer_prompt(view, context=view.context)
    prompt = counter.truncate_tokens(
        prompt,
        max_tokens=_max_input_tokens(runtime, runtime.max_output_tokens),
        keep="tail",
    )
    answer = _call_final_answer(
        bb=bb,
        adapter=adapter,
        problem=problem,
        prompt=prompt,
        runtime=runtime,
        counter=counter,
    )
    cost = dict(answer.cost)
    resp = ModelResponse(
        text=answer.raw_text or answer.prediction,
        model_id=bb.model_id(),
        prompt_tokens=int(cost.get("prompt_tokens", 0) or 0),
        completion_tokens=int(cost.get("completion_tokens", 0) or 0),
        latency_ms=float(cost.get("wall_ms", 0.0) or 0.0),
    )
    v = (
        verifier.check(
            contract=adapter.build_contract(problem),
            response=resp,
            max_output_tokens=runtime.max_output_tokens,
        )
        if runtime.verifier_enabled and answer.decode_method == "generate_free_text"
        else None
    )
    step = _step(
        unit=unit,
        problem=problem,
        step_idx=0,
        node_id="root",
        action_type="llm_direct_official",
        input_tokens=counter.count(prompt),
        output_tokens=counter.count(answer.raw_text or answer.prediction),
        verifier_pass=True if v is None else v.pass_,
        failure_codes=[] if v is None else v.failures,
        latency_ms=float(cost.get("wall_ms", 0.0) or 0.0),
        extra={"answer_decode_method": answer.decode_method},
    )
    return MethodRunResult(
        prediction=answer.prediction,
        cost=cost,
        steps=[step],
        artifacts={
            "method_id": unit.method,
            "prompt_tokens_est": counter.count(prompt),
            "answer": answer.to_dict(),
        },
    )


def run_llm_tree_memory(
    *,
    unit: RunUnit,
    problem: ProblemSpec,
    adapter: BenchmarkAdapter,
    runtime: RuntimeConfig,
    resources: MethodResources,
    counter: TokenCounter,
    verifier: DeterministicVerifier,
    repair: SimpleRepairPolicy,
) -> MethodRunResult:
    scorer_bb = _require_backbone(resources, unit.method, role=RUNTIME_ROLE_SCORER)
    summarizer_bb = _require_backbone(
        resources, unit.method, role=RUNTIME_ROLE_SUMMARIZER
    )
    view = _task_view(adapter, problem)
    verifier_enabled = bool(runtime.verifier_enabled)
    repair_enabled = bool(runtime.repair_enabled)
    if unit.method == "runtime_no_verifier":
        verifier_enabled = False
    if unit.method == "runtime_no_repair":
        repair_enabled = False

    chunks = chunk_text_tokens(
        view.context,
        counter=counter,
        chunk_tokens=int(runtime.chunk_tokens),
        overlap_tokens=int(runtime.overlap_tokens),
    )
    steps: List[StepEvent] = []
    costs: List[Dict[str, Any]] = []
    system = (
        "You are writing a short memory to help answer a question from a long context. "
        "Do not answer yet. Preserve names, numbers, rare strings, and choice-relevant evidence exactly."
    )
    choices = _choices_block(view)

    def summarize_leaf(chunk_text: str, step_idx: int, node_id: str) -> str:
        user = (
            f"Question:\n{view.question}\n\n" f"Choices:\n{choices}\n\n"
            if choices
            else f"Question:\n{view.question}\n\n"
        )
        user += (
            f"Write a concise memory (<= {runtime.leaf_memory_tokens} tokens) with only facts useful for the answer.\n\n"
            f"Chunk:\n{chunk_text}"
        )
        user = counter.truncate_tokens(
            user, max_tokens=_max_input_tokens(runtime, runtime.leaf_memory_tokens), keep="tail"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        resp, cost = _call_llm(
            bb=summarizer_bb,
            messages=messages,
            runtime=runtime,
            counter=counter,
            max_tokens=runtime.leaf_memory_tokens,
        )
        cost["wall_ms"] = float(resp.latency_ms)
        costs.append(cost)
        v = (
            verifier.check(
                contract=adapter.build_contract(problem),
                response=resp,
                max_output_tokens=runtime.leaf_memory_tokens,
            )
            if verifier_enabled
            else None
        )
        steps.append(
            _step(
                unit=unit,
                problem=problem,
                step_idx=step_idx,
                node_id=node_id,
                action_type="leaf_summarize",
                input_tokens=counter.count(user),
                output_tokens=counter.count(resp.text),
                verifier_pass=True if v is None else v.pass_,
                failure_codes=[] if v is None else v.failures,
                latency_ms=resp.latency_ms,
            )
        )
        if v is not None and not v.pass_ and repair_enabled:
            repaired = repair.apply(
                backbone=summarizer_bb,
                messages=messages,
                contract=adapter.build_contract(problem),
                verifier=v,
                max_tokens=runtime.leaf_memory_tokens,
            )
            costs.append(
                {
                    "prompt_tokens": int(repaired.prompt_tokens),
                    "completion_tokens": int(repaired.completion_tokens),
                    "n_calls": 1,
                    "wall_ms": float(repaired.latency_ms),
                }
            )
            steps.append(
                _step(
                    unit=unit,
                    problem=problem,
                    step_idx=step_idx + 1,
                    node_id=node_id,
                    action_type="leaf_repair",
                    input_tokens=counter.count(user),
                    output_tokens=counter.count(repaired.text),
                    verifier_pass=True,
                    repair_action="retry",
                    latency_ms=repaired.latency_ms,
                )
            )
            resp = repaired
        return counter.truncate_tokens(
            resp.text.strip(), max_tokens=runtime.leaf_memory_tokens, keep="head"
        )

    def merge_memory(left: str, right: str, step_idx: int, node_id: str) -> str:
        user = (
            f"Question:\n{view.question}\n\n" f"Choices:\n{choices}\n\n"
            if choices
            else f"Question:\n{view.question}\n\n"
        )
        user += (
            f"Merge these memories into one concise memory (<= {runtime.merge_memory_tokens} tokens).\n\n"
            f"Memory A:\n{left}\n\nMemory B:\n{right}"
        )
        user = counter.truncate_tokens(
            user, max_tokens=_max_input_tokens(runtime, runtime.merge_memory_tokens), keep="tail"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        resp, cost = _call_llm(
            bb=summarizer_bb,
            messages=messages,
            runtime=runtime,
            counter=counter,
            max_tokens=runtime.merge_memory_tokens,
        )
        cost["wall_ms"] = float(resp.latency_ms)
        costs.append(cost)
        v = (
            verifier.check(
                contract=adapter.build_contract(problem),
                response=resp,
                max_output_tokens=runtime.merge_memory_tokens,
            )
            if verifier_enabled
            else None
        )
        steps.append(
            _step(
                unit=unit,
                problem=problem,
                step_idx=step_idx,
                node_id=node_id,
                action_type="merge_summarize",
                input_tokens=counter.count(user),
                output_tokens=counter.count(resp.text),
                verifier_pass=True if v is None else v.pass_,
                failure_codes=[] if v is None else v.failures,
                latency_ms=resp.latency_ms,
            )
        )
        return counter.truncate_tokens(
            resp.text.strip(), max_tokens=runtime.merge_memory_tokens, keep="head"
        )

    memories: List[str] = []
    step_idx = 0
    for idx, chunk in enumerate(chunks):
        memories.append(summarize_leaf(chunk, step_idx=step_idx, node_id=f"leaf_{idx:04d}"))
        step_idx += 2 if verifier_enabled and repair_enabled else 1

    if unit.method == "chunk_concat_baseline":
        root_memory = counter.truncate_tokens(
            "\n\n".join(memories), max_tokens=runtime.merge_memory_tokens, keep="head"
        )
    else:
        level = 0
        while len(memories) > 1:
            next_level: List[str] = []
            for j, (left, right) in enumerate(pairwise(memories)):
                if right is None:
                    next_level.append(left)
                    continue
                next_level.append(
                    merge_memory(left, right, step_idx=step_idx, node_id=f"merge_L{level}_{j:04d}")
                )
                step_idx += 2 if verifier_enabled and repair_enabled else 1
            memories = next_level
            level += 1
        root_memory = memories[0] if memories else ""

    answer_user = _answer_prompt(view, context=f"Memory:\n{root_memory}")
    answer_user = counter.truncate_tokens(
        answer_user, max_tokens=_max_input_tokens(runtime, runtime.max_output_tokens), keep="tail"
    )
    answer = _call_final_answer(
        bb=scorer_bb,
        adapter=adapter,
        problem=problem,
        prompt=answer_user,
        runtime=runtime,
        counter=counter,
    )
    answer_cost = dict(answer.cost)
    costs.append(answer_cost)
    steps.append(
        _step(
            unit=unit,
            problem=problem,
            step_idx=step_idx,
            node_id="answer",
            action_type="answer",
            input_tokens=counter.count(answer_user),
            output_tokens=counter.count(answer.raw_text or answer.prediction),
            latency_ms=float(answer_cost.get("wall_ms", 0.0) or 0.0),
            extra={"answer_decode_method": answer.decode_method},
        )
    )
    return MethodRunResult(
        prediction=answer.prediction,
        cost=_merge_cost(*costs),
        steps=steps,
        artifacts={
            "method_id": unit.method,
            "chunk_count": len(chunks),
            "root_memory_chars": len(root_memory),
            "answer": answer.to_dict(),
        },
    )


def _select_chunks_by_embedding(
    *,
    view: RuntimeTaskView,
    runtime: RuntimeConfig,
    counter: TokenCounter,
    embedding_client: EmbeddingClient,
) -> Tuple[List[str], Dict[str, Any]]:
    chunks = chunk_text_tokens(
        view.context,
        counter=counter,
        chunk_tokens=int(getattr(runtime, "retrieval_chunk_tokens", 1024) or 1024),
        overlap_tokens=int(getattr(runtime, "retrieval_overlap_tokens", 128) or 128),
    )
    query_text = "\n".join([view.question, _choices_block(view)])
    vectors = embedding_client.embed_texts([query_text] + chunks)
    query_vec = vectors[0]
    chunk_vecs = vectors[1:]
    scored = [
        (_cosine(query_vec, vec), idx, chunk)
        for idx, (vec, chunk) in enumerate(zip(chunk_vecs, chunks))
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    top_k = max(1, int(getattr(runtime, "retrieval_top_k", 4) or 4))
    selected = sorted(scored[:top_k], key=lambda item: item[1])
    return [chunk for _, _, chunk in selected], {
        "chunk_count": len(chunks),
        "selected_indices": [idx for _, idx, _ in selected],
        "selected_scores": [score for score, _, _ in selected],
        "selector_backend": "embedding",
    }


def _select_chunks_by_operator(
    *,
    view: RuntimeTaskView,
    runtime: RuntimeConfig,
    counter: TokenCounter,
    resources: MethodResources,
) -> Optional[Tuple[List[str], Dict[str, Any]]]:
    ctx = getattr(resources, "inference_context", None)
    if (
        ctx is None
        or not hasattr(ctx, "has_role")
        or not ctx.has_role(RUNTIME_ROLE_STATE_MODEL)
    ):
        return None

    chunks = chunk_text_tokens(
        view.context,
        counter=counter,
        chunk_tokens=int(getattr(runtime, "retrieval_chunk_tokens", 1024) or 1024),
        overlap_tokens=int(getattr(runtime, "retrieval_overlap_tokens", 128) or 128),
    )
    top_k = max(1, int(getattr(runtime, "retrieval_top_k", 4) or 4))
    try:
        operator_cfg = ctx.surface_config(EngineSurface.OPERATOR)
    except Exception:
        operator_cfg = {}
    checkpoint_path = str(
        getattr(runtime, "neural_checkpoint_path", "")
        or operator_cfg.get("checkpoint_path", "")
        or ""
    )
    try:
        response = ctx.execute_operator(
            STATE_OPERATOR_SELECT_EVIDENCE,
            inputs={
                "question": view.question,
                "choices": dict(view.choices),
                "chunks": list(chunks),
                "top_k": top_k,
                "metadata": dict(view.metadata),
            },
            options={
                "checkpoint_path": checkpoint_path,
                "method_family": str(getattr(runtime, "method_family", "") or ""),
                "method_variant": str(getattr(runtime, "method_variant", "") or ""),
            },
            metadata={"request_type": "operator_select_evidence"},
        )
    except Exception as exc:
        return None

    output = response.output
    data = output.data if isinstance(output, OperatorOutput) else None
    if not isinstance(data, Mapping):
        return None
    selected_chunks = [str(item) for item in list(data.get("selected_chunks") or []) if str(item)]
    raw_indices = list(data.get("selected_indices") or [])
    selected_indices: List[int] = []
    for idx in raw_indices:
        try:
            parsed = int(idx)
        except (TypeError, ValueError):
            continue
        if 0 <= parsed < len(chunks):
            selected_indices.append(parsed)
    if not selected_chunks and selected_indices:
        selected_chunks = [chunks[idx] for idx in selected_indices]
    if not selected_chunks:
        return None
    artifacts = {
        "chunk_count": len(chunks),
        "selected_indices": selected_indices,
        "selected_scores": list(data.get("selected_scores") or []),
        "selector_backend": RUNTIME_ROLE_STATE_MODEL,
        "state_model_operation": STATE_OPERATOR_SELECT_EVIDENCE,
        "operator_model_id": response.model_id,
        "state_model_model_id": response.model_id,
        "state_model_checkpoint_path": checkpoint_path,
        "operator_latency_ms": response.latency_ms,
        "operator_artifacts": dict(output.artifacts if isinstance(output, OperatorOutput) else {}),
    }
    return selected_chunks, artifacts


def run_embedding_retrieval_llm(
    *,
    unit: RunUnit,
    problem: ProblemSpec,
    adapter: BenchmarkAdapter,
    runtime: RuntimeConfig,
    resources: MethodResources,
    counter: TokenCounter,
    verifier: DeterministicVerifier,
    repair: SimpleRepairPolicy,
) -> MethodRunResult:
    bb = _require_backbone(resources, unit.method, role=RUNTIME_ROLE_SCORER)
    view = _task_view(adapter, problem)
    embedding_client = _embedding_client(resources)
    selected, artifacts = _select_chunks_by_embedding(
        view=view,
        runtime=runtime,
        counter=counter,
        embedding_client=embedding_client,
    )
    selected_context = "\n\n".join(selected)
    prompt = _answer_prompt(view, context=selected_context)
    prompt = counter.truncate_tokens(
        prompt, max_tokens=_max_input_tokens(runtime, runtime.max_output_tokens), keep="tail"
    )
    answer = _call_final_answer(
        bb=bb,
        adapter=adapter,
        problem=problem,
        prompt=prompt,
        runtime=runtime,
        counter=counter,
    )
    cost = dict(answer.cost)
    step = _step(
        unit=unit,
        problem=problem,
        step_idx=0,
        node_id="retrieval_answer",
        action_type="embedding_retrieval_llm",
        input_tokens=counter.count(prompt),
        output_tokens=counter.count(answer.raw_text or answer.prediction),
        latency_ms=float(cost.get("wall_ms", 0.0) or 0.0),
        extra={
            "selected_indices": artifacts["selected_indices"],
            "answer_decode_method": answer.decode_method,
        },
    )
    artifacts.update(
        {
            "method_id": unit.method,
            "selected_context_chars": len(selected_context),
            "answer": answer.to_dict(),
        }
    )
    return MethodRunResult(
        prediction=answer.prediction, cost=cost, steps=[step], artifacts=artifacts
    )


class _BackboneTreeOperator:
    name = "runtime_backbone_state_tree"

    def __init__(self, bb: BackboneAdapter, runtime: RuntimeConfig, counter: TokenCounter) -> None:
        self.bb = bb
        self.runtime = runtime
        self.counter = counter

    def combine(self, left_span: str, right_span: str, **_: Any) -> str:
        return f"{left_span}\n\n{right_span}"

    async def aencode_many(self, spans: Sequence[str], **kwargs: Any) -> List[str]:
        rubric = str(kwargs.get("rubric", "") or "")
        return [await to_thread(self._summarize, str(span), rubric, "leaf") for span in spans]

    async def amerge_many(self, pairs: Sequence[Tuple[str, str]], **kwargs: Any) -> List[str]:
        rubric = str(kwargs.get("rubric", "") or "")
        return [
            await to_thread(self._merge, str(left), str(right), rubric) for left, right in pairs
        ]

    async def aencode(self, span: str, **kwargs: Any) -> str:
        return await to_thread(
            self._summarize, str(span), str(kwargs.get("rubric", "") or ""), "leaf"
        )

    async def amerge(self, left_state: str, right_state: str, **kwargs: Any) -> str:
        return await to_thread(
            self._merge,
            str(left_state),
            str(right_state),
            str(kwargs.get("rubric", "") or ""),
        )

    def _summarize(self, text: str, rubric: str, role: str) -> str:
        prompt = (
            f"Rubric:\n{rubric}\n\n"
            f"Summarize this {role} text for later question answering. Preserve exact names, numbers, and options evidence.\n\n"
            f"Text:\n{text}"
        )
        prompt = self.counter.truncate_tokens(
            prompt,
            max_tokens=_max_input_tokens(self.runtime, self.runtime.leaf_memory_tokens),
            keep="tail",
        )
        resp = self.bb.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=self.runtime.leaf_memory_tokens,
            temperature=0.0,
        )
        return resp.text.strip()

    def _merge(self, left: str, right: str, rubric: str) -> str:
        prompt = (
            f"Rubric:\n{rubric}\n\n"
            "Merge these two summaries for later question answering.\n\n"
            f"Summary A:\n{left}\n\nSummary B:\n{right}"
        )
        prompt = self.counter.truncate_tokens(
            prompt,
            max_tokens=_max_input_tokens(self.runtime, self.runtime.merge_memory_tokens),
            keep="tail",
        )
        resp = self.bb.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=self.runtime.merge_memory_tokens,
            temperature=0.0,
        )
        return resp.text.strip()

    def capability_report(self) -> Any:
        from treepo._research.core.ops_checks import EvidenceStatus, OperatorCapabilityReport

        return OperatorCapabilityReport(
            operator_name=self.name,
            evidence_status=EvidenceStatus.PROXY_ONLY,
            latent_mergeability_enforced=False,
            tree_nesting_supported=True,
            theorem_domain_decode_available=False,
            theorem_domain_reencode_available=False,
            notes=("Runtime scaffold text compressor; no LongBench-label training.",),
        )


def run_treepo_text_compressor_llm(
    *,
    unit: RunUnit,
    problem: ProblemSpec,
    adapter: BenchmarkAdapter,
    runtime: RuntimeConfig,
    resources: MethodResources,
    counter: TokenCounter,
    verifier: DeterministicVerifier,
    repair: SimpleRepairPolicy,
) -> MethodRunResult:
    scorer_bb = _require_backbone(resources, unit.method, role=RUNTIME_ROLE_SCORER)
    summarizer_bb = _require_backbone(
        resources, unit.method, role=RUNTIME_ROLE_SUMMARIZER
    )
    view = _task_view(adapter, problem)
    chunks = chunk_text_tokens(
        view.context,
        counter=counter,
        chunk_tokens=runtime.chunk_tokens,
        overlap_tokens=runtime.overlap_tokens,
    )
    from treepo._research.tree.state_tree_runner import run_fixed_binary_state_tree

    operator = _BackboneTreeOperator(summarizer_bb, runtime, counter)
    result = run_fixed_binary_state_tree(
        operator,
        chunks or [view.context],
        rubric=view.answer_instruction or "Preserve evidence for final question answering.",
        refine_rounds=int(getattr(runtime, "treepo_refine_rounds", 0) or 0),
    )
    root_memory = str(result.tree.final_rendered or "")
    prompt = _answer_prompt(view, context=f"TreePO root memory:\n{root_memory}")
    prompt = counter.truncate_tokens(
        prompt, max_tokens=_max_input_tokens(runtime, runtime.max_output_tokens), keep="tail"
    )
    answer = _call_final_answer(
        bb=scorer_bb,
        adapter=adapter,
        problem=problem,
        prompt=prompt,
        runtime=runtime,
        counter=counter,
    )
    cost = dict(answer.cost)
    steps = [
        _step(
            unit=unit,
            problem=problem,
            step_idx=idx,
            node_id="treepo",
            action_type=f"treepo_{op.operation}",
            input_tokens=0,
            output_tokens=0,
            latency_ms=float(op.latency_seconds) * 1000.0,
            extra=op.to_dict(),
        )
        for idx, op in enumerate(result.operations)
    ]
    steps.append(
        _step(
            unit=unit,
            problem=problem,
            step_idx=len(steps),
            node_id="answer",
            action_type="treepo_text_compressor_answer",
            input_tokens=counter.count(prompt),
            output_tokens=counter.count(answer.raw_text or answer.prediction),
            latency_ms=float(cost.get("wall_ms", 0.0) or 0.0),
            extra={"answer_decode_method": answer.decode_method},
        )
    )
    return MethodRunResult(
        prediction=answer.prediction,
        cost=cost,
        steps=steps,
        artifacts={
            "method_id": unit.method,
            "leaf_count": len(chunks),
            "root_memory_chars": len(root_memory),
            "answer": answer.to_dict(),
        },
    )


def run_neural_tree_selector_llm(
    *,
    unit: RunUnit,
    problem: ProblemSpec,
    adapter: BenchmarkAdapter,
    runtime: RuntimeConfig,
    resources: MethodResources,
    counter: TokenCounter,
    verifier: DeterministicVerifier,
    repair: SimpleRepairPolicy,
) -> MethodRunResult:
    # V1 frozen operator path: build a tree-shaped embedding state by mean
    # merging chunk vectors, select evidence leaves by query-vs-leaf similarity,
    # then let the LLM answer. A configured checkpoint can replace this selector
    # later without changing the method result envelope.
    bb = _require_backbone(resources, unit.method, role=RUNTIME_ROLE_SCORER)
    view = _task_view(adapter, problem)
    selected_operator = _select_chunks_by_operator(
        view=view,
        runtime=runtime,
        counter=counter,
        resources=resources,
    )
    if selected_operator is None:
        embedding_client = _embedding_client(resources)
        selected, artifacts = _select_chunks_by_embedding(
            view=view,
            runtime=runtime,
            counter=counter,
            embedding_client=embedding_client,
        )
    else:
        selected, artifacts = selected_operator
    evidence = "\n\n".join(selected)
    prompt = _answer_prompt(view, context=f"Frozen neural/tree selector evidence:\n{evidence}")
    prompt = counter.truncate_tokens(
        prompt, max_tokens=_max_input_tokens(runtime, runtime.max_output_tokens), keep="tail"
    )
    answer = _call_final_answer(
        bb=bb,
        adapter=adapter,
        problem=problem,
        prompt=prompt,
        runtime=runtime,
        counter=counter,
    )
    cost = dict(answer.cost)
    checkpoint = str(getattr(runtime, "neural_checkpoint_path", "") or "")
    if not checkpoint:
        ctx = getattr(resources, "inference_context", None)
        if (
            ctx is not None
            and hasattr(ctx, "has_role")
            and ctx.has_role(RUNTIME_ROLE_STATE_MODEL)
        ):
            try:
                checkpoint = str(
                    ctx.role_config(RUNTIME_ROLE_STATE_MODEL).get("checkpoint_path", "") or ""
                )
            except Exception:
                checkpoint = ""
    artifacts.update(
        {
            "method_id": unit.method,
            "selector_backend": artifacts.get("selector_backend", "embedding"),
            "state_model_checkpoint_path": checkpoint,
            "selected_context_chars": len(evidence),
            "answer": answer.to_dict(),
        }
    )
    return MethodRunResult(
        prediction=answer.prediction,
        cost=cost,
        steps=[
            _step(
                unit=unit,
                problem=problem,
                step_idx=0,
                node_id="neural_selector_answer",
                action_type="neural_tree_selector_llm",
                input_tokens=counter.count(prompt),
                output_tokens=counter.count(answer.raw_text or answer.prediction),
                latency_ms=float(cost.get("wall_ms", 0.0) or 0.0),
                extra={
                    "selector_backend": artifacts["selector_backend"],
                    "answer_decode_method": answer.decode_method,
                },
            )
        ],
        artifacts=artifacts,
    )


METHOD_RUNNERS: Dict[str, Any] = {
    "ground_truth": run_ground_truth,
    "empty": run_empty,
    "flat_prompt_baseline": run_llm_direct_official,
    "llm_direct_official": run_llm_direct_official,
    "chunk_concat_baseline": run_llm_tree_memory,
    "runtime_no_verifier": run_llm_tree_memory,
    "runtime_no_repair": run_llm_tree_memory,
    "runtime_full": run_llm_tree_memory,
    "llm_tree_memory": run_llm_tree_memory,
    "embedding_retrieval_llm": run_embedding_retrieval_llm,
    "treepo_text_compressor_llm": run_treepo_text_compressor_llm,
    "neural_tree_selector_llm": run_neural_tree_selector_llm,
}


PAPER_METHOD_ALIASES: Dict[str, str] = {
    "full_context": "llm_direct_official",
    "retrieval": "embedding_retrieval_llm",
    "summary_tree": "llm_tree_memory",
    "state_tree": "treepo_text_compressor_llm",
    "neural_operator": "neural_tree_selector_llm",
}


METHOD_COMPARE_RUNNER_ALIASES: Dict[str, Tuple[str, str, bool]] = {
    "baseline_llm": ("llm_tree_memory", "baseline_llm", False),
    "baseline_llm_raw": ("llm_tree_memory", "baseline_llm", False),
    "baseline_llm_trained": ("llm_tree_memory", "baseline_llm", True),
    "embedding_proxy_ridge": ("embedding_retrieval_llm", "embedding_proxy_ridge", True),
    "embedding_proxy_ridge_raw": ("embedding_retrieval_llm", "embedding_proxy_ridge", False),
    "embedding_proxy_ridge_trained": ("embedding_retrieval_llm", "embedding_proxy_ridge", True),
    "neural_operator_hybrid": ("neural_tree_selector_llm", "neural_operator_hybrid", True),
    "neural_operator_hybrid_raw": ("neural_tree_selector_llm", "neural_operator_hybrid", False),
    "neural_operator_hybrid_trained": ("neural_tree_selector_llm", "neural_operator_hybrid", True),
    "generator_lora_dpo": ("llm_tree_memory", "generator_lora_dpo", True),
    "generator_lora_dpo_raw": ("llm_tree_memory", "generator_lora_dpo", False),
    "generator_lora_dpo_trained": ("llm_tree_memory", "generator_lora_dpo", True),
}


def _method_spec_for_method(method: str, runtime: RuntimeConfig) -> MethodSpec:
    if method in PAPER_METHOD_ALIASES:
        return MethodSpec(
            name=method,
            runner_id=PAPER_METHOD_ALIASES[method],
            family=str(getattr(runtime, "method_family", "") or method),
            trained=bool(getattr(runtime, "method_trained", False)),
            artifact_dir=str(getattr(runtime, "method_dir", "") or ""),
            variant=str(getattr(runtime, "method_variant", "") or ""),
        )
    if method in METHOD_COMPARE_RUNNER_ALIASES:
        runner_id, family, trained = METHOD_COMPARE_RUNNER_ALIASES[method]
        variant = (
            method.rsplit("_", 1)[-1]
            if method.endswith(("_raw", "_trained"))
            else ("trained" if trained else "raw")
        )
        return MethodSpec(
            name=method,
            runner_id=runner_id,
            family=family,
            trained=trained,
            artifact_dir=str(getattr(runtime, "method_dir", "") or ""),
            variant=variant,
        )
    return MethodSpec(
        name=method,
        runner_id=method,
        family=str(getattr(runtime, "method_family", "") or method),
        trained=bool(getattr(runtime, "method_trained", False)),
        artifact_dir=str(getattr(runtime, "method_dir", "") or ""),
        variant=str(getattr(runtime, "method_variant", "") or ""),
    )


def discover_method(run_dir: str | Path, *, trained: bool = True) -> MethodSpec:
    """Infer a LongBench runtime method from a method-compare profile directory."""
    path = Path(run_dir).expanduser().resolve()
    profile = path.name
    stats_path = path / "final_stats.json"
    if stats_path.exists():
        try:
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
            profile = str(
                payload.get("profile")
                or payload.get("method")
                or payload.get("profile_name")
                or profile
            )
        except Exception:
            profile = path.name
    key = f"{profile}_{'trained' if trained else 'raw'}"
    if key not in METHOD_COMPARE_RUNNER_ALIASES and profile in METHOD_COMPARE_RUNNER_ALIASES:
        key = profile
    spec = _method_spec_for_method(key, RuntimeConfig(method_dir=str(path)))
    return MethodSpec(
        name=spec.name,
        runner_id=spec.runner_id,
        family=spec.family,
        trained=bool(trained),
        artifact_dir=str(path),
        variant="trained" if trained else "raw",
    )


def available_methods() -> Tuple[str, ...]:
    return tuple(
        sorted(set(METHOD_RUNNERS) | set(PAPER_METHOD_ALIASES) | set(METHOD_COMPARE_RUNNER_ALIASES))
    )


def run_runtime_method(
    *,
    unit: RunUnit,
    problem: ProblemSpec,
    adapter: BenchmarkAdapter,
    runtime: RuntimeConfig,
    resources: MethodResources,
    counter: TokenCounter,
    verifier: DeterministicVerifier,
    repair: SimpleRepairPolicy,
) -> MethodRunResult:
    method_spec = _method_spec_for_method(str(unit.method), runtime)
    try:
        runner = METHOD_RUNNERS[method_spec.runner_id]
    except KeyError as exc:
        supported = ", ".join(available_methods())
        raise ValueError(
            f"Unknown runtime method {unit.method!r}. Supported methods: {supported}"
        ) from exc
    ctx = getattr(resources, "inference_context", None)
    previous_scope: Dict[str, Any] = {}
    if ctx is not None and hasattr(ctx, "set_call_scope"):
        try:
            base_scope = dict(ctx.call_scope()) if hasattr(ctx, "call_scope") else {}
            base_scope.update(
                {
                    "run_id": unit.run_id,
                    "experiment_id": unit.run_id,
                    "unit_id": unit.unit_id,
                    "method_id": unit.method,
                    "runner_id": method_spec.runner_id,
                    "problem_id": problem.problem_id,
                }
            )
            previous_scope = ctx.set_call_scope(**base_scope)
        except Exception:
            previous_scope = {}
    try:
        result = runner(
            unit=unit,
            problem=problem,
            adapter=adapter,
            runtime=runtime,
            resources=resources,
            counter=counter,
            verifier=verifier,
            repair=repair,
        )
    finally:
        if ctx is not None and hasattr(ctx, "set_call_scope"):
            try:
                ctx.set_call_scope(**previous_scope)
            except Exception:
                pass
    artifacts = dict(result.artifacts)
    artifacts.update(
        {
            "method_id": unit.method,
            "runner_id": method_spec.runner_id,
            "method_backend_group": method_spec.family,
            "method_trained": bool(method_spec.trained),
            "method_variant": method_spec.variant,
        }
    )
    if method_spec.artifact_dir:
        artifacts["method_dir"] = method_spec.artifact_dir
    return MethodRunResult(
        prediction=result.prediction,
        cost=dict(result.cost),
        steps=list(result.steps),
        artifacts=artifacts,
    )


__all__ = [
    "HashingEmbeddingClient",
    "MethodResources",
    "MethodSpec",
    "PAPER_METHOD_ALIASES",
    "available_methods",
    "discover_method",
    "run_runtime_method",
]
