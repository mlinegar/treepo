"""Async compositional operator interfaces and adapters.

The existing TreePO abstractions (e.g., ``CompositionalOperator``) are largely
sync-first. For unified execution across chat engines, diffusion endpoints, and
pure-Python exact operators, we provide an async-first operator contract that
can still wrap sync implementations.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Mapping, Optional, Protocol, Sequence, Tuple, TypeVar, TYPE_CHECKING

from treepo._research.core.async_utils import to_thread
from treepo._research.core.ops_checks import (
    EvidenceStatus,
    LawCapabilityReport,
    LawKind,
    OperatorCapabilityReport,
)
from treepo._research.core.protocols import format_merge_input

if TYPE_CHECKING:  # pragma: no cover
    from treepo._research.diffusion.backends import DiffusionBackend

SpanT = TypeVar("SpanT")
StateT = TypeVar("StateT")


class AsyncCompositionalOperator(Protocol, Generic[SpanT, StateT]):
    """Async operator contract for theorem-domain reduction over arbitrary state."""

    name: str

    async def aencode(self, span: SpanT, **kwargs: Any) -> StateT:
        ...

    async def amerge(self, left_state: StateT, right_state: StateT, **kwargs: Any) -> StateT:
        ...

    def combine(self, left_span: SpanT, right_span: SpanT, **kwargs: Any) -> SpanT:
        ...

    async def adecode(self, state: StateT, **kwargs: Any) -> SpanT:
        ...

    async def aresummarize(self, state: StateT, **kwargs: Any) -> StateT:
        ...

    async def aencode_many(self, spans: Sequence[SpanT], **kwargs: Any) -> List[StateT]:
        ...

    async def amerge_many(self, pairs: Sequence[Tuple[StateT, StateT]], **kwargs: Any) -> List[StateT]:
        ...

    def capability_report(self) -> OperatorCapabilityReport:
        ...


async def _gather_with_semaphore(
    coros: Sequence[Callable[[], "asyncio.Future[Any]"] | Callable[[], Any]],
    *,
    max_concurrent: int,
) -> List[Any]:
    semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))

    async def run_one(thunk: Callable[[], Any]) -> Any:
        async with semaphore:
            result = thunk()
            if asyncio.iscoroutine(result):
                return await result
            return result

    return await asyncio.gather(*(run_one(thunk) for thunk in coros))


@dataclass
class AsyncFromSyncOperator(Generic[SpanT, StateT]):
    """Wrap a sync ``CompositionalOperator`` as an async operator."""

    operator: Any
    name: str = "async_from_sync"
    max_concurrent: int = 64

    def __post_init__(self) -> None:
        if hasattr(self.operator, "name"):
            self.name = str(getattr(self.operator, "name"))

    async def aencode(self, span: SpanT, **kwargs: Any) -> StateT:
        return await to_thread(self.operator.encode, span, **kwargs)

    async def amerge(self, left_state: StateT, right_state: StateT, **kwargs: Any) -> StateT:
        return await to_thread(self.operator.merge, left_state, right_state, **kwargs)

    def combine(self, left_span: SpanT, right_span: SpanT, **kwargs: Any) -> SpanT:
        if hasattr(self.operator, "combine"):
            return self.operator.combine(left_span, right_span, **kwargs)
        return left_span  # type: ignore[return-value]

    async def adecode(self, state: StateT, **kwargs: Any) -> SpanT:
        if not hasattr(self.operator, "decode"):
            raise NotImplementedError(f"{self.name} has no decode() method.")
        return await to_thread(self.operator.decode, state, **kwargs)

    async def aresummarize(self, state: StateT, **kwargs: Any) -> StateT:
        if hasattr(self.operator, "resummarize"):
            return await to_thread(self.operator.resummarize, state, **kwargs)
        if hasattr(self.operator, "decode") and hasattr(self.operator, "encode"):
            span = await to_thread(self.operator.decode, state, **kwargs)
            return await to_thread(self.operator.encode, span, **kwargs)
        raise NotImplementedError(f"{self.name} does not support resummarize.")

    async def aencode_many(self, spans: Sequence[SpanT], **kwargs: Any) -> List[StateT]:
        thunks = [lambda span=span: self.aencode(span, **kwargs) for span in spans]
        return list(await _gather_with_semaphore(thunks, max_concurrent=self.max_concurrent))

    async def amerge_many(self, pairs: Sequence[Tuple[StateT, StateT]], **kwargs: Any) -> List[StateT]:
        thunks = [
            lambda left=left, right=right: self.amerge(left, right, **kwargs)
            for left, right in pairs
        ]
        return list(await _gather_with_semaphore(thunks, max_concurrent=self.max_concurrent))

    def capability_report(self) -> OperatorCapabilityReport:
        if hasattr(self.operator, "capability_report"):
            return self.operator.capability_report()
        # Conservative fallback: proxy-only with no declared laws.
        return OperatorCapabilityReport(
            operator_name=self.name,
            evidence_status=EvidenceStatus.PROXY_ONLY,
            latent_mergeability_enforced=False,
            tree_nesting_supported=True,
            theorem_domain_decode_available=bool(hasattr(self.operator, "decode")),
            theorem_domain_reencode_available=bool(hasattr(self.operator, "encode")),
        )


@dataclass
class AsyncFromSummarizationStrategy:
    """Lift a text-only ``SummarizationStrategy`` into an async operator."""

    strategy: Any
    name: str = "summarization_strategy"
    max_concurrent: int = 128
    notes: Tuple[str, ...] = (
        "Async adapter over SummarizationStrategy (text-only).",
        "Laws are declared evaluable but evidence is proxy-only unless externally certified.",
    )

    async def aencode(self, span: str, **kwargs: Any) -> str:
        rubric = str(kwargs.get("rubric", "") or "")
        temperature = float(kwargs.get("temperature", kwargs.get("sampling_params", {}).get("temperature", 0.7)) or 0.7)
        try:
            return await self.strategy.summarize(str(span), rubric, temperature=temperature)
        except TypeError:
            # Backward-compatible fallback for legacy strategy adapters.
            return await self.strategy.summarize(str(span), rubric)

    async def amerge(self, left_state: str, right_state: str, **kwargs: Any) -> str:
        rubric = str(kwargs.get("rubric", "") or "")
        temperature = float(kwargs.get("temperature", kwargs.get("sampling_params", {}).get("temperature", 0.7)) or 0.7)
        try:
            return await self.strategy.merge(str(left_state), str(right_state), rubric, temperature=temperature)
        except TypeError:
            return await self.strategy.merge(str(left_state), str(right_state), rubric)

    def combine(self, left_span: str, right_span: str, **_: Any) -> str:
        return format_merge_input(str(left_span or ""), str(right_span or ""))

    async def adecode(self, state: str, **_: Any) -> str:
        return str(state)

    async def aresummarize(self, state: str, **kwargs: Any) -> str:
        return await self.aencode(str(state), **kwargs)

    async def aencode_many(self, spans: Sequence[str], **kwargs: Any) -> List[str]:
        rubric = str(kwargs.get("rubric", "") or "")
        temperature = float(kwargs.get("temperature", kwargs.get("sampling_params", {}).get("temperature", 0.7)) or 0.7)
        bulk = getattr(self.strategy, "summarize_many", None)
        if callable(bulk):
            doc_id = None
            try:
                from treepo._research.core.strategy import tournament_doc_id

                doc_id = tournament_doc_id.get()
            except Exception:
                doc_id = None
            items = [
                {
                    "content": str(span),
                    "rubric": rubric,
                    "temperature": float(temperature),
                    "doc_id": doc_id,
                }
                for span in spans
            ]
            outputs = await bulk(items)
            return [str(item) for item in list(outputs)]

        thunks = [lambda span=span: self.aencode(span, **kwargs) for span in spans]
        return [str(item) for item in await _gather_with_semaphore(thunks, max_concurrent=self.max_concurrent)]

    async def amerge_many(self, pairs: Sequence[Tuple[str, str]], **kwargs: Any) -> List[str]:
        rubric = str(kwargs.get("rubric", "") or "")
        temperature = float(kwargs.get("temperature", kwargs.get("sampling_params", {}).get("temperature", 0.7)) or 0.7)
        bulk = getattr(self.strategy, "merge_many", None)
        if callable(bulk):
            doc_id = None
            try:
                from treepo._research.core.strategy import tournament_doc_id

                doc_id = tournament_doc_id.get()
            except Exception:
                doc_id = None
            items = [
                {
                    "left": str(left),
                    "right": str(right),
                    "rubric": rubric,
                    "temperature": float(temperature),
                    "doc_id": doc_id,
                }
                for left, right in pairs
            ]
            outputs = await bulk(items)
            return [str(item) for item in list(outputs)]

        thunks = [
            lambda left=left, right=right: self.amerge(left, right, **kwargs)
            for left, right in pairs
        ]
        return [str(item) for item in await _gather_with_semaphore(thunks, max_concurrent=self.max_concurrent)]

    def capability_report(self) -> OperatorCapabilityReport:
        return OperatorCapabilityReport(
            operator_name=self.name,
            evidence_status=EvidenceStatus.PROXY_ONLY,
            latent_mergeability_enforced=False,
            tree_nesting_supported=True,
            theorem_domain_decode_available=True,
            theorem_domain_reencode_available=True,
            exact_reduction_supported=False,
            leaf_law=LawCapabilityReport(
                law_kind=LawKind.L1_LEAF,
                available=True,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                exact=False,
                notes="Leaf outputs are produced by a text summarization strategy.",
            ),
            merge_law=LawCapabilityReport(
                law_kind=LawKind.L2_MERGE,
                available=True,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                exact=False,
                notes="Merge outputs are produced by merging two child summaries.",
            ),
            idempotence_law=LawCapabilityReport(
                law_kind=LawKind.L3_IDEMPOTENCE,
                available=True,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                exact=False,
                notes="Idempotence is evaluable as re-summarizing a produced summary.",
            ),
            notes=tuple(self.notes),
        )


@dataclass
class AsyncFromInferenceEngine:
    """Text-only async operator backed by the universal ``InferenceEngine``.

    This adapter supports both:
    - chat engines (surface ``chat_openai``), producing a single text output
    - diffusion engines (surface ``diffusion_generate``), producing a single text output
    """

    engine: Any
    name: str = "inference_engine_operator"
    max_tokens: int = 512
    temperature: float = 0.0
    stop: Tuple[str, ...] = ()
    summarize_prompt_fn: Optional[Callable[[str, str], Any]] = None
    merge_prompt_fn: Optional[Callable[[str, str, str], Any]] = None
    diffusion_prompt_templates: Optional[Any] = None
    max_concurrent: int = 128

    def __post_init__(self) -> None:
        if hasattr(self.engine, "engine_type"):
            self.name = f"{getattr(self.engine, 'engine_type').value}_operator"

    def combine(self, left_span: str, right_span: str, **_: Any) -> str:
        return format_merge_input(str(left_span or ""), str(right_span or ""))

    async def adecode(self, state: str, **_: Any) -> str:
        return str(state)

    async def aresummarize(self, state: str, **kwargs: Any) -> str:
        # Default resummary is just re-encoding the decoded span, but diffusion
        # has a dedicated refinement template that we preserve.
        rubric = str(kwargs.get("rubric", "") or "")
        sampling_params = kwargs.get("sampling_params")
        engine_options = kwargs.get("engine_options")
        round_index = int(kwargs.get("round_index", kwargs.get("refine_round", 0) or 0) or 0)

        if self.engine.surface.value == "chat_openai":
            from treepo._research.core.prompting import default_resummary_prompt
            from treepo._research.runtime.contracts import ChatInput, InferenceRequest
            from treepo._research.core.engines import EngineSurface

            messages = default_resummary_prompt(str(state), rubric, round_index=round_index or None)
            response = await self.engine.aexecute(
                InferenceRequest(
                    surface=EngineSurface.CHAT_OPENAI,
                    input=ChatInput(
                        messages=list(messages),
                        max_tokens=self._resolve_max_tokens(sampling_params),
                        temperature=self._resolve_temperature(sampling_params),
                        stop=list(self.stop),
                        extra=dict(engine_options or {}),
                    ),
                    engine_options=dict(engine_options or {}),
                )
            )
            return str(response.to_model_response().text or "")

        from treepo._research.runtime.contracts import DiffusionInput, InferenceRequest, TextListOutput
        from treepo._research.core.engines import EngineSurface
        from treepo._research.tree.generate_prompting import GenerateTreePromptTemplates, refine_prompt

        templates = self.diffusion_prompt_templates or GenerateTreePromptTemplates()
        prompt = refine_prompt(str(state), rubric, round_index or 1, templates)
        response = await self.engine.aexecute(
            InferenceRequest(
                surface=EngineSurface.DIFFUSION_GENERATE,
                input=DiffusionInput(
                    texts=[prompt],
                    sampling_params=dict(sampling_params or {}),
                ),
                engine_options=dict(engine_options or {}),
            )
        )
        if not isinstance(response.output, TextListOutput):
            raise TypeError(
                f"Diffusion resummarize expected TextListOutput, received {type(response.output).__name__}."
            )
        return str(response.output.texts[0] if response.output.texts else "")

    def _resolve_temperature(self, sampling_params: Optional[Mapping[str, Any]]) -> float:
        if sampling_params is None:
            return float(self.temperature)
        if sampling_params.get("temperature") is not None:
            try:
                return float(sampling_params["temperature"])
            except (TypeError, ValueError):
                return float(self.temperature)
        return float(self.temperature)

    def _resolve_max_tokens(self, sampling_params: Optional[Mapping[str, Any]]) -> int:
        if sampling_params is None:
            return int(self.max_tokens)
        for key in ("max_tokens", "max_new_tokens"):
            if sampling_params.get(key) is not None:
                try:
                    return int(sampling_params[key])
                except (TypeError, ValueError):
                    return int(self.max_tokens)
        return int(self.max_tokens)

    async def aencode(self, span: str, **kwargs: Any) -> str:
        rubric = str(kwargs.get("rubric", "") or "")
        sampling_params = kwargs.get("sampling_params")
        engine_options = kwargs.get("engine_options")
        if self.engine.surface.value == "chat_openai":
            from treepo._research.core.prompting import default_summarize_prompt
            from treepo._research.runtime.contracts import ChatInput, InferenceRequest
            from treepo._research.core.engines import EngineSurface

            prompt_fn = self.summarize_prompt_fn or default_summarize_prompt
            messages = prompt_fn(str(span), rubric)
            response = await self.engine.aexecute(
                InferenceRequest(
                    surface=EngineSurface.CHAT_OPENAI,
                    input=ChatInput(
                        messages=list(messages),
                        max_tokens=self._resolve_max_tokens(sampling_params),
                        temperature=self._resolve_temperature(sampling_params),
                        stop=list(self.stop),
                        extra=dict(engine_options or {}),
                    ),
                    engine_options=dict(engine_options or {}),
                )
            )
            return str(response.to_model_response().text or "")

        # diffusion_generate
        from treepo._research.runtime.contracts import DiffusionInput, InferenceRequest, TextListOutput
        from treepo._research.core.engines import EngineSurface
        from treepo._research.tree.generate_prompting import GenerateTreePromptTemplates, leaf_prompt

        templates = self.diffusion_prompt_templates or GenerateTreePromptTemplates()
        prompt = leaf_prompt(str(span), rubric, templates)
        response = await self.engine.aexecute(
            InferenceRequest(
                surface=EngineSurface.DIFFUSION_GENERATE,
                input=DiffusionInput(
                    texts=[prompt],
                    sampling_params=dict(sampling_params or {}),
                ),
                engine_options=dict(engine_options or {}),
            )
        )
        if not isinstance(response.output, TextListOutput):
            raise TypeError(
                f"Diffusion encode expected TextListOutput, received {type(response.output).__name__}."
            )
        return str(response.output.texts[0] if response.output.texts else "")

    async def amerge(self, left_state: str, right_state: str, **kwargs: Any) -> str:
        rubric = str(kwargs.get("rubric", "") or "")
        sampling_params = kwargs.get("sampling_params")
        engine_options = kwargs.get("engine_options")
        if self.engine.surface.value == "chat_openai":
            from treepo._research.core.prompting import default_merge_prompt
            from treepo._research.runtime.contracts import ChatInput, InferenceRequest
            from treepo._research.core.engines import EngineSurface

            prompt_fn = self.merge_prompt_fn or default_merge_prompt
            messages = prompt_fn(str(left_state), str(right_state), rubric)
            response = await self.engine.aexecute(
                InferenceRequest(
                    surface=EngineSurface.CHAT_OPENAI,
                    input=ChatInput(
                        messages=list(messages),
                        max_tokens=self._resolve_max_tokens(sampling_params),
                        temperature=self._resolve_temperature(sampling_params),
                        stop=list(self.stop),
                        extra=dict(engine_options or {}),
                    ),
                    engine_options=dict(engine_options or {}),
                )
            )
            return str(response.to_model_response().text or "")

        from treepo._research.runtime.contracts import DiffusionInput, InferenceRequest, TextListOutput
        from treepo._research.core.engines import EngineSurface
        from treepo._research.tree.generate_prompting import GenerateTreePromptTemplates, merge_prompt

        templates = self.diffusion_prompt_templates or GenerateTreePromptTemplates()
        prompt = merge_prompt(str(left_state), str(right_state), rubric, templates)
        response = await self.engine.aexecute(
            InferenceRequest(
                surface=EngineSurface.DIFFUSION_GENERATE,
                input=DiffusionInput(
                    texts=[prompt],
                    sampling_params=dict(sampling_params or {}),
                ),
                engine_options=dict(engine_options or {}),
            )
        )
        if not isinstance(response.output, TextListOutput):
            raise TypeError(
                f"Diffusion merge expected TextListOutput, received {type(response.output).__name__}."
            )
        return str(response.output.texts[0] if response.output.texts else "")

    async def aencode_many(self, spans: Sequence[str], **kwargs: Any) -> List[str]:
        if self.engine.surface.value == "diffusion_generate":
            rubric = str(kwargs.get("rubric", "") or "")
            sampling_params = kwargs.get("sampling_params")
            engine_options = kwargs.get("engine_options")
            from treepo._research.runtime.contracts import DiffusionInput, InferenceRequest, TextListOutput
            from treepo._research.core.engines import EngineSurface
            from treepo._research.tree.generate_prompting import GenerateTreePromptTemplates, leaf_prompt

            templates = self.diffusion_prompt_templates or GenerateTreePromptTemplates()
            prompts = [leaf_prompt(str(span), rubric, templates) for span in spans]
            response = await self.engine.aexecute(
                InferenceRequest(
                    surface=EngineSurface.DIFFUSION_GENERATE,
                    input=DiffusionInput(
                        texts=list(prompts),
                        sampling_params=dict(sampling_params or {}),
                    ),
                    engine_options=dict(engine_options or {}),
                )
            )
            if not isinstance(response.output, TextListOutput):
                raise TypeError(
                    f"Diffusion batch encode expected TextListOutput, received {type(response.output).__name__}."
                )
            return [str(text) for text in list(response.output.texts)]

        thunks = [lambda span=span: self.aencode(span, **kwargs) for span in spans]
        return [str(item) for item in await _gather_with_semaphore(thunks, max_concurrent=self.max_concurrent)]

    async def amerge_many(self, pairs: Sequence[Tuple[str, str]], **kwargs: Any) -> List[str]:
        if self.engine.surface.value == "diffusion_generate":
            rubric = str(kwargs.get("rubric", "") or "")
            sampling_params = kwargs.get("sampling_params")
            engine_options = kwargs.get("engine_options")
            from treepo._research.runtime.contracts import DiffusionInput, InferenceRequest, TextListOutput
            from treepo._research.core.engines import EngineSurface
            from treepo._research.tree.generate_prompting import GenerateTreePromptTemplates, merge_prompt

            templates = self.diffusion_prompt_templates or GenerateTreePromptTemplates()
            prompts = [
                merge_prompt(str(left), str(right), rubric, templates)
                for left, right in pairs
            ]
            response = await self.engine.aexecute(
                InferenceRequest(
                    surface=EngineSurface.DIFFUSION_GENERATE,
                    input=DiffusionInput(
                        texts=list(prompts),
                        sampling_params=dict(sampling_params or {}),
                    ),
                    engine_options=dict(engine_options or {}),
                )
            )
            if not isinstance(response.output, TextListOutput):
                raise TypeError(
                    f"Diffusion batch merge expected TextListOutput, received {type(response.output).__name__}."
                )
            return [str(text) for text in list(response.output.texts)]

        thunks = [
            lambda left=left, right=right: self.amerge(left, right, **kwargs)
            for left, right in pairs
        ]
        return [str(item) for item in await _gather_with_semaphore(thunks, max_concurrent=self.max_concurrent)]

    def capability_report(self) -> OperatorCapabilityReport:
        surface = getattr(self.engine, "surface", None)
        notes = []
        if surface is not None:
            notes.append(f"Backed by inference surface {surface.value}.")
        return OperatorCapabilityReport(
            operator_name=self.name,
            evidence_status=EvidenceStatus.PROXY_ONLY,
            latent_mergeability_enforced=False,
            tree_nesting_supported=True,
            theorem_domain_decode_available=True,
            theorem_domain_reencode_available=True,
            exact_reduction_supported=False,
            leaf_law=LawCapabilityReport(
                law_kind=LawKind.L1_LEAF,
                available=True,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                exact=False,
            ),
            merge_law=LawCapabilityReport(
                law_kind=LawKind.L2_MERGE,
                available=True,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                exact=False,
            ),
            idempotence_law=LawCapabilityReport(
                law_kind=LawKind.L3_IDEMPOTENCE,
                available=True,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                exact=False,
            ),
            notes=tuple(notes),
        )


@dataclass
class AsyncFromDiffusionBackend:
    """Text-only async operator backed by a sync ``DiffusionBackend``."""

    backend: "DiffusionBackend"
    prompt_templates: Optional[Any] = None
    name: str = "diffusion_backend_operator"
    max_concurrent: int = 128

    def __post_init__(self) -> None:
        backend_name = getattr(self.backend, "backend_name", None)
        if backend_name:
            self.name = f"diffusion_backend:{backend_name}"

        if self.prompt_templates is None:
            from treepo._research.tree.generate_prompting import GenerateTreePromptTemplates

            self.prompt_templates = GenerateTreePromptTemplates()

    def combine(self, left_span: str, right_span: str, **_: Any) -> str:
        return format_merge_input(str(left_span or ""), str(right_span or ""))

    async def adecode(self, state: str, **_: Any) -> str:
        return str(state)

    async def aencode(self, span: str, **kwargs: Any) -> str:
        rubric = str(kwargs.get("rubric", "") or "")
        sampling_params = kwargs.get("sampling_params")
        engine_options = kwargs.get("engine_options")

        from treepo._research.tree.generate_prompting import leaf_prompt

        prompt = leaf_prompt(str(span), rubric, self.prompt_templates)
        batch = await to_thread(
            self.backend.generate,
            [prompt],
            sampling_params=sampling_params,
            engine_options=engine_options,
        )
        return str(batch.generations[0].output_text if batch.generations else "")

    async def amerge(self, left_state: str, right_state: str, **kwargs: Any) -> str:
        rubric = str(kwargs.get("rubric", "") or "")
        sampling_params = kwargs.get("sampling_params")
        engine_options = kwargs.get("engine_options")

        from treepo._research.tree.generate_prompting import merge_prompt

        prompt = merge_prompt(str(left_state), str(right_state), rubric, self.prompt_templates)
        batch = await to_thread(
            self.backend.generate,
            [prompt],
            sampling_params=sampling_params,
            engine_options=engine_options,
        )
        return str(batch.generations[0].output_text if batch.generations else "")

    async def aresummarize(self, state: str, **kwargs: Any) -> str:
        rubric = str(kwargs.get("rubric", "") or "")
        sampling_params = kwargs.get("sampling_params")
        engine_options = kwargs.get("engine_options")
        round_index = int(kwargs.get("round_index", kwargs.get("refine_round", 0) or 0) or 0)

        from treepo._research.tree.generate_prompting import refine_prompt

        prompt = refine_prompt(str(state), rubric, round_index or 1, self.prompt_templates)
        batch = await to_thread(
            self.backend.generate,
            [prompt],
            sampling_params=sampling_params,
            engine_options=engine_options,
        )
        return str(batch.generations[0].output_text if batch.generations else "")

    async def aencode_many(self, spans: Sequence[str], **kwargs: Any) -> List[str]:
        rubric = str(kwargs.get("rubric", "") or "")
        sampling_params = kwargs.get("sampling_params")
        engine_options = kwargs.get("engine_options")

        from treepo._research.tree.generate_prompting import leaf_prompt

        prompts = [leaf_prompt(str(span), rubric, self.prompt_templates) for span in spans]
        batch = await to_thread(
            self.backend.generate,
            list(prompts),
            sampling_params=sampling_params,
            engine_options=engine_options,
        )
        return [str(generation.output_text) for generation in batch.generations]

    async def amerge_many(self, pairs: Sequence[Tuple[str, str]], **kwargs: Any) -> List[str]:
        rubric = str(kwargs.get("rubric", "") or "")
        sampling_params = kwargs.get("sampling_params")
        engine_options = kwargs.get("engine_options")

        from treepo._research.tree.generate_prompting import merge_prompt

        prompts = [
            merge_prompt(str(left), str(right), rubric, self.prompt_templates)
            for left, right in pairs
        ]
        batch = await to_thread(
            self.backend.generate,
            list(prompts),
            sampling_params=sampling_params,
            engine_options=engine_options,
        )
        return [str(generation.output_text) for generation in batch.generations]

    def capability_report(self) -> OperatorCapabilityReport:
        notes = []
        backend_name = getattr(self.backend, "backend_name", None)
        if backend_name:
            notes.append(f"Backed by diffusion backend {backend_name}.")
        return OperatorCapabilityReport(
            operator_name=self.name,
            evidence_status=EvidenceStatus.PROXY_ONLY,
            latent_mergeability_enforced=False,
            tree_nesting_supported=True,
            theorem_domain_decode_available=True,
            theorem_domain_reencode_available=True,
            exact_reduction_supported=False,
            leaf_law=LawCapabilityReport(
                law_kind=LawKind.L1_LEAF,
                available=True,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                exact=False,
            ),
            merge_law=LawCapabilityReport(
                law_kind=LawKind.L2_MERGE,
                available=True,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                exact=False,
            ),
            idempotence_law=LawCapabilityReport(
                law_kind=LawKind.L3_IDEMPOTENCE,
                available=True,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                exact=False,
                notes="Refinement is implemented via a `/generate` refine prompt template.",
            ),
            notes=tuple(notes),
        )


@dataclass(frozen=True)
class MarkovToyOperator:
    """Exact fixed-binary Markov sketch operator (Span=list[str], State=MarkovToySketch)."""

    name: str = "markov_toy_exact"

    async def aencode(self, span: Sequence[str], **_: Any):
        from treepo._research.diffusion.markov_toy import encode_markov_path

        return encode_markov_path(list(span))

    async def amerge(self, left_state: Any, right_state: Any, **_: Any):
        from treepo._research.diffusion.markov_toy import merge_markov_sketch

        return merge_markov_sketch(left_state, right_state)

    def combine(self, left_span: Sequence[str], right_span: Sequence[str], **_: Any) -> List[str]:
        return list(left_span) + list(right_span)

    async def adecode(self, state: Any, **_: Any):
        raise NotImplementedError("MarkovToyOperator does not support decode().")

    async def aencode_many(self, spans: Sequence[Sequence[str]], **kwargs: Any) -> List[Any]:
        return [await self.aencode(span, **kwargs) for span in spans]

    async def amerge_many(self, pairs: Sequence[Tuple[Any, Any]], **kwargs: Any) -> List[Any]:
        return [await self.amerge(left, right, **kwargs) for left, right in pairs]

    def capability_report(self) -> OperatorCapabilityReport:
        return OperatorCapabilityReport(
            operator_name=self.name,
            evidence_status=EvidenceStatus.THEOREM_BACKED,
            latent_mergeability_enforced=True,
            tree_nesting_supported=True,
            theorem_domain_decode_available=False,
            theorem_domain_reencode_available=False,
            exact_reduction_supported=True,
            leaf_law=LawCapabilityReport(
                law_kind=LawKind.L1_LEAF,
                available=True,
                evidence_status=EvidenceStatus.THEOREM_BACKED,
                exact=True,
                notes="Leaf encoding is exact: encode_markov_path.",
            ),
            merge_law=LawCapabilityReport(
                law_kind=LawKind.L2_MERGE,
                available=True,
                evidence_status=EvidenceStatus.THEOREM_BACKED,
                exact=True,
                notes="Merge is exact: merge_markov_sketch.",
            ),
            idempotence_law=LawCapabilityReport(
                law_kind=LawKind.L3_IDEMPOTENCE,
                available=False,
                evidence_status=EvidenceStatus.THEOREM_BACKED,
                exact=None,
                notes="Decode/re-encode is not available for the MarkovToySketch state.",
            ),
            notes=(
                "This operator is an exact mergeable sketch, useful as a claim-bearing lane.",
            ),
        )


__all__ = [
    "SpanT",
    "StateT",
    "AsyncCompositionalOperator",
    "AsyncFromSyncOperator",
    "AsyncFromSummarizationStrategy",
    "AsyncFromInferenceEngine",
    "AsyncFromDiffusionBackend",
    "MarkovToyOperator",
]
