from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from treepo._research.core.engines import EngineSurface
from treepo._research.runtime.contracts import (
    EmbeddingInput,
    EmbeddingOutput,
    InferenceResponse,
    OperatorInput,
    OperatorOutput,
    RuntimeSurfaceCall,
    RuntimeSurfaceResult,
)


CallSink = Callable[[Dict[str, Any]], None]


class RuntimeCallScheduler:
    """Small typed scheduler for runtime surface calls."""

    def __init__(self, context: Any, *, sink: Optional[CallSink] = None) -> None:
        self.context = context
        self.sink = sink

    def _emit(self, result: RuntimeSurfaceResult) -> None:
        if self.sink is not None:
            self.sink(result.to_dict())

    def _with_scope(self, call: RuntimeSurfaceCall) -> RuntimeSurfaceCall:
        scope = {}
        if hasattr(self.context, "call_scope"):
            try:
                scope = dict(self.context.call_scope())
            except Exception:
                scope = {}
        values = {
            "run_id": call.run_id
            or str(scope.get("experiment_id", "") or scope.get("run_id", "") or ""),
            "unit_id": call.unit_id or str(scope.get("unit_id", "") or ""),
            "method_id": call.method_id or str(scope.get("method_id", "") or ""),
            "runner_id": call.runner_id or str(scope.get("runner_id", "") or ""),
            "problem_id": call.problem_id or str(scope.get("problem_id", "") or ""),
            "node_id": call.node_id or str(scope.get("node_id", "") or ""),
            "request_kind": call.request_kind or str(scope.get("request_kind", "") or ""),
            "role": call.role or str(scope.get("role", "") or ""),
        }
        metadata = dict(scope.get("metadata", {}) or {})
        metadata.update(dict(call.metadata))
        return RuntimeSurfaceCall(
            surface=call.surface,
            input=call.input,
            role=values["role"],
            call_id=call.call_id,
            request_id=call.request_id,
            run_id=values["run_id"],
            unit_id=values["unit_id"],
            method_id=values["method_id"],
            runner_id=values["runner_id"],
            problem_id=values["problem_id"],
            node_id=values["node_id"],
            request_kind=values["request_kind"] or call.surface.value,
            document_id=call.document_id,
            routing_key=call.routing_key,
            priority=call.priority,
            metadata=metadata,
            engine_options=dict(call.engine_options),
            artifacts=dict(call.artifacts),
            created_utc=call.created_utc,
        )

    async def aschedule(self, call: RuntimeSurfaceCall) -> RuntimeSurfaceResult:
        return (await self.aschedule_many([call]))[0]

    def schedule(self, call: RuntimeSurfaceCall) -> RuntimeSurfaceResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aschedule(call))
        raise RuntimeError(
            "RuntimeCallScheduler.schedule() cannot be used from an active event loop; "
            "use aschedule()."
        )

    async def aschedule_many(
        self, calls: Sequence[RuntimeSurfaceCall]
    ) -> List[RuntimeSurfaceResult]:
        scoped = [self._with_scope(call) for call in calls]
        if not scoped:
            return []

        responses_by_index: dict[int, InferenceResponse] = {}
        for (surface, role), items in _group_by_surface_and_role(scoped).items():
            try:
                engine = self.context.engine(surface, role=role or None)
            except TypeError:
                engine = self.context.engine(surface)
            requests = [call.to_inference_request() for _, call in items]
            responses = await engine.aexecute_many(requests)
            for (idx, _), response in zip(items, responses):
                responses_by_index[idx] = response

        results: List[RuntimeSurfaceResult] = []
        for idx, call in enumerate(scoped):
            result = RuntimeSurfaceResult(call=call, response=responses_by_index[idx])
            self._emit(result)
            results.append(result)
        return results

    def schedule_many(self, calls: Sequence[RuntimeSurfaceCall]) -> List[RuntimeSurfaceResult]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aschedule_many(calls))
        raise RuntimeError(
            "RuntimeCallScheduler.schedule_many() cannot be used from an active event loop; "
            "use aschedule_many()."
        )

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        request_kind: str = "embed_texts",
    ) -> list[list[float]]:
        result = self.schedule(
            RuntimeSurfaceCall(
                surface=EngineSurface.EMBEDDING,
                input=EmbeddingInput(texts=[str(text) for text in texts]),
                role="embedder",
                request_kind=request_kind,
                metadata=dict(metadata or {}),
            )
        )
        if not isinstance(result.response.output, EmbeddingOutput):
            raise TypeError(
                "Embedding surface returned "
                f"{type(result.response.output).__name__}, expected EmbeddingOutput."
            )
        return result.response.output.vectors

    def execute_operator(
        self,
        operation: str,
        *,
        inputs: Optional[Mapping[str, Any]] = None,
        batch: Optional[Sequence[Mapping[str, Any]]] = None,
        options: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> RuntimeSurfaceResult:
        return self.schedule(
            RuntimeSurfaceCall(
                surface=EngineSurface.OPERATOR,
                input=OperatorInput(
                    operation=str(operation),
                    inputs=dict(inputs or {}),
                    batch=[dict(item) for item in (batch or [])],
                    options=dict(options or {}),
                ),
                role="state_model",
                request_kind=f"state_model:{operation}",
                metadata=dict(metadata or {}),
            )
        )

    def batch_operator(
        self,
        operation: str,
        items: Sequence[Mapping[str, Any]],
        *,
        inputs: Optional[Mapping[str, Any]] = None,
        options: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> RuntimeSurfaceResult:
        return self.execute_operator(
            operation,
            inputs=dict(inputs or {}),
            batch=[dict(item) for item in items],
            options=dict(options or {}),
            metadata=dict(metadata or {}),
        )


def _group_by_surface_and_role(
    calls: Sequence[RuntimeSurfaceCall],
) -> Dict[tuple[EngineSurface, str], List[tuple[int, RuntimeSurfaceCall]]]:
    grouped: Dict[tuple[EngineSurface, str], List[tuple[int, RuntimeSurfaceCall]]] = defaultdict(list)
    for idx, call in enumerate(calls):
        grouped[(call.surface, str(call.role or ""))].append((idx, call))
    return dict(grouped)


def operator_result_data(result: RuntimeSurfaceResult) -> Any:
    output = result.response.output
    if isinstance(output, OperatorOutput):
        return output.data
    return None


__all__ = [
    "CallSink",
    "RuntimeCallScheduler",
    "operator_result_data",
]
