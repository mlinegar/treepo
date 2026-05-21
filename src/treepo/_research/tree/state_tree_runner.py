"""Canonical fixed-binary runner for stateful TreePO operators."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Generic, List, Mapping, Optional, Sequence, Tuple, TypeVar

from treepo._research.tree.async_operator import AsyncCompositionalOperator
from treepo._research.tree.state_tree import StateNode, StateTree
from treepo._research.tree.state_tree_verifiers import LawVerifier, _attach_law_checks

SpanT = TypeVar("SpanT")
StateT = TypeVar("StateT")


def _render_state(state: Any) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    try:
        import torch  # type: ignore

        if isinstance(state, torch.Tensor):
            payload = {
                "type": "torch.Tensor",
                "dtype": str(state.dtype),
                "shape": list(state.shape),
                "device": str(state.device),
            }
            return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:
        pass
    if is_dataclass(state):
        try:
            return json.dumps(asdict(state), ensure_ascii=False, sort_keys=True)
        except Exception:
            return repr(state)
    text = repr(state)
    if len(text) > 2000:
        return text[:2000] + "...(truncated)"
    return text


@dataclass(frozen=True)
class StateTreeOperationTrace:
    operation: str
    level: int
    round_index: int
    input_count: int
    output_count: int
    latency_seconds: float
    operator_name: str
    engine_options: Dict[str, Any] = field(default_factory=dict)
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    node_ids: List[str] = field(default_factory=list)
    carried_node_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "level": int(self.level),
            "round_index": int(self.round_index),
            "input_count": int(self.input_count),
            "output_count": int(self.output_count),
            "latency_seconds": float(self.latency_seconds),
            "operator_name": str(self.operator_name),
            "engine_options": dict(self.engine_options),
            "sampling_params": dict(self.sampling_params),
            "node_ids": list(self.node_ids),
            "carried_node_ids": list(self.carried_node_ids),
        }


@dataclass
class FixedBinaryStateTreeRunResult(Generic[SpanT, StateT]):
    tree: StateTree[SpanT, StateT]
    operations: List[StateTreeOperationTrace]
    leaf_spans: List[SpanT]
    operator_name: str
    engine_options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operator_name": self.operator_name,
            "leaf_count": len(self.leaf_spans),
            "final_rendered": self.tree.final_rendered,
            "tree": self.tree.to_dict(),
            "operations": [op.to_dict() for op in self.operations],
            "engine_options": dict(self.engine_options),
        }


async def arun_fixed_binary_state_tree(
    operator: AsyncCompositionalOperator[SpanT, StateT],
    leaf_spans: Sequence[SpanT],
    *,
    rubric: str = "",
    refine_rounds: int = 0,
    max_concurrent: int = 128,
    sampling_params: Optional[Mapping[str, Any]] = None,
    engine_options: Optional[Mapping[str, Any]] = None,
    verifiers: Optional[Sequence[LawVerifier]] = None,
) -> FixedBinaryStateTreeRunResult[SpanT, StateT]:
    """Run a fixed-binary reduction schedule over ``leaf_spans``.

    This is the canonical stateful runner. It is intentionally simple:
    - encode leaves
    - fixed-binary merges with carry-forward of an odd node
    - optional refine rounds: state := encode(decode(state))
    """
    if not leaf_spans:
        raise ValueError("arun_fixed_binary_state_tree requires at least one leaf span.")

    resolved_engine_options = dict(engine_options or {})
    resolved_sampling_params = dict(sampling_params or {})

    operations: List[StateTreeOperationTrace] = []

    # ---------------------------------------------------------------------
    # Encode leaves (batched when possible).
    # ---------------------------------------------------------------------
    started = time.time()
    if hasattr(operator, "aencode_many"):
        leaf_states = await operator.aencode_many(
            list(leaf_spans),
            rubric=rubric,
            sampling_params=resolved_sampling_params,
            engine_options=resolved_engine_options,
            max_concurrent=max_concurrent,
        )
    else:
        semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))

        async def encode_one(span: SpanT) -> StateT:
            async with semaphore:
                return await operator.aencode(
                    span,
                    rubric=rubric,
                    sampling_params=resolved_sampling_params,
                    engine_options=resolved_engine_options,
                )

        leaf_states = list(await asyncio.gather(*(encode_one(span) for span in leaf_spans)))

    leaf_nodes: List[StateNode[SpanT, StateT]] = []
    for index, (span, state) in enumerate(zip(leaf_spans, leaf_states)):
        node = StateNode(
            level=0,
            span=span,
            state=state,
            rendered=_render_state(state),
            metadata={
                "leaf_index": index,
                "leaf_start_index": index,
                "leaf_end_index": index,
                "range_label": f"{index}:{index}",
            },
        )
        leaf_nodes.append(node)

    if verifiers:
        for node in leaf_nodes:
            for verifier in verifiers:
                if hasattr(verifier, "check_leaf"):
                    checks = await verifier.check_leaf(
                        node, rubric=rubric, operator=operator, sampling_params=resolved_sampling_params, engine_options=resolved_engine_options
                    )
                    if checks:
                        _attach_law_checks(node, checks, verifier_name=getattr(verifier, "name", verifier.__class__.__name__))

    operations.append(
        StateTreeOperationTrace(
            operation="encode_leaves",
            level=0,
            round_index=0,
            input_count=len(leaf_spans),
            output_count=len(leaf_nodes),
            latency_seconds=time.time() - started,
            operator_name=str(getattr(operator, "name", "operator")),
            engine_options=dict(resolved_engine_options),
            sampling_params=dict(resolved_sampling_params),
            node_ids=[node.id for node in leaf_nodes],
            carried_node_ids=[],
        )
    )

    # ---------------------------------------------------------------------
    # Fixed-binary merges.
    # ---------------------------------------------------------------------
    current_nodes: List[StateNode[SpanT, StateT]] = leaf_nodes
    merge_level = 1
    while len(current_nodes) > 1:
        started = time.time()
        pairs: List[Tuple[StateNode[SpanT, StateT], StateNode[SpanT, StateT]]] = []
        merge_pairs: List[Tuple[StateT, StateT]] = []
        carried: List[StateNode[SpanT, StateT]] = []

        index = 0
        while index + 1 < len(current_nodes):
            left = current_nodes[index]
            right = current_nodes[index + 1]
            pairs.append((left, right))
            merge_pairs.append((left.state, right.state))  # type: ignore[arg-type]
            index += 2
        if index < len(current_nodes):
            carried_node = current_nodes[index]
            carried_node.metadata.setdefault("carried_levels", []).append(merge_level)
            carried.append(carried_node)

        merged_states: List[StateT] = []
        if pairs:
            if hasattr(operator, "amerge_many"):
                merged_states = await operator.amerge_many(
                    merge_pairs,
                    rubric=rubric,
                    sampling_params=resolved_sampling_params,
                    engine_options=resolved_engine_options,
                    max_concurrent=max_concurrent,
                )
            else:
                semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))

                async def merge_one(left_state: StateT, right_state: StateT) -> StateT:
                    async with semaphore:
                        return await operator.amerge(
                            left_state,
                            right_state,
                            rubric=rubric,
                            sampling_params=resolved_sampling_params,
                            engine_options=resolved_engine_options,
                        )

                merged_states = list(
                    await asyncio.gather(*(merge_one(left, right) for left, right in merge_pairs))
                )

        next_nodes: List[StateNode[SpanT, StateT]] = []
        created_ids: List[str] = []
        for (left, right), state in zip(pairs, merged_states):
            combined_span = operator.combine(left.span, right.span, rubric=rubric)  # type: ignore[arg-type]
            left_start = int(left.metadata.get("leaf_start_index", left.metadata.get("leaf_index", 0)))
            left_end = int(left.metadata.get("leaf_end_index", left.metadata.get("leaf_index", left_start)))
            right_start = int(right.metadata.get("leaf_start_index", right.metadata.get("leaf_index", left_end)))
            right_end = int(right.metadata.get("leaf_end_index", right.metadata.get("leaf_index", right_start)))
            leaf_start = min(left_start, right_start)
            leaf_end = max(left_end, right_end)
            parent = StateNode(
                level=max(int(left.level), int(right.level)) + 1,
                span=combined_span,
                state=state,
                rendered=_render_state(state),
                left_child=left,
                right_child=right,
                metadata={
                    "merge_level": merge_level,
                    "leaf_start_index": leaf_start,
                    "leaf_end_index": leaf_end,
                    "range_label": f"{leaf_start}:{leaf_end}",
                },
            )
            left.parent = parent
            right.parent = parent
            left.metadata.setdefault("child_side", "left")
            right.metadata.setdefault("child_side", "right")
            left.metadata.setdefault("parent_id", parent.id)
            right.metadata.setdefault("parent_id", parent.id)
            next_nodes.append(parent)
            created_ids.append(parent.id)

            if verifiers:
                for verifier in verifiers:
                    if hasattr(verifier, "check_merge"):
                        checks = await verifier.check_merge(
                            parent,
                            left,
                            right,
                            rubric=rubric,
                            operator=operator,
                            sampling_params=resolved_sampling_params,
                            engine_options=resolved_engine_options,
                        )
                        if checks:
                            _attach_law_checks(
                                parent,
                                checks,
                                verifier_name=getattr(verifier, "name", verifier.__class__.__name__),
                            )

        for carried_node in carried:
            next_nodes.append(carried_node)

        operations.append(
            StateTreeOperationTrace(
                operation="merge_level",
                level=int(merge_level),
                round_index=0,
                input_count=len(current_nodes),
                output_count=len(next_nodes),
                latency_seconds=time.time() - started,
                operator_name=str(getattr(operator, "name", "operator")),
                engine_options=dict(resolved_engine_options),
                sampling_params=dict(resolved_sampling_params),
                node_ids=created_ids,
                carried_node_ids=[node.id for node in carried],
            )
        )

        current_nodes = next_nodes
        merge_level += 1

    tree = StateTree(
        root=current_nodes[0],
        metadata={
            "mode": "fixed_binary_state_tree",
            "leaf_count": len(leaf_spans),
            "engine_options": dict(resolved_engine_options),
        },
    )
    tree.root.metadata.setdefault("child_side", "root")

    # ---------------------------------------------------------------------
    # Optional refine rounds: state := encode(decode(state)).
    # ---------------------------------------------------------------------
    capability = operator.capability_report()
    can_decode_encode = bool(getattr(capability, "supports_resummary_idempotence", False))
    prefers_aresummarize = hasattr(operator, "aresummarize")

    if refine_rounds > 0 and (prefers_aresummarize or can_decode_encode):
        tree.root.metadata.setdefault("refinement_history", [])
        for round_index in range(1, int(refine_rounds) + 1):
            started = time.time()
            prior_state = tree.root.state
            prior_rendered = tree.root.rendered
            refined_state: Optional[StateT] = None

            if prefers_aresummarize:
                try:
                    refined_state = await operator.aresummarize(  # type: ignore[attr-defined]
                        prior_state,  # type: ignore[arg-type]
                        rubric=rubric,
                        sampling_params=resolved_sampling_params,
                        engine_options=resolved_engine_options,
                        round_index=int(round_index),
                    )
                except NotImplementedError:
                    refined_state = None

            if refined_state is None and can_decode_encode:
                try:
                    decoded = await operator.adecode(
                        prior_state,  # type: ignore[arg-type]
                        rubric=rubric,
                        sampling_params=resolved_sampling_params,
                        engine_options=resolved_engine_options,
                    )
                    refined_state = await operator.aencode(
                        decoded,
                        rubric=rubric,
                        sampling_params=resolved_sampling_params,
                        engine_options=resolved_engine_options,
                    )
                except NotImplementedError:
                    refined_state = None

            if refined_state is None:
                tree.metadata.setdefault("refine_skipped", True)
                tree.metadata.setdefault(
                    "refine_skip_reason",
                    "operator_missing_aresummarize_and_decode_encode",
                )
                break

            tree.root.state = refined_state
            tree.root.rendered = _render_state(refined_state)
            tree.root.metadata["refinement_history"].append(
                {
                    "round_index": int(round_index),
                    "prior_rendered": str(prior_rendered or ""),
                    "rendered": str(tree.root.rendered or ""),
                }
            )
            operations.append(
                StateTreeOperationTrace(
                    operation="refine_round",
                    level=int(tree.height),
                    round_index=int(round_index),
                    input_count=1,
                    output_count=1,
                    latency_seconds=time.time() - started,
                    operator_name=str(getattr(operator, "name", "operator")),
                    engine_options=dict(resolved_engine_options),
                    sampling_params=dict(resolved_sampling_params),
                    node_ids=[tree.root.id],
                    carried_node_ids=[],
                )
            )
            if verifiers:
                for verifier in verifiers:
                    if hasattr(verifier, "check_idempotence"):
                        checks = await verifier.check_idempotence(
                            tree.root,
                            prior_rendered=str(prior_rendered or ""),
                            rubric=rubric,
                            operator=operator,
                            sampling_params=resolved_sampling_params,
                            engine_options=resolved_engine_options,
                        )
                        if checks:
                            _attach_law_checks(
                                tree.root,
                                checks,
                                verifier_name=getattr(verifier, "name", verifier.__class__.__name__),
                            )
    elif refine_rounds > 0 and not (prefers_aresummarize or can_decode_encode):
        tree.metadata.setdefault("refine_skipped", True)
        tree.metadata.setdefault(
            "refine_skip_reason",
            "operator_missing_aresummarize_and_decode_encode",
        )

    if verifiers:
        node_map = {node.id: node for node in tree.traverse_preorder()}
        for verifier in verifiers:
            if hasattr(verifier, "finalize_tree"):
                results = await verifier.finalize_tree(
                    tree, rubric=rubric, operator=operator, engine_options=resolved_engine_options, sampling_params=resolved_sampling_params
                )
                if not results:
                    continue
                for node_id, checks in results.items():
                    node = node_map.get(str(node_id))
                    if node is None:
                        continue
                    _attach_law_checks(
                        node,
                        checks,
                        verifier_name=getattr(verifier, "name", verifier.__class__.__name__),
                    )

    return FixedBinaryStateTreeRunResult(
        tree=tree,
        operations=operations,
        leaf_spans=list(leaf_spans),
        operator_name=str(getattr(operator, "name", "operator")),
        engine_options=dict(resolved_engine_options),
    )


def run_fixed_binary_state_tree(
    operator: AsyncCompositionalOperator[SpanT, StateT],
    leaf_spans: Sequence[SpanT],
    *,
    rubric: str = "",
    refine_rounds: int = 0,
    max_concurrent: int = 128,
    sampling_params: Optional[Mapping[str, Any]] = None,
    engine_options: Optional[Mapping[str, Any]] = None,
    verifiers: Optional[Sequence[LawVerifier]] = None,
) -> FixedBinaryStateTreeRunResult[SpanT, StateT]:
    """Sync wrapper around ``arun_fixed_binary_state_tree``."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            arun_fixed_binary_state_tree(
                operator,
                leaf_spans,
                rubric=rubric,
                refine_rounds=refine_rounds,
                max_concurrent=max_concurrent,
                sampling_params=sampling_params,
                engine_options=engine_options,
                verifiers=verifiers,
            )
        )
    raise RuntimeError("run_fixed_binary_state_tree cannot be called from an active event loop; use arun_fixed_binary_state_tree.")


__all__ = [
    "StateTreeOperationTrace",
    "FixedBinaryStateTreeRunResult",
    "arun_fixed_binary_state_tree",
    "run_fixed_binary_state_tree",
]
