"""Fixed-binary diffusion tree engine for research prototypes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import asyncio
import concurrent.futures
import warnings
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from treepo._research.core.data_models import Node, Tree, leaf, node
from treepo._research.core.protocols import format_merge_input
from treepo._research.diffusion.backends import DiffusionBackend, DiffusionBatchResponse


@dataclass(frozen=True)
class DiffusionPromptTemplates:
    """Prompt templates for the fixed-tree diffusion prototype."""

    leaf_system: str = (
        "You denoise a fixed-binary tree leaf checkpoint. Preserve theorem-relevant "
        "content, remove noise, and return only the cleaned summary."
    )
    merge_system: str = (
        "You denoise a fixed-binary tree merge checkpoint. Combine both child summaries "
        "without dropping theorem-relevant content. Return only the merged summary."
    )
    refine_system: str = (
        "You refine an existing fixed-binary tree checkpoint. Preserve the meaning, "
        "tighten the wording, and return only the revised summary."
    )


def format_diffusion_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    """Format a plain-text chat prompt that works with text diffusion backends."""
    return (
        f"<system>\n{system_prompt.strip()}\n</system>\n\n"
        f"<user>\n{user_prompt.strip()}\n</user>\n\n"
        "<assistant>\n"
    )


def _leaf_prompt(text: str, rubric: str, templates: DiffusionPromptTemplates) -> str:
    return format_diffusion_chat_prompt(
        templates.leaf_system,
        f"Rubric:\n{rubric.strip() or 'Preserve theorem-relevant content.'}\n\nLeaf text:\n{text.strip()}",
    )


def _merge_prompt(left_summary: str, right_summary: str, rubric: str, templates: DiffusionPromptTemplates) -> str:
    merge_input = format_merge_input(left_summary, right_summary)
    return format_diffusion_chat_prompt(
        templates.merge_system,
        f"Rubric:\n{rubric.strip() or 'Preserve theorem-relevant content.'}\n\nMerge input:\n{merge_input}",
    )


def _refine_prompt(summary: str, rubric: str, round_index: int, templates: DiffusionPromptTemplates) -> str:
    return format_diffusion_chat_prompt(
        templates.refine_system,
        (
            f"Rubric:\n{rubric.strip() or 'Preserve theorem-relevant content.'}\n\n"
            f"Refinement round: {round_index}\n\nCurrent checkpoint:\n{summary.strip()}"
        ),
    )


@dataclass
class DiffusionOperationTrace:
    """Telemetry for one diffusion operation batch."""

    operation: str
    level: int
    round_index: int
    input_count: int
    output_count: int
    latency_seconds: float
    backend_name: str
    engine_options: Dict[str, Any] = field(default_factory=dict)
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    node_ids: List[str] = field(default_factory=list)
    carried_node_ids: List[str] = field(default_factory=list)


@dataclass
class DiffusionRunResult:
    """End-to-end result for a fixed-tree diffusion run."""

    tree: Tree
    operations: List[DiffusionOperationTrace]
    leaf_texts: List[str]
    backend_name: str
    engine_options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the run for JSON reporting."""
        return {
            "final_summary": self.tree.final_summary,
            "tree": self.tree.to_dict(),
            "leaf_texts": list(self.leaf_texts),
            "backend_name": self.backend_name,
            "engine_options": dict(self.engine_options),
            "operations": [asdict(operation) for operation in self.operations],
        }


class DiffusionTreeEngine(Protocol):
    """Protocol for fixed-tree diffusion engines."""

    def summarize_leaves(
        self,
        leaves: Sequence[str],
        *,
        rubric: str = "",
        sampling_params: Optional[Mapping[str, Any]] = None,
        engine_options: Optional[Mapping[str, Any]] = None,
    ) -> List[Node]:
        ...

    def merge_level(
        self,
        nodes: Sequence[Node],
        *,
        rubric: str = "",
        level: int,
        sampling_params: Optional[Mapping[str, Any]] = None,
        engine_options: Optional[Mapping[str, Any]] = None,
    ) -> List[Node]:
        ...

    def refine_rounds(
        self,
        tree: Tree,
        *,
        rounds: int,
        rubric: str = "",
        sampling_params: Optional[Mapping[str, Any]] = None,
        engine_options: Optional[Mapping[str, Any]] = None,
    ) -> List[DiffusionOperationTrace]:
        ...


class FixedBinaryDiffusionTreeEngine:
    """Thin fixed-binary tree engine backed by a diffusion backend."""

    def __init__(
        self,
        backend: DiffusionBackend,
        *,
        prompt_templates: Optional[DiffusionPromptTemplates] = None,
    ) -> None:
        self.backend = backend
        self.prompt_templates = prompt_templates or DiffusionPromptTemplates()
        self._operation_traces: List[DiffusionOperationTrace] = []

    def summarize_leaves(
        self,
        leaves: Sequence[str],
        *,
        rubric: str = "",
        sampling_params: Optional[Mapping[str, Any]] = None,
        engine_options: Optional[Mapping[str, Any]] = None,
    ) -> List[Node]:
        resolved_engine_options = dict(engine_options or {})
        prompts = [_leaf_prompt(text, rubric, self.prompt_templates) for text in leaves]
        batch = self.backend.generate(
            prompts,
            sampling_params=sampling_params,
            engine_options=resolved_engine_options,
        )
        nodes: List[Node] = []
        for index, (text, generation) in enumerate(zip(leaves, batch.generations)):
            nodes.append(
                leaf(
                    text,
                    summary=generation.output_text,
                    metadata={
                        "diffusion_stage": "leaf",
                        "leaf_index": index,
                        "prompt": generation.input_text,
                    },
                )
            )
        self._operation_traces.append(
            self._trace_from_batch(
                batch,
                operation="summarize_leaves",
                level=0,
                round_index=0,
                node_ids=[node_.id for node_ in nodes],
                carried_node_ids=[],
                sampling_params=sampling_params,
                engine_options=resolved_engine_options,
            )
        )
        return nodes

    def merge_level(
        self,
        nodes: Sequence[Node],
        *,
        rubric: str = "",
        level: int,
        sampling_params: Optional[Mapping[str, Any]] = None,
        engine_options: Optional[Mapping[str, Any]] = None,
    ) -> List[Node]:
        current_nodes = list(nodes)
        if not current_nodes:
            return []
        resolved_engine_options = dict(engine_options or {})

        prompts: List[str] = []
        pairs: List[tuple[Node, Node]] = []
        carried_node_ids: List[str] = []
        next_level: List[Node] = []

        index = 0
        while index + 1 < len(current_nodes):
            left_child = current_nodes[index]
            right_child = current_nodes[index + 1]
            pairs.append((left_child, right_child))
            prompts.append(_merge_prompt(left_child.summary, right_child.summary, rubric, self.prompt_templates))
            index += 2

        batch = None
        if prompts:
            batch = self.backend.generate(
                prompts,
                sampling_params=sampling_params,
                engine_options=resolved_engine_options,
            )
            for generation, (left_child, right_child) in zip(batch.generations, pairs):
                next_level.append(
                    node(
                        left_child,
                        right_child,
                        generation.output_text,
                        metadata={
                            "diffusion_stage": "merge",
                            "prompt": generation.input_text,
                        },
                    )
                )

        if index < len(current_nodes):
            carried = current_nodes[index]
            carried_node_ids.append(carried.id)
            carried.metadata.setdefault("diffusion_carry_levels", []).append(level)
            next_level.append(carried)

        if batch is not None:
            self._operation_traces.append(
                self._trace_from_batch(
                    batch,
                    operation="merge_level",
                    level=level,
                    round_index=0,
                    node_ids=[node_.id for node_ in next_level if node_.id not in carried_node_ids],
                    carried_node_ids=carried_node_ids,
                    sampling_params=sampling_params,
                    engine_options=resolved_engine_options,
                )
            )
        return next_level

    def refine_rounds(
        self,
        tree: Tree,
        *,
        rounds: int,
        rubric: str = "",
        sampling_params: Optional[Mapping[str, Any]] = None,
        engine_options: Optional[Mapping[str, Any]] = None,
    ) -> List[DiffusionOperationTrace]:
        if rounds <= 0:
            return []
        resolved_engine_options = dict(engine_options or {})

        traces: List[DiffusionOperationTrace] = []
        for round_index in range(1, rounds + 1):
            prompt = _refine_prompt(tree.root.summary, rubric, round_index, self.prompt_templates)
            batch = self.backend.generate(
                [prompt],
                sampling_params=sampling_params,
                engine_options=resolved_engine_options,
            )
            tree.root.summary = batch.generations[0].output_text
            tree.root.metadata.setdefault("diffusion_refinement_history", []).append(
                {
                    "round_index": round_index,
                    "summary": tree.root.summary,
                    "prompt": batch.generations[0].input_text,
                }
            )
            trace = self._trace_from_batch(
                batch,
                operation="refine_round",
                level=tree.height,
                round_index=round_index,
                node_ids=[tree.root.id],
                carried_node_ids=[],
                sampling_params=sampling_params,
                engine_options=resolved_engine_options,
            )
            self._operation_traces.append(trace)
            traces.append(trace)
        return traces

    def run_fixed_tree(
        self,
        leaf_texts: Sequence[str],
        *,
        rubric: str = "",
        refine_rounds: int = 0,
        sampling_params: Optional[Mapping[str, Any]] = None,
        engine_options: Optional[Mapping[str, Any]] = None,
    ) -> DiffusionRunResult:
        warnings.warn(
            "FixedBinaryDiffusionTreeEngine.run_fixed_tree is deprecated; "
            "prefer src.tree.state_tree_runner.run_fixed_binary_state_tree with "
            "src.tree.async_operator.AsyncFromDiffusionBackend.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not leaf_texts:
            raise ValueError("FixedBinaryDiffusionTreeEngine.run_fixed_tree requires at least one leaf.")
        resolved_engine_options = dict(engine_options or {})

        from treepo._research.tree.async_operator import AsyncFromDiffusionBackend
        from treepo._research.tree.state_tree import state_tree_to_text_tree
        from treepo._research.tree.state_tree_runner import arun_fixed_binary_state_tree, run_fixed_binary_state_tree

        operator = AsyncFromDiffusionBackend(
            self.backend,
            prompt_templates=self.prompt_templates,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            state_result = run_fixed_binary_state_tree(
                operator,  # type: ignore[arg-type]
                list(leaf_texts),
                rubric=rubric,
                refine_rounds=refine_rounds,
                sampling_params=sampling_params,
                engine_options=resolved_engine_options,
            )
        else:
            def _run_in_thread():
                return asyncio.run(
                    arun_fixed_binary_state_tree(
                        operator,  # type: ignore[arg-type]
                        list(leaf_texts),
                        rubric=rubric,
                        refine_rounds=refine_rounds,
                        sampling_params=sampling_params,
                        engine_options=resolved_engine_options,
                    )
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                state_result = executor.submit(_run_in_thread).result()

        root_id = str(state_result.tree.root.id)
        node_rendered = {node.id: str(node.rendered or "") for node in state_result.tree.traverse_preorder()}
        refinement_history = list(state_result.tree.root.metadata.get("refinement_history", []) or [])
        refine_outputs_by_round: Dict[int, str] = {}
        pre_refine_root = None
        if refinement_history:
            try:
                pre_refine_root = str(refinement_history[0].get("prior_rendered", "") or "")
            except Exception:
                pre_refine_root = None
        for entry in refinement_history:
            try:
                refine_outputs_by_round[int(entry.get("round_index"))] = str(entry.get("rendered", "") or "")
            except Exception:
                continue

        operations: List[DiffusionOperationTrace] = []
        for op in state_result.operations:
            operation_name = "summarize_leaves" if op.operation == "encode_leaves" else str(op.operation)
            outputs = [node_rendered.get(node_id, "") for node_id in op.node_ids]
            if op.operation == "merge_level" and pre_refine_root is not None:
                outputs = [pre_refine_root if node_id == root_id else node_rendered.get(node_id, "") for node_id in op.node_ids]
            elif op.operation == "refine_round":
                outputs = [refine_outputs_by_round.get(int(op.round_index), node_rendered.get(node_id, "")) for node_id in op.node_ids]

            operations.append(
                DiffusionOperationTrace(
                    operation=operation_name,
                    level=int(op.level),
                    round_index=int(op.round_index),
                    input_count=int(op.input_count),
                    output_count=int(op.output_count),
                    latency_seconds=float(op.latency_seconds),
                    backend_name=str(self.backend.backend_name),
                    engine_options=dict(op.engine_options),
                    sampling_params=dict(op.sampling_params),
                    outputs=outputs,
                    node_ids=list(op.node_ids),
                    carried_node_ids=list(op.carried_node_ids),
                )
            )

        tree_metadata = {
            "mode": "fixed_binary_diffusion",
            "leaf_count": len(leaf_texts),
            "merge_levels": int(state_result.tree.height),
            "engine_options": dict(resolved_engine_options),
            "state_tree_operations": [op.to_dict() for op in state_result.operations],
        }
        law_checks_by_node = {}
        for state_node in state_result.tree.traverse_preorder():
            law_checks = state_node.audit.get("law_checks")
            if law_checks:
                law_checks_by_node[state_node.id] = law_checks
        if law_checks_by_node:
            tree_metadata["state_tree_law_checks"] = law_checks_by_node

        tree = state_tree_to_text_tree(
            state_result.tree,  # type: ignore[arg-type]
            rubric=rubric,
            metadata=tree_metadata,
        )

        self._operation_traces = list(operations)
        return DiffusionRunResult(
            tree=tree,
            operations=operations,
            leaf_texts=[str(text) for text in leaf_texts],
            backend_name=str(self.backend.backend_name),
            engine_options=dict(resolved_engine_options),
        )

    def _trace_from_batch(
        self,
        batch: DiffusionBatchResponse,
        *,
        operation: str,
        level: int,
        round_index: int,
        node_ids: Sequence[str],
        carried_node_ids: Sequence[str],
        sampling_params: Optional[Mapping[str, Any]],
        engine_options: Optional[Mapping[str, Any]],
    ) -> DiffusionOperationTrace:
        return DiffusionOperationTrace(
            operation=operation,
            level=level,
            round_index=round_index,
            input_count=len(batch.generations),
            output_count=len(batch.generations),
            latency_seconds=batch.latency_seconds,
            backend_name=self.backend.backend_name,
            engine_options=dict(engine_options or {}),
            sampling_params=dict(sampling_params or {}),
            outputs=batch.texts,
            node_ids=list(node_ids),
            carried_node_ids=list(carried_node_ids),
        )
