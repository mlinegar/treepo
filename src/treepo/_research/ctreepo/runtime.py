"""Runtime facade for C-TreePO programs and sketch adapters."""

from __future__ import annotations

import sys
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from treepo._research.ctreepo.contracts import CTreePOProgramSpec, MethodDescriptor
from treepo._research.tree.state_tree import (
    StateNode,
    StateTree,
    explicit_oracle_trace_kwargs,
    local_law_trace_metadata,
)


@runtime_checkable
class RuntimeProgram(Protocol):
    """Minimal runtime program surface."""

    spec: CTreePOProgramSpec

    def reduce_tree(self, inputs: Any, *, schedule: str = "balanced") -> Any: ...

    def score(self, state_or_summary: Any, *, query: Any = None) -> Any: ...


def _ensure_treepo_on_path() -> None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "treepo" / "src"
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            return


@dataclass
class SketchAdapterProgram:
    """Runtime wrapper around `treepo.sketches.SketchAdapter`."""

    spec: CTreePOProgramSpec
    adapter: Any

    def _state_metadata(
        self,
        state: Any,
        *,
        node_id: str,
        node_type: str,
        query: Any = None,
        target: Any = None,
    ) -> dict[str, Any]:
        adapter_name = str(getattr(self.adapter, "name", "") or self.spec.family or "sketch")
        metadata: dict[str, Any] = {
            "source_node_id": str(node_id),
            "node_type": str(node_type),
            "method_family": adapter_name,
            "state_kind": "classical_sketch_state",
            "law_channel": str(node_type),
            "state_descriptor": {
                "adapter_name": adapter_name,
                "adapter_config": dict(getattr(self.adapter, "config", {}) or {}),
            },
        }
        for key, fn_name in (
            ("serialized_size_bytes", "serialized_size_bytes"),
            ("memory_bytes", "memory_bytes"),
        ):
            fn = getattr(self.adapter, fn_name, None)
            if callable(fn):
                try:
                    metadata["state_descriptor"][key] = float(fn(state))
                except Exception:
                    pass
        try:
            readout = self.adapter.query(state, query)
        except Exception:
            readout = None
        try:
            readout_float = float(readout)
        except (TypeError, ValueError):
            readout_float = None
        if readout_float is not None and math.isfinite(readout_float):
            metadata["prediction"] = float(readout_float)
            metadata["readout_prediction"] = float(readout_float)
        if isinstance(target, Mapping):
            target_payload = dict(target)
            for key in (
                "target",
                "proxy_target",
                "oracle_target",
                "oracle_loss",
                "observed",
                "sampled",
                "propensity",
                "logged_propensity",
                "oracle_propensity",
                "label_source",
                "truth_label_source",
            ):
                if key in target_payload:
                    metadata[key] = target_payload[key]
            target = target_payload.get("target", target_payload.get("proxy_target"))
        if target is not None:
            try:
                target_float = float(target)
            except (TypeError, ValueError):
                target_float = None
            if target_float is not None and math.isfinite(target_float):
                metadata["target"] = float(target_float)
        return metadata

    def reduce_tree(self, inputs: Sequence[Iterable[Any]], *, schedule: str = "balanced") -> Any:
        cfg = dict(self.spec.backend_config or {})
        if bool(cfg.get("inputs_are_states", False)):
            _ensure_treepo_on_path()
            from treepo.sketches.tree_reducer import fold_states

            return fold_states(list(inputs), self.adapter, schedule=schedule)
        _ensure_treepo_on_path()
        from treepo.sketches.tree_reducer import treepo_reduce

        return treepo_reduce(list(inputs), self.adapter, schedule=schedule)

    def score(self, state_or_summary: Any, *, query: Any = None) -> Any:
        return self.adapter.query(state_or_summary, query)

    def reduce_tree_trace(
        self,
        inputs: Sequence[Iterable[Any]],
        *,
        schedule: str = "balanced",
        query: Any = None,
        doc_id: str = "",
        split: str = "",
        targets: Mapping[str, Any] | Sequence[Any] | None = None,
    ) -> StateTree[Any, Any]:
        """Return the full realized sketch-reduction tree as a ``StateTree``."""

        input_list = list(inputs)
        if not input_list:
            raise ValueError("inputs must be non-empty")

        cfg = dict(self.spec.backend_config or {})
        if bool(cfg.get("inputs_are_states", False)):
            leaf_states = list(input_list)
            leaf_spans = [{"leaf_index": idx, "inputs_are_states": True} for idx in range(len(leaf_states))]
        else:
            leaf_items = [list(items) for items in input_list]
            leaf_states = [self.adapter.encode(items) for items in leaf_items]
            leaf_spans = [
                {"leaf_index": idx, "item_count": len(items)}
                for idx, items in enumerate(leaf_items)
            ]

        def target_for(node_id: str, index: int | None = None) -> Any:
            if targets is None:
                return None
            if isinstance(targets, Mapping):
                return targets.get(node_id)
            if index is not None and 0 <= index < len(targets):
                return targets[index]
            return None

        def make_node(
            *,
            node_id: str,
            level: int,
            state: Any,
            span: Any,
            node_type: str,
            left: StateNode[Any, Any] | None = None,
            right: StateNode[Any, Any] | None = None,
            target: Any = None,
        ) -> StateNode[Any, Any]:
            metadata = self._state_metadata(
                state,
                node_id=node_id,
                node_type=node_type,
                query=query,
                target=target,
            )
            node = StateNode[Any, Any](
                id=str(node_id),
                level=int(level),
                span=span,
                state=state,
                rendered=str(metadata.get("readout_prediction", "")),
                left_child=left,
                right_child=right,
                metadata=metadata,
            )
            if left is not None:
                left.parent = node
            if right is not None:
                right.parent = node
            return node

        leaves = [
            make_node(
                node_id=f"leaf_{idx}",
                level=0,
                state=state,
                span=leaf_spans[idx],
                node_type="leaf",
                target=target_for(f"leaf_{idx}", idx),
            )
            for idx, state in enumerate(leaf_states)
        ]

        sched = str(schedule or "balanced")
        if len(leaves) == 1:
            root = leaves[0]
        elif sched == "balanced":
            current = list(leaves)
            level = 1
            while len(current) > 1:
                next_level: list[StateNode[Any, Any]] = []
                pair_index = 0
                idx = 0
                while idx < len(current):
                    if idx + 1 >= len(current):
                        next_level.append(current[idx])
                        idx += 1
                        continue
                    left = current[idx]
                    right = current[idx + 1]
                    node_id = f"merge_{level}_{pair_index}"
                    merged = self.adapter.merge(left.state, right.state)
                    next_level.append(
                        make_node(
                            node_id=node_id,
                            level=level,
                            state=merged,
                            span={"left_child_id": left.id, "right_child_id": right.id},
                            node_type="merge",
                            left=left,
                            right=right,
                            target=target_for(node_id),
                        )
                    )
                    pair_index += 1
                    idx += 2
                current = next_level
                level += 1
            root = current[0]
        elif sched in {"left_to_right", "right_to_left"}:
            ordered = list(leaves) if sched == "left_to_right" else list(reversed(leaves))
            root = ordered[0]
            for idx, node in enumerate(ordered[1:], start=1):
                node_id = f"merge_{sched}_{idx}"
                merged = self.adapter.merge(root.state, node.state)
                root = make_node(
                    node_id=node_id,
                    level=idx,
                    state=merged,
                    span={"left_child_id": root.id, "right_child_id": node.id},
                    node_type="merge",
                    left=root,
                    right=node,
                    target=target_for(node_id),
                )
        else:
            raise ValueError(f"unsupported schedule: {schedule!r}")

        root_target = target_for("root")
        if root_target is not None:
            root.metadata.update(
                self._state_metadata(
                    root.state,
                    node_id=str(root.id),
                    node_type="root",
                    query=query,
                    target=root_target,
                )
            )
        root.metadata["node_type"] = "root"

        def assign_depths(node: StateNode[Any, Any], depth: int) -> None:
            node.metadata["doc_id"] = str(doc_id or "")
            node.metadata["split"] = str(split or "")
            node.metadata["depth"] = int(depth)
            node.metadata["is_root"] = bool(depth == 0)
            node.metadata["is_leaf"] = bool(node.is_leaf)
            prediction = node.metadata.get("prediction")
            target = node.metadata.get("target")
            try:
                prediction_float = None if prediction is None else float(prediction)
                target_float = None if target is None else float(target)
            except (TypeError, ValueError):
                prediction_float = None
                target_float = None
            oracle_kwargs = explicit_oracle_trace_kwargs(node.metadata)
            node.metadata.update(
                local_law_trace_metadata(
                    prediction=prediction_float,
                    proxy_target=target_float,
                    oracle_target=oracle_kwargs["oracle_target"],
                    oracle_loss=oracle_kwargs["oracle_loss"],
                    observed=bool(oracle_kwargs["observed"]),
                    sampled=bool(oracle_kwargs["sampled"]),
                    propensity=oracle_kwargs["propensity"],
                    law_channel=str(node.metadata.get("node_type", "")),
                    state_kind="classical_sketch_state",
                    label_source=str(oracle_kwargs["label_source"] or "proxy_target"),
                )
            )
            for child in node.children:
                assign_depths(child, depth + 1)

        assign_depths(root, 0)
        return StateTree(
            root=root,
            metadata={
                "doc_id": str(doc_id or ""),
                "split": str(split or ""),
                "method_family": str(getattr(self.adapter, "name", "") or self.spec.family or "sketch"),
                "state_kind": "classical_sketch_state",
                "trace_schema": "state_tree_full_trace_v1",
                "schedule": sched,
            },
        )


@dataclass
class FamilyRuntimeProgram:
    """Runtime wrapper around an existing `src.ctreepo.alternating.FamilyRuntime`."""

    spec: CTreePOProgramSpec
    family: Any

    def reduce_tree(self, inputs: Any, *, schedule: str = "balanced") -> Any:
        raise NotImplementedError(
            f"{getattr(self.family, 'name', 'family')} runtime does not expose a "
            "generic reduce_tree method; score LabeledTree inputs through score()."
        )

    def score(self, state_or_summary: Any, *, query: Any = None) -> Any:
        del query
        trees = (
            list(state_or_summary)
            if isinstance(state_or_summary, Sequence)
            and not isinstance(state_or_summary, (str, bytes, bytearray))
            else [state_or_summary]
        )
        preds = self.family.score_roots_with_f(
            f=self.spec.f_artifact,
            g=self.spec.g_artifact,
            trees=trees,
        )
        return preds if len(trees) != 1 else preds[0]


@dataclass
class LearnedCheckpointProgram:
    """Thin wrapper for migrated learned-sketch checkpoint runtimes.

    Callers may pass a concrete `predictor` callable in `backend_config` while
    checkpoint-specific loaders continue to migrate out of prototype modules.
    """

    spec: CTreePOProgramSpec
    predictor: Any = None

    def reduce_tree(self, inputs: Any, *, schedule: str = "balanced") -> Any:
        if callable(self.predictor):
            return self.predictor("reduce_tree", inputs, schedule=schedule)
        raise NotImplementedError(
            "learned checkpoint runtime requires backend_config['predictor'] "
            "or a migrated loader for this checkpoint family"
        )

    def score(self, state_or_summary: Any, *, query: Any = None) -> Any:
        if callable(self.predictor):
            return self.predictor("score", state_or_summary, query=query)
        raise NotImplementedError(
            "learned checkpoint runtime requires backend_config['predictor'] "
            "or a migrated loader for this checkpoint family"
        )


def _coerce_spec(spec: CTreePOProgramSpec | Mapping[str, Any]) -> CTreePOProgramSpec:
    if isinstance(spec, CTreePOProgramSpec):
        return spec
    if isinstance(spec, Mapping):
        return CTreePOProgramSpec.from_mapping(spec)
    raise TypeError(f"expected CTreePOProgramSpec or mapping, got {type(spec).__name__}")


def _build_hll_adapter(spec: CTreePOProgramSpec) -> Any:
    _ensure_treepo_on_path()
    from treepo.sketches import make_hll_adapter

    cfg = dict(spec.backend_config or {})
    backend = str(cfg.get("backend") or "native")
    if backend not in {"native", "datasketches"}:
        raise ValueError(
            f"unsupported HLL backend={backend!r}; use backend_config['backend'] "
            "with 'native' or 'datasketches'"
        )
    return make_hll_adapter(
        backend=backend,
        precision=int(cfg.get("precision", cfg.get("lg_k", 10))),
    )


ProgramBuilder = Callable[[CTreePOProgramSpec], RuntimeProgram]


def _build_hll_program(spec: CTreePOProgramSpec) -> RuntimeProgram:
    return SketchAdapterProgram(spec=spec, adapter=_build_hll_adapter(spec))


def _build_learned_checkpoint_program(spec: CTreePOProgramSpec) -> RuntimeProgram:
    cfg = dict(spec.backend_config or {})
    checkpoint = spec.g_artifact or spec.f_artifact or spec.leaf_adapter_artifact
    if checkpoint is not None and str(checkpoint) and not Path(str(checkpoint)).exists():
        raise FileNotFoundError(f"learned checkpoint artifact not found: {checkpoint}")
    return LearnedCheckpointProgram(
        spec=spec,
        predictor=cfg.get("predictor"),
    )


_METHOD_REGISTRY: dict[str, tuple[MethodDescriptor, ProgramBuilder]] = {
    "hll": (
        MethodDescriptor(
            method_id="hll",
            method_family="classical_sketch",
            backend="treepo.sketches",
            runtime_hook="_build_hll_program",
        ),
        _build_hll_program,
    ),
    "learned_sketch": (
        MethodDescriptor(
            method_id="learned_sketch",
            method_family="learned_sketch",
            backend="checkpoint",
            runtime_hook="_build_learned_checkpoint_program",
        ),
        _build_learned_checkpoint_program,
    ),
}

def method_descriptors() -> dict[str, dict[str, Any]]:
    return {key: descriptor.to_dict() for key, (descriptor, _builder) in _METHOD_REGISTRY.items()}


def _method_id_for_spec(spec: CTreePOProgramSpec) -> str:
    requested = str(spec.method_id or "").strip().lower()
    if not requested:
        raise ValueError("CTreePOProgramSpec requires public method_id")
    if requested not in _METHOD_REGISTRY:
        raise ValueError(
            f"unsupported C-TreePO runtime method_id={requested!r}; "
            f"available method_ids are {sorted(_METHOD_REGISTRY)}"
        )
    return requested


def load_program(spec: CTreePOProgramSpec | Mapping[str, Any]) -> RuntimeProgram:
    """Load a runtime program from a public program spec."""

    program_spec = _coerce_spec(spec)
    cfg = dict(program_spec.backend_config or {})

    adapter = cfg.get("adapter")
    if adapter is not None:
        return SketchAdapterProgram(spec=program_spec, adapter=adapter)

    family_runtime = cfg.get("family_runtime")
    if family_runtime is not None:
        return FamilyRuntimeProgram(spec=program_spec, family=family_runtime)

    method_id = _method_id_for_spec(program_spec)
    entry = _METHOD_REGISTRY.get(method_id)
    if entry is not None:
        _descriptor, builder = entry
        return builder(program_spec)

    raise ValueError(
        f"unsupported C-TreePO runtime method_id={method_id!r}"
    )


def reduce_tree(
    program: RuntimeProgram | CTreePOProgramSpec | Mapping[str, Any],
    inputs: Any,
    *,
    schedule: str = "balanced",
) -> Any:
    loaded = program if isinstance(program, RuntimeProgram) else load_program(program)
    return loaded.reduce_tree(inputs, schedule=schedule)


def score(
    program: RuntimeProgram | CTreePOProgramSpec | Mapping[str, Any],
    state_or_summary: Any,
    *,
    query: Any = None,
) -> Any:
    loaded = program if isinstance(program, RuntimeProgram) else load_program(program)
    return loaded.score(state_or_summary, query=query)


def trace_tree(
    program: RuntimeProgram | CTreePOProgramSpec | Mapping[str, Any],
    inputs: Any,
    *,
    schedule: str = "balanced",
    query: Any = None,
    doc_id: str = "",
    split: str = "",
    targets: Mapping[str, Any] | Sequence[Any] | None = None,
) -> StateTree[Any, Any]:
    """Return a full realized tree trace when the runtime exposes one."""

    loaded = program if isinstance(program, RuntimeProgram) else load_program(program)
    reduce_trace = getattr(loaded, "reduce_tree_trace", None)
    if not callable(reduce_trace):
        raise NotImplementedError(
            f"{loaded.__class__.__name__} does not expose reduce_tree_trace"
        )
    return reduce_trace(
        inputs,
        schedule=schedule,
        query=query,
        doc_id=doc_id,
        split=split,
        targets=targets,
    )


__all__ = [
    "FamilyRuntimeProgram",
    "LearnedCheckpointProgram",
    "RuntimeProgram",
    "SketchAdapterProgram",
    "load_program",
    "reduce_tree",
    "score",
    "trace_tree",
]
