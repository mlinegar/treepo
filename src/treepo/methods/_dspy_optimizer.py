"""Optimizer-backed DSPy learning for the package DSPy family.

This module stays DSPy-import-free until a native compile or reload is
requested.  One g program receives explicitly tagged leaf and merge calls;
``reduce_g`` is only repeated application of that same program.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo.forest import oracle_vector_l1
from treepo.methods._dspy_records import (
    BinaryTopology,
    binary_topology,
    f_training_records,
    g_training_records,
    leaf_prompt,
    merge_prompt,
    split_examples,
)
from treepo.state import state_to_dict
from treepo.tree import TreeRecord


class DSPyOptimizerMixin:
    """Shared implementation mixed into :class:`treepo.methods.dspy.DSPyFamily`."""

    def _init_dspy_optimizer_runtime(
        self,
        *,
        f_program: Any = None,
        g_program: Any = None,
        compiler: Any = None,
        program_loader: Any = None,
        program_saver: Any = None,
        dspy_module: Any = None,
        token_count_fn: Any = None,
    ) -> None:
        if token_count_fn is not None and not callable(token_count_fn):
            raise TypeError("DSPy token_count_fn must be callable")
        self._dspy_f_program = f_program
        self._dspy_g_program = g_program
        self._dspy_compiler = compiler
        self._dspy_program_loader = program_loader
        self._dspy_program_saver = program_saver
        self._dspy_module = dspy_module
        self._dspy_token_count_fn = token_count_fn
        self._dspy_program_cache: dict[str, Any] = {}
        self._dspy_default_program_ids: set[int] = set()

    def _train_dspy_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
        rows = self._dspy_f_examples(traces, g=g)
        if not rows:
            raise ValueError(
                "DSPy train_f requires at least one observed root target; "
                "set optimizer='none' only for explicit inference/offline compatibility"
            )
        self._dspy_check_training_budgets(kind="f", rows=rows)
        trainset, valset, split = self._dspy_split_examples(rows)
        program = self._dspy_resolve_program(
            f_init,
            kind="f",
            fallback=self._dspy_f_program,
            create=True,
        )
        metric = self._dspy_f_metric
        compiled = self._dspy_compile(
            kind="f",
            program=program,
            metric=metric,
            trainset=trainset,
            valset=valset,
        )
        self._dspy_propagate_default_program_provenance(program, compiled)
        program_path, program_persistence = self._dspy_save_program(
            compiled,
            output_dir=Path(output_dir),
            kind="f",
            iteration=iteration,
        )
        self._dspy_f_program = compiled
        artifact = dict(
            self._dspy_learned_artifact(
                kind="f",
                iteration=iteration,
                rows=rows,
                split=split,
                program_path=program_path,
                program_persistence=program_persistence,
            )
        )
        source_order = (
            "identity_raw_text",
            "preference_prompt",
            "reference_state",
            "current_g_generated",
        )
        train_source_counts = {
            source: sum(
                str(getattr(row, "f_state_source", "unknown")) == source for row in trainset
            )
            for source in source_order
        }
        val_source_counts = {
            source: sum(str(getattr(row, "f_state_source", "unknown")) == source for row in valset)
            for source in source_order
        }
        artifact.update(
            {
                "f_record_source_requested": str(self.dspy_config.f_record_source),
                "f_state_sources": [
                    source for source in source_order if train_source_counts[source]
                ],
                "f_state_source_training_counts": train_source_counts,
                "f_state_source_validation_counts": val_source_counts,
                "f_training_row_count_scope": "optimizer_compile_trainset_only",
                "f_current_g_generation_executed": bool(train_source_counts["current_g_generated"]),
            }
        )
        self._last_f = artifact
        return artifact

    def _train_dspy_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        from treepo.methods.runtime import GTrainOutcome

        scheduled_rate = self._dspy_scheduled_sampling_rate(iteration=iteration)
        current_g_program = None
        generated_states_by_trace: dict[int, Mapping[str, str]] = {}
        if scheduled_rate > 0.0:
            current_g_program = self._dspy_resolve_program(
                g_init,
                kind="g",
                fallback=self._dspy_g_program,
                create=False,
            )
            if current_g_program is not None:
                for trace_index, trace in enumerate(traces):
                    record = TreeRecord.from_value(trace)
                    if not record.nodes:
                        continue
                    topology = binary_topology(record)
                    generated_states_by_trace[trace_index] = self._dspy_tree_states(
                        current_g_program,
                        record,
                        topology=topology,
                    )
        applied_scheduled_rate = scheduled_rate if current_g_program is not None else 0.0
        rows = self._dspy_g_examples(
            traces,
            generated_states_by_trace=generated_states_by_trace,
            scheduled_sampling_rate=applied_scheduled_rate,
        )
        if not rows:
            raise ValueError(
                "DSPy train_g requires at least one observed node target; "
                "set optimizer='none' only for explicit inference/offline compatibility"
            )
        self._dspy_check_training_budgets(kind="g", rows=rows)
        trainset, valset, split = self._dspy_split_examples(rows)
        f_program = self._dspy_resolve_program(
            f,
            kind="f",
            fallback=self._dspy_f_program,
            create=False,
        )
        if f_program is None:
            raise ValueError("DSPy train_g requires the current compiled f program as its judge")
        g_program = current_g_program
        if g_program is None:
            g_program = self._dspy_resolve_program(
                g_init,
                kind="g",
                fallback=self._dspy_g_program,
                create=True,
            )

        def metric(gold: Any, pred: Any, *args: Any, **kwargs: Any) -> float:
            del args, kwargs
            candidate_state = _prediction_state(pred)
            if not candidate_state.strip():
                return 0.0
            try:
                prediction = self._dspy_apply_f(f_program, candidate_state)
                target = self._dspy_decode_target(getattr(gold, "target_json", None))
                distance = oracle_vector_l1(
                    prediction,
                    target,
                    target_names=self._dspy_target_names(),
                )
            except (TypeError, ValueError):
                return 0.0
            weight = _finite_positive(getattr(gold, "effective_weight", 1.0), default=1.0)
            return self._dspy_weighted_reward(distance=distance, weight=weight)

        compiled = self._dspy_compile(
            kind="g",
            program=g_program,
            metric=metric,
            trainset=trainset,
            valset=valset,
        )
        self._dspy_propagate_default_program_provenance(g_program, compiled)
        program_path, program_persistence = self._dspy_save_program(
            compiled,
            output_dir=Path(output_dir),
            kind="g",
            iteration=iteration,
        )
        self._dspy_g_program = compiled
        artifact = dict(
            self._dspy_learned_artifact(
                kind="g",
                iteration=iteration,
                rows=rows,
                split=split,
                program_path=program_path,
                program_persistence=program_persistence,
            )
        )
        leaf_count = sum(str(getattr(row, "role", "")) == "leaf" for row in trainset)
        recompression_count = sum(
            str(getattr(row, "role", "")) == "recompression" for row in trainset
        )
        merge_count = sum(str(getattr(row, "role", "")) == "merge" for row in trainset)
        leaf_val_count = sum(str(getattr(row, "role", "")) == "leaf" for row in valset)
        recompression_val_count = sum(
            str(getattr(row, "role", "")) == "recompression" for row in valset
        )
        merge_val_count = sum(str(getattr(row, "role", "")) == "merge" for row in valset)
        identity_target_count = sum(
            bool(getattr(row, "identity_g_target", False)) for row in trainset
        )
        target_sources = sorted(
            {str(getattr(row, "g_target_source", "unknown")) for row in trainset}
        )
        scheduled_parent_count = sum(
            str(getattr(row, "role", "")) in {"recompression", "merge"} for row in rows
        )
        scheduled_used_count = sum(
            bool(getattr(row, "used_generated_children", False)) for row in rows
        )
        scheduled_sampling = {
            "configured_rate_cap": float(self.dspy_config.g_scheduled_sampling_rate),
            "configured_rate_start": float(self.dspy_config.g_scheduled_sampling_rate_start),
            "configured_ramp_per_iter": float(self.dspy_config.g_scheduled_sampling_ramp_per_iter),
            "effective_rate": float(applied_scheduled_rate),
            "selection_seed": int(self.dspy_config.split_seed),
            "selection_method": "sha256_64_per_tree_child",
            "current_g_generation_executed": bool(generated_states_by_trace),
            "generated_node_state_count": int(
                sum(len(states) for states in generated_states_by_trace.values())
            ),
            "parent_row_count": int(scheduled_parent_count),
            "used_generated_children_row_count": int(scheduled_used_count),
        }
        artifact.update(
            {
                "g_training_call_roles": [
                    role
                    for role, count in (
                        ("leaf", leaf_count),
                        ("recompression", recompression_count),
                        ("merge", merge_count),
                    )
                    if count
                ],
                "leaf_domain_training_row_count": int(leaf_count),
                "recompression_training_row_count": int(recompression_count),
                "merge_domain_training_row_count": int(merge_count),
                "leaf_domain_validation_row_count": int(leaf_val_count),
                "recompression_validation_row_count": int(recompression_val_count),
                "merge_domain_validation_row_count": int(merge_val_count),
                "g_training_row_count_scope": "optimizer_compile_trainset_only",
                "identity_g_target_row_count": int(identity_target_count),
                "g_target_sources": target_sources,
                "g_target_source": (target_sources[0] if len(target_sources) == 1 else "mixed"),
                "scheduled_sampling_rate": float(applied_scheduled_rate),
                "scheduled_sampling_parent_row_count": int(scheduled_parent_count),
                "scheduled_sampling_used_generated_children_row_count": int(scheduled_used_count),
                "g_scheduled_sampling": scheduled_sampling,
                "g_training_role_evidence_source": "dspy_compiler_mixed_examples",
                "same_g_across_node_roles": True,
                "reduce_g_is_derived": True,
                "compile_status": "success",
                "update_performed": True,
                # Compiling over both call domains is optimizer support, not a
                # faithful witness of universal state-level C3 closure.
                "universal_c3_established": False,
            }
        )
        self._last_g = artifact
        return GTrainOutcome(
            artifact=artifact,
            update_performed=True,
            reason="dspy_optimizer_compile_saved_nonempty_trainset",
        )

    def _score_roots_with_dspy(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[Any],
    ) -> list[Any | None] | None:
        f_program = self._dspy_resolve_program(
            f,
            kind="f",
            fallback=self._dspy_f_program,
            create=False,
        )
        if f_program is None:
            return None
        g_program = self._dspy_resolve_program(
            g,
            kind="g",
            fallback=self._dspy_g_program,
            create=False,
        )
        out: list[Any | None] = []
        for trace in trees:
            state = (
                self._dspy_reduce_tree(g_program, trace)
                if g_program is not None
                else _trace_text(trace)
            )
            if not state.strip():
                out.append(None)
                continue
            try:
                out.append(self._dspy_apply_f(f_program, state))
            except (TypeError, ValueError):
                out.append(None)
        return out

    def _validate_dspy_artifact(self, *, kind: str, artifact: Any) -> bool:
        if not isinstance(artifact, Mapping) or not artifact.get("program_path"):
            return False
        trained = str(artifact.get("trained") or kind)
        if trained != str(kind):
            raise ValueError(f"DSPy artifact trained={trained!r} cannot be used as {kind!r}")
        self._dspy_validate_artifact_contract(artifact, kind=kind)
        path = Path(str(artifact["program_path"]))
        if not path.exists():
            raise RuntimeError(f"DSPy {kind} program artifact does not exist: {path}")
        expected_digest = artifact.get("program_sha256")
        if expected_digest is not None and str(expected_digest) != _path_sha256(path):
            raise RuntimeError(f"DSPy {kind} program artifact digest mismatch: {path}")
        self._dspy_resolve_program(artifact, kind=kind, fallback=None, create=False)
        return True

    def _dspy_f_examples(self, traces: Sequence[Any], *, g: Any) -> list[Any]:
        return self._dspy_f_record_examples(traces, g=g)

    def _dspy_g_examples(
        self,
        traces: Sequence[Any],
        *,
        generated_states_by_trace: Mapping[int, Mapping[str, str]] | None = None,
        scheduled_sampling_rate: float = 0.0,
    ) -> list[Any]:
        records = g_training_records(
            traces,
            config=self.dspy_config,
            parse_target=self._parse_prediction,
            generated_states_by_trace=generated_states_by_trace,
            scheduled_sampling_rate=scheduled_sampling_rate,
            scheduled_sampling_seed=int(self.dspy_config.split_seed),
        )
        return [self._dspy_record_example(row) for row in records]

    def _dspy_scheduled_sampling_rate(self, *, iteration: int) -> float:
        cap = float(self.dspy_config.g_scheduled_sampling_rate)
        if cap <= 0.0:
            return 0.0
        start = float(self.dspy_config.g_scheduled_sampling_rate_start)
        ramp = float(self.dspy_config.g_scheduled_sampling_ramp_per_iter)
        if start <= 0.0 and ramp <= 0.0:
            return cap
        rate = start + max(0, int(iteration)) * ramp
        return float(min(cap, max(0.0, rate)))

    def _dspy_budget_enabled(self) -> bool:
        return self.dspy_config.lm_context_window_tokens is not None

    def _dspy_available_input_tokens(self) -> int:
        if not self._dspy_budget_enabled():
            raise RuntimeError("DSPy no-truncation budget is not configured")
        return (
            int(self.dspy_config.lm_context_window_tokens)
            - int(self.dspy_config.max_completion_tokens)
            - int(self.dspy_config.prompt_template_overhead_tokens)
        )

    def _dspy_count_tokens(self, text: Any) -> int:
        rendered = str(text or "")
        counter = self._dspy_token_count_fn
        if counter is None:
            # Portable conservative proxy for ordinary byte-backed subword
            # tokenizers. It intentionally counts bytes rather than using the
            # common chars/4 estimate. Exact model-tokenizer parity requires
            # an injected token_count_fn and is disclosed separately below.
            return len(rendered.encode("utf-8"))
        count = counter(rendered)
        if isinstance(count, bool):
            raise TypeError("DSPy token_count_fn must return a non-negative integer")
        try:
            parsed = int(count)
        except (TypeError, ValueError) as exc:
            raise TypeError("DSPy token_count_fn must return a non-negative integer") from exc
        if parsed < 0 or isinstance(count, float) and not float(count).is_integer():
            raise ValueError("DSPy token_count_fn must return a non-negative integer")
        return parsed

    def _dspy_assert_input_budget(
        self,
        *,
        label: str,
        fields: Mapping[str, Any],
    ) -> None:
        if not self._dspy_budget_enabled():
            return
        counts = {str(name): self._dspy_count_tokens(value) for name, value in fields.items()}
        total = int(sum(counts.values()))
        available = self._dspy_available_input_tokens()
        if total > available:
            raise RuntimeError(
                f"DSPy no-truncation budget failed for {label}: actual "
                f"input/demo-output tokens={total}, available input "
                f"budget={available} (lm_context_window_tokens="
                f"{self.dspy_config.lm_context_window_tokens} - "
                f"max_completion_tokens={self.dspy_config.max_completion_tokens} - "
                "prompt_template_overhead_tokens="
                f"{self.dspy_config.prompt_template_overhead_tokens}); "
                f"field_counts={counts}"
            )

    def _dspy_check_training_budgets(
        self,
        *,
        kind: str,
        rows: Sequence[Any],
    ) -> None:
        if not self._dspy_budget_enabled():
            return
        for index, row in enumerate(rows):
            values = row.toDict() if callable(getattr(row, "toDict", None)) else dict(row)
            if kind == "f":
                fields = {
                    "state": values.get("state", ""),
                    "prediction_json_demo_output": values.get("prediction_json", ""),
                }
            elif kind == "g":
                fields = {
                    "prompt": values.get("prompt", ""),
                    "state_demo_output": values.get("state", ""),
                }
            else:
                raise ValueError(f"unknown DSPy budget kind: {kind!r}")
            self._dspy_assert_input_budget(
                label=f"{kind} training record {index}",
                fields=fields,
            )

    def _dspy_budget_artifact(self) -> Mapping[str, Any]:
        if not self._dspy_budget_enabled():
            return {"enabled": False}
        return {
            "enabled": True,
            "leaf_size_tokens": int(self.dspy_config.leaf_size_tokens),
            "lm_context_window_tokens": int(self.dspy_config.lm_context_window_tokens),
            "max_completion_tokens": int(self.dspy_config.max_completion_tokens),
            "prompt_template_overhead_tokens": int(
                self.dspy_config.prompt_template_overhead_tokens
            ),
            "available_input_tokens": self._dspy_available_input_tokens(),
            "coverage": "per_training_record_and_direct_program_input",
            "compiled_demo_stack_verified": False,
            "absolute_prompt_fit_guarantee": False,
            "overhead_reservation_semantics": (
                "prompt_template_overhead_tokens must conservatively cover "
                "signature rendering and any optimizer-selected demo stack; "
                "treepo cannot inspect every provider/compiler-rendered prompt"
            ),
            "token_count_method": (
                "injected_exact_counter"
                if self._dspy_token_count_fn is not None
                else "conservative_utf8_byte_count_proxy"
            ),
        }

    def _dspy_f_metric(self, gold: Any, pred: Any, *args: Any, **kwargs: Any) -> float:
        del args, kwargs
        try:
            prediction = self._parse_prediction(pred)
            target = self._dspy_decode_target(getattr(gold, "target_json", None))
            distance = oracle_vector_l1(
                prediction,
                target,
                target_names=self._dspy_target_names(),
            )
        except (TypeError, ValueError):
            return 0.0
        weight = _finite_positive(getattr(gold, "effective_weight", 1.0), default=1.0)
        return self._dspy_weighted_reward(distance=distance, weight=weight)

    def _dspy_apply_f(self, program: Any, state: str) -> Any:
        self._dspy_assert_input_budget(
            label="f program input",
            fields={"state": state},
        )
        raw = _invoke_program(program, state=state)
        prediction = self._parse_prediction(raw)
        if prediction is None:
            raise ValueError("current DSPy f program returned no valid prediction")
        return prediction

    def _dspy_tree_states(
        self, g_program: Any, record: TreeRecord, *, topology: BinaryTopology
    ) -> dict[str, str]:
        cache: dict[str, str] = {}

        def reduce_node(node: Any) -> str:
            node_id = str(node.node_id)
            if node_id in cache:
                return cache[node_id]
            children = topology.children(node)
            if not children:
                prompt = leaf_prompt(str(node.text or ""))
            else:
                child_states = [reduce_node(child) for child in children]
                prompt = merge_prompt(
                    child_states[0], child_states[1] if len(child_states) > 1 else None
                )
            self._dspy_assert_input_budget(
                label=f"g program input for node {node_id}",
                fields={"prompt": prompt},
            )
            state = _prediction_state(_invoke_program(g_program, prompt=prompt))
            if not state:
                raise ValueError(f"DSPy g returned an empty state for node {node_id!r}")
            cache[node_id] = state
            return state

        reduce_node(topology.root)
        return cache

    def _dspy_reduce_tree(self, g_program: Any, trace: Any) -> str:
        record = TreeRecord.from_value(trace)
        if not record.nodes:
            prompt = leaf_prompt(_trace_text(trace))
            self._dspy_assert_input_budget(
                label="g program input for node-less trace",
                fields={"prompt": prompt},
            )
            return _prediction_state(_invoke_program(g_program, prompt=prompt))
        topology = binary_topology(record)
        states = self._dspy_tree_states(g_program, record, topology=topology)
        return states[str(topology.root.node_id)]

    def _dspy_trace_target(self, trace: Any) -> Any | None:
        metadata = _metadata(trace)
        candidates: list[Any] = []
        for key in (
            self.dspy_config.target_vector_key,
            self.dspy_config.target_key,
            "oracle_target",
            "target_vector",
            "topic_proportions",
            "teacher_score_native",
            "expert_score",
        ):
            if key:
                candidates.append(metadata.get(str(key)))
                candidates.append(getattr(trace, str(key), None))
        candidates.extend(
            [
                getattr(trace, "root_label", None),
                getattr(trace, "document_score", None),
            ]
        )
        record = TreeRecord.from_value(trace)
        root = record.root()
        if root is not None:
            candidates.extend([self._dspy_node_target(root), root.label])
        return self._dspy_first_target(candidates)

    def _dspy_node_target(self, node: Any) -> Any | None:
        metadata = _metadata(node)
        candidates: list[Any] = []
        for key in (
            self.dspy_config.node_target_key,
            self.dspy_config.target_vector_key,
            "oracle_target",
            "target_vector",
            "topic_proportions",
            "score",
            "oracle_score",
            "teacher_score_native",
        ):
            if key:
                candidates.append(metadata.get(str(key)))
                candidates.append(getattr(node, str(key), None))
        candidates.extend([getattr(node, "label", None), getattr(node, "state", None)])
        return self._dspy_first_target(candidates)

    def _dspy_first_target(self, candidates: Sequence[Any]) -> Any | None:
        for candidate in candidates:
            if candidate is None:
                continue
            parsed = self._parse_prediction(candidate)
            if parsed is not None:
                return parsed
        return None

    def _dspy_decode_target(self, value: Any) -> Any:
        if not isinstance(value, (str, bytes)):
            parsed = self._parse_prediction(value)
        else:
            parsed = self._parse_prediction(value.decode() if isinstance(value, bytes) else value)
        if parsed is None:
            raise ValueError("DSPy optimizer example has an invalid target_json")
        return parsed

    def _dspy_target_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in (self.dspy_config.target_names or ()))

    def _dspy_weighted_reward(self, *, distance: float, weight: float) -> float:
        if not math.isfinite(float(distance)) or float(distance) < 0.0:
            return 0.0
        return float(weight) / (1.0 + float(distance))

    def _dspy_f_record_examples(self, traces: Sequence[Any], *, g: Any) -> list[Any]:
        source = str(self.dspy_config.f_record_source)
        allow_identity_g_input = _is_identity_g_artifact(g)
        g_program = None
        if source == "generated_when_available":
            g_program = self._dspy_resolve_program(
                g,
                kind="g",
                fallback=self._dspy_g_program,
                create=False,
            )
        if source not in {"gold_state", "generated_when_available"}:
            raise ValueError(f"unknown DSPy f_record_source: {source!r}")
        reducer = None
        if g_program is not None:

            def use_generated(record: TreeRecord, topology: BinaryTopology) -> Mapping[str, str]:
                return self._dspy_tree_states(
                    g_program,
                    record,
                    topology=topology,
                )

            reducer = use_generated
        records = f_training_records(
            traces,
            config=self.dspy_config,
            parse_target=self._parse_prediction,
            reduce_states=reducer,
            allow_identity_g_input=allow_identity_g_input,
        )
        return [self._dspy_record_example(row) for row in records]

    def _dspy_record_example(self, row: Mapping[str, Any]) -> Any:
        values = dict(row)
        return self._dspy_example(inputs=values.pop("inputs"), **values)

    def _dspy_example(self, *, inputs: Sequence[str], **values: Any) -> Any:
        from treepo.methods.dspy import _DSPyTrainingExample

        return _DSPyTrainingExample(**values).with_inputs(*inputs)

    def _dspy_split_examples(
        self,
        rows: Sequence[Any],
    ) -> tuple[list[Any], list[Any], Mapping[str, Any]]:
        return split_examples(
            rows,
            validation_fraction=float(self.dspy_config.validation_fraction),
            seed=int(self.dspy_config.split_seed),
        )

    def _dspy_compile(
        self,
        *,
        kind: str,
        program: Any,
        metric: Any,
        trainset: Sequence[Any],
        valset: Sequence[Any],
    ) -> Any:
        if not trainset:
            raise ValueError("DSPy optimizer compile requires a non-empty trainset")
        if self._dspy_compiler is not None:
            compile_fn = getattr(self._dspy_compiler, "compile", self._dspy_compiler)
            if not callable(compile_fn):
                raise TypeError("dspy compiler must be callable or expose compile()")
            compiled = _invoke_supported(
                compile_fn,
                {
                    "program": program,
                    "student": program,
                    "metric": metric,
                    "trainset": list(trainset),
                    "valset": list(valset),
                    "kind": str(kind),
                    "family": self,
                    "config": self.dspy_config,
                },
            )
            if compiled is None:
                raise RuntimeError("injected DSPy compiler returned None")
            return compiled

        dspy = self._require_dspy()
        native_train = [_native_example(dspy, row) for row in trainset]
        native_val = [_native_example(dspy, row) for row in valset]
        lm = self._dspy_lm(dspy)
        context = (
            dspy.context(lm=lm) if lm is not None and hasattr(dspy, "context") else nullcontext()
        )
        with context:
            optimizer = self._dspy_native_optimizer(dspy, metric, lm=lm)
            try:
                compile_parameters = inspect.signature(optimizer.compile).parameters
            except (TypeError, ValueError):
                compile_parameters = {"valset": None}
            compile_kwargs: dict[str, Any] = {"trainset": native_train}
            if "valset" in compile_parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in compile_parameters.values()
                if parameter is not None
            ):
                compile_kwargs["valset"] = native_val
            compiled = optimizer.compile(program, **compile_kwargs)
        if compiled is None:
            raise RuntimeError("DSPy optimizer returned None")
        return compiled

    def _dspy_native_optimizer(self, dspy: Any, metric: Any, *, lm: Any = None) -> Any:
        name = str(self.dspy_config.optimizer)
        kwargs = dict(self.dspy_config.optimizer_kwargs or {})
        if name == "bootstrap":
            cls = getattr(dspy, "BootstrapFewShot")
            return cls(metric=metric, **kwargs)
        if name == "bootstrap_random_search":
            cls = getattr(dspy, "BootstrapFewShotWithRandomSearch")
            return cls(metric=metric, **kwargs)
        if name == "mipro":
            cls = getattr(dspy, "MIPROv2")
            kwargs.setdefault("auto", str(self.dspy_config.optimizer_budget))
            return cls(metric=metric, **kwargs)
        if name == "gepa":
            cls = getattr(dspy, "GEPA")
            kwargs.setdefault("auto", str(self.dspy_config.optimizer_budget))
            if lm is not None:
                kwargs.setdefault("reflection_lm", lm)
            return cls(metric=metric, **kwargs)
        raise ValueError(f"unsupported DSPy optimizer: {name!r}")

    def _dspy_lm(self, dspy: Any) -> Any:
        lm_config = dict(self.dspy_config.lm_config or {})
        explicit_lm = lm_config.get("lm", lm_config.get("current_lm"))
        if explicit_lm is not None:
            return explicit_lm
        if lm_config.get("use_current_dspy_lm") is True:
            return getattr(getattr(dspy, "settings", None), "lm", None)
        model = str(lm_config.get("model") or "").strip()
        if not model or not hasattr(dspy, "LM"):
            return None
        default_max_tokens = self.dspy_config.max_completion_tokens or 4096
        kwargs = {
            "model": model,
            "api_base": lm_config.get("api_base") or lm_config.get("base_url"),
            "api_key": lm_config.get("api_key"),
            "temperature": lm_config.get("temperature", 0.0),
            "max_tokens": lm_config.get("max_tokens", default_max_tokens),
        }
        return _invoke_supported(
            dspy.LM, {key: value for key, value in kwargs.items() if value is not None}
        )

    def _dspy_resolve_program(
        self,
        artifact: Any,
        *,
        kind: str,
        fallback: Any,
        create: bool,
    ) -> Any:
        if kind == "g" and _is_identity_g_artifact(artifact):
            return None
        if artifact is not None and not isinstance(artifact, Mapping):
            return artifact
        path_text = None
        if isinstance(artifact, Mapping):
            path_text = artifact.get("program_path")
        if path_text:
            self._dspy_validate_artifact_contract(artifact, kind=kind)
            path = Path(str(path_text))
            cache_key = str(path.resolve())
            if cache_key in self._dspy_program_cache:
                return self._dspy_program_cache[cache_key]
            if not path.exists():
                raise RuntimeError(f"DSPy {kind} program artifact does not exist: {path}")
            expected_digest = artifact.get("program_sha256")
            if expected_digest is not None and str(expected_digest) != _path_sha256(path):
                raise RuntimeError(f"DSPy {kind} program artifact digest mismatch: {path}")
            reconstruction = artifact.get("program_reconstruction")
            if reconstruction is None:
                warnings.warn(
                    "Loading a legacy DSPy artifact without program_reconstruction; "
                    "assuming the historical state-only default Predict contract. "
                    "Re-save custom programs with a program_loader.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                reconstruction = (
                    "program_loader_v1"
                    if self._dspy_program_loader is not None
                    else "treepo_default_predict_v1"
                )
            reconstruction = str(reconstruction)
            if reconstruction == "program_loader_v1":
                if self._dspy_program_loader is None:
                    raise RuntimeError(
                        f"DSPy {kind} artifact requires program_loader for fresh-process "
                        f"reconstruction: {path}"
                    )
                loaded = _invoke_supported(
                    self._dspy_program_loader,
                    {"path": path, "kind": kind, "family": self},
                )
            elif reconstruction == "treepo_default_predict_v1":
                dspy = self._require_dspy()
                loaded = self._dspy_new_program(kind, dspy=dspy)
                expected_class = artifact.get("program_class")
                if expected_class is not None and str(expected_class) != _qualified_class_name(
                    loaded
                ):
                    raise RuntimeError(
                        f"DSPy {kind} state artifact expects class {expected_class!r}, "
                        f"not {_qualified_class_name(loaded)!r}"
                    )
                loaded.load(str(path))
            elif reconstruction == "dspy_load_v1":
                if not self.dspy_config.allow_pickle_program_load:
                    raise RuntimeError(
                        f"DSPy {kind} artifact requires pickle loading; set "
                        "allow_pickle_program_load=True only for trusted artifacts"
                    )
                dspy = self._require_dspy()
                load = getattr(dspy, "load", None)
                if not callable(load):
                    raise RuntimeError("installed DSPy does not expose dspy.load()")
                loaded = _invoke_supported(
                    load,
                    {
                        "path": str(path),
                        # DSPy >=3.2 requires this explicit argument; DSPy 3.0.4
                        # does not accept it, and _invoke_supported filters it.
                        "allow_pickle": True,
                    },
                )
            else:
                raise RuntimeError(
                    f"unknown DSPy program_reconstruction={reconstruction!r} "
                    f"for {kind} artifact {path}"
                )
            if loaded is None:
                raise RuntimeError(f"DSPy {kind} loader returned None for {path}")
            expected_class = artifact.get("program_class")
            if expected_class is not None and str(expected_class) != _qualified_class_name(loaded):
                raise RuntimeError(
                    f"DSPy {kind} artifact class mismatch: expected "
                    f"{expected_class!r}, loaded {_qualified_class_name(loaded)!r}"
                )
            self._dspy_program_cache[cache_key] = loaded
            return loaded
        if fallback is not None:
            return fallback
        if not create:
            return None
        return self._dspy_new_program(kind)

    def _dspy_new_program(self, kind: str, *, dspy: Any = None) -> Any:
        module = dspy if dspy is not None else self._require_dspy()
        program = self._dspy_build_default_program(module, kind=kind)
        self._dspy_default_program_ids.add(id(program))
        return program

    def _dspy_build_default_program(self, module: Any, *, kind: str) -> Any:
        signature_factory = getattr(module, "Signature", None)
        input_field = getattr(module, "InputField", None)
        output_field = getattr(module, "OutputField", None)
        has_native_signature = all(
            callable(value) for value in (signature_factory, input_field, output_field)
        )
        if not has_native_signature:

            def ignored_field(**_kwargs: Any) -> None:
                return None

            input_field = ignored_field
            output_field = ignored_field
        if kind == "f":
            specification = "state -> prediction_json"
            instructions = self._dspy_signature_instructions("f")
            field_values = {
                "state": input_field(
                    desc="One task state produced by identity g or the shared g program."
                ),
                "prediction_json": output_field(
                    desc=(
                        "Exactly one JSON object containing every declared named "
                        "target coordinate and no extra coordinates."
                    )
                ),
            }
        elif kind == "g":
            specification = "prompt -> state"
            instructions = self._dspy_signature_instructions("g")
            field_values = {
                "prompt": input_field(
                    desc="A role-tagged leaf, unary recompression, or binary merge call."
                ),
                "state": output_field(
                    desc="One task state consumable by the same shared g and by f."
                ),
            }
        else:
            raise ValueError(f"unknown DSPy program kind: {kind!r}")
        if has_native_signature:
            try:
                signature = signature_factory(
                    field_values,
                    instructions=str(instructions),
                )
            except TypeError:
                signature = signature_factory(
                    specification,
                    instructions=str(instructions),
                )
            return module.Predict(signature)
        # Compatibility with minimal downstream DSPy shims. Native DSPy uses
        # the Signature path above.
        try:
            return module.Predict(specification, instructions=str(instructions))
        except TypeError:
            return module.Predict(specification)

    def _dspy_signature_instructions(self, kind: str) -> str:
        configured = getattr(
            self.dspy_config,
            f"{kind}_signature_instructions",
            None,
        )
        if kind not in {"f", "g"}:
            raise ValueError(f"unknown DSPy program kind: {kind!r}")
        return str(configured or self._dspy_default_signature_instructions(kind))

    def _dspy_default_signature_instructions(self, kind: str) -> str:
        names = self._dspy_target_names()
        oracle_ids = tuple(str(value) for value in self.dspy_config.target_oracle_ids)
        catalog = [
            {"target_name": name, "oracle_id": oracle_id}
            for name, oracle_id in zip(names, oracle_ids)
        ]
        rendered_catalog = json.dumps(catalog, separators=(",", ":"))
        if kind == "f":
            return (
                "Read one task state and emit the single declared joint target "
                "vector as a strict JSON object. Include every target_name exactly "
                "once in declared order, add no coordinates, and never return a "
                "scalar or positional array, including when K=1. The optimizer "
                "uses unnormalized sum-L1 over coordinates. Declared target/oracle "
                f"catalog: {rendered_catalog}."
            )
        if kind == "g":
            return (
                "Implement one shared task-state operator g for every tagged call "
                "role: leaf reduction, unary recompression, and binary merge. "
                "Preserve the information needed by the same joint readout f for "
                "all declared targets; do not create role-specific g programs. "
                "Unary recompression is the C2 domain and binary merge is the C3 "
                "training domain. Declared target/oracle catalog: "
                f"{rendered_catalog}."
            )
        raise ValueError(f"unknown DSPy program kind: {kind!r}")

    def _dspy_signature_contract(self, kind: str) -> Mapping[str, Any]:
        instructions = self._dspy_signature_instructions(kind)
        if kind == "f":
            specification = "state -> prediction_json"
        elif kind == "g":
            specification = "prompt -> state"
        else:
            raise ValueError(f"unknown DSPy program kind: {kind!r}")
        return {
            "version": "treepo.dspy.signature_contract.v1",
            "kind": str(kind),
            "specification": specification,
            "target_names": list(self._dspy_target_names()),
            "target_oracle_ids": [str(value) for value in self.dspy_config.target_oracle_ids],
            "instructions_source": (
                "configured_task_owned"
                if getattr(self.dspy_config, f"{kind}_signature_instructions")
                else "task_neutral_default"
            ),
            "instructions_sha256": hashlib.sha256(instructions.encode("utf-8")).hexdigest(),
        }

    def _dspy_validate_artifact_contract(
        self,
        artifact: Mapping[str, Any],
        *,
        kind: str,
    ) -> None:
        expected_names = self._dspy_target_names()
        if artifact.get("target_names") is not None:
            observed_names = tuple(str(value) for value in artifact["target_names"])
            if observed_names != expected_names:
                raise ValueError(
                    f"DSPy {kind} artifact target_names={observed_names!r} do not "
                    f"match current target_names={expected_names!r}"
                )
        expected_oracles = tuple(str(value) for value in self.dspy_config.target_oracle_ids)
        if artifact.get("target_oracle_ids") is not None:
            observed_oracles = tuple(str(value) for value in artifact["target_oracle_ids"])
            if observed_oracles != expected_oracles:
                raise ValueError(
                    f"DSPy {kind} artifact target_oracle_ids={observed_oracles!r} "
                    f"do not match current target_oracle_ids={expected_oracles!r}"
                )
        contract = artifact.get("signature_contract")
        if contract is None:
            return
        if not isinstance(contract, Mapping):
            raise TypeError("DSPy artifact signature_contract must be a mapping")
        if str(contract.get("kind") or "") != str(kind):
            raise ValueError(
                f"DSPy artifact signature_contract kind={contract.get('kind')!r} "
                f"cannot be used as {kind!r}"
            )
        reconstruction = str(artifact.get("program_reconstruction") or "")
        if reconstruction != "treepo_default_predict_v1":
            return
        expected_contract = self._dspy_signature_contract(kind)
        for field in (
            "version",
            "specification",
            "target_names",
            "target_oracle_ids",
            "instructions_sha256",
        ):
            if contract.get(field) != expected_contract[field]:
                raise ValueError(
                    f"DSPy {kind} artifact signature_contract.{field} is "
                    "incompatible with the current family configuration"
                )

    def _dspy_propagate_default_program_provenance(
        self,
        source: Any,
        compiled: Any,
    ) -> None:
        if id(source) in self._dspy_default_program_ids and type(compiled) is type(source):
            self._dspy_default_program_ids.add(id(compiled))

    def _dspy_save_program(
        self,
        program: Any,
        *,
        output_dir: Path,
        kind: str,
        iteration: int,
    ) -> tuple[Path, Mapping[str, Any]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        program_class = _qualified_class_name(program)
        if self._dspy_program_saver is not None and self._dspy_program_loader is None:
            raise ValueError(
                "program_saver requires program_loader so fresh-process DSPy "
                "artifacts cannot fall back to a bare Predict"
            )

        if self._dspy_program_loader is not None:
            path = output_dir / f"{kind}_dspy_iter_{int(iteration):02d}.json"
            persistence = {
                "program_format": "custom_program",
                "program_reconstruction": "program_loader_v1",
                "program_class": program_class,
                "program_requires_pickle": False,
            }
        else:
            dspy = self._require_dspy()
            predict_type = getattr(dspy, "Predict", None)
            if (
                isinstance(predict_type, type)
                and type(program) is predict_type
                and id(program) in self._dspy_default_program_ids
            ):
                path = output_dir / f"{kind}_dspy_iter_{int(iteration):02d}.json"
                persistence = {
                    "program_format": "dspy_json_state",
                    "program_reconstruction": "treepo_default_predict_v1",
                    "program_class": program_class,
                    "program_requires_pickle": False,
                }
            else:
                module_type = getattr(dspy, "Module", None)
                if not isinstance(module_type, type) or not isinstance(program, module_type):
                    raise TypeError(
                        "injected non-DSPy programs require program_loader and either "
                        "program_saver or a compatible save() method"
                    )
                elif not self.dspy_config.allow_pickle_program_load:
                    raise ValueError(
                        "custom DSPy Module persistence requires "
                        "allow_pickle_program_load=True for trusted artifacts, or "
                        "explicit program_loader/program_saver callbacks"
                    )
                else:
                    path = output_dir / f"{kind}_dspy_iter_{int(iteration):02d}"
                    persistence = {
                        "program_format": "dspy_whole_program_pickle",
                        "program_reconstruction": "dspy_load_v1",
                        "program_class": program_class,
                        "program_requires_pickle": True,
                    }

        if self._dspy_program_saver is not None:
            _invoke_supported(
                self._dspy_program_saver,
                {"program": program, "path": path, "kind": kind, "family": self},
            )
        else:
            save = getattr(program, "save", None)
            if not callable(save):
                raise TypeError("compiled DSPy program must expose save() or use program_saver")
            if persistence["program_reconstruction"] == "dspy_load_v1":
                save(str(path), save_program=True)
            else:
                try:
                    save(str(path), save_program=False)
                except TypeError:
                    save(str(path))
        if not path.exists():
            raise RuntimeError(
                f"DSPy {kind} compile succeeded but save produced no artifact: {path}"
            )
        self._dspy_program_cache[str(path.resolve())] = program
        return path, persistence

    def _dspy_learned_artifact(
        self,
        *,
        kind: str,
        iteration: int,
        rows: Sequence[Any],
        split: Mapping[str, Any],
        program_path: Path,
        program_persistence: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        weights = [float(getattr(row, "effective_weight", 1.0)) for row in rows]
        propensities = [float(getattr(row, "propensity", 1.0)) for row in rows]
        persistence = dict(
            program_persistence
            or {
                "program_format": "dspy_json_state",
                "program_reconstruction": "treepo_default_predict_v1",
                "program_class": None,
                "program_requires_pickle": False,
            }
        )
        return {
            "kind": f"treepo_dspy_{kind}",
            "trained": str(kind),
            "iteration": int(iteration),
            "n_examples": int(len(rows)),
            "n_train": int(split.get("train_count", len(rows))),
            "n_validation": int(split.get("validation_count", 0)),
            "optimizer": str(self.dspy_config.optimizer),
            "optimizer_budget": str(self.dspy_config.optimizer_budget),
            "compile_executed": True,
            "compile_status": "success",
            "program_path": str(program_path),
            **persistence,
            "program_sha256": _path_sha256(program_path),
            "target_names": list(self._dspy_target_names()),
            "target_oracle_ids": [str(value) for value in self.dspy_config.target_oracle_ids],
            "signature_contract": dict(self._dspy_signature_contract(kind)),
            "output_contract": (
                "strict_named_json" if self._dspy_target_names() else "scalar_numeric"
            ),
            "oracle_metric": "sum_l1",
            "no_truncation_budget": dict(self._dspy_budget_artifact()),
            "split": dict(split),
            "weighting": {
                "estimator": (
                    "inverse_propensity_weighted_optimizer_reward"
                    if self.dspy_config.importance_weight_cap is None
                    else "explicitly_capped_inverse_propensity_optimizer_reward"
                ),
                "estimator_claim": "no_horvitz_thompson_or_hajek_claim",
                "min_propensity": float(self.dspy_config.min_propensity),
                "propensity_floor_policy": ("fail_closed_below_min_no_propensity_clipping"),
                "importance_weight_cap": self.dspy_config.importance_weight_cap,
                "importance_weight_policy": (
                    "uncapped"
                    if self.dspy_config.importance_weight_cap is None
                    else "explicit_user_cap"
                ),
                "effective_weight_sum": float(sum(weights)),
                "effective_weight_max": float(max(weights)) if weights else 0.0,
                "propensity_min": float(min(propensities)) if propensities else None,
                "weight_sources": sorted(
                    {str(getattr(row, "weight_source", "unknown")) for row in rows}
                ),
            },
            "dspy_config": _redacted_config(self.dspy_config),
        }

    def _require_dspy(self) -> Any:
        if self._dspy_module is not None:
            return self._dspy_module
        try:
            import dspy
        except ImportError as exc:
            raise ImportError(
                "optimizer-backed DSPy learning requires the 'dspy' extra; "
                "install treepo[dspy] or set optimizer='none' for inference-only use"
            ) from exc
        self._dspy_module = dspy
        return dspy


def _is_identity_g_artifact(artifact: Any) -> bool:
    if not isinstance(artifact, Mapping):
        return False
    return (
        str(artifact.get("kind") or "") == "treepo_identity_g"
        or str(artifact.get("g_mode") or "").lower() == "identity"
        or str(artifact.get("operator") or "").lower() == "identity"
    )


def _invoke_program(program: Any, **values: Any) -> Any:
    if callable(program):
        fn = program
    else:
        fn = getattr(program, "forward", None) or getattr(program, "predict", None)
    if not callable(fn):
        raise TypeError("DSPy program must be callable or expose forward()/predict()")
    return _invoke_supported(fn, values)


def _invoke_supported(fn: Any, values: Mapping[str, Any]) -> Any:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**dict(values))
    parameters = signature.parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return fn(**dict(values))
    kwargs = {
        name: value
        for name, value in values.items()
        if name in parameters
        and parameters[name].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return fn(**kwargs)


def _native_example(dspy: Any, row: Any) -> Any:
    values = row.toDict()
    example = dspy.Example(**values)
    input_names = tuple(getattr(row, "_input_names", ()))
    return example.with_inputs(*input_names)


def _prediction_state(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("state", "completion", "summary", "output", "text", "content"):
            if key in value:
                return _prediction_state(value[key])
        return ""
    for attr in ("state", "completion", "summary", "output", "text", "content"):
        if hasattr(value, attr):
            nested = getattr(value, attr)
            if nested is not value:
                parsed = _prediction_state(nested)
                if parsed:
                    return parsed
    return str(value or "").strip()


def _target_json(value: Any) -> str:
    return json.dumps(state_to_dict(value), separators=(",", ":"), sort_keys=True)


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        metadata = value.get("metadata")
    else:
        metadata = getattr(value, "metadata", None)
    return dict(metadata or {}) if isinstance(metadata, Mapping) else {}


def _trace_text(trace: Any) -> str:
    if isinstance(trace, Mapping):
        value = trace.get("text", trace.get("content", trace.get("document_text", "")))
    else:
        value = getattr(
            trace,
            "text",
            getattr(trace, "content", getattr(trace, "document_text", "")),
        )
    if value:
        return str(value)
    return str(_metadata(trace).get("text") or "")


def _has_explicit_trace_g_reference(
    trace: Any,
    metadata: Mapping[str, Any],
) -> bool:
    if any(
        metadata.get(key) is not None and str(metadata.get(key)).strip()
        for key in (
            "reference_state",
            "summary",
            "teacher_summary",
            "target_summary",
            "completion",
            "response",
        )
    ):
        return True
    state = getattr(trace, "state", None)
    if isinstance(state, str) and state.strip():
        return True
    return str(metadata.get("preference_target") or "").lower() == "g"


def _has_explicit_node_g_reference(node: Any) -> bool:
    metadata = _metadata(node)
    if any(
        metadata.get(key) is not None and str(metadata.get(key)).strip()
        for key in (
            "summary",
            "teacher_summary",
            "target_summary",
            "reference_summary",
            "completion",
            "response",
        )
    ):
        return True
    state = getattr(node, "state", None)
    return isinstance(state, str) and bool(state.strip())


def _trace_g_reference(
    trace: Any,
    metadata: Mapping[str, Any],
    *,
    allow_identity: bool,
) -> str:
    for key in (
        "reference_state",
        "summary",
        "teacher_summary",
        "target_summary",
        "completion",
        "response",
    ):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    state = getattr(trace, "state", None)
    if isinstance(state, str) and state.strip():
        return state.strip()
    if str(metadata.get("preference_target") or "").lower() == "g" or allow_identity:
        return _trace_text(trace).strip()
    return ""


def _reference_state(node: Any, *, fallback: str, allow_fallback: bool) -> str:
    metadata = _metadata(node)
    for key in (
        "summary",
        "teacher_summary",
        "target_summary",
        "reference_summary",
        "completion",
        "response",
    ):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    state = getattr(node, "state", None)
    if isinstance(state, str) and state.strip():
        return state.strip()
    if not allow_fallback:
        return ""
    text = str(getattr(node, "text", "") or "").strip()
    return text or str(fallback or "").strip()


def _node_children(
    record: TreeRecord, node: Any, children_by_parent: Mapping[str, Sequence[Any]]
) -> list[Any]:
    children: list[Any] = []
    for child_id in (getattr(node, "left_child_id", None), getattr(node, "right_child_id", None)):
        child = record.get_node(child_id) if child_id is not None else None
        if child is not None and all(
            str(existing.node_id) != str(child.node_id) for existing in children
        ):
            children.append(child)
    for child in children_by_parent.get(str(node.node_id), ()):
        if all(str(existing.node_id) != str(child.node_id) for existing in children):
            children.append(child)
    return sorted(
        children,
        key=lambda child: (
            -1 if child.position is None else int(child.position),
            str(child.node_id),
        ),
    )


def _leaf_prompt(text: str) -> str:
    return (
        "[TREEPO_G_CALL=leaf]\n"
        "Apply the shared C-Tree state operator g to this leaf. Return only the state.\n"
        f"LEAF_TEXT:\n{text}"
    )


def _merge_prompt(left: str, right: str) -> str:
    return (
        "[TREEPO_G_CALL=merge]\n"
        "Apply the same shared C-Tree state operator g to these two child states. "
        "Return only the merged state.\n"
        f"LEFT_STATE:\n{left}\n\nRIGHT_STATE:\n{right}"
    )


def _split_name(metadata: Mapping[str, Any]) -> str:
    value = metadata.get("split", metadata.get("source_split", ""))
    return str(value or "").strip().lower()


def _row_id(trace: Any, *, index: int, role: str) -> str:
    metadata = _metadata(trace)
    value = (
        metadata.get("preference_unit_id")
        or metadata.get("row_id")
        or metadata.get("tree_id")
        or metadata.get("doc_id")
        or getattr(trace, "tree_id", None)
        or getattr(trace, "doc_id", None)
        or index
    )
    return f"{value}:{role}:{index}"


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = (
        [path]
        if path.is_file()
        else sorted(
            (item for item in path.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(path).as_posix(),
        )
    )
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _qualified_class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _row_digest(rows: Sequence[Any]) -> str:
    payload = json.dumps(
        [str(getattr(row, "row_id", "")) for row in rows],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nonnegative_weight(value: Any) -> float:
    number = _finite_number(value)
    if number is None or number < 0.0:
        raise ValueError(f"DSPy example weight must be finite and non-negative, got {value!r}")
    return float(number)


def _valid_propensity(value: Any) -> float:
    if value is None:
        return 1.0
    number = _finite_number(value)
    if number is None or not 0.0 < number <= 1.0:
        raise ValueError(f"DSPy example propensity must be finite and in (0, 1], got {value!r}")
    return float(number)


def _finite_positive(value: Any, *, default: float) -> float:
    number = _finite_number(value)
    if number is None or number <= 0.0:
        return float(default)
    return float(number)


def _redacted_config(config: Any) -> dict[str, Any]:
    from dataclasses import fields

    payload = {item.name: getattr(config, item.name) for item in fields(config)}
    lm = dict(payload.get("lm_config") or {})
    if lm.get("api_key"):
        lm["api_key"] = "<redacted>"
    for key in ("lm", "current_lm"):
        if lm.get(key) is not None:
            lm[key] = f"<{type(lm[key]).__name__}>"
    payload["lm_config"] = lm
    return payload


__all__ = ["DSPyOptimizerMixin"]
