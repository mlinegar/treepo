"""Registry-driven TreePO contract fit/evaluation runner.

The public surface is contract-first.  Callers describe objectives,
requirements, data, and neutral resources; registered adapters decide whether
they support the contract and own any specialized implementation details.
"""

from __future__ import annotations

import csv
import importlib
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from treepo._research.training.config_sections import RunConfig, TestConfig, TrainConfig, ValidationConfig, config_to_dict
from treepo._research.tree.treepo_stack import (
    OracleLaneSpec,
    SupervisionSourceSpec,
    TreePOContractSpec,
    TreePOModelSpec,
    build_treepo_stack,
)


RESOURCE_GENERATION = "generation"
RESOURCE_EMBEDDING = "embedding"
RESOURCE_SYMBOLIC_REFERENCE = "symbolic_reference"
RESOURCE_LABELED_TREE_ARTIFACT = "labeled_tree_artifact"
RESOURCE_LOCAL_LAW_ORACLE = "local_law_oracle"
RESOURCE_TRAINER = "trainer"

_RESOURCE_ALIASES = {
    "generation": RESOURCE_GENERATION,
    "generation_backend": RESOURCE_GENERATION,
    "backend": RESOURCE_GENERATION,
    "embedding": RESOURCE_EMBEDDING,
    "embedding_client": RESOURCE_EMBEDDING,
    "symbolic_reference": RESOURCE_SYMBOLIC_REFERENCE,
    "reference": RESOURCE_SYMBOLIC_REFERENCE,
    "labeled_tree_artifact": RESOURCE_LABELED_TREE_ARTIFACT,
    "labeled_trees": RESOURCE_LABELED_TREE_ARTIFACT,
    "local_law_oracle": RESOURCE_LOCAL_LAW_ORACLE,
    "score_fn": RESOURCE_LOCAL_LAW_ORACLE,
    "trainer": RESOURCE_TRAINER,
}


def _import_from_path(import_path: str) -> Any:
    module_path, sep, attr_path = str(import_path).partition(":")
    if not sep:
        parts = str(import_path).split(".")
        if len(parts) < 2:
            raise ValueError(f"Import path must include an attribute: {import_path!r}")
        module_path = ".".join(parts[:-1])
        attr_path = parts[-1]
    module = importlib.import_module(module_path)
    value: Any = module
    for part in attr_path.split("."):
        value = getattr(value, part)
    return value


def _canonical_identifier(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _canonical_resource_kind(value: Any) -> str:
    raw = _canonical_identifier(value)
    return _RESOURCE_ALIASES.get(raw, raw)


@dataclass(frozen=True, kw_only=True)
class TreePOResourceSpec:
    """Neutral resource descriptor for contract adapters."""

    kind: str = "object"
    value: Any = None
    import_path: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ResolvedTreePOResources:
    resources: Dict[str, Any] = field(default_factory=dict)
    resource_specs: Dict[str, Any] = field(default_factory=dict)

    def get(self, kind: str, default: Any = None) -> Any:
        return self.resources.get(_canonical_resource_kind(kind), default)

    def require(self, kind: str, *, contract_id: str, adapter_key: str) -> Any:
        canonical = _canonical_resource_kind(kind)
        if canonical not in self.resources or self.resources[canonical] is None:
            raise ValueError(
                f"Contract {contract_id!r} via adapter {adapter_key!r} requires "
                f"resource {canonical!r}."
            )
        return self.resources[canonical]

    @property
    def kinds(self) -> Tuple[str, ...]:
        return tuple(sorted(self.resources))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_kinds": list(self.kinds),
            "resource_specs": config_to_dict(self.resource_specs),
        }


@dataclass(frozen=True, kw_only=True)
class ResolvedTreePOContractRoute:
    contract_id: str
    adapter_key: str
    adapter_class: str
    resolved_model_class: str
    resolved_supervision_source: str
    capabilities: Dict[str, Any] = field(default_factory=dict)
    matched_requirements: Dict[str, Any] = field(default_factory=dict)
    resource_kinds: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return config_to_dict(self)


@dataclass(frozen=True, kw_only=True)
class TreePOContractFitResult:
    contract: Dict[str, Any]
    resolved_model_class: str
    resolved_supervision_source: str
    capabilities: Dict[str, Any]
    artifacts: Dict[str, Any]
    metrics: Dict[str, Any]
    route: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return config_to_dict(self)


@dataclass(frozen=True, kw_only=True)
class TreePOContractFitContext:
    contract: TreePOContractSpec
    route: ResolvedTreePOContractRoute
    model: TreePOModelSpec
    run: RunConfig
    train: TrainConfig
    validation: ValidationConfig
    test: TestConfig
    data: Dict[str, Any]
    supervision: Dict[str, Any]
    resources: ResolvedTreePOResources
    output_dir: Path


class TreePOContractAdapter:
    """Base class for registry adapters."""

    adapter_key = "base"
    resolved_model_class = "treepo_model"
    resolved_supervision_source = "unspecified"
    capabilities: Dict[str, Any] = {}

    def supports(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> bool:
        raise NotImplementedError

    def matched_requirements(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> Dict[str, Any]:
        return {
            "objective_kind": contract.objective_kind,
            "state_semantics": contract.state_semantics,
        }

    def resource_kinds(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> Tuple[str, ...]:
        return resources.kinds

    def resolve(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> ResolvedTreePOContractRoute:
        return ResolvedTreePOContractRoute(
            contract_id=_contract_id(contract),
            adapter_key=self.adapter_key,
            adapter_class=type(self).__name__,
            resolved_model_class=self.resolved_model_class,
            resolved_supervision_source=self.resolved_supervision_source,
            capabilities=dict(self.capabilities),
            matched_requirements=self.matched_requirements(contract, model, data, resources),
            resource_kinds=self.resource_kinds(contract, model, data, resources),
        )

    def fit(self, context: TreePOContractFitContext) -> TreePOContractFitResult:
        raise NotImplementedError


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(payload), indent=2, sort_keys=True) + "\n")
    return path


def _contract_dict(contract: TreePOContractSpec) -> Dict[str, Any]:
    payload = config_to_dict(contract)
    payload.pop("oracle_lane", None)
    payload.pop("supervision_source", None)
    return dict(payload)


def _contract_id(contract: TreePOContractSpec) -> str:
    value = str(getattr(contract, "contract_id", "") or "").strip()
    return value or "treepo_contract"


def _contract_adapter_preference(contract: TreePOContractSpec) -> Optional[str]:
    direct = getattr(contract, "adapter_preference", None)
    if direct:
        return _canonical_identifier(direct)
    metadata = dict(getattr(contract, "metadata", {}) or {})
    value = metadata.get("adapter_preference")
    if value:
        return _canonical_identifier(value)
    return None


def _resolve_one_resource(value: Any) -> tuple[Any, Any]:
    if isinstance(value, TreePOResourceSpec):
        spec = value
        if _canonical_identifier(spec.kind) in {"object", "provided"}:
            return spec.value, {"kind": "object", "type": type(spec.value).__name__}
        if _canonical_identifier(spec.kind) in {"import", "import_path"}:
            if not spec.import_path:
                raise ValueError("TreePOResourceSpec(kind='import_path') requires import_path")
            target = _import_from_path(spec.import_path)
            resolved = target(**dict(spec.kwargs or {})) if callable(target) else target
            return resolved, {"kind": "import_path", "import_path": spec.import_path}
        raise ValueError(f"Unsupported TreePOResourceSpec.kind={spec.kind!r}")

    if isinstance(value, Mapping) and "kind" in value:
        kind = _canonical_identifier(value.get("kind"))
        if kind in {"object", "provided"}:
            resolved = value.get("value")
            return resolved, {"kind": "object", "type": type(resolved).__name__}
        if kind in {"import", "import_path"}:
            import_path = value.get("import_path")
            if not import_path:
                raise ValueError("Resource mapping kind='import_path' requires import_path")
            kwargs = dict(value.get("kwargs") or {})
            target = _import_from_path(str(import_path))
            resolved = target(**kwargs) if callable(target) else target
            return resolved, {"kind": "import_path", "import_path": str(import_path)}
        raise ValueError(f"Unsupported resource spec kind={value.get('kind')!r}")

    return value, {"kind": "object", "type": type(value).__name__}


def resolve_treepo_resources(resources: Optional[Mapping[str, Any]] = None) -> ResolvedTreePOResources:
    """Resolve neutral resource specs into concrete adapter resources."""

    resolved: Dict[str, Any] = {}
    specs: Dict[str, Any] = {}
    for key, value in dict(resources or {}).items():
        canonical = _canonical_resource_kind(key)
        concrete, spec_meta = _resolve_one_resource(value)
        resolved[canonical] = concrete
        specs[canonical] = spec_meta
    return ResolvedTreePOResources(resources=resolved, resource_specs=specs)


def _base_result(
    *,
    context: TreePOContractFitContext,
    artifacts: Mapping[str, Any],
    metrics: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
) -> TreePOContractFitResult:
    return TreePOContractFitResult(
        contract=_contract_dict(context.contract),
        resolved_model_class=context.route.resolved_model_class,
        resolved_supervision_source=context.route.resolved_supervision_source,
        capabilities=dict(context.route.capabilities),
        artifacts=dict(artifacts),
        metrics=dict(metrics),
        route=context.route.to_dict(),
        metadata=dict(metadata or {}),
    )


class TextGenerationDistillationAdapter(TreePOContractAdapter):
    adapter_key = "text_generation_distillation"
    resolved_model_class = "generative_tree_operator"
    resolved_supervision_source = "labeled_tree_artifact"
    capabilities = {
        "tree_indexed_outputs": True,
        "summary_targets": True,
        "score_targets": True,
        "uses_distillation_fit": True,
    }

    def supports(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> bool:
        objective = _canonical_identifier(contract.objective_kind)
        semantics = _canonical_identifier(contract.state_semantics)
        return objective in {"node_summary_distillation", "text_node_summary"} or semantics in {
            "natural_language_summary",
            "text_summary_state",
        }

    def matched_requirements(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> Dict[str, Any]:
        return {
            "objective_kind": contract.objective_kind,
            "state_semantics": contract.state_semantics,
            "required_resources": [RESOURCE_GENERATION, RESOURCE_EMBEDDING],
            "operator_requirements": dict(contract.operator_requirements or {}),
            "oracle_requirements": dict(contract.oracle_requirements or {}),
        }

    def resource_kinds(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> Tuple[str, ...]:
        return tuple(sorted(set(resources.kinds).union({RESOURCE_GENERATION, RESOURCE_EMBEDDING})))

    def fit(self, context: TreePOContractFitContext) -> TreePOContractFitResult:
        from treepo._research.ctreepo.distillation import (
            DistillationContractConfig,
            DistillationTrainConfig,
            FEmbeddingConfig,
            ScoreTargetConfig,
            SummaryTargetConfig,
            TRAIN_TARGET_F,
            TRAIN_TARGET_G,
            STUDENT_MODEL_EMBEDDING_RIDGE_PROXY,
            STUDENT_MODEL_LM_SFT,
            build_f_embedding_examples,
            build_g_sft_records,
            build_labeled_tree_from_text,
            fit,
            write_labeled_trees_jsonl,
        )

        output_dir = context.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        leaves = [str(item) for item in context.data.get("leaf_spans", ())]
        if not leaves:
            raise ValueError("text contract requires data['leaf_spans']")

        rows = list(context.supervision.get("rows") or ())
        if not rows:
            rows = [
                {"text": leaf, "label": float(index) / max(1, len(leaves) - 1)}
                for index, leaf in enumerate(leaves)
            ]

        supervision_csv = output_dir / "proxy_supervision.csv"
        with open(supervision_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["text", "label"])
            writer.writeheader()
            for row in rows:
                writer.writerow({"text": str(row["text"]), "label": float(row["label"])})

        generation = context.resources.require(
            RESOURCE_GENERATION,
            contract_id=context.route.contract_id,
            adapter_key=context.route.adapter_key,
        )
        embedding = context.resources.require(
            RESOURCE_EMBEDDING,
            contract_id=context.route.contract_id,
            adapter_key=context.route.adapter_key,
        )
        local_law_oracle = context.resources.get(RESOURCE_LOCAL_LAW_ORACLE)

        proxy_path = output_dir / "treepo_stack_embedding_proxy.json"
        stack_model = replace(context.model, kind="diffusion_backend", backend=generation)
        stack_contract = replace(
            context.contract,
            supervision_source=SupervisionSourceSpec(
                kind="csv",
                path=str(supervision_csv),
                text_column="text",
                label_column="label",
                rubric=str(context.contract.rubric or ""),
                save_path=str(output_dir / "proxy_supervision_dataset.json"),
            ),
            oracle_lane=OracleLaneSpec(
                kind="embedding_proxy",
                embedding_client=embedding,
                ridge_lambda=float(context.supervision.get("ridge_lambda", 1e-8)),
                proxy_model_id=str(context.supervision.get("proxy_model_id", "paper_text_proxy")),
                proxy_artifact_path=str(proxy_path),
                value_name=str(context.supervision.get("value_name", "paper_text_score")),
            ),
        )
        stack = build_treepo_stack(stack_model, stack_contract)
        stack_result = stack.run_fixed_binary(
            leaves,
            document_id=str(context.data.get("document_id", context.route.contract_id)),
            refine_rounds=int(context.data.get("refine_rounds", 0)),
            sampling_params=dict(context.data.get("sampling_params") or {"max_tokens": 96, "temperature": 0.0}),
        )
        stack_path = _write_json(output_dir / "treepo_state_tree.json", stack_result.to_dict())

        def score_span(text: str) -> float:
            if callable(local_law_oracle):
                return float(local_law_oracle(text))
            return float(min(100.0, max(0.0, len(str(text)) / 2.0)))

        labeled_tree = build_labeled_tree_from_text(
            doc_id=str(context.data.get("document_id", context.route.contract_id)),
            text=str(context.data.get("document_text") or "\n\n".join(leaves)),
            document_score=float(context.data.get("document_score", 0.0)),
            split=str(context.data.get("split", "train")),
            score_fn=score_span,
            window_size=int(context.data.get("window_size", 48)),
            window_overlap=int(context.data.get("window_overlap", 0)),
            target_leaves_per_doc=context.data.get("target_leaves_per_doc"),
            label_source=str(context.data.get("label_source", "paper_demo_teacher")),
            root_summary=context.data.get("root_summary"),
            resummary_target=context.data.get("resummary_target"),
            fill_missing_summaries_from_span=bool(context.data.get("fill_missing_summaries_from_span", True)),
        )
        labeled_tree_path = write_labeled_trees_jsonl(output_dir / "labeled_trees.jsonl", [labeled_tree])

        teacher_model_spec = dict(context.data.get("teacher_model_spec") or {"kind": "paper_demo_teacher"})
        g_result = fit(
            [labeled_tree],
            DistillationTrainConfig(
                contract=DistillationContractConfig(
                    train_targets=(TRAIN_TARGET_G,),
                    student_model_class=STUDENT_MODEL_LM_SFT,
                    supervision_source="labeled_tree_artifact",
                    teacher_model_spec=teacher_model_spec,
                ),
                run=replace(context.run, output_dir=output_dir / "g_student"),
                train=context.train,
                validation=context.validation,
                test=context.test,
                summary_targets=SummaryTargetConfig(include_identity_targets=False),
            ),
        )
        f_result = fit(
            [labeled_tree],
            DistillationTrainConfig(
                contract=DistillationContractConfig(
                    train_targets=(TRAIN_TARGET_F,),
                    student_model_class=STUDENT_MODEL_EMBEDDING_RIDGE_PROXY,
                    supervision_source="labeled_tree_artifact",
                    teacher_model_spec=teacher_model_spec,
                ),
                run=replace(context.run, output_dir=output_dir / "f_student"),
                train=context.train,
                validation=context.validation,
                test=context.test,
                score_targets=ScoreTargetConfig(
                    target_min=float(context.data.get("target_min", -100.0)),
                    target_max=float(context.data.get("target_max", 100.0)),
                ),
                f_embedding=FEmbeddingConfig(ridge_lambda=float(context.supervision.get("ridge_lambda", 1e-8))),
            ),
            embedding_client=embedding,
        )

        summary_path = output_dir / "summary.json"
        result = _base_result(
            context=context,
            artifacts={
                "summary": str(summary_path),
                "state_tree": str(stack_path),
                "labeled_trees": str(labeled_tree_path),
                "g_sft_train": str(output_dir / "g_student" / "g_sft_train.jsonl"),
                "f_embedding_proxy": str(output_dir / "f_student" / "f_embedding_proxy.json"),
            },
            metrics={
                "leaf_count": len(leaves),
                "g_sft_record_count": len(build_g_sft_records([labeled_tree])),
                "f_embedding_example_count": len(build_f_embedding_examples([labeled_tree])),
                "root_rendered": stack_result.tree.final_rendered,
            },
            metadata={
                "fit_backends": ["tree_stack", "distillation_g", "distillation_f"],
                "resources": context.resources.to_dict(),
                "g_result": g_result.metadata,
                "f_result": f_result.metadata,
            },
        )
        _write_json(summary_path, result.to_dict())
        return result


class SymbolicReferenceAdapter(TreePOContractAdapter):
    adapter_key = "symbolic_reference"
    resolved_model_class = "exact_symbolic_operator"
    resolved_supervision_source = "theorem_backed_reference"
    capabilities = {
        "tree_indexed_outputs": True,
        "exact_reference_available": True,
        "theorem_backed": True,
    }

    def supports(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> bool:
        objective = _canonical_identifier(contract.objective_kind)
        return objective in {"symbolic_state_reduction", "exact_state_reduction"} or bool(contract.theorem_domain)

    def matched_requirements(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> Dict[str, Any]:
        return {
            "objective_kind": contract.objective_kind,
            "state_semantics": contract.state_semantics,
            "theorem_domain": dict(contract.theorem_domain or {}),
            "operator_requirements": dict(contract.operator_requirements or {}),
        }

    def resource_kinds(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> Tuple[str, ...]:
        return tuple(sorted(set(resources.kinds).union({RESOURCE_SYMBOLIC_REFERENCE})))

    def fit(self, context: TreePOContractFitContext) -> TreePOContractFitResult:
        from treepo._research.diffusion.markov_toy import encode_markov_path

        output_dir = context.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        sequence = [str(item) for item in context.data.get("sequence", ())]
        if not sequence:
            raise ValueError("symbolic contract requires data['sequence']")
        leaf_size = int(context.data.get("leaf_size", 2))
        leaf_spans = [sequence[index : index + leaf_size] for index in range(0, len(sequence), leaf_size)]

        stack = build_treepo_stack(
            replace(context.model, kind="markov_toy_exact"),
            replace(context.contract, oracle_lane=OracleLaneSpec(kind="markov_exact")),
        )
        result = stack.run_fixed_binary(
            leaf_spans,
            document_id=str(context.data.get("document_id", context.route.contract_id)),
        )
        reference = encode_markov_path(sequence)
        state_path = _write_json(output_dir / "treepo_state_tree.json", result.to_dict())
        summary_path = output_dir / "summary.json"
        fit_result = _base_result(
            context=context,
            artifacts={"summary": str(summary_path), "state_tree": str(state_path)},
            metrics={
                "leaf_count": len(leaf_spans),
                "root_matches_reference": bool(result.tree.root.state == reference),
                "reference_state": {
                    "changepoints": reference.changepoints,
                    "start_state": reference.start_state,
                    "end_state": reference.end_state,
                    "length": reference.length,
                },
            },
            metadata={
                "fit_backends": ["tree_stack"],
                "resources": context.resources.to_dict(),
                "train": config_to_dict(context.train),
                "validation": config_to_dict(context.validation),
                "test": config_to_dict(context.test),
                "run": config_to_dict(context.run),
            },
        )
        _write_json(summary_path, fit_result.to_dict())
        return fit_result


class LearnedStateSummaryAdapter(TreePOContractAdapter):
    adapter_key = "learned_tree_state"
    resolved_model_class = "learned_tree_operator"
    resolved_supervision_source = "local_law_oracle_queries"
    capabilities = {
        "tree_indexed_outputs": True,
        "decoded_summary_metrics": True,
        "local_law_metrics": True,
    }

    def supports(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> bool:
        objective = _canonical_identifier(contract.objective_kind)
        semantics = _canonical_identifier(contract.state_semantics)
        return objective in {"local_law_recovery", "learned_state_recovery"} or semantics in {
            "learned_state_summary",
            "compressed_state_summary",
        }

    def matched_requirements(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> Dict[str, Any]:
        return {
            "objective_kind": contract.objective_kind,
            "state_semantics": contract.state_semantics,
            "operator_requirements": dict(contract.operator_requirements or {}),
            "oracle_requirements": dict(contract.oracle_requirements or {}),
        }

    def resource_kinds(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> Tuple[str, ...]:
        return tuple(sorted(set(resources.kinds).union({RESOURCE_LOCAL_LAW_ORACLE})))

    def fit(self, context: TreePOContractFitContext) -> TreePOContractFitResult:
        from treepo._research.training.config_sections import OptimizerConfig
        from treepo._research.tree.learned_sketch import (
            LearnedSketchDataConfig,
            LearnedSketchEvaluationConfig,
            LearnedSketchModelConfig,
            LearnedSketchObjectiveConfig,
            LearnedSketchTrainingConfig,
            run_learning_curve,
        )
        from treepo._research.tree.mergeable_ablation import (
            run_default_ablation_suite,
            sketch_insufficiency_counterexample,
            worked_failure_examples,
        )

        output_dir = context.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        seed = int(context.run.seed)
        n_docs = int(context.data.get("n_docs", 48))
        n_steps = int(context.train.steps if context.train.steps is not None else context.data.get("steps", 12))
        state_dim = int(context.data.get("state_dim", 4))
        target_k = int(context.data.get("target_k", 4))
        ablations = run_default_ablation_suite(
            n_docs=n_docs,
            n_tokens=int(context.data.get("n_tokens", 32)),
            seed=seed,
        )
        counter_a, counter_b, shared_topm = sketch_insufficiency_counterexample(
            sketch_order=int(context.data.get("sketch_order", 3)),
            target_k=target_k,
            n_tokens=int(context.data.get("counterexample_tokens", 16)),
        )
        learned_config = LearnedSketchTrainingConfig(
            model=LearnedSketchModelConfig(
                state_dim=state_dim,
                target_k=target_k,
                hidden_dim=int(context.data.get("hidden_dim", 16)),
            ),
            data=LearnedSketchDataConfig(chunk_size=int(context.data.get("chunk_size", 4))),
            train=replace(context.train, steps=n_steps),
            optimizer=OptimizerConfig(learning_rate=float(context.data.get("learning_rate", 1e-3))),
            validation=context.validation,
            run=context.run,
            objective=LearnedSketchObjectiveConfig(n_audit=int(context.data.get("n_audit", 4))),
            evaluation=LearnedSketchEvaluationConfig(eval_docs=int(context.data.get("eval_docs", 16))),
        )
        learned = run_learning_curve(learned_config)
        summary_path = output_dir / "summary.json"
        learned_payload = asdict(learned)
        fit_result = _base_result(
            context=context,
            artifacts={"summary": str(summary_path)},
            metrics={
                "n_docs": n_docs,
                "training_steps": n_steps,
                "ablation_summaries": [asdict(item) for item in ablations],
                "worked_failure_examples": worked_failure_examples(),
                "insufficiency_counterexample": {
                    "sketch_order": int(context.data.get("sketch_order", 3)),
                    "target_k": target_k,
                    "doc_a_scores": list(counter_a),
                    "doc_b_scores": list(counter_b),
                    "shared_topm_signature": list(shared_topm),
                },
                "local_law_series": learned_payload.get("metrics", []),
                "final_l1_leaf_error": learned_payload.get("final_l1_leaf_error"),
                "final_l2_merge_error": learned_payload.get("final_l2_merge_error"),
                "final_l3_idemp_error": learned_payload.get("final_l3_idemp_error"),
            },
            metadata={
                "fit_backends": ["learned_tree_state"],
                "resources": context.resources.to_dict(),
                "training_result": learned_payload,
                "test": config_to_dict(context.test),
            },
        )
        _write_json(summary_path, fit_result.to_dict())
        return fit_result


class LabeledTreeDistillationAdapter(TreePOContractAdapter):
    adapter_key = "labeled_tree_distillation"
    resolved_model_class = "artifact_distillation"
    resolved_supervision_source = "labeled_tree_artifact"
    capabilities = {
        "tree_indexed_outputs": True,
        "uses_distillation_fit": True,
    }

    def supports(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> bool:
        objective = _canonical_identifier(contract.objective_kind)
        return objective in {"labeled_tree_distillation", "artifact_distillation"}

    def matched_requirements(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> Dict[str, Any]:
        return {
            "objective_kind": contract.objective_kind,
            "state_semantics": contract.state_semantics,
            "required_data": ["labeled_trees"],
        }

    def resource_kinds(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> Tuple[str, ...]:
        return tuple(sorted(set(resources.kinds).union({RESOURCE_LABELED_TREE_ARTIFACT})))

    def fit(self, context: TreePOContractFitContext) -> TreePOContractFitResult:
        from treepo._research.ctreepo.distillation import DistillationTrainConfig, fit

        output_dir = context.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        labeled_trees = list(context.data.get("labeled_trees") or ())
        if not labeled_trees:
            labeled_trees = list(context.resources.get(RESOURCE_LABELED_TREE_ARTIFACT) or ())
        if not labeled_trees:
            raise ValueError("labeled-tree distillation contract requires labeled trees")
        config = context.data.get("distillation_config")
        if config is None:
            config = DistillationTrainConfig(
                run=replace(context.run, output_dir=output_dir),
                train=context.train,
                validation=context.validation,
                test=context.test,
            )
        result = fit(
            labeled_trees,
            config,
            embedding_client=context.resources.get(RESOURCE_EMBEDDING),
            trainer=context.resources.get(RESOURCE_TRAINER),
        )
        summary_path = output_dir / "summary.json"
        fit_result = _base_result(
            context=context,
            artifacts={"summary": str(summary_path)},
            metrics={
                "train_count": int(result.train_count),
                "val_count": int(result.val_count),
                "test_count": int(result.test_count),
            },
            metadata={
                "fit_backends": ["distillation"],
                "resources": context.resources.to_dict(),
                "distillation_result": result.metadata,
            },
        )
        _write_json(summary_path, fit_result.to_dict())
        return fit_result


class EmbeddingFNONodeDistillationAdapter(TreePOContractAdapter):
    adapter_key = "embedding_fno_node_distillation"
    resolved_model_class = "embedding_coordinate_fno_tree_operator"
    resolved_supervision_source = "labeled_tree_artifact"
    capabilities = {
        "tree_indexed_outputs": True,
        "uses_embedding_resource": True,
        "fno_spatial_axis": "embedding_dimension",
        "compares_node_to_node": True,
    }

    def supports(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> bool:
        objective = _canonical_identifier(contract.objective_kind)
        model_kind = _canonical_identifier(getattr(model, "kind", ""))
        model_name = _canonical_identifier(getattr(model, "model", ""))
        return objective in {
            "embedding_fno_node_distillation",
            "embedding_coordinate_fno_node_distillation",
        } or model_kind in {
            "embedding_fno_tree_operator",
            "embedding_coordinate_fno_tree_operator",
        } or model_name in {
            "embedding_fno_tree_operator",
            "embedding_coordinate_fno_tree_operator",
        }

    def matched_requirements(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> Dict[str, Any]:
        return {
            "objective_kind": contract.objective_kind,
            "state_semantics": contract.state_semantics,
            "required_data": ["labeled_trees"],
            "required_resources": [RESOURCE_EMBEDDING],
            "fno_spatial_axis": "embedding_dimension",
        }

    def resource_kinds(
        self,
        contract: TreePOContractSpec,
        model: TreePOModelSpec,
        data: Mapping[str, Any],
        resources: ResolvedTreePOResources,
    ) -> Tuple[str, ...]:
        return tuple(sorted(set(resources.kinds).union({RESOURCE_EMBEDDING, RESOURCE_LABELED_TREE_ARTIFACT})))

    def fit(self, context: TreePOContractFitContext) -> TreePOContractFitResult:
        from treepo._research.ctreepo.embedding_fno import EmbeddingFNOTrainConfig, fit_embedding_fno_node_regressor

        output_dir = context.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        labeled_trees = list(context.data.get("labeled_trees") or ())
        if not labeled_trees:
            labeled_trees = list(context.resources.get(RESOURCE_LABELED_TREE_ARTIFACT) or ())
        if not labeled_trees:
            raise ValueError("embedding-FNO node distillation requires labeled trees")
        embedding_client = context.resources.require(
            RESOURCE_EMBEDDING,
            contract_id=_contract_id(context.contract),
            adapter_key=self.adapter_key,
        )
        config = context.data.get("embedding_fno_config")
        if config is None:
            config = EmbeddingFNOTrainConfig(
                run=replace(context.run, output_dir=output_dir),
                train=context.train,
                validation=context.validation,
                test=context.test,
            )
        result = fit_embedding_fno_node_regressor(
            labeled_trees,
            embedding_client=embedding_client,
            config=config,
        )
        summary_path = output_dir / "summary.json"
        fit_result = _base_result(
            context=context,
            artifacts={**dict(result.artifacts), "summary": str(summary_path)},
            metrics={
                "train_count": int(result.train_count),
                "val_count": int(result.val_count),
                "test_count": int(result.test_count),
                "embedding_dim": int(result.embedding_dim),
                "fit_metrics": dict(result.metrics),
            },
            metadata={
                "fit_backends": ["embedding_coordinate_fno"],
                "resources": context.resources.to_dict(),
                "embedding_fno_result": result.to_dict(),
            },
        )
        _write_json(summary_path, fit_result.to_dict())
        return fit_result


DEFAULT_TREEPO_CONTRACT_ADAPTERS: Tuple[TreePOContractAdapter, ...] = (
    TextGenerationDistillationAdapter(),
    SymbolicReferenceAdapter(),
    LearnedStateSummaryAdapter(),
    EmbeddingFNONodeDistillationAdapter(),
    LabeledTreeDistillationAdapter(),
)


def _matches_preference(adapter: TreePOContractAdapter, preference: str) -> bool:
    pref = _canonical_identifier(preference)
    return pref in {
        _canonical_identifier(adapter.adapter_key),
        _canonical_identifier(type(adapter).__name__),
        _canonical_identifier(adapter.resolved_model_class),
    }


def resolve_treepo_contract_adapter(
    contract: TreePOContractSpec,
    model: Optional[TreePOModelSpec] = None,
    *,
    data: Optional[Mapping[str, Any]] = None,
    resources: Optional[Mapping[str, Any] | ResolvedTreePOResources] = None,
    adapters: Optional[Sequence[TreePOContractAdapter]] = None,
) -> tuple[TreePOContractAdapter, ResolvedTreePOContractRoute, ResolvedTreePOResources]:
    """Resolve exactly one adapter for a contract."""

    model_spec = model or TreePOModelSpec()
    payload = dict(data or {})
    resolved_resources = (
        resources
        if isinstance(resources, ResolvedTreePOResources)
        else resolve_treepo_resources(resources)
    )
    candidates = tuple(adapters or DEFAULT_TREEPO_CONTRACT_ADAPTERS)
    matches = [
        adapter
        for adapter in candidates
        if adapter.supports(contract, model_spec, payload, resolved_resources)
    ]

    preference = _contract_adapter_preference(contract)
    if preference:
        matches = [adapter for adapter in matches if _matches_preference(adapter, preference)]

    if not matches:
        detail = f" with adapter_preference={preference!r}" if preference else ""
        raise ValueError(
            f"No TreePO contract adapter matched contract {_contract_id(contract)!r}{detail}."
        )
    if len(matches) > 1:
        keys = [adapter.adapter_key for adapter in matches]
        raise ValueError(
            f"Multiple TreePO contract adapters matched contract {_contract_id(contract)!r}: {keys}. "
            "Set contract.adapter_preference or contract.metadata['adapter_preference']."
        )

    adapter = matches[0]
    route = adapter.resolve(contract, model_spec, payload, resolved_resources)
    return adapter, route, resolved_resources


def resolve_treepo_contract_route(
    contract: TreePOContractSpec,
    model: Optional[TreePOModelSpec] = None,
    *,
    data: Optional[Mapping[str, Any]] = None,
    resources: Optional[Mapping[str, Any] | ResolvedTreePOResources] = None,
    adapters: Optional[Sequence[TreePOContractAdapter]] = None,
) -> ResolvedTreePOContractRoute:
    """Resolve a contract route without running the adapter."""

    _, route, _ = resolve_treepo_contract_adapter(
        contract,
        model,
        data=data,
        resources=resources,
        adapters=adapters,
    )
    return route


def fit_treepo_contract(
    *,
    contract: TreePOContractSpec,
    model: Optional[TreePOModelSpec] = None,
    run: Optional[RunConfig] = None,
    train: Optional[TrainConfig] = None,
    validation: Optional[ValidationConfig] = None,
    test: Optional[TestConfig] = None,
    data: Optional[Mapping[str, Any]] = None,
    supervision: Optional[Mapping[str, Any]] = None,
    output_dir: Optional[Path] = None,
    resources: Optional[Mapping[str, Any]] = None,
    adapters: Optional[Sequence[TreePOContractAdapter]] = None,
) -> TreePOContractFitResult:
    """Fit or evaluate one TreePO contract through the adapter registry."""

    run_cfg = run or RunConfig()
    train_cfg = train or TrainConfig()
    validation_cfg = validation or ValidationConfig()
    test_cfg = test or TestConfig()
    model_spec = model or TreePOModelSpec()
    payload = dict(data or {})
    adapter, route, resolved_resources = resolve_treepo_contract_adapter(
        contract,
        model_spec,
        data=payload,
        resources=resources,
        adapters=adapters,
    )
    base_output = Path(output_dir or run_cfg.output_dir or Path("outputs") / "treepo_contracts")
    context = TreePOContractFitContext(
        contract=contract,
        route=route,
        model=model_spec,
        run=run_cfg,
        train=train_cfg,
        validation=validation_cfg,
        test=test_cfg,
        data=payload,
        supervision=dict(supervision or {}),
        resources=resolved_resources,
        output_dir=base_output / route.contract_id,
    )
    return adapter.fit(context)


def find_contract_setup_bypasses(
    paths: Iterable[Path | str],
    *,
    forbidden_tokens: Sequence[str] = (
        "markov_exact",
        "markov_toy_exact",
        "learned_sketch",
        "mergeable_sketch",
        "diffusion_backend",
        "generation_backend",
        "llm",
    ),
) -> Dict[str, list[str]]:
    """Return forbidden implementation-route tokens found in public setup files."""

    findings: Dict[str, list[str]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        text = path.read_text()
        tokens = [token for token in forbidden_tokens if token in text]
        if tokens:
            findings[str(path)] = tokens
    return findings


__all__ = [
    "DEFAULT_TREEPO_CONTRACT_ADAPTERS",
    "RESOURCE_EMBEDDING",
    "RESOURCE_GENERATION",
    "RESOURCE_LABELED_TREE_ARTIFACT",
    "RESOURCE_LOCAL_LAW_ORACLE",
    "RESOURCE_SYMBOLIC_REFERENCE",
    "RESOURCE_TRAINER",
    "ResolvedTreePOContractRoute",
    "ResolvedTreePOResources",
    "TextGenerationDistillationAdapter",
    "TreePOContractAdapter",
    "TreePOContractFitContext",
    "TreePOContractFitResult",
    "TreePOResourceSpec",
    "fit_treepo_contract",
    "find_contract_setup_bypasses",
    "resolve_treepo_contract_adapter",
    "resolve_treepo_contract_route",
    "resolve_treepo_resources",
]
