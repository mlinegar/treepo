"""Generate-first TreePO stack builder.

This module provides a single entrypoint for assembling a unified TreePO stack:
- an operator ``g`` (summarize/merge/resummary) over either `/generate` or chat,
- an oracle lane (provided oracle or trained proxy),
- local-law verifiers wired into the canonical ``StateTree`` runner.

Key principle: "generate-first" is the default. Users specify "generate" and we
fall back to chat only when the chosen engine does not expose a `/generate`
surface.
"""

from __future__ import annotations

import csv
import importlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

from treepo._research.core.engines import EngineSurface, EngineType, resolve_engine_base_url
from treepo._research.core.inference_engine import build_inference_engine
from treepo._research.core.ops_checks import EvidenceStatus
from treepo._research.core.scoring import BoundedScale, ScoringOracle, SimilarityScorer
from treepo._research.core.supervision_metadata import judgment_supervision_metadata
from treepo._research.core.url_utils import normalize_generate_base_url
from treepo._research.training.embedding_proxy import LabeledEmbeddingExample, fit_embedding_ridge_proxy
from treepo._research.training.supervision.types import ResponseJudgment, SupervisionDataset
from treepo._research.tree.async_operator import (
    AsyncCompositionalOperator,
    AsyncFromDiffusionBackend,
    AsyncFromInferenceEngine,
    MarkovToyOperator,
)
from treepo._research.tree.generate_prompting import GenerateTreePromptTemplates
from treepo._research.tree.state_tree_runner import FixedBinaryStateTreeRunResult, run_fixed_binary_state_tree
from treepo._research.tree.state_tree_verifiers import LawVerifier, MarkovExactVerifier, TextAuditorAdapterVerifier


def _import_from_path(import_path: str) -> Any:
    text = str(import_path or "").strip()
    if not text:
        raise ValueError("Empty import_path")
    module_path: str
    attr_path: str
    if ":" in text:
        module_path, attr_path = text.split(":", 1)
    else:
        parts = text.split(".")
        if len(parts) < 2:
            raise ValueError(
                "Import path must be 'module:attr' or 'module.attr', "
                f"received {import_path!r}."
            )
        module_path, attr_path = ".".join(parts[:-1]), parts[-1]
    module = importlib.import_module(module_path)
    value: Any = module
    for part in attr_path.split("."):
        value = getattr(value, part)
    return value


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text


@dataclass(frozen=True)
class TreePOModelSpec:
    """Spec for the TreePO operator g (summarize/merge/resummary)."""

    kind: str = "inference_engine"  # "inference_engine" | "markov_toy_exact"
    engine: str | EngineType = "auto"
    model: str = "default"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    host: str = "localhost"
    port: Optional[int] = None
    timeout: float = 120.0

    surface: str = "generate"  # user-facing: "generate" (preferred) or "chat"
    prefer_generate: bool = True
    generate_path: str = "/generate"

    max_tokens: int = 512
    temperature: float = 0.0
    stop: Tuple[str, ...] = ()

    prompt_templates: Optional[GenerateTreePromptTemplates] = None
    backend: Optional[Any] = None  # For kind="diffusion_backend" tests/adapters.


@dataclass(frozen=True)
class TreePOLocalLawConfig:
    """Contract-side local-law config used to construct the legacy Auditor config."""

    enable_l1: bool = True
    enable_l2: bool = True
    enable_l3: bool = True

    discrepancy_threshold: float = 0.1
    sample_budget: int = 10
    idempotence_budget: int = 5
    sampling_probability: float = 1.0

    enable_substitution: bool = False
    substitution_budget: int = 5

    sampling_strategy: str = "random"
    random_seed: Optional[int] = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TreePOLocalLawConfig":
        data = dict(payload or {})
        if "sample_prob" in data and "sampling_probability" not in data:
            data["sampling_probability"] = data.pop("sample_prob")
        if "sampling_prob" in data and "sampling_probability" not in data:
            data["sampling_probability"] = data.pop("sampling_prob")
        if "sample_probability" in data and "sampling_probability" not in data:
            data["sampling_probability"] = data.pop("sample_probability")
        return cls(**data)

    def to_audit_config(self) -> Any:
        from treepo._research.tree.auditor import AuditConfig, SamplingStrategy

        strategy_raw = str(self.sampling_strategy or "random").strip().lower()
        try:
            strategy = SamplingStrategy(strategy_raw)
        except Exception:
            strategy = SamplingStrategy.RANDOM

        return AuditConfig(
            sample_budget=int(self.sample_budget),
            sampling_strategy=strategy,
            sampling_probability=float(self.sampling_probability),
            discrepancy_threshold=float(self.discrepancy_threshold),
            audit_leaves=bool(self.enable_l1),
            audit_internal=bool(self.enable_l2),
            audit_idempotence=bool(self.enable_l3),
            audit_substitution=bool(self.enable_substitution),
            idempotence_budget=int(self.idempotence_budget),
            substitution_budget=int(self.substitution_budget),
            random_seed=self.random_seed,
        )


@dataclass(frozen=True)
class SupervisionSourceSpec:
    """How to construct/load supervision for training proxy oracles."""

    kind: str  # "csv" | "jsonl" | "supervision_dataset_json"
    path: str

    text_column: str = "text"
    label_column: str = "label"
    example_id_column: Optional[str] = None
    doc_id_column: Optional[str] = None
    rubric_column: Optional[str] = None
    split_column: Optional[str] = None

    rubric: Optional[str] = None
    law_type: str = "document_level_target"
    response_signal_name: str = "document_score"
    response_signal_min: float = 0.0
    response_signal_max: float = 1.0

    save_path: Optional[str] = None
    max_rows: Optional[int] = None


@dataclass(frozen=True)
class OracleLaneSpec:
    kind: str  # "provided_scoring_oracle" | "embedding_proxy" | "markov_exact"

    # provided_scoring_oracle
    import_path: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    # embedding_proxy
    embedding_client: Optional[Any] = None
    embedding_engine: str | EngineType = EngineType.VLLM
    embedding_base_url: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_api_key: Optional[str] = None
    ridge_lambda: float = 1.0
    proxy_model_id: str = "embedding_proxy_v1"
    proxy_artifact_path: Optional[str] = None
    value_name: str = "value"


@dataclass(frozen=True)
class TreePOContractSpec:
    rubric: str
    local_law_config: TreePOLocalLawConfig = field(default_factory=TreePOLocalLawConfig)

    oracle_lane: Optional[OracleLaneSpec] = None
    supervision_source: Optional[SupervisionSourceSpec] = None

    oracle_scale_min: float = 0.0
    oracle_scale_max: float = 1.0

    contract_id: str = "treepo_contract"
    objective_kind: str = "tree_preservation"
    state_semantics: str = "summary_state"
    operator_requirements: Dict[str, Any] = field(default_factory=dict)
    oracle_requirements: Dict[str, Any] = field(default_factory=dict)
    theorem_domain: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    adapter_preference: Optional[str] = None


@dataclass
class TreePOStack:
    operator_g: AsyncCompositionalOperator[Any, Any]
    rubric: str
    verifiers: List[LawVerifier]
    capabilities: Dict[str, Any]
    oracle: Optional[ScoringOracle] = None

    engine: Optional[EngineType] = None
    model: Optional[str] = None
    surface: Optional[EngineSurface] = None
    surface_requested: Optional[str] = None
    surface_fallback_reason: Optional[str] = None
    base_url: Optional[str] = None

    inference_engine: Optional[Any] = None
    operator_prompt_templates: Optional[Any] = None

    def run_fixed_binary(
        self,
        leaf_spans: Sequence[Any],
        *,
        document_id: Optional[str] = None,
        refine_rounds: int = 0,
        sampling_params: Optional[Mapping[str, Any]] = None,
        engine_options: Optional[Mapping[str, Any]] = None,
        max_concurrent: int = 128,
        supervision: Optional[Any] = None,
        supervision_oracle: Optional[ScoringOracle] = None,
    ) -> FixedBinaryStateTreeRunResult[Any, Any]:
        per_run_verifiers: List[LawVerifier] = []
        for verifier in list(self.verifiers):
            if isinstance(verifier, TextAuditorAdapterVerifier):
                summarizer = _make_sync_resummarizer_from_stack(
                    self,
                    sampling_params=sampling_params,
                    engine_options=engine_options,
                )
                per_run_verifiers.append(
                    TextAuditorAdapterVerifier(
                        oracle=verifier.oracle,
                        audit_config=verifier.audit_config,
                        summarizer=summarizer,
                        theorem_operator=verifier.theorem_operator,
                        name=verifier.name,
                    )
                )
            else:
                per_run_verifiers.append(verifier)

        result = run_fixed_binary_state_tree(
            self.operator_g,
            list(leaf_spans),
            rubric=str(self.rubric or ""),
            refine_rounds=int(refine_rounds),
            sampling_params=sampling_params,
            engine_options=engine_options,
            max_concurrent=max_concurrent,
            verifiers=per_run_verifiers,
        )
        if document_id is not None:
            result.tree.metadata.setdefault("document_id", str(document_id))
        result.tree.metadata.setdefault("treepo_stack", {})
        result.tree.metadata["treepo_stack"].update(
            {
                "engine": self.engine.value if self.engine is not None else None,
                "model": self.model,
                "surface": self.surface.value if self.surface is not None else None,
                "surface_requested": self.surface_requested,
                "surface_fallback_reason": self.surface_fallback_reason,
                "base_url": self.base_url,
                "capabilities": dict(self.capabilities),
            }
        )

        effective_supervision = supervision
        if effective_supervision is None and (self.oracle is not None or supervision_oracle is not None):
            # Default: if an oracle lane is present, automatically emit at least
            # one scalar score judgment (root only). This keeps supervision
            # collection "on" by default while remaining cheap.
            from treepo._research.tree.treepo_supervision import TreePOSupervisionSpec

            effective_supervision = TreePOSupervisionSpec(
                mode="label_now",
                labeler_kind="oracle_score",
                doc_sample_probability=1.0,
                unit_selector="root",
                max_units=1,
                random_seed=0,
                output_dir="outputs/treepo_supervision_auto",
                response_signal_name="oracle_similarity",
                response_signal_min=0.0,
                response_signal_max=1.0,
                truth_label_source="oracle",
            )

        if effective_supervision is not None:
            from treepo._research.tree.treepo_supervision import (
                TreePOSupervisionSpec,
                build_supervision_dataset_from_state_tree,
                persist_supervision_dataset,
                should_collect_supervision,
            )

            spec = (
                effective_supervision
                if isinstance(effective_supervision, TreePOSupervisionSpec)
                else TreePOSupervisionSpec.from_dict(dict(effective_supervision))
            )
            if should_collect_supervision(spec, document_id=document_id):
                oracle = supervision_oracle if supervision_oracle is not None else self.oracle
                dataset = build_supervision_dataset_from_state_tree(
                    result.tree,  # type: ignore[arg-type]
                    rubric=str(self.rubric or ""),
                    spec=spec,
                    document_id=str(document_id) if document_id is not None else None,
                    oracle=oracle,
                )
                path = persist_supervision_dataset(
                    dataset,
                    spec=spec,
                    document_id=str(document_id) if document_id is not None else None,
                )
                result.tree.metadata.setdefault("treepo_supervision", {})
                result.tree.metadata["treepo_supervision"].update(
                    {
                        "mode": str(spec.mode),
                        "labeler_kind": str(spec.labeler_kind),
                        "dataset_path": str(path),
                        "judgment_count": int(len(dataset.response_judgments)),
                        "doc_sample_probability": float(spec.doc_sample_probability),
                        "unit_selector": str(spec.unit_selector),
                        "max_units": int(spec.max_units),
                    }
                )
            else:
                result.tree.metadata.setdefault("treepo_supervision", {})
                result.tree.metadata["treepo_supervision"].update(
                    {
                        "mode": str(getattr(effective_supervision, "mode", "off")),
                        "labeler_kind": str(getattr(effective_supervision, "labeler_kind", "oracle_score")),
                        "skipped": True,
                        "doc_sample_probability": float(getattr(effective_supervision, "doc_sample_probability", 0.0) or 0.0),
                    }
                )
        return result


def _resolve_surface(
    spec: TreePOModelSpec,
    *,
    engine_spec: Any,
) -> Tuple[EngineSurface, str, Optional[str]]:
    requested = str(spec.surface or "generate").strip().lower()
    if requested not in {"generate", "chat"}:
        raise ValueError(f"Unsupported surface '{spec.surface}'. Expected 'generate' or 'chat'.")

    if requested == "chat":
        return EngineSurface.CHAT_OPENAI, requested, None

    # requested == generate
    if engine_spec.supports_surface(EngineSurface.DIFFUSION_GENERATE):
        return EngineSurface.DIFFUSION_GENERATE, requested, None
    if bool(spec.prefer_generate):
        return EngineSurface.CHAT_OPENAI, requested, "engine_missing_generate_surface"
    return EngineSurface.CHAT_OPENAI, requested, "generate_not_preferred"


_AUTO_ENGINE_ALIASES = {"", "auto", "default", "infer"}


def _resolve_engine_type(model_spec: TreePOModelSpec) -> EngineType:
    raw = model_spec.engine
    if isinstance(raw, EngineType):
        return raw
    if raw is None:
        raw_text = ""
    else:
        raw_text = str(raw).strip().lower().replace("-", "_")
    if raw_text and raw_text not in _AUTO_ENGINE_ALIASES:
        return EngineType.normalize(raw_text)

    # Auto-infer from base_url when possible.
    if model_spec.base_url:
        parsed = urlparse(str(model_spec.base_url))
        host = str(parsed.hostname or "").lower()
        port = parsed.port
        path = str(parsed.path or "")

        if "openai.com" in host:
            return EngineType.OPENAI

        try:
            from treepo._research.core.engines import EngineRegistry

            for candidate in (EngineType.SGLANG, EngineType.VLLM, EngineType.VLLM_OMNI):
                spec = EngineRegistry.resolve(candidate)
                if port is not None and spec.default_port is not None and int(port) == int(spec.default_port):
                    return candidate
        except Exception:
            pass

        if "/generate" in path:
            if port == 8004:
                return EngineType.VLLM_OMNI
            if port == 30000:
                return EngineType.SGLANG
            return EngineType.CUSTOM_HTTP

        if "/v1" in path:
            if port == 8000:
                return EngineType.VLLM
            if port == 30000:
                return EngineType.SGLANG
            return EngineType.CUSTOM_HTTP

        return EngineType.CUSTOM_HTTP

    # No base_url: default to local SGLang if available.
    return EngineType.SGLANG


def _build_operator_g(model_spec: TreePOModelSpec) -> Tuple[AsyncCompositionalOperator[Any, Any], Dict[str, Any]]:
    kind = str(model_spec.kind or "inference_engine").strip().lower()
    if kind in {"markov", "markov_toy_exact", "markov_exact"}:
        operator: AsyncCompositionalOperator[Any, Any] = MarkovToyOperator()
        return operator, {
            "kind": "markov_toy_exact",
            "engine": "symbolic_local",
            "surface": EngineSurface.SYMBOLIC_EXACT.value,
        }

    if kind in {"diffusion_backend", "generate_backend"}:
        if model_spec.backend is None:
            raise ValueError("TreePOModelSpec.backend is required for kind='diffusion_backend'.")
        templates = model_spec.prompt_templates or GenerateTreePromptTemplates()
        operator = AsyncFromDiffusionBackend(
            model_spec.backend,
            prompt_templates=templates,
        )
        return operator, {
            "kind": "diffusion_backend",
            "engine": "custom_backend",
            "model": str(model_spec.model or "default"),
            "surface": EngineSurface.DIFFUSION_GENERATE.value,
            "surface_requested": "generate",
            "surface_fallback_reason": None,
            "base_url": None,
            "generate_path": str(model_spec.generate_path or "/generate"),
            "prompt_templates": templates,
            "backend": model_spec.backend,
        }

    engine_type = _resolve_engine_type(model_spec)
    from treepo._research.core.engines import EngineRegistry

    engine_spec = EngineRegistry.resolve(engine_type)
    surface, surface_requested, fallback_reason = _resolve_surface(
        model_spec,
        engine_spec=engine_spec,
    )

    resolved_base_url = model_spec.base_url or resolve_engine_base_url(
        engine_type,
        surface=surface,
        host=model_spec.host,
        port=model_spec.port,
    )
    if surface is EngineSurface.DIFFUSION_GENERATE and resolved_base_url is not None:
        resolved_base_url = normalize_generate_base_url(
            resolved_base_url,
            generate_path=model_spec.generate_path,
        )

    engine = build_inference_engine(
        engine_type,
        surface=surface,
        model=str(model_spec.model or "default"),
        host=str(model_spec.host or "localhost"),
        port=model_spec.port,
        base_url=resolved_base_url,
        api_key=model_spec.api_key,
        timeout=float(model_spec.timeout),
        generate_path=str(model_spec.generate_path or "/generate"),
    )

    templates = model_spec.prompt_templates or GenerateTreePromptTemplates()
    operator = AsyncFromInferenceEngine(
        engine,
        max_tokens=int(model_spec.max_tokens),
        temperature=float(model_spec.temperature),
        stop=tuple(model_spec.stop or ()),
        diffusion_prompt_templates=templates,
    )
    meta = {
        "kind": "inference_engine",
        "engine": engine_type.value,
        "model": str(model_spec.model or "default"),
        "surface": surface.value,
        "surface_requested": surface_requested,
        "surface_fallback_reason": fallback_reason,
        "base_url": resolved_base_url,
        "generate_path": str(model_spec.generate_path or "/generate"),
        "prompt_templates": templates,
        "inference_engine": engine,
    }
    return operator, meta


def _iter_jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            yield obj


def _build_supervision_dataset(source: SupervisionSourceSpec) -> Tuple[SupervisionDataset, Optional[Path]]:
    kind = str(source.kind or "").strip().lower()
    path = Path(str(source.path))
    if kind == "supervision_dataset_json":
        dataset = SupervisionDataset.load(path)
        return dataset, path

    judgments: List[ResponseJudgment] = []
    max_rows = int(source.max_rows) if source.max_rows is not None else None

    if kind == "csv":
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                if max_rows is not None and index >= max_rows:
                    break
                judgments.append(_judgment_from_row(row, source, index=index))
    elif kind == "jsonl":
        for index, row in enumerate(_iter_jsonl_rows(path)):
            if max_rows is not None and index >= max_rows:
                break
            judgments.append(_judgment_from_row(row, source, index=index))
    else:
        raise ValueError(
            f"Unsupported SupervisionSourceSpec.kind={source.kind!r}. "
            "Expected 'csv', 'jsonl', or 'supervision_dataset_json'."
        )

    dataset = SupervisionDataset(response_judgments=judgments)
    save_path = Path(source.save_path) if source.save_path else None
    if save_path is None:
        save_path = Path("outputs") / "treepo_stack" / f"supervision_dataset_{uuid.uuid4().hex}.json"
    dataset.save(save_path)
    return dataset, save_path


def _judgment_from_row(row: Mapping[str, Any], source: SupervisionSourceSpec, *, index: int) -> ResponseJudgment:
    text = str(row.get(source.text_column, "") or "")
    if not text:
        raise ValueError(f"Missing text in column '{source.text_column}' at row {index}.")
    raw_label = row.get(source.label_column)
    if raw_label is None:
        raise ValueError(f"Missing label in column '{source.label_column}' at row {index}.")
    try:
        label = float(raw_label)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid label {raw_label!r} at row {index}.") from exc

    example_id = None
    if source.example_id_column:
        example_id = row.get(source.example_id_column)
    if example_id is None:
        example_id = f"row_{index}"
    doc_id = None
    if source.doc_id_column:
        doc_id = row.get(source.doc_id_column)
    rubric = None
    if source.rubric_column:
        rubric = row.get(source.rubric_column)
    rubric_value = str(rubric or source.rubric or "")

    split = None
    if source.split_column:
        split = row.get(source.split_column)

    supervision_meta = judgment_supervision_metadata(
        law_type=str(source.law_type),
        response_signal_name=str(source.response_signal_name),
        response_signal_min=float(source.response_signal_min),
        response_signal_max=float(source.response_signal_max),
        metadata={
            "split": str(split) if split is not None else None,
            "source_kind": str(source.kind),
            "source_path": str(source.path),
        },
    )
    return ResponseJudgment(
        judgment_id=uuid.uuid4().hex,
        source_example_id=str(example_id),
        original_text=str(text),
        rubric=rubric_value,
        response=str(text),
        source_doc_id=str(doc_id) if doc_id is not None else None,
        law_type=str(source.law_type),
        truth_label_source="provided_dataset",
        response_signal_value=float(label),
        supervision_metadata=supervision_meta,
        metadata={
            "row_index": int(index),
            "split": str(split) if split is not None else None,
        },
    )


def _build_oracle_lane(
    contract: TreePOContractSpec,
    *,
    lane: OracleLaneSpec,
    supervision: Optional[SupervisionDataset],
) -> Tuple[Optional[ScoringOracle], Dict[str, Any]]:
    kind = str(lane.kind or "").strip().lower()

    if kind in {"markov_exact", "markov"}:
        return None, {
            "kind": "markov_exact",
            "evidence_status": EvidenceStatus.THEOREM_BACKED.value,
        }

    if kind == "provided_scoring_oracle":
        if not lane.import_path:
            raise ValueError("oracle_lane.import_path is required for provided_scoring_oracle.")
        target = _import_from_path(str(lane.import_path))
        oracle = target(**dict(lane.kwargs or {})) if callable(target) else target
        return oracle, {
            "kind": "provided_scoring_oracle",
            "import_path": str(lane.import_path),
        }

    if kind == "embedding_proxy":
        if supervision is None:
            raise ValueError("embedding_proxy oracle lane requires supervision data.")

        embedding_client = lane.embedding_client
        if embedding_client is None:
            from treepo._research.training.embedding_proxy import VLLMEmbeddingClient

            embedding_engine = EngineType.normalize(lane.embedding_engine)
            embedding_base_url = lane.embedding_base_url or resolve_engine_base_url(
                embedding_engine,
                surface=EngineSurface.EMBEDDING,
            )
            if not embedding_base_url:
                raise ValueError("embedding_proxy lane requires an embedding base_url.")
            embedding_client = VLLMEmbeddingClient(
                api_base=str(embedding_base_url),
                model=lane.embedding_model,
                api_key=str(lane.embedding_api_key or "EMPTY"),
                timeout_seconds=60.0,
            )

        scale = BoundedScale(float(contract.oracle_scale_min), float(contract.oracle_scale_max))
        denom = float(scale.range) if float(scale.range) != 0.0 else 1.0

        examples: List[LabeledEmbeddingExample] = []
        for judgment in list(supervision.response_judgments):
            raw = judgment.response_signal_value
            if raw is None:
                continue
            normalized = (float(raw) - float(scale.min_value)) / denom
            doc_id = judgment.source_doc_id or judgment.source_example_id or judgment.judgment_id
            examples.append(
                LabeledEmbeddingExample(
                    doc_id=str(doc_id),
                    text=str(judgment.response or ""),
                    target_score=float(normalized),
                    truth_label_source=str(judgment.truth_label_source or "unknown"),
                )
            )

        trained = fit_embedding_ridge_proxy(
            examples,
            embedding_client=embedding_client,
            ridge_lambda=float(lane.ridge_lambda),
            model_id=str(lane.proxy_model_id or "embedding_proxy_v1"),
        )

        artifact_path = None
        if lane.proxy_artifact_path:
            artifact_path = Path(str(lane.proxy_artifact_path))
        else:
            artifact_path = Path("outputs") / "treepo_stack" / f"{trained.model_id}.json"
        trained.save_json(artifact_path)

        def value_extractor(text: str) -> float:
            embedding = embedding_client.embed_texts([str(text or "")])[0]
            normalized_score = trained.predict_from_embedding(embedding)
            return float(scale.denormalize(float(normalized_score)))

        oracle = SimilarityScorer(value_extractor, scale, name=str(lane.value_name or "value"))
        return oracle, {
            "kind": "embedding_proxy",
            "proxy_model_id": trained.model_id,
            "artifact_path": str(artifact_path),
            "embedding_model": getattr(trained, "embedding_model", None),
            "embedding_dim": getattr(trained, "embedding_dim", None),
        }

    raise ValueError(f"Unsupported oracle_lane.kind={lane.kind!r}.")


def _resolve_chat_max_tokens(sampling_params: Optional[Mapping[str, Any]], *, default: int = 512) -> int:
    if sampling_params is None:
        return int(default)
    for key in ("max_tokens", "max_new_tokens"):
        value = sampling_params.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(default)


def _resolve_chat_temperature(sampling_params: Optional[Mapping[str, Any]], *, default: float = 0.0) -> float:
    if sampling_params is None:
        return float(default)
    value = sampling_params.get("temperature")
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _make_sync_resummarizer_from_stack(
    stack: TreePOStack,
    *,
    sampling_params: Optional[Mapping[str, Any]] = None,
    engine_options: Optional[Mapping[str, Any]] = None,
) -> Optional[Callable[[str, str], str]]:
    if stack.surface is None:
        return None
    surface = stack.surface
    templates = stack.operator_prompt_templates or GenerateTreePromptTemplates()
    resolved_sampling_params = dict(sampling_params or {})
    resolved_engine_options = dict(engine_options or {})

    if stack.inference_engine is None and surface is EngineSurface.DIFFUSION_GENERATE and hasattr(stack.operator_g, "backend"):
        from treepo._research.tree.generate_prompting import refine_prompt

        backend = getattr(stack.operator_g, "backend")
        operator_templates = getattr(stack.operator_g, "prompt_templates", None)
        if operator_templates is not None:
            templates = operator_templates

        def resummarize(text: str, rubric: str) -> str:
            prompt = refine_prompt(str(text or ""), str(rubric or ""), 1, templates)
            batch = backend.generate(
                [prompt],
                sampling_params=dict(resolved_sampling_params),
                engine_options=dict(resolved_engine_options),
            )
            if not getattr(batch, "generations", None):
                return ""
            return str(batch.generations[0].output_text)

        return resummarize

    if stack.inference_engine is None:
        return None

    engine = stack.inference_engine

    if surface is EngineSurface.CHAT_OPENAI:
        from treepo._research.core.prompting import default_resummary_prompt
        from treepo._research.runtime.contracts import ChatInput, InferenceRequest

        def resummarize(text: str, rubric: str) -> str:
            messages = default_resummary_prompt(str(text or ""), str(rubric or ""), round_index=None)
            response = engine.execute(
                InferenceRequest(
                    surface=EngineSurface.CHAT_OPENAI,
                    input=ChatInput(
                        messages=list(messages),
                        max_tokens=_resolve_chat_max_tokens(resolved_sampling_params),
                        temperature=_resolve_chat_temperature(resolved_sampling_params),
                        stop=[],
                        extra=dict(resolved_engine_options),
                    ),
                    engine_options=dict(resolved_engine_options),
                )
            )
            return str(response.to_model_response().text or "")

        return resummarize

    if surface is EngineSurface.DIFFUSION_GENERATE:
        from treepo._research.runtime.contracts import DiffusionInput, InferenceRequest, TextListOutput
        from treepo._research.tree.generate_prompting import refine_prompt

        def resummarize(text: str, rubric: str) -> str:
            prompt = refine_prompt(str(text or ""), str(rubric or ""), 1, templates)
            response = engine.execute(
                InferenceRequest(
                    surface=EngineSurface.DIFFUSION_GENERATE,
                    input=DiffusionInput(texts=[prompt], sampling_params=dict(resolved_sampling_params)),
                    engine_options=dict(resolved_engine_options),
                )
            )
            output = response.output
            if not isinstance(output, TextListOutput):
                raise TypeError(
                    f"Expected TextListOutput from diffusion resummary, got {type(output).__name__}."
                )
            return str(output.texts[0] if output.texts else "")

        return resummarize

    return None


def build_treepo_stack(
    model_spec: TreePOModelSpec | Mapping[str, Any],
    contract_spec: TreePOContractSpec | Mapping[str, Any],
) -> TreePOStack:
    """Build a ready-to-run TreePO stack (operator g + oracle lane + verifiers)."""

    resolved_model = model_spec if isinstance(model_spec, TreePOModelSpec) else TreePOModelSpec(**dict(model_spec))
    if isinstance(contract_spec, TreePOContractSpec):
        resolved_contract = contract_spec
    else:
        payload = dict(contract_spec)
        local = payload.get("local_law_config", TreePOLocalLawConfig())
        if not isinstance(local, TreePOLocalLawConfig):
            local = TreePOLocalLawConfig.from_dict(dict(local or {}))
        payload["local_law_config"] = local

        lane = payload.get("oracle_lane")
        if lane is not None and not isinstance(lane, OracleLaneSpec):
            lane = OracleLaneSpec(**dict(lane))
        payload["oracle_lane"] = lane

        source = payload.get("supervision_source")
        if source is not None and not isinstance(source, SupervisionSourceSpec):
            source = SupervisionSourceSpec(**dict(source))
        payload["supervision_source"] = source
        resolved_contract = TreePOContractSpec(**payload)

    operator_g, operator_meta = _build_operator_g(resolved_model)

    supervision_dataset: Optional[SupervisionDataset] = None
    supervision_path: Optional[Path] = None
    if resolved_contract.supervision_source is not None:
        supervision_dataset, supervision_path = _build_supervision_dataset(resolved_contract.supervision_source)

    oracle_lane = resolved_contract.oracle_lane
    if oracle_lane is None and supervision_dataset is not None:
        oracle_lane = OracleLaneSpec(kind="embedding_proxy")

    local = resolved_contract.local_law_config
    verification_enabled = bool(
        getattr(local, "enable_l1", True)
        or getattr(local, "enable_l2", True)
        or getattr(local, "enable_l3", True)
        or getattr(local, "enable_substitution", False)
    )

    if (
        oracle_lane is None
        and str(resolved_model.kind).strip().lower()
        not in {"markov", "markov_toy_exact", "markov_exact"}
        and verification_enabled
    ):
        raise ValueError(
            "build_treepo_stack requires an oracle_lane or supervision_source for text lanes "
            "when local-law verification is enabled."
        )

    oracle: Optional[ScoringOracle] = None
    oracle_meta: Dict[str, Any] = {}
    if oracle_lane is not None:
        oracle, oracle_meta = _build_oracle_lane(
            resolved_contract,
            lane=oracle_lane,
            supervision=supervision_dataset,
        )

    verifiers: List[LawVerifier] = []
    operator_kind = str(operator_meta.get("kind", "") or "").strip().lower()
    oracle_lane_kind = str(getattr(oracle_lane, "kind", "") or "").strip().lower() if oracle_lane is not None else ""
    if operator_kind in {"markov_toy_exact", "markov_exact"} or oracle_lane_kind in {"markov_exact", "markov"}:
        verifiers.append(MarkovExactVerifier())
    elif oracle is not None and verification_enabled:
        audit_config = resolved_contract.local_law_config.to_audit_config()
        placeholder_engine = None
        engine_value = operator_meta.get("engine")
        if engine_value:
            try:
                placeholder_engine = EngineType.normalize(engine_value)
            except Exception:
                placeholder_engine = None
        stack_placeholder = TreePOStack(
            operator_g=operator_g,
            rubric=str(resolved_contract.rubric or ""),
            verifiers=[],
            capabilities={},
            oracle=oracle,
            engine=placeholder_engine,
            model=str(operator_meta.get("model")) if operator_meta.get("model") else None,
            surface=EngineSurface(str(operator_meta.get("surface"))) if operator_meta.get("surface") else None,
            surface_requested=_optional_str(operator_meta.get("surface_requested")),
            surface_fallback_reason=operator_meta.get("surface_fallback_reason"),
            base_url=_optional_str(operator_meta.get("base_url")),
            inference_engine=operator_meta.get("inference_engine"),
            operator_prompt_templates=operator_meta.get("prompt_templates"),
        )
        summarizer = _make_sync_resummarizer_from_stack(stack_placeholder)
        verifiers.append(
            TextAuditorAdapterVerifier(
                oracle=oracle,
                audit_config=audit_config,
                summarizer=summarizer,
                theorem_operator=None,
            )
        )

    operator_cap = operator_g.capability_report().to_dict()
    capabilities = {
        "operator": operator_cap,
        "oracle_lane": dict(oracle_meta),
        "supervision_dataset_path": str(supervision_path) if supervision_path is not None else None,
    }

    engine_value = operator_meta.get("engine")
    engine_type = None
    if isinstance(engine_value, str) and engine_value:
        try:
            engine_type = EngineType.normalize(engine_value)
        except Exception:
            engine_type = None

    surface_value = operator_meta.get("surface")
    surface_enum = None
    if isinstance(surface_value, str) and surface_value:
        try:
            surface_enum = EngineSurface(surface_value)
        except Exception:
            surface_enum = None

    stack = TreePOStack(
        operator_g=operator_g,
        rubric=str(resolved_contract.rubric or ""),
        verifiers=verifiers,
        capabilities=capabilities,
        oracle=oracle,
        engine=engine_type,
        model=str(operator_meta.get("model")) if operator_meta.get("model") else None,
        surface=surface_enum,
        surface_requested=_optional_str(operator_meta.get("surface_requested")),
        surface_fallback_reason=operator_meta.get("surface_fallback_reason"),
        base_url=_optional_str(operator_meta.get("base_url")),
        inference_engine=operator_meta.get("inference_engine"),
        operator_prompt_templates=operator_meta.get("prompt_templates"),
    )
    return stack


__all__ = [
    "TreePOModelSpec",
    "TreePOContractSpec",
    "TreePOLocalLawConfig",
    "SupervisionSourceSpec",
    "OracleLaneSpec",
    "TreePOStack",
    "build_treepo_stack",
]
