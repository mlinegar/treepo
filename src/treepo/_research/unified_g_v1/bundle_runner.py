from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from treepo._research.unified_g_v1.bundles import (
    build_openai_record_eval_bundle,
    create_text_artifact_bundle,
    run_embedding_fno_train_bundle,
    run_embedding_sequence_smoke_bundle,
    run_preference_data_bundle,
    run_preference_optimization_bundle,
    run_text_audit_bundle,
    run_text_batch_bundle,
    run_text_dspy_optimize_bundle,
    run_dspy_rile_bundle,
    run_dspy_rile_tree_bundle,
    run_text_llm_trl_sft_bundle,
    run_token_sequence_fno_report_bundle,
    run_token_sequence_fno_smoke_bundle,
)
from treepo._research.unified_g_v1.core.artifact import UnifiedGArtifact
from treepo._research.unified_g_v1.core.contracts import MarkovRunSpec, MarkovScope, Profile, SupervisionPolicy
from treepo._research.unified_g_v1.core.manifest import now_iso, write_json
from treepo._research.unified_g_v1.core.supervision import UnifiedGSupervisionDataset
from treepo._research.unified_g_v1.realdoc.dspy_optimize import LawStressDSPyOptimizeConfig
from treepo._research.unified_g_v1.training.param_coerce import opt_float, opt_int, opt_str

from treepo._research.training.trl_training import TRLTrainingConfig


REPO_ROOT = Path(__file__).resolve().parents[4]
_BUNDLE_SCRIPT = REPO_ROOT / "parallel" / "unified_g_v1" / "scripts" / "run_unified_g_bundle.py"
_LONG_JOB_SCRIPT = REPO_ROOT / "scripts" / "long_job.py"


def _load_json_or_toml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser()
    if resolved.suffix.lower() == ".json":
        return dict(json.loads(resolved.read_text(encoding="utf-8")))
    import tomllib

    return dict(tomllib.loads(resolved.read_text(encoding="utf-8")))


def _load_reward_func(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError("reward_func must be in <module_or_file>:<symbol> form")
    module_spec, symbol = spec.split(":", 1)
    if module_spec.endswith(".py") and Path(module_spec).exists():
        module_path = Path(module_spec).expanduser()
        module_name = f"unified_g_v1_reward_{module_path.stem}"
        import_spec = importlib.util.spec_from_file_location(module_name, module_path)
        if import_spec is None or import_spec.loader is None:
            raise ValueError(f"Could not load reward function module from {module_path}")
        module = importlib.util.module_from_spec(import_spec)
        import_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_spec)
    return getattr(module, symbol)


def _coerce_artifact(path: str | Path | None) -> UnifiedGArtifact | None:
    if not path:
        return None
    return UnifiedGArtifact.load(Path(path).expanduser())


def _coerce_markov_spec(payload: Mapping[str, Any]) -> MarkovRunSpec:
    return MarkovRunSpec(
        scope=MarkovScope.parse(str(payload.get("scope", ""))),
        train_docs=int(payload.get("train_docs", 0)),
        root_share=int(payload.get("root_share", 0)),
        leaf_tokens=int(payload.get("leaf_tokens", 0)),
        supervision_policy=SupervisionPolicy.parse(str(payload.get("supervision_policy", ""))),
        profile=Profile.parse(str(payload.get("profile", ""))),
        seed=int(payload.get("seed", 0)),
    )


def _coerce_markov_scopes(values: Sequence[Any] | None) -> tuple[MarkovScope, ...]:
    if values is None:
        return (
            MarkovScope.RECOVERABLE_V5_T128,
            MarkovScope.R12_P079,
        )
    return tuple(MarkovScope.parse(str(value)) for value in values)


@dataclass(frozen=True)
class BundleApproach:
    name: str
    runner: Callable[[str | Path, Mapping[str, Any]], dict[str, Any]]
    description: str


@dataclass(frozen=True)
class BundleRunSpec:
    approach: str
    output_root: str | Path
    params: Mapping[str, Any] = field(default_factory=dict)
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "approach": str(self.approach),
            "output_root": str(self.output_root),
            "params": dict(self.params),
            "label": str(self.label),
        }


def _run_text_artifact(output_root: str | Path, params: Mapping[str, Any]) -> dict[str, Any]:
    return create_text_artifact_bundle(
        output_root,
        artifact=_coerce_artifact(params.get("artifact_path")),
        label=str(params.get("label", "baseline")),
    )


def _run_text_audit(output_root: str | Path, params: Mapping[str, Any]) -> dict[str, Any]:
    return run_text_audit_bundle(
        output_root,
        doc_id=str(params["doc_id"]),
        artifact=_coerce_artifact(params.get("artifact_path")),
        chunk_size=int(params.get("chunk_size", 8000)),
        min_chunk_chars=int(params.get("min_chunk_chars", 400)),
    )


def _run_text_batch(output_root: str | Path, params: Mapping[str, Any]) -> dict[str, Any]:
    doc_ids = params.get("doc_ids", params.get("ids"))
    if doc_ids is None:
        raise ValueError("text_batch requires doc_ids or ids")
    return run_text_batch_bundle(
        output_root,
        doc_ids=[str(value) for value in doc_ids],
        artifact=_coerce_artifact(params.get("artifact_path")),
        chunk_size=int(params.get("chunk_size", 8000)),
        min_chunk_chars=int(params.get("min_chunk_chars", 400)),
    )


def _run_record_eval(output_root: str | Path, params: Mapping[str, Any]) -> dict[str, Any]:
    records_path = params.get("records_path", params.get("records"))
    if not records_path:
        raise ValueError("record_eval requires records_path")
    scorer_base_url = params.get("scorer_base_url")
    scorer_model = params.get("scorer_model")
    if not scorer_base_url or not scorer_model:
        raise ValueError("record_eval requires scorer_base_url and scorer_model")
    return build_openai_record_eval_bundle(
        output_root,
        records_path=records_path,
        artifact=_coerce_artifact(params.get("artifact_path")),
        scorer_base_url=str(scorer_base_url),
        scorer_model=str(scorer_model),
        scorer_api_key=str(params.get("scorer_api_key", "EMPTY")),
        scorer_timeout_seconds=float(params.get("scorer_timeout_seconds", 120.0)),
        scorer_temperature=float(params.get("scorer_temperature", 0.0)),
        scorer_max_tokens=int(params.get("scorer_max_tokens", 16)),
        enable_thinking=bool(params.get("enable_thinking", False)),
        num_workers=int(params.get("num_workers", 2)),
    )


def _run_embedding_sequence_smoke(output_root: str | Path, params: Mapping[str, Any]) -> dict[str, Any]:
    doc_ids = params.get("doc_ids", params.get("ids"))
    if doc_ids is None:
        raise ValueError("embedding_sequence_smoke requires doc_ids or ids")
    return run_embedding_sequence_smoke_bundle(
        output_root,
        doc_ids=[str(value) for value in doc_ids],
        embedding_dim=int(params.get("embedding_dim", 64)),
        summary_dim=opt_int(params, "summary_dim"),
        state_dim=opt_int(params, "state_dim"),
        leaf_tokens=int(params.get("leaf_tokens", 1024)),
        subwindow_tokens=int(params.get("subwindow_tokens", 128)),
        token_encoding=str(params.get("token_encoding", "cl100k_base")),
        head_width=opt_int(params, "head_width"),
        operator_modes=opt_int(params, "operator_modes"),
        embedding_backend=str(params.get("embedding_backend", "hash")),
        embedding_api_base=params.get("embedding_api_base"),
        embedding_model=params.get("embedding_model"),
        seed=int(params.get("seed", 0)),
        salt=str(params.get("salt", "unified_g_v1")),
    )


def _build_trl_training_config(params: Mapping[str, Any]) -> TRLTrainingConfig | None:
    raw = params.get("trl_config")
    if raw is None:
        return None
    if isinstance(raw, TRLTrainingConfig):
        return raw
    if isinstance(raw, Mapping):
        return TRLTrainingConfig(**dict(raw))
    raise ValueError("trl_config must be a mapping or TRLTrainingConfig")


def _run_dspy_rile(output_root: str | Path, params: Mapping[str, Any]) -> dict[str, Any]:
    prepared_path = params.get("prepared_dataset_path") or params.get("prepared_dataset")
    if not prepared_path:
        raise ValueError("dspy_rile requires prepared_dataset_path")
    model_name = params.get("model_name")
    if not model_name:
        raise ValueError("dspy_rile requires model_name")
    return run_dspy_rile_bundle(
        output_root,
        prepared_dataset_path=str(prepared_path),
        api_base=str(params.get("api_base", "http://localhost:8005/v1")),
        model_name=str(model_name),
        api_key=str(params.get("api_key", "EMPTY")),
        temperature=float(params.get("temperature", 0.0)),
        max_tokens=int(params.get("max_tokens", 1024)),
        max_bootstrapped_demos=int(params.get("max_bootstrapped_demos", 4)),
        max_train_examples=int(params.get("max_train_examples", 0)),
        n_val=int(params.get("n_val", 0)),
        seed=int(params.get("seed", 0)),
        optimizer=str(params.get("optimizer", "gepa")),
        gepa_auto=str(params.get("gepa_auto", "medium")),
        gepa_num_threads=int(params.get("gepa_num_threads", 16)),
        gepa_max_metric_calls=int(params.get("gepa_max_metric_calls", 0)),
        gepa_reflection_minibatch_size=int(params.get("gepa_reflection_minibatch_size", 3)),
        gepa_valset_cap=int(params.get("gepa_valset_cap", 64)),
        reflection_api_base=str(params.get("reflection_api_base", "")),
        reflection_model_name=str(params.get("reflection_model_name", "")),
        reflection_max_tokens=int(params.get("reflection_max_tokens", 16384)),
    )


def _run_dspy_rile_tree(output_root: str | Path, params: Mapping[str, Any]) -> dict[str, Any]:
    model_name = params.get("model_name")
    if not model_name:
        raise ValueError("dspy_rile_tree requires model_name")
    return run_dspy_rile_tree_bundle(
        output_root,
        phase1_data_path=params.get("phase1_data_path"),
        split_ids_path=params.get("split_ids_path"),
        api_base=str(params.get("api_base", "http://localhost:8000/v1")),
        model_name=str(model_name),
        api_key=str(params.get("api_key", "EMPTY")),
        temperature=float(params.get("temperature", 0.0)),
        max_tokens=int(params.get("max_tokens", 1024)),
        leaf_tokens=int(params.get("leaf_tokens", 1024)),
        token_encoding=str(params.get("token_encoding", "cl100k_base")),
        max_bootstrapped_demos=int(params.get("max_bootstrapped_demos", 4)),
        max_train_examples=int(params.get("max_train_examples", 0)),
        n_val=int(params.get("n_val", 0)),
        seed=int(params.get("seed", 0)),
        optimizer=str(params.get("optimizer", "gepa")),
        gepa_auto=str(params.get("gepa_auto", "medium")),
        gepa_num_threads=int(params.get("gepa_num_threads", 16)),
        gepa_max_metric_calls=int(params.get("gepa_max_metric_calls", 0)),
        gepa_reflection_minibatch_size=int(params.get("gepa_reflection_minibatch_size", 3)),
        gepa_valset_cap=int(params.get("gepa_valset_cap", 64)),
        reflection_api_base=str(params.get("reflection_api_base", "")),
        reflection_model_name=str(params.get("reflection_model_name", "")),
        reflection_max_tokens=int(params.get("reflection_max_tokens", 16384)),
        local_law_weight=float(params.get("local_law_weight", 0.3)),
        c1_relative_weight=float(params.get("c1_relative_weight", 1.0)),
        c2_relative_weight=float(params.get("c2_relative_weight", 1.0)),
        c3_relative_weight=float(params.get("c3_relative_weight", 1.0)),
    )


def _run_text_llm_trl_sft(output_root: str | Path, params: Mapping[str, Any]) -> dict[str, Any]:
    prepared_path = params.get("prepared_dataset_path") or params.get("prepared_dataset")
    if not prepared_path:
        raise ValueError("text_llm_trl_sft requires prepared_dataset_path")
    model_name = params.get("model_name")
    if not model_name:
        raise ValueError("text_llm_trl_sft requires model_name")
    return run_text_llm_trl_sft_bundle(
        output_root,
        prepared_dataset_path=str(prepared_path),
        model_name=str(model_name),
        trl_config=_build_trl_training_config(params),
        execute=bool(params.get("execute", True)),
    )


def _run_embedding_fno_train(
    output_root: str | Path,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    return run_embedding_fno_train_bundle(
        output_root,
        phase1_data_path=params.get("phase1_data_path"),
        split_ids_path=params.get("split_ids_path"),
        embedding_api_base=str(params.get("embedding_api_base", "http://localhost:8006/v1")),
        embedding_model=params.get("embedding_model"),
        embedding_api_key=str(params.get("embedding_api_key", "EMPTY")),
        embedding_timeout_seconds=float(params.get("embedding_timeout_seconds", 60.0)),
        embedding_batch_size=int(params.get("embedding_batch_size", 32)),
        leaf_tokens=int(params.get("leaf_tokens", 1024)),
        subwindow_tokens=int(params.get("subwindow_tokens", 128)),
        token_encoding=str(params.get("token_encoding", "cl100k_base")),
        summary_dim=opt_int(params, "summary_dim"),
        state_dim=opt_int(params, "state_dim"),
        adapter_hidden_dim=opt_int(params, "adapter_hidden_dim"),
        g_hidden_dim=opt_int(params, "g_hidden_dim"),
        head_width=opt_int(params, "head_width"),
        operator_modes=opt_int(params, "operator_modes"),
        train_batch_size=int(params.get("train_batch_size", 4)),
        epochs=int(params.get("epochs", 8)),
        learning_rate=float(params.get("learning_rate", 3e-4)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
        grad_clip_norm=float(params.get("grad_clip_norm", 1.0)),
        seed=int(params.get("seed", 42)),
        device=str(params.get("device", "auto")),
        save_every_epoch=bool(params.get("save_every_epoch", False)),
        execute=bool(params.get("execute", True)),
    )


def _run_text_dspy_optimize(output_root: str | Path, params: Mapping[str, Any]) -> dict[str, Any]:
    records_path = params.get("records_path", params.get("records"))
    if not records_path:
        raise ValueError("text_dspy_optimize requires records_path")
    return run_text_dspy_optimize_bundle(
        output_root,
        config=LawStressDSPyOptimizeConfig(
            records_path=records_path,
            student_port=int(params.get("student_port", 8000)),
            student_model=params.get("student_model"),
            student_temperature=float(params.get("student_temperature", 0.2)),
            student_max_tokens=int(params.get("student_max_tokens", 0)),
            enable_thinking=bool(params.get("enable_thinking", False)),
            gepa_reflection_model=params.get("gepa_reflection_model"),
            gepa_reflection_temperature=float(params.get("gepa_reflection_temperature", 0.0)),
            gepa_reflection_max_tokens=int(params.get("gepa_reflection_max_tokens", 0)),
            embedding_url=str(params.get("embedding_url", "http://localhost:8003/v1")),
            embedding_model=str(params.get("embedding_model", "Qwen/Qwen3-Embedding-8B")),
            embedding_api_key=str(params.get("embedding_api_key", "EMPTY")),
            embedding_timeout_seconds=float(params.get("embedding_timeout_seconds", 60.0)),
            embedding_batch_size=int(params.get("embedding_batch_size", 32)),
            proxy_path=params.get("proxy_path"),
            ridge_lambda=float(params.get("ridge_lambda", 1.0)),
            proxy_model_id=str(params.get("proxy_model_id", "lawstress_embedding_ridge_proxy_v1")),
            gepa_budget=str(params.get("gepa_budget", "light")),
            num_threads=int(params.get("num_threads", 8)),
            gepa_max_metric_calls=int(params.get("gepa_max_metric_calls", 0)),
            gepa_max_full_evals=int(params.get("gepa_max_full_evals", 0)),
            seed=int(params.get("seed", 0)),
            c1_threshold_norm=float(params.get("c1_threshold_norm", 0.10)),
            c2_threshold_norm=float(params.get("c2_threshold_norm", 0.06)),
            c3_threshold_norm=float(params.get("c3_threshold_norm", 0.08)),
            objective_aggregate=str(params.get("objective_aggregate", "min")),
            objective_softmin_temperature=float(params.get("objective_softmin_temperature", 0.08)),
            objective_component_floor=float(params.get("objective_component_floor", 0.55)),
            verbose=bool(params.get("verbose", False)),
        ),
        execute=bool(params.get("execute", True)),
        python_executable=params.get("python_executable"),
        script_path=params.get("script_path"),
        strict=bool(params.get("strict", True)),
    )


def _run_token_fno_smoke(output_root: str | Path, params: Mapping[str, Any]) -> dict[str, Any]:
    raw_specs = params.get("specs")
    specs = None if raw_specs is None else [_coerce_markov_spec(dict(item)) for item in raw_specs]
    return run_token_sequence_fno_smoke_bundle(
        output_root,
        train_docs=int(params.get("train_docs", 1024)),
        seed=int(params.get("seed", 0)),
        scope=MarkovScope.parse(str(params.get("scope", MarkovScope.RECOVERABLE_V5_T128.value))),
        use_cuda=bool(params.get("use_cuda", False)),
        cuda_device=opt_int(params, "cuda_device"),
        torch_threads=int(params.get("torch_threads", 1)),
        reuse_existing=bool(params.get("reuse_existing", True)),
        specs=specs,
    )


def _run_token_fno_report(output_root: str | Path, params: Mapping[str, Any]) -> dict[str, Any]:
    return run_token_sequence_fno_report_bundle(
        output_root,
        train_docs=int(params.get("train_docs", 10240)),
        scopes=_coerce_markov_scopes(params.get("scopes")),
        root_shares=tuple(int(value) for value in params.get("root_shares", (100, 90, 80, 70, 60, 50, 40, 30, 20, 10))),
        leaf_tokens=tuple(int(value) for value in params.get("leaf_tokens", (128, 64, 32, 16, 8))),
        seed=int(params.get("seed", 0)),
        include_duplicate_local_label_one_leaf=bool(params.get("include_duplicate_local_label_one_leaf", True)),
        use_cuda=bool(params.get("use_cuda", True)),
        cuda_device=opt_int(params, "cuda_device"),
        torch_threads=int(params.get("torch_threads", 0)),
        reuse_existing=bool(params.get("reuse_existing", True)),
        config_overrides=dict(params.get("config_overrides", {}) or {}),
    )


def _run_preference_data(output_root: str | Path, params: Mapping[str, Any]) -> dict[str, Any]:
    supervision_path = params.get("supervision_path", params.get("supervision"))
    if not supervision_path:
        raise ValueError("preference_data requires supervision_path")
    resolved_supervision_path = Path(str(supervision_path)).expanduser()
    if not resolved_supervision_path.is_absolute():
        resolved_supervision_path = REPO_ROOT / resolved_supervision_path
    return run_preference_data_bundle(
        output_root,
        supervision_path=resolved_supervision_path,
        phase1_data_path=params.get("phase1_data_path"),
        split_ids_path=params.get("split_ids_path"),
        law_type=params.get("law_type"),
    )


def _run_preference_optimize(output_root: str | Path, params: Mapping[str, Any]) -> dict[str, Any]:
    supervision_path = params.get("supervision_path", params.get("supervision"))
    if not supervision_path:
        raise ValueError("preference_optimize requires supervision_path")
    resolved_supervision_path = Path(str(supervision_path)).expanduser()
    if not resolved_supervision_path.is_absolute():
        resolved_supervision_path = REPO_ROOT / resolved_supervision_path
    trl_config = None
    trl_config_json = params.get("trl_config_json")
    if trl_config_json:
        trl_config = TRLTrainingConfig(**_load_json_or_toml(trl_config_json))
    reward_funcs = None
    reward_func = params.get("reward_func")
    if reward_func:
        reward_funcs = _load_reward_func(str(reward_func))
    return run_preference_optimization_bundle(
        output_root,
        supervision_path=resolved_supervision_path,
        phase1_data_path=params.get("phase1_data_path"),
        split_ids_path=params.get("split_ids_path"),
        law_type=params.get("law_type"),
        train_mode=str(params.get("train_mode", "none")),
        model_name=params.get("model_name"),
        trl_config=trl_config,
        reward_funcs=reward_funcs,
    )


_APPROACHES: tuple[BundleApproach, ...] = (
    BundleApproach(
        name="text_artifact",
        runner=_run_text_artifact,
        description="Store a baseline or preloaded text Unified-G artifact bundle.",
    ),
    BundleApproach(
        name="text_audit",
        runner=_run_text_audit,
        description="Run one text-document audit bundle with the text Unified-G path.",
    ),
    BundleApproach(
        name="text_batch",
        runner=_run_text_batch,
        description="Run a batched text-document bundle with the text Unified-G path.",
    ),
    BundleApproach(
        name="record_eval",
        runner=_run_record_eval,
        description="Run record-based evaluation with an OpenAI-compatible scorer.",
    ),
    BundleApproach(
        name="embedding_sequence_smoke",
        runner=_run_embedding_sequence_smoke,
        description="Run the embedding-sequence smoke bundle.",
    ),
    BundleApproach(
        name="embedding_fno_train",
        runner=_run_embedding_fno_train,
        description="Run or plan embedding-sequence FNO/FNO training on an explicit split.",
    ),
    BundleApproach(
        name="text_dspy_optimize",
        runner=_run_text_dspy_optimize,
        description="Run or plan DSPy optimization for the text LLM/LLM path.",
    ),
    BundleApproach(
        name="text_llm_trl_sft",
        runner=_run_text_llm_trl_sft,
        description="Run TRL SFT fine-tuning on a prepared text_pairs_v1 dataset via fit().",
    ),
    BundleApproach(
        name="dspy_rile",
        runner=_run_dspy_rile,
        description="Bootstrap a DSPy RILE predictor against a running vLLM endpoint (no local GPU training).",
    ),
    BundleApproach(
        name="dspy_rile_tree",
        runner=_run_dspy_rile_tree,
        description="Bootstrap the tree-structured DSPy RILE program against a running vLLM endpoint.",
    ),
    BundleApproach(
        name="token_fno_smoke",
        runner=_run_token_fno_smoke,
        description="Run the reduced token-sequence FNO/FNO smoke bundle.",
    ),
    BundleApproach(
        name="token_fno_report",
        runner=_run_token_fno_report,
        description="Run the full fixed-report token-sequence FNO/FNO bundle.",
    ),
    BundleApproach(
        name="preference_data",
        runner=_run_preference_data,
        description="Stage preference data and export DPO/GRPO/reward datasets.",
    ),
    BundleApproach(
        name="preference_optimize",
        runner=_run_preference_optimize,
        description="Export preference bundles and optionally run TRL training.",
    ),
)

_APPROACH_BY_NAME = {entry.name: entry for entry in _APPROACHES}
_APPROACH_ALIASES = {
    "text_unified_g_artifact": "text_artifact",
    "manifesto_audit": "text_audit",
    "manifesto_batch": "text_batch",
    "lawstress_eval": "record_eval",
    "embedding_rile": "embedding_sequence_smoke",
    "manifesto_embedding_fno_train": "embedding_fno_train",
    "dspy_unified_g_optimize": "text_dspy_optimize",
    "markov_smoke": "token_fno_smoke",
    "markov_fixed_report": "token_fno_report",
    "trl_export": "preference_optimize",
}


def list_bundle_approaches() -> list[dict[str, str]]:
    return [
        {
            "name": entry.name,
            "description": entry.description,
        }
        for entry in _APPROACHES
    ]


def parse_bundle_run_spec(payload: Mapping[str, Any]) -> BundleRunSpec:
    approach = str(payload.get("approach", "") or "").strip()
    approach = _APPROACH_ALIASES.get(approach, approach)
    output_root = payload.get("output_root")
    if not approach:
        raise ValueError("bundle run spec requires approach")
    if not output_root:
        raise ValueError("bundle run spec requires output_root")
    params = dict(payload.get("params") or {})
    if not params:
        for key, value in payload.items():
            if key not in {"approach", "output_root", "label"}:
                params[key] = value
    return BundleRunSpec(
        approach=approach,
        output_root=output_root,
        params=params,
        label=str(payload.get("label", "") or ""),
    )


def _merge_shared_params(
    payload: Mapping[str, Any],
    *,
    shared_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(payload)
    if not shared_params:
        return merged
    params = dict(shared_params)
    params.update(dict(merged.get("params") or {}))
    merged["params"] = params
    return merged


def _resolve_collection_specs(
    specs: Sequence[BundleRunSpec | Mapping[str, Any]],
    *,
    collection_root: str | Path | None = None,
    shared_params: Mapping[str, Any] | None = None,
) -> tuple[Path | None, list[BundleRunSpec]]:
    collection_dir = Path(collection_root).expanduser() if collection_root is not None else None
    if collection_dir is not None:
        collection_dir.mkdir(parents=True, exist_ok=True)
    resolved_specs: list[BundleRunSpec] = []
    for index, spec in enumerate(specs):
        if isinstance(spec, BundleRunSpec):
            base_payload = spec.to_dict()
        else:
            base_payload = dict(spec)
        payload = _merge_shared_params(base_payload, shared_params=shared_params)
        if not payload.get("output_root"):
            if collection_dir is None:
                raise ValueError(
                    "bundle collection entries require output_root when collection_root is not set"
                )
            payload["output_root"] = f"{index + 1:02d}_{payload.get('approach', 'bundle')}"
        resolved_specs.append(parse_bundle_run_spec(payload))
    materialized_specs: list[BundleRunSpec] = []
    for index, spec in enumerate(resolved_specs):
        output_root = Path(spec.output_root).expanduser()
        if collection_dir is not None and not output_root.is_absolute() and str(output_root) in {"", "."}:
            suffix = f"{index + 1:02d}_{spec.approach}"
            output_root = collection_dir / suffix
        elif collection_dir is not None and str(spec.output_root).startswith("./"):
            output_root = collection_dir / str(spec.output_root)[2:]
        elif collection_dir is not None and not output_root.is_absolute():
            output_root = collection_dir / output_root
        materialized_specs.append(
            BundleRunSpec(
                approach=spec.approach,
                output_root=output_root,
                params=dict(spec.params),
                label=spec.label,
            )
        )
    return collection_dir, materialized_specs


def run_bundle(
    spec: BundleRunSpec | Mapping[str, Any],
) -> dict[str, Any]:
    resolved = spec if isinstance(spec, BundleRunSpec) else parse_bundle_run_spec(spec)
    entry = _APPROACH_BY_NAME.get(str(resolved.approach))
    if entry is None:
        valid = ", ".join(sorted(_APPROACH_BY_NAME))
        raise ValueError(f"unsupported bundle approach={resolved.approach!r}; expected one of: {valid}")
    result = entry.runner(resolved.output_root, dict(resolved.params))
    return {
        "approach": entry.name,
        "label": str(resolved.label),
        "output_root": str(resolved.output_root),
        "result": result,
    }


def run_bundle_collection(
    specs: Sequence[BundleRunSpec | Mapping[str, Any]],
    *,
    collection_root: str | Path | None = None,
    collection_name: str = "bundle_collection",
    shared_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    collection_dir, resolved_specs = _resolve_collection_specs(
        specs,
        collection_root=collection_root,
        shared_params=shared_params,
    )
    results: list[dict[str, Any]] = []
    for spec in resolved_specs:
        run_result = run_bundle(
            spec
        )
        results.append(run_result)
    payload = {
        "generated_at": now_iso(),
        "collection_name": str(collection_name),
        "collection_root": str(collection_dir) if collection_dir is not None else "",
        "run_count": len(results),
        "shared_params": dict(shared_params or {}),
        "runs": results,
    }
    if collection_dir is not None:
        write_json(collection_dir / "collection_manifest.json", payload)
    return payload


def launch_bundle_collection(
    specs: Sequence[BundleRunSpec | Mapping[str, Any]],
    *,
    collection_root: str | Path,
    collection_name: str = "bundle_collection",
    shared_params: Mapping[str, Any] | None = None,
    python_executable: str | Path | None = None,
    long_job_python: str | Path | None = None,
) -> dict[str, Any]:
    collection_dir, resolved_specs = _resolve_collection_specs(
        specs,
        collection_root=collection_root,
        shared_params=shared_params,
    )
    if collection_dir is None:
        raise ValueError("launch_bundle_collection requires collection_root")
    launch_specs_dir = collection_dir / "launch_specs"
    launchers_dir = collection_dir / "launchers"
    launch_specs_dir.mkdir(parents=True, exist_ok=True)
    launchers_dir.mkdir(parents=True, exist_ok=True)
    bundle_python = str(Path(python_executable or sys.executable).expanduser())
    launcher_python = str(Path(long_job_python or sys.executable).expanduser())
    launches: list[dict[str, Any]] = []
    for index, spec in enumerate(resolved_specs, start=1):
        spec_path = launch_specs_dir / f"{index:02d}_{spec.approach}.json"
        write_json(spec_path, spec.to_dict())
        job_root = launchers_dir / f"{index:02d}_{spec.approach}"
        launch_cmd = [
            launcher_python,
            str(_LONG_JOB_SCRIPT),
            "launch",
            "--name",
            f"{collection_name}_{index:02d}_{spec.approach}",
            "--job-root",
            str(job_root),
            "--cwd",
            str(REPO_ROOT),
            "--",
            bundle_python,
            str(_BUNDLE_SCRIPT),
            "run",
            "--config",
            str(spec_path),
        ]
        completed = subprocess.run(
            launch_cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if int(completed.returncode) != 0:
            raise RuntimeError(
                "Failed to launch bundle collection job "
                f"{spec.approach}: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        launch_payload = json.loads(completed.stdout or "{}")
        launches.append(
            {
                "approach": spec.approach,
                "label": spec.label,
                "output_root": str(spec.output_root),
                "spec_path": str(spec_path),
                "job_root": str(job_root),
                "launch": launch_payload,
            }
        )
    payload = {
        "generated_at": now_iso(),
        "collection_name": str(collection_name),
        "collection_root": str(collection_dir),
        "shared_params": dict(shared_params or {}),
        "launch_count": len(launches),
        "launches": launches,
    }
    write_json(collection_dir / "parallel_launch_manifest.json", payload)
    return payload


def load_bundle_collection_config(path: str | Path) -> dict[str, Any]:
    return _load_json_or_toml(path)
