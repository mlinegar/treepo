from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

from treepo._research.unified_g_v1.core.artifact import (
    TextUnifiedGProgram,
    UnifiedGArtifact,
    resolve_text_unified_g_program,
)
from treepo._research.unified_g_v1.core.bundle import StoredRunBundle, create_stored_run_bundle
from treepo._research.unified_g_v1.core.contracts import MarkovRunSpec, MarkovScope
from treepo._research.unified_g_v1.core.manifest import write_json
from treepo._research.unified_g_v1.core.splits import resolve_document_split_ids, write_split_ids_json
from treepo._research.unified_g_v1.core.specs import (
    build_embedding_sequence_fno_program_spec,
    build_llm_text_program_spec,
)
from treepo._research.unified_g_v1.core.supervision import UnifiedGSupervisionDataset
from treepo._research.unified_g_v1.markov.report import build_fixed_report_summary, render_fixed_report
from treepo._research.unified_g_v1.markov.runner import run_fixed_report_suite
from treepo._research.unified_g_v1.markov.smoke import run_markov_smoke_suite
from treepo._research.unified_g_v1.realdoc.dspy_optimize import (
    LawStressDSPyOptimizeConfig,
    build_lawstress_dspy_command,
    run_lawstress_dspy_optimization,
)
from treepo._research.unified_g_v1.realdoc.embedding import run_manifesto_embedding_smoke
from treepo._research.unified_g_v1.realdoc.embedding_fno_training import (
    EmbeddingFNOTrainingConfig,
    run_embedding_fno_training,
)
from treepo._research.unified_g_v1.realdoc.lawstress import (
    LawStressEvalConfig,
    OpenAIChatClient,
    build_numeric_score_fn,
    evaluate_lawstress_records,
    load_lawstress_records,
)
from treepo._research.unified_g_v1.realdoc.manifesto import audit_manifesto_document, run_manifesto_batch
from treepo._research.unified_g_v1.realdoc.trl import export_supervision_formats

from treepo._research.tasks.manifesto.lawstress_generator import LawStressRecord
from treepo._research.training.trl_training import TRLTrainingConfig


REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_BUNDLE_PYTHON = sys.executable
_DEFAULT_LLM_TRAIN_SCRIPT = REPO_ROOT / "scripts" / "run_training_pipeline.sh"
_DEFAULT_EMBEDDING_TRAIN_SCRIPT = (
    REPO_ROOT / "parallel" / "unified_g_v1" / "scripts" / "run_manifesto_embedding_fno_training.py"
)


def _bundle_program_and_artifact(
    artifact: UnifiedGArtifact | TextUnifiedGProgram | None,
    *,
    bundle: StoredRunBundle,
) -> tuple[TextUnifiedGProgram, str]:
    program = resolve_text_unified_g_program(artifact)
    saved_path = ""
    artifact_value: UnifiedGArtifact | None = None
    if artifact is None:
        artifact_value = UnifiedGArtifact.baseline()
    elif isinstance(artifact, UnifiedGArtifact):
        artifact_value = artifact
    else:
        runtime = artifact.runtime
        if isinstance(runtime, Mapping):
            candidate = runtime.get("artifact")
            if isinstance(candidate, UnifiedGArtifact):
                artifact_value = candidate
        elif runtime is not None:
            candidate = getattr(runtime, "artifact", None)
            if isinstance(candidate, UnifiedGArtifact):
                artifact_value = candidate
    if artifact_value is not None:
        saved_path = str(artifact_value.save(bundle.artifacts_dir / "unified_g_final.json"))
    return program, saved_path


def _resolve_and_save_split_ids(
    *,
    bundle: StoredRunBundle,
    phase1_data_path: str | Path | None = None,
    split_ids_path: str | Path | None = None,
) -> tuple[str, dict[str, list[str]]]:
    split_ids, split_source = resolve_document_split_ids(
        phase1_data_path=phase1_data_path,
        split_ids_path=split_ids_path,
    )
    saved_path = write_split_ids_json(split_ids, bundle.inputs_dir / "split_ids.json")
    return str(saved_path), {
        "train": list(split_ids.train_doc_ids),
        "val": list(split_ids.val_doc_ids),
        "test": list(split_ids.test_doc_ids),
        "source_path": str(split_source),
    }


def _write_command_payload(
    *,
    path: Path,
    argv: Sequence[str],
    cwd: str | Path,
    env: Mapping[str, str] | None = None,
) -> Path:
    return write_json(
        path,
        {
            "argv": [str(item) for item in argv],
            "cwd": str(Path(cwd).expanduser()),
            "env": dict(env or {}),
        },
    )


def _run_external_command(
    *,
    argv: Sequence[str],
    cwd: str | Path,
    stdout_path: str | Path,
    stderr_path: str | Path,
) -> dict[str, Any]:
    stdout_file = Path(stdout_path).expanduser()
    stderr_file = Path(stderr_path).expanduser()
    stdout_file.parent.mkdir(parents=True, exist_ok=True)
    stderr_file.parent.mkdir(parents=True, exist_ok=True)
    with stdout_file.open("w", encoding="utf-8") as stdout_handle, stderr_file.open(
        "w",
        encoding="utf-8",
    ) as stderr_handle:
        completed = subprocess.run(
            [str(item) for item in argv],
            cwd=str(Path(cwd).expanduser()),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    return {
        "returncode": int(completed.returncode),
        "stdout_log": str(stdout_file),
        "stderr_log": str(stderr_file),
    }


def create_text_artifact_bundle(
    output_root: str | Path,
    *,
    artifact: UnifiedGArtifact | TextUnifiedGProgram | None = None,
    label: str = "baseline",
) -> dict[str, Any]:
    bundle = create_stored_run_bundle(output_root, approach="text_artifact")
    program, artifact_path = _bundle_program_and_artifact(artifact, bundle=bundle)
    payload = {
        "artifact_path": artifact_path,
        "label": str(label),
        "program_contract": program.contract.to_dict(),
    }
    result_path = write_json(bundle.results_dir / "artifact_summary.json", payload)
    bundle.write_manifest(
        config={"label": str(label)},
        result_paths={
            "artifact_summary": str(result_path),
            "artifact_path": artifact_path,
        },
        program_contract=program.contract.to_dict(),
    )
    return {
        "bundle_root": str(bundle.root),
        "artifact_summary": str(result_path),
        "artifact_path": artifact_path,
        "bundle_manifest": str(bundle.manifest_path),
    }


def run_text_audit_bundle(
    output_root: str | Path,
    *,
    doc_id: str,
    artifact: UnifiedGArtifact | TextUnifiedGProgram | None = None,
    chunk_size: int = 8000,
    min_chunk_chars: int = 400,
    score_fn: Any | None = None,
) -> dict[str, Any]:
    bundle = create_stored_run_bundle(output_root, approach="text_audit")
    program, artifact_path = _bundle_program_and_artifact(artifact, bundle=bundle)
    payload = audit_manifesto_document(
        str(doc_id),
        artifact=program,
        chunk_size=int(chunk_size),
        min_chunk_chars=int(min_chunk_chars),
        score_fn=score_fn,
    ).to_dict()
    result_path = write_json(bundle.results_dir / "manifesto_audit.json", payload)
    bundle.write_manifest(
        config={
            "doc_id": str(doc_id),
            "chunk_size": int(chunk_size),
            "min_chunk_chars": int(min_chunk_chars),
        },
        result_paths={
            "audit_result": str(result_path),
            "artifact_path": artifact_path,
        },
        program_contract=program.contract.to_dict(),
    )
    return {
        "bundle_root": str(bundle.root),
        "audit_result": str(result_path),
        "artifact_path": artifact_path,
        "bundle_manifest": str(bundle.manifest_path),
    }


def run_manifesto_audit_bundle(
    output_root: str | Path,
    *,
    doc_id: str,
    artifact: UnifiedGArtifact | TextUnifiedGProgram | None = None,
    chunk_size: int = 8000,
    min_chunk_chars: int = 400,
    score_fn: Any | None = None,
) -> dict[str, Any]:
    return run_text_audit_bundle(
        output_root,
        doc_id=doc_id,
        artifact=artifact,
        chunk_size=chunk_size,
        min_chunk_chars=min_chunk_chars,
        score_fn=score_fn,
    )


def run_text_batch_bundle(
    output_root: str | Path,
    *,
    doc_ids: Sequence[str],
    artifact: UnifiedGArtifact | TextUnifiedGProgram | None = None,
    chunk_size: int = 8000,
    min_chunk_chars: int = 400,
    score_fn: Any | None = None,
) -> dict[str, Any]:
    bundle = create_stored_run_bundle(output_root, approach="text_batch")
    program, artifact_path = _bundle_program_and_artifact(artifact, bundle=bundle)
    payload = {
        "doc_ids": [str(doc_id) for doc_id in doc_ids],
        "results": run_manifesto_batch(
            [str(doc_id) for doc_id in doc_ids],
            artifact=program,
            chunk_size=int(chunk_size),
            min_chunk_chars=int(min_chunk_chars),
            score_fn=score_fn,
        ),
    }
    result_path = write_json(bundle.results_dir / "manifesto_batch.json", payload)
    bundle.write_manifest(
        config={
            "doc_ids": [str(doc_id) for doc_id in doc_ids],
            "chunk_size": int(chunk_size),
            "min_chunk_chars": int(min_chunk_chars),
        },
        result_paths={
            "batch_result": str(result_path),
            "artifact_path": artifact_path,
        },
        program_contract=program.contract.to_dict(),
    )
    return {
        "bundle_root": str(bundle.root),
        "batch_result": str(result_path),
        "artifact_path": artifact_path,
        "bundle_manifest": str(bundle.manifest_path),
    }


def run_manifesto_batch_bundle(
    output_root: str | Path,
    *,
    doc_ids: Sequence[str],
    artifact: UnifiedGArtifact | TextUnifiedGProgram | None = None,
    chunk_size: int = 8000,
    min_chunk_chars: int = 400,
    score_fn: Any | None = None,
) -> dict[str, Any]:
    return run_text_batch_bundle(
        output_root,
        doc_ids=doc_ids,
        artifact=artifact,
        chunk_size=chunk_size,
        min_chunk_chars=min_chunk_chars,
        score_fn=score_fn,
    )


def run_record_eval_bundle(
    output_root: str | Path,
    *,
    records: str | Path | Sequence[LawStressRecord],
    artifact: UnifiedGArtifact | TextUnifiedGProgram | None = None,
    score_fn: Any,
    judge_fn: Any | None = None,
    config: LawStressEvalConfig | None = None,
    num_workers: int = 2,
    records_path: str | Path | None = None,
    score_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bundle = create_stored_run_bundle(output_root, approach="record_eval")
    program, artifact_path = _bundle_program_and_artifact(artifact, bundle=bundle)
    if isinstance(records, (str, Path)):
        resolved_records = load_lawstress_records(records)
        records_source = str(Path(records).expanduser())
    else:
        resolved_records = list(records)
        records_source = str(records_path or "")
    payload = evaluate_lawstress_records(
        resolved_records,
        artifact=program,
        score_fn=score_fn,
        judge_fn=judge_fn,
        config=config,
        num_workers=int(num_workers),
    )
    result_path = write_json(bundle.results_dir / "lawstress_eval.json", payload)
    bundle.write_manifest(
        config={
            "record_count": len(resolved_records),
            "num_workers": int(num_workers),
            "records_source": records_source,
            "score_spec": dict(score_spec or {}),
        },
        result_paths={
            "lawstress_eval": str(result_path),
            "artifact_path": artifact_path,
        },
        program_contract=program.contract.to_dict(),
    )
    return {
        "bundle_root": str(bundle.root),
        "lawstress_eval": str(result_path),
        "artifact_path": artifact_path,
        "bundle_manifest": str(bundle.manifest_path),
    }


def run_lawstress_eval_bundle(
    output_root: str | Path,
    *,
    records: str | Path | Sequence[LawStressRecord],
    artifact: UnifiedGArtifact | TextUnifiedGProgram | None = None,
    score_fn: Any,
    judge_fn: Any | None = None,
    config: LawStressEvalConfig | None = None,
    num_workers: int = 2,
    records_path: str | Path | None = None,
    score_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return run_record_eval_bundle(
        output_root,
        records=records,
        artifact=artifact,
        score_fn=score_fn,
        judge_fn=judge_fn,
        config=config,
        num_workers=num_workers,
        records_path=records_path,
        score_spec=score_spec,
    )


def run_embedding_sequence_smoke_bundle(
    output_root: str | Path,
    *,
    doc_ids: Sequence[str],
    embedding_dim: int = 64,
    summary_dim: int | None = None,
    state_dim: int | None = None,
    leaf_tokens: int = 1024,
    subwindow_tokens: int = 128,
    token_encoding: str = "cl100k_base",
    head_width: int | None = None,
    operator_modes: int | None = None,
    embedding_backend: str = "hash",
    embedding_api_base: str | None = None,
    embedding_model: str | None = None,
    seed: int = 0,
    salt: str = "unified_g_v1",
) -> dict[str, Any]:
    bundle = create_stored_run_bundle(output_root, approach="embedding_sequence_smoke")
    payload = run_manifesto_embedding_smoke(
        doc_ids,
        embedding_dim=int(embedding_dim),
        summary_dim=summary_dim,
        state_dim=state_dim,
        leaf_tokens=int(leaf_tokens),
        subwindow_tokens=int(subwindow_tokens),
        token_encoding=str(token_encoding),
        head_width=head_width,
        operator_modes=operator_modes,
        embedding_backend=str(embedding_backend),
        embedding_api_base=embedding_api_base,
        embedding_model=embedding_model,
        seed=int(seed),
        salt=str(salt),
    )
    result_path = write_json(bundle.results_dir / "manifesto_embedding.json", payload)
    bundle.write_manifest(
        config={
            "doc_ids": [str(doc_id) for doc_id in doc_ids],
            "embedding_dim": int(embedding_dim),
            "summary_dim": None if summary_dim is None else int(summary_dim),
            "state_dim": None if state_dim is None else int(state_dim),
            "leaf_tokens": int(leaf_tokens),
            "subwindow_tokens": int(subwindow_tokens),
            "token_encoding": str(token_encoding),
            "embedding_backend": str(embedding_backend),
            "embedding_api_base": str(embedding_api_base or ""),
            "embedding_model": str(embedding_model or ""),
            "seed": int(seed),
            "salt": str(salt),
        },
        result_paths={"embedding_result": str(result_path)},
        program_contract=dict(payload.get("program_contract") or {}),
    )
    return {
        "bundle_root": str(bundle.root),
        "embedding_result": str(result_path),
        "bundle_manifest": str(bundle.manifest_path),
    }


def run_manifesto_embedding_bundle(
    output_root: str | Path,
    *,
    doc_ids: Sequence[str],
    embedding_dim: int = 64,
    summary_dim: int | None = None,
    state_dim: int | None = None,
    leaf_tokens: int = 1024,
    subwindow_tokens: int = 128,
    token_encoding: str = "cl100k_base",
    head_width: int | None = None,
    operator_modes: int | None = None,
    embedding_backend: str = "hash",
    embedding_api_base: str | None = None,
    embedding_model: str | None = None,
    seed: int = 0,
    salt: str = "unified_g_v1",
) -> dict[str, Any]:
    return run_embedding_sequence_smoke_bundle(
        output_root,
        doc_ids=doc_ids,
        embedding_dim=embedding_dim,
        summary_dim=summary_dim,
        state_dim=state_dim,
        leaf_tokens=leaf_tokens,
        subwindow_tokens=subwindow_tokens,
        token_encoding=token_encoding,
        head_width=head_width,
        operator_modes=operator_modes,
        embedding_backend=embedding_backend,
        embedding_api_base=embedding_api_base,
        embedding_model=embedding_model,
        seed=seed,
        salt=salt,
    )


def run_text_llm_train_bundle(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Removed. Use run_text_dspy_optimize_bundle or run_text_llm_trl_sft_bundle."""
    del args, kwargs
    raise RuntimeError(
        "run_text_llm_train_bundle (shell-script subprocess) was removed. "
        "Use run_text_dspy_optimize_bundle (DSPy bootstrap, in-process) or "
        "run_text_llm_trl_sft_bundle (TRL SFT via fit())."
    )


def run_manifesto_llm_train_bundle(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Removed alias. Use run_text_dspy_optimize_bundle or run_text_llm_trl_sft_bundle."""
    del args, kwargs
    raise RuntimeError("run_manifesto_llm_train_bundle was removed. See run_text_llm_train_bundle docstring.")


def run_text_llm_trl_sft_bundle(
    output_root: str | Path,
    *,
    prepared_dataset_path: str | Path,
    model_name: str,
    trl_config: TRLTrainingConfig | None = None,
    execute: bool = True,
) -> dict[str, Any]:
    """Run TRL SFT fine-tuning on a `text_pairs_v1` prepared dataset via `fit()`.

    Companion prep script: `parallel/unified_g_v1/scripts/prep_text_dataset.py`.
    """
    from treepo._research.unified_g_v1.training import (
        PreparedDataset,
        TrainerConfig,
        run_training_bundle,
    )

    if not execute:
        bundle = create_stored_run_bundle(output_root, approach="text_llm_trl_sft")
        return {
            "bundle_root": str(bundle.root),
            "approach": "text_llm_trl_sft",
            "status": "planned",
            "bundle_manifest": str(bundle.manifest_path),
        }
    prepared = PreparedDataset.load(prepared_dataset_path)
    return run_training_bundle(
        output_root,
        approach="text_llm_trl_sft",
        trainer_config=TrainerConfig(
            trainer="trl_sft",
            model_name=str(model_name),
            trl_config=trl_config,
        ),
        dataset=prepared,
        inputs={
            "prepared_dataset_path": str(prepared.root),
            "prepared_dataset_manifest": dict(prepared.manifest),
            "model_name": str(model_name),
            "trl_config": None if trl_config is None else trl_config.__dict__,
        },
    )


def run_embedding_fno_train_bundle(
    output_root: str | Path,
    *,
    phase1_data_path: str | Path | None = None,
    split_ids_path: str | Path | None = None,
    embedding_api_base: str = "http://localhost:8006/v1",
    embedding_model: str | None = None,
    embedding_api_key: str = "EMPTY",
    embedding_timeout_seconds: float = 60.0,
    embedding_batch_size: int = 32,
    leaf_tokens: int = 1024,
    subwindow_tokens: int = 128,
    token_encoding: str = "cl100k_base",
    # None => auto-size to embedding server's output dim (summary_dim)
    # or 2 * summary_dim (state_dim). Matches run_embedding_fno_training behavior.
    summary_dim: int | None = None,
    state_dim: int | None = None,
    adapter_hidden_dim: int | None = 512,
    g_hidden_dim: int | None = 512,
    head_width: int | None = None,
    operator_modes: int | None = 32,
    train_batch_size: int = 4,
    epochs: int = 8,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
    seed: int = 42,
    device: str = "auto",
    save_every_epoch: bool = False,
    # Canonical λ/ρ local-law balance — default matches `scripts/run_markov_publication_bundle.py`.
    local_law_weight: float = 0.3,
    c1_relative_weight: float = 1.0,
    c2_relative_weight: float = 1.0,
    c3_relative_weight: float = 1.0,
    execute: bool = True,
) -> dict[str, Any]:
    bundle = create_stored_run_bundle(output_root, approach="embedding_fno_train")
    split_ids_json, split_payload = _resolve_and_save_split_ids(
        bundle=bundle,
        phase1_data_path=phase1_data_path,
        split_ids_path=split_ids_path,
    )
    training_output_root = bundle.training_dir / "embedding_fno_train"
    config = EmbeddingFNOTrainingConfig(
        output_dir=str(training_output_root),
        embedding_api_base=str(embedding_api_base),
        phase1_data_path=None,
        split_ids_path=str(split_ids_json),
        embedding_model=None if embedding_model is None else str(embedding_model),
        embedding_api_key=str(embedding_api_key),
        embedding_timeout_seconds=float(embedding_timeout_seconds),
        embedding_batch_size=int(embedding_batch_size),
        leaf_tokens=int(leaf_tokens),
        subwindow_tokens=int(subwindow_tokens),
        token_encoding=str(token_encoding),
        summary_dim=None if summary_dim is None else int(summary_dim),
        state_dim=None if state_dim is None else int(state_dim),
        adapter_hidden_dim=None if adapter_hidden_dim is None else int(adapter_hidden_dim),
        g_hidden_dim=None if g_hidden_dim is None else int(g_hidden_dim),
        head_width=None if head_width is None else int(head_width),
        operator_modes=None if operator_modes is None else int(operator_modes),
        train_batch_size=int(train_batch_size),
        n_epochs=int(epochs),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        grad_clip_norm=float(grad_clip_norm),
        seed=int(seed),
        device=str(device),
        save_every_epoch=bool(save_every_epoch),
        local_law_weight=float(local_law_weight),
        c1_relative_weight=float(c1_relative_weight),
        c2_relative_weight=float(c2_relative_weight),
        c3_relative_weight=float(c3_relative_weight),
    )
    config_path = write_json(
        bundle.inputs_dir / "embedding_fno_train_config.json",
        {
            **config.__dict__,
            "resolved_split_ids_path": str(split_ids_json),
            "split_counts": {
                "train": int(len(split_payload["train"])),
                "val": int(len(split_payload["val"])),
                "test": int(len(split_payload["test"])),
            },
            "execute": bool(execute),
        },
    )
    embedding_command = [
        str(_DEFAULT_BUNDLE_PYTHON),
        str(_DEFAULT_EMBEDDING_TRAIN_SCRIPT),
        "--split-ids-path",
        str(split_ids_json),
        "--output-dir",
        str(training_output_root),
        "--embedding-api-base",
        str(embedding_api_base),
        "--embedding-api-key",
        str(embedding_api_key),
        "--embedding-timeout-seconds",
        str(float(embedding_timeout_seconds)),
        "--embedding-batch-size",
        str(int(embedding_batch_size)),
        "--leaf-tokens",
        str(int(leaf_tokens)),
        "--subwindow-tokens",
        str(int(subwindow_tokens)),
        "--token-encoding",
        str(token_encoding),
        "--train-batch-size",
        str(int(train_batch_size)),
        "--epochs",
        str(int(epochs)),
        "--learning-rate",
        str(float(learning_rate)),
        "--weight-decay",
        str(float(weight_decay)),
        "--grad-clip-norm",
        str(float(grad_clip_norm)),
        "--seed",
        str(int(seed)),
        "--device",
        str(device),
        "--local-law-weight",
        str(float(local_law_weight)),
        "--c1-relative-weight",
        str(float(c1_relative_weight)),
        "--c2-relative-weight",
        str(float(c2_relative_weight)),
        "--c3-relative-weight",
        str(float(c3_relative_weight)),
    ]
    if embedding_model is not None:
        embedding_command.extend(["--embedding-model", str(embedding_model)])
    if summary_dim is not None:
        embedding_command.extend(["--summary-dim", str(int(summary_dim))])
    if state_dim is not None:
        embedding_command.extend(["--state-dim", str(int(state_dim))])
    if adapter_hidden_dim is not None:
        embedding_command.extend(["--adapter-hidden-dim", str(int(adapter_hidden_dim))])
    if g_hidden_dim is not None:
        embedding_command.extend(["--g-hidden-dim", str(int(g_hidden_dim))])
    if head_width is not None:
        embedding_command.extend(["--head-width", str(int(head_width))])
    if operator_modes is not None:
        embedding_command.extend(["--operator-modes", str(int(operator_modes))])
    if save_every_epoch:
        embedding_command.append("--save-every-epoch")
    command_path = _write_command_payload(
        path=bundle.inputs_dir / "embedding_fno_train_command.json",
        argv=embedding_command,
        cwd=REPO_ROOT,
    )
    program_spec = build_embedding_sequence_fno_program_spec(
        feature_dim=int(summary_dim or 0),
        tokenizer_or_adapter_id=str(embedding_model or embedding_api_base),
        operator_width=int(head_width or max(1, int(state_dim or summary_dim or 1))),
        operator_modes=int(operator_modes or 32),
    )
    summary_payload: dict[str, Any] = {
        "status": "planned",
        "executed": False,
        "training_output_root": str(training_output_root),
        "summary_json": "",
        "best_model": "",
        "history_json": "",
        "train_predictions_json": "",
        "val_predictions_json": "",
        "split_ids_path": str(split_ids_json),
        "split_counts": {
            "train": int(len(split_payload["train"])),
            "val": int(len(split_payload["val"])),
            "test": int(len(split_payload["test"])),
        },
        "program_spec": program_spec.to_dict(),
    }
    if execute:
        payload = run_embedding_fno_training(config)
        artifacts = dict(payload.get("artifacts") or {})
        summary_payload.update(
            {
                "status": "completed",
                "executed": True,
                "summary_json": str(training_output_root / "summary.json"),
                "best_model": str(artifacts.get("best_model", "")),
                "history_json": str(artifacts.get("history_json", "")),
                "train_predictions_json": str(artifacts.get("train_predictions_json", "")),
                "val_predictions_json": str(artifacts.get("val_predictions_json", "")),
                "best_val_mae_raw": payload.get("best_val_mae_raw"),
                "program_contract": payload.get("program_contract"),
            }
        )
    summary_path = write_json(
        bundle.results_dir / "embedding_fno_train_summary.json",
        summary_payload,
    )
    bundle.write_manifest(
        config=json.loads(Path(config_path).read_text(encoding="utf-8")),
        result_paths={
            "config": str(config_path),
            "command": str(command_path),
            "summary": str(summary_path),
            "training_output_root": str(training_output_root),
            "split_ids_path": str(split_ids_json),
            "summary_json": summary_payload["summary_json"],
            "best_model": summary_payload["best_model"],
            "history_json": summary_payload["history_json"],
            "train_predictions_json": summary_payload["train_predictions_json"],
            "val_predictions_json": summary_payload["val_predictions_json"],
        },
        program_contract=dict(summary_payload.get("program_contract") or program_spec.to_dict()),
        extra={
            "status": str(summary_payload["status"]),
            "best_val_mae_raw": summary_payload.get("best_val_mae_raw"),
        },
    )
    return {
        "bundle_root": str(bundle.root),
        "config": str(config_path),
        "command": str(command_path),
        "summary": str(summary_path),
        "training_output_root": str(training_output_root),
        "bundle_manifest": str(bundle.manifest_path),
    }


def run_dspy_rile_bundle(
    output_root: str | Path,
    *,
    prepared_dataset_path: str | Path,
    api_base: str = "http://localhost:8005/v1",
    model_name: str = "google/gemma-4-31b-it",
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_bootstrapped_demos: int = 4,
    max_train_examples: int = 0,
    n_val: int = 0,
    seed: int = 0,
    optimizer: str = "gepa",
    gepa_auto: str = "medium",
    gepa_num_threads: int = 16,
    gepa_max_metric_calls: int = 0,
    gepa_reflection_minibatch_size: int = 3,
    gepa_valset_cap: int = 64,
    reflection_api_base: str = "",
    reflection_model_name: str = "",
    reflection_max_tokens: int = 16384,
) -> dict[str, Any]:
    """Bootstrap a DSPy RILE predictor against a running vLLM endpoint."""
    from treepo._research.unified_g_v1.training import (
        PreparedDataset,
        TrainerConfig,
        run_training_bundle,
    )
    from treepo._research.unified_g_v1.training.oracles import ManifestoRileTextOracle

    prepared = PreparedDataset.load(prepared_dataset_path)
    oracle = ManifestoRileTextOracle(prepared_dataset=prepared)
    return run_training_bundle(
        output_root,
        approach="dspy_rile",
        trainer_config=TrainerConfig(
            trainer="dspy_rile",
            oracle=oracle,
            model_name=str(model_name),
            seed=int(seed),
            extra={
                "api_base": str(api_base),
                "api_key": str(api_key),
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "max_bootstrapped_demos": int(max_bootstrapped_demos),
                "max_train_examples": int(max_train_examples),
                "n_val": int(n_val),
                "optimizer": str(optimizer),
                "gepa_auto": str(gepa_auto),
                "gepa_num_threads": int(gepa_num_threads),
                "gepa_max_metric_calls": int(gepa_max_metric_calls),
                "gepa_reflection_minibatch_size": int(gepa_reflection_minibatch_size),
                "gepa_valset_cap": int(gepa_valset_cap),
                "reflection_api_base": str(reflection_api_base),
                "reflection_model_name": str(reflection_model_name),
                "reflection_max_tokens": int(reflection_max_tokens),
            },
        ),
        inputs={
            "prepared_dataset_path": str(prepared.root),
            "api_base": str(api_base),
            "model_name": str(model_name),
            "max_bootstrapped_demos": int(max_bootstrapped_demos),
        },
    )


def run_dspy_rile_tree_bundle(
    output_root: str | Path,
    *,
    phase1_data_path: str | Path | None = None,
    split_ids_path: str | Path | None = None,
    api_base: str = "http://localhost:8000/v1",
    model_name: str = "google/gemma-4-31b-it",
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    leaf_tokens: int = 1024,
    token_encoding: str = "cl100k_base",
    max_bootstrapped_demos: int = 4,
    max_train_examples: int = 0,
    n_val: int = 0,
    seed: int = 0,
    optimizer: str = "gepa",
    gepa_auto: str = "medium",
    gepa_num_threads: int = 16,
    gepa_max_metric_calls: int = 0,
    gepa_reflection_minibatch_size: int = 3,
    gepa_valset_cap: int = 64,
    reflection_api_base: str = "",
    reflection_model_name: str = "",
    reflection_max_tokens: int = 16384,
    local_law_weight: float = 0.3,
    c1_relative_weight: float = 1.0,
    c2_relative_weight: float = 1.0,
    c3_relative_weight: float = 1.0,
) -> dict[str, Any]:
    """Bootstrap the tree-structured DSPy RILE program against a running vLLM endpoint."""
    from treepo._research.unified_g_v1.training import TrainerConfig, run_training_bundle
    from treepo._research.unified_g_v1.training.oracles import ManifestoRileTreeOracle

    oracle = ManifestoRileTreeOracle(
        phase1_data_path=phase1_data_path,
        split_ids_path=split_ids_path,
        leaf_tokens=int(leaf_tokens),
        token_encoding=str(token_encoding),
        enforce_local_laws=True,
    )
    return run_training_bundle(
        output_root,
        approach="dspy_rile_tree",
        trainer_config=TrainerConfig(
            trainer="dspy_rile_tree",
            oracle=oracle,
            model_name=str(model_name),
            seed=int(seed),
            extra={
                "api_base": str(api_base),
                "api_key": str(api_key),
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "max_bootstrapped_demos": int(max_bootstrapped_demos),
                "max_train_examples": int(max_train_examples),
                "n_val": int(n_val),
                "optimizer": str(optimizer),
                "gepa_auto": str(gepa_auto),
                "gepa_num_threads": int(gepa_num_threads),
                "gepa_max_metric_calls": int(gepa_max_metric_calls),
                "gepa_reflection_minibatch_size": int(gepa_reflection_minibatch_size),
                "gepa_valset_cap": int(gepa_valset_cap),
                "reflection_api_base": str(reflection_api_base),
                "reflection_model_name": str(reflection_model_name),
                "reflection_max_tokens": int(reflection_max_tokens),
                "local_law_weight": float(local_law_weight),
                "c1_relative_weight": float(c1_relative_weight),
                "c2_relative_weight": float(c2_relative_weight),
                "c3_relative_weight": float(c3_relative_weight),
            },
        ),
        inputs={
            "phase1_data_path": None if phase1_data_path is None else str(phase1_data_path),
            "split_ids_path": None if split_ids_path is None else str(split_ids_path),
            "api_base": str(api_base),
            "model_name": str(model_name),
            "leaf_tokens": int(leaf_tokens),
            "local_law_weight": float(local_law_weight),
            "c1_relative_weight": float(c1_relative_weight),
            "c2_relative_weight": float(c2_relative_weight),
            "c3_relative_weight": float(c3_relative_weight),
        },
    )


def run_manifesto_embedding_fno_train_bundle(
    output_root: str | Path,
    *,
    phase1_data_path: str | Path | None = None,
    split_ids_path: str | Path | None = None,
    embedding_api_base: str = "http://localhost:8006/v1",
    embedding_model: str | None = None,
    embedding_api_key: str = "EMPTY",
    embedding_timeout_seconds: float = 60.0,
    embedding_batch_size: int = 32,
    leaf_tokens: int = 1024,
    subwindow_tokens: int = 128,
    token_encoding: str = "cl100k_base",
    # None => auto-size to embedding server's output dim (summary_dim)
    # or 2 * summary_dim (state_dim). Matches run_embedding_fno_training behavior.
    summary_dim: int | None = None,
    state_dim: int | None = None,
    adapter_hidden_dim: int | None = 512,
    g_hidden_dim: int | None = 512,
    head_width: int | None = None,
    operator_modes: int | None = 32,
    train_batch_size: int = 4,
    epochs: int = 8,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
    seed: int = 42,
    device: str = "auto",
    save_every_epoch: bool = False,
    local_law_weight: float = 0.3,
    c1_relative_weight: float = 1.0,
    c2_relative_weight: float = 1.0,
    c3_relative_weight: float = 1.0,
    execute: bool = True,
) -> dict[str, Any]:
    return run_embedding_fno_train_bundle(
        output_root,
        phase1_data_path=phase1_data_path,
        split_ids_path=split_ids_path,
        embedding_api_base=embedding_api_base,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_timeout_seconds=embedding_timeout_seconds,
        embedding_batch_size=embedding_batch_size,
        leaf_tokens=leaf_tokens,
        subwindow_tokens=subwindow_tokens,
        token_encoding=token_encoding,
        summary_dim=summary_dim,
        state_dim=state_dim,
        adapter_hidden_dim=adapter_hidden_dim,
        g_hidden_dim=g_hidden_dim,
        head_width=head_width,
        operator_modes=operator_modes,
        train_batch_size=train_batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        seed=seed,
        device=device,
        save_every_epoch=save_every_epoch,
        local_law_weight=float(local_law_weight),
        c1_relative_weight=float(c1_relative_weight),
        c2_relative_weight=float(c2_relative_weight),
        c3_relative_weight=float(c3_relative_weight),
        execute=execute,
    )


def run_text_dspy_optimize_bundle(
    output_root: str | Path,
    *,
    config: LawStressDSPyOptimizeConfig,
    execute: bool = True,
    python_executable: str | Path | None = None,
    script_path: str | Path | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    bundle = create_stored_run_bundle(output_root, approach="text_dspy_optimize")
    config_path = write_json(bundle.inputs_dir / "text_dspy_optimize_config.json", config.to_dict())
    training_output_root = bundle.training_dir / "lawstress_bootstrap"
    planned_command = build_lawstress_dspy_command(
        config,
        output_dir=training_output_root,
        python_executable=python_executable,
        script_path=script_path,
    )
    command_path = write_json(
        bundle.inputs_dir / "text_dspy_optimize_command.json",
        {
            "argv": planned_command,
            "output_dir": str(training_output_root),
        },
    )
    from treepo._research.unified_g_v1.training import TrainerConfig, fit

    fit_result = fit(
        trainer_config=TrainerConfig(
            dspy_config=config,
            dspy_execute=bool(execute),
            dspy_python_executable=python_executable,
            dspy_script_path=script_path,
        ),
        output_dir=training_output_root,
    )
    payload = dict(fit_result.summary.get("raw_payload") or fit_result.summary)
    copied_artifact = ""
    program_contract = {}
    source_artifact = Path(str(payload.get("artifact_path", ""))).expanduser()
    if source_artifact.exists():
        copied_path = bundle.artifacts_dir / "unified_g_final.json"
        shutil.copy2(source_artifact, copied_path)
        copied_artifact = str(copied_path)
        program_contract = UnifiedGArtifact.baseline().contract.to_dict()
    summary_payload = {
        "status": str(payload.get("status", "planned")),
        "executed": bool(payload.get("executed", False)),
        "returncode": payload.get("returncode"),
        "command": list(payload.get("command") or []),
        "command_pretty": str(payload.get("command_pretty", "")),
        "training_output_root": str(training_output_root),
        "bootstrap_stats": payload.get("bootstrap_stats"),
        "source_artifact_path": str(payload.get("artifact_path", "")),
        "artifact_path": copied_artifact,
        "stdout_log": str(payload.get("stdout_log", "")),
        "stderr_log": str(payload.get("stderr_log", "")),
        "bootstrap_stats_path": str(payload.get("bootstrap_stats_path", "")),
    }
    summary_path = write_json(bundle.results_dir / "text_dspy_optimize_summary.json", summary_payload)
    bundle.write_manifest(
        config={
            **config.to_dict(),
            "execute": bool(execute),
            "strict": bool(strict),
            "python_executable": str(python_executable) if python_executable is not None else "",
            "script_path": str(Path(script_path).expanduser()) if script_path is not None else "",
        },
        result_paths={
            "config": str(config_path),
            "command": str(command_path),
            "summary": str(summary_path),
            "artifact_path": copied_artifact,
            "training_output_root": str(training_output_root),
            "stdout_log": str(payload.get("stdout_log", "")),
            "stderr_log": str(payload.get("stderr_log", "")),
            "bootstrap_stats_path": str(payload.get("bootstrap_stats_path", "")),
        },
        program_contract=program_contract,
        extra={
            "status": str(payload.get("status", "planned")),
            "returncode": payload.get("returncode"),
        },
    )
    if bool(execute) and bool(strict) and str(payload.get("status", "")) == "failed":
        raise RuntimeError(
            "DSPy Unified-G optimization failed; inspect "
            f"{payload.get('stdout_log', '')} and {payload.get('stderr_log', '')}"
        )
    return {
        "bundle_root": str(bundle.root),
        "config": str(config_path),
        "command": str(command_path),
        "summary": str(summary_path),
        "artifact_path": copied_artifact,
        "training_output_root": str(training_output_root),
        "bundle_manifest": str(bundle.manifest_path),
    }


def run_dspy_unified_g_optimize_bundle(
    output_root: str | Path,
    *,
    config: LawStressDSPyOptimizeConfig,
    execute: bool = True,
    python_executable: str | Path | None = None,
    script_path: str | Path | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    return run_text_dspy_optimize_bundle(
        output_root,
        config=config,
        execute=execute,
        python_executable=python_executable,
        script_path=script_path,
        strict=strict,
    )


def run_token_sequence_fno_smoke_bundle(
    output_root: str | Path,
    *,
    train_docs: int = 1024,
    seed: int = 0,
    scope: MarkovScope = MarkovScope.RECOVERABLE_V5_T128,
    use_cuda: bool = False,
    cuda_device: int | None = None,
    torch_threads: int = 1,
    reuse_existing: bool = True,
    specs: Sequence[MarkovRunSpec] | None = None,
) -> dict[str, Any]:
    bundle = create_stored_run_bundle(output_root, approach="token_fno_smoke")
    resolved_specs = None if specs is None else list(specs)
    payload = run_markov_smoke_suite(
        output_root=bundle.root,
        train_docs=int(train_docs),
        seed=int(seed),
        scope=scope,
        use_cuda=bool(use_cuda),
        cuda_device=cuda_device,
        torch_threads=int(torch_threads),
        reuse_existing=bool(reuse_existing),
        specs=resolved_specs,
    )
    bundle.write_manifest(
        config={
            "train_docs": int(train_docs),
            "seed": int(seed),
            "scope": scope.value,
            "use_cuda": bool(use_cuda),
            "cuda_device": cuda_device,
            "torch_threads": int(torch_threads),
            "reuse_existing": bool(reuse_existing),
            "spec_count": len(resolved_specs) if resolved_specs is not None else None,
        },
        result_paths={"smoke_manifest": str(bundle.root / "smoke_manifest.json")},
        extra={"run_count": int(payload.get("run_count", 0))},
    )
    return {
        "bundle_root": str(bundle.root),
        "smoke_manifest": str(bundle.root / "smoke_manifest.json"),
        "bundle_manifest": str(bundle.manifest_path),
    }


def run_markov_smoke_bundle(
    output_root: str | Path,
    *,
    train_docs: int = 1024,
    seed: int = 0,
    scope: MarkovScope = MarkovScope.RECOVERABLE_V5_T128,
    use_cuda: bool = False,
    cuda_device: int | None = None,
    torch_threads: int = 1,
    reuse_existing: bool = True,
    specs: Sequence[MarkovRunSpec] | None = None,
) -> dict[str, Any]:
    return run_token_sequence_fno_smoke_bundle(
        output_root,
        train_docs=train_docs,
        seed=seed,
        scope=scope,
        use_cuda=use_cuda,
        cuda_device=cuda_device,
        torch_threads=torch_threads,
        reuse_existing=reuse_existing,
        specs=specs,
    )


def run_token_sequence_fno_report_bundle(
    output_root: str | Path,
    *,
    train_docs: int = 10240,
    scopes: Sequence[MarkovScope] = (
        MarkovScope.RECOVERABLE_V5_T128,
        MarkovScope.R12_P079,
    ),
    root_shares: Sequence[int] = (100, 90, 80, 70, 60, 50, 40, 30, 20, 10),
    leaf_tokens: Sequence[int] = (128, 64, 32, 16, 8),
    seed: int = 0,
    include_duplicate_local_label_one_leaf: bool = True,
    use_cuda: bool = True,
    cuda_device: int | None = None,
    torch_threads: int = 0,
    reuse_existing: bool = True,
    config_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bundle = create_stored_run_bundle(output_root, approach="token_fno_report")
    records = run_fixed_report_suite(
        output_root=bundle.root,
        train_docs=int(train_docs),
        scopes=scopes,
        root_shares=root_shares,
        leaf_tokens=leaf_tokens,
        seed=int(seed),
        include_duplicate_local_label_one_leaf=bool(include_duplicate_local_label_one_leaf),
        use_cuda=bool(use_cuda),
        cuda_device=cuda_device,
        torch_threads=int(torch_threads),
        reuse_existing=bool(reuse_existing),
        config_overrides=config_overrides,
    )
    summary = build_fixed_report_summary(
        records,
        output_root=bundle.root,
        train_doc_count=int(train_docs),
    )
    rendered = render_fixed_report(summary, output_root=bundle.root)
    bundle.write_manifest(
        config={
            "train_docs": int(train_docs),
            "scopes": [scope.value for scope in scopes],
            "root_shares": [int(value) for value in root_shares],
            "leaf_tokens": [int(value) for value in leaf_tokens],
            "seed": int(seed),
            "include_duplicate_local_label_one_leaf": bool(include_duplicate_local_label_one_leaf),
            "use_cuda": bool(use_cuda),
            "cuda_device": cuda_device,
            "torch_threads": int(torch_threads),
            "reuse_existing": bool(reuse_existing),
            "config_overrides": dict(config_overrides or {}),
        },
        result_paths={
            "run_manifest": str(bundle.root / "run_manifest.json"),
            "report_summary": str(bundle.root / "summary.json"),
            "figures": dict(rendered.get("figures") or {}),
        },
        extra={"run_count": len(records)},
    )
    return {
        "bundle_root": str(bundle.root),
        "run_manifest": str(bundle.root / "run_manifest.json"),
        "report_summary": str(bundle.root / "summary.json"),
        "bundle_manifest": str(bundle.manifest_path),
    }


def run_markov_fixed_report_bundle(
    output_root: str | Path,
    *,
    train_docs: int = 10240,
    scopes: Sequence[MarkovScope] = (
        MarkovScope.RECOVERABLE_V5_T128,
        MarkovScope.R12_P079,
    ),
    root_shares: Sequence[int] = (100, 90, 80, 70, 60, 50, 40, 30, 20, 10),
    leaf_tokens: Sequence[int] = (128, 64, 32, 16, 8),
    seed: int = 0,
    include_duplicate_local_label_one_leaf: bool = True,
    use_cuda: bool = True,
    cuda_device: int | None = None,
    torch_threads: int = 0,
    reuse_existing: bool = True,
    config_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return run_token_sequence_fno_report_bundle(
        output_root,
        train_docs=train_docs,
        scopes=scopes,
        root_shares=root_shares,
        leaf_tokens=leaf_tokens,
        seed=seed,
        include_duplicate_local_label_one_leaf=include_duplicate_local_label_one_leaf,
        use_cuda=use_cuda,
        cuda_device=cuda_device,
        torch_threads=torch_threads,
        reuse_existing=reuse_existing,
        config_overrides=config_overrides,
    )


def run_preference_data_bundle(
    output_root: str | Path,
    *,
    dataset: UnifiedGSupervisionDataset | None = None,
    supervision_path: str | Path | None = None,
    phase1_data_path: str | Path | None = None,
    split_ids_path: str | Path | None = None,
    law_type: str | None = None,
) -> dict[str, Any]:
    if dataset is None and supervision_path is None:
        raise ValueError("run_preference_data_bundle requires either dataset or supervision_path")
    bundle = create_stored_run_bundle(output_root, approach="preference_data")
    split_ids_json = ""
    if phase1_data_path is not None or split_ids_path is not None:
        split_ids_json, _split_payload = _resolve_and_save_split_ids(
            bundle=bundle,
            phase1_data_path=phase1_data_path,
            split_ids_path=split_ids_path,
        )
    resolved_dataset = (
        dataset
        if dataset is not None
        else UnifiedGSupervisionDataset.load(Path(supervision_path).expanduser())
    )
    saved_supervision = resolved_dataset.save(bundle.inputs_dir / "supervision_dataset.json")
    export_paths = export_supervision_formats(
        resolved_dataset,
        output_dir=bundle.exports_dir,
        law_type=law_type,
    )
    bundle.write_manifest(
        config={
            "law_type": law_type,
            "split_ids_path": str(split_ids_json),
        },
        result_paths={
            "split_ids_path": str(split_ids_json),
            "supervision_dataset": str(saved_supervision),
            "exports": export_paths,
        },
        extra={"status": "ready"},
    )
    return {
        "bundle_root": str(bundle.root),
        "split_ids_path": str(split_ids_json),
        "supervision_dataset": str(saved_supervision),
        "export_paths": export_paths,
        "bundle_manifest": str(bundle.manifest_path),
    }


def run_preference_optimization_bundle(
    output_root: str | Path,
    *,
    dataset: UnifiedGSupervisionDataset | None = None,
    supervision_path: str | Path | None = None,
    phase1_data_path: str | Path | None = None,
    split_ids_path: str | Path | None = None,
    law_type: str | None = None,
    train_mode: str = "none",
    model_name: str | None = None,
    trl_config: TRLTrainingConfig | None = None,
    reward_funcs: Any | None = None,
) -> dict[str, Any]:
    if dataset is None and supervision_path is None:
        raise ValueError("run_preference_optimization_bundle requires either dataset or supervision_path")
    bundle = create_stored_run_bundle(output_root, approach="preference_optimize")
    split_ids_json = ""
    if phase1_data_path is not None or split_ids_path is not None:
        split_ids_json, _split_payload = _resolve_and_save_split_ids(
            bundle=bundle,
            phase1_data_path=phase1_data_path,
            split_ids_path=split_ids_path,
        )
    resolved_dataset = (
        dataset
        if dataset is not None
        else UnifiedGSupervisionDataset.load(Path(supervision_path).expanduser())
    )
    saved_supervision = resolved_dataset.save(bundle.inputs_dir / "supervision_dataset.json")
    trl_config_path = None
    if trl_config is not None:
        trl_config_path = write_json(bundle.inputs_dir / "trl_config.json", trl_config.__dict__)
    export_paths = export_supervision_formats(
        resolved_dataset,
        output_dir=bundle.exports_dir,
        law_type=law_type,
    )
    train_output = None
    if train_mode != "none":
        if not model_name:
            raise ValueError("model_name is required when train_mode != 'none'")
        from treepo._research.unified_g_v1.training import TrainerConfig, fit

        fit_result = fit(
            trainer_config=TrainerConfig(
                mode=str(train_mode),
                model_name=str(model_name),
                supervision_dataset=resolved_dataset,
                trl_config=trl_config,
                law_type=law_type,
                reward_funcs=reward_funcs,
            ),
            output_dir=bundle.training_dir / str(train_mode),
        )
        train_output = fit_result.summary.get("train_output")
    bundle.write_manifest(
        config={
            "law_type": law_type,
            "train_mode": str(train_mode),
            "model_name": model_name,
            "trl_config_path": str(trl_config_path) if trl_config_path is not None else "",
            "split_ids_path": str(split_ids_json),
        },
        result_paths={
            "split_ids_path": str(split_ids_json),
            "supervision_dataset": str(saved_supervision),
            "trl_config": str(trl_config_path) if trl_config_path is not None else "",
            "exports": export_paths,
            "training_output": train_output,
        },
        extra={"train_mode": str(train_mode)},
    )
    return {
        "bundle_root": str(bundle.root),
        "supervision_dataset": str(saved_supervision),
        "export_paths": export_paths,
        "train_output": train_output,
        "bundle_manifest": str(bundle.manifest_path),
    }


def run_trl_bundle(
    output_root: str | Path,
    *,
    dataset: UnifiedGSupervisionDataset | None = None,
    supervision_path: str | Path | None = None,
    phase1_data_path: str | Path | None = None,
    split_ids_path: str | Path | None = None,
    law_type: str | None = None,
    train_mode: str = "none",
    model_name: str | None = None,
    trl_config: TRLTrainingConfig | None = None,
    reward_funcs: Any | None = None,
) -> dict[str, Any]:
    return run_preference_optimization_bundle(
        output_root,
        dataset=dataset,
        supervision_path=supervision_path,
        phase1_data_path=phase1_data_path,
        split_ids_path=split_ids_path,
        law_type=law_type,
        train_mode=train_mode,
        model_name=model_name,
        trl_config=trl_config,
        reward_funcs=reward_funcs,
    )


def build_openai_lawstress_bundle(
    output_root: str | Path,
    *,
    records_path: str | Path,
    artifact: UnifiedGArtifact | TextUnifiedGProgram | None = None,
    scorer_base_url: str,
    scorer_model: str,
    scorer_api_key: str = "EMPTY",
    scorer_timeout_seconds: float = 120.0,
    scorer_temperature: float = 0.0,
    scorer_max_tokens: int = 16,
    enable_thinking: bool = False,
    num_workers: int = 2,
) -> dict[str, Any]:
    client = OpenAIChatClient(
        base_url=scorer_base_url,
        model=scorer_model,
        api_key=scorer_api_key,
        timeout_seconds=float(scorer_timeout_seconds),
        enable_thinking=bool(enable_thinking),
    )
    return run_lawstress_eval_bundle(
        output_root,
        records=records_path,
        artifact=artifact,
        score_fn=build_numeric_score_fn(
            client,
            temperature=float(scorer_temperature),
            max_tokens=int(scorer_max_tokens),
        ),
        num_workers=int(num_workers),
        score_spec={
            "kind": "openai_numeric_score_fn",
            "scorer_base_url": str(scorer_base_url),
            "scorer_model": str(scorer_model),
            "scorer_timeout_seconds": float(scorer_timeout_seconds),
            "scorer_temperature": float(scorer_temperature),
            "scorer_max_tokens": int(scorer_max_tokens),
            "enable_thinking": bool(enable_thinking),
        },
    )


def build_openai_record_eval_bundle(
    output_root: str | Path,
    *,
    records_path: str | Path,
    artifact: UnifiedGArtifact | TextUnifiedGProgram | None = None,
    scorer_base_url: str,
    scorer_model: str,
    scorer_api_key: str = "EMPTY",
    scorer_timeout_seconds: float = 120.0,
    scorer_temperature: float = 0.0,
    scorer_max_tokens: int = 16,
    enable_thinking: bool = False,
    num_workers: int = 2,
) -> dict[str, Any]:
    return build_openai_lawstress_bundle(
        output_root,
        records_path=records_path,
        artifact=artifact,
        scorer_base_url=scorer_base_url,
        scorer_model=scorer_model,
        scorer_api_key=scorer_api_key,
        scorer_timeout_seconds=scorer_timeout_seconds,
        scorer_temperature=scorer_temperature,
        scorer_max_tokens=scorer_max_tokens,
        enable_thinking=enable_thinking,
        num_workers=num_workers,
    )
