"""Thin helper to collapse the bundle bookkeeping boilerplate.

Every training bundle in `bundles.py` does the same dance:
  1. create_stored_run_bundle(output_root, approach=...)
  2. write config JSON into inputs_dir
  3. call fit() or a training function
  4. write summary JSON into results_dir
  5. bundle.write_manifest(config=..., result_paths=..., extra=...)
  6. return {bundle_root, summary, ...}

`run_training_bundle` does steps 1, 3, 4, 5, 6 in one call. Callers supply
inputs to persist and optional extra manifest fields.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from treepo._research.unified_g_v1.core.bundle import StoredRunBundle, create_stored_run_bundle
from treepo._research.unified_g_v1.core.manifest import write_json
from treepo._research.unified_g_v1.core.program import UnifiedFGProgram
from treepo._research.unified_g_v1.training.fit import FitResult, TrainerConfig, fit
from treepo._research.unified_g_v1.training.prepared_dataset import PreparedDataset


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _sanitize_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(k): _json_safe(v) for k, v in dict(value or {}).items()}


def run_training_bundle(
    output_root: str | Path,
    *,
    approach: str,
    trainer_config: TrainerConfig,
    dataset: PreparedDataset | None = None,
    program: UnifiedFGProgram | None = None,
    training_subdir: str | None = None,
    inputs: Mapping[str, Any] | None = None,
    manifest_extra: Mapping[str, Any] | None = None,
    program_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a StoredRunBundle, call `fit()`, persist manifest, return bundle dict.

    `inputs` is written as `inputs/<approach>_config.json`. `fit()`'s
    `FitResult.summary` is sanitized (non-JSON fields coerced to strings) and
    written as `results/<approach>_summary.json`.
    """
    bundle: StoredRunBundle = create_stored_run_bundle(output_root, approach=approach)
    inputs_payload = dict(inputs or {})
    config_path = write_json(
        bundle.inputs_dir / f"{approach}_config.json",
        _sanitize_mapping(inputs_payload),
    )
    training_output_root = bundle.training_dir / (training_subdir or approach)
    training_output_root.mkdir(parents=True, exist_ok=True)
    del program  # legacy arg retained in helper signature; fit() no longer accepts it
    result: FitResult = fit(
        trainer_config=trainer_config,
        dataset=dataset,
        output_dir=training_output_root,
    )
    summary_payload = {
        "backend": result.backend,
        "artifacts": {k: str(v) for k, v in result.artifacts.items()},
        **_sanitize_mapping(result.summary),
    }
    summary_path = write_json(
        bundle.results_dir / f"{approach}_summary.json",
        summary_payload,
    )
    manifest_result_paths: dict[str, str] = {
        "config": str(config_path),
        "summary": str(summary_path),
        "training_output_root": str(training_output_root),
    }
    for artifact_name, artifact_path in result.artifacts.items():
        manifest_result_paths[f"artifact_{artifact_name}"] = str(artifact_path)
    bundle.write_manifest(
        config=_sanitize_mapping(inputs_payload),
        result_paths=manifest_result_paths,
        program_contract=dict(program_contract or {}),
        extra={"backend": result.backend, **_sanitize_mapping(manifest_extra)},
    )
    return {
        "bundle_root": str(bundle.root),
        "approach": str(approach),
        "summary": str(summary_path),
        "training_output_root": str(training_output_root),
        "bundle_manifest": str(bundle.manifest_path),
        "backend": result.backend,
        "status": str(result.status),
        "metrics": dict(result.metrics),
    }
