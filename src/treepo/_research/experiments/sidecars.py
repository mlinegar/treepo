from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Any, Mapping, Sequence

from treepo._research.experiments.contracts import (
    ArtifactRef,
    BenchmarkRef,
    ExperimentSpec,
    MethodRef,
    PhaseSpec,
    ProgressSnapshot,
    ResultRow,
    benchmark_ref_from_parts,
    default_phase_specs,
)
from treepo._research.experiments.control_plane import (
    RESULTS_NAME,
    append_result_rows,
    canonical_artifact_refs_from_paths,
    merge_artifacts,
    write_experiment_manifest,
    write_experiment_status,
)


def sidecar_root_for_output_file(path: str | Path) -> Path:
    output_path = Path(path).expanduser().resolve()
    return output_path.parent / f"{output_path.stem}_experiment"


def output_sidecar_root(path_or_dir: str | Path, *, file_output: bool = False) -> Path:
    path = Path(path_or_dir).expanduser().resolve()
    return sidecar_root_for_output_file(path) if file_output else path


def write_canonical_sidecars(
    output_root: str | Path,
    *,
    title: str,
    adapter_id: str,
    benchmark_refs: Sequence[BenchmarkRef] = (),
    method_refs: Sequence[MethodRef] = (),
    phases: Sequence[PhaseSpec] | Sequence[str] = (),
    artifacts: Sequence[ArtifactRef] | Mapping[str, Any] = (),
    result_rows: Sequence[ResultRow] | Sequence[Mapping[str, Any]] = (),
    state: str = "completed",
    metadata: Mapping[str, Any] | None = None,
    launch_command: Sequence[str] = (),
    report_profiles: Sequence[str] = (),
    replace_results: bool = True,
) -> ExperimentSpec:
    root = Path(output_root).expanduser().resolve()
    phase_specs: Sequence[PhaseSpec]
    if phases and all(isinstance(item, str) for item in phases):  # type: ignore[arg-type]
        phase_specs = default_phase_specs(str(item) for item in phases)  # type: ignore[arg-type]
    else:
        phase_specs = phases  # type: ignore[assignment]

    artifact_refs: Sequence[ArtifactRef]
    artifact_mapping: Mapping[str, Any] | None = None
    if isinstance(artifacts, Mapping):
        artifact_mapping = artifacts
        artifact_refs = tuple(
            canonical_artifact_refs_from_paths(artifact_mapping, required=False)
        )
    else:
        artifact_refs = tuple(artifacts)

    spec = ExperimentSpec.create(
        adapter_id=str(adapter_id),
        output_root=str(root),
        title=str(title),
        benchmark_refs=tuple(benchmark_refs),
        method_refs=tuple(method_refs),
        phases=tuple(phase_specs),
        artifacts=tuple(artifact_refs),
        report_profiles=tuple(str(item) for item in report_profiles),
        launch_command=tuple(str(item) for item in launch_command),
        resume_command=tuple(str(item) for item in launch_command),
        metadata=dict(metadata or {}),
    )
    write_experiment_manifest(root, spec)
    write_experiment_status(
        root,
        ProgressSnapshot(
            experiment_id=spec.experiment_id,
            state=str(state),
            active_phase=tuple(phase_specs)[-1].phase_id if phase_specs else "",
            items_total=len(result_rows),
            completed_items=len(result_rows) if str(state) in {"completed", "dry_run"} else 0,
            failed_items=0,
            percent_complete=100.0 if str(state) in {"completed", "dry_run"} else 0.0,
            artifact_targets=tuple(item.artifact_id for item in artifact_refs),
            metadata=dict(metadata or {}),
        ),
    )
    if artifact_mapping is not None:
        merge_artifacts(root, artifact_mapping)
    else:
        merge_artifacts(root, artifact_refs)
    results_path = root / RESULTS_NAME
    normalized_rows = []
    for row in result_rows:
        if isinstance(row, ResultRow):
            normalized_rows.append(
                replace(row, experiment_id=spec.experiment_id)
                if not row.experiment_id
                else row
            )
        else:
            payload = dict(row)
            payload.setdefault("experiment_id", spec.experiment_id)
            normalized_rows.append(payload)
    if replace_results and results_path.exists():
        results_path.unlink()
    append_result_rows(root, normalized_rows)
    return spec


def simple_benchmark_ref(
    *,
    family: str,
    name: str = "",
    scope: str = "",
    dataset_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> BenchmarkRef:
    return benchmark_ref_from_parts(
        family=family,
        name=name or family,
        scope=scope,
        dataset_id=dataset_id,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "output_sidecar_root",
    "sidecar_root_for_output_file",
    "simple_benchmark_ref",
    "write_canonical_sidecars",
]
