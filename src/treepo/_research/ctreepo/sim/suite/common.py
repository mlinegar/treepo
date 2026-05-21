from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from treepo._research.ctreepo.sim.manifest import RunSpec, write_manifest_jsonl
from treepo._research.ctreepo.sim.resource_queue import detect_gpu_tokens, load_jobs, run_resource_queue


def resolve_output_root(
    *,
    run_id: str,
    output_root: str,
    default_prefix: str,
) -> Path:
    """Resolve output root, falling back to ``outputs/{default_prefix}_{run_id}``."""
    if str(output_root).strip():
        return Path(output_root)
    return Path(f"outputs/{default_prefix}_{str(run_id).strip()}")


def resolve_figures_root(*, figures_root: str, output_root: Path) -> Path:
    """Resolve figures root, falling back to ``output_root/figures``."""
    if str(figures_root).strip():
        return Path(figures_root)
    return output_root / "figures"


def utc_run_id(default: str | None = None) -> str:
    if default and str(default).strip():
        return str(default).strip()
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_items(text: str) -> List[str]:
    items: List[str] = []
    for raw in str(text).replace(",", " ").split():
        item = raw.strip()
        if item:
            items.append(item)
    return items


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def thread_env_vars() -> Dict[str, str]:
    return {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
    }


def thread_env_prefix() -> str:
    return " ".join(f"{key}={value}" for key, value in thread_env_vars().items())


def apply_thread_env() -> None:
    os.environ.update(thread_env_vars())


@dataclass(frozen=True)
class GroupedSuitePaths:
    output_root: Path
    suite_meta: Path
    suite_cmds: Path
    suite_manifest: Path
    group_cmd_dir: Path
    group_manifest_dir: Path
    queue_log_dir: Path


@dataclass(frozen=True)
class SuiteGroupRuns:
    key: str
    family: str
    runs: List[RunSpec]


@dataclass(frozen=True)
class GroupedSuiteArtifacts:
    group_cmd_files: Dict[str, str]
    group_manifest_files: Dict[str, str]
    group_families: Dict[str, str]
    counts_by_group: Dict[str, int]
    all_runs: List[RunSpec]
    all_cmds: List[str]


def resolve_grouped_suite_paths(
    output_root: Path,
    *,
    group_dir_name: str = "suite_groups",
) -> GroupedSuitePaths:
    return GroupedSuitePaths(
        output_root=output_root,
        suite_meta=output_root / "suite_meta.json",
        suite_cmds=output_root / "suite_cmds.txt",
        suite_manifest=output_root / "suite_manifest.jsonl",
        group_cmd_dir=output_root / group_dir_name / "cmds",
        group_manifest_dir=output_root / group_dir_name / "manifests",
        queue_log_dir=output_root / "queue_logs",
    )


def emit_grouped_suite_artifacts(
    paths: GroupedSuitePaths,
    groups: Sequence[SuiteGroupRuns],
) -> GroupedSuiteArtifacts:
    paths.output_root.mkdir(parents=True, exist_ok=True)
    paths.group_cmd_dir.mkdir(parents=True, exist_ok=True)
    paths.group_manifest_dir.mkdir(parents=True, exist_ok=True)

    group_cmd_files: Dict[str, str] = {}
    group_manifest_files: Dict[str, str] = {}
    group_families: Dict[str, str] = {}
    counts_by_group: Dict[str, int] = {}
    all_runs: List[RunSpec] = []
    all_cmds: List[str] = []

    for group in groups:
        cmd_path = paths.group_cmd_dir / f"{group.key}.txt"
        manifest_path = paths.group_manifest_dir / f"{group.key}.jsonl"
        cmds = [run.command for run in group.runs]
        write_text(cmd_path, "\n".join(cmds) + ("\n" if cmds else ""))
        write_manifest_jsonl(manifest_path, group.runs)
        group_cmd_files[group.key] = str(cmd_path)
        group_manifest_files[group.key] = str(manifest_path)
        group_families[group.key] = str(group.family)
        counts_by_group[group.key] = int(len(group.runs))
        all_runs.extend(group.runs)
        all_cmds.extend(cmds)

    write_text(paths.suite_cmds, "\n".join(all_cmds) + ("\n" if all_cmds else ""))
    write_manifest_jsonl(paths.suite_manifest, all_runs)
    return GroupedSuiteArtifacts(
        group_cmd_files=group_cmd_files,
        group_manifest_files=group_manifest_files,
        group_families=group_families,
        counts_by_group=counts_by_group,
        all_runs=all_runs,
        all_cmds=all_cmds,
    )


def read_suite_meta(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def runs_from_commands(
    *,
    commands: Sequence[str],
    family: str,
    group_key: str,
    base_config: Mapping[str, Any] | None = None,
    base_outputs: Mapping[str, str] | None = None,
    requires: Sequence[str] | None = None,
) -> List[RunSpec]:
    runs: List[RunSpec] = []
    config_base = dict(base_config or {})
    outputs_base = {str(k): str(v) for k, v in dict(base_outputs or {}).items()}
    reqs = [str(x) for x in (requires or [])]
    for idx, command in enumerate(commands):
        config = dict(config_base)
        config.setdefault("suite_group", str(group_key))
        config.setdefault("command_index", int(idx))
        runs.append(
            RunSpec.create(
                family=str(family),
                config=config,
                outputs=dict(outputs_base),
                command=str(command),
                requires=reqs,
            )
        )
    return runs


def build_suite_meta(
    *,
    suite_name: str,
    suite_role: str,
    run_id: str,
    output_root: Path,
    python_bin: str,
    cmds_file: Path,
    manifest_file: Path,
    selected_groups: Sequence[str],
    group_cmd_files: Mapping[str, str],
    group_manifest_files: Mapping[str, str],
    profile: str = "",
    policy: Mapping[str, Any] | None = None,
    group_families: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": "v2",
        "suite_name": str(suite_name),
        "suite_role": str(suite_role),
        "run_id": str(run_id),
        "profile": str(profile),
        "policy": dict(policy or {}),
        "python_bin": str(python_bin),
        "output_root": str(output_root),
        "cmds_file": str(cmds_file),
        "manifest_file": str(manifest_file),
        "selected_groups": [str(x) for x in selected_groups if str(x).strip()],
        "group_cmd_files": {str(k): str(v) for k, v in dict(group_cmd_files).items()},
        "group_manifest_files": {str(k): str(v) for k, v in dict(group_manifest_files).items()},
        "group_families": {str(k): str(v) for k, v in dict(group_families or {}).items()},
    }
    if extra:
        payload.update(dict(extra))
    return payload


def write_suite_meta(path: Path, payload: Mapping[str, Any]) -> None:
    write_text(path, json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def select_known_items(
    *,
    requested: Sequence[str],
    available: Sequence[str],
    item_name: str,
) -> List[str]:
    available_list = [str(x) for x in available if str(x).strip()]
    if not requested:
        return list(available_list)
    requested_list = [str(x) for x in requested if str(x).strip()]
    unknown = sorted(set(requested_list) - set(available_list))
    if unknown:
        raise ValueError(f"unknown {item_name}: {', '.join(unknown)}")
    return [key for key in available_list if key in requested_list]


def run_manifest_queue_suite(
    *,
    manifest_paths: Sequence[Path],
    cpu_workers: int,
    gpu_tokens: str,
    log_dir: Path,
    set_thread_env: bool = True,
) -> Dict[str, Any]:
    if bool(set_thread_env):
        apply_thread_env()
    jobs = load_jobs(manifest_paths=list(manifest_paths), cmd_files=[])
    resolved_gpu_tokens = detect_gpu_tokens(str(gpu_tokens))
    summary = run_resource_queue(
        jobs,
        cpu_workers=int(cpu_workers),
        gpu_tokens=resolved_gpu_tokens,
        log_dir=log_dir,
    )
    return {
        "manifest_paths": [str(path) for path in manifest_paths],
        "log_dir": str(log_dir),
        "gpu_tokens": list(resolved_gpu_tokens),
        "summary": summary,
    }


__all__ = [
    "GroupedSuiteArtifacts",
    "GroupedSuitePaths",
    "SuiteGroupRuns",
    "apply_thread_env",
    "build_suite_meta",
    "emit_grouped_suite_artifacts",
    "parse_items",
    "read_suite_meta",
    "resolve_grouped_suite_paths",
    "runs_from_commands",
    "run_manifest_queue_suite",
    "select_known_items",
    "thread_env_prefix",
    "thread_env_vars",
    "utc_run_id",
    "write_suite_meta",
    "write_text",
]
