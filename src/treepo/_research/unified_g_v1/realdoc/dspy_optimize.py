from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_BOOTSTRAP_SCRIPT = REPO_ROOT / "scripts" / "bootstrap_lawstress_summarizer.py"


@dataclass(frozen=True)
class LawStressDSPyOptimizeConfig:
    records_path: str | Path
    student_port: int = 8000
    student_model: str | None = None
    student_temperature: float = 0.2
    student_max_tokens: int = 0
    enable_thinking: bool = False
    gepa_reflection_model: str | None = None
    gepa_reflection_temperature: float = 0.0
    gepa_reflection_max_tokens: int = 0
    embedding_url: str = "http://localhost:8003/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    embedding_api_key: str = "EMPTY"
    embedding_timeout_seconds: float = 60.0
    embedding_batch_size: int = 32
    proxy_path: str | Path | None = None
    ridge_lambda: float = 1.0
    proxy_model_id: str = "lawstress_embedding_ridge_proxy_v1"
    gepa_budget: str = "light"
    num_threads: int = 8
    gepa_max_metric_calls: int = 0
    gepa_max_full_evals: int = 0
    seed: int = 0
    c1_threshold_norm: float = 0.10
    c2_threshold_norm: float = 0.06
    c3_threshold_norm: float = 0.08
    objective_aggregate: str = "min"
    objective_softmin_temperature: float = 0.08
    objective_component_floor: float = 0.55
    verbose: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_path": str(Path(self.records_path).expanduser()),
            "student_port": int(self.student_port),
            "student_model": self.student_model,
            "student_temperature": float(self.student_temperature),
            "student_max_tokens": int(self.student_max_tokens),
            "enable_thinking": bool(self.enable_thinking),
            "gepa_reflection_model": self.gepa_reflection_model,
            "gepa_reflection_temperature": float(self.gepa_reflection_temperature),
            "gepa_reflection_max_tokens": int(self.gepa_reflection_max_tokens),
            "embedding_url": str(self.embedding_url),
            "embedding_model": str(self.embedding_model),
            "embedding_api_key": str(self.embedding_api_key),
            "embedding_timeout_seconds": float(self.embedding_timeout_seconds),
            "embedding_batch_size": int(self.embedding_batch_size),
            "proxy_path": (
                str(Path(self.proxy_path).expanduser()) if self.proxy_path is not None else None
            ),
            "ridge_lambda": float(self.ridge_lambda),
            "proxy_model_id": str(self.proxy_model_id),
            "gepa_budget": str(self.gepa_budget),
            "num_threads": int(self.num_threads),
            "gepa_max_metric_calls": int(self.gepa_max_metric_calls),
            "gepa_max_full_evals": int(self.gepa_max_full_evals),
            "seed": int(self.seed),
            "c1_threshold_norm": float(self.c1_threshold_norm),
            "c2_threshold_norm": float(self.c2_threshold_norm),
            "c3_threshold_norm": float(self.c3_threshold_norm),
            "objective_aggregate": str(self.objective_aggregate),
            "objective_softmin_temperature": float(self.objective_softmin_temperature),
            "objective_component_floor": float(self.objective_component_floor),
            "verbose": bool(self.verbose),
        }


def build_lawstress_dspy_command(
    config: LawStressDSPyOptimizeConfig,
    *,
    output_dir: str | Path,
    python_executable: str | Path | None = None,
    script_path: str | Path | None = None,
) -> list[str]:
    resolved_output_dir = Path(output_dir).expanduser()
    resolved_script = Path(script_path).expanduser() if script_path is not None else DEFAULT_BOOTSTRAP_SCRIPT
    resolved_python = str(python_executable or sys.executable)
    command = [
        resolved_python,
        str(resolved_script),
        "--records",
        str(Path(config.records_path).expanduser()),
        "--output-dir",
        str(resolved_output_dir),
        "--student-port",
        str(int(config.student_port)),
        "--student-temperature",
        str(float(config.student_temperature)),
        "--student-max-tokens",
        str(int(config.student_max_tokens)),
        "--gepa-reflection-temperature",
        str(float(config.gepa_reflection_temperature)),
        "--gepa-reflection-max-tokens",
        str(int(config.gepa_reflection_max_tokens)),
        "--embedding-url",
        str(config.embedding_url),
        "--embedding-model",
        str(config.embedding_model),
        "--embedding-api-key",
        str(config.embedding_api_key),
        "--embedding-timeout-seconds",
        str(float(config.embedding_timeout_seconds)),
        "--embedding-batch-size",
        str(int(config.embedding_batch_size)),
        "--ridge-lambda",
        str(float(config.ridge_lambda)),
        "--proxy-model-id",
        str(config.proxy_model_id),
        "--gepa-budget",
        str(config.gepa_budget),
        "--num-threads",
        str(int(config.num_threads)),
        "--gepa-max-metric-calls",
        str(int(config.gepa_max_metric_calls)),
        "--gepa-max-full-evals",
        str(int(config.gepa_max_full_evals)),
        "--seed",
        str(int(config.seed)),
        "--c1-threshold-norm",
        str(float(config.c1_threshold_norm)),
        "--c2-threshold-norm",
        str(float(config.c2_threshold_norm)),
        "--c3-threshold-norm",
        str(float(config.c3_threshold_norm)),
        "--objective-aggregate",
        str(config.objective_aggregate),
        "--objective-softmin-temperature",
        str(float(config.objective_softmin_temperature)),
        "--objective-component-floor",
        str(float(config.objective_component_floor)),
    ]
    if config.student_model:
        command.extend(["--student-model", str(config.student_model)])
    if config.enable_thinking:
        command.append("--enable-thinking")
    if config.gepa_reflection_model:
        command.extend(["--gepa-reflection-model", str(config.gepa_reflection_model)])
    if config.proxy_path is not None:
        command.extend(["--proxy-path", str(Path(config.proxy_path).expanduser())])
    if config.verbose:
        command.append("--verbose")
    return command


def run_lawstress_dspy_optimization(
    config: LawStressDSPyOptimizeConfig,
    *,
    output_dir: str | Path,
    execute: bool = True,
    python_executable: str | Path | None = None,
    script_path: str | Path | None = None,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    resolved_output_dir = Path(output_dir).expanduser()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_cwd = Path(cwd).expanduser() if cwd is not None else REPO_ROOT
    command = build_lawstress_dspy_command(
        config,
        output_dir=resolved_output_dir,
        python_executable=python_executable,
        script_path=script_path,
    )
    stdout_path = resolved_output_dir / "bootstrap_stdout.log"
    stderr_path = resolved_output_dir / "bootstrap_stderr.log"
    artifact_path = resolved_output_dir / "trained_modules" / "unified_g_final.json"
    bootstrap_stats_path = resolved_output_dir / "bootstrap_stats.json"

    payload: dict[str, Any] = {
        "status": "planned",
        "command": list(command),
        "command_pretty": shlex.join(command),
        "cwd": str(resolved_cwd),
        "output_dir": str(resolved_output_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "artifact_path": str(artifact_path),
        "bootstrap_stats_path": str(bootstrap_stats_path),
        "config": config.to_dict(),
        "executed": bool(execute),
        "returncode": None,
        "bootstrap_stats": None,
    }
    if not execute:
        return payload

    completed = subprocess.run(
        command,
        cwd=str(resolved_cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    payload["returncode"] = int(completed.returncode)
    payload["status"] = "completed" if completed.returncode == 0 else "failed"
    if bootstrap_stats_path.exists():
        payload["bootstrap_stats"] = json.loads(bootstrap_stats_path.read_text(encoding="utf-8"))
    return payload
