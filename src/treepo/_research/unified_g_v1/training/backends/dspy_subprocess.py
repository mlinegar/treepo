from __future__ import annotations

from pathlib import Path
from typing import Any

from treepo._research.unified_g_v1.realdoc.dspy_optimize import (
    LawStressDSPyOptimizeConfig,
    run_lawstress_dspy_optimization,
)


def run_dspy_subprocess(
    *,
    config: LawStressDSPyOptimizeConfig,
    output_dir: Path,
    execute: bool = True,
    python_executable: str | Path | None = None,
    script_path: str | Path | None = None,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Thin wrapper around `run_lawstress_dspy_optimization`.

    The DSPy pipeline keeps its subprocess seam. We normalize the payload
    keys so downstream code can treat PyTorch, TRL, and DSPy results uniformly.
    """
    payload = run_lawstress_dspy_optimization(
        config,
        output_dir=output_dir,
        execute=bool(execute),
        python_executable=python_executable,
        script_path=script_path,
        cwd=cwd,
    )
    return {
        "backend": "dspy_subprocess",
        "status": str(payload.get("status", "planned")),
        "executed": bool(payload.get("executed", False)),
        "returncode": payload.get("returncode"),
        "artifact_path": str(payload.get("artifact_path", "")),
        "stdout_log": str(payload.get("stdout_log", "")),
        "stderr_log": str(payload.get("stderr_log", "")),
        "bootstrap_stats_path": str(payload.get("bootstrap_stats_path", "")),
        "bootstrap_stats": payload.get("bootstrap_stats"),
        "command": list(payload.get("command") or []),
        "command_pretty": str(payload.get("command_pretty", "")),
        "raw_payload": payload,
    }
