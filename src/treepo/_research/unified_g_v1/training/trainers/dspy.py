"""DSPy trainer — subprocess orchestration of DSPy bootstrap optimization."""
from __future__ import annotations

from pathlib import Path

from treepo._research.unified_g_v1.training.trainers import register_trainer


def dspy_trainer(cfg, output_dir: Path, dataset=None):
    del dataset
    from treepo._research.unified_g_v1.training.backends.dspy_subprocess import run_dspy_subprocess
    from treepo._research.unified_g_v1.training.fit import FitResult

    if cfg.dspy_config is None:
        raise ValueError("dspy_trainer requires cfg.dspy_config (LawStressDSPyOptimizeConfig)")
    summary = run_dspy_subprocess(
        config=cfg.dspy_config,
        output_dir=Path(output_dir),
        execute=bool(cfg.dspy_execute),
        python_executable=cfg.dspy_python_executable,
        script_path=cfg.dspy_script_path,
        cwd=cfg.dspy_cwd,
    )
    status_raw = str(summary.get("status", "planned"))
    return FitResult(
        backend="dspy_subprocess",
        summary=summary,
        status=status_raw if status_raw in {"completed", "failed", "planned"} else "completed",
        artifacts={
            "artifact_path": str(summary.get("artifact_path", "")),
            "stdout_log": str(summary.get("stdout_log", "")),
            "stderr_log": str(summary.get("stderr_log", "")),
        },
    )


register_trainer("dspy", dspy_trainer)
