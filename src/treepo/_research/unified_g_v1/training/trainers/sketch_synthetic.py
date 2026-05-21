"""Sketch synthetic trainer — dispatches when `cfg.run_spec` is a SketchRunSpec."""
from __future__ import annotations

from pathlib import Path

from treepo._research.unified_g_v1.training.trainers import register_trainer


def sketch_synthetic_trainer(cfg, output_dir: Path, dataset=None):
    del dataset
    from treepo._research.unified_g_v1.training.backends.sketch_synthetic import run_sketch_synthetic
    from treepo._research.unified_g_v1.training.fit import FitResult

    if cfg.run_spec is None:
        raise ValueError("sketch_synthetic_trainer requires cfg.run_spec (SketchRunSpec)")
    summary = run_sketch_synthetic(
        run_spec=cfg.run_spec,
        output_dir=Path(output_dir),
        reuse_existing=bool(cfg.reuse_existing),
    )
    return FitResult(
        backend="sketch_synthetic",
        summary=summary,
        status="completed",
        metrics={
            "final_train_mae": float(summary.get("final_train_mae", 0.0)),
            "final_val_mae": float(summary.get("final_val_mae", 0.0)),
            "merge_recon_mse": float(summary.get("merge_recon_mse", 0.0)),
        },
        artifacts={
            "run_dir": str(summary.get("run_dir", "")),
            "summary_path": str(summary.get("summary_path", "")),
        },
        history=list(summary.get("history") or []),
    )


register_trainer("sketch_synthetic", sketch_synthetic_trainer)
