"""Markov synthetic trainer — dispatches when `cfg.run_spec` is a MarkovRunSpec."""
from __future__ import annotations

from pathlib import Path

from treepo._research.unified_g_v1.training.trainers import register_trainer


def markov_synthetic_trainer(cfg, output_dir: Path, dataset=None):
    del dataset
    from treepo._research.unified_g_v1.training.backends.markov_synthetic import run_markov_synthetic
    from treepo._research.unified_g_v1.training.fit import FitResult

    if cfg.run_spec is None:
        raise ValueError("markov_synthetic_trainer requires cfg.run_spec (MarkovRunSpec)")
    summary = run_markov_synthetic(
        run_spec=cfg.run_spec,
        output_dir=Path(output_dir),
        use_cuda=bool(cfg.use_cuda),
        cuda_device=cfg.cuda_device,
        torch_threads=int(cfg.torch_threads),
        reuse_existing=bool(cfg.reuse_existing),
        config_overrides=cfg.config_overrides,
    )
    metrics_raw = summary.get("extracted_metrics") or {}
    metrics: dict[str, float] = {}
    for key, value in metrics_raw.items():
        try:
            metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return FitResult(
        backend="markov_synthetic",
        summary=summary,
        status="completed",
        metrics=metrics,
        artifacts={
            "run_dir": str(summary.get("run_dir", "")),
            "summary_path": str(summary.get("summary_path", "")),
        },
    )


register_trainer("markov_synthetic", markov_synthetic_trainer)
