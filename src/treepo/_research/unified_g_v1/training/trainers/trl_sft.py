"""TRL SFT trainer — default for `space_kind == "text"`.

Wraps `run_trl_sft` from `training/backends/trl_sft.py`. Reads `model_name`
and `trl_config` from the `TreeTaskConfig`; requires either a `PreparedDataset`
passed into `fit()` or an oracle that can materialize one.
"""
from __future__ import annotations

from pathlib import Path

from treepo._research.unified_g_v1.training.trainers import register_trainer


def trl_sft_trainer(cfg, output_dir: Path, dataset=None):
    from treepo._research.unified_g_v1.training.backends.trl_sft import run_trl_sft
    from treepo._research.unified_g_v1.training.fit import FitResult

    if not cfg.model_name:
        raise ValueError("trl_sft_trainer requires cfg.model_name")

    resolved_dataset = dataset
    if resolved_dataset is None:
        # Oracle may expose a `.prepared_dataset` attribute for the text lane.
        resolved_dataset = getattr(cfg.oracle, "prepared_dataset", None)
    if resolved_dataset is None:
        raise ValueError(
            "trl_sft_trainer requires a PreparedDataset (pass via fit(dataset=...) "
            "or have the oracle expose .prepared_dataset)"
        )

    summary = run_trl_sft(
        dataset=resolved_dataset,
        model_name=str(cfg.model_name),
        output_dir=Path(output_dir),
        trl_config=cfg.trl_config,
    )
    metrics_raw = summary.get("metrics") or {}
    metrics: dict[str, float] = {}
    for key, value in metrics_raw.items():
        try:
            metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return FitResult(
        backend="trl_sft",
        summary=summary,
        status="completed",
        metrics=metrics,
        artifacts={"output_dir": str(output_dir)},
    )


register_trainer("trl_sft", trl_sft_trainer)
