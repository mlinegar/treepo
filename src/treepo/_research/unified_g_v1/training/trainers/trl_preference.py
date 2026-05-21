"""TRL preference trainer — DPO/GRPO/reward-model/scalar-reward."""
from __future__ import annotations

from pathlib import Path

from treepo._research.unified_g_v1.training.trainers import register_trainer


def trl_preference_trainer(cfg, output_dir: Path, dataset=None):
    del dataset
    from treepo._research.unified_g_v1.training.backends.trl_preference import run_trl_preference
    from treepo._research.unified_g_v1.training.fit import FitResult

    if cfg.supervision_dataset is None:
        raise ValueError("trl_preference_trainer requires cfg.supervision_dataset")
    if not cfg.model_name:
        raise ValueError("trl_preference_trainer requires cfg.model_name")
    if not cfg.mode:
        raise ValueError("trl_preference_trainer requires cfg.mode")
    summary = run_trl_preference(
        dataset=cfg.supervision_dataset,
        mode=str(cfg.mode),
        model_name=str(cfg.model_name),
        output_dir=Path(output_dir),
        trl_config=cfg.trl_config,
        law_type=cfg.law_type,
        reward_funcs=cfg.reward_funcs,
    )
    return FitResult(
        backend="trl_preference",
        summary=summary,
        status="completed",
        artifacts={"output_dir": str(output_dir)},
    )


register_trainer("trl_preference", trl_preference_trainer)
