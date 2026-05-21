"""Explicit tree-GRPO trainer surface.

The actual online optimization still flows through the generic TRL preference
backend. This wrapper exists so the tree lane has an honest, named registry
surface and so callers get a clear contract error unless they provide a real
tree-aware reward function adapter.
"""
from __future__ import annotations

from pathlib import Path

from treepo._research.unified_g_v1.training.trainers import register_trainer


def grpo_tree_trainer(cfg, output_dir: Path, dataset=None):
    del dataset
    from treepo._research.unified_g_v1.training.fit import FitResult
    from treepo._research.unified_g_v1.training.trainers.trl_preference import trl_preference_trainer

    if str(getattr(cfg, "mode", "") or "").lower() != "grpo":
        raise ValueError("grpo_tree_trainer requires cfg.mode='grpo'")
    if cfg.reward_funcs is None:
        raise ValueError(
            "grpo_tree_trainer requires cfg.reward_funcs. "
            "Provide a tree-aware completion-to-rollout reward adapter; there is no "
            "default structured completion contract for tree GRPO in this repo."
        )

    result = trl_preference_trainer(cfg, output_dir, dataset=None)
    return FitResult(
        backend="grpo_tree",
        summary={**dict(result.summary), "backend": "grpo_tree"},
        status=result.status,
        metrics=dict(result.metrics),
        artifacts=dict(result.artifacts),
        history=tuple(result.history),
    )


register_trainer("grpo_tree", grpo_tree_trainer)
