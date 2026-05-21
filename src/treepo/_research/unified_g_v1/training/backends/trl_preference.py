from __future__ import annotations

from pathlib import Path
from typing import Any

from treepo._research.unified_g_v1.core.supervision import UnifiedGSupervisionDataset

from treepo._research.training.trl_training import TRLTrainingConfig


_SUPPORTED_MODES = frozenset({"dpo", "grpo", "reward_model", "scalar_reward"})


def run_trl_preference(
    *,
    dataset: UnifiedGSupervisionDataset,
    mode: str,
    model_name: str,
    output_dir: Path,
    trl_config: TRLTrainingConfig | None = None,
    law_type: str | None = None,
    reward_funcs: Any | None = None,
) -> dict[str, Any]:
    """Dispatch to the appropriate TRL preference trainer on `UnifiedGSupervisionDataset`.

    The underlying `train_*` methods on `UnifiedGSupervisionDataset` already
    wrap `src/training/trl_training.py`; this just gates `mode` and threads
    the per-mode kwargs.
    """
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"unsupported preference mode={mode!r}")
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "dpo":
        train_output = dataset.train_dpo(
            model_name=model_name,
            output_dir=output_dir,
            config=trl_config,
            law_type=law_type,
        )
    elif mode == "grpo":
        if reward_funcs is None:
            raise ValueError("reward_funcs is required for train_mode='grpo'")
        train_output = dataset.train_grpo(
            model_name=model_name,
            output_dir=output_dir,
            config=trl_config,
            law_type=law_type,
            reward_funcs=reward_funcs,
        )
    elif mode == "reward_model":
        train_output = dataset.train_reward_model(
            model_name=model_name,
            output_dir=output_dir,
            config=trl_config,
            law_type=law_type,
        )
    else:
        train_output = dataset.train_scalar_reward_model(
            model_name=model_name,
            output_dir=output_dir,
            config=trl_config,
            law_type=law_type,
        )
    return {
        "backend": "trl_preference",
        "mode": str(mode),
        "model_name": str(model_name),
        "law_type": law_type,
        "train_output": train_output,
        "output_dir": str(output_dir),
    }
