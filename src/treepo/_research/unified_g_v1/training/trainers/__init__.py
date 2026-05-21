"""Trainers are plain functions.

A `Trainer` is any callable `(cfg, output_dir, dataset) -> FitResult`. Built-in
trainers live here; users can register their own via `TRAINER_REGISTRY` or
pass a callable directly as `TrainerConfig.trainer`.

Default selection is driven by which fields of the config are populated. See
`resolve_trainer` for the precedence.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

if TYPE_CHECKING:
    from treepo._research.unified_g_v1.training.fit import FitResult
    from treepo._research.unified_g_v1.training.prepared_dataset import PreparedDataset
    from treepo._research.unified_g_v1.training.tree_task import TrainerConfig


Trainer = Callable[
    ["TrainerConfig", Path, Optional["PreparedDataset"]],
    "FitResult",
]


TRAINER_REGISTRY: dict[str, Trainer] = {}


def register_trainer(name: str, trainer: Trainer) -> Trainer:
    TRAINER_REGISTRY[str(name)] = trainer
    return trainer


def resolve_trainer(cfg: "TrainerConfig") -> Trainer:
    """Pick the trainer for a `TrainerConfig`.

    Precedence:
      1. `cfg.trainer` is callable              -> use it directly.
      2. `cfg.trainer` is a string              -> look up in TRAINER_REGISTRY.
      3. `cfg.run_spec` is a MarkovRunSpec      -> `markov_synthetic`.
      4. `cfg.run_spec` is a SketchRunSpec      -> `sketch_synthetic`.
      5. `cfg.mode` is a preference mode        -> `trl_preference`.
      6. `cfg.dspy_config` present              -> `dspy`.
      7. `cfg.oracle.space_kind == "text"`      -> `trl_sft`.
      8. otherwise                              -> `pytorch_tree`.
    """
    _ensure_builtins_registered()

    candidate = getattr(cfg, "trainer", None)
    if callable(candidate):
        return candidate
    if isinstance(candidate, str):
        if candidate not in TRAINER_REGISTRY:
            raise KeyError(
                f"unknown trainer {candidate!r}; registered: {sorted(TRAINER_REGISTRY)}"
            )
        return TRAINER_REGISTRY[candidate]

    # Field-driven dispatch.
    run_spec = getattr(cfg, "run_spec", None)
    if run_spec is not None:
        spec_name = type(run_spec).__name__
        if spec_name == "MarkovRunSpec":
            return TRAINER_REGISTRY["markov_synthetic"]
        if spec_name == "SketchRunSpec":
            return TRAINER_REGISTRY["sketch_synthetic"]

    if getattr(cfg, "mode", None) in {"dpo", "grpo", "reward_model", "scalar_reward"}:
        return TRAINER_REGISTRY["trl_preference"]

    if getattr(cfg, "dspy_config", None) is not None:
        return TRAINER_REGISTRY["dspy"]

    # Oracle-driven.
    oracle = getattr(cfg, "oracle", None)
    metadata: Mapping[str, Any] = oracle.metadata() if oracle is not None else {}
    space_kind = str(metadata.get("space_kind", "numeric_sequence"))
    if space_kind == "text":
        return TRAINER_REGISTRY["trl_sft"]
    return TRAINER_REGISTRY["pytorch_tree"]


_BUILTINS_REGISTERED = False


def _ensure_builtins_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    # Side-effect registration via import.
    from treepo._research.unified_g_v1.training.trainers import dspy as _dspy  # noqa: F401
    from treepo._research.unified_g_v1.training.trainers import dspy_online as _dspy_online  # noqa: F401
    from treepo._research.unified_g_v1.training.trainers import dspy_rile as _dspy_rile  # noqa: F401
    from treepo._research.unified_g_v1.training.trainers import dspy_rile_tree as _dspy_rile_tree  # noqa: F401
    from treepo._research.unified_g_v1.training.trainers import grpo_tree as _grpo_tree  # noqa: F401
    from treepo._research.unified_g_v1.training.trainers import learned_sketch_sequence as _seq  # noqa: F401
    from treepo._research.unified_g_v1.training.trainers import markov_synthetic as _mk  # noqa: F401
    from treepo._research.unified_g_v1.training.trainers import pytorch_tree as _pt  # noqa: F401
    from treepo._research.unified_g_v1.training.trainers import sft_best_of_n as _bon  # noqa: F401
    from treepo._research.unified_g_v1.training.trainers import sketch_synthetic as _sk  # noqa: F401
    from treepo._research.unified_g_v1.training.trainers import trl_preference as _pref  # noqa: F401
    from treepo._research.unified_g_v1.training.trainers import trl_sft as _sft  # noqa: F401

    _BUILTINS_REGISTERED = True


__all__ = [
    "TRAINER_REGISTRY",
    "Trainer",
    "register_trainer",
    "resolve_trainer",
]
