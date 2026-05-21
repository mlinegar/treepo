"""`TrainerConfig` — one flexible dataclass, any training paradigm.

A `TrainerConfig` is a bag of optional fields. Any trainer (a plain callable)
reads the fields it needs and ignores the rest. The trainer is resolved by
`resolve_trainer(cfg)` which inspects which fields are populated and, failing
that, the oracle's `space_kind`.

Typical shapes:

    # PyTorch tree task (sketch, FNO, any tree-structured module)
    TrainerConfig(oracle=my_oracle, model=my_module, objective=my_loss,
                  n_epochs=5, learning_rate=1e-3)

    # TRL SFT on text (auto-detected from oracle.space_kind=="text")
    TrainerConfig(oracle=ManifestoRileTextOracle.from_path(...),
                  model_name="Qwen/Qwen2-0.5B")

    # Synthetic Markov / mergeable-sketch runs
    TrainerConfig(run_spec=MarkovRunSpec(...))
    TrainerConfig(run_spec=SketchRunSpec(...))

    # TRL preference training (DPO / GRPO / reward model / scalar reward)
    TrainerConfig(supervision_dataset=ds, mode="dpo", model_name="...")

    # DSPy bootstrap
    TrainerConfig(dspy_config=LawStressDSPyOptimizeConfig(...))

    # Fully custom: plug in your own trainer function
    TrainerConfig(oracle=my_oracle, trainer=my_fn)

The tree abstractions (`TreeExample`, `TreeOracle`, `TreeModel`,
`TreeObjective`) live here because the PyTorch tree trainer uses them, but
they are not required for other trainers. Set only the fields your trainer
reads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Tree abstractions (used by the PyTorch tree trainer; optional elsewhere).
# ---------------------------------------------------------------------------


@dataclass
class TreeExample:
    """A single tree-structured training example.

    `leaves` is the per-leaf payload (tensor, dict, tokens, whatever the model
    expects). `target` is the supervision signal at the root. `extra` carries
    anything the objective or model needs.
    """

    leaves: Sequence[Any]
    target: Any
    extra: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class TreeOracle(Protocol):
    """Produces training + validation examples."""

    def train_examples(self) -> Sequence[TreeExample]: ...

    def val_examples(self) -> Sequence[TreeExample]: ...

    def metadata(self) -> Mapping[str, Any]: ...


@runtime_checkable
class TreeModel(Protocol):
    """A tree-structured model. Must be an `nn.Module`.

    `forward_tree(batch) -> (root_state, prediction)` is the one method the
    PyTorch tree trainer calls.
    """

    def forward_tree(self, batch: Sequence[TreeExample]) -> tuple[torch.Tensor, torch.Tensor]: ...


@runtime_checkable
class TreeObjective(Protocol):
    """Loss + evaluation over tree examples.

    `forward_aux` is an optional dict a `TreeModel` can populate alongside
    `(root_state, prediction)` when it wants to pass per-forward intermediate
    tensors (e.g. Markov's per-node C1/C2/C3 losses) straight to the objective
    without polluting the generic training loop. Objectives that don't need
    it can declare `forward_aux=None` and ignore it.
    """

    def compute_loss(
        self,
        *,
        root_state: torch.Tensor,
        prediction: torch.Tensor,
        batch: Sequence[TreeExample],
        forward_aux: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, int, Mapping[str, Any]]: ...

    def evaluate(
        self,
        *,
        model: nn.Module,
        items: Sequence[TreeExample],
        batch_size: int,
    ) -> Mapping[str, Any]: ...


# ---------------------------------------------------------------------------
# The general TrainerConfig.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainerConfig:
    """General training config. Fill in only what your trainer needs.

    ## Selecting a trainer

    If `trainer` is set, it wins (callable directly, or a name in the
    registry). Otherwise `resolve_trainer` picks by populated fields:

    * `run_spec: MarkovRunSpec`           -> markov synthetic
    * `run_spec: SketchRunSpec`           -> sketch synthetic (if no oracle)
    * `mode in {dpo, grpo, ...}`          -> TRL preference
    * `dspy_config` present               -> DSPy subprocess
    * `oracle.space_kind == "text"`       -> TRL SFT
    * `oracle` + `model` + `objective`    -> PyTorch tree trainer
    """

    # ---- task / data ------------------------------------------------------
    oracle: Any | None = None
    model: Any | None = None             # nn.Module (or anything the trainer wants)
    objective: Any | None = None         # TreeObjective-like, or a callable

    # ---- explicit trainer override ---------------------------------------
    trainer: Any = None                  # Callable | registered name | None

    # ---- shared training knobs -------------------------------------------
    n_epochs: int = 4
    train_batch_size: int = 32
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    seed: int = 0
    save_every_epoch: bool = False
    best_metric_key: str = "val_mae"
    optimizer_builder: Callable[[nn.Module, "TrainerConfig"], torch.optim.Optimizer] | None = None

    # ---- TRL SFT ---------------------------------------------------------
    model_name: str | None = None
    trl_config: Any | None = None

    # ---- TRL preference --------------------------------------------------
    supervision_dataset: Any | None = None          # UnifiedGSupervisionDataset
    mode: str | None = None                         # "dpo" | "grpo" | "reward_model" | "scalar_reward"
    reward_funcs: Any | None = None                 # for GRPO
    law_type: str | None = None

    # ---- DSPy ------------------------------------------------------------
    dspy_config: Any | None = None
    dspy_execute: bool = True
    dspy_python_executable: str | Path | None = None
    dspy_script_path: str | Path | None = None
    dspy_cwd: str | Path | None = None

    # ---- Synthetic run specs (Markov / Sketch) ---------------------------
    run_spec: Any | None = None
    reuse_existing: bool = True
    use_cuda: bool = True
    cuda_device: int | None = None
    torch_threads: int = 0
    config_overrides: Mapping[str, Any] | None = None

    # ---- Online / interactive training -----------------------------------
    # `feedback_fn(prompt, candidates) -> int | list[int]` returns either the
    # index of the preferred candidate or a full ranking (descending). Online
    # trainers (dspy_online, sft_best_of_n) call this per example.
    feedback_fn: Callable[[Any, Sequence[Any]], Any] | None = None
    candidates_per_example: int = 1
    # `base_module` is whatever the online trainer uses to generate candidates:
    # a dspy.Module for `dspy_online`; any `prompt -> str` callable for
    # `sft_best_of_n`. Interpretation is trainer-specific.
    base_module: Any | None = None

    # ---- Escape hatch for trainer-specific extras ------------------------
    extra: Mapping[str, Any] = field(default_factory=dict)


# Back-compat alias: the old name still points at the same general config.
# External callers that did `TreeTaskConfig(oracle=..., model=..., objective=...)`
# continue to work unchanged.
TreeTaskConfig = TrainerConfig


# ---------------------------------------------------------------------------
# Dispatch shim — routes a TrainerConfig to its resolved trainer.
# ---------------------------------------------------------------------------


def run_tree_task(
    cfg: TrainerConfig,
    *,
    output_dir: str | Path,
    dataset=None,
):
    """Resolve the trainer for `cfg` and call it. `fit()` delegates to this."""
    # Import lazily to avoid a circular import at module load time.
    from treepo._research.unified_g_v1.training.trainers import resolve_trainer

    trainer = resolve_trainer(cfg)
    return trainer(cfg, Path(output_dir), dataset)
