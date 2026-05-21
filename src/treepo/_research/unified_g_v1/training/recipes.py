"""Preset `TrainerConfig` builders.

Every preset returns the same type: a `TrainerConfig`. Users call `fit()` on
whatever the preset returns. Presets exist for convenience — they are
optional; you can always build a `TrainerConfig` from scratch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from treepo._research.unified_g_v1.core.contracts import MarkovRunSpec
from treepo._research.unified_g_v1.realdoc.manifesto_oracle import ManifestoRileTextOracle
from treepo._research.unified_g_v1.sketch.tree_task import mergeable_sketch_task
from treepo._research.unified_g_v1.training.tree_task import TrainerConfig


__all__ = [
    "manifesto_rile_embedding_fno_task",
    "manifesto_rile_tree_dspy_task",
    "manifesto_rile_text_llm_task",
    "markov_task",
    "mergeable_sketch_task",
]


def markov_task(
    spec: MarkovRunSpec,
    *,
    use_cuda: bool = True,
    cuda_device: int | None = None,
    torch_threads: int = 0,
    reuse_existing: bool = True,
    config_overrides: dict | None = None,
) -> TrainerConfig:
    """Preset: a single Markov-synthetic OPSCount run.

    The `MarkovRunSpec` populates `cfg.run_spec`, so `resolve_trainer` picks
    the `markov_synthetic` trainer automatically.
    """
    return TrainerConfig(
        run_spec=spec,
        use_cuda=bool(use_cuda),
        cuda_device=cuda_device,
        torch_threads=int(torch_threads),
        reuse_existing=bool(reuse_existing),
        config_overrides=config_overrides,
    )


def manifesto_rile_embedding_fno_task(
    *,
    embedding_api_base: str,
    phase1_data_path: str | Path | None = None,
    split_ids_path: str | Path | None = None,
    embedding_model: str | None = None,
    embedding_api_key: str = "EMPTY",
    embedding_batch_size: int = 32,
    leaf_tokens: int = 1024,
    subwindow_tokens: int = 128,
    token_encoding: str = "cl100k_base",
    # None => auto-size to the embedding server's output dim (and 2x that for state).
    summary_dim: int | None = None,
    state_dim: int | None = None,
    adapter_hidden_dim: int | None = 512,
    g_hidden_dim: int | None = 512,
    head_width: int | None = None,
    operator_modes: int | None = 32,
    train_batch_size: int = 4,
    epochs: int = 8,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
    seed: int = 42,
    device: str = "auto",
    # Local-law balance knobs. Canonical formula:
    # (1 - λ) · root + λ · Σ ρᵢ · Cᵢ / Σ ρᵢ
    local_law_weight: float = 0.3,
    c1_relative_weight: float = 1.0,
    c2_relative_weight: float = 1.0,
    c3_relative_weight: float = 1.0,
) -> TrainerConfig:
    """Preset: Manifesto RILE with embedding-sequence FNO, as a `TrainerConfig`.

    Building this preset is eager: it constructs the oracle, connects to the
    embedding service, materializes leaf sketches, then builds the
    `EmbeddingSequenceFNOTreeModel` sized to the observed embedding dim. Pass
    the returned config straight to `fit()`.
    """
    import random
    import torch

    from treepo._research.unified_g_v1.realdoc.embedding_fno_training import (
        EmbeddingSequenceFNOTreeModel,
        _resolve_device,
    )
    from treepo._research.unified_g_v1.eval.law_stress import predict_train_mean_baseline_mae
    from treepo._research.unified_g_v1.training.objectives import ManifestoRileEmbeddingObjective
    from treepo._research.unified_g_v1.training.oracles import ManifestoRileEmbeddingOracle
    from treepo._research.tasks.manifesto import RILE_SCALE

    random.seed(int(seed))
    torch.manual_seed(int(seed))

    oracle = ManifestoRileEmbeddingOracle(
        embedding_api_base=str(embedding_api_base),
        phase1_data_path=None if phase1_data_path is None else str(phase1_data_path),
        split_ids_path=None if split_ids_path is None else str(split_ids_path),
        embedding_model=embedding_model,
        embedding_api_key=str(embedding_api_key),
        embedding_batch_size=int(embedding_batch_size),
        leaf_tokens=int(leaf_tokens),
        subwindow_tokens=int(subwindow_tokens),
        token_encoding=str(token_encoding),
        device=str(device),
        seed=int(seed),
    )
    oracle._ensure_built()  # materialize embeddings + learn embedding_dim
    resolved_device = _resolve_device(str(device))
    model = EmbeddingSequenceFNOTreeModel(
        embedding_dim=int(oracle.embedding_dim or 0),
        summary_dim=summary_dim,
        state_dim=state_dim,
        adapter_hidden_dim=adapter_hidden_dim,
        g_hidden_dim=g_hidden_dim,
        head_width=head_width,
        operator_modes=operator_modes,
        target_min=float(RILE_SCALE.min_value),
        target_max=float(RILE_SCALE.max_value),
    ).to(resolved_device)
    # Pre-compute predict-train-mean baseline MAE so every epoch's evaluate()
    # can report val_mae_gain_frac on the local-law (-∞, 1.0] scale.
    baseline_val_mae = predict_train_mean_baseline_mae(
        oracle.train_examples(), oracle.val_examples(),
        target_getter=lambda ex: float(ex.target),
    )
    objective = ManifestoRileEmbeddingObjective(
        rile_scale=RILE_SCALE,
        baseline_val_mae=baseline_val_mae,
        local_law_weight=float(local_law_weight),
        c1_relative_weight=float(c1_relative_weight),
        c2_relative_weight=float(c2_relative_weight),
        c3_relative_weight=float(c3_relative_weight),
    )
    return TrainerConfig(
        oracle=oracle,
        model=model,
        objective=objective,
        n_epochs=int(epochs),
        train_batch_size=int(train_batch_size),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        grad_clip_norm=float(grad_clip_norm),
        seed=int(seed),
        best_metric_key="mae_raw",
    )


def manifesto_rile_text_llm_task(
    *,
    prepared_dataset_path: str | Path,
    model_name: str,
    trl_config: Any | None = None,
) -> TrainerConfig:
    """Preset: Manifesto RILE text-LLM.

    Oracle declares `space_kind="text"`, so `resolve_trainer` auto-picks the
    TRL SFT trainer. Override with `cfg.trainer=...` if needed.

    Companion prep: `parallel/unified_g_v1/scripts/prep_text_dataset.py`.
    """
    oracle = ManifestoRileTextOracle.from_path(prepared_dataset_path)
    return TrainerConfig(
        oracle=oracle,
        model_name=str(model_name),
        trl_config=trl_config,
    )


def manifesto_rile_tree_dspy_task(
    *,
    split_ids_path: str | Path | None = None,
    phase1_data_path: str | Path | None = None,
    model_name: str,
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    leaf_tokens: int = 1024,
    token_encoding: str = "cl100k_base",
    optimizer: str = "gepa",
    seed: int = 0,
    local_law_weight: float = 0.3,
    c1_relative_weight: float = 1.0,
    c2_relative_weight: float = 1.0,
    c3_relative_weight: float = 1.0,
) -> TrainerConfig:
    """Preset: tree-structured DSPy RILE optimization over raw manifesto docs."""
    from treepo._research.unified_g_v1.training.oracles import ManifestoRileTreeOracle

    oracle = ManifestoRileTreeOracle(
        phase1_data_path=phase1_data_path,
        split_ids_path=split_ids_path,
        leaf_tokens=int(leaf_tokens),
        token_encoding=str(token_encoding),
        enforce_local_laws=True,
    )
    return TrainerConfig(
        trainer="dspy_rile_tree",
        oracle=oracle,
        model_name=str(model_name),
        seed=int(seed),
        extra={
            "api_base": str(api_base),
            "api_key": str(api_key),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "optimizer": str(optimizer),
            "local_law_weight": float(local_law_weight),
            "c1_relative_weight": float(c1_relative_weight),
            "c2_relative_weight": float(c2_relative_weight),
            "c3_relative_weight": float(c3_relative_weight),
        },
    )
