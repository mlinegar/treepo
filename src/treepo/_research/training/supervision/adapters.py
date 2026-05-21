"""Shared optimizer adapters over the primary supervision surface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from treepo._research.training.supervision.types import (
    BinaryComparison,
    BinaryProjectionDataset,
    BinaryProjectionMode,
    ComparativeDataset,
    ComparativeJudgment,
    SupervisionDataset,
    coerce_supervision_dataset,
)
from treepo._research.training.supervision.comparative_types import PromptBuilder

BinaryProjection = BinaryProjectionMode


def coerce_binary_projection(
    supervision: Union[
        SupervisionDataset,
        BinaryProjectionDataset,
        ComparativeDataset,
        Sequence[BinaryComparison],
        Sequence[ComparativeJudgment],
    ],
    *,
    projection: BinaryProjection = "adjacent",
) -> BinaryProjectionDataset:
    """Coerce supervision inputs into a binary optimizer projection."""
    return coerce_supervision_dataset(supervision).project_binary(
        projection=projection,
    )


def coerce_comparative_dataset(
    supervision: Union[
        SupervisionDataset,
        BinaryProjectionDataset,
        ComparativeDataset,
        Sequence[BinaryComparison],
        Sequence[ComparativeJudgment],
    ],
    *,
    law_type: Optional[str] = None,
) -> ComparativeDataset:
    """Coerce supervision inputs into grouped comparative judgments."""
    return coerce_supervision_dataset(supervision).to_comparative_dataset(
        law_type=law_type
    )


def prepare_binary_optimizer_dataset(
    supervision: Union[
        SupervisionDataset,
        BinaryProjectionDataset,
        ComparativeDataset,
        Sequence[BinaryComparison],
        Sequence[ComparativeJudgment],
    ],
    *,
    projection: BinaryProjection = "adjacent",
    keep_existing: bool = True,
) -> BinaryProjectionDataset:
    """Return a binary optimizer view over any supervision input."""
    del keep_existing
    return coerce_supervision_dataset(supervision).project_binary(
        projection=projection,
    )


def build_dpo_training_records(
    supervision: Union[
        SupervisionDataset,
        BinaryProjectionDataset,
        ComparativeDataset,
        Sequence[BinaryComparison],
        Sequence[ComparativeJudgment],
    ],
    *,
    law_type: Optional[str] = None,
    prompt_builder: Optional[PromptBuilder] = None,
    projection: BinaryProjection = "adjacent",
    tree_objective_weighting_mode: str = "legacy_channel",
    discount_gamma: float = 1.0,
) -> List[Dict[str, Any]]:
    return coerce_supervision_dataset(supervision).to_dpo_records(
        law_type=law_type,
        prompt_builder=prompt_builder,
        projection=projection,
        tree_objective_weighting_mode=tree_objective_weighting_mode,
        discount_gamma=discount_gamma,
    )


def build_group_grpo_training_records(
    supervision: Union[
        SupervisionDataset,
        BinaryProjectionDataset,
        ComparativeDataset,
        Sequence[BinaryComparison],
        Sequence[ComparativeJudgment],
    ],
    *,
    law_type: Optional[str] = None,
    prompt_builder: Optional[PromptBuilder] = None,
    min_group_size: int = 2,
    tree_objective_weighting_mode: str = "legacy_channel",
    discount_gamma: float = 1.0,
) -> List[Dict[str, Any]]:
    return coerce_supervision_dataset(supervision).to_group_grpo_records(
        law_type=law_type,
        prompt_builder=prompt_builder,
        min_group_size=min_group_size,
        tree_objective_weighting_mode=tree_objective_weighting_mode,
        discount_gamma=discount_gamma,
    )


def build_scalar_reward_training_records(
    supervision: Union[
        SupervisionDataset,
        BinaryProjectionDataset,
        ComparativeDataset,
        Sequence[BinaryComparison],
        Sequence[ComparativeJudgment],
    ],
    *,
    law_type: Optional[str] = None,
    prompt_builder: Optional[PromptBuilder] = None,
    tree_objective_weighting_mode: str = "legacy_channel",
    discount_gamma: float = 1.0,
) -> List[Dict[str, Any]]:
    return coerce_supervision_dataset(supervision).to_scalar_reward_records(
        law_type=law_type,
        prompt_builder=prompt_builder,
        tree_objective_weighting_mode=tree_objective_weighting_mode,
        discount_gamma=discount_gamma,
    )


def build_dense_scalar_training_records(
    supervision: Union[
        SupervisionDataset,
        BinaryProjectionDataset,
        ComparativeDataset,
        Sequence[BinaryComparison],
        Sequence[ComparativeJudgment],
    ],
    *,
    law_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return coerce_supervision_dataset(supervision).to_dense_scalar_training_records(
        law_type=law_type,
    )


def build_reward_model_training_records(
    supervision: Union[
        SupervisionDataset,
        BinaryProjectionDataset,
        ComparativeDataset,
        Sequence[BinaryComparison],
        Sequence[ComparativeJudgment],
    ],
    *,
    law_type: Optional[str] = None,
    prompt_builder: Optional[PromptBuilder] = None,
    projection: BinaryProjection = "adjacent",
    include_oracle_scores: bool = True,
    tree_objective_weighting_mode: str = "legacy_channel",
    discount_gamma: float = 1.0,
) -> List[Dict[str, Any]]:
    return coerce_supervision_dataset(supervision).to_reward_pairs(
        law_type=law_type,
        prompt_builder=prompt_builder,
        projection=projection,
        include_oracle_scores=include_oracle_scores,
        tree_objective_weighting_mode=tree_objective_weighting_mode,
        discount_gamma=discount_gamma,
    )


__all__ = [
    "BinaryProjection",
    "build_dense_scalar_training_records",
    "build_dpo_training_records",
    "build_group_grpo_training_records",
    "build_reward_model_training_records",
    "build_scalar_reward_training_records",
    "coerce_binary_projection",
    "coerce_comparative_dataset",
    "prepare_binary_optimizer_dataset",
]
