"""Internal compatibility wrapper over ``src.training.supervision.adapters``."""

from treepo._research.training.supervision.adapters import (
    BinaryProjection,
    build_dense_scalar_training_records,
    build_dpo_training_records,
    build_group_grpo_training_records,
    build_reward_model_training_records,
    build_scalar_reward_training_records,
    coerce_binary_projection as coerce_preference_dataset,
    coerce_comparative_dataset,
    prepare_binary_optimizer_dataset,
)

__all__ = [
    "BinaryProjection",
    "build_dense_scalar_training_records",
    "build_dpo_training_records",
    "build_group_grpo_training_records",
    "build_reward_model_training_records",
    "build_scalar_reward_training_records",
    "coerce_comparative_dataset",
    "coerce_preference_dataset",
    "prepare_binary_optimizer_dataset",
]
