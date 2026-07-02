"""Config dataclasses for the method examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class HllSketchConfig:
    backend: str = "datasketches"
    precision: int = 14
    hash_bits: int = 64
    schedule: str = "balanced"
    n_trees: int = 6
    leaves_per_tree: int = 4
    doc_unit_kind: str = "item"
    leaf_unit_count: int = 24
    vocabulary_size: int = 200
    seed: int = 0
    max_iterations: int = 2


@dataclass
class FnoMarkovConfig:
    n_train: int = 8
    n_eval: int = 4
    doc_tokens: int = 32
    doc_unit_kind: str = "token"
    leaf_unit_count: int = 8
    vocabulary_size: int = 64
    seed: int = 0
    max_iterations: int = 3
    embedding_dim: int = 8
    hidden_channels: int = 4
    n_modes: int = 2
    n_layers: int = 1
    head_hidden_dim: int = 8
    epochs_per_iteration: int = 1
    batch_size: int = 4
    learning_rate: float = 0.01
    device: str = "cpu"
    preference_mode: str = "scores"


@dataclass
class NeuralOperatorCompareConfig:
    operator_kinds: Sequence[str] = field(default_factory=lambda: ("fno", "tfno", "uno"))
    n_train: int = 8
    n_eval: int = 5
    n_states: int = 4
    doc_tokens: int = 32
    doc_unit_kind: str = "token"
    leaf_unit_count: int = 8
    transition_prob: float = 0.15
    vocabulary_size: int = 64
    seed: int = 0
    max_iterations: int = 3
    embedding_dim: int = 8
    hidden_channels: int = 4
    n_modes: int = 2
    n_layers: int = 1
    conv_kernel_size: int = 3
    head_hidden_dim: int = 8
    epochs_per_iteration: int = 1
    batch_size: int = 4
    learning_rate: float = 0.01
    device: str = "cpu"
    fixture_device: str | None = None
    normalize_targets: bool = True
    numeric_transition_state_weight: float = 0.0
    numeric_transition_count_scale: float | None = None


@dataclass
class NeuralOperatorMarkovLeafGridConfig:
    operator_kinds: Sequence[str] = field(default_factory=lambda: ("fno", "tfno", "uno", "conv1d"))
    doc_unit_kind: str = "token"
    leaf_unit_counts: Sequence[int] = field(default_factory=lambda: (32, 64, 128))
    n_train: int = 512
    n_eval: int = 128
    n_states: int = 4
    doc_tokens: int = 256
    transition_prob: float = 0.15
    vocabulary_size: int = 256
    seed: int = 0
    max_iterations: int = 4
    embedding_dim: int = 16
    hidden_channels: int = 12
    n_modes: int = 4
    n_layers: int = 1
    conv_kernel_size: int = 3
    head_hidden_dim: int = 24
    epochs_per_iteration: int = 3
    batch_size: int = 32
    learning_rate: float = 0.01
    device: str = "cpu"
    fixture_device: str | None = None
    normalize_targets: bool = True
    numeric_transition_state_weight: float = 0.05
    numeric_transition_count_scale: float | None = None


@dataclass
class NeuralOperatorLDAConfig:
    operator_kinds: Sequence[str] = field(default_factory=lambda: ("fno", "tfno", "uno"))
    n_train: int = 256
    n_eval: int = 64
    n_topics: int = 8
    doc_tokens: int = 256
    doc_unit_kind: str = "token"
    leaf_unit_count: int = 32
    vocabulary_size: int = 160
    doc_topic_concentration: float = 0.7
    topic_word_concentration: float = 0.05
    target_topic: int = 0
    topic_seed: int = 0
    seed: int = 0
    max_iterations: int = 3
    embedding_dim: int = 16
    hidden_channels: int = 12
    n_modes: int = 4
    n_layers: int = 1
    head_hidden_dim: int = 24
    epochs_per_iteration: int = 3
    batch_size: int = 16
    learning_rate: float = 0.01
    device: str = "cpu"
    fixture_device: str | None = None
    sklearn_max_iter: int = 50
    run_sklearn_baseline: bool = True


@dataclass
class NeuralOperatorLDALeafGridConfig:
    operator_kinds: Sequence[str] = field(default_factory=lambda: ("fno", "tfno", "uno", "conv1d"))
    doc_unit_kind: str = "token"
    leaf_unit_counts: Sequence[int] = field(default_factory=lambda: (32, 64, 128))
    n_train: int = 256
    n_eval: int = 64
    n_topics: int = 8
    doc_tokens: int = 256
    vocabulary_size: int = 160
    doc_topic_concentration: float = 0.7
    topic_word_concentration: float = 0.05
    target_topic: int = 0
    topic_seed: int = 0
    seed: int = 0
    max_iterations: int = 3
    embedding_dim: int = 16
    hidden_channels: int = 12
    n_modes: int = 4
    n_layers: int = 1
    conv_kernel_size: int = 3
    head_hidden_dim: int = 24
    epochs_per_iteration: int = 3
    batch_size: int = 16
    learning_rate: float = 0.01
    device: str = "cpu"
    fixture_device: str | None = None
    sklearn_max_iter: int = 50
    run_sklearn_baseline: bool = True


@dataclass
class ManifestoReplicationConfig:
    family: str = "dspy"
    model: str = "local-dspy-program"
    max_iterations: int = 2
    use_oracle_predictor: bool = True
    doc_sample_size: int | None = None
    doc_sample_rate: float | None = None
    doc_sample_seed: int = 0
    qsentence_sample_size: int | None = None
    qsentence_sample_rate: float | None = None
    qsentence_sample_seed: int | None = None
    sample_size: int | None = None
    sample_rate: float | None = None
    sample_seed: int = 0
    prompt_template: str = ""
    preference_mode: str = "none"
    preference_scope: str = "both"
    doc_unit_kind: str = "qsentence"
    leaf_unit_count: int = 1
    leaf_unit_counts: tuple[int, ...] = ()
    supervision_grid: tuple[str, ...] = ()


@dataclass
class ManifestoEndToEndConfig:
    family: str = "dspy"
    model: str = "local-dspy-program"
    max_iterations: int = 2
    use_oracle_predictor: bool = True
    doc_sample_size: int | None = None
    doc_sample_rate: float | None = None
    doc_sample_seed: int = 0
    qsentence_sample_size: int | None = None
    qsentence_sample_rate: float | None = None
    qsentence_sample_seed: int = 0
    prompt_template: str = ""
    doc_unit_kind: str = "qsentence"
    leaf_unit_count: int = 1
    fit_preference_mode: str = "scores"
    fit_preference_scope: str = "both"
    reward_scopes: tuple[str, ...] = ("roots", "qsentences", "both")
    reward_modes: tuple[str, ...] = ("pairwise", "ranked")
    export_formats: tuple[str, ...] = ("general", "supervised", "dpo", "reward", "grpo")


@dataclass
class ManifestoRewardMechanismConfig:
    doc_sample_size: int | None = None
    doc_sample_rate: float | None = None
    doc_sample_seed: int = 0
    qsentence_sample_size: int | None = None
    qsentence_sample_rate: float | None = None
    qsentence_sample_seed: int = 0
    doc_unit_kind: str = "qsentence"
    leaf_unit_count: int = 1
    preference_scopes: tuple[str, ...] = ("roots", "qsentences", "both")
    preference_modes: tuple[str, ...] = ("pairwise", "ranked")
    export_formats: tuple[str, ...] = ("general", "supervised", "dpo", "reward", "grpo")

