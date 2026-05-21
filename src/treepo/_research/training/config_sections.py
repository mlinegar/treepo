"""Shared sectioned training configuration blocks.

These dataclasses are intentionally small and serializable.  Training
surfaces compose them with model- or objective-specific sections rather than
growing one-off flat config bags.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


def _tuple_of_strings(values: tuple[str, ...] | list[str] | str) -> Tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    return tuple(str(value) for value in values)


def config_to_dict(value: Any) -> Any:
    """Convert nested config sections into JSON-friendly primitives."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return config_to_dict(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): config_to_dict(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [config_to_dict(item) for item in value]
    return str(value)


@dataclass(frozen=True, kw_only=True)
class RunConfig:
    output_dir: Optional[Path] = None
    dry_run: bool = False
    seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return config_to_dict(self)


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    train_splits: Tuple[str, ...] = ("train",)
    batch_size: int = 1
    epochs: int = 1
    steps: Optional[int] = None
    shuffle: bool = True
    gradient_accumulation_steps: int = 1
    logging_steps: int = 10
    save_steps: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_splits", _tuple_of_strings(self.train_splits))

    def to_dict(self) -> Dict[str, Any]:
        return config_to_dict(self)


@dataclass(frozen=True, kw_only=True)
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    scheduler: str = "none"
    warmup_epochs: int = 0
    warmup_ratio: float = 0.0
    min_learning_rate: float = 0.0
    grad_clip_norm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return config_to_dict(self)


@dataclass(frozen=True, kw_only=True)
class ValidationConfig:
    val_splits: Tuple[str, ...] = ("val",)
    enabled: bool = True
    eval_every: int = 1
    eval_steps: int = 100
    selection_metric: str = "val_loss"
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "val_splits", _tuple_of_strings(self.val_splits))

    def to_dict(self) -> Dict[str, Any]:
        return config_to_dict(self)


@dataclass(frozen=True, kw_only=True)
class TestConfig:
    test_splits: Tuple[str, ...] = ("test",)
    enabled: bool = False
    metrics_mode: str = "default"

    def __post_init__(self) -> None:
        object.__setattr__(self, "test_splits", _tuple_of_strings(self.test_splits))

    def to_dict(self) -> Dict[str, Any]:
        return config_to_dict(self)


@dataclass(frozen=True, kw_only=True)
class RuntimeConfig:
    device: str = "auto"
    bf16: bool = True
    gradient_checkpointing: bool = True
    deterministic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return config_to_dict(self)


__all__ = [
    "OptimizerConfig",
    "RunConfig",
    "RuntimeConfig",
    "TestConfig",
    "TrainConfig",
    "ValidationConfig",
    "config_to_dict",
]
