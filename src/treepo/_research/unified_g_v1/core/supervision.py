from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import treepo._research.training.trl_training as trl_module
from treepo._research.training.supervision.types import (
    BinaryComparison,
    ComparativeJudgment,
    SupervisionDataset,
)
from treepo._research.training.trl_training import (
    TRLTrainingConfig,
    train_dpo,
    train_grpo,
    train_reward_model,
    train_scalar_reward_model,
)


@dataclass
class UnifiedGSupervisionDataset:
    """Lane-local wrapper over the canonical supervision surface."""

    dataset: SupervisionDataset = field(default_factory=SupervisionDataset)

    @contextmanager
    def _patched_trl_export_builders(self):
        original = {
            "build_dpo_training_records": trl_module.build_dpo_training_records,
            "build_group_grpo_training_records": trl_module.build_group_grpo_training_records,
            "build_reward_model_training_records": trl_module.build_reward_model_training_records,
            "build_scalar_reward_training_records": trl_module.build_scalar_reward_training_records,
        }
        trl_module.build_dpo_training_records = (
            lambda _supervision, *, law_type=None, **_kwargs: self.to_dpo_records(
                law_type=law_type
            )
        )
        trl_module.build_group_grpo_training_records = (
            lambda _supervision, *, law_type=None, **_kwargs: self.to_grpo_records(
                law_type=law_type
            )
        )
        trl_module.build_reward_model_training_records = (
            lambda _supervision, *, law_type=None, **_kwargs: self.to_reward_model_records(
                law_type=law_type
            )
        )
        trl_module.build_scalar_reward_training_records = (
            lambda _supervision, *, law_type=None, **_kwargs: self.to_scalar_reward_records(
                law_type=law_type
            )
        )
        try:
            yield
        finally:
            for name, value in original.items():
                setattr(trl_module, name, value)

    def add_binary_comparisons(
        self,
        comparisons: Iterable[BinaryComparison],
    ) -> None:
        self.dataset.add_comparative_judgments(
            comparison.to_comparative_judgment() for comparison in comparisons
        )

    def add_comparative_judgments(
        self,
        judgments: Iterable[ComparativeJudgment],
    ) -> None:
        self.dataset.comparative_judgments.extend(list(judgments))

    def to_dspy_examples(self) -> List[Any]:
        return self.dataset.project_binary().to_dspy_examples()

    def to_dense_scalar_records(self, *, law_type: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.dataset.to_dense_scalar_training_records(law_type=law_type)

    def to_dpo_records(self, *, law_type: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.dataset.project_binary().to_dpo_records(law_type=law_type)

    def to_grpo_records(self, *, law_type: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.dataset.to_group_grpo_records(law_type=law_type)

    def to_scalar_reward_records(
        self,
        *,
        law_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.dataset.to_scalar_reward_records(law_type=law_type)

    def to_reward_model_records(
        self,
        *,
        law_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.dataset.to_reward_pairs(law_type=law_type)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset.save(output_path)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "UnifiedGSupervisionDataset":
        return cls(dataset=SupervisionDataset.load(path))

    def train_dpo(
        self,
        *,
        model_name: str,
        output_dir: str | Path,
        config: Optional[TRLTrainingConfig] = None,
        law_type: Optional[str] = None,
    ) -> str:
        with self._patched_trl_export_builders():
            return train_dpo(
                self.dataset,
                model_name=model_name,
                output_dir=output_dir,
                config=config,
                law_type=law_type,
            )

    def train_grpo(
        self,
        *,
        model_name: str,
        output_dir: str | Path,
        reward_funcs: Any,
        config: Optional[TRLTrainingConfig] = None,
        law_type: Optional[str] = None,
    ) -> str:
        with self._patched_trl_export_builders():
            return train_grpo(
                self.dataset,
                model_name=model_name,
                output_dir=output_dir,
                config=config,
                law_type=law_type,
                reward_funcs=reward_funcs,
            )

    def train_reward_model(
        self,
        *,
        model_name: str,
        output_dir: str | Path,
        config: Optional[TRLTrainingConfig] = None,
        law_type: Optional[str] = None,
    ) -> str:
        with self._patched_trl_export_builders():
            return train_reward_model(
                self.dataset,
                model_name=model_name,
                output_dir=output_dir,
                config=config,
                law_type=law_type,
            )

    def train_scalar_reward_model(
        self,
        *,
        model_name: str,
        output_dir: str | Path,
        config: Optional[TRLTrainingConfig] = None,
        law_type: Optional[str] = None,
    ) -> str:
        with self._patched_trl_export_builders():
            return train_scalar_reward_model(
                self.dataset,
                model_name=model_name,
                output_dir=output_dir,
                config=config,
                law_type=law_type,
            )
