"""Shared artifact writers for canonical supervision bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from treepo._research.training.supervision.types import (
    BinaryProjectionMode,
    SupervisionDataset,
    SupervisionInput,
    coerce_supervision_dataset,
)


@dataclass(frozen=True)
class SupervisionArtifactBundlePaths:
    """Paths written for one saved supervision bundle."""

    supervision_path: Path
    binary_projection_path: Optional[Path] = None
    comparative_path: Optional[Path] = None
    dpo_path: Optional[Path] = None
    group_grpo_path: Optional[Path] = None
    scalar_reward_path: Optional[Path] = None
    stats_path: Optional[Path] = None


def _ensure_parent(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_binary_projection_compat(dataset: Any, path: Path) -> None:
    comparisons = list(getattr(dataset, "comparisons", None) or [])
    if not comparisons:
        comparisons = list(getattr(dataset, "pairs", []) or [])
    comparative_judgments = list(
        getattr(dataset, "comparative_judgments", None) or []
    )
    if not comparative_judgments:
        comparative_judgments = list(
            getattr(dataset, "comparative_records", []) or []
        )
    payload = {
        "version": "4.0",
        "num_binary_comparisons": len(comparisons),
        "binary_comparisons": [comparison.to_dict() for comparison in comparisons],
        "num_comparative_judgments": len(comparative_judgments),
        "comparative_judgments": [
            record.to_dict()
            for record in comparative_judgments
        ],
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def save_supervision_artifact_bundle(
    supervision: SupervisionInput,
    *,
    supervision_path: Path,
    binary_projection_path: Optional[Path] = None,
    comparative_path: Optional[Path] = None,
    dpo_path: Optional[Path] = None,
    group_grpo_path: Optional[Path] = None,
    scalar_reward_path: Optional[Path] = None,
    stats_path: Optional[Path] = None,
    stats: Optional[Dict[str, Any]] = None,
    law_type: Optional[str] = None,
    prompt_builder: Any = None,
    binary_projection: BinaryProjectionMode = "adjacent",
) -> SupervisionArtifactBundlePaths:
    """Save the primary supervision artifact plus optional optimizer projections."""

    dataset = coerce_supervision_dataset(supervision)
    supervision_path = Path(supervision_path)
    supervision_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(supervision_path)

    binary_projection_path = _ensure_parent(binary_projection_path)
    comparative_path = _ensure_parent(comparative_path)
    dpo_path = _ensure_parent(dpo_path)
    group_grpo_path = _ensure_parent(group_grpo_path)
    scalar_reward_path = _ensure_parent(scalar_reward_path)
    stats_path = _ensure_parent(stats_path)

    if binary_projection_path is not None:
        _save_binary_projection_compat(
            dataset.project_binary(projection=binary_projection),
            binary_projection_path,
        )

    if comparative_path is not None:
        dataset.to_comparative_dataset(law_type=law_type).save(comparative_path)

    if dpo_path is not None:
        with open(dpo_path, "w") as handle:
            json.dump(
                dataset.to_dpo_records(
                    law_type=law_type,
                    prompt_builder=prompt_builder,
                    projection=binary_projection,
                ),
                handle,
                indent=2,
            )

    if group_grpo_path is not None:
        with open(group_grpo_path, "w") as handle:
            json.dump(
                dataset.to_group_grpo_records(
                    law_type=law_type,
                    prompt_builder=prompt_builder,
                ),
                handle,
                indent=2,
            )

    if scalar_reward_path is not None:
        with open(scalar_reward_path, "w") as handle:
            json.dump(
                dataset.to_scalar_reward_records(
                    law_type=law_type,
                    prompt_builder=prompt_builder,
                ),
                handle,
                indent=2,
            )

    if stats_path is not None and stats is not None:
        with open(stats_path, "w") as handle:
            json.dump(stats, handle, indent=2)

    return SupervisionArtifactBundlePaths(
        supervision_path=supervision_path,
        binary_projection_path=binary_projection_path,
        comparative_path=comparative_path,
        dpo_path=dpo_path,
        group_grpo_path=group_grpo_path,
        scalar_reward_path=scalar_reward_path,
        stats_path=stats_path,
    )


__all__ = [
    "SupervisionArtifactBundlePaths",
    "save_supervision_artifact_bundle",
]
