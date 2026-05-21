from __future__ import annotations

from pathlib import Path
from typing import Any

from treepo._research.unified_g_v1.core.manifest import write_json
from treepo._research.unified_g_v1.core.supervision import UnifiedGSupervisionDataset


def _coerce_dspy_example(example: Any) -> dict[str, Any]:
    if hasattr(example, "toDict"):
        return dict(example.toDict())
    if hasattr(example, "to_dict"):
        return dict(example.to_dict())
    if hasattr(example, "__dict__"):
        return dict(example.__dict__)
    return {"value": repr(example)}


def export_supervision_formats(
    dataset: UnifiedGSupervisionDataset,
    *,
    output_dir: str | Path,
    law_type: str | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "dspy_examples": output_dir / "dspy_examples.json",
        "dense_scalar": output_dir / "dense_scalar_records.json",
        "dpo": output_dir / "dpo_records.json",
        "grpo": output_dir / "grpo_records.json",
        "scalar_reward": output_dir / "scalar_reward_records.json",
        "reward_model": output_dir / "reward_model_records.json",
    }
    write_json(
        paths["dspy_examples"],
        [_coerce_dspy_example(example) for example in dataset.to_dspy_examples()],
    )
    write_json(paths["dense_scalar"], dataset.to_dense_scalar_records(law_type=law_type))
    write_json(paths["dpo"], dataset.to_dpo_records(law_type=law_type))
    write_json(paths["grpo"], dataset.to_grpo_records(law_type=law_type))
    write_json(
        paths["scalar_reward"],
        dataset.to_scalar_reward_records(law_type=law_type),
    )
    write_json(
        paths["reward_model"],
        dataset.to_reward_model_records(law_type=law_type),
    )
    return {name: str(path) for name, path in paths.items()}
