"""TRL SFT backend — thin wrapper over `src.training.trl_training.train_sft`.

Delegates to the unified TRL/PEFT entry point so SFT inherits the same LoRA +
quantization handling as DPO, GRPO, reward-model, and scalar-reward training.
A single `TRLTrainingConfig(use_lora=..., lora_r=..., lora_target_modules=...,
load_in_4bit=..., ...)` controls all paths.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from treepo._research.unified_g_v1.training.prepared_dataset import PreparedDataset

from treepo._research.training.trl_training import TRLTrainingConfig, train_sft


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_text_pairs(dataset: PreparedDataset, split: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for shard in dataset.shard_paths(split):
        for record in _iter_jsonl(shard):
            prompt = str(record.get("prompt", ""))
            completion = str(record.get("completion", ""))
            if not prompt or not completion:
                continue
            records.append({"prompt": prompt, "completion": completion})
    return records


def run_trl_sft(
    *,
    dataset: PreparedDataset,
    model_name: str,
    output_dir: Path,
    trl_config: TRLTrainingConfig | None = None,
) -> dict[str, Any]:
    """Supervised fine-tune an LLM on `{prompt, completion}` pairs via TRL SFTTrainer.

    Lazy-delegates to `src.training.trl_training.train_sft`, which handles
    LoRA (PEFT) + quantization uniformly with the preference-training paths.
    """
    if dataset.payload_schema != "text_pairs_v1":
        raise ValueError(
            f"trl_sft requires payload_schema=text_pairs_v1, got {dataset.payload_schema!r}"
        )
    train_records = _load_text_pairs(dataset, "train")
    val_records = _load_text_pairs(dataset, "val") if dataset.has_split("val") else []
    if not train_records:
        raise ValueError("trl_sft: no training records found in prepared dataset")

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    final_path = train_sft(
        records=train_records,
        model_name=str(model_name),
        output_dir=output_dir,
        config=trl_config,
        eval_records=val_records or None,
    )

    return {
        "backend": "trl_sft",
        "model_name": str(model_name),
        "output_dir": str(output_dir),
        "final_model_path": str(final_path),
        "train_records": int(len(train_records)),
        "val_records": int(len(val_records)),
    }
