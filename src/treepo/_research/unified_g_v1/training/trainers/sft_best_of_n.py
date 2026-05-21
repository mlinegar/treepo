"""`sft_best_of_n` — online trainer: generate N candidates, user picks best.

General HITL/best-of-N data-collection trainer. For each training prompt:
  1. `cfg.base_module(prompt)` is called `cfg.candidates_per_example` times
     to produce candidate completions.
  2. `cfg.feedback_fn(prompt, candidates)` returns either the index of the
     preferred candidate (int) or a ranking (list of ints; head is best).
  3. The (prompt, best_candidate) pair is written to the run's output dir
     as a `text_pairs_v1` JSONL shard — consumable by `trl_sft` downstream.

Use this to bootstrap SFT data when you have a generator and a ranker (human
or judge model) but no gold completions. The resulting dataset can feed
directly into the `trl_sft` trainer via a `PreparedDataset`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from treepo._research.unified_g_v1.training.prepared_dataset import (
    DATASET_MANIFEST_VERSION,
    MANIFEST_FILENAME,
    write_dataset_manifest,
)
from treepo._research.unified_g_v1.training.trainers import register_trainer


def _best_index(feedback_result: Any) -> int:
    """feedback_fn may return an int index or a ranking list (best first)."""
    if isinstance(feedback_result, int):
        return int(feedback_result)
    if isinstance(feedback_result, Sequence) and len(feedback_result) > 0:
        return int(feedback_result[0])
    raise ValueError(
        f"feedback_fn must return an int index or a non-empty ranking list, got {feedback_result!r}"
    )


def _iter_prompts(cfg) -> list[tuple[str, Any]]:
    """Extract (prompt, target) pairs from cfg.oracle.train_examples().

    Each TreeExample's `leaves[0]` is treated as the prompt; `target` is the
    gold completion if known (otherwise None). This matches the convention
    used by `ManifestoRileTextOracle`.
    """
    items: list[tuple[str, Any]] = []
    for example in cfg.oracle.train_examples():
        prompt = example.leaves[0] if example.leaves else ""
        items.append((str(prompt), example.target))
    return items


def sft_best_of_n_trainer(cfg, output_dir: Path, dataset=None):
    del dataset
    from treepo._research.unified_g_v1.training.fit import FitResult

    if cfg.base_module is None or not callable(cfg.base_module):
        raise ValueError("sft_best_of_n_trainer requires callable cfg.base_module")
    if cfg.feedback_fn is None or not callable(cfg.feedback_fn):
        raise ValueError("sft_best_of_n_trainer requires callable cfg.feedback_fn")
    if cfg.oracle is None:
        raise ValueError("sft_best_of_n_trainer requires cfg.oracle")
    if int(cfg.candidates_per_example) < 1:
        raise ValueError("candidates_per_example must be >= 1")

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shards_dir / "train-0000.jsonl"

    n_candidates = int(cfg.candidates_per_example)
    items = _iter_prompts(cfg)

    agree_with_target = 0
    known_target = 0
    records: list[dict[str, Any]] = []
    with open(shard_path, "w", encoding="utf-8") as fh:
        for prompt, gold in items:
            candidates = [cfg.base_module(prompt) for _ in range(n_candidates)]
            best_idx = _best_index(cfg.feedback_fn(prompt, candidates))
            if not (0 <= best_idx < n_candidates):
                raise ValueError(f"feedback_fn returned out-of-range index {best_idx}")
            best = candidates[best_idx]
            record = {
                "prompt": prompt,
                "completion": str(best),
                "n_candidates": n_candidates,
                "chosen_index": best_idx,
            }
            if gold is not None:
                record["gold"] = str(gold)
                known_target += 1
                if str(best) == str(gold):
                    agree_with_target += 1
            fh.write(json.dumps(record) + "\n")
            records.append(record)

    oracle_meta = dict(cfg.oracle.metadata()) if cfg.oracle is not None else {}
    write_dataset_manifest(
        output_dir,
        space_kind="text",
        payload_schema="text_pairs_v1",
        shards={"train": [str(shard_path.relative_to(output_dir).as_posix())]},
        provenance={
            "trainer": "sft_best_of_n",
            "candidates_per_example": n_candidates,
            "oracle_metadata": oracle_meta,
        },
    )
    summary = {
        "backend": "sft_best_of_n",
        "n_prompts": len(items),
        "candidates_per_example": n_candidates,
        "agree_with_gold": agree_with_target,
        "known_targets": known_target,
        "agreement_rate": (
            float(agree_with_target) / float(known_target) if known_target else None
        ),
        "shard_path": str(shard_path),
        "manifest_path": str(output_dir / MANIFEST_FILENAME),
        "prepared_dataset_root": str(output_dir),
    }
    metrics: dict[str, float] = {
        "n_prompts": float(summary["n_prompts"]),
        "candidates_per_example": float(summary["candidates_per_example"]),
    }
    if summary.get("agreement_rate") is not None:
        metrics["agreement_rate"] = float(summary["agreement_rate"])
    return FitResult(
        backend="sft_best_of_n",
        summary=summary,
        status="completed",
        metrics=metrics,
        artifacts={
            "shard_path": str(shard_path),
            "prepared_dataset_root": str(output_dir),
        },
    )


register_trainer("sft_best_of_n", sft_best_of_n_trainer)
