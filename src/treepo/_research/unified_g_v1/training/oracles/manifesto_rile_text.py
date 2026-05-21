"""`TreeOracle` adapters for Manifesto RILE data.

`ManifestoRileTextOracle` wraps a `text_pairs_v1` `PreparedDataset` for the
text / TRL-SFT lane. It declares `space_kind="text"` in its metadata so the
default trainer resolution picks `trl_sft_trainer`.

The "tree" here is degenerate — a single leaf per example — because TRL SFT
is sequence-level. This is still the right abstraction: the oracle produces
examples, the trainer consumes them, and `fit()` doesn't care how many
leaves are in each example.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo._research.unified_g_v1.training.prepared_dataset import PreparedDataset
from treepo._research.unified_g_v1.training.tree_task import TreeExample


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            yield json.loads(raw)


def _load_split(dataset: PreparedDataset, split: str) -> list[TreeExample]:
    items: list[TreeExample] = []
    for shard in dataset.shard_paths(split):
        for record in _iter_jsonl(shard):
            prompt = str(record.get("prompt", ""))
            completion = str(record.get("completion", ""))
            if not prompt or not completion:
                continue
            items.append(
                TreeExample(
                    leaves=[prompt],
                    target=completion,
                    extra={"doc_id": record.get("doc_id", ""), "target_raw": record.get("target")},
                )
            )
    return items


@dataclass
class ManifestoRileTextOracle:
    """Text oracle backed by a `text_pairs_v1` prepared dataset."""

    prepared_dataset: PreparedDataset
    _train_cache: list[TreeExample] = field(default_factory=list, init=False, repr=False)
    _val_cache: list[TreeExample] = field(default_factory=list, init=False, repr=False)

    @classmethod
    def from_path(cls, path: str | Path) -> "ManifestoRileTextOracle":
        return cls(prepared_dataset=PreparedDataset.load(path))

    def __post_init__(self) -> None:
        schema = self.prepared_dataset.payload_schema
        if schema != "text_pairs_v1":
            raise ValueError(
                f"ManifestoRileTextOracle expects payload_schema=text_pairs_v1, got {schema!r}"
            )

    def train_examples(self) -> Sequence[TreeExample]:
        if not self._train_cache:
            self._train_cache = _load_split(self.prepared_dataset, "train")
        return self._train_cache

    def val_examples(self) -> Sequence[TreeExample]:
        if not self._val_cache and self.prepared_dataset.has_split("val"):
            self._val_cache = _load_split(self.prepared_dataset, "val")
        return self._val_cache

    def metadata(self) -> Mapping[str, Any]:
        return {
            "oracle": "manifesto_rile_text",
            "space_kind": "text",
            "payload_schema": self.prepared_dataset.payload_schema,
            "prepared_dataset_root": str(self.prepared_dataset.root),
        }
