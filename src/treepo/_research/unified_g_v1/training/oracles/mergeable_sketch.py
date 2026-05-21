"""Synthetic bigram-sketch oracle. Declares `space_kind="numeric_sequence"`.

Each example carries everything the objective needs to enforce local laws
at every node of the tree:

* `leaves` — per-leaf flat sketch tensors (input to the model).
* `extra["analytic_root"]` — full analytic root sketch (C2 reconstruction target).
* `extra["leaf_scalar_targets"]` — per-leaf target-bigram counts (C1).
* `extra["merge_scalar_targets"]` — per-internal-node target-bigram counts (C3),
  computed as the left-to-right cumulative merge of leaves [0..i+1] for i in
  0..n_leaves-2. The last entry equals the root target.
* `extra["analytic_merge_sketches"]` — the analytic merge sketch at each
  internal node (optional C2 reconstruction target per merge).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from treepo._research.unified_g_v1.sketch.sketch_data import (
    BigramSketch,
    SketchSyntheticConfig,
    SketchTreeExample,
    generate_sketch_dataset,
    merge_sketches,
)
from treepo._research.unified_g_v1.training.tree_task import TreeExample


def _cumulative_merges(leaves: Sequence[BigramSketch]) -> list[BigramSketch]:
    """Left-to-right cumulative merges. Returns one sketch per internal node.

    For leaves `[l0, l1, ..., l_{n-1}]` returns `[merge(l0,l1), merge(merge(l0,l1),l2), ...]`
    of length `n-1`. The last entry is the full-root sketch.
    """
    if len(leaves) <= 1:
        return []
    out: list[BigramSketch] = []
    running = leaves[0]
    for leaf in leaves[1:]:
        running = merge_sketches(running, leaf)
        out.append(running)
    return out


@dataclass
class MergeableSketchOracle:
    config: SketchSyntheticConfig
    train_docs: int
    val_docs: int
    seed: int

    def _to_tree_example(self, item: SketchTreeExample) -> TreeExample:
        leaves = [
            leaf.as_flat_tensor(vocab_size=self.config.vocab_size)
            for leaf in item.leaves
        ]
        target_a, target_b = self.config.target_bigram
        leaf_scalars = [
            float(leaf.bigram_counts[int(target_a), int(target_b)].item())
            for leaf in item.leaves
        ]
        analytic_merges = _cumulative_merges(item.leaves)
        merge_scalars = [
            float(sketch.bigram_counts[int(target_a), int(target_b)].item())
            for sketch in analytic_merges
        ]
        extra = {
            "tokens": list(item.tokens),
            "analytic_root": item.root,
            "analytic_merge_sketches": analytic_merges,
            "leaf_scalar_targets": leaf_scalars,
            "merge_scalar_targets": merge_scalars,
        }
        return TreeExample(leaves=leaves, target=float(item.target), extra=extra)

    def train_examples(self) -> Sequence[TreeExample]:
        raw = generate_sketch_dataset(self.config, n_docs=self.train_docs, seed=self.seed)
        return [self._to_tree_example(x) for x in raw]

    def val_examples(self) -> Sequence[TreeExample]:
        raw = generate_sketch_dataset(
            self.config, n_docs=self.val_docs, seed=self.seed + 1000
        )
        return [self._to_tree_example(x) for x in raw]

    def metadata(self) -> Mapping[str, Any]:
        return {
            "oracle": "mergeable_sketch",
            "space_kind": "numeric_sequence",
            "vocab_size": int(self.config.vocab_size),
            "seq_length": int(self.config.seq_length),
            "n_leaves": int(self.config.n_leaves),
            "target_bigram": list(self.config.target_bigram),
        }
