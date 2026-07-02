"""HLL cardinality fixture."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Tuple

import numpy as np

from treepo.methods.fixtures.common import exact_score_metadata, int_tuple


@dataclass
class _HLLLeaf:
    tokens: Tuple[int, ...]


@dataclass
class HLLItemTree:
    """Tree-shaped object with per-leaf item sequences and an exact
    distinct-count teacher on metadata."""

    leaves: Tuple[_HLLLeaf, ...]
    tokens: Tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@functools.lru_cache(maxsize=32)
def make_hll_item_trees(
    *,
    n_trees: int = 8,
    leaves_per_tree: int = 4,
    leaf_unit_count: int = 16,
    doc_unit_kind: str = "item",
    vocabulary_size: int = 64,
    seed: int = 0,
    split: str = "test",
) -> List[HLLItemTree]:
    """Generate deterministic item trees with precomputed exact unique counts."""
    if n_trees <= 0 or leaves_per_tree <= 0 or leaf_unit_count <= 0:
        raise ValueError("n_trees, leaves_per_tree, leaf_unit_count must be positive")
    rng = np.random.default_rng(int(seed))
    trees: List[HLLItemTree] = []
    for _ in range(int(n_trees)):
        leaves: List[_HLLLeaf] = []
        all_tokens: List[int] = []
        for _leaf_idx in range(int(leaves_per_tree)):
            leaf_tokens = rng.integers(
                low=0, high=int(vocabulary_size), size=int(leaf_unit_count)
            ).tolist()
            leaves.append(_HLLLeaf(tokens=int_tuple(leaf_tokens)))
            all_tokens.extend(int(t) for t in leaf_tokens)
        exact_unique = int(len(set(all_tokens)))
        trees.append(
            HLLItemTree(
                leaves=tuple(leaves),
                tokens=tuple(all_tokens),
                metadata={
                    "split": split,
                    **exact_score_metadata(exact_unique, target_scale="raw"),
                    "doc_unit_kind": str(doc_unit_kind),
                    "leaf_unit_count": int(leaf_unit_count),
                },
            )
        )
    return trees


__all__ = ["HLLItemTree", "make_hll_item_trees"]
