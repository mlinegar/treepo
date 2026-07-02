"""Synthetic Markov changepoint fixture."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np

from treepo.methods.fixtures.common import (
    exact_score_metadata,
    leaf_slices,
    require_cuda_torch,
)


@dataclass
class _MarkovLeaf:
    tokens: Sequence[int]
    regimes: Sequence[int]


@dataclass
class MarkovChangepointTree:
    """Tree-shaped Markov sequence with a changepoint-count teacher."""

    leaves: Tuple[_MarkovLeaf, ...]
    tokens: Sequence[int]
    regimes: Sequence[int]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@functools.lru_cache(maxsize=32)
def make_markov_changepoint_trees(
    *,
    n_trees: int = 8,
    n_states: int = 4,
    doc_tokens: int = 128,
    leaf_unit_count: int = 16,
    doc_unit_kind: str = "token",
    transition_prob: float = 0.15,
    vocabulary_size: int = 256,
    seed: int = 0,
    split: str = "test",
    generation_device: str = "cpu",
) -> List[MarkovChangepointTree]:
    """Generate deterministic Markov trees with exact changepoint targets."""

    if n_trees <= 0 or n_states <= 1 or doc_tokens <= 0 or leaf_unit_count <= 0:
        raise ValueError(
            "n_trees, doc_tokens, leaf_unit_count must be positive and n_states > 1"
        )
    if not 0.0 <= float(transition_prob) <= 1.0:
        raise ValueError("transition_prob must be in [0, 1]")

    if str(generation_device).startswith("cuda"):
        return _make_markov_changepoint_trees_torch(
            n_trees=int(n_trees),
            n_states=int(n_states),
            doc_tokens=int(doc_tokens),
            leaf_unit_count=int(leaf_unit_count),
            doc_unit_kind=str(doc_unit_kind),
            transition_prob=float(transition_prob),
            vocabulary_size=int(vocabulary_size),
            seed=int(seed),
            split=str(split),
            generation_device=str(generation_device),
        )

    rng = np.random.default_rng(int(seed))
    bucket = max(1, int(vocabulary_size) // int(n_states))
    trees: List[MarkovChangepointTree] = []
    for tree_idx in range(int(n_trees)):
        regimes: List[int] = [int(rng.integers(0, int(n_states)))]
        for _ in range(1, int(doc_tokens)):
            cur = int(regimes[-1])
            if float(rng.random()) < float(transition_prob):
                step = int(rng.integers(1, int(n_states)))
                regimes.append(int((cur + step) % int(n_states)))
            else:
                regimes.append(cur)

        tokens: List[int] = []
        for state in regimes:
            low = int(state) * bucket
            high = min(int(vocabulary_size), low + bucket)
            if high <= low:
                tokens.append(int(state))
            else:
                tokens.append(int(rng.integers(low, high)))

        leaves: List[_MarkovLeaf] = []
        for start, end in leaf_slices(doc_tokens, leaf_unit_count):
            leaves.append(
                _MarkovLeaf(
                    tokens=tuple(int(x) for x in tokens[start:end]),
                    regimes=tuple(int(x) for x in regimes[start:end]),
                )
            )
        changepoints = sum(
            1 for left, right in zip(regimes, regimes[1:]) if int(left) != int(right)
        )
        trees.append(
            MarkovChangepointTree(
                leaves=tuple(leaves),
                tokens=tuple(tokens),
                regimes=tuple(regimes),
                metadata={
                    "tree_id": f"markov_{int(seed)}_{tree_idx}",
                    "split": split,
                    **exact_score_metadata(changepoints, target_scale="raw"),
                    "n_states": int(n_states),
                    "doc_tokens": int(doc_tokens),
                    "doc_unit_kind": str(doc_unit_kind),
                    "leaf_unit_count": int(leaf_unit_count),
                    "vocabulary_size": int(vocabulary_size),
                    "transition_prob": float(transition_prob),
                },
            )
        )
    return trees


def _make_markov_changepoint_trees_torch(
    *,
    n_trees: int,
    n_states: int,
    doc_tokens: int,
    leaf_unit_count: int,
    doc_unit_kind: str,
    transition_prob: float,
    vocabulary_size: int,
    seed: int,
    split: str,
    generation_device: str,
) -> List[MarkovChangepointTree]:
    torch, device, rng_devices = require_cuda_torch(
        generation_device, fixture_name="Markov"
    )

    bucket = max(1, int(vocabulary_size) // int(n_states))
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        initial = torch.randint(0, int(n_states), (int(n_trees), 1), device=device)
        if int(doc_tokens) > 1:
            changes = torch.rand(
                (int(n_trees), int(doc_tokens) - 1), device=device
            ) < float(transition_prob)
            steps = torch.randint(
                1,
                int(n_states),
                (int(n_trees), int(doc_tokens) - 1),
                device=device,
            )
            increments = torch.where(changes, steps, torch.zeros_like(steps))
            regimes_t = torch.cat(
                [initial, (initial + torch.cumsum(increments, dim=1)) % int(n_states)],
                dim=1,
            )
        else:
            regimes_t = initial
        offsets = torch.randint(0, bucket, regimes_t.shape, device=device)
        tokens_t = regimes_t * int(bucket) + offsets
        if int(vocabulary_size) < int(n_states) * int(bucket):
            tokens_t = torch.clamp(tokens_t, max=int(vocabulary_size) - 1)
        changepoints_t = (
            (regimes_t[:, 1:] != regimes_t[:, :-1]).sum(dim=1)
            if int(doc_tokens) > 1
            else torch.zeros((int(n_trees),), device=device, dtype=torch.long)
        )

    regimes_np = regimes_t.to(device="cpu", dtype=torch.int64).numpy()
    tokens_np = tokens_t.to(device="cpu", dtype=torch.int64).numpy()
    changepoints_np = changepoints_t.to(device="cpu", dtype=torch.int64).numpy()
    trees: List[MarkovChangepointTree] = []
    for tree_idx in range(int(n_trees)):
        regimes = regimes_np[tree_idx]
        tokens = tokens_np[tree_idx]
        leaves: List[_MarkovLeaf] = []
        for start, end in leaf_slices(doc_tokens, leaf_unit_count):
            leaves.append(
                _MarkovLeaf(
                    tokens=tokens[start:end],
                    regimes=regimes[start:end],
                )
            )
        changepoints = int(changepoints_np[tree_idx])
        trees.append(
            MarkovChangepointTree(
                leaves=tuple(leaves),
                tokens=tokens,
                regimes=regimes,
                metadata={
                    "tree_id": f"markov_{int(seed)}_{tree_idx}",
                    "split": split,
                    **exact_score_metadata(changepoints, target_scale="raw"),
                    "n_states": int(n_states),
                    "doc_tokens": int(doc_tokens),
                    "doc_unit_kind": str(doc_unit_kind),
                    "leaf_unit_count": int(leaf_unit_count),
                    "vocabulary_size": int(vocabulary_size),
                    "transition_prob": float(transition_prob),
                },
            )
        )
    return trees


__all__ = ["MarkovChangepointTree", "make_markov_changepoint_trees"]
