"""Synthetic bigram-sketch data and analytic sketch operations.

Follows `lean3/FormalProofs/OPT/BigramSketch.lean`: a sketch over a token
sequence is a triple `(bigram_counts, first, last)` augmented with the
segment length. Merging two sketches concatenates bigram counts and adds the
cross-boundary bigram (last_of_left, first_of_right).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

import torch


@dataclass
class BigramSketch:
    """Analytic bigram sketch of a token segment.

    bigram_counts: V x V integer matrix (torch.long).
    first / last: token indices, or -1 for an empty segment.
    length: number of tokens covered.
    """

    bigram_counts: torch.Tensor
    first: int
    last: int
    length: int

    @classmethod
    def from_tokens(cls, tokens: Sequence[int], *, vocab_size: int) -> "BigramSketch":
        counts = torch.zeros((int(vocab_size), int(vocab_size)), dtype=torch.long)
        for a, b in zip(tokens, tokens[1:]):
            counts[int(a), int(b)] += 1
        if not tokens:
            return cls(bigram_counts=counts, first=-1, last=-1, length=0)
        return cls(
            bigram_counts=counts,
            first=int(tokens[0]),
            last=int(tokens[-1]),
            length=int(len(tokens)),
        )

    def as_flat_tensor(self, *, vocab_size: int) -> torch.Tensor:
        """Pack (counts, first_one_hot, last_one_hot, length) as a 1-D float tensor."""
        first_one_hot = torch.zeros(int(vocab_size), dtype=torch.float32)
        last_one_hot = torch.zeros(int(vocab_size), dtype=torch.float32)
        if self.first >= 0:
            first_one_hot[self.first] = 1.0
        if self.last >= 0:
            last_one_hot[self.last] = 1.0
        return torch.cat(
            [
                self.bigram_counts.reshape(-1).float(),
                first_one_hot,
                last_one_hot,
                torch.tensor([float(self.length)], dtype=torch.float32),
            ]
        )


def merge_sketches(left: BigramSketch, right: BigramSketch) -> BigramSketch:
    """Analytic merge, per BigramSketch.lean `bigramSketch_append`."""
    if left.length == 0:
        return right
    if right.length == 0:
        return left
    counts = left.bigram_counts.clone() + right.bigram_counts
    counts[int(left.last), int(right.first)] += 1  # boundary bigram
    return BigramSketch(
        bigram_counts=counts,
        first=int(left.first),
        last=int(right.last),
        length=int(left.length + right.length),
    )


def flat_sketch_dim(vocab_size: int) -> int:
    v = int(vocab_size)
    return v * v + v + v + 1


@dataclass(frozen=True)
class SketchSyntheticConfig:
    vocab_size: int = 8
    seq_length: int = 32
    n_leaves: int = 4
    train_docs: int = 256
    val_docs: int = 64
    seed: int = 0
    target_bigram: tuple[int, int] = (0, 1)


@dataclass
class SketchTreeExample:
    """A single training example: leaf sketches, precomputed root sketch, scalar target."""

    leaves: list[BigramSketch]
    root: BigramSketch
    target: float
    tokens: list[int] = field(default_factory=list)


def generate_sketch_dataset(
    config: SketchSyntheticConfig,
    *,
    n_docs: int,
    seed: int,
) -> list[SketchTreeExample]:
    rng = random.Random(int(seed))
    assert config.seq_length % config.n_leaves == 0, (
        "seq_length must be divisible by n_leaves for this basic setting"
    )
    leaf_len = config.seq_length // config.n_leaves
    target_a, target_b = config.target_bigram
    items: list[SketchTreeExample] = []
    for _ in range(int(n_docs)):
        tokens = [rng.randrange(config.vocab_size) for _ in range(config.seq_length)]
        leaves = [
            BigramSketch.from_tokens(
                tokens[idx * leaf_len : (idx + 1) * leaf_len],
                vocab_size=config.vocab_size,
            )
            for idx in range(config.n_leaves)
        ]
        # Reference root via the analytic merge.
        root = leaves[0]
        for leaf in leaves[1:]:
            root = merge_sketches(root, leaf)
        target = float(root.bigram_counts[target_a, target_b].item())
        items.append(SketchTreeExample(leaves=leaves, root=root, target=target, tokens=tokens))
    return items
