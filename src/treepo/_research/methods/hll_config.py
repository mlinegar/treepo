"""Research-only HLL sketch config."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HllSketchConfig:
    """HLL adapter and fixture knobs for research sketch helpers."""

    backend: str = "native"
    precision: int = 14
    hash_bits: int = 64
    schedule: str = "balanced"
    n_trees: int = 6
    leaves_per_tree: int = 4
    leaf_token_count: int = 24
    vocabulary_size: int = 200
    seed: int = 0


__all__ = ["HllSketchConfig"]
