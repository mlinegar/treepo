"""Oracles — producers of (leaf data, target) examples for tree tasks.

An `Oracle` implements `train_examples()`, `val_examples()`, `metadata()`.
The `metadata()["space_kind"]` drives trainer auto-selection in `fit()`.

Built-in oracles:
  * `MergeableSketchOracle` — synthetic bigram sequences (`numeric_sequence`).
  * `ManifestoRileTextOracle` — Manifesto RILE text pairs (`text`).
  * `ManifestoRileTreeOracle` — Manifesto RILE fixed-tree text scaffolds
     (`tree_text`).
  * `ManifestoRileEmbeddingOracle` — Manifesto RILE embedding-sequence trees
     (`embedding_sequence`).
"""
from treepo._research.unified_g_v1.training.oracles.manifesto_rile_embedding import (
    ManifestoRileEmbeddingOracle,
)
from treepo._research.unified_g_v1.training.oracles.manifesto_rile_text import ManifestoRileTextOracle
from treepo._research.unified_g_v1.training.oracles.manifesto_rile_tree import ManifestoRileTreeOracle
from treepo._research.unified_g_v1.training.oracles.mergeable_sketch import MergeableSketchOracle

__all__ = [
    "ManifestoRileEmbeddingOracle",
    "ManifestoRileTextOracle",
    "ManifestoRileTreeOracle",
    "MergeableSketchOracle",
]
