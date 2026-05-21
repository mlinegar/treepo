"""Objectives — loss + evaluation for tree tasks.

A `TreeObjective` has two methods:
  * `compute_loss(root_state, prediction, batch) -> (loss, n_terms, stats)`
  * `evaluate(model, items, batch_size) -> metrics_dict`

For lightweight cases, pass a plain callable as `TrainerConfig.objective` —
`pytorch_tree_trainer` wraps it in `SimpleObjective` automatically.
"""
from treepo._research.unified_g_v1.training.objectives.manifesto_rile_embedding import (
    ManifestoRileEmbeddingObjective,
)
from treepo._research.unified_g_v1.training.objectives.mergeable_sketch import MergeableSketchObjective
from treepo._research.unified_g_v1.training.objectives.simple import SimpleObjective, as_objective

__all__ = [
    "ManifestoRileEmbeddingObjective",
    "MergeableSketchObjective",
    "SimpleObjective",
    "as_objective",
]
