"""Sequenced learned-sketch trainer — registers the Pass-4 sequenced trainer.

The implementation lives in :mod:`unified_g_v1.sketch.learned_scalar_sketch` so
the model, single-stage task factory, and sequenced trainer stay co-located.
This module exists only to register the trainer under the name
``"learned_sketch_sequence"`` in :data:`TRAINER_REGISTRY` so callers that
prefer string lookup (or that resolve via a config field) can pick it up.
"""
from __future__ import annotations

from treepo._research.unified_g_v1.sketch.learned_scalar_sketch import sequenced_learned_sketch_trainer
from treepo._research.unified_g_v1.training.trainers import register_trainer


register_trainer("learned_sketch_sequence", sequenced_learned_sketch_trainer)
