"""Compatibility facade for Manifesto/RILE replication helpers."""

from __future__ import annotations

from treepo.tasks.manifesto.documents import (
    DEFAULT_MANIFESTO_REPLICATIONS,
    ManifestoDocument,
    ManifestoLeaf,
    ManifestoQSentence,
    ManifestoReplicationTree,
)
from treepo.tasks.manifesto.exports import export_manifesto_reward_views
from treepo.tasks.manifesto.preferences import make_manifesto_preferences
from treepo.tasks.manifesto.prompts import (
    manifesto_oracle_predict_fn,
    manifesto_prompt_template,
)
from treepo.tasks.manifesto.sampling import (
    manifesto_document_unit_sampling_rows,
    sample_manifesto_replication_trees,
)
from treepo.tasks.manifesto.trees import (
    make_manifesto_replication_trees,
    manifesto_tree_records,
    replication_payload,
)

__all__ = [
    "DEFAULT_MANIFESTO_REPLICATIONS",
    "ManifestoDocument",
    "ManifestoLeaf",
    "ManifestoQSentence",
    "ManifestoReplicationTree",
    "export_manifesto_reward_views",
    "make_manifesto_preferences",
    "make_manifesto_replication_trees",
    "manifesto_document_unit_sampling_rows",
    "manifesto_oracle_predict_fn",
    "manifesto_prompt_template",
    "manifesto_tree_records",
    "replication_payload",
    "sample_manifesto_replication_trees",
]
