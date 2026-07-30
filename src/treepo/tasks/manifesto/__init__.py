"""Minimal manifesto/RILE helpers for package examples."""

from treepo.tasks.manifesto.components import (
    CMP_POLICY_CODES,
    MANIFESTO_RILE_COMPONENT_SCHEMA_VERSION,
    MANIFESTO_RILE_NORMALIZED_SCHEMA_VERSION,
    RILE_CMP56_TARGET_NAMES,
    RILE_LEFT_CODES,
    RILE_NORMALIZED_ORACLE_ID,
    RILE_NORMALIZED_TARGET_KEY,
    RILE_NORMALIZED_TARGET_NAME,
    RILE_POLARITY_TARGET_NAMES,
    RILE_RIGHT_CODES,
    attach_manifesto_rile_components,
    manifesto_rile_component_fit_fragment,
    manifesto_rile_component_output_schema,
    manifesto_rile_component_readout,
    manifesto_rile_component_report,
    manifesto_rile_component_target_key,
    manifesto_rile_component_target_names,
    manifesto_rile_component_targets,
    manifesto_rile_components_from_codes,
    manifesto_rile_components_from_counts,
    manifesto_rile_normalized_fit_fragment,
    normalize_cmp_code,
)
from treepo.tasks.manifesto.documents import (
    DEFAULT_MANIFESTO_REPLICATIONS,
    ManifestoDocument,
    ManifestoLeaf,
    ManifestoQSentence,
    ManifestoReplicationTree,
)
from treepo.tasks.manifesto.exports import export_manifesto_reward_views
from treepo.tasks.manifesto.full_document import (
    MANIFESTO_RILE_MASS_SCHEMA_VERSION,
    manifesto_rile_mass_output_schema,
    manifesto_rile_mass_readout,
    manifesto_rile_mass_state,
    manifesto_rile_mass_state_from_value,
)
from treepo.tasks.manifesto.preferences import make_manifesto_preferences
from treepo.tasks.manifesto.prompts import (
    manifesto_oracle_predict_fn,
    manifesto_prompt_template,
)
from treepo.tasks.manifesto.rile import RILE_MAX, RILE_MIN, RILE_RANGE, clamp_rile
from treepo.tasks.manifesto.sampling import (
    manifesto_document_unit_sampling_rows,
    sample_manifesto_replication_trees,
)
from treepo.tasks.manifesto.semantic_forest_reporting import (
    RILE_SEMANTIC_FOREST_COMPARISON_SCHEMA_VERSION,
    manifesto_semantic_forest_comparison_report_schema,
    validate_manifesto_semantic_forest_comparison_row,
)
from treepo.tasks.manifesto.state import (
    MANIFESTO_POLICY_STATE_KIND,
    ManifestoPolicyStatistic,
    manifesto_policy_state_from_leaf,
)
from treepo.tasks.manifesto.trees import (
    make_manifesto_replication_trees,
    manifesto_tree_records,
    replication_payload,
)

__all__ = [
    "CMP_POLICY_CODES",
    "MANIFESTO_RILE_COMPONENT_SCHEMA_VERSION",
    "MANIFESTO_RILE_NORMALIZED_SCHEMA_VERSION",
    "MANIFESTO_POLICY_STATE_KIND",
    "MANIFESTO_RILE_MASS_SCHEMA_VERSION",
    "RILE_MAX",
    "RILE_MIN",
    "RILE_RANGE",
    "RILE_CMP56_TARGET_NAMES",
    "RILE_LEFT_CODES",
    "RILE_POLARITY_TARGET_NAMES",
    "RILE_NORMALIZED_ORACLE_ID",
    "RILE_NORMALIZED_TARGET_KEY",
    "RILE_NORMALIZED_TARGET_NAME",
    "RILE_RIGHT_CODES",
    "RILE_SEMANTIC_FOREST_COMPARISON_SCHEMA_VERSION",
    "attach_manifesto_rile_components",
    "ManifestoPolicyStatistic",
    "manifesto_rile_component_fit_fragment",
    "manifesto_rile_normalized_fit_fragment",
    "manifesto_rile_component_output_schema",
    "manifesto_rile_component_readout",
    "manifesto_rile_component_report",
    "manifesto_rile_component_target_key",
    "manifesto_rile_component_target_names",
    "manifesto_rile_component_targets",
    "manifesto_rile_components_from_codes",
    "manifesto_rile_components_from_counts",
    "manifesto_rile_mass_output_schema",
    "manifesto_rile_mass_readout",
    "manifesto_rile_mass_state",
    "manifesto_rile_mass_state_from_value",
    "manifesto_semantic_forest_comparison_report_schema",
    "manifesto_policy_state_from_leaf",
    "DEFAULT_MANIFESTO_REPLICATIONS",
    "ManifestoDocument",
    "ManifestoLeaf",
    "ManifestoQSentence",
    "ManifestoReplicationTree",
    "clamp_rile",
    "export_manifesto_reward_views",
    "make_manifesto_preferences",
    "make_manifesto_replication_trees",
    "manifesto_document_unit_sampling_rows",
    "manifesto_oracle_predict_fn",
    "manifesto_prompt_template",
    "manifesto_tree_records",
    "normalize_cmp_code",
    "replication_payload",
    "sample_manifesto_replication_trees",
    "validate_manifesto_semantic_forest_comparison_row",
]
