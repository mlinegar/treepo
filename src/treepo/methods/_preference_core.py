"""Unit-level preference data and trainer export views.

``PreferenceDataset`` is the package data boundary for labels, scored
candidates, ranked candidates, and pairwise preferences. Pairwise DPO rows are
one projection of this data, not the storage model.

This module is now a thin aggregation hub. The implementation is split by
responsibility so each piece reads on its own:

* ``_preference_normalize``: row parsing, scalar/JSON coercion (data-prep).
* ``_preference_views``: supervised/DPO/reward/GRPO projections (data-prep).
* ``_preference_dataset``: the ``Candidate``/``PreferenceRecord``/
  ``PreferenceDataset`` data model.
* ``_preference_io``: dataset IO, export, and view previews.
* ``_preference_tree``: ``TreeRecord``-derived units and tree filtering.
* ``_preference_adapters``: trainer-adapter export fan-out.

Symbols are re-exported here so existing
``from treepo.methods._preference_core import ...`` call sites stay stable.
"""

from __future__ import annotations

from treepo.methods._preference_adapters import export_adapter_views
from treepo.methods._preference_dataset import (
    Candidate,
    PreferenceDataset,
    PreferenceFormat,
    PreferenceRecord,
    PreferenceTarget,
    _prompt_from_pair,
    _record_from_pairwise,
)
from treepo.methods._preference_io import (
    export_preference_records,
    normalize_preference_data,
    summarize_preference_views,
)
from treepo.methods._preference_normalize import (
    _CANDIDATE_FIELDS,
    _TREE_FIELDS,
    _UNIT_FIELDS,
    _bool,
    _hf_candidate_row,
    _hf_unit_row,
    _is_flat_candidate_mapping,
    _is_pairwise_mapping,
    _json_default,
    _json_text,
    _maybe_json,
    _mean,
    _normalize_candidate_row,
    _normalize_unit_row,
    _optional_float,
    _optional_int,
    _optional_str,
    _preferred_ids,
    _rows_from_table,
    _sample_weight,
)
from treepo.methods._preference_tree import (
    _is_root_node,
    _parent_ids_by_node,
    _supervised_candidates,
    _tree_nodes,
    filter_units_for_tree,
    make_unit_id,
    preference_units_from_trees,
)
from treepo.methods._preference_views import (
    _candidate_record,
    _candidate_text,
    _context_text,
    _export_metadata,
    _grpo_ranks,
    _ordered_candidates,
    _pair_candidates,
    _preferred_side,
    _reward_scores,
    _score_sort_value,
    _top_candidates,
)

__all__ = [
    # Public data model and API.
    "Candidate",
    "PreferenceDataset",
    "PreferenceFormat",
    "PreferenceRecord",
    "PreferenceTarget",
    "export_adapter_views",
    "export_preference_records",
    "filter_units_for_tree",
    "make_unit_id",
    "normalize_preference_data",
    "preference_units_from_trees",
    "summarize_preference_views",
    # Re-exported internals kept stable for existing import sites.
    "_CANDIDATE_FIELDS",
    "_TREE_FIELDS",
    "_UNIT_FIELDS",
    "_bool",
    "_candidate_record",
    "_candidate_text",
    "_context_text",
    "_export_metadata",
    "_grpo_ranks",
    "_hf_candidate_row",
    "_hf_unit_row",
    "_is_flat_candidate_mapping",
    "_is_pairwise_mapping",
    "_is_root_node",
    "_json_default",
    "_json_text",
    "_maybe_json",
    "_mean",
    "_normalize_candidate_row",
    "_normalize_unit_row",
    "_optional_float",
    "_optional_int",
    "_optional_str",
    "_ordered_candidates",
    "_pair_candidates",
    "_parent_ids_by_node",
    "_preferred_ids",
    "_preferred_side",
    "_prompt_from_pair",
    "_record_from_pairwise",
    "_reward_scores",
    "_rows_from_table",
    "_sample_weight",
    "_score_sort_value",
    "_supervised_candidates",
    "_top_candidates",
    "_tree_nodes",
]
