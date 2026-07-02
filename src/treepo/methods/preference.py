"""Compatibility facade for unit-level preference data and export views."""

from __future__ import annotations

from treepo.methods._preference_core import (
    Candidate,
    PreferenceDataset,
    PreferenceFormat,
    PreferenceRecord,
    PreferenceTarget,
    export_adapter_views,
    export_preference_records,
    filter_units_for_tree,
    make_unit_id,
    normalize_preference_data,
    preference_units_from_trees,
    summarize_preference_views,
)

__all__ = [
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
]
