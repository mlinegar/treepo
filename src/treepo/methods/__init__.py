"""treepo.methods: internal fit implementation helpers.

The package-level surface is intentionally lazy. Importing ``treepo.methods``
must not import optional heavy dependencies. Classical sketches and the generic
neural-operator family are bundled; application-backed families can still
register from downstream packages.
"""

from __future__ import annotations

from treepo.local_law import (
    InfluenceWeightedAuditOverlap,
    LawKind,
    LocalLawAuditRow,
    LocalLawObjectiveSummary,
    compute_influence_weighted_overlap,
    corrected_local_law_loss,
    local_law_objective_summary,
)
from treepo.methods.contracts import (
    CTreePOFitResult,
    CTreePOLearningSpec,
    FamilyRuntime,
    FitResult,
    ObjectiveSpec,
)

_LAZY_EXPORTS = {
    "canonical_defaults": ("treepo.methods.canonical_defaults", None),
    "build_lm_config_dict": ("treepo.methods.canonical_defaults", "build_lm_config_dict"),
    "LmSection": ("treepo.methods.canonical_defaults", "LmSection"),
    "load_dataclass": ("treepo.methods.canonical_defaults", "load_dataclass"),
    "fit": ("treepo.methods.learning", "fit"),
    "Candidate": ("treepo.methods.preference", "Candidate"),
    "PreferenceRecord": ("treepo.methods.preference", "PreferenceRecord"),
    "PreferenceDataset": ("treepo.methods.preference", "PreferenceDataset"),
    "normalize_preference_data": ("treepo.methods.preference", "normalize_preference_data"),
    "export_preference_records": ("treepo.methods.preference", "export_preference_records"),
    "make_unit_id": ("treepo.methods.preference", "make_unit_id"),
    "preference_units_from_trees": ("treepo.methods.preference", "preference_units_from_trees"),
    "filter_units_for_tree": ("treepo.methods.preference", "filter_units_for_tree"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value

__all__ = [
    "Candidate",
    "PreferenceRecord",
    "PreferenceDataset",
    "normalize_preference_data",
    "export_preference_records",
    "make_unit_id",
    "preference_units_from_trees",
    "filter_units_for_tree",
    "fit",
    "CTreePOFitResult",
    "CTreePOLearningSpec",
    "FitResult",
    "FamilyRuntime",
    "ObjectiveSpec",
    "InfluenceWeightedAuditOverlap",
    "LawKind",
    "LocalLawAuditRow",
    "LocalLawObjectiveSummary",
    "compute_influence_weighted_overlap",
    "corrected_local_law_loss",
    "local_law_objective_summary",
    # Canonical defaults (see docs/training_defaults.md).
    "canonical_defaults",
    "build_lm_config_dict",
    "LmSection",
    "load_dataclass",
]
