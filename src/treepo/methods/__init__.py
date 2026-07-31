"""treepo.methods: internal fit implementation helpers.

The package-level surface is intentionally lazy. Importing ``treepo.methods``
must not import optional heavy dependencies. Classical sketches and the generic
neural-operator family are bundled; application-backed families can still
register from downstream packages.
"""

from __future__ import annotations

from treepo.methods.contracts import (
    G_MODE_FIXED,
    G_MODE_IDENTITY,
    G_MODE_LEARNED,
    G_MODES,
    CTreePOLearningSpec,
    FamilyRuntime,
    FitResult,
    ObjectiveSpec,
    normalize_g_mode,
)

_LAZY_EXPORTS = {
    "canonical_defaults": ("treepo.methods.canonical_defaults", None),
    "load_dataclass": ("treepo.methods.canonical_defaults", "load_dataclass"),
    "align_model_artifacts": ("treepo.methods.artifact_alignment", "align_model_artifacts"),
    "fit": ("treepo.methods.learning", "fit"),
    "GTrainOutcome": ("treepo.methods.runtime", "GTrainOutcome"),
    "Candidate": ("treepo.methods.preference", "Candidate"),
    "PreferenceRecord": ("treepo.methods.preference", "PreferenceRecord"),
    "PreferenceDataset": ("treepo.methods.preference", "PreferenceDataset"),
    "normalize_preference_data": ("treepo.methods.preference", "normalize_preference_data"),
    "export_preference_records": ("treepo.methods.preference", "export_preference_records"),
    "make_unit_id": ("treepo.methods.preference", "make_unit_id"),
    "preference_units_from_trees": ("treepo.methods.preference", "preference_units_from_trees"),
    "TradeoffCurve": ("treepo.methods.tradeoff", "TradeoffCurve"),
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
    "align_model_artifacts",
    "Candidate",
    "PreferenceRecord",
    "PreferenceDataset",
    "normalize_preference_data",
    "export_preference_records",
    "make_unit_id",
    "preference_units_from_trees",
    "fit",
    "CTreePOLearningSpec",
    "FitResult",
    "FamilyRuntime",
    "GTrainOutcome",
    "G_MODE_FIXED",
    "G_MODE_IDENTITY",
    "G_MODE_LEARNED",
    "G_MODES",
    "ObjectiveSpec",
    "normalize_g_mode",
    # Canonical defaults (see docs/training_defaults.md).
    "canonical_defaults",
    "load_dataclass",
]
