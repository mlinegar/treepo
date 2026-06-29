"""treepo.methods: lightweight fit() / run() surface.

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
    ObjectiveSpec,
)

_LAZY_EXPORTS = {
    "canonical_defaults": ("treepo.methods.canonical_defaults", None),
    "build_lm_config_dict": ("treepo.methods.canonical_defaults", "build_lm_config_dict"),
    "LmSection": ("treepo.methods.canonical_defaults", "LmSection"),
    "load_dataclass": ("treepo.methods.canonical_defaults", "load_dataclass"),
    "fit": ("treepo.methods.learning", "fit"),
    "run": ("treepo.methods.dispatch", "run"),
    "list_methods": ("treepo.methods.dispatch", "list_methods"),
    "method_info": ("treepo.methods.dispatch", "method_info"),
    "allowed_config_keys": ("treepo.methods.dispatch", "allowed_config_keys"),
    "register_method": ("treepo.methods.dispatch", "register_method"),
    "list_oracle_domains_with_fixtures": (
        "treepo.methods.dispatch",
        "list_oracle_domains_with_fixtures",
    ),
    "list_registered_oracles": ("treepo.methods.dispatch", "list_registered_oracles"),
    "list_families": ("treepo.methods.families", "list_families"),
    "resolve_family": ("treepo.methods.families", "resolve_family"),
    "register_family": ("treepo.methods.families", "register_family"),
    "EstimatorSpec": ("treepo.methods.estimators", "EstimatorSpec"),
    "EstimatorDescriptor": ("treepo.methods.estimators", "EstimatorDescriptor"),
    "list_estimators": ("treepo.methods.estimators", "list_estimators"),
    "resolve_estimator": ("treepo.methods.estimators", "resolve_estimator"),
    "register_estimator": ("treepo.methods.estimators", "register_estimator"),
    "PromptedLLMFamily": ("treepo.methods.llm", "PromptedLLMFamily"),
    "PromptedLLMFamilyConfig": ("treepo.methods.llm", "PromptedLLMFamilyConfig"),
    "build_llm_family": ("treepo.methods.llm", "build_llm_family"),
    "GEstimatorSpec": ("treepo.methods.g_estimators", "GEstimatorSpec"),
    "GEstimatorDescriptor": ("treepo.methods.g_estimators", "GEstimatorDescriptor"),
    "list_g_estimators": ("treepo.methods.g_estimators", "list_g_estimators"),
    "resolve_g_estimator": ("treepo.methods.g_estimators", "resolve_g_estimator"),
    "register_g_estimator": ("treepo.methods.g_estimators", "register_g_estimator"),
    "GridCell": ("treepo.methods.grid", "GridCell"),
    "grid_cell_name": ("treepo.methods.grid", "grid_cell_name"),
    "iter_grid": ("treepo.methods.grid", "iter_grid"),
    "write_grid_outputs": ("treepo.methods.grid", "write_grid_outputs"),
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
    # Centralized dispatch surface (3 axes — names are strings: "fit", "oracle", "audit").
    "run",
    "list_methods",
    "method_info",
    "allowed_config_keys",
    "register_method",
    "list_oracle_domains_with_fixtures",
    "list_registered_oracles",
    "list_families",
    "resolve_family",
    "register_family",
    "EstimatorSpec",
    "EstimatorDescriptor",
    "list_estimators",
    "resolve_estimator",
    "register_estimator",
    "PromptedLLMFamily",
    "PromptedLLMFamilyConfig",
    "build_llm_family",
    "GEstimatorSpec",
    "GEstimatorDescriptor",
    "list_g_estimators",
    "resolve_g_estimator",
    "register_g_estimator",
    "GridCell",
    "grid_cell_name",
    "iter_grid",
    "write_grid_outputs",
    # Underlying fit() and law surface for advanced use.
    "fit",
    "CTreePOFitResult",
    "CTreePOLearningSpec",
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
