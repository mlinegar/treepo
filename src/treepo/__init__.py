"""treepo: paper-facing TreePO / C-TreePO package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from treepo.core import (
    BenchmarkRef,
    ExperimentContext,
    MethodRef,
    NormalizedOutput,
    ROLE_EMBEDDER,
    ROLE_ORACLE,
    ROLE_SCORER,
    ROLE_STATE_MODEL,
    ROLE_SUMMARIZER,
    ResultRow,
    RoleRef,
    SamplingPlan,
    role_ref,
    roles_metadata,
)
from treepo.local_law import (
    InfluenceWeightedAuditOverlap,
    LawKind,
    LocalLawAuditRow,
    LocalLawObjectiveSummary,
    compute_influence_weighted_overlap,
    corrected_local_law_loss,
    local_law_objective_summary,
)

try:
    __version__ = version("treepo")
except (PackageNotFoundError, TypeError, KeyError):  # pragma: no cover
    __version__ = "0.1.0"

_LAZY_EXPORTS = {
    "FitConfig": ("treepo.learning", "FitConfig"),
    "FitResult": ("treepo.learning", "FitResult"),
    "fit": ("treepo.learning", "fit"),
    # --- treepo.methods surface (the unified fit() / run() axis-factored API) ---
    "run": ("treepo.methods", "run"),
    "list_methods": ("treepo.methods", "list_methods"),
    "list_families": ("treepo.methods", "list_families"),
    "method_info": ("treepo.methods", "method_info"),
    "allowed_config_keys": ("treepo.methods", "allowed_config_keys"),
    "register_method": ("treepo.methods", "register_method"),
    "list_registered_oracles": ("treepo.methods", "list_registered_oracles"),
    "list_oracle_domains_with_fixtures": (
        "treepo.methods", "list_oracle_domains_with_fixtures",
    ),
    "load_dataclass": ("treepo.methods", "load_dataclass"),
    "build_lm_config_dict": ("treepo.methods", "build_lm_config_dict"),
    "LmSection": ("treepo.methods", "LmSection"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "__version__",
    "BenchmarkRef",
    "ExperimentContext",
    "FitConfig",
    "FitResult",
    "MethodRef",
    "NormalizedOutput",
    "ROLE_EMBEDDER",
    "ROLE_ORACLE",
    "ROLE_SCORER",
    "ROLE_STATE_MODEL",
    "ROLE_SUMMARIZER",
    "ResultRow",
    "RoleRef",
    "SamplingPlan",
    "role_ref",
    "roles_metadata",
    "fit",
    # treepo.methods surface
    "run",
    "list_methods",
    "list_families",
    "method_info",
    "allowed_config_keys",
    "register_method",
    "list_registered_oracles",
    "list_oracle_domains_with_fixtures",
    "load_dataclass",
    "build_lm_config_dict",
    "LmSection",
    "LawKind",
    "LocalLawAuditRow",
    "LocalLawObjectiveSummary",
    "InfluenceWeightedAuditOverlap",
    "compute_influence_weighted_overlap",
    "corrected_local_law_loss",
    "local_law_objective_summary",
]
