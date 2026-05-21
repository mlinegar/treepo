"""treepo.cld: minimal unified fit() / run() surface (parallel workspace).

Implements the plan at ``docs/treepo_unified_fit_plan.md``. Eventual target:
merge into the ``treepo/`` package.
"""

from __future__ import annotations

from treepo.cld import canonical_defaults
from treepo.cld.canonical_defaults import (
    build_lm_config_dict,
    HllSketchConfig,
    LdaOracleConfig,
    LmSection,
    load_dataclass,
)
from treepo.cld.learning import fit
from treepo.cld.local_law import (
    InfluenceWeightedAuditOverlap,
    LawKind,
    LocalLawAuditRow,
    LocalLawObjectiveSummary,
    compute_influence_weighted_overlap,
    corrected_local_law_loss,
    local_law_objective_summary,
)
from treepo.cld.methods import (
    allowed_config_keys,
    list_methods,
    list_oracle_domains_with_fixtures,
    list_registered_oracles,
    list_sketch_kinds,
    method_info,
    register_method,
    run,
)

__all__ = [
    # Centralized dispatch surface (4 axes — names are strings: "fit", "oracle", "sketch", "audit").
    "run",
    "list_methods",
    "method_info",
    "allowed_config_keys",
    "register_method",
    "list_oracle_domains_with_fixtures",
    "list_registered_oracles",
    "list_sketch_kinds",
    # Underlying fit() and law surface for advanced use.
    "fit",
    "InfluenceWeightedAuditOverlap",
    "LawKind",
    "LocalLawAuditRow",
    "LocalLawObjectiveSummary",
    "compute_influence_weighted_overlap",
    "corrected_local_law_loss",
    "local_law_objective_summary",
    # Canonical defaults (see treepo.cld/docs/training_defaults.md).
    "canonical_defaults",
    "build_lm_config_dict",
    "HllSketchConfig",
    "LdaOracleConfig",
    "LmSection",
    "load_dataclass",
]
__version__ = "0.0.4"
