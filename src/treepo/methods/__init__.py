"""treepo.methods: unified fit() / run() surface for paper exercises."""

from __future__ import annotations

from treepo.methods import canonical_defaults
from treepo.methods.canonical_defaults import (
    build_lm_config_dict,
    LmSection,
    load_dataclass,
)
from treepo.methods.learning import fit
from treepo.local_law import (
    InfluenceWeightedAuditOverlap,
    LawKind,
    LocalLawAuditRow,
    LocalLawObjectiveSummary,
    compute_influence_weighted_overlap,
    corrected_local_law_loss,
    local_law_objective_summary,
)
from treepo.methods.dispatch import (
    allowed_config_keys,
    list_methods,
    list_oracle_domains_with_fixtures,
    list_registered_oracles,
    method_info,
    register_method,
    run,
)

__all__ = [
    # Centralized dispatch surface (3 axes — names are strings: "fit", "oracle", "audit").
    "run",
    "list_methods",
    "method_info",
    "allowed_config_keys",
    "register_method",
    "list_oracle_domains_with_fixtures",
    "list_registered_oracles",
    # Underlying fit() and law surface for advanced use.
    "fit",
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
