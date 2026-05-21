"""treepo: paper-facing TreePO / C-TreePO package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from treepo.core import (
    BenchmarkRef,
    ExperimentContext,
    MethodRef,
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
from treepo.hll import (
    HLLConfig,
    HyperLogLogSketch,
    hll_relative_standard_error,
    match_hll_precision_for_bits,
    reduce_hll_sketches,
)

try:
    __version__ = version("treepo")
except (PackageNotFoundError, TypeError, KeyError):  # pragma: no cover
    __version__ = "0.1.0"

_LAZY_EXPORTS = {
    "CardinalityRecoveryConfig": (
        "treepo.bench.cardinality_recovery",
        "CardinalityRecoveryConfig",
    ),
    "CardinalityRecoverySummary": (
        "treepo.bench.cardinality_recovery",
        "CardinalityRecoverySummary",
    ),
    "run_cardinality_recovery_experiment": (
        "treepo.bench.cardinality_recovery",
        "run_cardinality_recovery_experiment",
    ),
    "HLLMergeLearningConfig": (
        "treepo.bench.hll_merge_learning",
        "HLLMergeLearningConfig",
    ),
    "HLLMergeLearningSummary": (
        "treepo.bench.hll_merge_learning",
        "HLLMergeLearningSummary",
    ),
    "run_hll_merge_learning_experiment": (
        "treepo.bench.hll_merge_learning",
        "run_hll_merge_learning_experiment",
    ),
    "FitConfig": ("treepo.learning", "FitConfig"),
    "FitResult": ("treepo.learning", "FitResult"),
    "fit": ("treepo.learning", "fit"),
    # --- treepo.cld surface (the unified fit() / run() axis-factored API) ---
    # Source moved from the parallel ``treepo_cld`` package; ``treepo_cld``
    # is now a thin re-export shim. New code should use ``treepo.X`` (or
    # ``treepo.cld.X`` for the full namespace).
    "run": ("treepo.cld", "run"),
    "list_methods": ("treepo.cld", "list_methods"),
    "method_info": ("treepo.cld", "method_info"),
    "allowed_config_keys": ("treepo.cld", "allowed_config_keys"),
    "register_method": ("treepo.cld", "register_method"),
    "list_registered_oracles": ("treepo.cld", "list_registered_oracles"),
    "list_sketch_kinds": ("treepo.cld", "list_sketch_kinds"),
    "list_oracle_domains_with_fixtures": (
        "treepo.cld", "list_oracle_domains_with_fixtures",
    ),
    "load_dataclass": ("treepo.cld", "load_dataclass"),
    "build_lm_config_dict": ("treepo.cld", "build_lm_config_dict"),
    "LmSection": ("treepo.cld", "LmSection"),
    "HllSketchConfig": ("treepo.cld", "HllSketchConfig"),
    "LdaOracleConfig": ("treepo.cld", "LdaOracleConfig"),
    "LawKind": ("treepo.cld", "LawKind"),
    "LocalLawAuditRow": ("treepo.cld", "LocalLawAuditRow"),
    "LocalLawObjectiveSummary": ("treepo.cld", "LocalLawObjectiveSummary"),
    "InfluenceWeightedAuditOverlap": ("treepo.cld", "InfluenceWeightedAuditOverlap"),
    "compute_influence_weighted_overlap": (
        "treepo.cld", "compute_influence_weighted_overlap",
    ),
    "corrected_local_law_loss": ("treepo.cld", "corrected_local_law_loss"),
    "local_law_objective_summary": ("treepo.cld", "local_law_objective_summary"),
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
    "CardinalityRecoveryConfig",
    "CardinalityRecoverySummary",
    "ExperimentContext",
    "FitConfig",
    "FitResult",
    "HLLConfig",
    "HLLMergeLearningConfig",
    "HLLMergeLearningSummary",
    "HyperLogLogSketch",
    "MethodRef",
    "ROLE_EMBEDDER",
    "ROLE_ORACLE",
    "ROLE_SCORER",
    "ROLE_STATE_MODEL",
    "ROLE_SUMMARIZER",
    "ResultRow",
    "RoleRef",
    "SamplingPlan",
    "hll_relative_standard_error",
    "match_hll_precision_for_bits",
    "reduce_hll_sketches",
    "role_ref",
    "roles_metadata",
    "fit",
    "run_cardinality_recovery_experiment",
    "run_hll_merge_learning_experiment",
    # treepo.cld surface
    "run",
    "list_methods",
    "method_info",
    "allowed_config_keys",
    "register_method",
    "list_registered_oracles",
    "list_sketch_kinds",
    "list_oracle_domains_with_fixtures",
    "load_dataclass",
    "build_lm_config_dict",
    "LmSection",
    "HllSketchConfig",
    "LdaOracleConfig",
    "LawKind",
    "LocalLawAuditRow",
    "LocalLawObjectiveSummary",
    "InfluenceWeightedAuditOverlap",
    "compute_influence_weighted_overlap",
    "corrected_local_law_loss",
    "local_law_objective_summary",
]
