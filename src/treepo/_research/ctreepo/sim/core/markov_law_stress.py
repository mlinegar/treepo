"""
Markov-specific law-stress definitions and re-exports.

DGP-agnostic classification lives in ``law_stress_common``.  This module
keeps the Markov-specific constants (law packages, exact families) and
re-exports everything so existing imports remain valid.
"""
from __future__ import annotations

# Re-export all shared classification machinery (backward compatible)
from treepo._research.ctreepo.sim.core.law_stress_common import (  # noqa: F401
    DEFAULT_LAW_GAIN_THRESHOLD,
    DEFAULT_ROOT_RATIO_LIMIT,
    DEFAULT_SPREAD_GAIN_THRESHOLD,
    LawStressAssessment,
    classify_law_stress,
    infer_law_stress_failure_reason,
    law_bundle_score,
    markov_law_bundle_score,
)

# Markov-specific constants
VALID_LAW_PACKAGES = (
    "root_only",
    "c1_only",
    "c2_only",
    "c3_only",
    "c1c3",
    "all_laws",
    "sched_only",
    "all_laws_plus_sched",
)

VALID_EXACT_FAMILIES = (
    "exact",
    "leaf_bucket",
    "count_only",
    "flip_R2",
)


__all__ = [
    "DEFAULT_LAW_GAIN_THRESHOLD",
    "DEFAULT_ROOT_RATIO_LIMIT",
    "DEFAULT_SPREAD_GAIN_THRESHOLD",
    "LawStressAssessment",
    "VALID_EXACT_FAMILIES",
    "VALID_LAW_PACKAGES",
    "classify_law_stress",
    "infer_law_stress_failure_reason",
    "law_bundle_score",
    "markov_law_bundle_score",
]
