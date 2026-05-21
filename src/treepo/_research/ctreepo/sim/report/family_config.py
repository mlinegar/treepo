"""Family-specific configuration for unified reports.

Each simulation family (Markov, LDA, ...) provides a frozen config that drives
the unified report scripts.  All family-specific display names, law packages,
axis labels, and colour palettes live here so that the report logic itself is
family-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from treepo._research.ctreepo.contracts import (
    LAW_ID_LEAF_PRESERVATION,
    LAW_ID_MERGE_PRESERVATION,
    LAW_ID_ON_RANGE_IDEMPOTENCE,
)

@dataclass(frozen=True)
class FamilyReportConfig:
    """Everything the unified report scripts need to know about a family."""

    # ── identity ──────────────────────────────────────────────────────
    family: str  # canonical name used in JSON payloads
    display_name: str  # human-readable label

    # ── law packages ──────────────────────────────────────────────────
    valid_law_packages: Tuple[str, ...]
    expected_main_package: str
    fallback_main_package: str
    package_labels: Dict[str, str]
    package_colors: Dict[str, str]

    # ── primary metric ────────────────────────────────────────────────
    primary_metric_label: str  # e.g. "Root MAE" or "Utility error"
    primary_metric_direction: str  # "lower_is_better"

    # ── law-stress heatmap axes ───────────────────────────────────────
    heatmap_row_field: str
    heatmap_col_field: str
    heatmap_row_label: str
    heatmap_col_label: str

    # ── learnability sweep axes ───────────────────────────────────────
    sweep_field: str  # x-axis variable ("local_law_weight" / "tau")
    sweep_label: str  # display label for sweep variable
    sweep_group_fields: Tuple[str, ...]  # variables that create separate curves

    # ── normalisation ─────────────────────────────────────────────────
    normalization_scale_field: str  # JSON config key, or "" if none

    # ── baseline semantics ────────────────────────────────────────────
    baseline_field: Optional[str] = None  # no-local-law / task-only baseline axis
    baseline_label: Optional[str] = None
    baseline_value: Optional[float] = None
    required_local_law_weight_fields: Tuple[str, ...] = ()
    disallowed_lambda_interpretations: Tuple[str, ...] = ()

    # ── exact-family counterexamples ──────────────────────────────────
    valid_exact_families: Tuple[str, ...] = ()

    # ── law metric colours (consistent across families) ───────────────
    law_colors: Dict[str, str] = field(
        default_factory=lambda: {
            "c1": "#457b9d",
            "c2": "#e07a5f",
            "c3": "#8d99ae",
        }
    )


# ── Markov ops-count family ──────────────────────────────────────────────

MARKOV_CONFIG = FamilyReportConfig(
    family="markov_ops_count",
    display_name="Markov",
    valid_law_packages=(
        "root_only", "c1_only", "c2_only", "c3_only",
        "c1c3", "all_laws", "sched_only", "all_laws_plus_sched",
    ),
    expected_main_package="all_laws_plus_sched",
    fallback_main_package="all_laws",
    package_labels={
        "root_only": "root only\n(baseline)",
        "c1_only": "C1 only",
        "c2_only": "C2 only",
        "c3_only": "C3 only",
        "c1c3": "C1+C3",
        "all_laws": "C1+C2+C3",
        "sched_only": "sched only",
        "all_laws_plus_sched": "C1+C2+C3\n+sched",
    },
    package_colors={
        "root_only": "#6c757d",
        "c1_only": "#457b9d",
        "c2_only": "#e07a5f",
        "c3_only": "#8d99ae",
        "c1c3": "#2a9d8f",
        "all_laws": "#264653",
        "sched_only": "#f4a261",
        "all_laws_plus_sched": "#1d3557",
    },
    primary_metric_label="Root MAE",
    primary_metric_direction="lower_is_better",
    heatmap_row_field="train_docs",
    heatmap_col_field="audit_fraction",
    heatmap_row_label="Training docs",
    heatmap_col_label="q_audit",
    sweep_field="local_law_weight",
    sweep_label="local_law_weight",
    sweep_group_fields=("train_docs", "audit_fraction"),
    normalization_scale_field="max_segments",
    baseline_field="local_law_weight",
    baseline_label="local_law_weight",
    baseline_value=0.0,
    required_local_law_weight_fields=(
        LAW_ID_LEAF_PRESERVATION,
        LAW_ID_ON_RANGE_IDEMPOTENCE,
        LAW_ID_MERGE_PRESERVATION,
    ),
    valid_exact_families=("exact", "leaf_bucket", "count_only", "flip_R2"),
)


# ── Tree-relevant LDA family ────────────────────────────────────────────

LDA_CONFIG = FamilyReportConfig(
    family="tree_relevant_lda",
    display_name="LDA",
    valid_law_packages=(
        "root_only", "c1_only", "c3_only", "c1c3",
        "c2_only", "all_laws",
    ),
    expected_main_package="all_laws",
    fallback_main_package="all_laws",
    package_labels={
        "root_only": "root only\n(baseline)",
        "c1_only": "C1 only",
        "c2_only": "C2 only",
        "c3_only": "C3 only",
        "c1c3": "C1+C3",
        "all_laws": "C1+C2+C3",
    },
    package_colors={
        "root_only": "#6c757d",
        "c1_only": "#457b9d",
        "c2_only": "#e07a5f",
        "c3_only": "#8d99ae",
        "c1c3": "#2a9d8f",
        "all_laws": "#264653",
    },
    primary_metric_label="Utility error",
    primary_metric_direction="lower_is_better",
    heatmap_row_field="tau",
    heatmap_col_field="analysis_partition_mode",
    heatmap_row_label="τ (mixture concentration)",
    heatmap_col_label="Analysis mode",
    sweep_field="tau",
    sweep_label="τ",
    sweep_group_fields=("quadratic_utility_weight", "train_docs"),
    normalization_scale_field="",
    baseline_field="quadratic_utility_weight",
    baseline_label="quadratic_utility_weight",
    baseline_value=0.0,
    required_local_law_weight_fields=(
        "objective_local_law_weight_c1",
        "objective_local_law_weight_c2_proxy",
        "objective_local_law_weight_c3",
    ),
    disallowed_lambda_interpretations=("dgp_term_multiplier", "quadratic_utility_weight"),
    valid_exact_families=(
        "oracle", "scrambled_topics", "uniform_prior", "adversarial_merge",
    ),
)


_FAMILY_REGISTRY: Dict[str, FamilyReportConfig] = {
    "markov": MARKOV_CONFIG,
    "markov_ops_count": MARKOV_CONFIG,
    "lda": LDA_CONFIG,
    "tree_relevant_lda": LDA_CONFIG,
    "tree_relevant_lda_local_law": LDA_CONFIG,
}


def resolve_family(name: str) -> FamilyReportConfig:
    """Look up a family config by canonical or short name."""
    key = str(name).strip().lower()
    if key not in _FAMILY_REGISTRY:
        valid = sorted(set(_FAMILY_REGISTRY.keys()))
        raise ValueError(f"Unknown family {name!r}; valid: {valid}")
    return _FAMILY_REGISTRY[key]


__all__ = [
    "FamilyReportConfig",
    "LDA_CONFIG",
    "MARKOV_CONFIG",
    "resolve_family",
]
