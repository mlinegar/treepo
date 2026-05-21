from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Sequence

from treepo._research.ctreepo.sim.util import safe_float, safe_int


CONTRACT_STATUS_CURRENT = "current"
CONTRACT_STATUS_LOCKED_COMPARATOR = "locked_comparator"
CONTRACT_STATUS_DIAGNOSTIC_ONLY = "diagnostic_only"
CONTRACT_STATUS_LEGACY_QUARANTINED = "legacy_quarantined"
ALLOWED_CONTRACT_STATUSES = frozenset(
    {
        CONTRACT_STATUS_CURRENT,
        CONTRACT_STATUS_LOCKED_COMPARATOR,
        CONTRACT_STATUS_DIAGNOSTIC_ONLY,
        CONTRACT_STATUS_LEGACY_QUARANTINED,
    }
)
HEADLINE_CONTRACT_STATUSES = frozenset(
    {
        CONTRACT_STATUS_CURRENT,
        CONTRACT_STATUS_LOCKED_COMPARATOR,
    }
)
DOWNSTREAM_V3_REQUIRED_FIELDS = (
    "baseline_family",
    "comparison_mode",
    "comparison_semantics",
    "comparison_semantics_label",
    "run_intent_hash",
    "run_intent_validation_status",
    "requested_fixed_leaf_tokens",
    "executed_fixed_leaf_tokens",
    "depth_discount_gamma",
)


def _as_text(value: object) -> str:
    return str(value or "").strip()


def _is_known_invalid_one_leaf_root_only_tree_recipe(
    row: Mapping[str, Any],
) -> bool:
    baseline_family = _as_text(row.get("baseline_family"))
    if baseline_family not in {"tree_neural", "tree_neural_c2", "tree_neural_c2c3"}:
        return False
    package_name = _as_text(row.get("package_name"))
    if not package_name.startswith("full") or package_name == "full100":
        return False
    requested_fixed_leaf_tokens = int(
        safe_int(row.get("requested_fixed_leaf_tokens"), 0)
    )
    executed_fixed_leaf_tokens = int(
        safe_int(row.get("executed_fixed_leaf_tokens"), 0)
    )
    if max(requested_fixed_leaf_tokens, executed_fixed_leaf_tokens) != 128:
        return False
    tree_reference_label = _as_text(row.get("tree_reference_label"))
    if tree_reference_label != "unified_g_full_local_laws_v1":
        return False
    computed_leaf_mass_per_doc = float(
        safe_float(row.get("computed_leaf_mass_per_doc"), float("nan"))
    )
    computed_internal_mass_per_doc = float(
        safe_float(row.get("computed_internal_mass_per_doc"), float("nan"))
    )
    if not (
        math.isfinite(computed_leaf_mass_per_doc)
        and math.isfinite(computed_internal_mass_per_doc)
    ):
        return False
    return (
        computed_leaf_mass_per_doc <= 0.0
        and computed_internal_mass_per_doc <= 0.0
    )


def _is_known_invalid_one_leaf_matched_root_v1_recipe(
    row: Mapping[str, Any],
) -> bool:
    baseline_family = _as_text(row.get("baseline_family"))
    if baseline_family not in {"tree_neural", "tree_neural_c2", "tree_neural_c2c3"}:
        return False
    package_name = _as_text(row.get("package_name"))
    if not package_name.startswith("full") or package_name == "full100":
        return False
    requested_fixed_leaf_tokens = int(
        safe_int(row.get("requested_fixed_leaf_tokens"), 0)
    )
    executed_fixed_leaf_tokens = int(
        safe_int(row.get("executed_fixed_leaf_tokens"), 0)
    )
    if max(requested_fixed_leaf_tokens, executed_fixed_leaf_tokens) != 128:
        return False
    tree_reference_label = _as_text(row.get("tree_reference_label"))
    return tree_reference_label in {
        "recoverable_root_only_parity_matched_root_v1",
        "structural_root_only_parity_matched_root_v1",
    }


def _is_structural_one_leaf_partial_root_rescue_pending(
    row: Mapping[str, Any],
) -> bool:
    baseline_family = _as_text(row.get("baseline_family"))
    if baseline_family not in {"tree_neural", "tree_neural_c2", "tree_neural_c2c3"}:
        return False
    if _as_text(row.get("scope_key")) != "r12_seg10to12":
        return False
    package_name = _as_text(row.get("package_name"))
    if not package_name.startswith("full") or package_name == "full100":
        return False
    requested_fixed_leaf_tokens = int(
        safe_int(row.get("requested_fixed_leaf_tokens"), 0)
    )
    executed_fixed_leaf_tokens = int(
        safe_int(row.get("executed_fixed_leaf_tokens"), 0)
    )
    if max(requested_fixed_leaf_tokens, executed_fixed_leaf_tokens) != 128:
        return False
    tree_reference_label = _as_text(row.get("tree_reference_label"))
    return tree_reference_label in {
        "structural_root_only_parity_matched_root_v2",
        "structural_root_only_parity_matched_root_v3",
    }


def is_headline_contract_status(value: object) -> bool:
    return _as_text(value) in HEADLINE_CONTRACT_STATUSES


def filtered_headline_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if is_headline_contract_status((row or {}).get("contract_status"))
    ]


def filtered_quarantined_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if _as_text((row or {}).get("contract_status"))
        == CONTRACT_STATUS_LEGACY_QUARANTINED
    ]


def quarantine_sources_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    values: set[str] = set()
    for raw_row in rows:
        row = dict(raw_row or {})
        row_values: set[str] = set()
        for candidate in (
            row.get("source_summary_json"),
            row.get("source_path"),
            row.get("job_output_dir"),
            row.get("report_row_key"),
        ):
            text = _as_text(candidate)
            if text:
                row_values.add(text)
        if not row_values:
            scope_key = _as_text(row.get("scope_key"))
            package_name = _as_text(row.get("package_name"))
            family = _as_text(row.get("baseline_family"))
            geometry = _as_text(row.get("supervision_recovery_geometry_key"))
            if scope_key or package_name or family or geometry:
                row_values.add(
                    "::".join(
                        bit
                        for bit in (scope_key, package_name, family, geometry)
                        if bit
                    )
                )
        values.update(row_values)
    return sorted(values)


def annotate_downstream_v3_row(
    row: Mapping[str, Any],
    *,
    canonical_fno_families: Iterable[str],
    canonical_fno_fixed_leaf_tokens: int,
) -> Dict[str, Any]:
    normalized = dict(row or {})
    baseline_family = _as_text(normalized.get("baseline_family"))
    comparison_mode = _as_text(normalized.get("comparison_mode"))
    comparison_semantics = _as_text(normalized.get("comparison_semantics"))
    comparison_semantics_label = _as_text(
        normalized.get("comparison_semantics_label")
    )
    run_intent_hash = _as_text(normalized.get("run_intent_hash"))
    run_intent_validation_status = _as_text(
        normalized.get("run_intent_validation_status")
    )
    requested_fixed_leaf_tokens = int(
        safe_int(normalized.get("requested_fixed_leaf_tokens"), 0)
    )
    executed_fixed_leaf_tokens = int(
        safe_int(normalized.get("executed_fixed_leaf_tokens"), 0)
    )
    depth_discount_gamma = float(
        safe_float(normalized.get("depth_discount_gamma"), float("nan"))
    )

    failures: list[str] = []
    if not baseline_family:
        failures.append("missing_baseline_family")
    if not comparison_mode:
        failures.append("missing_comparison_mode")
    if comparison_semantics not in ALLOWED_CONTRACT_STATUSES:
        failures.append("missing_or_invalid_comparison_semantics")
    if not comparison_semantics_label:
        failures.append("missing_comparison_semantics_label")
    if not run_intent_hash:
        failures.append("missing_run_intent_hash")
    if not run_intent_validation_status:
        failures.append("missing_run_intent_validation_status")
    if requested_fixed_leaf_tokens <= 0:
        failures.append("missing_requested_fixed_leaf_tokens")
    if executed_fixed_leaf_tokens <= 0:
        failures.append("missing_executed_fixed_leaf_tokens")
    if not math.isfinite(depth_discount_gamma):
        failures.append("missing_depth_discount_gamma")
    if (
        comparison_semantics == CONTRACT_STATUS_LOCKED_COMPARATOR
        and run_intent_validation_status != CONTRACT_STATUS_LOCKED_COMPARATOR
    ):
        failures.append("locked_comparator_without_locked_validation")
    if (
        comparison_semantics == CONTRACT_STATUS_CURRENT
        and run_intent_validation_status == CONTRACT_STATUS_LOCKED_COMPARATOR
    ):
        failures.append("current_semantics_with_locked_validation")
    if _is_known_invalid_one_leaf_root_only_tree_recipe(normalized):
        failures.append("known_invalid_one_leaf_root_only_recipe")
    if _is_known_invalid_one_leaf_matched_root_v1_recipe(normalized):
        failures.append("known_invalid_one_leaf_matched_root_v1_recipe")
    diagnostic_only_reasons: list[str] = []
    if _is_structural_one_leaf_partial_root_rescue_pending(normalized):
        diagnostic_only_reasons.append(
            "structural_one_leaf_partial_root_rescue_pending"
        )

    canonical_fno_family_set = {str(item).strip() for item in canonical_fno_families}
    if baseline_family in canonical_fno_family_set:
        if executed_fixed_leaf_tokens != int(canonical_fno_fixed_leaf_tokens):
            failures.append(
                "canonical_fno_executed_leaf_tokens_mismatch"
            )
        if (
            requested_fixed_leaf_tokens != int(canonical_fno_fixed_leaf_tokens)
            and run_intent_validation_status != CONTRACT_STATUS_LOCKED_COMPARATOR
        ):
            failures.append(
                "canonical_fno_requested_leaf_tokens_mismatch"
            )

    contract_status = (
        comparison_semantics
        if comparison_semantics in ALLOWED_CONTRACT_STATUSES
        else CONTRACT_STATUS_LEGACY_QUARANTINED
    )
    if failures:
        contract_status = CONTRACT_STATUS_LEGACY_QUARANTINED
        comparison_semantics = CONTRACT_STATUS_LEGACY_QUARANTINED
        if not comparison_semantics_label:
            comparison_semantics_label = "legacy_quarantined_downstream_v3_contract"
    elif diagnostic_only_reasons:
        contract_status = CONTRACT_STATUS_DIAGNOSTIC_ONLY
        comparison_semantics = CONTRACT_STATUS_DIAGNOSTIC_ONLY
        if not comparison_semantics_label:
            comparison_semantics_label = (
                "diagnostic_only_structural_one_leaf_rescue_pending"
            )

    normalized.update(
        {
            "baseline_family": baseline_family,
            "comparison_mode": comparison_mode,
            "comparison_semantics": comparison_semantics,
            "comparison_semantics_label": comparison_semantics_label,
            "run_intent_hash": run_intent_hash,
            "run_intent_validation_status": run_intent_validation_status,
            "requested_fixed_leaf_tokens": int(requested_fixed_leaf_tokens),
            "executed_fixed_leaf_tokens": int(executed_fixed_leaf_tokens),
            "depth_discount_gamma": depth_discount_gamma,
            "contract_status": contract_status,
            "contract_headline_eligible": bool(
                contract_status in HEADLINE_CONTRACT_STATUSES
            ),
            "contract_failures": list(failures),
            "contract_failure_count": int(len(failures)),
            "contract_diagnostic_reasons": list(diagnostic_only_reasons),
        }
    )
    return normalized


__all__ = [
    "ALLOWED_CONTRACT_STATUSES",
    "CONTRACT_STATUS_CURRENT",
    "CONTRACT_STATUS_DIAGNOSTIC_ONLY",
    "CONTRACT_STATUS_LEGACY_QUARANTINED",
    "CONTRACT_STATUS_LOCKED_COMPARATOR",
    "DOWNSTREAM_V3_REQUIRED_FIELDS",
    "HEADLINE_CONTRACT_STATUSES",
    "annotate_downstream_v3_row",
    "filtered_headline_rows",
    "filtered_quarantined_rows",
    "is_headline_contract_status",
    "quarantine_sources_from_rows",
]
