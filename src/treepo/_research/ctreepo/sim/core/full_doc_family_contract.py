from __future__ import annotations

import math
from typing import Any, Dict, Mapping

from treepo._research.core.ops_checks import EvidenceStatus, LawKind
from treepo._research.ctreepo.sim.core.full_doc_config_codec import (
    canonicalize_full_doc_config_mapping,
)
from treepo._research.ctreepo.sim.core.fno_doc_baselines import (
    FNO_TREE_C2_EXACT_WITNESS_KIND,
    FNO_TREE_C2_METRIC_KIND,
    FNO_TREE_C2_PROXY_METRIC_KIND,
)
from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import OPSCountConfig


FAMILY_API_VERSION = "markov_full_doc_family_api_v1"
LAW_CONTRACT_VERSION = "markov_full_doc_law_contract_v1"

OFFICIAL_FNO_BASELINE_FAMILIES = frozenset({"official_fno", "official_fno_sumlen"})
TREE_NEURAL_BASELINE_FAMILIES = frozenset(
    {"tree_neural_c2", "tree_neural_c2c3", "tree_neural"}
)
CONTROL_BASELINE_FAMILIES = frozenset(
    {
        "cnn1d",
        "mlp_bigram",
        "palette_block_exact",
        "raw_token_ngram_ridge",
        "ridge_control",
        "tree_doc_ridge",
        "tree_ridge_leaf",
    }
)
FULL_DOC_BUDGET_SUBSET_FAMILIES = frozenset(
    {
        "official_fno",
        "official_fno_sumlen",
        "cnn1d",
        "mlp_bigram",
        "palette_block_exact",
        "raw_token_ngram_ridge",
        "tree_doc_ridge",
    }
)


def _mapping_from_config_like(
    config_like: Mapping[str, Any] | OPSCountConfig | None,
) -> Dict[str, Any]:
    return canonicalize_full_doc_config_mapping(
        config_like,
        allow_private_tree_aliases=True,
    )


def _clean_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _normalize_baseline_family(baseline_family: str) -> str:
    family = str(baseline_family or "").strip().lower()
    if family in {"tree_ridge", "tree_leaf_ridge"}:
        return "tree_ridge_leaf"
    return family


def _resolved_objective_flag(
    mapping: Mapping[str, Any],
    *,
    resolved_key: str,
    fallback_enabled: bool,
) -> bool:
    resolved = _optional_float(mapping.get(resolved_key))
    if resolved is not None:
        return bool(resolved > 1e-12)
    objective_weights_active = bool(mapping.get("objective_weights_active", False))
    return bool(objective_weights_active and fallback_enabled)


def _law_entry(
    law_kind: LawKind,
    *,
    available: bool,
    objective_enforced: bool,
    evidence_status: EvidenceStatus,
    train_semantics: str,
    primary_eval_metric_kind: str = "",
    proxy_eval_metric_kind: str = "",
    exact_witness_kind: str = "",
    nontriviality_status: str = "not_applicable",
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "law_kind": law_kind.value,
        "paper_condition": law_kind.paper_condition,
        "lean_name": law_kind.lean_name,
        "available": bool(available),
        "objective_enforced": bool(objective_enforced),
        "evidence_status": evidence_status.value,
        "train_semantics": str(train_semantics),
        "primary_eval_metric_kind": str(primary_eval_metric_kind),
        "proxy_eval_metric_kind": str(proxy_eval_metric_kind),
        "exact_witness_kind": str(exact_witness_kind),
        "nontriviality_status": str(nontriviality_status),
        "notes": str(notes),
    }


def full_doc_family_api_report(
    baseline_family: str,
    *,
    config_like: Mapping[str, Any] | OPSCountConfig | None = None,
) -> Dict[str, Any]:
    family = _normalize_baseline_family(baseline_family)
    mapping = _mapping_from_config_like(config_like)
    tree_exact_collapse_mode = str(
        mapping.get("tree_exact_collapse_mode", "") or ""
    ).strip()
    if family in OFFICIAL_FNO_BASELINE_FAMILIES:
        return {
            "family_api_version": FAMILY_API_VERSION,
            "baseline_family": str(family),
            "family_api_group": "markov_full_doc_neuraloperator",
            "family_runner_kind": "official_fno_doc_sequence",
            "shared_framework_group": "shared_markov_fno_encoder",
            "shared_setup_contract": "shared_markov_full_doc_root_task_surface",
            "budget_execution_mode": "subset_selected_docs",
            "runtime_delegate_family": "",
            "tree_exact_collapse_mode": "",
        }
    if family in TREE_NEURAL_BASELINE_FAMILIES:
        return {
            "family_api_version": FAMILY_API_VERSION,
            "baseline_family": str(family),
            "family_api_group": "markov_full_doc_neuraloperator",
            "family_runner_kind": "tree_fno_count_sketch",
            "shared_framework_group": "shared_markov_fno_encoder",
            "shared_setup_contract": "shared_markov_full_doc_root_task_surface",
            "budget_execution_mode": "manifest_or_rate_driven_tree_supervision",
            "runtime_delegate_family": (
                "official_fno" if tree_exact_collapse_mode else ""
            ),
            "tree_exact_collapse_mode": str(tree_exact_collapse_mode),
        }
    if family in {"cnn1d", "mlp_bigram"}:
        return {
            "family_api_version": FAMILY_API_VERSION,
            "baseline_family": str(family),
            "family_api_group": "markov_full_doc_proxy_neural",
            "family_runner_kind": str(family),
            "shared_framework_group": "standalone_proxy_encoder",
            "shared_setup_contract": "shared_markov_full_doc_root_task_surface",
            "budget_execution_mode": "subset_selected_docs",
            "runtime_delegate_family": "",
            "tree_exact_collapse_mode": "",
        }
    if family in CONTROL_BASELINE_FAMILIES:
        return {
            "family_api_version": FAMILY_API_VERSION,
            "baseline_family": str(family),
            "family_api_group": "markov_full_doc_control",
            "family_runner_kind": str(family),
            "shared_framework_group": "closed_form_or_linear_control",
            "shared_setup_contract": "shared_markov_full_doc_root_task_surface",
            "budget_execution_mode": (
                "subset_selected_docs"
                if family in FULL_DOC_BUDGET_SUBSET_FAMILIES
                else "full_tree_required"
            ),
            "runtime_delegate_family": "",
            "tree_exact_collapse_mode": "",
        }
    return {
        "family_api_version": FAMILY_API_VERSION,
        "baseline_family": str(family or "unknown"),
        "family_api_group": "markov_full_doc_unknown",
        "family_runner_kind": str(family or "unknown"),
        "shared_framework_group": "unknown",
        "shared_setup_contract": "shared_markov_full_doc_root_task_surface",
        "budget_execution_mode": "unknown",
        "runtime_delegate_family": "",
        "tree_exact_collapse_mode": "",
    }


def _tree_c2_nontriviality(mapping: Mapping[str, Any]) -> str:
    c2_mode = str(mapping.get("tree_c2_mode", "reconstruction") or "reconstruction")
    c2_mode = c2_mode.strip().lower() or "reconstruction"
    if c2_mode == "fiber":
        return "fiber_contrastive_nontrivial"
    if str(mapping.get("summary_spec_name", "") or "").strip():
        return "decoded_summary_replay"
    if str(mapping.get("aligned_sketch_surface", "") or "").strip():
        return "decoded_surface_replay"
    if str(mapping.get("tree_model_version", "") or "").strip() == "unified_g":
        return "shared_summary_encoder_replay"
    return "state_reencode_replay"


def full_doc_law_contract_report(
    baseline_family: str,
    *,
    config_like: Mapping[str, Any] | OPSCountConfig | None = None,
    objective_weights_active: bool = False,
    c2_metric_kind: str = "",
    c2_proxy_metric_kind: str = "",
    c2_exact_witness_kind: str = "",
    mean_leaves_per_doc: float | None = None,
) -> Dict[str, Any]:
    family = _normalize_baseline_family(baseline_family)
    mapping = _mapping_from_config_like(config_like)
    api = full_doc_family_api_report(family, config_like=mapping)
    objective_flag_raw = mapping.get("objective_weights_active", objective_weights_active)
    if isinstance(objective_flag_raw, str):
        resolved_objective_weights_active = (
            objective_flag_raw.strip().lower() in {"1", "true", "yes", "on"}
        )
    else:
        resolved_objective_weights_active = bool(objective_flag_raw)
    gaps: list[str] = []
    limitations: list[str] = []

    if family in OFFICIAL_FNO_BASELINE_FAMILIES:
        law_status = "proxy_only_reference"
        limitations.append("root_only_reference_no_tree_local_law_channel")
        laws = {
            "c1": _law_entry(
                LawKind.L1_LEAF,
                available=False,
                objective_enforced=False,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                train_semantics="no_tree_local_law_objective",
                notes="Official FNO is a root-task reference baseline only.",
            ),
            "c2": _law_entry(
                LawKind.L3_IDEMPOTENCE,
                available=False,
                objective_enforced=False,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                train_semantics="no_tree_local_law_objective",
                notes="Official FNO does not expose a theorem-facing C2 replay channel.",
            ),
            "c3": _law_entry(
                LawKind.L2_MERGE,
                available=False,
                objective_enforced=False,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                train_semantics="no_tree_local_law_objective",
                notes="Official FNO is not a tree-merge witness.",
            ),
        }
    elif family in TREE_NEURAL_BASELINE_FAMILIES:
        c1_enabled = _resolved_objective_flag(
            mapping,
            resolved_key="local_law_c1_weight",
            fallback_enabled=(family == "tree_neural"),
        )
        c2_enabled = _resolved_objective_flag(
            mapping,
            resolved_key="local_law_c2_weight",
            fallback_enabled=(family in TREE_NEURAL_BASELINE_FAMILIES),
        )
        c3_enabled = _resolved_objective_flag(
            mapping,
            resolved_key="local_law_c3_weight",
            fallback_enabled=(family in {"tree_neural_c2c3", "tree_neural"}),
        )
        c2_nontriviality = _tree_c2_nontriviality(mapping)
        if resolved_objective_weights_active and c2_enabled:
            limitations.append("c2_replay_proxy_not_exact_paper_idempotence")
            if c2_nontriviality != "fiber_contrastive_nontrivial":
                limitations.append(
                    "c2_lacks_external_fiber_contrast"
                )
        if mean_leaves_per_doc is not None and math.isfinite(float(mean_leaves_per_doc)):
            if (
                resolved_objective_weights_active
                and float(mean_leaves_per_doc) <= 1.0
                and (c1_enabled or c3_enabled)
            ):
                gaps.append("single_leaf_geometry_collapses_tree_local_laws")
        law_status = "approximate_with_gaps" if gaps else "approximate_audited"
        laws = {
            "c1": _law_entry(
                LawKind.L1_LEAF,
                available=True,
                objective_enforced=bool(resolved_objective_weights_active and c1_enabled),
                evidence_status=EvidenceStatus.APPROX_AUDITED,
                train_semantics=(
                    "leaf_local_supervision"
                    if resolved_objective_weights_active and c1_enabled
                    else "inactive"
                ),
                primary_eval_metric_kind="leaf_mae",
                notes="Leaf-law evidence is approximate and depends on nontrivial multi-leaf geometry.",
            ),
            "c2": _law_entry(
                LawKind.L3_IDEMPOTENCE,
                available=True,
                objective_enforced=bool(resolved_objective_weights_active and c2_enabled),
                evidence_status=EvidenceStatus.APPROX_AUDITED,
                train_semantics=(
                    "decode_encode_replay"
                    if str(mapping.get("tree_c2_mode", "reconstruction") or "reconstruction")
                    .strip()
                    .lower()
                    != "fiber"
                    else "fiber_contrastive_replay"
                ),
                primary_eval_metric_kind=(
                    str(c2_metric_kind or mapping.get("c2_metric_kind", "")).strip()
                    or FNO_TREE_C2_METRIC_KIND
                ),
                proxy_eval_metric_kind=(
                    str(
                        c2_proxy_metric_kind or mapping.get("c2_proxy_metric_kind", "")
                    ).strip()
                    or FNO_TREE_C2_PROXY_METRIC_KIND
                ),
                exact_witness_kind=(
                    str(
                        c2_exact_witness_kind
                        or mapping.get("c2_exact_witness_kind", "")
                    ).strip()
                    or FNO_TREE_C2_EXACT_WITNESS_KIND
                ),
                nontriviality_status=str(c2_nontriviality),
                notes=(
                    "This is an approximate Markov replay contract, not an exact paper-C2 theorem witness."
                ),
            ),
            "c3": _law_entry(
                LawKind.L2_MERGE,
                available=True,
                objective_enforced=bool(resolved_objective_weights_active and c3_enabled),
                evidence_status=EvidenceStatus.APPROX_AUDITED,
                train_semantics=(
                    "internal_merge_supervision"
                    if resolved_objective_weights_active and c3_enabled
                    else "inactive"
                ),
                primary_eval_metric_kind="merge_mae",
                notes="Merge-law evidence is approximate and requires nontrivial tree geometry.",
            ),
        }
    else:
        law_status = "control_or_proxy_only"
        laws = {
            "c1": _law_entry(
                LawKind.L1_LEAF,
                available=False,
                objective_enforced=False,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                train_semantics="control_baseline",
            ),
            "c2": _law_entry(
                LawKind.L3_IDEMPOTENCE,
                available=False,
                objective_enforced=False,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                train_semantics="control_baseline",
            ),
            "c3": _law_entry(
                LawKind.L2_MERGE,
                available=False,
                objective_enforced=False,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                train_semantics="control_baseline",
            ),
        }

    notes = []
    if family in TREE_NEURAL_BASELINE_FAMILIES:
        notes.append(
            "Tree and official FNO families share the same top-level full-doc API group but not the same local-law head semantics."
        )
        notes.append(
            "Paper C2 / Lean L3 is stricter than the runtime replay/count-drift proxy used by the Markov tree family."
        )
    elif family in OFFICIAL_FNO_BASELINE_FAMILIES:
        notes.append(
            "Official FNO and tree-neural share the same encoder framework but only the tree family exposes theorem-facing local-law heads."
        )
    contract = {
        "law_contract_version": LAW_CONTRACT_VERSION,
        "baseline_family": str(family),
        "family_api": dict(api),
        "objective_weights_active": bool(resolved_objective_weights_active),
        "law_alignment_status": str(law_status),
        "law_contract_gaps": list(gaps),
        "law_contract_limitations": list(limitations),
        "c1": dict(laws["c1"]),
        "c2": dict(laws["c2"]),
        "c3": dict(laws["c3"]),
        "notes": list(notes),
    }
    return {
        "law_contract_version": LAW_CONTRACT_VERSION,
        "family_api_version": str(api["family_api_version"]),
        "family_api_group": str(api["family_api_group"]),
        "family_runner_kind": str(api["family_runner_kind"]),
        "shared_framework_group": str(api["shared_framework_group"]),
        "law_alignment_status": str(law_status),
        "law_contract_gap_count": int(len(gaps)),
        "law_contract_gaps": list(gaps),
        "law_contract_limitation_count": int(len(limitations)),
        "law_contract_limitations": list(limitations),
        "c2_nontriviality_status": str(contract["c2"]["nontriviality_status"]),
        "c2_train_semantics": str(contract["c2"]["train_semantics"]),
        "law_contract": contract,
    }


__all__ = [
    "CONTROL_BASELINE_FAMILIES",
    "FAMILY_API_VERSION",
    "FULL_DOC_BUDGET_SUBSET_FAMILIES",
    "LAW_CONTRACT_VERSION",
    "OFFICIAL_FNO_BASELINE_FAMILIES",
    "TREE_NEURAL_BASELINE_FAMILIES",
    "full_doc_family_api_report",
    "full_doc_law_contract_report",
]
