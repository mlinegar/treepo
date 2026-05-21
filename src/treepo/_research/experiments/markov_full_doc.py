from __future__ import annotations

from typing import Any, Mapping

from treepo._research.ctreepo.sim.core.full_doc_config_codec import (
    canonicalize_full_doc_config_mapping,
)
from treepo._research.ctreepo.sim.core.full_doc_family_contract import (
    OFFICIAL_FNO_BASELINE_FAMILIES,
    TREE_NEURAL_BASELINE_FAMILIES,
    full_doc_family_api_report,
    full_doc_law_contract_report,
)
from treepo._research.experiments.contracts import (
    ControlRef,
    MethodRef,
    ReferenceModelRef,
    method_ref_from_parts,
    stable_hash,
)
from treepo._research.experiments.normalization import supervision_ref_from_markov_config
from treepo._research.experiments.roles import (
    ROLE_SCORER,
    ROLE_STATE_MODEL,
    metadata_with_roles,
    oracle_ref,
    role_ref,
    state_model_role_ref,
)


_PAPER_TO_LEAN_LAW_IDS = {
    "c1": "L1",
    "c2": "L3",
    "c3": "L2",
}


def _safe_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalized_family(family: str) -> str:
    text = str(family or "").strip().lower()
    if text in {"tree_ridge", "tree_leaf_ridge"}:
        return "tree_ridge_leaf"
    return text


def _mapping(config_like: Mapping[str, Any] | None) -> dict[str, Any]:
    return canonicalize_full_doc_config_mapping(config_like)


def _resolved_mean_leaves_per_doc(mapping: Mapping[str, Any]) -> float | None:
    for key in (
        "mean_leaves_per_doc",
        "test_mean_leaves_per_doc",
        "val_mean_leaves_per_doc",
        "train_mean_leaves_per_doc",
    ):
        value = _safe_float(mapping.get(key))
        if value is not None:
            return value
    return None


def _supervision_mapping_for_family(
    family: str,
    *,
    config_like: Mapping[str, Any] | None,
) -> dict[str, Any]:
    mapping = _mapping(config_like)
    normalized_family = _normalized_family(family)
    if normalized_family not in TREE_NEURAL_BASELINE_FAMILIES:
        mapping["leaf_supervision_kind"] = "none"
        mapping["leaf_label_rate"] = 0.0
        mapping["internal_supervision_kind"] = "none"
        mapping["internal_label_rate"] = 0.0
    return mapping


def reference_model_ref_from_markov_full_doc_family(
    family: str,
    *,
    config_like: Mapping[str, Any] | None = None,
) -> ReferenceModelRef | None:
    normalized_family = _normalized_family(family)
    api = full_doc_family_api_report(normalized_family, config_like=config_like)
    if api["family_api_group"] == "markov_full_doc_unknown":
        return None
    payload = {
        "family_api_group": str(api["family_api_group"]),
        "family_runner_kind": str(api["family_runner_kind"]),
        "shared_framework_group": str(api["shared_framework_group"]),
        "baseline_family": str(normalized_family),
    }
    return ReferenceModelRef(
        reference_id=stable_hash(payload)[:16],
        family=str(api["family_api_group"]),
        variant=str(api["family_runner_kind"]),
        engine=str(api["shared_framework_group"]),
        model=str(normalized_family),
        metadata=dict(api),
    )


def control_ref_from_markov_full_doc_contract(
    family: str,
    *,
    config_like: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    objective_weights_active: bool | None = None,
    c2_metric_kind: str = "",
    c2_proxy_metric_kind: str = "",
    c2_exact_witness_kind: str = "",
    mean_leaves_per_doc: float | None = None,
) -> ControlRef | None:
    normalized_family = _normalized_family(family)
    mapping = _mapping(config_like)
    resolved_objective_weights_active = (
        bool(objective_weights_active)
        if objective_weights_active is not None
        else bool(mapping.get("objective_weights_active", False))
    )
    contract = full_doc_law_contract_report(
        normalized_family,
        config_like=mapping,
        objective_weights_active=resolved_objective_weights_active,
        c2_metric_kind=str(c2_metric_kind or mapping.get("c2_metric_kind", "")),
        c2_proxy_metric_kind=str(
            c2_proxy_metric_kind or mapping.get("c2_proxy_metric_kind", "")
        ),
        c2_exact_witness_kind=str(
            c2_exact_witness_kind or mapping.get("c2_exact_witness_kind", "")
        ),
        mean_leaves_per_doc=(
            mean_leaves_per_doc
            if mean_leaves_per_doc is not None
            else _resolved_mean_leaves_per_doc(mapping)
        ),
    )
    if normalized_family not in TREE_NEURAL_BASELINE_FAMILIES:
        return None
    law_contract = dict(contract.get("law_contract") or {})
    law_ids = tuple(
        law_id
        for law_key, law_id in _PAPER_TO_LEAN_LAW_IDS.items()
        if bool(dict(law_contract.get(law_key) or {}).get("objective_enforced", False))
    )
    merged_metadata = {
        "baseline_family": str(normalized_family),
        "family_api_group": str(contract.get("family_api_group", "")),
        "family_runner_kind": str(contract.get("family_runner_kind", "")),
        "shared_framework_group": str(contract.get("shared_framework_group", "")),
        "law_contract_version": str(contract.get("law_contract_version", "")),
        "law_alignment_status": str(contract.get("law_alignment_status", "")),
        "law_contract_gap_count": int(contract.get("law_contract_gap_count", 0) or 0),
        "law_contract_gaps": list(contract.get("law_contract_gaps") or ()),
        "law_contract_limitation_count": int(
            contract.get("law_contract_limitation_count", 0) or 0
        ),
        "law_contract_limitations": list(
            contract.get("law_contract_limitations") or ()
        ),
        "c2_nontriviality_status": str(contract.get("c2_nontriviality_status", "")),
        "c2_train_semantics": str(contract.get("c2_train_semantics", "")),
        "law_contract": law_contract,
        **dict(metadata or {}),
    }
    return ControlRef(
        control_family="markov_full_doc_local_law",
        law_ids=law_ids,
        applies_to="tree_nodes",
        enabled=bool(law_ids),
        source_kind="approx_audited_train_objective",
        metadata=merged_metadata,
    )


def method_ref_from_markov_full_doc_run(
    *,
    family: str,
    variant: str = "",
    adapter: str = "markov_tree",
    config_like: Mapping[str, Any] | None = None,
    package_name: str = "",
    metadata: Mapping[str, Any] | None = None,
    objective_weights_active: bool | None = None,
    c2_metric_kind: str = "",
    c2_proxy_metric_kind: str = "",
    c2_exact_witness_kind: str = "",
    mean_leaves_per_doc: float | None = None,
    include_reference_model: bool = True,
) -> MethodRef:
    normalized_family = _normalized_family(family)
    mapping = _mapping(config_like)
    resolved_mean_leaves = (
        mean_leaves_per_doc
        if mean_leaves_per_doc is not None
        else _resolved_mean_leaves_per_doc(mapping)
    )
    resolved_objective_weights_active = (
        bool(objective_weights_active)
        if objective_weights_active is not None
        else bool(mapping.get("objective_weights_active", False))
    )
    family_api = full_doc_family_api_report(normalized_family, config_like=mapping)
    law_contract = full_doc_law_contract_report(
        normalized_family,
        config_like=mapping,
        objective_weights_active=resolved_objective_weights_active,
        c2_metric_kind=str(c2_metric_kind or mapping.get("c2_metric_kind", "")),
        c2_proxy_metric_kind=str(
            c2_proxy_metric_kind or mapping.get("c2_proxy_metric_kind", "")
        ),
        c2_exact_witness_kind=str(
            c2_exact_witness_kind or mapping.get("c2_exact_witness_kind", "")
        ),
        mean_leaves_per_doc=resolved_mean_leaves,
    )
    supervision = supervision_ref_from_markov_config(
        _supervision_mapping_for_family(normalized_family, config_like=mapping),
        package_name=package_name,
        metadata={
            "baseline_family": str(normalized_family),
            "family_api_group": str(law_contract.get("family_api_group", "")),
            "law_alignment_status": str(law_contract.get("law_alignment_status", "")),
        },
    )
    control_ref = control_ref_from_markov_full_doc_contract(
        normalized_family,
        config_like=mapping,
        metadata=metadata,
        objective_weights_active=resolved_objective_weights_active,
        c2_metric_kind=str(c2_metric_kind or mapping.get("c2_metric_kind", "")),
        c2_proxy_metric_kind=str(
            c2_proxy_metric_kind or mapping.get("c2_proxy_metric_kind", "")
        ),
        c2_exact_witness_kind=str(
            c2_exact_witness_kind or mapping.get("c2_exact_witness_kind", "")
        ),
        mean_leaves_per_doc=resolved_mean_leaves,
    )
    role_engine = str(family_api.get("shared_framework_group", "") or "")
    roles = {
        ROLE_SCORER: role_ref(
            role=ROLE_SCORER,
            surface="native",
            engine=role_engine,
            model=normalized_family,
            metadata={
                "family_api_group": str(family_api.get("family_api_group", "")),
                "family_runner_kind": str(family_api.get("family_runner_kind", "")),
            },
        )
    }
    if normalized_family in TREE_NEURAL_BASELINE_FAMILIES or normalized_family in OFFICIAL_FNO_BASELINE_FAMILIES:
        roles[ROLE_STATE_MODEL] = state_model_role_ref(
            engine=role_engine,
            model=normalized_family,
            execution_mode=str(family_api.get("family_runner_kind", "") or ""),
            metadata={
                "family_api_group": str(family_api.get("family_api_group", "")),
                "baseline_family": normalized_family,
            },
        )
    method_metadata = metadata_with_roles(
        {
        "baseline_family": str(normalized_family),
        "package_name": str(package_name or ""),
        "family_api_version": str(family_api.get("family_api_version", "")),
        "family_api_group": str(family_api.get("family_api_group", "")),
        "family_runner_kind": str(family_api.get("family_runner_kind", "")),
        "shared_framework_group": str(family_api.get("shared_framework_group", "")),
        "shared_setup_contract": str(family_api.get("shared_setup_contract", "")),
        "law_contract_version": str(law_contract.get("law_contract_version", "")),
        "law_alignment_status": str(law_contract.get("law_alignment_status", "")),
        "law_contract_gap_count": int(law_contract.get("law_contract_gap_count", 0) or 0),
        "law_contract_gaps": list(law_contract.get("law_contract_gaps") or ()),
        "law_contract_limitation_count": int(
            law_contract.get("law_contract_limitation_count", 0) or 0
        ),
        "law_contract_limitations": list(
            law_contract.get("law_contract_limitations") or ()
        ),
        "c2_nontriviality_status": str(
            law_contract.get("c2_nontriviality_status", "")
        ),
        "c2_train_semantics": str(law_contract.get("c2_train_semantics", "")),
        "family_api": dict(family_api),
        "law_contract": dict(law_contract.get("law_contract") or {}),
        **dict(metadata or {}),
        },
        roles=roles,
        oracle=oracle_ref(
            kind="markov_full_doc_targets",
            source="synthetic_markov_generator",
            metadata={"label_semantics": "full_doc_count_or_state_target"},
        ),
    )
    for key in (
        "comparison_mode",
        "comparison_semantics",
        "comparison_semantics_label",
        "run_intent_hash",
        "run_intent_validation_status",
        "tree_c2_mode",
        "depth_discount_gamma",
        "fixed_leaf_tokens",
        "slot_count",
        "study_name",
        "study_axis",
        "package_semantics",
    ):
        if key not in mapping:
            continue
        value = mapping[key]
        if value is None or value == "":
            continue
        method_metadata[key] = value
    return method_ref_from_parts(
        family=str(normalized_family),
        variant=str(variant or package_name or normalized_family),
        adapter=str(adapter),
        supervision=supervision,
        control_ref=control_ref,
        reference_model=(
            reference_model_ref_from_markov_full_doc_family(
                normalized_family,
                config_like=mapping,
            )
            if include_reference_model
            else None
        ),
        metadata=method_metadata,
    )


__all__ = [
    "control_ref_from_markov_full_doc_contract",
    "method_ref_from_markov_full_doc_run",
    "reference_model_ref_from_markov_full_doc_family",
]
