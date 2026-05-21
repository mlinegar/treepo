from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Dict, Mapping, Optional, Sequence

from treepo._research.ctreepo.contracts import (
    LAW_ID_LEAF_PRESERVATION,
    LAW_ID_MERGE_PRESERVATION,
    LAW_ID_ON_RANGE_IDEMPOTENCE,
    ProblemAdapterSpec,
    canonical_law_component_weights,
    default_law_set_specs,
)
from treepo._research.ctreepo.sim.util import safe_float
from treepo._research.core.ops_checks import (
    EvidenceStatus,
    LawCapabilityReport,
    LawKind,
    OperatorCapabilityReport,
)
from treepo._research.core.provenance import ORACLE_SOURCE
from treepo._research.core.runtime_capabilities import (
    FamilyRuntimeCapability,
    default_family_runtime_capability,
    markov_family_runtime_capability,
)
from treepo._research.ctreepo.sim.local_law_learnability import (
    LocalLawPolicyEvaluation,
    LocalLawRunSummary,
    PolicyRole,
    split_metric_views,
)
from treepo._research.tree.compositional_learning import (
    CompositionalLearningProblemSpec,
    OracleQueryPolicySpec,
    SHARED_SAMPLED_SUBSTRUCTURE_CHANNEL_NAME,
    SupervisionDeliveryMode,
    shared_full_document_supervision_channel,
    shared_protocol_problem_notes,
    shared_sampled_substructure_query_policy,
    shared_sampled_substructure_supervision_channel,
)


_MARKOV_EXACT_LAW_AVAILABILITY: Dict[str, Dict[LawKind, bool]] = {
    "exact": {
        LawKind.L1_LEAF: True,
        LawKind.L2_MERGE: True,
        LawKind.L3_IDEMPOTENCE: True,
    },
    "count_only": {
        LawKind.L1_LEAF: True,
        LawKind.L2_MERGE: False,
        LawKind.L3_IDEMPOTENCE: True,
    },
    "leaf_bucket": {
        LawKind.L1_LEAF: False,
        LawKind.L2_MERGE: True,
        LawKind.L3_IDEMPOTENCE: True,
    },
    "flip_R2": {
        LawKind.L1_LEAF: True,
        LawKind.L2_MERGE: True,
        LawKind.L3_IDEMPOTENCE: False,
    },
}


_safe_float = lambda v, default=0.0: safe_float(v, default=default)

_PROBLEM_ADAPTERS: Dict[str, ProblemAdapterSpec] = {
    "markov_ops_count": ProblemAdapterSpec(
        problem_id="markov_ops_count",
        document_type_name="markov_regime_documents",
        theorem_domain_name="changepoint_count_summary_states",
        oracle_label_sources=("root_state", "sampled_node_labels", "oracle_state"),
        law_sets=default_law_set_specs(
            (
                LAW_ID_LEAF_PRESERVATION,
                LAW_ID_MERGE_PRESERVATION,
                LAW_ID_ON_RANGE_IDEMPOTENCE,
            )
        ),
    ),
    "tree_relevant_lda_local_law": ProblemAdapterSpec(
        problem_id="leaf_local_mixture_utility",
        document_type_name="topic_mixture_documents",
        theorem_domain_name="analysis_summary_states",
        oracle_label_sources=("root_topic_target", "sampled_node_labels"),
    ),
}


def _family_problem_metadata(summary: LocalLawRunSummary) -> Dict[str, str]:
    adapter = _PROBLEM_ADAPTERS.get(str(summary.family))
    if adapter is not None:
        return {
            "problem_id": str(adapter.problem_id),
            "name": f"{adapter.problem_id}_local_law_learning",
            "document_type_name": str(adapter.document_type_name),
            "theorem_domain_name": str(adapter.theorem_domain_name),
        }
    return {
        "problem_id": str(summary.family or "generic"),
        "name": "local_law_learning",
        "document_type_name": "documents",
        "theorem_domain_name": "summary_states",
    }


def _policy_lookup(
    summary: LocalLawRunSummary,
    name: str,
) -> Optional[LocalLawPolicyEvaluation]:
    policies = dict(summary.policies or {})
    if name in policies:
        return policies[str(name)]
    for key, policy in policies.items():
        if str(key) == str(name) or str(policy.name) == str(name):
            return policy
    return None


def _selected_policy_name(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> str:
    selection = dict(summary.selection or {})
    selected = str(selection.get("selected_candidate", "") or "").strip()
    if selected:
        return selected
    if raw_payload is not None:
        local_law = dict(raw_payload.get("local_law", {}) or {})
        raw_selection = dict(local_law.get("selection", {}) or {})
        selected = str(raw_selection.get("selected_candidate", "") or "").strip()
        if selected:
            return selected
    return ""


def _primary_policy(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> Optional[LocalLawPolicyEvaluation]:
    learned = _policy_lookup(summary, "learned_g")
    if learned is not None:
        return learned
    selected_name = _selected_policy_name(summary, raw_payload=raw_payload)
    if selected_name:
        selected = _policy_lookup(summary, selected_name)
        if selected is not None:
            return selected
    if str(summary.study_role) == PolicyRole.ORACLE_G.value:
        oracle = _policy_lookup(summary, "oracle_g") or _policy_lookup(
            summary, summary.oracle_name
        )
        if oracle is not None:
            return oracle
    for policy in dict(summary.policies or {}).values():
        role = (
            policy.role.value
            if isinstance(policy.role, PolicyRole)
            else str(policy.role)
        )
        if role == PolicyRole.BASELINE_G.value:
            return policy
    for policy in dict(summary.policies or {}).values():
        return policy
    return None


def _first_objective_payload(
    policy: Optional[LocalLawPolicyEvaluation],
) -> Dict[str, Any]:
    if policy is None:
        return {}
    for split_name in ("test", "val", "train"):
        split_metrics = dict(policy.split_metrics or {}).get(split_name, {})
        if not isinstance(split_metrics, Mapping):
            continue
        _local, _downstream, objective = split_metric_views(split_metrics)
        if objective:
            return dict(objective)
    return {}


def _summary_objective_payload(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    policy_objective = _first_objective_payload(
        _primary_policy(summary, raw_payload=raw_payload)
    )
    if policy_objective:
        return policy_objective
    metadata = dict(summary.metadata or {})
    objective = dict(metadata.get("objective", {}) or {})
    if objective:
        return objective
    if raw_payload is not None:
        raw_objective = dict(raw_payload.get("objective", {}) or {})
        if raw_objective:
            return raw_objective
    return {}


def _has_ipw_logged_observation_artifact(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> bool:
    artifacts = dict(summary.logged_observation_artifacts or {})
    if not artifacts and raw_payload is not None:
        direct = dict(raw_payload.get("logged_observation_artifacts", {}) or {})
        if direct:
            artifacts = direct
    return any(
        bool(dict(payload or {}).get("supports_ipw_estimation", False))
        for payload in artifacts.values()
        if isinstance(payload, Mapping)
    )


def _has_legacy_estimator_payload(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> bool:
    objective = _summary_objective_payload(summary, raw_payload=raw_payload)
    available_estimators = [
        str(name)
        for name in list(objective.get("available_estimators", []) or [])
    ]
    if "hajek" in available_estimators or "ht" in available_estimators:
        return True
    selection_estimator = str(objective.get("selection_estimator", "") or "").strip()
    if selection_estimator in {"hajek", "ht"}:
        return True
    for key in objective.keys():
        if str(key).endswith("_hajek") or str(key).endswith("_ht"):
            return True
    if raw_payload is None:
        return False
    local_law = dict(raw_payload.get("local_law", {}) or {})
    if dict(local_law.get("ipw_evaluation", {}) or {}):
        return True
    if dict(local_law.get("split_ipw_evaluation", {}) or {}):
        return True
    stage3 = dict(raw_payload.get("stage3", {}) or {})
    if dict(stage3.get("ipw_evaluation", {}) or {}):
        return True
    return False


def _backfilled_logged_observation_artifacts(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    artifacts = dict(summary.logged_observation_artifacts or {})
    if artifacts or not _has_legacy_estimator_payload(summary, raw_payload=raw_payload):
        return artifacts
    return {
        "legacy_ipw_summary": {
            "artifact_id": "legacy_ipw_summary",
            "channel_name": SHARED_SAMPLED_SUBSTRUCTURE_CHANNEL_NAME,
            "format": "legacy_summary",
            "path": "",
            "count": 0,
            "unit_kinds": [],
            "propensity_fields_logged": ["joint_propensity"],
            "supports_ipw_estimation": True,
            "metadata": {"backfilled_from_legacy_ipw_payload": True},
        }
    }


def _sampled_supervision_active(summary: LocalLawRunSummary) -> bool:
    budget = summary.support_budget
    return bool(
        float(budget.leaf_query_rate) > 0.0
        or float(budget.internal_query_rate) > 0.0
        or float(budget.mean_leaf_labels_per_doc) > 0.0
        or float(budget.mean_internal_labels_per_doc) > 0.0
    )


def _full_document_supervision_active(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> bool:
    objective = _summary_objective_payload(summary, raw_payload=raw_payload)
    root_share = _safe_float(objective.get("root_share"), float("nan"))
    if math.isfinite(root_share):
        return bool(root_share > 0.0)
    return bool(float(summary.support_budget.root_query_rate) > 0.0)


def _sampled_targeted_laws(summary: LocalLawRunSummary) -> Sequence[LawKind]:
    targeted: list[LawKind] = []
    budget = summary.support_budget
    if (
        float(budget.leaf_query_rate) > 0.0
        or float(budget.mean_leaf_labels_per_doc) > 0.0
    ):
        targeted.append(LawKind.L1_LEAF)
    if (
        float(budget.internal_query_rate) > 0.0
        or float(budget.mean_internal_labels_per_doc) > 0.0
    ):
        targeted.append(LawKind.L2_MERGE)
    return targeted


def _query_strategy(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> str:
    metadata = dict(summary.support_budget.metadata or {})
    if str(summary.family) == "markov_ops_count":
        strategy = str(metadata.get("audit_policy", "") or "").strip()
        if strategy:
            return strategy
        if raw_payload is not None:
            cfg = dict(raw_payload.get("config", {}) or {})
            strategy = str(cfg.get("audit_policy", "") or "").strip()
            if strategy:
                return strategy
        return "uniform_random"
    if str(summary.family) == "tree_relevant_lda_local_law":
        leaf_design = str(metadata.get("law_leaf_query_design", "") or "").strip()
        internal_design = str(
            metadata.get("law_internal_query_design", "") or ""
        ).strip()
        if raw_payload is not None:
            cfg = dict(raw_payload.get("config", {}) or {})
            if not leaf_design:
                leaf_design = str(cfg.get("law_leaf_query_design", "") or "").strip()
            if not internal_design:
                internal_design = str(
                    cfg.get("law_internal_query_design", "") or ""
                ).strip()
        parts = []
        if leaf_design:
            parts.append(f"leaf={leaf_design}")
        if internal_design:
            parts.append(f"internal={internal_design}")
        return ",".join(parts) if parts else "uniform_random"
    return "uniform_random"


def _sampled_query_policy(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
    supports_unbiased_risk: bool,
) -> Optional[OracleQueryPolicySpec]:
    if not _sampled_supervision_active(summary):
        return None
    budget = {
        "leaf_query_rate": float(summary.support_budget.leaf_query_rate),
        "internal_query_rate": float(summary.support_budget.internal_query_rate),
        "root_query_rate": float(summary.support_budget.root_query_rate),
        "mean_leaf_labels_per_doc": float(summary.support_budget.mean_leaf_labels_per_doc),
        "mean_internal_labels_per_doc": float(
            summary.support_budget.mean_internal_labels_per_doc
        ),
        "mean_queries_per_doc": float(summary.support_budget.mean_queries_per_doc),
        "total_queries_estimate": float(summary.support_budget.total_queries_estimate),
    }
    notes = [
        "This channel represents on-demand oracle calls over sampled substructures rather than a fixed offline label table.",
    ]
    if not supports_unbiased_risk:
        notes.append(
            "The run summary does not retain enough realized propensity information to reconstruct IPW risk estimates end-to-end."
        )
    return shared_sampled_substructure_query_policy(
        selection_strategy=_query_strategy(summary, raw_payload=raw_payload),
        adaptive=bool(
            any(
                token in _query_strategy(summary, raw_payload=raw_payload)
                for token in ("weighted", "profile", "risk", "priority", "adversarial")
            )
        ),
        budget=budget,
        propensity_field_name="propensity",
        logs_realized_propensities=bool(supports_unbiased_risk),
        supports_ipw_estimation=bool(supports_unbiased_risk),
        notes=tuple(notes),
    )


def _objective_weight(
    objective: Mapping[str, Any],
    *,
    names: Sequence[str],
    container_name: str = "local_law_component_weights",
) -> float:
    nested = canonical_law_component_weights(
        dict(objective.get(container_name, {}) or {}),
        allow_aliases=True,
    )
    for name in names:
        canonical_names = canonical_law_component_weights({str(name): 1.0}, allow_aliases=True)
        for canonical_name in canonical_names:
            if canonical_name in nested:
                return _safe_float(nested.get(canonical_name), 0.0)
    for name in names:
        for key in (
            name,
            f"local_law_{name}_weight",
            f"{name}_weight",
        ):
            if key in objective:
                return _safe_float(objective.get(key), 0.0)
    return 0.0


def _proxy_operator_capabilities(
    *,
    operator_name: str,
    tree_nesting_supported: bool,
    leaf_objective_enforced: bool,
    merge_objective_enforced: bool,
    idempotence_objective_enforced: bool,
    notes: Sequence[str],
) -> OperatorCapabilityReport:
    return OperatorCapabilityReport(
        operator_name=str(operator_name),
        evidence_status=EvidenceStatus.PROXY_ONLY,
        latent_mergeability_enforced=False,
        tree_nesting_supported=bool(tree_nesting_supported),
        theorem_domain_decode_available=False,
        theorem_domain_reencode_available=False,
        exact_reduction_supported=False,
        leaf_law=LawCapabilityReport(
            law_kind=LawKind.L1_LEAF,
            available=False,
            evidence_status=EvidenceStatus.PROXY_ONLY,
            objective_enforced=bool(leaf_objective_enforced),
            exact=False,
            notes="Local leaf supervision is present, but no explicit theorem-backed operator assumptions were supplied.",
        ),
        merge_law=LawCapabilityReport(
            law_kind=LawKind.L2_MERGE,
            available=False,
            evidence_status=EvidenceStatus.PROXY_ONLY,
            objective_enforced=bool(merge_objective_enforced),
            exact=False,
            notes="Merge supervision is optimization-facing only in this artifact.",
        ),
        idempotence_law=LawCapabilityReport(
            law_kind=LawKind.L3_IDEMPOTENCE,
            available=False,
            evidence_status=EvidenceStatus.PROXY_ONLY,
            objective_enforced=bool(idempotence_objective_enforced),
            exact=False,
            notes="Idempotence appears only through an objective-side proxy unless a separate theorem operator is attached.",
        ),
        notes=tuple(str(note) for note in notes),
    )


def _markov_exact_family_name(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> str:
    metadata = dict(summary.metadata or {})
    exact_family = str(metadata.get("exact_family", "") or "").strip()
    if exact_family in _MARKOV_EXACT_LAW_AVAILABILITY:
        return exact_family
    selected = _selected_policy_name(summary, raw_payload=raw_payload)
    if selected == "oracle_g" or str(summary.study_role) == PolicyRole.ORACLE_G.value:
        return "exact"
    if selected in _MARKOV_EXACT_LAW_AVAILABILITY:
        return selected
    if raw_payload is not None:
        cfg = dict(raw_payload.get("config", {}) or {})
        exact_family = str(cfg.get("exact_family", "") or "").strip()
        if exact_family in _MARKOV_EXACT_LAW_AVAILABILITY:
            return exact_family
    return ""


def _markov_exact_capabilities(
    *,
    operator_name: str,
    exact_family: str,
) -> OperatorCapabilityReport:
    availability = dict(_MARKOV_EXACT_LAW_AVAILABILITY[exact_family])
    fully_theorem_backed = all(bool(value) for value in availability.values())

    def _law_report(
        law_kind: LawKind,
        *,
        available: bool,
    ) -> LawCapabilityReport:
        if available:
            return LawCapabilityReport(
                law_kind=law_kind,
                available=True,
                evidence_status=EvidenceStatus.THEOREM_BACKED,
                objective_enforced=True,
                exact=True,
                notes=f"Exact symbolic Markov family `{exact_family}` satisfies this law.",
            )
        return LawCapabilityReport(
            law_kind=law_kind,
            available=False,
            evidence_status=EvidenceStatus.PROXY_ONLY,
            objective_enforced=False,
            exact=False,
            notes=f"Exact symbolic Markov family `{exact_family}` is a deliberate counterexample for this law.",
        )

    return OperatorCapabilityReport(
        operator_name=str(operator_name),
        evidence_status=(
            EvidenceStatus.THEOREM_BACKED
            if fully_theorem_backed
            else EvidenceStatus.PROXY_ONLY
        ),
        latent_mergeability_enforced=True,
        tree_nesting_supported=True,
        theorem_domain_decode_available=True,
        theorem_domain_reencode_available=True,
        exact_reduction_supported=bool(fully_theorem_backed),
        leaf_law=_law_report(
            LawKind.L1_LEAF,
            available=bool(availability[LawKind.L1_LEAF]),
        ),
        merge_law=_law_report(
            LawKind.L2_MERGE,
            available=bool(availability[LawKind.L2_MERGE]),
        ),
        idempotence_law=_law_report(
            LawKind.L3_IDEMPOTENCE,
            available=bool(availability[LawKind.L3_IDEMPOTENCE]),
        ),
        notes=(
            "Capability surface inferred from the selected exact symbolic Markov family.",
            "No separate theorem-assumption bundle was supplied in the serialized run summary.",
        ),
    )


def _operator_name(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> str:
    primary = _primary_policy(summary, raw_payload=raw_payload)
    if primary is not None and str(primary.name or "").strip():
        return str(primary.name)
    selected = _selected_policy_name(summary, raw_payload=raw_payload)
    if selected:
        return str(selected)
    return str(summary.family or "local_law_operator")


def _operator_capabilities(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> Optional[OperatorCapabilityReport]:
    objective = _summary_objective_payload(summary, raw_payload=raw_payload)
    operator_name = _operator_name(summary, raw_payload=raw_payload)
    leaf_weight = _objective_weight(objective, names=("c1",))
    merge_weight = _objective_weight(objective, names=("c3",))
    idempotence_weight = _objective_weight(objective, names=("c2", "c2_proxy"))

    if str(summary.family) == "markov_ops_count":
        exact_family = _markov_exact_family_name(summary, raw_payload=raw_payload)
        if exact_family:
            return _markov_exact_capabilities(
                operator_name=operator_name,
                exact_family=exact_family,
            )
        return _proxy_operator_capabilities(
            operator_name=operator_name,
            tree_nesting_supported=True,
            leaf_objective_enforced=bool(leaf_weight > 0.0),
            merge_objective_enforced=bool(merge_weight > 0.0),
            idempotence_objective_enforced=bool(idempotence_weight > 0.0),
            notes=(
                "Markov learned operators share the same compositional-learning API as the LLM lane.",
                "This summary only records objective-side law pressure, not a supplied theorem-backed codec.",
            ),
        )

    if str(summary.family) == "tree_relevant_lda_local_law":
        return _proxy_operator_capabilities(
            operator_name=operator_name,
            tree_nesting_supported=True,
            leaf_objective_enforced=bool(leaf_weight > 0.0),
            merge_objective_enforced=bool(merge_weight > 0.0),
            idempotence_objective_enforced=bool(idempotence_weight > 0.0),
            notes=(
                "LDA local-law runs now use the same learning-problem manifest shape as Markov and LLM artifacts.",
                "The c2 term is a proxy-side surrogate rather than a theorem-domain idempotence certificate.",
            ),
        )

    return None


def _runtime_capabilities(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> FamilyRuntimeCapability:
    family = str(summary.family)
    if family == "markov_ops_count":
        exact_family = _markov_exact_family_name(summary, raw_payload=raw_payload) or None
        return markov_family_runtime_capability(
            family_name=family,
            exact_family=exact_family,
            notes=(
                f"study_role={summary.study_role}",
                "Chat-engine lanes remain empirical even when an exact symbolic Markov family is available.",
            ),
        )

    capabilities = _operator_capabilities(summary, raw_payload=raw_payload)
    theorem_backed_symbolic = bool(
        capabilities is not None
        and capabilities.theorem_domain_decode_available
        and capabilities.theorem_domain_reencode_available
        and capabilities.exact_reduction_supported
        and capabilities.evidence_status == EvidenceStatus.THEOREM_BACKED
    )
    return default_family_runtime_capability(
        family_name=family,
        theorem_backed_symbolic=theorem_backed_symbolic,
        notes=(
            f"study_role={summary.study_role}",
            "Runtime capability metadata is declarative only and does not auto-select a backend in v1.",
        ),
    )


def build_local_law_runtime_capability(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build declarative runtime capability metadata for reporting surfaces."""
    return _runtime_capabilities(summary, raw_payload=raw_payload).to_dict()


def build_local_law_learning_problem(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    family_meta = _family_problem_metadata(summary)
    objective = _summary_objective_payload(summary, raw_payload=raw_payload)
    sampled_active = _sampled_supervision_active(summary)
    full_document_active = _full_document_supervision_active(
        summary,
        raw_payload=raw_payload,
    )
    supports_unbiased_risk = bool(
        sampled_active and _has_ipw_logged_observation_artifact(summary, raw_payload=raw_payload)
    )
    sampled_notes = [
        "Leaf and internal supervision is recorded as a sampled substructure channel so Markov, LDA, and LLM artifacts share one API.",
    ]
    if _objective_weight(objective, names=("c2", "c2_proxy")) > 0.0:
        sampled_notes.append(
            "Idempotence enters only indirectly in this lane; sampled labels primarily target leaf and merge structure."
        )
    if sampled_active and not supports_unbiased_risk:
        sampled_notes.append(
            "This artifact does not retain enough estimator payload to certify unbiased-risk accounting end-to-end."
        )
    problem = CompositionalLearningProblemSpec(
        name=str(family_meta["name"]),
        document_type_name=str(family_meta["document_type_name"]),
        theorem_domain_name=str(family_meta["theorem_domain_name"]),
        operator_name=_operator_name(summary, raw_payload=raw_payload),
        operator_capabilities=_operator_capabilities(
            summary,
            raw_payload=raw_payload,
        ),
        supervision_channels=(
            shared_full_document_supervision_channel(
                active=bool(full_document_active),
                label_source=ORACLE_SOURCE,
                notes=(
                    "Whole-document oracle targets instantiate the document-level lane when this application exposes them.",
                ),
            ),
            shared_sampled_substructure_supervision_channel(
                active=bool(sampled_active),
                label_source=ORACLE_SOURCE,
                delivery_mode=SupervisionDeliveryMode.ONLINE_ORACLE_QUERY,
                query_policy=_sampled_query_policy(
                    summary,
                    raw_payload=raw_payload,
                    supports_unbiased_risk=supports_unbiased_risk,
                ),
                targeted_laws=tuple(_sampled_targeted_laws(summary)),
                requires_propensity_logging=bool(sampled_active),
                supports_unbiased_risk=bool(supports_unbiased_risk),
                notes=tuple(sampled_notes),
            ),
        ),
        notes=shared_protocol_problem_notes(
            application_name=str(family_meta["name"]),
            notes=(
            f"study_role={summary.study_role}",
            f"suite_role={summary.suite_role}",
            "This payload is inferred from LocalLawRunSummary so theorem assumptions stay empty unless another artifact supplied them explicitly.",
            ),
        ),
    )
    return problem.to_dict()


def attach_local_law_learning_problem(
    summary: LocalLawRunSummary,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
    overwrite: bool = False,
) -> LocalLawRunSummary:
    summary = replace(
        summary,
        logged_observation_artifacts=_backfilled_logged_observation_artifacts(
            summary,
            raw_payload=raw_payload,
        ),
    )
    if summary.compositional_learning_problem and not overwrite:
        return summary
    return replace(
        summary,
        compositional_learning_problem=build_local_law_learning_problem(
            summary,
            raw_payload=raw_payload,
        ),
    )


__all__ = [
    "attach_local_law_learning_problem",
    "build_local_law_learning_problem",
    "build_local_law_runtime_capability",
]
