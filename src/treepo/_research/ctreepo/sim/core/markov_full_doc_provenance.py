from __future__ import annotations

from importlib import metadata as importlib_metadata
from typing import Any, Dict

from treepo._research.core.ops_checks import EvidenceStatus, LawCapabilityReport, LawKind, OperatorCapabilityReport
from treepo._research.ctreepo.sim.core.full_doc_family_contract import (
    CONTROL_BASELINE_FAMILIES,
    OFFICIAL_FNO_BASELINE_FAMILIES,
    TREE_NEURAL_BASELINE_FAMILIES,
    full_doc_law_contract_report,
)


def _package_version(package_name: str) -> str:
    try:
        return str(importlib_metadata.version(str(package_name)))
    except importlib_metadata.PackageNotFoundError:
        return ""


def _law_report(
    law_kind: LawKind,
    *,
    available: bool,
    evidence_status: EvidenceStatus,
    objective_enforced: bool = False,
    notes: str = "",
) -> LawCapabilityReport:
    return LawCapabilityReport(
        law_kind=law_kind,
        available=bool(available),
        evidence_status=evidence_status,
        objective_enforced=bool(objective_enforced),
        notes=str(notes) or None,
    )


def full_doc_operator_capability_report(baseline_family: str) -> OperatorCapabilityReport:
    family = str(baseline_family or "").strip().lower()
    if family in OFFICIAL_FNO_BASELINE_FAMILIES:
        return OperatorCapabilityReport(
            operator_name="official_neuraloperator_fno",
            evidence_status=EvidenceStatus.PROXY_ONLY,
            latent_mergeability_enforced=False,
            tree_nesting_supported=False,
            theorem_domain_decode_available=False,
            theorem_domain_reencode_available=False,
            exact_reduction_supported=False,
            leaf_law=_law_report(
                LawKind.L1_LEAF,
                available=False,
                evidence_status=EvidenceStatus.PROXY_ONLY,
            ),
            merge_law=_law_report(
                LawKind.L2_MERGE,
                available=False,
                evidence_status=EvidenceStatus.PROXY_ONLY,
            ),
            idempotence_law=_law_report(
                LawKind.L3_IDEMPOTENCE,
                available=False,
                evidence_status=EvidenceStatus.PROXY_ONLY,
            ),
            notes=(
                "Official neuraloperator FNO baseline for the paper-facing full-doc lane.",
                "This is a publication baseline, not a theorem-backed local-law witness.",
            ),
        )
    if family == "palette_block_exact":
        return OperatorCapabilityReport(
            operator_name="palette_block_exact",
            evidence_status=EvidenceStatus.THEOREM_BACKED,
            latent_mergeability_enforced=True,
            tree_nesting_supported=True,
            theorem_domain_decode_available=False,
            theorem_domain_reencode_available=False,
            exact_reduction_supported=True,
            leaf_law=_law_report(
                LawKind.L1_LEAF,
                available=True,
                evidence_status=EvidenceStatus.THEOREM_BACKED,
                notes="Exact control on the disjoint-palette generator family.",
            ),
            merge_law=_law_report(
                LawKind.L2_MERGE,
                available=True,
                evidence_status=EvidenceStatus.THEOREM_BACKED,
                notes="Exact block-count control on the disjoint-palette generator family.",
            ),
            idempotence_law=_law_report(
                LawKind.L3_IDEMPOTENCE,
                available=False,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                notes="No decode/re-encode path is exposed for this closed-form control.",
            ),
            notes=(
                "Generator-exact control for the structural grid.",
            ),
        )
    if family in TREE_NEURAL_BASELINE_FAMILIES:
        c1_active = family == "tree_neural"
        c2_active = family in {"tree_neural_c2", "tree_neural_c2c3", "tree_neural"}
        c3_active = family in {"tree_neural_c2c3", "tree_neural"}
        return OperatorCapabilityReport(
            operator_name="tree_fno_count_sketch",
            evidence_status=EvidenceStatus.APPROX_AUDITED,
            latent_mergeability_enforced=True,
            tree_nesting_supported=True,
            theorem_domain_decode_available=True,
            theorem_domain_reencode_available=True,
            exact_reduction_supported=False,
            leaf_law=_law_report(
                LawKind.L1_LEAF,
                available=True,
                evidence_status=EvidenceStatus.APPROX_AUDITED,
                objective_enforced=c1_active,
                notes="Approximate tree-local-law supervision on leaf summaries.",
            ),
            merge_law=_law_report(
                LawKind.L2_MERGE,
                available=True,
                evidence_status=EvidenceStatus.APPROX_AUDITED,
                objective_enforced=c3_active,
                notes="Approximate merge-law supervision on tree merges.",
            ),
            idempotence_law=_law_report(
                LawKind.L3_IDEMPOTENCE,
                available=True,
                evidence_status=EvidenceStatus.APPROX_AUDITED,
                objective_enforced=c2_active,
                notes="Canonical theorem-facing C2 uses score drift under re-summary.",
            ),
            notes=(
                "Approximate/audited tree-local-law experiment for the paper-facing full-doc lane.",
            ),
        )
    return OperatorCapabilityReport(
        operator_name=family or "full_doc_control",
        evidence_status=EvidenceStatus.PROXY_ONLY,
        latent_mergeability_enforced=False,
        tree_nesting_supported=False,
        theorem_domain_decode_available=False,
        theorem_domain_reencode_available=False,
        exact_reduction_supported=False,
        leaf_law=_law_report(
            LawKind.L1_LEAF,
            available=False,
            evidence_status=EvidenceStatus.PROXY_ONLY,
        ),
        merge_law=_law_report(
            LawKind.L2_MERGE,
            available=False,
            evidence_status=EvidenceStatus.PROXY_ONLY,
        ),
        idempotence_law=_law_report(
            LawKind.L3_IDEMPOTENCE,
            available=False,
            evidence_status=EvidenceStatus.PROXY_ONLY,
        ),
        notes=(
            "Closed-form or generic proxy baseline with no active theorem-facing local-law objective.",
        ),
    )


def full_doc_baseline_provenance(
    baseline_family: str,
    *,
    objective_weights_active: bool,
    config_like: Any | None = None,
    c2_metric_kind: str = "",
    c2_proxy_metric_kind: str = "",
    c2_exact_witness_kind: str = "",
    mean_leaves_per_doc: float | None = None,
) -> Dict[str, Any]:
    family = str(baseline_family or "").strip().lower()
    capability = full_doc_operator_capability_report(family)
    if family in OFFICIAL_FNO_BASELINE_FAMILIES:
        backend_name = "official_neuraloperator_fno"
        backend_package = "neuraloperator"
        backend_version = _package_version("neuraloperator")
        operator_class = "neuralop.models.FNO"
        theorem_relevance = False
    elif family in TREE_NEURAL_BASELINE_FAMILIES:
        backend_name = "tree_neural_neuraloperator"
        backend_package = "neuraloperator"
        backend_version = _package_version("neuraloperator")
        operator_class = (
            "src.ctreepo.sim.core.markov_neural_operator_baselines.FNOCountSketch"
        )
        theorem_relevance = True
    elif family == "palette_block_exact":
        backend_name = "exact_generator_control"
        backend_package = "local_python"
        backend_version = ""
        operator_class = "palette_block_exact"
        theorem_relevance = True
    elif family in {"tree_doc_ridge", "tree_ridge_leaf", "raw_token_ngram_ridge", "ridge_control"}:
        backend_name = "ridge_control"
        backend_package = "local_python"
        backend_version = ""
        operator_class = family
        theorem_relevance = False
    elif family in {"cnn1d", "mlp_bigram"}:
        backend_name = "torch_proxy_baseline"
        backend_package = "torch"
        backend_version = _package_version("torch")
        operator_class = family
        theorem_relevance = False
    else:
        backend_name = "local_python"
        backend_package = "local_python"
        backend_version = ""
        operator_class = family or "unknown"
        theorem_relevance = False
    law_contract = full_doc_law_contract_report(
        family,
        config_like=config_like,
        objective_weights_active=bool(objective_weights_active),
        c2_metric_kind=str(c2_metric_kind),
        c2_proxy_metric_kind=str(c2_proxy_metric_kind),
        c2_exact_witness_kind=str(c2_exact_witness_kind),
        mean_leaves_per_doc=mean_leaves_per_doc,
    )
    return {
        "backend_name": str(backend_name),
        "backend_package": str(backend_package),
        "backend_version": str(backend_version),
        "operator_class": str(operator_class),
        "operator_evidence_status": str(capability.evidence_status.name),
        "theorem_relevance": bool(theorem_relevance),
        "objective_weights_active": bool(objective_weights_active),
        **law_contract,
    }


def official_fno_doc_sequence_provenance(*, objective_weights_active: bool = False) -> Dict[str, Any]:
    capability = full_doc_operator_capability_report("official_fno")
    law_contract = full_doc_law_contract_report(
        "official_fno",
        objective_weights_active=bool(objective_weights_active),
    )
    return {
        "backend_name": "official_neuraloperator_fno",
        "backend_package": "neuraloperator",
        "backend_version": _package_version("neuraloperator"),
        "operator_class": "neuralop.models.FNO",
        "operator_evidence_status": str(capability.evidence_status.name),
        "theorem_relevance": False,
        "objective_weights_active": bool(objective_weights_active),
        **law_contract,
    }


__all__ = [
    "CONTROL_BASELINE_FAMILIES",
    "OFFICIAL_FNO_BASELINE_FAMILIES",
    "TREE_NEURAL_BASELINE_FAMILIES",
    "full_doc_baseline_provenance",
    "full_doc_operator_capability_report",
    "official_fno_doc_sequence_provenance",
]
