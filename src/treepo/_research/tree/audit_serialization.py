"""Helpers for serializing audit reports into stable JSON artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict


def _as_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _report_payload(report: Any) -> Dict[str, Any]:
    payload = _as_mapping(report)
    if payload:
        return payload
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        payload = _as_mapping(to_dict())
        if payload:
            return payload
    return {}


def _attr(report: Any, name: str, default: Any) -> Any:
    value = getattr(report, name, default)
    return default if value is None else value


def _float_rate(
    payload: Mapping[str, Any],
    *,
    violations_key: str,
    samples_key: str,
    explicit_rate: Any = None,
) -> float:
    if explicit_rate is not None:
        try:
            return float(explicit_rate)
        except Exception:
            pass
    violations = int(payload.get(violations_key, 0) or 0)
    samples = int(payload.get(samples_key, 0) or 0)
    if samples <= 0:
        return 0.0
    return float(violations) / float(samples)


def audit_report_to_dict(report: Any) -> Dict[str, Any]:
    """Serialize an audit report while preserving compositional-learning metadata."""

    payload = _report_payload(report)
    checks = payload.get("checks")
    if not isinstance(checks, list):
        checks = [
            {
                "node_id": str(getattr(check, "node_id", "")),
                "check_type": str(getattr(check, "check_type", "")),
                "passed": bool(getattr(check, "passed", False)),
                "discrepancy_score": float(getattr(check, "discrepancy_score", 0.0)),
                "reasoning": str(getattr(check, "reasoning", "")),
                "input_a": str(getattr(check, "input_a", "")),
                "input_b": str(getattr(check, "input_b", "")),
                "skipped": bool(getattr(check, "skipped", False)),
                "skip_reason": getattr(check, "skip_reason", None),
                "inclusion_probability": getattr(check, "inclusion_probability", None),
                "sampling_design": getattr(check, "sampling_design", None),
            }
            for check in (_attr(report, "checks", []) or [])
        ]

    base: Dict[str, Any] = {
        "tree_id": str(payload.get("tree_id", _attr(report, "tree_id", "unknown"))),
        "source_doc_id": payload.get("source_doc_id", _attr(report, "source_doc_id", None)),
        "total_nodes": int(payload.get("total_nodes", _attr(report, "total_nodes", 0)) or 0),
        "nodes_audited": int(payload.get("nodes_audited", _attr(report, "nodes_audited", 0)) or 0),
        "nodes_passed": int(payload.get("nodes_passed", _attr(report, "nodes_passed", 0)) or 0),
        "nodes_failed": int(payload.get("nodes_failed", _attr(report, "nodes_failed", 0)) or 0),
        "failure_rate": float(payload.get("failure_rate", _attr(report, "failure_rate", 0.0)) or 0.0),
        "checks": checks,
        "failed_node_ids": list(payload.get("failed_node_ids", _attr(report, "failed_node_ids", [])) or []),
        "sufficiency_violations": int(
            payload.get("sufficiency_violations", _attr(report, "sufficiency_violations", 0)) or 0
        ),
        "merge_violations": int(
            payload.get("merge_violations", _attr(report, "merge_violations", 0)) or 0
        ),
        "idempotence_violations": int(
            payload.get("idempotence_violations", _attr(report, "idempotence_violations", 0)) or 0
        ),
        "substitution_violations": int(
            payload.get("substitution_violations", _attr(report, "substitution_violations", 0)) or 0
        ),
        "sufficiency_samples": int(
            payload.get("sufficiency_samples", _attr(report, "sufficiency_samples", 0)) or 0
        ),
        "merge_samples": int(payload.get("merge_samples", _attr(report, "merge_samples", 0)) or 0),
        "idempotence_samples": int(
            payload.get("idempotence_samples", _attr(report, "idempotence_samples", 0)) or 0
        ),
        "substitution_samples": int(
            payload.get("substitution_samples", _attr(report, "substitution_samples", 0)) or 0
        ),
        "leaf_population": int(payload.get("leaf_population", _attr(report, "leaf_population", 0)) or 0),
        "merge_population": int(payload.get("merge_population", _attr(report, "merge_population", 0)) or 0),
        "idempotence_population": int(
            payload.get("idempotence_population", _attr(report, "idempotence_population", 0)) or 0
        ),
        "substitution_population": int(
            payload.get("substitution_population", _attr(report, "substitution_population", 0)) or 0
        ),
        "sampling_strategy": str(
            payload.get("sampling_strategy", _attr(report, "sampling_strategy", "random")) or "random"
        ),
        "sampling_probability": float(
            payload.get("sampling_probability", _attr(report, "sampling_probability", 1.0)) or 1.0
        ),
        "operator_capabilities": _as_mapping(
            payload.get("operator_capabilities", _attr(report, "operator_capabilities", {}))
        ),
        "compositional_learning_problem": _as_mapping(
            payload.get(
                "compositional_learning_problem",
                _attr(report, "compositional_learning_problem", {}),
            )
        ),
        "logged_observations": list(
            payload.get("logged_observations", _attr(report, "logged_observations", [])) or []
        ),
        "logged_observation_artifacts": _as_mapping(
            payload.get(
                "logged_observation_artifacts",
                _attr(report, "logged_observation_artifacts", {}),
            )
        ),
        "inclusion_probability_map": _as_mapping(
            payload.get("inclusion_probability_map", _attr(report, "inclusion_probability_map", {}))
        ),
    }

    base["violation_rates"] = {
        "sufficiency": {
            "violations": int(base["sufficiency_violations"]),
            "samples": int(base["sufficiency_samples"]),
            "rate": _float_rate(
                base,
                violations_key="sufficiency_violations",
                samples_key="sufficiency_samples",
                explicit_rate=getattr(report, "sufficiency_rate", None),
            ),
        },
        "merge_consistency": {
            "violations": int(base["merge_violations"]),
            "samples": int(base["merge_samples"]),
            "rate": _float_rate(
                base,
                violations_key="merge_violations",
                samples_key="merge_samples",
                explicit_rate=getattr(report, "merge_rate", None),
            ),
        },
        "idempotence": {
            "violations": int(base["idempotence_violations"]),
            "samples": int(base["idempotence_samples"]),
            "rate": _float_rate(
                base,
                violations_key="idempotence_violations",
                samples_key="idempotence_samples",
                explicit_rate=getattr(report, "idempotence_rate", None),
            ),
        },
        "substitution": {
            "violations": int(base["substitution_violations"]),
            "samples": int(base["substitution_samples"]),
            "rate": _float_rate(
                base,
                violations_key="substitution_violations",
                samples_key="substitution_samples",
                explicit_rate=getattr(report, "substitution_rate", None),
            ),
        },
        "assoc": float(getattr(report, "assoc_rate", 0.0) or 0.0),
    }
    return base


def audit_problem_manifest(report_payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the manifest-level learning-problem summary for an audit run."""

    payload = dict(report_payload)
    problem = _as_mapping(payload.get("compositional_learning_problem", {}))
    return {
        "problem_name": str(problem.get("name", "") or ""),
        "uses_full_document_labels": bool(problem.get("uses_full_document_labels", False)),
        "uses_sampled_substructure_labels": bool(
            problem.get("uses_sampled_substructure_labels", False)
        ),
        "uses_online_oracle_queries": bool(
            problem.get("uses_online_oracle_queries", False)
        ),
        "requires_propensity_logging": bool(problem.get("requires_propensity_logging", False)),
        "supports_theorem_backing": bool(problem.get("supports_theorem_backing", False)),
        "supervision_channels": [
            dict(channel)
            for channel in (problem.get("supervision_channels", []) or [])
            if isinstance(channel, Mapping)
        ],
        "theorem_assumptions": (
            problem.get("theorem_assumptions")
            if isinstance(problem.get("theorem_assumptions"), Mapping)
            else None
        ),
        "operator_assumptions": (
            problem.get("operator_assumptions")
            if isinstance(problem.get("operator_assumptions"), Mapping)
            else None
        ),
        "operator_capabilities": (
            payload.get("operator_capabilities")
            if isinstance(payload.get("operator_capabilities"), Mapping)
            else (
                problem.get("operator_capabilities")
                if isinstance(problem.get("operator_capabilities"), Mapping)
                else None
            )
        ),
        "sampling_strategy": str(payload.get("sampling_strategy", "") or ""),
        "sampling_probability": float(payload.get("sampling_probability", 1.0) or 1.0),
        "logged_inclusion_probabilities": bool(payload.get("inclusion_probability_map")),
        "logged_observation_artifacts": _as_mapping(payload.get("logged_observation_artifacts", {})),
    }


__all__ = ["audit_report_to_dict", "audit_problem_manifest"]
