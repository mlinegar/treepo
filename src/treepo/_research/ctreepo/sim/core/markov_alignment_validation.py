from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from treepo._research.ctreepo.sim.core.full_doc_anchor_diagnostics import (
    DEV_SELECTION_METRIC,
    FULL_DOC_ONLY_BUDGET_FAMILIES,
    ORACLE_BUDGET_STUDY_NAME,
    PRIMARY_REPORT_METRIC,
    PRIMARY_REPORT_SPLIT,
    PRIMARY_REPORT_TARGET,
    PRIMARY_REPORT_WEIGHTING,
    TREE_NEURAL_BASELINE_FAMILIES,
    _is_headline_comparison_semantics,
    load_markov_full_doc_anchor_diagnostics_from_output_dir,
)
from treepo._research.ctreepo.sim.core.full_tree_ipw_grid import grid_rows_from_payload
from treepo._research.ctreepo.sim.core.markov_v3_row_contract import (
    CONTRACT_STATUS_LEGACY_QUARANTINED,
    annotate_downstream_v3_row,
    filtered_quarantined_rows,
    quarantine_sources_from_rows,
)
from treepo._research.ctreepo.sim.core.markov_alignment_spec import (
    MARKOV_ALIGNMENT_AUDIT_NOTE_PATH,
    MARKOV_FULL_DOC_OBJECTIVE_SURFACE,
    PAPER_TO_LEAN_LOCAL_LAW_MAPPING,
    TREEPO_REGULARIZED_OBJECTIVE_SURFACE,
    markov_alignment_spec,
    required_audit_note_phrases,
)
from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (
    BUDGETED_SAMPLING_SCHEME_RANDOM_WITHOUT_REPLACEMENT,
    OPSCountConfig,
    _build_objective_summary,
    _doc_leaf_and_internal_spans,
)
from treepo._research.ctreepo.sim.expectations import (
    ExpectationConfig,
    ExpectationReport,
    FAMILY_MARKOV,
    MarkovOPSAdapter,
)
from treepo._research.ctreepo.sim.theory_alignment import build_simulation_theory_alignment_report


REPO_ROOT = Path(__file__).resolve().parents[4]
LEAN_ROOT = REPO_ROOT / "lean3"
LEAN_BUILD_TARGET = "FormalProofs.OPT.MainTheorems"
FULL_TREE_IPW_SIMULATION = "markov_full_tree_ipw_grid"


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    status: str
    severity: str
    summary: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarkovAlignmentAuditReport:
    diagnostics_root: Optional[str]
    full_tree_ipw_root: Optional[str]
    ladder_json: Optional[str]
    bundle_manifest: Optional[str]
    alignment_spec: Dict[str, Any]
    surface_coverage: Dict[str, Any]
    lean_build: Dict[str, Any]
    expectation_report: Dict[str, Any]
    theory_alignment_report: Dict[str, Any]
    checks: List[ValidationCheck]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnostics_root": self.diagnostics_root,
            "full_tree_ipw_root": self.full_tree_ipw_root,
            "ladder_json": self.ladder_json,
            "bundle_manifest": self.bundle_manifest,
            "alignment_spec": dict(self.alignment_spec),
            "surface_coverage": dict(self.surface_coverage),
            "lean_build": dict(self.lean_build),
            "expectation_report": dict(self.expectation_report),
            "theory_alignment_report": dict(self.theory_alignment_report),
            "checks": [check.to_dict() for check in self.checks],
            "summary": dict(self.summary),
        }


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _finite(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return float(out)


def _is_finite(value: object) -> bool:
    return math.isfinite(_finite(value))


def _close(a: object, b: object, *, atol: float = 1e-9) -> bool:
    av = _finite(a)
    bv = _finite(b)
    if math.isnan(av) and math.isnan(bv):
        return True
    return math.isfinite(av) and math.isfinite(bv) and abs(av - bv) <= float(atol)


def _is_missing_metadata_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (float, np.floating)):
        return not np.isfinite(float(value))
    return False


def _expectation_report_for_extra_paths(*, paths: Sequence[Path]) -> ExpectationReport:
    adapter = MarkovOPSAdapter()
    rows = []
    for path in paths:
        resolved = path.resolve()
        if not adapter.can_load(resolved):
            continue
        rows.extend(adapter.load_rows(resolved))
    expectations = adapter.build_expectations(rows, config=ExpectationConfig()) if rows else []
    return ExpectationReport(
        input_root=None,
        manifest=None,
        families_scanned=[FAMILY_MARKOV] if rows else [],
        rows_scanned=int(len(rows)),
        expectations=expectations,
        summary={
            "n_pass": int(sum(1 for item in expectations if item.status == "pass")),
            "n_warn": int(sum(1 for item in expectations if item.status == "warn")),
            "n_fail": int(sum(1 for item in expectations if item.status == "fail")),
            "n_not_applicable": int(
                sum(1 for item in expectations if item.status == "not_applicable")
            ),
            "families_with_failures": sorted(
                {item.family for item in expectations if item.status == "fail"}
            ),
            "highest_priority_findings": [
                {
                    "status": item.status,
                    "family": item.family,
                    "title": item.title,
                    "scenario": item.scenario,
                }
                for item in expectations[:10]
            ],
        },
    )


def _run_lean_build() -> Dict[str, Any]:
    proc = subprocess.run(
        ["lake", "build", LEAN_BUILD_TARGET],
        cwd=str(LEAN_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "requested": True,
        "command": f"lake build {LEAN_BUILD_TARGET}",
        "returncode": int(proc.returncode),
        "passed": bool(proc.returncode == 0),
        "stdout_tail": "\n".join(str(proc.stdout).splitlines()[-20:]),
        "stderr_tail": "\n".join(str(proc.stderr).splitlines()[-20:]),
    }


def _load_full_tree_ipw_payload(
    root_or_file: Path | None,
) -> tuple[Optional[Dict[str, Any]], Optional[Path]]:
    if root_or_file is None:
        return None, None
    path = root_or_file.resolve()
    if path.is_file():
        return _load_json(path), path
    summary_json = path / "summary.json"
    if summary_json.exists():
        return _load_json(summary_json), summary_json
    return None, None


def _surface_coverage_summary(
    *,
    diagnostics_payload: Mapping[str, Any] | None,
    ladder_payload: Mapping[str, Any] | None,
    full_tree_payload: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    return {
        "has_full_doc_diagnostics": bool(diagnostics_payload),
        "has_full_doc_ladder": bool(ladder_payload),
        "has_full_tree_ipw_grid": bool(full_tree_payload),
        "full_doc_run_count": int(len(list((diagnostics_payload or {}).get("runs") or []))),
        "full_doc_aggregate_count": int(
            len(list((diagnostics_payload or {}).get("aggregate_rows") or []))
        ),
        "ladder_stage_count": int(len(list((ladder_payload or {}).get("stages") or []))),
        "full_tree_plane_count": int(len(list((full_tree_payload or {}).get("planes") or []))),
    }


def _aggregate_key_fields() -> tuple[str, ...]:
    return (
        "benchmark",
        "cell_id",
        "baseline_family",
        "train_doc_count",
        "fixed_leaf_tokens",
        "comparison_mode",
        "comparison_semantics_label",
        "run_intent_hash",
        "parameterization",
        "weighting_scheme",
        "optimization_root_weight",
        "local_law_c1_weight",
        "local_law_c2_weight",
        "local_law_c3_weight",
        "task_objective_weight_source",
        "proxy_schedule_consistency_weight",
        "c2_metric_kind",
        "backend_name",
        "backend_package",
        "backend_version",
        "operator_class",
        "operator_evidence_status",
        "theorem_relevance",
        "objective_weights_active",
        "tree_root_supervision_kind",
        "tree_leaf_fno_width",
        "tree_leaf_fno_n_modes",
        "tree_leaf_fno_n_layers",
        "tree_aux_doc_sequence_fraction",
        "config_label",
        "tuning_stage",
        "study_name",
        "study_axis",
        "axis_value",
        "locked_tree_neural_config_label",
        "selection_metric",
        "budget_total_calls",
        "budget_total_calls_per_doc",
        "full_doc_budget_share",
        "doc_consumption_mode",
        "local_split_mode",
        "local_allocation_policy",
    )


def _duplicate_aggregate_check(payload: Mapping[str, Any]) -> ValidationCheck:
    key_fields = _aggregate_key_fields()
    seen: Dict[tuple[Any, ...], int] = {}
    duplicates: List[Dict[str, Any]] = []
    for row in list(payload.get("aggregate_rows") or []):
        key = tuple(row.get(field) for field in key_fields)
        seen[key] = int(seen.get(key, 0)) + 1
        if int(seen[key]) > 1:
            duplicates.append(dict(row))
    return ValidationCheck(
        name="aggregate_grouping",
        status="fail" if duplicates else "pass",
        severity="hard",
        summary=(
            "aggregate rows are unique under the semantics/provenance grouping key"
            if not duplicates
            else "duplicate aggregate rows detected under the semantics/provenance grouping key"
        ),
        details={
            "key_fields": list(key_fields),
            "n_aggregate_rows": int(len(list(payload.get("aggregate_rows") or []))),
            "n_duplicates": int(len(duplicates)),
            "sample_duplicates": duplicates[:5],
        },
    )


def _score_contract_check(payload: Mapping[str, Any]) -> ValidationCheck:
    expected = {
        "primary_report_metric": PRIMARY_REPORT_METRIC,
        "primary_report_split": PRIMARY_REPORT_SPLIT,
        "primary_report_target": PRIMARY_REPORT_TARGET,
        "primary_report_weighting": PRIMARY_REPORT_WEIGHTING,
        "dev_selection_metric": DEV_SELECTION_METRIC,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    return ValidationCheck(
        name="score_contract",
        status="fail" if mismatches else "pass",
        severity="hard",
        summary=(
            "full-doc summary exposes the canonical test-first score contract"
            if not mismatches
            else "full-doc summary is missing or violating the canonical test-first score contract"
        ),
        details={"expected": expected, "mismatches": mismatches},
    )


def _official_fno_provenance_check(payload: Mapping[str, Any]) -> ValidationCheck:
    official_rows = [
        dict(row)
        for row in list(payload.get("runs") or [])
        if str(row.get("baseline_family", "")) in {"official_fno", "official_fno_sumlen"}
    ]
    bad_rows = [
        row
        for row in official_rows
        if str(row.get("backend_package", "")) != "neuraloperator"
        or str(row.get("operator_class", "")) != "neuralop.models.FNO"
        or not str(row.get("backend_version", "")).strip()
        or str(row.get("operator_evidence_status", "")) != "PROXY_ONLY"
    ]
    status = "warn" if not official_rows else ("fail" if bad_rows else "pass")
    return ValidationCheck(
        name="official_fno_provenance",
        status=status,
        severity="hard",
        summary=(
            "official FNO rows expose installed neuraloperator provenance"
            if status == "pass"
            else (
                "no official FNO rows were present"
                if status == "warn"
                else "official FNO rows are missing required official-backend provenance"
            )
        ),
        details={
            "n_rows": int(len(official_rows)),
            "n_bad_rows": int(len(bad_rows)),
            "sample_bad_rows": bad_rows[:5],
        },
    )


def _current_semantics_check(payload: Mapping[str, Any]) -> ValidationCheck:
    current_rows = [
        dict(row)
        for row in list(payload.get("runs") or [])
        if str(row.get("baseline_family", "")).startswith("tree_neural")
        and _is_headline_comparison_semantics(
            str(row.get("comparison_semantics", ""))
        )
    ]
    bad_rows = [
        row
        for row in current_rows
        if not str(row.get("c2_metric_kind", "")).strip()
        or not str(row.get("c2_proxy_metric_kind", "")).strip()
        or not str(row.get("semantics_version", "")).strip()
        or not str(row.get("parameterization", "")).strip()
        or not str(row.get("weighting_scheme", "")).strip()
        or str(row.get("operator_evidence_status", "")) != "APPROX_AUDITED"
        or "backend_name" not in row
        or "proxy_terms" not in row
        or "theorem_terms" not in row
        or not str(row.get("law_contract_version", "")).strip()
        or not str(row.get("family_api_group", "")).strip()
        or not str(row.get("law_alignment_status", "")).strip()
        or not isinstance(row.get("law_contract"), dict)
    ]
    status = "warn" if not current_rows else ("fail" if bad_rows else "pass")
    return ValidationCheck(
        name="current_tree_neural_semantics",
        status=status,
        severity="hard",
        summary=(
            "current tree-neural rows are semantics-complete"
            if status == "pass"
            else (
                "no current tree-neural rows were present"
                if status == "warn"
                else "current tree-neural rows are missing theorem-facing semantics metadata"
            )
        ),
        details={
            "n_rows": int(len(current_rows)),
            "n_bad_rows": int(len(bad_rows)),
            "sample_bad_rows": bad_rows[:5],
        },
    )


def _law_contract_gap_warning(payload: Mapping[str, Any]) -> ValidationCheck:
    current_tree_rows = [
        dict(row)
        for row in list(payload.get("runs") or [])
        if str(row.get("baseline_family", "")).startswith("tree_neural")
        and _is_headline_comparison_semantics(
            str(row.get("comparison_semantics", ""))
        )
    ]
    gap_rows = [
        row
        for row in current_tree_rows
        if int(row.get("law_contract_gap_count", 0) or 0) > 0
    ]
    return ValidationCheck(
        name="law_contract_gaps",
        status="warn" if gap_rows else ("warn" if not current_tree_rows else "pass"),
        severity="warning",
        summary=(
            "current tree-neural rows expose no law-contract gap warnings"
            if current_tree_rows and not gap_rows
            else (
                "no current tree-neural rows were present"
                if not current_tree_rows
                else "current tree-neural rows include law-contract gap warnings"
            )
        ),
        details={
            "n_rows": int(len(current_tree_rows)),
            "n_gap_rows": int(len(gap_rows)),
            "sample_gap_rows": gap_rows[:5],
        },
    )


def _law_contract_limitation_warning(payload: Mapping[str, Any]) -> ValidationCheck:
    current_tree_rows = [
        dict(row)
        for row in list(payload.get("runs") or [])
        if str(row.get("baseline_family", "")).startswith("tree_neural")
        and _is_headline_comparison_semantics(
            str(row.get("comparison_semantics", ""))
        )
    ]
    limitation_rows = [
        row
        for row in current_tree_rows
        if int(row.get("law_contract_limitation_count", 0) or 0) > 0
    ]
    return ValidationCheck(
        name="law_contract_limitations",
        status=(
            "warn"
            if limitation_rows
            else ("warn" if not current_tree_rows else "pass")
        ),
        severity="warning",
        summary=(
            "current tree-neural rows expose no architectural law-contract limitations"
            if current_tree_rows and not limitation_rows
            else (
                "no current tree-neural rows were present"
                if not current_tree_rows
                else "current tree-neural rows include architectural law-contract limitations"
            )
        ),
        details={
            "n_rows": int(len(current_tree_rows)),
            "n_limitation_rows": int(len(limitation_rows)),
            "sample_limitation_rows": limitation_rows[:5],
        },
    )


def _legacy_current_mixing_check(payload: Mapping[str, Any]) -> ValidationCheck:
    tree_rows = [
        dict(row)
        for row in list(payload.get("runs") or [])
        if str(row.get("baseline_family", "")).startswith("tree_neural")
    ]
    labels_by_mode: Dict[str, set[str]] = {}
    bad_rows: List[Dict[str, Any]] = []
    for row in tree_rows:
        mode = str(row.get("comparison_semantics", ""))
        label = str(row.get("comparison_semantics_label", ""))
        labels_by_mode.setdefault(mode, set()).add(label)
        if mode == "legacy" and (
            not bool(row.get("legacy_semantics", False)) or not label.strip()
        ):
            bad_rows.append(row)
    shared_labels = sorted(labels_by_mode.get("legacy", set()) & labels_by_mode.get("current", set()))
    if shared_labels:
        bad_rows.extend(tree_rows)
    status = "warn" if not tree_rows else ("fail" if bad_rows else "pass")
    return ValidationCheck(
        name="legacy_current_tree_neural_mixing",
        status=status,
        severity="hard",
        summary=(
            "legacy and current tree-neural rows stay explicitly separated"
            if status == "pass"
            else (
                "no tree-neural rows were present"
                if status == "warn"
                else "legacy/current tree-neural rows are not explicitly separated"
            )
        ),
        details={
            "modes_present": sorted(labels_by_mode.keys()),
            "labels_by_mode": {key: sorted(values) for key, values in labels_by_mode.items()},
            "shared_labels": shared_labels,
            "sample_bad_rows": bad_rows[:5],
        },
    )


def _ladder_pairing_check(ladder_payload: Optional[Mapping[str, Any]]) -> ValidationCheck:
    if not ladder_payload:
        return ValidationCheck(
            name="ladder_reference_reproduction_pairing",
            status="warn",
            severity="hard",
            summary="no ladder payload was provided",
            details={},
        )
    stage_by_name = {
        str(stage.get("stage_name", "")): dict(stage)
        for stage in list(ladder_payload.get("stages") or [])
    }
    mismatches: Dict[str, Dict[str, Any]] = {}
    n_pairs = 0
    compared_fields = (
        "observed_token_profile",
        "bundle_source",
        "train_docs",
        "val_docs",
        "test_docs",
        "state_dim",
        "hidden_dim",
        "n_epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "doc_sequence_backend_package",
        "doc_sequence_operator_class",
    )
    for stage_name, reproduction in stage_by_name.items():
        if not stage_name.endswith("_reproduction"):
            continue
        reference_name = stage_name[: -len("_reproduction")] + "_reference"
        reference = stage_by_name.get(reference_name)
        if reference is None:
            continue
        n_pairs += 1
        field_mismatches: Dict[str, Any] = {}
        for field_name in compared_fields:
            if reproduction.get(field_name) != reference.get(field_name):
                field_mismatches[field_name] = {
                    "reproduction": reproduction.get(field_name),
                    "reference": reference.get(field_name),
                }
        if field_mismatches:
            mismatches[stage_name] = field_mismatches
    status = "warn" if n_pairs == 0 else ("fail" if mismatches else "pass")
    return ValidationCheck(
        name="ladder_reference_reproduction_pairing",
        status=status,
        severity="hard",
        summary=(
            "ladder reference/reproduction pairs stay bundle/config matched"
            if status == "pass"
            else (
                "no ladder reference/reproduction pair was present"
                if status == "warn"
                else "ladder reference/reproduction pairs diverged on bundle or declared config"
            )
        ),
        details={
            "n_pairs": int(n_pairs),
            "compared_fields": list(compared_fields),
            "mismatches": mismatches,
        },
    )


def _load_family_grids_summary_payload(path: Path | None) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    payload = _load_json(path.resolve())
    return payload if payload else None


def _load_parity_root_payloads(roots: Sequence[Path]) -> List[Dict[str, Any]]:
    if not roots:
        return []
    from scripts.run_markov_supervision_recovery_parity_grid import load_parity_grid_root

    payloads: List[Dict[str, Any]] = []
    for root in roots:
        resolved = root.resolve()
        if not resolved.exists():
            continue
        payloads.append(dict(load_parity_grid_root(resolved)))
    return payloads


def _family_grids_recovery_rows(
    family_grids_summary: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if not family_grids_summary:
        return []
    recovery = dict((family_grids_summary.get("supervision_recovery") or {}))
    rows: List[Dict[str, Any]] = []
    raw_rows = list(recovery.get("all_family_rows") or [])
    if not raw_rows:
        raw_rows = list(recovery.get("family_rows") or []) + list(
            recovery.get("quarantined_family_rows") or []
        )
    for raw_row in raw_rows:
        rows.append(
            annotate_downstream_v3_row(
                dict(raw_row or {}),
                canonical_fno_families=("official_fno", "official_fno_sumlen"),
                canonical_fno_fixed_leaf_tokens=128,
            )
        )
    return rows


def _family_grids_v3_provenance_check(
    family_grids_summary: Optional[Mapping[str, Any]],
) -> ValidationCheck:
    if not family_grids_summary:
        return ValidationCheck(
            name="family_grids_v3_provenance",
            status="warn",
            severity="hard",
            summary="no family-grids summary was provided",
            details={},
        )
    rows = _family_grids_recovery_rows(family_grids_summary)
    bad_rows = [
        dict(row)
        for row in rows
        if _as_contract_status(row.get("contract_status"))
        == CONTRACT_STATUS_LEGACY_QUARANTINED
    ]
    status = "warn" if not rows else ("fail" if bad_rows else "pass")
    return ValidationCheck(
        name="family_grids_v3_provenance",
        status=status,
        severity="hard",
        summary=(
            "family-grid recovery rows carry complete v3 semantics and run-intent provenance"
            if status == "pass"
            else (
                "no recovery family rows were present in the family-grids summary"
                if status == "warn"
                else "family-grid recovery rows are missing required v3 provenance or semantics"
            )
        ),
        details={
            "n_rows": int(len(rows)),
            "n_bad_rows": int(len(bad_rows)),
            "sample_bad_rows": bad_rows[:5],
        },
    )


def _family_grids_quarantine_enforcement_check(
    family_grids_summary: Optional[Mapping[str, Any]],
) -> ValidationCheck:
    if not family_grids_summary:
        return ValidationCheck(
            name="family_grids_quarantine_enforcement",
            status="warn",
            severity="hard",
            summary="no family-grids summary was provided",
            details={},
        )
    recovery = dict((family_grids_summary.get("supervision_recovery") or {}))
    headline_rows = [
        annotate_downstream_v3_row(
            dict(row or {}),
            canonical_fno_families=("official_fno", "official_fno_sumlen"),
            canonical_fno_fixed_leaf_tokens=128,
        )
        for row in list(recovery.get("family_rows") or [])
    ]
    headline_quarantined = filtered_quarantined_rows(headline_rows)
    quarantined_rows = filtered_quarantined_rows(
        _family_grids_recovery_rows(family_grids_summary)
    )
    reported_count = int(
        (family_grids_summary.get("quarantined_row_count", recovery.get("quarantined_row_count", 0)) or 0)
    )
    reported_sources = {
        str(value)
        for value in list(
            family_grids_summary.get(
                "quarantined_sources",
                recovery.get("quarantined_sources", []),
            )
            or []
        )
        if str(value).strip()
    }
    actual_sources = set(quarantine_sources_from_rows(quarantined_rows))
    mismatches: List[Dict[str, Any]] = []
    if headline_quarantined:
        mismatches.append(
            {
                "reason": "headline_rows_include_quarantined_entries",
                "n_rows": int(len(headline_quarantined)),
            }
        )
    if reported_count != len(quarantined_rows):
        mismatches.append(
            {
                "reason": "quarantined_row_count_mismatch",
                "reported": int(reported_count),
                "actual": int(len(quarantined_rows)),
            }
        )
    if actual_sources and reported_sources != actual_sources:
        mismatches.append(
            {
                "reason": "quarantined_sources_mismatch",
                "reported": sorted(reported_sources),
                "actual": sorted(actual_sources),
            }
        )
    status = "warn" if not headline_rows and not quarantined_rows else ("fail" if mismatches else "pass")
    return ValidationCheck(
        name="family_grids_quarantine_enforcement",
        status=status,
        severity="hard",
        summary=(
            "family-grid recovery summaries quarantine incomplete rows out of headline evidence"
            if status == "pass"
            else (
                "no family-grid recovery rows were present"
                if status == "warn"
                else "family-grid recovery summaries leak quarantined rows or misreport quarantine counts"
            )
        ),
        details={
            "n_headline_rows": int(len(headline_rows)),
            "n_quarantined_rows": int(len(quarantined_rows)),
            "quarantined_sources": sorted(actual_sources),
            "mismatches": mismatches,
        },
    )


def _canonical_fno_geometry_check(
    family_grids_summary: Optional[Mapping[str, Any]],
    parity_root_payloads: Sequence[Mapping[str, Any]],
) -> ValidationCheck:
    recovery_rows = _family_grids_recovery_rows(family_grids_summary)
    bad_rows: List[Dict[str, Any]] = []
    fno_rows = [
        row
        for row in recovery_rows
        if str(row.get("baseline_family", "")).strip()
        in {"official_fno", "official_fno_sumlen"}
    ]
    for row in fno_rows:
        if int(row.get("executed_fixed_leaf_tokens", 0) or 0) != 128:
            bad_rows.append(
                {
                    "surface": "family_grids",
                    "scope_key": row.get("scope_key"),
                    "package_name": row.get("package_name"),
                    "baseline_family": row.get("baseline_family"),
                    "executed_fixed_leaf_tokens": row.get(
                        "executed_fixed_leaf_tokens"
                    ),
                }
            )
    for payload in parity_root_payloads:
        for raw_row in list(payload.get("rows") or []):
            row = dict(raw_row or {})
            if str(row.get("baseline_family", "")).strip() not in {
                "official_fno",
                "official_fno_sumlen",
            }:
                continue
            if str(row.get("claim_level", "")).strip() == "empirical_geometry":
                continue
            if int(row.get("fixed_leaf_tokens", 0) or 0) != 128:
                bad_rows.append(
                    {
                        "surface": "parity_grid",
                        "job_name": row.get("job_name"),
                        "baseline_family": row.get("baseline_family"),
                        "fixed_leaf_tokens": row.get("fixed_leaf_tokens"),
                    }
                )
    rows_checked = int(len(fno_rows)) + int(
        sum(
            1
            for payload in parity_root_payloads
            for raw_row in list(payload.get("rows") or [])
            if str((raw_row or {}).get("baseline_family", "")).strip()
            in {"official_fno", "official_fno_sumlen"}
            and str((raw_row or {}).get("claim_level", "")).strip()
            != "empirical_geometry"
        )
    )
    status = "warn" if rows_checked == 0 else ("fail" if bad_rows else "pass")
    return ValidationCheck(
        name="canonical_fno_geometry",
        status=status,
        severity="hard",
        summary=(
            "official FNO reference rows stay locked to full-doc leaf128 on recovery/parity surfaces"
            if status == "pass"
            else (
                "no official FNO reference rows were present on recovery/parity surfaces"
                if status == "warn"
                else "official FNO reference rows drifted away from full-doc leaf128"
            )
        ),
        details={
            "rows_checked": rows_checked,
            "n_bad_rows": int(len(bad_rows)),
            "sample_bad_rows": bad_rows[:5],
        },
    )


def _as_contract_status(value: object) -> str:
    return str(value or "").strip()


def _prepared_metadata_from_run(run: Mapping[str, Any]) -> Dict[str, Any]:
    config = dict(run.get("config") or {})
    prepared_root = str(config.get("prepared_data_root", "") or "").strip()
    prepared_signature = str(config.get("prepared_data_signature", "") or "").strip()
    if not prepared_root or not prepared_signature:
        return {}
    candidate_paths = (
        Path(prepared_root).expanduser() / f"prepared_{prepared_signature}" / "metadata.json",
        Path(prepared_root).expanduser() / "metadata.json",
    )
    for metadata_path in candidate_paths:
        payload = _load_json(metadata_path)
        if payload:
            return {
                "metadata_json": str(metadata_path),
                "train_prefix_counts": [
                    int(value) for value in list(payload.get("train_prefix_counts") or [])
                ],
                "train_prefix_signatures": {
                    str(key): str(value)
                    for key, value in dict(payload.get("train_prefix_signatures") or {}).items()
                    if str(key).strip() and str(value).strip()
                },
                "train_corpus_signature": str(payload.get("train_corpus_signature", "") or ""),
                "val_corpus_signature": str(payload.get("val_corpus_signature", "") or ""),
                "test_corpus_signature": str(payload.get("test_corpus_signature", "") or ""),
            }
    return {}


def _rung_nestedness_check(
    payload: Optional[Mapping[str, Any]],
    *,
    canonical_train_ladder: Sequence[int],
    authoritative_claimed: bool = False,
) -> ValidationCheck:
    if not payload:
        return ValidationCheck(
            name="rung_nestedness",
            status="warn",
            severity="hard",
            summary="no diagnostics payload was provided for rung nestedness",
            details={"canonical_train_ladder": [int(value) for value in canonical_train_ladder]},
        )
    target = tuple(int(value) for value in canonical_train_ladder)
    tree_runs = [
        dict(run)
        for run in list(payload.get("runs") or [])
        if str(run.get("baseline_family", "")).startswith("tree_neural")
        and int(run.get("train_doc_count", 0) or 0) in set(target)
    ]
    observed_train_docs = sorted({int(run.get("train_doc_count", 0) or 0) for run in tree_runs})
    bad_rows: List[Dict[str, Any]] = []
    signatures_by_key: Dict[tuple[str, int], str] = {}
    for run in tree_runs:
        metadata = _prepared_metadata_from_run(run)
        if not metadata:
            bad_rows.append(
                {
                    "reason": "missing_prepared_metadata",
                    "benchmark": run.get("benchmark"),
                    "train_doc_count": run.get("train_doc_count"),
                    "seed": run.get("seed"),
                }
            )
            continue
        counts = tuple(int(value) for value in list(metadata.get("train_prefix_counts") or []))
        if counts != target:
            bad_rows.append(
                {
                    "reason": "canonical_ladder_mismatch",
                    "benchmark": run.get("benchmark"),
                    "train_doc_count": run.get("train_doc_count"),
                    "actual_counts": list(counts),
                    "expected_counts": list(target),
                }
            )
            continue
        signatures = dict(metadata.get("train_prefix_signatures") or {})
        missing = [int(value) for value in target if str(int(value)) not in signatures]
        if missing:
            bad_rows.append(
                {
                    "reason": "missing_prefix_signatures",
                    "benchmark": run.get("benchmark"),
                    "train_doc_count": run.get("train_doc_count"),
                    "missing": missing,
                }
            )
            continue
        benchmark = str(run.get("benchmark", "") or run.get("cell_id", "") or "")
        for count in target:
            key = (benchmark, int(count))
            signature = str(signatures[str(int(count))])
            current = signatures_by_key.get(key)
            if current is None:
                signatures_by_key[key] = signature
            elif current != signature:
                bad_rows.append(
                    {
                        "reason": "prefix_signature_mismatch",
                        "benchmark": benchmark,
                        "train_doc_count": int(count),
                        "expected": current,
                        "actual": signature,
                    }
                )
    missing_counts = [int(value) for value in target if int(value) not in observed_train_docs]
    if missing_counts:
        bad_rows.append({"reason": "missing_train_rungs", "missing": missing_counts})
    status = "warn" if not tree_runs else (
        ("fail" if authoritative_claimed else "warn") if bad_rows else "pass"
    )
    grouped_signatures: Dict[str, Dict[str, str]] = {}
    for (benchmark, count), signature in signatures_by_key.items():
        grouped_signatures.setdefault(str(benchmark), {})[str(int(count))] = str(signature)
    return ValidationCheck(
        name="rung_nestedness",
        status=status,
        severity="hard",
        summary=(
            "canonical train rungs certify the same nested train prefixes"
            if status == "pass"
            else (
                (
                    "prepared-data train prefixes diverged from the canonical nested ladder, "
                    "but this bundle does not claim authoritative canonical-ladder evidence"
                )
                if bad_rows and not authoritative_claimed
                else (
                    "no tree-neural rows with prepared-data metadata were available for rung nestedness"
                    if status == "warn"
                    else "prepared-data train prefixes diverged from the canonical nested ladder"
                )
            )
        ),
        details={
            "canonical_train_ladder": list(target),
            "authoritative_claimed": bool(authoritative_claimed),
            "observed_train_docs": observed_train_docs,
            "train_prefix_counts": list(target),
            "train_prefix_signatures": grouped_signatures,
            "n_bad_rows": int(len(bad_rows)),
            "sample_bad_rows": bad_rows[:5],
        },
    )


def _bundle_consistency_check(
    payload: Optional[Mapping[str, Any]],
    *,
    canonical_train_ladder: Sequence[int],
) -> ValidationCheck:
    if not payload:
        return ValidationCheck(
            name="bundle_consistency",
            status="warn",
            severity="hard",
            summary="no diagnostics payload was provided for bundle consistency",
            details={"canonical_train_ladder": [int(value) for value in canonical_train_ladder]},
        )
    target = {int(value) for value in canonical_train_ladder}
    grouped: Dict[tuple[str, int], Dict[str, set[str]]] = {}
    for raw_run in list(payload.get("runs") or []):
        run = dict(raw_run)
        train_doc_count = int(run.get("train_doc_count", 0) or 0)
        if train_doc_count not in target:
            continue
        key = (str(run.get("benchmark", "") or run.get("cell_id", "") or ""), train_doc_count)
        group = grouped.setdefault(
            key,
            {
                "bundle_source": set(),
                "train_corpus_signature": set(),
                "val_corpus_signature": set(),
                "test_corpus_signature": set(),
            },
        )
        for field_name in tuple(group):
            value = str(run.get(field_name, "") or "").strip()
            if value:
                group[field_name].add(value)
    bad_groups: List[Dict[str, Any]] = []
    reference_bundle_source: Dict[str, str] = {}
    train_signatures: Dict[str, str] = {}
    val_signatures: Dict[str, str] = {}
    test_signatures: Dict[str, str] = {}
    for (benchmark, train_doc_count), fields_map in grouped.items():
        key_label = f"{benchmark}@{int(train_doc_count)}"
        for field_name, values in fields_map.items():
            if len(values) > 1:
                bad_groups.append(
                    {
                        "benchmark": benchmark,
                        "train_doc_count": int(train_doc_count),
                        "field": field_name,
                        "values": sorted(values),
                    }
                )
        if fields_map["bundle_source"]:
            reference_bundle_source[key_label] = sorted(fields_map["bundle_source"])[0]
        if fields_map["train_corpus_signature"]:
            train_signatures[key_label] = sorted(fields_map["train_corpus_signature"])[0]
        if fields_map["val_corpus_signature"]:
            val_signatures[key_label] = sorted(fields_map["val_corpus_signature"])[0]
        if fields_map["test_corpus_signature"]:
            test_signatures[key_label] = sorted(fields_map["test_corpus_signature"])[0]
    status = "warn" if not grouped else ("fail" if bad_groups else "pass")
    return ValidationCheck(
        name="bundle_consistency",
        status=status,
        severity="hard",
        summary=(
            "rows compared within each figure share the same held-out bundle signatures"
            if status == "pass"
            else (
                "no canonical ladder rows were available for bundle consistency"
                if status == "warn"
                else "rows within the same figure diverged on bundle source or held-out corpus signatures"
            )
        ),
        details={
            "canonical_train_ladder": [int(value) for value in canonical_train_ladder],
            "reference_bundle_source": reference_bundle_source,
            "train_corpus_signature": train_signatures,
            "val_corpus_signature": val_signatures,
            "test_corpus_signature": test_signatures,
            "n_bad_groups": int(len(bad_groups)),
            "sample_bad_groups": bad_groups[:5],
        },
    )


def _package_vs_parity_separation_check(
    family_grids_summary: Optional[Mapping[str, Any]],
) -> ValidationCheck:
    if not family_grids_summary:
        return ValidationCheck(
            name="package_vs_parity_separation",
            status="warn",
            severity="hard",
            summary="no merged family-grids summary was provided",
            details={},
        )
    bad_rows: List[Dict[str, Any]] = []
    for row in list(family_grids_summary.get("rows") or []):
        source_kind = str((row or {}).get("source_kind", "") or "")
        claim_level = str((row or {}).get("claim_level", "") or "")
        if source_kind == "supervision_recovery_parity_grid" or claim_level in {
            "empirical_geometry",
            "exact_collapse_candidate",
        }:
            bad_rows.append(dict(row))
    family_rows = list(
        (((family_grids_summary.get("supervision_recovery") or {}).get("family_rows")) or [])
    )
    for row in family_rows:
        claim_level = str((row or {}).get("claim_level", "") or "")
        if claim_level in {"empirical_geometry", "exact_collapse_candidate"}:
            bad_rows.append(dict(row))
    return ValidationCheck(
        name="package_vs_parity_separation",
        status="fail" if bad_rows else "pass",
        severity="hard",
        summary=(
            "package rows and geometry/parity rows stay in separate report sections"
            if not bad_rows
            else "geometry/parity rows leaked into the package ladder surfaces"
        ),
        details={
            "n_bad_rows": int(len(bad_rows)),
            "sample_bad_rows": bad_rows[:5],
        },
    )


def _stopped_root_exclusion_check(
    parity_root_payloads: Sequence[Mapping[str, Any]],
    *,
    family_grids_summary: Optional[Mapping[str, Any]],
) -> ValidationCheck:
    if not parity_root_payloads:
        return ValidationCheck(
            name="stopped_root_exclusion",
            status="warn",
            severity="hard",
            summary="no parity-grid roots were provided",
            details={},
        )
    stopped_roots = [
        dict(payload)
        for payload in parity_root_payloads
        if str(payload.get("evidence_status", "") or "") == "stopped"
    ]
    used_sources = {
        str(value)
        for value in list((family_grids_summary or {}).get("geometry_parity_sources") or [])
        if str(value).strip()
    }
    bad_roots: List[Dict[str, Any]] = []
    for payload in stopped_roots:
        root = str(payload.get("output_root", "") or "")
        completed_rows = [
            dict(row)
            for row in list(payload.get("rows") or [])
            if str((row or {}).get("state", "")).strip().lower() == "completed"
        ]
        if root in used_sources or completed_rows:
            bad_roots.append(
                {
                    "root": root,
                    "completed_rows": int(len(completed_rows)),
                    "used_as_geometry_source": bool(root in used_sources),
                }
            )
    status = "pass" if not bad_roots else "fail"
    if not stopped_roots and not bad_roots:
        status = "pass"
    return ValidationCheck(
        name="stopped_root_exclusion",
        status=status,
        severity="hard",
        summary=(
            "stopped or stale parity-grid roots are indexed only and never used as evidence"
            if status == "pass"
            else "a stopped parity-grid root still contributes evidence rows or sources"
        ),
        details={
            "n_stopped_roots": int(len(stopped_roots)),
            "n_bad_roots": int(len(bad_roots)),
            "sample_bad_roots": bad_roots[:5],
        },
    )


def _strict_collapse_readiness_check(
    parity_root_payloads: Sequence[Mapping[str, Any]],
) -> ValidationCheck:
    candidate_rows: List[Dict[str, Any]] = []
    bad_rows: List[Dict[str, Any]] = []
    for payload in parity_root_payloads:
        for raw_row in list(payload.get("rows") or []):
            row = dict(raw_row)
            if str(row.get("claim_level", "") or "") != "exact_collapse_candidate":
                continue
            candidate_rows.append(row)
            if not bool(row.get("strict_collapse_pass", False)) or dict(
                row.get("config_diff_vs_official_fno") or {}
            ):
                bad_rows.append(
                    {
                        "scope_label": row.get("scope_label"),
                        "state": row.get("state"),
                        "strict_collapse_pass": row.get("strict_collapse_pass"),
                        "diff_fields": sorted(
                            dict(row.get("config_diff_vs_official_fno") or {}).keys()
                        ),
                    }
                )
    status = "warn" if not candidate_rows else ("fail" if bad_rows else "pass")
    return ValidationCheck(
        name="strict_collapse_readiness",
        status=status,
        severity="hard",
        summary=(
            "one-leaf exact-collapse candidates clear the strict production-FNO surface check"
            if status == "pass"
            else (
                "no exact-collapse candidate rows were present"
                if status == "warn"
                else "exact-collapse candidate rows still differ from the production-FNO comparison surface"
            )
        ),
        details={
            "n_candidate_rows": int(len(candidate_rows)),
            "n_bad_rows": int(len(bad_rows)),
            "sample_bad_rows": bad_rows[:5],
        },
    )


def _comparable_surface_drift_check(
    diagnostics_payload: Mapping[str, Any],
) -> ValidationCheck:
    runs = [
        dict(run)
        for run in list(diagnostics_payload.get("runs") or [])
        if str(run.get("comparison_mode", "")).strip().lower() == "comparable"
    ]
    grouped: Dict[tuple[str, str, int, int], List[Dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(
            (
                str(run.get("benchmark", "")),
                str(run.get("cell_id", "")),
                int(run.get("train_doc_count", 0)),
                int(run.get("fixed_leaf_tokens", 0)),
            ),
            [],
        ).append(run)
    bad_groups: List[Dict[str, Any]] = []
    for (benchmark, cell_id, train_doc_count, fixed_leaf_tokens), group in grouped.items():
        families = {
            str(run.get("baseline_family", "")).strip()
            for run in group
            if str(run.get("baseline_family", "")).strip()
        }
        if not (
            families & set(TREE_NEURAL_BASELINE_FAMILIES)
            and families & {"official_fno", "official_fno_sumlen"}
        ):
            continue
        drifted = [
            {
                "baseline_family": str(run.get("baseline_family", "")),
                "comparison_surface_diff": dict(
                    run.get("comparison_surface_diff") or {}
                ),
            }
            for run in group
            if dict(run.get("comparison_surface_diff") or {})
        ]
        if drifted:
            bad_groups.append(
                {
                    "benchmark": benchmark,
                    "cell_id": cell_id,
                    "train_doc_count": int(train_doc_count),
                    "fixed_leaf_tokens": int(fixed_leaf_tokens),
                    "families": sorted(families),
                    "drifted_runs": drifted,
                }
            )
    status = "pass" if not bad_groups else "fail"
    return ValidationCheck(
        name="comparable_surface_drift",
        status=status,
        severity="hard",
        summary=(
            "mixed-family comparable runs have zero shared-surface drift"
            if status == "pass"
            else "mixed-family comparable runs still drift on the shared comparison surface"
        ),
        details={
            "n_groups_checked": int(len(grouped)),
            "n_bad_groups": int(len(bad_groups)),
            "sample_bad_groups": bad_groups[:5],
        },
    )


def _compat_objective_config_from_run(run: Mapping[str, Any]) -> Optional[OPSCountConfig]:
    config_payload = dict(run.get("config") or {})
    dataclass_fields = {field.name for field in fields(OPSCountConfig)}
    if not config_payload:
        return None
    # Require critical objective-weight fields to be present rather than
    # silently filling them with defaults (which would produce incorrect
    # validation reports for runs with non-default weights).
    _REQUIRED_OBJECTIVE_KEYS = {
        "c1_relative_weight",
        "c2_relative_weight",
        "c3_relative_weight",
        "leaf_label_rate",
        "depth_discount_gamma",
    }
    # Also check the run-level fallback locations
    combined_payload = {**config_payload}
    for key in (
        "local_law_weight",
        "task_objective_weight",
        "c1_relative_weight",
        "c2_relative_weight",
        "c3_relative_weight",
        "schedule_consistency_weight",
        "law_package",
        "root_weight",
        "leaf_weight",
        "c2_weight",
        "c3_weight",
    ):
        if key not in combined_payload and key in run:
            combined_payload[key] = run[key]
    missing = _REQUIRED_OBJECTIVE_KEYS - set(combined_payload.keys())
    if missing:
        return None
    defaults = asdict(OPSCountConfig())
    for key in dataclass_fields:
        if key in combined_payload:
            defaults[key] = combined_payload[key]
    try:
        return OPSCountConfig(**{key: defaults[key] for key in dataclass_fields})
    except Exception:
        return None


def _canonical_theorem_terms(terms: Sequence[Mapping[str, Any]]) -> List[tuple[str, float, bool, str]]:
    out: List[tuple[str, float, bool, str]] = []
    for term in terms:
        out.append(
            (
                str(term.get("law_kind", term.get("name", ""))),
                float(term.get("weight", float("nan"))),
                bool(term.get("active", False)),
                str(term.get("evidence_status", "")),
            )
        )
    return out


def _canonical_proxy_terms(terms: Sequence[Mapping[str, Any]]) -> List[tuple[str, float, bool, str]]:
    out: List[tuple[str, float, bool, str]] = []
    for term in terms:
        out.append(
            (
                str(term.get("name", "")),
                float(term.get("weight", float("nan"))),
                bool(term.get("active", False)),
                str(term.get("evidence_status", "")),
            )
        )
    return out


def _objective_weight_alignment_check(payload: Mapping[str, Any]) -> ValidationCheck:
    bad_rows: List[Dict[str, Any]] = []
    checked_rows = 0
    for run in list(payload.get("runs") or []):
        family = str(run.get("baseline_family", ""))
        if family not in TREE_NEURAL_BASELINE_FAMILIES:
            if (
                str(run.get("parameterization", "")) != "inactive_for_family"
                or str(run.get("weighting_scheme", "")) != "inactive_for_family"
                or bool(run.get("objective_weights_active", False))
                or abs(float(run.get("proxy_schedule_consistency_weight", 0.0))) > 1e-12
            ):
                bad_rows.append(
                    {
                        "family": family,
                        "reason": "non_tree_inactive_objective_metadata_mismatch",
                        "parameterization": run.get("parameterization"),
                        "weighting_scheme": run.get("weighting_scheme"),
                        "objective_weights_active": run.get("objective_weights_active"),
                    }
                )
            continue
        if not _is_headline_comparison_semantics(
            str(run.get("comparison_semantics", ""))
        ):
            continue
        checked_rows += 1
        cfg = _compat_objective_config_from_run(run)
        if cfg is None:
            bad_rows.append(
                {
                    "family": family,
                    "reason": "missing_or_invalid_objective_builder_inputs",
                    "config": dict(run.get("config") or {}),
                }
            )
            continue
        expected = _build_objective_summary(cfg)
        if (
            str(run.get("parameterization", "")) != str(expected.get("parameterization", ""))
            or str(run.get("weighting_scheme", "")) != str(expected.get("weighting_scheme", ""))
            or not _close(
                run.get("optimization_root_weight"),
                expected.get("optimization_root_weight"),
            )
            or not _close(run.get("local_law_c1_weight"), expected.get("local_law_c1_weight"))
            or not _close(run.get("local_law_c2_weight"), expected.get("local_law_c2_weight"))
            or not _close(run.get("local_law_c3_weight"), expected.get("local_law_c3_weight"))
            or str(run.get("task_objective_weight_source", ""))
            != str(expected.get("task_objective_weight_source", ""))
            or not _close(
                run.get("proxy_schedule_consistency_weight"),
                expected.get("proxy_schedule_consistency_weight"),
            )
            or _canonical_theorem_terms(list(run.get("theorem_terms") or []))
            != _canonical_theorem_terms(list(expected.get("theorem_terms") or []))
            or _canonical_proxy_terms(list(run.get("proxy_terms") or []))
            != _canonical_proxy_terms(list(expected.get("proxy_terms") or []))
        ):
            bad_rows.append(
                {
                    "family": family,
                    "seed": run.get("seed"),
                    "reason": "resolved_objective_mismatch",
                    "expected": {
                        "parameterization": expected.get("parameterization"),
                        "weighting_scheme": expected.get("weighting_scheme"),
                        "optimization_root_weight": expected.get("optimization_root_weight"),
                        "local_law_c1_weight": expected.get("local_law_c1_weight"),
                        "local_law_c2_weight": expected.get("local_law_c2_weight"),
                        "local_law_c3_weight": expected.get("local_law_c3_weight"),
                        "task_objective_weight_source": expected.get("task_objective_weight_source"),
                        "proxy_schedule_consistency_weight": expected.get("proxy_schedule_consistency_weight"),
                    },
                    "actual": {
                        "parameterization": run.get("parameterization"),
                        "weighting_scheme": run.get("weighting_scheme"),
                        "optimization_root_weight": run.get("optimization_root_weight"),
                        "local_law_c1_weight": run.get("local_law_c1_weight"),
                        "local_law_c2_weight": run.get("local_law_c2_weight"),
                        "local_law_c3_weight": run.get("local_law_c3_weight"),
                        "task_objective_weight_source": run.get("task_objective_weight_source"),
                        "proxy_schedule_consistency_weight": run.get("proxy_schedule_consistency_weight"),
                    },
                }
            )
    status = "pass" if not bad_rows else "fail"
    if checked_rows == 0 and not bad_rows:
        status = "warn"
    return ValidationCheck(
        name="objective_weight_alignment",
        status=status,
        severity="hard",
        summary=(
            "full-doc objective weights and theorem/proxy term payloads match the shared builder"
            if status == "pass"
            else (
                "no current tree-neural rows were available for strict objective comparison"
                if status == "warn"
                else "full-doc emitted objective semantics diverged from the shared objective builder"
            )
        ),
        details={
            "rows_checked": int(checked_rows),
            "n_bad_rows": int(len(bad_rows)),
            "sample_bad_rows": bad_rows[:5],
        },
    )


def _theorem_proxy_labeling_check(payload: Mapping[str, Any]) -> ValidationCheck:
    bad_rows: List[Dict[str, Any]] = []
    for run in list(payload.get("runs") or []):
        theorem_terms = list(run.get("theorem_terms") or [])
        proxy_terms = list(run.get("proxy_terms") or [])
        theorem_names = {
            str(term.get("name", "")) for term in theorem_terms if isinstance(term, Mapping)
        }
        proxy_names = {
            str(term.get("name", "")) for term in proxy_terms if isinstance(term, Mapping)
        }
        if "schedule_consistency" in theorem_names:
            bad_rows.append({"family": run.get("baseline_family"), "reason": "schedule_term_in_theorem_terms"})
            continue
        if proxy_names and proxy_names != {"schedule_consistency"}:
            bad_rows.append({"family": run.get("baseline_family"), "reason": "unexpected_proxy_term_names", "proxy_names": sorted(proxy_names)})
            continue
        if any(
            str(term.get("evidence_status", "")).strip().upper() != "PROXY_ONLY"
            for term in proxy_terms
            if isinstance(term, Mapping)
        ):
            bad_rows.append({"family": run.get("baseline_family"), "reason": "proxy_term_not_labeled_proxy_only"})
            continue
        theorem_total = float(
            run.get(
                "theorem_local_law_total_weight",
                float(run.get("local_law_c1_weight", 0.0))
                + float(run.get("local_law_c2_weight", 0.0))
                + float(run.get("local_law_c3_weight", 0.0)),
            )
        )
        if not _close(
            theorem_total,
            float(run.get("local_law_c1_weight", 0.0))
            + float(run.get("local_law_c2_weight", 0.0))
            + float(run.get("local_law_c3_weight", 0.0)),
        ):
            bad_rows.append({"family": run.get("baseline_family"), "reason": "theorem_total_mismatch"})
            continue
    return ValidationCheck(
        name="theorem_proxy_labeling",
        status="fail" if bad_rows else "pass",
        severity="hard",
        summary=(
            "theorem-facing local-law totals exclude proxy schedule terms"
            if not bad_rows
            else "proxy schedule terms are being counted or labeled as theorem local laws"
        ),
        details={"n_bad_rows": int(len(bad_rows)), "sample_bad_rows": bad_rows[:5]},
    )


def _parity_metadata_check(payload: Mapping[str, Any]) -> ValidationCheck:
    parity_rows = [
        dict(row)
        for row in list(payload.get("runs") or [])
        if str(row.get("config_label", "")).strip() == "fair_fno_v1"
    ]
    bad_rows = [
        row
        for row in parity_rows
        if not str(row.get("tree_root_supervision_kind", "")).strip()
        or not int(row.get("tree_leaf_fno_width", 0))
        or not int(row.get("tree_leaf_fno_n_modes", 0))
        or not int(row.get("tree_leaf_fno_n_layers", 0))
        or not _close(row.get("tree_aux_doc_sequence_fraction"), 0.0)
    ]
    status = "warn" if not parity_rows else ("fail" if bad_rows else "pass")
    return ValidationCheck(
        name="parity_tree_metadata",
        status=status,
        severity="hard",
        summary=(
            "parity-tagged tree rows carry explicit root-supervision and leaf-FNO capacity metadata"
            if status == "pass"
            else (
                "no parity-tagged tree rows were present"
                if status == "warn"
                else "parity-tagged tree rows are missing required capacity/root-supervision metadata"
            )
        ),
        details={
            "n_rows": int(len(parity_rows)),
            "n_bad_rows": int(len(bad_rows)),
            "sample_bad_rows": bad_rows[:5],
        },
    )


def _budget_frontier_field_check(payload: Mapping[str, Any]) -> ValidationCheck:
    budget_rows = [
        dict(row)
        for row in list(payload.get("runs") or [])
        if int(row.get("budget_total_calls", 0)) > 0
        or float(row.get("budget_total_calls_per_doc", 0.0)) > 0.0
        or str(row.get("study_name", "")) == ORACLE_BUDGET_STUDY_NAME
    ]
    required_fields = (
        "budget_total_calls",
        "budget_total_calls_per_doc",
        "full_doc_budget_share",
        "doc_consumption_mode",
        "local_split_mode",
        "effective_full_doc_mass_total",
        "effective_full_doc_mass_per_doc",
        "budget_manifest",
    )
    bad_rows: List[Dict[str, Any]] = []
    for row in budget_rows:
        missing = [
            field_name
            for field_name in required_fields
            if field_name not in row
            or _is_missing_metadata_value(row.get(field_name))
        ]
        if missing:
            bad_rows.append(
                {
                    "family": row.get("baseline_family"),
                    "seed": row.get("seed"),
                    "reason": "missing_budget_fields",
                    "missing": missing,
                }
            )
            continue
        if (
            str(row.get("baseline_family", "")) in FULL_DOC_ONLY_BUDGET_FAMILIES
            and abs(float(row.get("full_doc_budget_share", 1.0)) - 1.0) > 1e-12
        ):
            bad_rows.append(
                {
                    "family": row.get("baseline_family"),
                    "seed": row.get("seed"),
                    "reason": "non_tree_budget_share_not_one",
                    "full_doc_budget_share": row.get("full_doc_budget_share"),
                }
            )
            continue
        if str(row.get("local_allocation_policy", "")) not in {"", "breadth_first"}:
            bad_rows.append(
                {
                    "family": row.get("baseline_family"),
                    "seed": row.get("seed"),
                    "reason": "unsupported_local_allocation_policy",
                    "local_allocation_policy": row.get("local_allocation_policy"),
                }
            )
        manifest = dict(row.get("budget_manifest") or {})
        sampling_scheme = str(
            row.get("sampling_scheme", manifest.get("sampling_scheme", "")) or ""
        )
        if sampling_scheme != BUDGETED_SAMPLING_SCHEME_RANDOM_WITHOUT_REPLACEMENT:
            bad_rows.append(
                {
                    "family": row.get("baseline_family"),
                    "seed": row.get("seed"),
                    "reason": "unsupported_sampling_scheme",
                    "sampling_scheme": sampling_scheme,
                }
            )
    status = "warn" if not budget_rows else ("fail" if bad_rows else "pass")
    return ValidationCheck(
        name="budget_frontier_fields",
        status=status,
        severity="hard",
        summary=(
            "budget-frontier rows expose the required budget/share semantics"
            if status == "pass"
            else (
                "no budget-frontier rows were present"
                if status == "warn"
                else "budget-frontier rows are missing required budget/share semantics"
            )
        ),
        details={
            "n_rows": int(len(budget_rows)),
            "n_bad_rows": int(len(bad_rows)),
            "sample_bad_rows": bad_rows[:5],
        },
    )


def _recompute_budget_accounting_for_run(run: Mapping[str, Any]) -> Dict[str, float]:
    manifest = dict(run.get("budget_manifest") or {})
    plans = list(manifest.get("doc_plans") or [])
    fixed_leaf_tokens = int(run.get("fixed_leaf_tokens", run.get("config", {}).get("fixed_leaf_tokens", 0)) or 0)
    document_calls_total = 0
    leaf_calls_total = 0
    internal_calls_total = 0
    document_mass_total = 0.0
    leaf_mass_total = 0.0
    internal_mass_total = 0.0
    budget_total_calls_used = 0
    touched_docs_total = 0
    for plan in plans:
        doc_tokens = int(plan.get("doc_tokens", 0))
        document_mode = str(plan.get("document_mode", "")).strip()
        leaf_indices = [int(value) for value in list(plan.get("leaf_indices") or [])]
        internal_indices = [int(value) for value in list(plan.get("internal_indices") or [])]
        leaf_spans, internal_spans = _doc_leaf_and_internal_spans(
            n_tokens=int(doc_tokens),
            leaf_tokens=max(1, int(fixed_leaf_tokens)),
        )
        leaf_mass = sum(
            float(max(0, int(leaf_spans[idx][1]) - int(leaf_spans[idx][0]))) / float(max(1, doc_tokens))
            for idx in leaf_indices
            if 0 <= idx < len(leaf_spans)
        )
        internal_mass = sum(
            float(max(0, int(internal_spans[idx][1]) - int(internal_spans[idx][0]))) / float(max(1, doc_tokens))
            for idx in internal_indices
            if 0 <= idx < len(internal_spans)
        )
        expected_raw_call_cost = int(bool(document_mode)) + int(len(leaf_indices)) + int(len(internal_indices))
        if int(plan.get("raw_call_cost", 0)) != int(expected_raw_call_cost):
            raise ValueError(
                f"raw_call_cost mismatch for doc_index={plan.get('doc_index')}: "
                f"expected {expected_raw_call_cost}, got {plan.get('raw_call_cost')}"
            )
        expected_mass = float(int(bool(document_mode)) + float(leaf_mass) + float(internal_mass))
        if abs(float(plan.get("effective_full_doc_mass", float("nan"))) - expected_mass) > 1e-9:
            raise ValueError(
                f"effective_full_doc_mass mismatch for doc_index={plan.get('doc_index')}: "
                f"expected {expected_mass}, got {plan.get('effective_full_doc_mass')}"
            )
        if expected_raw_call_cost > 0:
            touched_docs_total += 1
        budget_total_calls_used += expected_raw_call_cost
        document_calls_total += int(bool(document_mode))
        leaf_calls_total += int(len(leaf_indices))
        internal_calls_total += int(len(internal_indices))
        document_mass_total += float(int(bool(document_mode)))
        leaf_mass_total += float(leaf_mass)
        internal_mass_total += float(internal_mass)
    effective_full_doc_mass_total = float(
        document_mass_total + leaf_mass_total + internal_mass_total
    )
    return {
        "budget_total_calls_used": float(budget_total_calls_used),
        "full_doc_calls_total": float(document_calls_total),
        "local_calls_total": float(leaf_calls_total + internal_calls_total),
        "document_call_share": float(document_calls_total / max(1, budget_total_calls_used)),
        "leaf_call_share": float(leaf_calls_total / max(1, budget_total_calls_used)),
        "internal_call_share": float(internal_calls_total / max(1, budget_total_calls_used)),
        "effective_full_doc_mass_total": float(effective_full_doc_mass_total),
        "effective_full_doc_mass_per_doc": float(
            effective_full_doc_mass_total / max(1, len(plans))
        ),
        "document_mass_share": float(
            document_mass_total / max(1e-12, effective_full_doc_mass_total)
        ),
        "leaf_mass_share": float(
            leaf_mass_total / max(1e-12, effective_full_doc_mass_total)
        ),
        "internal_mass_share": float(
            internal_mass_total / max(1e-12, effective_full_doc_mass_total)
        ),
        "doc_touch_rate": float(touched_docs_total / max(1, len(plans))),
        "mean_labels_per_touched_doc": float(
            budget_total_calls_used / max(1, touched_docs_total)
        ),
        "document_calls_total_recomputed": float(document_calls_total),
        "leaf_calls_total_recomputed": float(leaf_calls_total),
        "internal_calls_total_recomputed": float(internal_calls_total),
        "touched_docs_total": float(touched_docs_total),
    }


def _budget_manifest_accounting_check(payload: Mapping[str, Any]) -> ValidationCheck:
    budget_rows = [
        dict(row)
        for row in list(payload.get("runs") or [])
        if int(row.get("budget_total_calls", 0)) > 0
        or float(row.get("budget_total_calls_per_doc", 0.0)) > 0.0
        or str(row.get("study_name", "")) == ORACLE_BUDGET_STUDY_NAME
    ]
    bad_rows: List[Dict[str, Any]] = []
    checked_rows = 0
    for row in budget_rows:
        manifest = dict(row.get("budget_manifest") or {})
        if not list(manifest.get("doc_plans") or []):
            bad_rows.append(
                {
                    "family": row.get("baseline_family"),
                    "seed": row.get("seed"),
                    "reason": "missing_doc_plans",
                }
            )
            continue
        checked_rows += 1
        try:
            recomputed = _recompute_budget_accounting_for_run(row)
        except Exception as exc:
            bad_rows.append(
                {
                    "family": row.get("baseline_family"),
                    "seed": row.get("seed"),
                    "reason": "budget_manifest_recompute_failed",
                    "error": str(exc),
                }
            )
            continue
        for field_name in (
            "budget_total_calls_used",
            "full_doc_calls_total",
            "local_calls_total",
            "document_call_share",
            "leaf_call_share",
            "internal_call_share",
            "effective_full_doc_mass_total",
            "effective_full_doc_mass_per_doc",
            "document_mass_share",
            "leaf_mass_share",
            "internal_mass_share",
            "doc_touch_rate",
            "mean_labels_per_touched_doc",
            "touched_docs_total",
        ):
            if not _close(row.get(field_name), recomputed.get(field_name), atol=1e-9):
                bad_rows.append(
                    {
                        "family": row.get("baseline_family"),
                        "seed": row.get("seed"),
                        "reason": "budget_field_mismatch",
                        "field": field_name,
                        "actual": row.get(field_name),
                        "expected": recomputed.get(field_name),
                    }
                )
                break
        requested_full_doc = int(
            max(
                0,
                min(
                    int(row.get("budget_total_calls", 0)),
                    round(
                        float(row.get("full_doc_budget_share", 1.0))
                        * int(row.get("budget_total_calls", 0))
                    ),
                ),
            )
        )
        if int(row.get("full_doc_calls_requested", requested_full_doc)) != requested_full_doc:
            bad_rows.append(
                {
                    "family": row.get("baseline_family"),
                    "seed": row.get("seed"),
                    "reason": "full_doc_calls_requested_mismatch",
                    "actual": row.get("full_doc_calls_requested"),
                    "expected": requested_full_doc,
                }
            )
            continue
        if str(row.get("local_allocation_policy", "")) not in {"", "breadth_first"}:
            bad_rows.append(
                {
                    "family": row.get("baseline_family"),
                    "seed": row.get("seed"),
                    "reason": "allocation_policy_not_breadth_first",
                    "local_allocation_policy": row.get("local_allocation_policy"),
                }
            )
            continue
        sampling_scheme = str(
            row.get("sampling_scheme", manifest.get("sampling_scheme", "")) or ""
        )
        if sampling_scheme != BUDGETED_SAMPLING_SCHEME_RANDOM_WITHOUT_REPLACEMENT:
            bad_rows.append(
                {
                    "family": row.get("baseline_family"),
                    "seed": row.get("seed"),
                    "reason": "sampling_scheme_not_seeded_random_without_replacement",
                    "sampling_scheme": sampling_scheme,
                }
            )
            continue
    return ValidationCheck(
        name="budget_manifest_accounting",
        status=("warn" if not budget_rows else ("fail" if bad_rows else "pass")),
        severity="hard",
        summary=(
            "budget manifests and emitted aggregate fields are internally consistent"
            if budget_rows and not bad_rows
            else (
                "no budget-manifest rows were present"
                if not budget_rows
                else "budget manifest accounting diverged from the emitted budget-share summaries"
            )
        ),
        details={
            "rows_checked": int(checked_rows),
            "n_bad_rows": int(len(bad_rows)),
            "sample_bad_rows": bad_rows[:5],
        },
    )


def _objective_surface_distinction_check(payload: Mapping[str, Any]) -> ValidationCheck:
    note_exists = MARKOV_ALIGNMENT_AUDIT_NOTE_PATH.exists()
    note_text = (
        MARKOV_ALIGNMENT_AUDIT_NOTE_PATH.read_text(encoding="utf-8")
        if note_exists
        else ""
    )
    missing_phrases = [
        phrase for phrase in required_audit_note_phrases() if phrase not in note_text
    ]
    bad_rows = [
        dict(row)
        for row in list(payload.get("runs") or [])
        if str(row.get("objective_surface_name", MARKOV_FULL_DOC_OBJECTIVE_SURFACE))
        == TREEPO_REGULARIZED_OBJECTIVE_SURFACE
        or TREEPO_REGULARIZED_OBJECTIVE_SURFACE
        not in list(row.get("objective_surface_distinct_from") or [TREEPO_REGULARIZED_OBJECTIVE_SURFACE])
    ]
    failed = bool(missing_phrases or bad_rows)
    return ValidationCheck(
        name="objective_surface_distinction",
        status="fail" if failed else "pass",
        severity="hard",
        summary=(
            "the Markov full-doc objective surface is explicitly separated from the TreePO regularized objective"
            if not failed
            else "the audit note or emitted metadata no longer clearly separates the Markov full-doc objective from the TreePO regularized objective"
        ),
        details={
            "audit_note_path": str(MARKOV_ALIGNMENT_AUDIT_NOTE_PATH),
            "missing_phrases": missing_phrases,
            "n_bad_rows": int(len(bad_rows)),
            "sample_bad_rows": bad_rows[:5],
        },
    )


def _full_tree_ipw_semantics_check(payload: Optional[Mapping[str, Any]]) -> ValidationCheck:
    if not payload:
        return ValidationCheck(
            name="full_tree_ipw_semantics",
            status="warn",
            severity="hard",
            summary="no full-tree IPW payload was provided",
            details={},
        )
    semantics = dict(payload.get("semantics") or {})
    expected = {
        "estimand_name": "realized_full_tree_node_mean_loss",
        "population_kind": "realized_tree_nodes",
        "sampling_design": "bernoulli_realized_node_sampling",
        "propensity_field": "unit_propensity",
        "document_channel": "always_observed_document_top_loss",
        "node_channel": "sampled_realized_tree_nodes",
        "ci_semantics": "point_estimation_only",
    }
    mismatches = {
        key: {"expected": value, "actual": semantics.get(key)}
        for key, value in expected.items()
        if semantics.get(key) != value
    }
    estimators = list(semantics.get("estimator_families") or [])
    if estimators != ["naive", "ht", "hajek"]:
        mismatches["estimator_families"] = {
            "expected": ["naive", "ht", "hajek"],
            "actual": estimators,
        }
    return ValidationCheck(
        name="full_tree_ipw_semantics",
        status="fail" if mismatches else "pass",
        severity="hard",
        summary=(
            "full-tree IPW payload exposes the explicit realized-node estimand semantics"
            if not mismatches
            else "full-tree IPW payload is missing required estimand/sampling semantics"
        ),
        details={"mismatches": mismatches},
    )


def _full_tree_ipw_endpoint_check(payload: Optional[Mapping[str, Any]]) -> ValidationCheck:
    if not payload:
        return ValidationCheck(
            name="full_tree_ipw_endpoints",
            status="warn",
            severity="hard",
            summary="no full-tree IPW payload was provided",
            details={},
        )
    rows = grid_rows_from_payload(payload)
    full_tree_rows = [
        row
        for row in rows
        if abs(float(row.get("p_internal", float("nan"))) - 1.0) <= 1e-12
        and abs(float(row.get("p_leaf", float("nan"))) - 1.0) <= 1e-12
    ]
    doc_only_rows = [
        row
        for row in rows
        if abs(float(row.get("p_internal", float("nan"))) - 0.0) <= 1e-12
        and abs(float(row.get("p_leaf", float("nan"))) - 0.0) <= 1e-12
    ]
    bad_rows: List[Dict[str, Any]] = []
    for row in full_tree_rows:
        if (
            not _is_finite(row.get("test_sampled_node_ht_abs_error"))
            or not _is_finite(row.get("test_sampled_node_hajek_abs_error"))
            or float(row.get("test_sampled_node_ht_abs_error", float("inf"))) > 1e-6
            or float(row.get("test_sampled_node_hajek_abs_error", float("inf"))) > 1e-6
        ):
            bad_rows.append(
                {
                    "reason": "full_tree_endpoint_recovery_failed",
                    "row": row,
                }
            )
    for row in doc_only_rows:
        if (
            abs(float(row.get("test_sample_fraction", float("nan")))) > 1e-12
            or int(row.get("test_sampled_nodes", 0)) != 0
        ):
            bad_rows.append(
                {
                    "reason": "doc_only_endpoint_is_not_document_only",
                    "row": row,
                }
            )
    status = "pass" if (full_tree_rows or doc_only_rows) and not bad_rows else ("warn" if not (full_tree_rows or doc_only_rows) else "fail")
    return ValidationCheck(
        name="full_tree_ipw_endpoints",
        status=status,
        severity="hard",
        summary=(
            "full-tree IPW endpoints behave as the estimand story claims"
            if status == "pass"
            else (
                "no endpoint rows were present in the full-tree IPW payload"
                if status == "warn"
                else "full-tree IPW endpoints diverged from the claimed estimand story"
            )
        ),
        details={
            "n_full_tree_rows": int(len(full_tree_rows)),
            "n_doc_only_rows": int(len(doc_only_rows)),
            "n_bad_rows": int(len(bad_rows)),
            "sample_bad_rows": bad_rows[:5],
        },
    )


def _full_tree_ipw_sampling_check(payload: Optional[Mapping[str, Any]]) -> ValidationCheck:
    if not payload:
        return ValidationCheck(
            name="full_tree_ipw_sampling_diagnostics",
            status="warn",
            severity="hard",
            summary="no full-tree IPW payload was provided",
            details={},
        )
    rows = grid_rows_from_payload(payload)
    sampled_rows = [
        row
        for row in rows
        if float(row.get("test_sample_fraction", float("nan"))) > 0.0
    ]
    bad_rows = [
        row
        for row in sampled_rows
        if not _is_finite(row.get("test_effective_sample_size"))
        or not _is_finite(row.get("test_max_weight"))
        or float(row.get("test_effective_sample_size", float("nan"))) <= 0.0
        or float(row.get("test_effective_sample_size", float("nan")))
        > float(row.get("test_sampled_nodes", float("inf"))) + 1e-9
        or float(row.get("test_max_weight", float("nan"))) <= 0.0
    ]
    status = "pass" if sampled_rows and not bad_rows else ("warn" if not sampled_rows else "fail")
    return ValidationCheck(
        name="full_tree_ipw_sampling_diagnostics",
        status=status,
        severity="hard",
        summary=(
            "full-tree IPW propensity, ESS, and max-weight diagnostics are finite and internally consistent"
            if status == "pass"
            else (
                "no sampled full-tree IPW rows were present"
                if status == "warn"
                else "full-tree IPW propensity, ESS, or max-weight diagnostics are inconsistent"
            )
        ),
        details={
            "n_rows": int(len(sampled_rows)),
            "n_bad_rows": int(len(bad_rows)),
            "sample_bad_rows": bad_rows[:5],
        },
    )


def _underperformance_warning(payload: Mapping[str, Any]) -> ValidationCheck:
    aggregate_rows = list(payload.get("aggregate_rows") or [])
    control_rows = [
        row
        for row in aggregate_rows
        if str(row.get("baseline_family", "")) in {"tree_doc_ridge", "raw_token_ngram_ridge", "palette_block_exact"}
    ]
    approximate_rows = [
        row
        for row in aggregate_rows
        if str(row.get("baseline_family", "")).startswith("tree_neural")
        or str(row.get("baseline_family", "")) in {"official_fno", "official_fno_sumlen"}
    ]
    if not control_rows or not approximate_rows:
        return ValidationCheck(
            name="approximate_lane_underperformance",
            status="not_applicable",
            severity="warning",
            summary="insufficient control or approximate rows to score underperformance",
            details={},
        )
    best_control = min(control_rows, key=lambda row: float(row.get("test_root_mae_mean", float("inf"))))
    best_approx = min(approximate_rows, key=lambda row: float(row.get("test_root_mae_mean", float("inf"))))
    gap = float(best_approx.get("test_root_mae_mean", float("nan"))) - float(
        best_control.get("test_root_mae_mean", float("nan"))
    )
    return ValidationCheck(
        name="approximate_lane_underperformance",
        status="warn" if gap > 0.01 else "pass",
        severity="warning",
        summary=(
            "best approximate/proxy lane still trails the exact or closed-form control"
            if gap > 0.01
            else "best approximate/proxy lane is within the control envelope"
        ),
        details={
            "best_control_family": str(best_control.get("baseline_family", "")),
            "best_control_root_mae_mean": float(best_control.get("test_root_mae_mean", float("nan"))),
            "best_approx_family": str(best_approx.get("baseline_family", "")),
            "best_approx_root_mae_mean": float(best_approx.get("test_root_mae_mean", float("nan"))),
            "gap_to_best_control": float(gap),
        },
    )


def _partial_coverage_warning(payload: Mapping[str, Any]) -> ValidationCheck:
    expected = max((int(len(list(payload.get("seeds") or []))), 1))
    cell_counts: Dict[
        tuple[str, str, int, int, str, str, str, str, str, str, int, int, int, float],
        int,
    ] = {}
    for row in list(payload.get("runs") or []):
        key = (
            str(row.get("cell_id", row.get("benchmark", ""))),
            str(row.get("baseline_family", "")),
            int(row.get("train_doc_count", 0)),
            int(row.get("fixed_leaf_tokens", 0)),
            str(row.get("config_label", "")),
            str(row.get("tree_root_supervision_kind", "")),
            str(row.get("tuning_stage", "")),
            str(row.get("study_name", "")),
            str(row.get("study_axis", "")),
            str(row.get("axis_value", "")),
            int(row.get("tree_leaf_fno_width", 0)),
            int(row.get("tree_leaf_fno_n_modes", 0)),
            int(row.get("tree_leaf_fno_n_layers", 0)),
            float(row.get("tree_aux_doc_sequence_fraction", 0.0)),
        )
        cell_counts[key] = int(cell_counts.get(key, 0)) + 1
    incomplete = {
        "|".join(
            [
                key[0],
                key[1],
                str(key[2]),
                str(key[3]),
                key[4],
                key[5],
                key[6],
                key[7],
                key[8],
                key[9],
                str(key[10]),
                str(key[11]),
                str(key[12]),
                f"{float(key[13]):.6g}",
            ]
        ): count
        for key, count in cell_counts.items()
        if count < expected
    }
    return ValidationCheck(
        name="partial_coverage",
        status="warn" if incomplete else "pass",
        severity="warning",
        summary=(
            "some family/train-count cells are still missing seeds"
            if incomplete
            else "all family/train-count cells have the expected seed coverage"
        ),
        details={
            "expected_seeds_per_cell": int(expected),
            "incomplete_cells": incomplete,
        },
    )


def _full_tree_ipw_point_estimate_warning(payload: Optional[Mapping[str, Any]]) -> ValidationCheck:
    if not payload:
        return ValidationCheck(
            name="full_tree_ipw_honesty_scope",
            status="not_applicable",
            severity="warning",
            summary="no full-tree IPW payload was provided",
            details={},
        )
    semantics = dict(payload.get("semantics") or {})
    ci_semantics = str(semantics.get("ci_semantics", ""))
    return ValidationCheck(
        name="full_tree_ipw_honesty_scope",
        status="warn" if ci_semantics == "point_estimation_only" else "pass",
        severity="warning",
        summary=(
            "full-tree IPW grid is point-estimation only and does not claim honest CI semantics"
            if ci_semantics == "point_estimation_only"
            else "full-tree IPW payload includes a non-point-estimation CI scope"
        ),
        details={"ci_semantics": ci_semantics},
    )


def _full_doc_family_semantics_summary(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    families: Dict[str, Dict[str, Any]] = {}
    for row in list(payload.get("aggregate_rows") or []):
        family = str(row.get("baseline_family", ""))
        if family in families:
            continue
        if family in TREE_NEURAL_BASELINE_FAMILIES:
            optimization_problem = "root_count objective plus theorem-facing C1/C2/C3 local-law terms"
            theorem_side = "C1/L1, C2/L3, C3/L2"
            proxy_side = "schedule_consistency / schedule spread"
        elif family in {"official_fno", "official_fno_sumlen"}:
            optimization_problem = "full-document count classification via official neuraloperator baseline"
            theorem_side = "none"
            proxy_side = "all reported local-law diagnostics are proxy-only comparisons"
        else:
            optimization_problem = "closed-form or generic control fit without active theorem local-law objective"
            theorem_side = "none"
            proxy_side = "all local-law diagnostics are proxy-only comparisons"
        families[family] = {
            "baseline_family": family,
            "optimization_problem": optimization_problem,
            "theorem_side_diagnostics": theorem_side,
            "proxy_only_diagnostics": proxy_side,
            "operator_evidence_status": str(row.get("operator_evidence_status", "")),
            "weighting_scheme": str(row.get("weighting_scheme", "")),
            "parameterization": str(row.get("parameterization", "")),
        }
    return [families[key] for key in sorted(families)]


def build_markov_alignment_audit_report(
    *,
    diagnostics_root: Path | None = None,
    full_tree_ipw_root: Path | None = None,
    ladder_json: Path | None = None,
    bundle_manifest_path: Path | None = None,
    family_grids_summary_json: Path | None = None,
    parity_grid_roots: Sequence[Path] = (),
    canonical_train_ladder: Sequence[int] = (1024, 4096, 10240),
    run_lean_build: bool = False,
) -> MarkovAlignmentAuditReport:
    diagnostics_payload: Dict[str, Any] | None = None
    diagnostics_summary_json: Path | None = None
    if diagnostics_root is not None:
        diagnostics_path = diagnostics_root.resolve()
        diagnostics_payload = load_markov_full_doc_anchor_diagnostics_from_output_dir(
            diagnostics_path
        )
        diagnostics_summary_json = diagnostics_path / "summary.json"
    else:
        diagnostics_path = None

    ladder_payload = (
        _load_json(ladder_json.resolve())
        if ladder_json is not None and ladder_json.exists()
        else None
    )
    full_tree_payload, full_tree_summary_json = _load_full_tree_ipw_payload(
        full_tree_ipw_root
    )
    family_grids_summary = _load_family_grids_summary_payload(family_grids_summary_json)
    parity_root_payloads = _load_parity_root_payloads(parity_grid_roots)
    authoritative_canonical_ladder_claimed = bool(
        str((family_grids_summary or {}).get("evidence_status", "")).strip()
        == "authoritative"
    )

    explicit_paths: List[Path] = []
    if diagnostics_summary_json is not None and diagnostics_summary_json.exists():
        explicit_paths.append(diagnostics_summary_json)
    if ladder_payload is not None and ladder_json is not None:
        explicit_paths.append(ladder_json.resolve())
    if full_tree_summary_json is not None and full_tree_summary_json.exists():
        explicit_paths.append(full_tree_summary_json)

    expectation_report = _expectation_report_for_extra_paths(paths=explicit_paths)
    theory_alignment_report = build_simulation_theory_alignment_report(
        formal_root=(
            diagnostics_path
            if diagnostics_path is not None
            else (
                full_tree_ipw_root.resolve()
                if full_tree_ipw_root is not None
                else None
            )
        ),
        expectation_report=expectation_report,
        bundle_manifest_path=(
            bundle_manifest_path.resolve()
            if bundle_manifest_path is not None and bundle_manifest_path.exists()
            else None
        ),
    )

    checks: List[ValidationCheck] = []
    if diagnostics_payload is not None:
        checks.extend(
            [
                _duplicate_aggregate_check(diagnostics_payload),
                _score_contract_check(diagnostics_payload),
                _objective_weight_alignment_check(diagnostics_payload),
                _theorem_proxy_labeling_check(diagnostics_payload),
                _official_fno_provenance_check(diagnostics_payload),
                _current_semantics_check(diagnostics_payload),
                _law_contract_gap_warning(diagnostics_payload),
                _law_contract_limitation_warning(diagnostics_payload),
                _parity_metadata_check(diagnostics_payload),
                _legacy_current_mixing_check(diagnostics_payload),
                _comparable_surface_drift_check(diagnostics_payload),
                _rung_nestedness_check(
                    diagnostics_payload,
                    canonical_train_ladder=canonical_train_ladder,
                    authoritative_claimed=authoritative_canonical_ladder_claimed,
                ),
                _bundle_consistency_check(
                    diagnostics_payload,
                    canonical_train_ladder=canonical_train_ladder,
                ),
                _budget_frontier_field_check(diagnostics_payload),
                _budget_manifest_accounting_check(diagnostics_payload),
                _ladder_pairing_check(ladder_payload),
                _objective_surface_distinction_check(diagnostics_payload),
                _underperformance_warning(diagnostics_payload),
                _partial_coverage_warning(diagnostics_payload),
            ]
        )
    else:
        checks.append(
            ValidationCheck(
                name="full_doc_surface_presence",
                status="warn",
                severity="warning",
                summary="no full-doc diagnostics root was provided",
                details={},
            )
        )

    checks.extend(
        [
            _family_grids_v3_provenance_check(family_grids_summary),
            _family_grids_quarantine_enforcement_check(family_grids_summary),
            _canonical_fno_geometry_check(
                family_grids_summary,
                parity_root_payloads,
            ),
            _package_vs_parity_separation_check(family_grids_summary),
            _stopped_root_exclusion_check(
                parity_root_payloads,
                family_grids_summary=family_grids_summary,
            ),
            _strict_collapse_readiness_check(parity_root_payloads),
            _full_tree_ipw_semantics_check(full_tree_payload),
            _full_tree_ipw_endpoint_check(full_tree_payload),
            _full_tree_ipw_sampling_check(full_tree_payload),
            _full_tree_ipw_point_estimate_warning(full_tree_payload),
        ]
    )

    lean_build = _run_lean_build() if run_lean_build else {"requested": False}
    if run_lean_build and not bool(lean_build.get("passed", False)):
        checks.append(
            ValidationCheck(
                name="lean_build",
                status="fail",
                severity="hard",
                summary="Lean preflight failed",
                details=dict(lean_build),
            )
        )

    n_fail = int(sum(1 for check in checks if check.status == "fail"))
    n_warn = int(sum(1 for check in checks if check.status == "warn"))
    alignment_spec_payload = markov_alignment_spec()
    surface_coverage = _surface_coverage_summary(
        diagnostics_payload=diagnostics_payload,
        ladder_payload=ladder_payload,
        full_tree_payload=full_tree_payload,
    )
    summary = {
        "status": "fail" if n_fail > 0 else ("warn" if n_warn > 0 else "pass"),
        "n_checks": int(len(checks)),
        "n_fail": n_fail,
        "n_warn": n_warn,
        "n_pass": int(sum(1 for check in checks if check.status == "pass")),
        "hard_failures": [
            check.name for check in checks if check.severity == "hard" and check.status == "fail"
        ],
        "warning_checks": [check.name for check in checks if check.status == "warn"],
        "surface_coverage": surface_coverage,
        "expectation_summary": dict(expectation_report.summary),
        "theory_alignment_summary": dict(theory_alignment_report.summary),
        "canonical_train_ladder": [int(value) for value in canonical_train_ladder],
    }
    check_by_name = {check.name: check for check in checks}
    rung_check = check_by_name.get("rung_nestedness")
    bundle_check = check_by_name.get("bundle_consistency")
    if rung_check is not None:
        summary["train_prefix_counts"] = list(
            rung_check.details.get("train_prefix_counts") or []
        )
        summary["train_prefix_signatures"] = dict(
            rung_check.details.get("train_prefix_signatures") or {}
        )
    if bundle_check is not None:
        summary["reference_bundle_source"] = dict(
            bundle_check.details.get("reference_bundle_source") or {}
        )
        summary["train_corpus_signature"] = dict(
            bundle_check.details.get("train_corpus_signature") or {}
        )
        summary["val_corpus_signature"] = dict(
            bundle_check.details.get("val_corpus_signature") or {}
        )
        summary["test_corpus_signature"] = dict(
            bundle_check.details.get("test_corpus_signature") or {}
        )
    if family_grids_summary is not None:
        summary["evidence_status"] = str(
            family_grids_summary.get("evidence_status")
            or (
                "authoritative"
                if bool(family_grids_summary.get("historical_grid_present", False))
                else "partial"
            )
        )
        summary["quarantined_row_count"] = int(
            family_grids_summary.get("quarantined_row_count", 0) or 0
        )
        summary["quarantined_sources"] = list(
            family_grids_summary.get("quarantined_sources") or []
        )
        summary["contract_gate_status"] = str(
            family_grids_summary.get("contract_gate_status", "") or ""
        )
    elif parity_root_payloads:
        statuses = {
            str(payload.get("evidence_status", "") or "")
            for payload in parity_root_payloads
            if str(payload.get("evidence_status", "") or "").strip()
        }
        summary["evidence_status"] = (
            sorted(statuses)[0] if len(statuses) == 1 else "partial"
        )
    elif diagnostics_path is not None and "exploratory" in str(diagnostics_path).lower():
        summary["evidence_status"] = "exploratory"
    else:
        summary["evidence_status"] = "authoritative"
    if diagnostics_payload is not None:
        summary["full_doc_family_semantics"] = _full_doc_family_semantics_summary(
            diagnostics_payload
        )
    return MarkovAlignmentAuditReport(
        diagnostics_root=str(diagnostics_path) if diagnostics_path is not None else None,
        full_tree_ipw_root=(
            str(full_tree_ipw_root.resolve()) if full_tree_ipw_root is not None else None
        ),
        ladder_json=str(ladder_json.resolve()) if ladder_json is not None and ladder_json.exists() else None,
        bundle_manifest=(
            str(bundle_manifest_path.resolve())
            if bundle_manifest_path is not None and bundle_manifest_path.exists()
            else None
        ),
        alignment_spec=alignment_spec_payload,
        surface_coverage=surface_coverage,
        lean_build=lean_build,
        expectation_report=expectation_report.to_dict(),
        theory_alignment_report=theory_alignment_report.to_dict(),
        checks=checks,
        summary=summary,
    )


def render_markov_alignment_audit_markdown(report: MarkovAlignmentAuditReport) -> str:
    lines = [
        "# Markov Alignment Audit",
        "",
        f"- diagnostics root: `{report.diagnostics_root or 'not_provided'}`",
        f"- full-tree IPW root: `{report.full_tree_ipw_root or 'not_provided'}`",
        f"- ladder json: `{report.ladder_json or 'not_provided'}`",
        f"- bundle manifest: `{report.bundle_manifest or 'not_provided'}`",
        f"- overall status: `{str(report.summary.get('status', 'unknown'))}`",
        f"- evidence status: `{str(report.summary.get('evidence_status', 'unknown'))}`",
        f"- canonical train ladder: `{list(report.summary.get('canonical_train_ladder') or [])}`",
        "",
        "## Surface Coverage",
        "",
    ]
    for key, value in report.surface_coverage.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Alignment Spec",
            "",
            f"- audit note: `{report.alignment_spec.get('audit_note_path', '')}`",
            "- paper-to-Lean local-law mapping:",
            f"  `C1 -> {report.alignment_spec.get('paper_to_lean_local_law_mapping', {}).get('C1', '')}`, "
            f"`C2 -> {report.alignment_spec.get('paper_to_lean_local_law_mapping', {}).get('C2', '')}`, "
            f"`C3 -> {report.alignment_spec.get('paper_to_lean_local_law_mapping', {}).get('C3', '')}`",
            "",
            "## Checks",
            "",
            "| Check | Severity | Status | Summary |",
            "| --- | --- | --- | --- |",
        ]
    )
    for check in report.checks:
        lines.append(
            f"| {check.name} | `{check.severity}` | `{check.status}` | {check.summary} |"
        )
    family_semantics = list(report.summary.get("full_doc_family_semantics") or [])
    if family_semantics:
        lines.extend(
            [
                "",
                "## Full-Doc Family Semantics",
                "",
                "| Family | Optimization Problem | Theorem-Side Diagnostics | Proxy-Only Diagnostics |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in family_semantics:
            lines.append(
                "| "
                f"{str(row.get('baseline_family', ''))} | "
                f"{str(row.get('optimization_problem', ''))} | "
                f"{str(row.get('theorem_side_diagnostics', ''))} | "
                f"{str(row.get('proxy_only_diagnostics', ''))} |"
            )
    lines.extend(
        [
            "",
            "## Expectation Summary",
            "",
            f"- pass: `{report.expectation_report.get('summary', {}).get('n_pass', 0)}`",
            f"- warn: `{report.expectation_report.get('summary', {}).get('n_warn', 0)}`",
            f"- fail: `{report.expectation_report.get('summary', {}).get('n_fail', 0)}`",
            "",
            "## Theory Alignment Summary",
            "",
            f"- aligned families: `{report.theory_alignment_report.get('summary', {}).get('aligned_families', 0)}`",
            f"- provisionally aligned families: `{report.theory_alignment_report.get('summary', {}).get('provisionally_aligned_families', 0)}`",
            f"- misaligned families: `{report.theory_alignment_report.get('summary', {}).get('misaligned_families', 0)}`",
            "",
            "## Lean Build",
            "",
            f"- requested: `{bool(report.lean_build.get('requested', False))}`",
            f"- passed: `{bool(report.lean_build.get('passed', False))}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def write_markov_alignment_audit_report(
    report: MarkovAlignmentAuditReport,
    *,
    output_json: Path,
    output_markdown: Path,
) -> Dict[str, str]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    output_markdown.write_text(
        render_markov_alignment_audit_markdown(report),
        encoding="utf-8",
    )
    return {
        "output_json": str(output_json.resolve()),
        "output_markdown": str(output_markdown.resolve()),
    }


__all__ = [
    "LEAN_BUILD_TARGET",
    "MarkovAlignmentAuditReport",
    "ValidationCheck",
    "_duplicate_aggregate_check",
    "_legacy_current_mixing_check",
    "_score_contract_check",
    "build_markov_alignment_audit_report",
    "render_markov_alignment_audit_markdown",
    "write_markov_alignment_audit_report",
]
