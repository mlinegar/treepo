from __future__ import annotations

from pathlib import Path
from typing import Dict

from treepo._research.ctreepo.sim.core.markov_alignment_validation import (
    LEAN_BUILD_TARGET,
    MarkovAlignmentAuditReport,
    ValidationCheck,
    _duplicate_aggregate_check,
    _legacy_current_mixing_check,
    _score_contract_check,
    build_markov_alignment_audit_report,
    render_markov_alignment_audit_markdown,
    write_markov_alignment_audit_report,
)


MarkovTreeFNOValidationReport = MarkovAlignmentAuditReport


def build_markov_tree_fno_validation_report(
    *,
    diagnostics_root: Path,
    ladder_json: Path | None = None,
    bundle_manifest_path: Path | None = None,
    family_grids_summary_json: Path | None = None,
    parity_grid_roots: tuple[Path, ...] = (),
    canonical_train_ladder: tuple[int, ...] = (1024, 4096, 10240),
    run_lean_build: bool = False,
) -> MarkovTreeFNOValidationReport:
    return build_markov_alignment_audit_report(
        diagnostics_root=diagnostics_root,
        full_tree_ipw_root=None,
        ladder_json=ladder_json,
        bundle_manifest_path=bundle_manifest_path,
        family_grids_summary_json=family_grids_summary_json,
        parity_grid_roots=parity_grid_roots,
        canonical_train_ladder=canonical_train_ladder,
        run_lean_build=run_lean_build,
    )


def render_markov_tree_fno_validation_markdown(
    report: MarkovTreeFNOValidationReport,
) -> str:
    return render_markov_alignment_audit_markdown(report)


def write_markov_tree_fno_validation_report(
    report: MarkovTreeFNOValidationReport,
    *,
    output_json: Path,
    output_markdown: Path,
) -> Dict[str, str]:
    return write_markov_alignment_audit_report(
        report,
        output_json=output_json,
        output_markdown=output_markdown,
    )


__all__ = [
    "LEAN_BUILD_TARGET",
    "MarkovTreeFNOValidationReport",
    "ValidationCheck",
    "_duplicate_aggregate_check",
    "_legacy_current_mixing_check",
    "_score_contract_check",
    "build_markov_tree_fno_validation_report",
    "render_markov_tree_fno_validation_markdown",
    "write_markov_tree_fno_validation_report",
]
