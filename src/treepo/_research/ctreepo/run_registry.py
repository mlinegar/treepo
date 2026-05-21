"""General C-TreePO run target registry.

The registry is deliberately broader than the paper build.  Paper-facing runs
are ordinary targets with stricter audit requirements, while exploratory and
legacy targets share the same RunManifest envelope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from treepo._research.ctreepo.contracts import RUN_MANIFEST_SCHEMA_VERSION, TREE_BUNDLE_SCHEMA_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET_STATUSES = {
    "canonical",
    "thin_wrapper",
    "legacy_compat",
    "not_publication_safe",
    "exploratory",
}


@dataclass(frozen=True)
class RunTargetRecord:
    target: str
    path: str
    domain: str
    role: str
    backend: str
    status: str
    expected_input_contract: str = ""
    output_contract: str = RUN_MANIFEST_SCHEMA_VERSION
    audit_policy: str = ""
    publication_ready: bool = False
    suites: Sequence[str] = field(default_factory=tuple)
    command: Sequence[str] = field(default_factory=tuple)
    notes: str = ""

    @property
    def publication_facing(self) -> bool:
        return bool(self.publication_ready)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _python_command(*parts: str) -> tuple[str, ...]:
    return ("{python}", *parts)


_BASE_TARGETS: tuple[RunTargetRecord, ...] = (
    RunTargetRecord(
        target="manifesto.coverage_split",
        path="scripts/build_manifesto_coverage_split.py",
        domain="manifesto_rile",
        role="coverage_split_builder",
        backend="data_prep",
        status="canonical",
        expected_input_contract="manifesto_dimension_results",
        audit_policy="run_manifest_required",
        publication_ready=True,
        suites=("manifesto", "publication"),
        command=_python_command(
            "scripts/build_manifesto_coverage_split.py",
            "--output-dir",
            "{output_root}",
        ),
        notes="All-six Manifesto coverage split with soft inverse-sqrt-length training sampling.",
    ),
    RunTargetRecord(
        target="manifesto.full_doc_gemma4_benchmark",
        path="scripts/run_manifesto_full_doc_gemma4_benchmark.py",
        domain="manifesto_rile",
        role="full_doc_direct_scorer",
        backend="vllm",
        status="canonical",
        expected_input_contract="manifesto_coverage_split",
        audit_policy="experiment_sidecars_and_run_manifest_required",
        publication_ready=True,
        suites=("manifesto", "publication"),
        command=_python_command(
            "scripts/run_manifesto_full_doc_gemma4_benchmark.py",
            "--split-dir",
            "{split_dir}",
            "--output-dir",
            "{output_root}",
        ),
        notes="{split_dir} must point at a manifesto.coverage_split output directory.",
    ),
    RunTargetRecord(
        target="manifesto.full_doc_dspy_global_f",
        path="scripts/run_manifesto_full_doc_dspy_global_f.py",
        domain="manifesto_rile",
        role="full_doc_dspy_global_f",
        backend="dspy",
        status="canonical",
        expected_input_contract="manifesto_coverage_split",
        audit_policy="experiment_sidecars_and_run_manifest_required",
        publication_ready=True,
        suites=("manifesto", "publication"),
        command=_python_command(
            "scripts/run_manifesto_full_doc_dspy_global_f.py",
            "--split-dir",
            "{split_dir}",
            "--output-dir",
            "{output_root}",
        ),
        notes="Raw one-leaf/document global f trained with DSPy over all requested dimensions.",
    ),
    RunTargetRecord(
        target="manifesto.teacher_leaf_grid",
        path="scripts/run_manifesto_teacher_fg_leaf_grid.py",
        domain="manifesto_rile",
        role="teacher_tree_bundle",
        backend="dspy",
        status="canonical",
        expected_input_contract=TREE_BUNDLE_SCHEMA_VERSION,
        audit_policy="fail_hard_tree_bundle_raw_default",
        publication_ready=True,
        suites=("manifesto", "publication"),
        command=_python_command("scripts/run_manifesto_teacher_fg_leaf_grid.py", "--output-dir", "{output_root}"),
    ),
    RunTargetRecord(
        target="manifesto.teacher_joint_leaf_grid",
        path="scripts/run_manifesto_teacher_fg_joint_leaf_grid.py",
        domain="manifesto_rile",
        role="joint_teacher_tree_bundle",
        backend="dspy",
        status="canonical",
        expected_input_contract=TREE_BUNDLE_SCHEMA_VERSION,
        audit_policy="fail_hard_tree_bundle_raw_default",
        publication_ready=True,
        suites=("manifesto", "publication"),
        command=_python_command("scripts/run_manifesto_teacher_fg_joint_leaf_grid.py", "--output-dir", "{output_root}"),
    ),
    RunTargetRecord(
        target="manifesto.alternating_ladder",
        path="scripts/run_alternating_ladder.py",
        domain="manifesto_rile",
        role="fg_ladder_runner",
        backend="dspy",
        status="canonical",
        expected_input_contract=TREE_BUNDLE_SCHEMA_VERSION,
        audit_policy="preflight_required",
        publication_ready=True,
        suites=("manifesto", "publication"),
        command=_python_command("scripts/run_alternating_ladder.py", "--output-dir", "{output_root}"),
    ),
    RunTargetRecord(
        target="manifesto.benoit_supervised_ladder",
        path="scripts/run_benoit_supervised_dspy_ladder.sh",
        domain="manifesto_rile",
        role="single_dimension_launcher",
        backend="dspy",
        status="thin_wrapper",
        expected_input_contract=TREE_BUNDLE_SCHEMA_VERSION,
        audit_policy="shell_audit_required",
        publication_ready=True,
        suites=("manifesto", "publication"),
        command=("bash", "scripts/run_benoit_supervised_dspy_ladder.sh"),
    ),
    RunTargetRecord(
        target="manifesto.benoit_combined_ladder",
        path="scripts/run_benoit_combined_dspy_ladder.sh",
        domain="manifesto_rile",
        role="combined_scalar_launcher",
        backend="dspy",
        status="thin_wrapper",
        expected_input_contract=TREE_BUNDLE_SCHEMA_VERSION,
        audit_policy="shell_audit_required",
        publication_ready=True,
        suites=("manifesto", "publication"),
        command=("bash", "scripts/run_benoit_combined_dspy_ladder.sh"),
    ),
    RunTargetRecord(
        target="manifesto.benoit_joint_ladder",
        path="scripts/run_benoit_combined_joint_teacher_dspy_ladder.sh",
        domain="manifesto_rile",
        role="joint_launcher",
        backend="dspy",
        status="thin_wrapper",
        expected_input_contract=TREE_BUNDLE_SCHEMA_VERSION,
        audit_policy="shell_audit_required",
        publication_ready=True,
        suites=("manifesto", "publication"),
        command=("bash", "scripts/run_benoit_combined_joint_teacher_dspy_ladder.sh"),
    ),
    RunTargetRecord(
        target="markov.publication_bundle",
        path="scripts/run_markov_publication_bundle.py",
        domain="markov",
        role="publication_bundle",
        backend="fno",
        status="canonical",
        expected_input_contract=TREE_BUNDLE_SCHEMA_VERSION,
        audit_policy="manifest_contract_required",
        publication_ready=True,
        suites=("markov", "publication"),
        command=_python_command("scripts/run_markov_publication_bundle.py", "--output-root", "{output_root}"),
    ),
    RunTargetRecord(
        target="markov.tradeoff_pipeline",
        path="scripts/run_markov_optimization_tradeoff_pipeline.py",
        domain="markov",
        role="tradeoff_pipeline",
        backend="fno",
        status="canonical",
        expected_input_contract=TREE_BUNDLE_SCHEMA_VERSION,
        audit_policy="manifest_contract_required",
        publication_ready=True,
        suites=("markov", "publication"),
        command=_python_command(
            "scripts/run_markov_optimization_tradeoff_pipeline.py",
            "--output-root",
            "{output_root}",
        ),
    ),
    RunTargetRecord(
        target="classical_sketch.paper_bundle",
        path="scripts/run_classical_sketches_paper_bundle.py",
        domain="classical_sketch",
        role="paper_bundle",
        backend="treepo",
        status="canonical",
        expected_input_contract=TREE_BUNDLE_SCHEMA_VERSION,
        audit_policy="post_report_required",
        publication_ready=True,
        suites=("classical_sketch", "publication"),
        command=_python_command("scripts/run_classical_sketches_paper_bundle.py", "--out-root", "{output_root}"),
    ),
    RunTargetRecord(
        target="classical_sketch.treepo_bench",
        path="treepo/src/treepo/bench/runner.py",
        domain="classical_sketch",
        role="bench_runner",
        backend="treepo",
        status="thin_wrapper",
        expected_input_contract=TREE_BUNDLE_SCHEMA_VERSION,
        audit_policy="learned_requires_unified_g",
        publication_ready=True,
        suites=("classical_sketch", "publication"),
    ),
    RunTargetRecord(
        target="classical_sketch.unified_g_grid",
        path="parallel/unified_g_v1/src/unified_g_v1/sketch/classical_sketch_grid.py",
        domain="classical_sketch",
        role="unified_g_grid",
        backend="unified_g",
        status="canonical",
        expected_input_contract=TREE_BUNDLE_SCHEMA_VERSION,
        audit_policy="row_metadata_required",
        publication_ready=True,
        suites=("classical_sketch", "publication"),
    ),
    RunTargetRecord(
        target="classical_sketch.learned_overlay",
        path="parallel/unified_g_v1/src/unified_g_v1/sketch/learned_sketch_grid.py",
        domain="classical_sketch",
        role="learned_overlay",
        backend="unified_g",
        status="canonical",
        expected_input_contract=TREE_BUNDLE_SCHEMA_VERSION,
        audit_policy="row_metadata_required",
        publication_ready=True,
        suites=("classical_sketch", "publication"),
    ),
    RunTargetRecord(
        target="runtime.longbench",
        path="scripts/run_runtime_eval.py",
        domain="runtime_eval",
        role="longbench_runtime",
        backend="runtime",
        status="canonical",
        expected_input_contract="runtime_eval_manifest",
        audit_policy="run_manifest_required",
        publication_ready=False,
        suites=("runtime", "exploratory"),
        command=_python_command("scripts/run_runtime_eval.py", "--output-dir", "{output_root}"),
    ),
    RunTargetRecord(
        target="manifesto.legacy_existing_ladder_eval",
        path="scripts/evaluate_manifesto_existing_dspy_ladder.py",
        domain="manifesto_rile",
        role="legacy_evaluation",
        backend="dspy",
        status="legacy_compat",
        expected_input_contract="legacy_or_tree_bundle_v1",
        audit_policy="explicit_legacy_only",
        publication_ready=False,
        suites=("legacy",),
    ),
    RunTargetRecord(
        target="classical_sketch.legacy_learned_sketch_simulation",
        path="scripts/run_learned_sketch_simulation.py",
        domain="classical_sketch",
        role="legacy_sketch_simulation",
        backend="torch",
        status="not_publication_safe",
        expected_input_contract="none",
        audit_policy="excluded_from_publication_defaults",
        publication_ready=False,
        suites=("legacy",),
    ),
    RunTargetRecord(
        target="classical_sketch.fno_mergeable_diagnostic",
        path="scripts/run_fno_mergeable_sketch_diagnostic.py",
        domain="classical_sketch",
        role="diagnostic",
        backend="fno",
        status="not_publication_safe",
        expected_input_contract="none",
        audit_policy="excluded_from_publication_defaults",
        publication_ready=False,
        suites=("legacy", "diagnostic"),
    ),
)


def _simulation_suite_targets() -> tuple[RunTargetRecord, ...]:
    try:
        from treepo._research.ctreepo.sim.suite.registry import iter_canonical_suite_targets
    except Exception:
        return ()

    rows: list[RunTargetRecord] = []
    for target in iter_canonical_suite_targets(bundle_roles=("paper", "appendix", "diagnostic")):
        publication = str(target.bundle_role) in {"paper", "appendix"}
        rows.append(
            RunTargetRecord(
                target=f"sim.{target.key}",
                path="src/ctreepo/sim/suite/registry.py",
                domain="simulation",
                role=str(target.bundle_role),
                backend="ctreepo_sim_suite",
                status="thin_wrapper",
                expected_input_contract=RUN_MANIFEST_SCHEMA_VERSION,
                audit_policy="run_manifest_required",
                publication_ready=publication,
                suites=("simulation", "publication" if publication else "diagnostic"),
                command=_python_command(
                    "-m",
                    "src.ctreepo.cli",
                    "sim",
                    "suite",
                    str(target.cli_suite_name),
                    "report",
                    "--output-root",
                    "{output_root}",
                ),
                notes=str(target.description),
            )
        )
    return tuple(rows)


def iter_run_targets(*, suites: Iterable[str] | None = None) -> list[RunTargetRecord]:
    requested = {str(item) for item in (suites or ()) if str(item).strip()}
    rows = list(_BASE_TARGETS) + list(_simulation_suite_targets())
    if requested:
        rows = [
            row
            for row in rows
            if requested.intersection({str(suite) for suite in row.suites})
        ]
    return sorted(rows, key=lambda row: row.target)


def run_targets_by_name() -> Mapping[str, RunTargetRecord]:
    return {row.target: row for row in iter_run_targets()}


def get_run_target(target: str) -> RunTargetRecord:
    rows = run_targets_by_name()
    try:
        return rows[str(target)]
    except KeyError as exc:
        choices = ", ".join(sorted(rows))
        raise KeyError(f"unknown C-TreePO run target {target!r}; choices: {choices}") from exc


def audit_target_records(records: Iterable[RunTargetRecord] | None = None) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for record in records if records is not None else iter_run_targets():
        if record.target in seen:
            errors.append(f"{record.target}: duplicate target")
        seen.add(record.target)
        if record.status not in TARGET_STATUSES:
            errors.append(f"{record.target}: invalid status {record.status!r}")
        if record.output_contract != RUN_MANIFEST_SCHEMA_VERSION:
            errors.append(f"{record.target}: output_contract must be {RUN_MANIFEST_SCHEMA_VERSION}")
        if not record.audit_policy:
            errors.append(f"{record.target}: missing audit_policy")
        if record.publication_ready and record.status not in {"canonical", "thin_wrapper"}:
            errors.append(f"{record.target}: publication-ready target has unsafe status {record.status!r}")
        if record.publication_ready and record.expected_input_contract in {"", "none"}:
            errors.append(f"{record.target}: publication-ready target lacks input contract")
        path = PROJECT_ROOT / record.path
        if not path.exists():
            errors.append(f"{record.target}: registered path does not exist: {record.path}")
    return errors


__all__ = [
    "RunTargetRecord",
    "TARGET_STATUSES",
    "audit_target_records",
    "get_run_target",
    "iter_run_targets",
    "run_targets_by_name",
]
