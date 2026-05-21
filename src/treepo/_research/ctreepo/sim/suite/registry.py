from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class SuiteModuleSpec:
    name: str
    module: str
    default_output_root_stem: str


@dataclass(frozen=True)
class CanonicalSuiteTarget:
    key: str
    title: str
    description: str
    cli_suite_name: str
    root_rel: str
    report_args: Sequence[str]
    expected_outputs: Sequence[str]
    bundle_role: str
    theory_alignment_key: str
    theory_title: str
    theory_families: Sequence[str]
    theory_role: str
    theory_demonstrates: str
    pending_note: str = ""
    always_run: bool = False


_SUITE_MODULES: Dict[str, SuiteModuleSpec] = {
    "identifiable-zero": SuiteModuleSpec(
        name="identifiable-zero",
        module="src.ctreepo.sim.suite.identifiable_zero",
        default_output_root_stem="identifiable_zero_suite",
    ),
    "identifiable-zero-publication": SuiteModuleSpec(
        name="identifiable-zero-publication",
        module="src.ctreepo.sim.suite.identifiable_zero_publication",
        default_output_root_stem="identifiable_zero_publication",
    ),
    "publication-ctreepo": SuiteModuleSpec(
        name="publication-ctreepo",
        module="src.ctreepo.sim.suite.publication_ctreepo",
        default_output_root_stem="identifiable_zero_publication_ctreepo",
    ),
    "identifiable-zero-learnability": SuiteModuleSpec(
        name="identifiable-zero-learnability",
        module="src.ctreepo.sim.suite.identifiable_zero_learnability",
        default_output_root_stem="identifiable_zero_learnability",
    ),
    "law-stress": SuiteModuleSpec(
        name="law-stress",
        module="src.ctreepo.sim.suite.law_stress",
        default_output_root_stem="law_stress_suite",
    ),
    "cpu-megasweep": SuiteModuleSpec(
        name="cpu-megasweep",
        module="src.ctreepo.sim.suite.cpu_megasweep_v2",
        default_output_root_stem="cpu_megasweep",
    ),
    "simulation-buildout": SuiteModuleSpec(
        name="simulation-buildout",
        module="src.ctreepo.sim.suite.simulation_buildout_v2",
        default_output_root_stem="simulation_buildout",
    ),
    "identifiable-zero-neural-operator": SuiteModuleSpec(
        name="identifiable-zero-neural-operator",
        module="src.ctreepo.sim.suite.identifiable_zero_neural_operator",
        default_output_root_stem="identifiable_zero_neural_operator_v2",
    ),
    "identifiable-zero-lda-leafnoise": SuiteModuleSpec(
        name="identifiable-zero-lda-leafnoise",
        module="src.ctreepo.sim.suite.identifiable_zero_lda_leafnoise",
        default_output_root_stem="identifiable_zero_lda_leafnoise",
    ),
    "identifiable-zero-dtm-lda": SuiteModuleSpec(
        name="identifiable-zero-dtm-lda",
        module="src.ctreepo.sim.suite.identifiable_zero_dtm_lda",
        default_output_root_stem="identifiable_zero_dtm_lda",
    ),
    "lda-tree-recovery-progress": SuiteModuleSpec(
        name="lda-tree-recovery-progress",
        module="src.ctreepo.sim.suite.lda_tree_recovery_progress",
        default_output_root_stem="lda_tree_recovery_production",
    ),
    "learned-sketch-smoke": SuiteModuleSpec(
        name="learned-sketch-smoke",
        module="src.ctreepo.sim.suite.learned_sketch_smoke",
        default_output_root_stem="learned_sketch_smoke",
    ),
    "markov-observed-token": SuiteModuleSpec(
        name="markov-observed-token",
        module="src.ctreepo.sim.suite.markov_observed_token",
        default_output_root_stem="markov_observed_token",
    ),
    "markov-full-doc-anchor": SuiteModuleSpec(
        name="markov-full-doc-anchor",
        module="src.ctreepo.sim.suite.markov_full_doc_anchor",
        default_output_root_stem="markov_full_doc_anchor",
    ),
    "markov-full-tree-ipw": SuiteModuleSpec(
        name="markov-full-tree-ipw",
        module="src.ctreepo.sim.suite.markov_full_tree_ipw",
        default_output_root_stem="markov_full_tree_ipw",
    ),
}


_CANONICAL_TARGETS: Dict[str, CanonicalSuiteTarget] = {
    "cpu_megasweep": CanonicalSuiteTarget(
        key="cpu_megasweep",
        title="CPU Megasweep",
        description="Baseline anchor figures and reference diagnostics.",
        cli_suite_name="cpu-megasweep",
        root_rel="cpu_megasweep",
        report_args=(),
        expected_outputs=(
            "{root}/figures/megasweep_consolidated_report.md",
            "{root}/figures/megasweep_consolidated_readable_report.md",
        ),
        bundle_role="paper",
        theory_alignment_key="cpu_megasweep",
        theory_title="CPU Megasweep",
        theory_families=("markov_ops_count", "segment_lda_ops_weight_recovery", "segmented_lda_ctreepo"),
        theory_role="Baseline anchor",
        theory_demonstrates="Exact ceilings, wrong-state controls, and core budget trends across the main simulation families.",
    ),
    "simulation_buildout": CanonicalSuiteTarget(
        key="simulation_buildout",
        title="Simulation Buildout",
        description="Focused stress/mechanism buildout layer.",
        cli_suite_name="simulation-buildout",
        root_rel="simulation_buildout",
        report_args=(),
        expected_outputs=("{root}/figures/simulation_buildout_report.md",),
        bundle_role="paper",
        theory_alignment_key="simulation_buildout",
        theory_title="Simulation Buildout",
        theory_families=(
            "markov_ops_count",
            "segment_lda_ops_weight_recovery",
            "segmented_lda_ctreepo",
            "mergeable_ablation",
            "local_law_learnability",
        ),
        theory_role="Stress / mechanism layer",
        theory_demonstrates="Why the method works, where it fails, and how estimator stress and control baselines map to the theory.",
        always_run=True,
        pending_note="The buildout root exists but may still lack figure JSONs. Draft markdown is still useful.",
    ),
    "publication_clean": CanonicalSuiteTarget(
        key="publication_clean",
        title="Identifiable-Zero Publication Clean",
        description="Main clean cross-family publication draft.",
        cli_suite_name="identifiable-zero-publication",
        root_rel="identifiable_zero_longrun_clean",
        report_args=("--profile", "publication_clean"),
        expected_outputs=(
            "{root}/figures/identifiable_zero_publication_report_latest.md",
            "{root}/figures/identifiable_zero_publication_report_latest_diagnostics.json",
        ),
        bundle_role="paper",
        theory_alignment_key="publication_clean",
        theory_title="Identifiable-Zero Publication Clean",
        theory_families=("markov_ops_count", "segment_lda_ops_weight_recovery", "segmented_lda_ctreepo"),
        theory_role="Main paper slice",
        theory_demonstrates="Compact cross-family publication figures for the clean exact-plus-approx story.",
    ),
    "publication_ctreepo_progress": CanonicalSuiteTarget(
        key="publication_ctreepo_progress",
        title="Publication C-TreePO Progress",
        description="Partial-run C-TreePO publication progress report.",
        cli_suite_name="publication-ctreepo",
        root_rel="identifiable_zero_publication_ctreepo",
        report_args=(),
        expected_outputs=(
            "{root}/figures/publication_progress/publication_ctreepo_progress_latest.md",
            "{root}/figures/publication_progress/publication_ctreepo_progress_diagnostics.json",
        ),
        bundle_role="paper",
        theory_alignment_key="publication_ctreepo_progress",
        theory_title="Publication C-TreePO Progress",
        theory_families=("segmented_lda_ctreepo",),
        theory_role="Approximate-tree progress slice",
        theory_demonstrates="The calibrated/budgeted approximate local-law story while the richer C-TreePO sweep is in flight.",
    ),
    "learnability": CanonicalSuiteTarget(
        key="learnability",
        title="Identifiable-Zero Learnability",
        description="Appendix-quality learnability report.",
        cli_suite_name="identifiable-zero-learnability",
        root_rel="identifiable_zero_learnability",
        report_args=(),
        expected_outputs=("{root}/figures/learnability/identifiable_zero_learnability_latest.md",),
        bundle_role="appendix",
        theory_alignment_key="learnability",
        theory_title="Identifiable-Zero Learnability",
        theory_families=("markov_ops_count", "segmented_lda_ctreepo", "local_law_learnability"),
        theory_role="Appendix learnability slice",
        theory_demonstrates="Whether held-out local-law supervision actually translates into downstream gains.",
        pending_note="Learnability reruns have not been generated in the current formal root yet.",
    ),
    "neural_operator_overnight": CanonicalSuiteTarget(
        key="neural_operator_overnight",
        title="Identifiable-Zero Neural Operator Overnight",
        description="Neural-operator overnight robustness report.",
        cli_suite_name="identifiable-zero-neural-operator",
        root_rel="identifiable_zero_neural_operator_v2",
        report_args=(),
        expected_outputs=(
            "{root}/figures/neural_operator_overnight/identifiable_zero_neural_operator_overnight_latest.md",
        ),
        bundle_role="appendix",
        theory_alignment_key="neural_operator_overnight",
        theory_title="Identifiable-Zero Neural Operator Overnight",
        theory_families=("segmented_lda_ctreepo", "local_law_learnability"),
        theory_role="Operator-capacity robustness",
        theory_demonstrates="Robustness of the approximate/audited operator story under neural-operator capacity changes.",
    ),
    "lda_leafnoise": CanonicalSuiteTarget(
        key="lda_leafnoise",
        title="Identifiable-Zero LDA Leafnoise",
        description="Appendix-style leaf-noise progression report.",
        cli_suite_name="identifiable-zero-lda-leafnoise",
        root_rel="identifiable_zero_lda_leafnoise",
        report_args=(),
        expected_outputs=("{root}/figures/lda_leafnoise/identifiable_zero_lda_leafnoise_latest.md",),
        bundle_role="appendix",
        theory_alignment_key="lda_leafnoise",
        theory_title="Identifiable-Zero LDA Leafnoise",
        theory_families=("segment_lda_ops_weight_recovery", "segmented_lda_ctreepo"),
        theory_role="Appendix robustness slice",
        theory_demonstrates="How the theorem-friendly LDA story degrades under leaf noise.",
        pending_note="Leaf-noise reruns have not been generated in the current formal root yet.",
    ),
    "dtm_lda": CanonicalSuiteTarget(
        key="dtm_lda",
        title="Identifiable-Zero DTM-LDA",
        description="DTM-LDA appendix/robustness suite.",
        cli_suite_name="identifiable-zero-dtm-lda",
        root_rel="identifiable_zero_dtm_lda",
        report_args=(),
        expected_outputs=("{root}/figures/dtm_lda/identifiable_zero_dtm_lda_latest.md",),
        bundle_role="appendix",
        theory_alignment_key="dtm_lda",
        theory_title="Identifiable-Zero DTM-LDA",
        theory_families=("segmented_lda_ctreepo",),
        theory_role="Appendix robustness slice",
        theory_demonstrates="A broader topic-model robustness pass for the approximate/audited tree stack.",
        pending_note="DTM-LDA reruns have not been generated in the current formal root yet.",
    ),
    "lda_tree_recovery_progress": CanonicalSuiteTarget(
        key="lda_tree_recovery_progress",
        title="Diagnostic: LDA Tree Recovery Progress",
        description="Diagnostic-only LDA tree-recovery production progress report.",
        cli_suite_name="lda-tree-recovery-progress",
        root_rel="lda_tree_recovery_production",
        report_args=(),
        expected_outputs=(
            "{root}/report/lda_tree_recovery_progress_report.pdf",
            "{root}/report/lda_tree_recovery_progress_summary.json",
        ),
        bundle_role="diagnostic",
        theory_alignment_key="lda_tree_recovery_progress",
        theory_title="Diagnostic: LDA Tree Recovery Progress",
        theory_families=("segment_lda_ops_weight_recovery",),
        theory_role="Diagnostic exact-control slice",
        theory_demonstrates="The exact-vs-learned tree recovery ladder for the LDA mergeable baseline family.",
    ),
    "markov_observed_token": CanonicalSuiteTarget(
        key="markov_observed_token",
        title="Observed-Token Markov Comparison",
        description="Appendix-facing fixed-bundle comparison between root-only learning and sampled local labels.",
        cli_suite_name="markov-observed-token",
        root_rel="markov_observed_token",
        report_args=(),
        expected_outputs=(
            "{root}/figures/markov_observed_token/markov_observed_token_latest.md",
            "{root}/figures/markov_observed_token/markov_observed_token_latest_diagnostics.json",
        ),
        bundle_role="appendix",
        theory_alignment_key="markov_observed_token",
        theory_title="Observed-Token Markov Comparison",
        theory_families=("markov_ops_count", "local_law_learnability"),
        theory_role="Appendix observed-token slice",
        theory_demonstrates=(
            "That the Markov changepoint task is learnable from observed tokens alone, "
            "and how fixed-bundle sampled local labels compare against full-document controls."
        ),
        pending_note="Observed-token reruns have not been generated in the current formal root yet.",
    ),
    "markov_full_doc_anchor": CanonicalSuiteTarget(
        key="markov_full_doc_anchor",
        title="Diagnostic: Markov Full-Doc Anchor",
        description="Paper-facing full-doc diagnostics, ladder alignment, and provenance checks.",
        cli_suite_name="markov-full-doc-anchor",
        root_rel="markov_full_doc_anchor",
        report_args=(),
        expected_outputs=(
            "{root}/figures/markov_full_doc_anchor/markov_full_doc_anchor_latest.md",
            "{root}/figures/markov_full_doc_anchor/markov_full_doc_anchor_latest_diagnostics.json",
        ),
        bundle_role="diagnostic",
        theory_alignment_key="markov_full_doc_anchor",
        theory_title="Diagnostic: Markov Full-Doc Anchor",
        theory_families=("markov_ops_count",),
        theory_role="Paper-facing approximate/proxy full-doc slice",
        theory_demonstrates=(
            "Whether the official FNO and tree-neural publication lanes are semantically aligned, "
            "provenance-labeled, and compared on matched fixed bundles."
        ),
    ),
    "markov_full_tree_ipw": CanonicalSuiteTarget(
        key="markov_full_tree_ipw",
        title="Diagnostic: Markov Full-Tree IPW",
        description="Diagnostic full-tree IPW estimand/probability alignment and point-estimation checks.",
        cli_suite_name="markov-full-tree-ipw",
        root_rel="markov_full_tree_ipw",
        report_args=(),
        expected_outputs=(
            "{root}/figures/markov_full_tree_ipw/markov_full_tree_ipw_latest.md",
            "{root}/figures/markov_full_tree_ipw/markov_full_tree_ipw_latest_diagnostics.json",
        ),
        bundle_role="diagnostic",
        theory_alignment_key="markov_full_tree_ipw",
        theory_title="Diagnostic: Markov Full-Tree IPW",
        theory_families=("markov_ops_count",),
        theory_role="Diagnostic IPW estimand slice",
        theory_demonstrates=(
            "Whether the full-tree IPW grid is reporting the realized-node estimand, "
            "sampling design, and endpoint behavior claimed by the Markov audit story."
        ),
    ),
}


def suite_module_specs() -> Mapping[str, SuiteModuleSpec]:
    return dict(_SUITE_MODULES)


def suite_module_names() -> List[str]:
    return sorted(_SUITE_MODULES)


def suite_module_spec(name: str) -> SuiteModuleSpec:
    return _SUITE_MODULES[str(name)]


def canonical_suite_targets() -> Mapping[str, CanonicalSuiteTarget]:
    return dict(_CANONICAL_TARGETS)


def canonical_suite_target(key: str) -> CanonicalSuiteTarget:
    return _CANONICAL_TARGETS[str(key)]


def iter_canonical_suite_targets(*, bundle_roles: Iterable[str] | None = None) -> List[CanonicalSuiteTarget]:
    allowed = {str(x) for x in (bundle_roles or []) if str(x).strip()}
    rows = list(_CANONICAL_TARGETS.values())
    if allowed:
        rows = [row for row in rows if str(row.bundle_role) in allowed]
    return rows


__all__ = [
    "CanonicalSuiteTarget",
    "SuiteModuleSpec",
    "canonical_suite_target",
    "canonical_suite_targets",
    "iter_canonical_suite_targets",
    "suite_module_names",
    "suite_module_spec",
    "suite_module_specs",
]
