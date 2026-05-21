from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from treepo._research.ctreepo.sim.expectations import ExpectationFinding, ExpectationReport, build_expectation_report
from treepo._research.ctreepo.sim.suite.registry import iter_canonical_suite_targets


@dataclass(frozen=True)
class LeanRef:
    label: str
    theorem: str
    file: str
    note: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "LeanRef":
        return cls(
            label=str(d.get("label", "")),
            theorem=str(d.get("theorem", "")),
            file=str(d.get("file", "")),
            note=str(d.get("note", "")),
        )


@dataclass(frozen=True)
class FamilyTheorySpec:
    family: str
    title: str
    simulation_claim: str
    why_it_demonstrates_the_method: str
    key_assumptions: List[str]
    exact_lanes: List[str]
    approx_lanes: List[str]
    proxy_lanes: List[str]
    lean_refs: List[LeanRef]
    canonical_suites: List[str]
    caveats: List[str]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["lean_refs"] = [ref.to_dict() for ref in self.lean_refs]
        return payload


@dataclass(frozen=True)
class SuiteTheorySpec:
    name: str
    title: str
    families: List[str]
    role: str
    demonstrates: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FamilyAlignmentStatus:
    family: str
    title: str
    expectation_counts: Dict[str, int]
    relevant_suites: List[Dict[str, object]]
    overall_status: str
    note: str
    spec: FamilyTheorySpec

    def to_dict(self) -> Dict[str, object]:
        return {
            "family": self.family,
            "title": self.title,
            "expectation_counts": dict(self.expectation_counts),
            "relevant_suites": [dict(x) for x in self.relevant_suites],
            "overall_status": self.overall_status,
            "note": self.note,
            "spec": self.spec.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "FamilyAlignmentStatus":
        spec = d.get("spec", {}) or {}
        lean_refs = [LeanRef.from_dict(x) for x in (spec.get("lean_refs", []) if isinstance(spec, Mapping) else [])]
        theory_spec = FamilyTheorySpec(
            family=str(spec.get("family", "")) if isinstance(spec, Mapping) else "",
            title=str(spec.get("title", "")) if isinstance(spec, Mapping) else "",
            simulation_claim=str(spec.get("simulation_claim", "")) if isinstance(spec, Mapping) else "",
            why_it_demonstrates_the_method=str(spec.get("why_it_demonstrates_the_method", "")) if isinstance(spec, Mapping) else "",
            key_assumptions=[str(x) for x in ((spec.get("key_assumptions", []) if isinstance(spec, Mapping) else []) or [])],
            exact_lanes=[str(x) for x in ((spec.get("exact_lanes", []) if isinstance(spec, Mapping) else []) or [])],
            approx_lanes=[str(x) for x in ((spec.get("approx_lanes", []) if isinstance(spec, Mapping) else []) or [])],
            proxy_lanes=[str(x) for x in ((spec.get("proxy_lanes", []) if isinstance(spec, Mapping) else []) or [])],
            lean_refs=lean_refs,
            canonical_suites=[str(x) for x in ((spec.get("canonical_suites", []) if isinstance(spec, Mapping) else []) or [])],
            caveats=[str(x) for x in ((spec.get("caveats", []) if isinstance(spec, Mapping) else []) or [])],
        )
        return cls(
            family=str(d.get("family", "")),
            title=str(d.get("title", "")),
            expectation_counts={str(k): int(v) for k, v in dict(d.get("expectation_counts", {}) or {}).items()},
            relevant_suites=[dict(x) for x in (d.get("relevant_suites", []) or [])],
            overall_status=str(d.get("overall_status", "")),
            note=str(d.get("note", "")),
            spec=theory_spec,
        )


@dataclass(frozen=True)
class SimulationTheoryAlignmentReport:
    formal_root: Optional[str]
    expectation_source: Optional[str]
    bundle_manifest: Optional[str]
    family_statuses: List[FamilyAlignmentStatus]
    suites: List[Dict[str, object]]
    summary: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "formal_root": self.formal_root,
            "expectation_source": self.expectation_source,
            "bundle_manifest": self.bundle_manifest,
            "family_statuses": [x.to_dict() for x in self.family_statuses],
            "suites": [dict(x) for x in self.suites],
            "summary": dict(self.summary),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "SimulationTheoryAlignmentReport":
        return cls(
            formal_root=str(d["formal_root"]) if d.get("formal_root") is not None else None,
            expectation_source=str(d["expectation_source"]) if d.get("expectation_source") is not None else None,
            bundle_manifest=str(d["bundle_manifest"]) if d.get("bundle_manifest") is not None else None,
            family_statuses=[FamilyAlignmentStatus.from_dict(x) for x in (d.get("family_statuses", []) or [])],
            suites=[dict(x) for x in (d.get("suites", []) or [])],
            summary=dict(d.get("summary", {}) or {}),
        )


def _family_specs() -> Dict[str, FamilyTheorySpec]:
    return {
        "markov_ops_count": FamilyTheorySpec(
            family="markov_ops_count",
            title="Markov OPS Count",
            simulation_claim=(
                "Exact Markov sketches give a zero-distortion theorem-backed ceiling, "
                "while count-only and flip controls isolate missing state and missing L3/idempotence. "
                "Learned neural lanes are approximate local-law experiments rather than exact guarantees, "
                "and the paper-facing full-doc anchor lane should be read as an audited publication surface rather than "
                "a direct theorem witness."
            ),
            why_it_demonstrates_the_method=(
                "This is the clearest exact-vs-learned-vs-wrong-baseline demonstration of local-law structure: "
                "the exact lane should sit at zero, the count-only baseline should remain separated, and the learned "
                "lane should only be expected to improve where the approximate local laws are actually learned. "
                "The full-doc anchor diagnostics extend that story to the public paper lane by checking provenance, "
                "semantic labeling, and matched fixed-bundle comparisons."
            ),
            key_assumptions=[
                "Exact encoded-feature controls satisfy the theorem-domain local laws exactly.",
                "Learned lanes are only approximate/audited unless they are backed by an explicit theorem witness.",
                "Flip controls are deliberate violations of on-range stability or schedule stability.",
                "Full-doc official FNO and tree-neural baselines are publication-facing proxy/approximate lanes, not exact Lean-backed operators.",
            ],
            exact_lanes=["exact", "oracle_tree", "one_pass_reference"],
            approx_lanes=["learned", "tree_neural_* full-doc lane"],
            proxy_lanes=["undersupported", "flip_R1", "flip_R2", "official_fno full-doc lane"],
            lean_refs=[
                LeanRef(
                    label="Markov path local laws",
                    theorem="markov_path_local_laws_of_encoded_state",
                    file="lean3/FormalProofs/OPT/MainTheorems.lean",
                    note="Raw regime-path documents induce exact local laws through the exact endpoint+count encoder.",
                ),
                LeanRef(
                    label="Markov path utility exact on tree",
                    theorem="markov_path_state_exact_on_tree",
                    file="lean3/FormalProofs/OPT/MainTheorems.lean",
                    note="Utilities on the exact Markov sketch state are preserved exactly under tree reduction of raw paths.",
                ),
                LeanRef(
                    label="Markov path count exact on tree",
                    theorem="markov_path_count_exact_on_tree",
                    file="lean3/FormalProofs/OPT/MainTheorems.lean",
                    note="The changepoint count itself is preserved exactly on every tree of realized paths.",
                ),
                LeanRef(
                    label="Count-only counterexample",
                    theorem="markov_countOnly_mergeFold_counterexample",
                    file="lean3/FormalProofs/OPT/MainTheorems.lean",
                    note="The count-only undersupported control is not compositionally sufficient, even on a tiny tree.",
                ),
                LeanRef(
                    label="L3 is genuinely extra",
                    theorem="not_L3_gFlip",
                    file="lean3/FormalProofs/OPT/MarkovCountSketchExample.lean",
                    note="A flip-style control can preserve other structure while failing on-range stability.",
                ),
            ],
            canonical_suites=[
                "cpu_megasweep",
                "publication_clean",
                "learnability",
                "simulation_buildout",
                "markov_observed_token",
                "markov_full_doc_anchor",
                "markov_full_tree_ipw",
            ],
            caveats=[
                "Learned Markov operators remain approximate/audited rather than theorem-backed unless an explicit theorem-domain witness is attached.",
                "The strongest paper-facing cross-family slice is only as clean as the matched baseline coverage in the rerun root.",
                "Full-doc anchor diagnostics should not silently mix legacy tree-neural semantics with current score-drift semantics.",
                "The full-tree IPW grid is a point-estimation diagnostic surface unless an explicit honest/CI wrapper is attached.",
            ],
        ),
        "segment_lda_ops_weight_recovery": FamilyTheorySpec(
            family="segment_lda_ops_weight_recovery",
            title="Segment-LDA OPS Weight Recovery",
            simulation_claim=(
                "Ordinary bag-of-words quantities are exactly mergeable, while boundary-sensitive objectives create a principled gap between pooled and leaf-aware summaries."
            ),
            why_it_demonstrates_the_method=(
                "This family shows both sides of the method: exact mergeable ceilings when the sufficient statistic is right, "
                "and a controlled failure mode when nonlinear local structure matters."
            ),
            key_assumptions=[
                "The bag-of-words/count-sketch control is an exact mergeable statistic.",
                "Boundary-sensitive targets are intentionally not recoverable from pooled counts alone.",
                "Ridge and inferred-topic lanes are approximation experiments, not theorem-backed exact controls.",
            ],
            exact_lanes=["exact", "ridge_true_topics"],
            approx_lanes=["ridge", "ridge_infer_true_phi", "ridge_infer_est_phi"],
            proxy_lanes=["undersupported", "flip_R1", "flip_R2"],
            lean_refs=[
                LeanRef(
                    label="Bag-of-words exact tree recovery",
                    theorem="sketchReduce_countSketch_eq_bagOfWords",
                    file="lean3/FormalProofs/OPT/BagOfWordsLDARecovery.lean",
                    note="Exact count-sketch tree reduction recovers the full bag-of-words histogram.",
                ),
                LeanRef(
                    label="LDA likelihood exact on tree",
                    theorem="ldaDocumentLikelihood_exact_on_tree",
                    file="lean3/FormalProofs/OPT/BagOfWordsLDARecovery.lean",
                    note="Ordinary bag-of-words LDA document likelihood is exactly preserved by the mergeable sketch.",
                ),
                LeanRef(
                    label="Nonlinear local-mixture gap",
                    theorem="affineQuadratic_gap_eq_quadratic_gap",
                    file="lean3/FormalProofs/OPT/LeafLocalMixtureUtilityGap.lean",
                    note="The pooled-vs-leaf gap is exactly the nonlinear interaction term.",
                ),
                LeanRef(
                    label="Null control at lambda = 0",
                    theorem="affineQuadratic_gap_zero_lambda",
                    file="lean3/FormalProofs/OPT/LeafLocalMixtureUtilityGap.lean",
                    note="When the nonlinear term is turned off, the extra leaf-local gap disappears.",
                ),
            ],
            canonical_suites=["cpu_megasweep", "publication_clean", "lda_tree_recovery_progress", "lda_leafnoise"],
            caveats=[
                "Boundary-sensitive lanes test a deliberately richer target than ordinary bag-of-words mergeability.",
                "Some legacy roots reuse a quadratic utility weight field that should not be read as the normalized paper lambda.",
            ],
        ),
        "segmented_lda_ctreepo": FamilyTheorySpec(
            family="segmented_lda_ctreepo",
            title="Segmented-LDA C-TreePO",
            simulation_claim=(
                "Calibrated, budgeted, approximate local-law pipelines should tighten root error and decomposition slack relative to uncalibrated baselines."
            ),
            why_it_demonstrates_the_method=(
                "This is the main approximate/audit story: it is where theorem-backed exactness gives way to "
                "audited local-law budgets, oracle measurement envelopes, and expected-tree optimization."
            ),
            key_assumptions=[
                "Oracle-tree lanes are exact/theorem-backed controls.",
                "Estimated/calibrated/budgeted lanes are approximate/audited rather than exact.",
                "Decomposition lanes should be read against the audited upper-bound/objective story, not as exact equalities.",
            ],
            exact_lanes=["oracle_tree"],
            approx_lanes=["estimated_uncalibrated", "estimated_calibrated", "estimated_calibrated_budgeted", "decomposition"],
            proxy_lanes=[],
            lean_refs=[
                LeanRef(
                    label="Expected-tree optimization under oracle uncertainty",
                    theorem="dpo_expected_tree_argmin_subset_true_pointwiseEpsilonArgmin_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement",
                    file="lean3/FormalProofs/OPT/OptimizationPerturbation.lean",
                    note="Adaptive approximate local laws plus oracle uncertainty still justify optimizer transfer.",
                ),
                LeanRef(
                    label="Training-path composition with oracle measurement",
                    theorem="training_path_bundle_epsilon_optimal_with_oracleMeasurement",
                    file="lean3/FormalProofs/OPT/TrainingPipeline.lean",
                    note="Multi-stage training gaps compose additively with oracle measurement error.",
                ),
                LeanRef(
                    label="TreeIPW joint interval validity",
                    theorem="computeDSLBound_valid_from_joint_interval_event_with_oracleMeasurement_export",
                    file="lean3/FormalProofs/DSL/MainTheorems.lean",
                    note="Clustered intervals, calibration, and oracle measurement feed into the final DSL certificate.",
                ),
                LeanRef(
                    label="Oracle measurement decomposition",
                    theorem="expected_feature_utility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz",
                    file="lean3/FormalProofs/OPT/README.lean",
                    note="Approximate theorem-backed transport decomposes into transport error plus measurement error.",
                ),
            ],
            canonical_suites=["cpu_megasweep", "publication_clean", "publication_ctreepo_progress", "learnability", "dtm_lda"],
            caveats=[
                "These are audited approximate simulations, not exact theorem-backed neural operators.",
                "Partial suite status should be taken seriously when reading publication-progress artifacts.",
            ],
        ),
        "mergeable_ablation": FamilyTheorySpec(
            family="mergeable_ablation",
            title="Mergeable Ablation",
            simulation_claim=(
                "Strictly mergeable controls should preserve the target when the sufficient statistic is right and degrade cleanly when chunk quality or retained state is wrong."
            ),
            why_it_demonstrates_the_method=(
                "This family is the commutative sanity check for the broader theory: it shows the exact sketch ceiling and what is lost when one keeps the wrong state."
            ),
            key_assumptions=[
                "The one-pass reference lane is the exact mergeable ceiling.",
                "Budgeted/fixed-grid lanes demonstrate state-retention loss rather than theorem failure of the exact control.",
            ],
            exact_lanes=["one_pass_reference", "perfect_token_leaves_all"],
            approx_lanes=["grid_fixed_s1_b6", "grid_fixed_s2_b6"],
            proxy_lanes=["grid_fixed_s1_b1", "grid_fixed_s2_b1"],
            lean_refs=[
                LeanRef(
                    label="Broad theorem-backed interface",
                    theorem="ExactTheoremBacked.ofLocalLaws",
                    file="lean3/FormalProofs/OPT/README.lean",
                    note="Exact local laws are the broadest sufficient interface for theorem-backed reductions.",
                ),
                LeanRef(
                    label="Sketch route to theorem-backedness",
                    theorem="SketchCodecExactAssumptions.toExactTheoremBacked",
                    file="lean3/FormalProofs/OPT/README.lean",
                    note="A supplied exact sketch/codec witness induces exact theorem-backedness.",
                ),
                LeanRef(
                    label="Sketches as classical mergeable summaries",
                    theorem="sketchCodecExactAssumptions_imply_classical_mergeable",
                    file="lean3/FormalProofs/OPT/README.lean",
                    note="Exact sketch/codec assumptions reduce to the classical mergeable-summary interface.",
                ),
            ],
            canonical_suites=["simulation_buildout", "cpu_megasweep"],
            caveats=[
                "These simulations demonstrate the mergeable-sketch subcase, not the full noncommutative OPS story.",
            ],
        ),
        "local_law_learnability": FamilyTheorySpec(
            family="local_law_learnability",
            title="Unified Local-Law Learnability",
            simulation_claim=(
                "Held-out local-law scores and downstream gains can be compared under a common protocol across DGP families."
            ),
            why_it_demonstrates_the_method=(
                "This is the empirical supervision layer for the theory: it checks whether pushing on C1/C2/C3-style objectives actually improves downstream targets."
            ),
            key_assumptions=[
                "Exact local laws imply theorem-backedness; approximate local laws imply approximate theorem-backedness.",
                "Selection must be read against held-out configured objectives, not arbitrary train-side proxies.",
            ],
            exact_lanes=["oracle_g"],
            approx_lanes=["learned_g", "candidate_g"],
            proxy_lanes=["baseline_g", "root_only", "naive controls"],
            lean_refs=[
                LeanRef(
                    label="Exact theorem-backed from local laws",
                    theorem="ExactTheoremBacked.ofLocalLaws",
                    file="lean3/FormalProofs/OPT/README.lean",
                    note="A full exact local-law bundle is enough for theorem-backed correctness.",
                ),
                LeanRef(
                    label="Approx theorem-backed from audited local laws",
                    theorem="ApproxTheoremBacked.ofApproxLocalLaws",
                    file="lean3/FormalProofs/OPT/README.lean",
                    note="An audited approximate local-law bundle induces approximate theorem-backedness.",
                ),
                LeanRef(
                    label="Event-level DSL validity",
                    theorem="computeDSLBound_valid_from_joint_interval_event",
                    file="lean3/FormalProofs/DSL/MainTheorems.lean",
                    note="Held-out law estimates and calibration events can feed a valid DSL certificate.",
                ),
            ],
            canonical_suites=["learnability", "simulation_buildout", "markov_observed_token"],
            caveats=[
                "Learnability reports are evidence about optimization and generalization, not by themselves proofs of theorem-backedness for a learned operator.",
            ],
        ),
    }


def _suite_specs() -> Dict[str, SuiteTheorySpec]:
    return {
        target.theory_alignment_key: SuiteTheorySpec(
            name=target.theory_alignment_key,
            title=target.theory_title,
            families=list(target.theory_families),
            role=target.theory_role,
            demonstrates=target.theory_demonstrates,
        )
        for target in iter_canonical_suite_targets(bundle_roles=("paper", "appendix", "diagnostic"))
    }


def _load_bundle_manifest(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _status_priority(status: str) -> int:
    if status == "fail":
        return 0
    if status == "warn":
        return 1
    if status == "partial":
        return 2
    if status == "completed":
        return 3
    if status == "pass":
        return 4
    return 5


def _alignment_priority(status: str) -> int:
    if status == "misaligned":
        return 0
    if status == "incomplete":
        return 1
    if status == "provisionally_aligned":
        return 2
    if status == "mostly_aligned":
        return 3
    if status == "aligned":
        return 4
    return 5


def _bundle_results_by_name(bundle_manifest: Mapping[str, Any]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for result in bundle_manifest.get("results", []) or []:
        if not isinstance(result, Mapping):
            continue
        name = str(result.get("name", "")).strip()
        if name:
            out[name] = dict(result)
    return out


def _counts_for_family(report: ExpectationReport, family: str) -> Dict[str, int]:
    counts = {"pass": 0, "warn": 0, "fail": 0, "not_applicable": 0}
    for finding in report.expectations:
        if str(finding.family) != str(family):
            continue
        status = str(finding.status)
        counts[status] = int(counts.get(status, 0)) + 1
    return counts


def _overall_family_status(*, counts: Mapping[str, int], suite_statuses: Sequence[str]) -> tuple[str, str]:
    total_expectations = sum(int(v) for v in counts.values())
    n_fail = int(counts.get("fail", 0))
    n_warn = int(counts.get("warn", 0))
    if n_fail > 0:
        return "misaligned", "At least one qualitative expectation check currently fails."
    if total_expectations == 0:
        return "incomplete", "No aligned expectation evidence was found for this family in the scanned artifacts."
    if any(str(s) in {"failed", "pending"} for s in suite_statuses):
        return "incomplete", "At least one canonical suite is still pending or failed."
    if any(str(s) == "partial" for s in suite_statuses):
        return "provisionally_aligned", "The qualitative checks pass so far, but at least one canonical suite is still partial."
    if n_warn > 0:
        return "mostly_aligned", "No hard failures, but some expectation checks remain warning-level."
    return "aligned", "Canonical suites and qualitative checks are currently aligned with the intended theorem story."


def _suite_rows(
    suite_specs: Mapping[str, SuiteTheorySpec],
    bundle_results: Mapping[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name, spec in suite_specs.items():
        result = dict(bundle_results.get(name, {}) or {})
        rows.append(
            {
                "name": name,
                "title": spec.title,
                "families": list(spec.families),
                "role": spec.role,
                "demonstrates": spec.demonstrates,
                "status": str(result.get("status", "missing")),
                "root": result.get("root"),
                "json_count": result.get("json_count"),
                "bundle_role": result.get("bundle_role"),
                "primary_outputs": sorted((result.get("outputs", {}) or {}).keys()) if isinstance(result.get("outputs"), Mapping) else [],
            }
        )
    rows.sort(key=lambda x: (_status_priority(str(x.get("status", ""))), str(x.get("title", ""))))
    return rows


def build_simulation_theory_alignment_report(
    *,
    formal_root: Optional[Path] = None,
    expectation_report: Optional[ExpectationReport] = None,
    expectation_json_path: Optional[Path] = None,
    bundle_manifest_path: Optional[Path] = None,
) -> SimulationTheoryAlignmentReport:
    if expectation_report is None:
        if expectation_json_path is not None and expectation_json_path.exists():
            expectation_report = ExpectationReport.from_dict(
                json.loads(expectation_json_path.read_text(encoding="utf-8"))
            )
        elif formal_root is not None:
            expectation_report = build_expectation_report(output_root=formal_root.resolve())
        else:
            raise ValueError("expected one of expectation_report, expectation_json_path, or formal_root")

    bundle_manifest = _load_bundle_manifest(bundle_manifest_path)
    bundle_results = _bundle_results_by_name(bundle_manifest)
    family_specs = _family_specs()
    suite_specs = _suite_specs()

    family_statuses: List[FamilyAlignmentStatus] = []
    all_families = sorted(
        {
            *family_specs.keys(),
            *[str(x) for x in expectation_report.families_scanned],
        }
    )
    for family in all_families:
        spec = family_specs.get(str(family))
        if spec is None:
            continue
        relevant_suites: List[Dict[str, object]] = []
        suite_statuses: List[str] = []
        for suite_name in spec.canonical_suites:
            suite_spec = suite_specs.get(suite_name)
            result = dict(bundle_results.get(suite_name, {}) or {})
            status = str(result.get("status", "missing"))
            suite_statuses.append(status)
            relevant_suites.append(
                {
                    "name": suite_name,
                    "title": suite_spec.title if suite_spec is not None else suite_name,
                    "status": status,
                    "root": result.get("root"),
                    "bundle_role": result.get("bundle_role"),
                    "demonstrates": suite_spec.demonstrates if suite_spec is not None else "",
                }
            )
        counts = _counts_for_family(expectation_report, str(family))
        overall_status, note = _overall_family_status(counts=counts, suite_statuses=suite_statuses)
        family_statuses.append(
            FamilyAlignmentStatus(
                family=str(family),
                title=spec.title,
                expectation_counts=counts,
                relevant_suites=relevant_suites,
                overall_status=overall_status,
                note=note,
                spec=spec,
            )
        )

    family_statuses.sort(key=lambda x: (_alignment_priority(x.overall_status), x.title))
    suite_rows = _suite_rows(suite_specs, bundle_results)
    summary = {
        "n_families": int(len(family_statuses)),
        "aligned_families": int(sum(1 for x in family_statuses if x.overall_status == "aligned")),
        "mostly_aligned_families": int(sum(1 for x in family_statuses if x.overall_status == "mostly_aligned")),
        "provisionally_aligned_families": int(sum(1 for x in family_statuses if x.overall_status == "provisionally_aligned")),
        "misaligned_families": int(sum(1 for x in family_statuses if x.overall_status == "misaligned")),
        "incomplete_families": int(sum(1 for x in family_statuses if x.overall_status == "incomplete")),
        "expectation_summary": dict(expectation_report.summary),
    }
    return SimulationTheoryAlignmentReport(
        formal_root=str(formal_root.resolve()) if formal_root is not None else expectation_report.input_root,
        expectation_source=(
            str(expectation_json_path.resolve())
            if expectation_json_path is not None and expectation_json_path.exists()
            else expectation_report.input_root
        ),
        bundle_manifest=str(bundle_manifest_path.resolve()) if bundle_manifest_path is not None and bundle_manifest_path.exists() else None,
        family_statuses=family_statuses,
        suites=suite_rows,
        summary=summary,
    )


def render_simulation_theory_alignment_markdown(report: SimulationTheoryAlignmentReport) -> str:
    lines: List[str] = []
    lines.append("# Simulation Theory Alignment")
    lines.append("")
    if report.formal_root is not None:
        lines.append(f"- Formal root: `{report.formal_root}`")
    if report.expectation_source is not None:
        lines.append(f"- Expectation source: `{report.expectation_source}`")
    if report.bundle_manifest is not None:
        lines.append(f"- Bundle manifest: `{report.bundle_manifest}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Families aligned: `{report.summary.get('aligned_families', 0)}`")
    lines.append(f"- Families mostly aligned: `{report.summary.get('mostly_aligned_families', 0)}`")
    lines.append(f"- Families provisionally aligned: `{report.summary.get('provisionally_aligned_families', 0)}`")
    lines.append(f"- Families incomplete: `{report.summary.get('incomplete_families', 0)}`")
    lines.append(f"- Families misaligned: `{report.summary.get('misaligned_families', 0)}`")
    lines.append("")
    lines.append("## Canonical Suites")
    lines.append("")
    lines.append("| Suite | Status | Families | Role |")
    lines.append("| --- | --- | --- | --- |")
    for suite in report.suites:
        families = ", ".join(str(x) for x in (suite.get("families", []) or []))
        lines.append(
            f"| {suite.get('title', suite.get('name', ''))} | `{suite.get('status', 'missing')}` | `{families}` | {suite.get('role', '')} |"
        )
    lines.append("")
    for family_status in report.family_statuses:
        spec = family_status.spec
        lines.append(f"## {family_status.title}")
        lines.append("")
        lines.append(f"- Alignment status: `{family_status.overall_status}`")
        lines.append(f"- Simulation claim: {spec.simulation_claim}")
        lines.append(f"- Why it demonstrates the method: {spec.why_it_demonstrates_the_method}")
        lines.append(f"- Expectation checks: pass=`{family_status.expectation_counts.get('pass', 0)}`, warn=`{family_status.expectation_counts.get('warn', 0)}`, fail=`{family_status.expectation_counts.get('fail', 0)}`, n/a=`{family_status.expectation_counts.get('not_applicable', 0)}`")
        lines.append(f"- Current read: {family_status.note}")
        if spec.key_assumptions:
            lines.append("- Key assumptions:")
            for item in spec.key_assumptions:
                lines.append(f"  - {item}")
        if spec.exact_lanes:
            lines.append(f"- Exact/theorem-backed control lanes: `{', '.join(spec.exact_lanes)}`")
        if spec.approx_lanes:
            lines.append(f"- Approximate/audited lanes: `{', '.join(spec.approx_lanes)}`")
        if spec.proxy_lanes:
            lines.append(f"- Proxy/wrong-control lanes: `{', '.join(spec.proxy_lanes)}`")
        if spec.lean_refs:
            lines.append("- Lean anchors:")
            for ref in spec.lean_refs:
                lines.append(f"  - `{ref.theorem}` in `{ref.file}`: {ref.note}")
        if family_status.relevant_suites:
            lines.append("- Canonical suites:")
            for suite in family_status.relevant_suites:
                lines.append(
                    f"  - `{suite.get('title', suite.get('name', ''))}`: `{suite.get('status', 'missing')}`. {suite.get('demonstrates', '')}"
                )
        if spec.caveats:
            lines.append("- Caveats:")
            for item in spec.caveats:
                lines.append(f"  - {item}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_simulation_theory_alignment_report(
    report: SimulationTheoryAlignmentReport,
    *,
    output_json: Optional[Path],
    output_markdown: Optional[Path],
) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"output_json": None, "output_markdown": None}
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        out["output_json"] = str(output_json.resolve())
    if output_markdown is not None:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(render_simulation_theory_alignment_markdown(report), encoding="utf-8")
        out["output_markdown"] = str(output_markdown.resolve())
    return out


__all__ = [
    "FamilyAlignmentStatus",
    "FamilyTheorySpec",
    "LeanRef",
    "SimulationTheoryAlignmentReport",
    "SuiteTheorySpec",
    "build_simulation_theory_alignment_report",
    "render_simulation_theory_alignment_markdown",
    "write_simulation_theory_alignment_report",
]
