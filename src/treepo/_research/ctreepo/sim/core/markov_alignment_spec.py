from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from treepo._research.core.runtime_capabilities import markov_family_runtime_capability


REPO_ROOT = Path(__file__).resolve().parents[4]
MARKOV_ALIGNMENT_AUDIT_NOTE_PATH = REPO_ROOT / "docs" / "markov_alignment_audit.md"
PAPER_TO_LEAN_LOCAL_LAW_MAPPING = {
    "C1": "L1",
    "C2": "L3",
    "C3": "L2",
}
MARKOV_FULL_DOC_OBJECTIVE_SURFACE = "markov_full_doc_normalized_task_local_law_surface"
TREEPO_REGULARIZED_OBJECTIVE_SURFACE = "treepo_regularized_objective"


def _surface_runtime_capability(surface_name: str) -> Dict[str, Any]:
    if surface_name == "markov_observed_token":
        return markov_family_runtime_capability(
            family_name=surface_name,
            exact_family="exact",
            notes=(
                "Observed-token Markov runs expose theorem anchors plus approximate empirical chat lanes.",
            ),
        ).to_dict()
    if surface_name == "markov_full_tree_ipw_grid":
        return markov_family_runtime_capability(
            family_name=surface_name,
            notes=(
                "This estimator grid is an evaluation/audit surface rather than a symbolic execution lane.",
            ),
        ).to_dict()
    return markov_family_runtime_capability(
        family_name=surface_name,
        notes=(
            "This surface is empirical/provenance-facing even when theorem anchors are reported alongside it.",
        ),
    ).to_dict()


def markov_alignment_spec() -> Dict[str, Any]:
    return {
        "paper_to_lean_local_law_mapping": dict(PAPER_TO_LEAN_LOCAL_LAW_MAPPING),
        "objective_surface_distinctions": [
            {
                "surface_name": MARKOV_FULL_DOC_OBJECTIVE_SURFACE,
                "distinct_from": TREEPO_REGULARIZED_OBJECTIVE_SURFACE,
                "note": (
                    "The Markov full-doc normalized task/local-law weighting surface is "
                    "an empirical full-doc objective surface. It is not the same "
                    "objective family as the separate TreePO Regularized Objective note."
                ),
            }
        ],
        "surfaces": [
            {
                "name": "markov_observed_token",
                "learning_target_or_estimand": "document root-count prediction under fixed observed-token bundles",
                "supervision_channels": [
                    "document root supervision",
                    "sampled leaf local-law labels",
                    "sampled internal merge-law labels",
                ],
                "objective_or_estimator_semantics": (
                    "Observed-token Markov task/local-law weighting surface with theorem-facing "
                    "C1/L1, C2/L3, C3/L2 terms and proxy schedule diagnostics kept separate."
                ),
                "status": "approximate_audited_plus_theorem_anchors",
                "lean_references": [
                    "markov_path_local_laws_of_encoded_state",
                    "markov_path_state_exact_on_tree",
                    "markov_path_count_exact_on_tree",
                ],
                "paper_notation_mapping": dict(PAPER_TO_LEAN_LOCAL_LAW_MAPPING),
                "runtime_capabilities": _surface_runtime_capability("markov_observed_token"),
            },
            {
                "name": "markov_full_doc_anchor_diagnostics",
                "learning_target_or_estimand": "paper-facing full-document root-count learning under fixed recoverable bundles",
                "supervision_channels": [
                    "full-document root supervision",
                    "optional tree-local C1/C2/C3 supervision for tree baselines",
                    "optional budgeted local supervision via explicit manifest",
                ],
                "objective_or_estimator_semantics": (
                    "Full-doc empirical objective surface. Theorem-facing totals include only "
                    "C1/L1, C2/L3, C3/L2. Schedule spread/consistency remains proxy-only."
                ),
                "status": "approximate_audited_or_proxy_only",
                "lean_references": [
                    "ApproxTheoremBacked.ofApproxLocalLaws",
                    "markov_path_local_laws_of_encoded_state",
                ],
                "paper_notation_mapping": dict(PAPER_TO_LEAN_LOCAL_LAW_MAPPING),
                "config_tagged_slices": [
                    "fair_fno_v1 parity",
                    "capacity tuning configs fair_fno_v1_w{width}_m{modes}_l{layers}",
                    "oracle budget share frontier",
                ],
                "runtime_capabilities": _surface_runtime_capability("markov_full_doc_anchor_diagnostics"),
            },
            {
                "name": "markov_full_doc_anchor_ladder",
                "learning_target_or_estimand": "reference/reproduction provenance alignment for the full-doc doc-sequence baseline",
                "supervision_channels": ["full-document document-sequence supervision"],
                "objective_or_estimator_semantics": (
                    "Provenance and matched-bundle reproduction surface; not a theorem witness."
                ),
                "status": "proxy_only",
                "lean_references": [],
                "paper_notation_mapping": {},
                "runtime_capabilities": _surface_runtime_capability("markov_full_doc_anchor_ladder"),
            },
            {
                "name": "markov_full_tree_ipw_grid",
                "learning_target_or_estimand": "realized_full_tree_node_mean_loss under Bernoulli realized-node sampling",
                "supervision_channels": [
                    "always-observed document-top loss",
                    "sampled realized tree-node losses with unit propensities",
                ],
                "objective_or_estimator_semantics": (
                    "Point-estimation diagnostic surface for naive, HT, and Hajek estimators "
                    "over realized tree nodes. This surface is not an honest-CI claim unless "
                    "a separate confidence wrapper is attached."
                ),
                "status": "approximate_audited_estimand_diagnostic",
                "lean_references": [
                    "ApproxTheoremBacked.ofApproxLocalLaws",
                    "computeDSLBound_valid_from_joint_interval_event",
                ],
                "paper_notation_mapping": dict(PAPER_TO_LEAN_LOCAL_LAW_MAPPING),
                "runtime_capabilities": _surface_runtime_capability("markov_full_tree_ipw_grid"),
            },
        ],
        "audit_note_path": str(MARKOV_ALIGNMENT_AUDIT_NOTE_PATH),
    }


def required_audit_note_phrases() -> List[str]:
    return [
        "paper C1 = Lean L1",
        "paper C2 = Lean L3",
        "paper C3 = Lean L2",
        "not the same objective family as the TreePO Regularized Objective",
        MARKOV_FULL_DOC_OBJECTIVE_SURFACE,
        TREEPO_REGULARIZED_OBJECTIVE_SURFACE,
    ]


__all__ = [
    "MARKOV_ALIGNMENT_AUDIT_NOTE_PATH",
    "MARKOV_FULL_DOC_OBJECTIVE_SURFACE",
    "PAPER_TO_LEAN_LOCAL_LAW_MAPPING",
    "TREEPO_REGULARIZED_OBJECTIVE_SURFACE",
    "markov_alignment_spec",
    "required_audit_note_phrases",
]
