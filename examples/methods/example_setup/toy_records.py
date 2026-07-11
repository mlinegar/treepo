"""Toy records for task-neutral example walkthroughs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def toy_finetune_preferences() -> Any:
    from treepo import Candidate, PreferenceDataset, PreferenceRecord, TaskState

    positive_state = TaskState(
        kind="policy_signal",
        counts={"positive": 2.0},
        measures={"score": 0.8},
        text="specific positive evidence",
        metadata={"source": "example"},
    )
    return PreferenceDataset(
        [
            PreferenceRecord(
                record_id="root_supervised",
                unit_id="doc1:root",
                unit_type="root",
                target="f",
                context="Estimate the document-level score.",
                tree_id="doc1",
                doc_id="doc1",
                node_id="root",
                level=1,
                position=0,
                left_child_id="leaf0",
                right_child_id="leaf1",
                candidates=(Candidate(id="gold", value="score: 0.8", score=1.0, preferred=True),),
            ),
            PreferenceRecord(
                record_id="leaf_supervised",
                unit_id="doc1:leaf0",
                unit_type="leaf",
                target="g",
                context="Encode leaf0 as a composable state.",
                tree_id="doc1",
                doc_id="doc1",
                node_id="leaf0",
                level=0,
                position=0,
                parent_id="root",
                candidates=(Candidate(id="gold", value=positive_state, score=1.0, preferred=True),),
            ),
            PreferenceRecord(
                record_id="leaf_pair",
                unit_id="doc1:leaf0:pair",
                unit_type="leaf",
                target="g",
                context="Choose the better state for leaf0.",
                tree_id="doc1",
                doc_id="doc1",
                node_id="leaf0",
                level=0,
                position=0,
                parent_id="root",
                candidates=(
                    Candidate(id="specific", value=positive_state, score=0.95, preferred=True),
                    Candidate(id="generic", value="generic campaign language", score=0.2),
                ),
                weight=2.0,
                propensity=0.5,
            ),
            PreferenceRecord(
                record_id="ranked_merge",
                unit_id="doc1:root:ranked",
                unit_type="merge",
                target="g",
                context="Rank candidate merged states.",
                tree_id="doc1",
                doc_id="doc1",
                node_id="root",
                level=1,
                position=0,
                left_child_id="leaf0",
                right_child_id="leaf1",
                candidates=(
                    Candidate(id="best", value={"score": 0.8, "evidence": "specific"}, score=0.9, rank=1),
                    Candidate(id="tie_a", value={"score": 0.7, "evidence": "partial a"}, score=0.7, rank=2),
                    Candidate(id="tie_b", value={"score": 0.7, "evidence": "partial b"}, score=0.7, rank=2),
                ),
            ),
        ]
    )


def toy_optimizer_trees_and_preferences() -> tuple[list[Any], Any]:
    from treepo import (
        Candidate,
        PreferenceDataset,
        PreferenceRecord,
        TaskState,
        TreeNode,
        TreeRecord,
    )

    train_trees = [
        TreeRecord(
            tree_id="doc_a",
            doc_id="doc_a",
            text="Leaf A supports a positive root. Leaf B is neutral.",
            root_label=0.7,
            nodes=(
                TreeNode(
                    node_id="leaf_a0",
                    unit_type="leaf",
                    text="positive evidence",
                    level=0,
                    position=0,
                    parent_id="root",
                    label=0.8,
                ),
                TreeNode(
                    node_id="leaf_a1",
                    unit_type="leaf",
                    text="neutral evidence",
                    level=0,
                    position=1,
                    parent_id="root",
                    label=0.0,
                ),
                TreeNode(
                    node_id="root",
                    unit_type="root",
                    text="positive plus neutral",
                    level=1,
                    position=0,
                    left_child_id="leaf_a0",
                    right_child_id="leaf_a1",
                    label=0.7,
                ),
            ),
        ),
        TreeRecord(
            tree_id="doc_b",
            doc_id="doc_b",
            text="Leaf C supports a negative root. Leaf D is neutral.",
            root_label=-0.5,
            nodes=(
                TreeNode(
                    node_id="leaf_b0",
                    unit_type="leaf",
                    text="negative evidence",
                    level=0,
                    position=0,
                    parent_id="root",
                    label=-0.6,
                ),
                TreeNode(
                    node_id="leaf_b1",
                    unit_type="leaf",
                    text="neutral evidence",
                    level=0,
                    position=1,
                    parent_id="root",
                    label=0.0,
                ),
                TreeNode(
                    node_id="root",
                    unit_type="root",
                    text="negative plus neutral",
                    level=1,
                    position=0,
                    left_child_id="leaf_b0",
                    right_child_id="leaf_b1",
                    label=-0.5,
                ),
            ),
        ),
    ]

    positive_state = TaskState(kind="toy_signal", counts={"positive": 1.0}, measures={"score": 0.8}, text="positive evidence")
    negative_state = TaskState(kind="toy_signal", counts={"negative": 1.0}, measures={"score": -0.6}, text="negative evidence")
    preferences = PreferenceDataset(
        [
            PreferenceRecord(
                record_id="root_doc_a",
                unit_id="doc_a:root",
                unit_type="root",
                target="f",
                context="Score document A.",
                tree_id="doc_a",
                doc_id="doc_a",
                node_id="root",
                level=1,
                candidates=(Candidate(id="gold", value=0.7, score=1.0, preferred=True),),
            ),
            PreferenceRecord(
                record_id="root_doc_b",
                unit_id="doc_b:root",
                unit_type="root",
                target="f",
                context="Score document B.",
                tree_id="doc_b",
                doc_id="doc_b",
                node_id="root",
                level=1,
                candidates=(Candidate(id="gold", value=-0.5, score=1.0, preferred=True),),
            ),
            PreferenceRecord(
                record_id="leaf_doc_a",
                unit_id="doc_a:leaf_a0",
                unit_type="leaf",
                target="g",
                context="Encode leaf A0 as a composable state.",
                tree_id="doc_a",
                doc_id="doc_a",
                node_id="leaf_a0",
                level=0,
                parent_id="root",
                candidates=(Candidate(id="gold", value=positive_state, score=1.0, preferred=True),),
            ),
            PreferenceRecord(
                record_id="leaf_doc_b",
                unit_id="doc_b:leaf_b0",
                unit_type="leaf",
                target="g",
                context="Encode leaf B0 as a composable state.",
                tree_id="doc_b",
                doc_id="doc_b",
                node_id="leaf_b0",
                level=0,
                parent_id="root",
                candidates=(Candidate(id="gold", value=negative_state, score=1.0, preferred=True),),
            ),
            PreferenceRecord(
                record_id="pair_leaf_a0",
                unit_id="doc_a:leaf_a0:pair",
                unit_type="leaf",
                target="g",
                context="Choose the better state for leaf A0.",
                tree_id="doc_a",
                doc_id="doc_a",
                node_id="leaf_a0",
                level=0,
                parent_id="root",
                candidates=(
                    Candidate(id="chosen", value=positive_state, score=0.9, preferred=True),
                    Candidate(id="rejected", value="generic positive text", score=0.2),
                ),
                propensity=0.5,
            ),
            PreferenceRecord(
                record_id="rank_merge_a",
                unit_id="doc_a:root:ranked",
                unit_type="merge",
                target="g",
                context="Rank candidate merged states for document A.",
                tree_id="doc_a",
                doc_id="doc_a",
                node_id="root",
                level=1,
                candidates=(
                    Candidate(id="best", value={"score": 0.7, "evidence": "specific"}, score=0.95, rank=1),
                    Candidate(id="ok", value={"score": 0.5, "evidence": "partial"}, score=0.65, rank=2),
                    Candidate(id="weak", value={"score": 0.0, "evidence": "generic"}, score=0.1, rank=3),
                ),
            ),
        ]
    )
    return train_trees, preferences


def toy_dspy_fit_config(*, output_dir: Path, train_trees: Sequence[Any], preferences: Any) -> dict[str, Any]:
    return {
        "family": "dspy",
        "train_data": train_trees,
        "eval_data": train_trees,
        "preference_data": preferences,
        "backend_config": {
            "output_dir": str(output_dir / "fit"),
            "model": "local-dspy-program",
            "predict_fn": toy_prompt_predict,
            "prompt_template": (
                "Estimate the root score from this document.\n\n"
                "Document:\n{text}\n\n"
                "f examples:\n{f_supervised_examples}\n\n"
                "g examples:\n{g_supervised_examples}\n\n"
                "Return only one number."
            ),
            "min_score": -1.0,
            "max_score": 1.0,
        },
        "axis": {"max_iterations": 2, "axis_value": 2, "axis_kind": "leaf_count"},
    }


def toy_prompt_predict(*, prompt: str, tree: Any, messages: list[dict[str, Any]], config: Any) -> dict[str, float]:
    del prompt, messages, config
    label = getattr(tree, "root_label", None)
    if label is None:
        label = getattr(tree, "document_score", 0.0)
    return {"score": float(label)}


def toy_local_law_rows() -> Sequence[Any]:
    from treepo.local_law import LawKind, LocalLawAuditRow

    return (
        LocalLawAuditRow(
            row_id="doc-1:leaf-0:c1",
            law_kind=LawKind.C1_LEAF,
            proxy_loss=0.06,
            oracle_loss=0.04,
            observed=True,
            propensity=0.5,
            node_weight=1.0,
            depth=2,
            metadata={"tree_id": "doc-1", "node_id": "leaf-0"},
        ),
        LocalLawAuditRow(
            row_id="doc-1:leaf-1:c1",
            law_kind=LawKind.C1_LEAF,
            proxy_loss=0.08,
            observed=False,
            propensity=0.5,
            node_weight=1.0,
            depth=2,
            metadata={"tree_id": "doc-1", "node_id": "leaf-1"},
        ),
        LocalLawAuditRow(
            row_id="doc-1:leaf-2:c1",
            law_kind=LawKind.C1_LEAF,
            proxy_loss=0.02,
            oracle_loss=0.03,
            observed=True,
            propensity=0.5,
            node_weight=1.0,
            depth=2,
            metadata={"tree_id": "doc-1", "node_id": "leaf-2"},
        ),
        LocalLawAuditRow(
            row_id="doc-1:leaf-0:c2",
            law_kind=LawKind.C2_IDEMPOTENCE,
            proxy_loss=0.01,
            oracle_loss=0.01,
            observed=True,
            propensity=1.0,
            node_weight=0.5,
            depth=2,
            metadata={"tree_id": "doc-1", "node_id": "leaf-0"},
        ),
        LocalLawAuditRow(
            row_id="doc-1:internal-0:c3",
            law_kind=LawKind.C3_MERGE,
            proxy_loss=0.10,
            oracle_loss=0.12,
            observed=True,
            propensity=0.75,
            node_weight=2.0,
            depth=1,
            metadata={"tree_id": "doc-1", "node_id": "internal-0"},
        ),
        LocalLawAuditRow(
            row_id="doc-1:root:c3",
            law_kind=LawKind.C3_MERGE,
            proxy_loss=0.04,
            observed=False,
            propensity=0.75,
            node_weight=2.5,
            depth=0,
            metadata={"tree_id": "doc-1", "node_id": "root"},
        ),
    )


def write_local_law_rows(path: Path, rows: Sequence[Any]) -> None:
    path.write_text(
        "".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def toy_certificate_preferences() -> Any:
    from treepo import Candidate, PreferenceDataset, PreferenceRecord, TaskState

    dataset = PreferenceDataset()
    dataset.append(
        PreferenceRecord(
            record_id="doc-1:root:score",
            unit_id="doc-1:root",
            unit_type="root",
            target="f",
            context={"prompt": "Predict the document-level policy count."},
            candidates=(
                Candidate(id="teacher_root", value=3.0, score=1.0, preferred=True),
                Candidate(id="under_count", value=2.0, score=0.5),
            ),
            tree_id="doc-1",
            node_id="root",
            level=0,
            metadata={"evidence_tier": "root_observed"},
        )
    )
    dataset.append(
        PreferenceRecord(
            record_id="doc-1:leaf-0:state",
            unit_id="doc-1:leaf-0",
            unit_type="leaf",
            target="g",
            context={"prompt": "Encode local policy mentions for this leaf."},
            candidates=(
                Candidate(
                    id="teacher_leaf_state",
                    value=TaskState(
                        kind="toy_policy_state",
                        counts={"policy_mentions": 1.0},
                        metadata={"source": "sampled_oracle"},
                    ),
                    score=1.0,
                    preferred=True,
                ),
                Candidate(
                    id="empty_state",
                    value=TaskState(kind="toy_policy_state", counts={"policy_mentions": 0.0}),
                    score=0.0,
                ),
            ),
            weight=1.0,
            propensity=0.5,
            tree_id="doc-1",
            node_id="leaf-0",
            level=2,
            metadata={"evidence_tier": "sampled_node", "law_type": "c1_leaf"},
        )
    )
    return dataset


def toy_statistic_artifact(*, audit: dict[str, Any], rows_path: Path, audit_dir: Path, row_count: int) -> dict[str, Any]:
    from treepo.statistic import StatisticInfo

    return {
        "info": StatisticInfo(
            name="toy_policy_count",
            state_kind="toy_policy_state",
            exact=False,
            supports_local_laws=True,
            metadata={"example": "sampled_local_law_certificate"},
        ).to_dict(),
        "local_law_summary": audit["local_law_objective"],
        "local_law_row_count": int(row_count),
        "files": {"rows": str(rows_path), "audit_summary": str(audit_dir / "audit_summary.json")},
    }


def toy_root_metrics() -> dict[str, float]:
    return {
        "n": 1.0,
        "internal_f_mae": 0.18,
        "mean_prediction": 2.82,
        "mean_teacher": 3.0,
    }


def toy_error_certificate(*, root_metrics: dict[str, float], audit: dict[str, Any]) -> Any:
    from treepo.certificate import CommonMechanismEnvelopeEvidence
    from treepo.local_law import build_triangle_local_law_error_certificate

    observed_root_radius = abs(float(root_metrics["mean_prediction"]) - float(root_metrics["mean_teacher"]))
    return build_triangle_local_law_error_certificate(
        reported_estimate=float(root_metrics["internal_f_mae"]),
        audit=audit,
        root_down_radius=observed_root_radius,
        common_mechanism_envelopes=(
            CommonMechanismEnvelopeEvidence(
                observed_root_radius=observed_root_radius,
                amplification=1.0,
                slack=0.05,
                source="toy_root_error_bound",
                artifact_ids=("audit_summary",),
                metadata={"replace_with": "held-out root-error finite-sample bound"},
            ),
        ),
        confidence_delta=0.05,
        source="sampled_local_law_rows",
        artifact_ids=("sampled_local_law_rows", "audit_summary"),
        metadata={
            "example_only": True,
            "note": (
                "The ledger shape is real; the toy root and common-mechanism "
                "radii are illustrative."
            ),
        },
    )
