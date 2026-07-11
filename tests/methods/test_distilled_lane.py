"""Phase 2 of the fit-grid plan: the cached-teacher distillation lane.

``fit()`` with ``local_label_mix='llm_distilled'`` must train a student purely
from a ``teacher_node_rows.jsonl`` cache (or a callable predictor) with no LLM
client constructed: rows load, attach to the training trees by node id or char
span, feed the node-supervision loss through an exclusive metadata key, and
unmatched nodes stay unobserved instead of falling back to gold labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from treepo import fit
from treepo.distilled import (
    DISTILLED_NODE_KEY,
    attach_distilled_labels,
    attach_predictor_labels,
    load_teacher_node_rows,
)
from treepo.methods._fno_config import NeuralOperatorFamilyConfig
from treepo.methods._fno_targets import _node_supervision_targets
from treepo.tree import TreeNode, TreeRecord


def _labeled_tree(doc_id: str, *, offset: float = 0.0, split: str = "train") -> TreeRecord:
    """Four gold-labeled leaves, two merges, one root — mirrors the Phase 1 fixture."""

    leaf_scores = [0.1 + offset, 0.2 + offset, 0.3 + offset, 0.4 + offset]
    leaves = [
        TreeNode(
            node_id=f"{doc_id}_l{i}",
            unit_type="leaf",
            text=f"{doc_id} unit {i} alpha beta gamma delta",
            level=0,
            position=i,
            label=score,
            metadata={"char_start": i * 10, "char_end": i * 10 + 9},
        )
        for i, score in enumerate(leaf_scores)
    ]
    merges = [
        TreeNode(
            node_id=f"{doc_id}_m0",
            unit_type="merge",
            level=1,
            position=0,
            left_child_id=f"{doc_id}_l0",
            right_child_id=f"{doc_id}_l1",
            label=0.15 + offset,
        ),
        TreeNode(
            node_id=f"{doc_id}_m1",
            unit_type="merge",
            level=1,
            position=1,
            left_child_id=f"{doc_id}_l2",
            right_child_id=f"{doc_id}_l3",
            label=0.35 + offset,
        ),
    ]
    root = TreeNode(
        node_id=f"{doc_id}_root",
        unit_type="root",
        level=2,
        position=0,
        left_child_id=f"{doc_id}_m0",
        right_child_id=f"{doc_id}_m1",
        label=0.25 + offset,
    )
    return TreeRecord(
        tree_id=doc_id,
        doc_id=doc_id,
        root_label=0.25 + offset,
        nodes=(*leaves, *merges, root),
        metadata={"split": split},
    )


def _trees(n: int = 4) -> list[TreeRecord]:
    return [_labeled_tree(f"doc_{i:02d}", offset=0.05 * i) for i in range(n)]


def _teacher_rows_path(tmp_path: Path, trees: list[TreeRecord], *, skip_docs: int = 0) -> Path:
    """Write teacher-grid (schema A) rows for every leaf of every tree."""

    path = tmp_path / "teacher_node_rows.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for tree in trees[skip_docs:]:
            for node in tree.nodes:
                if node.unit_type != "leaf":
                    continue
                row = {
                    "doc_id": tree.doc_id,
                    "node_id": node.node_id,
                    "level": node.level,
                    "char_start": node.metadata["char_start"],
                    "char_end": node.metadata["char_end"],
                    "dimension": "economic",
                    "score_1_7": 4.0 + float(node.position or 0),
                    "is_leaf": True,
                    "split": "train",
                }
                handle.write(json.dumps(row) + "\n")
    return path


def _fit_config(tmp_path: Path, trees, **spec_extra: object) -> dict[str, object]:
    return {
        "family": "neural_operator",
        "train_data": trees,
        "eval_data": trees,
        "axis": {"axis_kind": "leaf_count", "axis_value": 4},
        "backend_config": {
            "operator_kind": "conv1d",
            "embedding_dim": 8,
            "hidden_channels": 4,
            "n_layers": 1,
            "head_hidden_dim": 8,
            "epochs_per_iteration": 1,
            "batch_size": 4,
            "learning_rate": 0.01,
            "device": "cpu",
            "seed": 3,
            "output_dir": str(tmp_path / "fit"),
        },
        **spec_extra,
    }


# ------------------------------ loader ------------------------------ #


def test_loader_reads_teacher_grid_rows(tmp_path: Path) -> None:
    trees = _trees(2)
    path = _teacher_rows_path(tmp_path, trees)
    labels = load_teacher_node_rows(path)
    assert labels.score_key == "score_1_7"
    assert labels.n_rows == 8
    assert labels.lookup(doc_id="doc_00", node_id="doc_00_l1") == 5.0
    # Span key fallback resolves the same row.
    assert labels.lookup(doc_id="doc_00", level=0, char_start=10, char_end=19) == 5.0


def test_loader_reads_rile_grid_rows(tmp_path: Path) -> None:
    path = tmp_path / "teacher_node_rows.jsonl"
    rows = [
        {
            "doc_id": "92622",
            "node_id": "node_l0_00000",
            "level": 0,
            "char_start": 0,
            "char_end": 59,
            "rile_norm": 0.5,
            "dimension_scores_0_1": {"rile": 0.5, "domain_3": 0.25},
        }
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    by_default = load_teacher_node_rows(path)
    assert by_default.score_key == "rile_norm"
    assert by_default.lookup(doc_id="92622", node_id="node_l0_00000") == 0.5

    by_dimension = load_teacher_node_rows(path, dimension="domain_3")
    assert by_dimension.lookup(doc_id="92622", node_id="node_l0_00000") == 0.25


def test_loader_rejects_multi_dimension_files_without_selector(tmp_path: Path) -> None:
    path = tmp_path / "teacher_node_rows.jsonl"
    rows = [
        {"doc_id": "d", "node_id": "n0", "dimension": "economic", "score_1_7": 3.0},
        {"doc_id": "d", "node_id": "n1", "dimension": "social", "score_1_7": 5.0},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="multiple dimensions"):
        load_teacher_node_rows(path)
    labels = load_teacher_node_rows(path, dimension="economic")
    assert labels.lookup(doc_id="d", node_id="n0") == 3.0
    assert labels.lookup(doc_id="d", node_id="n1") is None


def test_loader_rejects_conflicting_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "teacher_node_rows.jsonl"
    rows = [
        {"doc_id": "d", "node_id": "n0", "score_1_7": 3.0},
        {"doc_id": "d", "node_id": "n0", "score_1_7": 6.0},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting teacher scores"):
        load_teacher_node_rows(path)


def test_loader_accepts_directory_containing_rows_file(tmp_path: Path) -> None:
    trees = _trees(1)
    _teacher_rows_path(tmp_path, trees)
    labels = load_teacher_node_rows(tmp_path)
    assert labels.n_rows == 4


# ------------------------------ attach ------------------------------ #


def test_attach_writes_exclusive_key_and_reports_counts(tmp_path: Path) -> None:
    trees = _trees(3)
    labels = load_teacher_node_rows(_teacher_rows_path(tmp_path, trees, skip_docs=1))
    report = attach_distilled_labels(trees, labels)
    # doc_00 has no teacher rows; its 7 nodes stay unlabeled. The other two
    # trees have leaf rows only, so merges/roots stay unlabeled as well.
    assert report["n_nodes"] == 21
    assert report["n_nodes_attached"] == 8
    assert report["n_matched_by_node_id"] == 8
    assert report["n_docs_with_unmatched_nodes"] == 3
    labeled = [n for t in trees for n in t.nodes if DISTILLED_NODE_KEY in n.metadata]
    assert len(labeled) == 8
    assert all(n.unit_type == "leaf" for n in labeled)
    # Gold labels are untouched.
    assert trees[1].nodes[0].label == pytest.approx(0.15)


def test_attach_falls_back_to_span_key(tmp_path: Path) -> None:
    trees = _trees(1)
    path = tmp_path / "teacher_node_rows.jsonl"
    row = {
        "doc_id": "doc_00",
        "node_id": "some_other_naming_scheme",
        "level": 0,
        "char_start": 0,
        "char_end": 9,
        "score_1_7": 2.0,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = attach_distilled_labels(trees, load_teacher_node_rows(path))
    assert report["n_matched_by_span"] == 1
    assert trees[0].nodes[0].metadata[DISTILLED_NODE_KEY] == 2.0


def test_attach_errors_when_nothing_matches(tmp_path: Path) -> None:
    trees = _trees(1)
    path = tmp_path / "teacher_node_rows.jsonl"
    row = {"doc_id": "unrelated_doc", "node_id": "n0", "score_1_7": 2.0}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no node .* matched"):
        attach_distilled_labels(trees, load_teacher_node_rows(path))


def test_attach_predictor_labels_scores_node_texts() -> None:
    trees = _trees(1)

    def scorer(text: str) -> float:
        return float(len(text) % 7)

    report = attach_predictor_labels(trees, scorer)
    # Only leaves carry text in the fixture.
    assert report["n_nodes_attached"] == 4
    assert report["labels_source"] == "node_oracle_predictor"


# --------------------- exclusive node-target reads --------------------- #


def test_exclusive_key_means_unmatched_nodes_stay_unobserved() -> None:
    trees = _trees(1)
    trees[0].nodes[0].metadata[DISTILLED_NODE_KEY] = 5.5
    config = NeuralOperatorFamilyConfig(
        node_target_key=DISTILLED_NODE_KEY,
        node_target_exclusive=True,
        leaf_weight=1.0,
        merge_weight=1.0,
    )
    supervision = _node_supervision_targets(trees, config, width=1)
    assert supervision is not None
    targets, observed = supervision[0]
    assert observed[0] is True and targets[0] == [5.5]
    # Every other node has gold labels but NO distilled key: unobserved.
    assert sum(observed) == 1


def test_exclusive_without_key_is_a_loud_error() -> None:
    trees = _trees(1)
    config = NeuralOperatorFamilyConfig(node_target_exclusive=True, leaf_weight=1.0)
    with pytest.raises(ValueError, match="node_target_exclusive"):
        _node_supervision_targets(trees, config, width=1)


# ------------------------------ fit() e2e ------------------------------ #


def test_fit_trains_from_cached_teacher_rows_only(tmp_path: Path) -> None:
    trees = _trees(4)
    rows_path = _teacher_rows_path(tmp_path, trees)
    result = fit(
        _fit_config(
            tmp_path,
            trees,
            local_label_mix="llm_distilled",
            distilled_labels_path=str(rows_path),
            supervision_level="node",
        )
    )
    assert result.status == "success"
    mix = result.summary["grid_axes"]["local_label_mix"]
    assert mix["labels_source"] == "cached_jsonl"
    assert mix["n_nodes_attached"] == 16
    assert mix["n_teacher_rows"] == 16
    assert mix["distilled_node_key"] == DISTILLED_NODE_KEY
    supervision = result.artifacts["g"]["node_supervision"]
    # 4 leaf rows per tree consumed; merges have no distilled rows.
    assert supervision["n_leaf_rows"] == 16
    assert supervision["n_merge_rows"] == 0


def test_fit_distilled_requires_a_consuming_channel(tmp_path: Path) -> None:
    trees = _trees(2)
    rows_path = _teacher_rows_path(tmp_path, trees)
    with pytest.raises(ValueError, match="no loss channel consumes"):
        fit(
            _fit_config(
                tmp_path,
                trees,
                local_label_mix="llm_distilled",
                distilled_labels_path=str(rows_path),
            )
        )


def test_fit_distilled_respects_nested_node_target_key(tmp_path: Path) -> None:
    """A nested family-config node_target_key must not be clobbered by the lane."""

    trees = _trees(2)
    for tree in trees:
        for node in tree.nodes:
            if node.unit_type == "leaf":
                node.metadata["my_key"] = 1.0
    rows_path = _teacher_rows_path(tmp_path, trees)
    config = _fit_config(
        tmp_path,
        trees,
        local_label_mix="llm_distilled",
        distilled_labels_path=str(rows_path),
        supervision_level="node",
    )
    config["backend_config"]["neural_operator_config"] = {"node_target_key": "my_key"}
    result = fit(config)
    assert result.status == "success"
    node_supervision = result.artifacts["g"]["node_supervision"]
    assert node_supervision["n_leaf_rows"] == 8


def test_fit_distilled_via_predictor_callable(tmp_path: Path) -> None:
    trees = _trees(3)
    calls: list[str] = []

    def scorer(text: str) -> float:
        calls.append(text)
        return 0.5

    config = _fit_config(
        tmp_path,
        trees,
        local_label_mix="llm_distilled",
        supervision_level="node",
    )
    config["backend_config"]["node_oracle_predictor"] = scorer
    result = fit(config)
    assert result.status == "success"
    mix = result.summary["grid_axes"]["local_label_mix"]
    assert mix["labels_source"] == "node_oracle_predictor"
    assert mix["n_nodes_attached"] == 12
    assert calls  # the predictor really scored node texts


def test_fit_refuses_degenerate_placeholder_teacher_cache(tmp_path: Path) -> None:
    """The leafq001 bug shape: every teacher row carries the same score."""

    trees = _trees(2)
    path = tmp_path / "teacher_node_rows.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for tree in trees:
            for node in tree.nodes:
                if node.unit_type == "leaf":
                    row = {"doc_id": tree.doc_id, "node_id": node.node_id, "rile_norm": 0.5}
                    handle.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="degenerate"):
        fit(
            _fit_config(
                tmp_path,
                trees,
                local_label_mix="llm_distilled",
                distilled_labels_path=str(path),
                supervision_level="node",
            )
        )
