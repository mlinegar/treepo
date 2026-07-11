"""Phase 1 of the fit-grid plan: per-node supervision through ``fit()``.

Covers the named supervision levels (TT ladder vocabulary), trace-aligned
node-target extraction from labeled ``TreeRecord`` bundles, the weighted
node-mean loss, the ObjectiveSpec fold (node terms feed the C1/C3 law
channels), the gold_fraction gating, and the bundle -> fit end-to-end path.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from treepo import fit
from treepo.bundles import load_labeled_tree_bundle
from treepo.methods._fno_config import NeuralOperatorFamilyConfig
from treepo.methods._fno_targets import _node_supervision_targets
from treepo.methods._supervision import (
    SUPERVISION_LEVELS,
    normalize_supervision_level,
    resolve_supervision,
)
from treepo.methods.contracts import CTreePOLearningSpec
from treepo.objective import ObjectiveSpec
from treepo.tree import TreeNode, TreeRecord, tree_root_target


def _labeled_tree(doc_id: str, *, offset: float = 0.0, split: str = "train") -> TreeRecord:
    """Four labeled leaves, two labeled merges, one labeled root."""

    leaf_scores = [0.1 + offset, 0.2 + offset, 0.3 + offset, 0.4 + offset]
    leaves = [
        TreeNode(
            node_id=f"{doc_id}_l{i}",
            unit_type="leaf",
            text=f"{doc_id} unit {i} alpha beta gamma delta",
            level=0,
            position=i,
            label=score,
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


def _trees(n: int = 6) -> list[TreeRecord]:
    return [_labeled_tree(f"doc_{i:02d}", offset=0.05 * i) for i in range(n)]


def _backend_config(tmp_path: Path, **extra: object) -> dict[str, object]:
    return {
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
        "output_dir": str(tmp_path),
        **extra,
    }


def _fit_config(tmp_path: Path, trees, **spec_extra: object) -> dict[str, object]:
    return {
        "family": "neural_operator",
        "train_data": trees,
        "eval_data": trees,
        "axis": {"axis_kind": "leaf_count", "axis_value": 4},
        "backend_config": _backend_config(tmp_path),
        **spec_extra,
    }


# --------------------------- level resolution --------------------------- #


def test_supervision_levels_mirror_tt_ladder_names() -> None:
    assert set(SUPERVISION_LEVELS) == {"default", "root", "leaf", "node", "mix"}
    assert SUPERVISION_LEVELS["root"] == {
        "root_weight": 1.0,
        "leaf_weight": 0.0,
        "merge_weight": 0.0,
    }
    assert SUPERVISION_LEVELS["node"] == {
        "root_weight": 0.0,
        "leaf_weight": 1.0,
        "merge_weight": 1.0,
    }
    assert SUPERVISION_LEVELS["mix"] == {
        "root_weight": 3.0,
        "leaf_weight": 1.0,
        "merge_weight": 1.0,
    }
    assert SUPERVISION_LEVELS["default"] == {}


def test_resolve_supervision_default_passthrough_and_explicit_weights() -> None:
    assert resolve_supervision(SimpleNamespace(supervision_level="default")) == {}
    resolved = resolve_supervision(
        SimpleNamespace(supervision_level="default", leaf_weight=0.5, merge_weight=None)
    )
    assert resolved == {"leaf_weight": 0.5}


def test_resolve_supervision_rejects_level_plus_explicit_weight() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_supervision(SimpleNamespace(supervision_level="node", leaf_weight=2.0))


def test_unknown_supervision_level_errors() -> None:
    with pytest.raises(ValueError, match="unknown supervision_level"):
        normalize_supervision_level("balanced")


def test_spec_round_trips_supervision_fields() -> None:
    spec = CTreePOLearningSpec(
        space_kind="tree",
        family="fno",
        schedule="balanced",
        supervision_level="mix",
        local_law_weight=0.25,
    )
    payload = spec.to_dict()
    assert payload["supervision_level"] == "mix"
    assert payload["local_law_weight"] == 0.25
    again = CTreePOLearningSpec.from_mapping(payload)
    assert again.supervision_level == "mix"
    assert again.local_law_weight == 0.25
    assert again.leaf_weight is None


# ------------------------- node-target extraction ------------------------ #


def test_node_targets_align_to_trace_order() -> None:
    config = NeuralOperatorFamilyConfig()
    rows = _node_supervision_targets([_labeled_tree("doc")], config, width=1)
    assert rows is not None
    targets, observed = rows[0]
    # 4 leaves -> 7 trace nodes; root row (last) is never observed here.
    assert len(targets) == 7
    assert observed == [True, True, True, True, True, True, False]
    assert [row[0] for row in targets[:4]] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    # Balanced schedule: merge 4 = leaves (0,1), merge 5 = leaves (2,3).
    assert targets[4][0] == pytest.approx(0.15)
    assert targets[5][0] == pytest.approx(0.35)


def test_node_targets_channel_flags() -> None:
    config = NeuralOperatorFamilyConfig()
    rows = _node_supervision_targets(
        [_labeled_tree("doc")], config, width=1, include_merges=False
    )
    assert rows is not None
    _targets, observed = rows[0]
    assert observed[:4] == [True] * 4 and not any(observed[4:])
    rows = _node_supervision_targets(
        [_labeled_tree("doc")], config, width=1, include_leaves=False
    )
    assert rows is not None
    _targets, observed = rows[0]
    assert observed == [False, False, False, False, True, True, False]


def test_node_targets_respect_supervised_unit_pinning() -> None:
    config = NeuralOperatorFamilyConfig(
        supervised_node_units=("doc::doc_l0", "doc::doc_l2")
    )
    rows = _node_supervision_targets([_labeled_tree("doc")], config, width=1)
    assert rows is not None
    _targets, observed = rows[0]
    assert observed[:4] == [True, False, True, False]


def test_tree_root_target_reads_bundle_fields() -> None:
    record = _labeled_tree("doc")
    assert tree_root_target(record) == pytest.approx(0.25)
    plain = SimpleNamespace(metadata={"teacher_score_native": 1.5})
    assert tree_root_target(plain) == pytest.approx(1.5)
    assert tree_root_target(SimpleNamespace(metadata={})) is None


# ------------------------ per-level loss activation ---------------------- #


@pytest.mark.parametrize(
    ("level", "expect_leaf", "expect_merge", "expect_root_weight"),
    [
        ("default", 0, 0, 1.0),
        ("root", 0, 0, 1.0),
        ("leaf", 24, 0, 0.0),
        ("node", 24, 12, 0.0),
        ("mix", 24, 12, 3.0),
    ],
)
def test_fit_trains_each_named_level(
    tmp_path: Path, level: str, expect_leaf: int, expect_merge: int, expect_root_weight: float
) -> None:
    result = fit(_fit_config(tmp_path, _trees(6), supervision_level=level))
    assert result.status == "success"
    assert result.summary["supervision"]["level"] == level
    payload = result.artifacts["g"]["node_supervision"]
    assert payload["n_leaf_rows"] == expect_leaf
    assert payload["n_merge_rows"] == expect_merge
    assert payload["root_weight"] == pytest.approx(expect_root_weight)
    assert all(v is not None for v in (result.artifacts["f"], result.artifacts["g"]))


def test_gold_fraction_axis_gates_consumed_leaf_labels(tmp_path: Path) -> None:
    trees = _trees(6)
    result = fit(
        _fit_config(
            tmp_path,
            trees,
            supervision_level="leaf",
            local_label_mix="gold_fraction",
            gold_fraction_p=0.5,
            seed=7,
        )
    )
    assert result.status == "success"
    payload = result.artifacts["g"]["node_supervision"]
    # 6 trees x 4 leaves = 24 units; p=0.5 pins round(12) of them.
    assert payload["n_leaf_rows"] == 12
    selected = result.summary["grid_axes"]["local_label_mix"]["selected_node_units"]
    assert len(selected) == 12


def test_node_level_without_node_labels_is_loud(tmp_path: Path) -> None:
    unlabeled = [
        TreeRecord(
            tree_id=f"plain_{i}",
            root_label=0.5,
            nodes=(
                TreeNode(node_id=f"p{i}_l0", unit_type="leaf", text="aa bb", level=0, position=0),
                TreeNode(node_id=f"p{i}_l1", unit_type="leaf", text="cc dd", level=0, position=1),
            ),
            metadata={"split": "train"},
        )
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="no per-node targets"):
        fit(_fit_config(tmp_path, unlabeled, supervision_level="node"))


def test_node_supervision_requires_capable_family(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="per-node supervision support"):
        fit(
            {
                "family": "learnable_constant",
                "train_data": _trees(3),
                "supervision_level": "node",
                "backend_config": {"output_dir": str(tmp_path)},
            }
        )


# ------------------------- objective (law) fold --------------------------- #


def test_objective_law_channels_consume_node_targets(tmp_path: Path) -> None:
    config = _fit_config(tmp_path, _trees(6))
    config["backend_config"]["objective"] = ObjectiveSpec(
        objective_family="root_plus_local_laws",
        local_law_estimator="oracle_state",
        local_law_weight=0.5,
        root_share=0.5,
        local_law_component_weights={
            "leaf_preservation": 0.25,
            "merge_preservation": 0.25,
        },
    )
    result = fit(config)
    assert result.status == "success"
    payload = result.artifacts["g"]["node_supervision"]
    assert payload["law_source"] == "node_targets"
    assert payload["n_leaf_rows"] == 24
    assert payload["n_merge_rows"] == 12


def test_spec_local_law_weight_builds_canonical_objective(tmp_path: Path) -> None:
    result = fit(_fit_config(tmp_path, _trees(6), local_law_weight=0.5))
    assert result.status == "success"
    objective = result.summary["objective"]
    assert objective["root_share"] == pytest.approx(0.5)
    assert objective["local_law_weight"] == pytest.approx(0.5)
    assert result.artifacts["g"]["node_supervision"]["law_source"] == "node_targets"


def test_spec_local_law_weight_conflicts_with_explicit_objective(tmp_path: Path) -> None:
    config = _fit_config(tmp_path, _trees(3), local_law_weight=0.5)
    config["backend_config"]["objective"] = ObjectiveSpec()
    with pytest.raises(ValueError, match="mutually exclusive"):
        fit(config)


def test_objective_excludes_node_weights(tmp_path: Path) -> None:
    config = _fit_config(tmp_path, _trees(3), supervision_level="node")
    config["backend_config"]["objective"] = ObjectiveSpec(
        objective_family="root_plus_local_laws",
        local_law_estimator="oracle_state",
        local_law_weight=0.5,
        root_share=0.5,
        local_law_component_weights={"leaf_preservation": 0.5},
    )
    with pytest.raises(ValueError, match="single weight source"):
        fit(config)


# ------------------------------ structure -------------------------------- #


def test_single_leaf_tree_reduces_to_root_supervision(tmp_path: Path) -> None:
    single = [
        TreeRecord(
            tree_id=f"one_{i}",
            root_label=0.2 + 0.1 * i,
            nodes=(
                TreeNode(
                    node_id=f"one_{i}_l0",
                    unit_type="leaf",
                    text=f"solo unit {i}",
                    level=0,
                    position=0,
                    label=0.2 + 0.1 * i,
                ),
            ),
            metadata={"split": "train"},
        )
        for i in range(4)
    ]
    # leaf_001: the lone leaf IS the root; fg reduces to f (root term only),
    # so even the densest level trains without node rows.
    result = fit(
        _fit_config(tmp_path, single, supervision_level="mix", axis={"axis_kind": "leaf_count", "axis_value": 1})
    )
    assert result.status == "success"
    payload = result.artifacts["g"]["node_supervision"]
    assert payload["n_leaf_rows"] == 0
    assert payload["n_merge_rows"] == 0


# ---------------------------- bundle -> fit ------------------------------- #


def _write_bundle(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    trees = []
    for i in range(4):
        doc_id = f"bdoc_{i}"
        record = _labeled_tree(doc_id, offset=0.05 * i)
        nodes = {}
        for node in record.nodes:
            nodes[node.node_id] = {
                "node_id": node.node_id,
                "level": node.level,
                "text": node.text,
                "score": node.label,
                "dimension_scores": {"rile": node.label},
                "left_child_id": node.left_child_id,
                "right_child_id": node.right_child_id,
                "metadata": {"is_leaf": node.unit_type == "leaf"},
            }
        trees.append(
            {
                "version": "3.0",
                "doc_id": doc_id,
                "document_text": " ".join(n.text for n in record.nodes if n.unit_type == "leaf"),
                "document_score": record.root_label,
                "nodes": nodes,
                "levels": [
                    [n.node_id for n in record.nodes if n.level == 0],
                    [n.node_id for n in record.nodes if n.level == 1],
                    [n.node_id for n in record.nodes if n.level == 2],
                ],
                "metadata": {"split": "train" if i < 3 else "test"},
                "label_source": "synthetic_test_v1",
            }
        )
    (directory / "labeled_trees.jsonl").write_text(
        "\n".join(json.dumps(t) for t in trees) + "\n", encoding="utf-8"
    )
    (directory / "split_ids.json").write_text(
        json.dumps({"train": [t["doc_id"] for t in trees[:3]], "val": [], "test": [trees[3]["doc_id"]]}),
        encoding="utf-8",
    )
    return directory


def test_bundle_fit_end_to_end_learnable_constant(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    train = load_labeled_tree_bundle(bundle, split="train")
    eval_trees = load_labeled_tree_bundle(bundle, split="test")
    result = fit(
        {
            "family": "learnable_constant",
            "train_data": train,
            "eval_data": eval_trees,
            "backend_config": {"output_dir": str(tmp_path / "lc")},
        }
    )
    assert result.status == "success"
    # The trained constant is the mean root target of the loaded bundle trees.
    expected = sum(float(t.root_label) for t in train) / len(train)
    assert float(result.artifacts["f"]) == pytest.approx(expected)


def test_bundle_fit_end_to_end_node_supervision(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    train = load_labeled_tree_bundle(bundle, split="train", dimension="rile")
    result = fit(
        _fit_config(tmp_path / "no", train, supervision_level="node", eval_data=train)
    )
    assert result.status == "success"
    payload = result.artifacts["g"]["node_supervision"]
    assert payload["n_leaf_rows"] == 12  # 3 train docs x 4 leaves
    assert payload["n_merge_rows"] == 6  # 3 train docs x 2 internal merges
