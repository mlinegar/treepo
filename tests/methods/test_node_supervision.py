"""Phase 1 of the fit-grid plan: per-node supervision through ``fit()``.

Covers the named supervision levels (TT ladder vocabulary), trace-aligned
node-target extraction from labeled ``TreeRecord`` bundles, the weighted
node-mean loss, the ObjectiveSpec fold (node terms feed the C1/C3 law
channels), the gold_fraction gating, and the bundle -> fit end-to-end path.
"""

from __future__ import annotations

import json
from dataclasses import replace
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
from treepo.methods.fno import NeuralOperatorFamily
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
        schedule="fg",
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
    assert again.schedule == "fg"


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
    rows = _node_supervision_targets([_labeled_tree("doc")], config, width=1, include_merges=False)
    assert rows is not None
    _targets, observed = rows[0]
    assert observed[:4] == [True] * 4 and not any(observed[4:])
    rows = _node_supervision_targets([_labeled_tree("doc")], config, width=1, include_leaves=False)
    assert rows is not None
    _targets, observed = rows[0]
    assert observed == [False, False, False, False, True, True, False]


def test_singleton_named_node_supervision_reduces_exactly_to_scalar(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    trees = _trees(6)
    scalar_config = NeuralOperatorFamilyConfig()
    singleton_config = NeuralOperatorFamilyConfig(
        target_names=("score",),
        target_oracle_ids=("exact_panel:v1",),
    )

    scalar_targets = _node_supervision_targets(trees, scalar_config, width=1)
    singleton_targets = _node_supervision_targets(trees, singleton_config, width=1)
    assert singleton_targets == scalar_targets

    scalar = fit(
        _fit_config(
            tmp_path / "scalar",
            trees,
            supervision_level="mix",
        )
    )
    singleton = fit(
        _fit_config(
            tmp_path / "singleton",
            trees,
            supervision_level="mix",
            oracle_targets=[
                {
                    "target_name": "score",
                    "oracle_id": "exact_panel:v1",
                }
            ],
        )
    )

    assert scalar.status == singleton.status == "success"
    for artifact_name in ("f", "g"):
        scalar_artifact = scalar.artifacts[artifact_name]
        singleton_artifact = singleton.artifacts[artifact_name]
        assert singleton_artifact["loss"] == scalar_artifact["loss"]
        assert singleton_artifact["target_center"] == scalar_artifact["target_center"]
        assert singleton_artifact["target_scale"] == scalar_artifact["target_scale"]
        assert singleton_artifact["node_supervision"] == scalar_artifact["node_supervision"]
        scalar_state = torch.load(
            scalar_artifact["weights_path"], map_location="cpu", weights_only=True
        )
        singleton_state = torch.load(
            singleton_artifact["weights_path"], map_location="cpu", weights_only=True
        )
        assert scalar_state.keys() == singleton_state.keys()
        assert all(torch.equal(scalar_state[key], singleton_state[key]) for key in scalar_state)


def test_node_targets_respect_supervised_unit_pinning() -> None:
    config = NeuralOperatorFamilyConfig(supervised_node_units=("doc::doc_l0", "doc::doc_l2"))
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


# ---------------------- independent root observations -------------------- #


def _prediction_scalars(result) -> list[float]:
    rows = result.history[-1]["extra"]["prediction_rows"]
    return [float(row["prediction_scalar"]) for row in rows]


def _masked_fit_config(
    tmp_path: Path,
    trees,
    *,
    root_ids: tuple[str, ...],
    micro_batch_size: int | None = None,
    root_weight: float = 1.0,
    leaf_weight: float = 1.0,
) -> dict[str, object]:
    config = _fit_config(
        tmp_path,
        trees,
        root_observed_doc_ids=root_ids,
        root_weight=root_weight,
        leaf_weight=leaf_weight,
        merge_weight=0.0,
        axis={"axis_kind": "leaf_count", "axis_value": 4, "max_iterations": 1},
    )
    config["backend_config"] = _backend_config(
        tmp_path,
        micro_batch_size=micro_batch_size,
        normalize_targets=False,
    )
    return config


def test_masked_root_labels_do_not_leak_and_local_trees_stay_in_training(
    tmp_path: Path,
) -> None:
    trees = _trees(6)
    root_ids = ("doc_00", "doc_03")
    changed = [
        tree if tree.doc_id in root_ids else replace(tree, root_label=1000.0 + index)
        for index, tree in enumerate(trees)
    ]

    first = fit(_masked_fit_config(tmp_path / "a", trees, root_ids=root_ids))
    second = fit(_masked_fit_config(tmp_path / "b", changed, root_ids=root_ids))

    assert _prediction_scalars(first) == pytest.approx(_prediction_scalars(second), abs=1.0e-8)
    supervision = first.artifacts["f"]["node_supervision"]
    assert supervision["n_trees"] == 6
    assert supervision["n_root_rows"] == 2
    assert supervision["n_leaf_rows"] == 24


def test_empty_root_mask_trains_from_local_rows(tmp_path: Path) -> None:
    result = fit(_masked_fit_config(tmp_path, _trees(6), root_ids=(), root_weight=0.0))
    assert result.status == "success"
    supervision = result.artifacts["f"]["node_supervision"]
    assert supervision["n_trees"] == 6
    assert supervision["n_root_rows"] == 0
    assert supervision["n_leaf_rows"] == 24


def test_zero_local_budget_falls_back_to_observed_roots(tmp_path: Path) -> None:
    trees = _trees(6)
    config = _masked_fit_config(
        tmp_path,
        trees,
        root_ids=("doc_00", "doc_03"),
        root_weight=1.0,
        leaf_weight=1.0,
    )
    config.update(
        {
            "local_label_mix": "gold_fraction",
            "gold_fraction_p": 0.0,
        }
    )
    result = fit(config)
    assert result.status == "success"
    supervision = result.artifacts["f"]["node_supervision"]
    assert supervision["n_root_rows"] == 2
    assert supervision["n_leaf_rows"] == 0
    assert result.summary["grid_axes"]["local_label_mix"]["selected_node_units"] == []


def test_sparse_root_only_mask_allows_batches_with_no_observed_root(
    tmp_path: Path,
) -> None:
    config = _masked_fit_config(
        tmp_path,
        _trees(6),
        root_ids=("doc_00",),
        root_weight=1.0,
        leaf_weight=0.0,
    )
    config["backend_config"] = _backend_config(
        tmp_path, batch_size=2, micro_batch_size=1, normalize_targets=False
    )
    result = fit(config)
    assert result.status == "success"
    assert result.artifacts["f"]["node_supervision"]["n_root_rows"] == 1


def test_sparse_root_only_full_batch_skips_decay_only_updates(
    tmp_path: Path,
) -> None:
    trees = _trees(6)
    roots = ("doc_00",)

    def run(path: Path, *, micro_batch_size: int | None):
        config = _masked_fit_config(
            path,
            trees,
            root_ids=roots,
            root_weight=1.0,
            leaf_weight=0.0,
        )
        config["backend_config"] = _backend_config(
            path,
            batch_size=2,
            micro_batch_size=micro_batch_size,
            normalize_targets=False,
            weight_decay=0.5,
        )
        return fit(config)

    full = run(tmp_path / "full", micro_batch_size=None)
    micro = run(tmp_path / "micro", micro_batch_size=1)
    assert _prediction_scalars(full) == pytest.approx(_prediction_scalars(micro), abs=2.0e-7)


@pytest.mark.parametrize("micro_batch_size", [None, 1])
def test_g_stage_leaf_only_batch_updates_the_single_shared_g(
    tmp_path: Path, micro_batch_size: int | None
) -> None:
    """Leaf targets update the same g that recursive reduction will call."""

    torch = pytest.importorskip("torch")
    trees = _trees(4)
    config = _masked_fit_config(
        tmp_path,
        trees,
        root_ids=(),
        root_weight=0.0,
        leaf_weight=1.0,
    )
    config["axis"] = {"axis_kind": "leaf_count", "axis_value": 4, "max_iterations": 2}
    config["backend_config"] = _backend_config(
        tmp_path,
        batch_size=2,
        micro_batch_size=micro_batch_size,
        normalize_targets=False,
        root_readout="root_state",
        weight_decay=0.5,
    )

    result = fit(config)
    f_state = torch.load(
        result.artifacts["f"]["weights_path"], map_location="cpu", weights_only=True
    )
    g_state = torch.load(
        result.artifacts["g"]["weights_path"], map_location="cpu", weights_only=True
    )
    g_keys = [key for key in f_state if key.startswith("g.")]
    assert g_keys
    assert any(not torch.equal(f_state[key], g_state[key]) for key in g_keys)
    assert not any(key.startswith(("leaf_fno.", "merge_fno.", "merge.")) for key in g_state)
    assert result.artifacts["g"]["g_role_activity"] == {
        "shared_g_optimizer_eligible": True,
        "leaf_domain_gradient_path_present": True,
        "merge_domain_gradient_path_present": False,
    }


@pytest.mark.parametrize("micro_batch_size", [None, 1])
def test_g_stage_observed_multileaf_root_updates_merge_parameters(
    tmp_path: Path, micro_batch_size: int | None
) -> None:
    torch = pytest.importorskip("torch")
    trees = _trees(4)
    config = _masked_fit_config(
        tmp_path,
        trees,
        root_ids=("doc_00",),
        root_weight=1.0,
        leaf_weight=1.0,
    )
    config["axis"] = {"axis_kind": "leaf_count", "axis_value": 4, "max_iterations": 2}
    config["backend_config"] = _backend_config(
        tmp_path,
        batch_size=4,
        micro_batch_size=micro_batch_size,
        normalize_targets=False,
        root_readout="root_state",
        weight_decay=0.5,
    )

    result = fit(config)
    f_state = torch.load(
        result.artifacts["f"]["weights_path"], map_location="cpu", weights_only=True
    )
    g_state = torch.load(
        result.artifacts["g"]["weights_path"], map_location="cpu", weights_only=True
    )
    g_keys = [key for key in f_state if key.startswith("g.")]
    assert g_keys
    assert any(not torch.equal(f_state[key], g_state[key]) for key in g_keys)
    assert result.artifacts["g"]["g_role_activity"] == {
        "shared_g_optimizer_eligible": True,
        "leaf_domain_gradient_path_present": True,
        "merge_domain_gradient_path_present": True,
    }


def test_masked_root_microbatch_matches_full_batch(tmp_path: Path) -> None:
    trees = _trees(6)
    roots = ("doc_00", "doc_02", "doc_04")
    full = fit(_masked_fit_config(tmp_path / "full", trees, root_ids=roots, micro_batch_size=None))
    micro = fit(_masked_fit_config(tmp_path / "micro", trees, root_ids=roots, micro_batch_size=1))
    assert _prediction_scalars(full) == pytest.approx(_prediction_scalars(micro), abs=2.0e-6)


def test_masked_root_per_tree_loss_matches_across_chunk_sizes(tmp_path: Path) -> None:
    trees = _trees(6)
    roots = ("doc_00",)
    local_units = ("doc_01::doc_01_l0",)

    def run(path: Path, *, micro_batch_size: int | None):
        config = _masked_fit_config(
            path,
            trees,
            root_ids=roots,
            root_weight=1.0,
            leaf_weight=1.0,
        )
        config.update(
            {
                "local_label_mix": "gold_fraction",
                "gold_fraction_p": 1.0 / 24.0,
            }
        )
        config["backend_config"] = _backend_config(
            path,
            batch_size=2,
            micro_batch_size=micro_batch_size,
            normalize_targets=False,
            per_tree_loss_lambda=0.5,
            supervised_node_units=local_units,
            weight_decay=0.5,
        )
        return fit(config)

    one_tree_chunks = run(tmp_path / "one", micro_batch_size=None)
    two_tree_chunks = run(tmp_path / "two", micro_batch_size=2)
    assert _prediction_scalars(one_tree_chunks) == pytest.approx(
        _prediction_scalars(two_tree_chunks), abs=2.0e-7
    )


def test_per_tree_active_denominator_counts_root_only_and_local_only_trees() -> None:
    torch = pytest.importorskip("torch")
    family = NeuralOperatorFamily(
        NeuralOperatorFamilyConfig(
            operator_kind="conv1d",
            embedding_dim=4,
            hidden_channels=2,
            leaf_weight=1.0,
            merge_weight=0.0,
            per_tree_loss_lambda=0.5,
        )
    )
    targets = torch.zeros((7, 1))
    none = torch.zeros(7, dtype=torch.bool)
    one_leaf = none.clone()
    one_leaf[0] = True
    count = family._active_tree_count(
        [0, 1, 2],
        root_observed=torch.tensor([True, False, False]),
        node_supervision=[
            (targets, none),
            (targets, one_leaf),
            (targets, none),
        ],
    )
    assert count == 2

    family.config.per_tree_loss_lambda = 0.0
    assert (
        family._active_tree_count(
            [0, 1, 2],
            root_observed=torch.tensor([True, False, False]),
            node_supervision=[
                (targets, none),
                (targets, one_leaf),
                (targets, none),
            ],
        )
        == 1
    )
    family.config.per_tree_loss_lambda = 1.0
    assert (
        family._active_tree_count(
            [0, 1, 2],
            root_observed=torch.tensor([True, False, False]),
            node_supervision=[
                (targets, none),
                (targets, one_leaf),
                (targets, none),
            ],
        )
        == 1
    )


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_per_tree_loss_lambda_rejects_nonconvex_values(value: float) -> None:
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        NeuralOperatorFamilyConfig(per_tree_loss_lambda=value)


def test_masked_root_weighted_denominator_counts_observed_roots_only() -> None:
    torch = pytest.importorskip("torch")
    family = NeuralOperatorFamily(
        NeuralOperatorFamilyConfig(
            operator_kind="conv1d",
            embedding_dim=4,
            hidden_channels=2,
            root_weight=1.0,
            leaf_weight=1.0,
        )
    )
    predictions = torch.tensor([[1.0], [9.0]], requires_grad=True)
    targets = torch.zeros_like(predictions)
    root_loss, root_count = family._masked_root_loss(
        predictions, targets, torch.tensor([True, False])
    )
    node_rows = (
        torch.tensor([4.0, 16.0], requires_grad=True),
        torch.tensor([0, 0]),
        torch.tensor([True, True]),
    )
    loss = family._node_weighted_loss(root_loss, node_rows, root_count=root_count)
    assert root_count == 1
    assert float(loss.detach()) == pytest.approx((1.0 + 4.0 + 16.0) / 3.0)


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
    # The lone leaf is also the root: root loss trains f(g(X)), while its
    # retained C1 row supervises g's leaf-build role. There are no C3 rows.
    result = fit(
        _fit_config(
            tmp_path,
            single,
            supervision_level="default",
            root_weight=3.0,
            leaf_weight=1.0,
            merge_weight=0.0,
            axis={"axis_kind": "leaf_count", "axis_value": 1},
        )
    )
    assert result.status == "success"
    payload = result.artifacts["g"]["node_supervision"]
    assert payload["n_leaf_rows"] == 4
    assert payload["n_merge_rows"] == 0
    assert result.artifacts["g"]["optimizer_step_count"] > 0
    assert result.artifacts["g"]["g_role_activity"] == {
        "shared_g_optimizer_eligible": True,
        "leaf_domain_gradient_path_present": True,
        "merge_domain_gradient_path_present": False,
    }


def test_single_leaf_root_only_trains_shared_g_from_leaf_domain(tmp_path: Path) -> None:
    single = [
        TreeRecord(
            tree_id=f"root_only_{i}",
            root_label=0.2 + 0.1 * i,
            nodes=(
                TreeNode(
                    node_id=f"root_only_{i}_l0",
                    unit_type="leaf",
                    text=f"complete singleton document {i}",
                    level=0,
                    position=0,
                ),
            ),
            metadata={"split": "train"},
        )
        for i in range(4)
    ]
    result = fit(
        _fit_config(
            tmp_path,
            single,
            axis={"axis_kind": "leaf_count", "axis_value": 1, "max_iterations": 2},
        )
    )

    assert result.status == "success"
    assert result.artifacts["g"]["optimizer_step_count"] > 0
    assert result.artifacts["g"]["g_role_activity"] == {
        "shared_g_optimizer_eligible": True,
        "leaf_domain_gradient_path_present": True,
        "merge_domain_gradient_path_present": False,
    }
    contract = result.summary["g_contract"]
    assert contract["summarized_singleton"] is True
    assert contract["composition_present"] is False
    assert contract["same_g_across_node_roles"] is True
    assert contract["reduce_g_is_derived"] is True
    assert contract["merge_domain_training_observed"] is False
    assert contract["shared_g_updated_with_merge_domain"] is False


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
        json.dumps(
            {"train": [t["doc_id"] for t in trees[:3]], "val": [], "test": [trees[3]["doc_id"]]}
        ),
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
    result = fit(_fit_config(tmp_path / "no", train, supervision_level="node", eval_data=train))
    assert result.status == "success"
    payload = result.artifacts["g"]["node_supervision"]
    assert payload["n_leaf_rows"] == 12  # 3 train docs x 4 leaves
    assert payload["n_merge_rows"] == 6  # 3 train docs x 2 internal merges
