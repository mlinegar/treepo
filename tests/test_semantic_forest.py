from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from treepo import OracleTargetSpec, fit
from treepo.forest import normalize_oracle_targets
from treepo.methods._fno_config import NeuralOperatorFamilyConfig
from treepo.methods._fno_targets import _coerce_node_target, _target_vector
from treepo.methods._runtime_evaluation import _prediction_rows, _split_metrics
from treepo.methods.contracts import CTreePOLearningSpec
from treepo.methods.fixtures import make_markov_changepoint_trees
from treepo.tree import TreeRecord

TARGETS = (
    OracleTargetSpec(
        target_name="economic_left_right",
        oracle_id="expert_panel_2026_v3",
        metadata={"scale": "unit_interval"},
    ),
    OracleTargetSpec(
        target_name="immigration",
        oracle_id="expert_panel_2026_v3",
    ),
)
TARGET_NAMES = tuple(target.target_name for target in TARGETS)
ORACLE_IDS = tuple(target.oracle_id for target in TARGETS)


def _tree(
    tree_id: str,
    *,
    economic_left_right: float,
    immigration: float,
) -> TreeRecord:
    # Deliberately reverse mapping insertion order. The OracleTargetSpec
    # sequence, not input-map order, is the canonical model coordinate order.
    return TreeRecord(
        tree_id=tree_id,
        root_label={
            "immigration": float(immigration),
            "economic_left_right": float(economic_left_right),
        },
        metadata={"split": "test"},
    )


class CountingJointVectorFamily:
    """Dependency-free fixture for one shared-g, joint-vector fit."""

    name = "counting_joint_vector"

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            target_names=TARGET_NAMES,
            target_oracle_ids=ORACLE_IDS,
            target_vector_key=None,
            root_observed_doc_ids=None,
        )
        self.f_calls = 0
        self.g_calls = 0

    def train_f(self, *, f_init, g, traces, output_dir, iteration):
        del f_init, g, output_dir
        self.f_calls += 1
        return {
            "kind": "joint_vector_f",
            "iteration": int(iteration),
            "n_train": len(traces),
            "target_order": list(TARGET_NAMES),
        }

    def train_g(self, *, g_init, f, traces, output_dir, iteration):
        del g_init, f, output_dir
        self.g_calls += 1
        return {
            "kind": "one_shared_g",
            "iteration": int(iteration),
            "n_train": len(traces),
            "target_order": list(TARGET_NAMES),
        }

    def score_roots_with_f(self, *, f, g, trees):
        del f, g
        return [[0.25, 0.75] for _tree_value in trees]

    def validate_artifact(self, *, kind, artifact):
        assert kind in {"f", "g"}
        assert isinstance(artifact, dict)


class ScalarOnlyFamily:
    """Implements FamilyRuntime but intentionally has no vector schema."""

    name = "scalar_only"
    config = SimpleNamespace(root_observed_doc_ids=None)

    def train_f(self, *, f_init, g, traces, output_dir, iteration):
        del g, traces, output_dir, iteration
        return f_init

    def train_g(self, *, g_init, f, traces, output_dir, iteration):
        del f, traces, output_dir, iteration
        return g_init

    def score_roots_with_f(self, *, f, g, trees):
        del f, g
        return [0.0 for _tree_value in trees]

    def validate_artifact(self, *, kind, artifact):
        del kind, artifact


def test_oracle_target_normalization_and_learning_spec_round_trip() -> None:
    targets = normalize_oracle_targets(
        [
            TARGETS[0],
            {
                "target_name": "immigration",
                "oracle_id": "expert_panel_2026_v3",
                "metadata": {"source": "hand_coding"},
            },
        ]
    )

    assert tuple(target.target_name for target in targets) == TARGET_NAMES
    assert tuple(target.oracle_id for target in targets) == ORACLE_IDS
    assert (
        targets[0].contract_digest
        == OracleTargetSpec.from_value(targets[0].to_dict()).contract_digest
    )

    spec = CTreePOLearningSpec.from_mapping(
        {
            "space_kind": "shared_semantic_state.v1",
            "family": "neural_operator",
            "schedule": "fg",
            "oracle_targets": [target.to_dict() for target in targets],
        }
    )
    payload = spec.to_dict()
    restored = CTreePOLearningSpec.from_mapping(payload)

    assert [row["target_name"] for row in payload["oracle_targets"]] == list(TARGET_NAMES)
    assert tuple(target.target_name for target in restored.oracle_targets) == TARGET_NAMES
    assert tuple(target.oracle_id for target in restored.oracle_targets) == ORACLE_IDS

    with pytest.raises(ValueError, match="unique"):
        normalize_oracle_targets([TARGETS[0], TARGETS[0]])
    with pytest.raises(TypeError, match="ordered iterable"):
        normalize_oracle_targets({"economic_left_right": "panel"})
    with pytest.raises(ValueError, match="per-target data"):
        normalize_oracle_targets(
            [
                {
                    "target_name": "economic_left_right",
                    "oracle_id": "panel",
                    "train_data": ["separate-child-fit-is-not-allowed"],
                }
            ]
        )


def test_dense_mapping_targets_follow_declared_order_and_require_every_coordinate() -> None:
    config = NeuralOperatorFamilyConfig(
        operator_kind="conv1d",
        target_names=TARGET_NAMES,
        target_oracle_ids=ORACLE_IDS,
        normalize_targets=False,
    )
    tree = _tree(
        "doc_0",
        economic_left_right=0.2,
        immigration=0.8,
    )

    assert _target_vector(tree, config) == pytest.approx([0.2, 0.8])
    assert _coerce_node_target(
        {"immigration": 0.7, "economic_left_right": 0.3},
        width=2,
        target_names=TARGET_NAMES,
    ) == pytest.approx([0.3, 0.7])

    missing_root = TreeRecord(
        tree_id="missing_root",
        root_label={"economic_left_right": 0.2},
    )
    with pytest.raises(ValueError, match="exactly cover.*missing=.*immigration"):
        _target_vector(missing_root, config)
    with pytest.raises(ValueError, match="exactly cover.*missing=.*immigration"):
        _coerce_node_target(
            {"economic_left_right": 0.3},
            width=2,
            target_names=TARGET_NAMES,
        )

    scalar_tree = TreeRecord(
        tree_id="scalar",
        root_label=9.0,
        metadata={"declared_score": 0.35, "teacher_score_native": 0.7},
    )
    legacy_scalar = NeuralOperatorFamilyConfig(
        operator_kind="conv1d",
        target_key="declared_score",
        normalize_targets=False,
    )
    named_singleton = NeuralOperatorFamilyConfig(
        operator_kind="conv1d",
        target_key="declared_score",
        target_names=("declared_score",),
        target_oracle_ids=("oracle:v1",),
        normalize_targets=False,
    )
    assert (
        _target_vector(scalar_tree, named_singleton)
        == _target_vector(scalar_tree, legacy_scalar)
        == [0.35]
    )
    scalar_vector_key = NeuralOperatorFamilyConfig(
        operator_kind="conv1d",
        target_vector_key="declared_score",
        target_names=("declared_score",),
        target_oracle_ids=("oracle:v1",),
        normalize_targets=False,
    )
    assert _target_vector(scalar_tree, scalar_vector_key) == [0.35]
    conflicting_scalar = TreeRecord(
        tree_id="conflicting_scalar",
        root_label=0.9,
        metadata={"declared_score": 0.35},
    )
    assert _target_vector(conflicting_scalar, scalar_vector_key) == [0.35]
    assert _prediction_rows(
        [[0.4]],
        [conflicting_scalar],
        target_names=("declared_score",),
        target_oracle_ids=("oracle:v1",),
        target_vector_key="declared_score",
    )[0]["target_by_name"] == {"declared_score": 0.35}
    assert _coerce_node_target(
        0.35,
        width=1,
        target_names=("declared_score",),
    ) == [0.35]


def test_one_fit_trains_one_joint_f_and_one_shared_g_and_keeps_named_outputs(
    tmp_path,
) -> None:
    family = CountingJointVectorFamily()
    trees = [
        _tree("doc_0", economic_left_right=0.2, immigration=0.8),
        _tree("doc_1", economic_left_right=0.4, immigration=0.6),
    ]

    result = fit(
        {
            "space_kind": "shared_semantic_state.v1",
            "family": family.name,
            "schedule": "fg",
            "axis": {"max_iterations": 2},
            "oracle_targets": [target.to_dict() for target in TARGETS],
            "train_data": trees,
            "eval_data": trees,
            "backend_config": {"family_runtime": family},
        },
        output_dir=tmp_path / "joint",
    )

    assert result.status == "success"
    assert family.f_calls == 1
    assert family.g_calls == 1
    assert result.artifacts["f"]["kind"] == "joint_vector_f"
    assert result.artifacts["g"]["kind"] == "one_shared_g"
    assert "targets" not in result.artifacts

    assert result.summary["definition"] == "single_shared_g_joint_vector_f_star"
    assert result.summary["target_order"] == list(TARGET_NAMES)
    assert result.summary["oracle_ids_by_target"] == dict(zip(TARGET_NAMES, ORACLE_IDS))
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    results = json.loads(Path(result.artifacts["results_json"]).read_text(encoding="utf-8"))
    for payload in (manifest, results):
        assert payload["definition"] == result.summary["definition"]
        assert payload["target_order"] == result.summary["target_order"]
        assert payload["target_schema_digest"] == result.summary["target_schema_digest"]
    assert [target["target_name"] for target in manifest["spec"]["oracle_targets"]] == list(
        TARGET_NAMES
    )
    assert results["paired_rows"]["named_vector_fields"] == {
        "key": "tree_id",
        "target_order": "target_order",
        "prediction": "prediction_by_target",
        "gold": "target_by_name",
        "oracle_ids": "oracle_ids_by_target",
        "split": "split",
    }

    rows = result.history[-1]["extra"]["prediction_rows"]
    assert len(rows) == 2
    assert rows[0]["target_order"] == list(TARGET_NAMES)
    assert rows[0]["oracle_ids_by_target"] == dict(zip(TARGET_NAMES, ORACLE_IDS))
    assert rows[0]["prediction_by_target"] == {
        "economic_left_right": pytest.approx(0.25),
        "immigration": pytest.approx(0.75),
    }
    assert rows[0]["target_by_name"] == {
        "economic_left_right": pytest.approx(0.2),
        "immigration": pytest.approx(0.8),
    }
    assert rows[0]["prediction_scalar"] is None

    # Named vector evaluation remains coordinatewise. There is no implicit
    # coordinate-zero scalar metric or pooled cross-target average.
    assert "internal_f_mae" not in result.metrics
    assert result.metrics["economic_left_right_internal_f_mae"] == pytest.approx(0.1)
    assert result.metrics["immigration_internal_f_mae"] == pytest.approx(0.1)
    assert result.metrics["n"] == pytest.approx(2.0)


def test_named_joint_vector_rejects_a_scalar_only_family(tmp_path) -> None:
    tree = _tree("doc_0", economic_left_right=0.2, immigration=0.8)

    with pytest.raises(ValueError, match="named joint vector|does not consume"):
        fit(
            {
                "space_kind": "shared_semantic_state.v1",
                "family": "scalar_only",
                "schedule": "fg",
                "axis": {"max_iterations": 0},
                "oracle_targets": [target.to_dict() for target in TARGETS],
                "train_data": [tree],
                "eval_data": [tree],
                "backend_config": {"family_runtime": ScalarOnlyFamily()},
            },
            output_dir=tmp_path / "scalar",
        )


def test_singleton_named_fit_is_exactly_scalar_after_forgetting_schema(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    train = make_markov_changepoint_trees(
        n_trees=5,
        doc_tokens=24,
        leaf_unit_count=6,
        vocabulary_size=48,
        seed=81,
        split="train",
    )
    evaluation = make_markov_changepoint_trees(
        n_trees=3,
        doc_tokens=24,
        leaf_unit_count=6,
        vocabulary_size=48,
        seed=82,
        split="test",
    )
    backend_config = {
        "operator_kind": "conv1d",
        "embedding_dim": 8,
        "hidden_channels": 4,
        "conv_kernel_size": 3,
        "head_hidden_dim": 8,
        "epochs_per_iteration": 1,
        "batch_size": 5,
        "learning_rate": 0.01,
        "device": "cpu",
        "seed": 83,
    }
    common = {
        "space_kind": "markov_count_state.v1",
        "family": "neural_operator",
        "schedule": "fg",
        "train_data": train,
        "eval_data": evaluation,
        "backend_config": backend_config,
        "axis": {"max_iterations": 2, "axis_value": 0},
    }

    scalar = fit(common, output_dir=tmp_path / "scalar")
    singleton_spec = {
        **common,
        "oracle_targets": [
            {
                "target_name": "changepoint_count",
                "oracle_id": "markov_exact:v1",
            }
        ],
    }
    singleton = fit(singleton_spec, output_dir=tmp_path / "singleton")

    assert scalar.status == singleton.status == "success"
    for artifact_name in ("f", "g"):
        scalar_artifact = scalar.artifacts[artifact_name]
        singleton_artifact = singleton.artifacts[artifact_name]
        assert singleton_artifact["loss"] == scalar_artifact["loss"]
        assert singleton_artifact["target_center"] == scalar_artifact["target_center"]
        assert singleton_artifact["target_scale"] == scalar_artifact["target_scale"]
        scalar_state = torch.load(
            scalar_artifact["weights_path"], map_location="cpu", weights_only=True
        )
        singleton_state = torch.load(
            singleton_artifact["weights_path"], map_location="cpu", weights_only=True
        )
        assert scalar_state.keys() == singleton_state.keys()
        assert all(torch.equal(scalar_state[key], singleton_state[key]) for key in scalar_state)

    scalar_rows = scalar.history[-1]["extra"]["prediction_rows"]
    singleton_rows = singleton.history[-1]["extra"]["prediction_rows"]
    assert len(scalar_rows) == len(singleton_rows)
    for scalar_row, singleton_row in zip(scalar_rows, singleton_rows):
        for field_name in (
            "tree_id",
            "split",
            "prediction",
            "prediction_scalar",
            "teacher_score",
            "expert_score",
        ):
            assert singleton_row[field_name] == scalar_row[field_name]
        assert singleton_row["prediction_by_target"] == {
            "changepoint_count": scalar_row["prediction_scalar"]
        }

    for metric_name, scalar_value in scalar.metrics.items():
        assert singleton.metrics[metric_name] == scalar_value
    assert singleton.metrics["changepoint_count_internal_f_mae"] == scalar.metrics["internal_f_mae"]
    assert (
        singleton.metrics["test_changepoint_count_internal_f_mae"]
        == scalar.metrics["test_internal_f_mae"]
    )

    resumed = fit(
        {
            **singleton_spec,
            "initial_artifacts": {
                "f": singleton.artifacts["f"],
                "g": singleton.artifacts["g"],
            },
            "axis": {"max_iterations": 0, "axis_value": 0},
        },
        output_dir=tmp_path / "singleton_resumed",
    )
    assert [row["prediction"] for row in resumed.history[-1]["extra"]["prediction_rows"]] == [
        row["prediction"] for row in singleton_rows
    ]

    anonymous_to_named = fit(
        {
            **singleton_spec,
            "initial_artifacts": {
                "f": scalar.artifacts["f"],
                "g": scalar.artifacts["g"],
            },
            "axis": {"max_iterations": 0, "axis_value": 0},
        },
        output_dir=tmp_path / "anonymous_to_named",
    )
    assert [
        row["prediction"] for row in anonymous_to_named.history[-1]["extra"]["prediction_rows"]
    ] == [row["prediction"] for row in scalar_rows]

    named_to_anonymous = fit(
        {
            **common,
            "initial_artifacts": {
                "f": singleton.artifacts["f"],
                "g": singleton.artifacts["g"],
            },
            "axis": {"max_iterations": 0, "axis_value": 0},
        },
        output_dir=tmp_path / "named_to_anonymous",
    )
    assert [
        row["prediction"] for row in named_to_anonymous.history[-1]["extra"]["prediction_rows"]
    ] == [row["prediction"] for row in singleton_rows]


def test_singleton_named_fit_is_exactly_scalar_for_concrete_fno_leaf_mean(
    tmp_path,
) -> None:
    """The public FNO route, normalization, and leaf rollup are conservative."""

    torch = pytest.importorskip("torch")
    pytest.importorskip("neuralop")
    train = make_markov_changepoint_trees(
        n_trees=5,
        doc_tokens=24,
        leaf_unit_count=6,
        vocabulary_size=48,
        seed=91,
        split="train",
    )
    evaluation = make_markov_changepoint_trees(
        n_trees=3,
        doc_tokens=24,
        leaf_unit_count=6,
        vocabulary_size=48,
        seed=92,
        split="test",
    )
    common = {
        "space_kind": "markov_count_state.v1",
        "family": "fno",
        "schedule": "fg",
        "train_data": train,
        "eval_data": evaluation,
        "backend_config": {
            "embedding_dim": 8,
            "hidden_channels": 4,
            "n_modes": 2,
            "n_layers": 1,
            "head_hidden_dim": 8,
            "epochs_per_iteration": 1,
            "batch_size": 5,
            "learning_rate": 0.01,
            "device": "cpu",
            "seed": 93,
            "normalize_targets": True,
            "root_readout": "leaf_mean",
        },
        "axis": {"max_iterations": 2, "axis_value": 0},
    }

    scalar = fit(common, output_dir=tmp_path / "fno_scalar")
    singleton = fit(
        {
            **common,
            "oracle_targets": [
                {
                    "target_name": "changepoint_count",
                    "oracle_id": "markov_exact:v1",
                }
            ],
        },
        output_dir=tmp_path / "fno_singleton",
    )

    assert scalar.status == singleton.status == "success"
    for artifact_name in ("f", "g"):
        scalar_artifact = scalar.artifacts[artifact_name]
        singleton_artifact = singleton.artifacts[artifact_name]
        assert scalar_artifact["operator_kind"] == singleton_artifact["operator_kind"] == "fno"
        assert scalar_artifact["root_readout"] == singleton_artifact["root_readout"] == "leaf_mean"
        assert (
            scalar_artifact["normalize_targets"] is singleton_artifact["normalize_targets"] is True
        )
        assert singleton_artifact["loss"] == scalar_artifact["loss"]
        assert singleton_artifact["target_center"] == scalar_artifact["target_center"]
        assert singleton_artifact["target_scale"] == scalar_artifact["target_scale"]
        assert scalar_artifact["target_scale"] != [1.0]

        scalar_state = torch.load(
            scalar_artifact["weights_path"], map_location="cpu", weights_only=True
        )
        singleton_state = torch.load(
            singleton_artifact["weights_path"], map_location="cpu", weights_only=True
        )
        assert scalar_state.keys() == singleton_state.keys()
        assert all(torch.equal(scalar_state[key], singleton_state[key]) for key in scalar_state)

    scalar_rows = scalar.history[-1]["extra"]["prediction_rows"]
    singleton_rows = singleton.history[-1]["extra"]["prediction_rows"]
    assert [row["prediction_scalar"] for row in singleton_rows] == [
        row["prediction_scalar"] for row in scalar_rows
    ]
    for metric_name, scalar_value in scalar.metrics.items():
        assert singleton.metrics[metric_name] == scalar_value


def test_singleton_internal_truth_does_not_fabricate_external_expert_labels() -> None:
    teacher_only = TreeRecord(
        tree_id="teacher_only",
        metadata={"split": "test", "teacher_score_native": 0.5},
    )
    rows = _prediction_rows(
        [0.4],
        [teacher_only],
        target_names=("score",),
        target_oracle_ids=("teacher:v1",),
    )
    metrics = _split_metrics([0.4], [teacher_only], target_names=("score",))

    assert rows[0]["prediction_scalar"] == pytest.approx(0.4)
    assert rows[0]["teacher_score"] == pytest.approx(0.5)
    assert rows[0]["expert_score"] is None
    assert metrics.internal_f_mae == pytest.approx(0.1)
    assert metrics.external_expert_mae is None

    custom_target_only = TreeRecord(
        tree_id="custom_only",
        metadata={"split": "test", "custom_target": 0.6},
    )
    legacy_rows = _prediction_rows([0.4], [custom_target_only])
    named_rows = _prediction_rows(
        [0.4],
        [custom_target_only],
        target_names=("score",),
        target_oracle_ids=("custom:v1",),
        target_key="custom_target",
    )
    legacy_metrics = _split_metrics([0.4], [custom_target_only])
    named_metrics = _split_metrics(
        [0.4],
        [custom_target_only],
        target_names=("score",),
        target_key="custom_target",
    )
    assert named_rows[0]["prediction_scalar"] == legacy_rows[0]["prediction_scalar"]
    assert named_rows[0]["teacher_score"] == legacy_rows[0]["teacher_score"] is None
    assert named_rows[0]["expert_score"] == legacy_rows[0]["expert_score"] is None
    assert named_metrics.internal_f_mae == legacy_metrics.internal_f_mae is None
    assert named_metrics.external_expert_mae == legacy_metrics.external_expert_mae is None
    assert named_metrics.n == legacy_metrics.n == 0
    assert named_metrics.per_dimension["score"]["n"] == 1

    mapped_root = TreeRecord(tree_id="mapped", root_label={"score": 0.5})
    mapped_rows = _prediction_rows(
        [0.4],
        [mapped_root],
        target_names=("score",),
        target_oracle_ids=("expert:v1",),
    )
    assert mapped_rows[0]["teacher_score"] == pytest.approx(0.5)
    assert mapped_rows[0]["expert_score"] == pytest.approx(0.5)
