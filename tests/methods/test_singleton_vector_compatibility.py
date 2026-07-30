"""Package-wide regressions for the width-one named-vector compatibility path.

Adding one ``oracle_targets`` record to an established scalar fit must lift the
scalar family into the unified vector contract without changing its scalar
computation. Reporting adds named sidecars; projecting those sidecars away
recovers the legacy rows and metrics exactly.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from treepo import fit
from treepo.methods.fixtures import (
    make_hll_item_trees,
    make_markov_changepoint_trees,
)
from treepo.tree import TreeNode, TreeRecord, tree_root_target

TARGET_NAME = "singleton_score"
ORACLE_ID = "test_scalar_oracle:v1"
ORACLE_TARGETS = [{"target_name": TARGET_NAME, "oracle_id": ORACLE_ID}]
ROW_SIDECAR_KEYS = {
    "target_order",
    "oracle_ids_by_target",
    "prediction_by_target",
    "target_by_name",
}


class _InjectedScalarRuntime:
    """A downstream scalar runtime with no native named-vector capability."""

    name = "injected_scalar"
    config = SimpleNamespace(root_observed_doc_ids=None)

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
        del f_init, g, output_dir
        values = [tree_root_target(tree) for tree in traces]
        observed = [float(value) for value in values if value is not None]
        return {
            "kind": "injected_scalar_f",
            "iteration": int(iteration),
            "value": sum(observed) / len(observed),
        }

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
        del g_init, traces, output_dir
        return {
            "kind": "injected_scalar_g",
            "iteration": int(iteration),
            "f_value": f["value"],
        }

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[Any],
    ) -> list[float | None]:
        del g
        value = None if not isinstance(f, Mapping) else float(f["value"])
        return [value for _tree in trees]

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        assert kind in {"f", "g"}
        assert artifact is None or isinstance(artifact, Mapping)


def _scalar_trees() -> list[TreeRecord]:
    trees: list[TreeRecord] = []
    for index, score in enumerate((0.2, 0.4, 0.7, 0.9)):
        leaves = (
            TreeNode(
                node_id=f"doc_{index}_left",
                unit_type="leaf",
                text=f"document {index} left policy",
                level=0,
                position=0,
            ),
            TreeNode(
                node_id=f"doc_{index}_right",
                unit_type="leaf",
                text=f"document {index} right policy",
                level=0,
                position=1,
            ),
        )
        trees.append(
            TreeRecord(
                tree_id=f"doc_{index}",
                doc_id=f"doc_{index}",
                root_label=float(score),
                nodes=leaves,
                metadata={
                    "split": "test",
                    "teacher_score_native": float(score),
                    "expert_score_for_objective": float(score),
                    "observed": True,
                    "propensity": 1.0,
                },
            )
        )
    return trees


def _scalar_case_spec(case: str) -> dict[str, Any]:
    if case in {"oracle", "classical_sketch"}:
        evaluation = make_hll_item_trees(
            n_trees=4,
            leaves_per_tree=4,
            leaf_unit_count=16,
            vocabulary_size=48,
            seed=301,
            split="test",
        )
        if case == "oracle":
            return {
                "family": "oracle",
                "train_data": [],
                "eval_data": evaluation,
                "backend_config": {"oracle_name": "hll_exact"},
                "axis": {"max_iterations": 0, "axis_value": 0},
            }
        return {
            "family": "classical_sketch",
            "train_data": evaluation,
            "eval_data": evaluation,
            "backend_config": {"sketch": "hll", "backend": "datasketches"},
            "axis": {"max_iterations": 2, "axis_value": 0},
        }

    trees = _scalar_trees()
    common: dict[str, Any] = {
        "train_data": trees,
        "eval_data": trees,
        "axis": {"max_iterations": 2, "axis_value": 0},
    }
    if case == "learnable_constant":
        return {**common, "family": "learnable_constant"}
    if case == "llm":
        return {
            **common,
            "family": "llm",
            "backend_config": {
                "default_prediction": 0.55,
                "audit_laws": False,
            },
        }
    if case == "injected":
        return {
            **common,
            "family": "injected_scalar",
            "backend_config": {"family_runtime": _InjectedScalarRuntime()},
        }
    raise AssertionError(f"unknown scalar test case {case!r}")


def _named_metric_key(key: str) -> bool:
    return key.startswith(f"{TARGET_NAME}_") or f"_{TARGET_NAME}_" in key


def _assert_named_row_projects_to_scalar(
    scalar_row: Mapping[str, Any],
    named_row: Mapping[str, Any],
) -> None:
    assert set(named_row).difference(scalar_row) == ROW_SIDECAR_KEYS
    for key, value in scalar_row.items():
        assert named_row[key] == value, key

    assert named_row["target_order"] == [TARGET_NAME]
    assert named_row["oracle_ids_by_target"] == {TARGET_NAME: ORACLE_ID}
    prediction = scalar_row["prediction_scalar"]
    assert named_row["prediction_by_target"] == (
        None if prediction is None else {TARGET_NAME: prediction}
    )
    teacher = scalar_row["teacher_score"]
    assert named_row["target_by_name"] == (None if teacher is None else {TARGET_NAME: teacher})


def _assert_named_split_projects_to_scalar(
    scalar_split: Mapping[str, Any],
    named_split: Mapping[str, Any],
) -> None:
    scalar_base = dict(scalar_split)
    named_base = dict(named_split)
    scalar_dimensions = dict(scalar_base.pop("per_dimension") or {})
    named_dimensions = dict(named_base.pop("per_dimension") or {})
    assert scalar_dimensions == {}
    assert named_base == scalar_base
    if scalar_split["n"] == 0 and scalar_split["mean_prediction"] is None:
        assert named_dimensions == {}
        return
    assert set(named_dimensions) == {TARGET_NAME}

    mirror = named_dimensions[TARGET_NAME]
    for key in (
        "internal_f_pearson",
        "internal_f_mae",
        "mean_prediction",
        "mean_teacher",
        "n",
    ):
        assert mirror[key] == scalar_split[key]


def _assert_named_result_projects_to_scalar(
    scalar: Any,
    named: Any,
    *,
    artifacts_equal: bool,
) -> None:
    assert named.status == scalar.status == "success"
    if artifacts_equal:
        assert named.artifacts.get("f") == scalar.artifacts.get("f")
        assert named.artifacts.get("g") == scalar.artifacts.get("g")

    for key, value in scalar.metrics.items():
        assert named.metrics[key] == value, key
    additions = set(named.metrics).difference(scalar.metrics)
    assert additions
    assert all(_named_metric_key(key) for key in additions)
    assert f"{TARGET_NAME}_internal_f_mae" in named.metrics
    assert f"{TARGET_NAME}_n" in named.metrics

    assert len(named.history) == len(scalar.history)
    for scalar_record, named_record in zip(scalar.history, named.history):
        for key, value in scalar_record.items():
            if key in {"f_artifact", "g_artifact", "split_metrics", "extra"}:
                continue
            assert named_record[key] == value, key
        if artifacts_equal:
            assert named_record["f_artifact"] == scalar_record["f_artifact"]
            assert named_record["g_artifact"] == scalar_record["g_artifact"]

        assert set(named_record["split_metrics"]) == set(scalar_record["split_metrics"])
        for split_name, scalar_split in scalar_record["split_metrics"].items():
            _assert_named_split_projects_to_scalar(
                scalar_split,
                named_record["split_metrics"][split_name],
            )

        scalar_extra = scalar_record["extra"]
        named_extra = named_record["extra"]
        assert set(named_extra) == set(scalar_extra)
        for key, value in scalar_extra.items():
            if key == "statistic":
                scalar_statistic = copy.deepcopy(value)
                named_statistic = copy.deepcopy(named_extra[key])
                scalar_schema = scalar_statistic["info"]["metadata"].pop("target_schema", None)
                named_schema = named_statistic["info"]["metadata"].pop("target_schema", None)
                assert scalar_schema is None
                if named_schema is not None:
                    assert named_schema["target_order"] == [TARGET_NAME]
                assert named_statistic == scalar_statistic
            elif key != "prediction_rows":
                assert named_extra[key] == value, key
        scalar_rows = scalar_extra.get("prediction_rows") or []
        named_rows = named_extra.get("prediction_rows") or []
        assert len(named_rows) == len(scalar_rows)
        for scalar_row, named_row in zip(scalar_rows, named_rows):
            _assert_named_row_projects_to_scalar(scalar_row, named_row)

    assert named.summary["definition"] == "single_shared_g_joint_vector_f_star"
    assert named.summary["target_order"] == [TARGET_NAME]
    assert named.summary["oracle_ids_by_target"] == {TARGET_NAME: ORACLE_ID}
    assert named.summary["target_schema_digest"]

    manifest = json.loads(Path(named.manifest_path).read_text(encoding="utf-8"))
    results = json.loads(Path(named.artifacts["results_json"]).read_text(encoding="utf-8"))
    for payload in (manifest, results):
        assert payload["target_order"] == [TARGET_NAME]
        assert payload["oracle_ids_by_target"] == {TARGET_NAME: ORACLE_ID}
        assert payload["target_schema_digest"] == named.summary["target_schema_digest"]
    assert results["paired_rows"]["named_vector_fields"] == {
        "key": "tree_id",
        "target_order": "target_order",
        "prediction": "prediction_by_target",
        "gold": "target_by_name",
        "oracle_ids": "oracle_ids_by_target",
        "split": "split",
    }


@pytest.mark.parametrize(
    "case",
    ["oracle", "classical_sketch", "learnable_constant", "llm", "injected"],
)
def test_one_oracle_target_lifts_scalar_runtime_without_changing_scalar_outputs(
    tmp_path: Path,
    case: str,
) -> None:
    scalar_spec = _scalar_case_spec(case)
    # The public compatibility operation changes no runtime knob or data
    # shape; it adds exactly one coordinate provenance record.
    named_spec = {**scalar_spec, "oracle_targets": ORACLE_TARGETS}

    scalar = fit(scalar_spec, output_dir=tmp_path / case / "scalar")
    named = fit(named_spec, output_dir=tmp_path / case / "named")

    _assert_named_result_projects_to_scalar(
        scalar,
        named,
        artifacts_equal=True,
    )
    if case == "classical_sketch":
        for result in (scalar, named):
            contract = result.summary["g_contract"]
            assert contract["train_g_call_count"] == 1
            assert contract["g_update_count"] == 0
            assert contract["learned_this_run"] is False
            assert contract["operator"] == "trainable_not_updated_this_run"


@pytest.mark.parametrize("case", ["learnable_constant", "injected"])
def test_multiple_oracle_targets_still_reject_scalar_only_runtimes(
    tmp_path: Path,
    case: str,
) -> None:
    spec = _scalar_case_spec(case)
    with pytest.raises(ValueError):
        fit(
            {
                **spec,
                "oracle_targets": [
                    *ORACLE_TARGETS,
                    {
                        "target_name": "second_score",
                        "oracle_id": "test_second_oracle:v1",
                    },
                ],
            },
            output_dir=tmp_path / case,
        )


def _tiny_operator_spec(family: str) -> dict[str, Any]:
    train = make_markov_changepoint_trees(
        n_trees=4,
        doc_tokens=16,
        leaf_unit_count=4,
        vocabulary_size=32,
        seed=401,
        split="train",
    )
    evaluation = make_markov_changepoint_trees(
        n_trees=2,
        doc_tokens=16,
        leaf_unit_count=4,
        vocabulary_size=32,
        seed=402,
        split="test",
    )
    backend: dict[str, Any] = {
        "embedding_dim": 4,
        "hidden_channels": 2,
        "n_modes": 2,
        "n_layers": 1,
        "head_hidden_dim": 4,
        "epochs_per_iteration": 1,
        "batch_size": 4,
        "learning_rate": 0.01,
        "device": "cpu",
        "seed": 403,
        "normalize_targets": False,
    }
    if family == "neural_operator":
        backend["operator_kind"] = "conv1d"
    return {
        "space_kind": "markov_count_state.v1",
        "family": family,
        "schedule": "fg",
        "train_data": train,
        "eval_data": evaluation,
        "backend_config": backend,
        "axis": {"max_iterations": 2, "axis_value": 0},
    }


def _prediction_scalars(result: Any) -> list[float | None]:
    return [row["prediction_scalar"] for row in result.history[-1]["extra"]["prediction_rows"]]


@pytest.mark.parametrize("family", ["neural_operator", "fno"])
def test_scalar_and_named_singleton_checkpoints_resume_in_both_directions(
    tmp_path: Path,
    family: str,
) -> None:
    torch = pytest.importorskip("torch")
    if family == "fno":
        pytest.importorskip("neuralop")

    scalar_spec = _tiny_operator_spec(family)
    named_spec = {**scalar_spec, "oracle_targets": ORACLE_TARGETS}
    scalar = fit(scalar_spec, output_dir=tmp_path / family / "scalar_train")
    named = fit(named_spec, output_dir=tmp_path / family / "named_train")

    for kind in ("f", "g"):
        scalar_artifact = scalar.artifacts[kind]
        named_artifact = named.artifacts[kind]
        assert scalar_artifact["output_dim"] == named_artifact["output_dim"] == 1
        assert scalar_artifact["loss"] == named_artifact["loss"]
        assert scalar_artifact["target_center"] == named_artifact["target_center"]
        assert scalar_artifact["target_scale"] == named_artifact["target_scale"]
        scalar_state = torch.load(
            scalar_artifact["weights_path"],
            map_location="cpu",
            weights_only=True,
        )
        named_state = torch.load(
            named_artifact["weights_path"],
            map_location="cpu",
            weights_only=True,
        )
        assert scalar_state.keys() == named_state.keys()
        assert all(torch.equal(scalar_state[key], named_state[key]) for key in scalar_state)

    named_from_scalar = fit(
        {
            **named_spec,
            "initial_artifacts": {
                "f": scalar.artifacts["f"],
                "g": scalar.artifacts["g"],
            },
            "axis": {"max_iterations": 0, "axis_value": 0},
        },
        output_dir=tmp_path / family / "named_from_scalar",
    )
    scalar_from_named = fit(
        {
            **scalar_spec,
            "initial_artifacts": {
                "f": named.artifacts["f"],
                "g": named.artifacts["g"],
            },
            "axis": {"max_iterations": 0, "axis_value": 0},
        },
        output_dir=tmp_path / family / "scalar_from_named",
    )

    assert _prediction_scalars(named_from_scalar) == _prediction_scalars(scalar)
    assert _prediction_scalars(scalar_from_named) == _prediction_scalars(named)
    assert named_from_scalar.artifacts["f"] == scalar.artifacts["f"]
    assert named_from_scalar.artifacts["g"] == scalar.artifacts["g"]
    assert scalar_from_named.artifacts["f"] == named.artifacts["f"]
    assert scalar_from_named.artifacts["g"] == named.artifacts["g"]

    scalar_projection = fit(
        {
            **scalar_spec,
            "initial_artifacts": {
                "f": scalar.artifacts["f"],
                "g": scalar.artifacts["g"],
            },
            "axis": {"max_iterations": 0, "axis_value": 0},
        },
        output_dir=tmp_path / family / "scalar_projection",
    )
    _assert_named_result_projects_to_scalar(
        scalar_projection,
        named_from_scalar,
        artifacts_equal=True,
    )
