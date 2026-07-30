"""Exact metadata state laws keep the task/root readout scalar."""

from __future__ import annotations

from pathlib import Path

import pytest

from treepo import fit
from treepo.methods._fno_config import NeuralOperatorFamilyConfig
from treepo.methods._fno_law_state import metadata_law_state_targets
from treepo.tree import TreeNode, TreeRecord

STATE_KEY = "__exact_additive_state__"


def _tree(doc_id: str, *, root_observed: bool, expose_state: bool) -> TreeRecord:
    signed = [-1.0, 0.0, 1.0, 0.5]
    mass = [1.0, 1.0, 1.0, 1.0]

    def state(indices: list[int]) -> list[float]:
        # Coordinate zero is the stable fixed task readout.  The remaining
        # coordinates retain the exact additive (S, N) witness.
        signed_count = sum(signed[i] for i in indices)
        total_mass = sum(mass[i] for i in indices)
        rile = 0.5 if total_mass <= 0 else 0.5 * (1.0 + signed_count / total_mass)
        return [rile, signed_count / 8.0, total_mass / 8.0]

    def metadata(indices: list[int]) -> dict[str, object]:
        return {STATE_KEY: state(indices)} if expose_state else {}

    leaves = [
        TreeNode(
            node_id=f"{doc_id}_l{i}",
            unit_type="leaf",
            text=f"{doc_id} policy sentence {i}",
            level=0,
            position=i,
            metadata=metadata([i]),
        )
        for i in range(4)
    ]
    merges = [
        TreeNode(
            node_id=f"{doc_id}_m0",
            unit_type="merge",
            level=1,
            position=0,
            left_child_id=f"{doc_id}_l0",
            right_child_id=f"{doc_id}_l1",
            metadata=metadata([0, 1]),
        ),
        TreeNode(
            node_id=f"{doc_id}_m1",
            unit_type="merge",
            level=1,
            position=1,
            left_child_id=f"{doc_id}_l2",
            right_child_id=f"{doc_id}_l3",
            metadata=metadata([2, 3]),
        ),
    ]
    root_scalar = 0.5 + sum(signed) / (2.0 * sum(mass))
    # The partial/local-only documents deliberately do not reveal a root state
    # key: their two child subtrees are closed, but the full document is not.
    root = TreeNode(
        node_id=f"{doc_id}_root",
        unit_type="root",
        level=2,
        position=0,
        left_child_id=f"{doc_id}_m0",
        right_child_id=f"{doc_id}_m1",
        metadata=({STATE_KEY: state([0, 1, 2, 3])} if expose_state and root_observed else {}),
    )
    return TreeRecord(
        tree_id=doc_id,
        doc_id=doc_id,
        root_label=root_scalar,
        nodes=(*leaves, *merges, root),
        metadata={"split": "train", "teacher_score_native": root_scalar},
    )


def _records() -> list[TreeRecord]:
    return [
        _tree("root_0", root_observed=True, expose_state=False),
        _tree("root_1", root_observed=True, expose_state=False),
        _tree("local_0", root_observed=False, expose_state=True),
        _tree("local_1", root_observed=False, expose_state=True),
    ]


def _config(output_dir: Path, *, micro_batch_size: int | None) -> dict[str, object]:
    return {
        "family": "neural_operator",
        "train_data": _records(),
        "eval_data": _records(),
        "root_observed_doc_ids": ["root_0", "root_1"],
        "axis": {"axis_kind": "leaf_count", "axis_value": 4, "max_iterations": 2},
        "backend_config": {
            "operator_kind": "conv1d",
            "embedding_dim": 8,
            "hidden_channels": 4,
            "n_layers": 1,
            "head_hidden_dim": 8,
            "epochs_per_iteration": 1,
            "batch_size": 4,
            "micro_batch_size": micro_batch_size,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "grad_clip_norm": None,
            "device": "cpu",
            "seed": 17,
            "normalize_targets": False,
            "root_readout": "root_state",
            "law_state_target_key": STATE_KEY,
            "law_state_target_dim": 3,
            "law_state_root_readout": "first_coordinate",
            "law_state_leaf_propensity": 1.0,
            "law_state_merge_propensity": 1.0,
            "output_dir": str(output_dir),
            "objective": {
                "objective_family": "scalar_root_plus_exact_state",
                "local_law_estimator": "oracle_state",
                "root_share": 0.5,
                "local_law_weight": 0.5,
                "local_law_component_weights": {"c1": 0.25, "c3": 0.25},
            },
        },
    }


def _endpoint_config(
    output_dir: Path,
    *,
    endpoint: str,
    learning_rate: float,
) -> dict[str, object]:
    payload = _config(output_dir, micro_batch_size=1)
    backend = payload["backend_config"]
    assert isinstance(backend, dict)
    backend["learning_rate"] = float(learning_rate)
    objective = backend["objective"]
    assert isinstance(objective, dict)
    if endpoint == "root":
        payload["root_observed_doc_ids"] = ["root_0", "root_1"]
        objective.update(
            {
                "local_law_estimator": "none",
                "root_share": 1.0,
                "local_law_weight": 0.0,
                "local_law_component_weights": {},
            }
        )
    elif endpoint == "local":
        payload["root_observed_doc_ids"] = []
        objective.update(
            {
                "local_law_estimator": "oracle_state",
                "root_share": 0.0,
                "local_law_weight": 1.0,
                "local_law_component_weights": {"c1": 0.5, "c3": 0.5},
            }
        )
    else:  # pragma: no cover - test helper guard
        raise ValueError(endpoint)
    return payload


def test_metadata_state_extractor_is_sparse_and_includes_explicit_root_c3() -> None:
    torch = pytest.importorskip("torch")
    record = _tree("doc", root_observed=True, expose_state=True)
    config = NeuralOperatorFamilyConfig(
        law_state_target_key=STATE_KEY,
        law_state_target_dim=3,
        law_state_leaf_propensity=0.25,
        law_state_merge_propensity=0.5,
    )
    rows = metadata_law_state_targets([record], config, torch=torch, device="cpu")
    assert rows is not None
    row = rows[0]
    assert row.values.shape == (7, 3)
    assert row.observed.tolist() == [True] * 7
    assert row.propensity.tolist() == pytest.approx([0.25] * 4 + [0.5] * 3)
    assert row.values[-1].tolist() == pytest.approx([0.5625, 0.0625, 0.5])


def test_fixed_coordinate_readout_is_exact_and_freezes_learned_head() -> None:
    torch = pytest.importorskip("torch")
    from treepo.methods._neural_operator_core import NeuralOperatorFamily

    family = NeuralOperatorFamily(
        NeuralOperatorFamilyConfig(
            operator_kind="conv1d",
            embedding_dim=4,
            hidden_channels=4,
            normalize_targets=False,
            law_state_root_readout="first_coordinate",
        )
    )
    family._ensure_model(output_dim=1)
    assert family._model is not None
    states = torch.tensor([[0.25, 9.0, 4.0, -3.0]])
    assert family._model._read(states).tolist() == [[0.25]]
    family._set_trainable(train_f=True, train_g=False)
    assert not any(parameter.requires_grad for parameter in family._model.g.parameters())
    assert not any(parameter.requires_grad for parameter in family._model.readout.parameters())
    family._set_trainable(train_f=False, train_g=True)
    assert any(parameter.requires_grad for parameter in family._model.g.parameters())
    assert family._model.g_module_names == ("g",)
    assert not any(parameter.requires_grad for parameter in family._model.readout.parameters())


@pytest.mark.parametrize("endpoint", ["root", "local"])
def test_fixed_coordinate_endpoints_train_state_but_not_learned_readout(
    tmp_path: Path,
    endpoint: str,
) -> None:
    torch = pytest.importorskip("torch")
    frozen = fit(
        _endpoint_config(tmp_path / endpoint / "frozen", endpoint=endpoint, learning_rate=0.0)
    )
    trained = fit(
        _endpoint_config(tmp_path / endpoint / "trained", endpoint=endpoint, learning_rate=0.02)
    )
    frozen_state = torch.load(
        frozen.artifacts["g"]["weights_path"], map_location="cpu", weights_only=True
    )
    trained_state = torch.load(
        trained.artifacts["g"]["weights_path"], map_location="cpu", weights_only=True
    )
    readout_keys = [name for name in frozen_state if name.startswith("readout.")]
    state_keys = [name for name in frozen_state if not name.startswith("readout.")]
    assert readout_keys
    assert state_keys
    assert all(torch.equal(frozen_state[name], trained_state[name]) for name in readout_keys)
    assert any(not torch.equal(frozen_state[name], trained_state[name]) for name in state_keys)


def test_fit_keeps_scalar_root_and_trains_disjoint_exact_state_rows_with_microbatch(
    tmp_path: Path,
) -> None:
    result = fit(_config(tmp_path, micro_batch_size=1))
    assert result.status == "success"
    artifact = result.artifacts["g"]
    assert artifact["output_dim"] == 1
    assert artifact["task_readout"] == {
        "kind": "fixed_state_coordinate",
        "state_coordinate": 0,
        "output_dim": 1,
        "learned_readout_trainable": False,
        "training_scale": "configured_normalized_task_coordinate",
        "public_prediction_clamped_to_target_bounds": True,
    }
    assert artifact["node_supervision"]["n_root_rows"] == 2
    state = artifact["law_state_supervision"]
    assert state["n_leaf_rows"] == 8
    assert state["n_merge_rows_including_root"] == 4
    assert state["root_merge_is_separate_from_root_task_loss"] is True
    assert state["c2_status"] == "not_applicable_one_pass_no_recompressor"
    assert state["c3_scope"] == "observed_realized_canonical_merges_not_universal_closure"
    components = artifact["objective_components"]
    assert components["exact_gradient_accumulation"] is True
    assert components["outer_batch_size"] == 4
    assert components["micro_batch_size"] == 1
    assert components["combined"] == pytest.approx(
        0.5 * components["root"] + 0.25 * components["c1"] + 0.25 * components["c3"],
        abs=1.0e-7,
    )
    prediction = result.history[-1]["extra"]["prediction_rows"][0]
    assert isinstance(prediction["prediction_scalar"], float)
    assert not isinstance(prediction["prediction"], list)


def test_exact_law_microbatch_gradient_matches_single_pass_outer_batch(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    full = fit(_config(tmp_path / "full", micro_batch_size=None))
    micro = fit(_config(tmp_path / "micro", micro_batch_size=1))
    full_state = torch.load(
        full.artifacts["g"]["weights_path"], map_location="cpu", weights_only=True
    )
    micro_state = torch.load(
        micro.artifacts["g"]["weights_path"], map_location="cpu", weights_only=True
    )
    assert set(full_state) == set(micro_state)
    for name in full_state:
        assert torch.allclose(full_state[name], micro_state[name], atol=2.0e-7, rtol=1.0e-6), name
    full_components = full.artifacts["g"]["objective_components"]
    micro_components = micro.artifacts["g"]["objective_components"]
    for name in ("combined", "root", "c1", "c3"):
        assert micro_components[name] == pytest.approx(full_components[name], abs=2.0e-7)
