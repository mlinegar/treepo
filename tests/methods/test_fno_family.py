from __future__ import annotations

import math
from pathlib import Path

import pytest

from treepo import ComposableStatistic, fit
from treepo.methods.families import list_families, resolve_family
from treepo.methods.fixtures import make_markov_changepoint_trees
from treepo.methods.fno import (
    FNOFamily,
    FNOFamilyConfig,
    NeuralOperatorFamily,
    NeuralOperatorFamilyConfig,
)


def _tiny_fno_config() -> dict[str, object]:
    return {
        "embedding_dim": 8,
        "hidden_channels": 4,
        "n_modes": 2,
        "n_layers": 1,
        "head_hidden_dim": 8,
        "epochs_per_iteration": 1,
        "batch_size": 4,
        "learning_rate": 0.01,
        "device": "cpu",
        "seed": 3,
    }


def test_fno_family_is_builtin() -> None:
    assert "fno" in list_families()
    family = resolve_family("fno", _tiny_fno_config())
    assert isinstance(family, FNOFamily)
    assert isinstance(family.config, FNOFamilyConfig)
    assert family.name == "fno"
    assert family.operator_kind == "fno"
    assert family.config.embedding_dim == 8
    assert family.config.embedding_salt == "treepo_fno"


def test_neural_operator_family_is_builtin_with_fno_option() -> None:
    assert "neural_operator" in list_families()
    family = resolve_family(
        "neural_operator",
        {
            **_tiny_fno_config(),
            "operator_kind": "fno",
        },
    )
    assert isinstance(family, NeuralOperatorFamily)
    assert not isinstance(family, FNOFamily)
    assert isinstance(family.config, NeuralOperatorFamilyConfig)
    assert family.name == "neural_operator"
    assert family.operator_kind == "fno"


def test_neural_operator_family_rejects_unknown_operator_kind() -> None:
    with pytest.raises(ValueError, match="does not support operator_kind='fourier'"):
        resolve_family(
            "neural_operator",
            {
                **_tiny_fno_config(),
                "operator_kind": "fourier",
            },
        )


def test_neural_operator_family_is_builtin_with_conv1d_option() -> None:
    family = resolve_family(
        "neural_operator",
        {
            **_tiny_fno_config(),
            "operator_kind": "conv1d",
            "conv_kernel_size": 3,
        },
    )
    assert isinstance(family, NeuralOperatorFamily)
    assert family.name == "neural_operator"
    assert family.operator_kind == "conv1d"


def test_neural_operator_family_accepts_neuralop_model_names() -> None:
    family = resolve_family(
        "neural_operator",
        {
            **_tiny_fno_config(),
            "operator_kind": "tfno",
        },
    )
    assert isinstance(family, NeuralOperatorFamily)
    assert family.operator_kind == "tfno"
    family._ensure_model()


def test_neural_operator_statistic_is_available_after_training(tmp_path: Path) -> None:
    family = resolve_family(
        "fno",
        {
            **_tiny_fno_config(),
            "numeric_transition_state_weight": 0.05,
        },
    )
    assert family.as_statistic() is None
    train = make_markov_changepoint_trees(
        n_trees=5,
        doc_tokens=32,
        leaf_unit_count=8,
        vocabulary_size=64,
        seed=41,
        split="train",
    )
    eval_trees = make_markov_changepoint_trees(
        n_trees=3,
        doc_tokens=32,
        leaf_unit_count=8,
        vocabulary_size=64,
        seed=42,
        split="test",
    )
    f_artifact = family.train_f(
        f_init=None,
        g=None,
        traces=train,
        output_dir=tmp_path / "f",
        iteration=1,
    )
    g_artifact = family.train_g(
        g_init=None,
        f=f_artifact,
        traces=train,
        output_dir=tmp_path / "g",
        iteration=2,
    )
    statistic = family.as_statistic(f=f_artifact, g=g_artifact)
    assert isinstance(statistic, ComposableStatistic)
    assert statistic.info.exact is False
    assert statistic.info.state_kind == "fno"

    via_family = family.score_roots_with_f(f=f_artifact, g=g_artifact, trees=eval_trees)
    via_statistic = [statistic.predict_tree(tree) for tree in eval_trees]
    assert via_statistic == pytest.approx(via_family)

    rows = statistic.local_law_rows(eval_trees)
    assert rows
    assert {row.metadata["check"] for row in rows} == {"numeric_transition_state"}


def test_fno_route_rejects_non_fno_operator_kind() -> None:
    try:
        resolve_family("fno", {"operator_kind": "conv1d"})
    except ValueError as exc:
        assert "family='fno' only supports operator_kind='fno'" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected fno route to reject operator_kind='conv1d'")


def test_neural_operator_rejects_unknown_operator_kind_with_available_names() -> None:
    try:
        resolve_family("neural_operator", {"operator_kind": "deeponet"})
    except ValueError as exc:
        message = str(exc)
        assert "supported operator_kind values:" in message
        assert "fno" in message
        assert "tfno" in message
    else:  # pragma: no cover
        raise AssertionError("expected unsupported operator_kind to fail")


def test_neural_operator_rejects_geometry_query_neuralop_kind_for_sequence_adapter() -> None:
    family = resolve_family(
        "neural_operator",
        {
            **_tiny_fno_config(),
            "operator_kind": "gino",
        },
    )
    try:
        family._ensure_model()
    except ValueError as exc:
        message = str(exc)
        assert "one embedded leaf-sequence tensor" in message
        assert "downstream family" in message
    else:  # pragma: no cover
        raise AssertionError("expected GINO to require a downstream query-input family")


def test_neural_operator_reports_required_neuralop_constructor_kwargs() -> None:
    family = resolve_family(
        "neural_operator",
        {
            **_tiny_fno_config(),
            "operator_kind": "uqno",
        },
    )
    try:
        family._ensure_model()
    except ValueError as exc:
        assert "operator_kwargs" in str(exc)
        assert "base_model" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected UQNO without base_model to fail clearly")


def test_fit_runs_builtin_fno_on_markov_fixture(tmp_path: Path) -> None:
    train = make_markov_changepoint_trees(
        n_trees=6,
        doc_tokens=32,
        leaf_unit_count=8,
        vocabulary_size=64,
        seed=11,
        split="train",
    )
    eval_trees = make_markov_changepoint_trees(
        n_trees=4,
        doc_tokens=32,
        leaf_unit_count=8,
        vocabulary_size=64,
        seed=12,
        split="test",
    )

    result = fit(
        {
            "family": "fno",
            "train_data": train,
            "eval_data": eval_trees,
            "backend_config": {
                **_tiny_fno_config(),
                "output_dir": str(tmp_path),
            },
            "axis": {"max_iterations": 3, "axis_value": 0},
        },
    )

    assert result.status == "success"
    assert result.summary["family"] == "fno"
    assert result.artifacts["f"]["kind"] == "treepo_fno"
    assert result.artifacts["g"]["kind"] == "treepo_fno_g"
    assert result.artifacts["g"]["trained"] == "g"
    assert result.artifacts["statistic"]["info"]["state_kind"] == "fno"
    assert result.metrics["n"] == 4.0
    assert result.metrics["internal_f_mae"] >= 0.0
    assert result.artifacts["prediction_records"]


def test_fit_runs_neural_operator_fno_on_markov_fixture(tmp_path: Path) -> None:
    train = make_markov_changepoint_trees(
        n_trees=6,
        doc_tokens=32,
        leaf_unit_count=8,
        vocabulary_size=64,
        seed=21,
        split="train",
    )
    eval_trees = make_markov_changepoint_trees(
        n_trees=4,
        doc_tokens=32,
        leaf_unit_count=8,
        vocabulary_size=64,
        seed=22,
        split="test",
    )

    result = fit(
        {
            "family": "neural_operator",
            "train_data": train,
            "eval_data": eval_trees,
            "backend_config": {
                **_tiny_fno_config(),
                "operator_kind": "fno",
                "output_dir": str(tmp_path),
            },
            "axis": {"max_iterations": 3, "axis_value": 0},
        },
    )

    assert result.status == "success"
    assert result.summary["family"] == "neural_operator"
    assert result.artifacts["f"]["kind"] == "treepo_fno"
    assert result.artifacts["f"]["operator_kind"] == "fno"
    assert result.artifacts["g"]["trained"] == "g"
    assert result.summary["statistic"]["state_kind"] == "fno"
    assert result.metrics["n"] == 4.0
    assert result.metrics["internal_f_mae"] >= 0.0


def test_neural_operator_fno_matches_fno_alias_on_markov_fixture(tmp_path: Path) -> None:
    train = make_markov_changepoint_trees(
        n_trees=6,
        doc_tokens=32,
        leaf_unit_count=8,
        vocabulary_size=64,
        seed=24,
        split="train",
    )
    eval_trees = make_markov_changepoint_trees(
        n_trees=4,
        doc_tokens=32,
        leaf_unit_count=8,
        vocabulary_size=64,
        seed=25,
        split="test",
    )
    common_backend = {
        **_tiny_fno_config(),
        "operator_kind": "fno",
    }

    fno_result = fit(
        {
            "family": "fno",
            "train_data": train,
            "eval_data": eval_trees,
            "backend_config": {
                **common_backend,
                "output_dir": str(tmp_path / "fno"),
            },
            "axis": {"max_iterations": 3, "axis_value": 0},
        },
    )
    generic_result = fit(
        {
            "family": "neural_operator",
            "train_data": train,
            "eval_data": eval_trees,
            "backend_config": {
                **common_backend,
                "output_dir": str(tmp_path / "neural_operator"),
            },
            "axis": {"max_iterations": 3, "axis_value": 0},
        },
    )

    assert fno_result.status == generic_result.status == "success"
    assert fno_result.artifacts["f"]["kind"] == "treepo_fno"
    assert generic_result.artifacts["f"]["kind"] == "treepo_fno"
    assert fno_result.artifacts["f"]["operator_kind"] == "fno"
    assert generic_result.artifacts["f"]["operator_kind"] == "fno"
    assert fno_result.artifacts["g"]["trained"] == "g"
    assert generic_result.artifacts["g"]["trained"] == "g"
    assert fno_result.metrics["n"] == generic_result.metrics["n"] == 4.0
    assert math.isfinite(float(fno_result.metrics["internal_f_mae"]))
    assert math.isfinite(float(generic_result.metrics["internal_f_mae"]))


def test_neural_operator_compares_dense_official_kinds_and_conv1d_on_markov_fixture(tmp_path: Path) -> None:
    train = make_markov_changepoint_trees(
        n_trees=8,
        doc_tokens=32,
        leaf_unit_count=8,
        vocabulary_size=64,
        seed=31,
        split="train",
    )
    eval_trees = make_markov_changepoint_trees(
        n_trees=5,
        doc_tokens=32,
        leaf_unit_count=8,
        vocabulary_size=64,
        seed=32,
        split="test",
    )
    by_kind = {}
    for operator_kind in ("fno", "tfno", "uno", "conv1d"):
        result = fit(
            {
                "family": "neural_operator",
                "train_data": train,
                "eval_data": eval_trees,
                "backend_config": {
                    **_tiny_fno_config(),
                    "operator_kind": operator_kind,
                    "conv_kernel_size": 3,
                    "output_dir": str(tmp_path / operator_kind),
                },
                "axis": {"max_iterations": 3, "axis_value": 0},
            },
        )
        assert result.status == "success"
        assert result.artifacts["f"]["operator_kind"] == operator_kind
        assert result.artifacts["g"]["trained"] == "g"
        assert result.metrics["n"] == 5.0
        assert math.isfinite(float(result.metrics["internal_f_mae"]))
        by_kind[operator_kind] = result.metrics["internal_f_mae"]

    assert set(by_kind) == {"fno", "tfno", "uno", "conv1d"}
