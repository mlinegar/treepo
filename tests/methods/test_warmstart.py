"""Artifacts are a real warmstart contract, not decorative metadata.

A neural-operator artifact carries ``weights_path``; handing it to a fresh
family (via ``initial_artifacts`` in a spec, or ``f=``/``g=`` at scoring
time) must reconstruct the trained model exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from treepo import fit
from treepo.methods.families import resolve_family
from treepo.methods.fixtures import make_markov_changepoint_trees


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


def _markov_trees(n_trees: int, *, seed: int, split: str) -> list[Any]:
    return make_markov_changepoint_trees(
        n_trees=n_trees,
        doc_tokens=32,
        leaf_unit_count=8,
        vocabulary_size=64,
        seed=seed,
        split=split,
    )


def test_fresh_family_scores_from_artifact_weights(tmp_path: Path) -> None:
    train = _markov_trees(6, seed=61, split="train")
    eval_trees = _markov_trees(4, seed=62, split="test")

    trained = resolve_family("fno", _tiny_fno_config())
    f_artifact = trained.train_f(
        f_init=None, g=None, traces=train, output_dir=tmp_path / "f", iteration=1
    )
    g_artifact = trained.train_g(
        g_init=None, f=f_artifact, traces=train, output_dir=tmp_path / "g", iteration=2
    )
    assert f_artifact["weights_path"] is not None
    assert Path(f_artifact["weights_path"]).exists()
    expected = trained.score_roots_with_f(f=f_artifact, g=g_artifact, trees=eval_trees)

    fresh = resolve_family("fno", _tiny_fno_config())
    got = fresh.score_roots_with_f(f=g_artifact, g=None, trees=eval_trees)

    assert got == pytest.approx(expected)


def test_fit_initial_artifacts_reproduce_predictions_without_training(tmp_path: Path) -> None:
    train = _markov_trees(6, seed=63, split="train")
    eval_trees = _markov_trees(4, seed=64, split="test")

    first = fit(
        {
            "family": "fno",
            "train_data": train,
            "eval_data": eval_trees,
            "backend_config": {**_tiny_fno_config(), "output_dir": str(tmp_path / "run1")},
            "axis": {"max_iterations": 2, "axis_value": 0},
        },
    )
    assert first.status == "success"

    resumed = fit(
        {
            "family": "fno",
            "train_data": train,
            "eval_data": eval_trees,
            "initial_artifacts": {
                "f": first.artifacts["f"],
                "g": first.artifacts["g"],
            },
            "backend_config": {**_tiny_fno_config(), "output_dir": str(tmp_path / "run2")},
            "axis": {"max_iterations": 0, "axis_value": 0},
        },
    )
    assert resumed.status == "success"

    first_rows = _final_prediction_rows(first)
    resumed_rows = _final_prediction_rows(resumed)
    assert set(resumed_rows) == set(first_rows)
    for tree_id, prediction in resumed_rows.items():
        assert prediction == pytest.approx(first_rows[tree_id])


def _final_prediction_rows(result: Any) -> dict[str, Any]:
    import json

    path = result.artifacts["prediction_records"][-1]
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {row["tree_id"]: row["prediction"] for row in rows}


def test_warmstart_rejects_missing_weights_file(tmp_path: Path) -> None:
    family = resolve_family("fno", _tiny_fno_config())
    with pytest.raises(ValueError, match="missing"):
        family.train_f(
            f_init={"weights_path": str(tmp_path / "gone.pt"), "output_dim": 1},
            g=None,
            traces=_markov_trees(2, seed=65, split="train"),
            output_dir=tmp_path,
            iteration=1,
        )


def test_warmstart_rejects_operator_kind_mismatch(tmp_path: Path) -> None:
    trained = resolve_family("fno", _tiny_fno_config())
    artifact = trained.train_f(
        f_init=None,
        g=None,
        traces=_markov_trees(3, seed=66, split="train"),
        output_dir=tmp_path,
        iteration=1,
    )
    other = resolve_family(
        "neural_operator",
        {**_tiny_fno_config(), "operator_kind": "conv1d", "conv_kernel_size": 3},
    )
    with pytest.raises(ValueError, match="operator_kind mismatch"):
        other.score_roots_with_f(f=artifact, g=None, trees=_markov_trees(1, seed=67, split="test"))
