"""A declared ObjectiveSpec is executed, not just recorded.

The neural-operator families adopt a resolved ``ObjectiveSpec`` through
``configure_objective`` and minimize the convex combination
``root_share * root + sum_c share_c * law_c``, with every law channel routed
through the canonical depth-discounted objective in
:mod:`treepo.training.local_law`. ``fit()`` refuses law-bearing objectives on
families that cannot execute them.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from treepo import fit
from treepo.local_law import LawKind
from treepo.methods.families import resolve_family
from treepo.methods.fixtures import make_markov_changepoint_trees
from treepo.objective import ObjectiveSpec


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


def _law_objective(**overrides: Any) -> ObjectiveSpec:
    kwargs: dict[str, Any] = {
        "objective_family": "root_plus_laws",
        "local_law_estimator": "oracle_state",
        "root_share": 0.5,
        "local_law_component_weights": {"c1": 0.25, "c3": 0.25},
    }
    kwargs.update(overrides)
    return ObjectiveSpec(**kwargs)


def _markov_trees(n_trees: int, *, seed: int, split: str) -> list[Any]:
    return make_markov_changepoint_trees(
        n_trees=n_trees,
        doc_tokens=32,
        leaf_unit_count=8,
        vocabulary_size=64,
        seed=seed,
        split=split,
    )


def test_assembled_loss_is_the_convex_depth_discounted_combination() -> None:
    family = resolve_family("fno", _tiny_fno_config())
    family.configure_objective(
        _law_objective(terms={"local_law_corrected": {"gamma_depth": 0.5}})
    )
    torch = family._torch

    proxy = torch.tensor([1.0, 2.0, 3.0, 4.0])
    depths = torch.tensor([1, 1, 0, 0])
    is_leaf = torch.tensor([True, True, False, False])
    root_loss = torch.tensor(2.0)

    loss = family._assemble_loss(root_loss, (proxy, depths, is_leaf))

    # C1 channel (leaves, depth 1, gamma 0.5): (0.5*1 + 0.5*2) / 1.0 = 1.5.
    # C3 channel (merges, depth 0): (3 + 4) / 2 = 3.5.
    # Total: 0.5*2.0 + 0.25*1.5 + 0.25*3.5 = 2.25.
    assert float(loss) == pytest.approx(2.25)


def test_root_only_objective_scales_root_loss_only() -> None:
    family = resolve_family("fno", _tiny_fno_config())
    family.configure_objective(ObjectiveSpec())
    torch = family._torch

    loss = family._assemble_loss(torch.tensor(2.0), None)

    assert float(loss) == pytest.approx(2.0)


def test_fit_executes_law_bearing_objective_on_markov_fixture(tmp_path: Path) -> None:
    result = fit(
        {
            "family": "fno",
            "train_data": _markov_trees(6, seed=51, split="train"),
            "eval_data": _markov_trees(3, seed=52, split="test"),
            "backend_config": {
                **_tiny_fno_config(),
                "output_dir": str(tmp_path),
                "objective": {
                    "objective_family": "root_plus_laws",
                    "local_law_estimator": "oracle_state",
                    "root_share": 0.5,
                    "local_law_component_weights": {"c1": 0.25, "c3": 0.25},
                    "terms": {"local_law_corrected": {"gamma_depth": 0.9}},
                },
            },
            "axis": {"max_iterations": 2, "axis_value": 0},
        },
    )

    assert result.status == "success"
    for side in ("f", "g"):
        artifact = result.artifacts[side]
        assert artifact["objective_executed"] is True
        recorded = artifact["objective"]
        assert recorded["root_share"] == pytest.approx(0.5)
        assert recorded["local_law_estimator"] == "oracle_state"
        assert recorded["local_law_weight"] == pytest.approx(0.5)
        weights = recorded["local_law_component_weights"]
        assert weights[LawKind.C1_LEAF.value] == pytest.approx(0.25)
        assert weights[LawKind.C3_MERGE.value] == pytest.approx(0.25)


def test_fit_records_root_only_objective_as_executed_with_zero_law_weight(
    tmp_path: Path,
) -> None:
    result = fit(
        {
            "family": "fno",
            "train_data": _markov_trees(6, seed=53, split="train"),
            "eval_data": _markov_trees(3, seed=54, split="test"),
            "backend_config": {
                **_tiny_fno_config(),
                "output_dir": str(tmp_path),
                "objective": {"objective_family": "root_only"},
            },
            "axis": {"max_iterations": 2, "axis_value": 0},
        },
    )

    assert result.status == "success"
    for side in ("f", "g"):
        artifact = result.artifacts[side]
        assert artifact["objective_executed"] is True
        assert artifact["objective"]["local_law_weight"] == pytest.approx(0.0)
        assert artifact["objective"]["root_share"] == pytest.approx(1.0)


class _RootOnlyRuntime:
    """Minimal FamilyRuntime without a configure_objective hook."""

    name = "root_only_stub"

    def train_f(self, *, f_init, g, traces, output_dir, iteration):
        return {"kind": "stub", "trained": "f"}

    def train_g(self, *, g_init, f, traces, output_dir, iteration):
        return {"kind": "stub", "trained": "g"}

    def score_roots_with_f(self, *, f, g, trees):
        return [0.0 for _ in trees]

    def validate_artifact(self, *, kind, artifact):
        return None


def test_fit_rejects_law_objective_on_family_without_hook(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provenance-only"):
        fit(
            {
                "family": "root_only_stub",
                "train_data": _markov_trees(2, seed=55, split="train"),
                "eval_data": _markov_trees(2, seed=56, split="test"),
                "backend_config": {
                    "family_runtime": _RootOnlyRuntime(),
                    "output_dir": str(tmp_path),
                    "objective": _law_objective(),
                },
                "axis": {"max_iterations": 1, "axis_value": 0},
            },
        )


def test_objective_conflicts_with_legacy_state_weight() -> None:
    family = resolve_family(
        "fno",
        {
            **_tiny_fno_config(),
            "numeric_transition_state_weight": 0.05,
        },
    )
    with pytest.raises(ValueError, match="single weight source"):
        family.configure_objective(_law_objective())


def test_objective_rejects_c2_component_for_neural_operator() -> None:
    family = resolve_family("fno", _tiny_fno_config())
    with pytest.raises(ValueError, match="no C2"):
        family.configure_objective(
            _law_objective(local_law_component_weights={"c2": 0.5})
        )


def test_objective_requires_oracle_state_estimator() -> None:
    family = resolve_family("fno", _tiny_fno_config())
    with pytest.raises(ValueError, match="oracle_state"):
        family.configure_objective(_law_objective(local_law_estimator="corrected"))


def test_objective_requires_transition_metadata_on_training_trees(
    tmp_path: Path,
) -> None:
    trees = [
        dataclasses.replace(
            tree,
            metadata={
                key: value
                for key, value in dict(tree.metadata).items()
                if key not in {"n_states", "vocabulary_size"}
            },
        )
        for tree in _markov_trees(4, seed=57, split="train")
    ]
    family = resolve_family("fno", _tiny_fno_config())
    family.configure_objective(_law_objective())

    with pytest.raises(ValueError, match="numeric transition supervision"):
        family.train_g(
            g_init=None,
            f=None,
            traces=trees,
            output_dir=tmp_path,
            iteration=1,
        )


def test_law_channel_without_rows_is_an_error_not_a_silent_zero() -> None:
    family = resolve_family("fno", _tiny_fno_config())
    family.configure_objective(_law_objective())
    torch = family._torch

    leaf_only_rows = (
        torch.tensor([1.0]),
        torch.tensor([0]),
        torch.tensor([True]),
    )
    with pytest.raises(ValueError, match="merge_preservation"):
        family._assemble_loss(torch.tensor(2.0), leaf_only_rows)


def test_law_kind_channel_names_match_the_statistic_convention() -> None:
    assert LawKind.C1_LEAF.value == "leaf_preservation"
    assert LawKind.C3_MERGE.value == "merge_preservation"
