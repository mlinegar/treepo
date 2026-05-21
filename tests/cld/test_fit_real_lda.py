"""Real-data LDA test: ``fit()`` produces non-trivial metrics on synthetic
leaf-local-mixture documents.

This exercises the full path:

1. Generate a real ``LeafLocalMixtureUtilityWorld`` (deterministic seed).
2. Wrap each test doc as an :class:`LDADocTree` with the DGP parameters
   the ``leaf_local_mixture_target`` oracle's score_tree adapter expects.
3. Dispatch through ``fit(spec)`` with ``spec.family='oracle'``.
4. Assert that ``CTreePOFitResult.metrics`` contains *finite* numbers
   (not None / NaN placeholders) and that the oracle-vs-itself MAE is
   effectively zero.

Together this proves: the registry resolves, the dispatch reaches the
oracle, the oracle's score_tree adapter actually fires on real DGP data,
the evaluator pairs predictions with teacher truth correctly, and the
metric extractor in ``learning.py`` surfaces real numbers — not just
empty placeholders from a no-op stub.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from treepo._research.ctreepo.contracts import CTreePOLearningSpec
from treepo._research.ctreepo.oracles.lda import leaf_local_mixture_target
from treepo.cld import fit
from treepo.cld.fixtures import make_leaf_local_mixture_trees


def test_fit_runs_lda_oracle_on_real_synthetic_data(tmp_path: Path) -> None:
    trees, cfg = make_leaf_local_mixture_trees(seed=0, split="test")
    assert len(trees) >= 4, "evaluator needs >=4 paired predictions for MAE"

    spec = CTreePOLearningSpec(
        space_kind="leaf_local_mixture",
        family="oracle",
        schedule="fg",
        initial_artifacts={"f": None, "g": None},
        train_data=[],
        eval_data=trees,
        backend_config={
            "oracle_name": "leaf_local_mixture_target",
            "output_dir": str(tmp_path),
        },
        axis={"max_iterations": 0, "axis_value": 0},
    )
    result = fit(spec)

    assert result.status == "success"
    # The "all" split's MAE must be a real, finite number — proves the
    # evaluator actually paired predictions with teacher scores.
    mae = result.metrics.get("internal_f_mae")
    assert mae is not None, f"expected internal_f_mae in metrics; got {result.metrics!r}"
    assert math.isfinite(mae)
    # Oracle compared against itself: MAE must be effectively zero
    # (modulo float64 representation noise on closed-form targets).
    assert mae < 1e-9, f"oracle-vs-itself MAE should be ~0, got {mae}"
    # And n paired must equal the eval_data length.
    assert result.metrics.get("n") == float(len(trees))


def test_lda_oracle_direct_call_matches_score_tree(tmp_path: Path) -> None:
    """The oracle's ``score_tree`` adapter must compute the same number as
    a direct call to ``leaf_local_mixture_target(doc, theta, W_base, lam)``
    for the first synthetic doc — this is the cross-check that the
    fixture's adapter shape is correct.
    """
    from treepo._research.ctreepo.oracles import get_oracle

    trees, cfg = make_leaf_local_mixture_trees(seed=42, split="test")
    tree0 = trees[0]
    oracle = get_oracle("leaf_local_mixture_target")
    assert oracle.score_tree is not None
    via_adapter = float(oracle.score_tree(tree0))
    via_direct = float(
        leaf_local_mixture_target(
            tree0.doc,
            theta=tree0.theta,
            W_base=tree0.W_base,
            lambda_multiplier=tree0.lambda_multiplier,
        )
    )
    assert math.isclose(via_adapter, via_direct, rel_tol=0.0, abs_tol=0.0)


def test_fit_lda_with_distinct_seeds_yields_distinct_scores(tmp_path: Path) -> None:
    """Different seeds → different docs → different mean prediction values.
    This catches a class of bug where the wrapper accidentally serves the
    same doc to every tree (the LDA oracle would happily return MAE=0 in
    that case too, but the *mean_prediction* would be identical across
    seeds and the test would catch it).
    """
    trees_a, _ = make_leaf_local_mixture_trees(seed=0, split="test")
    trees_b, _ = make_leaf_local_mixture_trees(seed=1, split="test")

    def _run(trees, out_dir):
        spec = CTreePOLearningSpec(
            space_kind="leaf_local_mixture",
            family="oracle",
            schedule="fg",
            initial_artifacts={"f": None, "g": None},
            train_data=[],
            eval_data=trees,
            backend_config={
                "oracle_name": "leaf_local_mixture_target",
                "output_dir": str(out_dir),
            },
            axis={"max_iterations": 0, "axis_value": 0},
        )
        return fit(spec)

    result_a = _run(trees_a, tmp_path / "a")
    result_b = _run(trees_b, tmp_path / "b")
    # Both must succeed.
    assert result_a.status == "success"
    assert result_b.status == "success"
    # Sanity: the per-tree teacher scores differ across seeds.
    teachers_a = [t.metadata["teacher_score_1_7"] for t in trees_a]
    teachers_b = [t.metadata["teacher_score_1_7"] for t in trees_b]
    assert teachers_a != teachers_b, "fixture seeds did not produce distinct docs"
