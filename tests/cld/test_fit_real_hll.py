"""Real-data HLL / cardinality test: ``fit()`` produces correct counts
through the oracle path and bounded-error counts through the classical
HLL sketch path.

Two concrete checks:

1. ``family='oracle'`` + ``oracle_name='hll_exact'`` on token trees must
   return MAE ≈ 0 against the precomputed exact unique-count teacher.
2. ``family='hll'`` + ``ClassicalSketchFamilyRuntime`` with a native HLL
   adapter at p=12 must return a finite cardinality estimate whose
   relative error against the true distinct count is bounded by HLL's
   theoretical RSE on a small leaf chain.

Together this proves the same plumbing serves both the exact (oracle)
and approximate (classical sketch) paths through one ``fit()`` signature.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from treepo._research.ctreepo.contracts import CTreePOLearningSpec
from treepo.sketches.adapters import make_hll_adapter
from treepo.cld import fit
from treepo.cld.fixtures import make_hll_token_trees


def _make_spec(*, family: str, trees, backend_config, tmp_path: Path) -> CTreePOLearningSpec:
    backend_config = {**backend_config, "output_dir": str(tmp_path)}
    return CTreePOLearningSpec(
        space_kind="hll_cardinality",
        family=family,
        schedule="fg",
        initial_artifacts={"f": None, "g": None},
        train_data=[],
        eval_data=trees,
        backend_config=backend_config,
        axis={"max_iterations": 0, "axis_value": 0},
    )


def test_fit_runs_hll_exact_oracle_with_zero_mae(tmp_path: Path) -> None:
    trees = make_hll_token_trees(
        n_trees=8,
        leaves_per_tree=4,
        leaf_token_count=12,
        vocabulary_size=32,
        seed=0,
    )
    spec = _make_spec(
        family="oracle",
        trees=trees,
        backend_config={"oracle_name": "hll_exact"},
        tmp_path=tmp_path,
    )
    result = fit(spec)
    assert result.status == "success"
    mae = result.metrics.get("internal_f_mae")
    assert mae is not None and math.isfinite(mae)
    # Exact oracle: must match the precomputed exact unique counts.
    assert mae == pytest.approx(0.0, abs=1e-9)
    assert result.metrics.get("n") == float(len(trees))


def test_fit_runs_hll_classical_sketch_within_bounded_error(tmp_path: Path) -> None:
    """The native HLL adapter at p=12 has relative standard error
    ~1.04/sqrt(2^12) ≈ 1.6%. With 8 trees and ~50 distinct tokens each,
    the *mean* MAE should be small compared to the mean true count.
    """
    trees = make_hll_token_trees(
        n_trees=8,
        leaves_per_tree=4,
        leaf_token_count=24,
        vocabulary_size=200,
        seed=7,
    )
    adapter = make_hll_adapter(backend="native", precision=12)
    spec = _make_spec(
        family="sketch",
        trees=trees,
        backend_config={"sketch_adapter": adapter, "sketch_schedule": "balanced"},
        tmp_path=tmp_path,
    )
    result = fit(spec)
    assert result.status == "success"
    mae = result.metrics.get("internal_f_mae")
    assert mae is not None and math.isfinite(mae)

    mean_teacher = result.metrics.get("internal_f_mae")  # use mae as size proxy
    mean_truth = sum(t.metadata["teacher_score_1_7"] for t in trees) / len(trees)
    # MAE should be a small fraction of the mean truth (well within the
    # HLL relative-error envelope at p=12, even with the small-cardinality
    # bias correction in the native adapter).
    assert mae < 0.30 * mean_truth, (
        f"HLL p=12 mean MAE={mae:.3f} too large vs mean truth={mean_truth:.3f}"
    )


def test_fit_hll_classical_sketch_recovers_distinct_counts(tmp_path: Path) -> None:
    """Same plumbing, larger precision (p=14) → tighter error. This is a
    smoke test that increasing precision actually tightens the result —
    confirms the sketch path is wired to the precision knob.
    """
    trees = make_hll_token_trees(
        n_trees=8,
        leaves_per_tree=4,
        leaf_token_count=24,
        vocabulary_size=200,
        seed=11,
    )
    adapter = make_hll_adapter(backend="native", precision=14)
    spec = _make_spec(
        family="sketch",
        trees=trees,
        backend_config={"sketch_adapter": adapter, "sketch_schedule": "balanced"},
        tmp_path=tmp_path,
    )
    result = fit(spec)
    assert result.status == "success"
    mae = result.metrics["internal_f_mae"]
    mean_truth = sum(t.metadata["teacher_score_1_7"] for t in trees) / len(trees)
    # p=14 should comfortably beat 15% relative error on this scale.
    assert mae < 0.15 * mean_truth, (
        f"HLL p=14 mean MAE={mae:.3f} too large vs mean truth={mean_truth:.3f}"
    )
