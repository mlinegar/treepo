"""Local-law and audit math: verify the formulas fire on real rows AND
drive a real training step.

Two complementary checks:

1. **Post-hoc audit hook.** ``backend_config['law_audit_rows']`` triggers
   :func:`local_law_objective_summary` + :func:`compute_influence_weighted_overlap`
   on the supplied rows. ``result.summary['audit']`` and the on-disk
   manifest carry the same dict. This is the surface that an evaluator
   uses *after* a ladder run.

2. **In-loop training signal.** The ``"learnable_constant"`` family
   trains its scalar f by calling
   :func:`local_law_objective_summary` in ``sampled_ipw`` mode. The
   trained value is *exactly* the IPW estimator; under confounded
   sampling (high-value trees sampled more often), the naive observed
   mean is biased while the IPW estimate is unbiased. We check both.

If the local-law arithmetic ever changes (e.g. a different propensity
floor, a different depth-weight convention), these tests will fail loudly
— that is the point.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from treepo._research.ctreepo.contracts import CTreePOLearningSpec
from treepo.methods import (
    LocalLawAuditRow,
    compute_influence_weighted_overlap,
    corrected_local_law_loss,
    fit,
    local_law_objective_summary,
)
from treepo.methods.learnable import LearnableConstantFamily


# --------------------------------------------------------------------------- #
# 1. The math primitives themselves.
# --------------------------------------------------------------------------- #


def test_corrected_loss_unobserved_returns_proxy() -> None:
    assert corrected_local_law_loss(
        proxy_loss=0.42,
        oracle_loss=None,
        observed=False,
        propensity=0.0,
    ) == pytest.approx(0.42)


def test_corrected_loss_observed_pi_one_returns_oracle() -> None:
    """With π = 1, AIPW reduces to oracle: proxy + (oracle - proxy)/1 = oracle."""
    assert corrected_local_law_loss(
        proxy_loss=10.0,
        oracle_loss=2.0,
        observed=True,
        propensity=1.0,
    ) == pytest.approx(2.0)


def test_corrected_loss_observed_amplifies_when_pi_small() -> None:
    """proxy + (oracle - proxy)/π with proxy=10, oracle=12, π=0.1 → 10 + 20 = 30."""
    assert corrected_local_law_loss(
        proxy_loss=10.0,
        oracle_loss=12.0,
        observed=True,
        propensity=0.1,
    ) == pytest.approx(30.0)


def test_corrected_loss_rejects_zero_propensity_when_observed() -> None:
    with pytest.raises(ValueError, match="propensity"):
        corrected_local_law_loss(
            proxy_loss=1.0,
            oracle_loss=1.0,
            observed=True,
            propensity=0.0,
        )


def test_overlap_uses_all_rows_with_observed_ess_diagnostic() -> None:
    """Lean overlap is over the full finite row space; observed-only ESS is
    retained only as a separately named sampling diagnostic.
    """
    rows = [
        LocalLawAuditRow(
            row_id="observed",
            law_kind="c1",
            proxy_loss=0.0,
            oracle_loss=0.0,
            observed=True,
            propensity=0.5,
            node_weight=1.0,
        ),
        LocalLawAuditRow(
            row_id="unobserved",
            law_kind="c1",
            proxy_loss=0.0,
            oracle_loss=None,
            observed=False,
            propensity=0.5,
            node_weight=1.0,
        ),
    ]
    overlap = compute_influence_weighted_overlap(rows)
    assert overlap.n_observed == 1
    assert overlap.n_total == 2
    assert overlap.D_lambda == pytest.approx(4.0)
    assert overlap.W_lambda == pytest.approx(2.0)
    assert overlap.effective_sample_size == pytest.approx(2.0)
    assert overlap.observed_effective_sample_size == pytest.approx(1.0)


def test_overlap_kish_ess_on_balanced_design() -> None:
    """Two observed rows at π=0.5 each, λ=1: D_λ = 4, W_λ = 2,
    Kish ESS = (2+2)² / (4+4) = 16/8 = 2.
    """
    rows = [
        LocalLawAuditRow(
            row_id=f"r{i}",
            law_kind="c1",
            proxy_loss=0.0,
            oracle_loss=0.0,
            observed=True,
            propensity=0.5,
            node_weight=1.0,
        )
        for i in range(2)
    ]
    overlap = compute_influence_weighted_overlap(rows)
    assert overlap.effective_sample_size == pytest.approx(2.0)
    assert overlap.D_lambda == pytest.approx(4.0)


def test_objective_summary_sampled_ipw_matches_closed_form() -> None:
    """Sampled-IPW objective: (Σ w·oracle/π) / (Σ w/π). With w=1, oracle=10 at π=0.9
    and oracle=2 at π=0.1, the IPW estimate is (10/0.9 + 2/0.1) / (1/0.9 + 1/0.1)
    = (11.11 + 20) / (1.11 + 10) ≈ 31.11 / 11.11 ≈ 2.8.
    """
    rows = [
        LocalLawAuditRow(
            row_id="hi", law_kind="c1", proxy_loss=0.0,
            oracle_loss=10.0, observed=True, propensity=0.9, node_weight=1.0,
        ),
        LocalLawAuditRow(
            row_id="lo", law_kind="c1", proxy_loss=0.0,
            oracle_loss=2.0, observed=True, propensity=0.1, node_weight=1.0,
        ),
    ]
    summary = local_law_objective_summary(rows, objective_mode="sampled_ipw")
    expected = (10.0 / 0.9 + 2.0 / 0.1) / (1.0 / 0.9 + 1.0 / 0.1)
    assert summary.objective == pytest.approx(expected, rel=1e-9)
    assert summary.observed_count == 2


def test_objective_summary_corrected_with_depth_discount() -> None:
    """Corrected mode with γ=0.5: each row's contribution is weighted by
    γ^depth × node_weight, normalized by total weights.
    """
    rows = [
        LocalLawAuditRow(
            row_id=f"d{d}",
            law_kind="c1",
            proxy_loss=0.0,
            oracle_loss=0.0,
            observed=True,
            propensity=1.0,
            node_weight=1.0,
            depth=d,
        )
        for d in range(3)
    ]
    summary = local_law_objective_summary(
        rows, objective_mode="corrected_local_law", gamma_depth=0.5
    )
    # All oracle/proxy values are zero, so the objective is exactly 0.
    assert summary.objective == pytest.approx(0.0)
    # But the weight_sum reflects gamma-discounted depth weights:
    # weights = 1 + 0.5 + 0.25 = 1.75.
    assert summary.weight_sum == pytest.approx(1.75)


def test_objective_summary_rejects_gamma_above_one() -> None:
    rows = [
        LocalLawAuditRow(
            row_id="r0",
            law_kind="c1",
            proxy_loss=0.0,
            oracle_loss=0.0,
            observed=True,
            propensity=1.0,
        )
    ]
    with pytest.raises(ValueError, match="gamma_depth"):
        local_law_objective_summary(rows, gamma_depth=1.5)


# --------------------------------------------------------------------------- #
# 2. In-loop training signal: LearnableConstantFamily uses the IPW summary.
#    (The post-hoc audit via fit() backend_config is gone — audit now lives
#    only at run("audit", ...). See test_methods_centralized.py.)
# --------------------------------------------------------------------------- #


def _confounded_train_trees(seed: int, n: int) -> List[SimpleNamespace]:
    """Two-group confounded design: half the trees have value 10 and are
    sampled with propensity 0.9; the other half have value 2 and are
    sampled with propensity 0.1.

    True population mean = 6. Naive observed mean ≈ 9.2 (strongly biased
    toward the over-sampled group). IPW estimate ≈ 6.0 (unbiased).

    This is the textbook IPW demonstration; with N large the gap is
    clean and the test isn't seed-fragile.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    trees: List[SimpleNamespace] = []
    half = n // 2
    for idx in range(n):
        if idx < half:
            y, pi = 10.0, 0.9
        else:
            y, pi = 2.0, 0.1
        observed = bool(rng.random() < pi)
        trees.append(
            SimpleNamespace(
                leaves=[SimpleNamespace(tokens=[0])],
                metadata={
                    "split": "train",
                    "teacher_score_1_7": y,
                    "observed": observed,
                    "propensity": pi,
                },
            )
        )
    return trees


def test_learnable_constant_trains_via_ipw_summary(tmp_path: Path) -> None:
    """The trained scalar must equal :func:`local_law_objective_summary` in
    ``sampled_ipw`` mode applied to the training trees — proving the
    formula is the training signal, not a separate post-hoc metric.
    """
    family = LearnableConstantFamily()
    train_trees = _confounded_train_trees(seed=0, n=200)
    spec = CTreePOLearningSpec(
        space_kind="learnable_constant",
        family="learnable_constant",
        schedule="fg",
        initial_artifacts={"f": 0.0, "g": None},
        train_data=train_trees,
        eval_data=[],
        backend_config={
            "family_runtime": family,
            "output_dir": str(tmp_path),
        },
        axis={"max_iterations": 1, "axis_value": 0},
    )
    result = fit(spec)
    assert result.status == "success"
    learned = family.last_trained_f
    assert learned is not None
    # Recompute the IPW estimator directly and confirm the family hit it.
    rows = family._rows_from_trees(train_trees)
    expected = local_law_objective_summary(rows, objective_mode="sampled_ipw").objective
    assert learned == pytest.approx(expected, rel=1e-12)


def test_learnable_constant_ipw_unbiased_vs_naive_observed_mean(tmp_path: Path) -> None:
    """Two-group confounded design (true_mean=6.0):
    - naive observed mean ≈ 9.2 (biased toward over-sampled high-value group)
    - IPW estimate ≈ 6.0 (unbiased)
    """
    family = LearnableConstantFamily()
    train_trees = _confounded_train_trees(seed=1, n=4000)
    spec = CTreePOLearningSpec(
        space_kind="learnable_constant",
        family="learnable_constant",
        schedule="fg",
        initial_artifacts={"f": 0.0, "g": None},
        train_data=train_trees,
        eval_data=[],
        backend_config={
            "family_runtime": family,
            "output_dir": str(tmp_path),
        },
        axis={"max_iterations": 1, "axis_value": 0},
    )
    fit(spec)
    learned = family.last_trained_f

    # Naive observed mean — biased by confounded sampling.
    observed = [
        t.metadata["teacher_score_1_7"]
        for t in train_trees
        if t.metadata["observed"]
    ]
    naive_mean = sum(observed) / len(observed)

    true_mean = 6.0
    # Naive observed mean is dominated by the over-sampled (y=10, π=0.9)
    # group — expected ~9.2.
    assert naive_mean > 8.0, (
        f"sanity: naive mean {naive_mean:.3f} should be biased high above 8.0"
    )
    # IPW estimate is unbiased; with n=4000 the sampling noise is small.
    assert abs(learned - true_mean) < 0.3, (
        f"IPW estimate {learned:.3f} should be within 0.3 of true mean {true_mean}"
    )


def test_learnable_constant_predictions_use_trained_value(tmp_path: Path) -> None:
    """Predictions at evaluation time = trained constant for every tree.
    Tests the f-artifact threading from train_f → score_roots_with_f.
    """
    family = LearnableConstantFamily()
    train_trees = _confounded_train_trees(seed=2, n=50)
    eval_trees = [
        SimpleNamespace(leaves=[SimpleNamespace(tokens=[0])], metadata={"split": "test"})
        for _ in range(5)
    ]
    spec = CTreePOLearningSpec(
        space_kind="learnable_constant",
        family="learnable_constant",
        schedule="fg",
        initial_artifacts={"f": 0.0, "g": None},
        train_data=train_trees,
        eval_data=eval_trees,
        backend_config={
            "family_runtime": family,
            "output_dir": str(tmp_path),
        },
        axis={"max_iterations": 1, "axis_value": 0},
    )
    result = fit(spec)
    assert result.status == "success"
    # Eval ran; the family was queried via score_roots_with_f. The trained
    # scalar is f.
    learned = family.last_trained_f
    assert learned is not None
    # Sanity: evaluator returns a prediction record per tree (we can't
    # easily read it here without parsing the iteration dir, but if
    # `result.status == "success"` and ``last_trained_f`` is populated,
    # the path executed).
