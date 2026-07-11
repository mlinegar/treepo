"""The learnable-constant family's train path runs and the IPW math bites.

The family docstring promises the simplest training problem where IPW
correction matters: with confounded propensities the naive observed mean is
biased, and the Hájek estimate corrects it. These tests exercise ``train_f``
end-to-end (it used to crash on a non-canonical law-kind spelling).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from treepo.methods.learnable import LearnableConstantFamily


def _tree(*, value: float | None, observed: bool, propensity: float) -> SimpleNamespace:
    metadata: dict[str, object] = {
        "observed": observed,
        "propensity": propensity,
    }
    if value is not None:
        metadata["teacher_score_native"] = value
    return SimpleNamespace(metadata=metadata)


def test_train_f_returns_hajek_ipw_mean(tmp_path: Path) -> None:
    # Confounded design: the low-value tree was undersampled (pi = 0.2), the
    # high-value tree always sampled (pi = 1.0). Naive observed mean = 5.5;
    # the Hájek estimate is (1/0.2 * 1 + 1/1 * 10) / (1/0.2 + 1/1) = 2.5.
    trees = [
        _tree(value=1.0, observed=True, propensity=0.2),
        _tree(value=10.0, observed=True, propensity=1.0),
        _tree(value=None, observed=False, propensity=0.2),
    ]
    family = LearnableConstantFamily()

    learned = family.train_f(
        f_init=None,
        g=None,
        traces=trees,
        output_dir=tmp_path,
        iteration=1,
    )

    assert learned == pytest.approx(2.5)
    assert family.last_trained_f == pytest.approx(2.5)
    summary = family.last_objective_summary
    assert summary is not None
    assert summary["objective"] == pytest.approx(2.5)


def test_train_f_ipw_estimate_differs_from_naive_observed_mean(tmp_path: Path) -> None:
    trees = [
        _tree(value=1.0, observed=True, propensity=0.2),
        _tree(value=10.0, observed=True, propensity=1.0),
    ]
    naive_mean = (1.0 + 10.0) / 2.0

    learned = LearnableConstantFamily().train_f(
        f_init=None,
        g=None,
        traces=trees,
        output_dir=tmp_path,
        iteration=1,
    )

    assert learned != pytest.approx(naive_mean)
    assert learned < naive_mean  # undersampled low value is up-weighted


def test_train_f_uniform_design_reduces_to_plain_mean(tmp_path: Path) -> None:
    trees = [
        _tree(value=2.0, observed=True, propensity=1.0),
        _tree(value=4.0, observed=True, propensity=1.0),
    ]

    learned = LearnableConstantFamily().train_f(
        f_init=None,
        g=None,
        traces=trees,
        output_dir=tmp_path,
        iteration=1,
    )

    assert learned == pytest.approx(3.0)


def test_scoring_uses_the_trained_constant(tmp_path: Path) -> None:
    family = LearnableConstantFamily()
    learned = family.train_f(
        f_init=None,
        g=None,
        traces=[_tree(value=2.0, observed=True, propensity=1.0)],
        output_dir=tmp_path,
        iteration=1,
    )
    scores = family.score_roots_with_f(f=learned, g=None, trees=[object(), object()])
    assert scores == [pytest.approx(2.0), pytest.approx(2.0)]
