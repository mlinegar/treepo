"""sklearn_lda is a registered family: LDA trains only inside fit().

The count-vector statistic is exactly compositional, so its law rows are
exact-zero across all three channels on the synthetic fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from treepo import fit
from treepo.local_law import LawKind
from treepo.methods.families import list_families, resolve_family
from treepo.methods.fixtures import make_lda_topic_trees


def _trees(n_trees: int, *, seed: int, split: str):
    return make_lda_topic_trees(
        n_trees=n_trees,
        n_topics=3,
        doc_tokens=96,
        leaf_unit_count=16,
        vocabulary_size=45,
        seed=seed,
        split=split,
    )


def test_sklearn_lda_family_is_builtin() -> None:
    assert "sklearn_lda" in list_families()
    family = resolve_family(
        "sklearn_lda", {"n_topics": 3, "vocabulary_size": 45, "max_iter": 5}
    )
    assert family.name == "sklearn_lda"
    assert family.config.n_topics == 3


def test_fit_runs_sklearn_lda_on_topic_fixture(tmp_path: Path) -> None:
    result = fit(
        {
            "family": "sklearn_lda",
            "train_data": _trees(24, seed=31, split="train"),
            "eval_data": _trees(8, seed=32, split="test"),
            "backend_config": {
                "n_topics": 3,
                "vocabulary_size": 45,
                "max_iter": 20,
                "output_dir": str(tmp_path),
            },
            "axis": {"max_iterations": 1, "axis_value": 0},
        },
    )

    assert result.status == "success"
    assert result.artifacts["f"]["kind"] == "treepo_sklearn_lda"
    assert len(result.artifacts["f"]["topic_order"]) == 3
    assert result.artifacts["statistic"]["info"]["state_kind"] == "token_counts"


def test_lda_count_statistic_laws_are_exact() -> None:
    family = resolve_family("sklearn_lda", {"n_topics": 3, "vocabulary_size": 45})
    statistic = family.as_statistic()
    trees = _trees(4, seed=33, split="test")

    rows = statistic.local_law_rows(trees)

    kinds = {LawKind.from_value(row.law_kind) for row in rows}
    assert kinds == {LawKind.C1_LEAF, LawKind.C2_IDEMPOTENCE, LawKind.C3_MERGE}
    assert all(row.proxy_loss == 0.0 for row in rows)


def test_baseline_helper_matches_direct_family_training() -> None:
    from treepo.methods.lda import fit_sklearn_lda_baseline

    train = _trees(24, seed=34, split="train")
    eval_trees = _trees(8, seed=35, split="test")

    baseline = fit_sklearn_lda_baseline(
        train,
        eval_trees,
        n_topics=3,
        vocabulary_size=45,
        max_iter=20,
        random_state=0,
        target_topic=0,
    )

    family = resolve_family(
        "sklearn_lda",
        {"n_topics": 3, "vocabulary_size": 45, "max_iter": 20, "random_state": 0},
    )
    family.train_f(f_init=None, g=None, traces=train, output_dir=Path("."), iteration=1)
    direct = family.score_roots_with_f(f=None, g=None, trees=eval_trees)

    import numpy as np

    assert np.allclose(np.asarray(baseline.predictions), np.asarray(direct))
