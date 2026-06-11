from __future__ import annotations


def test_lda_bench_experiments_are_research_only() -> None:
    from treepo.bench.runner import VALID_EXPERIMENTS

    assert "segmented-lda-ctreepo" not in VALID_EXPERIMENTS
    assert "segment-lda-ops-weight-recovery" not in VALID_EXPERIMENTS
    assert "learned-segment-lda-ops-g" not in VALID_EXPERIMENTS
    assert "learned-segmented-lda-theta-g" not in VALID_EXPERIMENTS

