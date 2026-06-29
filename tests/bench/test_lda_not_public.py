from __future__ import annotations


def test_lda_bench_experiments_are_not_public() -> None:
    from treepo.bench.runner import VALID_EXPERIMENTS

    assert "lda" not in VALID_EXPERIMENTS
    assert "manifesto" not in VALID_EXPERIMENTS
    assert "segmented-lda-ctreepo" not in VALID_EXPERIMENTS
    assert "segment-lda-ops-weight-recovery" not in VALID_EXPERIMENTS
    assert "learned-segment-lda-ops-g" not in VALID_EXPERIMENTS
    assert "learned-segmented-lda-theta-g" not in VALID_EXPERIMENTS
