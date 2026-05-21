from __future__ import annotations

from pathlib import Path

from treepo.bench.suites.cardinality import build_cardinality_paper_suite
from treepo.bench.suites.identifiable_zero import (
    build_identifiable_zero_dtm_lda,
    build_identifiable_zero_lda_leafnoise,
    build_identifiable_zero_publication_ctreepo,
)


def test_identifiable_zero_dtm_lda_grid_size() -> None:
    specs = build_identifiable_zero_dtm_lda(out_root=Path("outputs/tmp"), skip_existing=False)
    assert len(specs) == 5 * 5 * 2 * 2 * 6


def test_identifiable_zero_lda_leafnoise_grid_size() -> None:
    specs = build_identifiable_zero_lda_leafnoise(out_root=Path("outputs/tmp"), skip_existing=False)
    assert len(specs) == 5 * 8 * 2 * 6


def test_identifiable_zero_publication_ctreepo_grid_size() -> None:
    specs = build_identifiable_zero_publication_ctreepo(out_root=Path("outputs/tmp"), skip_existing=False)
    # See suite definition: total across lanes/regimes.
    assert len(specs) == 9456


def test_cardinality_paper_grid_size() -> None:
    specs = build_cardinality_paper_suite(out_root=Path("outputs/tmp"), skip_existing=False)
    assert len(specs) == 4 * 3
