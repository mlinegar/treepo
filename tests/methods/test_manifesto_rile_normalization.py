"""RILE normalization conventions on the vendored manifesto task modules.

The repo standard (decided by the Step 0 reconstruction gate, 2026-06-09)
is the ``total_non_header`` denominator: it matches the published MPDS
``rile`` ~3x better (Pearson 0.9975 / MAE 0.49 vs 0.9944 / 1.35).
``span_rile`` defaults to it and routes through ``targets_from_counts`` so
leaf, internal, and root targets share one code path; ``denominator="all"``
keeps the legacy all-quasi-sentence convention for comparisons.
"""

from __future__ import annotations

from treepo._research.tasks.manifesto.rile_codes import (
    ManifestoCodings,
    QuasiSentenceSpan,
    rile_sign,
)
from treepo._research.tasks.manifesto.span_targets import targets_from_counts


def _synthetic_codings(pattern: list[str]) -> ManifestoCodings:
    spans: list[QuasiSentenceSpan] = []
    cursor = 0
    for code in pattern:
        spans.append(
            QuasiSentenceSpan(
                start_char=cursor,
                end_char=cursor + 1,
                code=code,
                sign=rile_sign(code),
            )
        )
        cursor += 2
    return ManifestoCodings(
        manifesto_id="synthetic",
        text=" ".join(["x"] * len(pattern)),
        spans=tuple(spans),
    )


def test_span_rile_denominator_conventions_diverge_on_headers() -> None:
    codings = _synthetic_codings(["104", "104", "504", "H", "H", "000"])
    window = (0, len(codings.text) + 1)
    standard = codings.span_rile(*window)
    legacy = codings.span_rile(*window, denominator="all")
    assert standard is not None and legacy is not None
    assert abs(standard - 100.0 * (2 - 1) / 4) < 1e-6  # H excluded -> /4
    assert abs(legacy - 100.0 * (2 - 1) / 6) < 1e-6  # all spans -> /6


def test_span_rile_shares_code_path_with_targets_from_counts() -> None:
    codings = _synthetic_codings(["104", "605.1", "504", "H", "000", "403"])
    window = (0, len(codings.text) + 1)
    counts = codings.span_counts(*window)
    assert counts is not None
    assert counts == {"104": 1, "605": 1, "504": 1, "H": 1, "000": 1, "403": 1}
    expected = targets_from_counts(counts)["rile_raw"]
    got = codings.span_rile(*window)
    assert got is not None
    assert abs(got - expected) < 1e-9
    assert codings.span_rile(500, 600) is None
    assert codings.span_rile(500, 600, denominator="all") is None


def test_targets_from_counts_denominator_option() -> None:
    counts = {"104": 2, "504": 1, "H": 2, "000": 1}
    non_header = targets_from_counts(counts)
    legacy = targets_from_counts(counts, denominator="all")
    assert abs(non_header["rile_raw"] - 100.0 * (2 - 1) / 4) < 1e-6
    assert abs(legacy["rile_raw"] - 100.0 * (2 - 1) / 6) < 1e-6
    try:
        targets_from_counts(counts, denominator="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown denominator")
