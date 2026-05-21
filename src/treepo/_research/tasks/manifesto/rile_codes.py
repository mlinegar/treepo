"""Manifesto Project RILE code → ideology sign mapping + per-span RILE helper.

The Manifesto Project RILE score is defined (Laver & Budge 1992) as

    RILE = (right_count - left_count) / total_quasi_sentences * 100

where `right_count` and `left_count` are counts of quasi-sentences coded
with specific CMP 3-digit categories, and `total_quasi_sentences` includes
ALL quasi-sentences (including neutral/uncoded), giving RILE in (-100, 100).

This module:

* Defines the canonical left and right CMP-code sets.
* Exposes a `rile_sign(code)` lookup → {-1, 0, +1}.
* Exposes a `RILECorpusIndex` class that loads the per-quasi-sentence CSV
  at `data/raw/manifesto_project_full/manifesto_corpus_df.csv` once and
  provides `span_rile(manifesto_id, start_char, end_char)` for any
  character-span query. The character positions are computed by joining
  the quasi-sentence texts with a single space separator and tracking the
  cumulative start/end offsets of each.

The index is built lazily on first `manifesto(...)` call and cached so a
full grid of span queries over the same manifesto is cheap.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Canonical RILE code sets (Laver & Budge 1992).
# ---------------------------------------------------------------------------


RIGHT_CODES: frozenset[str] = frozenset(
    {
        "104",  # Military: Positive
        "201",  # Freedom and Human Rights
        "203",  # Constitutionalism: Positive
        "305",  # Political Authority
        "401",  # Free Market Economy
        "402",  # Incentives
        "407",  # Protectionism: Negative
        "414",  # Economic Orthodoxy
        "505",  # Welfare State Limitation
        "601",  # National Way of Life: Positive
        "603",  # Traditional Morality: Positive
        "605",  # Law and Order
        "606",  # Civic Mindedness: Positive
    }
)


LEFT_CODES: frozenset[str] = frozenset(
    {
        "103",  # Anti-Imperialism
        "105",  # Military: Negative
        "106",  # Peace
        "107",  # Internationalism: Positive
        "202",  # Democracy
        "403",  # Market Regulation
        "404",  # Economic Planning
        "406",  # Protectionism: Positive
        "412",  # Controlled Economy
        "413",  # Nationalisation
        "504",  # Welfare State Expansion
        "506",  # Education: Positive
        "701",  # Labour Groups: Positive
    }
)


def rile_sign(code: object) -> int:
    """Return -1 for left, +1 for right, 0 for neutral/missing/uncoded."""
    if code is None:
        return 0
    token = str(code).strip()
    if not token or token.lower() == "nan":
        return 0
    # Codes sometimes carry trailing decimals (e.g. "403.0"); normalize.
    if "." in token:
        token = token.split(".", 1)[0]
    if token in RIGHT_CODES:
        return 1
    if token in LEFT_CODES:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Per-manifesto quasi-sentence index + span RILE.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuasiSentenceSpan:
    """A single coded quasi-sentence with its char range in the joined text."""

    start_char: int
    end_char: int
    code: str  # raw CMP code as string (may be "000", "H", "", etc.)
    sign: int  # -1 / 0 / +1


@dataclass
class ManifestoCodings:
    """All coded quasi-sentences of one manifesto, aligned to joined text."""

    manifesto_id: str
    text: str  # quasi-sentences joined with a single space
    spans: tuple[QuasiSentenceSpan, ...]

    @property
    def total_quasi_sentences(self) -> int:
        return len(self.spans)

    def span_rile(self, start_char: int, end_char: int) -> float | None:
        """RILE score for quasi-sentences whose start lies in [start, end).

        Returns None if no quasi-sentences fall in the window (prevents
        divide-by-zero). The numerator is (right - left) of codes whose
        start_char is inside the window; the denominator is the count of
        such quasi-sentences (standard RILE normalization applied
        locally).
        """
        if end_char <= start_char or not self.spans:
            return None
        right = 0
        left = 0
        total = 0
        for span in self.spans:
            if span.start_char >= end_char:
                break
            if span.start_char < start_char:
                continue
            total += 1
            if span.sign > 0:
                right += 1
            elif span.sign < 0:
                left += 1
        if total == 0:
            return None
        return 100.0 * float(right - left) / float(total)


@dataclass
class RILECorpusIndex:
    """Lazy loader for per-manifesto quasi-sentence codings + span RILE.

    On first `manifesto(...)` call, reads `manifesto_corpus_df.csv` once,
    groups by `manifesto_id`, joins quasi-sentence texts with a single
    space and computes cumulative character spans. Subsequent calls hit the
    cache.
    """

    data_dir: Path | None = None
    _cache: dict[str, ManifestoCodings] = field(default_factory=dict, init=False, repr=False)
    _csv_loaded: bool = field(default=False, init=False, repr=False)
    _grouped: Mapping[str, Sequence[tuple[str, object]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def _resolved_csv_path(self) -> Path:
        # Priority: explicit data_dir > env var > repo-default.
        if self.data_dir is not None:
            return Path(self.data_dir) / "manifesto_corpus_df.csv"
        env = os.environ.get("MANIFESTO_FULL_DIR")
        if env:
            return Path(env).expanduser() / "manifesto_corpus_df.csv"
        # Default: repo `data/raw/manifesto_project_full/`.
        here = Path(__file__).resolve()
        return here.parents[3] / "data" / "raw" / "manifesto_project_full" / "manifesto_corpus_df.csv"

    def _ensure_csv_loaded(self) -> None:
        if self._csv_loaded:
            return
        import pandas as pd

        csv_path = self._resolved_csv_path()
        df = pd.read_csv(
            csv_path,
            low_memory=False,
            usecols=["text", "cmp_code", "pos", "manifesto_id"],
        )
        df = df.sort_values(["manifesto_id", "pos"]).reset_index(drop=True)
        # Group text + code rows per manifesto; store as simple list of pairs.
        grouped: dict[str, list[tuple[str, object]]] = {}
        for mid, sub in df.groupby("manifesto_id", sort=False):
            rows: list[tuple[str, object]] = []
            for _, row in sub.iterrows():
                text = row["text"]
                if text is None:
                    continue
                text_str = "" if _is_nan(text) else str(text)
                rows.append((text_str, row["cmp_code"]))
            grouped[str(mid)] = rows
        self._grouped = grouped  # type: ignore[assignment]
        self._csv_loaded = True

    def manifesto(self, manifesto_id: str) -> ManifestoCodings | None:
        """Return coded spans for a manifesto, or None if not granularly coded."""
        mid = str(manifesto_id)
        if mid in self._cache:
            return self._cache[mid]
        self._ensure_csv_loaded()
        rows = self._grouped.get(mid)
        if not rows or len(rows) < 2:
            # Manifestos with a single row are uncoded placeholders.
            return None
        spans: list[QuasiSentenceSpan] = []
        parts: list[str] = []
        cursor = 0
        for text, code in rows:
            if not text:
                continue
            start = cursor
            parts.append(text)
            end = start + len(text)
            spans.append(
                QuasiSentenceSpan(
                    start_char=start,
                    end_char=end,
                    code=str(code) if code is not None else "",
                    sign=rile_sign(code),
                )
            )
            # Single-space separator between quasi-sentences.
            cursor = end + 1
        joined = " ".join(parts)
        codings = ManifestoCodings(
            manifesto_id=mid,
            text=joined,
            spans=tuple(spans),
        )
        self._cache[mid] = codings
        return codings

    def coded_manifesto_ids(self) -> list[str]:
        """All manifesto ids that have granular codings (more than one row)."""
        self._ensure_csv_loaded()
        return sorted(
            mid for mid, rows in self._grouped.items() if rows and len(rows) > 1
        )


def _is_nan(value: object) -> bool:
    try:
        return value != value  # NaN ≠ NaN
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# Module-level singleton convenience (builds on first use).
# ---------------------------------------------------------------------------


_DEFAULT_INDEX: RILECorpusIndex | None = None


def default_index() -> RILECorpusIndex:
    global _DEFAULT_INDEX
    if _DEFAULT_INDEX is None:
        _DEFAULT_INDEX = RILECorpusIndex()
    return _DEFAULT_INDEX
