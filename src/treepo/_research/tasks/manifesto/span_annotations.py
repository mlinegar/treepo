"""Load Manifesto Project quasi-sentence annotations.

The local ``manifesto_corpus_df.csv`` contains one row per corpus item with
``text``, ``cmp_code``, ``pos``, ``manifesto_id``, and annotation flags.  The
helpers here expose annotated rows as ordered quasi-sentences and reconstruct a
stable document string with character spans for tree building.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd

from .span_targets import normalize_cmp_code


DEFAULT_QSENTENCE_CORPUS = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "raw"
    / "manifesto_project_full"
    / "manifesto_corpus_df.csv"
)


@dataclass(frozen=True)
class ManifestoQSentence:
    manifesto_id: str
    pos: int
    text: str
    cmp_code_raw: str
    cmp_code: str
    language: str = ""
    annotations: bool = True
    translation_en: bool = False


@dataclass(frozen=True)
class PositionedQSentence:
    item: ManifestoQSentence
    char_start: int
    char_end: int


@dataclass(frozen=True)
class ReconstructedManifesto:
    manifesto_id: str
    text: str
    qsentences: tuple[PositionedQSentence, ...]


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def load_manifesto_qsentences(
    corpus_csv: Path | str = DEFAULT_QSENTENCE_CORPUS,
    *,
    manifesto_ids: Optional[Iterable[str]] = None,
    require_annotations: bool = True,
    chunksize: int = 200_000,
) -> dict[str, list[ManifestoQSentence]]:
    """Load annotated quasi-sentences grouped by manifesto id."""
    corpus_csv = Path(corpus_csv)
    wanted = {str(item) for item in manifesto_ids} if manifesto_ids is not None else None
    usecols = [
        "text",
        "cmp_code",
        "pos",
        "manifesto_id",
        "language",
        "annotations",
        "translation_en",
    ]
    grouped: dict[str, list[ManifestoQSentence]] = {}
    for chunk in pd.read_csv(corpus_csv, usecols=usecols, chunksize=int(chunksize), low_memory=False):
        chunk["manifesto_id"] = chunk["manifesto_id"].astype(str)
        if wanted is not None:
            chunk = chunk[chunk["manifesto_id"].isin(wanted)]
        if require_annotations:
            chunk = chunk[chunk["annotations"].map(_boolish)]
        chunk = chunk[chunk["cmp_code"].notna()]
        chunk = chunk[chunk["text"].notna()]
        if chunk.empty:
            continue
        for row in chunk.itertuples(index=False):
            code = normalize_cmp_code(getattr(row, "cmp_code"))
            text = str(getattr(row, "text") or "").strip()
            if code is None or not text:
                continue
            try:
                pos = int(getattr(row, "pos"))
            except (TypeError, ValueError):
                continue
            manifesto_id = str(getattr(row, "manifesto_id"))
            grouped.setdefault(manifesto_id, []).append(
                ManifestoQSentence(
                    manifesto_id=manifesto_id,
                    pos=pos,
                    text=text,
                    cmp_code_raw=str(getattr(row, "cmp_code")),
                    cmp_code=code,
                    language=str(getattr(row, "language") or ""),
                    annotations=_boolish(getattr(row, "annotations")),
                    translation_en=_boolish(getattr(row, "translation_en")),
                )
            )
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item.pos))
    return grouped


def reconstruct_manifesto(
    manifesto_id: str,
    rows: Sequence[ManifestoQSentence],
    *,
    separator: str = "\n",
) -> ReconstructedManifesto:
    """Return a stable document text and char spans for ordered rows."""
    parts: list[str] = []
    positioned: list[PositionedQSentence] = []
    cursor = 0
    for idx, row in enumerate(rows):
        if idx:
            parts.append(separator)
            cursor += len(separator)
        start = cursor
        parts.append(str(row.text))
        cursor += len(str(row.text))
        positioned.append(PositionedQSentence(item=row, char_start=start, char_end=cursor))
    return ReconstructedManifesto(
        manifesto_id=str(manifesto_id),
        text="".join(parts),
        qsentences=tuple(positioned),
    )


def qsentences_in_span(
    reconstructed: ReconstructedManifesto,
    *,
    char_start: int,
    char_end: int,
) -> list[PositionedQSentence]:
    start = int(char_start)
    end = int(char_end)
    return [
        item
        for item in reconstructed.qsentences
        if int(item.char_start) >= start and int(item.char_end) <= end
    ]


def indexed_manifesto_ids(grouped: Mapping[str, Sequence[ManifestoQSentence]]) -> list[str]:
    return sorted(str(key) for key, rows in grouped.items() if rows)

