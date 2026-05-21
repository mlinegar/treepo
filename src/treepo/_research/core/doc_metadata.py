"""
Document metadata helpers.

This module provides a small, typed metadata container used to:
  - attach stable "WHO/WHEN/WHERE" context to documents
  - optionally embed metadata text alongside document content (for flexible retrieval)

We keep metadata formatting deterministic so embeddings/caches are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DocMetadata:
    """Lightweight document metadata (task-agnostic)."""

    doc_id: str
    source: str = ""

    country: Optional[str] = None
    party: Optional[str] = None
    party_abbrev: Optional[str] = None
    year: Optional[int] = None
    date_code: Optional[int] = None
    election_date: Optional[str] = None
    party_family: Optional[int] = None
    rile: Optional[float] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_manifesto_sample(sample: Any) -> "DocMetadata":
        """Best-effort conversion from ``ManifestoSample`` to ``DocMetadata``."""
        doc_id = str(getattr(sample, "manifesto_id", "") or getattr(sample, "doc_id", "") or "").strip()
        if not doc_id:
            doc_id = str(getattr(sample, "id", "") or "").strip()
        if not doc_id:
            raise ValueError("Cannot infer doc_id from sample")

        year = getattr(sample, "year", None)
        if year is None:
            date_code = getattr(sample, "date_code", None)
            if isinstance(date_code, int):
                year = date_code // 100

        return DocMetadata(
            doc_id=doc_id,
            source="manifesto",
            country=_clean_optional_str(getattr(sample, "country_name", None)),
            party=_clean_optional_str(getattr(sample, "party_name", None)),
            party_abbrev=_clean_optional_str(getattr(sample, "party_abbrev", None)),
            year=_clean_optional_int(year),
            date_code=_clean_optional_int(getattr(sample, "date_code", None)),
            election_date=_clean_optional_str(getattr(sample, "election_date", None)),
            party_family=_clean_optional_int(getattr(sample, "party_family", None)),
            rile=_clean_optional_float(getattr(sample, "rile", None)),
            extra={},
        )


def _clean_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _clean_optional_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if converted != converted:
        return None
    return converted


def format_doc_meta_prompt_block(meta: DocMetadata) -> str:
    """
    Render metadata as a compact prompt block.

    This is intended for optional injection into LLM prompts (not required for
    embedding-based models).
    """
    lines = ["[DOC_META]"]
    for key, value in _ordered_kv(meta):
        lines.append(f"{key}: {value}")
    return "\n".join(lines).strip()


def format_doc_meta_embedding_text(meta: DocMetadata) -> str:
    """
    Render metadata as deterministic text for embedding.

    We keep it short and structured so it composes well with multilingual
    embedding models.
    """
    lines = ["DOC_META"]
    for key, value in _ordered_kv(meta):
        lines.append(f"{key}: {value}")
    return "\n".join(lines).strip()


def _ordered_kv(meta: DocMetadata) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    out.append(("doc_id", str(meta.doc_id)))
    if meta.source:
        out.append(("source", str(meta.source)))
    if meta.country:
        out.append(("country", str(meta.country)))
    if meta.party:
        out.append(("party", str(meta.party)))
    if meta.party_abbrev:
        out.append(("party_abbrev", str(meta.party_abbrev)))
    if meta.election_date:
        out.append(("election_date", str(meta.election_date)))
    if meta.date_code is not None:
        out.append(("date_code", str(int(meta.date_code))))
    if meta.year is not None:
        out.append(("year", str(int(meta.year))))
    if meta.party_family is not None:
        out.append(("party_family", str(int(meta.party_family))))
    if meta.rile is not None:
        out.append(("rile", f"{float(meta.rile):.6g}"))
    return out

