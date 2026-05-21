"""
Engram-style "conditional memory" utilities for ThinkingTrees.

This module implements a cheap, deterministic "static memory" extractor for
local/stereotyped patterns (named entities, IDs, URLs, etc.) that are easy to
lose during compression. The extracted items can be injected into prompts so
the model can treat them like a lookup table rather than reconstructing them
through generation.

Implementation note:
The text normalization pipeline is adapted from DeepSeek's Engram demo
(`deepseek-ai/Engram`, `engram_demo_v1.py`, Apache-2.0). We keep a stdlib
fallback to avoid a hard dependency on `tokenizers`.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

from treepo._research.core.conditional_memory import canonical_hash

if TYPE_CHECKING:
    from treepo._research.core.conditional_memory import ConditionalMemory

_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}\b"
)
_URL_RE = re.compile(r"\bhttps?://[^\s<>()\"']+\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]{8,}\b")


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


class EngramTextNormalizer:
    """
    Normalize text to a canonical form suitable for deduplication.

    Tries to match Engram's demo normalizer:
      NFKC → NFD → strip accents → lowercase → whitespace collapse → strip
    """

    def __init__(self):
        self._use_tokenizers = False
        self._normalizer = None
        try:
            from tokenizers import Regex, normalizers  # type: ignore

            sentinel = "\uE000"
            self._normalizer = normalizers.Sequence(
                [
                    normalizers.NFKC(),
                    normalizers.NFD(),
                    normalizers.StripAccents(),
                    normalizers.Lowercase(),
                    normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                    normalizers.Replace(Regex(r"^ $"), sentinel),
                    normalizers.Strip(),
                    normalizers.Replace(sentinel, " "),
                ]
            )
            self._use_tokenizers = True
        except Exception:
            self._use_tokenizers = False
            self._normalizer = None

    def normalize(self, text: str) -> str:
        if not text:
            return ""

        raw = str(text)
        if self._use_tokenizers and self._normalizer is not None:
            try:
                return str(self._normalizer.normalize_str(raw))
            except Exception:
                # Fall through to stdlib path.
                pass

        out = unicodedata.normalize("NFKC", raw)
        out = _strip_accents(out)
        out = out.casefold()
        out = re.sub(r"[ \t\r\n]+", " ", out).strip()
        return out


@dataclass(frozen=True)
class EngramMemoryConfig:
    """Controls extraction and formatting of Engram-style static memory."""

    enabled: bool = False
    max_items: int = 32
    max_chars: int = 1200
    max_item_chars: int = 120

    include_named_entities: bool = True
    include_single_proper_nouns: bool = True
    include_urls: bool = True
    include_emails: bool = True
    include_uuids: bool = True
    include_hex: bool = True
    include_numbers: bool = True
    include_identifiers: bool = True
    include_acronyms: bool = True
    skip_shouty_phrases: bool = True

    # Heuristics / thresholds.
    max_named_entity_words: int = 6
    min_single_proper_len: int = 10
    min_number_digits: int = 4
    min_identifier_len: int = 10
    min_acronym_len: int = 2
    max_acronym_len: int = 5


def _stable_json_dumps(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _config_cache_key(config: EngramMemoryConfig) -> str:
    """Stable cache discriminator for ConditionalMemory namespaces."""
    return canonical_hash(_stable_json_dumps(asdict(config)), normalize=False)


def _find_spans(pattern: re.Pattern[str], text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in pattern.finditer(text)]


def _dedup_by_normalized(
    spans: Iterable[Tuple[int, int, str]],
    *,
    normalizer: EngramTextNormalizer,
) -> List[Tuple[int, int, str]]:
    seen: set[str] = set()
    out: List[Tuple[int, int, str]] = []
    for start, end, value in sorted(spans, key=lambda s: (s[0], -(s[1] - s[0]))):
        key = normalizer.normalize(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((start, end, value))
    return out


def _extract_named_entities(
    text: str,
    *,
    max_words: int,
    include_single: bool,
    min_single_len: int,
) -> List[Tuple[int, int, str]]:
    connectors = r"(?:of|the|and|de|van|von|da|dos|di|la|le|du|del|&)"
    multi = re.compile(
        rf"\b[A-Z][a-zA-Z]+(?:\s+(?:[A-Z][a-zA-Z]+|{connectors})){{1,{max(1, max_words - 1)}}}\b"
    )
    spans: List[Tuple[int, int, str]] = _find_spans(multi, text)

    if include_single and min_single_len > 0:
        single = re.compile(rf"\b[A-Z][a-zA-Z]{{{int(min_single_len) - 1},}}\b")
        spans.extend(_find_spans(single, text))

    return spans


def _extract_numbers(text: str, *, min_digits: int) -> List[Tuple[int, int, str]]:
    if min_digits <= 0:
        min_digits = 1
    number = re.compile(rf"\b\d[\d,]{{{int(min_digits) - 1},}}(?:\.\d+)?%?\b")
    return _find_spans(number, text)


def _extract_identifiers(text: str, *, min_len: int) -> List[Tuple[int, int, str]]:
    if min_len <= 0:
        min_len = 1
    snake = re.compile(rf"\b[a-zA-Z_][a-zA-Z0-9_]{{{int(min_len) - 1},}}\b")
    camel = re.compile(rf"\b[a-z]+[A-Z][A-Za-z0-9]{{{max(1, int(min_len) - 2)},}}\b")
    spans: List[Tuple[int, int, str]] = []
    for start, end, value in _find_spans(snake, text):
        if "_" in value or any(ch.isdigit() for ch in value):
            spans.append((start, end, value))
    spans.extend(_find_spans(camel, text))
    return spans


def _extract_acronyms(text: str, *, min_len: int, max_len: int) -> List[Tuple[int, int, str]]:
    min_len = max(2, int(min_len))
    max_len = max(min_len, int(max_len))
    # Unicode letters only (no digits/underscore), length-bounded.
    token = re.compile(rf"\b[^\W\d_]{{{min_len},{max_len}}}\b", flags=re.UNICODE)
    spans: List[Tuple[int, int, str]] = []
    for start, end, value in _find_spans(token, text):
        if not value:
            continue
        if not value.isalpha():
            continue
        # Only include cased all-uppercase tokens (skip scripts without case).
        if value == value.upper() and value != value.lower():
            spans.append((start, end, value))
    return spans


def _is_shouty_phrase(value: str) -> bool:
    """Heuristic to drop ALL-CAPS multiword headings that crowd out useful items."""
    rendered = str(value or "")
    if " " not in rendered:
        return False
    words = [w for w in rendered.split() if w]
    if len(words) < 3:
        return False
    letters = [ch for ch in rendered if ch.isalpha()]
    if len(letters) < 12:
        return False
    upper = sum(1 for ch in letters if ch.isupper())
    return (upper / float(len(letters))) >= 0.9


def _is_all_caps_token(value: str) -> bool:
    rendered = str(value or "")
    letters = [ch for ch in rendered if ch.isalpha()]
    if not letters:
        return False
    upper = sum(1 for ch in letters if ch.isupper())
    lower = sum(1 for ch in letters if ch.islower())
    # Only treat as ALL-CAPS if the script is cased (has upper/lower).
    return upper > 0 and lower == 0


def _is_mostly_alpha_phrase(value: str) -> bool:
    """True for phrases that are essentially just letters + separators."""
    rendered = str(value or "")
    cleaned = re.sub(r"[\s'’\"“”\\-–—]", "", rendered)
    return bool(cleaned) and cleaned.isalpha()


def extract_engram_memory_items(
    text: str,
    config: EngramMemoryConfig,
    memory: Optional["ConditionalMemory"] = None,
) -> List[str]:
    """
    Extract a compact list of verbatim strings worth preserving exactly.

    When a ``ConditionalMemory`` instance is provided, the result is cached
    under the ``"engram_items"`` score head so repeated documents skip the
    full regex extraction pipeline.

    Returns:
        A list of strings in original surface form.
    """
    if not config.enabled:
        return []
    if not text or not str(text).strip():
        return []

    # Check ConditionalMemory for cached extraction results.
    if memory is not None:
        namespace = f"engram_items:{_config_cache_key(config)}:{memory.namespace_version}"
        key = canonical_hash(text)
        cached = memory.get_json(namespace, key)
        if isinstance(cached, list) and all(isinstance(x, str) for x in cached):
            return list(cached)

    normalizer = EngramTextNormalizer()
    raw = str(text)

    spans: List[Tuple[int, int, str]] = []
    if config.include_urls:
        spans.extend(_find_spans(_URL_RE, raw))
    if config.include_emails:
        spans.extend(_find_spans(_EMAIL_RE, raw))
    if config.include_uuids:
        spans.extend(_find_spans(_UUID_RE, raw))
    if config.include_hex:
        spans.extend(_find_spans(_HEX_RE, raw))
        spans.extend(_find_spans(re.compile(r"\b[0-9a-fA-F]{16,}\b"), raw))

    if config.include_numbers:
        spans.extend(_extract_numbers(raw, min_digits=config.min_number_digits))

    if config.include_identifiers:
        spans.extend(_extract_identifiers(raw, min_len=config.min_identifier_len))

    if config.include_acronyms:
        spans.extend(_extract_acronyms(raw, min_len=config.min_acronym_len, max_len=config.max_acronym_len))

    if config.include_named_entities:
        spans.extend(
            _extract_named_entities(
                raw,
                max_words=config.max_named_entity_words,
                include_single=config.include_single_proper_nouns,
                min_single_len=config.min_single_proper_len,
            )
        )

    deduped = _dedup_by_normalized(spans, normalizer=normalizer)

    items: List[str] = []
    total_chars = 0
    max_items = max(0, int(config.max_items))
    max_chars = max(0, int(config.max_chars))
    max_item_chars = max(0, int(getattr(config, "max_item_chars", 0) or 0))

    for _, __, value in deduped:
        if max_items and len(items) >= max_items:
            break
        if max_item_chars and len(value) > max_item_chars:
            continue
        if config.skip_shouty_phrases:
            # Drop ALL-CAPS headings and other "shouty" fragments.
            if _is_shouty_phrase(value):
                continue
            if (
                _is_all_caps_token(value)
                and len(value) > int(config.max_acronym_len)
                and _is_mostly_alpha_phrase(value)
            ):
                continue
        if max_chars and (total_chars + len(value)) > max_chars:
            continue
        items.append(value)
        total_chars += len(value)

    # Cache in ConditionalMemory for cross-run reuse.
    if memory is not None and items:
        namespace = f"engram_items:{_config_cache_key(config)}:{memory.namespace_version}"
        key = canonical_hash(text)
        memory.set_json(namespace, key, items)

    return items


def format_metadata_preamble(metadata: dict) -> str:
    """Format structured document metadata for injection into STATIC MEMORY.

    Picks out useful fields (party, country, year) when available and formats
    them as a compact preamble above the extracted items.
    """
    if not metadata:
        return ""

    field_map = [
        ("party_name", "PARTY"),
        ("party_abbrev", "PARTY_ABBREV"),
        ("country_name", "COUNTRY"),
        ("country_code", "COUNTRY_CODE"),
        ("year", "YEAR"),
        ("date_code", "DATE"),
        ("election_date", "ELECTION"),
    ]
    parts: list[str] = []
    for attr, label in field_map:
        val = metadata.get(attr)
        if val is not None and str(val).strip():
            parts.append(f"{label}: {val}")
    return " / ".join(parts)


def format_engram_memory_block(
    items: List[str],
    context_metadata: Optional[dict] = None,
) -> str:
    """Render memory items as a prompt-ready block.

    When ``context_metadata`` is provided, a compact preamble with structured
    document metadata (party, country, year, etc.) is prepended.
    """
    preamble = format_metadata_preamble(context_metadata) if context_metadata else ""
    if not items and not preamble:
        return ""
    lines: list[str] = [
        "STATIC MEMORY (verbatim strings from the input; preserve exactly if relevant):",
    ]
    if preamble:
        lines.append(preamble)
    lines.extend(f"- {item}" for item in items)
    return "\n".join(lines).strip()
