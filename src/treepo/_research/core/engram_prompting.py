"""
Prompt-builder wrappers for Engram-style static memory injection.

These utilities are task-agnostic: they wrap an existing prompt builder and
append a compact "STATIC MEMORY" block derived from the input text.

This mirrors Engram's core idea: offload stereotyped/local pattern handling to
deterministic lookup so model compute can focus on reasoning.

When per-document metadata (party, country, year, etc.) is available, it is
prepended to the STATIC MEMORY block via a ContextVar set by the pipeline.
"""

from __future__ import annotations

import contextvars
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional

from treepo._research.core.engram_memory import (
    EngramMemoryConfig,
    extract_engram_memory_items,
    format_engram_memory_block,
)

if TYPE_CHECKING:
    from treepo._research.core.conditional_memory import ConditionalMemory

ENGRAM_PROMPT_VERSION = "engram_static_memory_v2"

# Per-document metadata injected by the pipeline before calling prompt builders.
# Set this ContextVar around prompt builder calls so the Engram wrapper can
# include structured metadata (party, country, year) in the STATIC MEMORY block.
engram_document_metadata: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("engram_document_metadata", default=None)
)

_prompt_metadata_lock = threading.RLock()
_prompt_metadata_registry: Dict[str, Dict[str, Any]] = {}

_PROMPT_METADATA_KEYS = (
    "doc_id",
    "manifesto_id",
    "party_name",
    "party_abbrev",
    "party_id",
    "country_name",
    "country",
    "country_code",
    "year",
    "date_code",
    "election_date",
    "party_family",
    "language",
    "source",
)

_PROMPT_METADATA_LABELS = {
    "doc_id": "DOC_ID",
    "manifesto_id": "MANIFESTO_ID",
    "party_name": "PARTY",
    "party_abbrev": "PARTY_ABBREV",
    "party_id": "PARTY_ID",
    "country_name": "COUNTRY",
    "country": "COUNTRY",
    "country_code": "COUNTRY_CODE",
    "year": "YEAR",
    "date_code": "DATE",
    "election_date": "ELECTION",
    "party_family": "PARTY_FAMILY",
    "language": "LANGUAGE",
    "source": "SOURCE",
}


def _clean_prompt_metadata_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    rendered = str(value).strip()
    if not rendered:
        return None
    if rendered.lower() in {"nan", "none", "null"}:
        return None
    return rendered[:160]


def sanitize_prompt_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep only non-label contextual metadata safe for prompt injection."""
    if not isinstance(metadata, dict) or not metadata:
        return {}

    sanitized: Dict[str, Any] = {}
    for key in _PROMPT_METADATA_KEYS:
        cleaned = _clean_prompt_metadata_value(metadata.get(key))
        if cleaned is not None:
            sanitized[key] = cleaned
    return sanitized


def register_prompt_metadata_for_doc(doc_id: str, metadata: Optional[Dict[str, Any]]) -> None:
    rendered_doc_id = str(doc_id or "").strip()
    if not rendered_doc_id:
        return
    sanitized = sanitize_prompt_metadata(metadata)
    with _prompt_metadata_lock:
        if sanitized:
            _prompt_metadata_registry[rendered_doc_id] = sanitized
        else:
            _prompt_metadata_registry.pop(rendered_doc_id, None)


def clear_prompt_metadata_registry(doc_ids: Optional[Iterable[str]] = None) -> None:
    with _prompt_metadata_lock:
        if doc_ids is None:
            _prompt_metadata_registry.clear()
            return
        for doc_id in doc_ids:
            _prompt_metadata_registry.pop(str(doc_id or "").strip(), None)


def resolve_prompt_metadata() -> Optional[Dict[str, Any]]:
    """Resolve per-document metadata from context or tournament doc-id registry."""
    current = sanitize_prompt_metadata(engram_document_metadata.get(None))
    if current:
        return current

    try:
        from treepo._research.core.strategy import tournament_doc_id

        doc_id = str(tournament_doc_id.get() or "").strip()
    except Exception:
        doc_id = ""

    if not doc_id:
        return None

    with _prompt_metadata_lock:
        cached = _prompt_metadata_registry.get(doc_id)
        if not cached:
            return None
        return dict(cached)


def format_prompt_metadata_block(
    metadata: Optional[Dict[str, Any]] = None,
    *,
    heading: str = "DOCUMENT METADATA",
) -> str:
    resolved = sanitize_prompt_metadata(metadata) if metadata is not None else resolve_prompt_metadata()
    if not resolved:
        return ""

    # Reuse Engram's compact metadata line when available.
    compact = format_engram_memory_block([], context_metadata=resolved)
    if compact:
        lines = [line for line in compact.splitlines() if line.strip()]
        if len(lines) >= 2:
            return f"{heading} (context only): {lines[1].strip()}"

    parts: List[str] = []
    for key in _PROMPT_METADATA_KEYS:
        value = resolved.get(key)
        if value is None:
            continue
        label = _PROMPT_METADATA_LABELS.get(key, key.upper())
        parts.append(f"{label}: {value}")
    if not parts:
        return ""
    return f"{heading} (context only): {' / '.join(parts)}"


def _inject_memory_into_messages(
    messages: List[Dict[str, str]],
    memory_block: str,
) -> List[Dict[str, str]]:
    if not memory_block or not messages:
        return messages

    # Copy to avoid mutating task-provided builders.
    out: List[Dict[str, str]] = [dict(m) for m in messages]

    for msg in out:
        if msg.get("role") == "system" and msg.get("content"):
            msg["content"] = (
                msg["content"].rstrip()
                + "\n- Preserve any STATIC MEMORY items exactly if they appear.\n"
                + "- Do not output the STATIC MEMORY list.\n"
                + f"- Prompt policy version: {ENGRAM_PROMPT_VERSION}.\n"
            )
            break

    for msg in out:
        if msg.get("role") == "user":
            msg["content"] = (msg.get("content") or "").rstrip() + "\n\n" + memory_block
            break

    return out


def _inject_metadata_into_messages(
    messages: List[Dict[str, str]],
    metadata_block: str,
) -> List[Dict[str, str]]:
    if not metadata_block or not messages:
        return messages

    out: List[Dict[str, str]] = [dict(m) for m in messages]
    for msg in out:
        if msg.get("role") == "system" and msg.get("content"):
            msg["content"] = (
                msg["content"].rstrip()
                + "\n- Use DOCUMENT METADATA as contextual priors only (time/place/party)."
                + "\n- Do not treat metadata as a direct score label."
            )
            break

    for msg in out:
        if msg.get("role") == "user":
            msg["content"] = (msg.get("content") or "").rstrip() + "\n\n" + metadata_block
            break

    return out


def wrap_summarize_prompt_with_engram_memory(
    prompt_fn: Callable[[str, str], List[Dict[str, str]]],
    config: EngramMemoryConfig,
    memory: Optional["ConditionalMemory"] = None,
) -> Callable[[str, str], List[Dict[str, str]]]:
    """Wrap a (text, rubric)->messages prompt builder with Engram static memory.

    When ``memory`` is provided, extracted entities are cached in
    ConditionalMemory so repeated documents skip regex extraction.

    Per-document metadata is read from the ``engram_document_metadata``
    ContextVar (set by the pipeline) and prepended to the STATIC MEMORY block.
    """

    def wrapped(text: str, rubric: str) -> List[Dict[str, str]]:
        base = prompt_fn(text, rubric)
        items = extract_engram_memory_items(text, config, memory=memory)
        meta = resolve_prompt_metadata()
        block = format_engram_memory_block(items, context_metadata=meta)
        return _inject_memory_into_messages(base, block)

    return wrapped


def wrap_merge_prompt_with_engram_memory(
    prompt_fn: Callable[[str, str, str], List[Dict[str, str]]],
    config: EngramMemoryConfig,
    memory: Optional["ConditionalMemory"] = None,
) -> Callable[[str, str, str], List[Dict[str, str]]]:
    """Wrap a (left, right, rubric)->messages prompt builder with Engram static memory.

    When ``memory`` is provided, extracted entities are cached in
    ConditionalMemory so repeated documents skip regex extraction.

    Per-document metadata is read from the ``engram_document_metadata``
    ContextVar (set by the pipeline) and prepended to the STATIC MEMORY block.
    """

    def wrapped(left: str, right: str, rubric: str) -> List[Dict[str, str]]:
        base = prompt_fn(left, right, rubric)
        joined = f"{left}\n\n{right}"
        items = extract_engram_memory_items(joined, config, memory=memory)
        meta = resolve_prompt_metadata()
        block = format_engram_memory_block(items, context_metadata=meta)
        return _inject_memory_into_messages(base, block)

    return wrapped


def wrap_score_prompt_with_engram_metadata(
    prompt_fn: Callable[[str, str], List[Dict[str, str]]],
) -> Callable[[str, str], List[Dict[str, str]]]:
    """Wrap score prompts with safe per-document metadata context."""

    def wrapped(summary: str, task_context: str) -> List[Dict[str, str]]:
        base = prompt_fn(summary, task_context)
        metadata_block = format_prompt_metadata_block()
        return _inject_metadata_into_messages(base, metadata_block)

    return wrapped


__all__ = [
    "ENGRAM_PROMPT_VERSION",
    "engram_document_metadata",
    "sanitize_prompt_metadata",
    "register_prompt_metadata_for_doc",
    "clear_prompt_metadata_registry",
    "resolve_prompt_metadata",
    "format_prompt_metadata_block",
    "wrap_summarize_prompt_with_engram_memory",
    "wrap_merge_prompt_with_engram_memory",
    "wrap_score_prompt_with_engram_metadata",
]
