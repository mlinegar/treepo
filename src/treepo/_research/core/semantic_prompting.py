"""
Prompt-builder wrappers for semantic memory injection.

This layer is additive to Engram static-memory prompting and consumes
precomputed per-document retrieval payloads.
"""

from __future__ import annotations

import contextvars
import threading
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from treepo._research.core.semantic_memory import SemanticNeighbor

SEMANTIC_PROMPT_VERSION = "semantic_memory_v1"

# Per-document semantic memory payload (set by the pipeline when available).
semantic_document_memory: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("semantic_document_memory", default=None)
)

_registry_lock = threading.RLock()
_semantic_memory_registry: Dict[str, Dict[str, Any]] = {}


def register_semantic_memory_for_doc(doc_id: str, payload: Dict[str, Any]) -> None:
    rendered_doc_id = str(doc_id or "").strip()
    if not rendered_doc_id:
        return
    with _registry_lock:
        _semantic_memory_registry[rendered_doc_id] = dict(payload or {})


def clear_semantic_memory_registry(doc_ids: Optional[Iterable[str]] = None) -> None:
    with _registry_lock:
        if doc_ids is None:
            _semantic_memory_registry.clear()
            return
        for doc_id in doc_ids:
            _semantic_memory_registry.pop(str(doc_id or "").strip(), None)


def _neighbor_to_dict(row: Any) -> Dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    if isinstance(row, SemanticNeighbor):
        return asdict(row)
    if is_dataclass(row):
        return asdict(row)
    out: Dict[str, Any] = {}
    for key in (
        "doc_id",
        "score",
        "similarity",
        "scope",
        "year",
        "date_code",
        "rile",
        "delta_rile",
        "party_id",
        "country_code",
        "snippets",
    ):
        value = getattr(row, key, None)
        if value is not None:
            out[key] = value
    return out


def format_semantic_memory_block(payload: Optional[Dict[str, Any]]) -> str:
    if not payload:
        return ""
    neighbors_raw = payload.get("neighbors", [])
    if not isinstance(neighbors_raw, list) or not neighbors_raw:
        return ""

    lines: List[str] = [
        "SEMANTIC MEMORY (cross-document retrieved context; use only if relevant):",
    ]
    for idx, item in enumerate(neighbors_raw, start=1):
        row = _neighbor_to_dict(item)
        if not row:
            continue
        doc_id = str(row.get("doc_id", "") or "").strip() or "unknown_doc"
        score = row.get("score")
        similarity = row.get("similarity")
        scope = str(row.get("scope", "") or "").strip()
        year = row.get("year")
        rile = row.get("rile")
        delta = row.get("delta_rile")

        parts = [f"{idx}. doc={doc_id}"]
        if scope:
            parts.append(f"scope={scope}")
        if year is not None:
            parts.append(f"year={year}")
        if similarity is not None:
            parts.append(f"sim={float(similarity):.4f}")
        if score is not None:
            parts.append(f"score={float(score):.4f}")
        if rile is not None:
            parts.append(f"rile={float(rile):+.2f}")
        if delta is not None:
            parts.append(f"delta={float(delta):+.3f}")
        lines.append(" - " + " | ".join(parts))

        snippets = row.get("snippets", [])
        if isinstance(snippets, list):
            for snip in snippets:
                if not isinstance(snip, dict):
                    continue
                text = str(snip.get("text", "") or "").strip()
                if not text:
                    continue
                sim = snip.get("similarity")
                prefix = "   · snippet"
                if sim is not None:
                    prefix += f" sim={float(sim):.4f}"
                lines.append(f"{prefix}: {text}")

    if len(lines) <= 1:
        return ""
    return "\n".join(lines).strip()


def _resolve_payload() -> Optional[Dict[str, Any]]:
    payload = semantic_document_memory.get(None)
    if payload:
        return payload
    try:
        from treepo._research.core.strategy import tournament_doc_id

        doc_id = str(tournament_doc_id.get() or "").strip()
    except Exception:
        doc_id = ""
    if not doc_id:
        return None
    with _registry_lock:
        cached = _semantic_memory_registry.get(doc_id)
        if cached is None:
            return None
        return dict(cached)


def _inject_memory_into_messages(messages: List[Dict[str, str]], block: str) -> List[Dict[str, str]]:
    if not block or not messages:
        return messages
    out = [dict(m) for m in messages]
    for msg in out:
        if msg.get("role") == "system" and msg.get("content"):
            msg["content"] = (
                msg["content"].rstrip()
                + "\n- Use SEMANTIC MEMORY only as supporting context when relevant.\n"
                + "- Do not copy unsupported claims from SEMANTIC MEMORY.\n"
                + "- Do not output the SEMANTIC MEMORY list.\n"
                + f"- Prompt policy version: {SEMANTIC_PROMPT_VERSION}.\n"
            )
            break
    for msg in out:
        if msg.get("role") == "user":
            msg["content"] = (msg.get("content") or "").rstrip() + "\n\n" + block
            break
    return out


def wrap_summarize_prompt_with_semantic_memory(
    prompt_fn: Callable[[str, str], List[Dict[str, str]]],
) -> Callable[[str, str], List[Dict[str, str]]]:
    def wrapped(text: str, rubric: str) -> List[Dict[str, str]]:
        base = prompt_fn(text, rubric)
        payload = _resolve_payload()
        block = format_semantic_memory_block(payload)
        return _inject_memory_into_messages(base, block)

    return wrapped


def wrap_merge_prompt_with_semantic_memory(
    prompt_fn: Callable[[str, str, str], List[Dict[str, str]]],
) -> Callable[[str, str, str], List[Dict[str, str]]]:
    def wrapped(left: str, right: str, rubric: str) -> List[Dict[str, str]]:
        base = prompt_fn(left, right, rubric)
        payload = _resolve_payload()
        block = format_semantic_memory_block(payload)
        return _inject_memory_into_messages(base, block)

    return wrapped


__all__ = [
    "SEMANTIC_PROMPT_VERSION",
    "semantic_document_memory",
    "register_semantic_memory_for_doc",
    "clear_semantic_memory_registry",
    "format_semantic_memory_block",
    "wrap_summarize_prompt_with_semantic_memory",
    "wrap_merge_prompt_with_semantic_memory",
]
