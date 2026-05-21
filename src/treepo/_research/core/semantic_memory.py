"""
Persistent semantic memory index for multilingual retrieval.

This module complements exact/string-level Engram memory with vector retrieval
over document/chunk embeddings and soft-recency temporal weighting.
"""

from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import json
import math
import threading
import time
import zlib

import numpy as np

from treepo._research.core.conditional_memory import canonical_hash


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(v) for v in value]
    return value


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _normalize_vector(vector: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty vector")
    denom = float(np.linalg.norm(arr) + 1e-12)
    return (arr / denom).astype(np.float32, copy=False)


def _encode_f32z(vector: np.ndarray) -> tuple[bytes, Dict[str, Any]]:
    arr = array("f", (float(v) for v in np.asarray(vector, dtype=np.float32).reshape(-1)))
    raw = arr.tobytes()
    return zlib.compress(raw), {"dtype": "float32", "dim": int(len(arr))}


def _decode_f32z(blob: bytes) -> np.ndarray:
    raw = zlib.decompress(bytes(blob))
    arr = array("f")
    arr.frombytes(raw)
    return np.asarray(arr, dtype=np.float32)


def _clip_text(value: str, max_chars: int) -> str:
    rendered = " ".join(str(value or "").split())
    if max_chars <= 0:
        return rendered
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max(0, int(max_chars) - 1)].rstrip() + "…"


@dataclass(frozen=True)
class SemanticMemoryConfig:
    enabled: bool = False
    index_dir: Path = Path("outputs/semantic_memory")
    top_k: int = 5
    lambda_year: float = 0.08
    scope_bonus_party_country: float = 0.05
    scope_bonus_family_country: float = 0.02
    index_granularity: str = "doc_chunk"  # doc | chunk | doc_chunk
    max_windows: int = 0  # 0 = unlimited
    update_policy: str = "post_score"  # post_score only in phase 1
    inject_prompts: bool = True
    model_features: bool = True
    temporal_mode: bool = True
    max_chunk_snippets_per_neighbor: int = 2
    max_snippet_chars: int = 240

    def resolved_index_dir(self) -> Path:
        return Path(self.index_dir).expanduser()


@dataclass(frozen=True)
class SemanticMemoryEntry:
    entry_id: str
    kind: str  # doc | chunk
    doc_id: str
    vector_path: str
    vector_dim: int

    party_id: Optional[int] = None
    country_code: Optional[int] = None
    party_family: Optional[int] = None
    year: Optional[int] = None
    date_code: Optional[int] = None
    rile: Optional[float] = None
    delta_rile: Optional[float] = None

    chunk_index: Optional[int] = None
    chunk_text: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: float(time.time()))


@dataclass(frozen=True)
class SemanticNeighbor:
    entry_id: str
    doc_id: str
    similarity: float
    score: float
    recency_weight: float
    scope: str
    year_gap: Optional[int]

    party_id: Optional[int] = None
    country_code: Optional[int] = None
    party_family: Optional[int] = None
    year: Optional[int] = None
    date_code: Optional[int] = None
    rile: Optional[float] = None
    delta_rile: Optional[float] = None
    snippets: List[Dict[str, Any]] = field(default_factory=list)


def normalize_rile_delta(
    *,
    current_rile: Optional[float],
    previous_rile: Optional[float],
    rile_range: float = 200.0,
) -> Optional[float]:
    current = _safe_float(current_rile)
    previous = _safe_float(previous_rile)
    if current is None or previous is None:
        return None
    if abs(float(rile_range)) <= 1e-12:
        return None
    raw = (float(current) - float(previous)) / float(rile_range)
    return max(-1.0, min(1.0, float(raw)))


class SemanticMemoryIndex:
    """Persistent in-memory+disk semantic index with doc/chunk vectors."""

    SCHEMA_VERSION = 1

    def __init__(self, config: Optional[SemanticMemoryConfig] = None):
        self.config = config or SemanticMemoryConfig()
        self._index_dir = self.config.resolved_index_dir()
        self._vectors_dir = self._index_dir / "vectors"
        self._entries_path = self._index_dir / "entries.jsonl"
        self._manifest_path = self._index_dir / "manifest.json"
        self._lock = threading.RLock()

        self._entries: List[SemanticMemoryEntry] = []
        self._vectors: List[np.ndarray] = []
        self._entry_id_to_idx: Dict[str, int] = {}
        self._doc_indices: List[int] = []
        self._chunk_indices_by_doc: Dict[str, List[int]] = {}
        self._writes = 0
        self._loads = 0

        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._vectors_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def _vector_blob_path(self, entry_id: str) -> Path:
        return self._vectors_dir / f"{entry_id}.f32z"

    def _build_entry_id(
        self,
        *,
        kind: str,
        doc_id: str,
        source_key: str,
        year: Optional[int],
        date_code: Optional[int],
        chunk_index: Optional[int],
    ) -> str:
        payload = {
            "kind": str(kind),
            "doc_id": str(doc_id),
            "source_key": str(source_key),
            "year": _safe_int(year),
            "date_code": _safe_int(date_code),
            "chunk_index": _safe_int(chunk_index),
        }
        stable = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return canonical_hash(stable, normalize=False)

    def _register_in_memory(self, entry: SemanticMemoryEntry, vector: np.ndarray) -> None:
        idx = len(self._entries)
        self._entries.append(entry)
        self._vectors.append(vector)
        self._entry_id_to_idx[entry.entry_id] = idx
        if entry.kind == "doc":
            self._doc_indices.append(idx)
        elif entry.kind == "chunk":
            self._chunk_indices_by_doc.setdefault(entry.doc_id, []).append(idx)

    def _append_entry_to_disk(self, entry: SemanticMemoryEntry, vector: np.ndarray) -> None:
        blob_path = self._vector_blob_path(entry.entry_id)
        blob, _meta = _encode_f32z(vector)
        blob_path.write_bytes(blob)
        with self._entries_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_sanitize_json(asdict(entry)), ensure_ascii=False) + "\n")
        self._writes += 1
        self._write_manifest()

    def _write_manifest(self) -> None:
        payload = {
            "schema_version": int(self.SCHEMA_VERSION),
            "created_at": float(time.time()),
            "entries_total": int(len(self._entries)),
            "doc_entries": int(len(self._doc_indices)),
            "chunk_entries": int(sum(len(v) for v in self._chunk_indices_by_doc.values())),
            "writes": int(self._writes),
            "loads": int(self._loads),
            "config": _sanitize_json(asdict(self.config)),
        }
        self._manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load(self) -> None:
        if not self._entries_path.exists():
            self._write_manifest()
            return
        loaded = 0
        with self._entries_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                try:
                    entry = SemanticMemoryEntry(
                        entry_id=str(payload.get("entry_id", "") or "").strip(),
                        kind=str(payload.get("kind", "") or "").strip(),
                        doc_id=str(payload.get("doc_id", "") or "").strip(),
                        vector_path=str(payload.get("vector_path", "") or "").strip(),
                        vector_dim=int(payload.get("vector_dim", 0) or 0),
                        party_id=_safe_int(payload.get("party_id")),
                        country_code=_safe_int(payload.get("country_code")),
                        party_family=_safe_int(payload.get("party_family")),
                        year=_safe_int(payload.get("year")),
                        date_code=_safe_int(payload.get("date_code")),
                        rile=_safe_float(payload.get("rile")),
                        delta_rile=_safe_float(payload.get("delta_rile")),
                        chunk_index=_safe_int(payload.get("chunk_index")),
                        chunk_text=str(payload.get("chunk_text", "") or "") or None,
                        provenance=payload.get("provenance", {}) if isinstance(payload.get("provenance", {}), dict) else {},
                        created_at=float(payload.get("created_at", time.time()) or time.time()),
                    )
                except Exception:
                    continue
                if not entry.entry_id or not entry.doc_id or entry.kind not in {"doc", "chunk"}:
                    continue
                blob_path = Path(entry.vector_path)
                if not blob_path.is_absolute():
                    blob_path = (self._index_dir / blob_path).resolve()
                if not blob_path.exists():
                    blob_path = self._vector_blob_path(entry.entry_id)
                    if not blob_path.exists():
                        continue
                try:
                    vec = _normalize_vector(_decode_f32z(blob_path.read_bytes()))
                except Exception:
                    continue
                self._register_in_memory(entry, vec)
                loaded += 1
        self._loads = loaded
        self._write_manifest()

    def report(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries_total": int(len(self._entries)),
                "doc_entries": int(len(self._doc_indices)),
                "chunk_entries": int(sum(len(v) for v in self._chunk_indices_by_doc.values())),
                "index_dir": str(self._index_dir),
                "writes": int(self._writes),
                "loads": int(self._loads),
                "enabled": bool(self.config.enabled),
            }

    def get_temporal_predecessor(
        self,
        *,
        party_id: Optional[int],
        country_code: Optional[int],
        year: Optional[int],
        date_code: Optional[int],
        exclude_doc_id: Optional[str] = None,
    ) -> Optional[SemanticMemoryEntry]:
        pid = _safe_int(party_id)
        ccode = _safe_int(country_code)
        q_year = _safe_int(year)
        q_date = _safe_int(date_code)
        if pid is None or ccode is None:
            return None

        best_entry: Optional[SemanticMemoryEntry] = None
        best_key: Optional[tuple[int, int]] = None
        for idx in self._doc_indices:
            entry = self._entries[idx]
            if exclude_doc_id and entry.doc_id == str(exclude_doc_id):
                continue
            if entry.party_id != pid or entry.country_code != ccode:
                continue
            if q_year is not None and entry.year is not None and entry.year > q_year:
                continue
            if (
                q_year is not None
                and q_date is not None
                and entry.year == q_year
                and entry.date_code is not None
                and entry.date_code >= q_date
            ):
                continue
            candidate_key = (
                int(entry.year) if entry.year is not None else -10**9,
                int(entry.date_code) if entry.date_code is not None else -10**9,
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_entry = entry
        return best_entry

    def add_entry(
        self,
        *,
        kind: str,
        doc_id: str,
        vector: Sequence[float],
        metadata: Optional[Dict[str, Any]] = None,
        source_key: Optional[str] = None,
        chunk_index: Optional[int] = None,
        chunk_text: Optional[str] = None,
    ) -> Optional[SemanticMemoryEntry]:
        if not self.config.enabled:
            return None
        rendered_kind = str(kind or "").strip().lower()
        if rendered_kind not in {"doc", "chunk"}:
            raise ValueError(f"Unsupported semantic entry kind: {kind!r}")
        rendered_doc_id = str(doc_id or "").strip()
        if not rendered_doc_id:
            raise ValueError("doc_id is required")
        meta = metadata or {}

        source = str(source_key or "")
        if not source:
            source = canonical_hash(
                json.dumps(
                    {
                        "doc_id": rendered_doc_id,
                        "kind": rendered_kind,
                        "chunk_index": _safe_int(chunk_index),
                        "date_code": _safe_int(meta.get("date_code")),
                        "year": _safe_int(meta.get("year")),
                        "chunk_hash": canonical_hash(str(chunk_text or ""), normalize=True) if chunk_text else "",
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ),
                normalize=False,
            )

        entry_id = self._build_entry_id(
            kind=rendered_kind,
            doc_id=rendered_doc_id,
            source_key=source,
            year=_safe_int(meta.get("year")),
            date_code=_safe_int(meta.get("date_code")),
            chunk_index=_safe_int(chunk_index),
        )

        vector_norm = _normalize_vector(vector)
        with self._lock:
            existing_idx = self._entry_id_to_idx.get(entry_id)
            if existing_idx is not None:
                return self._entries[existing_idx]

            vector_path = self._vector_blob_path(entry_id)
            entry = SemanticMemoryEntry(
                entry_id=entry_id,
                kind=rendered_kind,
                doc_id=rendered_doc_id,
                vector_path=str(vector_path.relative_to(self._index_dir)),
                vector_dim=int(vector_norm.shape[0]),
                party_id=_safe_int(meta.get("party_id")),
                country_code=_safe_int(meta.get("country_code")),
                party_family=_safe_int(meta.get("party_family")),
                year=_safe_int(meta.get("year")),
                date_code=_safe_int(meta.get("date_code")),
                rile=_safe_float(meta.get("rile")),
                delta_rile=_safe_float(meta.get("delta_rile")),
                chunk_index=_safe_int(chunk_index),
                chunk_text=(str(chunk_text) if chunk_text is not None else None),
                provenance=_sanitize_json(meta.get("provenance", {})) if isinstance(meta.get("provenance", {}), dict) else {},
                created_at=float(time.time()),
            )
            self._register_in_memory(entry, vector_norm)
            self._append_entry_to_disk(entry, vector_norm)
            return entry

    def add_document(
        self,
        *,
        doc_id: str,
        vector: Sequence[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SemanticMemoryEntry]:
        meta = dict(metadata or {})
        if meta.get("delta_rile") is None:
            predecessor = self.get_temporal_predecessor(
                party_id=meta.get("party_id"),
                country_code=meta.get("country_code"),
                year=meta.get("year"),
                date_code=meta.get("date_code"),
                exclude_doc_id=doc_id,
            )
            if predecessor is not None:
                delta = normalize_rile_delta(
                    current_rile=_safe_float(meta.get("rile")),
                    previous_rile=predecessor.rile,
                    rile_range=200.0,
                )
                if delta is not None:
                    meta["delta_rile"] = float(delta)
        return self.add_entry(kind="doc", doc_id=doc_id, vector=vector, metadata=meta)

    def add_chunks(
        self,
        *,
        doc_id: str,
        vectors: Sequence[Sequence[float]],
        chunk_texts: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        if not self.config.enabled:
            return 0
        count = 0
        texts = list(chunk_texts) if chunk_texts is not None else []
        for idx, vector in enumerate(vectors):
            snippet = texts[idx] if idx < len(texts) else None
            entry = self.add_entry(
                kind="chunk",
                doc_id=doc_id,
                vector=vector,
                metadata=metadata,
                chunk_index=idx,
                chunk_text=snippet,
            )
            if entry is not None:
                count += 1
        return int(count)

    def _scope_of(self, entry: SemanticMemoryEntry, query_meta: Dict[str, Any]) -> str:
        q_party = _safe_int(query_meta.get("party_id"))
        q_country = _safe_int(query_meta.get("country_code"))
        q_family = _safe_int(query_meta.get("party_family"))
        if (
            q_party is not None
            and q_country is not None
            and entry.party_id == q_party
            and entry.country_code == q_country
        ):
            return "same_party_country"
        if (
            q_family is not None
            and q_country is not None
            and entry.party_family == q_family
            and entry.country_code == q_country
        ):
            return "same_family_country"
        return "global"

    def _is_temporally_allowed(self, entry: SemanticMemoryEntry, query_meta: Dict[str, Any]) -> bool:
        if not self.config.temporal_mode:
            return True
        q_year = _safe_int(query_meta.get("year"))
        q_date = _safe_int(query_meta.get("date_code"))
        if q_year is None:
            return True
        if entry.year is None:
            return True
        if entry.year > q_year:
            return False
        if q_date is not None and entry.year == q_year and entry.date_code is not None and entry.date_code > q_date:
            return False
        return True

    def query(
        self,
        *,
        query_vector: Sequence[float],
        query_meta: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        exclude_doc_id: Optional[str] = None,
    ) -> List[SemanticNeighbor]:
        with self._lock:
            if not self._doc_indices:
                return []
            q_meta = dict(query_meta or {})
            qvec = _normalize_vector(query_vector)
            q_year = _safe_int(q_meta.get("year"))
            k = int(top_k if top_k is not None else self.config.top_k)
            k = max(1, k)

            buckets: Dict[str, List[SemanticNeighbor]] = {
                "same_party_country": [],
                "same_family_country": [],
                "global": [],
            }
            for idx in self._doc_indices:
                entry = self._entries[idx]
                if exclude_doc_id and entry.doc_id == str(exclude_doc_id):
                    continue
                if not self._is_temporally_allowed(entry, q_meta):
                    continue

                similarity = float(np.dot(qvec, self._vectors[idx]))
                year_gap: Optional[int] = None
                recency_weight = 1.0
                if q_year is not None and entry.year is not None:
                    year_gap = max(0, int(q_year) - int(entry.year))
                    recency_weight = float(math.exp(-float(self.config.lambda_year) * float(year_gap)))

                scope = self._scope_of(entry, q_meta)
                if scope == "same_party_country":
                    bonus = float(self.config.scope_bonus_party_country)
                elif scope == "same_family_country":
                    bonus = float(self.config.scope_bonus_family_country)
                else:
                    bonus = 0.0

                score = float(similarity * recency_weight + bonus)
                buckets[scope].append(
                    SemanticNeighbor(
                        entry_id=entry.entry_id,
                        doc_id=entry.doc_id,
                        similarity=float(similarity),
                        score=float(score),
                        recency_weight=float(recency_weight),
                        scope=scope,
                        year_gap=year_gap,
                        party_id=entry.party_id,
                        country_code=entry.country_code,
                        party_family=entry.party_family,
                        year=entry.year,
                        date_code=entry.date_code,
                        rile=entry.rile,
                        delta_rile=entry.delta_rile,
                        snippets=[],
                    )
                )

            for scope in buckets:
                buckets[scope].sort(key=lambda n: n.score, reverse=True)

            ordered = buckets["same_party_country"] + buckets["same_family_country"] + buckets["global"]
            return ordered[:k]

    def top_chunks_for_doc(
        self,
        *,
        doc_id: str,
        query_vector: Sequence[float],
        max_items: int = 2,
        max_chars: int = 240,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            indices = list(self._chunk_indices_by_doc.get(str(doc_id), []))
            if not indices:
                return []
            qvec = _normalize_vector(query_vector)
            scored: List[tuple[float, SemanticMemoryEntry]] = []
            for idx in indices:
                entry = self._entries[idx]
                similarity = float(np.dot(qvec, self._vectors[idx]))
                scored.append((similarity, entry))
            scored.sort(key=lambda row: row[0], reverse=True)
            out: List[Dict[str, Any]] = []
            for similarity, entry in scored[: max(1, int(max_items))]:
                out.append(
                    {
                        "chunk_index": entry.chunk_index,
                        "similarity": float(similarity),
                        "text": _clip_text(str(entry.chunk_text or ""), int(max_chars)),
                    }
                )
            return out

    def query_with_snippets(
        self,
        *,
        query_vector: Sequence[float],
        query_meta: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        exclude_doc_id: Optional[str] = None,
        max_chunk_snippets_per_neighbor: Optional[int] = None,
        max_snippet_chars: Optional[int] = None,
    ) -> List[SemanticNeighbor]:
        neighbors = self.query(
            query_vector=query_vector,
            query_meta=query_meta,
            top_k=top_k,
            exclude_doc_id=exclude_doc_id,
        )
        if not neighbors:
            return []
        query_norm = _normalize_vector(query_vector)
        snippet_k = int(
            max_chunk_snippets_per_neighbor
            if max_chunk_snippets_per_neighbor is not None
            else self.config.max_chunk_snippets_per_neighbor
        )
        snippet_chars = int(max_snippet_chars if max_snippet_chars is not None else self.config.max_snippet_chars)
        out: List[SemanticNeighbor] = []
        for neighbor in neighbors:
            snippets = self.top_chunks_for_doc(
                doc_id=neighbor.doc_id,
                query_vector=query_norm,
                max_items=snippet_k,
                max_chars=snippet_chars,
            )
            out.append(
                SemanticNeighbor(
                    entry_id=neighbor.entry_id,
                    doc_id=neighbor.doc_id,
                    similarity=neighbor.similarity,
                    score=neighbor.score,
                    recency_weight=neighbor.recency_weight,
                    scope=neighbor.scope,
                    year_gap=neighbor.year_gap,
                    party_id=neighbor.party_id,
                    country_code=neighbor.country_code,
                    party_family=neighbor.party_family,
                    year=neighbor.year,
                    date_code=neighbor.date_code,
                    rile=neighbor.rile,
                    delta_rile=neighbor.delta_rile,
                    snippets=snippets,
                )
            )
        return out

    @staticmethod
    def retrieval_features(neighbors: Sequence[SemanticNeighbor]) -> np.ndarray:
        sims = [float(n.similarity) for n in neighbors]
        rile = [float(n.rile) for n in neighbors if n.rile is not None]
        delta = [float(n.delta_rile) for n in neighbors if n.delta_rile is not None]
        rec = [float(n.recency_weight) for n in neighbors]

        feat = np.zeros((6,), dtype=np.float32)
        if rile:
            feat[0] = float(np.mean(rile))
            feat[1] = float(np.std(rile))
        if delta:
            feat[2] = float(np.mean(delta))
        if sims:
            feat[3] = float(np.mean(sims))
            feat[4] = float(np.max(sims))
        if rec:
            feat[5] = float(np.mean(rec))
        return feat


def temporal_delta_targets(
    *,
    rows: Sequence[Dict[str, Any]],
    index: SemanticMemoryIndex,
    rile_key: str = "true_rile",
) -> List[Optional[float]]:
    """Compute normalized delta-RILE targets from temporal predecessors."""
    out: List[Optional[float]] = []
    for row in rows:
        predecessor = index.get_temporal_predecessor(
            party_id=row.get("party_id"),
            country_code=row.get("country_code"),
            year=row.get("year"),
            date_code=row.get("date_code"),
            exclude_doc_id=row.get("manifesto_id"),
        )
        if predecessor is None:
            out.append(None)
            continue
        delta = normalize_rile_delta(
            current_rile=_safe_float(row.get(rile_key)),
            previous_rile=predecessor.rile,
            rile_range=200.0,
        )
        out.append(delta)
    return out


__all__ = [
    "SemanticMemoryConfig",
    "SemanticMemoryEntry",
    "SemanticNeighbor",
    "SemanticMemoryIndex",
    "normalize_rile_delta",
    "temporal_delta_targets",
]
