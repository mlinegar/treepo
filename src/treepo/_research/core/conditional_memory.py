"""
Persistent ConditionalMemory (L1 in-process + L2 SQLite) for ThinkingTrees.

Implements an Engram-inspired "conditional memory" layer:
- Deterministic addressing via canonicalized hashes (keys only; LLM sees raw text).
- Namespacing + namespace_version to avoid cross-contamination across tasks/commits.
- Two tiers: L1 (LRU OrderedDict) + L2 (SQLite WAL) for cross-run reuse.

This module is intentionally backend-agnostic: it stores bytes and small JSON
payloads used by higher-level caches (oracle value extraction, embeddings, etc.).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Tuple, TypeVar

Mode = Literal["off", "read", "write", "readwrite"]


_WHITESPACE = None


def canonicalize_for_key(
    text: str,
    *,
    nfkc: bool = True,
    canonicalize_whitespace: bool = True,
    casefold: bool = False,
) -> str:
    """Mild canonicalization used for cache keys only (LLM sees original text)."""
    out = str(text or "")
    if nfkc:
        out = unicodedata.normalize("NFKC", out)
    if canonicalize_whitespace:
        global _WHITESPACE
        if _WHITESPACE is None:
            import re

            _WHITESPACE = re.compile(r"[ \t\r\n]+")
        out = _WHITESPACE.sub(" ", out)
    out = out.strip()
    if casefold:
        out = out.casefold()
    return out


def canonical_hash(
    text: str,
    *,
    normalize: bool = True,
    nfkc: bool = True,
    canonicalize_whitespace: bool = True,
    casefold: bool = False,
) -> str:
    """Stable SHA256 over (optionally) canonicalized text."""
    payload = str(text or "")
    if normalize:
        payload = canonicalize_for_key(
            payload,
            nfkc=nfkc,
            canonicalize_whitespace=canonicalize_whitespace,
            casefold=casefold,
        )
    return hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()


def hash_payload(payload: Any) -> str:
    """Stable SHA256 over a JSON payload (sorted keys, compact separators)."""

    def _normalize(obj: Any) -> Any:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, (bytes, bytearray)):
            return {"__bytes__": True, "hex": bytes(obj).hex()}
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            items = ((str(k), _normalize(v)) for k, v in obj.items())
            return {k: v for k, v in sorted(items, key=lambda kv: kv[0])}
        if isinstance(obj, (list, tuple)):
            return [_normalize(v) for v in obj]
        if isinstance(obj, set):
            normalized = [_normalize(v) for v in obj]
            try:
                return sorted(normalized)
            except TypeError:
                rendered = [json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for v in normalized]
                return [v for _, v in sorted(zip(rendered, normalized), key=lambda kv: kv[0])]
        return str(obj)

    stable = json.dumps(
        _normalize(payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(stable.encode("utf-8", "surrogatepass")).hexdigest()


@dataclass(frozen=True)
class ConditionalMemoryConfig:
    enabled: bool = False
    root_dir: Path = Path("outputs/conditional_memory")
    mode: Mode = "off"
    namespace_version: str = ""
    l1_capacity: int = 4096
    l2_path: Path = Path("conditional_memory.db")
    l2_shards: int = 1
    max_l2_entries: Optional[int] = None
    sqlite_timeout_seconds: float = 30.0

    # To avoid write-amplifying high-parallel read paths, L2 "touch" updates are
    # buffered and flushed in batches (best-effort) by writers.
    touch_flush_interval_seconds: float = 10.0
    touch_flush_max_entries: int = 5000

    canonicalize_whitespace: bool = True
    nfkc: bool = True
    casefold: bool = False

    def resolved_l2_path(self) -> Path:
        l2 = Path(self.l2_path)
        if not l2.is_absolute():
            l2 = Path(self.root_dir) / l2
        return l2

    def resolved_l2_paths(self) -> list[Path]:
        base = self.resolved_l2_path()
        try:
            shard_count = max(1, int(self.l2_shards))
        except (TypeError, ValueError):
            shard_count = 1
        if shard_count <= 1:
            return [base]

        stem = base.stem
        suffix = base.suffix
        parent = base.parent
        return [parent / f"{stem}.shard{i:02d}{suffix}" for i in range(shard_count)]


@dataclass(frozen=True)
class ConditionalMemoryEntry:
    namespace: str
    key: str
    value_type: str
    value: bytes
    meta: Dict[str, Any]
    created_at: float
    last_accessed: float
    access_count: int

    @property
    def size_bytes(self) -> int:
        return int(len(self.value))


@dataclass(frozen=True)
class MemoryRecord:
    """Compatibility record for high-level lookup/store APIs."""

    key: str
    value: Any = None
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0

    @property
    def canonical_hash(self) -> str:
        return self.key


@dataclass
class ConditionalMemoryStats:
    l1_hits: int = 0
    l2_hits: int = 0
    misses: int = 0
    writes: int = 0
    bytes_written: int = 0
    l1_evictions: int = 0
    l2_evictions: int = 0

    @property
    def lookups(self) -> int:
        return int(self.l1_hits + self.l2_hits + self.misses)

    @property
    def hit_rate(self) -> float:
        total = self.lookups
        return (self.l1_hits + self.l2_hits) / total if total > 0 else 0.0

    def report(self) -> Dict[str, Any]:
        return {
            "l1_hits": int(self.l1_hits),
            "l2_hits": int(self.l2_hits),
            "misses": int(self.misses),
            "writes": int(self.writes),
            "bytes_written": int(self.bytes_written),
            "l1_evictions": int(self.l1_evictions),
            "l2_evictions": int(self.l2_evictions),
            "hit_rate": float(self.hit_rate),
        }


T = TypeVar("T")


class ConditionalMemory:
    """Two-tier persistent key/value store."""

    def __init__(
        self,
        config: Optional[ConditionalMemoryConfig] = None,
        *,
        enabled: bool = True,
        mode: Optional[Mode] = None,
        l1_capacity: int = 4096,
        l2_path: Optional[Path] = None,
        l2_shards: int = 1,
        sqlite_path: Optional[Path] = None,
        namespace_version: str = "",
        gate_threshold: float = 0.0,
        promotion_threshold: int = 3,
    ):
        if config is None:
            resolved_mode = str(mode or ("readwrite" if enabled else "off")).strip().lower()
            if resolved_mode not in {"off", "read", "write", "readwrite"}:
                resolved_mode = "off"
            try:
                resolved_l2_shards = max(1, int(l2_shards))
            except (TypeError, ValueError):
                resolved_l2_shards = 1
            config = ConditionalMemoryConfig(
                enabled=bool(enabled),
                mode=resolved_mode,  # type: ignore[arg-type]
                l1_capacity=max(1, int(l1_capacity)),
                l2_path=Path(sqlite_path or l2_path or "conditional_memory.db"),
                l2_shards=resolved_l2_shards,
                namespace_version=str(namespace_version or ""),
            )

        self.config = config
        self.gate_threshold = float(gate_threshold)
        self.promotion_threshold = max(1, int(promotion_threshold))
        enabled_flag = bool(config.enabled) and str(config.mode).strip().lower() != "off"
        self._mode: Mode = str(config.mode).strip().lower() if enabled_flag else "off"  # type: ignore[assignment]
        if self._mode not in {"off", "read", "write", "readwrite"}:
            self._mode = "off"

        self.stats = ConditionalMemoryStats()
        self._lock = threading.RLock()
        self._l1: "OrderedDict[Tuple[str, str], ConditionalMemoryEntry]" = OrderedDict()

        self._db_paths: list[Path] = []
        self._db_available: list[bool] = []
        self._write_locks: list[threading.Lock] = []
        self._write_conns: list[Optional[sqlite3.Connection]] = []
        self._read_locals: list[threading.local] = []
        self._read_conns: list[list[sqlite3.Connection]] = []
        self._pending_touches: list[Dict[Tuple[str, str], Tuple[int, float]]] = []
        self._last_touch_flush: list[float] = []
        self._shard_count: int = 1

        if self._mode != "off":
            self._open_db()

    @property
    def enabled(self) -> bool:
        return self._mode != "off"

    @property
    def can_read(self) -> bool:
        return self._mode in {"read", "readwrite"}

    @property
    def can_write(self) -> bool:
        return self._mode in {"write", "readwrite"}

    @property
    def namespace_version(self) -> str:
        return str(self.config.namespace_version or "")

    def close(self) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            return

        # Best-effort flush of any buffered L2 touches before shutdown.
        try:
            self._flush_pending_touches(force=True)
        except Exception:
            pass

        with lock:
            read_conns_by_shard = [list(bucket) for bucket in self._read_conns]
            self._read_conns = [[] for _ in range(self._shard_count)]

        for bucket in read_conns_by_shard:
            for conn in bucket:
                try:
                    conn.close()
                except Exception:
                    pass

        for shard_idx in range(self._shard_count):
            write_lock = self._write_locks[shard_idx] if shard_idx < len(self._write_locks) else None
            if write_lock is None:
                continue
            with write_lock:
                if shard_idx < len(self._write_conns) and self._write_conns[shard_idx] is not None:
                    try:
                        self._write_conns[shard_idx].close()
                    except Exception:
                        pass
                    self._write_conns[shard_idx] = None
                if shard_idx < len(self._db_available):
                    self._db_available[shard_idx] = False

    def report(self) -> Dict[str, Any]:
        """Compact stats snapshot for logging/JSON artifacts."""
        available_paths = [
            str(path)
            for shard_idx, path in enumerate(self._db_paths)
            if shard_idx < len(self._db_available) and self._db_available[shard_idx]
        ]
        return {
            **self.stats.report(),
            "mode": str(self._mode),
            "l1_entries": int(self.l1_size),
            "l2_entries": int(self.l2_size),
            "namespace_version": self.namespace_version,
            "l2_shards": int(self._shard_count),
            "l2_path": available_paths[0] if len(available_paths) == 1 else None,
            "l2_paths": available_paths if len(available_paths) > 1 else None,
        }

    @property
    def l1_size(self) -> int:
        with self._lock:
            return int(len(self._l1))

    @property
    def l2_size(self) -> int:
        total = 0
        for shard_idx in range(self._shard_count):
            conn = self._get_read_conn(shard_idx)
            if conn is None:
                continue
            try:
                row = conn.execute("SELECT COUNT(*) FROM entries").fetchone()
                total += int(row[0]) if row is not None else 0
            except Exception:
                continue
        return int(total)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _open_db(self) -> None:
        paths = self.config.resolved_l2_paths()
        self._db_paths = list(paths)
        self._shard_count = len(paths)
        self._db_available = [False for _ in range(self._shard_count)]
        self._write_locks = [threading.Lock() for _ in range(self._shard_count)]
        self._write_conns = [None for _ in range(self._shard_count)]
        self._read_locals = [threading.local() for _ in range(self._shard_count)]
        self._read_conns = [[] for _ in range(self._shard_count)]
        self._pending_touches = [{} for _ in range(self._shard_count)]
        now = time.time()
        self._last_touch_flush = [now for _ in range(self._shard_count)]

        for shard_idx, path in enumerate(paths):
            # Read-only mode: never create shard files as a side-effect.
            if not self.can_write:
                self._db_available[shard_idx] = bool(path.exists())
                continue

            # Write-capable: ensure directory exists and initialize schema once.
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                conn = self._connect(path)
                try:
                    conn.execute("PRAGMA journal_mode=WAL;")
                except Exception:
                    pass
                self._init_schema(conn)
            except Exception:
                # Fail open: keep memory enabled for L1, but disable this shard's L2.
                self._write_conns[shard_idx] = None
                self._db_available[shard_idx] = False
                continue

            self._write_conns[shard_idx] = conn
            self._db_available[shard_idx] = True

    def _sqlite_timeout_seconds(self) -> float:
        try:
            value = float(self.config.sqlite_timeout_seconds)
        except (TypeError, ValueError):
            return 30.0
        return max(0.0, value)

    def _connect(self, path: Path) -> sqlite3.Connection:
        timeout = self._sqlite_timeout_seconds()
        conn = sqlite3.connect(
            str(path),
            check_same_thread=False,
            timeout=timeout,
        )
        # Best-effort pragmas for high-parallel read/write workloads.
        try:
            conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)};")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA temp_store=MEMORY;")
        except Exception:
            pass
        return conn

    @staticmethod
    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value_type TEXT NOT NULL,
                value BLOB NOT NULL,
                meta_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER NOT NULL,
                PRIMARY KEY (namespace, key)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_access ON entries(namespace, access_count DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_last_accessed ON entries(namespace, last_accessed DESC)"
        )
        conn.commit()

    def _stable_hash_int(self, value: str) -> int:
        digest = hashlib.sha256(str(value).encode("utf-8", "surrogatepass")).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def _shard_index_for(self, namespace: str, key: str) -> int:
        if self._shard_count <= 1:
            return 0
        shard_key = f"{namespace}\x00{key}"
        return int(self._stable_hash_int(shard_key) % self._shard_count)

    def _get_read_conn(self, shard_idx: int) -> Optional[sqlite3.Connection]:
        if not self.enabled or not self.can_read:
            return None
        if shard_idx < 0 or shard_idx >= self._shard_count:
            return None
        if shard_idx >= len(self._db_available) or not self._db_available[shard_idx]:
            return None
        if shard_idx >= len(self._db_paths):
            return None

        read_local = self._read_locals[shard_idx]
        conn = getattr(read_local, "conn", None)
        if conn is not None:
            return conn

        # Lazy per-thread read connection. Use check_same_thread=False so we can
        # always close from the main thread during teardown.
        try:
            conn = self._connect(self._db_paths[shard_idx])
            if not self.can_write:
                try:
                    conn.execute("PRAGMA query_only=ON;")
                except Exception:
                    pass
        except Exception:
            return None

        read_local.conn = conn
        with self._lock:
            self._read_conns[shard_idx].append(conn)
        return conn

    def _l1_get(self, namespace: str, key: str) -> Optional[ConditionalMemoryEntry]:
        entry_key = (str(namespace), str(key))
        entry = self._l1.get(entry_key)
        if entry is None:
            return None
        self._l1.move_to_end(entry_key)
        return entry

    def _l1_put(self, entry: ConditionalMemoryEntry) -> None:
        entry_key = (entry.namespace, entry.key)
        self._l1[entry_key] = entry
        self._l1.move_to_end(entry_key)
        while len(self._l1) > max(1, int(self.config.l1_capacity)):
            self._l1.popitem(last=False)
            self.stats.l1_evictions += 1

    def _get_entry(self, namespace: str, key: str) -> Optional[ConditionalMemoryEntry]:
        if not self.enabled or not self.can_read:
            return None
        namespace = str(namespace)
        key = str(key)
        shard_idx = self._shard_index_for(namespace, key)

        now = time.time()

        # Fast-path: L1 lookup under lock.
        with self._lock:
            entry = self._l1_get(namespace, key)
            if entry is not None:
                self.stats.l1_hits += 1
                updated = ConditionalMemoryEntry(
                    namespace=entry.namespace,
                    key=entry.key,
                    value_type=entry.value_type,
                    value=entry.value,
                    meta=dict(entry.meta),
                    created_at=entry.created_at,
                    last_accessed=now,
                    access_count=int(entry.access_count) + 1,
                )
                self._l1_put(updated)
                return updated

        # L2 lookup outside the L1 lock so hot-path hits don't contend with disk I/O.
        conn = self._get_read_conn(shard_idx)
        if conn is None:
            with self._lock:
                self.stats.misses += 1
            return None

        try:
            row = conn.execute(
                "SELECT value_type, value, meta_json, created_at, last_accessed, access_count "
                "FROM entries WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()
        except Exception:
            with self._lock:
                self.stats.misses += 1
            return None

        if row is None:
            with self._lock:
                self.stats.misses += 1
            return None

        value_type, value, meta_json, created_at, last_accessed, access_count = row
        try:
            meta = json.loads(str(meta_json) or "{}")
            if not isinstance(meta, dict):
                meta = {}
        except Exception:
            meta = {}

        entry = ConditionalMemoryEntry(
            namespace=namespace,
            key=key,
            value_type=str(value_type),
            value=bytes(value),
            meta=meta,
            created_at=float(created_at),
            last_accessed=now,
            access_count=int(access_count) + 1,
        )

        should_flush = False
        with self._lock:
            self.stats.l2_hits += 1
            self._l1_put(entry)
            if self.can_write:
                self._record_touch_locked(shard_idx, namespace, key, accessed_at=now, delta=1)
                should_flush = self._should_flush_pending_touches_locked(shard_idx, now)

        if should_flush:
            self._flush_pending_touches(force=False, shard_idx=shard_idx)

        return entry

    @staticmethod
    def _entry_to_record(entry: ConditionalMemoryEntry) -> MemoryRecord:
        value: Any
        if entry.value_type == "utf8":
            try:
                value = entry.value.decode("utf-8", errors="strict")
            except Exception:
                value = entry.value.decode("utf-8", errors="replace")
        elif entry.value_type == "json":
            try:
                value = json.loads(entry.value.decode("utf-8", errors="strict"))
            except Exception:
                value = None
        else:
            value = bytes(entry.value)
        return MemoryRecord(
            key=str(entry.key),
            value=value,
            scores=dict(entry.meta.get("scores", {}) if isinstance(entry.meta, dict) else {}),
            metadata=dict(entry.meta if isinstance(entry.meta, dict) else {}),
            created_at=float(entry.created_at),
            last_accessed=float(entry.last_accessed),
            access_count=int(entry.access_count),
        )

    def get(
        self,
        namespace: str,
        key: Optional[str] = None,
    ) -> Optional[ConditionalMemoryEntry | MemoryRecord]:
        """Dual API:
        - ``get(namespace, key)`` -> ``ConditionalMemoryEntry`` (low-level).
        - ``get(key)`` -> ``MemoryRecord`` from default namespace (compat mode).
        """
        if key is None:
            entry = self._get_entry("default", str(namespace))
            if entry is None:
                return None
            return self._entry_to_record(entry)
        return self._get_entry(str(namespace), str(key))

    def _record_touch_locked(
        self,
        shard_idx: int,
        namespace: str,
        key: str,
        *,
        accessed_at: float,
        delta: int = 1,
    ) -> None:
        if shard_idx < 0 or shard_idx >= self._shard_count:
            return
        touches = self._pending_touches[shard_idx]
        touch_key = (str(namespace), str(key))
        prev = touches.get(touch_key)
        if prev is None:
            touches[touch_key] = (max(1, int(delta)), float(accessed_at))
            return
        prev_delta, prev_accessed_at = prev
        touches[touch_key] = (
            int(prev_delta) + max(1, int(delta)),
            max(float(prev_accessed_at), float(accessed_at)),
        )

    def _should_flush_pending_touches_locked(self, shard_idx: int, now: float) -> bool:
        if shard_idx < 0 or shard_idx >= self._shard_count:
            return False
        if (
            not self.enabled
            or not self.can_write
            or shard_idx >= len(self._write_conns)
            or self._write_conns[shard_idx] is None
        ):
            return False
        touches = self._pending_touches[shard_idx]
        if not touches:
            return False
        try:
            max_entries = max(1, int(self.config.touch_flush_max_entries))
        except (TypeError, ValueError):
            max_entries = 5000
        try:
            interval = max(0.0, float(self.config.touch_flush_interval_seconds))
        except (TypeError, ValueError):
            interval = 10.0
        if len(touches) >= max_entries:
            return True
        return (now - float(self._last_touch_flush[shard_idx])) >= interval

    def _flush_pending_touches_locked(self, shard_idx: int, conn: sqlite3.Connection, *, force: bool) -> None:
        now = time.time()
        with self._lock:
            if not force and not self._should_flush_pending_touches_locked(shard_idx, now):
                return
            touches = self._pending_touches[shard_idx]
            self._pending_touches[shard_idx] = {}
            self._last_touch_flush[shard_idx] = now

        if not touches:
            return

        rows = [
            (float(accessed_at), int(delta), str(namespace), str(key))
            for (namespace, key), (delta, accessed_at) in touches.items()
        ]
        try:
            conn.executemany(
                "UPDATE entries SET last_accessed = ?, access_count = access_count + ? "
                "WHERE namespace = ? AND key = ?",
                rows,
            )
        except Exception:
            return

    def _flush_pending_touches(self, *, force: bool, shard_idx: Optional[int] = None) -> None:
        if not self.enabled or not self.can_write:
            return
        shard_indices = [int(shard_idx)] if shard_idx is not None else list(range(self._shard_count))
        for idx in shard_indices:
            if idx < 0 or idx >= self._shard_count:
                continue
            conn = self._write_conns[idx] if idx < len(self._write_conns) else None
            lock = self._write_locks[idx] if idx < len(self._write_locks) else None
            if conn is None or lock is None:
                continue
            with lock:
                self._flush_pending_touches_locked(idx, conn, force=force)
                try:
                    conn.commit()
                except Exception:
                    pass

    def _touch_l2(self, namespace: str, key: str, access_count: int) -> None:
        # Back-compat helper for callers that expect an eager touch API.
        if not self.enabled or not self.can_write:
            return
        shard_idx = self._shard_index_for(str(namespace), str(key))
        now = time.time()
        with self._lock:
            self._record_touch_locked(shard_idx, namespace, key, accessed_at=now, delta=1)
            should_flush = self._should_flush_pending_touches_locked(shard_idx, now)
        if should_flush:
            self._flush_pending_touches(force=False, shard_idx=shard_idx)

    def set(
        self,
        namespace: str,
        key: str,
        *,
        value_type: str,
        value: bytes,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled or not self.can_write:
            return
        namespace = str(namespace)
        key = str(key)
        shard_idx = self._shard_index_for(namespace, key)
        now = time.time()

        meta_json = json.dumps(
            dict(meta or {}),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        entry = ConditionalMemoryEntry(
            namespace=namespace,
            key=key,
            value_type=str(value_type),
            value=bytes(value),
            meta=dict(meta or {}),
            created_at=now,
            last_accessed=now,
            access_count=1,
        )

        with self._lock:
            self._l1_put(entry)

        conn = self._write_conns[shard_idx] if shard_idx < len(self._write_conns) else None
        write_lock = self._write_locks[shard_idx] if shard_idx < len(self._write_locks) else None
        if conn is None or write_lock is None:
            return

        with write_lock:
            try:
                conn.execute(
                    """
                    INSERT INTO entries (namespace, key, value_type, value, meta_json, created_at, last_accessed, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(namespace, key) DO UPDATE SET
                        value_type=excluded.value_type,
                        value=excluded.value,
                        meta_json=excluded.meta_json,
                        created_at=MIN(entries.created_at, excluded.created_at),
                        last_accessed=excluded.last_accessed,
                        access_count=entries.access_count + 1
                    """,
                    (namespace, key, str(value_type), sqlite3.Binary(bytes(value)), meta_json, now, now, 1),
                )

                # Flush buffered touches opportunistically so eviction sees fresher stats.
                self._flush_pending_touches_locked(shard_idx, conn, force=False)
                self._evict_l2_if_needed_locked(shard_idx, conn)

                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                return

        with self._lock:
            self.stats.writes += 1
            self.stats.bytes_written += int(len(value))

    def delete(self, namespace: str, key: str) -> None:
        if not self.enabled or not self.can_write:
            return
        namespace = str(namespace)
        key = str(key)
        shard_idx = self._shard_index_for(namespace, key)
        with self._lock:
            self._l1.pop((namespace, key), None)
        conn = self._write_conns[shard_idx] if shard_idx < len(self._write_conns) else None
        write_lock = self._write_locks[shard_idx] if shard_idx < len(self._write_locks) else None
        if conn is None or write_lock is None:
            return
        with write_lock:
            try:
                conn.execute(
                    "DELETE FROM entries WHERE namespace = ? AND key = ?",
                    (namespace, key),
                )
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass

    def set_json(
        self,
        namespace: str,
        key: str,
        obj: Any,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.set(namespace, key, value_type="json", value=payload, meta=meta)

    def get_json(self, namespace: str, key: str) -> Optional[Any]:
        entry = self.get(namespace, key)
        if entry is None:
            return None
        if entry.value_type != "json":
            return None
        try:
            return json.loads(entry.value.decode("utf-8", errors="strict"))
        except Exception:
            return None

    def set_text(
        self,
        namespace: str,
        key: str,
        text: str,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = str(text or "").encode("utf-8", "surrogatepass")
        self.set(namespace, key, value_type="utf8", value=payload, meta=meta)

    def get_text(self, namespace: str, key: str) -> Optional[str]:
        entry = self.get(namespace, key)
        if entry is None:
            return None
        if entry.value_type != "utf8":
            return None
        try:
            return entry.value.decode("utf-8", errors="strict")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Compatibility helpers (Engram/DualPath integration paths)
    # ------------------------------------------------------------------

    def put(
        self,
        key: str,
        value: Any,
        *,
        namespace: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if isinstance(value, str):
            self.set_text(namespace, str(key), value, meta=metadata)
            return
        self.set_json(namespace, str(key), value, meta=metadata)

    def remember_text(
        self,
        text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        key = canonical_hash(text)
        meta = dict(metadata or {})
        meta.setdefault("value_chars", len(str(text or "")))
        self.set_text("remembered_text", key, str(text or ""), meta=meta)
        return key

    def should_inject(
        self,
        key: str,
        *,
        max_value_chars: int,
        required_tag: Optional[str] = None,
    ) -> bool:
        entry = self._get_entry("remembered_text", str(key))
        if entry is None:
            return False
        try:
            value = entry.value.decode("utf-8", errors="strict")
        except Exception:
            return False
        if len(value) > max(0, int(max_value_chars)):
            return False
        if required_tag:
            tags = entry.meta.get("tags", []) if isinstance(entry.meta, dict) else []
            if isinstance(tags, str):
                tags = [tags]
            if required_tag not in list(tags):
                return False
        return True

    def lookup(
        self,
        text: str,
        *,
        score_heads: Optional[Iterable[str]] = None,
        normalize: bool = True,
    ) -> Optional[MemoryRecord]:
        key = canonical_hash(text, normalize=normalize)
        entry = self._get_entry("scores", key)
        if entry is None or entry.value_type != "json":
            return None
        try:
            payload = json.loads(entry.value.decode("utf-8", errors="strict"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        raw_scores = payload.get("scores", {})
        scores = {
            str(k): float(v)
            for k, v in (raw_scores.items() if isinstance(raw_scores, dict) else [])
            if isinstance(k, str)
        }
        requested = [str(h) for h in (score_heads or [])]
        if requested and not any(head in scores for head in requested):
            return None
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        return MemoryRecord(
            key=key,
            value=payload.get("preview", ""),
            scores=scores,
            metadata=metadata,
            created_at=float(entry.created_at),
            last_accessed=float(entry.last_accessed),
            access_count=int(entry.access_count),
        )

    def store(
        self,
        text: str,
        *,
        scores: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        normalize: bool = True,
    ) -> MemoryRecord:
        key = canonical_hash(text, normalize=normalize)
        existing_payload = self.get_json("scores", key)
        existing_scores: Dict[str, float] = {}
        existing_meta: Dict[str, Any] = {}
        if isinstance(existing_payload, dict):
            maybe_scores = existing_payload.get("scores", {})
            if isinstance(maybe_scores, dict):
                for k, v in maybe_scores.items():
                    try:
                        existing_scores[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
            maybe_meta = existing_payload.get("metadata", {})
            if isinstance(maybe_meta, dict):
                existing_meta = dict(maybe_meta)

        merged_scores = dict(existing_scores)
        for k, v in (scores or {}).items():
            try:
                merged_scores[str(k)] = float(v)
            except (TypeError, ValueError):
                continue

        merged_meta = dict(existing_meta)
        merged_meta.update(dict(metadata or {}))
        payload = {
            "preview": str(text or "")[:512],
            "scores": merged_scores,
            "metadata": merged_meta,
        }
        self.set_json("scores", key, payload, meta={"scores": merged_scores, **merged_meta})
        entry = self._get_entry("scores", key)
        if entry is None:
            return MemoryRecord(key=key, value=payload["preview"], scores=merged_scores, metadata=merged_meta)
        return MemoryRecord(
            key=key,
            value=payload["preview"],
            scores=merged_scores,
            metadata=merged_meta,
            created_at=float(entry.created_at),
            last_accessed=float(entry.last_accessed),
            access_count=int(entry.access_count),
        )

    def get_or_compute_json(
        self,
        namespace: str,
        key: str,
        compute_fn: Callable[[], T],
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> T:
        cached = self.get_json(namespace, key)
        if cached is not None:
            return cached
        value = compute_fn()
        self.set_json(namespace, key, value, meta=meta)
        return value

    def warmup_top(self, namespace: str, limit: int = 256) -> int:
        if not self.enabled or not self.can_read:
            return 0
        namespace = str(namespace)
        limit = max(0, int(limit))
        if limit <= 0:
            return 0

        candidates: list[tuple[int, float, tuple[Any, ...]]] = []
        for shard_idx in range(self._shard_count):
            conn = self._get_read_conn(shard_idx)
            if conn is None:
                continue
            try:
                rows = conn.execute(
                    "SELECT key, value_type, value, meta_json, created_at, last_accessed, access_count "
                    "FROM entries WHERE namespace = ? "
                    "ORDER BY access_count DESC, last_accessed DESC LIMIT ?",
                    (namespace, limit),
                ).fetchall()
            except Exception:
                continue
            for row in rows:
                access_count = int(row[6]) if len(row) > 6 else 0
                last_accessed = float(row[5]) if len(row) > 5 else 0.0
                candidates.append((access_count, last_accessed, row))

        if not candidates:
            return 0

        candidates.sort(key=lambda item: (-item[0], -item[1]))
        selected_rows = [row for _, _, row in candidates[:limit]]

        warmed = 0
        with self._lock:
            for key, value_type, value, meta_json, created_at, last_accessed, access_count in selected_rows:
                try:
                    meta = json.loads(str(meta_json) or "{}")
                    if not isinstance(meta, dict):
                        meta = {}
                except Exception:
                    meta = {}
                entry = ConditionalMemoryEntry(
                    namespace=namespace,
                    key=str(key),
                    value_type=str(value_type),
                    value=bytes(value),
                    meta=meta,
                    created_at=float(created_at),
                    last_accessed=float(last_accessed),
                    access_count=int(access_count),
                )
                self._l1_put(entry)
                warmed += 1
        return warmed

    def _evict_l2_if_needed_locked(self, shard_idx: int, conn: sqlite3.Connection) -> None:
        max_entries = self.config.max_l2_entries
        if max_entries is None:
            return
        try:
            max_entries_int = int(max_entries)
        except (TypeError, ValueError):
            return
        if max_entries_int <= 0:
            return

        # Keep lock ordering simple by enforcing an independent bound per shard.
        per_shard_limit = max(1, (max_entries_int + self._shard_count - 1) // self._shard_count)

        # Ensure eviction decisions see recent access deltas.
        self._flush_pending_touches_locked(shard_idx, conn, force=True)

        try:
            row = conn.execute("SELECT COUNT(*) FROM entries").fetchone()
            count = int(row[0]) if row is not None else 0
        except Exception:
            return

        excess = count - per_shard_limit
        if excess <= 0:
            return

        try:
            conn.execute(
                """
                DELETE FROM entries
                WHERE rowid IN (
                    SELECT rowid FROM entries
                    ORDER BY access_count ASC, last_accessed ASC
                    LIMIT ?
                )
                """,
                (int(excess),),
            )
        except Exception:
            return

        with self._lock:
            self.stats.l2_evictions += int(excess)

    def _evict_l2_if_needed(self) -> None:
        if not self.enabled or not self.can_write:
            return
        for shard_idx in range(self._shard_count):
            conn = self._write_conns[shard_idx] if shard_idx < len(self._write_conns) else None
            write_lock = self._write_locks[shard_idx] if shard_idx < len(self._write_locks) else None
            if conn is None or write_lock is None:
                continue
            with write_lock:
                self._evict_l2_if_needed_locked(shard_idx, conn)
                try:
                    conn.commit()
                except Exception:
                    pass


_DEFAULT_MEMORY_LOCK = threading.Lock()
_DEFAULT_MEMORY: Optional[ConditionalMemory] = None


def _default_namespace_version() -> str:
    raw = str(os.getenv("TT_CONDITIONAL_MEMORY_NAMESPACE_VERSION", "") or "").strip()
    if raw:
        return raw
    task_name = str(os.getenv("TASK", "") or os.getenv("TT_TASK", "") or "").strip() or "task"
    sha = "unknown"
    try:
        import subprocess

        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8", errors="replace")
            .strip()
            or sha
        )
    except Exception:
        pass
    return f"{sha}:{task_name}"


def get_default_memory() -> Optional[ConditionalMemory]:
    """Lazy singleton ConditionalMemory configured via env vars."""
    global _DEFAULT_MEMORY
    with _DEFAULT_MEMORY_LOCK:
        if _DEFAULT_MEMORY is not None:
            return _DEFAULT_MEMORY

        mode_raw = str(os.getenv("TT_CONDITIONAL_MEMORY_MODE", "") or "").strip().lower() or "off"
        if mode_raw not in {"off", "read", "write", "readwrite"}:
            mode_raw = "off"

        root = Path(str(os.getenv("TT_CONDITIONAL_MEMORY_DIR", "") or "outputs/conditional_memory")).expanduser()
        l2_path = Path(str(os.getenv("TT_CONDITIONAL_MEMORY_L2_PATH", "") or "conditional_memory.db"))
        try:
            l2_shards_raw = str(os.getenv("TT_CONDITIONAL_MEMORY_L2_SHARDS", "") or "").strip()
            l2_shards = int(l2_shards_raw) if l2_shards_raw else 1
        except (TypeError, ValueError):
            l2_shards = 1
        try:
            l1_cap = int(os.getenv("TT_CONDITIONAL_MEMORY_L1_CAP", "4096") or 4096)
        except (TypeError, ValueError):
            l1_cap = 4096
        try:
            max_l2_entries = os.getenv("TT_CONDITIONAL_MEMORY_MAX_L2_ENTRIES")
            max_l2_entries_int = int(max_l2_entries) if max_l2_entries is not None and str(max_l2_entries).strip() else None
        except (TypeError, ValueError):
            max_l2_entries_int = None
        try:
            sqlite_timeout_raw = str(os.getenv("TT_CONDITIONAL_MEMORY_SQLITE_TIMEOUT_SECONDS", "") or "").strip()
            sqlite_timeout_seconds = float(sqlite_timeout_raw) if sqlite_timeout_raw else 30.0
        except (TypeError, ValueError):
            sqlite_timeout_seconds = 30.0
        try:
            touch_interval_raw = str(os.getenv("TT_CONDITIONAL_MEMORY_TOUCH_FLUSH_INTERVAL_SECONDS", "") or "").strip()
            touch_flush_interval_seconds = float(touch_interval_raw) if touch_interval_raw else 10.0
        except (TypeError, ValueError):
            touch_flush_interval_seconds = 10.0
        try:
            touch_max_raw = str(os.getenv("TT_CONDITIONAL_MEMORY_TOUCH_FLUSH_MAX_ENTRIES", "") or "").strip()
            touch_flush_max_entries = int(touch_max_raw) if touch_max_raw else 5000
        except (TypeError, ValueError):
            touch_flush_max_entries = 5000

        cfg = ConditionalMemoryConfig(
            enabled=mode_raw != "off",
            root_dir=root,
            mode=mode_raw,  # type: ignore[arg-type]
            namespace_version=_default_namespace_version(),
            l1_capacity=max(1, int(l1_cap)),
            l2_path=l2_path,
            l2_shards=max(1, int(l2_shards)),
            max_l2_entries=max_l2_entries_int,
            sqlite_timeout_seconds=float(sqlite_timeout_seconds),
            touch_flush_interval_seconds=float(touch_flush_interval_seconds),
            touch_flush_max_entries=int(touch_flush_max_entries),
        )
        if cfg.mode == "off" or not cfg.enabled:
            _DEFAULT_MEMORY = None
            return None

        _DEFAULT_MEMORY = ConditionalMemory(cfg)
        return _DEFAULT_MEMORY
