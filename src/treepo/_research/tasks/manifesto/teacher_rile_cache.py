"""Deprecated score-only cache of teacher RILE scores keyed by node text.

This module is compatibility glue for older offline C-TreePO runs. The
canonical distillation artifact is now the score-plus-summary ``LabeledTree``
JSONL produced by ``src.ctreepo.distillation`` and teacher-trace generation:
it can recover the same scalar score examples while also carrying the
teacher summaries needed for ``g`` and ``f`` student distillation.

The legacy cache still works when a score-only
``Callable[[str], float]`` is required. A large LLM teacher (e.g.
DSPy-wrapped ``RILEScore``) is run once per rendered span in a pre-built
training corpus and scalar scores are dumped to JSONL. During compatibility
training the same tree topology replays, and
:func:`create_cached_rile_oracle` returns a ``Callable[[str], float]`` that
drops into ``CTreePOTrainer(node_oracle_predictor=...)`` without any other
code change.

The cache is append-only JSONL; each record is::

    {"key": "<sha256>", "score": <float>, "text_preview": "<first 128 chars>"}

Lookups are by SHA-256 of the raw rendered text (byte-identical match
with what ``_predict_node_oracle_score`` passes in).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

_PREVIEW_CHARS = 128

TEACHER_RILE_CACHE_DEPRECATION = (
    "TeacherRILECache is deprecated compatibility glue. Use C-TreePO "
    "LabeledTree artifacts for score-plus-summary distillation; the legacy "
    "score cache is only a score-only example that can be derived from those "
    "tree artifacts."
)


def _warn_deprecated() -> None:
    warnings.warn(
        TEACHER_RILE_CACHE_DEPRECATION,
        DeprecationWarning,
        stacklevel=3,
    )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    key: str
    score: float
    text_preview: str


class TeacherRILECache:
    """Deprecated append-only JSONL cache of rendered-text -> RILE score.

    Loading is eager: the full file is read into an in-memory dict on
    construction. Writes are serialized by an internal lock so concurrent
    builders (e.g. multiple tree workers) can share one file.
    """

    def __init__(self, path: Path | str):
        _warn_deprecated()
        self.path = Path(path)
        self.deprecated = True
        self._by_key: Dict[str, float] = {}
        self._write_lock = threading.Lock()
        self._loaded = False

    def load(self) -> "TeacherRILECache":
        if self._loaded:
            return self
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                for line_num, raw in enumerate(fh, start=1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "Skipping malformed line %d in %s: %s",
                            line_num, self.path, exc,
                        )
                        continue
                    key = record.get("key")
                    score = record.get("score")
                    if not isinstance(key, str) or not isinstance(score, (int, float)):
                        continue
                    self._by_key[str(key)] = float(score)
        self._loaded = True
        logger.info("Loaded %d cached RILE scores from %s", len(self._by_key), self.path)
        return self

    def __len__(self) -> int:
        return len(self._by_key)

    def __contains__(self, text: str) -> bool:
        return _hash_text(str(text)) in self._by_key

    def get(self, text: str) -> Optional[float]:
        return self._by_key.get(_hash_text(str(text)))

    def put(self, text: str, score: float) -> None:
        """Add a record to the in-memory dict and append it to disk."""
        text_s = str(text)
        key = _hash_text(text_s)
        score_f = float(score)
        with self._write_lock:
            if key in self._by_key and abs(self._by_key[key] - score_f) < 1e-9:
                return
            self._by_key[key] = score_f
            self.path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "key": key,
                "score": score_f,
                "text_preview": text_s[:_PREVIEW_CHARS],
            }
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def bulk_put(self, items: Iterable[tuple[str, float]]) -> int:
        n = 0
        for text, score in items:
            self.put(text, score)
            n += 1
        return n


def create_cached_rile_oracle(
    cache_path: Path | str,
    *,
    strict: bool = True,
    fallback: Optional[Callable[[str], float]] = None,
    write_back: bool = False,
) -> Callable[[str], float]:
    """Return a legacy ``Callable[[str], float]`` backed by a cache.

    Parameters
    ----------
    cache_path
        Path to the JSONL cache. Must already exist (populated by
        ``scripts/build_rile_cache.py``) unless ``fallback`` is provided.
    strict
        If ``True`` and a rendered span is missing from the cache, raise
        ``KeyError``. If ``False``, return ``0.0`` on miss so training can
        proceed with reduced supervision (not recommended: it silently
        discards C1/C2 labels for unknown spans).
    fallback
        Optional live callable (e.g. DSPy ``RILEScore``) invoked on cache
        miss. When set, overrides ``strict``.
    write_back
        If ``True`` and ``fallback`` is used, append the fallback result
        into the cache for the next run.
    """
    cache = TeacherRILECache(cache_path).load()

    def _oracle(text: str) -> float:
        score = cache.get(text)
        if score is not None:
            return float(score)
        if fallback is not None:
            value = float(fallback(text))
            if write_back:
                cache.put(text, value)
            return value
        if strict:
            raise KeyError(
                f"Rendered span not in teacher RILE cache at {cache.path}: "
                f"preview={text[:80]!r}. Re-run the cache builder or pass "
                f"fallback= to allow live filling."
            )
        return 0.0

    _oracle.__ctreepo_oracle_kind__ = "cached_rile"  # type: ignore[attr-defined]
    _oracle.__ctreepo_cache_path__ = str(cache.path)  # type: ignore[attr-defined]
    _oracle.__ctreepo_oracle_deprecated__ = True  # type: ignore[attr-defined]
    _oracle.__ctreepo_oracle_replacement__ = "labeled_tree_artifacts"  # type: ignore[attr-defined]
    return _oracle


def create_cached_rile_oracle_from_env() -> Callable[[str], float]:
    """Env-var variant usable as a ``module:function`` CLI spec.

    Reads ``CTREEPO_RILE_CACHE`` (required) and optional
    ``CTREEPO_RILE_CACHE_STRICT`` (default ``1``).
    """
    path = os.environ.get("CTREEPO_RILE_CACHE")
    if not path:
        raise RuntimeError(
            "CTREEPO_RILE_CACHE env var must be set to the cache JSONL path."
        )
    strict = os.environ.get("CTREEPO_RILE_CACHE_STRICT", "1") not in {"0", "false", "False"}
    return create_cached_rile_oracle(path, strict=strict)


def dump_nodes_to_cache(
    trees: Iterable,  # Sequence[(nodes, rile, doc_id)]
    cache: TeacherRILECache,
) -> Dict[str, int]:
    """Walk built trees and persist every node's RILE score to the cache.

    Deprecated compatibility helper. Prefer emitting ``LabeledTree`` JSONL
    artifacts so the same run captures scalar scores and teacher summary
    targets.

    Expects trees in the shape CTreePOTrainer uses internally: each entry is
    ``(nodes, rile, doc_id)`` where ``nodes`` is a list of
    ``EmbeddingTreeNode`` already labeled via
    ``_label_tree_nodes_with_oracle_scores`` (so each node's
    ``oracle_scores["rile"]`` is populated by a live predictor).

    Returns a summary dict with leaf / merge / skipped counts.
    """
    leaf = 0
    merge = 0
    skipped = 0
    for entry in trees:
        try:
            nodes, _rile, _doc_id = entry
        except Exception:
            continue
        for node in nodes:
            score = getattr(node, "oracle_scores", {}).get("rile")
            text = str(getattr(node, "text_span", "") or "")
            if score is None or not text.strip():
                skipped += 1
                continue
            cache.put(text, float(score))
            if bool(getattr(node, "is_leaf", False)):
                leaf += 1
            else:
                merge += 1
    return {"leaf": leaf, "merge": merge, "skipped": skipped, "total": leaf + merge}


def _node_field(node: Any, name: str, default: Any = None) -> Any:
    if isinstance(node, Mapping):
        return node.get(name, default)
    return getattr(node, name, default)


def dump_labeled_trees_to_cache(
    labeled_trees: Iterable[Any],
    cache: TeacherRILECache,
) -> Dict[str, int]:
    """Derive the legacy score-only cache from ``LabeledTree`` artifacts.

    This is the preferred compatibility bridge when an old consumer still
    requires ``TeacherRILECache``: emit canonical labeled trees first, then
    project their node text and scalar scores into the old JSONL shape.
    """
    leaf = 0
    merge = 0
    skipped = 0
    for tree in labeled_trees:
        if isinstance(tree, Mapping):
            raw_nodes = tree.get("nodes", {})
        else:
            raw_nodes = getattr(tree, "nodes", {})
        nodes = raw_nodes.values() if isinstance(raw_nodes, Mapping) else raw_nodes
        for node in nodes or []:
            score = _node_field(node, "score")
            text = str(_node_field(node, "text", "") or "")
            if score is None or not text.strip():
                skipped += 1
                continue
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                skipped += 1
                continue
            cache.put(text, score_f)
            try:
                level = int(_node_field(node, "level", 0) or 0)
            except (TypeError, ValueError):
                level = 0
            if level == 0:
                leaf += 1
            else:
                merge += 1
    return {"leaf": leaf, "merge": merge, "skipped": skipped, "total": leaf + merge}
