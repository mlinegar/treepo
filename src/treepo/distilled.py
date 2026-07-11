"""Cached-teacher node labels: read ``teacher_node_rows.jsonl`` into trees.

The distillation lane of the fit-grid plan (Phase 2) trains a student purely
from labels an LLM teacher wrote to disk in an earlier pass — fit-time work is
pure cache consumption, no teacher client is ever constructed. Producers write
one flat JSONL row per tree node; two row schemas exist in the wild:

* **teacher-grid rows** carry ``score_1_7`` plus a ``dimension`` name
  (``outputs/manifesto_teacher_fg_leaf_grid/*/leaf_*/teacher_node_rows.jsonl``).
* **rile-grid rows** carry ``rile_norm`` and a ``dimension_scores_0_1`` map
  (``outputs/mpds_rile_llmseg_grid/*/teacher_node_rows.jsonl``).

Rows are matched to :class:`treepo.tree.TreeRecord` nodes by
``(doc_id, node_id)`` first and by the span key
``(doc_id, level, char_start, char_end)`` as a fallback — the same stable keys
the producers use. Matched scores land on node ``metadata`` under a dedicated
key (default ``"distilled_score"``) that the per-node supervision path only
reads when explicitly configured, so attached labels never leak into cells
that did not ask for them.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "DISTILLED_NODE_KEY",
    "DistilledNodeLabels",
    "attach_distilled_labels",
    "attach_predictor_labels",
    "load_teacher_node_rows",
]

#: Node-metadata key the distillation lane writes and the supervision path
#: reads. Deliberately NOT ``score`` / ``oracle_score`` (generic fallbacks in
#: node-target extraction): the distilled value must only be consumed when
#: ``node_target_key`` names it explicitly.
DISTILLED_NODE_KEY = "distilled_score"

_SpanKey = tuple[str, int, int, int]
_NodeKey = tuple[str, str]


@dataclass(frozen=True)
class DistilledNodeLabels:
    """Pinned view of one cached-teacher ``teacher_node_rows.jsonl`` file."""

    source_path: str
    score_key: str
    dimension: str | None
    n_rows: int
    n_skipped: int
    by_node: Mapping[_NodeKey, float]
    by_span: Mapping[_SpanKey, float]

    @property
    def n_distinct_scores(self) -> int:
        return len(set(self.by_node.values()) | set(self.by_span.values()))

    @property
    def degenerate_scores(self) -> bool:
        """Every row carries the same score — a placeholder-label cache.

        The known instance is ``outputs/mpds_rile_llmseg_grid/leafq001`` where
        all 373K ``rile_norm`` values are literally 0.5; training a distilled
        cell on a constant teacher fits a constant and must fail loudly.
        """

        return self.n_rows > 1 and self.n_distinct_scores <= 1

    def lookup(
        self,
        *,
        doc_id: str,
        node_id: str | None = None,
        level: int | None = None,
        char_start: int | None = None,
        char_end: int | None = None,
    ) -> float | None:
        if node_id is not None:
            value = self.by_node.get((str(doc_id), str(node_id)))
            if value is not None:
                return value
        if level is not None and char_start is not None and char_end is not None:
            return self.by_span.get((str(doc_id), int(level), int(char_start), int(char_end)))
        return None


def load_teacher_node_rows(
    path: str | Path,
    *,
    dimension: str | None = None,
) -> DistilledNodeLabels:
    """Load one cached-teacher JSONL file into lookup tables.

    ``dimension`` selects the label channel: for teacher-grid rows it filters
    on the row's ``dimension`` field, for rile-grid rows it reads
    ``dimension_scores_0_1[dimension]``. Without it, single-dimension
    teacher-grid files load as-is (multi-dimension files are ambiguous and
    error) and rile-grid rows fall back to ``rile_norm``.
    """

    file_path = _resolve_rows_file(Path(path))
    by_node: dict[_NodeKey, float] = {}
    by_span: dict[_SpanKey, float] = {}
    score_keys: set[str] = set()
    dimensions_seen: set[str] = set()
    n_rows = 0
    n_skipped = 0

    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_no}: not valid JSON: {exc}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"{file_path}:{line_no}: expected a JSON object per line")
            n_rows += 1
            row_dimension = row.get("dimension")
            if row_dimension is not None:
                dimensions_seen.add(str(row_dimension))
            score, score_key = _row_score(row, dimension=dimension)
            if score is None:
                n_skipped += 1
                continue
            score_keys.add(score_key)

            doc_id = row.get("doc_id")
            if doc_id is None:
                raise ValueError(f"{file_path}:{line_no}: row has no doc_id")
            doc_id = str(doc_id)

            node_id = row.get("node_id")
            if node_id is not None:
                _put_strict(
                    by_node, (doc_id, str(node_id)), score, where=f"{file_path}:{line_no}"
                )
            level = row.get("level")
            char_start = row.get("char_start")
            char_end = row.get("char_end")
            if level is not None and char_start is not None and char_end is not None:
                _put_strict(
                    by_span,
                    (doc_id, int(level), int(char_start), int(char_end)),
                    score,
                    where=f"{file_path}:{line_no}",
                )

    if dimension is None and len(dimensions_seen) > 1:
        raise ValueError(
            f"{file_path} carries rows for multiple dimensions "
            f"{sorted(dimensions_seen)}; pass dimension= to select one"
        )
    if not by_node and not by_span:
        raise ValueError(
            f"{file_path}: no usable teacher rows"
            + (f" for dimension {dimension!r}" if dimension else "")
            + f" ({n_rows} rows read, {n_skipped} skipped)"
        )
    return DistilledNodeLabels(
        source_path=str(file_path),
        score_key="|".join(sorted(score_keys)),
        dimension=dimension,
        n_rows=n_rows,
        n_skipped=n_skipped,
        by_node=by_node,
        by_span=by_span,
    )


def attach_distilled_labels(
    records: Sequence[Any],
    labels: DistilledNodeLabels,
    *,
    node_key: str = DISTILLED_NODE_KEY,
) -> dict[str, Any]:
    """Attach cached-teacher scores to matching nodes; return an attach report.

    Node ``metadata`` dicts are updated in place (mirroring the TT stage-1
    ``attach_labeled_tree_scores`` semantics). Unmatched nodes get nothing —
    with ``node_target_exclusive`` set they stay *unobserved* rather than
    falling back to gold labels.
    """

    n_nodes = 0
    n_attached = 0
    n_by_node_id = 0
    n_by_span = 0
    unmatched_docs: set[str] = set()
    matched_keys: set[_NodeKey] = set()
    for record in records or ():
        doc_id = _record_doc_id(record)
        for node in _record_nodes(record):
            n_nodes += 1
            node_id = _node_attr(node, "node_id")
            meta = _node_metadata(node)
            value = None
            if node_id is not None:
                value = labels.by_node.get((doc_id, str(node_id)))
                if value is not None:
                    n_by_node_id += 1
                    matched_keys.add((doc_id, str(node_id)))
            if value is None:
                span = _node_span_key(node, meta, doc_id)
                if span is not None:
                    value = labels.by_span.get(span)
                    if value is not None:
                        n_by_span += 1
            if value is None:
                unmatched_docs.add(doc_id)
                continue
            _write_node_metadata(node, meta, node_key, float(value))
            n_attached += 1
    if n_attached == 0:
        raise ValueError(
            f"no node of {len(list(records or ()))} trees matched a teacher row from "
            f"{labels.source_path} (tried doc_id::node_id and doc_id/level/char-span "
            "keys); check that the cache and the bundle come from the same grid"
        )
    return {
        "labels_source": "cached_jsonl",
        "distilled_labels_path": labels.source_path,
        "distilled_node_key": str(node_key),
        "score_key": labels.score_key,
        "dimension": labels.dimension,
        "n_teacher_rows": labels.n_rows,
        "n_teacher_rows_skipped": labels.n_skipped,
        "n_nodes": n_nodes,
        "n_nodes_attached": n_attached,
        "n_nodes_unmatched": n_nodes - n_attached,
        "n_matched_by_node_id": n_by_node_id,
        "n_matched_by_span": n_by_span,
        "n_docs_with_unmatched_nodes": len(unmatched_docs),
    }


def attach_predictor_labels(
    records: Sequence[Any],
    predictor: Callable[..., Any],
    *,
    node_key: str = DISTILLED_NODE_KEY,
) -> dict[str, Any]:
    """Attach scores from a callable node predictor (text -> score).

    The TT contract: the predictor receives the node's rendered text and
    returns a scalar (or a mapping with a ``"score"`` entry). Nodes without
    text, and nodes the predictor declines (``None`` / non-finite), stay
    unlabeled.
    """

    n_nodes = 0
    n_attached = 0
    for record in records or ():
        for node in _record_nodes(record):
            n_nodes += 1
            text = _node_attr(node, "text")
            if not text:
                continue
            raw = predictor(str(text))
            if isinstance(raw, Mapping):
                raw = raw.get("score")
            if raw is None:
                continue
            value = float(raw)
            if not math.isfinite(value):
                continue
            meta = _node_metadata(node)
            _write_node_metadata(node, meta, node_key, value)
            n_attached += 1
    if n_attached == 0:
        raise ValueError(
            "node_oracle_predictor produced no usable node score across "
            f"{n_nodes} nodes; it must return a finite scalar (or a mapping "
            "with a 'score') for at least one node text"
        )
    return {
        "labels_source": "node_oracle_predictor",
        "predictor": getattr(predictor, "__name__", type(predictor).__name__),
        "distilled_node_key": str(node_key),
        "n_nodes": n_nodes,
        "n_nodes_attached": n_attached,
        "n_nodes_unmatched": n_nodes - n_attached,
    }


def _row_score(row: Mapping[str, Any], *, dimension: str | None) -> tuple[float | None, str]:
    """Resolve one row's label channel; ``(None, key)`` means row is skipped."""

    if "score_1_7" in row:
        row_dimension = row.get("dimension")
        if dimension is not None and row_dimension is not None and str(row_dimension) != dimension:
            return None, "score_1_7"
        return _finite_or_none(row.get("score_1_7")), "score_1_7"
    scores = row.get("dimension_scores_0_1")
    if dimension is not None:
        if isinstance(scores, Mapping) and dimension in scores:
            return _finite_or_none(scores.get(dimension)), f"dimension_scores_0_1[{dimension}]"
        return None, f"dimension_scores_0_1[{dimension}]"
    if "rile_norm" in row:
        return _finite_or_none(row.get("rile_norm")), "rile_norm"
    return None, "unknown"


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _put_strict(
    table: dict[Any, float], key: Any, value: float, *, where: str
) -> None:
    existing = table.get(key)
    if existing is not None and abs(existing - value) > 1e-12:
        raise ValueError(
            f"{where}: conflicting teacher scores for {key!r}: "
            f"{existing!r} vs {value!r} (is the file a concatenation of runs?)"
        )
    table[key] = value


def _resolve_rows_file(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "teacher_node_rows.jsonl"
        if not candidate.is_file():
            raise FileNotFoundError(
                f"{path} is a directory without a teacher_node_rows.jsonl"
            )
        return candidate
    if not path.is_file():
        raise FileNotFoundError(f"no such teacher rows file: {path}")
    return path


def _record_doc_id(record: Any) -> str:
    for attr in ("doc_id", "tree_id", "id"):
        value = getattr(record, attr, None)
        if value is not None:
            return str(value)
    if isinstance(record, Mapping):
        for key in ("doc_id", "tree_id", "id"):
            if record.get(key) is not None:
                return str(record[key])
    return ""


def _record_nodes(record: Any) -> Sequence[Any]:
    nodes = getattr(record, "nodes", None)
    if nodes is None and isinstance(record, Mapping):
        nodes = record.get("nodes")
    if isinstance(nodes, Mapping):
        return list(nodes.values())
    return list(nodes or ())


def _node_attr(node: Any, key: str) -> Any:
    value = getattr(node, key, None)
    if value is None and isinstance(node, Mapping):
        value = node.get(key)
    return value


def _node_metadata(node: Any) -> dict[str, Any] | None:
    meta = getattr(node, "metadata", None)
    if meta is None and isinstance(node, Mapping):
        meta = node.get("metadata")
    return meta if isinstance(meta, dict) else None


def _node_span_key(
    node: Any, meta: Mapping[str, Any] | None, doc_id: str
) -> _SpanKey | None:
    level = _node_attr(node, "level")
    source: Mapping[str, Any] = meta if meta is not None else {}
    char_start = source.get("char_start", _node_attr(node, "char_start"))
    char_end = source.get("char_end", _node_attr(node, "char_end"))
    if level is None or char_start is None or char_end is None:
        return None
    return (doc_id, int(level), int(char_start), int(char_end))


def _write_node_metadata(
    node: Any, meta: dict[str, Any] | None, key: str, value: float
) -> None:
    if meta is not None:
        meta[str(key)] = value
        return
    if isinstance(node, dict):
        inner = node.setdefault("metadata", {})
        if isinstance(inner, dict):
            inner[str(key)] = value
            return
    raise TypeError(
        f"cannot attach distilled label: node {node!r} has no mutable metadata dict"
    )
