"""Task-agnostic loader for external ``LabeledTree`` bundles.

Downstream producers (the Manifesto/RILE sentence and qsentence grids, and
later Markov or Benoit-econ tasks) publish labeled trees to a shared on-disk
contract: a ``labeled_trees.jsonl`` file where every line is one document tree
with per-node payload (spans, ``cmp_counts``, ``dimension_scores``, sentence and
qsentence index ranges) plus a bundle-level ``split_ids.json`` that pins the
train/val/test partition. This module reads that contract into the package-owned
:class:`treepo.tree.TreeRecord` shape so real grids run through ``fit()`` without
porting any task into treepo.

The loader is deliberately thin and lossless: the root target becomes the
tree-level ``root_label`` that ``fit()`` consumes, each node's score (or the
selected dimension's score) becomes the node ``label`` that per-node
supervision consumes (``supervision_level`` on the spec; see
``treepo.methods._supervision``), and *every* remaining per-node field is
preserved on the node ``metadata`` (an ``extras`` payload). It never resamples
splits — when a bundle pins ``split_ids`` those ids are authoritative.

The full field-by-field contract lives in ``docs/bundle_contract.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from treepo.tree import TreeNode, TreeRecord

__all__ = [
    "BundleFormatError",
    "KNOWN_TREE_VERSIONS",
    "load_labeled_tree_bundle",
]

#: Filenames used by the shared bundle contract.
TREE_FILENAME = "labeled_trees.jsonl"
SPLIT_IDS_FILENAME = "split_ids.json"
MANIFEST_FILENAME = "manifest.json"

#: Tree ``version`` strings this loader has been validated against. Unknown
#: versions are *tolerated* (extra/unknown fields never fail the load); the value
#: is preserved as ``schema_version`` in tree metadata so callers can gate on it.
KNOWN_TREE_VERSIONS = frozenset({"3.0"})

#: Fields a bundle line must carry to be loadable. Missing ones raise.
_REQUIRED_TREE_FIELDS = ("doc_id", "nodes")
_REQUIRED_NODE_FIELDS = ("node_id", "level")

#: Node-level typed fields lifted into ``metadata`` so nothing in the source row
#: is dropped (``metadata`` already carries the rest of the payload).
_LIFTED_NODE_FIELDS = (
    "score",
    "dimension_scores",
    "reasoning",
    "confidence",
    "timestamp",
    "doc_id",
    "level",
)


class BundleFormatError(ValueError):
    """Raised when a bundle is missing a required field or is otherwise malformed."""


def load_labeled_tree_bundle(
    path: str | Path,
    *,
    split: str | None = None,
    dimension: str | None = None,
) -> list[TreeRecord]:
    """Load an external ``LabeledTree`` bundle into ``TreeRecord`` objects.

    Parameters
    ----------
    path:
        Either a ``labeled_trees.jsonl`` file, a directory containing one (a
        single leaf-scale run), or a top-level bundle directory that holds a
        ``manifest.json`` and exactly one leaf-scale run. Top-level bundles with
        several leaf-scale runs are ambiguous by design; the error names the run
        directories to pass instead.
    split:
        Optional ``"train"``/``"val"``/``"test"`` filter. When the bundle pins
        ``split_ids.json`` those ids are authoritative and are never resampled;
        otherwise the per-tree ``metadata["split"]`` field is used. Requesting a
        split a bundle cannot resolve raises :class:`BundleFormatError`.
    dimension:
        Optional target dimension (e.g. ``"rile"``, ``"domain_3"``). When given,
        each node label and the tree-level ``root_label`` are taken from that
        dimension's ``dimension_scores``; when omitted, the scalar node ``score``
        and the tree ``document_score`` are used.

    Returns
    -------
    list[TreeRecord]
        One record per document, with all per-node payload preserved on node
        ``metadata`` and topology (parent/child edges, levels) reconstructed.
    """

    tree_file, split_ids_dirs = _resolve_tree_file(Path(path))
    split_ids = _load_split_ids(split_ids_dirs)

    records: list[TreeRecord] = []
    available_dimensions: set[str] | None = None
    with tree_file.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise BundleFormatError(
                    f"{tree_file}:{line_no}: line is not valid JSON: {exc}"
                ) from exc
            if not isinstance(row, Mapping):
                raise BundleFormatError(
                    f"{tree_file}:{line_no}: each line must be a JSON object, got {type(row).__name__}"
                )
            if available_dimensions is None:
                available_dimensions = _dimensions_of(row)
                if dimension is not None and dimension not in available_dimensions:
                    raise BundleFormatError(
                        f"{tree_file}: dimension {dimension!r} not in bundle; "
                        f"available: {sorted(available_dimensions)}"
                    )
            records.append(_record_from_row(row, dimension=dimension, where=f"{tree_file}:{line_no}"))

    if split is not None:
        records = _filter_by_split(records, split=split, split_ids=split_ids, where=str(tree_file))
    return records


def _resolve_tree_file(path: Path) -> tuple[Path, tuple[Path, ...]]:
    """Return the ``labeled_trees.jsonl`` file and directories to search for split ids."""

    if not path.exists():
        raise BundleFormatError(f"bundle path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() != ".jsonl":
            raise BundleFormatError(f"bundle file must be a .jsonl labeled-trees file: {path}")
        return path, (path.parent, path.parent.parent)
    # Directory: a leaf-scale run, or a top-level bundle with a manifest.
    direct = path / TREE_FILENAME
    if direct.is_file():
        return direct, (path, path.parent)
    if (path / MANIFEST_FILENAME).is_file():
        run_files = sorted(
            child / TREE_FILENAME
            for child in path.iterdir()
            if child.is_dir() and (child / TREE_FILENAME).is_file()
        )
        if len(run_files) == 1:
            # Split ids live at the top-level bundle root.
            return run_files[0], (run_files[0].parent, path)
        if not run_files:
            raise BundleFormatError(
                f"manifest bundle has no leaf-scale runs with {TREE_FILENAME}: {path}"
            )
        run_dirs = ", ".join(str(rf.parent.name) for rf in run_files)
        raise BundleFormatError(
            f"top-level bundle {path} holds several leaf-scale runs ({run_dirs}); "
            f"pass one run directory (e.g. {run_files[0].parent})"
        )
    raise BundleFormatError(
        f"directory has no {TREE_FILENAME} and no {MANIFEST_FILENAME}: {path}"
    )


def _load_split_ids(search_dirs: tuple[Path, ...]) -> dict[str, set[str]] | None:
    for directory in search_dirs:
        candidate = directory / SPLIT_IDS_FILENAME
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise BundleFormatError(f"{candidate}: split-ids file must be a JSON object")
            return {
                str(name): {str(doc_id) for doc_id in (ids or [])}
                for name, ids in payload.items()
                if isinstance(ids, (list, tuple))
            }
    return None


def _filter_by_split(
    records: list[TreeRecord],
    *,
    split: str,
    split_ids: dict[str, set[str]] | None,
    where: str,
) -> list[TreeRecord]:
    if split_ids is not None:
        if split not in split_ids:
            raise BundleFormatError(
                f"{where}: split {split!r} not in split_ids; available: {sorted(split_ids)}"
            )
        allowed = split_ids[split]
        return [record for record in records if str(record.doc_id) in allowed]
    # No pinned ids: fall back to per-tree metadata, but only if it exists.
    if not any("split" in dict(record.metadata or {}) for record in records):
        raise BundleFormatError(
            f"{where}: split={split!r} requested but bundle has no split_ids.json "
            "and no per-tree metadata 'split' field"
        )
    return [record for record in records if str(dict(record.metadata or {}).get("split")) == split]


def _dimensions_of(row: Mapping[str, Any]) -> set[str]:
    dims: set[str] = set()
    nodes = row.get("nodes")
    if isinstance(nodes, Mapping):
        for node in nodes.values():
            scores = node.get("dimension_scores") if isinstance(node, Mapping) else None
            if isinstance(scores, Mapping):
                dims.update(str(key) for key in scores)
    return dims


def _record_from_row(row: Mapping[str, Any], *, dimension: str | None, where: str) -> TreeRecord:
    for field in _REQUIRED_TREE_FIELDS:
        if field not in row or row[field] is None:
            raise BundleFormatError(f"{where}: tree is missing required field {field!r}")
    nodes_raw = row["nodes"]
    if not isinstance(nodes_raw, Mapping):
        raise BundleFormatError(f"{where}: 'nodes' must be a mapping of node_id -> node")

    doc_id = str(row["doc_id"])
    levels = row.get("levels")
    position_of = _positions_from_levels(levels)
    parent_of = _parents_from_children(nodes_raw, where=where)
    child_ids = set(parent_of)
    root_id = _root_node_id(nodes_raw, child_ids=child_ids, where=where)

    nodes: list[TreeNode] = []
    for order_index, (node_id, node) in enumerate(nodes_raw.items()):
        if not isinstance(node, Mapping):
            raise BundleFormatError(f"{where}: node {node_id!r} must be an object")
        for field in _REQUIRED_NODE_FIELDS:
            if field not in node or node[field] is None:
                raise BundleFormatError(f"{where}: node {node_id!r} missing required field {field!r}")
        level = _as_int(node["level"], where=f"{where}: node {node_id!r} level")
        node_id_str = str(node["node_id"])
        is_leaf = bool(_node_metadata(node).get("is_leaf")) or level == 0
        if node_id_str == root_id:
            unit_type = "root"
        elif is_leaf:
            unit_type = "leaf"
        else:
            unit_type = "merge"
        nodes.append(
            TreeNode(
                node_id=node_id_str,
                unit_type=unit_type,
                text=str(node.get("text") or ""),
                level=level,
                position=position_of.get(node_id_str, order_index),
                parent_id=parent_of.get(node_id_str),
                left_child_id=_optional_str(node.get("left_child_id")),
                right_child_id=_optional_str(node.get("right_child_id")),
                label=_node_label(node, dimension=dimension),
                state=None,
                metadata=_preserved_node_payload(node),
            )
        )

    root_label = _root_label(row, nodes_raw.get(root_id) if root_id else None, dimension=dimension)
    metadata = {
        **dict(row.get("metadata") or {}),
        "schema_version": row.get("version"),
        "doc_id": doc_id,
        "document_score": row.get("document_score"),
        "label_source": row.get("label_source"),
        "selected_dimension": dimension,
    }
    return TreeRecord(
        tree_id=doc_id,
        doc_id=doc_id,
        text=str(row.get("document_text") or ""),
        root_label=root_label,
        nodes=tuple(nodes),
        metadata=metadata,
    )


def _preserved_node_payload(node: Mapping[str, Any]) -> dict[str, Any]:
    """Return a lossless copy of the node payload as tree-node metadata."""

    preserved: dict[str, Any] = dict(_node_metadata(node))
    for field in _LIFTED_NODE_FIELDS:
        if field in node and field not in preserved:
            preserved[field] = node[field]
    return preserved


def _node_metadata(node: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = node.get("metadata")
    return meta if isinstance(meta, Mapping) else {}


def _node_label(node: Mapping[str, Any], *, dimension: str | None) -> Any:
    if dimension is not None:
        scores = node.get("dimension_scores")
        if isinstance(scores, Mapping) and dimension in scores and scores[dimension] is not None:
            return float(scores[dimension])
        # Fall through to the scalar score when a node lacks the dimension.
    score = node.get("score")
    return None if score is None else float(score)


def _root_label(row: Mapping[str, Any], root_node: Mapping[str, Any] | None, *, dimension: str | None) -> Any:
    if dimension is not None and isinstance(root_node, Mapping):
        scores = root_node.get("dimension_scores")
        if isinstance(scores, Mapping) and dimension in scores and scores[dimension] is not None:
            return float(scores[dimension])
    score = row.get("document_score")
    return None if score is None else float(score)


def _positions_from_levels(levels: Any) -> dict[str, int]:
    positions: dict[str, int] = {}
    if isinstance(levels, (list, tuple)):
        for level_ids in levels:
            if isinstance(level_ids, (list, tuple)):
                for position, node_id in enumerate(level_ids):
                    positions[str(node_id)] = int(position)
    return positions


def _parents_from_children(nodes_raw: Mapping[str, Any], *, where: str) -> dict[str, str]:
    parent_of: dict[str, str] = {}
    for node_id, node in nodes_raw.items():
        if not isinstance(node, Mapping):
            continue
        for child in (node.get("left_child_id"), node.get("right_child_id")):
            child_id = _optional_str(child)
            if child_id is not None:
                parent_of[child_id] = str(node.get("node_id") or node_id)
    return parent_of


def _root_node_id(nodes_raw: Mapping[str, Any], *, child_ids: set[str], where: str) -> str | None:
    unparented: list[tuple[int, str]] = []
    for node_id, node in nodes_raw.items():
        node_id_str = str(node.get("node_id") or node_id) if isinstance(node, Mapping) else str(node_id)
        if node_id_str in child_ids:
            continue
        level = 0
        if isinstance(node, Mapping):
            level = _as_int(node.get("level", 0), where=f"{where}: node {node_id_str!r} level", default=0)
        unparented.append((level, node_id_str))
    if not unparented:
        return None
    # The root is the unparented node at the greatest depth.
    unparented.sort(key=lambda item: item[0])
    return unparented[-1][1]


def _as_int(value: Any, *, where: str, default: int | None = None) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        if default is not None:
            return default
        raise BundleFormatError(f"{where} must be an integer, got {value!r}") from exc


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
