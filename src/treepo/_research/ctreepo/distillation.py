"""Distillation helpers for C-TreePO labeled-node training.

This module keeps the teacher-first distillation path explicit:

1. Stage 0 writes offline ``LabeledTree`` artifacts with node labels.
2. Stage 1 attaches those labels to runtime embedding-tree nodes as
   ``node.oracle_scores["rile"]``.
3. The existing CTreePO trainer consumes those labels through its normal
   sparse local-law supervision path.

No teacher/scorer calls happen in this module.  Callers must supply already
materialized labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from treepo._research.training.config_sections import (
    OptimizerConfig,
    RunConfig,
    TestConfig,
    TrainConfig,
    ValidationConfig,
    config_to_dict,
)
from treepo._research.ctreepo.contracts import (
    SOURCE_KIND_EXTERNAL_STATE,
    SOURCE_KIND_RAW_INPUT,
    normalize_tree_bundle_manifest,
)
from treepo._research.tree.embedding_tree import EmbeddingTreeNode
from treepo._research.tree.labeled import LabeledNode, LabeledTree


LABEL_TREE_VERSION = "ctreepo_labeled_node_distillation_v1"

TRAIN_TARGET_TREE_OPERATOR = "tree_operator"
TRAIN_TARGET_G = "g"
TRAIN_TARGET_F = "f"

STUDENT_MODEL_CTREEPO_EMBEDDING_TREE = "ctreepo_embedding_tree"
STUDENT_MODEL_LM_SFT = "lm_sft"
STUDENT_MODEL_EMBEDDING_RIDGE_PROXY = "embedding_ridge_proxy"
STUDENT_MODEL_LM_SCALAR_REGRESSION = "lm_scalar_regression"

SUPERVISION_SOURCE_LABELED_TREE_ARTIFACT = "labeled_tree_artifact"
SUPERVISION_SOURCE_EMPIRICAL_ROOT_LABELS = "empirical_root_labels"

FULL_DOC_ANCHOR_OFF = "off"
FULL_DOC_ANCHOR_STORED_SUMMARY = "stored_summary"
FULL_DOC_ANCHOR_RAW_DOCUMENT = "raw_document"
FULL_DOC_ANCHOR_BOTH = "both"
FULL_DOC_ANCHOR_TARGET_EXPERT = "expert"
FULL_DOC_ANCHOR_TARGET_TEACHER = "teacher"
NODE_WEIGHT_NORMALIZATION_PER_TREE = "per_tree"
NODE_WEIGHT_NORMALIZATION_NONE = "none"
DEFAULT_LOCAL_LAW_WEIGHT_WITH_ANCHORS = 0.25
ROOT_LABEL_SOURCE_STORED_SUMMARY = FULL_DOC_ANCHOR_STORED_SUMMARY
ROOT_LABEL_SOURCE_RAW_DOCUMENT = FULL_DOC_ANCHOR_RAW_DOCUMENT
ROOT_LABEL_TARGET_EXPERT = FULL_DOC_ANCHOR_TARGET_EXPERT
ROOT_LABEL_TARGET_TEACHER = FULL_DOC_ANCHOR_TARGET_TEACHER

PAPER_TO_LEAN_LOCAL_LAW_MAPPING = {
    "C1": "Lean L1 leaf preservation",
    "C2": "Lean L3 on-range idempotence",
    "C3": "Lean L2 merge preservation against node span",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_score(
    value: float,
    *,
    target_min: float = -100.0,
    target_max: float = 100.0,
) -> float:
    span = float(target_max) - float(target_min)
    if span <= 0.0:
        return 0.5
    return _clamp01((float(value) - float(target_min)) / span)


def _normalise_anchor_mode(value: str) -> str:
    mode = str(value or FULL_DOC_ANCHOR_OFF).strip().lower().replace("-", "_")
    aliases = {
        "none": FULL_DOC_ANCHOR_OFF,
        "summary": FULL_DOC_ANCHOR_STORED_SUMMARY,
        "stored": FULL_DOC_ANCHOR_STORED_SUMMARY,
        "raw": FULL_DOC_ANCHOR_RAW_DOCUMENT,
    }
    mode = aliases.get(mode, mode)
    allowed = {
        FULL_DOC_ANCHOR_OFF,
        FULL_DOC_ANCHOR_STORED_SUMMARY,
        FULL_DOC_ANCHOR_RAW_DOCUMENT,
        FULL_DOC_ANCHOR_BOTH,
    }
    if mode not in allowed:
        raise ValueError(f"unknown full-doc anchor mode: {value!r}")
    return mode


def _normalise_root_label_sources(value: Optional[Sequence[str] | str]) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return tuple()
        lowered = raw.lower().replace("-", "_")
        if lowered in {FULL_DOC_ANCHOR_OFF, "none"}:
            return tuple()
        if lowered == FULL_DOC_ANCHOR_BOTH:
            return (ROOT_LABEL_SOURCE_STORED_SUMMARY, ROOT_LABEL_SOURCE_RAW_DOCUMENT)
        values: Sequence[str] = tuple(part.strip() for part in raw.split(",") if part.strip())
    else:
        values = tuple(str(part).strip() for part in value if str(part).strip())
    out: List[str] = []
    for item in values:
        source = str(item).strip().lower().replace("-", "_")
        aliases = {
            "summary": ROOT_LABEL_SOURCE_STORED_SUMMARY,
            "stored": ROOT_LABEL_SOURCE_STORED_SUMMARY,
            "raw": ROOT_LABEL_SOURCE_RAW_DOCUMENT,
        }
        source = aliases.get(source, source)
        if source not in {ROOT_LABEL_SOURCE_STORED_SUMMARY, ROOT_LABEL_SOURCE_RAW_DOCUMENT}:
            raise ValueError(
                f"unknown root label source {item!r}; expected stored_summary or raw_document"
            )
        if source not in out:
            out.append(source)
    return tuple(out)


def _anchor_mode_from_root_label_sources(value: Optional[Sequence[str] | str]) -> str:
    sources = _normalise_root_label_sources(value)
    if not sources:
        return FULL_DOC_ANCHOR_OFF
    if set(sources) == {ROOT_LABEL_SOURCE_STORED_SUMMARY, ROOT_LABEL_SOURCE_RAW_DOCUMENT}:
        return FULL_DOC_ANCHOR_BOTH
    return sources[0]


def _normalise_anchor_target(value: str) -> str:
    target = str(value or FULL_DOC_ANCHOR_TARGET_EXPERT).strip().lower().replace("-", "_")
    allowed = {FULL_DOC_ANCHOR_TARGET_EXPERT, FULL_DOC_ANCHOR_TARGET_TEACHER}
    if target not in allowed:
        raise ValueError(f"unknown full-doc anchor target: {value!r}")
    return target


def _normalise_root_label_target(value: str) -> str:
    return _normalise_anchor_target(value)


def _reject_legacy_anchor_kwargs(kwargs: Mapping[str, Any]) -> None:
    legacy = sorted(str(key) for key in kwargs if str(key).startswith("full_doc_anchor_"))
    if legacy:
        raise TypeError(
            "legacy full_doc_anchor_* public parameters are not supported: "
            + ", ".join(legacy)
            + ". Use root_label_sources and root_label_target."
        )


def _normalise_node_weight_normalization(value: str) -> str:
    mode = str(value or NODE_WEIGHT_NORMALIZATION_PER_TREE).strip().lower().replace("-", "_")
    aliases = {"tree": NODE_WEIGHT_NORMALIZATION_PER_TREE}
    mode = aliases.get(mode, mode)
    allowed = {NODE_WEIGHT_NORMALIZATION_PER_TREE, NODE_WEIGHT_NORMALIZATION_NONE}
    if mode not in allowed:
        raise ValueError(f"unknown node weight normalization: {value!r}")
    return mode


def _validate_local_law_weight(value: float) -> float:
    raw = float(value)
    if not math.isfinite(raw) or raw < 0.0 or raw > 1.0:
        raise ValueError(f"local_law_weight must be in [0, 1], got {value!r}")
    return float(raw)


def _objective_masses(
    *,
    full_doc_anchor_mode: str,
    local_law_weight: Optional[float] = None,
) -> Tuple[float, float]:
    anchor_mode = _normalise_anchor_mode(full_doc_anchor_mode)
    if anchor_mode == FULL_DOC_ANCHOR_OFF:
        if local_law_weight is not None:
            local_mass = _validate_local_law_weight(float(local_law_weight))
            if not math.isclose(local_mass, 1.0):
                raise ValueError("full_doc_anchor_mode=off requires local_law_weight=1.0")
        # Preserve legacy node-only training when anchors are not enabled.
        return 0.0, 1.0
    local_mass = _validate_local_law_weight(
        DEFAULT_LOCAL_LAW_WEIGHT_WITH_ANCHORS
        if local_law_weight is None
        else float(local_law_weight)
    )
    return float(1.0 - local_mass), float(local_mass)


def _node_record_weight(
    *,
    n_teacher_records_for_tree: int,
    full_doc_anchor_mode: str,
    local_law_weight: Optional[float] = None,
    node_weight_normalization: str,
) -> float:
    _, teacher_mass = _objective_masses(
        full_doc_anchor_mode=full_doc_anchor_mode,
        local_law_weight=local_law_weight,
    )
    if teacher_mass <= 0.0:
        return 0.0
    if _normalise_anchor_mode(full_doc_anchor_mode) == FULL_DOC_ANCHOR_OFF:
        return 1.0
    norm = _normalise_node_weight_normalization(node_weight_normalization)
    if norm == NODE_WEIGHT_NORMALIZATION_PER_TREE:
        return float(teacher_mass) / float(max(1, int(n_teacher_records_for_tree)))
    return float(teacher_mass)


def _root_anchor_weight(
    *,
    full_doc_anchor_mode: str,
    local_law_weight: Optional[float] = None,
) -> float:
    gold_mass, _ = _objective_masses(
        full_doc_anchor_mode=full_doc_anchor_mode,
        local_law_weight=local_law_weight,
    )
    return gold_mass


def _uniform_windows(
    text_len: int,
    window_size: int,
    window_overlap: int = 0,
) -> List[Tuple[int, int]]:
    """Mirror ``src.tree.embedding_tree._uniform_windows`` without embeddings."""
    if text_len <= 0 or window_size <= 0:
        return [(0, max(text_len, 0))]
    if text_len <= window_size:
        return [(0, text_len)]

    step = max(1, int(window_size) - int(window_overlap))
    windows: List[Tuple[int, int]] = []
    start = 0
    while start < text_len:
        end = min(start + int(window_size), text_len)
        windows.append((start, end))
        if end >= text_len:
            break
        start += step
    return windows


def _target_leaf_windows(text_len: int, target_leaves: int) -> List[Tuple[int, int]]:
    """Return near-even non-overlapping windows with at most ``target_leaves`` leaves."""
    n = max(1, int(target_leaves))
    text_len = max(0, int(text_len))
    if text_len == 0:
        return [(0, 0)]
    if text_len <= n:
        return [(idx, idx + 1) for idx in range(text_len)]
    windows: List[Tuple[int, int]] = []
    for idx in range(n):
        start = int(math.floor(idx * text_len / n))
        end = int(math.floor((idx + 1) * text_len / n))
        if end <= start:
            end = min(text_len, start + 1)
        windows.append((start, end))
    return windows


@dataclass(frozen=True)
class _TreeNodeSpec:
    node_id: str
    level: int
    char_start: int
    char_end: int
    left_child_id: Optional[str] = None
    right_child_id: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        return self.left_child_id is None and self.right_child_id is None


def _build_binary_node_specs(
    *,
    text_len: int,
    window_size: int,
    window_overlap: int,
    target_leaves_per_doc: Optional[int] = None,
    explicit_char_windows: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[List[_TreeNodeSpec], List[List[str]], List[Dict[str, str]]]:
    if explicit_char_windows is not None:
        windows = _normalize_explicit_char_windows(
            explicit_char_windows,
            text_len=text_len,
        )
    elif target_leaves_per_doc is not None and int(target_leaves_per_doc) > 0:
        windows = _target_leaf_windows(text_len, int(target_leaves_per_doc))
    else:
        windows = _uniform_windows(
            text_len,
            window_size=int(window_size),
            window_overlap=int(window_overlap),
        )
    specs: List[_TreeNodeSpec] = []
    levels: List[List[str]] = []
    triples: List[Dict[str, str]] = []

    leaf_ids: List[str] = []
    for idx, (start, end) in enumerate(windows):
        node_id = f"node_l0_{idx:05d}"
        specs.append(
            _TreeNodeSpec(
                node_id=node_id,
                level=0,
                char_start=int(start),
                char_end=int(end),
            )
        )
        leaf_ids.append(node_id)
    levels.append(leaf_ids)

    current_specs = specs[:]
    level = 1
    while len(current_specs) > 1:
        next_specs: List[_TreeNodeSpec] = []
        level_ids: List[str] = []
        for pair_idx in range(0, len(current_specs), 2):
            left = current_specs[pair_idx]
            if pair_idx + 1 < len(current_specs):
                right = current_specs[pair_idx + 1]
            else:
                right = left
            node_id = f"node_l{level}_{len(next_specs):05d}"
            parent = _TreeNodeSpec(
                node_id=node_id,
                level=level,
                char_start=int(left.char_start),
                char_end=int(right.char_end),
                left_child_id=left.node_id,
                right_child_id=right.node_id,
            )
            specs.append(parent)
            next_specs.append(parent)
            level_ids.append(node_id)
            triples.append(
                {
                    "left_node_id": left.node_id,
                    "right_node_id": right.node_id,
                    "parent_node_id": node_id,
                }
            )
        levels.append(level_ids)
        current_specs = next_specs
        level += 1

    return specs, levels, triples


def _normalize_explicit_char_windows(
    windows: Sequence[Tuple[int, int]],
    *,
    text_len: int,
) -> List[Tuple[int, int]]:
    """Validate exact caller-supplied leaf windows.

    Size-token teacher traces compute leaves with a tokenizer and then replay
    those char spans here. The spans must cover the document exactly; otherwise
    downstream tree replay would silently drop or duplicate text.
    """
    text_len = max(0, int(text_len))
    if not windows:
        return [(0, text_len)]
    normalized: List[Tuple[int, int]] = []
    expected_start = 0
    for idx, pair in enumerate(windows):
        if len(pair) != 2:
            raise ValueError(f"explicit char window {idx} is not a (start, end) pair: {pair!r}")
        start, end = int(pair[0]), int(pair[1])
        if start != expected_start:
            raise ValueError(
                "explicit char windows must be contiguous and cover the document: "
                f"window {idx} starts at {start}, expected {expected_start}"
            )
        if end < start:
            raise ValueError(f"explicit char window {idx} has end < start: {(start, end)!r}")
        if end > text_len:
            raise ValueError(
                f"explicit char window {idx} ends at {end}, beyond text_len={text_len}"
            )
        normalized.append((start, end))
        expected_start = end
    if expected_start != text_len:
        raise ValueError(
            "explicit char windows must cover the full document: "
            f"last end={expected_start}, text_len={text_len}"
        )
    return normalized


def build_labeled_tree_from_text(
    *,
    doc_id: str,
    text: str,
    document_score: float,
    split: str,
    score_fn: Callable[[str], float],
    window_size: int,
    window_overlap: int = 0,
    target_leaves_per_doc: Optional[int] = None,
    explicit_char_windows: Optional[Sequence[Tuple[int, int]]] = None,
    label_source: str = "teacher",
    root_summary: Optional[str] = None,
    resummary_target: Optional[str] = None,
    node_summaries: Optional[Mapping[str, str]] = None,
    node_summary_fn: Optional[Callable[[str, Mapping[str, Any]], Optional[str]]] = None,
    fill_missing_summaries_from_span: bool = False,
    summary_source: str = "teacher",
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> LabeledTree:
    """Build a node-labeled binary tree from text and an offline scorer.

    ``score_fn`` is called once per node span.  Optional summary targets are
    stored in node metadata so the same artifact can supervise both the text
    student ``g`` and the scalar score head ``f``.  Use this only in Stage 0
    while teacher/scorer outputs are intentionally available; training-time
    replay should load the resulting artifact instead.
    """
    rendered = str(text or "")
    specs, levels, triples = _build_binary_node_specs(
        text_len=len(rendered),
        window_size=int(window_size),
        window_overlap=int(window_overlap),
        target_leaves_per_doc=target_leaves_per_doc,
        explicit_char_windows=explicit_char_windows,
    )
    if explicit_char_windows is not None:
        topology_policy = {
            "kind": "explicit_char_windows",
            "n_windows": int(len(explicit_char_windows)),
            "actual_leaves": int(len(levels[0]) if levels else 0),
        }
    elif target_leaves_per_doc is not None and int(target_leaves_per_doc) > 0:
        topology_policy = {
            "kind": "target_leaves_per_doc",
            "target_leaves_per_doc": int(target_leaves_per_doc),
            "actual_leaves": int(len(levels[0]) if levels else 0),
        }
    else:
        topology_policy = {
            "kind": "fixed_char_windows",
            "leaf_size_chars": int(window_size),
            "window_overlap_chars": int(window_overlap),
        }
    metadata: Dict[str, Any] = {
        "artifact_version": LABEL_TREE_VERSION,
        "split": str(split),
        "leaf_size_chars": int(window_size),
        "window_overlap_chars": int(window_overlap),
        "target_leaves_per_doc": (
            int(target_leaves_per_doc)
            if target_leaves_per_doc is not None and int(target_leaves_per_doc) > 0
            else None
        ),
        "explicit_char_windows": (
            [(int(start), int(end)) for start, end in explicit_char_windows]
            if explicit_char_windows is not None
            else None
        ),
        "topology_policy": topology_policy,
        "topology_replay": "exact_artifact_spans",
        "sibling_triples": triples,
        "idempotence_pairs": [],
        "label_source": str(label_source),
        "paper_to_lean_local_law_mapping": dict(PAPER_TO_LEAN_LOCAL_LAW_MAPPING),
        "distillation_state_contract": {
            "g_target_kind": "text_summary",
            "f_input_kind": "summary_embedding",
            "hidden_state": {
                "backend": None,
                "layer": None,
                "pooling": None,
            },
        },
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))

    if root_summary and resummary_target:
        metadata["idempotence_pairs"].append(
            {
                "input_summary": str(root_summary),
                "target_resummary": str(resummary_target),
                "source": "teacher_trace_root",
            }
        )

    tree = LabeledTree(
        doc_id=str(doc_id),
        document_text=rendered,
        document_score=float(document_score),
        metadata=metadata,
        label_source=str(label_source),
    )

    tree.levels = [list(row) for row in levels]
    summary_by_node_id: Dict[str, str] = {}
    explicit_node_summaries = dict(node_summaries or {})
    root_node_id = specs[-1].node_id if specs else ""
    for spec in specs:
        span = rendered[int(spec.char_start) : int(spec.char_end)]
        score = float(score_fn(span))
        left_summary = (
            summary_by_node_id.get(str(spec.left_child_id))
            if spec.left_child_id
            else None
        )
        right_summary = (
            summary_by_node_id.get(str(spec.right_child_id))
            if spec.right_child_id
            else None
        )
        summary_target: Optional[str] = None
        summary_target_source = str(summary_source or "teacher")
        if spec.node_id in explicit_node_summaries:
            summary_target = str(explicit_node_summaries[spec.node_id] or "").strip()
            summary_target_source = "node_summaries"
        elif root_summary and spec.node_id == root_node_id:
            summary_target = str(root_summary).strip()
            summary_target_source = str(summary_source or "teacher_trace_root")
        elif node_summary_fn is not None:
            summary_context = {
                "node_id": str(spec.node_id),
                "level": int(spec.level),
                "char_start": int(spec.char_start),
                "char_end": int(spec.char_end),
                "is_leaf": bool(spec.is_leaf),
                "left_child_id": spec.left_child_id,
                "right_child_id": spec.right_child_id,
                "left_summary": left_summary,
                "right_summary": right_summary,
            }
            maybe_summary = node_summary_fn(span, summary_context)
            if maybe_summary is not None and str(maybe_summary).strip():
                summary_target = str(maybe_summary).strip()
                summary_target_source = "node_summary_fn"
        elif bool(fill_missing_summaries_from_span):
            summary_target = span.strip()
            summary_target_source = "span_identity_fallback"

        node_metadata: Dict[str, Any] = {
            "char_start": int(spec.char_start),
            "char_end": int(spec.char_end),
            "node_id": str(spec.node_id),
            "is_leaf": bool(spec.is_leaf),
            "label_source": str(label_source),
            "g_training_role": "leaf" if spec.is_leaf else "merge",
            "f_input_kind": "summary_embedding",
            "hidden_state_backend": None,
            "hidden_state_layer": None,
            "hidden_state_pooling": None,
        }
        if summary_target:
            node_metadata["teacher_summary"] = str(summary_target)
            node_metadata["target_summary"] = str(summary_target)
            node_metadata["teacher_summary_source"] = str(summary_target_source)
            if spec.is_leaf:
                node_metadata["teacher_leaf_summary"] = str(summary_target)
            else:
                node_metadata["teacher_merge_summary"] = str(summary_target)
            summary_by_node_id[str(spec.node_id)] = str(summary_target)
        else:
            node_metadata["missing_teacher_summary"] = True
        if root_summary and spec.node_id == specs[-1].node_id:
            if resummary_target:
                node_metadata["teacher_resummary"] = str(resummary_target)
        tree.add_node(
            LabeledNode(
                node_id=str(spec.node_id),
                doc_id=str(doc_id),
                level=int(spec.level),
                text=span,
                score=score,
                left_child_id=spec.left_child_id,
                right_child_id=spec.right_child_id,
                metadata=node_metadata,
            )
        )
    annotate_labeled_tree_summary_coverage(tree)
    return tree


def annotate_labeled_tree_summary_coverage(tree: LabeledTree) -> LabeledTree:
    """Attach in-memory summary coverage metadata for full/partial artifacts."""
    missing: List[str] = []
    present = 0
    for node in tree.nodes.values():
        if _summary_target_for_node(node, include_identity_targets=False):
            present += 1
            if isinstance(node.metadata, dict):
                node.metadata.pop("missing_teacher_summary", None)
        else:
            missing.append(str(node.node_id))
            if isinstance(node.metadata, dict):
                node.metadata["missing_teacher_summary"] = True
                node.metadata.setdefault("g_training_role", "leaf" if int(node.level) == 0 else "merge")
    total = len(tree.nodes)
    metadata = dict(tree.metadata or {})
    metadata.setdefault("artifact_version", LABEL_TREE_VERSION)
    metadata.setdefault("paper_to_lean_local_law_mapping", dict(PAPER_TO_LEAN_LOCAL_LAW_MAPPING))
    metadata["summary_coverage"] = {
        "total_nodes": int(total),
        "nodes_with_teacher_summary": int(present),
        "nodes_missing_teacher_summary": int(len(missing)),
        "missing_node_ids": missing,
        "partial_artifact": bool(missing),
    }
    tree.metadata = metadata
    return tree


def repair_labeled_tree_missing_summaries(
    trees: Sequence[LabeledTree],
    summary_fn: Callable[..., Optional[str]],
    *,
    source: str = "summary_repair",
) -> List[LabeledTree]:
    """Fill missing node summary targets with a caller-supplied teacher function."""
    repaired: List[LabeledTree] = []
    for tree in trees:
        for node in tree.nodes.values():
            if _summary_target_for_node(node, include_identity_targets=False):
                continue
            context = {
                "doc_id": tree.doc_id,
                "node_id": node.node_id,
                "level": int(node.level),
                "is_leaf": int(node.level) == 0,
                "left_child_id": node.left_child_id,
                "right_child_id": node.right_child_id,
                "score": float(node.score),
                "metadata": dict(node.metadata or {}),
            }
            try:
                maybe = summary_fn(str(node.text), context)
            except TypeError:
                maybe = summary_fn(str(node.text))
            if maybe is None or not str(maybe).strip():
                continue
            summary = str(maybe).strip()
            node.metadata["teacher_summary"] = summary
            node.metadata["target_summary"] = summary
            node.metadata["teacher_summary_source"] = str(source)
            node.metadata.pop("missing_teacher_summary", None)
            if int(node.level) == 0:
                node.metadata["teacher_leaf_summary"] = summary
            else:
                node.metadata["teacher_merge_summary"] = summary
        annotate_labeled_tree_summary_coverage(tree)
        repaired.append(tree)
    return repaired


def write_labeled_trees_jsonl(path: Path, trees: Iterable[LabeledTree]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for tree in trees:
            metadata = dict(tree.metadata or {})
            metadata.setdefault("source_artifact", str(path))
            metadata.setdefault("labeled_trees_path", str(path))
            tree.metadata = metadata
            handle.write(json.dumps(tree.to_dict(), ensure_ascii=True) + "\n")
    return path


def write_jsonl_records(path: Path, records: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=True) + "\n")
    return path


def load_labeled_trees(path: Path | str) -> List[LabeledTree]:
    """Load one labeled tree file, a JSONL bundle, or all JSON files in a dir."""
    root = Path(path)
    if root.is_dir():
        trees: List[LabeledTree] = []
        for child in sorted(root.glob("*.json")):
            trees.append(LabeledTree.load(child))
        for child in sorted(root.glob("*.jsonl")):
            trees.extend(load_labeled_trees(child))
        return trees

    if root.suffix.lower() == ".jsonl":
        loaded: List[LabeledTree] = []
        with root.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                tree = annotate_labeled_tree_summary_coverage(LabeledTree.from_dict(json.loads(text)))
                metadata = dict(tree.metadata or {})
                metadata.setdefault("source_artifact", str(root))
                metadata.setdefault("labeled_trees_path", str(root))
                tree.metadata = metadata
                loaded.append(tree)
        return loaded

    return [annotate_labeled_tree_summary_coverage(LabeledTree.load(root))]


def _label_lookup(labeled_tree: LabeledTree) -> Dict[Tuple[int, int, int], LabeledNode]:
    lookup: Dict[Tuple[int, int, int], LabeledNode] = {}
    for node in labeled_tree.nodes.values():
        meta = dict(node.metadata or {})
        if "char_start" not in meta or "char_end" not in meta:
            continue
        key = (int(node.level), int(meta["char_start"]), int(meta["char_end"]))
        lookup[key] = node
    return lookup


def attach_labeled_tree_scores(
    nodes: Sequence[EmbeddingTreeNode],
    labeled_tree: LabeledTree,
    *,
    head: str = "rile",
) -> Dict[str, Any]:
    """Attach cached labeled-node scores to embedding-tree nodes.

    Matching is by ``(level, char_start, char_end)``, which is stable when the
    training run uses the same fixed leaf size and overlap as the artifact.
    """
    lookup = _label_lookup(labeled_tree)
    attached = 0
    missing = 0
    leaf_attached = 0
    internal_attached = 0
    for node in nodes:
        key = (int(node.level), int(node.char_start), int(node.char_end))
        labeled = lookup.get(key)
        if labeled is None:
            missing += 1
            continue
        node.oracle_scores[str(head)] = float(labeled.score)
        node.summary = str(labeled.metadata.get("teacher_summary", "") or node.summary or "")
        attached += 1
        if node.is_leaf:
            leaf_attached += 1
        else:
            internal_attached += 1
    return {
        "attached": int(attached),
        "missing": int(missing),
        "leaf_attached": int(leaf_attached),
        "internal_attached": int(internal_attached),
        "total_runtime_nodes": int(len(nodes)),
        "total_labeled_nodes": int(len(labeled_tree.nodes)),
    }


def _node_span(node: LabeledNode) -> Tuple[int, int]:
    meta = dict(node.metadata or {})
    start = int(_safe_float(meta.get("char_start"), 0.0))
    end = int(_safe_float(meta.get("char_end"), start + len(str(node.text or ""))))
    return start, end


def _ordered_labeled_nodes_by_level(tree: LabeledTree) -> List[LabeledNode]:
    ordered: List[LabeledNode] = []
    seen: set[str] = set()
    for level_ids in list(tree.levels or []):
        for node_id in level_ids:
            node = tree.get_node(str(node_id))
            if node is None or str(node.node_id) in seen:
                continue
            ordered.append(node)
            seen.add(str(node.node_id))
    for node in sorted(tree.nodes.values(), key=lambda n: (int(n.level), _node_span(n), str(n.node_id))):
        if str(node.node_id) not in seen:
            ordered.append(node)
            seen.add(str(node.node_id))
    return ordered


def build_embedding_tree_from_labeled_tree(
    labeled_tree: LabeledTree,
    *,
    embedding_client: Any,
    head: str = "rile",
) -> Tuple[List[EmbeddingTreeNode], Dict[str, Any]]:
    """Build runtime embedding nodes by replaying exact artifact topology."""
    if embedding_client is None:
        raise ValueError("embedding_client is required for exact labeled-tree replay")

    from treepo._research.tree.packed_execution import canonicalize_leaf_embedding

    ordered = _ordered_labeled_nodes_by_level(labeled_tree)
    leaves = [node for node in ordered if int(node.level) == 0]
    leaf_embeddings = embedding_client.embed_texts([str(node.text or "") for node in leaves])
    embedding_by_leaf_id = {
        str(node.node_id): canonicalize_leaf_embedding(embedding)
        for node, embedding in zip(leaves, leaf_embeddings)
    }

    runtime_nodes: List[EmbeddingTreeNode] = []
    runtime_index_by_id: Dict[str, int] = {}
    attached = 0
    leaf_attached = 0
    internal_attached = 0
    missing_children = 0
    for labeled in ordered:
        start, end = _node_span(labeled)
        summary = _summary_target_for_node(labeled, include_identity_targets=False) or ""
        children: Optional[Tuple[int, int]] = None
        embedding = None
        if int(labeled.level) == 0:
            embedding = embedding_by_leaf_id.get(str(labeled.node_id))
        else:
            left_id = str(labeled.left_child_id or "")
            right_id = str(labeled.right_child_id or labeled.left_child_id or "")
            if left_id in runtime_index_by_id and right_id in runtime_index_by_id:
                children = (runtime_index_by_id[left_id], runtime_index_by_id[right_id])
            else:
                missing_children += 1
                raise ValueError(
                    "LabeledTree topology is not replayable: node "
                    f"{labeled.node_id!r} references missing child ids "
                    f"{left_id!r}, {right_id!r}"
                )

        runtime = EmbeddingTreeNode(
            level=int(labeled.level),
            text_span=str(labeled.text or ""),
            char_start=int(start),
            char_end=int(end),
            embedding=embedding,
            children=children,
            oracle_scores={str(head): float(labeled.score)},
            summary=str(summary),
        )
        runtime_nodes.append(runtime)
        runtime_index_by_id[str(labeled.node_id)] = len(runtime_nodes) - 1
        attached += 1
        if runtime.is_leaf:
            leaf_attached += 1
        else:
            internal_attached += 1

    stats = {
        "attached": int(attached),
        "missing": 0,
        "leaf_attached": int(leaf_attached),
        "internal_attached": int(internal_attached),
        "missing_children": int(missing_children),
        "total_runtime_nodes": int(len(runtime_nodes)),
        "total_labeled_nodes": int(len(labeled_tree.nodes)),
        "topology_replay": "exact_artifact_spans",
    }
    return runtime_nodes, stats


@dataclass(frozen=True, kw_only=True)
class DistillationContractConfig:
    train_targets: Tuple[str, ...] = (TRAIN_TARGET_TREE_OPERATOR,)
    student_model_class: str = STUDENT_MODEL_CTREEPO_EMBEDDING_TREE
    supervision_source: str = SUPERVISION_SOURCE_LABELED_TREE_ARTIFACT
    teacher_model_spec: Optional[Dict[str, Any]] = None


@dataclass(frozen=True, kw_only=True)
class SummaryTargetConfig:
    include_identity_targets: bool = False


@dataclass(frozen=True, kw_only=True)
class ScoreTargetConfig:
    include_identity_targets: bool = False
    target_min: float = -100.0
    target_max: float = 100.0


@dataclass(frozen=True, kw_only=True)
class GLMConfig:
    run_trl_sft: bool = False
    model_name: Optional[str] = None
    trl_config: Optional[Any] = None


@dataclass(frozen=True, kw_only=True)
class FEmbeddingConfig:
    method: str = "ridge"
    ridge_lambda: float = 1.0
    epochs: int = 25
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    model_id: str = "ctreepo_f_embedding_proxy"


@dataclass(frozen=True, kw_only=True)
class FLMConfig:
    run_trl_scalar_reward: bool = False
    model_name: Optional[str] = None
    trl_config: Optional[Any] = None


@dataclass(frozen=True, kw_only=True)
class DistillationTrainConfig:
    """Sectioned distillation contract for labeled-tree training."""

    contract: DistillationContractConfig = field(default_factory=DistillationContractConfig)
    run: RunConfig = field(default_factory=RunConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    test: TestConfig = field(default_factory=TestConfig)
    summary_targets: SummaryTargetConfig = field(default_factory=SummaryTargetConfig)
    score_targets: ScoreTargetConfig = field(default_factory=ScoreTargetConfig)
    g_lm: GLMConfig = field(default_factory=GLMConfig)
    f_embedding: FEmbeddingConfig = field(default_factory=FEmbeddingConfig)
    f_lm: FLMConfig = field(default_factory=FLMConfig)


@dataclass
class DistillationFitResult:
    train_count: int
    val_count: int
    test_count: int = 0
    output_dir: Optional[str] = None
    trained_artifact: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    train_targets: Tuple[str, ...] = field(default_factory=tuple)
    student_model_class: str = ""
    supervision_source: str = ""
    teacher_model_spec: Optional[Dict[str, Any]] = None


def _canonical_train_targets(train_targets: Sequence[str] | str) -> Tuple[str, ...]:
    raw_items: List[str]
    if isinstance(train_targets, str):
        raw_items = [part for part in train_targets.replace("+", ",").split(",")]
    else:
        raw_items = []
        for item in train_targets:
            raw_items.extend(str(item).replace("+", ",").split(","))
    targets = tuple(str(item).strip().lower().replace("-", "_") for item in raw_items if str(item).strip())
    if not targets:
        raise ValueError("Distillation train_targets must not be empty")
    allowed = {TRAIN_TARGET_TREE_OPERATOR, TRAIN_TARGET_G, TRAIN_TARGET_F}
    unknown = [target for target in targets if target not in allowed]
    if unknown:
        raise ValueError(f"Unknown distillation train target(s): {unknown!r}")
    return targets


def _canonical_student_model_class(value: str) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "ctreepo": STUDENT_MODEL_CTREEPO_EMBEDDING_TREE,
        "ctreepo_embedding_tree": STUDENT_MODEL_CTREEPO_EMBEDDING_TREE,
        "embedding_tree": STUDENT_MODEL_CTREEPO_EMBEDDING_TREE,
        "lm_sft": STUDENT_MODEL_LM_SFT,
        "sft": STUDENT_MODEL_LM_SFT,
        "embedding_proxy": STUDENT_MODEL_EMBEDDING_RIDGE_PROXY,
        "embedding_ridge_proxy": STUDENT_MODEL_EMBEDDING_RIDGE_PROXY,
        "ridge_proxy": STUDENT_MODEL_EMBEDDING_RIDGE_PROXY,
        "lm_scalar_regression": STUDENT_MODEL_LM_SCALAR_REGRESSION,
        "lm_regression": STUDENT_MODEL_LM_SCALAR_REGRESSION,
        "sequence_classification_regression": STUDENT_MODEL_LM_SCALAR_REGRESSION,
    }
    if raw not in aliases:
        raise ValueError(f"Unknown distillation student_model_class: {value!r}")
    return aliases[raw]


def resolve_distillation_train_config(
    config: Optional[DistillationTrainConfig] = None,
) -> DistillationTrainConfig:
    """Return a canonical explicit distillation contract."""

    cfg = config or DistillationTrainConfig()
    if isinstance(cfg, DistillationTrainConfig):
        contract = DistillationContractConfig(
            train_targets=_canonical_train_targets(cfg.contract.train_targets),
            student_model_class=_canonical_student_model_class(
                cfg.contract.student_model_class
            ),
            supervision_source=str(
                cfg.contract.supervision_source
                or SUPERVISION_SOURCE_LABELED_TREE_ARTIFACT
            ),
            teacher_model_spec=(
                dict(cfg.contract.teacher_model_spec)
                if cfg.contract.teacher_model_spec
                else None
            ),
        )
        return DistillationTrainConfig(
            contract=contract,
            run=cfg.run,
            train=cfg.train,
            validation=cfg.validation,
            test=cfg.test,
            summary_targets=cfg.summary_targets,
            score_targets=cfg.score_targets,
            g_lm=cfg.g_lm,
            f_embedding=cfg.f_embedding,
            f_lm=cfg.f_lm,
        )

    raise TypeError(f"Unsupported distillation config type: {type(cfg).__name__}")


def _distillation_contract_metadata(cfg: DistillationTrainConfig) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "train_targets": list(cfg.contract.train_targets),
        "student_model_class": str(cfg.contract.student_model_class),
        "supervision_source": str(cfg.contract.supervision_source),
        "teacher_model_spec": (
            dict(cfg.contract.teacher_model_spec)
            if cfg.contract.teacher_model_spec
            else None
        ),
    }
    return payload


def _metadata_with_distillation_contract(
    metadata: Mapping[str, Any],
    cfg: DistillationTrainConfig,
) -> Dict[str, Any]:
    contract = _distillation_contract_metadata(cfg)
    merged = dict(metadata)
    merged.update(
        {
            "train_targets": list(cfg.contract.train_targets),
            "student_model_class": str(cfg.contract.student_model_class),
            "supervision_source": str(cfg.contract.supervision_source),
            "teacher_model_spec": (
                dict(cfg.contract.teacher_model_spec)
                if cfg.contract.teacher_model_spec
                else None
            ),
            "distillation_contract": contract,
            "run": config_to_dict(cfg.run),
            "train": config_to_dict(cfg.train),
            "validation": config_to_dict(cfg.validation),
            "test": config_to_dict(cfg.test),
        }
    )
    return merged


def _fit_result(
    cfg: DistillationTrainConfig,
    *,
    train_count: int,
    val_count: int,
    test_count: int = 0,
    trained_artifact: Optional[Any] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> DistillationFitResult:
    return DistillationFitResult(
        train_count=int(train_count),
        val_count=int(val_count),
        test_count=int(test_count),
        output_dir=str(cfg.run.output_dir) if cfg.run.output_dir else None,
        trained_artifact=trained_artifact,
        metadata=_metadata_with_distillation_contract(metadata or {}, cfg),
        train_targets=tuple(cfg.contract.train_targets),
        student_model_class=str(cfg.contract.student_model_class),
        supervision_source=str(cfg.contract.supervision_source),
        teacher_model_spec=(
            dict(cfg.contract.teacher_model_spec)
            if cfg.contract.teacher_model_spec
            else None
        ),
    )


def split_labeled_trees(
    trees: Sequence[LabeledTree],
    *,
    train_splits: Sequence[str] = ("train",),
    val_splits: Sequence[str] = ("val",),
) -> Tuple[List[LabeledTree], List[LabeledTree]]:
    train_keys = {str(value).lower() for value in train_splits}
    val_keys = {str(value).lower() for value in val_splits}
    train: List[LabeledTree] = []
    val: List[LabeledTree] = []
    for tree in trees:
        split = str((tree.metadata or {}).get("split", "") or "").lower()
        if split in train_keys:
            train.append(tree)
        elif split in val_keys:
            val.append(tree)
    if not train and trees:
        train = list(trees)
    return train, val


def _select_labeled_trees_by_splits(
    trees: Sequence[LabeledTree],
    splits: Sequence[str],
) -> List[LabeledTree]:
    keys = {str(value).lower() for value in splits}
    selected: List[LabeledTree] = []
    for tree in trees:
        split = str((tree.metadata or {}).get("split", "") or "").lower()
        if split in keys:
            selected.append(tree)
    return selected


def _summary_target_for_node(
    node: LabeledNode,
    *,
    include_identity_targets: bool = False,
) -> Optional[str]:
    metadata = dict(node.metadata or {})
    for key in (
        "teacher_summary",
        "teacher_leaf_summary",
        "teacher_merge_summary",
        "target_summary",
        "summary",
    ):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    if include_identity_targets and str(node.text or "").strip():
        return str(node.text)
    return None


def _root_labeled_node(tree: LabeledTree) -> Optional[LabeledNode]:
    levels = list(tree.levels or [])
    for level_ids in reversed(levels):
        for node_id in level_ids:
            node = tree.get_node(str(node_id))
            if node is not None:
                return node
    if not tree.nodes:
        return None
    return max(tree.nodes.values(), key=lambda node: (int(node.level), str(node.node_id)))


def _stored_summary_anchor_text(tree: LabeledTree) -> Optional[str]:
    metadata = dict(tree.metadata or {})
    for key in (
        "stored_summary",
        "source_summary",
        "existing_summary",
        "full_doc_summary",
    ):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    bundle = normalize_tree_bundle_manifest(metadata)
    source_kind = str(bundle.get("source_kind") or "").strip().lower()
    tree_bundle_kind = str(metadata.get("tree_bundle_kind") or "").strip().lower()
    tree_text_source = str(metadata.get("tree_text_source") or "").strip().lower()
    if (
        source_kind == SOURCE_KIND_EXTERNAL_STATE
        or tree_bundle_kind == "external_summary_token_tree"
        or (tree_text_source and tree_text_source != "aligned_text")
    ) and str(tree.document_text or "").strip():
        return str(tree.document_text).strip()
    root = _root_labeled_node(tree)
    if root is not None:
        root_summary = _summary_target_for_node(root, include_identity_targets=False)
        if root_summary:
            return root_summary
    if str(tree.document_text or "").strip():
        return str(tree.document_text).strip()
    return None


def _raw_document_anchor_text(tree: LabeledTree) -> Optional[str]:
    metadata = dict(tree.metadata or {})
    for key in (
        "raw_document_text",
        "aligned_text",
        "full_document_text",
        "manifesto_text",
    ):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    bundle = normalize_tree_bundle_manifest(metadata)
    source_kind = str(bundle.get("source_kind") or "").strip().lower()
    tree_bundle_kind = str(metadata.get("tree_bundle_kind") or "").strip().lower()
    tree_text_source = str(metadata.get("tree_text_source") or "").strip().lower()
    summary_representation = str(metadata.get("summary_representation") or "").strip().lower()
    if (
        source_kind == SOURCE_KIND_RAW_INPUT
        or tree_bundle_kind == "raw_manifesto_token_tree"
        or tree_text_source == "aligned_text"
        or summary_representation == "raw_whole_document"
    ):
        if str(tree.document_text or "").strip():
            return str(tree.document_text).strip()
    return None


def _full_doc_anchor_target(
    tree: LabeledTree,
    *,
    target_source: str,
    prefer_native_expert: bool = False,
) -> Tuple[Optional[float], Optional[str]]:
    metadata = dict(tree.metadata or {})
    target = _normalise_anchor_target(target_source)
    if target == FULL_DOC_ANCHOR_TARGET_EXPERT:
        native_keys = (
            "expert_score_native",
            "expert_target_native",
            "benoit_expert_mean_raw",
            "expert_score_raw_benoit",
            "expert_score_for_objective",
            "benoit_expert_mean",
        )
        legacy_keys = ("expert_score_1_7", "expert_score_for_objective", "expert_score", "benoit_expert_mean")
        keys = native_keys + legacy_keys if prefer_native_expert else legacy_keys + native_keys
    else:
        keys = (
            "teacher_score_1_7_existing_root",
            "teacher_score_1_7",
            "llm_score_1_7",
            "document_score",
        )
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            parsed = _safe_float(value, default=float("nan"))
            if math.isfinite(parsed):
                return float(parsed), key
    if target == FULL_DOC_ANCHOR_TARGET_TEACHER:
        root = _root_labeled_node(tree)
        if root is not None and root.score is not None:
            parsed = _safe_float(root.score, default=float("nan"))
            if math.isfinite(parsed):
                return float(parsed), "root_node.score"
        parsed = _safe_float(getattr(tree, "document_score", None), default=float("nan"))
        if math.isfinite(parsed):
            return float(parsed), "tree.document_score"
    return None, None


def _score_bounds_for_target(
    *,
    anchor_target: str,
    target_min: float,
    target_max: float,
    scorer_output_min: float,
    scorer_output_max: float,
) -> Tuple[float, float, str]:
    if _normalise_anchor_target(anchor_target) == FULL_DOC_ANCHOR_TARGET_EXPERT:
        return float(target_min), float(target_max), "expert_target"
    return float(scorer_output_min), float(scorer_output_max), "scorer_output"


def _full_doc_anchor_sources(
    tree: LabeledTree,
    *,
    mode: str,
) -> List[Tuple[str, str]]:
    normalized = _normalise_anchor_mode(mode)
    if normalized == FULL_DOC_ANCHOR_OFF:
        return []
    sources: List[Tuple[str, str]] = []
    if normalized in {FULL_DOC_ANCHOR_STORED_SUMMARY, FULL_DOC_ANCHOR_BOTH}:
        text = _stored_summary_anchor_text(tree)
        if text:
            sources.append((FULL_DOC_ANCHOR_STORED_SUMMARY, text))
    if normalized in {FULL_DOC_ANCHOR_RAW_DOCUMENT, FULL_DOC_ANCHOR_BOTH}:
        text = _raw_document_anchor_text(tree)
        if text:
            sources.append((FULL_DOC_ANCHOR_RAW_DOCUMENT, text))
    return sources


def _record_weight_metadata(
    metadata: Dict[str, Any],
    *,
    weight: float,
) -> Dict[str, Any]:
    out = dict(metadata)
    out["example_weight"] = float(weight)
    return out


def _objective_weight_metadata(
    *,
    root_share: float,
    local_law_weight: float,
) -> Dict[str, Any]:
    return {
        "root_share": float(root_share),
        "local_law_weight": float(local_law_weight),
        "local_law_component_weights": {
            "teacher_node": float(local_law_weight),
        },
    }


def build_g_sft_records(
    labeled_trees: Sequence[LabeledTree],
    *,
    include_identity_targets: bool = False,
    target_min: float = -100.0,
    target_max: float = 100.0,
    scorer_output_min: Optional[float] = None,
    scorer_output_max: Optional[float] = None,
    root_label_sources: Optional[Sequence[str] | str] = None,
    root_label_target: str = ROOT_LABEL_TARGET_EXPERT,
    local_law_weight: Optional[float] = None,
    node_weight_normalization: str = NODE_WEIGHT_NORMALIZATION_PER_TREE,
    **legacy_anchor_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Build TRL SFT records for the generative ``g`` target.

    Records are ``{"prompt": str, "completion": str}`` rows accepted by
    ``src.training.trl_training.train_sft``.  The builder consumes only cached
    artifact fields; it never calls a teacher.  Nodes without teacher text
    targets are skipped unless ``include_identity_targets`` is enabled.
    """
    _reject_legacy_anchor_kwargs(legacy_anchor_kwargs)
    records: List[Dict[str, Any]] = []
    anchor_mode = _anchor_mode_from_root_label_sources(root_label_sources)
    anchor_target = _normalise_root_label_target(root_label_target)
    node_weight_norm = _normalise_node_weight_normalization(node_weight_normalization)
    scorer_min = float(target_min if scorer_output_min is None else scorer_output_min)
    scorer_max = float(target_max if scorer_output_max is None else scorer_output_max)
    anchor_weight = _root_anchor_weight(
        full_doc_anchor_mode=anchor_mode,
        local_law_weight=local_law_weight,
    )
    _, teacher_local_law_weight = _objective_masses(
        full_doc_anchor_mode=anchor_mode,
        local_law_weight=local_law_weight,
    )
    objective_weight_metadata = _objective_weight_metadata(
        root_share=float(anchor_weight),
        local_law_weight=float(teacher_local_law_weight),
    )
    prefer_native_expert = not (
        math.isclose(float(target_min), 1.0) and math.isclose(float(target_max), 7.0)
    )
    for tree in labeled_trees:
        split = str((tree.metadata or {}).get("split", "") or "")
        tree_records: List[Dict[str, Any]] = []
        for node in tree.nodes.values():
            target = _summary_target_for_node(
                node,
                include_identity_targets=bool(include_identity_targets),
            )
            if not target:
                continue
            metadata = {
                "doc_id": tree.doc_id,
                "node_id": node.node_id,
                "split": split,
                "level": int(node.level),
                "is_leaf": bool((node.metadata or {}).get("is_leaf", node.level == 0)),
                "law_role": "leaf_g" if int(node.level) == 0 else "merge_g",
                "target_score_raw": float(node.score),
                "target_score_normalized": _normalize_score(
                    float(node.score),
                    target_min=scorer_min,
                    target_max=scorer_max,
                ),
                "target_score_scale": "scorer_output",
                "target_min": scorer_min,
                "target_max": scorer_max,
                "scorer_output_min": scorer_min,
                "scorer_output_max": scorer_max,
                "target_source": "teacher_node_score",
                "label_source": str(
                    (node.metadata or {}).get("label_source")
                    or tree.label_source
                    or (tree.metadata or {}).get("label_source", "")
                ),
            }
            if int(node.level) == 0:
                prompt = (
                    "Summarize the following leaf span for score-preserving "
                    "C-TreePO distillation.\n\nLEAF:\n"
                    f"{node.text}"
                )
            else:
                left = tree.get_node(str(node.left_child_id)) if node.left_child_id else None
                right = tree.get_node(str(node.right_child_id)) if node.right_child_id else None
                left_text = _summary_target_for_node(
                    left,
                    include_identity_targets=True,
                ) if left is not None else ""
                right_text = _summary_target_for_node(
                    right,
                    include_identity_targets=True,
                ) if right is not None else ""
                prompt = (
                    "Merge the two child summaries into one score-preserving "
                    "C-TreePO parent summary.\n\nLEFT:\n"
                    f"{left_text}\n\nRIGHT:\n{right_text}"
                )
            tree_records.append(
                {
                    "prompt": prompt,
                    "completion": target,
                    "metadata": metadata,
                }
            )

        for pair_index, pair in enumerate((tree.metadata or {}).get("idempotence_pairs", []) or []):
            if not isinstance(pair, Mapping):
                continue
            source_summary = str(pair.get("input_summary", "") or "").strip()
            target_summary = str(pair.get("target_resummary", "") or "").strip()
            if not source_summary or not target_summary:
                continue
            tree_records.append(
                {
                    "prompt": (
                        "Resummarize the following summary while preserving the "
                        "same directional score-relevant content.\n\nSUMMARY:\n"
                        f"{source_summary}"
                    ),
                    "completion": target_summary,
                    "metadata": {
                        "doc_id": tree.doc_id,
                        "node_id": f"idempotence_pair_{pair_index}",
                        "split": split,
                        "level": None,
                        "is_leaf": False,
                        "law_role": "idempotence_proxy",
                        "target_source": "teacher_idempotence_pair",
                        "label_source": str(
                            tree.label_source or (tree.metadata or {}).get("label_source", "")
                        ),
                    },
                }
            )
        node_weight = _node_record_weight(
            n_teacher_records_for_tree=len(tree_records),
            full_doc_anchor_mode=anchor_mode,
            local_law_weight=local_law_weight,
            node_weight_normalization=node_weight_norm,
        )
        for row in tree_records:
            row["weight"] = float(node_weight)
            metadata = dict(row.get("metadata") or {})
            metadata.update(objective_weight_metadata)
            row["metadata"] = _record_weight_metadata(metadata, weight=float(node_weight))
            if node_weight > 0.0:
                records.append(row)

        target_raw, target_key = _full_doc_anchor_target(
            tree,
            target_source=anchor_target,
            prefer_native_expert=prefer_native_expert,
        )
        if target_raw is None:
            continue
        if anchor_weight <= 0.0:
            continue
        anchor_min, anchor_max, anchor_scale = _score_bounds_for_target(
            anchor_target=anchor_target,
            target_min=float(target_min),
            target_max=float(target_max),
            scorer_output_min=scorer_min,
            scorer_output_max=scorer_max,
        )
        target_normalized = _normalize_score(
            float(target_raw),
            target_min=anchor_min,
            target_max=anchor_max,
        )
        stored_summary = _stored_summary_anchor_text(tree)
        anchor_sources: List[Tuple[str, str, str]] = []
        for source_name, source_text in _full_doc_anchor_sources(tree, mode=anchor_mode):
            if source_name == FULL_DOC_ANCHOR_RAW_DOCUMENT and not stored_summary:
                continue
            completion = stored_summary if source_name == FULL_DOC_ANCHOR_RAW_DOCUMENT else source_text
            if not completion or not str(source_text).strip():
                continue
            anchor_sources.append((source_name, str(source_text), str(completion)))
        if not anchor_sources:
            continue
        anchor_row_weight = float(anchor_weight)
        for source_name, source_text, completion in anchor_sources:
            anchor_node_id = f"full_doc_anchor_{source_name}"
            metadata = {
                "doc_id": tree.doc_id,
                "node_id": anchor_node_id,
                "split": split,
                "level": None,
                "is_leaf": False,
                "law_role": "full_doc_g_anchor",
                "anchor_text_source": source_name,
                "target_score_raw": float(target_raw),
                "target_score_normalized": float(target_normalized),
                "target_score_scale": anchor_scale,
                **objective_weight_metadata,
                "target_min": float(anchor_min),
                "target_max": float(anchor_max),
                "scorer_output_min": scorer_min,
                "scorer_output_max": scorer_max,
                "target_source": f"{anchor_target}:{target_key}",
                "observed_target": anchor_target == FULL_DOC_ANCHOR_TARGET_EXPERT,
                "label_source": str(
                    tree.label_source or (tree.metadata or {}).get("label_source", "")
                ),
                "example_weight": float(anchor_row_weight),
            }
            records.append(
                {
                    "prompt": (
                        "Summarize the following full document input for score-preserving "
                        "C-TreePO distillation.\n\nFULL_DOC_INPUT:\n"
                        f"{source_text}"
                    ),
                    "completion": str(completion),
                    "weight": float(anchor_row_weight),
                    "metadata": metadata,
                }
            )
    return records


def build_f_embedding_examples(
    labeled_trees: Sequence[LabeledTree],
    *,
    include_identity_targets: bool = False,
    target_min: float = -100.0,
    target_max: float = 100.0,
) -> List[Any]:
    """Build scalar ``f`` examples from cached node summaries.

    The returned objects are ``LabeledEmbeddingExample`` rows accepted by the
    existing embedding-proxy trainers.  Targets are normalized to ``[0, 1]`` so
    RILE's raw ``[-100, 100]`` scale fits the proxy contract.
    """
    from treepo._research.training.embedding_proxy import LabeledEmbeddingExample

    examples: List[LabeledEmbeddingExample] = []
    for tree in labeled_trees:
        for node in tree.nodes.values():
            target_text = _summary_target_for_node(
                node,
                include_identity_targets=bool(include_identity_targets),
            )
            if not target_text:
                continue
            source = str(
                (node.metadata or {}).get("label_source")
                or tree.label_source
                or (tree.metadata or {}).get("label_source", "")
                or "unknown"
            )
            examples.append(
                LabeledEmbeddingExample(
                    doc_id=f"{tree.doc_id}:{node.node_id}",
                    text=str(target_text),
                    target_score=_normalize_score(
                        float(node.score),
                        target_min=float(target_min),
                        target_max=float(target_max),
                    ),
                    truth_label_source=source,
                )
            )
    return examples


def build_f_lm_regression_records(
    labeled_trees: Sequence[LabeledTree],
    *,
    include_identity_targets: bool = False,
    target_min: float = -100.0,
    target_max: float = 100.0,
    scorer_output_min: Optional[float] = None,
    scorer_output_max: Optional[float] = None,
    root_label_sources: Optional[Sequence[str] | str] = None,
    root_label_target: str = ROOT_LABEL_TARGET_EXPERT,
    local_law_weight: Optional[float] = None,
    node_weight_normalization: str = NODE_WEIGHT_NORMALIZATION_PER_TREE,
    **legacy_anchor_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Build scalar-regression rows for a small LM ``f`` target."""
    _reject_legacy_anchor_kwargs(legacy_anchor_kwargs)
    records: List[Dict[str, Any]] = []
    anchor_mode = _anchor_mode_from_root_label_sources(root_label_sources)
    anchor_target = _normalise_root_label_target(root_label_target)
    node_weight_norm = _normalise_node_weight_normalization(node_weight_normalization)
    scorer_min = float(target_min if scorer_output_min is None else scorer_output_min)
    scorer_max = float(target_max if scorer_output_max is None else scorer_output_max)
    anchor_weight = _root_anchor_weight(
        full_doc_anchor_mode=anchor_mode,
        local_law_weight=local_law_weight,
    )
    _, teacher_local_law_weight = _objective_masses(
        full_doc_anchor_mode=anchor_mode,
        local_law_weight=local_law_weight,
    )
    objective_weight_metadata = _objective_weight_metadata(
        root_share=float(anchor_weight),
        local_law_weight=float(teacher_local_law_weight),
    )
    prefer_native_expert = not (
        math.isclose(float(target_min), 1.0) and math.isclose(float(target_max), 7.0)
    )
    for tree in labeled_trees:
        split = str((tree.metadata or {}).get("split", "") or "")
        tree_records: List[Dict[str, Any]] = []
        for node in tree.nodes.values():
            target_text = _summary_target_for_node(
                node,
                include_identity_targets=bool(include_identity_targets),
            )
            if not target_text:
                continue
            law_role = "leaf_f" if int(node.level) == 0 else "merge_f"
            normalized = _normalize_score(
                float(node.score),
                target_min=scorer_min,
                target_max=scorer_max,
            )
            tree_records.append(
                {
                    "prompt": (
                        "Predict the normalized scalar RILE score in [0, 1] "
                        "for this C-TreePO node summary."
                    ),
                    "response": str(target_text),
                    "score": float(normalized),
                    "metadata": {
                        "doc_id": tree.doc_id,
                        "node_id": node.node_id,
                        "split": split,
                        "level": int(node.level),
                        "law_role": law_role,
                        "target_score_raw": float(node.score),
                        "target_score_normalized": float(normalized),
                        "target_score_scale": "scorer_output",
                        "target_min": scorer_min,
                        "target_max": scorer_max,
                        "scorer_output_min": scorer_min,
                        "scorer_output_max": scorer_max,
                        "target_source": "teacher_node_score",
                        "paper_to_lean_local_law_mapping": dict(PAPER_TO_LEAN_LOCAL_LAW_MAPPING),
                        "label_source": str(
                            (node.metadata or {}).get("label_source")
                            or tree.label_source
                            or (tree.metadata or {}).get("label_source", "")
                        ),
                    },
                }
            )
        node_weight = _node_record_weight(
            n_teacher_records_for_tree=len(tree_records),
            full_doc_anchor_mode=anchor_mode,
            local_law_weight=local_law_weight,
            node_weight_normalization=node_weight_norm,
        )
        for row in tree_records:
            row["weight"] = float(node_weight)
            metadata = dict(row.get("metadata") or {})
            metadata.update(objective_weight_metadata)
            row["metadata"] = _record_weight_metadata(metadata, weight=float(node_weight))
            if node_weight > 0.0:
                records.append(row)

        target_raw, target_key = _full_doc_anchor_target(
            tree,
            target_source=anchor_target,
            prefer_native_expert=prefer_native_expert,
        )
        if target_raw is None:
            continue
        if anchor_weight <= 0.0:
            continue
        anchor_min, anchor_max, anchor_scale = _score_bounds_for_target(
            anchor_target=anchor_target,
            target_min=float(target_min),
            target_max=float(target_max),
            scorer_output_min=scorer_min,
            scorer_output_max=scorer_max,
        )
        target_normalized = _normalize_score(
            float(target_raw),
            target_min=anchor_min,
            target_max=anchor_max,
        )
        anchor_sources = [
            (source_name, str(source_text))
            for source_name, source_text in _full_doc_anchor_sources(tree, mode=anchor_mode)
            if str(source_text or "").strip()
        ]
        if not anchor_sources:
            continue
        anchor_row_weight = float(anchor_weight)
        for source_name, source_text in anchor_sources:
            if not str(source_text or "").strip():
                continue
            metadata = {
                "doc_id": tree.doc_id,
                "node_id": f"full_doc_anchor_{source_name}",
                "split": split,
                "level": None,
                "law_role": "full_doc_f_anchor",
                "anchor_text_source": source_name,
                "target_score_raw": float(target_raw),
                "target_score_normalized": float(target_normalized),
                "target_score_scale": anchor_scale,
                **objective_weight_metadata,
                "target_min": float(anchor_min),
                "target_max": float(anchor_max),
                "scorer_output_min": scorer_min,
                "scorer_output_max": scorer_max,
                "target_source": f"{anchor_target}:{target_key}",
                "observed_target": anchor_target == FULL_DOC_ANCHOR_TARGET_EXPERT,
                "paper_to_lean_local_law_mapping": dict(PAPER_TO_LEAN_LOCAL_LAW_MAPPING),
                "label_source": str(
                    tree.label_source or (tree.metadata or {}).get("label_source", "")
                ),
                "example_weight": float(anchor_row_weight),
            }
            records.append(
                {
                    "prompt": (
                        "Predict the normalized scalar RILE score in [0, 1] "
                        "for this full-document anchor text."
                    ),
                    "response": str(source_text),
                    "score": float(target_normalized),
                    "weight": float(anchor_row_weight),
                    "metadata": metadata,
                }
            )
    return records


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def evaluate_labeled_tree_local_laws(
    labeled_trees: Sequence[LabeledTree],
    *,
    score_fn: Optional[Callable[[str], float]] = None,
    include_identity_targets: bool = False,
    score_outputs_normalized: bool = True,
    target_min: float = -100.0,
    target_max: float = 100.0,
) -> Dict[str, Any]:
    """Report Lean-mapped C1/C2/C3 metrics over artifact-aligned nodes."""
    c1_errors: List[float] = []
    c3_errors: List[float] = []
    c2_score_drifts: List[float] = []
    c2_text_distances: List[float] = []
    c1_count = 0
    c3_count = 0
    missing_summary = 0

    def _score(text: str) -> Optional[float]:
        if score_fn is None:
            return None
        value = float(score_fn(text))
        if score_outputs_normalized:
            return _clamp01(value)
        return _normalize_score(value, target_min=float(target_min), target_max=float(target_max))

    for tree in labeled_trees:
        for node in tree.nodes.values():
            target_text = _summary_target_for_node(
                node,
                include_identity_targets=bool(include_identity_targets),
            )
            if not target_text:
                missing_summary += 1
                continue
            target = _normalize_score(
                float(node.score),
                target_min=float(target_min),
                target_max=float(target_max),
            )
            pred = _score(str(target_text))
            if int(node.level) == 0:
                c1_count += 1
                if pred is not None:
                    c1_errors.append(abs(float(pred) - float(target)))
            else:
                c3_count += 1
                if pred is not None:
                    c3_errors.append(abs(float(pred) - float(target)))

        for pair in (tree.metadata or {}).get("idempotence_pairs", []) or []:
            if not isinstance(pair, Mapping):
                continue
            source = str(pair.get("input_summary", "") or "").strip()
            target = str(pair.get("target_resummary", "") or "").strip()
            if not source or not target:
                continue
            c2_text_distances.append(1.0 - SequenceMatcher(None, source, target).ratio())
            source_score = _score(source)
            target_score = _score(target)
            if source_score is not None and target_score is not None:
                c2_score_drifts.append(abs(float(source_score) - float(target_score)))

    return {
        "paper_to_lean_local_law_mapping": dict(PAPER_TO_LEAN_LOCAL_LAW_MAPPING),
        "score_outputs_normalized": bool(score_outputs_normalized),
        "missing_summary_targets": int(missing_summary),
        "C1": {
            "lean_law": PAPER_TO_LEAN_LOCAL_LAW_MAPPING["C1"],
            "count": int(c1_count),
            "mae_normalized": _mean(c1_errors),
            "scored_count": int(len(c1_errors)),
        },
        "C2": {
            "lean_law": PAPER_TO_LEAN_LOCAL_LAW_MAPPING["C2"],
            "idempotence_pairs": int(len(c2_text_distances)),
            "theorem_domain_text_pair_available": bool(c2_text_distances),
            "proxy_score_drift_available": bool(c2_score_drifts),
            "mean_text_distance": _mean(c2_text_distances),
            "mean_proxy_score_drift_normalized": _mean(c2_score_drifts),
            "metric_kind": "teacher_resummary_text_distance_plus_f_score_drift",
        },
        "C3": {
            "lean_law": PAPER_TO_LEAN_LOCAL_LAW_MAPPING["C3"],
            "count": int(c3_count),
            "mae_normalized": _mean(c3_errors),
            "scored_count": int(len(c3_errors)),
        },
    }


def _embedding_proxy_eval_metrics(
    model: Any,
    examples: Sequence[Any],
    *,
    embedding_client: Any,
    target_min: float = -100.0,
    target_max: float = 100.0,
) -> Dict[str, Any]:
    if not examples:
        return {"count": 0}
    embeddings = embedding_client.embed_texts([str(ex.text) for ex in examples])
    preds = [float(model.predict_from_embedding(vec)) for vec in embeddings]
    targets = [float(ex.target_score) for ex in examples]
    abs_errors = [abs(pred - target) for pred, target in zip(preds, targets)]
    span = float(target_max) - float(target_min)
    mean_norm = sum(abs_errors) / float(len(abs_errors)) if abs_errors else 0.0
    return {
        "count": int(len(examples)),
        "mae_normalized": float(mean_norm),
        "mae_raw": float(mean_norm * span),
        "max_abs_error_normalized": float(max(abs_errors) if abs_errors else 0.0),
    }


def fit_f_embedding_proxy(
    labeled_trees: Sequence[LabeledTree],
    *,
    embedding_client: Any,
    output_path: Optional[Path] = None,
    train_splits: Sequence[str] = ("train",),
    val_splits: Sequence[str] = ("val",),
    include_identity_targets: bool = False,
    method: str = "ridge",
    ridge_lambda: float = 1.0,
    epochs: int = 25,
    learning_rate: float = 5e-3,
    weight_decay: float = 1e-4,
    model_id: str = "ctreepo_f_embedding_proxy",
    target_min: float = -100.0,
    target_max: float = 100.0,
) -> DistillationFitResult:
    """Fit the scalar ``f`` target from labeled-tree summary embeddings."""
    train, val = split_labeled_trees(
        labeled_trees,
        train_splits=train_splits,
        val_splits=val_splits,
    )
    train_examples = build_f_embedding_examples(
        train,
        include_identity_targets=bool(include_identity_targets),
        target_min=float(target_min),
        target_max=float(target_max),
    )
    val_examples = build_f_embedding_examples(
        val,
        include_identity_targets=bool(include_identity_targets),
        target_min=float(target_min),
        target_max=float(target_max),
    )
    if not train_examples:
        raise ValueError("No f-target embedding examples were available")

    method_key = str(method or "ridge").strip().lower()
    if method_key == "ridge":
        from treepo._research.training.embedding_proxy import fit_embedding_ridge_proxy

        model = fit_embedding_ridge_proxy(
            train_examples,
            embedding_client=embedding_client,
            ridge_lambda=float(ridge_lambda),
            model_id=str(model_id),
        )
    elif method_key in {"linear_sgd", "sgd", "linear"}:
        from treepo._research.training.embedding_proxy import fit_embedding_linear_sgd_proxy

        model = fit_embedding_linear_sgd_proxy(
            train_examples,
            embedding_client=embedding_client,
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            model_id=str(model_id),
        )
    else:
        raise ValueError("Unsupported f-target embedding method: %r" % method)

    saved_path: Optional[str] = None
    if output_path is not None:
        saved_path = str(model.save_json(Path(output_path)))

    metadata = {
        "method": method_key,
        "train_examples": int(len(train_examples)),
        "val_examples": int(len(val_examples)),
        "include_identity_targets": bool(include_identity_targets),
        "model_path": saved_path,
        "train_metrics": _embedding_proxy_eval_metrics(
            model,
            train_examples,
            embedding_client=embedding_client,
            target_min=float(target_min),
            target_max=float(target_max),
        ),
        "val_metrics": _embedding_proxy_eval_metrics(
            model,
            val_examples,
            embedding_client=embedding_client,
            target_min=float(target_min),
            target_max=float(target_max),
        ),
    }
    default_cfg = DistillationTrainConfig(
        contract=DistillationContractConfig(
            train_targets=(TRAIN_TARGET_F,),
            student_model_class=STUDENT_MODEL_EMBEDDING_RIDGE_PROXY,
            supervision_source=SUPERVISION_SOURCE_LABELED_TREE_ARTIFACT,
        ),
        run=RunConfig(
            output_dir=Path(output_path).parent if output_path is not None else None
        ),
        train=TrainConfig(train_splits=tuple(train_splits)),
        validation=ValidationConfig(val_splits=tuple(val_splits)),
        score_targets=ScoreTargetConfig(
            include_identity_targets=bool(include_identity_targets),
            target_min=float(target_min),
            target_max=float(target_max),
        ),
        f_embedding=FEmbeddingConfig(
            method=method_key,
            ridge_lambda=float(ridge_lambda),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            model_id=str(model_id),
        ),
    )
    return DistillationFitResult(
        train_count=len(train),
        val_count=len(val),
        test_count=0,
        output_dir=str(Path(output_path).parent) if output_path is not None else None,
        trained_artifact=model,
        metadata=_metadata_with_distillation_contract(metadata, default_cfg),
        train_targets=(TRAIN_TARGET_F,),
        student_model_class=STUDENT_MODEL_EMBEDDING_RIDGE_PROXY,
        supervision_source=SUPERVISION_SOURCE_LABELED_TREE_ARTIFACT,
    )


def fit(
    labeled_trees: Sequence[LabeledTree],
    config: Optional[DistillationTrainConfig] = None,
    *,
    embedding_client: Any = None,
    trainer: Any = None,
) -> DistillationFitResult:
    """Fit a target/model-class distillation contract from labeled trees.

    Routing is explicit: callers choose ``train_targets`` and
    ``student_model_class`` through ``DistillationTrainConfig``.
    """
    cfg = resolve_distillation_train_config(config)
    train, val = split_labeled_trees(
        labeled_trees,
        train_splits=cfg.train.train_splits,
        val_splits=cfg.validation.val_splits if cfg.validation.enabled else (),
    )
    test = (
        _select_labeled_trees_by_splits(labeled_trees, cfg.test.test_splits)
        if cfg.test.enabled
        else []
    )

    if cfg.run.dry_run:
        return _fit_result(
            cfg,
            train_count=len(train),
            val_count=len(val),
            test_count=len(test),
            metadata={"dry_run": True},
        )

    route = (tuple(cfg.contract.train_targets), str(cfg.contract.student_model_class))
    if route == ((TRAIN_TARGET_TREE_OPERATOR,), STUDENT_MODEL_CTREEPO_EMBEDDING_TREE):
        if trainer is None:
            raise ValueError("tree_operator fit requires an initialized CTreePOTrainer")
        if embedding_client is not None:
            trainer.embedding_client = embedding_client
        trainer.prepare_trees_from_labeled_trees(train, split="train")
        if val:
            trainer.prepare_trees_from_labeled_trees(val, split="val")
        trained = trainer.train(output_dir=cfg.run.output_dir)
        return _fit_result(
            cfg,
            train_count=len(train),
            val_count=len(val),
            test_count=len(test),
            trained_artifact=trained,
            metadata={"trainer": type(trainer).__name__},
        )

    if route == ((TRAIN_TARGET_G,), STUDENT_MODEL_LM_SFT):
        include_identity_targets = bool(cfg.summary_targets.include_identity_targets)
        train_sft_records = build_g_sft_records(
            train,
            include_identity_targets=include_identity_targets,
        )
        val_sft_records = build_g_sft_records(
            val,
            include_identity_targets=include_identity_targets,
        )
        test_sft_records = build_g_sft_records(
            test,
            include_identity_targets=include_identity_targets,
        )
        metadata: Dict[str, Any] = {
            "dataset_ready": True,
            "sft_train_records": len(train_sft_records),
            "sft_val_records": len(val_sft_records),
            "sft_test_records": len(test_sft_records),
            "include_identity_targets": include_identity_targets,
            "paper_to_lean_local_law_mapping": dict(PAPER_TO_LEAN_LOCAL_LAW_MAPPING),
            "local_law_eval": evaluate_labeled_tree_local_laws(
                val or train,
                include_identity_targets=include_identity_targets,
            ),
            "note": "Use TRL SFT/DPO trainers with datasets derived from these labeled trees.",
        }
        trained: Optional[Any] = None
        if cfg.test.enabled:
            metadata["test_local_law_eval"] = evaluate_labeled_tree_local_laws(
                test,
                include_identity_targets=include_identity_targets,
            )
        if cfg.run.output_dir is not None:
            output_dir = Path(cfg.run.output_dir)
            train_path = write_jsonl_records(
                output_dir / "g_sft_train.jsonl",
                train_sft_records,
            )
            val_path = write_jsonl_records(
                output_dir / "g_sft_val.jsonl",
                val_sft_records,
            )
            test_path = write_jsonl_records(
                output_dir / "g_sft_test.jsonl",
                test_sft_records,
            )
            metadata["sft_train_path"] = str(train_path)
            metadata["sft_val_path"] = str(val_path)
            metadata["sft_test_path"] = str(test_path)

        if bool(cfg.g_lm.run_trl_sft):
            model_name = cfg.g_lm.model_name
            if not model_name:
                raise ValueError("g-target TRL SFT requires g_lm.model_name")
            if not train_sft_records:
                raise ValueError("g-target TRL SFT requested but no SFT records were available")
            from treepo._research.training.trl_training import TRLTrainingConfig, train_sft

            trl_config = cfg.g_lm.trl_config or TRLTrainingConfig()
            trained = train_sft(
                records=train_sft_records,
                eval_records=val_sft_records or None,
                model_name=str(model_name),
                output_dir=Path(cfg.run.output_dir or "outputs/g_sft"),
                config=trl_config,
            )
            metadata["trl_sft_ran"] = True
        return _fit_result(
            cfg,
            train_count=len(train),
            val_count=len(val),
            test_count=len(test),
            trained_artifact=trained,
            metadata=metadata,
        )

    if route == ((TRAIN_TARGET_F,), STUDENT_MODEL_EMBEDDING_RIDGE_PROXY):
        if embedding_client is None:
            raise ValueError("f embedding proxy fit requires an embedding_client")
        output_path = None
        if cfg.run.output_dir is not None:
            output_path = Path(cfg.run.output_dir) / "f_embedding_proxy.json"
        result = fit_f_embedding_proxy(
            labeled_trees,
            embedding_client=embedding_client,
            output_path=output_path,
            train_splits=cfg.train.train_splits,
            val_splits=cfg.validation.val_splits if cfg.validation.enabled else (),
            include_identity_targets=bool(cfg.score_targets.include_identity_targets),
            method=str(cfg.f_embedding.method),
            ridge_lambda=float(cfg.f_embedding.ridge_lambda),
            epochs=int(cfg.f_embedding.epochs),
            learning_rate=float(cfg.f_embedding.learning_rate),
            weight_decay=float(cfg.f_embedding.weight_decay),
            model_id=str(cfg.f_embedding.model_id),
            target_min=float(cfg.score_targets.target_min),
            target_max=float(cfg.score_targets.target_max),
        )
        target_min = float(cfg.score_targets.target_min)
        target_max = float(cfg.score_targets.target_max)

        def _proxy_score(text: str) -> float:
            embedding = embedding_client.embed_texts([text])[0]
            return float(result.trained_artifact.predict_from_embedding(embedding))

        result.metadata["paper_to_lean_local_law_mapping"] = dict(PAPER_TO_LEAN_LOCAL_LAW_MAPPING)
        result.metadata["local_law_eval"] = evaluate_labeled_tree_local_laws(
            val or train,
            score_fn=_proxy_score,
            include_identity_targets=bool(cfg.score_targets.include_identity_targets),
            score_outputs_normalized=True,
            target_min=target_min,
            target_max=target_max,
        )
        if cfg.test.enabled:
            result.metadata["test_local_law_eval"] = evaluate_labeled_tree_local_laws(
                test,
                score_fn=_proxy_score,
                include_identity_targets=bool(cfg.score_targets.include_identity_targets),
                score_outputs_normalized=True,
                target_min=target_min,
                target_max=target_max,
            )
        result.train_targets = tuple(cfg.contract.train_targets)
        result.student_model_class = str(cfg.contract.student_model_class)
        result.supervision_source = str(cfg.contract.supervision_source)
        result.teacher_model_spec = (
            dict(cfg.contract.teacher_model_spec)
            if cfg.contract.teacher_model_spec
            else None
        )
        result.test_count = len(test)
        result.output_dir = str(cfg.run.output_dir) if cfg.run.output_dir else None
        result.metadata = _metadata_with_distillation_contract(result.metadata, cfg)
        return result

    if route == ((TRAIN_TARGET_F,), STUDENT_MODEL_LM_SCALAR_REGRESSION):
        include_identity_targets = bool(cfg.score_targets.include_identity_targets)
        target_min = float(cfg.score_targets.target_min)
        target_max = float(cfg.score_targets.target_max)
        train_records = build_f_lm_regression_records(
            train,
            include_identity_targets=include_identity_targets,
            target_min=target_min,
            target_max=target_max,
        )
        val_records = build_f_lm_regression_records(
            val,
            include_identity_targets=include_identity_targets,
            target_min=target_min,
            target_max=target_max,
        )
        test_records = build_f_lm_regression_records(
            test,
            include_identity_targets=include_identity_targets,
            target_min=target_min,
            target_max=target_max,
        )
        metadata: Dict[str, Any] = {
            "dataset_ready": True,
            "lm_regression_train_records": int(len(train_records)),
            "lm_regression_val_records": int(len(val_records)),
            "lm_regression_test_records": int(len(test_records)),
            "include_identity_targets": bool(include_identity_targets),
            "paper_to_lean_local_law_mapping": dict(PAPER_TO_LEAN_LOCAL_LAW_MAPPING),
            "local_law_eval": evaluate_labeled_tree_local_laws(
                val or train,
                include_identity_targets=include_identity_targets,
                target_min=target_min,
                target_max=target_max,
            ),
        }
        if cfg.test.enabled:
            metadata["test_local_law_eval"] = evaluate_labeled_tree_local_laws(
                test,
                include_identity_targets=include_identity_targets,
                target_min=target_min,
                target_max=target_max,
            )
        if cfg.run.output_dir is not None:
            output_dir = Path(cfg.run.output_dir)
            train_path = write_jsonl_records(
                output_dir / "f_lm_regression_train.jsonl",
                train_records,
            )
            val_path = write_jsonl_records(
                output_dir / "f_lm_regression_val.jsonl",
                val_records,
            )
            test_path = write_jsonl_records(
                output_dir / "f_lm_regression_test.jsonl",
                test_records,
            )
            metadata["lm_regression_train_path"] = str(train_path)
            metadata["lm_regression_val_path"] = str(val_path)
            metadata["lm_regression_test_path"] = str(test_path)

        trained: Optional[Any] = None
        if bool(cfg.f_lm.run_trl_scalar_reward):
            model_name = cfg.f_lm.model_name
            if not model_name:
                raise ValueError("F-LM scalar regression requires f_lm.model_name")
            if not train_records:
                raise ValueError("F-LM scalar regression requested but no records were available")
            from treepo._research.training.trl_training import TRLTrainingConfig, train_scalar_reward_records

            trl_config = cfg.f_lm.trl_config or TRLTrainingConfig()
            trained = train_scalar_reward_records(
                records=train_records,
                eval_records=val_records or None,
                model_name=str(model_name),
                output_dir=Path(cfg.run.output_dir or "outputs/f_lm_regression"),
                config=trl_config,
            )
            metadata["trl_scalar_reward_ran"] = True
            metadata["model_path"] = str(trained)

        return _fit_result(
            cfg,
            train_count=len(train),
            val_count=len(val),
            test_count=len(test),
            trained_artifact=trained,
            metadata=metadata,
        )

    raise ValueError(
        "Unsupported distillation contract: "
        f"train_targets={cfg.contract.train_targets!r}, "
        f"student_model_class={cfg.contract.student_model_class!r}"
    )


__all__ = [
    "LABEL_TREE_VERSION",
    "PAPER_TO_LEAN_LOCAL_LAW_MAPPING",
    "DistillationContractConfig",
    "DistillationFitResult",
    "DistillationTrainConfig",
    "FEmbeddingConfig",
    "FLMConfig",
    "GLMConfig",
    "ScoreTargetConfig",
    "STUDENT_MODEL_CTREEPO_EMBEDDING_TREE",
    "STUDENT_MODEL_EMBEDDING_RIDGE_PROXY",
    "STUDENT_MODEL_LM_SCALAR_REGRESSION",
    "STUDENT_MODEL_LM_SFT",
    "SummaryTargetConfig",
    "SUPERVISION_SOURCE_EMPIRICAL_ROOT_LABELS",
    "SUPERVISION_SOURCE_LABELED_TREE_ARTIFACT",
    "TRAIN_TARGET_F",
    "TRAIN_TARGET_G",
    "TRAIN_TARGET_TREE_OPERATOR",
    "annotate_labeled_tree_summary_coverage",
    "attach_labeled_tree_scores",
    "build_embedding_tree_from_labeled_tree",
    "build_f_embedding_examples",
    "build_f_lm_regression_records",
    "build_labeled_tree_from_text",
    "build_g_sft_records",
    "evaluate_labeled_tree_local_laws",
    "fit_f_embedding_proxy",
    "fit",
    "load_labeled_trees",
    "repair_labeled_tree_missing_summaries",
    "split_labeled_trees",
    "write_jsonl_records",
    "write_labeled_trees_jsonl",
]
