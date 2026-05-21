"""Stateful tree data structures for generic TreePO operators.

This module is intentionally separate from ``src.core.data_models``:
- ``Node``/``Tree`` there are text-first and assume a ``summary`` string.
- ``StateNode``/``StateTree`` here carry an arbitrary internal ``state`` object
  alongside a theorem-domain ``span`` object, plus a ``rendered`` string for
  logs/UI.

The primary requirement is that ``StateTree.to_dict()`` is JSON-safe and does
not attempt to serialize large tensors or opaque objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
import uuid
from typing import Any, Callable, Dict, Generic, Iterator, List, Mapping, Optional, Sequence, TypeVar

SpanT = TypeVar("SpanT")
StateT = TypeVar("StateT")

JsonSerializer = Callable[[Any], Optional[Any]]


def _finite_metadata_float(metadata: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key not in metadata:
            continue
        value = metadata.get(key)
        if value is None:
            continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if out == out and out not in (float("inf"), float("-inf")):
            return out
    return None


def _metadata_truthy(metadata: Mapping[str, Any], keys: Sequence[str]) -> bool:
    for key in keys:
        if key not in metadata:
            continue
        value = metadata.get(key)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "observed", "sampled"}:
                return True
            if normalized in {"0", "false", "no", "n", "none", ""}:
                continue
        elif bool(value):
            return True
    return False


def explicit_oracle_trace_kwargs(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract explicit oracle-observation kwargs for ``local_law_trace_metadata``.

    Plain ``target`` fields are proxy supervision. An oracle row exists only
    when the producer provides an oracle loss/target and marks it observed or
    sampled with a logged propensity.
    """

    data = dict(metadata or {})
    oracle_loss = _finite_metadata_float(data, ("oracle_loss", "loss_oracle"))
    oracle_target = _finite_metadata_float(
        data,
        ("oracle_target", "oracle_score", "oracle_target_score"),
    )
    observed = _metadata_truthy(data, ("oracle_observed", "observed", "sampled"))
    if not observed:
        return {
            "oracle_loss": None,
            "oracle_target": None,
            "observed": False,
            "sampled": False,
            "propensity": None,
            "label_source": "",
        }
    if oracle_loss is None and oracle_target is None:
        raise ValueError("observed local-law trace rows require explicit oracle_loss or oracle_target")
    propensity = _finite_metadata_float(
        data,
        ("oracle_propensity", "propensity", "logged_propensity"),
    )
    if propensity is None:
        raise ValueError("observed local-law trace rows require a logged propensity")
    sampled = _metadata_truthy(data, ("sampled", "oracle_sampled")) or observed
    return {
        "oracle_loss": oracle_loss,
        "oracle_target": oracle_target,
        "observed": True,
        "sampled": bool(sampled),
        "propensity": float(propensity),
        "label_source": str(data.get("label_source", data.get("truth_label_source", "oracle")) or "oracle"),
    }


def _default_json_safe(value: Any, *, serializer: Optional[JsonSerializer] = None) -> Any:
    """Convert arbitrary Python objects into JSON-safe structures.

    Rules:
    - dataclasses -> dict of fields (recursing)
    - torch.Tensor -> structural metadata (dtype/shape/device), no raw values
    - unknown objects -> repr(...)
    """
    if serializer is not None:
        overridden = serializer(value)
        if overridden is not None:
            return overridden

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value):
        return {f.name: _default_json_safe(getattr(value, f.name), serializer=serializer) for f in fields(value)}

    # Avoid importing torch unless it is available.
    try:
        import torch  # type: ignore

        if isinstance(value, torch.Tensor):
            return {
                "type": "torch.Tensor",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "device": str(value.device),
            }
    except Exception:
        pass

    if isinstance(value, Mapping):
        return {str(k): _default_json_safe(v, serializer=serializer) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_default_json_safe(v, serializer=serializer) for v in list(value)]

    return repr(value)


@dataclass
class StateNode(Generic[SpanT, StateT]):
    """One node in a stateful TreePO tree."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    level: int = 0

    # Theorem-domain object for this node (e.g., text span, list-of-states, etc.).
    span: Optional[SpanT] = None

    # Internal mergeable state for this node (e.g., summary string, sketch tensor).
    state: Optional[StateT] = None

    # Human-readable view of the node state for logs/UI.
    rendered: str = ""

    left_child: Optional["StateNode[SpanT, StateT]"] = None
    right_child: Optional["StateNode[SpanT, StateT]"] = None
    parent: Optional["StateNode[SpanT, StateT]"] = None

    metadata: Dict[str, Any] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return self.left_child is None and self.right_child is None

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def children(self) -> List["StateNode[SpanT, StateT]"]:
        children: List["StateNode[SpanT, StateT]"] = []
        if self.left_child is not None:
            children.append(self.left_child)
        if self.right_child is not None:
            children.append(self.right_child)
        return children

    def validate(self) -> List[str]:
        violations: List[str] = []
        if self.is_leaf:
            if self.level != 0:
                violations.append(f"Leaf node has non-zero level: {self.level}")
        else:
            if self.level <= 0:
                violations.append("Internal node has non-positive level")
            if (self.left_child is None) != (self.right_child is None):
                violations.append("Node has exactly one child (must have 0 or 2)")
        for child in self.children:
            if child.parent is not self:
                violations.append(f"Child {child.id} missing parent reference")
        return violations

    def to_dict(self, *, serializer: Optional[JsonSerializer] = None) -> Dict[str, Any]:
        return {
            "id": self.id,
            "level": int(self.level),
            "span": _default_json_safe(self.span, serializer=serializer),
            "state": _default_json_safe(self.state, serializer=serializer),
            "rendered": str(self.rendered or ""),
            "left_child_id": self.left_child.id if self.left_child is not None else None,
            "right_child_id": self.right_child.id if self.right_child is not None else None,
            "parent_id": self.parent.id if self.parent is not None else None,
            "metadata": _default_json_safe(self.metadata, serializer=serializer),
            "audit": _default_json_safe(self.audit, serializer=serializer),
        }


@dataclass
class StateTree(Generic[SpanT, StateT]):
    """A stateful TreePO tree rooted at ``root``."""

    root: StateNode[SpanT, StateT]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        if self.root is None:
            return 0
        return self._height(self.root)

    def _height(self, node: StateNode[SpanT, StateT]) -> int:
        if node.is_leaf:
            return 0
        left_h = self._height(node.left_child) if node.left_child is not None else 0
        right_h = self._height(node.right_child) if node.right_child is not None else 0
        return 1 + max(left_h, right_h)

    @property
    def node_count(self) -> int:
        return sum(1 for _ in self.traverse_preorder())

    @property
    def final_rendered(self) -> str:
        return str(self.root.rendered or "")

    def traverse_preorder(self) -> Iterator[StateNode[SpanT, StateT]]:
        stack: List[StateNode[SpanT, StateT]] = [self.root]
        while stack:
            node = stack.pop()
            yield node
            if node.right_child is not None:
                stack.append(node.right_child)
            if node.left_child is not None:
                stack.append(node.left_child)

    def to_dict(self, *, serializer: Optional[JsonSerializer] = None) -> Dict[str, Any]:
        nodes = list(self.traverse_preorder())
        return {
            "root_id": self.root.id,
            "final_rendered": self.final_rendered,
            "height": int(self.height),
            "node_count": int(len(nodes)),
            "nodes": {node.id: node.to_dict(serializer=serializer) for node in nodes},
            "metadata": _default_json_safe(self.metadata, serializer=serializer),
        }


def _labeled_tree_ordered_nodes(tree: Any) -> List[Any]:
    out: List[Any] = []
    seen: set[str] = set()
    levels = list(getattr(tree, "levels", None) or [])
    for level_ids in levels:
        for node_id in level_ids:
            node = tree.get_node(str(node_id)) if hasattr(tree, "get_node") else None
            if node is not None and str(getattr(node, "node_id", node_id)) not in seen:
                out.append(node)
                seen.add(str(getattr(node, "node_id", node_id)))
    raw_nodes = getattr(tree, "nodes", {}) or {}
    node_values = raw_nodes.values() if isinstance(raw_nodes, Mapping) else raw_nodes
    for node in sorted(
        list(node_values),
        key=lambda item: (int(getattr(item, "level", 0)), str(getattr(item, "node_id", ""))),
    ):
        node_id = str(getattr(node, "node_id", ""))
        if node_id and node_id not in seen:
            out.append(node)
            seen.add(node_id)
    return out


def _labeled_tree_root_node(tree: Any) -> Any:
    levels = list(getattr(tree, "levels", None) or [])
    for level_ids in reversed(levels):
        for node_id in level_ids:
            node = tree.get_node(str(node_id)) if hasattr(tree, "get_node") else None
            if node is not None:
                return node
    ordered = _labeled_tree_ordered_nodes(tree)
    if not ordered:
        raise ValueError("labeled tree has no nodes")
    return ordered[-1]


def state_tree_skeleton_from_labeled_tree(
    tree: Any,
    *,
    method_family: str = "",
    state_kind: str = "",
    split: Optional[str] = None,
    include_rendered_text: bool = True,
) -> StateTree[Any, Any]:
    """Create a stable full-tree trace skeleton from a ``LabeledTree``-like object."""

    doc_id = str(getattr(tree, "doc_id", "") or "")
    tree_meta = dict(getattr(tree, "metadata", {}) or {})
    resolved_split = str(split if split is not None else tree_meta.get("split", "") or "")
    root_source = _labeled_tree_root_node(tree)
    root_level = int(getattr(root_source, "level", 0))
    cache: Dict[str, StateNode[Any, Any]] = {}

    def build(source_node: Any) -> StateNode[Any, Any]:
        source_id = str(getattr(source_node, "node_id", ""))
        if not source_id:
            raise ValueError("labeled tree node missing node_id")
        if source_id in cache:
            return cache[source_id]

        source_level = int(getattr(source_node, "level", 0))
        left_id = getattr(source_node, "left_child_id", None)
        right_id = getattr(source_node, "right_child_id", None)
        left_node = tree.get_node(str(left_id)) if left_id and hasattr(tree, "get_node") else None
        right_node = tree.get_node(str(right_id)) if right_id and hasattr(tree, "get_node") else None
        is_leaf = left_node is None and right_node is None
        node_type = "root" if source_id == str(getattr(root_source, "node_id", "")) else ("leaf" if is_leaf else "merge")
        depth = max(0, int(root_level - source_level))
        metadata = dict(getattr(source_node, "metadata", {}) or {})
        metadata.update(
            {
                "doc_id": doc_id,
                "source_node_id": source_id,
                "split": resolved_split,
                "node_type": node_type,
                "is_leaf": bool(is_leaf),
                "is_root": bool(node_type == "root"),
                "depth": int(depth),
                "source_level": int(source_level),
                "target": float(getattr(source_node, "score", 0.0)),
            }
        )
        if method_family:
            metadata["method_family"] = str(method_family)
        if state_kind:
            metadata["state_kind"] = str(state_kind)
        rendered = str(getattr(source_node, "text", "") or "") if bool(include_rendered_text) else ""
        node = StateNode[Any, Any](
            id=source_id,
            level=source_level,
            span=str(getattr(source_node, "text", "") or ""),
            state=None,
            rendered=rendered,
            metadata=metadata,
        )
        cache[source_id] = node
        if left_node is not None:
            node.left_child = build(left_node)
            node.left_child.parent = node
        if right_node is not None:
            node.right_child = build(right_node)
            node.right_child.parent = node
        return node

    root = build(root_source)
    return StateTree(
        root=root,
        metadata={
            "doc_id": doc_id,
            "split": resolved_split,
            "method_family": str(method_family or ""),
            "state_kind": str(state_kind or ""),
            "trace_schema": "state_tree_full_trace_v1",
            **tree_meta,
        },
    )


def update_state_tree_node(
    tree: StateTree[Any, Any],
    node_id: str,
    *,
    rendered: Optional[str] = None,
    state: Any = None,
    metadata: Optional[Mapping[str, Any]] = None,
    audit: Optional[Mapping[str, Any]] = None,
) -> StateNode[Any, Any]:
    """Update node payload fields while preserving tree topology."""

    target_id = str(node_id)
    for node in tree.traverse_preorder():
        if str(node.id) != target_id:
            continue
        if rendered is not None:
            node.rendered = str(rendered)
        if state is not None:
            node.state = state
        if metadata:
            node.metadata.update(dict(metadata))
        if audit:
            node.audit.update(dict(audit))
        return node
    raise KeyError(f"state tree node not found: {node_id!r}")


def local_law_trace_metadata(
    *,
    prediction: Optional[float] = None,
    proxy_target: Optional[float] = None,
    proxy_loss: Optional[float] = None,
    oracle_target: Optional[float] = None,
    oracle_loss: Optional[float] = None,
    observed: bool = False,
    sampled: Optional[bool] = None,
    propensity: Optional[float] = None,
    node_weight: Optional[float] = None,
    law_channel: str = "",
    state_kind: str = "",
    label_source: str = "",
) -> Dict[str, Any]:
    """Return canonical local-law trace metadata for one node.

    Proxy rows may be dense. Oracle rows are only emitted when the node is
    actually observed under the logging design.
    """

    payload: Dict[str, Any] = {}
    pred_value: Optional[float] = None
    if prediction is not None:
        pred_value = float(prediction)
        payload["prediction"] = pred_value
    if proxy_target is not None:
        target_value = float(proxy_target)
        payload["proxy_target"] = target_value
        if proxy_loss is None and pred_value is not None:
            proxy_loss = float((pred_value - target_value) ** 2)
    if proxy_loss is not None:
        payload["proxy_loss"] = float(proxy_loss)

    observed_value = bool(observed)
    sampled_value = bool(observed_value if sampled is None else sampled)
    if observed_value:
        if propensity is None:
            raise ValueError("observed local-law trace rows require propensity")
        prop_value = float(propensity)
        if prop_value <= 0.0 or prop_value > 1.0:
            raise ValueError(f"observed local-law trace propensity must be in (0, 1], got {propensity!r}")
        if oracle_target is not None:
            oracle_target_value = float(oracle_target)
            payload["oracle_target"] = oracle_target_value
            if oracle_loss is None and pred_value is not None:
                oracle_loss = float((pred_value - oracle_target_value) ** 2)
        if oracle_loss is None:
            raise ValueError("observed local-law trace rows require oracle_loss or oracle_target")
        payload["oracle_loss"] = float(oracle_loss)
        payload["observed"] = True
        payload["sampled"] = sampled_value
        payload["propensity"] = prop_value
    else:
        payload["observed"] = False
        payload["sampled"] = False
        payload["propensity"] = 0.0

    if node_weight is not None:
        payload["node_weight"] = float(node_weight)
    if law_channel:
        payload["law_channel"] = str(law_channel)
    if state_kind:
        payload["state_kind"] = str(state_kind)
    if label_source:
        payload["label_source"] = str(label_source)
    return payload


def state_tree_trace_metrics(trees: Sequence[StateTree[Any, Any]]) -> Dict[str, Any]:
    """Return compact metrics for a collection of full-tree traces."""

    tree_list = list(trees)
    node_count = 0
    proxy_row_count = 0
    oracle_row_count = 0
    observed_count = 0
    by_state_kind: Dict[str, int] = {}
    for tree in tree_list:
        for node in tree.traverse_preorder():
            node_count += 1
            metadata = dict(node.metadata or {})
            has_proxy_loss = metadata.get("proxy_loss") is not None
            if not has_proxy_loss:
                has_proxy_loss = (
                    metadata.get("prediction") is not None
                    or metadata.get("readout_prediction") is not None
                    or metadata.get("scorer_output") is not None
                ) and (
                    metadata.get("proxy_target") is not None
                    or metadata.get("target") is not None
                    or metadata.get("target_score") is not None
                )
            if has_proxy_loss:
                proxy_row_count += 1
            if metadata.get("oracle_loss") is not None:
                oracle_row_count += 1
            if bool(metadata.get("observed", metadata.get("sampled", False))):
                observed_count += 1
            kind = str(metadata.get("state_kind", "") or "")
            if kind:
                by_state_kind[kind] = int(by_state_kind.get(kind, 0)) + 1
    return {
        "count_trees": int(len(tree_list)),
        "count_nodes": int(node_count),
        "count_proxy_rows": int(proxy_row_count),
        "count_oracle_rows": int(oracle_row_count),
        "count_observed_nodes": int(observed_count),
        "state_kind_counts": by_state_kind,
    }


def write_state_trees_jsonl(
    trees: Sequence[StateTree[Any, Any]],
    path: str | Path,
    *,
    serializer: Optional[JsonSerializer] = None,
) -> Path:
    """Write full-tree traces as one JSON-safe ``StateTree`` per line."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for tree in trees:
            handle.write(json.dumps(tree.to_dict(serializer=serializer), sort_keys=True) + "\n")
    return output


def state_tree_to_text_tree(
    tree: StateTree[str, str],
    *,
    rubric: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
) -> "Tree":
    """Convert a text StateTree into the legacy ``src.core.data_models.Tree``."""

    from treepo._research.core.data_models import Node, Tree

    node_map: Dict[str, Node] = {}

    def build(node: StateNode[str, str]) -> Node:
        if node.id in node_map:
            return node_map[node.id]
        if node.is_leaf:
            legacy = Node(
                id=node.id,
                level=0,
                raw_text_span=str(node.span or ""),
                ops_span=str(node.span or ""),
                summary=str(node.rendered or ""),
                metadata=dict(node.metadata or {}),
            )
            node_map[node.id] = legacy
            return legacy

        left = build(node.left_child) if node.left_child is not None else None
        right = build(node.right_child) if node.right_child is not None else None
        legacy = Node(
            id=node.id,
            level=int(node.level),
            raw_text_span=None,
            ops_span=str(node.span or ""),
            summary=str(node.rendered or ""),
            left_child=left,
            right_child=right,
            metadata=dict(node.metadata or {}),
        )
        if left is not None:
            left.parent = legacy
        if right is not None:
            right.parent = legacy
        node_map[node.id] = legacy
        return legacy

    root = build(tree.root)
    return Tree(
        root=root,
        rubric=str(rubric or ""),
        metadata=dict(metadata or {}),
    )


__all__ = [
    "JsonSerializer",
    "SpanT",
    "StateT",
    "explicit_oracle_trace_kwargs",
    "StateNode",
    "StateTree",
    "state_tree_skeleton_from_labeled_tree",
    "state_tree_trace_metrics",
    "state_tree_to_text_tree",
    "update_state_tree_node",
    "write_state_trees_jsonl",
]
