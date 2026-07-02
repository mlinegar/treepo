"""Manifesto policy task-state helpers.

The generic package abstraction is ``TaskState``. This module defines the
small Manifesto/RILE state kind used by examples and downstream qsentence
bridges.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from treepo.local_law import LawKind, LocalLawAuditRow
from treepo.state import TaskState, state_from_value, state_to_dict
from treepo.statistic import StatisticInfo
from treepo.tasks.manifesto.rile import clamp_rile


MANIFESTO_POLICY_STATE_KIND = "manifesto_policy"


def manifesto_policy_state_from_leaf(leaf: Any) -> TaskState:
    """Build a policy state from one gold qsentence/leaf-like object."""

    text = str(getattr(leaf, "text", "") or "")
    qid = str(getattr(leaf, "qid", getattr(leaf, "node_id", "")) or "")
    code = str(getattr(leaf, "code", "") or "")
    score = _optional_float(getattr(leaf, "score", None))
    weight = float(getattr(leaf, "weight", 1.0) or 1.0)
    item = {
        "id": qid,
        "text": text,
        "code": code,
        "score": score,
        "weight": weight,
    }
    counts: dict[str, float] = {"qsentences": 1.0, "non_header": weight}
    if code:
        counts[f"code:{code}"] = weight
        domain = _cmp_domain(code)
        if domain:
            counts[f"domain:{domain}"] = weight
    score_weight = weight if score is not None else 0.0
    score_sum = (float(score) * weight) if score is not None else 0.0
    counts["left"] = weight if score is not None and float(score) < 0.0 else 0.0
    counts["right"] = weight if score is not None and float(score) > 0.0 else 0.0
    measures = {
        "rile": clamp_rile(score or 0.0),
        "rile_from_leaves": clamp_rile(score or 0.0),
    }
    return TaskState(
        kind=MANIFESTO_POLICY_STATE_KIND,
        items=(item,),
        counts=counts,
        measures=measures,
        text=text,
        metadata={
            "score_sum": score_sum,
            "score_weight": score_weight,
            "state_source": "gold_qsentence",
        },
    )


def merge_manifesto_policy_states(left: Any, right: Any) -> TaskState:
    """Merge two Manifesto policy states by concatenating items and summing mass."""

    states = [_coerce_policy_state(left), _coerce_policy_state(right)]
    items: list[Any] = []
    counts: dict[str, float] = {}
    score_sum = 0.0
    score_weight = 0.0
    for state in states:
        items.extend(list(state.items or ()))
        for key, value in dict(state.counts or {}).items():
            counts[str(key)] = float(counts.get(str(key), 0.0)) + float(value)
        meta = dict(state.metadata or {})
        score_sum += float(meta.get("score_sum", 0.0) or 0.0)
        score_weight += float(meta.get("score_weight", 0.0) or 0.0)
    leaf_rile = clamp_rile(score_sum / score_weight) if score_weight > 0.0 else 0.0
    text = "\n".join(str(item.get("text", "")) for item in items if isinstance(item, Mapping) and item.get("text"))
    return TaskState(
        kind=MANIFESTO_POLICY_STATE_KIND,
        items=tuple(items),
        counts=counts,
        measures={
            "rile": leaf_rile,
            "rile_from_leaves": leaf_rile,
        },
        text=text or None,
        metadata={
            "score_sum": score_sum,
            "score_weight": score_weight,
            "state_source": "merged_gold_qsentences",
        },
    )


def attach_manifesto_root_label(state: Any, root_label: float | None) -> TaskState:
    """Attach an observed document/root label to a merged policy state."""

    policy = _coerce_policy_state(state)
    measures = dict(policy.measures or {})
    if root_label is not None:
        measures["rile"] = clamp_rile(float(root_label))
        measures["root_rile"] = clamp_rile(float(root_label))
    metadata = dict(policy.metadata or {})
    metadata["state_source"] = "gold_manifesto_tree"
    if root_label is not None:
        metadata["root_label"] = clamp_rile(float(root_label))
    return TaskState(
        kind=policy.kind,
        items=tuple(policy.items or ()),
        counts=dict(policy.counts or {}),
        measures=measures,
        text=policy.text,
        metadata=metadata,
    )


def manifesto_policy_readout(state: Any, query: Any = None) -> Any:
    """Read a scalar or mapping from a Manifesto policy state."""

    policy = _coerce_policy_state(state)
    measures = dict(policy.measures or {})
    if query in {"state", "measures"}:
        return measures
    key = str(query or "rile")
    if key in measures:
        value = measures[key]
        return clamp_rile(float(value)) if key.startswith("rile") or key == "root_rile" else value
    return measures


def state_to_manifesto_policy_dict(state: Any) -> dict[str, Any]:
    return state_to_dict(_coerce_policy_state(state))


class ManifestoPolicyStatistic:
    """Exact fixture statistic for gold Manifesto qsentence labels."""

    info = StatisticInfo(
        name="manifesto_policy",
        state_kind=MANIFESTO_POLICY_STATE_KIND,
        exact=True,
        supports_local_laws=True,
        metadata={"readout": "rile"},
    )

    def encode_leaf(self, leaf: Any) -> TaskState:
        return manifesto_policy_state_from_leaf(leaf)

    def merge(self, left: Any, right: Any) -> TaskState:
        return merge_manifesto_policy_states(left, right)

    def readout(self, state: Any, query: Any = None) -> Any:
        return manifesto_policy_readout(state, query=query)

    def encode_tree(self, tree: Any) -> TaskState:
        leaves = list(getattr(tree, "leaves", None) or ())
        if not leaves:
            base = TaskState(kind=MANIFESTO_POLICY_STATE_KIND)
        else:
            state = self.encode_leaf(leaves[0])
            for leaf in leaves[1:]:
                state = self.merge(state, self.encode_leaf(leaf))
            base = state
        return attach_manifesto_root_label(base, _root_label(tree))

    def local_law_rows(
        self,
        units: Sequence[Any],
        *,
        query: Any = None,
        oracle: Any = None,
    ) -> Sequence[LocalLawAuditRow]:
        del oracle
        rows: list[LocalLawAuditRow] = []
        for tree_idx, tree in enumerate(list(units or ())):
            tree_id = str(getattr(tree, "doc_id", f"tree_{tree_idx}"))
            for leaf_idx, leaf in enumerate(list(getattr(tree, "leaves", None) or ())):
                score = _optional_float(getattr(leaf, "score", None))
                if score is None:
                    continue
                pred = float(self.readout(self.encode_leaf(leaf), query=query))
                loss = float((pred - clamp_rile(score)) ** 2)
                rows.append(
                    LocalLawAuditRow(
                        row_id=f"{tree_id}:leaf:{leaf_idx}",
                        law_kind=LawKind.C1_LEAF,
                        proxy_loss=loss,
                        oracle_loss=loss,
                        observed=True,
                        propensity=1.0,
                        metadata={
                            "statistic": self.info.name,
                            "state_kind": self.info.state_kind,
                            "unit_type": "qsentence",
                            "tree_id": tree_id,
                            "node_id": getattr(leaf, "qid", leaf_idx),
                            "check": "gold_leaf_readout",
                        },
                    )
                )
            root = _root_label(tree)
            if root is None:
                continue
            state = self.encode_tree(tree)
            pred = float(self.readout(state, query=query))
            loss = float((pred - clamp_rile(root)) ** 2)
            rows.append(
                LocalLawAuditRow(
                    row_id=f"{tree_id}:root",
                    law_kind=LawKind.C3_MERGE,
                    proxy_loss=loss,
                    oracle_loss=loss,
                    observed=True,
                    propensity=1.0,
                    metadata={
                        "statistic": self.info.name,
                        "state_kind": self.info.state_kind,
                        "unit_type": "root",
                        "tree_id": tree_id,
                        "node_id": "root",
                        "check": "gold_root_readout",
                        "rile_from_leaves": dict(state.measures).get("rile_from_leaves"),
                    },
                )
            )
        return tuple(rows)


def _coerce_policy_state(value: Any) -> TaskState:
    state = state_from_value(value)
    if not isinstance(state, TaskState):
        if isinstance(value, Mapping):
            state = TaskState(
                kind=str(value.get("kind") or MANIFESTO_POLICY_STATE_KIND),
                items=tuple(value.get("items") or ()),
                counts=dict(value.get("counts") or {}),
                measures=dict(value.get("measures") or {}),
                text=None if value.get("text") is None else str(value.get("text")),
                metadata=dict(value.get("metadata") or {}),
            )
        else:
            raise TypeError(f"expected manifesto policy TaskState, got {type(value).__name__}")
    if state.kind != MANIFESTO_POLICY_STATE_KIND:
        raise ValueError(f"expected state kind {MANIFESTO_POLICY_STATE_KIND!r}, got {state.kind!r}")
    return state


def _root_label(tree: Any) -> float | None:
    meta = dict(getattr(tree, "metadata", None) or {})
    for key in ("root_label", "teacher_score_native", "teacher_score_1_7"):
        value = meta.get(key)
        parsed = _optional_float(value)
        if parsed is not None:
            return clamp_rile(parsed)
    return None


def _cmp_domain(code: str) -> str:
    digits = "".join(ch for ch in str(code) if ch.isdigit())
    return digits[:1] if digits else ""


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "MANIFESTO_POLICY_STATE_KIND",
    "ManifestoPolicyStatistic",
    "attach_manifesto_root_label",
    "manifesto_policy_readout",
    "manifesto_policy_state_from_leaf",
    "merge_manifesto_policy_states",
    "state_to_manifesto_policy_dict",
]
