"""Row normalization and scalar/JSON coercion for preference data.

The messy data-prep layer of the preference boundary: parse arbitrary unit and
candidate rows into canonical dicts, coerce optional scalars, and JSON-encode
state values. Pure functions with no dependency on the ``PreferenceDataset``
data model, so the model and its export views can import freely from here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from typing import Any, Mapping, Sequence

from treepo.state import state_from_value, state_to_dict

_UNIT_FIELDS = (
    "unit_id",
    "unit_type",
    "target",
    "context",
    "weight",
    "propensity",
    "metadata",
    "tree_id",
    "doc_id",
    "node_id",
    "level",
    "position",
    "parent_id",
    "left_child_id",
    "right_child_id",
)
_TREE_FIELDS = (
    "tree_id",
    "doc_id",
    "node_id",
    "level",
    "position",
    "parent_id",
    "left_child_id",
    "right_child_id",
)
_CANDIDATE_FIELDS = (
    "unit_id",
    "candidate_id",
    "value",
    "score",
    "rank",
    "preferred",
    "metadata",
)


def _preferred_ids(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        return {str(item) for item in value}
    return {str(value)}


def _rows_from_table(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_list") and callable(value.to_list):
        return [dict(row) for row in value.to_list()]
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        return [dict(row) for row in value if isinstance(row, MappingABC)]
    raise TypeError(f"expected HF Dataset or sequence of mappings, got {type(value).__name__}")


def _is_pairwise_mapping(row: Mapping[str, Any]) -> bool:
    keys = set(row.keys())
    has_left = bool(keys & {"response_a", "candidate_a", "summary_a"})
    has_right = bool(keys & {"response_b", "candidate_b", "summary_b"})
    has_preference = bool(keys & {"preferred", "winner"})
    return has_left and has_right and has_preference and "candidates" not in keys


def _is_flat_candidate_mapping(row: Mapping[str, Any]) -> bool:
    keys = set(row.keys())
    return "unit_id" in keys and bool(keys & {"candidate_id", "response_id", "value", "response"})


def _normalize_unit_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    unit_id = str(out.get("unit_id") or out.get("node_id") or out.get("doc_id") or "")
    out["unit_id"] = unit_id
    out["unit_type"] = str(out.get("unit_type") or out.get("kind") or "unit")
    out["target"] = str(out.get("target") or "g")
    out["context"] = _maybe_json(out.get("context_json", out.get("context", out.get("prompt", ""))))
    out["weight"] = float(out.get("weight", out.get("sample_weight", 1.0)) or 1.0)
    out["propensity"] = float(out.get("propensity", out.get("joint_propensity", 1.0)) or 1.0)
    if out["propensity"] <= 0.0:
        raise ValueError("propensity must be positive")
    out["sample_weight"] = _sample_weight(out["weight"], out["propensity"])
    out["metadata"] = dict(_maybe_json(out.get("metadata_json", out.get("metadata") or {})) or {})
    out["record_id"] = str(out.get("record_id") or out.get("id") or unit_id)
    for key in _TREE_FIELDS:
        out.setdefault(key, None)
    out["level"] = _optional_int(out.get("level"))
    out["position"] = _optional_int(out.get("position"))
    return {key: out.get(key) for key in (*_UNIT_FIELDS, "sample_weight", "record_id")}


def _normalize_candidate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["unit_id"] = str(out.get("unit_id") or "")
    out["candidate_id"] = str(out.get("candidate_id") or out.get("id") or out.get("response_id") or "")
    out["value"] = state_to_dict(
        state_from_value(_maybe_json(out.get("value_json", out.get("value", out.get("response", "")))))
    )
    out["score"] = _optional_float(out.get("score", out.get("reward")))
    out["rank"] = _optional_int(out.get("rank"))
    out["preferred"] = _bool(out.get("preferred", False))
    out["metadata"] = dict(_maybe_json(out.get("metadata_json", out.get("metadata") or {})) or {})
    return {key: out.get(key) for key in _CANDIDATE_FIELDS}


def _hf_unit_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = _normalize_unit_row(row)
    out["context"] = _json_text(out.get("context", ""))
    out["metadata"] = _json_text(out.get("metadata", {}))
    return out


def _hf_candidate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = _normalize_candidate_row(row)
    out["value"] = _json_text(out.get("value", ""))
    out["metadata"] = _json_text(out.get("metadata", {}))
    return out


def _sample_weight(weight: Any, propensity: Any, *, min_propensity: float = 1e-8) -> float:
    return float(weight or 1.0) / max(float(propensity or 1.0), min_propensity)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _json_text(value: Any) -> str:
    value = state_to_dict(value)
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return json.dumps(value, sort_keys=True, default=str)


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{\"0123456789-tfn":
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _mean(values: Sequence[float] | Any) -> float | None:
    rows = [float(value) for value in values]
    if not rows:
        return None
    return float(sum(rows) / len(rows))


def _json_default(value: Any) -> Any:
    state_value = state_to_dict(value)
    if state_value is not value:
        return state_value
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            pass
    return str(value)


__all__ = [
    "_CANDIDATE_FIELDS",
    "_TREE_FIELDS",
    "_UNIT_FIELDS",
    "_bool",
    "_hf_candidate_row",
    "_hf_unit_row",
    "_is_flat_candidate_mapping",
    "_is_pairwise_mapping",
    "_json_default",
    "_json_text",
    "_maybe_json",
    "_mean",
    "_normalize_candidate_row",
    "_normalize_unit_row",
    "_optional_float",
    "_optional_int",
    "_optional_str",
    "_preferred_ids",
    "_rows_from_table",
    "_sample_weight",
]
