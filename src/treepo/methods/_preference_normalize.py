"""Row normalization and scalar/JSON coercion for preference data.

The messy data-prep layer of the preference boundary: parse arbitrary unit and
candidate rows into canonical dicts, coerce optional scalars, and JSON-encode
state values. Pure functions with no dependency on the ``PreferenceDataset``
data model, so the model and its export views can import freely from here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from typing import Any, Mapping, Sequence

from treepo.methods._run_manifest import json_default
from treepo.sampling import ResolvedPropensity, resolve_effective_propensity
from treepo.state import state_from_value, state_to_dict

_UNIT_FIELDS = (
    "unit_id",
    "unit_type",
    "target",
    "context",
    "weight",
    "propensity",
    "sample_weight",
    "propensity_source",
    "sample_weight_source",
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
_PROPENSITY_INPUT_FIELDS = (
    "effective_propensity",
    "joint_propensity",
    "inclusion_probability",
    "sampling",
    "document_propensity",
    "unit_propensity",
    "label_propensity",
    "propensity",
    "supports_ipw_estimation",
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
    metadata = _preference_metadata(out)
    resolved = _resolve_preference_propensity(out, metadata=metadata)
    weight, sample_weight, sample_weight_source = _resolve_preference_weights(
        out,
        propensity=resolved.propensity,
    )
    unit_id = str(out.get("unit_id") or out.get("node_id") or out.get("doc_id") or "")
    out["unit_id"] = unit_id
    out["unit_type"] = str(out.get("unit_type") or out.get("kind") or "unit")
    out["target"] = str(out.get("target") or "g")
    out["context"] = _maybe_json(out.get("context_json", out.get("context", out.get("prompt", ""))))
    out["weight"] = weight
    out["propensity"] = resolved.propensity
    out["sample_weight"] = sample_weight
    out["propensity_source"] = resolved.source
    out["sample_weight_source"] = sample_weight_source
    metadata.update(
        {
            "propensity_source": resolved.source,
            "resolved_propensity": resolved.propensity,
            "sample_weight_source": sample_weight_source,
        }
    )
    out["metadata"] = metadata
    out["record_id"] = str(out.get("record_id") or out.get("id") or unit_id)
    for key in _TREE_FIELDS:
        out.setdefault(key, None)
    out["level"] = _optional_int(out.get("level"))
    out["position"] = _optional_int(out.get("position"))
    return {key: out.get(key) for key in (*_UNIT_FIELDS, "record_id")}


def _normalize_candidate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["unit_id"] = str(out.get("unit_id") or "")
    out["candidate_id"] = str(
        out.get("candidate_id") or out.get("id") or out.get("response_id") or ""
    )
    out["value"] = state_to_dict(
        state_from_value(
            _maybe_json(out.get("value_json", out.get("value", out.get("response", ""))))
        )
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


def _sample_weight(
    weight: Any,
    propensity: Any,
    *,
    precomputed: Any = None,
    min_propensity: float = 1e-8,
) -> float:
    """Return an exact IPW weight, never a propensity-clipped approximation."""

    resolved = resolve_effective_propensity({"propensity": propensity})
    probability = float(resolved.propensity)
    minimum = float(min_propensity)
    if not math.isfinite(minimum) or minimum <= 0.0 or minimum > 1.0:
        raise ValueError("min_propensity must be finite and in (0, 1]")
    if probability < minimum:
        raise ValueError(
            f"preference propensity {probability!r} is below "
            f"min_propensity={minimum!r}; propensity flooring is not allowed"
        )
    if precomputed is not None:
        return _nonnegative_finite_weight(precomputed, "sample_weight")
    base_weight = _nonnegative_finite_weight(
        1.0 if weight is None else weight,
        "weight",
    )
    return float(base_weight / probability)


def _unit_sample_weight(unit: Mapping[str, Any]) -> float:
    """Read a normalized unit weight without recomputing a precomputed value."""

    return _sample_weight(
        unit.get("weight"),
        unit.get("propensity"),
        precomputed=unit.get("sample_weight"),
    )


def _resolve_preference_propensity(
    row: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ResolvedPropensity:
    """Resolve one preference unit design with canonical precedence."""

    parsed_metadata = dict(metadata) if metadata is not None else _preference_metadata(row)
    payload = dict(parsed_metadata)
    for key in _PROPENSITY_INPUT_FIELDS:
        if key in row:
            payload[key] = row.get(key)
    resolved = resolve_effective_propensity(payload)
    declared_source = row.get("propensity_source", parsed_metadata.get("propensity_source"))
    if (
        declared_source is not None
        and str(declared_source).strip()
        and resolved.source in {"metadata.propensity", "default"}
    ):
        return ResolvedPropensity(
            propensity=resolved.propensity,
            source=str(declared_source).strip(),
        )
    return resolved


def _resolve_preference_weights(
    row: Mapping[str, Any],
    *,
    propensity: float,
) -> tuple[float, float, str]:
    """Normalize base/effective weights without applying IPW twice."""

    raw_weight = row.get("weight")
    precomputed = row.get("sample_weight")
    if precomputed is not None:
        effective = _sample_weight(
            raw_weight,
            propensity,
            precomputed=precomputed,
        )
        base = (
            _nonnegative_finite_weight(raw_weight, "weight")
            if raw_weight is not None
            else float(effective * float(propensity))
        )
        source = row.get("sample_weight_source") or "precomputed_sample_weight"
        return base, effective, str(source)
    base = _nonnegative_finite_weight(
        1.0 if raw_weight is None else raw_weight,
        "weight",
    )
    return (
        base,
        _sample_weight(base, propensity),
        "computed_weight_over_propensity",
    )


def _preference_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    value = _maybe_json(
        row.get(
            "metadata_json",
            row.get("unit_metadata", row.get("metadata") or {}),
        )
    )
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise ValueError("preference metadata must be a mapping")
    return dict(value)


def _nonnegative_finite_weight(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return parsed


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
    if text[0] not in '[{"0123456789-tfn':
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


# Canonical JSON fallback shared with the run-manifest writer.
_json_default = json_default


__all__ = [
    "_CANDIDATE_FIELDS",
    "_PROPENSITY_INPUT_FIELDS",
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
    "_preference_metadata",
    "_resolve_preference_propensity",
    "_resolve_preference_weights",
    "_optional_float",
    "_optional_int",
    "_optional_str",
    "_preferred_ids",
    "_rows_from_table",
    "_sample_weight",
    "_unit_sample_weight",
]
