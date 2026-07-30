"""Convert preference datasets into generic f/g training traces."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

from treepo.methods._coerce import float_vector as _as_float_vector
from treepo.methods._coerce import safe_float as _as_float
from treepo.methods.preference import PreferenceDataset
from treepo.state import state_from_value, state_to_dict


def preference_training_rows(
    dataset: PreferenceDataset,
    *,
    target: str,
) -> tuple[Any, ...]:
    """Convert preferences without discarding their structured supervision."""

    if len(dataset) == 0:
        return ()
    filtered = dataset.filter_target(target)
    units_by_id = {str(unit.get("unit_id") or ""): unit for unit in filtered.units}
    rows: list[Any] = []
    for record in filtered.to_records("supervised"):
        label = _supervised_label(record, target=target)
        if label is None:
            continue
        unit_id = str(record.get("unit_id") or "")
        unit = units_by_id.get(unit_id, {})
        metadata = dict(record.get("metadata") or {})
        prompt = str(
            metadata.get("g_prompt")
            if target == "g" and metadata.get("g_prompt") is not None
            else record.get("prompt") or ""
        )
        completion = str(record.get("completion") or "")
        text = _supervised_text(prompt=prompt, completion=completion, unit_id=unit_id)
        label_payload = state_to_dict(label)
        propensity = float(unit.get("propensity", 1.0))
        sample_weight = float(record.get("sample_weight", unit.get("sample_weight", 1.0)))
        # DSPy consumes a base weight and divides by propensity exactly once.
        # Reconstruct that base from the authoritative effective weight so a
        # supplied/pre-capped sample_weight is never divided a second time.
        weight = float(sample_weight * propensity)
        propensity_source = str(
            unit.get("propensity_source")
            or metadata.get("propensity_source")
            or "metadata.propensity"
        )
        sample_weight_source = str(
            unit.get("sample_weight_source")
            or metadata.get("sample_weight_source")
            or "computed_weight_over_propensity"
        )
        call_role = _preference_call_role(record, unit, metadata) if target == "g" else None
        metadata.update(
            {
                "preference_unit_id": unit_id,
                "preference_unit_type": str(record.get("unit_type") or "unit"),
                "preference_candidate_id": str(record.get("candidate_id") or ""),
                "preference_target": str(target),
                "text": text,
                "prompt": prompt,
                "completion": completion,
                "oracle_target": label_payload,
                "observed": True,
                "weight": weight,
                "sample_weight": sample_weight,
                "propensity": propensity,
                "effective_propensity": propensity,
                "propensity_source": propensity_source,
                "sample_weight_source": sample_weight_source,
                "preference_input_weight": unit.get("weight"),
            }
        )
        if call_role is not None:
            metadata["call_role"] = call_role
        scalar = _as_float(label)
        vector = _as_float_vector(label)
        if scalar is not None:
            metadata["teacher_score_native"] = scalar
            document_score: float | None = scalar
        elif vector is not None:
            metadata["target_vector"] = vector
            metadata["topic_proportions"] = vector
            document_score = None
        else:
            document_score = None
        rows.append(
            SimpleNamespace(
                text=text,
                content=text,
                tokens=text.split(),
                prompt=prompt,
                completion=completion,
                value=label_payload,
                oracle_target=label_payload,
                document_score=document_score,
                topic_proportions=vector,
                weight=weight,
                sample_weight=sample_weight,
                propensity=propensity,
                call_role=call_role,
                unit_id=unit_id,
                unit_type=str(record.get("unit_type") or "unit"),
                candidate_id=str(record.get("candidate_id") or ""),
                preference_target=str(target),
                metadata=metadata,
            )
        )
    return tuple(rows)


def _supervised_label(row: Mapping[str, Any], *, target: str) -> Any:
    if target == "g":
        value = state_from_value(row.get("value"))
        if _is_present_g_state(value):
            return state_to_dict(value)
        completion = row.get("completion")
        if _is_present_g_state(completion):
            return state_to_dict(completion)
    return _numeric_supervised_label(row)


def _numeric_supervised_label(row: Mapping[str, Any]) -> Any:
    value = state_from_value(row.get("value"))
    value_payload = state_to_dict(value)
    if isinstance(value_payload, Mapping):
        return value_payload
    vector = _as_float_vector(value)
    if vector is not None:
        return vector
    scalar = _as_float(value)
    if scalar is not None:
        return scalar
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("oracle_target", "label", "target_value"):
            nested = metadata.get(key)
            vector = _as_float_vector(nested)
            if vector is not None:
                return vector
            scalar = _as_float(nested)
            if scalar is not None:
                return scalar
    score = _as_float(row.get("score"))
    if score is not None:
        return score
    return _as_float(row.get("completion"))


def _is_present_g_state(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _preference_call_role(
    record: Mapping[str, Any],
    unit: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str | None:
    unit_metadata = unit.get("metadata")
    unit_metadata = unit_metadata if isinstance(unit_metadata, Mapping) else {}
    for value in (
        record.get("call_role"),
        metadata.get("call_role"),
        metadata.get("g_call_role"),
        unit_metadata.get("call_role"),
        unit_metadata.get("g_call_role"),
    ):
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    child_ids = tuple(
        child_id
        for child_id in (
            _preference_topology_value(
                "left_child_id",
                record=record,
                unit=unit,
                metadata=metadata,
                unit_metadata=unit_metadata,
            ),
            _preference_topology_value(
                "right_child_id",
                record=record,
                unit=unit,
                metadata=metadata,
                unit_metadata=unit_metadata,
            ),
        )
        if child_id is not None
    )
    if child_ids:
        return "recompression" if len(child_ids) == 1 else "merge"
    unit_type = str(record.get("unit_type") or unit.get("unit_type") or "")
    if unit_type in {"leaf", "recompression", "merge"}:
        return unit_type
    return "leaf"


def _preference_topology_value(
    key: str,
    *,
    record: Mapping[str, Any],
    unit: Mapping[str, Any],
    metadata: Mapping[str, Any],
    unit_metadata: Mapping[str, Any],
) -> str | None:
    for source in (record, metadata, unit, unit_metadata):
        value = source.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _supervised_text(*, prompt: str, completion: str, unit_id: str) -> str:
    stripped_prompt = str(prompt).strip()
    stripped_completion = str(completion).strip()
    if stripped_prompt and stripped_completion:
        return f"{stripped_prompt}\n{stripped_completion}"
    return stripped_prompt or stripped_completion or str(unit_id)


__all__ = ["preference_training_rows"]
