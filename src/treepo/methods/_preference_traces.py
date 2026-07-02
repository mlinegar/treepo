"""Convert preference datasets into generic f/g training traces."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

from treepo.methods._coerce import float_vector as _as_float_vector, safe_float as _as_float
from treepo.methods.preference import PreferenceDataset
from treepo.state import state_from_value, state_to_dict


def preference_training_rows(
    dataset: PreferenceDataset,
    *,
    target: str,
) -> tuple[Any, ...]:
    """Convert unit/candidate preferences into generic family training traces."""

    if len(dataset) == 0:
        return ()
    rows: list[Any] = []
    for record in dataset.filter_target(target).to_records("supervised"):
        label = _supervised_label(record)
        if label is None:
            continue
        text = _supervised_text(record)
        metadata = dict(record.get("metadata") or {})
        label_payload = state_to_dict(label)
        metadata.update(
            {
                "preference_unit_id": str(record.get("unit_id") or ""),
                "preference_unit_type": str(record.get("unit_type") or "unit"),
                "preference_candidate_id": str(record.get("candidate_id") or ""),
                "preference_target": str(target),
                "text": text,
                "oracle_target": label_payload,
                "observed": True,
                "propensity": 1.0,
            }
        )
        scalar = _as_float(label)
        vector = _as_float_vector(label)
        if scalar is not None:
            metadata["teacher_score_native"] = scalar
            document_score: float | None = scalar
        else:
            metadata["target_vector"] = vector
            metadata["topic_proportions"] = vector
            document_score = None
        rows.append(
            SimpleNamespace(
                text=text,
                content=text,
                tokens=text.split(),
                document_score=document_score,
                topic_proportions=vector,
                metadata=metadata,
            )
        )
    return tuple(rows)


def _supervised_label(row: Mapping[str, Any]) -> Any:
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


def _supervised_text(row: Mapping[str, Any]) -> str:
    prompt = str(row.get("prompt") or "").strip()
    completion = str(row.get("completion") or "").strip()
    if prompt and completion:
        return f"{prompt}\n{completion}"
    return prompt or completion or str(row.get("unit_id") or "")


__all__ = ["preference_training_rows"]
