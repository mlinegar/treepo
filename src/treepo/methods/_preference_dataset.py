"""Unit-level preference data model.

``PreferenceDataset`` is the package data boundary for labels, scored
candidates, ranked candidates, and pairwise preferences. Pairwise DPO rows are
one projection of this data, not the storage model. This module owns only the
data model (``Candidate`` / ``PreferenceRecord`` / ``PreferenceDataset``); row
normalization lives in ``_preference_normalize`` and the optimizer-facing
projections live in ``_preference_views``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from treepo.methods._preference_normalize import (
    _CANDIDATE_FIELDS,
    _TREE_FIELDS,
    _UNIT_FIELDS,
    _bool,
    _hf_candidate_row,
    _hf_unit_row,
    _is_flat_candidate_mapping,
    _is_pairwise_mapping,
    _json_default,
    _maybe_json,
    _mean,
    _normalize_candidate_row,
    _normalize_unit_row,
    _optional_float,
    _optional_int,
    _optional_str,
    _preferred_ids,
    _rows_from_table,
    _sample_weight,
)
from treepo.methods._preference_views import (
    _candidate_record,
    _candidate_text,
    _context_text,
    _export_metadata,
    _grpo_ranks,
    _ordered_candidates,
    _pair_candidates,
    _preferred_side,
    _reward_scores,
    _top_candidates,
)
from treepo.state import state_from_value, state_to_dict

PreferenceTarget = Literal["f", "g", "both"]
PreferenceFormat = Literal["general", "supervised", "dpo", "reward", "grpo"]


@dataclass(frozen=True)
class Candidate:
    id: str
    value: Any
    score: float | None = None
    rank: int | None = None
    preferred: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "Candidate":
        if isinstance(value, Candidate):
            return value
        if isinstance(value, MappingABC):
            row = dict(value)
            candidate_id = row.get("id", row.get("candidate_id", row.get("response_id", "")))
            candidate_value = row.get("value", row.get("response", row.get("text", "")))
            return cls(
                id=str(candidate_id),
                value=state_from_value(_maybe_json(row.get("value_json", candidate_value))),
                score=_optional_float(row.get("score", row.get("reward"))),
                rank=_optional_int(row.get("rank")),
                preferred=_bool(row.get("preferred", False)),
                metadata=dict(_maybe_json(row.get("metadata_json", row.get("metadata") or {})) or {}),
            )
        raise TypeError(f"candidate entries must be Candidate or mapping; got {type(value).__name__}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["value"] = state_to_dict(self.value)
        payload["metadata"] = dict(self.metadata or {})
        return payload

    def to_row(self, unit_id: str) -> dict[str, Any]:
        return {
            "unit_id": str(unit_id),
            "candidate_id": str(self.id),
            "value": state_to_dict(self.value),
            "score": None if self.score is None else float(self.score),
            "rank": None if self.rank is None else int(self.rank),
            "preferred": bool(self.preferred),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class PreferenceRecord:
    unit_id: str
    unit_type: str
    target: PreferenceTarget
    context: Any
    candidates: tuple[Candidate, ...] = ()
    record_id: str = ""
    weight: float = 1.0
    propensity: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tree_id: str | None = None
    doc_id: str | None = None
    node_id: str | None = None
    level: int | None = None
    position: int | None = None
    parent_id: str | None = None
    left_child_id: str | None = None
    right_child_id: str | None = None

    def __post_init__(self) -> None:
        if self.target not in {"f", "g", "both"}:
            raise ValueError("target must be 'f', 'g', or 'both'")
        if float(self.propensity) <= 0.0:
            raise ValueError("propensity must be positive")
        object.__setattr__(
            self,
            "candidates",
            tuple(Candidate.from_value(candidate) for candidate in self.candidates),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PreferenceRecord":
        row = dict(payload)
        if _is_pairwise_mapping(row):
            return _record_from_pairwise(row)
        candidates = row.get("candidates", ())
        if not isinstance(candidates, SequenceABC) or isinstance(candidates, (str, bytes)):
            raise TypeError("preference record candidates must be a sequence")
        raw_preferred = row.get("preferred")
        parsed_candidates = [Candidate.from_value(candidate) for candidate in candidates]
        if raw_preferred is not None:
            preferred_ids = _preferred_ids(raw_preferred)
            parsed_candidates = [
                Candidate(
                    id=candidate.id,
                    value=candidate.value,
                    score=candidate.score,
                    rank=candidate.rank,
                    preferred=candidate.preferred or candidate.id in preferred_ids,
                    metadata=candidate.metadata,
                )
                for candidate in parsed_candidates
            ]
        return cls(
            record_id=str(row.get("record_id") or row.get("id") or ""),
            unit_id=str(row.get("unit_id") or row.get("node_id") or row.get("doc_id") or ""),
            unit_type=str(row.get("unit_type") or row.get("kind") or "unit"),
            target=str(row.get("target") or "g"),  # type: ignore[arg-type]
            context=_maybe_json(row.get("context_json", row.get("context", row.get("prompt", "")))),
            candidates=tuple(parsed_candidates),
            weight=float(row.get("weight", row.get("sample_weight", 1.0)) or 1.0),
            propensity=float(row.get("propensity", row.get("joint_propensity", 1.0)) or 1.0),
            metadata=dict(_maybe_json(row.get("metadata_json", row.get("metadata") or {})) or {}),
            tree_id=_optional_str(row.get("tree_id")),
            doc_id=_optional_str(row.get("doc_id", row.get("source_doc_id"))),
            node_id=_optional_str(row.get("node_id")),
            level=_optional_int(row.get("level")),
            position=_optional_int(row.get("position")),
            parent_id=_optional_str(row.get("parent_id")),
            left_child_id=_optional_str(row.get("left_child_id")),
            right_child_id=_optional_str(row.get("right_child_id")),
        )

    def sample_weight(self, *, min_propensity: float = 1e-8, max_weight: float | None = None) -> float:
        value = float(self.weight) / max(float(self.propensity), float(min_propensity))
        if max_weight is not None:
            value = min(value, float(max_weight))
        return float(value)

    def to_unit_row(self) -> dict[str, Any]:
        row = {
            "unit_id": str(self.unit_id),
            "unit_type": str(self.unit_type),
            "target": str(self.target),
            "context": self.context,
            "weight": float(self.weight),
            "propensity": float(self.propensity),
            "sample_weight": self.sample_weight(),
            "metadata": dict(self.metadata or {}),
            "record_id": str(self.record_id or self.unit_id),
        }
        for key in _TREE_FIELDS:
            value = getattr(self, key)
            row[key] = value
        return row

    def to_dict(self) -> dict[str, Any]:
        row = self.to_unit_row()
        row["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return row


class PreferenceDataset:
    """HF-compatible unit/candidate preference dataset."""

    def __init__(
        self,
        records: Any = (),
        *,
        units: Sequence[Mapping[str, Any]] | None = None,
        candidates: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.units: list[dict[str, Any]] = []
        self.candidates: list[dict[str, Any]] = []
        if units is not None or candidates is not None:
            self.units = [_normalize_unit_row(row) for row in (units or ())]
            self.candidates = [_normalize_candidate_row(row) for row in (candidates or ())]
            return
        if records is None:
            return
        if isinstance(records, (list, tuple)) and len(records) == 0:
            return
        other = self.from_value(records)
        self.units = other.units
        self.candidates = other.candidates

    @classmethod
    def from_value(cls, value: Any) -> "PreferenceDataset":
        if value is None:
            return cls()
        if isinstance(value, PreferenceDataset):
            return value
        if isinstance(value, PreferenceRecord):
            return cls.from_records((value,))
        if isinstance(value, Candidate):
            raise TypeError("a bare Candidate must be wrapped in a PreferenceRecord")
        if hasattr(value, "keys") and {"units", "candidates"} <= set(value.keys()):
            return cls.from_tables(value["units"], value["candidates"])
        if hasattr(value, "to_list") and callable(value.to_list):
            return cls.from_value(value.to_list())
        if isinstance(value, (str, Path)):
            return cls.load(value)
        if isinstance(value, MappingABC):
            if {"units", "candidates"} <= set(value.keys()):
                return cls.from_tables(value["units"], value["candidates"])
            rows = value.get("records", value.get("preference_records"))
            if rows is not None:
                return cls.from_value(rows)
            if _is_flat_candidate_mapping(value):
                return cls.from_flat_rows((value,))
            return cls.from_records((PreferenceRecord.from_mapping(value),))
        if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
            rows = list(value)
            if not rows:
                return cls()
            if all(isinstance(row, PreferenceRecord) for row in rows):
                return cls.from_records(rows)
            if all(isinstance(row, MappingABC) and _is_flat_candidate_mapping(row) for row in rows):
                return cls.from_flat_rows(rows)  # type: ignore[arg-type]
            return cls.from_records(
                [
                    row if isinstance(row, PreferenceRecord) else PreferenceRecord.from_mapping(row)
                    for row in rows
                    if isinstance(row, (PreferenceRecord, MappingABC))
                ]
            )
        raise TypeError(
            "preference_data must be a PreferenceDataset, DatasetDict, Dataset, "
            f"path, mapping, or sequence; got {type(value).__name__}"
        )

    @classmethod
    def from_records(cls, records: Sequence[PreferenceRecord | Mapping[str, Any]]) -> "PreferenceDataset":
        units: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        for item in records:
            record = item if isinstance(item, PreferenceRecord) else PreferenceRecord.from_mapping(item)
            unit = record.to_unit_row()
            units.append(unit)
            for candidate in record.candidates:
                candidates.append(candidate.to_row(record.unit_id))
        return cls(units=units, candidates=candidates)

    @classmethod
    def from_tables(cls, units: Any, candidates: Any) -> "PreferenceDataset":
        unit_rows = _rows_from_table(units)
        candidate_rows = _rows_from_table(candidates)
        return cls(units=unit_rows, candidates=candidate_rows)

    @classmethod
    def from_flat_rows(cls, rows: Sequence[Mapping[str, Any]]) -> "PreferenceDataset":
        units_by_id: dict[str, dict[str, Any]] = {}
        candidates: list[dict[str, Any]] = []
        for raw in rows:
            row = dict(raw)
            unit_id = str(row.get("unit_id") or row.get("node_id") or row.get("doc_id") or "")
            if not unit_id:
                unit_id = str(row.get("record_id") or row.get("id") or len(units_by_id))
            if unit_id not in units_by_id:
                units_by_id[unit_id] = _normalize_unit_row(
                    {
                        **{key: row.get(key) for key in _UNIT_FIELDS if key in row},
                        "unit_id": unit_id,
                        "unit_type": row.get("unit_type", row.get("kind", "unit")),
                        "target": row.get("target", "g"),
                        "context": _maybe_json(row.get("context_json", row.get("context", row.get("prompt", "")))),
                        "weight": row.get("weight", row.get("sample_weight", 1.0)),
                        "propensity": row.get("propensity", row.get("joint_propensity", 1.0)),
                        "metadata": _maybe_json(row.get("unit_metadata", row.get("metadata", {}))) or {},
                    }
                )
            candidates.append(
                _normalize_candidate_row(
                    {
                        "unit_id": unit_id,
                        "candidate_id": row.get("candidate_id", row.get("id", row.get("response_id", ""))),
                        "value": _maybe_json(row.get("value_json", row.get("value", row.get("response", "")))),
                        "score": row.get("score", row.get("reward")),
                        "rank": row.get("rank"),
                        "preferred": row.get("preferred", False),
                        "metadata": _maybe_json(row.get("candidate_metadata", row.get("metadata", {}))) or {},
                    }
                )
            )
        return cls(units=list(units_by_id.values()), candidates=candidates)

    def __len__(self) -> int:
        return len(self.units)

    def __iter__(self):
        return iter(self.to_records("general"))

    def append(self, record: PreferenceRecord | Mapping[str, Any]) -> "PreferenceDataset":
        other = PreferenceDataset.from_records((record,))
        self.units.extend(other.units)
        self.candidates.extend(other.candidates)
        return self

    def extend(self, records: Sequence[PreferenceRecord | Mapping[str, Any]]) -> "PreferenceDataset":
        for record in records:
            self.append(record)
        return self

    def sample(
        self,
        *,
        sample_size: int | None = None,
        sample_rate: float | None = None,
        seed: int = 0,
    ) -> "PreferenceDataset":
        if sample_size is not None and sample_rate is not None:
            raise ValueError("pass sample_size or sample_rate, not both")
        n = len(self.units)
        if n == 0:
            return PreferenceDataset()
        if sample_rate is not None:
            if sample_rate < 0.0 or sample_rate > 1.0:
                raise ValueError("sample_rate must be in [0, 1]")
            k = min(n, int(round(n * float(sample_rate))))
        elif sample_size is not None:
            if sample_size < 0:
                raise ValueError("sample_size must be non-negative")
            k = min(n, int(sample_size))
        else:
            k = n
        rng = random.Random(int(seed))
        indices = sorted(rng.sample(range(n), k)) if k < n else list(range(n))
        sampled_units = [self.units[idx] for idx in indices]
        sampled_ids = {str(unit["unit_id"]) for unit in sampled_units}
        sampled_candidates = [
            row for row in self.candidates if str(row.get("unit_id")) in sampled_ids
        ]
        return PreferenceDataset(units=sampled_units, candidates=sampled_candidates)

    def filter_target(self, target: PreferenceTarget) -> "PreferenceDataset":
        unit_rows = [
            row
            for row in self.units
            if row.get("target") == target or row.get("target") == "both" or target == "both"
        ]
        unit_ids = {str(row["unit_id"]) for row in unit_rows}
        candidate_rows = [
            row for row in self.candidates if str(row.get("unit_id")) in unit_ids
        ]
        return PreferenceDataset(units=unit_rows, candidates=candidate_rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "2.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_units": len(self.units),
            "n_candidates": len(self.candidates),
            "units": [dict(row) for row in self.units],
            "candidates": [dict(row) for row in self.candidates],
        }

    def summary(self) -> dict[str, Any]:
        return {
            "n_units": len(self.units),
            "n_candidates": len(self.candidates),
            "targets": sorted({str(row.get("target") or "") for row in self.units if row.get("target")}),
            "unit_types": sorted({str(row.get("unit_type") or "") for row in self.units if row.get("unit_type")}),
            "mean_sample_weight": _mean(
                _sample_weight(row.get("weight", 1.0), row.get("propensity", 1.0))
                for row in self.units
            ),
        }

    def to_hf_dataset_dict(self) -> Any:
        from datasets import Dataset, DatasetDict

        return DatasetDict(
            {
                "units": Dataset.from_list([_hf_unit_row(row) for row in self.units]),
                "candidates": Dataset.from_list(
                    [_hf_candidate_row(row) for row in self.candidates]
                ),
            }
        )

    def to_records(self, format: PreferenceFormat = "general") -> list[dict[str, Any]]:
        if format == "general":
            return [self._general_record(unit) for unit in self.units]
        if format == "supervised":
            return self._supervised_records()
        if format == "dpo":
            return [
                row
                for unit in self.units
                for row in self._pairwise_records(unit, format_name="dpo")
            ]
        if format == "reward":
            return [
                row
                for unit in self.units
                for row in self._pairwise_records(unit, format_name="reward")
            ]
        if format == "grpo":
            return [
                row
                for unit in self.units
                if (row := self._grpo_record(unit)) is not None
            ]
        raise ValueError("format must be 'general', 'supervised', 'dpo', 'reward', or 'grpo'")

    def save(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: Path | str) -> "PreferenceDataset":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_value(payload)

    def _unit_candidates(self, unit_id: str) -> list[dict[str, Any]]:
        return [
            row for row in self.candidates if str(row.get("unit_id")) == str(unit_id)
        ]

    def _general_record(self, unit: Mapping[str, Any]) -> dict[str, Any]:
        unit_id = str(unit["unit_id"])
        return {
            **dict(unit),
            "sample_weight": _sample_weight(unit.get("weight", 1.0), unit.get("propensity", 1.0)),
            "candidates": [
                _candidate_record(candidate)
                for candidate in self._unit_candidates(unit_id)
            ],
        }

    def _supervised_records(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for unit in self.units:
            for candidate in _top_candidates(self._unit_candidates(str(unit["unit_id"]))):
                rows.append(
                    {
                        "unit_id": unit["unit_id"],
                        "unit_type": unit.get("unit_type", "unit"),
                        "target": unit.get("target", "g"),
                        "prompt": _context_text(unit.get("context", "")),
                        "completion": _candidate_text(candidate),
                        "value": candidate.get("value"),
                        "candidate_id": candidate.get("candidate_id"),
                        "score": candidate.get("score"),
                        "rank": candidate.get("rank"),
                        "sample_weight": _sample_weight(unit.get("weight", 1.0), unit.get("propensity", 1.0)),
                        "metadata": _export_metadata(unit, candidate, format_name="supervised"),
                    }
                )
        return rows

    def _pairwise_records(self, unit: Mapping[str, Any], *, format_name: str) -> list[dict[str, Any]]:
        pairs = _pair_candidates(self._unit_candidates(str(unit["unit_id"])))
        rows: list[dict[str, Any]] = []
        for left, right in pairs:
            preferred_side = _preferred_side(left, right)
            if preferred_side == "tie":
                continue
            chosen, rejected = (left, right) if preferred_side == "A" else (right, left)
            row = {
                "prompt": _context_text(unit.get("context", "")),
                "chosen": _candidate_text(chosen),
                "rejected": _candidate_text(rejected),
                "sample_weight": _sample_weight(unit.get("weight", 1.0), unit.get("propensity", 1.0)),
                "metadata": _export_metadata(unit, chosen, rejected, format_name=format_name),
            }
            if format_name == "reward":
                chosen_score, rejected_score = _reward_scores(chosen, rejected)
                row["chosen_score"] = chosen_score
                row["rejected_score"] = rejected_score
            rows.append(row)
        return rows

    def _grpo_record(self, unit: Mapping[str, Any]) -> dict[str, Any] | None:
        candidates = _ordered_candidates(self._unit_candidates(str(unit["unit_id"])))
        if len(candidates) < 2:
            return None
        return {
            "prompt": _context_text(unit.get("context", "")),
            "responses": [_candidate_text(candidate) for candidate in candidates],
            "ranks": _grpo_ranks(candidates),
            "scores": [
                None if candidate.get("score") is None else float(candidate["score"])
                for candidate in candidates
            ],
            "sample_weight": _sample_weight(unit.get("weight", 1.0), unit.get("propensity", 1.0)),
            "metadata": _export_metadata(unit, format_name="grpo"),
        }


def _record_from_pairwise(row: Mapping[str, Any]) -> PreferenceRecord:
    response_a = row.get("response_a", row.get("summary_a", row.get("candidate_a", "")))
    response_b = row.get("response_b", row.get("summary_b", row.get("candidate_b", "")))
    preferred = row.get("preferred", row.get("winner", "tie"))
    preferred = {"a": "A", "left": "A", "chosen_a": "A", "b": "B", "right": "B", "chosen_b": "B"}.get(
        str(preferred),
        preferred,
    )
    candidate_a = Candidate(
        id="A",
        value=response_a,
        score=_optional_float(row.get("score_a", row.get("score_estimate_a"))),
        preferred=preferred == "A",
    )
    candidate_b = Candidate(
        id="B",
        value=response_b,
        score=_optional_float(row.get("score_b", row.get("score_estimate_b"))),
        preferred=preferred == "B",
    )
    if preferred == "tie":
        candidate_a = Candidate(id="A", value=response_a, score=candidate_a.score, rank=1)
        candidate_b = Candidate(id="B", value=response_b, score=candidate_b.score, rank=1)
    return PreferenceRecord(
        record_id=str(row.get("pair_id") or row.get("example_id") or row.get("id") or ""),
        unit_id=str(row.get("source_example_id") or row.get("unit_id") or row.get("doc_id") or row.get("pair_id") or ""),
        unit_type=str(row.get("unit_type") or "pair"),
        target=str(row.get("target") or "g"),  # type: ignore[arg-type]
        context=row.get("prompt") or _prompt_from_pair(row),
        candidates=(candidate_a, candidate_b),
        weight=float(row.get("weight", row.get("sample_weight", 1.0)) or 1.0),
        propensity=float(row.get("propensity", row.get("joint_propensity", 1.0)) or 1.0),
        metadata={
            "law_type": row.get("law_type", "preference"),
            "confidence": float(row.get("confidence", 1.0) or 1.0),
            "reasoning": str(row.get("reasoning") or ""),
            **dict(row.get("metadata") or {}),
        },
        doc_id=_optional_str(row.get("source_doc_id", row.get("doc_id"))),
    )


def _prompt_from_pair(row: Mapping[str, Any]) -> str:
    parts = []
    if row.get("rubric"):
        parts.append(f"Rubric:\n{row['rubric']}")
    if row.get("original_text") or row.get("text"):
        parts.append(f"Input:\n{row.get('original_text') or row.get('text')}")
    return "\n\n".join(parts)


__all__ = [
    "Candidate",
    "PreferenceDataset",
    "PreferenceFormat",
    "PreferenceRecord",
    "PreferenceTarget",
]
