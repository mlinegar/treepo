from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = "treepo.run_manifest.v1"
MIN_PROPENSITY = 1e-12


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def stable_digest(payload: Mapping[str, Any]) -> str:
    rendered = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _finite_float(value: float, *, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    unit: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.start) < 0:
            raise ValueError(f"span start must be non-negative, got {self.start!r}")
        if int(self.end) < int(self.start):
            raise ValueError(f"span end must be >= start, got {self.start!r}, {self.end!r}")
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "end", int(self.end))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def valid_for_length(self, length: int) -> bool:
        return 0 <= int(self.start) <= int(self.end) <= int(length)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Span":
        data = dict(payload or {})
        return cls(
            start=int(data.get("start", 0)),
            end=int(data.get("end", 0)),
            unit=str(data.get("unit") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class TopLevelUnit:
    unit_id: str
    length: int
    source_ref: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.unit_id):
            raise ValueError("top-level unit_id is required")
        if int(self.length) <= 0:
            raise ValueError(f"top-level unit length must be positive, got {self.length!r}")
        object.__setattr__(self, "unit_id", str(self.unit_id))
        object.__setattr__(self, "length", int(self.length))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TopLevelUnit":
        data = dict(payload or {})
        return cls(
            unit_id=str(data.get("unit_id") or data.get("id") or ""),
            length=int(data.get("length", 0)),
            source_ref=str(data.get("source_ref") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    kind: str = ""
    uri: str = ""
    digest: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.artifact_id):
            raise ValueError("artifact_id is required")
        object.__setattr__(self, "artifact_id", str(self.artifact_id))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | str) -> "ArtifactRef":
        if isinstance(payload, str):
            return cls(artifact_id=payload)
        data = dict(payload or {})
        return cls(
            artifact_id=str(data.get("artifact_id") or data.get("id") or ""),
            kind=str(data.get("kind") or ""),
            uri=str(data.get("uri") or ""),
            digest=str(data.get("digest") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RoleTuple:
    chunker: str = "eval"
    g: str = "eval"
    oracle: str = "eval"

    def __post_init__(self) -> None:
        for name in ("chunker", "g", "oracle"):
            value = str(getattr(self, name) or "")
            if not value:
                raise ValueError(f"{name} role is required")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, str]:
        return {"chunker": self.chunker, "g": self.g, "oracle": self.oracle}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "RoleTuple":
        data = dict(payload or {})
        return cls(
            chunker=str(data.get("chunker") or data.get("chunk") or "eval"),
            g=str(data.get("g") or data.get("summarizer") or "eval"),
            oracle=str(data.get("oracle") or "eval"),
        )


@dataclass(frozen=True)
class ArtifactLineage:
    chunker: str
    g: str
    f: str
    oracle_online: str
    oracle_eval: str
    query_policy: str
    proxy: str = ""

    def __post_init__(self) -> None:
        for name in ("chunker", "g", "f", "oracle_online", "oracle_eval", "query_policy"):
            value = str(getattr(self, name) or "")
            if not value:
                raise ValueError(f"{name} artifact id is required")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "proxy", str(self.proxy or ""))

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactLineage":
        data = dict(payload or {})
        return cls(
            chunker=str(data.get("chunker") or ""),
            g=str(data.get("g") or ""),
            f=str(data.get("f") or ""),
            oracle_online=str(data.get("oracle_online") or data.get("oracleOnline") or ""),
            oracle_eval=str(data.get("oracle_eval") or data.get("oracleEval") or ""),
            query_policy=str(data.get("query_policy") or data.get("queryPolicy") or ""),
            proxy=str(data.get("proxy") or ""),
        )


@dataclass(frozen=True)
class ManifestRow:
    row_id: str
    top_level_unit_id: str
    source_unit_id: str = ""
    fold_id: str = "0"
    split_seed: int = 0
    roles: RoleTuple = field(default_factory=RoleTuple)
    artifacts: ArtifactLineage | None = None
    law_kind: str = ""
    support: Span = field(default_factory=lambda: Span(0, 0))
    node_id: str = ""
    pair_id: str = ""
    observed: bool = False
    propensity: float = 0.0
    effective_propensity: float | None = None
    influence_weight: float | None = None
    truth_source: str = ""
    approx_source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.row_id):
            raise ValueError("manifest row_id is required")
        if not str(self.top_level_unit_id):
            raise ValueError("manifest row top_level_unit_id is required")
        roles = self.roles if isinstance(self.roles, RoleTuple) else RoleTuple.from_dict(self.roles)
        support = self.support if isinstance(self.support, Span) else Span.from_dict(self.support)
        artifacts = (
            self.artifacts
            if self.artifacts is None or isinstance(self.artifacts, ArtifactLineage)
            else ArtifactLineage.from_dict(self.artifacts)
        )
        propensity = _finite_float(self.propensity, name="propensity")
        if propensity < 0.0 or propensity > 1.0:
            raise ValueError(f"propensity must be in [0, 1], got {self.propensity!r}")
        if bool(self.observed) and propensity <= 0.0:
            raise ValueError("observed manifest rows require positive propensity")
        effective = (
            max(float(propensity), MIN_PROPENSITY)
            if self.effective_propensity is None
            else _finite_float(self.effective_propensity, name="effective_propensity")
        )
        if effective <= 0.0 or effective > 1.0:
            raise ValueError(f"effective_propensity must be in (0, 1], got {effective!r}")
        influence = (
            1.0 / effective
            if self.influence_weight is None
            else _finite_float(self.influence_weight, name="influence_weight")
        )
        object.__setattr__(self, "row_id", str(self.row_id))
        object.__setattr__(self, "top_level_unit_id", str(self.top_level_unit_id))
        object.__setattr__(
            self,
            "source_unit_id",
            str(self.source_unit_id or self.top_level_unit_id),
        )
        object.__setattr__(self, "fold_id", str(self.fold_id))
        object.__setattr__(self, "split_seed", int(self.split_seed))
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "propensity", float(propensity))
        object.__setattr__(self, "effective_propensity", float(effective))
        object.__setattr__(self, "influence_weight", float(influence))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def with_artifacts(self, artifacts: ArtifactLineage) -> "ManifestRow":
        return replace(self, artifacts=artifacts)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ManifestRow":
        data = dict(payload or {})
        return cls(
            row_id=str(data.get("row_id") or data.get("rowId") or ""),
            top_level_unit_id=str(data.get("top_level_unit_id") or data.get("topLevelUnit") or ""),
            source_unit_id=str(data.get("source_unit_id") or data.get("sourceUnit") or ""),
            fold_id=str(data.get("fold_id") or data.get("foldId") or "0"),
            split_seed=int(data.get("split_seed", data.get("splitSeed", 0)) or 0),
            roles=RoleTuple.from_dict(data.get("roles") or {}),
            artifacts=(
                None
                if data.get("artifacts") in (None, "")
                else ArtifactLineage.from_dict(data.get("artifacts") or {})
            ),
            law_kind=str(data.get("law_kind") or data.get("lawKind") or ""),
            support=Span.from_dict(data.get("support") or {}),
            node_id=str(data.get("node_id") or data.get("nodeId") or ""),
            pair_id=str(data.get("pair_id") or data.get("pairId") or ""),
            observed=bool(data.get("observed", False)),
            propensity=float(data.get("propensity", 0.0) or 0.0),
            effective_propensity=(
                None
                if data.get("effective_propensity", data.get("effectivePropensity")) is None
                else float(data.get("effective_propensity", data.get("effectivePropensity")))
            ),
            influence_weight=(
                None
                if data.get("influence_weight", data.get("influenceWeight")) is None
                else float(data.get("influence_weight", data.get("influenceWeight")))
            ),
            truth_source=str(data.get("truth_source") or data.get("truthSource") or ""),
            approx_source=str(data.get("approx_source") or data.get("approxSource") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ManifestValidationReport:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"ok": bool(self.ok), "errors": list(self.errors), "warnings": list(self.warnings)}


@dataclass(frozen=True)
class RunManifestContract:
    run_id: str
    top_level_units: tuple[TopLevelUnit, ...] = ()
    rows: tuple[ManifestRow, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    schema_version: str = MANIFEST_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "top_level_units",
            tuple(
                unit if isinstance(unit, TopLevelUnit) else TopLevelUnit.from_dict(unit)
                for unit in self.top_level_units
            ),
        )
        object.__setattr__(
            self,
            "rows",
            tuple(row if isinstance(row, ManifestRow) else ManifestRow.from_dict(row) for row in self.rows),
        )
        object.__setattr__(
            self,
            "artifacts",
            tuple(
                art if isinstance(art, ArtifactRef) else ArtifactRef.from_dict(art)
                for art in self.artifacts
            ),
        )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def digest(self) -> str:
        return stable_digest(self.to_dict())

    def validate(self, *, require_artifacts: bool = True) -> ManifestValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        unit_by_id = {unit.unit_id: unit for unit in self.top_level_units}
        if len(unit_by_id) != len(self.top_level_units):
            errors.append("top_level_units must have unique unit_id values")
        artifact_ids = {artifact.artifact_id for artifact in self.artifacts}
        row_ids: set[str] = set()
        for row in self.rows:
            if row.row_id in row_ids:
                errors.append(f"duplicate manifest row_id: {row.row_id}")
            row_ids.add(row.row_id)
            unit = unit_by_id.get(row.top_level_unit_id)
            if unit is None:
                errors.append(f"row {row.row_id} references missing top-level unit {row.top_level_unit_id}")
                continue
            if row.source_unit_id and row.source_unit_id not in unit_by_id:
                errors.append(f"row {row.row_id} references missing source unit {row.source_unit_id}")
            if not row.support.valid_for_length(unit.length):
                errors.append(f"row {row.row_id} support span is outside unit {unit.unit_id}")
            if require_artifacts:
                if row.artifacts is None:
                    errors.append(f"row {row.row_id} is missing artifact lineage")
                else:
                    for artifact_id in row.artifacts.to_dict().values():
                        if artifact_id and isinstance(artifact_id, str) and artifact_ids and artifact_id not in artifact_ids:
                            warnings.append(f"row {row.row_id} artifact {artifact_id} is not listed in artifacts")
        return ManifestValidationReport(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "top_level_units": [unit.to_dict() for unit in self.top_level_units],
            "rows": [row.to_dict() for row in self.rows],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metadata": _jsonable(dict(self.metadata or {})),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunManifestContract":
        data = dict(payload or {})
        return cls(
            schema_version=str(data.get("schema_version") or MANIFEST_SCHEMA_VERSION),
            run_id=str(data.get("run_id") or ""),
            top_level_units=tuple(
                TopLevelUnit.from_dict(item) for item in list(data.get("top_level_units") or [])
            ),
            rows=tuple(ManifestRow.from_dict(item) for item in list(data.get("rows") or [])),
            artifacts=tuple(ArtifactRef.from_dict(item) for item in list(data.get("artifacts") or [])),
            metadata=dict(data.get("metadata") or {}),
        )


def manifest_digest(manifest: RunManifestContract | Mapping[str, Any]) -> str:
    payload = manifest.to_dict() if isinstance(manifest, RunManifestContract) else dict(manifest)
    return stable_digest(payload)


__all__ = [
    "ArtifactLineage",
    "ArtifactRef",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestRow",
    "ManifestValidationReport",
    "RoleTuple",
    "RunManifestContract",
    "Span",
    "TopLevelUnit",
    "manifest_digest",
    "stable_digest",
]
