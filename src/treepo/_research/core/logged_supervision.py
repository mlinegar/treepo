"""Canonical logged-supervision records for propensity-aware learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Dict, Generic, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple, TypeVar

from treepo._research.core.provenance import ORACLE_SOURCE, TruthLabelSource, normalize_truth_label_source

MIN_PROPENSITY = 1e-12
MAX_PROPENSITY = 1.0
DEFAULT_PROPENSITY = 1.0

DocT = TypeVar("DocT")
UnitT = TypeVar("UnitT")
LabelT = TypeVar("LabelT")


def _normalize_propensity(value: Optional[float], name: str) -> float:
    if value is None:
        return DEFAULT_PROPENSITY
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0 or parsed > MAX_PROPENSITY:
        raise ValueError(f"{name} must be finite and in (0, {MAX_PROPENSITY}], got {value!r}")
    return parsed


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_serialize(v) for v in value]
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    return value


class ObservationUnitKind(str, Enum):
    DOCUMENT = "document"
    LEAF = "leaf"
    INTERNAL = "internal"
    MERGE = "merge"
    RESUMMARY = "resummary"
    SUBSTITUTION = "substitution"
    PAIR = "pair"


@dataclass(frozen=True)
class SamplingMetadata:
    """Canonical sampling/IPW metadata shared across supervision lanes."""

    document_propensity: float = DEFAULT_PROPENSITY
    unit_propensity: float = DEFAULT_PROPENSITY
    label_propensity: float = DEFAULT_PROPENSITY
    joint_propensity: Optional[float] = None
    sampling_scheme: Optional[str] = None
    policy_name: Optional[str] = None
    unit_kind: Optional[ObservationUnitKind] = None
    supports_ipw_estimation: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_propensity",
            _normalize_propensity(self.document_propensity, "document_propensity"),
        )
        object.__setattr__(
            self,
            "unit_propensity",
            _normalize_propensity(self.unit_propensity, "unit_propensity"),
        )
        object.__setattr__(
            self,
            "label_propensity",
            _normalize_propensity(self.label_propensity, "label_propensity"),
        )
        if self.joint_propensity is not None:
            object.__setattr__(
                self,
                "joint_propensity",
                _normalize_propensity(self.joint_propensity, "joint_propensity"),
            )
        if self.unit_kind is not None and not isinstance(self.unit_kind, ObservationUnitKind):
            object.__setattr__(self, "unit_kind", ObservationUnitKind(str(self.unit_kind)))
        if not isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", dict(self.metadata))

    def effective_joint_propensity(self, *, min_propensity: float = MIN_PROPENSITY) -> float:
        if self.joint_propensity is not None:
            joint = float(self.joint_propensity)
        else:
            joint = (
                float(self.document_propensity)
                * float(self.unit_propensity)
                * float(self.label_propensity)
            )
        return max(float(min_propensity), float(joint))

    def ipw_weight(
        self,
        *,
        min_propensity: float = MIN_PROPENSITY,
        max_weight: Optional[float] = None,
    ) -> float:
        weight = 1.0 / self.effective_joint_propensity(min_propensity=min_propensity)
        if max_weight is not None:
            weight = min(weight, float(max_weight))
        return float(weight)

    def with_updates(self, **kwargs: Any) -> "SamplingMetadata":
        return replace(self, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SamplingMetadata":
        data = dict(payload or {})
        unit_kind = data.get("unit_kind")
        if unit_kind in ("", None):
            unit_kind = None
        return cls(
            document_propensity=data.get(
                "document_propensity",
                data.get("doc_propensity", DEFAULT_PROPENSITY),
            ),
            unit_propensity=data.get(
                "unit_propensity",
                data.get("node_propensity", DEFAULT_PROPENSITY),
            ),
            label_propensity=data.get(
                "label_propensity",
                data.get("label_propensity", DEFAULT_PROPENSITY),
            ),
            joint_propensity=data.get("joint_propensity"),
            sampling_scheme=data.get("sampling_scheme"),
            policy_name=data.get("policy_name"),
            unit_kind=unit_kind,
            supports_ipw_estimation=bool(data.get("supports_ipw_estimation", True)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class LoggedLabelObservation(Generic[LabelT]):
    """Canonical realized supervision record."""

    observation_id: str
    document_id: str
    unit_id: str
    unit_kind: ObservationUnitKind
    target_name: str
    label: LabelT
    truth_label_source: TruthLabelSource = ORACLE_SOURCE
    sampling: SamplingMetadata = field(default_factory=SamplingMetadata)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", str(self.document_id))
        object.__setattr__(self, "unit_id", str(self.unit_id))
        object.__setattr__(self, "target_name", str(self.target_name))
        object.__setattr__(
            self,
            "truth_label_source",
            normalize_truth_label_source(self.truth_label_source),
        )
        if not isinstance(self.unit_kind, ObservationUnitKind):
            object.__setattr__(self, "unit_kind", ObservationUnitKind(str(self.unit_kind)))
        if not isinstance(self.sampling, SamplingMetadata):
            object.__setattr__(self, "sampling", SamplingMetadata.from_dict(self.sampling))
        if self.sampling.unit_kind is None:
            object.__setattr__(
                self,
                "sampling",
                self.sampling.with_updates(unit_kind=self.unit_kind),
            )
        if not isinstance(self.context, dict):
            object.__setattr__(self, "context", dict(self.context))

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(
            {
                "observation_id": self.observation_id,
                "document_id": self.document_id,
                "unit_id": self.unit_id,
                "unit_kind": self.unit_kind,
                "target_name": self.target_name,
                "label": self.label,
                "truth_label_source": self.truth_label_source,
                "sampling": self.sampling.to_dict(),
                "context": self.context,
                "timestamp": self.timestamp,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LoggedLabelObservation[Any]":
        data = dict(payload or {})
        sampling = data.get("sampling")
        if not isinstance(sampling, Mapping):
            sampling = {
                "joint_propensity": data.get("joint_propensity"),
                "sampling_scheme": data.get("sampling_scheme"),
                "unit_kind": data.get("unit_kind"),
            }
        return cls(
            observation_id=str(data.get("observation_id", "")),
            document_id=str(data.get("document_id", "")),
            unit_id=str(data.get("unit_id", data.get("document_id", ""))),
            unit_kind=ObservationUnitKind(str(data.get("unit_kind", ObservationUnitKind.DOCUMENT.value))),
            target_name=str(data.get("target_name", "label")),
            label=data.get("label"),
            truth_label_source=data.get("truth_label_source", ORACLE_SOURCE),
            sampling=SamplingMetadata.from_dict(sampling),
            context=dict(data.get("context", {}) or {}),
            timestamp=str(
                data.get("timestamp", datetime.now(timezone.utc).isoformat())
            ),
        )


@dataclass(frozen=True)
class LoggedObservationArtifact:
    """Manifest entry for a persisted logged-observation sidecar."""

    artifact_id: str
    channel_name: str
    format: str
    path: str
    count: int
    unit_kinds: Tuple[ObservationUnitKind, ...] = field(default_factory=tuple)
    propensity_fields_logged: Tuple[str, ...] = field(default_factory=tuple)
    supports_ipw_estimation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kinds = tuple(
            kind if isinstance(kind, ObservationUnitKind) else ObservationUnitKind(str(kind))
            for kind in self.unit_kinds
        )
        object.__setattr__(self, "unit_kinds", kinds)
        if not isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(
            {
                "artifact_id": self.artifact_id,
                "channel_name": self.channel_name,
                "format": self.format,
                "path": self.path,
                "count": int(self.count),
                "unit_kinds": self.unit_kinds,
                "propensity_fields_logged": tuple(self.propensity_fields_logged),
                "supports_ipw_estimation": bool(self.supports_ipw_estimation),
                "metadata": self.metadata,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LoggedObservationArtifact":
        data = dict(payload or {})
        return cls(
            artifact_id=str(data.get("artifact_id", "")),
            channel_name=str(data.get("channel_name", "")),
            format=str(data.get("format", "jsonl")),
            path=str(data.get("path", "")),
            count=int(data.get("count", 0) or 0),
            unit_kinds=tuple(
                ObservationUnitKind(str(kind))
                for kind in (data.get("unit_kinds", []) or [])
            ),
            propensity_fields_logged=tuple(
                str(name) for name in (data.get("propensity_fields_logged", []) or [])
            ),
            supports_ipw_estimation=bool(data.get("supports_ipw_estimation", False)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


class SamplingPolicy(Protocol, Generic[DocT, UnitT]):
    """Adapter that selects units and logs realized sampling metadata."""

    def sample_units(
        self,
        document: DocT,
        *,
        rng: Optional[Any] = None,
    ) -> Sequence[Tuple[str, UnitT, SamplingMetadata, Dict[str, Any]]]:
        ...


class OracleLabeler(Protocol, Generic[DocT, UnitT, LabelT]):
    """Adapter that returns a label for a sampled unit."""

    def label_unit(
        self,
        document: DocT,
        unit: UnitT,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> LabelT:
        ...


def collect_logged_observations(
    document: DocT,
    *,
    document_id: str,
    target_name: str,
    sampling_policy: SamplingPolicy[DocT, UnitT],
    oracle_labeler: OracleLabeler[DocT, UnitT, LabelT],
    truth_label_source: TruthLabelSource = ORACLE_SOURCE,
    rng: Optional[Any] = None,
) -> List[LoggedLabelObservation[LabelT]]:
    """Collect realized logged observations under a shared policy interface."""

    observations: List[LoggedLabelObservation[LabelT]] = []
    for unit_id, unit, sampling, context in sampling_policy.sample_units(document, rng=rng):
        observation_id = f"{document_id}:{target_name}:{unit_id}"
        label = oracle_labeler.label_unit(document, unit, context=context)
        observations.append(
            LoggedLabelObservation(
                observation_id=observation_id,
                document_id=document_id,
                unit_id=str(unit_id),
                unit_kind=sampling.unit_kind or ObservationUnitKind.DOCUMENT,
                target_name=target_name,
                label=label,
                truth_label_source=truth_label_source,
                sampling=sampling,
                context=dict(context or {}),
            )
        )
    return observations


def write_logged_observations_jsonl(
    path: Path,
    observations: Iterable[LoggedLabelObservation[Any]],
    *,
    channel_name: Optional[str] = None,
) -> LoggedObservationArtifact:
    rows = [observation.to_dict() for observation in observations]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    unit_kinds = tuple(
        dict.fromkeys(
            ObservationUnitKind(str(row["unit_kind"]))
            for row in rows
            if row.get("unit_kind") is not None
        )
    )
    supports_ipw = any(
        bool(dict(row.get("sampling", {}) or {}).get("supports_ipw_estimation", False))
        for row in rows
    )
    return LoggedObservationArtifact(
        artifact_id=path.stem,
        channel_name=str(channel_name or path.stem),
        format="jsonl",
        path=str(path),
        count=len(rows),
        unit_kinds=unit_kinds,
        propensity_fields_logged=(
            "document_propensity",
            "unit_propensity",
            "label_propensity",
            "joint_propensity",
        ),
        supports_ipw_estimation=supports_ipw,
    )


def read_logged_observations_jsonl(path: Path) -> List[LoggedLabelObservation[Any]]:
    observations: List[LoggedLabelObservation[Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            observations.append(LoggedLabelObservation.from_dict(json.loads(stripped)))
    return observations


def summarize_logged_observations(
    observations: Iterable[LoggedLabelObservation[Any]],
) -> Dict[str, Any]:
    rows = list(observations)
    propensities = [row.sampling.effective_joint_propensity() for row in rows] or [1.0]
    return {
        "count": len(rows),
        "unit_kinds": sorted({row.unit_kind.value for row in rows}),
        "supports_ipw_estimation": any(
            bool(row.sampling.supports_ipw_estimation) for row in rows
        ),
        "joint_propensity_min": float(min(propensities)),
        "joint_propensity_max": float(max(propensities)),
        "joint_propensity_mean": float(sum(propensities) / max(1, len(propensities))),
    }


__all__ = [
    "DEFAULT_PROPENSITY",
    "MAX_PROPENSITY",
    "MIN_PROPENSITY",
    "LoggedLabelObservation",
    "LoggedObservationArtifact",
    "ObservationUnitKind",
    "OracleLabeler",
    "SamplingMetadata",
    "SamplingPolicy",
    "collect_logged_observations",
    "read_logged_observations_jsonl",
    "summarize_logged_observations",
    "write_logged_observations_jsonl",
]
