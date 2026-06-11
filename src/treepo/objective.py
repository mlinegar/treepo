from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from treepo.common import finite_float


OBJECTIVE_SCHEMA_VERSION = "treepo.objective.v1"
THEOREM_BOUND_LIMITATION = (
    "v0.1 objective metadata records root/local-law training weights only. "
    "Lipschitz readout and measurement-error constants must be included by "
    "callers in supplied certificate radii; first-class components are v0.2 work."
)
OBJECTIVE_TERM_ROOT = "root"
OBJECTIVE_TERM_LOCAL_LAW_CORRECTED = "local_law_corrected"
LOCAL_LAW_ESTIMATOR_NONE = "none"
LOCAL_LAW_ESTIMATOR_CORRECTED = "corrected"
LOCAL_LAW_ESTIMATOR_PROXY_ONLY = "proxy_only"
LOCAL_LAW_ESTIMATOR_ORACLE_EXACT = "oracle_exact"
CANONICAL_LAW_COMPONENTS = (
    "leaf_preservation",
    "on_range_idempotence",
    "merge_preservation",
)
_ALLOWED_LOCAL_LAW_ESTIMATORS = {
    LOCAL_LAW_ESTIMATOR_NONE,
    LOCAL_LAW_ESTIMATOR_CORRECTED,
    LOCAL_LAW_ESTIMATOR_PROXY_ONLY,
    LOCAL_LAW_ESTIMATOR_ORACLE_EXACT,
}
_LEGACY_PUBLIC_FIELDS = {
    "gap_weight",
    "oracle_gap_weight",
    "lambda_eff",
    "reliability",
}
_LEGACY_TERM_NAMES = {"oracle_gap", "gap", "f_star_gap"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _stable_digest(payload: Mapping[str, Any]) -> str:
    rendered = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def canonical_law_component_weights(weights: Mapping[str, float]) -> dict[str, float]:
    aliases = {
        "c1": "leaf_preservation",
        "l1": "leaf_preservation",
        "leaf": "leaf_preservation",
        "c2": "on_range_idempotence",
        "l3": "on_range_idempotence",
        "idempotence": "on_range_idempotence",
        "c3": "merge_preservation",
        "l2": "merge_preservation",
        "merge": "merge_preservation",
    }
    out = {name: 0.0 for name in CANONICAL_LAW_COMPONENTS}
    for raw_name, raw_weight in dict(weights or {}).items():
        name = aliases.get(str(raw_name).strip().lower(), str(raw_name).strip().lower())
        if name not in out:
            raise ValueError(f"unknown local-law component weight: {raw_name!r}")
        out[name] = float(raw_weight)
    return out


@dataclass(frozen=True)
class ObjectiveSpec:
    schema_version: str = OBJECTIVE_SCHEMA_VERSION
    objective_family: str = "root_only"
    local_law_estimator: str = LOCAL_LAW_ESTIMATOR_NONE
    local_law_weight: float | None = None
    root_share: float = 1.0
    local_law_component_weights: Mapping[str, float] = field(default_factory=dict)
    terms: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    allow_nonconvex_objective: bool = False

    def __post_init__(self) -> None:
        estimator = str(self.local_law_estimator or "").strip().lower()
        if estimator == "aipw":
            estimator = LOCAL_LAW_ESTIMATOR_CORRECTED
        if estimator == "oracle":
            estimator = LOCAL_LAW_ESTIMATOR_ORACLE_EXACT
        if estimator not in _ALLOWED_LOCAL_LAW_ESTIMATORS:
            raise ValueError(f"unsupported local_law_estimator: {estimator!r}")
        object.__setattr__(self, "local_law_estimator", estimator)
        root_share = finite_float(self.root_share, name="root_share")
        if root_share < 0.0 or root_share > 1.0:
            raise ValueError("root_share must be in [0, 1]")
        weights = canonical_law_component_weights(self.local_law_component_weights or {})
        for name, value in weights.items():
            weight = finite_float(value, name=f"local_law_component_weights[{name}]")
            if weight < 0.0:
                raise ValueError("local-law component weights must be non-negative")
        local_weight = (
            finite_float(self.local_law_weight, name="local_law_weight")
            if self.local_law_weight is not None
            else float(sum(weights.values()))
        )
        if local_weight < 0.0:
            raise ValueError("local_law_weight must be non-negative")
        enabled = estimator != LOCAL_LAW_ESTIMATOR_NONE
        has_positive_component = any(float(v) > 0.0 for v in weights.values())
        if enabled and (local_weight <= 0.0 or not has_positive_component):
            raise ValueError(
                "local-law estimator enabled requires at least one positive law component"
            )
        if not enabled and (local_weight > 0.0 or has_positive_component):
            raise ValueError("local-law weights require a non-none local_law_estimator")
        if not bool(self.allow_nonconvex_objective) and not abs(root_share + local_weight - 1.0) <= 1e-9:
            raise ValueError(
                "objective weights must form a convex combination; set "
                "allow_nonconvex_objective=True to opt into non-convex weighting"
            )
        object.__setattr__(self, "root_share", root_share)
        object.__setattr__(self, "local_law_weight", local_weight)
        object.__setattr__(self, "local_law_component_weights", weights)
        object.__setattr__(self, "allow_nonconvex_objective", bool(self.allow_nonconvex_objective))

    def _normalized_terms(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for raw_name, raw_payload in dict(self.terms or {}).items():
            name = str(raw_name).strip().lower()
            if name in _LEGACY_TERM_NAMES:
                raise ValueError(
                    f"objective term {raw_name!r} is evidence metadata, not a training term"
                )
            if name not in {OBJECTIVE_TERM_ROOT, OBJECTIVE_TERM_LOCAL_LAW_CORRECTED}:
                raise ValueError(
                    f"objective term {raw_name!r} is not public; use 'root' and "
                    "'local_law_corrected'"
                )
            out[name] = dict(raw_payload or {})
        weights = canonical_law_component_weights(self.local_law_component_weights or {})
        local_weight = (
            float(self.local_law_weight)
            if self.local_law_weight is not None
            else float(sum(weights.values()))
        )
        out.setdefault(OBJECTIVE_TERM_ROOT, {})
        out.setdefault(OBJECTIVE_TERM_LOCAL_LAW_CORRECTED, {})
        out[OBJECTIVE_TERM_ROOT].setdefault("weight", float(self.root_share))
        out[OBJECTIVE_TERM_ROOT].setdefault("metric", "root_loss")
        out[OBJECTIVE_TERM_LOCAL_LAW_CORRECTED].setdefault("weight", local_weight)
        out[OBJECTIVE_TERM_LOCAL_LAW_CORRECTED].setdefault("estimator", self.local_law_estimator)
        out[OBJECTIVE_TERM_LOCAL_LAW_CORRECTED].setdefault("component_weights", weights)
        return out

    def to_dict(self) -> dict[str, Any]:
        weights = canonical_law_component_weights(self.local_law_component_weights or {})
        local_weight = (
            float(self.local_law_weight)
            if self.local_law_weight is not None
            else float(sum(weights.values()))
        )
        return {
            "schema_version": str(self.schema_version or OBJECTIVE_SCHEMA_VERSION),
            "objective_family": str(self.objective_family or "root_only"),
            "local_law_estimator": str(self.local_law_estimator),
            "local_law_weight": float(local_weight),
            "root_share": float(self.root_share),
            "local_law_component_weights": weights,
            "terms": _jsonable(self._normalized_terms()),
            "metadata": _jsonable(dict(self.metadata or {})),
            "allow_nonconvex_objective": bool(self.allow_nonconvex_objective),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ObjectiveSpec":
        data = dict(payload or {})
        nested = data.get("objective") or data.get("objective_spec")
        if isinstance(nested, Mapping):
            merged = dict(data)
            merged.update(dict(nested))
            data = merged
        legacy_fields = sorted(key for key in _LEGACY_PUBLIC_FIELDS if key in data)
        if legacy_fields:
            raise ValueError(
                "legacy public objective fields are not supported: "
                + ", ".join(legacy_fields)
            )
        terms = dict(data.get("terms") or {})
        for name in terms:
            if str(name).strip().lower() in _LEGACY_TERM_NAMES:
                raise ValueError("oracle_gap belongs in evidence metadata, not objective terms")
        weights = canonical_law_component_weights(dict(data.get("local_law_component_weights") or {}))
        estimator = str(
            data.get("local_law_estimator")
            or (LOCAL_LAW_ESTIMATOR_NONE if not any(weights.values()) else LOCAL_LAW_ESTIMATOR_CORRECTED)
        )
        return cls(
            schema_version=str(data.get("schema_version") or OBJECTIVE_SCHEMA_VERSION),
            objective_family=str(data.get("objective_family") or data.get("name") or "root_only"),
            local_law_estimator=estimator,
            local_law_weight=(
                None if data.get("local_law_weight") is None else float(data.get("local_law_weight"))
            ),
            root_share=float(data.get("root_share", 1.0)),
            local_law_component_weights=weights,
            terms=terms,
            metadata=dict(data.get("metadata") or {}),
            allow_nonconvex_objective=bool(data.get("allow_nonconvex_objective", False)),
        )

    @property
    def digest(self) -> str:
        return _stable_digest(self.to_dict())


def normalize_objective_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
    return ObjectiveSpec.from_mapping(payload).to_dict()


def objective_spec_digest(payload: Mapping[str, Any]) -> str:
    return _stable_digest(normalize_objective_spec(payload))


def objective_metadata(**kwargs: Any) -> dict[str, Any]:
    return ObjectiveSpec(**kwargs).to_dict()


__all__ = [
    "CANONICAL_LAW_COMPONENTS",
    "LOCAL_LAW_ESTIMATOR_CORRECTED",
    "LOCAL_LAW_ESTIMATOR_NONE",
    "LOCAL_LAW_ESTIMATOR_ORACLE_EXACT",
    "LOCAL_LAW_ESTIMATOR_PROXY_ONLY",
    "OBJECTIVE_SCHEMA_VERSION",
    "OBJECTIVE_TERM_LOCAL_LAW_CORRECTED",
    "OBJECTIVE_TERM_ROOT",
    "ObjectiveSpec",
    "THEOREM_BOUND_LIMITATION",
    "canonical_law_component_weights",
    "normalize_objective_spec",
    "objective_metadata",
    "objective_spec_digest",
]
