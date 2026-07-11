"""Training objective specification and canonical digesting.

Defines ``ObjectiveSpec`` (the convex root / local-law objective contract),
its normalization and validation rules, and content-addressed digests over
normalized specs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from treepo.common import finite_float, jsonable, stable_digest
from treepo.local_law import LawKind


OBJECTIVE_SCHEMA_VERSION = "treepo.objective.v1"
OBJECTIVE_TERM_ROOT = "root"
OBJECTIVE_TERM_LOCAL_LAW_CORRECTED = "local_law_corrected"
LOCAL_LAW_ESTIMATOR_NONE = "none"
LOCAL_LAW_ESTIMATOR_CORRECTED = "corrected"
LOCAL_LAW_ESTIMATOR_PROXY_ONLY = "proxy_only"
LOCAL_LAW_ESTIMATOR_ORACLE_EXACT = "oracle_exact"
LOCAL_LAW_ESTIMATOR_ORACLE_STATE = "oracle_state"
LOCAL_LAW_ESTIMATOR_EXTERNAL_PASSTHROUGH = "external_passthrough"
CANONICAL_LAW_COMPONENTS = tuple(kind.value for kind in LawKind)
_ALLOWED_LOCAL_LAW_ESTIMATORS = {
    LOCAL_LAW_ESTIMATOR_NONE,
    LOCAL_LAW_ESTIMATOR_CORRECTED,
    LOCAL_LAW_ESTIMATOR_PROXY_ONLY,
    LOCAL_LAW_ESTIMATOR_ORACLE_EXACT,
    LOCAL_LAW_ESTIMATOR_ORACLE_STATE,
    LOCAL_LAW_ESTIMATOR_EXTERNAL_PASSTHROUGH,
}
_UNSUPPORTED_PUBLIC_FIELDS = {
    "gap_weight",
    "oracle_gap_weight",
    "lambda_eff",
    "reliability",
}
_EVIDENCE_ONLY_TERM_NAMES = {"oracle_gap", "gap", "f_star_gap"}


def canonical_law_component_weights(weights: Mapping[str, float]) -> dict[str, float]:
    out = {name: 0.0 for name in CANONICAL_LAW_COMPONENTS}
    for raw_name, raw_weight in dict(weights or {}).items():
        try:
            name = LawKind.from_value(str(raw_name)).value
        except ValueError:
            raise ValueError(f"unknown local-law component weight: {raw_name!r}") from None
        out[name] = float(raw_weight)
    return out


def _canonical_law_component_name(value: str) -> str:
    weights = canonical_law_component_weights({str(value): 1.0})
    active = [name for name, weight in weights.items() if float(weight) > 0.0]
    if len(active) != 1:
        raise ValueError(f"unknown local-law component: {value!r}")
    return active[0]


def _finite_nonnegative(value: object, *, name: str) -> float:
    out = finite_float(value, name=name)
    if out < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return out


def _finite_probability(value: object, *, name: str) -> float:
    out = finite_float(value, name=name)
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return out


@dataclass(frozen=True)
class ResolvedObjectiveWeights:
    input_mode: str
    weighting_scheme: str
    root_share: float
    local_law_shares: Mapping[str, float] = field(default_factory=dict)

    @property
    def local_law_weight(self) -> float:
        return float(sum(float(v) for v in dict(self.local_law_shares).values()))

    def as_metadata(self) -> dict[str, Any]:
        return {
            "objective_input_mode": str(self.input_mode),
            "weighting_scheme": str(self.weighting_scheme),
            "root_share": float(self.root_share),
            "local_law_weight": float(self.local_law_weight),
            "local_law_shares": {
                str(k): float(v) for k, v in dict(self.local_law_shares).items()
            },
        }


def resolve_root_local_objective_weights(
    *,
    local_law_weight: float | None,
    active_laws: tuple[str, ...] | list[str],
    explicit_root_weight: float | None = None,
    explicit_law_weights: Mapping[str, float] | None = None,
    objective_context: str = "objective",
) -> ResolvedObjectiveWeights:
    """Resolve root/local-law weights through the canonical objective contract."""

    laws: list[str] = []
    for law in tuple(active_laws or ()):
        name = _canonical_law_component_name(str(law))
        if name not in laws:
            laws.append(name)

    explicit_weights = dict(explicit_law_weights or {})
    if local_law_weight is not None:
        if explicit_root_weight is not None or explicit_weights:
            raise ValueError(
                f"{objective_context}: local_law_weight is mutually exclusive "
                "with explicit root/law weights"
            )
        lam = _finite_probability(local_law_weight, name="local_law_weight")
        if lam > 0.0 and not laws:
            raise ValueError(
                f"{objective_context}: local_law_weight > 0 requires at least one active local law"
            )
        share = float(lam / float(len(laws))) if laws and lam > 0.0 else 0.0
        return ResolvedObjectiveWeights(
            input_mode="lambda",
            weighting_scheme="normalized_lambda_tradeoff",
            root_share=float(1.0 - lam),
            local_law_shares={law: float(share) for law in laws},
        )

    root_raw = _finite_nonnegative(
        1.0 if explicit_root_weight is None else explicit_root_weight,
        name="explicit_root_weight",
    )
    law_raw = canonical_law_component_weights(explicit_weights)
    law_raw = {
        str(name): _finite_nonnegative(value, name=f"explicit_law_weights[{name}]")
        for name, value in law_raw.items()
    }
    total = float(root_raw + sum(float(v) for v in law_raw.values()))
    if total <= 0.0:
        return ResolvedObjectiveWeights(
            input_mode="explicit_weights",
            weighting_scheme="normalized_explicit_weights",
            root_share=1.0,
            local_law_shares={str(name): 0.0 for name in law_raw.keys()},
        )
    return ResolvedObjectiveWeights(
        input_mode="explicit_weights",
        weighting_scheme="normalized_explicit_weights",
        root_share=float(root_raw / total),
        local_law_shares={
            str(name): float(value / total) for name, value in law_raw.items()
        },
    )


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
            if name in _EVIDENCE_ONLY_TERM_NAMES:
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
            "terms": jsonable(self._normalized_terms()),
            "metadata": jsonable(dict(self.metadata or {})),
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
        unsupported_fields = sorted(key for key in _UNSUPPORTED_PUBLIC_FIELDS if key in data)
        if unsupported_fields:
            raise ValueError(
                "unsupported objective fields: "
                + ", ".join(unsupported_fields)
            )
        terms = dict(data.get("terms") or {})
        for name in terms:
            if str(name).strip().lower() in _EVIDENCE_ONLY_TERM_NAMES:
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
    def gamma_depth(self) -> float:
        """Depth-discount factor declared on the ``local_law_corrected`` term."""
        term = dict(dict(self.terms or {}).get(OBJECTIVE_TERM_LOCAL_LAW_CORRECTED) or {})
        gamma = finite_float(term.get("gamma_depth", 1.0), name="gamma_depth")
        if gamma < 0.0 or gamma > 1.0:
            raise ValueError(f"gamma_depth must be in [0, 1], got {gamma!r}")
        return gamma

    @property
    def digest(self) -> str:
        return stable_digest(self.to_dict())


def normalize_objective_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate ``payload`` and return its canonical ``ObjectiveSpec`` dict."""
    return ObjectiveSpec.from_mapping(payload).to_dict()


__all__ = [
    "CANONICAL_LAW_COMPONENTS",
    "LOCAL_LAW_ESTIMATOR_CORRECTED",
    "LOCAL_LAW_ESTIMATOR_NONE",
    "LOCAL_LAW_ESTIMATOR_ORACLE_EXACT",
    "LOCAL_LAW_ESTIMATOR_ORACLE_STATE",
    "LOCAL_LAW_ESTIMATOR_PROXY_ONLY",
    "LOCAL_LAW_ESTIMATOR_EXTERNAL_PASSTHROUGH",
    "OBJECTIVE_SCHEMA_VERSION",
    "OBJECTIVE_TERM_LOCAL_LAW_CORRECTED",
    "OBJECTIVE_TERM_ROOT",
    "ObjectiveSpec",
    "ResolvedObjectiveWeights",
    "canonical_law_component_weights",
    "normalize_objective_spec",
    "resolve_root_local_objective_weights",
]
