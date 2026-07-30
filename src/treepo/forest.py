"""Named coordinates for one shared-state, joint-vector C-Tree.

A Semantic Forest uses one declared compositional state operator ``g`` and one
joint vector readout ``f``. ``g`` may be learned, explicitly fixed, or the
fixed identity used by the canonical one-leaf full-document baseline.
``OracleTargetSpec`` identifies a coordinate of that joint oracle response; it
does not own a separate dataset, objective, or tree learner.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from treepo.common import jsonable, stable_digest

ORACLE_METRIC_SCHEMA_VERSION = "treepo.oracle_metric.v1"
SEMANTIC_FOREST_SCHEMA_VERSION = "treepo.semantic_forest.v1"


@dataclass(frozen=True)
class OracleTargetSpec:
    """One ordered semantic coordinate of a joint vector oracle.

    ``target_name`` is the stable estimand/coordinate. ``oracle_id`` records
    the versioned evaluator or labeling process supplying that coordinate.
    Several coordinates may share an oracle, and an oracle revision does not
    rename the estimand.
    """

    target_name: str
    oracle_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("target_name", "oracle_id"):
            value = str(getattr(self, field_name) or "").strip()
            if not value:
                raise ValueError(f"OracleTargetSpec.{field_name} must be non-empty")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_value(cls, value: "OracleTargetSpec | Mapping[str, Any]") -> "OracleTargetSpec":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                "each oracle_targets member must be an OracleTargetSpec or mapping, "
                f"got {type(value).__name__}"
            )
        payload = dict(value)
        allowed = {"schema_version", "target_name", "oracle_id", "metadata"}
        unknown = sorted(str(key) for key in set(payload).difference(allowed))
        if unknown:
            raise ValueError(
                "oracle_targets members describe coordinates of one joint oracle; "
                "per-target data, objectives, state kinds, or initial artifacts are "
                f"not supported (unknown fields: {unknown!r})"
            )
        schema_version = payload.get("schema_version")
        if schema_version is not None and str(schema_version) != SEMANTIC_FOREST_SCHEMA_VERSION:
            raise ValueError(
                "unsupported OracleTargetSpec schema_version "
                f"{schema_version!r}; expected {SEMANTIC_FOREST_SCHEMA_VERSION!r}"
            )
        return cls(
            target_name=str(payload.get("target_name") or ""),
            oracle_id=str(payload.get("oracle_id") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )

    @property
    def contract_digest(self) -> str:
        return stable_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SEMANTIC_FOREST_SCHEMA_VERSION,
            "target_name": self.target_name,
            "oracle_id": self.oracle_id,
            "metadata": jsonable(dict(self.metadata or {})),
        }


def normalize_oracle_targets(
    value: Iterable[OracleTargetSpec | Mapping[str, Any]],
) -> tuple[OracleTargetSpec, ...]:
    """Return a non-empty ordered coordinate family with unique target names."""

    if isinstance(value, (str, bytes, Mapping, set, frozenset)) or not isinstance(value, Iterable):
        raise TypeError("oracle_targets must be an ordered iterable of coordinate records")
    targets = tuple(OracleTargetSpec.from_value(member) for member in value)
    if not targets:
        raise ValueError("oracle_targets must contain at least one coordinate")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for target in targets:
        if target.target_name in seen:
            duplicates.add(target.target_name)
        seen.add(target.target_name)
    if duplicates:
        raise ValueError(f"oracle target_name values must be unique, got {sorted(duplicates)!r}")
    return targets


def l1_oracle_metric_schema() -> dict[str, Any]:
    """Return the width-independent metric contract for joint oracle values.

    The point distance is the sum of absolute coordinate errors. A scalar is
    therefore exactly a one-coordinate vector; there is no ``K=1`` branch.
    Evaluation and oracle geometry use this same L1 contract at every width.
    """

    return {
        "schema_version": ORACLE_METRIC_SCHEMA_VERSION,
        "metric": "l1",
        "point_distance": "sum_absolute_coordinate_error",
        "row_reduction": "arithmetic_mean",
        "coordinate_space": "declared_oracle_target_order",
    }


def oracle_vector_l1(
    prediction: Any,
    target: Any,
    *,
    target_names: Iterable[str] = (),
) -> float:
    """Return ``sum_j |prediction_j - target_j|`` for any oracle width.

    Mappings are aligned by the declared target order and must cover it
    exactly. Scalars are accepted only when the declared width is one (or no
    names are supplied), which is the same one-element vector calculation.
    """

    names = tuple(str(name) for name in target_names)
    predicted = _oracle_vector(prediction, names=names, source="prediction")
    observed = _oracle_vector(target, names=names, source="target")
    if len(predicted) != len(observed):
        raise ValueError(
            "prediction and target must have the same oracle width; "
            f"got {len(predicted)} and {len(observed)}"
        )
    return float(sum(abs(left - right) for left, right in zip(predicted, observed)))


def _oracle_vector(
    value: Any,
    *,
    names: tuple[str, ...],
    source: str,
) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        if not names:
            raise ValueError(f"{source} mappings require target_names")
        by_name = {str(key): component for key, component in value.items()}
        if len(by_name) != len(value):
            raise ValueError(f"{source} keys collide after string coercion")
        expected = set(names)
        actual = set(by_name)
        if actual != expected:
            raise ValueError(
                f"{source} mapping must exactly cover target_names; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        raw = tuple(by_name[name] for name in names)
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        raw = tuple(value)
    else:
        raw = (value,)
    if names and len(raw) != len(names):
        raise ValueError(
            f"{source} width {len(raw)} does not match target_names width {len(names)}"
        )
    numbers: list[float] = []
    for component in raw:
        try:
            number = float(component)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source} must contain only finite numeric values") from exc
        if not math.isfinite(number):
            raise ValueError(f"{source} must contain only finite numeric values")
        numbers.append(number)
    if not numbers:
        raise ValueError(f"{source} must contain at least one coordinate")
    return tuple(numbers)


__all__ = [
    "ORACLE_METRIC_SCHEMA_VERSION",
    "OracleTargetSpec",
    "SEMANTIC_FOREST_SCHEMA_VERSION",
    "l1_oracle_metric_schema",
    "normalize_oracle_targets",
    "oracle_vector_l1",
]
