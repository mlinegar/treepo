"""Identification weights for partially observed additive tree targets.

These helpers describe how a document-level additive/share label constrains a
node through its subtree mass. They are identification/sensitivity weights,
not sampling propensities; callers that use them in objectives should pass
them through ``node_weight`` while leaving logged design propensities intact.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from treepo.common import finite_float

IDENTIFICATION_WEIGHT_NONE = "none"
IDENTIFICATION_WEIGHT_SENSITIVITY = "sensitivity"
IDENTIFICATION_WEIGHT_INFORMATION = "information"
IDENTIFICATION_WEIGHT_PROFILES = (
    IDENTIFICATION_WEIGHT_NONE,
    IDENTIFICATION_WEIGHT_SENSITIVITY,
    IDENTIFICATION_WEIGHT_INFORMATION,
)


def additive_root_sensitivity(node_mass: float, document_mass: float) -> float:
    """Return the root-share sensitivity ``m / M`` for an additive node."""

    m, M = _validated_masses(node_mass, document_mass)
    return float(m / M)


def additive_root_information_weight(node_mass: float, document_mass: float) -> float:
    """Return the squared additive root sensitivity ``(m / M)^2``."""

    sensitivity = additive_root_sensitivity(node_mass, document_mass)
    return float(sensitivity * sensitivity)


def pairwise_trace_node_masses(leaf_masses: Sequence[float]) -> list[float]:
    """Return node masses in the package pairwise-with-carry trace order.

    Leaves appear first, followed by each realized merge node. Odd leaves are
    carried to the next level without creating a duplicate trace row.
    """

    cur = [_validated_leaf_mass(value) for value in list(leaf_masses or ())]
    if not cur:
        return []
    rows = list(cur)
    while len(cur) > 1:
        next_level: list[float] = []
        for idx in range(0, len(cur) - 1, 2):
            merged = float(cur[idx] + cur[idx + 1])
            next_level.append(merged)
            rows.append(merged)
        if len(cur) % 2:
            next_level.append(cur[-1])
        cur = next_level
    return rows


def additive_identification_metadata(
    *,
    node_mass: float,
    document_mass: float,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return JSONable additive-identification metadata for one node."""

    sensitivity = additive_root_sensitivity(node_mass, document_mass)
    payload: dict[str, Any] = {
        "node_mass": float(node_mass),
        "document_mass": float(document_mass),
        "additive_root_sensitivity": float(sensitivity),
        "additive_root_information_weight": float(sensitivity * sensitivity),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def identification_node_weight(
    *,
    node_mass: float,
    document_mass: float,
    profile: str,
) -> float:
    """Return the objective ``node_weight`` for an identification profile."""

    normalized = normalize_identification_weight_profile(profile)
    if normalized == IDENTIFICATION_WEIGHT_NONE:
        return 1.0
    if normalized == IDENTIFICATION_WEIGHT_SENSITIVITY:
        return additive_root_sensitivity(node_mass, document_mass)
    return additive_root_information_weight(node_mass, document_mass)


def annotate_additive_identification_rows(
    rows: Sequence[Any],
    *,
    node_masses: Sequence[float],
    document_mass: float | None = None,
    profile: str = IDENTIFICATION_WEIGHT_NONE,
    mass_source: str = "provided",
) -> tuple[Any, ...]:
    """Attach additive-identification metadata and optional weights to rows.

    This is deliberately model-agnostic: callers pass already-built
    ``LocalLawAuditRow`` values plus their node masses. Logged propensities are
    preserved. When ``profile="none"``, existing row ``node_weight`` values are
    preserved; ``sensitivity`` and ``information`` replace ``node_weight`` with
    ``m/M`` and ``(m/M)^2`` respectively.
    """

    from treepo.local_law import LocalLawAuditRow

    row_list = tuple(
        row if isinstance(row, LocalLawAuditRow) else LocalLawAuditRow(**dict(row))
        for row in rows
    )
    masses = tuple(_validated_leaf_mass(value, name="node_mass") for value in node_masses)
    if len(row_list) != len(masses):
        raise ValueError(f"got {len(row_list)} rows but {len(masses)} node masses")
    if not row_list:
        return ()
    M = finite_float(document_mass if document_mass is not None else masses[-1], name="document_mass")
    if M <= 0.0:
        raise ValueError("document_mass must be positive")
    normalized = normalize_identification_weight_profile(profile)
    out: list[LocalLawAuditRow] = []
    for row, node_mass in zip(row_list, masses):
        metadata = additive_identification_metadata(
            node_mass=node_mass,
            document_mass=M,
            extra={
                "identification_weight_kind": "additive_root_share",
                "identification_weight_profile": normalized,
                "node_mass_source": str(mass_source),
            },
        )
        node_weight = float(row.node_weight)
        if normalized != IDENTIFICATION_WEIGHT_NONE:
            node_weight = identification_node_weight(
                node_mass=node_mass,
                document_mass=M,
                profile=normalized,
            )
        out.append(
            replace(
                row,
                node_weight=float(node_weight),
                metadata={**dict(row.metadata or {}), **metadata},
            )
        )
    return tuple(out)


def normalize_identification_weight_profile(value: Any) -> str:
    """Normalize and validate an identification weight profile name."""

    profile = str(value or IDENTIFICATION_WEIGHT_NONE).strip().lower().replace("-", "_")
    if profile not in IDENTIFICATION_WEIGHT_PROFILES:
        allowed = ", ".join(IDENTIFICATION_WEIGHT_PROFILES)
        raise ValueError(f"unsupported identification_weight_profile {value!r}; expected one of {allowed}")
    return profile


def _validated_masses(node_mass: float, document_mass: float) -> tuple[float, float]:
    m = _validated_leaf_mass(node_mass, name="node_mass")
    M = finite_float(document_mass, name="document_mass")
    if M <= 0.0:
        raise ValueError("document_mass must be positive")
    return float(m), float(M)


def _validated_leaf_mass(value: float, *, name: str = "leaf_mass") -> float:
    mass = finite_float(value, name=name)
    if mass < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return float(mass)


__all__ = [
    "IDENTIFICATION_WEIGHT_INFORMATION",
    "IDENTIFICATION_WEIGHT_NONE",
    "IDENTIFICATION_WEIGHT_PROFILES",
    "IDENTIFICATION_WEIGHT_SENSITIVITY",
    "annotate_additive_identification_rows",
    "additive_identification_metadata",
    "additive_root_information_weight",
    "additive_root_sensitivity",
    "identification_node_weight",
    "normalize_identification_weight_profile",
    "pairwise_trace_node_masses",
]
