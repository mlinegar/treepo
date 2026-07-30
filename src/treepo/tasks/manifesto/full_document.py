"""Validated full-document Manifesto/RILE mass-state contract.

This module owns the task arithmetic only. Prompt optimization, provider
requests, batch execution, document loading, and benchmark split policy remain
downstream concerns.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, cast

from treepo.common import finite_float, jsonable
from treepo.state import TaskState, state_from_value
from treepo.tasks.manifesto.state import MANIFESTO_POLICY_STATE_KIND

MANIFESTO_RILE_MASS_SCHEMA_VERSION = "treepo_manifesto_rile_mass_v1"
_MASS_OUTPUT_FIELDS = ("left_mass", "right_mass", "other_mass", "header_mass")


def manifesto_rile_mass_output_schema() -> dict[str, Any]:
    """Return the strict JSON Schema for one full-document mass estimate.

    The returned object is the schema itself, without a provider-specific
    wrapper, so callers can embed it directly wherever their structured-output
    transport expects a JSON Schema.
    """

    return {
        "type": "object",
        "description": (
            "Full-document policy-mass estimate for deterministic Manifesto RILE readout. "
            "Masses may be counts, shares, or another common non-negative unit."
        ),
        "properties": {
            "left_mass": {
                "type": "number",
                "description": "Substantive left-coded policy mass included in the denominator.",
            },
            "right_mass": {
                "type": "number",
                "description": "Substantive right-coded policy mass included in the denominator.",
            },
            "other_mass": {
                "type": "number",
                "description": (
                    "Substantive policy mass included in the denominator but coded neither "
                    "left nor right."
                ),
            },
            "header_mass": {
                "type": "number",
                "description": (
                    "Header, administrative, or otherwise non-substantive mass excluded from "
                    "the RILE denominator."
                ),
            },
        },
        "required": list(_MASS_OUTPUT_FIELDS),
        "additionalProperties": False,
    }


def manifesto_rile_mass_state(
    *,
    left_mass: float,
    right_mass: float,
    other_mass: float,
    header_mass: float,
    metadata: Mapping[str, Any] | None = None,
) -> TaskState:
    """Build a validated full-document mass estimate as a ``TaskState``.

    RILE is read deterministically as
    ``100 * (right_mass - left_mass) / non_header_mass``. All four masses
    share one arbitrary non-negative unit, so the result is scale invariant.
    At least one non-header mass must be positive.
    """

    left = _mass(left_mass, name="left_mass")
    right = _mass(right_mass, name="right_mass")
    other = _mass(other_mass, name="other_mass")
    header = _mass(header_mass, name="header_mass")
    non_header = left + right + other
    if non_header <= 0.0:
        raise ValueError("left_mass + right_mass + other_mass must be positive")
    total = non_header + header
    rile = 100.0 * (right - left) / non_header
    state_metadata = _json_metadata(metadata)
    state_metadata.update(
        {
            "schema_version": MANIFESTO_RILE_MASS_SCHEMA_VERSION,
            "state_source": "full_document_mass_estimate",
            "document_scope": "full_document",
            "mass_semantics": "common_nonnegative_unit",
        }
    )
    return TaskState(
        kind=MANIFESTO_POLICY_STATE_KIND,
        counts={
            "left": left,
            "right": right,
            "other": other,
            "header": header,
            "non_header": non_header,
            "total": total,
        },
        measures={
            "rile": rile,
            "rile_from_masses": rile,
        },
        metadata=state_metadata,
    )


def manifesto_rile_mass_state_from_value(value: Any) -> TaskState:
    """Validate raw structured output or round-trip a mass ``TaskState``."""

    state = state_from_value(value)
    if isinstance(state, TaskState):
        if state.kind != MANIFESTO_POLICY_STATE_KIND:
            raise ValueError(
                f"expected state kind {MANIFESTO_POLICY_STATE_KIND!r}, got {state.kind!r}"
            )
        if state.items:
            raise ValueError("full-document RILE mass state cannot contain item-level labels")
        counts = dict(state.counts or {})
        required_counts = {"left", "right", "other", "header"}
        missing = sorted(required_counts - set(counts))
        if missing:
            raise ValueError(f"full-document RILE mass state missing counts: {missing}")
        normalized = manifesto_rile_mass_state(
            left_mass=counts["left"],
            right_mass=counts["right"],
            other_mass=counts["other"],
            header_mass=counts["header"],
            metadata=dict(state.metadata or {}),
        )
        _check_derived_value(counts, "non_header", normalized.counts["non_header"])
        _check_derived_value(counts, "total", normalized.counts["total"])
        measures = dict(state.measures or {})
        _check_derived_value(measures, "rile", normalized.measures["rile"])
        _check_derived_value(
            measures,
            "rile_from_masses",
            normalized.measures["rile_from_masses"],
        )
        return normalized

    if not isinstance(value, Mapping):
        raise TypeError(
            "full-document RILE mass output must be a mapping or manifesto policy TaskState"
        )
    row = dict(value)
    unknown = sorted(set(row) - set(_MASS_OUTPUT_FIELDS))
    missing = sorted(set(_MASS_OUTPUT_FIELDS) - set(row))
    if unknown:
        raise ValueError(f"full-document RILE mass output has unknown fields: {unknown}")
    if missing:
        raise ValueError(f"full-document RILE mass output is missing fields: {missing}")
    return manifesto_rile_mass_state(
        left_mass=row["left_mass"],
        right_mass=row["right_mass"],
        other_mass=row["other_mass"],
        header_mass=row["header_mass"],
    )


def manifesto_rile_mass_readout(value: Any) -> float:
    """Return deterministic RILE from raw full-document masses or state."""

    state = manifesto_rile_mass_state_from_value(value)
    return float(state.measures["rile"])


def _mass(value: Any, *, name: str) -> float:
    try:
        number = finite_float(float(value), name=name)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _json_metadata(value: Any) -> dict[str, Any]:
    normalized = jsonable(value or {})
    if not isinstance(normalized, Mapping):
        raise TypeError("metadata must be a JSON object")
    try:
        decoded = json.loads(json.dumps(normalized, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain only finite JSON values") from exc
    return cast(dict[str, Any], decoded)


def _check_derived_value(values: Mapping[str, Any], key: str, expected: Any) -> None:
    if key not in values:
        return
    try:
        observed = finite_float(float(values[key]), name=key)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be finite") from exc
    if not math.isclose(observed, float(expected), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"inconsistent {key}: expected {float(expected)!r}, got {observed!r}")


__all__ = [
    "MANIFESTO_RILE_MASS_SCHEMA_VERSION",
    "manifesto_rile_mass_output_schema",
    "manifesto_rile_mass_readout",
    "manifesto_rile_mass_state",
    "manifesto_rile_mass_state_from_value",
]
