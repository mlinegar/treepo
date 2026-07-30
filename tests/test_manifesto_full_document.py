from __future__ import annotations

import json
import math

import pytest

from treepo import TaskState
from treepo.tasks.manifesto import (
    MANIFESTO_POLICY_STATE_KIND,
    MANIFESTO_RILE_MASS_SCHEMA_VERSION,
    manifesto_rile_mass_output_schema,
    manifesto_rile_mass_readout,
    manifesto_rile_mass_state,
    manifesto_rile_mass_state_from_value,
)
from treepo.tasks.manifesto.state import manifesto_policy_readout


def test_full_document_mass_schema_is_directly_embeddable_json_schema() -> None:
    schema = manifesto_rile_mass_output_schema()

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "left_mass",
        "right_mass",
        "other_mass",
        "header_mass",
    }
    assert set(schema["properties"]) == set(schema["required"])
    assert json.loads(json.dumps(schema, allow_nan=False)) == schema
    responses_format = {
        "type": "json_schema",
        "name": "manifesto_rile_mass",
        "strict": True,
        "schema": schema,
    }
    assert json.loads(json.dumps(responses_format))["schema"] == schema


def test_full_document_mass_state_has_exact_rile_readout() -> None:
    state = manifesto_rile_mass_state(
        left_mass=30,
        right_mass=50,
        other_mass=20,
        header_mass=5,
        metadata={"doc_id": "blind-doc-1"},
    )

    assert isinstance(state, TaskState)
    assert state.kind == MANIFESTO_POLICY_STATE_KIND
    assert state.counts == {
        "left": 30.0,
        "right": 50.0,
        "other": 20.0,
        "header": 5.0,
        "non_header": 100.0,
        "total": 105.0,
    }
    assert manifesto_rile_mass_readout(state) == pytest.approx(20.0)
    assert manifesto_policy_readout(state) == pytest.approx(20.0)
    assert state.metadata["schema_version"] == MANIFESTO_RILE_MASS_SCHEMA_VERSION
    assert state.metadata["document_scope"] == "full_document"
    assert state.metadata["doc_id"] == "blind-doc-1"


def test_raw_mass_output_and_task_state_round_trip() -> None:
    raw = {
        "left_mass": 4,
        "right_mass": 1,
        "other_mass": 5,
        "header_mass": 2,
    }
    state = manifesto_rile_mass_state_from_value(raw)
    restored = manifesto_rile_mass_state_from_value(state.to_dict())

    assert manifesto_rile_mass_readout(raw) == pytest.approx(-30.0)
    assert restored.to_dict() == state.to_dict()


@pytest.mark.parametrize(
    "raw,match",
    [
        (
            {"left_mass": 0, "right_mass": 0, "other_mass": 0, "header_mass": 1},
            "must be positive",
        ),
        (
            {"left_mass": -1, "right_mass": 1, "other_mass": 1, "header_mass": 0},
            "left_mass must be non-negative",
        ),
        (
            {
                "left_mass": math.inf,
                "right_mass": 1,
                "other_mass": 1,
                "header_mass": 0,
            },
            "finite non-negative",
        ),
        (
            {
                "left_mass": 1,
                "right_mass": 1,
                "other_mass": 1,
                "header_mass": 0,
                "rile": 0,
            },
            "unknown fields",
        ),
    ],
)
def test_full_document_mass_output_rejects_invalid_values(raw, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        manifesto_rile_mass_state_from_value(raw)


def test_round_trip_rejects_inconsistent_derived_arithmetic() -> None:
    payload = manifesto_rile_mass_state(
        left_mass=1,
        right_mass=2,
        other_mass=1,
        header_mass=0,
    ).to_dict()
    payload["measures"]["rile"] = 99

    with pytest.raises(ValueError, match="inconsistent rile"):
        manifesto_rile_mass_state_from_value(payload)


def test_mass_state_rejects_non_json_metadata() -> None:
    with pytest.raises(ValueError, match="finite JSON"):
        manifesto_rile_mass_state(
            left_mass=1,
            right_mass=1,
            other_mass=1,
            header_mass=0,
            metadata={"bad": math.nan},
        )
