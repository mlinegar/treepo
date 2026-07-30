from __future__ import annotations

import math

import pytest

from treepo.llm import (
    JSONRequestRecord,
    JSONResultRecord,
    TokenUsage,
    parse_json_object,
    validate_json_object,
)


def test_json_object_helpers_are_strict_and_detached() -> None:
    source = {"mass": [1.0, 2.0]}
    validated = validate_json_object(source)
    source["mass"].append(3.0)

    assert validated == {"mass": [1.0, 2.0]}
    assert parse_json_object(b'{"mass": 4}') == {"mass": 4}
    with pytest.raises(TypeError, match="JSON object"):
        parse_json_object("[1, 2]")
    with pytest.raises(ValueError, match="exactly one JSON object"):
        parse_json_object('```json\n{"mass": 4}\n```')
    with pytest.raises(ValueError, match="finite JSON"):
        validate_json_object({"mass": math.nan})


def test_token_usage_normalizes_details_and_preserves_extensions() -> None:
    usage = TokenUsage.from_value(
        {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 80, "audio_tokens": 2},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 12, "accepted_tokens": 4},
            "total_tokens": 120,
            "cache_write_tokens": 20,
        }
    )

    assert usage.input_tokens == 100
    assert usage.cached_input_tokens == 80
    assert usage.cache_write_input_tokens == 20
    assert usage.output_tokens == 20
    assert usage.reasoning_tokens == 12
    assert usage.total_tokens == 120
    assert usage.metadata == {
        "input_tokens_details": {"audio_tokens": 2},
        "output_tokens_details": {"accepted_tokens": 4},
    }
    assert TokenUsage.from_value(usage.to_dict()) == usage


@pytest.mark.parametrize(
    ("payload", "expected_read", "expected_write"),
    [
        (
            {
                "input_tokens": 100,
                "cache_read_input_tokens": 70,
                "cache_creation_input_tokens": 20,
            },
            70,
            20,
        ),
        (
            {
                "input_tokens": 100,
                "input_tokens_details": {
                    "cache_read_tokens": 60,
                    "cache_write_input_tokens": 30,
                },
            },
            60,
            30,
        ),
        (
            {
                "prompt_tokens": 100,
                "prompt_tokens_details": {
                    "cached_input_tokens": 50,
                    "cache_creation_input_tokens": 40,
                },
            },
            50,
            40,
        ),
    ],
)
def test_token_usage_normalizes_cache_aliases(
    payload: dict[str, object], expected_read: int, expected_write: int
) -> None:
    usage = TokenUsage.from_value(payload)

    assert usage.cached_input_tokens == expected_read
    assert usage.cache_write_input_tokens == expected_write
    assert usage.metadata == {}


def test_token_usage_cache_categories_are_nonnegative_but_not_assumed_disjoint() -> None:
    # Some providers include cache counters in input_tokens while others report
    # them beside ordinary input. Even inclusion-style providers may expose
    # read/write classifications whose relationship is not contractual.
    usage = TokenUsage(
        input_tokens=100,
        cached_input_tokens=90,
        cache_write_input_tokens=80,
    )
    beside_input = TokenUsage.from_value(
        {
            "input_tokens": 10,
            "cache_read_input_tokens": 70,
            "cache_creation_input_tokens": 20,
        }
    )

    assert usage.cached_input_tokens + usage.cache_write_input_tokens == 170
    assert beside_input.cached_input_tokens == 70
    assert beside_input.cache_write_input_tokens == 20
    with pytest.raises(ValueError, match="cache_write_input_tokens must be non-negative"):
        TokenUsage(cache_write_input_tokens=-1)


def test_token_usage_new_cache_write_field_preserves_positional_constructor() -> None:
    usage = TokenUsage(10, 2, 3, 1, 13, {"provider": "legacy"})

    assert usage.output_tokens == 3
    assert usage.reasoning_tokens == 1
    assert usage.metadata == {"provider": "legacy"}
    assert usage.cache_write_input_tokens == 0


def test_usage_and_artifact_metadata_reject_credential_fields() -> None:
    with pytest.raises(ValueError, match="credential field"):
        TokenUsage.from_value({"input_tokens": 1, "api_key": "do-not-store"})
    with pytest.raises(ValueError, match="credential field"):
        JSONRequestRecord(
            request_id="doc-1",
            model="model-a",
            payload={"input": "document", "authorization": "secret"},
        )


def test_json_request_and_result_records_round_trip() -> None:
    request = JSONRequestRecord(
        request_id="doc-1:model-a:rep-0",
        model="model-a",
        payload={
            "input": "complete document",
            "schema": {"type": "object", "additionalProperties": False},
        },
        metadata={"split": "validation", "replicate": 0},
    )
    restored_request = JSONRequestRecord.from_value(request.to_dict())
    assert restored_request == request
    assert restored_request.digest == request.digest

    result = JSONResultRecord(
        request_id=request.request_id,
        model=request.model,
        response_id="response-1",
        output={"left_mass": 20, "right_mass": 30, "other_mass": 50, "header_mass": 2},
        usage={
            "prompt_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 75},
            "completion_tokens": 10,
            "completion_tokens_details": {"reasoning_tokens": 4},
            "total_tokens": 110,
        },
        metadata={"program_digest": "abc"},
    )
    restored_result = JSONResultRecord.from_value(result.to_dict())

    assert restored_result == result
    assert restored_result.digest == result.digest
    assert restored_result.usage.cached_input_tokens == 75
    assert restored_result.output["right_mass"] == 30
