"""Provider-neutral JSON request, result, and token-usage records.

The records in this module are artifact boundaries, not transport clients.
Downstream packages may place the validated payloads inside synchronous or
batched provider requests, then persist the normalized result without storing
credentials or a provider SDK object.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from treepo.common import jsonable, stable_digest

_SECRET_FIELD_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth_token",
        "authorization",
        "bearer_token",
        "client_secret",
        "id_token",
        "password",
        "private_key",
        "refresh_token",
        "secret",
    }
)


def validate_json_object(value: Any, *, name: str = "value") -> dict[str, Any]:
    """Return a detached, strictly JSON-serializable object.

    Mappings, dataclasses, paths, enums, and objects with ``to_dict`` are
    normalized through :func:`treepo.common.jsonable`. Non-object roots,
    non-finite numbers, and values unsupported by JSON are rejected.
    """

    normalized = jsonable(value)
    if not isinstance(normalized, Mapping):
        raise TypeError(f"{name} must be a JSON object, got {type(normalized).__name__}")
    try:
        encoded = json.dumps(normalized, allow_nan=False, sort_keys=True, separators=(",", ":"))
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only finite JSON values") from exc
    if not isinstance(decoded, dict):  # pragma: no cover - guarded by the mapping check
        raise TypeError(f"{name} must be a JSON object")
    return decoded


def parse_json_object(value: Any, *, name: str = "output") -> dict[str, Any]:
    """Parse a JSON object from text/bytes or validate an existing mapping.

    The text form is deliberately strict: Markdown fences and prose around the
    object are not repaired. Benchmark harnesses can therefore distinguish a
    schema-valid model response from an application-side recovery.
    """

    if isinstance(value, (str, bytes, bytearray)):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"{name} must be exactly one JSON object") from exc
        return validate_json_object(parsed, name=name)
    return validate_json_object(value, name=name)


@dataclass(frozen=True)
class TokenUsage:
    """Canonical token accounting with provider extensions in ``metadata``.

    Cache-read and cache-write counters retain the provider-reported input
    categories. Providers differ on whether ``input_tokens`` includes those
    categories and on whether read/write categories can overlap, so only
    non-negativity is imposed on them; their sum is deliberately not checked
    against ``input_tokens``. Reasoning tokens remain a subset of output
    tokens. Unknown provider usage fields are retained by :meth:`from_value`,
    while credential-like field names are rejected so the record remains safe
    to persist as a benchmark artifact.
    """

    input_tokens: int = 0
    cached_input_tokens: int = 0
    # Keyword-only so adding this counter does not shift the historical
    # positional constructor arguments for output/total/metadata.
    cache_write_input_tokens: int = field(default=0, kw_only=True)
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        input_tokens = _nonnegative_int(self.input_tokens, name="input_tokens")
        cached_input_tokens = _nonnegative_int(self.cached_input_tokens, name="cached_input_tokens")
        cache_write_input_tokens = _nonnegative_int(
            self.cache_write_input_tokens, name="cache_write_input_tokens"
        )
        output_tokens = _nonnegative_int(self.output_tokens, name="output_tokens")
        reasoning_tokens = _nonnegative_int(self.reasoning_tokens, name="reasoning_tokens")
        total_tokens = (
            input_tokens + output_tokens
            if self.total_tokens is None
            else _nonnegative_int(self.total_tokens, name="total_tokens")
        )
        if reasoning_tokens > output_tokens:
            raise ValueError("reasoning_tokens cannot exceed output_tokens")
        metadata = _artifact_metadata(self.metadata, name="usage metadata")
        object.__setattr__(self, "input_tokens", input_tokens)
        object.__setattr__(self, "cached_input_tokens", cached_input_tokens)
        object.__setattr__(self, "cache_write_input_tokens", cache_write_input_tokens)
        object.__setattr__(self, "output_tokens", output_tokens)
        object.__setattr__(self, "reasoning_tokens", reasoning_tokens)
        object.__setattr__(self, "total_tokens", total_tokens)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_value(cls, value: Any) -> "TokenUsage":
        """Normalize canonical or conventional provider usage mappings."""

        if isinstance(value, cls):
            return value
        row = validate_json_object(value or {}, name="usage")
        input_details = _mapping_or_empty(
            row.get("input_tokens_details", row.get("prompt_tokens_details"))
        )
        output_details = _mapping_or_empty(
            row.get("output_tokens_details", row.get("completion_tokens_details"))
        )
        explicit_metadata = _mapping_or_empty(row.get("metadata"))

        input_tokens = row.get("input_tokens", row.get("prompt_tokens", 0))
        output_tokens = row.get("output_tokens", row.get("completion_tokens", 0))
        cached_input_tokens = _first_present(
            row,
            ("cached_input_tokens", "cache_read_input_tokens", "cache_read_tokens"),
            input_details,
            (
                "cached_tokens",
                "cached_input_tokens",
                "cache_read_input_tokens",
                "cache_read_tokens",
            ),
        )
        cache_write_input_tokens = _first_present(
            row,
            (
                "cache_write_input_tokens",
                "cache_write_tokens",
                "cache_creation_input_tokens",
            ),
            input_details,
            (
                "cache_write_input_tokens",
                "cache_write_tokens",
                "cache_creation_input_tokens",
            ),
        )
        reasoning_tokens = row.get(
            "reasoning_tokens",
            output_details.get("reasoning_tokens", 0),
        )

        consumed = {
            "cached_input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "cache_read_tokens",
            "cache_write_input_tokens",
            "cache_write_tokens",
            "completion_tokens",
            "completion_tokens_details",
            "input_tokens",
            "input_tokens_details",
            "metadata",
            "output_tokens",
            "output_tokens_details",
            "prompt_tokens",
            "prompt_tokens_details",
            "reasoning_tokens",
            "total_tokens",
        }
        metadata = dict(explicit_metadata)
        for key, item in row.items():
            if key not in consumed:
                metadata[str(key)] = item

        remaining_input_details = {
            str(key): item
            for key, item in input_details.items()
            if key
            not in {
                "cached_tokens",
                "cached_input_tokens",
                "cache_read_input_tokens",
                "cache_read_tokens",
                "cache_write_input_tokens",
                "cache_write_tokens",
                "cache_creation_input_tokens",
            }
        }
        remaining_output_details = {
            str(key): item for key, item in output_details.items() if key != "reasoning_tokens"
        }
        if remaining_input_details:
            metadata["input_tokens_details"] = remaining_input_details
        if remaining_output_details:
            metadata["output_tokens_details"] = remaining_output_details

        return cls(
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_input_tokens=cache_write_input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=row.get("total_tokens"),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": int(self.input_tokens),
            "cached_input_tokens": int(self.cached_input_tokens),
            "cache_write_input_tokens": int(self.cache_write_input_tokens),
            "output_tokens": int(self.output_tokens),
            "reasoning_tokens": int(self.reasoning_tokens),
            "total_tokens": int(self.total_tokens or 0),
            "metadata": validate_json_object(self.metadata, name="usage metadata"),
        }


@dataclass(frozen=True)
class JSONRequestRecord:
    """One provider-neutral structured-output request artifact."""

    request_id: str
    model: str
    payload: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        request_id = str(self.request_id).strip()
        model = str(self.model).strip()
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if not model:
            raise ValueError("model must be non-empty")
        payload = validate_json_object(self.payload, name="request payload")
        _reject_secret_fields(payload, name="request payload")
        metadata = _artifact_metadata(self.metadata, name="request metadata")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_value(cls, value: Any) -> "JSONRequestRecord":
        if isinstance(value, cls):
            return value
        row = validate_json_object(value, name="request record")
        return cls(
            request_id=str(row.get("request_id") or ""),
            model=str(row.get("model") or ""),
            payload=_mapping_or_empty(row.get("payload")),
            metadata=_mapping_or_empty(row.get("metadata")),
        )

    @property
    def digest(self) -> str:
        """Stable digest over the complete record except the digest itself."""

        return stable_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "payload": validate_json_object(self.payload, name="request payload"),
            "metadata": validate_json_object(self.metadata, name="request metadata"),
        }


@dataclass(frozen=True)
class JSONResultRecord:
    """One validated structured JSON result joined to normalized usage."""

    request_id: str
    output: Mapping[str, Any]
    model: str | None = None
    response_id: str | None = None
    usage: TokenUsage | Mapping[str, Any] = field(default_factory=TokenUsage)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        request_id = str(self.request_id).strip()
        if not request_id:
            raise ValueError("request_id must be non-empty")
        model = None if self.model is None else str(self.model).strip() or None
        response_id = None if self.response_id is None else str(self.response_id).strip() or None
        output = validate_json_object(self.output, name="result output")
        usage = TokenUsage.from_value(self.usage)
        metadata = _artifact_metadata(self.metadata, name="result metadata")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "response_id", response_id)
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "usage", usage)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_value(cls, value: Any) -> "JSONResultRecord":
        if isinstance(value, cls):
            return value
        row = validate_json_object(value, name="result record")
        return cls(
            request_id=str(row.get("request_id") or ""),
            output=_mapping_or_empty(row.get("output")),
            model=None if row.get("model") is None else str(row["model"]),
            response_id=(None if row.get("response_id") is None else str(row["response_id"])),
            usage=_mapping_or_empty(row.get("usage")),
            metadata=_mapping_or_empty(row.get("metadata")),
        )

    @property
    def digest(self) -> str:
        """Stable digest over the complete normalized result record."""

        return stable_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "response_id": self.response_id,
            "output": validate_json_object(self.output, name="result output"),
            "usage": cast(TokenUsage, self.usage).to_dict(),
            "metadata": validate_json_object(self.metadata, name="result metadata"),
        }


def _artifact_metadata(value: Any, *, name: str) -> dict[str, Any]:
    metadata = validate_json_object(value or {}, name=name)
    _reject_secret_fields(metadata, name=name)
    return metadata


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(
    primary: Mapping[str, Any],
    primary_keys: tuple[str, ...],
    secondary: Mapping[str, Any],
    secondary_keys: tuple[str, ...],
) -> Any:
    """Return the first explicitly present usage alias, or zero.

    Presence rather than truthiness matters: an explicit zero from a canonical
    field must not be replaced by a nonzero legacy alias later in the payload.
    """

    for source, keys in ((primary, primary_keys), (secondary, secondary_keys)):
        for key in keys:
            if key in source:
                return source[key]
    return 0


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if float(value) != float(number):
        raise ValueError(f"{name} must be an integer")
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _reject_secret_fields(value: Any, *, name: str, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            normalized = key_text.strip().lower().replace("-", "_")
            item_path = f"{path}.{key_text}" if path else key_text
            if normalized in _SECRET_FIELD_NAMES:
                raise ValueError(f"{name} cannot contain credential field {item_path!r}")
            _reject_secret_fields(item, name=name, path=item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secret_fields(item, name=name, path=f"{path}[{index}]")


__all__ = [
    "JSONRequestRecord",
    "JSONResultRecord",
    "TokenUsage",
    "parse_json_object",
    "validate_json_object",
]
