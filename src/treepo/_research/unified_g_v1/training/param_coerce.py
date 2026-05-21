"""Small helpers for coercing JSON-derived bundle params to typed kwargs.

Bundle runners in `bundle_runner.py` translate a JSON mapping into typed
kwargs. The same `None if params.get(...) is None else int(...)` dance
appeared dozens of times; these helpers collapse it.
"""
from __future__ import annotations

from typing import Any, Mapping


_MISSING = object()


def opt_int(params: Mapping[str, Any], key: str) -> int | None:
    value = params.get(key, _MISSING)
    if value is _MISSING or value is None or value == "":
        return None
    return int(value)


def opt_float(params: Mapping[str, Any], key: str) -> float | None:
    value = params.get(key, _MISSING)
    if value is _MISSING or value is None or value == "":
        return None
    return float(value)


def opt_str(params: Mapping[str, Any], key: str) -> str | None:
    value = params.get(key, _MISSING)
    if value is _MISSING or value is None:
        return None
    return str(value)
