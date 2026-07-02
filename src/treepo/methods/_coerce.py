"""Scalar and vector coercion helpers shared across methods modules."""

from __future__ import annotations

import math
from typing import Any, Mapping


def safe_float(value: Any, *, require_finite: bool = False) -> float | None:
    """Coerce ``value`` to ``float``; return ``None`` for non-coercible values.

    With ``require_finite=True``, NaN/inf also coerce to ``None``.
    """
    if value is None or isinstance(value, (list, tuple, Mapping)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if require_finite and not math.isfinite(out):
        return None
    return out


def float_vector(value: Any) -> list[float] | None:
    """Coerce an iterable of scalars to a non-empty float list, else ``None``."""
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        out = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return out if out else None


__all__ = ["float_vector", "safe_float"]
