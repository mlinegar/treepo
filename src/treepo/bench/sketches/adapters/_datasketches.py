"""Shared gated `datasketches` import and item coercion for adapter modules."""

from __future__ import annotations

from typing import Any

try:
    import datasketches as _ds
except ImportError:  # pragma: no cover
    _ds = None  # type: ignore[assignment]


def _require_datasketches() -> None:
    if _ds is None:
        raise ImportError(
            "datasketches is required for official sketch benchmarks. "
            "Install with: uv sync --extra sketches"
        )


def _split_weighted_item(item: Any) -> tuple[Any, float | int]:
    """Accept either a bare key or a ``(key, weight)`` update."""
    if isinstance(item, tuple) and len(item) == 2:
        return item[0], item[1]
    return item, 1
