"""Shared utility functions for the ctreepo simulation package.

Centralizes _safe_float / _safe_int and other helpers that were
previously copy-pasted across 30+ files.
"""

from __future__ import annotations

import math
from typing import Any, Optional


def safe_float(value: Any, default: float = float("nan")) -> float:
    """Convert *value* to float, returning *default* on failure or non-finite."""
    if value is None:
        return default
    try:
        f = float(value)
        if not math.isfinite(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Convert *value* to int, returning *default* on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Config string helpers
# ---------------------------------------------------------------------------

def norm_str(value: Any, default: str = "") -> str:
    """Normalize a config value to a stripped lowercase string.

    Replaces the pervasive ``str(x or "").strip().lower()`` pattern.
    """
    if value is None or value == "":
        return default
    return str(value).strip().lower() or default


def get_str(mapping: Any, key: str, default: str = "") -> str:
    """Read a string config key, normalized.

    Replaces ``str(payload.get("key", "") or "").strip().lower()``.
    """
    return norm_str(mapping.get(key, default), default=default)


# ---------------------------------------------------------------------------
# Simplex normalization (previously copy-pasted across 3 files)
# ---------------------------------------------------------------------------

def normalize_simplex_vec(x: "np.ndarray") -> "np.ndarray":
    """Project *x* onto the probability simplex (non-negative, sums to 1)."""
    import numpy as np
    arr = np.maximum(np.asarray(x, dtype=np.float64).reshape(-1), 0.0)
    total = float(np.sum(arr))
    if arr.size == 0:
        return arr.astype(np.float64, copy=False)
    if not math.isfinite(total) or total <= 0.0:
        return np.full((arr.size,), 1.0 / float(arr.size), dtype=np.float64)
    return arr / total


def normalize_simplex_rows(x: "np.ndarray") -> "np.ndarray":
    """Normalize each row of a 2D array onto the probability simplex."""
    import numpy as np
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return normalize_simplex_vec(arr)
    out = np.zeros_like(arr, dtype=np.float64)
    for idx in range(int(arr.shape[0])):
        out[idx] = normalize_simplex_vec(arr[idx])
    return out
