"""Shared weighting helpers for TreePO benchmark summaries."""

from __future__ import annotations

from enum import Enum
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class WeightingMode(str, Enum):
    DOC = "doc"
    LEAF = "leaf"
    TOKEN = "token"


DEFAULT_WEIGHTING_MODES: Tuple[WeightingMode, ...] = (
    WeightingMode.DOC,
    WeightingMode.LEAF,
    WeightingMode.TOKEN,
)


def parse_weighting_modes(modes: Optional[Sequence[str]] = None) -> Tuple[WeightingMode, ...]:
    if modes is None:
        return DEFAULT_WEIGHTING_MODES
    out: List[WeightingMode] = []
    seen: set[str] = set()
    valid = {m.value for m in WeightingMode}
    for raw in modes:
        mode = raw.value if isinstance(raw, WeightingMode) else str(raw).strip().lower()
        if mode == "":
            continue
        if mode not in valid:
            raise ValueError(f"unsupported weighting mode: {raw!r}")
        if mode in seen:
            continue
        seen.add(mode)
        out.append(WeightingMode(mode))
    if len(out) == 0:
        raise ValueError("weighting modes must be non-empty")
    return tuple(out)


def validate_legacy_weighting_mode(
    legacy_mode: str,
    *,
    weighting_modes: Sequence[WeightingMode],
) -> WeightingMode:
    mode = WeightingMode(str(legacy_mode).strip().lower())
    if mode not in set(weighting_modes):
        raise ValueError(
            f"legacy_weighting_mode={legacy_mode!r} must be included in weighting_modes="
            f"{[m.value for m in weighting_modes]}"
        )
    return mode


def _safe_mean(xs: Sequence[float]) -> float:
    if len(xs) == 0:
        return float("nan")
    return float(np.mean(np.asarray(xs, dtype=np.float64)))


def _safe_var(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return float(np.var(np.asarray(xs, dtype=np.float64), ddof=1))


def ci95_for_mean(xs: Sequence[float]) -> Tuple[float, float, float]:
    mean = _safe_mean(xs)
    if not math.isfinite(mean):
        return (float("nan"), float("nan"), float("nan"))
    n = int(len(xs))
    if n <= 1:
        return (0.0, mean, mean)
    se = math.sqrt(max(0.0, _safe_var(xs)) / float(n))
    z = 1.96
    return (float(se), float(mean - z * se), float(mean + z * se))


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights):
        raise ValueError("values and weights must align")
    if len(values) == 0:
        return float("nan")
    vals = np.asarray(values, dtype=np.float64)
    ws = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    wsum = float(np.sum(ws))
    if wsum <= 0.0:
        return float(np.mean(vals))
    return float(np.sum(ws * vals) / wsum)


def weighted_mean_ci95(values: Sequence[float], weights: Sequence[float]) -> Dict[str, float]:
    if len(values) != len(weights):
        raise ValueError("values and weights must align")
    if len(values) == 0:
        return {
            "mean": float("nan"),
            "se": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "weight_sum": 0.0,
            "effective_n": 0.0,
        }

    vals = np.asarray(values, dtype=np.float64)
    ws = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    wsum = float(np.sum(ws))
    if wsum <= 0.0:
        mu = float(np.mean(vals))
        se, lo, hi = ci95_for_mean(vals.tolist())
        return {
            "mean": float(mu),
            "se": float(se),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "weight_sum": 0.0,
            "effective_n": float(len(vals)),
        }

    mu = float(np.sum(ws * vals) / wsum)
    centered = vals - mu
    v_w = float(np.sum(ws * centered * centered) / wsum)
    w2sum = float(np.sum(ws * ws))
    n_eff = float((wsum * wsum) / w2sum) if w2sum > 0 else float(len(vals))
    se = 0.0 if n_eff <= 1.0 else math.sqrt(max(0.0, v_w) / n_eff)
    z = 1.96
    return {
        "mean": float(mu),
        "se": float(se),
        "ci95_low": float(mu - z * se),
        "ci95_high": float(mu + z * se),
        "weight_sum": float(wsum),
        "effective_n": float(n_eff),
    }


__all__ = [
    "DEFAULT_WEIGHTING_MODES",
    "WeightingMode",
    "parse_weighting_modes",
    "validate_legacy_weighting_mode",
    "weighted_mean",
    "weighted_mean_ci95",
]
