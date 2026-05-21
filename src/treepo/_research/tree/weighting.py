"""Shared weighting helpers for simulation summaries."""

from __future__ import annotations

from enum import Enum
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class WeightingMode(str, Enum):
    """Supported weighting schemes for aggregate metrics."""

    DOC = "doc"
    LEAF = "leaf"
    TOKEN = "token"


DEFAULT_WEIGHTING_MODES: Tuple[WeightingMode, ...] = (
    WeightingMode.DOC,
    WeightingMode.LEAF,
    WeightingMode.TOKEN,
)


def parse_weighting_modes(modes: Optional[Sequence[str]] = None) -> Tuple[WeightingMode, ...]:
    """Parse/validate weighting mode names with stable order and de-dup."""
    if modes is None:
        return DEFAULT_WEIGHTING_MODES
    out: List[WeightingMode] = []
    seen: set[str] = set()
    valid = {m.value for m in WeightingMode}
    for raw in modes:
        if isinstance(raw, WeightingMode):
            mode = raw.value
        else:
            mode = str(raw).strip().lower()
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
    """Validate legacy mode and ensure it is present in weighting_modes."""
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
    arr = np.asarray(xs, dtype=np.float64)
    return float(np.var(arr, ddof=1))


def ci95_for_mean(xs: Sequence[float]) -> Tuple[float, float, float]:
    """Normal-approx 95% CI for an unweighted sample mean."""
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
    """Weighted mean with graceful zero-weight fallback."""
    if len(values) != len(weights):
        raise ValueError("values and weights must align")
    if len(values) == 0:
        return float("nan")
    vals = np.asarray(values, dtype=np.float64)
    ws = np.asarray(weights, dtype=np.float64)
    ws = np.maximum(ws, 0.0)
    wsum = float(np.sum(ws))
    if wsum <= 0:
        return float(np.mean(vals))
    return float(np.sum(ws * vals) / wsum)


def weighted_mean_ci95(values: Sequence[float], weights: Sequence[float]) -> Dict[str, float]:
    """Weighted mean with approximate 95% CI and effective sample size."""
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
    ws = np.asarray(weights, dtype=np.float64)
    ws = np.maximum(ws, 0.0)
    wsum = float(np.sum(ws))
    if wsum <= 0:
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
    if n_eff <= 1.0:
        se = 0.0
    else:
        se = math.sqrt(max(0.0, v_w) / n_eff)
    z = 1.96
    return {
        "mean": float(mu),
        "se": float(se),
        "ci95_low": float(mu - z * se),
        "ci95_high": float(mu + z * se),
        "weight_sum": float(wsum),
        "effective_n": float(n_eff),
    }


def build_weighting_views_from_replicates(
    *,
    estimates_by_mode: Dict[str, Sequence[float]],
    sample_targets_by_mode: Dict[str, Sequence[float]],
    true_target: float,
) -> Dict[str, Dict[str, float]]:
    """Build common summary block for doc/leaf/token replicate estimates."""
    views: Dict[str, Dict[str, float]] = {}
    for mode, estimates in estimates_by_mode.items():
        est = [float(x) for x in estimates]
        if len(est) == 0:
            views[mode] = {
                "mean_hat": float("nan"),
                "bias": float("nan"),
                "mean_abs_bias": float("nan"),
                "rmse": float("nan"),
                "sample_target_bias": float("nan"),
                "se": float("nan"),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
                "n_replicates": 0.0,
            }
            continue
        n = float(len(est))
        sample_t = [float(x) for x in sample_targets_by_mode.get(mode, ())]
        if len(sample_t) != len(est):
            sample_t = [float("nan")] * len(est)
        mean_hat = float(np.mean(np.asarray(est, dtype=np.float64)))
        se, lo, hi = ci95_for_mean(est)
        sample_bias_vals = [
            (x - y)
            for x, y in zip(est, sample_t)
            if math.isfinite(float(x)) and math.isfinite(float(y))
        ]
        sample_target_bias = float(np.mean(np.asarray(sample_bias_vals, dtype=np.float64))) if sample_bias_vals else float("nan")
        views[mode] = {
            "mean_hat": float(mean_hat),
            "bias": float(mean_hat - float(true_target)),
            "mean_abs_bias": float(np.mean(np.abs(np.asarray(est, dtype=np.float64) - float(true_target)))),
            "rmse": float(math.sqrt(float(np.mean((np.asarray(est, dtype=np.float64) - float(true_target)) ** 2)))),
            "sample_target_bias": float(sample_target_bias),
            "se": float(se),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "n_replicates": float(n),
        }
    return views


__all__ = [
    "WeightingMode",
    "DEFAULT_WEIGHTING_MODES",
    "parse_weighting_modes",
    "validate_legacy_weighting_mode",
    "weighted_mean",
    "weighted_mean_ci95",
    "build_weighting_views_from_replicates",
]
