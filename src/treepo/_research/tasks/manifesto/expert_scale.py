"""Explicit scale handling for Benoit expert-survey targets.

Benoit's released ``data_experts.rda`` keeps expert means on the survey-side
scale used in their replication files.  Those values are correlated directly
against 1-7 LLM scores in the paper, which is valid for Pearson but misleading
for MAE/calibration and for any supervised target that assumes a 1-7 range.

This module keeps both surfaces named:

``raw_benoit``
    The exact released ``expert_mean`` value.

``normalized_1_7``
    A derived target on the same 1-7 range as LLM outputs.  Non-EU Benoit
    dimensions are treated as 0-10 expert scales; EU is already native 1-7.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from .dimensions import PolicyDimension


EXPERT_SCALE_RAW = "raw_benoit"
EXPERT_SCALE_NORMALIZED_1_7 = "normalized_1_7"
EXPERT_SCALE_CHOICES = (EXPERT_SCALE_NORMALIZED_1_7, EXPERT_SCALE_RAW)
SCORER_OUTPUT_SCALE_1_7 = "scorer_1_7"
SCORER_OUTPUT_MIN_1_7 = 1.0
SCORER_OUTPUT_MAX_1_7 = 7.0


def _coerce_dimension(dimension: PolicyDimension | str) -> PolicyDimension:
    if isinstance(dimension, PolicyDimension):
        return dimension
    return PolicyDimension(str(dimension))


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def normalize_to_unit(value: Any, *, minimum: float, maximum: float) -> Optional[float]:
    raw = safe_float(value)
    if raw is None:
        return None
    span = float(maximum) - float(minimum)
    if span <= 0.0:
        return None
    return max(0.0, min(1.0, (float(raw) - float(minimum)) / span))


def denormalize_from_unit(value: Any, *, minimum: float, maximum: float) -> Optional[float]:
    raw = safe_float(value)
    if raw is None:
        return None
    unit = max(0.0, min(1.0, float(raw)))
    return float(minimum) + unit * (float(maximum) - float(minimum))


def scorer_output_bounds_1_7() -> tuple[float, float]:
    return SCORER_OUTPUT_MIN_1_7, SCORER_OUTPUT_MAX_1_7


def expert_scale_bounds(
    *,
    dimension: PolicyDimension | str,
    scale: str = EXPERT_SCALE_NORMALIZED_1_7,
) -> tuple[float, float]:
    """Return bounds for a Benoit expert target scale."""
    if scale == EXPERT_SCALE_NORMALIZED_1_7:
        return 1.0, 7.0
    if scale == EXPERT_SCALE_RAW:
        dim = _coerce_dimension(dimension)
        if dim == PolicyDimension.EU:
            return 1.0, 7.0
        return 0.0, 10.0
    raise ValueError(f"unknown Benoit expert scale {scale!r}; expected one of {EXPERT_SCALE_CHOICES}")


def scorer_1_7_to_unit(value: Any) -> Optional[float]:
    lo, hi = scorer_output_bounds_1_7()
    return normalize_to_unit(value, minimum=lo, maximum=hi)


def scorer_1_7_to_expert_target(
    value: Any,
    *,
    dimension: PolicyDimension | str,
    scale: str,
) -> Optional[float]:
    """Map a seven-point LLM/scorer output onto an expert target scale."""
    unit = scorer_1_7_to_unit(value)
    if unit is None:
        return None
    lo, hi = expert_scale_bounds(dimension=dimension, scale=scale)
    return denormalize_from_unit(unit, minimum=lo, maximum=hi)


def expert_target_to_unit(
    value: Any,
    *,
    dimension: PolicyDimension | str,
    scale: str,
) -> Optional[float]:
    lo, hi = expert_scale_bounds(dimension=dimension, scale=scale)
    return normalize_to_unit(value, minimum=lo, maximum=hi)


def normalize_benoit_expert_mean(value: Any, dimension: PolicyDimension | str) -> Optional[float]:
    """Map a released Benoit expert mean onto the 1-7 LLM score scale."""
    raw = safe_float(value)
    if raw is None:
        return None
    dim = _coerce_dimension(dimension)
    if dim == PolicyDimension.EU:
        return max(1.0, min(7.0, raw))
    bounded = max(0.0, min(10.0, raw))
    return 1.0 + 6.0 * (bounded / 10.0)


def benoit_expert_transform_name(dimension: PolicyDimension | str) -> str:
    dim = _coerce_dimension(dimension)
    if dim == PolicyDimension.EU:
        return "identity_clamped_1_7"
    return "one_plus_six_times_raw_over_ten_clamped_0_10"


def _mapping_value(mapping: Any, key: str) -> Optional[float]:
    if not isinstance(mapping, Mapping):
        return None
    return safe_float(mapping.get(key))


def raw_benoit_expert_from_row(row: Mapping[str, Any], *, dimension: PolicyDimension | str) -> Optional[float]:
    """Resolve the exact released expert value from common row schemas."""
    dim = str(_coerce_dimension(dimension).value)
    for key in (
        "expert_score_native",
        "expert_target_native",
        "benoit_expert_mean_raw",
        "expert_score_raw_benoit",
        "benoit_expert_mean",
        "expert_mean_raw",
        "expert_mean",
    ):
        value = safe_float(row.get(key))
        if value is not None:
            return value
    for key in ("expert_means_raw", "expert_means"):
        value = _mapping_value(row.get(key), dim)
        if value is not None:
            return value
    return None


def normalized_benoit_expert_from_row(
    row: Mapping[str, Any],
    *,
    dimension: PolicyDimension | str,
) -> Optional[float]:
    """Resolve or derive the 1-7 expert target from common row schemas."""
    dim = str(_coerce_dimension(dimension).value)
    for key in ("benoit_expert_mean_1_7", "expert_mean_1_7"):
        value = safe_float(row.get(key))
        if value is not None:
            return value
    for key in ("expert_means_1_7", "expert_dimension_scores_1_7"):
        value = _mapping_value(row.get(key), dim)
        if value is not None:
            return value
    raw = raw_benoit_expert_from_row(row, dimension=dimension)
    if raw is not None:
        return normalize_benoit_expert_mean(raw, dimension)
    value = safe_float(row.get("expert_score_1_7"))
    if value is not None:
        return value
    return None


def resolve_benoit_expert_target(
    row: Mapping[str, Any],
    *,
    dimension: PolicyDimension | str,
    scale: str = EXPERT_SCALE_NORMALIZED_1_7,
) -> Optional[float]:
    """Return the expert target on the requested scale."""
    if scale == EXPERT_SCALE_RAW:
        return raw_benoit_expert_from_row(row, dimension=dimension)
    if scale == EXPERT_SCALE_NORMALIZED_1_7:
        return normalized_benoit_expert_from_row(row, dimension=dimension)
    raise ValueError(f"unknown Benoit expert scale {scale!r}; expected one of {EXPERT_SCALE_CHOICES}")


def expert_scale_metadata(
    *,
    dimension: PolicyDimension | str,
    scale: str = EXPERT_SCALE_NORMALIZED_1_7,
) -> dict[str, object]:
    target_min, target_max = expert_scale_bounds(dimension=dimension, scale=scale)
    scorer_min, scorer_max = scorer_output_bounds_1_7()
    return {
        "expert_target_scale": str(scale),
        "expert_target_min": float(target_min),
        "expert_target_max": float(target_max),
        "scorer_output_scale": SCORER_OUTPUT_SCALE_1_7,
        "scorer_output_min": float(scorer_min),
        "scorer_output_max": float(scorer_max),
        "expert_target_transform": (
            "none_released_benoit_expert_mean"
            if scale == EXPERT_SCALE_RAW
            else benoit_expert_transform_name(dimension)
        ),
    }
