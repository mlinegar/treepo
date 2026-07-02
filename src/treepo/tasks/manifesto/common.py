"""Private helpers shared by Manifesto fixture modules."""

from __future__ import annotations

import math
import random
from typing import Any, Mapping


def root_label(tree: Any) -> float:
    metadata = dict(getattr(tree, "metadata", None) or {})
    return float(metadata.get("root_label", metadata.get("teacher_score_native", 0.0)) or 0.0)


def sample_indices(
    n: int,
    *,
    sample_size: int | None,
    sample_rate: float | None,
    seed: int,
    min_sampled: int,
) -> tuple[list[int], float]:
    if sample_size is not None and sample_rate is not None:
        raise ValueError("pass sample_size or sample_rate, not both")
    total = max(0, int(n))
    if total == 0:
        return [], 0.0
    if sample_size is not None:
        if int(sample_size) < 0:
            raise ValueError("sample_size must be non-negative")
        k = min(total, int(sample_size))
    elif sample_rate is not None:
        rate = float(sample_rate)
        if rate < 0.0 or rate > 1.0:
            raise ValueError("sample_rate must be in [0, 1]")
        k = 0 if rate == 0.0 else min(total, max(int(min_sampled), int(math.ceil(total * rate))))
    else:
        k = total
    if k <= 0:
        return [], 0.0
    if k >= total:
        return list(range(total)), 1.0
    rng = random.Random(int(seed))
    return sorted(rng.sample(range(total), k)), float(k / total)


def document_propensity(tree: Any) -> float:
    meta = dict(getattr(tree, "metadata", None) or {})
    value = meta.get("document_propensity")
    if value is None:
        sampling = meta.get("document_sampling")
        if isinstance(sampling, Mapping):
            value = sampling.get("document_propensity")
    if value is None:
        return 1.0
    parsed = float(value)
    if parsed <= 0.0 or parsed > 1.0:
        raise ValueError(f"document_propensity must be in (0, 1], got {value!r}")
    return parsed


def slug(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


__all__ = ["document_propensity", "root_label", "sample_indices", "slug"]
