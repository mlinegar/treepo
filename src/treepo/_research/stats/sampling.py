"""Shared weighted sampling utilities."""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence


def normalize_positive_weights(weights: Sequence[float]) -> List[float]:
    """Normalize nonnegative weights, with uniform fallback for degenerate inputs."""
    if not weights:
        return []

    cleaned = [max(0.0, float(weight)) for weight in weights]
    total = sum(cleaned)
    if total <= 0:
        return [1.0 / len(cleaned)] * len(cleaned)
    return [weight / total for weight in cleaned]


def largest_remainder_allocation(total_count: int, quotas: Sequence[float]) -> List[int]:
    """Allocate integer counts from real quotas via Hamilton rounding."""
    if total_count <= 0 or not quotas:
        return [0] * len(quotas)

    base = [int(math.floor(max(0.0, quota))) for quota in quotas]
    remainder = total_count - sum(base)
    if remainder <= 0:
        return base

    fractions = [
        (index, max(0.0, quota) - base[index])
        for index, quota in enumerate(quotas)
    ]
    fractions.sort(key=lambda item: item[1], reverse=True)
    for index, _ in fractions[:remainder]:
        base[index] += 1
    return base


def pps_inclusion_probabilities(weights: Sequence[float], sample_size: int) -> List[float]:
    """
    Compute first-order inclusion probabilities for fixed-size PPS sampling.

    Uses certainty selection and proportional reallocation for remaining units.
    """
    n = len(weights)
    if n == 0 or sample_size <= 0:
        return [0.0] * n

    k = min(sample_size, n)
    normalized = normalize_positive_weights(weights)
    inclusion = [0.0] * n
    remaining_indices = set(range(n))
    remaining_k = float(k)

    while remaining_indices and remaining_k > 1e-12:
        remaining_mass = sum(normalized[index] for index in remaining_indices)
        if remaining_mass <= 0:
            uniform = min(1.0, remaining_k / len(remaining_indices))
            for index in remaining_indices:
                inclusion[index] = uniform
            break

        provisional = {
            index: (remaining_k * normalized[index] / remaining_mass)
            for index in remaining_indices
        }
        certainty = [index for index, prob in provisional.items() if prob >= (1.0 - 1e-12)]
        if not certainty:
            for index, prob in provisional.items():
                inclusion[index] = max(0.0, min(1.0, prob))
            break

        for index in certainty:
            inclusion[index] = 1.0
            remaining_indices.remove(index)
            remaining_k -= 1.0

    return inclusion


def systematic_pps_sample_indices(
    inclusion_probs: Sequence[float],
    sample_size: int,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Draw fixed-size systematic PPS indices using first-order inclusion probabilities."""
    if not inclusion_probs or sample_size <= 0:
        return []

    k = min(sample_size, len(inclusion_probs))
    clipped = [max(0.0, min(1.0, float(prob))) for prob in inclusion_probs]
    sum_pi = sum(clipped)
    if sum_pi <= 0:
        return []

    target_k = min(k, max(1, int(round(sum_pi))))
    draw = random.random if rng is None else rng.random
    u = draw()
    thresholds = [u + float(i) for i in range(target_k)]

    selected: List[int] = []
    cumulative = 0.0
    index = 0
    n = len(clipped)
    for threshold in thresholds:
        while index < n and (cumulative + clipped[index]) < (threshold - 1e-12):
            cumulative += clipped[index]
            index += 1
        if index >= n:
            break
        selected.append(index)

    if len(selected) < target_k:
        chosen = set(selected)
        missing = target_k - len(selected)
        extras = [
            idx
            for idx, _ in sorted(
                enumerate(clipped),
                key=lambda item: item[1],
                reverse=True,
            )
            if idx not in chosen
        ][:missing]
        selected.extend(extras)

    return selected
