from __future__ import annotations


RILE_MIN = -100.0
RILE_MAX = 100.0
RILE_RANGE = RILE_MAX - RILE_MIN


def clamp_rile(value: float) -> float:
    return max(RILE_MIN, min(RILE_MAX, float(value)))


__all__ = ["RILE_MAX", "RILE_MIN", "RILE_RANGE", "clamp_rile"]
