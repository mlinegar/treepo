"""Iteration schedule helpers for the alternating f/g runtime."""

from __future__ import annotations


def stage_powers_for_iteration(k: int) -> tuple[int, int]:
    if k < 0:
        raise ValueError(f"iteration must be >= 0, got {k}")
    f_degree = 1
    g_degree = 1
    side = "f"
    for _ in range(int(k)):
        if side == "f":
            f_degree += 1
            side = "g"
        else:
            g_degree += 1
            side = "f"
    return f_degree, g_degree


def stage_name_for_iteration(k: int) -> str:
    if k == 0:
        return "fg"
    return "fg" + "".join("f" if i % 2 == 0 else "g" for i in range(k))


def stage_label_for_iteration(k: int) -> str:
    f_degree, g_degree = stage_powers_for_iteration(k)
    return f"f^{f_degree} g^{g_degree}"


def trains_f_at_iteration(k: int) -> bool:
    return k >= 1 and k % 2 == 1


def trains_g_at_iteration(k: int) -> bool:
    return k >= 1 and k % 2 == 0


__all__ = [
    "stage_label_for_iteration",
    "stage_name_for_iteration",
    "stage_powers_for_iteration",
    "trains_f_at_iteration",
    "trains_g_at_iteration",
]
