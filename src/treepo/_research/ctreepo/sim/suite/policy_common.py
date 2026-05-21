from __future__ import annotations

from typing import Iterable, Sequence, Tuple


def parse_int_list(text: str | None, *, default: Sequence[int]) -> Tuple[int, ...]:
    raw = str(text or "").replace(",", " ").split()
    if not raw:
        return tuple(int(x) for x in default)
    return tuple(int(x) for x in raw)


def parse_float_list(
    text: str | None,
    *,
    default: Sequence[float],
) -> Tuple[float, ...]:
    raw = str(text or "").replace(",", " ").split()
    if not raw:
        return tuple(float(x) for x in default)
    return tuple(float(x) for x in raw)


def join_items(values: Iterable[int | float | str]) -> str:
    return " ".join(str(x) for x in values)


__all__ = ["join_items", "parse_float_list", "parse_int_list"]
