"""Shared internals for deterministic method fixtures."""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Tuple


def exact_score_metadata(target: float, *, target_scale: str) -> dict[str, float | str]:
    """Return the standard exact-score metadata fields used by fixtures."""
    value = float(target)
    return {
        "teacher_score_1_7": value,
        "teacher_score_native": value,
        "expert_score_1_7": value,
        "expert_score_native": value,
        "expert_target_scale": str(target_scale),
        "expert_score_for_objective": value,
    }


def int_tuple(values: Iterable[Any]) -> Tuple[int, ...]:
    """Normalize generated integer arrays/lists to immutable Python ints."""
    return tuple(int(value) for value in values)


def leaf_slices(total_units: int, leaf_unit_count: int) -> Iterator[tuple[int, int]]:
    """Yield half-open leaf ranges for fixed-width synthetic fixtures."""
    for start in range(0, int(total_units), int(leaf_unit_count)):
        yield start, min(int(total_units), start + int(leaf_unit_count))


def require_cuda_torch(generation_device: str, *, fixture_name: str) -> tuple[Any, Any, list[int]]:
    """Load torch and validate a CUDA fixture generation request."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("generation_device='cuda' requires torch") from exc

    device = torch.device(str(generation_device))
    if device.type != "cuda":
        raise ValueError(f"torch {fixture_name} fixture generation is only used for CUDA devices")
    if not torch.cuda.is_available():  # pragma: no cover - environment dependent
        raise RuntimeError("generation_device='cuda' requested but CUDA is unavailable")

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    return torch, device, [device_index]


__all__ = [
    "exact_score_metadata",
    "int_tuple",
    "leaf_slices",
    "require_cuda_torch",
]
