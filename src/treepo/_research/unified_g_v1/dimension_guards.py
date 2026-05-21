"""Public dimension invariant guards for C-TreePO neural components."""

from __future__ import annotations

import warnings


class DimensionInvariantWarning(UserWarning):
    """Warns that a requested model width violated a package invariant.

    Constructors that use this warning promote the offending width to the
    required floor instead of silently building an under-capacity model.
    """


def promote_dim(
    *,
    name: str,
    requested: int | None,
    default: int,
    minimum: int,
    context: str,
    reason: str,
) -> int:
    """Resolve a dimension and warn if it is below the public invariant floor."""

    resolved = int(default if requested is None else requested)
    floor = int(minimum)
    if resolved >= floor:
        return resolved
    source = "default" if requested is None else "requested"
    warnings.warn(
        (
            f"{context}: {source} {name}={resolved} is below the required "
            f"floor {floor} ({reason}); promoting {name} to {floor}."
        ),
        DimensionInvariantWarning,
        stacklevel=2,
    )
    return floor


def require_dim(
    *,
    name: str,
    value: int,
    minimum: int,
    context: str,
    reason: str,
) -> int:
    """Validate a dimension where automatic promotion would change semantics."""

    resolved = int(value)
    floor = int(minimum)
    if resolved >= floor:
        return resolved
    raise ValueError(
        f"{context}: {name}={resolved} is below the required floor {floor} "
        f"({reason})"
    )


__all__ = ["DimensionInvariantWarning", "promote_dim", "require_dim"]
