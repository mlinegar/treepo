"""Input normalization helpers for ``treepo.methods.learning.fit``."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from treepo.methods.contracts import FamilyRuntime, ObjectiveSpec
from treepo.methods.families import resolve_family


def resolve_runtime_family(
    spec: Any,
    backend_config: Mapping[str, Any],
) -> Any:
    injected = backend_config.get("family_runtime")
    if injected is not None:
        if not isinstance(injected, FamilyRuntime):
            raise TypeError(
                "spec.backend_config['family_runtime'] must implement "
                f"FamilyRuntime; got {type(injected).__name__}"
            )
        return injected
    family_name = str(getattr(spec, "family", "") or "")
    if not family_name:
        raise ValueError(
            "spec.family is empty and no family_runtime was supplied. "
            "Set spec.family or spec.backend_config['family_runtime']."
        )
    return resolve_family(family_name, backend_config)


def as_sequence(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return value
    return tuple(value)


def optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def resolve_objective(backend_config: Mapping[str, Any]) -> Any | None:
    """Resolve the optional objective spec from ``backend_config``.

    ``fit()`` hands the resolved spec to the family's ``configure_objective``
    hook to execute during training, and records it in the run manifest. A
    law-bearing objective on a family without the hook is an error: a declared
    training objective must not be provenance-only.
    """
    raw = backend_config.get("objective")
    if raw is None:
        return None
    if isinstance(raw, ObjectiveSpec):
        return raw
    if isinstance(raw, Mapping):
        return ObjectiveSpec(**dict(raw))
    raise TypeError(
        "backend_config['objective'] must be an ObjectiveSpec or mapping; "
        f"got {type(raw).__name__}"
    )


__all__ = [
    "as_sequence",
    "optional_int",
    "resolve_objective",
    "resolve_runtime_family",
]
