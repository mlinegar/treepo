"""Oracle f* registry for the unified C-TreePO ladder.

Synthetic-DGP simulations (Markov changepoint count, LDA leaf-local-mixture
target, classical sketches) all knew their oracle f* before this module — but
each one buried it inline, making it impossible to plug into the alternating
ladder as ``--f-init oracle:<name>``. This registry centralizes oracles so:

1. the unified runner can resolve ``--f-init oracle:hll_exact`` (etc.) via
   :func:`get_oracle`;
2. any backend family can use the same oracle as its f initializer by name,
   not by code reference;
3. the user's intended workflow ("fix f=oracle → learn g_opt → swap g →
   learn f_opt") works end-to-end through the same surface as the LM ladder.

Each :class:`OracleSpec` keeps the native callable signature so existing call
sites can be migrated to thin re-exports without changing semantics. The
optional :attr:`OracleSpec.score_tree` adapter lets the ladder treat any
oracle uniformly as ``Callable[[Tree], float]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional


@dataclass(frozen=True)
class OracleSpec:
    """Registry record for an oracle f* (or g*) callable.

    ``f_callable`` retains the oracle's native signature so existing call
    sites can re-export it without behavioural change. ``score_tree`` is an
    optional adapter ``Callable[[tree], float]`` that lets the alternating
    ladder treat every oracle uniformly. ``g_callable`` is rare — used for
    named-but-explicit lossy-native merges (e.g. HLL register-max).

    Fields are aligned to the TreeBundle v1 vocabulary in
    :mod:`src.ctreepo.contracts` so that an oracle can validate it is being
    paired with a compatible bundle (matching ``leaf_unit`` and ``domain``).
    """

    name: str
    domain: str
    leaf_unit: str
    f_callable: Callable[..., Any]
    g_callable: Optional[Callable[..., Any]] = None
    score_tree: Optional[Callable[..., Any]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


_ORACLES: Dict[str, OracleSpec] = {}


def register_oracle(spec: OracleSpec, *, replace: bool = False) -> None:
    """Add an :class:`OracleSpec` to the global registry.

    Raises :class:`ValueError` if an oracle with the same name is already
    registered, unless ``replace=True``.
    """
    name = str(spec.name).strip()
    if not name:
        raise ValueError("OracleSpec.name must be non-empty")
    if not replace and name in _ORACLES:
        raise ValueError(
            f"oracle {name!r} already registered; pass replace=True to override"
        )
    _ORACLES[name] = spec


def get_oracle(name: str) -> OracleSpec:
    """Look up a registered oracle by name. Raises ``KeyError`` if absent."""
    key = str(name).strip()
    if key not in _ORACLES:
        available = ", ".join(sorted(_ORACLES)) or "<empty>"
        raise KeyError(
            f"oracle {key!r} not registered; available: {available}"
        )
    return _ORACLES[key]


def list_oracles() -> tuple[str, ...]:
    """Return the names of all registered oracles, sorted."""
    return tuple(sorted(_ORACLES))


def has_oracle(name: str) -> bool:
    return str(name).strip() in _ORACLES


def _clear_for_tests() -> None:
    """Clear the registry. Tests only — module imports re-register defaults."""
    _ORACLES.clear()


# Eagerly import the per-domain registration modules so that oracles are
# available the moment any caller imports ``src.ctreepo.oracles``. Each
# submodule registers via :func:`register_oracle` at import time and is
# safe to import multiple times.
from . import sketches as _sketches  # noqa: E402,F401
from . import markov as _markov  # noqa: E402,F401
from . import lda as _lda  # noqa: E402,F401


__all__ = [
    "OracleSpec",
    "get_oracle",
    "has_oracle",
    "list_oracles",
    "register_oracle",
]
