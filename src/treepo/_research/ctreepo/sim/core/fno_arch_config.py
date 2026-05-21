"""Unified FNO architecture configuration.

Single source of truth for resolving FNO width / n_modes / n_layers,
replacing 5+ duplicated fallback chains across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class FNOArchConfig:
    """Resolved FNO architecture hyperparameters."""

    width: int = 64
    n_modes: int = 8
    n_layers: int = 2


def resolve_fno_arch(config: Any) -> FNOArchConfig:
    """Resolve FNO architecture from an OPSCountConfig or compatible object.

    Uses tree_leaf_fno_* if set, otherwise falls back to fno_*.
    This is the ONLY place this resolution logic should live.
    """
    tree_w = getattr(config, "tree_leaf_fno_width", None)
    tree_m = getattr(config, "tree_leaf_fno_n_modes", None)
    tree_l = getattr(config, "tree_leaf_fno_n_layers", None)
    return FNOArchConfig(
        width=int(tree_w) if tree_w is not None else int(config.fno_width),
        n_modes=int(tree_m) if tree_m is not None else int(config.fno_n_modes),
        n_layers=int(tree_l) if tree_l is not None else int(config.fno_n_layers),
    )


def resolve_fno_arch_from_mapping(
    mapping: Mapping[str, Any],
    *,
    fallback_width: int = 64,
    fallback_n_modes: int = 8,
    fallback_n_layers: int = 2,
) -> FNOArchConfig:
    """Resolve FNO architecture from a dict/mapping (e.g. config dicts, payloads).

    Checks tree_leaf_fno_* first, then fno_*, then provided fallbacks.
    """
    def _resolve(tree_key: str, base_key: str, fallback: int) -> int:
        tree_val = mapping.get(tree_key)
        if tree_val is not None and tree_val != "" and tree_val != 0:
            return int(tree_val)
        base_val = mapping.get(base_key)
        if base_val is not None and base_val != "" and base_val != 0:
            return int(base_val)
        return fallback

    return FNOArchConfig(
        width=_resolve("tree_leaf_fno_width", "fno_width", fallback_width),
        n_modes=_resolve("tree_leaf_fno_n_modes", "fno_n_modes", fallback_n_modes),
        n_layers=_resolve("tree_leaf_fno_n_layers", "fno_n_layers", fallback_n_layers),
    )
