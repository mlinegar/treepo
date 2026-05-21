"""Back-compat re-export for dimension invariant guards."""

from __future__ import annotations

from treepo._research.unified_g_v1.dimension_guards import (
    DimensionInvariantWarning,
    promote_dim,
    require_dim,
)


__all__ = ["DimensionInvariantWarning", "promote_dim", "require_dim"]
