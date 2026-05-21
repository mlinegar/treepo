from __future__ import annotations

import importlib
from types import ModuleType


class MissingOptionalDependency(ImportError):
    """Raised when an optional package extra is needed but not installed."""


def require_optional(module_name: str, *, extra: str, purpose: str = "") -> ModuleType:
    """Import an optional dependency or raise an actionable package-extra error."""

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - exercised by callers
        detail = f" for {purpose}" if purpose else ""
        raise MissingOptionalDependency(
            f"`{module_name}` is required{detail}. Install with `pip install treepo[{extra}]`."
        ) from exc


__all__ = ["MissingOptionalDependency", "require_optional"]
