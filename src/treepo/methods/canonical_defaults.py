"""Generic dataclass-from-TOML loading for :mod:`treepo.methods`.

Application families define their own defaults in their own package and use
the generic loader here.

``load_dataclass(path, cls, section=...)``
   hydrates any dataclass from a TOML, with optional dotted-key
   overrides. Recursive: a dataclass field whose type is another
   dataclass is built from a nested table.

Usage::

    from treepo.methods.canonical_defaults import load_dataclass
    from your_package import YourFamilyConfig

    cfg = load_dataclass("config.toml", YourFamilyConfig)
"""

from __future__ import annotations

import typing
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Type, TypeVar

try:  # Python 3.11+ stdlib
    import tomllib as _toml_loader
except ModuleNotFoundError:  # pragma: no cover
    import tomli as _toml_loader  # type: ignore[no-redef]

# ===========================================================================
# Generic loader
# ===========================================================================

_T = TypeVar("_T")


def load_dataclass(
    path: str | Path | None,
    cls: Type[_T],
    *,
    section: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> _T:
    """Load a TOML file into ``cls`` (any dataclass).

    - ``path=None`` returns ``cls()`` (the in-code defaults).
    - ``section`` reads only that top-level table from the TOML.
    - ``overrides`` is a flat dict of dotted-path overrides applied after
      loading; ``None`` values are skipped (so CLI args the user did not
      pass don't clobber TOML values).

    Recursive: when ``cls`` has a field whose type is another dataclass,
    a nested TOML table builds that sub-dataclass. Unknown TOML keys
    raise ``ValueError`` at load time, not at iteration 50.
    """
    if path is None:
        instance: _T = cls()  # type: ignore[call-arg]
    else:
        data = _toml_loader.loads(Path(path).read_text(encoding="utf-8"))
        if section is not None:
            data = data.get(section) or {}
        instance = _build(cls, data, ctx=cls.__name__)
    if overrides:
        instance = _apply_overrides(instance, overrides)
    return instance


def _build(cls: Type[_T], data: Mapping[str, Any], *, ctx: str) -> _T:
    """Recursively instantiate ``cls`` from a mapping."""
    if not is_dataclass(cls):
        raise TypeError(f"{ctx}: expected dataclass, got {cls!r}")
    hints = typing.get_type_hints(cls)
    allowed = {f.name: hints.get(f.name, f.type) for f in fields(cls)}
    unknown = set(data) - set(allowed)
    if unknown:
        raise ValueError(
            f"{ctx}: unknown field(s) {sorted(unknown)} "
            f"(allowed: {sorted(allowed)})"
        )
    built: dict[str, Any] = {}
    for name, ftype in allowed.items():
        if name not in data:
            continue
        raw = data[name]
        if is_dataclass(ftype) and isinstance(raw, Mapping):
            built[name] = _build(ftype, raw, ctx=f"{ctx}.{name}")
        else:
            built[name] = raw
    return cls(**built)  # type: ignore[call-arg]


def _apply_overrides(obj: _T, overrides: Mapping[str, Any]) -> _T:
    """Return a copy of a dataclass tree with non-None overrides applied.

    Keys are dotted paths (``"family.optimizer"``, ``"lm.endpoints"``).
    """
    new = _clone(obj)
    for key, value in overrides.items():
        if value is None:
            continue
        parts = key.split(".")
        cur: Any = new
        for part in parts[:-1]:
            if not hasattr(cur, part):
                raise ValueError(f"unknown override path: {key!r}")
            cur = getattr(cur, part)
        leaf = parts[-1]
        if not hasattr(cur, leaf):
            raise ValueError(f"unknown override field: {key!r}")
        setattr(cur, leaf, value)
    return new


def _clone(obj: Any) -> Any:
    """Shallow-clone every dataclass node in the tree (so overrides don't mutate the input)."""
    if not is_dataclass(obj) or isinstance(obj, type):
        return obj
    return replace(obj, **{f.name: _clone(getattr(obj, f.name)) for f in fields(obj)})


__all__ = [
    "load_dataclass",
]
