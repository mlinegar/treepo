"""Lightweight defaults and dataclass loading for :mod:`treepo.methods`.

`treepo` keeps only dependency-light defaults. Application families can define
their richer defaults in their own package and still use the generic loader
here.

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
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Type, TypeVar

try:  # Python 3.11+ stdlib
    import tomllib as _toml_loader
except ModuleNotFoundError:  # pragma: no cover
    import tomli as _toml_loader  # type: ignore[no-redef]

# ===========================================================================
# Defaults
# ===========================================================================

DEFAULT_BATCH_MAX_CONCURRENT = 512
DEFAULT_BATCH_SIZE = 64
DEFAULT_BATCH_TIMEOUT_SECONDS = 0.02
DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS = 300.0
DEFAULT_BATCH_ROUTING_POLICY = "affinity_load_aware"
BATCH_DEFAULTS: dict[str, Any] = {
    "batch_size": DEFAULT_BATCH_SIZE,
    "batch_max_concurrent": DEFAULT_BATCH_MAX_CONCURRENT,
    "batch_timeout": DEFAULT_BATCH_TIMEOUT_SECONDS,
    "batch_request_timeout": DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS,
    "batch_routing_policy": DEFAULT_BATCH_ROUTING_POLICY,
}

CONCAT_RATIO: float = 2.0
DEFAULT_TARGET_RATIO: float = 0.15
DEFAULT_PROMPT_OVERHEAD_TOKENS: int = 1500
DEFAULT_MANIFESTO_WORKERS: int = 4
DEFAULT_SUMMARY_WORKERS: int = 4
DEFAULT_SCORING_WORKERS: int = 4
DEFAULT_SCORER_MAX_TOKENS: int = 256
GEPA_STRONG_DEFAULTS: dict[str, Any] = {
    "use_merge": True,
    "max_merge_invocations": 5,
    "track_stats": True,
    "reflection_minibatch_size": 8,
    "use_wandb": False,
    "use_mlflow": False,
}


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


# ===========================================================================
# Scenario wrappers — only the knobs upstream classes DON'T carry
# ===========================================================================


@dataclass
class LmSection:
    """LM endpoint config (shared across LLM-driven families)."""

    model: str = "nvidia/Gemma-4-31B-IT-NVFP4"
    endpoints: list[str] = field(
        default_factory=lambda: ["http://localhost:8000/v1"]
    )
    temperature: float = 0.0
    cache: bool = False


# ===========================================================================
# DSPy: just an LM-config helper. Strong GEPA defaults are now baked into
# ``DSPyFamilyConfig.gepa_kwargs`` (a field default factory); no monkey-patch.
# ===========================================================================


def build_lm_config_dict(lm: LmSection, *, max_tokens: int) -> dict[str, Any]:
    """Build the ``lm_config`` dict that ``DSPyFamilyConfig.lm_config`` expects."""
    return {
        "model": f"openai/{lm.model}",
        "api_bases": list(lm.endpoints),
        "api_key": "EMPTY",
        "temperature": float(lm.temperature),
        "max_tokens": int(max_tokens),
        "cache": bool(lm.cache),
    }


__all__ = [
    # Constants
    "GEPA_STRONG_DEFAULTS", "BATCH_DEFAULTS",
    "CONCAT_RATIO", "DEFAULT_TARGET_RATIO", "DEFAULT_SCORER_MAX_TOKENS",
    "DEFAULT_PROMPT_OVERHEAD_TOKENS", "DEFAULT_MANIFESTO_WORKERS",
    "DEFAULT_SUMMARY_WORKERS", "DEFAULT_SCORING_WORKERS",
    # Generic loader
    "load_dataclass",
    # Cross-family scenario wrappers
    "LmSection",
    # DSPy LM-config helper (strong GEPA defaults are now field defaults on
    # DSPyFamilyConfig — no monkey-patch needed).
    "build_lm_config_dict",
]
