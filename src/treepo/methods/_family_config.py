"""The single config-ingestion mechanism for family builders.

Every family config is a dataclass; every builder ingests it the same way:
merge the optional nested payload (``backend_config["<family>_config"]``)
with flat field-named keys, filtered to the dataclass's fields, with flat
keys taking precedence. Legacy flat spellings are declared as aliases, not
hand-rolled per family.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping, Type, TypeVar

_ConfigT = TypeVar("_ConfigT")


def dataclass_field_subset(values: Mapping[str, Any] | None, config_cls: Type[Any]) -> dict[str, Any]:
    """Return the entries of ``values`` naming actual fields of ``config_cls``."""

    allowed = {f.name for f in fields(config_cls)}
    return {key: value for key, value in dict(values or {}).items() if key in allowed}


def coerce_family_config(
    config_cls: Type[_ConfigT],
    backend_config: Mapping[str, Any] | None,
    *,
    nested_key: str,
    aliases: Mapping[str, str] | None = None,
) -> _ConfigT:
    """Build ``config_cls`` from ``backend_config``.

    Precedence: flat field-named keys override the nested
    ``backend_config[nested_key]`` payload. ``aliases`` maps legacy flat
    spellings to canonical field names. Keys naming no config field are
    ignored — they belong to the runtime (``predict_fn``, ``output_dir``,
    ``objective``, ...).
    """

    payload = dict(backend_config or {})
    for alias, canonical in dict(aliases or {}).items():
        if payload.get(alias) is not None and canonical not in payload:
            payload[canonical] = payload[alias]
    nested = payload.get(nested_key)
    if nested is not None and not isinstance(nested, Mapping):
        raise TypeError(
            f"backend_config[{nested_key!r}] must be a mapping; got {type(nested).__name__}"
        )
    data = dataclass_field_subset(nested, config_cls)
    data.update(dataclass_field_subset(payload, config_cls))
    return config_cls(**data)


__all__ = ["coerce_family_config", "dataclass_field_subset"]
