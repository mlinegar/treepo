"""Extension stub for the optional TRL family."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class TRLFamilyConfig:
    """Small config record for downstream registration code."""

    metadata: Mapping[str, Any] = field(default_factory=dict)


def build_trl_family(backend_config: Mapping[str, Any]) -> Any:
    del backend_config
    raise ImportError(
        "family='trl' is an optional application family and is not included "
        "in treepo. Register a TRL family from an external package before use."
    )


def __getattr__(name: str) -> Any:
    if name == "TRLFamily":
        raise ImportError(
            "TRLFamily is not included in treepo; import it from the "
            "external package that owns TRL dependencies."
        )
    raise AttributeError(name)


__all__ = ["TRLFamily", "TRLFamilyConfig", "build_trl_family"]
