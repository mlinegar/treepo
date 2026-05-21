from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from treepo.bench.io import load_yaml_or_json


@dataclass(frozen=True)
class SweepOutputSpec:
    layout: str = "hash"  # "hash" | "identifiable_zero"


@dataclass(frozen=True)
class SweepSpec:
    base: Dict[str, Any]
    grid: Dict[str, List[Any]]
    output: SweepOutputSpec


def load_sweep_spec(path: Path) -> SweepSpec:
    raw = load_yaml_or_json(Path(path))
    if not isinstance(raw, dict):
        raise ValueError("sweep spec must be a YAML/JSON mapping with keys: base, grid, output")

    allowed = {"base", "grid", "output"}
    unknown = sorted([k for k in raw.keys() if k not in allowed])
    if unknown:
        raise ValueError(f"unknown sweep spec keys: {unknown} (allowed: {sorted(allowed)})")

    base = raw.get("base", {})
    grid = raw.get("grid", {})
    output_raw = raw.get("output", {}) or {}

    if not isinstance(base, dict):
        raise ValueError("spec.base must be a mapping")
    if not isinstance(grid, dict):
        raise ValueError("spec.grid must be a mapping of key -> list")
    if not isinstance(output_raw, dict):
        raise ValueError("spec.output must be a mapping")

    grid_norm: Dict[str, List[Any]] = {}
    for k, v in grid.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError("spec.grid keys must be non-empty strings")
        if not isinstance(v, list):
            raise ValueError(f"spec.grid[{k!r}] must be a list")
        grid_norm[k] = list(v)

    layout = str(output_raw.get("layout", "hash")).strip() or "hash"
    if layout not in {"hash", "identifiable_zero"}:
        raise ValueError("spec.output.layout must be one of: 'hash', 'identifiable_zero'")

    return SweepSpec(base=dict(base), grid=grid_norm, output=SweepOutputSpec(layout=layout))

