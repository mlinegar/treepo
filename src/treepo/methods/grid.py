"""Small helpers for reproducible method grids."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class GridCell:
    values: Mapping[str, Any]
    output_dir: Path


def iter_grid(
    axes: Mapping[str, Sequence[Any]],
    *,
    output_root: str | Path | None = None,
) -> Iterable[GridCell]:
    normalized = {str(key): tuple(values) for key, values in axes.items()}
    for key, values in normalized.items():
        if not values:
            raise ValueError(f"grid axis {key!r} must be non-empty")
    keys = tuple(normalized)
    root = Path(output_root) if output_root is not None else Path(".")
    if not keys:
        yield GridCell(values={}, output_dir=root)
        return
    for combo in product(*(normalized[key] for key in keys)):
        values = dict(zip(keys, combo))
        yield GridCell(values=values, output_dir=root / grid_cell_name(values))


def grid_cell_name(values: Mapping[str, Any]) -> str:
    if not values:
        return "cell"
    return "__".join(f"{_slug(key)}_{_slug(value)}" for key, value in values.items())


def write_grid_outputs(
    *,
    json_out: str | Path,
    csv_out: str | Path,
    payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    json_path = Path(json_out)
    csv_path = Path(csv_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_path, rows)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key_str = str(key)
            if key_str not in seen:
                seen.add(key_str)
                fieldnames.append(key_str)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _slug(value: Any) -> str:
    text = str(value).strip().lower().replace(".", "p")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "none"


__all__ = ["GridCell", "grid_cell_name", "iter_grid", "write_grid_outputs"]
