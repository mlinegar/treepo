from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from treepo.bench.io import (
    add_runtime_meta,
    atomic_write_text,
    dump_json,
    write_csv_rows,
)


@dataclass(frozen=True)
class BenchmarkResult:
    payload: Mapping[str, Any]
    rows: Sequence[Mapping[str, object]]


@dataclass(frozen=True)
class BenchmarkDefinition:
    name: str
    allowed_config_keys: frozenset[str]
    run: Callable[[Mapping[str, object], Path], BenchmarkResult]


def validate_benchmark_config(
    definition: BenchmarkDefinition,
    config: Mapping[str, object],
) -> None:
    unknown = sorted(str(key) for key in config if str(key) not in definition.allowed_config_keys)
    if unknown:
        raise ValueError(f"unknown config keys for {definition.name}: {unknown}")


def run_benchmark_cell(
    definition: BenchmarkDefinition,
    config: Mapping[str, object],
    *,
    json_out: str | Path,
    csv_out: str | Path,
    print_json: bool = False,
    skip_existing: bool = False,
) -> dict[str, object]:
    json_path = Path(json_out)
    csv_path = Path(csv_out)
    validate_benchmark_config(definition, config)
    if skip_existing and json_path.exists() and csv_path.exists():
        return {"status": "skipped", "json_out": str(json_path), "csv_out": str(csv_path)}

    try:
        result = definition.run(dict(config), json_path.parent / f"{definition.name}_work")
        payload = add_runtime_meta(dict(result.payload))
        atomic_write_text(json_path, dump_json(payload))
        write_csv_rows(csv_path, [dict(row) for row in result.rows])
    except Exception:
        err_path = (
            json_path.parent / "error.txt"
            if json_path.name == "summary.json"
            else json_path.with_suffix(".error.txt")
        )
        atomic_write_text(err_path, traceback.format_exc())
        raise

    if print_json:
        print(json_path.read_text(encoding="utf-8"))
    return {"status": "ok", "json_out": str(json_path), "csv_out": str(csv_path)}


__all__ = [
    "BenchmarkDefinition",
    "BenchmarkResult",
    "run_benchmark_cell",
    "validate_benchmark_config",
]
