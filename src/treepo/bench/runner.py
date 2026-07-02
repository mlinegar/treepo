from __future__ import annotations

import traceback
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from treepo.bench.classical_sketches import (
    ClassicalSketchComparisonConfig,
    run_classical_sketch_comparison,
)
from treepo.bench.env import apply_cpu_thread_limits
from treepo.bench.io import (
    add_runtime_meta,
    atomic_write_text,
    dump_json,
    summary_to_csv_rows_classical_sketches,
    write_csv_rows,
)
from treepo.bench.tasks import (
    TaskBenchmarkConfig,
    list_task_benchmarks,
    run_task_benchmark,
)
from treepo.bench.tasks import (
    experiment_rows as task_experiment_rows,
)

ExperimentName = str
_EXPERIMENT_CLASSICAL_SKETCHES = "classical-sketches"


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
) -> dict[str, object]:
    json_path = Path(json_out)
    csv_path = Path(csv_out)
    validate_benchmark_config(definition, config)

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


def _dataclass_keys(cls: type[object]) -> frozenset[str]:
    return frozenset(field.name for field in fields(cls))


def _run_classical_sketches(
    config: Mapping[str, object],
    output_dir: Path,
) -> BenchmarkResult:
    del output_dir
    cfg = ClassicalSketchComparisonConfig(**dict(config))
    summary = run_classical_sketch_comparison(cfg)
    return BenchmarkResult(
        payload=summary.to_dict(),
        rows=summary_to_csv_rows_classical_sketches(summary),
    )


def _make_task_definition(experiment: ExperimentName) -> BenchmarkDefinition:
    def _run_task(config: Mapping[str, object], output_dir: Path) -> BenchmarkResult:
        cfg = TaskBenchmarkConfig(**dict(config))
        summary = run_task_benchmark(experiment, cfg, output_dir=output_dir)
        return BenchmarkResult(
            payload=summary.to_dict(),
            rows=[dict(row) for row in task_experiment_rows(summary)],
        )

    return BenchmarkDefinition(
        name=experiment,
        allowed_config_keys=_dataclass_keys(TaskBenchmarkConfig),
        run=_run_task,
    )


def _build_benchmarks() -> dict[ExperimentName, BenchmarkDefinition]:
    benchmarks: dict[ExperimentName, BenchmarkDefinition] = {
        _EXPERIMENT_CLASSICAL_SKETCHES: BenchmarkDefinition(
            name=_EXPERIMENT_CLASSICAL_SKETCHES,
            allowed_config_keys=_dataclass_keys(ClassicalSketchComparisonConfig),
            run=_run_classical_sketches,
        ),
    }
    for experiment in list_task_benchmarks():
        benchmarks[str(experiment)] = _make_task_definition(str(experiment))
    return benchmarks


BENCHMARKS: dict[ExperimentName, BenchmarkDefinition] = _build_benchmarks()
VALID_EXPERIMENTS: tuple[ExperimentName, ...] = tuple(BENCHMARKS)


def validate_config_dict(experiment: ExperimentName, config: Mapping[str, object]) -> None:
    try:
        definition = BENCHMARKS[str(experiment)]
    except KeyError as exc:
        raise ValueError(f"unknown experiment: {experiment!r}") from exc
    validate_benchmark_config(definition, config)


def run_single(
    *,
    experiment: ExperimentName,
    config: Mapping[str, object],
    json_out: Path,
    csv_out: Path,
    print_json: bool = False,
) -> dict[str, object]:
    apply_cpu_thread_limits(threads=1)
    try:
        definition = BENCHMARKS[str(experiment)]
    except KeyError as exc:
        raise ValueError(f"unknown experiment: {experiment!r}") from exc
    return run_benchmark_cell(
        definition,
        config,
        json_out=Path(json_out),
        csv_out=Path(csv_out),
        print_json=print_json,
    )


__all__ = [
    "BENCHMARKS",
    "BenchmarkDefinition",
    "BenchmarkResult",
    "VALID_EXPERIMENTS",
    "run_benchmark_cell",
    "run_single",
    "validate_benchmark_config",
    "validate_config_dict",
]
