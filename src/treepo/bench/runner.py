from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Mapping

from treepo.bench.classical_sketches import (
    ClassicalSketchComparisonConfig,
    run_classical_sketch_comparison,
)
from treepo.bench.env import apply_cpu_thread_limits
from treepo.bench.grid import (
    BenchmarkDefinition,
    BenchmarkResult,
    run_benchmark_cell,
    validate_benchmark_config,
)
from treepo.bench.io import (
    summary_to_csv_rows_classical_sketches,
    summary_to_csv_rows_task,
)
from treepo.bench.tasks import (
    TaskBenchmarkConfig,
    list_task_benchmarks,
    run_task_benchmark,
    task_benchmark_config_keys,
)

ExperimentName = str
EXPERIMENT_CLASSICAL_SKETCHES = "classical-sketches"
BASE_EXPERIMENTS: tuple[ExperimentName, ...] = (
    EXPERIMENT_CLASSICAL_SKETCHES,
)


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
        payload=json.loads(summary.to_json()),
        rows=summary_to_csv_rows_classical_sketches(summary),
    )


def _make_task_definition(experiment: ExperimentName) -> BenchmarkDefinition:
    def _run_task(config: Mapping[str, object], output_dir: Path) -> BenchmarkResult:
        cfg = TaskBenchmarkConfig(**dict(config))
        summary = run_task_benchmark(experiment, cfg, output_dir=output_dir)
        return BenchmarkResult(
            payload=summary.to_dict(),
            rows=summary_to_csv_rows_task(summary),
        )

    return BenchmarkDefinition(
        name=experiment,
        allowed_config_keys=frozenset(task_benchmark_config_keys(experiment)),
        run=_run_task,
    )


def _build_benchmarks() -> dict[ExperimentName, BenchmarkDefinition]:
    benchmarks: dict[ExperimentName, BenchmarkDefinition] = {
        EXPERIMENT_CLASSICAL_SKETCHES: BenchmarkDefinition(
            name=EXPERIMENT_CLASSICAL_SKETCHES,
            allowed_config_keys=_dataclass_keys(ClassicalSketchComparisonConfig),
            run=_run_classical_sketches,
        ),
    }
    for experiment in list_task_benchmarks():
        benchmarks[str(experiment)] = _make_task_definition(str(experiment))
    return benchmarks


BENCHMARKS: dict[ExperimentName, BenchmarkDefinition] = _build_benchmarks()
VALID_EXPERIMENTS: tuple[ExperimentName, ...] = tuple(BENCHMARKS)


def allowed_config_keys(experiment: ExperimentName) -> set[str]:
    try:
        return set(BENCHMARKS[str(experiment)].allowed_config_keys)
    except KeyError as exc:
        raise ValueError(f"unknown experiment: {experiment!r}") from exc


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
    "BASE_EXPERIMENTS",
    "BENCHMARKS",
    "EXPERIMENT_CLASSICAL_SKETCHES",
    "VALID_EXPERIMENTS",
    "allowed_config_keys",
    "run_single",
    "validate_config_dict",
]
