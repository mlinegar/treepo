from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from treepo.methods.oracles import score_oracle


@dataclass(frozen=True)
class TaskBenchmarkConfig:
    method: str = "oracle"
    scorer: str = ""
    seed: int = 0
    split: str = "test"
    n_trees: int = 8
    task_config: Mapping[str, Any] = field(default_factory=dict)
    method_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskBenchmarkSpec:
    name: str
    default_method: str
    default_scorer: str
    default_task_config: Mapping[str, Any] = field(default_factory=dict)
    allowed_task_config_keys: tuple[str, ...] = ()
    supported_scorers: tuple[str, ...] = ()
    build_method_config: (
        Callable[[TaskBenchmarkConfig, Mapping[str, Any], str, Path | None], Mapping[str, Any]]
        | None
    ) = None


@dataclass(frozen=True)
class TaskBenchmarkSummary:
    experiment: str
    config: dict[str, Any]
    result: dict[str, Any]
    rows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "config": self.config,
            "result": self.result,
            "rows": list(self.rows),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


_REGISTRY: dict[str, TaskBenchmarkSpec] = {}


def register_task_benchmark(spec: TaskBenchmarkSpec, *, replace: bool = False) -> None:
    name = _normalize(spec.name)
    if not name:
        raise ValueError("task benchmark name must be non-empty")
    if name in _REGISTRY and not bool(replace):
        raise ValueError(f"task benchmark {name!r} is already registered")
    _REGISTRY[name] = dataclasses.replace(spec, name=name)


def list_task_benchmarks() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def run_task_benchmark(
    task: str,
    config: TaskBenchmarkConfig,
    *,
    output_dir: str | Path | None = None,
) -> TaskBenchmarkSummary:
    spec = _lookup(task)
    method = _normalize(config.method or spec.default_method)
    if method != _normalize(spec.default_method):
        raise ValueError(
            f"task benchmark {spec.name!r} supports method={spec.default_method!r}; "
            f"got {config.method!r}"
        )

    scorer = _normalize(config.scorer or spec.default_scorer)
    supported = tuple(_normalize(name) for name in spec.supported_scorers or (spec.default_scorer,))
    if supported and scorer not in supported:
        raise ValueError(
            f"task benchmark {spec.name!r} supports scorer(s) {sorted(supported)}; "
            f"got {config.scorer!r}"
        )

    task_config = _merged_task_config(spec, config.task_config)
    out_path = Path(output_dir) if output_dir is not None else None
    if spec.build_method_config is not None:
        method_config = dict(spec.build_method_config(config, task_config, scorer, out_path))
    else:
        method_config = _default_oracle_method_config(config, task_config, scorer, out_path)

    result = score_oracle(method_config)
    result_payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    config_payload = _config_payload(config, task_config=task_config, scorer=scorer, method=method)
    row = _row_from_result(
        experiment=spec.name,
        method=method,
        scorer=scorer,
        config=config,
        task_config=task_config,
        result=result_payload,
    )
    return TaskBenchmarkSummary(
        experiment=spec.name,
        config=config_payload,
        result=result_payload,
        rows=(row,),
    )


def experiment_rows(summary: TaskBenchmarkSummary) -> list[dict[str, Any]]:
    return [dict(row) for row in summary.rows]


def _lookup(task: str) -> TaskBenchmarkSpec:
    name = _normalize(task)
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown task benchmark {task!r}; available: {', '.join(list_task_benchmarks())}"
        )
    return _REGISTRY[name]


def _normalize(value: str) -> str:
    return str(value).strip().lower()


def _merged_task_config(spec: TaskBenchmarkSpec, supplied: Mapping[str, Any]) -> dict[str, Any]:
    supplied_dict = {str(key): value for key, value in dict(supplied or {}).items()}
    allowed = set(spec.allowed_task_config_keys)
    unknown = sorted(key for key in supplied_dict if key not in allowed)
    if unknown:
        raise ValueError(
            f"unknown task_config keys for {spec.name!r}: {unknown}; allowed: {sorted(allowed)}"
        )
    merged = dict(spec.default_task_config or {})
    merged.update(supplied_dict)
    return merged


def _default_oracle_method_config(
    config: TaskBenchmarkConfig,
    task_config: Mapping[str, Any],
    scorer: str,
    output_dir: Path | None,
) -> dict[str, Any]:
    method_config = dict(config.method_config or {})
    method_config.update({
        "oracle_name": scorer,
        "seed": int(config.seed),
        "split": str(config.split),
        "n_trees": int(config.n_trees),
    })
    method_config.update(dict(task_config))
    if output_dir is not None:
        method_config["output_dir"] = str(output_dir)
    return method_config


def _config_payload(
    config: TaskBenchmarkConfig,
    *,
    task_config: Mapping[str, Any],
    scorer: str,
    method: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "scorer": scorer,
        "seed": int(config.seed),
        "split": str(config.split),
        "n_trees": int(config.n_trees),
        "task_config": dict(task_config),
        "method_config": dict(config.method_config or {}),
    }


def _row_from_result(
    *,
    experiment: str,
    method: str,
    scorer: str,
    config: TaskBenchmarkConfig,
    task_config: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment": experiment,
        "task": experiment,
        "method": method,
        "scorer": scorer,
        "seed": int(config.seed),
        "split": str(config.split),
        "n_trees": int(config.n_trees),
        "status": str(result.get("status") or ""),
        "manifest_path": str(result.get("manifest_path") or ""),
    }
    if method == "oracle":
        row["oracle_name"] = scorer
    row.update({str(key): value for key, value in task_config.items()})
    row.update({str(key): value for key, value in dict(result.get("metrics") or {}).items()})
    return row


register_task_benchmark(
    TaskBenchmarkSpec(
        name="markov",
        default_method="oracle",
        default_scorer="markov_changepoint_count",
        supported_scorers=("markov_changepoint_count",),
        default_task_config={
            "n_states": 4,
            "doc_tokens": 128,
            "doc_unit_kind": "token",
            "leaf_unit_count": 16,
            "transition_prob": 0.15,
            "vocabulary_size": 256,
        },
        allowed_task_config_keys=(
            "n_states",
            "doc_tokens",
            "doc_unit_kind",
            "leaf_unit_count",
            "transition_prob",
            "vocabulary_size",
        ),
    )
)


__all__ = [
    "TaskBenchmarkConfig",
    "TaskBenchmarkSpec",
    "TaskBenchmarkSummary",
    "experiment_rows",
    "list_task_benchmarks",
    "register_task_benchmark",
    "run_task_benchmark",
]
