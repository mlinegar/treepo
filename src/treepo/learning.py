from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class FitConfig:
    mode: str = ""
    experiment: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)
    output_dir: str | Path = "outputs/treepo_fit"
    json_out: str | Path | None = None
    csv_out: str | Path | None = None
    spec: Mapping[str, Any] = field(default_factory=dict)
    train_data: Any = None
    eval_data: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FitConfig":
        data = dict(payload or {})
        nested = data.get("fit")
        if isinstance(nested, Mapping):
            merged = dict(data)
            merged.update(dict(nested))
            data = merged
        config = data.get("config")
        spec = data.get("spec")
        if config is None and not spec:
            config = {
                k: v
                for k, v in data.items()
                if k
                not in {
                    "mode",
                    "kind",
                    "experiment",
                    "output_dir",
                    "json_out",
                    "csv_out",
                    "metadata",
                    "fit",
                }
            }
        return cls(
            mode=str(data.get("mode") or data.get("kind") or ""),
            experiment=str(data.get("experiment") or ""),
            config=dict(config or {}),
            output_dir=data.get("output_dir") or "outputs/treepo_fit",
            json_out=data.get("json_out"),
            csv_out=data.get("csv_out"),
            spec=dict(spec or {}),
            train_data=data.get("train_data"),
            eval_data=data.get("eval_data"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class FitResult:
    status: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    history: tuple[Mapping[str, Any], ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)
    manifest_path: str | None = None
    mode: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "mode": str(self.mode),
            "metrics": dict(self.metrics or {}),
            "artifacts": dict(self.artifacts or {}),
            "history": [dict(item) for item in self.history],
            "summary": _jsonable(dict(self.summary or {})),
            "manifest_path": self.manifest_path,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return value


def _infer_mode(cfg: FitConfig, *, task: str | None = None, backend: str | None = None) -> str:
    if cfg.mode:
        return str(cfg.mode).strip().lower()
    if task:
        return "paper_experiment"
    payload = dict(cfg.config or cfg.spec or {})
    if "methods" in payload and "benchmark" in payload:
        return "runtime"
    if "family" in payload and ("schedule" in payload or "backend_config" in payload):
        return "learning"
    if cfg.experiment or "experiment" in payload:
        return "paper_experiment"
    if backend in {"runtime", "longbench"}:
        return "runtime"
    raise ValueError("could not infer fit mode; provide mode='paper_experiment', 'runtime', or 'learning'")


def _as_fit_config(config: FitConfig | Mapping[str, Any] | None, **kwargs: Any) -> FitConfig:
    payload: dict[str, Any] = {}
    if isinstance(config, FitConfig):
        payload = asdict(config)
    elif isinstance(config, Mapping):
        payload = dict(config)
    elif config is not None:
        raise TypeError(f"fit config must be a mapping or FitConfig, got {type(config).__name__}")
    payload.update({k: v for k, v in kwargs.items() if v is not None})
    return FitConfig.from_mapping(payload)


def fit(
    config: FitConfig | Mapping[str, Any] | None = None,
    *,
    task: str | None = None,
    backend: str | None = None,
    output_dir: str | Path | None = None,
    train_data: Any = None,
    eval_data: Any = None,
    **kwargs: Any,
) -> FitResult:
    """Run a TreePO exercise through the smallest compatible public envelope.

    The dispatcher follows the config shapes already used by the paper:

    - ``{"experiment": ..., "config": ...}`` routes through ``treepo-bench``.
    - LongBench/runtime configs with ``methods`` and ``benchmark`` route through
      ``treepo.runtime``.
    - f/g ladder specs with ``family`` and ``schedule`` route through the
      existing ``src.ctreepo.learning.fit`` implementation when the monorepo is
      available.
    """

    cfg = _as_fit_config(
        config,
        output_dir=output_dir,
        train_data=train_data,
        eval_data=eval_data,
        **kwargs,
    )
    mode = _infer_mode(cfg, task=task, backend=backend)
    if mode in {"paper", "paper_experiment", "bench", "suite"}:
        return _fit_paper_experiment(cfg, task=task, mode=mode)
    if mode in {"runtime", "longbench", "runtime_eval"}:
        return _fit_runtime(cfg, mode=mode)
    if mode in {"learning", "ladder", "family_runtime", "fg"}:
        return _fit_learning(cfg, mode=mode)
    raise ValueError(f"unsupported fit mode: {mode!r}")


def _fit_paper_experiment(cfg: FitConfig, *, task: str | None, mode: str) -> FitResult:
    runner = importlib.import_module("treepo.bench.runner")
    experiment = str(task or cfg.experiment or cfg.config.get("experiment") or "")
    if not experiment:
        raise ValueError("paper_experiment fit requires an experiment/task name")
    run_config = dict(cfg.config.get("config") or cfg.config)
    run_config.pop("experiment", None)
    output_root = Path(cfg.output_dir)
    json_out = Path(cfg.json_out) if cfg.json_out is not None else output_root / experiment / "summary.json"
    csv_out = Path(cfg.csv_out) if cfg.csv_out is not None else output_root / experiment / "summary.csv"
    result = runner.run_single(
        experiment=experiment,
        config=run_config,
        json_out=json_out,
        csv_out=csv_out,
    )
    summary = _read_json(json_out)
    return FitResult(
        status=str(result.get("status", "ok")),
        artifacts={"json_out": str(json_out), "csv_out": str(csv_out)},
        summary=summary,
        mode=mode,
    )


def _fit_runtime(cfg: FitConfig, *, mode: str) -> FitResult:
    runtime = importlib.import_module("treepo.runtime")
    run_config = dict(cfg.config or cfg.spec or {})
    output_root = Path(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = runtime.run_runtime_eval(run_config)
    summary_payload = summary.to_dict()
    json_out = Path(cfg.json_out) if cfg.json_out is not None else output_root / "runtime_summary.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = {str(k): float(v) for k, v in dict(summary.metrics).items() if isinstance(v, (int, float))}
    return FitResult(
        status="ok",
        metrics=metrics,
        artifacts={"json_out": str(json_out)},
        summary=summary_payload,
        mode=mode,
    )


def _fit_learning(cfg: FitConfig, *, mode: str) -> FitResult:
    learning = importlib.import_module("treepo._research.ctreepo.learning")
    spec = dict(cfg.spec or cfg.config or {})
    if cfg.train_data is not None:
        spec["train_data"] = cfg.train_data
    if cfg.eval_data is not None:
        spec["eval_data"] = cfg.eval_data
    result = learning.fit(spec, output_dir=cfg.output_dir)
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    return FitResult(
        status=str(payload.get("status", "ok")),
        metrics=dict(payload.get("metrics") or {}),
        artifacts=dict(payload.get("artifacts") or {}),
        history=tuple(dict(item) for item in list(payload.get("history") or [])),
        summary=dict(payload.get("summary") or {}),
        manifest_path=payload.get("manifest_path"),
        mode=mode,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return payload if isinstance(payload, dict) else {"value": payload}


__all__ = ["FitConfig", "FitResult", "fit"]
