from __future__ import annotations

import json
import inspect
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from treepo.core.refs import BenchmarkRef, MethodRef, ResultRow


@dataclass(frozen=True)
class SamplingPlan:
    seed: int | None = None
    split: str = ""
    strategy: str = ""
    sample_budget: int | None = None
    sampling_probability: float | None = None
    unit: str = ""
    frame: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))


@dataclass(frozen=True)
class NormalizedOutput:
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentContext:
    experiment_id: str
    output_root: str | Path
    benchmark_ref: BenchmarkRef
    method_ref: MethodRef
    sampling: SamplingPlan = field(default_factory=SamplingPlan)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root)

    def train(self, method: Any, train_data: Any, validation_data: Any = None, **kwargs: Any):
        if not hasattr(method, "train"):
            raise TypeError("experiment method must expose train(...)")
        if _looks_like_raw_module_train(method):
            raise TypeError(
                "raw model modules are not experiment methods; wrap the module in an object "
                "whose train(...) owns the experiment training loop"
            )
        result = method.train(train_data, validation_data=validation_data, context=self, **kwargs)
        return self.record(result, phase="train")

    def evaluate(self, method: Any, data: Any, *, split: str = "test", **kwargs: Any):
        if not hasattr(method, "evaluate"):
            raise TypeError("experiment method must expose evaluate(...)")
        result = method.evaluate(data, context=self, split=split, **kwargs)
        return self.record(result, phase="evaluate")

    def predict(self, method: Any, inputs: Any, **kwargs: Any):
        if not hasattr(method, "predict"):
            raise TypeError("experiment method must expose predict(...)")
        result = method.predict(inputs, context=self, **kwargs)
        return self.record(result, phase="predict")

    def record(self, result: Any, *, phase: str = "evaluate") -> NormalizedOutput:
        normalized = normalize_output(result)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.output_dir / "experiment_manifest.json", self._manifest())
        _write_json(
            self.output_dir / "experiment_status.json",
            {
                "experiment_id": self.experiment_id,
                "state": "completed",
                "metadata": {
                    **dict(self.metadata),
                    "sampling": self.sampling.to_dict(),
                },
            },
        )
        _write_json(self.output_dir / "artifacts.json", dict(normalized.artifacts))
        rows = [
            ResultRow(
                experiment_id=self.experiment_id,
                phase=str(phase),
                metric_name=str(key),
                metric_value=value,
                benchmark_ref=self.benchmark_ref,
                method_ref=self.method_ref,
                seed=self.sampling.seed,
                split=self.sampling.split,
                metadata=dict(normalized.metadata),
            ).to_dict()
            for key, value in normalized.metrics.items()
        ]
        with (self.output_dir / "results.jsonl").open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return normalized

    def call_metadata(
        self,
        *,
        role: str,
        request_kind: str,
        method_id: str | None = None,
        problem_id: str = "",
        node_id: str = "",
    ) -> dict[str, Any]:
        return _drop_empty(
            {
                "experiment_id": self.experiment_id,
                "role": role,
                "request_kind": request_kind,
                "method_id": method_id or self.method_ref.to_dict().get("method_id", ""),
                "problem_id": problem_id,
                "node_id": node_id,
            }
        )

    def _manifest(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "benchmark_ref": self.benchmark_ref.to_dict(),
            "method_ref": self.method_ref.to_dict(),
            "metadata": {
                **dict(self.metadata),
                "sampling": self.sampling.to_dict(),
            },
        }


def normalize_output(result: Any) -> NormalizedOutput:
    if isinstance(result, NormalizedOutput):
        return result
    if is_dataclass(result):
        raw = asdict(result)
    elif isinstance(result, Mapping):
        raw = dict(result)
    else:
        raw = {"metrics": {"value": result}}
    metrics = dict(raw.get("metrics") or {})
    artifacts = dict(raw.get("artifacts") or {})
    metadata = dict(raw.get("metadata") or {})
    for key, value in raw.items():
        if key in {"metrics", "artifacts", "metadata", "model"}:
            continue
        if isinstance(value, (int, float, bool)):
            metrics.setdefault(str(key), value)
        elif key.endswith("_path") or key.endswith("_dir") or key in {"output_dir", "checkpoint"}:
            artifacts.setdefault(str(key), str(value))
    return NormalizedOutput(metrics=metrics, artifacts=artifacts, metadata=metadata)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _drop_empty(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if value not in ("", None, {}, [])
    }


def _looks_like_raw_module_train(method: Any) -> bool:
    if not all(hasattr(method, name) for name in ("state_dict", "parameters", "train")):
        return False
    try:
        signature = inspect.signature(method.train)
    except (TypeError, ValueError):
        return False
    params = list(signature.parameters.values())
    return len(params) <= 1 and all(
        param.name == "mode" or param.default is not inspect.Parameter.empty
        for param in params
    )


__all__ = ["ExperimentContext", "NormalizedOutput", "SamplingPlan", "normalize_output"]
