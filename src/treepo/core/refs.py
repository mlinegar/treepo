from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class BenchmarkRef:
    family: str
    name: str = ""
    split: str = ""
    dataset_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(asdict(self))


@dataclass(frozen=True)
class MethodRef:
    family: str
    method_id: str = ""
    variant: str = ""
    adapter: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data["method_id"]:
            data["method_id"] = _join_id(self.family, self.variant)
        return _drop_empty(data)


@dataclass(frozen=True)
class ResultRow:
    experiment_id: str
    phase: str
    metric_name: str
    metric_value: Any
    benchmark_ref: BenchmarkRef
    method_ref: MethodRef
    seed: int | None = None
    split: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["benchmark_ref"] = self.benchmark_ref.to_dict()
        data["method_ref"] = self.method_ref.to_dict()
        return _drop_empty(data)


def _join_id(*parts: str) -> str:
    return "::".join(str(part) for part in parts if str(part))


def _drop_empty(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if value not in ("", None, {}, [])
    }


__all__ = ["BenchmarkRef", "MethodRef", "ResultRow"]
