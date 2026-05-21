from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from treepo._research.experiments.contracts import BenchmarkRef, MethodRef
from treepo._research.experiments.method_api import (
    ExperimentMethodRun,
    ExperimentMethodSpec,
    _record_method_output,
    _run_method_phase,
)
from treepo._research.experiments.roles import method_ref_with_roles


@dataclass(frozen=True)
class SamplingPlan:
    """Experiment-level sampling design recorded with every experiment."""

    seed: int | None = None
    split: str = ""
    strategy: str = ""
    sample_budget: int | None = None
    sampling_probability: float | None = None
    unit: str = ""
    frame: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: "SamplingPlan | Mapping[str, Any] | None") -> "SamplingPlan":
        if isinstance(value, SamplingPlan):
            return value
        payload = dict(value or {})
        return cls(
            seed=(
                None
                if payload.get("seed") in {None, ""}
                else int(payload.get("seed"))
            ),
            split=str(payload.get("split", "") or ""),
            strategy=str(payload.get("strategy", "") or ""),
            sample_budget=(
                None
                if payload.get("sample_budget") in {None, ""}
                else int(payload.get("sample_budget"))
            ),
            sampling_probability=(
                None
                if payload.get("sampling_probability") in {None, ""}
                else float(payload.get("sampling_probability"))
            ),
            unit=str(payload.get("unit", "") or ""),
            frame=str(payload.get("frame", "") or ""),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "split": self.split,
            "strategy": self.strategy,
            "sample_budget": self.sample_budget,
            "sampling_probability": self.sampling_probability,
            "unit": self.unit,
            "frame": self.frame,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExperimentContext:
    """Single public context for experiment lifecycle entrypoints."""

    output_root: Path
    benchmark_ref: BenchmarkRef
    method_ref: MethodRef
    title: str = ""
    adapter_id: str = "experiment_method"
    phases: tuple[str, ...] = ("train",)
    sampling: SamplingPlan | Mapping[str, Any] | None = None
    report_profiles: tuple[str, ...] = ()
    roles: Mapping[str, Any] | None = None
    oracle: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None
    launch_command: tuple[str, ...] = ()
    call_sink: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_root",
            Path(self.output_root).expanduser().resolve(),
        )
        object.__setattr__(self, "sampling", SamplingPlan.from_value(self.sampling))
        if self.roles is not None or self.oracle is not None:
            object.__setattr__(
                self,
                "method_ref",
                method_ref_with_roles(
                    self.method_ref,
                    roles=self.roles,
                    oracle=self.oracle,
                ),
            )

    def with_method(self, method_ref: MethodRef) -> "ExperimentContext":
        return replace(self, method_ref=method_ref)

    def with_sampling(self, sampling: SamplingPlan | Mapping[str, Any] | None) -> "ExperimentContext":
        return replace(self, sampling=SamplingPlan.from_value(sampling))

    def sampling_dict(self) -> dict[str, Any]:
        sampling = SamplingPlan.from_value(self.sampling)
        return sampling.to_dict()

    def method_spec(self) -> ExperimentMethodSpec:
        metadata = dict(self.metadata or {})
        metadata["sampling"] = self.sampling_dict()
        return ExperimentMethodSpec(
            benchmark_ref=self.benchmark_ref,
            method_ref=self.method_ref,
            title=self.title,
            adapter_id=self.adapter_id,
            phases=self.phases,
            report_profiles=self.report_profiles,
            metadata=metadata,
        )

    def _seed(self) -> int | None:
        return SamplingPlan.from_value(self.sampling).seed

    def _split(self, default: str = "") -> str:
        return SamplingPlan.from_value(self.sampling).split or str(default or "")

    def record(
        self,
        result: Any,
        *,
        phase: str = "train",
        metrics: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        state: str = "completed",
        train_docs: int | None = None,
    ) -> ExperimentMethodRun:
        return _record_method_output(
            result,
            output_root=self.output_root,
            method_spec=self.method_spec(),
            phase=phase,
            metrics=metrics,
            artifacts=artifacts,
            metadata=metadata,
            launch_command=self.launch_command,
            state=state,
            split=self._split(),
            seed=self._seed(),
            train_docs=train_docs,
        )

    def train(
        self,
        method: Any,
        *train_args: Any,
        validation_data: Any | None = None,
        config: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        train_docs: int | None = None,
        method_kwargs: Mapping[str, Any] | None = None,
    ) -> ExperimentMethodRun:
        call_kwargs = dict(method_kwargs or {})
        if validation_data is not None:
            call_kwargs.setdefault("validation_data", validation_data)
        return _run_method_phase(
            method,
            "train",
            *train_args,
            output_root=self.output_root,
            context=self,
            method_spec=self.method_spec(),
            config=config,
            metrics=metrics,
            artifacts=artifacts,
            metadata=metadata,
            launch_command=self.launch_command,
            split=self._split(),
            seed=self._seed(),
            train_docs=train_docs,
            method_kwargs=call_kwargs,
        )

    def evaluate(
        self,
        method: Any,
        data: Any,
        *,
        config: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        train_docs: int | None = None,
        method_kwargs: Mapping[str, Any] | None = None,
    ) -> ExperimentMethodRun:
        split = self._split("test")
        return _run_method_phase(
            method,
            "evaluate",
            data,
            output_root=self.output_root,
            context=self,
            method_spec=self.method_spec(),
            config=config,
            metrics=metrics,
            artifacts=artifacts,
            metadata=metadata,
            launch_command=self.launch_command,
            split=split,
            seed=self._seed(),
            train_docs=train_docs,
            method_kwargs=method_kwargs,
        )

    def predict(
        self,
        method: Any,
        inputs: Any,
        *,
        config: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        train_docs: int | None = None,
        method_kwargs: Mapping[str, Any] | None = None,
    ) -> ExperimentMethodRun:
        return _run_method_phase(
            method,
            "predict",
            inputs,
            output_root=self.output_root,
            context=self,
            method_spec=self.method_spec(),
            config=config,
            metrics=metrics,
            artifacts=artifacts,
            metadata=metadata,
            launch_command=self.launch_command,
            split=self._split(),
            seed=self._seed(),
            train_docs=train_docs,
            method_kwargs=method_kwargs,
        )

    def call_metadata(
        self,
        *,
        role: str,
        request_kind: str,
        problem_id: str = "",
        node_id: str = "",
        runner_id: str = "",
        artifacts: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "method_id": str(self.method_ref.method_id),
            "runner_id": str(runner_id or self.method_ref.adapter or self.adapter_id),
            "problem_id": str(problem_id or ""),
            "node_id": str(node_id or ""),
            "request_kind": str(request_kind),
            "role": str(role),
            "sampling": self.sampling_dict(),
            "artifacts": dict(artifacts or {}),
        }


__all__ = ["ExperimentContext", "SamplingPlan"]
