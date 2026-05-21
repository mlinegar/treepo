from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_hash(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except Exception:
        return None


@dataclass(frozen=True)
class PreparedDataRef:
    dataset_id: str = ""
    root: str = ""
    signature: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "PreparedDataRef":
        data = dict(payload or {})
        return cls(
            dataset_id=str(data.get("dataset_id", "") or ""),
            root=str(data.get("root", "") or ""),
            signature=str(data.get("signature", "") or ""),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class ReferenceModelRef:
    reference_id: str = ""
    family: str = ""
    variant: str = ""
    engine: str = ""
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        family = str(payload.pop("family", "") or "")
        if family:
            payload.setdefault("reference_model_id", family)
        return _json_safe(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ReferenceModelRef":
        data = dict(payload or {})
        return cls(
            reference_id=str(data.get("reference_id", "") or ""),
            family=str(data.get("reference_model_id", data.get("family", "")) or ""),
            variant=str(data.get("variant", "") or ""),
            engine=str(data.get("engine", "") or ""),
            model=str(data.get("model", "") or ""),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class SupervisionRef:
    root_rate: float | None = None
    leaf_kind: str = ""
    leaf_rate: float | None = None
    internal_kind: str = ""
    internal_rate: float | None = None
    topology_scope: str = ""
    unit_selector: str = ""
    supervision_kind: str = ""
    label_source: str = ""
    labeler_kind: str = ""
    doc_sample_probability: float | None = None
    unit_sampling_probability: float | None = None
    sampling_strategy: str = ""
    max_units: int | None = None
    coverage_label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "SupervisionRef":
        data = dict(payload or {})
        return cls(
            root_rate=_optional_float(data.get("root_rate")),
            leaf_kind=str(data.get("leaf_kind", "") or ""),
            leaf_rate=_optional_float(data.get("leaf_rate")),
            internal_kind=str(data.get("internal_kind", "") or ""),
            internal_rate=_optional_float(data.get("internal_rate")),
            topology_scope=str(data.get("topology_scope", "") or ""),
            unit_selector=str(data.get("unit_selector", "") or ""),
            supervision_kind=str(data.get("supervision_kind", "") or ""),
            label_source=str(data.get("label_source", "") or ""),
            labeler_kind=str(data.get("labeler_kind", "") or ""),
            doc_sample_probability=_optional_float(data.get("doc_sample_probability")),
            unit_sampling_probability=_optional_float(data.get("unit_sampling_probability")),
            sampling_strategy=str(data.get("sampling_strategy", "") or ""),
            max_units=_optional_int(data.get("max_units")),
            coverage_label=str(data.get("coverage_label", "") or ""),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class ControlRef:
    control_family: str = ""
    law_ids: tuple[str, ...] = ()
    applies_to: str = ""
    enabled: bool = False
    source_kind: str = ""
    sample_budget: int | None = None
    sampling_probability: float | None = None
    sampling_strategy: str = ""
    threshold: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ControlRef":
        data = dict(payload or {})
        return cls(
            control_family=str(data.get("control_family", "") or ""),
            law_ids=tuple(str(item) for item in list(data.get("law_ids") or ())),
            applies_to=str(data.get("applies_to", "") or ""),
            enabled=bool(data.get("enabled", False)),
            source_kind=str(data.get("source_kind", "") or ""),
            sample_budget=_optional_int(data.get("sample_budget")),
            sampling_probability=_optional_float(data.get("sampling_probability")),
            sampling_strategy=str(data.get("sampling_strategy", "") or ""),
            threshold=_optional_float(data.get("threshold")),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class BenchmarkRef:
    benchmark_id: str
    family: str
    scope: str = ""
    cell: str = ""
    dataset_id: str = ""
    name: str = ""
    prepared_data: PreparedDataRef | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        family = str(payload.pop("family", "") or "")
        payload["problem_id"] = family
        if self.prepared_data is not None:
            payload["prepared_data"] = self.prepared_data.to_dict()
        return _json_safe(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkRef":
        data = dict(payload or {})
        prepared_data = data.get("prepared_data")
        return cls(
            benchmark_id=str(data.get("benchmark_id", "") or ""),
            family=str(data.get("problem_id", data.get("family", "")) or ""),
            scope=str(data.get("scope", "") or ""),
            cell=str(data.get("cell", "") or ""),
            dataset_id=str(data.get("dataset_id", "") or ""),
            name=str(data.get("name", "") or ""),
            prepared_data=(
                PreparedDataRef.from_dict(prepared_data)
                if isinstance(prepared_data, Mapping)
                else None
            ),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class MethodRef:
    method_id: str
    family: str
    variant: str = ""
    engine: str = ""
    model: str = ""
    adapter: str = ""
    supervision: SupervisionRef | None = None
    control_ref: ControlRef | None = None
    reference_model: ReferenceModelRef | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.pop("family", None)
        if self.supervision is not None:
            payload["supervision"] = self.supervision.to_dict()
        if self.control_ref is not None:
            payload["control_ref"] = self.control_ref.to_dict()
        if self.reference_model is not None:
            payload["reference_model"] = self.reference_model.to_dict()
        return _json_safe(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MethodRef":
        data = dict(payload or {})
        supervision = data.get("supervision")
        control_ref = data.get("control_ref")
        reference_model = data.get("reference_model")
        return cls(
            method_id=str(data.get("method_id", "") or ""),
            family=str(data.get("family", data.get("method_id", "")) or ""),
            variant=str(data.get("variant", "") or ""),
            engine=str(data.get("engine", "") or ""),
            model=str(data.get("model", "") or ""),
            adapter=str(data.get("adapter", "") or ""),
            supervision=(
                SupervisionRef.from_dict(supervision)
                if isinstance(supervision, Mapping)
                else None
            ),
            control_ref=(
                ControlRef.from_dict(control_ref)
                if isinstance(control_ref, Mapping)
                else None
            ),
            reference_model=(
                ReferenceModelRef.from_dict(reference_model)
                if isinstance(reference_model, Mapping)
                else None
            ),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    artifact_type: str
    path: str
    phase_id: str = ""
    required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactRef":
        data = dict(payload or {})
        return cls(
            artifact_id=str(data.get("artifact_id", "") or ""),
            artifact_type=str(data.get("artifact_type", "") or ""),
            path=str(data.get("path", "") or ""),
            phase_id=str(data.get("phase_id", "") or ""),
            required=bool(data.get("required", True)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    phase_id: str
    task_kind: str
    command: tuple[str, ...] = ()
    deps: tuple[str, ...] = ()
    benchmark_ref: BenchmarkRef | None = None
    method_ref: MethodRef | None = None
    expected_artifacts: tuple[str, ...] = ()
    resources: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.benchmark_ref is not None:
            payload["benchmark_ref"] = self.benchmark_ref.to_dict()
        if self.method_ref is not None:
            payload["method_ref"] = self.method_ref.to_dict()
        return _json_safe(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskSpec":
        data = dict(payload or {})
        benchmark_ref = data.get("benchmark_ref")
        method_ref = data.get("method_ref")
        return cls(
            task_id=str(data.get("task_id", "") or ""),
            phase_id=str(data.get("phase_id", "") or ""),
            task_kind=str(data.get("task_kind", "") or ""),
            command=tuple(str(item) for item in list(data.get("command") or ())),
            deps=tuple(str(item) for item in list(data.get("deps") or ())),
            benchmark_ref=(
                BenchmarkRef.from_dict(benchmark_ref)
                if isinstance(benchmark_ref, Mapping)
                else None
            ),
            method_ref=(
                MethodRef.from_dict(method_ref)
                if isinstance(method_ref, Mapping)
                else None
            ),
            expected_artifacts=tuple(str(item) for item in list(data.get("expected_artifacts") or ())),
            resources=dict(data.get("resources", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class PhaseSpec:
    phase_id: str
    phase_role: str
    task_ids: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    summary_artifacts: tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhaseSpec":
        data = dict(payload or {})
        return cls(
            phase_id=str(data.get("phase_id", "") or ""),
            phase_role=str(data.get("phase_role", "") or ""),
            task_ids=tuple(str(item) for item in list(data.get("task_ids") or ())),
            depends_on=tuple(str(item) for item in list(data.get("depends_on") or ())),
            summary_artifacts=tuple(str(item) for item in list(data.get("summary_artifacts") or ())),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class ProgressSnapshot:
    experiment_id: str
    state: str
    active_phase: str = ""
    items_total: int = 0
    completed_items: int = 0
    failed_items: int = 0
    active_items: int = 0
    pending_items: int = 0
    percent_complete: float = 0.0
    artifact_targets: tuple[str, ...] = ()
    live_child_status_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgressSnapshot":
        data = dict(payload or {})
        return cls(
            experiment_id=str(data.get("experiment_id", "") or ""),
            state=str(data.get("state", "") or ""),
            active_phase=str(data.get("active_phase", "") or ""),
            items_total=int(data.get("items_total", 0) or 0),
            completed_items=int(data.get("completed_items", 0) or 0),
            failed_items=int(data.get("failed_items", 0) or 0),
            active_items=int(data.get("active_items", 0) or 0),
            pending_items=int(data.get("pending_items", 0) or 0),
            percent_complete=float(data.get("percent_complete", 0.0) or 0.0),
            artifact_targets=tuple(str(item) for item in list(data.get("artifact_targets") or ())),
            live_child_status_path=str(data.get("live_child_status_path", "") or ""),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class ResultRow:
    experiment_id: str
    phase: str
    benchmark_ref: BenchmarkRef
    method_ref: MethodRef
    split: str = ""
    seed: int | None = None
    train_docs: int | None = None
    supervision_ref: SupervisionRef | None = None
    control_ref: ControlRef | None = None
    metric_name: str = ""
    metric_value: float | int | str | bool | None = None
    artifact_refs: tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["benchmark_ref"] = self.benchmark_ref.to_dict()
        payload["method_ref"] = self.method_ref.to_dict()
        if self.supervision_ref is not None:
            payload["supervision_ref"] = self.supervision_ref.to_dict()
        if self.control_ref is not None:
            payload["control_ref"] = self.control_ref.to_dict()
        return _json_safe(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResultRow":
        data = dict(payload or {})
        supervision_ref = data.get("supervision_ref")
        control_ref = data.get("control_ref")
        return cls(
            experiment_id=str(data.get("experiment_id", "") or ""),
            phase=str(data.get("phase", "") or ""),
            benchmark_ref=BenchmarkRef.from_dict(dict(data.get("benchmark_ref") or {})),
            method_ref=MethodRef.from_dict(dict(data.get("method_ref") or {})),
            split=str(data.get("split", "") or ""),
            seed=(
                None
                if data.get("seed") in {None, ""}
                else int(data.get("seed"))
            ),
            train_docs=(
                None
                if data.get("train_docs") in {None, ""}
                else int(data.get("train_docs"))
            ),
            supervision_ref=(
                SupervisionRef.from_dict(supervision_ref)
                if isinstance(supervision_ref, Mapping)
                else None
            ),
            control_ref=(
                ControlRef.from_dict(control_ref)
                if isinstance(control_ref, Mapping)
                else None
            ),
            metric_name=str(data.get("metric_name", "") or ""),
            metric_value=data.get("metric_value"),
            artifact_refs=tuple(str(item) for item in list(data.get("artifact_refs") or ())),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    adapter_id: str
    created_utc: str
    output_root: str
    title: str = ""
    benchmark_refs: tuple[BenchmarkRef, ...] = ()
    method_refs: tuple[MethodRef, ...] = ()
    phases: tuple[PhaseSpec, ...] = ()
    tasks: tuple[TaskSpec, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    report_profiles: tuple[str, ...] = ()
    launch_command: tuple[str, ...] = ()
    resume_command: tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "adapter_id": self.adapter_id,
            "created_utc": self.created_utc,
            "output_root": self.output_root,
            "title": self.title,
            "benchmark_refs": [item.to_dict() for item in self.benchmark_refs],
            "method_refs": [item.to_dict() for item in self.method_refs],
            "phases": [item.to_dict() for item in self.phases],
            "tasks": [item.to_dict() for item in self.tasks],
            "artifacts": [item.to_dict() for item in self.artifacts],
            "report_profiles": [str(item) for item in self.report_profiles],
            "launch_command": [str(item) for item in self.launch_command],
            "resume_command": [str(item) for item in self.resume_command],
            "metadata": _json_safe(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentSpec":
        data = dict(payload or {})
        return cls(
            experiment_id=str(data.get("experiment_id", "") or ""),
            adapter_id=str(data.get("adapter_id", "") or ""),
            created_utc=str(data.get("created_utc", "") or ""),
            output_root=str(data.get("output_root", "") or ""),
            title=str(data.get("title", "") or ""),
            benchmark_refs=tuple(
                BenchmarkRef.from_dict(item)
                for item in list(data.get("benchmark_refs") or ())
                if isinstance(item, Mapping)
            ),
            method_refs=tuple(
                MethodRef.from_dict(item)
                for item in list(data.get("method_refs") or ())
                if isinstance(item, Mapping)
            ),
            phases=tuple(
                PhaseSpec.from_dict(item)
                for item in list(data.get("phases") or ())
                if isinstance(item, Mapping)
            ),
            tasks=tuple(
                TaskSpec.from_dict(item)
                for item in list(data.get("tasks") or ())
                if isinstance(item, Mapping)
            ),
            artifacts=tuple(
                ArtifactRef.from_dict(item)
                for item in list(data.get("artifacts") or ())
                if isinstance(item, Mapping)
            ),
            report_profiles=tuple(str(item) for item in list(data.get("report_profiles") or ())),
            launch_command=tuple(str(item) for item in list(data.get("launch_command") or ())),
            resume_command=tuple(str(item) for item in list(data.get("resume_command") or ())),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    @classmethod
    def create(
        cls,
        *,
        adapter_id: str,
        output_root: str,
        title: str = "",
        benchmark_refs: Sequence[BenchmarkRef] = (),
        method_refs: Sequence[MethodRef] = (),
        phases: Sequence[PhaseSpec] = (),
        tasks: Sequence[TaskSpec] = (),
        artifacts: Sequence[ArtifactRef] = (),
        report_profiles: Sequence[str] = (),
        launch_command: Sequence[str] = (),
        resume_command: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "ExperimentSpec":
        payload = {
            "adapter_id": str(adapter_id),
            "output_root": str(output_root),
            "title": str(title),
            "benchmarks": [item.to_dict() for item in benchmark_refs],
            "methods": [item.to_dict() for item in method_refs],
            "phases": [item.to_dict() for item in phases],
            "tasks": [item.to_dict() for item in tasks],
            "report_profiles": [str(item) for item in report_profiles],
            "metadata": dict(metadata or {}),
        }
        experiment_id = stable_hash(payload)[:16]
        return cls(
            experiment_id=experiment_id,
            adapter_id=str(adapter_id),
            created_utc=utc_now_iso(),
            output_root=str(output_root),
            title=str(title),
            benchmark_refs=tuple(benchmark_refs),
            method_refs=tuple(method_refs),
            phases=tuple(phases),
            tasks=tuple(tasks),
            artifacts=tuple(artifacts),
            report_profiles=tuple(str(item) for item in report_profiles),
            launch_command=tuple(str(item) for item in launch_command),
            resume_command=tuple(str(item) for item in resume_command),
            metadata=dict(metadata or {}),
        )


def benchmark_ref_from_parts(
    *,
    family: str,
    problem_id: str = "",
    scope: str = "",
    cell: str = "",
    dataset_id: str = "",
    name: str = "",
    prepared_data: PreparedDataRef | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BenchmarkRef:
    resolved_problem_id = str(problem_id or family)
    benchmark_id = stable_hash(
        {
            "problem_id": resolved_problem_id,
            "scope": str(scope),
            "cell": str(cell),
            "dataset_id": str(dataset_id),
            "name": str(name),
            "prepared_data": prepared_data.to_dict() if prepared_data is not None else {},
        }
    )[:16]
    return BenchmarkRef(
        benchmark_id=benchmark_id,
        family=resolved_problem_id,
        scope=str(scope),
        cell=str(cell),
        dataset_id=str(dataset_id),
        name=str(name),
        prepared_data=prepared_data,
        metadata=dict(metadata or {}),
    )


def method_ref_from_parts(
    *,
    family: str,
    method_id: str = "",
    variant: str = "",
    engine: str = "",
    model: str = "",
    adapter: str = "",
    supervision: SupervisionRef | None = None,
    control_ref: ControlRef | None = None,
    reference_model: ReferenceModelRef | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MethodRef:
    resolved_method_id = str(method_id or family)
    method_key = stable_hash(
        {
            "method_id": resolved_method_id,
            "variant": str(variant),
            "engine": str(engine),
            "model": str(model),
            "adapter": str(adapter),
            "supervision": supervision.to_dict() if supervision is not None else {},
            "control_ref": control_ref.to_dict() if control_ref is not None else {},
            "reference_model": reference_model.to_dict() if reference_model is not None else {},
        }
    )[:16]
    return MethodRef(
        method_id=resolved_method_id or method_key,
        family=str(family),
        variant=str(variant),
        engine=str(engine),
        model=str(model),
        adapter=str(adapter),
        supervision=supervision,
        control_ref=control_ref,
        reference_model=reference_model,
        metadata=dict(metadata or {}),
    )


def default_phase_specs(phase_ids: Iterable[str]) -> tuple[PhaseSpec, ...]:
    phases: list[PhaseSpec] = []
    for raw_phase in phase_ids:
        phase_name = str(raw_phase or "").strip()
        if not phase_name:
            continue
        phases.append(
            PhaseSpec(
                phase_id=phase_name,
                phase_role=phase_name,
            )
        )
    return tuple(phases)
