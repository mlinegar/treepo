from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Protocol, Sequence, runtime_checkable

from treepo._research.experiments.contracts import (
    BenchmarkRef,
    ExperimentSpec,
    MethodRef,
    ResultRow,
    method_ref_from_parts,
)
from treepo._research.experiments.roles import method_ref_with_roles
from treepo._research.experiments.sidecars import write_canonical_sidecars


ScalarMetric = float | int | str | bool | None
MethodCallable = Callable[..., Any]

_RESERVED_RESULT_KEYS = {
    "artifacts",
    "artifact_refs",
    "metadata",
    "metrics",
    "model",
    "trained_artifact",
}
_ARTIFACT_FIELD_NAMES = {
    "checkpoint",
    "checkpoint_path",
    "manifest",
    "manifest_path",
    "model_path",
    "output",
    "output_dir",
    "output_file",
    "output_json",
    "output_jsonl",
    "path",
}


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _safe_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _string_path(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value
    return ""


def _merge_metric_value(target: Dict[str, ScalarMetric], key: str, value: Any) -> None:
    if _is_scalar(value):
        target[str(key)] = value


def _flatten_scalar_metrics(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
) -> Dict[str, ScalarMetric]:
    out: Dict[str, ScalarMetric] = {}
    for key, value in dict(payload).items():
        metric_name = f"{prefix}.{key}" if prefix else str(key)
        if _is_scalar(value):
            out[metric_name] = value
        elif isinstance(value, Mapping):
            out.update(_flatten_scalar_metrics(value, prefix=metric_name))
    return out


@dataclass(frozen=True)
class ExperimentMethodSpec:
    """Canonical identity for one train/evaluate/predict method invocation."""

    benchmark_ref: BenchmarkRef
    method_ref: MethodRef
    title: str = ""
    adapter_id: str = "experiment_method"
    phases: tuple[str, ...] = ("train",)
    report_profiles: tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def phase_tuple(self, active_phase: str) -> tuple[str, ...]:
        phases = tuple(str(item) for item in self.phases if str(item))
        if not phases:
            return (str(active_phase),)
        if str(active_phase) in phases:
            return phases
        return (*phases, str(active_phase))


@dataclass(frozen=True)
class NormalizedMethodOutput:
    """Serializable view of a method result.

    The raw trained model or estimator is intentionally kept on the Python
    object returned by the caller, not copied into experiment sidecars.
    """

    raw_result: Any
    metrics: Dict[str, ScalarMetric] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentMethodRun:
    """Return value from a canonical method execution wrapper."""

    raw_result: Any
    normalized: NormalizedMethodOutput
    experiment_spec: ExperimentSpec
    sidecar_root: Path


@runtime_checkable
class ExperimentMethod(Protocol):
    """Canonical identity provider for experiment methods."""

    def method_spec(self) -> ExperimentMethodSpec: ...


@runtime_checkable
class _SupportsTrain(Protocol):
    def train(
        self,
        train_data: Any,
        validation_data: Any | None = None,
        *,
        context: Any,
        config: Mapping[str, Any] | None = None,
    ) -> Any: ...


@runtime_checkable
class _SupportsEvaluate(Protocol):
    def evaluate(
        self,
        data: Any,
        *,
        context: Any,
        split: str = "test",
        config: Mapping[str, Any] | None = None,
    ) -> Any: ...


@runtime_checkable
class _SupportsPredict(Protocol):
    def predict(
        self,
        inputs: Any,
        *,
        context: Any,
        config: Mapping[str, Any] | None = None,
    ) -> Any: ...


@runtime_checkable
class _SupportsArtifacts(Protocol):
    def export_artifacts(self, output_root: str | Path) -> Mapping[str, Any]: ...


@runtime_checkable
class _SupportsSave(Protocol):
    def save(self, path: str | Path) -> Any: ...


@runtime_checkable
class _SupportsLoad(Protocol):
    def load(self, path: str | Path) -> Any: ...

def experiment_method_ref(
    *,
    family: str,
    method_id: str = "",
    variant: str = "",
    engine: str = "",
    model: str = "",
    adapter: str = "",
    roles: Mapping[str, Any] | None = None,
    oracle: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MethodRef:
    """Create a MethodRef with canonical role/oracle metadata attached."""

    method_ref = method_ref_from_parts(
        family=family,
        method_id=method_id or family,
        variant=variant,
        engine=engine,
        model=model,
        adapter=adapter,
        metadata=dict(metadata or {}),
    )
    return method_ref_with_roles(method_ref, roles=roles, oracle=oracle)


def normalize_method_output(
    result: Any,
    *,
    metrics: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> NormalizedMethodOutput:
    """Normalize common train/evaluate/predict return shapes into sidecar fields."""

    normalized_metrics: Dict[str, ScalarMetric] = {}
    normalized_artifacts: Dict[str, Any] = {}
    normalized_metadata: Dict[str, Any] = {}

    if isinstance(result, Mapping):
        payload = dict(result)
        normalized_metrics.update(
            _flatten_scalar_metrics(_safe_mapping(payload.get("metrics")))
        )
        normalized_artifacts.update(_safe_mapping(payload.get("artifacts")))
        normalized_metadata.update(_safe_mapping(payload.get("metadata")))
        for key, value in payload.items():
            key_text = str(key)
            if key_text in _RESERVED_RESULT_KEYS:
                continue
            if key_text in _ARTIFACT_FIELD_NAMES:
                path_text = _string_path(value)
                if path_text:
                    normalized_artifacts.setdefault(key_text, path_text)
                continue
            _merge_metric_value(normalized_metrics, key_text, value)
    elif is_dataclass(result) and not isinstance(result, type):
        for field_info in fields(result):
            key = str(field_info.name)
            if key in _RESERVED_RESULT_KEYS:
                if key == "metadata":
                    normalized_metadata.update(_safe_mapping(getattr(result, key, None)))
                if key == "artifacts":
                    normalized_artifacts.update(_safe_mapping(getattr(result, key, None)))
                if key == "metrics":
                    normalized_metrics.update(
                        _flatten_scalar_metrics(_safe_mapping(getattr(result, key, None)))
                    )
                continue
            value = getattr(result, key)
            if key in _ARTIFACT_FIELD_NAMES:
                path_text = _string_path(value)
                if path_text:
                    normalized_artifacts.setdefault(key, path_text)
                continue
            if isinstance(value, (int, float, bool)) or value is None:
                normalized_metrics.setdefault(key, value)
            elif _is_scalar(value):
                normalized_metadata.setdefault(key, value)
            elif isinstance(value, Mapping):
                normalized_metadata.setdefault(key, dict(value))
            elif isinstance(value, (list, tuple)):
                normalized_metadata.setdefault(key, list(value))
    else:
        normalized_metrics.update(
            _flatten_scalar_metrics(_safe_mapping(getattr(result, "metrics", None)))
        )
        normalized_artifacts.update(_safe_mapping(getattr(result, "artifacts", None)))
        normalized_metadata.update(_safe_mapping(getattr(result, "metadata", None)))

    if metrics:
        normalized_metrics.update(_flatten_scalar_metrics(metrics))
    if artifacts:
        normalized_artifacts.update(dict(artifacts))
    if metadata:
        normalized_metadata.update(dict(metadata))

    return NormalizedMethodOutput(
        raw_result=result,
        metrics=normalized_metrics,
        artifacts=normalized_artifacts,
        metadata=normalized_metadata,
    )


def result_rows_from_method_output(
    output: NormalizedMethodOutput,
    *,
    method_spec: ExperimentMethodSpec,
    phase: str,
    split: str = "",
    seed: int | None = None,
    train_docs: int | None = None,
) -> tuple[ResultRow, ...]:
    artifact_ids = tuple(str(key) for key in output.artifacts.keys())
    rows = []
    for metric_name, metric_value in output.metrics.items():
        rows.append(
            ResultRow(
                experiment_id="",
                phase=str(phase),
                benchmark_ref=method_spec.benchmark_ref,
                method_ref=method_spec.method_ref,
                split=str(split or ""),
                seed=seed,
                train_docs=train_docs,
                metric_name=str(metric_name),
                metric_value=metric_value,
                artifact_refs=artifact_ids,
                metadata=dict(output.metadata),
            )
        )
    return tuple(rows)


def _record_failed_method_phase(
    *,
    output_root: str | Path,
    method_spec: ExperimentMethodSpec,
    phase: str,
    exc: BaseException,
    metadata: Mapping[str, Any] | None = None,
    launch_command: Sequence[str] = (),
) -> None:
    root = Path(output_root).expanduser().resolve()
    failure_metadata = dict(method_spec.metadata)
    if metadata:
        failure_metadata.update(dict(metadata))
    failure_metadata["error"] = {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }
    write_canonical_sidecars(
        root,
        title=method_spec.title or method_spec.method_ref.family,
        adapter_id=method_spec.adapter_id,
        benchmark_refs=(method_spec.benchmark_ref,),
        method_refs=(method_spec.method_ref,),
        phases=method_spec.phase_tuple(phase),
        artifacts={},
        result_rows=(),
        state="failed",
        metadata=failure_metadata,
        launch_command=launch_command,
        report_profiles=method_spec.report_profiles,
    )


def _record_method_output(
    result: Any,
    *,
    output_root: str | Path,
    method_spec: ExperimentMethodSpec,
    phase: str = "train",
    metrics: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    launch_command: Sequence[str] = (),
    state: str = "completed",
    split: str = "",
    seed: int | None = None,
    train_docs: int | None = None,
    replace_results: bool = True,
) -> ExperimentMethodRun:
    """Write canonical experiment sidecars for an already-computed method result."""

    root = Path(output_root).expanduser().resolve()
    normalized = normalize_method_output(
        result,
        metrics=metrics,
        artifacts=artifacts,
        metadata=metadata,
    )
    rows = result_rows_from_method_output(
        normalized,
        method_spec=method_spec,
        phase=phase,
        split=split,
        seed=seed,
        train_docs=train_docs,
    )
    sidecar_metadata = dict(method_spec.metadata)
    sidecar_metadata.update(normalized.metadata)
    spec = write_canonical_sidecars(
        root,
        title=method_spec.title or method_spec.method_ref.family,
        adapter_id=method_spec.adapter_id,
        benchmark_refs=(method_spec.benchmark_ref,),
        method_refs=(method_spec.method_ref,),
        phases=method_spec.phase_tuple(phase),
        artifacts=normalized.artifacts,
        result_rows=rows,
        state=state,
        metadata=sidecar_metadata,
        launch_command=launch_command,
        report_profiles=method_spec.report_profiles,
        replace_results=replace_results,
    )
    return ExperimentMethodRun(
        raw_result=result,
        normalized=normalized,
        experiment_spec=spec,
        sidecar_root=root,
    )


def _resolve_method_spec(
    method: Any,
    *,
    context: Any = None,
    method_spec: ExperimentMethodSpec | None = None,
) -> ExperimentMethodSpec:
    if method_spec is not None:
        return method_spec
    method_method_spec = getattr(method, "method_spec", None)
    if callable(method_method_spec):
        resolved = method_method_spec()
        if isinstance(resolved, ExperimentMethodSpec):
            return resolved
        raise TypeError(
            "method.method_spec() must return ExperimentMethodSpec, "
            f"received {type(resolved).__name__}."
        )
    context_method_spec = getattr(context, "method_spec", None)
    if callable(context_method_spec):
        resolved = context_method_spec()
        if isinstance(resolved, ExperimentMethodSpec):
            return resolved
    raise RuntimeError(
        "Method phase requires an ExperimentMethodSpec, either from "
        "method_spec=..., method.method_spec(), or context.method_spec()."
    )


def _resolve_output_root(output_root: str | Path | None, *, context: Any = None) -> Path:
    if output_root is not None:
        return Path(output_root).expanduser().resolve()
    context_output_root = getattr(context, "output_root", None)
    if context_output_root is not None:
        return Path(context_output_root).expanduser().resolve()
    raise RuntimeError("Method phase requires output_root=... or context.output_root.")


def _resolve_launch_command(
    launch_command: Sequence[str] = (),
    *,
    context: Any = None,
) -> Sequence[str]:
    if launch_command:
        return launch_command
    value = getattr(context, "launch_command", ())
    return tuple(str(item) for item in list(value or ()))


def _callable_accepts_keyword(fn: Callable[..., Any], key: str) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return True
    return str(key) in signature.parameters


def _filter_supported_kwargs(
    fn: Callable[..., Any],
    kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return {str(key): value for key, value in dict(kwargs).items()}
    allowed = set(signature.parameters.keys())
    return {
        str(key): value
        for key, value in dict(kwargs).items()
        if str(key) in allowed
    }


def _looks_like_pytorch_mode_train(fn: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = [
        param
        for param in signature.parameters.values()
        if param.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    ]
    return (
        len(params) == 1
        and params[0].name == "mode"
        and params[0].default is not inspect.Parameter.empty
    )


def _export_method_artifacts(method: Any, output_root: Path) -> Dict[str, Any]:
    export = getattr(method, "export_artifacts", None)
    if not callable(export):
        return {}
    if _callable_accepts_keyword(export, "output_root"):
        payload = export(output_root=output_root)
    else:
        try:
            signature = inspect.signature(export)
        except (TypeError, ValueError):
            payload = export(output_root)
        else:
            positional = [
                param
                for param in signature.parameters.values()
                if param.kind
                in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }
            ]
            payload = export(output_root) if positional else export()
    return _safe_mapping(payload)


def _run_method_phase(
    method: Any,
    phase: str,
    *phase_args: Any,
    output_root: str | Path | None = None,
    context: Any = None,
    method_spec: ExperimentMethodSpec | None = None,
    config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    launch_command: Sequence[str] = (),
    split: str = "",
    seed: int | None = None,
    train_docs: int | None = None,
    method_kwargs: Mapping[str, Any] | None = None,
    replace_results: bool = False,
) -> ExperimentMethodRun:
    """Run one method phase and commit canonical sidecars.

    The wrapper deliberately calls only the named phase. PyTorch-style
    ``module.train()``/``module.eval()`` mode changes belong inside the
    experiment method implementation, not in this generic sidecar layer.
    """

    phase_name = str(phase or "").strip()
    if not phase_name:
        raise ValueError("method phase must be non-empty")
    root = _resolve_output_root(output_root, context=context)
    spec = _resolve_method_spec(
        method,
        context=context,
        method_spec=method_spec,
    )
    launch = _resolve_launch_command(launch_command, context=context)
    fn = getattr(method, phase_name, None)
    if not callable(fn):
        raise RuntimeError(
            f"Experiment method {method.__class__.__name__} does not support phase {phase_name!r}."
        )
    if phase_name == "train" and _looks_like_pytorch_mode_train(fn):
        raise RuntimeError(
            f"{method.__class__.__name__}.train(...) looks like a PyTorch mode toggle. "
            "Wrap the module in an ExperimentMethod whose train(...) runs the training loop."
        )

    call_kwargs: Dict[str, Any] = {}
    if context is not None:
        call_kwargs["context"] = context
    if config is not None:
        call_kwargs["config"] = dict(config)
    if split:
        call_kwargs["split"] = str(split)
    if method_kwargs:
        call_kwargs.update(dict(method_kwargs))
    call_kwargs = _filter_supported_kwargs(fn, call_kwargs)

    try:
        result = fn(*phase_args, **call_kwargs)
        exported_artifacts = _export_method_artifacts(method, root)
    except Exception as exc:
        _record_failed_method_phase(
            output_root=root,
            method_spec=spec,
            phase=phase_name,
            exc=exc,
            metadata=metadata,
            launch_command=launch,
        )
        raise

    combined_artifacts = dict(exported_artifacts)
    if artifacts:
        combined_artifacts.update(dict(artifacts))
    return _record_method_output(
        result,
        output_root=root,
        method_spec=spec,
        phase=phase_name,
        metrics=metrics,
        artifacts=combined_artifacts,
        metadata=metadata,
        launch_command=launch,
        split=split,
        seed=seed,
        train_docs=train_docs,
        replace_results=replace_results,
    )

__all__ = [
    "ExperimentMethod",
    "ExperimentMethodRun",
    "ExperimentMethodSpec",
    "MethodCallable",
    "NormalizedMethodOutput",
    "ScalarMetric",
    "experiment_method_ref",
    "normalize_method_output",
    "result_rows_from_method_output",
]
