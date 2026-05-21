"""Public learning facade for C-TreePO f/g ladders."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from treepo._research.ctreepo.contracts import (
    CTreePOFitResult,
    CTreePOLearningSpec,
    fg_lineage_metadata,
    jsonable,
)
from treepo._research.ctreepo.ladder import (
    DEFAULT_MANIFEST_NAME,
    LadderResult,
    LadderStageContext,
    LadderStageOutput,
    StageTrainFn,
    load_ladder_manifest,
    run_component_ladder,
    write_ladder_manifest,
)
from treepo._research.ctreepo.ladder import continue_ladder as continue_component_ladder


def _ensure_treepo_on_path() -> None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "treepo" / "src"
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            return


def schedule_from_max_iterations(
    max_iterations: int,
    *,
    first_train_side: str = "f",
) -> str:
    """Translate legacy `max_iterations` into an explicit training schedule.

    `max_iterations=0` means baseline evaluation only, so the returned schedule
    is the empty string.  Each subsequent character is one trained component.
    """

    n = int(max_iterations)
    if n < 0:
        raise ValueError(f"max_iterations must be >= 0, got {max_iterations}")
    side = str(first_train_side or "f").strip().lower()
    if side not in {"f", "g"}:
        raise ValueError(f"first_train_side must be 'f' or 'g', got {first_train_side!r}")
    out: list[str] = []
    current = side
    for _ in range(n):
        out.append(current)
        current = "g" if current == "f" else "f"
    return "".join(out)


def preflight(spec: CTreePOLearningSpec) -> Mapping[str, Any]:
    """Run cheap spec-level checks without loading training artifacts."""

    family = str(spec.family or "").strip().lower()
    cfg = dict(spec.backend_config or {})
    axis = dict(spec.axis or {})
    leaf_size_tokens = cfg.get("leaf_size_tokens", axis.get("leaf_size_tokens"))
    if family in {"dspy", "trl"} and leaf_size_tokens is not None:
        from treepo._research.ctreepo.fg_arity import two_child_lm_budget_report

        report = two_child_lm_budget_report(
            family_name=family,
            leaf_size_tokens=int(leaf_size_tokens),
            lm_context_window_tokens=int(
                cfg.get("lm_context_window_tokens", cfg.get("lm_context_tokens", 12000))
            ),
            max_completion_tokens=(
                None
                if cfg.get("max_completion_tokens", cfg.get("max_tokens")) is None
                else int(cfg.get("max_completion_tokens", cfg.get("max_tokens")))
            ),
            prompt_template_overhead_tokens=int(
                cfg.get("prompt_template_overhead_tokens", cfg.get("prompt_overhead_tokens", 1500))
            ),
        )
        if not report.ok:
            raise RuntimeError("; ".join(report.violations))
        return {"family": family, "budget_report": asdict(report)}
    return {"family": family, "ok": True}


def _coerce_spec(spec: CTreePOLearningSpec | Mapping[str, Any]) -> CTreePOLearningSpec:
    if isinstance(spec, CTreePOLearningSpec):
        return spec
    if isinstance(spec, Mapping):
        return CTreePOLearningSpec.from_mapping(spec)
    raise TypeError(f"expected CTreePOLearningSpec or mapping, got {type(spec).__name__}")


def _fit_result_from_ladder(result: LadderResult) -> CTreePOFitResult:
    return CTreePOFitResult(
        status=result.status,
        metrics={"stage_count": float(len(result.stages))},
        artifacts=jsonable(dict(result.component_artifacts or {})),
        history=[stage.to_dict() for stage in result.stages],
        summary={
            "schedule": list(result.schedule),
            "shared_artifacts": jsonable(dict(result.shared_artifacts or {})),
            "manifest_path": str(result.manifest_path),
        },
        manifest_path=str(result.manifest_path),
    )


def train_ladder(
    spec: CTreePOLearningSpec | Mapping[str, Any],
    *,
    output_dir: str | Path,
    train_stage: StageTrainFn | None = None,
) -> CTreePOFitResult:
    """Train a C-TreePO ladder from a public learning spec."""

    learning_spec = _coerce_spec(spec)
    cfg = dict(learning_spec.backend_config or {})
    family_runtime = cfg.get("family_runtime")
    if family_runtime is not None:
        return run_family_runtime_ladder(
            family=family_runtime,
            f_init=(learning_spec.initial_artifacts or {}).get("f", cfg.get("f_init")),
            g_init=(learning_spec.initial_artifacts or {}).get("g", cfg.get("g_init")),
            initial_shared_artifacts=dict(cfg.get("initial_shared_artifacts") or {}),
            traces=learning_spec.train_data or (),
            eval_trees=learning_spec.eval_data or learning_spec.train_data or (),
            schedule=str(learning_spec.schedule or ""),
            output_dir=output_dir,
            axis_kind=str((learning_spec.axis or {}).get("axis_kind", "leaf_count")),
            axis_value=int((learning_spec.axis or {}).get("axis_value", 0) or 0),
            leaf_count=(learning_spec.axis or {}).get("leaf_count"),
            leaf_size_tokens=(learning_spec.axis or {}).get("leaf_size_tokens"),
            first_train_side=str(cfg.get("first_train_side", "f")),
            initial_f_degree=int(cfg.get("initial_f_degree", 1)),
            initial_g_degree=int(cfg.get("initial_g_degree", 1)),
            stage_naming=str(cfg.get("stage_naming", "legacy")),
            previous_manifest=cfg.get("previous_manifest"),
            schedule_prefix=str(cfg.get("schedule_prefix", "")),
            metadata={
                "spec": learning_spec.to_dict(),
                "preflight": dict(preflight(learning_spec)),
            },
        )

    stage_fn = train_stage or cfg.get("train_stage")
    if not callable(stage_fn):
        raise ValueError(
            "train_ladder requires either backend_config['family_runtime'] or a "
            "callable train_stage"
        )

    result = run_component_ladder(
        schedule=str(learning_spec.schedule),
        output_dir=output_dir,
        train_stage=stage_fn,
        initial_component_artifacts=dict(learning_spec.initial_artifacts or {}),
        initial_shared_artifacts=dict(cfg.get("initial_shared_artifacts") or {}),
        allowed_components=frozenset(str(c) for c in cfg.get("allowed_components", ("f", "g"))),
        metadata={"spec": learning_spec.to_dict()},
    )
    return _fit_result_from_ladder(result)


def continue_ladder(
    previous_manifest: str | Path,
    *,
    schedule: str,
    output_dir: str | Path,
    spec: CTreePOLearningSpec | Mapping[str, Any] | None = None,
    train_stage: StageTrainFn | None = None,
) -> CTreePOFitResult:
    """Continue a ladder from a previous manifest.

    When a `spec` with `backend_config['family_runtime']` is supplied, the
    current f/g artifacts are loaded from the previous manifest and dispatched
    through the family runtime.  Otherwise this delegates to the generic ladder
    runner with the supplied `train_stage`.
    """

    manifest = load_ladder_manifest(previous_manifest)
    latest_components = dict(manifest.get("component_artifacts") or {})
    latest_shared = dict(manifest.get("shared_artifacts") or {})

    if spec is not None:
        learning_spec = _coerce_spec(spec).with_schedule(schedule).with_initial_artifacts(
            latest_components
        )
        cfg = dict(learning_spec.backend_config or {})
        cfg.setdefault("initial_shared_artifacts", latest_shared)
        cfg.setdefault("previous_manifest", str(previous_manifest))
        prior_schedule = manifest.get("metadata", {}).get("combined_schedule")
        if prior_schedule is None:
            prior_schedule = "".join(str(ch) for ch in manifest.get("schedule", []) or [])
        cfg.setdefault("schedule_prefix", str(prior_schedule or ""))
        learning_spec = CTreePOLearningSpec(
            space_kind=learning_spec.space_kind,
            family=learning_spec.family,
            schedule=learning_spec.schedule,
            initial_artifacts=learning_spec.initial_artifacts,
            train_data=learning_spec.train_data,
            eval_data=learning_spec.eval_data,
            backend_config=cfg,
            axis=learning_spec.axis,
        )
        return train_ladder(learning_spec, output_dir=output_dir, train_stage=train_stage)

    stage_fn = train_stage
    if not callable(stage_fn):
        raise ValueError("continue_ladder without a spec requires a callable train_stage")
    result = continue_component_ladder(
        previous_manifest=previous_manifest,
        schedule=schedule,
        output_dir=output_dir,
        train_stage=stage_fn,
        allowed_components=frozenset({"f", "g"}),
    )
    return _fit_result_from_ladder(result)


def fit(
    spec: CTreePOLearningSpec | Mapping[str, Any],
    *,
    output_dir: str | Path,
    train_stage: StageTrainFn | None = None,
) -> CTreePOFitResult:
    """Alias for `train_ladder` to match the broader one-call API shape."""

    return train_ladder(spec, output_dir=output_dir, train_stage=train_stage)


def _stage_name_from_prefix(
    prefix: str,
    *,
    first_train_side: str,
    initial_f_degree: int,
    initial_g_degree: int,
    stage_naming: str,
) -> tuple[str, str, int, int]:
    from treepo._research.ctreepo.alternating import stage_label_for_iteration, stage_name_for_iteration

    f_degree = int(initial_f_degree) + prefix.count("f")
    g_degree = int(initial_g_degree) + prefix.count("g")
    canonical = schedule_from_max_iterations(
        len(prefix),
        first_train_side=first_train_side,
    )
    if prefix == canonical:
        stage_name = stage_name_for_iteration(
            len(prefix),
            first_train_side=first_train_side,
            initial_f_degree=initial_f_degree,
            initial_g_degree=initial_g_degree,
            naming=stage_naming,
        )
        stage_label = stage_label_for_iteration(
            len(prefix),
            first_train_side=first_train_side,
            initial_f_degree=initial_f_degree,
            initial_g_degree=initial_g_degree,
        )
    else:
        stage_name = f"f{f_degree}g{g_degree}"
        stage_label = f"f^{f_degree} g^{g_degree}"
    return stage_name, stage_label, f_degree, g_degree


def _evaluate_family_iteration(
    *,
    family: Any,
    f: Any,
    g: Any,
    trees: Sequence[Any],
    prediction_records_path: Path | None,
) -> tuple[Mapping[str, Any], Optional[str]]:
    from treepo._research.ctreepo.alternating import evaluate_iteration

    try:
        metrics = evaluate_iteration(
            family=family,
            f=f,
            g=g,
            trees=list(trees),
            prediction_records_path=prediction_records_path,
        )
        return metrics, None
    except NotImplementedError as exc:
        return {}, f"evaluation NotImplementedError: {exc}"
    except Exception as exc:
        return {}, f"evaluation {type(exc).__name__}: {exc}"


def _iteration_record_dict(
    *,
    iteration: int,
    prefix: str,
    family: Any,
    trained: str,
    f_artifact: Any,
    g_artifact: Any,
    split_metrics: Mapping[str, Any],
    error: str | None,
    axis_kind: str,
    axis_value: int,
    leaf_count: int | None,
    leaf_size_tokens: int | None,
    first_train_side: str,
    initial_f_degree: int,
    initial_g_degree: int,
    stage_naming: str,
    trace_artifacts: Mapping[str, Any] | None = None,
    trace_metrics: Mapping[str, Any] | None = None,
    trace_errors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from treepo._research.ctreepo.alternating import IterationRecord

    stage_name, stage_label, f_degree, g_degree = _stage_name_from_prefix(
        prefix,
        first_train_side=first_train_side,
        initial_f_degree=initial_f_degree,
        initial_g_degree=initial_g_degree,
        stage_naming=stage_naming,
    )
    extra = {"error": error} if error else {}
    if trace_artifacts:
        extra["trace_artifacts"] = jsonable(dict(trace_artifacts))
    if trace_metrics:
        extra["trace_metrics"] = jsonable(dict(trace_metrics))
    if trace_errors:
        extra["trace_errors"] = jsonable(dict(trace_errors))
    record = IterationRecord(
        iteration=int(iteration),
        stage_name=stage_name,
        stage_label=stage_label,
        family=str(getattr(family, "name", "family")),
        f_degree=int(f_degree),
        g_degree=int(g_degree),
        axis_kind=str(axis_kind),
        axis_value=int(axis_value),
        leaf_count=int(leaf_count) if leaf_count is not None else None,
        leaf_size_tokens=int(leaf_size_tokens) if leaf_size_tokens is not None else None,
        trained=str(trained),
        f_artifact=None if f_artifact is None else str(f_artifact),
        g_artifact=None if g_artifact is None else str(g_artifact),
        split_metrics=dict(split_metrics or {}),
        extra=extra,
    )
    return asdict(record)


def _write_family_step_checkpoint(
    *,
    output_dir: Path,
    family: Any,
    axis_kind: str,
    axis_value: int,
    leaf_count: int | None,
    leaf_size_tokens: int | None,
    iteration: int,
    stage_name: str,
    stage_label: str | None,
    f_degree: int | None,
    g_degree: int | None,
    trained: str,
    phase: str,
    f_artifact: Any,
    g_artifact: Any,
    iteration_dir: Path | None = None,
    split_metrics: Mapping[str, Any] | None = None,
    error: str | None = None,
    artifact_validation: Mapping[str, Any] | None = None,
    trace_artifacts: Mapping[str, Any] | None = None,
    trace_metrics: Mapping[str, Any] | None = None,
    trace_errors: Mapping[str, Any] | None = None,
) -> Path:
    checkpoints_dir = Path(output_dir) / "step_checkpoints"
    checkpoint_path = checkpoints_dir / f"iter_{int(iteration):02d}_{phase}.json"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "family": str(getattr(family, "name", "family")),
        "axis_kind": str(axis_kind),
        "axis_value": int(axis_value),
        "leaf_count": int(leaf_count) if leaf_count is not None else None,
        "leaf_size_tokens": int(leaf_size_tokens) if leaf_size_tokens is not None else None,
        "iteration": int(iteration),
        "stage_name": str(stage_name),
        "stage_label": str(stage_label) if stage_label is not None else None,
        "f_degree": int(f_degree) if f_degree is not None else None,
        "g_degree": int(g_degree) if g_degree is not None else None,
        "trained": str(trained),
        "phase": str(phase),
        "f_artifact": None if f_artifact is None else str(f_artifact),
        "g_artifact": None if g_artifact is None else str(g_artifact),
        "iteration_dir": str(iteration_dir) if iteration_dir is not None else None,
        "error": error,
        "artifact_validation": jsonable(dict(artifact_validation or {})),
        "checkpoint_path": str(checkpoint_path),
    }
    if trace_artifacts:
        payload["trace_artifacts"] = jsonable(dict(trace_artifacts))
    if trace_metrics:
        payload["trace_metrics"] = jsonable(dict(trace_metrics))
    if trace_errors:
        payload["trace_errors"] = jsonable(dict(trace_errors))
    if split_metrics is not None:
        payload["split_metrics"] = jsonable(dict(split_metrics or {}))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
    tmp_path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(checkpoint_path)
    latest_path = checkpoints_dir / "latest.json"
    latest_tmp = latest_path.with_name(f"{latest_path.name}.tmp")
    latest_tmp.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    latest_tmp.replace(latest_path)
    return checkpoint_path


def _artifact_validation_from_stage_result(result: Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        validation = result.get("artifact_validation")
        if isinstance(validation, Mapping):
            return dict(validation)
    return {}


def run_family_runtime_ladder(
    *,
    family: Any,
    f_init: Any,
    g_init: Any,
    traces: Sequence[Any],
    eval_trees: Sequence[Any],
    schedule: str,
    output_dir: str | Path,
    axis_kind: str = "leaf_count",
    axis_value: int = 0,
    leaf_count: int | None = None,
    leaf_size_tokens: int | None = None,
    first_train_side: str = "f",
    initial_f_degree: int = 1,
    initial_g_degree: int = 1,
    stage_naming: str = "legacy",
    initial_shared_artifacts: Mapping[str, Any] | None = None,
    previous_manifest: str | Path | None = None,
    schedule_prefix: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> CTreePOFitResult:
    """Run an existing `FamilyRuntime` through the new ladder manifest API."""

    from treepo._research.ctreepo.alternating import (
        _validate_family_artifact,
        export_ladder_full_tree_traces,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    schedule = str(schedule or "")
    if any(ch not in {"f", "g"} for ch in schedule):
        raise ValueError(f"family runtime schedule must contain only f/g, got {schedule!r}")

    initial_components = {"f": f_init, "g": g_init}
    family_name = str(getattr(family, "name", "family"))
    combined_schedule = f"{str(schedule_prefix or '')}{schedule}"
    family_metadata = {
        "family": family_name,
        "family_runtime": type(family).__name__,
        "schedule_prefix": str(schedule_prefix or ""),
        "combined_schedule": combined_schedule,
        "f_init": None if f_init is None else str(f_init),
        "g_init": None if g_init is None else str(g_init),
        "fg_lineage": fg_lineage_metadata(
            f_init="" if f_init is None else str(f_init),
            g_init="" if g_init is None else str(g_init),
            schedule=combined_schedule,
            f_lineage={"init_artifact": None if f_init is None else str(f_init)},
            g_lineage={"init_artifact": None if g_init is None else str(g_init)},
        ),
        "trl_train_g_uses_current_f_reward": False if family_name == "trl" else None,
        **dict(metadata or {}),
    }

    def _train_stage(context: LadderStageContext) -> LadderStageOutput:
        component = str(context.component)
        iteration = int(context.index) + 1
        if component == "f":
            artifact = family.train_f(
                f_init=context.component_artifacts.get("f"),
                g=context.component_artifacts.get("g"),
                traces=traces,
                output_dir=context.stage_dir,
                iteration=iteration,
            )
        elif component == "g":
            artifact = family.train_g(
                g_init=context.component_artifacts.get("g"),
                f=context.component_artifacts.get("f"),
                traces=traces,
                output_dir=context.stage_dir,
                iteration=iteration,
            )
        else:
            raise ValueError(f"unsupported family ladder component: {component!r}")
        validation = _validate_family_artifact(family, kind=component, artifact=artifact)
        return LadderStageOutput(
            component_artifact=artifact,
            result={"artifact_validation": {component: validation}},
            metrics={},
        )

    if schedule:
        ladder_result = run_component_ladder(
            schedule=schedule,
            output_dir=output_root,
            train_stage=_train_stage,
            initial_component_artifacts=initial_components,
            initial_shared_artifacts=dict(initial_shared_artifacts or {}),
            allowed_components=frozenset({"f", "g"}),
            stage_dir_name=lambda idx, comp: f"iter_{int(idx) + 1:02d}_train_{comp}",
            manifest_name=DEFAULT_MANIFEST_NAME,
            previous_manifest=previous_manifest,
            metadata=family_metadata,
        )
        stages = list(ladder_result.stages)
        final_artifacts = dict(ladder_result.component_artifacts or {})
        manifest_path = ladder_result.manifest_path
    else:
        manifest_path = output_root / DEFAULT_MANIFEST_NAME
        stages = []
        final_artifacts = dict(initial_components)
        write_ladder_manifest(
            manifest_path,
            output_dir=output_root,
            schedule=(),
            component_artifacts=final_artifacts,
            shared_artifacts={},
            stages=(),
            status="completed",
            previous_manifest=previous_manifest,
            metadata=family_metadata,
        )

    history: list[dict[str, Any]] = []
    prediction_dir = output_root / "prediction_records"
    all_trace_artifacts: dict[str, Any] = {}
    eval_items = list(eval_trees or traces or ())

    current_f = f_init
    current_g = g_init
    stage_name, stage_label, f_degree, g_degree = _stage_name_from_prefix(
        "",
        first_train_side=first_train_side,
        initial_f_degree=initial_f_degree,
        initial_g_degree=initial_g_degree,
        stage_naming=stage_naming,
    )
    _write_family_step_checkpoint(
        output_dir=output_root,
        family=family,
        axis_kind=axis_kind,
        axis_value=axis_value,
        leaf_count=leaf_count,
        leaf_size_tokens=leaf_size_tokens,
        iteration=0,
        stage_name=stage_name,
        stage_label=stage_label,
        f_degree=f_degree,
        g_degree=g_degree,
        trained="none",
        phase="post_train",
        f_artifact=current_f,
        g_artifact=current_g,
    )
    metrics, error = _evaluate_family_iteration(
        family=family,
        f=current_f,
        g=current_g,
        trees=eval_items,
        prediction_records_path=prediction_dir / "iter_00_post_eval.jsonl",
    )
    trace_export = export_ladder_full_tree_traces(
        family=family,
        f=current_f,
        g=current_g,
        tree_sets={"train": list(traces), "eval": eval_items},
        output_dir=output_root,
        iteration=0,
    )
    all_trace_artifacts.update(dict(trace_export.get("artifacts") or {}))
    _write_family_step_checkpoint(
        output_dir=output_root,
        family=family,
        axis_kind=axis_kind,
        axis_value=axis_value,
        leaf_count=leaf_count,
        leaf_size_tokens=leaf_size_tokens,
        iteration=0,
        stage_name=stage_name,
        stage_label=stage_label,
        f_degree=f_degree,
        g_degree=g_degree,
        trained="none",
        phase="post_eval",
        f_artifact=current_f,
        g_artifact=current_g,
        split_metrics=metrics,
        error=error,
        trace_artifacts=trace_export.get("artifacts"),
        trace_metrics=trace_export.get("metrics"),
        trace_errors=trace_export.get("errors"),
    )
    history.append(
        _iteration_record_dict(
            iteration=0,
            prefix="",
            family=family,
            trained="none",
            f_artifact=current_f,
            g_artifact=current_g,
            split_metrics=metrics,
            error=error,
            axis_kind=axis_kind,
            axis_value=axis_value,
            leaf_count=leaf_count,
            leaf_size_tokens=leaf_size_tokens,
            first_train_side=first_train_side,
            initial_f_degree=initial_f_degree,
            initial_g_degree=initial_g_degree,
            stage_naming=stage_naming,
            trace_artifacts=trace_export.get("artifacts"),
            trace_metrics=trace_export.get("metrics"),
            trace_errors=trace_export.get("errors"),
        )
    )

    prefix = ""
    for stage in stages:
        prefix += str(stage.component)
        components_after = dict(stage.input_component_artifacts or {})
        components_after[str(stage.component)] = stage.output_component_artifact
        current_f = components_after.get("f")
        current_g = components_after.get("g")
        stage_name, stage_label, f_degree, g_degree = _stage_name_from_prefix(
            prefix,
            first_train_side=first_train_side,
            initial_f_degree=initial_f_degree,
            initial_g_degree=initial_g_degree,
            stage_naming=stage_naming,
        )
        artifact_validation = _artifact_validation_from_stage_result(stage.result)
        _write_family_step_checkpoint(
            output_dir=output_root,
            family=family,
            axis_kind=axis_kind,
            axis_value=axis_value,
            leaf_count=leaf_count,
            leaf_size_tokens=leaf_size_tokens,
            iteration=int(stage.index) + 1,
            stage_name=stage_name,
            stage_label=stage_label,
            f_degree=f_degree,
            g_degree=g_degree,
            trained=str(stage.component),
            phase="post_train",
            f_artifact=current_f,
            g_artifact=current_g,
            iteration_dir=stage.stage_dir,
            artifact_validation=artifact_validation,
        )
        metrics, error = _evaluate_family_iteration(
            family=family,
            f=current_f,
            g=current_g,
            trees=eval_items,
            prediction_records_path=(
                prediction_dir / f"iter_{int(stage.index) + 1:02d}_post_eval.jsonl"
            ),
        )
        trace_export = export_ladder_full_tree_traces(
            family=family,
            f=current_f,
            g=current_g,
            tree_sets={"train": list(traces), "eval": eval_items},
            output_dir=output_root,
            iteration=int(stage.index) + 1,
        )
        all_trace_artifacts.update(dict(trace_export.get("artifacts") or {}))
        _write_family_step_checkpoint(
            output_dir=output_root,
            family=family,
            axis_kind=axis_kind,
            axis_value=axis_value,
            leaf_count=leaf_count,
            leaf_size_tokens=leaf_size_tokens,
            iteration=int(stage.index) + 1,
            stage_name=stage_name,
            stage_label=stage_label,
            f_degree=f_degree,
            g_degree=g_degree,
            trained=str(stage.component),
            phase="post_eval",
            f_artifact=current_f,
            g_artifact=current_g,
            iteration_dir=stage.stage_dir,
            split_metrics=metrics,
            error=error,
            artifact_validation=artifact_validation,
            trace_artifacts=trace_export.get("artifacts"),
            trace_metrics=trace_export.get("metrics"),
            trace_errors=trace_export.get("errors"),
        )
        history.append(
            _iteration_record_dict(
                iteration=int(stage.index) + 1,
                prefix=prefix,
                family=family,
                trained=str(stage.component),
                f_artifact=current_f,
                g_artifact=current_g,
                split_metrics=metrics,
                error=error,
                axis_kind=axis_kind,
                axis_value=axis_value,
                leaf_count=leaf_count,
                leaf_size_tokens=leaf_size_tokens,
                first_train_side=first_train_side,
                initial_f_degree=initial_f_degree,
                initial_g_degree=initial_g_degree,
                stage_naming=stage_naming,
                trace_artifacts=trace_export.get("artifacts"),
                trace_metrics=trace_export.get("metrics"),
                trace_errors=trace_export.get("errors"),
            )
        )

    result_artifacts = dict(final_artifacts)
    result_artifacts.update(all_trace_artifacts)
    if all_trace_artifacts:
        result_artifacts["full_tree_traces_dir"] = str(output_root / "full_tree_traces")
        write_ladder_manifest(
            manifest_path,
            output_dir=output_root,
            schedule=tuple(str(ch) for ch in schedule),
            component_artifacts=final_artifacts,
            shared_artifacts={
                **dict(initial_shared_artifacts or {}),
                **dict(all_trace_artifacts),
                "full_tree_traces_dir": str(output_root / "full_tree_traces"),
            },
            stages=tuple(stages),
            status="completed",
            previous_manifest=previous_manifest,
            metadata=family_metadata,
        )

    return CTreePOFitResult(
        status="completed",
        metrics={"stage_count": float(len(stages)), "iteration_count": float(len(history))},
        artifacts=jsonable(result_artifacts),
        history=history,
        summary={
            "family": family_name,
            "schedule": list(schedule),
            "schedule_prefix": str(schedule_prefix or ""),
            "combined_schedule": combined_schedule,
            "f_init": None if f_init is None else str(f_init),
            "g_init": None if g_init is None else str(g_init),
            "manifest_path": str(manifest_path),
            "previous_manifest": str(previous_manifest) if previous_manifest else None,
            "trl_train_g_uses_current_f_reward": (
                False if family_name == "trl" else None
            ),
        },
        manifest_path=str(manifest_path),
    )


def fit_classical_sketch_grid(config: Any, *, output_dir: str | Path) -> Any:
    """Run the broad classical-sketch grid through the canonical facade."""

    _ensure_treepo_on_path()
    from treepo.bench.classical_sketches import ClassicalSketchComparisonConfig

    cfg = (
        config
        if isinstance(config, ClassicalSketchComparisonConfig)
        else ClassicalSketchComparisonConfig(**dict(config or {}))
    )
    try:
        from unified_g_v1.sketch.classical_sketch_grid import classical_sketch_grid_task
        from unified_g_v1.training.fit import fit as unified_fit
    except ImportError:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "parallel" / "unified_g_v1" / "src"
            if candidate.exists():
                import sys

                for path in (str(parent), str(candidate)):
                    if path not in sys.path:
                        sys.path.insert(0, path)
                break
        from unified_g_v1.sketch.classical_sketch_grid import classical_sketch_grid_task
        from unified_g_v1.training.fit import fit as unified_fit

    return unified_fit(
        trainer_config=classical_sketch_grid_task(config=cfg),
        output_dir=output_dir,
    )


__all__ = [
    "continue_ladder",
    "fit",
    "fit_classical_sketch_grid",
    "preflight",
    "run_family_runtime_ladder",
    "schedule_from_max_iterations",
    "train_ladder",
]
