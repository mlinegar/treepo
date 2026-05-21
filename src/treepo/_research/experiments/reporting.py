from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from treepo._research.experiments.contracts import ArtifactRef, ExperimentSpec, ResultRow
from treepo._research.experiments.control_plane import experiment_paths, load_json
from treepo._research.experiments.normalization import derive_markov_coverage_label


SUPERVISED_PRIMARY_METRIC_NAMES = {"mae", "root_mae", "test_root_mae"}
CORRELATION_METRIC_NAMES = {"pearson_r", "spearman_r"}
DIRECTIONAL_METRIC_NAMES = {"same_side_of_neutral_pct", "within_5pct", "within_10pct"}
CONTROL_OUTCOME_SUFFIXES = ("violation_rate", "failure_rate", "pass_rate")


@dataclass(frozen=True)
class PlotSpec:
    plot_kind: str
    comparison_domain: str
    metric_name: str
    match_policy: str
    reference_policy: str
    facet_policy: str
    main_body: bool
    scope_key: str
    title: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _row_supervision_ref(row: ResultRow) -> Any:
    return row.supervision_ref or row.method_ref.supervision


def _row_control_ref(row: ResultRow) -> Any:
    return row.control_ref or row.method_ref.control_ref


def _safe_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _flatten_scalar_metrics(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
    max_depth: int = 6,
) -> dict[str, int | float | bool]:
    metrics: dict[str, int | float | bool] = {}
    if max_depth <= 0:
        return metrics
    for key, value in dict(payload or {}).items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, bool):
            metrics[name] = bool(value)
        elif isinstance(value, (int, float)):
            metrics[name] = value
        elif isinstance(value, Mapping):
            metrics.update(
                _flatten_scalar_metrics(
                    dict(value),
                    prefix=name,
                    max_depth=max_depth - 1,
                )
            )
    return metrics


def _legacy_ctreepo_result_rows(output_root: str | Path) -> list[ResultRow]:
    paths = experiment_paths(output_root)
    manifest_payload = load_json(paths["manifest"])
    if not manifest_payload:
        return []
    try:
        spec = ExperimentSpec.from_dict(manifest_payload)
    except Exception:
        return []
    if str(spec.adapter_id or "") != "ctreepo_sim":
        return []
    artifact_entries = dict(load_json(paths["artifacts"]).get("artifacts", {}) or {})
    if not artifact_entries:
        artifact_entries = {
            str(artifact.artifact_id): artifact.to_dict()
            for artifact in spec.artifacts
        }
    tasks_by_id = {str(task.task_id): task for task in spec.tasks}
    rows: list[ResultRow] = []
    for artifact_id, raw_artifact in artifact_entries.items():
        payload = dict(raw_artifact or {})
        path_text = str(payload.get("path", "") or "").strip()
        if not path_text or not path_text.endswith(".json"):
            continue
        path = Path(path_text)
        if not path.exists():
            continue
        try:
            summary_payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(summary_payload, Mapping):
            continue
        metadata = dict(payload.get("metadata", {}) or {})
        run_id = str(metadata.get("run_id", "") or "")
        if not run_id and ":" in str(artifact_id):
            run_id = str(artifact_id).split(":", 1)[0]
        task = tasks_by_id.get(run_id)
        if task is None:
            continue
        flattened = _flatten_scalar_metrics(summary_payload)
        for metric_name, metric_value in flattened.items():
            rows.append(
                ResultRow(
                    experiment_id=str(spec.experiment_id),
                    phase=str(task.phase_id),
                    benchmark_ref=task.benchmark_ref,
                    method_ref=task.method_ref,
                    split="",
                    seed=None,
                    train_docs=None,
                    supervision_ref=task.method_ref.supervision,
                    control_ref=task.method_ref.control_ref,
                    metric_name=str(metric_name),
                    metric_value=metric_value,
                    artifact_refs=(str(artifact_id),),
                    metadata={
                        **dict(task.metadata or {}),
                        "legacy_backfill": True,
                        "artifact_id": str(artifact_id),
                        "artifact_path": str(path),
                    },
                )
            )
    return rows


def load_canonical_result_rows(output_root: str | Path) -> list[ResultRow]:
    path = experiment_paths(output_root)["results"]
    rows: list[ResultRow] = []
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            rows.append(ResultRow.from_dict(dict(json.loads(line))))
    if rows:
        return rows
    return _legacy_ctreepo_result_rows(output_root)


def load_canonical_artifacts(output_root: str | Path) -> dict[str, Any]:
    paths = experiment_paths(output_root)
    payload = dict(load_json(paths["artifacts"]).get("artifacts", {}) or {})
    if payload:
        return payload
    manifest_payload = load_json(paths["manifest"])
    if not manifest_payload:
        return {}
    try:
        spec = ExperimentSpec.from_dict(manifest_payload)
    except Exception:
        return {}
    return {
        str(artifact.artifact_id): artifact.to_dict()
        for artifact in spec.artifacts
    }


def supervision_label_for_row(row: ResultRow) -> str:
    ref = _row_supervision_ref(row)
    if ref is None:
        return ""
    if str(ref.coverage_label or "").strip():
        return str(ref.coverage_label)
    return derive_markov_coverage_label(
        root_rate=ref.root_rate,
        leaf_rate=ref.leaf_rate,
        internal_rate=ref.internal_rate,
        package_name=str(ref.metadata.get("package_name", "") or ""),
    )


def control_label_for_row(row: ResultRow) -> str:
    ref = _row_control_ref(row)
    if ref is None or not bool(ref.enabled):
        return ""
    law_suffix = "+".join(
        str(item)
        for item in tuple(ref.law_ids or ())
        if str(item).strip()
    )
    if law_suffix:
        return f"{ref.control_family}:{law_suffix}"
    return str(ref.control_family or "")


def group_rows_by_metric(rows: Sequence[ResultRow]) -> dict[str, list[ResultRow]]:
    grouped: dict[str, list[ResultRow]] = defaultdict(list)
    for row in rows:
        grouped[str(row.metric_name)].append(row)
    return grouped


def artifact_refs_from_payload(payload: Mapping[str, Any]) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    for key, value in dict(payload or {}).items():
        if isinstance(value, Mapping) and str(value.get("path", "") or "").strip():
            refs.append(ArtifactRef.from_dict({"artifact_id": key, **dict(value)}))
    return refs


def comparable_rows(
    rows: Iterable[ResultRow],
    *,
    metric_names: Sequence[str] = (),
) -> list[ResultRow]:
    metric_filter = set(str(item) for item in list(metric_names or ()))
    out: list[ResultRow] = []
    for row in rows:
        if metric_filter and str(row.metric_name) not in metric_filter:
            continue
        out.append(row)
    return out


def _metric_semantics(row: ResultRow) -> str:
    metric_name = str(row.metric_name or "").strip().lower()
    split = str(row.split or "").strip().lower()
    if metric_name == "test_root_mae":
        return "test_mae"
    if metric_name in SUPERVISED_PRIMARY_METRIC_NAMES and split in {"test", ""}:
        return "test_mae"
    if metric_name in CORRELATION_METRIC_NAMES and split in {"test", ""}:
        return "correlation"
    if metric_name in DIRECTIONAL_METRIC_NAMES and split in {"test", ""}:
        return "directional_accuracy"
    if metric_name == "score":
        return "runtime_score"
    if metric_name.endswith(CONTROL_OUTCOME_SUFFIXES):
        return "control_outcome"
    return metric_name


def _benchmark_scope_key(row: ResultRow) -> str:
    parts = [
        str(row.benchmark_ref.family or "").strip(),
        str(row.benchmark_ref.scope or "").strip(),
        str(row.benchmark_ref.cell or "").strip(),
        str(row.benchmark_ref.dataset_id or "").strip(),
    ]
    return "::".join(part for part in parts if part)


def _comparison_domain(row: ResultRow) -> str:
    problem_id = str(row.benchmark_ref.family or "").strip()
    method_id = str(row.method_ref.method_id or row.method_ref.family or "").strip()
    metric_semantics = _metric_semantics(row)
    if problem_id == "runtime_benchmark" or method_id == "runtime_eval":
        return "runtime_context_eval"
    if problem_id == "markov_full_doc" and metric_semantics in {"test_mae", "control_outcome"}:
        return "supervised_root_regression"
    if problem_id == "treepo_task" and metric_semantics in {"test_mae", "control_outcome", "correlation", "directional_accuracy"}:
        return "supervised_doc_regression"
    if problem_id == "ctreepo_sim":
        return "tree_support_recovery"
    return "problem_specific"


def _direct_label_budget(row: ResultRow) -> dict[str, Any]:
    ref = _row_supervision_ref(row)
    train_docs = _safe_int(row.train_docs)
    problem_id = str(row.benchmark_ref.family or "").strip()
    method_id = str(row.method_ref.method_id or row.method_ref.family or "").strip()
    if problem_id == "markov_full_doc":
        rate = _safe_float(getattr(ref, "root_rate", None)) if ref is not None else None
        label = ""
        if rate is not None:
            label = f"R{int(round(100.0 * float(rate)))}"
        elif str(supervision_label_for_row(row)).strip():
            label = str(supervision_label_for_row(row)).split("+", 1)[0]
        count = int(round(float(train_docs) * float(rate))) if train_docs is not None and rate is not None else None
        return {
            "kind": "root_labels",
            "rate": rate,
            "count": count,
            "label": label,
            "maximal": bool(rate is not None and rate >= 0.999999),
        }
    if problem_id == "treepo_task":
        rate = _safe_float(getattr(ref, "doc_sample_probability", None)) if ref is not None else None
        if rate is None and ref is not None and getattr(ref, "topology_scope", "") == "document":
            rate = 1.0
        if rate is None and method_id in {
            "llm_prompt_optimization",
            "embedding_proxy",
            "generator_finetune",
            "ctreepo",
            "mergeable_sketch",
        }:
            rate = 1.0
        count = int(round(float(train_docs) * float(rate))) if train_docs is not None and rate is not None else None
        if ref is not None and str(getattr(ref, "coverage_label", "") or "").strip():
            label = str(getattr(ref, "coverage_label"))
        elif rate is not None:
            label = f"{int(round(100.0 * float(rate)))}% labeled docs"
        else:
            label = ""
        return {
            "kind": "document_labels",
            "rate": rate,
            "count": count,
            "label": label,
            "maximal": bool(rate is not None and rate >= 0.999999),
        }
    return {
        "kind": "",
        "rate": None,
        "count": None,
        "label": "",
        "maximal": False,
    }


def _local_supervision_budget(row: ResultRow) -> dict[str, Any]:
    ref = _row_supervision_ref(row)
    metadata = dict(row.metadata or {})
    label = str(metadata.get("local_supervision_budget_label", "") or "")
    if label:
        return {"label": label}
    if ref is None:
        return {"label": ""}
    if str(row.benchmark_ref.family or "") == "markov_full_doc":
        leaf_rate = _safe_float(getattr(ref, "leaf_rate", None))
        internal_rate = _safe_float(getattr(ref, "internal_rate", None))
        if (leaf_rate or 0.0) <= 0.0 and (internal_rate or 0.0) <= 0.0:
            return {"label": "none"}
        parts = []
        if leaf_rate is not None and leaf_rate > 0.0:
            parts.append(f"Lc{int(round(100.0 * float(leaf_rate)))}")
        if internal_rate is not None and internal_rate > 0.0:
            parts.append(f"Ia{int(round(100.0 * float(internal_rate)))}")
        return {"label": "+".join(parts)}
    if str(getattr(ref, "coverage_label", "") or "").strip() and getattr(ref, "topology_scope", "") == "tree":
        return {"label": str(ref.coverage_label)}
    return {"label": ""}


def _control_budget(row: ResultRow) -> dict[str, Any]:
    ref = _row_control_ref(row)
    label = control_label_for_row(row)
    if ref is None:
        return {"label": ""}
    return {
        "label": label,
        "enabled": bool(ref.enabled),
        "control_family": str(ref.control_family or ""),
        "law_ids": list(ref.law_ids or ()),
    }


def _comparison_group_key(view: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(view.get("comparison_domain", "")),
            str(view.get("benchmark_scope_key", "")),
            str(view.get("split", "")),
            str(view.get("metric_semantics", "")),
            str(view.get("train_docs", "")),
            str(dict(view.get("direct_label_budget") or {}).get("kind", "")),
            str(dict(view.get("direct_label_budget") or {}).get("label", "")),
        ]
    )


def _source_cohort_key(row: ResultRow) -> str:
    metadata = dict(row.metadata or {})
    explicit = str(metadata.get("source_cohort_key", "") or "").strip()
    if explicit:
        return explicit
    output_root = str(metadata.get("source_output_root", "") or metadata.get("output_root", "") or "").strip()
    if output_root:
        return output_root
    return ""


def _method_full_label_reference_key(view: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(view.get("source_cohort_key", "")),
            str(view.get("comparison_domain", "")),
            str(view.get("benchmark_scope_key", "")),
            str(view.get("method_id", "")),
            str(view.get("split", "")),
            str(view.get("metric_semantics", "")),
            str(view.get("train_docs", "")),
        ]
    )


def _observed_frontier_key(view: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(view.get("source_cohort_key", "")),
            str(view.get("comparison_domain", "")),
            str(view.get("benchmark_scope_key", "")),
            str(view.get("split", "")),
            str(view.get("metric_semantics", "")),
            str(view.get("train_docs", "")),
        ]
    )


def derive_comparison_view(row: ResultRow) -> dict[str, Any]:
    direct_label_budget = _direct_label_budget(row)
    view = {
        "experiment_id": str(row.experiment_id or ""),
        "source_cohort_key": _source_cohort_key(row),
        "phase": str(row.phase or ""),
        "method_id": str(row.method_ref.method_id or row.method_ref.family or ""),
        "method_variant": str(row.method_ref.variant or ""),
        "benchmark_scope_key": _benchmark_scope_key(row),
        "problem_id": str(row.benchmark_ref.family or ""),
        "benchmark_scope": str(row.benchmark_ref.scope or row.benchmark_ref.name or ""),
        "split": str(row.split or ""),
        "seed": row.seed,
        "train_docs": _safe_int(row.train_docs),
        "metric_name": str(row.metric_name or ""),
        "metric_semantics": _metric_semantics(row),
        "metric_value": _safe_float(row.metric_value),
        "comparison_domain": _comparison_domain(row),
        "supervision_label": supervision_label_for_row(row),
        "direct_label_budget": direct_label_budget,
        "local_supervision_budget": _local_supervision_budget(row),
        "control_budget": _control_budget(row),
    }
    view["comparison_group_key"] = _comparison_group_key(view)
    view["method_full_label_reference_key"] = _method_full_label_reference_key(view)
    view["observed_frontier_key"] = _observed_frontier_key(view)
    return view


def _best_rows_by_key(
    views: Sequence[Mapping[str, Any]],
    *,
    key_name: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for view in views:
        grouped[str(view.get(key_name, ""))].append(view)
    out: dict[str, dict[str, Any]] = {}
    for key, group in grouped.items():
        valid_group = [
            view
            for view in group
            if _safe_float(view.get("metric_value")) is not None
        ]
        if not valid_group:
            continue
        best = min(valid_group, key=lambda item: float(item.get("metric_value") or 0.0))
        out[str(key)] = {
            "metric_value": float(best.get("metric_value") or 0.0),
            "method_id": str(best.get("method_id", "")),
            "train_docs": best.get("train_docs"),
            "direct_label_budget": dict(best.get("direct_label_budget") or {}),
            "benchmark_scope_key": str(best.get("benchmark_scope_key", "")),
        }
    return out


def _method_full_label_baselines(
    views: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for view in views:
        grouped[str(view.get("method_full_label_reference_key", ""))].append(view)
    out: dict[str, dict[str, Any]] = {}
    for key, group in grouped.items():
        if not group:
            continue
        max_budget_score = max(
            (
                float(dict(view.get("direct_label_budget") or {}).get("rate"))
                if dict(view.get("direct_label_budget") or {}).get("rate") is not None
                else float(dict(view.get("direct_label_budget") or {}).get("count") or 0.0)
            )
            for view in group
        )
        candidate_rows = []
        for view in group:
            budget = dict(view.get("direct_label_budget") or {})
            if budget.get("rate") is not None:
                score = float(budget["rate"])
            else:
                score = float(budget.get("count") or 0.0)
            if abs(score - max_budget_score) < 1e-9:
                candidate_rows.append(view)
        if not candidate_rows:
            continue
        best = min(
            candidate_rows,
            key=lambda item: float(item.get("metric_value") or 0.0),
        )
        out[str(key)] = {
            "metric_value": float(best.get("metric_value") or 0.0),
            "method_id": str(best.get("method_id", "")),
            "direct_label_budget": dict(best.get("direct_label_budget") or {}),
            "benchmark_scope_key": str(best.get("benchmark_scope_key", "")),
            "train_docs": best.get("train_docs"),
        }
    return out


def _plot_caption_contract(spec: PlotSpec) -> Dict[str, str]:
    metadata = dict(spec.metadata or {})
    direct_label_label = str(metadata.get("direct_label_label", "") or "")
    train_docs = _safe_int(metadata.get("train_docs"))
    example = str(metadata.get("example_note", "") or "")
    if not example and direct_label_label and train_docs is not None:
        if direct_label_label.startswith("R"):
            try:
                root_pct = int(direct_label_label[1:].split("+", 1)[0])
                root_count = int(round(float(train_docs) * float(root_pct) / 100.0))
                example = (
                    f"{direct_label_label} @ train_docs={train_docs} means {train_docs} training docs "
                    f"and {root_count} directly root-labeled docs."
                )
            except Exception:
                example = ""
        elif direct_label_label.endswith("% labeled docs"):
            try:
                pct = int(direct_label_label.split("%", 1)[0])
                count = int(round(float(train_docs) * float(pct) / 100.0))
                example = (
                    f"{direct_label_label} @ train_docs={train_docs} means {train_docs} training docs "
                    f"and {count} directly labeled documents."
                )
            except Exception:
                example = ""
    return {
        "match_note": str(
            metadata.get(
                "match_note",
                "Same benchmark, split, train-doc count, and direct document/root label budget.",
            )
        ),
        "exclusion_note": str(
            metadata.get(
                "exclusion_note",
                "Local supervision and verifier/local-law controls are extra structure, not counted as direct labels.",
            )
        ),
        "reference_note": str(
            metadata.get(
                "reference_note",
                "Method baseline means the maximal direct-label run for the same method_id; frontier means the best observed comparable test MAE at this train-doc count.",
            )
        ),
        "example_note": example,
    }


def _selected_plot_specs(views: Sequence[Mapping[str, Any]]) -> tuple[list[PlotSpec], list[PlotSpec]]:
    main_specs: list[PlotSpec] = []
    appendix_specs: list[PlotSpec] = []
    supervised_primary = [
        view
        for view in views
        if str(view.get("comparison_domain", "")) in {"supervised_doc_regression", "supervised_root_regression"}
        and str(view.get("metric_semantics", "")) == "test_mae"
        and _safe_float(view.get("metric_value")) is not None
    ]
    grouped_by_direct_budget: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    grouped_by_train_docs: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    grouped_for_local: dict[tuple[str, str, int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for view in supervised_primary:
        benchmark_scope_key = str(view.get("benchmark_scope_key", ""))
        train_docs = _safe_int(view.get("train_docs"))
        direct_label_label = str(dict(view.get("direct_label_budget") or {}).get("label", ""))
        source_cohort_key = str(view.get("source_cohort_key", ""))
        if direct_label_label:
            grouped_by_direct_budget[
                (str(view.get("comparison_domain", "")), benchmark_scope_key, direct_label_label, source_cohort_key)
            ].append(view)
        if train_docs is not None:
            grouped_by_train_docs[
                (str(view.get("comparison_domain", "")), benchmark_scope_key, train_docs, source_cohort_key)
            ].append(view)
            grouped_for_local[
                (str(view.get("comparison_domain", "")), benchmark_scope_key, train_docs, direct_label_label, source_cohort_key)
            ].append(view)
    for (domain, scope_key, direct_label_label, source_cohort_key), group in sorted(grouped_by_direct_budget.items()):
        train_doc_values = sorted({_safe_int(view.get("train_docs")) for view in group if _safe_int(view.get("train_docs")) is not None})
        if len(train_doc_values) > 1:
            spec = PlotSpec(
                plot_kind="train_doc_scaling",
                comparison_domain=domain,
                metric_name="test_mae",
                match_policy="budget_matched_direct_labels",
                reference_policy="method_full_label_plus_observed_frontier",
                facet_policy="by_direct_label_budget",
                main_body=True,
                scope_key=scope_key,
                title=f"Train-doc scaling | {scope_key} | {direct_label_label}",
                metadata={
                    "direct_label_label": direct_label_label,
                    "train_docs": train_doc_values[-1],
                    "source_cohort_key": source_cohort_key,
                },
            )
            main_specs.append(spec)
    for (domain, scope_key, train_docs, source_cohort_key), group in sorted(grouped_by_train_docs.items()):
        direct_labels = sorted(
            {
                str(dict(view.get("direct_label_budget") or {}).get("label", ""))
                for view in group
                if str(dict(view.get("direct_label_budget") or {}).get("label", "")).strip()
            }
        )
        if len(direct_labels) > 1:
            main_specs.append(
                PlotSpec(
                    plot_kind="direct_label_budget_ladder",
                    comparison_domain=domain,
                    metric_name="test_mae",
                    match_policy="fixed_train_docs",
                    reference_policy="method_full_label_plus_observed_frontier",
                    facet_policy="by_train_docs",
                    main_body=True,
                    scope_key=scope_key,
                    title=f"Direct-label budget ladder | {scope_key} | train_docs={train_docs}",
                    metadata={
                        "train_docs": train_docs,
                        "direct_label_label": direct_labels[-1],
                        "source_cohort_key": source_cohort_key,
                    },
                )
            )
            main_specs.append(
                PlotSpec(
                    plot_kind="gap_to_reference",
                    comparison_domain=domain,
                    metric_name="test_mae",
                    match_policy="fixed_train_docs",
                    reference_policy="method_full_label_plus_observed_frontier",
                    facet_policy="by_train_docs",
                    main_body=True,
                    scope_key=scope_key,
                    title=f"Gap to reference | {scope_key} | train_docs={train_docs}",
                    metadata={"train_docs": train_docs, "source_cohort_key": source_cohort_key},
                )
            )
    for (domain, scope_key, train_docs, direct_label_label, source_cohort_key), group in sorted(grouped_for_local.items()):
        local_labels = sorted(
            {
                str(dict(view.get("local_supervision_budget") or {}).get("label", ""))
                for view in group
                if str(dict(view.get("local_supervision_budget") or {}).get("label", "")).strip()
            }
        )
        control_labels = sorted(
            {
                str(dict(view.get("control_budget") or {}).get("label", ""))
                for view in group
                if str(dict(view.get("control_budget") or {}).get("label", "")).strip()
            }
        )
        if len(local_labels) > 1 or len(control_labels) > 1:
            appendix_specs.append(
                PlotSpec(
                    plot_kind="extra_local_supervision_or_control",
                    comparison_domain=domain,
                    metric_name="test_mae",
                    match_policy="fixed_train_docs_and_direct_labels",
                    reference_policy="method_full_label_plus_observed_frontier",
                    facet_policy="by_local_supervision_or_control",
                    main_body=False,
                    scope_key=scope_key,
                    title=f"Local supervision/control benefit | {scope_key} | train_docs={train_docs}",
                    metadata={
                        "train_docs": train_docs,
                        "direct_label_label": direct_label_label,
                        "source_cohort_key": source_cohort_key,
                    },
                )
            )
    control_rows = [
        view
        for view in views
        if str(view.get("metric_semantics", "")) == "control_outcome"
        and str(view.get("comparison_domain", "")) != "runtime_context_eval"
    ]
    control_scopes = sorted({str(view.get("benchmark_scope_key", "")) for view in control_rows if str(view.get("benchmark_scope_key", "")).strip()})
    for scope_key in control_scopes:
        appendix_specs.append(
            PlotSpec(
                plot_kind="control_outcome",
                comparison_domain="problem_specific",
                metric_name="control_outcome",
                match_policy="within_family_only",
                reference_policy="none",
                facet_policy="by_control_setting",
                main_body=False,
                scope_key=scope_key,
                title=f"Control outcomes | {scope_key}",
                metadata={},
            )
        )
    runtime_rows = [
        view
        for view in views
        if str(view.get("comparison_domain", "")) == "runtime_context_eval"
    ]
    if runtime_rows:
        appendix_specs.append(
            PlotSpec(
                plot_kind="runtime_context_scaling",
                comparison_domain="runtime_context_eval",
                metric_name="runtime_score",
                match_policy="within_runtime_domain",
                reference_policy="observed_frontier_only",
                facet_policy="by_context_length",
                main_body=False,
                scope_key="runtime_context_eval",
                title="Runtime context scaling",
                metadata={},
            )
        )
    return main_specs, appendix_specs


def build_canonical_report_views(rows: Sequence[ResultRow]) -> dict[str, Any]:
    method_ids = sorted(
        {
            str(row.method_ref.method_id or row.method_ref.family or "")
            for row in rows
            if str(row.method_ref.method_id or row.method_ref.family or "").strip()
        }
    )
    metric_names = sorted(
        {
            str(row.metric_name)
            for row in rows
            if str(row.metric_name).strip()
        }
    )
    benchmark_scopes = sorted(
        {
            str(row.benchmark_ref.scope or row.benchmark_ref.name or "")
            for row in rows
            if str(row.benchmark_ref.scope or row.benchmark_ref.name or "").strip()
        }
    )
    supervision_labels = sorted(
        {
            label
            for label in (supervision_label_for_row(row) for row in rows)
            if label
        }
    )
    control_families = sorted(
        {
            str(ref.control_family)
            for ref in (_row_control_ref(row) for row in rows)
            if ref is not None and str(ref.control_family or "").strip()
        }
    )
    control_labels = sorted(
        {
            label
            for label in (control_label_for_row(row) for row in rows)
            if label
        }
    )
    row_views = [derive_comparison_view(row) for row in rows]
    comparison_domains = sorted(
        {
            str(view.get("comparison_domain", ""))
            for view in row_views
            if str(view.get("comparison_domain", "")).strip()
        }
    )
    comparable_metrics: dict[str, Any] = {}
    for metric_name, metric_rows in group_rows_by_metric(rows).items():
        families = sorted(
            {
                str(row.method_ref.method_id or row.method_ref.family or "")
                for row in metric_rows
                if str(row.method_ref.method_id or row.method_ref.family or "").strip()
            }
        )
        if len(families) < 2:
            continue
        comparable_metrics[str(metric_name)] = {
            "method_ids": families,
            "row_count": len(metric_rows),
            "benchmark_scopes": sorted(
                {
                    str(row.benchmark_ref.scope or row.benchmark_ref.name or "")
                    for row in metric_rows
                    if str(row.benchmark_ref.scope or row.benchmark_ref.name or "").strip()
                }
            ),
            "supervision_labels": sorted(
                {
                    label
                    for label in (supervision_label_for_row(row) for row in metric_rows)
                    if label
                }
            ),
            "control_labels": sorted(
                {
                    label
                    for label in (control_label_for_row(row) for row in metric_rows)
                    if label
                }
            ),
        }
    method_sections: dict[str, Any] = {}
    for method_id in method_ids:
        family_rows = [
            row
            for row in rows
            if str(row.method_ref.method_id or row.method_ref.family or "") == method_id
        ]
        method_sections[method_id] = {
            "row_count": len(family_rows),
            "metric_names": sorted(
                {
                    str(row.metric_name)
                    for row in family_rows
                    if str(row.metric_name).strip()
                }
            ),
            "benchmark_scopes": sorted(
                {
                    str(row.benchmark_ref.scope or row.benchmark_ref.name or "")
                    for row in family_rows
                    if str(row.benchmark_ref.scope or row.benchmark_ref.name or "").strip()
                }
            ),
            "supervision_labels": sorted(
                {
                    label
                    for label in (supervision_label_for_row(row) for row in family_rows)
                    if label
                }
            ),
            "control_labels": sorted(
                {
                    label
                    for label in (control_label_for_row(row) for row in family_rows)
                    if label
                }
            ),
        }
    primary_views = [
        view
        for view in row_views
        if str(view.get("metric_semantics", "")) == "test_mae"
        and _safe_float(view.get("metric_value")) is not None
    ]
    method_full_label_baselines = _method_full_label_baselines(primary_views)
    observed_frontiers = _best_rows_by_key(primary_views, key_name="observed_frontier_key")
    main_body_specs, appendix_specs = _selected_plot_specs(row_views)
    return {
        "row_count": len(rows),
        "method_ids": method_ids,
        "metric_names": metric_names,
        "benchmark_scopes": benchmark_scopes,
        "comparison_domains": comparison_domains,
        "supervision_labels": supervision_labels,
        "control_families": control_families,
        "control_labels": control_labels,
        "comparable_metrics": comparable_metrics,
        "method_sections": method_sections,
        "row_views": row_views,
        "reference_summaries": {
            "method_full_label_baselines": method_full_label_baselines,
            "observed_frontiers": observed_frontiers,
        },
        "main_body_plot_specs": [spec.to_dict() for spec in main_body_specs],
        "appendix_plot_specs": [spec.to_dict() for spec in appendix_specs],
        "caption_contracts": {
            spec.title: _plot_caption_contract(spec)
            for spec in (*main_body_specs, *appendix_specs)
        },
    }
