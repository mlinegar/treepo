from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import fmean
import textwrap
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from treepo._research.ctreepo.sim.util import safe_float
from treepo._research.ctreepo.sim.local_law_backfill import (
    collect_law_stress_assessments,
    load_or_backfill_local_law_payload,
)
from treepo._research.ctreepo.sim.local_law_learnability import (
    LocalLawRunSummary,
    PolicyRole,
    selected_policy_role,
    split_metric_views,
)


@dataclass(frozen=True)
class LoadedLocalLawRun:
    path: str
    summary: LocalLawRunSummary
    payload: Dict[str, Any]


_safe_float = safe_float


def _safe_mean(values: Iterable[object]) -> float:
    xs = [float(v) for v in (_safe_float(x) for x in values) if math.isfinite(float(v))]
    if not xs:
        return float("nan")
    return float(fmean(xs))


def _safe_percentile(values: Iterable[object], q: float) -> float:
    xs = np.asarray(
        [float(v) for v in (_safe_float(x) for x in values) if math.isfinite(float(v))],
        dtype=np.float64,
    )
    if xs.size == 0:
        return float("nan")
    return float(np.percentile(xs, float(q)))


def _role_value(role: object) -> str:
    value = getattr(role, "value", role)
    return str(value)


def _format_range(values: Iterable[object]) -> str:
    xs = sorted({int(v) for v in values if math.isfinite(_safe_float(v))})
    if not xs:
        return "-"
    if len(xs) == 1:
        return str(xs[0])
    return f"{xs[0]}..{xs[-1]}"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _expected_main_package(family: str, packages: Sequence[str]) -> Optional[str]:
    available = {str(x).strip() for x in packages if str(x).strip()}
    if not available:
        return None
    family = str(family).strip()
    if family == "markov_ops_count":
        if "all_laws_plus_sched" in available:
            return "all_laws_plus_sched"
        if "all_laws" in available:
            return "all_laws"
    if family == "tree_relevant_lda_local_law":
        if "all_laws" in available:
            return "all_laws"
    for candidate in ("all_laws_plus_sched", "all_laws"):
        if candidate in available:
            return candidate
    return None


def load_local_law_runs(input_root: Path) -> List[LoadedLocalLawRun]:
    runs: List[LoadedLocalLawRun] = []
    for path in sorted(input_root.rglob("*.json")):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        loaded = load_or_backfill_local_law_payload(payload, source_path=str(path))
        if loaded is None:
            continue
        summary, augmented = loaded
        runs.append(LoadedLocalLawRun(path=str(path), summary=summary, payload=augmented))
    return runs


def _policy_metrics(
    summary: LocalLawRunSummary,
    *,
    split: str,
    role: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    for policy in dict(summary.policies).values():
        if _role_value(policy.role) != str(role):
            continue
        payload = dict(policy.split_metrics).get(split, {})
        if not isinstance(payload, dict):
            continue
        local, downstream, objective = split_metric_views(payload)
        return local, downstream, objective
    return {}, {}, {}


def _primary_metric_value(local: Mapping[str, Any], objective: Mapping[str, Any]) -> float:
    selection_metric_name = str(objective.get("selection_metric_name", "") or "").strip()
    if selection_metric_name:
        value = _safe_float(objective.get(selection_metric_name))
        if math.isfinite(value):
            return float(value)
    selection_metric_value = _safe_float(objective.get("selection_metric_value"))
    if math.isfinite(selection_metric_value):
        return float(selection_metric_value)
    value = _safe_float(objective.get("full_objective_value", objective.get("value", float("nan"))))
    if math.isfinite(value):
        return float(value)
    return _safe_float(local.get("combined"))


def _scenario_signature(run: LoadedLocalLawRun, *, include_support: bool = False) -> str:
    summary = run.summary
    payload = run.payload
    cfg = dict(payload.get("config", {}) or {})
    metadata = dict(summary.metadata or {})
    parts: Dict[str, object] = {
        "family": str(summary.family),
        "dgp": str(summary.dgp),
        "suite_role": str(summary.suite_role),
    }
    for key in (
        "analysis_partition_mode",
        "law_leaf_query_design",
        "law_internal_query_design",
        "fixed_leaf_tokens",
        "feature_mode",
        "model_family",
        "n_regimes",
    ):
        value = metadata.get(key, cfg.get(key))
        if value is not None and value != "":
            parts[str(key)] = value
    quadratic_weight = metadata.get(
        "quadratic_utility_weight",
        metadata.get("lambda_multiplier", cfg.get("quadratic_utility_weight", cfg.get("lambda_multiplier"))),
    )
    if quadratic_weight is not None and quadratic_weight != "":
        parts["quadratic_weight"] = quadratic_weight
    if include_support:
        parts["train_docs"] = int(summary.support_budget.train_docs)
        parts["total_queries_estimate"] = float(summary.support_budget.total_queries_estimate)
    return "|".join(f"{key}={parts[key]}" for key in sorted(parts.keys()))


def build_local_law_report_core(runs: Sequence[LoadedLocalLawRun]) -> Dict[str, Any]:
    families = sorted({str(run.summary.family) for run in runs})
    suite_roles = sorted(
        {str(run.summary.suite_role) for run in runs if str(run.summary.suite_role)}
    )
    selection_counts: Dict[str, int] = {}
    selection_violations = 0
    artifact_counts: List[int] = []
    artifact_roles: Dict[str, int] = {}
    suite_role_rows: List[Dict[str, Any]] = []

    by_suite_role: Dict[str, List[LoadedLocalLawRun]] = {}
    for run in runs:
        summary = run.summary
        by_suite_role.setdefault(str(summary.suite_role or "unlabeled"), []).append(run)
        selection = dict(summary.selection or {})
        split = str(selection.get("selection_split", "") or "unset")
        selection_counts[split] = int(selection_counts.get(split, 0)) + 1
        uses_test = bool(
            selection.get("uses_test_metrics", False)
            or selection.get("test_metrics_used_for_selection", False)
        )
        selected_role = selected_policy_role(summary)
        requires_validation = selected_role in {PolicyRole.LEARNED_G, PolicyRole.CANDIDATE_G}
        if requires_validation and (
            uses_test
            or (
                int(summary.support_budget.val_docs) > 0
                and str(selection.get("selected_candidate", "") or "").strip()
                and split != "val"
            )
        ):
            selection_violations += 1
        g_artifacts = dict(run.payload.get("g_artifacts", {}) or {})
        artifact_counts.append(int(len(g_artifacts)))
        for artifact in g_artifacts.values():
            if not isinstance(artifact, dict):
                continue
            role = str(artifact.get("role", "") or "")
            if role:
                artifact_roles[role] = int(artifact_roles.get(role, 0)) + 1

    for suite_role, group in sorted(by_suite_role.items()):
        baseline_objective = []
        learned_objective = []
        baseline_combined = []
        learned_combined = []
        baseline_task = []
        learned_task = []
        baseline_local_law_objective = []
        learned_local_law_objective = []
        baseline_downstream = []
        learned_downstream = []
        for run in group:
            base_local, base_down, base_objective_metrics = _policy_metrics(
                run.summary,
                split="test",
                role="baseline_g",
            )
            learned_local, learned_down, learned_objective_metrics = _policy_metrics(
                run.summary,
                split="test",
                role="learned_g",
            )
            if base_local or base_objective_metrics:
                baseline_objective.append(_primary_metric_value(base_local, base_objective_metrics))
                baseline_task.append(base_objective_metrics.get("task_objective_value"))
                baseline_local_law_objective.append(
                    base_objective_metrics.get(
                        "local_law_objective_value",
                        base_local.get("combined"),
                    )
                )
            if learned_local or learned_objective_metrics:
                learned_objective.append(
                    _primary_metric_value(learned_local, learned_objective_metrics)
                )
                learned_task.append(learned_objective_metrics.get("task_objective_value"))
                learned_local_law_objective.append(
                    learned_objective_metrics.get(
                        "local_law_objective_value",
                        learned_local.get("combined"),
                    )
                )
            if base_local:
                baseline_combined.append(base_local.get("combined"))
            if learned_local:
                learned_combined.append(learned_local.get("combined"))
            if base_down:
                baseline_downstream.append(base_down.get("oracle_target_abs_error"))
            if learned_down:
                learned_downstream.append(learned_down.get("oracle_target_abs_error"))
        suite_role_rows.append(
            {
                "suite_role": str(suite_role),
                "n_runs": int(len(group)),
                "train_docs": _format_range(run.summary.support_budget.train_docs for run in group),
                "val_docs": _format_range(run.summary.support_budget.val_docs for run in group),
                "test_docs": _format_range(run.summary.support_budget.test_docs for run in group),
                "mean_queries": _safe_mean(
                    run.summary.support_budget.total_queries_estimate for run in group
                ),
                "mean_baseline_objective_test": _safe_mean(baseline_objective),
                "mean_learned_objective_test": _safe_mean(learned_objective),
                "mean_baseline_combined_test": _safe_mean(baseline_combined),
                "mean_learned_combined_test": _safe_mean(learned_combined),
                "mean_baseline_task_objective_test": _safe_mean(baseline_task),
                "mean_learned_task_objective_test": _safe_mean(learned_task),
                "mean_baseline_local_law_objective_test": _safe_mean(baseline_local_law_objective),
                "mean_learned_local_law_objective_test": _safe_mean(learned_local_law_objective),
                "mean_baseline_downstream_abs_error_test": _safe_mean(baseline_downstream),
                "mean_learned_downstream_abs_error_test": _safe_mean(learned_downstream),
            }
        )

    support_scaling: List[Dict[str, Any]] = []
    support_groups: Dict[str, List[LoadedLocalLawRun]] = {}
    for run in runs:
        if str(run.summary.suite_role) != "support_scaling":
            continue
        support_groups.setdefault(_scenario_signature(run), []).append(run)
    for scenario, group in sorted(support_groups.items()):
        buckets: Dict[float, Dict[str, List[float]]] = {}
        for run in group:
            summary = run.summary
            if str(summary.family) == "markov_ops_count":
                support = float(summary.support_budget.train_docs)
            else:
                support = float(summary.support_budget.total_queries_estimate)
                if not math.isfinite(support) or support <= 0.0:
                    support = float(summary.support_budget.train_docs)
            bucket = buckets.setdefault(
                support,
                {
                    "baseline_objective": [],
                    "learned_objective": [],
                    "baseline_combined": [],
                    "learned_combined": [],
                    "baseline_task_objective": [],
                    "learned_task_objective": [],
                    "baseline_local_law_objective": [],
                    "learned_local_law_objective": [],
                },
            )
            base_local, _base_down, base_objective_metrics = _policy_metrics(
                summary,
                split="test",
                role="baseline_g",
            )
            learned_local, _learned_down, learned_objective_metrics = _policy_metrics(
                summary,
                split="test",
                role="learned_g",
            )
            if base_local or base_objective_metrics:
                bucket["baseline_objective"].append(
                    _primary_metric_value(base_local, base_objective_metrics)
                )
                bucket["baseline_task_objective"].append(
                    _safe_float(base_objective_metrics.get("task_objective_value"))
                )
                bucket["baseline_local_law_objective"].append(
                    _safe_float(
                        base_objective_metrics.get(
                            "local_law_objective_value",
                            base_local.get("combined"),
                        )
                    )
                )
            if learned_local or learned_objective_metrics:
                bucket["learned_objective"].append(
                    _primary_metric_value(learned_local, learned_objective_metrics)
                )
                bucket["learned_task_objective"].append(
                    _safe_float(learned_objective_metrics.get("task_objective_value"))
                )
                bucket["learned_local_law_objective"].append(
                    _safe_float(
                        learned_objective_metrics.get(
                            "local_law_objective_value",
                            learned_local.get("combined"),
                        )
                    )
                )
            if base_local:
                bucket["baseline_combined"].append(base_local.get("combined"))
            if learned_local:
                bucket["learned_combined"].append(learned_local.get("combined"))
        supports = sorted(buckets.keys())
        if not supports:
            continue
        baseline_vals = [_safe_mean(buckets[s]["baseline_objective"]) for s in supports]
        learned_vals = [_safe_mean(buckets[s]["learned_objective"]) for s in supports]
        if not any(math.isfinite(v) for v in learned_vals):
            continue
        support_scaling.append(
            {
                "scenario": str(scenario),
                "support_values": [float(v) for v in supports],
                "baseline_objective": [float(v) for v in baseline_vals],
                "learned_objective": [float(v) for v in learned_vals],
                "baseline_combined": [
                    float(_safe_mean(buckets[s]["baseline_combined"])) for s in supports
                ],
                "learned_combined": [
                    float(_safe_mean(buckets[s]["learned_combined"])) for s in supports
                ],
                "baseline_task_objective": [
                    float(_safe_mean(buckets[s]["baseline_task_objective"])) for s in supports
                ],
                "learned_task_objective": [
                    float(_safe_mean(buckets[s]["learned_task_objective"])) for s in supports
                ],
                "baseline_local_law_objective": [
                    float(_safe_mean(buckets[s]["baseline_local_law_objective"])) for s in supports
                ],
                "learned_local_law_objective": [
                    float(_safe_mean(buckets[s]["learned_local_law_objective"])) for s in supports
                ],
                "gap_values": [
                    float(l - b) if math.isfinite(l) and math.isfinite(b) else float("nan")
                    for l, b in zip(learned_vals, baseline_vals)
                ],
            }
        )

    failure_modes: List[Dict[str, Any]] = []
    failure_groups: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        for counterexample in run.summary.counterexamples:
            payload = dict(counterexample.metrics or {}).get("test", {})
            local = dict(payload.get("local_law", {}) or payload.get("local_law_metrics", {}) or {})
            failure_groups.setdefault(str(counterexample.name), []).append(
                {
                    "targeted_laws": list(counterexample.targeted_laws),
                    "c1": local.get("c1"),
                    "c2": local.get("c2"),
                    "c3": local.get("c3"),
                }
            )
    for name, group in sorted(failure_groups.items()):
        targeted = sorted({law for row in group for law in row["targeted_laws"]})
        failure_modes.append(
            {
                "name": str(name),
                "targeted_laws": targeted,
                "n_runs": int(len(group)),
                "mean_c1": _safe_mean(row.get("c1") for row in group),
                "mean_c2": _safe_mean(row.get("c2") for row in group),
                "mean_c3": _safe_mean(row.get("c3") for row in group),
            }
        )

    quadratic_weight_zero_controls: List[Dict[str, Any]] = []
    null_groups: Dict[str, List[LoadedLocalLawRun]] = {}
    for run in runs:
        lam = _safe_float(
            dict(run.summary.metadata or {}).get(
                "lambda_multiplier",
                dict(run.payload.get("config", {}) or {}).get("lambda_multiplier"),
            )
        )
        if math.isfinite(lam) and abs(lam) <= 1e-12:
            null_groups.setdefault(_scenario_signature(run), []).append(run)
    for scenario, group in sorted(null_groups.items()):
        pooled_deltas: List[float] = []
        objective_gaps: List[float] = []
        law_gaps: List[float] = []
        primary_gains: List[float] = []
        for run in group:
            base_local, base_down, base_objective_metrics = _policy_metrics(
                run.summary,
                split="test",
                role="baseline_g",
            )
            learned_local, learned_down, learned_objective_metrics = _policy_metrics(
                run.summary,
                split="test",
                role="learned_g",
            )
            if base_down:
                pooled_deltas.append(abs(_safe_float(base_down.get("oracle_target_delta"))))
            if learned_down:
                pooled_deltas.append(abs(_safe_float(learned_down.get("oracle_target_delta"))))
            if base_local and learned_local:
                law_gaps.append(
                    abs(
                        _safe_float(learned_local.get("combined"))
                        - _safe_float(base_local.get("combined"))
                    )
                )
                objective_gaps.append(
                    abs(
                        _primary_metric_value(learned_local, learned_objective_metrics)
                        - _primary_metric_value(base_local, base_objective_metrics)
                    )
                )
            base_primary = (
                _safe_float(base_down.get("oracle_target_abs_error")) if base_down else float("nan")
            )
            learned_primary = (
                _safe_float(learned_down.get("oracle_target_abs_error"))
                if learned_down
                else float("nan")
            )
            if math.isfinite(base_primary) and math.isfinite(learned_primary):
                if float(base_primary) <= 0.0:
                    primary_gains.append(0.0 if float(learned_primary) <= 0.0 else float("inf"))
                else:
                    primary_gains.append(
                        (float(base_primary) - float(learned_primary)) / float(base_primary)
                    )
        quadratic_weight_zero_controls.append(
            {
                "scenario": str(scenario),
                "n_runs": int(len(group)),
                "n_learned_runs": int(len(primary_gains)),
                "max_abs_delta": max(
                    [abs(v) for v in pooled_deltas if math.isfinite(v)], default=float("nan")
                ),
                "mean_objective_gap": _safe_mean(objective_gaps),
                "mean_primary_gain": _safe_mean(primary_gains),
                "median_abs_primary_gain": _safe_percentile((abs(v) for v in primary_gains), 50.0),
                "p90_abs_primary_gain": _safe_percentile((abs(v) for v in primary_gains), 90.0),
                "primary_gain_positive_rate": _safe_mean((float(v) > 0.0 for v in primary_gains)),
                "mean_law_gap": _safe_mean(law_gaps),
            }
        )

    # Law-stress classification: pass/fail rates by family and law_package.
    law_stress_by_group: Dict[str, List[Dict[str, Any]]] = {}
    law_stress_records = collect_law_stress_assessments(
        [(run.path, run.summary, run.payload) for run in runs]
    )
    for record in law_stress_records:
        family = str(record.get("family", ""))
        law_package = str(record.get("law_package", "") or "").strip() or "unknown"
        stress = dict(record.get("assessment", {}) or {})
        if not stress:
            continue
        key = f"{family}|{law_package}"
        law_stress_by_group.setdefault(key, []).append(stress)

    law_stress_summary: List[Dict[str, Any]] = []
    for key, assessments in sorted(law_stress_by_group.items()):
        parts = key.split("|", 1)
        family = parts[0]
        law_package = parts[1] if len(parts) > 1 else ""
        n = len(assessments)
        law_stress_summary.append(
            {
                "family": family,
                "law_package": law_package,
                "n_runs": n,
                "primary_pass_rate": float(
                    fmean(
                        1.0 if bool(a.get("primary_pass", a.get("bundle_full_success"))) else 0.0
                        for a in assessments
                    )
                ),
                "c1_pass_rate": float(
                    fmean(1.0 if bool(a.get("c1_pass")) else 0.0 for a in assessments)
                ),
                "c2_pass_rate": float(
                    fmean(1.0 if bool(a.get("c2_pass")) else 0.0 for a in assessments)
                ),
                "c3_pass_rate": float(
                    fmean(1.0 if bool(a.get("c3_pass")) else 0.0 for a in assessments)
                ),
                "all_laws_pass_rate": float(
                    fmean(1.0 if bool(a.get("all_laws_pass")) else 0.0 for a in assessments)
                ),
                "mean_laws_improved": float(
                    fmean(float(a.get("laws_improved", 0)) for a in assessments)
                ),
                "mean_primary_gain": float(
                    fmean(float(a.get("primary_gain_frac", 0)) for a in assessments)
                ),
            }
        )

    law_stress_guidance: List[Dict[str, Any]] = []
    family_to_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in law_stress_summary:
        family_to_rows.setdefault(str(row.get("family", "")), []).append(dict(row))
    for family, rows_for_family in sorted(family_to_rows.items()):
        expected_main = _expected_main_package(
            family,
            [str(row.get("law_package", "")) for row in rows_for_family],
        )
        main_row = next(
            (
                row
                for row in rows_for_family
                if str(row.get("law_package", "")) == str(expected_main)
            ),
            None,
        )
        ablation_rows = [
            row for row in rows_for_family if str(row.get("law_package", "")) != str(expected_main)
        ]
        strongest_ablation = None
        if ablation_rows:
            strongest_ablation = max(
                ablation_rows,
                key=lambda row: (
                    _safe_float(row.get("primary_pass_rate"), 0.0),
                    _safe_float(row.get("mean_primary_gain"), float("-inf")),
                    _safe_float(row.get("mean_laws_improved"), float("-inf")),
                ),
            )

        if expected_main is None:
            note = (
                "No canonical full-package row is present for this family in the current inventory."
            )
        elif main_row is None:
            note = f"The expected full-package row `{expected_main}` is missing from the current inventory."
        elif strongest_ablation is None:
            note = f"Only the expected full-package row `{expected_main}` is present here; the ablation story is not covered by this inventory."
        else:
            main_gain = _safe_float(main_row.get("mean_primary_gain"))
            main_pass = _safe_float(main_row.get("primary_pass_rate"))
            ab_pkg = str(strongest_ablation.get("law_package", ""))
            ab_gain = _safe_float(strongest_ablation.get("mean_primary_gain"))
            ab_pass = _safe_float(strongest_ablation.get("primary_pass_rate"))
            if ab_pass > main_pass or ab_gain > main_gain:
                note = (
                    f"The expected claim row is `{expected_main}`. `{ab_pkg}` is stronger on downstream metrics in this sweep, "
                    "but it is an ablation and should be read as a mechanism diagnostic rather than a replacement main claim."
                )
            else:
                note = f"`{expected_main}` remains the expected claim row. Ablations are present as mechanism checks and should not be read as alternative success criteria."

        law_stress_guidance.append(
            {
                "family": family,
                "expected_main_package": expected_main,
                "main_row": main_row,
                "strongest_ablation_row": strongest_ablation,
                "note": note,
            }
        )

    return {
        "n_protocol_runs": int(len(runs)),
        "families": families,
        "suite_roles": suite_roles,
        "selection": {
            "selection_split_counts": selection_counts,
            "test_metric_selection_violations": int(selection_violations),
        },
        "artifacts": {
            "runs_with_artifacts": int(sum(1 for x in artifact_counts if int(x) > 0)),
            "mean_artifacts_per_run": _safe_mean(artifact_counts),
            "role_counts": artifact_roles,
        },
        "suite_role_overview": suite_role_rows,
        "support_scaling": support_scaling,
        "failure_modes": failure_modes,
        "quadratic_weight_zero_controls": quadratic_weight_zero_controls,
        "law_stress_summary": law_stress_summary,
        "law_stress_guidance": law_stress_guidance,
    }


def render_local_law_report_markdown(core: Mapping[str, Any]) -> List[str]:
    lines: List[str] = [
        "## Unified Core",
        "",
        "- Both families are summarized under the same local-law contract: `oracle_g`, matched `baseline_g`, selected `learned_g`, held-out downstream error, and serialized `g` artifacts.",
        f"- Protocol runs loaded: `{int(core.get('n_protocol_runs', 0))}`.",
        f"- Families present: `{', '.join(core.get('families', [])) or 'none'}`.",
        f"- Suite roles present: `{', '.join(core.get('suite_roles', [])) or 'none'}`.",
    ]
    selection = dict(core.get("selection", {}) or {})
    lines.append(
        "- Selection protocol: "
        f"`{selection.get('selection_split_counts', {})}` splits, "
        f"`{int(selection.get('test_metric_selection_violations', 0))}` test-selection violations."
    )
    artifacts = dict(core.get("artifacts", {}) or {})
    lines.append(
        "- Artifact coverage: "
        f"`{int(artifacts.get('runs_with_artifacts', 0))}` runs with artifacts, "
        f"mean `{_safe_float(artifacts.get('mean_artifacts_per_run')):.2f}` artifacts/run."
    )
    lines.extend(["", "### Suite Role Overview", ""])
    for row in list(core.get("suite_role_overview", []) or []):
        lines.append(
            "- "
            f"`{row.get('suite_role', 'unknown')}`: "
            f"runs `{row.get('n_runs', 0)}`, train_docs `{row.get('train_docs', '-')}`, "
            f"queries `{_safe_float(row.get('mean_queries')):.1f}`, "
            f"baseline objective `{_safe_float(row.get('mean_baseline_objective_test')):.4f}`, "
            f"learned objective `{_safe_float(row.get('mean_learned_objective_test')):.4f}`, "
            f"baseline combined `{_safe_float(row.get('mean_baseline_combined_test')):.4f}`, "
            f"learned combined `{_safe_float(row.get('mean_learned_combined_test')):.4f}`, "
            f"baseline task `{_safe_float(row.get('mean_baseline_task_objective_test')):.4f}`, "
            f"learned task `{_safe_float(row.get('mean_learned_task_objective_test')):.4f}`."
        )
    if core.get("failure_modes"):
        lines.extend(["", "### Failure Modes", ""])
        for row in list(core.get("failure_modes", []) or []):
            lines.append(
                "- "
                f"`{row.get('name', 'unknown')}` targets `{row.get('targeted_laws', [])}` "
                f"with mean `(C1, C2, C3)=({_safe_float(row.get('mean_c1')):.3f}, {_safe_float(row.get('mean_c2')):.3f}, {_safe_float(row.get('mean_c3')):.3f})`."
            )
    if core.get("quadratic_weight_zero_controls"):
        lines.extend(["", "### Downstream Null Controls", ""])
        for row in list(core.get("quadratic_weight_zero_controls", []) or []):
            lines.append(
                "- "
                f"`{row.get('scenario', 'unknown')}`: "
                f"median `|primary gain|` `{_safe_float(row.get('median_abs_primary_gain')):.4f}`, "
                f"p90 `|primary gain|` `{_safe_float(row.get('p90_abs_primary_gain')):.4f}`, "
                f"diagnostic max `|Delta_vs_pooled|` `{_safe_float(row.get('max_abs_delta')):.4f}`, "
                f"mean law-gap `{_safe_float(row.get('mean_law_gap')):.4f}`."
            )
    if core.get("law_stress_summary"):
        lines.extend(["", "### Law-Stress Classification (Cross-DGP)", ""])
        lines.append(
            "- `PrimGain` is downstream fractional improvement versus the matched baseline; positive is better, negative is worse."
        )
        lines.append(
            "- Only the expected full package should be read as the paper claim. Single-law or partial-law rows are ablations/mechanism diagnostics."
        )
        guidance_rows = list(core.get("law_stress_guidance", []) or [])
        for row in guidance_rows:
            family = str(row.get("family", "") or "unknown")
            expected = str(row.get("expected_main_package", "") or "n/a")
            lines.append(
                f"- `{family}`: expected claim package `{expected}`. {row.get('note', '')}"
            )
        lines.extend([""])
        header = f"{'Family':<35} {'Package':<14} {'N':>5} {'Prim%':>6} {'C1%':>5} {'C2%':>5} {'C3%':>5} {'Laws':>5} {'PrimGain':>9}"
        lines.append("```")
        lines.append(header)
        lines.append("-" * len(header))
        for row in list(core.get("law_stress_summary", []) or []):
            lines.append(
                f"{row.get('family', ''):<35} {row.get('law_package', ''):<14} "
                f"{row.get('n_runs', 0):>5} "
                f"{_safe_float(row.get('primary_pass_rate')):>5.1%} "
                f"{_safe_float(row.get('c1_pass_rate')):>4.0%} "
                f"{_safe_float(row.get('c2_pass_rate')):>4.0%} "
                f"{_safe_float(row.get('c3_pass_rate')):>4.0%} "
                f"{_safe_float(row.get('mean_laws_improved')):>5.1f} "
                f"{_safe_float(row.get('mean_primary_gain')):>8.1%}"
            )
        lines.append("```")
    return lines


def _write_text_page(pdf: PdfPages, *, title: str, lines: Sequence[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.90])
    ax.axis("off")
    ax.text(0.0, 1.0, title, fontsize=16, fontweight="bold", va="top")
    y = 0.95
    for raw in lines:
        chunks = textwrap.wrap(
            str(raw), width=105, break_long_words=False, break_on_hyphens=False
        ) or [""]
        for chunk in chunks:
            ax.text(0.0, y, chunk, fontsize=10.0, va="top")
            y -= 0.024
            if y < 0.05:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_axes([0.06, 0.05, 0.88, 0.90])
                ax.axis("off")
                y = 0.97
    pdf.savefig(fig)
    plt.close(fig)


def _write_support_scaling_page(
    pdf: PdfPages, *, title: str, curves: Sequence[Mapping[str, Any]]
) -> None:
    shown = list(curves)[:4]
    if not shown:
        return
    ncols = 2 if len(shown) > 1 else 1
    nrows = int(math.ceil(len(shown) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 8.5))
    axes_arr = np.atleast_1d(axes).reshape(-1)
    for ax, curve in zip(axes_arr, shown):
        xs = np.asarray(curve.get("support_values", []), dtype=np.float64)
        baseline = np.asarray(curve.get("baseline_objective", []), dtype=np.float64)
        learned = np.asarray(curve.get("learned_objective", []), dtype=np.float64)
        ax.plot(xs, baseline, color="#666666", lw=2.0, marker="o", label="baseline_g")
        ax.plot(xs, learned, color="#1f77b4", lw=2.0, marker="o", label="learned_g")
        ax.set_title(str(curve.get("scenario", ""))[:96], fontsize=9)
        ax.set_xlabel("Support")
        ax.set_ylabel("Configured objective")
        ax.grid(alpha=0.25, lw=0.6)
    for ax in axes_arr[len(shown) :]:
        ax.axis("off")
    axes_arr[0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout(rect=(0.03, 0.03, 0.98, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _write_failure_modes_page(
    pdf: PdfPages, *, title: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    shown = list(rows)
    if not shown:
        return
    names = [str(row.get("name", "")) for row in shown]
    c1 = [_safe_float(row.get("mean_c1")) for row in shown]
    c2 = [_safe_float(row.get("mean_c2")) for row in shown]
    c3 = [_safe_float(row.get("mean_c3")) for row in shown]
    x = np.arange(len(names), dtype=np.float64)
    width = 0.22
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.bar(x - width, c1, width=width, label="C1", color="#457b9d")
    ax.bar(x, c2, width=width, label="C2", color="#e07a5f")
    ax.bar(x + width, c3, width=width, label="C3", color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean local-law metric")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.25)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def write_local_law_report_core_pages(
    pdf: PdfPages,
    *,
    title: str,
    core: Mapping[str, Any],
) -> None:
    overview_lines = [
        "This report now shares a theorem-facing core with the other local-law families.",
        f"Protocol runs loaded: {int(core.get('n_protocol_runs', 0))}.",
        f"Families present: {', '.join(core.get('families', [])) or 'none'}.",
        f"Suite roles: {', '.join(core.get('suite_roles', [])) or 'none'}.",
    ]
    selection = dict(core.get("selection", {}) or {})
    overview_lines.append(
        "Selection protocol: "
        f"{selection.get('selection_split_counts', {})} split counts; "
        f"{int(selection.get('test_metric_selection_violations', 0))} runs violate validation-only selection."
    )
    artifacts = dict(core.get("artifacts", {}) or {})
    overview_lines.append(
        "Artifact coverage: "
        f"{int(artifacts.get('runs_with_artifacts', 0))} runs with serialized g artifacts; "
        f"mean artifacts/run={_safe_float(artifacts.get('mean_artifacts_per_run')):.2f}."
    )
    if core.get("suite_role_overview"):
        overview_lines.append("")
        overview_lines.append("Per-suite-role overview:")
        for row in list(core.get("suite_role_overview", []) or []):
            overview_lines.append(
                f"{row.get('suite_role', 'unknown')}: "
                f"runs={row.get('n_runs', 0)}, train_docs={row.get('train_docs', '-')}, "
                f"baseline_objective={_safe_float(row.get('mean_baseline_objective_test')):.4f}, "
                f"learned_objective={_safe_float(row.get('mean_learned_objective_test')):.4f}, "
                f"baseline_combined={_safe_float(row.get('mean_baseline_combined_test')):.4f}, "
                f"learned_combined={_safe_float(row.get('mean_learned_combined_test')):.4f}, "
                f"baseline_task={_safe_float(row.get('mean_baseline_task_objective_test')):.4f}, "
                f"learned_task={_safe_float(row.get('mean_learned_task_objective_test')):.4f}, "
                f"baseline_downstream={_safe_float(row.get('mean_baseline_downstream_abs_error_test')):.4f}, "
                f"learned_downstream={_safe_float(row.get('mean_learned_downstream_abs_error_test')):.4f}."
            )
    _write_text_page(pdf, title=f"{title} | Unified Core", lines=overview_lines)
    _write_support_scaling_page(
        pdf,
        title=f"{title} | Shared Support Scaling",
        curves=list(core.get("support_scaling", []) or []),
    )
    _write_failure_modes_page(
        pdf, title=f"{title} | Shared Failure Modes", rows=list(core.get("failure_modes", []) or [])
    )
    if core.get("quadratic_weight_zero_controls"):
        null_lines = [
            "Null-control scenarios where quadratic weight=0 should keep learned-vs-baseline primary gains modest even if pooled-relative Delta remains diagnostic-only.",
        ]
        for row in list(core.get("quadratic_weight_zero_controls", []) or []):
            null_lines.append(
                f"{row.get('scenario', 'unknown')}: "
                f"n_runs={row.get('n_runs', 0)}, "
                f"n_learned_runs={row.get('n_learned_runs', 0)}, "
                f"median |primary gain|={_safe_float(row.get('median_abs_primary_gain')):.4f}, "
                f"p90 |primary gain|={_safe_float(row.get('p90_abs_primary_gain')):.4f}, "
                f"max |Delta_vs_pooled|={_safe_float(row.get('max_abs_delta')):.4f}, "
                f"mean objective-gap={_safe_float(row.get('mean_objective_gap')):.4f}, "
                f"mean law-gap={_safe_float(row.get('mean_law_gap')):.4f}."
            )
        _write_text_page(pdf, title=f"{title} | Shared Downstream Null Controls", lines=null_lines)

    if core.get("law_stress_summary"):
        stress_lines = [
            "Law-stress classification: pass/fail rates by DGP family and law package.",
            "Primary metric = root MAE improvement >= 10%. Laws (C1/C2/C3) are diagnostics.",
            "PrimGain = downstream fractional improvement vs matched baseline; positive is better, negative is worse.",
            "Only the expected full package should be read as the paper claim. Single-law or partial-law rows are ablations/mechanism diagnostics.",
            "",
        ]
        for row in list(core.get("law_stress_guidance", []) or []):
            stress_lines.append(
                f"{row.get('family', 'unknown')}: expected claim package={row.get('expected_main_package', 'n/a')} | {row.get('note', '')}"
            )
        stress_lines.append("")
        header = f"{'Family':<35} {'Package':<14} {'N':>5} {'Prim%':>6} {'C1%':>5} {'C2%':>5} {'C3%':>5} {'Laws':>5} {'PrimGain':>9}"
        stress_lines.append(header)
        stress_lines.append("-" * len(header))
        for row in list(core.get("law_stress_summary", []) or []):
            stress_lines.append(
                f"{row.get('family', ''):<35} {row.get('law_package', ''):<14} "
                f"{row.get('n_runs', 0):>5} "
                f"{_safe_float(row.get('primary_pass_rate')):>5.1%} "
                f"{_safe_float(row.get('c1_pass_rate')):>4.0%} "
                f"{_safe_float(row.get('c2_pass_rate')):>4.0%} "
                f"{_safe_float(row.get('c3_pass_rate')):>4.0%} "
                f"{_safe_float(row.get('mean_laws_improved')):>5.1f} "
                f"{_safe_float(row.get('mean_primary_gain')):>8.1%}"
            )
        _write_text_page(
            pdf, title=f"{title} | Cross-DGP Law-Stress Classification", lines=stress_lines
        )


__all__ = [
    "LoadedLocalLawRun",
    "build_local_law_report_core",
    "load_local_law_runs",
    "render_local_law_report_markdown",
    "write_local_law_report_core_pages",
]
