from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from treepo.bench.io import atomic_write_text, dump_json, write_csv_rows

CAPACITY_ORDER = {"small": 0, "medium": 1, "large": 2}


def _scan_rows(output_root: Path) -> List[dict]:
    rows: List[dict] = []
    for path in Path(output_root).rglob("*.json"):
        if "reports" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        candidate = payload.get("rows")
        if not isinstance(candidate, list) or not candidate or not isinstance(candidate[0], dict):
            continue
        if "family" not in candidate[0] or "sketch" not in candidate[0]:
            continue
        config = payload.get("config")
        config = config if isinstance(config, dict) else {}
        for r in candidate:
            row = dict(r)
            for key in ("min_tokens", "max_tokens", "n_docs"):
                if key in config:
                    row.setdefault(f"_config_{key}", config.get(key))
            rows.append(row)
    return rows


def _rate_key(row: dict, name: str) -> str:
    try:
        return f"{float(row.get(name, 1.0)):.12g}"
    except Exception:
        return "1"


def _aggregate(rows: Sequence[dict]) -> List[dict]:
    groups: Dict[Tuple[str, str, str, str, str, str, str, str, str, str, str, str, str], List[dict]] = {}
    for row in rows:
        learned_variant = (
            str(row.get("learned_variant", ""))
            if str(row.get("implementation_status", "")) == "learned_empirical"
            else ""
        )
        readout_arch = (
            str(row.get("readout_arch", ""))
            if str(row.get("implementation_status", "")) == "learned_empirical"
            else ""
        )
        leaf_axis = str(row.get("leaf_axis", "n_leaves"))
        key = (
            str(row.get("family")),
            str(row.get("sketch")),
            str(row.get("query")),
            str(row.get("capacity_label", "single")),
            str(row.get("n_leaves", "-1")),
            str(row.get("leaf_size", "-1")),
            leaf_axis,
            _rate_key(row, "learned_root_query_rate"),
            _rate_key(row, "learned_leaf_query_rate"),
            _rate_key(row, "learned_internal_query_rate"),
            str(row.get("learned_supervision_sampling_policy", "separate_axes")),
            learned_variant,
            readout_arch,
        )
        groups.setdefault(key, []).append(row)
    out: List[dict] = []
    for (
        family,
        sketch,
        query,
        capacity_label,
        n_leaves,
        leaf_size,
        leaf_axis,
        learned_root_query_rate,
        learned_leaf_query_rate,
        learned_internal_query_rate,
        learned_supervision_sampling_policy,
        _learned_variant,
        _readout_arch,
    ), grows in sorted(groups.items()):
        def arr(name: str) -> np.ndarray:
            vals = []
            for r in grows:
                try:
                    vals.append(float(r.get(name, np.nan)))
                except Exception:
                    vals.append(float("nan"))
            return np.asarray(vals, dtype=np.float64)

        def nanmean(values: np.ndarray) -> float:
            finite = values[np.isfinite(values)]
            return float(np.mean(finite)) if len(finite) else float("nan")

        def nanstd(values: np.ndarray) -> float:
            finite = values[np.isfinite(values)]
            return float(np.std(finite)) if len(finite) else float("nan")

        def ci95(values: np.ndarray) -> float:
            finite = values[np.isfinite(values)]
            if len(finite) < 2:
                return 0.0 if len(finite) == 1 else float("nan")
            return float(1.96 * np.std(finite, ddof=1) / math.sqrt(float(len(finite))))

        def nanint(name: str, default: int = -1) -> int:
            value = nanmean(arr(name))
            return int(round(value)) if np.isfinite(value) else int(default)

        rel = arr("relative_rmse")
        spread = arr("schedule_spread_mean")
        dist = arr("distance_to_official_floor")
        floor = arr("official_floor_rel_rmse")
        coverage = arr("bound_coverage_2sigma")
        theory = arr("theoretical_error")
        n_leaves_int = int(float(n_leaves))
        leaf_size_int = int(float(leaf_size))
        tokens_per_leaf_mean = nanmean(arr("tokens_per_leaf_mean"))
        if not np.isfinite(tokens_per_leaf_mean):
            if leaf_axis == "leaf_size" and leaf_size_int > 0:
                tokens_per_leaf_mean = float(leaf_size_int)
            elif n_leaves_int > 0:
                min_tok = nanmean(arr("_config_min_tokens"))
                max_tok = nanmean(arr("_config_max_tokens"))
                if np.isfinite(min_tok) and np.isfinite(max_tok):
                    tokens_per_leaf_mean = 0.5 * (float(min_tok) + float(max_tok)) / float(n_leaves_int)
        row = {
            "family": family,
            "sketch": sketch,
            "query": query,
            "capacity_label": capacity_label,
            "n_leaves": n_leaves_int,
            "leaf_size": leaf_size_int,
            "leaf_axis": leaf_axis,
            "learned_root_query_rate": float(learned_root_query_rate),
            "learned_leaf_query_rate": float(learned_leaf_query_rate),
            "learned_internal_query_rate": float(learned_internal_query_rate),
            "learned_supervision_sampling_policy": str(learned_supervision_sampling_policy),
            "leaf_count_min": nanint("leaf_count_min"),
            "leaf_count_mean": nanmean(arr("leaf_count_mean")),
            "leaf_count_max": nanint("leaf_count_max"),
            "tokens_per_leaf_min": nanmean(arr("tokens_per_leaf_min")),
            "tokens_per_leaf_mean": tokens_per_leaf_mean,
            "tokens_per_leaf_max": nanmean(arr("tokens_per_leaf_max")),
            "n_runs": int(len(grows)),
            "implementation_status": str(grows[0].get("implementation_status", "")),
            "formal_status": str(grows[0].get("formal_status", "")),
            "relative_rmse_mean": nanmean(rel),
            "relative_rmse_std": nanstd(rel),
            "relative_rmse_ci95": ci95(rel),
            "schedule_spread_mean": nanmean(spread),
            "schedule_spread_ci95": ci95(spread),
            "bound_coverage_2sigma_mean": nanmean(coverage),
            "theoretical_error_mean": nanmean(theory),
            "official_floor_rel_rmse_mean": nanmean(floor),
            "distance_to_official_floor_mean": nanmean(dist),
            "distance_to_official_floor_ci95": ci95(dist),
            "memory_bytes_mean": nanmean(arr("memory_bytes_mean")),
            "memory_bytes_ci95": ci95(arr("memory_bytes_mean")),
        }
        for key in (
            "distinct_lg_k",
            "theta_lg_k",
            "cms_num_hashes",
            "cms_num_buckets",
            "frequent_lg_max_map_size",
            "kll_k",
            "quantiles_k",
            "req_k",
            "tdigest_k",
            "tuple_lg_k",
            "varopt_k",
        ):
            if key in grows[0]:
                try:
                    row[key] = int(float(grows[0].get(key)))
                except Exception:
                    row[key] = grows[0].get(key)
        for key in (
            "learned_target_kind",
            "learned_variant",
            "learned_codename",
            "learned_run_slug",
            "projection_kind",
            "readout_arch",
            "learned_readout_arch",
            "exact_state_mode",
            "state_space_kind",
            "merge_kind",
            "readout_kind",
            "leaf_feature_mode",
            "learned_gpu_ids",
            "learned_stage_components",
            "learned_trained_stage_components",
            "learned_supervision_sampling_policy",
            "learned_reused_prefix",
            "learned_prefix_variant",
            "learned_suffix_variant",
            "final_f_checkpoint",
            "final_g_checkpoint",
            "final_leaf_adapter_checkpoint",
        ):
            if key in grows[0]:
                row[key] = grows[0].get(key)
        for key in (
            "learned_embedding_dim",
            "learned_summary_dim",
            "learned_state_dim",
            "learned_hidden_dim",
            "learned_leaf_width_floor",
            "leaf_feature_dim",
            "g_input_dim",
            "output_dim",
            "learned_batch_size",
            "learned_effective_batch_size",
            "learned_effective_batch_size_uncapped",
            "learned_batch_size_base",
            "learned_batch_reference_leaf_size",
            "learned_max_batch_size",
            "learned_leaf_token_batch_budget",
            "learned_cuda_device",
            "learned_target_jobs",
        ):
            if key in grows[0]:
                row[key] = nanint(key, default=0)
        out.append(row)
    out.sort(
        key=lambda r: (
            str(r.get("family")),
            str(r.get("sketch")),
            str(r.get("query")),
            CAPACITY_ORDER.get(str(r.get("capacity_label")), 999),
            int(r.get("leaf_size", -1)),
            int(r.get("n_leaves", -1)),
            float(r.get("learned_root_query_rate", 1.0)),
            float(r.get("learned_leaf_query_rate", 1.0)),
            float(r.get("learned_internal_query_rate", 1.0)),
            str(r.get("learned_supervision_sampling_policy", "separate_axes")),
            str(r.get("learned_variant", "")),
        )
    )
    return out


def _capacity_x(label: object) -> int:
    return CAPACITY_ORDER.get(str(label), 999)


def _series_label(sketch: str, query: str) -> str:
    if query in {"cardinality", "top5_point_frequency", "total_weight", "accumulator_summary_sum"}:
        return sketch
    return f"{sketch}:{query}"


METHOD_GROUPS = ("official", "learned_f", "learned_g", "learned_joint", "learned_other")
PRIMARY_METHOD_GROUPS = ("official", "learned_g", "learned_joint")
METHOD_LABELS = {
    "official": "Official/oracle",
    "learned_f": r"learned $f$",
    "learned_g": r"Learned $g$ + fixed $f^\star$",
    "learned_joint": r"Learned mergeable projection",
    "learned_other": r"learned (other variant)",
}
METHOD_COLORS = {
    "official": "#2f5f8f",
    "learned_f": "#7b3294",
    "learned_g": "#d95f02",
    "learned_joint": "#1b7837",
    "learned_other": "#666666",
}
SKETCH_COLORS = [
    "#2f5f8f",
    "#d95f02",
    "#1b7837",
    "#762a83",
    "#a6761d",
    "#666666",
    "#e7298a",
    "#66a61e",
]
CAPACITY_COLORS = {
    "small": "#8b8b8b",
    "medium": "#4c78a8",
    "large": "#f58518",
}


def _method_group(row: dict) -> str:
    sketch = str(row.get("sketch", ""))
    status = str(row.get("implementation_status", ""))
    variant = str(row.get("learned_variant", ""))
    if variant:
        if variant == "f":
            return "learned_f"
        if variant == "g":
            return "learned_g"
        if all(c in ("f", "g") for c in variant):
            return "learned_joint"
    if sketch.startswith("learned_joint_"):
        return "learned_joint"
    # Legacy split names now report as the single joint codename.
    if sketch.startswith("learned_fg_") or sketch.startswith("learned_gf_"):
        return "learned_joint"
    if sketch.startswith("learned_f_"):
        return "learned_f"
    if sketch.startswith("learned_g_"):
        return "learned_g"
    if sketch.startswith("learned_") and status == "learned_empirical":
        parts = sketch.split("_", 2)
        if len(parts) >= 2 and parts[1] and all(c in ("f", "g") for c in parts[1]):
            return "learned_joint"
        return "learned_other"
    if status in {"official_empirical", "lean_backed"}:
        return "official"
    return status


def _finite_float(row: dict, key: str) -> float:
    try:
        value = float(row.get(key, np.nan))
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _has_learned_rows(rows: Sequence[dict]) -> bool:
    return any(str(r.get("implementation_status", "")) == "learned_empirical" for r in rows)


def _plot_mode(rows: Sequence[dict]) -> str:
    return "projection_gap" if _has_learned_rows(rows) else "relative_rmse"


def _plot_value(row: dict, mode: str) -> float:
    if mode == "projection_gap" and _method_group(row) == "official":
        return 0.0
    return _finite_float(row, "relative_rmse_mean")


def _plot_ci(row: dict, mode: str) -> float:
    try:
        if int(float(row.get("n_runs", 1))) <= 1:
            return 0.0
    except Exception:
        return 0.0
    if mode == "projection_gap" and _method_group(row) == "official":
        return 0.0
    val = _finite_float(row, "relative_rmse_ci95")
    return 0.0 if not np.isfinite(val) else float(val)


def _plot_ylabel(mode: str, *, family_detail: bool = False) -> str:
    if mode == "projection_gap":
        return "projection gap (relative RMSE)"
    return "relative/rank RMSE" if family_detail else "mean relative RMSE"


def _draw_series(
    ax,
    xs: Sequence[int],
    ys: Sequence[float],
    yerr: Sequence[float],
    *,
    color: str,
    label: str,
    linewidth: float,
    markersize: float,
) -> None:
    if any(float(e) > 0.0 for e in yerr):
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker="o",
            linewidth=linewidth,
            markersize=markersize,
            capsize=2.0,
            color=color,
            label=label,
        )
    else:
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=linewidth,
            markersize=markersize,
            color=color,
            label=label,
        )


def _best_row(
    rows: Sequence[dict],
    *,
    method: str,
    family: str,
    query: str,
    capacity: str | None,
    n_leaves: int,
    metric: str,
) -> dict | None:
    candidates: List[tuple[float, dict]] = []
    for row in rows:
        if _method_group(row) != method:
            continue
        if str(row.get("family")) != family or str(row.get("query")) != query:
            continue
        if capacity is not None and str(row.get("capacity_label")) != capacity:
            continue
        if int(row.get("n_leaves", -1)) != int(n_leaves):
            continue
        value = _finite_float(row, metric)
        if np.isfinite(value):
            candidates.append((value, row))
    return min(candidates, key=lambda x: x[0])[1] if candidates else None


def _panel_axes(rows: Sequence[dict]) -> tuple[list[tuple[str, str]], list[int], list[str]]:
    panels = sorted({(str(r.get("family")), str(r.get("query"))) for r in rows})
    leaves = sorted({int(r.get("n_leaves", -1)) for r in rows if int(r.get("n_leaves", -1)) > 0})
    capacities = sorted({str(r.get("capacity_label", "")) for r in rows}, key=_capacity_x)
    return panels, leaves, capacities


def _plot_summary(rows: Sequence[dict], output: Path) -> None:
    if not rows:
        return
    families = sorted({str(r.get("family")) for r in rows})
    leaves = sorted({int(r.get("n_leaves", -1)) for r in rows if int(r.get("n_leaves", -1)) > 0})
    capacities = sorted({str(r.get("capacity_label", "")) for r in rows}, key=_capacity_x)
    preferred_capacity = "large" if "large" in capacities else capacities[-1]
    ncols = 3
    nrows = int(np.ceil(len(families) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15, max(3.2, 3.2 * nrows)),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, family in zip(axes.ravel(), families):
        family_rows = [r for r in rows if str(r.get("family")) == family]
        queries = sorted({str(r.get("query")) for r in family_rows})
        for method in METHOD_GROUPS:
            xs: list[int] = []
            ys: list[float] = []
            for n_leaves in leaves:
                vals: list[float] = []
                for query in queries:
                    best = _best_row(
                        rows,
                        method=method,
                        family=family,
                        query=query,
                        capacity=preferred_capacity,
                        n_leaves=n_leaves,
                        metric="relative_rmse_mean",
                    )
                    if best is not None:
                        vals.append(_finite_float(best, "relative_rmse_mean"))
                finite = [v for v in vals if np.isfinite(v)]
                if finite:
                    xs.append(n_leaves)
                    ys.append(float(np.mean(finite)))
            if xs:
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=1.8,
                    markersize=4.0,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                )
        ax.set_title(family)
        ax.set_xlabel("leaf count L")
        ax.set_ylabel("mean best RMSE")
        ax.set_xticks(leaves)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best", frameon=False)
    for ax in axes.ravel()[len(families) :]:
        ax.axis("off")
    fig.suptitle(
        "Broad Sketch Summary: Best Large-Capacity Row Per Method",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _plot_method_group(rows: Sequence[dict], method: str, output: Path) -> None:
    method_rows = [r for r in rows if _method_group(r) == method]
    panels, leaves, capacities = _panel_axes(method_rows)
    if not panels:
        return
    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(13, max(3.2, 2.85 * nrows)),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, (family, query) in zip(axes.ravel(), panels):
        for capacity in capacities:
            xs: list[int] = []
            ys: list[float] = []
            yerr: list[float] = []
            for n_leaves in leaves:
                best = _best_row(
                    rows,
                    method=method,
                    family=family,
                    query=query,
                    capacity=capacity,
                    n_leaves=n_leaves,
                    metric="relative_rmse_mean",
                )
                if best is None:
                    continue
                xs.append(n_leaves)
                ys.append(_finite_float(best, "relative_rmse_mean"))
                yerr.append(_finite_float(best, "relative_rmse_ci95"))
            if xs:
                clean_err = [0.0 if not np.isfinite(v) else v for v in yerr]
                ax.errorbar(
                    xs,
                    ys,
                    yerr=clean_err,
                    label=capacity,
                    color=CAPACITY_COLORS.get(capacity, "#555555"),
                    marker="o",
                    linewidth=1.6,
                    markersize=3.5,
                    capsize=2.0,
                )
        ax.set_title(f"{family}: {_pretty_query(query)}", fontsize=9)
        ax.set_xlabel("leaf count L")
        ax.set_ylabel("relative/rank RMSE")
        ax.set_xticks(leaves)
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(fontsize=7, frameon=False, loc="best")
    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")
    fig.suptitle(f"{METHOD_LABELS[method]}: Raw Error by Capacity", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=185)
    plt.close(fig)


def _plot_gold_gap(rows: Sequence[dict], output: Path) -> None:
    panels, leaves, capacities = _panel_axes(rows)
    if not panels:
        return
    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(13, max(3.2, 2.85 * nrows)),
        squeeze=False,
        constrained_layout=True,
    )
    linestyle = {"learned_f": ":", "learned_g": "--", "learned_joint": "-"}
    marker = {"learned_f": "^", "learned_g": "s", "learned_joint": "o"}
    for ax, (family, query) in zip(axes.ravel(), panels):
        for method in ("learned_f", "learned_g", "learned_joint"):
            for capacity in capacities:
                xs: list[int] = []
                ys: list[float] = []
                for n_leaves in leaves:
                    best = _best_row(
                        rows,
                        method=method,
                        family=family,
                        query=query,
                        capacity=capacity,
                        n_leaves=n_leaves,
                        metric="distance_to_official_floor_mean",
                    )
                    if best is None:
                        continue
                    value = _finite_float(best, "distance_to_official_floor_mean")
                    if not np.isfinite(value):
                        continue
                    xs.append(n_leaves)
                    ys.append(max(0.0, value))
                if xs:
                    ax.plot(
                        xs,
                        ys,
                        label=f"{METHOD_LABELS[method]}, {capacity}",
                        color=CAPACITY_COLORS.get(capacity, "#555555"),
                        linestyle=linestyle[method],
                        marker=marker[method],
                        linewidth=1.4,
                        markersize=3.2,
                    )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        ax.set_title(f"{family}: {query}", fontsize=9)
        ax.set_xlabel("leaf count L")
        ax.set_ylabel("excess RMSE over official floor")
        ax.set_xticks(leaves)
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(fontsize=6, frameon=False, loc="best")
    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")
    fig.suptitle("Learned Excess Error Over the Gold-Standard Floor", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=185)
    plt.close(fig)


def _pretty_query(value: object) -> str:
    query = str(value)
    labels = {
        "a_not_b": "A not B",
        "accumulator_summary_sum": "tuple sum",
        "cardinality": "cardinality",
        "intersection": "intersection",
        "rank_at_q0.5": "rank @ median",
        "rank_at_q0.95": "rank @ q=.95",
        "top5_point_frequency": "top-5 freq.",
        "total_weight": "total weight",
        "union": "union",
    }
    return labels.get(query, query.replace("_", " "))


def _pretty_sketch(value: object) -> str:
    sketch = str(value)
    labels = {
        "count_min_datasketches": "Count-Min",
        "cpc_datasketches": "CPC",
        "frequent_strings_datasketches": "Frequent Items",
        "hll_datasketches": "HLL DataSketches",
        "hll_native": "HLL native",
        "kll_floats_datasketches": "KLL",
        "quantiles_floats_datasketches": "classic quantiles",
        "req_floats_datasketches": "REQ",
        "tdigest_double_datasketches": "t-digest",
        "theta_datasketches": "Theta",
        "tuple_accumulator_datasketches": "Tuple accumulator",
        "varopt_strings_datasketches": "VarOpt",
    }
    return labels.get(sketch, sketch.replace("_datasketches", "").replace("_", " "))


def _is_learned_row(row: dict) -> bool:
    return str(row.get("implementation_status", "")) == "learned_empirical"


def _is_exact_state_row(row: dict) -> bool:
    if not _is_learned_row(row):
        return False
    mode = str(row.get("exact_state_mode", ""))
    projection = str(row.get("projection_kind", ""))
    state_kind = str(row.get("state_space_kind", ""))
    return bool(mode) or projection.endswith("_oracle_state") or state_kind == "fixed_numeric_vector"


def _is_projection_row(row: dict) -> bool:
    if not _is_learned_row(row):
        return False
    return str(row.get("projection_kind", "")) == "mergeable_projection"


_PAPERPLOT_MODULE = None


def _paperplot_module():
    global _PAPERPLOT_MODULE
    if _PAPERPLOT_MODULE is False:
        return None
    if _PAPERPLOT_MODULE is not None:
        return _PAPERPLOT_MODULE
    for parent in Path(__file__).resolve().parents:
        scripts_dir = parent / "paper" / "ctreepo" / "scripts"
        if (scripts_dir / "paperplot.py").exists():
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            try:
                import paperplot  # type: ignore

                paperplot.rcparams()
                _PAPERPLOT_MODULE = paperplot
                return paperplot
            except Exception:
                break
    _PAPERPLOT_MODULE = False
    return None


def _save_paper_figure(fig, output_stem: Path) -> None:
    paperplot = _paperplot_module()
    stem = output_stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    if paperplot is not None:
        paperplot.save(fig, stem)
    else:
        fig.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def _axis_key(rows: Sequence[dict]) -> str:
    if any(str(r.get("leaf_axis", "n_leaves")) == "leaf_size" for r in rows):
        return "leaf_size"
    if any(int(float(r.get("n_leaves", -1))) < 0 for r in rows):
        return "leaf_size"
    return "n_leaves"


def _axis_values(rows: Sequence[dict], x_key: str) -> list[int]:
    out: list[int] = []
    for row in rows:
        try:
            value = int(float(row.get(x_key, -1)))
        except Exception:
            continue
        if value > 0 and value not in out:
            out.append(value)
    return sorted(out)


def _axis_label(x_key: str) -> str:
    return "leaf size (tokens)" if x_key == "leaf_size" else "leaf count L"


def _leaf_display(row: dict) -> object:
    if str(row.get("leaf_axis", "n_leaves")) == "leaf_size":
        return row.get("leaf_size", "--")
    return row.get("n_leaves", "--")


def _capacity_label(row: dict) -> str:
    cap = str(row.get("capacity_label", ""))
    if str(row.get("family")) == "distinct":
        p = row.get("distinct_lg_k")
        if p is not None:
            return f"{cap} (p={p})"
    return cap


def _preferred_capacity(rows: Sequence[dict]) -> str | None:
    capacities = sorted({str(r.get("capacity_label", "")) for r in rows}, key=_capacity_x)
    if not capacities:
        return None
    return "large" if "large" in capacities else capacities[-1]


def _best_row_axis(
    rows: Sequence[dict],
    *,
    method: str,
    family: str,
    query: str,
    capacity: str | None,
    x_key: str,
    x_value: int,
    metric: str,
    row_filter=None,
) -> dict | None:
    candidates: list[tuple[float, dict]] = []
    for row in rows:
        if _method_group(row) != method:
            continue
        if str(row.get("family")) != family or str(row.get("query")) != query:
            continue
        if capacity is not None and str(row.get("capacity_label")) != capacity:
            continue
        try:
            if int(float(row.get(x_key, -1))) != int(x_value):
                continue
        except Exception:
            continue
        if row_filter is not None and not row_filter(row):
            continue
        value = _finite_float(row, metric)
        if np.isfinite(value):
            candidates.append((value, row))
    return min(candidates, key=lambda x: x[0])[1] if candidates else None


def _format_axis_number(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    if value >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}"


def _mean_by_x(rows: Sequence[dict], x_key: str, value_key: str, x_value: int) -> float:
    vals: list[float] = []
    for row in rows:
        try:
            if int(float(row.get(x_key, -1))) != int(x_value):
                continue
            value = float(row.get(value_key, np.nan))
        except Exception:
            continue
        if np.isfinite(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else float("nan")


def _setup_leaf_axis(
    ax,
    x_values: Sequence[int],
    x_key: str,
    rows: Sequence[dict] | None = None,
) -> None:
    if len(x_values) >= 2 and all(v > 0 for v in x_values):
        ax.set_xscale("log", base=2)
    ax.set_xticks(list(x_values))
    ax.set_xticklabels([str(v) for v in x_values])
    ax.set_xlabel(_axis_label(x_key))
    ax.grid(alpha=0.25)
    if rows is None or not x_values:
        return
    top_key = "leaf_count_mean" if x_key == "leaf_size" else "tokens_per_leaf_mean"
    top_label = "leaves/doc" if x_key == "leaf_size" else "tokens/leaf"
    top_values = [_mean_by_x(rows, x_key, top_key, int(x)) for x in x_values]
    if not any(np.isfinite(v) for v in top_values):
        return
    top = ax.twiny()
    top.set_xscale(ax.get_xscale())
    top.set_xlim(ax.get_xlim())
    top.set_xticks(list(x_values))
    top.set_xticklabels([_format_axis_number(v) for v in top_values])
    top.set_xlabel(top_label)
    top.tick_params(axis="x", direction="in", labelsize=7)


def _bottom_legend(fig, handles, labels, *, ncol: int | None = None) -> None:
    if not handles:
        return
    seen: set[str] = set()
    deduped_handles = []
    deduped_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        deduped_handles.append(handle)
        deduped_labels.append(label)
    if not deduped_handles:
        return
    fig.legend(
        deduped_handles,
        deduped_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=ncol or min(len(deduped_handles), 4),
        frameon=False,
    )


def _finish_figure(
    fig,
    title: str | None = None,
    *,
    top: float = 0.90,
    bottom: float = 0.15,
    left: float = 0.10,
    right: float = 0.98,
    wspace: float = 0.35,
    hspace: float = 0.75,
    title_size: float = 10.0,
) -> None:
    if title:
        fig.suptitle(title, y=0.985, fontsize=title_size)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)


def _plot_leafsize_summary(rows: Sequence[dict], output_stem: Path) -> None:
    _paperplot_module()
    x_key = _axis_key(rows)
    x_values = _axis_values(rows, x_key)
    if not x_values:
        return
    capacity = _preferred_capacity(rows)
    families = sorted({str(r.get("family")) for r in rows})
    ncols = min(3, max(1, len(families)))
    nrows = int(np.ceil(len(families) / ncols))
    mode = _plot_mode(rows)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.0, max(2.8, 2.45 * nrows)),
        squeeze=False,
        constrained_layout=False,
    )
    for panel_idx, (ax, family) in enumerate(zip(axes.ravel(), families)):
        family_rows = [r for r in rows if str(r.get("family")) == family]
        queries = sorted({str(r.get("query")) for r in family_rows})
        methods = ("official",) if mode == "relative_rmse" else PRIMARY_METHOD_GROUPS
        for method in methods:
            xs: list[int] = []
            ys: list[float] = []
            yerr: list[float] = []
            for x_value in x_values:
                if mode == "relative_rmse":
                    candidates = [
                        r
                        for r in family_rows
                        if _method_group(r) == "official"
                        and (capacity is None or str(r.get("capacity_label")) == capacity)
                        and int(float(r.get(x_key, -1))) == int(x_value)
                    ]
                    vals = [_plot_value(r, mode) for r in candidates]
                    errs = [_plot_ci(r, mode) for r in candidates]
                else:
                    vals = []
                    errs = []
                    for query in queries:
                        best = _best_row_axis(
                            rows,
                            method=method,
                            family=family,
                            query=query,
                            capacity=capacity,
                            x_key=x_key,
                            x_value=x_value,
                            metric="relative_rmse_mean",
                        )
                        if best is None:
                            continue
                        vals.append(_plot_value(best, mode))
                        errs.append(_plot_ci(best, mode))
                finite = [v for v in vals if np.isfinite(v)]
                if finite:
                    xs.append(x_value)
                    ys.append(float(np.mean(finite)))
                    err_finite = [v for v in errs if np.isfinite(v)]
                    yerr.append(float(np.mean(err_finite)) if err_finite else 0.0)
            if xs:
                _draw_series(
                    ax,
                    xs,
                    ys,
                    yerr,
                    color=METHOD_COLORS[method],
                    label="Official/oracle mean" if mode == "relative_rmse" else METHOD_LABELS[method],
                    linewidth=1.7,
                    markersize=3.8,
                )
        ax.set_title(family, pad=4)
        ax.set_ylabel(_plot_ylabel(mode) if panel_idx % ncols == 0 else "")
        _setup_leaf_axis(ax, x_values, x_key, family_rows)
    for ax in axes.ravel()[len(families) :]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    _bottom_legend(fig, handles, labels, ncol=len(handles))
    # The caption carries the global title in the paper; keeping the plot body
    # title-free avoids collisions with the dual top axis on compact grids.
    _finish_figure(fig, None, top=0.90, bottom=0.14, left=0.08, wspace=0.42, hspace=0.78)
    _save_paper_figure(fig, output_stem)
    plt.close(fig)


def _plot_leafsize_hll(rows: Sequence[dict], output_stem: Path) -> None:
    _paperplot_module()
    subset = [
        r
        for r in rows
        if str(r.get("family")) == "distinct" and str(r.get("query")) == "cardinality"
    ]
    if not subset:
        return
    x_key = _axis_key(subset)
    x_values = _axis_values(subset, x_key)
    capacities = sorted({str(r.get("capacity_label", "")) for r in subset}, key=_capacity_x)
    if not x_values or not capacities:
        return
    mode = _plot_mode(subset)
    fig, axes = plt.subplots(
        1,
        len(capacities),
        figsize=(7.0, 3.15),
        squeeze=False,
        sharey=True,
        constrained_layout=False,
    )
    method_filters = {
        "official": lambda r: "hll" in str(r.get("sketch", "")).lower(),
        "learned_g": lambda r: str(r.get("learned_target_kind", "")) == "hll_register_space",
        "learned_joint": lambda r: str(r.get("learned_target_kind", "")) == "hll_register_space",
    }
    for panel_idx, (ax, capacity) in enumerate(zip(axes.ravel(), capacities)):
        panel_rows = [r for r in subset if str(r.get("capacity_label")) == capacity]
        label = _capacity_label(panel_rows[0]) if panel_rows else capacity
        for method in PRIMARY_METHOD_GROUPS:
            xs: list[int] = []
            ys: list[float] = []
            yerr: list[float] = []
            for x_value in x_values:
                best = _best_row_axis(
                    subset,
                    method=method,
                    family="distinct",
                    query="cardinality",
                    capacity=capacity,
                    x_key=x_key,
                    x_value=x_value,
                    metric="relative_rmse_mean",
                    row_filter=method_filters[method],
                )
                if best is None:
                    continue
                xs.append(x_value)
                ys.append(_plot_value(best, mode))
                yerr.append(_plot_ci(best, mode))
            if xs:
                _draw_series(
                    ax,
                    xs,
                    ys,
                    yerr,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                    linewidth=1.8,
                    markersize=4.0,
                )
        theory = [
            _finite_float(r, "theoretical_error_mean")
            for r in panel_rows
            if _method_group(r) == "official" and "hll" in str(r.get("sketch", "")).lower()
        ]
        theory = [v for v in theory if np.isfinite(v)]
        if theory:
            ax.axhline(float(np.mean(theory)), color="#555555", linestyle=":", linewidth=1.0)
        ax.text(
            0.03,
            0.93,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )
        ax.set_ylabel(_plot_ylabel(mode) if panel_idx == 0 else "")
        _setup_leaf_axis(ax, x_values, x_key, panel_rows)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    _bottom_legend(fig, handles, labels, ncol=min(3, len(handles)))
    title = (
        r"HLL projection gap: fixed $f^\star$ versus learned state"
        if mode == "projection_gap"
        else "Official HLL over leaf size"
    )
    _finish_figure(fig, title, top=0.78, bottom=0.21, left=0.08, wspace=0.24, title_size=9.5)
    _save_paper_figure(fig, output_stem)
    plt.close(fig)


def _plot_leafsize_family_detail(rows: Sequence[dict], family: str, output_stem: Path) -> None:
    _paperplot_module()
    subset = [r for r in rows if str(r.get("family")) == family]
    if not subset:
        return
    x_key = _axis_key(subset)
    x_values = _axis_values(subset, x_key)
    capacity = _preferred_capacity(subset)
    queries = sorted({str(r.get("query")) for r in subset})
    if not x_values or not queries:
        return
    ncols = min(3, len(queries))
    nrows = int(np.ceil(len(queries) / ncols))
    mode = _plot_mode(subset)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.0, max(2.7, 2.55 * nrows)),
        squeeze=False,
        sharey=False,
        constrained_layout=False,
    )
    for panel_idx, (ax, query) in enumerate(zip(axes.ravel(), queries)):
        query_rows = [r for r in subset if str(r.get("query")) == query]
        if mode == "relative_rmse":
            sketches = sorted(
                {
                    str(r.get("sketch"))
                    for r in query_rows
                    if _method_group(r) == "official"
                    and (capacity is None or str(r.get("capacity_label")) == capacity)
                }
            )
            for sketch_idx, sketch in enumerate(sketches):
                xs: list[int] = []
                ys: list[float] = []
                yerr: list[float] = []
                for x_value in x_values:
                    candidates = [
                        r
                        for r in query_rows
                        if _method_group(r) == "official"
                        and str(r.get("sketch")) == sketch
                        and (capacity is None or str(r.get("capacity_label")) == capacity)
                        and int(float(r.get(x_key, -1))) == int(x_value)
                    ]
                    if not candidates:
                        continue
                    row = candidates[0]
                    xs.append(x_value)
                    ys.append(_plot_value(row, mode))
                    yerr.append(_plot_ci(row, mode))
                if xs:
                    _draw_series(
                        ax,
                        xs,
                        ys,
                        yerr,
                        color=SKETCH_COLORS[sketch_idx % len(SKETCH_COLORS)],
                        label=_pretty_sketch(sketch),
                        linewidth=1.5,
                        markersize=3.4,
                    )
        else:
            for method in PRIMARY_METHOD_GROUPS:
                xs: list[int] = []
                ys: list[float] = []
                yerr: list[float] = []
                for x_value in x_values:
                    best = _best_row_axis(
                        subset,
                        method=method,
                        family=family,
                        query=query,
                        capacity=capacity,
                        x_key=x_key,
                        x_value=x_value,
                        metric="relative_rmse_mean",
                    )
                    if best is None:
                        continue
                    xs.append(x_value)
                    ys.append(_plot_value(best, mode))
                    yerr.append(_plot_ci(best, mode))
                if xs:
                    _draw_series(
                        ax,
                        xs,
                        ys,
                        yerr,
                        color=METHOD_COLORS[method],
                        label=METHOD_LABELS[method],
                        linewidth=1.6,
                        markersize=3.5,
                    )
        ax.set_title(_pretty_query(query), pad=4)
        ax.set_ylabel(_plot_ylabel(mode, family_detail=True) if panel_idx % ncols == 0 else "")
        _setup_leaf_axis(ax, x_values, x_key, query_rows)
    for ax in axes.ravel()[len(queries) :]:
        ax.axis("off")
    handles = []
    labels = []
    for ax in axes.ravel()[: len(queries)]:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)
    _bottom_legend(fig, handles, labels, ncol=min(3, len(handles)))
    _finish_figure(fig, None, top=0.90, bottom=0.27, left=0.08, wspace=0.42, hspace=0.78)
    _save_paper_figure(fig, output_stem)
    plt.close(fig)


def _plot_leafsize_learned_diagnostic(rows: Sequence[dict], output_stem: Path) -> None:
    _paperplot_module()
    subset = [
        r
        for r in rows
        if str(r.get("family")) == "distinct"
        and str(r.get("query")) == "cardinality"
        and (
            _method_group(r) == "official"
            or str(r.get("learned_target_kind", "")) == "hll_register_space"
        )
    ]
    if not subset:
        return
    x_key = _axis_key(subset)
    x_values = _axis_values(subset, x_key)
    capacity = _preferred_capacity(subset)
    if not x_values or capacity is None:
        return
    mode = _plot_mode(subset)
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.55), constrained_layout=False)
    for method in PRIMARY_METHOD_GROUPS:
        xs: list[int] = []
        ys: list[float] = []
        yerr: list[float] = []
        row_filter = (
            (lambda r: "hll" in str(r.get("sketch", "")).lower())
            if method == "official"
            else (lambda r: str(r.get("learned_target_kind", "")) == "hll_register_space")
        )
        for x_value in x_values:
            best = _best_row_axis(
                subset,
                method=method,
                family="distinct",
                query="cardinality",
                capacity=capacity,
                x_key=x_key,
                x_value=x_value,
                metric="relative_rmse_mean",
                row_filter=row_filter,
            )
            if best is None:
                continue
            xs.append(x_value)
            ys.append(_plot_value(best, mode))
            yerr.append(_plot_ci(best, mode))
        if xs:
            _draw_series(
                ax,
                xs,
                ys,
                yerr,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                linewidth=1.7,
                markersize=3.8,
            )
    theory_vals = [
        _finite_float(r, "theoretical_error_mean")
        for r in subset
        if str(r.get("capacity_label")) == capacity
        and _method_group(r) == "official"
        and "hll" in str(r.get("sketch", "")).lower()
    ]
    theory_vals = [v for v in theory_vals if np.isfinite(v)]
    if theory_vals:
        ax.axhline(float(np.mean(theory_vals)), color="#555555", linestyle=":", linewidth=1.0, label="HLL theory floor")
    ax.set_ylabel(_plot_ylabel(mode))
    _setup_leaf_axis(ax, x_values, x_key, subset)
    handles, labels = ax.get_legend_handles_labels()
    _bottom_legend(fig, handles, labels, ncol=min(4, len(handles)))
    ax.text(
        0.03,
        0.93,
        "large-capacity HLL",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    _finish_figure(fig, "Controlled HLL register-state check", top=0.74, bottom=0.24, title_size=9.5)
    _save_paper_figure(fig, output_stem)
    plt.close(fig)


def _plot_paper_summary_figures(rows: Sequence[dict], out_dir: Path) -> None:
    _plot_leafsize_summary(rows, out_dir / "classical_sketches_summary")
    _plot_leafsize_hll(rows, out_dir / "classical_sketches_hll_leaf_size")
    _plot_leafsize_learned_diagnostic(rows, out_dir / "learned_sketch_leaf_size_diagnostic")


def _write_figure_manifest(rows: Sequence[dict], out_dir: Path, output_root: Path) -> None:
    stems = [
        "classical_sketches_summary",
        "classical_sketches_hll_leaf_size",
        "learned_sketch_leaf_size_diagnostic",
    ]
    families = sorted({str(r.get("family")) for r in rows if str(r.get("family", ""))})
    stems.extend(f"classical_sketches_{family}_detail" for family in families)
    leaf_sizes = sorted(
        {
            int(float(r.get("leaf_size", -1)))
            for r in rows
            if str(r.get("leaf_axis", "n_leaves")) == "leaf_size"
            and int(float(r.get("leaf_size", -1))) > 0
        }
    )
    payload = {
        "data_root": str(Path(output_root).resolve()),
        "seed_count": int(max([int(r.get("n_runs", 0) or 0) for r in rows] or [0])),
        "leaf_sizes": leaf_sizes,
        "include_learned": bool(_has_learned_rows(rows)),
        "learned_rows": int(sum(1 for r in rows if str(r.get("implementation_status")) == "learned_empirical")),
        "figures": [
            {
                "stem": stem,
                "pdf": str((out_dir / f"{stem}.pdf").resolve()),
                "png": str((out_dir / f"{stem}.png").resolve()),
                "pdf_exists": bool((out_dir / f"{stem}.pdf").exists()),
                "png_exists": bool((out_dir / f"{stem}.png").exists()),
            }
            for stem in stems
        ],
    }
    atomic_write_text(out_dir / "classical_sketches_figure_manifest.json", dump_json(payload))


def _markdown(rows: Sequence[dict]) -> str:
    def fmt(value: object) -> str:
        try:
            v = float(value)
        except Exception:
            return "—"
        return f"{v:.4g}" if np.isfinite(v) else "—"

    has_rate_axis = any(
        abs(_finite_float(r, "learned_root_query_rate") - 1.0) > 1e-12
        or abs(_finite_float(r, "learned_leaf_query_rate") - 1.0) > 1e-12
        or abs(_finite_float(r, "learned_internal_query_rate") - 1.0) > 1e-12
        for r in rows
    )
    has_sampling_policy_axis = any(
        str(r.get("learned_supervision_sampling_policy", "separate_axes")) != "separate_axes"
        for r in rows
    )
    header_cells = [
        "family",
        "sketch",
        "query",
        "capacity",
        "leaf",
    ]
    rule_cells = ["---", "---:", "---:", "---:", "---:"]
    if has_rate_axis:
        header_cells.extend(["root R", "leaf R", "internal R"])
        rule_cells.extend(["---:", "---:", "---:"])
    if has_sampling_policy_axis:
        header_cells.append("sampling")
        rule_cells.append("---:")
    header_cells.extend(
        [
            "implementation",
            "formal",
            "rel/rank RMSE",
            "official floor",
            "distance",
            "2σ coverage",
            "schedule spread",
            "memory bytes",
        ]
    )
    rule_cells.extend(["---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"])
    lines = [
        "# Classical Mergeable Sketch Comparison",
        "",
        (
            f"Learned exact-state recovery rows: {sum(1 for r in rows if _is_exact_state_row(r))}. "
            f"Learned mergeable-projection diagnostic rows: {sum(1 for r in rows if _is_projection_row(r))}. "
            "Exact-state rows use exposed deterministic vector states; projection rows are best-fit "
            "mergeable summaries under the empirical loss, not claims about opaque library internals."
        ),
        "",
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(rule_cells) + " |",
    ]
    for r in rows:
        rate_values = (
            f" | {fmt(r.get('learned_root_query_rate', 1.0))}"
            f" | {fmt(r.get('learned_leaf_query_rate', 1.0))}"
            f" | {fmt(r.get('learned_internal_query_rate', 1.0))}"
            if has_rate_axis
            else ""
        )
        policy_value = (
            f" | {r.get('learned_supervision_sampling_policy', 'separate_axes')}"
            if has_sampling_policy_axis
            else ""
        )
        lines.append(
            "| {family} | {sketch} | {query} | {capacity} | {n_leaves}{rate_values}{policy_value} | {status} | {formal} | {rel} | {floor} | {dist} | {coverage} | {spread} | {mem} |".format(
                family=r["family"],
                sketch=r["sketch"],
                query=r["query"],
                capacity=r.get("capacity_label", "single"),
                n_leaves=_leaf_display(r),
                rate_values=rate_values,
                policy_value=policy_value,
                status=r["implementation_status"],
                formal=r.get("formal_status", ""),
                rel=fmt(r.get("relative_rmse_mean", float("nan"))),
                floor=fmt(r.get("official_floor_rel_rmse_mean", float("nan"))),
                dist=fmt(r.get("distance_to_official_floor_mean", float("nan"))),
                coverage=fmt(r.get("bound_coverage_2sigma_mean", float("nan"))),
                spread=fmt(r.get("schedule_spread_mean", float("nan"))),
                mem=fmt(r.get("memory_bytes_mean", float("nan"))),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _latex_escape(value: object) -> str:
    return str(value).replace("_", "\\_")


def _latex_table(rows: Sequence[dict]) -> str:
    def fmt(value: object) -> str:
        try:
            v = float(value)
        except Exception:
            return "--"
        return f"{v:.4g}" if np.isfinite(v) else "--"

    lines = [
        "% Auto-generated by treepo.bench.reports.classical_sketches; do not edit.",
        "\\begin{tabular}{lllrlrrrrrrr}",
        "\\toprule",
        "family & sketch & query & cap. & leaf & rel RMSE & 95\\% CI & floor & dist. & spread & bytes \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            "{family} & {sketch} & {query} & {capacity} & {n_leaves} & {rel} & {ci} & {floor} & {dist} & {spread} & {mem} \\\\".format(
                family=_latex_escape(r["family"]),
                sketch=_latex_escape(r["sketch"]),
                query=_latex_escape(r["query"]),
                capacity=_latex_escape(r.get("capacity_label", "")),
                n_leaves=_leaf_display(r),
                rel=fmt(r.get("relative_rmse_mean")),
                ci=fmt(r.get("relative_rmse_ci95")),
                floor=fmt(r.get("official_floor_rel_rmse_mean")),
                dist=fmt(r.get("distance_to_official_floor_mean")),
                spread=fmt(r.get("schedule_spread_mean")),
                mem=fmt(r.get("memory_bytes_mean")),
            )
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def _best_compact_rows(rows: Sequence[dict]) -> List[dict]:
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for row in rows:
        groups.setdefault((str(row.get("family")), str(row.get("query"))), []).append(row)
    out: List[dict] = []
    for (family, query), grows in sorted(groups.items()):
        preferred = [r for r in grows if str(r.get("capacity_label")) == "large"] or list(grows)

        def best(
            status: str,
            sketch_prefix: str | None = None,
            *,
            row_filter=None,
        ) -> dict | None:
            candidates = [r for r in preferred if str(r.get("implementation_status")) == status]
            if sketch_prefix is not None:
                candidates = [r for r in candidates if str(r.get("sketch", "")).startswith(sketch_prefix)]
            if row_filter is not None:
                candidates = [r for r in candidates if row_filter(r)]
            finite = []
            for r in candidates:
                try:
                    value = float(r.get("relative_rmse_mean", np.nan))
                except Exception:
                    continue
                if np.isfinite(value):
                    finite.append((value, r))
            return min(finite, key=lambda x: x[0])[1] if finite else None

        official = best("official_empirical")
        # Use _method_group so legacy split names and current joint names land
        # in the same report bucket while exact variants remain in metadata.
        def best_in_group(group: str):
            return best(
                "learned_empirical",
                None,
                row_filter=lambda r: _method_group(r) == group,
            )

        learned_f = best_in_group("learned_f")
        learned_g = best_in_group("learned_g")
        # Joint = any multi-letter variant. The aggregate already keys by
        # learned_variant, so fg, gf, fgf, etc. live as distinct rows; this
        # picks the best one for the joint column. The chosen row's
        # `learned_variant` field is reported as `joint_variant` so the table
        # records *which* schedule won.
        learned_joint = best_in_group("learned_joint")
        learned_any = best("learned_empirical")
        if all(x is None for x in (official, learned_f, learned_g, learned_joint)):
            continue
        out.append(
            {
                "family": family,
                "query": query,
                "official_sketch": official.get("sketch", "--") if official else "--",
                "official_rel_rmse": official.get("relative_rmse_mean", np.nan) if official else np.nan,
                "official_L": _leaf_display(official) if official else "--",
                "learned_sketch": learned_any.get("sketch", "--") if learned_any else "--",
                "learned_rel_rmse": learned_any.get("relative_rmse_mean", np.nan) if learned_any else np.nan,
                "learned_L": _leaf_display(learned_any) if learned_any else "--",
                "learned_distance": learned_any.get("distance_to_official_floor_mean", np.nan) if learned_any else np.nan,
                "learned_f_sketch": learned_f.get("sketch", "--") if learned_f else "--",
                "learned_f_rel_rmse": learned_f.get("relative_rmse_mean", np.nan) if learned_f else np.nan,
                "learned_f_L": _leaf_display(learned_f) if learned_f else "--",
                "learned_f_distance": learned_f.get("distance_to_official_floor_mean", np.nan) if learned_f else np.nan,
                "learned_g_sketch": learned_g.get("sketch", "--") if learned_g else "--",
                "learned_g_rel_rmse": learned_g.get("relative_rmse_mean", np.nan) if learned_g else np.nan,
                "learned_g_L": _leaf_display(learned_g) if learned_g else "--",
                "learned_g_distance": learned_g.get("distance_to_official_floor_mean", np.nan) if learned_g else np.nan,
                "learned_joint_sketch": learned_joint.get("sketch", "--") if learned_joint else "--",
                "learned_joint_variant": str(learned_joint.get("learned_variant", "--")) if learned_joint else "--",
                "learned_joint_rel_rmse": learned_joint.get("relative_rmse_mean", np.nan) if learned_joint else np.nan,
                "learned_joint_L": _leaf_display(learned_joint) if learned_joint else "--",
                "learned_joint_distance": learned_joint.get("distance_to_official_floor_mean", np.nan) if learned_joint else np.nan,
            }
        )
    return out


def _compact_markdown(rows: Sequence[dict]) -> str:
    def fmt(value: object) -> str:
        try:
            v = float(value)
        except Exception:
            return "—"
        return f"{v:.4g}" if np.isfinite(v) else "—"

    lines = [
        "# Classical Sketch Compact Learned Overlay",
        "",
        "| family | query | best official | official RMSE | learned f | f RMSE | learned g | g RMSE | learned joint (best) | joint variant | joint RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _best_compact_rows(rows):
        lines.append(
            "| {family} | {query} | {official} (L={official_l}) | {official_rmse} | "
            "{learned_f} (L={learned_f_l}) | {learned_f_rmse} | "
            "{learned_g} (L={learned_g_l}) | {learned_g_rmse} | "
            "{learned_joint} (L={learned_joint_l}) | {joint_variant} | {learned_joint_rmse} |".format(
                family=row["family"],
                query=row["query"],
                official=row["official_sketch"],
                official_l=row["official_L"],
                official_rmse=fmt(row["official_rel_rmse"]),
                learned_f=row["learned_f_sketch"],
                learned_f_l=row["learned_f_L"],
                learned_f_rmse=fmt(row["learned_f_rel_rmse"]),
                learned_g=row["learned_g_sketch"],
                learned_g_l=row["learned_g_L"],
                learned_g_rmse=fmt(row["learned_g_rel_rmse"]),
                learned_joint=row["learned_joint_sketch"],
                learned_joint_l=row["learned_joint_L"],
                joint_variant=row["learned_joint_variant"],
                learned_joint_rmse=fmt(row["learned_joint_rel_rmse"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _compact_latex(rows: Sequence[dict]) -> str:
    def fmt(value: object) -> str:
        try:
            v = float(value)
        except Exception:
            return "--"
        return f"{v:.4g}" if np.isfinite(v) else "--"

    lines = [
        "% Auto-generated by treepo.bench.reports.classical_sketches; do not edit.",
        "\\begin{tabular}{lllrrrrrrrr}",
        "\\toprule",
        "family & query & best official & official & "
        "learned $f$ & $f$ RMSE & learned $g$ & $g$ RMSE & "
        "learned joint (best) & joint variant & joint RMSE \\\\",
        "\\midrule",
    ]
    for row in _best_compact_rows(rows):
        lines.append(
            "{family} & {query} & {official} & {official_rmse} & "
            "{learned_f} & {learned_f_rmse} & {learned_g} & {learned_g_rmse} & "
            "{learned_joint} & {joint_variant} & {learned_joint_rmse} \\\\".format(
                family=_latex_escape(row["family"]),
                query=_latex_escape(row["query"]),
                official=_latex_escape(row["official_sketch"]),
                official_rmse=fmt(row["official_rel_rmse"]),
                learned_f=_latex_escape(row["learned_f_sketch"]),
                learned_f_rmse=fmt(row["learned_f_rel_rmse"]),
                learned_g=_latex_escape(row["learned_g_sketch"]),
                learned_g_rmse=fmt(row["learned_g_rel_rmse"]),
                learned_joint=_latex_escape(row["learned_joint_sketch"]),
                joint_variant=_latex_escape(row["learned_joint_variant"]),
                learned_joint_rmse=fmt(row["learned_joint_rel_rmse"]),
            )
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def _plot_frequency_family(rows: Sequence[dict], output: Path) -> None:
    subset = [
        r
        for r in rows
        if str(r.get("family")) == "frequency"
        and str(r.get("query")) == "top5_point_frequency"
        and str(r.get("sketch")) in {"count_min_datasketches", "frequent_strings_datasketches"}
        and _method_group(r) == "official"
    ]
    if not subset:
        return

    leaves = sorted({int(r.get("n_leaves", -1)) for r in subset if int(r.get("n_leaves", -1)) > 0})
    capacities = sorted({str(r.get("capacity_label", "")) for r in subset}, key=_capacity_x)
    sketches = [
        ("count_min_datasketches", "Count-Min point frequency"),
        ("frequent_strings_datasketches", "Frequent Items heavy hitters"),
    ]

    def capacity_label(row: dict, sketch: str) -> str:
        if sketch == "count_min_datasketches":
            hashes = row.get("cms_num_hashes")
            buckets = row.get("cms_num_buckets")
            if hashes is not None and buckets is not None:
                return f"{hashes}x{buckets} counters"
        if sketch == "frequent_strings_datasketches":
            lg_map = row.get("frequent_lg_max_map_size")
            if lg_map is not None:
                return f"map 2^{lg_map}"
        return str(row.get("capacity_label", ""))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), squeeze=False, constrained_layout=True)
    for ax, (sketch, title) in zip(axes.ravel(), sketches):
        ymax = 0.0
        final_points: list[tuple[float, float, str, str]] = []
        for capacity in capacities:
            xs: list[int] = []
            ys: list[float] = []
            yerr: list[float] = []
            series_label: str | None = None
            for n_leaves in leaves:
                candidates = [
                    r
                    for r in subset
                    if str(r.get("sketch")) == sketch
                    and str(r.get("capacity_label")) == capacity
                    and int(r.get("n_leaves", -1)) == n_leaves
                ]
                if not candidates:
                    continue
                row = candidates[0]
                if series_label is None:
                    series_label = capacity_label(row, sketch)
                xs.append(n_leaves)
                y = _finite_float(row, "relative_rmse_mean")
                err = _finite_float(row, "relative_rmse_ci95")
                ys.append(y)
                yerr.append(0.0 if not np.isfinite(err) else err)
                if np.isfinite(y):
                    ymax = max(ymax, y + (0.0 if not np.isfinite(err) else err))
            if not xs:
                continue
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                color=CAPACITY_COLORS.get(capacity, "#555555"),
                marker="o",
                linewidth=2.0,
                markersize=4.8,
                capsize=2.6,
            )
            final_points.append(
                (
                    float(xs[-1]),
                    float(ys[-1]),
                    series_label or str(capacity),
                    CAPACITY_COLORS.get(capacity, "#555555"),
                )
            )
        ax.set_title(title, fontsize=11)
        ax.set_xscale("log", base=2)
        ax.set_xlim(left=float(leaves[0]) / 1.15, right=float(leaves[-1]) * 1.75)
        ax.set_xticks(leaves)
        ax.set_xticklabels([str(x) for x in leaves])
        ax.set_xlabel("leaf count L")
        ax.set_ylabel("relative RMSE")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
        ax.set_ylim(bottom=0.0, top=max(0.01, ymax * 1.18))
        ax.grid(alpha=0.24)
        close_to_zero = [p for p in final_points if abs(p[1]) < 5e-4]
        zero_offsets: dict[str, int] = {}
        if len(close_to_zero) > 1:
            offsets = [8, 20, 32, 44]
            for idx, point in enumerate(close_to_zero):
                zero_offsets[point[2]] = offsets[idx % len(offsets)]
        for x, y, label, color in final_points:
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(7, zero_offsets.get(label, 0)),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=8.2,
                color=color,
                clip_on=False,
            )
    fig.suptitle(
        "Official Frequency Sketches: Error by Capacity and Tree Granularity",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=210, bbox_inches="tight")
    plt.close(fig)


def _plot_family(rows: Sequence[dict], family: str, output: Path) -> None:
    if family == "frequency":
        _plot_frequency_family(rows, output)
        return
    subset = [r for r in rows if str(r.get("family")) == family]
    if not subset:
        return
    leaf_counts = sorted({int(r.get("n_leaves", -1)) for r in subset})
    capacities = sorted({str(r.get("capacity_label", "single")) for r in subset}, key=_capacity_x)
    cap_x = np.arange(len(capacities), dtype=np.float64)
    cap_pos = {cap: idx for idx, cap in enumerate(capacities)}
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for row in subset:
        groups.setdefault((str(row.get("sketch")), str(row.get("query"))), []).append(row)

    fig, axes = plt.subplots(
        1,
        max(1, len(leaf_counts)),
        figsize=(4.2 * max(1, len(leaf_counts)), 3.4),
        sharey=True,
        squeeze=False,
    )
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf", "#bcbd22"]
    for panel_idx, n_leaves in enumerate(leaf_counts):
        ax = axes[0, panel_idx]
        for idx, ((sketch, query), grows) in enumerate(sorted(groups.items())):
            by_cap = {str(r.get("capacity_label", "single")): r for r in grows if int(r.get("n_leaves", -1)) == n_leaves}
            xs: List[float] = []
            ys: List[float] = []
            yerr: List[float] = []
            for cap in capacities:
                row = by_cap.get(cap)
                if row is None:
                    continue
                xs.append(float(cap_pos[cap]))
                ys.append(float(row.get("relative_rmse_mean", np.nan)))
                yerr.append(float(row.get("relative_rmse_ci95", 0.0)))
            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerr,
                    label=_series_label(sketch, query),
                    color=palette[idx % len(palette)],
                    linestyle="-" if "official" in str(grows[0].get("implementation_status")) else "--",
                    marker="o",
                    linewidth=1.2,
                    markersize=3.0,
                    capsize=2.0,
                )
        floors = []
        for cap in capacities:
            vals = [
                float(r.get("official_floor_rel_rmse_mean", np.nan))
                for r in subset
                if int(r.get("n_leaves", -1)) == n_leaves and str(r.get("capacity_label", "single")) == cap
            ]
            vals = [v for v in vals if np.isfinite(v)]
            floors.append(min(vals) if vals else np.nan)
        if any(np.isfinite(floors)):
            ax.plot(cap_x, floors, linestyle=":", color="black", linewidth=1.0, label="official floor")
        ax.set_xticks(cap_x)
        ax.set_xticklabels(capacities, rotation=20, ha="right")
        ax.set_xlabel("capacity preset")
        if panel_idx == 0:
            ax.set_ylabel("relative RMSE / rank RMSE")
        ax.set_title(f"L = {n_leaves}")
        ax.grid(True, alpha=0.3)
        if panel_idx == len(leaf_counts) - 1:
            ax.legend(fontsize=6, loc="best", frameon=False)
    fig.suptitle(f"Classical sketch grid: {family}", fontsize=11)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_family_figures(rows: Sequence[dict], out_dir: Path) -> None:
    for family in sorted({str(r.get("family")) for r in rows}):
        _plot_leafsize_family_detail(rows, family, out_dir / f"classical_sketches_{family}_detail")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate the classical-sketch comparison report.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--tables-dir", type=Path, default=Path("paper/ctreepo/tables"))
    p.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.output_root) / "reports" / "classical_sketches"
    rows = _scan_rows(Path(args.output_root))
    agg = _aggregate(rows)
    exact_state_rows = [r for r in agg if _is_exact_state_row(r)]
    projection_rows = [r for r in agg if _is_projection_row(r)]
    write_csv_rows(out_dir / "classical_sketches_aggregate.csv", agg)
    atomic_write_text(out_dir / "classical_sketches_aggregate.json", dump_json({"rows": agg}))
    write_csv_rows(out_dir / "classical_sketches_exact_state_recovery.csv", exact_state_rows)
    atomic_write_text(
        out_dir / "classical_sketches_exact_state_recovery.json",
        dump_json({"rows": exact_state_rows}),
    )
    write_csv_rows(out_dir / "classical_sketches_projection_diagnostics.csv", projection_rows)
    atomic_write_text(
        out_dir / "classical_sketches_projection_diagnostics.json",
        dump_json({"rows": projection_rows}),
    )
    atomic_write_text(out_dir / "classical_sketches_report.md", _markdown(agg))
    atomic_write_text(out_dir / "classical_sketches_grid.md", _markdown(agg))
    atomic_write_text(out_dir / "classical_sketches_grid.tex", _latex_table(agg))
    atomic_write_text(out_dir / "classical_sketches_compact.md", _compact_markdown(agg))
    atomic_write_text(out_dir / "classical_sketches_compact.tex", _compact_latex(agg))
    if args.tables_dir is not None:
        tables_dir = Path(args.tables_dir)
        atomic_write_text(tables_dir / "classical_sketches_grid.md", _markdown(agg))
        atomic_write_text(tables_dir / "classical_sketches_grid.tex", _latex_table(agg))
        atomic_write_text(tables_dir / "classical_sketches_compact.md", _compact_markdown(agg))
        atomic_write_text(tables_dir / "classical_sketches_compact.tex", _compact_latex(agg))
        write_csv_rows(
            tables_dir / "classical_sketches_exact_state_recovery.csv",
            exact_state_rows,
        )
        write_csv_rows(
            tables_dir / "classical_sketches_projection_diagnostics.csv",
            projection_rows,
        )
    _plot_paper_summary_figures(agg, out_dir)
    _plot_family_figures(agg, out_dir)
    _write_figure_manifest(agg, out_dir, Path(args.output_root))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
