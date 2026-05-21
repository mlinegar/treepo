"""Family-agnostic plotting functions for unified reports."""
from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from treepo._research.ctreepo.sim.report.family_config import FamilyReportConfig
from treepo._research.ctreepo.sim.report.pdf_utils import safe_float, safe_mean, safe_sem


# ── heatmap ──────────────────────────────────────────────────────────────


def plot_heatmap(
    rows: Sequence[dict],
    *,
    row_field: str,
    col_field: str,
    value_key: str,
    row_label: str,
    col_label: str,
    title: str,
    output_path: Path,
    cmap: str,
    fmt: str = ".2f",
) -> None:
    """Generic heatmap: row_field × col_field → value_key."""
    row_vals = sorted({rows[i][row_field] for i in range(len(rows))})
    col_vals = sorted({rows[i][col_field] for i in range(len(rows))})
    if not row_vals or not col_vals:
        return

    matrix = np.full((len(row_vals), len(col_vals)), np.nan, dtype=np.float64)
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            match = [
                row for row in rows
                if row[row_field] == rv and (
                    row[col_field] == cv
                    if not isinstance(cv, float)
                    else math.isclose(float(row[col_field]), float(cv), abs_tol=1e-12)
                )
            ]
            if match:
                matrix[i, j] = safe_float(match[0][value_key])

    fig, ax = plt.subplots(figsize=(1.8 + 1.55 * len(col_vals), 1.8 + 1.1 * len(row_vals)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)

    # Column labels
    col_labels = []
    for v in col_vals:
        if isinstance(v, float):
            col_labels.append(f"{100.0 * v:.1f}%" if v < 0.1 else f"{100.0 * v:.0f}%")
        else:
            col_labels.append(str(v))
    ax.set_xticks(range(len(col_vals)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_vals)))
    ax.set_yticklabels([str(v) for v in row_vals])
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    ax.set_title(title)

    for i in range(len(row_vals)):
        for j in range(len(col_vals)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, format(float(matrix[i, j]), fmt), ha="center", va="center", fontsize=9, color="#111111")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ── ablation bar chart ───────────────────────────────────────────────────


def plot_ablation_bar_chart(
    aggregated_rows: Sequence[dict],
    *,
    family: FamilyReportConfig,
    output_path: Path,
) -> None:
    """Dual-panel figure: downstream PrimGain + law pass rates by package."""
    packages_order = [p for p in family.valid_law_packages]
    pkg_stats: Dict[str, dict] = {}
    for pkg in packages_order:
        pkg_rows = [row for row in aggregated_rows if str(row.get("law_package", "")) == pkg]
        if not pkg_rows:
            continue
        n_total = sum(int(row.get("n_runs", 1)) for row in pkg_rows)
        prim_gains = [1.0 - float(row["root_ratio"]) for row in pkg_rows]
        pkg_stats[pkg] = {
            "root_ratio": float(fmean(float(row["root_ratio"]) for row in pkg_rows)),
            "prim_gain": float(fmean(prim_gains)),
            "prim_gain_sem": float(safe_sem(prim_gains)),
            "c1_pass": float(fmean(float(row.get("c1_pass_rate", 0.0)) for row in pkg_rows)),
            "c2_pass": float(fmean(float(row.get("c2_pass_rate", 0.0)) for row in pkg_rows)),
            "c3_pass": float(fmean(float(row.get("c3_pass_rate", 0.0)) for row in pkg_rows)),
            "n_configs": len(pkg_rows),
            "n_runs": n_total,
        }

    present = [pkg for pkg in packages_order if pkg in pkg_stats]
    if len(present) < 2:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.8), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(top=0.88, bottom=0.18, left=0.08, right=0.96, wspace=0.32)

    # Left panel: PrimGain by package
    x = np.arange(len(present), dtype=np.float64)
    gains = [float(pkg_stats[pkg]["prim_gain"]) for pkg in present]
    sems = [float(pkg_stats[pkg]["prim_gain_sem"]) for pkg in present]
    colors = [family.package_colors.get(pkg, "#333333") for pkg in present]
    bars = ax1.bar(
        x, [100.0 * g for g in gains], yerr=[100.0 * s for s in sems],
        color=colors, width=0.65, edgecolor="#222222", linewidth=0.8,
        capsize=4, error_kw={"linewidth": 1.2},
    )
    ax1.axhline(0.0, color="#555555", linewidth=1.2, linestyle="--", zorder=0)
    ax1.axhline(10.0, color="#2a9d8f", linewidth=0.9, linestyle=":", alpha=0.7, zorder=0, label="10% pass threshold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([family.package_labels.get(pkg, pkg) for pkg in present], fontsize=9.5)
    ax1.set_ylabel(f"Downstream gain (%)\n(PrimGain = 1 − {family.primary_metric_label} ratio)", fontsize=10.5)
    ax1.set_title(f"Downstream {family.primary_metric_label} improvement by law package", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="y", linewidth=0.7, alpha=0.3)
    ax1.legend(loc="upper left", frameon=False, fontsize=9)
    for bar_obj, gain in zip(bars, gains):
        y_pos = bar_obj.get_height()
        ax1.text(
            bar_obj.get_x() + bar_obj.get_width() / 2.0,
            y_pos + (1.5 if y_pos >= 0 else -3.0),
            f"{100.0 * gain:+.1f}%",
            ha="center", va="bottom" if y_pos >= 0 else "top",
            fontsize=9, fontweight="bold",
        )

    # Right panel: law pass rates
    c1_rates = [100.0 * float(pkg_stats[pkg]["c1_pass"]) for pkg in present]
    c2_rates = [100.0 * float(pkg_stats[pkg]["c2_pass"]) for pkg in present]
    c3_rates = [100.0 * float(pkg_stats[pkg]["c3_pass"]) for pkg in present]
    width = 0.22
    ax2.bar(x - width, c1_rates, width=width, color=family.law_colors["c1"], label="C1 (leaf)", edgecolor="#222222", linewidth=0.5)
    ax2.bar(x, c2_rates, width=width, color=family.law_colors["c2"], label="C2 (resum.)", edgecolor="#222222", linewidth=0.5)
    ax2.bar(x + width, c3_rates, width=width, color=family.law_colors["c3"], label="C3 (merge)", edgecolor="#222222", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([family.package_labels.get(pkg, pkg) for pkg in present], fontsize=9.5)
    ax2.set_ylabel("Law pass rate (%)", fontsize=10.5)
    ax2.set_ylim(0, 115)
    ax2.set_title("Local law satisfaction by package", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", linewidth=0.7, alpha=0.3)
    ax2.legend(loc="upper left", frameon=False, fontsize=9)

    fig.suptitle(
        "Ablation: which local laws improve downstream error?",
        fontsize=14, fontweight="bold", y=0.96,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ── mechanism pareto ─────────────────────────────────────────────────────


def plot_mechanism_pareto(
    rows: Sequence[dict],
    *,
    family: FamilyReportConfig,
    output_path: Path,
) -> None:
    """Scatter: primary metric vs C1+C2+C3 bundle score, coloured by package."""
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    packages = sorted({str(row["law_package"]) for row in rows})
    for package in packages:
        subset = [row for row in rows if str(row["law_package"]) == package]
        ax.scatter(
            [float(row["test_primary"]) for row in subset],
            [float(row["test_bundle_score"]) for row in subset],
            color=family.package_colors.get(package, "#333333"),
            label=package, s=70, alpha=0.85,
        )
    ax.set_xlabel(f"test {family.primary_metric_label} (normalised)")
    ax.set_ylabel("test C1+C2+C3 score")
    ax.set_title("Mechanism Pareto")
    ax.grid(True, linewidth=0.8, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ── exact-family counterexamples ─────────────────────────────────────────


def plot_exact_family_counterexamples(
    rows: Sequence[dict],
    *,
    output_path: Path,
) -> None:
    """Bar chart: C1/C2/C3/Root for each exact family."""
    families = sorted({str(row["exact_family"]) for row in rows})
    metrics = ["test_c1", "test_c2", "test_c3", "test_primary"]
    labels = ["C1", "C2", "C3", "Root"]
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    x = np.arange(len(families), dtype=np.float64)
    width = 0.18
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        vals = []
        for fam in families:
            fam_rows = [row for row in rows if str(row["exact_family"]) == fam]
            vals.append(safe_mean([row[metric] for row in fam_rows]))
        ax.bar(x + width * (idx - 1.5), vals, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel("normalised error")
    ax.set_title("Exact-family counterexamples")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Learnability plotting helpers
# ═══════════════════════════════════════════════════════════════════════════

TRAIN_DOC_BASE_COLORS = [
    "#1d3557", "#d17c00", "#2a9d8f", "#c44e52", "#6c5ce7", "#7f5539",
]
SCW_LINESTYLES = [
    "solid", (0, (5, 2)), (0, (3, 2, 1.2, 2)), (0, (1.5, 1.5)), (0, (7, 2, 1.5, 2)),
]
SCW_MARKERS = ["o", "s", "^", "D", "P", "X"]
AX_FACE = "#f7f6f2"
GRID_COLOR = "#d8d2c7"

CapacityKey = tuple  # (state_dim, hidden_dim, n_epochs, feature_mode)


def _build_color_map(vals: Sequence, base_colors: Sequence[str] = TRAIN_DOC_BASE_COLORS) -> dict:
    if len(vals) <= len(base_colors):
        return {v: base_colors[i] for i, v in enumerate(vals)}
    cmap = plt.get_cmap("cividis")
    return {v: cmap(i / max(1, len(vals) - 1)) for i, v in enumerate(vals)}


def _build_style_map(vals: Sequence, styles: Sequence) -> dict:
    return {v: styles[i % len(styles)] for i, v in enumerate(sorted(float(x) for x in vals))}


def _capacity_key(row: dict) -> CapacityKey:
    return (int(row.get("state_dim", 0)), int(row.get("hidden_dim", 0)),
            int(row.get("n_epochs", 0)), str(row.get("feature_mode", "")))


def _capacity_label(cap: CapacityKey, *, show_fm: bool = True) -> str:
    parts = [f"state_dim={cap[0]}", f"hidden_dim={cap[1]}", f"epochs={cap[2]}"]
    if show_fm:
        parts.append(f"feature_mode={cap[3]}")
    return ", ".join(parts)


def _format_pct(frac: float) -> str:
    return f"{100.0 * float(frac):.0f}%"


def _format_weight(value: float) -> str:
    return f"{float(value):g}"


def _set_sweep_ticks(ax, vals: Sequence[float], label: str = "") -> None:
    ordered = sorted({float(v) for v in vals})
    if not ordered:
        return
    if len(ordered) <= 7:
        ticks = ordered
    else:
        targets = [ordered[0], 0.1, 0.25, 0.5, 0.8, ordered[-1]]
        ticks = []
        for t in targets:
            actual = min(ordered, key=lambda v: abs(v - t))
            if not any(np.isclose(actual, e) for e in ticks):
                ticks.append(float(actual))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])


def _apply_axis_style(ax, *, ylabel: str, xlabel: str = "", zero_line: bool = False) -> None:
    ax.set_facecolor(AX_FACE)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if zero_line:
        ax.axhline(0.0, color="#7a7368", linewidth=1.1, linestyle=(0, (3, 2)), alpha=0.9, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _add_series_legends(
    fig,
    *,
    group1_vals: Sequence,
    group1_colors: dict,
    group1_title: str,
    group2_vals: Sequence,
    group2_linestyles: dict,
    group2_markers: dict,
    group2_title: str,
) -> None:
    from matplotlib.lines import Line2D
    g1_handles = [
        Line2D([0], [0], color=group1_colors[v], linewidth=2.6, label=f"{v}")
        for v in group1_vals
    ]
    g2_handles = [
        Line2D([0], [0], color="#404040", linewidth=2.2,
               linestyle=group2_linestyles[v], marker=group2_markers[v],
               markersize=5.5, label=f"{_format_weight(float(v))}")
        for v in group2_vals
    ]
    leg1 = fig.legend(handles=g1_handles, title=group1_title,
                      loc="upper left", bbox_to_anchor=(0.04, 0.985),
                      ncol=max(1, min(4, len(g1_handles))), frameon=False)
    fig.add_artist(leg1)
    fig.legend(handles=g2_handles, title=group2_title,
               loc="upper right", bbox_to_anchor=(0.98, 0.985),
               ncol=max(1, min(4, len(g2_handles))), frameon=False)


def _series_mean(values: Sequence[float]) -> float:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    return float(arr.mean()) if arr.size > 0 else float("nan")


def _value_matches(lhs: object, rhs: object) -> bool:
    try:
        lhs_f = float(lhs)
        rhs_f = float(rhs)
        if np.isfinite(lhs_f) and np.isfinite(rhs_f):
            return bool(np.isclose(lhs_f, rhs_f))
    except (TypeError, ValueError):
        pass
    return lhs == rhs


def _build_metric_series(
    group_rows: Sequence[dict],
    metric: str,
    sweep_field: str,
) -> tuple:
    xs, ys = [], []
    for sv in sorted({float(r[sweep_field]) for r in group_rows}):
        vals = [float(r[metric]) for r in group_rows if np.isclose(float(r[sweep_field]), sv)]
        center = _series_mean(vals)
        if np.isfinite(center):
            xs.append(float(sv))
            ys.append(center)
    return xs, ys


def _build_gain_series(
    panel_rows: Sequence[dict],
    metric: str,
    sweep_field: str,
    *,
    line_filters: Dict[str, object],
    baseline_field: str,
    baseline_value: object,
) -> tuple:
    line_rows = [
        row for row in panel_rows
        if all(_value_matches(row.get(key), value) for key, value in line_filters.items())
    ]
    sv_vals = sorted({float(r[sweep_field]) for r in line_rows})
    if not sv_vals:
        return [], []
    baseline_filters = dict(line_filters)
    baseline_filters[baseline_field] = baseline_value
    baseline_rows = [
        row for row in panel_rows
        if all(_value_matches(row.get(key), value) for key, value in baseline_filters.items())
    ]
    baseline_by_seed = {
        (float(r[sweep_field]), int(r.get("effective_data_seed", 0)), int(r.get("effective_model_seed", 0))): float(r[metric])
        for r in baseline_rows
    }
    xs, ys = [], []
    for sv in sv_vals:
        gains = []
        for row in line_rows:
            if not np.isclose(float(row[sweep_field]), sv):
                continue
            seed_key = (int(row.get("effective_data_seed", 0)), int(row.get("effective_model_seed", 0)))
            base = baseline_by_seed.get((float(sv), seed_key[0], seed_key[1]))
            cur = float(row[metric])
            if base is not None and np.isfinite(base) and np.isfinite(cur):
                gains.append(float(base - cur))
        center = _series_mean(gains)
        if np.isfinite(center):
            xs.append(float(sv))
            ys.append(center)
    return xs, ys


# ── sweep grid ───────────────────────────────────────────────────────────


def plot_sweep_grid(
    rows: Sequence[dict],
    *,
    family: FamilyReportConfig,
    output_path: Path,
    metric_defs: Sequence[tuple],
    title_prefix: str,
    panel_field: str,
    series_fields: tuple,
    capacity: CapacityKey | None = None,
) -> None:
    """Grid: panels=panel_field values, columns=metrics, series=series_fields combos."""
    if capacity is not None:
        rows = [r for r in rows if _capacity_key(r) == capacity]
    panel_vals = sorted({float(r[panel_field]) for r in rows})
    if not panel_vals:
        return

    # Series grouping: first field → color, second → linestyle
    s1_field = series_fields[0] if series_fields else "train_docs"
    s2_field = series_fields[1] if len(series_fields) > 1 else None
    s1_vals = sorted({r[s1_field] for r in rows})
    s2_vals = sorted({float(r[s2_field]) for r in rows}) if s2_field else [0.0]
    s1_colors = _build_color_map(s1_vals)
    s2_linestyles = _build_style_map(s2_vals, SCW_LINESTYLES)
    s2_markers = _build_style_map(s2_vals, SCW_MARKERS)

    sweep_field = family.sweep_field
    sweep_vals = sorted({float(r[sweep_field]) for r in rows})

    fig, axes = plt.subplots(
        len(panel_vals), len(metric_defs),
        figsize=(4.8 * len(metric_defs) + 1.4, 3.35 * len(panel_vals) + 1.8),
        squeeze=False,
    )
    fig.subplots_adjust(top=0.80, bottom=0.10, left=0.08, right=0.98, hspace=0.32, wspace=0.28)

    for row_idx, pv in enumerate(panel_vals):
        panel_subset = [r for r in rows if np.isclose(float(r[panel_field]), pv)]
        for col_idx, (metric, title, ylabel) in enumerate(metric_defs):
            ax = axes[row_idx][col_idx]
            for s1 in s1_vals:
                for s2 in s2_vals:
                    group = [r for r in panel_subset if r[s1_field] == s1]
                    if s2_field:
                        group = [r for r in group if np.isclose(float(r[s2_field]), s2)]
                    xs, ys = _build_metric_series(group, metric, sweep_field)
                    if not xs:
                        continue
                    ax.plot(xs, ys, color=s1_colors[s1],
                            linestyle=s2_linestyles[float(s2)],
                            marker=s2_markers[float(s2)],
                            markersize=4.8, linewidth=2.3)
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.text(-0.38, 0.5, f"{panel_field}={pv}",
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=11,
                        fontweight="bold", color="#3b352e")
            _set_sweep_ticks(ax, sweep_vals)
            _apply_axis_style(ax, ylabel=ylabel,
                              xlabel=family.sweep_label if row_idx == len(panel_vals) - 1 else "")

    _add_series_legends(
        fig,
        group1_vals=s1_vals, group1_colors=s1_colors,
        group1_title=f"Color = {s1_field}",
        group2_vals=s2_vals, group2_linestyles=s2_linestyles,
        group2_markers=s2_markers,
        group2_title=f"Style = {s2_field}" if s2_field else "",
    )
    fig.suptitle(title_prefix, fontsize=14, y=0.94)
    if capacity is not None:
        fig.text(0.5, 0.905, _capacity_label(capacity),
                 ha="center", va="center", fontsize=9.5, color="#4a433b")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# ── gain grid ────────────────────────────────────────────────────────────


def plot_gain_grid(
    rows: Sequence[dict],
    *,
    family: FamilyReportConfig,
    output_path: Path,
    metric_defs: Sequence[tuple],
    title_prefix: str,
    panel_field: str,
    series_fields: tuple,
    capacity: CapacityKey | None = None,
    baseline_field: str | None = None,
    baseline_value: object | None = None,
) -> None:
    """Like sweep_grid but shows gain vs sweep_value=0 baseline."""
    if capacity is not None:
        rows = [r for r in rows if _capacity_key(r) == capacity]
    panel_vals = sorted({float(r[panel_field]) for r in rows})
    if not panel_vals:
        return

    s1_field = series_fields[0] if series_fields else "train_docs"
    s2_field = series_fields[1] if len(series_fields) > 1 else None
    s1_vals = sorted({r[s1_field] for r in rows})
    s2_vals = sorted({float(r[s2_field]) for r in rows}) if s2_field else [0.0]
    s1_colors = _build_color_map(s1_vals)
    s2_linestyles = _build_style_map(s2_vals, SCW_LINESTYLES)
    s2_markers = _build_style_map(s2_vals, SCW_MARKERS)

    sweep_field = family.sweep_field
    sweep_vals = sorted({float(r[sweep_field]) for r in rows})

    fig, axes = plt.subplots(
        len(panel_vals), len(metric_defs),
        figsize=(4.8 * len(metric_defs) + 1.4, 3.35 * len(panel_vals) + 1.8),
        squeeze=False,
    )
    fig.subplots_adjust(top=0.80, bottom=0.10, left=0.08, right=0.98, hspace=0.32, wspace=0.28)

    for row_idx, pv in enumerate(panel_vals):
        panel_subset = [r for r in rows if np.isclose(float(r[panel_field]), pv)]
        for col_idx, (metric, title, ylabel) in enumerate(metric_defs):
            ax = axes[row_idx][col_idx]
            for s1 in s1_vals:
                for s2 in s2_vals:
                    group = [r for r in panel_subset if _value_matches(r.get(s1_field), s1)]
                    if s2_field:
                        group = [r for r in group if _value_matches(r.get(s2_field), s2)]
                    line_filters = {s1_field: s1}
                    if s2_field:
                        line_filters[s2_field] = s2
                    base_field = str(baseline_field or sweep_field)
                    base_value = baseline_value if baseline_value is not None else 0.0
                    xs, ys = _build_gain_series(
                        panel_subset,
                        metric,
                        sweep_field,
                        line_filters=line_filters,
                        baseline_field=base_field,
                        baseline_value=base_value,
                    )
                    if not xs:
                        continue
                    ax.plot(xs, ys, color=s1_colors[s1],
                            linestyle=s2_linestyles[float(s2)],
                            marker=s2_markers[float(s2)],
                            markersize=4.8, linewidth=2.3)
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.text(-0.38, 0.5, f"{panel_field}={pv}",
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=11,
                        fontweight="bold", color="#3b352e")
            _set_sweep_ticks(ax, sweep_vals)
            _apply_axis_style(ax, ylabel=ylabel,
                              xlabel=family.sweep_label if row_idx == len(panel_vals) - 1 else "",
                              zero_line=True)

    _add_series_legends(
        fig,
        group1_vals=s1_vals, group1_colors=s1_colors,
        group1_title=f"Color = {s1_field}",
        group2_vals=s2_vals, group2_linestyles=s2_linestyles,
        group2_markers=s2_markers,
        group2_title=f"Style = {s2_field}" if s2_field else "",
    )
    fig.suptitle(title_prefix, fontsize=14, y=0.94)
    if capacity is not None:
        fig.text(0.5, 0.905, _capacity_label(capacity),
                 ha="center", va="center", fontsize=9.5, color="#4a433b")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# ── audit summary ────────────────────────────────────────────────────────


def plot_audit_summary(
    aggregated_rows: Sequence[dict],
    *,
    family: FamilyReportConfig,
    output_path: Path,
    best_row_fn,
    theorem_score_fn,
    capacity: CapacityKey | None = None,
) -> None:
    """At each audit fraction, show metrics at the objective-optimal sweep value."""
    from matplotlib.lines import Line2D

    if capacity is not None:
        aggregated_rows = [r for r in aggregated_rows if _capacity_key(r) == capacity]

    audit_field = "audit_fraction"
    audit_vals = sorted({float(r[audit_field]) for r in aggregated_rows})
    if not audit_vals:
        return

    s1_field = family.sweep_group_fields[0] if family.sweep_group_fields else "train_docs"
    s2_field = family.sweep_group_fields[1] if len(family.sweep_group_fields) > 1 else None
    s1_vals = sorted({r[s1_field] for r in aggregated_rows})
    s2_vals = sorted({float(r[s2_field]) for r in aggregated_rows}) if s2_field else [0.0]
    s1_colors = _build_color_map(s1_vals)
    s2_linestyles = _build_style_map(s2_vals, SCW_LINESTYLES)
    s2_markers = _build_style_map(s2_vals, SCW_MARKERS)

    x_positions = np.arange(len(audit_vals), dtype=np.float64)
    x_labels = [_format_pct(v) for v in audit_vals]

    metric_defs = [
        ("learned_root_mae_n", f"{family.primary_metric_label} at objective optimum", "normalized error"),
        ("theorem_score", "Held-out theorem score at objective optimum", "normalized theorem error"),
        ("learned_spread_n", "Sensitivity at objective optimum", "normalized error"),
        (family.sweep_field, f"{family.sweep_label} at objective optimum", "weight"),
    ]

    fig, axes = plt.subplots(1, len(metric_defs), figsize=(15.2, 4.6), squeeze=False)
    fig.subplots_adjust(top=0.76, bottom=0.16, left=0.06, right=0.98, wspace=0.26)
    for col_idx, (metric, title, ylabel) in enumerate(metric_defs):
        ax = axes[0][col_idx]
        for s1 in s1_vals:
            for s2 in s2_vals:
                xs, ys = [], []
                for pos, av in zip(x_positions, audit_vals):
                    filters = {s1_field: s1, audit_field: float(av)}
                    if s2_field:
                        filters[s2_field] = float(s2)
                    best = best_row_fn(aggregated_rows, **filters)
                    if best is None:
                        continue
                    val = theorem_score_fn(best) if metric == "theorem_score" else float(best.get(metric, float("nan")))
                    if np.isfinite(val):
                        xs.append(float(pos))
                        ys.append(float(val))
                if not xs:
                    continue
                ax.plot(xs, ys, color=s1_colors[s1],
                        linestyle=s2_linestyles[float(s2)],
                        marker=s2_markers[float(s2)],
                        markersize=5.0, linewidth=2.1)
        ax.set_title(title)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        _apply_axis_style(ax, ylabel=ylabel, xlabel="q_audit")

    _add_series_legends(
        fig,
        group1_vals=s1_vals, group1_colors=s1_colors,
        group1_title=f"Color = {s1_field}",
        group2_vals=s2_vals, group2_linestyles=s2_linestyles,
        group2_markers=s2_markers,
        group2_title=f"Style = {s2_field}" if s2_field else "",
    )
    fig.suptitle(f"Sparse vs full audit at objective-optimal {family.sweep_label}", fontsize=14, y=0.935)
    if capacity is not None:
        fig.text(0.5, 0.895, _capacity_label(capacity),
                 ha="center", va="center", fontsize=9.5, color="#4a433b")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# ── capacity summary ────────────────────────────────────────────────────


def plot_capacity_summary(
    aggregated_rows: Sequence[dict],
    *,
    family: FamilyReportConfig,
    output_path: Path,
    selection_objective_fn,
    theorem_score_fn,
) -> None:
    """Compare capacities at their objective-optimal sweep value."""
    from matplotlib.lines import Line2D

    capacity_keys = sorted({_capacity_key(r) for r in aggregated_rows})
    if len(capacity_keys) <= 1:
        return
    audit_vals = sorted({float(r["audit_fraction"]) for r in aggregated_rows})
    if not audit_vals:
        return

    x_positions = np.arange(len(capacity_keys), dtype=np.float64)
    x_labels = [f"sd={c[0]}\nhd={c[1]}\nep={c[2]}" for c in capacity_keys]
    audit_color_map = _build_color_map([int(round(float(a) * 1000.0)) for a in audit_vals])

    metric_defs = [
        ("learned_root_mae_n", f"{family.primary_metric_label} at objective optimum", "normalized error"),
        ("theorem_score", "Held-out theorem score", "normalized theorem error"),
        ("learned_spread_n", "Sensitivity at objective optimum", "normalized error"),
        (family.sweep_field, f"{family.sweep_label} at objective optimum", "weight"),
    ]

    fig, axes = plt.subplots(1, len(metric_defs), figsize=(15.4, 4.9), squeeze=False)
    fig.subplots_adjust(top=0.78, bottom=0.23, left=0.06, right=0.98, wspace=0.28)
    for col_idx, (metric, title, ylabel) in enumerate(metric_defs):
        ax = axes[0][col_idx]
        for av in audit_vals:
            xs, ys = [], []
            for pos, cap in zip(x_positions, capacity_keys):
                candidates = [r for r in aggregated_rows
                              if _capacity_key(r) == cap
                              and np.isclose(float(r["audit_fraction"]), av)]
                if not candidates:
                    continue
                best = min(candidates, key=selection_objective_fn)
                val = theorem_score_fn(best) if metric == "theorem_score" else float(best.get(metric, float("nan")))
                if np.isfinite(val):
                    xs.append(float(pos))
                    ys.append(float(val))
            if not xs:
                continue
            color = audit_color_map[int(round(float(av) * 1000.0))]
            ax.plot(xs, ys, color=color, marker="o", linewidth=2.1, markersize=5.2,
                    label=_format_pct(av))
        ax.set_title(title)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        _apply_axis_style(ax, ylabel=ylabel, xlabel="capacity")

    handles = [
        Line2D([0], [0], color=audit_color_map[int(round(float(a) * 1000.0))],
               marker="o", linewidth=2.1, label=_format_pct(a))
        for a in audit_vals
    ]
    fig.legend(handles=handles, title="Color = q_audit",
               loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncol=max(1, min(4, len(handles))), frameon=False)
    fig.suptitle(f"Capacity summary at objective-optimal {family.sweep_label}", fontsize=14, y=0.96)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# ── optimization appendix ───────────────────────────────────────────────


def plot_optimization_appendix(
    rows: Sequence[dict],
    *,
    family: FamilyReportConfig,
    output_path: Path,
    panel_field: str,
    series_fields: tuple,
    capacity: CapacityKey | None = None,
) -> None:
    """Generalization gap and train loss vs sweep variable."""
    if capacity is not None:
        rows = [r for r in rows if _capacity_key(r) == capacity]
    panel_vals = sorted({float(r[panel_field]) for r in rows})
    if not panel_vals:
        return

    s1_field = series_fields[0] if series_fields else "train_docs"
    s2_field = series_fields[1] if len(series_fields) > 1 else None
    s1_vals = sorted({r[s1_field] for r in rows})
    s2_vals = sorted({float(r[s2_field]) for r in rows}) if s2_field else [0.0]
    s1_colors = _build_color_map(s1_vals)
    s2_linestyles = _build_style_map(s2_vals, SCW_LINESTYLES)
    s2_markers = _build_style_map(s2_vals, SCW_MARKERS)

    sweep_field = family.sweep_field
    sweep_vals = sorted({float(r[sweep_field]) for r in rows})

    metric_defs = [
        ("generalization_gap_law_score_n", "Held-out minus train theorem score", "gap"),
        ("train_loss_final", "Final train loss", "optimization loss"),
    ]

    fig, axes = plt.subplots(
        len(panel_vals), len(metric_defs),
        figsize=(10.2, 3.0 * len(panel_vals) + 1.4),
        squeeze=False,
    )
    fig.subplots_adjust(top=0.71, bottom=0.12, left=0.08, right=0.98, hspace=0.34, wspace=0.26)
    for row_idx, pv in enumerate(panel_vals):
        panel_subset = [r for r in rows if np.isclose(float(r[panel_field]), pv)]
        for col_idx, (metric, title, ylabel) in enumerate(metric_defs):
            ax = axes[row_idx][col_idx]
            for s1 in s1_vals:
                for s2 in s2_vals:
                    group = [r for r in panel_subset if r[s1_field] == s1]
                    if s2_field:
                        group = [r for r in group if np.isclose(float(r[s2_field]), s2)]
                    xs, ys = _build_metric_series(group, metric, sweep_field)
                    if not xs:
                        continue
                    ax.plot(xs, ys, color=s1_colors[s1],
                            linestyle=s2_linestyles[float(s2)],
                            marker=s2_markers[float(s2)],
                            markersize=4.8, linewidth=2.3)
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.text(-0.44, 0.5, f"{panel_field}={pv}",
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=11,
                        fontweight="bold", color="#3b352e")
            _set_sweep_ticks(ax, sweep_vals)
            _apply_axis_style(ax, ylabel=ylabel,
                              xlabel=family.sweep_label if row_idx == len(panel_vals) - 1 else "",
                              zero_line=(metric == "generalization_gap_law_score_n"))

    _add_series_legends(
        fig,
        group1_vals=s1_vals, group1_colors=s1_colors,
        group1_title=f"Color = {s1_field}",
        group2_vals=s2_vals, group2_linestyles=s2_linestyles,
        group2_markers=s2_markers,
        group2_title=f"Style = {s2_field}" if s2_field else "",
    )
    fig.suptitle("Optimization appendix: theorem gap and train loss", fontsize=14, y=0.88)
    if capacity is not None:
        fig.text(0.5, 0.845, _capacity_label(capacity),
                 ha="center", va="center", fontsize=9.5, color="#4a433b")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


__all__ = [
    "plot_ablation_bar_chart",
    "plot_audit_summary",
    "plot_capacity_summary",
    "plot_exact_family_counterexamples",
    "plot_gain_grid",
    "plot_heatmap",
    "plot_mechanism_pareto",
    "plot_optimization_appendix",
    "plot_sweep_grid",
]
