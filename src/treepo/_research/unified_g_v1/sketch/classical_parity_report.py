"""Report generator for the classical-HLL parity sweep.

Consumes the `summary.csv` written by `scripts/run_classical_parity_benchmark.py`
and emits:

- `curve.{pdf,png}` — supplied-oracle error vs leaves per document, one panel
  per HLL precision.
- `paper/ctreepo/tables/classical_parity_hll.{md,tex}` — a paper-facing table
  aggregating across seeds.

The CSV format matches `FitResult.metrics` keys so this module has no
dependency on any trainer beyond reading the CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = (
    "precision",
    "n_leaves",
    "backend",
    "seed",
    "root_rel_mae",
    "hll_rse_theory",
)


METHOD_STYLE = {
    "classical_native": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
    "classical_datasketches": {"color": "#7f7f7f", "linestyle": "-", "marker": "s"},
    "learned_g": {"color": "#2ca02c", "linestyle": "--", "marker": "^"},
    "learned_g_oracle_state": {"color": "#9467bd", "linestyle": "-.", "marker": "P"},
    "learned_joint": {"color": "#d62728", "linestyle": "--", "marker": "D"},
    "learned_fg": {"color": "#d62728", "linestyle": "--", "marker": "D"},  # legacy CSVs
}


METHOD_LABEL = {
    "classical_native": "official/native",
    "classical_datasketches": "official/DataSketches",
    "learned_g": "learned leaf+g + fixed f*",
    "learned_g_oracle_state": "learned g only + fixed f*",
    "learned_joint": "learned f+g",
    "learned_fg": "learned f+g",
}

PLOT_LABEL = {
    **METHOD_LABEL,
    "classical_native": "official/native (exact)",
}

METHOD_ORDER = (
    "classical_native",
    "classical_datasketches",
    "learned_g_oracle_state",
    "learned_g",
    "learned_joint",
)

PLOT_METHOD_ORDER = (
    "classical_native",
    "classical_datasketches",
    "learned_g_oracle_state",
    "learned_joint",
)


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if rows:
        missing = set(REQUIRED_COLUMNS) - set(rows[0].keys())
        if missing:
            raise ValueError(f"missing columns in {csv_path}: {sorted(missing)}")
    return rows


def _group_by(rows: list[dict[str, str]], keys: Iterable[str]) -> dict[tuple, list[dict[str, str]]]:
    keys = tuple(keys)
    out: dict[tuple, list[dict[str, str]]] = {}
    for row in rows:
        k = tuple(row[key] for key in keys)
        out.setdefault(k, []).append(row)
    return out


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def _ci95(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(1.96 * statistics.stdev(xs) / math.sqrt(len(xs)))


def _method_of(row: dict[str, str]) -> str:
    """Back-compat: older CSVs have no `method` column, so derive from backend."""
    m = row.get("method")
    if m:
        return "learned_joint" if str(m) == "learned_fg" else str(m)
    return f"classical_{row['backend']}"


def _method_label(method: str) -> str:
    return METHOD_LABEL.get(str(method), str(method))


def _plot_label(method: str) -> str:
    return PLOT_LABEL.get(str(method), _method_label(method))


def _method_sort_key(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), method)


def write_curve_png(rows: list[dict[str, str]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        out_path.write_text("matplotlib not installed; skipping curve.png\n")
        return
    try:
        import importlib.util

        paperplot_path = Path("paper/ctreepo/scripts/paperplot.py")
        spec = importlib.util.spec_from_file_location("ctreepo_paperplot", paperplot_path)
        if spec is None or spec.loader is None:
            raise ImportError(str(paperplot_path))
        paperplot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(paperplot)
        paperplot.rcparams()
    except Exception:
        pass

    plot_rows = [row for row in rows if row.get("oracle_kind") == "hll_reference"]
    if not plot_rows:
        plot_rows = rows

    available_methods = {_method_of(row) for row in plot_rows}
    methods = [method for method in PLOT_METHOD_ORDER if method in available_methods]
    methods.extend(
        method for method in sorted(available_methods, key=_method_sort_key)
        if method not in methods and method != "learned_g"
    )
    n_leaves_values = sorted({int(row["n_leaves"]) for row in plot_rows})
    precisions = sorted({int(row["precision"]) for row in plot_rows})
    positive_values: list[float] = []
    for row in plot_rows:
        try:
            value = float(row.get("root_rel_mae", "nan"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            positive_values.append(value)
    min_positive = min(positive_values) if positive_values else 1e-3
    max_positive = max(positive_values) if positive_values else 1.0
    display_floor = max(1e-4, min_positive / 8.0)
    y_top = max(0.2, max_positive * 1.9)

    n_panels = len(precisions)
    fig, axes = plt.subplots(
        1,
        max(1, n_panels),
        figsize=((4.65 if n_panels == 1 else 3.45 * max(1, n_panels)), 3.15),
        sharey=True,
        squeeze=False,
    )

    legend_handles = None
    legend_labels = None
    for panel_idx, precision in enumerate(precisions):
        ax = axes[0, panel_idx]
        for method in methods:
            style = METHOD_STYLE.get(method, {"color": None, "linestyle": "-", "marker": "o"})
            xs, ys, errs = [], [], []
            for n_leaves in n_leaves_values:
                seeds = [
                    row for row in plot_rows
                    if int(row["precision"]) == precision
                    and int(row["n_leaves"]) == n_leaves
                    and _method_of(row) == method
                ]
                if not seeds:
                    continue
                vals = [float(row["root_rel_mae"]) for row in seeds if row.get("root_rel_mae") not in (None, "", "nan")]
                if not vals:
                    continue
                mean = _mean(vals)
                xs.append(n_leaves)
                ys.append(max(mean, display_floor))
                errs.append(0.0 if mean <= 0 else _ci95(vals))
            if xs:
                ax.errorbar(
                    xs, ys, yerr=errs,
                    label=_plot_label(method),
                    linewidth=1.65,
                    markersize=5.0,
                    capsize=2.5,
                    **style,
                )
        # Theoretical HLL floor.
        floor = 1.04 / math.sqrt(1 << precision)
        ax.plot(
            n_leaves_values,
            [floor for _ in n_leaves_values],
            linestyle=":",
            color="black",
            linewidth=1.15,
            label="HLL RSE floor",
        )
        ax.set_xlabel("leaves per document")
        if panel_idx == 0:
            ax.set_ylabel("relative MAE (lower is better)")
        ax.set_title(f"HLL precision $p={precision}$")
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_leaves_values)
        ax.set_xticklabels([str(x) for x in n_leaves_values])
        ax.set_yscale("log")
        ax.set_ylim(display_floor / 1.6, y_top)
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=5))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=range(2, 10), numticks=8))
        ax.grid(True, which="major", alpha=0.28)
        ax.grid(True, which="minor", alpha=0.08)
        legend_handles, legend_labels = ax.get_legend_handles_labels()

    if n_panels == 1:
        axes[0, 0].set_title(f"HLL supplied-oracle parity ($p={precisions[0]}$)")
    else:
        fig.suptitle("HLL supplied-oracle parity", fontsize=10.5)
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.035),
            ncol=min(3, len(legend_labels)),
            fontsize=7,
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0.21, 1, 0.98 if n_panels == 1 else 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _summary_table(rows: list[dict[str, str]]) -> list[dict[str, str | float]]:
    # Expected columns in new CSVs: method, backend, oracle_kind, precision,
    # n_leaves. Fall back gracefully for pre-method-column CSVs.
    has_method = bool(rows) and "method" in rows[0]
    has_oracle = bool(rows) and "oracle_kind" in rows[0]
    if has_method and has_oracle:
        group_keys = ("method", "oracle_kind", "precision", "n_leaves")
    elif has_oracle:
        group_keys = ("backend", "oracle_kind", "precision", "n_leaves")
    else:
        group_keys = ("backend", "precision", "n_leaves")
    grouped = _group_by(rows, group_keys)
    summary: list[dict[str, str | float]] = []
    for key, group in sorted(grouped.items()):
        backend = group[0].get("backend", "")
        if has_method and has_oracle:
            method, oracle_kind, precision, n_leaves = key
        elif has_oracle:
            method = _method_of(group[0])
            backend, oracle_kind, precision, n_leaves = key
        else:
            method = _method_of(group[0])
            backend, precision, n_leaves = key
            oracle_kind = "analytic"

        def _nums(key: str) -> list[float]:
            out: list[float] = []
            for r in group:
                v = r.get(key)
                if v in (None, "", "nan"):
                    continue
                try:
                    out.append(float(v))
                except (TypeError, ValueError):
                    continue
            return out

        deltas_abs = _nums("flat_vs_tree_abs_mean")
        deltas_rel = _nums("flat_vs_tree_rel_mean")
        state_equal = _nums("state_equal_rate")
        bytes_equal = _nums("state_bytes_equal_rate")
        tree_ms = _nums("tree_wall_ms_mean")
        flat_ms = _nums("flat_wall_ms_mean")
        root_rel_mae = _nums("root_rel_mae")
        c1 = _nums("c1_mae")
        c3 = _nums("c3_mae")
        merge_state = _nums("merge_state_mae")
        rse = float(group[0]["hll_rse_theory"])
        wall_ratio = (_mean(tree_ms) / max(1e-9, _mean(flat_ms))) if flat_ms and tree_ms else float("nan")
        summary.append(
            {
                "method": method,
                "backend": backend,
                "oracle_kind": str(oracle_kind),
                "precision": int(precision),
                "n_leaves": int(n_leaves),
                "hll_rse_theory": rse,
                "mean_abs_delta": _mean(deltas_abs) if deltas_abs else float("nan"),
                "max_abs_delta": float(max(deltas_abs)) if deltas_abs else float("nan"),
                "mean_rel_delta": _mean(deltas_rel) if deltas_rel else float("nan"),
                "ci95_rel_delta": _ci95(deltas_rel) if deltas_rel else float("nan"),
                "state_equal_rate": _mean(state_equal) if state_equal else float("nan"),
                "bytes_equal_rate": _mean(bytes_equal) if bytes_equal else float("nan"),
                "root_rel_mae_mean": _mean(root_rel_mae) if root_rel_mae else float("nan"),
                "c1_mae_mean": _mean(c1) if c1 else float("nan"),
                "c3_mae_mean": _mean(c3) if c3 else float("nan"),
                "merge_state_mae_mean": _mean(merge_state) if merge_state else float("nan"),
                "wall_clock_ratio": wall_ratio,
                "n_seeds": len(group),
            }
        )
    return summary


def _fmt(x: float | str) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if v != v:  # NaN
        return "—"
    return f"{v:.4g}"


def write_markdown_table(summary: list[dict[str, str | float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| method | oracle | p | L | HLL RSE | mean |Δ_rel| | 95% CI | max |Δ| | state= | bytes= | rel MAE | c1 | c3 | merge state | wall × | seeds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {method} | {oracle} | {p} | {L} | {rse} | {mrel} | {ci} | {mabs} | "
            "{seq} | {beq} | {rmae} | {c1} | {c3} | {mstate} | {wall} | {n} |".format(
                method=_method_label(str(row.get("method", row.get("backend", "")))),
                oracle=row["oracle_kind"],
                p=row["precision"],
                L=row["n_leaves"],
                rse=_fmt(row["hll_rse_theory"]),
                mrel=_fmt(row["mean_rel_delta"]),
                ci=_fmt(row["ci95_rel_delta"]),
                mabs=_fmt(row["max_abs_delta"]),
                seq=_fmt(row["state_equal_rate"]),
                beq=_fmt(row["bytes_equal_rate"]),
                rmae=_fmt(row["root_rel_mae_mean"]),
                c1=_fmt(row.get("c1_mae_mean", float("nan"))),
                c3=_fmt(row.get("c3_mae_mean", float("nan"))),
                mstate=_fmt(row.get("merge_state_mae_mean", float("nan"))),
                wall=_fmt(row["wall_clock_ratio"]),
                n=row["n_seeds"],
            )
        )
    out_path.write_text("\n".join(lines) + "\n")


def _tex_fmt(x: float | str) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    if v != v:
        return "--"
    return f"{v:.4g}"


def write_latex_table(summary: list[dict[str, str | float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by unified_g_v1.sketch.classical_parity_report; do not edit.",
        "\\begin{tabular}{llrrrrrrrrrrrrrr}",
        "\\toprule",
        "method & oracle & $p$ & $L$ & RSE & $\\overline{|\\Delta_{rel}|}$ & 95\\% CI "
        "& $\\max|\\Delta|$ & state$=$ & bytes$=$ & rel MAE & C1 & C3 & merge state & wall$\\times$ & $n$ \\\\",
        "\\midrule",
    ]
    for row in summary:
        method = _method_label(str(row.get("method", row.get("backend", "")))).replace("_", "\\_")
        lines.append(
            "{method} & {oracle} & {p} & {L} & {rse} & {mrel} & {ci} & {mabs} & "
            "{seq} & {beq} & {rmae} & {c1} & {c3} & {mstate} & {wall} & {n} \\\\".format(
                method=method,
                oracle=str(row["oracle_kind"]).replace("_", "\\_"),
                p=row["precision"],
                L=row["n_leaves"],
                rse=_tex_fmt(row["hll_rse_theory"]),
                mrel=_tex_fmt(row["mean_rel_delta"]),
                ci=_tex_fmt(row["ci95_rel_delta"]),
                mabs=_tex_fmt(row["max_abs_delta"]),
                seq=_tex_fmt(row["state_equal_rate"]),
                beq=_tex_fmt(row["bytes_equal_rate"]),
                rmae=_tex_fmt(row["root_rel_mae_mean"]),
                c1=_tex_fmt(row.get("c1_mae_mean", float("nan"))),
                c3=_tex_fmt(row.get("c3_mae_mean", float("nan"))),
                mstate=_tex_fmt(row.get("merge_state_mae_mean", float("nan"))),
                wall=_tex_fmt(row["wall_clock_ratio"]),
                n=row["n_seeds"],
            )
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n")


def write_supplied_oracle_sanity(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    methods = ("classical_native", "classical_datasketches")
    metrics = ("root_mae", "root_rel_mae", "c1_mae", "c3_mae")
    lines = [
        "# Supplied-Oracle Single-Leaf Sanity",
        "",
        "Rows included here use `oracle_kind=hll_reference` and `n_leaves=1`.",
        "",
        "| method | max root MAE | max root rel MAE | max C1 MAE | max C3 MAE | rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        selected = [
            row for row in rows
            if _method_of(row) == method
            and row.get("oracle_kind") == "hll_reference"
            and str(row.get("n_leaves")) == "1"
        ]
        vals: dict[str, float] = {}
        for metric in metrics:
            nums: list[float] = []
            for row in selected:
                try:
                    nums.append(float(row.get(metric, "nan")))
                except (TypeError, ValueError):
                    pass
            vals[metric] = max(nums) if nums else float("nan")
        lines.append(
            "| {method} | {root} | {rel} | {c1} | {c3} | {n} |".format(
                method=_method_label(method),
                root=_fmt(vals["root_mae"]),
                rel=_fmt(vals["root_rel_mae"]),
                c1=_fmt(vals["c1_mae"]),
                c3=_fmt(vals["c3_mae"]),
                n=len(selected),
            )
        )
    out_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render classical-HLL parity report from summary.csv")
    ap.add_argument("--summary", type=Path, required=True, help="Path to summary.csv")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory for curve.png output")
    ap.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("paper/ctreepo/tables"),
        help="Directory to write classical_parity_hll.md and .tex",
    )
    args = ap.parse_args(argv)

    rows = _read_rows(args.summary)
    summary = _summary_table(rows)

    write_curve_png(rows, args.out_dir / "curve.png")
    write_curve_png(rows, args.out_dir / "curve.pdf")
    write_supplied_oracle_sanity(rows, args.out_dir / "supplied_oracle_sanity.md")
    write_markdown_table(summary, args.tables_dir / "classical_parity_hll.md")
    write_latex_table(summary, args.tables_dir / "classical_parity_hll.tex")
    print(f"wrote {args.out_dir / 'curve.png'}")
    print(f"wrote {args.out_dir / 'curve.pdf'}")
    print(f"wrote {args.out_dir / 'supplied_oracle_sanity.md'}")
    print(f"wrote {args.tables_dir / 'classical_parity_hll.md'}")
    print(f"wrote {args.tables_dir / 'classical_parity_hll.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
