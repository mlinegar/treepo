#!/usr/bin/env python3
"""Ceiling-focused plots for Segment-LDA OPS weight-recovery simulation.

This script makes the "upper bounds" explicit:

- `exact`: oracle mergeable sketch (absolute ceiling, ~0 distortion)
- `undersupported`: insufficient sketch family (approximation-bias floor)
- `ridge_true_topics`: best-case downstream estimator (true topics, same audit budgets)
- `ridge_infer_true_phi`: topic inference only (oracle phi, inferred topics)
- `ridge_infer_est_phi`: topic inference + upstream phi estimation
- `ridge`: the configured pipeline (typically topic_source=infer)

Inputs are per-run JSON outputs from `ctreepo sim run segment-lda-ops`.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import statistics
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ceilings/floors and gap-to-ceiling for Segment-LDA OPS sims.")
    p.add_argument(
        "--input-glob",
        type=str,
        default="outputs/segment_lda_ops_weight_recovery/**/*seed_*.json",
        help="Glob for per-run JSON outputs.",
    )
    p.add_argument(
        "--audit-strategy",
        type=str,
        default="random",
        help="Filter to this audit_strategy (empty means no filter).",
    )
    p.add_argument("--topic-phi-estimator", type=str, default="", help="Optional exact filter.")
    p.add_argument("--topic-process", type=str, default="", help="Optional exact filter (segments/bag_of_words).")
    p.add_argument("--lambda-multiplier", type=float, default=float("nan"), help="Optional exact filter.")
    p.add_argument("--train-docs", type=int, default=-1, help="Optional exact filter.")
    p.add_argument(
        "--x-axis",
        choices=["oracle_cost_ratio", "oracle_cost_total", "oracle_queries_total", "train_docs"],
        default="oracle_cost_ratio",
        help="X axis for the plots.",
    )
    p.add_argument(
        "--aggregate",
        choices=["median", "mean"],
        default="median",
        help="How to aggregate across seeds per x value.",
    )
    p.add_argument(
        "--band",
        choices=["none", "p10_p90", "p25_p75"],
        default="p10_p90",
        help="Optional quantile band across seeds for the main ridge curve.",
    )
    p.add_argument("--log-x", action="store_true")
    p.add_argument(
        "--output-figure",
        type=str,
        default="outputs/segment_lda_ops_weight_recovery_ceilings.png",
        help="Output PNG figure path.",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="outputs/segment_lda_ops_weight_recovery_ceilings_report.json",
        help="Output JSON report path.",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def _reduce(vals: List[float], *, agg: str) -> float:
    vals2 = [float(x) for x in vals if np.isfinite(float(x))]
    if not vals2:
        return float("nan")
    if agg == "mean":
        return float(np.mean(np.asarray(vals2, dtype=np.float64)))
    if agg == "median":
        return float(statistics.median(vals2))
    raise ValueError(f"unsupported aggregate: {agg!r}")


def _percentile(vals: List[float], q: float) -> float:
    vals2 = np.asarray([float(x) for x in vals if np.isfinite(float(x))], dtype=np.float64)
    if vals2.size == 0:
        return float("nan")
    return float(np.percentile(vals2, q))


def _band_quantiles(kind: str) -> Optional[Tuple[float, float]]:
    if kind == "none":
        return None
    if kind == "p10_p90":
        return (10.0, 90.0)
    if kind == "p25_p75":
        return (25.0, 75.0)
    raise ValueError(f"unsupported band: {kind!r}")


def _collect_rows(files: Iterable[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        metrics = payload.get("metrics", {}) or {}
        if not isinstance(metrics, dict):
            continue

        ridge = metrics.get("ridge", {}) or {}
        if not isinstance(ridge, dict):
            continue

        row = {
            "path": str(path),
            "seed": int(cfg.get("seed", -1)),
            "audit_strategy": str(cfg.get("audit_strategy", "")),
            "audit_fraction": float(cfg.get("audit_fraction", float("nan"))),
            "topic_phi_estimator": str(cfg.get("topic_phi_estimator", "")),
            "topic_process": str(cfg.get("topic_process", "")),
            "lambda_multiplier": float(cfg.get("lambda_multiplier", float("nan"))),
            "train_docs": int(cfg.get("train_docs", -1)),
            "x": {
                "oracle_cost_ratio": float(ridge.get("oracle_cost_ratio", float("nan"))),
                "oracle_cost_total": float(ridge.get("oracle_cost_total", float("nan"))),
                "oracle_queries_total": float(ridge.get("oracle_queries_total", float("nan"))),
                "train_docs": float(cfg.get("train_docs", float("nan"))),
            },
            "y": {},
        }

        def _get_root_mae(key: str) -> float:
            block = metrics.get(key, {}) or {}
            if not isinstance(block, dict):
                return float("nan")
            return float(block.get("root_mae", float("nan")))

        row["y"]["exact"] = _get_root_mae("exact")
        row["y"]["undersupported"] = _get_root_mae("undersupported")
        row["y"]["ridge"] = _get_root_mae("ridge")
        row["y"]["ridge_true_topics"] = _get_root_mae("ridge_true_topics")
        row["y"]["ridge_infer_true_phi"] = _get_root_mae("ridge_infer_true_phi")
        row["y"]["ridge_infer_est_phi"] = _get_root_mae("ridge_infer_est_phi")

        rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    files = [Path(p) for p in sorted(glob.glob(str(args.input_glob), recursive=True))]
    if not files:
        raise ValueError(f"no files matched: {args.input_glob}")

    rows = _collect_rows(files)

    if str(args.audit_strategy):
        rows = [r for r in rows if str(r["audit_strategy"]) == str(args.audit_strategy)]
    if str(args.topic_phi_estimator):
        rows = [r for r in rows if str(r["topic_phi_estimator"]) == str(args.topic_phi_estimator)]
    if str(args.topic_process):
        rows = [r for r in rows if str(r["topic_process"]) == str(args.topic_process)]
    if int(args.train_docs) >= 0:
        rows = [r for r in rows if int(r["train_docs"]) == int(args.train_docs)]
    if np.isfinite(float(args.lambda_multiplier)):
        lam = float(args.lambda_multiplier)
        rows = [r for r in rows if np.isfinite(float(r["lambda_multiplier"])) and float(r["lambda_multiplier"]) == lam]

    if not rows:
        raise ValueError("no rows after filters")

    x_key = str(args.x_axis)
    qband = _band_quantiles(str(args.band))

    by_x: Dict[float, List[dict]] = {}
    for r in rows:
        x = float(r["x"].get(x_key, float("nan")))
        if not np.isfinite(x):
            continue
        by_x.setdefault(x, []).append(r)
    if not by_x:
        raise ValueError("no finite x values for plotting")

    xs = sorted(by_x.keys())

    def _series(name: str) -> Tuple[List[float], List[float], List[float]]:
        ys = []
        ylo = []
        yhi = []
        for x in xs:
            vals = [float(rr["y"].get(name, float("nan"))) for rr in by_x[x]]
            ys.append(_reduce(vals, agg=str(args.aggregate)))
            if qband is None:
                ylo.append(float("nan"))
                yhi.append(float("nan"))
            else:
                ylo.append(_percentile(vals, qband[0]))
                yhi.append(_percentile(vals, qband[1]))
        return ys, ylo, yhi

    exact_line = _reduce([float(r["y"]["exact"]) for r in rows], agg=str(args.aggregate))
    undersupported_line = _reduce([float(r["y"]["undersupported"]) for r in rows], agg=str(args.aggregate))
    full_audit_rows = [
        r
        for r in rows
        if np.isfinite(float(r.get("audit_fraction", float("nan"))))
        and abs(float(r["audit_fraction"]) - 1.0) <= 1e-12
    ]
    full_audit_diagnostic = {
        "present": bool(full_audit_rows),
        "n_rows": int(len(full_audit_rows)),
        "ridge_root_mae": _reduce([float(r["y"].get("ridge", float("nan"))) for r in full_audit_rows], agg=str(args.aggregate)),
        "exact_root_mae": _reduce([float(r["y"].get("exact", float("nan"))) for r in full_audit_rows], agg=str(args.aggregate)),
        "undersupported_root_mae": _reduce(
            [float(r["y"].get("undersupported", float("nan"))) for r in full_audit_rows],
            agg=str(args.aggregate),
        ),
    }

    series_payload: Dict[str, dict] = {}

    fig, axes = plt.subplots(1, 2, figsize=(16.4, 6.4), constrained_layout=True)
    ax0, ax1 = axes

    curves = [
        ("ridge_true_topics", "ceiling: ridge (true topics)", "#111111", "--"),
        ("ridge_infer_true_phi", "topic inference only", "#9467bd", "-."),
        ("ridge_infer_est_phi", "topic+phi estimation", "#ff7f0e", "-."),
        ("ridge", "pipeline ridge", "#2ca02c", "-"),
    ]

    for key, label, color, ls in curves:
        ys, ylo, yhi = _series(key)
        if all(not np.isfinite(y) for y in ys):
            continue
        series_payload[key] = {
            "label": label,
            "color": color,
            "linestyle": ls,
            "x": [float(x) for x in xs],
            "y": [float(y) for y in ys],
            "y_lo": [float(v) for v in ylo],
            "y_hi": [float(v) for v in yhi],
        }
        ax0.plot(xs, ys, marker="o", linewidth=2.2, markersize=5.5, color=color, linestyle=ls, label=label)
        if qband is not None and key == "ridge":
            lo_arr = np.asarray(ylo, dtype=np.float64)
            hi_arr = np.asarray(yhi, dtype=np.float64)
            ok = np.isfinite(lo_arr) & np.isfinite(hi_arr)
            if np.any(ok):
                ax0.fill_between(
                    np.asarray(xs, dtype=np.float64)[ok],
                    lo_arr[ok],
                    hi_arr[ok],
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )

    ax0.axhline(exact_line, color="#222222", linestyle=":", linewidth=2.0, label="exact (absolute ceiling)")
    ax0.axhline(undersupported_line, color="#444444", linestyle="--", linewidth=2.0, label="undersupported (bias floor)")
    if not bool(full_audit_diagnostic["present"]):
        ax0.text(
            0.02,
            0.98,
            "No audit_fraction=1.0 runs in input",
            transform=ax0.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#b22222",
        )
    ax0.set_xlabel(x_key, fontsize=12)
    ax0.set_ylabel("Root MAE", fontsize=12)
    ax0.set_title("Ceilings + Floor: Root Distortion", fontsize=14)
    ax0.tick_params(axis="both", labelsize=11)
    if bool(args.log_x):
        ax0.set_xscale("log")
    ax0.grid(alpha=0.25)
    ax0.legend(frameon=False, fontsize=10)

    gap_payload: Dict[str, object] = {"has_ridge_true_topics": False}
    if any(np.isfinite(float(r["y"].get("ridge_true_topics", float("nan")))) for r in rows):
        gap_payload["has_ridge_true_topics"] = True
        gaps = []
        for x in xs:
            vals = []
            for rr in by_x[x]:
                y = float(rr["y"].get("ridge", float("nan")))
                ceil = float(rr["y"].get("ridge_true_topics", float("nan")))
                if np.isfinite(y) and np.isfinite(ceil):
                    vals.append(y - ceil)
            gaps.append(_reduce(vals, agg=str(args.aggregate)) if vals else float("nan"))
        gap_payload["x"] = [float(x) for x in xs]
        gap_payload["gap_ridge_minus_ceiling"] = [float(g) for g in gaps]
        ax1.plot(xs, gaps, marker="o", linewidth=2.2, markersize=5.5, color="#2ca02c", label="ridge − ridge_true_topics")
        ax1.axhline(0.0, color="#444444", linewidth=1.0)
        ax1.set_ylabel("Excess root MAE over ceiling", fontsize=12)
        ax1.legend(frameon=False, fontsize=10)
    else:
        ax1.text(
            0.5,
            0.5,
            "No ridge_true_topics in inputs.\nRe-run sims with --run-all-feature-modes.",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax1.set_axis_off()
    ax1.set_xlabel(x_key, fontsize=12)
    ax1.set_title("Gap to Ceiling", fontsize=14)
    ax1.tick_params(axis="both", labelsize=11)
    if bool(args.log_x) and ax1.get_xaxis().get_scale() != "log":
        ax1.set_xscale("log")
    ax1.grid(alpha=0.25)

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)

    report = {
        "input_glob": str(args.input_glob),
        "n_files": int(len(files)),
        "n_rows": int(len(rows)),
        "filters": {
            "audit_strategy": str(args.audit_strategy),
            "topic_phi_estimator": str(args.topic_phi_estimator),
            "topic_process": str(args.topic_process),
            "train_docs": int(args.train_docs),
            "lambda_multiplier": float(args.lambda_multiplier),
        },
        "x_axis": x_key,
        "aggregate": str(args.aggregate),
        "band": str(args.band),
        "baseline": {"exact_root_mae": exact_line, "undersupported_root_mae": undersupported_line},
        "diagnostics": {"full_audit": full_audit_diagnostic},
        "series": series_payload,
        "gap": gap_payload,
        "output_figure": str(out_fig),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_figure": str(out_fig), "output_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

