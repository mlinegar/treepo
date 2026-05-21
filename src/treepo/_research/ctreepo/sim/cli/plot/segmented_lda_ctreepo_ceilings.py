#!/usr/bin/env python3
"""Ceiling + ablation plots for segmented-LDA end-to-end C-TreePO sweeps.

Goals:
1) Make the absolute ceiling explicit (oracle_tree has 0 distortion by construction).
2) Show ablation gains as we add components:
   truth -> oracle_proxy -> estimated_uncalibrated -> estimated_calibrated -> budgeted guidance.
3) Show that the end-to-end decomposition upper bound tracks realized error (tightness).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ceilings/ablations for segmented-LDA C-TreePO sweeps.")
    p.add_argument("--input-glob", type=str, default="outputs/segmented_lda_ctreepo/**/*.json")
    p.add_argument("--topic-phi-estimator", type=str, default="", help="Optional exact filter.")
    p.add_argument("--train-docs", type=int, default=-1, help="Optional exact filter.")
    p.add_argument(
        "--min-calibration-samples",
        type=int,
        default=-1,
        help="Optional filter: require calibration_samples >= this (default: no filter).",
    )
    p.add_argument(
        "--calibration-leaf-query-rates",
        type=str,
        default="",
        help="Optional comma/space list filter on calibration_leaf_query_rate (exact match within tolerance).",
    )
    p.add_argument(
        "--x-axis",
        choices=[
            "oracle_cost_ratio",
            "eval_internal_query_rate",
            "eval_leaf_query_rate",
            "calibration_leaf_query_rate",
            "topic_phi_l2_error_mean",
        ],
        default="oracle_cost_ratio",
    )
    p.add_argument("--aggregate", choices=["median", "mean"], default="median")
    p.add_argument("--band", choices=["none", "p10_p90", "p25_p75"], default="p10_p90")
    p.add_argument("--log-x", action="store_true")
    p.add_argument(
        "--output-figure",
        type=str,
        default="outputs/segmented_lda_ctreepo/ceilings.png",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="outputs/segmented_lda_ctreepo/ceilings_report.json",
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


def _parse_floats(text: str) -> List[float]:
    out: List[float] = []
    for raw in str(text).replace(",", " ").split():
        x = raw.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def _float_in_set(x: float, targets: List[float], *, tol: float = 1e-12) -> bool:
    if not targets:
        return True
    return any(abs(float(x) - float(t)) <= tol for t in targets)


def _extract_row(payload: dict) -> dict:
    cfg = payload.get("config", {}) or {}
    topic_meta = payload.get("topic_meta", {}) or {}
    m = payload.get("metrics", {}) or {}
    d = payload.get("decomposition", {}) or {}
    calibration_samples = int(payload.get("calibration_samples", 0) or 0)

    oracle = m.get("oracle_tree", {}) or {}
    budgeted = m.get("estimated_calibrated_budgeted", {}) or {}
    oracle_q = float(oracle.get("mean_total_queries", float("nan")))
    budget_q = float(budgeted.get("mean_total_queries", float("nan")))
    oracle_cost_ratio = float(budget_q / oracle_q) if np.isfinite(oracle_q) and oracle_q > 0 else float("nan")

    def _root_l1(policy: str) -> float:
        block = m.get(policy, {}) or {}
        return float(block.get("root_l1_mean", float("nan"))) if isinstance(block, dict) else float("nan")

    return {
        "topic_phi_estimator": str(cfg.get("topic_phi_estimator", "")),
        "train_docs": int(cfg.get("n_books_train", -1)),
        "calibration_leaf_query_rate": float(cfg.get("calibration_leaf_query_rate", float("nan"))),
        "calibration_samples": int(calibration_samples),
        "eval_internal_query_rate": float(cfg.get("eval_internal_query_rate", float("nan"))),
        "eval_leaf_query_rate": float(cfg.get("eval_leaf_query_rate", float("nan"))),
        "topic_phi_l2_error_mean": float(topic_meta.get("topic_phi_l2_error_mean", float("nan"))),
        "oracle_cost_ratio": float(oracle_cost_ratio),
        "root_l1": {
            "oracle_proxy": _root_l1("oracle_proxy"),
            "estimated_uncalibrated": _root_l1("estimated_uncalibrated"),
            "estimated_calibrated": _root_l1("estimated_calibrated"),
            "estimated_calibrated_budgeted": _root_l1("estimated_calibrated_budgeted"),
            "oracle_tree": _root_l1("oracle_tree"),
        },
        "decomposition": {
            "total": float(d.get("total_root_l1_mean", float("nan"))),
            "topic": float(d.get("topic_component_mean", float("nan"))),
            "calibration": float(d.get("calibration_component_mean", float("nan"))),
            "guidance": float(d.get("guidance_component_mean", float("nan"))),
            "oracle_proxy": float(d.get("oracle_proxy_component_mean", float("nan"))),
            "upper": float(d.get("upper_bound_mean", float("nan"))),
            "slack": float(d.get("slack_mean", float("nan"))),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = [Path(p) for p in sorted(glob.glob(str(args.input_glob), recursive=True))]
    if not paths:
        raise ValueError(f"no files matched: {args.input_glob}")

    rows = [_extract_row(json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    if str(args.topic_phi_estimator):
        rows = [r for r in rows if str(r["topic_phi_estimator"]) == str(args.topic_phi_estimator)]
    if int(args.train_docs) >= 0:
        rows = [r for r in rows if int(r["train_docs"]) == int(args.train_docs)]
    if int(args.min_calibration_samples) >= 0:
        rows = [r for r in rows if int(r.get("calibration_samples", 0)) >= int(args.min_calibration_samples)]
    cal_targets = _parse_floats(str(args.calibration_leaf_query_rates))
    if cal_targets:
        rows = [
            r
            for r in rows
            if _float_in_set(float(r.get("calibration_leaf_query_rate", float("nan"))), cal_targets)
        ]
    if not rows:
        raise ValueError("no rows after filters")

    x_key = str(args.x_axis)
    qband = _band_quantiles(str(args.band))

    by_x: Dict[float, List[dict]] = {}
    for r in rows:
        x = float(r.get(x_key, float("nan")))
        if not np.isfinite(x):
            continue
        by_x.setdefault(x, []).append(r)
    if not by_x:
        raise ValueError("no finite x values")

    xs = sorted(by_x.keys())
    policies = [
        ("oracle_proxy", "oracle proxy", "#1f77b4", "-."),
        ("estimated_uncalibrated", "estimated (uncalibrated)", "#ff7f0e", "-."),
        ("estimated_calibrated", "estimated (calibrated)", "#9467bd", "-."),
        ("estimated_calibrated_budgeted", "estimated + budgeted guidance", "#2ca02c", "-"),
    ]

    def _policy_series(policy: str) -> Tuple[List[float], List[float], List[float]]:
        ys = []
        lo = []
        hi = []
        for x in xs:
            vals = [float(rr["root_l1"].get(policy, float("nan"))) for rr in by_x[x]]
            ys.append(_reduce(vals, agg=str(args.aggregate)))
            if qband is None:
                lo.append(float("nan"))
                hi.append(float("nan"))
            else:
                lo.append(_percentile(vals, qband[0]))
                hi.append(_percentile(vals, qband[1]))
        return ys, lo, hi

    oracle_line = _reduce([float(r["root_l1"]["oracle_tree"]) for r in rows], agg=str(args.aggregate))
    full_guidance_rows = [
        r
        for r in rows
        if np.isfinite(float(r.get("eval_leaf_query_rate", float("nan"))))
        and np.isfinite(float(r.get("eval_internal_query_rate", float("nan"))))
        and abs(float(r["eval_leaf_query_rate"]) - 1.0) <= 1e-12
        and abs(float(r["eval_internal_query_rate"]) - 1.0) <= 1e-12
    ]
    full_guidance_diagnostic = {
        "present": bool(full_guidance_rows),
        "n_rows": int(len(full_guidance_rows)),
        "estimated_calibrated_budgeted_root_l1": _reduce(
            [float(r["root_l1"].get("estimated_calibrated_budgeted", float("nan"))) for r in full_guidance_rows],
            agg=str(args.aggregate),
        ),
        "oracle_tree_root_l1": _reduce(
            [float(r["root_l1"].get("oracle_tree", float("nan"))) for r in full_guidance_rows],
            agg=str(args.aggregate),
        ),
    }

    series_payload: Dict[str, dict] = {}
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.1), constrained_layout=True)
    ax0, ax1 = axes

    for key, label, color, ls in policies:
        ys, ylo, yhi = _policy_series(key)
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
        ax0.plot(xs, ys, marker="o", linewidth=1.9, color=color, linestyle=ls, label=label)
        if qband is not None and key == "estimated_calibrated_budgeted":
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

    ax0.axhline(oracle_line, color="#222222", linestyle=":", linewidth=2.0, label="oracle tree (ceiling)")
    if not bool(full_guidance_diagnostic["present"]):
        ax0.text(
            0.02,
            0.98,
            "No full-guidance runs (leaf=1, internal=1) in input",
            transform=ax0.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#b22222",
        )
    ax0.set_xlabel(x_key)
    ax0.set_ylabel("Root L1 mean")
    ax0.set_title("Ablation Gains vs Ceiling")
    if bool(args.log_x):
        ax0.set_xscale("log")
    ax0.grid(alpha=0.25)
    ax0.legend(frameon=False, fontsize=9)

    pts = [
        (
            float(r["decomposition"]["total"]),
            float(r["decomposition"]["upper"]),
            float(r.get(x_key, float("nan"))),
        )
        for r in rows
        if np.isfinite(float(r["decomposition"]["total"])) and np.isfinite(float(r["decomposition"]["upper"]))
    ]
    total_vals = [t for t, _, _ in pts]
    upper_vals = [u for _, u, _ in pts]
    color_vals = [x for _, _, x in pts]

    color_arr = np.asarray(color_vals, dtype=np.float64)
    use_color = bool(np.any(np.isfinite(color_arr)))
    if use_color:
        sc = ax1.scatter(total_vals, upper_vals, s=26, alpha=0.55, c=color_arr, cmap="viridis")
    else:
        sc = ax1.scatter(total_vals, upper_vals, s=26, alpha=0.55, color="#1f77b4")
    lo = min(total_vals + upper_vals) if total_vals and upper_vals else 0.0
    hi = max(total_vals + upper_vals) if total_vals and upper_vals else 1.0
    ax1.plot([lo, hi], [lo, hi], color="#444444", linewidth=1.2, linestyle="--", label="y=x")
    ax1.set_xlabel("Total root L1 (realized)")
    ax1.set_ylabel("Upper bound (topic+calib+guidance+proxy)")
    ax1.set_title("Decomposition Tightness")
    ax1.grid(alpha=0.25)
    if use_color:
        cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.03)
        cbar.set_label(x_key)
    ax1.legend(frameon=False, fontsize=9)

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=170)
    plt.close(fig)

    report = {
        "input_glob": str(args.input_glob),
        "n_files": int(len(paths)),
        "n_rows_after_filters": int(len(rows)),
        "filters": {
            "topic_phi_estimator": str(args.topic_phi_estimator),
            "train_docs": int(args.train_docs),
            "min_calibration_samples": int(args.min_calibration_samples),
            "calibration_leaf_query_rates": cal_targets,
        },
        "x_axis": x_key,
        "aggregate": str(args.aggregate),
        "band": str(args.band),
        "oracle_ceiling_root_l1": float(oracle_line),
        "diagnostics": {"full_guidance": full_guidance_diagnostic},
        "series": series_payload,
        "tightness": {
            "n_points": int(len(total_vals)),
            "mean_slack": float(
                _reduce(
                    [
                        float(r["decomposition"]["slack"])
                        for r in rows
                        if np.isfinite(float(r["decomposition"]["slack"]))
                    ],
                    agg="mean",
                )
            ),
            "corr_total_vs_upper": (
                float(np.corrcoef(np.asarray(total_vals), np.asarray(upper_vals))[0, 1])
                if len(total_vals) >= 3 and len(upper_vals) == len(total_vals)
                else float("nan")
            ),
        },
        "output_figure": str(out_fig),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_figure": str(out_fig), "output_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

