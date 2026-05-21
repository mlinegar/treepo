#!/usr/bin/env python3
"""Phase-style heatmap for segmented-LDA C-TreePO sweeps."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import fmean, median
from typing import Dict, List, Sequence

np = None
plt = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot segmented-LDA phase heatmap from per-run JSON outputs.")
    p.add_argument("--input-glob", type=str, default="outputs/segmented_lda_ctreepo/**/*.json")
    p.add_argument("--topic-phi-estimator", type=str, default="", help="Optional filter.")
    p.add_argument("--train-docs", type=int, default=-1, help="Optional exact filter.")
    p.add_argument("--metric", type=str, default="decomposition_total_root_l1_mean")
    p.add_argument("--aggregate", choices=["mean", "median"], default="median")
    p.add_argument(
        "--output-figure",
        type=str,
        default="outputs/segmented_lda_ctreepo/phase.png",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="outputs/segmented_lda_ctreepo/phase_report.json",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def _agg(xs: List[float], kind: str) -> float:
    vals = [float(x) for x in xs if np.isfinite(float(x))]
    if not vals:
        return float("nan")
    if kind == "mean":
        return float(fmean(vals))
    return float(median(vals))


def _extract_metric(payload: dict, name: str) -> float:
    m = payload.get("metrics", {}) or {}
    d = payload.get("decomposition", {}) or {}
    pol = m.get("estimated_calibrated_budgeted", {}) or {}

    table = {
        "decomposition_total_root_l1_mean": d.get("total_root_l1_mean", float("nan")),
        "decomposition_topic_component_mean": d.get("topic_component_mean", float("nan")),
        "decomposition_calibration_component_mean": d.get("calibration_component_mean", float("nan")),
        "decomposition_guidance_component_mean": d.get("guidance_component_mean", float("nan")),
        "decomposition_oracle_proxy_component_mean": d.get("oracle_proxy_component_mean", float("nan")),
        "decomposition_upper_bound_mean": d.get("upper_bound_mean", float("nan")),
        "decomposition_slack_mean": d.get("slack_mean", float("nan")),
        "budgeted_root_l1_mean": pol.get("root_l1_mean", float("nan")),
        "budgeted_c3_violation_rate": pol.get("c3_violation_rate", float("nan")),
        "budgeted_c1_violation_rate": pol.get("c1_violation_rate", float("nan")),
        "budgeted_mean_total_queries": pol.get("mean_total_queries", float("nan")),
    }
    return float(table.get(name, float("nan")))


def _load_rows(paths: List[Path], *, metric: str) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        m = payload.get("metrics", {}) or {}
        oracle = m.get("oracle_tree", {}) or {}
        budgeted = m.get("estimated_calibrated_budgeted", {}) or {}
        oracle_q = float(oracle.get("mean_total_queries", float("nan")))
        budget_q = float(budgeted.get("mean_total_queries", float("nan")))
        rows.append(
            {
                "path": str(path),
                "mode": str(cfg.get("topic_phi_estimator", "")),
                "train_docs": int(cfg.get("n_books_train", -1)),
                "calibration_leaf_query_rate": float(cfg.get("calibration_leaf_query_rate", float("nan"))),
                "eval_internal_query_rate": float(cfg.get("eval_internal_query_rate", float("nan"))),
                "oracle_cost_ratio": float(budget_q / oracle_q)
                if np.isfinite(oracle_q) and oracle_q > 0
                else float("nan"),
                "metric": _extract_metric(payload, metric),
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    global np, plt
    import numpy as np  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    paths = [Path(p) for p in sorted(glob.glob(str(args.input_glob), recursive=True))]
    if not paths:
        raise ValueError(f"no files matched: {args.input_glob}")

    rows = _load_rows(paths, metric=str(args.metric))
    if args.topic_phi_estimator:
        rows = [r for r in rows if str(r["mode"]) == str(args.topic_phi_estimator)]
    if int(args.train_docs) >= 0:
        rows = [r for r in rows if int(r["train_docs"]) == int(args.train_docs)]
    if not rows:
        raise ValueError("no rows after filters")

    x_vals = sorted(
        {float(r["eval_internal_query_rate"]) for r in rows if np.isfinite(float(r["eval_internal_query_rate"]))}
    )
    y_vals = sorted(
        {float(r["calibration_leaf_query_rate"]) for r in rows if np.isfinite(float(r["calibration_leaf_query_rate"]))}
    )
    if not x_vals or not y_vals:
        raise ValueError("missing x/y values for heatmap")

    z = np.full((len(y_vals), len(x_vals)), np.nan, dtype=np.float64)
    n = np.zeros((len(y_vals), len(x_vals)), dtype=np.int64)
    agg_map: Dict[str, Dict[str, Dict[str, float]]] = {}

    for iy, y in enumerate(y_vals):
        key_y = f"{y:.6g}"
        agg_map[key_y] = {}
        for ix, x in enumerate(x_vals):
            vals = [
                float(r["metric"])
                for r in rows
                if abs(float(r["calibration_leaf_query_rate"]) - y) <= 1e-12
                and abs(float(r["eval_internal_query_rate"]) - x) <= 1e-12
                and np.isfinite(float(r["metric"]))
            ]
            z[iy, ix] = _agg(vals, str(args.aggregate))
            n[iy, ix] = int(len(vals))
            agg_map[key_y][f"{x:.6g}"] = {"n": int(len(vals)), str(args.aggregate): float(z[iy, ix])}

    fig, ax = plt.subplots(figsize=(9.8, 6.3), constrained_layout=True)
    im = ax.imshow(np.ma.masked_invalid(z), origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xlabel("eval_internal_query_rate")
    ax.set_ylabel("calibration_leaf_query_rate")
    ax.set_xticks(list(range(len(x_vals))))
    ax.set_xticklabels([f"{x:.3g}" for x in x_vals], rotation=45, ha="right")
    ax.set_yticks(list(range(len(y_vals))))
    ax.set_yticklabels([f"{y:.3g}" for y in y_vals])
    title_parts = [f"metric={args.metric}", f"aggregate={args.aggregate}"]
    if args.topic_phi_estimator:
        title_parts.append(f"phi={args.topic_phi_estimator}")
    if int(args.train_docs) >= 0:
        title_parts.append(f"train_docs={int(args.train_docs)}")
    ax.set_title("Segmented-LDA Phase | " + ", ".join(title_parts))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(str(args.metric))

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=170)
    plt.close(fig)

    report = {
        "input_glob": str(args.input_glob),
        "n_files": int(len(paths)),
        "n_rows_after_filters": int(len(rows)),
        "metric": str(args.metric),
        "aggregate": str(args.aggregate),
        "topic_phi_estimator": str(args.topic_phi_estimator),
        "train_docs": int(args.train_docs),
        "x_values": x_vals,
        "y_values": y_vals,
        "aggregated": agg_map,
        "output_figure": str(out_fig),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"output_figure": str(out_fig), "output_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

