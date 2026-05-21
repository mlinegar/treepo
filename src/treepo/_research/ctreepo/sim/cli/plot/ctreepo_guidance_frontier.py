#!/usr/bin/env python3
"""Plot guidance-efficiency frontier for segmented-LDA C-TreePO.

This expects per-run JSON outputs from `ctreepo sim run segmented-lda-ctreepo`.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import statistics
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot C-TreePO guidance frontier (leaf/internal query rates).")
    p.add_argument(
        "--input-glob",
        type=str,
        default="outputs/cpu_megasweep_20260302_megasweep_paper_v2/segmented_lda_ctreepo/**/*.json",
    )
    p.add_argument("--topic-phi-estimator", type=str, default="", help="Optional exact filter.")
    p.add_argument("--train-docs", type=str, default="", help="Optional comma/space list filter.")
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
    p.add_argument("--aggregate", choices=["median", "mean"], default="median")
    p.add_argument(
        "--output-figure",
        type=str,
        default="outputs/ctreepo_guidance_frontier.png",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="outputs/ctreepo_guidance_frontier_report.json",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def _parse_ints(text: str) -> List[int]:
    out: List[int] = []
    for raw in str(text).replace(",", " ").split():
        x = raw.strip()
        if not x:
            continue
        out.append(int(x))
    return out


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


def _reduce(vals: List[float], agg: str) -> float:
    clean = [float(x) for x in vals if np.isfinite(float(x))]
    if not clean:
        return float("nan")
    if agg == "mean":
        return float(np.mean(np.asarray(clean, dtype=np.float64)))
    return float(statistics.median(clean))


def _extract_row(payload: dict) -> dict:
    cfg = payload.get("config", {}) or {}
    m = payload.get("metrics", {}) or {}
    budget = m.get("estimated_calibrated_budgeted", {}) or {}
    oracle = m.get("oracle_tree", {}) or {}
    calibration_samples = int(payload.get("calibration_samples", 0) or 0)
    root_l1 = float(budget.get("root_l1_mean", float("nan")))
    oracle_l1 = float(oracle.get("root_l1_mean", float("nan")))
    budget_q = float(budget.get("mean_total_queries", float("nan")))
    oracle_q = float(oracle.get("mean_total_queries", float("nan")))
    cost_ratio = (
        budget_q / oracle_q if np.isfinite(budget_q) and np.isfinite(oracle_q) and oracle_q > 0 else float("nan")
    )
    return {
        "topic_phi_estimator": str(cfg.get("topic_phi_estimator", "")),
        "train_docs": int(cfg.get("n_books_train", -1)),
        "calibration_leaf_query_rate": float(cfg.get("calibration_leaf_query_rate", float("nan"))),
        "calibration_samples": int(calibration_samples),
        "leaf_rate": float(cfg.get("eval_leaf_query_rate", float("nan"))),
        "internal_rate": float(cfg.get("eval_internal_query_rate", float("nan"))),
        "root_l1": root_l1,
        "oracle_root_l1": oracle_l1,
        "gap_to_oracle": root_l1 - oracle_l1 if np.isfinite(root_l1) and np.isfinite(oracle_l1) else float("nan"),
        "cost_ratio": float(cost_ratio),
    }


def _grid_from_rows(rows: List[dict], value_key: str, agg: str) -> Tuple[List[float], List[float], np.ndarray]:
    leaf_vals = sorted({float(r["leaf_rate"]) for r in rows if np.isfinite(float(r["leaf_rate"]))})
    int_vals = sorted({float(r["internal_rate"]) for r in rows if np.isfinite(float(r["internal_rate"]))})
    grid = np.full((len(leaf_vals), len(int_vals)), np.nan, dtype=np.float64)
    for i, lr in enumerate(leaf_vals):
        for j, ir in enumerate(int_vals):
            vals = [
                float(r[value_key])
                for r in rows
                if np.isfinite(float(r["leaf_rate"]))
                and np.isfinite(float(r["internal_rate"]))
                and abs(float(r["leaf_rate"]) - lr) <= 1e-12
                and abs(float(r["internal_rate"]) - ir) <= 1e-12
            ]
            grid[i, j] = _reduce(vals, agg)
    return leaf_vals, int_vals, grid


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    agg = str(args.aggregate)

    paths = glob.glob(str(args.input_glob), recursive=True)
    rows = [_extract_row(json.loads(Path(p).read_text(encoding="utf-8"))) for p in paths]
    if str(args.topic_phi_estimator):
        rows = [r for r in rows if str(r["topic_phi_estimator"]) == str(args.topic_phi_estimator)]
    td_filter = _parse_ints(str(args.train_docs))
    if td_filter:
        td_set = set(td_filter)
        rows = [r for r in rows if int(r["train_docs"]) in td_set]
    if int(args.min_calibration_samples) >= 0:
        rows = [r for r in rows if int(r.get("calibration_samples", 0)) >= int(args.min_calibration_samples)]
    cal_targets = _parse_floats(str(args.calibration_leaf_query_rates))
    if cal_targets:
        rows = [
            r
            for r in rows
            if _float_in_set(float(r.get("calibration_leaf_query_rate", float("nan"))), cal_targets)
        ]
    rows = [r for r in rows if np.isfinite(float(r["leaf_rate"])) and np.isfinite(float(r["internal_rate"]))]
    if not rows:
        raise ValueError("No rows available after filters.")

    leaf_rates, internal_rates, gap_grid = _grid_from_rows(rows, "gap_to_oracle", agg)
    _, _, cost_grid = _grid_from_rows(rows, "cost_ratio", agg)

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.6), constrained_layout=True)
    ax0, ax1 = axes

    im0 = ax0.imshow(
        gap_grid,
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
        extent=[min(internal_rates), max(internal_rates), min(leaf_rates), max(leaf_rates)],
    )
    ax0.set_xlabel("eval_internal_query_rate", fontsize=12)
    ax0.set_ylabel("eval_leaf_query_rate", fontsize=12)
    ax0.set_title(f"Gap to Oracle ({agg})", fontsize=14)
    ax0.tick_params(axis="both", labelsize=11)
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03)
    cbar0.set_label("budgeted_root_l1 - oracle_root_l1", fontsize=11)
    cbar0.ax.tick_params(labelsize=10)
    if np.isfinite(gap_grid).any():
        xx, yy = np.meshgrid(np.asarray(internal_rates, dtype=np.float64), np.asarray(leaf_rates, dtype=np.float64))
        try:
            cs = ax0.contour(xx, yy, gap_grid, levels=6, colors="white", linewidths=1.0, alpha=0.85)
            ax0.clabel(cs, fmt="%.3g", fontsize=9, inline=True)
        except Exception:
            pass

    im1 = ax1.imshow(
        cost_grid,
        origin="lower",
        aspect="auto",
        cmap="magma_r",
        extent=[min(internal_rates), max(internal_rates), min(leaf_rates), max(leaf_rates)],
    )
    ax1.set_xlabel("eval_internal_query_rate", fontsize=12)
    ax1.set_ylabel("eval_leaf_query_rate", fontsize=12)
    ax1.set_title(f"Oracle Cost Ratio ({agg})", fontsize=14)
    ax1.tick_params(axis="both", labelsize=11)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cbar1.set_label("mean_total_queries / oracle_queries", fontsize=11)
    cbar1.ax.tick_params(labelsize=10)

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)

    report = {
        "input_glob": str(args.input_glob),
        "aggregate": agg,
        "filters": {
            "topic_phi_estimator": str(args.topic_phi_estimator),
            "train_docs": td_filter,
            "min_calibration_samples": int(args.min_calibration_samples),
            "calibration_leaf_query_rates": cal_targets,
        },
        "n_rows_after_filters": int(len(rows)),
        "leaf_rates": [float(x) for x in leaf_rates],
        "internal_rates": [float(x) for x in internal_rates],
        "gap_grid": gap_grid.tolist(),
        "cost_grid": cost_grid.tolist(),
        "output_figure": str(out_fig),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_figure": str(out_fig), "output_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

