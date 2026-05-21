#!/usr/bin/env python3
"""Plot full-budget/full-guidance gap-to-ceiling diagnostics across simulation families."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import statistics
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot full-budget gap-to-ceiling diagnostics.")
    p.add_argument(
        "--markov-glob",
        type=str,
        default="outputs/cpu_megasweep_20260302_megasweep_paper_v2/markov_changepoint_ops_count/**/*seed_*.json",
    )
    p.add_argument(
        "--segment-glob",
        type=str,
        default="outputs/cpu_megasweep_20260302_megasweep_paper_v2/segment_lda_ops_weight_recovery/**/*seed_*.json",
    )
    p.add_argument(
        "--ctree-glob",
        type=str,
        default="outputs/cpu_megasweep_20260302_megasweep_paper_v2/segmented_lda_ctreepo/**/*.json",
    )
    p.add_argument("--aggregate", choices=["median", "mean"], default="median")
    p.add_argument(
        "--output-figure",
        type=str,
        default="outputs/full_budget_gap_suite.png",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="outputs/full_budget_gap_suite_report.json",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def _reduce(vals: List[float], agg: str) -> float:
    clean = [float(x) for x in vals if np.isfinite(float(x))]
    if not clean:
        return float("nan")
    if agg == "mean":
        return float(np.mean(np.asarray(clean, dtype=np.float64)))
    return float(statistics.median(clean))


def _q(vals: List[float], p: float) -> float:
    clean = np.asarray([float(x) for x in vals if np.isfinite(float(x))], dtype=np.float64)
    if clean.size == 0:
        return float("nan")
    return float(np.percentile(clean, float(p)))


def _collect_markov(markov_glob: str) -> Dict[int, Dict[str, List[float]]]:
    out: Dict[int, Dict[str, List[float]]] = {}
    for fp in glob.glob(markov_glob, recursive=True):
        payload = json.loads(Path(fp).read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        if abs(float(cfg.get("audit_fraction", float("nan"))) - 1.0) > 1e-12:
            continue
        td = int(cfg.get("train_docs", -1))
        m = payload.get("metrics", {}) or {}
        out.setdefault(td, {"learned": [], "exact": [], "undersupported": []})
        out[td]["learned"].append(float((m.get("learned", {}) or {}).get("root_mae", float("nan"))))
        out[td]["exact"].append(float((m.get("exact", {}) or {}).get("root_mae", float("nan"))))
        out[td]["undersupported"].append(float((m.get("undersupported", {}) or {}).get("root_mae", float("nan"))))
    return out


def _collect_segment(segment_glob: str) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    for fp in glob.glob(segment_glob, recursive=True):
        payload = json.loads(Path(fp).read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        if abs(float(cfg.get("audit_fraction", float("nan"))) - 1.0) > 1e-12:
            continue
        est = str(cfg.get("topic_phi_estimator", ""))
        m = payload.get("metrics", {}) or {}
        out.setdefault(est, {"ridge": [], "exact": [], "undersupported": []})
        out[est]["ridge"].append(float((m.get("ridge", {}) or {}).get("root_mae", float("nan"))))
        out[est]["exact"].append(float((m.get("exact", {}) or {}).get("root_mae", float("nan"))))
        out[est]["undersupported"].append(float((m.get("undersupported", {}) or {}).get("root_mae", float("nan"))))
    return out


def _collect_ctree(ctree_glob: str) -> Dict[int, Dict[str, List[float]]]:
    out: Dict[int, Dict[str, List[float]]] = {}
    for fp in glob.glob(ctree_glob, recursive=True):
        payload = json.loads(Path(fp).read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        leaf = float(cfg.get("eval_leaf_query_rate", float("nan")))
        internal = float(cfg.get("eval_internal_query_rate", float("nan")))
        if abs(leaf - 1.0) > 1e-12 or abs(internal - 1.0) > 1e-12:
            continue
        td = int(cfg.get("n_books_train", -1))
        m = payload.get("metrics", {}) or {}
        out.setdefault(td, {"budgeted": [], "oracle_tree": []})
        out[td]["budgeted"].append(
            float((m.get("estimated_calibrated_budgeted", {}) or {}).get("root_l1_mean", float("nan")))
        )
        out[td]["oracle_tree"].append(float((m.get("oracle_tree", {}) or {}).get("root_l1_mean", float("nan"))))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    agg = str(args.aggregate)

    markov = _collect_markov(str(args.markov_glob))
    segment = _collect_segment(str(args.segment_glob))
    ctree = _collect_ctree(str(args.ctree_glob))

    if not markov and not segment and not ctree:
        raise ValueError("No full-budget/full-guidance rows found in any input.")

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2), constrained_layout=True)
    ax0, ax1, ax2 = axes

    # Panel A: Markov full-audit gap to ceiling.
    markov_payload: Dict[str, object] = {}
    if markov:
        xs = sorted(markov.keys())
        learned = [_reduce(markov[x]["learned"], agg) for x in xs]
        exact = [_reduce(markov[x]["exact"], agg) for x in xs]
        floor = [_reduce(markov[x]["undersupported"], agg) for x in xs]
        ax0.plot(xs, learned, marker="o", color="#2ca02c", linewidth=2.0, label="learned")
        ax0.plot(xs, exact, marker="o", color="#222222", linestyle=":", linewidth=1.8, label="exact ceiling")
        ax0.plot(xs, floor, marker="o", color="#444444", linestyle="--", linewidth=1.8, label="undersupported floor")
        ax0.set_xlabel("train_docs")
        ax0.set_ylabel("Root MAE")
        ax0.set_title("Markov @ audit_fraction=1")
        ax0.grid(alpha=0.25)
        ax0.legend(frameon=False, fontsize=9)
        markov_payload = {
            "n_rows": int(sum(len(markov[x]["learned"]) for x in xs)),
            "train_docs": [int(x) for x in xs],
            "learned": [float(y) for y in learned],
            "exact": [float(y) for y in exact],
            "undersupported": [float(y) for y in floor],
        }
    else:
        ax0.text(0.5, 0.5, "No full-audit Markov rows", ha="center", va="center")
        ax0.set_axis_off()

    # Panel B: Segment-LDA full-audit gap by estimator.
    segment_payload: Dict[str, object] = {}
    if segment:
        estimators = sorted(segment.keys())
        est_pos = np.arange(len(estimators), dtype=np.float64)
        ridge = [_reduce(segment[e]["ridge"], agg) for e in estimators]
        exact = [_reduce(segment[e]["exact"], agg) for e in estimators]
        gap = [float(r - ex) if np.isfinite(r) and np.isfinite(ex) else float("nan") for r, ex in zip(ridge, exact)]
        bars = ax1.bar(est_pos, gap, color="#ff7f0e", alpha=0.85, label="ridge - exact")
        ax1.axhline(0.0, color="#222222", linestyle=":", linewidth=1.2)
        for b, e in zip(bars, estimators):
            vals = segment[e]["ridge"]
            yerr_lo = _reduce(vals, agg) - _q(vals, 10.0)
            yerr_hi = _q(vals, 90.0) - _reduce(vals, agg)
            if np.isfinite(yerr_lo) and np.isfinite(yerr_hi):
                ax1.errorbar(
                    [b.get_x() + b.get_width() / 2.0],
                    [b.get_height()],
                    yerr=[[max(0.0, yerr_lo)], [max(0.0, yerr_hi)]],
                    fmt="none",
                    ecolor="#333333",
                    elinewidth=1.1,
                    capsize=2,
                )
        ax1.set_xticks(est_pos)
        ax1.set_xticklabels(estimators, rotation=25, ha="right")
        ax1.set_ylabel("Gap to exact ceiling")
        ax1.set_title("Segment-LDA OPS @ audit_fraction=1")
        ax1.grid(axis="y", alpha=0.25)
        segment_payload = {
            "n_rows": int(sum(len(segment[e]["ridge"]) for e in estimators)),
            "estimators": estimators,
            "ridge": [float(v) for v in ridge],
            "exact": [float(v) for v in exact],
            "gap": [float(v) for v in gap],
        }
    else:
        ax1.text(0.5, 0.5, "No full-audit Segment-LDA rows", ha="center", va="center")
        ax1.set_axis_off()

    # Panel C: C-TreePO full-guidance gap.
    ctree_payload: Dict[str, object] = {}
    if ctree:
        xs = sorted(ctree.keys())
        budgeted = [_reduce(ctree[x]["budgeted"], agg) for x in xs]
        oracle = [_reduce(ctree[x]["oracle_tree"], agg) for x in xs]
        ax2.plot(xs, budgeted, marker="o", color="#1f77b4", linewidth=2.0, label="budgeted guidance")
        ax2.plot(xs, oracle, marker="o", color="#222222", linestyle=":", linewidth=1.8, label="oracle tree")
        ax2.set_xlabel("n_books_train")
        ax2.set_ylabel("Root L1")
        ax2.set_title("Segmented-LDA C-TreePO @ leaf=1,internal=1")
        ax2.grid(alpha=0.25)
        ax2.legend(frameon=False, fontsize=9)
        ctree_payload = {
            "n_rows": int(sum(len(ctree[x]["budgeted"]) for x in xs)),
            "train_docs": [int(x) for x in xs],
            "budgeted": [float(y) for y in budgeted],
            "oracle_tree": [float(y) for y in oracle],
        }
    else:
        ax2.text(0.5, 0.5, "No full-guidance C-TreePO rows", ha="center", va="center")
        ax2.set_axis_off()

    fig.suptitle("Full-Budget Gap to Theoretical Ceiling", fontsize=13)
    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=190)
    plt.close(fig)

    report = {
        "aggregate": agg,
        "input_globs": {
            "markov": str(args.markov_glob),
            "segment": str(args.segment_glob),
            "ctree": str(args.ctree_glob),
        },
        "markov": markov_payload,
        "segment": segment_payload,
        "ctree": ctree_payload,
        "output_figure": str(out_fig),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_figure": str(out_fig), "output_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

