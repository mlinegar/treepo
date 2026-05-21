#!/usr/bin/env python3
"""Plot a train_docs × internal-label-budget grid for Segment-LDA OPS weight recovery.

This expects per-run JSON outputs from `ctreepo sim run segment-lda-ops`.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Segment-LDA OPS weight-recovery grid.")
    parser.add_argument(
        "--input-glob",
        type=str,
        default="outputs/segment_lda_ops_weight_recovery/**/*seed_*.json",
        help="Glob for per-run JSON outputs.",
    )
    parser.add_argument(
        "--ridge-key",
        type=str,
        default="ridge",
        help="Which metrics entry to plot (e.g. ridge, ridge_true_topics, ridge_infer_true_phi, ridge_infer_est_phi).",
    )
    parser.add_argument(
        "--audit-strategy",
        type=str,
        default="random",
        help="Filter to this audit_strategy (e.g. random, active_small).",
    )
    parser.add_argument(
        "--topic-phi-estimator",
        type=str,
        default=None,
        help="Optional filter on cfg_topic_phi_estimator (e.g. true, noisy_theory, neural_hybrid).",
    )
    parser.add_argument(
        "--topic-phi-docs",
        type=int,
        default=None,
        help="Optional filter on cfg_topic_phi_docs (<=0 means 'use train_docs' in the sim).",
    )
    parser.add_argument(
        "--oracle-noise-std",
        type=float,
        default=None,
        help="Optional filter on cfg_oracle_noise_std. If omitted and multiple noise levels are "
        "present in the input, the script will error to avoid mixing runs.",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default="outputs/segment_lda_ops_weight_recovery_grid.png",
        help="Output PNG figure path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/segment_lda_ops_weight_recovery_grid_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--include-design",
        action="store_true",
        help="Also plot ridge design diagnostics (rank/conditioning) if present in the JSON.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _collect_rows(
    files: List[Path],
    *,
    ridge_key: str,
    audit_strategy: str,
    topic_phi_estimator: str | None,
    topic_phi_docs: int | None,
) -> List[dict]:
    rows: List[dict] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        geom = payload.get("training_geometry", {})
        metrics = payload.get("metrics", {})
        ridge = metrics.get(str(ridge_key), {}) if isinstance(metrics, dict) else {}
        if not isinstance(ridge, dict):
            continue

        strat = str(cfg.get("audit_strategy", ""))
        if audit_strategy and str(strat) != str(audit_strategy):
            continue

        if topic_phi_estimator is not None:
            if str(cfg.get("topic_phi_estimator", "")) != str(topic_phi_estimator):
                continue
        if topic_phi_docs is not None:
            if int(cfg.get("topic_phi_docs", 0)) != int(topic_phi_docs):
                continue

        train_docs = int(cfg.get("train_docs", -1))
        seed = int(cfg.get("seed", -1))
        lam = float(cfg.get("lambda_multiplier", float("nan")))
        oracle_noise_std = float(cfg.get("oracle_noise_std", 0.0))
        mean_leaves = float(geom.get("mean_leaves", float("nan")))
        mean_internal = float(geom.get("mean_internal_labels", float("nan")))
        if train_docs <= 0 or not np.isfinite(lam) or not np.isfinite(mean_leaves) or mean_leaves <= 0:
            continue
        internal_per_leaf = float(mean_internal) / float(mean_leaves)

        rows.append(
            {
                "train_docs": int(train_docs),
                "seed": int(seed),
                "lambda_multiplier": float(lam),
                "oracle_noise_std": float(oracle_noise_std),
                "internal_per_leaf": float(internal_per_leaf),
                "ridge_root_mae": float(ridge.get("root_mae", float("nan"))),
                "ridge_theta_cosine": float(ridge.get("theta_cosine", float("nan"))),
                "ridge_bigram_cosine": float(ridge.get("bigram_cosine", float("nan"))),
                "ridge_lambda_abs_error": float(ridge.get("lambda_abs_error", float("nan"))),
                "ridge_rank_over_d": (
                    float(ridge.get("rank", float("nan"))) / float(ridge.get("d", float("nan")))
                    if float(ridge.get("d", float("nan"))) > 0
                    else float("nan")
                ),
                "ridge_log10_a_condition": (
                    float(math.log10(float(ridge.get("a_condition"))))
                    if np.isfinite(float(ridge.get("a_condition", float("nan"))))
                    and float(ridge.get("a_condition", float("nan"))) > 0
                    else float("nan")
                ),
                "ridge_train_rmse": float(ridge.get("train_rmse", float("nan"))),
            }
        )
    return rows


def _heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    *,
    xlabels: List[str],
    ylabels: List[str],
    title: str,
    cmap: str,
) -> None:
    im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="lower")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("train_docs", fontsize=9)
    ax.set_ylabel("internal labels / leaf", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    files = [Path(p) for p in sorted(glob.glob(str(args.input_glob), recursive=True))]
    if not files:
        raise ValueError(f"no files matched: {args.input_glob}")

    rows = _collect_rows(
        files,
        ridge_key=str(args.ridge_key),
        audit_strategy=str(args.audit_strategy),
        topic_phi_estimator=(str(args.topic_phi_estimator) if args.topic_phi_estimator is not None else None),
        topic_phi_docs=(int(args.topic_phi_docs) if args.topic_phi_docs is not None else None),
    )
    if not rows:
        raise ValueError("no ridge rows found (check audit_strategy filter and input_glob)")

    noise_values = sorted({float(r["oracle_noise_std"]) for r in rows if np.isfinite(float(r["oracle_noise_std"]))})
    if args.oracle_noise_std is not None:
        target = float(args.oracle_noise_std)
        rows = [r for r in rows if np.isfinite(float(r["oracle_noise_std"])) and float(r["oracle_noise_std"]) == target]
        if not rows:
            raise ValueError(f"no rows matched oracle_noise_std={target:g}")
        noise_values = [target]
    elif len(noise_values) > 1:
        raise ValueError(
            f"multiple oracle_noise_std values present ({noise_values}); pass --oracle-noise-std to filter"
        )

    train_docs_values = sorted({int(r["train_docs"]) for r in rows})
    budgets = sorted({float(r["internal_per_leaf"]) for r in rows})
    lambdas = sorted({float(r["lambda_multiplier"]) for r in rows})

    def _budget_label(x: float) -> str:
        if x < 1.0:
            return f"{x:.3f}".rstrip("0").rstrip(".") + "/leaf"
        return f"{x:.2f}".rstrip("0").rstrip(".") + "/leaf"

    xlabels = [str(x) for x in train_docs_values]
    ylabels = [_budget_label(b) for b in budgets]

    metrics_to_plot = [
        ("ridge_root_mae", "Ridge | Root MAE (↓)", "viridis_r"),
        ("ridge_theta_cosine", "Ridge | cos(theta_hat, theta_true) (↑)", "viridis"),
        ("ridge_bigram_cosine", "Ridge | cos(w_big_hat, w_big_true) (↑)", "viridis"),
        ("ridge_lambda_abs_error", "Ridge | |lambda_hat - lambda| (↓)", "viridis_r"),
    ]
    if bool(args.include_design):
        metrics_to_plot.extend(
            [
                ("ridge_rank_over_d", "Ridge | design rank / d (↑)", "viridis"),
                ("ridge_log10_a_condition", "Ridge | log10(cond(XᵀX+λI)) (↓)", "viridis_r"),
                ("ridge_train_rmse", "Ridge | train RMSE (↓)", "viridis_r"),
            ]
        )

    n_rows = len(lambdas)
    n_cols = len(metrics_to_plot)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.4 * n_cols + 1.0, 3.4 * n_rows + 1.0),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.asarray([axes])
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for li, lam in enumerate(lambdas):
        for ci, (metric, title, cmap) in enumerate(metrics_to_plot):
            mat = np.full((len(budgets), len(train_docs_values)), np.nan, dtype=np.float64)
            for bi, b in enumerate(budgets):
                for xi, td in enumerate(train_docs_values):
                    vals = [
                        float(r[metric])
                        for r in rows
                        if int(r["train_docs"]) == int(td)
                        and float(r["internal_per_leaf"]) == float(b)
                        and float(r["lambda_multiplier"]) == float(lam)
                        and np.isfinite(float(r[metric]))
                    ]
                    if vals:
                        mat[bi, xi] = float(fmean(vals))
            ax = axes[li, ci]
            masked = np.ma.masked_invalid(mat)
            _heatmap(
                ax,
                masked,
                xlabels=xlabels,
                ylabels=ylabels,
                title=f"λ={lam:g} | {title}",
                cmap=cmap,
            )

    fig.suptitle(
        f"Segment-LDA OPS Weight Recovery Grid | audit_strategy={args.audit_strategy}"
        + (f" | oracle_noise_std={noise_values[0]:g}" if noise_values else ""),
        fontsize=12,
    )

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)

    report = {
        "input_files": len(files),
        "rows": len(rows),
        "ridge_key": str(args.ridge_key),
        "audit_strategy": str(args.audit_strategy),
        "topic_phi_estimator": (str(args.topic_phi_estimator) if args.topic_phi_estimator is not None else None),
        "topic_phi_docs": (int(args.topic_phi_docs) if args.topic_phi_docs is not None else None),
        "oracle_noise_std": (float(noise_values[0]) if noise_values else None),
        "include_design": bool(args.include_design),
        "train_docs_values": train_docs_values,
        "budgets_internal_per_leaf": budgets,
        "lambda_values": lambdas,
        "metrics": {m: t for m, t, _c in metrics_to_plot},
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

