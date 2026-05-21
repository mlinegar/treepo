#!/usr/bin/env python3
"""Readable focus plot for Segment-LDA oracle-gap behavior.

Produces contour-overlaid heatmaps with large text so we can inspect:
1) absolute ridge root MAE,
2) inference gap to the true-topic ridge ceiling.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import statistics
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot readable Segment-LDA oracle-gap focus heatmaps.")
    p.add_argument(
        "--input-glob",
        type=str,
        default="outputs/segment_lda_ops_weight_recovery/**/*seed_*.json",
    )
    p.add_argument("--topic-phi-estimator", type=str, default="true")
    p.add_argument(
        "--lambda-multipliers",
        type=str,
        default="",
        help="Optional comma/space list filter on lambda_multiplier (exact match within tolerance).",
    )
    p.add_argument("--aggregate", choices=["median", "mean"], default="median")
    p.add_argument(
        "--output-figure",
        type=str,
        default="outputs/segment_lda_oracle_gap_focus.png",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="outputs/segment_lda_oracle_gap_focus_report.json",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def _reduce(vals: List[float], agg: str) -> float:
    clean = [float(x) for x in vals if np.isfinite(float(x))]
    if not clean:
        return float("nan")
    if agg == "mean":
        return float(np.mean(np.asarray(clean, dtype=np.float64)))
    return float(statistics.median(clean))


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


def _collect_rows(input_glob: str, estimator: str) -> List[dict]:
    rows: List[dict] = []
    for fp in sorted(glob.glob(str(input_glob), recursive=True)):
        payload = json.loads(Path(fp).read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        if str(cfg.get("topic_phi_estimator", "")) != str(estimator):
            continue
        metrics = payload.get("metrics", {}) or {}
        ridge = metrics.get("ridge", {}) or {}
        ridge_true = metrics.get("ridge_true_topics", {}) or {}
        exact = metrics.get("exact", {}) or {}
        rows.append(
            {
                "train_docs": int(cfg.get("train_docs", -1)),
                "audit_fraction": float(cfg.get("audit_fraction", float("nan"))),
                "lambda_multiplier": float(cfg.get("lambda_multiplier", float("nan"))),
                "ridge_root_mae": float(ridge.get("root_mae", float("nan"))),
                "ridge_true_root_mae": float(ridge_true.get("root_mae", float("nan"))),
                "exact_root_mae": float(exact.get("root_mae", float("nan"))),
            }
        )
    return rows


def _grid(
    rows: List[dict],
    *,
    train_docs: List[int],
    audits: List[float],
    lam: float,
    key: str,
    agg: str,
) -> np.ndarray:
    z = np.full((len(audits), len(train_docs)), np.nan, dtype=np.float64)
    for iy, af in enumerate(audits):
        for ix, td in enumerate(train_docs):
            vals = [
                float(r[key])
                for r in rows
                if int(r["train_docs"]) == int(td)
                and abs(float(r["audit_fraction"]) - float(af)) <= 1e-12
                and abs(float(r["lambda_multiplier"]) - float(lam)) <= 1e-12
            ]
            z[iy, ix] = _reduce(vals, agg)
    return z


def _plot_heat(
    ax: plt.Axes,
    z: np.ndarray,
    *,
    title: str,
    train_docs: List[int],
    audits: List[float],
    cmap: str,
    full_audit_idx: int | None,
) -> None:
    im = ax.imshow(np.ma.masked_invalid(z), origin="lower", aspect="auto", cmap=cmap)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(list(range(len(train_docs))))
    ax.set_xticklabels([str(x) for x in train_docs], rotation=45, ha="right", fontsize=10)
    ax.set_yticks(list(range(len(audits))))
    ax.set_yticklabels([f"{a:g}" for a in audits], fontsize=10)
    ax.set_xlabel("train_docs", fontsize=11)
    ax.set_ylabel("audit_fraction", fontsize=11)

    finite = np.asarray(z, dtype=np.float64)
    valid = np.isfinite(finite)
    if np.any(valid):
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        if hi > lo:
            levels = np.linspace(lo, hi, 6)
            yy, xx = np.mgrid[0 : z.shape[0], 0 : z.shape[1]]
            cs = ax.contour(xx, yy, finite, levels=levels, colors="white", linewidths=1.1, alpha=0.9)
            ax.clabel(cs, fmt="%.2g", fontsize=8, inline=True)

    if full_audit_idx is not None:
        ax.axhline(float(full_audit_idx), color="white", linestyle=":", linewidth=1.4, alpha=0.95)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=9)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = _collect_rows(str(args.input_glob), estimator=str(args.topic_phi_estimator))
    if not rows:
        raise ValueError("no rows matched filters")

    lam_targets = _parse_floats(str(args.lambda_multipliers))
    if lam_targets:
        rows = [
            r
            for r in rows
            if np.isfinite(float(r.get("lambda_multiplier", float("nan"))))
            and _float_in_set(float(r["lambda_multiplier"]), lam_targets)
        ]
        if not rows:
            raise ValueError("no rows matched lambda_multipliers filter")

    train_docs = sorted({int(r["train_docs"]) for r in rows if int(r["train_docs"]) > 0})
    audits = sorted({float(r["audit_fraction"]) for r in rows if np.isfinite(float(r["audit_fraction"]))})
    lambdas = sorted({float(r["lambda_multiplier"]) for r in rows if np.isfinite(float(r["lambda_multiplier"]))})
    if not train_docs or not audits or not lambdas:
        raise ValueError("insufficient finite axes for plotting")

    try:
        full_audit_idx = audits.index(1.0)
    except ValueError:
        full_audit_idx = None

    ridge_by_lam: Dict[str, List[List[float]]] = {}
    infgap_by_lam: Dict[str, List[List[float]]] = {}
    exactgap_by_lam: Dict[str, List[List[float]]] = {}
    log_exactgap_by_lam: Dict[str, List[List[float]]] = {}
    full_audit_by_lam: Dict[str, Dict[str, Dict[str, float]]] = {}

    for lam in lambdas:
        ridge = _grid(
            rows,
            train_docs=train_docs,
            audits=audits,
            lam=float(lam),
            key="ridge_root_mae",
            agg=str(args.aggregate),
        )
        ridge_true = _grid(
            rows,
            train_docs=train_docs,
            audits=audits,
            lam=float(lam),
            key="ridge_true_root_mae",
            agg=str(args.aggregate),
        )
        exact = _grid(
            rows,
            train_docs=train_docs,
            audits=audits,
            lam=float(lam),
            key="exact_root_mae",
            agg=str(args.aggregate),
        )
        inf_gap = ridge - ridge_true
        exact_gap = ridge - exact
        log_exact_gap = np.log10(np.maximum(exact_gap, 1e-12))
        k = f"{lam:g}"
        ridge_by_lam[k] = ridge.tolist()
        infgap_by_lam[k] = inf_gap.tolist()
        exactgap_by_lam[k] = exact_gap.tolist()
        log_exactgap_by_lam[k] = log_exact_gap.tolist()

        fa: Dict[str, Dict[str, float]] = {}
        if full_audit_idx is not None:
            for ix, td in enumerate(train_docs):
                fa[str(td)] = {
                    "ridge": float(ridge[full_audit_idx, ix]),
                    "ridge_true_topics": float(ridge_true[full_audit_idx, ix]),
                    "exact": float(exact[full_audit_idx, ix]),
                    "inference_gap": float(inf_gap[full_audit_idx, ix]),
                    "gap_to_exact": float(exact_gap[full_audit_idx, ix]),
                }
        full_audit_by_lam[k] = fa

    ncol = len(lambdas)
    fig, axes = plt.subplots(2, ncol, figsize=(6.0 * ncol + 1.4, 10.0), constrained_layout=True)
    if ncol == 1:
        axes = np.asarray([[axes[0]], [axes[1]]], dtype=object)

    for j, lam in enumerate(lambdas):
        k = f"{lam:g}"
        ridge = np.asarray(ridge_by_lam[k], dtype=np.float64)
        log_exact_gap = np.asarray(log_exactgap_by_lam[k], dtype=np.float64)

        _plot_heat(
            axes[0, j],
            ridge,
            title=f"lambda={lam:g} | ridge root MAE",
            train_docs=train_docs,
            audits=audits,
            cmap="viridis_r",
            full_audit_idx=full_audit_idx,
        )
        _plot_heat(
            axes[1, j],
            log_exact_gap,
            title=f"lambda={lam:g} | log10(ridge - exact + 1e-12)",
            train_docs=train_docs,
            audits=audits,
            cmap="magma_r",
            full_audit_idx=full_audit_idx,
        )

    fig.suptitle(
        f"Segment-LDA Focus (phi={args.topic_phi_estimator}) | aggregate={args.aggregate}",
        fontsize=16,
        y=1.01,
    )
    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)

    report = {
        "input_glob": str(args.input_glob),
        "topic_phi_estimator": str(args.topic_phi_estimator),
        "lambda_multipliers": lam_targets,
        "aggregate": str(args.aggregate),
        "n_rows": int(len(rows)),
        "train_docs": [int(x) for x in train_docs],
        "audit_fractions": [float(x) for x in audits],
        "lambdas": [float(x) for x in lambdas],
        "full_audit_index": (int(full_audit_idx) if full_audit_idx is not None else None),
        "ridge_root_mae_by_lambda": ridge_by_lam,
        "inference_gap_by_lambda": infgap_by_lam,
        "gap_to_exact_by_lambda": exactgap_by_lam,
        "log10_gap_to_exact_by_lambda": log_exactgap_by_lam,
        "full_audit_by_lambda_train_docs": full_audit_by_lam,
        "output_figure": str(out_fig),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_figure": str(out_fig), "output_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

