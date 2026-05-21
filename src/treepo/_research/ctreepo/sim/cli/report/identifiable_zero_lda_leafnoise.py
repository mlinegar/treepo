#!/usr/bin/env python3
"""Leaf-noise progression report for the Identifiable-Zero LDA baseline.

This report is meant to answer: does LDA learn the underlying LDA DGP, and how does
shrinking leaf sizes (fixed_leaf_tokens) degrade root estimation?

Inputs: an output root produced by `scripts/run_identifiable_zero_lda_leafnoise_overnight.sh`.
Outputs:
  - <out-dir>/identifiable_zero_lda_leafnoise_latest.md
  - <out-dir>/identifiable_zero_lda_leafnoise_latest.pdf (if pandoc+pdflatex are available)
  - <out-dir>/identifiable_zero_lda_leafnoise_latest_diagnostics.json
  - <out-dir>/pages/*.png
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors


GOOD_COLOR = "#1a9850"  # green
MID_COLOR = "#ffffbf"   # yellow
BAD_COLOR = "#d73027"   # red
# For error metrics: lower is better -> green, higher is worse -> red.
GOOD_BAD_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "good_bad_error",
    [GOOD_COLOR, MID_COLOR, BAD_COLOR],
    N=256,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a leaf-noise progression report (LDA baseline).")
    p.add_argument("--output-root", type=Path, required=True, help="Sweep output root.")
    p.add_argument(
        "--ctreepo-root",
        type=Path,
        default=None,
        help="Optional output root with a neural C-TreePO attempt to append in the same report.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <output-root>/figures/lda_leafnoise).",
    )
    p.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(list(argv) if argv is not None else None)


def _load_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _as_float(x: object) -> Optional[float]:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _median(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(statistics.median(vals))


def _fmt_cell(x: float) -> str:
    if not math.isfinite(float(x)):
        return "—"
    if float(x) == 0.0:
        return "0"
    if abs(float(x)) < 1e-3 or abs(float(x)) >= 1000.0:
        return f"{float(x):.1e}"
    return f"{float(x):.3g}"


def _heatmap(
    ax: plt.Axes,
    *,
    grid: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    title: str,
    vmin: float,
    vmax: float,
    xlabel: str,
    ylabel: str,
    cmap=GOOD_BAD_CMAP,
) -> plt.Axes:
    im = ax.imshow(grid, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(list(range(len(x_labels))), labels=list(x_labels))
    ax.set_yticks(list(range(len(y_labels))), labels=list(y_labels))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)

    ny, nx = grid.shape
    for yi in range(ny):
        for xi in range(nx):
            val = float(grid[yi, xi])
            rgba = im.cmap(im.norm(val))
            # Perceptual luminance for contrast-aware text color.
            lum = float(0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2])
            txt_color = "black" if lum >= 0.58 else "white"
            ax.text(
                xi,
                yi,
                _fmt_cell(val),
                ha="center",
                va="center",
                fontsize=9,
                color=txt_color,
            )
    return ax


def _build_grid(
    rows: Iterable[Tuple[int, int, float]],
    *,
    x_vals: Sequence[int],
    y_vals: Sequence[int],
) -> np.ndarray:
    # rows: (x, y, value)
    cell: Dict[Tuple[int, int], List[float]] = {}
    for x, y, v in rows:
        cell.setdefault((int(x), int(y)), []).append(float(v))
    grid = np.full((len(y_vals), len(x_vals)), float("nan"), dtype=np.float64)
    for yi, y in enumerate(y_vals):
        for xi, x in enumerate(x_vals):
            grid[yi, xi] = float(_median(cell.get((int(x), int(y)), [])))
    return grid


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _run_pandoc(md_path: Path, pdf_path: Path) -> bool:
    if shutil.which("pandoc") is None or shutil.which("pdflatex") is None:
        return False
    subprocess.run(
        [
            "pandoc",
            str(md_path.name),
            "-o",
            str(pdf_path.name),
            "--pdf-engine=pdflatex",
        ],
        cwd=str(md_path.parent),
        check=True,
    )
    return True


@dataclass(frozen=True)
class _Row:
    n_books_train: int
    fixed_leaf_tokens: int
    calibration_rate: float
    seed: int
    root_l1_mean: float
    topic_phi_l2_error_mean: Optional[float]
    leaf_theta_l1_mean: Optional[float]
    corpus_signature_test: Optional[str]
    dgp_key: Tuple[object, ...]


def _scan(output_root: Path) -> List[_Row]:
    rows: List[_Row] = []
    glob_pat = str(output_root / "segmented_lda_ctreepo" / "equivalence" / "**" / "*.json")
    for fp in glob.glob(glob_pat, recursive=True):
        path = Path(fp)
        payload = _load_json(path)
        if not payload:
            continue
        cfg = payload.get("config") or {}
        met = payload.get("metrics") or {}
        topic_meta = payload.get("topic_meta") or {}
        if not isinstance(cfg, dict) or not isinstance(met, dict) or not isinstance(topic_meta, dict):
            continue

        # Focus on the LDA baseline: phi=sklearn_lda, theta=sklearn_lda, q_infer=0.
        if str(cfg.get("topic_process", "")).strip().lower() != "bag_of_words":
            continue
        if str(cfg.get("topic_phi_estimator", "")).strip().lower() != "sklearn_lda":
            continue
        if str(cfg.get("leaf_theta_estimator", "")).strip().lower() != "sklearn_lda":
            continue
        if abs(float(cfg.get("eval_leaf_query_rate", 0.0)) - 0.0) > 1e-12:
            continue
        if abs(float(cfg.get("eval_internal_query_rate", 0.0)) - 0.0) > 1e-12:
            continue

        policy = met.get("estimated_calibrated_budgeted") or {}
        if not isinstance(policy, dict):
            continue

        n_books_train = int(cfg.get("n_books_train", -1))
        fixed_leaf_tokens = int(cfg.get("fixed_leaf_tokens", -1))
        cal_rate = float(cfg.get("calibration_leaf_query_rate", float("nan")))
        seed = int(cfg.get("seed", -1))
        root_l1_mean = _as_float(policy.get("root_l1_mean"))
        if n_books_train < 0 or fixed_leaf_tokens < 0 or seed < 0 or root_l1_mean is None:
            continue

        topic_phi_l2 = _as_float(topic_meta.get("topic_phi_l2_error_mean"))
        leaf_theta_l1 = _as_float(topic_meta.get("leaf_theta_l1_mean"))
        corpus_sig = topic_meta.get("corpus_signature_test")
        corpus_sig_s: Optional[str] = str(corpus_sig) if corpus_sig is not None else None

        dgp_key = (
            cfg.get("n_topics"),
            cfg.get("vocab_size"),
            cfg.get("min_segments"),
            cfg.get("max_segments"),
            cfg.get("min_seg_tokens"),
            cfg.get("max_seg_tokens"),
            cfg.get("alpha_topic"),
            cfg.get("beta_word"),
            cfg.get("segment_concentration"),
            cfg.get("segment_background"),
        )
        rows.append(
            _Row(
                n_books_train=int(n_books_train),
                fixed_leaf_tokens=int(fixed_leaf_tokens),
                calibration_rate=float(cal_rate),
                seed=int(seed),
                root_l1_mean=float(root_l1_mean),
                topic_phi_l2_error_mean=float(topic_phi_l2) if topic_phi_l2 is not None else None,
                leaf_theta_l1_mean=float(leaf_theta_l1) if leaf_theta_l1 is not None else None,
                corpus_signature_test=corpus_sig_s,
                dgp_key=dgp_key,
            )
        )
    return rows


@dataclass(frozen=True)
class _CtreeAttemptRow:
    topic_phi_estimator: str
    topic_phi_docs: int
    neural_topic_seed_fraction: Optional[float]
    seed: int
    root_l1_mean: float
    topic_phi_l2_error_mean: Optional[float]
    calibration_rate: float
    eval_leaf_rate: float
    eval_internal_rate: float


def _scan_ctreepo_attempt(ctreepo_root: Path) -> List[_CtreeAttemptRow]:
    rows: List[_CtreeAttemptRow] = []
    glob_pat = str(ctreepo_root / "segmented_lda_ctreepo" / "equivalence" / "**" / "*.json")
    for fp in glob.glob(glob_pat, recursive=True):
        path = Path(fp)
        payload = _load_json(path)
        if not payload:
            continue
        cfg = payload.get("config") or {}
        met = payload.get("metrics") or {}
        topic_meta = payload.get("topic_meta") or {}
        if not isinstance(cfg, dict) or not isinstance(met, dict) or not isinstance(topic_meta, dict):
            continue

        policy = met.get("estimated_calibrated_budgeted") or {}
        if not isinstance(policy, dict):
            continue

        est = str(cfg.get("topic_phi_estimator", "")).strip().lower()
        docs = int(cfg.get("topic_phi_docs", -1))
        seed = int(cfg.get("seed", -1))
        root_l1 = _as_float(policy.get("root_l1_mean"))
        if not est or docs < 0 or seed < 0 or root_l1 is None:
            continue

        seed_frac_raw = cfg.get("neural_topic_seed_fraction", None)
        seed_frac = _as_float(seed_frac_raw) if seed_frac_raw is not None else None
        topic_phi_l2 = _as_float(topic_meta.get("topic_phi_l2_error_mean"))
        cal = _as_float(cfg.get("calibration_leaf_query_rate"))
        lq = _as_float(cfg.get("eval_leaf_query_rate"))
        iq = _as_float(cfg.get("eval_internal_query_rate"))

        rows.append(
            _CtreeAttemptRow(
                topic_phi_estimator=est,
                topic_phi_docs=int(docs),
                neural_topic_seed_fraction=float(seed_frac) if seed_frac is not None else None,
                seed=int(seed),
                root_l1_mean=float(root_l1),
                topic_phi_l2_error_mean=float(topic_phi_l2) if topic_phi_l2 is not None else None,
                calibration_rate=float(cal) if cal is not None else float("nan"),
                eval_leaf_rate=float(lq) if lq is not None else float("nan"),
                eval_internal_rate=float(iq) if iq is not None else float("nan"),
            )
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = Path(args.output_root)
    out_dir = Path(args.out_dir) if args.out_dir is not None else (output_root / "figures" / "lda_leafnoise")
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    now = datetime.now(timezone.utc)
    rows = _scan(output_root)
    diagnostics: Dict[str, object] = {
        "generated_at_utc": now.isoformat(),
        "output_root": str(output_root),
        "n_rows": int(len(rows)),
    }

    if not rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "identifiable_zero_lda_leafnoise_latest_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0

    # Test-set stability: same seed + DGP should have identical test signature across train size sweeps.
    stability: Dict[str, List[str]] = {}
    for r in rows:
        key = str((r.seed, r.dgp_key))
        if r.corpus_signature_test:
            stability.setdefault(key, []).append(str(r.corpus_signature_test))
    n_fail = 0
    examples: List[Dict[str, object]] = []
    for k, sigs in stability.items():
        uniq = sorted(set(sigs))
        if len(uniq) > 1:
            n_fail += 1
            if len(examples) < 10:
                examples.append({"group": k, "unique_signatures": uniq[:5]})
    diagnostics["test_set_stability"] = {"n_groups": int(len(stability)), "n_fail": int(n_fail), "fail_examples": examples}

    # Assume a single DGP for this report (otherwise facet by DGP key).
    dgp_keys = sorted({r.dgp_key for r in rows}, key=str)
    diagnostics["n_dgp_keys"] = int(len(dgp_keys))
    if len(dgp_keys) > 1:
        diagnostics["dgp_keys"] = [list(k) for k in dgp_keys[:5]]

    cal_vals = sorted({float(r.calibration_rate) for r in rows})
    x_vals = sorted({int(r.n_books_train) for r in rows})
    y_vals = sorted({int(r.fixed_leaf_tokens) for r in rows})
    x_labels = [str(x) for x in x_vals]
    y_labels = [str(y) for y in y_vals]

    # Root error heatmaps (train size × leaf size), one per calibration rate.
    root_pages: List[Path] = []
    for cal in cal_vals:
        sel = [(r.n_books_train, r.fixed_leaf_tokens, r.root_l1_mean) for r in rows if abs(r.calibration_rate - cal) <= 1e-12]
        grid = _build_grid(sel, x_vals=x_vals, y_vals=y_vals)
        vals = grid[np.isfinite(grid)]
        vmin = float(np.min(vals)) if vals.size else 0.0
        vmax = float(np.max(vals)) if vals.size else 1.0
        if not math.isfinite(vmin):
            vmin = 0.0
        if not math.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        fig, ax = plt.subplots(1, 1, figsize=(12.5, 6.5), constrained_layout=True)
        _heatmap(
            ax,
            grid=grid,
            x_labels=x_labels,
            y_labels=y_labels,
            title=f"LDA baseline | calibration_rate={cal:g} | root L1 mean (median over seeds)",
            vmin=vmin,
            vmax=vmax,
            xlabel="n_books_train",
            ylabel="fixed_leaf_tokens",
        )
        cbar = fig.colorbar(ax.images[0], ax=[ax], fraction=0.040, pad=0.02)
        cbar.set_label("root_l1_mean (green=better, red=worse)")
        out_png = pages_dir / f"root_l1_mean_cal_{str(cal).replace('.','p')}.png"
        _save_fig(fig, out_png)
        root_pages.append(out_png)

    # Topic recovery curve (phi L2 error vs train size), averaged over leaf sizes and calibration rates.
    phi_rows: Dict[int, List[float]] = {x: [] for x in x_vals}
    for r in rows:
        if r.topic_phi_l2_error_mean is None:
            continue
        phi_rows[int(r.n_books_train)].append(float(r.topic_phi_l2_error_mean))
    phi_meds = [float(_median(phi_rows[x])) for x in x_vals]
    fig, ax = plt.subplots(1, 1, figsize=(11.5, 5.5), constrained_layout=True)
    ax.plot(x_vals, phi_meds, linewidth=2.0, color="#666666", zorder=2)
    finite_phi = np.asarray([v for v in phi_meds if math.isfinite(float(v))], dtype=np.float64)
    if finite_phi.size > 0:
        pmin = float(np.min(finite_phi))
        pmax = float(np.max(finite_phi))
        if not math.isfinite(pmin):
            pmin = 0.0
        if not math.isfinite(pmax) or pmax <= pmin:
            pmax = pmin + 1.0
    else:
        pmin, pmax = 0.0, 1.0
    pnorm = mcolors.Normalize(vmin=pmin, vmax=pmax)
    ax.scatter(
        x_vals,
        phi_meds,
        c=phi_meds,
        cmap=GOOD_BAD_CMAP,
        norm=pnorm,
        s=52,
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n_books_train (log2)")
    ax.set_ylabel("topic_phi_l2_error_mean (median)")
    ax.set_title("Topic recovery improves with more training books (sklearn LDA)")
    ax.grid(True, which="both", alpha=0.25)
    sm = plt.cm.ScalarMappable(norm=pnorm, cmap=GOOD_BAD_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax], fraction=0.040, pad=0.02)
    cbar.set_label("topic_phi_l2_error_mean (green=better, red=worse)")
    out_phi = pages_dir / "topic_phi_l2_error_vs_train.png"
    _save_fig(fig, out_phi)

    # Leaf-theta error heatmap (train size × leaf size) for the *proxy* (pre-calibration).
    leaf_pages: List[Path] = []
    leaf_rows = [(r.n_books_train, r.fixed_leaf_tokens, float(r.leaf_theta_l1_mean)) for r in rows if r.leaf_theta_l1_mean is not None]
    if leaf_rows:
        grid = _build_grid(leaf_rows, x_vals=x_vals, y_vals=y_vals)
        vals = grid[np.isfinite(grid)]
        vmin = float(np.min(vals)) if vals.size else 0.0
        vmax = float(np.max(vals)) if vals.size else 1.0
        if not math.isfinite(vmin):
            vmin = 0.0
        if not math.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        fig, ax = plt.subplots(1, 1, figsize=(12.5, 6.5), constrained_layout=True)
        _heatmap(
            ax,
            grid=grid,
            x_labels=x_labels,
            y_labels=y_labels,
            title="LDA baseline | test leaf theta L1 mean (proxy estimator; median over seeds)",
            vmin=vmin,
            vmax=vmax,
            xlabel="n_books_train",
            ylabel="fixed_leaf_tokens",
        )
        cbar = fig.colorbar(ax.images[0], ax=[ax], fraction=0.040, pad=0.02)
        cbar.set_label("leaf_theta_l1_mean (green=better, red=worse)")
        out_png = pages_dir / "leaf_theta_l1_mean.png"
        _save_fig(fig, out_png)
        leaf_pages.append(out_png)

    # Optional section: append a separate C-TreePO attempt root (e.g., neural_ctreepo sweeps).
    ctree_rows: List[_CtreeAttemptRow] = []
    ctree_pages: List[Path] = []
    if args.ctreepo_root is not None:
        ctree_rows = _scan_ctreepo_attempt(Path(args.ctreepo_root))
        ctree_q0 = [
            r
            for r in ctree_rows
            if abs(float(r.eval_leaf_rate) - 0.0) <= 1e-12 and abs(float(r.eval_internal_rate) - 0.0) <= 1e-12
        ]
        docs_q0 = sorted({int(r.topic_phi_docs) for r in ctree_q0})
        seeds_q0 = sorted({int(r.seed) for r in ctree_q0})
        est_counts: Dict[str, int] = {}
        for r in ctree_q0:
            est_counts[r.topic_phi_estimator] = int(est_counts.get(r.topic_phi_estimator, 0) + 1)
        est_missing: Dict[str, int] = {}
        for est in sorted(est_counts):
            est_rows = [r for r in ctree_q0 if r.topic_phi_estimator == est]
            if est == "neural_ctreepo":
                sfs = sorted({float(r.neural_topic_seed_fraction) for r in est_rows if r.neural_topic_seed_fraction is not None})
                expected = {(int(d), float(sf), int(s)) for d in docs_q0 for sf in sfs for s in seeds_q0}
                got = {
                    (int(r.topic_phi_docs), float(r.neural_topic_seed_fraction), int(r.seed))
                    for r in est_rows
                    if r.neural_topic_seed_fraction is not None
                }
            else:
                expected = {(int(d), int(s)) for d in docs_q0 for s in seeds_q0}
                got = {(int(r.topic_phi_docs), int(r.seed)) for r in est_rows}
            est_missing[est] = int(len(expected - got))
        diagnostics["ctreepo_attempt"] = {
            "ctreepo_root": str(args.ctreepo_root),
            "n_rows_all": int(len(ctree_rows)),
            "n_rows_q0": int(len(ctree_q0)),
            "estimators_q0": est_counts,
            "missing_cartesian_q0": est_missing,
            "topic_phi_docs_q0": docs_q0,
            "neural_seed_fractions_q0": sorted(
                {
                    float(r.neural_topic_seed_fraction)
                    for r in ctree_q0
                    if r.topic_phi_estimator == "neural_ctreepo" and r.neural_topic_seed_fraction is not None
                }
            ),
        }

        if ctree_q0:
            docs_vals = sorted({int(r.topic_phi_docs) for r in ctree_q0})
            docs_labels = [str(d) for d in docs_vals]

            # Figure A: root error vs topic_phi_docs by estimator (+ pooled neural_ctreepo line).
            non_neural = sorted(
                {
                    r.topic_phi_estimator
                    for r in ctree_q0
                    if not str(r.topic_phi_estimator).startswith("neural_")
                }
            )
            pooled_neural = [r for r in ctree_q0 if r.topic_phi_estimator == "neural_ctreepo"]
            plot_groups: List[Tuple[str, Dict[int, float], Dict[int, List[float]]]] = []

            for est in non_neural:
                raw: Dict[int, List[float]] = {}
                for r in ctree_q0:
                    if r.topic_phi_estimator != est:
                        continue
                    raw.setdefault(int(r.topic_phi_docs), []).append(float(r.root_l1_mean))
                med = {d: float(_median(raw.get(d, []))) for d in docs_vals}
                plot_groups.append((est, med, raw))

            if pooled_neural:
                raw_n: Dict[int, List[float]] = {}
                for r in pooled_neural:
                    raw_n.setdefault(int(r.topic_phi_docs), []).append(float(r.root_l1_mean))
                med_n = {d: float(_median(raw_n.get(d, []))) for d in docs_vals}
                plot_groups.append(("neural_ctreepo (pooled seed_fracs)", med_n, raw_n))

            if plot_groups:
                all_vals: List[float] = []
                for _name, _med, raw in plot_groups:
                    for d in docs_vals:
                        all_vals.extend([v for v in raw.get(d, []) if math.isfinite(float(v))])
                if all_vals:
                    vmin = float(np.min(np.asarray(all_vals, dtype=np.float64)))
                    vmax = float(np.max(np.asarray(all_vals, dtype=np.float64)))
                else:
                    vmin, vmax = 0.0, 1.0
                if not math.isfinite(vmax) or vmax <= vmin:
                    vmax = vmin + 1.0
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

                fig, ax = plt.subplots(1, 1, figsize=(12.5, 6.5), constrained_layout=True)
                line_colors = [
                    "#1f77b4",
                    "#ff7f0e",
                    "#2ca02c",
                    "#8c564b",
                    "#17becf",
                    "#7f7f7f",
                ]
                for i, (name, med, _raw) in enumerate(plot_groups):
                    ys = [float(med.get(d, float("nan"))) for d in docs_vals]
                    finite_idx = [j for j, y in enumerate(ys) if math.isfinite(float(y))]
                    if not finite_idx:
                        continue
                    x_f = [docs_vals[j] for j in finite_idx]
                    y_f = [ys[j] for j in finite_idx]
                    ax.plot(
                        x_f,
                        y_f,
                        marker="o",
                        linewidth=2.2,
                        color=line_colors[i % len(line_colors)],
                        label=name,
                        alpha=0.95,
                    )
                    ax.scatter(
                        x_f,
                        y_f,
                        c=y_f,
                        cmap=GOOD_BAD_CMAP,
                        norm=norm,
                        s=58,
                        edgecolors="black",
                        linewidths=0.5,
                        zorder=3,
                    )

                ax.set_xscale("log", base=2)
                ax.set_xticks(docs_vals, labels=docs_labels)
                ax.set_xlabel("topic_phi_docs (log2)")
                ax.set_ylabel("root_l1_mean (median over seeds)")
                ax.set_title("C-TreePO attempt lane | q=0 | root error vs topic-phi docs")
                ax.grid(True, which="both", alpha=0.25)
                ax.legend(loc="best", fontsize=10)
                sm = plt.cm.ScalarMappable(norm=norm, cmap=GOOD_BAD_CMAP)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=[ax], fraction=0.040, pad=0.02)
                cbar.set_label("root_l1_mean (green=better, red=worse)")
                out_png = pages_dir / "ctree_attempt_root_l1_vs_topic_phi_docs.png"
                _save_fig(fig, out_png)
                ctree_pages.append(out_png)

            # Figure B: neural_ctreepo heatmap across (topic_phi_docs x seed_fraction), if available.
            neural_rows = [
                r for r in ctree_q0 if r.topic_phi_estimator == "neural_ctreepo" and r.neural_topic_seed_fraction is not None
            ]
            if neural_rows:
                x_docs = sorted({int(r.topic_phi_docs) for r in neural_rows})
                y_sfs = sorted({float(r.neural_topic_seed_fraction) for r in neural_rows})
                grid = np.full((len(y_sfs), len(x_docs)), float("nan"), dtype=np.float64)
                for yi, sf in enumerate(y_sfs):
                    for xi, d in enumerate(x_docs):
                        vals = [
                            float(r.root_l1_mean)
                            for r in neural_rows
                            if int(r.topic_phi_docs) == int(d)
                            and r.neural_topic_seed_fraction is not None
                            and abs(float(r.neural_topic_seed_fraction) - float(sf)) <= 1e-12
                        ]
                        grid[yi, xi] = float(_median(vals))
                vals = grid[np.isfinite(grid)]
                vmin = float(np.min(vals)) if vals.size else 0.0
                vmax = float(np.max(vals)) if vals.size else 1.0
                if not math.isfinite(vmin):
                    vmin = 0.0
                if not math.isfinite(vmax) or vmax <= vmin:
                    vmax = vmin + 1.0
                fig, ax = plt.subplots(1, 1, figsize=(12.5, 6.5), constrained_layout=True)
                _heatmap(
                    ax,
                    grid=grid,
                    x_labels=[str(d) for d in x_docs],
                    y_labels=[f"{sf:g}" for sf in y_sfs],
                    title="neural_ctreepo attempt | q=0 | root L1 by docs × seed fraction",
                    vmin=vmin,
                    vmax=vmax,
                    xlabel="topic_phi_docs",
                    ylabel="neural_topic_seed_fraction",
                )
                cbar = fig.colorbar(ax.images[0], ax=[ax], fraction=0.040, pad=0.02)
                cbar.set_label("root_l1_mean (green=better, red=worse)")
                out_png = pages_dir / "ctree_attempt_neural_root_l1_heatmap.png"
                _save_fig(fig, out_png)
                ctree_pages.append(out_png)

    diag_path = out_dir / "identifiable_zero_lda_leafnoise_latest_diagnostics.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_path = out_dir / "identifiable_zero_lda_leafnoise_latest.md"
    pdf_path = out_dir / "identifiable_zero_lda_leafnoise_latest.pdf"

    def img(p: Path) -> str:
        rel = p.relative_to(out_dir)
        return f"![]({str(rel)}){{ width=100% }}"

    md_lines: List[str] = []
    md_lines.extend(
        [
            "---",
            "title: Identifiable-Zero LDA Leaf-Noise Progression (v1)",
            f"date: {now.strftime('%Y-%m-%d')}",
            "fontsize: 12pt",
            "geometry: margin=0.7in",
            "---",
            "",
            f"**Output root:** `{output_root}`  ",
            f"**Generated (UTC):** `{now.isoformat()}`",
            (
                f"**C-TreePO attempt root:** `{args.ctreepo_root}`"
                if args.ctreepo_root is not None
                else "**C-TreePO attempt root:** `(not provided)`"
            ),
            "",
            "## Setup (pedantically explicit)",
            "- Color convention (all error plots): **green = better (lower error), red = worse (higher error)**.",
            "- DGP: `topic_process=bag_of_words` (standard LDA bag-of-words generator).",
            "- Varied: `n_books_train` (more training books) and `fixed_leaf_tokens` (leaf size; smaller = noisier leaf documents).",
            "- Held-out evaluation: fixed `n_books_test` per seed; test set is generated from a dedicated RNG stream so it is stable across `n_books_train` sweeps.",
            "- Method: scikit-learn variational Bayes LDA fit on training *books*; leaf topic mixtures inferred via `transform()` on leaf DTMs; merged to a root estimate via the existing C-TreePO reduction.",
            "",
            f"**Diagnostics JSON:** `{diag_path}`",
            "",
            "\\newpage",
        ]
    )

    for p in root_pages:
        md_lines.extend([f"## Root error surface | {p.name}", "", img(p), "", "\\newpage"])
    md_lines.extend(["## Topic recovery curve", "", img(out_phi), "", "\\newpage"])
    for p in leaf_pages:
        md_lines.extend(["## Leaf-theta prediction error (proxy)", "", img(p), "", "\\newpage"])
    if ctree_pages:
        md_lines.extend(
            [
                "## C-TreePO attempt (separate run merged here)",
                "- These pages are loaded from the optional `--ctreepo-root` and filtered to `q_leaf=0, q_internal=0`.",
                "- `neural_ctreepo (pooled seed_fracs)` pools all seed-fraction settings in that run.",
                "",
            ]
        )
        for p in ctree_pages:
            md_lines.extend([f"### {p.name}", "", img(p), "", "\\newpage"])

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    if bool(args.emit_pdf):
        _run_pandoc(md_path, pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
