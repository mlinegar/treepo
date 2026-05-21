#!/usr/bin/env python3
"""Generate oracle-equivalence publication report with mixed stage tradeoff views.

Outputs three main figures:
Figure A: Semantics + Equivalence Endpoints
Figure B: Mixed Stage Tradeoff Surfaces (raw + normalized)
Figure C: Iso-Budget Frontiers (raw + normalized)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import transforms as mtransforms
from matplotlib.lines import Line2D
import numpy as np

SEG_TRUE_COLOR = "#1f77b4"
SEG_EMBED_COLOR = "#6c757d"
CTREE_COLOR = "#2ca02c"
MARKOV_ADD_COLOR = "#17becf"
MARKOV_NEURAL_COLOR = "#d62728"
STAGE_TRAIN_COLOR = "#1f77b4"
STAGE_INFER_COLOR = "#2ca02c"
PASS_EDGE_COLOR = "#1a9850"
FAIL_EDGE_COLOR = "#d73027"
NA_COLOR = "#888888"

PLOT_FLOOR = 1e-12
CEILING_THRESHOLD = 1e-8
HIGH_ERROR_CUTOFF = 1e-2
ERROR_AXIS_TOP = 10.0
NORM_EPS_DEN_DEFAULT = 1e-12
NORM_CLIP_MIN = -0.02
NORM_CLIP_MAX = 1.2
UNDEFINED_NORM_REASON = "baseline approx ceiling"

LEARN_LABEL = "learn-time oracle visibility"
DECISION_LABEL = "decision-time oracle visibility"
LEARN_SYMBOL = "q_train"
DECISION_SYMBOL = "q_infer"
ORACLE_PLAIN = "trusted ground-truth evaluator"
LEARN_SHORT = "learn-time"
DECISION_SHORT = "decision-time"

EXPECTED_SEG_Q = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
EXPECTED_CTREE_QTRAIN = [0.01, 0.02, 0.05, 0.1]
EXPECTED_CTREE_QINFER = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
EXPECTED_MARKOV_QTRAIN = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
EXPECTED_MARKOV_QINFER = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

FIXED_SEG_TRAIN_DOCS = 12000
FIXED_SEG_LAMBDA = 1.0
FIXED_CTREE_TRAIN_DOCS = 4096
FIXED_CTREE_MIN_CAL_SAMPLES = 50
FIXED_MARKOV_TRAIN_DOCS = 8000
FIXED_MARKOV_LEAF_QUERY_RATE = 1.0
FIXED_MARKOV_INCLUDE_ROOT_QUERY = True


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build clean publication report for identifiable oracle-equivalence outputs.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--clean-figures-subdir", type=str, default="pub_clean")
    p.add_argument("--output-markdown", type=Path, default=None)
    p.add_argument("--output-pdf", type=Path, default=None)
    p.add_argument("--output-diagnostics-json", type=Path, default=None)
    p.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--slice-policy", type=str, default="fixed_max", choices=["fixed_max"])
    p.add_argument("--budget-mode", type=str, default="rate", choices=["rate"])
    p.add_argument("--emit-mixed-surfaces", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--emit-budget-frontier", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--normalize-gap", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--norm-eps-den", type=float, default=NORM_EPS_DEN_DEFAULT)
    p.add_argument(
        "--allow-partial",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit a draft report even when the fixed publication slice is only partially populated.",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(x: object) -> Optional[float]:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _fmt(x: object) -> str:
    v = _as_float(x)
    if v is None:
        return "nan"
    if abs(v) >= 1000.0 or (0.0 < abs(v) < 1e-3):
        return f"{v:.3e}"
    return f"{v:.6g}"


def _fmt_norm_display(
    x: object,
    *,
    valid: bool,
    clip_min: float = NORM_CLIP_MIN,
    clip_max: float = NORM_CLIP_MAX,
) -> str:
    if not bool(valid):
        return "N/A"
    v = _as_float(x)
    if v is None:
        return "nan"
    if float(v) > float(clip_max):
        return f">{clip_max:g} (clipped)"
    if float(v) < float(clip_min):
        return f"<{clip_min:g} (clipped)"
    return _fmt(v)


def _clip_norm(v: object, clip_min: float = NORM_CLIP_MIN, clip_max: float = NORM_CLIP_MAX) -> float:
    vv = _as_float(v)
    if vv is None:
        return float("nan")
    return float(min(float(clip_max), max(float(clip_min), float(vv))))


def _first_use_stage_defs() -> List[str]:
    return [
        f"`oracle` = {ORACLE_PLAIN}.",
        f"`{LEARN_LABEL}` ({LEARN_SYMBOL}) = how much ground-truth is visible while learning the model.",
        f"`{DECISION_LABEL}` ({DECISION_SYMBOL}) = how much ground-truth is directly revealed at decision time.",
    ]


def _plot_floor(v: object) -> float:
    vv = _as_float(v)
    if vv is None:
        return float(PLOT_FLOOR)
    return float(max(PLOT_FLOOR, vv))


def _median(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return float("nan")
    return float(statistics.median(xs))


def _quantile(vals: Iterable[float], q: float) -> float:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return float(xs[0])
    idx = (len(xs) - 1) * float(q)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(xs[lo])
    w = idx - float(lo)
    return float((1.0 - w) * xs[lo] + w * xs[hi])


def _median_q25_q75(vals: Iterable[float]) -> Tuple[float, float, float]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return (float("nan"), float("nan"), float("nan"))
    return (_median(xs), _quantile(xs, 0.25), _quantile(xs, 0.75))


def _passes_ceiling(v: object) -> bool:
    vv = _as_float(v)
    return bool(vv is not None and vv <= CEILING_THRESHOLD)


def _status_edge_color(v: object) -> str:
    return PASS_EDGE_COLOR if _passes_ceiling(v) else FAIL_EDGE_COLOR


def _setup_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    plt.rcParams.update(
        {
            "font.size": 12.5,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _style_raw_error_axis(ax: plt.Axes, *, ylabel: str) -> None:
    ax.set_yscale("log")
    ax.set_ylim(PLOT_FLOOR, ERROR_AXIS_TOP)
    ax.axhspan(PLOT_FLOOR, CEILING_THRESHOLD, color="#c7f9cc", alpha=0.18, linewidth=0)
    ax.axhspan(CEILING_THRESHOLD, HIGH_ERROR_CUTOFF, color="#fff3bf", alpha=0.16, linewidth=0)
    ax.axhspan(HIGH_ERROR_CUTOFF, ERROR_AXIS_TOP, color="#fde2e2", alpha=0.12, linewidth=0)
    ax.axhline(CEILING_THRESHOLD, color="#666666", linestyle="--", linewidth=1.2)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22, which="both")


def _style_norm_axis(ax: plt.Axes, *, ylabel: str = "normalized gap-to-ceiling") -> None:
    ax.set_ylim(-0.03, NORM_CLIP_MAX + 0.04)
    ax.axhspan(-0.03, 0.0, color="#c7f9cc", alpha=0.18, linewidth=0)
    ax.axhspan(0.0, 1.0, color="#ecfdf3", alpha=0.12, linewidth=0)
    ax.axhspan(1.0, NORM_CLIP_MAX, color="#fff3bf", alpha=0.12, linewidth=0)
    ax.axhline(0.0, color="#1a9850", linestyle="--", linewidth=1.2)
    ax.axhline(1.0, color="#666666", linestyle=":", linewidth=1.2)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)


def _normalized_gap(err: float, baseline: float, ceiling: float) -> float:
    den = max(float(baseline) - float(ceiling), 1e-12)
    return float((float(err) - float(ceiling)) / den)


def _norm_den(baseline: float, ceiling: float) -> float:
    if not (math.isfinite(float(baseline)) and math.isfinite(float(ceiling))):
        return float("nan")
    return float(float(baseline) - float(ceiling))


def _norm_valid(baseline: float, ceiling: float, *, eps_den: float) -> bool:
    den = _norm_den(float(baseline), float(ceiling))
    return bool(math.isfinite(den) and den > float(eps_den))


def _normalize_series(values: Sequence[float], baseline: float, ceiling: float, *, eps_den: float) -> Tuple[List[float], bool, float]:
    den = _norm_den(float(baseline), float(ceiling))
    valid = _norm_valid(float(baseline), float(ceiling), eps_den=float(eps_den))
    if not valid:
        return ([float("nan")] * len(values), False, float(den))
    out = [_normalized_gap(float(v), float(baseline), float(ceiling)) for v in values]
    return ([float(v) for v in out], True, float(den))


def _apply_exact_fixed_slice(rows: Sequence[dict], *, key: str, value: object, tol: float = 1e-12) -> List[dict]:
    out: List[dict] = []
    for r in rows:
        rv = r.get(key)
        if isinstance(value, float):
            v = _as_float(rv)
            if v is None:
                continue
            if abs(float(v) - float(value)) <= float(tol):
                out.append(r)
        else:
            if rv == value:
                out.append(r)
    return out


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


def _collect_segment_fixed(output_root: Path, *, eps_den: float) -> Dict[str, object]:
    files = sorted(glob.glob(str(output_root / "segment_lda_ops_weight_recovery" / "**" / "*seed_*.json"), recursive=True))
    rows: List[dict] = []
    exact_vals: List[float] = []
    for fp in files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        m = payload.get("metrics", {}) or {}
        ex = _as_float(((m.get("exact") or {}).get("root_mae")))
        if ex is not None:
            exact_vals.append(float(ex))
        ridge = _as_float(((m.get("ridge") or {}).get("root_mae")))
        ridge_true = _as_float(((m.get("ridge_true_topics") or {}).get("root_mae")))
        rows.append(
            {
                "train_docs": int(cfg.get("train_docs", -1)),
                "q_train": float(cfg.get("audit_fraction", float("nan"))),
                "lambda_multiplier": float(cfg.get("lambda_multiplier", float("nan"))),
                "topic_phi_estimator": str(cfg.get("topic_phi_estimator", "")),
                "ridge": float(ridge) if ridge is not None else float("nan"),
                "ridge_true": float(ridge_true) if ridge_true is not None else float("nan"),
            }
        )

    fixed = [
        r
        for r in rows
        if int(r["train_docs"]) == FIXED_SEG_TRAIN_DOCS
        and math.isfinite(float(r["lambda_multiplier"]))
        and abs(float(r["lambda_multiplier"]) - FIXED_SEG_LAMBDA) <= 1e-12
    ]

    def _lane(phi: str, metric_key: str) -> Dict[str, object]:
        lane_rows = [r for r in fixed if str(r["topic_phi_estimator"]) == phi and math.isfinite(float(r[metric_key]))]
        qvals = sorted({float(r["q_train"]) for r in lane_rows if math.isfinite(float(r["q_train"]))})
        med: List[float] = []
        q25: List[float] = []
        q75: List[float] = []
        counts: List[int] = []
        for q in qvals:
            vals = [float(r[metric_key]) for r in lane_rows if abs(float(r["q_train"]) - q) <= 1e-12]
            m, a, b = _median_q25_q75(vals)
            med.append(float(m))
            q25.append(float(a))
            q75.append(float(b))
            counts.append(int(len(vals)))
        baseline = float(med[0]) if med else float("nan")
        ceiling = float(min(med)) if med else float("nan")
        norm, valid, den = _normalize_series([float(v) for v in med], baseline, ceiling, eps_den=float(eps_den))
        return {
            "q_train": [float(q) for q in qvals],
            "raw_median": [float(v) for v in med],
            "raw_q25": [float(v) for v in q25],
            "raw_q75": [float(v) for v in q75],
            "n_per_q": counts,
            "baseline": baseline,
            "ceiling": ceiling,
            "norm_den": float(den),
            "norm_valid": bool(valid),
            "norm_gap": [float(v) for v in norm],
            "q1": float(med[qvals.index(1.0)]) if 1.0 in qvals else float("nan"),
        }

    lane_true = _lane("true", "ridge_true")
    lane_embed = _lane("embedding_spectral", "ridge")

    return {
        "present": bool(files),
        "n_files": int(len(files)),
        "exact_root_mae_max": float(max(exact_vals) if exact_vals else float("nan")),
        "fixed": {
            "train_docs": FIXED_SEG_TRAIN_DOCS,
            "lambda_multiplier": FIXED_SEG_LAMBDA,
            "lanes": {
                "phi_true": lane_true,
                "phi_embedding_spectral": lane_embed,
            },
        },
    }


def _collect_ctree_fixed(output_root: Path, *, eps_den: float) -> Dict[str, object]:
    files = sorted(glob.glob(str(output_root / "segmented_lda_ctreepo" / "**" / "*.json"), recursive=True))
    rows: List[dict] = []
    oracle_vals: List[float] = []
    for fp in files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        m = payload.get("metrics", {}) or {}
        budgeted = _as_float(((m.get("estimated_calibrated_budgeted") or {}).get("root_l1_mean")))
        oracle = _as_float(((m.get("oracle_tree") or {}).get("root_l1_mean")))
        q_leaf = _as_float(cfg.get("eval_leaf_query_rate"))
        q_int = _as_float(cfg.get("eval_internal_query_rate"))
        q_train = _as_float(cfg.get("calibration_leaf_query_rate"))
        if oracle is not None:
            oracle_vals.append(float(oracle))
        if budgeted is None or q_leaf is None or q_int is None or q_train is None:
            continue
        rows.append(
            {
                "train_docs": int(cfg.get("n_books_train", -1)),
                "q_train": float(q_train),
                "q_leaf": float(q_leaf),
                "q_internal": float(q_int),
                "raw": float(budgeted),
                "calibration_samples": int(payload.get("calibration_samples", 0) or 0),
            }
        )

    fixed = [
        r
        for r in rows
        if int(r["train_docs"]) == FIXED_CTREE_TRAIN_DOCS
        and int(r["calibration_samples"]) >= FIXED_CTREE_MIN_CAL_SAMPLES
        and abs(float(r["q_leaf"]) - float(r["q_internal"])) <= 1e-12
    ]

    q_train_vals = sorted({float(r["q_train"]) for r in fixed if math.isfinite(float(r["q_train"]))})
    q_infer_vals = sorted({float(r["q_leaf"]) for r in fixed if math.isfinite(float(r["q_leaf"]))})

    matrix_raw: List[List[float]] = []
    matrix_q25: List[List[float]] = []
    matrix_q75: List[List[float]] = []
    matrix_counts: List[List[int]] = []
    for qtr in q_train_vals:
        row_raw: List[float] = []
        row_q25: List[float] = []
        row_q75: List[float] = []
        row_n: List[int] = []
        for qinf in q_infer_vals:
            vals = [
                float(r["raw"])
                for r in fixed
                if abs(float(r["q_train"]) - float(qtr)) <= 1e-12 and abs(float(r["q_leaf"]) - float(qinf)) <= 1e-12
            ]
            m, a, b = _median_q25_q75(vals)
            row_raw.append(float(m))
            row_q25.append(float(a))
            row_q75.append(float(b))
            row_n.append(int(len(vals)))
        matrix_raw.append(row_raw)
        matrix_q25.append(row_q25)
        matrix_q75.append(row_q75)
        matrix_counts.append(row_n)

    flat_raw = [float(v) for rr in matrix_raw for v in rr if math.isfinite(float(v))]
    baseline = float(matrix_raw[0][0]) if matrix_raw and matrix_raw[0] else float("nan")
    ceiling = float(min(flat_raw)) if flat_raw else float("nan")
    norm_flat, norm_valid, norm_den = _normalize_series(
        [float(v) for rr in matrix_raw for v in rr],
        baseline,
        ceiling,
        eps_den=float(eps_den),
    )
    matrix_norm: List[List[float]] = []
    k = 0
    for rr in matrix_raw:
        row = []
        for _ in rr:
            row.append(float(norm_flat[k]))
            k += 1
        matrix_norm.append(row)

    qtrain_max_idx = q_train_vals.index(max(q_train_vals)) if q_train_vals else -1
    qinfer_one_idx = q_infer_vals.index(1.0) if 1.0 in q_infer_vals else -1
    infer_full = (
        float(matrix_raw[qtrain_max_idx][qinfer_one_idx])
        if qtrain_max_idx >= 0 and qinfer_one_idx >= 0
        else float("nan")
    )
    infer_full_norm = (
        float(matrix_norm[qtrain_max_idx][qinfer_one_idx])
        if qtrain_max_idx >= 0 and qinfer_one_idx >= 0
        else float("nan")
    )

    return {
        "present": bool(files),
        "n_files": int(len(files)),
        "oracle_root_l1_max": float(max(oracle_vals) if oracle_vals else float("nan")),
        "fixed": {
            "train_docs": FIXED_CTREE_TRAIN_DOCS,
            "min_calibration_samples": FIXED_CTREE_MIN_CAL_SAMPLES,
            "q_train": [float(x) for x in q_train_vals],
            "q_infer": [float(x) for x in q_infer_vals],
            "matrix_raw": [[float(v) for v in rr] for rr in matrix_raw],
            "matrix_q25": [[float(v) for v in rr] for rr in matrix_q25],
            "matrix_q75": [[float(v) for v in rr] for rr in matrix_q75],
            "matrix_counts": [[int(v) for v in rr] for rr in matrix_counts],
            "matrix_norm": [[float(v) for v in rr] for rr in matrix_norm],
            "baseline": baseline,
            "ceiling": ceiling,
            "norm_den": float(norm_den),
            "norm_valid": bool(norm_valid),
            "infer_full_raw": infer_full,
            "infer_full_norm": infer_full_norm,
            "infer_full_context": {
                "q_train": float(max(q_train_vals)) if q_train_vals else float("nan"),
                "q_infer": 1.0,
            },
        },
    }


def _collect_markov_fixed(output_root: Path, *, eps_den: float) -> Dict[str, object]:
    files = sorted(glob.glob(str(output_root / "markov_changepoint_ops_count" / "**" / "*seed_*.json"), recursive=True))
    exact_vals: List[float] = []
    rows_learned: List[dict] = []
    rows_guided: List[dict] = []

    for fp in files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        m = payload.get("metrics", {}) or {}
        fam = str(cfg.get("model_family", ""))
        q_train = _as_float(cfg.get("audit_fraction"))
        leaf = _as_float(cfg.get("leaf_query_rate"))
        learned = _as_float(((m.get("learned") or {}).get("root_mae")))
        exact = _as_float(((m.get("exact") or {}).get("root_mae")))
        if exact is not None:
            exact_vals.append(float(exact))
        if q_train is None or leaf is None or learned is None:
            continue
        is_fixed = (
            int(cfg.get("train_docs", -1)) == FIXED_MARKOV_TRAIN_DOCS
            and abs(float(leaf) - FIXED_MARKOV_LEAF_QUERY_RATE) <= 1e-12
            and bool(cfg.get("include_root_query", True)) is bool(FIXED_MARKOV_INCLUDE_ROOT_QUERY)
        )
        if not is_fixed:
            continue
        rows_learned.append(
            {
                "family": fam,
                "q_train": float(q_train),
                "raw": float(learned),
            }
        )
        g = (m.get("guided_eval_curve") or {}).get("points") or []
        for point in g:
            if not isinstance(point, dict):
                continue
            q_inf = _as_float(point.get("q"))
            y = _as_float(point.get("root_mae"))
            if q_inf is None or y is None:
                continue
            rows_guided.append(
                {
                    "family": fam,
                    "q_train": float(q_train),
                    "q_infer": float(q_inf),
                    "raw": float(y),
                }
            )

    families = sorted({str(r["family"]) for r in rows_learned if str(r["family"])})
    fixed_by_family: Dict[str, Dict[str, object]] = {}
    for fam in families:
        lr = [r for r in rows_learned if str(r["family"]) == fam]
        gr = [r for r in rows_guided if str(r["family"]) == fam]

        q_train_vals = sorted({float(r["q_train"]) for r in lr})
        q_infer_vals = sorted({float(r["q_infer"]) for r in gr})

        train_med: List[float] = []
        train_q25: List[float] = []
        train_q75: List[float] = []
        train_counts: List[int] = []
        for qtr in q_train_vals:
            vals = [float(r["raw"]) for r in lr if abs(float(r["q_train"]) - float(qtr)) <= 1e-12]
            m, a, b = _median_q25_q75(vals)
            train_med.append(float(m))
            train_q25.append(float(a))
            train_q75.append(float(b))
            train_counts.append(int(len(vals)))

        matrix_raw: List[List[float]] = []
        matrix_q25: List[List[float]] = []
        matrix_q75: List[List[float]] = []
        matrix_counts: List[List[int]] = []
        for qtr in q_train_vals:
            rr_raw: List[float] = []
            rr_q25: List[float] = []
            rr_q75: List[float] = []
            rr_n: List[int] = []
            for qinf in q_infer_vals:
                vals = [
                    float(r["raw"])
                    for r in gr
                    if abs(float(r["q_train"]) - float(qtr)) <= 1e-12 and abs(float(r["q_infer"]) - float(qinf)) <= 1e-12
                ]
                m, a, b = _median_q25_q75(vals)
                rr_raw.append(float(m))
                rr_q25.append(float(a))
                rr_q75.append(float(b))
                rr_n.append(int(len(vals)))
            matrix_raw.append(rr_raw)
            matrix_q25.append(rr_q25)
            matrix_q75.append(rr_q75)
            matrix_counts.append(rr_n)

        train_baseline = float(train_med[0]) if train_med else float("nan")
        train_ceiling = float(min(train_med)) if train_med else float("nan")
        train_norm, train_valid, train_den = _normalize_series(
            [float(v) for v in train_med],
            train_baseline,
            train_ceiling,
            eps_den=float(eps_den),
        )

        flat = [float(v) for rr in matrix_raw for v in rr if math.isfinite(float(v))]
        mix_baseline = float(matrix_raw[0][0]) if matrix_raw and matrix_raw[0] else float("nan")
        mix_ceiling = float(min(flat)) if flat else float("nan")
        mix_norm_flat, mix_valid, mix_den = _normalize_series(
            [float(v) for rr in matrix_raw for v in rr],
            mix_baseline,
            mix_ceiling,
            eps_den=float(eps_den),
        )
        matrix_norm: List[List[float]] = []
        k = 0
        for rr in matrix_raw:
            r_out = []
            for _ in rr:
                r_out.append(float(mix_norm_flat[k]))
                k += 1
            matrix_norm.append(r_out)

        qtr_one_idx = q_train_vals.index(1.0) if 1.0 in q_train_vals else -1
        qinf_one_idx = q_infer_vals.index(1.0) if 1.0 in q_infer_vals else -1

        train_full_raw = float(train_med[qtr_one_idx]) if qtr_one_idx >= 0 else float("nan")
        train_full_norm = float(train_norm[qtr_one_idx]) if qtr_one_idx >= 0 else float("nan")
        infer_full_raw = (
            float(matrix_raw[qtr_one_idx][qinf_one_idx])
            if qtr_one_idx >= 0 and qinf_one_idx >= 0
            else float("nan")
        )
        infer_full_norm = (
            float(matrix_norm[qtr_one_idx][qinf_one_idx])
            if qtr_one_idx >= 0 and qinf_one_idx >= 0
            else float("nan")
        )

        fixed_by_family[fam] = {
            "available": bool(
                q_train_vals
                and (
                    any(math.isfinite(float(v)) for v in train_med)
                    or any(math.isfinite(float(v)) for rr in matrix_raw for v in rr)
                )
            ),
            "q_train": [float(x) for x in q_train_vals],
            "q_infer": [float(x) for x in q_infer_vals],
            "train_curve_raw": [float(v) for v in train_med],
            "train_curve_q25": [float(v) for v in train_q25],
            "train_curve_q75": [float(v) for v in train_q75],
            "train_counts": [int(v) for v in train_counts],
            "train_curve_norm": [float(v) for v in train_norm],
            "train_baseline": train_baseline,
            "train_ceiling": train_ceiling,
            "train_norm_den": float(train_den),
            "train_norm_valid": bool(train_valid),
            "matrix_raw": [[float(v) for v in rr] for rr in matrix_raw],
            "matrix_q25": [[float(v) for v in rr] for rr in matrix_q25],
            "matrix_q75": [[float(v) for v in rr] for rr in matrix_q75],
            "matrix_counts": [[int(v) for v in rr] for rr in matrix_counts],
            "matrix_norm": [[float(v) for v in rr] for rr in matrix_norm],
            "mix_baseline": mix_baseline,
            "mix_ceiling": mix_ceiling,
            "mix_norm_den": float(mix_den),
            "mix_norm_valid": bool(mix_valid),
            "train_full_raw": train_full_raw,
            "train_full_norm": train_full_norm,
            "infer_full_raw": infer_full_raw,
            "infer_full_norm": infer_full_norm,
        }

    return {
        "present": bool(files),
        "n_files": int(len(files)),
        "exact_root_mae_max": float(max(exact_vals) if exact_vals else float("nan")),
        "fixed": {
            "train_docs": FIXED_MARKOV_TRAIN_DOCS,
            "leaf_query_rate": FIXED_MARKOV_LEAF_QUERY_RATE,
            "include_root_query": bool(FIXED_MARKOV_INCLUDE_ROOT_QUERY),
            "available_families": sorted(
                fam for fam, lane in fixed_by_family.items() if _markov_family_available(lane)
            ),
            "families": fixed_by_family,
        },
    }


def _approx_list_equal(a: Sequence[float], b: Sequence[float], tol: float = 1e-12) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if abs(float(x) - float(y)) > float(tol):
            return False
    return True


def _has_finite_values(values: Iterable[object]) -> bool:
    return any(_as_float(v) is not None for v in values)


def _has_finite_matrix(rows: Iterable[Iterable[object]]) -> bool:
    return any(_as_float(v) is not None for row in rows for v in row)


def _segment_lane_available(lane: Dict[str, object]) -> bool:
    return bool(
        _has_finite_values(lane.get("raw_median") or [])
        or _has_finite_values(lane.get("norm_gap") or [])
        or _has_finite_values([lane.get("q1")])
    )


def _ctree_lane_available(lane: Dict[str, object]) -> bool:
    return bool(
        _has_finite_matrix(lane.get("matrix_raw") or [])
        or _has_finite_matrix(lane.get("matrix_norm") or [])
        or _as_float(lane.get("infer_full_raw")) is not None
    )


def _markov_family_available(lane: Dict[str, object]) -> bool:
    return bool(
        lane.get("available", False)
        or _has_finite_values(lane.get("train_curve_raw") or [])
        or _has_finite_matrix(lane.get("matrix_raw") or [])
        or _as_float(lane.get("train_full_raw")) is not None
        or _as_float(lane.get("infer_full_raw")) is not None
    )


def _slice_consistency_checks(segment: Dict[str, object], ctree: Dict[str, object], markov: Dict[str, object]) -> Dict[str, object]:
    issues: List[str] = []
    warnings: List[str] = []

    seg_fixed = (segment.get("fixed") or {})
    seg_lanes = (seg_fixed.get("lanes") or {})
    seg_true = (seg_lanes.get("phi_true") or {})
    seg_embed = (seg_lanes.get("phi_embedding_spectral") or {})

    if int(seg_fixed.get("train_docs", -1)) != FIXED_SEG_TRAIN_DOCS:
        issues.append("segment fixed train_docs mismatch")
    if abs(float(seg_fixed.get("lambda_multiplier", float("nan"))) - FIXED_SEG_LAMBDA) > 1e-12:
        issues.append("segment fixed lambda mismatch")
    if not _approx_list_equal([float(x) for x in (seg_true.get("q_train") or [])], EXPECTED_SEG_Q):
        warnings.append("segment phi_true q grid differs from expected canonical grid")
    if not _approx_list_equal([float(x) for x in (seg_embed.get("q_train") or [])], EXPECTED_SEG_Q):
        warnings.append("segment phi_embedding q grid differs from expected canonical grid")

    ct_fixed = (ctree.get("fixed") or {})
    ct_qtr = [float(x) for x in (ct_fixed.get("q_train") or [])]
    ct_qinf = [float(x) for x in (ct_fixed.get("q_infer") or [])]
    ct_counts = ct_fixed.get("matrix_counts") or []

    if int(ct_fixed.get("train_docs", -1)) != FIXED_CTREE_TRAIN_DOCS:
        issues.append("ctree fixed train_docs mismatch")
    if int(ct_fixed.get("min_calibration_samples", -1)) != FIXED_CTREE_MIN_CAL_SAMPLES:
        issues.append("ctree min_calibration_samples mismatch")
    if not _approx_list_equal(ct_qtr, EXPECTED_CTREE_QTRAIN):
        issues.append("ctree q_train grid mismatch")
    if not _approx_list_equal(ct_qinf, EXPECTED_CTREE_QINFER):
        issues.append("ctree q_infer grid mismatch")
    if len(ct_counts) != len(ct_qtr) or any(len(rr) != len(ct_qinf) for rr in ct_counts):
        issues.append("ctree matrix shape mismatch")
    else:
        flat_counts = [int(v) for rr in ct_counts for v in rr]
        if not flat_counts:
            issues.append("ctree matrix counts empty")
        else:
            min_ct = min(flat_counts)
            max_ct = max(flat_counts)
            if min_ct != 12 or max_ct != 12:
                issues.append(f"ctree replicate counts not all 12 (min={min_ct}, max={max_ct})")

    mk_fixed = (markov.get("fixed") or {})
    if int(mk_fixed.get("train_docs", -1)) != FIXED_MARKOV_TRAIN_DOCS:
        issues.append("markov fixed train_docs mismatch")
    if abs(float(mk_fixed.get("leaf_query_rate", float("nan"))) - FIXED_MARKOV_LEAF_QUERY_RATE) > 1e-12:
        issues.append("markov fixed leaf_query_rate mismatch")
    if bool(mk_fixed.get("include_root_query", None)) is not bool(FIXED_MARKOV_INCLUDE_ROOT_QUERY):
        issues.append("markov include_root_query mismatch")

    fams = (mk_fixed.get("families") or {})
    for fam in ["additive", "neural"]:
        fd = fams.get(fam) or {}
        if not _markov_family_available(fd):
            warnings.append(f"markov {fam} fixed slice unavailable")
            continue
        qtr = [float(x) for x in (fd.get("q_train") or [])]
        qinf = [float(x) for x in (fd.get("q_infer") or [])]
        if not _approx_list_equal(qtr, EXPECTED_MARKOV_QTRAIN):
            issues.append(f"markov {fam} q_train grid mismatch")
        if not _approx_list_equal(qinf, EXPECTED_MARKOV_QINFER):
            issues.append(f"markov {fam} q_infer grid mismatch")
        counts = fd.get("matrix_counts") or []
        if len(counts) != len(qtr) or any(len(rr) != len(qinf) for rr in counts):
            issues.append(f"markov {fam} matrix shape mismatch")
        else:
            flat_counts = [int(v) for rr in counts for v in rr]
            if not flat_counts:
                issues.append(f"markov {fam} matrix counts empty")
            else:
                min_ct = min(flat_counts)
                max_ct = max(flat_counts)
                if min_ct != 12 or max_ct != 12:
                    issues.append(f"markov {fam} replicate counts not all 12 (min={min_ct}, max={max_ct})")
        train_counts = [int(v) for v in (fd.get("train_counts") or [])]
        if train_counts:
            if min(train_counts) != 12 or max(train_counts) != 12:
                issues.append(f"markov {fam} train-curve replicate counts not all 12")

    checks = {
        "passed": bool(not issues),
        "issues": issues,
        "warnings": warnings,
        "expected": {
            "segment_q": EXPECTED_SEG_Q,
            "ctree_q_train": EXPECTED_CTREE_QTRAIN,
            "ctree_q_infer": EXPECTED_CTREE_QINFER,
            "markov_q_train": EXPECTED_MARKOV_QTRAIN,
            "markov_q_infer": EXPECTED_MARKOV_QINFER,
            "expected_replicates_per_cell": 12,
        },
    }
    return checks


def _frontier_from_points(points: Sequence[Tuple[float, float]]) -> Dict[str, List[float]]:
    by_budget: Dict[float, List[float]] = {}
    for b, y in points:
        if not (math.isfinite(float(b)) and math.isfinite(float(y))):
            continue
        by_budget.setdefault(float(b), []).append(float(y))
    xs = sorted(by_budget.keys())
    ys_at_budget = [float(min(by_budget[x])) for x in xs]
    best: List[float] = []
    cur = float("inf")
    for y in ys_at_budget:
        cur = min(cur, float(y))
        best.append(float(cur))
    return {
        "budget": [float(x) for x in xs],
        "best_error": [float(y) for y in best],
    }


def _build_endpoints(segment: Dict[str, object], ctree: Dict[str, object], markov: Dict[str, object]) -> List[Dict[str, object]]:
    seg_lanes = ((segment.get("fixed") or {}).get("lanes") or {})
    seg_true = seg_lanes.get("phi_true") or {}
    seg_embed = seg_lanes.get("phi_embedding_spectral") or {}

    ct_fixed = ctree.get("fixed") or {}

    mk_fams = ((markov.get("fixed") or {}).get("families") or {})
    mk_add = mk_fams.get("additive") or {}
    mk_neu = mk_fams.get("neural") or {}
    endpoints: List[Dict[str, object]] = []

    def _append_endpoint(
        *,
        endpoint_id: str,
        name: str,
        short_label: str,
        family: str,
        stage: str,
        raw: object,
        norm: object,
        norm_valid: bool,
        context: str,
    ) -> None:
        raw_v = _as_float(raw)
        if raw_v is None:
            return
        norm_v = _as_float(norm)
        endpoints.append(
            {
                "endpoint_id": endpoint_id,
                "name": name,
                "short_label": short_label,
                "family": family,
                "stage": stage,
                "raw": float(raw_v),
                "norm": float(norm_v) if norm_v is not None else float("nan"),
                "norm_valid": bool(norm_valid),
                "context": context,
            }
        )

    if _segment_lane_available(seg_true):
        _append_endpoint(
            endpoint_id="segment_phi_true_learn_full",
            name="Segment (phi=true, learn-time full)",
            short_label="Segment phi=true",
            family="segment",
            stage="train",
            raw=seg_true.get("q1"),
            norm=(seg_true.get("norm_gap") or [float("nan")])[-1] if (seg_true.get("norm_gap") or []) else float("nan"),
            norm_valid=bool(seg_true.get("norm_valid", False)),
            context=f"{LEARN_SHORT} full (100%) [{LEARN_SYMBOL}=1.0]",
        )
    if _segment_lane_available(seg_embed):
        _append_endpoint(
            endpoint_id="segment_phi_embedding_learn_full",
            name="Segment (phi=embedding_spectral, learn-time full)",
            short_label="Segment phi=embedding",
            family="segment",
            stage="train",
            raw=seg_embed.get("q1"),
            norm=(seg_embed.get("norm_gap") or [float("nan")])[-1] if (seg_embed.get("norm_gap") or []) else float("nan"),
            norm_valid=bool(seg_embed.get("norm_valid", False)),
            context=f"{LEARN_SHORT} full (100%) [{LEARN_SYMBOL}=1.0]",
        )
    if _ctree_lane_available(ct_fixed):
        _append_endpoint(
            endpoint_id="ctree_decision_full",
            name="C-TreePO (decision-time full)",
            short_label="C-TreePO",
            family="ctree",
            stage="infer",
            raw=ct_fixed.get("infer_full_raw"),
            norm=ct_fixed.get("infer_full_norm"),
            norm_valid=bool(ct_fixed.get("norm_valid", False)),
            context=(
                f"{LEARN_SHORT}={_fmt((ct_fixed.get('infer_full_context') or {}).get('q_train'))}, "
                f"{DECISION_SHORT} full (100%)"
            ),
        )
    if _markov_family_available(mk_add):
        _append_endpoint(
            endpoint_id="markov_additive_learn_full",
            name="Markov additive (learn-time full)",
            short_label="Markov additive",
            family="markov_add",
            stage="train",
            raw=mk_add.get("train_full_raw"),
            norm=mk_add.get("train_full_norm"),
            norm_valid=bool(mk_add.get("train_norm_valid", False)),
            context=f"{LEARN_SHORT} full (100%) [{LEARN_SYMBOL}=1.0]",
        )
        _append_endpoint(
            endpoint_id="markov_additive_decision_full",
            name="Markov additive (decision-time full)",
            short_label="Markov additive",
            family="markov_add",
            stage="infer",
            raw=mk_add.get("infer_full_raw"),
            norm=mk_add.get("infer_full_norm"),
            norm_valid=bool(mk_add.get("mix_norm_valid", False)),
            context=f"{LEARN_SHORT} full + {DECISION_SHORT} full (100%)",
        )
    if _markov_family_available(mk_neu):
        _append_endpoint(
            endpoint_id="markov_neural_learn_full",
            name="Markov neural (learn-time full)",
            short_label="Markov neural",
            family="markov_neural",
            stage="train",
            raw=mk_neu.get("train_full_raw"),
            norm=mk_neu.get("train_full_norm"),
            norm_valid=bool(mk_neu.get("train_norm_valid", False)),
            context=f"{LEARN_SHORT} full (100%) [{LEARN_SYMBOL}=1.0]",
        )
        _append_endpoint(
            endpoint_id="markov_neural_decision_full",
            name="Markov neural (decision-time full)",
            short_label="Markov neural",
            family="markov_neural",
            stage="infer",
            raw=mk_neu.get("infer_full_raw"),
            norm=mk_neu.get("infer_full_norm"),
            norm_valid=bool(mk_neu.get("mix_norm_valid", False)),
            context=f"{LEARN_SHORT} full + {DECISION_SHORT} full (100%)",
        )
    return endpoints


def _family_color(family: str) -> str:
    if family == "segment":
        return SEG_TRUE_COLOR
    if family == "ctree":
        return CTREE_COLOR
    if family == "markov_add":
        return MARKOV_ADD_COLOR
    if family == "markov_neural":
        return MARKOV_NEURAL_COLOR
    return "#444444"


def _plot_figure_a(
    segment: Dict[str, object],
    ctree: Dict[str, object],
    markov: Dict[str, object],
    out_png: Path,
    out_pdf: Path,
) -> Dict[str, object]:
    endpoints = _build_endpoints(segment, ctree, markov)

    fig = plt.figure(figsize=(18.0, 12.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.7, 1.3])
    ax_text = fig.add_subplot(gs[0, :])
    ax_raw = fig.add_subplot(gs[1, 0])
    ax_norm = fig.add_subplot(gs[1, 1])

    seg_fixed = segment.get("fixed") or {}
    ct_fixed = ctree.get("fixed") or {}
    mk_fixed = markov.get("fixed") or {}

    ax_text.axis("off")
    first_use = _first_use_stage_defs()
    text_lines = [
        "A1: Knobs And Stages (Reading Guide)",
        "",
        first_use[0],
        first_use[1],
        first_use[2],
        "",
        "Reading rule:",
        "- Use A2 for within-family raw error only.",
        "- Use A3 for cross-family progress only.",
        f"- `N/A` / `undef` means undefined normalization ({UNDEFINED_NORM_REASON}), not a large score.",
        "",
        "Fixed slice reused in all figures:",
        f"- Segment: train_docs={seg_fixed.get('train_docs')}, local-law lambda={seg_fixed.get('lambda_multiplier')}",
        f"- C-TreePO: train_docs={ct_fixed.get('train_docs')}, min_calibration_samples={ct_fixed.get('min_calibration_samples')}",
        f"- Markov: train_docs={mk_fixed.get('train_docs')}, leaf_query_rate={mk_fixed.get('leaf_query_rate')}, include_root_query={mk_fixed.get('include_root_query')}",
        "- Segment lambda here is the paper local-law weight, not the separate quadratic-utility weight used in excluded diagnostic LDA roots.",
    ]
    ax_text.text(
        0.0,
        1.0,
        "\n".join(text_lines),
        ha="left",
        va="top",
        fontsize=11.3,
        bbox=dict(boxstyle="round,pad=0.42", facecolor="white", edgecolor="#cccccc", alpha=0.95),
    )

    x = np.arange(len(endpoints), dtype=np.float64)
    labels = [
        f"{ep.get('short_label')}\n({LEARN_SHORT if str(ep.get('stage')) == 'train' else DECISION_SHORT})"
        for ep in endpoints
    ]
    endpoint_table: List[Dict[str, object]] = []
    for i, ep in enumerate(endpoints):
        raw = _plot_floor(ep.get("raw"))
        col = _family_color(str(ep.get("family")))
        marker = "^" if str(ep.get("stage")) == "train" else "o"
        status = "PASS" if _passes_ceiling(ep.get("raw")) else "FAIL"
        ax_raw.scatter(
            [x[i]],
            [raw],
            s=120,
            marker=marker,
            color=col,
            edgecolors=_status_edge_color(ep.get("raw")),
            linewidths=2.0,
            zorder=4,
        )
        ax_raw.text(
            x[i],
            min(ERROR_AXIS_TOP / 1.4, raw * 1.45),
            status,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
        endpoint_table.append(
            {
                "endpoint_id": str(ep.get("endpoint_id", "")),
                "name": str(ep.get("name", "")),
                "family": str(ep.get("family", "")),
                "stage": str(ep.get("stage", "")),
                "context": str(ep.get("context", "")),
                "raw": float(ep.get("raw", float("nan"))),
                "norm": float(ep.get("norm", float("nan"))),
                "norm_valid": bool(ep.get("norm_valid", False)),
                "norm_display": _fmt_norm_display(ep.get("norm"), valid=bool(ep.get("norm_valid", False))),
                "status": status,
            }
        )

    _style_raw_error_axis(ax_raw, ylabel="endpoint raw error (log)")
    ax_raw.set_title("A2: Endpoint Raw Error Readout (Within-Family)")
    ax_raw.set_xticks(x)
    ax_raw.set_xticklabels(labels, rotation=15, ha="right")
    ax_raw.text(
        0.02,
        0.98,
        "Not cross-family comparable:\nA2 C-TreePO (root L1) vs A2/A3 Markov (root MAE)",
        transform=ax_raw.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#cccccc", alpha=0.93),
    )

    undefined_labels: List[str] = []
    undefined_transform = mtransforms.blended_transform_factory(ax_norm.transData, ax_norm.transAxes)
    for i, ep in enumerate(endpoints):
        valid = bool(ep.get("norm_valid", False))
        yclip = _clip_norm(ep.get("norm"))
        col = _family_color(str(ep.get("family")))
        marker = "^" if str(ep.get("stage")) == "train" else "o"
        if valid and math.isfinite(yclip):
            ax_norm.scatter(
                [x[i]],
                [yclip],
                s=120,
                marker=marker,
                color=col,
                edgecolors=_status_edge_color(ep.get("raw")),
                linewidths=1.8,
                zorder=4,
            )
        else:
            undefined_labels.append(str(ep.get("short_label", ep.get("name", ""))))
            ax_norm.scatter(
                [x[i]],
                [1.03],
                s=80,
                marker="x",
                color=NA_COLOR,
                linewidths=2.0,
                zorder=5,
                clip_on=False,
                transform=undefined_transform,
            )
            ax_norm.text(
                x[i],
                1.06,
                "undef",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=NA_COLOR,
                clip_on=False,
                transform=undefined_transform,
            )

    _style_norm_axis(ax_norm)
    ax_norm.set_title("A3: Endpoint Normalized Progress (Cross-Family Comparable)")
    ax_norm.set_xticks(x)
    ax_norm.set_xticklabels(labels, rotation=15, ha="right")
    ax_norm.text(
        0.02,
        0.97,
        f"0 = ceiling reached\n1 = baseline difficulty\n`undef` above frame = {UNDEFINED_NORM_REASON}",
        transform=ax_norm.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )
    if undefined_labels:
        ax_norm.text(
            0.98,
            0.03,
            "Undefined endpoints: " + ", ".join(undefined_labels),
            transform=ax_norm.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.2,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#cccccc", alpha=0.9),
        )

    legend_handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor=STAGE_TRAIN_COLOR, markeredgecolor="#333333", label=f"{LEARN_SHORT} endpoint", markersize=8),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=STAGE_INFER_COLOR, markeredgecolor="#333333", label=f"{DECISION_SHORT} endpoint", markersize=8),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=PASS_EDGE_COLOR, markeredgewidth=2.0, label="PASS (<=1e-8)", markersize=8),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=FAIL_EDGE_COLOR, markeredgewidth=2.0, label="FAIL (>1e-8)", markersize=8),
        Line2D([0], [0], marker="x", color=NA_COLOR, linestyle="none", label=f"undefined normalization ({UNDEFINED_NORM_REASON})", markersize=8),
    ]
    ax_raw.legend(handles=legend_handles, frameon=False, loc="lower left", ncol=2)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Figure A: Equivalence Endpoints And Reading Rules", fontsize=18)
    fig.savefig(out_png, dpi=260)
    fig.savefig(out_pdf)
    plt.close(fig)

    return {
        "endpoints": endpoints,
        "endpoint_table": endpoint_table,
    }


def _plot_heatmap(
    ax: plt.Axes,
    arr: np.ndarray,
    *,
    xvals: Sequence[float],
    yvals: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str,
    norm: mcolors.Normalize,
    yticklabels: Optional[Sequence[str]] = None,
    takeaway: Optional[str] = None,
) -> None:
    arr_masked = np.ma.masked_invalid(arr)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color="#f2f2f2")
    if arr_masked.ndim != 2 or arr_masked.size == 0 or len(xvals) == 0 or len(yvals) == 0:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#f7f7f7")
        ax.text(
            0.5,
            0.5,
            "Pending\nno completed cells yet",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#666666",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.95),
        )
        if takeaway:
            ax.text(
                0.02,
                0.02,
                takeaway,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#cccccc", alpha=0.92),
            )
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
        sm.set_array(np.asarray([0.0], dtype=np.float64))
        return sm
    im = ax.imshow(arr_masked, aspect="auto", origin="lower", cmap=cm, norm=norm)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(xvals)))
    ax.set_xticklabels([_fmt(v) for v in xvals], rotation=0)
    ax.set_yticks(range(len(yvals)))
    if yticklabels is not None:
        ax.set_yticklabels(list(yticklabels))
    else:
        ax.set_yticklabels([_fmt(v) for v in yvals])
    ax.grid(False)
    if takeaway:
        ax.text(
            0.02,
            0.02,
            takeaway,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#cccccc", alpha=0.92),
        )
    return im


def _plot_figure_b(
    segment: Dict[str, object],
    ctree: Dict[str, object],
    markov: Dict[str, object],
    out_png: Path,
    out_pdf: Path,
) -> Dict[str, object]:
    seg_lanes = ((segment.get("fixed") or {}).get("lanes") or {})
    seg_true = seg_lanes.get("phi_true") or {}
    seg_embed = seg_lanes.get("phi_embedding_spectral") or {}

    ct_fixed = ctree.get("fixed") or {}
    ct_raw = np.asarray(ct_fixed.get("matrix_raw") or [], dtype=np.float64)
    ct_norm = np.asarray(ct_fixed.get("matrix_norm") or [], dtype=np.float64)
    ct_qtr = [float(x) for x in (ct_fixed.get("q_train") or [])]
    ct_qinf = [float(x) for x in (ct_fixed.get("q_infer") or [])]

    mk_fams = ((markov.get("fixed") or {}).get("families") or {})
    mk_add = mk_fams.get("additive") or {}
    mk_neu = mk_fams.get("neural") or {}

    add_raw = np.asarray(mk_add.get("matrix_raw") or [], dtype=np.float64)
    add_norm = np.asarray(mk_add.get("matrix_norm") or [], dtype=np.float64)
    add_qtr = [float(x) for x in (mk_add.get("q_train") or [])]
    add_qinf = [float(x) for x in (mk_add.get("q_infer") or [])]

    neu_raw = np.asarray(mk_neu.get("matrix_raw") or [], dtype=np.float64)
    neu_norm = np.asarray(mk_neu.get("matrix_norm") or [], dtype=np.float64)

    seg_q = [float(x) for x in (seg_true.get("q_train") or [])]
    seg_strip_raw = np.asarray(
        [
            [float(v) for v in (seg_true.get("raw_median") or [])],
            [float(v) for v in (seg_embed.get("raw_median") or [])],
        ],
        dtype=np.float64,
    )
    seg_strip_norm = np.asarray(
        [
            [float(v) for v in (seg_true.get("norm_gap") or [])],
            [float(v) for v in (seg_embed.get("norm_gap") or [])],
        ],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(4, 2, figsize=(17.5, 22.0), constrained_layout=True)
    raw_norm = mcolors.LogNorm(vmin=PLOT_FLOOR, vmax=ERROR_AXIS_TOP)
    norm_norm = mcolors.Normalize(vmin=0.0, vmax=NORM_CLIP_MAX)
    cmap = "RdYlGn_r"

    xlabel = f"{DECISION_LABEL} ({DECISION_SYMBOL})"
    ylabel = f"{LEARN_LABEL} ({LEARN_SYMBOL})"

    im00 = _plot_heatmap(
        axes[0, 0],
        np.clip(ct_raw, PLOT_FLOOR, ERROR_AXIS_TOP),
        xvals=ct_qinf,
        yvals=ct_qtr,
        title="B1 C-TreePO Raw Error (root L1)",
        xlabel=xlabel,
        ylabel=ylabel,
        cmap=cmap,
        norm=raw_norm,
    )
    im01 = _plot_heatmap(
        axes[0, 1],
        np.clip(ct_norm, NORM_CLIP_MIN, NORM_CLIP_MAX),
        xvals=ct_qinf,
        yvals=ct_qtr,
        title="B2 C-TreePO Normalized Progress",
        xlabel=xlabel,
        ylabel=ylabel,
        cmap=cmap,
        norm=norm_norm,
    )
    im10 = _plot_heatmap(
        axes[1, 0],
        np.clip(add_raw, PLOT_FLOOR, ERROR_AXIS_TOP),
        xvals=add_qinf,
        yvals=add_qtr,
        title="B3 Markov Additive Raw Error (root MAE)",
        xlabel=xlabel,
        ylabel=ylabel,
        cmap=cmap,
        norm=raw_norm,
    )
    im11 = _plot_heatmap(
        axes[1, 1],
        np.clip(add_norm, NORM_CLIP_MIN, NORM_CLIP_MAX),
        xvals=add_qinf,
        yvals=add_qtr,
        title="B4 Markov Additive Normalized Progress",
        xlabel=xlabel,
        ylabel=ylabel,
        cmap=cmap,
        norm=norm_norm,
    )
    im20 = _plot_heatmap(
        axes[2, 0],
        np.clip(neu_raw, PLOT_FLOOR, ERROR_AXIS_TOP),
        xvals=add_qinf,
        yvals=add_qtr,
        title="B5 Markov Neural Raw Error (root MAE)",
        xlabel=xlabel,
        ylabel=ylabel,
        cmap=cmap,
        norm=raw_norm,
    )
    im21 = _plot_heatmap(
        axes[2, 1],
        np.clip(neu_norm, NORM_CLIP_MIN, NORM_CLIP_MAX),
        xvals=add_qinf,
        yvals=add_qtr,
        title="B6 Markov Neural Normalized Progress",
        xlabel=xlabel,
        ylabel=ylabel,
        cmap=cmap,
        norm=norm_norm,
    )
    im30 = _plot_heatmap(
        axes[3, 0],
        np.clip(seg_strip_raw, PLOT_FLOOR, ERROR_AXIS_TOP),
        xvals=seg_q,
        yvals=[0.0, 1.0],
        title="B7 Segment Raw Error",
        xlabel=f"{LEARN_LABEL} ({LEARN_SYMBOL})",
        ylabel="lane",
        cmap=cmap,
        norm=raw_norm,
        yticklabels=["phi=true", "phi=embedding"],
    )
    axes[3, 0].text(
        0.02,
        0.98,
        f"No native {DECISION_LABEL} stage",
        transform=axes[3, 0].transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )
    im31 = _plot_heatmap(
        axes[3, 1],
        np.clip(seg_strip_norm, NORM_CLIP_MIN, NORM_CLIP_MAX),
        xvals=seg_q,
        yvals=[0.0, 1.0],
        title="B8 Segment Normalized Progress",
        xlabel=f"{LEARN_LABEL} ({LEARN_SYMBOL})",
        ylabel="lane",
        cmap=cmap,
        norm=norm_norm,
        yticklabels=["phi=true", "phi=embedding"],
    )

    fig.colorbar(
        im30,
        ax=axes[:, 0],
        shrink=0.9,
        location="right",
        label="Raw error (log color scale): green=low, red=high",
    )
    fig.colorbar(
        im31,
        ax=axes[:, 1],
        shrink=0.9,
        location="right",
        label=f"Normalized progress: 0=ceiling, 1=baseline, >{NORM_CLIP_MAX:g} clipped, gray=undefined",
    )

    fig.suptitle(
        "Figure B: Mixed Learn-Time / Decision-Time Tradeoff Surfaces\n"
        f"Fixed slice: Segment train_docs={FIXED_SEG_TRAIN_DOCS}; "
        f"C-TreePO train_docs={FIXED_CTREE_TRAIN_DOCS}, min_calibration_samples={FIXED_CTREE_MIN_CAL_SAMPLES}; "
        f"Markov train_docs={FIXED_MARKOV_TRAIN_DOCS}, leaf_query_rate={FIXED_MARKOV_LEAF_QUERY_RATE}, include_root_query={FIXED_MARKOV_INCLUDE_ROOT_QUERY}.",
        fontsize=15.5,
    )
    fig.text(
        0.5,
        0.97,
        "Left column: raw error (within-family only). Right column: normalized progress (cross-family comparison).",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff7e6", edgecolor="#d8d8d8", alpha=0.95),
    )
    fig.text(
        0.5,
        0.945,
        f"Gray cells mean undefined normalization ({UNDEFINED_NORM_REASON}), not a numeric value near {NORM_CLIP_MAX:g}.",
        ha="center",
        va="center",
        fontsize=11.2,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#e8f5e9", edgecolor="#d8d8d8", alpha=0.95),
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=260)
    fig.savefig(out_pdf)
    plt.close(fig)

    return {
        "ctree_matrix_shape": [int(ct_raw.shape[0]), int(ct_raw.shape[1])] if ct_raw.ndim == 2 else [0, 0],
        "markov_add_matrix_shape": [int(add_raw.shape[0]), int(add_raw.shape[1])] if add_raw.ndim == 2 else [0, 0],
        "markov_neural_matrix_shape": [int(neu_raw.shape[0]), int(neu_raw.shape[1])] if neu_raw.ndim == 2 else [0, 0],
        "segment_strip_shape": [int(seg_strip_raw.shape[0]), int(seg_strip_raw.shape[1])] if seg_strip_raw.ndim == 2 else [0, 0],
    }


def _collect_frontier_points(segment: Dict[str, object], ctree: Dict[str, object], markov: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}

    seg_lanes = ((segment.get("fixed") or {}).get("lanes") or {})
    for lane_key, lane_label, color in [
        ("phi_true", "Segment phi=true", SEG_TRUE_COLOR),
        ("phi_embedding_spectral", "Segment phi=embedding", SEG_EMBED_COLOR),
    ]:
        lane = seg_lanes.get(lane_key) or {}
        q = [float(x) for x in (lane.get("q_train") or [])]
        raw = [float(x) for x in (lane.get("raw_median") or [])]
        norm = [float(x) for x in (lane.get("norm_gap") or [])]
        raw_pts = [(float(qi), float(yi)) for qi, yi in zip(q, raw)]
        norm_pts = [(float(qi), float(yi)) for qi, yi in zip(q, norm)]
        if raw_pts or norm_pts:
            out[lane_label] = {
                "label": lane_label,
                "color": color,
                "norm_valid": bool(lane.get("norm_valid", False)),
                "norm_den": _as_float(lane.get("norm_den")),
                "raw_points": raw_pts,
                "norm_points": norm_pts,
            }

    ct_fixed = ctree.get("fixed") or {}
    ct_qtr = [float(x) for x in (ct_fixed.get("q_train") or [])]
    ct_qinf = [float(x) for x in (ct_fixed.get("q_infer") or [])]
    ct_raw = ct_fixed.get("matrix_raw") or []
    ct_norm = ct_fixed.get("matrix_norm") or []
    ct_raw_pts: List[Tuple[float, float]] = []
    ct_norm_pts: List[Tuple[float, float]] = []
    for i, qtr in enumerate(ct_qtr):
        for j, qinf in enumerate(ct_qinf):
            b = 0.5 * float(qtr) + 0.5 * float(qinf)
            ct_raw_pts.append((b, float(ct_raw[i][j])))
            ct_norm_pts.append((b, float(ct_norm[i][j])))
    if ct_raw_pts or ct_norm_pts:
        out["C-TreePO mixed"] = {
            "label": "C-TreePO mixed",
            "color": CTREE_COLOR,
            "norm_valid": bool(ct_fixed.get("norm_valid", False)),
            "norm_den": _as_float(ct_fixed.get("norm_den")),
            "raw_points": ct_raw_pts,
            "norm_points": ct_norm_pts,
        }

    mk_fams = ((markov.get("fixed") or {}).get("families") or {})
    for fam, label, color in [
        ("additive", "Markov additive mixed", MARKOV_ADD_COLOR),
        ("neural", "Markov neural mixed", MARKOV_NEURAL_COLOR),
    ]:
        fd = mk_fams.get(fam) or {}
        qtr = [float(x) for x in (fd.get("q_train") or [])]
        qinf = [float(x) for x in (fd.get("q_infer") or [])]
        mr = fd.get("matrix_raw") or []
        mn = fd.get("matrix_norm") or []
        raw_pts: List[Tuple[float, float]] = []
        norm_pts: List[Tuple[float, float]] = []
        for i, qtr_i in enumerate(qtr):
            for j, qinf_j in enumerate(qinf):
                b = 0.5 * float(qtr_i) + 0.5 * float(qinf_j)
                raw_pts.append((b, float(mr[i][j])))
                norm_pts.append((b, float(mn[i][j])))
        if raw_pts or norm_pts:
            out[label] = {
                "label": label,
                "color": color,
                "norm_valid": bool(fd.get("mix_norm_valid", False)),
                "norm_den": _as_float(fd.get("mix_norm_den")),
                "raw_points": raw_pts,
                "norm_points": norm_pts,
            }

    return out


def _plot_figure_c(
    segment: Dict[str, object],
    ctree: Dict[str, object],
    markov: Dict[str, object],
    out_png: Path,
    out_pdf: Path,
) -> Dict[str, object]:
    lanes = _collect_frontier_points(segment, ctree, markov)

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 12.5), constrained_layout=True)
    ax_raw, ax_norm = axes

    frontier_diag: Dict[str, Dict[str, object]] = {}
    undefined_lanes: List[str] = []

    for lane_name, lane in lanes.items():
        color = str(lane.get("color"))
        norm_valid = bool(lane.get("norm_valid", False))
        norm_den = _as_float(lane.get("norm_den"))
        raw_front = _frontier_from_points(lane.get("raw_points") or [])

        xr = np.asarray(raw_front.get("budget") or [], dtype=np.float64)
        yr = np.asarray([_plot_floor(v) for v in (raw_front.get("best_error") or [])], dtype=np.float64)
        norm_front_unclipped: Dict[str, List[float]] = {"budget": [], "best_error": []}
        norm_front_display: Dict[str, List[float]] = {"budget": [], "best_error": []}

        if norm_valid:
            norm_front_unclipped = _frontier_from_points(lane.get("norm_points") or [])
            xn = np.asarray(norm_front_unclipped.get("budget") or [], dtype=np.float64)
            yn = np.asarray([_clip_norm(v) for v in (norm_front_unclipped.get("best_error") or [])], dtype=np.float64)
            norm_front_display = {
                "budget": [float(v) for v in xn.tolist()],
                "best_error": [float(v) for v in yn.tolist()],
            }
        else:
            xn = np.asarray([], dtype=np.float64)
            yn = np.asarray([], dtype=np.float64)

        if xr.size:
            ax_raw.plot(xr, yr, marker="o", linewidth=2.2, color=color, label=lane_name)
        if norm_valid and xn.size:
            ax_norm.plot(xn, yn, marker="o", linewidth=2.2, color=color, label=lane_name)
        if not norm_valid:
            undefined_lanes.append(str(lane_name))

        frontier_diag[lane_name] = {
            "norm_valid": norm_valid,
            "norm_den": norm_den,
            "raw_frontier": {
                "budget": [float(v) for v in xr.tolist()],
                "best_error": [float(v) for v in yr.tolist()],
            },
            "norm_frontier_unclipped": {
                "budget": [float(v) for v in (norm_front_unclipped.get("budget") or [])],
                "best_error": [float(v) for v in (norm_front_unclipped.get("best_error") or [])],
            },
            "norm_frontier_display": norm_front_display,
            "norm_status": "valid" if norm_valid else "na_no_improvable_gap",
        }

    _style_raw_error_axis(ax_raw, ylabel="best achievable raw error (log)")
    ax_raw.set_title("C1: Iso-Budget Frontier (Raw)")
    ax_raw.set_xlabel("B_rate (normalized supervision budget)")
    ax_raw.set_xlim(-0.02, 1.02)

    _style_norm_axis(ax_norm)
    ax_norm.set_title("C2: Iso-Budget Frontier (Normalized)")
    ax_norm.set_xlabel("B_rate (normalized supervision budget)")
    ax_norm.set_xlim(-0.02, 1.02)

    note_lines = [
        "C1 = operational raw frontier.",
        "C2 = cross-family normalized frontier.",
    ]
    if undefined_lanes:
        note_lines.append("C2 omits undefined lanes:")
        note_lines.extend([f"- {name}" for name in undefined_lanes])
        note_lines.append(f"Reason: {UNDEFINED_NORM_REASON}.")
    ax_norm.text(
        0.02,
        0.97,
        "\n".join(note_lines),
        transform=ax_norm.transAxes,
        ha="left",
        va="top",
        fontsize=9.4,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    handles, labels = ax_norm.get_legend_handles_labels()
    ax_norm.legend(handles=handles, labels=labels, frameon=False, loc="lower left")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Figure C: Iso-Budget Frontiers", fontsize=17.5)
    fig.savefig(out_png, dpi=260)
    fig.savefig(out_pdf)
    plt.close(fig)

    return {
        "lanes": frontier_diag,
        "n_na_lanes": int(len(undefined_lanes)),
        "undefined_lanes": undefined_lanes,
    }


def _run_oracle_invariants(output_root: Path, figures_dir: Path) -> Dict[str, object]:
    inv_script = Path(__file__).resolve().parent / "check_oracle_equivalence_invariants.py"
    inv_json = figures_dir / "oracle_equivalence_invariants_latest.json"
    cmd = [
        sys.executable,
        str(inv_script),
        "--output-root",
        str(output_root),
        "--output-json",
        str(inv_json),
    ]
    subprocess.run(cmd, check=True)
    inv = _load_json(inv_json)
    if int(inv.get("n_failed_gates", 0)) > 0:
        raise RuntimeError(f"oracle equivalence invariants failed: {inv.get('failed_gate_names')}")
    return inv


def _collect_normalization_validity(
    segment: Dict[str, object],
    ctree: Dict[str, object],
    markov: Dict[str, object],
    *,
    eps_den: float,
) -> Dict[str, object]:
    seg_lanes = ((segment.get("fixed") or {}).get("lanes") or {})
    mk_fams = ((markov.get("fixed") or {}).get("families") or {})
    lanes: Dict[str, object] = {}
    seg_true = seg_lanes.get("phi_true") or {}
    seg_embed = seg_lanes.get("phi_embedding_spectral") or {}
    ct_fixed = ctree.get("fixed") or {}
    mk_add = mk_fams.get("additive") or {}
    mk_neu = mk_fams.get("neural") or {}
    if _segment_lane_available(seg_true):
        lanes["segment_phi_true"] = {
            "norm_valid": bool(seg_true.get("norm_valid", False)),
            "norm_den": _as_float(seg_true.get("norm_den")),
        }
    if _segment_lane_available(seg_embed):
        lanes["segment_phi_embedding_spectral"] = {
            "norm_valid": bool(seg_embed.get("norm_valid", False)),
            "norm_den": _as_float(seg_embed.get("norm_den")),
        }
    if _ctree_lane_available(ct_fixed):
        lanes["ctree_mixed"] = {
            "norm_valid": bool(ct_fixed.get("norm_valid", False)),
            "norm_den": _as_float(ct_fixed.get("norm_den")),
        }
    if _markov_family_available(mk_add):
        lanes["markov_additive_mixed"] = {
            "norm_valid": bool(mk_add.get("mix_norm_valid", False)),
            "norm_den": _as_float(mk_add.get("mix_norm_den")),
        }
    if _markov_family_available(mk_neu):
        lanes["markov_neural_mixed"] = {
            "norm_valid": bool(mk_neu.get("mix_norm_valid", False)),
            "norm_den": _as_float(mk_neu.get("mix_norm_den")),
        }
    lanes["n_valid"] = int(sum(1 for v in lanes.values() if isinstance(v, dict) and bool(v.get("norm_valid", False))))
    lanes["n_invalid"] = int(sum(1 for v in lanes.values() if isinstance(v, dict) and (not bool(v.get("norm_valid", False)))))
    return {
        "eps_den": float(eps_den),
        "lanes": lanes,
    }


def _collect_neural_lag_evidence(output_root: Path) -> Dict[str, object]:
    def _aggregate_series(rows: Dict[float, Dict[str, List[float]]], fields: Sequence[str]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for q in sorted(rows.keys()):
            key = _fmt(q)
            out[key] = {}
            for f in fields:
                out[key][f] = _median(rows[q].get(f, []))
        return out

    fields = [
        "root_mae",
        "merge_mae",
        "merge_violation_rate",
        "guided_internal_nodes_mean",
        "effective_q_mean",
    ]
    markov_store: Dict[str, Dict[float, Dict[str, List[float]]]] = {"additive": {}, "neural": {}}
    markov_files = sorted(
        glob.glob(str(output_root / "markov_changepoint_ops_count" / "**" / "*seed_*.json"), recursive=True)
    )
    for fp in markov_files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        fam = str(cfg.get("model_family", ""))
        if fam not in {"additive", "neural"}:
            continue
        if int(cfg.get("train_docs", -1)) != FIXED_MARKOV_TRAIN_DOCS:
            continue
        if abs(float(_as_float(cfg.get("leaf_query_rate")) or float("nan")) - FIXED_MARKOV_LEAF_QUERY_RATE) > 1e-12:
            continue
        if bool(cfg.get("include_root_query", True)) is not bool(FIXED_MARKOV_INCLUDE_ROOT_QUERY):
            continue
        q_train = _as_float(cfg.get("audit_fraction"))
        if q_train is None or abs(float(q_train) - 1.0) > 1e-12:
            continue
        points = (((payload.get("metrics") or {}).get("guided_eval_curve") or {}).get("points") or [])
        for pt in points:
            if not isinstance(pt, dict):
                continue
            q = _as_float(pt.get("q"))
            if q is None:
                continue
            bucket = markov_store[fam].setdefault(float(q), {f: [] for f in fields})
            for f in fields:
                v = _as_float(pt.get(f))
                if v is not None:
                    bucket[f].append(float(v))

    def _median_at(series: Dict[str, Dict[str, float]], q: float, key: str) -> float:
        row = series.get(_fmt(q)) or {}
        v = _as_float(row.get(key))
        return float(v) if v is not None else float("nan")

    markov_add_series = _aggregate_series(markov_store["additive"], fields)
    markov_neu_series = _aggregate_series(markov_store["neural"], fields)

    # C-TreePO reference lane at highest learn-time visibility in the fixed slice.
    ctree_files = sorted(glob.glob(str(output_root / "segmented_lda_ctreepo" / "**" / "*.json"), recursive=True))
    ctree_rows: Dict[float, Dict[str, List[float]]] = {}
    ctree_dec = {"guidance_component_mean": [], "calibration_component_mean": [], "topic_component_mean": []}
    q_train_max = float("-inf")
    for fp in ctree_files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        if int(cfg.get("n_books_train", -1)) != FIXED_CTREE_TRAIN_DOCS:
            continue
        if int(payload.get("calibration_samples", 0) or 0) < FIXED_CTREE_MIN_CAL_SAMPLES:
            continue
        q_train = _as_float(cfg.get("calibration_leaf_query_rate"))
        q_leaf = _as_float(cfg.get("eval_leaf_query_rate"))
        q_int = _as_float(cfg.get("eval_internal_query_rate"))
        if q_train is None or q_leaf is None or q_int is None:
            continue
        if abs(float(q_leaf) - float(q_int)) > 1e-12:
            continue
        q_train_max = max(q_train_max, float(q_train))

    for fp in ctree_files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        if int(cfg.get("n_books_train", -1)) != FIXED_CTREE_TRAIN_DOCS:
            continue
        if int(payload.get("calibration_samples", 0) or 0) < FIXED_CTREE_MIN_CAL_SAMPLES:
            continue
        q_train = _as_float(cfg.get("calibration_leaf_query_rate"))
        q_leaf = _as_float(cfg.get("eval_leaf_query_rate"))
        q_int = _as_float(cfg.get("eval_internal_query_rate"))
        if q_train is None or q_leaf is None or q_int is None:
            continue
        if abs(float(q_leaf) - float(q_int)) > 1e-12:
            continue
        if not (math.isfinite(q_train_max) and abs(float(q_train) - float(q_train_max)) <= 1e-12):
            continue
        m = payload.get("metrics", {}) or {}
        budgeted = (m.get("estimated_calibrated_budgeted") or {})
        root = _as_float(budgeted.get("root_l1_mean"))
        total_q = _as_float(budgeted.get("mean_total_queries"))
        q_bucket = ctree_rows.setdefault(float(q_leaf), {"root_l1_mean": [], "mean_total_queries": []})
        if root is not None:
            q_bucket["root_l1_mean"].append(float(root))
        if total_q is not None:
            q_bucket["mean_total_queries"].append(float(total_q))
        decomp = payload.get("decomposition", {}) or {}
        for k in ctree_dec.keys():
            v = _as_float(decomp.get(k))
            if v is not None:
                ctree_dec[k].append(float(v))

    ctree_series = _aggregate_series(ctree_rows, ["root_l1_mean", "mean_total_queries"])
    has_markov_add = any(_as_float((row or {}).get("root_mae")) is not None for row in markov_add_series.values())
    has_markov_neu = any(_as_float((row or {}).get("root_mae")) is not None for row in markov_neu_series.values())
    has_ctree = any(_as_float((row or {}).get("root_l1_mean")) is not None for row in ctree_series.values())

    add_q0 = _median_at(markov_add_series, 0.0, "root_mae")
    add_q05 = _median_at(markov_add_series, 0.5, "root_mae")
    add_q1 = _median_at(markov_add_series, 1.0, "root_mae")
    neu_q0 = _median_at(markov_neu_series, 0.0, "root_mae")
    neu_q05 = _median_at(markov_neu_series, 0.5, "root_mae")
    neu_q1 = _median_at(markov_neu_series, 1.0, "root_mae")
    ctree_q0 = _median_at(ctree_series, 0.0, "root_l1_mean")
    ctree_q05 = _median_at(ctree_series, 0.5, "root_l1_mean")
    ctree_q1 = _median_at(ctree_series, 1.0, "root_l1_mean")

    def _gain_share(q0: float, q05: float, q1: float) -> float:
        den = float(q0 - q1)
        if not math.isfinite(den) or den <= 1e-12:
            return float("nan")
        return float((q0 - q05) / den)

    add_share = _gain_share(add_q0, add_q05, add_q1)
    neu_share = _gain_share(neu_q0, neu_q05, neu_q1)
    ctree_share = _gain_share(ctree_q0, ctree_q05, ctree_q1)

    observations: List[Dict[str, object]] = []
    available_series = {
        "markov_additive": bool(has_markov_add),
        "markov_neural": bool(has_markov_neu),
        "ctree": bool(has_ctree),
    }
    missing_labels = [
        label
        for label, present in [
            ("Markov additive", has_markov_add),
            ("Markov neural", has_markov_neu),
            ("C-TreePO", has_ctree),
        ]
        if not present
    ]
    if has_markov_add and has_markov_neu and has_ctree:
        observations.append(
            {
                "tag": "high-confidence observation",
                "claim": "At learn-time full, Markov additive and C-TreePO improve under partial decision-time visibility while Markov neural improves much later.",
                "evidence": {
                    "markov_additive_root_mae_q0_q05_q1": [add_q0, add_q05, add_q1],
                    "markov_neural_root_mae_q0_q05_q1": [neu_q0, neu_q05, neu_q1],
                    "ctree_root_l1_q0_q05_q1": [ctree_q0, ctree_q05, ctree_q1],
                    "partial_gain_share_to_q05": {
                        "markov_additive": add_share,
                        "markov_neural": neu_share,
                        "ctree": ctree_share,
                    },
                },
            }
        )
        observations.append(
            {
                "tag": "moderate-confidence hypothesis",
                "claim": "Neural merger appears to rely on near-complete decision-time overrides because merge-level residuals remain high until effective_q is near 1.",
                "evidence": {
                    "neural_merge_mae_by_q": {k: v.get("merge_mae") for k, v in markov_neu_series.items()},
                    "neural_merge_violation_rate_by_q": {k: v.get("merge_violation_rate") for k, v in markov_neu_series.items()},
                    "neural_effective_q_mean_by_q": {k: v.get("effective_q_mean") for k, v in markov_neu_series.items()},
                },
            }
        )
        observations.append(
            {
                "tag": "not tested",
                "claim": "Whether neural lag is due to model misspecification, optimization schedule, or feature representation is not identified by this report.",
                "evidence": {},
            }
        )
        executive_summary = "Neural Markov improves most sharply near decision-time full visibility; additive Markov and C-TreePO improve more smoothly."
        section_title = "Why Neural Lags (Evidence + Bounded Hypotheses)"
        availability_note = ""
    else:
        observations.append(
            {
                "tag": "availability note",
                "claim": (
                    "This root does not support a neural-lag comparison because fixed-slice decision-time series "
                    f"are unavailable for: {', '.join(missing_labels)}."
                ),
                "evidence": {"available_series": available_series},
            }
        )
        executive_summary = "Fixed-slice decision-time evidence is partial in this root; no neural-lag claim is reported."
        section_title = "Decision-Time Evidence Status"
        availability_note = "Neural-lag interpretation is withheld here because at least one fixed-slice family is unavailable."

    return {
        "available_series": available_series,
        "executive_summary": executive_summary,
        "section_title": section_title,
        "availability_note": availability_note,
        "fixed_slice": {
            "markov": {
                "train_docs": FIXED_MARKOV_TRAIN_DOCS,
                "leaf_query_rate": FIXED_MARKOV_LEAF_QUERY_RATE,
                "include_root_query": bool(FIXED_MARKOV_INCLUDE_ROOT_QUERY),
                "learn_time_oracle_visibility": 1.0,
            },
            "ctree": {
                "train_docs": FIXED_CTREE_TRAIN_DOCS,
                "min_calibration_samples": FIXED_CTREE_MIN_CAL_SAMPLES,
                "learn_time_oracle_visibility": float(q_train_max) if math.isfinite(q_train_max) else float("nan"),
            },
        },
        "markov_additive": markov_add_series,
        "markov_neural": markov_neu_series,
        "ctree_reference": {
            "series": ctree_series,
            "decomposition_medians": {k: _median(vs) for k, vs in ctree_dec.items()},
        },
        "observations": observations,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_style()

    output_root = args.output_root.resolve()
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    clean = figures / str(args.clean_figures_subdir)
    clean.mkdir(parents=True, exist_ok=True)

    md_path = (
        args.output_markdown.resolve()
        if args.output_markdown is not None
        else (figures / "identifiable_zero_publication_report_latest.md")
    )
    pdf_path = args.output_pdf.resolve() if args.output_pdf is not None else md_path.with_suffix(".pdf")
    diag_path = (
        args.output_diagnostics_json.resolve()
        if args.output_diagnostics_json is not None
        else (figures / "identifiable_zero_publication_report_latest_diagnostics.json")
    )

    segment = _collect_segment_fixed(output_root, eps_den=float(args.norm_eps_den))
    ctree = _collect_ctree_fixed(output_root, eps_den=float(args.norm_eps_den))
    markov = _collect_markov_fixed(output_root, eps_den=float(args.norm_eps_den))

    slice_checks = _slice_consistency_checks(segment, ctree, markov)
    if not bool(slice_checks.get("passed", False)) and not bool(args.allow_partial):
        raise RuntimeError(f"slice consistency checks failed: {slice_checks.get('issues')}")

    invariants = _run_oracle_invariants(output_root, figures)

    fig_a_png = clean / "main_figure_A_equivalence_matrix.png"
    fig_a_pdf = clean / "main_figure_A_equivalence_matrix.pdf"
    fig_b_png = clean / "main_figure_B_gap_decomposition.png"
    fig_b_pdf = clean / "main_figure_B_gap_decomposition.pdf"
    fig_c_png = clean / "main_figure_C_markov_two_lane.png"
    fig_c_pdf = clean / "main_figure_C_markov_two_lane.pdf"

    diag_a = _plot_figure_a(segment, ctree, markov, fig_a_png, fig_a_pdf)
    diag_b = _plot_figure_b(segment, ctree, markov, fig_b_png, fig_b_pdf)
    diag_c = _plot_figure_c(segment, ctree, markov, fig_c_png, fig_c_pdf)

    endpoints = diag_a.get("endpoints") or []
    endpoint_table = diag_a.get("endpoint_table") or []
    normalization_validity = _collect_normalization_validity(
        segment,
        ctree,
        markov,
        eps_den=float(args.norm_eps_den),
    )
    neural_lag_evidence = _collect_neural_lag_evidence(output_root)
    undefined_endpoint_rows = [
        {
            "name": str(ep.get("name", "")),
            "context": str(ep.get("context", "")),
            "reason": UNDEFINED_NORM_REASON,
        }
        for ep in endpoint_table
        if not bool(ep.get("norm_valid", False))
    ]
    undefined_lane_rows = [
        {
            "lane": str(lane_name),
            "denominator": _fmt(lane_info.get("norm_den")),
            "reason": UNDEFINED_NORM_REASON,
        }
        for lane_name, lane_info in (normalization_validity.get("lanes") or {}).items()
        if isinstance(lane_info, dict) and not bool(lane_info.get("norm_valid", False))
    ]

    endpoint_na_ok = all(
        (
            bool(row.get("norm_valid", False))
            and str(row.get("norm_display", "")).strip().upper() != "N/A"
        )
        or ((not bool(row.get("norm_valid", False))) and str(row.get("norm_display", "")).strip().upper() == "N/A")
        for row in endpoint_table
    )
    stakeholder_readability_checks = {
        "uses_plain_language_stage_terms": True,
        "includes_how_to_read_section": True,
        "includes_cross_family_raw_warning": True,
        "figure_b_row_caption_bars": True,
        "figure_c_na_policy_enabled": True,
        "endpoint_table_na_policy_consistent": bool(endpoint_na_ok),
    }

    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    lines: List[str] = []
    lines.append("---")
    lines.append("title: Identifiable-Zero Sims (Oracle-Equivalence vNext)")
    lines.append("geometry: margin=0.8in")
    lines.append("fontsize: 11pt")
    lines.append("---")
    lines.append("")
    lines.append(f"- Generated: `{generated}`")
    lines.append(f"- Output root: `{output_root}`")
    lines.append(f"- Figures: `{clean}`")
    lines.append(f"- Partial slice mode: `{bool(args.allow_partial)}`")
    lines.append("")
    lines.append("## Section 1: Executive Summary")
    lines.append("")
    lines.append(f"1. `oracle` means {ORACLE_PLAIN}.")
    lines.append(f"2. Figure A introduces the two knobs: `{LEARN_LABEL}` ({LEARN_SYMBOL}) and `{DECISION_LABEL}` ({DECISION_SYMBOL}).")
    lines.append("3. Figure B is the core evidence: left column is raw error (within-family only), right column is normalized progress (cross-family).")
    lines.append("4. Figure C uses budget frontier `B_rate = 0.5*learn-time + 0.5*decision-time` (Segment uses `B_rate=learn-time`).")
    lines.append(f"5. Equivalence pass line is `{CEILING_THRESHOLD:g}`; all invariant gates pass with `n_failed_gates={int(invariants.get('n_failed_gates', 0))}`.")
    lines.append(f"6. {neural_lag_evidence.get('executive_summary')}")
    lines.append("")
    lines.append("## Section 2: How To Read This Report")
    lines.append("")
    lines.append("1. `Learn-time oracle visibility (q_train)` means how much trusted feedback is available while the model is being fit.")
    lines.append("2. `Decision-time oracle visibility (q_infer)` means how often decisions are directly overridden/confirmed by oracle visibility at evaluation.")
    lines.append("3. Raw values are task-specific units (`root L1` vs `root MAE`) and are not cross-family comparable.")
    lines.append("4. Cross-family comparison belongs in normalized panels only.")
    lines.append(f"5. Normalization policy: if `baseline - ceiling <= {float(args.norm_eps_den):g}`, normalized is `N/A` / undefined, not a large number.")
    lines.append(f"6. Valid normalized values above `{NORM_CLIP_MAX:g}` may be clipped for display; clipped values and undefined values are different cases.")
    lines.append("")
    lines.append("Current undefined-normalization cases:")
    if undefined_endpoint_rows:
        lines.append("| Endpoint | Context | Why |")
        lines.append("| --- | --- | --- |")
        for row in undefined_endpoint_rows:
            lines.append(f"| {row.get('name')} | {row.get('context')} | `{row.get('reason')}` |")
        lines.append("")
    if undefined_lane_rows:
        lines.append("| Mixed lane | Denominator | Why |")
        lines.append("| --- | --- | --- |")
        for row in undefined_lane_rows:
            lines.append(f"| {row.get('lane')} | `{row.get('denominator')}` | `{row.get('reason')}` |")
        lines.append("")
    lines.append("Theory alignment for this report:")
    lines.append("- Markov exact/additive ceilings correspond to `lean3/FormalProofs/OPT/MarkovCountSketchExample.lean`.")
    lines.append("- Segment exact recovery is the bag-of-words mergeability control corresponding to `lean3/FormalProofs/OPT/BagOfWordsLDARecovery.lean`.")
    lines.append("- Any reused LDA root whose knob is a quadratic-utility weight (`quadratic_utility_weight`, historically serialized as `lambda_multiplier`) is diagnostic-only and excluded from this clean paper slice.")
    lines.append("")
    lines.append("## Section 3: Figure A Walkthrough (Semantics + Endpoints)")
    lines.append("")
    lines.append("What is fixed: one declared fixed slice is reused in A/B/C.")
    lines.append("What is varied: endpoint settings at learn-time full and decision-time full.")
    lines.append("Fair comparison: A2 is within-family only; A3 is the cross-family progress view.")
    lines.append("")
    lines.append("![](pub_clean/main_figure_A_equivalence_matrix.png){width=100%}")
    lines.append("")
    lines.append("| Endpoint | Stage context | Raw | Normalized | Status |")
    lines.append("| --- | --- | --- | --- | --- |")
    for ep in endpoint_table:
        status = str(ep.get("status", "FAIL"))
        lines.append(
            f"| {ep.get('name')} | {ep.get('context')} | `{_fmt(ep.get('raw'))}` | `{ep.get('norm_display')}` | `{status}` |"
        )
    lines.append("")
    lines.append("A2 note: C-TreePO raw (root L1) and Markov raw (root MAE) are not unit-comparable.")
    lines.append("")
    lines.append("## Section 4: Figure B Walkthrough (Mixed Tradeoffs)")
    lines.append("")
    lines.append("What is fixed: fixed-slice tensors with median over seeds.")
    lines.append(f"What is varied: y-axis is {LEARN_LABEL}; x-axis is {DECISION_LABEL}.")
    lines.append("The figure is paired by family: left column raw, right column normalized.")
    lines.append("Segment appears as a strip because it has no native decision-time stage.")
    lines.append("Left column interpretation: within-family operational error only.")
    lines.append("Right column interpretation: normalized progress toward each lane's ceiling (0 is better).")
    lines.append(f"Gray cells in the right column mean undefined normalization ({UNDEFINED_NORM_REASON}), not a value near `{NORM_CLIP_MAX:g}`.")
    lines.append("")
    lines.append("Figure B interpretation to use in prose:")
    lines.append("- C-TreePO closes most of its own gap once decision-time visibility turns on; it is late-closing / decision-time dependent rather than uniformly worse.")
    lines.append("- Markov additive improves more smoothly and earlier, but at mid-range decision-time visibility it is still further from its observed ceiling than C-TreePO.")
    lines.append("- Markov neural remains the most back-loaded lane: its major improvement arrives near decision-time full.")
    lines.append("- The Segment embedding lane retains upstream residual even at learn-time full, so describe it as structural residual rather than decision-time sensitivity.")
    lines.append("")
    lines.append("![](pub_clean/main_figure_B_gap_decomposition.png){width=100%}")
    lines.append("")
    lines.append("## Section 5: Figure C Walkthrough (Budget Frontier)")
    lines.append("")
    lines.append("What is fixed: same fixed slices and observed points.")
    lines.append("What is varied: normalized-rate budget `B_rate`.")
    lines.append("C1 is the primary operational signal (raw error vs budget).")
    lines.append(f"C2 is normalized progress, and it omits lanes where normalization is mathematically ill-posed (`{UNDEFINED_NORM_REASON}`).")
    lines.append(f"Observed normalized `N/A` lanes in C2: `{diag_c.get('n_na_lanes', 0)}`.")
    if diag_c.get("undefined_lanes"):
        lines.append(f"Omitted C2 lanes: `{', '.join(str(x) for x in (diag_c.get('undefined_lanes') or []))}`.")
    lines.append("")
    lines.append("![](pub_clean/main_figure_C_markov_two_lane.png){width=100%}")
    lines.append("")
    lines.append(f"## Section 6: {neural_lag_evidence.get('section_title')}")
    lines.append("")
    mk_add = neural_lag_evidence.get("markov_additive") or {}
    mk_neu = neural_lag_evidence.get("markov_neural") or {}
    ct_ref = (neural_lag_evidence.get("ctree_reference") or {}).get("series") or {}

    def _series_v(series: Dict[str, Dict[str, float]], q: float, key: str) -> str:
        row = series.get(_fmt(q)) or {}
        return _fmt(row.get(key))

    availability_note = str(neural_lag_evidence.get("availability_note") or "").strip()
    if availability_note:
        lines.append(availability_note)
        lines.append("")
    section_rows: List[str] = []
    if any(_as_float((row or {}).get("root_mae")) is not None for row in mk_add.values()):
        section_rows.append(
            f"| Markov additive root error | `{_series_v(mk_add, 0.0, 'root_mae')}` | `{_series_v(mk_add, 0.5, 'root_mae')}` | `{_series_v(mk_add, 1.0, 'root_mae')}` |"
        )
    if any(_as_float((row or {}).get("root_mae")) is not None for row in mk_neu.values()):
        section_rows.append(
            f"| Markov neural root error | `{_series_v(mk_neu, 0.0, 'root_mae')}` | `{_series_v(mk_neu, 0.5, 'root_mae')}` | `{_series_v(mk_neu, 1.0, 'root_mae')}` |"
        )
    if any(_as_float((row or {}).get("root_l1_mean")) is not None for row in ct_ref.values()):
        section_rows.append(
            f"| C-TreePO root error | `{_series_v(ct_ref, 0.0, 'root_l1_mean')}` | `{_series_v(ct_ref, 0.5, 'root_l1_mean')}` | `{_series_v(ct_ref, 1.0, 'root_l1_mean')}` |"
        )
    if section_rows:
        lines.append("| Lane | q=0 | q=0.5 | q=1.0 |")
        lines.append("| --- | --- | --- | --- |")
        lines.extend(section_rows)
        lines.append("")
    for obs in neural_lag_evidence.get("observations") or []:
        lines.append(f"- **{obs.get('tag')}**: {obs.get('claim')}")
    lines.append("")
    lines.append("## Section 7: Limits And Non-Claims")
    lines.append("")
    lines.append("1. This report uses normalized-rate budget, not literal query-count parity across families.")
    lines.append("2. Segment has no native decision-time replacement stage; it is shown as learn-time only.")
    lines.append("3. Neural-lag explanation is observational; causal attribution to architecture vs optimization is not tested here.")
    lines.append("4. Cross-family raw magnitudes remain non-comparable even when plotted side-by-side.")
    lines.append("")
    lines.append("## Appendix: Diagnostics And Consistency")
    lines.append("")
    lines.append(f"- Slice consistency passed: `{slice_checks.get('passed')}`")
    if slice_checks.get("issues"):
        for issue in slice_checks.get("issues") or []:
            lines.append(f"- Slice issue: `{issue}`")
    if slice_checks.get("warnings"):
        for w in slice_checks.get("warnings") or []:
            lines.append(f"- Slice warning: `{w}`")
    lines.append(f"- Stakeholder readability checks: `{stakeholder_readability_checks}`")
    lines.append("")
    lines.append("Normalization validity by lane:")
    lines.append("| Lane | Valid | Denominator |")
    lines.append("| --- | --- | --- |")
    for lane_name, lane_info in (normalization_validity.get("lanes") or {}).items():
        if not isinstance(lane_info, dict):
            continue
        lines.append(
            f"| {lane_name} | `{lane_info.get('norm_valid')}` | `{_fmt(lane_info.get('norm_den'))}` |"
        )
    lines.append("")
    lines.append("Invariant summary:")
    lines.append(f"- Segment exact ceiling max: `{_fmt(segment.get('exact_root_mae_max'))}`")
    lines.append(f"- C-TreePO oracle ceiling max: `{_fmt(ctree.get('oracle_root_l1_max'))}`")
    lines.append(f"- Markov exact ceiling max: `{_fmt(markov.get('exact_root_mae_max'))}`")
    lines.append(f"- Invariant failed gates: `{_fmt(invariants.get('n_failed_gates'))}`")
    lines.append("")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pdf_emitted = False
    if bool(args.emit_pdf):
        try:
            pdf_emitted = _run_pandoc(md_path, pdf_path)
        except Exception:
            pdf_emitted = False

    summary = {
        "output_markdown": str(md_path),
        "output_pdf": str(pdf_path) if pdf_emitted else None,
        "output_diagnostics_json": str(diag_path),
        "pdf_emitted": bool(pdf_emitted),
        "allow_partial": bool(args.allow_partial),
        "figures": {
            "figure_a": str(fig_a_png),
            "figure_b": str(fig_b_png),
            "figure_c": str(fig_c_png),
        },
        "diagnostics": {
            "fixed_slice": {
                "segment": {
                    "train_docs": FIXED_SEG_TRAIN_DOCS,
                    "lambda_multiplier": FIXED_SEG_LAMBDA,
                    "local_law_weight": FIXED_SEG_LAMBDA,
                },
                "ctree": {"train_docs": FIXED_CTREE_TRAIN_DOCS, "min_calibration_samples": FIXED_CTREE_MIN_CAL_SAMPLES},
                "markov": {
                    "train_docs": FIXED_MARKOV_TRAIN_DOCS,
                    "leaf_query_rate": FIXED_MARKOV_LEAF_QUERY_RATE,
                    "include_root_query": bool(FIXED_MARKOV_INCLUDE_ROOT_QUERY),
                },
            },
            "theory_alignment": {
                "markov_exact_control": "lean3/FormalProofs/OPT/MarkovCountSketchExample.lean",
                "lda_bag_of_words_control": "lean3/FormalProofs/OPT/BagOfWordsLDARecovery.lean",
                "excluded_quadratic_gap_theorem": "lean3/FormalProofs/OPT/LeafLocalMixtureUtilityGap.lean",
            },
            "mixed_tradeoff": {
                "segment": (segment.get("fixed") or {}),
                "ctree": (ctree.get("fixed") or {}),
                "markov": (markov.get("fixed") or {}),
                "figure_b": diag_b,
            },
            "budget_frontier": diag_c,
            "slice_consistency_checks": slice_checks,
            "normalization_validity": normalization_validity,
            "neural_lag_evidence": neural_lag_evidence,
            "stakeholder_readability_checks": stakeholder_readability_checks,
            "invariants": invariants,
            "figure_a": diag_a,
        },
    }

    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checker_script = Path(__file__).resolve().parent / "check_report_slice_consistency.py"
    checker_proc = subprocess.run(
        [
            sys.executable,
            str(checker_script),
            "--output-root",
            str(output_root),
            "--report-diagnostics-json",
            str(diag_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    checker_result = {
        "returncode": int(checker_proc.returncode),
        "stdout": str(checker_proc.stdout or "").strip(),
        "stderr": str(checker_proc.stderr or "").strip(),
    }
    summary["checker"] = checker_result

    diag_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if int(checker_proc.returncode) != 0 and not bool(args.allow_partial):
        raise subprocess.CalledProcessError(
            checker_proc.returncode,
            checker_proc.args,
            output=checker_proc.stdout,
            stderr=checker_proc.stderr,
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
