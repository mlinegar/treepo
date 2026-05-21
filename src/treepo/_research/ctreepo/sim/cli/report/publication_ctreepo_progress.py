#!/usr/bin/env python3
"""Interim progress plots for publication C-TreePO CPU passes.

This is designed for partial outputs while long runs are still in flight. It scans
`<output-root>/segmented_lda_ctreepo/equivalence/**` and produces lane-aware progress
and learnability figures without requiring full completion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


GOOD_BAD_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "good_bad_error",
    ["#1a9850", "#ffffbf", "#d73027"],  # green -> yellow -> red
    N=256,
)


@dataclass(frozen=True)
class _Row:
    regime: str
    lane: str
    train: int
    leaf_tokens: int
    cal_rate: float
    q_leaf: float
    q_internal: float
    seed: int
    root_l1: float
    topic_phi_l2: Optional[float]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create interim progress plots for publication C-TreePO CPU pass.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <output-root>/figures/publication_progress",
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
    if abs(float(x)) < 1e-3:
        return f"{float(x):.1e}"
    return f"{float(x):.3g}"


def _run_pandoc(md_path: Path, pdf_path: Path) -> bool:
    if shutil.which("pandoc") is None or shutil.which("pdflatex") is None:
        return False
    subprocess.run(
        ["pandoc", str(md_path.name), "-o", str(pdf_path.name), "--pdf-engine=pdflatex"],
        cwd=str(md_path.parent),
        check=True,
    )
    return True


def _scan_rows(output_root: Path) -> List[_Row]:
    rows: List[_Row] = []
    pat = str(output_root / "segmented_lda_ctreepo" / "equivalence" / "**" / "*.json")
    for fp in glob.glob(pat, recursive=True):
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

        parts = path.parts
        lane = next((p for p in parts if str(p).startswith("lane_")), "lane_unknown")
        regime = "unknown"
        if "equivalence" in parts:
            i = parts.index("equivalence")
            if i + 1 < len(parts):
                regime = str(parts[i + 1])

        train = int(cfg.get("n_books_train", -1))
        leaf = int(cfg.get("fixed_leaf_tokens", -1))
        cal = _as_float(cfg.get("calibration_leaf_query_rate"))
        ql = _as_float(cfg.get("eval_leaf_query_rate"))
        qi = _as_float(cfg.get("eval_internal_query_rate"))
        seed = int(cfg.get("seed", -1))
        root = _as_float(policy.get("root_l1_mean"))
        phi_l2 = _as_float(topic_meta.get("topic_phi_l2_error_mean"))
        if (
            train < 0
            or leaf < 0
            or seed < 0
            or cal is None
            or ql is None
            or qi is None
            or root is None
        ):
            continue

        rows.append(
            _Row(
                regime=regime,
                lane=lane,
                train=train,
                leaf_tokens=leaf,
                cal_rate=float(cal),
                q_leaf=float(ql),
                q_internal=float(qi),
                seed=seed,
                root_l1=float(root),
                topic_phi_l2=float(phi_l2) if phi_l2 is not None else None,
            )
        )
    return rows


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _heatmap(
    ax: plt.Axes,
    *,
    grid: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    title: str,
    vmin: float,
    vmax: float,
    cmap,
) -> None:
    im = ax.imshow(grid, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(list(range(len(x_labels))), labels=list(x_labels))
    ax.set_yticks(list(range(len(y_labels))), labels=list(y_labels))
    ax.set_xlabel("train")
    ax.set_ylabel("leaf_tokens")
    for yi in range(grid.shape[0]):
        for xi in range(grid.shape[1]):
            v = float(grid[yi, xi])
            if not math.isfinite(v):
                continue
            rgba = im.cmap(im.norm(v))
            lum = float(0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2])
            txt = "black" if lum >= 0.58 else "white"
            ax.text(xi, yi, _fmt_cell(v), ha="center", va="center", fontsize=8, color=txt)


def _grid_from_cells(
    xs: Sequence[int],
    ys: Sequence[int],
    cell_values: Dict[Tuple[int, int], List[float]],
) -> np.ndarray:
    out = np.full((len(ys), len(xs)), float("nan"), dtype=np.float64)
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            out[yi, xi] = _median(cell_values.get((int(x), int(y)), []))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_root)
    out_dir = Path(args.out_dir) if args.out_dir is not None else (output_root / "figures" / "publication_progress")
    pages = out_dir / "pages"
    pages.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    rows = _scan_rows(output_root)
    now = datetime.now(timezone.utc)
    diagnostics: Dict[str, object] = {
        "generated_at_utc": now.isoformat(),
        "output_root": str(output_root),
        "n_rows": int(len(rows)),
    }

    if not rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "publication_ctreepo_progress_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0

    # Global summary counts.
    lane_counts: Dict[str, int] = {}
    for r in rows:
        lane_counts[r.lane] = lane_counts.get(r.lane, 0) + 1
    diagnostics["lane_counts"] = lane_counts

    summary_png = pages / "progress_counts_by_lane.png"
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5), constrained_layout=True)
    lane_names = sorted(lane_counts.keys())
    lane_vals = [int(lane_counts[n]) for n in lane_names]
    ax.bar(lane_names, lane_vals, color="#4c78a8")
    ax.set_title("Completed result files by lane (partial run)")
    ax.set_ylabel("n completed JSON rows")
    ax.tick_params(axis="x", labelrotation=20)
    for i, v in enumerate(lane_vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
    _save_fig(fig, summary_png)

    page_paths: List[Path] = [summary_png]

    # Per-lane diagnostic pages.
    lane_keys = sorted({(r.regime, r.lane) for r in rows})
    lane_diag: Dict[str, object] = {}
    for regime, lane in lane_keys:
        sub = [r for r in rows if r.regime == regime and r.lane == lane]
        seeds = sorted({int(r.seed) for r in sub})
        n_seeds = int(len(seeds)) if seeds else 1
        q_pairs = sorted({(float(r.q_leaf), float(r.q_internal)) for r in sub})
        cal_vals = sorted({float(r.cal_rate) for r in sub})
        trains = sorted({int(r.train) for r in sub})
        leafs = sorted({int(r.leaf_tokens) for r in sub})
        if not trains or not leafs:
            continue

        # "Primary" slice: q=0 if present; otherwise min q. cal=0.1 if present; otherwise median-like.
        q_primary = (0.0, 0.0) if (0.0, 0.0) in q_pairs else q_pairs[0]
        if 0.1 in cal_vals:
            cal_primary = 0.1
        else:
            cal_primary = cal_vals[min(len(cal_vals) - 1, max(0, len(cal_vals) // 2))]

        prim = [
            r
            for r in sub
            if abs(r.q_leaf - q_primary[0]) <= 1e-12
            and abs(r.q_internal - q_primary[1]) <= 1e-12
            and abs(r.cal_rate - cal_primary) <= 1e-12
        ]

        cov_cells: Dict[Tuple[int, int], List[float]] = {}
        root_cells: Dict[Tuple[int, int], List[float]] = {}
        for t in trains:
            for lf in leafs:
                done = len([r for r in prim if r.train == t and r.leaf_tokens == lf])
                cov = float(done) / float(n_seeds)
                cov_cells.setdefault((t, lf), []).append(cov)
        for r in prim:
            root_cells.setdefault((int(r.train), int(r.leaf_tokens)), []).append(float(r.root_l1))

        cov_grid = _grid_from_cells(trains, leafs, cov_cells)
        root_grid = _grid_from_cells(trains, leafs, root_cells)
        root_vals = root_grid[np.isfinite(root_grid)]
        rvmin = float(np.min(root_vals)) if root_vals.size else 0.0
        rvmax = float(np.max(root_vals)) if root_vals.size else 1.0
        if not math.isfinite(rvmax) or rvmax <= rvmin:
            rvmax = rvmin + 1.0

        fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), constrained_layout=True)
        _heatmap(
            axes[0],
            grid=cov_grid,
            x_labels=[str(x) for x in trains],
            y_labels=[str(y) for y in leafs],
            title=f"{regime}/{lane} | completion fraction | q={q_primary[0]:g}, cal={cal_primary:g}",
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
        )
        cb0 = fig.colorbar(axes[0].images[0], ax=[axes[0]], fraction=0.046, pad=0.02)
        cb0.set_label("fraction complete")

        _heatmap(
            axes[1],
            grid=root_grid,
            x_labels=[str(x) for x in trains],
            y_labels=[str(y) for y in leafs],
            title=f"{regime}/{lane} | root_l1 (median) | q={q_primary[0]:g}, cal={cal_primary:g}",
            vmin=rvmin,
            vmax=rvmax,
            cmap=GOOD_BAD_CMAP,
        )
        cb1 = fig.colorbar(axes[1].images[0], ax=[axes[1]], fraction=0.046, pad=0.02)
        cb1.set_label("root_l1 (green=better, red=worse)")
        p_heat = pages / f"{regime}_{lane}_completion_and_root.png"
        _save_fig(fig, p_heat)
        page_paths.append(p_heat)

        # Learnability curve by leaf_tokens at primary q/cal.
        fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.5), constrained_layout=True)
        for lf in leafs:
            ys = []
            for t in trains:
                vals = [r.root_l1 for r in prim if r.train == t and r.leaf_tokens == lf]
                ys.append(_median(vals))
            finite = [(t, y) for t, y in zip(trains, ys) if math.isfinite(float(y))]
            if not finite:
                continue
            x_f = [x for x, _ in finite]
            y_f = [y for _, y in finite]
            ax.plot(x_f, y_f, marker="o", linewidth=2.0, label=f"leaf={lf}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("n_books_train (log2)")
        ax.set_ylabel("root_l1_mean (median)")
        ax.set_title(f"{regime}/{lane} | learnability slice | q={q_primary[0]:g}, cal={cal_primary:g}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best", fontsize=10)
        p_line = pages / f"{regime}_{lane}_learnability_line.png"
        _save_fig(fig, p_line)
        page_paths.append(p_line)

        lane_diag[f"{regime}/{lane}"] = {
            "n_rows": int(len(sub)),
            "n_seeds_observed": int(n_seeds),
            "q_pairs": [[float(a), float(b)] for a, b in q_pairs],
            "cal_rates": [float(c) for c in cal_vals],
            "trains": [int(t) for t in trains],
            "leaf_tokens": [int(lf) for lf in leafs],
            "primary_slice": {"q": [float(q_primary[0]), float(q_primary[1])], "cal": float(cal_primary)},
        }

    diagnostics["lanes"] = lane_diag
    diag_path = out_dir / "publication_ctreepo_progress_diagnostics.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_path = out_dir / "publication_ctreepo_progress_latest.md"
    pdf_path = out_dir / "publication_ctreepo_progress_latest.pdf"

    def _img(p: Path) -> str:
        return f"![]({str(p.relative_to(out_dir))}){{ width=100% }}"

    lines: List[str] = [
        "---",
        "title: Publication C-TreePO Progress (Interim)",
        f"date: {now.strftime('%Y-%m-%d')}",
        "fontsize: 12pt",
        "geometry: margin=0.7in",
        "---",
        "",
        f"**Output root:** `{output_root}`  ",
        f"**Generated (UTC):** `{now.isoformat()}`",
        "",
        "- Color convention for error plots: **green = better (lower error), red = worse (higher error)**.",
        "- This is an interim snapshot from partial results; missing cells indicate jobs not finished yet.",
        "",
        f"**Diagnostics JSON:** `{diag_path}`",
        "",
        "\\newpage",
    ]
    for p in page_paths:
        lines.extend([f"## {p.name}", "", _img(p), "", "\\newpage"])

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if bool(args.emit_pdf):
        _run_pandoc(md_path, pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

