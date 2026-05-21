#!/usr/bin/env python3
"""Report/visualize overnight neural-operator sweeps for the identifiable-zero suite.

This is intentionally separate from `report_identifiable_zero_suite_journal_appendix.py`:
- It targets *operator ablations* (capacity/info-density) rather than the baseline suite slice.
- It expects an output root created by `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-neural-operator ...`.
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

PLOT_FLOOR = 1e-12
ERROR_AXIS_TOP = 10.0
CEILING_THRESHOLD = 1e-8


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize identifiable-zero neural-operator overnight sweeps.")
    p.add_argument("--overnight-output-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
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


def _float_or_nan(x: object) -> float:
    v = _as_float(x)
    return float(v) if v is not None else float("nan")


def _median(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return float("nan")
    return float(statistics.median(xs))


def _plot_floor(v: object) -> float:
    x = _as_float(v)
    if x is None:
        return float("nan")
    return float(max(PLOT_FLOOR, float(x)))


def _setup_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    plt.rcParams.update(
        {
            "font.size": 13.0,
            "axes.titlesize": 15.0,
            "axes.labelsize": 13.0,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
        }
    )


def _run_pandoc(md_path: Path, pdf_path: Path) -> bool:
    if shutil.which("pandoc") is None or shutil.which("pdflatex") is None:
        return False
    subprocess.run(
        ["pandoc", str(md_path.name), "-o", str(pdf_path.name), "--pdf-engine=pdflatex"],
        cwd=str(md_path.parent),
        check=True,
    )
    return True


@dataclass(frozen=True)
class MarkovGuidedRow:
    q_train: float
    llw: float
    scw: float
    gov: str
    feature_mode: str
    state_dim: int
    hidden_dim: int
    seed: int
    q_infer: float
    root_mae: float
    merge_mae: float
    merge_violation_rate: float
    effective_q_mean: float
    guided_internal_nodes_mean: float


@dataclass(frozen=True)
class MarkovLearnedRow:
    q_train: float
    llw: float
    scw: float
    gov: str
    feature_mode: str
    state_dim: int
    hidden_dim: int
    seed: int
    root_mae: float
    merge_mae: float
    merge_violation_rate: float
    schedule_spread_mean: float
    schedule_spread_p95: float


def _scan_markov_rows(root: Path) -> Tuple[List[MarkovLearnedRow], List[MarkovGuidedRow]]:
    files = sorted(glob.glob(str(root / "markov_changepoint_ops_count" / "**" / "*.json"), recursive=True))
    learned: List[MarkovLearnedRow] = []
    guided: List[MarkovGuidedRow] = []
    for fp in files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        metrics = payload.get("metrics", {}) or {}

        if str(cfg.get("model_family", "")).strip().lower() != "neural":
            continue

        q_train = _as_float(cfg.get("audit_fraction"))
        if q_train is None:
            continue
        objective = payload.get("objective", {}) or {}
        llw = float(
            _as_float(
                objective.get(
                    "local_law_weight",
                    cfg.get(
                        "local_law_weight",
                        float(cfg.get("leaf_weight", 0.0)) + float(cfg.get("c3_weight", 0.0)),
                    ),
                )
            )
            or 0.0
        )
        scw = float(_as_float(cfg.get("schedule_consistency_weight")) or 0.0)
        gov = str(cfg.get("guidance_override_mode", "reset")).strip().lower() or "reset"
        feature_mode = str(cfg.get("feature_mode", "full")).strip() or "full"
        state_dim = int(cfg.get("state_dim", 0) or 0)
        hidden_dim = int(cfg.get("hidden_dim", 0) or 0)
        seed = int(cfg.get("seed", -1) or -1)
        if state_dim <= 0 or hidden_dim <= 0 or seed < 0:
            continue

        learned_metrics = metrics.get("learned") or {}
        learned.append(
            MarkovLearnedRow(
                q_train=float(q_train),
                llw=float(llw),
                scw=float(scw),
                gov=str(gov),
                feature_mode=str(feature_mode),
                state_dim=int(state_dim),
                hidden_dim=int(hidden_dim),
                seed=int(seed),
                root_mae=_float_or_nan(learned_metrics.get("root_mae")),
                merge_mae=_float_or_nan(learned_metrics.get("merge_mae")),
                merge_violation_rate=_float_or_nan(learned_metrics.get("merge_violation_rate")),
                schedule_spread_mean=_float_or_nan(learned_metrics.get("schedule_spread_mean")),
                schedule_spread_p95=_float_or_nan(learned_metrics.get("schedule_spread_p95")),
            )
        )

        pts = ((metrics.get("guided_eval_curve") or {}).get("points") or [])
        for pt in pts:
            if not isinstance(pt, dict):
                continue
            q = _as_float(pt.get("q"))
            if q is None:
                continue
            guided.append(
                MarkovGuidedRow(
                    q_train=float(q_train),
                    llw=float(llw),
                    scw=float(scw),
                    gov=str(gov),
                    feature_mode=str(feature_mode),
                    state_dim=int(state_dim),
                    hidden_dim=int(hidden_dim),
                    seed=int(seed),
                    q_infer=float(q),
                    root_mae=_float_or_nan(pt.get("root_mae")),
                    merge_mae=_float_or_nan(pt.get("merge_mae")),
                    merge_violation_rate=_float_or_nan(pt.get("merge_violation_rate")),
                    effective_q_mean=_float_or_nan(pt.get("effective_q_mean")),
                    guided_internal_nodes_mean=_float_or_nan(pt.get("guided_internal_nodes_mean")),
                )
            )
    return learned, guided


def _plot_markov_capacity_pages(*, guided: Sequence[MarkovGuidedRow], out_dir: Path) -> List[Path]:
    out_paths: List[Path] = []
    q_train_vals = sorted({float(r.q_train) for r in guided})
    llw_vals = sorted({float(r.llw) for r in guided})
    scw_vals = sorted({float(r.scw) for r in guided})
    gov_vals = sorted({str(r.gov) for r in guided})

    feature_modes_all = sorted({str(r.feature_mode) for r in guided})
    colors = {8: "#1f77b4", 16: "#ff7f0e", 32: "#2ca02c", 64: "#d62728", 128: "#9467bd"}

    for qtr in q_train_vals:
        for llw in llw_vals:
            for scw in scw_vals:
                for gov in gov_vals:
                    sub = [
                        r
                        for r in guided
                        if abs(float(r.q_train) - qtr) <= 1e-12
                        and abs(float(r.llw) - llw) <= 1e-12
                        and abs(float(r.scw) - scw) <= 1e-12
                        and str(r.gov) == gov
                    ]
                    if not sub:
                        continue
                    feature_modes = [fm for fm in feature_modes_all if any(str(r.feature_mode) == fm for r in sub)]
                    if not feature_modes:
                        continue
                    state_dims = sorted({int(r.state_dim) for r in sub})
                    q_infer_vals = sorted({float(r.q_infer) for r in sub})

                    ncols = int(len(feature_modes))
                    fig, axes = plt.subplots(2, ncols, figsize=(6.9 * ncols, 8.8), constrained_layout=True)
                    if ncols == 1:
                        axes = np.asarray(axes).reshape(2, 1)

                    for col, fm in enumerate(feature_modes):
                        ax_root = axes[0, col]
                        ax_merge = axes[1, col]
                        fm_rows = [r for r in sub if str(r.feature_mode) == fm]
                        if not fm_rows:
                            continue
                        for sd in state_dims:
                            rows_sd = [r for r in fm_rows if int(r.state_dim) == int(sd)]
                            if not rows_sd:
                                continue
                            xs = q_infer_vals
                            root = []
                            merge = []
                            for q in xs:
                                bucket = [r for r in rows_sd if abs(float(r.q_infer) - float(q)) <= 1e-12]
                                root.append(_plot_floor(_median(r.root_mae for r in bucket)))
                                merge.append(_plot_floor(_median(r.merge_mae for r in bucket)))
                            colr = colors.get(int(sd), None)
                            ax_root.plot(xs, root, marker="o", linewidth=2.0, label=f"state_dim={sd}", color=colr)
                            ax_merge.plot(xs, merge, marker="o", linewidth=2.0, label=f"state_dim={sd}", color=colr)

                        for axx, ylabel in [(ax_root, "root MAE (log)"), (ax_merge, "merge MAE (log)")]:
                            axx.set_yscale("log")
                            axx.set_ylim(PLOT_FLOOR, ERROR_AXIS_TOP)
                            axx.axhline(CEILING_THRESHOLD, color="#666666", linestyle="--", linewidth=1.0)
                            axx.set_xlim(-0.02, 1.02)
                            axx.set_xlabel("decision-time oracle visibility (q_infer)")
                            axx.set_ylabel(ylabel)
                            axx.grid(True, which="major", alpha=0.4)

                        ax_root.set_title(f"Root error vs q_infer (feature_mode={fm})")
                        ax_merge.set_title(f"Merge error vs q_infer (feature_mode={fm})")

                    title = f"Markov neural capacity sweep | q_train={qtr} | llw={llw} | scw={scw} | gov={gov}"
                    fig.suptitle(title, fontsize=14.5)
                    axes[0, -1].legend(frameon=False, loc="upper right")

                    out_png = out_dir / f"M_markov_capacity_qtr_{qtr:.3g}_llw_{llw:.3g}_scw_{scw:.3g}_gov_{gov}.png"
                    out_pdf = out_dir / f"M_markov_capacity_qtr_{qtr:.3g}_llw_{llw:.3g}_scw_{scw:.3g}_gov_{gov}.pdf"
                    fig.savefig(out_png, dpi=240)
                    fig.savefig(out_pdf)
                    plt.close(fig)
                    out_paths.append(out_png)
    return out_paths


@dataclass(frozen=True)
class CTreePhiRow:
    estimator: str
    phi_docs: int
    seed_topics: int
    n_topics: int
    seed: int
    q_leaf: float
    q_internal: float
    phi_l2_mean: float
    root_l1: float


def _scan_ctree_phi_rows(root: Path) -> List[CTreePhiRow]:
    files = sorted(glob.glob(str(root / "segmented_lda_ctreepo" / "**" / "*.json"), recursive=True))
    rows: List[CTreePhiRow] = []
    for fp in files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        m = payload.get("metrics", {}) or {}

        est = str(cfg.get("topic_phi_estimator", "")).strip()
        if not est:
            continue
        phi_docs = int(cfg.get("topic_phi_docs", 0) or 0)
        n_topics = int(cfg.get("n_topics", 0) or 0)
        seed = int(cfg.get("seed", -1) or -1)
        q_leaf = float(_as_float(cfg.get("eval_leaf_query_rate")) or 0.0)
        q_internal = float(_as_float(cfg.get("eval_internal_query_rate")) or 0.0)
        root_l1 = _as_float(((m.get("estimated_calibrated_budgeted") or {}).get("root_l1_mean")))
        if root_l1 is None or seed < 0:
            continue
        phi_l2 = _as_float(((payload.get("topic_meta") or {}).get("topic_phi_l2_error_mean")))

        seed_topics = -1
        if str(est).strip().lower().startswith("neural_"):
            topic_meta = payload.get("topic_meta") or {}
            try:
                seed_topics = int(topic_meta.get("topic_phi_neural_seed_count", -1) or -1)
            except Exception:
                seed_topics = -1

        rows.append(
            CTreePhiRow(
                estimator=est,
                phi_docs=int(phi_docs),
                seed_topics=int(seed_topics),
                n_topics=int(n_topics),
                seed=int(seed),
                q_leaf=float(q_leaf),
                q_internal=float(q_internal),
                phi_l2_mean=float(phi_l2) if phi_l2 is not None else float("nan"),
                root_l1=float(root_l1),
            )
        )
    return rows


def _plot_ctree_phi_density(*, rows: Sequence[CTreePhiRow], out_dir: Path) -> Path:
    rows0 = [r for r in rows if abs(float(r.q_leaf)) <= 1e-12 and abs(float(r.q_internal)) <= 1e-12]
    docs_vals = sorted({int(r.phi_docs) for r in rows0 if int(r.phi_docs) > 0})
    if len(docs_vals) < 2:
        raise ValueError("ctree phi-density plot requires >=2 distinct topic_phi_docs values at q_infer=0")

    rows = rows0
    n_topics_max = max((int(r.n_topics) for r in rows if int(r.n_topics) > 0), default=0)
    if n_topics_max <= 0:
        n_topics_max = 4

    # Aggregate median across seeds.
    groups: Dict[Tuple[str, int, int], Dict[str, List[float]]] = {}
    for r in rows:
        key = (str(r.estimator), int(r.phi_docs), int(r.seed_topics) if int(r.seed_topics) >= 0 else -1)
        groups.setdefault(key, {"phi": [], "root": []})
        groups[key]["phi"].append(float(r.phi_l2_mean))
        groups[key]["root"].append(float(r.root_l1))

    # Build series: estimator -> list of (docs, median_phi, median_root).
    series: Dict[str, List[Tuple[int, float, float, float]]] = {}
    for (est, docs, seed_topics), vals in groups.items():
        phi_med = _median(vals["phi"])
        root_med = _median(vals["root"])
        series.setdefault(str(est), []).append((int(docs), float(phi_med), float(root_med), float(seed_topics)))

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 4.6), constrained_layout=True)
    ax_phi, ax_root = axes

    def _label(est: str, seed_topics: float) -> str:
        if est != "neural_ctreepo":
            return est
        if not math.isfinite(float(seed_topics)) or float(seed_topics) < 0:
            return "neural_ctreepo"
        return f"neural_ctreepo (seed_topics={int(seed_topics)}/{int(n_topics_max)})"

    palette = {
        "spectral_numpy": "#2ca02c",
        "tensor_lda": "#ff7f0e",
        "online_tensor_lda": "#1f77b4",
        "neural_ctreepo": "#7a1fa2",
    }

    for est, pts in sorted(series.items()):
        # For neural_ctreepo, split by number of seed topics.
        if est == "neural_ctreepo":
            fracs = sorted({float(p[3]) for p in pts if p[3] >= 0})
            for frac in fracs:
                sub = [p for p in pts if abs(float(p[3]) - float(frac)) <= 1e-12]
                sub.sort(key=lambda x: int(x[0]))
                xs = [int(p[0]) for p in sub]
                ys_phi = [float(p[1]) for p in sub]
                ys_root = [_plot_floor(float(p[2])) for p in sub]
                ax_phi.plot(xs, ys_phi, marker="o", linewidth=2.0, color=palette.get(est, "#444444"), alpha=0.85, label=_label(est, frac))
                ax_root.plot(xs, ys_root, marker="o", linewidth=2.0, color=palette.get(est, "#444444"), alpha=0.85, label=_label(est, frac))
            continue

        pts.sort(key=lambda x: int(x[0]))
        xs = [int(p[0]) for p in pts]
        ys_phi = [float(p[1]) for p in pts]
        ys_root = [_plot_floor(float(p[2])) for p in pts]
        ax_phi.plot(xs, ys_phi, marker="o", linewidth=2.2, color=palette.get(est, "#444444"), label=_label(est, float("nan")))
        ax_root.plot(xs, ys_root, marker="o", linewidth=2.2, color=palette.get(est, "#444444"), label=_label(est, float("nan")))

    for axx in (ax_phi, ax_root):
        axx.set_xscale("log")
        axx.set_xlabel("topic_phi_docs (log)")
        axx.set_xlim(32, None)
        axx.grid(True, which="major", alpha=0.4)

    ax_phi.set_title("Upstream topic-phi error vs docs")
    ax_phi.set_ylabel("topic phi L2 error (mean, aligned)")

    ax_root.set_title("Downstream root error vs docs (q_infer=0)")
    ax_root.set_yscale("log")
    ax_root.set_ylim(PLOT_FLOOR, ERROR_AXIS_TOP)
    ax_root.axhline(CEILING_THRESHOLD, color="#666666", linestyle="--", linewidth=1.0)
    ax_root.set_ylabel("root L1 (log)")

    ax_root.legend(frameon=False, loc="upper right", fontsize=9.5)

    out_png = out_dir / "C_ctree_phi_density.png"
    out_pdf = out_dir / "C_ctree_phi_density.pdf"
    fig.savefig(out_png, dpi=240)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png


def _plot_ctree_budget_sweep(*, rows: Sequence[CTreePhiRow], out_dir: Path) -> List[Path]:
    n_topics_max = max((int(r.n_topics) for r in rows if int(r.n_topics) > 0), default=0)
    if n_topics_max <= 0:
        n_topics_max = 4

    qs = sorted({float(r.q_leaf) for r in rows})
    if len(qs) < 2:
        raise ValueError("ctree budget plot requires >=2 distinct q_infer values")

    phi_docs_vals = sorted({int(r.phi_docs) for r in rows if int(r.phi_docs) > 0})
    if not phi_docs_vals:
        phi_docs_vals = [0]

    palette = {
        "spectral_numpy": "#2ca02c",
        "tensor_lda": "#ff7f0e",
        "online_tensor_lda": "#1f77b4",
        "neural_ctreepo": "#7a1fa2",
        "neural_mergeable_sketch": "#d62728",
        "neural_hybrid": "#8c564b",
        "neural_embedding_hybrid": "#9467bd",
    }

    def _label(est: str, seed_topics: int) -> str:
        est_l = str(est).strip()
        if not est_l.lower().startswith("neural_"):
            return est_l
        if seed_topics < 0:
            return est_l
        return f"{est_l} (seed_topics={seed_topics}/{int(n_topics_max)})"

    out_paths: List[Path] = []
    for phi_docs in phi_docs_vals:
        sub = [r for r in rows if int(r.phi_docs) == int(phi_docs)]
        if not sub:
            continue

        # group_key -> q -> list[root_l1]
        buckets: Dict[Tuple[str, int], Dict[float, List[float]]] = {}
        for r in sub:
            est = str(r.estimator)
            seed_topics = int(r.seed_topics) if str(est).strip().lower().startswith("neural_") else -1
            key = (est, seed_topics)
            q = float(r.q_leaf)
            buckets.setdefault(key, {}).setdefault(q, []).append(float(r.root_l1))

        fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.8), constrained_layout=True)
        for (est, seed_topics), by_q in sorted(buckets.items()):
            xs = sorted(by_q.keys())
            ys = [_plot_floor(_median(by_q[q])) for q in xs]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.2,
                color=palette.get(str(est), "#444444"),
                alpha=0.9,
                label=_label(str(est), int(seed_topics)),
            )

        ax.set_title(f"C-TreePO end-to-end error vs q_infer (topic_phi_docs={int(phi_docs)})")
        ax.set_xlabel("decision-time oracle visibility (q_infer)")
        ax.set_ylabel("root L1 (log)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_yscale("log")
        ax.set_ylim(PLOT_FLOOR, ERROR_AXIS_TOP)
        ax.axhline(CEILING_THRESHOLD, color="#666666", linestyle="--", linewidth=1.0)
        ax.grid(True, which="major", alpha=0.4)
        ax.legend(frameon=False, loc="upper right", fontsize=9.5)

        suffix = f"{int(phi_docs)}" if int(phi_docs) > 0 else "auto"
        out_png = out_dir / f"D_ctree_budget_sweep_docs_{suffix}.png"
        out_pdf = out_dir / f"D_ctree_budget_sweep_docs_{suffix}.pdf"
        fig.savefig(out_png, dpi=240)
        fig.savefig(out_pdf)
        plt.close(fig)
        out_paths.append(out_png)

    return out_paths


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_style()

    overnight_root = Path(args.overnight_output_root)
    out_dir = Path(args.out_dir) if args.out_dir is not None else (overnight_root / "figures" / "neural_operator_overnight")
    out_dir.mkdir(parents=True, exist_ok=True)

    learned, guided = _scan_markov_rows(overnight_root)
    ctree_rows = _scan_ctree_phi_rows(overnight_root)

    fig_paths: List[Path] = []
    if guided:
        fig_paths.extend(_plot_markov_capacity_pages(guided=guided, out_dir=out_dir))
    if ctree_rows:
        qs = sorted({float(r.q_leaf) for r in ctree_rows})
        if len(qs) >= 2:
            try:
                fig_paths.extend(_plot_ctree_budget_sweep(rows=ctree_rows, out_dir=out_dir))
            except Exception:
                pass
        try:
            fig_paths.append(_plot_ctree_phi_density(rows=ctree_rows, out_dir=out_dir))
        except Exception:
            pass

    md_path = out_dir / "identifiable_zero_neural_operator_overnight_latest.md"
    pdf_path = out_dir / "identifiable_zero_neural_operator_overnight_latest.pdf"
    lines: List[str] = []
    lines.append("---")
    lines.append("title: Identifiable-Zero Neural-Operator Overnight Sweep (Auto-Report)")
    lines.append("geometry: margin=0.7in")
    lines.append("fontsize: 12pt")
    lines.append("toc: true")
    lines.append("toc-depth: 2")
    lines.append("---")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Overnight output root: `{overnight_root}`")
    lines.append("")
    lines.append("## 1. Markov capacity / information-density")
    lines.append("")
    lines.append("Pages below facet by `q_train`, `local_law_weight`, `schedule_consistency_weight`, and `guidance_override_mode`.")
    lines.append("")
    for p in sorted([p for p in fig_paths if p.name.startswith("M_markov_capacity_")]):
        lines.append(f"![]({p.name}){{width=100%}}")
        lines.append("")
        lines.append("\\newpage")
        lines.append("")
    lines.append("## 2. C-TreePO topic-phi information density")
    lines.append("")
    for p in sorted([p for p in fig_paths if p.name.startswith("D_ctree_budget_sweep")]):
        lines.append(f"![]({p.name}){{width=100%}}")
        lines.append("")
        lines.append("\\newpage")
        lines.append("")
    for p in sorted([p for p in fig_paths if p.name.startswith("C_ctree_phi_density")]):
        lines.append(f"![]({p.name}){{width=100%}}")
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote_markdown | {md_path}")

    pdf_ok = False
    if bool(args.emit_pdf):
        try:
            pdf_ok = _run_pandoc(md_path, pdf_path)
        except Exception as e:
            print(f"pandoc_failed | {e}")
            pdf_ok = False
    print(f"wrote_pdf={bool(pdf_ok)} | {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
