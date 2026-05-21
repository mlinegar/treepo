#!/usr/bin/env python3
"""Appendix-quality learnability report for the Identifiable-Zero learnability suite.

This report is explicitly about "learnability" (performance improving with more data),
not budget-frontier aesthetics.

Inputs: an output root produced by `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-learnability ...`.
Outputs:
  - <out-dir>/identifiable_zero_learnability_latest.md
  - <out-dir>/identifiable_zero_learnability_latest.pdf (if pandoc+pdflatex are available)
  - <out-dir>/identifiable_zero_learnability_latest_diagnostics.json
  - <out-dir>/pages/*.png (one page/figure each)
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
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from treepo._research.ctreepo.sim.manifest import read_manifest_jsonl
from treepo._research.ctreepo.sim.suite.common import read_suite_meta, resolve_grouped_suite_paths
from treepo._research.ctreepo.sim.suite.learnability_policy import (  # type: ignore[import-not-found]
    IdentifiableZeroLearnabilityPolicy,
)

PLOT_EPS = 1e-12


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate an appendix-style learnability report (Identifiable-Zero v1).")
    p.add_argument("--output-root", type=Path, required=True, help="Learnability sweep output root.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <output-root>/figures/learnability).",
    )
    p.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(list(argv) if argv is not None else None)


def _load_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _suite_summary_paths(output_root: Path) -> tuple[dict[str, List[Path]], Dict[str, object]]:
    family_paths: dict[str, List[Path]] = {"markov": [], "ctree": []}
    paths = resolve_grouped_suite_paths(output_root.resolve())
    if not paths.suite_meta.exists():
        return family_paths, {}
    meta = read_suite_meta(paths.suite_meta)
    selected_groups = [str(x) for x in (meta.get("selected_groups", []) or []) if str(x).strip()]
    group_manifest_files = dict(meta.get("group_manifest_files", {}) or {})
    manifest_paths: List[Path] = []
    for group in selected_groups:
        manifest_path = Path(str(group_manifest_files.get(group, "")))
        if manifest_path.exists():
            manifest_paths.append(manifest_path)
    if not manifest_paths and paths.suite_manifest.exists():
        manifest_paths = [paths.suite_manifest]

    for manifest_path in manifest_paths:
        for run in read_manifest_jsonl(manifest_path):
            out_path = Path(str(run.outputs.get("json_summary", "")))
            if not out_path.exists():
                continue
            family = str(run.family).strip().lower()
            if family == "markov-ops-count":
                family_paths["markov"].append(out_path.resolve())
            elif family == "segmented-lda-ctreepo":
                family_paths["ctree"].append(out_path.resolve())
    family_paths = {key: sorted(set(value)) for key, value in family_paths.items()}
    return family_paths, meta


def _subset_mode(meta: Dict[str, object], *, markov_rows: Sequence[object], ctree_rows: Sequence[object]) -> str:
    selected_groups = [str(x) for x in (meta.get("selected_groups", []) or []) if str(x).strip()]
    group_families = dict(meta.get("group_families", {}) or {})
    selected_families = {str(group_families.get(group, "")).strip().lower() for group in selected_groups}
    selected_families.discard("")
    if selected_families:
        has_markov = "markov-ops-count" in selected_families
        has_ctree = "segmented-lda-ctreepo" in selected_families
        if has_markov and has_ctree:
            return "cross_family"
        if has_markov:
            return "markov_only"
        if has_ctree:
            return "ctree_only"
    if markov_rows and ctree_rows:
        return "cross_family"
    if markov_rows:
        return "markov_only"
    if ctree_rows:
        return "ctree_only"
    return "cross_family"


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


def _try_rel_regime(output_root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(output_root)
    except Exception:
        rel = path
    parts = [str(p) for p in rel.parts]
    if "equivalence" in parts:
        idx = parts.index("equivalence")
        if idx + 1 < len(parts):
            cand = parts[idx + 1]
            if cand in {"baseline", "hard", "lda"}:
                return cand
    # Fallback: search anywhere.
    if "baseline" in parts:
        return "baseline"
    if "hard" in parts:
        return "hard"
    if "lda" in parts:
        return "lda"
    return "unknown"


def _heatmap(
    ax: plt.Axes,
    *,
    grid: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
) -> plt.Axes:
    im = ax.imshow(grid, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(list(range(len(x_labels))), labels=list(x_labels))
    ax.set_yticks(list(range(len(y_labels))), labels=list(y_labels))
    ax.set_xlabel("train set size")
    ax.set_ylabel("oracle label rate")
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)

    ny, nx = grid.shape
    for yi in range(ny):
        for xi in range(nx):
            val = float(grid[yi, xi])
            ax.text(
                xi,
                yi,
                _fmt_cell(val),
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.65,
                },
            )
    # Colorbar is handled at the figure level so multi-panel pages share a scale.
    return ax


def _build_grid(
    rows: Iterable[Tuple[int, float, float]],
    *,
    x_vals: Sequence[int],
    y_vals: Sequence[float],
) -> np.ndarray:
    # rows: (x, y, value)
    cell: Dict[Tuple[int, float], List[float]] = {}
    for x, y, v in rows:
        cell.setdefault((int(x), float(y)), []).append(float(v))
    grid = np.full((len(y_vals), len(x_vals)), float("nan"), dtype=np.float64)
    for yi, y in enumerate(y_vals):
        for xi, x in enumerate(x_vals):
            grid[yi, xi] = float(_median(cell.get((int(x), float(y)), [])))
    return grid


def _monotonicity_violations(
    grid: np.ndarray,
    *,
    axis: int,
    tol: float = 0.0,
) -> Dict[str, object]:
    # axis=1: x increases; axis=0: y increases.
    n_viol = 0
    n_pairs = 0
    if axis == 1:
        for yi in range(grid.shape[0]):
            for xi in range(grid.shape[1] - 1):
                a = float(grid[yi, xi])
                b = float(grid[yi, xi + 1])
                if not (math.isfinite(a) and math.isfinite(b)):
                    continue
                n_pairs += 1
                if b > a + float(tol):
                    n_viol += 1
    elif axis == 0:
        for xi in range(grid.shape[1]):
            for yi in range(grid.shape[0] - 1):
                a = float(grid[yi, xi])
                b = float(grid[yi + 1, xi])
                if not (math.isfinite(a) and math.isfinite(b)):
                    continue
                n_pairs += 1
                if b > a + float(tol):
                    n_viol += 1
    else:
        raise ValueError("axis must be 0 or 1")
    return {
        "n_pairs": int(n_pairs),
        "n_violations": int(n_viol),
        "violation_rate": (float(n_viol) / float(n_pairs) if n_pairs else float("nan")),
    }


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
class _MarkovRow:
    regime: str
    train_docs: int
    test_docs: int
    audit_fraction: float
    seed: int
    model_family: str
    learned_root_mae: float
    leaf_bucket_root_mae: Optional[float]
    leaf_ridge_tree_root_mae: Optional[float]
    leaf_endpoint_table_tree_root_mae: Optional[float]
    leaf_dt_tree_root_mae: Optional[float]
    leaf_knn_tree_root_mae: Optional[float]
    leaf_rf_tree_root_mae: Optional[float]
    doc_level_root_mae: Optional[float]
    doc_level_ridge_root_mae: Optional[float]
    rf_root_mae: Optional[float]


@dataclass(frozen=True)
class _MarkovSampledLeafRow:
    regime: str
    train_docs: int
    test_docs: int
    audit_fraction: float
    seed: int
    leaf_budget: int
    baseline_kind: str
    root_mae: float
    sampled_leaves_mean_test: float
    sampled_tokens_mean_test: float
    sampled_token_fraction_mean_test: float


@dataclass(frozen=True)
class _CTreeRow:
    regime: str
    n_books_train: int
    n_books_test: int
    calibration_rate: float
    eval_leaf_rate: float
    eval_internal_rate: float
    seed: int
    topic_phi_estimator: str
    leaf_theta_estimator: str
    root_l1_mean: float
    leaf_theta_l1_mean: Optional[float]
    topic_phi_l2_error_mean: Optional[float]
    topic_component_mean: Optional[float]
    oracle_proxy_component_mean: Optional[float]
    oracle_proxy_root_l1_mean: Optional[float]
    full_doc_rf_root_l1_mean: Optional[float]
    full_doc_mlp_root_l1_mean: Optional[float]
    corpus_signature_test: Optional[str]
    dgp_key: Tuple[object, ...]


def _scan_markov(output_root: Path, candidate_paths: Optional[Sequence[Path]] = None) -> List[_MarkovRow]:
    rows: List[_MarkovRow] = []
    paths = (
        [Path(path) for path in candidate_paths]
        if candidate_paths is not None
        else [
            Path(fp)
            for fp in glob.glob(
                str(output_root / "markov_changepoint_ops_count" / "equivalence" / "**" / "*.json"),
                recursive=True,
            )
        ]
    )
    for path in paths:
        payload = _load_json(path)
        if not payload:
            continue
        cfg = payload.get("config") or {}
        met = payload.get("metrics") or {}
        learned = (met.get("learned") or {}) if isinstance(met, dict) else {}
        if not isinstance(cfg, dict) or not isinstance(met, dict) or not isinstance(learned, dict):
            continue

        train_docs = int(cfg.get("train_docs", -1))
        audit_fraction = _float_or_nan(cfg.get("audit_fraction"))
        seed = int(cfg.get("seed", -1))
        fam = str(cfg.get("model_family", ""))
        learned_root_mae = _float_or_nan(learned.get("root_mae"))
        leaf_bucket = met.get("leaf_bucket")
        leaf_bucket_root_mae: Optional[float] = None
        if isinstance(leaf_bucket, dict):
            v = _as_float(leaf_bucket.get("root_mae"))
            leaf_bucket_root_mae = float(v) if v is not None else None
        leaf_ridge_tree = met.get("leaf_ridge_tree")
        leaf_ridge_tree_root_mae: Optional[float] = None
        if isinstance(leaf_ridge_tree, dict):
            v = _as_float(leaf_ridge_tree.get("root_mae"))
            leaf_ridge_tree_root_mae = float(v) if v is not None else None
        leaf_endpoint_table_tree = met.get("leaf_endpoint_table_tree")
        leaf_endpoint_table_tree_root_mae: Optional[float] = None
        if isinstance(leaf_endpoint_table_tree, dict):
            v = _as_float(leaf_endpoint_table_tree.get("root_mae"))
            leaf_endpoint_table_tree_root_mae = float(v) if v is not None else None
        leaf_dt_tree = met.get("leaf_dt_tree")
        leaf_dt_tree_root_mae: Optional[float] = None
        if isinstance(leaf_dt_tree, dict):
            v = _as_float(leaf_dt_tree.get("root_mae"))
            leaf_dt_tree_root_mae = float(v) if v is not None else None
        leaf_knn_tree = met.get("leaf_knn_tree")
        leaf_knn_tree_root_mae: Optional[float] = None
        if isinstance(leaf_knn_tree, dict):
            v = _as_float(leaf_knn_tree.get("root_mae"))
            leaf_knn_tree_root_mae = float(v) if v is not None else None
        leaf_rf_tree = met.get("leaf_rf_tree")
        leaf_rf_tree_root_mae: Optional[float] = None
        if isinstance(leaf_rf_tree, dict):
            v = _as_float(leaf_rf_tree.get("root_mae"))
            leaf_rf_tree_root_mae = float(v) if v is not None else None
        doc_level = met.get("doc_level")
        doc_level_root_mae: Optional[float] = None
        if isinstance(doc_level, dict):
            v = _as_float(doc_level.get("root_mae"))
            doc_level_root_mae = float(v) if v is not None else None
        doc_level_ridge = met.get("doc_level_ridge")
        doc_level_ridge_root_mae: Optional[float] = None
        if isinstance(doc_level_ridge, dict):
            v = _as_float(doc_level_ridge.get("root_mae"))
            doc_level_ridge_root_mae = float(v) if v is not None else None
        rf_root = met.get("rf_root")
        rf_root_mae: Optional[float] = None
        if isinstance(rf_root, dict):
            v = _as_float(rf_root.get("root_mae"))
            rf_root_mae = float(v) if v is not None else None

        if train_docs < 0 or seed < 0 or not math.isfinite(audit_fraction) or not math.isfinite(learned_root_mae):
            continue
        regime = _try_rel_regime(output_root, path)
        rows.append(
            _MarkovRow(
                regime=regime,
                train_docs=int(train_docs),
                test_docs=int(cfg.get("test_docs", -1)),
                audit_fraction=float(audit_fraction),
                seed=int(seed),
                model_family=str(fam),
                learned_root_mae=float(learned_root_mae),
                leaf_bucket_root_mae=leaf_bucket_root_mae,
                leaf_ridge_tree_root_mae=leaf_ridge_tree_root_mae,
                leaf_endpoint_table_tree_root_mae=leaf_endpoint_table_tree_root_mae,
                leaf_dt_tree_root_mae=leaf_dt_tree_root_mae,
                leaf_knn_tree_root_mae=leaf_knn_tree_root_mae,
                leaf_rf_tree_root_mae=leaf_rf_tree_root_mae,
                doc_level_root_mae=doc_level_root_mae,
                doc_level_ridge_root_mae=doc_level_ridge_root_mae,
                rf_root_mae=rf_root_mae,
            )
        )
    return rows


def _scan_markov_sampled_leaf_pool(
    output_root: Path,
    candidate_paths: Optional[Sequence[Path]] = None,
) -> List[_MarkovSampledLeafRow]:
    rows: List[_MarkovSampledLeafRow] = []
    paths = (
        [Path(path) for path in candidate_paths]
        if candidate_paths is not None
        else [
            Path(fp)
            for fp in glob.glob(
                str(output_root / "markov_changepoint_ops_count" / "equivalence" / "**" / "*.json"),
                recursive=True,
            )
        ]
    )
    for path in paths:
        payload = _load_json(path)
        if not payload:
            continue
        cfg = payload.get("config") or {}
        met = payload.get("metrics") or {}
        if not isinstance(cfg, dict) or not isinstance(met, dict):
            continue
        sweep = met.get("sampled_leaf_pool_budget_sweep")
        if not isinstance(sweep, dict):
            continue
        points = list(sweep.get("points") or [])
        if not points:
            continue

        train_docs = int(cfg.get("train_docs", -1))
        audit_fraction = _float_or_nan(cfg.get("audit_fraction"))
        seed = int(cfg.get("seed", -1))
        regime = _try_rel_regime(output_root, path)
        test_docs = int(cfg.get("test_docs", -1))
        if train_docs < 0 or seed < 0 or not math.isfinite(audit_fraction):
            continue

        for point in points:
            if not isinstance(point, dict):
                continue
            leaf_budget = int(point.get("leaf_budget", 0))
            obs = point.get("test_observation") or {}
            if not isinstance(obs, dict) or leaf_budget <= 0:
                continue
            sampled_leaves_mean_test = _float_or_nan(obs.get("sampled_leaves_mean"))
            sampled_tokens_mean_test = _float_or_nan(obs.get("sampled_tokens_mean"))
            sampled_token_fraction_mean_test = _float_or_nan(
                obs.get("sampled_token_fraction_mean")
            )
            for metric_key, baseline_kind in (
                ("ridge", "sampled_leaf_pool_ridge"),
                ("rf", "sampled_leaf_pool_rf"),
            ):
                metric_payload = point.get(metric_key)
                if not isinstance(metric_payload, dict):
                    continue
                root_mae = _float_or_nan(metric_payload.get("root_mae"))
                if not (
                    math.isfinite(root_mae)
                    and math.isfinite(sampled_leaves_mean_test)
                    and math.isfinite(sampled_tokens_mean_test)
                    and math.isfinite(sampled_token_fraction_mean_test)
                ):
                    continue
                rows.append(
                    _MarkovSampledLeafRow(
                        regime=regime,
                        train_docs=int(train_docs),
                        test_docs=int(test_docs),
                        audit_fraction=float(audit_fraction),
                        seed=int(seed),
                        leaf_budget=int(leaf_budget),
                        baseline_kind=str(baseline_kind),
                        root_mae=float(root_mae),
                        sampled_leaves_mean_test=float(sampled_leaves_mean_test),
                        sampled_tokens_mean_test=float(sampled_tokens_mean_test),
                        sampled_token_fraction_mean_test=float(
                            sampled_token_fraction_mean_test
                        ),
                    )
                )
    return rows


def _scan_ctree(output_root: Path, candidate_paths: Optional[Sequence[Path]] = None) -> List[_CTreeRow]:
    rows: List[_CTreeRow] = []
    paths = (
        [Path(path) for path in candidate_paths]
        if candidate_paths is not None
        else [
            Path(fp)
            for fp in glob.glob(
                str(output_root / "segmented_lda_ctreepo" / "equivalence" / "**" / "*.json"),
                recursive=True,
            )
        ]
    )
    for path in paths:
        payload = _load_json(path)
        if not payload:
            continue
        cfg = payload.get("config") or {}
        met = payload.get("metrics") or {}
        topic_meta = payload.get("topic_meta") or {}
        decomp = payload.get("decomposition") or {}
        if not isinstance(cfg, dict) or not isinstance(met, dict) or not isinstance(topic_meta, dict):
            continue
        if not isinstance(decomp, dict):
            decomp = {}

        policy = (met.get("estimated_calibrated_budgeted") or {}) if isinstance(met, dict) else {}
        if not isinstance(policy, dict):
            continue
        oracle_proxy = (met.get("oracle_proxy") or {}) if isinstance(met, dict) else {}
        if not isinstance(oracle_proxy, dict):
            oracle_proxy = {}
        full_doc_rf = (met.get("full_doc_rf") or {}) if isinstance(met, dict) else {}
        if not isinstance(full_doc_rf, dict):
            full_doc_rf = {}
        full_doc_mlp = (met.get("full_doc_mlp") or {}) if isinstance(met, dict) else {}
        if not isinstance(full_doc_mlp, dict):
            full_doc_mlp = {}

        n_books_train = int(cfg.get("n_books_train", -1))
        cal_rate = _float_or_nan(cfg.get("calibration_leaf_query_rate"))
        eval_leaf_rate = _float_or_nan(cfg.get("eval_leaf_query_rate"))
        eval_internal_rate = _float_or_nan(cfg.get("eval_internal_query_rate"))
        seed = int(cfg.get("seed", -1))
        topic_phi_estimator = str(cfg.get("topic_phi_estimator", ""))
        leaf_theta_estimator = str(cfg.get("leaf_theta_estimator", "lstsq"))
        root_l1_mean = _float_or_nan(policy.get("root_l1_mean"))
        leaf_theta_l1_mean_v = _as_float(topic_meta.get("leaf_theta_l1_mean"))
        topic_phi_l2_error_mean_v = _as_float(topic_meta.get("topic_phi_l2_error_mean"))
        topic_component_mean_v = _as_float(decomp.get("topic_component_mean"))
        oracle_proxy_component_mean_v = _as_float(decomp.get("oracle_proxy_component_mean"))
        oracle_proxy_root_l1_mean_v = _as_float(oracle_proxy.get("root_l1_mean"))
        full_doc_rf_root_l1_mean_v = _as_float(full_doc_rf.get("root_l1_mean"))
        full_doc_mlp_root_l1_mean_v = _as_float(full_doc_mlp.get("root_l1_mean"))
        corpus_sig = topic_meta.get("corpus_signature_test")
        corpus_sig_s: Optional[str] = str(corpus_sig) if corpus_sig is not None else None

        if (
            n_books_train < 0
            or seed < 0
            or not math.isfinite(cal_rate)
            or not math.isfinite(eval_leaf_rate)
            or not math.isfinite(eval_internal_rate)
            or not math.isfinite(root_l1_mean)
        ):
            continue

        dgp_key = (
            cfg.get("n_topics"),
            cfg.get("vocab_size"),
            cfg.get("min_segments"),
            cfg.get("max_segments"),
            cfg.get("min_seg_tokens"),
            cfg.get("max_seg_tokens"),
            cfg.get("fixed_leaf_tokens"),
            cfg.get("alpha_topic"),
            cfg.get("beta_word"),
            cfg.get("segment_concentration"),
            cfg.get("segment_background"),
        )
        regime = _try_rel_regime(output_root, path)
        rows.append(
            _CTreeRow(
                regime=regime,
                n_books_train=int(n_books_train),
                n_books_test=int(cfg.get("n_books_test", -1)),
                calibration_rate=float(cal_rate),
                eval_leaf_rate=float(eval_leaf_rate),
                eval_internal_rate=float(eval_internal_rate),
                seed=int(seed),
                topic_phi_estimator=str(topic_phi_estimator),
                leaf_theta_estimator=str(leaf_theta_estimator),
                root_l1_mean=float(root_l1_mean),
                leaf_theta_l1_mean=float(leaf_theta_l1_mean_v) if leaf_theta_l1_mean_v is not None else None,
                topic_phi_l2_error_mean=float(topic_phi_l2_error_mean_v) if topic_phi_l2_error_mean_v is not None else None,
                topic_component_mean=float(topic_component_mean_v) if topic_component_mean_v is not None else None,
                oracle_proxy_component_mean=float(oracle_proxy_component_mean_v)
                if oracle_proxy_component_mean_v is not None
                else None,
                oracle_proxy_root_l1_mean=float(oracle_proxy_root_l1_mean_v) if oracle_proxy_root_l1_mean_v is not None else None,
                full_doc_rf_root_l1_mean=float(full_doc_rf_root_l1_mean_v)
                if full_doc_rf_root_l1_mean_v is not None
                else None,
                full_doc_mlp_root_l1_mean=float(full_doc_mlp_root_l1_mean_v)
                if full_doc_mlp_root_l1_mean_v is not None
                else None,
                corpus_signature_test=corpus_sig_s,
                dgp_key=dgp_key,
            )
        )
    return rows


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = Path(args.output_root)
    out_dir = Path(args.out_dir) if args.out_dir is not None else (output_root / "figures" / "learnability")
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    now = datetime.now(timezone.utc)
    suite_paths, suite_meta = _suite_summary_paths(output_root)
    markov_rows = _scan_markov(output_root, candidate_paths=suite_paths["markov"] or None)
    markov_sampled_leaf_rows = _scan_markov_sampled_leaf_pool(
        output_root,
        candidate_paths=suite_paths["markov"] or None,
    )
    ctree_rows = _scan_ctree(output_root, candidate_paths=suite_paths["ctree"] or None)
    subset_mode = _subset_mode(suite_meta, markov_rows=markov_rows, ctree_rows=ctree_rows)

    diagnostics: Dict[str, object] = {
        "generated_at_utc": now.isoformat(),
        "output_root": str(output_root),
        "policy_defaults": IdentifiableZeroLearnabilityPolicy().to_dict(),
        "subset_mode": str(subset_mode),
        "suite_meta": {
            "present": bool(suite_meta),
            "selected_groups": list(suite_meta.get("selected_groups", []) or []),
            "group_families": dict(suite_meta.get("group_families", {}) or {}),
        },
        "markov": {
            "n_rows": int(len(markov_rows)),
            "sampled_leaf_pool_rows": int(len(markov_sampled_leaf_rows)),
        },
        "ctree": {"n_rows": int(len(ctree_rows))},
    }

    markov_train_grid = sorted({int(r.train_docs) for r in markov_rows})
    ctree_train_grid = sorted({int(r.n_books_train) for r in ctree_rows})
    markov_label_grid = sorted({float(r.audit_fraction) for r in markov_rows})
    ctree_label_grid = sorted({float(r.calibration_rate) for r in ctree_rows})
    markov_test_grid = sorted({int(r.test_docs) for r in markov_rows if int(r.test_docs) > 0})
    ctree_test_grid = sorted({int(r.n_books_test) for r in ctree_rows if int(r.n_books_test) > 0})
    cross_family_match = bool(
        markov_rows
        and ctree_rows
        and markov_train_grid == ctree_train_grid
        and markov_label_grid == ctree_label_grid
        and markov_test_grid == ctree_test_grid
    )
    diagnostics["setup_alignment"] = {
        "subset_mode": str(subset_mode),
        "applicable": bool(subset_mode == "cross_family"),
        "markov_train_docs": markov_train_grid,
        "ctree_train_docs": ctree_train_grid,
        "train_docs_match": bool(markov_train_grid == ctree_train_grid and markov_train_grid),
        "markov_label_rates": markov_label_grid,
        "ctree_label_rates": ctree_label_grid,
        "label_rates_match": bool(markov_label_grid == ctree_label_grid and markov_label_grid),
        "markov_test_docs": markov_test_grid,
        "ctree_test_docs": ctree_test_grid,
        "heldout_size_match": bool(markov_test_grid == ctree_test_grid and markov_test_grid),
        "both_families_present": bool(markov_rows and ctree_rows),
        "matches": (bool(cross_family_match) if subset_mode == "cross_family" else None),
        "safe_to_quote_cross_family": bool(subset_mode == "cross_family" and cross_family_match),
    }

    # --- Markov pages (baseline + hard) ---
    for regime in ("baseline", "hard"):
        reg_rows = [r for r in markov_rows if r.regime == regime]
        if not reg_rows:
            continue
        x_vals = sorted({int(r.train_docs) for r in reg_rows})
        y_vals = sorted({float(r.audit_fraction) for r in reg_rows})
        x_labels = [str(x) for x in x_vals]
        y_labels = [f"{y:.3g}" for y in y_vals]

        # Aggregate medians across seeds.
        neural_rows = [(r.train_docs, r.audit_fraction, r.learned_root_mae) for r in reg_rows if r.model_family == "neural"]
        leaf_bucket_rows = [
            (r.train_docs, r.audit_fraction, float(r.leaf_bucket_root_mae))
            for r in reg_rows
            if r.leaf_bucket_root_mae is not None
            and math.isfinite(float(r.leaf_bucket_root_mae))
        ]
        leaf_ridge_rows = [
            (r.train_docs, r.audit_fraction, float(r.leaf_ridge_tree_root_mae))
            for r in reg_rows
            if r.leaf_ridge_tree_root_mae is not None
            and math.isfinite(float(r.leaf_ridge_tree_root_mae))
        ]
        leaf_endpoint_table_rows = [
            (r.train_docs, r.audit_fraction, float(r.leaf_endpoint_table_tree_root_mae))
            for r in reg_rows
            if r.leaf_endpoint_table_tree_root_mae is not None
            and math.isfinite(float(r.leaf_endpoint_table_tree_root_mae))
        ]
        leaf_dt_rows = [
            (r.train_docs, r.audit_fraction, float(r.leaf_dt_tree_root_mae))
            for r in reg_rows
            if r.leaf_dt_tree_root_mae is not None
            and math.isfinite(float(r.leaf_dt_tree_root_mae))
        ]
        leaf_knn_rows = [
            (r.train_docs, r.audit_fraction, float(r.leaf_knn_tree_root_mae))
            for r in reg_rows
            if r.leaf_knn_tree_root_mae is not None
            and math.isfinite(float(r.leaf_knn_tree_root_mae))
        ]
        leaf_rf_rows = [
            (r.train_docs, r.audit_fraction, float(r.leaf_rf_tree_root_mae))
            for r in reg_rows
            if r.leaf_rf_tree_root_mae is not None
            and math.isfinite(float(r.leaf_rf_tree_root_mae))
        ]
        doc_neural_rows = [
            (r.train_docs, r.audit_fraction, float(r.doc_level_root_mae))
            for r in reg_rows
            if r.model_family == "neural"
            and r.doc_level_root_mae is not None
            and math.isfinite(float(r.doc_level_root_mae))
        ]
        additive_rows = [
            (r.train_docs, r.audit_fraction, r.learned_root_mae) for r in reg_rows if r.model_family == "additive"
        ]
        doc_add_rows = [
            (r.train_docs, r.audit_fraction, float(r.doc_level_root_mae))
            for r in reg_rows
            if r.model_family == "additive"
            and r.doc_level_root_mae is not None
            and math.isfinite(float(r.doc_level_root_mae))
        ]
        rf_rows = [
            (r.train_docs, r.audit_fraction, float(r.rf_root_mae))
            for r in reg_rows
            if r.rf_root_mae is not None and math.isfinite(float(r.rf_root_mae))
        ]
        ridge_rows = [
            (r.train_docs, r.audit_fraction, float(r.doc_level_ridge_root_mae))
            for r in reg_rows
            if r.doc_level_ridge_root_mae is not None
            and math.isfinite(float(r.doc_level_ridge_root_mae))
        ]
        grid_neural = _build_grid(neural_rows, x_vals=x_vals, y_vals=y_vals)
        grid_leaf_bucket = _build_grid(leaf_bucket_rows, x_vals=x_vals, y_vals=y_vals)
        grid_leaf_ridge = _build_grid(leaf_ridge_rows, x_vals=x_vals, y_vals=y_vals)
        grid_leaf_endpoint_table = _build_grid(
            leaf_endpoint_table_rows,
            x_vals=x_vals,
            y_vals=y_vals,
        )
        grid_leaf_dt = _build_grid(leaf_dt_rows, x_vals=x_vals, y_vals=y_vals)
        grid_leaf_knn = _build_grid(leaf_knn_rows, x_vals=x_vals, y_vals=y_vals)
        grid_leaf_rf = _build_grid(leaf_rf_rows, x_vals=x_vals, y_vals=y_vals)
        grid_doc_neural = _build_grid(doc_neural_rows, x_vals=x_vals, y_vals=y_vals)
        grid_add = _build_grid(additive_rows, x_vals=x_vals, y_vals=y_vals)
        grid_doc_add = _build_grid(doc_add_rows, x_vals=x_vals, y_vals=y_vals)
        grid_rf = _build_grid(rf_rows, x_vals=x_vals, y_vals=y_vals)
        grid_ridge = _build_grid(ridge_rows, x_vals=x_vals, y_vals=y_vals)

        all_vals = np.concatenate(
            [
                grid_neural[np.isfinite(grid_neural)],
                grid_leaf_bucket[np.isfinite(grid_leaf_bucket)],
                grid_leaf_ridge[np.isfinite(grid_leaf_ridge)],
                grid_leaf_endpoint_table[np.isfinite(grid_leaf_endpoint_table)],
                grid_leaf_dt[np.isfinite(grid_leaf_dt)],
                grid_leaf_knn[np.isfinite(grid_leaf_knn)],
                grid_leaf_rf[np.isfinite(grid_leaf_rf)],
                grid_doc_neural[np.isfinite(grid_doc_neural)],
                grid_add[np.isfinite(grid_add)],
                grid_doc_add[np.isfinite(grid_doc_add)],
                grid_ridge[np.isfinite(grid_ridge)],
                grid_rf[np.isfinite(grid_rf)],
            ]
        )
        vmin = float(np.min(all_vals)) if all_vals.size else 0.0
        vmax = float(np.max(all_vals)) if all_vals.size else 1.0
        if not math.isfinite(vmin):
            vmin = 0.0
        if not math.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0

        panel_specs = [
            ("tree neural", grid_neural),
            ("leaf bucket", grid_leaf_bucket),
            ("leaf ridge tree", grid_leaf_ridge),
            ("leaf endpoint table", grid_leaf_endpoint_table),
            ("leaf DT tree", grid_leaf_dt),
            ("leaf kNN tree", grid_leaf_knn),
            ("leaf RF tree", grid_leaf_rf),
            ("full-doc neural", grid_doc_neural),
            ("tree additive", grid_add),
            ("full-doc additive", grid_doc_add),
            ("full-doc ridge", grid_ridge),
            ("RF root baseline", grid_rf),
        ]
        n_cols = min(4, len(panel_specs))
        n_rows = int(math.ceil(float(len(panel_specs)) / float(max(1, n_cols))))
        fig_w = max(16.0, 9.0 * float(n_cols))
        fig_h = max(8.0, 5.5 * float(n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True)
        axes_flat = np.atleast_1d(axes).reshape(-1)
        used_axes = []
        for ax, (label, grid) in zip(axes_flat, panel_specs):
            _heatmap(
                ax,
                grid=grid,
                x_labels=x_labels,
                y_labels=y_labels,
                title=f"Markov ({regime}) | {label} | root MAE",
                vmin=vmin,
                vmax=vmax,
            )
            used_axes.append(ax)
        for ax in axes_flat[len(panel_specs) :]:
            ax.axis("off")
        if used_axes:
            fig.colorbar(used_axes[0].images[0], ax=used_axes, fraction=0.030, pad=0.02)
        out_png = pages_dir / f"markov_{regime}_root_mae.png"
        _save_fig(fig, out_png)

        diagnostics.setdefault("markov", {}).setdefault("monotonicity", {})[regime] = {
            "neural": {
                "train_axis": _monotonicity_violations(grid_neural, axis=1, tol=0.0),
                "label_axis": _monotonicity_violations(grid_neural, axis=0, tol=0.0),
            },
            "leaf_bucket": {
                "train_axis": _monotonicity_violations(grid_leaf_bucket, axis=1, tol=0.0),
                "label_axis": _monotonicity_violations(grid_leaf_bucket, axis=0, tol=0.0),
            },
            "full_doc_neural": {
                "train_axis": _monotonicity_violations(grid_doc_neural, axis=1, tol=0.0),
                "label_axis": _monotonicity_violations(grid_doc_neural, axis=0, tol=0.0),
            },
            "leaf_ridge_tree": {
                "train_axis": _monotonicity_violations(grid_leaf_ridge, axis=1, tol=0.0),
                "label_axis": _monotonicity_violations(grid_leaf_ridge, axis=0, tol=0.0),
            },
            "leaf_endpoint_table_tree": {
                "train_axis": _monotonicity_violations(
                    grid_leaf_endpoint_table, axis=1, tol=0.0
                ),
                "label_axis": _monotonicity_violations(
                    grid_leaf_endpoint_table, axis=0, tol=0.0
                ),
            },
            "leaf_dt_tree": {
                "train_axis": _monotonicity_violations(grid_leaf_dt, axis=1, tol=0.0),
                "label_axis": _monotonicity_violations(grid_leaf_dt, axis=0, tol=0.0),
            },
            "leaf_knn_tree": {
                "train_axis": _monotonicity_violations(grid_leaf_knn, axis=1, tol=0.0),
                "label_axis": _monotonicity_violations(grid_leaf_knn, axis=0, tol=0.0),
            },
            "leaf_rf_tree": {
                "train_axis": _monotonicity_violations(grid_leaf_rf, axis=1, tol=0.0),
                "label_axis": _monotonicity_violations(grid_leaf_rf, axis=0, tol=0.0),
            },
            "additive": {
                "train_axis": _monotonicity_violations(grid_add, axis=1, tol=0.0),
                "label_axis": _monotonicity_violations(grid_add, axis=0, tol=0.0),
            },
            "full_doc_additive": {
                "train_axis": _monotonicity_violations(grid_doc_add, axis=1, tol=0.0),
                "label_axis": _monotonicity_violations(grid_doc_add, axis=0, tol=0.0),
            },
            "full_doc_ridge": {
                "train_axis": _monotonicity_violations(grid_ridge, axis=1, tol=0.0),
                "label_axis": _monotonicity_violations(grid_ridge, axis=0, tol=0.0),
            },
            "rf_root": {
                "train_axis": _monotonicity_violations(grid_rf, axis=1, tol=0.0),
                "label_axis": _monotonicity_violations(grid_rf, axis=0, tol=0.0),
            },
        }

        reg_sampled_rows = [r for r in markov_sampled_leaf_rows if r.regime == regime]
        if reg_sampled_rows:
            slice_train_docs = max(int(r.train_docs) for r in reg_sampled_rows)
            slice_audit_fraction = max(float(r.audit_fraction) for r in reg_sampled_rows)
            sample_slice = [
                r
                for r in reg_sampled_rows
                if int(r.train_docs) == int(slice_train_docs)
                and abs(float(r.audit_fraction) - float(slice_audit_fraction)) <= 1e-12
            ]
            point_payload: Dict[str, List[Dict[str, float | int]]] = {}
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
            for baseline_kind, color in (
                ("sampled_leaf_pool_ridge", "#D55E00"),
                ("sampled_leaf_pool_rf", "#0072B2"),
            ):
                kind_rows = [r for r in sample_slice if r.baseline_kind == baseline_kind]
                if not kind_rows:
                    continue
                budgets = sorted({int(r.leaf_budget) for r in kind_rows})
                xs_leaves: List[float] = []
                xs_frac: List[float] = []
                ys: List[float] = []
                summary_rows: List[Dict[str, float | int]] = []
                for budget in budgets:
                    budget_rows = [r for r in kind_rows if int(r.leaf_budget) == int(budget)]
                    xs_leaves.append(
                        float(
                            _median([float(r.sampled_leaves_mean_test) for r in budget_rows])
                        )
                    )
                    xs_frac.append(
                        float(
                            _median(
                                [float(r.sampled_token_fraction_mean_test) for r in budget_rows]
                            )
                        )
                    )
                    ys.append(float(_median([float(r.root_mae) for r in budget_rows])))
                    summary_rows.append(
                        {
                            "leaf_budget": int(budget),
                            "sampled_leaves_mean_test": float(xs_leaves[-1]),
                            "sampled_token_fraction_mean_test": float(xs_frac[-1]),
                            "root_mae": float(ys[-1]),
                        }
                    )
                label = (
                    "sampled pooled ridge"
                    if baseline_kind == "sampled_leaf_pool_ridge"
                    else "sampled pooled RF"
                )
                axes[0].plot(xs_leaves, ys, marker="o", linewidth=2.0, color=color, label=label)
                axes[1].plot(xs_frac, ys, marker="o", linewidth=2.0, color=color, label=label)
                point_payload[baseline_kind] = summary_rows

            slice_refs = [
                r
                for r in reg_rows
                if int(r.train_docs) == int(slice_train_docs)
                and abs(float(r.audit_fraction) - float(slice_audit_fraction)) <= 1e-12
            ]
            ref_specs = [
                (
                    "leaf bucket",
                    _median(
                        [
                            float(r.leaf_bucket_root_mae)
                            for r in slice_refs
                            if r.leaf_bucket_root_mae is not None
                            and math.isfinite(float(r.leaf_bucket_root_mae))
                        ]
                    ),
                    "#666666",
                ),
                (
                    "tree neural",
                    _median(
                        [float(r.learned_root_mae) for r in slice_refs if r.model_family == "neural"]
                    ),
                    "#009E73",
                ),
                (
                    "tree additive",
                    _median(
                        [float(r.learned_root_mae) for r in slice_refs if r.model_family == "additive"]
                    ),
                    "#CC79A7",
                ),
                (
                    "leaf endpoint table",
                    _median(
                        [
                            float(r.leaf_endpoint_table_tree_root_mae)
                            for r in slice_refs
                            if r.leaf_endpoint_table_tree_root_mae is not None
                            and math.isfinite(float(r.leaf_endpoint_table_tree_root_mae))
                        ]
                    ),
                    "#999999",
                ),
                (
                    "leaf DT tree",
                    _median(
                        [
                            float(r.leaf_dt_tree_root_mae)
                            for r in slice_refs
                            if r.leaf_dt_tree_root_mae is not None
                            and math.isfinite(float(r.leaf_dt_tree_root_mae))
                        ]
                    ),
                    "#7E6148",
                ),
                (
                    "leaf ridge tree",
                    _median(
                        [
                            float(r.leaf_ridge_tree_root_mae)
                            for r in slice_refs
                            if r.leaf_ridge_tree_root_mae is not None
                            and math.isfinite(float(r.leaf_ridge_tree_root_mae))
                        ]
                    ),
                    "#D55E00",
                ),
                (
                    "leaf kNN tree",
                    _median(
                        [
                            float(r.leaf_knn_tree_root_mae)
                            for r in slice_refs
                            if r.leaf_knn_tree_root_mae is not None
                            and math.isfinite(float(r.leaf_knn_tree_root_mae))
                        ]
                    ),
                    "#009E73",
                ),
                (
                    "leaf RF tree",
                    _median(
                        [
                            float(r.leaf_rf_tree_root_mae)
                            for r in slice_refs
                            if r.leaf_rf_tree_root_mae is not None
                            and math.isfinite(float(r.leaf_rf_tree_root_mae))
                        ]
                    ),
                    "#3C5488",
                ),
                (
                    "full-doc ridge",
                    _median(
                        [
                            float(r.doc_level_ridge_root_mae)
                            for r in slice_refs
                            if r.doc_level_ridge_root_mae is not None
                            and math.isfinite(float(r.doc_level_ridge_root_mae))
                        ]
                    ),
                    "#F0E442",
                ),
                (
                    "full-doc RF",
                    _median(
                        [
                            float(r.rf_root_mae)
                            for r in slice_refs
                            if r.rf_root_mae is not None and math.isfinite(float(r.rf_root_mae))
                        ]
                    ),
                    "#56B4E9",
                ),
            ]
            ref_payload: Dict[str, float] = {}
            for label, value, color in ref_specs:
                if value is None or not math.isfinite(float(value)):
                    continue
                axes[0].axhline(float(value), linestyle="--", linewidth=1.2, color=color, alpha=0.9, label=label)
                axes[1].axhline(float(value), linestyle="--", linewidth=1.2, color=color, alpha=0.9, label=label)
                ref_payload[str(label)] = float(value)

            axes[0].set_title(
                f"Markov ({regime}) | sampled-leaf pooled baselines | error vs leaves touched"
            )
            axes[0].set_xlabel("Mean sampled leaves per test doc")
            axes[0].set_ylabel("Root MAE")
            axes[1].set_title(
                f"Markov ({regime}) | sampled-leaf pooled baselines | error vs token coverage"
            )
            axes[1].set_xlabel("Mean sampled token fraction per test doc")
            axes[1].set_ylabel("Root MAE")
            for ax in axes:
                ax.grid(alpha=0.25)
                ax.legend(fontsize=9)
            out_png = pages_dir / f"markov_{regime}_sampled_leaf_efficiency.png"
            _save_fig(fig, out_png)
            diagnostics.setdefault("markov", {}).setdefault("sampled_leaf_pool_efficiency", {})[
                regime
            ] = {
                "selected_slice": {
                    "train_docs": int(slice_train_docs),
                    "audit_fraction": float(slice_audit_fraction),
                },
                "points": point_payload,
                "references": ref_payload,
            }

        ladder_train_docs = max(int(r.train_docs) for r in reg_rows)
        ladder_audit_fraction = max(float(r.audit_fraction) for r in reg_rows)
        ladder_slice_refs = [
            r
            for r in reg_rows
            if int(r.train_docs) == int(ladder_train_docs)
            and abs(float(r.audit_fraction) - float(ladder_audit_fraction)) <= 1e-12
        ]
        ladder_specs = [
            (
                "leaf bucket",
                _median(
                    [
                        float(r.leaf_bucket_root_mae)
                        for r in ladder_slice_refs
                        if r.leaf_bucket_root_mae is not None
                        and math.isfinite(float(r.leaf_bucket_root_mae))
                    ]
                ),
                "#666666",
            ),
            (
                "leaf endpoint table",
                _median(
                    [
                        float(r.leaf_endpoint_table_tree_root_mae)
                        for r in ladder_slice_refs
                        if r.leaf_endpoint_table_tree_root_mae is not None
                        and math.isfinite(float(r.leaf_endpoint_table_tree_root_mae))
                    ]
                ),
                "#999999",
            ),
            (
                "leaf DT tree",
                _median(
                    [
                        float(r.leaf_dt_tree_root_mae)
                        for r in ladder_slice_refs
                        if r.leaf_dt_tree_root_mae is not None
                        and math.isfinite(float(r.leaf_dt_tree_root_mae))
                    ]
                ),
                "#7E6148",
            ),
            (
                "leaf ridge tree",
                _median(
                    [
                        float(r.leaf_ridge_tree_root_mae)
                        for r in ladder_slice_refs
                        if r.leaf_ridge_tree_root_mae is not None
                        and math.isfinite(float(r.leaf_ridge_tree_root_mae))
                    ]
                ),
                "#D55E00",
            ),
            (
                "leaf kNN tree",
                _median(
                    [
                        float(r.leaf_knn_tree_root_mae)
                        for r in ladder_slice_refs
                        if r.leaf_knn_tree_root_mae is not None
                        and math.isfinite(float(r.leaf_knn_tree_root_mae))
                    ]
                ),
                "#009E73",
            ),
            (
                "leaf RF tree",
                _median(
                    [
                        float(r.leaf_rf_tree_root_mae)
                        for r in ladder_slice_refs
                        if r.leaf_rf_tree_root_mae is not None
                        and math.isfinite(float(r.leaf_rf_tree_root_mae))
                    ]
                ),
                "#3C5488",
            ),
            (
                "tree neural",
                _median(
                    [
                        float(r.learned_root_mae)
                        for r in ladder_slice_refs
                        if r.model_family == "neural"
                    ]
                ),
                "#00A087",
            ),
            (
                "full-doc ridge",
                _median(
                    [
                        float(r.doc_level_ridge_root_mae)
                        for r in ladder_slice_refs
                        if r.doc_level_ridge_root_mae is not None
                        and math.isfinite(float(r.doc_level_ridge_root_mae))
                    ]
                ),
                "#E1BE6A",
            ),
            (
                "full-doc RF",
                _median(
                    [
                        float(r.rf_root_mae)
                        for r in ladder_slice_refs
                        if r.rf_root_mae is not None and math.isfinite(float(r.rf_root_mae))
                    ]
                ),
                "#4DBBD5",
            ),
            (
                "tree additive",
                _median(
                    [
                        float(r.learned_root_mae)
                        for r in ladder_slice_refs
                        if r.model_family == "additive"
                    ]
                ),
                "#CC79A7",
            ),
        ]
        ladder_points = [
            {"label": str(label), "root_mae": float(value), "color": str(color)}
            for label, value, color in ladder_specs
            if value is not None and math.isfinite(float(value))
        ]
        if ladder_points:
            fig, ax = plt.subplots(1, 1, figsize=(11, 6), constrained_layout=True)
            xs = np.arange(len(ladder_points), dtype=np.float64)
            vals = [float(point["root_mae"]) for point in ladder_points]
            colors = [str(point["color"]) for point in ladder_points]
            labels = [str(point["label"]) for point in ladder_points]
            ax.bar(xs, vals, color=colors, alpha=0.9)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("Root MAE")
            ax.set_title(
                f"Markov ({regime}) | local capacity ladder | train={ladder_train_docs}, audit={ladder_audit_fraction:.3g}"
            )
            ax.grid(axis="y", alpha=0.25)
            out_png = pages_dir / f"markov_{regime}_capacity_ladder.png"
            _save_fig(fig, out_png)
            diagnostics.setdefault("markov", {}).setdefault("capacity_ladder", {})[
                regime
            ] = {
                "selected_slice": {
                    "train_docs": int(ladder_train_docs),
                    "audit_fraction": float(ladder_audit_fraction),
                },
                "points": ladder_points,
            }

    # --- C-TreePO pages (baseline + hard + lda; eval pairs) ---
    # Test-set stability diagnostics (per regime/seed/DGP key).
    stability: Dict[str, object] = {}
    for r in ctree_rows:
        key = (r.regime, r.seed, r.dgp_key)
        entry = stability.setdefault(str(key), {"regime": r.regime, "seed": r.seed, "dgp_key": list(r.dgp_key), "sigs": []})
        sig = r.corpus_signature_test
        if sig is not None:
            entry["sigs"].append(str(sig))
    stability_summary: Dict[str, object] = {"n_groups": int(len(stability)), "n_fail": 0, "fail_examples": []}
    for k, v in stability.items():
        sigs = [str(s) for s in (v.get("sigs") or []) if str(s)]
        uniq = sorted(set(sigs))
        if len(uniq) > 1:
            stability_summary["n_fail"] += 1
            if len(stability_summary["fail_examples"]) < 10:
                stability_summary["fail_examples"].append({"group": k, "unique_signatures": uniq[:5]})
    diagnostics.setdefault("ctree", {})["test_set_stability"] = stability_summary

    desired_eval_pairs = [(0.0, 0.0), (0.5, 0.5)]
    for regime in ("baseline", "hard", "lda"):
        reg_rows = [r for r in ctree_rows if r.regime == regime]
        if not reg_rows:
            continue
        x_vals = sorted({int(r.n_books_train) for r in reg_rows})
        y_vals = sorted({float(r.calibration_rate) for r in reg_rows})
        x_labels = [str(x) for x in x_vals]
        y_labels = [f"{y:.3g}" for y in y_vals]

        for eval_leaf, eval_internal in desired_eval_pairs:
            pair_rows = [
                r
                for r in reg_rows
                if abs(float(r.eval_leaf_rate) - float(eval_leaf)) <= 1e-12
                and abs(float(r.eval_internal_rate) - float(eval_internal)) <= 1e-12
            ]
            if not pair_rows:
                continue

            def sel(theta: str, phi: Optional[str] = None) -> List[Tuple[int, float, float]]:
                out: List[Tuple[int, float, float]] = []
                for rr in pair_rows:
                    if rr.leaf_theta_estimator != theta:
                        continue
                    if phi is not None and rr.topic_phi_estimator != phi:
                        continue
                    out.append((rr.n_books_train, rr.calibration_rate, rr.root_l1_mean))
                return out

            phi_present = sorted({str(r.topic_phi_estimator) for r in pair_rows if r.leaf_theta_estimator == "lstsq"})
            phi_priority = [
                "spectral_numpy",
                "embedding_spectral",
                "tensor_lda",
                "online_tensor_lda",
                "true",
                "noisy_theory",
            ]

            def phi_sort_key(phi: str) -> Tuple[int, str]:
                try:
                    idx = phi_priority.index(str(phi))
                except ValueError:
                    idx = 999
                return (int(idx), str(phi))

            phi_list = sorted(phi_present, key=phi_sort_key)
            grids_phi = {phi: _build_grid(sel("lstsq", phi), x_vals=x_vals, y_vals=y_vals) for phi in phi_list}
            grid_sklearn = _build_grid(sel("sklearn_lda", None), x_vals=x_vals, y_vals=y_vals)
            grid_rf = _build_grid(sel("rf", None), x_vals=x_vals, y_vals=y_vals)
            grid_mlp = _build_grid(sel("mlp", None), x_vals=x_vals, y_vals=y_vals)
            full_doc_rf_rows = [
                (rr.n_books_train, rr.calibration_rate, float(rr.full_doc_rf_root_l1_mean))
                for rr in pair_rows
                if rr.full_doc_rf_root_l1_mean is not None
                and math.isfinite(float(rr.full_doc_rf_root_l1_mean))
            ]
            full_doc_mlp_rows = [
                (rr.n_books_train, rr.calibration_rate, float(rr.full_doc_mlp_root_l1_mean))
                for rr in pair_rows
                if rr.full_doc_mlp_root_l1_mean is not None
                and math.isfinite(float(rr.full_doc_mlp_root_l1_mean))
            ]
            grid_full_doc_rf = _build_grid(full_doc_rf_rows, x_vals=x_vals, y_vals=y_vals)
            grid_full_doc_mlp = _build_grid(full_doc_mlp_rows, x_vals=x_vals, y_vals=y_vals)

            all_chunks = [
                grid_sklearn[np.isfinite(grid_sklearn)],
                grid_rf[np.isfinite(grid_rf)],
                grid_mlp[np.isfinite(grid_mlp)],
                grid_full_doc_rf[np.isfinite(grid_full_doc_rf)],
                grid_full_doc_mlp[np.isfinite(grid_full_doc_mlp)],
            ]
            for grid in grids_phi.values():
                all_chunks.append(grid[np.isfinite(grid)])
            all_vals = np.concatenate(all_chunks) if any(chunk.size for chunk in all_chunks) else np.asarray([], dtype=np.float64)
            vmin = float(np.min(all_vals)) if all_vals.size else 0.0
            vmax = float(np.max(all_vals)) if all_vals.size else 1.0
            if not math.isfinite(vmin):
                vmin = 0.0
            if not math.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0

            n_phi = int(len(phi_list))
            n_panels = int(n_phi + 5)  # + sklearn_lda + rf + mlp + full-doc rf + full-doc mlp
            n_cols = 2 if n_panels <= 4 else 3
            n_rows = int(math.ceil(float(n_panels) / float(n_cols)))
            fig_w = 2.0 + 6.0 * float(n_cols)
            fig_h = 1.0 + 5.0 * float(n_rows)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True)
            axes_flat = np.asarray(axes).reshape(-1)
            panel_axes = axes_flat[:n_panels]
            extra_axes = axes_flat[n_panels:]

            panel_idx = 0
            for phi in phi_list:
                _heatmap(
                    panel_axes[panel_idx],
                    grid=grids_phi[phi],
                    x_labels=x_labels,
                    y_labels=y_labels,
                    title=(
                        f"C-TreePO ({regime}) | eval=({eval_leaf:g},{eval_internal:g}) | θ=lstsq, φ={phi} | root L1 mean"
                    ),
                    vmin=vmin,
                    vmax=vmax,
                )
                panel_idx += 1

            _heatmap(
                panel_axes[panel_idx],
                grid=grid_sklearn,
                x_labels=x_labels,
                y_labels=y_labels,
                title=f"C-TreePO ({regime}) | eval=({eval_leaf:g},{eval_internal:g}) | θ=sklearn_lda | root L1 mean",
                vmin=vmin,
                vmax=vmax,
            )
            panel_idx += 1
            _heatmap(
                panel_axes[panel_idx],
                grid=grid_rf,
                x_labels=x_labels,
                y_labels=y_labels,
                title=f"C-TreePO ({regime}) | eval=({eval_leaf:g},{eval_internal:g}) | θ=rf (supervised) | root L1 mean",
                vmin=vmin,
                vmax=vmax,
            )
            panel_idx += 1
            _heatmap(
                panel_axes[panel_idx],
                grid=grid_mlp,
                x_labels=x_labels,
                y_labels=y_labels,
                title=f"C-TreePO ({regime}) | eval=({eval_leaf:g},{eval_internal:g}) | θ=mlp (supervised) | root L1 mean",
                vmin=vmin,
                vmax=vmax,
            )
            panel_idx += 1
            _heatmap(
                panel_axes[panel_idx],
                grid=grid_full_doc_rf,
                x_labels=x_labels,
                y_labels=y_labels,
                title=(
                    f"C-TreePO ({regime}) | eval=({eval_leaf:g},{eval_internal:g}) | "
                    f"full-doc rf (no tree) | root L1 mean"
                ),
                vmin=vmin,
                vmax=vmax,
            )
            panel_idx += 1
            _heatmap(
                panel_axes[panel_idx],
                grid=grid_full_doc_mlp,
                x_labels=x_labels,
                y_labels=y_labels,
                title=(
                    f"C-TreePO ({regime}) | eval=({eval_leaf:g},{eval_internal:g}) | "
                    f"full-doc mlp (no tree) | root L1 mean"
                ),
                vmin=vmin,
                vmax=vmax,
            )

            for ax in extra_axes.tolist():
                ax.set_axis_off()

            if panel_axes.size > 0 and panel_axes[0].images:
                fig.colorbar(panel_axes[0].images[0], ax=axes_flat.tolist(), fraction=0.030, pad=0.02)
            tag = f"{eval_leaf:g}".replace(".", "p")
            out_png = pages_dir / f"ctree_{regime}_eval_{tag}_root_l1_mean.png"
            _save_fig(fig, out_png)

            theta_lstsq_diag = {
                phi: {
                    "train_axis": _monotonicity_violations(grid, axis=1, tol=0.0),
                    "label_axis": _monotonicity_violations(grid, axis=0, tol=0.0),
                }
                for phi, grid in grids_phi.items()
            }
            diagnostics.setdefault("ctree", {}).setdefault("monotonicity", {}).setdefault(regime, {})[
                f"eval_{eval_leaf:g}"
            ] = {
                "theta_lstsq": theta_lstsq_diag,
                "theta_rf": {
                    "train_axis": _monotonicity_violations(grid_rf, axis=1, tol=0.0),
                    "label_axis": _monotonicity_violations(grid_rf, axis=0, tol=0.0),
                },
                "theta_mlp": {
                    "train_axis": _monotonicity_violations(grid_mlp, axis=1, tol=0.0),
                    "label_axis": _monotonicity_violations(grid_mlp, axis=0, tol=0.0),
                },
                "full_doc_rf": {
                    "train_axis": _monotonicity_violations(grid_full_doc_rf, axis=1, tol=0.0),
                    "label_axis": _monotonicity_violations(grid_full_doc_rf, axis=0, tol=0.0),
                },
                "full_doc_mlp": {
                    "train_axis": _monotonicity_violations(grid_full_doc_mlp, axis=1, tol=0.0),
                    "label_axis": _monotonicity_violations(grid_full_doc_mlp, axis=0, tol=0.0),
                },
            }

        # Topic-phi recovery + decomposition components.
        # These are easiest to interpret at eval=(0,0), but are independent of eval-time guidance.
        diag_rows = [
            r
            for r in reg_rows
            if abs(float(r.eval_leaf_rate) - 0.0) <= 1e-12 and abs(float(r.eval_internal_rate) - 0.0) <= 1e-12
        ]
        if not diag_rows:
            diag_rows = list(reg_rows)

        phi_present_diag = sorted({str(r.topic_phi_estimator) for r in diag_rows if r.leaf_theta_estimator == "lstsq"})
        phi_priority_diag = [
            "spectral_numpy",
            "embedding_spectral",
            "tensor_lda",
            "online_tensor_lda",
            "sklearn_lda",
            "true",
            "noisy_theory",
        ]

        def phi_sort_key_diag(phi: str) -> Tuple[int, str]:
            try:
                idx = phi_priority_diag.index(str(phi))
            except ValueError:
                idx = 999
            return (int(idx), str(phi))

        phi_list_diag = sorted(phi_present_diag, key=phi_sort_key_diag)

        def sel_diag(phi: str, *, field: str) -> List[Tuple[int, float, float]]:
            out: List[Tuple[int, float, float]] = []
            for rr in diag_rows:
                if rr.leaf_theta_estimator != "lstsq":
                    continue
                if str(rr.topic_phi_estimator) != str(phi):
                    continue
                val = getattr(rr, field)
                if val is None or not math.isfinite(float(val)):
                    continue
                out.append((rr.n_books_train, rr.calibration_rate, float(val)))
            return out

        # (1) φ recovery (topic-word L2 error; permutation-aligned).
        grids_phi_l2 = {phi: _build_grid(sel_diag(phi, field="topic_phi_l2_error_mean"), x_vals=x_vals, y_vals=y_vals) for phi in phi_list_diag}
        all_vals = np.concatenate([g[np.isfinite(g)] for g in grids_phi_l2.values()]) if grids_phi_l2 else np.asarray([], dtype=np.float64)
        vmin = float(np.min(all_vals)) if all_vals.size else 0.0
        vmax = float(np.max(all_vals)) if all_vals.size else 1.0
        if not math.isfinite(vmin):
            vmin = 0.0
        if not math.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        if grids_phi_l2:
            n_phi = int(len(phi_list_diag))
            n_cols = 2 if n_phi <= 4 else 3
            n_rows = int(math.ceil(float(n_phi) / float(n_cols)))
            fig_w = 2.0 + 6.0 * float(n_cols)
            fig_h = 1.0 + 5.0 * float(n_rows)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True)
            axes_flat = np.asarray(axes).reshape(-1)
            for ax, phi in zip(axes_flat.tolist(), phi_list_diag):
                _heatmap(
                    ax,
                    grid=grids_phi_l2[phi],
                    x_labels=x_labels,
                    y_labels=y_labels,
                    title=f"C-TreePO ({regime}) | θ=lstsq, φ={phi} | topic-word L2 error mean",
                    vmin=vmin,
                    vmax=vmax,
                )
            for ax in axes_flat[len(phi_list_diag) :].tolist():
                ax.set_axis_off()
            if axes_flat.size > 0 and axes_flat[0].images:
                fig.colorbar(axes_flat[0].images[0], ax=axes_flat.tolist(), fraction=0.030, pad=0.02)
            out_png = pages_dir / f"ctree_{regime}_topic_phi_l2_error_mean.png"
            _save_fig(fig, out_png)

        # (2) Topic-estimation contribution to root error (truth->oracle_proxy vs estimated->oracle_proxy).
        grids_topic_comp = {
            phi: _build_grid(sel_diag(phi, field="topic_component_mean"), x_vals=x_vals, y_vals=y_vals) for phi in phi_list_diag
        }
        all_vals = np.concatenate([g[np.isfinite(g)] for g in grids_topic_comp.values()]) if grids_topic_comp else np.asarray([], dtype=np.float64)
        vmin = float(np.min(all_vals)) if all_vals.size else 0.0
        vmax = float(np.max(all_vals)) if all_vals.size else 1.0
        if not math.isfinite(vmin):
            vmin = 0.0
        if not math.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        if grids_topic_comp:
            n_phi = int(len(phi_list_diag))
            n_cols = 2 if n_phi <= 4 else 3
            n_rows = int(math.ceil(float(n_phi) / float(n_cols)))
            fig_w = 2.0 + 6.0 * float(n_cols)
            fig_h = 1.0 + 5.0 * float(n_rows)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True)
            axes_flat = np.asarray(axes).reshape(-1)
            for ax, phi in zip(axes_flat.tolist(), phi_list_diag):
                _heatmap(
                    ax,
                    grid=grids_topic_comp[phi],
                    x_labels=x_labels,
                    y_labels=y_labels,
                    title=f"C-TreePO ({regime}) | θ=lstsq, φ={phi} | topic component (root L1)",
                    vmin=vmin,
                    vmax=vmax,
                )
            for ax in axes_flat[len(phi_list_diag) :].tolist():
                ax.set_axis_off()
            if axes_flat.size > 0 and axes_flat[0].images:
                fig.colorbar(axes_flat[0].images[0], ax=axes_flat.tolist(), fraction=0.030, pad=0.02)
            out_png = pages_dir / f"ctree_{regime}_topic_component_mean.png"
            _save_fig(fig, out_png)

        # (3) Oracle-proxy floor (error even with true φ; dominated by finite leaf tokens).
        floor_rows: List[Tuple[int, float, float]] = []
        for rr in diag_rows:
            val = rr.oracle_proxy_component_mean
            if val is None or not math.isfinite(float(val)):
                continue
            floor_rows.append((rr.n_books_train, rr.calibration_rate, float(val)))
        if floor_rows:
            grid_floor = _build_grid(floor_rows, x_vals=x_vals, y_vals=y_vals)
            vals = grid_floor[np.isfinite(grid_floor)]
            vmin = float(np.min(vals)) if vals.size else 0.0
            vmax = float(np.max(vals)) if vals.size else 1.0
            if not math.isfinite(vmin):
                vmin = 0.0
            if not math.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
            fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
            _heatmap(
                ax,
                grid=grid_floor,
                x_labels=x_labels,
                y_labels=y_labels,
                title=f"C-TreePO ({regime}) | oracle-proxy floor (root L1 mean; true φ, no queries)",
                vmin=vmin,
                vmax=vmax,
            )
            fig.colorbar(ax.images[0], ax=[ax], fraction=0.030, pad=0.02)
            out_png = pages_dir / f"ctree_{regime}_oracle_proxy_component_mean.png"
            _save_fig(fig, out_png)

        # Leaf-theta error pages (supervised estimators only; eval rates irrelevant).
        def theta_err_rows(theta: str) -> List[Tuple[int, float, float]]:
            out: List[Tuple[int, float, float]] = []
            for rr in reg_rows:
                if rr.leaf_theta_estimator != theta:
                    continue
                if rr.leaf_theta_l1_mean is None or not math.isfinite(float(rr.leaf_theta_l1_mean)):
                    continue
                out.append((rr.n_books_train, rr.calibration_rate, float(rr.leaf_theta_l1_mean)))
            return out

        grid_err_rf = _build_grid(theta_err_rows("rf"), x_vals=x_vals, y_vals=y_vals)
        grid_err_mlp = _build_grid(theta_err_rows("mlp"), x_vals=x_vals, y_vals=y_vals)
        all_vals = np.concatenate([grid_err_rf[np.isfinite(grid_err_rf)], grid_err_mlp[np.isfinite(grid_err_mlp)]])
        vmin = float(np.min(all_vals)) if all_vals.size else 0.0
        vmax = float(np.max(all_vals)) if all_vals.size else 1.0
        if not math.isfinite(vmin):
            vmin = 0.0
        if not math.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        _heatmap(
            axes[0],
            grid=grid_err_rf,
            x_labels=x_labels,
            y_labels=y_labels,
            title=f"Leaf-θ error ({regime}) | RF | test leaf θ L1 mean",
            vmin=vmin,
            vmax=vmax,
        )
        _heatmap(
            axes[1],
            grid=grid_err_mlp,
            x_labels=x_labels,
            y_labels=y_labels,
            title=f"Leaf-θ error ({regime}) | MLP | test leaf θ L1 mean",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(axes[0].images[0], ax=list(axes), fraction=0.030, pad=0.02)
        out_png = pages_dir / f"ctree_{regime}_leaf_theta_l1_mean.png"
        _save_fig(fig, out_png)

    diag_path = out_dir / "identifiable_zero_learnability_latest_diagnostics.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_path = out_dir / "identifiable_zero_learnability_latest.md"
    pdf_path = out_dir / "identifiable_zero_learnability_latest.pdf"

    def img(p: Path) -> str:
        rel = p.relative_to(out_dir)
        return f"![]({str(rel)}){{ width=100% }}"

    md_lines: List[str] = []
    md_lines.extend(
        [
            "---",
            "title: Identifiable-Zero Learnability Benchmarks (v1)",
            f"date: {now.strftime('%Y-%m-%d')}",
            "fontsize: 12pt",
            "geometry: margin=0.7in",
            "---",
            "",
            f"**Output root:** `{output_root}`  ",
            f"**Generated (UTC):** `{now.isoformat()}`",
            "",
            "## Read this first (what is varied, what is fixed)",
            "",
            "**Markov (OPS-count)**",
            "- Varied: `train_docs` (more training documents), `audit_fraction` (more internal-node oracle labels).",
            "- Fixed: held-out `test_docs` per run; generated with seed offset so the test set is stable across `train_docs`.",
            "- Metric: `root_mae` on held-out test docs at decision-time oracle visibility `q_infer=0` (no inference guidance).",
            "- Comparison lanes now include a local-capacity ladder: theorem-facing `leaf bucket`, piecewise-constant `leaf endpoint table`, then a single-tree `leaf DT tree`, followed by `leaf ridge tree`, `leaf kNN tree`, and `leaf RF tree`. All preserve the same exact additive merge law for the global document result.",
            "- No-tree lanes include the same `neural` or `additive` sketch family trained on a single full-document leaf (`full-doc neural` / `full-doc additive`), plus a closed-form pooled-feature ridge regressor (`full-doc ridge`).",
            "- When available, the Markov pages also add a fairness-oriented sampled-leaf pooled comparison: uniformly sample `k` realized leaves, pool those leaf features without a tree, and track `root_mae` against leaves touched and token coverage.",
            "",
            "**C-TreePO (segmented LDA)**",
            "- Varied: `n_books_train` (more training books), `calibration_leaf_query_rate` (more leaf oracle labels for calibration/training).",
            "- Fixed: held-out `n_books_test` per run; generated from a dedicated RNG stream so the test set is stable across `n_books_train`.",
            "- Metric: `estimated_calibrated_budgeted.root_l1_mean` on the fixed held-out test set.",
            "- Supervised leaf-theta baselines (`leaf_theta_estimator in {rf,mlp}`) also report held-out test leaf theta error in `topic_meta.leaf_theta_l1_mean`.",
            "- When available, those supervised estimators also include matched no-tree baselines (`full-doc rf` / `full-doc mlp`) trained on full-document counts and root topic mixtures from the same train-book slice.",
            "- Diagnostics: `topic_meta.topic_phi_l2_error_mean` (topic recovery), `decomposition.topic_component_mean` (root error attributable to topic estimation), `decomposition.oracle_proxy_component_mean` (irreducible floor from finite leaf tokens; even with true phi).",
            "",
            f"**Diagnostics JSON:** `{diag_path}`",
            "",
            "## Theory alignment",
            "",
            "- Markov pages are the learnability counterpart to `lean3/FormalProofs/OPT/MarkovCountSketchExample.lean`: exact mergeability is the control, while held-out `root_mae` is the main downstream objective.",
            "- The LDA/C-TreePO pages separate two Lean stories. `lean3/FormalProofs/OPT/BagOfWordsLDARecovery.lean` is the exact bag-of-words control where pooled counts are sufficient.",
            "- `lean3/FormalProofs/OPT/LeafLocalMixtureUtilityGap.lean` is the separate nonlinear local-utility theorem explaining when leafwise information can matter beyond the pooled document average.",
            "- In this report, `root_mae` (Markov) and `root_l1_mean` (C-TreePO) are the primary operational metrics. C1/C3/topic-recovery/oracle-proxy pages are supporting diagnostics about why those primary errors move.",
            "- Only quote Markov-vs-LDA cross-family comparisons when the setup check below says the train grids, label-rate grids, and held-out test sizes match, and only when `subset_mode` is `cross_family`.",
            "",
            "## Suite subset and setup check",
            "",
            f"- `subset_mode`: `{subset_mode}`",
        ]
    )
    if subset_mode == "cross_family":
        md_lines.extend(
            [
                f"- Matched Markov/LDA grids: `{bool((diagnostics.get('setup_alignment') or {}).get('matches', False))}`",
                f"- Safe to quote cross-family comparison: `{bool((diagnostics.get('setup_alignment') or {}).get('safe_to_quote_cross_family', False))}`",
                f"- Markov train docs: `{markov_train_grid}`",
                f"- LDA train docs: `{ctree_train_grid}`",
                f"- Markov label rates: `{markov_label_grid}`",
                f"- LDA label rates: `{ctree_label_grid}`",
                f"- Markov held-out docs: `{markov_test_grid}`",
                f"- LDA held-out docs: `{ctree_test_grid}`",
                "- If this check is false, treat the current pages as family-specific diagnostics rather than a matched Markov/LDA comparison.",
                "",
                "\\newpage",
            ]
        )
    elif subset_mode == "markov_only":
        md_lines.extend(
            [
                "- This output root intentionally includes only the Markov learnability groups.",
                "- Cross-family alignment checks are not applicable for this subset.",
                f"- Markov train docs: `{markov_train_grid}`",
                f"- Markov label rates: `{markov_label_grid}`",
                f"- Markov held-out docs: `{markov_test_grid}`",
                "",
                "\\newpage",
            ]
        )
    else:
        md_lines.extend(
            [
                "- This output root intentionally includes only the C-TreePO learnability groups.",
                "- Cross-family alignment checks are not applicable for this subset.",
                f"- LDA train docs: `{ctree_train_grid}`",
                f"- LDA label rates: `{ctree_label_grid}`",
                f"- LDA held-out docs: `{ctree_test_grid}`",
                "",
                "\\newpage",
            ]
        )

    for regime in ("baseline", "hard", "lda"):
        p = pages_dir / f"markov_{regime}_root_mae.png"
        if p.exists():
            md_lines.extend([f"## Markov learnability ({regime})", "", img(p), "", "\\newpage"])

    for regime in ("baseline", "hard", "lda"):
        p = pages_dir / f"markov_{regime}_sampled_leaf_efficiency.png"
        if p.exists():
            md_lines.extend(
                [
                    f"## Markov sampled-leaf pooled efficiency ({regime})",
                    "",
                    img(p),
                    "",
                    "\\newpage",
                ]
            )

    for regime in ("baseline", "hard", "lda"):
        p = pages_dir / f"markov_{regime}_capacity_ladder.png"
        if p.exists():
            md_lines.extend(
                [
                    f"## Markov local capacity ladder ({regime})",
                    "",
                    img(p),
                    "",
                    "\\newpage",
                ]
            )

    for regime in ("baseline", "hard", "lda"):
        for q in ("0", "0p5"):
            p = pages_dir / f"ctree_{regime}_eval_{q}_root_l1_mean.png"
            if p.exists():
                md_lines.extend([f"## C-TreePO learnability ({regime}) | eval q={q.replace('p','.')}", "", img(p), "", "\\newpage"])

    for regime in ("baseline", "hard", "lda"):
        p = pages_dir / f"ctree_{regime}_topic_phi_l2_error_mean.png"
        if p.exists():
            md_lines.extend([f"## C-TreePO topic recovery ({regime})", "", img(p), "", "\\newpage"])

    for regime in ("baseline", "hard", "lda"):
        p = pages_dir / f"ctree_{regime}_topic_component_mean.png"
        if p.exists():
            md_lines.extend([f"## C-TreePO topic contribution ({regime})", "", img(p), "", "\\newpage"])

    for regime in ("baseline", "hard", "lda"):
        p = pages_dir / f"ctree_{regime}_oracle_proxy_component_mean.png"
        if p.exists():
            md_lines.extend([f"## C-TreePO oracle-proxy floor ({regime})", "", img(p), "", "\\newpage"])

    for regime in ("baseline", "hard", "lda"):
        p = pages_dir / f"ctree_{regime}_leaf_theta_l1_mean.png"
        if p.exists():
            md_lines.extend([f"## Leaf-theta prediction error ({regime})", "", img(p), "", "\\newpage"])

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    pdf_ok = False
    if bool(args.emit_pdf):
        try:
            pdf_ok = _run_pandoc(md_path, pdf_path)
        except Exception:
            pdf_ok = False

    print(f"wrote_markdown | {md_path}")
    print(f"wrote_diagnostics | {diag_path}")
    if bool(args.emit_pdf):
        print(f"wrote_pdf | {pdf_path} | ok={pdf_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
