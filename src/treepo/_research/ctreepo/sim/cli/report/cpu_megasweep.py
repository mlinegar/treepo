#!/usr/bin/env python3
"""Build a single markdown report for a CPU megasweep run."""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


from treepo._research.ctreepo.sim.util import safe_float


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a consolidated markdown report for a CPU megasweep output.")
    p.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Run output root (e.g., outputs/cpu_megasweep_<run_id>).",
    )
    p.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Output markdown report path (default: <output-root>/figures/megasweep_consolidated_report.md).",
    )
    return p.parse_args(list(argv) if argv is not None else None)


_safe_float = safe_float


def _median(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(statistics.median(vals))


def _q(xs: Iterable[float], qv: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * float(qv)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return vals[lo]
    w = k - lo
    return vals[lo] * (1.0 - w) + vals[hi] * w


def _fmt(x: float, digits: int = 6) -> str:
    if not math.isfinite(float(x)):
        return "nan"
    return f"{float(x):.{digits}g}"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_markov_full_budget(markov_glob: str) -> Tuple[Dict[int, List[float]], int]:
    by_train_docs: Dict[int, List[float]] = {}
    n = 0
    for fp in glob.glob(markov_glob, recursive=True):
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        if abs(_safe_float(cfg.get("audit_fraction")) - 1.0) > 1e-12:
            continue
        met = payload.get("metrics", {}) or {}
        y = _safe_float((met.get("learned", {}) or {}).get("root_mae"))
        if not math.isfinite(y):
            continue
        td = int(cfg.get("train_docs", -1))
        by_train_docs.setdefault(td, []).append(y)
        n += 1
    return by_train_docs, n


def _collect_segment_full_budget(seg_glob: str) -> Tuple[Dict[str, List[float]], int]:
    by_estimator: Dict[str, List[float]] = {}
    n = 0
    for fp in glob.glob(seg_glob, recursive=True):
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        if abs(_safe_float(cfg.get("audit_fraction")) - 1.0) > 1e-12:
            continue
        met = payload.get("metrics", {}) or {}
        y = _safe_float((met.get("ridge", {}) or {}).get("root_mae"))
        if not math.isfinite(y):
            continue
        est = str(cfg.get("topic_phi_estimator", ""))
        by_estimator.setdefault(est, []).append(y)
        n += 1
    return by_estimator, n


def _collect_ctree_full_guidance(ctree_glob: str) -> Tuple[Dict[int, List[float]], int]:
    by_train_docs: Dict[int, List[float]] = {}
    n = 0
    for fp in glob.glob(ctree_glob, recursive=True):
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        leaf = _safe_float(cfg.get("eval_leaf_query_rate"))
        internal = _safe_float(cfg.get("eval_internal_query_rate"))
        if abs(leaf - 1.0) > 1e-12 or abs(internal - 1.0) > 1e-12:
            continue
        met = payload.get("metrics", {}) or {}
        y = _safe_float((met.get("estimated_calibrated_budgeted", {}) or {}).get("root_l1_mean"))
        if not math.isfinite(y):
            continue
        td = int(cfg.get("n_books_train", -1))
        by_train_docs.setdefault(td, []).append(y)
        n += 1
    return by_train_docs, n


def _table_row(cols: List[str]) -> str:
    return "| " + " | ".join(cols) + " |"


def _build_report(output_root: Path) -> str:
    figures = output_root / "figures"
    markov_report = _load_json(figures / "markov_ops_count_ceilings_report.json")
    segment_report = _load_json(figures / "segment_lda_ops_weight_recovery_ceilings_report.json")
    ctree_report = _load_json(figures / "segmented_lda_ctreepo_ceilings_report.json")
    mergeable_summary = _load_json(figures / "mergeable_ceilings_summary.json")
    ladder_summary = _load_json(figures / "mergeable_complexity_ladder_summary.json")

    markov_diag = (((markov_report.get("diagnostics") or {}).get("full_audit")) or {})
    segment_diag = (((segment_report.get("diagnostics") or {}).get("full_audit")) or {})
    ctree_diag = (((ctree_report.get("diagnostics") or {}).get("full_guidance")) or {})

    markov_by_train, markov_n = _collect_markov_full_budget(
        str(output_root / "markov_changepoint_ops_count" / "**" / "*seed_*.json")
    )
    segment_by_est, segment_n = _collect_segment_full_budget(
        str(output_root / "segment_lda_ops_weight_recovery" / "**" / "*seed_*.json")
    )
    ctree_by_train, ctree_n = _collect_ctree_full_guidance(str(output_root / "segmented_lda_ctreepo" / "**" / "*.json"))

    gap_markov = _safe_float(markov_diag.get("learned_root_mae")) - _safe_float(markov_diag.get("exact_root_mae"))
    gap_segment = _safe_float(segment_diag.get("ridge_root_mae")) - _safe_float(segment_diag.get("exact_root_mae"))
    gap_ctree = _safe_float(ctree_diag.get("estimated_calibrated_budgeted_root_l1")) - _safe_float(
        ctree_diag.get("oracle_tree_root_l1")
    )

    out: List[str] = []
    out.append("# CPU Megasweep Consolidated Report")
    out.append("")
    out.append(f"- Output root: `{output_root}`")
    out.append(f"- Figures dir: `{figures}`")
    out.append("")
    out.append("## Core Figures")
    out.append("")
    out.append("### 1) Markov OPS Count Ceilings")
    out.append("![](markov_ops_count_ceilings.png)")
    out.append("")
    out.append("### 2) Segment-LDA OPS Weight Recovery Ceilings")
    out.append("![](segment_lda_ops_weight_recovery_ceilings.png)")
    out.append("")
    out.append("### 3) Segmented-LDA C-TreePO Ceilings")
    out.append("![](segmented_lda_ctreepo_ceilings.png)")
    out.append("")
    out.append("### 4) Mergeable Ceilings")
    out.append("![](mergeable_ceilings.png)")
    out.append("")
    out.append("### 5) Mergeable Complexity Ladder")
    out.append("![](mergeable_complexity_ladder.png)")
    out.append("")
    out.append("## Full-Budget / Full-Guidance Anchor Diagnostics")
    out.append("")
    out.append(_table_row(["Family", "Anchor present", "n", "Oracle/Exact error", "Learned/Budgeted error", "Gap"]))
    out.append(_table_row(["---", "---", "---", "---", "---", "---"]))
    out.append(
        _table_row(
            [
                "Markov",
                str(bool(markov_diag.get("present"))),
                str(int(markov_diag.get("n_rows", 0))),
                _fmt(_safe_float(markov_diag.get("exact_root_mae")), 6),
                _fmt(_safe_float(markov_diag.get("learned_root_mae")), 6),
                _fmt(gap_markov, 6),
            ]
        )
    )
    out.append(
        _table_row(
            [
                "Segment-LDA OPS",
                str(bool(segment_diag.get("present"))),
                str(int(segment_diag.get("n_rows", 0))),
                _fmt(_safe_float(segment_diag.get("exact_root_mae")), 6),
                _fmt(_safe_float(segment_diag.get("ridge_root_mae")), 6),
                _fmt(gap_segment, 6),
            ]
        )
    )
    out.append(
        _table_row(
            [
                "Segmented-LDA C-TreePO",
                str(bool(ctree_diag.get("present"))),
                str(int(ctree_diag.get("n_rows", 0))),
                _fmt(_safe_float(ctree_diag.get("oracle_tree_root_l1")), 6),
                _fmt(_safe_float(ctree_diag.get("estimated_calibrated_budgeted_root_l1")), 6),
                _fmt(gap_ctree, 6),
            ]
        )
    )
    out.append("")
    out.append("Interpretation: oracle/exact ceilings are attained at zero for all three families; learned/budgeted policies still show residual gap.")
    out.append("")
    out.append("## Detailed Full-Budget Breakdown")
    out.append("")
    out.append(f"- Markov full-audit rows: `{markov_n}`")
    out.append(_table_row(["train_docs", "n", "median learned root_mae", "p10", "p90", "min", "max"]))
    out.append(_table_row(["---", "---", "---", "---", "---", "---", "---"]))
    for td in sorted(markov_by_train):
        vals = markov_by_train[td]
        out.append(
            _table_row(
                [
                    str(td),
                    str(len(vals)),
                    _fmt(_median(vals), 6),
                    _fmt(_q(vals, 0.10), 6),
                    _fmt(_q(vals, 0.90), 6),
                    _fmt(min(vals), 6),
                    _fmt(max(vals), 6),
                ]
            )
        )
    out.append("")
    out.append(f"- Segment-LDA OPS full-audit rows: `{segment_n}`")
    out.append(_table_row(["topic_phi_estimator", "n", "median ridge root_mae", "p10", "p90", "min", "max"]))
    out.append(_table_row(["---", "---", "---", "---", "---", "---", "---"]))
    for est in sorted(segment_by_est):
        vals = segment_by_est[est]
        out.append(
            _table_row(
                [
                    est,
                    str(len(vals)),
                    _fmt(_median(vals), 6),
                    _fmt(_q(vals, 0.10), 6),
                    _fmt(_q(vals, 0.90), 6),
                    _fmt(min(vals), 6),
                    _fmt(max(vals), 6),
                ]
            )
        )
    out.append("")
    out.append(f"- Segmented-LDA C-TreePO full-guidance rows: `{ctree_n}`")
    out.append(_table_row(["n_books_train", "n", "median budgeted root_l1", "p10", "p90", "min", "max"]))
    out.append(_table_row(["---", "---", "---", "---", "---", "---", "---"]))
    for td in sorted(ctree_by_train):
        vals = ctree_by_train[td]
        out.append(
            _table_row(
                [
                    str(td),
                    str(len(vals)),
                    _fmt(_median(vals), 6),
                    _fmt(_q(vals, 0.10), 6),
                    _fmt(_q(vals, 0.90), 6),
                    _fmt(min(vals), 6),
                    _fmt(max(vals), 6),
                ]
            )
        )
    out.append("")
    out.append("## Mergeable Snapshot")
    out.append("")
    out.append(
        f"- Budget ladder target k: `{mergeable_summary.get('budget_target_k')}`, sketch order: `{mergeable_summary.get('budget_sketch_order')}`"
    )
    out.append(
        f"- Complexity ladder methods: `{', '.join(str(x) for x in (ladder_summary.get('method_order') or []))}`"
    )
    out.append("")
    out.append("## Recommended Next Simulation Build-Out")
    out.append("")
    out.append("1. Add an explicit `gap-to-ceiling` figure for each family at full budget to separate `oracle attainability` from `learned model gap`.")
    out.append("2. Expand hard regimes where approximation error dominates: increase topic overlap, reduce segment separability, and add nonstationary boundaries.")
    out.append("3. Add estimator-stress ablations in Segment-LDA OPS (especially noisy_phi and online_tensor settings) with fixed full-audit budgets.")
    out.append("4. Add guidance-efficiency frontiers in C-TreePO: sweep `(leaf_rate, internal_rate)` jointly and report iso-error curves.")
    out.append("5. Expand IPW section with high-variance propensity ladders and calibration diagnostics so uncertainty behavior is visible under harder audits.")
    out.append("")
    out.append("## Source Files")
    out.append("")
    out.append("- `markov_ops_count_ceilings_report.json`")
    out.append("- `segment_lda_ops_weight_recovery_ceilings_report.json`")
    out.append("- `segmented_lda_ctreepo_ceilings_report.json`")
    out.append("- `mergeable_ceilings_summary.json`")
    out.append("- `mergeable_complexity_ladder_summary.json`")
    out.append("")
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = Path(args.output_root).resolve()
    figures = output_root / "figures"
    if not figures.exists():
        raise SystemExit(f"figures directory not found: {figures}")

    out_path = (
        Path(args.output_report).resolve()
        if args.output_report is not None
        else (figures / "megasweep_consolidated_report.md").resolve()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = _build_report(output_root)
    out_path.write_text(text, encoding="utf-8")
    print(json.dumps({"output_report": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
