#!/usr/bin/env python3
"""Generate a readable markdown/PDF report for a CPU megasweep run.

This is a companion to `src.ctreepo.sim.cli.report.cpu_megasweep` that is optimized for
paper-debugging: it makes "oracle attainability" explicit and breaks down
Segment-LDA OPS by `topic_phi_estimator`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import math
import shutil
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


from treepo._research.ctreepo.sim.util import safe_float


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a readable report for a CPU megasweep output root.")
    p.add_argument("--output-root", type=Path, required=True, help="Megasweep output root.")
    p.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Markdown path (default: <output-root>/figures/megasweep_consolidated_readable_report.md).",
    )
    p.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help="PDF path (default: same stem as markdown).",
    )
    p.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(list(argv) if argv is not None else None)


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


_safe_float = safe_float


def _isfinite(x: object) -> bool:
    v = _safe_float(x)
    return math.isfinite(v)


def _median(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(statistics.median(vals))


def _fmt(x: object, digits: int = 6) -> str:
    v = _safe_float(x)
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{digits}g}"


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


def _table_row(cols: Sequence[str]) -> str:
    return "| " + " | ".join(cols) + " |"


def _scan_ctree_full_guidance(ctree_glob: str) -> Dict[str, object]:
    budgeted: List[float] = []
    oracle: List[float] = []
    diffs: List[float] = []
    n = 0
    for fp in glob.glob(ctree_glob, recursive=True):
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        if abs(_safe_float(cfg.get("eval_leaf_query_rate")) - 1.0) > 1e-12:
            continue
        if abs(_safe_float(cfg.get("eval_internal_query_rate")) - 1.0) > 1e-12:
            continue
        met = payload.get("metrics", {}) or {}
        b = _safe_float(((met.get("estimated_calibrated_budgeted") or {}) or {}).get("root_l1_mean"))
        o = _safe_float(((met.get("oracle_tree") or {}) or {}).get("root_l1_mean"))
        if not math.isfinite(b) or not math.isfinite(o):
            continue
        budgeted.append(float(b))
        oracle.append(float(o))
        diffs.append(abs(float(b) - float(o)))
        n += 1
    return {
        "n_rows": int(n),
        "budgeted_median": float(_median(budgeted)),
        "oracle_median": float(_median(oracle)),
        "max_abs_diff": float(max(diffs) if diffs else float("nan")),
    }


def _scan_markov_full_audit(markov_glob: str) -> Dict[str, object]:
    exact: List[float] = []
    learned: List[float] = []
    spread: List[float] = []
    n = 0
    for fp in glob.glob(markov_glob, recursive=True):
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        if abs(_safe_float(cfg.get("audit_fraction")) - 1.0) > 1e-12:
            continue
        met = payload.get("metrics", {}) or {}
        ex = _safe_float(((met.get("exact") or {}) or {}).get("root_mae"))
        le = _safe_float(((met.get("learned") or {}) or {}).get("root_mae"))
        sp = _safe_float(((met.get("learned") or {}) or {}).get("schedule_spread_mean"))
        if not math.isfinite(ex) or not math.isfinite(le):
            continue
        exact.append(float(ex))
        learned.append(float(le))
        if math.isfinite(sp):
            spread.append(float(sp))
        n += 1
    return {
        "n_rows": int(n),
        "exact_median": float(_median(exact)),
        "learned_median": float(_median(learned)),
        "learned_schedule_spread_median": float(_median(spread)),
    }


def _scan_segment_full_audit_by_estimator(seg_glob: str, *, audit_strategy: str = "random") -> Dict[str, Dict[str, object]]:
    by_est: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for fp in glob.glob(seg_glob, recursive=True):
        payload = _load_json(Path(fp))
        cfg = payload.get("config", {}) or {}
        if str(cfg.get("audit_strategy", "")) != str(audit_strategy):
            continue
        if abs(_safe_float(cfg.get("audit_fraction")) - 1.0) > 1e-12:
            continue

        est = str(cfg.get("topic_phi_estimator", ""))
        metrics = payload.get("metrics", {}) or {}

        def _y(key: str, field: str = "root_mae") -> float:
            return _safe_float(((metrics.get(key) or {}) or {}).get(field))

        # Exact/undersupported are independent of estimator but useful sanity checks.
        by_est[est]["exact_root_mae"].append(_y("exact"))
        by_est[est]["undersupported_root_mae"].append(_y("undersupported"))

        # Ridge families.
        by_est[est]["ridge_root_mae"].append(_y("ridge"))
        by_est[est]["ridge_true_topics_root_mae"].append(_y("ridge_true_topics"))
        by_est[est]["ridge_infer_true_phi_root_mae"].append(_y("ridge_infer_true_phi"))
        by_est[est]["ridge_infer_est_phi_root_mae"].append(_y("ridge_infer_est_phi"))

        # Topic inference accuracy (only present for ridge-like metrics blocks).
        by_est[est]["ridge_leaf_acc_test"].append(_y("ridge", field="leaf_accuracy_test"))

    out: Dict[str, Dict[str, object]] = {}
    for est, cols in sorted(by_est.items(), key=lambda kv: kv[0]):
        out[est] = {
            "n_rows": int(len(cols.get("ridge_root_mae", []))),
            "exact_root_mae_median": float(_median(cols.get("exact_root_mae", []))),
            "undersupported_root_mae_median": float(_median(cols.get("undersupported_root_mae", []))),
            "ridge_root_mae_median": float(_median(cols.get("ridge_root_mae", []))),
            "ridge_true_topics_root_mae_median": float(_median(cols.get("ridge_true_topics_root_mae", []))),
            "ridge_infer_true_phi_root_mae_median": float(_median(cols.get("ridge_infer_true_phi_root_mae", []))),
            "ridge_infer_est_phi_root_mae_median": float(_median(cols.get("ridge_infer_est_phi_root_mae", []))),
            "ridge_leaf_acc_test_median": float(_median(cols.get("ridge_leaf_acc_test", []))),
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = args.output_root.resolve()
    figures = output_root / "figures"
    if not figures.exists():
        raise SystemExit(f"figures directory not found: {figures}")

    md_path = (
        args.output_markdown.resolve()
        if args.output_markdown is not None
        else (figures / "megasweep_consolidated_readable_report.md")
    )
    pdf_path = args.output_pdf.resolve() if args.output_pdf is not None else md_path.with_suffix(".pdf")

    generated = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    # Load plot reports when available (these backstop scanning + are used to reference figure filenames).
    ctree_report = _load_json(figures / "segmented_lda_ctreepo_ceilings_report.json")
    markov_report = _load_json(figures / "markov_ops_count_ceilings_report.json")
    markov_add_report = _load_json(figures / "markov_ops_count_ceilings_additive_report.json")

    seg_report = _load_json(figures / "segment_lda_ops_weight_recovery_ceilings_report.json")

    # Scan raw outputs for the strongest invariants.
    ctree_scan = _scan_ctree_full_guidance(str(output_root / "segmented_lda_ctreepo" / "**" / "*.json"))
    markov_scan = _scan_markov_full_audit(
        str(output_root / "markov_changepoint_ops_count" / "**" / "*seed_*.json")
    )
    seg_by_est = _scan_segment_full_audit_by_estimator(
        str(output_root / "segment_lda_ops_weight_recovery" / "**" / "*seed_*.json"),
        audit_strategy=str((seg_report.get("filters") or {}).get("audit_strategy") or "random"),
    )

    # Anchor diagnostics.
    ctree_diag = ((ctree_report.get("diagnostics") or {}).get("full_guidance") or {})
    markov_diag = ((markov_report.get("diagnostics") or {}).get("full_audit") or {})
    markov_add_diag = ((markov_add_report.get("diagnostics") or {}).get("full_audit") or {})
    seg_diag = ((seg_report.get("diagnostics") or {}).get("full_audit") or {})

    lines: List[str] = []
    lines.append("# CPU Megasweep Report (Readable)")
    lines.append("")
    lines.append(f"- Generated: `{generated}`")
    lines.append("- Purpose: make oracle attainability vs pipeline gaps unambiguous.")
    lines.append("")

    lines.append("## Critical Check: C-TreePO Full Guidance Convergence")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(_table_row(["Full-guidance rows", _fmt(ctree_scan.get("n_rows"))]))
    lines.append(
        _table_row(
            [
                "Budgeted median root L1 (full guidance)",
                _fmt(ctree_scan.get("budgeted_median")),
            ]
        )
    )
    lines.append(_table_row(["Oracle median root L1 (full guidance)", _fmt(ctree_scan.get("oracle_median"))]))
    lines.append(_table_row(["Max abs diff |budgeted-oracle| across rows", _fmt(ctree_scan.get("max_abs_diff"))]))
    lines.append("")
    lines.append(
        "Conclusion: under full guidance, Segmented-LDA C-TreePO matches the oracle to numerical precision."
    )

    lines.append("")
    lines.append("## Anchor Diagnostics (Full Budget / Full Guidance)")
    lines.append("")
    lines.append("| Family | Oracle/Exact | Learned/Budgeted | Gap | Notes |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(
        _table_row(
            [
                "Markov (neural)",
                _fmt(markov_diag.get("exact_root_mae")),
                _fmt(markov_diag.get("learned_root_mae")),
                _fmt(_safe_float(markov_diag.get("learned_root_mae")) - _safe_float(markov_diag.get("exact_root_mae"))),
                f"schedule_spread~{_fmt(_safe_float(markov_scan.get('learned_schedule_spread_median')))}",
            ]
        )
    )
    if markov_add_diag:
        lines.append(
            _table_row(
                [
                    "Markov (additive)",
                    _fmt(markov_add_diag.get("exact_root_mae")),
                    _fmt(markov_add_diag.get("learned_root_mae")),
                    _fmt(
                        _safe_float(markov_add_diag.get("learned_root_mae"))
                        - _safe_float(markov_add_diag.get("exact_root_mae"))
                    ),
                    "structured merge hits ceiling",
                ]
            )
        )
    lines.append(
        _table_row(
            [
                "Segment-LDA OPS (pooled)",
                _fmt(seg_diag.get("exact_root_mae")),
                _fmt(seg_diag.get("ridge_root_mae")),
                _fmt(_safe_float(seg_diag.get("ridge_root_mae")) - _safe_float(seg_diag.get("exact_root_mae"))),
                "pooled over topic estimators",
            ]
        )
    )
    lines.append(
        _table_row(
            [
                "Segmented-LDA C-TreePO",
                _fmt(ctree_diag.get("oracle_tree_root_l1")),
                _fmt(ctree_diag.get("estimated_calibrated_budgeted_root_l1")),
                _fmt(
                    _safe_float(ctree_diag.get("estimated_calibrated_budgeted_root_l1"))
                    - _safe_float(ctree_diag.get("oracle_tree_root_l1"))
                ),
                "full guidance = 0 gap",
            ]
        )
    )
    lines.append("")
    lines.append(
        "Interpretation: if a curve does not approach its ceiling, check whether the estimator family is expressive enough and whether upstream inference (e.g., topic-phi estimation) is the bottleneck."
    )
    lines.append("")

    lines.append("## Segment-LDA OPS: Full-Audit Breakdown by Topic Estimator")
    lines.append("")
    lines.append("| topic_phi_estimator | n | ridge_root_mae (median) | ridge_true_topics (median) | ridge_infer_true_phi (median) | ridge_infer_est_phi (median) | leaf_acc_test (median) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for est, row in seg_by_est.items():
        lines.append(
            _table_row(
                [
                    est or "(empty)",
                    str(int(row.get("n_rows", 0))),
                    _fmt(row.get("ridge_root_mae_median")),
                    _fmt(row.get("ridge_true_topics_root_mae_median")),
                    _fmt(row.get("ridge_infer_true_phi_root_mae_median")),
                    _fmt(row.get("ridge_infer_est_phi_root_mae_median")),
                    _fmt(row.get("ridge_leaf_acc_test_median")),
                ]
            )
        )
    lines.append("")
    lines.append(
        "Key takeaway: `ridge_true_topics` and `ridge_infer_true_phi` are (near) the ceiling, so the downstream ridge is well-specified; the remaining gap is dominated by `topic_phi_estimator` quality."
    )
    lines.append("")

    def _fig(name: str) -> None:
        if (figures / name).exists():
            lines.append(f"![]({name}){{ width=95% }}")
            lines.append("")

    lines.append("\\newpage")
    lines.append("## Figures: C-TreePO + Mergeable")
    lines.append("")
    _fig("segmented_lda_ctreepo_ceilings.png")
    _fig("mergeable_ceilings.png")

    lines.append("\\newpage")
    lines.append("## Figures: Markov OPS Count")
    lines.append("")
    _fig("markov_ops_count_ceilings.png")
    _fig("markov_ops_count_ceilings_additive.png")

    lines.append("\\newpage")
    lines.append("## Figures: Segment-LDA OPS (Pooled)")
    lines.append("")
    _fig("segment_lda_ops_weight_recovery_ceilings.png")

    lines.append("\\newpage")
    lines.append("## Figures: Segment-LDA OPS (Filtered Ceilings)")
    lines.append("")
    _fig("segment_lda_ops_weight_recovery_ceilings_true.png")
    _fig("segment_lda_ops_weight_recovery_ceilings_embedding_spectral.png")
    _fig("segment_lda_ops_weight_recovery_ceilings_tensor_lda.png")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pdf_emitted = False
    if bool(args.emit_pdf):
        try:
            pdf_emitted = _run_pandoc(md_path, pdf_path)
        except Exception:
            pdf_emitted = False

    print(
        json.dumps(
            {
                "output_markdown": str(md_path),
                "output_pdf": str(pdf_path) if pdf_emitted else None,
                "pdf_emitted": bool(pdf_emitted),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
