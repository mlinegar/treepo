#!/usr/bin/env python3
"""Generate a markdown/PDF report for identifiable-only simulation sweeps."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import shutil
import subprocess
from typing import Dict, List, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build report for identifiable-zero suite outputs.")
    p.add_argument("--output-root", type=Path, required=True, help="Suite output root.")
    p.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Markdown report path (default: <output-root>/figures/identifiable_zero_suite_report.md).",
    )
    p.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help="PDF report path (default: same stem as markdown).",
    )
    p.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(list(argv) if argv is not None else None)


def _load_json(path: Path | None) -> Dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x: object) -> str:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return "nan"
    if not (v == v):
        return "nan"
    return f"{v:.6g}"


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


def _diag_row(name: str, diag: Dict, key_learned: str, key_exact: str) -> str:
    exact_v = float(diag.get(key_exact, float("nan")))
    learned_v = float(diag.get(key_learned, float("nan")))
    gap = learned_v - exact_v if exact_v == exact_v and learned_v == learned_v else float("nan")
    return f"| {name} | {_fmt(diag.get('n_rows'))} | {_fmt(exact_v)} | {_fmt(learned_v)} | {_fmt(gap)} |"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    figures = output_root / "figures"
    if not figures.exists():
        raise SystemExit(f"figures directory not found: {figures}")

    md_path = (
        args.output_markdown.resolve()
        if args.output_markdown is not None
        else (figures / "identifiable_zero_suite_report.md")
    )
    pdf_path = args.output_pdf.resolve() if args.output_pdf is not None else md_path.with_suffix(".pdf")

    seg_true = _load_json(figures / "segment_lda_ops_weight_recovery_ceilings_true_report.json")
    seg_focus = _load_json(figures / "segment_lda_oracle_gap_focus_true_report.json")
    seg_emb = _load_json(figures / "segment_lda_ops_weight_recovery_ceilings_embedding_spectral_report.json")
    ctree = _load_json(figures / "segmented_lda_ctreepo_ceilings_report.json")
    frontier = _load_json(figures / "ctreepo_guidance_frontier_report.json")
    frontier_focus_path = None
    focus_jsons = sorted(glob.glob(str(figures / "ctreepo_guidance_frontier_focus_train*_report.json")))
    if focus_jsons:
        frontier_focus_path = Path(focus_jsons[-1])
    frontier_focus = _load_json(frontier_focus_path) if frontier_focus_path is not None else {}
    gap = _load_json(figures / "full_budget_gap_suite_report.json")

    seg_true_diag = ((seg_true.get("diagnostics") or {}).get("full_audit") or {})
    seg_emb_diag = ((seg_emb.get("diagnostics") or {}).get("full_audit") or {})
    ctree_diag = ((ctree.get("diagnostics") or {}).get("full_guidance") or {})

    lines: List[str] = []
    lines.append("# Identifiable-Zero Simulation Suite Report")
    lines.append("")
    lines.append(f"- Output root: `{output_root}`")
    lines.append(f"- Figures dir: `{figures}`")
    lines.append("")
    lines.append("## Anchor Diagnostics (Full Budget / Full Guidance)")
    lines.append("")
    lines.append("| Family | n rows | Oracle/Exact | Learned/Budgeted | Gap |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(_diag_row("Segment-LDA OPS (phi=true)", seg_true_diag, "ridge_root_mae", "exact_root_mae"))
    if seg_emb_diag:
        lines.append(
            _diag_row(
                "Segment-LDA OPS (phi=embedding_spectral)",
                seg_emb_diag,
                "ridge_root_mae",
                "exact_root_mae",
            )
        )
    lines.append(
        _diag_row(
            "Segmented-LDA C-TreePO",
            ctree_diag,
            "estimated_calibrated_budgeted_root_l1",
            "oracle_tree_root_l1",
        )
    )
    lines.append("")
    lines.append("## Segment-LDA Focus")
    lines.append("")
    if (figures / "segment_lda_oracle_gap_focus_true.png").exists():
        lines.append("![](segment_lda_oracle_gap_focus_true.png)")
        lines.append("")
    lines.append("\\\\newpage")
    lines.append("")
    if (figures / "segment_lda_ops_weight_recovery_grid_true.png").exists():
        lines.append("![](segment_lda_ops_weight_recovery_grid_true.png)")
        lines.append("")
    if (figures / "segment_lda_ops_weight_recovery_ceilings_true.png").exists():
        lines.append("![](segment_lda_ops_weight_recovery_ceilings_true.png)")
        lines.append("")
    if (figures / "segment_lda_ops_weight_recovery_ceilings_embedding_spectral.png").exists():
        lines.append("![](segment_lda_ops_weight_recovery_ceilings_embedding_spectral.png)")
        lines.append("")
    lines.append("## C-TreePO Focus")
    lines.append("")
    if frontier_focus_path is not None:
        focus_png = frontier_focus_path.name.replace("_report.json", ".png")
        if (figures / focus_png).exists():
            lines.append(f"![]({focus_png})")
            lines.append("")
    if (figures / "ctreepo_guidance_frontier_focus_train1024.png").exists() and frontier_focus_path is None:
        lines.append("![](ctreepo_guidance_frontier_focus_train1024.png)")
        lines.append("")
    if (figures / "segmented_lda_ctreepo_ceilings.png").exists():
        lines.append("![](segmented_lda_ctreepo_ceilings.png)")
        lines.append("")
    if (figures / "segmented_lda_ctreepo_phase.png").exists():
        lines.append("![](segmented_lda_ctreepo_phase.png)")
        lines.append("")
    if (figures / "ctreepo_guidance_frontier.png").exists():
        lines.append("![](ctreepo_guidance_frontier.png)")
        lines.append("")
    lines.append("## Cross-Family Gap View")
    lines.append("")
    lines.append("\\\\newpage")
    lines.append("")
    if (figures / "full_budget_gap_suite.png").exists():
        lines.append("![](full_budget_gap_suite.png)")
        lines.append("")
    lines.append("## Quick Notes")
    lines.append("")
    lines.append(
        f"- Segment true full-audit ridge root MAE: `{_fmt(seg_true_diag.get('ridge_root_mae'))}`, exact: `{_fmt(seg_true_diag.get('exact_root_mae'))}`"
    )
    if seg_focus:
        try:
            lam0 = sorted(float(x) for x in (seg_focus.get("lambdas") or []))[0]
            lamk = f"{lam0:g}"
            fa = (seg_focus.get("full_audit_by_lambda_train_docs") or {}).get(lamk, {}) or {}
            if fa:
                td_max = max(int(k) for k in fa.keys())
                td_row = fa.get(str(td_max), {}) or {}
                lines.append(
                    f"- Segment focus (lambda={lamk}, train_docs={td_max}) full-audit gap-to-exact: `{_fmt(td_row.get('gap_to_exact'))}`"
                )
        except Exception:
            pass
    if seg_emb_diag:
        lines.append(
            f"- Segment embedding full-audit ridge root MAE: `{_fmt(seg_emb_diag.get('ridge_root_mae'))}`, exact: `{_fmt(seg_emb_diag.get('exact_root_mae'))}`"
        )
    lines.append(
        f"- C-TreePO full-guidance budgeted root L1: `{_fmt(ctree_diag.get('estimated_calibrated_budgeted_root_l1'))}`, oracle: `{_fmt(ctree_diag.get('oracle_tree_root_l1'))}`"
    )
    lines.append(f"- C-TreePO frontier rows after filters: `{_fmt(frontier.get('n_rows_after_filters'))}`")
    if frontier_focus:
        lines.append(f"- C-TreePO focus frontier rows: `{_fmt(frontier_focus.get('n_rows_after_filters'))}`")
    if gap:
        lines.append(
            f"- Full-budget suite includes families: `{'/'.join(k for k in ['markov', 'segment', 'ctree'] if gap.get(k))}`"
        )
    lines.append("")

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
