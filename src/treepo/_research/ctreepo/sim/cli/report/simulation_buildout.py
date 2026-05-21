#!/usr/bin/env python3
"""Generate a single markdown (and optional PDF) report for the 1-5 simulation buildout."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build consolidated report for simulation buildout outputs.")
    p.add_argument("--output-root", type=Path, required=True, help="Buildout output root.")
    p.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Markdown report path (default: <output-root>/figures/simulation_buildout_report.md).",
    )
    p.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help="PDF report path (default: same stem as markdown).",
    )
    p.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(list(argv) if argv is not None else None)


def _load_json(path: Path) -> Dict:
    if not path.exists():
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
    if shutil.which("pandoc") is None:
        return False
    if shutil.which("pdflatex") is None:
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = args.output_root.resolve()
    figures = output_root / "figures"
    if not figures.exists():
        raise SystemExit(f"figures directory not found: {figures}")

    md_path = args.output_markdown.resolve() if args.output_markdown else (figures / "simulation_buildout_report.md")
    pdf_path = args.output_pdf.resolve() if args.output_pdf else md_path.with_suffix(".pdf")

    gap = _load_json(figures / "full_budget_gap_suite_report.json")
    hard = _load_json(figures / "hard_regime_summary_report.json")
    est = _load_json(figures / "segment_lda_estimator_stress_report.json")
    frontier = _load_json(figures / "ctreepo_guidance_frontier_report.json")
    ipw_v = _load_json(figures / "ipw_propensity_diagnostics_violation_report.json")

    lines: List[str] = []
    lines.append("# Simulation Buildout Report (Items 1-5)")
    lines.append("")
    lines.append(f"- Output root: `{output_root}`")
    lines.append(f"- Figures: `{figures}`")
    lines.append("")
    lines.append("## Item 1: Full-Budget Gap-to-Ceiling")
    lines.append("![](full_budget_gap_suite.png)")
    lines.append("")
    lines.append("| Family | Oracle/Exact | Learned/Budgeted |")
    lines.append("| --- | --- | --- |")
    if gap.get("markov"):
        mk = gap["markov"]
        lines.append(
            f"| Markov | {_fmt((mk.get('exact') or [float('nan')])[-1])} | {_fmt((mk.get('learned') or [float('nan')])[-1])} |"
        )
    if gap.get("segment"):
        sg = gap["segment"]
        seg_exact = (sg.get("exact") or [float("nan")])[0] if isinstance(sg.get("exact"), list) else float("nan")
        seg_ridge = (sg.get("ridge") or [float("nan")])[0] if isinstance(sg.get("ridge"), list) else float("nan")
        lines.append(f"| Segment-LDA OPS | {_fmt(seg_exact)} | {_fmt(seg_ridge)} |")
    if gap.get("ctree"):
        ct = gap["ctree"]
        lines.append(
            f"| Segmented-LDA C-TreePO | {_fmt((ct.get('oracle_tree') or [float('nan')])[-1])} | {_fmt((ct.get('budgeted') or [float('nan')])[-1])} |"
        )
    lines.append("")
    lines.append("## Item 2: Hard-Regime Sweeps")
    lines.append("![](hard_regime_summary.png)")
    lines.append("")
    lines.append("## Item 3: Segment-LDA Estimator Stress")
    lines.append("![](segment_lda_estimator_stress.png)")
    lines.append("")
    lines.append("## Item 4: C-TreePO Guidance Frontier")
    lines.append("![](ctreepo_guidance_frontier.png)")
    lines.append("")
    lines.append("## Item 5: Expanded IPW Diagnostics")
    lines.append("![](ipw_stress_ladder_violation.png)")
    lines.append("")
    lines.append("![](ipw_stress_ladder_preference.png)")
    lines.append("")
    lines.append("![](ipw_propensity_diagnostics_violation.png)")
    lines.append("")
    lines.append("![](ipw_propensity_diagnostics_preference.png)")
    lines.append("")
    lines.append("## Quick Diagnostics")
    lines.append("")
    lines.append(f"- Hard-regime families with rows: `{', '.join(sorted((hard.keys() if isinstance(hard, dict) else [])))}`")
    lines.append(f"- Estimator stress estimators: `{', '.join(est.get('estimators', []))}`")
    lines.append(f"- Guidance frontier rows: `{frontier.get('n_rows_after_filters', 'nan')}`")
    lines.append(f"- IPW cases (violation view): `{', '.join(ipw_v.get('cases', []))}`")
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
