#!/usr/bin/env python3
"""Generate a paper-facing DTM-LDA report from the LDA-only learnability layout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a DTM-LDA report for Identifiable-Zero sweeps.")
    p.add_argument("--output-root", type=Path, required=True, help="DTM-LDA sweep output root.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <output-root>/figures/dtm_lda).",
    )
    p.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = args.output_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else (output_root / "figures" / "dtm_lda")
    out_dir.mkdir(parents=True, exist_ok=True)

    from treepo._research.ctreepo.sim.cli.report.identifiable_zero_learnability import main as _learnability_report_main

    base_argv = [
        "--output-root",
        str(output_root),
        "--out-dir",
        str(out_dir),
        "--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf",
    ]
    rc = int(_learnability_report_main(base_argv))
    if rc != 0:
        return rc

    md_src = out_dir / "identifiable_zero_learnability_latest.md"
    pdf_src = out_dir / "identifiable_zero_learnability_latest.pdf"
    diag_src = out_dir / "identifiable_zero_learnability_latest_diagnostics.json"

    md_dst = out_dir / "identifiable_zero_dtm_lda_latest.md"
    pdf_dst = out_dir / "identifiable_zero_dtm_lda_latest.pdf"
    diag_dst = out_dir / "identifiable_zero_dtm_lda_latest_diagnostics.json"

    md_text = md_src.read_text(encoding="utf-8")
    md_text = md_text.replace(
        "title: Identifiable-Zero Learnability Benchmarks (v1)",
        "title: Identifiable-Zero DTM-LDA Benchmarks (v1)",
        1,
    )
    md_text = md_text.replace(
        "**Output root:**",
        "**DTM-LDA output root:**",
        1,
    )
    md_text = md_text.replace(
        "identifiable_zero_learnability_latest_diagnostics.json",
        "identifiable_zero_dtm_lda_latest_diagnostics.json",
    )
    md_text = md_text.replace(
        "**Markov (OPS-count)**\n"
        "- Varied: `train_docs` (more training documents), `audit_fraction` (more internal-node oracle labels).\n"
        "- Fixed: held-out `test_docs` per run; generated with seed offset so the test set is stable across `train_docs`.\n"
        "- Metric: `root_mae` on held-out test docs at decision-time oracle visibility `q_infer=0` (no inference guidance).\n",
        "**Markov (OPS-count)**\n"
        "- Not included in this DTM-LDA robustness suite.\n",
        1,
    )
    md_text = md_text.replace(
        "Only quote Markov-vs-LDA cross-family comparisons when the setup check below says the train grids, label-rate grids, and held-out test sizes match.",
        "Treat this report as LDA-only unless a separate matched Markov control is provided alongside it.",
        1,
    )
    md_dst.write_text(md_text, encoding="utf-8")

    shutil.copy2(diag_src, diag_dst)
    if bool(args.emit_pdf) and pdf_src.exists():
        shutil.copy2(pdf_src, pdf_dst)

    # The reused learnability reporter writes its default filenames first.
    # Remove those compatibility leftovers so the DTM report directory only
    # exposes DTM-LDA-facing artifact names.
    for stale in (md_src, diag_src, pdf_src):
        if stale.exists():
            stale.unlink()

    print(f"wrote_markdown | {md_dst}")
    print(f"wrote_diagnostics | {diag_dst}")
    if bool(args.emit_pdf):
        print(f"wrote_pdf | {pdf_dst} | ok={pdf_dst.exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
