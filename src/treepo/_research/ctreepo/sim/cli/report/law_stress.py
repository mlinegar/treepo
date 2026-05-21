#!/usr/bin/env python3
"""Unified law-stress report for any simulation family (Markov, LDA, ...).

Canonical suite usage:
    python -m src.ctreepo.cli sim suite law-stress report --family markov --output-root outputs/...
    python -m src.ctreepo.cli sim suite law-stress report --family lda    --output-root outputs/...

Produces a markdown report, JSON summary, CSV of assessed/aggregated rows,
publication-ready figures, and a PDF — all in the output directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matplotlib.backends.backend_pdf import PdfPages

from treepo._research.ctreepo.sim.core.law_stress_common import (
    DEFAULT_LAW_GAIN_THRESHOLD,
    DEFAULT_ROOT_RATIO_LIMIT,
    DEFAULT_SPREAD_GAIN_THRESHOLD,
)
from treepo._research.ctreepo.sim.local_law_report_common import (
    build_local_law_report_core,
    load_local_law_runs,
    render_local_law_report_markdown,
    write_local_law_report_core_pages,
)
from treepo._research.ctreepo.sim.report.aggregation import (
    aggregate_law_stress,
    build_downstream_table,
)
from treepo._research.ctreepo.sim.report.data_loading import (
    assess_law_stress_rows,
    load_law_stress_records,
)
from treepo._research.ctreepo.sim.report.family_config import FamilyReportConfig, resolve_family
from treepo._research.ctreepo.sim.report.pdf_utils import (
    safe_sem,
    write_csv,
    write_image_page,
    write_text_page,
)
from treepo._research.ctreepo.sim.report.plots import (
    plot_ablation_bar_chart,
    plot_exact_family_counterexamples,
    plot_heatmap,
    plot_mechanism_pareto,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified law-stress report.")
    p.add_argument("--family", type=str, required=True, help="markov | lda")
    p.add_argument("--input-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--pdf-path", type=str, default=None)
    p.add_argument("--expected-run-count", type=int, default=None)
    return p.parse_args(list(argv) if argv is not None else None)


# ── claim readout ────────────────────────────────────────────────────────


def _claim_readout(
    unified_core: Dict,
    *,
    family: FamilyReportConfig,
) -> Dict:
    rows = [
        dict(row)
        for row in list(unified_core.get("law_stress_summary", []) or [])
        if str(dict(row).get("family", "")) == family.family
    ]
    main_package = family.expected_main_package
    main_row = next((r for r in rows if str(r.get("law_package", "")) == main_package), None)
    if main_row is None:
        main_package = family.fallback_main_package
        main_row = next((r for r in rows if str(r.get("law_package", "")) == main_package), None)

    ablation_rows = [r for r in rows if str(r.get("law_package", "")) != main_package]
    strongest_ablation = None
    if ablation_rows:
        strongest_ablation = max(
            ablation_rows,
            key=lambda r: (
                float(r.get("primary_pass_rate", 0.0)),
                float(r.get("mean_primary_gain", float("-inf"))),
                float(r.get("mean_laws_improved", float("-inf"))),
            ),
        )

    status = "unknown"
    if main_row is not None:
        if float(main_row.get("primary_pass_rate", 0.0)) > 0.0 and float(main_row.get("mean_primary_gain", 0.0)) > 0.0:
            status = "passes_downstream"
        else:
            status = "fails_downstream"

    note = "No expected full-package claim row was found in the current report."
    if main_row is not None and strongest_ablation is not None:
        note = (
            f"The expected claim row is `{main_package}`. "
            f"`{strongest_ablation.get('law_package', '')}` is the strongest ablation on downstream metrics in this sweep, "
            "but it remains diagnostic-only and does not replace the claim row."
        )
    elif main_row is not None:
        note = (
            f"The expected claim row is `{main_package}`. No ablation rows are present in this report, "
            "so the mechanism comparison is not available here."
        )

    return {
        "main_package": str(main_package),
        "status": status,
        "main_row": main_row,
        "strongest_ablation_row": strongest_ablation,
        "note": note,
    }


# ── narrative ────────────────────────────────────────────────────────────


def _build_narrative(
    *,
    family: FamilyReportConfig,
    aggregated_rows: Sequence[dict],
    claim: dict,
    boundary_candidates: Sequence[dict],
    n_learned: int,
    n_exact: int,
    n_total: int,
) -> List[str]:
    """Build the bullet-point narrative section."""
    primary = family.primary_metric_label
    ablation_pkg_stats: Dict[str, dict] = {}
    for pkg in family.valid_law_packages:
        pkg_rows = [r for r in aggregated_rows if str(r.get("law_package", "")) == pkg]
        if pkg_rows:
            ablation_pkg_stats[pkg] = {
                "prim_gain": float(fmean(1.0 - float(r["root_ratio"]) for r in pkg_rows)),
                "root_ratio": float(fmean(float(r["root_ratio"]) for r in pkg_rows)),
            }

    narrative: List[str] = [
        f"**Primary metric**: held-out {primary} compared to the root-only baseline (no local laws). "
        f"`PrimGain = 1 − {primary} ratio`; positive = lower error = better.",
        "",
    ]

    # Ablation key finding (auto-detect strongest and weakest packages)
    non_baseline = {k: v for k, v in ablation_pkg_stats.items() if k != "root_only"}
    if len(non_baseline) >= 2:
        best_pkg = max(non_baseline, key=lambda k: non_baseline[k]["prim_gain"])
        main_pkg = family.expected_main_package
        main_stat = ablation_pkg_stats.get(main_pkg)
        best_stat = non_baseline[best_pkg]
        if main_stat and best_pkg != main_pkg:
            narrative.extend([
                f"**Key finding**: `{best_pkg}` is the strongest ablation on downstream {primary} "
                f"(PrimGain = {100.0 * best_stat['prim_gain']:+.1f}%). The full bundle (`{main_pkg}`) "
                f"has PrimGain = {100.0 * main_stat['prim_gain']:+.1f}%.",
                "",
            ])

    narrative.extend([
        f"This report uses direct per-law metrics: C1 leaf preservation, C2 re-summary idempotence, C3 merge preservation.",
        "Schedule consistency is reported separately as a proxy diagnostic.",
        f"For regularisation strength tuning, see the learnability report.",
        f"Rows loaded: {n_total} total, {n_learned} learned, {n_exact} exact-family.",
    ])

    claim_main = dict(claim.get("main_row", {}) or {})
    claim_ablation = dict(claim.get("strongest_ablation_row", {}) or {})
    if claim_main:
        narrative.append(
            f"Claim row status: `{claim.get('main_package', '')}` has "
            f"Prim%={100.0 * float(claim_main.get('primary_pass_rate', 0.0)):.1f}%, "
            f"PrimGain={100.0 * float(claim_main.get('mean_primary_gain', 0.0)):.1f}%, "
            f"C1={100.0 * float(claim_main.get('c1_pass_rate', 0.0)):.0f}%, "
            f"C2={100.0 * float(claim_main.get('c2_pass_rate', 0.0)):.0f}%, "
            f"C3={100.0 * float(claim_main.get('c3_pass_rate', 0.0)):.0f}%."
        )
    if claim_ablation:
        narrative.append(
            f"Strongest ablation: `{claim_ablation.get('law_package', '')}` has "
            f"Prim%={100.0 * float(claim_ablation.get('primary_pass_rate', 0.0)):.1f}%, "
            f"PrimGain={100.0 * float(claim_ablation.get('mean_primary_gain', 0.0)):.1f}%. "
            "This is a mechanism diagnostic, not the main claim."
        )
    if claim.get("note"):
        narrative.append(str(claim.get("note")))
    if boundary_candidates:
        top = boundary_candidates[0]
        row_field = family.heatmap_row_field
        col_field = family.heatmap_col_field
        narrative.append(
            f"Closest transition-boundary cell: "
            f"{row_field}={top.get(row_field, '?')}, "
            f"{col_field}={top.get(col_field, '?')}, "
            f"bundle_success_rate={float(top['bundle_full_success_rate']):.2f}, "
            f"bundle_margin_mean={float(top['bundle_margin_mean']):.3f}."
        )
    return narrative


# ── main ─────────────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    family = resolve_family(args.family)
    input_root = Path(args.input_root)
    title = args.title or f"{family.display_name} Local-Law Stress Report"
    output_dir = Path(args.output_dir) if args.output_dir else (input_root / "law_stress_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ────────────────────────────────────────────────────
    learned_records, exact_records = load_law_stress_records(input_root, family)
    if not learned_records:
        raise SystemExit(f"No learned records found under {input_root}")

    protocol_runs = load_local_law_runs(input_root)
    unified_core = build_local_law_report_core(protocol_runs)

    # ── assess & aggregate ───────────────────────────────────────────
    assessed_rows = assess_law_stress_rows(learned_records)

    # Group keys: family-specific config fields + law_package
    # Use all config fields present in the records, excluding metric/path fields
    _metric_fields = {
        "path", "run_kind", "exact_family", "seed",
        "test_c1", "test_c2", "test_c3", "test_spread", "test_primary", "test_bundle_score",
        "val_c1", "val_c2", "val_c3", "val_spread", "val_primary", "val_bundle_score",
        "test_c2_r4", "test_resummary_root_drift_r4",
        "baseline_package", "baseline_test_c1", "baseline_test_c2", "baseline_test_c3",
        "baseline_test_spread", "baseline_test_primary", "baseline_test_bundle_score",
        "failure_reason",
    }
    # Also exclude all assessment fields
    _assessment_fields = {
        "primary_pass", "primary_gain_frac", "primary_margin",
        "bundle_status", "bundle_full_success",
        "c1_pass", "c2_pass", "c3_pass", "root_pass", "spread_pass",
        "laws_improved", "all_laws_pass",
        "c1_gain_frac", "c2_gain_frac", "c3_gain_frac", "spread_gain_frac", "root_ratio",
        "c1_margin", "c2_margin", "c3_margin", "spread_margin", "root_margin",
    }
    exclude = _metric_fields | _assessment_fields
    if assessed_rows:
        group_keys = [k for k in assessed_rows[0].keys() if k not in exclude]
    else:
        group_keys = ["law_package"]

    aggregated_rows = aggregate_law_stress(assessed_rows, group_keys=group_keys)

    # Filter main package rows for heatmaps
    main_pkg = family.expected_main_package
    main_rows = [r for r in aggregated_rows if str(r.get("law_package", "")) == main_pkg]
    if not main_rows:
        main_pkg = family.fallback_main_package
        main_rows = [r for r in aggregated_rows if str(r.get("law_package", "")) == main_pkg]

    # ── figures ──────────────────────────────────────────────────────
    figure_paths: List[str] = []
    figure_titles: Dict[str, str] = {}

    # Ablation bar chart (always first)
    ablation_fig = output_dir / "ablation_downstream.png"
    plot_ablation_bar_chart(aggregated_rows, family=family, output_path=ablation_fig)
    if ablation_fig.exists():
        figure_paths.append(str(ablation_fig))
        figure_titles[str(ablation_fig)] = "Ablation: downstream gain by law package"

    # Heatmaps on main-package rows
    if main_rows:
        for key, htitle, cmap, fmt in (
            ("c1_pass_rate", "C1 pass rate", "YlGnBu", ".2f"),
            ("c2_pass_rate", "C2 pass rate", "YlGnBu", ".2f"),
            ("c3_pass_rate", "C3 pass rate", "YlGnBu", ".2f"),
            ("bundle_full_success_rate", "Bundle full-success rate", "YlGnBu", ".2f"),
            ("root_ratio", f"{family.primary_metric_label} ratio vs matched baseline", "magma_r", ".2f"),
        ):
            path = output_dir / f"{key}.png"
            plot_heatmap(
                main_rows,
                row_field=family.heatmap_row_field,
                col_field=family.heatmap_col_field,
                value_key=key,
                row_label=family.heatmap_row_label,
                col_label=family.heatmap_col_label,
                title=htitle,
                output_path=path,
                cmap=cmap,
                fmt=fmt,
            )
            figure_paths.append(str(path))
            figure_titles[str(path)] = htitle

    # Mechanism pareto
    if assessed_rows:
        mech_fig = output_dir / "mechanism_pareto.png"
        plot_mechanism_pareto(assessed_rows, family=family, output_path=mech_fig)
        figure_paths.append(str(mech_fig))
        figure_titles[str(mech_fig)] = f"Mechanism Pareto: {family.primary_metric_label} vs C1+C2+C3"

    # Exact-family counterexamples
    exact_with_family = [r for r in exact_records if str(r.get("exact_family", ""))]
    if exact_with_family:
        exact_fig = output_dir / "exact_family_counterexamples.png"
        plot_exact_family_counterexamples(exact_with_family, output_path=exact_fig)
        figure_paths.append(str(exact_fig))
        figure_titles[str(exact_fig)] = "Exact-family counterexamples"

    # ── boundary candidates ──────────────────────────────────────────
    boundary_candidates = sorted(
        [r for r in main_rows if str(r.get("law_package", "")) in {family.expected_main_package, family.fallback_main_package}],
        key=lambda r: (
            abs(float(r["bundle_full_success_rate"]) - 0.5),
            abs(float(r["bundle_margin_mean"])),
        ),
    )

    # ── claim readout ────────────────────────────────────────────────
    claim = _claim_readout(unified_core, family=family)

    # ── narrative ────────────────────────────────────────────────────
    narrative = _build_narrative(
        family=family,
        aggregated_rows=aggregated_rows,
        claim=claim,
        boundary_candidates=boundary_candidates,
        n_learned=len(learned_records),
        n_exact=len(exact_records),
        n_total=len(learned_records) + len(exact_records),
    )

    # ── JSON summary ─────────────────────────────────────────────────
    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "family": family.family,
        "input_root": str(input_root),
        "raw_row_count": len(learned_records) + len(exact_records),
        "learned_row_count": len(learned_records),
        "exact_family_row_count": len(exact_records),
        "assessed_row_count": len(assessed_rows),
        "aggregated_row_count": len(aggregated_rows),
        "main_package": str(main_pkg),
        "thresholds": {
            "law_gain_threshold": float(DEFAULT_LAW_GAIN_THRESHOLD),
            "spread_gain_threshold": float(DEFAULT_SPREAD_GAIN_THRESHOLD),
            "root_ratio_limit": float(DEFAULT_ROOT_RATIO_LIMIT),
        },
        "claim_readout": claim,
        "boundary_candidates": boundary_candidates[:8],
        "figures": figure_paths,
        "figure_titles": figure_titles,
        "unified_core": unified_core,
    }
    summary_path = output_dir / "law_stress_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_csv(output_dir / "law_stress_assessed_rows.csv", assessed_rows)
    write_csv(output_dir / "law_stress_aggregated_rows.csv", aggregated_rows)

    # ── Markdown ─────────────────────────────────────────────────────
    pkg_labels_flat = {k: v.replace("\n", " ") for k, v in family.package_labels.items()}
    md_lines = [
        f"# {title}",
        "",
        f"- **Primary metric**: held-out {family.primary_metric_label} vs matched root-only baseline. "
        f"PrimGain = 1 − {family.primary_metric_label} ratio.",
        "- **Local laws** (C1, C2, C3) are regularisation diagnostics. They explain *why* a learned g works, "
        "but downstream error is the success criterion.",
        "- For regularisation strength tuning, see the learnability report.",
        "",
        "## Ablation Summary",
        "",
    ]
    md_lines.extend(build_downstream_table(
        aggregated_rows,
        packages_order=list(family.valid_law_packages),
        package_labels=pkg_labels_flat,
        primary_label=family.primary_metric_label,
    ))

    # Claim status
    claim_main = dict(claim.get("main_row", {}) or {})
    claim_ablation = dict(claim.get("strongest_ablation_row", {}) or {})
    md_lines.extend(["## Claim Status", ""])
    if claim_main:
        md_lines.extend([
            f"- Expected claim package: `{claim.get('main_package', '')}`.",
            f"- Downstream status: `{claim.get('status', 'unknown')}`.",
            (
                f"- Claim-row readout: `Prim%={100.0 * float(claim_main.get('primary_pass_rate', 0.0)):.1f}%`, "
                f"`PrimGain={100.0 * float(claim_main.get('mean_primary_gain', 0.0)):.1f}%`, "
                f"`C1={100.0 * float(claim_main.get('c1_pass_rate', 0.0)):.0f}%`, "
                f"`C2={100.0 * float(claim_main.get('c2_pass_rate', 0.0)):.0f}%`, "
                f"`C3={100.0 * float(claim_main.get('c3_pass_rate', 0.0)):.0f}%`."
            ),
        ])
    if claim_ablation:
        md_lines.append(
            f"- Strongest ablation: `{claim_ablation.get('law_package', '')}` with "
            f"`Prim%={100.0 * float(claim_ablation.get('primary_pass_rate', 0.0)):.1f}%` and "
            f"`PrimGain={100.0 * float(claim_ablation.get('mean_primary_gain', 0.0)):.1f}%`. "
            "This remains diagnostic-only."
        )
    if claim.get("note"):
        md_lines.append(f"- {claim['note']}")

    # Narrative
    md_lines.extend(["", "## Narrative", ""])
    md_lines.extend(f"- {line}" if line else "" for line in narrative)

    # Unified core
    md_lines.extend([""])
    md_lines.extend(render_local_law_report_markdown(unified_core))

    # Figures
    pdf_path = Path(args.pdf_path) if args.pdf_path else (output_dir / "law_stress_report.pdf")
    md_lines.extend(["", "## Figures", ""])
    for fig in figure_paths:
        md_lines.append(f"- {figure_titles.get(fig, Path(fig).name)}: `{fig}`")
    md_lines.append(f"- PDF: `{pdf_path}`")
    (output_dir / "law_stress.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # ── PDF ──────────────────────────────────────────────────────────
    with PdfPages(pdf_path) as pdf:
        # Claim page
        claim_lines = []
        if claim_main:
            claim_lines.extend([
                f"Expected claim package: {claim.get('main_package', '')}",
                f"Downstream status: {claim.get('status', 'unknown')}",
                (
                    "Claim-row readout: "
                    f"Prim%={100.0 * float(claim_main.get('primary_pass_rate', 0.0)):.1f}%, "
                    f"PrimGain={100.0 * float(claim_main.get('mean_primary_gain', 0.0)):.1f}%, "
                    f"C1={100.0 * float(claim_main.get('c1_pass_rate', 0.0)):.0f}%, "
                    f"C2={100.0 * float(claim_main.get('c2_pass_rate', 0.0)):.0f}%, "
                    f"C3={100.0 * float(claim_main.get('c3_pass_rate', 0.0)):.0f}%."
                ),
            ])
        if claim_ablation:
            claim_lines.append(
                f"Strongest ablation: {claim_ablation.get('law_package', '')} | "
                f"Prim%={100.0 * float(claim_ablation.get('primary_pass_rate', 0.0)):.1f}% | "
                f"PrimGain={100.0 * float(claim_ablation.get('mean_primary_gain', 0.0)):.1f}% | diagnostic only."
            )
        if claim.get("note"):
            claim_lines.append(str(claim["note"]))
        write_text_page(pdf, title=f"{title} | Claim Status", lines=claim_lines or ["No claim summary available."])

        # Narrative page
        write_text_page(pdf, title=title, lines=narrative)

        # Unified core pages
        write_local_law_report_core_pages(pdf, title=title, core=unified_core)

        # Boundary candidates
        boundary_lines = [
            (
                f"{family.heatmap_row_field}={r.get(family.heatmap_row_field, '?')} | "
                f"{family.heatmap_col_field}={r.get(family.heatmap_col_field, '?')} | "
                f"bundle_success_rate={float(r['bundle_full_success_rate']):.2f} | "
                f"bundle_margin_mean={float(r['bundle_margin_mean']):.3f} | "
                f"failure_reason={r.get('dominant_failure_reason', 'n/a') or 'n/a'}"
            )
            for r in boundary_candidates[:8]
        ] or ["No boundary candidates available."]
        write_text_page(pdf, title=f"{title} | Boundary Candidates", lines=boundary_lines)

        # Figure pages
        for fig in figure_paths:
            write_image_page(pdf, image_path=Path(fig), title=figure_titles.get(fig, Path(fig).name))

    summary["pdf"] = str(pdf_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "pdf": str(pdf_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
