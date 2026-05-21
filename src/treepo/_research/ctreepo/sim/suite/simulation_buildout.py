#!/usr/bin/env python3
"""Build command lists for the 5-part simulation buildout suite."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _now_run_id() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _fmt_float(x: float) -> str:
    return f"{float(x):.6g}".replace("-", "m").replace(".", "p")


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = "\n".join(str(x) for x in lines if str(x).strip())
    path.write_text(txt + ("\n" if txt else ""), encoding="utf-8")


def _append_sim_cmd(
    *,
    cmds: List[str],
    cmd: str,
    json_path: Path,
    csv_path: Path,
    skip_existing: bool,
) -> bool:
    if bool(skip_existing) and json_path.exists() and csv_path.exists():
        return False
    cmds.append(cmd)
    return True


def _profile_lists(profile: str) -> Tuple[List[int], List[int], List[float]]:
    if profile == "smoke":
        return [200], [0, 1], [0.5, 1.0]
    if profile == "full":
        return [200, 500, 1000, 2000], list(range(8)), [0.1, 0.5, 1.0]
    # paper
    return [200, 500, 1000], [0, 1, 2, 3], [0.1, 0.5, 1.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build command files for full simulation buildout items (1-5).")
    p.add_argument("--python-bin", type=str, default="venv/bin/python")
    p.add_argument("--profile", choices=["smoke", "paper", "full"], default="paper")
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--output-root", type=str, default="")
    p.add_argument(
        "--baseline-root",
        type=str,
        default="outputs/cpu_megasweep_20260302_megasweep_paper_v2",
        help="Existing baseline run used for immediate item-1 gap plots.",
    )
    p.add_argument(
        "--ipw-source-summary",
        type=str,
        default="outputs/ipw_stress_ladder_hard_large_20260302_183753/summary_rows.csv",
        help="Optional pre-existing IPW summary CSV for immediate diagnostics.",
    )
    p.add_argument("--out-cmds", type=str, default="")
    p.add_argument("--out-plot-cmds", type=str, default="")
    p.add_argument("--out-meta", type=str, default="")
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--markov-n-epochs", type=int, default=12)
    p.add_argument("--markov-device", type=str, default="auto")
    p.add_argument("--markov-cuda-device", type=int, default=None)
    p.add_argument("--ipw-jobs", type=int, default=128)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_id = str(args.run_id).strip() or _now_run_id()
    output_root = Path(args.output_root) if str(args.output_root).strip() else Path(f"outputs/simulation_buildout_{run_id}")
    baseline_root = Path(str(args.baseline_root))
    figures_root = output_root / "figures"
    out_cmds = Path(args.out_cmds) if str(args.out_cmds).strip() else Path(f"logs/simulation_buildout_{run_id}_cmds.txt")
    out_plot_cmds = (
        Path(args.out_plot_cmds)
        if str(args.out_plot_cmds).strip()
        else Path(f"logs/simulation_buildout_{run_id}_plot_cmds.txt")
    )
    out_meta = Path(args.out_meta) if str(args.out_meta).strip() else Path(f"logs/simulation_buildout_{run_id}_meta.json")

    output_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    out_cmds.parent.mkdir(parents=True, exist_ok=True)

    python_bin = str(args.python_bin)
    profile = str(args.profile)
    skip_existing = bool(args.skip_existing)
    markov_device_flag = f"--device {str(args.markov_device)} --torch-threads {int(args.torch_threads)}"
    if args.markov_cuda_device is not None:
        markov_device_flag += f" --cuda-device {int(args.markov_cuda_device)}"

    cmds: List[str] = []
    counts: Dict[str, int] = {
        "item2_hard_markov": 0,
        "item2_hard_segment_lda_ops": 0,
        "item2_hard_ctreepo": 0,
        "item3_estimator_stress_segment_lda_ops": 0,
        "item4_guidance_frontier_ctreepo": 0,
        "item5_ipw_expanded": 0,
    }

    train_docs_common, seeds_common, audit_fracs_common = _profile_lists(profile)

    # ------------------------------------------------------------------
    # Item 2: Hard regimes
    # ------------------------------------------------------------------
    hard_markov_root = output_root / "hard_regimes" / "markov"
    hard_segment_root = output_root / "hard_regimes" / "segment_lda_ops"
    hard_ctree_root = output_root / "hard_regimes" / "segmented_lda_ctreepo"

    markov_regimes = [
        ("moderate", 1.75, 14, 28, 6, 24),
        ("severe", 2.50, 18, 36, 4, 18),
    ]
    for regime_name, tstd, min_seg, max_seg, min_seg_len, max_seg_len in markov_regimes:
        for td in train_docs_common:
            for af in audit_fracs_common:
                for seed in seeds_common:
                    out_json = hard_markov_root / f"regime_{regime_name}" / f"train_{td}" / f"audit_{_fmt_float(af)}" / f"seed_{seed}.json"
                    out_csv = out_json.with_suffix(".csv")
                    cmd = (
                        f"{python_bin} -u scripts/run_markov_changepoint_ops_count_simulation.py "
                        f"--model-family neural "
                        f"--train-docs {td} --test-docs 1500 "
                        f"--audit-fraction {af} --leaf-query-rate 1.0 "
                        f"--min-segments {min_seg} --max-segments {max_seg} "
                        f"--min-seg-len {min_seg_len} --max-seg-len {max_seg_len} "
                        f"--transition-log-std {tstd} "
                        f"--c3-audit-strategy uniform "
                        f"--n-epochs {int(args.markov_n_epochs)} "
                        f"{markov_device_flag} "
                        f"--seed {seed} "
                        f"--json-summary {out_json} --csv-summary {out_csv}"
                    )
                    if _append_sim_cmd(cmds=cmds, cmd=cmd, json_path=out_json, csv_path=out_csv, skip_existing=skip_existing):
                        counts["item2_hard_markov"] += 1

    segment_hard_regimes = [
        ("moderate", 1.2, 0.6, 12.0, 0.5, 0.4),
        ("severe", 2.5, 1.2, 6.0, 0.9, 0.0),
    ]
    segment_train_docs = [x for x in train_docs_common if x >= 200]
    for regime_name, doc_alpha, topic_alpha, anchor_mult, b_strength, seg_power in segment_hard_regimes:
        for td in segment_train_docs:
            for af in audit_fracs_common:
                for seed in seeds_common:
                    out_json = (
                        hard_segment_root
                        / f"regime_{regime_name}"
                        / "phi_tensor_lda"
                        / f"train_{td}"
                        / f"audit_{_fmt_float(af)}"
                        / f"seed_{seed}.json"
                    )
                    out_csv = out_json.with_suffix(".csv")
                    cmd = (
                        f"{python_bin} -u scripts/run_segment_lda_ops_weight_recovery_simulation.py "
                        f"--train-docs {td} --test-docs 2000 "
                        f"--audit-fraction {af} --audit-policy fraction "
                        f"--topic-source infer --feature-inference hard "
                        f"--topic-phi-estimator tensor_lda --topic-phi-docs 0 "
                        f"--run-all-feature-modes "
                        f"--topic-process segments "
                        f"--doc-topic-concentration {doc_alpha} "
                        f"--topic-concentration {topic_alpha} "
                        f"--anchor-multiplier {anchor_mult} "
                        f"--boundary-profile random "
                        f"--boundary-profile-strength {b_strength} "
                        f"--segment-length-power {seg_power} "
                        f"--seed {seed} "
                        f"--json-summary {out_json} --csv-summary {out_csv}"
                    )
                    if _append_sim_cmd(cmds=cmds, cmd=cmd, json_path=out_json, csv_path=out_csv, skip_existing=skip_existing):
                        counts["item2_hard_segment_lda_ops"] += 1

    ctree_train_docs = [128, 256, 512] if profile != "smoke" else [256]
    ctree_seeds = [0, 1] if profile == "smoke" else ([0, 1, 2, 3] if profile == "paper" else list(range(6)))
    ctree_hard_regimes = [
        ("moderate", 0.50, 0.25, 40.0, 4.0),
        ("severe", 1.00, 0.40, 20.0, 8.0),
    ]
    ctree_rate_pairs = [(0.5, 0.5), (1.0, 1.0)]
    for regime_name, alpha_topic, beta_word, seg_conc, seg_bg in ctree_hard_regimes:
        for td in ctree_train_docs:
            for leaf_rate, int_rate in ctree_rate_pairs:
                for seed in ctree_seeds:
                    out_json = (
                        hard_ctree_root
                        / f"regime_{regime_name}"
                        / f"train_{td}"
                        / f"leaf_{_fmt_float(leaf_rate)}"
                        / f"int_{_fmt_float(int_rate)}"
                        / f"seed_{seed}.json"
                    )
                    out_csv = out_json.with_suffix(".csv")
                    cmd = (
                        f"{python_bin} -u scripts/run_segmented_lda_ctreepo_simulation.py "
                        f"--topic-phi-estimator spectral_numpy --topic-phi-docs 0 "
                        f"--n-books-train {td} --n-books-test 2000 "
                        f"--alpha-topic {alpha_topic} --beta-word {beta_word} "
                        f"--segment-concentration {seg_conc} --segment-background {seg_bg} "
                        f"--calibration-leaf-query-rate 0.1 --calibration-policy uniform "
                        f"--eval-leaf-query-rate {leaf_rate} --eval-internal-query-rate {int_rate} "
                        f"--eval-internal-query-design risk "
                        f"--seed {seed} "
                        f"--json-summary {out_json} --csv-summary {out_csv}"
                    )
                    if _append_sim_cmd(cmds=cmds, cmd=cmd, json_path=out_json, csv_path=out_csv, skip_existing=skip_existing):
                        counts["item2_hard_ctreepo"] += 1

    # ------------------------------------------------------------------
    # Item 3: Segment-LDA estimator stress at full audit
    # ------------------------------------------------------------------
    estimator_stress_root = output_root / "estimator_stress" / "segment_lda_ops"
    est_train_docs = [100, 200] if profile == "smoke" else ([100, 200, 500, 1000] if profile == "paper" else [100, 200, 500, 1000, 2000])
    est_seeds = [0, 1] if profile == "smoke" else ([0, 1, 2, 3, 4, 5] if profile == "paper" else list(range(8)))
    est_lambdas = [0.0, 1.0] if profile == "smoke" else [0.0, 0.25, 1.0]
    estimators = ["true", "embedding_spectral", "tensor_lda", "online_tensor_lda", "noisy_theory"]
    for est in estimators:
        for td in est_train_docs:
            for lam in est_lambdas:
                for seed in est_seeds:
                    out_json = (
                        estimator_stress_root
                        / f"phi_{est}"
                        / f"train_{td}"
                        / f"lam_{_fmt_float(lam)}"
                        / f"seed_{seed}.json"
                    )
                    out_csv = out_json.with_suffix(".csv")
                    cmd = (
                        f"{python_bin} -u scripts/run_segment_lda_ops_weight_recovery_simulation.py "
                        f"--train-docs {td} --test-docs 2000 "
                        f"--audit-fraction 1.0 --audit-policy fraction "
                        f"--topic-source infer --feature-inference hard "
                        f"--topic-phi-estimator {est} --topic-phi-docs 0 "
                        f"--topic-process segments "
                        f"--lambda-multiplier {lam} "
                        f"--run-all-feature-modes "
                        f"--seed {seed} "
                        f"--json-summary {out_json} --csv-summary {out_csv}"
                    )
                    if _append_sim_cmd(cmds=cmds, cmd=cmd, json_path=out_json, csv_path=out_csv, skip_existing=skip_existing):
                        counts["item3_estimator_stress_segment_lda_ops"] += 1

    # ------------------------------------------------------------------
    # Item 4: C-TreePO guidance frontier (leaf/internal grid)
    # ------------------------------------------------------------------
    frontier_root = output_root / "guidance_frontier" / "segmented_lda_ctreepo"
    frontier_train = [256] if profile == "smoke" else ([128, 256, 512] if profile == "paper" else [64, 128, 256, 512])
    frontier_seeds = [0, 1] if profile == "smoke" else ([0, 1, 2] if profile == "paper" else [0, 1, 2, 3, 4, 5])
    frontier_leaf_rates = [0.0, 0.5, 1.0] if profile == "smoke" else [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
    frontier_int_rates = [0.0, 0.5, 1.0] if profile == "smoke" else [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
    frontier_cal_rates = [0.1] if profile == "smoke" else ([0.0, 0.1, 0.25] if profile == "paper" else [0.0, 0.05, 0.1, 0.25])
    for td in frontier_train:
        for cal in frontier_cal_rates:
            for lr in frontier_leaf_rates:
                for ir in frontier_int_rates:
                    for seed in frontier_seeds:
                        out_json = (
                            frontier_root
                            / f"train_{td}"
                            / f"cal_{_fmt_float(cal)}"
                            / f"leaf_{_fmt_float(lr)}"
                            / f"int_{_fmt_float(ir)}"
                            / f"seed_{seed}.json"
                        )
                        out_csv = out_json.with_suffix(".csv")
                        cmd = (
                            f"{python_bin} -u scripts/run_segmented_lda_ctreepo_simulation.py "
                            f"--topic-phi-estimator spectral_numpy --topic-phi-docs 0 "
                            f"--n-books-train {td} --n-books-test 2000 "
                            f"--calibration-leaf-query-rate {cal} --calibration-policy uniform "
                            f"--eval-leaf-query-rate {lr} --eval-internal-query-rate {ir} "
                            f"--eval-internal-query-design risk "
                            f"--seed {seed} "
                            f"--json-summary {out_json} --csv-summary {out_csv}"
                        )
                        if _append_sim_cmd(
                            cmds=cmds,
                            cmd=cmd,
                            json_path=out_json,
                            csv_path=out_csv,
                            skip_existing=skip_existing,
                        ):
                            counts["item4_guidance_frontier_ctreepo"] += 1

    # ------------------------------------------------------------------
    # Item 5: Expanded IPW ladders + diagnostics
    # ------------------------------------------------------------------
    ipw_root = output_root / "ipw_expanded"
    ipw_root.mkdir(parents=True, exist_ok=True)
    n_docs_values = "60,120,240" if profile == "smoke" else ("60,120,240,480,960,1920" if profile == "paper" else "60,120,240,480,960,1920,3840")
    ipw_trials = 800 if profile == "smoke" else (2000 if profile == "paper" else 4000)
    ipw_pop_seeds = 8 if profile == "smoke" else (20 if profile == "paper" else 30)
    ipw_summary_csv = ipw_root / "summary_rows.csv"
    if (not skip_existing) or (not ipw_summary_csv.exists()):
        cmd = (
            f"{python_bin} -u scripts/run_ipw_stress_ladder.py "
            f"--case-set both "
            f"--n-docs-values {n_docs_values} "
            f"--trials {ipw_trials} "
            f"--n-population-seeds {ipw_pop_seeds} "
            f"--jobs {int(args.ipw_jobs)} "
            f"--output-csv {ipw_root / 'raw_rows.csv'} "
            f"--output-summary-csv {ipw_root / 'summary_rows.csv'} "
            f"--output-json {ipw_root / 'summary.json'}"
        )
        cmds.append(cmd)
        counts["item5_ipw_expanded"] += 1

    toy_matrix_json = ipw_root / "ci_toy_matrix.json"
    if (not skip_existing) or (not toy_matrix_json.exists()):
        cmd = (
            f"mkdir -p {ipw_root} && "
            f"{python_bin} -u scripts/run_ipw_ci_simulation.py "
            f"--population-model toy --toy-matrix --design compare "
            f"--trials {max(600, ipw_trials // 2)} --delta 0.1 --json "
            f"> {toy_matrix_json}"
        )
        cmds.append(cmd)
        counts["item5_ipw_expanded"] += 1

    toy_mergeable_json = ipw_root / "ci_toy_mergeable_examples.json"
    if (not skip_existing) or (not toy_mergeable_json.exists()):
        cmd = (
            f"mkdir -p {ipw_root} && "
            f"{python_bin} -u scripts/run_ipw_ci_simulation.py "
            f"--population-model toy --toy-mergeable-examples --design compare "
            f"--trials {max(600, ipw_trials // 2)} --delta 0.1 --json "
            f"> {toy_mergeable_json}"
        )
        cmds.append(cmd)
        counts["item5_ipw_expanded"] += 1

    _write_lines(out_cmds, cmds)

    # ------------------------------------------------------------------
    # Plot/report command list
    # ------------------------------------------------------------------
    plot_cmds: List[str] = []

    # Item 1: full-budget gap suite (baseline)
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_full_budget_gap_suite.py",
                "--markov-glob",
                f"'{baseline_root}/markov_changepoint_ops_count/**/*seed_*.json'",
                "--segment-glob",
                f"'{baseline_root}/segment_lda_ops_weight_recovery/**/*seed_*.json'",
                "--ctree-glob",
                f"'{baseline_root}/segmented_lda_ctreepo/**/*.json'",
                "--output-figure",
                f"'{figures_root}/full_budget_gap_suite.png'",
                "--output-json",
                f"'{figures_root}/full_budget_gap_suite_report.json'",
            ]
        )
    )

    # Item 2: hard-regime summary
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_hard_regime_summary.py",
                "--markov-glob",
                f"'{hard_markov_root}/**/*.json'",
                "--segment-glob",
                f"'{hard_segment_root}/**/*.json'",
                "--ctree-glob",
                f"'{hard_ctree_root}/**/*.json'",
                "--output-figure",
                f"'{figures_root}/hard_regime_summary.png'",
                "--output-json",
                f"'{figures_root}/hard_regime_summary_report.json'",
            ]
        )
    )

    # Item 3: estimator stress
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_segment_lda_estimator_stress.py",
                "--input-glob",
                f"'{estimator_stress_root}/**/*.json'",
                "--output-figure",
                f"'{figures_root}/segment_lda_estimator_stress.png'",
                "--output-json",
                f"'{figures_root}/segment_lda_estimator_stress_report.json'",
            ]
        )
    )

    # Item 4: guidance frontier
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_ctreepo_guidance_frontier.py",
                "--input-glob",
                f"'{frontier_root}/**/*.json'",
                "--output-figure",
                f"'{figures_root}/ctreepo_guidance_frontier.png'",
                "--output-json",
                f"'{figures_root}/ctreepo_guidance_frontier_report.json'",
            ]
        )
    )

    # Item 5: IPW diagnostics from newly generated summary.
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_ipw_stress_ladder.py",
                "--input-csv",
                f"'{ipw_root / 'summary_rows.csv'}'",
                "--metric",
                "violation",
                "--output-figure",
                f"'{figures_root}/ipw_stress_ladder_violation.png'",
                "--output-json",
                f"'{figures_root}/ipw_stress_ladder_violation_report.json'",
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_ipw_stress_ladder.py",
                "--input-csv",
                f"'{ipw_root / 'summary_rows.csv'}'",
                "--metric",
                "preference",
                "--output-figure",
                f"'{figures_root}/ipw_stress_ladder_preference.png'",
                "--output-json",
                f"'{figures_root}/ipw_stress_ladder_preference_report.json'",
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_ipw_propensity_diagnostics.py",
                "--input-csv",
                f"'{ipw_root / 'summary_rows.csv'}'",
                "--metric",
                "violation",
                "--output-figure",
                f"'{figures_root}/ipw_propensity_diagnostics_violation.png'",
                "--output-json",
                f"'{figures_root}/ipw_propensity_diagnostics_violation_report.json'",
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_ipw_propensity_diagnostics.py",
                "--input-csv",
                f"'{ipw_root / 'summary_rows.csv'}'",
                "--metric",
                "preference",
                "--output-figure",
                f"'{figures_root}/ipw_propensity_diagnostics_preference.png'",
                "--output-json",
                f"'{figures_root}/ipw_propensity_diagnostics_preference_report.json'",
            ]
        )
    )

    # Optional immediate IPW diagnostics from pre-existing source summary.
    ipw_source_summary = Path(str(args.ipw_source_summary))
    if ipw_source_summary.exists():
        plot_cmds.append(
            " ".join(
                [
                    python_bin,
                    "-u",
                    "scripts/plot_ipw_propensity_diagnostics.py",
                    "--input-csv",
                    f"'{ipw_source_summary}'",
                    "--metric",
                    "violation",
                    "--output-figure",
                    f"'{figures_root}/ipw_propensity_diagnostics_existing_violation.png'",
                    "--output-json",
                    f"'{figures_root}/ipw_propensity_diagnostics_existing_violation_report.json'",
                ]
            )
        )

    _write_lines(out_plot_cmds, plot_cmds)

    meta = {
        "run_id": run_id,
        "profile": profile,
        "python_bin": python_bin,
        "skip_existing": skip_existing,
        "markov_device": str(args.markov_device),
        "markov_cuda_device": int(args.markov_cuda_device) if args.markov_cuda_device is not None else None,
        "torch_threads": int(args.torch_threads),
        "output_root": str(output_root),
        "baseline_root": str(baseline_root),
        "figures_root": str(figures_root),
        "cmds_file": str(out_cmds),
        "plot_cmds_file": str(out_plot_cmds),
        "counts_by_suite": counts,
        "n_sim_commands_total": int(len(cmds)),
        "n_plot_commands_total": int(len(plot_cmds)),
    }
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
