#!/usr/bin/env python3
"""Build one unified command list for the core CPU simulation suite.

This is a thin orchestrator that calls the existing per-sim builders, writes:

1) a single xargs-friendly sims command file, and
2) a follow-on plots command file (ceilings/grids) to run after sims finish.

Design goals:
- deterministic run id (timestamp) for reproducibility
- outputs isolated under one output root
- sweep-friendly defaults (skip-existing + CPU-safe threading in runner)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
import subprocess
import sys
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]


def _now_run_id() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.lstrip().startswith("#")]


def _append_unique_items(base: str, extras: List[str]) -> str:
    items: List[str] = []
    seen = set()
    for raw in str(base).replace(",", " ").split():
        token = str(raw).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        items.append(token)
    for raw in extras:
        token = str(raw).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        items.append(token)
    return " ".join(items)


def _call_builder(args: List[str]) -> None:
    proc = subprocess.run(args, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"builder_failed rc={proc.returncode}: {' '.join(args)}\n{proc.stdout}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified CPU megasweep command lists (sims + plots).")
    p.add_argument("--python-bin", type=str, default="venv/bin/python")
    p.add_argument("--run-id", type=str, default="", help="Run id suffix (default: UTC timestamp).")
    p.add_argument("--output-root", type=str, default="", help="Base output dir (default: outputs/cpu_megasweep_<run-id>).")
    p.add_argument("--figures-root", type=str, default="", help="Figures dir (default: <output-root>/figures).")
    p.add_argument("--out-cmds", type=str, default="", help="Unified sims cmds file (default: logs/cpu_megasweep_<run-id>_cmds.txt).")
    p.add_argument("--out-plot-cmds", type=str, default="", help="Unified plot cmds file (default: logs/cpu_megasweep_<run-id>_plot_cmds.txt).")
    p.add_argument("--out-meta", type=str, default="", help="Metadata JSON path (default: logs/cpu_megasweep_<run-id>_meta.json).")

    p.add_argument("--profile", choices=["smoke", "paper", "full"], default="paper")

    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)

    # Optional “knobs” that keep the mega sweep CPU-friendly.
    p.add_argument(
        "--include-neural-topic-estimators",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, include neural topic-phi estimators in Segment-LDA sweeps (can be much slower on CPU).",
    )
    p.add_argument(
        "--markov-extra-c3-strategies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, include non-uniform C3 audit strategies (for selection-bias diagnostics).",
    )
    p.add_argument(
        "--markov-n-epochs",
        type=int,
        default=10,
        help="Epochs per Markov OPS-count run (CPU cost driver).",
    )
    p.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch thread count per Markov OPS-count process (sweep-friendly).",
    )
    p.add_argument("--markov-device", type=str, default="auto")
    p.add_argument("--markov-cuda-device", type=int, default=None)
    p.add_argument("--segment-device", type=str, default="auto")
    p.add_argument("--segment-cuda-device", type=int, default=None)
    p.add_argument("--ctree-device", type=str, default="auto")
    p.add_argument("--ctree-cuda-device", type=int, default=None)
    p.add_argument(
        "--include-full-budget-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include explicit 1.0 budget/full-guidance points in smoke/paper profiles.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_id = str(args.run_id).strip() or _now_run_id()

    output_root = Path(args.output_root) if str(args.output_root).strip() else Path(f"outputs/cpu_megasweep_{run_id}")
    figures_root = Path(args.figures_root) if str(args.figures_root).strip() else (output_root / "figures")

    out_cmds = Path(args.out_cmds) if str(args.out_cmds).strip() else Path(f"logs/cpu_megasweep_{run_id}_cmds.txt")
    out_plot_cmds = (
        Path(args.out_plot_cmds)
        if str(args.out_plot_cmds).strip()
        else Path(f"logs/cpu_megasweep_{run_id}_plot_cmds.txt")
    )
    out_meta = Path(args.out_meta) if str(args.out_meta).strip() else Path(f"logs/cpu_megasweep_{run_id}_meta.json")

    python_bin = str(args.python_bin)
    output_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    out_cmds.parent.mkdir(parents=True, exist_ok=True)

    profile = str(args.profile)
    skip_flag = "--skip-existing" if bool(args.skip_existing) else "--no-skip-existing"

    # ----------------------------
    # Markov OPS-count sweep
    # ----------------------------
    markov_out = output_root / "markov_changepoint_ops_count"
    markov_cmds = out_cmds.with_name(out_cmds.stem + "_markov.txt")
    c3_strats = "uniform"
    if bool(args.markov_extra_c3_strategies):
        c3_strats = "uniform top_span span_weighted hybrid_top_span"
    if profile == "smoke":
        train_docs = "50 100"
        audit_fracs = "0.1 0.2"
        seeds = "0 1"
    elif profile == "full":
        train_docs = "50 100 200 500 1000 2000"
        audit_fracs = "0.05 0.1 0.2 0.5 1.0"
        seeds = "0 1 2 3 4 5 6 7"
    else:  # paper
        train_docs = "50 100 200 500 1000"
        audit_fracs = "0.05 0.1 0.2 0.5 1.0"
        seeds = "0 1 2 3 4 5"
    if bool(args.include_full_budget_anchors):
        audit_fracs = _append_unique_items(audit_fracs, ["1.0"])

    markov_builder_args = [
        python_bin,
        "-u",
        "scripts/build_markov_changepoint_ops_count_cmds.py",
        "--out-cmds",
        str(markov_cmds),
        "--output-root",
        str(markov_out),
        "--train-docs",
        train_docs,
        "--audit-fractions",
        audit_fracs,
        "--c3-audit-strategies",
        c3_strats,
        "--seeds",
        seeds,
        "--n-epochs",
        str(int(args.markov_n_epochs)),
        "--device",
        str(args.markov_device),
        "--torch-threads",
        str(int(args.torch_threads)),
        skip_flag,
    ]
    if args.markov_cuda_device is not None:
        markov_builder_args.extend(["--cuda-device", str(int(args.markov_cuda_device))])
    _call_builder(markov_builder_args)

    # ----------------------------
    # Segment-LDA OPS weight recovery sweep
    # ----------------------------
    seg_out = output_root / "segment_lda_ops_weight_recovery"
    seg_cmds = out_cmds.with_name(out_cmds.stem + "_segment_lda_ops.txt")

    if profile == "smoke":
        seg_train_docs = "100 200"
        seg_audit_fracs = "0.1 0.2"
        seg_seeds = "0 1"
        seg_lambda = "0 1.0"
        seg_processes = "segments"
        seg_phi_docs = "0"
    elif profile == "full":
        seg_train_docs = "100 200 500 1000 2000"
        seg_audit_fracs = "0.05 0.1 0.2 0.5 1.0"
        seg_seeds = "0 1 2 3 4 5 6 7"
        seg_lambda = "0 0.25 1.0"
        seg_processes = "segments bag_of_words"
        seg_phi_docs = "0"
    else:  # paper
        seg_train_docs = "100 200 500 1000"
        seg_audit_fracs = "0.05 0.1 0.2 0.5"
        seg_seeds = "0 1 2 3 4 5"
        seg_lambda = "0 0.25 1.0"
        seg_processes = "segments"
        seg_phi_docs = "0"
    if bool(args.include_full_budget_anchors):
        seg_audit_fracs = _append_unique_items(seg_audit_fracs, ["1.0"])

    base_estimators = [
        "true",
        "noisy_theory",
        "tensor_lda",
        "online_tensor_lda",
        "embedding_spectral",
    ]
    if bool(args.include_neural_topic_estimators):
        base_estimators.extend(
            [
                "neural_ctreepo",
                "neural_mergeable_sketch",
                "neural_hybrid",
                "neural_embedding_hybrid",
            ]
        )
    seg_estimators = " ".join(base_estimators)

    seg_builder_args = [
        python_bin,
        "-u",
        "scripts/build_segment_lda_ops_weight_recovery_cmds.py",
        "--out-cmds",
        str(seg_cmds),
        "--output-root",
        str(seg_out),
        "--train-docs",
        seg_train_docs,
        "--audit-fractions",
        seg_audit_fracs,
        "--topic-phi-docs",
        seg_phi_docs,
        "--topic-phi-estimators",
        seg_estimators,
        "--topic-processes",
        seg_processes,
        "--lambda-multipliers",
        seg_lambda,
        "--seeds",
        seg_seeds,
        "--topic-source",
        "infer",
        "--feature-inference",
        "hard",
        "--device",
        str(args.segment_device),
        "--torch-threads",
        str(int(args.torch_threads)),
        "--run-all-feature-modes",
        skip_flag,
    ]
    if args.segment_cuda_device is not None:
        seg_builder_args.extend(["--cuda-device", str(int(args.segment_cuda_device))])
    _call_builder(seg_builder_args)

    # ----------------------------
    # Segmented-LDA C-TreePO sweep
    # ----------------------------
    ctree_out = output_root / "segmented_lda_ctreepo"
    ctree_cmds = out_cmds.with_name(out_cmds.stem + "_segmented_lda_ctreepo.txt")

    if profile == "smoke":
        ctree_train = "64 128"
        ctree_cal = "0 0.1"
        ctree_int = "0 0.5"
        ctree_leaf = "0"
        ctree_seeds = "0 1"
    elif profile == "full":
        ctree_train = "64 128 256 512"
        ctree_cal = "0 0.05 0.1 0.25 0.5"
        ctree_int = "0 0.05 0.1 0.25 0.5 1.0"
        ctree_leaf = "0 1.0"
        ctree_seeds = "0 1 2 3 4 5 6 7"
    else:  # paper
        ctree_train = "64 128 256 512"
        ctree_cal = "0 0.05 0.1 0.25"
        ctree_int = "0 0.05 0.1 0.25 0.5"
        ctree_leaf = "0"
        ctree_seeds = "0 1 2 3 4 5"
    if bool(args.include_full_budget_anchors):
        ctree_int = _append_unique_items(ctree_int, ["1.0"])
        ctree_leaf = _append_unique_items(ctree_leaf, ["1.0"])

    ctree_builder_args = [
        python_bin,
        "-u",
        "scripts/build_segmented_lda_ctreepo_cmds.py",
        "--out-cmds",
        str(ctree_cmds),
        "--output-root",
        str(ctree_out),
        "--train-docs",
        ctree_train,
        "--calibration-rates",
        ctree_cal,
        "--eval-internal-rates",
        ctree_int,
        "--eval-leaf-rates",
        ctree_leaf,
        "--seeds",
        ctree_seeds,
        "--topic-phi-estimator",
        "spectral_numpy",
        "--eval-internal-query-design",
        "risk",
        "--device",
        str(args.ctree_device),
        "--torch-threads",
        str(int(args.torch_threads)),
        skip_flag,
    ]
    if args.ctree_cuda_device is not None:
        ctree_builder_args.extend(["--cuda-device", str(int(args.ctree_cuda_device))])
    _call_builder(ctree_builder_args)

    # ----------------------------
    # Merge unified sim cmds
    # ----------------------------
    cmd_sources = {
        "markov": markov_cmds,
        "segment_lda_ops": seg_cmds,
        "segmented_lda_ctreepo": ctree_cmds,
    }
    all_cmds: List[str] = []
    counts: Dict[str, int] = {}
    for key, path in cmd_sources.items():
        lines = _read_lines(path)
        counts[key] = int(len(lines))
        all_cmds.extend(lines)

    _write_text(out_cmds, "\n".join(all_cmds) + ("\n" if all_cmds else ""))

    # ----------------------------
    # Plot commands (run after sims)
    # ----------------------------
    plot_cmds: List[str] = []

    # Markov plots
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_markov_changepoint_ops_count_grid.py",
                "--input-glob",
                f"'{markov_out}/**/*seed_*.json'",
                "--layout",
                "honesty",
                "--aggregate",
                "median",
                "--normalize",
                "--output-figure",
                f"'{figures_root}/markov_ops_count_honesty_grid.png'",
                "--output-json",
                f"'{figures_root}/markov_ops_count_honesty_grid_report.json'",
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_markov_changepoint_ops_count_ceilings.py",
                "--input-glob",
                f"'{markov_out}/**/*seed_*.json'",
                "--output-figure",
                f"'{figures_root}/markov_ops_count_ceilings.png'",
                "--output-json",
                f"'{figures_root}/markov_ops_count_ceilings_report.json'",
                "--aggregate",
                "median",
                "--band",
                "p10_p90",
            ]
        )
    )

    # Segment-LDA plots
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_segment_lda_ops_weight_recovery_grid.py",
                "--input-glob",
                f"'{seg_out}/**/*seed_*.json'",
                "--audit-strategy",
                "random",
                "--output-figure",
                f"'{figures_root}/segment_lda_ops_weight_recovery_grid.png'",
                "--output-json",
                f"'{figures_root}/segment_lda_ops_weight_recovery_grid_report.json'",
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_segment_lda_ops_weight_recovery_ceilings.py",
                "--input-glob",
                f"'{seg_out}/**/*seed_*.json'",
                "--audit-strategy",
                "random",
                "--output-figure",
                f"'{figures_root}/segment_lda_ops_weight_recovery_ceilings.png'",
                "--output-json",
                f"'{figures_root}/segment_lda_ops_weight_recovery_ceilings_report.json'",
                "--aggregate",
                "median",
                "--band",
                "p10_p90",
            ]
        )
    )

    # Segmented-LDA C-TreePO plots
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_segmented_lda_ctreepo_phase.py",
                "--input-glob",
                f"'{ctree_out}/**/*.json'",
                "--metric",
                "decomposition_total_root_l1_mean",
                "--aggregate",
                "median",
                "--output-figure",
                f"'{figures_root}/segmented_lda_ctreepo_phase.png'",
                "--output-json",
                f"'{figures_root}/segmented_lda_ctreepo_phase_report.json'",
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_segmented_lda_ctreepo_ceilings.py",
                "--input-glob",
                f"'{ctree_out}/**/*.json'",
                "--aggregate",
                "median",
                "--band",
                "p10_p90",
                "--output-figure",
                f"'{figures_root}/segmented_lda_ctreepo_ceilings.png'",
                "--output-json",
                f"'{figures_root}/segmented_lda_ctreepo_ceilings_report.json'",
            ]
        )
    )

    # Mergeable ceilings (self-contained; writes figures directly).
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_mergeable_ceilings.py",
                "--output",
                f"'{figures_root}/mergeable_ceilings.png'",
                "--json-summary",
                f"'{figures_root}/mergeable_ceilings_summary.json'",
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_mergeable_complexity_ladder.py",
                "--output",
                f"'{figures_root}/mergeable_complexity_ladder.png'",
                "--json-summary",
                f"'{figures_root}/mergeable_complexity_ladder_summary.json'",
            ]
        )
    )

    _write_text(out_plot_cmds, "\n".join(plot_cmds) + ("\n" if plot_cmds else ""))

    meta = {
        "run_id": run_id,
        "profile": profile,
        "python_bin": python_bin,
        "skip_existing": bool(args.skip_existing),
        "include_full_budget_anchors": bool(args.include_full_budget_anchors),
        "markov_device": str(args.markov_device),
        "markov_cuda_device": int(args.markov_cuda_device) if args.markov_cuda_device is not None else None,
        "segment_device": str(args.segment_device),
        "segment_cuda_device": int(args.segment_cuda_device) if args.segment_cuda_device is not None else None,
        "ctree_device": str(args.ctree_device),
        "ctree_cuda_device": int(args.ctree_cuda_device) if args.ctree_cuda_device is not None else None,
        "torch_threads": int(args.torch_threads),
        "output_root": str(output_root),
        "figures_root": str(figures_root),
        "cmds_file": str(out_cmds),
        "plot_cmds_file": str(out_plot_cmds),
        "counts_by_family": counts,
        "n_sim_commands_total": int(len(all_cmds)),
        "n_plot_commands_total": int(len(plot_cmds)),
        "builder_cmd_files": {k: str(v) for k, v in cmd_sources.items()},
    }
    _write_text(out_meta, json.dumps(meta, indent=2, sort_keys=True) + "\n")

    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
