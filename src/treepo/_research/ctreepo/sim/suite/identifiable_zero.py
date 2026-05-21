from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
from typing import Dict, List, Optional, Sequence, Tuple

from treepo._research.ctreepo.sim.manifest import RunSpec, read_manifest_jsonl, write_manifest_jsonl
from treepo._research.ctreepo.sim.runner import read_cmds_file
from treepo._research.ctreepo.sim.suite.common import (
    build_suite_meta,
    read_suite_meta,
    run_manifest_queue_suite,
    utc_run_id,
    write_suite_meta,
    write_text,
)
from treepo._research.ctreepo.sim.suite.identifiable_zero_policy import resolve_identifiable_zero_policy
from treepo._research.ctreepo.sim.suite.policy_common import join_items


def _q(x: object) -> str:
    return shlex.quote(str(x))


@dataclass(frozen=True)
class SuitePaths:
    output_root: Path
    figures_root: Path
    suite_cmds: Path
    suite_plot_cmds: Path
    suite_meta: Path
    suite_manifest: Path
    seg_cmds: Path
    seg_manifest: Path
    ctree_cmds: Path
    ctree_manifest: Path
    markov_cmds: Path
    markov_manifest: Path


def _resolve_paths(*, output_root: Path, figures_root: Path) -> SuitePaths:
    return SuitePaths(
        output_root=output_root,
        figures_root=figures_root,
        suite_cmds=output_root / "suite_cmds.txt",
        suite_plot_cmds=output_root / "suite_plot_cmds.txt",
        suite_meta=output_root / "suite_meta.json",
        suite_manifest=output_root / "suite_manifest.jsonl",
        seg_cmds=output_root / "suite_cmds_segment_lda_ops.txt",
        seg_manifest=output_root / "suite_manifest_segment_lda_ops.jsonl",
        ctree_cmds=output_root / "suite_cmds_segmented_lda_ctreepo.txt",
        ctree_manifest=output_root / "suite_manifest_segmented_lda_ctreepo.jsonl",
        markov_cmds=output_root / "suite_cmds_markov_additive.txt",
        markov_manifest=output_root / "suite_manifest_markov_additive.jsonl",
    )


def build_suite(
    *,
    run_id: str,
    profile: str,
    python_bin: str,
    output_root: Path,
    figures_root: Path,
    skip_existing: bool,
    include_markov: bool,
    include_embedding_estimator: bool,
    segment_test_docs: int,
    ctree_test_books: int,
    markov_test_docs: int,
    markov_n_epochs: int,
    segment_device: str,
    segment_cuda_device: int | None,
    ctree_device: str,
    ctree_cuda_device: int | None,
    markov_device: str,
    markov_cuda_device: int | None,
    torch_threads: int,
) -> dict:
    output_root = output_root.resolve()
    figures_root = figures_root.resolve()
    paths = _resolve_paths(output_root=output_root, figures_root=figures_root)

    output_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    policy = resolve_identifiable_zero_policy(str(profile))
    seg_train = join_items(policy.segment_train_docs)
    seg_audit = join_items(policy.segment_audit_fractions)
    seg_lam = join_items(policy.segment_lambda_multipliers)
    seg_seeds = join_items(policy.segment_seeds)
    ctree_train = join_items(policy.ctree_train_docs)
    ctree_cal = join_items(policy.ctree_calibration_rates)
    ctree_leaf = join_items(policy.ctree_eval_leaf_rates)
    ctree_int = join_items(policy.ctree_eval_internal_rates)
    ctree_seeds = join_items(policy.ctree_seeds)
    markov_train = join_items(policy.markov_train_docs)
    markov_audit = join_items(policy.markov_audit_fractions)
    markov_seeds = join_items(policy.markov_seeds)
    ctree_focus_train = max(int(x) for x in policy.ctree_train_docs)

    skip_flag = "--skip-existing" if bool(skip_existing) else "--no-skip-existing"

    # ----------------------------
    # Segment-LDA OPS (identifiable estimators only)
    # ----------------------------
    from treepo._research.ctreepo.sim.cli.sweep_segment_lda_ops_weight_recovery import (  # noqa: WPS433
        main as _seg_sweep,
    )

    seg_out = output_root / "segment_lda_ops_weight_recovery"
    seg_estimators = ["true"]
    if bool(include_embedding_estimator):
        seg_estimators.append("embedding_spectral")
    seg_estimators_str = " ".join(seg_estimators)
    seg_argv = [
        "--python-bin",
        str(python_bin),
        "--out-cmds",
        str(paths.seg_cmds),
        "--out-manifest",
        str(paths.seg_manifest),
        "--output-root",
        str(seg_out),
        "--train-docs",
        str(seg_train),
        "--test-docs",
        str(int(segment_test_docs)),
        "--audit-fractions",
        str(seg_audit),
        "--topic-phi-docs",
        "0",
        "--topic-phi-estimators",
        str(seg_estimators_str),
        "--topic-processes",
        "segments",
        "--lambda-multipliers",
        str(seg_lam),
        "--seeds",
        str(seg_seeds),
        "--topic-source",
        "infer",
        "--feature-inference",
        "hard",
        "--device",
        str(segment_device),
        "--torch-threads",
        str(int(torch_threads)),
        "--run-all-feature-modes",
        skip_flag,
    ]
    if segment_cuda_device is not None:
        seg_argv.extend(["--cuda-device", str(int(segment_cuda_device))])
    _seg_sweep(seg_argv)

    # ----------------------------
    # Segmented-LDA C-TreePO
    # ----------------------------
    from treepo._research.ctreepo.sim.cli.sweep_segmented_lda_ctreepo import (  # noqa: WPS433
        main as _ctree_sweep,
    )

    ctree_out = output_root / "segmented_lda_ctreepo"
    ctree_argv = [
        "--python-bin",
        str(python_bin),
        "--out-cmds",
        str(paths.ctree_cmds),
        "--out-manifest",
        str(paths.ctree_manifest),
        "--output-root",
        str(ctree_out),
        "--train-docs",
        str(ctree_train),
        "--seeds",
        str(ctree_seeds),
        "--calibration-rates",
        str(ctree_cal),
        "--eval-leaf-rates",
        str(ctree_leaf),
        "--eval-internal-rates",
        str(ctree_int),
        "--topic-phi-estimator",
        "spectral_numpy",
        "--topic-phi-docs",
        "0",
        "--n-books-test",
        str(int(ctree_test_books)),
        "--eval-internal-query-design",
        "risk",
        "--device",
        str(ctree_device),
        "--torch-threads",
        str(int(torch_threads)),
        skip_flag,
    ]
    if ctree_cuda_device is not None:
        ctree_argv.extend(["--cuda-device", str(int(ctree_cuda_device))])
    _ctree_sweep(ctree_argv)

    # ----------------------------
    # Optional Markov additive-only
    # ----------------------------
    cmd_sources: Dict[str, Path] = {
        "segment_lda_ops": paths.seg_cmds,
        "segmented_lda_ctreepo": paths.ctree_cmds,
    }
    manifest_sources: Dict[str, Path] = {
        "segment_lda_ops": paths.seg_manifest,
        "segmented_lda_ctreepo": paths.ctree_manifest,
    }

    markov_out = output_root / "markov_changepoint_ops_count"
    if bool(include_markov):
        from treepo._research.ctreepo.sim.cli.sweep_markov_changepoint_ops_count import (  # noqa: WPS433
            main as _markov_sweep,
        )

        markov_argv = [
            "--python-bin",
            str(python_bin),
            "--out-cmds",
            str(paths.markov_cmds),
            "--out-manifest",
            str(paths.markov_manifest),
            "--output-root",
            str(markov_out),
            "--train-docs",
            str(markov_train),
            "--test-docs",
            str(int(markov_test_docs)),
            "--audit-fractions",
            str(markov_audit),
            "--model-family",
            "additive",
            "--c3-audit-strategies",
            "uniform",
            "--leaf-query-rates",
            "1.0",
            "--root-weights",
            "1.0",
            "--schedule-consistency-weights",
            "0.0",
            "--seeds",
            str(markov_seeds),
            "--n-epochs",
            str(int(markov_n_epochs)),
            "--device",
            str(markov_device),
            "--torch-threads",
            str(int(torch_threads)),
            skip_flag,
        ]
        if markov_cuda_device is not None:
            markov_argv.extend(["--cuda-device", str(int(markov_cuda_device))])
        _markov_sweep(markov_argv)
        cmd_sources["markov_additive"] = paths.markov_cmds
        manifest_sources["markov_additive"] = paths.markov_manifest

    # ----------------------------
    # Merge sim commands + manifest
    # ----------------------------
    all_cmds: List[str] = []
    counts: Dict[str, int] = {}
    all_runs: List[RunSpec] = []
    for key, cmd_path in cmd_sources.items():
        lines = read_cmds_file(cmd_path) if cmd_path.exists() else []
        counts[key] = int(len(lines))
        all_cmds.extend(lines)
        mf_path = manifest_sources.get(key)
        if mf_path is not None and mf_path.exists():
            all_runs.extend(read_manifest_jsonl(mf_path))

    write_text(paths.suite_cmds, "\n".join(all_cmds) + ("\n" if all_cmds else ""))
    write_manifest_jsonl(paths.suite_manifest, all_runs)

    # ----------------------------
    # Plot/report commands
    # ----------------------------
    plot_cmds: List[str] = []
    plot_cmds.append(
        " ".join(
            [
                str(python_bin),
                "-m",
                "src.ctreepo.cli",
                "sim",
                "plot",
                "segment-lda-ops-grid",
                "--input-glob",
                _q(f"{seg_out}/**/*seed_*.json"),
                "--audit-strategy",
                "random",
                "--topic-phi-estimator",
                "true",
                "--output-figure",
                _q(figures_root / "segment_lda_ops_weight_recovery_grid_true.png"),
                "--output-json",
                _q(figures_root / "segment_lda_ops_weight_recovery_grid_true_report.json"),
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                str(python_bin),
                "-m",
                "src.ctreepo.cli",
                "sim",
                "plot",
                "segment-lda-oracle-gap",
                "--input-glob",
                _q(f"{seg_out}/**/*seed_*.json"),
                "--topic-phi-estimator",
                "true",
                "--aggregate",
                "median",
                "--output-figure",
                _q(figures_root / "segment_lda_oracle_gap_focus_true.png"),
                "--output-json",
                _q(figures_root / "segment_lda_oracle_gap_focus_true_report.json"),
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                str(python_bin),
                "-m",
                "src.ctreepo.cli",
                "sim",
                "plot",
                "segment-lda-ops-ceilings",
                "--input-glob",
                _q(f"{seg_out}/**/*seed_*.json"),
                "--audit-strategy",
                "random",
                "--topic-phi-estimator",
                "true",
                "--aggregate",
                "median",
                "--band",
                "p10_p90",
                "--output-figure",
                _q(figures_root / "segment_lda_ops_weight_recovery_ceilings_true.png"),
                "--output-json",
                _q(figures_root / "segment_lda_ops_weight_recovery_ceilings_true_report.json"),
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                str(python_bin),
                "-m",
                "src.ctreepo.cli",
                "sim",
                "plot",
                "ctreepo-guidance-frontier",
                "--input-glob",
                _q(f"{ctree_out}/**/*.json"),
                "--train-docs",
                str(int(ctree_focus_train)),
                "--aggregate",
                "median",
                "--output-figure",
                _q(figures_root / f"ctreepo_guidance_frontier_focus_train{int(ctree_focus_train)}.png"),
                "--output-json",
                _q(figures_root / f"ctreepo_guidance_frontier_focus_train{int(ctree_focus_train)}_report.json"),
            ]
        )
    )
    if bool(include_embedding_estimator):
        plot_cmds.append(
            " ".join(
                [
                    str(python_bin),
                    "-m",
                    "src.ctreepo.cli",
                    "sim",
                    "plot",
                    "segment-lda-ops-ceilings",
                    "--input-glob",
                    _q(f"{seg_out}/**/*seed_*.json"),
                    "--audit-strategy",
                    "random",
                    "--topic-phi-estimator",
                    "embedding_spectral",
                    "--aggregate",
                    "median",
                    "--band",
                    "p10_p90",
                    "--output-figure",
                    _q(figures_root / "segment_lda_ops_weight_recovery_ceilings_embedding_spectral.png"),
                    "--output-json",
                    _q(figures_root / "segment_lda_ops_weight_recovery_ceilings_embedding_spectral_report.json"),
                ]
            )
        )

    plot_cmds.append(
        " ".join(
            [
                str(python_bin),
                "-m",
                "src.ctreepo.cli",
                "sim",
                "plot",
                "segmented-lda-ctreepo-phase",
                "--input-glob",
                _q(f"{ctree_out}/**/*.json"),
                "--metric",
                "decomposition_total_root_l1_mean",
                "--aggregate",
                "median",
                "--output-figure",
                _q(figures_root / "segmented_lda_ctreepo_phase.png"),
                "--output-json",
                _q(figures_root / "segmented_lda_ctreepo_phase_report.json"),
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                str(python_bin),
                "-m",
                "src.ctreepo.cli",
                "sim",
                "plot",
                "segmented-lda-ctreepo-ceilings",
                "--input-glob",
                _q(f"{ctree_out}/**/*.json"),
                "--aggregate",
                "median",
                "--band",
                "p10_p90",
                "--output-figure",
                _q(figures_root / "segmented_lda_ctreepo_ceilings.png"),
                "--output-json",
                _q(figures_root / "segmented_lda_ctreepo_ceilings_report.json"),
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                str(python_bin),
                "-m",
                "src.ctreepo.cli",
                "sim",
                "plot",
                "ctreepo-guidance-frontier",
                "--input-glob",
                _q(f"{ctree_out}/**/*.json"),
                "--aggregate",
                "median",
                "--output-figure",
                _q(figures_root / "ctreepo_guidance_frontier.png"),
                "--output-json",
                _q(figures_root / "ctreepo_guidance_frontier_report.json"),
            ]
        )
    )

    markov_glob = f"{markov_out}/**/*seed_*.json" if bool(include_markov) else f"{output_root}/_none_markov/**/*.json"
    plot_cmds.append(
        " ".join(
            [
                str(python_bin),
                "-m",
                "src.ctreepo.cli",
                "sim",
                "plot",
                "full-budget-gap-suite",
                "--markov-glob",
                _q(markov_glob),
                "--segment-glob",
                _q(f"{seg_out}/**/*seed_*.json"),
                "--ctree-glob",
                _q(f"{ctree_out}/**/*.json"),
                "--aggregate",
                "median",
                "--output-figure",
                _q(figures_root / "full_budget_gap_suite.png"),
                "--output-json",
                _q(figures_root / "full_budget_gap_suite_report.json"),
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                str(python_bin),
                "-m",
                "src.ctreepo.cli",
                "sim",
                "report",
                "identifiable-zero",
                "--output-root",
                _q(output_root),
                "--emit-pdf",
            ]
        )
    )
    write_text(paths.suite_plot_cmds, "\n".join(plot_cmds) + ("\n" if plot_cmds else ""))

    group_cmd_files = {k: str(v) for k, v in cmd_sources.items()}
    group_manifest_files = {k: str(v) for k, v in manifest_sources.items()}
    meta = build_suite_meta(
        suite_name="identifiable-zero",
        suite_role="paper",
        run_id=str(run_id),
        profile=str(profile),
        policy=policy.to_dict(),
        python_bin=str(python_bin),
        output_root=output_root,
        cmds_file=paths.suite_cmds,
        manifest_file=paths.suite_manifest,
        selected_groups=list(group_manifest_files),
        group_cmd_files=group_cmd_files,
        group_manifest_files=group_manifest_files,
        group_families={k: k for k in group_manifest_files},
        extra={
            "skip_existing": bool(skip_existing),
            "include_markov": bool(include_markov),
            "include_embedding_estimator": bool(include_embedding_estimator),
            "figures_root": str(figures_root),
            "plot_cmds_file": str(paths.suite_plot_cmds),
            "suite_manifest_file": str(paths.suite_manifest),
            "counts_by_family": counts,
            "n_sim_commands_total": int(len(all_cmds)),
            "n_plot_commands_total": int(len(plot_cmds)),
            "builder_cmd_files": group_cmd_files,
            "builder_manifest_files": group_manifest_files,
            "segment_test_docs": int(segment_test_docs),
            "ctree_test_books": int(ctree_test_books),
            "markov_test_docs": int(markov_test_docs),
            "markov_n_epochs": int(markov_n_epochs),
            "segment_device": str(segment_device),
            "segment_cuda_device": int(segment_cuda_device) if segment_cuda_device is not None else None,
            "ctree_device": str(ctree_device),
            "ctree_cuda_device": int(ctree_cuda_device) if ctree_cuda_device is not None else None,
            "markov_device": str(markov_device),
            "markov_cuda_device": int(markov_cuda_device) if markov_cuda_device is not None else None,
            "torch_threads": int(torch_threads),
        },
    )
    write_suite_meta(paths.suite_meta, meta)
    return meta


def _add_build_args(p: argparse.ArgumentParser, *, require_output_root: bool) -> None:
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--profile", choices=["smoke", "paper", "walk_long"], default="walk_long")
    p.add_argument("--python-bin", type=str, default="")

    p.add_argument(
        "--output-root",
        type=str,
        default="" if not require_output_root else None,
        required=bool(require_output_root),
        help="Default: outputs/identifiable_zero_suite_<run_id>",
    )
    p.add_argument(
        "--figures-root",
        type=str,
        default="",
        help="Default: <output-root>/figures",
    )

    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--include-markov",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, include Markov additive-family runs.",
    )
    p.add_argument(
        "--include-embedding-estimator",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, include Segment-LDA estimator=embedding_spectral in addition to true.",
    )

    p.add_argument("--segment-test-docs", type=int, default=4000)
    p.add_argument("--ctree-test-books", type=int, default=4000)
    p.add_argument("--markov-test-docs", type=int, default=2000)
    p.add_argument("--markov-n-epochs", type=int, default=10)
    p.add_argument("--segment-device", type=str, default="auto")
    p.add_argument("--segment-cuda-device", type=int, default=None)
    p.add_argument("--ctree-device", type=str, default="auto")
    p.add_argument("--ctree-cuda-device", type=int, default=None)
    p.add_argument("--markov-device", type=str, default="auto")
    p.add_argument("--markov-cuda-device", type=int, default=None)
    p.add_argument("--torch-threads", type=int, default=1)


def _resolve_output_root(*, run_id: str, output_root: str) -> Path:
    from treepo._research.ctreepo.sim.suite.common import resolve_output_root
    return resolve_output_root(run_id=run_id, output_root=output_root, default_prefix="identifiable_zero_suite")


def _resolve_figures_root(*, figures_root: str, output_root: Path) -> Path:
    from treepo._research.ctreepo.sim.suite.common import resolve_figures_root
    return resolve_figures_root(figures_root=figures_root, output_root=output_root)


def _resolve_python_bin(python_bin: str) -> str:
    if str(python_bin).strip():
        return str(python_bin).strip()
    # Avoid hardcoding venv paths; default to the current interpreter.
    import sys

    return sys.executable


def _ensure_built(ns: argparse.Namespace) -> Tuple[SuitePaths, dict]:
    run_id = utc_run_id(getattr(ns, "run_id", ""))
    output_root = _resolve_output_root(run_id=run_id, output_root=getattr(ns, "output_root", ""))
    figures_root = _resolve_figures_root(figures_root=getattr(ns, "figures_root", ""), output_root=output_root)
    paths = _resolve_paths(output_root=output_root, figures_root=figures_root)
    rebuild = bool(getattr(ns, "rebuild", False))
    if (
        rebuild
        or not paths.suite_meta.exists()
        or not paths.suite_manifest.exists()
        or not paths.suite_cmds.exists()
        or not paths.suite_plot_cmds.exists()
    ):
        meta = build_suite(
            run_id=run_id,
            profile=str(getattr(ns, "profile", "walk_long")),
            python_bin=_resolve_python_bin(str(getattr(ns, "python_bin", ""))),
            output_root=output_root,
            figures_root=figures_root,
            skip_existing=bool(getattr(ns, "skip_existing", True)),
            include_markov=bool(getattr(ns, "include_markov", False)),
            include_embedding_estimator=bool(getattr(ns, "include_embedding_estimator", False)),
            segment_test_docs=int(getattr(ns, "segment_test_docs", 4000)),
            ctree_test_books=int(getattr(ns, "ctree_test_books", 4000)),
            markov_test_docs=int(getattr(ns, "markov_test_docs", 2000)),
            markov_n_epochs=int(getattr(ns, "markov_n_epochs", 10)),
            segment_device=str(getattr(ns, "segment_device", "auto")),
            segment_cuda_device=getattr(ns, "segment_cuda_device", None),
            ctree_device=str(getattr(ns, "ctree_device", "auto")),
            ctree_cuda_device=getattr(ns, "ctree_cuda_device", None),
            markov_device=str(getattr(ns, "markov_device", "auto")),
            markov_cuda_device=getattr(ns, "markov_cuda_device", None),
            torch_threads=int(getattr(ns, "torch_threads", 1)),
        )
        return paths, meta
    return paths, read_suite_meta(paths.suite_meta)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Identifiable-Zero suite orchestration.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build suite command lists and metadata.")
    _add_build_args(b, require_output_root=False)

    r = sub.add_parser("run", help="Execute the suite command list.")
    _add_build_args(r, require_output_root=False)
    r.add_argument("--rebuild", action=argparse.BooleanOptionalAction, default=False)
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--gpu-tokens", type=str, default="auto")
    r.add_argument("--log-dir", type=str, default="")
    r.add_argument("--set-thread-env", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)

    pl = sub.add_parser("plot", help="Execute suite plotting command list.")
    _add_build_args(pl, require_output_root=False)
    pl.add_argument("--rebuild", action=argparse.BooleanOptionalAction, default=False)
    pl.add_argument("--jobs", type=int, default=1)
    pl.add_argument("--log-dir", type=str, default="")
    pl.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)

    rep = sub.add_parser("report", help="Generate suite report (markdown/PDF).")
    rep.add_argument("--output-root", type=str, required=True)
    rep.add_argument("--output-markdown", type=str, default="")
    rep.add_argument("--output-pdf", type=str, default="")
    rep.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    ns = _build_parser().parse_args(list(argv) if argv is not None else None)

    if ns.cmd == "build":
        run_id = utc_run_id(ns.run_id)
        output_root = _resolve_output_root(run_id=run_id, output_root=str(ns.output_root or ""))
        figures_root = _resolve_figures_root(figures_root=str(ns.figures_root), output_root=output_root)
        meta = build_suite(
            run_id=run_id,
            profile=str(ns.profile),
            python_bin=_resolve_python_bin(str(ns.python_bin)),
            output_root=output_root,
            figures_root=figures_root,
            skip_existing=bool(ns.skip_existing),
            include_markov=bool(ns.include_markov),
            include_embedding_estimator=bool(ns.include_embedding_estimator),
            segment_test_docs=int(ns.segment_test_docs),
            ctree_test_books=int(ns.ctree_test_books),
            markov_test_docs=int(ns.markov_test_docs),
            markov_n_epochs=int(ns.markov_n_epochs),
            segment_device=str(ns.segment_device),
            segment_cuda_device=ns.segment_cuda_device,
            ctree_device=str(ns.ctree_device),
            ctree_cuda_device=ns.ctree_cuda_device,
            markov_device=str(ns.markov_device),
            markov_cuda_device=ns.markov_cuda_device,
            torch_threads=int(ns.torch_threads),
        )
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0

    if ns.cmd == "run":
        paths, meta = _ensure_built(ns)
        if bool(ns.fail_fast):
            from treepo._research.ctreepo.sim.cli.exec_cmds import main as _exec_main  # noqa: WPS433

            exec_argv: List[str] = ["--cmds", str(paths.suite_cmds), "--jobs", str(int(ns.jobs))]
            if str(ns.log_dir).strip():
                exec_argv.extend(["--log-dir", str(ns.log_dir)])
            exec_argv.append("--fail-fast")
            return int(_exec_main(exec_argv))

        log_dir = Path(ns.log_dir).resolve() if str(ns.log_dir).strip() else (paths.output_root / "queue_logs")
        queue_payload = run_manifest_queue_suite(
            manifest_paths=[paths.suite_manifest],
            cpu_workers=int(ns.jobs),
            gpu_tokens=str(ns.gpu_tokens),
            log_dir=log_dir,
            set_thread_env=bool(ns.set_thread_env),
        )
        payload = {
            "output_root": str(paths.output_root),
            "profile": str(meta.get("profile", "")),
            **queue_payload,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if int(queue_payload["summary"].get("n_fail", 0)) == 0 else 1

    if ns.cmd == "plot":
        paths, _meta = _ensure_built(ns)
        from treepo._research.ctreepo.sim.cli.exec_cmds import main as _exec_main  # noqa: WPS433

        exec_argv = ["--cmds", str(paths.suite_plot_cmds), "--jobs", str(int(ns.jobs))]
        if str(ns.log_dir).strip():
            exec_argv.extend(["--log-dir", str(ns.log_dir)])
        if bool(ns.fail_fast):
            exec_argv.append("--fail-fast")
        return int(_exec_main(exec_argv))

    if ns.cmd == "report":
        from treepo._research.ctreepo.sim.cli.report.identifiable_zero_suite import (  # noqa: WPS433
            main as _report_main,
        )

        argv_out: List[str] = ["--output-root", str(Path(ns.output_root).resolve())]
        if str(ns.output_markdown).strip():
            argv_out.extend(["--output-markdown", str(Path(ns.output_markdown).resolve())])
        if str(ns.output_pdf).strip():
            argv_out.extend(["--output-pdf", str(Path(ns.output_pdf).resolve())])
        argv_out.append("--emit-pdf" if bool(ns.emit_pdf) else "--no-emit-pdf")
        return int(_report_main(argv_out))

    raise ValueError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
