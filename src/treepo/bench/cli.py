from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from treepo.bench.io import dump_json, load_yaml_or_json
from treepo.bench.runner import (
    VALID_EXPERIMENTS,
    emit_commands,
    run_single,
    run_specs,
    run_sweep,
)
from treepo.bench.suites.cardinality import build_cardinality_paper_suite
from treepo.bench.suites.classical_sketches import build_classical_sketches_suite
from treepo.bench.suites.identifiable_zero import (
    build_identifiable_zero_dtm_lda,
    build_identifiable_zero_lda_leafnoise,
    build_identifiable_zero_publication_ctreepo,
)
from treepo.bench.suites.paper import build_paper_grids_suite, build_paper_smoke_suite
from treepo.release import audit_launch_gate, audit_migration_inventory, audit_package_hygiene


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="treepo-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------
    # run
    # ------------------------------------------------------------
    p_run = sub.add_parser("run", help="Run a single experiment and write JSON/CSV.")
    p_run.add_argument("experiment", choices=list(VALID_EXPERIMENTS))
    p_run.add_argument("--config", type=Path, required=True)
    p_run.add_argument("--json-out", type=Path, required=True)
    p_run.add_argument("--csv-out", type=Path, required=True)
    p_run.add_argument("--print-json", action="store_true", default=False)

    # ------------------------------------------------------------
    # sweep
    # ------------------------------------------------------------
    p_sweep = sub.add_parser("sweep", help="Run a grid sweep defined by a YAML/JSON spec.")
    p_sweep.add_argument("experiment", choices=list(VALID_EXPERIMENTS))
    p_sweep.add_argument("--spec", type=Path, required=True)
    p_sweep.add_argument("--out-root", type=Path, required=True)
    p_sweep.add_argument("--jobs", type=int, required=True)
    p_sweep.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    p_sweep.add_argument("--emit-commands", type=Path, default=None)
    p_sweep.add_argument("--commands-only", action=argparse.BooleanOptionalAction, default=False)

    # ------------------------------------------------------------
    # suite
    # ------------------------------------------------------------
    p_suite = sub.add_parser("suite", help="Run a named benchmark suite.")
    p_suite.add_argument(
        "suite",
        choices=[
            "identifiable-zero-dtm-lda",
            "identifiable-zero-lda-leafnoise",
            "identifiable-zero-publication-ctreepo",
            "cardinality-paper",
            "classical-sketches",
            "paper-smoke",
            "paper-grids",
        ],
    )
    p_suite.add_argument("--out-root", type=Path, required=True)
    p_suite.add_argument("--jobs", type=int, required=True)
    p_suite.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    p_suite.add_argument("--emit-commands", type=Path, default=None)
    p_suite.add_argument("--commands-only", action=argparse.BooleanOptionalAction, default=False)
    p_suite.add_argument("--seeds", type=str, default=None)
    p_suite.add_argument("--topic-phi-estimators", type=str, default=None)
    p_suite.add_argument("--leaf-counts", type=str, default=None)
    p_suite.add_argument("--leaf-sizes", type=str, default=None)
    p_suite.add_argument("--capacities", type=str, default=None)
    p_suite.add_argument(
        "--execution-backend",
        choices=["unified_g", "treepo"],
        default="unified_g",
        help="Execution path for classical-sketches; unified_g routes through fit().",
    )
    p_suite.add_argument("--include-learned", action=argparse.BooleanOptionalAction, default=False)
    p_suite.add_argument("--learned-targets", type=str, default=None)
    p_suite.add_argument("--learned-variants", type=str, default=None)
    p_suite.add_argument("--learned-readout-archs", type=str, default=None)
    p_suite.add_argument("--learned-epochs", type=int, default=150)
    p_suite.add_argument("--learned-n-train", type=int, default=128)
    p_suite.add_argument("--learned-n-val", type=int, default=48)
    p_suite.add_argument("--learned-batch-size", type=int, default=1024)
    p_suite.add_argument(
        "--learned-target-jobs",
        type=str,
        default="auto",
        help=(
            "Concurrent learned target chains. 'auto' uses four workers per "
            "visible GPU by default; override the multiplier with "
            "TREEPO_LEARNED_TARGET_JOBS_PER_GPU."
        ),
    )
    p_suite.add_argument("--learned-gpu-ids", type=str, default="auto")
    p_suite.add_argument("--learned-batch-reference-leaf-size", type=int, default=128)
    p_suite.add_argument("--learned-max-batch-size", type=int, default=8192)
    p_suite.add_argument("--learned-eval-every", type=int, default=25)
    p_suite.add_argument(
        "--learned-local-label-rates",
        type=str,
        default=None,
        help="Comma-separated R-grid rates applied to both learned leaf and internal local labels, e.g. 0.1,0.2,...,1.0.",
    )
    p_suite.add_argument(
        "--learned-leaf-query-rates",
        type=str,
        default=None,
        help="Comma-separated learned leaf-label rates. Overrides --learned-local-label-rates for leaves.",
    )
    p_suite.add_argument(
        "--learned-root-query-rates",
        type=str,
        default=None,
        help="Comma-separated learned document-root label rates. Defaults to 1.0 for separate axes and to the node R for uniform_all_nodes.",
    )
    p_suite.add_argument(
        "--learned-internal-query-rates",
        type=str,
        default=None,
        help="Comma-separated learned internal-node label rates. Overrides --learned-local-label-rates for internal nodes.",
    )
    p_suite.add_argument(
        "--learned-supervision-sampling-policy",
        choices=("separate_axes", "uniform_all_nodes"),
        default="separate_axes",
        help="How learned local labels are sampled: separate leaf/internal axes or one uniform root+leaf+internal node pool.",
    )
    p_suite.add_argument("--include-runtime", action=argparse.BooleanOptionalAction, default=True)

    # ------------------------------------------------------------
    # report
    # ------------------------------------------------------------
    p_report = sub.add_parser("report", help="Generate a report from existing outputs.")
    rep = p_report.add_subparsers(dest="report", required=True)

    p_rep_leaf = rep.add_parser("lda-leafnoise", help="Leaf-noise progression report (LDA baseline).")
    p_rep_leaf.add_argument("--output-root", type=Path, required=True)
    p_rep_leaf.add_argument("--ctreepo-root", type=Path, default=None)
    p_rep_leaf.add_argument("--out-dir", type=Path, default=None)
    p_rep_leaf.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)

    p_rep_pub = rep.add_parser("publication-progress", help="Interim progress plots for publication suite.")
    p_rep_pub.add_argument("--output-root", type=Path, required=True)
    p_rep_pub.add_argument("--out-dir", type=Path, default=None)
    p_rep_pub.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)

    p_rep_learned_g = rep.add_parser("learned-g-overnight", help="Progress report for learned-g overnight runs.")
    p_rep_learned_g.add_argument("--output-root", type=Path, required=True)
    p_rep_learned_g.add_argument("--out-dir", type=Path, default=None)
    p_rep_learned_g.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)

    p_rep_card = rep.add_parser("cardinality", help="Cardinality/HLL report and figures.")
    p_rep_card.add_argument("--output-root", type=Path, required=True)
    p_rep_card.add_argument("--out-dir", type=Path, default=None)
    p_rep_card.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)

    p_rep_classical = rep.add_parser("classical-sketches", help="Classical sketch comparison report.")
    p_rep_classical.add_argument("--output-root", type=Path, required=True)
    p_rep_classical.add_argument("--out-dir", type=Path, default=None)
    p_rep_classical.add_argument("--tables-dir", type=Path, default=Path("paper/ctreepo/tables"))
    p_rep_classical.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=False)

    # ------------------------------------------------------------
    # check
    # ------------------------------------------------------------
    p_check = sub.add_parser("check", help="Run package release checks.")
    check_sub = p_check.add_subparsers(dest="check", required=True)
    p_check_inv = check_sub.add_parser("inventory", help="Validate the migration inventory.")
    p_check_inv.add_argument("--json", action="store_true", default=False)
    p_check_hygiene = check_sub.add_parser("hygiene", help="Validate package import/hygiene rules.")
    p_check_hygiene.add_argument("--json", action="store_true", default=False)
    p_check_launch = check_sub.add_parser("launch", help="Run the GitHub source-release launch gate.")
    p_check_launch.add_argument("--json", action="store_true", default=False)

    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.cmd == "run":
        payload = load_yaml_or_json(Path(args.config))
        if not isinstance(payload, dict):
            raise SystemExit("--config must contain a JSON/YAML mapping")
        res = run_single(
            experiment=str(args.experiment),
            config=payload,
            json_out=Path(args.json_out),
            csv_out=Path(args.csv_out),
            print_json=bool(args.print_json),
        )
        if not bool(args.print_json):
            print(dump_json(res))
        return 0

    if args.cmd == "sweep":
        res = run_sweep(
            experiment=str(args.experiment),
            spec_path=Path(args.spec),
            out_root=Path(args.out_root),
            jobs=int(args.jobs),
            skip_existing=bool(args.skip_existing),
            emit_commands_path=Path(args.emit_commands) if args.emit_commands is not None else None,
            commands_only=bool(args.commands_only),
        )
        print(dump_json({"results": res, "n_results": len(res)}))
        return 0

    if args.cmd == "suite":
        suite = str(args.suite)
        skip = bool(args.skip_existing)
        out_root = Path(args.out_root)

        if suite == "identifiable-zero-dtm-lda":
            specs = build_identifiable_zero_dtm_lda(
                out_root=out_root,
                skip_existing=skip,
                seeds=args.seeds,
                topic_phi_estimators=args.topic_phi_estimators,
            )
        elif suite == "identifiable-zero-lda-leafnoise":
            specs = build_identifiable_zero_lda_leafnoise(out_root=out_root, skip_existing=skip, seeds=args.seeds)
        elif suite == "identifiable-zero-publication-ctreepo":
            specs = build_identifiable_zero_publication_ctreepo(out_root=out_root, skip_existing=skip, seeds=args.seeds)
        elif suite == "cardinality-paper":
            specs = build_cardinality_paper_suite(out_root=out_root, skip_existing=skip, seeds=args.seeds)
        elif suite == "classical-sketches":
            specs = build_classical_sketches_suite(
                out_root=out_root,
                skip_existing=skip,
                seeds=args.seeds,
                leaf_counts=args.leaf_counts,
                leaf_sizes=args.leaf_sizes,
                capacities=args.capacities,
                execution_backend=args.execution_backend,
                include_learned=bool(args.include_learned),
                learned_targets=args.learned_targets,
                learned_variants=args.learned_variants,
                learned_readout_archs=args.learned_readout_archs,
                learned_n_epochs=int(args.learned_epochs),
                learned_n_train=int(args.learned_n_train),
                learned_n_val=int(args.learned_n_val),
                learned_batch_size=int(args.learned_batch_size),
                learned_target_jobs=args.learned_target_jobs,
                learned_gpu_ids=args.learned_gpu_ids,
                learned_batch_reference_leaf_size=int(args.learned_batch_reference_leaf_size),
                learned_max_batch_size=int(args.learned_max_batch_size),
                learned_eval_every_n_epochs=int(args.learned_eval_every),
                learned_local_label_rates=args.learned_local_label_rates,
                learned_root_query_rates=args.learned_root_query_rates,
                learned_leaf_query_rates=args.learned_leaf_query_rates,
                learned_internal_query_rates=args.learned_internal_query_rates,
                learned_supervision_sampling_policy=args.learned_supervision_sampling_policy,
            )
        elif suite == "paper-smoke":
            specs = build_paper_smoke_suite(out_root=out_root, skip_existing=skip)
        elif suite == "paper-grids":
            specs = build_paper_grids_suite(
                out_root=out_root,
                skip_existing=skip,
                seeds=args.seeds,
                topic_phi_estimators=args.topic_phi_estimators,
                leaf_counts=args.leaf_counts,
                leaf_sizes=args.leaf_sizes,
                capacities=args.capacities,
                execution_backend=args.execution_backend,
                include_learned=bool(args.include_learned),
                learned_targets=args.learned_targets,
                learned_variants=args.learned_variants,
                learned_n_epochs=int(args.learned_epochs),
                learned_n_train=int(args.learned_n_train),
                learned_n_val=int(args.learned_n_val),
                include_runtime=bool(args.include_runtime),
            )
        else:  # pragma: no cover
            raise SystemExit(f"unknown suite: {suite}")

        if args.emit_commands is not None:
            emit_commands(specs, out_path=Path(args.emit_commands))
        if args.commands_only:
            print(dump_json({"status": "commands_only", "n_runs": len(specs)}))
            return 0

        results = run_specs(specs, jobs=int(args.jobs), skip_existing=skip)
        print(dump_json({"suite": suite, "n_runs": len(specs), "results": results}))
        return 0

    if args.cmd == "report":
        if args.report == "lda-leafnoise":
            from treepo.bench.reports import lda_leafnoise as report_lda_leafnoise

            argv2: list[str] = ["--output-root", str(Path(args.output_root))]
            if args.ctreepo_root is not None:
                argv2 += ["--ctreepo-root", str(Path(args.ctreepo_root))]
            if args.out_dir is not None:
                argv2 += ["--out-dir", str(Path(args.out_dir))]
            argv2 += ["--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf"]
            return int(report_lda_leafnoise.main(argv2))
        if args.report == "publication-progress":
            from treepo.bench.reports import publication_progress as report_publication_progress

            argv2 = ["--output-root", str(Path(args.output_root))]
            if args.out_dir is not None:
                argv2 += ["--out-dir", str(Path(args.out_dir))]
            argv2 += ["--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf"]
            return int(report_publication_progress.main(argv2))
        if args.report == "learned-g-overnight":
            from treepo.bench.reports import learned_g_overnight as report_learned_g_overnight

            argv2 = ["--output-root", str(Path(args.output_root))]
            if args.out_dir is not None:
                argv2 += ["--out-dir", str(Path(args.out_dir))]
            argv2 += ["--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf"]
            return int(report_learned_g_overnight.main(argv2))
        if args.report == "cardinality":
            from treepo.bench.reports import cardinality as report_cardinality

            argv2 = ["--output-root", str(Path(args.output_root))]
            if args.out_dir is not None:
                argv2 += ["--out-dir", str(Path(args.out_dir))]
            argv2 += ["--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf"]
            return int(report_cardinality.main(argv2))
        if args.report == "classical-sketches":
            from treepo.bench.reports import classical_sketches as report_classical_sketches

            argv2 = ["--output-root", str(Path(args.output_root))]
            if args.out_dir is not None:
                argv2 += ["--out-dir", str(Path(args.out_dir))]
            if args.tables_dir is not None:
                argv2 += ["--tables-dir", str(Path(args.tables_dir))]
            argv2 += ["--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf"]
            return int(report_classical_sketches.main(argv2))
        raise SystemExit(f"unknown report: {args.report}")

    if args.cmd == "check":
        if args.check == "inventory":
            report = audit_migration_inventory()
        elif args.check == "hygiene":
            report = audit_package_hygiene()
        elif args.check == "launch":
            report = audit_launch_gate()
        else:  # pragma: no cover
            raise SystemExit(f"unknown check: {args.check}")
        if bool(args.json):
            print(dump_json(report))
        else:
            print("ok" if report.get("ok") else "FAILED")
            for failure in list(report.get("failures") or []):
                print(dump_json(failure))
        return 0 if bool(report.get("ok")) else 1

    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
