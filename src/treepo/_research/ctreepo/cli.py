from __future__ import annotations

import argparse
import importlib
import sys
from typing import Sequence

from treepo._research.ctreepo.sim.suite.registry import suite_module_names, suite_module_spec


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ctreepo", description="C-TreePO simulations and tooling.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sim = sub.add_parser("sim", help="Simulation families, sweeps, plots, and suites.")
    sim_sub = sim.add_subparsers(dest="sim_cmd", required=True)

    run = sim_sub.add_parser("run", help="Run a single simulation instance.")
    run.add_argument(
        "family",
        choices=[
            "markov-ops-count",
            "contextual-sbijax",
            "segment-lda-ops",
            "segmented-lda-ctreepo",
            "tensor-lda-books",
        ],
    )
    run.add_argument("args", nargs=argparse.REMAINDER)

    sweep = sim_sub.add_parser("sweep", help="Build (and optionally execute) sweep commands.")
    sweep.add_argument(
        "family",
        choices=[
            "markov-ops-count",
            "segment-lda-ops",
            "segmented-lda-ctreepo",
            "tensor-lda-books",
        ],
    )
    sweep.add_argument("args", nargs=argparse.REMAINDER)

    exec_p = sim_sub.add_parser("exec", help="Execute a command list with bounded parallelism.")
    exec_p.add_argument("args", nargs=argparse.REMAINDER)

    plot = sim_sub.add_parser("plot", help="Run plot utilities.")
    plot.add_argument(
        "name",
        choices=[
            "segment-lda-ops-grid",
            "segment-lda-oracle-gap",
            "segment-lda-ops-ceilings",
            "segmented-lda-ctreepo-phase",
            "segmented-lda-ctreepo-ceilings",
            "ctreepo-guidance-frontier",
            "full-budget-gap-suite",
        ],
    )
    plot.add_argument("args", nargs=argparse.REMAINDER)

    report = sim_sub.add_parser("report", help="Generate reports from sweep outputs.")
    report.add_argument("name", choices=["identifiable-zero", "publication-ctreepo-progress"])
    report.add_argument("args", nargs=argparse.REMAINDER)

    suite = sim_sub.add_parser("suite", help="Curated multi-family suites.")
    suite.add_argument(
        "suite_name",
        choices=suite_module_names(),
        help="Suite name.",
    )
    suite.add_argument("suite_cmd", help="Suite subcommand.")
    suite.add_argument("args", nargs=argparse.REMAINDER)

    return p


def _dispatch_run(family: str, argv: Sequence[str]) -> int:
    if family == "markov-ops-count":
        from treepo._research.ctreepo.sim.cli.run_markov_changepoint_ops_count import main as _main

        return int(_main(argv))
    if family == "contextual-sbijax":
        from treepo._research.ctreepo.sim.cli.probe_contextual_sbijax import main as _main

        return int(_main(argv))
    if family == "segment-lda-ops":
        from treepo._research.ctreepo.sim.cli.run_segment_lda_ops_weight_recovery import main as _main

        return int(_main(argv))
    if family == "segmented-lda-ctreepo":
        from treepo._research.ctreepo.sim.cli.run_segmented_lda_ctreepo import main as _main

        return int(_main(argv))
    if family == "tensor-lda-books":
        from treepo._research.ctreepo.sim.cli.run_tensor_lda_book_benchmark import main as _main

        return int(_main(argv))
    raise ValueError(f"Unknown family: {family}")


def _dispatch_sweep(family: str, argv: Sequence[str]) -> int:
    if family == "markov-ops-count":
        from treepo._research.ctreepo.sim.cli.sweep_markov_changepoint_ops_count import main as _main

        return int(_main(argv))
    if family == "segment-lda-ops":
        from treepo._research.ctreepo.sim.cli.sweep_segment_lda_ops_weight_recovery import main as _main

        return int(_main(argv))
    if family == "segmented-lda-ctreepo":
        from treepo._research.ctreepo.sim.cli.sweep_segmented_lda_ctreepo import main as _main

        return int(_main(argv))
    if family == "tensor-lda-books":
        from treepo._research.ctreepo.sim.cli.sweep_tensor_lda_book_benchmark import main as _main

        return int(_main(argv))
    raise ValueError(f"Unknown family: {family}")


def _dispatch_plot(name: str, argv: Sequence[str]) -> int:
    if name == "segment-lda-ops-grid":
        from treepo._research.ctreepo.sim.cli.plot.segment_lda_ops_weight_recovery_grid import main as _main

        return int(_main(argv))
    if name == "segment-lda-oracle-gap":
        from treepo._research.ctreepo.sim.cli.plot.segment_lda_oracle_gap_focus import main as _main

        return int(_main(argv))
    if name == "segment-lda-ops-ceilings":
        from treepo._research.ctreepo.sim.cli.plot.segment_lda_ops_weight_recovery_ceilings import main as _main

        return int(_main(argv))
    if name == "segmented-lda-ctreepo-phase":
        from treepo._research.ctreepo.sim.cli.plot.segmented_lda_ctreepo_phase import main as _main

        return int(_main(argv))
    if name == "segmented-lda-ctreepo-ceilings":
        from treepo._research.ctreepo.sim.cli.plot.segmented_lda_ctreepo_ceilings import main as _main

        return int(_main(argv))
    if name == "ctreepo-guidance-frontier":
        from treepo._research.ctreepo.sim.cli.plot.ctreepo_guidance_frontier import main as _main

        return int(_main(argv))
    if name == "full-budget-gap-suite":
        from treepo._research.ctreepo.sim.cli.plot.full_budget_gap_suite import main as _main

        return int(_main(argv))
    raise ValueError(f"Unknown plot name: {name}")


def _dispatch_report(name: str, argv: Sequence[str]) -> int:
    if name == "identifiable-zero":
        from treepo._research.ctreepo.sim.cli.report.identifiable_zero_suite import main as _main

        return int(_main(argv))
    if name == "publication-ctreepo-progress":
        from treepo._research.ctreepo.sim.cli.report.publication_ctreepo_progress import main as _main

        return int(_main(argv))
    raise ValueError(f"Unknown report name: {name}")


def _dispatch_suite(suite_name: str, cmd: str, argv: Sequence[str]) -> int:
    spec = suite_module_spec(str(suite_name))
    module = importlib.import_module(str(spec.module))
    main_fn = getattr(module, "main")
    return int(main_fn([cmd, *argv]))


def main(argv: Sequence[str] | None = None) -> int:
    args_in = list(sys.argv[1:] if argv is None else argv)
    p = _build_parser()
    ns, extras = p.parse_known_args(args_in)
    if extras:
        if hasattr(ns, "args"):
            ns.args = list(getattr(ns, "args", []) or []) + list(extras)
        else:
            p.error(f"unrecognized arguments: {' '.join(extras)}")

    if ns.cmd == "sim":
        if ns.sim_cmd == "run":
            return _dispatch_run(str(ns.family), list(ns.args))
        if ns.sim_cmd == "sweep":
            return _dispatch_sweep(str(ns.family), list(ns.args))
        if ns.sim_cmd == "exec":
            from treepo._research.ctreepo.sim.cli.exec_cmds import main as _main

            return int(_main(list(ns.args)))
        if ns.sim_cmd == "plot":
            return _dispatch_plot(str(ns.name), list(ns.args))
        if ns.sim_cmd == "report":
            return _dispatch_report(str(ns.name), list(ns.args))
        if ns.sim_cmd == "suite":
            return _dispatch_suite(str(ns.suite_name), str(ns.suite_cmd), list(ns.args))

    raise ValueError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
