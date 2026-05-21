from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence

from treepo._research.ctreepo.sim.cli.exec_cmds import main as exec_cmds_main
from treepo._research.ctreepo.sim.runner import read_cmds_file
from treepo._research.ctreepo.sim.suite.common import (
    SuiteGroupRuns,
    build_suite_meta,
    emit_grouped_suite_artifacts,
    parse_items,
    read_suite_meta,
    resolve_grouped_suite_paths,
    run_manifest_queue_suite,
    runs_from_commands,
    select_known_items,
    utc_run_id,
    write_suite_meta,
)
from treepo._research.ctreepo.sim.suite.cpu_megasweep import main as legacy_build_main


_GROUP_ORDER = ("markov", "segment_lda_ops", "segmented_lda_ctreepo")


def build_suite(
    *,
    run_id: str,
    python_bin: str,
    output_root: Path,
    profile: str,
    requested_groups: Sequence[str],
    skip_existing: bool,
    include_neural_topic_estimators: bool,
    markov_extra_c3_strategies: bool,
    markov_n_epochs: int,
    torch_threads: int,
    markov_device: str,
    markov_cuda_device: int | None,
    segment_device: str,
    segment_cuda_device: int | None,
    ctree_device: str,
    ctree_cuda_device: int | None,
    include_full_budget_anchors: bool,
) -> Dict[str, object]:
    output_root = output_root.resolve()
    paths = resolve_grouped_suite_paths(output_root)
    legacy_dir = output_root / "_legacy_builder"
    legacy_out_cmds = legacy_dir / "cpu_megasweep_cmds.txt"
    legacy_plot_cmds = legacy_dir / "cpu_megasweep_plot_cmds.txt"
    legacy_meta = legacy_dir / "cpu_megasweep_meta.json"

    build_argv = [
        "--python-bin", str(python_bin),
        "--run-id", str(run_id),
        "--profile", str(profile),
        "--output-root", str(output_root),
        "--figures-root", str(output_root / "figures"),
        "--out-cmds", str(legacy_out_cmds),
        "--out-plot-cmds", str(legacy_plot_cmds),
        "--out-meta", str(legacy_meta),
        "--markov-n-epochs", str(int(markov_n_epochs)),
        "--torch-threads", str(int(torch_threads)),
        "--markov-device", str(markov_device),
        "--segment-device", str(segment_device),
        "--ctree-device", str(ctree_device),
        "--skip-existing" if bool(skip_existing) else "--no-skip-existing",
        "--include-neural-topic-estimators" if bool(include_neural_topic_estimators) else "--no-include-neural-topic-estimators",
        "--markov-extra-c3-strategies" if bool(markov_extra_c3_strategies) else "--no-markov-extra-c3-strategies",
        "--include-full-budget-anchors" if bool(include_full_budget_anchors) else "--no-include-full-budget-anchors",
    ]
    if markov_cuda_device is not None:
        build_argv.extend(["--markov-cuda-device", str(int(markov_cuda_device))])
    if segment_cuda_device is not None:
        build_argv.extend(["--segment-cuda-device", str(int(segment_cuda_device))])
    if ctree_cuda_device is not None:
        build_argv.extend(["--ctree-cuda-device", str(int(ctree_cuda_device))])
    legacy_build_main(build_argv)

    legacy_payload = json.loads(legacy_meta.read_text(encoding="utf-8"))
    group_cmd_map = {
        "markov": legacy_out_cmds.with_name(legacy_out_cmds.stem + "_markov.txt"),
        "segment_lda_ops": legacy_out_cmds.with_name(legacy_out_cmds.stem + "_segment_lda_ops.txt"),
        "segmented_lda_ctreepo": legacy_out_cmds.with_name(legacy_out_cmds.stem + "_segmented_lda_ctreepo.txt"),
    }
    selected_groups = select_known_items(
        requested=requested_groups,
        available=_GROUP_ORDER,
        item_name="cpu-megasweep groups",
    )
    groups: List[SuiteGroupRuns] = []
    for key in _GROUP_ORDER:
        if key not in selected_groups:
            continue
        commands = read_cmds_file(group_cmd_map[key])
        groups.append(
            SuiteGroupRuns(
                key=key,
                family=key,
                runs=runs_from_commands(commands=commands, family=key, group_key=key),
            )
        )
    artifacts = emit_grouped_suite_artifacts(paths, groups)
    plot_cmds = read_cmds_file(legacy_plot_cmds)
    (output_root / "suite_plot_cmds.txt").write_text("\n".join(plot_cmds) + ("\n" if plot_cmds else ""), encoding="utf-8")

    meta = build_suite_meta(
        suite_name="cpu-megasweep",
        suite_role="paper",
        run_id=str(run_id),
        profile=str(profile),
        policy={},
        python_bin=str(python_bin),
        output_root=output_root,
        cmds_file=paths.suite_cmds,
        manifest_file=paths.suite_manifest,
        selected_groups=selected_groups,
        group_cmd_files=artifacts.group_cmd_files,
        group_manifest_files=artifacts.group_manifest_files,
        group_families=artifacts.group_families,
        extra={
            "plot_cmds_file": str(output_root / "suite_plot_cmds.txt"),
            "legacy_builder_meta": legacy_payload,
            "counts_by_group": artifacts.counts_by_group,
            "n_commands_total": int(len(artifacts.all_cmds)),
        },
    )
    write_suite_meta(paths.suite_meta, meta)
    return meta


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU megasweep v2 suite.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("build", "run", "plot", "report"):
        subp = sub.add_parser(name)
        subp.add_argument("--run-id", type=str, default="")
        subp.add_argument("--python-bin", type=str, default="")
        subp.add_argument("--output-root", type=str, default="" if name == "build" else None, required=name != "build")
        if name in {"build", "run"}:
            subp.add_argument("--profile", choices=["smoke", "paper", "full"], default="paper")
            subp.add_argument("--groups", type=str, default="")
            subp.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
            subp.add_argument("--include-neural-topic-estimators", action=argparse.BooleanOptionalAction, default=False)
            subp.add_argument("--markov-extra-c3-strategies", action=argparse.BooleanOptionalAction, default=True)
            subp.add_argument("--markov-n-epochs", type=int, default=10)
            subp.add_argument("--torch-threads", type=int, default=1)
            subp.add_argument("--markov-device", type=str, default="auto")
            subp.add_argument("--markov-cuda-device", type=int, default=None)
            subp.add_argument("--segment-device", type=str, default="auto")
            subp.add_argument("--segment-cuda-device", type=int, default=None)
            subp.add_argument("--ctree-device", type=str, default="auto")
            subp.add_argument("--ctree-cuda-device", type=int, default=None)
            subp.add_argument("--include-full-budget-anchors", action=argparse.BooleanOptionalAction, default=True)
        if name == "run":
            subp.add_argument("--jobs", type=int, default=1)
            subp.add_argument("--gpu-tokens", type=str, default="auto")
            subp.add_argument("--log-dir", type=str, default="")
            subp.add_argument("--rebuild", action=argparse.BooleanOptionalAction, default=False)
            subp.add_argument("--set-thread-env", action=argparse.BooleanOptionalAction, default=True)
        if name in {"plot", "report"}:
            subp.add_argument("--jobs", type=int, default=1)
            subp.add_argument("--log-dir", type=str, default="")
        if name == "report":
            subp.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
            subp.add_argument("--skip-plots", action=argparse.BooleanOptionalAction, default=False)
    return parser


def _ensure_built(args: argparse.Namespace) -> tuple[Path, Dict[str, object]]:
    output_root = Path(args.output_root).resolve()
    paths = resolve_grouped_suite_paths(output_root)
    if bool(getattr(args, "rebuild", False)) or not paths.suite_meta.exists():
        build_suite(
            run_id=utc_run_id(args.run_id or output_root.name),
            python_bin=str(args.python_bin).strip() or sys.executable,
            output_root=output_root,
            profile=str(getattr(args, "profile", "paper")),
            requested_groups=parse_items(str(getattr(args, "groups", ""))),
            skip_existing=bool(getattr(args, "skip_existing", True)),
            include_neural_topic_estimators=bool(getattr(args, "include_neural_topic_estimators", False)),
            markov_extra_c3_strategies=bool(getattr(args, "markov_extra_c3_strategies", True)),
            markov_n_epochs=int(getattr(args, "markov_n_epochs", 10)),
            torch_threads=int(getattr(args, "torch_threads", 1)),
            markov_device=str(getattr(args, "markov_device", "auto")),
            markov_cuda_device=getattr(args, "markov_cuda_device", None),
            segment_device=str(getattr(args, "segment_device", "auto")),
            segment_cuda_device=getattr(args, "segment_cuda_device", None),
            ctree_device=str(getattr(args, "ctree_device", "auto")),
            ctree_cuda_device=getattr(args, "ctree_cuda_device", None),
            include_full_budget_anchors=bool(getattr(args, "include_full_budget_anchors", True)),
        )
    return output_root, read_suite_meta(paths.suite_meta)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    if args.cmd == "build":
        output_root = Path(args.output_root) if str(args.output_root).strip() else Path(f"outputs/cpu_megasweep_{utc_run_id(args.run_id)}")
        meta = build_suite(
            run_id=utc_run_id(args.run_id),
            python_bin=str(args.python_bin).strip() or sys.executable,
            output_root=output_root,
            profile=str(args.profile),
            requested_groups=parse_items(args.groups),
            skip_existing=bool(args.skip_existing),
            include_neural_topic_estimators=bool(args.include_neural_topic_estimators),
            markov_extra_c3_strategies=bool(args.markov_extra_c3_strategies),
            markov_n_epochs=int(args.markov_n_epochs),
            torch_threads=int(args.torch_threads),
            markov_device=str(args.markov_device),
            markov_cuda_device=args.markov_cuda_device,
            segment_device=str(args.segment_device),
            segment_cuda_device=args.segment_cuda_device,
            ctree_device=str(args.ctree_device),
            ctree_cuda_device=args.ctree_cuda_device,
            include_full_budget_anchors=bool(args.include_full_budget_anchors),
        )
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0

    output_root, meta = _ensure_built(args)
    paths = resolve_grouped_suite_paths(output_root)

    if args.cmd == "run":
        built_groups = [str(x) for x in (meta.get("selected_groups", []) or []) if str(x).strip()]
        selected_groups = select_known_items(
            requested=parse_items(args.groups),
            available=built_groups,
            item_name="cpu-megasweep groups",
        )
        manifest_files = dict(meta.get("group_manifest_files", {}) or {})
        manifest_paths = [Path(str(manifest_files[key])) for key in selected_groups]
        payload = run_manifest_queue_suite(
            manifest_paths=manifest_paths,
            cpu_workers=int(args.jobs),
            gpu_tokens=str(args.gpu_tokens),
            log_dir=Path(args.log_dir).resolve() if str(args.log_dir).strip() else paths.queue_log_dir,
            set_thread_env=bool(args.set_thread_env),
        )
        print(json.dumps({"output_root": str(output_root), "selected_groups": selected_groups, **payload}, indent=2, sort_keys=True))
        return 0 if int(payload["summary"].get("n_fail", 0)) == 0 else 1

    if args.cmd == "plot":
        return int(exec_cmds_main(["--cmds", str(output_root / "suite_plot_cmds.txt"), "--jobs", str(int(args.jobs)), *(["--log-dir", str(Path(args.log_dir).resolve())] if str(args.log_dir).strip() else [])]))

    if args.cmd == "report":
        if not bool(args.skip_plots):
            rc = exec_cmds_main(["--cmds", str(output_root / "suite_plot_cmds.txt"), "--jobs", str(int(args.jobs)), *(["--log-dir", str(Path(args.log_dir).resolve())] if str(args.log_dir).strip() else [])])
            if int(rc) != 0:
                return int(rc)
        from treepo._research.ctreepo.sim.cli.report.cpu_megasweep import main as _main_report
        from treepo._research.ctreepo.sim.cli.report.cpu_megasweep_readable import main as _readable_report

        rc_main = int(_main_report(["--output-root", str(output_root), "--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf"]))
        rc_readable = int(_readable_report(["--output-root", str(output_root), "--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf"]))
        return 0 if rc_main == 0 and rc_readable == 0 else 1

    raise ValueError("unreachable")


__all__ = ["build_suite", "main"]
