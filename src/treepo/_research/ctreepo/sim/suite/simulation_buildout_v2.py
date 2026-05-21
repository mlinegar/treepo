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
from treepo._research.ctreepo.sim.suite.simulation_buildout import main as legacy_build_main


_GROUP_ORDER = (
    "item2_hard_markov",
    "item2_hard_segment_lda_ops",
    "item2_hard_ctreepo",
    "item3_estimator_stress_segment_lda_ops",
    "item4_guidance_frontier_ctreepo",
    "item5_ipw_expanded",
)
_GROUP_FAMILY = {
    "item2_hard_markov": "markov_changepoint_ops_count",
    "item2_hard_segment_lda_ops": "segment_lda_ops_weight_recovery",
    "item2_hard_ctreepo": "segmented_lda_ctreepo",
    "item3_estimator_stress_segment_lda_ops": "segment_lda_ops_weight_recovery",
    "item4_guidance_frontier_ctreepo": "segmented_lda_ctreepo",
    "item5_ipw_expanded": "ipw",
}


def build_suite(
    *,
    run_id: str,
    python_bin: str,
    output_root: Path,
    profile: str,
    requested_groups: Sequence[str],
    baseline_root: str,
    ipw_source_summary: str,
    torch_threads: int,
    markov_n_epochs: int,
    markov_device: str,
    markov_cuda_device: int | None,
    ipw_jobs: int,
    skip_existing: bool,
) -> Dict[str, object]:
    output_root = output_root.resolve()
    paths = resolve_grouped_suite_paths(output_root)
    legacy_dir = output_root / "_legacy_builder"
    legacy_out_cmds = legacy_dir / "simulation_buildout_cmds.txt"
    legacy_plot_cmds = legacy_dir / "simulation_buildout_plot_cmds.txt"
    legacy_meta = legacy_dir / "simulation_buildout_meta.json"

    build_argv = [
        "--python-bin", str(python_bin),
        "--run-id", str(run_id),
        "--profile", str(profile),
        "--output-root", str(output_root),
        "--baseline-root", str(baseline_root),
        "--ipw-source-summary", str(ipw_source_summary),
        "--out-cmds", str(legacy_out_cmds),
        "--out-plot-cmds", str(legacy_plot_cmds),
        "--out-meta", str(legacy_meta),
        "--torch-threads", str(int(torch_threads)),
        "--markov-n-epochs", str(int(markov_n_epochs)),
        "--markov-device", str(markov_device),
        "--ipw-jobs", str(int(ipw_jobs)),
        "--skip-existing" if bool(skip_existing) else "--no-skip-existing",
    ]
    if markov_cuda_device is not None:
        build_argv.extend(["--markov-cuda-device", str(int(markov_cuda_device))])
    legacy_build_main(build_argv)

    legacy_payload = json.loads(legacy_meta.read_text(encoding="utf-8"))
    selected_groups = select_known_items(
        requested=requested_groups,
        available=_GROUP_ORDER,
        item_name="simulation buildout groups",
    )
    all_commands = read_cmds_file(legacy_out_cmds)
    counts_by_suite = dict(legacy_payload.get("counts_by_suite", {}) or {})
    group_commands: Dict[str, List[str]] = {}
    idx = 0
    for key in _GROUP_ORDER:
        count = int(counts_by_suite.get(key, 0) or 0)
        group_commands[key] = all_commands[idx : idx + count]
        idx += count
    groups = [
        SuiteGroupRuns(
            key=key,
            family=_GROUP_FAMILY[key],
            runs=runs_from_commands(commands=group_commands[key], family=_GROUP_FAMILY[key], group_key=key),
        )
        for key in _GROUP_ORDER
        if key in selected_groups
    ]
    artifacts = emit_grouped_suite_artifacts(paths, groups)
    plot_cmds = read_cmds_file(legacy_plot_cmds)
    (output_root / "suite_plot_cmds.txt").write_text("\n".join(plot_cmds) + ("\n" if plot_cmds else ""), encoding="utf-8")
    meta = build_suite_meta(
        suite_name="simulation-buildout",
        suite_role="paper",
        run_id=str(run_id),
        profile=str(profile),
        policy={
            "baseline_root": str(baseline_root),
            "ipw_source_summary": str(ipw_source_summary),
        },
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
    parser = argparse.ArgumentParser(description="Simulation buildout v2 suite.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("build", "run", "plot", "report"):
        subp = sub.add_parser(name)
        subp.add_argument("--run-id", type=str, default="")
        subp.add_argument("--python-bin", type=str, default="")
        subp.add_argument("--output-root", type=str, default="" if name == "build" else None, required=name != "build")
        if name in {"build", "run"}:
            subp.add_argument("--profile", choices=["smoke", "paper", "full"], default="paper")
            subp.add_argument("--groups", type=str, default="")
            subp.add_argument("--baseline-root", type=str, default="outputs/cpu_megasweep")
            subp.add_argument("--ipw-source-summary", type=str, default="outputs/ipw_stress_ladder/summary_rows.csv")
            subp.add_argument("--torch-threads", type=int, default=1)
            subp.add_argument("--markov-n-epochs", type=int, default=12)
            subp.add_argument("--markov-device", type=str, default="auto")
            subp.add_argument("--markov-cuda-device", type=int, default=None)
            subp.add_argument("--ipw-jobs", type=int, default=128)
            subp.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
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
            baseline_root=str(getattr(args, "baseline_root", "outputs/cpu_megasweep")),
            ipw_source_summary=str(getattr(args, "ipw_source_summary", "outputs/ipw_stress_ladder/summary_rows.csv")),
            torch_threads=int(getattr(args, "torch_threads", 1)),
            markov_n_epochs=int(getattr(args, "markov_n_epochs", 12)),
            markov_device=str(getattr(args, "markov_device", "auto")),
            markov_cuda_device=getattr(args, "markov_cuda_device", None),
            ipw_jobs=int(getattr(args, "ipw_jobs", 128)),
            skip_existing=bool(getattr(args, "skip_existing", True)),
        )
    return output_root, read_suite_meta(paths.suite_meta)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    if args.cmd == "build":
        output_root = Path(args.output_root) if str(args.output_root).strip() else Path(f"outputs/simulation_buildout_{utc_run_id(args.run_id)}")
        meta = build_suite(
            run_id=utc_run_id(args.run_id),
            python_bin=str(args.python_bin).strip() or sys.executable,
            output_root=output_root,
            profile=str(args.profile),
            requested_groups=parse_items(args.groups),
            baseline_root=str(args.baseline_root),
            ipw_source_summary=str(args.ipw_source_summary),
            torch_threads=int(args.torch_threads),
            markov_n_epochs=int(args.markov_n_epochs),
            markov_device=str(args.markov_device),
            markov_cuda_device=args.markov_cuda_device,
            ipw_jobs=int(args.ipw_jobs),
            skip_existing=bool(args.skip_existing),
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
            item_name="simulation buildout groups",
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
        exec_argv = ["--cmds", str(output_root / "suite_plot_cmds.txt"), "--jobs", str(int(args.jobs))]
        if str(args.log_dir).strip():
            exec_argv.extend(["--log-dir", str(Path(args.log_dir).resolve())])
        return int(exec_cmds_main(exec_argv))

    if args.cmd == "report":
        if not bool(args.skip_plots):
            exec_argv = ["--cmds", str(output_root / "suite_plot_cmds.txt"), "--jobs", str(int(args.jobs))]
            if str(args.log_dir).strip():
                exec_argv.extend(["--log-dir", str(Path(args.log_dir).resolve())])
            rc = int(exec_cmds_main(exec_argv))
            if rc != 0:
                return rc
        from treepo._research.ctreepo.sim.cli.report.simulation_buildout import main as _report_main

        return int(_report_main(["--output-root", str(output_root), "--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf"]))

    raise ValueError("unreachable")


__all__ = ["build_suite", "main"]
