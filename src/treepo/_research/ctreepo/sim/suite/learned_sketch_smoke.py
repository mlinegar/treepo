from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import Dict, List, Sequence

from treepo._research.ctreepo.sim.manifest import RunSpec
from treepo._research.ctreepo.sim.suite.common import (
    SuiteGroupRuns,
    build_suite_meta,
    emit_grouped_suite_artifacts,
    parse_items,
    read_suite_meta,
    resolve_grouped_suite_paths,
    run_manifest_queue_suite,
    select_known_items,
    utc_run_id,
    write_suite_meta,
)
from treepo._research.ctreepo.sim.suite.learned_sketch_smoke_policy import (
    LearnedSketchSmokePolicy,
    resolve_learned_sketch_smoke_policy,
)


_GROUPS: tuple[str, ...] = ("proxy_baseline",)


def _resolve_policy_from_args(args: argparse.Namespace) -> LearnedSketchSmokePolicy:
    return resolve_learned_sketch_smoke_policy(
        state_dims=(str(args.state_dims) if str(args.state_dims).strip() else None),
        train_sizes=(str(args.train_sizes) if str(args.train_sizes).strip() else None),
        zipf_alphas=(str(args.zipf_alphas) if str(args.zipf_alphas).strip() else None),
        n_val=(int(args.n_val) if int(args.n_val) > 0 else None),
        n_test=(int(args.n_test) if int(args.n_test) > 0 else None),
        hidden_dim=(int(args.hidden_dim) if int(args.hidden_dim) > 0 else None),
        n_epochs=(int(args.n_epochs) if int(args.n_epochs) > 0 else None),
        batch_size=(int(args.batch_size) if int(args.batch_size) > 0 else None),
        device=(str(args.device) if str(args.device).strip() else None),
        torch_threads=(int(args.torch_threads) if int(args.torch_threads) > 0 else None),
        seed=(int(args.seed) if int(args.seed) >= 0 else None),
        simulation_mode=(str(args.simulation_mode) if str(args.simulation_mode).strip() else None),
    )


def _resources_for_policy(policy: LearnedSketchSmokePolicy) -> Dict[str, object]:
    device = str(policy.device).strip().lower()
    accelerator = "cpu"
    gpu_eligible = False
    gpu_preferred = False
    if device == "cuda":
        accelerator = "gpu"
        gpu_eligible = True
        gpu_preferred = True
    elif device == "auto":
        accelerator = "auto"
        gpu_eligible = True
        gpu_preferred = False
    return {
        "accelerator": accelerator,
        "device_mode": device or "cpu",
        "gpu_eligible": bool(gpu_eligible),
        "gpu_preferred": bool(gpu_preferred),
        "cpu_threads": int(max(1, policy.torch_threads)),
        "torch_threads": int(policy.torch_threads),
    }


def _build_groups(
    *,
    python_bin: str,
    output_root: Path,
    policy: LearnedSketchSmokePolicy,
    skip_existing: bool,
) -> List[SuiteGroupRuns]:
    summary_root = output_root / "learned_sketch_simulation" / "proxy_baseline"
    out_json = summary_root / f"seed_{int(policy.seed)}.json"
    out_csv = summary_root / f"seed_{int(policy.seed)}.csv"
    if bool(skip_existing) and out_json.exists() and out_csv.exists():
        runs: List[RunSpec] = []
    else:
        argv = [
            str(python_bin),
            "scripts/run_learned_sketch_simulation.py",
            "--state-dims",
            ",".join(str(x) for x in policy.state_dims),
            "--train-sizes",
            ",".join(str(x) for x in policy.train_sizes),
            "--zipf-alphas",
            ",".join(str(x) for x in policy.zipf_alphas),
            "--n-val",
            str(int(policy.n_val)),
            "--n-test",
            str(int(policy.n_test)),
            "--hidden-dim",
            str(int(policy.hidden_dim)),
            "--n-epochs",
            str(int(policy.n_epochs)),
            "--batch-size",
            str(int(policy.batch_size)),
            "--device",
            str(policy.device),
            "--torch-threads",
            str(int(policy.torch_threads)),
            "--seed",
            str(int(policy.seed)),
            "--simulation-mode",
            str(policy.simulation_mode),
            "--json-summary",
            str(out_json),
            "--csv-summary",
            str(out_csv),
        ]
        command = " ".join(shlex.quote(part) for part in argv)
        runs = [
            RunSpec.create(
                family="learned-sketch-smoke",
                config={
                    **policy.to_dict(),
                    "suite_group": "proxy_baseline",
                },
                outputs={
                    "json_summary": str(out_json),
                    "csv_summary": str(out_csv),
                },
                command=command,
                resources=_resources_for_policy(policy),
            )
        ]

    return [
        SuiteGroupRuns(
            key="proxy_baseline",
            family="learned-sketch-smoke",
            runs=runs,
        )
    ]


def build_suite(
    *,
    run_id: str,
    python_bin: str,
    output_root: Path,
    requested_groups: Sequence[str],
    policy: LearnedSketchSmokePolicy,
    skip_existing: bool,
) -> Dict[str, object]:
    output_root = output_root.resolve()
    paths = resolve_grouped_suite_paths(output_root)
    groups = _build_groups(
        python_bin=python_bin,
        output_root=output_root,
        policy=policy,
        skip_existing=skip_existing,
    )
    selected_groups = select_known_items(
        requested=requested_groups,
        available=_GROUPS,
        item_name="learned-sketch smoke groups",
    )
    filtered_groups = [group for group in groups if group.key in selected_groups]
    artifacts = emit_grouped_suite_artifacts(paths, filtered_groups)
    meta = build_suite_meta(
        suite_name="learned-sketch-smoke",
        suite_role="diagnostic",
        run_id=str(run_id),
        profile="smoke",
        policy=policy.to_dict(),
        python_bin=str(python_bin),
        output_root=output_root,
        cmds_file=paths.suite_cmds,
        manifest_file=paths.suite_manifest,
        selected_groups=selected_groups,
        group_cmd_files=artifacts.group_cmd_files,
        group_manifest_files=artifacts.group_manifest_files,
        group_families=artifacts.group_families,
        extra={
            "skip_existing": bool(skip_existing),
            "counts_by_group": artifacts.counts_by_group,
            "n_commands_total": int(len(artifacts.all_cmds)),
        },
    )
    write_suite_meta(paths.suite_meta, meta)
    return meta


def _add_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-dims", type=str, default="")
    parser.add_argument("--train-sizes", type=str, default="")
    parser.add_argument("--zipf-alphas", type=str, default="")
    parser.add_argument("--n-val", type=int, default=0)
    parser.add_argument("--n-test", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--n-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--simulation-mode", type=str, default="")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Learned-sketch smoke suite.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("build", "run"):
        subp = sub.add_parser(name)
        subp.add_argument("--run-id", type=str, default="")
        subp.add_argument("--python-bin", type=str, default="")
        subp.add_argument("--output-root", type=str, default="" if name == "build" else None, required=name == "run")
        subp.add_argument("--groups", type=str, default="")
        subp.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
        _add_policy_args(subp)
        if name == "run":
            subp.add_argument("--jobs", type=int, default=1)
            subp.add_argument("--gpu-tokens", type=str, default="auto")
            subp.add_argument("--log-dir", type=str, default="")
            subp.add_argument("--rebuild", action=argparse.BooleanOptionalAction, default=False)
            subp.add_argument("--set-thread-env", action=argparse.BooleanOptionalAction, default=True)

    report = sub.add_parser("report")
    report.add_argument("--output-root", type=str, required=True)
    report.add_argument("--out-dir", type=str, default="")
    report.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    if args.cmd == "build":
        run_id = utc_run_id(args.run_id)
        output_root = (
            Path(args.output_root)
            if str(args.output_root).strip()
            else Path(f"outputs/learned_sketch_smoke_{run_id}")
        )
        meta = build_suite(
            run_id=run_id,
            python_bin=str(args.python_bin).strip() or sys.executable,
            output_root=output_root,
            requested_groups=parse_items(args.groups),
            policy=_resolve_policy_from_args(args),
            skip_existing=bool(args.skip_existing),
        )
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0

    if args.cmd == "run":
        output_root = Path(args.output_root).resolve()
        paths = resolve_grouped_suite_paths(output_root)
        if bool(args.rebuild) or not paths.suite_meta.exists() or not paths.suite_manifest.exists():
            build_suite(
                run_id=utc_run_id(args.run_id or output_root.name),
                python_bin=str(args.python_bin).strip() or sys.executable,
                output_root=output_root,
                requested_groups=parse_items(args.groups),
                policy=_resolve_policy_from_args(args),
                skip_existing=bool(args.skip_existing),
            )
        meta = read_suite_meta(paths.suite_meta)
        built_groups = [str(x) for x in (meta.get("selected_groups", []) or []) if str(x).strip()]
        selected_groups = select_known_items(
            requested=parse_items(args.groups),
            available=built_groups,
            item_name="learned-sketch smoke groups",
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
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "selected_groups": selected_groups,
                    **payload,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if int(payload["summary"].get("n_fail", 0)) == 0 else 1

    if args.cmd == "report":
        from treepo._research.ctreepo.sim.cli.report.learned_sketch_smoke import main as _report_main

        report_argv: List[str] = ["--output-root", str(Path(args.output_root).resolve())]
        if str(args.out_dir).strip():
            report_argv.extend(["--out-dir", str(Path(args.out_dir).resolve())])
        report_argv.append("--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf")
        return int(_report_main(report_argv))

    raise ValueError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
