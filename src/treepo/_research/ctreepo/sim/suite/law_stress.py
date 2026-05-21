from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

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
from treepo._research.ctreepo.sim.suite.law_stress_builders import (
    build_lda_law_stress_suites,
    build_markov_law_stress_suites,
)


_GROUP_SPECS: tuple[Tuple[str, str, str], ...] = (
    ("markov_sanity_suite", "markov", "sanity_suite"),
    ("markov_transition_map_suite", "markov", "transition_map_suite"),
    ("markov_mechanism_suite", "markov", "mechanism_suite"),
    ("markov_capacity_appendix_suite", "markov", "capacity_appendix_suite"),
    ("markov_cross_dgp_suite", "markov", "cross_dgp_suite"),
    ("markov_weight_ablation_suite", "markov", "weight_ablation_suite"),
    ("lda_sanity_suite", "lda", "sanity_suite"),
    ("lda_transition_map_suite", "lda", "transition_map_suite"),
    ("lda_mechanism_suite", "lda", "mechanism_suite"),
)


def _group_spec_map() -> Dict[str, Tuple[str, str]]:
    return {key: (family, suite_name) for key, family, suite_name in _GROUP_SPECS}


def _build_markov_group(
    *,
    group_key: str,
    suite_name: str,
    python_bin: str,
    output_root: Path,
    cmd_dir: Path,
    smoke: bool,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
    transition_summary: str,
) -> Tuple[SuiteGroupRuns, Dict[str, object]]:
    result = build_markov_law_stress_suites(
        suite=str(suite_name),
        output_root=output_root,
        cmd_dir=cmd_dir,
        python_bin=str(python_bin),
        device=str(device),
        cuda_device=(int(cuda_device) if cuda_device is not None else None),
        torch_threads=int(torch_threads),
        transition_summary=(Path(transition_summary).resolve() if str(transition_summary).strip() else None),
        smoke=bool(smoke),
    )
    return (
        SuiteGroupRuns(key=str(group_key), family="markov-law-stress", runs=result.runs),
        dict(result.manifest.get("policy", {}) or {}),
    )


def _build_lda_group(
    *,
    group_key: str,
    suite_name: str,
    python_bin: str,
    output_root: Path,
    cmd_dir: Path,
    smoke: bool,
    skip_existing: bool,
) -> Tuple[SuiteGroupRuns, Dict[str, object]]:
    result = build_lda_law_stress_suites(
        suite=str(suite_name),
        output_root=output_root,
        cmd_dir=cmd_dir,
        python_bin=str(python_bin),
        skip_existing=bool(skip_existing),
        smoke=bool(smoke),
    )
    return (
        SuiteGroupRuns(key=str(group_key), family="lda-law-stress", runs=result.runs),
        dict(result.manifest.get("policy", {}) or {}),
    )


def _build_groups(
    *,
    selected_groups: Sequence[str],
    python_bin: str,
    output_root: Path,
    smoke: bool,
    skip_existing: bool,
    markov_device: str,
    markov_cuda_device: int | None,
    torch_threads: int,
    transition_summary: str,
) -> Tuple[List[SuiteGroupRuns], Dict[str, object]]:
    groups: List[SuiteGroupRuns] = []
    group_policies: Dict[str, object] = {}
    spec_map = _group_spec_map()

    for group_key in selected_groups:
        family, suite_name = spec_map[group_key]
        raw_cmd_dir = output_root / "_raw" / group_key
        if family == "markov":
            built_group, policy = _build_markov_group(
                group_key=group_key,
                suite_name=suite_name,
                python_bin=python_bin,
                output_root=output_root / "markov",
                cmd_dir=raw_cmd_dir,
                smoke=bool(smoke),
                device=str(markov_device),
                cuda_device=markov_cuda_device,
                torch_threads=int(torch_threads),
                transition_summary=str(transition_summary),
            )
            groups.append(built_group)
            group_policies[group_key] = policy
        else:
            built_group, policy = _build_lda_group(
                group_key=group_key,
                suite_name=suite_name,
                python_bin=python_bin,
                output_root=output_root / "lda",
                cmd_dir=raw_cmd_dir,
                smoke=bool(smoke),
                skip_existing=bool(skip_existing),
            )
            groups.append(built_group)
            group_policies[group_key] = policy
    return groups, group_policies


def build_suite(
    *,
    run_id: str,
    python_bin: str,
    output_root: Path,
    requested_groups: Sequence[str],
    smoke: bool,
    skip_existing: bool,
    markov_device: str,
    markov_cuda_device: int | None,
    torch_threads: int,
    transition_summary: str,
) -> Dict[str, object]:
    output_root = output_root.resolve()
    paths = resolve_grouped_suite_paths(output_root)
    selected_groups = select_known_items(
        requested=requested_groups,
        available=[key for key, _family, _suite_name in _GROUP_SPECS],
        item_name="law-stress groups",
    )
    group_builds, group_policies = _build_groups(
        selected_groups=selected_groups,
        python_bin=python_bin,
        output_root=output_root,
        smoke=bool(smoke),
        skip_existing=bool(skip_existing),
        markov_device=str(markov_device),
        markov_cuda_device=markov_cuda_device,
        torch_threads=int(torch_threads),
        transition_summary=str(transition_summary),
    )
    artifacts = emit_grouped_suite_artifacts(paths, group_builds)

    meta: Dict[str, object] = build_suite_meta(
        suite_name="law-stress",
        suite_role="appendix",
        run_id=str(run_id),
        profile="v1",
        policy={},
        python_bin=str(python_bin),
        output_root=output_root,
        cmds_file=paths.suite_cmds,
        manifest_file=paths.suite_manifest,
        selected_groups=list(selected_groups),
        group_cmd_files=artifacts.group_cmd_files,
        group_manifest_files=artifacts.group_manifest_files,
        group_families=artifacts.group_families,
        extra={
            "smoke": bool(smoke),
            "skip_existing": bool(skip_existing),
            "markov_device": str(markov_device),
            "markov_cuda_device": markov_cuda_device,
            "torch_threads": int(torch_threads),
            "transition_summary": str(transition_summary),
            "group_policies": group_policies,
            "counts_by_group": artifacts.counts_by_group,
            "n_commands_total": int(len(artifacts.all_cmds)),
        },
    )
    write_suite_meta(paths.suite_meta, meta)
    return meta


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified law-stress suite orchestration.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Build manifests and command files for the law-stress suite.")
    build.add_argument("--run-id", type=str, default="")
    build.add_argument("--python-bin", type=str, default="")
    build.add_argument("--output-root", type=str, default="")
    build.add_argument("--groups", type=str, default="")
    build.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=False)
    build.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    build.add_argument("--markov-device", type=str, default="auto")
    build.add_argument("--markov-cuda-device", type=int, default=None)
    build.add_argument("--torch-threads", type=int, default=1)
    build.add_argument("--transition-summary", type=str, default="")

    run = sub.add_parser("run", help="Build if needed, then execute the law-stress suite.")
    run.add_argument("--run-id", type=str, default="")
    run.add_argument("--python-bin", type=str, default="")
    run.add_argument("--output-root", type=str, required=True)
    run.add_argument("--groups", type=str, default="")
    run.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--markov-device", type=str, default="auto")
    run.add_argument("--markov-cuda-device", type=int, default=None)
    run.add_argument("--torch-threads", type=int, default=1)
    run.add_argument("--transition-summary", type=str, default="")
    run.add_argument("--jobs", type=int, default=1)
    run.add_argument("--gpu-tokens", type=str, default="auto")
    run.add_argument("--log-dir", type=str, default="")
    run.add_argument("--rebuild", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--set-thread-env", action=argparse.BooleanOptionalAction, default=True)

    report = sub.add_parser("report", help="Generate a family-specific law-stress report.")
    report.add_argument("--output-root", type=str, required=True)
    report.add_argument("--family", choices=["markov", "lda"], required=True)
    report.add_argument("--output-dir", type=str, default="")
    report.add_argument("--title", type=str, default="")
    report.add_argument("--pdf-path", type=str, default="")
    report.add_argument("--expected-run-count", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    if args.cmd == "build":
        run_id = utc_run_id(args.run_id)
        python_bin = str(args.python_bin).strip() or sys.executable
        output_root = (
            Path(args.output_root)
            if str(args.output_root).strip()
            else Path(f"outputs/law_stress_suite_{run_id}")
        )
        meta = build_suite(
            run_id=run_id,
            python_bin=python_bin,
            output_root=output_root,
            requested_groups=parse_items(args.groups),
            smoke=bool(args.smoke),
            skip_existing=bool(args.skip_existing),
            markov_device=str(args.markov_device),
            markov_cuda_device=(int(args.markov_cuda_device) if args.markov_cuda_device is not None else None),
            torch_threads=int(args.torch_threads),
            transition_summary=str(args.transition_summary),
        )
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0

    if args.cmd == "run":
        output_root = Path(args.output_root).resolve()
        paths = resolve_grouped_suite_paths(output_root)
        if bool(args.rebuild) or not paths.suite_meta.exists() or not paths.suite_manifest.exists():
            run_id = utc_run_id(args.run_id or output_root.name)
            build_suite(
                run_id=run_id,
                python_bin=(str(args.python_bin).strip() or sys.executable),
                output_root=output_root,
                requested_groups=parse_items(args.groups),
                smoke=bool(args.smoke),
                skip_existing=bool(args.skip_existing),
                markov_device=str(args.markov_device),
                markov_cuda_device=(int(args.markov_cuda_device) if args.markov_cuda_device is not None else None),
                torch_threads=int(args.torch_threads),
                transition_summary=str(args.transition_summary),
            )

        meta = read_suite_meta(paths.suite_meta)
        built_groups = [str(x) for x in (meta.get("selected_groups", []) or []) if str(x).strip()]
        requested_groups = parse_items(args.groups)
        if requested_groups:
            unknown = sorted(set(requested_groups) - set(built_groups))
            if unknown:
                raise SystemExit(
                    f"requested groups were not built under {output_root}: {', '.join(unknown)}"
                )
            selected_groups = [key for key in built_groups if key in requested_groups]
        else:
            selected_groups = built_groups

        manifest_files = dict(meta.get("group_manifest_files", {}) or {})
        manifest_paths = [Path(str(manifest_files[key])) for key in selected_groups]
        if not manifest_paths:
            raise SystemExit("no manifest paths selected")

        log_dir = Path(args.log_dir).resolve() if str(args.log_dir).strip() else paths.queue_log_dir
        queue_payload = run_manifest_queue_suite(
            manifest_paths=manifest_paths,
            cpu_workers=int(args.jobs),
            gpu_tokens=str(args.gpu_tokens),
            log_dir=log_dir,
            set_thread_env=bool(args.set_thread_env),
        )
        payload = {
            "output_root": str(output_root),
            "selected_groups": list(selected_groups),
            **queue_payload,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if int(queue_payload["summary"].get("n_fail", 0)) == 0 else 1

    if args.cmd == "report":
        from treepo._research.ctreepo.sim.cli.report.law_stress import main as _report_main  # noqa: WPS433

        family_root = Path(args.output_root).resolve() / str(args.family)
        report_argv: List[str] = ["--family", str(args.family), "--input-root", str(family_root)]
        if str(args.output_dir).strip():
            report_argv.extend(["--output-dir", str(Path(args.output_dir).resolve())])
        if str(args.title).strip():
            report_argv.extend(["--title", str(args.title)])
        if str(args.pdf_path).strip():
            report_argv.extend(["--pdf-path", str(Path(args.pdf_path).resolve())])
        if args.expected_run_count is not None:
            report_argv.extend(["--expected-run-count", str(int(args.expected_run_count))])
        return int(_report_main(report_argv))

    raise ValueError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
