from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
from typing import Dict, List, Sequence, Tuple

from treepo._research.ctreepo.sim.manifest import RunSpec, read_manifest_jsonl, write_manifest_jsonl
from treepo._research.ctreepo.sim.runner import read_cmds_file
from treepo._research.ctreepo.sim.suite.common import (
    build_suite_meta,
    read_suite_meta,
    run_manifest_queue_suite,
    thread_env_prefix,
    utc_run_id,
    write_suite_meta,
    write_text,
)
from treepo._research.ctreepo.sim.suite.publication_lanes import resolve_publication_lane_calls
from treepo._research.ctreepo.sim.suite.publication_policy import (
    resolve_publication_ctreepo_policy,
)


def _q(x: object) -> str:
    return shlex.quote(str(x))


@dataclass(frozen=True)
class SuitePaths:
    output_root: Path
    suite_meta: Path
    suite_cmds: Path
    suite_manifest: Path
    lane_dir: Path


def _resolve_paths(output_root: Path) -> SuitePaths:
    return SuitePaths(
        output_root=output_root,
        suite_meta=output_root / "suite_meta.json",
        suite_cmds=output_root / "suite_cmds.txt",
        suite_manifest=output_root / "suite_manifest.jsonl",
        lane_dir=output_root / "suite_lanes",
    )


def build_suite(
    *,
    run_id: str,
    profile: str,
    python_bin: str,
    output_root: Path,
    skip_existing: bool,
    set_thread_env: bool,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
) -> dict:
    output_root = output_root.resolve()
    paths = _resolve_paths(output_root)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    paths.lane_dir.mkdir(parents=True, exist_ok=True)

    policy = resolve_publication_ctreepo_policy(str(profile))
    lane_calls = resolve_publication_lane_calls(output_root=output_root, policy=policy)

    from treepo._research.ctreepo.sim.cli.sweep_segmented_lda_ctreepo import (  # noqa: WPS433
        main as _ctree_sweep,
    )

    lane_cmds: Dict[str, Path] = {}
    lane_manifests: Dict[str, Path] = {}
    counts: Dict[str, int] = {}
    all_cmds: List[str] = []
    all_runs: List[RunSpec] = []
    env_prefix = thread_env_prefix() if bool(set_thread_env) else ""

    for lane in lane_calls:
        out_cmds = paths.lane_dir / f"{lane.key}_cmds.txt"
        out_manifest = paths.lane_dir / f"{lane.key}_manifest.jsonl"
        lane_cmds[lane.key] = out_cmds
        lane_manifests[lane.key] = out_manifest

        lane_cmd_lines: List[str] = []
        lane_runs: List[RunSpec] = []
        for lt in lane.fixed_leaf_tokens_grid:
            tmp_cmds = paths.lane_dir / f"{lane.key}_lt{int(lt)}_cmds.tmp.txt"
            tmp_manifest = paths.lane_dir / f"{lane.key}_lt{int(lt)}_manifest.tmp.jsonl"
            sweep_argv = [
                "--python-bin",
                str(python_bin),
                "--out-cmds",
                str(tmp_cmds),
                "--out-manifest",
                str(tmp_manifest),
                "--skip-existing" if bool(skip_existing) else "--no-skip-existing",
                "--fixed-leaf-tokens",
                str(int(lt)),
                "--device",
                str(device),
                "--torch-threads",
                str(int(torch_threads)),
                *lane.argv_base,
            ]
            if cuda_device is not None:
                sweep_argv.extend(["--cuda-device", str(int(cuda_device))])
            _ctree_sweep(sweep_argv)
            lane_cmd_lines.extend(read_cmds_file(tmp_cmds) if tmp_cmds.exists() else [])
            lane_runs.extend(read_manifest_jsonl(tmp_manifest))

        if env_prefix:
            lane_cmd_lines = [f"{env_prefix} {c}" for c in lane_cmd_lines]
            lane_runs = [
                RunSpec(
                    id=r.id,
                    family=r.family,
                    config=dict(r.config),
                    outputs=dict(r.outputs),
                    command=f"{env_prefix} {r.command}",
                    requires=list(r.requires),
                    resources=dict(r.resources),
                )
                for r in lane_runs
            ]

        write_text(out_cmds, "\n".join(lane_cmd_lines) + ("\n" if lane_cmd_lines else ""))
        write_manifest_jsonl(out_manifest, lane_runs)

        counts[lane.key] = int(len(lane_cmd_lines))
        all_cmds.extend(lane_cmd_lines)
        all_runs.extend(lane_runs)

    write_text(paths.suite_cmds, "\n".join(all_cmds) + ("\n" if all_cmds else ""))
    write_manifest_jsonl(paths.suite_manifest, all_runs)

    meta = build_suite_meta(
        suite_name="publication-ctreepo",
        suite_role="paper",
        run_id=str(run_id),
        profile=str(profile),
        policy=policy.to_dict(),
        python_bin=str(python_bin),
        output_root=output_root,
        cmds_file=paths.suite_cmds,
        manifest_file=paths.suite_manifest,
        selected_groups=list(lane_cmds),
        group_cmd_files={k: str(v) for k, v in lane_cmds.items()},
        group_manifest_files={k: str(v) for k, v in lane_manifests.items()},
        group_families={lane.key: "segmented_lda_ctreepo" for lane in lane_calls},
        extra={
            "skip_existing": bool(skip_existing),
            "set_thread_env": bool(set_thread_env),
            "device": str(device),
            "cuda_device": int(cuda_device) if cuda_device is not None else None,
            "torch_threads": int(torch_threads),
            "lane_catalog": {lane.key: lane.to_dict() for lane in lane_calls},
            "lane_cmd_files": {k: str(v) for k, v in lane_cmds.items()},
            "lane_manifest_files": {k: str(v) for k, v in lane_manifests.items()},
            "counts_by_lane": counts,
            "counts_by_group": counts,
            "n_commands_total": int(len(all_cmds)),
        },
    )
    write_suite_meta(paths.suite_meta, meta)
    return meta


def _add_build_args(parser: argparse.ArgumentParser, *, require_output_root: bool) -> None:
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--profile", choices=["smoke", "publication"], default="publication")
    parser.add_argument("--python-bin", type=str, default="")
    parser.add_argument(
        "--output-root",
        type=str,
        default="" if not require_output_root else None,
        required=bool(require_output_root),
        help="Default: outputs/identifiable_zero_publication_ctreepo_<run_id>",
    )
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--set-thread-env",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefix commands with low-thread env vars (OMP/MKL/OpenBLAS/etc).",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=1)


def _resolve_output_root(*, run_id: str, output_root: str) -> Path:
    from treepo._research.ctreepo.sim.suite.common import resolve_output_root
    return resolve_output_root(run_id=run_id, output_root=output_root, default_prefix="identifiable_zero_publication_ctreepo")


def _ensure_built(ns: argparse.Namespace) -> Tuple[SuitePaths, dict]:
    run_id = utc_run_id(getattr(ns, "run_id", ""))
    output_root = _resolve_output_root(run_id=run_id, output_root=str(getattr(ns, "output_root", "")))
    paths = _resolve_paths(output_root.resolve())
    rebuild = bool(getattr(ns, "rebuild", False))
    if rebuild or not paths.suite_meta.exists() or not paths.suite_manifest.exists() or not paths.suite_cmds.exists():
        meta = build_suite(
            run_id=run_id,
            profile=str(getattr(ns, "profile", "publication")),
            python_bin=str(getattr(ns, "python_bin", "")).strip() or __import__("sys").executable,
            output_root=output_root,
            skip_existing=bool(getattr(ns, "skip_existing", True)),
            set_thread_env=bool(getattr(ns, "set_thread_env", True)),
            device=str(getattr(ns, "device", "auto")),
            cuda_device=getattr(ns, "cuda_device", None),
            torch_threads=int(getattr(ns, "torch_threads", 1)),
        )
        return paths, meta
    return paths, read_suite_meta(paths.suite_meta)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Publication C-TreePO benchmark suite orchestration.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build command lists + manifest for the publication benchmark suite.")
    _add_build_args(b, require_output_root=False)

    r = sub.add_parser("run", help="Execute the suite command list.")
    _add_build_args(r, require_output_root=True)
    r.add_argument("--rebuild", action=argparse.BooleanOptionalAction, default=False)
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--gpu-tokens", type=str, default="auto")
    r.add_argument("--log-dir", type=str, default="")
    r.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)

    rep = sub.add_parser("report", help="Generate the canonical publication progress report.")
    rep.add_argument("--output-root", type=str, required=True)
    rep.add_argument("--out-dir", type=str, default="")
    rep.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)

    prog = sub.add_parser("progress", help="Alias for `report`.")
    prog.add_argument("--output-root", type=str, required=True)
    prog.add_argument("--out-dir", type=str, default="")
    prog.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)

    exp = sub.add_parser("expectations", help="Generate simulation expectation JSON/Markdown reports.")
    exp.add_argument("--output-root", type=str, required=True)
    exp.add_argument("--output-json", type=str, default="")
    exp.add_argument("--output-markdown", type=str, default="")
    exp.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    exp.add_argument("--seed-aggregate", choices=["median", "mean"], default="median")
    exp.add_argument("--min-effect", type=float, default=0.10)
    exp.add_argument("--adjacent-tolerance", type=float, default=0.01)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    ns = _build_parser().parse_args(list(argv) if argv is not None else None)

    if ns.cmd == "build":
        run_id = utc_run_id(ns.run_id)
        python_bin = str(ns.python_bin).strip() or __import__("sys").executable
        output_root = _resolve_output_root(run_id=run_id, output_root=str(ns.output_root or ""))
        meta = build_suite(
            run_id=run_id,
            profile=str(ns.profile),
            python_bin=python_bin,
            output_root=output_root,
            skip_existing=bool(ns.skip_existing),
            set_thread_env=bool(ns.set_thread_env),
            device=str(ns.device),
            cuda_device=ns.cuda_device,
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
            # Publication suite already bakes the thread env policy into each command line.
            set_thread_env=False,
        )
        payload = {
            "output_root": str(paths.output_root),
            "profile": str(meta.get("profile", "")),
            **queue_payload,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if int(queue_payload["summary"].get("n_fail", 0)) == 0 else 1

    if ns.cmd in {"report", "progress"}:
        from treepo._research.ctreepo.sim.cli.report.publication_ctreepo_progress import (  # noqa: WPS433
            main as _progress_main,
        )

        argv_out: List[str] = ["--output-root", str(Path(ns.output_root).resolve())]
        if str(ns.out_dir).strip():
            argv_out.extend(["--out-dir", str(Path(ns.out_dir).resolve())])
        argv_out.append("--emit-pdf" if bool(ns.emit_pdf) else "--no-emit-pdf")
        return int(_progress_main(argv_out))

    if ns.cmd == "expectations":
        from treepo._research.ctreepo.sim.expectations import (  # noqa: WPS433
            ExpectationConfig,
            build_expectation_report,
            write_expectation_report,
        )

        output_root = Path(ns.output_root).resolve()
        report = build_expectation_report(
            output_root=output_root,
            config=ExpectationConfig(
                seed_aggregate=str(ns.seed_aggregate),
                min_effect_rel=float(ns.min_effect),
                adjacent_tolerance=float(ns.adjacent_tolerance),
            ),
        )
        out_json = (
            Path(ns.output_json).resolve()
            if str(ns.output_json).strip()
            else output_root / "simulation_expectations.json"
        )
        out_markdown = (
            Path(ns.output_markdown).resolve()
            if str(ns.output_markdown).strip()
            else output_root / "simulation_expectations.md"
        )
        outputs = write_expectation_report(report, output_json=out_json, output_markdown=out_markdown)
        print(
            json.dumps(
                {
                    "output_json": outputs["output_json"],
                    "output_markdown": outputs["output_markdown"],
                    "summary": report.summary,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1 if bool(ns.strict) and int(report.summary.get("n_fail", 0)) > 0 else 0

    raise ValueError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
