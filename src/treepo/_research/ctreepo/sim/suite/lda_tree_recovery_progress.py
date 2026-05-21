from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence

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
from treepo._research.ctreepo.sim.suite.lda_tree_recovery_exact_builder import main as exact_builder_main
from treepo._research.ctreepo.sim.suite.lda_tree_recovery_world_batch_builder import main as world_batch_builder_main


@dataclass(frozen=True)
class LdaTreeRecoveryProgressPolicy:
    exact_leaf_tokens: tuple[int, ...]
    exact_doc_topic_concentrations: tuple[float, ...]
    exact_quadratic_weights: tuple[float, ...]
    exact_seeds: tuple[int, ...]
    learned_gpu_doc_topic_concentrations: tuple[float, ...]
    learned_gpu_leaf_tokens: tuple[int, ...]
    learned_gpu_train_docs: tuple[int, ...]
    learned_gpu_state_dims: tuple[int, ...]
    learned_gpu_seeds: tuple[int, ...]
    learned_cpu_doc_topic_concentrations: tuple[float, ...]
    learned_cpu_leaf_tokens: tuple[int, ...]
    learned_cpu_train_docs: tuple[int, ...]
    learned_cpu_state_dims: tuple[int, ...]
    learned_cpu_seeds: tuple[int, ...]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def resolve_policy() -> LdaTreeRecoveryProgressPolicy:
    return LdaTreeRecoveryProgressPolicy(
        exact_leaf_tokens=(384, 192, 96, 48, 24, 16),
        exact_doc_topic_concentrations=(0.2, 0.6, 1.5),
        exact_quadratic_weights=(0.0, 1.0, 2.0),
        exact_seeds=tuple(range(32)),
        learned_gpu_doc_topic_concentrations=(0.2, 0.6, 1.5),
        learned_gpu_leaf_tokens=(384, 192, 96, 16),
        learned_gpu_train_docs=(128, 512, 2048),
        learned_gpu_state_dims=(8, 16, 32, 64, 128, 256, 512),
        learned_gpu_seeds=tuple(range(8)),
        learned_cpu_doc_topic_concentrations=(0.6,),
        learned_cpu_leaf_tokens=(384, 96, 16),
        learned_cpu_train_docs=(128, 512, 2048),
        learned_cpu_state_dims=(8, 32, 128, 512),
        learned_cpu_seeds=(0, 1, 2, 3),
    )


def _write_spec(path: Path, *, output_root: Path) -> None:
    lines = [
        f"generated_at_utc={utc_run_id()}",
        f"output_root={output_root}",
        f"exact_root={output_root / 'exact_cpu'}",
        f"learned_gpu_root={output_root / 'learned_gpu'}",
        f"learned_cpu_root={output_root / 'learned_cpu_shadow'}",
        f"learned_world_cache_dir={output_root / 'learned_shared_cache' / 'world_cache'}",
        f"learned_prepared_cache_dir={output_root / 'learned_shared_cache' / 'prepared_cache'}",
        "exact_matrix=leaf(384,192,96,48,24,16) x dtc(0.2,0.6,1.5) x quadratic_weight(0,1,2) x seeds(32), test_docs=2048",
        "learned_gpu_matrix=bundled by (dtc,seed) over leaf(384,192,96,16) x quadratic_weight(0,1,2) x train(128,512,2048) x state(8,16,32,64,128,256,512), dtc(0.2,0.6,1.5), seeds(8), test_docs=512, epochs=80",
        "learned_cpu_shadow_matrix=bundled by (dtc,seed) over leaf(384,96,16) x quadratic_weight(0,1,2) x train(128,512,2048) x state(8,32,128,512), dtc(0.6), seeds(4), test_docs=512, epochs=80",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_suite(
    *,
    run_id: str,
    python_bin: str,
    output_root: Path,
    requested_groups: Sequence[str],
    gpu_device: str,
    gpu_cuda_device: int | None,
    cpu_device: str,
    torch_threads: int,
    skip_existing: bool,
) -> Dict[str, object]:
    output_root = output_root.resolve()
    paths = resolve_grouped_suite_paths(output_root)
    policy = resolve_policy()

    exact_cmds = output_root / "exact_cpu" / "commands.txt"
    learned_gpu_cmds = output_root / "learned_gpu" / "commands.txt"
    learned_cpu_cmds = output_root / "learned_cpu_shadow" / "commands.txt"
    world_cache_dir = output_root / "learned_shared_cache" / "world_cache"
    prepared_cache_dir = output_root / "learned_shared_cache" / "prepared_cache"
    world_cache_dir.mkdir(parents=True, exist_ok=True)
    prepared_cache_dir.mkdir(parents=True, exist_ok=True)

    exact_builder_main(
        [
            "--python-bin", str(python_bin),
            "--out-cmds", str(exact_cmds),
            "--output-root", str(output_root / "exact_cpu" / "results"),
            "--leaf-tokens", " ".join(str(x) for x in policy.exact_leaf_tokens),
            "--doc-topic-concentrations", " ".join(str(x) for x in policy.exact_doc_topic_concentrations),
            "--quadratic-utility-weights", " ".join(str(int(x)) for x in policy.exact_quadratic_weights),
            "--seeds", " ".join(str(x) for x in policy.exact_seeds),
            "--test-docs", "2048",
            "--skip-existing" if bool(skip_existing) else "--no-skip-existing",
        ]
    )
    world_batch_builder_main(
        [
            "--out-cmds", str(learned_gpu_cmds),
            "--output-root", str(output_root / "learned_gpu" / "results"),
            "--world-cache-dir", str(world_cache_dir),
            "--prepared-cache-dir", str(prepared_cache_dir),
            "--doc-topic-concentrations", " ".join(str(x) for x in policy.learned_gpu_doc_topic_concentrations),
            "--quadratic-utility-weights", "0 1 2",
            "--leaf-tokens-grid", " ".join(str(x) for x in policy.learned_gpu_leaf_tokens),
            "--train-docs-grid", " ".join(str(x) for x in policy.learned_gpu_train_docs),
            "--state-dims", " ".join(str(x) for x in policy.learned_gpu_state_dims),
            "--max-train-docs-capacity", "2048",
            "--test-docs", "512",
            "--full-hidden-dim", "256",
            "--full-n-layers", "3",
            "--n-epochs", "80",
            "--batch-size", "128",
            "--device", str(gpu_device),
            "--torch-threads", str(int(torch_threads)),
            "--seeds", " ".join(str(x) for x in policy.learned_gpu_seeds),
            "--skip-existing" if bool(skip_existing) else "--no-skip-existing",
            *(["--cuda-device", str(int(gpu_cuda_device))] if gpu_cuda_device is not None else []),
        ]
    )
    world_batch_builder_main(
        [
            "--out-cmds", str(learned_cpu_cmds),
            "--output-root", str(output_root / "learned_cpu_shadow" / "results"),
            "--world-cache-dir", str(world_cache_dir),
            "--prepared-cache-dir", str(prepared_cache_dir),
            "--doc-topic-concentrations", " ".join(str(x) for x in policy.learned_cpu_doc_topic_concentrations),
            "--quadratic-utility-weights", "0 1 2",
            "--leaf-tokens-grid", " ".join(str(x) for x in policy.learned_cpu_leaf_tokens),
            "--train-docs-grid", " ".join(str(x) for x in policy.learned_cpu_train_docs),
            "--state-dims", " ".join(str(x) for x in policy.learned_cpu_state_dims),
            "--max-train-docs-capacity", "2048",
            "--test-docs", "512",
            "--full-hidden-dim", "256",
            "--full-n-layers", "3",
            "--n-epochs", "80",
            "--batch-size", "128",
            "--device", str(cpu_device),
            "--torch-threads", str(int(torch_threads)),
            "--seeds", " ".join(str(x) for x in policy.learned_cpu_seeds),
            "--skip-existing" if bool(skip_existing) else "--no-skip-existing",
        ]
    )
    _write_spec(output_root / "sweep_spec.txt", output_root=output_root)

    available_groups = ("exact_cpu", "learned_cpu_shadow", "learned_gpu")
    selected_groups = select_known_items(
        requested=requested_groups,
        available=available_groups,
        item_name="lda tree recovery groups",
    )
    cmd_map = {
        "exact_cpu": exact_cmds,
        "learned_cpu_shadow": learned_cpu_cmds,
        "learned_gpu": learned_gpu_cmds,
    }
    groups: List[SuiteGroupRuns] = []
    for key in available_groups:
        if key not in selected_groups:
            continue
        groups.append(
            SuiteGroupRuns(
                key=key,
                family="lda_tree_recovery",
                runs=runs_from_commands(
                    commands=read_cmds_file(cmd_map[key]),
                    family="lda_tree_recovery",
                    group_key=key,
                ),
            )
        )
    artifacts = emit_grouped_suite_artifacts(paths, groups)
    meta = build_suite_meta(
        suite_name="lda-tree-recovery-progress",
        suite_role="diagnostic",
        run_id=str(run_id),
        profile="production",
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
            "gpu_device": str(gpu_device),
            "gpu_cuda_device": int(gpu_cuda_device) if gpu_cuda_device is not None else None,
            "cpu_device": str(cpu_device),
            "torch_threads": int(torch_threads),
            "skip_existing": bool(skip_existing),
            "sweep_spec": str(output_root / "sweep_spec.txt"),
            "counts_by_group": artifacts.counts_by_group,
            "n_commands_total": int(len(artifacts.all_cmds)),
        },
    )
    write_suite_meta(paths.suite_meta, meta)
    return meta


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LDA tree-recovery diagnostic suite.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("build", "run"):
        subp = sub.add_parser(name)
        subp.add_argument("--run-id", type=str, default="")
        subp.add_argument("--python-bin", type=str, default="")
        subp.add_argument("--output-root", type=str, default="" if name == "build" else None, required=name == "run")
        subp.add_argument("--groups", type=str, default="")
        subp.add_argument("--gpu-device", type=str, default="auto")
        subp.add_argument("--gpu-cuda-device", type=int, default=None)
        subp.add_argument("--cpu-device", type=str, default="cpu")
        subp.add_argument("--torch-threads", type=int, default=1)
        subp.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
        if name == "run":
            subp.add_argument("--jobs", type=int, default=1)
            subp.add_argument("--gpu-tokens", type=str, default="auto")
            subp.add_argument("--log-dir", type=str, default="")
            subp.add_argument("--rebuild", action=argparse.BooleanOptionalAction, default=False)
            subp.add_argument("--set-thread-env", action=argparse.BooleanOptionalAction, default=True)
    report = sub.add_parser("report")
    report.add_argument("--output-root", type=str, required=True)
    report.add_argument("--output-dir", type=str, default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    if args.cmd == "build":
        output_root = Path(args.output_root) if str(args.output_root).strip() else Path(f"outputs/lda_tree_recovery_production_{utc_run_id(args.run_id)}")
        meta = build_suite(
            run_id=utc_run_id(args.run_id),
            python_bin=str(args.python_bin).strip() or sys.executable,
            output_root=output_root,
            requested_groups=parse_items(args.groups),
            gpu_device=str(args.gpu_device),
            gpu_cuda_device=(int(args.gpu_cuda_device) if args.gpu_cuda_device is not None else None),
            cpu_device=str(args.cpu_device),
            torch_threads=int(args.torch_threads),
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
                gpu_device=str(args.gpu_device),
                gpu_cuda_device=(int(args.gpu_cuda_device) if args.gpu_cuda_device is not None else None),
                cpu_device=str(args.cpu_device),
                torch_threads=int(args.torch_threads),
                skip_existing=bool(args.skip_existing),
            )
        meta = read_suite_meta(paths.suite_meta)
        built_groups = [str(x) for x in (meta.get("selected_groups", []) or []) if str(x).strip()]
        selected_groups = select_known_items(
            requested=parse_items(args.groups),
            available=built_groups,
            item_name="lda tree recovery groups",
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

    if args.cmd == "report":
        from treepo._research.ctreepo.sim.cli.report.lda_tree_recovery_progress import main as _report_main  # noqa: WPS433

        report_argv = ["--input-root", str(Path(args.output_root).resolve())]
        if str(args.output_dir).strip():
            report_argv.extend(["--output-dir", str(Path(args.output_dir).resolve())])
        return int(_report_main(report_argv))

    raise ValueError("unreachable")


__all__ = ["build_suite", "main", "resolve_policy"]
