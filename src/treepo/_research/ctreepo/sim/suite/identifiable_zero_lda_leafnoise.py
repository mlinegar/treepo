from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence

from treepo._research.ctreepo.sim.cli.sweep_segmented_lda_ctreepo import _iter_runs as _iter_ctree_runs
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


@dataclass(frozen=True)
class IdentifiableZeroLdaLeafnoisePolicy:
    train_docs: tuple[int, ...]
    leaf_tokens: tuple[int, ...]
    calibration_rates: tuple[float, ...]
    seeds: tuple[int, ...]
    n_books_test: int
    n_topics: int
    vocab_size: int
    doc_tokens: int
    alpha_topic: float
    beta_word: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def resolve_policy() -> IdentifiableZeroLdaLeafnoisePolicy:
    return IdentifiableZeroLdaLeafnoisePolicy(
        train_docs=(16, 32, 64, 128, 256, 512, 1024, 2048),
        leaf_tokens=(2048, 512, 128, 32, 8),
        calibration_rates=(0.0, 0.1),
        seeds=(0, 1, 2, 3, 4, 5),
        n_books_test=2000,
        n_topics=4,
        vocab_size=256,
        doc_tokens=2048,
        alpha_topic=0.20,
        beta_word=0.10,
    )


def _build_groups(
    *,
    python_bin: str,
    output_root: Path,
    policy: IdentifiableZeroLdaLeafnoisePolicy,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
    skip_existing: bool,
) -> List[SuiteGroupRuns]:
    groups: List[SuiteGroupRuns] = []
    for leaf_tokens in policy.leaf_tokens:
        runs = _iter_ctree_runs(
            python_bin=str(python_bin),
            train_docs=policy.train_docs,
            seeds=policy.seeds,
            calibration_rates=policy.calibration_rates,
            eval_internal_rates=(0.0,),
            eval_leaf_rates=(0.0,),
            output_root=output_root / "segmented_lda_ctreepo" / "equivalence" / "lda_leafnoise",
            topic_phi_estimators=("sklearn_lda",),
            topic_phi_docs_values=(0,),
            leaf_theta_estimators=("sklearn_lda",),
            topic_processes=("bag_of_words",),
            n_topics=int(policy.n_topics),
            vocab_size=int(policy.vocab_size),
            min_segments=1,
            max_segments=1,
            min_seg_tokens=int(policy.doc_tokens),
            max_seg_tokens=int(policy.doc_tokens),
            fixed_leaf_tokens=int(leaf_tokens),
            n_books_test=int(policy.n_books_test),
            alpha_topic=float(policy.alpha_topic),
            beta_word=float(policy.beta_word),
            segment_concentration=80.0,
            segment_background=2.0,
            calibration_policy="uniform",
            eval_internal_query_design="risk",
            spectral_svd_dim_extra=2,
            spectral_max_leaves=4000,
            spectral_kmeans_inits=6,
            spectral_kmeans_max_iter=60,
            tlda_delta=0.10,
            tlda_rate_constant=1.0,
            tlda_sigmaK_floor=1e-6,
            topic_phi_permute=True,
            online_tensor_lda_burn_in_docs=0,
            online_tensor_lda_batch_docs=32,
            online_tensor_lda_passes=1,
            online_tensor_lda_lr=0.1,
            online_tensor_lda_grad_clip_norm=1.0,
            embedding_topic_svd_dim_extra=4,
            embedding_topic_kmeans_inits=8,
            embedding_topic_kmeans_max_iter=80,
            embedding_topic_assignment_temperature=0.35,
            embedding_topic_ppmi_shift=1.0,
            neural_topic_base_estimator="tensor_lda",
            neural_topic_seed_fraction_default=0.35,
            neural_topic_seed_fractions=(0.35,),
            neural_topic_hidden_dim=48,
            neural_topic_steps=60,
            neural_topic_lr=3e-3,
            neural_topic_weight_decay=1e-4,
            neural_topic_mix_samples=128,
            neural_topic_mix_temperature=1.0,
            neural_topic_operator_boost=1.4,
            neural_topic_seed_llm_min_weight=0.2,
            neural_topic_seed_llm_max_weight=0.55,
            neural_topic_similarity_temperature=0.15,
            neural_topic_ridge=1e-3,
            selection_audit_trials=0,
            leaf_theta_rf_n_estimators=200,
            leaf_theta_rf_max_depth=16,
            leaf_theta_rf_min_samples_leaf=5,
            leaf_theta_mlp_hidden_dim=128,
            leaf_theta_mlp_epochs=10,
            leaf_theta_mlp_batch_size=256,
            leaf_theta_mlp_lr=1e-3,
            leaf_theta_mlp_weight_decay=1e-4,
            include_full_doc_theta_baseline=False,
            device=str(device),
            cuda_device=(int(cuda_device) if cuda_device is not None else None),
            torch_threads=int(torch_threads),
            skip_existing=bool(skip_existing),
        )
        groups.append(
            SuiteGroupRuns(
                key=f"leaf_{int(leaf_tokens)}",
                family="segmented_lda_ctreepo",
                runs=runs,
            )
        )
    return groups


def build_suite(
    *,
    run_id: str,
    python_bin: str,
    output_root: Path,
    requested_groups: Sequence[str],
    device: str,
    cuda_device: int | None,
    torch_threads: int,
    skip_existing: bool,
) -> Dict[str, object]:
    output_root = output_root.resolve()
    paths = resolve_grouped_suite_paths(output_root)
    policy = resolve_policy()
    groups = _build_groups(
        python_bin=python_bin,
        output_root=output_root,
        policy=policy,
        device=device,
        cuda_device=cuda_device,
        torch_threads=torch_threads,
        skip_existing=skip_existing,
    )
    selected_groups = select_known_items(
        requested=requested_groups,
        available=[group.key for group in groups],
        item_name="lda leafnoise groups",
    )
    filtered_groups = [group for group in groups if group.key in selected_groups]
    artifacts = emit_grouped_suite_artifacts(paths, filtered_groups)
    meta = build_suite_meta(
        suite_name="identifiable-zero-lda-leafnoise",
        suite_role="appendix",
        run_id=str(run_id),
        profile="v1",
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
            "device": str(device),
            "cuda_device": int(cuda_device) if cuda_device is not None else None,
            "torch_threads": int(torch_threads),
            "skip_existing": bool(skip_existing),
            "counts_by_group": artifacts.counts_by_group,
            "n_commands_total": int(len(artifacts.all_cmds)),
        },
    )
    write_suite_meta(paths.suite_meta, meta)
    return meta


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Identifiable-Zero LDA leaf-noise suite.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("build", "run"):
        subp = sub.add_parser(name)
        subp.add_argument("--run-id", type=str, default="")
        subp.add_argument("--python-bin", type=str, default="")
        subp.add_argument("--output-root", type=str, default="" if name == "build" else None, required=name == "run")
        subp.add_argument("--groups", type=str, default="")
        subp.add_argument("--device", type=str, default="auto")
        subp.add_argument("--cuda-device", type=int, default=None)
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
    report.add_argument("--ctreepo-root", type=str, default="")
    report.add_argument("--out-dir", type=str, default="")
    report.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    if args.cmd == "build":
        run_id = utc_run_id(args.run_id)
        output_root = Path(args.output_root) if str(args.output_root).strip() else Path(f"outputs/identifiable_zero_lda_leafnoise_{run_id}")
        meta = build_suite(
            run_id=run_id,
            python_bin=str(args.python_bin).strip() or sys.executable,
            output_root=output_root,
            requested_groups=parse_items(args.groups),
            device=str(args.device),
            cuda_device=(int(args.cuda_device) if args.cuda_device is not None else None),
            torch_threads=int(args.torch_threads),
            skip_existing=bool(args.skip_existing),
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
                python_bin=str(args.python_bin).strip() or sys.executable,
                output_root=output_root,
                requested_groups=parse_items(args.groups),
                device=str(args.device),
                cuda_device=(int(args.cuda_device) if args.cuda_device is not None else None),
                torch_threads=int(args.torch_threads),
                skip_existing=bool(args.skip_existing),
            )
        meta = read_suite_meta(paths.suite_meta)
        built_groups = [str(x) for x in (meta.get("selected_groups", []) or []) if str(x).strip()]
        selected_groups = select_known_items(
            requested=parse_items(args.groups),
            available=built_groups,
            item_name="lda leafnoise groups",
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
        from treepo._research.ctreepo.sim.cli.report.identifiable_zero_lda_leafnoise import main as _report_main  # noqa: WPS433

        report_argv = ["--output-root", str(Path(args.output_root).resolve())]
        if str(args.ctreepo_root).strip():
            report_argv.extend(["--ctreepo-root", str(Path(args.ctreepo_root).resolve())])
        if str(args.out_dir).strip():
            report_argv.extend(["--out-dir", str(Path(args.out_dir).resolve())])
        report_argv.append("--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf")
        return int(_report_main(report_argv))

    raise ValueError("unreachable")


__all__ = ["build_suite", "main", "resolve_policy"]
