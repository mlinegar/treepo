from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence

from treepo._research.ctreepo.sim.cli.sweep_markov_changepoint_ops_count import _iter_runs as _iter_markov_runs
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
class IdentifiableZeroNeuralOperatorPolicy:
    markov_schedule_consistency_weights: tuple[float, ...]
    markov_seeds: tuple[int, ...]
    ctree_eval_rates: tuple[float, ...]
    ctree_topic_estimators: tuple[str, ...]
    ctree_seed_fractions: tuple[float, ...]
    ctree_seeds: tuple[int, ...]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def resolve_policy() -> IdentifiableZeroNeuralOperatorPolicy:
    return IdentifiableZeroNeuralOperatorPolicy(
        markov_schedule_consistency_weights=(0.0, 0.01, 0.03, 0.1, 0.3),
        markov_seeds=tuple(range(12)),
        ctree_eval_rates=(0.0, 0.5, 1.0),
        ctree_topic_estimators=(
            "spectral_numpy",
            "tensor_lda",
            "online_tensor_lda",
            "neural_ctreepo",
            "neural_mergeable_sketch",
            "neural_hybrid",
        ),
        ctree_seed_fractions=(0.35, 0.5, 0.75, 1.0),
        ctree_seeds=(0, 1, 2, 3, 4, 5),
    )


def _build_groups(
    *,
    python_bin: str,
    output_root: Path,
    policy: IdentifiableZeroNeuralOperatorPolicy,
    markov_device: str,
    markov_cuda_device: int | None,
    ctree_device: str,
    ctree_cuda_device: int | None,
    torch_threads: int,
    skip_existing: bool,
) -> List[SuiteGroupRuns]:
    markov_runs = _iter_markov_runs(
        python_bin=str(python_bin),
        n_regimes=4,
        vocab_size=96,
        min_tokens=384,
        max_tokens=384,
        min_segments=12,
        max_segments=24,
        fixed_leaf_tokens=16,
        train_docs=(8000,),
        val_docs=0,
        test_docs=2000,
        audit_fractions=(0.1, 1.0),
        c3_audit_strategies=("uniform",),
        c3_include_root=True,
        leaf_query_rates=(1.0,),
        include_root_queries=(True,),
        local_law_weights=(0.0,),
        task_objective_weights=(),
        c1_relative_weights=(1.0,),
        c2_relative_weights=(1.0,),
        c3_relative_weights=(1.0,),
        c2_weights=(0.0,),
        root_weights=(1.0,),
        schedule_consistency_weights=policy.markov_schedule_consistency_weights,
        guidance_override_modes=("adjust",),
        eval_guidance_qs=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
        eval_guidance_trials=8,
        eval_guidance_seed_offset=100000,
        eval_guidance_include_root=True,
        include_rf_root_baseline=False,
        include_doc_level_baseline=False,
        rf_n_estimators=200,
        rf_max_depth=16,
        rf_min_samples_leaf=5,
        data_seeds=(),
        seeds=policy.markov_seeds,
        output_root=output_root / "markov_changepoint_ops_count" / "equivalence",
        model_families=("neural",),
        feature_modes=("full",),
        state_dims=(32,),
        hidden_dims=(),
        hidden_dim_multiplier=4.0,
        hidden_dim_min=64,
        n_epochs=12,
        device=str(markov_device),
        cuda_device=(int(markov_cuda_device) if markov_cuda_device is not None else None),
        violation_tau=0.0,
        torch_threads=int(torch_threads),
        skip_existing=bool(skip_existing),
        suite_role="neural_operator",
    )

    ctree_runs = _iter_ctree_runs(
        python_bin=str(python_bin),
        train_docs=(4096,),
        seeds=policy.ctree_seeds,
        calibration_rates=(0.1,),
        eval_internal_rates=policy.ctree_eval_rates,
        eval_leaf_rates=policy.ctree_eval_rates,
        output_root=output_root / "segmented_lda_ctreepo" / "equivalence",
        topic_phi_estimators=policy.ctree_topic_estimators,
        topic_phi_docs_values=(4096,),
        leaf_theta_estimators=("lstsq",),
        topic_processes=("segments",),
        n_topics=4,
        vocab_size=256,
        min_segments=6,
        max_segments=6,
        min_seg_tokens=24,
        max_seg_tokens=48,
        fixed_leaf_tokens=32,
        n_books_test=5000,
        alpha_topic=0.4,
        beta_word=0.2,
        segment_concentration=12.0,
        segment_background=4.0,
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
        neural_topic_seed_fractions=policy.ctree_seed_fractions,
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
        device=str(ctree_device),
        cuda_device=(int(ctree_cuda_device) if ctree_cuda_device is not None else None),
        torch_threads=int(torch_threads),
        skip_existing=bool(skip_existing),
    )

    return [
        SuiteGroupRuns(key="markov_schedule_consistency", family="markov_changepoint_ops_count", runs=markov_runs),
        SuiteGroupRuns(key="ctree_operator_family", family="segmented_lda_ctreepo", runs=ctree_runs),
    ]


def build_suite(
    *,
    run_id: str,
    python_bin: str,
    output_root: Path,
    requested_groups: Sequence[str],
    markov_device: str,
    markov_cuda_device: int | None,
    ctree_device: str,
    ctree_cuda_device: int | None,
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
        markov_device=markov_device,
        markov_cuda_device=markov_cuda_device,
        ctree_device=ctree_device,
        ctree_cuda_device=ctree_cuda_device,
        torch_threads=torch_threads,
        skip_existing=skip_existing,
    )
    selected_groups = select_known_items(
        requested=requested_groups,
        available=[group.key for group in groups],
        item_name="neural-operator groups",
    )
    filtered_groups = [group for group in groups if group.key in selected_groups]
    artifacts = emit_grouped_suite_artifacts(paths, filtered_groups)
    meta = build_suite_meta(
        suite_name="identifiable-zero-neural-operator",
        suite_role="appendix",
        run_id=str(run_id),
        profile="v2",
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
            "markov_device": str(markov_device),
            "markov_cuda_device": int(markov_cuda_device) if markov_cuda_device is not None else None,
            "ctree_device": str(ctree_device),
            "ctree_cuda_device": int(ctree_cuda_device) if ctree_cuda_device is not None else None,
            "torch_threads": int(torch_threads),
            "skip_existing": bool(skip_existing),
            "counts_by_group": artifacts.counts_by_group,
            "n_commands_total": int(len(artifacts.all_cmds)),
        },
    )
    write_suite_meta(paths.suite_meta, meta)
    return meta


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Identifiable-Zero neural-operator suite.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("build", "run"):
        subp = sub.add_parser(name)
        subp.add_argument("--run-id", type=str, default="")
        subp.add_argument("--python-bin", type=str, default="")
        subp.add_argument("--output-root", type=str, default="" if name == "build" else None, required=name == "run")
        subp.add_argument("--groups", type=str, default="")
        subp.add_argument("--markov-device", type=str, default="auto")
        subp.add_argument("--markov-cuda-device", type=int, default=None)
        subp.add_argument("--ctree-device", type=str, default="auto")
        subp.add_argument("--ctree-cuda-device", type=int, default=None)
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
    report.add_argument("--out-dir", type=str, default="")
    report.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    if args.cmd == "build":
        run_id = utc_run_id(args.run_id)
        output_root = Path(args.output_root) if str(args.output_root).strip() else Path(f"outputs/identifiable_zero_neural_operator_v2_{run_id}")
        meta = build_suite(
            run_id=run_id,
            python_bin=str(args.python_bin).strip() or sys.executable,
            output_root=output_root,
            requested_groups=parse_items(args.groups),
            markov_device=str(args.markov_device),
            markov_cuda_device=(int(args.markov_cuda_device) if args.markov_cuda_device is not None else None),
            ctree_device=str(args.ctree_device),
            ctree_cuda_device=(int(args.ctree_cuda_device) if args.ctree_cuda_device is not None else None),
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
                markov_device=str(args.markov_device),
                markov_cuda_device=(int(args.markov_cuda_device) if args.markov_cuda_device is not None else None),
                ctree_device=str(args.ctree_device),
                ctree_cuda_device=(int(args.ctree_cuda_device) if args.ctree_cuda_device is not None else None),
                torch_threads=int(args.torch_threads),
                skip_existing=bool(args.skip_existing),
            )
        meta = read_suite_meta(paths.suite_meta)
        built_groups = [str(x) for x in (meta.get("selected_groups", []) or []) if str(x).strip()]
        selected_groups = select_known_items(
            requested=parse_items(args.groups),
            available=built_groups,
            item_name="neural-operator groups",
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
        from treepo._research.ctreepo.sim.cli.report.identifiable_zero_neural_operator import main as _report_main  # noqa: WPS433

        report_argv = ["--overnight-output-root", str(Path(args.output_root).resolve())]
        if str(args.out_dir).strip():
            report_argv.extend(["--out-dir", str(Path(args.out_dir).resolve())])
        report_argv.append("--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf")
        return int(_report_main(report_argv))

    raise ValueError("unreachable")


__all__ = ["build_suite", "main", "resolve_policy"]
