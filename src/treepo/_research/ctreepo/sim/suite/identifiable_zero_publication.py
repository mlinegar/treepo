from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import shlex
from typing import Dict, Iterable, List, Sequence, Tuple

from treepo._research.ctreepo.sim.cli.sweep_markov_changepoint_ops_count import _iter_runs as _iter_markov_runs
from treepo._research.ctreepo.sim.cli.sweep_segment_lda_ops_weight_recovery import _iter_runs as _iter_segment_runs
from treepo._research.ctreepo.sim.cli.sweep_segmented_lda_ctreepo import _iter_runs as _iter_ctree_runs
from treepo._research.ctreepo.sim.manifest import RunSpec, read_manifest_jsonl, write_manifest_jsonl
from treepo._research.ctreepo.sim.suite.common import (
    build_suite_meta,
    parse_items,
    read_suite_meta,
    run_manifest_queue_suite,
    select_known_items,
    utc_run_id,
    write_suite_meta,
    write_text,
)
from treepo._research.ctreepo.sim.suite.identifiable_zero_publication_policy import (
    CtreeSweepPolicy,
    IdentifiableZeroLongrunPolicy,
    IdentifiableZeroPublicationCleanPolicy,
    MarkovSweepPolicy,
    SegmentOpsSweepPolicy,
    resolve_identifiable_zero_longrun_policy,
    resolve_identifiable_zero_publication_clean_policy,
)


def _q(x: object) -> str:
    return shlex.quote(str(x))


@dataclass(frozen=True)
class SuitePaths:
    output_root: Path
    figures_root: Path
    suite_meta: Path
    suite_cmds: Path
    suite_manifest: Path
    suite_plot_cmds: Path
    group_cmd_dir: Path
    group_manifest_dir: Path
    queue_log_dir: Path


@dataclass(frozen=True)
class SuiteArtifacts:
    group_cmd_files: Dict[str, str]
    group_manifest_files: Dict[str, str]
    counts_by_group: Dict[str, int]


def _resolve_paths(*, output_root: Path, figures_root: Path) -> SuitePaths:
    return SuitePaths(
        output_root=output_root,
        figures_root=figures_root,
        suite_meta=output_root / "suite_meta.json",
        suite_cmds=output_root / "suite_cmds.txt",
        suite_manifest=output_root / "suite_manifest.jsonl",
        suite_plot_cmds=output_root / "suite_plot_cmds.txt",
        group_cmd_dir=output_root / "suite_groups" / "cmds",
        group_manifest_dir=output_root / "suite_groups" / "manifests",
        queue_log_dir=output_root / "queue_logs",
    )


def _available_groups(profile: str) -> List[str]:
    if str(profile) == "publication_clean":
        return ["cpu", "gpu"]
    if str(profile) == "longrun_equiv_v1":
        return ["equiv", "scale", "pilot"]
    raise ValueError(f"unknown publication profile: {profile}")


def _default_groups(profile: str) -> List[str]:
    if str(profile) == "publication_clean":
        return ["cpu", "gpu"]
    if str(profile) == "longrun_equiv_v1":
        return ["equiv", "scale"]
    raise ValueError(f"unknown publication profile: {profile}")


def _selected_groups(*, profile: str, requested: Sequence[str]) -> List[str]:
    if not requested:
        return _default_groups(str(profile))
    return select_known_items(
        requested=requested,
        available=_available_groups(str(profile)),
        item_name=f"{profile} groups",
    )


def _emit_grouped_artifacts(
    *,
    paths: SuitePaths,
    groups: Dict[str, List[RunSpec]],
    aggregate_groups: Sequence[str],
) -> SuiteArtifacts:
    paths.output_root.mkdir(parents=True, exist_ok=True)
    paths.figures_root.mkdir(parents=True, exist_ok=True)
    paths.group_cmd_dir.mkdir(parents=True, exist_ok=True)
    paths.group_manifest_dir.mkdir(parents=True, exist_ok=True)

    group_cmd_files: Dict[str, str] = {}
    group_manifest_files: Dict[str, str] = {}
    counts_by_group: Dict[str, int] = {}
    aggregate_cmds: List[str] = []
    aggregate_runs: List[RunSpec] = []

    for group_key, runs in groups.items():
        cmd_path = paths.group_cmd_dir / f"{group_key}.txt"
        manifest_path = paths.group_manifest_dir / f"{group_key}.jsonl"
        cmds = [run.command for run in runs]
        write_text(cmd_path, "\n".join(cmds) + ("\n" if cmds else ""))
        write_manifest_jsonl(manifest_path, runs)
        group_cmd_files[group_key] = str(cmd_path)
        group_manifest_files[group_key] = str(manifest_path)
        counts_by_group[group_key] = int(len(runs))
        if group_key in aggregate_groups:
            aggregate_cmds.extend(cmds)
            aggregate_runs.extend(runs)

    write_text(paths.suite_cmds, "\n".join(aggregate_cmds) + ("\n" if aggregate_cmds else ""))
    write_manifest_jsonl(paths.suite_manifest, aggregate_runs)
    return SuiteArtifacts(
        group_cmd_files=group_cmd_files,
        group_manifest_files=group_manifest_files,
        counts_by_group=counts_by_group,
    )


def _segment_runs(
    *,
    policy: SegmentOpsSweepPolicy,
    python_bin: str,
    output_root: Path,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
    skip_existing: bool,
) -> List[RunSpec]:
    return _iter_segment_runs(
        python_bin=str(python_bin),
        train_docs=policy.train_docs,
        test_docs=int(policy.test_docs),
        audit_fractions=policy.audit_fractions,
        topic_phi_docs=policy.topic_phi_docs,
        topic_phi_estimators=policy.topic_phi_estimators,
        topic_processes=policy.topic_processes,
        lambda_multipliers=policy.lambda_multipliers,
        seeds=policy.seeds,
        output_root=output_root,
        topic_source=str(policy.topic_source),
        feature_inference=str(policy.feature_inference),
        n_topics=int(policy.n_topics),
        vocab_size=int(policy.vocab_size),
        min_tokens=int(policy.min_tokens),
        max_tokens=int(policy.max_tokens),
        leaf_tokens=int(policy.leaf_tokens),
        device=str(device),
        cuda_device=(int(cuda_device) if cuda_device is not None else None),
        torch_threads=int(torch_threads),
        run_all_feature_modes=bool(policy.run_all_feature_modes),
        skip_existing=bool(skip_existing),
    )


def _ctree_runs(
    *,
    policy: CtreeSweepPolicy,
    python_bin: str,
    output_root: Path,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
    skip_existing: bool,
    coupled_only: bool,
) -> List[RunSpec]:
    if not coupled_only:
        return _iter_ctree_runs(
            python_bin=str(python_bin),
            train_docs=policy.train_docs,
            seeds=policy.seeds,
            calibration_rates=policy.calibration_rates,
            eval_internal_rates=policy.eval_internal_rates,
            eval_leaf_rates=policy.eval_leaf_rates,
            output_root=output_root,
            topic_phi_estimators=policy.topic_phi_estimators,
            topic_phi_docs_values=policy.topic_phi_docs_values,
            leaf_theta_estimators=policy.leaf_theta_estimators,
            topic_processes=policy.topic_processes,
            n_topics=int(policy.n_topics),
            vocab_size=int(policy.vocab_size),
            min_segments=int(policy.min_segments),
            max_segments=int(policy.max_segments),
            min_seg_tokens=int(policy.min_seg_tokens),
            max_seg_tokens=int(policy.max_seg_tokens),
            fixed_leaf_tokens=int(policy.fixed_leaf_tokens),
            n_books_test=int(policy.n_books_test),
            alpha_topic=float(policy.alpha_topic),
            beta_word=float(policy.beta_word),
            segment_concentration=float(policy.segment_concentration),
            segment_background=float(policy.segment_background),
            calibration_policy=str(policy.calibration_policy),
            eval_internal_query_design=str(policy.eval_internal_query_design),
            spectral_svd_dim_extra=2,
            spectral_max_leaves=4000,
            spectral_kmeans_inits=6,
            spectral_kmeans_max_iter=60,
            tlda_delta=0.10,
            tlda_rate_constant=1.0,
            tlda_sigmaK_floor=1e-6,
            topic_phi_permute=bool(policy.topic_phi_permute),
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
            neural_topic_seed_fractions=[0.35],
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
            include_full_doc_theta_baseline=bool(policy.include_full_doc_theta_baseline),
            device=str(device),
            cuda_device=(int(cuda_device) if cuda_device is not None else None),
            torch_threads=int(torch_threads),
            skip_existing=bool(skip_existing),
        )

    runs: List[RunSpec] = []
    for q in policy.eval_leaf_rates:
        runs.extend(
            _iter_ctree_runs(
                python_bin=str(python_bin),
                train_docs=policy.train_docs,
                seeds=policy.seeds,
                calibration_rates=policy.calibration_rates,
                eval_internal_rates=[float(q)],
                eval_leaf_rates=[float(q)],
                output_root=output_root,
                topic_phi_estimators=policy.topic_phi_estimators,
                topic_phi_docs_values=policy.topic_phi_docs_values,
                leaf_theta_estimators=policy.leaf_theta_estimators,
                topic_processes=policy.topic_processes,
                n_topics=int(policy.n_topics),
                vocab_size=int(policy.vocab_size),
                min_segments=int(policy.min_segments),
                max_segments=int(policy.max_segments),
                min_seg_tokens=int(policy.min_seg_tokens),
                max_seg_tokens=int(policy.max_seg_tokens),
                fixed_leaf_tokens=int(policy.fixed_leaf_tokens),
                n_books_test=int(policy.n_books_test),
                alpha_topic=float(policy.alpha_topic),
                beta_word=float(policy.beta_word),
                segment_concentration=float(policy.segment_concentration),
                segment_background=float(policy.segment_background),
                calibration_policy=str(policy.calibration_policy),
                eval_internal_query_design=str(policy.eval_internal_query_design),
                spectral_svd_dim_extra=2,
                spectral_max_leaves=4000,
                spectral_kmeans_inits=6,
                spectral_kmeans_max_iter=60,
                tlda_delta=0.10,
                tlda_rate_constant=1.0,
                tlda_sigmaK_floor=1e-6,
                topic_phi_permute=bool(policy.topic_phi_permute),
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
                neural_topic_seed_fractions=[0.35],
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
                include_full_doc_theta_baseline=bool(policy.include_full_doc_theta_baseline),
                device=str(device),
                cuda_device=(int(cuda_device) if cuda_device is not None else None),
                torch_threads=int(torch_threads),
                skip_existing=bool(skip_existing),
            )
        )
    return runs


def _markov_runs(
    *,
    policy: MarkovSweepPolicy,
    python_bin: str,
    output_root: Path,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
    skip_existing: bool,
    model_families: Iterable[str] | None = None,
) -> List[RunSpec]:
    return _iter_markov_runs(
        python_bin=str(python_bin),
        n_regimes=int(policy.n_regimes),
        vocab_size=int(policy.vocab_size),
        min_tokens=int(policy.min_tokens),
        max_tokens=int(policy.max_tokens),
        min_segments=int(policy.min_segments),
        max_segments=int(policy.max_segments),
        fixed_leaf_tokens=int(policy.fixed_leaf_tokens),
        train_docs=policy.train_docs,
        val_docs=int(policy.val_docs),
        test_docs=int(policy.test_docs),
        audit_fractions=policy.audit_fractions,
        c3_audit_strategies=policy.c3_audit_strategies,
        c3_include_root=bool(policy.c3_include_root),
        leaf_query_rates=policy.leaf_query_rates,
        include_root_queries=policy.include_root_queries,
        local_law_weights=policy.local_law_weights,
        task_objective_weights=policy.task_objective_weights,
        c1_relative_weights=policy.c1_relative_weights,
        c2_relative_weights=policy.c2_relative_weights,
        c3_relative_weights=policy.c3_relative_weights,
        c2_weights=policy.c2_weights,
        root_weights=policy.root_weights,
        schedule_consistency_weights=policy.schedule_consistency_weights,
        guidance_override_modes=policy.guidance_override_modes,
        eval_guidance_qs=policy.eval_guidance_qs,
        eval_guidance_trials=int(policy.eval_guidance_trials),
        eval_guidance_seed_offset=int(policy.eval_guidance_seed_offset),
        eval_guidance_include_root=bool(policy.eval_guidance_include_root),
        include_rf_root_baseline=bool(policy.include_rf_root_baseline),
        include_doc_level_baseline=bool(policy.include_doc_level_baseline),
        rf_n_estimators=int(policy.rf_n_estimators),
        rf_max_depth=int(policy.rf_max_depth),
        rf_min_samples_leaf=int(policy.rf_min_samples_leaf),
        data_seeds=policy.data_seeds,
        seeds=policy.seeds,
        output_root=output_root,
        model_families=tuple(model_families) if model_families is not None else policy.model_families,
        feature_modes=policy.feature_modes,
        state_dims=policy.state_dims,
        hidden_dims=policy.hidden_dims,
        hidden_dim_multiplier=policy.hidden_dim_multiplier,
        hidden_dim_min=int(policy.hidden_dim_min),
        n_epochs=int(policy.n_epochs),
        device=str(device),
        cuda_device=(int(cuda_device) if cuda_device is not None else None),
        violation_tau=float(policy.violation_tau),
        torch_threads=int(torch_threads),
        skip_existing=bool(skip_existing),
        suite_role=str(policy.suite_role),
        law_packages=(),
        exact_families=(),
    )


def _build_publication_clean_groups(
    *,
    needed_groups: Sequence[str],
    policy: IdentifiableZeroPublicationCleanPolicy,
    python_bin: str,
    output_root: Path,
    skip_existing: bool,
    segment_device: str,
    segment_cuda_device: int | None,
    ctree_device: str,
    ctree_cuda_device: int | None,
    markov_additive_device: str,
    markov_additive_cuda_device: int | None,
    markov_neural_device: str,
    markov_neural_cuda_device: int | None,
    torch_threads: int,
) -> Tuple[Dict[str, List[RunSpec]], Dict[str, int]]:
    needed = set(str(x) for x in needed_groups)
    build_cpu = "cpu" in needed
    build_gpu = "gpu" in needed
    segment_runs = (
        _segment_runs(
            policy=policy.segment,
            python_bin=python_bin,
            output_root=output_root / "segment_lda_ops_weight_recovery" / "publication_clean",
            device=str(segment_device),
            cuda_device=segment_cuda_device,
            torch_threads=int(torch_threads),
            skip_existing=bool(skip_existing),
        )
        if build_cpu
        else []
    )
    ctree_runs = (
        _ctree_runs(
            policy=policy.ctree,
            python_bin=python_bin,
            output_root=output_root / "segmented_lda_ctreepo" / "equivalence",
            device=str(ctree_device),
            cuda_device=ctree_cuda_device,
            torch_threads=int(torch_threads),
            skip_existing=bool(skip_existing),
            coupled_only=True,
        )
        if build_cpu
        else []
    )
    markov_additive_runs = (
        _markov_runs(
            policy=policy.markov,
            python_bin=python_bin,
            output_root=output_root / "markov_changepoint_ops_count" / "equivalence",
            device=str(markov_additive_device),
            cuda_device=markov_additive_cuda_device,
            torch_threads=int(torch_threads),
            skip_existing=bool(skip_existing),
            model_families=("additive",),
        )
        if build_cpu
        else []
    )
    markov_neural_runs = (
        _markov_runs(
            policy=policy.markov,
            python_bin=python_bin,
            output_root=output_root / "markov_changepoint_ops_count" / "equivalence" / "family_neural",
            device=str(markov_neural_device),
            cuda_device=markov_neural_cuda_device,
            torch_threads=int(torch_threads),
            skip_existing=bool(skip_existing),
            model_families=("neural",),
        )
        if build_gpu
        else []
    )
    groups = {
        "cpu": [*segment_runs, *ctree_runs, *markov_additive_runs],
        "gpu": [*markov_neural_runs],
    }
    counts = {
        "segment": int(len(segment_runs)),
        "ctree_coupled": int(len(ctree_runs)),
        "markov_additive": int(len(markov_additive_runs)),
        "markov_neural": int(len(markov_neural_runs)),
        "cpu_total": int(len(groups["cpu"])),
        "gpu_total": int(len(groups["gpu"])),
    }
    return groups, counts


def _build_longrun_groups(
    *,
    needed_groups: Sequence[str],
    policy: IdentifiableZeroLongrunPolicy,
    python_bin: str,
    output_root: Path,
    skip_existing: bool,
    segment_device: str,
    segment_cuda_device: int | None,
    ctree_device: str,
    ctree_cuda_device: int | None,
    markov_device: str,
    markov_cuda_device: int | None,
    torch_threads: int,
) -> Tuple[Dict[str, List[RunSpec]], Dict[str, int]]:
    needed = set(str(x) for x in needed_groups)
    build_scale = "scale" in needed or "pilot" in needed
    build_equiv = "equiv" in needed or "pilot" in needed
    segment_runs = (
        _segment_runs(
            policy=policy.segment_scale,
            python_bin=python_bin,
            output_root=output_root / "segment_lda_ops_weight_recovery" / "scale",
            device=str(segment_device),
            cuda_device=segment_cuda_device,
            torch_threads=int(torch_threads),
            skip_existing=bool(skip_existing),
        )
        if build_scale
        else []
    )
    ctree_scale_runs = (
        _ctree_runs(
            policy=policy.ctree_scale,
            python_bin=python_bin,
            output_root=output_root / "segmented_lda_ctreepo" / "scale",
            device=str(ctree_device),
            cuda_device=ctree_cuda_device,
            torch_threads=int(torch_threads),
            skip_existing=bool(skip_existing),
            coupled_only=False,
        )
        if build_scale
        else []
    )
    ctree_equiv_runs = (
        _ctree_runs(
            policy=policy.ctree_equiv,
            python_bin=python_bin,
            output_root=output_root / "segmented_lda_ctreepo" / "equivalence",
            device=str(ctree_device),
            cuda_device=ctree_cuda_device,
            torch_threads=int(torch_threads),
            skip_existing=bool(skip_existing),
            coupled_only=True,
        )
        if build_equiv
        else []
    )
    markov_scale_runs = (
        _markov_runs(
            policy=policy.markov_scale,
            python_bin=python_bin,
            output_root=output_root / "markov_changepoint_ops_count" / "scale",
            device=str(markov_device),
            cuda_device=markov_cuda_device,
            torch_threads=int(torch_threads),
            skip_existing=bool(skip_existing),
        )
        if build_scale
        else []
    )
    markov_equiv_runs = (
        _markov_runs(
            policy=policy.markov_equiv,
            python_bin=python_bin,
            output_root=output_root / "markov_changepoint_ops_count" / "equivalence",
            device=str(markov_device),
            cuda_device=markov_cuda_device,
            torch_threads=int(torch_threads),
            skip_existing=bool(skip_existing),
        )
        if build_equiv
        else []
    )
    scale_runs = [*segment_runs, *ctree_scale_runs, *markov_scale_runs]
    equiv_runs = [*ctree_equiv_runs, *markov_equiv_runs]
    pilot_runs = [*equiv_runs, *scale_runs][: int(max(0, policy.pilot_cmd_count))]
    groups = {
        "equiv": equiv_runs,
        "scale": scale_runs,
        "pilot": pilot_runs,
    }
    counts = {
        "segment_scale": int(len(segment_runs)),
        "ctree_scale": int(len(ctree_scale_runs)),
        "ctree_equiv_coupled": int(len(ctree_equiv_runs)),
        "markov_scale": int(len(markov_scale_runs)),
        "markov_equiv": int(len(markov_equiv_runs)),
        "equiv_total": int(len(equiv_runs)),
        "scale_total": int(len(scale_runs)),
        "pilot_total": int(len(pilot_runs)),
    }
    return groups, counts


def _apply_clean_overrides(
    *,
    policy: IdentifiableZeroPublicationCleanPolicy,
    segment_test_docs: int,
    ctree_test_books: int,
    markov_test_docs: int,
    markov_n_epochs: int,
) -> IdentifiableZeroPublicationCleanPolicy:
    segment = replace(policy.segment, test_docs=int(segment_test_docs))
    ctree = replace(policy.ctree, n_books_test=int(ctree_test_books))
    markov = replace(policy.markov, test_docs=int(markov_test_docs), n_epochs=int(markov_n_epochs))
    return IdentifiableZeroPublicationCleanPolicy(profile=policy.profile, segment=segment, ctree=ctree, markov=markov)


def _apply_longrun_overrides(
    *,
    policy: IdentifiableZeroLongrunPolicy,
    segment_test_docs: int,
    ctree_test_books: int,
    markov_test_docs: int,
    markov_n_epochs: int,
) -> IdentifiableZeroLongrunPolicy:
    return IdentifiableZeroLongrunPolicy(
        profile=policy.profile,
        segment_scale=replace(policy.segment_scale, test_docs=int(segment_test_docs)),
        ctree_scale=replace(policy.ctree_scale, n_books_test=int(ctree_test_books)),
        ctree_equiv=replace(policy.ctree_equiv, n_books_test=int(ctree_test_books)),
        markov_scale=replace(policy.markov_scale, test_docs=int(markov_test_docs), n_epochs=int(markov_n_epochs)),
        markov_equiv=replace(policy.markov_equiv, test_docs=int(markov_test_docs), n_epochs=int(markov_n_epochs)),
        pilot_cmd_count=int(policy.pilot_cmd_count),
        target_main_jobs=int(policy.target_main_jobs),
        target_pilot_minutes=int(policy.target_pilot_minutes),
    )


def _write_plot_cmds(*, profile: str, python_bin: str, output_root: Path, figures_root: Path, path: Path) -> None:
    plot_cmds = [
        " ".join(
            [
                str(python_bin),
                "-u",
                "scripts/check_oracle_equivalence_invariants.py",
                "--output-root",
                _q(output_root),
                "--ceiling-threshold",
                "1e-8",
                "--hard-guided-threshold",
                "1e-12",
                "--output-json",
                _q(figures_root / "oracle_equivalence_invariants.json"),
            ]
        ),
        " ".join(
            [
                str(python_bin),
                "-m",
                "src.ctreepo.cli",
                "sim",
                "suite",
                "identifiable-zero-publication",
                "report",
                "--profile",
                "publication_clean",
                "--output-root",
                _q(output_root),
                "--emit-pdf",
            ]
        ),
    ]
    write_text(path, "\n".join(plot_cmds) + ("\n" if plot_cmds else ""))


def build_suite(
    *,
    run_id: str,
    profile: str,
    python_bin: str,
    output_root: Path,
    figures_root: Path,
    requested_groups: Sequence[str],
    skip_existing: bool,
    segment_device: str,
    segment_cuda_device: int | None,
    ctree_device: str,
    ctree_cuda_device: int | None,
    markov_device: str,
    markov_cuda_device: int | None,
    markov_additive_device: str,
    markov_additive_cuda_device: int | None,
    markov_neural_device: str,
    markov_neural_cuda_device: int | None,
    torch_threads: int,
    n_seeds: int,
    markov_local_law_weight: float,
    markov_task_objective_weight: float,
    pilot_cmd_count: int,
    target_main_jobs: int,
    target_pilot_minutes: int,
    segment_test_docs: int,
    ctree_test_books: int,
    markov_test_docs: int,
    markov_n_epochs: int,
) -> Dict[str, object]:
    output_root = output_root.resolve()
    figures_root = figures_root.resolve()
    paths = _resolve_paths(output_root=output_root, figures_root=figures_root)
    selected_groups = _selected_groups(profile=str(profile), requested=requested_groups)

    if str(profile) == "publication_clean":
        policy = resolve_identifiable_zero_publication_clean_policy(
            n_seeds=int(n_seeds),
            markov_local_law_weight=float(markov_local_law_weight),
            markov_task_objective_weight=float(markov_task_objective_weight),
        )
        policy = _apply_clean_overrides(
            policy=policy,
            segment_test_docs=int(segment_test_docs),
            ctree_test_books=int(ctree_test_books),
            markov_test_docs=int(markov_test_docs),
            markov_n_epochs=int(markov_n_epochs),
        )
        groups, counts = _build_publication_clean_groups(
            needed_groups=selected_groups,
            policy=policy,
            python_bin=python_bin,
            output_root=output_root,
            skip_existing=bool(skip_existing),
            segment_device=str(segment_device),
            segment_cuda_device=segment_cuda_device,
            ctree_device=str(ctree_device),
            ctree_cuda_device=ctree_cuda_device,
            markov_additive_device=str(markov_additive_device),
            markov_additive_cuda_device=markov_additive_cuda_device,
            markov_neural_device=str(markov_neural_device),
            markov_neural_cuda_device=markov_neural_cuda_device,
            torch_threads=int(torch_threads),
        )
        policy_dict = policy.to_dict()
    elif str(profile) == "longrun_equiv_v1":
        policy = resolve_identifiable_zero_longrun_policy(
            n_seeds=int(n_seeds),
            pilot_cmd_count=int(pilot_cmd_count),
            target_main_jobs=int(target_main_jobs),
            target_pilot_minutes=int(target_pilot_minutes),
        )
        policy = _apply_longrun_overrides(
            policy=policy,
            segment_test_docs=int(segment_test_docs),
            ctree_test_books=int(ctree_test_books),
            markov_test_docs=int(markov_test_docs),
            markov_n_epochs=int(markov_n_epochs),
        )
        groups, counts = _build_longrun_groups(
            needed_groups=selected_groups,
            policy=policy,
            python_bin=python_bin,
            output_root=output_root,
            skip_existing=bool(skip_existing),
            segment_device=str(segment_device),
            segment_cuda_device=segment_cuda_device,
            ctree_device=str(ctree_device),
            ctree_cuda_device=ctree_cuda_device,
            markov_device=str(markov_device),
            markov_cuda_device=markov_cuda_device,
            torch_threads=int(torch_threads),
        )
        policy_dict = policy.to_dict()
    else:
        raise ValueError(f"unknown publication profile: {profile}")

    artifacts = _emit_grouped_artifacts(paths=paths, groups=groups, aggregate_groups=selected_groups)
    _write_plot_cmds(
        profile=str(profile),
        python_bin=str(python_bin),
        output_root=output_root,
        figures_root=figures_root,
        path=paths.suite_plot_cmds,
    )

    meta: Dict[str, object] = build_suite_meta(
        suite_name="identifiable-zero-publication",
        suite_role="paper",
        run_id=str(run_id),
        profile=str(profile),
        policy=policy_dict,
        python_bin=str(python_bin),
        output_root=output_root,
        cmds_file=paths.suite_cmds,
        manifest_file=paths.suite_manifest,
        selected_groups=list(selected_groups),
        group_cmd_files=artifacts.group_cmd_files,
        group_manifest_files=artifacts.group_manifest_files,
        group_families={key: str(profile) for key in artifacts.group_manifest_files},
        extra={
            "figures_root": str(figures_root),
            "available_groups": _available_groups(str(profile)),
            "skip_existing": bool(skip_existing),
            "segment_device": str(segment_device),
            "segment_cuda_device": int(segment_cuda_device) if segment_cuda_device is not None else None,
            "ctree_device": str(ctree_device),
            "ctree_cuda_device": int(ctree_cuda_device) if ctree_cuda_device is not None else None,
            "markov_device": str(markov_device),
            "markov_cuda_device": int(markov_cuda_device) if markov_cuda_device is not None else None,
            "markov_additive_device": str(markov_additive_device),
            "markov_additive_cuda_device": (
                int(markov_additive_cuda_device) if markov_additive_cuda_device is not None else None
            ),
            "markov_neural_device": str(markov_neural_device),
            "markov_neural_cuda_device": (
                int(markov_neural_cuda_device) if markov_neural_cuda_device is not None else None
            ),
            "torch_threads": int(torch_threads),
            "plot_cmds_file": str(paths.suite_plot_cmds),
            "counts_by_group": artifacts.counts_by_group,
            "counts": counts,
            "n_commands_total": int(sum(len(runs) for key, runs in groups.items() if key in selected_groups)),
        },
    )
    write_suite_meta(paths.suite_meta, meta)
    return meta


def _resolve_output_root(*, profile: str, run_id: str, output_root: str) -> Path:
    from treepo._research.ctreepo.sim.suite.common import resolve_output_root
    prefix = "identifiable_zero_publication_clean" if str(profile) == "publication_clean" else "identifiable_zero_longrun"
    return resolve_output_root(run_id=run_id, output_root=output_root, default_prefix=prefix)


def _resolve_figures_root(*, figures_root: str, output_root: Path) -> Path:
    from treepo._research.ctreepo.sim.suite.common import resolve_figures_root
    return resolve_figures_root(figures_root=figures_root, output_root=output_root)


def _ensure_built(ns: argparse.Namespace) -> Tuple[SuitePaths, Dict[str, object]]:
    run_id = utc_run_id(getattr(ns, "run_id", ""))
    profile = str(getattr(ns, "profile", "publication_clean"))
    output_root = _resolve_output_root(profile=profile, run_id=run_id, output_root=str(getattr(ns, "output_root", "")))
    figures_root = _resolve_figures_root(figures_root=str(getattr(ns, "figures_root", "")), output_root=output_root)
    paths = _resolve_paths(output_root=output_root.resolve(), figures_root=figures_root.resolve())
    rebuild = bool(getattr(ns, "rebuild", False))
    if rebuild or not paths.suite_meta.exists() or not paths.suite_manifest.exists() or not paths.suite_cmds.exists():
        meta = build_suite(
            run_id=run_id,
            profile=profile,
            python_bin=str(getattr(ns, "python_bin", "")).strip() or __import__("sys").executable,
            output_root=output_root,
            figures_root=figures_root,
            requested_groups=parse_items(str(getattr(ns, "groups", ""))),
            skip_existing=bool(getattr(ns, "skip_existing", True)),
            segment_device=str(getattr(ns, "segment_device", "auto")),
            segment_cuda_device=getattr(ns, "segment_cuda_device", None),
            ctree_device=str(getattr(ns, "ctree_device", "auto")),
            ctree_cuda_device=getattr(ns, "ctree_cuda_device", None),
            markov_device=str(getattr(ns, "markov_device", "cpu")),
            markov_cuda_device=getattr(ns, "markov_cuda_device", None),
            markov_additive_device=str(getattr(ns, "markov_additive_device", "cpu")),
            markov_additive_cuda_device=getattr(ns, "markov_additive_cuda_device", None),
            markov_neural_device=str(getattr(ns, "markov_neural_device", "auto")),
            markov_neural_cuda_device=getattr(ns, "markov_neural_cuda_device", None),
            torch_threads=int(getattr(ns, "torch_threads", 1)),
            n_seeds=int(getattr(ns, "n_seeds", 12)),
            markov_local_law_weight=float(getattr(ns, "markov_local_law_weight", 0.2)),
            markov_task_objective_weight=float(getattr(ns, "markov_task_objective_weight", 1.0)),
            pilot_cmd_count=int(getattr(ns, "pilot_cmd_count", 240)),
            target_main_jobs=int(getattr(ns, "target_main_jobs", 48)),
            target_pilot_minutes=int(getattr(ns, "target_pilot_minutes", 20)),
            segment_test_docs=int(getattr(ns, "segment_test_docs", 5000)),
            ctree_test_books=int(getattr(ns, "ctree_test_books", 5000)),
            markov_test_docs=int(getattr(ns, "markov_test_docs", 2000)),
            markov_n_epochs=int(getattr(ns, "markov_n_epochs", 12)),
        )
        return paths, meta
    return paths, read_suite_meta(paths.suite_meta)


def _add_build_args(parser: argparse.ArgumentParser, *, require_output_root: bool) -> None:
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--profile", choices=["publication_clean", "longrun_equiv_v1"], default="publication_clean")
    parser.add_argument("--python-bin", type=str, default="")
    parser.add_argument(
        "--output-root",
        type=str,
        default="" if not require_output_root else None,
        required=bool(require_output_root),
    )
    parser.add_argument("--figures-root", type=str, default="")
    parser.add_argument("--groups", type=str, default="")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segment-device", type=str, default="auto")
    parser.add_argument("--segment-cuda-device", type=int, default=None)
    parser.add_argument("--ctree-device", type=str, default="auto")
    parser.add_argument("--ctree-cuda-device", type=int, default=None)
    parser.add_argument("--markov-device", type=str, default="cpu")
    parser.add_argument("--markov-cuda-device", type=int, default=None)
    parser.add_argument("--markov-additive-device", type=str, default="cpu")
    parser.add_argument("--markov-additive-cuda-device", type=int, default=None)
    parser.add_argument("--markov-neural-device", type=str, default="auto")
    parser.add_argument("--markov-neural-cuda-device", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--segment-test-docs", type=int, default=5000)
    parser.add_argument("--ctree-test-books", type=int, default=5000)
    parser.add_argument("--markov-test-docs", type=int, default=2000)
    parser.add_argument("--markov-n-epochs", type=int, default=12)
    parser.add_argument("--n-seeds", type=int, default=12)
    parser.add_argument("--markov-local-law-weight", type=float, default=0.2)
    parser.add_argument("--markov-task-objective-weight", type=float, default=1.0)
    parser.add_argument("--pilot-cmd-count", type=int, default=240)
    parser.add_argument("--target-main-jobs", type=int, default=48)
    parser.add_argument("--target-pilot-minutes", type=int, default=20)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper-facing identifiable-zero publication/longrun suite.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Build grouped manifests and command files.")
    _add_build_args(build, require_output_root=False)

    run = sub.add_parser("run", help="Build if needed, then execute selected suite groups.")
    _add_build_args(run, require_output_root=False)
    run.add_argument("--rebuild", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--jobs", type=int, default=1)
    run.add_argument("--gpu-tokens", type=str, default="auto")
    run.add_argument("--log-dir", type=str, default="")

    plot = sub.add_parser("plot", help="Execute plot/report command list.")
    _add_build_args(plot, require_output_root=False)
    plot.add_argument("--rebuild", action=argparse.BooleanOptionalAction, default=False)
    plot.add_argument("--jobs", type=int, default=1)
    plot.add_argument("--log-dir", type=str, default="")
    plot.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)

    report = sub.add_parser("report", help="Generate the publication-clean report.")
    report.add_argument("--output-root", type=str, required=True)
    report.add_argument("--profile", choices=["publication_clean", "longrun_equiv_v1"], default="publication_clean")
    report.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    ns = _build_parser().parse_args(list(argv) if argv is not None else None)

    if ns.cmd == "build":
        run_id = utc_run_id(ns.run_id)
        profile = str(ns.profile)
        output_root = _resolve_output_root(profile=profile, run_id=run_id, output_root=str(ns.output_root or ""))
        figures_root = _resolve_figures_root(figures_root=str(ns.figures_root), output_root=output_root)
        meta = build_suite(
            run_id=run_id,
            profile=profile,
            python_bin=str(ns.python_bin).strip() or __import__("sys").executable,
            output_root=output_root,
            figures_root=figures_root,
            requested_groups=parse_items(str(ns.groups)),
            skip_existing=bool(ns.skip_existing),
            segment_device=str(ns.segment_device),
            segment_cuda_device=ns.segment_cuda_device,
            ctree_device=str(ns.ctree_device),
            ctree_cuda_device=ns.ctree_cuda_device,
            markov_device=str(ns.markov_device),
            markov_cuda_device=ns.markov_cuda_device,
            markov_additive_device=str(ns.markov_additive_device),
            markov_additive_cuda_device=ns.markov_additive_cuda_device,
            markov_neural_device=str(ns.markov_neural_device),
            markov_neural_cuda_device=ns.markov_neural_cuda_device,
            torch_threads=int(ns.torch_threads),
            n_seeds=int(ns.n_seeds),
            markov_local_law_weight=float(ns.markov_local_law_weight),
            markov_task_objective_weight=float(ns.markov_task_objective_weight),
            pilot_cmd_count=int(ns.pilot_cmd_count),
            target_main_jobs=int(ns.target_main_jobs),
            target_pilot_minutes=int(ns.target_pilot_minutes),
            segment_test_docs=int(ns.segment_test_docs),
            ctree_test_books=int(ns.ctree_test_books),
            markov_test_docs=int(ns.markov_test_docs),
            markov_n_epochs=int(ns.markov_n_epochs),
        )
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0

    if ns.cmd == "run":
        paths, meta = _ensure_built(ns)
        requested_groups = parse_items(str(ns.groups))
        if requested_groups:
            selected_groups = select_known_items(
                requested=requested_groups,
                available=list((meta.get("group_manifest_files", {}) or {}).keys()),
                item_name="publication suite groups",
            )
        else:
            selected_groups = [str(x) for x in (meta.get("selected_groups", []) or []) if str(x).strip()]
        manifest_files = dict(meta.get("group_manifest_files", {}) or {})
        manifest_paths = [Path(str(manifest_files[key])) for key in selected_groups]
        if not manifest_paths:
            raise SystemExit("no manifest paths selected")
        log_dir = Path(ns.log_dir).resolve() if str(ns.log_dir).strip() else paths.queue_log_dir
        payload = run_manifest_queue_suite(
            manifest_paths=manifest_paths,
            cpu_workers=int(ns.jobs),
            gpu_tokens=str(ns.gpu_tokens),
            log_dir=log_dir,
            set_thread_env=True,
        )
        print(
            json.dumps(
                {
                    "output_root": str(paths.output_root),
                    "profile": str(meta.get("profile", "")),
                    "selected_groups": list(selected_groups),
                    **payload,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if int(payload["summary"].get("n_fail", 0)) == 0 else 1

    if ns.cmd == "plot":
        paths, _meta = _ensure_built(ns)
        from treepo._research.ctreepo.sim.cli.exec_cmds import main as _exec_main  # noqa: WPS433

        exec_argv: List[str] = ["--cmds", str(paths.suite_plot_cmds), "--jobs", str(int(ns.jobs))]
        if str(ns.log_dir).strip():
            exec_argv.extend(["--log-dir", str(ns.log_dir)])
        if bool(ns.fail_fast):
            exec_argv.append("--fail-fast")
        return int(_exec_main(exec_argv))

    if ns.cmd == "report":
        if str(ns.profile) != "publication_clean":
            raise SystemExit("suite report is only defined for --profile publication_clean")
        from treepo._research.ctreepo.sim.cli.report.identifiable_zero_publication_clean import (  # noqa: WPS433
            main as _report_main,
        )

        return int(
            _report_main(
                [
                    "--output-root",
                    str(Path(ns.output_root).resolve()),
                    "--emit-pdf" if bool(ns.emit_pdf) else "--no-emit-pdf",
                ]
            )
        )

    raise ValueError("unreachable")


__all__ = ["build_suite", "main"]
