from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence

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
from treepo._research.ctreepo.sim.suite.learnability_policy import (
    IdentifiableZeroLearnabilityPolicy,
    resolve_identifiable_zero_learnability_policy,
)


def _hero_train_docs(policy: IdentifiableZeroLearnabilityPolicy) -> List[int]:
    return [int(max(policy.train_docs_grid))]


def _hero_label_rates(policy: IdentifiableZeroLearnabilityPolicy) -> List[float]:
    rates = [float(x) for x in policy.label_rate_grid if float(x) >= 0.05]
    return rates or [float(x) for x in policy.label_rate_grid]


def _hero_guidance_rates(policy: IdentifiableZeroLearnabilityPolicy) -> List[float]:
    rates = [float(x) for x in policy.ctree_eval_guidance_rates if abs(float(x)) <= 1e-12]
    if rates:
        return rates
    return [float(policy.ctree_eval_guidance_rates[0])]


_GROUP_ORDER: tuple[str, ...] = (
    "markov_baseline",
    "markov_hard",
    "markov_hard_hero",
    "ctree_baseline_lstsq",
    "ctree_baseline_theta",
    "ctree_hard_lstsq",
    "ctree_hard_theta",
    "ctree_hard_hero_lstsq",
    "ctree_hard_hero_theta",
    "ctree_lda_lstsq",
    "ctree_lda_theta",
)
_HERO_GROUPS = {"markov_hard_hero", "ctree_hard_hero_lstsq", "ctree_hard_hero_theta"}


def _available_groups(*, hero: bool) -> List[str]:
    return [key for key in _GROUP_ORDER if hero or key not in _HERO_GROUPS]


def _selected_groups(*, requested: Sequence[str], hero: bool) -> List[str]:
    return select_known_items(
        requested=requested,
        available=_available_groups(hero=hero),
        item_name="learnability groups",
    )


def _markov_common_runs(
    *,
    key: str,
    python_bin: str,
    output_root: Path,
    train_docs: Iterable[int],
    audit_fractions: Iterable[float],
    seeds: Iterable[int],
    n_regimes: int,
    vocab_size: int,
    min_tokens: int,
    max_tokens: int,
    min_segments: int,
    max_segments: int,
    test_docs: int,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
    skip_existing: bool,
    sampled_leaf_pool_leaf_counts: Iterable[int],
) -> SuiteGroupRuns:
    sampled_leaf_pool_counts = [int(x) for x in sampled_leaf_pool_leaf_counts]
    runs = _iter_markov_runs(
        python_bin=python_bin,
        n_regimes=int(n_regimes),
        vocab_size=int(vocab_size),
        min_tokens=int(min_tokens),
        max_tokens=int(max_tokens),
        min_segments=int(min_segments),
        max_segments=int(max_segments),
        fixed_leaf_tokens=16,
        train_docs=[int(x) for x in train_docs],
        val_docs=0,
        test_docs=int(test_docs),
        audit_fractions=[float(x) for x in audit_fractions],
        c3_audit_strategies=["uniform"],
        c3_include_root=True,
        leaf_query_rates=[1.0],
        include_root_queries=[True],
        local_law_weights=[0.0],
        task_objective_weights=[],
        c1_relative_weights=[1.0],
        c2_relative_weights=[1.0],
        c3_relative_weights=[1.0],
        root_weights=[1.0],
        schedule_consistency_weights=[0.0],
        guidance_override_modes=["reset"],
        eval_guidance_qs=[],
        eval_guidance_trials=0,
        eval_guidance_seed_offset=100_000,
        eval_guidance_include_root=True,
        include_rf_root_baseline=True,
        include_doc_level_baseline=True,
        include_doc_level_ridge_baseline=True,
        include_leaf_ridge_tree_baseline=True,
        include_leaf_endpoint_table_tree_baseline=True,
        include_leaf_dt_tree_baseline=True,
        include_leaf_knn_tree_baseline=True,
        include_leaf_rf_tree_baseline=True,
        rf_n_estimators=200,
        rf_max_depth=16,
        rf_min_samples_leaf=5,
        doc_level_ridge_alpha=1.0,
        leaf_knn_neighbors=32,
        include_sampled_leaf_pool_ridge_baseline=bool(sampled_leaf_pool_counts),
        include_sampled_leaf_pool_rf_baseline=bool(sampled_leaf_pool_counts),
        sampled_leaf_pool_leaf_counts=sampled_leaf_pool_counts,
        sampled_leaf_pool_seed_offset=200_000,
        data_seeds=[],
        seeds=[int(x) for x in seeds],
        output_root=output_root,
        model_families=["neural", "additive"],
        feature_modes=["full"],
        state_dims=[32],
        hidden_dims=[128],
        hidden_dim_multiplier=None,
        hidden_dim_min=64,
        n_epochs=10,
        device=str(device),
        cuda_device=cuda_device,
        violation_tau=0.0,
        torch_threads=int(torch_threads),
        skip_existing=bool(skip_existing),
        suite_role=str(key),
        law_packages=[],
        exact_families=[],
        c2_weights=[0.0],
    )
    return SuiteGroupRuns(key=str(key), family="markov-ops-count", runs=runs)


def _ctree_common_runs(
    *,
    key: str,
    python_bin: str,
    output_root: Path,
    train_docs: Iterable[int],
    calibration_rates: Iterable[float],
    eval_rates: Iterable[float],
    seeds: Iterable[int],
    topic_phi_estimators: Iterable[str],
    leaf_theta_estimators: Iterable[str],
    topic_processes: Iterable[str],
    n_topics: int,
    vocab_size: int,
    min_segments: int,
    max_segments: int,
    min_seg_tokens: int,
    max_seg_tokens: int,
    fixed_leaf_tokens: int,
    alpha_topic: float,
    beta_word: float,
    segment_concentration: float,
    segment_background: float,
    n_books_test: int,
    include_full_doc_theta_baseline: bool,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
    skip_existing: bool,
) -> SuiteGroupRuns:
    runs = _iter_ctree_runs(
        python_bin=python_bin,
        train_docs=[int(x) for x in train_docs],
        seeds=[int(x) for x in seeds],
        calibration_rates=[float(x) for x in calibration_rates],
        eval_internal_rates=[float(x) for x in eval_rates],
        eval_leaf_rates=[float(x) for x in eval_rates],
        output_root=output_root,
        topic_phi_estimators=[str(x) for x in topic_phi_estimators],
        topic_phi_docs_values=[0],
        leaf_theta_estimators=[str(x) for x in leaf_theta_estimators],
        topic_processes=[str(x) for x in topic_processes],
        n_topics=int(n_topics),
        vocab_size=int(vocab_size),
        min_segments=int(min_segments),
        max_segments=int(max_segments),
        min_seg_tokens=int(min_seg_tokens),
        max_seg_tokens=int(max_seg_tokens),
        fixed_leaf_tokens=int(fixed_leaf_tokens),
        n_books_test=int(n_books_test),
        alpha_topic=float(alpha_topic),
        beta_word=float(beta_word),
        segment_concentration=float(segment_concentration),
        segment_background=float(segment_background),
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
        include_full_doc_theta_baseline=bool(include_full_doc_theta_baseline),
        device=str(device),
        cuda_device=cuda_device,
        torch_threads=int(torch_threads),
        skip_existing=bool(skip_existing),
    )
    return SuiteGroupRuns(key=str(key), family="segmented-lda-ctreepo", runs=runs)


def _build_groups(
    *,
    selected_groups: Sequence[str],
    policy: IdentifiableZeroLearnabilityPolicy,
    python_bin: str,
    output_root: Path,
    markov_device: str,
    markov_cuda_device: int | None,
    ctree_device: str,
    ctree_cuda_device: int | None,
    torch_threads: int,
    skip_existing: bool,
) -> List[SuiteGroupRuns]:
    groups: List[SuiteGroupRuns] = []
    selected = set(selected_groups)

    markov_base_out = output_root / "markov_changepoint_ops_count" / "equivalence" / "baseline"
    markov_hard_out = output_root / "markov_changepoint_ops_count" / "equivalence" / "hard"
    ctree_base_out = output_root / "segmented_lda_ctreepo" / "equivalence" / "baseline"
    ctree_hard_out = output_root / "segmented_lda_ctreepo" / "equivalence" / "hard"
    ctree_lda_out = output_root / "segmented_lda_ctreepo" / "equivalence" / "lda"

    if "markov_baseline" in selected:
        groups.append(
            _markov_common_runs(
                key="markov_baseline",
                python_bin=python_bin,
                output_root=markov_base_out,
                train_docs=policy.train_docs_grid,
                audit_fractions=policy.label_rate_grid,
                seeds=policy.base_seeds,
                n_regimes=4,
                vocab_size=96,
                min_tokens=384,
                max_tokens=384,
                min_segments=12,
                max_segments=24,
                test_docs=policy.heldout_docs,
                device=markov_device,
                cuda_device=markov_cuda_device,
                torch_threads=torch_threads,
                skip_existing=skip_existing,
                sampled_leaf_pool_leaf_counts=policy.markov_sampled_leaf_pool_leaf_counts,
            )
        )
    if "markov_hard" in selected:
        groups.append(
            _markov_common_runs(
                key="markov_hard",
                python_bin=python_bin,
                output_root=markov_hard_out,
                train_docs=policy.train_docs_grid,
                audit_fractions=policy.label_rate_grid,
                seeds=policy.base_seeds,
                n_regimes=6,
                vocab_size=128,
                min_tokens=768,
                max_tokens=768,
                min_segments=24,
                max_segments=48,
                test_docs=policy.heldout_docs,
                device=markov_device,
                cuda_device=markov_cuda_device,
                torch_threads=torch_threads,
                skip_existing=skip_existing,
                sampled_leaf_pool_leaf_counts=policy.markov_sampled_leaf_pool_leaf_counts,
            )
        )
    if "markov_hard_hero" in selected:
        groups.append(
            _markov_common_runs(
                key="markov_hard_hero",
                python_bin=python_bin,
                output_root=markov_hard_out,
                train_docs=_hero_train_docs(policy),
                audit_fractions=_hero_label_rates(policy),
                seeds=policy.hero_seeds,
                n_regimes=6,
                vocab_size=128,
                min_tokens=768,
                max_tokens=768,
                min_segments=24,
                max_segments=48,
                test_docs=policy.heldout_docs,
                device=markov_device,
                cuda_device=markov_cuda_device,
                torch_threads=torch_threads,
                skip_existing=skip_existing,
                sampled_leaf_pool_leaf_counts=policy.markov_sampled_leaf_pool_leaf_counts,
            )
        )

    if "ctree_baseline_lstsq" in selected:
        groups.append(
            _ctree_common_runs(
                key="ctree_baseline_lstsq",
                python_bin=python_bin,
                output_root=ctree_base_out,
                train_docs=policy.train_docs_grid,
                calibration_rates=policy.label_rate_grid,
                eval_rates=policy.ctree_eval_guidance_rates,
                seeds=policy.base_seeds,
                topic_phi_estimators=["spectral_numpy", "embedding_spectral"],
                leaf_theta_estimators=["lstsq"],
                topic_processes=["segments"],
                n_topics=4,
                vocab_size=256,
                min_segments=6,
                max_segments=6,
                min_seg_tokens=24,
                max_seg_tokens=48,
                fixed_leaf_tokens=32,
                alpha_topic=0.20,
                beta_word=0.10,
                segment_concentration=80.0,
                segment_background=2.0,
                n_books_test=policy.heldout_docs,
                include_full_doc_theta_baseline=False,
                device=ctree_device,
                cuda_device=ctree_cuda_device,
                torch_threads=torch_threads,
                skip_existing=skip_existing,
            )
        )
    if "ctree_baseline_theta" in selected:
        groups.append(
            _ctree_common_runs(
                key="ctree_baseline_theta",
                python_bin=python_bin,
                output_root=ctree_base_out,
                train_docs=policy.train_docs_grid,
                calibration_rates=policy.label_rate_grid,
                eval_rates=policy.ctree_eval_guidance_rates,
                seeds=policy.base_seeds,
                topic_phi_estimators=["spectral_numpy"],
                leaf_theta_estimators=["rf", "mlp"],
                topic_processes=["segments"],
                n_topics=4,
                vocab_size=256,
                min_segments=6,
                max_segments=6,
                min_seg_tokens=24,
                max_seg_tokens=48,
                fixed_leaf_tokens=32,
                alpha_topic=0.20,
                beta_word=0.10,
                segment_concentration=80.0,
                segment_background=2.0,
                n_books_test=policy.heldout_docs,
                include_full_doc_theta_baseline=True,
                device=ctree_device,
                cuda_device=ctree_cuda_device,
                torch_threads=torch_threads,
                skip_existing=skip_existing,
            )
        )
    if "ctree_hard_lstsq" in selected:
        groups.append(
            _ctree_common_runs(
                key="ctree_hard_lstsq",
                python_bin=python_bin,
                output_root=ctree_hard_out,
                train_docs=policy.train_docs_grid,
                calibration_rates=policy.label_rate_grid,
                eval_rates=policy.ctree_eval_guidance_rates,
                seeds=policy.base_seeds,
                topic_phi_estimators=["spectral_numpy", "embedding_spectral"],
                leaf_theta_estimators=["lstsq"],
                topic_processes=["segments"],
                n_topics=8,
                vocab_size=512,
                min_segments=8,
                max_segments=8,
                min_seg_tokens=16,
                max_seg_tokens=32,
                fixed_leaf_tokens=16,
                alpha_topic=0.30,
                beta_word=0.30,
                segment_concentration=20.0,
                segment_background=5.0,
                n_books_test=policy.heldout_docs,
                include_full_doc_theta_baseline=False,
                device=ctree_device,
                cuda_device=ctree_cuda_device,
                torch_threads=torch_threads,
                skip_existing=skip_existing,
            )
        )
    if "ctree_hard_theta" in selected:
        groups.append(
            _ctree_common_runs(
                key="ctree_hard_theta",
                python_bin=python_bin,
                output_root=ctree_hard_out,
                train_docs=policy.train_docs_grid,
                calibration_rates=policy.label_rate_grid,
                eval_rates=policy.ctree_eval_guidance_rates,
                seeds=policy.base_seeds,
                topic_phi_estimators=["spectral_numpy"],
                leaf_theta_estimators=["rf", "mlp"],
                topic_processes=["segments"],
                n_topics=8,
                vocab_size=512,
                min_segments=8,
                max_segments=8,
                min_seg_tokens=16,
                max_seg_tokens=32,
                fixed_leaf_tokens=16,
                alpha_topic=0.30,
                beta_word=0.30,
                segment_concentration=20.0,
                segment_background=5.0,
                n_books_test=policy.heldout_docs,
                include_full_doc_theta_baseline=True,
                device=ctree_device,
                cuda_device=ctree_cuda_device,
                torch_threads=torch_threads,
                skip_existing=skip_existing,
            )
        )
    if "ctree_hard_hero_lstsq" in selected:
        groups.append(
            _ctree_common_runs(
                key="ctree_hard_hero_lstsq",
                python_bin=python_bin,
                output_root=ctree_hard_out,
                train_docs=_hero_train_docs(policy),
                calibration_rates=_hero_label_rates(policy),
                eval_rates=_hero_guidance_rates(policy),
                seeds=policy.hero_seeds,
                topic_phi_estimators=["spectral_numpy"],
                leaf_theta_estimators=["lstsq"],
                topic_processes=["segments"],
                n_topics=8,
                vocab_size=512,
                min_segments=8,
                max_segments=8,
                min_seg_tokens=16,
                max_seg_tokens=32,
                fixed_leaf_tokens=16,
                alpha_topic=0.30,
                beta_word=0.30,
                segment_concentration=20.0,
                segment_background=5.0,
                n_books_test=policy.heldout_docs,
                include_full_doc_theta_baseline=False,
                device=ctree_device,
                cuda_device=ctree_cuda_device,
                torch_threads=torch_threads,
                skip_existing=skip_existing,
            )
        )
    if "ctree_hard_hero_theta" in selected:
        groups.append(
            _ctree_common_runs(
                key="ctree_hard_hero_theta",
                python_bin=python_bin,
                output_root=ctree_hard_out,
                train_docs=_hero_train_docs(policy),
                calibration_rates=_hero_label_rates(policy),
                eval_rates=_hero_guidance_rates(policy),
                seeds=policy.hero_seeds,
                topic_phi_estimators=["spectral_numpy"],
                leaf_theta_estimators=["rf", "mlp"],
                topic_processes=["segments"],
                n_topics=8,
                vocab_size=512,
                min_segments=8,
                max_segments=8,
                min_seg_tokens=16,
                max_seg_tokens=32,
                fixed_leaf_tokens=16,
                alpha_topic=0.30,
                beta_word=0.30,
                segment_concentration=20.0,
                segment_background=5.0,
                n_books_test=policy.heldout_docs,
                include_full_doc_theta_baseline=True,
                device=ctree_device,
                cuda_device=ctree_cuda_device,
                torch_threads=torch_threads,
                skip_existing=skip_existing,
            )
        )
    if "ctree_lda_lstsq" in selected:
        groups.append(
            _ctree_common_runs(
                key="ctree_lda_lstsq",
                python_bin=python_bin,
                output_root=ctree_lda_out,
                train_docs=policy.train_docs_grid,
                calibration_rates=policy.label_rate_grid,
                eval_rates=policy.ctree_eval_guidance_rates,
                seeds=policy.base_seeds,
                topic_phi_estimators=["spectral_numpy", "embedding_spectral", "tensor_lda", "sklearn_lda"],
                leaf_theta_estimators=["lstsq"],
                topic_processes=["bag_of_words"],
                n_topics=4,
                vocab_size=256,
                min_segments=6,
                max_segments=6,
                min_seg_tokens=24,
                max_seg_tokens=48,
                fixed_leaf_tokens=32,
                alpha_topic=0.20,
                beta_word=0.10,
                segment_concentration=80.0,
                segment_background=2.0,
                n_books_test=policy.heldout_docs,
                include_full_doc_theta_baseline=False,
                device=ctree_device,
                cuda_device=ctree_cuda_device,
                torch_threads=torch_threads,
                skip_existing=skip_existing,
            )
        )
    if "ctree_lda_theta" in selected:
        groups.append(
            _ctree_common_runs(
                key="ctree_lda_theta",
                python_bin=python_bin,
                output_root=ctree_lda_out,
                train_docs=policy.train_docs_grid,
                calibration_rates=policy.label_rate_grid,
                eval_rates=policy.ctree_eval_guidance_rates,
                seeds=policy.base_seeds,
                topic_phi_estimators=["spectral_numpy"],
                leaf_theta_estimators=["rf", "mlp"],
                topic_processes=["bag_of_words"],
                n_topics=4,
                vocab_size=256,
                min_segments=6,
                max_segments=6,
                min_seg_tokens=24,
                max_seg_tokens=48,
                fixed_leaf_tokens=32,
                alpha_topic=0.20,
                beta_word=0.10,
                segment_concentration=80.0,
                segment_background=2.0,
                n_books_test=policy.heldout_docs,
                include_full_doc_theta_baseline=True,
                device=ctree_device,
                cuda_device=ctree_cuda_device,
                torch_threads=torch_threads,
                skip_existing=skip_existing,
            )
        )

    return groups


def build_suite(
    *,
    run_id: str,
    python_bin: str,
    output_root: Path,
    hero: bool,
    requested_groups: Sequence[str],
    policy: IdentifiableZeroLearnabilityPolicy,
    markov_device: str,
    markov_cuda_device: int | None,
    ctree_device: str,
    ctree_cuda_device: int | None,
    torch_threads: int,
    skip_existing: bool,
) -> Dict[str, object]:
    output_root = output_root.resolve()
    paths = resolve_grouped_suite_paths(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    selected_groups = _selected_groups(requested=requested_groups, hero=bool(hero))
    group_builds = _build_groups(
        selected_groups=selected_groups,
        policy=policy,
        python_bin=python_bin,
        output_root=output_root,
        markov_device=markov_device,
        markov_cuda_device=markov_cuda_device,
        ctree_device=ctree_device,
        ctree_cuda_device=ctree_cuda_device,
        torch_threads=torch_threads,
        skip_existing=skip_existing,
    )

    artifacts = emit_grouped_suite_artifacts(paths, group_builds)

    meta: Dict[str, object] = build_suite_meta(
        suite_name="identifiable-zero-learnability",
        suite_role="appendix",
        run_id=str(run_id),
        profile=str(policy.profile),
        policy=policy.to_dict(),
        python_bin=str(python_bin),
        output_root=output_root,
        cmds_file=paths.suite_cmds,
        manifest_file=paths.suite_manifest,
        selected_groups=list(selected_groups),
        group_cmd_files=artifacts.group_cmd_files,
        group_manifest_files=artifacts.group_manifest_files,
        group_families=artifacts.group_families,
        extra={
            "hero": bool(hero),
            "markov_device": str(markov_device),
            "markov_cuda_device": markov_cuda_device,
            "ctree_device": str(ctree_device),
            "ctree_cuda_device": ctree_cuda_device,
            "torch_threads": int(torch_threads),
            "skip_existing": bool(skip_existing),
            "counts_by_group": artifacts.counts_by_group,
            "n_commands_total": int(len(artifacts.all_cmds)),
        },
    )
    write_suite_meta(paths.suite_meta, meta)
    return meta


def _read_suite_meta(output_root: Path) -> Dict[str, object]:
    paths = resolve_grouped_suite_paths(output_root.resolve())
    if not paths.suite_meta.exists():
        raise FileNotFoundError(f"suite meta not found: {paths.suite_meta}")
    return read_suite_meta(paths.suite_meta)


def _add_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=["paper", "smoke"], default="paper")
    parser.add_argument("--train-docs-grid", type=str, default="")
    parser.add_argument("--label-rate-grid", type=str, default="")
    parser.add_argument("--heldout-docs", type=int, default=0)
    parser.add_argument("--base-seeds", type=str, default="")
    parser.add_argument("--hero-seeds", type=str, default="")
    parser.add_argument("--ctree-eval-guidance-rates", type=str, default="")
    parser.add_argument("--markov-sampled-leaf-pool-leaf-counts", type=str, default="")


def _resolve_policy_from_args(args: argparse.Namespace) -> IdentifiableZeroLearnabilityPolicy:
    return resolve_identifiable_zero_learnability_policy(
        profile_name=str(getattr(args, "profile", "paper")),
        train_docs_grid=(str(args.train_docs_grid) if str(args.train_docs_grid).strip() else None),
        label_rate_grid=(str(args.label_rate_grid) if str(args.label_rate_grid).strip() else None),
        heldout_docs=(int(args.heldout_docs) if int(args.heldout_docs) > 0 else None),
        base_seeds=(str(args.base_seeds) if str(args.base_seeds).strip() else None),
        hero_seeds=(str(args.hero_seeds) if str(args.hero_seeds).strip() else None),
        ctree_eval_guidance_rates=(
            str(args.ctree_eval_guidance_rates) if str(args.ctree_eval_guidance_rates).strip() else None
        ),
        markov_sampled_leaf_pool_leaf_counts=(
            str(args.markov_sampled_leaf_pool_leaf_counts)
            if str(args.markov_sampled_leaf_pool_leaf_counts).strip()
            else None
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Identifiable-Zero learnability suite orchestration.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Build manifests and command files for the learnability suite.")
    build.add_argument("--run-id", type=str, default="")
    build.add_argument("--python-bin", type=str, default="")
    build.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Default: outputs/identifiable_zero_learnability_v1_<run_id>",
    )
    build.add_argument("--groups", type=str, default="")
    build.add_argument("--hero", action=argparse.BooleanOptionalAction, default=None)
    build.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    build.add_argument("--markov-device", type=str, default="auto")
    build.add_argument("--markov-cuda-device", type=int, default=None)
    build.add_argument("--ctree-device", type=str, default="auto")
    build.add_argument("--ctree-cuda-device", type=int, default=None)
    build.add_argument("--torch-threads", type=int, default=1)
    _add_policy_args(build)

    run = sub.add_parser("run", help="Build if needed, then execute the learnability suite.")
    run.add_argument("--run-id", type=str, default="")
    run.add_argument("--python-bin", type=str, default="")
    run.add_argument("--output-root", type=str, required=True)
    run.add_argument("--groups", type=str, default="")
    run.add_argument("--hero", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--markov-device", type=str, default="auto")
    run.add_argument("--markov-cuda-device", type=int, default=None)
    run.add_argument("--ctree-device", type=str, default="auto")
    run.add_argument("--ctree-cuda-device", type=int, default=None)
    run.add_argument("--torch-threads", type=int, default=1)
    run.add_argument("--jobs", type=int, default=1)
    run.add_argument("--gpu-tokens", type=str, default="auto")
    run.add_argument("--log-dir", type=str, default="")
    run.add_argument("--rebuild", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--set-thread-env", action=argparse.BooleanOptionalAction, default=True)
    _add_policy_args(run)

    report = sub.add_parser("report", help="Generate the canonical learnability report.")
    report.add_argument("--output-root", type=str, required=True)
    report.add_argument("--output-markdown", type=str, default="")
    report.add_argument("--output-pdf", type=str, default="")
    report.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    if args.cmd == "build":
        run_id = utc_run_id(args.run_id)
        python_bin = str(args.python_bin).strip() or sys.executable
        policy = _resolve_policy_from_args(args)
        hero = bool(args.hero) if args.hero is not None else bool(policy.profile != "smoke")
        output_root = (
            Path(args.output_root)
            if str(args.output_root).strip()
            else Path(f"outputs/identifiable_zero_learnability_v1_{run_id}")
        )
        meta = build_suite(
            run_id=run_id,
            python_bin=python_bin,
            output_root=output_root,
            hero=bool(hero),
            requested_groups=parse_items(args.groups),
            policy=policy,
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
        policy = _resolve_policy_from_args(args)
        hero = bool(args.hero) if args.hero is not None else bool(policy.profile != "smoke")
        if bool(args.rebuild) or not paths.suite_meta.exists() or not paths.suite_manifest.exists():
            run_id = utc_run_id(args.run_id or output_root.name)
            build_suite(
                run_id=run_id,
                python_bin=(str(args.python_bin).strip() or sys.executable),
                output_root=output_root,
                hero=bool(hero),
                requested_groups=parse_items(args.groups),
                policy=policy,
                markov_device=str(args.markov_device),
                markov_cuda_device=(int(args.markov_cuda_device) if args.markov_cuda_device is not None else None),
                ctree_device=str(args.ctree_device),
                ctree_cuda_device=(int(args.ctree_cuda_device) if args.ctree_cuda_device is not None else None),
                torch_threads=int(args.torch_threads),
                skip_existing=bool(args.skip_existing),
            )

        meta = _read_suite_meta(output_root)
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
        from treepo._research.ctreepo.sim.cli.report.identifiable_zero_learnability import (  # noqa: WPS433
            main as _report_main,
        )

        report_argv: List[str] = ["--output-root", str(Path(args.output_root).resolve())]
        if str(args.output_markdown).strip():
            report_argv.extend(["--output-markdown", str(Path(args.output_markdown).resolve())])
        if str(args.output_pdf).strip():
            report_argv.extend(["--output-pdf", str(Path(args.output_pdf).resolve())])
        report_argv.append("--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf")
        return int(_report_main(report_argv))

    raise ValueError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
