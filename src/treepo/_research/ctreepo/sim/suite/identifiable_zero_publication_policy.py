from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class SegmentOpsSweepPolicy:
    train_docs: Tuple[int, ...]
    test_docs: int
    audit_fractions: Tuple[float, ...]
    topic_phi_docs: Tuple[int, ...]
    topic_phi_estimators: Tuple[str, ...]
    topic_processes: Tuple[str, ...]
    lambda_multipliers: Tuple[float, ...]
    seeds: Tuple[int, ...]
    topic_source: str
    feature_inference: str
    n_topics: int
    vocab_size: int
    min_tokens: int
    max_tokens: int
    leaf_tokens: int
    run_all_feature_modes: bool


@dataclass(frozen=True)
class CtreeSweepPolicy:
    train_docs: Tuple[int, ...]
    seeds: Tuple[int, ...]
    calibration_rates: Tuple[float, ...]
    eval_leaf_rates: Tuple[float, ...]
    eval_internal_rates: Tuple[float, ...]
    topic_phi_estimators: Tuple[str, ...]
    topic_phi_docs_values: Tuple[int, ...]
    leaf_theta_estimators: Tuple[str, ...]
    topic_processes: Tuple[str, ...]
    n_topics: int
    vocab_size: int
    min_segments: int
    max_segments: int
    min_seg_tokens: int
    max_seg_tokens: int
    fixed_leaf_tokens: int
    n_books_test: int
    alpha_topic: float
    beta_word: float
    segment_concentration: float
    segment_background: float
    calibration_policy: str
    eval_internal_query_design: str
    topic_phi_permute: bool
    include_full_doc_theta_baseline: bool


@dataclass(frozen=True)
class MarkovSweepPolicy:
    n_regimes: int
    vocab_size: int
    min_tokens: int
    max_tokens: int
    min_segments: int
    max_segments: int
    fixed_leaf_tokens: int
    train_docs: Tuple[int, ...]
    val_docs: int
    test_docs: int
    audit_fractions: Tuple[float, ...]
    c3_audit_strategies: Tuple[str, ...]
    c3_include_root: bool
    leaf_query_rates: Tuple[float, ...]
    include_root_queries: Tuple[bool, ...]
    local_law_weights: Tuple[float, ...]
    task_objective_weights: Tuple[float, ...]
    c1_relative_weights: Tuple[float, ...]
    c2_relative_weights: Tuple[float, ...]
    c3_relative_weights: Tuple[float, ...]
    c2_weights: Tuple[float, ...]
    root_weights: Tuple[float, ...]
    schedule_consistency_weights: Tuple[float, ...]
    guidance_override_modes: Tuple[str, ...]
    eval_guidance_qs: Tuple[float, ...]
    eval_guidance_trials: int
    eval_guidance_seed_offset: int
    eval_guidance_include_root: bool
    include_rf_root_baseline: bool
    include_doc_level_baseline: bool
    rf_n_estimators: int
    rf_max_depth: int
    rf_min_samples_leaf: int
    data_seeds: Tuple[int, ...]
    seeds: Tuple[int, ...]
    model_families: Tuple[str, ...]
    feature_modes: Tuple[str, ...]
    state_dims: Tuple[int, ...]
    hidden_dims: Tuple[int, ...]
    hidden_dim_multiplier: float | None
    hidden_dim_min: int
    n_epochs: int
    violation_tau: float
    suite_role: str


@dataclass(frozen=True)
class IdentifiableZeroPublicationCleanPolicy:
    profile: str
    segment: SegmentOpsSweepPolicy
    ctree: CtreeSweepPolicy
    markov: MarkovSweepPolicy

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class IdentifiableZeroLongrunPolicy:
    profile: str
    segment_scale: SegmentOpsSweepPolicy
    ctree_scale: CtreeSweepPolicy
    ctree_equiv: CtreeSweepPolicy
    markov_scale: MarkovSweepPolicy
    markov_equiv: MarkovSweepPolicy
    pilot_cmd_count: int
    target_main_jobs: int
    target_pilot_minutes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_identifiable_zero_publication_clean_policy(
    *,
    n_seeds: int = 12,
    markov_local_law_weight: float = 0.2,
    markov_task_objective_weight: float = 1.0,
) -> IdentifiableZeroPublicationCleanPolicy:
    seeds = tuple(range(max(0, int(n_seeds))))
    return IdentifiableZeroPublicationCleanPolicy(
        profile="publication_clean",
        segment=SegmentOpsSweepPolicy(
            train_docs=(12000,),
            test_docs=5000,
            audit_fractions=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
            topic_phi_docs=(0,),
            topic_phi_estimators=("true", "embedding_spectral"),
            topic_processes=("segments",),
            lambda_multipliers=(1.0,),
            seeds=seeds,
            topic_source="infer",
            feature_inference="hard",
            n_topics=8,
            vocab_size=512,
            min_tokens=384,
            max_tokens=384,
            leaf_tokens=16,
            run_all_feature_modes=True,
        ),
        ctree=CtreeSweepPolicy(
            train_docs=(4096,),
            seeds=seeds,
            calibration_rates=(0.01, 0.02, 0.05, 0.1),
            eval_leaf_rates=(0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
            eval_internal_rates=(0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
            topic_phi_estimators=("spectral_numpy",),
            topic_phi_docs_values=(0,),
            leaf_theta_estimators=("lstsq",),
            topic_processes=("segments",),
            n_topics=8,
            vocab_size=512,
            min_segments=8,
            max_segments=16,
            min_seg_tokens=32,
            max_seg_tokens=96,
            fixed_leaf_tokens=32,
            n_books_test=5000,
            alpha_topic=0.4,
            beta_word=0.2,
            segment_concentration=12.0,
            segment_background=4.0,
            calibration_policy="uniform",
            eval_internal_query_design="risk",
            topic_phi_permute=True,
            include_full_doc_theta_baseline=False,
        ),
        markov=MarkovSweepPolicy(
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
            audit_fractions=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
            c3_audit_strategies=("uniform",),
            c3_include_root=True,
            leaf_query_rates=(1.0,),
            include_root_queries=(True,),
            local_law_weights=(float(markov_local_law_weight),),
            task_objective_weights=(float(markov_task_objective_weight),),
            c1_relative_weights=(0.0,),
            c2_relative_weights=(0.0,),
            c3_relative_weights=(1.0,),
            c2_weights=(0.0,),
            root_weights=(1.0,),
            schedule_consistency_weights=(0.0,),
            guidance_override_modes=("reset",),
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
            seeds=seeds,
            model_families=("additive", "neural"),
            feature_modes=("full",),
            state_dims=(32,),
            hidden_dims=(128,),
            hidden_dim_multiplier=None,
            hidden_dim_min=64,
            n_epochs=12,
            violation_tau=0.0,
            suite_role="publication_clean",
        ),
    )


def resolve_identifiable_zero_longrun_policy(
    *,
    n_seeds: int = 12,
    pilot_cmd_count: int = 240,
    target_main_jobs: int = 48,
    target_pilot_minutes: int = 20,
) -> IdentifiableZeroLongrunPolicy:
    seeds = tuple(range(max(0, int(n_seeds))))
    common_markov = dict(
        n_regimes=4,
        vocab_size=96,
        min_tokens=384,
        max_tokens=384,
        min_segments=12,
        max_segments=24,
        fixed_leaf_tokens=16,
        val_docs=0,
        test_docs=2000,
        audit_fractions=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
        c3_audit_strategies=("uniform",),
        c3_include_root=True,
        include_root_queries=(True, False),
        local_law_weights=(0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0),
        task_objective_weights=(),
        c1_relative_weights=(1.0,),
        c2_relative_weights=(1.0,),
        c3_relative_weights=(1.0,),
        c2_weights=(0.0,),
        root_weights=(1.0,),
        schedule_consistency_weights=(0.0,),
        guidance_override_modes=("reset",),
        eval_guidance_qs=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
        eval_guidance_seed_offset=100000,
        eval_guidance_include_root=True,
        include_rf_root_baseline=False,
        include_doc_level_baseline=False,
        rf_n_estimators=200,
        rf_max_depth=16,
        rf_min_samples_leaf=5,
        data_seeds=(),
        seeds=seeds,
        model_families=("additive", "neural"),
        feature_modes=("full",),
        state_dims=(32,),
        hidden_dims=(128,),
        hidden_dim_multiplier=None,
        hidden_dim_min=64,
        n_epochs=12,
        violation_tau=0.0,
    )
    return IdentifiableZeroLongrunPolicy(
        profile="longrun_equiv_v1",
        segment_scale=SegmentOpsSweepPolicy(
            train_docs=(100, 200, 500, 1000, 2000, 4000, 8000, 12000),
            test_docs=5000,
            audit_fractions=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
            topic_phi_docs=(0,),
            topic_phi_estimators=("true", "embedding_spectral"),
            topic_processes=("segments",),
            lambda_multipliers=(1.0,),
            seeds=seeds,
            topic_source="infer",
            feature_inference="hard",
            n_topics=8,
            vocab_size=512,
            min_tokens=384,
            max_tokens=384,
            leaf_tokens=16,
            run_all_feature_modes=True,
        ),
        ctree_scale=CtreeSweepPolicy(
            train_docs=(256, 512, 1024, 2048, 4096),
            seeds=seeds,
            calibration_rates=(0.01, 0.02, 0.05, 0.1),
            eval_leaf_rates=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
            eval_internal_rates=(0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
            topic_phi_estimators=("spectral_numpy",),
            topic_phi_docs_values=(0,),
            leaf_theta_estimators=("lstsq",),
            topic_processes=("segments",),
            n_topics=8,
            vocab_size=512,
            min_segments=8,
            max_segments=16,
            min_seg_tokens=32,
            max_seg_tokens=96,
            fixed_leaf_tokens=32,
            n_books_test=5000,
            alpha_topic=0.4,
            beta_word=0.2,
            segment_concentration=12.0,
            segment_background=4.0,
            calibration_policy="uniform",
            eval_internal_query_design="risk",
            topic_phi_permute=True,
            include_full_doc_theta_baseline=False,
        ),
        ctree_equiv=CtreeSweepPolicy(
            train_docs=(256, 512, 1024, 2048, 4096),
            seeds=seeds,
            calibration_rates=(0.01, 0.02, 0.05, 0.1),
            eval_leaf_rates=(0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
            eval_internal_rates=(0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
            topic_phi_estimators=("spectral_numpy",),
            topic_phi_docs_values=(0,),
            leaf_theta_estimators=("lstsq",),
            topic_processes=("segments",),
            n_topics=8,
            vocab_size=512,
            min_segments=8,
            max_segments=16,
            min_seg_tokens=32,
            max_seg_tokens=96,
            fixed_leaf_tokens=32,
            n_books_test=5000,
            alpha_topic=0.4,
            beta_word=0.2,
            segment_concentration=12.0,
            segment_background=4.0,
            calibration_policy="uniform",
            eval_internal_query_design="risk",
            topic_phi_permute=True,
            include_full_doc_theta_baseline=False,
        ),
        markov_scale=MarkovSweepPolicy(
            train_docs=(100, 200, 500, 1000, 2000, 4000, 8000),
            leaf_query_rates=(0.0, 0.05, 0.1, 0.25, 0.5, 1.0),
            eval_guidance_trials=3,
            suite_role="longrun_scale",
            **common_markov,
        ),
        markov_equiv=MarkovSweepPolicy(
            train_docs=(1000, 4000, 8000),
            leaf_query_rates=(0.0, 0.05, 0.1, 0.25, 0.5, 1.0),
            eval_guidance_trials=8,
            suite_role="longrun_equiv",
            **common_markov,
        ),
        pilot_cmd_count=int(pilot_cmd_count),
        target_main_jobs=int(target_main_jobs),
        target_pilot_minutes=int(target_pilot_minutes),
    )


__all__ = [
    "CtreeSweepPolicy",
    "IdentifiableZeroLongrunPolicy",
    "IdentifiableZeroPublicationCleanPolicy",
    "MarkovSweepPolicy",
    "SegmentOpsSweepPolicy",
    "resolve_identifiable_zero_longrun_policy",
    "resolve_identifiable_zero_publication_clean_policy",
]
