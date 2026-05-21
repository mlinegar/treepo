from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

from treepo._research.ctreepo.contracts import (
    LAW_SET_ALL,
    LAW_SET_LEAF_AND_MERGE_PRESERVATION,
    LAW_SET_LEAF_PRESERVATION_ONLY,
    LAW_SET_MERGE_PRESERVATION_ONLY,
    LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY,
    LAW_SET_ROOT_ONLY,
)


@dataclass(frozen=True)
class LawWeightProfile:
    label: str
    c1_relative_weight: float
    c2_relative_weight: float
    c3_relative_weight: float


@dataclass(frozen=True)
class MarkovSanityPolicy:
    train_docs: Tuple[int, ...]
    n_regimes: Tuple[int, ...]
    fixed_leaf_tokens: Tuple[int, ...]
    data_seeds: Tuple[int, ...]
    model_seeds: Tuple[int, ...]
    val_docs: int
    test_docs: int
    learned_law_packages: Tuple[str, ...]
    exact_families: Tuple[str, ...]
    state_dims: Tuple[int, ...]
    hidden_dims: Tuple[int, ...]
    root_weights: Tuple[float, ...]
    n_epochs: int


@dataclass(frozen=True)
class MarkovTransitionMapPolicy:
    n_regimes: int
    fixed_leaf_tokens: int
    train_docs: Tuple[int, ...]
    val_docs: int
    test_docs: int
    audit_fractions: Tuple[float, ...]
    law_packages: Tuple[str, ...]
    data_seeds: Tuple[int, ...]
    model_seeds: Tuple[int, ...]
    state_dims: Tuple[int, ...]
    hidden_dims: Tuple[int, ...]
    root_weights: Tuple[float, ...]
    n_epochs: int


@dataclass(frozen=True)
class MarkovMechanismPolicy:
    selection_limit: int
    law_packages: Tuple[str, ...]
    root_weights: Tuple[float, ...]
    data_seeds: Tuple[int, ...]
    model_seeds: Tuple[int, ...]


@dataclass(frozen=True)
class MarkovCapacityAppendixPolicy:
    caps: Tuple[Tuple[int, int], ...]
    n_regimes: int
    fixed_leaf_tokens: int
    train_docs: Tuple[int, ...]
    val_docs: int
    test_docs: int
    audit_fractions: Tuple[float, ...]
    law_packages: Tuple[str, ...]
    data_seeds: Tuple[int, ...]
    model_seeds: Tuple[int, ...]
    root_weights: Tuple[float, ...]
    n_epochs: int


@dataclass(frozen=True)
class MarkovCrossDgpPolicy:
    caps: Tuple[Tuple[int, int], ...]
    n_regimes: int
    fixed_leaf_tokens: int
    train_docs: Tuple[int, ...]
    val_docs: int
    test_docs: int
    audit_fractions: Tuple[float, ...]
    law_packages: Tuple[str, ...]
    data_seeds: Tuple[int, ...]
    model_seeds: Tuple[int, ...]
    root_weights: Tuple[float, ...]
    n_epochs: int


@dataclass(frozen=True)
class MarkovWeightAblationPolicy:
    caps: Tuple[Tuple[int, int], ...]
    n_regimes: int
    fixed_leaf_tokens: int
    train_docs: Tuple[int, ...]
    val_docs: int
    test_docs: int
    audit_fractions: Tuple[float, ...]
    baseline_law_packages: Tuple[str, ...]
    data_seeds: Tuple[int, ...]
    model_seeds: Tuple[int, ...]
    n_epochs: int
    weight_profiles: Tuple[LawWeightProfile, ...]


@dataclass(frozen=True)
class MarkovLawStressPolicy:
    smoke: bool
    sanity: MarkovSanityPolicy
    transition_map: MarkovTransitionMapPolicy
    mechanism: MarkovMechanismPolicy
    capacity_appendix: MarkovCapacityAppendixPolicy
    cross_dgp: MarkovCrossDgpPolicy
    weight_ablation: MarkovWeightAblationPolicy

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LDASanityPolicy:
    taus: Tuple[float, ...]
    quadratic_utility_weights: Tuple[float, ...]
    seeds: Tuple[int, ...]
    train_docs: int
    test_docs: int
    learned_law_set_ids: Tuple[str, ...]
    exact_families: Tuple[str, ...]
    law_leaf_query_rate: float
    law_internal_query_rate: float
    analysis_partition_mode: str


@dataclass(frozen=True)
class LDATransitionMapPolicy:
    taus: Tuple[float, ...]
    quadratic_utility_weights: Tuple[float, ...]
    law_set_ids: Tuple[str, ...]
    seeds: Tuple[int, ...]
    train_docs: int
    test_docs: int
    law_leaf_query_rate: float
    law_internal_query_rate: float
    analysis_partition_mode: str


@dataclass(frozen=True)
class LDAMechanismPolicy:
    taus: Tuple[float, ...]
    quadratic_utility_weights: Tuple[float, ...]
    analysis_partition_modes: Tuple[str, ...]
    law_set_ids: Tuple[str, ...]
    seeds: Tuple[int, ...]
    train_docs: int
    test_docs: int
    law_leaf_query_rate: float
    law_internal_query_rate: float


@dataclass(frozen=True)
class LDALawStressPolicy:
    smoke: bool
    sanity: LDASanityPolicy
    transition_map: LDATransitionMapPolicy
    mechanism: LDAMechanismPolicy

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_markov_law_stress_policy(*, smoke: bool) -> MarkovLawStressPolicy:
    if bool(smoke):
        return MarkovLawStressPolicy(
            smoke=True,
            sanity=MarkovSanityPolicy(
                train_docs=(32,),
                n_regimes=(2,),
                fixed_leaf_tokens=(8,),
                data_seeds=(0,),
                model_seeds=(0,),
                val_docs=64,
                test_docs=64,
                learned_law_packages=(
                    "root_only",
                    "c1_only",
                    "c2_only",
                    "c3_only",
                    "c1c3",
                    "all_laws",
                    "all_laws_plus_sched",
                ),
                exact_families=("exact", "leaf_bucket", "count_only", "flip_R2"),
                state_dims=(64,),
                hidden_dims=(256,),
                root_weights=(1.0,),
                n_epochs=2,
            ),
            transition_map=MarkovTransitionMapPolicy(
                n_regimes=4,
                fixed_leaf_tokens=16,
                train_docs=(64, 128),
                val_docs=128,
                test_docs=128,
                audit_fractions=(0.1, 1.0),
                law_packages=("root_only", "c1c3", "c2_only", "all_laws", "all_laws_plus_sched"),
                data_seeds=(0,),
                model_seeds=(0,),
                state_dims=(64,),
                hidden_dims=(256,),
                root_weights=(1.0,),
                n_epochs=2,
            ),
            mechanism=MarkovMechanismPolicy(
                selection_limit=1,
                law_packages=(
                    "root_only",
                    "c1_only",
                    "c2_only",
                    "c3_only",
                    "c1c3",
                    "all_laws",
                    "sched_only",
                    "all_laws_plus_sched",
                ),
                root_weights=(1.0,),
                data_seeds=(0,),
                model_seeds=(0,),
            ),
            capacity_appendix=MarkovCapacityAppendixPolicy(
                caps=((32, 128), (64, 256)),
                n_regimes=4,
                fixed_leaf_tokens=16,
                train_docs=(128,),
                val_docs=128,
                test_docs=128,
                audit_fractions=(0.1,),
                law_packages=("root_only", "all_laws_plus_sched"),
                data_seeds=(0,),
                model_seeds=(0,),
                root_weights=(1.0,),
                n_epochs=2,
            ),
            cross_dgp=MarkovCrossDgpPolicy(
                caps=((64, 256),),
                n_regimes=4,
                fixed_leaf_tokens=16,
                train_docs=(64,),
                val_docs=128,
                test_docs=128,
                audit_fractions=(0.1, 1.0),
                law_packages=(
                    "root_only",
                    "c1_only",
                    "c2_only",
                    "c3_only",
                    "c1c3",
                    "all_laws",
                    "all_laws_plus_sched",
                ),
                data_seeds=(0,),
                model_seeds=(0,),
                root_weights=(1.0,),
                n_epochs=2,
            ),
            weight_ablation=MarkovWeightAblationPolicy(
                caps=((64, 256),),
                n_regimes=4,
                fixed_leaf_tokens=16,
                train_docs=(64,),
                val_docs=128,
                test_docs=128,
                audit_fractions=(0.1, 1.0),
                baseline_law_packages=("root_only",),
                data_seeds=(0,),
                model_seeds=(0,),
                n_epochs=2,
                weight_profiles=(
                    LawWeightProfile("pure_c2", 0.0, 1.0, 0.0),
                    LawWeightProfile("no_c2", 1.0, 0.0, 4.0),
                    LawWeightProfile("c2_trace_c1c3", 0.05, 1.0, 0.05),
                    LawWeightProfile("c2_light_c1c3", 0.1, 1.0, 0.1),
                    LawWeightProfile("c2_mild_c1c3", 0.25, 1.0, 0.25),
                    LawWeightProfile("c2_moderate_c1c3", 0.5, 1.0, 0.5),
                    LawWeightProfile("c2_very_dominant", 1.0, 8.0, 1.0),
                    LawWeightProfile("c2_dominant", 1.0, 4.0, 1.0),
                    LawWeightProfile("c2_heavy", 1.0, 2.0, 1.0),
                    LawWeightProfile("equal", 1.0, 1.0, 1.0),
                    LawWeightProfile("c1c3_heavy", 2.0, 1.0, 2.0),
                    LawWeightProfile("c3_dominant", 1.0, 1.0, 4.0),
                ),
            ),
        )

    return MarkovLawStressPolicy(
        smoke=False,
        sanity=MarkovSanityPolicy(
            train_docs=(128, 512, 2048),
            n_regimes=(2, 4),
            fixed_leaf_tokens=(8, 16),
            data_seeds=(0, 1),
            model_seeds=(0, 1),
            val_docs=256,
            test_docs=512,
            learned_law_packages=(
                "root_only",
                "c1_only",
                "c2_only",
                "c3_only",
                "c1c3",
                "all_laws",
                "all_laws_plus_sched",
            ),
            exact_families=("exact", "leaf_bucket", "count_only", "flip_R2"),
            state_dims=(64,),
            hidden_dims=(256,),
            root_weights=(1.0,),
            n_epochs=12,
        ),
        transition_map=MarkovTransitionMapPolicy(
            n_regimes=4,
            fixed_leaf_tokens=16,
            train_docs=(128, 512, 2048, 4096),
            val_docs=256,
            test_docs=512,
            audit_fractions=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
            law_packages=("root_only", "c1c3", "c2_only", "all_laws", "all_laws_plus_sched"),
            data_seeds=(0, 1),
            model_seeds=(0, 1),
            state_dims=(64,),
            hidden_dims=(256,),
            root_weights=(1.0,),
            n_epochs=12,
        ),
        mechanism=MarkovMechanismPolicy(
            selection_limit=2,
            law_packages=(
                "root_only",
                "c1_only",
                "c2_only",
                "c3_only",
                "c1c3",
                "all_laws",
                "sched_only",
                "all_laws_plus_sched",
            ),
            root_weights=(0.5, 1.0, 2.0),
            data_seeds=(0, 1),
            model_seeds=(0, 1),
        ),
        capacity_appendix=MarkovCapacityAppendixPolicy(
            caps=((32, 128), (64, 256), (128, 512)),
            n_regimes=4,
            fixed_leaf_tokens=16,
            train_docs=(512, 2048),
            val_docs=256,
            test_docs=512,
            audit_fractions=(0.1, 1.0),
            law_packages=("root_only", "all_laws_plus_sched"),
            data_seeds=(0, 1),
            model_seeds=(0, 1),
            root_weights=(1.0,),
            n_epochs=12,
        ),
        cross_dgp=MarkovCrossDgpPolicy(
            caps=((64, 256), (128, 512)),
            n_regimes=4,
            fixed_leaf_tokens=16,
            train_docs=(256, 1024),
            val_docs=256,
            test_docs=512,
            audit_fractions=(0.1, 0.5, 1.0),
            law_packages=(
                "root_only",
                "c1_only",
                "c2_only",
                "c3_only",
                "c1c3",
                "all_laws",
                "all_laws_plus_sched",
            ),
            data_seeds=(0, 1),
            model_seeds=(0,),
            root_weights=(1.0,),
            n_epochs=12,
        ),
        weight_ablation=MarkovWeightAblationPolicy(
            caps=((64, 256), (128, 512)),
            n_regimes=4,
            fixed_leaf_tokens=16,
            train_docs=(256, 1024),
            val_docs=256,
            test_docs=512,
            audit_fractions=(0.1, 0.5, 1.0),
            baseline_law_packages=("root_only",),
            data_seeds=(0, 1),
            model_seeds=(0,),
            n_epochs=12,
            weight_profiles=(
                LawWeightProfile("pure_c2", 0.0, 1.0, 0.0),
                LawWeightProfile("no_c2", 1.0, 0.0, 4.0),
                LawWeightProfile("c2_trace_c1c3", 0.05, 1.0, 0.05),
                LawWeightProfile("c2_light_c1c3", 0.1, 1.0, 0.1),
                LawWeightProfile("c2_mild_c1c3", 0.25, 1.0, 0.25),
                LawWeightProfile("c2_moderate_c1c3", 0.5, 1.0, 0.5),
                LawWeightProfile("c2_very_dominant", 1.0, 8.0, 1.0),
                LawWeightProfile("c2_dominant", 1.0, 4.0, 1.0),
                LawWeightProfile("c2_heavy", 1.0, 2.0, 1.0),
                LawWeightProfile("equal", 1.0, 1.0, 1.0),
                LawWeightProfile("c1c3_heavy", 2.0, 1.0, 2.0),
                LawWeightProfile("c3_dominant", 1.0, 1.0, 4.0),
            ),
        ),
    )


def resolve_lda_law_stress_policy(*, smoke: bool) -> LDALawStressPolicy:
    if bool(smoke):
        return LDALawStressPolicy(
            smoke=True,
            sanity=LDASanityPolicy(
                taus=(1.0, 8.0),
                quadratic_utility_weights=(0.0, 1.5),
                seeds=(0,),
                train_docs=32,
                test_docs=16,
                learned_law_set_ids=(
                    LAW_SET_ROOT_ONLY,
                    LAW_SET_LEAF_PRESERVATION_ONLY,
                    LAW_SET_MERGE_PRESERVATION_ONLY,
                    LAW_SET_ALL,
                ),
                exact_families=("oracle", "scrambled_topics", "uniform_prior", "adversarial_merge"),
                law_leaf_query_rate=0.25,
                law_internal_query_rate=0.25,
                analysis_partition_mode="aligned",
            ),
            transition_map=LDATransitionMapPolicy(
                taus=(1.0, 8.0),
                quadratic_utility_weights=(0.0, 1.5),
                law_set_ids=(LAW_SET_ROOT_ONLY, LAW_SET_ALL),
                seeds=(0,),
                train_docs=32,
                test_docs=16,
                law_leaf_query_rate=0.10,
                law_internal_query_rate=0.10,
                analysis_partition_mode="aligned",
            ),
            mechanism=LDAMechanismPolicy(
                taus=(1.0, 8.0),
                quadratic_utility_weights=(1.5,),
                analysis_partition_modes=("aligned", "shift_half"),
                law_set_ids=(LAW_SET_ALL,),
                seeds=(0,),
                train_docs=32,
                test_docs=16,
                law_leaf_query_rate=0.10,
                law_internal_query_rate=0.10,
            ),
        )

    return LDALawStressPolicy(
        smoke=False,
        sanity=LDASanityPolicy(
            taus=(1.0, 4.0, 16.0),
            quadratic_utility_weights=(0.0, 0.5, 1.5),
            seeds=(0, 1, 2),
            train_docs=128,
            test_docs=64,
            learned_law_set_ids=(
                LAW_SET_ROOT_ONLY,
                LAW_SET_LEAF_PRESERVATION_ONLY,
                LAW_SET_MERGE_PRESERVATION_ONLY,
                LAW_SET_ALL,
            ),
            exact_families=("oracle", "scrambled_topics", "uniform_prior", "adversarial_merge"),
            law_leaf_query_rate=0.25,
            law_internal_query_rate=0.25,
            analysis_partition_mode="aligned",
        ),
        transition_map=LDATransitionMapPolicy(
            taus=(1.0, 2.0, 4.0, 8.0, 16.0),
            quadratic_utility_weights=(0.0, 0.1, 0.5, 1.0, 1.5, 3.0),
            law_set_ids=(
                LAW_SET_ROOT_ONLY,
                LAW_SET_LEAF_PRESERVATION_ONLY,
                LAW_SET_MERGE_PRESERVATION_ONLY,
                LAW_SET_LEAF_AND_MERGE_PRESERVATION,
                LAW_SET_ALL,
            ),
            seeds=(0, 1, 2, 3),
            train_docs=256,
            test_docs=128,
            law_leaf_query_rate=0.10,
            law_internal_query_rate=0.10,
            analysis_partition_mode="aligned",
        ),
        mechanism=LDAMechanismPolicy(
            taus=(1.0, 4.0, 16.0),
            quadratic_utility_weights=(0.5, 1.5),
            analysis_partition_modes=("aligned", "coarsen_2x", "shift_half", "random_same_count"),
            law_set_ids=(LAW_SET_ROOT_ONLY, LAW_SET_ALL),
            seeds=(0, 1, 2, 3),
            train_docs=256,
            test_docs=128,
            law_leaf_query_rate=0.10,
            law_internal_query_rate=0.10,
        ),
    )


__all__ = [
    "LDALawStressPolicy",
    "LDAMechanismPolicy",
    "LDASanityPolicy",
    "LDATransitionMapPolicy",
    "LawWeightProfile",
    "MarkovCapacityAppendixPolicy",
    "MarkovCrossDgpPolicy",
    "MarkovLawStressPolicy",
    "MarkovMechanismPolicy",
    "MarkovSanityPolicy",
    "MarkovTransitionMapPolicy",
    "MarkovWeightAblationPolicy",
    "resolve_lda_law_stress_policy",
    "resolve_markov_law_stress_policy",
]
