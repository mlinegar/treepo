"""
Stage-2 and Stage-3 tree-relevant LDA family: local topic mixtures with nonlinear utility.

Stage 2 asks when per-section analysis can beat pooling under a latent local-mixture DGP:

    pi_d ~ Dir(alpha)
    pi_{d,b} | pi_d ~ Dir(tau * pi_d)
    z_{d,t} | b(t)=b ~ Cat(pi_{d,b})
    x_{d,t} | z_{d,t}=k ~ Cat(phi_k)

    y_d = sum_b N_{d,b} h(pi_{d,b})
    h(pi) = theta^T pi + quadratic_weight * pi^T W pi

Stage 3 keeps the same ladder but adds:
- variable latent section lengths built from atomic token blocks,
- analysis partitions that may misalign with the latent sections,
- weighted vs unweighted aggregation checks,
- budgeted section supervision with IPW-inspired ridge training,
- HT/Hajek held-out evaluation diagnostics.

Backward compatibility:
- old defaults reproduce the original equal-leaf aligned setting,
- legacy method names remain present,
- Stage-2 builders can continue to call this module unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from treepo._research.core.logged_supervision import (
    LoggedLabelObservation,
    ObservationUnitKind,
    SamplingMetadata,
    summarize_logged_observations,
    write_logged_observations_jsonl,
)
from treepo._research.core.ops_checks import LawKind
from treepo._research.ctreepo.contracts import (
    LAW_ID_LEAF_PRESERVATION,
    LAW_ID_MERGE_PRESERVATION,
    LAW_ID_ON_RANGE_IDEMPOTENCE,
    LAW_SET_ALL,
    LAW_SET_LEAF_AND_MERGE_PRESERVATION,
    LAW_SET_LEAF_PRESERVATION_ONLY,
    LAW_SET_MERGE_PRESERVATION_ONLY,
    LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY,
    LAW_SET_ROOT_ONLY,
)
from treepo._research.ctreepo.sim.composite_objective import (
    OBJECTIVE_ESTIMATOR_KEYS,
    CompositeObjectiveSpec,
    evaluate_composite_objective,
    evaluate_composite_objective_from_metrics,
    objective_estimator_alias,
    resolve_root_local_objective_weights,
    scalarize_objective_estimates,
)
from treepo._research.ctreepo.sim.core import lda_tree_recovery as _base
from treepo._research.ctreepo.sim.core.lda_tree_utility_vector import (
    VALID_EMISSION_MODES,
    VALID_UTILITY_DESIGNS,
    leaf_fraction_label,
    leaf_tokens_from_fraction,
    sample_topic_anchored_utility_matrix,
    utility_vector_from_counts,
)
from treepo._research.ctreepo.sim.core.segment_lda_ops_weight_recovery import (
    _splitmix64,
    sample_sparse_oracle_weights,
    sample_topic_distributions,
)
from treepo._research.ctreepo.sim.learning_problem import attach_local_law_learning_problem
from treepo._research.ctreepo.sim.objective_semantics import latent_quadratic_utility_objective_semantics
from treepo._research.tree.compositional_learning import shared_logged_substructure_observation
from treepo._research.tree.ipw import (
    MIN_PROPENSITY,
    NodeType,
    TreeSample,
    effective_sample_size,
    empirical_bernstein_ci,
    hajek_estimate,
    hajek_ht_comparison,
    horvitz_thompson_mean,
    max_weight,
)
from treepo._research.ctreepo.sim.local_law_learnability import (
    DownstreamMetrics,
    GArtifact,
    LocalLawCounterexampleEvaluation,
    LocalLawMetrics,
    LocalLawPolicyEvaluation,
    LocalLawRunSummary,
    PolicyRole,
    SupportBudgetSummary,
    artifact_index,
    write_json_g_artifact,
)
from treepo._research.training.supervision import (
    DenseScalarRidgeModelConfig,
    DenseScalarRidgeTrainingConfig,
    DenseSupervisionExample,
    OPTIMIZER_FAMILY_CLOSED_FORM_LINEAR,
    REPRESENTATION_DENSE_FEATURE_VECTOR,
    TARGET_SCALAR,
    build_dense_sampled_substructure_supervision_dataset,
    fit_dense_scalar_ridge_regressor,
    supervision_training_contract,
)


VALID_BUDGET_REGIMES: Tuple[str, ...] = ("fixed_oracle_budget", "all_leaves_labeled")
VALID_LATENT_PARTITION_MODES: Tuple[str, ...] = ("equal", "variable")
VALID_LATENT_LENGTH_PROFILES: Tuple[str, ...] = ("equal", "bimodal", "long_tail")
VALID_ANALYSIS_PARTITION_MODES: Tuple[str, ...] = (
    "aligned",
    "coarsen_2x",
    "refine_2x",
    "shift_half",
    "random_same_count",
)
VALID_QUERY_DESIGNS: Tuple[str, ...] = ("uniform", "proxy_priority", "proxy_adversarial")
VALID_PROPENSITY_PROXIES: Tuple[str, ...] = ("l1_deviation",)
VALID_LOCAL_LAW_MODES: Tuple[str, ...] = ("off", "diagnostics", "diagnostics_and_learned")
VALID_LAW_INTERNAL_QUERY_DESIGNS: Tuple[str, ...] = ("uniform", "risk")
Span = Tuple[int, int]


@dataclass(frozen=True)
class LeafLocalMixtureDoc:
    tokens: Tuple[int, ...]
    topics: Tuple[int, ...]
    global_topic_weights: Tuple[float, ...]
    local_topic_weights: Tuple[Tuple[float, ...], ...]
    latent_section_spans: Tuple[Span, ...]
    latent_section_block_spans: Tuple[Span, ...]
    atomic_block_tokens: int


@dataclass(frozen=True)
class AnalysisPartitionView:
    analysis_section_spans: Tuple[Span, ...]
    analysis_section_block_spans: Tuple[Span, ...]
    overlap_tokens: np.ndarray  # [analysis, latent]
    latent_weights: Tuple[float, ...]
    analysis_weights: Tuple[float, ...]
    analysis_topic_weights: Tuple[Tuple[float, ...], ...]
    overlap_row_normalized: np.ndarray
    overlap_col_normalized: np.ndarray


@dataclass(frozen=True)
class LeafLocalMixtureUtilityConfig:
    n_topics: int = 8
    vocab_size: int = 512
    doc_tokens: int = 384
    doc_topic_concentration: float = 0.6

    topic_concentration: float = 0.2
    emission_mode: str = "anchored"
    anchor_words_per_topic: int = 20
    anchor_multiplier: float = 25.0

    utility_dim: int = 16
    utility_design: str = "topic_anchored_sparse"

    atomic_block_tokens: int = 16
    latent_leaf_tokens: int = 16
    latent_partition_mode: str = "equal"
    latent_length_profile: str = "equal"
    leaf_fraction: float = 1.0 / 24.0
    analysis_partition_mode: str = "aligned"
    analysis_leaf_tokens: int = 0
    local_mixture_concentration: float = 1.0

    relevant_topics: int = 2
    theta_scale: float = 1.0
    zero_diagonal: bool = False
    # Legacy internal name for the quadratic utility weight.
    lambda_multiplier: float = 1.0

    train_docs: int = 512
    val_docs: int = 0
    test_docs: int = 256
    val_seed_offset: int = 5_000
    test_seed_offset: int = 10_000

    budget_regime: str = "all_leaves_labeled"
    leaf_label_budget: float = 8.0
    ridge_alpha: float = 1e-3

    query_design: str = "uniform"
    doc_sample_rate: float = 1.0
    heldout_doc_sample_rate: float = 0.5
    target_query_budget_per_doc: float = 0.0
    propensity_floor: float = 0.10
    propensity_ceiling: float = 0.90
    propensity_proxy: str = "l1_deviation"
    ipw_stabilized_clip: float = 20.0
    ipw_delta: float = 0.05

    local_law_mode: str = "off"
    law_package: str = "all_laws"
    exact_family: str = ""
    law_leaf_query_rate: float = 0.10
    law_internal_query_rate: float = 0.10
    law_leaf_query_design: str = "uniform"
    law_internal_query_design: str = "uniform"
    local_law_weight: Optional[float] = None
    law_task_objective_weight: float = 1.0
    law_c1_weight: float = 1.0 / 3.0
    law_c3_weight: float = 1.0 / 3.0
    law_c2_proxy_weight: float = 1.0 / 3.0
    law_calibration_ridge: float = 1e-3
    law_eval_leaf_sample_rate: float = 0.25
    law_eval_internal_sample_rate: float = 0.25
    law_c1_threshold: float = 0.20
    law_c3_threshold: float = 0.20
    law_c2_threshold: float = 0.20
    suite_role: str = ""
    artifact_dir: str = ""
    save_logged_observations: bool = False

    inference_prior_mass: float = 0.25
    inference_max_iter: int = 200
    inference_tol: float = 1e-9

    seed: int = 0


@dataclass(frozen=True)
class LeafLocalMixtureUtilityWorld:
    signature: Dict[str, object]
    topic_meta: Dict[str, object]
    topics_phi: Tuple[np.ndarray, ...]
    utility_matrix: np.ndarray
    utility_topic_meta: Dict[str, object]
    theta_true: np.ndarray
    W_base: np.ndarray
    docs_train: Tuple[LeafLocalMixtureDoc, ...]
    docs_val: Tuple[LeafLocalMixtureDoc, ...]
    docs_test: Tuple[LeafLocalMixtureDoc, ...]


@dataclass(frozen=True)
class LeafLocalMixtureUtilitySummary:
    family: str
    target_kind: str
    config: Dict[str, object]
    topic_meta: Dict[str, object]
    utility_topic_meta: Dict[str, object]
    utility_truth: Dict[str, object]
    world_stats: Dict[str, object]
    heterogeneity: Dict[str, object]
    methods: Dict[str, object]
    objective: Dict[str, object] = field(default_factory=dict)
    stage3: Dict[str, object] = field(default_factory=dict)
    local_law: Dict[str, object] = field(default_factory=dict)
    local_law_learnability: Dict[str, object] = field(default_factory=dict)
    g_artifacts: Dict[str, object] = field(default_factory=dict)
    is_stale_generation: bool = False

    def to_json(self) -> str:
        payload = {
            "problem_id": self.family,
            "target_kind": self.target_kind,
            "config": self.config,
            "topic_meta": self.topic_meta,
            "utility_topic_meta": self.utility_topic_meta,
            "utility_truth": self.utility_truth,
            "world_stats": self.world_stats,
            "heterogeneity": self.heterogeneity,
            "methods": self.methods,
            "is_stale_generation": bool(self.is_stale_generation),
        }
        if self.objective:
            payload["objective"] = self.objective
        if self.stage3:
            payload["stage3"] = self.stage3
        if self.local_law:
            payload["local_law"] = self.local_law
        if self.local_law_learnability:
            payload["local_law_learnability"] = self.local_law_learnability
        if self.g_artifacts:
            payload["g_artifacts"] = self.g_artifacts
        return json.dumps(payload, indent=2, sort_keys=True)


def _validate_config(config: LeafLocalMixtureUtilityConfig) -> None:
    if int(config.n_topics) < 2:
        raise ValueError("n_topics must be >= 2")
    if int(config.vocab_size) < 2:
        raise ValueError("vocab_size must be >= 2")
    if int(config.doc_tokens) <= 0:
        raise ValueError("doc_tokens must be positive")
    if float(config.doc_topic_concentration) <= 0.0:
        raise ValueError("doc_topic_concentration must be positive")
    if float(config.topic_concentration) <= 0.0:
        raise ValueError("topic_concentration must be positive")
    if str(config.emission_mode).strip().lower() not in VALID_EMISSION_MODES:
        raise ValueError(f"emission_mode must be one of {VALID_EMISSION_MODES}")
    if int(config.utility_dim) <= 0:
        raise ValueError("utility_dim must be positive")
    if str(config.utility_design).strip().lower() not in VALID_UTILITY_DESIGNS:
        raise ValueError(f"utility_design must be one of {VALID_UTILITY_DESIGNS}")
    if int(config.atomic_block_tokens) <= 0:
        raise ValueError("atomic_block_tokens must be positive")
    if int(config.doc_tokens) % int(config.atomic_block_tokens) != 0:
        raise ValueError("doc_tokens must be divisible by atomic_block_tokens")
    if int(config.latent_leaf_tokens) <= 0:
        raise ValueError("latent_leaf_tokens must be positive")
    if int(config.latent_leaf_tokens) % int(config.atomic_block_tokens) != 0:
        raise ValueError("latent_leaf_tokens must be a multiple of atomic_block_tokens")
    if str(config.latent_partition_mode).strip().lower() not in VALID_LATENT_PARTITION_MODES:
        raise ValueError(f"latent_partition_mode must be one of {VALID_LATENT_PARTITION_MODES}")
    if str(config.latent_length_profile).strip().lower() not in VALID_LATENT_LENGTH_PROFILES:
        raise ValueError(f"latent_length_profile must be one of {VALID_LATENT_LENGTH_PROFILES}")
    if str(config.analysis_partition_mode).strip().lower() not in VALID_ANALYSIS_PARTITION_MODES:
        raise ValueError(f"analysis_partition_mode must be one of {VALID_ANALYSIS_PARTITION_MODES}")
    if int(config.analysis_leaf_tokens) < 0:
        raise ValueError("analysis_leaf_tokens must be non-negative")
    if (
        int(config.analysis_leaf_tokens) > 0
        and int(config.analysis_leaf_tokens) % int(config.atomic_block_tokens) != 0
    ):
        raise ValueError("analysis_leaf_tokens must be a multiple of atomic_block_tokens")
    eval_leaf_tokens = leaf_tokens_from_fraction(
        int(config.doc_tokens), float(config.leaf_fraction)
    )
    if int(config.doc_tokens) % int(eval_leaf_tokens) != 0:
        raise ValueError("doc_tokens must be divisible by the evaluation leaf size")
    if float(config.local_mixture_concentration) <= 0.0:
        raise ValueError("local_mixture_concentration must be positive")
    if int(config.relevant_topics) <= 0:
        raise ValueError("relevant_topics must be positive")
    if int(config.train_docs) <= 0 or int(config.test_docs) <= 0:
        raise ValueError("train_docs and test_docs must be positive")
    if int(config.val_docs) < 0:
        raise ValueError("val_docs must be non-negative")
    if int(config.val_seed_offset) < 0 or int(config.test_seed_offset) < 0:
        raise ValueError("val_seed_offset and test_seed_offset must be non-negative")
    if str(config.budget_regime).strip().lower() not in VALID_BUDGET_REGIMES:
        raise ValueError(f"budget_regime must be one of {VALID_BUDGET_REGIMES}")
    if float(config.leaf_label_budget) <= 0.0:
        raise ValueError("leaf_label_budget must be positive")
    if float(config.ridge_alpha) < 0.0:
        raise ValueError("ridge_alpha must be non-negative")
    if str(config.query_design).strip().lower() not in VALID_QUERY_DESIGNS:
        raise ValueError(f"query_design must be one of {VALID_QUERY_DESIGNS}")
    for name, value in (
        ("doc_sample_rate", config.doc_sample_rate),
        ("heldout_doc_sample_rate", config.heldout_doc_sample_rate),
        ("propensity_floor", config.propensity_floor),
        ("propensity_ceiling", config.propensity_ceiling),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0 or float(value) > 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if float(config.propensity_floor) > float(config.propensity_ceiling):
        raise ValueError("propensity_floor must be <= propensity_ceiling")
    if float(config.target_query_budget_per_doc) < 0.0:
        raise ValueError("target_query_budget_per_doc must be non-negative")
    if str(config.propensity_proxy).strip().lower() not in VALID_PROPENSITY_PROXIES:
        raise ValueError(f"propensity_proxy must be one of {VALID_PROPENSITY_PROXIES}")
    if float(config.ipw_stabilized_clip) <= 0.0:
        raise ValueError("ipw_stabilized_clip must be positive")
    if float(config.ipw_delta) <= 0.0 or float(config.ipw_delta) >= 1.0:
        raise ValueError("ipw_delta must be in (0, 1)")
    if str(config.local_law_mode).strip().lower() not in VALID_LOCAL_LAW_MODES:
        raise ValueError(f"local_law_mode must be one of {VALID_LOCAL_LAW_MODES}")
    if str(config.law_leaf_query_design).strip().lower() not in VALID_QUERY_DESIGNS:
        raise ValueError(f"law_leaf_query_design must be one of {VALID_QUERY_DESIGNS}")
    if (
        str(config.law_internal_query_design).strip().lower()
        not in VALID_LAW_INTERNAL_QUERY_DESIGNS
    ):
        raise ValueError(
            f"law_internal_query_design must be one of {VALID_LAW_INTERNAL_QUERY_DESIGNS}"
        )
    for name, value in (
        ("law_leaf_query_rate", config.law_leaf_query_rate),
        ("law_internal_query_rate", config.law_internal_query_rate),
        ("law_eval_leaf_sample_rate", config.law_eval_leaf_sample_rate),
        ("law_eval_internal_sample_rate", config.law_eval_internal_sample_rate),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0 or float(value) > 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    for name, value in (
        ("law_task_objective_weight", config.law_task_objective_weight),
        ("law_c1_weight", config.law_c1_weight),
        ("law_c3_weight", config.law_c3_weight),
        ("law_c2_proxy_weight", config.law_c2_proxy_weight),
        ("law_calibration_ridge", config.law_calibration_ridge),
        ("law_c1_threshold", config.law_c1_threshold),
        ("law_c3_threshold", config.law_c3_threshold),
        ("law_c2_threshold", config.law_c2_threshold),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if config.local_law_weight is not None:
        if (
            not math.isfinite(float(config.local_law_weight))
            or float(config.local_law_weight) < 0.0
            or float(config.local_law_weight) > 1.0
        ):
            raise ValueError("local_law_weight must be in [0, 1]")
        explicit_overrides = (
            not math.isclose(float(config.law_task_objective_weight), 1.0),
            not math.isclose(float(config.law_c1_weight), 1.0 / 3.0),
            not math.isclose(float(config.law_c3_weight), 1.0 / 3.0),
            not math.isclose(float(config.law_c2_proxy_weight), 1.0 / 3.0),
        )
        if any(explicit_overrides):
            raise ValueError(
                "local_law_weight is mutually exclusive with explicit law/root weights"
            )
    if float(config.inference_prior_mass) < 0.0:
        raise ValueError("inference_prior_mass must be non-negative")
    if int(config.inference_max_iter) <= 0:
        raise ValueError("inference_max_iter must be positive")
    if float(config.inference_tol) < 0.0:
        raise ValueError("inference_tol must be non-negative")


def _local_law_objective_spec(
    config: LeafLocalMixtureUtilityConfig,
) -> CompositeObjectiveSpec:
    task_weight, law_weights, input_mode = _resolve_lda_objective_weights(config)
    return CompositeObjectiveSpec(
        name="configured_objective",
        selection_metric_name="configured_objective",
        root_metric_name="mean_aux_oracle_target_abs_error",
        root_share=float(task_weight),
        local_law_component_weights=law_weights,
        auxiliary_diagnostic_weights={},
        weighting_scheme=str(input_mode),
        root_share_source=(
            "derived_from_local_law_weight"
            if str(input_mode) == "lambda"
            else "normalized_explicit_weights"
        ),
        metadata={
            "objective_input_mode": str(input_mode),
            "root_metric_name": "mean_aux_oracle_target_abs_error",
            "task_delta_metric_name": "mean_aux_oracle_target_delta",
            "local_law_metric_names": {
                LAW_ID_LEAF_PRESERVATION: "mean_c1",
                LAW_ID_ON_RANGE_IDEMPOTENCE: "mean_c2_proxy",
                LAW_ID_MERGE_PRESERVATION: "mean_c3",
            },
        },
    )


def _world_signature(config: LeafLocalMixtureUtilityConfig) -> Dict[str, object]:
    return {
        "problem_id": "leaf_local_mixture_utility",
        "n_topics": int(config.n_topics),
        "vocab_size": int(config.vocab_size),
        "doc_tokens": int(config.doc_tokens),
        "doc_topic_concentration": float(config.doc_topic_concentration),
        "topic_concentration": float(config.topic_concentration),
        "emission_mode": str(config.emission_mode),
        "anchor_words_per_topic": int(config.anchor_words_per_topic),
        "anchor_multiplier": float(config.anchor_multiplier),
        "utility_dim": int(config.utility_dim),
        "utility_design": str(config.utility_design),
        "atomic_block_tokens": int(config.atomic_block_tokens),
        "latent_leaf_tokens": int(config.latent_leaf_tokens),
        "latent_partition_mode": str(config.latent_partition_mode),
        "latent_length_profile": str(config.latent_length_profile),
        "local_mixture_concentration": float(config.local_mixture_concentration),
        "relevant_topics": int(config.relevant_topics),
        "theta_scale": float(config.theta_scale),
        "zero_diagonal": bool(config.zero_diagonal),
        "seed": int(config.seed),
    }


def leaf_local_mixture_utility_world_cache_signature(
    config: LeafLocalMixtureUtilityConfig,
    *,
    train_docs_capacity: int,
    val_docs_capacity: int,
    test_docs_capacity: int,
) -> Dict[str, object]:
    return {
        **_world_signature(config),
        "train_docs_capacity": int(train_docs_capacity),
        "val_docs_capacity": int(val_docs_capacity),
        "test_docs_capacity": int(test_docs_capacity),
    }


def _resolve_analysis_leaf_tokens(config: LeafLocalMixtureUtilityConfig) -> int:
    if int(config.analysis_leaf_tokens) > 0:
        return int(config.analysis_leaf_tokens)
    return int(leaf_tokens_from_fraction(int(config.doc_tokens), float(config.leaf_fraction)))


def _leaf_metadata(config: LeafLocalMixtureUtilityConfig) -> Dict[str, object]:
    eval_leaf_tokens = int(
        leaf_tokens_from_fraction(int(config.doc_tokens), float(config.leaf_fraction))
    )
    return {
        "doc_tokens": int(config.doc_tokens),
        "atomic_block_tokens": int(config.atomic_block_tokens),
        "latent_leaf_tokens": int(config.latent_leaf_tokens),
        "latent_leaf_fraction": float(int(config.latent_leaf_tokens) / int(config.doc_tokens)),
        "latent_leaf_fraction_label": leaf_fraction_label(
            float(int(config.latent_leaf_tokens) / int(config.doc_tokens))
        ),
        "leaf_tokens": int(eval_leaf_tokens),
        "leaf_fraction": float(config.leaf_fraction),
        "leaf_fraction_label": leaf_fraction_label(float(config.leaf_fraction)),
        "leaf_percent_of_doc": float(100.0 * float(config.leaf_fraction)),
        "analysis_leaf_tokens": int(_resolve_analysis_leaf_tokens(config)),
        "analysis_partition_mode": str(config.analysis_partition_mode),
        "latent_partition_mode": str(config.latent_partition_mode),
        "latent_length_profile": str(config.latent_length_profile),
    }


def _fit_lengths_to_total(lengths: Sequence[int], total: int, *, min_len: int = 1) -> List[int]:
    out = [max(int(min_len), int(x)) for x in lengths]
    if not out:
        return [int(total)]
    cur = int(sum(out))
    idx = 0
    while cur < int(total):
        out[idx % len(out)] += 1
        cur += 1
        idx += 1
    idx = 0
    while cur > int(total):
        pos = idx % len(out)
        if out[pos] > int(min_len):
            out[pos] -= 1
            cur -= 1
        idx += 1
        if idx > 10000:
            break
    return out


def _block_spans_from_lengths(lengths: Sequence[int]) -> Tuple[Span, ...]:
    spans: List[Span] = []
    start = 0
    for length in lengths:
        end = int(start + int(length))
        spans.append((int(start), int(end)))
        start = end
    return tuple(spans)


def _variable_partition_lengths(
    total_blocks: int,
    *,
    target_blocks: int,
    profile: str,
    rng: np.random.Generator,
) -> List[int]:
    prof = str(profile).strip().lower()
    if prof == "equal":
        n_sections = max(1, int(round(float(total_blocks) / float(max(1, target_blocks)))))
        base = [int(target_blocks)] * int(n_sections)
        return _fit_lengths_to_total(base, int(total_blocks), min_len=1)

    if prof == "bimodal":
        n_sections = max(2, int(round(float(total_blocks) / float(max(1, target_blocks)))))
        small = max(1, int(math.floor(0.5 * float(target_blocks))))
        large = max(small + 1, int(math.ceil(1.5 * float(target_blocks))))
        first_large = bool(rng.integers(0, 2))
        base: List[int] = []
        for idx in range(n_sections):
            use_large = bool((idx % 2 == 0) == first_large)
            base.append(large if use_large else small)
        return _fit_lengths_to_total(base, int(total_blocks), min_len=1)

    if prof == "long_tail":
        n_sections = max(3, int(round(float(total_blocks) / float(max(1, target_blocks)))))
        small = max(1, int(math.floor(0.5 * float(target_blocks))))
        while (n_sections - 1) * small >= int(total_blocks):
            n_sections = max(2, n_sections - 1)
        big = int(total_blocks) - (n_sections - 1) * int(small)
        base = [int(small)] * int(n_sections - 1) + [int(big)]
        pos = int(rng.integers(0, len(base)))
        rotated = base[pos:] + base[:pos]
        return _fit_lengths_to_total(rotated, int(total_blocks), min_len=1)

    raise ValueError(f"unsupported latent_length_profile: {profile!r}")


def _latent_section_block_spans(
    *,
    doc_tokens: int,
    atomic_block_tokens: int,
    latent_leaf_tokens: int,
    latent_partition_mode: str,
    latent_length_profile: str,
    rng: np.random.Generator,
) -> Tuple[Span, ...]:
    total_blocks = int(doc_tokens) // int(atomic_block_tokens)
    target_blocks = max(1, int(latent_leaf_tokens) // int(atomic_block_tokens))
    if str(latent_partition_mode).strip().lower() == "equal":
        if int(doc_tokens) % int(latent_leaf_tokens) != 0:
            raise ValueError("doc_tokens must be divisible by latent_leaf_tokens in equal mode")
        return _block_spans_from_lengths(
            [int(target_blocks)] * (int(total_blocks) // int(target_blocks))
        )
    lengths = _variable_partition_lengths(
        int(total_blocks),
        target_blocks=int(target_blocks),
        profile=str(latent_length_profile),
        rng=rng,
    )
    return _block_spans_from_lengths(lengths)


def _token_spans_from_block_spans(
    block_spans: Sequence[Span], *, atomic_block_tokens: int
) -> Tuple[Span, ...]:
    return tuple(
        (int(lo) * int(atomic_block_tokens), int(hi) * int(atomic_block_tokens))
        for lo, hi in block_spans
    )


def _sample_leaf_local_mixture_docs(
    n_docs: int,
    *,
    topics_phi: Sequence[np.ndarray],
    doc_tokens: int,
    latent_leaf_tokens: int,
    doc_topic_concentration: float,
    local_mixture_concentration: float,
    seed: int,
    atomic_block_tokens: int = 16,
    latent_partition_mode: str = "equal",
    latent_length_profile: str = "equal",
) -> Tuple[Tuple[LeafLocalMixtureDoc, ...], Dict[str, float]]:
    rng = np.random.default_rng(int(seed))
    k = int(len(topics_phi))
    v = int(np.asarray(topics_phi[0], dtype=np.float64).size)

    docs: List[LeafLocalMixtureDoc] = []
    global_entropy: List[float] = []
    local_dispersion: List[float] = []
    n_sections_list: List[float] = []
    section_tokens_list: List[float] = []
    for _ in range(int(n_docs)):
        pi_doc = rng.dirichlet(np.full((k,), float(doc_topic_concentration), dtype=np.float64))
        block_spans = _latent_section_block_spans(
            doc_tokens=int(doc_tokens),
            atomic_block_tokens=int(atomic_block_tokens),
            latent_leaf_tokens=int(latent_leaf_tokens),
            latent_partition_mode=str(latent_partition_mode),
            latent_length_profile=str(latent_length_profile),
            rng=rng,
        )
        token_spans = _token_spans_from_block_spans(
            block_spans, atomic_block_tokens=int(atomic_block_tokens)
        )
        tokens = np.zeros((int(doc_tokens),), dtype=np.int64)
        topics = np.zeros((int(doc_tokens),), dtype=np.int64)
        local_pis: List[Tuple[float, ...]] = []
        for span in token_spans:
            lo, hi = int(span[0]), int(span[1])
            n_tok = int(hi - lo)
            alpha = np.clip(float(local_mixture_concentration) * pi_doc, 1e-9, None)
            pi_leaf = rng.dirichlet(alpha.astype(np.float64, copy=False))
            z = rng.choice(k, size=int(n_tok), replace=True, p=pi_leaf).astype(np.int64, copy=False)
            x = np.zeros((int(n_tok),), dtype=np.int64)
            for topic_id in range(k):
                idx = np.flatnonzero(z == int(topic_id))
                if idx.size == 0:
                    continue
                x[idx] = rng.choice(
                    v,
                    size=int(idx.size),
                    replace=True,
                    p=np.asarray(topics_phi[topic_id], dtype=np.float64),
                )
            tokens[lo:hi] = x
            topics[lo:hi] = z
            local_pis.append(tuple(float(t) for t in pi_leaf.tolist()))
            section_tokens_list.append(float(n_tok))
        pi_local = np.asarray(local_pis, dtype=np.float64)
        centered = pi_local - np.mean(pi_local, axis=0, keepdims=True)
        local_dispersion.append(float(np.mean(np.sum(centered**2, axis=1))))
        global_entropy.append(float(-np.sum(pi_doc * np.log(np.clip(pi_doc, 1e-12, 1.0)))))
        n_sections_list.append(float(len(token_spans)))
        docs.append(
            LeafLocalMixtureDoc(
                tokens=tuple(int(t) for t in tokens.tolist()),
                topics=tuple(int(t) for t in topics.tolist()),
                global_topic_weights=tuple(float(t) for t in pi_doc.tolist()),
                local_topic_weights=tuple(local_pis),
                latent_section_spans=tuple((int(lo), int(hi)) for lo, hi in token_spans),
                latent_section_block_spans=tuple((int(lo), int(hi)) for lo, hi in block_spans),
                atomic_block_tokens=int(atomic_block_tokens),
            )
        )
    stats = {
        "mean_global_topic_entropy": _base._safe_stat(global_entropy, kind="mean"),
        "mean_local_mixture_dispersion": _base._safe_stat(local_dispersion, kind="mean"),
        "mean_tokens": float(doc_tokens),
        "mean_base_leaves": _base._safe_stat(n_sections_list, kind="mean"),
        "mean_latent_sections": _base._safe_stat(n_sections_list, kind="mean"),
        "mean_latent_section_tokens": _base._safe_stat(section_tokens_list, kind="mean"),
    }
    return tuple(docs), stats


def _leaf_additive_utility(
    pi: np.ndarray,
    *,
    theta: np.ndarray,
    W_base: np.ndarray,
    lambda_multiplier: float,
) -> float:
    return _base._utility_from_pi(
        np.asarray(pi, dtype=np.float64),
        theta=np.asarray(theta, dtype=np.float64),
        W_base=np.asarray(W_base, dtype=np.float64),
        lambda_multiplier=float(lambda_multiplier),
    )


def _section_lengths_from_spans(spans: Sequence[Span]) -> np.ndarray:
    return np.asarray([float(int(hi) - int(lo)) for lo, hi in spans], dtype=np.float64)


def _base_leaf_utilities(
    doc: LeafLocalMixtureDoc,
    *,
    theta: np.ndarray,
    W_base: np.ndarray,
    lambda_multiplier: float,
    latent_leaf_tokens: int,
) -> np.ndarray:
    del latent_leaf_tokens  # kept for backward-compatible test signatures
    local = np.asarray(doc.local_topic_weights, dtype=np.float64)
    lengths = _section_lengths_from_spans(doc.latent_section_spans)
    per_leaf = np.asarray(
        [
            float(lengths[idx])
            * _leaf_additive_utility(
                row,
                theta=theta,
                W_base=W_base,
                lambda_multiplier=lambda_multiplier,
            )
            for idx, row in enumerate(local)
        ],
        dtype=np.float64,
    )
    return per_leaf


def _true_span_contribution(
    doc: LeafLocalMixtureDoc,
    span: Span,
    *,
    theta: np.ndarray,
    W_base: np.ndarray,
    lambda_multiplier: float,
) -> float:
    total = 0.0
    for latent_span, pi_leaf in zip(doc.latent_section_spans, doc.local_topic_weights):
        lo = max(int(span[0]), int(latent_span[0]))
        hi = min(int(span[1]), int(latent_span[1]))
        if hi <= lo:
            continue
        total += float(hi - lo) * _leaf_additive_utility(
            np.asarray(pi_leaf, dtype=np.float64),
            theta=theta,
            W_base=W_base,
            lambda_multiplier=lambda_multiplier,
        )
    return float(total)


def _equal_token_spans(*, doc_tokens: int, span_tokens: int) -> Tuple[Span, ...]:
    if int(doc_tokens) % int(span_tokens) != 0:
        raise ValueError("span_tokens must divide doc_tokens")
    spans: List[Span] = []
    start = 0
    while start < int(doc_tokens):
        end = int(start + int(span_tokens))
        spans.append((int(start), int(end)))
        start = end
    return tuple(spans)


def _aggregate_partition_spans(
    doc: LeafLocalMixtureDoc,
    spans: Sequence[Span],
    *,
    utility_matrix: np.ndarray,
    theta: np.ndarray,
    W_base: np.ndarray,
    lambda_multiplier: float,
) -> Tuple[Tuple[np.ndarray, ...], Tuple[float, ...], Tuple[float, ...]]:
    out_u: List[np.ndarray] = []
    out_y: List[float] = []
    out_n: List[float] = []
    tokens = np.asarray(doc.tokens, dtype=np.int64)
    for lo, hi in spans:
        counts = _base._counts_from_tokens(
            tokens[int(lo) : int(hi)], vocab_size=int(utility_matrix.shape[1])
        )
        out_u.append(utility_vector_from_counts(counts, utility_matrix))
        out_y.append(
            _true_span_contribution(
                doc,
                (int(lo), int(hi)),
                theta=theta,
                W_base=W_base,
                lambda_multiplier=lambda_multiplier,
            )
        )
        out_n.append(float(int(hi) - int(lo)))
    return tuple(out_u), tuple(float(x) for x in out_y), tuple(float(x) for x in out_n)


def _ridge_feature_map(u: np.ndarray, *, n_tokens: float) -> np.ndarray:
    arr = np.asarray(u, dtype=np.float64).reshape(-1)
    scale = float(max(1.0, n_tokens))
    base = arr / scale
    feats: List[float] = [1.0]
    feats.extend(float(x) for x in base.tolist())
    for i in range(int(base.size)):
        for j in range(i, int(base.size)):
            feats.append(float(base[i] * base[j]))
    return np.asarray(feats, dtype=np.float64)


def _analysis_section_supervision_row(
    *,
    split: str,
    doc_id: str,
    section_id: str,
    features: np.ndarray,
    target_density: float,
    n_tokens: float,
    sampling: SamplingMetadata,
    training_variant: str,
    metadata: Optional[Mapping[str, object]] = None,
) -> DenseSupervisionExample:
    row_metadata = {
        "dgp": "leaf_local_mixture_utility",
        "input_view": "analysis_section_utility_features",
        "training_variant": str(training_variant),
        "n_tokens": float(n_tokens),
        **dict(metadata or {}),
    }
    return DenseSupervisionExample(
        example_id=f"{doc_id}:{section_id}:{training_variant}",
        features=[float(value) for value in np.asarray(features, dtype=np.float64).reshape(-1)],
        scalar_target=float(target_density),
        original_text=f"leaf_local_mixture_utility::{doc_id}",
        rubric="Predict scalar analysis-section utility density from dense section features.",
        response="analysis_section_candidate",
        response_id=f"{doc_id}:{section_id}",
        unit_kind=ObservationUnitKind.LEAF,
        reference_score=0.0,
        source_doc_id=str(doc_id),
        truth_label_source="oracle",
        sampling=sampling,
        metadata=row_metadata,
    )


def _analysis_section_supervision_dataset(
    rows: Sequence[DenseSupervisionExample],
    *,
    split: str,
    training_variant: str,
) -> "SupervisionDataset":
    return build_dense_sampled_substructure_supervision_dataset(
        rows,
        application_name="leaf_local_mixture_utility",
        supervision_signal_name="substructure_level_target",
        response_signal_name="analysis_section_utility_density",
        law_type="section_utility_target",
        split=str(split),
        metadata={
            "dgp": "leaf_local_mixture_utility",
            "input_view": "analysis_section_utility_features",
            "training_variant": str(training_variant),
            "target_kind": "scalar",
        },
    )


def _l1(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(u, dtype=np.float64) - np.asarray(v, dtype=np.float64))))


from treepo._research.ctreepo.sim.util import normalize_simplex_vec, normalize_simplex_rows

_normalize_simplex_vec = normalize_simplex_vec
_normalize_simplex_rows = normalize_simplex_rows


def _infer_analysis_section_topics(
    doc: LeafLocalMixtureDoc,
    view: AnalysisPartitionView,
    *,
    world: LeafLocalMixtureUtilityWorld,
    config: LeafLocalMixtureUtilityConfig,
) -> np.ndarray:
    if not view.analysis_section_spans:
        return np.zeros((0, int(config.n_topics)), dtype=np.float64)
    rows = [
        _infer_section_topic_from_span(doc, span, world=world, config=config)
        for span in view.analysis_section_spans
    ]
    return _normalize_simplex_rows(np.asarray(rows, dtype=np.float64))


def _analysis_target_from_topic_rows(
    topic_rows: np.ndarray,
    *,
    weights: Sequence[float],
    theta: np.ndarray,
    W_base: np.ndarray,
    lambda_multiplier: float,
    doc_tokens: int,
    weighted: bool,
) -> float:
    rows = np.asarray(topic_rows, dtype=np.float64)
    if rows.size == 0:
        return 0.0
    per_section = np.asarray(
        [
            _leaf_additive_utility(
                row,
                theta=theta,
                W_base=W_base,
                lambda_multiplier=lambda_multiplier,
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    if weighted:
        wt = np.asarray(weights, dtype=np.float64).reshape(-1)
    else:
        wt = np.full((per_section.size,), 1.0 / float(max(1, per_section.size)), dtype=np.float64)
    return float(doc_tokens) * float(np.sum(wt * per_section))


def _fit_affine_summary_calibrator(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    x_arr = _normalize_simplex_rows(np.asarray(x, dtype=np.float64))
    y_arr = _normalize_simplex_rows(np.asarray(y, dtype=np.float64))
    if x_arr.ndim != 2 or y_arr.ndim != 2 or x_arr.shape != y_arr.shape or x_arr.shape[0] == 0:
        dim = (
            int(y_arr.shape[1])
            if y_arr.ndim == 2
            else int(x_arr.shape[1]) if x_arr.ndim == 2 else 1
        )
        return {
            "w": np.eye(dim, dtype=np.float64),
            "b": np.zeros((dim,), dtype=np.float64),
        }
    features = np.concatenate(
        [x_arr, np.ones((int(x_arr.shape[0]), 1), dtype=np.float64)],
        axis=1,
    )
    if sample_weight is None:
        weights = np.ones((int(x_arr.shape[0]),), dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if weights.size != x_arr.shape[0]:
            raise ValueError("sample_weight has wrong shape for affine summary calibrator")
    sqrt_w = np.sqrt(np.clip(weights, 0.0, None))
    feat_w = features * sqrt_w[:, None]
    y_w = y_arr * sqrt_w[:, None]
    gram = feat_w.T @ feat_w
    beta = np.linalg.solve(
        gram + float(max(0.0, ridge)) * np.eye(int(gram.shape[0]), dtype=np.float64),
        feat_w.T @ y_w,
    )
    return {
        "w": np.asarray(beta[:-1, :], dtype=np.float64),
        "b": np.asarray(beta[-1, :], dtype=np.float64),
    }


def _apply_affine_summary_calibrator(x: np.ndarray, *, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        out = np.asarray(arr, dtype=np.float64) @ np.asarray(w, dtype=np.float64) + np.asarray(
            b, dtype=np.float64
        )
        return _normalize_simplex_vec(out)
    out = (
        np.asarray(arr, dtype=np.float64) @ np.asarray(w, dtype=np.float64)
        + np.asarray(b, dtype=np.float64)[None, :]
    )
    return _normalize_simplex_rows(out)


def _apply_calibrator(policy: Dict[str, object], x: np.ndarray) -> np.ndarray:
    kind = str(policy.get("kind", "identity"))
    if kind == "oracle":
        raise ValueError("oracle policies must pass truth summaries directly")
    if kind == "identity":
        return _normalize_simplex_rows(np.asarray(x, dtype=np.float64))
    if kind == "affine":
        return _apply_affine_summary_calibrator(
            np.asarray(x, dtype=np.float64),
            w=np.asarray(policy.get("w"), dtype=np.float64),
            b=np.asarray(policy.get("b"), dtype=np.float64),
        )
    raise ValueError(f"unsupported calibrator policy kind: {kind!r}")


def _lda_artifact_dir(config: LeafLocalMixtureUtilityConfig) -> Optional[Path]:
    text = str(config.artifact_dir).strip()
    if not text:
        return None
    path = Path(text)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _samples_to_logged_observations(
    samples: Sequence[TreeSample],
    *,
    supervision_signal_name: str,
) -> List[LoggedLabelObservation[float]]:
    signal_to_law_kind = {
        "c1": LawKind.L1_LEAF,
        "c2_proxy": LawKind.L3_IDEMPOTENCE,
        "c3": LawKind.L2_MERGE,
    }
    observations: List[LoggedLabelObservation[float]] = []
    for sample in samples:
        observations.append(
            shared_logged_substructure_observation(
                document_id=str(sample.doc_id),
                unit_id=str(sample.node_id),
                unit_kind=sample.sampling.unit_kind or (
                    ObservationUnitKind.LEAF
                    if sample.node_type == NodeType.LEAF
                    else ObservationUnitKind.MERGE
                ),
                label=float(sample.preference_loss),
                application_name="tree_relevant_lda_local_law",
                supervision_signal_name=supervision_signal_name,
                law_kind=signal_to_law_kind.get(supervision_signal_name),
                sampling=sample.sampling,
                context=dict(sample.metadata or {}),
            )
        )
    return observations


def _lda_split_id(*, split: str, seed: int, n_docs: int) -> str:
    return f"tree_relevant_lda:{str(split)}:seed={int(seed)}:docs={int(n_docs)}"


def _local_law_metrics_from_summary(metrics: Mapping[str, object]) -> LocalLawMetrics:
    return LocalLawMetrics(
        c1=float(metrics.get("mean_c1", float("nan"))),
        c2=float(metrics.get("mean_c2_proxy", float("nan"))),
        c3=float(metrics.get("mean_c3", float("nan"))),
        combined=float(metrics.get("combined_law_score", float("nan"))),
        root_error=float(metrics.get("mean_root_c3_error", float("nan"))),
        c1_violation_rate=float(metrics.get("c1_violation_rate", float("nan"))),
        c2_violation_rate=float(metrics.get("c2_proxy_violation_rate", float("nan"))),
        c3_violation_rate=float(metrics.get("c3_violation_rate", float("nan"))),
    )


def _downstream_metrics_from_summary(metrics: Mapping[str, object]) -> DownstreamMetrics:
    return DownstreamMetrics(
        oracle_target_abs_error=float(
            metrics.get("mean_aux_oracle_target_abs_error", float("nan"))
        ),
        oracle_target_delta=float(metrics.get("mean_aux_oracle_target_delta", float("nan"))),
        root_error=float(metrics.get("mean_root_c3_error", float("nan"))),
    )


def _objective_metrics_from_summary(
    metrics: Mapping[str, object],
    *,
    objective_spec: CompositeObjectiveSpec,
    config: LeafLocalMixtureUtilityConfig,
) -> Dict[str, object]:
    objective_name = str(objective_spec.name or objective_spec.selection_metric_name)
    objective_eval = evaluate_composite_objective_from_metrics(
        objective_spec,
        metrics=metrics,
    )
    local_law_weight_total = float(sum(float(v) for v in objective_spec.local_law_weights.values()))
    auxiliary_weight_total = float(sum(float(v) for v in objective_spec.proxy_weights.values()))
    total_weight_without_proxy = float(objective_spec.total_weight_without_proxy())
    local_law_weight = float(objective_spec.local_law_weight())
    local_law_objective_term = float(sum(float(v) for v in objective_eval.local_law_terms.values()))
    local_law_objective_value = (
        float(local_law_objective_term / local_law_weight)
        if local_law_weight > 0.0
        else 0.0
    )
    proxy_objective_value = float(sum(float(v) for v in objective_eval.proxy_raw.values()))
    proxy_objective_term = float(sum(float(v) for v in objective_eval.proxy_terms.values()))
    normalized_task_share = float(objective_spec.normalized_task_share())
    normalized_local_law_share = float(local_law_weight)
    out = {
        "objective_name": objective_name,
        "selection_metric_name": objective_name,
        "weighting_scheme": str(objective_spec.weighting_scheme),
        "root_metric_name": str(
            dict(objective_spec.metadata).get("root_metric_name", objective_spec.root_metric_name)
        ),
        "local_law_weight_total": local_law_weight_total,
        "auxiliary_diagnostic_weight_total": auxiliary_weight_total,
        "total_weight_without_proxy": total_weight_without_proxy,
        "root_share": normalized_task_share,
        "local_law_weight": normalized_local_law_share,
        "local_law_component_weights": objective_spec.normalized_local_law_weights(),
        "quadratic_utility_weight": float(config.lambda_multiplier),
        "model_quadratic_utility_weight": float(config.lambda_multiplier),
        "full_objective_value": float(objective_eval.total),
        "task_objective_value": float(objective_eval.task_raw),
        "task_objective_term": float(objective_eval.task_term),
        "regular_objective_value": float(objective_eval.task_raw),
        "regular_objective_term": float(objective_eval.task_term),
        "local_law_objective_value": local_law_objective_value,
        "local_law_objective_term": local_law_objective_term,
        "proxy_objective_value": proxy_objective_value,
        "proxy_objective_term": proxy_objective_term,
        "auxiliary_diagnostic_weights": {
            str(name): float(value) for name, value in dict(objective_spec.proxy_weights).items()
        },
    }
    estimator_payload = dict(metrics.get("objective_estimator_payload", {}) or {})
    if estimator_payload:
        out["selection_metric_name"] = str(
            estimator_payload.get("selection_metric_name", out["selection_metric_name"])
        )
        for key in (
            "selection_estimator",
            "selection_metric_value",
            "available_estimators",
            "estimator_components",
        ):
            if key in estimator_payload:
                out[str(key)] = estimator_payload[key]
        for estimator in OBJECTIVE_ESTIMATOR_KEYS:
            alias = objective_estimator_alias(objective_name, estimator)
            for key in (
                str(alias),
                f"{alias}_task_objective_value",
                f"{alias}_task_objective_term",
                f"{alias}_local_law_objective_value",
                f"{alias}_local_law_objective_term",
                f"{alias}_proxy_objective_value",
                f"{alias}_proxy_objective_term",
                f"{alias}_root_share",
                f"{alias}_local_law_weight",
            ):
                if key in estimator_payload:
                    out[str(key)] = estimator_payload[key]
        width_key = f"{objective_name}_eb_width"
        if width_key in estimator_payload:
            out[width_key] = estimator_payload[width_key]
        selection_value_key = f"{objective_name}_selection_value"
        if selection_value_key in estimator_payload:
            out[selection_value_key] = estimator_payload[selection_value_key]
    return out


def _constant_estimator_map(value: float) -> Dict[str, float]:
    return {str(name): float(value) for name in OBJECTIVE_ESTIMATOR_KEYS}


def _augment_summary_metrics_with_objective_estimators(
    metrics: Mapping[str, object],
    *,
    objective_spec: CompositeObjectiveSpec,
    ipw_policy_eval: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    out = dict(metrics)
    task_metric_name = str(
        dict(objective_spec.metadata).get("root_metric_name", objective_spec.root_metric_name)
    )
    task_value = float(out.get(task_metric_name, float("nan")))
    task_estimates = _constant_estimator_map(task_value)

    local_law_metric_names = dict(
        dict(objective_spec.metadata).get("local_law_metric_names", {}) or {}
    )
    ipw_payload = dict(ipw_policy_eval or {})
    local_law_estimates: Dict[str, Dict[str, float]] = {}
    for law_name in dict(objective_spec.local_law_weights).keys():
        raw_metric_name = str(local_law_metric_names.get(str(law_name), str(law_name)))
        exact_value = float(out.get(raw_metric_name, float("nan")))
        law_estimates = {"exact": float(exact_value)}
        law_eval = dict(ipw_payload.get(str(law_name), {}) or {})
        if law_eval:
            law_estimates["ht"] = float(law_eval.get("ht_mean", float("nan")))
            law_estimates["hajek"] = float(law_eval.get("hajek", float("nan")))
            law_estimates["eb_lo"] = float(law_eval.get("eb_lo", float("nan")))
            law_estimates["eb_hi"] = float(law_eval.get("eb_hi", float("nan")))
        local_law_estimates[str(law_name)] = law_estimates

    proxy_metric_names = dict(dict(objective_spec.metadata).get("proxy_metric_names", {}) or {})
    proxy_estimates: Dict[str, Dict[str, float]] = {}
    for proxy_name in dict(objective_spec.proxy_weights).keys():
        raw_metric_name = str(proxy_metric_names.get(str(proxy_name), str(proxy_name)))
        exact_value = float(out.get(raw_metric_name, float("nan")))
        proxy_estimates[str(proxy_name)] = _constant_estimator_map(exact_value)

    prefer_hajek = False
    weighted_laws = {
        str(name): float(weight)
        for name, weight in dict(objective_spec.local_law_weights).items()
        if float(weight) > 0.0
    }
    if weighted_laws:
        hajek_ready = all(
            math.isfinite(float(dict(local_law_estimates.get(name, {}) or {}).get("hajek", float("nan"))))
            for name in weighted_laws
        )
        prefer_hajek = bool(hajek_ready)

    estimator_payload = scalarize_objective_estimates(
        objective_spec,
        task_estimates=task_estimates,
        local_law_estimates=local_law_estimates,
        proxy_estimates=proxy_estimates,
        selection_preference=("hajek" if prefer_hajek else "exact"),
    )
    out["objective_estimator_payload"] = estimator_payload
    out["objective_name"] = str(estimator_payload.get("objective_name", objective_spec.name))
    out["selection_metric_name"] = str(
        estimator_payload.get("selection_metric_name", objective_spec.selection_metric_name)
    )
    out["selection_estimator"] = str(estimator_payload.get("selection_estimator", "exact"))
    out["selection_metric_value"] = float(
        estimator_payload.get("selection_metric_value", float("nan"))
    )
    out["available_objective_estimators"] = list(
        estimator_payload.get("available_estimators", [])
    )
    base_name = str(estimator_payload.get("objective_name", objective_spec.name))
    for estimator in OBJECTIVE_ESTIMATOR_KEYS:
        alias = objective_estimator_alias(base_name, estimator)
        if alias in estimator_payload:
            out[str(alias)] = estimator_payload[alias]
        for key in tuple(estimator_payload.keys()):
            if not str(key).startswith(f"{alias}_"):
                continue
            if key in estimator_payload:
                out[str(key)] = estimator_payload[key]
    width_key = f"{base_name}_eb_width"
    if width_key in estimator_payload:
        out[width_key] = estimator_payload[width_key]
    selection_value_key = f"{base_name}_selection_value"
    if selection_value_key in estimator_payload:
        out[selection_value_key] = estimator_payload[selection_value_key]
    local_law_objective_term_key = f"{base_name}_local_law_objective_term"
    if local_law_objective_term_key in estimator_payload:
        out[f"{base_name}_local_law_term_total"] = estimator_payload[
            local_law_objective_term_key
        ]
    return out


def _serialize_law_policy_artifact(
    *,
    output_dir: Optional[Path],
    artifact_id: str,
    name: str,
    role: PolicyRole,
    policy: Mapping[str, object],
    config: LeafLocalMixtureUtilityConfig,
) -> Optional[GArtifact]:
    if output_dir is None:
        return None
    payload = {
        "kind": str(policy.get("kind", "identity")),
        "policy": dict(policy),
        "n_topics": int(config.n_topics),
        "analysis_partition_mode": str(config.analysis_partition_mode),
        "law_leaf_query_design": str(config.law_leaf_query_design),
        "law_internal_query_design": str(config.law_internal_query_design),
    }
    return write_json_g_artifact(
        output_dir=output_dir,
        artifact_id=str(artifact_id),
        name=str(name),
        role=role,
        family="tree_relevant_lda_local_law",
        dgp="leaf_local_mixture_utility",
        payload=payload,
        metadata={
            "suite_role": str(config.suite_role),
            "quadratic_utility_weight": float(config.lambda_multiplier),
        },
    )


def _select_local_law_candidate(
    val_policy_metrics: Mapping[str, Mapping[str, object]],
    *,
    objective_spec: CompositeObjectiveSpec,
) -> Optional[str]:
    candidate_order = [
        "law_calibrated_ipw_stabilized",
        "law_calibrated_ipw",
        "law_calibrated_naive",
    ]
    scored: List[Tuple[float, float, int, str]] = []
    for idx, name in enumerate(candidate_order):
        metrics = dict(val_policy_metrics.get(name, {}) or {})
        if not metrics:
            continue
        preferred_metric_name = str(
            metrics.get(
                "selection_metric_name",
                objective_estimator_alias(str(objective_spec.name), "hajek"),
            )
        )
        preferred_value = float(metrics.get(preferred_metric_name, float("nan")))
        if not math.isfinite(preferred_value):
            objective_eval = evaluate_composite_objective_from_metrics(
                objective_spec,
                metrics=metrics,
            )
            preferred_value = float(objective_eval.total)
        scored.append(
            (
                float(preferred_value),
                float(metrics.get("mean_aux_oracle_target_abs_error", float("inf"))),
                int(idx),
                str(name),
            )
        )
    if not scored:
        return None
    scored.sort()
    return str(scored[0][3])


def _pooled_prediction_and_oracle_target(
    doc: LeafLocalMixtureDoc,
    *,
    world: LeafLocalMixtureUtilityWorld,
    config: LeafLocalMixtureUtilityConfig,
    theta_true: np.ndarray,
    W_base: np.ndarray,
) -> Tuple[float, float]:
    counts_full = _base._counts_from_tokens(doc.tokens, vocab_size=int(config.vocab_size))
    pi_doc_hat = _base._infer_topic_mixture_from_counts(
        counts_full,
        topics_phi=world.topics_phi,
        prior_mass=float(config.inference_prior_mass),
        max_iter=int(config.inference_max_iter),
        tol=float(config.inference_tol),
    )
    pooled_pred = float(int(config.doc_tokens)) * _leaf_additive_utility(
        pi_doc_hat,
        theta=theta_true,
        W_base=W_base,
        lambda_multiplier=float(config.lambda_multiplier),
    )
    oracle_true = _true_doc_target(
        doc,
        theta=theta_true,
        W_base=W_base,
        lambda_multiplier=float(config.lambda_multiplier),
    )
    return float(pooled_pred), float(oracle_true)


def _evaluate_local_law_split(
    docs: Sequence[LeafLocalMixtureDoc],
    *,
    split_name: str,
    split_doc_offset: int,
    learned_policies: Mapping[str, Mapping[str, object]],
    objective_spec: CompositeObjectiveSpec,
    world: LeafLocalMixtureUtilityWorld,
    config: LeafLocalMixtureUtilityConfig,
    theta_true: np.ndarray,
    W_base: np.ndarray,
) -> Dict[str, object]:
    per_policy_doc_metrics: Dict[str, List[Dict[str, object]]] = {
        "oracle_true_summary": [],
        "infer_identity": [],
    }
    for name in learned_policies:
        per_policy_doc_metrics[str(name)] = []

    law_doc_records: List[Dict[str, object]] = []
    calibrated_aux_preds: Dict[str, List[float]] = {str(name): [] for name in learned_policies}
    oracle_targets: List[float] = []
    pooled_errors: List[float] = []

    for doc_idx, doc in enumerate(docs):
        pooled_pred, oracle_true = _pooled_prediction_and_oracle_target(
            doc,
            world=world,
            config=config,
            theta_true=theta_true,
            W_base=W_base,
        )
        oracle_targets.append(float(oracle_true))
        pooled_errors.append(abs(float(pooled_pred) - float(oracle_true)))

        view = _build_analysis_partition_view(
            doc,
            config,
            doc_index=int(split_doc_offset) + int(doc_idx),
        )
        truth_rows = np.asarray(view.analysis_topic_weights, dtype=np.float64)
        hat_rows = _infer_analysis_section_topics(doc, view, world=world, config=config)
        masses = _section_lengths_from_spans(view.analysis_section_spans)
        identity_tree = _balanced_merge_internal_nodes(
            hat_rows,
            truth_rows,
            leaf_masses=masses,
        )
        leaf_scores = _section_proxy_scores(
            doc,
            view.analysis_section_spans,
            world=world,
            config=config,
        )
        internal_scores = [float(node["risk"]) for node in identity_tree["internal_nodes"]]

        oracle_metrics = _evaluate_local_law_doc_policy(
            truth_rows,
            truth_rows,
            masses=masses,
            reapply_fn=lambda x: _normalize_simplex_rows(np.asarray(x, dtype=np.float64)),
            world=world,
            config=config,
            theta_true=theta_true,
            W_base=W_base,
        )
        identity_metrics = _evaluate_local_law_doc_policy(
            hat_rows,
            truth_rows,
            masses=masses,
            reapply_fn=lambda x: _normalize_simplex_rows(np.asarray(x, dtype=np.float64)),
            world=world,
            config=config,
            theta_true=theta_true,
            W_base=W_base,
        )
        per_policy_doc_metrics["oracle_true_summary"].append(oracle_metrics)
        per_policy_doc_metrics["infer_identity"].append(identity_metrics)
        rec_policies: Dict[str, object] = {
            "oracle_true_summary": oracle_metrics,
            "infer_identity": identity_metrics,
        }

        for name, policy in learned_policies.items():
            calibrated_rows = _apply_calibrator(dict(policy), hat_rows)
            metrics = _evaluate_local_law_doc_policy(
                calibrated_rows,
                truth_rows,
                masses=masses,
                reapply_fn=lambda x, pol=dict(policy): _apply_calibrator(pol, x),
                world=world,
                config=config,
                theta_true=theta_true,
                W_base=W_base,
            )
            per_policy_doc_metrics[str(name)].append(metrics)
            rec_policies[str(name)] = metrics
            calibrated_aux_preds[str(name)].append(float(metrics["aux_oracle_target"]))

        law_doc_records.append(
            {
                "doc_idx": int(doc_idx),
                "split": str(split_name),
                "leaf_scores": [
                    float(x) for x in np.asarray(leaf_scores, dtype=np.float64).tolist()
                ],
                "internal_scores": [float(x) for x in internal_scores],
                "policies": rec_policies,
            }
        )

    # Extract infer_identity aux targets as the baseline for delta computation.
    # Delta measures improvement over the no-calibration baseline within the
    # same analysis-partition framework (not cross-framework pooled vs aux).
    _baseline_aux_targets: Optional[List[float]] = None
    _identity_docs = per_policy_doc_metrics.get("infer_identity")
    if _identity_docs:
        _baseline_aux_targets = [
            float(m.get("aux_oracle_target", float("nan"))) for m in _identity_docs
        ]

    policy_metrics: Dict[str, object] = {}
    violation_rates: Dict[str, object] = {}
    mediation: Dict[str, object] = {}
    for name, doc_metrics in per_policy_doc_metrics.items():
        metrics = _summarize_local_law_policy(
            doc_metrics,
            oracle_targets=oracle_targets,
            pooled_errors=pooled_errors,
            objective_spec=objective_spec,
            config=config,
            baseline_aux_targets=_baseline_aux_targets,
        )
        policy_metrics[str(name)] = metrics
        violation_rates[str(name)] = {
            "c1": float(metrics["c1_violation_rate"]),
            "c3": float(metrics["c3_violation_rate"]),
            "c2_proxy": float(metrics["c2_proxy_violation_rate"]),
        }
        mediation[str(name)] = {
            "combined_law_score": float(metrics["combined_law_score"]),
            "mean_aux_oracle_target_abs_error": float(metrics["mean_aux_oracle_target_abs_error"]),
            "mean_aux_oracle_target_delta": float(metrics["mean_aux_oracle_target_delta"]),
            "law_score_aux_abs_error_correlation": float(
                metrics["law_score_aux_abs_error_correlation"]
            ),
        }

    ipw_evaluation, logged_observations_by_policy = _local_law_ipw_evaluation(
        law_doc_records,
        config=config,
    )
    for name, metrics in list(policy_metrics.items()):
        policy_metrics[str(name)] = _augment_summary_metrics_with_objective_estimators(
            dict(metrics),
            objective_spec=objective_spec,
            ipw_policy_eval=dict(ipw_evaluation.get(str(name), {}) or {}),
        )

    return {
        "oracle_targets": oracle_targets,
        "pooled_errors": pooled_errors,
        "policy_metrics": policy_metrics,
        "violation_rates": violation_rates,
        "mediation": mediation,
        "ipw_evaluation": ipw_evaluation,
        "logged_observations_by_policy": logged_observations_by_policy,
        "doc_records": law_doc_records,
        "aux_preds": calibrated_aux_preds,
    }


def _expected_counts_from_summary(
    summary_row: np.ndarray,
    *,
    n_tokens: float,
    topics_phi: Sequence[np.ndarray],
) -> np.ndarray:
    pi = _normalize_simplex_vec(summary_row)
    phi = np.stack([np.asarray(t, dtype=np.float64).reshape(-1) for t in topics_phi], axis=0)
    probs = np.clip(np.asarray(pi, dtype=np.float64) @ phi, 1e-18, None)
    probs = probs / float(np.sum(probs))
    return float(max(0.0, n_tokens)) * probs


def _reinfer_expected_summary(
    summary_row: np.ndarray,
    *,
    n_tokens: float,
    world: LeafLocalMixtureUtilityWorld,
    config: LeafLocalMixtureUtilityConfig,
) -> np.ndarray:
    counts = _expected_counts_from_summary(
        np.asarray(summary_row, dtype=np.float64),
        n_tokens=float(n_tokens),
        topics_phi=world.topics_phi,
    )
    return _base._infer_topic_mixture_from_counts(
        counts,
        topics_phi=world.topics_phi,
        prior_mass=float(config.inference_prior_mass),
        max_iter=int(config.inference_max_iter),
        tol=float(config.inference_tol),
    )


def _balanced_merge_internal_nodes(
    leaf_est: np.ndarray,
    leaf_truth: np.ndarray,
    *,
    leaf_masses: np.ndarray,
) -> Dict[str, object]:
    est_rows = _normalize_simplex_rows(np.asarray(leaf_est, dtype=np.float64))
    truth_rows = _normalize_simplex_rows(np.asarray(leaf_truth, dtype=np.float64))
    masses = np.clip(np.asarray(leaf_masses, dtype=np.float64).reshape(-1), 1e-12, None)
    if est_rows.shape != truth_rows.shape:
        raise ValueError("leaf_est and leaf_truth must have the same shape")
    if est_rows.shape[0] != masses.size:
        raise ValueError("leaf_masses must align with leaf rows")

    nodes = [
        {
            "est": est_rows[idx].copy(),
            "truth": truth_rows[idx].copy(),
            "mass": float(masses[idx]),
            "span": (int(idx), int(idx + 1)),
        }
        for idx in range(int(est_rows.shape[0]))
    ]
    internals: List[Dict[str, object]] = []
    level = 0
    while len(nodes) > 1:
        next_nodes: List[Dict[str, object]] = []
        idx = 0
        pair_index = 0
        while idx < len(nodes):
            if idx + 1 >= len(nodes):
                next_nodes.append(nodes[idx])
                idx += 1
                continue
            left = nodes[idx]
            right = nodes[idx + 1]
            total_mass = float(left["mass"]) + float(right["mass"])
            est_parent = (
                float(left["mass"]) * np.asarray(left["est"], dtype=np.float64)
                + float(right["mass"]) * np.asarray(right["est"], dtype=np.float64)
            ) / float(total_mass)
            truth_parent = (
                float(left["mass"]) * np.asarray(left["truth"], dtype=np.float64)
                + float(right["mass"]) * np.asarray(right["truth"], dtype=np.float64)
            ) / float(total_mass)
            span = (int(left["span"][0]), int(right["span"][1]))
            risk = _l1(
                np.asarray(left["est"], dtype=np.float64),
                np.asarray(right["est"], dtype=np.float64),
            )
            error = _l1(est_parent, truth_parent)
            node = {
                "level": int(level),
                "pair_index": int(pair_index),
                "span": span,
                "mass": float(total_mass),
                "est": _normalize_simplex_vec(est_parent),
                "truth": _normalize_simplex_vec(truth_parent),
                "risk": float(risk),
                "error": float(error),
            }
            internals.append(node)
            next_nodes.append(
                {
                    "est": np.asarray(node["est"], dtype=np.float64).copy(),
                    "truth": np.asarray(node["truth"], dtype=np.float64).copy(),
                    "mass": float(total_mass),
                    "span": span,
                }
            )
            idx += 2
            pair_index += 1
        nodes = next_nodes
        level += 1
    if nodes:
        root_est = np.asarray(nodes[0]["est"], dtype=np.float64)
        root_truth = np.asarray(nodes[0]["truth"], dtype=np.float64)
    else:
        root_est = np.zeros((0,), dtype=np.float64)
        root_truth = np.zeros((0,), dtype=np.float64)
    return {
        "internal_nodes": internals,
        "leaf_c1": [_l1(est_rows[idx], truth_rows[idx]) for idx in range(int(est_rows.shape[0]))],
        "root_error": _l1(root_est, root_truth) if root_est.size > 0 else 0.0,
        "root_est": root_est,
        "root_truth": root_truth,
    }


def _local_law_query_probabilities(
    scores: np.ndarray,
    *,
    target_rate: float,
    design: str,
    floor: float,
    ceiling: float,
) -> np.ndarray:
    n = int(np.asarray(scores, dtype=np.float64).size)
    if n <= 0 or float(target_rate) <= 0.0:
        return np.zeros((max(0, n),), dtype=np.float64)
    return _query_probabilities(
        np.asarray(scores, dtype=np.float64),
        target_budget=float(target_rate) * float(n),
        query_design=str(design),
        floor=float(floor),
        ceiling=float(ceiling),
    )


def sample_leaf_local_mixture_utility_world(
    config: LeafLocalMixtureUtilityConfig,
    *,
    train_docs_capacity: Optional[int] = None,
    val_docs_capacity: Optional[int] = None,
    test_docs_capacity: Optional[int] = None,
) -> LeafLocalMixtureUtilityWorld:
    _validate_config(config)
    train_cap = int(config.train_docs if train_docs_capacity is None else train_docs_capacity)
    val_cap = int(config.val_docs if val_docs_capacity is None else val_docs_capacity)
    test_cap = int(config.test_docs if test_docs_capacity is None else test_docs_capacity)
    if train_cap <= 0 or val_cap < 0 or test_cap <= 0:
        raise ValueError(
            "train_docs_capacity must be positive, val_docs_capacity non-negative, and test_docs_capacity positive"
        )

    topics_phi, topic_meta = sample_topic_distributions(
        vocab_size=int(config.vocab_size),
        n_topics=int(config.n_topics),
        topic_concentration=float(config.topic_concentration),
        emission_mode=str(config.emission_mode),
        anchor_words_per_topic=int(config.anchor_words_per_topic),
        anchor_multiplier=float(config.anchor_multiplier),
        seed=int(_splitmix64(int(config.seed) + 101) & 0xFFFFFFFF),
    )
    utility_matrix, utility_topic_meta = sample_topic_anchored_utility_matrix(
        topics_phi=topics_phi,
        topic_meta=topic_meta,
        utility_dim=int(config.utility_dim),
        seed=int(_splitmix64(int(config.seed) + 303) & 0xFFFFFFFF),
    )
    relevant_topics, theta_true, W_base = sample_sparse_oracle_weights(
        n_topics=int(config.n_topics),
        relevant_topics=int(config.relevant_topics),
        theta_scale=float(config.theta_scale),
        zero_diagonal=bool(config.zero_diagonal),
        seed=int(_splitmix64(int(config.seed) + 202) & 0xFFFFFFFF),
    )
    docs_train, _ = _sample_leaf_local_mixture_docs(
        train_cap,
        topics_phi=topics_phi,
        doc_tokens=int(config.doc_tokens),
        latent_leaf_tokens=int(config.latent_leaf_tokens),
        doc_topic_concentration=float(config.doc_topic_concentration),
        local_mixture_concentration=float(config.local_mixture_concentration),
        seed=int(_splitmix64(int(config.seed) + 7) & 0xFFFFFFFF),
        atomic_block_tokens=int(config.atomic_block_tokens),
        latent_partition_mode=str(config.latent_partition_mode),
        latent_length_profile=str(config.latent_length_profile),
    )
    docs_val, _ = _sample_leaf_local_mixture_docs(
        val_cap,
        topics_phi=topics_phi,
        doc_tokens=int(config.doc_tokens),
        latent_leaf_tokens=int(config.latent_leaf_tokens),
        doc_topic_concentration=float(config.doc_topic_concentration),
        local_mixture_concentration=float(config.local_mixture_concentration),
        seed=int(_splitmix64(int(config.seed) + int(config.val_seed_offset)) & 0xFFFFFFFF),
        atomic_block_tokens=int(config.atomic_block_tokens),
        latent_partition_mode=str(config.latent_partition_mode),
        latent_length_profile=str(config.latent_length_profile),
    )
    docs_test, _ = _sample_leaf_local_mixture_docs(
        test_cap,
        topics_phi=topics_phi,
        doc_tokens=int(config.doc_tokens),
        latent_leaf_tokens=int(config.latent_leaf_tokens),
        doc_topic_concentration=float(config.doc_topic_concentration),
        local_mixture_concentration=float(config.local_mixture_concentration),
        seed=int(_splitmix64(int(config.seed) + int(config.test_seed_offset)) & 0xFFFFFFFF),
        atomic_block_tokens=int(config.atomic_block_tokens),
        latent_partition_mode=str(config.latent_partition_mode),
        latent_length_profile=str(config.latent_length_profile),
    )
    utility_topic_meta = dict(utility_topic_meta)
    utility_topic_meta["relevant_topics"] = list(int(x) for x in relevant_topics)
    return LeafLocalMixtureUtilityWorld(
        signature=_world_signature(config),
        topic_meta=dict(topic_meta),
        topics_phi=tuple(np.asarray(t, dtype=np.float64).copy() for t in topics_phi),
        utility_matrix=np.asarray(utility_matrix, dtype=np.float64).copy(),
        utility_topic_meta=utility_topic_meta,
        theta_true=np.asarray(theta_true, dtype=np.float64).copy(),
        W_base=np.asarray(W_base, dtype=np.float64).copy(),
        docs_train=tuple(docs_train),
        docs_val=tuple(docs_val),
        docs_test=tuple(docs_test),
    )


def _method_metric_dict(
    *,
    supervision_kind: str,
    budget_regime: str,
    n_docs: int,
    utility_abs_to_true: Sequence[float],
    queried_leaves_per_doc: Sequence[float],
    queried_cost_per_doc: Sequence[float],
    pooled_abs_to_true: Optional[Sequence[float]] = None,
    diagnostics: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    out = {
        "supervision_kind": str(supervision_kind),
        "budget_regime": str(budget_regime),
        "n_docs": int(n_docs),
        "utility_abs_to_true_mean": _base._safe_stat(utility_abs_to_true, kind="mean"),
        "utility_abs_to_true_median": _base._safe_stat(utility_abs_to_true, kind="median"),
        "utility_abs_to_true_p95": _base._safe_stat(utility_abs_to_true, kind="p95"),
        "mean_queried_leaves_train_per_doc": _base._safe_stat(queried_leaves_per_doc, kind="mean"),
        "mean_queried_cost_train_per_doc": _base._safe_stat(queried_cost_per_doc, kind="mean"),
    }
    if pooled_abs_to_true is not None:
        delta = np.asarray(pooled_abs_to_true, dtype=np.float64) - np.asarray(
            utility_abs_to_true, dtype=np.float64
        )
        out["delta_mean"] = _base._safe_stat(delta.tolist(), kind="mean")
        out["delta_median"] = _base._safe_stat(delta.tolist(), kind="median")
        out["delta_p95"] = _base._safe_stat(delta.tolist(), kind="p95")
        out["wins_vs_pooled_rate"] = (
            float(np.mean((delta > 0.0).astype(np.float64))) if delta.size > 0 else float("nan")
        )
    if diagnostics:
        out.update({str(k): v for k, v in diagnostics.items()})
    return out


def _pooled_true_target(
    doc: LeafLocalMixtureDoc,
    *,
    theta: np.ndarray,
    W_base: np.ndarray,
    lambda_multiplier: float,
    doc_tokens: int,
) -> float:
    local = np.asarray(doc.local_topic_weights, dtype=np.float64)
    weights = _section_lengths_from_spans(doc.latent_section_spans) / float(max(1, int(doc_tokens)))
    mean_local = np.sum(weights[:, None] * local, axis=0)
    return float(int(doc_tokens)) * _leaf_additive_utility(
        mean_local,
        theta=theta,
        W_base=W_base,
        lambda_multiplier=lambda_multiplier,
    )


def _true_doc_target(
    doc: LeafLocalMixtureDoc,
    *,
    theta: np.ndarray,
    W_base: np.ndarray,
    lambda_multiplier: float,
) -> float:
    """Closed-form leaf-local-mixture target.

    Thin re-export for back-compat: the canonical implementation lives in
    :mod:`src.ctreepo.oracles.lda` and is registered as
    ``oracle:leaf_local_mixture_target`` in the unified ladder registry.
    Internal call sites still import from this module by name; the registry
    is the single source of truth.
    """
    from treepo._research.ctreepo.oracles.lda import leaf_local_mixture_target

    return leaf_local_mixture_target(
        doc,
        theta=theta,
        W_base=W_base,
        lambda_multiplier=lambda_multiplier,
    )


def _merge_pairs(spans: Sequence[Span]) -> Tuple[Span, ...]:
    out: List[Span] = []
    idx = 0
    while idx < len(spans):
        lo = int(spans[idx][0])
        hi = int(spans[min(idx + 1, len(spans) - 1)][1])
        out.append((lo, hi))
        idx += 2
    return tuple(out)


def _split_span_in_half(span: Span) -> Tuple[Span, ...]:
    lo, hi = int(span[0]), int(span[1])
    width = int(hi - lo)
    if width <= 1:
        return ((lo, hi),)
    mid = int(lo + width // 2)
    if mid <= lo or mid >= hi:
        return ((lo, hi),)
    return ((lo, mid), (mid, hi))


def _random_same_count_block_spans(
    total_blocks: int, n_sections: int, *, rng: np.random.Generator
) -> Tuple[Span, ...]:
    if n_sections <= 1:
        return ((0, int(total_blocks)),)
    n_sections = int(max(1, min(int(total_blocks), int(n_sections))))
    cut_pool = np.arange(1, int(total_blocks), dtype=np.int64)
    if n_sections - 1 > cut_pool.size:
        n_sections = int(cut_pool.size + 1)
    cuts = sorted(
        int(x) for x in rng.choice(cut_pool, size=int(n_sections - 1), replace=False).tolist()
    )
    points = [0] + cuts + [int(total_blocks)]
    return tuple((int(points[i]), int(points[i + 1])) for i in range(len(points) - 1))


def _shift_half_block_spans(
    total_blocks: int,
    *,
    section_blocks: int,
) -> Tuple[Span, ...]:
    sec = max(1, int(section_blocks))
    shift = max(1, int(sec // 2))
    boundaries = [0]
    if shift < int(total_blocks):
        boundaries.append(int(shift))
    cur = int(shift)
    while cur + int(sec) < int(total_blocks):
        cur += int(sec)
        boundaries.append(int(cur))
    if boundaries[-1] != int(total_blocks):
        boundaries.append(int(total_blocks))
    out: List[Span] = []
    for idx in range(len(boundaries) - 1):
        lo = int(boundaries[idx])
        hi = int(boundaries[idx + 1])
        if hi > lo:
            out.append((lo, hi))
    if out and out[-1][1] < int(total_blocks):
        out.append((int(out[-1][1]), int(total_blocks)))
    elif not out:
        out.append((0, int(total_blocks)))
    return tuple(out)


def _build_analysis_partition_view(
    doc: LeafLocalMixtureDoc,
    config: LeafLocalMixtureUtilityConfig,
    *,
    doc_index: int,
) -> AnalysisPartitionView:
    latent_blocks = tuple((int(lo), int(hi)) for lo, hi in doc.latent_section_block_spans)
    total_blocks = int(config.doc_tokens) // int(config.atomic_block_tokens)
    mode = str(config.analysis_partition_mode).strip().lower()
    rng = np.random.default_rng(
        int(_splitmix64(int(config.seed) + 5003 + int(doc_index)) & 0xFFFFFFFF)
    )
    if mode == "aligned":
        analysis_blocks = latent_blocks
    elif mode == "coarsen_2x":
        analysis_blocks = _merge_pairs(latent_blocks)
    elif mode == "refine_2x":
        pieces: List[Span] = []
        for span in latent_blocks:
            pieces.extend(_split_span_in_half(span))
        analysis_blocks = tuple(pieces)
    elif mode == "shift_half":
        section_blocks = max(
            1, int(_resolve_analysis_leaf_tokens(config)) // int(config.atomic_block_tokens)
        )
        analysis_blocks = _shift_half_block_spans(total_blocks, section_blocks=section_blocks)
    elif mode == "random_same_count":
        analysis_blocks = _random_same_count_block_spans(total_blocks, len(latent_blocks), rng=rng)
    else:
        raise ValueError(f"unsupported analysis_partition_mode: {config.analysis_partition_mode!r}")

    analysis_spans = _token_spans_from_block_spans(
        analysis_blocks, atomic_block_tokens=int(config.atomic_block_tokens)
    )
    latent_spans = tuple((int(lo), int(hi)) for lo, hi in doc.latent_section_spans)
    overlap = np.zeros((len(analysis_spans), len(latent_spans)), dtype=np.float64)
    for a_idx, a_span in enumerate(analysis_spans):
        for l_idx, l_span in enumerate(latent_spans):
            lo = max(int(a_span[0]), int(l_span[0]))
            hi = min(int(a_span[1]), int(l_span[1]))
            if hi > lo:
                overlap[a_idx, l_idx] = float(hi - lo)
    analysis_lengths = np.sum(overlap, axis=1)
    latent_lengths = np.sum(overlap, axis=0)
    local = np.asarray(doc.local_topic_weights, dtype=np.float64)
    analysis_pis: List[Tuple[float, ...]] = []
    for row in overlap:
        mass = float(np.sum(row))
        if mass <= 0.0:
            pi = np.full((local.shape[1],), 1.0 / float(local.shape[1]), dtype=np.float64)
        else:
            pi = (row / mass) @ local
        analysis_pis.append(tuple(float(x) for x in pi.tolist()))
    row_norm = np.divide(
        overlap,
        np.clip(analysis_lengths[:, None], 1e-12, None),
        out=np.zeros_like(overlap),
        where=analysis_lengths[:, None] > 0.0,
    )
    col_norm = np.divide(
        overlap,
        np.clip(latent_lengths[None, :], 1e-12, None),
        out=np.zeros_like(overlap),
        where=latent_lengths[None, :] > 0.0,
    )
    return AnalysisPartitionView(
        analysis_section_spans=tuple((int(lo), int(hi)) for lo, hi in analysis_spans),
        analysis_section_block_spans=tuple((int(lo), int(hi)) for lo, hi in analysis_blocks),
        overlap_tokens=overlap,
        latent_weights=tuple(float(x) / float(config.doc_tokens) for x in latent_lengths.tolist()),
        analysis_weights=tuple(
            float(x) / float(config.doc_tokens) for x in analysis_lengths.tolist()
        ),
        analysis_topic_weights=tuple(analysis_pis),
        overlap_row_normalized=row_norm,
        overlap_col_normalized=col_norm,
    )


def _analysis_target(
    view: AnalysisPartitionView,
    *,
    theta: np.ndarray,
    W_base: np.ndarray,
    lambda_multiplier: float,
    doc_tokens: int,
    weighted: bool,
) -> float:
    return _analysis_target_from_topic_rows(
        np.asarray(view.analysis_topic_weights, dtype=np.float64),
        weights=view.analysis_weights,
        theta=theta,
        W_base=W_base,
        lambda_multiplier=lambda_multiplier,
        doc_tokens=doc_tokens,
        weighted=weighted,
    )


def _infer_section_topic_from_span(
    doc: LeafLocalMixtureDoc,
    span: Span,
    *,
    world: LeafLocalMixtureUtilityWorld,
    config: LeafLocalMixtureUtilityConfig,
) -> np.ndarray:
    counts = _base._counts_from_tokens(
        np.asarray(doc.tokens, dtype=np.int64)[int(span[0]) : int(span[1])],
        vocab_size=int(config.vocab_size),
    )
    return _base._infer_topic_mixture_from_counts(
        counts,
        topics_phi=world.topics_phi,
        prior_mass=float(config.inference_prior_mass),
        max_iter=int(config.inference_max_iter),
        tol=float(config.inference_tol),
    )


def _infer_analysis_target(
    doc: LeafLocalMixtureDoc,
    view: AnalysisPartitionView,
    *,
    world: LeafLocalMixtureUtilityWorld,
    config: LeafLocalMixtureUtilityConfig,
    theta: np.ndarray,
    W_base: np.ndarray,
    weighted: bool,
) -> float:
    inferred = _infer_analysis_section_topics(doc, view, world=world, config=config)
    return _analysis_target_from_topic_rows(
        inferred,
        weights=view.analysis_weights,
        theta=theta,
        W_base=W_base,
        lambda_multiplier=float(config.lambda_multiplier),
        doc_tokens=int(config.doc_tokens),
        weighted=weighted,
    )


def _legacy_query_mask(
    n_leaves: int,
    *,
    budget_regime: str,
    budget_equiv: float,
    n_base: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if int(n_leaves) <= 0:
        return np.zeros((0,), dtype=bool)
    regime = str(budget_regime).strip().lower()
    if regime == "all_leaves_labeled":
        return np.ones((int(n_leaves),), dtype=bool)
    p = min(1.0, float(budget_equiv) / float(max(1, n_base)))
    mask = rng.random(int(n_leaves)) < p
    return mask.astype(bool, copy=False)


def _section_proxy_scores(
    doc: LeafLocalMixtureDoc,
    spans: Sequence[Span],
    *,
    world: LeafLocalMixtureUtilityWorld,
    config: LeafLocalMixtureUtilityConfig,
) -> np.ndarray:
    proxy_kind = str(config.propensity_proxy).strip().lower()
    counts_full = _base._counts_from_tokens(doc.tokens, vocab_size=int(config.vocab_size))
    pi_doc_hat = _base._infer_topic_mixture_from_counts(
        counts_full,
        topics_phi=world.topics_phi,
        prior_mass=float(config.inference_prior_mass),
        max_iter=int(config.inference_max_iter),
        tol=float(config.inference_tol),
    )
    if proxy_kind != "l1_deviation":
        raise ValueError(f"unsupported propensity_proxy: {config.propensity_proxy!r}")
    scores: List[float] = []
    for span in spans:
        pi_span_hat = _infer_section_topic_from_span(doc, span, world=world, config=config)
        scores.append(float(np.sum(np.abs(np.asarray(pi_span_hat) - np.asarray(pi_doc_hat)))))
    return np.asarray(scores, dtype=np.float64)


def _query_probabilities(
    scores: np.ndarray,
    *,
    target_budget: float,
    query_design: str,
    floor: float,
    ceiling: float,
) -> np.ndarray:
    n = int(scores.size)
    if n <= 0 or float(target_budget) <= 0.0:
        return np.zeros((max(0, n),), dtype=np.float64)
    if str(query_design).strip().lower() == "uniform":
        probs = np.full((n,), float(target_budget) / float(n), dtype=np.float64)
    else:
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        if str(query_design).strip().lower() == "proxy_priority":
            base = np.clip(s, 0.0, None)
        elif str(query_design).strip().lower() == "proxy_adversarial":
            hi = float(np.max(s)) if s.size > 0 else 0.0
            base = np.clip(hi - s, 0.0, None)
        else:
            raise ValueError(f"unsupported query_design: {query_design!r}")
        if float(np.sum(base)) <= 0.0:
            base = np.ones((n,), dtype=np.float64)
        probs = float(target_budget) * base / float(np.sum(base))
    probs = np.clip(probs, float(floor), float(ceiling))
    probs = np.clip(probs, 0.0, 1.0)
    return probs.astype(np.float64, copy=False)


def _train_budgeted_analysis_ridge(
    docs_train: Sequence[LeafLocalMixtureDoc],
    *,
    world: LeafLocalMixtureUtilityWorld,
    config: LeafLocalMixtureUtilityConfig,
    theta_true: np.ndarray,
    W_base: np.ndarray,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    full_rows: List[DenseSupervisionExample] = []
    naive_rows: List[DenseSupervisionExample] = []
    ipw_rows: List[DenseSupervisionExample] = []
    stabilized_weights: List[float] = []
    queried_per_doc: List[float] = []
    mean_propensity_per_doc: List[float] = []
    max_propensity_per_doc: List[float] = []
    train_samples: List[TreeSample] = []

    train_doc_rate = float(config.doc_sample_rate)
    for doc_idx, doc in enumerate(docs_train):
        rng = np.random.default_rng(
            int(_splitmix64(int(config.seed) + 7001 + int(doc_idx)) & 0xFFFFFFFF)
        )
        if float(train_doc_rate) < 1.0 and float(rng.random()) >= float(train_doc_rate):
            queried_per_doc.append(0.0)
            mean_propensity_per_doc.append(0.0)
            max_propensity_per_doc.append(0.0)
            continue
        view = _build_analysis_partition_view(doc, config, doc_index=doc_idx)
        spans = view.analysis_section_spans
        scores = _section_proxy_scores(doc, spans, world=world, config=config)
        target_budget = float(config.target_query_budget_per_doc)
        if target_budget <= 0.0:
            if str(config.budget_regime).strip().lower() == "all_leaves_labeled":
                target_budget = float(len(spans))
            else:
                target_budget = float(config.leaf_label_budget)
        probs = _query_probabilities(
            scores,
            target_budget=target_budget,
            query_design=str(config.query_design),
            floor=float(config.propensity_floor),
            ceiling=float(config.propensity_ceiling),
        )
        keep = rng.random(len(spans)) < probs
        queried_per_doc.append(float(np.sum(keep)))
        mean_propensity_per_doc.append(float(np.mean(probs)) if probs.size > 0 else 0.0)
        max_propensity_per_doc.append(float(np.max(probs)) if probs.size > 0 else 0.0)
        for sec_idx, (span, p_sec, do_keep) in enumerate(zip(spans, probs.tolist(), keep.tolist())):
            counts = _base._counts_from_tokens(
                np.asarray(doc.tokens, dtype=np.int64)[int(span[0]) : int(span[1])],
                vocab_size=int(config.vocab_size),
            )
            u = utility_vector_from_counts(
                counts, np.asarray(world.utility_matrix, dtype=np.float64)
            )
            n_tok = float(int(span[1]) - int(span[0]))
            x = _ridge_feature_map(u, n_tokens=n_tok)
            y = _true_span_contribution(
                doc,
                span,
                theta=theta_true,
                W_base=W_base,
                lambda_multiplier=float(config.lambda_multiplier),
            ) / float(max(1.0, n_tok))
            doc_id = f"train_{doc_idx}"
            section_id = f"section_{sec_idx}"
            full_rows.append(
                _analysis_section_supervision_row(
                    split="train_full",
                    doc_id=doc_id,
                    section_id=section_id,
                    features=x,
                    target_density=float(y),
                    n_tokens=n_tok,
                    sampling=SamplingMetadata(
                        document_propensity=1.0,
                        unit_propensity=1.0,
                        label_propensity=1.0,
                        joint_propensity=1.0,
                        sampling_scheme="sampled_substructure_supervision",
                        policy_name="all_sections",
                        unit_kind=ObservationUnitKind.LEAF,
                        supports_ipw_estimation=False,
                    ),
                    training_variant="full_labels",
                    metadata={
                        "query_design": "all",
                        "doc_index": int(doc_idx),
                        "section_index": int(sec_idx),
                    },
                )
            )
            if not bool(do_keep):
                continue
            joint = max(MIN_PROPENSITY, float(train_doc_rate) * float(max(MIN_PROPENSITY, p_sec)))
            ipw_weight = 1.0 / float(joint)
            stab_weight = min(float(config.ipw_stabilized_clip), ipw_weight)
            naive_rows.append(
                _analysis_section_supervision_row(
                    split="train_budgeted",
                    doc_id=doc_id,
                    section_id=section_id,
                    features=x,
                    target_density=float(y),
                    n_tokens=n_tok,
                    sampling=SamplingMetadata(
                        document_propensity=1.0,
                        unit_propensity=1.0,
                        label_propensity=1.0,
                        joint_propensity=1.0,
                        sampling_scheme="sampled_substructure_supervision",
                        policy_name=str(config.query_design),
                        unit_kind=ObservationUnitKind.LEAF,
                        supports_ipw_estimation=False,
                        metadata={
                            "doc_sample_rate": float(train_doc_rate),
                            "section_propensity": float(p_sec),
                        },
                    ),
                    training_variant="budgeted_naive",
                    metadata={
                        "query_design": str(config.query_design),
                        "doc_index": int(doc_idx),
                        "section_index": int(sec_idx),
                    },
                )
            )
            ipw_rows.append(
                _analysis_section_supervision_row(
                    split="train_budgeted",
                    doc_id=doc_id,
                    section_id=section_id,
                    features=x,
                    target_density=float(y),
                    n_tokens=n_tok,
                    sampling=SamplingMetadata(
                        document_propensity=max(MIN_PROPENSITY, float(train_doc_rate)),
                        unit_propensity=max(MIN_PROPENSITY, float(p_sec)),
                        label_propensity=1.0,
                        joint_propensity=float(joint),
                        sampling_scheme="sampled_substructure_supervision",
                        policy_name=str(config.query_design),
                        unit_kind=ObservationUnitKind.LEAF,
                        supports_ipw_estimation=True,
                        metadata={
                            "doc_sample_rate": float(train_doc_rate),
                            "section_propensity": float(p_sec),
                        },
                    ),
                    training_variant="budgeted_ipw",
                    metadata={
                        "query_design": str(config.query_design),
                        "doc_index": int(doc_idx),
                        "section_index": int(sec_idx),
                    },
                )
            )
            stabilized_weights.append(float(stab_weight))
            train_samples.append(
                TreeSample(
                    doc_id=doc_id,
                    node_id=section_id,
                    node_type=NodeType.LEAF,
                    violation=0,
                    preference_loss=float(max(0.0, min(1.0, p_sec))),
                    sampling=SamplingMetadata(
                        document_propensity=max(MIN_PROPENSITY, float(train_doc_rate)),
                        unit_propensity=max(MIN_PROPENSITY, float(p_sec)),
                        unit_kind=ObservationUnitKind.LEAF,
                    ),
                    metadata={
                        "raw_target_density": float(y),
                        "proxy_score": (
                            float(scores[sec_idx]) if sec_idx < scores.size else float("nan")
                        ),
                    },
                )
            )

    if not full_rows:
        raise ValueError("no analysis-section training examples were constructed")
    if not naive_rows:
        first = full_rows[0]
        fallback_sampling = SamplingMetadata(
            document_propensity=1.0,
            unit_propensity=1.0,
            label_propensity=1.0,
            joint_propensity=1.0,
            sampling_scheme="sampled_substructure_supervision",
            policy_name="fallback_single_row",
            unit_kind=ObservationUnitKind.LEAF,
            supports_ipw_estimation=False,
        )
        naive_rows = [
            _analysis_section_supervision_row(
                split="train_budgeted",
                doc_id=str(first.source_doc_id or "train_fallback"),
                section_id=str(first.response_id or "section_0"),
                features=np.asarray(first.features, dtype=np.float64),
                target_density=float(first.scalar_target if first.scalar_target is not None else 0.0),
                n_tokens=float(first.metadata.get("n_tokens", 1.0)),
                sampling=fallback_sampling,
                training_variant="budgeted_naive",
                metadata=dict(first.metadata),
            )
        ]
        ipw_rows = [
            _analysis_section_supervision_row(
                split="train_budgeted",
                doc_id=str(first.source_doc_id or "train_fallback"),
                section_id=str(first.response_id or "section_0"),
                features=np.asarray(first.features, dtype=np.float64),
                target_density=float(first.scalar_target if first.scalar_target is not None else 0.0),
                n_tokens=float(first.metadata.get("n_tokens", 1.0)),
                sampling=fallback_sampling,
                training_variant="budgeted_ipw",
                metadata=dict(first.metadata),
            )
        ]
        stabilized_weights = [1.0]

    full_supervision = _analysis_section_supervision_dataset(
        full_rows,
        split="train_full",
        training_variant="full_labels",
    )
    naive_supervision = _analysis_section_supervision_dataset(
        naive_rows,
        split="train_budgeted",
        training_variant="budgeted_naive",
    )
    ipw_supervision = _analysis_section_supervision_dataset(
        ipw_rows,
        split="train_budgeted",
        training_variant="budgeted_ipw",
    )
    full_model, full_fit = fit_dense_scalar_ridge_regressor(
        full_supervision,
        config=DenseScalarRidgeTrainingConfig(
            model=DenseScalarRidgeModelConfig(ridge_alpha=float(config.ridge_alpha))
        ),
    )
    naive_model, naive_fit = fit_dense_scalar_ridge_regressor(
        naive_supervision,
        config=DenseScalarRidgeTrainingConfig(
            model=DenseScalarRidgeModelConfig(ridge_alpha=float(config.ridge_alpha))
        ),
    )
    ipw_model, ipw_fit = fit_dense_scalar_ridge_regressor(
        ipw_supervision,
        config=DenseScalarRidgeTrainingConfig(
            model=DenseScalarRidgeModelConfig(ridge_alpha=float(config.ridge_alpha))
        ),
    )
    stab_arr = np.asarray(stabilized_weights, dtype=np.float64)
    if stab_arr.size > 0 and float(np.mean(stab_arr)) > 0.0:
        stab_arr = stab_arr / float(np.mean(stab_arr))
    stab_model, stab_fit = fit_dense_scalar_ridge_regressor(
        ipw_supervision,
        config=DenseScalarRidgeTrainingConfig(
            model=DenseScalarRidgeModelConfig(ridge_alpha=float(config.ridge_alpha))
        ),
        sample_weights=stab_arr,
    )
    diagnostics = {
        "sampled_doc_rate_train": float(config.doc_sample_rate),
        "query_design": str(config.query_design),
        "target_query_budget_per_doc": float(
            config.target_query_budget_per_doc
            if float(config.target_query_budget_per_doc) > 0.0
            else config.leaf_label_budget
        ),
        "mean_sections_queried_train_per_doc": _base._safe_stat(queried_per_doc, kind="mean"),
        "mean_section_propensity_train": _base._safe_stat(mean_propensity_per_doc, kind="mean"),
        "max_section_propensity_train": _base._safe_stat(max_propensity_per_doc, kind="mean"),
        "train_sample_count": int(len(train_samples)),
        "train_effective_sample_size": float(effective_sample_size(train_samples)),
        "train_max_weight": float(max_weight(train_samples)),
        "full_labels_rows": int(full_fit.n_train_rows),
        "budgeted_rows": int(ipw_fit.n_train_rows),
        **supervision_training_contract(
            representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
            target_kind=TARGET_SCALAR,
            optimizer_family=OPTIMIZER_FAMILY_CLOSED_FORM_LINEAR,
            optimizer_backend="closed_form_ridge",
            n_train_rows=int(ipw_fit.n_train_rows),
        ),
    }
    return {
        "analysis_ridge_full_labels": full_model,
        "budgeted_leaf_ridge_naive": naive_model,
        "budgeted_leaf_ridge_ipw": ipw_model,
        "budgeted_leaf_ridge_ipw_stabilized": stab_model,
    }, diagnostics


def _predict_analysis_ridge(
    model: object,
    doc: LeafLocalMixtureDoc,
    view: AnalysisPartitionView,
    *,
    world: LeafLocalMixtureUtilityWorld,
) -> float:
    total = 0.0
    for span in view.analysis_section_spans:
        counts = _base._counts_from_tokens(
            np.asarray(doc.tokens, dtype=np.int64)[int(span[0]) : int(span[1])],
            vocab_size=int(world.utility_matrix.shape[1]),
        )
        u = utility_vector_from_counts(counts, np.asarray(world.utility_matrix, dtype=np.float64))
        n_tok = float(int(span[1]) - int(span[0]))
        pred_density = float(
            np.asarray(
                getattr(model, "predict")(
                    np.asarray([_ridge_feature_map(u, n_tokens=n_tok)], dtype=np.float64)
                ),
                dtype=np.float64,
            )[0]
        )
        total += float(n_tok) * pred_density
    return float(total)


def _normalized_ci_and_coverage(
    samples: Sequence[TreeSample],
    *,
    exact_value: float,
    raw_values_population: Sequence[float],
    population_size: float,
    delta: float,
) -> Dict[str, float]:
    raw_arr = np.asarray(raw_values_population, dtype=np.float64)
    lo = float(np.min(raw_arr)) if raw_arr.size > 0 else 0.0
    hi = float(np.max(raw_arr)) if raw_arr.size > 0 else 1.0
    scale = max(1e-12, hi - lo)
    if raw_arr.size == 0 or scale <= 1e-12:
        return {
            "value_min": float(lo),
            "value_max": float(hi),
            "scale": float(scale),
            "ht_mean": float(exact_value),
            "hajek": float(exact_value),
            "ht_abs_error": 0.0,
            "hajek_abs_error": 0.0,
            "eb_lo": float(exact_value),
            "eb_hi": float(exact_value),
            "eb_width": 0.0,
            "eb_contains_exact": 1.0,
            "effective_sample_size": float(effective_sample_size(samples)),
            "max_weight": float(max_weight(samples)),
            "sample_count": float(len(samples)),
            "weight_sum": float(sum(s.weight for s in samples)),
        }

    normalized_samples = [
        TreeSample(
            doc_id=str(sample.doc_id),
            node_id=str(sample.node_id),
            node_type=sample.node_type,
            violation=int(sample.violation),
            preference_loss=float(np.clip((float(sample.preference_loss) - lo) / scale, 0.0, 1.0)),
            sampling=sample.sampling,
            metadata=dict(sample.metadata),
        )
        for sample in samples
    ]
    comp = hajek_ht_comparison(
        normalized_samples,
        lambda s: float(s.preference_loss),
        population_size=float(population_size),
    )
    ci = empirical_bernstein_ci(
        normalized_samples,
        lambda s: float(s.preference_loss),
        float(delta),
        value_min=0.0,
        value_max=1.0,
    )
    exact_norm = (float(exact_value) - lo) / scale
    return {
        "value_min": float(lo),
        "value_max": float(hi),
        "scale": float(scale),
        "ht_mean": float(lo + scale * float(comp["ht_mean"])),
        "hajek": float(lo + scale * float(comp["hajek"])),
        "ht_abs_error": abs(float(lo + scale * float(comp["ht_mean"])) - float(exact_value)),
        "hajek_abs_error": abs(float(lo + scale * float(comp["hajek"])) - float(exact_value)),
        "eb_lo": float(lo + scale * float(ci[0])),
        "eb_hi": float(lo + scale * float(ci[1])),
        "eb_width": float(scale * max(0.0, float(ci[1]) - float(ci[0]))),
        "eb_contains_exact": (
            1.0 if float(ci[0]) - 1e-12 <= exact_norm <= float(ci[1]) + 1e-12 else 0.0
        ),
        "effective_sample_size": float(effective_sample_size(samples)),
        "max_weight": float(max_weight(samples)),
        "sample_count": float(len(samples)),
        "weight_sum": float(comp["weight_sum"]),
    }


def _propensity_quantiles(samples: Sequence[TreeSample]) -> Dict[str, float]:
    if not samples:
        return {
            "q00": float("nan"),
            "q10": float("nan"),
            "q50": float("nan"),
            "q90": float("nan"),
            "q100": float("nan"),
        }
    vals = np.asarray([float(s.joint_propensity) for s in samples], dtype=np.float64)
    return {
        "q00": float(np.quantile(vals, 0.00)),
        "q10": float(np.quantile(vals, 0.10)),
        "q50": float(np.quantile(vals, 0.50)),
        "q90": float(np.quantile(vals, 0.90)),
        "q100": float(np.quantile(vals, 1.00)),
    }


def _legacy_ridge_predictions(
    docs_train: Sequence[LeafLocalMixtureDoc],
    docs_test: Sequence[LeafLocalMixtureDoc],
    *,
    world: LeafLocalMixtureUtilityWorld,
    config: LeafLocalMixtureUtilityConfig,
    theta_true: np.ndarray,
    W_base: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    utility_matrix = np.asarray(world.utility_matrix, dtype=np.float64)
    rng_base = np.random.default_rng(int(_splitmix64(int(config.seed) + 701) & 0xFFFFFFFF))
    rng_coarse = np.random.default_rng(int(_splitmix64(int(config.seed) + 907) & 0xFFFFFFFF))
    eval_leaf_tokens = int(
        leaf_tokens_from_fraction(int(config.doc_tokens), float(config.leaf_fraction))
    )

    base_rows: List[DenseSupervisionExample] = []
    coarse_rows: List[DenseSupervisionExample] = []
    queried_leaves_base: List[float] = []
    queried_cost_base: List[float] = []
    queried_leaves_coarse: List[float] = []
    queried_cost_coarse: List[float] = []

    for doc in docs_train:
        base_u, base_y, base_n = _aggregate_partition_spans(
            doc,
            doc.latent_section_spans,
            utility_matrix=utility_matrix,
            theta=theta_true,
            W_base=W_base,
            lambda_multiplier=float(config.lambda_multiplier),
        )
        mask_base = _legacy_query_mask(
            len(base_u),
            budget_regime=str(config.budget_regime),
            budget_equiv=float(config.leaf_label_budget),
            n_base=max(1, len(base_u)),
            rng=rng_base,
        )
        queried_leaves_base.append(float(np.sum(mask_base)))
        queried_cost_base.append(float(np.sum(mask_base)))
        for leaf_idx, (u_leaf, y_leaf, n_leaf, keep) in enumerate(
            zip(base_u, base_y, base_n, mask_base)
        ):
            if not bool(keep):
                continue
            base_rows.append(
                _analysis_section_supervision_row(
                    split="train_legacy_base",
                    doc_id=f"train_{len(queried_leaves_base) - 1}",
                    section_id=f"leaf_{leaf_idx}",
                    features=_ridge_feature_map(u_leaf, n_tokens=float(n_leaf)),
                    target_density=float(y_leaf) / float(max(1.0, n_leaf)),
                    n_tokens=float(n_leaf),
                    sampling=SamplingMetadata(
                        document_propensity=1.0,
                        unit_propensity=1.0,
                        label_propensity=1.0,
                        joint_propensity=1.0,
                        sampling_scheme="sampled_substructure_supervision",
                        policy_name="legacy_leaf_partition",
                        unit_kind=ObservationUnitKind.LEAF,
                        supports_ipw_estimation=False,
                    ),
                    training_variant="legacy_base",
                    metadata={"partition_kind": "latent_leaf_partition"},
                )
            )

        coarse_spans = _equal_token_spans(
            doc_tokens=int(config.doc_tokens), span_tokens=int(eval_leaf_tokens)
        )
        coarse_u, coarse_y, coarse_n = _aggregate_partition_spans(
            doc,
            coarse_spans,
            utility_matrix=utility_matrix,
            theta=theta_true,
            W_base=W_base,
            lambda_multiplier=float(config.lambda_multiplier),
        )
        mask_coarse = _legacy_query_mask(
            len(coarse_u),
            budget_regime=str(config.budget_regime),
            budget_equiv=float(config.leaf_label_budget),
            n_base=max(1, len(base_u)),
            rng=rng_coarse,
        )
        queried_leaves_coarse.append(float(np.sum(mask_coarse)))
        queried_cost_coarse.append(
            float(
                np.sum(mask_coarse)
                * max(1.0, float(eval_leaf_tokens) / float(max(1, config.latent_leaf_tokens)))
            )
        )
        for leaf_idx, (u_leaf, y_leaf, n_leaf, keep) in enumerate(
            zip(coarse_u, coarse_y, coarse_n, mask_coarse)
        ):
            if not bool(keep):
                continue
            coarse_rows.append(
                _analysis_section_supervision_row(
                    split="train_legacy_coarse",
                    doc_id=f"train_{len(queried_leaves_coarse) - 1}",
                    section_id=f"leaf_{leaf_idx}",
                    features=_ridge_feature_map(u_leaf, n_tokens=float(n_leaf)),
                    target_density=float(y_leaf) / float(max(1.0, n_leaf)),
                    n_tokens=float(n_leaf),
                    sampling=SamplingMetadata(
                        document_propensity=1.0,
                        unit_propensity=1.0,
                        label_propensity=1.0,
                        joint_propensity=1.0,
                        sampling_scheme="sampled_substructure_supervision",
                        policy_name="legacy_coarse_partition",
                        unit_kind=ObservationUnitKind.LEAF,
                        supports_ipw_estimation=False,
                    ),
                    training_variant="legacy_coarse",
                    metadata={"partition_kind": "equal_token_partition"},
                )
            )

    if not base_rows:
        base_rows = [
            _analysis_section_supervision_row(
                split="train_legacy_base",
                doc_id="train_fallback",
                section_id="leaf_0",
                features=_ridge_feature_map(
                    np.zeros((utility_matrix.shape[0],), dtype=np.float64),
                    n_tokens=1.0,
                ),
                target_density=0.0,
                n_tokens=1.0,
                sampling=SamplingMetadata(
                    document_propensity=1.0,
                    unit_propensity=1.0,
                    label_propensity=1.0,
                    joint_propensity=1.0,
                    sampling_scheme="sampled_substructure_supervision",
                    policy_name="legacy_leaf_partition",
                    unit_kind=ObservationUnitKind.LEAF,
                    supports_ipw_estimation=False,
                ),
                training_variant="legacy_base",
            )
        ]
    if not coarse_rows:
        coarse_rows = [
            _analysis_section_supervision_row(
                split="train_legacy_coarse",
                doc_id="train_fallback",
                section_id="leaf_0",
                features=_ridge_feature_map(
                    np.zeros((utility_matrix.shape[0],), dtype=np.float64),
                    n_tokens=1.0,
                ),
                target_density=0.0,
                n_tokens=1.0,
                sampling=SamplingMetadata(
                    document_propensity=1.0,
                    unit_propensity=1.0,
                    label_propensity=1.0,
                    joint_propensity=1.0,
                    sampling_scheme="sampled_substructure_supervision",
                    policy_name="legacy_coarse_partition",
                    unit_kind=ObservationUnitKind.LEAF,
                    supports_ipw_estimation=False,
                ),
                training_variant="legacy_coarse",
            )
        ]

    base_model, base_fit = fit_dense_scalar_ridge_regressor(
        _analysis_section_supervision_dataset(
            base_rows,
            split="train_legacy_base",
            training_variant="legacy_base",
        ),
        config=DenseScalarRidgeTrainingConfig(
            model=DenseScalarRidgeModelConfig(ridge_alpha=float(config.ridge_alpha))
        ),
    )
    coarse_model, coarse_fit = fit_dense_scalar_ridge_regressor(
        _analysis_section_supervision_dataset(
            coarse_rows,
            split="train_legacy_coarse",
            training_variant="legacy_coarse",
        ),
        config=DenseScalarRidgeTrainingConfig(
            model=DenseScalarRidgeModelConfig(ridge_alpha=float(config.ridge_alpha))
        ),
    )

    base_preds: List[float] = []
    coarse_preds: List[float] = []
    for doc in docs_test:
        base_u, _base_y, base_n = _aggregate_partition_spans(
            doc,
            doc.latent_section_spans,
            utility_matrix=utility_matrix,
            theta=theta_true,
            W_base=W_base,
            lambda_multiplier=float(config.lambda_multiplier),
        )
        base_pred = 0.0
        for u_leaf, n_leaf in zip(base_u, base_n):
            base_pred += float(n_leaf) * float(
                np.asarray(
                    getattr(base_model, "predict")(
                        np.asarray(
                            [_ridge_feature_map(u_leaf, n_tokens=float(n_leaf))],
                            dtype=np.float64,
                        )
                    ),
                    dtype=np.float64,
                )[0]
            )
        base_preds.append(float(base_pred))

        coarse_spans = _equal_token_spans(
            doc_tokens=int(config.doc_tokens), span_tokens=int(eval_leaf_tokens)
        )
        coarse_u, _coarse_y, coarse_n = _aggregate_partition_spans(
            doc,
            coarse_spans,
            utility_matrix=utility_matrix,
            theta=theta_true,
            W_base=W_base,
            lambda_multiplier=float(config.lambda_multiplier),
        )
        coarse_pred = 0.0
        for u_leaf, n_leaf in zip(coarse_u, coarse_n):
            coarse_pred += float(n_leaf) * float(
                np.asarray(
                    getattr(coarse_model, "predict")(
                        np.asarray(
                            [_ridge_feature_map(u_leaf, n_tokens=float(n_leaf))],
                            dtype=np.float64,
                        )
                    ),
                    dtype=np.float64,
                )[0]
            )
        coarse_preds.append(float(coarse_pred))

    return (
        np.asarray(base_preds, dtype=np.float64),
        np.asarray(coarse_preds, dtype=np.float64),
        {
            "queried_leaves_base": queried_leaves_base,
            "queried_cost_base": queried_cost_base,
            "queried_leaves_coarse": queried_leaves_coarse,
            "queried_cost_coarse": queried_cost_coarse,
            "beta_base_dim": int(base_fit.input_dim),
            "beta_coarse_dim": int(coarse_fit.input_dim),
            **supervision_training_contract(
                representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                target_kind=TARGET_SCALAR,
                optimizer_family=OPTIMIZER_FAMILY_CLOSED_FORM_LINEAR,
                optimizer_backend="closed_form_ridge",
                n_train_rows=int(base_fit.n_train_rows),
            ),
        },
    )


def _law_internal_probabilities(
    scores: np.ndarray, *, target_rate: float, design: str
) -> np.ndarray:
    mode = str(design).strip().lower()
    if mode == "uniform":
        return np.full(
            (int(np.asarray(scores, dtype=np.float64).size),), float(target_rate), dtype=np.float64
        )
    if mode == "risk":
        return _local_law_query_probabilities(
            np.asarray(scores, dtype=np.float64),
            target_rate=float(target_rate),
            design="proxy_priority",
            floor=0.0,
            ceiling=1.0,
        )
    raise ValueError(f"unsupported law_internal_query_design: {design!r}")


VALID_LDA_LAW_PACKAGES = (
    "root_only",
    "c1_only",
    "c3_only",
    "c1c3",
    "c2_only",
    "all_laws",
)

_LDA_PACKAGE_TO_LAW_SET_ID = {
    "all_laws": LAW_SET_ALL,
    "root_only": LAW_SET_ROOT_ONLY,
    "c1_only": LAW_SET_LEAF_PRESERVATION_ONLY,
    "c3_only": LAW_SET_MERGE_PRESERVATION_ONLY,
    "c2_only": LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY,
    "c1c3": LAW_SET_LEAF_AND_MERGE_PRESERVATION,
}


def _lda_public_law_set_id(config: LeafLocalMixtureUtilityConfig) -> str:
    return _LDA_PACKAGE_TO_LAW_SET_ID.get(
        str(getattr(config, "law_package", "all_laws")).strip().lower(),
        LAW_SET_ALL,
    )


def _active_lda_laws(config: LeafLocalMixtureUtilityConfig) -> Tuple[str, ...]:
    pkg = str(getattr(config, "law_package", "all_laws")).strip().lower()
    if pkg == "root_only":
        return tuple()
    if pkg == "c1_only":
        return (LAW_ID_LEAF_PRESERVATION,)
    if pkg == "c3_only":
        return (LAW_ID_MERGE_PRESERVATION,)
    if pkg == "c1c3":
        return (LAW_ID_LEAF_PRESERVATION, LAW_ID_MERGE_PRESERVATION)
    if pkg == "c2_only":
        return (LAW_ID_ON_RANGE_IDEMPOTENCE,)
    return (
        LAW_ID_LEAF_PRESERVATION,
        LAW_ID_ON_RANGE_IDEMPOTENCE,
        LAW_ID_MERGE_PRESERVATION,
    )


def _resolve_lda_objective_weights(
    config: LeafLocalMixtureUtilityConfig,
) -> Tuple[float, Dict[str, float], str]:
    active = _active_lda_laws(config)
    if config.local_law_weight is not None:
        explicit_overrides = (
            not math.isclose(float(config.law_task_objective_weight), 1.0),
            not math.isclose(float(config.law_c1_weight), 1.0 / 3.0),
            not math.isclose(float(config.law_c3_weight), 1.0 / 3.0),
            not math.isclose(float(config.law_c2_proxy_weight), 1.0 / 3.0),
        )
        if any(explicit_overrides):
            raise ValueError(
                "local_law_weight is mutually exclusive with explicit law/root weights"
            )
        resolved = resolve_root_local_objective_weights(
            local_law_weight=float(config.local_law_weight),
            active_laws=active,
            objective_context="lda objective",
        )
        return (
            float(resolved.root_share),
            {
                LAW_ID_LEAF_PRESERVATION: float(resolved.local_law_shares.get(LAW_ID_LEAF_PRESERVATION, 0.0)),
                LAW_ID_ON_RANGE_IDEMPOTENCE: float(resolved.local_law_shares.get(LAW_ID_ON_RANGE_IDEMPOTENCE, 0.0)),
                LAW_ID_MERGE_PRESERVATION: float(resolved.local_law_shares.get(LAW_ID_MERGE_PRESERVATION, 0.0)),
            },
            "lambda",
        )
    resolved = resolve_root_local_objective_weights(
        local_law_weight=None,
        active_laws=active,
        explicit_root_weight=float(config.law_task_objective_weight),
        explicit_law_weights=_resolved_lda_law_weight_map(config),
        objective_context="lda objective",
    )
    return (
        float(resolved.root_share),
        {
            LAW_ID_LEAF_PRESERVATION: float(resolved.local_law_shares.get(LAW_ID_LEAF_PRESERVATION, 0.0)),
            LAW_ID_ON_RANGE_IDEMPOTENCE: float(resolved.local_law_shares.get(LAW_ID_ON_RANGE_IDEMPOTENCE, 0.0)),
            LAW_ID_MERGE_PRESERVATION: float(resolved.local_law_shares.get(LAW_ID_MERGE_PRESERVATION, 0.0)),
        },
        "explicit_weights",
    )


def _resolve_lda_law_weights(config: LeafLocalMixtureUtilityConfig) -> Tuple[float, float, float]:
    """Resolve effective (c1, c3, c2_proxy) weights from law_package or config fields."""
    if config.local_law_weight is not None:
        _, law_weights, _ = _resolve_lda_objective_weights(config)
        return (
            float(law_weights.get(LAW_ID_LEAF_PRESERVATION, 0.0)),
            float(law_weights.get(LAW_ID_MERGE_PRESERVATION, 0.0)),
            float(law_weights.get(LAW_ID_ON_RANGE_IDEMPOTENCE, 0.0)),
        )
    pkg = str(getattr(config, "law_package", "all_laws")).strip().lower()
    if pkg == "all_laws":
        return (
            float(config.law_c1_weight),
            float(config.law_c3_weight),
            float(config.law_c2_proxy_weight),
        )
    if pkg == "root_only":
        return 0.0, 0.0, 0.0
    if pkg == "c1_only":
        return float(config.law_c1_weight), 0.0, 0.0
    if pkg == "c3_only":
        return 0.0, float(config.law_c3_weight), 0.0
    if pkg == "c1c3":
        return float(config.law_c1_weight), float(config.law_c3_weight), 0.0
    if pkg == "c2_only":
        return 0.0, 0.0, float(config.law_c2_proxy_weight)
    return (
        float(config.law_c1_weight),
        float(config.law_c3_weight),
        float(config.law_c2_proxy_weight),
    )


def _resolved_lda_law_weight_map(config: LeafLocalMixtureUtilityConfig) -> Dict[str, float]:
    eff_c1_w, eff_c3_w, eff_c2_w = _resolve_lda_law_weights(config)
    return {
        LAW_ID_LEAF_PRESERVATION: float(eff_c1_w),
        LAW_ID_ON_RANGE_IDEMPOTENCE: float(eff_c2_w),
        LAW_ID_MERGE_PRESERVATION: float(eff_c3_w),
    }


def _weighted_lda_law_score(
    config: LeafLocalMixtureUtilityConfig,
    *,
    c1: float,
    c2_proxy: float,
    c3: float,
) -> float:
    weights = _resolved_lda_law_weight_map(config)
    return float(
        float(weights[LAW_ID_LEAF_PRESERVATION]) * float(c1)
        + float(weights[LAW_ID_ON_RANGE_IDEMPOTENCE]) * float(c2_proxy)
        + float(weights[LAW_ID_MERGE_PRESERVATION]) * float(c3)
    )


def _collect_local_law_training_examples(
    docs_train: Sequence[LeafLocalMixtureDoc],
    *,
    world: LeafLocalMixtureUtilityWorld,
    config: LeafLocalMixtureUtilityConfig,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    eff_c1_w, eff_c3_w, eff_c2_w = _resolve_lda_law_weights(config)

    examples: List[Dict[str, object]] = []
    c2_sources: List[Dict[str, object]] = []
    leaf_train_samples: List[TreeSample] = []
    internal_train_samples: List[TreeSample] = []
    leaf_props: List[float] = []
    internal_props: List[float] = []
    leaf_scores_sample: List[float] = []
    internal_scores_sample: List[float] = []

    for doc_idx, doc in enumerate(docs_train):
        rng = np.random.default_rng(
            int(_splitmix64(int(config.seed) + 21001 + int(doc_idx)) & 0xFFFFFFFF)
        )
        view = _build_analysis_partition_view(doc, config, doc_index=doc_idx + 300000)
        truth_rows = np.asarray(view.analysis_topic_weights, dtype=np.float64)
        hat_rows = _infer_analysis_section_topics(doc, view, world=world, config=config)
        masses = _section_lengths_from_spans(view.analysis_section_spans)
        leaf_scores = _section_proxy_scores(
            doc, view.analysis_section_spans, world=world, config=config
        )
        leaf_probs = _local_law_query_probabilities(
            leaf_scores,
            target_rate=float(config.law_leaf_query_rate),
            design=str(config.law_leaf_query_design),
            floor=0.0,
            ceiling=1.0,
        )
        tree = _balanced_merge_internal_nodes(hat_rows, truth_rows, leaf_masses=masses)
        internal_nodes = list(tree["internal_nodes"])
        internal_scores = np.asarray(
            [float(node["risk"]) for node in internal_nodes], dtype=np.float64
        )
        internal_probs = _law_internal_probabilities(
            internal_scores,
            target_rate=float(config.law_internal_query_rate),
            design=str(config.law_internal_query_design),
        )

        for sec_idx in range(int(hat_rows.shape[0])):
            p_leaf = float(leaf_probs[sec_idx]) if sec_idx < leaf_probs.size else 0.0
            if p_leaf <= 0.0 or float(rng.random()) >= p_leaf:
                continue
            examples.append(
                {
                    "x": np.asarray(hat_rows[sec_idx], dtype=np.float64).copy(),
                    "y": np.asarray(truth_rows[sec_idx], dtype=np.float64).copy(),
                    "kind": "leaf",
                    "base_weight": float(eff_c1_w),
                    "joint_propensity": float(max(MIN_PROPENSITY, p_leaf)),
                    "n_tokens": float(masses[sec_idx]),
                }
            )
            c2_sources.append(
                {
                    "raw": np.asarray(hat_rows[sec_idx], dtype=np.float64).copy(),
                    "joint_propensity": float(max(MIN_PROPENSITY, p_leaf)),
                    "n_tokens": float(masses[sec_idx]),
                }
            )
            leaf_props.append(float(p_leaf))
            leaf_scores_sample.append(
                float(leaf_scores[sec_idx]) if sec_idx < leaf_scores.size else float("nan")
            )
            leaf_train_samples.append(
                TreeSample(
                    doc_id=f"law_train_{doc_idx}",
                    node_id=f"leaf_{sec_idx}",
                    node_type=NodeType.LEAF,
                    violation=0,
                    preference_loss=float(max(0.0, min(1.0, p_leaf))),
                    sampling=SamplingMetadata(
                        unit_propensity=max(MIN_PROPENSITY, p_leaf),
                        unit_kind=ObservationUnitKind.LEAF,
                    ),
                )
            )

        for node_idx, node in enumerate(internal_nodes):
            p_internal = float(internal_probs[node_idx]) if node_idx < internal_probs.size else 0.0
            if p_internal <= 0.0 or float(rng.random()) >= p_internal:
                continue
            examples.append(
                {
                    "x": np.asarray(node["est"], dtype=np.float64).copy(),
                    "y": np.asarray(node["truth"], dtype=np.float64).copy(),
                    "kind": "internal",
                    "base_weight": float(eff_c3_w),
                    "joint_propensity": float(max(MIN_PROPENSITY, p_internal)),
                    "n_tokens": float(node["mass"]),
                }
            )
            internal_props.append(float(p_internal))
            internal_scores_sample.append(float(node["risk"]))
            internal_train_samples.append(
                TreeSample(
                    doc_id=f"law_train_{doc_idx}",
                    node_id=f"internal_{node_idx}",
                    node_type=NodeType.MERGE,
                    violation=0,
                    preference_loss=float(max(0.0, min(1.0, p_internal))),
                    sampling=SamplingMetadata(
                        unit_propensity=max(MIN_PROPENSITY, p_internal),
                        unit_kind=ObservationUnitKind.MERGE,
                    ),
                )
            )

    diag = {
        "leaf_label_count": int(sum(1 for ex in examples if str(ex.get("kind")) == "leaf")),
        "internal_label_count": int(sum(1 for ex in examples if str(ex.get("kind")) == "internal")),
        "leaf_mean_propensity": _base._safe_stat(leaf_props, kind="mean"),
        "internal_mean_propensity": _base._safe_stat(internal_props, kind="mean"),
        "leaf_effective_sample_size": float(effective_sample_size(leaf_train_samples)),
        "internal_effective_sample_size": float(effective_sample_size(internal_train_samples)),
        "leaf_max_weight": float(max_weight(leaf_train_samples)),
        "internal_max_weight": float(max_weight(internal_train_samples)),
        "leaf_proxy_score_mean": _base._safe_stat(leaf_scores_sample, kind="mean"),
        "internal_risk_score_mean": _base._safe_stat(internal_scores_sample, kind="mean"),
    }
    return examples, c2_sources, diag


def _fit_local_law_calibrator_variants(
    examples: Sequence[Dict[str, object]],
    c2_sources: Sequence[Dict[str, object]],
    *,
    world: LeafLocalMixtureUtilityWorld,
    config: LeafLocalMixtureUtilityConfig,
) -> Dict[str, Dict[str, object]]:
    eff_c1_w, eff_c3_w, eff_c2_w = _resolve_lda_law_weights(config)
    # root_only package: all law weights are zero → return identity calibrators
    if not examples or (eff_c1_w == 0.0 and eff_c3_w == 0.0 and eff_c2_w == 0.0):
        dim = int(config.n_topics)
        ident = {
            "kind": "affine",
            "w": np.eye(dim, dtype=np.float64),
            "b": np.zeros((dim,), dtype=np.float64),
        }
        return {
            "law_calibrated_naive": dict(ident),
            "law_calibrated_ipw": dict(ident),
            "law_calibrated_ipw_stabilized": dict(ident),
        }

    x_base = np.stack([np.asarray(ex["x"], dtype=np.float64) for ex in examples], axis=0)
    y_base = np.stack([np.asarray(ex["y"], dtype=np.float64) for ex in examples], axis=0)

    def _variant_weights(name: str) -> np.ndarray:
        vals: List[float] = []
        for ex in examples:
            prop = float(max(MIN_PROPENSITY, ex.get("joint_propensity", 1.0)))
            base_w = float(ex.get("base_weight", 1.0))
            if name == "law_calibrated_naive":
                w = base_w
            elif name == "law_calibrated_ipw":
                w = base_w / prop
            elif name == "law_calibrated_ipw_stabilized":
                w = base_w * min(float(config.ipw_stabilized_clip), 1.0 / prop)
            else:
                raise ValueError(f"unsupported local-law calibrator variant: {name!r}")
            vals.append(float(w))
        arr = np.asarray(vals, dtype=np.float64)
        if name == "law_calibrated_ipw_stabilized" and arr.size > 0 and float(np.mean(arr)) > 0.0:
            arr = arr / float(np.mean(arr))
        return arr

    variants: Dict[str, Dict[str, object]] = {}
    for name in ("law_calibrated_naive", "law_calibrated_ipw", "law_calibrated_ipw_stabilized"):
        weights = _variant_weights(name)
        cal = _fit_affine_summary_calibrator(
            x_base,
            y_base,
            ridge=float(config.law_calibration_ridge),
            sample_weight=weights,
        )
        if float(eff_c2_w) > 0.0 and c2_sources:
            c2_x: List[np.ndarray] = []
            c2_y: List[np.ndarray] = []
            c2_w: List[float] = []
            for item in c2_sources:
                summary0 = _apply_affine_summary_calibrator(
                    np.asarray(item["raw"], dtype=np.float64),
                    w=np.asarray(cal["w"], dtype=np.float64),
                    b=np.asarray(cal["b"], dtype=np.float64),
                )
                reinferred = _reinfer_expected_summary(
                    summary0,
                    n_tokens=float(item["n_tokens"]),
                    world=world,
                    config=config,
                )
                prop = float(max(MIN_PROPENSITY, item.get("joint_propensity", 1.0)))
                if name == "law_calibrated_naive":
                    w_c2 = float(eff_c2_w)
                elif name == "law_calibrated_ipw":
                    w_c2 = float(eff_c2_w) / prop
                else:
                    w_c2 = float(eff_c2_w) * min(float(config.ipw_stabilized_clip), 1.0 / prop)
                c2_x.append(np.asarray(reinferred, dtype=np.float64))
                c2_y.append(np.asarray(summary0, dtype=np.float64))
                c2_w.append(float(w_c2))
            c2_arr = np.asarray(c2_w, dtype=np.float64)
            if (
                name == "law_calibrated_ipw_stabilized"
                and c2_arr.size > 0
                and float(np.mean(c2_arr)) > 0.0
            ):
                c2_arr = c2_arr / float(np.mean(c2_arr))
            cal = _fit_affine_summary_calibrator(
                np.concatenate([x_base, np.stack(c2_x, axis=0)], axis=0),
                np.concatenate([y_base, np.stack(c2_y, axis=0)], axis=0),
                ridge=float(config.law_calibration_ridge),
                sample_weight=np.concatenate([weights, c2_arr], axis=0),
            )
        variants[name] = {"kind": "affine", **cal}
    return variants


def _evaluate_local_law_doc_policy(
    summary_rows: np.ndarray,
    truth_rows: np.ndarray,
    *,
    masses: np.ndarray,
    reapply_fn: Callable[[np.ndarray], np.ndarray],
    world: LeafLocalMixtureUtilityWorld,
    config: LeafLocalMixtureUtilityConfig,
    theta_true: np.ndarray,
    W_base: np.ndarray,
) -> Dict[str, object]:
    tree = _balanced_merge_internal_nodes(summary_rows, truth_rows, leaf_masses=masses)
    c1_values = [float(x) for x in tree["leaf_c1"]]
    internal_nodes_seq = list(tree["internal_nodes"])
    c3_values = [float(node["error"]) for node in internal_nodes_seq]
    per_node_merge_residual = list(c3_values)
    per_node_level = [int(node.get("level", 0)) for node in internal_nodes_seq]
    if per_node_level:
        max_level = max(per_node_level)
        per_node_depth = [int(max_level - lvl) for lvl in per_node_level]
    else:
        per_node_depth = []
    c2_values: List[float] = []
    for row, n_tok in zip(
        np.asarray(summary_rows, dtype=np.float64), np.asarray(masses, dtype=np.float64)
    ):
        reinferred = _reinfer_expected_summary(
            row,
            n_tokens=float(n_tok),
            world=world,
            config=config,
        )
        reapplied = np.asarray(
            reapply_fn(np.asarray(reinferred, dtype=np.float64)), dtype=np.float64
        )
        c2_values.append(_l1(reapplied, np.asarray(row, dtype=np.float64)))
    aux_target = _analysis_target_from_topic_rows(
        np.asarray(summary_rows, dtype=np.float64),
        weights=(
            np.asarray(masses, dtype=np.float64)
            / float(np.sum(np.asarray(masses, dtype=np.float64)))
        ),
        theta=theta_true,
        W_base=W_base,
        lambda_multiplier=float(config.lambda_multiplier),
        doc_tokens=int(np.sum(np.asarray(masses, dtype=np.float64))),
        weighted=True,
    )
    total_mass = float(np.sum(np.asarray(masses, dtype=np.float64)))
    if total_mass > 0.0:
        pooled_row = np.sum(
            np.asarray(summary_rows, dtype=np.float64)
            * (np.asarray(masses, dtype=np.float64)[:, None] / total_mass),
            axis=0,
        )
    else:
        pooled_row = np.mean(np.asarray(summary_rows, dtype=np.float64), axis=0)
    pooled_target = float(
        int(np.sum(np.asarray(masses, dtype=np.float64)))
    ) * _leaf_additive_utility(
        _normalize_simplex_vec(np.asarray(pooled_row, dtype=np.float64)),
        theta=theta_true,
        W_base=W_base,
        lambda_multiplier=float(config.lambda_multiplier),
    )
    mean_c1 = _base._safe_stat(c1_values, kind="mean")
    mean_c3 = _base._safe_stat(c3_values, kind="mean")
    mean_c2 = _base._safe_stat(c2_values, kind="mean")
    return {
        "c1_values": c1_values,
        "c3_values": c3_values,
        "c2_proxy_values": c2_values,
        "mean_c1": mean_c1,
        "mean_c3": mean_c3,
        "mean_c2_proxy": mean_c2,
        "combined_law_score": _weighted_lda_law_score(
            config,
            c1=mean_c1,
            c2_proxy=mean_c2,
            c3=mean_c3,
        ),
        "root_c3_error": float(tree["root_error"]),
        "aux_oracle_target": float(aux_target),
        "aux_oracle_target_delta": float(aux_target - pooled_target),
        # Per-node merge residuals retained as diagnostics: each entry is
        # ‖fhat(p) − merge(fhat(children(p)))‖ at internal node p.
        "per_node_merge_residual": per_node_merge_residual,
        "per_node_depth": per_node_depth,
    }


def _summarize_local_law_policy(
    doc_metrics: Sequence[Dict[str, object]],
    *,
    oracle_targets: Sequence[float],
    pooled_errors: Sequence[float],
    objective_spec: CompositeObjectiveSpec,
    config: LeafLocalMixtureUtilityConfig,
    baseline_aux_targets: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    c1_all: List[float] = []
    c3_all: List[float] = []
    c2_all: List[float] = []
    combined_doc: List[float] = []
    root_errors: List[float] = []
    aux_abs_errors: List[float] = []
    aux_delta: List[float] = []
    for idx, metrics in enumerate(doc_metrics):
        c1_vals = [float(x) for x in metrics.get("c1_values", [])]
        c3_vals = [float(x) for x in metrics.get("c3_values", [])]
        c2_vals = [float(x) for x in metrics.get("c2_proxy_values", [])]
        c1_all.extend(c1_vals)
        c3_all.extend(c3_vals)
        c2_all.extend(c2_vals)
        combined_doc.append(float(metrics.get("combined_law_score", float("nan"))))
        root_errors.append(float(metrics.get("root_c3_error", float("nan"))))
        oracle_target = float(oracle_targets[idx]) if idx < len(oracle_targets) else float("nan")
        aux_err = abs(float(metrics.get("aux_oracle_target", float("nan"))) - oracle_target)
        aux_abs_errors.append(float(aux_err))
        if baseline_aux_targets is not None and idx < len(baseline_aux_targets):
            # Delta = policy_aux - baseline_aux (within same analysis-partition
            # framework). Positive means policy is farther from oracle than
            # the infer_identity baseline.
            aux_delta.append(
                float(metrics.get("aux_oracle_target", float("nan")))
                - float(baseline_aux_targets[idx])
            )
        else:
            # Fallback: use per-doc cross-framework delta (aux vs pooled).
            policy_delta = float(metrics.get("aux_oracle_target_delta", float("nan")))
            if math.isfinite(policy_delta):
                aux_delta.append(policy_delta)
            else:
                pooled_err = float(pooled_errors[idx]) if idx < len(pooled_errors) else float("nan")
                aux_delta.append(
                    float(pooled_err) - float(aux_err)
                    if math.isfinite(pooled_err)
                    else float("nan")
                )
    mean_c1 = _base._safe_stat(c1_all, kind="mean")
    mean_c3 = _base._safe_stat(c3_all, kind="mean")
    mean_c2 = _base._safe_stat(c2_all, kind="mean")
    combined_mean = _weighted_lda_law_score(
        config,
        c1=mean_c1,
        c2_proxy=mean_c2,
        c3=mean_c3,
    )
    law_doc_arr = np.asarray(combined_doc, dtype=np.float64)
    aux_err_arr = np.asarray(aux_abs_errors, dtype=np.float64)
    mean_aux_abs_error = _base._safe_stat(aux_abs_errors, kind="mean")
    objective_eval = evaluate_composite_objective(
        objective_spec,
        task_value=float(mean_aux_abs_error),
        local_law_values={
            LAW_ID_LEAF_PRESERVATION: float(mean_c1),
            LAW_ID_ON_RANGE_IDEMPOTENCE: float(mean_c2),
            LAW_ID_MERGE_PRESERVATION: float(mean_c3),
        },
    )
    corr = (
        float(np.corrcoef(law_doc_arr, aux_err_arr)[0, 1])
        if law_doc_arr.size >= 2 and np.std(law_doc_arr) > 1e-12 and np.std(aux_err_arr) > 1e-12
        else float("nan")
    )
    return {
        "mean_c1": mean_c1,
        "mean_c3": mean_c3,
        "mean_c2_proxy": mean_c2,
        "combined_law_score": combined_mean,
        "c1_violation_rate": (
            float(
                np.mean(
                    (np.asarray(c1_all, dtype=np.float64) > float(config.law_c1_threshold)).astype(
                        np.float64
                    )
                )
            )
            if c1_all
            else 0.0
        ),
        "c3_violation_rate": (
            float(
                np.mean(
                    (np.asarray(c3_all, dtype=np.float64) > float(config.law_c3_threshold)).astype(
                        np.float64
                    )
                )
            )
            if c3_all
            else 0.0
        ),
        "c2_proxy_violation_rate": (
            float(
                np.mean(
                    (np.asarray(c2_all, dtype=np.float64) > float(config.law_c2_threshold)).astype(
                        np.float64
                    )
                )
            )
            if c2_all
            else 0.0
        ),
        "mean_root_c3_error": _base._safe_stat(root_errors, kind="mean"),
        "mean_aux_oracle_target_abs_error": mean_aux_abs_error,
        "mean_aux_oracle_target_delta": _base._safe_stat(aux_delta, kind="mean"),
        "law_score_aux_abs_error_correlation": corr,
        **objective_eval.to_flat_dict(prefix=str(objective_spec.selection_metric_name)),
    }


def _local_law_ipw_evaluation(
    doc_records: Sequence[Dict[str, object]],
    *,
    config: LeafLocalMixtureUtilityConfig,
) -> Tuple[Dict[str, object], Dict[str, List[LoggedLabelObservation[float]]]]:
    if not doc_records:
        return {}, {}
    eval_rng = np.random.default_rng(int(_splitmix64(int(config.seed) + 23003) & 0xFFFFFFFF))
    policy_names = list(
        dict.fromkeys(str(name) for rec in doc_records for name in rec.get("policies", {}).keys())
    )
    leaf_pop: Dict[str, Dict[str, List[float]]] = {
        name: {"c1": [], "c2_proxy": []} for name in policy_names
    }
    internal_pop: Dict[str, Dict[str, List[float]]] = {name: {"c3": []} for name in policy_names}
    leaf_samples: Dict[str, Dict[str, List[TreeSample]]] = {
        name: {"c1": [], "c2_proxy": []} for name in policy_names
    }
    internal_samples: Dict[str, Dict[str, List[TreeSample]]] = {
        name: {"c3": []} for name in policy_names
    }
    leaf_meta_samples: List[TreeSample] = []
    internal_meta_samples: List[TreeSample] = []

    for rec in doc_records:
        policies = dict(rec.get("policies", {}))
        leaf_scores = np.asarray(rec.get("leaf_scores", []), dtype=np.float64)
        internal_scores = np.asarray(rec.get("internal_scores", []), dtype=np.float64)
        leaf_probs = _local_law_query_probabilities(
            leaf_scores,
            target_rate=float(config.law_eval_leaf_sample_rate),
            design=str(config.law_leaf_query_design),
            floor=0.0,
            ceiling=1.0,
        )
        internal_probs = _law_internal_probabilities(
            internal_scores,
            target_rate=float(config.law_eval_internal_sample_rate),
            design=str(config.law_internal_query_design),
        )
        for policy_name in policy_names:
            policy_metrics = dict(policies.get(policy_name, {}))
            leaf_pop[policy_name]["c1"].extend(
                float(x) for x in policy_metrics.get("c1_values", [])
            )
            leaf_pop[policy_name]["c2_proxy"].extend(
                float(x) for x in policy_metrics.get("c2_proxy_values", [])
            )
            internal_pop[policy_name]["c3"].extend(
                float(x) for x in policy_metrics.get("c3_values", [])
            )
        if float(eval_rng.random()) >= float(config.heldout_doc_sample_rate):
            continue
        doc_prop = max(MIN_PROPENSITY, float(config.heldout_doc_sample_rate))
        for idx, p_leaf in enumerate(leaf_probs.tolist()):
            if float(eval_rng.random()) >= float(p_leaf):
                continue
            leaf_meta_samples.append(
                TreeSample(
                    doc_id=f"law_eval_{rec['doc_idx']}",
                    node_id=f"leaf_{idx}",
                    node_type=NodeType.LEAF,
                    violation=0,
                    preference_loss=float(max(0.0, min(1.0, p_leaf))),
                    sampling=SamplingMetadata(
                        document_propensity=doc_prop,
                        unit_propensity=max(MIN_PROPENSITY, float(p_leaf)),
                        unit_kind=ObservationUnitKind.LEAF,
                    ),
                )
            )
            for policy_name in policy_names:
                policy_metrics = dict(policies.get(policy_name, {}))
                c1_vals = [float(x) for x in policy_metrics.get("c1_values", [])]
                c2_vals = [float(x) for x in policy_metrics.get("c2_proxy_values", [])]
                if idx < len(c1_vals):
                    leaf_samples[policy_name]["c1"].append(
                        TreeSample(
                            doc_id=f"law_eval_{rec['doc_idx']}",
                            node_id=f"leaf_c1_{idx}",
                            node_type=NodeType.LEAF,
                            violation=0,
                            preference_loss=float(c1_vals[idx]),
                            sampling=SamplingMetadata(
                                document_propensity=doc_prop,
                                unit_propensity=max(MIN_PROPENSITY, float(p_leaf)),
                                unit_kind=ObservationUnitKind.LEAF,
                            ),
                        )
                    )
                if idx < len(c2_vals):
                    leaf_samples[policy_name]["c2_proxy"].append(
                        TreeSample(
                            doc_id=f"law_eval_{rec['doc_idx']}",
                            node_id=f"leaf_c2_{idx}",
                            node_type=NodeType.LEAF,
                            violation=0,
                            preference_loss=float(c2_vals[idx]),
                            sampling=SamplingMetadata(
                                document_propensity=doc_prop,
                                unit_propensity=max(MIN_PROPENSITY, float(p_leaf)),
                                unit_kind=ObservationUnitKind.LEAF,
                            ),
                        )
                    )
        for idx, p_internal in enumerate(internal_probs.tolist()):
            if float(eval_rng.random()) >= float(p_internal):
                continue
            internal_meta_samples.append(
                TreeSample(
                    doc_id=f"law_eval_{rec['doc_idx']}",
                    node_id=f"internal_{idx}",
                    node_type=NodeType.MERGE,
                    violation=0,
                    preference_loss=float(max(0.0, min(1.0, p_internal))),
                    sampling=SamplingMetadata(
                        document_propensity=doc_prop,
                        unit_propensity=max(MIN_PROPENSITY, float(p_internal)),
                        unit_kind=ObservationUnitKind.MERGE,
                    ),
                )
            )
            for policy_name in policy_names:
                policy_metrics = dict(policies.get(policy_name, {}))
                c3_vals = [float(x) for x in policy_metrics.get("c3_values", [])]
                if idx < len(c3_vals):
                    internal_samples[policy_name]["c3"].append(
                        TreeSample(
                            doc_id=f"law_eval_{rec['doc_idx']}",
                            node_id=f"internal_c3_{idx}",
                            node_type=NodeType.MERGE,
                            violation=0,
                            preference_loss=float(c3_vals[idx]),
                            sampling=SamplingMetadata(
                                document_propensity=doc_prop,
                                unit_propensity=max(MIN_PROPENSITY, float(p_internal)),
                                unit_kind=ObservationUnitKind.MERGE,
                            ),
                        )
                    )

    out: Dict[str, object] = {}
    logged_by_policy: Dict[str, List[LoggedLabelObservation[float]]] = {}
    for policy_name in policy_names:
        c1_exact = _base._safe_stat(leaf_pop[policy_name]["c1"], kind="mean")
        c2_exact = _base._safe_stat(leaf_pop[policy_name]["c2_proxy"], kind="mean")
        c3_exact = _base._safe_stat(internal_pop[policy_name]["c3"], kind="mean")
        c1_eval = _normalized_ci_and_coverage(
            leaf_samples[policy_name]["c1"],
            exact_value=c1_exact,
            raw_values_population=leaf_pop[policy_name]["c1"],
            population_size=float(max(1, len(leaf_pop[policy_name]["c1"]))),
            delta=float(config.ipw_delta),
        )
        c2_eval = _normalized_ci_and_coverage(
            leaf_samples[policy_name]["c2_proxy"],
            exact_value=c2_exact,
            raw_values_population=leaf_pop[policy_name]["c2_proxy"],
            population_size=float(max(1, len(leaf_pop[policy_name]["c2_proxy"]))),
            delta=float(config.ipw_delta),
        )
        c3_eval = _normalized_ci_and_coverage(
            internal_samples[policy_name]["c3"],
            exact_value=c3_exact,
            raw_values_population=internal_pop[policy_name]["c3"],
            population_size=float(max(1, len(internal_pop[policy_name]["c3"]))),
            delta=float(config.ipw_delta),
        )
        combined_exact = _weighted_lda_law_score(
            config,
            c1=c1_exact,
            c2_proxy=c2_exact,
            c3=c3_exact,
        )
        combined_ht = _weighted_lda_law_score(
            config,
            c1=float(c1_eval["ht_mean"]),
            c2_proxy=float(c2_eval["ht_mean"]),
            c3=float(c3_eval["ht_mean"]),
        )
        combined_hajek = _weighted_lda_law_score(
            config,
            c1=float(c1_eval["hajek"]),
            c2_proxy=float(c2_eval["hajek"]),
            c3=float(c3_eval["hajek"]),
        )
        combined_lo = _weighted_lda_law_score(
            config,
            c1=float(c1_eval["eb_lo"]),
            c2_proxy=float(c2_eval["eb_lo"]),
            c3=float(c3_eval["eb_lo"]),
        )
        combined_hi = _weighted_lda_law_score(
            config,
            c1=float(c1_eval["eb_hi"]),
            c2_proxy=float(c2_eval["eb_hi"]),
            c3=float(c3_eval["eb_hi"]),
        )
        out[policy_name] = {
            "c1": {
                **c1_eval,
                "population_exact_mean": c1_exact,
            },
            "c2_proxy": {
                **c2_eval,
                "population_exact_mean": c2_exact,
            },
            "c3": {
                **c3_eval,
                "population_exact_mean": c3_exact,
            },
            "combined": {
                "population_exact_mean": combined_exact,
                "ht_mean": combined_ht,
                "hajek": combined_hajek,
                "ht_abs_error": abs(combined_ht - combined_exact),
                "hajek_abs_error": abs(combined_hajek - combined_exact),
                "eb_lo": combined_lo,
                "eb_hi": combined_hi,
                "eb_width": float(max(0.0, combined_hi - combined_lo)),
                "eb_contains_exact": (
                    1.0 if combined_lo - 1e-12 <= combined_exact <= combined_hi + 1e-12 else 0.0
                ),
                "leaf_effective_sample_size": float(
                    effective_sample_size(leaf_samples[policy_name]["c1"])
                ),
                "internal_effective_sample_size": float(
                    effective_sample_size(internal_samples[policy_name]["c3"])
                ),
            },
            "diagnostics": {
                "leaf_sample_count": int(len(leaf_samples[policy_name]["c1"])),
                "internal_sample_count": int(len(internal_samples[policy_name]["c3"])),
                "leaf_effective_sample_size": float(
                    effective_sample_size(leaf_samples[policy_name]["c1"])
                ),
                "internal_effective_sample_size": float(
                    effective_sample_size(internal_samples[policy_name]["c3"])
                ),
                "leaf_max_weight": float(max_weight(leaf_samples[policy_name]["c1"])),
                "internal_max_weight": float(max_weight(internal_samples[policy_name]["c3"])),
                "leaf_propensity_quantiles": _propensity_quantiles(leaf_meta_samples),
                "internal_propensity_quantiles": _propensity_quantiles(internal_meta_samples),
            },
        }
        logged_observations = _samples_to_logged_observations(
            leaf_samples[policy_name]["c1"],
            supervision_signal_name="c1",
        )
        logged_observations.extend(
            _samples_to_logged_observations(
                leaf_samples[policy_name]["c2_proxy"],
                supervision_signal_name="c2_proxy",
            )
        )
        logged_observations.extend(
            _samples_to_logged_observations(
                internal_samples[policy_name]["c3"],
                supervision_signal_name="c3",
            )
        )
        logged_by_policy[str(policy_name)] = logged_observations
    return out, logged_by_policy


def _build_local_law_payload(
    docs_train: Sequence[LeafLocalMixtureDoc],
    docs_val: Sequence[LeafLocalMixtureDoc],
    docs_test: Sequence[LeafLocalMixtureDoc],
    *,
    world: LeafLocalMixtureUtilityWorld,
    config: LeafLocalMixtureUtilityConfig,
    theta_true: np.ndarray,
    W_base: np.ndarray,
    oracle_arr: np.ndarray,
    pooled_err: np.ndarray,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object]]:
    if str(config.local_law_mode).strip().lower() == "off":
        return {}, {}, {}, {}

    objective_spec = _local_law_objective_spec(config)
    training_examples, c2_sources, training_diag = _collect_local_law_training_examples(
        docs_train,
        world=world,
        config=config,
    )
    learned_policies: Dict[str, Dict[str, object]] = {}
    if str(config.local_law_mode).strip().lower() == "diagnostics_and_learned":
        learned_policies = _fit_local_law_calibrator_variants(
            training_examples,
            c2_sources,
            world=world,
            config=config,
        )

    # Inject exact counterexample family calibrator if requested
    exact_fam = str(getattr(config, "exact_family", "")).strip()
    if exact_fam:
        from treepo._research.ctreepo.sim.core.lda_law_stress import build_exact_family_calibrator

        exact_cal = build_exact_family_calibrator(
            exact_fam,
            n_topics=int(config.n_topics),
            seed=int(config.seed),
        )
        learned_policies[f"exact_{exact_fam}"] = exact_cal

    split_evals = {
        "train": _evaluate_local_law_split(
            docs_train,
            split_name="train",
            split_doc_offset=300_000,
            learned_policies=learned_policies,
            objective_spec=objective_spec,
            world=world,
            config=config,
            theta_true=theta_true,
            W_base=W_base,
        ),
        "val": _evaluate_local_law_split(
            docs_val,
            split_name="val",
            split_doc_offset=400_000,
            learned_policies=learned_policies,
            objective_spec=objective_spec,
            world=world,
            config=config,
            theta_true=theta_true,
            W_base=W_base,
        ),
        "test": _evaluate_local_law_split(
            docs_test,
            split_name="test",
            split_doc_offset=500_000,
            learned_policies=learned_policies,
            objective_spec=objective_spec,
            world=world,
            config=config,
            theta_true=theta_true,
            W_base=W_base,
        ),
    }
    test_eval = dict(split_evals["test"])
    policy_metrics = dict(test_eval.get("policy_metrics", {}) or {})
    violation_rates = dict(test_eval.get("violation_rates", {}) or {})
    mediation = dict(test_eval.get("mediation", {}) or {})

    selected_candidate: Optional[str] = None
    selection_split = ""
    selection_metric = ""
    if learned_policies:
        if docs_val:
            selected_candidate = _select_local_law_candidate(
                dict(split_evals["val"].get("policy_metrics", {}) or {}),
                objective_spec=objective_spec,
            )
            selection_split = "val"
            if selected_candidate is not None:
                selected_val_metrics = dict(
                    dict(split_evals["val"].get("policy_metrics", {}) or {}).get(
                        selected_candidate,
                        {},
                    )
                )
                selection_metric = str(
                    selected_val_metrics.get(
                        "selection_metric_name",
                        objective_estimator_alias(str(objective_spec.name), "hajek"),
                    )
                )
        if selected_candidate is None:
            for name in (
                "law_calibrated_ipw_stabilized",
                "law_calibrated_ipw",
                "law_calibrated_naive",
            ):
                if name in learned_policies:
                    selected_candidate = str(name)
                    break
            if selected_candidate is not None and not selection_metric:
                selection_metric = "fallback_candidate_priority"

    # Law-stress classification: compare each learned policy against identity baseline
    law_stress_assessments: Dict[str, object] = {}
    if learned_policies and "infer_identity" in policy_metrics:
        from treepo._research.ctreepo.sim.core.law_stress_common import (
            classify_law_stress as _classify_law_stress,
        )

        _baseline = policy_metrics["infer_identity"]
        for _lp_name in learned_policies:
            _sel = policy_metrics[_lp_name]
            law_stress_assessments[_lp_name] = _classify_law_stress(
                baseline_c1=float(_baseline["mean_c1"]),
                baseline_c2=float(_baseline["mean_c2_proxy"]),
                baseline_c3=float(_baseline["mean_c3"]),
                baseline_spread=0.0,
                baseline_root_mae=float(_baseline["mean_aux_oracle_target_abs_error"]),
                selected_c1=float(_sel["mean_c1"]),
                selected_c2=float(_sel["mean_c2_proxy"]),
                selected_c3=float(_sel["mean_c3"]),
                selected_spread=0.0,
                selected_root_mae=float(_sel["mean_aux_oracle_target_abs_error"]),
            ).to_dict()

    exact_metrics = {
        "oracle_true_summary": dict(policy_metrics["oracle_true_summary"]),
        "infer_identity": dict(policy_metrics["infer_identity"]),
    }

    split_policy_metrics = {
        split: dict(payload.get("policy_metrics", {}) or {})
        for split, payload in split_evals.items()
    }
    split_violation_rates = {
        split: dict(payload.get("violation_rates", {}) or {})
        for split, payload in split_evals.items()
    }
    split_mediation = {
        split: dict(payload.get("mediation", {}) or {}) for split, payload in split_evals.items()
    }
    split_ipw_evaluation = {
        split: dict(payload.get("ipw_evaluation", {}) or {})
        for split, payload in split_evals.items()
    }
    objective_payload = dict(objective_spec.to_dict())
    objective_payload["objective_name"] = str(objective_spec.name)
    preferred_objective_metric = objective_estimator_alias(str(objective_spec.name), "hajek")
    any_hajek_available = any(
        math.isfinite(
            float(dict(metrics).get(preferred_objective_metric, float("nan")))
        )
        for metrics in policy_metrics.values()
        if isinstance(metrics, Mapping)
    )
    if selection_metric and str(selection_metric).startswith(str(objective_spec.name)):
        objective_payload["selection_metric_name"] = str(selection_metric)
    elif any_hajek_available:
        objective_payload["selection_metric_name"] = str(preferred_objective_metric)
    else:
        objective_payload["selection_metric_name"] = str(objective_spec.selection_metric_name)
    objective_payload["available_estimators"] = (
        list(
            next(
                (
                    dict(metrics).get("available_objective_estimators", [])
                    for metrics in policy_metrics.values()
                    if isinstance(metrics, Mapping)
                    and dict(metrics).get("available_objective_estimators")
                ),
                [],
            )
        )
    )

    payload = {
        "config": {
            "local_law_mode": str(config.local_law_mode),
            "law_set_id": _lda_public_law_set_id(config),
            "exact_family": str(getattr(config, "exact_family", "")),
            "law_leaf_query_rate": float(config.law_leaf_query_rate),
            "law_internal_query_rate": float(config.law_internal_query_rate),
            "law_leaf_query_design": str(config.law_leaf_query_design),
            "law_internal_query_design": str(config.law_internal_query_design),
            "local_law_weight": (
                None if config.local_law_weight is None else float(config.local_law_weight)
            ),
            "root_share": float(objective_spec.normalized_task_share()),
            "local_law_component_weights": {
                str(name): float(value)
                for name, value in objective_spec.normalized_local_law_weights().items()
            },
            "law_calibration_ridge": float(config.law_calibration_ridge),
            "law_eval_leaf_sample_rate": float(config.law_eval_leaf_sample_rate),
            "law_eval_internal_sample_rate": float(config.law_eval_internal_sample_rate),
            "law_c1_threshold": float(config.law_c1_threshold),
            "law_c3_threshold": float(config.law_c3_threshold),
            "law_c2_threshold": float(config.law_c2_threshold),
        },
        "objective": objective_payload,
        "exact_metrics": exact_metrics,
        "violation_rates": violation_rates,
        "policy_metrics": policy_metrics,
        "ipw_evaluation": dict(test_eval.get("ipw_evaluation", {}) or {}),
        "mediation": mediation,
        "training": training_diag,
        "law_stress": law_stress_assessments,
        "selection": {
            "selection_split": selection_split,
            "selection_metric": selection_metric,
            "selection_tie_breaker": "mean_aux_oracle_target_abs_error",
            "selected_candidate": selected_candidate,
            "test_metrics_used_for_selection": False,
        },
        "split_policy_metrics": split_policy_metrics,
        "split_violation_rates": split_violation_rates,
        "split_mediation": split_mediation,
        "split_ipw_evaluation": split_ipw_evaluation,
    }

    artifacts: List[GArtifact] = []
    artifact_dir = _lda_artifact_dir(config)
    oracle_artifact = _serialize_law_policy_artifact(
        output_dir=artifact_dir,
        artifact_id="oracle_g",
        name="oracle_true_summary",
        role=PolicyRole.ORACLE_G,
        policy={"kind": "oracle"},
        config=config,
    )
    if oracle_artifact is not None:
        artifacts.append(oracle_artifact)
    baseline_artifact = _serialize_law_policy_artifact(
        output_dir=artifact_dir,
        artifact_id="baseline_g",
        name="infer_identity",
        role=PolicyRole.BASELINE_G,
        policy={"kind": "identity"},
        config=config,
    )
    if baseline_artifact is not None:
        artifacts.append(baseline_artifact)

    candidate_artifacts: Dict[str, GArtifact] = {}
    for name, policy in learned_policies.items():
        artifact = _serialize_law_policy_artifact(
            output_dir=artifact_dir,
            artifact_id=f"candidate_{str(name)}",
            name=str(name),
            role=PolicyRole.CANDIDATE_G,
            policy=policy,
            config=config,
        )
        if artifact is not None:
            artifacts.append(artifact)
            candidate_artifacts[str(name)] = artifact

    learned_artifact: Optional[GArtifact] = None
    if selected_candidate is not None and selected_candidate in learned_policies:
        learned_artifact = _serialize_law_policy_artifact(
            output_dir=artifact_dir,
            artifact_id="learned_g",
            name=str(selected_candidate),
            role=PolicyRole.LEARNED_G,
            policy=learned_policies[selected_candidate],
            config=config,
        )
        if learned_artifact is not None:
            artifacts.append(learned_artifact)

    def _policy_split_payload(name: str) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for split_name, split_payload in split_evals.items():
            metrics = dict(split_payload.get("policy_metrics", {}) or {}).get(name)
            if not isinstance(metrics, dict) or not metrics:
                continue
            out[str(split_name)] = {
                "local_law": _local_law_metrics_from_summary(metrics).to_dict(),
                "downstream": _downstream_metrics_from_summary(metrics).to_dict(),
                "objective": _objective_metrics_from_summary(
                    metrics,
                    objective_spec=objective_spec,
                    config=config,
                ),
            }
        return out

    policies: Dict[str, LocalLawPolicyEvaluation] = {
        "oracle_true_summary": LocalLawPolicyEvaluation(
            name="oracle_true_summary",
            role=PolicyRole.ORACLE_G,
            artifact_id=oracle_artifact.artifact_id if oracle_artifact is not None else None,
            split_metrics=_policy_split_payload("oracle_true_summary"),
            metadata={"policy_kind": "oracle"},
        ),
        "infer_identity": LocalLawPolicyEvaluation(
            name="infer_identity",
            role=PolicyRole.BASELINE_G,
            artifact_id=baseline_artifact.artifact_id if baseline_artifact is not None else None,
            split_metrics=_policy_split_payload("infer_identity"),
            metadata={"policy_kind": "identity"},
        ),
    }
    for name in learned_policies:
        val_metric = float("nan")
        val_payload = dict(split_evals["val"].get("policy_metrics", {}) or {}).get(name)
        if isinstance(val_payload, dict):
            metric_name = str(
                val_payload.get(
                    "selection_metric_name",
                    objective_estimator_alias(str(objective_spec.name), "hajek"),
                )
            )
            val_metric = float(val_payload.get(metric_name, float("nan")))
        policies[str(name)] = LocalLawPolicyEvaluation(
            name=str(name),
            role=PolicyRole.CANDIDATE_G,
            artifact_id=(
                candidate_artifacts[str(name)].artifact_id
                if str(name) in candidate_artifacts
                else None
            ),
            selection_metric_value=val_metric,
            split_metrics=_policy_split_payload(str(name)),
            metadata={"policy_kind": "affine", "selection_split": selection_split},
        )
    if selected_candidate is not None:
        selected_metrics = dict(split_evals["val"].get("policy_metrics", {}) or {}).get(
            selected_candidate,
            {},
        )
        selected_metric_name = str(
            selected_metrics.get(
                "selection_metric_name",
                objective_estimator_alias(str(objective_spec.name), "hajek"),
            )
        )
        policies["learned_g"] = LocalLawPolicyEvaluation(
            name=str(selected_candidate),
            role=PolicyRole.LEARNED_G,
            artifact_id=learned_artifact.artifact_id if learned_artifact is not None else None,
            selection_metric_value=float(selected_metrics.get(selected_metric_name, float("nan"))),
            split_metrics=_policy_split_payload(str(selected_candidate)),
            metadata={
                "selected_candidate": str(selected_candidate),
                "selection_split": selection_split,
                "selection_metric_name": str(selected_metric_name),
                "selection_tie_breaker": "mean_aux_oracle_target_abs_error",
            },
        )
    logged_policy_name = str(selected_candidate or "infer_identity")
    selected_logged_observations = list(
        dict(test_eval.get("logged_observations_by_policy", {}) or {}).get(
            logged_policy_name,
            [],
        )
    )
    logged_observation_artifacts: Dict[str, Any] = {}
    logged_observations_summary = (
        summarize_logged_observations(selected_logged_observations)
        if selected_logged_observations
        else {
            "count": 0,
            "unit_kinds": [],
            "supports_ipw_estimation": False,
            "joint_propensity_min": 1.0,
            "joint_propensity_max": 1.0,
            "joint_propensity_mean": 1.0,
        }
    )
    if selected_logged_observations and artifact_dir is not None and bool(config.save_logged_observations):
        artifact = write_logged_observations_jsonl(
            artifact_dir / "local_law_logged_observations.jsonl",
            selected_logged_observations,
            channel_name="sampled_substructure_supervision",
        )
        logged_observation_artifacts = {artifact.channel_name: artifact.to_dict()}

    learnability_summary = LocalLawRunSummary(
        family="tree_relevant_lda_local_law",
        dgp="leaf_local_mixture_utility",
        oracle_name="oracle_true_summary",
        study_role=str(config.local_law_mode),
        split_ids={
            "train": _lda_split_id(split="train", seed=int(config.seed), n_docs=len(docs_train)),
            "val": _lda_split_id(
                split="val",
                seed=int(config.seed) + int(config.val_seed_offset),
                n_docs=len(docs_val),
            ),
            "test": _lda_split_id(
                split="test",
                seed=int(config.seed) + int(config.test_seed_offset),
                n_docs=len(docs_test),
            ),
        },
        support_budget=SupportBudgetSummary(
            train_docs=int(len(docs_train)),
            val_docs=int(len(docs_val)),
            test_docs=int(len(docs_test)),
            leaf_query_rate=float(config.law_leaf_query_rate),
            internal_query_rate=float(config.law_internal_query_rate),
            mean_leaf_labels_per_doc=float(training_diag.get("leaf_label_count", 0.0))
            / float(max(1, len(docs_train))),
            mean_internal_labels_per_doc=float(training_diag.get("internal_label_count", 0.0))
            / float(max(1, len(docs_train))),
            mean_queries_per_doc=float(
                float(training_diag.get("leaf_label_count", 0.0))
                + float(training_diag.get("internal_label_count", 0.0))
            )
            / float(max(1, len(docs_train))),
            total_queries_estimate=float(
                float(training_diag.get("leaf_label_count", 0.0))
                + float(training_diag.get("internal_label_count", 0.0))
            ),
            metadata={
                "law_leaf_query_design": str(config.law_leaf_query_design),
                "law_internal_query_design": str(config.law_internal_query_design),
                "analysis_partition_mode": str(config.analysis_partition_mode),
            },
        ),
        selection={
            "selection_split": selection_split,
            "selection_metric": selection_metric,
            "selection_tie_breaker": "mean_aux_oracle_target_abs_error",
            "selected_candidate": selected_candidate,
            "test_metrics_used_for_selection": False,
        },
        policies=policies,
        counterexamples=[],
        thresholds={
            "c1": float(config.law_c1_threshold),
            "c2": float(config.law_c2_threshold),
            "c3": float(config.law_c3_threshold),
        },
        suite_role=str(config.suite_role),
        logged_observation_artifacts=logged_observation_artifacts,
        metadata={
            "analysis_partition_mode": str(config.analysis_partition_mode),
            "quadratic_utility_weight": float(config.lambda_multiplier),
            "law_set_id": _lda_public_law_set_id(config),
            "oracle_target_metric": "mean_aux_oracle_target_abs_error",
            "configured_objective_name": str(objective_spec.selection_metric_name),
            "resolved_local_law_weights": {
                str(name): float(value)
                for name, value in dict(objective_spec.local_law_weights).items()
            },
            "objective": objective_spec.to_dict(),
            "logged_observations_policy": logged_policy_name,
            "logged_observations_summary": logged_observations_summary,
        },
    )
    learnability_summary = attach_local_law_learning_problem(learnability_summary)

    methods_patch: Dict[str, object] = {}
    if selected_candidate is not None and selected_candidate in dict(
        test_eval.get("aux_preds", {}) or {}
    ):
        law_arr = np.asarray(
            dict(test_eval.get("aux_preds", {}) or {})[selected_candidate],
            dtype=np.float64,
        )
        law_err = np.abs(law_arr - oracle_arr)
        methods_patch["analysis_infer_law_calibrated_oracle_target"] = _method_metric_dict(
            supervision_kind="local_law_calibrated_analysis_summary",
            budget_regime=str(config.budget_regime),
            n_docs=len(docs_test),
            utility_abs_to_true=law_err.tolist(),
            queried_leaves_per_doc=[
                float(training_diag.get("leaf_label_count", 0.0) / max(1, len(docs_train)))
            ]
            * len(docs_train),
            queried_cost_per_doc=[
                float(training_diag.get("leaf_label_count", 0.0) / max(1, len(docs_train)))
            ]
            * len(docs_train),
            pooled_abs_to_true=pooled_err.tolist(),
            diagnostics={
                "model_kind": "analysis_partition_inference_law_calibrated_oracle_target",
                "calibration_variant": str(selected_candidate),
                "selection_split": selection_split,
            },
        )
    return payload, methods_patch, learnability_summary.to_dict(), artifact_index(artifacts)


def run_leaf_local_mixture_utility_experiment_from_world(
    config: LeafLocalMixtureUtilityConfig,
    world: LeafLocalMixtureUtilityWorld,
) -> LeafLocalMixtureUtilitySummary:
    _validate_config(config)
    if dict(world.signature) != _world_signature(config):
        raise ValueError("config is incompatible with the provided fixed world")
    if int(config.train_docs) > len(world.docs_train):
        raise ValueError("config.train_docs exceeds fixed world train_docs capacity")
    if int(config.val_docs) > len(world.docs_val):
        raise ValueError("config.val_docs exceeds fixed world val_docs capacity")
    if int(config.test_docs) > len(world.docs_test):
        raise ValueError("config.test_docs exceeds fixed world test_docs capacity")

    leaf_meta = _leaf_metadata(config)
    theta_true = np.asarray(world.theta_true, dtype=np.float64)
    W_base = np.asarray(world.W_base, dtype=np.float64)
    docs_train = tuple(world.docs_train[: int(config.train_docs)])
    docs_val = tuple(world.docs_val[: int(config.val_docs)])
    docs_test = tuple(world.docs_test[: int(config.test_docs)])

    heterogeneity_signal_train: List[float] = []
    heterogeneity_signal_test: List[float] = []
    pooled_abs_gap_test: List[float] = []
    partition_gap_test: List[float] = []
    misweight_gap_test: List[float] = []
    inference_tax_test: List[float] = []
    latent_sections_train: List[float] = []
    latent_sections_test: List[float] = []
    analysis_sections_test: List[float] = []

    pooled_preds: List[float] = []
    oracle_true_targets: List[float] = []
    analysis_weighted_oracle_preds: List[float] = []
    analysis_unweighted_oracle_preds: List[float] = []
    analysis_weighted_infer_preds: List[float] = []
    analysis_unweighted_infer_preds: List[float] = []
    leaf_infer_preds: List[float] = []
    leaf_oracle_preds: List[float] = []
    sample_overlap_tokens: List[List[float]] = []
    sample_analysis_weights: List[float] = []

    for doc_idx, doc in enumerate(docs_train):
        latent_sections_train.append(float(len(doc.latent_section_spans)))
        oracle_true = _true_doc_target(
            doc,
            theta=theta_true,
            W_base=W_base,
            lambda_multiplier=float(config.lambda_multiplier),
        )
        pooled_true = _pooled_true_target(
            doc,
            theta=theta_true,
            W_base=W_base,
            lambda_multiplier=float(config.lambda_multiplier),
            doc_tokens=int(config.doc_tokens),
        )
        heterogeneity_signal_train.append(float(oracle_true - pooled_true))

    budgeted_betas, budgeted_training_diag = _train_budgeted_analysis_ridge(
        docs_train,
        world=world,
        config=config,
        theta_true=theta_true,
        W_base=W_base,
    )
    legacy_base_preds, legacy_coarse_preds, legacy_diag = _legacy_ridge_predictions(
        docs_train,
        docs_test,
        world=world,
        config=config,
        theta_true=theta_true,
        W_base=W_base,
    )

    full_ridge_preds: List[float] = []
    budget_naive_preds: List[float] = []
    budget_ipw_preds: List[float] = []
    budget_stab_preds: List[float] = []

    for doc_idx, doc in enumerate(docs_test):
        latent_sections_test.append(float(len(doc.latent_section_spans)))
        view = _build_analysis_partition_view(doc, config, doc_index=doc_idx + 100000)
        analysis_sections_test.append(float(len(view.analysis_section_spans)))
        if doc_idx == 0:
            sample_overlap_tokens = [
                [float(x) for x in row] for row in view.overlap_tokens.tolist()
            ]
            sample_analysis_weights = [float(x) for x in view.analysis_weights]

        counts_full = _base._counts_from_tokens(doc.tokens, vocab_size=int(config.vocab_size))
        pi_doc_hat = _base._infer_topic_mixture_from_counts(
            counts_full,
            topics_phi=world.topics_phi,
            prior_mass=float(config.inference_prior_mass),
            max_iter=int(config.inference_max_iter),
            tol=float(config.inference_tol),
        )
        pooled_pred = float(int(config.doc_tokens)) * _leaf_additive_utility(
            pi_doc_hat,
            theta=theta_true,
            W_base=W_base,
            lambda_multiplier=float(config.lambda_multiplier),
        )
        oracle_true = _true_doc_target(
            doc,
            theta=theta_true,
            W_base=W_base,
            lambda_multiplier=float(config.lambda_multiplier),
        )
        pooled_true = _pooled_true_target(
            doc,
            theta=theta_true,
            W_base=W_base,
            lambda_multiplier=float(config.lambda_multiplier),
            doc_tokens=int(config.doc_tokens),
        )
        analysis_oracle_weighted = _analysis_target(
            view,
            theta=theta_true,
            W_base=W_base,
            lambda_multiplier=float(config.lambda_multiplier),
            doc_tokens=int(config.doc_tokens),
            weighted=True,
        )
        analysis_oracle_unweighted = _analysis_target(
            view,
            theta=theta_true,
            W_base=W_base,
            lambda_multiplier=float(config.lambda_multiplier),
            doc_tokens=int(config.doc_tokens),
            weighted=False,
        )
        analysis_infer_weighted = _infer_analysis_target(
            doc,
            view,
            world=world,
            config=config,
            theta=theta_true,
            W_base=W_base,
            weighted=True,
        )
        analysis_infer_unweighted = _infer_analysis_target(
            doc,
            view,
            world=world,
            config=config,
            theta=theta_true,
            W_base=W_base,
            weighted=False,
        )

        leaf_infer = 0.0
        for span in doc.latent_section_spans:
            pi_hat = _infer_section_topic_from_span(doc, span, world=world, config=config)
            leaf_infer += float(int(span[1]) - int(span[0])) * _leaf_additive_utility(
                pi_hat,
                theta=theta_true,
                W_base=W_base,
                lambda_multiplier=float(config.lambda_multiplier),
            )

        pooled_preds.append(float(pooled_pred))
        oracle_true_targets.append(float(oracle_true))
        analysis_weighted_oracle_preds.append(float(analysis_oracle_weighted))
        analysis_unweighted_oracle_preds.append(float(analysis_oracle_unweighted))
        analysis_weighted_infer_preds.append(float(analysis_infer_weighted))
        analysis_unweighted_infer_preds.append(float(analysis_infer_unweighted))
        leaf_infer_preds.append(float(leaf_infer))
        leaf_oracle_preds.append(float(oracle_true))
        full_ridge_preds.append(
            _predict_analysis_ridge(
                budgeted_betas["analysis_ridge_full_labels"], doc, view, world=world
            )
        )
        budget_naive_preds.append(
            _predict_analysis_ridge(
                budgeted_betas["budgeted_leaf_ridge_naive"], doc, view, world=world
            )
        )
        budget_ipw_preds.append(
            _predict_analysis_ridge(
                budgeted_betas["budgeted_leaf_ridge_ipw"], doc, view, world=world
            )
        )
        budget_stab_preds.append(
            _predict_analysis_ridge(
                budgeted_betas["budgeted_leaf_ridge_ipw_stabilized"], doc, view, world=world
            )
        )

        heterogeneity_signal_test.append(float(oracle_true - pooled_true))
        pooled_abs_gap_test.append(abs(float(pooled_pred) - float(oracle_true)))
        partition_gap_test.append(float(oracle_true - analysis_oracle_weighted))
        misweight_gap_test.append(float(analysis_oracle_weighted - analysis_oracle_unweighted))
        inference_tax_test.append(float(analysis_oracle_weighted - analysis_infer_weighted))

    pooled_arr = np.asarray(pooled_preds, dtype=np.float64)
    oracle_arr = np.asarray(oracle_true_targets, dtype=np.float64)
    analysis_weighted_oracle_arr = np.asarray(analysis_weighted_oracle_preds, dtype=np.float64)
    analysis_unweighted_oracle_arr = np.asarray(analysis_unweighted_oracle_preds, dtype=np.float64)
    analysis_weighted_infer_arr = np.asarray(analysis_weighted_infer_preds, dtype=np.float64)
    analysis_unweighted_infer_arr = np.asarray(analysis_unweighted_infer_preds, dtype=np.float64)
    leaf_infer_arr = np.asarray(leaf_infer_preds, dtype=np.float64)
    leaf_oracle_arr = np.asarray(leaf_oracle_preds, dtype=np.float64)
    full_ridge_arr = np.asarray(full_ridge_preds, dtype=np.float64)
    budget_naive_arr = np.asarray(budget_naive_preds, dtype=np.float64)
    budget_ipw_arr = np.asarray(budget_ipw_preds, dtype=np.float64)
    budget_stab_arr = np.asarray(budget_stab_preds, dtype=np.float64)

    pooled_err = np.abs(pooled_arr - oracle_arr)
    leaf_oracle_err = np.abs(leaf_oracle_arr - oracle_arr)
    leaf_infer_err = np.abs(leaf_infer_arr - oracle_arr)
    analysis_weighted_oracle_err = np.abs(analysis_weighted_oracle_arr - oracle_arr)
    analysis_unweighted_oracle_err = np.abs(analysis_unweighted_oracle_arr - oracle_arr)
    analysis_weighted_infer_err = np.abs(analysis_weighted_infer_arr - oracle_arr)
    analysis_unweighted_infer_err = np.abs(analysis_unweighted_infer_arr - oracle_arr)
    full_ridge_err = np.abs(full_ridge_arr - oracle_arr)
    budget_naive_err = np.abs(budget_naive_arr - oracle_arr)
    budget_ipw_err = np.abs(budget_ipw_arr - oracle_arr)
    budget_stab_err = np.abs(budget_stab_arr - oracle_arr)
    base_err = np.abs(legacy_base_preds - oracle_arr)
    coarse_err = np.abs(legacy_coarse_preds - oracle_arr)

    utility_truth = {
        "relevant_topics": list(
            int(x) for x in world.utility_topic_meta.get("relevant_topics", [])
        ),
        "theta_true": [float(x) for x in theta_true.tolist()],
        "W_base": [[float(x) for x in row] for row in W_base.tolist()],
        "quadratic_utility_weight": float(config.lambda_multiplier),
    }
    heterogeneity = {
        "mean_train_gap_signal": _base._safe_stat(heterogeneity_signal_train, kind="mean"),
        "mean_test_gap_signal": _base._safe_stat(heterogeneity_signal_test, kind="mean"),
        "mean_test_abs_pooled_gap": _base._safe_stat(pooled_abs_gap_test, kind="mean"),
        "mean_test_partition_gap": _base._safe_stat(partition_gap_test, kind="mean"),
        "mean_test_misweight_gap": _base._safe_stat(misweight_gap_test, kind="mean"),
        "mean_test_inference_tax_analysis_weighted": _base._safe_stat(
            inference_tax_test, kind="mean"
        ),
        "gap_signal_correlation_with_pooled_abs_error": (
            float(
                np.corrcoef(
                    np.asarray(heterogeneity_signal_test, dtype=np.float64),
                    np.asarray(pooled_abs_gap_test, dtype=np.float64),
                )[0, 1]
            )
            if len(heterogeneity_signal_test) >= 2
            else float("nan")
        ),
    }

    methods = {
        "pooled_doc_wrong_model": _method_metric_dict(
            supervision_kind="count_ceiling",
            budget_regime=str(config.budget_regime),
            n_docs=len(docs_test),
            utility_abs_to_true=pooled_err.tolist(),
            queried_leaves_per_doc=[0.0] * len(docs_train),
            queried_cost_per_doc=[0.0] * len(docs_train),
            pooled_abs_to_true=pooled_err.tolist(),
            diagnostics={"model_kind": "single_document_mixture"},
        ),
        "leaf_oracle_sum": _method_metric_dict(
            supervision_kind="leaf_scalar_labels",
            budget_regime=str(config.budget_regime),
            n_docs=len(docs_test),
            utility_abs_to_true=leaf_oracle_err.tolist(),
            queried_leaves_per_doc=latent_sections_train,
            queried_cost_per_doc=latent_sections_train,
            pooled_abs_to_true=pooled_err.tolist(),
            diagnostics={"model_kind": "oracle"},
        ),
        "leaf_infer_sum": _method_metric_dict(
            supervision_kind="count_ceiling",
            budget_regime=str(config.budget_regime),
            n_docs=len(docs_test),
            utility_abs_to_true=leaf_infer_err.tolist(),
            queried_leaves_per_doc=[0.0] * len(docs_train),
            queried_cost_per_doc=[0.0] * len(docs_train),
            pooled_abs_to_true=pooled_err.tolist(),
            diagnostics={
                "model_kind": "leaf_topic_inference",
                "leaf_fraction": float(int(config.latent_leaf_tokens) / int(config.doc_tokens)),
                "leaf_fraction_label": leaf_fraction_label(
                    float(int(config.latent_leaf_tokens) / int(config.doc_tokens))
                ),
            },
        ),
        "analysis_oracle_weighted_sum": _method_metric_dict(
            supervision_kind="analysis_partition_oracle",
            budget_regime=str(config.budget_regime),
            n_docs=len(docs_test),
            utility_abs_to_true=analysis_weighted_oracle_err.tolist(),
            queried_leaves_per_doc=analysis_sections_test,
            queried_cost_per_doc=analysis_sections_test,
            pooled_abs_to_true=pooled_err.tolist(),
            diagnostics={
                "model_kind": "analysis_partition_oracle",
                "analysis_partition_mode": str(config.analysis_partition_mode),
                "aggregation_weighting": "token_weighted",
            },
        ),
        "analysis_oracle_unweighted_sum": _method_metric_dict(
            supervision_kind="analysis_partition_oracle",
            budget_regime=str(config.budget_regime),
            n_docs=len(docs_test),
            utility_abs_to_true=analysis_unweighted_oracle_err.tolist(),
            queried_leaves_per_doc=analysis_sections_test,
            queried_cost_per_doc=analysis_sections_test,
            pooled_abs_to_true=pooled_err.tolist(),
            diagnostics={
                "model_kind": "analysis_partition_oracle",
                "analysis_partition_mode": str(config.analysis_partition_mode),
                "aggregation_weighting": "uniform",
            },
        ),
        "analysis_infer_weighted_sum": _method_metric_dict(
            supervision_kind="count_ceiling",
            budget_regime=str(config.budget_regime),
            n_docs=len(docs_test),
            utility_abs_to_true=analysis_weighted_infer_err.tolist(),
            queried_leaves_per_doc=[0.0] * len(docs_train),
            queried_cost_per_doc=[0.0] * len(docs_train),
            pooled_abs_to_true=pooled_err.tolist(),
            diagnostics={
                "model_kind": "analysis_partition_inference",
                "analysis_partition_mode": str(config.analysis_partition_mode),
                "aggregation_weighting": "token_weighted",
            },
        ),
        "analysis_infer_unweighted_sum": _method_metric_dict(
            supervision_kind="count_ceiling",
            budget_regime=str(config.budget_regime),
            n_docs=len(docs_test),
            utility_abs_to_true=analysis_unweighted_infer_err.tolist(),
            queried_leaves_per_doc=[0.0] * len(docs_train),
            queried_cost_per_doc=[0.0] * len(docs_train),
            pooled_abs_to_true=pooled_err.tolist(),
            diagnostics={
                "model_kind": "analysis_partition_inference",
                "analysis_partition_mode": str(config.analysis_partition_mode),
                "aggregation_weighting": "uniform",
            },
        ),
        "analysis_ridge_full_labels": _method_metric_dict(
            supervision_kind="analysis_section_labels",
            budget_regime="all_leaves_labeled",
            n_docs=len(docs_test),
            utility_abs_to_true=full_ridge_err.tolist(),
            queried_leaves_per_doc=analysis_sections_test,
            queried_cost_per_doc=analysis_sections_test,
            pooled_abs_to_true=pooled_err.tolist(),
            diagnostics={
                "model_kind": "ridge_from_analysis_u",
                "query_design": "all",
                "ridge_alpha": float(config.ridge_alpha),
                **supervision_training_contract(
                    representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                    target_kind=TARGET_SCALAR,
                    optimizer_family=OPTIMIZER_FAMILY_CLOSED_FORM_LINEAR,
                    optimizer_backend="closed_form_ridge",
                    n_train_rows=int(budgeted_training_diag["full_labels_rows"]),
                ),
            },
        ),
        "budgeted_leaf_ridge_naive": _method_metric_dict(
            supervision_kind="budgeted_analysis_section_labels",
            budget_regime=str(config.budget_regime),
            n_docs=len(docs_test),
            utility_abs_to_true=budget_naive_err.tolist(),
            queried_leaves_per_doc=[
                float(budgeted_training_diag["mean_sections_queried_train_per_doc"])
            ]
            * len(docs_train),
            queried_cost_per_doc=[
                float(budgeted_training_diag["mean_sections_queried_train_per_doc"])
            ]
            * len(docs_train),
            pooled_abs_to_true=pooled_err.tolist(),
            diagnostics={
                **budgeted_training_diag,
                "model_kind": "ridge_from_analysis_u_budgeted",
                "estimator": "naive",
            },
        ),
        "budgeted_leaf_ridge_ipw": _method_metric_dict(
            supervision_kind="budgeted_analysis_section_labels",
            budget_regime=str(config.budget_regime),
            n_docs=len(docs_test),
            utility_abs_to_true=budget_ipw_err.tolist(),
            queried_leaves_per_doc=[
                float(budgeted_training_diag["mean_sections_queried_train_per_doc"])
            ]
            * len(docs_train),
            queried_cost_per_doc=[
                float(budgeted_training_diag["mean_sections_queried_train_per_doc"])
            ]
            * len(docs_train),
            pooled_abs_to_true=pooled_err.tolist(),
            diagnostics={
                **budgeted_training_diag,
                "model_kind": "ridge_from_analysis_u_budgeted",
                "estimator": "ipw",
            },
        ),
        "budgeted_leaf_ridge_ipw_stabilized": _method_metric_dict(
            supervision_kind="budgeted_analysis_section_labels",
            budget_regime=str(config.budget_regime),
            n_docs=len(docs_test),
            utility_abs_to_true=budget_stab_err.tolist(),
            queried_leaves_per_doc=[
                float(budgeted_training_diag["mean_sections_queried_train_per_doc"])
            ]
            * len(docs_train),
            queried_cost_per_doc=[
                float(budgeted_training_diag["mean_sections_queried_train_per_doc"])
            ]
            * len(docs_train),
            pooled_abs_to_true=pooled_err.tolist(),
            diagnostics={
                **budgeted_training_diag,
                "model_kind": "ridge_from_analysis_u_budgeted",
                "estimator": "ipw_stabilized",
                "stabilized_clip": float(config.ipw_stabilized_clip),
            },
        ),
        "leaf_ridge_from_u": _method_metric_dict(
            supervision_kind="leaf_scalar_labels",
            budget_regime=str(config.budget_regime),
            n_docs=len(docs_test),
            utility_abs_to_true=base_err.tolist(),
            queried_leaves_per_doc=legacy_diag["queried_leaves_base"],
            queried_cost_per_doc=legacy_diag["queried_cost_base"],
            pooled_abs_to_true=pooled_err.tolist(),
            diagnostics={
                "leaf_fraction": float(int(config.latent_leaf_tokens) / int(config.doc_tokens)),
                "leaf_fraction_label": leaf_fraction_label(
                    float(int(config.latent_leaf_tokens) / int(config.doc_tokens))
                ),
                "ridge_feature_dim": int(legacy_diag["beta_base_dim"]),
                "ridge_alpha": float(config.ridge_alpha),
                "training_surface": str(legacy_diag["training_surface"]),
                "supervision_mode": str(legacy_diag["supervision_mode"]),
                "representation_kind": str(legacy_diag["representation_kind"]),
                "target_kind": str(legacy_diag["target_kind"]),
                "optimizer_family": str(legacy_diag["optimizer_family"]),
                "optimizer_backend": str(legacy_diag["optimizer_backend"]),
            },
        ),
        "coarse_leaf_ridge_from_u": _method_metric_dict(
            supervision_kind="leaf_scalar_labels",
            budget_regime=str(config.budget_regime),
            n_docs=len(docs_test),
            utility_abs_to_true=coarse_err.tolist(),
            queried_leaves_per_doc=legacy_diag["queried_leaves_coarse"],
            queried_cost_per_doc=legacy_diag["queried_cost_coarse"],
            pooled_abs_to_true=pooled_err.tolist(),
            diagnostics={
                "leaf_fraction": float(config.leaf_fraction),
                "leaf_fraction_label": leaf_fraction_label(float(config.leaf_fraction)),
                "ridge_feature_dim": int(legacy_diag["beta_coarse_dim"]),
                "ridge_alpha": float(config.ridge_alpha),
                "training_surface": str(legacy_diag["training_surface"]),
                "supervision_mode": str(legacy_diag["supervision_mode"]),
                "representation_kind": str(legacy_diag["representation_kind"]),
                "target_kind": str(legacy_diag["target_kind"]),
                "optimizer_family": str(legacy_diag["optimizer_family"]),
                "optimizer_backend": str(legacy_diag["optimizer_backend"]),
            },
        ),
    }

    eval_doc_rng = np.random.default_rng(int(_splitmix64(int(config.seed) + 11003) & 0xFFFFFFFF))
    section_samples: List[TreeSample] = []
    sampled_doc_ids: List[int] = []
    exact_delta_population: Dict[str, List[float]] = {}
    prediction_errors: Dict[str, np.ndarray] = {
        "analysis_infer_weighted_sum": analysis_weighted_infer_err,
        "analysis_oracle_weighted_sum": analysis_weighted_oracle_err,
        "analysis_oracle_unweighted_sum": analysis_unweighted_oracle_err,
        "analysis_infer_unweighted_sum": analysis_unweighted_infer_err,
        "budgeted_leaf_ridge_naive": budget_naive_err,
        "budgeted_leaf_ridge_ipw": budget_ipw_err,
        "budgeted_leaf_ridge_ipw_stabilized": budget_stab_err,
        "analysis_ridge_full_labels": full_ridge_err,
        "leaf_infer_sum": leaf_infer_err,
        "leaf_ridge_from_u": base_err,
    }
    for method_name, err_arr in prediction_errors.items():
        exact_delta_population[method_name] = (pooled_err - err_arr).tolist()

    for doc_idx, doc in enumerate(docs_test):
        if float(eval_doc_rng.random()) >= float(config.heldout_doc_sample_rate):
            continue
        sampled_doc_ids.append(int(doc_idx))
        view = _build_analysis_partition_view(doc, config, doc_index=doc_idx + 200000)
        spans = view.analysis_section_spans
        scores = _section_proxy_scores(doc, spans, world=world, config=config)
        target_budget = float(config.target_query_budget_per_doc)
        if target_budget <= 0.0:
            target_budget = float(max(1, len(spans)))
        probs = _query_probabilities(
            scores,
            target_budget=target_budget,
            query_design=str(config.query_design),
            floor=float(config.propensity_floor),
            ceiling=float(config.propensity_ceiling),
        )
        for sec_idx, (span, p_sec) in enumerate(zip(spans, probs.tolist())):
            if float(eval_doc_rng.random()) >= float(p_sec):
                continue
            raw_contrib = _true_span_contribution(
                doc,
                span,
                theta=theta_true,
                W_base=W_base,
                lambda_multiplier=float(config.lambda_multiplier),
            )
            section_samples.append(
                TreeSample(
                    doc_id=f"test_{doc_idx}",
                    node_id=f"section_{sec_idx}",
                    node_type=NodeType.LEAF,
                    violation=0,
                    preference_loss=float(raw_contrib),
                    sampling=SamplingMetadata(
                        document_propensity=max(MIN_PROPENSITY, float(config.heldout_doc_sample_rate)),
                        unit_propensity=max(MIN_PROPENSITY, float(p_sec)),
                        unit_kind=ObservationUnitKind.LEAF,
                    ),
                    metadata={"span_tokens": int(span[1] - span[0])},
                )
            )

    target_doc_samples: List[TreeSample] = []
    for doc_idx in sampled_doc_ids:
        raw_value = float(oracle_arr[doc_idx])
        target_doc_samples.append(
            TreeSample(
                doc_id=f"target_{doc_idx}",
                node_id="doc",
                node_type=NodeType.LEAF,
                violation=0,
                preference_loss=raw_value,
                sampling=SamplingMetadata(
                    document_propensity=max(MIN_PROPENSITY, float(config.heldout_doc_sample_rate)),
                    unit_kind=ObservationUnitKind.DOCUMENT,
                ),
            )
        )
    target_eval = _normalized_ci_and_coverage(
        target_doc_samples,
        exact_value=float(np.mean(oracle_arr)),
        raw_values_population=oracle_arr.tolist(),
        population_size=float(len(docs_test)),
        delta=float(config.ipw_delta),
    )
    target_eval["population_exact_mean"] = float(np.mean(oracle_arr))
    target_eval["sampled_doc_count"] = float(len(target_doc_samples))
    target_eval["section_sample_count"] = float(len(section_samples))
    target_eval["section_effective_sample_size"] = float(effective_sample_size(section_samples))
    target_eval["section_max_weight"] = float(max_weight(section_samples))
    target_eval["section_propensity_quantiles"] = _propensity_quantiles(section_samples)
    target_eval["doc_propensity_quantiles"] = _propensity_quantiles(target_doc_samples)

    delta_eval: Dict[str, object] = {}
    for method_name, deltas in exact_delta_population.items():
        doc_samples: List[TreeSample] = []
        for doc_idx in sampled_doc_ids:
            doc_samples.append(
                TreeSample(
                    doc_id=f"delta_{method_name}_{doc_idx}",
                    node_id="doc",
                    node_type=NodeType.LEAF,
                    violation=0,
                    preference_loss=float(deltas[doc_idx]),
                    sampling=SamplingMetadata(
                        document_propensity=max(MIN_PROPENSITY, float(config.heldout_doc_sample_rate)),
                        unit_kind=ObservationUnitKind.DOCUMENT,
                    ),
                )
            )
        delta_eval[method_name] = {
            **_normalized_ci_and_coverage(
                doc_samples,
                exact_value=float(np.mean(np.asarray(deltas, dtype=np.float64))),
                raw_values_population=deltas,
                population_size=float(len(docs_test)),
                delta=float(config.ipw_delta),
            ),
            "population_exact_mean": float(np.mean(np.asarray(deltas, dtype=np.float64))),
            "sampled_doc_count": float(len(doc_samples)),
            "propensity_quantiles": _propensity_quantiles(doc_samples),
        }

    stage3 = {
        "partition_stats": {
            "latent_partition_mode": str(config.latent_partition_mode),
            "latent_length_profile": str(config.latent_length_profile),
            "analysis_partition_mode": str(config.analysis_partition_mode),
            "analysis_leaf_tokens": int(_resolve_analysis_leaf_tokens(config)),
            "mean_latent_sections_train": _base._safe_stat(latent_sections_train, kind="mean"),
            "mean_latent_sections_test": _base._safe_stat(latent_sections_test, kind="mean"),
            "mean_analysis_sections_test": _base._safe_stat(analysis_sections_test, kind="mean"),
            "sample_overlap_tokens": sample_overlap_tokens,
            "sample_analysis_weights": sample_analysis_weights,
        },
        "oracle_decomposition": {
            "mean_true_target": float(np.mean(oracle_arr)),
            "mean_pooled_target": float(
                np.mean(
                    [
                        _pooled_true_target(
                            doc,
                            theta=theta_true,
                            W_base=W_base,
                            lambda_multiplier=float(config.lambda_multiplier),
                            doc_tokens=int(config.doc_tokens),
                        )
                        for doc in docs_test
                    ]
                )
            ),
            "mean_analysis_weighted_target": float(np.mean(analysis_weighted_oracle_arr)),
            "mean_analysis_unweighted_target": float(np.mean(analysis_unweighted_oracle_arr)),
            "mean_structural_gap": _base._safe_stat(heterogeneity_signal_test, kind="mean"),
            "mean_partition_gap": _base._safe_stat(partition_gap_test, kind="mean"),
            "mean_misweight_gap": _base._safe_stat(misweight_gap_test, kind="mean"),
            "mean_analysis_inference_tax": _base._safe_stat(inference_tax_test, kind="mean"),
        },
        "budgeted_training": dict(budgeted_training_diag),
        "ipw_evaluation": {
            "query_design": str(config.query_design),
            "heldout_doc_sample_rate": float(config.heldout_doc_sample_rate),
            "target_query_budget_per_doc": float(
                config.target_query_budget_per_doc
                if float(config.target_query_budget_per_doc) > 0.0
                else max(1, len(sample_analysis_weights))
            ),
            "target": target_eval,
            "delta": delta_eval,
        },
    }
    local_law, local_law_methods, local_law_learnability, g_artifacts = _build_local_law_payload(
        docs_train,
        docs_val,
        docs_test,
        world=world,
        config=config,
        theta_true=theta_true,
        W_base=W_base,
        oracle_arr=oracle_arr,
        pooled_err=pooled_err,
    )
    if local_law_methods:
        methods.update(local_law_methods)

    world_stats = {
        **leaf_meta,
        "train_docs_fit": int(config.train_docs),
        "val_docs_fit": int(config.val_docs),
        "test_docs_evaluated": int(config.test_docs),
        "mean_test_tokens": float(config.doc_tokens),
        "mean_test_base_leaves": _base._safe_stat(latent_sections_test, kind="mean"),
        "mean_test_analysis_sections": _base._safe_stat(analysis_sections_test, kind="mean"),
        "budget_regime": str(config.budget_regime),
        "leaf_label_budget": float(config.leaf_label_budget),
        "train_doc_sample_rate": float(config.doc_sample_rate),
        "heldout_doc_sample_rate": float(config.heldout_doc_sample_rate),
        "query_design": str(config.query_design),
        "target_query_budget_per_doc": float(config.target_query_budget_per_doc),
        "train_split_id": _lda_split_id(
            split="train", seed=int(config.seed), n_docs=len(docs_train)
        ),
        "val_split_id": _lda_split_id(
            split="val",
            seed=int(config.seed) + int(config.val_seed_offset),
            n_docs=len(docs_val),
        ),
        "test_split_id": _lda_split_id(
            split="test",
            seed=int(config.seed) + int(config.test_seed_offset),
            n_docs=len(docs_test),
        ),
    }
    cfg_payload = asdict(config)
    objective_spec_for_config = _local_law_objective_spec(config)
    for legacy_key in (
        "law_package",
        "law_task_objective_weight",
        "law_c1_weight",
        "law_c3_weight",
        "law_c2_proxy_weight",
        "lambda_multiplier",
    ):
        cfg_payload.pop(legacy_key, None)
    cfg_payload["law_set_id"] = _lda_public_law_set_id(config)
    cfg_payload["root_share"] = float(objective_spec_for_config.normalized_task_share())
    cfg_payload["local_law_weight"] = float(objective_spec_for_config.local_law_weight())
    cfg_payload["local_law_component_weights"] = {
        str(name): float(value)
        for name, value in objective_spec_for_config.normalized_local_law_weights().items()
    }
    cfg_payload["quadratic_utility_weight"] = float(config.lambda_multiplier)
    cfg_payload.update(leaf_meta)
    return LeafLocalMixtureUtilitySummary(
        family="leaf_local_mixture_utility",
        target_kind="local_nonlinear_leaf_sum",
        config=cfg_payload,
        topic_meta=dict(world.topic_meta),
        utility_topic_meta=dict(world.utility_topic_meta),
        utility_truth=utility_truth,
        world_stats=world_stats,
        heterogeneity=heterogeneity,
        methods=methods,
        objective=latent_quadratic_utility_objective_semantics(
            name="leaf_local_mixture_utility_target",
            optimized_against="document_level_local_mixture_utility",
            quadratic_utility_weight=float(config.lambda_multiplier),
            linear_component_name="topic_mixture_linear_term",
            interaction_component_name="local_topic_mixture_quadratic_term",
            weighting_scheme="linear_plus_lambda_local_quadratic_utility",
            metadata={
                "problem_id": "leaf_local_mixture_utility",
                "target_kind": "local_nonlinear_leaf_sum",
                "latent_partition_mode": str(config.latent_partition_mode),
                "analysis_partition_mode": str(config.analysis_partition_mode),
            },
        ),
        stage3=stage3,
        local_law=local_law,
        local_law_learnability=local_law_learnability,
        g_artifacts=g_artifacts,
        is_stale_generation=False,
    )


def run_leaf_local_mixture_utility_experiment(
    config: LeafLocalMixtureUtilityConfig,
) -> LeafLocalMixtureUtilitySummary:
    world = sample_leaf_local_mixture_utility_world(config)
    return run_leaf_local_mixture_utility_experiment_from_world(config, world)


__all__ = [
    "AnalysisPartitionView",
    "LeafLocalMixtureDoc",
    "LeafLocalMixtureUtilityConfig",
    "LeafLocalMixtureUtilitySummary",
    "LeafLocalMixtureUtilityWorld",
    "VALID_ANALYSIS_PARTITION_MODES",
    "VALID_BUDGET_REGIMES",
    "VALID_LATENT_LENGTH_PROFILES",
    "VALID_LATENT_PARTITION_MODES",
    "VALID_LAW_INTERNAL_QUERY_DESIGNS",
    "VALID_LOCAL_LAW_MODES",
    "VALID_PROPENSITY_PROXIES",
    "VALID_QUERY_DESIGNS",
    "_base_leaf_utilities",
    "_build_analysis_partition_view",
    "_leaf_additive_utility",
    "_sample_leaf_local_mixture_docs",
    "_true_doc_target",
    "_true_span_contribution",
    "leaf_local_mixture_utility_world_cache_signature",
    "run_leaf_local_mixture_utility_experiment",
    "run_leaf_local_mixture_utility_experiment_from_world",
    "sample_leaf_local_mixture_utility_world",
]
