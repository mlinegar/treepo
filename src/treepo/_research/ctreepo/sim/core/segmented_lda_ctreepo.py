"""
Segmented-LDA end-to-end simulation for ThinkingTrees / C-TreePO.

This module is the "full decomposition" benchmark:

1) Upstream topic recovery / topic-word estimation (Tensor-LDA-inspired options):
   - `topic_phi_estimator="true"` uses the true topic-word matrix `φ`.
   - `topic_phi_estimator="noisy_theory"` perturbs `φ` with magnitude calibrated to a
     Lean-mirrored Thm-5.1-shaped `O(1/√N)` bound (simulation proxy for TLDA rates).
   - `topic_phi_estimator="tensor_lda"` estimates `φ̂` from unlabeled books via centered moments
     + whitening + tensor power method + recentering (batch baseline).
   - `topic_phi_estimator="online_tensor_lda"` estimates `φ̂` via burn-in whitening + STGD-style
     mini-batch factor updates (online baseline).
   - `topic_phi_estimator="sklearn_lda"` fits scikit-learn's variational Bayes LDA on the DTM.
   - `topic_phi_estimator="embedding_spectral"` estimates `φ̂` from shifted-PPMI word embeddings
     (SVD + k-means + soft assignment).
   - `topic_phi_estimator="spectral_numpy"` runs a lightweight spectral proxy on training leaves
     (center + SVD projection + k-means in spectral space).
   - `topic_phi_estimator in {"neural_ctreepo","neural_mergeable_sketch","neural_hybrid","neural_embedding_hybrid"}`
     refines a base estimator with a CPU neural-operator layer using oracle-seeded topics.

2) Midstream summary-learning/calibration error:
   - Learn an affine calibration from queried leaves on training books.

3) Downstream merge/audit error:
   - Tree aggregation over leaf summaries with optional eval-time leaf/internal oracle guidance.

The simulation reports per-policy OPS-style local discrepancy metrics (C1/C3 proxies),
query accounting, selection-bias audit summaries, and an end-to-end triangle decomposition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import itertools
from statistics import fmean
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from treepo._research.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from treepo._research.ctreepo.sim.device_runtime import (
    VALID_DEVICE_MODES,
    configure_torch_runtime,
    resolve_torch_device,
)
from treepo._research.ctreepo.sim.objective_semantics import discrepancy_benchmark_objective_semantics
from treepo._research.training.supervision import (
    AffineSimplexCalibrationConfig,
    DenseSimplexForestModelConfig,
    DenseSimplexTrainingConfig,
    DenseSimplexForestTrainingConfig,
    DenseSimplexModelConfig,
    DenseSupervisionExample,
    OPTIMIZER_FAMILY_AFFINE_VECTOR_CALIBRATION,
    OPTIMIZER_FAMILY_GRADIENT_DENSE,
    OPTIMIZER_FAMILY_TREE_ENSEMBLE,
    REPRESENTATION_DENSE_FEATURE_VECTOR,
    REPRESENTATION_SIMPLEX_VECTOR,
    TARGET_SIMPLEX_VECTOR,
    apply_dense_affine_simplex_calibrator,
    build_dense_full_document_supervision_dataset,
    build_dense_sampled_substructure_supervision_dataset,
    dense_vector_rows_to_numpy,
    fit_dense_affine_simplex_calibrator,
    fit_dense_simplex_forest_regressor,
    fit_dense_simplex_regressor,
    predict_dense_simplex_forest_regressor,
    predict_dense_simplex_regressor,
    supervision_training_contract,
)
from treepo._research.training.config_sections import OptimizerConfig, RunConfig, RuntimeConfig, TrainConfig


from treepo._research.ctreepo.sim.core.segment_lda_ops_weight_recovery import (  # noqa: E402
    VALID_TOPIC_PHI_ESTIMATORS as _OPS_VALID_TOPIC_PHI_ESTIMATORS,
    estimate_topic_distributions,
)


TopicPhiEstimatorName = str
VALID_TOPIC_PHI_ESTIMATORS: Tuple[TopicPhiEstimatorName, ...] = tuple(_OPS_VALID_TOPIC_PHI_ESTIMATORS) + (
    "spectral_numpy",
)
VALID_CALIBRATION_POLICIES: Tuple[str, ...] = ("uniform", "entropy")
VALID_INTERNAL_QUERY_DESIGNS: Tuple[str, ...] = ("none", "uniform", "risk")
VALID_LEAF_THETA_ESTIMATORS: Tuple[str, ...] = ("lstsq", "rf", "mlp", "sklearn_lda")
VALID_TOPIC_PROCESSES: Tuple[str, ...] = ("segments", "bag_of_words")


@dataclass(frozen=True)
class SegmentedLDACtreePOConfig:
    # Core LDA parameters.
    n_topics: int = 5
    vocab_size: int = 600
    alpha_topic: float = 0.20
    beta_word: float = 0.10

    # Segmentation DGP.
    topic_process: str = "segments"  # segments|bag_of_words
    n_books_train: int = 256
    n_books_test: int = 256
    min_segments: int = 8
    max_segments: int = 20
    min_seg_tokens: int = 24
    max_seg_tokens: int = 64
    segment_concentration: float = 80.0
    segment_background: float = 2.0

    # Leaf partition for C-TreePO aggregation.
    fixed_leaf_tokens: int = 32

    # Leaf-theta estimator (word counts -> θ̂ for each leaf).
    leaf_theta_estimator: str = "lstsq"  # lstsq|rf|mlp|sklearn_lda
    leaf_theta_rf_n_estimators: int = 200
    leaf_theta_rf_max_depth: int = 16
    leaf_theta_rf_min_samples_leaf: int = 5
    leaf_theta_mlp_hidden_dim: int = 128
    leaf_theta_mlp_epochs: int = 10
    leaf_theta_mlp_batch_size: int = 256
    leaf_theta_mlp_lr: float = 1e-3
    leaf_theta_mlp_weight_decay: float = 1e-4
    include_full_doc_theta_baseline: bool = False

    # Topic-word estimation (Tensor-LDA-inspired upstream step).
    topic_phi_estimator: TopicPhiEstimatorName = "noisy_theory"
    topic_phi_docs: int = 0  # if <=0, defaults to n_books_train for the estimator's effective N
    tlda_delta: float = 0.10
    tlda_rate_constant: float = 1.0
    tlda_sigmaK_floor: float = 1e-6
    topic_phi_permute: bool = True  # simulate topic unidentifiability (up to permutation)

    # Online Tensor-LDA knobs (used only when topic_phi_estimator="online_tensor_lda").
    online_tensor_lda_burn_in_docs: int = 0  # 0 => auto
    online_tensor_lda_batch_docs: int = 32
    online_tensor_lda_passes: int = 1
    online_tensor_lda_lr: float = 0.1
    online_tensor_lda_grad_clip_norm: float = 1.0

    # Embedding topic estimator knobs (used by embedding_spectral and neural_embedding_hybrid).
    embedding_topic_svd_dim_extra: int = 4
    embedding_topic_kmeans_inits: int = 8
    embedding_topic_kmeans_max_iter: int = 80
    embedding_topic_assignment_temperature: float = 0.35
    embedding_topic_ppmi_shift: float = 1.0

    # Neural topic refiner knobs (used when topic_phi_estimator starts with "neural_").
    neural_topic_base_estimator: str = "tensor_lda"
    neural_topic_seed_fraction: float = 0.35
    neural_topic_hidden_dim: int = 48
    neural_topic_steps: int = 60
    neural_topic_lr: float = 3e-3
    neural_topic_weight_decay: float = 1e-4
    neural_topic_mix_samples: int = 128
    neural_topic_mix_temperature: float = 1.0
    neural_topic_operator_boost: float = 1.4
    neural_topic_seed_llm_min_weight: float = 0.2
    neural_topic_seed_llm_max_weight: float = 0.55
    neural_topic_similarity_temperature: float = 0.15
    neural_topic_ridge: float = 1e-3

    # Lightweight spectral proxy knobs (used only when topic_phi_estimator="spectral_numpy").
    spectral_svd_dim_extra: int = 2
    spectral_max_leaves: int = 4000
    spectral_kmeans_inits: int = 6
    spectral_kmeans_max_iter: int = 60

    # Calibration from queried training leaves.
    calibration_leaf_query_rate: float = 0.10
    calibration_policy: str = "uniform"  # uniform|entropy
    calibration_ridge: float = 1e-4
    calibration_pi_min: float = 0.01

    # Evaluation-time oracle query budgets.
    eval_leaf_query_rate: float = 0.00
    eval_internal_query_rate: float = 0.00
    eval_internal_query_design: str = "none"  # none|uniform|risk

    # OPS discrepancy thresholds.
    c1_threshold: float = 0.20
    c3_threshold: float = 0.20

    # Optional selection-bias audit over internal-node discrepancy population.
    selection_audit_trials: int = 0
    selection_audit_sample_rate: float = 0.10
    selection_audit_pi_min: float = 0.01

    # Runtime.
    device: str = "auto"
    cuda_device: Optional[int] = None
    torch_threads: int = 0

    seed: int = 0


@dataclass(frozen=True)
class SegmentedBook:
    token_words: np.ndarray  # [T]
    token_topics: np.ndarray  # [T]
    boundaries: np.ndarray  # [B], cut-after indices
    book_topic_weights: np.ndarray  # [K]


@dataclass(frozen=True)
class SegmentedCorpus:
    topic_word_true: np.ndarray  # [K, V]
    books: Tuple[SegmentedBook, ...]


@dataclass(frozen=True)
class PolicyMetrics:
    n_books: int
    root_l1_mean: float
    root_l1_median: float
    root_l1_p95: float
    root_l2_mean: float
    c1_violation_rate: float
    c3_violation_rate: float
    mean_leaf_queries: float
    mean_internal_queries: float
    mean_total_queries: float


@dataclass(frozen=True)
class EndToEndDecompositionMetrics:
    n_books: int
    total_root_l1_mean: float
    topic_component_mean: float
    calibration_component_mean: float
    guidance_component_mean: float
    oracle_proxy_component_mean: float
    upper_bound_mean: float
    slack_mean: float


@dataclass(frozen=True)
class EstimatorStats:
    mean: float
    bias: float
    variance: float
    rmse: float


@dataclass(frozen=True)
class SelectionAuditSummary:
    n_units: int
    true_mean_discrepancy: float
    true_violation_rate: float
    trials: int
    target_sample_rate: float
    pi_min: float
    mean_sample_size: float
    mean_effective_sample_size: float
    naive_mean_discrepancy: EstimatorStats
    ipw_mean_discrepancy: EstimatorStats
    dsl0_mean_discrepancy: EstimatorStats
    dsl_oracle_mean_discrepancy: EstimatorStats
    naive_violation_rate: EstimatorStats
    ipw_violation_rate: EstimatorStats
    dsl0_violation_rate: EstimatorStats
    dsl_oracle_violation_rate: EstimatorStats
    ipw_violation_ci_coverage: float
    ipw_violation_ci_mean_radius: float


@dataclass(frozen=True)
class SegmentedLDACtreePOSummary:
    config: Dict[str, object]
    topic_meta: Dict[str, object]
    calibration_samples: int
    metrics: Dict[str, PolicyMetrics]
    decomposition: EndToEndDecompositionMetrics
    selection_audit: Optional[SelectionAuditSummary]
    objective: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "topic_meta": self.topic_meta,
            "calibration_samples": int(self.calibration_samples),
            "metrics": {k: asdict(v) for k, v in self.metrics.items()},
            "decomposition": asdict(self.decomposition),
            "selection_audit": asdict(self.selection_audit) if self.selection_audit is not None else None,
            "objective": self.objective,
        }
        return json.dumps(payload, indent=2, sort_keys=True)


@dataclass
class _TreeNode:
    est: np.ndarray
    truth: np.ndarray
    mass: float


def _safe_mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(fmean(vals))


def _median(xs: Sequence[float]) -> float:
    vals = np.asarray([float(x) for x in xs if math.isfinite(float(x))], dtype=np.float64)
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def _p95(xs: Sequence[float]) -> float:
    vals = np.asarray([float(x) for x in xs if math.isfinite(float(x))], dtype=np.float64)
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, 0.95))


def _l1(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(u, dtype=np.float64) - np.asarray(v, dtype=np.float64))))


def _l2(u: np.ndarray, v: np.ndarray) -> float:
    d = np.asarray(u, dtype=np.float64) - np.asarray(v, dtype=np.float64)
    return float(np.sqrt(np.sum(d * d)))


from treepo._research.ctreepo.sim.util import normalize_simplex_vec, normalize_simplex_rows

_normalize_simplex_vec = normalize_simplex_vec
_normalize_simplex_rows = normalize_simplex_rows


def _inclusion_probs_from_scores(scores: np.ndarray, *, target_rate: float, pi_min: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    target_rate = float(max(pi_min, min(1.0, target_rate)))
    pi_min = float(max(1e-9, min(1.0, pi_min)))
    if scores.size == 0:
        return np.zeros((0,), dtype=np.float64)
    s = scores.copy()
    s -= np.min(s)
    if float(np.max(s)) > 0.0:
        s /= float(np.max(s))
    c = float(max(0.0, target_rate - pi_min))
    pi = pi_min + c * s
    if float(np.mean(pi)) > target_rate:
        lo = 0.0
        hi = max(c, 1.0)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            cur = np.clip(pi_min + mid * s, pi_min, 1.0)
            if float(np.mean(cur)) > target_rate:
                hi = mid
            else:
                lo = mid
        pi = np.clip(pi_min + lo * s, pi_min, 1.0)
    return np.asarray(pi, dtype=np.float64)


def _bernoulli_sample(pi: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    pi = np.asarray(pi, dtype=np.float64)
    return rng.random(size=pi.shape) < pi


def _effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    if s2 <= 0.0:
        return 0.0
    return float((s1 * s1) / s2)


def _estimator_stats(estimates: Sequence[float], *, truth: float) -> EstimatorStats:
    vals = np.asarray([float(x) for x in estimates if math.isfinite(float(x))], dtype=np.float64)
    if vals.size == 0:
        return EstimatorStats(mean=float("nan"), bias=float("nan"), variance=float("nan"), rmse=float("nan"))
    mean = float(np.mean(vals))
    bias = float(mean - truth)
    var = float(np.var(vals))
    rmse = float(np.sqrt(np.mean((vals - truth) ** 2)))
    return EstimatorStats(mean=mean, bias=bias, variance=var, rmse=rmse)


def _validate_config(config: SegmentedLDACtreePOConfig) -> None:
    if config.n_topics < 2:
        raise ValueError("n_topics must be >= 2")
    if config.vocab_size < config.n_topics:
        raise ValueError("vocab_size must be >= n_topics")
    if config.alpha_topic <= 0 or config.beta_word <= 0:
        raise ValueError("alpha_topic and beta_word must be > 0")
    proc = str(config.topic_process).strip().lower()
    if proc not in VALID_TOPIC_PROCESSES:
        raise ValueError(f"topic_process must be one of {VALID_TOPIC_PROCESSES}")
    if config.n_books_train < 1 or config.n_books_test < 1:
        raise ValueError("n_books_train and n_books_test must be >= 1")
    if config.min_segments < 1 or config.max_segments < config.min_segments:
        raise ValueError("invalid segment bounds")
    if config.min_seg_tokens < 2 or config.max_seg_tokens < config.min_seg_tokens:
        raise ValueError("invalid segment token bounds")
    if config.segment_concentration <= 0 or config.segment_background <= 0:
        raise ValueError("segment_concentration and segment_background must be > 0")
    if config.fixed_leaf_tokens < 2:
        raise ValueError("fixed_leaf_tokens must be >= 2")
    leaf_theta = str(config.leaf_theta_estimator).strip().lower()
    if leaf_theta not in VALID_LEAF_THETA_ESTIMATORS:
        raise ValueError(f"leaf_theta_estimator must be one of {VALID_LEAF_THETA_ESTIMATORS}")
    if leaf_theta == "sklearn_lda" and str(config.topic_phi_estimator).strip().lower() != "sklearn_lda":
        raise ValueError("leaf_theta_estimator='sklearn_lda' requires topic_phi_estimator='sklearn_lda'")
    if int(config.leaf_theta_rf_n_estimators) < 1:
        raise ValueError("leaf_theta_rf_n_estimators must be >= 1")
    if int(config.leaf_theta_rf_max_depth) < 1:
        raise ValueError("leaf_theta_rf_max_depth must be >= 1")
    if int(config.leaf_theta_rf_min_samples_leaf) < 1:
        raise ValueError("leaf_theta_rf_min_samples_leaf must be >= 1")
    if int(config.leaf_theta_mlp_hidden_dim) < 1:
        raise ValueError("leaf_theta_mlp_hidden_dim must be >= 1")
    if int(config.leaf_theta_mlp_epochs) < 1:
        raise ValueError("leaf_theta_mlp_epochs must be >= 1")
    if int(config.leaf_theta_mlp_batch_size) < 1:
        raise ValueError("leaf_theta_mlp_batch_size must be >= 1")
    if float(config.leaf_theta_mlp_lr) <= 0:
        raise ValueError("leaf_theta_mlp_lr must be > 0")
    if float(config.leaf_theta_mlp_weight_decay) < 0:
        raise ValueError("leaf_theta_mlp_weight_decay must be >= 0")
    if str(config.topic_phi_estimator) not in VALID_TOPIC_PHI_ESTIMATORS:
        raise ValueError(f"topic_phi_estimator must be one of {VALID_TOPIC_PHI_ESTIMATORS}")
    if config.topic_phi_docs < 0:
        raise ValueError("topic_phi_docs must be >= 0")
    if not (0.0 < float(config.tlda_delta) < 1.0):
        raise ValueError("tlda_delta must be in (0, 1)")
    if float(config.tlda_rate_constant) <= 0:
        raise ValueError("tlda_rate_constant must be > 0")
    if float(config.tlda_sigmaK_floor) <= 0:
        raise ValueError("tlda_sigmaK_floor must be > 0")
    if config.online_tensor_lda_burn_in_docs < 0:
        raise ValueError("online_tensor_lda_burn_in_docs must be >= 0")
    if config.online_tensor_lda_batch_docs < 1:
        raise ValueError("online_tensor_lda_batch_docs must be >= 1")
    if config.online_tensor_lda_passes < 1:
        raise ValueError("online_tensor_lda_passes must be >= 1")
    if float(config.online_tensor_lda_lr) <= 0:
        raise ValueError("online_tensor_lda_lr must be > 0")
    if float(config.online_tensor_lda_grad_clip_norm) <= 0:
        raise ValueError("online_tensor_lda_grad_clip_norm must be > 0")
    if int(config.embedding_topic_svd_dim_extra) < 0:
        raise ValueError("embedding_topic_svd_dim_extra must be >= 0")
    if int(config.embedding_topic_kmeans_inits) < 1:
        raise ValueError("embedding_topic_kmeans_inits must be >= 1")
    if int(config.embedding_topic_kmeans_max_iter) < 1:
        raise ValueError("embedding_topic_kmeans_max_iter must be >= 1")
    if float(config.embedding_topic_assignment_temperature) <= 0:
        raise ValueError("embedding_topic_assignment_temperature must be > 0")
    if float(config.embedding_topic_ppmi_shift) <= 0:
        raise ValueError("embedding_topic_ppmi_shift must be > 0")
    base_est = str(config.neural_topic_base_estimator).strip().lower()
    if base_est not in VALID_TOPIC_PHI_ESTIMATORS:
        raise ValueError(f"neural_topic_base_estimator must be one of {VALID_TOPIC_PHI_ESTIMATORS}")
    if base_est.startswith("neural_"):
        raise ValueError("neural_topic_base_estimator must be a non-neural estimator")
    if not (0.0 < float(config.neural_topic_seed_fraction) <= 1.0):
        raise ValueError("neural_topic_seed_fraction must be in (0, 1]")
    if int(config.neural_topic_hidden_dim) < 1:
        raise ValueError("neural_topic_hidden_dim must be >= 1")
    if int(config.neural_topic_steps) < 1:
        raise ValueError("neural_topic_steps must be >= 1")
    if float(config.neural_topic_lr) <= 0:
        raise ValueError("neural_topic_lr must be > 0")
    if float(config.neural_topic_weight_decay) < 0:
        raise ValueError("neural_topic_weight_decay must be >= 0")
    if int(config.neural_topic_mix_samples) < 0:
        raise ValueError("neural_topic_mix_samples must be >= 0")
    if float(config.neural_topic_mix_temperature) <= 0:
        raise ValueError("neural_topic_mix_temperature must be > 0")
    if float(config.neural_topic_operator_boost) <= 0:
        raise ValueError("neural_topic_operator_boost must be > 0")
    if not (
        0.0 <= float(config.neural_topic_seed_llm_min_weight) <= float(config.neural_topic_seed_llm_max_weight) <= 1.0
    ):
        raise ValueError("neural_topic_seed_llm_min_weight/max_weight must satisfy 0<=min<=max<=1")
    if float(config.neural_topic_similarity_temperature) <= 0:
        raise ValueError("neural_topic_similarity_temperature must be > 0")
    if float(config.neural_topic_ridge) <= 0:
        raise ValueError("neural_topic_ridge must be > 0")
    if config.spectral_svd_dim_extra < 0:
        raise ValueError("spectral_svd_dim_extra must be >= 0")
    if config.spectral_max_leaves < 1:
        raise ValueError("spectral_max_leaves must be >= 1")
    if config.spectral_kmeans_inits < 1:
        raise ValueError("spectral_kmeans_inits must be >= 1")
    if config.spectral_kmeans_max_iter < 1:
        raise ValueError("spectral_kmeans_max_iter must be >= 1")
    if not (0.0 <= config.calibration_leaf_query_rate <= 1.0):
        raise ValueError("calibration_leaf_query_rate must be in [0, 1]")
    if config.calibration_policy not in VALID_CALIBRATION_POLICIES:
        raise ValueError(f"calibration_policy must be one of {VALID_CALIBRATION_POLICIES}")
    if config.calibration_ridge < 0:
        raise ValueError("calibration_ridge must be >= 0")
    if not (0.0 < config.calibration_pi_min <= 1.0):
        raise ValueError("calibration_pi_min must be in (0, 1]")
    if not (0.0 <= config.eval_leaf_query_rate <= 1.0):
        raise ValueError("eval_leaf_query_rate must be in [0, 1]")
    if not (0.0 <= config.eval_internal_query_rate <= 1.0):
        raise ValueError("eval_internal_query_rate must be in [0, 1]")
    if config.eval_internal_query_design not in VALID_INTERNAL_QUERY_DESIGNS:
        raise ValueError(f"eval_internal_query_design must be one of {VALID_INTERNAL_QUERY_DESIGNS}")
    if config.c1_threshold < 0 or config.c3_threshold < 0:
        raise ValueError("c1_threshold and c3_threshold must be >= 0")
    if config.selection_audit_trials < 0:
        raise ValueError("selection_audit_trials must be >= 0")
    if not (0.0 <= config.selection_audit_sample_rate <= 1.0):
        raise ValueError("selection_audit_sample_rate must be in [0, 1]")
    if not (0.0 < config.selection_audit_pi_min <= 1.0):
        raise ValueError("selection_audit_pi_min must be in (0, 1]")
    if str(config.device).strip().lower() not in VALID_DEVICE_MODES:
        raise ValueError(f"device must be one of {VALID_DEVICE_MODES}")
    if int(config.torch_threads) < 0:
        raise ValueError("torch_threads must be >= 0")


def _sample_topic_word_matrix(config: SegmentedLDACtreePOConfig, *, rng: np.random.Generator) -> np.ndarray:
    beta = np.full((int(config.vocab_size),), float(config.beta_word), dtype=np.float64)
    return np.asarray(rng.dirichlet(beta, size=int(config.n_topics)), dtype=np.float64)


def _sample_segmented_book(
    config: SegmentedLDACtreePOConfig,
    *,
    topic_word_true: np.ndarray,
    rng: np.random.Generator,
) -> SegmentedBook:
    k = int(config.n_topics)
    alpha = np.full((k,), float(config.alpha_topic), dtype=np.float64)
    w_book = np.asarray(rng.dirichlet(alpha), dtype=np.float64)

    n_seg = int(rng.integers(int(config.min_segments), int(config.max_segments) + 1))
    seg_lens = rng.integers(int(config.min_seg_tokens), int(config.max_seg_tokens) + 1, size=n_seg, dtype=np.int64)
    seg_lens = [int(x) for x in seg_lens]

    token_words: List[int] = []
    token_topics: List[int] = []
    boundaries: List[int] = []

    proc = str(config.topic_process).strip().lower()
    if proc == "bag_of_words":
        total_len = int(sum(seg_lens))
        if total_len <= 0:
            total_len = int(max(1, int(config.min_seg_tokens)))
        z = np.asarray(rng.choice(np.arange(k), size=total_len, p=w_book), dtype=np.int64)
        token_topics.extend(int(t) for t in z)
        for t in z:
            w = int(rng.choice(np.arange(topic_word_true.shape[1]), p=topic_word_true[int(t)]))
            token_words.append(w)
        # Keep "boundaries" for signature/debugging parity, even though the topic process is i.i.d.
        pos = 0
        for seg_idx, seg_len in enumerate(seg_lens):
            pos += int(seg_len)
            if seg_idx < n_seg - 1:
                boundaries.append(int(pos - 1))
        return SegmentedBook(
            token_words=np.asarray(token_words, dtype=np.int64),
            token_topics=np.asarray(token_topics, dtype=np.int64),
            boundaries=np.asarray(boundaries, dtype=np.int64),
            book_topic_weights=np.asarray(w_book, dtype=np.float64),
        )

    if proc != "segments":
        raise ValueError(f"unknown topic_process: {proc!r}")

    for seg_idx, seg_len in enumerate(seg_lens):
        dominant = int(rng.choice(np.arange(k), p=w_book))
        dir_param = (
            float(config.segment_background) * w_book
            + float(config.segment_concentration) * np.eye(k, dtype=np.float64)[dominant]
            + 1e-9
        )
        theta_seg = np.asarray(rng.dirichlet(dir_param), dtype=np.float64)

        z = np.asarray(rng.choice(np.arange(k), size=seg_len, p=theta_seg), dtype=np.int64)
        token_topics.extend(int(t) for t in z)
        for t in z:
            w = int(rng.choice(np.arange(topic_word_true.shape[1]), p=topic_word_true[int(t)]))
            token_words.append(w)

        if seg_idx < n_seg - 1:
            boundaries.append(len(token_words) - 1)

    return SegmentedBook(
        token_words=np.asarray(token_words, dtype=np.int64),
        token_topics=np.asarray(token_topics, dtype=np.int64),
        boundaries=np.asarray(boundaries, dtype=np.int64),
        book_topic_weights=np.asarray(w_book, dtype=np.float64),
    )


def _generate_segmented_corpus(
    config: SegmentedLDACtreePOConfig,
    *,
    topic_word_true: np.ndarray,
    n_books: int,
    rng: np.random.Generator,
) -> SegmentedCorpus:
    books = tuple(_sample_segmented_book(config, topic_word_true=topic_word_true, rng=rng) for _ in range(int(n_books)))
    return SegmentedCorpus(topic_word_true=np.asarray(topic_word_true, dtype=np.float64), books=books)


def _corpus_signature(corpus: SegmentedCorpus) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(corpus.topic_word_true, dtype=np.float64).tobytes())
    for book in corpus.books:
        h.update(np.asarray(book.token_words, dtype=np.int64).tobytes())
        h.update(np.asarray(book.token_topics, dtype=np.int64).tobytes())
        h.update(np.asarray(book.boundaries, dtype=np.int64).tobytes())
        h.update(np.asarray(book.book_topic_weights, dtype=np.float64).tobytes())
    return h.hexdigest()


def _leaf_spans(n_tokens: int, *, leaf_tokens: int) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    i = 0
    while i < int(n_tokens):
        j = min(int(n_tokens), i + int(leaf_tokens))
        spans.append((int(i), int(j)))
        i = j
    if not spans:
        spans = [(0, 0)]
    return spans


def _span_topic_theta(token_topics: np.ndarray, *, start: int, end: int, n_topics: int) -> np.ndarray:
    if int(end) <= int(start):
        return np.full((int(n_topics),), 1.0 / float(n_topics), dtype=np.float64)
    z = np.asarray(token_topics[int(start) : int(end)], dtype=np.int64)
    c = np.bincount(z, minlength=int(n_topics)).astype(np.float64)
    return _normalize_simplex_vec(c)


def _span_word_counts(token_words: np.ndarray, *, start: int, end: int, vocab_size: int) -> np.ndarray:
    if int(end) <= int(start):
        return np.zeros((int(vocab_size),), dtype=np.float64)
    w = np.asarray(token_words[int(start) : int(end)], dtype=np.int64)
    c = np.bincount(w, minlength=int(vocab_size)).astype(np.float64)
    return np.asarray(c, dtype=np.float64)


def _estimate_theta_from_counts(counts: np.ndarray, *, topic_word_est: np.ndarray) -> np.ndarray:
    x = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(x))
    k = int(topic_word_est.shape[0])
    if total <= 0.0:
        return np.full((k,), 1.0 / float(k), dtype=np.float64)
    freq = x / total
    raw, *_ = np.linalg.lstsq(topic_word_est.T, freq, rcond=None)
    return _normalize_simplex_vec(np.asarray(raw, dtype=np.float64))


def _collect_train_leaf_count_matrix(
    books: Sequence[SegmentedBook],
    *,
    vocab_size: int,
    leaf_tokens: int,
    max_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    rows: List[np.ndarray] = []
    for book in books:
        spans = _leaf_spans(len(book.token_words), leaf_tokens=leaf_tokens)
        for (s, e) in spans:
            rows.append(_span_word_counts(book.token_words, start=s, end=e, vocab_size=vocab_size))
    if not rows:
        return np.zeros((0, int(vocab_size)), dtype=np.float64)
    x = np.asarray(rows, dtype=np.float64)
    n = int(x.shape[0])
    if n > int(max_rows):
        idx = rng.choice(np.arange(n, dtype=np.int64), size=int(max_rows), replace=False)
        x = np.asarray(x[idx], dtype=np.float64)
    return x


def _kmeans_lloyd(
    x: np.ndarray,
    *,
    k: int,
    n_init: int,
    max_iter: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if n == 0:
        return np.zeros((int(k), int(d)), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    best_inertia = float("inf")
    best_centers = np.zeros((int(k), int(d)), dtype=np.float64)
    best_labels = np.zeros((n,), dtype=np.int64)

    for _ in range(int(max(1, n_init))):
        if n >= int(k):
            init_ids = rng.choice(np.arange(n, dtype=np.int64), size=int(k), replace=False)
        else:
            init_ids = rng.choice(np.arange(n, dtype=np.int64), size=int(k), replace=True)
        centers = np.asarray(x[init_ids], dtype=np.float64).copy()
        labels_prev: Optional[np.ndarray] = None

        for _it in range(int(max(1, max_iter))):
            # Squared Euclidean distances.
            dist2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dist2, axis=1).astype(np.int64)
            if labels_prev is not None and np.array_equal(labels, labels_prev):
                break
            labels_prev = labels

            for j in range(int(k)):
                idx = np.where(labels == j)[0]
                if idx.size == 0:
                    centers[j] = x[int(rng.integers(0, n))]
                else:
                    centers[j] = np.mean(x[idx], axis=0)

        final_dist2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        inertia = float(np.sum(np.min(final_dist2, axis=1)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = np.asarray(centers, dtype=np.float64).copy()
            best_labels = np.argmin(final_dist2, axis=1).astype(np.int64)

    return best_centers, best_labels


def _estimate_topic_word_matrix_spectral_numpy(
    config: SegmentedLDACtreePOConfig,
    *,
    train_books: Sequence[SegmentedBook],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, object]]:
    k = int(config.n_topics)
    v = int(config.vocab_size)
    x_counts = _collect_train_leaf_count_matrix(
        train_books,
        vocab_size=v,
        leaf_tokens=int(config.fixed_leaf_tokens),
        max_rows=int(config.spectral_max_leaves),
        rng=rng,
    )
    spectral_meta: Dict[str, object] = {
        "topic_phi_estimator": "spectral_numpy",
        "spectral_numpy_leaf_rows": int(x_counts.shape[0]),
        "spectral_numpy_svd_dim_extra": int(config.spectral_svd_dim_extra),
        "spectral_numpy_max_leaves": int(config.spectral_max_leaves),
        "spectral_numpy_kmeans_inits": int(config.spectral_kmeans_inits),
        "spectral_numpy_kmeans_max_iter": int(config.spectral_kmeans_max_iter),
    }
    if x_counts.shape[0] == 0:
        return np.full((k, v), 1.0 / float(v), dtype=np.float64), spectral_meta

    row_sum = np.sum(x_counts, axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, 1.0)
    x = x_counts / row_sum
    m1 = np.mean(x, axis=0)
    xc = x - m1[None, :]

    if float(np.linalg.norm(xc)) < 1e-12:
        noisy = np.maximum(m1[None, :] + rng.normal(0.0, 1e-6, size=(k, v)), 1e-12)
        return _normalize_simplex_rows(noisy), spectral_meta

    d = int(
        min(
            max(1, k + int(config.spectral_svd_dim_extra)),
            xc.shape[0],
            xc.shape[1],
        )
    )
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    del u
    sd = np.asarray(s[:d], dtype=np.float64)
    vd = np.asarray(vt[:d, :], dtype=np.float64)
    eps = 1e-8

    x_proj = xc @ vd.T
    x_white = x_proj / np.maximum(sd[None, :], eps)
    centers_w, _labels = _kmeans_lloyd(
        x_white,
        k=k,
        n_init=int(config.spectral_kmeans_inits),
        max_iter=int(config.spectral_kmeans_max_iter),
        rng=rng,
    )
    centers_proj = centers_w * np.maximum(sd[None, :], eps)
    topics = centers_proj @ vd + m1[None, :]
    topics = np.maximum(topics, 1e-12)
    spectral_meta["spectral_numpy_svd_dim"] = int(d)
    return _normalize_simplex_rows(topics), spectral_meta


def _best_topic_permutation_l2(
    topics_est: Sequence[np.ndarray],
    topics_true: Sequence[np.ndarray],
) -> Tuple[Tuple[int, ...], np.ndarray]:
    """
    Find σ : est_index ↦ true_index minimizing Σ_i ||φ̂_i - φ_{σ(i)}||₂.

    Returns (perm, cost_matrix) where perm[i]=σ(i).
    """

    k = int(len(topics_true))
    if int(len(topics_est)) != k:
        raise ValueError("topics_est and topics_true must have same length")
    if k <= 0:
        return (tuple(), np.zeros((0, 0), dtype=np.float64))

    est = np.stack([np.asarray(t, dtype=np.float64).reshape(-1) for t in topics_est], axis=0)
    tru = np.stack([np.asarray(t, dtype=np.float64).reshape(-1) for t in topics_true], axis=0)
    if est.shape != tru.shape:
        raise ValueError("topics_est and topics_true must have aligned shapes")

    cost = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        diff = tru[None, :, :] - est[i : i + 1, None, :]
        cost[i] = np.linalg.norm(diff.reshape(k, -1), axis=1)

    # Exact assignment via brute force for small K.
    if k <= 9:
        best_perm: Tuple[int, ...] = tuple(range(k))
        best = float("inf")
        for perm in itertools.permutations(range(k)):
            total = 0.0
            for i, j in enumerate(perm):
                total += float(cost[i, j])
                if total >= best:
                    break
            if total < best:
                best = total
                best_perm = tuple(int(x) for x in perm)
        return best_perm, cost

    # Greedy fallback for large K (kept simple; this is a simulation helper).
    remaining = set(range(k))
    perm_out: List[int] = [-1 for _ in range(k)]
    for i in range(k):
        j = min(remaining, key=lambda jj: float(cost[i, jj]))
        perm_out[i] = int(j)
        remaining.remove(j)
    return (tuple(perm_out), cost)


def _invert_perm(perm: Sequence[int]) -> Tuple[int, ...]:
    k = int(len(perm))
    inv = [-1 for _ in range(k)]
    for i, j in enumerate(perm):
        jj = int(j)
        if jj < 0 or jj >= k:
            raise ValueError("perm must be a bijection on [0,K)")
        inv[jj] = int(i)
    if any(x < 0 for x in inv):
        raise ValueError("perm must be a bijection on [0,K)")
    return tuple(int(x) for x in inv)


def _estimate_topic_word_matrix(
    config: SegmentedLDACtreePOConfig,
    *,
    topic_word_true: np.ndarray,
    train_books: Sequence[SegmentedBook],
    n_train_docs: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, object]]:
    est = str(config.topic_phi_estimator).strip().lower()
    topics_true = [np.asarray(row, dtype=np.float64).reshape(-1) for row in np.asarray(topic_word_true, dtype=np.float64)]
    k = int(len(topics_true))
    if k <= 0:
        raise ValueError("need at least one topic")

    # Spectral proxy (leaf-count SVD + kmeans).
    if est == "spectral_numpy":
        topic_word_est, spectral_meta = _estimate_topic_word_matrix_spectral_numpy(config, train_books=train_books, rng=rng)
        topics_est = [np.asarray(row, dtype=np.float64).reshape(-1) for row in np.asarray(topic_word_est, dtype=np.float64)]
        perm_est_to_true, cost = _best_topic_permutation_l2(topics_est, topics_true)
        aligned_err = np.asarray([float(cost[i, perm_est_to_true[i]]) for i in range(k)], dtype=np.float64)
        meta: Dict[str, object] = {
            **spectral_meta,
            "topic_phi_perm_est_to_true": [int(x) for x in perm_est_to_true],
            "topic_phi_l2_error_mean": float(np.mean(aligned_err)) if aligned_err.size else 0.0,
            "topic_phi_l2_error_p95": float(np.percentile(aligned_err, 95.0)) if aligned_err.size else 0.0,
            "topic_phi_l2_error_max": float(np.max(aligned_err)) if aligned_err.size else 0.0,
        }
        inv = _invert_perm(perm_est_to_true)
        aligned = np.asarray(topic_word_est, dtype=np.float64)[np.asarray(inv, dtype=np.int64)]
        return aligned, meta

    # Tensor-LDA / noisy-theory / oracle baselines via shared estimator.
    phi_docs_effective = int(config.topic_phi_docs) if int(config.topic_phi_docs) > 0 else int(n_train_docs)
    phi_docs_effective = int(max(0, phi_docs_effective))

    docs_phi: List[np.ndarray] = [np.asarray(b.token_words, dtype=np.int64) for b in train_books]
    phi_extra = int(max(0, phi_docs_effective - len(docs_phi)))
    for _ in range(phi_extra):
        extra = _sample_segmented_book(config, topic_word_true=np.asarray(topic_word_true, dtype=np.float64), rng=rng)
        docs_phi.append(np.asarray(extra.token_words, dtype=np.int64))
    docs_phi = docs_phi[:phi_docs_effective]

    topics_est, meta_raw, perm_est_to_true = estimate_topic_distributions(
        topics_true,
        estimator=est,
        n_docs=int(max(1, phi_docs_effective)) if est != "true" else int(phi_docs_effective),
        doc_topic_concentration=float(config.alpha_topic),
        tlda_delta=float(config.tlda_delta),
        tlda_rate_constant=float(config.tlda_rate_constant),
        sigmaK_floor=float(config.tlda_sigmaK_floor),
        permute=bool(config.topic_phi_permute),
        seed=int(rng.integers(0, 2**31 - 1)),
        topic_word_concentration=float(config.beta_word),
        docs_tokens=[d.tolist() for d in docs_phi] if docs_phi else None,
        online_burn_in_docs=int(config.online_tensor_lda_burn_in_docs),
        online_batch_docs=int(config.online_tensor_lda_batch_docs),
        online_passes=int(config.online_tensor_lda_passes),
        online_lr=float(config.online_tensor_lda_lr),
        online_grad_clip_norm=float(config.online_tensor_lda_grad_clip_norm),
        embedding_svd_dim_extra=int(config.embedding_topic_svd_dim_extra),
        embedding_kmeans_inits=int(config.embedding_topic_kmeans_inits),
        embedding_kmeans_max_iter=int(config.embedding_topic_kmeans_max_iter),
        embedding_assignment_temperature=float(config.embedding_topic_assignment_temperature),
        embedding_ppmi_shift=float(config.embedding_topic_ppmi_shift),
        neural_base_estimator=str(config.neural_topic_base_estimator),
        neural_seed_fraction=float(config.neural_topic_seed_fraction),
        neural_hidden_dim=int(config.neural_topic_hidden_dim),
        neural_steps=int(config.neural_topic_steps),
        neural_lr=float(config.neural_topic_lr),
        neural_weight_decay=float(config.neural_topic_weight_decay),
        neural_mix_samples=int(config.neural_topic_mix_samples),
        neural_mix_temperature=float(config.neural_topic_mix_temperature),
        neural_operator_boost=float(config.neural_topic_operator_boost),
        neural_seed_min_weight=float(config.neural_topic_seed_llm_min_weight),
        neural_seed_max_weight=float(config.neural_topic_seed_llm_max_weight),
        neural_similarity_temperature=float(config.neural_topic_similarity_temperature),
        neural_ridge=float(config.neural_topic_ridge),
        device=str(config.device),
        cuda_device=config.cuda_device,
        torch_threads=int(config.torch_threads),
    )
    meta: Dict[str, object] = dict(meta_raw)
    meta["topic_phi_perm_est_to_true"] = [int(x) for x in perm_est_to_true]

    inv = _invert_perm(perm_est_to_true)
    aligned_topics = tuple(np.asarray(topics_est[int(i)], dtype=np.float64).reshape(-1) for i in inv)
    topic_word_est = np.stack(aligned_topics, axis=0).astype(np.float64, copy=False)
    return topic_word_est, meta


def _docs_to_count_matrix(docs_tokens: Sequence[Sequence[int]], *, vocab_size: int) -> np.ndarray:
    v = int(vocab_size)
    X = np.zeros((len(docs_tokens), v), dtype=np.int64)
    for i, doc in enumerate(docs_tokens):
        w = np.asarray(list(doc), dtype=np.int64).reshape(-1)
        if w.size:
            X[i] = np.bincount(w, minlength=v).astype(np.int64, copy=False)
    return X


def _fit_sklearn_lda_topic_model(
    config: SegmentedLDACtreePOConfig,
    *,
    topic_word_true: np.ndarray,
    train_books: Sequence[SegmentedBook],
    n_train_docs: int,
    rng: np.random.Generator,
    max_iter: int = 60,
) -> Tuple[object, np.ndarray, Dict[str, object], Tuple[int, ...]]:
    """Fit scikit-learn LDA and return (model, aligned_phi, topic_meta, inv_perm_true_to_est)."""

    try:
        from sklearn.decomposition import LatentDirichletAllocation  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for leaf_theta_estimator='sklearn_lda'. "
            "Install with: pip install scikit-learn>=1.4.2"
        ) from e

    topics_true = [np.asarray(row, dtype=np.float64).reshape(-1) for row in np.asarray(topic_word_true, dtype=np.float64)]
    k = int(len(topics_true))
    if k <= 0:
        raise ValueError("need at least one topic")
    v = int(topics_true[0].size)

    phi_docs_effective = int(config.topic_phi_docs) if int(config.topic_phi_docs) > 0 else int(n_train_docs)
    phi_docs_effective = int(max(0, phi_docs_effective))

    docs_phi: List[np.ndarray] = [np.asarray(b.token_words, dtype=np.int64) for b in train_books]
    phi_extra = int(max(0, phi_docs_effective - len(docs_phi)))
    for _ in range(phi_extra):
        extra = _sample_segmented_book(config, topic_word_true=np.asarray(topic_word_true, dtype=np.float64), rng=rng)
        docs_phi.append(np.asarray(extra.token_words, dtype=np.int64))
    docs_phi = docs_phi[:phi_docs_effective]
    docs_list: List[List[int]] = [d.reshape(-1).astype(np.int64).tolist() for d in docs_phi]

    X = _docs_to_count_matrix(docs_list, vocab_size=v)
    D = int(X.shape[0])
    if D <= 0:
        raise ValueError("sklearn_lda requires at least one document")

    iters = int(max_iter)
    if iters < 1:
        raise ValueError("max_iter must be >= 1")

    lda = LatentDirichletAllocation(
        n_components=int(k),
        max_iter=int(iters),
        learning_method="batch",
        evaluate_every=-1,
        random_state=int(rng.integers(0, 2**31 - 1)),
        n_jobs=1,
        doc_topic_prior=float(config.alpha_topic),
        topic_word_prior=float(config.beta_word),
    )
    lda.fit(X)

    comps = np.asarray(getattr(lda, "components_"), dtype=np.float64)
    if comps.shape != (k, v):
        raise RuntimeError("sklearn_lda components_ shape mismatch")
    comps = np.clip(comps, 1e-12, None)
    topic_word = comps / np.sum(comps, axis=1, keepdims=True)
    topics_est = [np.asarray(topic_word[i], dtype=np.float64).reshape(-1) for i in range(k)]

    perm_est_to_true, cost = _best_topic_permutation_l2(topics_est, topics_true)
    aligned_err = np.asarray([float(cost[i, perm_est_to_true[i]]) for i in range(k)], dtype=np.float64)

    inv = _invert_perm(perm_est_to_true)
    aligned_phi = np.asarray(topic_word, dtype=np.float64)[np.asarray(inv, dtype=np.int64)]
    meta: Dict[str, object] = {
        "topic_phi_estimator": "sklearn_lda",
        "topic_phi_docs_effective": float(D),
        "sklearn_lda_max_iter": float(iters),
        "sklearn_lda_n_iter": float(getattr(lda, "n_iter_", float("nan"))),
        "sklearn_lda_doc_topic_prior": float(config.alpha_topic),
        "sklearn_lda_topic_word_prior": float(config.beta_word),
        "topic_phi_perm_est_to_true": [int(x) for x in perm_est_to_true],
        "topic_phi_l2_error_mean": float(np.mean(aligned_err)) if aligned_err.size else 0.0,
        "topic_phi_l2_error_p95": float(np.percentile(aligned_err, 95.0)) if aligned_err.size else 0.0,
        "topic_phi_l2_error_max": float(np.max(aligned_err)) if aligned_err.size else 0.0,
    }
    return lda, np.asarray(aligned_phi, dtype=np.float64), meta, inv


def _predict_leaf_thetas_sklearn_lda(
    lda_model: object,
    book: SegmentedBook,
    *,
    leaf_tokens: int,
    vocab_size: int,
    inv_perm_true_to_est: Tuple[int, ...],
) -> np.ndarray:
    spans = _leaf_spans(len(book.token_words), leaf_tokens=int(leaf_tokens))
    if not spans:
        return np.zeros((0, int(len(inv_perm_true_to_est))), dtype=np.float64)

    X = np.stack(
        [_span_word_counts(book.token_words, start=s, end=e, vocab_size=int(vocab_size)) for (s, e) in spans],
        axis=0,
    ).astype(np.int64, copy=False)
    raw = getattr(lda_model, "transform")(X)  # [n_leaves, K] in estimator topic order
    raw = np.asarray(raw, dtype=np.float64)
    aligned = raw[:, np.asarray(inv_perm_true_to_est, dtype=np.int64)]
    return _normalize_simplex_rows(np.maximum(aligned, 0.0))


def _sample_leaf_query_mask(
    proxy_leaf_thetas: np.ndarray,
    *,
    rate: float,
    policy: str,
    pi_min: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    b, c, _ = proxy_leaf_thetas.shape
    rate = float(max(0.0, min(1.0, rate)))
    pi_min = float(max(1e-9, min(1.0, pi_min)))
    if rate <= 0.0:
        return np.zeros((b, c), dtype=bool), np.zeros((b, c), dtype=np.float64)
    if policy == "uniform":
        pi = np.full((b, c), float(max(rate, pi_min)), dtype=np.float64)
    elif policy == "entropy":
        p = np.clip(np.asarray(proxy_leaf_thetas, dtype=np.float64), 1e-12, 1.0)
        entropy = -np.sum(p * np.log(p), axis=2)
        pi = _inclusion_probs_from_scores(entropy.reshape(-1), target_rate=rate, pi_min=pi_min).reshape(b, c)
    else:
        raise ValueError(f"unknown calibration policy: {policy}")
    return np.asarray(_bernoulli_sample(pi, rng=rng), dtype=bool), np.asarray(pi, dtype=np.float64)


def _counts_to_freq_rows(counts: np.ndarray) -> np.ndarray:
    x = np.asarray(counts, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("counts must be 2D [n, vocab]")
    s = np.sum(x, axis=1, keepdims=True)
    s = np.maximum(s, 1e-12)
    return np.asarray(x / s, dtype=np.float64)


def _full_doc_theta_supervision_dataset(
    books: Sequence[SegmentedBook],
    *,
    split: str,
    n_topics: int,
    vocab_size: int,
) -> "SupervisionDataset":
    rows: List[DenseSupervisionExample] = []
    rubric = (
        "Predict the full-document topic mixture from a dense document representation."
    )
    for idx, book in enumerate(books):
        doc_id = f"{split}_doc_{idx}"
        rows.append(
            DenseSupervisionExample(
                example_id=doc_id,
                features=_counts_to_freq_rows(
                    _span_word_counts(
                        book.token_words,
                        start=0,
                        end=len(book.token_words),
                        vocab_size=int(vocab_size),
                    ).reshape(1, -1)
                )[0].tolist(),
                vector_target=_aggregate_root_truth(book, n_topics=int(n_topics)).tolist(),
                original_text=f"segmented_lda_ctreepo::{doc_id}",
                rubric=rubric,
                response="single_full_document_candidate",
                response_id=f"{doc_id}:single_candidate",
                reference_score=0.0,
                source_doc_id=doc_id,
                truth_label_source="oracle",
                metadata={
                    "dgp": "segmented_lda_ctreepo",
                    "input_view": "single_full_document_bow",
                    "uses_tree_merges": False,
                    "n_topics": int(n_topics),
                },
            )
        )
    return build_dense_full_document_supervision_dataset(
        rows,
        application_name="segmented_lda_ctreepo",
        supervision_signal_name="document_level_target",
        response_signal_name="document_topic_mixture",
        law_type="document_level_target",
        split=str(split),
        response_signal_min=0.0,
        response_signal_max=1.0,
        metadata={
            "dgp": "segmented_lda_ctreepo",
            "input_view": "single_full_document_bow",
            "uses_tree_merges": False,
            "target_structure": "simplex",
            "n_topics": int(n_topics),
        },
    )


def _leaf_calibration_supervision_dataset(
    proxy_leaf_thetas: np.ndarray,
    true_leaf_thetas: np.ndarray,
    queried_mask_pad: np.ndarray,
    *,
    inclusion_probs_pad: Optional[np.ndarray],
    split: str,
    query_policy: str,
) -> "SupervisionDataset":
    rows: List[DenseSupervisionExample] = []
    proxy = _normalize_simplex_rows(np.asarray(proxy_leaf_thetas, dtype=np.float64).reshape(-1, proxy_leaf_thetas.shape[2]))
    truth = _normalize_simplex_rows(np.asarray(true_leaf_thetas, dtype=np.float64).reshape(-1, true_leaf_thetas.shape[2]))
    proxy = proxy.reshape(proxy_leaf_thetas.shape)
    truth = truth.reshape(true_leaf_thetas.shape)
    mask = np.asarray(queried_mask_pad, dtype=bool)
    pi = (
        np.asarray(inclusion_probs_pad, dtype=np.float64)
        if inclusion_probs_pad is not None
        else np.ones_like(mask, dtype=np.float64)
    )
    if mask.ndim != 2 or pi.ndim != 2:
        raise ValueError("queried_mask_pad and inclusion_probs_pad must be 2D [books, leaves]")
    n_topics = int(proxy.shape[2])
    rubric = "Calibrate sampled proxy leaf topic mixtures to oracle leaf topic mixtures."
    for i in range(int(proxy.shape[0])):
        if i >= mask.shape[0]:
            break
        p_book = proxy[i]
        t_book = truth[i]
        if p_book.shape != t_book.shape:
            raise ValueError("proxy and truth calibration rows must align")
        m = mask[i, : p_book.shape[0]]
        p = np.clip(pi[i, : p_book.shape[0]], 1e-9, 1.0)
        if not np.any(m):
            continue
        for j in np.nonzero(m)[0].tolist():
            doc_id = f"{split}_book_{i}"
            leaf_id = f"{doc_id}:leaf_{int(j)}"
            rows.append(
                DenseSupervisionExample(
                    example_id=f"{leaf_id}:affine_calibration",
                    features=p_book[int(j)].tolist(),
                    vector_target=t_book[int(j)].tolist(),
                    original_text=f"segmented_lda_ctreepo::{doc_id}",
                    rubric=rubric,
                    response="sampled_proxy_leaf_candidate",
                    response_id=f"{leaf_id}:proxy",
                    unit_kind=ObservationUnitKind.LEAF,
                    reference_score=0.0,
                    source_doc_id=doc_id,
                    truth_label_source="oracle",
                    sampling=SamplingMetadata(
                        document_propensity=1.0,
                        unit_propensity=float(p[int(j)]),
                        label_propensity=1.0,
                        joint_propensity=float(p[int(j)]),
                        sampling_scheme="sampled_substructure_supervision",
                        policy_name=str(query_policy),
                        unit_kind=ObservationUnitKind.LEAF,
                        supports_ipw_estimation=True,
                        metadata={
                            "split": str(split),
                            "leaf_index": int(j),
                            "query_policy": str(query_policy),
                            "calibration_stage": "affine",
                        },
                    ),
                    metadata={
                        "dgp": "segmented_lda_ctreepo",
                        "input_view": "proxy_leaf_topic_mixture",
                        "uses_tree_merges": False,
                        "n_topics": n_topics,
                        "leaf_index": int(j),
                        "query_policy": str(query_policy),
                        "calibration_stage": "affine",
                    },
                )
            )
    return build_dense_sampled_substructure_supervision_dataset(
        rows,
        application_name="segmented_lda_ctreepo",
        supervision_signal_name="substructure_level_target",
        response_signal_name="leaf_topic_mixture",
        law_type="sufficiency",
        split=str(split),
        response_signal_min=0.0,
        response_signal_max=1.0,
        metadata={
            "dgp": "segmented_lda_ctreepo",
            "input_view": "proxy_leaf_topic_mixture",
            "uses_tree_merges": False,
            "target_structure": "simplex",
            "n_topics": n_topics,
            "query_policy": str(query_policy),
            "calibration_stage": "affine",
        },
    )


def _leaf_theta_supervision_dataset(
    leaf_counts: Sequence[np.ndarray],
    leaf_truth: Sequence[np.ndarray],
    queried_mask_pad: np.ndarray,
    *,
    inclusion_probs_pad: Optional[np.ndarray],
    split: str,
    n_topics: int,
    query_policy: str,
) -> "SupervisionDataset":
    rows: List[DenseSupervisionExample] = []
    mask = np.asarray(queried_mask_pad, dtype=bool)
    pi = (
        np.asarray(inclusion_probs_pad, dtype=np.float64)
        if inclusion_probs_pad is not None
        else np.ones_like(mask, dtype=np.float64)
    )
    if mask.ndim != 2 or pi.ndim != 2:
        raise ValueError("queried_mask_pad and inclusion_probs_pad must be 2D [books, leaves]")
    rubric = "Predict the sampled leaf topic mixture from a dense leaf representation."
    for i, (counts, truth) in enumerate(zip(leaf_counts, leaf_truth)):
        c = _counts_to_freq_rows(np.asarray(counts, dtype=np.float64))
        t = _normalize_simplex_rows(np.asarray(truth, dtype=np.float64))
        if c.shape[0] != t.shape[0]:
            raise ValueError("counts and truth leaf rows must align")
        if i >= mask.shape[0]:
            break
        m = mask[i, : c.shape[0]]
        p = np.clip(pi[i, : c.shape[0]], 1e-9, 1.0)
        if not np.any(m):
            continue
        for j in np.nonzero(m)[0].tolist():
            doc_id = f"{split}_book_{i}"
            leaf_id = f"{doc_id}:leaf_{int(j)}"
            rows.append(
                DenseSupervisionExample(
                    example_id=leaf_id,
                    features=c[int(j)].tolist(),
                    vector_target=t[int(j)].tolist(),
                    original_text=f"segmented_lda_ctreepo::{doc_id}",
                    rubric=rubric,
                    response="sampled_leaf_candidate",
                    response_id=leaf_id,
                    unit_kind=ObservationUnitKind.LEAF,
                    reference_score=0.0,
                    source_doc_id=doc_id,
                    truth_label_source="oracle",
                    sampling=SamplingMetadata(
                        document_propensity=1.0,
                        unit_propensity=float(p[int(j)]),
                        label_propensity=1.0,
                        joint_propensity=float(p[int(j)]),
                        sampling_scheme="sampled_substructure_supervision",
                        policy_name=str(query_policy),
                        unit_kind=ObservationUnitKind.LEAF,
                        supports_ipw_estimation=True,
                        metadata={
                            "split": str(split),
                            "leaf_index": int(j),
                            "query_policy": str(query_policy),
                        },
                    ),
                    metadata={
                        "dgp": "segmented_lda_ctreepo",
                        "input_view": "sampled_leaf_bow",
                        "uses_tree_merges": False,
                        "n_topics": int(n_topics),
                        "leaf_index": int(j),
                        "query_policy": str(query_policy),
                    },
                )
            )
    return build_dense_sampled_substructure_supervision_dataset(
        rows,
        application_name="segmented_lda_ctreepo",
        supervision_signal_name="substructure_level_target",
        response_signal_name="leaf_topic_mixture",
        law_type="sufficiency",
        split=str(split),
        response_signal_min=0.0,
        response_signal_max=1.0,
        metadata={
            "dgp": "segmented_lda_ctreepo",
            "input_view": "sampled_leaf_bow",
            "uses_tree_merges": False,
            "target_structure": "simplex",
            "n_topics": int(n_topics),
            "query_policy": str(query_policy),
        },
    )


def _predict_leaf_theta_model(model: object, counts: np.ndarray) -> np.ndarray:
    x = _counts_to_freq_rows(np.asarray(counts, dtype=np.float64))
    try:
        import torch
    except Exception:
        torch = None  # type: ignore[assignment]

    if torch is not None:
        # torch model
        if hasattr(model, "parameters") and hasattr(model, "__call__"):
            try:
                model_device = next(model.parameters()).device
            except Exception:
                model_device = torch.device("cpu")
            with torch.no_grad():
                pred = (
                    model(torch.tensor(np.asarray(x, dtype=np.float32), device=model_device))
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
            return _normalize_simplex_rows(np.maximum(pred, 0.0))

    # sklearn-style model
    if hasattr(model, "predict"):
        pred = np.asarray(model.predict(np.asarray(x, dtype=np.float32)), dtype=np.float64)
        return _normalize_simplex_rows(np.maximum(pred, 0.0))

    raise TypeError(f"unsupported leaf theta model type: {type(model)!r}")


def _eval_full_doc_theta_predictions(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    c1_threshold: float,
    c3_threshold: float,
) -> PolicyMetrics:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError("predicted and true full-doc theta arrays must align")
    root_l1 = [_l1(p[i], t[i]) for i in range(int(p.shape[0]))]
    root_l2 = [_l2(p[i], t[i]) for i in range(int(p.shape[0]))]
    return _build_policy_metrics(
        root_l1=root_l1,
        root_l2=root_l2,
        c1_errors=[],
        c3_errors=[],
        leaf_queries=[0.0] * int(p.shape[0]),
        internal_queries=[0.0] * int(p.shape[0]),
        c1_threshold=float(c1_threshold),
        c3_threshold=float(c3_threshold),
    )


def _reduce_balanced_tree_with_guidance(
    leaf_est: np.ndarray,
    leaf_truth: np.ndarray,
    *,
    leaf_query_rate: float,
    internal_query_rate: float,
    internal_query_design: str,
    rng: np.random.Generator,
    leaf_masses: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[float], List[float], int, int, List[float], List[float]]:
    c, _k = leaf_est.shape
    leaf_query_rate = float(max(0.0, min(1.0, leaf_query_rate)))
    internal_query_rate = float(max(0.0, min(1.0, internal_query_rate)))
    if leaf_masses is None:
        masses = np.ones((c,), dtype=np.float64)
    else:
        masses = np.asarray(leaf_masses, dtype=np.float64).reshape(-1)
        if masses.shape[0] != c:
            raise ValueError("leaf_masses must align with leaf_est/leaf_truth rows")
        masses = np.clip(masses, 1e-12, None)

    leaf_mask = rng.random(size=(c,)) < leaf_query_rate
    est = np.asarray(leaf_est, dtype=np.float64).copy()
    est[leaf_mask] = np.asarray(leaf_truth, dtype=np.float64)[leaf_mask]

    c1_errors = [_l1(est[i], leaf_truth[i]) for i in range(c)]
    leaf_queries = int(np.sum(leaf_mask))
    internal_queries = 0
    c3_errors: List[float] = []
    internal_population_errors: List[float] = []
    internal_population_scores: List[float] = []

    nodes: List[_TreeNode] = [
        _TreeNode(est=est[i].copy(), truth=np.asarray(leaf_truth[i], dtype=np.float64).copy(), mass=float(masses[i]))
        for i in range(c)
    ]

    while len(nodes) > 1:
        next_nodes: List[_TreeNode] = []
        merged_flags: List[bool] = []
        merge_scores: List[float] = []

        i = 0
        while i < len(nodes):
            if i + 1 >= len(nodes):
                next_nodes.append(nodes[i])
                merged_flags.append(False)
                merge_scores.append(float("-inf"))
                i += 1
                continue

            left = nodes[i]
            right = nodes[i + 1]
            n = float(left.mass + right.mass)
            est_merge = (left.est * float(left.mass) + right.est * float(right.mass)) / float(n)
            truth_merge = (left.truth * float(left.mass) + right.truth * float(right.mass)) / float(n)
            pre_err = _l1(est_merge, truth_merge)
            score = _l1(left.est, right.est)
            internal_population_errors.append(pre_err)
            internal_population_scores.append(score)

            next_nodes.append(_TreeNode(est=est_merge, truth=truth_merge, mass=n))
            merged_flags.append(True)
            merge_scores.append(score)
            i += 2

        candidate_ids = [idx for idx, flag in enumerate(merged_flags) if flag]
        n_candidates = len(candidate_ids)
        selected: set[int] = set()
        if n_candidates > 0 and internal_query_rate > 0.0 and internal_query_design != "none":
            q = int(round(internal_query_rate * float(n_candidates)))
            q = max(0, min(n_candidates, q))
            if q > 0:
                if internal_query_design == "uniform":
                    chosen = rng.choice(np.asarray(candidate_ids, dtype=np.int64), size=q, replace=False)
                    selected = {int(x) for x in np.asarray(chosen, dtype=np.int64)}
                elif internal_query_design == "risk":
                    ranked = sorted(candidate_ids, key=lambda idx: float(merge_scores[idx]), reverse=True)
                    selected = set(ranked[:q])
                else:
                    raise ValueError(f"unknown internal_query_design: {internal_query_design}")

        for idx in candidate_ids:
            node = next_nodes[idx]
            if idx in selected:
                node.est = node.truth.copy()
                internal_queries += 1
            c3_errors.append(_l1(node.est, node.truth))

        nodes = next_nodes

    root_est = _normalize_simplex_vec(nodes[0].est)
    return (
        root_est,
        c1_errors,
        c3_errors,
        leaf_queries,
        internal_queries,
        internal_population_errors,
        internal_population_scores,
    )


def _violation_rate(errs: Sequence[float], *, threshold: float) -> float:
    vals = [float(x) for x in errs if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=np.float64) > float(threshold)))


def _extract_leaf_arrays(
    books: Sequence[SegmentedBook],
    *,
    n_topics: int,
    vocab_size: int,
    leaf_tokens: int,
    topic_word_est: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    # Returns per-book arrays:
    # - leaf_truth_theta[b]: [L_b, K]
    # - leaf_est_theta[b]: [L_b, K]
    # - leaf_counts[b]: [L_b, V]
    all_truth: List[np.ndarray] = []
    all_est: List[np.ndarray] = []
    all_counts: List[np.ndarray] = []
    for book in books:
        spans = _leaf_spans(len(book.token_words), leaf_tokens=leaf_tokens)
        truth_list: List[np.ndarray] = []
        est_list: List[np.ndarray] = []
        counts_list: List[np.ndarray] = []
        for (s, e) in spans:
            theta_truth = _span_topic_theta(book.token_topics, start=s, end=e, n_topics=n_topics)
            wc = _span_word_counts(book.token_words, start=s, end=e, vocab_size=vocab_size)
            theta_est = _estimate_theta_from_counts(wc, topic_word_est=topic_word_est)
            truth_list.append(theta_truth)
            est_list.append(theta_est)
            counts_list.append(wc)
        all_truth.append(np.asarray(truth_list, dtype=np.float64))
        all_est.append(np.asarray(est_list, dtype=np.float64))
        all_counts.append(np.asarray(counts_list, dtype=np.float64))
    return all_truth, all_est, all_counts


def _aggregate_root_truth(book: SegmentedBook, *, n_topics: int) -> np.ndarray:
    return _span_topic_theta(book.token_topics, start=0, end=len(book.token_topics), n_topics=n_topics)


def _build_policy_metrics(
    *,
    root_l1: Sequence[float],
    root_l2: Sequence[float],
    c1_errors: Sequence[float],
    c3_errors: Sequence[float],
    leaf_queries: Sequence[float],
    internal_queries: Sequence[float],
    c1_threshold: float,
    c3_threshold: float,
) -> PolicyMetrics:
    tot = [float(a + b) for a, b in zip(leaf_queries, internal_queries)]
    return PolicyMetrics(
        n_books=len(root_l1),
        root_l1_mean=_safe_mean(root_l1),
        root_l1_median=_median(root_l1),
        root_l1_p95=_p95(root_l1),
        root_l2_mean=_safe_mean(root_l2),
        c1_violation_rate=_violation_rate(c1_errors, threshold=float(c1_threshold)),
        c3_violation_rate=_violation_rate(c3_errors, threshold=float(c3_threshold)),
        mean_leaf_queries=_safe_mean(leaf_queries),
        mean_internal_queries=_safe_mean(internal_queries),
        mean_total_queries=_safe_mean(tot),
    )


def _run_selection_bias_audit(
    *,
    discrepancies: np.ndarray,
    violations: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    trials: int,
    sample_rate: float,
    pi_min: float,
    seed: int,
) -> SelectionAuditSummary:
    rng = np.random.default_rng(int(seed))
    disc = np.asarray(discrepancies, dtype=np.float64)
    viol = np.asarray(violations, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n = int(disc.size)
    if n == 0:
        nan_stats = EstimatorStats(mean=float("nan"), bias=float("nan"), variance=float("nan"), rmse=float("nan"))
        return SelectionAuditSummary(
            n_units=0,
            true_mean_discrepancy=float("nan"),
            true_violation_rate=float("nan"),
            trials=int(trials),
            target_sample_rate=float(sample_rate),
            pi_min=float(pi_min),
            mean_sample_size=float("nan"),
            mean_effective_sample_size=float("nan"),
            naive_mean_discrepancy=nan_stats,
            ipw_mean_discrepancy=nan_stats,
            dsl0_mean_discrepancy=nan_stats,
            dsl_oracle_mean_discrepancy=nan_stats,
            naive_violation_rate=nan_stats,
            ipw_violation_rate=nan_stats,
            dsl0_violation_rate=nan_stats,
            dsl_oracle_violation_rate=nan_stats,
            ipw_violation_ci_coverage=float("nan"),
            ipw_violation_ci_mean_radius=float("nan"),
        )

    pi = _inclusion_probs_from_scores(scores, target_rate=float(sample_rate), pi_min=float(pi_min))
    truth_mu = float(np.mean(disc))
    truth_p = float(np.mean(viol > float(threshold)))
    viol01 = (viol > float(threshold)).astype(np.float64)

    naive_mu: List[float] = []
    ipw_mu: List[float] = []
    dsl0_mu: List[float] = []
    dsl_oracle_mu: List[float] = []

    naive_p: List[float] = []
    ipw_p: List[float] = []
    dsl0_p: List[float] = []
    dsl_oracle_p: List[float] = []

    sample_sizes: List[float] = []
    ess_vals: List[float] = []
    ci_covered: List[float] = []
    ci_radius: List[float] = []

    pred0_mu = np.zeros_like(disc)
    pred0_p = np.zeros_like(viol01)
    pred_oracle_mu = disc.copy()
    pred_oracle_p = viol01.copy()

    for _ in range(int(trials)):
        idx = _bernoulli_sample(pi, rng=rng)
        m = int(np.sum(idx))
        if m <= 0:
            naive_mu.append(float("nan"))
            ipw_mu.append(float("nan"))
            dsl0_mu.append(float("nan"))
            dsl_oracle_mu.append(float("nan"))
            naive_p.append(float("nan"))
            ipw_p.append(float("nan"))
            dsl0_p.append(float("nan"))
            dsl_oracle_p.append(float("nan"))
            sample_sizes.append(0.0)
            ess_vals.append(0.0)
            ci_covered.append(float("nan"))
            ci_radius.append(float("nan"))
            continue

        w = idx.astype(np.float64) / pi
        ess = _effective_sample_size(w)
        sample_sizes.append(float(m))
        ess_vals.append(float(ess))

        naive_mu_t = float(np.mean(disc[idx]))
        naive_p_t = float(np.mean(viol01[idx]))
        ipw_mu_t = float(np.sum(w * disc) / float(n))
        ipw_p_t = float(np.sum(w * viol01) / float(n))
        dsl0_mu_t = float(np.mean(pred0_mu) + np.sum(w * (disc - pred0_mu)) / float(n))
        dsl0_p_t = float(np.mean(pred0_p) + np.sum(w * (viol01 - pred0_p)) / float(n))
        dsl_oracle_mu_t = float(np.mean(pred_oracle_mu) + np.sum(w * (disc - pred_oracle_mu)) / float(n))
        dsl_oracle_p_t = float(np.mean(pred_oracle_p) + np.sum(w * (viol01 - pred_oracle_p)) / float(n))

        naive_mu.append(naive_mu_t)
        ipw_mu.append(ipw_mu_t)
        dsl0_mu.append(dsl0_mu_t)
        dsl_oracle_mu.append(dsl_oracle_mu_t)
        naive_p.append(naive_p_t)
        ipw_p.append(ipw_p_t)
        dsl0_p.append(dsl0_p_t)
        dsl_oracle_p.append(dsl_oracle_p_t)

        rad = float(1.96 * math.sqrt(max(ipw_p_t * (1.0 - ipw_p_t), 1e-9) / max(ess, 1e-9)))
        ci_radius.append(rad)
        ci_covered.append(float(abs(ipw_p_t - truth_p) <= rad))

    return SelectionAuditSummary(
        n_units=int(n),
        true_mean_discrepancy=truth_mu,
        true_violation_rate=truth_p,
        trials=int(trials),
        target_sample_rate=float(sample_rate),
        pi_min=float(pi_min),
        mean_sample_size=_safe_mean(sample_sizes),
        mean_effective_sample_size=_safe_mean(ess_vals),
        naive_mean_discrepancy=_estimator_stats(naive_mu, truth=truth_mu),
        ipw_mean_discrepancy=_estimator_stats(ipw_mu, truth=truth_mu),
        dsl0_mean_discrepancy=_estimator_stats(dsl0_mu, truth=truth_mu),
        dsl_oracle_mean_discrepancy=_estimator_stats(dsl_oracle_mu, truth=truth_mu),
        naive_violation_rate=_estimator_stats(naive_p, truth=truth_p),
        ipw_violation_rate=_estimator_stats(ipw_p, truth=truth_p),
        dsl0_violation_rate=_estimator_stats(dsl0_p, truth=truth_p),
        dsl_oracle_violation_rate=_estimator_stats(dsl_oracle_p, truth=truth_p),
        ipw_violation_ci_coverage=_safe_mean(ci_covered),
        ipw_violation_ci_mean_radius=_safe_mean(ci_radius),
    )


def run_segmented_lda_ctreepo_simulation(
    config: SegmentedLDACtreePOConfig,
) -> SegmentedLDACtreePOSummary:
    _validate_config(config)
    seed = int(config.seed)
    seed_topic = int(seed)
    seed_train = int(seed) + 10_000
    seed_test = int(seed) + 20_000
    seed_sim = int(seed) + 30_000
    rng_topic = np.random.default_rng(int(seed_topic))
    rng_train = np.random.default_rng(int(seed_train))
    rng_test = np.random.default_rng(int(seed_test))
    rng = np.random.default_rng(int(seed_sim))

    topic_word_true = _sample_topic_word_matrix(config, rng=rng_topic)
    train = _generate_segmented_corpus(
        config, topic_word_true=topic_word_true, n_books=int(config.n_books_train), rng=rng_train
    )
    test = _generate_segmented_corpus(
        config, topic_word_true=topic_word_true, n_books=int(config.n_books_test), rng=rng_test
    )
    test_sig = _corpus_signature(test)

    leaf_theta_mode = str(config.leaf_theta_estimator).strip().lower()
    lda_model: Optional[object] = None
    inv_perm_true_to_est: Optional[Tuple[int, ...]] = None
    if leaf_theta_mode == "sklearn_lda":
        if str(config.topic_phi_estimator).strip().lower() != "sklearn_lda":
            raise ValueError("leaf_theta_estimator='sklearn_lda' requires topic_phi_estimator='sklearn_lda'")
        lda_model, topic_word_est, topic_meta, inv_perm_true_to_est = _fit_sklearn_lda_topic_model(
            config,
            topic_word_true=topic_word_true,
            train_books=train.books,
            n_train_docs=int(config.n_books_train),
            rng=rng,
        )
    else:
        topic_word_est, topic_meta = _estimate_topic_word_matrix(
            config,
            topic_word_true=topic_word_true,
            train_books=train.books,
            n_train_docs=int(config.n_books_train),
            rng=rng,
        )
    topic_meta = dict(topic_meta)
    topic_meta["corpus_seed_topic"] = int(seed_topic)
    topic_meta["corpus_seed_train"] = int(seed_train)
    topic_meta["corpus_seed_test"] = int(seed_test)
    topic_meta["corpus_seed_sim"] = int(seed_sim)
    topic_meta["corpus_signature_test"] = str(test_sig)

    # Leaf arrays for train/test under estimated topics and oracle topics.
    train_truth, train_est_lstsq, train_counts = _extract_leaf_arrays(
        train.books,
        n_topics=int(config.n_topics),
        vocab_size=int(config.vocab_size),
        leaf_tokens=int(config.fixed_leaf_tokens),
        topic_word_est=topic_word_est,
    )
    test_truth, test_est_lstsq, test_counts = _extract_leaf_arrays(
        test.books,
        n_topics=int(config.n_topics),
        vocab_size=int(config.vocab_size),
        leaf_tokens=int(config.fixed_leaf_tokens),
        topic_word_est=topic_word_est,
    )
    _test_truth2, test_oracle_proxy, _test_counts2 = _extract_leaf_arrays(
        test.books,
        n_topics=int(config.n_topics),
        vocab_size=int(config.vocab_size),
        leaf_tokens=int(config.fixed_leaf_tokens),
        topic_word_est=topic_word_true,
    )

    # Choose the initial proxy leaf-theta estimator (lstsq or sklearn_lda).
    train_proxy_list: List[np.ndarray] = list(train_est_lstsq)
    test_proxy_list: List[np.ndarray] = list(test_est_lstsq)
    if leaf_theta_mode == "sklearn_lda":
        if lda_model is None or inv_perm_true_to_est is None:
            raise RuntimeError("sklearn_lda leaf-theta requested but model is missing")
        train_proxy_list = [
            _predict_leaf_thetas_sklearn_lda(
                lda_model,
                b,
                leaf_tokens=int(config.fixed_leaf_tokens),
                vocab_size=int(config.vocab_size),
                inv_perm_true_to_est=inv_perm_true_to_est,
            )
            for b in train.books
        ]
        test_proxy_list = [
            _predict_leaf_thetas_sklearn_lda(
                lda_model,
                b,
                leaf_tokens=int(config.fixed_leaf_tokens),
                vocab_size=int(config.vocab_size),
                inv_perm_true_to_est=inv_perm_true_to_est,
            )
            for b in test.books
        ]

    # Build train tensors (ragged -> padded stack for query sampling).
    max_train_leaves = max(arr.shape[0] for arr in train_proxy_list)
    k = int(config.n_topics)
    train_proxy_pad = np.zeros((len(train_est_lstsq), max_train_leaves, k), dtype=np.float64)
    train_truth_pad = np.zeros((len(train_truth), max_train_leaves, k), dtype=np.float64)
    train_mask = np.zeros((len(train_est_lstsq), max_train_leaves), dtype=bool)
    for i, (a, b) in enumerate(zip(train_proxy_list, train_truth)):
        l = a.shape[0]
        train_proxy_pad[i, :l] = a
        train_truth_pad[i, :l] = b
        train_mask[i, :l] = True

    query_mask_pad, pi_train = _sample_leaf_query_mask(
        train_proxy_pad,
        rate=float(config.calibration_leaf_query_rate),
        policy=str(config.calibration_policy),
        pi_min=float(config.calibration_pi_min),
        rng=rng,
    )
    query_mask_pad = query_mask_pad & train_mask

    leaf_theta_meta: Dict[str, object] = {
        "leaf_theta_estimator": str(leaf_theta_mode),
        "leaf_theta_train_samples": 0,
        "leaf_theta_fallback": "",
    }
    train_est = list(train_proxy_list)
    test_est = list(test_proxy_list)
    full_doc_policy_metrics: Dict[str, PolicyMetrics] = {}
    full_doc_theta_meta: Dict[str, object] = {}
    if leaf_theta_mode in {"rf", "mlp"}:
        leaf_supervision = _leaf_theta_supervision_dataset(
            train_counts,
            train_truth,
            query_mask_pad,
            inclusion_probs_pad=pi_train,
            split="train",
            n_topics=int(config.n_topics),
            query_policy=str(config.calibration_policy),
        )
        leaf_rows = leaf_supervision.to_dense_vector_training_records(law_type="sufficiency")
        leaf_theta_meta["leaf_theta_train_samples"] = int(len(leaf_rows))
        if int(len(leaf_rows)) <= 0:
            leaf_theta_meta["leaf_theta_fallback"] = "lstsq_no_labels"
        else:
            model: object
            if leaf_theta_mode == "rf":
                model, fit_result = fit_dense_simplex_forest_regressor(
                    leaf_supervision,
                    config=DenseSimplexForestTrainingConfig(
                        model=DenseSimplexForestModelConfig(
                            n_estimators=int(config.leaf_theta_rf_n_estimators),
                            max_depth=int(config.leaf_theta_rf_max_depth),
                            min_samples_leaf=int(config.leaf_theta_rf_min_samples_leaf),
                        ),
                        run=RunConfig(seed=int(seed) + 40_000),
                    ),
                )
                fit_meta = {
                    "leaf_theta_model": "rf",
                    "leaf_theta_rf_n_estimators": int(config.leaf_theta_rf_n_estimators),
                    "leaf_theta_rf_max_depth": int(config.leaf_theta_rf_max_depth),
                    "leaf_theta_rf_min_samples_leaf": int(config.leaf_theta_rf_min_samples_leaf),
                    "leaf_theta_device_requested": "cpu",
                    "leaf_theta_device_used": "cpu",
                    **supervision_training_contract(
                        representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                        target_kind=TARGET_SIMPLEX_VECTOR,
                        optimizer_family=OPTIMIZER_FAMILY_TREE_ENSEMBLE,
                        optimizer_backend="random_forest",
                        selection_mode=str(fit_result.selection_mode),
                        selection_split=str(fit_result.selection_split),
                        selection_metric_name=str(fit_result.selection_metric_name),
                        selection_metric_value=float(fit_result.selection_metric_value),
                        best_epoch=int(fit_result.best_epoch),
                        n_train_rows=int(fit_result.n_train_rows),
                    ),
                }
            else:
                try:
                    import torch
                except Exception as e:  # pragma: no cover
                    raise ImportError(
                        "PyTorch is required for leaf_theta_estimator='mlp'. "
                        "Install with: pip install torch>=2.0.0"
                    ) from e
                configure_torch_runtime(torch, torch_threads=int(config.torch_threads))
                leaf_theta_device, leaf_theta_runtime_meta = resolve_torch_device(
                    torch_module=torch,
                    device=str(config.device),
                    cuda_device=config.cuda_device,
                )
                model, fit_result = fit_dense_simplex_regressor(
                    leaf_supervision,
                    config=DenseSimplexTrainingConfig(
                        model=DenseSimplexModelConfig(
                            hidden_dims=(int(config.leaf_theta_mlp_hidden_dim),),
                        ),
                        optimizer=OptimizerConfig(
                            learning_rate=float(config.leaf_theta_mlp_lr),
                            weight_decay=float(config.leaf_theta_mlp_weight_decay),
                        ),
                        train=TrainConfig(
                            batch_size=int(config.leaf_theta_mlp_batch_size),
                            epochs=int(config.leaf_theta_mlp_epochs),
                        ),
                        runtime=RuntimeConfig(
                            device=str(leaf_theta_device),
                            bf16=False,
                            gradient_checkpointing=False,
                        ),
                        run=RunConfig(seed=int(seed) + 40_000),
                    ),
                )
                fit_meta = {
                    "leaf_theta_model": "mlp",
                    "leaf_theta_mlp_hidden_dim": int(config.leaf_theta_mlp_hidden_dim),
                    "leaf_theta_mlp_epochs": int(config.leaf_theta_mlp_epochs),
                    "leaf_theta_mlp_batch_size": int(config.leaf_theta_mlp_batch_size),
                    "leaf_theta_mlp_lr": float(config.leaf_theta_mlp_lr),
                    "leaf_theta_mlp_weight_decay": float(config.leaf_theta_mlp_weight_decay),
                    "leaf_theta_mlp_train_loss_final": float(fit_result.train_loss_final),
                    **{f"leaf_theta_{k}": v for k, v in leaf_theta_runtime_meta.items()},
                    **supervision_training_contract(
                        representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                        target_kind=TARGET_SIMPLEX_VECTOR,
                        optimizer_family=OPTIMIZER_FAMILY_GRADIENT_DENSE,
                        optimizer_backend="torch_mlp",
                        selection_mode=str(fit_result.selection_mode),
                        selection_split=str(fit_result.selection_split),
                        selection_metric_name="leaf_theta_mlp_train_loss_final",
                        selection_metric_value=float(fit_result.train_loss_final),
                        best_epoch=int(fit_result.best_epoch),
                        n_train_rows=int(fit_result.n_train_rows),
                    ),
                }
            leaf_theta_meta.update({str(k): v for k, v in fit_meta.items()})
            train_est = [_predict_leaf_theta_model(model, c) for c in train_counts]
            test_est = [_predict_leaf_theta_model(model, c) for c in test_counts]
    if bool(config.include_full_doc_theta_baseline) and leaf_theta_mode in {"rf", "mlp"}:
        full_doc_train_supervision = _full_doc_theta_supervision_dataset(
            train.books,
            split="train",
            n_topics=int(config.n_topics),
            vocab_size=int(config.vocab_size),
        )
        full_doc_train_rows = full_doc_train_supervision.to_dense_vector_training_records(
            law_type="document_level_target"
        )
        baseline_name = f"full_doc_{leaf_theta_mode}"
        full_doc_theta_meta[f"{baseline_name}_train_samples"] = int(len(full_doc_train_rows))
        if int(len(full_doc_train_rows)) <= 0:
            full_doc_theta_meta[f"{baseline_name}_fallback"] = "no_doc_labels"
        else:
            full_doc_test_supervision = _full_doc_theta_supervision_dataset(
                test.books,
                split="test",
                n_topics=int(config.n_topics),
                vocab_size=int(config.vocab_size),
            )
            _x_doc_test, y_doc_test, _w_doc_test = dense_vector_rows_to_numpy(
                full_doc_test_supervision.to_dense_vector_training_records(
                    law_type="document_level_target"
                ),
                normalize_targets_to_simplex=True,
            )
            if leaf_theta_mode == "rf":
                full_doc_model, full_doc_fit = fit_dense_simplex_forest_regressor(
                    full_doc_train_supervision,
                    config=DenseSimplexForestTrainingConfig(
                        model=DenseSimplexForestModelConfig(
                            n_estimators=int(config.leaf_theta_rf_n_estimators),
                            max_depth=int(config.leaf_theta_rf_max_depth),
                            min_samples_leaf=int(config.leaf_theta_rf_min_samples_leaf),
                        ),
                        run=RunConfig(seed=int(seed) + 50_000),
                    ),
                )
                pred_doc = predict_dense_simplex_forest_regressor(
                    full_doc_model,
                    _x_doc_test,
                )
                full_doc_fit_meta = {
                    "leaf_theta_model": "rf",
                    **supervision_training_contract(
                        representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                        target_kind=TARGET_SIMPLEX_VECTOR,
                        optimizer_family=OPTIMIZER_FAMILY_TREE_ENSEMBLE,
                        optimizer_backend="random_forest",
                        selection_mode=str(full_doc_fit.selection_mode),
                        selection_split=str(full_doc_fit.selection_split),
                        selection_metric_name=str(full_doc_fit.selection_metric_name),
                        selection_metric_value=float(full_doc_fit.selection_metric_value),
                        best_epoch=int(full_doc_fit.best_epoch),
                        n_train_rows=int(full_doc_fit.n_train_rows),
                    ),
                }
                full_doc_policy_metrics[baseline_name] = _eval_full_doc_theta_predictions(
                    pred_doc,
                    y_doc_test,
                    c1_threshold=float(config.c1_threshold),
                    c3_threshold=float(config.c3_threshold),
                )
            else:
                try:
                    import torch
                except Exception as e:  # pragma: no cover
                    raise ImportError(
                        "PyTorch is required for include_full_doc_theta_baseline with "
                        "leaf_theta_estimator='mlp'. Install with: pip install torch>=2.0.0"
                    ) from e
                configure_torch_runtime(torch, torch_threads=int(config.torch_threads))
                full_doc_device, full_doc_runtime_meta = resolve_torch_device(
                    torch_module=torch,
                    device=str(config.device),
                    cuda_device=config.cuda_device,
                )
                full_doc_model, full_doc_fit = fit_dense_simplex_regressor(
                    full_doc_train_supervision,
                    config=DenseSimplexTrainingConfig(
                        model=DenseSimplexModelConfig(
                            hidden_dims=(int(config.leaf_theta_mlp_hidden_dim),),
                        ),
                        optimizer=OptimizerConfig(
                            learning_rate=float(config.leaf_theta_mlp_lr),
                            weight_decay=float(config.leaf_theta_mlp_weight_decay),
                        ),
                        train=TrainConfig(
                            batch_size=int(config.leaf_theta_mlp_batch_size),
                            epochs=int(config.leaf_theta_mlp_epochs),
                        ),
                        runtime=RuntimeConfig(
                            device=str(full_doc_device),
                            bf16=False,
                            gradient_checkpointing=False,
                        ),
                        run=RunConfig(seed=int(seed) + 50_000),
                    ),
                )
                pred_doc = predict_dense_simplex_regressor(
                    full_doc_model,
                    supervision=full_doc_test_supervision,
                    device=str(full_doc_device),
                )
                full_doc_fit_meta = {
                    "leaf_theta_model": "mlp",
                    "train_loss_final": float(full_doc_fit.train_loss_final),
                    "train_loss_curve": [float(x) for x in full_doc_fit.train_loss_curve],
                    "epochs_completed": int(full_doc_fit.epochs_completed),
                    "selection_metric_curve": [
                        float(x) for x in full_doc_fit.selection_metric_curve
                    ],
                    **{f"full_doc_{k}": v for k, v in full_doc_runtime_meta.items()},
                    **supervision_training_contract(
                        representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                        target_kind=TARGET_SIMPLEX_VECTOR,
                        optimizer_family=OPTIMIZER_FAMILY_GRADIENT_DENSE,
                        optimizer_backend="torch_mlp",
                        selection_mode=str(full_doc_fit.selection_mode),
                        selection_split=str(full_doc_fit.selection_split),
                        selection_metric_name=str(full_doc_fit.selection_metric_name),
                        selection_metric_value=float(full_doc_fit.selection_metric_value),
                        best_epoch=int(full_doc_fit.best_epoch),
                        n_train_rows=int(full_doc_fit.n_train_rows),
                    ),
                }
                full_doc_policy_metrics[baseline_name] = _eval_full_doc_theta_predictions(
                    pred_doc,
                    y_doc_test,
                    c1_threshold=float(config.c1_threshold),
                    c3_threshold=float(config.c3_threshold),
                )
            full_doc_theta_meta.update(
                {
                    f"{baseline_name}_{str(k)}": v
                    for k, v in dict(full_doc_fit_meta).items()
                }
            )

    # Report held-out leaf-theta prediction error for the proxy estimator (before affine calibration).
    leaf_l1: List[float] = []
    for pred, truth in zip(test_est, test_truth):
        p = np.asarray(pred, dtype=np.float64)
        t = np.asarray(truth, dtype=np.float64)
        if p.shape != t.shape:
            raise RuntimeError("leaf theta shape mismatch between prediction and truth")
        leaf_l1.extend(float(np.sum(np.abs(p[i] - t[i]))) for i in range(int(p.shape[0])))
    leaf_theta_meta["leaf_theta_l1_mean"] = _safe_mean(leaf_l1)
    leaf_theta_meta["leaf_theta_l1_p95"] = _p95(leaf_l1)
    topic_meta.update(leaf_theta_meta)
    topic_meta.update(full_doc_theta_meta)

    # Rebuild padded train proxy from the chosen leaf-theta estimator (lstsq/rf/mlp) for calibration.
    train_proxy_pad = np.zeros((len(train_est), max_train_leaves, k), dtype=np.float64)
    for i, a in enumerate(train_est):
        l = int(a.shape[0])
        train_proxy_pad[i, :l] = np.asarray(a, dtype=np.float64)

    calibration_supervision = _leaf_calibration_supervision_dataset(
        train_proxy_pad,
        train_truth_pad,
        query_mask_pad,
        inclusion_probs_pad=pi_train,
        split="train_calibration",
        query_policy=str(config.calibration_policy),
    )
    calibration_rows = calibration_supervision.to_dense_vector_training_records(law_type="sufficiency")
    if calibration_rows:
        calibrator, calibration_fit = fit_dense_affine_simplex_calibrator(
            calibration_supervision,
            config=AffineSimplexCalibrationConfig(
                ridge=float(config.calibration_ridge),
                use_sample_weights=True,
            ),
        )
        n_calib = int(calibration_fit.n_train_rows)
        calibration_contract = supervision_training_contract(
            prefix="calibration",
            representation_kind=REPRESENTATION_SIMPLEX_VECTOR,
            target_kind=TARGET_SIMPLEX_VECTOR,
            optimizer_family=OPTIMIZER_FAMILY_AFFINE_VECTOR_CALIBRATION,
            optimizer_backend="closed_form_affine_ridge",
            selection_mode=str(calibration_fit.selection_mode),
            selection_split=str(calibration_fit.selection_split),
            selection_metric_name=str(calibration_fit.selection_metric_name),
            selection_metric_value=float(calibration_fit.selection_metric_value),
            best_epoch=int(calibration_fit.best_epoch),
            n_train_rows=int(calibration_fit.n_train_rows),
        )
        topic_meta.update(calibration_contract)
        topic_meta["calibration_mode"] = str(calibration_contract["calibration_supervision_mode"])
        topic_meta["calibration_uses_sample_weights"] = bool(
            calibration_fit.uses_sample_weights
        )
    else:
        calibrator = None
        n_calib = 0
        calibration_contract = supervision_training_contract(
            prefix="calibration",
            representation_kind=REPRESENTATION_SIMPLEX_VECTOR,
            target_kind=TARGET_SIMPLEX_VECTOR,
            optimizer_family=OPTIMIZER_FAMILY_AFFINE_VECTOR_CALIBRATION,
            optimizer_backend="closed_form_affine_ridge",
            n_train_rows=0,
        )
        topic_meta.update(calibration_contract)
        topic_meta["calibration_mode"] = str(calibration_contract["calibration_supervision_mode"])
        topic_meta["calibration_fallback"] = "identity_no_labels"

    # Policy accumulators.
    policy_names = (
        "oracle_proxy",
        "estimated_uncalibrated",
        "estimated_calibrated",
        "estimated_calibrated_budgeted",
        "oracle_tree",
    )
    root_l1: Dict[str, List[float]] = {p: [] for p in policy_names}
    root_l2: Dict[str, List[float]] = {p: [] for p in policy_names}
    c1_err: Dict[str, List[float]] = {p: [] for p in policy_names}
    c3_err: Dict[str, List[float]] = {p: [] for p in policy_names}
    q_leaf: Dict[str, List[float]] = {p: [] for p in policy_names}
    q_internal: Dict[str, List[float]] = {p: [] for p in policy_names}

    # Decomposition components per book (L1 metric).
    decomp_total: List[float] = []
    decomp_topic: List[float] = []
    decomp_calib: List[float] = []
    decomp_guidance: List[float] = []
    decomp_oracle_proxy: List[float] = []
    decomp_upper: List[float] = []
    decomp_slack: List[float] = []

    audit_disc_population: List[float] = []
    audit_score_population: List[float] = []

    for i, book in enumerate(test.books):
        truth_root = _aggregate_root_truth(book, n_topics=int(config.n_topics))

        leaf_truth = np.asarray(test_truth[i], dtype=np.float64)
        leaf_est = np.asarray(test_est[i], dtype=np.float64)
        leaf_oracle_proxy = np.asarray(test_oracle_proxy[i], dtype=np.float64)
        spans = _leaf_spans(len(book.token_words), leaf_tokens=int(config.fixed_leaf_tokens))
        leaf_masses = np.asarray([float(e - s) for (s, e) in spans], dtype=np.float64)
        if leaf_masses.shape[0] != leaf_truth.shape[0]:
            raise RuntimeError("leaf mass shape mismatch with extracted leaf arrays")

        # Policy A: oracle topics but still projected from words (oracle proxy baseline).
        root_op, c1_op, c3_op, lq_op, iq_op, _e0, _s0 = _reduce_balanced_tree_with_guidance(
            leaf_oracle_proxy,
            leaf_truth,
            leaf_query_rate=0.0,
            internal_query_rate=0.0,
            internal_query_design="none",
            rng=rng,
            leaf_masses=leaf_masses,
        )
        root_l1["oracle_proxy"].append(_l1(root_op, truth_root))
        root_l2["oracle_proxy"].append(_l2(root_op, truth_root))
        c1_err["oracle_proxy"].extend(c1_op)
        c3_err["oracle_proxy"].extend(c3_op)
        q_leaf["oracle_proxy"].append(float(lq_op))
        q_internal["oracle_proxy"].append(float(iq_op))

        # Policy B: estimated topics, uncalibrated.
        root_est_u, c1_u, c3_u, lq_u, iq_u, _e1, _s1 = _reduce_balanced_tree_with_guidance(
            leaf_est,
            leaf_truth,
            leaf_query_rate=0.0,
            internal_query_rate=0.0,
            internal_query_design="none",
            rng=rng,
            leaf_masses=leaf_masses,
        )
        root_l1["estimated_uncalibrated"].append(_l1(root_est_u, truth_root))
        root_l2["estimated_uncalibrated"].append(_l2(root_est_u, truth_root))
        c1_err["estimated_uncalibrated"].extend(c1_u)
        c3_err["estimated_uncalibrated"].extend(c3_u)
        q_leaf["estimated_uncalibrated"].append(float(lq_u))
        q_internal["estimated_uncalibrated"].append(float(iq_u))

        # Policy C: estimated topics + calibration.
        if calibrator is None:
            leaf_cal = np.asarray(leaf_est, dtype=np.float64)
        else:
            leaf_cal = apply_dense_affine_simplex_calibrator(calibrator, leaf_est)
        root_est_c, c1_c, c3_c, lq_c, iq_c, pop_e, pop_s = _reduce_balanced_tree_with_guidance(
            leaf_cal,
            leaf_truth,
            leaf_query_rate=0.0,
            internal_query_rate=0.0,
            internal_query_design="none",
            rng=rng,
            leaf_masses=leaf_masses,
        )
        root_l1["estimated_calibrated"].append(_l1(root_est_c, truth_root))
        root_l2["estimated_calibrated"].append(_l2(root_est_c, truth_root))
        c1_err["estimated_calibrated"].extend(c1_c)
        c3_err["estimated_calibrated"].extend(c3_c)
        q_leaf["estimated_calibrated"].append(float(lq_c))
        q_internal["estimated_calibrated"].append(float(iq_c))
        audit_disc_population.extend(float(x) for x in pop_e)
        audit_score_population.extend(float(x) for x in pop_s)

        # Policy D: estimated topics + calibration + eval-time oracle budget.
        root_est_b, c1_b, c3_b, lq_b, iq_b, _e2, _s2 = _reduce_balanced_tree_with_guidance(
            leaf_cal,
            leaf_truth,
            leaf_query_rate=float(config.eval_leaf_query_rate),
            internal_query_rate=float(config.eval_internal_query_rate),
            internal_query_design=str(config.eval_internal_query_design),
            rng=rng,
            leaf_masses=leaf_masses,
        )
        root_l1["estimated_calibrated_budgeted"].append(_l1(root_est_b, truth_root))
        root_l2["estimated_calibrated_budgeted"].append(_l2(root_est_b, truth_root))
        c1_err["estimated_calibrated_budgeted"].extend(c1_b)
        c3_err["estimated_calibrated_budgeted"].extend(c3_b)
        q_leaf["estimated_calibrated_budgeted"].append(float(lq_b))
        q_internal["estimated_calibrated_budgeted"].append(float(iq_b))

        # Policy E: oracle tree (true leaf summaries, full guidance) through same reducer path.
        root_o, c1_o, c3_o, lq_o, iq_o, _e3, _s3 = _reduce_balanced_tree_with_guidance(
            leaf_truth,
            leaf_truth,
            leaf_query_rate=1.0,
            internal_query_rate=1.0,
            internal_query_design="risk",
            rng=rng,
            leaf_masses=leaf_masses,
        )
        root_l1["oracle_tree"].append(_l1(root_o, truth_root))
        root_l2["oracle_tree"].append(_l2(root_o, truth_root))
        c1_err["oracle_tree"].extend(c1_o)
        c3_err["oracle_tree"].extend(c3_o)
        q_leaf["oracle_tree"].append(float(lq_o))
        q_internal["oracle_tree"].append(float(iq_o))

        # End-to-end decomposition chain:
        # truth -> oracle_proxy -> estimated_uncalibrated -> estimated_calibrated -> estimated_calibrated_budgeted
        total = _l1(root_est_b, truth_root)
        comp_topic = _l1(root_est_u, root_op)
        comp_calib = _l1(root_est_c, root_est_u)
        comp_guidance = _l1(root_est_b, root_est_c)
        comp_oracle_proxy = _l1(root_op, truth_root)
        upper = comp_topic + comp_calib + comp_guidance + comp_oracle_proxy
        slack = upper - total

        decomp_total.append(total)
        decomp_topic.append(comp_topic)
        decomp_calib.append(comp_calib)
        decomp_guidance.append(comp_guidance)
        decomp_oracle_proxy.append(comp_oracle_proxy)
        decomp_upper.append(upper)
        decomp_slack.append(slack)

    metrics: Dict[str, PolicyMetrics] = {}
    for p in policy_names:
        metrics[p] = _build_policy_metrics(
            root_l1=root_l1[p],
            root_l2=root_l2[p],
            c1_errors=c1_err[p],
            c3_errors=c3_err[p],
            leaf_queries=q_leaf[p],
            internal_queries=q_internal[p],
            c1_threshold=float(config.c1_threshold),
            c3_threshold=float(config.c3_threshold),
        )
    metrics.update(full_doc_policy_metrics)

    decomposition = EndToEndDecompositionMetrics(
        n_books=int(config.n_books_test),
        total_root_l1_mean=_safe_mean(decomp_total),
        topic_component_mean=_safe_mean(decomp_topic),
        calibration_component_mean=_safe_mean(decomp_calib),
        guidance_component_mean=_safe_mean(decomp_guidance),
        oracle_proxy_component_mean=_safe_mean(decomp_oracle_proxy),
        upper_bound_mean=_safe_mean(decomp_upper),
        slack_mean=_safe_mean(decomp_slack),
    )

    selection_audit: Optional[SelectionAuditSummary] = None
    if int(config.selection_audit_trials) > 0 and len(audit_disc_population) > 0:
        disc = np.asarray(audit_disc_population, dtype=np.float64)
        viol = (disc > float(config.c3_threshold)).astype(np.float64)
        score = np.asarray(audit_score_population, dtype=np.float64)
        selection_audit = _run_selection_bias_audit(
            discrepancies=disc,
            violations=viol,
            scores=score,
            threshold=float(config.c3_threshold),
            trials=int(config.selection_audit_trials),
            sample_rate=float(config.selection_audit_sample_rate),
            pi_min=float(config.selection_audit_pi_min),
            seed=int(config.seed),
        )

    return SegmentedLDACtreePOSummary(
        config=asdict(config),
        topic_meta=topic_meta,
        calibration_samples=int(n_calib),
        metrics=metrics,
        decomposition=decomposition,
        selection_audit=selection_audit,
        objective=discrepancy_benchmark_objective_semantics(
            name="segmented_lda_ctreepo_benchmark",
            optimized_against="ridge_calibration_on_queried_leaves",
            benchmark_metric_name="root_l1_mean",
            metadata={
                "family": "segmented_lda_ctreepo",
                "calibration_leaf_query_rate": float(config.calibration_leaf_query_rate),
                "eval_leaf_query_rate": float(config.eval_leaf_query_rate),
                "eval_internal_query_rate": float(config.eval_internal_query_rate),
            },
        ),
    )


__all__ = [
    "SegmentedLDACtreePOConfig",
    "SegmentedBook",
    "SegmentedCorpus",
    "PolicyMetrics",
    "EndToEndDecompositionMetrics",
    "EstimatorStats",
    "SelectionAuditSummary",
    "SegmentedLDACtreePOSummary",
    "VALID_DEVICE_MODES",
    "VALID_TOPIC_PHI_ESTIMATORS",
    "run_segmented_lda_ctreepo_simulation",
]
