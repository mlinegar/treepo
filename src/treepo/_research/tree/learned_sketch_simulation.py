"""Compatibility wrapper for the TreePO cardinality recovery module."""

from __future__ import annotations

from pathlib import Path
import sys


TREEPO_SRC = Path(__file__).resolve().parents[2] / "treepo" / "src"
if str(TREEPO_SRC) not in sys.path:
    sys.path.insert(0, str(TREEPO_SRC))

from treepo.common import VALID_AUDIT_POLICIES, VALID_SCHEDULES, audit_sample_count
from treepo.hll import (
    HLLConfig,
    HyperLogLogSketch,
    hll_relative_standard_error,
    match_hll_precision_for_bits,
    reduce_hll_sketches,
)
from treepo.bench.cardinality_recovery import (  # noqa: F401,F403
    APPROX_AUDITED_EVIDENCE,
    DEFAULT_REGULARIZER_WEIGHT,
    DEFAULT_SUMMARY_SHARE,
    DEFAULT_LAW_STRENGTH,
    DEFAULT_LAW_COMPONENT_SHARE,
    PROXY_ONLY_EVIDENCE,
    CardinalityBaselineMetrics,
    CardinalityDocument,
    CardinalityRecoveryConfig,
    CardinalityRecoveryRun,
    CardinalityRecoverySummary,
    ExperimentSummary,
    HLLMetrics,
    LearningRunSummary,
    LearnedMergeableSketch,
    ModelEvalMetrics,
    RegularizedObjectiveMetrics,
    SimulationConfig,
    VALID_SIMULATION_MODES,
    compute_regularized_objective_metrics,
    compute_theoretical_floor_rmse,
    evaluate_exact_set_baseline,
    evaluate_hll_baseline,
    evaluate_learned_model,
    evaluate_model_loss,
    evaluate_sum_leaf_uniques_baseline,
    experiment_rows,
    generate_cardinality_documents,
    run_cardinality_recovery_experiment,
    run_learning_vs_hll_experiment,
    train_learned_model,
)

__all__ = [
    "APPROX_AUDITED_EVIDENCE",
    "DEFAULT_REGULARIZER_WEIGHT",
    "DEFAULT_SUMMARY_SHARE",
    "DEFAULT_LAW_STRENGTH",
    "DEFAULT_LAW_COMPONENT_SHARE",
    "CardinalityBaselineMetrics",
    "CardinalityDocument",
    "CardinalityRecoveryConfig",
    "CardinalityRecoveryRun",
    "CardinalityRecoverySummary",
    "ExperimentSummary",
    "HLLConfig",
    "HLLMetrics",
    "HyperLogLogSketch",
    "LearningRunSummary",
    "LearnedMergeableSketch",
    "ModelEvalMetrics",
    "PROXY_ONLY_EVIDENCE",
    "RegularizedObjectiveMetrics",
    "SimulationConfig",
    "VALID_AUDIT_POLICIES",
    "VALID_SCHEDULES",
    "VALID_SIMULATION_MODES",
    "audit_sample_count",
    "compute_regularized_objective_metrics",
    "compute_theoretical_floor_rmse",
    "evaluate_exact_set_baseline",
    "evaluate_hll_baseline",
    "evaluate_learned_model",
    "evaluate_model_loss",
    "evaluate_sum_leaf_uniques_baseline",
    "experiment_rows",
    "generate_cardinality_documents",
    "hll_relative_standard_error",
    "match_hll_precision_for_bits",
    "reduce_hll_sketches",
    "run_cardinality_recovery_experiment",
    "run_learning_vs_hll_experiment",
    "train_learned_model",
]
