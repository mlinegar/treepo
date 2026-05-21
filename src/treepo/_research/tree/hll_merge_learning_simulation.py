"""Compatibility wrapper for the TreePO HLL merge-learning module."""

from __future__ import annotations

from pathlib import Path
import sys


TREEPO_SRC = Path(__file__).resolve().parents[2] / "treepo" / "src"
if str(TREEPO_SRC) not in sys.path:
    sys.path.insert(0, str(TREEPO_SRC))

from treepo.bench.hll_merge_learning import (  # noqa: F401,F403
    ExactMaxMerger,
    HLLBaselineMetrics,
    HLLMergeLearningConfig,
    HLLMergeLearningRun,
    HLLMergeLearningSummary,
    LearnedHLLMerger,
    LearnedMergerMetrics,
    MeanMerger,
    MergeEvalMetrics,
    TokenStreamDoc,
    evaluate_hll_baseline,
    evaluate_merger_on_docs,
    evaluate_merger_on_docs_with_weighting,
    experiment_rows,
    experiment_summary_json,
    generate_token_stream_docs,
    leaf_hll_registers,
    merge_leaf_states,
    run_hll_merge_learning_experiment,
)

__all__ = [
    "ExactMaxMerger",
    "HLLBaselineMetrics",
    "HLLMergeLearningConfig",
    "HLLMergeLearningRun",
    "HLLMergeLearningSummary",
    "LearnedHLLMerger",
    "LearnedMergerMetrics",
    "MeanMerger",
    "MergeEvalMetrics",
    "TokenStreamDoc",
    "evaluate_hll_baseline",
    "evaluate_merger_on_docs",
    "evaluate_merger_on_docs_with_weighting",
    "experiment_rows",
    "experiment_summary_json",
    "generate_token_stream_docs",
    "leaf_hll_registers",
    "merge_leaf_states",
    "run_hll_merge_learning_experiment",
]
