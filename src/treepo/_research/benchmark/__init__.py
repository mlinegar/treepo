"""
Benchmark tools for vLLM model comparison.

This module provides throughput benchmarking utilities to compare
generation speed between different vLLM model deployments.
"""

from treepo._research.core.engines import build_server_manager

from .component_microbench import (
    available_benchmarks,
    run_selected_benchmarks,
)
from .perf_suite import (
    compare_suite_results,
    load_suite_config,
    render_comparison_markdown,
    render_suite_markdown,
    run_performance_suite,
    save_suite_results,
)
from .pipeline_limits import (
    GENRM_MODE_CONFIGS,
    StepSummary,
    StepSweepResult,
    SweepPoint,
    default_output_path,
    expand_genrm_steps,
    format_human_summary,
    parse_concurrency_grid,
    parse_genrm_modes,
    run_pipeline_throughput_suite,
    write_suite_csv,
    write_suite_json,
)
from .throughput import (
    BackendCapabilities,
    ComparisonResult,
    ServerManager,
    SGLangServerManager,
    ThroughputBenchmark,
    ThroughputComparison,
    ThroughputResult,
    VLLMServerManager,
    load_model_config,
    run_parallel_comparison,
    run_sequential_comparison,
    save_results,
)
from .tree_batching import (
    TreeBatchBudgetReport,
    TreeBatchPointConfig,
    TreeBatchPointResult,
    TreeBatchSuiteResult,
    compute_tree_batch_budget,
    expand_tree_batch_grid,
    parse_positive_float_grid,
    parse_positive_int_grid,
    render_tree_batch_markdown,
    summarize_metrics_snapshots,
    summarize_tree_batch_results,
    write_tree_batch_jsonl,
    write_tree_batch_markdown,
)

__all__ = [
    "BackendCapabilities",
    "ServerManager",
    "build_server_manager",
    "ThroughputResult",
    "ComparisonResult",
    "ThroughputBenchmark",
    "ThroughputComparison",
    "VLLMServerManager",
    "SGLangServerManager",
    "run_sequential_comparison",
    "run_parallel_comparison",
    "load_model_config",
    "save_results",
    "SweepPoint",
    "StepSummary",
    "StepSweepResult",
    "GENRM_MODE_CONFIGS",
    "parse_concurrency_grid",
    "parse_genrm_modes",
    "expand_genrm_steps",
    "run_pipeline_throughput_suite",
    "write_suite_json",
    "write_suite_csv",
    "format_human_summary",
    "default_output_path",
    "available_benchmarks",
    "run_selected_benchmarks",
    "load_suite_config",
    "run_performance_suite",
    "save_suite_results",
    "render_suite_markdown",
    "compare_suite_results",
    "render_comparison_markdown",
    "TreeBatchPointConfig",
    "TreeBatchPointResult",
    "TreeBatchSuiteResult",
    "TreeBatchBudgetReport",
    "parse_positive_int_grid",
    "parse_positive_float_grid",
    "expand_tree_batch_grid",
    "compute_tree_batch_budget",
    "summarize_metrics_snapshots",
    "summarize_tree_batch_results",
    "render_tree_batch_markdown",
    "write_tree_batch_jsonl",
    "write_tree_batch_markdown",
]
