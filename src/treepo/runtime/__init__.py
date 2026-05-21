"""Runtime benchmark helpers with optional LLM integrations."""

from treepo.runtime.longbench import (
    LongBenchRow,
    load_longbench_jsonl,
    parse_choice,
    render_longbench_prompt,
    score_choice_accuracy,
)
from treepo.runtime.eval import (
    RUNTIME_CONFIG_KEYS,
    RUNTIME_METHODS,
    RuntimeCall,
    RuntimeEvalSummary,
    RuntimePrediction,
    run_runtime_eval,
    runtime_summary_to_csv_rows,
    validate_runtime_config,
)

__all__ = [
    "RUNTIME_CONFIG_KEYS",
    "RUNTIME_METHODS",
    "LongBenchRow",
    "RuntimeCall",
    "RuntimeEvalSummary",
    "RuntimePrediction",
    "load_longbench_jsonl",
    "parse_choice",
    "render_longbench_prompt",
    "run_runtime_eval",
    "runtime_summary_to_csv_rows",
    "score_choice_accuracy",
    "validate_runtime_config",
]
