"""Internal compatibility wrapper over ``src.training.supervision.judge_capabilities``."""

from treepo._research.training.supervision.judge_capabilities import (
    ComparativeJudgeResult,
    PairwiseJudgeResult,
    invoke_comparative_judgment_async,
    invoke_comparative_judgment_sync,
    invoke_pairwise_judgment_async,
    invoke_pairwise_judgment_sync,
    judge_backend_name,
    supports_direct_comparative_judging,
    supports_pairwise_judging,
)

__all__ = [
    "ComparativeJudgeResult",
    "PairwiseJudgeResult",
    "invoke_comparative_judgment_async",
    "invoke_comparative_judgment_sync",
    "invoke_pairwise_judgment_async",
    "invoke_pairwise_judgment_sync",
    "judge_backend_name",
    "supports_direct_comparative_judging",
    "supports_pairwise_judging",
]
