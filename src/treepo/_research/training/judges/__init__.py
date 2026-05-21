"""Public judge backends and capability helpers for supervision collection."""

from treepo._research.training.judges.base import (
    JudgeResult,
    JudgeConfig,
    JudgeError,
    BaseJudge,
    AsyncJudge,
    CompilableJudge,
)

from treepo._research.training.judges.dspy import DSPyJudge
from treepo._research.training.judges.genrm import GenRMJudge, GenRMJudgeWrapper
from treepo._research.training.judges.oracle import OracleJudge
from treepo._research.training.judges.large_dspy import (
    LargeJudgeComparisonModule,
    LargeJudgeListwiseModule,
)
from treepo._research.training.judges.oracle_pairwise import OracleJudgeResult, OraclePairwiseJudge
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
    # Base types
    "JudgeResult",
    "JudgeConfig",
    "JudgeError",
    "BaseJudge",
    "AsyncJudge",
    "CompilableJudge",
    # Implementations
    "DSPyJudge",
    "GenRMJudge",
    "GenRMJudgeWrapper",
    "LargeJudgeComparisonModule",
    "LargeJudgeListwiseModule",
    "OracleJudge",
    "OracleJudgeResult",
    "OraclePairwiseJudge",
    # Capability helpers
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
