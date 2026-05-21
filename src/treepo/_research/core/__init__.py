"""Core public exports with lazy loading to keep imports lightweight."""

from __future__ import annotations

from importlib import import_module
from typing import Dict


_MODULE_EXPORTS = {
    "src.core.data_models": (
        "Node",
        "Tree",
        "AuditStatus",
        "AuditResult",
        "leaf",
        "node",
    ),
    "src.core.llm_client": (
        "ServerType",
        "LLMConfig",
        "LLMResponse",
        "LLMClient",
        "MockLLMClient",
        "create_client",
        "create_summarizer",
        "engine_client",
        "vllm_client",
        "sglang_client",
        "openai_client",
    ),
    "src.core.engines": (
        "EngineRegistry",
        "EngineSpec",
        "EngineSurface",
        "EngineType",
        "build_server_manager",
    ),
    "src.core.runtime_capabilities": (
        "FamilyRuntimeCapability",
        "RuntimeClaimStatus",
        "default_family_runtime_capability",
        "markov_family_runtime_capability",
    ),
    "src.core.signatures": (
        "RecursiveSummary",
        "OracleJudge",
        "SufficiencyCheck",
        "MergeConsistencyCheck",
        "Summarizer",
        "Judge",
        "SufficiencyChecker",
        "MergeChecker",
        "OracleFuncApproximation",
        "OracleFuncReviewer",
    ),
    "src.core.strategy": (
        "SummarizationStrategy",
        "DSPyStrategy",
        "BatchedStrategy",
    ),
    "src.core.prompting": (
        "PromptBuilders",
        "default_summarize_prompt",
        "default_merge_prompt",
        "parse_numeric_score",
    ),
    "src.core.checkpoints": (
        "CheckpointManager",
        "CheckpointMetadata",
        "CHECKPOINT_VERSION",
    ),
    "src.core.scorers": (
        "ScaleScorer",
        "PairwiseScorer",
    ),
    "src.core.summarization": (
        "GenericSummarizer",
        "GenericMerger",
        "SummarizationResult",
        "create_summarizers",
    ),
    "src.core.conditional_memory": (
        "ConditionalMemoryEntry",
        "ConditionalMemoryConfig",
        "ConditionalMemory",
        "canonical_hash",
        "hash_payload",
        "get_default_memory",
    ),
    "src.core.semantic_memory": (
        "SemanticMemoryConfig",
        "SemanticMemoryEntry",
        "SemanticMemoryIndex",
        "SemanticNeighbor",
        "normalize_rile_delta",
        "temporal_delta_targets",
    ),
    "src.core.logged_supervision": (
        "LoggedLabelObservation",
        "LoggedObservationArtifact",
        "ObservationUnitKind",
        "OracleLabeler",
        "SamplingMetadata",
        "SamplingPolicy",
        "collect_logged_observations",
        "read_logged_observations_jsonl",
        "summarize_logged_observations",
        "write_logged_observations_jsonl",
    ),
    "src.core.local_law_adjustment": (
        "LOCAL_LAW_OBJECTIVE_CORRECTED",
        "LOCAL_LAW_OBJECTIVE_SAMPLED_IPW",
        "LocalLawAggregate",
        "LocalLawObservation",
        "VALID_LOCAL_LAW_OBJECTIVE_MODES",
        "aggregate_local_law_observations",
        "corrected_local_law_loss",
        "depth_discount",
        "local_law_objective_mean",
        "normalize_local_law_objective_mode",
    ),
    "src.core.supervision_metadata": (
        "JUDGMENT_SUPERVISION_CHANNEL_NAME",
        "JUDGMENT_SUPERVISION_SIGNAL_NAME",
        "JudgmentSupervisionMetadata",
        "judgment_supervision_metadata",
        "normalize_judgment_law_type",
    ),
}

_NAME_TO_MODULE: Dict[str, str] = {
    name: module_name
    for module_name, names in _MODULE_EXPORTS.items()
    for name in names
}

__all__ = [
    "Node",
    "Tree",
    "AuditStatus",
    "AuditResult",
    "leaf",
    "node",
    "ServerType",
    "EngineRegistry",
    "EngineSpec",
    "EngineSurface",
    "EngineType",
    "LLMConfig",
    "LLMResponse",
    "LLMClient",
    "MockLLMClient",
    "build_server_manager",
    "create_client",
    "create_summarizer",
    "engine_client",
    "vllm_client",
    "sglang_client",
    "openai_client",
    "FamilyRuntimeCapability",
    "RuntimeClaimStatus",
    "default_family_runtime_capability",
    "markov_family_runtime_capability",
    "RecursiveSummary",
    "OracleJudge",
    "SufficiencyCheck",
    "MergeConsistencyCheck",
    "Summarizer",
    "Judge",
    "SufficiencyChecker",
    "MergeChecker",
    "OracleFuncApproximation",
    "OracleFuncReviewer",
    "SummarizationStrategy",
    "DSPyStrategy",
    "BatchedStrategy",
    "PromptBuilders",
    "default_summarize_prompt",
    "default_merge_prompt",
    "parse_numeric_score",
    "CheckpointManager",
    "CheckpointMetadata",
    "CHECKPOINT_VERSION",
    "ScaleScorer",
    "PairwiseScorer",
    "GenericSummarizer",
    "GenericMerger",
    "SummarizationResult",
    "create_summarizers",
    "ConditionalMemoryEntry",
    "ConditionalMemoryConfig",
    "MemoryRecord",
    "ConditionalMemory",
    "canonical_hash",
    "hash_payload",
    "get_default_memory",
    "SemanticMemoryConfig",
    "SemanticMemoryEntry",
    "SemanticMemoryIndex",
    "SemanticNeighbor",
    "normalize_rile_delta",
    "temporal_delta_targets",
    "LoggedLabelObservation",
    "LoggedObservationArtifact",
    "ObservationUnitKind",
    "OracleLabeler",
    "SamplingMetadata",
    "SamplingPolicy",
    "collect_logged_observations",
    "read_logged_observations_jsonl",
    "summarize_logged_observations",
    "write_logged_observations_jsonl",
    "LOCAL_LAW_OBJECTIVE_CORRECTED",
    "LOCAL_LAW_OBJECTIVE_SAMPLED_IPW",
    "LocalLawAggregate",
    "LocalLawObservation",
    "VALID_LOCAL_LAW_OBJECTIVE_MODES",
    "aggregate_local_law_observations",
    "corrected_local_law_loss",
    "depth_discount",
    "local_law_objective_mean",
    "normalize_local_law_objective_mode",
    "JUDGMENT_SUPERVISION_CHANNEL_NAME",
    "JUDGMENT_SUPERVISION_SIGNAL_NAME",
    "JudgmentSupervisionMetadata",
    "judgment_supervision_metadata",
    "normalize_judgment_law_type",
]


def __getattr__(name: str):
    if name == "MemoryRecord":
        module = import_module("src.core.conditional_memory")
        return getattr(module, "ConditionalMemoryEntry")
    module_name = _NAME_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
