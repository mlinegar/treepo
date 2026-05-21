"""
Generalized feedback collection for ThinkingTrees.

Type-agnostic feedback system supporting pairwise preferences, scalar ratings,
written critiques, and arbitrary combinations. Works with LLM judges,
oracle scoring functions, and human reviewers (via API).

Core Types:
    FeedbackRequest     -- declares what feedback is wanted
    FeedbackResponse    -- carries collected feedback
    FeedbackDimension   -- specifies a single feedback dimension
    FeedbackDataset     -- collection with export/diagnostics

Protocol:
    FeedbackCollector   -- generalized collection interface

Built-in Collectors:
    PreferenceDeriverAdapter -- wraps existing PreferenceDeriver
    OracleCollector          -- oracle scoring function
    LLMJudgeCollector        -- LLM-based multi-dimensional feedback
    HumanCollector           -- queues for human review via API
    CompositeCollector       -- combines multiple collectors

Registry:
    register_collector(name)  -- decorator to register collectors
    get_collector(name, ...)  -- factory to instantiate by name
    list_collectors()         -- list registered names

Usage:
    from treepo._research.feedback import (
        FeedbackRequest,
        FeedbackResponse,
        FeedbackDataset,
        get_collector,
    )

    collector = get_collector("oracle", oracle_predict=my_fn)
    request = FeedbackRequest(
        request_id="r1",
        text_a="Summary to rate...",
        original_text="Source text...",
        rubric="Preserve key arguments",
    )
    response = collector.collect(request)
    print(response.to_dspy_metric())
"""

from treepo._research.feedback.types import (
    FeedbackDimension,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackDataset,
)
from treepo._research.feedback.collector import (
    FeedbackCollector,
    PreferenceDeriverAdapter,
    get_collector,
    list_collectors,
    register_collector,
)
from treepo._research.feedback.store import FeedbackStore
from treepo._research.feedback.collectors.human import HumanCollector

# Import concrete collectors to trigger registration
import treepo._research.feedback.collectors  # noqa: F401

__all__ = [
    "FeedbackDimension",
    "FeedbackRequest",
    "FeedbackResponse",
    "FeedbackDataset",
    "FeedbackCollector",
    "FeedbackStore",
    "HumanCollector",
    "PreferenceDeriverAdapter",
    "get_collector",
    "list_collectors",
    "register_collector",
]
