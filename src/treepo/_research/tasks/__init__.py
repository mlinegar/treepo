"""
Task registry.

Tasks describe what we do with documents (summarization, scoring, extraction).
"""

from treepo._research.tasks.base import (
    TaskPlugin,
    AbstractTask,
)
from treepo._research.tasks.registry import (
    TaskRegistry,
    register_task,
)

# Import task modules to trigger registration
from treepo._research.tasks import document_analysis
from treepo._research.tasks import manifesto_task
from treepo._research.tasks import scoring

# Helper functions using registry
def get_task(name: str, **kwargs):
    """Get a task by name from the registry."""
    return TaskRegistry.get(name, **kwargs)

def list_tasks():
    """List all registered tasks."""
    return TaskRegistry.list_tasks()

from .prompting import PromptBuilders, default_merge_prompt, default_summarize_prompt, parse_numeric_score


__all__ = [
    "TaskPlugin",
    "AbstractTask",
    "TaskRegistry",
    "register_task",
    "get_task",
    "list_tasks",
    "PromptBuilders",
    "default_merge_prompt",
    "default_summarize_prompt",
    "parse_numeric_score",
]
