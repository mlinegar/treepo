"""Lazy task registry namespace."""

from __future__ import annotations

from importlib import import_module

from treepo._research.tasks.base import AbstractTask, TaskPlugin
from treepo._research.tasks.registry import TaskRegistry, register_task


_DEFAULT_TASK_MODULES = (
    "treepo._research.tasks.document_analysis",
    "treepo._research.tasks.manifesto_task",
    "treepo._research.tasks.scoring",
)
_TASKS_REGISTERED = False


def _ensure_default_tasks_registered() -> None:
    global _TASKS_REGISTERED
    if _TASKS_REGISTERED:
        return
    for module_name in _DEFAULT_TASK_MODULES:
        import_module(module_name)
    _TASKS_REGISTERED = True


def get_task(name: str, **kwargs):
    """Get a task by name."""
    _ensure_default_tasks_registered()
    return TaskRegistry.get(name, **kwargs)


def list_tasks():
    """List registered tasks."""
    _ensure_default_tasks_registered()
    return TaskRegistry.list_tasks()


def __getattr__(name: str):
    if name in {
        "PromptBuilders",
        "default_merge_prompt",
        "default_summarize_prompt",
        "parse_numeric_score",
    }:
        module = import_module("treepo._research.tasks.prompting")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))


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
