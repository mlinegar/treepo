"""
Task registry for managing task plugins.

This module provides a central registry for task implementations,
allowing dynamic discovery and selection of tasks.
"""

import logging
from typing import Dict, Type, Union, Iterable

from treepo._research.core.registry import GenericRegistry, create_register_decorator
from .base import TaskPlugin

logger = logging.getLogger(__name__)


class TaskRegistry(GenericRegistry[TaskPlugin]):
    """
    Registry for task implementations.

    Provides a central place to register and retrieve task plugins.
    Tasks can be registered by name and retrieved dynamically.
    """

    _registry: Dict[str, Type[TaskPlugin]] = {}
    _instances: Dict[str, TaskPlugin] = {}
    _item_type = "task"

    @classmethod
    def list_tasks(cls) -> Dict[str, Dict]:
        """List all registered tasks with their metadata."""
        return cls.list_registered()


# Decorator for registering tasks
register_task = create_register_decorator(TaskRegistry)
