"""
Generic registry pattern for managing registered classes.

Provides a base class that can be specialized for different
registration needs (tasks, optimizers, etc.).
"""

import inspect
import logging
from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union, Iterable

logger = logging.getLogger(__name__)

T = TypeVar('T')


class GenericRegistry(Generic[T]):
    """
    Generic registry for managing registered classes.

    Subclasses should define class-level storage:
        _registry: Dict[str, Type[T]] = {}
        _instances: Dict[str, T] = {}

    Example:
        class MyRegistry(GenericRegistry[MyBaseClass]):
            _registry: Dict[str, Type[MyBaseClass]] = {}
            _instances: Dict[str, MyBaseClass] = {}
            _item_type = "widget"
    """

    # Subclasses must define these
    _registry: Dict[str, Type[T]]
    _instances: Dict[str, T]
    _item_type: str = "item"  # For error messages: "optimizer", "task", etc.

    @classmethod
    def register(cls, name: str, item_class: Type[T]) -> None:
        """
        Register an implementation.

        Args:
            name: Unique name for the item
            item_class: Class to register
        """
        if name in cls._registry:
            logger.warning(f"Overwriting existing {cls._item_type}: {name}")
        cls._registry[name] = item_class
        logger.debug(f"Registered {cls._item_type}: {name}")

    @classmethod
    def get_class(cls, name: str) -> Type[T]:
        """
        Get registered class by name.

        Args:
            name: Registered name

        Returns:
            Registered class

        Raises:
            KeyError: If name not found
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"Unknown {cls._item_type}: {name}. Available: {available}"
            )
        return cls._registry[name]

    @classmethod
    def get(cls, name: str, **kwargs) -> T:
        """
        Create an instance of a registered class.

        Args:
            name: Registered name
            **kwargs: Arguments to pass to constructor

        Returns:
            New instance
        """
        item_class = cls.get_class(name)
        if not kwargs:
            return item_class()

        try:
            signature = inspect.signature(item_class.__init__)
        except (TypeError, ValueError):
            logger.debug(
                "Registry %s: unable to inspect __init__ for '%s'; passing kwargs as-is",
                cls._item_type,
                name,
            )
            return item_class(**kwargs)

        params = signature.parameters
        accepts_var_kw = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        )
        if accepts_var_kw:
            return item_class(**kwargs)

        allowed = {
            param_name
            for param_name, param in params.items()
            if param_name != "self"
            and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        filtered = {key: value for key, value in kwargs.items() if key in allowed}
        unknown = sorted(set(kwargs) - set(filtered))
        if unknown:
            logger.warning(
                "Ignoring unknown %s config keys for '%s': %s",
                cls._item_type,
                name,
                ", ".join(unknown),
            )

        return item_class(**filtered)

    @classmethod
    def get_singleton(cls, name: str, **kwargs) -> T:
        """
        Get a cached singleton instance.

        Args:
            name: Registered name
            **kwargs: Arguments (only used on first call)

        Returns:
            Cached instance
        """
        if name not in cls._instances:
            cls._instances[name] = cls.get(name, **kwargs)
        return cls._instances[name]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a name is registered."""
        return name in cls._registry

    @classmethod
    def list_registered(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all registered items with metadata.

        Returns:
            Dict mapping names to metadata
        """
        result = {}
        for name, item_class in cls._registry.items():
            try:
                result[name] = {
                    'class': item_class.__name__,
                    'module': item_class.__module__,
                }
            except Exception:
                result[name] = {
                    'class': str(item_class),
                    'module': 'unknown',
                }
        return result

    @classmethod
    def clear(cls) -> None:
        """Clear all registered items (mainly for testing)."""
        cls._registry.clear()
        cls._instances.clear()


def create_register_decorator(registry_class: Type[GenericRegistry[T]]):
    """
    Factory to create a registration decorator for a registry.

    Args:
        registry_class: The registry class to register with

    Returns:
        A decorator function

    Example:
        register_widget = create_register_decorator(WidgetRegistry)

        @register_widget("my_widget")
        class MyWidget(BaseWidget):
            ...
    """
    def register(name: Union[str, Iterable[str]]):
        def decorator(cls: Type[T]) -> Type[T]:
            names = [name] if isinstance(name, str) else list(name)
            for item in names:
                registry_class.register(item, cls)
            return cls
        return decorator
    return register
