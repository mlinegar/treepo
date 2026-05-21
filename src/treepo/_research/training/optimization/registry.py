"""
Optimizer registry with auto-selection.

This module provides a registry for optimizer implementations and
automatic selection based on dataset size and computational budget.
"""

import logging
from typing import Any, Dict, Optional, Type, TYPE_CHECKING

from treepo._research.core.registry import GenericRegistry, create_register_decorator
from .base import BaseOptimizer

if TYPE_CHECKING:
    from ..config import OptimizationConfig

logger = logging.getLogger(__name__)


class OptimizerRegistry(GenericRegistry[BaseOptimizer]):
    """
    Registry for optimizer implementations with auto-selection.

    Provides a central place to register and retrieve optimizer classes,
    as well as automatic selection based on dataset size.
    """

    _registry: Dict[str, Type[BaseOptimizer]] = {}
    _instances: Dict[str, BaseOptimizer] = {}
    _item_type = "optimizer"

    @classmethod
    def list_optimizers(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all registered optimizers with their properties.

        Returns:
            Dict mapping names to optimizer metadata
        """
        result = {}
        for name, opt_class in cls._registry.items():
            try:
                dummy = opt_class(None)
                result[name] = {
                    'class': opt_class.__name__,
                    'supports_parallel': dummy.supports_parallel,
                    'module': opt_class.__module__,
                }
            except Exception:
                result[name] = {
                    'class': opt_class.__name__,
                    'supports_parallel': 'unknown',
                    'module': opt_class.__module__,
                }
        return result

    @classmethod
    def auto_select(
        cls,
        dataset_size: int,
        config: Optional['OptimizationConfig'] = None,
    ) -> str:
        """
        Auto-select optimizer based on dataset size and config.

        Selection logic follows DSPy best practices:
        - Very few examples (~10): BootstrapFewShot
        - Medium dataset (50+): BootstrapFewShotWithRandomSearch
        - Large dataset (200+): MIPROv2 with longer runs
        - Very large or custom budget: GEPA

        Args:
            dataset_size: Number of training examples
            config: Optional config with custom thresholds

        Returns:
            Name of recommended optimizer
        """
        # Get thresholds from config or use defaults
        if config is not None:
            bootstrap_threshold = getattr(config, 'bootstrap_threshold', 10)
            random_search_threshold = getattr(config, 'random_search_threshold', 120)
            mipro_threshold = getattr(config, 'mipro_threshold', 200)
        else:
            from ..config import OptimizationConfig

            defaults = OptimizationConfig()
            bootstrap_threshold = int(getattr(defaults, 'bootstrap_threshold', 10))
            random_search_threshold = int(getattr(defaults, 'random_search_threshold', 120))
            mipro_threshold = int(getattr(defaults, 'mipro_threshold', 200))

        # Select based on dataset size
        if dataset_size <= bootstrap_threshold:
            selected = "bootstrap"
            reason = f"dataset_size ({dataset_size}) <= bootstrap_threshold ({bootstrap_threshold})"
        elif dataset_size <= random_search_threshold:
            selected = "bootstrap_random_search"
            reason = f"dataset_size ({dataset_size}) <= random_search_threshold ({random_search_threshold})"
        elif dataset_size <= mipro_threshold:
            selected = "mipro"
            reason = f"dataset_size ({dataset_size}) <= mipro_threshold ({mipro_threshold})"
        else:
            selected = "gepa"
            reason = f"dataset_size ({dataset_size}) > mipro_threshold ({mipro_threshold})"

        # Verify the selected optimizer is registered
        if selected not in cls._registry:
            fallbacks = ["bootstrap", "bootstrap_random_search", "gepa", "mipro"]
            for fallback in fallbacks:
                if fallback in cls._registry:
                    logger.warning(
                        f"Auto-selected '{selected}' not registered, "
                        f"falling back to '{fallback}'"
                    )
                    return fallback
            raise RuntimeError("No optimizers registered")

        logger.info(f"Auto-selected optimizer: {selected} ({reason})")
        return selected


# Decorator for registering optimizers
register_optimizer = create_register_decorator(OptimizerRegistry)
