"""
Parallel module optimizer.

This module provides utilities for optimizing multiple independent
DSPy modules concurrently, improving training throughput.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import dspy

from treepo._research.core.async_utils import to_thread
from .base import AbstractOptimizer, OptimizationResult
from .registry import OptimizerRegistry

if TYPE_CHECKING:
    from ..config import OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class ModuleOptimizationConfig:
    """Configuration for optimizing a single module."""

    optimizer_type: str = "bootstrap_random_search"
    weight: float = 1.0  # Relative weight for budget allocation

    # Optimizer-specific settings (passed to optimizer)
    max_bootstrapped_demos: int = 2
    max_labeled_demos: int = 4
    max_rounds: int = 1
    num_candidate_programs: int = 10
    num_threads: int = 64


@dataclass
class ParallelOptimizationResult:
    """Result of parallel module optimization."""

    timestamp: str
    total_duration_seconds: float
    module_results: Dict[str, OptimizationResult]
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'total_duration_seconds': self.total_duration_seconds,
            'module_results': {
                name: result.to_dict()
                for name, result in self.module_results.items()
            },
            'success': self.success,
            'error_message': self.error_message,
        }


class ParallelModuleOptimizer:
    """
    Optimize multiple independent modules concurrently.

    This optimizer runs separate optimization processes for each module
    in parallel, improving overall training throughput when modules
    don't have direct dependencies during optimization.

    Example use cases:
    - Optimizing oracle classifier + leaf summarizer + merge summarizer
    - Optimizing multiple independent classifiers
    - Running the same optimizer with different configs in parallel

    Example:
        parallel_opt = ParallelModuleOptimizer(
            module_configs={
                'oracle': ModuleOptimizationConfig(optimizer_type='gepa'),
                'leaf_summarizer': ModuleOptimizationConfig(optimizer_type='bootstrap'),
                'merge_summarizer': ModuleOptimizationConfig(optimizer_type='bootstrap'),
            },
            max_concurrent=3,
        )

        results = await parallel_opt.compile_async(
            modules={'oracle': oracle, 'leaf_summarizer': leaf, 'merge_summarizer': merge},
            trainsets={'oracle': oracle_trainset, ...},
            metrics={'oracle': oracle_metric, ...},
        )
    """

    def __init__(
        self,
        module_configs: Dict[str, ModuleOptimizationConfig],
        max_concurrent: int = 3,
        shared_config: Optional['OptimizationConfig'] = None,
    ):
        """
        Initialize parallel module optimizer.

        Args:
            module_configs: Config for each module to optimize
            max_concurrent: Maximum concurrent optimizations
            shared_config: Shared OptimizationConfig for common settings
        """
        self.module_configs = module_configs
        self.max_concurrent = max_concurrent
        self.shared_config = shared_config
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)

    def compile(
        self,
        modules: Dict[str, dspy.Module],
        trainsets: Dict[str, List[dspy.Example]],
        valsets: Optional[Dict[str, List[dspy.Example]]] = None,
        metrics: Optional[Dict[str, Callable]] = None,
    ) -> Tuple[Dict[str, dspy.Module], ParallelOptimizationResult]:
        """
        Synchronous wrapper for compile_async.

        Args:
            modules: Dict mapping module names to DSPy modules
            trainsets: Dict mapping module names to training sets
            valsets: Optional dict mapping module names to validation sets
            metrics: Optional dict mapping module names to metrics

        Returns:
            Tuple of (optimized_modules, result)
        """
        return asyncio.run(
            self.compile_async(modules, trainsets, valsets, metrics)
        )

    async def compile_async(
        self,
        modules: Dict[str, dspy.Module],
        trainsets: Dict[str, List[dspy.Example]],
        valsets: Optional[Dict[str, List[dspy.Example]]] = None,
        metrics: Optional[Dict[str, Callable]] = None,
    ) -> Tuple[Dict[str, dspy.Module], ParallelOptimizationResult]:
        """
        Optimize modules concurrently.

        Args:
            modules: Dict mapping module names to DSPy modules
            trainsets: Dict mapping module names to training sets
            valsets: Optional dict mapping module names to validation sets
            metrics: Optional dict mapping module names to metrics

        Returns:
            Tuple of (optimized_modules, result)
        """
        timestamp = datetime.now().isoformat()
        start_time = asyncio.get_event_loop().time()

        valsets = valsets or {}
        metrics = metrics or {}

        # Validate inputs
        for name in modules.keys():
            if name not in trainsets:
                raise ValueError(f"Missing trainset for module: {name}")
            if name not in self.module_configs:
                logger.warning(f"No config for module {name}, using default")
                self.module_configs[name] = ModuleOptimizationConfig()

        # Create tasks for each module
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def optimize_module(name: str) -> Tuple[str, dspy.Module, OptimizationResult]:
            async with semaphore:
                logger.info(f"Starting optimization for module: {name}")
                return await to_thread(
                    self._optimize_single_module,
                    name,
                    modules[name],
                    trainsets[name],
                    valsets.get(name),
                    metrics.get(name),
                )

        # Run all optimizations concurrently
        try:
            tasks = [optimize_module(name) for name in modules.keys()]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            optimized_modules = {}
            module_results = {}

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Module optimization failed: {result}")
                    continue

                name, optimized, opt_result = result
                optimized_modules[name] = optimized
                module_results[name] = opt_result

            end_time = asyncio.get_event_loop().time()

            return optimized_modules, ParallelOptimizationResult(
                timestamp=timestamp,
                total_duration_seconds=end_time - start_time,
                module_results=module_results,
                success=len(optimized_modules) == len(modules),
            )

        except Exception as e:
            logger.error(f"Parallel optimization failed: {e}")
            end_time = asyncio.get_event_loop().time()

            return modules, ParallelOptimizationResult(
                timestamp=timestamp,
                total_duration_seconds=end_time - start_time,
                module_results={},
                success=False,
                error_message=str(e),
            )

    def _optimize_single_module(
        self,
        name: str,
        module: dspy.Module,
        trainset: List[dspy.Example],
        valset: Optional[List[dspy.Example]],
        metric: Optional[Callable],
    ) -> Tuple[str, dspy.Module, OptimizationResult]:
        """
        Optimize a single module (runs in thread).

        Args:
            name: Module name
            module: DSPy module to optimize
            trainset: Training examples
            valset: Validation examples
            metric: Metric function

        Returns:
            Tuple of (name, optimized_module, result)
        """
        config = self.module_configs[name]
        timestamp = datetime.now().isoformat()

        try:
            # Get optimizer from registry
            optimizer = OptimizerRegistry.get(
                config.optimizer_type,
                self.shared_config,
            )

            # Compile
            valset = valset or trainset
            optimized = optimizer.compile(
                student=module,
                trainset=trainset,
                valset=valset,
                metric=metric,
            )

            # Create result
            result = OptimizationResult(
                iteration=1,
                stage=name,
                timestamp=timestamp,
                metric_before=0.0,  # Would need to evaluate before
                metric_after=0.0,   # Would need to evaluate after
                improvement=0.0,
                examples_used=len(trainset),
                trainset_size=len(trainset),
                valset_size=len(valset),
                optimizer_name=config.optimizer_type,
                optimizer_config={
                    'weight': config.weight,
                    'max_bootstrapped_demos': config.max_bootstrapped_demos,
                    'max_labeled_demos': config.max_labeled_demos,
                },
            )

            logger.info(f"Completed optimization for module: {name}")
            return name, optimized, result

        except Exception as e:
            logger.error(f"Failed to optimize module {name}: {e}")
            result = OptimizationResult(
                iteration=1,
                stage=name,
                timestamp=timestamp,
                metric_before=0.0,
                metric_after=0.0,
                improvement=0.0,
                examples_used=len(trainset),
                trainset_size=len(trainset),
                valset_size=len(valset or trainset),
                optimizer_name=config.optimizer_type,
                error_message=str(e),
            )
            return name, module, result

    def estimate_total_budget(self, trainset_sizes: Dict[str, int]) -> int:
        """
        Estimate total metric calls across all modules.

        Args:
            trainset_sizes: Dict mapping module names to trainset sizes

        Returns:
            Total estimated metric calls
        """
        total = 0
        for name, config in self.module_configs.items():
            size = trainset_sizes.get(name, 0)
            try:
                optimizer = OptimizerRegistry.get(config.optimizer_type, self.shared_config)
                total += optimizer.estimate_budget(size)
            except Exception:
                # Fallback estimate
                total += size * 10
        return total


def create_parallel_optimizer(
    module_names: List[str],
    optimizer_type: str = "bootstrap_random_search",
    weights: Optional[Dict[str, float]] = None,
    max_concurrent: int = 3,
    shared_config: Optional['OptimizationConfig'] = None,
) -> ParallelModuleOptimizer:
    """
    Convenience function to create a ParallelModuleOptimizer.

    Args:
        module_names: Names of modules to optimize
        optimizer_type: Optimizer type for all modules (default: bootstrap_random_search)
        weights: Optional weight allocation per module
        max_concurrent: Max concurrent optimizations
        shared_config: Shared configuration

    Returns:
        ParallelModuleOptimizer instance
    """
    weights = weights or {}

    module_configs = {
        name: ModuleOptimizationConfig(
            optimizer_type=optimizer_type,
            weight=weights.get(name, 1.0),
        )
        for name in module_names
    }

    return ParallelModuleOptimizer(
        module_configs=module_configs,
        max_concurrent=max_concurrent,
        shared_config=shared_config,
    )
