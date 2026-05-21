"""
Base optimizer protocol and result classes.

This module defines the interface for all optimizer implementations in the
training framework. Optimizers can be registered and selected dynamically.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

import dspy

logger = logging.getLogger(__name__)


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class OptimizationResult:
    """Result of a single optimization run or stage."""

    iteration: int
    stage: str  # "full", "classify", "retrieve", etc.
    timestamp: str

    # Metrics
    metric_before: float
    metric_after: float
    improvement: float

    # Data info
    examples_used: int
    trainset_size: int
    valset_size: int

    # Optimizer info
    optimizer_name: str = "unknown"
    optimizer_config: Optional[Dict[str, Any]] = None

    # Checkpoint
    checkpoint_path: Optional[Path] = None

    # Additional metadata
    config_snapshot: Optional[Dict] = None
    error_message: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error_message is None

    def to_dict(self) -> Dict:
        return {
            'iteration': self.iteration,
            'stage': self.stage,
            'timestamp': self.timestamp,
            'metric_before': self.metric_before,
            'metric_after': self.metric_after,
            'improvement': self.improvement,
            'examples_used': self.examples_used,
            'trainset_size': self.trainset_size,
            'valset_size': self.valset_size,
            'optimizer_name': self.optimizer_name,
            'optimizer_config': self.optimizer_config,
            'checkpoint_path': str(self.checkpoint_path) if self.checkpoint_path else None,
            'config_snapshot': self.config_snapshot,
            'error_message': self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'OptimizationResult':
        data = dict(data)
        if data.get('checkpoint_path'):
            data['checkpoint_path'] = Path(data['checkpoint_path'])
        return cls(**data)


# =============================================================================
# Optimizer Protocol
# =============================================================================

@runtime_checkable
class BaseOptimizer(Protocol):
    """
    Protocol for all optimizer implementations.

    All optimizers must implement this interface to be usable with the
    optimizer registry and training framework.
    """

    @property
    def name(self) -> str:
        """Unique name for this optimizer type."""
        ...

    @property
    def supports_parallel(self) -> bool:
        """Whether this optimizer can run evaluations in parallel."""
        ...

    def compile(
        self,
        student: dspy.Module,
        trainset: List[dspy.Example],
        valset: Optional[List[dspy.Example]] = None,
        metric: Optional[Callable] = None,
        teacher: Optional[dspy.Module] = None,
        **kwargs,
    ) -> dspy.Module:
        """
        Compile/optimize a DSPy module.

        Args:
            student: The module to optimize
            trainset: Training examples
            valset: Optional validation examples
            metric: DSPy metric function (example, prediction, trace?) -> float
            teacher: Optional teacher module for bootstrap
            **kwargs: Additional optimizer-specific arguments

        Returns:
            Optimized module
        """
        ...

    def estimate_budget(self, dataset_size: int) -> int:
        """
        Estimate number of metric calls for optimization.

        Args:
            dataset_size: Number of training examples

        Returns:
            Estimated metric call count
        """
        ...


# =============================================================================
# Abstract Base Class
# =============================================================================

class AbstractOptimizer(ABC):
    """
    Abstract base class for optimizer implementations.

    Provides common functionality like metric wrapping and logging.
    Subclasses should implement the abstract methods.
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize optimizer.

        Args:
            config: Optimizer configuration (OptimizationConfig or similar)
        """
        self.config = config
        self._compile_count = 0
        self._last_compile_audit: Dict[str, Any] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this optimizer type."""
        pass

    @property
    @abstractmethod
    def supports_parallel(self) -> bool:
        """Whether this optimizer can run evaluations in parallel."""
        pass

    @abstractmethod
    def compile(
        self,
        student: dspy.Module,
        trainset: List[dspy.Example],
        valset: Optional[List[dspy.Example]] = None,
        metric: Optional[Callable] = None,
        teacher: Optional[dspy.Module] = None,
        **kwargs,
    ) -> dspy.Module:
        """Compile/optimize a DSPy module."""
        pass

    @abstractmethod
    def estimate_budget(self, dataset_size: int) -> int:
        """Estimate number of metric calls for optimization."""
        pass

    def wrap_metric(self, metric: Callable, extract_score: bool = True) -> Callable:
        """
        Wrap a metric to ensure consistent return type.

        Many DSPy optimizers expect float returns, but our metrics may
        return dicts with 'score' and 'feedback' keys.

        Args:
            metric: Original metric function
            extract_score: If True, extract 'score' from dict returns

        Returns:
            Wrapped metric function
        """
        if not extract_score:
            return metric

        def wrapped(example, prediction, trace=None, *args, **kwargs):
            result = metric(example, prediction, trace, *args, **kwargs)
            if isinstance(result, dict):
                return result.get('score', 0.0)
            return result

        return wrapped

    def _log_compile_start(self, trainset_size: int, valset_size: int):
        """Log the start of a compile operation."""
        self._compile_count += 1
        self._last_compile_audit = {
            "optimizer_requested": self.name,
            "optimizer_used": self.name,
            "compile_status": "running",
            "trainset_size": int(trainset_size),
            "valset_size": int(valset_size),
            "supports_parallel": bool(self.supports_parallel),
        }
        logger.info(
            f"[{self.name}] Starting optimization #{self._compile_count} "
            f"(train={trainset_size}, val={valset_size})"
        )

    def _log_compile_end(self, metric_before: float, metric_after: float):
        """Log the end of a compile operation."""
        improvement = metric_after - metric_before
        logger.info(
            f"[{self.name}] Optimization complete: "
            f"{metric_before:.4f} -> {metric_after:.4f} ({improvement:+.4f})"
        )

    @property
    def last_compile_audit(self) -> Dict[str, Any]:
        """Best-effort metadata about the most recent compile() call."""
        return dict(self._last_compile_audit)

    def _update_compile_audit(self, **fields: Any) -> None:
        """Record wrapper-specific compile metadata for pipeline diagnostics."""
        self._last_compile_audit.update(fields)
