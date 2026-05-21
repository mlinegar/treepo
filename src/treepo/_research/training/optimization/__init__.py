"""
DSPy optimization infrastructure for OPS training.

Provides a registry of optimizer implementations and utilities
for training oracle approximation models.
"""

# Base types
from treepo._research.training.optimization.base import (
    OptimizationResult,
    BaseOptimizer,
    AbstractOptimizer,
)
from treepo._research.training.optimization.performance import (
    OptimizerRunRecord,
    OptimizerCellSummary,
    dataset_regime_label,
    metric_gain,
    summarize_optimizer_runs,
)

# Registry
from treepo._research.training.optimization.registry import (
    OptimizerRegistry,
    register_optimizer,
)

# Helper functions using registry
def get_optimizer(name: str, config=None):
    """Get an optimizer by name from the registry."""
    return OptimizerRegistry.get(name, config=config)

def auto_select_optimizer(dataset_size: int, config=None) -> str:
    """Auto-select an optimizer based on dataset size."""
    return OptimizerRegistry.auto_select(dataset_size, config)

def list_optimizers():
    """List all registered optimizers."""
    return OptimizerRegistry.list_optimizers()

# Implementations
from treepo._research.training.optimization.gepa import GEPAOptimizer
from treepo._research.training.optimization.bootstrap import (
    BootstrapOptimizer,
    BootstrapRandomSearchOptimizer,
    LabeledFewShotOptimizer,
)
from treepo._research.training.optimization.mipro import MIPROOptimizer
from treepo._research.training.optimization.parallel import (
    ParallelModuleOptimizer,
    ModuleOptimizationConfig,
)

# High-level wrapper
from treepo._research.training.optimization.optimizer import (
    OracleOptimizer,
)

__all__ = [
    # Base
    "OptimizationResult",
    "OptimizerRunRecord",
    "OptimizerCellSummary",
    "BaseOptimizer",
    "AbstractOptimizer",
    # Registry
    "OptimizerRegistry",
    "register_optimizer",
    "get_optimizer",
    "auto_select_optimizer",
    "list_optimizers",
    "dataset_regime_label",
    "metric_gain",
    "summarize_optimizer_runs",
    # Implementations
    "GEPAOptimizer",
    "BootstrapOptimizer",
    "BootstrapRandomSearchOptimizer",
    "LabeledFewShotOptimizer",
    "MIPROOptimizer",
    "ParallelModuleOptimizer",
    "ModuleOptimizationConfig",
    # High-level
    "OracleOptimizer",
]
