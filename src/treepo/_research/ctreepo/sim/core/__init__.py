"""Core (library) simulation implementations for C-TreePO.

These modules are intended to have minimal dependencies and stable APIs so the
`ctreepo` namespace can be extracted into its own distribution later.
"""

from __future__ import annotations

from treepo._research.ctreepo.sim.core.boundary_topic_treepo_preference import (
    BoundaryTopicExactUtilityConfig,
    BoundaryTopicExactUtilityDGP,
    run_boundary_topic_exact_utility_experiment,
)
from treepo._research.ctreepo.sim.core.exact_utility_common import (
    BudgetVector,
    ExactStateRecovery,
    ExactUtilityMetrics,
    ExactUtilityRunConfig,
    ExactUtilitySummary,
    ObjectiveFamily,
    StructuralArm,
    UtilityLane,
)
from treepo._research.ctreepo.sim.core.markov_treepo_preference import (
    MarkovExactUtilityConfig,
    MarkovExactUtilityDGP,
    run_markov_exact_utility_experiment,
)
from treepo._research.ctreepo.sim.core.nonseparable_treepo_preference import (
    NonseparableExactUtilityConfig,
    NonseparableExactUtilityDGP,
    run_nonseparable_exact_utility_experiment,
)

__all__ = [
    "BoundaryTopicExactUtilityConfig",
    "BoundaryTopicExactUtilityDGP",
    "BudgetVector",
    "ExactStateRecovery",
    "ExactUtilityMetrics",
    "ExactUtilityRunConfig",
    "ExactUtilitySummary",
    "MarkovExactUtilityConfig",
    "MarkovExactUtilityDGP",
    "NonseparableExactUtilityConfig",
    "NonseparableExactUtilityDGP",
    "ObjectiveFamily",
    "StructuralArm",
    "UtilityLane",
    "run_boundary_topic_exact_utility_experiment",
    "run_markov_exact_utility_experiment",
    "run_nonseparable_exact_utility_experiment",
]
