"""
Manifesto RILE Scoring - Example Components.

This module provides RILE-specific components (data loader, scorer, rubrics)
that can be used with the generic ScoringTask.

RILE scores range from -100 (far left) to +100 (far right).

Usage:
    from treepo._research.tasks.base import ScoringTask, ScaleDefinition
    from treepo._research.tasks.manifesto import (
        RILE_SCALE,
        RILE_PRESERVATION_RUBRIC,
        ManifestoDataset,
        RILEScorer,
    )

    # Create task with RILE configuration
    task = ScoringTask(
        name="rile",
        scale=RILE_SCALE,
        rubric=RILE_PRESERVATION_RUBRIC,
        data_loader_factory=lambda: ManifestoDataset(),
        predictor_factory=lambda: RILEScorer(),
    )
"""

from treepo._research.tasks.base import ScaleDefinition

from .constants import (
    RILE_RANGE,
    RILE_MIN,
    RILE_MAX,
)
from .rubrics import (
    RILE_PRESERVATION_RUBRIC,
    RILE_TASK_CONTEXT,
)

# Oracle
from .oracle import create_rile_oracle

# Data loading
from .data_loader import (
    ManifestoDataset,
    ManifestoSample,
    create_pilot_dataset,
)

# Summarizer
from .summarizer import (
    LeafSummarizer,
    MergeSummarizer,
    GenericSummarizer,
)

# DSPy signatures
from .dspy_signatures import (
    RILEScore,
    RILEScorer,
    RILEComparison,
    RILEComparator,
    SimpleScore,
    PairwiseSummaryComparison,
)

# Pipeline
from .pipeline import (
    # Signatures
    RILESummarize,
    RILEMerge,
    RILEScoreSignature,
    # Modules
    UnifiedManifestoG,
    ManifestoSummarizer,
    ManifestoMerger,
    ManifestoScorer,
    StrategyCompatibleSummarizer,
    StrategyCompatibleMerger,
    # Pipelines
    ManifestoPipeline,
    ManifestoPipelineWithStrategy,
    # Training helpers
    create_training_examples,
    rile_metric,
    is_placeholder,
)
from .metrics import (
    create_rile_summarization_metric,
    create_rile_merge_metric,
)


# =============================================================================
# RILE Scale Definition
# =============================================================================

RILE_SCALE = ScaleDefinition(
    name="rile",
    min_value=-100.0,
    max_value=100.0,
    description="Right-Left ideological scale. -100 = far left, +100 = far right",
    higher_is_better=True,
    neutral_value=0.0,
)


__all__ = [
    # Scale
    "RILE_SCALE",

    # Constants
    "RILE_RANGE",
    "RILE_MIN",
    "RILE_MAX",

    # Rubrics
    "RILE_PRESERVATION_RUBRIC",
    "RILE_TASK_CONTEXT",

    # Oracle
    "create_rile_oracle",

    # Data loading
    "ManifestoDataset",
    "ManifestoSample",
    "create_pilot_dataset",

    # Summarizers
    "LeafSummarizer",
    "MergeSummarizer",
    "GenericSummarizer",

    # DSPy signatures and modules
    "RILEScore",
    "RILEScorer",
    "RILEComparison",
    "RILEComparator",
    "SimpleScore",
    "PairwiseSummaryComparison",

    # Pipeline signatures
    "RILESummarize",
    "RILEMerge",
    "RILEScoreSignature",

    # Pipeline modules
    "UnifiedManifestoG",
    "ManifestoSummarizer",
    "ManifestoMerger",
    "ManifestoScorer",
    "StrategyCompatibleSummarizer",
    "StrategyCompatibleMerger",

    # Full pipelines
    "ManifestoPipeline",
    "ManifestoPipelineWithStrategy",

    # Training helpers
    "create_training_examples",
    "rile_metric",
    "is_placeholder",
    "create_rile_summarization_metric",
    "create_rile_merge_metric",
]
