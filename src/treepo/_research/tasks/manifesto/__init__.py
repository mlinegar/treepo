"""Lazy Manifesto/RILE research namespace.

Importing a small constants module such as `pipeline_config` should not load
DSPy signatures, training metrics, or tree builders. Resolve the historical
top-level names only on demand.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_ATTR_MODULES = {
    "RILE_RANGE": "treepo._research.tasks.manifesto.constants",
    "RILE_MIN": "treepo._research.tasks.manifesto.constants",
    "RILE_MAX": "treepo._research.tasks.manifesto.constants",
    "RILE_PRESERVATION_RUBRIC": "treepo._research.tasks.manifesto.rubrics",
    "RILE_TASK_CONTEXT": "treepo._research.tasks.manifesto.rubrics",
    "create_rile_oracle": "treepo._research.tasks.manifesto.oracle",
    "ManifestoDataset": "treepo._research.tasks.manifesto.data_loader",
    "ManifestoSample": "treepo._research.tasks.manifesto.data_loader",
    "create_pilot_dataset": "treepo._research.tasks.manifesto.data_loader",
    "LeafSummarizer": "treepo._research.tasks.manifesto.summarizer",
    "MergeSummarizer": "treepo._research.tasks.manifesto.summarizer",
    "GenericSummarizer": "treepo._research.tasks.manifesto.summarizer",
    "RILEScore": "treepo._research.tasks.manifesto.dspy_signatures",
    "RILEScorer": "treepo._research.tasks.manifesto.dspy_signatures",
    "RILEComparison": "treepo._research.tasks.manifesto.dspy_signatures",
    "RILEComparator": "treepo._research.tasks.manifesto.dspy_signatures",
    "SimpleScore": "treepo._research.tasks.manifesto.dspy_signatures",
    "PairwiseSummaryComparison": "treepo._research.tasks.manifesto.dspy_signatures",
    "RILESummarize": "treepo._research.tasks.manifesto.pipeline",
    "RILEMerge": "treepo._research.tasks.manifesto.pipeline",
    "RILEScoreSignature": "treepo._research.tasks.manifesto.pipeline",
    "UnifiedManifestoG": "treepo._research.tasks.manifesto.pipeline",
    "ManifestoSummarizer": "treepo._research.tasks.manifesto.pipeline",
    "ManifestoMerger": "treepo._research.tasks.manifesto.pipeline",
    "ManifestoScorer": "treepo._research.tasks.manifesto.pipeline",
    "StrategyCompatibleSummarizer": "treepo._research.tasks.manifesto.pipeline",
    "StrategyCompatibleMerger": "treepo._research.tasks.manifesto.pipeline",
    "ManifestoPipeline": "treepo._research.tasks.manifesto.pipeline",
    "ManifestoPipelineWithStrategy": "treepo._research.tasks.manifesto.pipeline",
    "create_training_examples": "treepo._research.tasks.manifesto.training_data",
    "rile_metric": "treepo._research.tasks.manifesto.metrics",
    "is_placeholder": "treepo._research.tasks.manifesto.metrics",
    "create_rile_summarization_metric": "treepo._research.tasks.manifesto.metrics",
    "create_rile_merge_metric": "treepo._research.tasks.manifesto.metrics",
}


def _build_rile_scale() -> Any:
    from treepo._research.tasks.base import ScaleDefinition

    return ScaleDefinition(
        name="rile",
        min_value=-100.0,
        max_value=100.0,
        description="Right-Left ideological scale. -100 = far left, +100 = far right",
        higher_is_better=True,
        neutral_value=0.0,
    )


def __getattr__(name: str) -> Any:
    if name == "RILE_SCALE":
        value = _build_rile_scale()
        globals()[name] = value
        return value
    module_name = _ATTR_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))


__all__ = ["RILE_SCALE", *_ATTR_MODULES]
