"""
Provenance utilities shared across the codebase.

Today we mainly standardize "truth label source" values used for:
  - training-data filtering (what supervision is trusted enough to learn from)
  - preference/feedback provenance reporting
  - adaptive chunking / proxy-model metadata

Canonical sources are intentionally small so downstream stats are stable:
  - human: manual annotation / review
  - dataset: trusted existing labels
  - oracle: trusted model-based labels (e.g., a large judge/reward model)
  - unknown: fallback when provenance is missing or unrecognized
"""

from __future__ import annotations

from typing import Any, Dict

TruthLabelSource = str

HUMAN_SOURCE: TruthLabelSource = "human"
DATASET_SOURCE: TruthLabelSource = "dataset"
ORACLE_SOURCE: TruthLabelSource = "oracle"
UNKNOWN_SOURCE: TruthLabelSource = "unknown"

CANONICAL_TRUTH_SOURCES: tuple[TruthLabelSource, ...] = (
    HUMAN_SOURCE,
    DATASET_SOURCE,
    ORACLE_SOURCE,
    UNKNOWN_SOURCE,
)

_TRUTH_SOURCE_ALIASES: Dict[str, TruthLabelSource] = {
    # Human labels.
    "human": HUMAN_SOURCE,
    "labeler": HUMAN_SOURCE,
    "annotator": HUMAN_SOURCE,
    "manual": HUMAN_SOURCE,
    # Dataset labels.
    "dataset": DATASET_SOURCE,
    "data": DATASET_SOURCE,
    "gold": DATASET_SOURCE,
    # Trusted model-based labels.
    "oracle": ORACLE_SOURCE,
    "task_oracle": ORACLE_SOURCE,
    "oracle_callback": ORACLE_SOURCE,
    "model_backed_teacher": ORACLE_SOURCE,
    "task": ORACLE_SOURCE,
    "judge": ORACLE_SOURCE,
    "llm_judge": ORACLE_SOURCE,
    "reward_model": ORACLE_SOURCE,
    "rm": ORACLE_SOURCE,
    "genrm": ORACLE_SOURCE,
    # Fallback.
    "unknown": UNKNOWN_SOURCE,
}


def normalize_truth_label_source(value: Any, *, default: TruthLabelSource = UNKNOWN_SOURCE) -> TruthLabelSource:
    """Normalize truth-label provenance into {human, dataset, oracle, unknown}."""
    if value is None:
        return default
    key = str(value).strip().lower()
    if not key:
        return default
    return _TRUTH_SOURCE_ALIASES.get(key, default)
