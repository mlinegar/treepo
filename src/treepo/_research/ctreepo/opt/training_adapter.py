from __future__ import annotations

from typing import Any, Iterable

from .records import PairwisePreference


def to_training_preference_dataset(records: Iterable[PairwisePreference]) -> Any:
    """Convert opt-layer records into the repo's binary projection dataset."""
    from treepo._research.training.supervision import BinaryProjectionDataset

    pairs = [record.to_training_preference_pair() for record in records]
    return BinaryProjectionDataset(comparisons=pairs)
