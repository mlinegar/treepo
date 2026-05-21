"""
Dataset plugin base types.

Datasets describe where documents come from (manifestos, jsonl, etc.).
Tasks describe what we do with those documents (summarization, scoring, IE).
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

from treepo._research.core.documents import DocumentSample


@dataclass
class DatasetInfo:
    """Metadata describing a dataset plugin."""
    name: str
    description: str = ""
    supports_reference_scores: bool = True


@runtime_checkable
class DatasetPlugin(Protocol):
    """Protocol for dataset plugins."""

    @property
    def name(self) -> str:
        """Unique identifier for this dataset."""
        ...

    def load_samples(self, **kwargs: Any) -> List[DocumentSample]:
        """Load samples from the dataset."""
        ...

    def get_info(self) -> DatasetInfo:
        """Return dataset metadata."""
        ...


class DatasetRegistry:
    """Registry for dataset plugins."""

    _datasets: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, dataset_class: type) -> None:
        if name in cls._datasets:
            raise ValueError(f"Dataset '{name}' already registered")
        cls._datasets[name] = dataset_class

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> DatasetPlugin:
        if name not in cls._datasets:
            available = ", ".join(sorted(cls._datasets.keys()))
            raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
        return cls._datasets[name](**kwargs)

    @classmethod
    def list_datasets(cls) -> Dict[str, DatasetInfo]:
        info = {}
        for name, dataset_class in cls._datasets.items():
            dataset = dataset_class()
            info[name] = dataset.get_info()
        return info

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._datasets


def register_dataset(name: str):
    """Decorator to register dataset plugins."""
    def decorator(cls):
        DatasetRegistry.register(name, cls)
        return cls
    return decorator
