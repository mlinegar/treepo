"""
Dataset registry.

Datasets describe where documents come from (manifestos, JSONL, etc.).
"""

from .base import (
    DatasetInfo,
    DatasetPlugin,
    DatasetRegistry,
    register_dataset,
)

from .manifesto import ManifestoDataset
from .jsonl import JSONLDataset
from .pdf import PDFDataset


def get_dataset(name: str, **kwargs):
    """Get a dataset instance by name."""
    return DatasetRegistry.get(name, **kwargs)


def list_datasets():
    """List all registered datasets."""
    return DatasetRegistry.list_datasets()


__all__ = [
    "DatasetInfo",
    "DatasetPlugin",
    "DatasetRegistry",
    "register_dataset",
    "get_dataset",
    "list_datasets",
    "ManifestoDataset",
    "JSONLDataset",
    "PDFDataset",
]
