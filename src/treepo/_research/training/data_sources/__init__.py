"""
Data sources for training and preference collection.

This module consolidates data source abstractions:
- preference: Data sources for preference collection (documents, labeled trees, synthetic)
- training: Training data sources (human-reviewed, document labels, auto-reviewed)
"""

# Preference collection data sources
from treepo._research.training.data_sources.preference import (
    DataSourceExample,
    PreferenceDataSource,
    DirectDocumentSource,
    LabeledTreeSource,
    SyntheticDataSource,
    create_data_source,
)

# Training data sources
from treepo._research.training.data_sources.training import (
    NodeLevelHumanSource,
    FullDocumentLabelSource,
    OracleAutoReviewSource,
    UnifiedTrainingCollector,
)

__all__ = [
    # Preference collection
    "DataSourceExample",
    "PreferenceDataSource",
    "DirectDocumentSource",
    "LabeledTreeSource",
    "SyntheticDataSource",
    "create_data_source",
    # Training
    "NodeLevelHumanSource",
    "FullDocumentLabelSource",
    "OracleAutoReviewSource",
    "UnifiedTrainingCollector",
]
