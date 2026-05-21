"""
Training pipeline module for OPS optimization.

This module provides CLI entry points for running training pipelines,
integrating the OPS training framework with task + dataset plugins.

Key Components:
    - run_training_pipeline: Main training pipeline
    - JudgeOptimizer: Single-pass judge optimization
    - TournamentOfTournamentsTrainer: Full iterative judge optimization loop
    - create_judge_trainset: Create training data for judge optimization
    - collect_preferences: Unified preference collection CLI

Preference Collection:
    - PreferenceDataSource: Protocol for data sources
    - DirectDocumentSource: Load from task data loader
    - LabeledTreeSource: Load from labeled trees (oracle-scored)
    - SyntheticDataSource: Load from JSONL/JSON files
    - PreferenceCollectionConfig: Configuration dataclasses
"""

from importlib import import_module
from typing import Any

_LAZY_ATTRS = {
    "run_training_pipeline": ("src.training.run_pipeline", "run_training_pipeline"),
    "main": ("src.training.run_pipeline", "main"),
    "JudgeOptimizer": ("src.training.judge_optimization", "JudgeOptimizer"),
    "JudgeOptimizationConfig": ("src.training.judge_optimization", "JudgeOptimizationConfig"),
    "create_judge_trainset": ("src.training.judge_optimization", "create_judge_trainset"),
    "SkippedReasons": ("src.training.judge_optimization", "SkippedReasons"),
    "optimize_judge_from_preferences": ("src.training.judge_optimization", "optimize_judge_from_preferences"),
    "load_optimized_judge": ("src.training.judge_optimization", "load_optimized_judge"),
    "TournamentOfTournamentsTrainer": ("src.training.tournament_loop", "TournamentOfTournamentsTrainer"),
    "ToTConfig": ("src.training.tournament_loop", "ToTConfig"),
    "ToTResult": ("src.training.tournament_loop", "ToTResult"),
    "run_tournament_of_tournaments": ("src.training.tournament_loop", "run_tournament_of_tournaments"),
    "collect_preferences": ("src.training.collect_preferences", "main"),
    "collect_preferences_main": ("src.training.collect_preferences", "main"),
    "PreferenceDataSource": ("src.training.data_sources", "PreferenceDataSource"),
    "DirectDocumentSource": ("src.training.data_sources", "DirectDocumentSource"),
    "LabeledTreeSource": ("src.training.data_sources", "LabeledTreeSource"),
    "SyntheticDataSource": ("src.training.data_sources", "SyntheticDataSource"),
    "DataSourceExample": ("src.training.data_sources", "DataSourceExample"),
    "create_data_source": ("src.training.data_sources", "create_data_source"),
    "PreferenceCollectionConfig": ("src.training.preference_config", "PreferenceCollectionConfig"),
    "JudgeType": ("src.training.preference_config", "JudgeType"),
    "DataSourceType": ("src.training.preference_config", "DataSourceType"),
    "ServerConfig": ("src.training.preference_config", "ServerConfig"),
    "GenerationSettings": ("src.training.preference_config", "GenerationSettings"),
    "JudgeSettings": ("src.training.preference_config", "JudgeSettings"),
    "DataSourceSettings": ("src.training.preference_config", "DataSourceSettings"),
}


def __getattr__(name: str) -> Any:
    """Lazily import training symbols to avoid side-effect imports."""
    module_info = _LAZY_ATTRS.get(name)
    if module_info is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module_path, attr_name = module_info
    module = import_module(module_path)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(_LAZY_ATTRS))

__all__ = [
    'run_training_pipeline',
    'main',
    # Judge optimization (single pass)
    'JudgeOptimizer',
    'JudgeOptimizationConfig',
    'create_judge_trainset',
    'SkippedReasons',
    'optimize_judge_from_preferences',
    'load_optimized_judge',
    # Tournament of Tournaments (full iterative loop)
    'TournamentOfTournamentsTrainer',
    'ToTConfig',
    'ToTResult',
    'run_tournament_of_tournaments',
    # Unified preference collection
    'collect_preferences',
    'collect_preferences_main',
    # Data sources
    'PreferenceDataSource',
    'DirectDocumentSource',
    'LabeledTreeSource',
    'SyntheticDataSource',
    'DataSourceExample',
    'create_data_source',
    # Preference config
    'PreferenceCollectionConfig',
    'JudgeType',
    'DataSourceType',
    'ServerConfig',
    'GenerationSettings',
    'JudgeSettings',
    'DataSourceSettings',
]
