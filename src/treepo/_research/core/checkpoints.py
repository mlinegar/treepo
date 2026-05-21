"""
Checkpoint utilities for resumable training pipelines.

This module provides a unified CheckpointManager for saving and loading
pipeline state, enabling resumable training runs.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Checkpoint format version (increment when format changes)
CHECKPOINT_VERSION = "1.0"


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    version: str = CHECKPOINT_VERSION
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    phase: str = ""
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """
    Manager for saving and loading checkpoints.

    Provides a unified API for checkpoint operations across the pipeline.
    Supports both JSON (for metadata) and pickle (for data) formats.

    Usage:
        manager = CheckpointManager(output_dir / 'checkpoints')

        # Save a phase checkpoint
        manager.save_phase('phase1', {'train_count': 100}, data={'results': results})

        # Check if phase is complete
        if manager.is_phase_complete('phase1'):
            data = manager.load_phase_data('phase1')
    """

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for storing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _metadata_path(self, name: str) -> Path:
        """Get path for metadata file."""
        return self.checkpoint_dir / f'{name}_complete.json'

    def _data_path(self, name: str) -> Path:
        """Get path for data file."""
        return self.checkpoint_dir / f'{name}_data.pkl'

    def save_phase(
        self,
        phase_name: str,
        metadata: Dict[str, Any],
        data: Optional[Any] = None,
        description: str = "",
    ) -> None:
        """
        Save a phase checkpoint.

        Args:
            phase_name: Name of the phase (e.g., 'phase1', 'phase1_5')
            metadata: Dict of metadata to save (JSON-serializable)
            data: Optional data object to pickle
            description: Human-readable description
        """
        # Create checkpoint metadata
        checkpoint_meta = CheckpointMetadata(
            phase=phase_name,
            description=description,
            extra=metadata,
        )

        # Save metadata
        meta_path = self._metadata_path(phase_name)
        with open(meta_path, 'w') as f:
            json.dump({
                'version': checkpoint_meta.version,
                'created_at': checkpoint_meta.created_at,
                'phase': checkpoint_meta.phase,
                'description': checkpoint_meta.description,
                **metadata,
            }, f, indent=2)

        # Save data if provided
        if data is not None:
            data_path = self._data_path(phase_name)
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)

        logger.debug(f"Saved checkpoint: {phase_name}")

    def is_phase_complete(self, phase_name: str) -> bool:
        """Check if a phase checkpoint exists."""
        return self._metadata_path(phase_name).exists()

    def has_data(self, phase_name: str) -> bool:
        """Check if a phase has data saved."""
        return self._data_path(phase_name).exists()

    def load_phase_metadata(self, phase_name: str) -> Dict[str, Any]:
        """
        Load phase metadata.

        Args:
            phase_name: Name of the phase

        Returns:
            Metadata dict

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        meta_path = self._metadata_path(phase_name)
        with open(meta_path, 'r') as f:
            return json.load(f)

    def load_phase_data(self, phase_name: str) -> Any:
        """
        Load phase data.

        Args:
            phase_name: Name of the phase

        Returns:
            Unpickled data object

        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        data_path = self._data_path(phase_name)
        with open(data_path, 'rb') as f:
            return pickle.load(f)

    def save_iteration(
        self,
        iteration: int,
        stats: Dict[str, Any],
        prefix: str = "iteration",
    ) -> Path:
        """
        Save an iteration checkpoint.

        Args:
            iteration: Iteration number
            stats: Stats dict to save
            prefix: Filename prefix

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f'{prefix}_{iteration}.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(stats, f, indent=2)
        return checkpoint_path

    def list_checkpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available checkpoints.

        Returns:
            Dict mapping phase names to metadata
        """
        result = {}
        for meta_file in self.checkpoint_dir.glob('*_complete.json'):
            phase_name = meta_file.stem.replace('_complete', '')
            try:
                with open(meta_file, 'r') as f:
                    result[phase_name] = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load checkpoint {phase_name}: {e}")
        return result

    def clear(self, phase_name: Optional[str] = None) -> None:
        """
        Clear checkpoints.

        Args:
            phase_name: If provided, clear only this phase. Otherwise clear all.
        """
        if phase_name:
            self._metadata_path(phase_name).unlink(missing_ok=True)
            self._data_path(phase_name).unlink(missing_ok=True)
        else:
            for f in self.checkpoint_dir.glob('*.json'):
                f.unlink()
            for f in self.checkpoint_dir.glob('*.pkl'):
                f.unlink()
