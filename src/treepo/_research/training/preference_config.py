"""
Configuration dataclasses for unified preference collection.

Provides composable configuration objects that can be built from:
- CLI arguments
- settings.yaml
- Environment variables

The configuration follows the priority: CLI > env vars > YAML > defaults

Example usage:
    settings = load_settings()
    config = PreferenceCollectionConfig.from_cli_and_settings(args, settings)
"""

import argparse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from treepo._research.config.settings import DEFAULT_TASK


class JudgeType(Enum):
    """Type of judge for preference derivation."""

    GENRM = "genrm"
    """GenRM HTTP API (RANKING_SCORE_THRESHOLD strategy)."""

    ORACLE = "oracle"
    """Numeric oracle scorer (ERROR_DIFFERENCE strategy)."""

    DSPY = "dspy"
    """DSPy listwise judge over multiple candidate summaries."""


class DataSourceType(Enum):
    """Type of data source for preference collection."""

    DIRECT = "direct"
    """Direct documents from task data loader."""

    LABELED = "labeled"
    """Pre-generated labeled trees (oracle-scored)."""

    SYNTHETIC = "synthetic"
    """Synthetic data files (JSONL/JSON)."""


@dataclass
class ServerConfig:
    """
    Server connection configuration for summarizer and judge.

    Both summarizer and judge run as vLLM servers with OpenAI-compatible APIs.
    """

    summarizer_port: int = 8000
    """Port for the summarizer model server."""

    summarizer_model: str = "openai/qwen-30b-thinking"
    """Model name for the summarizer (as expected by vLLM)."""

    judge_port: int = 8001
    """Port for the judge/oracle server."""

    judge_model: Optional[str] = None
    """Judge model name (auto-detected if None)."""

    @property
    def summarizer_url(self) -> str:
        """Get full URL for summarizer API."""
        return f"http://localhost:{self.summarizer_port}/v1"

    @property
    def judge_url(self) -> str:
        """Get full URL for judge API."""
        return f"http://localhost:{self.judge_port}/v1"

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "ServerConfig":
        """
        Load server config from settings.yaml.

        Args:
            settings: Loaded settings dictionary

        Returns:
            ServerConfig with values from settings
        """
        servers = settings.get("servers", {})
        return cls(
            summarizer_port=int(servers.get("task_model_port", 8000)),
            summarizer_model=servers.get("summarizer_model", "openai/qwen-30b-thinking"),
            judge_port=int(servers.get("genrm_port", 8001)),
            judge_model=servers.get("genrm_model"),
        )

    @classmethod
    def from_cli(
        cls,
        args: argparse.Namespace,
        settings: Dict[str, Any],
    ) -> "ServerConfig":
        """
        Build config from CLI args, falling back to settings.

        Args:
            args: Parsed CLI arguments
            settings: Loaded settings dictionary

        Returns:
            ServerConfig with CLI overrides
        """
        base = cls.from_settings(settings)
        return cls(
            summarizer_port=getattr(args, "summarizer_port", None) or base.summarizer_port,
            summarizer_model=getattr(args, "summarizer_model", None) or base.summarizer_model,
            judge_port=getattr(args, "judge_port", None) or base.judge_port,
            judge_model=getattr(args, "judge_model", None) or base.judge_model,
        )


@dataclass
class GenerationSettings:
    """
    Settings for candidate summary generation.

    Controls how many candidates are generated and with what parameters.

    The k_candidates parameter determines how many candidate summaries are
    generated per input. Pairwise backends induce O(k^2) comparisons, while
    listwise backends can judge all k candidates in one call.
    """

    k_candidates: int = 4
    """Number of candidate summaries per input (must be >= 2)."""

    temperatures: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9])
    """Temperatures for diverse generation (one per candidate)."""

    summarizer_temperature: float = 0.5
    """Base temperature for summarizer (may be overridden per candidate)."""

    summarizer_max_tokens: int = 2048
    """Maximum tokens for summary generation."""

    def __post_init__(self):
        """Validate settings after initialization."""
        if self.k_candidates < 2:
            raise ValueError(
                f"k_candidates must be >= 2 (got {self.k_candidates}). "
                "At least 2 candidates are needed for pairwise comparison."
            )

    @property
    def num_comparisons(self) -> int:
        """Number of pairwise comparisons for k candidates."""
        return self.k_candidates * (self.k_candidates - 1) // 2

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "GenerationSettings":
        """Load generation settings from settings.yaml."""
        gen = settings.get("generation", {})
        summ = gen.get("summarizer", {})
        return cls(
            k_candidates=gen.get("k_candidates", 4),
            temperatures=summ.get("candidate_temperatures", [0.3, 0.5, 0.7, 0.9]),
            summarizer_temperature=summ.get("temperature", 0.5),
            summarizer_max_tokens=summ.get("max_tokens", 2048),
        )

    @classmethod
    def from_cli(
        cls,
        args: argparse.Namespace,
        settings: Dict[str, Any],
    ) -> "GenerationSettings":
        """Build config from CLI args, falling back to settings."""
        base = cls.from_settings(settings)
        return cls(
            k_candidates=getattr(args, "k_candidates", None) or base.k_candidates,
            temperatures=getattr(args, "temperatures", None) or base.temperatures,
            summarizer_temperature=base.summarizer_temperature,
            summarizer_max_tokens=base.summarizer_max_tokens,
        )


@dataclass
class JudgeSettings:
    """
    Settings for the preference judge/oracle.

    Supports both GenRM judges and numeric oracle scorers.
    """

    judge_type: JudgeType = JudgeType.GENRM
    """Type of judge to use."""

    # GenRM settings
    judge_temperature: float = 0.6
    """Temperature for GenRM judge."""

    judge_top_p: float = 0.95
    """Top-p sampling for GenRM judge."""

    judge_max_tokens: int = 2048
    """Maximum tokens for GenRM response."""

    # Oracle settings
    tie_margin: float = 5.0
    """Margin below which scores are considered a tie (oracle only)."""

    confidence_floor: float = 0.5
    """Minimum confidence for oracle predictions."""

    @classmethod
    def from_settings(
        cls,
        settings: Dict[str, Any],
        judge_type: JudgeType = JudgeType.GENRM,
    ) -> "JudgeSettings":
        """Load judge settings from settings.yaml."""
        gen = settings.get("generation", {})
        judge = gen.get("genrm_judge", {})
        oracle = gen.get("oracle", {})
        return cls(
            judge_type=judge_type,
            judge_temperature=judge.get("temperature", 0.6),
            judge_top_p=judge.get("top_p", 0.95),
            judge_max_tokens=judge.get("max_tokens", 2048),
            tie_margin=oracle.get("tie_margin", 5.0),
            confidence_floor=oracle.get("confidence_floor", 0.5),
        )

    @classmethod
    def from_cli(
        cls,
        args: argparse.Namespace,
        settings: Dict[str, Any],
    ) -> "JudgeSettings":
        """Build config from CLI args, falling back to settings."""
        # Parse judge type from CLI
        judge_type_str = getattr(args, "judge_type", "genrm")
        judge_type = JudgeType(judge_type_str)

        base = cls.from_settings(settings, judge_type)

        return cls(
            judge_type=judge_type,
            judge_temperature=base.judge_temperature,
            judge_top_p=base.judge_top_p,
            judge_max_tokens=base.judge_max_tokens,
            tie_margin=getattr(args, "tie_margin", None) or base.tie_margin,
            confidence_floor=getattr(args, "confidence_floor", None) or base.confidence_floor,
        )


@dataclass
class DataSourceSettings:
    """
    Settings for the preference data source.

    Configures which data source to use and its parameters.
    """

    source_type: DataSourceType = DataSourceType.DIRECT
    """Type of data source."""

    # Direct source settings
    max_documents: Optional[int] = None
    """Maximum documents to process (None = all)."""

    train_only: bool = False
    """Only use training split (for direct source)."""

    # Labeled tree source settings
    labels_dir: Optional[Path] = None
    """Directory containing labeled trees (oracle-scored)."""

    max_trees: Optional[int] = None
    """Maximum trees to process (labeled source)."""

    max_nodes_per_tree: Optional[int] = None
    """Maximum nodes per tree (labeled source)."""

    # Synthetic source settings
    synthetic_data_path: Optional[Path] = None
    """Path to synthetic data file (JSONL/JSON)."""

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> "DataSourceSettings":
        """Build settings from CLI args."""
        source_type_str = getattr(args, "source_type", "direct")
        source_type = DataSourceType(source_type_str)

        return cls(
            source_type=source_type,
            max_documents=getattr(args, "max_documents", None),
            train_only=getattr(args, "train_only", False),
            labels_dir=getattr(args, "labels_dir", None),
            max_trees=getattr(args, "max_trees", None),
            max_nodes_per_tree=getattr(args, "max_nodes_per_tree", None),
            synthetic_data_path=getattr(args, "synthetic_data", None),
        )


@dataclass
class PreferenceCollectionConfig:
    """
    Complete configuration for preference collection.

    This is the top-level config that combines all sub-configs.
    Use from_cli_and_settings() to build from CLI args and settings.yaml.
    """

    # Task configuration
    task_name: str = DEFAULT_TASK
    """Name of the task (from TaskRegistry)."""

    law_type: str = "sufficiency"
    """OPS law type: sufficiency, idempotence, merge, or all."""

    # Sub-configs
    server: ServerConfig = field(default_factory=ServerConfig)
    """Server connection configuration."""

    generation: GenerationSettings = field(default_factory=GenerationSettings)
    """Candidate generation settings."""

    judge: JudgeSettings = field(default_factory=JudgeSettings)
    """Judge/oracle settings."""

    data_source: DataSourceSettings = field(default_factory=DataSourceSettings)
    """Data source configuration."""

    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("data/preferences"))
    """Output directory for preference data."""

    save_dpo_format: bool = True
    """Also save in DPO training format."""

    output_prefix: Optional[str] = None
    """Prefix for output files (auto-generated if None)."""

    # Misc
    seed: int = 42
    """Random seed for reproducibility."""

    verbose: bool = False
    """Enable verbose logging."""

    # Convenience properties for k configuration
    @property
    def k_candidates(self) -> int:
        """Number of candidate summaries per input (convenience accessor)."""
        return self.generation.k_candidates

    @property
    def num_comparisons(self) -> int:
        """Number of pairwise comparisons for current k setting."""
        return self.generation.num_comparisons

    @classmethod
    def from_cli_and_settings(
        cls,
        args: argparse.Namespace,
        settings: Dict[str, Any],
    ) -> "PreferenceCollectionConfig":
        """
        Build complete config from CLI args and settings.yaml.

        CLI arguments take precedence over settings.yaml values.

        Args:
            args: Parsed CLI arguments
            settings: Loaded settings dictionary

        Returns:
            Complete PreferenceCollectionConfig
        """
        return cls(
            task_name=getattr(args, "task", DEFAULT_TASK),
            law_type=getattr(args, "law_type", "sufficiency"),
            server=ServerConfig.from_cli(args, settings),
            generation=GenerationSettings.from_cli(args, settings),
            judge=JudgeSettings.from_cli(args, settings),
            data_source=DataSourceSettings.from_cli(args),
            output_dir=Path(getattr(args, "output_dir", "data/preferences")),
            save_dpo_format=not getattr(args, "no_dpo", False),
            output_prefix=getattr(args, "output_prefix", None),
            seed=getattr(args, "seed", 42),
            verbose=getattr(args, "verbose", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving."""
        return {
            "task_name": self.task_name,
            "law_type": self.law_type,
            "server": {
                "summarizer_port": self.server.summarizer_port,
                "summarizer_model": self.server.summarizer_model,
                "judge_port": self.server.judge_port,
                "judge_model": self.server.judge_model,
            },
            "generation": {
                "k_candidates": self.generation.k_candidates,
                "num_comparisons": self.generation.num_comparisons,
                "temperatures": self.generation.temperatures,
                "summarizer_temperature": self.generation.summarizer_temperature,
                "summarizer_max_tokens": self.generation.summarizer_max_tokens,
            },
            "judge": {
                "judge_type": self.judge.judge_type.value,
                "judge_temperature": self.judge.judge_temperature,
                "judge_top_p": self.judge.judge_top_p,
                "judge_max_tokens": self.judge.judge_max_tokens,
                "tie_margin": self.judge.tie_margin,
                "confidence_floor": self.judge.confidence_floor,
            },
            "data_source": {
                "source_type": self.data_source.source_type.value,
                "max_documents": self.data_source.max_documents,
                "train_only": self.data_source.train_only,
                "labels_dir": str(self.data_source.labels_dir)
                if self.data_source.labels_dir
                else None,
                "max_trees": self.data_source.max_trees,
                "max_nodes_per_tree": self.data_source.max_nodes_per_tree,
                "synthetic_data_path": str(self.data_source.synthetic_data_path)
                if self.data_source.synthetic_data_path
                else None,
            },
            "output_dir": str(self.output_dir),
            "save_dpo_format": self.save_dpo_format,
            "seed": self.seed,
        }
