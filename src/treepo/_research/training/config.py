"""
Configuration classes for the Oracle Approximation Training Framework.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

# Default thread count for parallel metric evaluation - use all available cores
DEFAULT_NUM_THREADS = os.cpu_count() or 64


@dataclass
class OracleIRRConfig:
    """Configuration for Oracle Infer-Retrieve-Rank module."""

    # Retriever settings
    retriever_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    retriever_cache_dir: Optional[Path] = None

    # Pipeline settings
    rank_topk: int = 5
    skip_retrieve: bool = False

    # Confidence thresholds
    auto_approve_threshold: float = 0.8
    auto_reject_threshold: float = 0.8
    confidence_threshold: float = 0.6

    # Stage compilation flags
    infer_compile: bool = True
    rank_compile: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {}
        for k, v in vars(self).items():
            if not k.startswith('_'):
                d[k] = str(v) if isinstance(v, Path) else v
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OracleIRRConfig':
        """Create from dictionary."""
        data = dict(data)
        if 'retriever_cache_dir' in data and data['retriever_cache_dir']:
            data['retriever_cache_dir'] = Path(data['retriever_cache_dir'])
        return cls(**data)


@dataclass
class OptimizationConfig:
    """Configuration for DSPy optimization."""

    # General optimizer settings
    # Options: 'auto', 'gepa', 'bootstrap', 'bootstrap_random_search', 'mipro', 'labeled_fewshot'
    optimizer_type: str = "gepa"

    # DSPy Bootstrap settings (for 'bootstrap' optimizer_type)
    max_bootstrapped_demos: int = 4  # Reduced from 8 to leave room in 8K context
    max_labeled_demos: int = 8  # Reduced from 16 to leave room in 8K context
    max_rounds: int = 4  # More iterations for bootstrap convergence
    num_candidate_programs: int = 20
    num_threads: int = DEFAULT_NUM_THREADS  # Parallel metric evaluations

    # GEPA-specific settings
    gepa_auto: str = "heavy"  # 'light', 'medium', 'heavy'
    # Optional cap for GEPA worker threads. None means "do not cap" and
    # honor num_threads directly.
    gepa_max_threads: Optional[int] = None
    gepa_reflection_minibatch_size: int = 8
    gepa_add_format_failure_as_feedback: bool = True

    # MIPRO-specific settings
    mipro_auto: str = "medium"  # 'light', 'medium', 'heavy'
    mipro_view_data_batch_size: int = 3  # examples shown to data-aware proposer
    mipro_data_aware_proposer: bool = True
    mipro_max_example_chars: int = 0  # 0 disables truncation; enable explicitly if needed
    mipro_drop_optional_original_content: bool = True
    max_metric_calls: Optional[int] = None  # Direct control (overrides gepa_auto budget)
    reflection_lm_model: Optional[str] = None  # Use same LM if None
    enable_merge: bool = True
    max_merge_invocations: int = 5
    track_stats: bool = True
    log_dir: Optional[Path] = None

    # Auto-selection thresholds (when optimizer_type='auto')
    auto_select_enabled: bool = False
    bootstrap_threshold: int = 10      # Dataset size <= this: use bootstrap
    random_search_threshold: int = 120  # Dataset size <= this: use bootstrap_random_search
    mipro_threshold: int = 200         # Dataset size <= this: use mipro
    # Dataset size > mipro_threshold: use gepa

    # Parallel module optimization
    parallel_modules: bool = False
    parallel_max_concurrent: int = 3
    module_weight_allocation: Dict[str, float] = field(default_factory=dict)

    # Ensemble settings (future)
    ensemble_enabled: bool = False
    ensemble_optimizers: List[str] = field(default_factory=list)
    ensemble_reduce_fn: str = "majority"  # 'majority', 'weighted_average', 'best_score'
    ensemble_weights: Dict[str, float] = field(default_factory=dict)

    # Training data settings
    min_training_examples: int = 4
    balance_ratio: float = 1.0
    max_examples: int = 100

    # Dataset splitting (for random sampling)
    train_samples: Optional[int] = None  # Number of train examples (random draw, None = use all)
    val_samples: Optional[int] = None    # Number of validation examples (random draw)
    test_samples: Optional[int] = None   # Number of test examples (random draw)

    # LabeledFewShot settings
    labeled_k: int = 8  # Number of demos for LabeledFewShot optimizer

    # Stage control
    infer_compile: bool = True
    rank_compile: bool = True

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: Path = field(
        default_factory=lambda: Path("data/oracle_irr_checkpoints")
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {}
        for k, v in vars(self).items():
            if not k.startswith('_'):
                d[k] = str(v) if isinstance(v, Path) else v
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationConfig':
        """Create from dictionary."""
        data = dict(data)
        if 'checkpoint_dir' in data and data['checkpoint_dir']:
            data['checkpoint_dir'] = Path(data['checkpoint_dir'])
        if 'log_dir' in data and data['log_dir']:
            data['log_dir'] = Path(data['log_dir'])
        return cls(**data)


@dataclass
class TrainingDataConfig:
    """Configuration for training data collection."""

    # Source weighting
    node_human_confidence: float = 1.0
    document_label_confidence: float = 0.75
    oracle_auto_confidence: float = 0.6

    # Document label source settings
    error_threshold_high: float = 30.0  # Errors > this are violations
    error_threshold_low: float = 10.0   # Errors < this are good

    # Oracle auto-review settings
    min_auto_confidence: float = 0.9
    require_verification: bool = True

    # Balancing
    balance_positive_negative: bool = True
    target_balance_ratio: float = 1.0

    # Persistence
    training_data_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {}
        for k, v in vars(self).items():
            if not k.startswith('_'):
                d[k] = str(v) if isinstance(v, Path) else v
        return d


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning with human-in-the-loop."""

    # Retraining triggers
    retrain_threshold: int = 10  # Retrain after N new examples
    min_examples_for_retrain: int = 5  # Minimum examples needed

    # Review confidence thresholds
    human_review_threshold: float = 0.6  # Below this, request human review
    auto_apply_threshold: float = 0.8  # Above this, auto-apply decisions

    # Feedback collapse prevention
    include_auto_reviewed: bool = False  # Default: exclude to prevent collapse
    auto_reviewed_discount: float = 0.5  # Confidence discount for auto-reviewed

    # Data limits
    max_examples_per_retrain: int = 100
    balance_positive_negative: bool = True

    # Persistence
    state_file: Optional[Path] = None

    # Embedded optimization config
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'retrain_threshold': self.retrain_threshold,
            'min_examples_for_retrain': self.min_examples_for_retrain,
            'human_review_threshold': self.human_review_threshold,
            'auto_apply_threshold': self.auto_apply_threshold,
            'include_auto_reviewed': self.include_auto_reviewed,
            'auto_reviewed_discount': self.auto_reviewed_discount,
            'max_examples_per_retrain': self.max_examples_per_retrain,
            'balance_positive_negative': self.balance_positive_negative,
            'state_file': str(self.state_file) if self.state_file else None,
            'optimization': self.optimization.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OnlineLearningConfig':
        """Create from dictionary."""
        data = dict(data)
        if data.get('state_file'):
            data['state_file'] = Path(data['state_file'])
        if 'optimization' in data:
            data['optimization'] = OptimizationConfig.from_dict(data['optimization'])
        return cls(**data)


@dataclass
class FrameworkConfig:
    """Master configuration combining all sub-configs."""

    oracle_irr: OracleIRRConfig = field(default_factory=OracleIRRConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    training_data: TrainingDataConfig = field(default_factory=TrainingDataConfig)
    online_learning: OnlineLearningConfig = field(default_factory=OnlineLearningConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'oracle_irr': self.oracle_irr.to_dict(),
            'optimization': self.optimization.to_dict(),
            'training_data': self.training_data.to_dict(),
            'online_learning': self.online_learning.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FrameworkConfig':
        """Create from dictionary."""
        return cls(
            oracle_irr=OracleIRRConfig.from_dict(data.get('oracle_irr', {})),
            optimization=OptimizationConfig.from_dict(data.get('optimization', {})),
            training_data=TrainingDataConfig(**data.get('training_data', {})),
            online_learning=OnlineLearningConfig.from_dict(data.get('online_learning', {})),
        )
