"""
Core types and protocols for the Oracle Approximation Training Framework.

This module defines the fundamental data structures and interfaces used
throughout the training framework.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Protocol, runtime_checkable

import dspy


class ViolationType(Enum):
    """Categories of OPS law violations (ontology)."""
    SUFFICIENCY = "sufficiency"        # C1: leaf doesn't preserve rubric info
    MERGE_CONSISTENCY = "merge"        # C3B: internal merge loses info
    IDEMPOTENCE = "idempotence"        # C2: re-summarization changes oracle
    SUBSTITUTION = "substitution"      # C3A: boundary inconsistency
    NONE = "none"                      # No violation (false positive)

    @classmethod
    def from_check_type(cls, check_type: str) -> 'ViolationType':
        """Map audit check type to violation type."""
        mapping = {
            'sufficiency': cls.SUFFICIENCY,
            'merge_consistency': cls.MERGE_CONSISTENCY,
            'merge': cls.MERGE_CONSISTENCY,
            'joint_to_disjoint_drift': cls.MERGE_CONSISTENCY,
            'readout_aggregation_drift': cls.MERGE_CONSISTENCY,
            'idempotence': cls.IDEMPOTENCE,
            'substitution': cls.SUBSTITUTION,
        }
        return mapping.get(check_type.lower(), cls.SUFFICIENCY)


class TrainingExampleLabel(Enum):
    """Label for training examples."""
    POSITIVE = "positive"   # True violation - confirmed needs fixing
    NEGATIVE = "negative"   # False positive - actually acceptable


@dataclass
class UnifiedTrainingExample:
    """
    Unified format for training examples from any source.

    This abstracts over:
    - Node-level human validation (direct examples)
    - Full-document labels (bootstrapped to node level)
    - Oracle approximation auto-reviews (filtered carefully)

    All training data sources convert to this format before being
    used for training.
    """
    # Identifiers
    example_id: str
    source_type: str  # "node_human", "document_label", "oracle_auto"

    # Input features (what the oracle sees)
    original_content: str
    summary: str
    rubric: str

    # Additional context
    context: Dict[str, Any] = field(default_factory=dict)

    # Labels
    label: TrainingExampleLabel = TrainingExampleLabel.POSITIVE
    violation_type: ViolationType = ViolationType.SUFFICIENCY

    # For positive examples: the corrected summary
    corrected_summary: Optional[str] = None
    human_reasoning: Optional[str] = None

    # Metadata
    confidence: float = 1.0  # Source trust score (human=1.0, auto=0.6-0.9)
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dspy_example(self) -> dspy.Example:
        """Convert to DSPy format for training."""
        is_violation = self.label == TrainingExampleLabel.POSITIVE

        # Get label for metric comparison:
        # - For ordinal scales: use discretized_label from context
        # - For categorical: use violation_type
        label = self.context.get('discretized_label')
        if label is None:
            label = self.violation_type.value if is_violation else "none"

        return dspy.Example(
            original_content=self.original_content,
            summary=self.summary,
            rubric=self.rubric,
            check_type=self.violation_type.value,
            is_true_violation=is_violation,
            violation_type=self.violation_type.value if is_violation else "none",
            label=str(label),  # For metric comparison
            confidence=self.confidence,
            corrected_summary=self.corrected_summary or "",
            reasoning=self.human_reasoning or "",
        ).with_inputs("original_content", "summary", "rubric")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'example_id': self.example_id,
            'source_type': self.source_type,
            'original_content': self.original_content,
            'summary': self.summary,
            'rubric': self.rubric,
            'context': self.context,
            'label': self.label.value,
            'violation_type': self.violation_type.value,
            'corrected_summary': self.corrected_summary,
            'human_reasoning': self.human_reasoning,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedTrainingExample':
        """Deserialize from dictionary."""
        data = dict(data)
        data['label'] = TrainingExampleLabel(data.get('label', 'positive'))
        data['violation_type'] = ViolationType(data.get('violation_type', 'sufficiency'))
        return cls(**data)

    @classmethod
    def create_audit_example(
        cls,
        example_id: str,
        check_type: str,
        is_violation: bool,
        original_content: str,
        current_summary: str,
        rubric: str,
        discrepancy: float = 0.0,
        oracle_original: Optional[float] = None,
        oracle_summary: Optional[float] = None,
        node_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> 'UnifiedTrainingExample':
        """
        Create an audit training example directly.

        This is the preferred way to create training examples from audit results,
        replacing the deprecated AuditTrainingExample class.

        Args:
            example_id: Unique identifier for this example
            check_type: Type of check ("sufficiency", "merge", "idempotence", "substitution")
            is_violation: Whether this is a true violation
            original_content: The original text being summarized
            current_summary: The summary that was checked
            rubric: The rubric/criteria for summarization
            discrepancy: How different the oracle values are (0 = same)
            oracle_original: Oracle score on original (optional)
            oracle_summary: Oracle score on summary (optional)
            node_id: Node identifier (optional)
            document_id: Document identifier (optional)

        Returns:
            UnifiedTrainingExample configured for audit use
        """
        return cls(
            example_id=example_id,
            source_type="audit_bootstrap",
            original_content=original_content,
            summary=current_summary,
            rubric=rubric,
            context={
                'oracle_original': oracle_original,
                'oracle_summary': oracle_summary,
                'discrepancy': discrepancy,
                'node_id': node_id,
                'document_id': document_id,
                'check_type': check_type,
            },
            label=TrainingExampleLabel.POSITIVE if is_violation else TrainingExampleLabel.NEGATIVE,
            violation_type=ViolationType.from_check_type(check_type),
            confidence=min(1.0, 0.5 + abs(discrepancy) / 100),  # Higher discrepancy = higher confidence
        )


@runtime_checkable
class TrainingDataSource(Protocol):
    """
    Protocol for training data sources.

    Any class implementing this protocol can be used as a training
    data source for the oracle approximation framework.
    """

    def get_examples(self) -> List[UnifiedTrainingExample]:
        """Return all available training examples."""
        ...

    def get_positive_examples(self) -> List[UnifiedTrainingExample]:
        """Return positive (true violation) examples."""
        ...

    def get_negative_examples(self) -> List[UnifiedTrainingExample]:
        """Return negative (false positive) examples."""
        ...

    @property
    def source_type(self) -> str:
        """Identify the source type."""
        ...


@dataclass
class OracleReviewResult:
    """Result of reviewing a single item with the oracle."""
    item_id: str
    is_violation: bool
    violation_type: ViolationType
    confidence: float
    reasoning: str
    corrected_summary: Optional[str] = None
    candidates: List[str] = field(default_factory=list)

    @property
    def auto_decided(self) -> bool:
        """Whether confidence is high enough for automatic decision."""
        return self.confidence >= 0.8


# Note: LabelSpace, CategoricalLabelSpace, and OrdinalLabelSpace have been removed.
# Use continuous score prediction instead of classification with discretized labels.


# =============================================================================
# Prediction and Verification Results
# =============================================================================

@dataclass
class Prediction:
    """Result of classifying an item."""
    label: str
    confidence: float
    reasoning: str
    raw_scores: Optional[Dict[str, float]] = None  # Per-label scores if available

    def to_dict(self) -> Dict[str, Any]:
        return {
            'label': self.label,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'raw_scores': self.raw_scores,
        }


@dataclass
class LawCheckResult:
    """Result of checking an OPS law at a tree node."""
    law: str  # "sufficiency", "idempotence", "merge_consistency", "substitution"
    passed: bool
    discrepancy: float  # Distance between expected and actual (0 if passed)

    # Predictions that were compared
    original_prediction: Optional[Prediction] = None
    summary_prediction: Optional[Prediction] = None
    expected_label: Optional[str] = None  # For merge consistency

    # Context
    node_id: Optional[str] = None
    reasoning: Optional[str] = None

    # Skipped checks (e.g., no summarizer provided)
    skipped: bool = False
    skip_reason: Optional[str] = None

    @property
    def was_evaluated(self) -> bool:
        """True if the check was actually performed (not skipped)."""
        return not self.skipped

    def to_training_example(
        self,
        original_content: str,
        summary: str,
        rubric: str,
        example_id: str,
    ) -> UnifiedTrainingExample:
        """Convert law check result to training example."""
        return UnifiedTrainingExample(
            example_id=example_id,
            source_type="law_violation",
            original_content=original_content,
            summary=summary,
            rubric=rubric,
            label=TrainingExampleLabel.POSITIVE if not self.passed else TrainingExampleLabel.NEGATIVE,
            violation_type=ViolationType.from_check_type(self.law),
            context={
                'law': self.law,
                'discrepancy': self.discrepancy,
                'node_id': self.node_id,
            },
            human_reasoning=self.reasoning,
            confidence=0.8 if not self.passed else 0.9,  # Slightly lower confidence for violations
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'law': self.law,
            'passed': self.passed,
            'discrepancy': self.discrepancy,
            'original_prediction': self.original_prediction.to_dict() if self.original_prediction else None,
            'summary_prediction': self.summary_prediction.to_dict() if self.summary_prediction else None,
            'expected_label': self.expected_label,
            'node_id': self.node_id,
            'reasoning': self.reasoning,
        }
