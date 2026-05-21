"""
Training data source implementations for the Oracle Approximation Training Framework.

This module provides concrete implementations of TrainingDataSource for
different types of training data:
- NodeLevelHumanSource: Human-reviewed audit failures
- FullDocumentLabelSource: Bootstrap from document-level labels
- OracleAutoReviewSource: High-confidence auto-reviewed items
- UnifiedTrainingCollector: Aggregates multiple sources
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict, Any

import dspy

from treepo._research.training.core import (
    UnifiedTrainingExample,
    TrainingExampleLabel,
    ViolationType,
    TrainingDataSource,
)
from treepo._research.training.config import TrainingDataConfig
from treepo._research.tasks.base import ScaleDefinition
from treepo._research.training.labeling import ThresholdLabeler

logger = logging.getLogger(__name__)


class NodeLevelHumanSource:
    """
    Training data from human-reviewed node-level audit failures.

    Integrates with existing ReviewQueue and FlaggedItem infrastructure.
    Only includes items that were reviewed by humans (not auto-reviewed)
    to prevent feedback collapse.
    """

    def __init__(self, review_queue: 'ReviewQueue'):
        """
        Args:
            review_queue: ReviewQueue instance with reviewed items
        """
        self.review_queue = review_queue
        self._source_type = "node_human"

    @property
    def source_type(self) -> str:
        return self._source_type

    def get_examples(self) -> List[UnifiedTrainingExample]:
        """Extract reviewed items from queue as training examples."""
        examples = []

        for item in self.review_queue.get_reviewed_items():
            # Skip auto-reviewed to prevent feedback collapse
            if getattr(item, 'review_source', 'human') == 'oracle_func_auto':
                continue

            # Determine label from review result
            # review_result=True means approved (false positive) -> NEGATIVE
            # review_result=False means needs fix (true violation) -> POSITIVE
            label = (
                TrainingExampleLabel.NEGATIVE if item.review_result
                else TrainingExampleLabel.POSITIVE
            )

            example = UnifiedTrainingExample(
                example_id=f"node_{item.item_id}",
                source_type=self._source_type,
                original_content=item.input_a,
                summary=item.input_b,
                rubric=item.rubric,
                context={
                    "node_id": getattr(item, 'node_id', None),
                    "tree_id": getattr(item, 'tree_id', None),
                    "node_level": getattr(item, 'node_level', None),
                    "check_type": item.check_type,
                    "approx_discrepancy": item.approx_discrepancy,
                },
                label=label,
                violation_type=ViolationType.from_check_type(item.check_type),
                corrected_summary=item.corrected_summary,
                human_reasoning=item.review_reasoning,
                confidence=1.0,  # Human reviews are highest confidence
                timestamp=item.reviewed_at,
            )
            examples.append(example)

        return examples

    def get_positive_examples(self) -> List[UnifiedTrainingExample]:
        """Return only positive (true violation) examples."""
        return [e for e in self.get_examples()
                if e.label == TrainingExampleLabel.POSITIVE]

    def get_negative_examples(self) -> List[UnifiedTrainingExample]:
        """Return only negative (false positive) examples."""
        return [e for e in self.get_examples()
                if e.label == TrainingExampleLabel.NEGATIVE]


class FullDocumentLabelSource:
    """
    Training data from full-document ground truth labels.

    Uses end-to-end task performance to bootstrap node-level training examples.
    Error thresholds are normalized (0-1 scale representing percentage of scale range).

    Strategy:
    - Low error predictions -> negative examples (summaries worked)
    - High error predictions -> positive examples (info lost somewhere)
    """

    def __init__(
        self,
        scale: Optional[ScaleDefinition] = None,
        error_threshold_high: float = 0.3,  # Normalized: 30% of scale range
        error_threshold_low: float = 0.1,   # Normalized: 10% of scale range
        confidence: float = 0.75,
    ):
        """
        Args:
            scale: ScaleDefinition for the task (used to normalize errors)
            error_threshold_high: Normalized threshold (0-1); errors > this are violations
            error_threshold_low: Normalized threshold (0-1); errors < this are good
            confidence: Confidence level for bootstrapped examples
        """
        self.scale = scale
        self.confidence = confidence
        self._source_type = "document_label"
        self._results: List[Dict[str, Any]] = []
        self._labeler = ThresholdLabeler(
            threshold_high=error_threshold_high,
            threshold_low=error_threshold_low,
        )

    @property
    def source_type(self) -> str:
        return self._source_type

    def add_result(
        self,
        document_id: str,
        predicted_value: float,
        ground_truth_value: float,
        final_summary: str,
        original_content: Optional[str] = None,
        rubric: str = "Preserve task-relevant information",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a processed document result.

        Args:
            document_id: Unique identifier for the document
            predicted_value: Model's prediction (task score)
            ground_truth_value: True label
            final_summary: The summary produced by OPS
            original_content: Original document content (if available)
            rubric: The rubric used for summarization
            metadata: Additional metadata
        """
        self._results.append({
            'document_id': document_id,
            'predicted': predicted_value,
            'ground_truth': ground_truth_value,
            'final_summary': final_summary,
            'original_content': original_content or "",
            'rubric': rubric,
            'metadata': metadata or {},
        })

    def get_examples(self) -> List[UnifiedTrainingExample]:
        """Bootstrap node-level examples from document-level labels."""
        examples = []

        for result in self._results:
            predicted = result['predicted']
            ground_truth = result['ground_truth']

            if predicted is None:
                continue

            raw_error = abs(predicted - ground_truth)

            # Use labeler to determine label (handles normalization internally)
            label = self._labeler.label_from_error(raw_error, scale=self.scale)
            if label is None:
                continue  # Ambiguous - skip

            # Compute normalized error for reporting
            if self.scale:
                normalized_error = raw_error / self.scale.range
            else:
                normalized_error = raw_error

            if label == TrainingExampleLabel.NEGATIVE:
                # Good prediction -> summary preserved info (negative example)
                examples.append(UnifiedTrainingExample(
                    example_id=f"doc_{result['document_id']}_good",
                    source_type=self._source_type,
                    original_content=result['original_content'] or "",  # Use full text
                    summary=result['final_summary'],
                    rubric=result['rubric'],
                    context={
                        'document_id': result['document_id'],
                        'predicted': predicted,
                        'ground_truth': ground_truth,
                        'error': raw_error,
                        'normalized_error': normalized_error,
                        **result.get('metadata', {}),
                    },
                    label=TrainingExampleLabel.NEGATIVE,
                    violation_type=ViolationType.NONE,
                    human_reasoning=f"Predicted {predicted:.1f}, actual {ground_truth:.1f}, error {raw_error:.1f} ({normalized_error:.1%} of scale) - acceptable",
                    confidence=self.confidence,
                ))

            else:  # label == TrainingExampleLabel.POSITIVE
                # Bad prediction -> info lost somewhere (positive example)
                examples.append(UnifiedTrainingExample(
                    example_id=f"doc_{result['document_id']}_violation",
                    source_type=self._source_type,
                    original_content=result['original_content'] or "",  # Use full text
                    summary=result['final_summary'],
                    rubric=result['rubric'],
                    context={
                        'document_id': result['document_id'],
                        'predicted': predicted,
                        'ground_truth': ground_truth,
                        'error': raw_error,
                        'normalized_error': normalized_error,
                        **result.get('metadata', {}),
                    },
                    label=TrainingExampleLabel.POSITIVE,
                    violation_type=ViolationType.SUFFICIENCY,  # Most likely cause
                    human_reasoning=f"Predicted {predicted:.1f}, actual {ground_truth:.1f}, error {raw_error:.1f} ({normalized_error:.1%} of scale) - information loss",
                    corrected_summary=f"[Expected result: {ground_truth:.1f}]",
                    confidence=self.confidence * 0.9,  # Slightly lower for violations
                ))

        return examples

    def get_positive_examples(self) -> List[UnifiedTrainingExample]:
        return [e for e in self.get_examples()
                if e.label == TrainingExampleLabel.POSITIVE]

    def get_negative_examples(self) -> List[UnifiedTrainingExample]:
        return [e for e in self.get_examples()
                if e.label == TrainingExampleLabel.NEGATIVE]


class OracleAutoReviewSource:
    """
    Training data from high-confidence oracle auto-reviews.

    IMPORTANT: Use carefully to avoid feedback collapse!
    Only include auto-reviews that were later verified by humans or
    have very high confidence.
    """

    def __init__(
        self,
        min_confidence: float = 0.9,
        require_verification: bool = True,
        confidence: float = 0.6,
    ):
        """
        Args:
            min_confidence: Minimum oracle confidence to include
            require_verification: If True, only include human-verified auto-reviews
            confidence: Base confidence for auto-reviewed examples
        """
        self.min_confidence = min_confidence
        self.require_verification = require_verification
        self.base_confidence = confidence
        self._source_type = "oracle_auto"
        self._reviews: List[Dict[str, Any]] = []
        self._verified: Set[str] = set()

    @property
    def source_type(self) -> str:
        return self._source_type

    def add_review(
        self,
        item_id: str,
        original_content: str,
        summary: str,
        rubric: str,
        check_type: str,
        is_violation: bool,
        confidence: float,
        reasoning: str,
        corrected_summary: Optional[str] = None,
    ) -> None:
        """Add an auto-reviewed item."""
        if confidence >= self.min_confidence:
            self._reviews.append({
                'item_id': item_id,
                'original_content': original_content,
                'summary': summary,
                'rubric': rubric,
                'check_type': check_type,
                'is_violation': is_violation,
                'confidence': confidence,
                'reasoning': reasoning,
                'corrected_summary': corrected_summary,
            })

    def verify(self, item_id: str) -> None:
        """Mark an auto-review as human-verified."""
        self._verified.add(item_id)

    def get_examples(self) -> List[UnifiedTrainingExample]:
        """Get examples from verified or high-confidence auto-reviews."""
        examples = []

        for review in self._reviews:
            item_id = review['item_id']

            if self.require_verification and item_id not in self._verified:
                continue

            is_verified = item_id in self._verified
            confidence = self.base_confidence + 0.3 if is_verified else self.base_confidence

            examples.append(UnifiedTrainingExample(
                example_id=f"auto_{item_id}",
                source_type=self._source_type,
                original_content=review['original_content'],
                summary=review['summary'],
                rubric=review['rubric'],
                context={
                    'auto_confidence': review['confidence'],
                    'verified': is_verified,
                },
                label=(TrainingExampleLabel.POSITIVE if review['is_violation']
                       else TrainingExampleLabel.NEGATIVE),
                violation_type=ViolationType.from_check_type(review['check_type']),
                corrected_summary=review['corrected_summary'],
                human_reasoning=review['reasoning'],
                confidence=confidence,
            ))

        return examples

    def get_positive_examples(self) -> List[UnifiedTrainingExample]:
        return [e for e in self.get_examples()
                if e.label == TrainingExampleLabel.POSITIVE]

    def get_negative_examples(self) -> List[UnifiedTrainingExample]:
        return [e for e in self.get_examples()
                if e.label == TrainingExampleLabel.NEGATIVE]


class UnifiedTrainingCollector:
    """
    Collects and manages training data from multiple sources.

    Responsibilities:
    - Aggregate examples from all registered sources
    - Balance positive/negative examples
    - Weight examples by source trust score
    - Export to DSPy format
    - Persist/load training data
    """

    def __init__(self, config: Optional[TrainingDataConfig] = None):
        """
        Args:
            config: Training data configuration
        """
        self.config = config or TrainingDataConfig()
        self._sources: List[TrainingDataSource] = []
        self._cache: Optional[List[UnifiedTrainingExample]] = None

    def add_source(self, source: TrainingDataSource) -> None:
        """Register a training data source."""
        self._sources.append(source)
        self._cache = None  # Invalidate cache

    def clear_cache(self) -> None:
        """Clear the example cache to force refresh."""
        self._cache = None

    def get_all_examples(self) -> List[UnifiedTrainingExample]:
        """Collect examples from all sources."""
        if self._cache is not None:
            return self._cache

        examples = []
        source_counts = []
        for source in self._sources:
            source_examples = source.get_examples()
            examples.extend(source_examples)
            if source_examples:  # Only track non-empty sources
                source_name = source.source_type or "unnamed"
                source_counts.append(f"{len(source_examples)} from {source_name}")

        if source_counts:
            logger.info(f"Collected {len(examples)} examples ({', '.join(source_counts)})")
        else:
            logger.info(f"Collected {len(examples)} examples")

        self._cache = examples
        return examples

    def get_positive_examples(self) -> List[UnifiedTrainingExample]:
        """Return all positive (true violation) examples."""
        return [e for e in self.get_all_examples()
                if e.label == TrainingExampleLabel.POSITIVE]

    def get_negative_examples(self) -> List[UnifiedTrainingExample]:
        """Return all negative (false positive) examples."""
        return [e for e in self.get_all_examples()
                if e.label == TrainingExampleLabel.NEGATIVE]

    def get_balanced_examples(
        self,
        max_examples: Optional[int] = None,
        balance_ratio: float = 1.0,
        weight_by_confidence: bool = True,
    ) -> List[UnifiedTrainingExample]:
        """
        Get balanced dataset with optional weighting.

        Args:
            max_examples: Maximum total examples to return
            balance_ratio: Ratio of negative to positive (1.0 = equal)
            weight_by_confidence: If True, prefer high-confidence examples

        Returns:
            Balanced list of training examples
        """
        positives = self.get_positive_examples()
        negatives = self.get_negative_examples()

        if not positives:
            logger.warning("No positive examples available")
            return list(negatives)
        if not negatives:
            logger.warning("No negative examples available")
            return list(positives)

        # Sort by confidence if weighting
        if weight_by_confidence:
            positives = sorted(positives, key=lambda x: x.confidence, reverse=True)
            negatives = sorted(negatives, key=lambda x: x.confidence, reverse=True)

        # Calculate balanced counts
        max_examples = max_examples or (len(positives) + len(negatives))
        n_positive = min(len(positives), max_examples // 2)
        n_negative = min(int(n_positive * balance_ratio), len(negatives))

        # Ensure we don't exceed max
        if n_positive + n_negative > max_examples:
            scale = max_examples / (n_positive + n_negative)
            n_positive = int(n_positive * scale)
            n_negative = int(n_negative * scale)

        # Sample
        sampled_positive = positives[:n_positive]
        sampled_negative = negatives[:n_negative]

        result = sampled_positive + sampled_negative
        random.shuffle(result)

        return result

    def get_dspy_trainset(
        self,
        max_examples: int = 50,
        balanced: bool = True,
    ) -> List[dspy.Example]:
        """
        Export as DSPy training set.

        Args:
            max_examples: Maximum examples to export
            balanced: Whether to balance positive/negative

        Returns:
            List of DSPy Example objects
        """
        if balanced:
            examples = self.get_balanced_examples(max_examples)
        else:
            examples = self.get_all_examples()[:max_examples]

        return [e.to_dspy_example() for e in examples]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected training data."""
        all_examples = self.get_all_examples()
        positives = self.get_positive_examples()
        negatives = self.get_negative_examples()

        # Group by source
        by_source = {}
        for e in all_examples:
            by_source[e.source_type] = by_source.get(e.source_type, 0) + 1

        # Group by violation type
        by_type = {}
        for e in all_examples:
            by_type[e.violation_type.value] = by_type.get(e.violation_type.value, 0) + 1

        return {
            'total_examples': len(all_examples),
            'positive_examples': len(positives),
            'negative_examples': len(negatives),
            'balance_ratio': len(negatives) / len(positives) if positives else 0.0,
            'by_source': by_source,
            'by_violation_type': by_type,
            'avg_confidence_positive': (
                sum(e.confidence for e in positives) / len(positives)
                if positives else 0.0
            ),
            'avg_confidence_negative': (
                sum(e.confidence for e in negatives) / len(negatives)
                if negatives else 0.0
            ),
        }

    def save(self, filepath: Path) -> None:
        """Save all training data to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        examples = self.get_all_examples()
        data = {
            'saved_at': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'examples': [e.to_dict() for e in examples],
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(examples)} training examples to {filepath}")

    def load(self, filepath: Path) -> int:
        """
        Load training data from JSON file.

        Returns:
            Number of examples loaded
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Training data file not found: {filepath}")
            return 0

        with open(filepath) as f:
            data = json.load(f)

        # Create a simple source from loaded examples
        loaded_examples = [
            UnifiedTrainingExample.from_dict(e)
            for e in data.get('examples', [])
        ]

        if loaded_examples:
            self.add_source(_LoadedExamplesSource(loaded_examples))

        logger.info(f"Loaded {len(loaded_examples)} examples from {filepath}")
        return len(loaded_examples)

    def __len__(self) -> int:
        return len(self.get_all_examples())


class _LoadedExamplesSource:
    """Internal source for loaded examples."""

    def __init__(self, examples: List[UnifiedTrainingExample]):
        self._examples = examples
        self._source_type = "loaded"

    @property
    def source_type(self) -> str:
        return self._source_type

    def get_examples(self) -> List[UnifiedTrainingExample]:
        return self._examples

    def get_positive_examples(self) -> List[UnifiedTrainingExample]:
        return [e for e in self._examples if e.label == TrainingExampleLabel.POSITIVE]

    def get_negative_examples(self) -> List[UnifiedTrainingExample]:
        return [e for e in self._examples if e.label == TrainingExampleLabel.NEGATIVE]
