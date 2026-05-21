"""
Synthetic Data Generation for Oracle-Preserving Summarization Training.

Inspired by NVIDIA Nemotron-3's synthetic data approach:
1. Use a large oracle model to generate training tasks/challenges
2. Use the same model to provide high-quality solutions
3. Train smaller models on these (task, solution) pairs

For OPS, this translates to:
1. Step 1 - Challenge Generation: Given a document and rubric, the large model:
   - Identifies critical information that MUST be preserved
   - Generates potential pitfalls (what might get lost during compression)
   - Creates verification questions

2. Step 2 - Reference Generation: The large model:
   - Produces a high-quality summary preserving the critical information
   - Provides reasoning about preservation choices

3. Training: Small model learns to produce similar quality summaries

Key components:
- OracleChallenge: Represents a summarization challenge
- ReferenceSummary: High-quality summary with preservation reasoning
- SyntheticDataGenerator: Orchestrates the generation pipeline
- SyntheticDataset: Manages generated examples for training
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import dspy

logger = logging.getLogger(__name__)


# =============================================================================
# DSPy Signatures for Synthetic Data Generation
# =============================================================================

class IdentifyCriticalInfo(dspy.Signature):
    """
    Identify information that MUST be preserved during summarization.

    Given a document and a rubric describing what to preserve,
    identify the specific critical pieces of information.
    """
    document: str = dspy.InputField(
        desc="Full document to be summarized"
    )
    rubric: str = dspy.InputField(
        desc="Description of what information must be preserved (e.g., political stance, key claims)"
    )

    critical_elements: str = dspy.OutputField(
        desc="List of specific critical information elements that MUST be preserved"
    )
    preservation_rationale: str = dspy.OutputField(
        desc="Why each element is critical for the rubric"
    )
    potential_pitfalls: str = dspy.OutputField(
        desc="Common ways this information might get lost or distorted during summarization"
    )


class GenerateVerificationQuestions(dspy.Signature):
    """
    Generate questions to verify if a summary preserves critical information.

    These questions can be used to evaluate summary quality.
    """
    document: str = dspy.InputField(
        desc="Original document"
    )
    rubric: str = dspy.InputField(
        desc="What information must be preserved"
    )
    critical_elements: str = dspy.InputField(
        desc="Identified critical information elements"
    )

    verification_questions: str = dspy.OutputField(
        desc="3-5 questions that can verify if a summary preserves the critical information"
    )
    expected_answers: str = dspy.OutputField(
        desc="Expected answers to the verification questions based on the original document"
    )


class GenerateReferenceSummary(dspy.Signature):
    """
    Generate a high-quality summary that preserves critical information.

    The summary should be concise while maintaining all elements
    identified as critical for the rubric.
    """
    document: str = dspy.InputField(
        desc="Full document to summarize"
    )
    rubric: str = dspy.InputField(
        desc="What information must be preserved"
    )
    critical_elements: str = dspy.InputField(
        desc="Specific elements that must be preserved"
    )
    target_compression: float = dspy.InputField(
        desc="Target compression ratio (e.g., 0.2 means summary should be ~20% of original length)"
    )

    summary: str = dspy.OutputField(
        desc="High-quality summary preserving all critical elements"
    )
    preservation_notes: str = dspy.OutputField(
        desc="How each critical element was preserved or represented in the summary"
    )
    compression_achieved: float = dspy.OutputField(
        desc="Actual compression ratio achieved"
    )


class EvaluateSummaryQuality(dspy.Signature):
    """
    Evaluate whether a summary preserves critical information.

    Used for validating generated summaries and comparing candidates.
    """
    document: str = dspy.InputField(
        desc="Original document"
    )
    summary: str = dspy.InputField(
        desc="Summary to evaluate"
    )
    rubric: str = dspy.InputField(
        desc="What information should be preserved"
    )
    critical_elements: str = dspy.InputField(
        desc="Specific elements that should be preserved"
    )
    verification_questions: str = dspy.InputField(
        desc="Questions to verify preservation"
    )

    preservation_score: float = dspy.OutputField(
        desc="Score from 0-100 indicating how well critical information is preserved"
    )
    element_checklist: str = dspy.OutputField(
        desc="For each critical element, whether it was preserved (yes/no/partial)"
    )
    quality_reasoning: str = dspy.OutputField(
        desc="Detailed reasoning about the quality of preservation"
    )
    improvement_suggestions: str = dspy.OutputField(
        desc="How the summary could better preserve the critical information"
    )


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class OracleChallenge:
    """
    Represents a summarization challenge with identified critical information.

    This is the output of Step 1: Challenge Generation.
    """
    challenge_id: str
    document: str
    rubric: str

    # Identified by large model
    critical_elements: str
    preservation_rationale: str
    potential_pitfalls: str
    verification_questions: str
    expected_answers: str

    # Metadata
    document_length: int = 0
    timestamp: Optional[str] = None
    oracle_model: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.document_length == 0:
            self.document_length = len(self.document)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "document": self.document,
            "rubric": self.rubric,
            "critical_elements": self.critical_elements,
            "preservation_rationale": self.preservation_rationale,
            "potential_pitfalls": self.potential_pitfalls,
            "verification_questions": self.verification_questions,
            "expected_answers": self.expected_answers,
            "document_length": self.document_length,
            "timestamp": self.timestamp,
            "oracle_model": self.oracle_model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OracleChallenge':
        return cls(**data)


@dataclass
class ReferenceSummary:
    """
    A high-quality reference summary generated by the large oracle model.

    This is the output of Step 2: Reference Generation.
    """
    reference_id: str
    challenge_id: str

    # The summary and its metadata
    summary: str
    preservation_notes: str
    compression_ratio: float

    # Quality validation
    preservation_score: float
    element_checklist: str
    quality_reasoning: str

    # Metadata
    target_compression: float = 0.2
    timestamp: Optional[str] = None
    oracle_model: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "challenge_id": self.challenge_id,
            "summary": self.summary,
            "preservation_notes": self.preservation_notes,
            "compression_ratio": self.compression_ratio,
            "preservation_score": self.preservation_score,
            "element_checklist": self.element_checklist,
            "quality_reasoning": self.quality_reasoning,
            "target_compression": self.target_compression,
            "timestamp": self.timestamp,
            "oracle_model": self.oracle_model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReferenceSummary':
        return cls(**data)


@dataclass
class SyntheticExample:
    """
    A complete synthetic training example combining challenge and reference.

    This is what gets used for training the small model.
    """
    example_id: str

    # Input for training
    document: str
    rubric: str

    # Target output for training
    reference_summary: str

    # Additional context (can be used for enhanced training)
    critical_elements: str
    preservation_notes: str

    # Quality metrics
    preservation_score: float
    compression_ratio: float

    # Metadata
    oracle_model: str = ""
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "document": self.document,
            "rubric": self.rubric,
            "reference_summary": self.reference_summary,
            "critical_elements": self.critical_elements,
            "preservation_notes": self.preservation_notes,
            "preservation_score": self.preservation_score,
            "compression_ratio": self.compression_ratio,
            "oracle_model": self.oracle_model,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyntheticExample':
        return cls(**data)

    def to_dspy_example(self) -> dspy.Example:
        """Convert to DSPy example for training."""
        return dspy.Example(
            content=self.document,
            rubric=self.rubric,
            summary=self.reference_summary,
            critical_elements=self.critical_elements,
        ).with_inputs("content", "rubric")


# =============================================================================
# Generation Modules
# =============================================================================

class ChallengeGenerator(dspy.Module):
    """
    Generates summarization challenges from documents.

    Step 1 of the synthetic data pipeline.
    """

    def __init__(self, use_cot: bool = True):
        super().__init__()
        predictor = dspy.ChainOfThought if use_cot else dspy.Predict
        self.identify_critical = predictor(IdentifyCriticalInfo)
        self.generate_questions = predictor(GenerateVerificationQuestions)

    def forward(
        self,
        document: str,
        rubric: str,
        oracle_model: str = "",
    ) -> OracleChallenge:
        """
        Generate a summarization challenge for the given document.

        Args:
            document: Document to create challenge for
            rubric: What information must be preserved
            oracle_model: Name of the oracle model being used

        Returns:
            OracleChallenge with critical info and verification questions
        """
        # Step 1a: Identify critical information
        critical_result = self.identify_critical(
            document=document,
            rubric=rubric,
        )

        # Step 1b: Generate verification questions
        questions_result = self.generate_questions(
            document=document,
            rubric=rubric,
            critical_elements=critical_result.critical_elements,
        )

        # Create challenge
        challenge_id = f"challenge_{hash(document) % 1000000:06d}"

        return OracleChallenge(
            challenge_id=challenge_id,
            document=document,
            rubric=rubric,
            critical_elements=str(critical_result.critical_elements),
            preservation_rationale=str(critical_result.preservation_rationale),
            potential_pitfalls=str(critical_result.potential_pitfalls),
            verification_questions=str(questions_result.verification_questions),
            expected_answers=str(questions_result.expected_answers),
            oracle_model=oracle_model,
        )


class ReferenceGenerator(dspy.Module):
    """
    Generates high-quality reference summaries for challenges.

    Step 2 of the synthetic data pipeline.
    """

    def __init__(self, use_cot: bool = True):
        super().__init__()
        predictor = dspy.ChainOfThought if use_cot else dspy.Predict
        self.generate_summary = predictor(GenerateReferenceSummary)
        self.evaluate_quality = predictor(EvaluateSummaryQuality)

    def forward(
        self,
        challenge: OracleChallenge,
        target_compression: float = 0.2,
        oracle_model: str = "",
    ) -> ReferenceSummary:
        """
        Generate a reference summary for the challenge.

        Args:
            challenge: The summarization challenge
            target_compression: Target compression ratio
            oracle_model: Name of the oracle model being used

        Returns:
            ReferenceSummary with high-quality summary and quality metrics
        """
        # Generate the reference summary
        summary_result = self.generate_summary(
            document=challenge.document,
            rubric=challenge.rubric,
            critical_elements=challenge.critical_elements,
            target_compression=target_compression,
        )

        # Validate the summary quality
        eval_result = self.evaluate_quality(
            document=challenge.document,
            summary=str(summary_result.summary),
            rubric=challenge.rubric,
            critical_elements=challenge.critical_elements,
            verification_questions=challenge.verification_questions,
        )

        # Parse compression ratio
        try:
            compression = float(summary_result.compression_achieved)
        except (ValueError, TypeError):
            compression = len(str(summary_result.summary)) / len(challenge.document)

        # Parse preservation score
        try:
            score = float(eval_result.preservation_score)
            score = max(0, min(100, score))
        except (ValueError, TypeError):
            score = 0.0

        reference_id = f"ref_{challenge.challenge_id}"

        return ReferenceSummary(
            reference_id=reference_id,
            challenge_id=challenge.challenge_id,
            summary=str(summary_result.summary),
            preservation_notes=str(summary_result.preservation_notes),
            compression_ratio=compression,
            preservation_score=score,
            element_checklist=str(eval_result.element_checklist),
            quality_reasoning=str(eval_result.quality_reasoning),
            target_compression=target_compression,
            oracle_model=oracle_model,
        )


class SyntheticDataGenerator:
    """
    Orchestrates the full synthetic data generation pipeline.

    Combines challenge generation and reference generation to produce
    complete training examples.
    """

    def __init__(
        self,
        challenge_generator: Optional[ChallengeGenerator] = None,
        reference_generator: Optional[ReferenceGenerator] = None,
        min_quality_score: float = 70.0,
        target_compression: float = 0.2,
        oracle_model_name: str = "",
    ):
        """
        Initialize the generator.

        Args:
            challenge_generator: Module for generating challenges
            reference_generator: Module for generating references
            min_quality_score: Minimum quality score to accept a reference
            target_compression: Target compression ratio for summaries
            oracle_model_name: Name of the oracle model for metadata
        """
        self.challenge_gen = challenge_generator or ChallengeGenerator()
        self.reference_gen = reference_generator or ReferenceGenerator()
        self.min_quality_score = min_quality_score
        self.target_compression = target_compression
        self.oracle_model_name = oracle_model_name

        self._example_counter = 0
        self._challenges: List[OracleChallenge] = []
        self._references: List[ReferenceSummary] = []
        self._examples: List[SyntheticExample] = []

    def generate_example(
        self,
        document: str,
        rubric: str,
        max_retries: int = 3,
    ) -> Optional[SyntheticExample]:
        """
        Generate a complete synthetic training example.

        Args:
            document: Document to create example from
            rubric: What information must be preserved
            max_retries: Maximum attempts to generate quality example

        Returns:
            SyntheticExample if successful, None if quality threshold not met
        """
        # Step 1: Generate challenge
        try:
            challenge = self.challenge_gen(
                document=document,
                rubric=rubric,
                oracle_model=self.oracle_model_name,
            )
            self._challenges.append(challenge)
        except Exception as e:
            logger.error(f"Failed to generate challenge: {e}")
            return None

        # Step 2: Generate reference (with retries for quality)
        best_reference = None
        best_score = 0.0

        for attempt in range(max_retries):
            try:
                reference = self.reference_gen(
                    challenge=challenge,
                    target_compression=self.target_compression,
                    oracle_model=self.oracle_model_name,
                )

                if reference.preservation_score > best_score:
                    best_score = reference.preservation_score
                    best_reference = reference

                if reference.preservation_score >= self.min_quality_score:
                    break

            except Exception as e:
                logger.warning(f"Reference generation attempt {attempt + 1} failed: {e}")
                continue

        if best_reference is None:
            logger.warning(f"Failed to generate quality reference for challenge {challenge.challenge_id}")
            return None

        if best_reference.preservation_score < self.min_quality_score:
            logger.warning(
                f"Best reference score {best_reference.preservation_score:.1f} "
                f"below threshold {self.min_quality_score}"
            )
            # Still use it but log the warning

        self._references.append(best_reference)

        # Create synthetic example
        self._example_counter += 1
        example = SyntheticExample(
            example_id=f"synth_{self._example_counter:06d}",
            document=document,
            rubric=rubric,
            reference_summary=best_reference.summary,
            critical_elements=challenge.critical_elements,
            preservation_notes=best_reference.preservation_notes,
            preservation_score=best_reference.preservation_score,
            compression_ratio=best_reference.compression_ratio,
            oracle_model=self.oracle_model_name,
        )
        self._examples.append(example)

        return example

    def generate_batch(
        self,
        documents: List[str],
        rubric: str,
        max_retries: int = 3,
    ) -> List[SyntheticExample]:
        """
        Generate examples for a batch of documents.

        Args:
            documents: List of documents
            rubric: What information must be preserved
            max_retries: Maximum attempts per document

        Returns:
            List of successfully generated examples
        """
        examples = []
        for i, doc in enumerate(documents):
            logger.info(f"Generating example {i + 1}/{len(documents)}")
            example = self.generate_example(doc, rubric, max_retries)
            if example is not None:
                examples.append(example)

        logger.info(f"Generated {len(examples)}/{len(documents)} examples")
        return examples

    def get_all_examples(self) -> List[SyntheticExample]:
        """Return all generated examples."""
        return self._examples

    def get_high_quality_examples(self, threshold: float = 80.0) -> List[SyntheticExample]:
        """Return examples above quality threshold."""
        return [e for e in self._examples if e.preservation_score >= threshold]

    def get_statistics(self) -> Dict[str, Any]:
        """Return generation statistics."""
        if not self._examples:
            return {"total": 0}

        scores = [e.preservation_score for e in self._examples]
        compressions = [e.compression_ratio for e in self._examples]

        return {
            "total_challenges": len(self._challenges),
            "total_references": len(self._references),
            "total_examples": len(self._examples),
            "avg_quality_score": sum(scores) / len(scores),
            "min_quality_score": min(scores),
            "max_quality_score": max(scores),
            "avg_compression": sum(compressions) / len(compressions),
            "high_quality_count": sum(1 for s in scores if s >= 80),
        }


# =============================================================================
# Dataset Management
# =============================================================================

class SyntheticDataset:
    """
    Manages synthetic training data.

    Supports saving, loading, filtering, and conversion to training formats.
    """

    def __init__(self, examples: Optional[List[SyntheticExample]] = None):
        """
        Initialize the dataset.

        Args:
            examples: Initial list of examples
        """
        self.examples = examples or []

    def add_example(self, example: SyntheticExample):
        """Add an example to the dataset."""
        self.examples.append(example)

    def add_examples(self, examples: List[SyntheticExample]):
        """Add multiple examples."""
        self.examples.extend(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SyntheticExample:
        return self.examples[idx]

    def filter_by_quality(self, min_score: float) -> 'SyntheticDataset':
        """Return new dataset with examples above quality threshold."""
        filtered = [e for e in self.examples if e.preservation_score >= min_score]
        return SyntheticDataset(filtered)

    def split(
        self,
        train_ratio: float = 0.8,
        shuffle: bool = True,
    ) -> Tuple['SyntheticDataset', 'SyntheticDataset']:
        """
        Split into train and validation sets.

        Args:
            train_ratio: Fraction for training set
            shuffle: Whether to shuffle before splitting

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        examples = self.examples.copy()
        if shuffle:
            random.shuffle(examples)

        split_idx = int(len(examples) * train_ratio)
        return (
            SyntheticDataset(examples[:split_idx]),
            SyntheticDataset(examples[split_idx:]),
        )

    def to_dspy_examples(self) -> List[dspy.Example]:
        """Convert to DSPy examples for training."""
        return [e.to_dspy_example() for e in self.examples]

    def to_sft_format(self) -> List[Dict[str, Any]]:
        """
        Convert to SFT (Supervised Fine-Tuning) format.

        Returns:
            List of dicts with prompt and completion
        """
        sft_data = []
        for example in self.examples:
            prompt = f"""Summarize the following text while preserving: {example.rubric}

Critical information to preserve:
{example.critical_elements}

Text:
{example.document}

Summary:"""

            sft_data.append({
                "prompt": prompt,
                "completion": example.reference_summary,
                "metadata": {
                    "example_id": example.example_id,
                    "preservation_score": example.preservation_score,
                    "compression_ratio": example.compression_ratio,
                },
            })

        return sft_data

    def save(self, path: Path):
        """Save dataset to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "num_examples": len(self.examples),
            "examples": [e.to_dict() for e in self.examples],
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.examples)} synthetic examples to {path}")

    @classmethod
    def load(cls, path: Path) -> 'SyntheticDataset':
        """Load dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)

        examples = [SyntheticExample.from_dict(e) for e in data["examples"]]
        logger.info(f"Loaded {len(examples)} synthetic examples from {path}")

        return cls(examples)

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics about the dataset."""
        if not self.examples:
            return {"total": 0}

        scores = [e.preservation_score for e in self.examples]
        compressions = [e.compression_ratio for e in self.examples]

        return {
            "total_examples": len(self.examples),
            "avg_quality_score": sum(scores) / len(scores),
            "min_quality_score": min(scores),
            "max_quality_score": max(scores),
            "avg_compression": sum(compressions) / len(compressions),
            "high_quality_count": sum(1 for s in scores if s >= 80),
            "oracle_models": list(set(e.oracle_model for e in self.examples)),
        }
