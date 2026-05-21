"""
Base Preference Collector for OPS Summarization.

Provides abstract base class that consolidates shared logic across
GenRM-based and Oracle-based preference collectors.
"""

import logging
import random
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import dspy

from treepo._research.core.supervision_metadata import (
    judgment_supervision_metadata as preference_supervision_metadata,
)
from .comparative_types import (
    ComparativeDataset,
    ComparativeJudgmentRecord,
    GenerationConfig,
    PreferencePair,
    PreferenceDataset,
)
from treepo._research.config.constants import DIVERSE_TEMPERATURES
from treepo._research.core.protocols import format_merge_input

logger = logging.getLogger(__name__)


@contextmanager
def _lm_params(lm, temperature: Optional[float] = None, top_p: Optional[float] = None, max_tokens: Optional[int] = None):
    """Context manager to temporarily set LM parameters and restore on exit.

    Args:
        lm: The language model object with kwargs dict
        temperature: Optional temperature to set
        top_p: Optional top_p to set
        max_tokens: Optional max_tokens to set

    Yields:
        None (parameters are set as a side effect)
    """
    if lm is None or not hasattr(lm, "kwargs"):
        yield
        return

    # Save original values
    saved = {}
    for key, val in [("temperature", temperature), ("top_p", top_p), ("max_tokens", max_tokens)]:
        if val is not None:
            saved[key] = lm.kwargs.get(key)
            lm.kwargs[key] = val

    try:
        yield
    finally:
        # Restore original values
        for key, orig_val in saved.items():
            if orig_val is None:
                lm.kwargs.pop(key, None)
            else:
                lm.kwargs[key] = orig_val


# Type variable for candidate metadata (can be GenerationConfig, dict, etc.)
CandidateMeta = TypeVar("CandidateMeta")


@dataclass
class CollectionStatistics:
    """Comprehensive statistics for preference collection.

    Tracks both successful operations and failures for full visibility
    into data quality and collection pipeline health.
    """
    # Example-level counts
    total_examples_attempted: int = 0
    successful_examples: int = 0
    failed_examples: int = 0

    # Generation-level counts
    total_candidates_attempted: int = 0
    successful_generations: int = 0
    failed_generations: int = 0

    # Pair-level counts
    pairs_collected: int = 0
    pairs_dropped_preference_error: int = 0
    pairs_dropped_no_candidates: int = 0
    pairs_dropped_resummary_failed: int = 0
    comparative_records_collected: int = 0
    comparative_candidates_total: int = 0

    # Error type breakdown
    oracle_errors: int = 0
    genrm_errors: int = 0
    network_errors: int = 0
    other_errors: int = 0

    # Preference distribution (for bias detection)
    prefer_a: int = 0
    prefer_b: int = 0
    ties: int = 0
    confidence_sum: float = 0.0

    def record_example_success(self) -> None:
        self.total_examples_attempted += 1
        self.successful_examples += 1

    def record_example_failure(self) -> None:
        self.total_examples_attempted += 1
        self.failed_examples += 1

    def record_generation_success(self) -> None:
        self.total_candidates_attempted += 1
        self.successful_generations += 1

    def record_generation_failure(self) -> None:
        self.total_candidates_attempted += 1
        self.failed_generations += 1

    def record_pair_collected(self, preferred: str, confidence: float) -> None:
        self.pairs_collected += 1
        if preferred == "A":
            self.prefer_a += 1
        elif preferred == "B":
            self.prefer_b += 1
        else:
            self.ties += 1
        self.confidence_sum += confidence

    def record_pair_dropped(self, reason: str = "preference_error") -> None:
        if reason == "preference_error":
            self.pairs_dropped_preference_error += 1
        elif reason == "no_candidates":
            self.pairs_dropped_no_candidates += 1
        elif reason == "resummary_failed":
            self.pairs_dropped_resummary_failed += 1

    def record_comparative_record(self, num_candidates: int) -> None:
        self.comparative_records_collected += 1
        self.comparative_candidates_total += max(0, int(num_candidates))

    def record_error(self, error_type: str) -> None:
        if error_type == "oracle":
            self.oracle_errors += 1
        elif error_type == "genrm":
            self.genrm_errors += 1
        elif error_type == "network":
            self.network_errors += 1
        else:
            self.other_errors += 1

    @property
    def avg_confidence(self) -> float:
        return self.confidence_sum / max(1, self.pairs_collected)

    @property
    def position_balance(self) -> float:
        """Measure of A/B preference balance. 0 = perfectly balanced, 1 = all one side."""
        total = self.prefer_a + self.prefer_b
        if total == 0:
            return 0.0
        return abs(self.prefer_a - self.prefer_b) / total

    @property
    def success_rate(self) -> float:
        """Fraction of attempted examples that succeeded."""
        if self.total_examples_attempted == 0:
            return 0.0
        return self.successful_examples / self.total_examples_attempted

    @property
    def generation_success_rate(self) -> float:
        """Fraction of candidate generations that succeeded."""
        if self.total_candidates_attempted == 0:
            return 0.0
        return self.successful_generations / self.total_candidates_attempted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "examples": {
                "attempted": self.total_examples_attempted,
                "successful": self.successful_examples,
                "failed": self.failed_examples,
                "success_rate": self.success_rate,
            },
            "generations": {
                "attempted": self.total_candidates_attempted,
                "successful": self.successful_generations,
                "failed": self.failed_generations,
                "success_rate": self.generation_success_rate,
            },
            "pairs": {
                "collected": self.pairs_collected,
                "dropped_preference_error": self.pairs_dropped_preference_error,
                "dropped_no_candidates": self.pairs_dropped_no_candidates,
                "dropped_resummary_failed": self.pairs_dropped_resummary_failed,
            },
            "comparative": {
                "records_collected": self.comparative_records_collected,
                "candidates_total": self.comparative_candidates_total,
                "mean_candidates_per_record": (
                    self.comparative_candidates_total / self.comparative_records_collected
                    if self.comparative_records_collected
                    else 0.0
                ),
            },
            "errors": {
                "oracle": self.oracle_errors,
                "genrm": self.genrm_errors,
                "network": self.network_errors,
                "other": self.other_errors,
                "total": self.oracle_errors + self.genrm_errors + self.network_errors + self.other_errors,
            },
            "preferences": {
                "prefer_a": self.prefer_a,
                "prefer_b": self.prefer_b,
                "ties": self.ties,
                "avg_confidence": self.avg_confidence,
                "position_balance": self.position_balance,
            },
        }


@dataclass
class PreferenceResult:
    """Result from preference derivation."""

    preferred: str  # "A", "B", or "tie"
    confidence: float
    reasoning: str
    score_estimate_a: Optional[float] = None
    score_estimate_b: Optional[float] = None
    comparison_signal_value: Optional[float] = None
    comparison_signal_name: Optional[str] = None
    comparison_signal_min: Optional[float] = None
    comparison_signal_max: Optional[float] = None
    response_signal_name: Optional[str] = None
    response_signal_min: Optional[float] = None
    response_signal_max: Optional[float] = None
    oracle_error_a: Optional[float] = None
    oracle_error_b: Optional[float] = None
    extra_metadata: Optional[Dict[str, Any]] = None


@dataclass
class CandidateInfo(Generic[CandidateMeta]):
    """Information about a generated candidate summary."""

    summary: str
    metadata: CandidateMeta
    index: int


class BasePreferenceCollector(ABC, Generic[CandidateMeta]):
    """
    Abstract base class for preference collectors.

    Provides common infrastructure for:
    - Generating candidate summaries with varying parameters
    - Iterating over candidate pairs
    - Collecting preferences for OPS laws (sufficiency, idempotence, merge)
    - Building PreferencePair objects

    Subclasses must implement:
    - _derive_preference(): Compare two candidates and return preference
    - _create_candidate_metadata(): Create metadata for each candidate

    Subclasses may optionally override:
    - generate_candidates(): Custom candidate generation
    - _get_generation_config_dict(): Custom metadata serialization
    """

    def __init__(
        self,
        summarizer,
        k_candidates: int = 4,
        generation_configs: Optional[List[GenerationConfig]] = None,
    ):
        """
        Initialize the base collector.

        Args:
            summarizer: DSPy module or callable that produces summaries
            k_candidates: Number of candidate summaries per input
            generation_configs: List of generation configurations
        """
        self.summarizer = summarizer
        self.k_candidates = k_candidates
        if generation_configs is None:
            prompt_variants = ["concise", "default", "detailed", "creative"]
            self.generation_configs = [
                GenerationConfig(temperature=temp, prompt_variant=variant)
                for temp, variant in zip(DIVERSE_TEMPERATURES, prompt_variants)
            ]
        else:
            self.generation_configs = generation_configs

        self._pairs: List[PreferencePair] = []
        self._comparative_records: List[ComparativeJudgmentRecord] = []
        self._pair_counter = 0
        self.stats = CollectionStatistics()

    @abstractmethod
    def _derive_preference(
        self,
        candidate_a: CandidateInfo[CandidateMeta],
        candidate_b: CandidateInfo[CandidateMeta],
        context: Dict[str, Any],
    ) -> PreferenceResult:
        """
        Derive preference between two candidates.

        Args:
            candidate_a: First candidate with summary and metadata
            candidate_b: Second candidate with summary and metadata
            context: Dictionary with keys:
                - original_text: Source text
                - rubric: Preservation criteria
                - reference_score: Optional ground truth
                - law_type: OPS law type
                - extra_context: Optional additional context (e.g., for idempotence)

        Returns:
            PreferenceResult with preference, confidence, and reasoning
        """
        pass

    @abstractmethod
    def _create_candidate_metadata(
        self,
        gen_config: GenerationConfig,
        index: int,
    ) -> CandidateMeta:
        """
        Create metadata object for a candidate.

        Args:
            gen_config: Generation configuration used
            index: Index in the candidates list

        Returns:
            Metadata object (type depends on subclass)
        """
        pass

    def _get_generation_config_dict(
        self,
        metadata: CandidateMeta,
    ) -> Dict[str, Any]:
        """
        Convert candidate metadata to dictionary for PreferencePair.

        Default implementation handles GenerationConfig and dict types.
        Override for custom metadata types.

        Args:
            metadata: Candidate metadata

        Returns:
            Dictionary representation
        """
        if hasattr(metadata, "to_dict"):
            return metadata.to_dict()
        elif isinstance(metadata, dict):
            return metadata
        else:
            return {"value": str(metadata)}

    def generate_candidates(
        self,
        content: str,
        rubric: str,
    ) -> List[CandidateInfo[CandidateMeta]]:
        """
        Generate k candidate summaries for the input.

        Args:
            content: Input content to summarize
            rubric: Information preservation rubric

        Returns:
            List of CandidateInfo with summaries and metadata
        """
        candidates: List[CandidateInfo[CandidateMeta]] = []

        lm = getattr(dspy, "settings", None)
        lm = getattr(lm, "lm", None)

        for idx, gen_config in enumerate(self.generation_configs[: self.k_candidates]):
            with _lm_params(lm, gen_config.temperature, gen_config.top_p, gen_config.max_tokens):
                try:
                    result = self.summarizer(content=content, rubric=rubric)
                    summary = getattr(result, "summary", str(result))
                    metadata = self._create_candidate_metadata(gen_config, idx)
                    candidates.append(CandidateInfo(summary=summary, metadata=metadata, index=idx))
                    self.stats.record_generation_success()
                except Exception as e:
                    logger.warning(f"Failed to generate candidate {idx}: {e}")
                    self.stats.record_generation_failure()
                    self.stats.record_error("other")

        return candidates

    def _iterate_pairs(
        self,
        candidates: List[CandidateInfo[CandidateMeta]],
    ) -> List[Tuple[CandidateInfo[CandidateMeta], CandidateInfo[CandidateMeta], bool]]:
        """
        Generate all pairs of candidates with random swapping.

        Args:
            candidates: List of candidates to pair

        Returns:
            List of (candidate_a, candidate_b, swapped) tuples
        """
        pairs = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]
                swapped = random.random() < 0.5
                if swapped:
                    a, b = b, a
                pairs.append((a, b, swapped))
        return pairs

    def _create_pair(
        self,
        pair_id_prefix: str,
        example_id: str,
        candidate_a: CandidateInfo[CandidateMeta],
        candidate_b: CandidateInfo[CandidateMeta],
        pref_result: PreferenceResult,
        original_text: str,
        rubric: str,
        reference_score: float,
        law_type: str,
        judge_model: str = "",
    ) -> PreferencePair:
        """
        Create a PreferencePair from candidates and preference result.

        Args:
            pair_id_prefix: Prefix for pair ID (e.g., "genrm", "oracle")
            example_id: Source example identifier
            candidate_a: First candidate
            candidate_b: Second candidate
            pref_result: Result from preference derivation
            original_text: Original source text
            rubric: Preservation criteria
            reference_score: Ground truth score
            law_type: OPS law type
            judge_model: Name of judge model

        Returns:
            PreferencePair object
        """
        self._pair_counter += 1
        idx_a, idx_b = candidate_a.index, candidate_b.index

        supervision = preference_supervision_metadata(
            application_name="preference_collection",
            law_type=law_type,
            comparison_signal_name=pref_result.comparison_signal_name,
            comparison_signal_min=pref_result.comparison_signal_min,
            comparison_signal_max=pref_result.comparison_signal_max,
            response_signal_name=pref_result.response_signal_name,
            response_signal_min=pref_result.response_signal_min,
            response_signal_max=pref_result.response_signal_max,
        )

        return PreferencePair(
            pair_id=f"{pair_id_prefix}_{self._pair_counter:06d}",
            source_example_id=example_id,
            original_text=original_text,
            rubric=rubric,
            reference_score=reference_score,
            law_type=law_type,
            preference_supervision=supervision,
            summary_a=candidate_a.summary,
            summary_b=candidate_b.summary,
            preferred=pref_result.preferred,
            reasoning=f"{pref_result.reasoning} (candidates {idx_a},{idx_b})",
            confidence=pref_result.confidence,
            score_estimate_a=pref_result.score_estimate_a,
            score_estimate_b=pref_result.score_estimate_b,
            comparison_signal_value=pref_result.comparison_signal_value,
            oracle_error_a=pref_result.oracle_error_a,
            oracle_error_b=pref_result.oracle_error_b,
            judge_model=judge_model,
            generation_config_a=self._get_generation_config_dict(candidate_a.metadata),
            generation_config_b=self._get_generation_config_dict(candidate_b.metadata),
        )

    def collect_pairs_for_example(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        reference_score: float = 0.0,
        law_type: str = "sufficiency",
        **kwargs,
    ) -> List[PreferencePair]:
        """
        Generate candidates and collect all pairwise preferences.

        Args:
            example_id: Unique identifier
            original_text: Original text to summarize
            rubric: What information to preserve
            reference_score: Optional ground truth score
            law_type: OPS law type (sufficiency, idempotence, merge)
            **kwargs: Additional arguments passed to law-specific methods

        Returns:
            List of preference pairs
        """
        if law_type == "idempotence":
            return self._collect_idempotence_pairs(
                example_id, original_text, rubric, reference_score, **kwargs
            )
        if law_type == "merge":
            return self._collect_merge_pairs(
                example_id, original_text, rubric, reference_score, **kwargs
            )
        return self._collect_sufficiency_pairs(
            example_id, original_text, rubric, reference_score, **kwargs
        )

    def _collect_sufficiency_pairs(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        reference_score: float,
        **kwargs,
    ) -> List[PreferencePair]:
        """Collect preference pairs for sufficiency law."""
        candidates = self.generate_candidates(original_text, rubric)

        if len(candidates) < 2:
            logger.warning(f"Only {len(candidates)} candidates for {example_id}")
            self.stats.record_pair_dropped("no_candidates")
            self.stats.record_example_failure()
            return []

        context = {
            "original_text": original_text,
            "rubric": rubric,
            "reference_score": reference_score,
            "law_type": "sufficiency",
        }
        context.update(kwargs)

        pairs: List[PreferencePair] = []
        for candidate_a, candidate_b, _swapped in self._iterate_pairs(candidates):
            try:
                pref_result = self._derive_preference(candidate_a, candidate_b, context)

                pair = self._create_pair(
                    pair_id_prefix=self._get_pair_id_prefix(),
                    example_id=example_id,
                    candidate_a=candidate_a,
                    candidate_b=candidate_b,
                    pref_result=pref_result,
                    original_text=original_text,
                    rubric=rubric,
                    reference_score=reference_score,
                    law_type="sufficiency",
                    judge_model=self._get_judge_model_name(),
                )
                pairs.append(pair)
                self._pairs.append(pair)
                self.stats.record_pair_collected(pref_result.preferred, pref_result.confidence)

            except Exception as e:
                logger.error(f"Failed to derive preference: {e}")
                self.stats.record_pair_dropped("preference_error")
                self.stats.record_error("other")

        if pairs:
            self.stats.record_example_success()
        else:
            self.stats.record_example_failure()

        return pairs

    def _collect_idempotence_pairs(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        reference_score: float,
        **kwargs,
    ) -> List[PreferencePair]:
        """
        Collect preference pairs for idempotence law.

        Idempotence: summarize(summarize(x)) should approximately equal summarize(x)
        """
        candidates = self.generate_candidates(original_text, rubric)

        if len(candidates) < 2:
            logger.warning(f"Only {len(candidates)} candidates for {example_id}")
            self.stats.record_pair_dropped("no_candidates")
            self.stats.record_example_failure()
            return []

        # Generate re-summaries for each candidate
        resummaries: Dict[int, str] = {}
        for candidate in candidates:
            try:
                result = self.summarizer(content=candidate.summary, rubric=rubric)
                resummary = getattr(result, "summary", str(result))
                resummaries[candidate.index] = resummary
                self.stats.record_generation_success()
            except Exception as e:
                logger.warning(f"Failed to generate resummary: {e}")
                resummaries[candidate.index] = ""
                self.stats.record_generation_failure()
                self.stats.record_error("other")

        pairs: List[PreferencePair] = []
        for candidate_a, candidate_b, _swapped in self._iterate_pairs(candidates):
            resummary_a = resummaries.get(candidate_a.index, "")
            resummary_b = resummaries.get(candidate_b.index, "")

            if not resummary_a or not resummary_b:
                self.stats.record_pair_dropped("resummary_failed")
                continue

            extra_context = (
                "Idempotence check (re-summaries):\n"
                f"Candidate A resummary:\n{resummary_a}\n\n"
                f"Candidate B resummary:\n{resummary_b}"
            )

            context = {
                "original_text": original_text,
                "rubric": rubric,
                "reference_score": reference_score,
                "law_type": "idempotence",
                "extra_context": extra_context,
                "resummary_a": resummary_a,
                "resummary_b": resummary_b,
            }
            context.update(kwargs)

            try:
                pref_result = self._derive_preference(candidate_a, candidate_b, context)

                pair = self._create_pair(
                    pair_id_prefix=f"{self._get_pair_id_prefix()}_idem",
                    example_id=example_id,
                    candidate_a=candidate_a,
                    candidate_b=candidate_b,
                    pref_result=pref_result,
                    original_text=original_text,
                    rubric=rubric,
                    reference_score=reference_score,
                    law_type="idempotence",
                    judge_model=self._get_judge_model_name(),
                )
                pairs.append(pair)
                self._pairs.append(pair)
                self.stats.record_pair_collected(pref_result.preferred, pref_result.confidence)

            except Exception as e:
                logger.error(f"Failed to derive preference: {e}")
                self.stats.record_pair_dropped("preference_error")
                self.stats.record_error("other")

        if pairs:
            self.stats.record_example_success()
        else:
            self.stats.record_example_failure()

        return pairs

    def _collect_merge_pairs(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        reference_score: float,
        **kwargs,
    ) -> List[PreferencePair]:
        """
        Collect preference pairs for merge law.

        Merge: summarize(summary_a + summary_b) should preserve information from both.
        """
        words = original_text.split()
        if not words:
            logger.warning(f"No text for merge pairing on {example_id}")
            self.stats.record_example_failure()
            return []

        # Split text in half
        mid = len(words) // 2
        text_a = " ".join(words[:mid])
        text_b = " ".join(words[mid:])

        # Generate candidates for each half
        k_per_half = max(2, self.k_candidates // 2)
        candidates_a = self.generate_candidates(text_a, rubric)[:k_per_half]
        candidates_b = self.generate_candidates(text_b, rubric)[:k_per_half]

        if not candidates_a or not candidates_b:
            logger.warning(f"Insufficient candidates for merge on {example_id}")
            self.stats.record_pair_dropped("no_candidates")
            self.stats.record_example_failure()
            return []

        # Create merged summaries
        @dataclass
        class MergedCandidate:
            """Represents a merged candidate from two halves."""

            left: CandidateInfo[CandidateMeta]
            right: CandidateInfo[CandidateMeta]
            merged_summary: str
            combined_index: int

        merged_candidates: List[MergedCandidate] = []
        combined_idx = 0
        for left in candidates_a:
            for right in candidates_b:
                merged_text = format_merge_input(left.summary, right.summary)
                try:
                    result = self.summarizer(content=merged_text, rubric=rubric)
                    merged_summary = getattr(result, "summary", str(result))
                    merged_candidates.append(
                        MergedCandidate(
                            left=left,
                            right=right,
                            merged_summary=merged_summary,
                            combined_index=combined_idx,
                        )
                    )
                    combined_idx += 1
                    self.stats.record_generation_success()
                except Exception as e:
                    logger.warning(f"Failed to generate merged summary: {e}")
                    self.stats.record_generation_failure()
                    self.stats.record_error("other")

        if len(merged_candidates) < 2:
            logger.warning(f"Only {len(merged_candidates)} merged candidates for {example_id}")
            self.stats.record_pair_dropped("no_candidates")
            self.stats.record_example_failure()
            return []

        pairs: List[PreferencePair] = []
        for i in range(len(merged_candidates)):
            for j in range(i + 1, len(merged_candidates)):
                mc_a, mc_b = merged_candidates[i], merged_candidates[j]
                idx_a, idx_b = mc_a.combined_index, mc_b.combined_index

                swapped = random.random() < 0.5
                if swapped:
                    mc_a, mc_b = mc_b, mc_a
                    idx_a, idx_b = idx_b, idx_a

                extra_context = (
                    "Merge check (child summaries):\n"
                    f"Candidate A left summary:\n{mc_a.left.summary}\n"
                    f"Candidate A right summary:\n{mc_a.right.summary}\n\n"
                    f"Candidate B left summary:\n{mc_b.left.summary}\n"
                    f"Candidate B right summary:\n{mc_b.right.summary}"
                )

                # Create pseudo CandidateInfo for merged summaries
                merged_info_a = CandidateInfo(
                    summary=mc_a.merged_summary,
                    metadata=mc_a.left.metadata,  # Use left's metadata as base
                    index=idx_a,
                )
                merged_info_b = CandidateInfo(
                    summary=mc_b.merged_summary,
                    metadata=mc_b.left.metadata,
                    index=idx_b,
                )

                context = {
                    "original_text": original_text,
                    "rubric": rubric,
                    "reference_score": reference_score,
                    "law_type": "merge",
                    "extra_context": extra_context,
                    "merged_candidate_a": mc_a,
                    "merged_candidate_b": mc_b,
                }
                context.update(kwargs)

                try:
                    pref_result = self._derive_preference(merged_info_a, merged_info_b, context)

                    # Create custom generation configs for merged candidates
                    self._pair_counter += 1
                    supervision = preference_supervision_metadata(
                        application_name="preference_collection",
                        law_type="merge",
                        comparison_signal_name=pref_result.comparison_signal_name,
                        comparison_signal_min=pref_result.comparison_signal_min,
                        comparison_signal_max=pref_result.comparison_signal_max,
                        response_signal_name=pref_result.response_signal_name,
                        response_signal_min=pref_result.response_signal_min,
                        response_signal_max=pref_result.response_signal_max,
                    )

                    pair = PreferencePair(
                        pair_id=f"{self._get_pair_id_prefix()}_merge_{self._pair_counter:06d}",
                        source_example_id=example_id,
                        original_text=original_text,
                        rubric=rubric,
                        reference_score=reference_score,
                        law_type="merge",
                        preference_supervision=supervision,
                        summary_a=mc_a.merged_summary,
                        summary_b=mc_b.merged_summary,
                        preferred=pref_result.preferred,
                        reasoning=f"{pref_result.reasoning} (candidates {idx_a},{idx_b})",
                        confidence=pref_result.confidence,
                        score_estimate_a=pref_result.score_estimate_a,
                        score_estimate_b=pref_result.score_estimate_b,
                        comparison_signal_value=pref_result.comparison_signal_value,
                        oracle_error_a=pref_result.oracle_error_a,
                        oracle_error_b=pref_result.oracle_error_b,
                        judge_model=self._get_judge_model_name(),
                        generation_config_a={
                            "left": self._get_generation_config_dict(mc_a.left.metadata),
                            "right": self._get_generation_config_dict(mc_a.right.metadata),
                        },
                        generation_config_b={
                            "left": self._get_generation_config_dict(mc_b.left.metadata),
                            "right": self._get_generation_config_dict(mc_b.right.metadata),
                        },
                    )
                    pairs.append(pair)
                    self._pairs.append(pair)
                    self.stats.record_pair_collected(pref_result.preferred, pref_result.confidence)

                except Exception as e:
                    logger.error(f"Failed to derive preference: {e}")
                    self.stats.record_pair_dropped("preference_error")
                    self.stats.record_error("other")

        if pairs:
            self.stats.record_example_success()
        else:
            self.stats.record_example_failure()

        return pairs

    def _get_pair_id_prefix(self) -> str:
        """
        Get prefix for pair IDs.

        Override in subclasses to customize pair ID format.
        """
        return "pref"

    def _get_judge_model_name(self) -> str:
        """
        Get name of the judge model.

        Override in subclasses to return appropriate model name.
        """
        return ""

    def get_all_pairs(self) -> List[PreferencePair]:
        """Return all collected pairs."""
        return self._pairs

    def get_dataset(self) -> PreferenceDataset:
        """Return pairs as a compatibility PreferenceDataset."""
        return PreferenceDataset(self._pairs)

    def get_supervision_dataset(self):
        """Return the primary supervision dataset for collected judgments."""
        from treepo._research.training.supervision import SupervisionDataset

        dataset = SupervisionDataset(
            comparative_judgments=list(self._comparative_records),
        )
        if self._pairs:
            dataset.add_comparative_judgments(
                [pair.to_comparative_judgment() for pair in self._pairs]
            )
        return dataset

    def get_all_comparative_records(self) -> List[ComparativeJudgmentRecord]:
        """Return all collected comparative records."""
        return self._comparative_records

    def get_comparative_dataset(self) -> ComparativeDataset:
        """Return comparative records as a ComparativeDataset."""
        return ComparativeDataset(self._comparative_records)

    def get_statistics(self) -> Dict[str, Any]:
        """Return comprehensive collection statistics.

        Returns the full CollectionStatistics as a dictionary for visibility
        into data quality, error rates, and preference distribution.
        """
        return self.stats.to_dict()

    def get_collection_stats(self) -> CollectionStatistics:
        """Return the raw CollectionStatistics object."""
        return self.stats
