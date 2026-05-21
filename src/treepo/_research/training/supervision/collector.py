"""
Preference Learning Infrastructure for Oracle-Preserving Summarization.

This module provides infrastructure for training oracle/judge models using
pairwise preference data. The workflow is:
1. Generate multiple candidate summaries from a smaller model (e.g., Nemotron-nano)
2. Use a large oracle model (e.g., Nemotron-253B) to compare and rank outputs
3. Build preference pairs for training
4. Train a smaller judge model to mimic the large oracle's preferences
5. Use the judge to guide distillation training of the summarizer

Key components:
- PreferencePair: A single pairwise preference judgment
- PairwiseJudge: DSPy module for comparing summaries
- PreferenceCollector: Generates diverse outputs and collects preferences
- PreferenceDataset: Manages preference pairs for training
"""

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Dict, Any, Callable, Literal, Set

from treepo._research.core.conditional_memory import canonical_hash

if TYPE_CHECKING:
    from treepo._research.core.conditional_memory import ConditionalMemory

import dspy

# Import generic signature from core; task-specific versions live under src/tasks
from treepo._research.config.constants import DIVERSE_TEMPERATURES
from treepo._research.core.signatures import PairwiseComparison
from treepo._research.core.output_parser import NormalizedOutputAccessor
from treepo._research.core.supervision_metadata import (
    judgment_supervision_metadata as preference_supervision_metadata,
)

# Import shared data types (separated to avoid circular imports)
from .comparative_types import (
    ComparativeCandidate,
    ComparativeJudgmentRecord,
    GenerationConfig,
    PreferencePair,
    PreferenceDataset,
)
from .judge_capabilities import (
    invoke_comparative_judgment_sync,
    invoke_pairwise_judgment_sync,
    judge_backend_name,
    supports_direct_comparative_judging,
)

# Import base class for OPS law support (idempotence, merge)
from .base import (
    BasePreferenceCollector,
    CandidateInfo,
    PreferenceResult,
    CollectionStatistics,
)

logger = logging.getLogger(__name__)


class PairwiseJudge(dspy.Module):
    """
    DSPy module for comparing two summaries using a large oracle model.

    Uses chain-of-thought reasoning to determine which summary
    better preserves the target information.
    """

    def __init__(self, use_cot: bool = True):
        """
        Initialize the judge module.

        Args:
            use_cot: Whether to use chain-of-thought reasoning
        """
        super().__init__()
        if use_cot:
            self.compare = dspy.ChainOfThought(PairwiseComparison)
        else:
            self.compare = dspy.Predict(PairwiseComparison)

    def forward(
        self,
        original_text: str,
        summary_a: str,
        summary_b: str,
        rubric: str,
        reference_score: float,
    ) -> Dict[str, Any]:
        """
        Compare two summaries and determine which is better.

        Args:
            original_text: Original source text
            summary_a: First candidate summary
            summary_b: Second candidate summary
            rubric: Information preservation criteria
            reference_score: Ground truth score for original text

        Returns:
            Dictionary with preference judgment and reasoning
        """
        result = self.compare(
            rubric=rubric,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            reference_score=reference_score,
        )

        # Use normalized accessor to handle key casing variations
        accessor = NormalizedOutputAccessor(result)

        # Normalize preferred to uppercase
        preferred = str(accessor.get('preferred', 'tie')).upper().strip()
        if preferred not in ["A", "B", "TIE"]:
            # Try to extract from reasoning
            if "A" in preferred and "B" not in preferred:
                preferred = "A"
            elif "B" in preferred and "A" not in preferred:
                preferred = "B"
            else:
                preferred = "tie"
        elif preferred == "TIE":
            preferred = "tie"

        # Parse confidence (using normalized accessor)
        try:
            confidence = float(accessor.get('confidence', 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5

        # Parse score estimates (using normalized accessor for case-insensitive field access)
        score_a = None
        raw_score_a = accessor.get('score_estimate_a')
        if raw_score_a is not None:
            try:
                score_a = float(raw_score_a)
            except (ValueError, TypeError):
                score_a = None

        score_b = None
        raw_score_b = accessor.get('score_estimate_b')
        if raw_score_b is not None:
            try:
                score_b = float(raw_score_b)
            except (ValueError, TypeError):
                score_b = None

        return {
            "preferred": preferred,
            "reasoning": str(accessor.get('reasoning', '')),
            "confidence": confidence,
            "score_estimate_a": score_a,
            "score_estimate_b": score_b,
        }


class PreferenceCollector(BasePreferenceCollector[GenerationConfig]):
    """
    Unified preference collector with strategy-based preference derivation.

    Inherits from BasePreferenceCollector to support all three OPS laws:
    - Sufficiency: original text → summary preserves information
    - Idempotence: summarize(summary) ≈ summary
    - Merge: summarize(A + B) preserves information from both

    Supports three strategies for deriving preferences:
    - "judge": Uses PairwiseJudge for LLM-based comparison (default)
    - "genrm": Uses GenRMJudge for NVIDIA's Nemotron GenRM comparison
    - "oracle": Uses oracle predictions to compute error-based preferences

    Workflow:
    1. For each input example, generate k candidate summaries
    2. Create all pairwise comparisons (k choose 2)
    3. Use the configured strategy to compare each pair
    4. Store the preference pairs

    Example:
        # Judge-based (default, backward compatible)
        collector = PreferenceCollector(summarizer, judge=my_judge)

        # GenRM-based
        from treepo._research.training.preference.genrm import GenRMJudge
        genrm = GenRMJudge(base_url="http://localhost:8000")
        collector = PreferenceCollector(summarizer, strategy="genrm", genrm_judge=genrm)

        # Oracle-based
        collector = PreferenceCollector(
            summarizer,
            strategy="oracle",
            oracle_predict=lambda text: my_oracle(text),
        )

        # Collect pairs for different OPS laws
        sufficiency_pairs = collector.collect_pairs_for_example(
            example_id="doc1", original_text=text, rubric=rubric, law_type="sufficiency"
        )
        idempotence_pairs = collector.collect_pairs_for_example(
            example_id="doc1", original_text=text, rubric=rubric, law_type="idempotence"
        )
        merge_pairs = collector.collect_pairs_for_example(
            example_id="doc1", original_text=text, rubric=rubric, law_type="merge"
        )
    """

    def __init__(
        self,
        summarizer: dspy.Module,
        judge: Optional[PairwiseJudge] = None,
        k: int = 4,
        generation_configs: Optional[List[GenerationConfig]] = None,
        # Strategy support
        strategy: Literal["judge", "genrm", "oracle"] = "judge",
        genrm_judge: Optional[Any] = None,  # GenRMJudge
        oracle_predict: Optional[Any] = None,  # Callable[[str], float]
        tie_margin: float = 5.0,  # For oracle strategy
        comparison_mode: Literal["pairwise", "listwise", "auto"] = "pairwise",
        memory: Optional["ConditionalMemory"] = None,
    ):
        """
        Initialize the collector.

        Args:
            summarizer: DSPy module for generating summaries
            judge: PairwiseJudge for comparing summaries (required for strategy="judge")
            k: Number of candidate summaries to generate per input
            generation_configs: Configurations for generating diverse outputs
            strategy: Preference derivation strategy ("judge", "genrm", "oracle")
            genrm_judge: GenRMJudge instance (required for strategy="genrm")
            oracle_predict: Function (text) -> float (required for strategy="oracle")
            tie_margin: Error margin for ties in oracle strategy
            memory: Optional ConditionalMemory for cross-run oracle score persistence
        """
        # Build generation configs first (needed for super().__init__)
        if generation_configs is None:
            prompt_variants = ["concise", "default", "detailed", "creative"]
            generation_configs = [
                GenerationConfig(temperature=temp, prompt_variant=variant)
                for temp, variant in zip(DIVERSE_TEMPERATURES, prompt_variants)
            ]

        # Initialize base class (provides OPS law support)
        super().__init__(
            summarizer=summarizer,
            k_candidates=k,
            generation_configs=generation_configs,
        )

        self.k = k
        self.strategy = strategy
        self.tie_margin = tie_margin
        self.comparison_mode = str(comparison_mode or "pairwise").strip().lower()

        # Strategy-specific components
        self.judge = judge
        self.genrm_judge = genrm_judge
        self.oracle_predict = oracle_predict

        # Validate strategy configuration
        if strategy == "judge" and judge is None:
            raise ValueError("judge is required when strategy='judge'")
        if strategy == "genrm" and genrm_judge is None:
            raise ValueError("genrm_judge is required when strategy='genrm'")
        if strategy == "oracle" and oracle_predict is None:
            raise ValueError("oracle_predict is required when strategy='oracle'")

        self._oracle_cache: Dict[str, float] = {}  # For oracle strategy
        self._memory = memory

    def _get_active_comparative_backend(self) -> Optional[Any]:
        """Return the active judge-like backend for direct comparative judgments."""
        if self.strategy == "judge":
            return self.judge
        if self.strategy == "genrm":
            return self.genrm_judge
        return None

    def _supports_listwise_judging(self) -> bool:
        backend = self._get_active_comparative_backend()
        return backend is not None and supports_direct_comparative_judging(backend)

    def _should_use_listwise(self) -> bool:
        if self.comparison_mode == "listwise":
            return True
        if self.comparison_mode == "auto":
            return self._supports_listwise_judging()
        return False

    def _build_comparative_record(
        self,
        *,
        example_id: str,
        original_text: str,
        rubric: str,
        reference_score: float,
        law_type: str,
        candidates: List[CandidateInfo[GenerationConfig]],
        candidate_views: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ComparativeJudgmentRecord:
        if len(candidates) != len(candidate_views):
            raise ValueError("candidates and candidate_views must have the same length")
        if len(candidates) < 2:
            raise ValueError("Need at least two candidates for comparative judging")
        if not self._supports_listwise_judging():
            raise ValueError("Current judge does not support listwise ranking")

        backend = self._get_active_comparative_backend()
        if backend is None:
            raise ValueError("Current strategy does not expose a direct comparative judge")

        result = invoke_comparative_judgment_sync(
            backend,
            context=rubric,
            original_text=original_text,
            candidate_summaries=candidate_views,
            law_type=law_type,
            reference_score=reference_score,
        )
        ordered_ids = [str(value).upper() for value in list(result.ordered_candidate_ids or [])]
        if not ordered_ids:
            ordered_ids = [f"C{idx}" for idx in range(1, len(candidates) + 1)]
        rank_by_id = {
            candidate_id: rank
            for rank, candidate_id in enumerate(ordered_ids, start=1)
        }
        raw_scores = dict(result.candidate_scores or {})
        score_by_id = {
            str(candidate_id).upper(): float(score)
            for candidate_id, score in raw_scores.items()
        }
        supervision = preference_supervision_metadata(
            application_name="preference_collection",
            law_type=law_type,
            comparison_signal_name=result.comparison_signal_name,
            comparison_signal_min=result.comparison_signal_min,
            comparison_signal_max=result.comparison_signal_max,
            response_signal_name=result.response_signal_name or "listwise_candidate_score",
            response_signal_min=result.response_signal_min,
            response_signal_max=result.response_signal_max,
            metadata={"collection_mode": "listwise"},
        ).with_updates(preference_family="groupwise")

        comparative_candidates: List[ComparativeCandidate] = []
        for idx, candidate in enumerate(candidates, start=1):
            candidate_id = f"C{idx}"
            comparative_candidates.append(
                ComparativeCandidate(
                    candidate_id=candidate_id,
                    response=str(candidate.summary or ""),
                    rank=rank_by_id.get(candidate_id, idx),
                    response_signal_value=score_by_id.get(candidate_id),
                    metadata={
                        "candidate_index": candidate.index,
                        "generation_config": self._get_generation_config_dict(candidate.metadata),
                        "candidate_view": candidate_views[idx - 1],
                    },
                )
            )

        self.stats.record_comparative_record(len(comparative_candidates))
        return ComparativeJudgmentRecord(
            record_id=f"{law_type}_cmp_{len(self._comparative_records) + 1:06d}",
            source_example_id=example_id,
            original_text=original_text,
            rubric=rubric,
            reference_score=reference_score,
            law_type=law_type,
            candidates=comparative_candidates,
            preference_supervision=supervision,
            judge_model=self._get_judge_model_name(),
            metadata={
                "reasoning": str(result.reasoning or ""),
                "confidence": float(result.confidence or 0.5),
                "judge_payload": dict(result.raw_payload or {}),
                **dict(metadata or {}),
            },
        )

    def collect_comparative_for_example(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        reference_score: float = 0.0,
        law_type: str = "sufficiency",
        **kwargs,
    ) -> ComparativeJudgmentRecord:
        if not self._supports_listwise_judging():
            fallback_pairs = self.collect_pairs_for_example(
                example_id=example_id,
                original_text=original_text,
                rubric=rubric,
                reference_score=reference_score,
                law_type=law_type,
                **kwargs,
            )
            comparative_dataset = PreferenceDataset(fallback_pairs).to_comparative_dataset(
                law_type=law_type
            )
            if not comparative_dataset.records:
                raise ValueError("No comparative record could be constructed")
            return comparative_dataset.records[0]

        if law_type == "idempotence":
            return self._collect_idempotence_comparative_record(
                example_id,
                original_text,
                rubric,
                reference_score,
                **kwargs,
            )
        if law_type == "merge":
            return self._collect_merge_comparative_record(
                example_id,
                original_text,
                rubric,
                reference_score,
                **kwargs,
            )
        return self._collect_sufficiency_comparative_record(
            example_id,
            original_text,
            rubric,
            reference_score,
            **kwargs,
        )

    def _collect_sufficiency_comparative_record(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        reference_score: float,
        **kwargs,
    ) -> ComparativeJudgmentRecord:
        candidates = self.generate_candidates(original_text, rubric)
        if len(candidates) < 2:
            logger.warning(f"Only {len(candidates)} candidates for {example_id}")
            self.stats.record_pair_dropped("no_candidates")
            self.stats.record_example_failure()
            raise ValueError("Insufficient candidates for comparative judgment")
        record = self._build_comparative_record(
            example_id=example_id,
            original_text=original_text,
            rubric=rubric,
            reference_score=reference_score,
            law_type="sufficiency",
            candidates=candidates,
            candidate_views=[candidate.summary for candidate in candidates],
            metadata=kwargs,
        )
        self._comparative_records.append(record)
        self.stats.record_example_success()
        return record

    def _collect_idempotence_comparative_record(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        reference_score: float,
        **kwargs,
    ) -> ComparativeJudgmentRecord:
        candidates = self.generate_candidates(original_text, rubric)
        if len(candidates) < 2:
            logger.warning(f"Only {len(candidates)} candidates for {example_id}")
            self.stats.record_pair_dropped("no_candidates")
            self.stats.record_example_failure()
            raise ValueError("Insufficient candidates for comparative judgment")

        resummaries: Dict[int, str] = {}
        for candidate in candidates:
            try:
                result = self.summarizer(content=candidate.summary, rubric=rubric)
                resummaries[candidate.index] = getattr(result, "summary", str(result))
                self.stats.record_generation_success()
            except Exception as e:
                logger.warning(f"Failed to generate resummary: {e}")
                self.stats.record_generation_failure()
                self.stats.record_error("other")
                resummaries[candidate.index] = ""

        filtered_candidates: List[CandidateInfo[GenerationConfig]] = []
        candidate_views: List[str] = []
        for candidate in candidates:
            resummary = resummaries.get(candidate.index, "")
            if not resummary:
                continue
            filtered_candidates.append(candidate)
            candidate_views.append(
                "Candidate Summary:\n"
                f"{candidate.summary}\n\n"
                "Re-summarized Candidate:\n"
                f"{resummary}"
            )
        if len(filtered_candidates) < 2:
            self.stats.record_pair_dropped("resummary_failed")
            self.stats.record_example_failure()
            raise ValueError("Insufficient successful resummaries for comparative judgment")

        record = self._build_comparative_record(
            example_id=example_id,
            original_text=original_text,
            rubric=rubric,
            reference_score=reference_score,
            law_type="idempotence",
            candidates=filtered_candidates,
            candidate_views=candidate_views,
            metadata=kwargs,
        )
        self._comparative_records.append(record)
        self.stats.record_example_success()
        return record

    def _collect_merge_comparative_record(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        reference_score: float,
        **kwargs,
    ) -> ComparativeJudgmentRecord:
        words = original_text.split()
        if not words:
            self.stats.record_example_failure()
            raise ValueError("No text for merge comparative judgment")

        mid = len(words) // 2
        text_a = " ".join(words[:mid])
        text_b = " ".join(words[mid:])
        k_per_half = max(2, self.k_candidates // 2)
        candidates_a = self.generate_candidates(text_a, rubric)[:k_per_half]
        candidates_b = self.generate_candidates(text_b, rubric)[:k_per_half]
        if not candidates_a or not candidates_b:
            self.stats.record_pair_dropped("no_candidates")
            self.stats.record_example_failure()
            raise ValueError("Insufficient candidates for merge comparative judgment")

        @dataclass
        class MergedCandidate:
            left: CandidateInfo[GenerationConfig]
            right: CandidateInfo[GenerationConfig]
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
            self.stats.record_pair_dropped("no_candidates")
            self.stats.record_example_failure()
            raise ValueError("Insufficient merged candidates for comparative judgment")

        comparative_candidates: List[CandidateInfo[GenerationConfig]] = []
        candidate_views: List[str] = []
        for merged in merged_candidates:
            comparative_candidates.append(
                CandidateInfo(
                    summary=merged.merged_summary,
                    metadata=merged.left.metadata,
                    index=merged.combined_index,
                )
            )
            candidate_views.append(
                "Merged Summary:\n"
                f"{merged.merged_summary}\n\n"
                "Left Child Summary:\n"
                f"{merged.left.summary}\n\n"
                "Right Child Summary:\n"
                f"{merged.right.summary}"
            )

        record = self._build_comparative_record(
            example_id=example_id,
            original_text=original_text,
            rubric=rubric,
            reference_score=reference_score,
            law_type="merge",
            candidates=comparative_candidates,
            candidate_views=candidate_views,
            metadata=kwargs,
        )
        self._comparative_records.append(record)
        self.stats.record_example_success()
        return record

    def _create_candidate_metadata(
        self,
        gen_config: GenerationConfig,
        index: int,
    ) -> GenerationConfig:
        """Create metadata for a candidate (returns the GenerationConfig itself)."""
        return gen_config

    def _get_generation_config_dict(
        self,
        metadata: GenerationConfig,
    ) -> Dict[str, Any]:
        """Convert GenerationConfig metadata to dictionary."""
        return metadata.to_dict()

    def _get_pair_id_prefix(self) -> str:
        """Return strategy-specific pair ID prefix."""
        return {"judge": "pair", "genrm": "genrm", "oracle": "oracle"}.get(self.strategy, "pref")

    def _get_judge_model_name(self) -> str:
        """Return name of the judge model."""
        if self.strategy == "judge" and self.judge is not None:
            return judge_backend_name(self.judge)
        if self.strategy == "genrm" and self.genrm_judge is not None:
            return judge_backend_name(self.genrm_judge)
        return ""

    def _get_oracle_score(self, text: str) -> float:
        """Get oracle score with caching (for oracle strategy).

        Checks ConditionalMemory first (cross-run persistent), then local dict.
        """
        text_hash = canonical_hash(text)

        # Check ConditionalMemory first
        if self._memory is not None:
            namespace = f"oracle_pref:{self._memory.namespace_version}"
            cached = self._memory.get_json(namespace, text_hash)
            if cached is not None:
                try:
                    return float(cached)
                except (TypeError, ValueError):
                    pass

        if text_hash in self._oracle_cache:
            return self._oracle_cache[text_hash]

        score = self.oracle_predict(text)
        self._oracle_cache[text_hash] = score

        # Persist to ConditionalMemory
        if self._memory is not None:
            namespace = f"oracle_pref:{self._memory.namespace_version}"
            self._memory.set_json(namespace, text_hash, float(score))

        return score

    def _build_enrichment_context(
        self,
        original_text: str,
        summary_a: str,
        summary_b: str,
    ) -> Optional[str]:
        """Build entity preservation context from ConditionalMemory enrichment.

        Looks up enrichment metadata for the original text and computes
        entity preservation rates for each candidate summary. Returns a
        compact string for injection into GenRM extra_context, or None
        if enrichment data is unavailable.
        """
        if self._memory is None:
            return None

        # Look up enrichment metadata for the original text
        key = canonical_hash(original_text)
        enrichment_data = self._memory.get_json(
            f"enrichment:{self._memory.namespace_version}", key
        )
        if enrichment_data is None:
            # Try the generic enrichment namespace
            enrichment_data = self._memory.get_json("enrichment", key)
        if not isinstance(enrichment_data, dict):
            return None

        key_entities: List[str] = enrichment_data.get("key_entities", [])
        key_numbers: List[str] = enrichment_data.get("key_numbers", [])
        if not key_entities and not key_numbers:
            return None

        # Check entity preservation in each summary
        def _preservation_rate(summary: str, entities: List[str]) -> float:
            if not entities:
                return 1.0
            lower = summary.lower()
            found = sum(1 for e in entities if e.lower() in lower)
            return found / len(entities)

        ent_rate_a = _preservation_rate(summary_a, key_entities)
        ent_rate_b = _preservation_rate(summary_b, key_entities)
        num_rate_a = _preservation_rate(summary_a, key_numbers)
        num_rate_b = _preservation_rate(summary_b, key_numbers)

        parts = []
        parts.append(f"Key entities ({len(key_entities)}): {', '.join(key_entities[:8])}")
        parts.append(f"Entity preservation: A={ent_rate_a:.0%}, B={ent_rate_b:.0%}")
        if key_numbers:
            parts.append(f"Number preservation: A={num_rate_a:.0%}, B={num_rate_b:.0%}")
        return "[ENRICHMENT CONTEXT] " + " | ".join(parts)

    def _derive_preference(
        self,
        candidate_a: CandidateInfo[GenerationConfig],
        candidate_b: CandidateInfo[GenerationConfig],
        context: Dict[str, Any],
    ) -> PreferenceResult:
        """
        Derive preference using the configured strategy.

        Implements the abstract method from BasePreferenceCollector.

        Args:
            candidate_a: First candidate with summary and metadata
            candidate_b: Second candidate with summary and metadata
            context: Dictionary with original_text, rubric, reference_score, law_type

        Returns:
            PreferenceResult with preference, confidence, and reasoning
        """
        summary_a = candidate_a.summary
        summary_b = candidate_b.summary
        original_text = context.get("original_text", "")
        rubric = context.get("rubric", "")
        reference_score = context.get("reference_score", 0.0)
        law_type = context.get("law_type", "sufficiency")

        if self.strategy == "judge":
            result = invoke_pairwise_judgment_sync(
                self.judge,
                context=rubric,
                original_text=original_text,
                summary_a=summary_a,
                summary_b=summary_b,
                law_type=law_type,
                reference_score=reference_score,
            )
            return PreferenceResult(
                preferred=result.preferred,
                reasoning=result.reasoning,
                confidence=result.confidence,
                score_estimate_a=result.score_estimate_a,
                score_estimate_b=result.score_estimate_b,
                comparison_signal_value=result.comparison_signal_value,
                comparison_signal_name=result.comparison_signal_name,
                comparison_signal_min=result.comparison_signal_min,
                comparison_signal_max=result.comparison_signal_max,
                response_signal_name=(
                    result.response_signal_name
                    or (
                        "judge_score_estimate"
                        if (
                            result.score_estimate_a is not None
                            or result.score_estimate_b is not None
                        )
                        else None
                    )
                ),
                response_signal_min=result.response_signal_min,
                response_signal_max=result.response_signal_max,
            )

        elif self.strategy == "genrm":
            # Build memory-augmented enrichment context (Phase 6.2)
            enrichment_ctx = self._build_enrichment_context(
                original_text, summary_a, summary_b
            )
            result = invoke_pairwise_judgment_sync(
                self.genrm_judge,
                context=rubric,
                original_text=original_text,
                summary_a=summary_a,
                summary_b=summary_b,
                law_type=law_type,
                reference_score=reference_score,
                extra_context=enrichment_ctx,
            )
            if result.error_message is not None:
                return PreferenceResult(
                    preferred="tie",
                    reasoning=f"GenRM error: {result.error_message}",
                    confidence=0.0,
                )
            return PreferenceResult(
                preferred=result.preferred,
                reasoning=result.reasoning,
                confidence=result.confidence,
                score_estimate_a=result.score_estimate_a,
                score_estimate_b=result.score_estimate_b,
                comparison_signal_value=result.comparison_signal_value,
                comparison_signal_name=result.comparison_signal_name,
                comparison_signal_min=result.comparison_signal_min,
                comparison_signal_max=result.comparison_signal_max,
                response_signal_name=result.response_signal_name,
                response_signal_min=result.response_signal_min,
                response_signal_max=result.response_signal_max,
            )

        elif self.strategy == "oracle":
            # Oracle-based comparison using error difference
            score_a = self._get_oracle_score(summary_a)
            score_b = self._get_oracle_score(summary_b)

            error_a = abs(score_a - reference_score)
            error_b = abs(score_b - reference_score)
            error_diff = error_a - error_b

            # Lower error is better, so positive diff means A is worse
            if error_diff > self.tie_margin:
                preferred = "B"
                confidence = min(0.5 + abs(error_diff) / 50, 0.95)
            elif error_diff < -self.tie_margin:
                preferred = "A"
                confidence = min(0.5 + abs(error_diff) / 50, 0.95)
            else:
                preferred = "tie"
                confidence = 0.5

            return PreferenceResult(
                preferred=preferred,
                reasoning=f"Oracle errors: A={error_a:.2f}, B={error_b:.2f}",
                confidence=confidence,
                score_estimate_a=score_a,
                score_estimate_b=score_b,
                response_signal_name="oracle_score_estimate",
                oracle_error_a=error_a,
                oracle_error_b=error_b,
            )

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    # Backward compatibility: collect_pairs delegates to collect_pairs_for_example
    def collect_pairs(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        reference_score: float,
        law_type: str = "sufficiency",
        judge_model: str = "",
    ) -> List[PreferencePair]:
        """
        Generate candidates and collect all pairwise preferences for one example.

        This is a backward-compatible wrapper around collect_pairs_for_example.

        Args:
            example_id: Unique identifier for this example
            original_text: Original source text
            rubric: Information preservation rubric
            reference_score: Ground truth score for original
            law_type: OPS law type (sufficiency, idempotence, merge)
            judge_model: Name of the judge model being used (ignored, uses strategy)

        Returns:
            List of preference pairs
        """
        return self.collect_pairs_for_example(
            example_id=example_id,
            original_text=original_text,
            rubric=rubric,
            reference_score=reference_score,
            law_type=law_type,
        )

    @property
    def pairs(self) -> List[PreferencePair]:
        """Return all collected preference pairs (property for backward compatibility)."""
        return self.get_all_pairs()

    def get_non_tie_pairs(self) -> List[PreferencePair]:
        """Return only pairs with clear preferences (no ties)."""
        return [p for p in self.get_all_pairs() if p.preferred != "tie"]

    def get_high_confidence_pairs(self, threshold: float = 0.7) -> List[PreferencePair]:
        """Return pairs with confidence above threshold."""
        return [p for p in self.get_all_pairs() if p.confidence >= threshold]

    def get_statistics(self) -> Dict[str, Any]:
        """Return comprehensive collection statistics."""
        return self.stats.to_dict()
