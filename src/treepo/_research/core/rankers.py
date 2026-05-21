"""
Pluggable ranker protocols for candidate selection.

This module provides a unified interface for ranking/scoring candidate summaries,
supporting the tournament-based selection system. The design mirrors re-rankers
in RAG systems:

    RAG: query + docs → relevance scores
    OPS: context + summaries → quality scores

Available Rankers:
- GenRMRanker: Uses GenRM model for pairwise comparisons → tournament scores
- EmbeddingRanker: Fast cosine similarity scoring (placeholder)
- EnsembleRanker: Weighted combination of multiple rankers (future)

Usage:
    from treepo._research.core.rankers import GenRMRanker, get_ranker

    # Create ranker wrapping existing GenRM judge
    ranker = GenRMRanker(genrm_judge)

    # Score candidates
    scores = ranker.score(context="Original text...", candidates=["sum1", "sum2", "sum3"])
    # Returns: [0.85, 0.72, 0.91]

    # Rank candidates (returns indices sorted by score descending)
    ranks = ranker.rank(context="...", candidates=["sum1", "sum2", "sum3"])
    # Returns: [2, 0, 1]  (meaning candidates[2] is best, then [0], then [1])

    # Or use factory
    ranker = get_ranker("genrm", judge=my_judge)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

logger = logging.getLogger(__name__)


# =============================================================================
# Ranker Protocol
# =============================================================================

@dataclass
class RankingResult:
    """Result from ranking candidates."""

    scores: List[float]
    """Scores for each candidate (higher = better)."""

    ranks: List[int]
    """Rank for each candidate (1 = best)."""

    best_idx: int
    """Index of the best candidate."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Optional metadata (e.g., pairwise comparison details)."""


@runtime_checkable
class Ranker(Protocol):
    """
    Protocol for candidate rankers.

    Rankers score/rank candidate summaries against a context (original text).
    This abstraction supports:
    - Tournament-based selection (existing GenRM flow)
    - Fast embedding-based scoring (future)
    - Ensemble combinations (future)
    """

    def score(
        self,
        context: str,
        candidates: List[str],
        rubric: Optional[str] = None,
    ) -> List[float]:
        """
        Score each candidate summary against the context.

        Args:
            context: Original text / context to compare against
            candidates: List of candidate summaries to score
            rubric: Optional criteria for comparison

        Returns:
            List of scores (one per candidate), higher = better
        """
        ...

    def rank(
        self,
        context: str,
        candidates: List[str],
        rubric: Optional[str] = None,
    ) -> RankingResult:
        """
        Rank candidates and return full result with scores and ranks.

        Args:
            context: Original text / context to compare against
            candidates: List of candidate summaries to rank
            rubric: Optional criteria for comparison

        Returns:
            RankingResult with scores, ranks, and best index
        """
        ...


# =============================================================================
# Base Ranker with Common Logic
# =============================================================================

class BaseRanker(ABC):
    """Base class for rankers with common ranking logic."""

    @abstractmethod
    def score(
        self,
        context: str,
        candidates: List[str],
        rubric: Optional[str] = None,
    ) -> List[float]:
        """Score each candidate. Must be implemented by subclasses."""
        pass

    def rank(
        self,
        context: str,
        candidates: List[str],
        rubric: Optional[str] = None,
    ) -> RankingResult:
        """
        Rank candidates based on scores.

        Default implementation uses score() and sorts.
        """
        scores = self.score(context, candidates, rubric)

        # Create (score, original_idx) pairs and sort descending
        indexed_scores = [(s, i) for i, s in enumerate(scores)]
        sorted_pairs = sorted(indexed_scores, key=lambda x: x[0], reverse=True)

        # Compute ranks (1 = best)
        ranks = [0] * len(candidates)
        for rank_idx, (score, orig_idx) in enumerate(sorted_pairs):
            ranks[orig_idx] = rank_idx + 1

        best_idx = sorted_pairs[0][1] if sorted_pairs else 0

        return RankingResult(
            scores=scores,
            ranks=ranks,
            best_idx=best_idx,
        )


# =============================================================================
# GenRM Ranker (Wraps Existing GenRM Judge)
# =============================================================================

class GenRMRanker(BaseRanker):
    """
    Ranker using GenRM model for pairwise comparisons.

    Uses tournament-style comparison: runs all pairwise comparisons
    and aggregates to produce scores for each candidate.

    This wraps the existing GenRMJudge from src/training/preference/genrm.py
    """

    def __init__(
        self,
        judge: Any,  # GenRMJudge instance
        aggregation: str = "win_rate",
    ):
        """
        Initialize GenRM ranker.

        Args:
            judge: GenRMJudge instance with .compare() method
            aggregation: How to aggregate pairwise comparisons
                - "win_rate": Score = wins / (wins + losses)
                - "elo": Use Elo-style rating (future)
        """
        self.judge = judge
        self.aggregation = aggregation

    def score(
        self,
        context: str,
        candidates: List[str],
        rubric: Optional[str] = None,
    ) -> List[float]:
        """
        Score candidates via pairwise tournament.

        For k candidates, runs k*(k-1)/2 comparisons and aggregates.
        """
        if len(candidates) < 2:
            return [1.0] * len(candidates)

        k = len(candidates)
        wins = [0] * k
        comparisons = [0] * k

        # Run all pairwise comparisons
        for i in range(k):
            for j in range(i + 1, k):
                try:
                    result = self.judge.compare(
                        context=rubric or "Compare these summaries",
                        original_text=context,
                        summary_a=candidates[i],
                        summary_b=candidates[j],
                    )

                    comparisons[i] += 1
                    comparisons[j] += 1

                    if hasattr(result, 'preferred'):
                        if result.preferred == "A":
                            wins[i] += 1
                        elif result.preferred == "B":
                            wins[j] += 1
                        else:  # tie
                            wins[i] += 0.5
                            wins[j] += 0.5

                except Exception as e:
                    logger.warning(f"GenRM comparison failed: {e}")
                    # On error, treat as tie
                    wins[i] += 0.5
                    wins[j] += 0.5
                    comparisons[i] += 1
                    comparisons[j] += 1

        # Compute scores based on aggregation method
        if self.aggregation == "win_rate":
            scores = [
                wins[i] / max(comparisons[i], 1)
                for i in range(k)
            ]
        else:
            # Default to win rate
            scores = [wins[i] / max(comparisons[i], 1) for i in range(k)]

        return scores

    def rank(
        self,
        context: str,
        candidates: List[str],
        rubric: Optional[str] = None,
    ) -> RankingResult:
        """Rank with metadata about pairwise comparisons."""
        result = super().rank(context, candidates, rubric)
        result.metadata["ranker_type"] = "genrm"
        result.metadata["aggregation"] = self.aggregation
        result.metadata["num_comparisons"] = len(candidates) * (len(candidates) - 1) // 2
        return result


# =============================================================================
# Embedding Ranker (Placeholder for Future)
# =============================================================================

class EmbeddingRanker(BaseRanker):
    """
    Ranker using embedding similarity.

    Fast and cheap alternative to GenRM for initial filtering.
    Uses cosine similarity between context and candidate embeddings.

    Note: This is a placeholder. Full implementation requires:
    - sentence-transformers or similar embedding model
    - vLLM embedding endpoint integration
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize embedding ranker.

        Args:
            embed_fn: Custom embedding function (str -> vector)
            model_name: Sentence transformer model name (if embed_fn not provided)
        """
        self.embed_fn = embed_fn
        self.model_name = model_name
        self._model = None

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts."""
        if self.embed_fn:
            return [self.embed_fn(t) for t in texts]

        # Lazy load sentence-transformers
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "EmbeddingRanker requires sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                )

        return self._model.encode(texts).tolist()

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def score(
        self,
        context: str,
        candidates: List[str],
        rubric: Optional[str] = None,
    ) -> List[float]:
        """
        Score candidates via embedding similarity.

        Higher similarity to context = better score.
        """
        if not candidates:
            return []

        # Include rubric in context if provided
        full_context = f"{context}\n\nCriteria: {rubric}" if rubric else context

        # Get embeddings
        all_texts = [full_context] + candidates
        embeddings = self._get_embeddings(all_texts)

        context_emb = embeddings[0]
        candidate_embs = embeddings[1:]

        # Compute similarity scores
        scores = [
            self._cosine_similarity(context_emb, cand_emb)
            for cand_emb in candidate_embs
        ]

        # Normalize to 0-1 range (cosine sim is already -1 to 1)
        scores = [(s + 1) / 2 for s in scores]

        return scores

    def rank(
        self,
        context: str,
        candidates: List[str],
        rubric: Optional[str] = None,
    ) -> RankingResult:
        """Rank with embedding metadata."""
        result = super().rank(context, candidates, rubric)
        result.metadata["ranker_type"] = "embedding"
        result.metadata["model_name"] = self.model_name
        return result


# =============================================================================
# Ensemble Ranker (Future)
# =============================================================================

class EnsembleRanker(BaseRanker):
    """
    Ensemble of multiple rankers with weighted combination.

    Useful for combining fast (embedding) with accurate (GenRM) rankers:
    - Use embedding for initial filtering (top-k from many)
    - Use GenRM for final ranking (expensive but accurate)

    Or for combining different signal sources.
    """

    def __init__(
        self,
        rankers: List[Tuple[BaseRanker, float]],
        normalize: bool = True,
    ):
        """
        Initialize ensemble ranker.

        Args:
            rankers: List of (ranker, weight) tuples
            normalize: Whether to normalize scores before combining
        """
        self.rankers = rankers
        self.normalize = normalize

        # Validate weights sum to 1
        total_weight = sum(w for _, w in rankers)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Ensemble weights sum to {total_weight}, normalizing")
            self.rankers = [(r, w / total_weight) for r, w in rankers]

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return scores
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [0.5] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def score(
        self,
        context: str,
        candidates: List[str],
        rubric: Optional[str] = None,
    ) -> List[float]:
        """
        Score candidates using weighted ensemble.
        """
        if not candidates:
            return []

        # Collect scores from all rankers
        all_scores = []
        for ranker, weight in self.rankers:
            scores = ranker.score(context, candidates, rubric)
            if self.normalize:
                scores = self._normalize_scores(scores)
            all_scores.append((scores, weight))

        # Weighted combination
        combined = [0.0] * len(candidates)
        for scores, weight in all_scores:
            for i, s in enumerate(scores):
                combined[i] += s * weight

        return combined

    def rank(
        self,
        context: str,
        candidates: List[str],
        rubric: Optional[str] = None,
    ) -> RankingResult:
        """Rank with ensemble metadata."""
        result = super().rank(context, candidates, rubric)
        result.metadata["ranker_type"] = "ensemble"
        result.metadata["num_rankers"] = len(self.rankers)
        result.metadata["ranker_weights"] = [w for _, w in self.rankers]
        return result


# =============================================================================
# Factory Function
# =============================================================================

_RANKER_REGISTRY: Dict[str, type] = {
    "genrm": GenRMRanker,
    "embedding": EmbeddingRanker,
    "ensemble": EnsembleRanker,
}


def get_ranker(
    ranker_type: str,
    **kwargs,
) -> BaseRanker:
    """
    Factory function to create rankers.

    Args:
        ranker_type: Type of ranker ("genrm", "embedding", "ensemble")
        **kwargs: Arguments passed to ranker constructor

    Returns:
        Ranker instance

    Example:
        ranker = get_ranker("genrm", judge=my_genrm_judge)
        ranker = get_ranker("embedding", model_name="all-MiniLM-L6-v2")
    """
    if ranker_type not in _RANKER_REGISTRY:
        raise ValueError(
            f"Unknown ranker type: {ranker_type}. "
            f"Available: {list(_RANKER_REGISTRY.keys())}"
        )

    return _RANKER_REGISTRY[ranker_type](**kwargs)


def register_ranker(name: str, ranker_class: type) -> None:
    """Register a custom ranker type."""
    _RANKER_REGISTRY[name] = ranker_class
