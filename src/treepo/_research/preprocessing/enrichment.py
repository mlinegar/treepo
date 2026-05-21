"""
Pre-merge enrichment layer for ThinkingTrees (Engram WS4).

Enriches text chunks with metadata *before* tree building so the LLM
can focus on reasoning rather than content identification.

Two tiers:
  - Tier 1 (regex, ~0ms): entity extraction, key phrases, boilerplate ratio
  - Tier 2 (embedding, ~5ms): topic clustering, semantic key phrases

The enrichment metadata can be injected into merge prompts to help
the LLM focus on preserving the most important information.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from treepo._research.core.engram_memory import (
    EngramMemoryConfig,
    EngramTextNormalizer,
    extract_engram_memory_items,
)
from treepo._research.core.conditional_memory import canonical_hash

if TYPE_CHECKING:
    from treepo._research.core.conditional_memory import ConditionalMemory


@dataclass
class ChunkEnrichment:
    """Enrichment metadata for a single text chunk."""

    # Tier 1: deterministic regex-based (always available)
    word_count: int = 0
    entity_count: int = 0
    entity_density: float = 0.0
    boilerplate_ratio: float = 0.0
    key_entities: List[str] = field(default_factory=list)
    key_numbers: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)

    # Tier 2: embedding-based (optional, requires embedding client)
    topic_keywords: List[str] = field(default_factory=list)
    semantic_complexity: float = 0.0

    @property
    def is_low_complexity(self) -> bool:
        """Heuristic: chunk is mostly boilerplate / repetitive."""
        return self.boilerplate_ratio > 0.5 and self.entity_density < 0.05

    def to_prompt_block(self) -> str:
        """Format enrichment as a compact prompt block for merge context."""
        parts = []
        if self.key_entities:
            parts.append(f"Key entities: {', '.join(self.key_entities[:10])}")
        if self.key_numbers:
            parts.append(f"Key numbers: {', '.join(self.key_numbers[:5])}")
        if self.topic_keywords:
            parts.append(f"Topics: {', '.join(self.topic_keywords[:5])}")
        if not parts:
            return ""
        return "[ENRICHMENT: " + " | ".join(parts) + "]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word_count": self.word_count,
            "entity_count": self.entity_count,
            "entity_density": round(self.entity_density, 4),
            "boilerplate_ratio": round(self.boilerplate_ratio, 4),
            "key_entities": self.key_entities,
            "key_numbers": self.key_numbers,
            "urls": self.urls,
            "topic_keywords": self.topic_keywords,
            "semantic_complexity": round(self.semantic_complexity, 4),
        }


# Pre-compiled patterns for Tier 1 extraction
_CAPITALIZED_WORD = re.compile(r"\b[A-Z][a-z]+\b")
_NUMBER_PATTERN = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")
_URL_PATTERN = re.compile(r"\bhttps?://[^\s<>()\"']+\b")


class ChunkEnricher:
    """Enriches text chunks with metadata for better merge prompts.

    Parameters
    ----------
    engram_config : Optional[EngramMemoryConfig]
        Config for Engram-style entity extraction. If None, uses defaults.
    memory : Optional[ConditionalMemory]
        Shared memory for caching enrichment results.
    enable_tier2 : bool
        Whether to compute Tier 2 (embedding-based) enrichment.
    """

    def __init__(
        self,
        engram_config: Optional[EngramMemoryConfig] = None,
        memory: Optional["ConditionalMemory"] = None,
        enable_tier2: bool = False,
    ):
        self._engram_config = engram_config or EngramMemoryConfig(enabled=True)
        self._memory = memory
        self._enable_tier2 = enable_tier2
        self._normalizer = EngramTextNormalizer()

    def enrich(self, text: str) -> ChunkEnrichment:
        """Compute enrichment metadata for a text chunk.

        Tier 1 is always computed. Tier 2 is optional.
        Results are cached in ConditionalMemory if available.
        """
        # Check cache.
        if self._memory is not None:
            namespace = f"enrichment:{int(self._enable_tier2)}:{self._memory.namespace_version}"
            key = canonical_hash(text)
            cached = self._memory.get_json(namespace, key)
            if isinstance(cached, dict):
                return self._from_cached(cached)

        enrichment = self._tier1(text)

        if self._enable_tier2:
            self._tier2(text, enrichment)

        # Cache result.
        if self._memory is not None:
            namespace = f"enrichment:{int(self._enable_tier2)}:{self._memory.namespace_version}"
            key = canonical_hash(text)
            self._memory.set_json(namespace, key, enrichment.to_dict(), meta={"entity_density": enrichment.entity_density})

        return enrichment

    def enrich_batch(self, texts: List[str]) -> List[ChunkEnrichment]:
        """Enrich multiple chunks."""
        return [self.enrich(text) for text in texts]

    # ------------------------------------------------------------------
    # Tier 1: deterministic regex-based enrichment (~0ms per chunk)
    # ------------------------------------------------------------------

    def _tier1(self, text: str) -> ChunkEnrichment:
        words = text.split()
        word_count = len(words)

        # Entity extraction via Engram
        entities = extract_engram_memory_items(text, self._engram_config)

        # Separate entities by type
        key_entities = [e for e in entities if not e[0].isdigit() and "://" not in e]
        key_numbers = [e for e in entities if e[0].isdigit()]
        urls = _URL_PATTERN.findall(text)

        # Entity density
        capitalized = _CAPITALIZED_WORD.findall(text)
        entity_count = len(capitalized)
        entity_density = entity_count / max(1, word_count)

        # Boilerplate ratio (repeated bigrams / total bigrams)
        boilerplate_ratio = 0.0
        if word_count >= 4:
            lower_words = [w.lower() for w in words]
            bigrams = [f"{lower_words[i]} {lower_words[i+1]}" for i in range(len(lower_words) - 1)]
            unique_bigrams = len(set(bigrams))
            total_bigrams = len(bigrams)
            if total_bigrams > 0:
                # Inverse: more unique = less boilerplate
                boilerplate_ratio = 1.0 - (unique_bigrams / total_bigrams)

        return ChunkEnrichment(
            word_count=word_count,
            entity_count=entity_count,
            entity_density=entity_density,
            boilerplate_ratio=boilerplate_ratio,
            key_entities=key_entities[:15],
            key_numbers=key_numbers[:10],
            urls=urls[:5],
        )

    # ------------------------------------------------------------------
    # Tier 2: embedding-based enrichment (~5ms per chunk)
    # ------------------------------------------------------------------

    def _tier2(self, text: str, enrichment: ChunkEnrichment) -> None:
        """Compute embedding-based enrichment (topic keywords, complexity).

        This is a placeholder that uses keyword extraction heuristics.
        Full implementation would use the VLLMEmbeddingClient for topic
        clustering via cosine similarity to topic prototypes.
        """
        # Extract top keywords by TF-like scoring (word frequency in chunk)
        words = text.lower().split()
        word_freq: Dict[str, int] = {}
        for w in words:
            if len(w) > 4:  # Skip short words
                word_freq[w] = word_freq.get(w, 0) + 1

        # Top keywords by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
        enrichment.topic_keywords = [w for w, _ in sorted_words[:5]]

        # Semantic complexity: vocabulary diversity as proxy
        unique_count = len(set(words))
        total_count = len(words)
        enrichment.semantic_complexity = unique_count / max(1, total_count)

    # ------------------------------------------------------------------
    # Cache deserialization
    # ------------------------------------------------------------------

    @staticmethod
    def _from_cached(data: Dict[str, Any]) -> ChunkEnrichment:
        return ChunkEnrichment(
            word_count=data.get("word_count", 0),
            entity_count=data.get("entity_count", 0),
            entity_density=data.get("entity_density", 0.0),
            boilerplate_ratio=data.get("boilerplate_ratio", 0.0),
            key_entities=data.get("key_entities", []),
            key_numbers=data.get("key_numbers", []),
            urls=data.get("urls", []),
            topic_keywords=data.get("topic_keywords", []),
            semantic_complexity=data.get("semantic_complexity", 0.0),
        )
