"""
Retriever for Example-based Few-Shot Learning.

This module provides semantic retrieval for training examples,
useful for few-shot learning and context selection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from .core import UnifiedTrainingExample


# =============================================================================
# Retriever
# =============================================================================

class Retriever:
    """
    Semantic retriever using SentenceTransformer embeddings.

    Supports example retrieval: Find similar training examples for few-shot context.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the retriever.

        Args:
            model_name: SentenceTransformer model name
            cache_dir: Directory for caching embeddings
            device: Device to use ("cpu", "cuda", or None for auto)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or Path("data/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy load model
        self._model = None
        self._device = device

        # Cache for example embeddings
        self._example_embeddings: Optional[torch.Tensor] = None
        self._examples: List[UnifiedTrainingExample] = []

    @property
    def model(self):
        """Lazy load the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            if self._device:
                self._model.to(self._device)
            else:
                self._model.to("cpu")
        return self._model

    def index_examples(
        self,
        examples: List[UnifiedTrainingExample],
        force_recompute: bool = False,
    ):
        """
        Index training examples for retrieval.

        Args:
            examples: List of training examples to index
            force_recompute: Whether to recompute embeddings even if cached
        """
        self._examples = examples

        # Create text representation of each example (full content for accurate embeddings)
        example_texts = [
            f"Content: {ex.original_content}\nSummary: {ex.summary}"
            for ex in examples
        ]

        # Compute embeddings
        self._example_embeddings = self.model.encode(
            example_texts,
            convert_to_tensor=True,
            show_progress_bar=len(example_texts) > 100,
        )

    def retrieve_examples(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Tuple[float, UnifiedTrainingExample]]:
        """
        Retrieve top-k training examples most similar to the query.

        Args:
            query: Query text to match against examples
            top_k: Number of examples to retrieve

        Returns:
            List of (score, example) tuples, sorted by score descending
        """
        if self._example_embeddings is None:
            raise ValueError("Examples not indexed. Call index_examples() first.")

        from sentence_transformers import util as st_util

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = st_util.cos_sim(query_embedding, self._example_embeddings)[0]

        # Get top-k indices
        top_indices = torch.topk(scores, min(top_k, len(self._examples))).indices

        return [
            (scores[i].item(), self._examples[i])
            for i in top_indices
        ]
