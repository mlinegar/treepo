"""LLM client utilities.

This package is import-light. Concrete providers are imported lazily by their
modules and should remain behind the `treepo[llm]` extra where possible.
"""

from treepo.llm.embedding import (
    DenseHashEmbeddingClient,
    DiskCachedEmbeddingClient,
    EmbeddingClient,
    EmbeddingClientConfig,
    HashEmbeddingClient,
    HashingEmbeddingClient,
    OpenAICompatibleEmbeddingClient,
    OpenAIEmbeddingClient,
    TransformersEmbeddingClient,
    VLLMEmbeddingClient,
    build_embedding_client,
)
from treepo.llm.openai_compatible import ChatMessage, OpenAICompatibleConfig, render_chat_payload

__all__ = [
    "ChatMessage",
    "DenseHashEmbeddingClient",
    "DiskCachedEmbeddingClient",
    "EmbeddingClient",
    "EmbeddingClientConfig",
    "HashEmbeddingClient",
    "HashingEmbeddingClient",
    "OpenAICompatibleConfig",
    "OpenAICompatibleEmbeddingClient",
    "OpenAIEmbeddingClient",
    "TransformersEmbeddingClient",
    "VLLMEmbeddingClient",
    "build_embedding_client",
    "render_chat_payload",
]
