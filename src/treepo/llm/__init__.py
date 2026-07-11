"""LLM client utilities.

This package is import-light. Concrete providers are imported lazily by their
modules and should remain behind the `treepo[llm]` extra where possible.
"""

from treepo.llm.embedding import (
    DiskCachedEmbeddingClient,
    EmbeddingClient,
    HashingEmbeddingClient,
    OpenAICompatibleEmbeddingClient,
    build_embedding_client,
)
from treepo.llm.openai_compatible import (
    ChatMessage,
    OpenAICompatibleChatClient,
    build_chat_client,
    render_chat_payload,
)

__all__ = [
    "ChatMessage",
    "DiskCachedEmbeddingClient",
    "EmbeddingClient",
    "HashingEmbeddingClient",
    "OpenAICompatibleChatClient",
    "OpenAICompatibleEmbeddingClient",
    "build_chat_client",
    "build_embedding_client",
    "render_chat_payload",
]
