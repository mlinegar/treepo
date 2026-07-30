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
from treepo.llm.json_records import (
    JSONRequestRecord,
    JSONResultRecord,
    TokenUsage,
    parse_json_object,
    validate_json_object,
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
    "JSONRequestRecord",
    "JSONResultRecord",
    "OpenAICompatibleChatClient",
    "OpenAICompatibleEmbeddingClient",
    "TokenUsage",
    "build_chat_client",
    "build_embedding_client",
    "parse_json_object",
    "render_chat_payload",
    "validate_json_object",
]
