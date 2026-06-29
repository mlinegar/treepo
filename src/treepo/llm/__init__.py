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

_LAZY_EXPORTS = {
    "DiffusionBackendConfig": ("treepo.llm.diffusion", "DiffusionBackendConfig"),
    "build_diffusion_backend": ("treepo.llm.diffusion", "build_diffusion_backend"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


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
