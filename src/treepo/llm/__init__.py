"""LLM client utilities.

This package is import-light. Concrete providers are imported lazily by their
modules and should remain behind the `treepo[llm]` extra where possible.
"""

from treepo.llm.openai_compatible import ChatMessage, OpenAICompatibleConfig, render_chat_payload

__all__ = ["ChatMessage", "OpenAICompatibleConfig", "render_chat_payload"]
