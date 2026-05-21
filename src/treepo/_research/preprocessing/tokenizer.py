"""
Token counting and tokenization utilities for OPS.

Provides token-aware text processing using tiktoken for accurate
token counting that matches OpenAI models and most modern LLMs.
"""

from typing import List, Optional, Tuple
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# Lazy import tiktoken to avoid slow import on module load
_tiktoken = None


def _get_tiktoken():
    """Lazy load tiktoken module."""
    global _tiktoken
    if _tiktoken is None:
        try:
            import tiktoken
            _tiktoken = tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for token counting. "
                "Install with: pip install tiktoken"
            )
    return _tiktoken


@lru_cache(maxsize=8)
def _get_encoding(encoding_name: str):
    """Get a tiktoken encoding, cached for reuse."""
    tiktoken = _get_tiktoken()
    return tiktoken.get_encoding(encoding_name)


@lru_cache(maxsize=8)
def _get_encoding_for_model(model_name: str):
    """Get tiktoken encoding for a specific model, cached for reuse."""
    tiktoken = _get_tiktoken()
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Default to cl100k_base for unknown models (standard for modern LLMs)
        # This is expected for local models with full paths (Nemotron, Qwen, etc.)
        logger.debug(
            f"Model '{model_name}' not in tiktoken registry, using cl100k_base encoding"
        )
        return tiktoken.get_encoding("cl100k_base")


class TokenCounter:
    """
    Token counter using tiktoken for accurate token counting.

    Supports both direct encoding names and model-based encoding selection.
    Uses caching for efficient repeated use.

    Example:
        >>> counter = TokenCounter()  # Uses cl100k_base (GPT-4)
        >>> counter.count("Hello, world!")
        4

        >>> counter = TokenCounter(model="gpt-4")
        >>> tokens = counter.encode("Hello, world!")
        >>> counter.decode(tokens)
        'Hello, world!'
    """

    # Common encodings with approximate chars-per-token ratios
    ENCODING_INFO = {
        "cl100k_base": {"chars_per_token": 4.0, "models": ["gpt-4", "gpt-4o", "gpt-3.5-turbo"]},
        "o200k_base": {"chars_per_token": 4.5, "models": ["gpt-4o-mini"]},
        "p50k_base": {"chars_per_token": 3.5, "models": ["text-davinci-003"]},
    }

    # Default context windows for common models
    MODEL_CONTEXT_WINDOWS = {
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-3.5-turbo": 16385,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "qwen3": 262144,  # 256k context
    }

    def __init__(
        self,
        model: Optional[str] = None,
        encoding: str = "cl100k_base"
    ):
        """
        Initialize the token counter.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-4o"). If provided,
                   encoding is determined from the model.
            encoding: Tiktoken encoding name. Only used if model is None.
        """
        if model:
            self._encoding = _get_encoding_for_model(model)
            self._model = model
        else:
            self._encoding = _get_encoding(encoding)
            self._model = None

        self._encoding_name = self._encoding.name

    @property
    def encoding_name(self) -> str:
        """Get the name of the encoding being used."""
        return self._encoding_name

    @property
    def model(self) -> Optional[str]:
        """Get the model name if one was specified."""
        return self._model

    def count(self, text: str) -> int:
        """
        Count the number of tokens in text.

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        if not text:
            return []
        return self._encoding.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        if not tokens:
            return ""
        return self._encoding.decode(tokens)

    def encode_with_offsets(self, text: str) -> List[Tuple[int, int, int]]:
        """
        Encode text and return token IDs with character offsets.

        Returns:
            List of (token_id, start_char, end_char) tuples
        """
        if not text:
            return []

        tokens = self._encoding.encode(text)
        result = []
        char_pos = 0

        for token_id in tokens:
            token_text = self._encoding.decode([token_id])
            token_len = len(token_text)
            result.append((token_id, char_pos, char_pos + token_len))
            char_pos += token_len

        return result

    def split_at_token_boundary(
        self,
        text: str,
        max_tokens: int
    ) -> Tuple[str, str]:
        """
        Split text at a token boundary, respecting max_tokens for first part.

        Args:
            text: Text to split
            max_tokens: Maximum tokens for first part

        Returns:
            Tuple of (first_part, remaining_part)
        """
        if not text:
            return ("", "")

        tokens = self._encoding.encode(text)

        if len(tokens) <= max_tokens:
            return (text, "")

        first_tokens = tokens[:max_tokens]
        remaining_tokens = tokens[max_tokens:]

        first_text = self._encoding.decode(first_tokens)
        remaining_text = self._encoding.decode(remaining_tokens)

        return (first_text, remaining_text)

    def get_context_window(self, model: Optional[str] = None) -> int:
        """
        Get the context window size for a model.

        Args:
            model: Model name (uses self.model if not provided)

        Returns:
            Context window size in tokens, or 8192 as default
        """
        model = model or self._model
        if model:
            # Normalize model name
            model_lower = model.lower()
            for known_model, context in self.MODEL_CONTEXT_WINDOWS.items():
                if known_model in model_lower:
                    return context
        return 8192  # Conservative default

    def estimate_tokens_from_chars(self, char_count: int) -> int:
        """
        Estimate token count from character count.

        This is a rough estimate - actual token count depends on text content.

        Args:
            char_count: Number of characters

        Returns:
            Estimated token count
        """
        info = self.ENCODING_INFO.get(self._encoding_name, {"chars_per_token": 4.0})
        return int(char_count / info["chars_per_token"])

    def estimate_chars_from_tokens(self, token_count: int) -> int:
        """
        Estimate character count from token count.

        This is a rough estimate - actual char count depends on text content.

        Args:
            token_count: Number of tokens

        Returns:
            Estimated character count
        """
        info = self.ENCODING_INFO.get(self._encoding_name, {"chars_per_token": 4.0})
        return int(token_count * info["chars_per_token"])


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Convenience function to count tokens in text.

    Args:
        text: Text to tokenize
        model: Model to use for encoding

    Returns:
        Number of tokens
    """
    counter = TokenCounter(model=model)
    return counter.count(text)


def get_default_max_tokens(model: str = "gpt-4", fraction: float = 0.5) -> int:
    """
    Get recommended max tokens for chunking based on model context.

    Args:
        model: Model name
        fraction: Fraction of context to use (default 0.5)

    Returns:
        Recommended max tokens for chunks
    """
    counter = TokenCounter(model=model)
    context_window = counter.get_context_window()
    return int(context_window * fraction)
