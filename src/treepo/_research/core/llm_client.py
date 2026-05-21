"""
LLM client for OPS (Oracle-Preserving Summarization).

Provides a unified OpenAI-compatible client that works with:
- vLLM (default local inference)
- SGLang
- OpenAI API
- Any OpenAI-compatible endpoint

Designed for simplicity - just point at a server and go.
"""

from __future__ import annotations

import os
import time
import random
import threading
import logging
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Any
from enum import Enum

from treepo._research.core.engines import (
    EngineRegistry,
    EngineSurface,
    EngineType,
    resolve_engine_base_url,
    resolve_engine_for_usage,
)

logger = logging.getLogger(__name__)


class ServerType(Enum):
    """Backward-compatible server naming wrapper around `EngineType`."""

    VLLM = EngineType.VLLM.value
    SGLANG = EngineType.SGLANG.value
    OPENAI = EngineType.OPENAI.value
    CUSTOM = EngineType.CUSTOM_HTTP.value

    @classmethod
    def from_engine_type(cls, engine_type: EngineType) -> "ServerType":
        mapping = {
            EngineType.VLLM: cls.VLLM,
            EngineType.SGLANG: cls.SGLANG,
            EngineType.OPENAI: cls.OPENAI,
            EngineType.CUSTOM_HTTP: cls.CUSTOM,
        }
        if engine_type not in mapping:
            raise ValueError(f"Engine '{engine_type.value}' does not have a ServerType compatibility alias.")
        return mapping[engine_type]

    @classmethod
    def normalize(cls, value: str | EngineType | "ServerType") -> "ServerType":
        if isinstance(value, cls):
            return value
        return cls.from_engine_type(EngineType.normalize(value))

    def to_engine_type(self) -> EngineType:
        return EngineType.normalize(self.value)

    @property
    def default_port(self) -> Optional[int]:
        return self.to_engine_type().default_port


@dataclass
class LLMConfig:
    """
    Configuration for OpenAI-compatible LLM server.

    Examples:
        # vLLM local server
        config = LLMConfig.vllm(model="meta-llama/Llama-2-7b-chat-hf")

        # SGLang server
        config = LLMConfig.sglang(port=30000)

        # OpenAI API
        config = LLMConfig.openai(model="gpt-4o")
    """
    base_url: str = "http://localhost:8000/v1"
    model: str = "default"
    api_key: str = "EMPTY"  # vLLM/SGLang don't need real keys
    max_tokens: int = 8192  # Fixed typo (was 8196)
    temperature: float = 0.7
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 120.0
    server_type: ServerType = ServerType.VLLM

    @classmethod
    def vllm(
        cls,
        model: str = "default",
        host: str = "localhost",
        port: int = 8000,
        **kwargs
    ) -> 'LLMConfig':
        """Create config for vLLM server.

        If model is "default", auto-detect from the vLLM /v1/models endpoint.
        """
        from treepo._research.core.model_detection import detect_model_sync

        base_url = f"http://{host}:{port}/v1"

        # Auto-detect model name if using "default"
        if model == "default":
            model = detect_model_sync(base_url, fallback="default")

        return cls(
            base_url=base_url,
            model=model,
            api_key="EMPTY",
            server_type=ServerType.VLLM,
            **kwargs
        )

    @classmethod
    def sglang(
        cls,
        model: str = "default",
        host: str = "localhost",
        port: int = 30000,
        **kwargs
    ) -> 'LLMConfig':
        """Create config for SGLang server."""
        return cls(
            base_url=f"http://{host}:{port}/v1",
            model=model,
            api_key="EMPTY",
            server_type=ServerType.SGLANG,
            **kwargs
        )

    @classmethod
    def openai(
        cls,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        **kwargs
    ) -> 'LLMConfig':
        """Create config for OpenAI API."""
        key = api_key or os.getenv('OPENAI_API_KEY', '')
        return cls(
            base_url="https://api.openai.com/v1",
            model=model,
            api_key=key,
            server_type=ServerType.OPENAI,
            **kwargs
        )

    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create config from environment variables."""
        base_url = os.getenv('LLM_BASE_URL', 'http://localhost:8000/v1')
        model = os.getenv('LLM_MODEL', 'default')
        api_key = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY', 'EMPTY')
        return cls(base_url=base_url, model=model, api_key=api_key)

    @classmethod
    def from_engine(
        cls,
        engine: str | EngineType = EngineType.VLLM,
        *,
        model: str = "default",
        host: str = "localhost",
        port: Optional[int] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> 'LLMConfig':
        """Create a config from a generic engine selector."""
        engine_type = EngineType.normalize(engine)
        if engine_type is EngineType.OPENAI:
            return cls.openai(model=model, api_key=api_key, **kwargs)
        if engine_type is EngineType.CUSTOM_HTTP:
            if not base_url:
                raise ValueError("LLMConfig.from_engine(..., engine='custom_http') requires base_url.")
            return cls(
                base_url=base_url,
                model=model,
                api_key=api_key or "EMPTY",
                server_type=ServerType.CUSTOM,
                **kwargs,
            )
        resolve_engine_for_usage(
            engine_type,
            surface=EngineSurface.CHAT_OPENAI,
            usage="LLMConfig chat endpoint resolution",
        )
        resolved_base_url = base_url or resolve_engine_base_url(
            engine_type,
            surface=EngineSurface.CHAT_OPENAI,
            role="task",
            host=host,
            port=port,
        )
        if not resolved_base_url:
            raise ValueError(f"Could not resolve chat endpoint for engine '{engine_type.value}'.")
        return cls(
            base_url=resolved_base_url,
            model=model,
            api_key=api_key or "EMPTY",
            server_type=ServerType.from_engine_type(engine_type),
            **kwargs,
        )

    @property
    def engine_type(self) -> EngineType:
        """Primary engine-first accessor retained alongside `server_type`."""
        return self.server_type.to_engine_type()


@dataclass
class LLMResponse:
    """Response from LLM call."""
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_response: Optional[Any] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """
    OpenAI-compatible LLM client.

    Works with vLLM, SGLang, OpenAI, and any OpenAI-compatible server.

    Example:
        # With vLLM
        client = LLMClient(LLMConfig.vllm(model="llama-2-7b"))
        response = client("Summarize this text...")

        # With SGLang
        client = LLMClient(LLMConfig.sglang())
        response = client.generate("Hello, world!")

        # With OpenAI
        client = LLMClient(LLMConfig.openai(model="gpt-4o"))
    """

    def __init__(self, config: Optional[LLMConfig] = None, enable_cache: bool = True, cache_size: int = 10000):
        self.config = config or LLMConfig()
        self._client = None
        self._usage_lock = threading.Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._call_count = 0

        # LRU cache for responses (using OrderedDict for O(1) LRU operations)
        self._enable_cache = enable_cache
        self._cache: OrderedDict[str, LLMResponse] = OrderedDict()
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

            try:
                self._client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                    max_retries=0,
                )
            except TypeError:
                # Older OpenAI client versions may not accept max_retries.
                self._client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                )
                # Best-effort retry disable for clients that expose with_options.
                with_options = getattr(self._client, "with_options", None)
                if callable(with_options):
                    try:
                        self._client = with_options(max_retries=0)
                    except Exception:
                        logger.debug("OpenAI client with_options(max_retries=0) unavailable")
        return self._client

    def __call__(self, prompt: str, **kwargs) -> str:
        """Call the LLM and return just the content."""
        response = self.generate(prompt, **kwargs)
        return response.content

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: User message/prompt
            system: Optional system message
            **kwargs: Additional args passed to chat.completions.create

        Returns:
            LLMResponse with content and usage info
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return self.chat(messages, **kwargs)

    def _get_cache_key(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate cache key from messages and parameters."""
        # Create deterministic key from messages + params
        cache_data = {
            'messages': messages,
            'model': kwargs.get('model', self.config.model),
            'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            'temperature': kwargs.get('temperature', self.config.temperature),
        }
        # Only include other kwargs that affect output determinism
        for key in ['top_p', 'presence_penalty', 'frequency_penalty']:
            if key in kwargs:
                cache_data[key] = kwargs[key]

        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[LLMResponse]:
        """Get response from cache if available (O(1) LRU operation)."""
        if not self._enable_cache:
            return None

        with self._usage_lock:
            if cache_key in self._cache:
                # Move to end (most recently used) - O(1) with OrderedDict
                self._cache.move_to_end(cache_key)
                self._cache_hits += 1
                logger.debug(f"Cache hit (hits={self._cache_hits}, misses={self._cache_misses})")
                return self._cache[cache_key]

            self._cache_misses += 1
            return None

    def _add_to_cache(self, cache_key: str, response: LLMResponse) -> None:
        """Add response to cache with LRU eviction (O(1) operations)."""
        if not self._enable_cache:
            return

        with self._usage_lock:
            # If key exists, update and move to end
            if cache_key in self._cache:
                self._cache[cache_key] = response
                self._cache.move_to_end(cache_key)
                return

            # Evict oldest if at capacity - O(1) with OrderedDict.popitem
            while len(self._cache) >= self._cache_size:
                self._cache.popitem(last=False)  # Remove oldest (first) item

            # Add new entry at end (most recently used)
            self._cache[cache_key] = response

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """
        Send chat messages to the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional args passed to chat.completions.create

        Returns:
            LLMResponse with content and usage info
        """
        # Check cache first
        cache_key = self._get_cache_key(messages, **kwargs)
        cached_response = self._get_from_cache(cache_key)
        if cached_response is not None:
            return cached_response

        client = self._get_client()
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = client.chat.completions.create(
                    model=kwargs.pop('model', self.config.model),
                    messages=messages,
                    max_tokens=kwargs.pop('max_tokens', self.config.max_tokens),
                    temperature=kwargs.pop('temperature', self.config.temperature),
                    **kwargs
                )

                # Track usage
                with self._usage_lock:
                    self._call_count += 1
                    if response.usage:
                        self._prompt_tokens += response.usage.prompt_tokens
                        self._completion_tokens += response.usage.completion_tokens

                llm_response = LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    raw_response=response
                )

                # Add to cache
                self._add_to_cache(cache_key, llm_response)

                return llm_response

            except Exception as e:
                last_error = e
                delay = self.config.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}. Retrying in {delay:.1f}s")
                time.sleep(delay)

        raise RuntimeError(f"LLM call failed after {self.config.max_retries} attempts: {last_error}")

    def get_usage(self) -> Dict[str, int]:
        """Get current token usage and cache statistics."""
        with self._usage_lock:
            total_requests = self._cache_hits + self._cache_misses
            cache_hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
            return {
                'prompt_tokens': self._prompt_tokens,
                'completion_tokens': self._completion_tokens,
                'total_tokens': self._prompt_tokens + self._completion_tokens,
                'call_count': self._call_count,
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'cache_hit_rate': cache_hit_rate,
                'cache_size': len(self._cache)
            }

    def reset_usage(self) -> Dict[str, int]:
        """Get usage and reset counters."""
        with self._usage_lock:
            usage = {
                'prompt_tokens': self._prompt_tokens,
                'completion_tokens': self._completion_tokens,
                'total_tokens': self._prompt_tokens + self._completion_tokens,
                'call_count': self._call_count
            }
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._call_count = 0
            return usage


class MockLLMClient:
    """Mock LLM client for testing without a server."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        response_fn: Optional[Callable[[str], str]] = None
    ):
        self.config = config or LLMConfig()
        self.response_fn = response_fn or self._default_response
        self.calls: List[str] = []
        self._call_count = 0

    def _default_response(self, prompt: str) -> str:
        """Default mock response: echo input with a simple label."""
        if len(prompt) > 100:
            return f"Summary: {prompt}"
        return f"Response to: {prompt}"

    def __call__(self, prompt: str, **kwargs) -> str:
        """Call the mock LLM."""
        return self.generate(prompt, **kwargs).content

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate mock response."""
        self.calls.append(prompt)
        self._call_count += 1
        content = self.response_fn(prompt)
        return LLMResponse(
            content=content,
            model="mock",
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(content.split())
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Handle chat-style input."""
        prompt = messages[-1]['content'] if messages else ""
        return self.generate(prompt, **kwargs)

    def reset(self) -> None:
        """Reset call history."""
        self.calls = []
        self._call_count = 0

    def get_usage(self) -> Dict[str, int]:
        """Get usage stats."""
        return {'call_count': self._call_count, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}


def create_client(config: Optional[LLMConfig] = None, mock: bool = False) -> LLMClient:
    """
    Create an LLM client.

    Args:
        config: LLM configuration (uses defaults if None)
        mock: If True, return MockLLMClient for testing

    Returns:
        LLMClient or MockLLMClient
    """
    if mock:
        return MockLLMClient(config)
    return LLMClient(config)


def create_summarizer(
    client: Optional[LLMClient] = None,
    system_prompt: Optional[str] = None
) -> Callable[[str, str], str]:
    """
    Create a summarizer function compatible with TreeBuilder.

    Args:
        client: LLM client (creates mock if None)
        system_prompt: Optional system prompt for summarization

    Returns:
        Callable that takes (content, rubric) and returns summary
    """
    if client is None:
        client = MockLLMClient()

    default_system = "You are a precise summarizer. Preserve all information specified in the rubric."

    def summarizer(content: str, rubric: str) -> str:
        prompt = f"""Summarize the following content while preserving information specified in the rubric.

Rubric: {rubric}

Content:
{content}

Summary:"""
        if hasattr(client, 'generate'):
            return client.generate(prompt, system=system_prompt or default_system).content
        return client(prompt)

    return summarizer


# Convenience aliases
def vllm_client(model: str = "default", host: str = "localhost", port: int = 8000, **kwargs) -> LLMClient:
    """Create client for vLLM server."""
    return LLMClient(LLMConfig.vllm(model=model, host=host, port=port, **kwargs))


def sglang_client(model: str = "default", host: str = "localhost", port: int = 30000, **kwargs) -> LLMClient:
    """Create client for SGLang server."""
    return LLMClient(LLMConfig.sglang(model=model, host=host, port=port, **kwargs))


def openai_client(model: str = "gpt-4o", api_key: Optional[str] = None, **kwargs) -> LLMClient:
    """Create client for OpenAI API."""
    return LLMClient(LLMConfig.openai(model=model, api_key=api_key, **kwargs))


def engine_client(
    engine: str | ServerType = ServerType.VLLM,
    *,
    model: str = "default",
    host: str = "localhost",
    port: Optional[int] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """Create an LLM client from a generic engine selector."""
    return LLMClient(
        LLMConfig.from_engine(
            engine,
            model=model,
            host=host,
            port=port,
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )
    )
