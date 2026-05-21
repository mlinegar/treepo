"""
DSPy configuration with XMLAdapter for robust output parsing.

This module provides centralized DSPy configuration that uses XMLAdapter
instead of the default ChatAdapter. XMLAdapter uses <field_name>value</field_name>
format which is more robust for parsing than the [[ ## field_name ## ]] format.

Also provides a unified LM factory for local OpenAI-compatible DSPy language models.
"""

import atexit
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import dspy
from dspy.adapters import XMLAdapter

from treepo._research.config.context_window import DEFAULT_CONTEXT_WINDOW, ContextWindowManager
from treepo._research.core.dspy_batch_client import BatchedDSPyLM
from treepo._research.core.engines import EngineType, LocalChatEndpoints, resolve_local_chat_endpoints
from treepo._research.core.model_detection import detect_model_from_port, get_context_window_from_port

logger = logging.getLogger(__name__)


_xml_adapter: Optional[XMLAdapter] = None
_dspy_cache_runtime_signature: Optional[Tuple[bool, bool, str, int, int]] = None
_dspy_cache_cleanup_registered = False


def get_xml_adapter() -> XMLAdapter:
    """
    Get or create a singleton XMLAdapter instance.

    Returns:
        XMLAdapter instance for use with dspy.configure()
    """
    global _xml_adapter
    if _xml_adapter is None:
        _xml_adapter = XMLAdapter()
    return _xml_adapter


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _close_cache_instance(cache_obj: Any) -> None:
    if cache_obj is None:
        return
    try:
        disk_cache = getattr(cache_obj, "disk_cache", None)
        close_fn = getattr(disk_cache, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception as exc:
        logger.debug("DSPy cache close failed: %s", exc)


def close_dspy_cache(*, reset_memory_cache: bool = False) -> None:
    """
    Close any open DSPy cache handles.

    This is intended for end-of-run cleanup in long-lived training processes.
    A later `configure_dspy(...)` call will recreate the cache if needed.
    """
    global _dspy_cache_runtime_signature

    cache_obj = getattr(dspy, "cache", None)
    if cache_obj is None:
        _dspy_cache_runtime_signature = None
        return

    if reset_memory_cache:
        try:
            reset_fn = getattr(cache_obj, "reset_memory_cache", None)
            if callable(reset_fn):
                reset_fn()
        except Exception as exc:
            logger.debug("DSPy memory cache reset failed: %s", exc)

    _close_cache_instance(cache_obj)
    _dspy_cache_runtime_signature = None


def configure_dspy_cache_from_env(*, force: bool = False) -> tuple[bool, bool]:
    """
    Reconfigure DSPy's global cache from environment flags.

    Supported env vars:
    - `TT_DSPY_ENABLE_DISK_CACHE`  (default: enabled)
    - `TT_DSPY_ENABLE_MEMORY_CACHE` (default: enabled)
    - `DSPY_CACHEDIR`
    - `DSPY_CACHE_LIMIT`
    - `TT_DSPY_MEMORY_CACHE_MAX_ENTRIES`
    """
    global _dspy_cache_runtime_signature
    global _dspy_cache_cleanup_registered

    enable_disk_cache = _env_flag("TT_DSPY_ENABLE_DISK_CACHE", True)
    enable_memory_cache = _env_flag("TT_DSPY_ENABLE_MEMORY_CACHE", True)
    disk_cache_dir = str(
        os.getenv("DSPY_CACHEDIR")
        or (Path.home() / ".dspy_cache")
    )
    try:
        disk_size_limit_bytes = int(float(os.getenv("DSPY_CACHE_LIMIT", "30000000000")))
    except (TypeError, ValueError):
        disk_size_limit_bytes = int(3e10)
    try:
        memory_max_entries = int(float(os.getenv("TT_DSPY_MEMORY_CACHE_MAX_ENTRIES", "1000000")))
    except (TypeError, ValueError):
        memory_max_entries = 1000000

    signature = (
        bool(enable_disk_cache),
        bool(enable_memory_cache),
        str(disk_cache_dir),
        int(disk_size_limit_bytes),
        int(memory_max_entries),
    )
    if not force and signature == _dspy_cache_runtime_signature:
        return enable_disk_cache, enable_memory_cache

    old_cache = getattr(dspy, "cache", None)
    dspy.configure_cache(
        enable_disk_cache=enable_disk_cache,
        enable_memory_cache=enable_memory_cache,
        disk_cache_dir=disk_cache_dir,
        disk_size_limit_bytes=disk_size_limit_bytes,
        memory_max_entries=memory_max_entries,
    )
    new_cache = getattr(dspy, "cache", None)
    if old_cache is not None and old_cache is not new_cache:
        _close_cache_instance(old_cache)

    _dspy_cache_runtime_signature = signature
    if not _dspy_cache_cleanup_registered:
        atexit.register(close_dspy_cache)
        _dspy_cache_cleanup_registered = True

    logger.info(
        "DSPy cache config: disk=%s memory=%s dir=%s",
        str(enable_disk_cache).lower(),
        str(enable_memory_cache).lower(),
        disk_cache_dir,
    )
    return enable_disk_cache, enable_memory_cache


def configure_dspy(
    lm: dspy.LM,
    adapter: Optional[Any] = None,
    **kwargs
) -> None:
    """
    Configure DSPy with XMLAdapter by default.

    This is a drop-in replacement for dspy.configure() that uses XMLAdapter
    for more robust output parsing.

    Args:
        lm: The DSPy language model to use
        adapter: Optional custom adapter (defaults to XMLAdapter)
        **kwargs: Additional arguments passed to dspy.configure()
            (e.g., async_max_workers)

    Example:
        from treepo._research.config.dspy_config import configure_dspy

        lm = dspy.LM("openai/model", api_base="...", api_key="...")
        configure_dspy(lm=lm)
    """
    configure_dspy_cache_from_env()

    if adapter is None:
        adapter = get_xml_adapter()

    # Merge optional DSPy runtime overrides from settings.yaml.
    try:
        from treepo._research.config.settings import load_settings

        settings = load_settings()
        dspy_overrides = settings.get("dspy", {}) if isinstance(settings, dict) else {}
    except Exception:
        dspy_overrides = {}

    for key, value in dspy_overrides.items():
        if key not in kwargs:
            kwargs[key] = value

    dspy.configure(lm=lm, adapter=adapter, **kwargs)


def create_local_engine_lm(
    *,
    engine: Optional[str | EngineType] = None,
    endpoints: Optional[LocalChatEndpoints] = None,
    ports: Optional[Sequence[int]] = None,
    port: Optional[int] = None,
    model: Optional[str] = None,
    settings: Optional[Mapping[str, Any]] = None,
    temperature: float = 0.5,
    max_tokens: Optional[int] = None,
    cache: bool = True,
    batch_max_concurrent: int = 200,
    batch_size: int = 50,
    batch_timeout: float = 0.02,
    batch_request_timeout: float = 300.0,
    batch_await_response_timeout: Optional[float] = None,
    routing_policy: str = "affinity_load_aware",
    **kwargs,
) -> dspy.LM:
    """Create the standard local OpenAI-compatible DSPy LM.

    This is the streamlined path for local vLLM/SGLang inference: DSPy keeps
    program composition and parsing, while HTTP transport always goes through
    ``AsyncBatchLLMClient`` / ``MultiServerBatchClient``.
    """
    if settings is None:
        try:
            from treepo._research.config.settings import load_settings

            settings = load_settings()
        except Exception:
            settings = None

    if engine is None:
        try:
            from treepo._research.config.settings import get_inference_backend_config

            engine = get_inference_backend_config(dict(settings or {})).get("task_backend") or "vllm"
        except Exception:
            engine = "vllm"

    if endpoints is None:
        endpoints = resolve_local_chat_endpoints(
            engine=engine,
            ports=ports,
            port=port,
            settings=settings,
            usage="DSPy local engine LM",
            allowed_engines=(EngineType.VLLM, EngineType.SGLANG),
        )
    base_urls = list(endpoints.base_urls)
    primary_port = endpoints.primary_port
    if model is None and primary_port is not None:
        model = detect_model_from_port(port=primary_port)
    if model is None:
        model = "default"

    if max_tokens is None:
        context_window = (
            get_context_window_from_port(port=primary_port)
            if primary_port is not None
            else DEFAULT_CONTEXT_WINDOW
        )
        manager = ContextWindowManager(context_window=context_window)
        max_tokens = manager.max_output_tokens

    return BatchedDSPyLM(
        model=f"openai/{model}",
        api_bases=base_urls,
        api_key="EMPTY",
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        max_concurrent=batch_max_concurrent,
        batch_size=batch_size,
        batch_timeout=batch_timeout,
        request_timeout=batch_request_timeout,
        await_response_timeout=batch_await_response_timeout,
        routing_policy=routing_policy,
        **kwargs,
    )


def create_vllm_lm(
    port: int,
    model: Optional[str] = None,
    temperature: float = 0.5,
    max_tokens: Optional[int] = None,
    cache: bool = True,
    batch_max_concurrent: int = 200,
    batch_size: int = 50,
    batch_timeout: float = 0.02,
    batch_request_timeout: float = 300.0,
    batch_await_response_timeout: Optional[float] = None,
    **kwargs,
) -> dspy.LM:
    """Compatibility wrapper for the standard batched local vLLM DSPy LM."""
    return create_local_engine_lm(
        engine="vllm",
        port=port,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        batch_max_concurrent=batch_max_concurrent,
        batch_size=batch_size,
        batch_timeout=batch_timeout,
        batch_request_timeout=batch_request_timeout,
        batch_await_response_timeout=batch_await_response_timeout,
        **kwargs,
    )


def create_vllm_lm_multi(
    ports: Sequence[int],
    model: Optional[str] = None,
    temperature: float = 0.5,
    max_tokens: Optional[int] = None,
    cache: bool = True,
    batch_max_concurrent: int = 200,
    batch_size: int = 50,
    batch_timeout: float = 0.02,
    batch_request_timeout: float = 300.0,
    batch_await_response_timeout: Optional[float] = None,
    routing_policy: str = "affinity_load_aware",
    **kwargs,
) -> dspy.LM:
    """Compatibility wrapper for the standard batched multi-vLLM DSPy LM."""
    return create_local_engine_lm(
        engine="vllm",
        ports=ports,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        batch_max_concurrent=batch_max_concurrent,
        batch_size=batch_size,
        batch_timeout=batch_timeout,
        batch_request_timeout=batch_request_timeout,
        batch_await_response_timeout=batch_await_response_timeout,
        routing_policy=routing_policy,
        **kwargs,
    )


def create_local_engine_lm_with_manager(
    *,
    engine: Optional[str | EngineType] = None,
    endpoints: Optional[LocalChatEndpoints] = None,
    ports: Optional[Sequence[int]] = None,
    port: Optional[int] = None,
    model: Optional[str] = None,
    settings: Optional[Mapping[str, Any]] = None,
    temperature: float = 0.5,
    cache: bool = True,
    task: str = "default",
    **kwargs,
) -> Tuple[dspy.LM, ContextWindowManager]:
    """Create a local-engine DSPy LM and its ContextWindowManager.

    Use this when callers need the same engine-neutral batched DSPy transport
    plus a context manager for request-specific token budgeting.
    """
    from treepo._research.config.context_window import create_manager_for_task

    if settings is None:
        try:
            from treepo._research.config.settings import load_settings

            settings = load_settings()
        except Exception:
            settings = None

    if engine is None and endpoints is None:
        try:
            from treepo._research.config.settings import get_inference_backend_config

            engine = get_inference_backend_config(dict(settings or {})).get("task_backend") or "vllm"
        except Exception:
            engine = "vllm"

    if endpoints is None:
        endpoints = resolve_local_chat_endpoints(
            engine=engine,
            ports=ports,
            port=port,
            settings=settings,
            usage="DSPy local engine LM with manager",
            allowed_engines=(EngineType.VLLM, EngineType.SGLANG),
        )

    primary_port = endpoints.primary_port
    if model is None and primary_port is not None:
        model = detect_model_from_port(port=primary_port)

    context_window = (
        get_context_window_from_port(port=primary_port)
        if primary_port is not None
        else DEFAULT_CONTEXT_WINDOW
    )
    manager = create_manager_for_task(context_window=context_window, task=task)

    lm = create_local_engine_lm(
        engine=engine,
        endpoints=endpoints,
        model=model,
        settings=settings,
        temperature=temperature,
        max_tokens=manager.max_output_tokens,
        cache=cache,
        **kwargs,
    )

    return lm, manager


def create_vllm_lm_with_manager(
    port: int,
    model: Optional[str] = None,
    temperature: float = 0.5,
    cache: bool = True,
    task: str = "default",
    **kwargs,
) -> Tuple[dspy.LM, ContextWindowManager]:
    """Compatibility wrapper for vLLM callers that need a context manager."""
    return create_local_engine_lm_with_manager(
        engine="vllm",
        port=port,
        model=model,
        temperature=temperature,
        cache=cache,
        task=task,
        **kwargs,
    )
