"""
Centralized model detection for vLLM servers.

This module consolidates model auto-detection logic that was previously
duplicated in llm_client.py, batch_processor.py, and llm_utils.py.
"""

import logging
from typing import Optional

import aiohttp
import requests

logger = logging.getLogger(__name__)


def detect_model_sync(
    base_url: str,
    fallback: str = "default",
    timeout: float = 5.0,
) -> str:
    """
    Synchronously detect vLLM model name from server.

    Args:
        base_url: Base URL of the vLLM server (e.g., "http://localhost:8000/v1")
        fallback: Model name to return if detection fails
        timeout: Request timeout in seconds

    Returns:
        Detected model ID or fallback
    """
    try:
        response = requests.get(f"{base_url}/models", timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if data.get("data") and len(data["data"]) > 0:
            model_id = data["data"][0]["id"]
            logger.debug(f"Auto-detected model: {model_id}")
            return model_id
    except requests.RequestException as e:
        logger.debug(f"Failed to auto-detect model from {base_url}: {e}")
    except (KeyError, IndexError, ValueError) as e:
        logger.debug(f"Failed to parse model response: {e}")
    return fallback


async def detect_model_async(
    base_url: str,
    fallback: str = "default",
    timeout: float = 5.0,
) -> str:
    """
    Asynchronously detect vLLM model name from server.

    Args:
        base_url: Base URL of the vLLM server (e.g., "http://localhost:8000/v1")
        fallback: Model name to return if detection fails
        timeout: Request timeout in seconds

    Returns:
        Detected model ID or fallback
    """
    try:
        timeout_config = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.get(f"{base_url}/models") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("data") and len(data["data"]) > 0:
                        model_id = data["data"][0]["id"]
                        logger.info(f"Auto-detected model: {model_id}")
                        return model_id
    except aiohttp.ClientError as e:
        logger.warning(f"Failed to auto-detect model from {base_url}: {e}")
    except (KeyError, IndexError, ValueError) as e:
        logger.warning(f"Failed to parse model response: {e}")
    return fallback


def detect_model_from_port(
    port: int = 8000,
    host: str = "localhost",
    fallback: str = "default",
    timeout: float = 5.0,
) -> str:
    """
    Convenience function to detect model from host:port.

    Args:
        port: vLLM server port
        host: vLLM server host
        fallback: Model name to return if detection fails
        timeout: Request timeout in seconds

    Returns:
        Detected model ID or fallback
    """
    base_url = f"http://{host}:{port}/v1"
    return detect_model_sync(base_url, fallback, timeout)


# Default context window if detection fails
DEFAULT_CONTEXT_WINDOW = 32768


def get_context_window_sync(
    base_url: str,
    fallback: int = DEFAULT_CONTEXT_WINDOW,
    timeout: float = 5.0,
) -> int:
    """
    Synchronously get model's context window from vLLM server.

    vLLM's /v1/models endpoint may include max_model_len in the response.
    If not available, falls back to looking up from settings.yaml or default.

    Args:
        base_url: Base URL of the vLLM server (e.g., "http://localhost:8000/v1")
        fallback: Context window to return if detection fails
        timeout: Request timeout in seconds

    Returns:
        Model's context window size in tokens
    """
    try:
        response = requests.get(f"{base_url}/models", timeout=timeout)
        response.raise_for_status()
        data = response.json()

        if data.get("data") and len(data["data"]) > 0:
            model_data = data["data"][0]

            # Try to get max_model_len from vLLM response
            # vLLM includes this in some versions
            if "max_model_len" in model_data:
                context_window = model_data["max_model_len"]
                logger.debug(f"Detected context window from vLLM: {context_window}")
                return context_window

            # Otherwise try to look up from settings.yaml
            model_id = model_data.get("id", "")
            context_window = _lookup_context_window_from_settings(model_id)
            if context_window:
                return context_window

    except requests.RequestException as e:
        logger.debug(f"Failed to get context window from {base_url}: {e}")
    except (KeyError, IndexError, ValueError) as e:
        logger.debug(f"Failed to parse context window response: {e}")

    logger.debug(f"Using fallback context window: {fallback}")
    return fallback


async def get_context_window_async(
    base_url: str,
    fallback: int = DEFAULT_CONTEXT_WINDOW,
    timeout: float = 5.0,
) -> int:
    """
    Asynchronously get model's context window from vLLM server.

    Args:
        base_url: Base URL of the vLLM server (e.g., "http://localhost:8000/v1")
        fallback: Context window to return if detection fails
        timeout: Request timeout in seconds

    Returns:
        Model's context window size in tokens
    """
    try:
        timeout_config = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.get(f"{base_url}/models") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("data") and len(data["data"]) > 0:
                        model_data = data["data"][0]

                        # Try to get max_model_len from vLLM response
                        if "max_model_len" in model_data:
                            context_window = model_data["max_model_len"]
                            logger.debug(f"Detected context window from vLLM: {context_window}")
                            return context_window

                        # Otherwise try to look up from settings.yaml
                        model_id = model_data.get("id", "")
                        context_window = _lookup_context_window_from_settings(model_id)
                        if context_window:
                            return context_window

    except aiohttp.ClientError as e:
        logger.debug(f"Failed to get context window from {base_url}: {e}")
    except (KeyError, IndexError, ValueError) as e:
        logger.debug(f"Failed to parse context window response: {e}")

    logger.debug(f"Using fallback context window: {fallback}")
    return fallback


def get_context_window_from_port(
    port: int = 8000,
    host: str = "localhost",
    fallback: int = DEFAULT_CONTEXT_WINDOW,
    timeout: float = 5.0,
) -> int:
    """
    Convenience function to get context window from host:port.

    Args:
        port: vLLM server port
        host: vLLM server host
        fallback: Context window to return if detection fails
        timeout: Request timeout in seconds

    Returns:
        Model's context window size in tokens
    """
    base_url = f"http://{host}:{port}/v1"
    return get_context_window_sync(base_url, fallback, timeout)


def _lookup_context_window_from_settings(model_id: str) -> Optional[int]:
    """
    Look up context window for a model from settings.yaml.

    Args:
        model_id: Model identifier to look up

    Returns:
        Context window if found, None otherwise
    """
    try:
        from treepo._research.config.settings import load_settings

        settings = load_settings()
        vllm_models = settings.get("vllm", {}).get("models", {})

        # Try exact match first
        for model_name, config in vllm_models.items():
            if model_name in model_id or model_id in str(config.get("path", "")):
                max_model_len = config.get("max_model_len")
                if max_model_len:
                    logger.debug(
                        f"Found context window for {model_id} from settings: {max_model_len}"
                    )
                    return max_model_len

    except Exception as e:
        logger.debug(f"Could not look up context window from settings: {e}")

    return None
