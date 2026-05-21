"""
Settings loader for YAML configuration.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from treepo._research.core.engines import normalize_engine_name, normalize_fallback_engine_name


# Default server URLs
DEFAULT_TASK_MODEL_URL = "http://localhost:8000/v1"
DEFAULT_GENRM_URL = "http://localhost:8001/v1"
DEFAULT_EMBEDDING_URL = "http://localhost:8003/v1"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

DEFAULT_INFERENCE_BACKEND: Dict[str, Any] = {
    "task_backend": "vllm",
    "genrm_backend": "vllm",
    "fallback_backend": "vllm",
    "routing_policy": "affinity_load_aware",
    "metrics_poll_seconds": 2.0,
    "sglang_venv_path": "/home/mlinegar/sglang-env",
    "vllm_venv_path": "/home/mlinegar/vllm-env",
}


def default_settings_path() -> Path:
    """Return default settings.yaml path within the repo."""
    return Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def load_settings(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load settings from a YAML file.

    Args:
        path: Optional path override. Defaults to repo config/settings.yaml.

    Returns:
        Parsed settings dict (empty if file missing).
    """
    settings_path = Path(path) if path else default_settings_path()
    if not settings_path.exists():
        return {}

    with open(settings_path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def get_task_model_url(settings: Optional[Dict[str, Any]] = None) -> str:
    """
    Get the task model (vLLM) server URL.

    Priority:
    1. TASK_MODEL_URL environment variable
    2. settings.yaml servers.task_model_url
    3. Default: http://localhost:8000/v1

    Args:
        settings: Pre-loaded settings dict, or None to load from file.

    Returns:
        Task model server URL.
    """
    # Environment variable takes precedence
    env_url = os.environ.get("TASK_MODEL_URL")
    if env_url:
        return env_url.rstrip("/")

    # Load settings if not provided
    if settings is None:
        settings = load_settings()

    # Check servers section
    servers = settings.get("servers", {})
    if servers.get("task_model_url"):
        return servers["task_model_url"].rstrip("/")

    return DEFAULT_TASK_MODEL_URL


def get_genrm_url(settings: Optional[Dict[str, Any]] = None) -> str:
    """
    Get the GenRM (generative reward model) server URL.

    Priority:
    1. GENRM_URL environment variable
    2. settings.yaml servers.genrm_url
    3. Default: http://localhost:8001/v1

    Args:
        settings: Pre-loaded settings dict, or None to load from file.

    Returns:
        GenRM server URL.
    """
    # Environment variable takes precedence
    env_url = os.environ.get("GENRM_URL")
    if env_url:
        return env_url.rstrip("/")

    # Load settings if not provided
    if settings is None:
        settings = load_settings()

    # Check servers section
    servers = settings.get("servers", {})
    if servers.get("genrm_url"):
        return servers["genrm_url"].rstrip("/")

    return DEFAULT_GENRM_URL


def get_embedding_url(settings: Optional[Dict[str, Any]] = None) -> str:
    """
    Get the embedding-model server URL.

    Priority:
    1. EMBEDDING_URL environment variable
    2. settings.yaml servers.embedding_url
    3. Default: http://localhost:8003/v1
    """
    env_url = os.environ.get("EMBEDDING_URL")
    if env_url:
        return env_url.rstrip("/")

    if settings is None:
        settings = load_settings()

    servers = settings.get("servers", {})
    if servers.get("embedding_url"):
        return servers["embedding_url"].rstrip("/")

    return DEFAULT_EMBEDDING_URL


def get_embedding_model(settings: Optional[Dict[str, Any]] = None) -> str:
    """
    Get the embedding-model id served by the embedding endpoint.

    Priority:
    1. EMBEDDING_MODEL environment variable
    2. settings.yaml servers.embedding_model
    3. settings.yaml chunking.adaptive.embedding_proxy.model (backward-compatible fallback)
    4. Default: Qwen/Qwen3-Embedding-8B
    """
    env_model = os.environ.get("EMBEDDING_MODEL")
    if env_model:
        return str(env_model).strip()

    if settings is None:
        settings = load_settings()

    servers = settings.get("servers", {}) if isinstance(settings, dict) else {}
    if isinstance(servers, dict) and servers.get("embedding_model"):
        return str(servers["embedding_model"]).strip()

    chunking = settings.get("chunking", {}) if isinstance(settings, dict) else {}
    adaptive = chunking.get("adaptive", {}) if isinstance(chunking, dict) else {}
    proxy = adaptive.get("embedding_proxy", {}) if isinstance(adaptive, dict) else {}
    if isinstance(proxy, dict) and proxy.get("model"):
        return str(proxy["model"]).strip()

    return DEFAULT_EMBEDDING_MODEL


def get_server_urls(settings: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Get server endpoints (and embedding model id) as a dict.

    Returns:
        Dict with 'task_model_url', 'genrm_url', 'embedding_url', and
        'embedding_model' keys.
    """
    if settings is None:
        settings = load_settings()

    return {
        "task_model_url": get_task_model_url(settings),
        "genrm_url": get_genrm_url(settings),
        "embedding_url": get_embedding_url(settings),
        "embedding_model": get_embedding_model(settings),
    }

def get_inference_backend_config(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return normalized inference backend settings.

    Priority:
    1. Environment variables (TT_*).
    2. settings.yaml inference.backend.
    3. Built-in defaults.
    """
    if settings is None:
        settings = load_settings()

    backend_cfg: Dict[str, Any] = dict(DEFAULT_INFERENCE_BACKEND)
    inference = settings.get("inference", {}) if isinstance(settings, dict) else {}
    if isinstance(inference, dict):
        raw_backend = inference.get("backend", {})
        if isinstance(raw_backend, dict):
            backend_cfg.update(raw_backend)

    env_task_backend = os.environ.get("TT_TASK_BACKEND")
    env_genrm_backend = os.environ.get("TT_GENRM_BACKEND")
    env_fallback_backend = os.environ.get("TT_FALLBACK_BACKEND")
    env_routing_policy = os.environ.get("TT_ROUTING_POLICY")
    env_metrics_poll = os.environ.get("TT_METRICS_POLL_SECONDS")
    env_sglang_venv = os.environ.get("TT_SGLANG_VENV_PATH")
    env_vllm_venv = os.environ.get("TT_VLLM_VENV_PATH")

    if env_task_backend:
        backend_cfg["task_backend"] = env_task_backend
    if env_genrm_backend:
        backend_cfg["genrm_backend"] = env_genrm_backend
    if env_fallback_backend:
        backend_cfg["fallback_backend"] = env_fallback_backend
    if env_routing_policy:
        backend_cfg["routing_policy"] = env_routing_policy
    if env_metrics_poll:
        backend_cfg["metrics_poll_seconds"] = env_metrics_poll
    if env_sglang_venv:
        backend_cfg["sglang_venv_path"] = env_sglang_venv
    if env_vllm_venv:
        backend_cfg["vllm_venv_path"] = env_vllm_venv

    routing_policy = str(backend_cfg.get("routing_policy", "affinity_load_aware") or "").strip().lower()
    if routing_policy not in {"round_robin", "document_affinity", "affinity_load_aware"}:
        routing_policy = "affinity_load_aware"

    try:
        metrics_poll_seconds = float(backend_cfg.get("metrics_poll_seconds", 2.0))
    except (TypeError, ValueError):
        metrics_poll_seconds = 2.0
    metrics_poll_seconds = max(0.25, metrics_poll_seconds)

    return {
        "task_backend": normalize_engine_name(backend_cfg.get("task_backend"), default="vllm"),
        "genrm_backend": normalize_engine_name(backend_cfg.get("genrm_backend"), default="vllm"),
        "fallback_backend": normalize_fallback_engine_name(
            backend_cfg.get("fallback_backend"),
            default="vllm",
        ),
        "routing_policy": routing_policy,
        "metrics_poll_seconds": metrics_poll_seconds,
        "sglang_venv_path": str(backend_cfg.get("sglang_venv_path") or DEFAULT_INFERENCE_BACKEND["sglang_venv_path"]),
        "vllm_venv_path": str(backend_cfg.get("vllm_venv_path") or DEFAULT_INFERENCE_BACKEND["vllm_venv_path"]),
    }


# Defaults
DEFAULT_TASK = "document_analysis"
DEFAULT_DATASET = "jsonl"  # Generic format, manifesto for manifesto-specific


def get_default_task(settings: Optional[Dict[str, Any]] = None) -> str:
    """
    Get the default task name.

    Priority:
    1. TASK environment variable
    2. settings.yaml tasks.default
    3. DEFAULT_TASK constant

    Args:
        settings: Pre-loaded settings dict, or None to load from file.

    Returns:
        Task name.
    """
    # Environment variable takes precedence
    env_task = os.environ.get("TASK")
    if env_task:
        return env_task

    # Load settings if not provided
    if settings is None:
        settings = load_settings()

    # Check tasks section
    tasks = settings.get("tasks", {})
    if tasks.get("default"):
        return tasks["default"]

    return DEFAULT_TASK


def get_default_dataset(settings: Optional[Dict[str, Any]] = None) -> str:
    """
    Get the default dataset name.

    Priority:
    1. DATASET environment variable
    2. settings.yaml datasets.default
    3. Default: jsonl
    """
    env_dataset = os.environ.get("DATASET")
    if env_dataset:
        return env_dataset

    if settings is None:
        settings = load_settings()

    datasets = settings.get("datasets", {})
    if datasets.get("default"):
        return datasets["default"]

    return DEFAULT_DATASET


def get_task_config(
    task_name: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get configuration for a specific task."""
    if settings is None:
        settings = load_settings()

    if task_name is None:
        task_name = get_default_task(settings)

    tasks = settings.get("tasks", {})
    return tasks.get(task_name, {})


def get_dataset_config(
    dataset_name: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get configuration for a specific dataset."""
    if settings is None:
        settings = load_settings()

    if dataset_name is None:
        dataset_name = get_default_dataset(settings)

    datasets = settings.get("datasets", {})
    return datasets.get(dataset_name, {})
