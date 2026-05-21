"""Config public exports with lazy loading for lightweight import safety."""

from __future__ import annotations

from importlib import import_module
from typing import Dict

_MODULE_EXPORTS = {
    "src.config.concurrency": (
        "ConcurrencyConfig",
        "DEFAULT_CONCURRENCY",
        "get_concurrency_config",
        "create_low_resource_config",
        "create_high_throughput_config",
    ),
    "src.config.settings": (
        "default_settings_path",
        "load_settings",
        "get_task_model_url",
        "get_genrm_url",
        "get_inference_backend_config",
        "get_server_urls",
        "get_default_task",
        "get_default_dataset",
        "get_task_config",
        "get_dataset_config",
        "DEFAULT_TASK_MODEL_URL",
        "DEFAULT_GENRM_URL",
        "DEFAULT_TASK",
        "DEFAULT_DATASET",
    ),
    "src.config.dspy_config": (
        "get_xml_adapter",
        "configure_dspy",
        "create_local_engine_lm",
        "create_local_engine_lm_with_manager",
        "create_vllm_lm",
    ),
    "src.config.local_inference": (
        "LocalInferenceConfig",
        "add_local_inference_args",
        "resolve_local_inference_config",
    ),
    "src.config.logging": (
        "setup_logging",
        "get_logger",
    ),
    "src.config.constants": (
        "LOG_TRUNCATE_LENGTH",
        "DEFAULT_TASK_MODEL_PORT",
        "DEFAULT_JUDGE_MODEL_PORT",
        "DEFAULT_SGLANG_PORT",
        "DEFAULT_MAX_TOKENS",
        "DEFAULT_CHUNK_TOKENS",
        "DEFAULT_MAX_INIT_PROMPT_TOKENS",
        "HEALTH_CHECK_TIMEOUT",
        "REQUEST_TIMEOUT",
        "BATCH_POLL_INTERVAL",
        "ERROR_THRESHOLD_HIGH",
        "ERROR_THRESHOLD_LOW",
        "BOOTSTRAP_TARGET_P_SUFF",
        "BOOTSTRAP_TARGET_P_MERGE",
        "BOOTSTRAP_TARGET_P_IDEM",
        "BOOTSTRAP_CONVERGENCE_THRESHOLD",
        "BOOTSTRAP_SAMPLE_RATE",
        "BOOTSTRAP_MAX_TRAINING_EXAMPLES",
        "DEFAULT_MAX_CONCURRENT_REQUESTS",
        "DEFAULT_MAX_CONCURRENT_DOCUMENTS",
        "AUDIT_MAX_WORKERS",
        "HIGH_THROUGHPUT_MAX_REQUESTS",
        "PRECACHE_MAX_WORKERS",
        "DEFAULT_BATCH_SIZE",
        "DEFAULT_TEMPERATURE",
        "DEFAULT_TOP_P",
        "DIVERSE_TEMPERATURES",
        "MIN_SUMMARY_WORDS",
        "MIN_TRAINING_EXAMPLES",
        "MIN_PREFERENCE_EXAMPLES",
        "DOCUMENT_LABEL_CONFIDENCE",
        "MAX_CHUNK_CHARS",
        "MIN_CHUNK_CHARS",
        "DEFAULT_MAX_DOC_CHARS",
        "EXTENDED_MAX_DOC_CHARS",
    ),
}

_NAME_TO_MODULE: Dict[str, str] = {
    name: module_name
    for module_name, names in _MODULE_EXPORTS.items()
    for name in names
}

__all__ = [
    "ConcurrencyConfig",
    "DEFAULT_CONCURRENCY",
    "get_concurrency_config",
    "create_low_resource_config",
    "create_high_throughput_config",
    "default_settings_path",
    "load_settings",
    "get_task_model_url",
    "get_genrm_url",
    "get_inference_backend_config",
    "get_server_urls",
    "get_default_task",
    "get_default_dataset",
    "get_task_config",
    "get_dataset_config",
    "DEFAULT_TASK_MODEL_URL",
    "DEFAULT_GENRM_URL",
    "DEFAULT_TASK",
    "DEFAULT_DATASET",
    "get_xml_adapter",
    "configure_dspy",
    "create_local_engine_lm",
    "create_local_engine_lm_with_manager",
    "create_vllm_lm",
    "LocalInferenceConfig",
    "add_local_inference_args",
    "resolve_local_inference_config",
    "setup_logging",
    "get_logger",
    "LOG_TRUNCATE_LENGTH",
    "DEFAULT_TASK_MODEL_PORT",
    "DEFAULT_JUDGE_MODEL_PORT",
    "DEFAULT_SGLANG_PORT",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_CHUNK_TOKENS",
    "DEFAULT_MAX_INIT_PROMPT_TOKENS",
    "HEALTH_CHECK_TIMEOUT",
    "REQUEST_TIMEOUT",
    "BATCH_POLL_INTERVAL",
    "ERROR_THRESHOLD_HIGH",
    "ERROR_THRESHOLD_LOW",
    "BOOTSTRAP_TARGET_P_SUFF",
    "BOOTSTRAP_TARGET_P_MERGE",
    "BOOTSTRAP_TARGET_P_IDEM",
    "BOOTSTRAP_CONVERGENCE_THRESHOLD",
    "BOOTSTRAP_SAMPLE_RATE",
    "BOOTSTRAP_MAX_TRAINING_EXAMPLES",
    "DEFAULT_MAX_CONCURRENT_REQUESTS",
    "DEFAULT_MAX_CONCURRENT_DOCUMENTS",
    "AUDIT_MAX_WORKERS",
    "HIGH_THROUGHPUT_MAX_REQUESTS",
    "PRECACHE_MAX_WORKERS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "DIVERSE_TEMPERATURES",
    "MIN_SUMMARY_WORDS",
    "MIN_TRAINING_EXAMPLES",
    "MIN_PREFERENCE_EXAMPLES",
    "DOCUMENT_LABEL_CONFIDENCE",
    "MAX_CHUNK_CHARS",
    "MIN_CHUNK_CHARS",
    "DEFAULT_MAX_DOC_CHARS",
    "EXTENDED_MAX_DOC_CHARS",
]


def __getattr__(name: str):
    module_name = _NAME_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
