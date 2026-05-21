"""
Batched Document Pipeline for High-Throughput Processing.

This module implements truly parallel processing of documents:
- Multiple documents processed concurrently
- All LLM requests pooled and batched
- Optimal vLLM GPU utilization

Example throughput comparison:
- Sequential: 1 doc/min × 100 docs = 100 minutes
- Batched (50 concurrent): ~5 minutes for same 100 docs

Usage:
    from treepo._research.pipelines.batched import BatchedDocPipeline

    pipeline = BatchedDocPipeline(config)
    results = await pipeline.process_batch_async(samples)

    # Or from sync code:
    results = pipeline.process_batch(samples)
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import Counter
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field as dataclass_field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

if TYPE_CHECKING:
    from treepo._research.core.conditional_memory import ConditionalMemory
    from treepo._research.core.semantic_memory import SemanticMemoryIndex

from treepo._research.core.batch_processor import (
    AsyncBatchLLMClient,
    BatchRequest,
    BatchStats,
    MultiServerBatchClient,
    parse_routing_policy,
)
from treepo._research.core.batch_transport import (
    DEFAULT_BATCH_MAX_CONCURRENT,
    DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_BATCH_ROUTING_POLICY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_TIMEOUT_SECONDS,
)
from treepo._research.core.batch_orchestrator import BatchTreeOrchestrator
from treepo._research.core.engines import LocalChatEndpoints
from treepo._research.core.strategy import (
    SummarizationStrategy,
    DSPyStrategy,
    BatchedStrategy,
    CallableStrategy,
)
from treepo._research.core.unified_runtime import (
    BatchTelemetry,
    WorkItem,
    get_named_plan_cache,
    normalize_runtime_mode,
    plan_work_batches,
)
from treepo._research.core.data_models import Node, Tree, leaf, node
from treepo._research.tree.builder import AsyncTreeBuilder, BuildConfig, BuildResult
from treepo._research.core.progress import (
    PipelineProgress,
    display_batch_summary,
)
from treepo._research.core.async_utils import cancel_tasks
from treepo._research.config.concurrency import ConcurrencyConfig, get_concurrency_config
from treepo._research.config import get_task_model_url, get_genrm_url
from treepo._research.core.documents import DocumentSample, DocumentResult
from treepo._research.core.engram_memory import EngramMemoryConfig
from treepo._research.core.engram_prompting import (
    clear_prompt_metadata_registry,
    engram_document_metadata,
    register_prompt_metadata_for_doc,
    wrap_score_prompt_with_engram_metadata,
)
from treepo._research.core.semantic_memory import SemanticMemoryConfig
from treepo._research.core.semantic_prompting import semantic_document_memory
from treepo._research.core.prompting import clean_summary_text, is_degenerate_summary_text
from treepo._research.tasks.prompting import PromptBuilders, default_merge_prompt, default_summarize_prompt
from treepo._research.preprocessing.chunker import chunk_for_ops
from treepo._research.unified_g_v1.core.specs import (
    UnifiedFGSpec,
    build_ctreepo_program_spec,
    build_llm_text_program_spec,
    build_mergeable_sketch_program_spec,
    build_semantic_embedding_program_spec,
    resolve_program_spec_alias,
)

logger = logging.getLogger(__name__)

_LLM_TEXT_PROGRAM_SPEC = build_llm_text_program_spec(tokenizer_or_adapter_id="cl100k_base")
_SEMANTIC_EMBEDDING_PROGRAM_SPEC = build_semantic_embedding_program_spec()
_CTREEPO_PROGRAM_SPEC = build_ctreepo_program_spec()
_MERGEABLE_SKETCH_PROGRAM_SPEC = build_mergeable_sketch_program_spec()

_BASE_REPRESENTATION_BACKENDS: Tuple[str, ...] = (
    str(_LLM_TEXT_PROGRAM_SPEC.program_family),
    str(_SEMANTIC_EMBEDDING_PROGRAM_SPEC.program_family),
    str(_CTREEPO_PROGRAM_SPEC.program_family),
    str(_MERGEABLE_SKETCH_PROGRAM_SPEC.program_family),
)
_ALL_REPRESENTATION_BACKENDS: Tuple[str, ...] = _BASE_REPRESENTATION_BACKENDS + ("ensemble",)
_REPRESENTATION_BACKEND_ALIASES: Dict[str, str] = {
    "llm": str(_LLM_TEXT_PROGRAM_SPEC.program_family),
    "oracle": str(_LLM_TEXT_PROGRAM_SPEC.program_family),
    "scorer": str(_LLM_TEXT_PROGRAM_SPEC.program_family),
    "score": str(_LLM_TEXT_PROGRAM_SPEC.program_family),
    "embedding": str(_SEMANTIC_EMBEDDING_PROGRAM_SPEC.program_family),
    "ctreepo": str(_CTREEPO_PROGRAM_SPEC.program_family),
    "neural_operator": str(_CTREEPO_PROGRAM_SPEC.program_family),
    "neural-operator": str(_CTREEPO_PROGRAM_SPEC.program_family),
    "operator": str(_CTREEPO_PROGRAM_SPEC.program_family),
    "mergeable_sketch": str(_MERGEABLE_SKETCH_PROGRAM_SPEC.program_family),
    "sketch": str(_MERGEABLE_SKETCH_PROGRAM_SPEC.program_family),
    "mergeable": str(_MERGEABLE_SKETCH_PROGRAM_SPEC.program_family),
    "mergeable_embedding_sketch": str(_MERGEABLE_SKETCH_PROGRAM_SPEC.program_family),
}


def _extract_host_and_ports(server_urls: List[str]) -> tuple[Optional[str], List[int]]:
    host: Optional[str] = None
    ports: List[int] = []
    for raw in server_urls:
        parsed = urlparse(str(raw))
        if parsed.hostname is None or parsed.port is None:
            continue
        if host is None:
            host = parsed.hostname
        if host != parsed.hostname:
            return None, []
        ports.append(int(parsed.port))
    return host, ports

def _extract_reference_score(sample: Any) -> Optional[float]:
    reference = getattr(sample, "reference_score", None)
    if reference is None:
        reference = getattr(sample, "score", None)
    return reference


def _extract_metadata(sample: Any) -> Dict[str, Any]:
    """Extract metadata from a sample object.

    Copies any existing metadata dict and adds any additional public attributes
    from the sample that look like metadata (excluding text content and known
    data fields).

    This is task-agnostic: it will capture any task-specific fields like
    party_name, country_code, etc. without hardcoding them.
    """
    metadata = dict(getattr(sample, "metadata", {}) or {})

    # Fields to exclude (text content and known data fields)
    exclude = {
        "text", "content", "doc_id", "id", "metadata",
        "reference_score", "score", "label",
    }

    # Copy additional public attributes that look like metadata
    for attr in dir(sample):
        # Skip private attributes, excluded fields, and methods
        if attr.startswith("_") or attr in exclude:
            continue
        if attr in metadata:
            continue

        try:
            value = getattr(sample, attr, None)
            # Only include non-callable, non-None values
            if value is not None and not callable(value):
                metadata[attr] = value
        except Exception:
            # Skip attributes that raise on access
            pass

    return metadata


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _semantic_meta_from_payload(doc_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = dict(metadata or {})
    return {
        "doc_id": str(doc_id or "").strip(),
        "party_id": _safe_int(meta.get("party_id")),
        "country_code": _safe_int(meta.get("country_code")),
        "party_family": _safe_int(meta.get("party_family")),
        "year": _safe_int(meta.get("year")),
        "date_code": _safe_int(meta.get("date_code")),
        "rile": _safe_float(meta.get("rile")),
        "delta_rile": _safe_float(meta.get("delta_rile")),
        "provenance": {
            "source": str(meta.get("source", "") or meta.get("dataset", "") or ""),
        },
    }


def _normalize_representation_backend(raw: Any) -> Optional[str]:
    value = str(raw or "").strip().lower()
    if not value:
        return None
    if value == "auto":
        return "auto"
    value = _REPRESENTATION_BACKEND_ALIASES.get(value, value)
    if value in _ALL_REPRESENTATION_BACKENDS:
        return value
    return None


def _parse_representation_backend_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        candidates = [part.strip() for part in re.split(r"[,\s;+|]+", raw) if part.strip()]
    elif isinstance(raw, (list, tuple, set)):
        candidates = list(raw)
    else:
        candidates = [raw]
    parsed: List[str] = []
    for candidate in candidates:
        normalized = _normalize_representation_backend(candidate)
        if normalized is None:
            continue
        if normalized not in parsed:
            parsed.append(normalized)
    return parsed


def _normalize_representation_weights(raw: Any) -> Dict[str, float]:
    if raw is None:
        return {}
    parsed: Dict[str, Any]
    if isinstance(raw, str):
        parsed = {}
        for token in [part.strip() for part in raw.split(",") if part.strip()]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            parsed[str(key).strip()] = value
    elif isinstance(raw, dict):
        parsed = dict(raw)
    else:
        return {}
    normalized: Dict[str, float] = {}
    for key, value in parsed.items():
        backend = _normalize_representation_backend(key)
        if backend is None or backend == "auto":
            continue
        weight = _safe_float(value)
        if weight is None:
            continue
        normalized[backend] = max(0.0, float(weight))
    return normalized


def _clip_rile_score(value: Any) -> Optional[float]:
    score = _safe_float(value)
    if score is None:
        return None
    return max(-100.0, min(100.0, float(score)))


def _clip01(value: Any) -> Optional[float]:
    scalar = _safe_float(value)
    if scalar is None:
        return None
    return max(0.0, min(1.0, float(scalar)))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BatchedPipelineConfig:
    """Configuration for batched pipeline."""

    # vLLM server settings - can be single URL or list of URLs for load balancing
    # Defaults are loaded from config/settings.yaml or environment variables
    task_model_endpoints: Optional[LocalChatEndpoints] = None
    task_model_url: str = dataclass_field(default_factory=get_task_model_url)
    task_model_urls: Optional[List[str]] = None  # Multiple servers for load balancing
    # Optional callback(base_url)->bool used for automatic server recovery.
    task_model_recovery_callback: Optional[Callable[[str], bool]] = None
    task_model_recovery_cooldown_seconds: float = 120.0
    routing_policy: str = DEFAULT_BATCH_ROUTING_POLICY
    metrics_poll_seconds: Optional[float] = None

    # Batching settings
    max_concurrent_requests: int = DEFAULT_BATCH_MAX_CONCURRENT    # Max concurrent HTTP requests
    batch_size: int = DEFAULT_BATCH_SIZE  # Requests per batch (independent from max_concurrent)
    max_concurrent_documents: int = 30    # Max documents in parallel (increased from 20)
    batch_timeout: float = DEFAULT_BATCH_TIMEOUT_SECONDS  # Max wait to fill batch
    request_timeout_seconds: Optional[float] = None
    await_response_timeout_seconds: Optional[float] = None

    # Tree building
    # Increased from 2000 to 4000 to reduce chunk count and tree depth
    max_chunk_chars: int = 4000
    max_chunk_tokens: Optional[int] = None
    chunk_token_encoding: str = "cl100k_base"
    max_tokens_summary: int = 500
    max_tokens_score: int = 200
    # Degenerate-summary safeguards for manual/guarded runs.
    fail_on_degenerate_summary: bool = False
    max_degenerate_leaf_fallbacks: int = 0
    max_degenerate_merge_fallbacks: int = 0

    # Concurrency configuration (prevents thread explosion)
    concurrency: ConcurrencyConfig = dataclass_field(default_factory=get_concurrency_config)

    # Processing options
    run_baseline: bool = True
    runtime_mode: str = "legacy"

    # Progress reporting
    show_progress: bool = True
    call_trace_sink: Optional[Callable[[Mapping[str, Any]], None]] = None

    # Task configuration (to be supplied by task plugins)
    rubric: str = ""
    task_context: str = ""
    prompt_builders: Optional[PromptBuilders] = None
    score_parser: Optional[Callable[[str], Optional[float]]] = None
    # Optional Engram-style static memory injection into summarize/merge prompts.
    engram_memory: EngramMemoryConfig = dataclass_field(default_factory=EngramMemoryConfig)
    # Optional semantic memory retrieval layer (multilingual embedding neighbors).
    semantic_memory: SemanticMemoryConfig = dataclass_field(default_factory=SemanticMemoryConfig)
    # Optional shared semantic memory index instance.
    semantic_memory_index: Optional["SemanticMemoryIndex"] = None
    # Optional shared ConditionalMemory store for Engram extraction caching.
    conditional_memory: Optional["ConditionalMemory"] = None
    # Optional CTreePO model for fast sketch-based scoring (loaded from checkpoint).
    ctreepo_model_path: Optional[str] = None
    # Optional strictly-mergeable embedding sketch model (MergeableEmbeddingSketch checkpoint).
    mergeable_sketch_model_path: Optional[str] = None

    # Canonical program-family routing for score selection.
    # Legacy representation_backends flags are accepted only as parser aliases.
    program_families: Optional[List[str]] = None
    primary_program_family: str = "auto"
    program_weights: Optional[Dict[str, float]] = None

    # Legacy aliases preserved at the config boundary only.
    representation_backends: Optional[List[str]] = None
    primary_representation_backend: str = "auto"
    representation_weights: Optional[Dict[str, float]] = None
    fallback_to_available_backend: bool = True
    # If no representation backend yields a score, assign this default score
    # instead of leaving estimated_score unset.
    missing_score_default: Optional[float] = 0.0
    # Hybrid mode: treat LLM score as oracle seed and blend with embedding/operator backends.
    hybrid_oracle_seeded_ensemble: bool = False
    hybrid_seed_llm_min_weight: float = 0.20
    hybrid_seed_llm_max_weight: float = 0.55
    hybrid_operator_boost: float = 1.40

    # Allow disabling LLM summarize/merge path while keeping a shared tree pipeline.
    llm_text_path_enabled: bool = True

    # Unified tree architecture: shared windows and dual representations.
    # When enabled, both LLM summarisation and CTreePO sketch scoring operate on
    # the same embedding-driven tree topology.
    unified_tree: bool = False
    # Use adaptive (coarse-to-fine) windowing instead of fixed uniform windows.
    adaptive_windows: bool = False
    # Feed oracle audit scores back to improve window boundaries.
    oracle_feedback_to_chunks: bool = False
    # Optional MIL proxy model path for window-importance scoring (Phase 5).
    mil_proxy_model_path: Optional[str] = None
    # Optional document-artifact cache root (per-run path), used for tree dumps.
    cache_artifacts_dir: Optional[Path] = None
    # Optional split label used when writing cached artifacts ("train"/"val"/"test").
    cache_artifacts_split: Optional[str] = None
    # Persist full tree structures to disk during document processing.
    cache_full_trees: bool = False

    def __post_init__(self):
        if self.task_model_endpoints is not None:
            self.task_model_url = self.task_model_endpoints.primary_base_url
            self.task_model_urls = self.task_model_endpoints.pipeline_base_urls

        self.routing_policy = parse_routing_policy(self.routing_policy).value
        self.runtime_mode = normalize_runtime_mode(self.runtime_mode)
        if self.metrics_poll_seconds is None:
            try:
                from treepo._research.config.settings import load_settings, get_inference_backend_config

                cfg = get_inference_backend_config(load_settings())
                self.metrics_poll_seconds = float(cfg.get("metrics_poll_seconds", 0.0))
            except Exception:
                self.metrics_poll_seconds = 0.0
        try:
            self.metrics_poll_seconds = float(self.metrics_poll_seconds)
        except (TypeError, ValueError):
            self.metrics_poll_seconds = 0.0
        self.metrics_poll_seconds = max(0.0, float(self.metrics_poll_seconds))

        try:
            request_timeout = (
                DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS
                if self.request_timeout_seconds is None
                else float(self.request_timeout_seconds)
            )
        except (TypeError, ValueError):
            request_timeout = DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS
        self.request_timeout_seconds = max(1.0, request_timeout)

        try:
            await_timeout = (
                600.0
                if self.await_response_timeout_seconds is None
                else float(self.await_response_timeout_seconds)
            )
        except (TypeError, ValueError):
            await_timeout = 600.0
        self.await_response_timeout_seconds = max(
            float(self.request_timeout_seconds) + 1.0,
            await_timeout,
        )
        if self.missing_score_default is not None:
            parsed_missing_default = _safe_float(self.missing_score_default)
            self.missing_score_default = (
                None if parsed_missing_default is None else float(parsed_missing_default)
            )
        if self.prompt_builders is None:
            try:
                from treepo._research.config.settings import load_settings, get_default_task, get_task_config
                from treepo._research.tasks import get_task

                settings = load_settings()
                task_name = get_default_task(settings)
                task = get_task(task_name, **get_task_config(task_name, settings))

                if not self.rubric:
                    self.rubric = task.create_rubric()
                if not self.task_context:
                    self.task_context = task.get_task_context()

                self.prompt_builders = task.create_prompt_builders()
                if self.score_parser is None:
                    self.score_parser = task.parse_score
            except Exception:
                self.prompt_builders = PromptBuilders(
                    summarize=default_summarize_prompt,
                    merge=default_merge_prompt,
                    score=None,
                    audit=None,
                )

        # Allow semantic-memory defaults to come from settings when not explicitly enabled.
        try:
            from treepo._research.config.settings import load_settings

            settings = load_settings()
            sem_cfg = settings.get("semantic_memory", {}) if isinstance(settings, dict) else {}
            if isinstance(sem_cfg, dict) and sem_cfg:
                current_sem = self.semantic_memory
                if not bool(current_sem.enabled) and bool(sem_cfg.get("enabled", False)):
                    self.semantic_memory = SemanticMemoryConfig(
                        enabled=bool(sem_cfg.get("enabled", current_sem.enabled)),
                        index_dir=Path(str(sem_cfg.get("index_dir", current_sem.index_dir))),
                        top_k=int(sem_cfg.get("top_k", current_sem.top_k) or current_sem.top_k),
                        lambda_year=float(sem_cfg.get("lambda_year", current_sem.lambda_year) or current_sem.lambda_year),
                        scope_bonus_party_country=float(
                            sem_cfg.get("scope_bonus_party_country", current_sem.scope_bonus_party_country)
                            or current_sem.scope_bonus_party_country
                        ),
                        scope_bonus_family_country=float(
                            sem_cfg.get("scope_bonus_family_country", current_sem.scope_bonus_family_country)
                            or current_sem.scope_bonus_family_country
                        ),
                        index_granularity=str(sem_cfg.get("index_granularity", current_sem.index_granularity) or current_sem.index_granularity),
                        max_windows=int(sem_cfg.get("max_windows", current_sem.max_windows) or 0),
                        update_policy=str(sem_cfg.get("update_policy", current_sem.update_policy) or current_sem.update_policy),
                        inject_prompts=bool(sem_cfg.get("inject_prompts", current_sem.inject_prompts)),
                        model_features=bool(sem_cfg.get("model_features", current_sem.model_features)),
                        temporal_mode=bool(sem_cfg.get("temporal_mode", current_sem.temporal_mode)),
                        max_chunk_snippets_per_neighbor=int(
                            sem_cfg.get(
                                "max_chunk_snippets_per_neighbor",
                                current_sem.max_chunk_snippets_per_neighbor,
                            )
                            or current_sem.max_chunk_snippets_per_neighbor
                        ),
                        max_snippet_chars=int(sem_cfg.get("max_snippet_chars", current_sem.max_snippet_chars) or current_sem.max_snippet_chars),
                    )
        except Exception:
            pass

        # Wrap prompt builders with Engram-style static memory injection (optional).
        if getattr(self.engram_memory, "enabled", False) and self.prompt_builders is not None:
            try:
                from treepo._research.core.engram_prompting import (
                    wrap_merge_prompt_with_engram_memory,
                    wrap_summarize_prompt_with_engram_memory,
                )

                self.prompt_builders = PromptBuilders(
                    summarize=wrap_summarize_prompt_with_engram_memory(
                        self.prompt_builders.summarize,
                        self.engram_memory,
                        memory=self.conditional_memory,
                    ),
                    merge=wrap_merge_prompt_with_engram_memory(
                        self.prompt_builders.merge,
                        self.engram_memory,
                        memory=self.conditional_memory,
                    ),
                    score=self.prompt_builders.score,
                    audit=self.prompt_builders.audit,
                )
            except Exception:
                # Fail open: memory injection is optional and should not block runs.
                pass

        # Wrap prompt builders with semantic-memory injection (optional).
        if (
            getattr(self.semantic_memory, "enabled", False)
            and bool(getattr(self.semantic_memory, "inject_prompts", True))
            and self.prompt_builders is not None
        ):
            try:
                from treepo._research.core.semantic_prompting import (
                    wrap_merge_prompt_with_semantic_memory,
                    wrap_summarize_prompt_with_semantic_memory,
                )

                self.prompt_builders = PromptBuilders(
                    summarize=wrap_summarize_prompt_with_semantic_memory(
                        self.prompt_builders.summarize,
                    ),
                    merge=wrap_merge_prompt_with_semantic_memory(
                        self.prompt_builders.merge,
                    ),
                    score=self.prompt_builders.score,
                    audit=self.prompt_builders.audit,
                )
            except Exception:
                pass

        # Always allow score prompts to consume safe per-document metadata when
        # available (via contextvar or doc-id registry).
        if self.prompt_builders is not None and self.prompt_builders.score is not None:
            try:
                self.prompt_builders = PromptBuilders(
                    summarize=self.prompt_builders.summarize,
                    merge=self.prompt_builders.merge,
                    score=wrap_score_prompt_with_engram_metadata(self.prompt_builders.score),
                    audit=self.prompt_builders.audit,
                )
            except Exception:
                pass

        # Auto-enable optional sketch models from settings when not explicitly configured.
        try:
            from treepo._research.config.settings import load_settings

            settings = load_settings()
            if not self.ctreepo_model_path:
                ctreepo_cfg = settings.get("ctreepo", {}) if isinstance(settings, dict) else {}
                if (
                    isinstance(ctreepo_cfg, dict)
                    and ctreepo_cfg.get("enabled")
                    and ctreepo_cfg.get("model_path")
                ):
                    self.ctreepo_model_path = str(ctreepo_cfg.get("model_path"))
            if not self.mergeable_sketch_model_path:
                sketch_cfg = settings.get("mergeable_sketch", {}) if isinstance(settings, dict) else {}
                if (
                    isinstance(sketch_cfg, dict)
                    and sketch_cfg.get("enabled")
                    and sketch_cfg.get("model_path")
                ):
                    self.mergeable_sketch_model_path = str(sketch_cfg.get("model_path"))
            rep_cfg = settings.get("representation_pipeline", {}) if isinstance(settings, dict) else {}
            if isinstance(rep_cfg, dict):
                if self.program_families is None and self.representation_backends is None:
                    parsed = _parse_representation_backend_list(rep_cfg.get("backends"))
                    if parsed:
                        self.program_families = parsed
                if (
                    str(self.primary_program_family or self.primary_representation_backend or "").strip().lower() in {"", "auto"}
                    and rep_cfg.get("primary_backend") is not None
                ):
                    self.primary_program_family = str(rep_cfg.get("primary_backend") or "auto")
                if self.program_weights is None and self.representation_weights is None and isinstance(rep_cfg.get("weights"), dict):
                    self.program_weights = dict(rep_cfg.get("weights"))
                if rep_cfg.get("fallback_to_available_backend") is not None:
                    self.fallback_to_available_backend = bool(rep_cfg.get("fallback_to_available_backend"))
                if (
                    rep_cfg.get("hybrid_oracle_seeded_ensemble") is not None
                    and bool(self.hybrid_oracle_seeded_ensemble) is False
                ):
                    self.hybrid_oracle_seeded_ensemble = bool(rep_cfg.get("hybrid_oracle_seeded_ensemble"))
                if (
                    rep_cfg.get("hybrid_seed_llm_min_weight") is not None
                    and abs(float(self.hybrid_seed_llm_min_weight) - 0.20) < 1e-12
                ):
                    self.hybrid_seed_llm_min_weight = float(rep_cfg.get("hybrid_seed_llm_min_weight"))
                if (
                    rep_cfg.get("hybrid_seed_llm_max_weight") is not None
                    and abs(float(self.hybrid_seed_llm_max_weight) - 0.55) < 1e-12
                ):
                    self.hybrid_seed_llm_max_weight = float(rep_cfg.get("hybrid_seed_llm_max_weight"))
                if (
                    rep_cfg.get("hybrid_operator_boost") is not None
                    and abs(float(self.hybrid_operator_boost) - 1.40) < 1e-12
                ):
                    self.hybrid_operator_boost = float(rep_cfg.get("hybrid_operator_boost"))
                if rep_cfg.get("llm_text_path_enabled") is not None:
                    self.llm_text_path_enabled = bool(rep_cfg.get("llm_text_path_enabled"))
            # Auto-enable unified tree from settings.
            if isinstance(settings, dict):
                ctreepo_cfg = settings.get("ctreepo", {})
                if isinstance(ctreepo_cfg, dict):
                    if not self.unified_tree and ctreepo_cfg.get("unified_tree"):
                        self.unified_tree = True
                    if not self.adaptive_windows and ctreepo_cfg.get("adaptive_windows"):
                        self.adaptive_windows = True
                    if not self.oracle_feedback_to_chunks and ctreepo_cfg.get("oracle_feedback_to_chunks"):
                        self.oracle_feedback_to_chunks = True
                    if not self.mil_proxy_model_path and ctreepo_cfg.get("mil_proxy_model_path"):
                        self.mil_proxy_model_path = str(ctreepo_cfg["mil_proxy_model_path"])
        except Exception:
            pass

        parsed_backends = _parse_representation_backend_list(
            self.program_families if self.program_families is not None else self.representation_backends
        )
        if not parsed_backends:
            parsed_backends = [str(_LLM_TEXT_PROGRAM_SPEC.program_family)]
        self.program_families = parsed_backends
        self.representation_backends = parsed_backends

        primary_backend = _normalize_representation_backend(
            self.primary_program_family
            if str(self.primary_program_family or "").strip()
            else self.primary_representation_backend
        )
        self.primary_program_family = primary_backend or "auto"
        self.primary_representation_backend = self.primary_program_family

        normalized_weights = _normalize_representation_weights(
            self.program_weights if self.program_weights is not None else self.representation_weights
        )
        self.program_weights = normalized_weights
        self.representation_weights = normalized_weights
        self.fallback_to_available_backend = bool(self.fallback_to_available_backend)
        self.hybrid_oracle_seeded_ensemble = bool(self.hybrid_oracle_seeded_ensemble)
        self.hybrid_seed_llm_min_weight = _clip01(self.hybrid_seed_llm_min_weight) or 0.20
        self.hybrid_seed_llm_max_weight = _clip01(self.hybrid_seed_llm_max_weight) or 0.55
        if self.hybrid_seed_llm_min_weight > self.hybrid_seed_llm_max_weight:
            self.hybrid_seed_llm_min_weight, self.hybrid_seed_llm_max_weight = (
                self.hybrid_seed_llm_max_weight,
                self.hybrid_seed_llm_min_weight,
            )
        self.hybrid_operator_boost = max(0.0, float(self.hybrid_operator_boost))
        self.llm_text_path_enabled = bool(self.llm_text_path_enabled)
        self.fail_on_degenerate_summary = bool(self.fail_on_degenerate_summary)
        self.max_degenerate_leaf_fallbacks = max(0, int(self.max_degenerate_leaf_fallbacks or 0))
        self.max_degenerate_merge_fallbacks = max(0, int(self.max_degenerate_merge_fallbacks or 0))
        self.cache_full_trees = bool(self.cache_full_trees)
        cache_dir_rendered = str(self.cache_artifacts_dir or "").strip()
        self.cache_artifacts_dir = Path(cache_dir_rendered) if cache_dir_rendered else None
        cache_split_rendered = str(self.cache_artifacts_split or "").strip().lower()
        self.cache_artifacts_split = cache_split_rendered or None
    # DSPy module support (for training/optimization mode)
    # When set, uses DSPy modules instead of raw prompts
    use_dspy_modules: bool = False


# =============================================================================
# Batched Pipeline
# =============================================================================

class BatchedDocPipeline:
    """
    High-throughput batched pipeline for document processing.

    Processes multiple documents concurrently, pooling all LLM requests
    for optimal GPU utilization.

    For DSPy optimization, compose tasks from core building blocks:
        from treepo._research.tasks import ScoringTask
        from treepo._research.core.scorers import ScaleScorer
        from treepo._research.core.summarization import GenericSummarizer, GenericMerger

        task = ScoringTask(
            name="my_task",
            scale=MY_SCALE,
            rubric="...",
            task_context="...",
            predictor_factory=lambda: ScaleScorer(MySignature),
        )
        strategy = DSPyStrategy(
            leaf_module=GenericSummarizer(),
            merge_module=GenericMerger(),
        )
        pipeline = BatchedDocPipeline(config=config)
        # Use process_batch_with_strategy for training
        results = await pipeline.process_batch_with_strategy([sample], strategy)
    """

    def __init__(
        self,
        config: Optional[BatchedPipelineConfig] = None,
    ):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or BatchedPipelineConfig()
        self._results: List[DocumentResult] = []
        self._last_stats: Optional[BatchStats] = None
        self._last_diagnostics: Optional[Dict[str, Any]] = None
        self._semantic_runtime_stats: Dict[str, Any] = {
            "retrieval_calls": 0,
            "retrieval_ms_total": 0.0,
            "writes_doc": 0,
            "writes_chunk": 0,
            "neighbors_total": 0,
        }
        self._semantic_written_doc_ids: set[str] = set()
        self._semantic_payload_by_doc_id: Dict[str, Dict[str, Any]] = {}

    def _reset_semantic_runtime_stats(self) -> None:
        self._semantic_runtime_stats = {
            "retrieval_calls": 0,
            "retrieval_ms_total": 0.0,
            "writes_doc": 0,
            "writes_chunk": 0,
            "neighbors_total": 0,
        }
        self._semantic_written_doc_ids = set()
        self._semantic_payload_by_doc_id = {}

    @staticmethod
    def _cache_safe_component(raw: Any, *, fallback: str) -> str:
        rendered = str(raw or "").strip()
        if not rendered:
            return fallback
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", rendered).strip("._-")
        return cleaned or fallback

    @classmethod
    def _json_safe_value(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): cls._json_safe_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe_value(v) for v in value]
        if isinstance(value, set):
            try:
                ordered = sorted(value, key=lambda item: repr(item))
            except Exception:
                ordered = list(value)
            return [cls._json_safe_value(v) for v in ordered]
        return repr(value)

    def _persist_tree_artifact(
        self,
        *,
        tree: Tree,
        doc_id: str,
        result_idx: int,
    ) -> Optional[str]:
        cache_root = self.config.cache_artifacts_dir
        cache_split = self.config.cache_artifacts_split
        if not self.config.cache_full_trees or cache_root is None or not cache_split:
            return None
        safe_split = self._cache_safe_component(cache_split, fallback="unknown")
        safe_doc_id = self._cache_safe_component(doc_id, fallback=f"doc_{result_idx}")
        digest = hashlib.sha1(
            f"{doc_id}|{result_idx}".encode("utf-8", errors="ignore")
        ).hexdigest()[:10]
        tree_dir = Path(cache_root) / safe_split / "trees"
        tree_dir.mkdir(parents=True, exist_ok=True)
        tree_path = tree_dir / f"{safe_doc_id}_{digest}.tree.json"
        try:
            payload = self._json_safe_value(tree.to_dict())
            tmp_path = tree_path.with_suffix(f"{tree_path.suffix}.tmp")
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            tmp_path.replace(tree_path)
            return str(tree_path)
        except Exception as exc:
            logger.warning("Failed to cache tree for %s: %s", doc_id or f"doc_{result_idx}", exc)
            return None

    def _ensure_semantic_components(self) -> bool:
        if not bool(getattr(self.config.semantic_memory, "enabled", False)):
            return False
        if hasattr(self, "_semantic_index") and hasattr(self, "_semantic_doc_embedder"):
            return True

        from treepo._research.config.settings import load_settings, get_embedding_model, get_embedding_url
        from treepo._research.embeddings.document_embedder import DocumentEmbedder, DocumentEmbeddingConfig
        from treepo._research.training.embedding_proxy import VLLMEmbeddingClient
        from treepo._research.core.semantic_memory import SemanticMemoryIndex

        sem_cfg = self.config.semantic_memory
        if self.config.semantic_memory_index is not None:
            self._semantic_index = self.config.semantic_memory_index
        else:
            self._semantic_index = SemanticMemoryIndex(sem_cfg)
            self.config.semantic_memory_index = self._semantic_index

        settings = load_settings()
        api_base = get_embedding_url(settings).rstrip("/")
        emb_model = get_embedding_model(settings)
        self._semantic_emb_client = VLLMEmbeddingClient(
            api_base=api_base,
            model=emb_model,
            timeout_seconds=60.0,
            batch_size=32,
            memory=self.config.conditional_memory,
        )
        self._semantic_doc_embedder = DocumentEmbedder(
            embedding_client=self._semantic_emb_client,
            config=DocumentEmbeddingConfig(
                window_chars=6000,
                overlap_chars=0,
                max_windows=int(getattr(sem_cfg, "max_windows", 0) or 0),
                l2_normalize=True,
                embed_metadata=True,
                meta_weight=0.25,
            ),
        )
        return True

    def _semantic_payload_for_document(
        self,
        *,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not self._ensure_semantic_components():
            return None
        from treepo._research.core.doc_metadata import DocMetadata

        text_raw = str(text or "")
        if not text_raw.strip():
            return None

        sem_meta = _semantic_meta_from_payload(doc_id, metadata)
        doc_meta = DocMetadata(
            doc_id=str(doc_id or ""),
            source=str((metadata or {}).get("source", "") or (metadata or {}).get("dataset", "") or ""),
            country=str((metadata or {}).get("country_name", "") or (metadata or {}).get("country", "") or "") or None,
            party=str((metadata or {}).get("party_name", "") or (metadata or {}).get("party", "") or "") or None,
            party_abbrev=str((metadata or {}).get("party_abbrev", "") or "") or None,
            year=_safe_int((metadata or {}).get("year")),
            date_code=_safe_int((metadata or {}).get("date_code")),
            election_date=str((metadata or {}).get("election_date", "") or "") or None,
            party_family=_safe_int((metadata or {}).get("party_family")),
            rile=_safe_float((metadata or {}).get("rile")),
            extra={},
        )

        started = time.time()
        embedded = self._semantic_doc_embedder.embed_document(text_raw, meta=doc_meta)
        query_vec = embedded.combined_vector if embedded.combined_vector is not None else embedded.text_vector
        if query_vec is None:
            return None

        neighbors = self._semantic_index.query_with_snippets(
            query_vector=query_vec,
            query_meta=sem_meta,
            top_k=int(getattr(self.config.semantic_memory, "top_k", 5) or 5),
            exclude_doc_id=str(doc_id or ""),
        )
        retrieval_ms = max(0.0, (time.time() - started) * 1000.0)
        features = self._semantic_index.retrieval_features(neighbors)
        payload = {
            "neighbors": [asdict(n) for n in neighbors],
            "features": [float(v) for v in features.reshape(-1)],
            "query_vector": query_vec.astype(np.float32, copy=False),
            "retrieval_ms": float(retrieval_ms),
            "neighbor_count": int(len(neighbors)),
        }
        self._semantic_runtime_stats["retrieval_calls"] = int(self._semantic_runtime_stats["retrieval_calls"]) + 1
        self._semantic_runtime_stats["retrieval_ms_total"] = float(self._semantic_runtime_stats["retrieval_ms_total"]) + float(
            retrieval_ms
        )
        self._semantic_runtime_stats["neighbors_total"] = int(self._semantic_runtime_stats["neighbors_total"]) + int(
            len(neighbors)
        )
        return payload

    def _semantic_features_from_query_vector(
        self,
        *,
        query_vector: np.ndarray,
        doc_id: str,
        metadata: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        if not self._ensure_semantic_components():
            return np.zeros((6,), dtype=np.float32)
        sem_meta = _semantic_meta_from_payload(doc_id, metadata)
        neighbors = self._semantic_index.query(
            query_vector=query_vector,
            query_meta=sem_meta,
            top_k=int(getattr(self.config.semantic_memory, "top_k", 5) or 5),
            exclude_doc_id=str(doc_id or ""),
        )
        return self._semantic_index.retrieval_features(neighbors)

    def _semantic_write_after_score(
        self,
        *,
        result: DocumentResult,
        query_vector: Optional[np.ndarray] = None,
    ) -> None:
        if not self._ensure_semantic_components():
            return
        if str(getattr(self.config.semantic_memory, "update_policy", "post_score")).strip().lower() != "post_score":
            return
        if result.error:
            return
        if result.doc_id in self._semantic_written_doc_ids:
            return

        write_score = result.reference_score if result.reference_score is not None else result.estimated_score
        if write_score is None:
            return

        sem_meta = _semantic_meta_from_payload(result.doc_id, result.metadata)
        sem_meta["rile"] = float(write_score)
        if query_vector is None:
            payload = self._semantic_payload_for_document(
                text=str(result.original_content or ""),
                doc_id=str(result.doc_id or ""),
                metadata=result.metadata,
            )
            if payload is None:
                return
            query_vector = np.asarray(payload.get("query_vector"), dtype=np.float32)

        doc_entry = self._semantic_index.add_document(
            doc_id=str(result.doc_id or ""),
            vector=query_vector,
            metadata=sem_meta,
        )
        result.metadata["semantic_memory_doc_written"] = bool(doc_entry is not None)
        if doc_entry is not None:
            self._semantic_runtime_stats["writes_doc"] = int(self._semantic_runtime_stats["writes_doc"]) + 1

        granularity = str(getattr(self.config.semantic_memory, "index_granularity", "doc_chunk") or "doc_chunk").strip().lower()
        written_chunks = 0
        if "chunk" in granularity:
            chunk_texts = [str(c or "") for c in (result.chunks or []) if str(c or "").strip()]
            if chunk_texts:
                chunk_vectors = self._semantic_emb_client.embed_texts(chunk_texts)
                written_chunks = self._semantic_index.add_chunks(
                    doc_id=str(result.doc_id or ""),
                    vectors=chunk_vectors,
                    chunk_texts=chunk_texts,
                    metadata=sem_meta,
                )
                self._semantic_runtime_stats["writes_chunk"] = int(self._semantic_runtime_stats["writes_chunk"]) + int(
                    written_chunks
                )
        result.metadata["semantic_memory_chunks_written"] = int(written_chunks)
        result.metadata["semantic_memory_write_policy"] = "post_score"
        result.metadata["semantic_memory_runtime_writes_doc"] = int(self._semantic_runtime_stats.get("writes_doc", 0) or 0)
        result.metadata["semantic_memory_runtime_writes_chunk"] = int(
            self._semantic_runtime_stats.get("writes_chunk", 0) or 0
        )

        self._semantic_written_doc_ids.add(str(result.doc_id or ""))
        report = self._semantic_index.report()
        result.metadata["semantic_memory_index_doc_entries"] = int(report.get("doc_entries", 0))
        result.metadata["semantic_memory_index_chunk_entries"] = int(report.get("chunk_entries", 0))

    def _attach_semantic_diagnostics(self) -> None:
        try:
            report = self._semantic_index.report() if hasattr(self, "_semantic_index") else {}
        except Exception:
            report = {}
        sem_diag = dict(self._semantic_runtime_stats)
        sem_diag.update(
            {
                "retrieval_ms_avg": (
                    float(sem_diag.get("retrieval_ms_total", 0.0))
                    / max(1, int(sem_diag.get("retrieval_calls", 0)))
                ),
                "index_doc_entries": int(report.get("doc_entries", 0) or 0),
                "index_chunk_entries": int(report.get("chunk_entries", 0) or 0),
                "index_entries_total": int(report.get("entries_total", 0) or 0),
            }
        )
        if self._last_diagnostics is None:
            self._last_diagnostics = {}
        if not isinstance(self._last_diagnostics, dict):
            self._last_diagnostics = {"base_diagnostics": self._last_diagnostics}
        self._last_diagnostics["semantic_memory"] = sem_diag

    def _get_ctreepo_prediction(self, text: str):
        """Compute CTreePO sketch + RILE prediction (lazy-loaded model).

        Reuses unified components when available to avoid duplicate model loads.
        """
        # Reuse unified components when they've been initialised
        if hasattr(self, "_unified_ctreepo_model"):
            if self._unified_ctreepo_model is None:
                return None, 0.0
            from treepo._research.training.ctreepo_trainer import extract_root_sketch

            ws = self._unified_ctreepo_settings.get("window_size", 1200)
            sketch, rile = extract_root_sketch(
                self._unified_ctreepo_model, text, self._unified_emb_client,
                window_size=ws,
            )
            return sketch, rile

        if not hasattr(self, "_ctreepo_model"):
            from treepo._research.tree.ctreepo_model import load_ctreepo_model_checkpoint
            from treepo._research.config.settings import load_settings, get_embedding_url, get_embedding_model
            from treepo._research.training.embedding_proxy import VLLMEmbeddingClient

            settings = load_settings()
            ctreepo_cfg = settings.get("ctreepo", {})
            model_path = self.config.ctreepo_model_path or ctreepo_cfg.get("model_path")
            if not model_path:
                self._ctreepo_model = None
                return None, 0.0

            model, _config = load_ctreepo_model_checkpoint(
                model_path,
                config_overrides=ctreepo_cfg,
                map_location="cpu",
            )
            self._ctreepo_model = model

            api_base = get_embedding_url(settings).rstrip("/")
            emb_model = get_embedding_model(settings)
            self._ctreepo_emb_client = VLLMEmbeddingClient(
                api_base=api_base, model=emb_model, timeout_seconds=30.0, batch_size=32,
            )
            self._ctreepo_window_size = ctreepo_cfg.get("window_size", 1200)

        if self._ctreepo_model is None:
            return None, 0.0

        from treepo._research.training.ctreepo_trainer import extract_root_sketch
        sketch, rile = extract_root_sketch(
            self._ctreepo_model, text, self._ctreepo_emb_client,
            window_size=self._ctreepo_window_size,
        )
        return sketch, rile

    def _get_mergeable_sketch_prediction(
        self,
        *,
        text: str,
        leaf_texts: Optional[List[str]] = None,
        doc_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        """
        Compute a strictly-mergeable embedding-sketch prediction.

        Returns:
            (pred_score01, pred_rile, window_count)
        """
        if not hasattr(self, "_mergeable_sketch_model"):
            import torch
            import numpy as np

            from treepo._research.config.settings import load_settings, get_embedding_url, get_embedding_model
            from treepo._research.training.embedding_proxy import VLLMEmbeddingClient
            from treepo._research.training.embedding_sketch import EmbeddingSketchConfig, MergeableEmbeddingSketch
            from treepo._research.embeddings.document_embedder import DocumentEmbedder, DocumentEmbeddingConfig

            settings = load_settings()
            sketch_cfg = settings.get("mergeable_sketch", {})
            model_path = self.config.mergeable_sketch_model_path or sketch_cfg.get("model_path")
            if not model_path:
                self._mergeable_sketch_model = None
                return None, None, None

            ckpt = torch.load(model_path, map_location="cpu")
            cfg_dict = ckpt.get("sketch_config", {}) if isinstance(ckpt, dict) else {}
            if not isinstance(cfg_dict, dict) or not cfg_dict.get("embedding_dim"):
                raise RuntimeError("Invalid mergeable sketch checkpoint (missing sketch_config)")

            cfg = EmbeddingSketchConfig(**{k: v for k, v in cfg_dict.items() if k in EmbeddingSketchConfig.__annotations__})
            model = MergeableEmbeddingSketch(cfg)
            state_dict = ckpt.get("model_state_dict", {}) if isinstance(ckpt, dict) else {}
            model.load_state_dict(state_dict)
            model.eval()
            self._mergeable_sketch_model = model
            self._mergeable_sketch_cfg = cfg

            ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
            if not isinstance(ckpt_args, dict):
                ckpt_args = {}
            self._mergeable_sketch_windowing_mode = str(ckpt_args.get("windowing_mode", "uniform") or "uniform").strip().lower()

            window_chars = int(ckpt_args.get("window_chars", 6000) or 6000)
            overlap_chars = int(ckpt_args.get("overlap_chars", 0) or 0)
            max_windows = int(ckpt_args.get("max_windows", 8) or 8)
            self._mergeable_sketch_window_chars = window_chars
            self._mergeable_sketch_max_windows = max_windows

            api_base = get_embedding_url(settings).rstrip("/")
            emb_model = get_embedding_model(settings)
            self._mergeable_sketch_emb_client = VLLMEmbeddingClient(
                api_base=api_base,
                model=emb_model,
                timeout_seconds=60.0,
                batch_size=32,
                memory=self.config.conditional_memory,
            )

            self._mergeable_sketch_doc_embedder = None
            if self._mergeable_sketch_windowing_mode == "uniform":
                self._mergeable_sketch_doc_embedder = DocumentEmbedder(
                    embedding_client=self._mergeable_sketch_emb_client,
                    config=DocumentEmbeddingConfig(
                        window_chars=window_chars,
                        overlap_chars=overlap_chars,
                        max_windows=max_windows,
                        l2_normalize=True,
                        embed_metadata=True,
                        meta_weight=float(ckpt_args.get("meta_weight", 0.25) or 0.25),
                    ),
                )

        if self._mergeable_sketch_model is None:
            return None, None, None

        import torch
        import numpy as np

        windows: List[List[float]] = []
        mode = str(getattr(self, "_mergeable_sketch_windowing_mode", "uniform") or "uniform").strip().lower()
        if mode == "chunker":
            texts = leaf_texts
            if texts is None:
                chunks = chunk_for_ops(text, max_chars=int(getattr(self, "_mergeable_sketch_window_chars", 6000) or 6000), strategy="axis")
                texts = [c.text for c in chunks]
            texts = [str(t) for t in (texts or []) if str(t).strip()]
            max_windows = int(getattr(self, "_mergeable_sketch_max_windows", 0) or 0)
            if max_windows > 0 and len(texts) > max_windows:
                stride = max(1, int(np.ceil(len(texts) / float(max_windows))))
                reduced = list(texts[::stride])
                if reduced and reduced[-1] != texts[-1]:
                    if len(reduced) >= max_windows:
                        reduced[-1] = texts[-1]
                    else:
                        reduced.append(texts[-1])
                texts = reduced[:max_windows]
            windows = self._mergeable_sketch_emb_client.embed_texts(texts or [])
        else:
            embedder = getattr(self, "_mergeable_sketch_doc_embedder", None)
            if embedder is None:
                raise RuntimeError("Mergeable sketch embedder not initialized")
            _win, _texts, window_embeddings = embedder.embed_text(text)
            windows = list(window_embeddings)

        if not windows:
            return None, None, 0

        mat = np.asarray(windows, dtype=np.float32)
        if mat.ndim != 2 or mat.shape[0] <= 0 or mat.shape[1] <= 0:
            return None, None, 0

        meta_tensor = None
        meta_arr: Optional[np.ndarray] = None
        if getattr(self._mergeable_sketch_cfg, "include_meta", False):
            try:
                from treepo._research.core.doc_metadata import DocMetadata, format_doc_meta_embedding_text

                meta = metadata or {}
                dm = DocMetadata(
                    doc_id=str(doc_id or ""),
                    source=str(meta.get("source", "") or meta.get("dataset", "") or ""),
                    country=str(meta.get("country_name", "") or meta.get("country", "") or "") or None,
                    party=str(meta.get("party_name", "") or meta.get("party", "") or "") or None,
                    party_abbrev=str(meta.get("party_abbrev", "") or "") or None,
                    year=int(meta.get("year")) if meta.get("year") is not None else None,
                    date_code=int(meta.get("date_code")) if meta.get("date_code") is not None else None,
                    election_date=str(meta.get("election_date", "") or "") or None,
                    party_family=int(meta.get("party_family")) if meta.get("party_family") is not None else None,
                    rile=float(meta.get("rile")) if meta.get("rile") is not None else None,
                    extra={},
                )
                meta_text = format_doc_meta_embedding_text(dm)
                meta_vec = self._mergeable_sketch_emb_client.embed_texts([meta_text])[0]
                meta_arr = np.asarray(meta_vec, dtype=np.float32)
                if meta_arr.ndim == 1 and meta_arr.shape[0] == mat.shape[1]:
                    meta_arr = meta_arr / (float(np.linalg.norm(meta_arr) + 1e-12))
                    meta_tensor = torch.from_numpy(meta_arr).to(dtype=torch.float32).unsqueeze(0)
            except Exception:
                meta_tensor = None
                meta_arr = None

        retrieval_tensor = None
        if (
            getattr(self.config.semantic_memory, "enabled", False)
            and getattr(self.config.semantic_memory, "model_features", True)
            and bool(getattr(self._mergeable_sketch_cfg, "include_retrieval_features", False))
        ):
            try:
                query_vec = np.asarray(mat.mean(axis=0), dtype=np.float32)
                query_vec = query_vec / float(np.linalg.norm(query_vec) + 1e-12)
                if meta_arr is not None and meta_arr.shape == query_vec.shape:
                    query_vec = query_vec + 0.25 * meta_arr
                    query_vec = query_vec / float(np.linalg.norm(query_vec) + 1e-12)
                sem_features = self._semantic_features_from_query_vector(
                    query_vector=query_vec,
                    doc_id=str(doc_id or ""),
                    metadata=metadata or {},
                )
                retrieval_tensor = torch.from_numpy(sem_features.astype(np.float32, copy=False)).unsqueeze(0)
            except Exception:
                retrieval_tensor = None

        xb = torch.from_numpy(mat).to(dtype=torch.float32).unsqueeze(0)  # [1,W,D]
        cb = torch.tensor([int(mat.shape[0])], dtype=torch.int64)
        with torch.no_grad():
            pred_out = self._mergeable_sketch_model(
                xb,
                counts=cb,
                meta_embeddings=meta_tensor,
                retrieval_features=retrieval_tensor,
                return_dict=bool(getattr(self._mergeable_sketch_cfg, "include_delta_head", False)),
            )
            if isinstance(pred_out, dict):
                pred_tensor = pred_out.get("rile")
                if pred_tensor is None:
                    return None, None, int(mat.shape[0])
                pred01 = pred_tensor.detach().cpu().numpy().reshape(-1)[0]
            else:
                pred01 = pred_out.detach().cpu().numpy().reshape(-1)[0]

        pred01_f = float(pred01)
        pred01_f = max(0.0, min(1.0, pred01_f))
        pred_rile = -100.0 + 200.0 * pred01_f
        return pred01_f, float(pred_rile), int(mat.shape[0])

    def _build_local_text_strategy(self) -> CallableStrategy:
        """Deterministic summarize/merge fallback when LLM text path is disabled."""
        max_chars = max(256, min(int(self.config.max_chunk_chars), 4000))
        max_tokens = max(0, int(getattr(self.config, "max_chunk_tokens", 0) or 0))

        def _clip_text(text: str) -> str:
            collapsed = " ".join(str(text or "").split())
            if max_tokens > 0:
                from treepo._research.preprocessing.tokenizer import TokenCounter

                counter = TokenCounter(
                    model=None,
                    encoding=str(getattr(self.config, "chunk_token_encoding", "cl100k_base") or "cl100k_base"),
                )
                first, rest = counter.split_at_token_boundary(collapsed, max_tokens=max_tokens)
                if not rest:
                    return collapsed
                return first.rstrip() + "..."
            if len(collapsed) <= max_chars:
                return collapsed
            return collapsed[: max_chars - 3].rstrip() + "..."

        def _summarize(content: str, rubric: str) -> str:
            return _clip_text(content)

        def _merge(left_summary: str, right_summary: str, rubric: str) -> str:
            combined = f"{str(left_summary or '').strip()} {str(right_summary or '').strip()}".strip()
            return _clip_text(combined)

        return CallableStrategy(summarizer=_summarize, merge_fn=_merge)

    def _is_representation_backend_configured(self, backend: str) -> bool:
        name = _normalize_representation_backend(backend)
        if name == str(_LLM_TEXT_PROGRAM_SPEC.program_family):
            return True
        if name == str(_SEMANTIC_EMBEDDING_PROGRAM_SPEC.program_family):
            return bool(getattr(self.config.semantic_memory, "enabled", False))
        if name == str(_CTREEPO_PROGRAM_SPEC.program_family):
            return bool(self.config.ctreepo_model_path) or bool(self.config.unified_tree)
        if name == str(_MERGEABLE_SKETCH_PROGRAM_SPEC.program_family):
            return bool(self.config.mergeable_sketch_model_path)
        if name == "ensemble":
            return True
        return False

    def _resolve_representation_backends(self) -> List[str]:
        requested = _parse_representation_backend_list(self.config.program_families)
        if not requested:
            requested = [str(_LLM_TEXT_PROGRAM_SPEC.program_family)]

        resolved: List[str] = []
        for backend in requested:
            if backend != "auto":
                if backend not in resolved:
                    resolved.append(backend)
                continue
            auto_backends = [str(_LLM_TEXT_PROGRAM_SPEC.program_family)]
            for candidate in (
                str(_SEMANTIC_EMBEDDING_PROGRAM_SPEC.program_family),
                str(_CTREEPO_PROGRAM_SPEC.program_family),
                str(_MERGEABLE_SKETCH_PROGRAM_SPEC.program_family),
            ):
                if self._is_representation_backend_configured(candidate):
                    auto_backends.append(candidate)
            for candidate in auto_backends:
                if candidate not in resolved:
                    resolved.append(candidate)
        if not resolved:
            resolved = [str(_LLM_TEXT_PROGRAM_SPEC.program_family)]
        return resolved

    def _resolve_primary_representation_backend(self) -> str:
        primary = _normalize_representation_backend(self.config.primary_program_family)
        return primary or "auto"

    def _resolve_backend_request_set(self, requested_backends: List[str], primary_backend: str) -> set[str]:
        requested_set = {b for b in requested_backends if b in _ALL_REPRESENTATION_BACKENDS}
        if primary_backend in _ALL_REPRESENTATION_BACKENDS:
            requested_set.add(primary_backend)
        if "ensemble" in requested_set:
            for candidate in _BASE_REPRESENTATION_BACKENDS:
                if self._is_representation_backend_configured(candidate):
                    requested_set.add(candidate)
        return requested_set

    def _score_from_semantic_payload(self, payload: Optional[Dict[str, Any]]) -> Tuple[Optional[float], int]:
        if not isinstance(payload, dict):
            return None, 0

        neighbors = payload.get("neighbors", [])
        if not isinstance(neighbors, list):
            neighbors = []

        weighted_sum = 0.0
        total_weight = 0.0
        support = 0
        for row in neighbors:
            if not isinstance(row, dict):
                continue
            rile = _clip_rile_score(row.get("rile"))
            if rile is None:
                continue
            weight = _safe_float(row.get("score"))
            if weight is None:
                weight = _safe_float(row.get("similarity"))
            if weight is None or weight <= 0.0:
                weight = 1.0
            weighted_sum += float(weight) * float(rile)
            total_weight += float(weight)
            support += 1

        if total_weight > 0.0:
            return float(weighted_sum / total_weight), int(support)

        features = payload.get("features", [])
        if isinstance(features, (list, tuple)) and features:
            fallback = _clip_rile_score(features[0])
            if fallback is not None:
                return fallback, int(support)
        return None, int(support)

    def _score_from_embedding_metadata(self, metadata: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[str]]:
        if not isinstance(metadata, dict):
            return None, None
        # Accept both retrieval-style embedding scores and proxy-head scores.
        candidate_keys = (
            "embedding_retrieval_score",
            "embedding_proxy_score",
            "proxy_estimated_score",
            "proxy_score",
            "cheap_proxy_score",
        )
        for key in candidate_keys:
            value = _clip_rile_score(metadata.get(key))
            if value is not None:
                return float(value), str(key)
        return None, None

    def _score_from_embedding_sources(
        self,
        *,
        payload: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[float], int, str]:
        score, support = self._score_from_semantic_payload(payload)
        if score is not None:
            return float(score), int(support), "semantic_neighbors"
        meta_score, meta_source = self._score_from_embedding_metadata(metadata)
        if meta_score is not None:
            return float(meta_score), int(support), f"metadata:{meta_source}"
        return None, int(support), "missing"

    def _resolve_ensemble_weights(
        self,
        backend_scores: Dict[str, float],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        candidates = {
            key: float(value)
            for key, value in backend_scores.items()
            if key in _BASE_REPRESENTATION_BACKENDS and _safe_float(value) is not None
        }
        if not candidates:
            return {}, {"mode": "empty"}

        raw_weights = self.config.program_weights or {}
        weights: Dict[str, float] = {}
        for backend in candidates:
            fallback = 1.0
            if isinstance(raw_weights, dict):
                fallback = float(max(0.0, _safe_float(raw_weights.get(backend)) or 0.0))
                if fallback <= 0.0:
                    fallback = 1.0
            weights[backend] = fallback

        def _normalize(local_weights: Dict[str, float]) -> Dict[str, float]:
            denom = float(sum(local_weights.values()))
            if denom <= 0.0:
                uniform = 1.0 / float(max(1, len(local_weights)))
                return {key: uniform for key in local_weights}
            return {key: float(value) / denom for key, value in local_weights.items()}

        normalized = _normalize(weights)
        diagnostics: Dict[str, Any] = {
            "mode": "static",
            "candidates": sorted(candidates.keys()),
            "weights_static": {k: float(v) for k, v in normalized.items()},
        }

        # Hybrid mode: use LLM as seed/oracle anchor while boosting embedding/operator
        # corrections when they are available and sufficiently supported.
        llm_family = str(_LLM_TEXT_PROGRAM_SPEC.program_family)
        embedding_family = str(_SEMANTIC_EMBEDDING_PROGRAM_SPEC.program_family)
        ctreepo_family = str(_CTREEPO_PROGRAM_SPEC.program_family)
        mergeable_family = str(_MERGEABLE_SKETCH_PROGRAM_SPEC.program_family)
        non_llm_candidates = [b for b in candidates.keys() if b != llm_family]
        if (
            not self.config.hybrid_oracle_seeded_ensemble
            or llm_family not in candidates
            or not non_llm_candidates
        ):
            return normalized, diagnostics

        metadata = metadata or {}
        confidence: Dict[str, float] = {backend: 1.0 for backend in candidates}

        if embedding_family in candidates:
            support = _safe_float(metadata.get("semantic_program_support"))
            source = str(metadata.get("semantic_program_source", "") or "").strip().lower()
            top_k = max(1, int(getattr(self.config.semantic_memory, "top_k", 5) or 5))
            if support is not None:
                confidence[embedding_family] = max(0.2, min(1.0, float(support) / float(top_k)))
            elif source.startswith("metadata:"):
                confidence[embedding_family] = 0.55
            else:
                confidence[embedding_family] = 0.35

        if ctreepo_family in candidates:
            confidence[ctreepo_family] = _clip01(metadata.get("ctreepo_confidence")) or 0.70

        if mergeable_family in candidates:
            window_count = _safe_float(metadata.get("mergeable_sketch_window_count"))
            if window_count is None:
                confidence[mergeable_family] = 0.60
            else:
                confidence[mergeable_family] = max(0.3, min(1.0, float(window_count) / 8.0))

        hybrid_weights = dict(weights)
        operator_boost = max(0.0, float(self.config.hybrid_operator_boost))
        for backend in non_llm_candidates:
            hybrid_weights[backend] = float(hybrid_weights[backend]) * max(0.05, float(confidence.get(backend, 1.0))) * operator_boost

        hybrid_norm = _normalize(hybrid_weights)
        llm_weight = float(hybrid_norm.get(llm_family, 0.0))
        min_llm = float(self.config.hybrid_seed_llm_min_weight)
        max_llm = float(self.config.hybrid_seed_llm_max_weight)
        llm_target = max(min_llm, min(max_llm, llm_weight))

        adjusted = dict(hybrid_norm)
        non_llm_sum = float(sum(adjusted.get(b, 0.0) for b in non_llm_candidates))
        remaining = max(0.0, 1.0 - llm_target)
        if non_llm_sum > 0.0:
            scale = remaining / non_llm_sum
            for backend in non_llm_candidates:
                adjusted[backend] = float(adjusted.get(backend, 0.0)) * scale
            adjusted[llm_family] = llm_target
        else:
            adjusted = {llm_family: 1.0}
            for backend in non_llm_candidates:
                adjusted[backend] = 0.0

        diagnostics = {
            "mode": "hybrid_oracle_seeded",
            "candidates": sorted(candidates.keys()),
            "operator_boost": operator_boost,
            "confidence_by_backend": {k: float(v) for k, v in confidence.items()},
            "weights_static": {k: float(v) for k, v in normalized.items()},
            "weights_hybrid_preclamp": {k: float(v) for k, v in hybrid_norm.items()},
            "weights_hybrid": {k: float(v) for k, v in adjusted.items()},
            "llm_weight_bounds": {"min": min_llm, "max": max_llm},
        }
        return adjusted, diagnostics

    def _select_representation_score(
        self,
        *,
        backend_scores: Dict[str, float],
        requested_backends: List[str],
        primary_backend: str,
    ) -> Tuple[Optional[str], Optional[float], bool]:
        selected_backend: Optional[str] = None
        selected_score: Optional[float] = None
        fallback_used = False

        selection_order: List[str] = []
        if primary_backend != "auto":
            selection_order.append(primary_backend)
        for backend in requested_backends:
            if backend not in selection_order:
                selection_order.append(backend)
        if primary_backend == "auto":
            selection_order = list(requested_backends)

        if not self.config.fallback_to_available_backend:
            target = primary_backend
            if target == "auto":
                target = selection_order[0] if selection_order else "llm"
            value = _safe_float(backend_scores.get(target))
            if value is not None:
                selected_backend = target
                selected_score = float(value)
            return selected_backend, selected_score, False

        for backend in selection_order:
            value = _safe_float(backend_scores.get(backend))
            if value is None:
                continue
            selected_backend = backend
            selected_score = float(value)
            break

        if selected_backend is None and backend_scores:
            # Last-resort fail-open: keep one available backend score.
            first_backend = next(iter(backend_scores.keys()))
            value = _safe_float(backend_scores.get(first_backend))
            if value is not None:
                selected_backend = first_backend
                selected_score = float(value)

        if (
            selected_backend is not None
            and primary_backend not in {"", "auto"}
            and selected_backend != primary_backend
        ):
            fallback_used = True
        return selected_backend, selected_score, fallback_used

    def _resolve_effective_selected_score(
        self,
        *,
        selected_score: Optional[float],
        metadata: Dict[str, Any],
    ) -> Tuple[Optional[float], bool]:
        parsed_score = _safe_float(selected_score)
        if parsed_score is not None:
            metadata["missing_score_default_applied"] = False
            metadata["missing_score_default_value"] = None
            metadata.pop("missing_score_default_reason", None)
            return float(parsed_score), False

        default_score = _safe_float(self.config.missing_score_default)
        if default_score is None:
            metadata["missing_score_default_applied"] = False
            metadata["missing_score_default_value"] = None
            return None, False

        metadata["missing_score_default_applied"] = True
        metadata["missing_score_default_value"] = float(default_score)
        metadata["missing_score_default_reason"] = str(
            metadata.get("missing_score_default_reason")
            or metadata.get("llm_score_failure_reason")
            or "missing_backend_score"
        )
        return float(default_score), True

    def _program_spec_for_family(self, program_family: str) -> Optional[UnifiedFGSpec]:
        family = str(program_family or "").strip()
        if not family or family in {"auto", "ensemble"}:
            return None
        if family == str(_LLM_TEXT_PROGRAM_SPEC.program_family):
            return build_llm_text_program_spec(
                tokenizer_or_adapter_id=str(self.config.chunk_token_encoding or "cl100k_base")
            )
        if family == str(_SEMANTIC_EMBEDDING_PROGRAM_SPEC.program_family):
            feature_dim = int(_safe_float(getattr(self.config.semantic_memory, "top_k", 0)) or 0)
            return build_semantic_embedding_program_spec(
                feature_dim=feature_dim,
                tokenizer_or_adapter_id="semantic_memory",
            )
        if family == str(_CTREEPO_PROGRAM_SPEC.program_family):
            return build_ctreepo_program_spec(
                feature_dim=0,
                tokenizer_or_adapter_id="embedding_tree",
            )
        if family == str(_MERGEABLE_SKETCH_PROGRAM_SPEC.program_family):
            return build_mergeable_sketch_program_spec(
                feature_dim=0,
                tokenizer_or_adapter_id="mergeable_sketch",
            )
        resolved = resolve_program_spec_alias(family)
        return resolved

    def _write_program_routing_metadata(
        self,
        *,
        metadata: Dict[str, Any],
        requested_backends: List[str],
        primary_backend: str,
        backend_scores: Dict[str, float],
        selected_backend: Optional[str],
        selected_score: Optional[float],
        effective_selected_score: Optional[float] = None,
        fallback_used: bool,
        ensemble_diag: Optional[Dict[str, Any]] = None,
        ensemble_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        metadata["program_families_requested"] = list(requested_backends)
        metadata["primary_program_family"] = str(primary_backend)
        metadata["program_family_scores"] = {
            backend: float(score) for backend, score in backend_scores.items()
        }
        metadata["selected_program_family"] = selected_backend
        metadata["selected_program_score_raw"] = (
            None if selected_score is None else float(selected_score)
        )
        metadata["selected_program_score"] = (
            None if effective_selected_score is None else float(effective_selected_score)
        )
        metadata["program_fallback_used"] = bool(fallback_used)
        selected_spec = self._program_spec_for_family(str(selected_backend or ""))
        if selected_spec is not None:
            metadata["selected_program_spec"] = selected_spec.to_dict()
            metadata["space_kind"] = selected_spec.space.space_kind
            metadata["g_learner_kind"] = selected_spec.g_learner.learner_kind
            metadata["f_learner_kind"] = selected_spec.f_learner.learner_kind
            metadata["program_family"] = selected_spec.program_family
            metadata["feature_dim"] = int(selected_spec.feature_dim)
            metadata["operator_width"] = (
                None if selected_spec.operator_width is None else int(selected_spec.operator_width)
            )
            metadata["tokenizer_or_adapter_id"] = selected_spec.space.tokenizer_or_adapter_id
        if ensemble_weights:
            metadata["program_ensemble_weights"] = {
                backend: float(weight) for backend, weight in ensemble_weights.items()
            }
        if ensemble_diag:
            metadata["program_ensemble_mode"] = str(ensemble_diag.get("mode", "static"))
            metadata["program_ensemble_diagnostics"] = dict(ensemble_diag)

    @property
    def last_stats(self) -> Optional[BatchStats]:
        """Latest BatchStats snapshot from the most recent run (best-effort)."""
        return self._last_stats

    @property
    def last_diagnostics(self) -> Optional[Dict[str, Any]]:
        """Latest per-run client diagnostics (routing/recovery/cache)."""
        return self._last_diagnostics

    def _ensure_unified_components(self):
        """Lazy-initialise components for the unified tree pipeline.

        Sets up:
          - ``_unified_ctreepo_model`` / ``_unified_ctreepo_config``:
            CTreePO model for sketch scoring.
          - ``_unified_emb_client``: Embedding client shared by tree building
            and CTreePO.
          - ``_unified_mil_model`` (optional): MIL proxy for window importance.
          - ``_unified_ctreepo_settings``: Windowing & merge settings from YAML.
        """
        if hasattr(self, "_unified_emb_client"):
            return  # already initialised

        import torch
        from treepo._research.tree.ctreepo_model import CTreePOConfig, CTreePOModel
        from treepo._research.config.settings import load_settings, get_embedding_url, get_embedding_model
        from treepo._research.training.embedding_proxy import VLLMEmbeddingClient

        settings = load_settings()
        ctreepo_cfg = settings.get("ctreepo", {}) if isinstance(settings, dict) else {}

        # --- Embedding client ---
        api_base = get_embedding_url(settings).rstrip("/")
        emb_model = get_embedding_model(settings)
        self._unified_emb_client = VLLMEmbeddingClient(
            api_base=api_base, model=emb_model, timeout_seconds=30.0, batch_size=32,
        )

        # --- CTreePO model ---
        model_path = self.config.ctreepo_model_path or (
            ctreepo_cfg.get("model_path") if isinstance(ctreepo_cfg, dict) else None
        )
        if model_path:
            from treepo._research.tree.ctreepo_model import load_ctreepo_model_checkpoint

            model, cfg = load_ctreepo_model_checkpoint(
                model_path,
                config_overrides=ctreepo_cfg,
                map_location="cpu",
            )
            self._unified_ctreepo_model = model
            self._unified_ctreepo_config = cfg
        else:
            self._unified_ctreepo_model = None
            self._unified_ctreepo_config = None

        # --- Window config ---
        self._unified_ctreepo_settings = {
            "window_size": ctreepo_cfg.get("window_size", 1200),
            "coarse_window_size": ctreepo_cfg.get("coarse_window_size", 4000),
            "merge_drift_threshold": ctreepo_cfg.get("merge_drift_threshold", 0.03),
        }

        # --- MIL proxy (optional, Phase 5) ---
        self._unified_mil_model = None
        mil_path = self.config.mil_proxy_model_path
        if mil_path:
            try:
                from treepo._research.training.embedding_proxy import load_embedding_proxy_model
                from pathlib import Path

                self._unified_mil_model = load_embedding_proxy_model(Path(mil_path))
            except Exception as e:
                logger.warning("Failed to load MIL proxy model: %s", e)

    def _build_all_unified_trees(
        self,
        samples: List[DocumentSample],
        sample_doc_ids: List[str],
    ) -> Dict[int, List[Any]]:
        """Build embedding-based unified trees for all documents.

        Embedding calls are CPU/network (not LLM), so this runs before
        the orchestrator handles LLM summarisation.
        """
        from treepo._research.tree.embedding_tree import build_unified_tree

        sw_cfg = self._unified_ctreepo_settings
        score_cb = None
        if self._unified_mil_model is not None:
            mil = self._unified_mil_model
            score_cb = lambda embs: mil.get_mil_attention_scores(embs)

        trees: Dict[int, List[Any]] = {}
        for idx, sample in enumerate(samples):
            doc_id = sample_doc_ids[idx] if idx < len(sample_doc_ids) else str(idx)
            try:
                nodes = build_unified_tree(
                    sample.text,
                    self._unified_emb_client,
                    coarse_window_size=sw_cfg["coarse_window_size"],
                    fine_window_size=sw_cfg["window_size"],
                    merge_drift_threshold=sw_cfg["merge_drift_threshold"],
                    adaptive=self.config.adaptive_windows,
                    score_windows_callback=score_cb,
                )
                trees[idx] = nodes
            except Exception as e:
                logger.error("Failed to build unified tree for %s: %s", doc_id, e)
        return trees

    def _ctreepo_postprocess(
        self,
        unified_trees: Dict[int, List[Any]],
        build_results: List[Any],
    ) -> None:
        """Run CTreePO sketch scoring on unified trees after LLM summarisation.

        Updates ``build_results`` metadata in-place with sketch scores,
        audit candidates, and feedback signals.
        """
        if self._unified_ctreepo_model is None:
            return

        from treepo._research.tree.embedding_tree import forward_ctreepo_batch

        # Collect valid trees for batched forward pass.
        valid_entries: list[tuple[int, list[Any]]] = []
        for idx, nodes in unified_trees.items():
            if idx >= len(build_results):
                continue
            if build_results[idx].errors:
                continue
            valid_entries.append((idx, nodes))

        # Batched forward pass across all valid trees at once.
        if valid_entries:
            try:
                forward_ctreepo_batch(
                    self._unified_ctreepo_model,
                    [nodes for _, nodes in valid_entries],
                )
            except Exception as e:
                logger.debug("Batched CTreePO forward failed: %s", e)
                return

        for idx, nodes in valid_entries:
            build_result = build_results[idx]

            try:
                pass  # Forward pass already done above in batch

                # Populate sketch scores and confidence
                try:
                    from treepo._research.training.ctreepo_trainer import CTreePOTrainer, CTreePOTrainingConfig
                    trainer = CTreePOTrainer(
                        CTreePOTrainingConfig(model=self._unified_ctreepo_config)
                    )
                    trainer.model = self._unified_ctreepo_model
                    trainer.populate_sketch_scores(nodes, head="rile")

                    audit_indices = trainer.select_audit_nodes(nodes, n_audit=5)
                    build_result.tree.metadata["unified_audit_candidates"] = audit_indices
                except Exception as e:
                    logger.debug("Sketch scoring/audit selection failed: %s", e)

                # Write root-level CTreePO prediction into tree metadata
                root_node = nodes[-1]
                if root_node.sketch_scores.get("rile") is not None:
                    build_result.tree.metadata["ctreepo_rile"] = round(
                        root_node.sketch_scores["rile"], 2
                    )
                if root_node.sketch is not None:
                    try:
                        conf = self._unified_ctreepo_model.predict_confidence(root_node.sketch, "rile")
                        mean, lower, upper, std = self._unified_ctreepo_model.predict_interval(
                            root_node.sketch,
                            "rile",
                        )
                        build_result.tree.metadata["ctreepo_confidence"] = round(float(conf.item()), 4)
                        build_result.tree.metadata["ctreepo_std_proxy"] = round(float(std.item()), 4)
                        build_result.tree.metadata["ctreepo_interval_95"] = [
                            round(float(lower.item()), 2),
                            round(float(upper.item()), 2),
                        ]
                        build_result.tree.metadata["ctreepo_rile"] = round(float(mean.item()), 2)
                    except Exception as e:
                        logger.debug("CTreePO uncertainty metadata failed: %s", e)
                if root_node.sketch is not None:
                    build_result.tree.metadata["ctreepo_sketch_dim"] = int(
                        root_node.sketch.shape[0]
                    )

                # Chunking metadata
                leaves = [n for n in nodes if n.is_leaf]
                build_result.tree.metadata["chunking"] = {
                    "strategy": "unified_adaptive" if self.config.adaptive_windows else "unified_uniform",
                    "fine_window_size": self._unified_ctreepo_settings["window_size"],
                    "coarse_window_size": self._unified_ctreepo_settings["coarse_window_size"],
                    "merge_drift_threshold": self._unified_ctreepo_settings["merge_drift_threshold"],
                    "adaptive": self.config.adaptive_windows,
                    "mil_proxy": self._unified_mil_model is not None,
                }

                # Oracle feedback signals
                if self.config.oracle_feedback_to_chunks:
                    try:
                        from treepo._research.preprocessing.chunker import oracle_to_feedback_signals
                        signals = oracle_to_feedback_signals(nodes)
                        if signals:
                            build_result.tree.metadata["feedback_signal_count"] = len(signals)
                            build_result.tree.metadata["feedback_signals"] = [
                                {
                                    "char_start": getattr(s, "char_start", 0),
                                    "char_end": getattr(s, "char_end", 0),
                                    "score": getattr(s, "score", 0.0),
                                    "role": getattr(s, "role", "boundary"),
                                }
                                for s in signals
                            ]
                    except Exception as e:
                        logger.debug("Feedback signal generation failed: %s", e)

            except Exception as e:
                logger.debug("CTreePO postprocess failed for doc %d: %s", idx, e)

    def _ctreepo_postprocess_single(
        self,
        nodes: List[Any],
        metadata: Dict[str, Any],
    ) -> None:
        """Run CTreePO sketch scoring on a single document's unified nodes.

        Populates sketch scores, audit candidates, and root-level predictions
        in *metadata* dict (in-place).
        """
        try:
            from treepo._research.training.ctreepo_trainer import CTreePOTrainer, CTreePOTrainingConfig
            trainer = CTreePOTrainer(
                CTreePOTrainingConfig(model=self._unified_ctreepo_config)
            )
            trainer.model = self._unified_ctreepo_model
            trainer.populate_sketch_scores(nodes, head="rile")

            audit_indices = trainer.select_audit_nodes(nodes, n_audit=5)
            metadata["unified_audit_candidates"] = audit_indices
        except Exception as e:
            logger.debug("Sketch scoring/audit selection failed: %s", e)

        root_node = nodes[-1]
        if root_node.sketch is not None:
            if root_node.sketch_scores.get("rile") is not None:
                metadata["ctreepo_rile"] = round(root_node.sketch_scores["rile"], 2)
            try:
                conf = self._unified_ctreepo_model.predict_confidence(root_node.sketch, "rile")
                mean, lower, upper, std = self._unified_ctreepo_model.predict_interval(
                    root_node.sketch,
                    "rile",
                )
                metadata["ctreepo_confidence"] = round(float(conf.item()), 4)
                metadata["ctreepo_std_proxy"] = round(float(std.item()), 4)
                metadata["ctreepo_interval_95"] = [
                    round(float(lower.item()), 2),
                    round(float(upper.item()), 2),
                ]
                metadata["ctreepo_rile"] = round(float(mean.item()), 2)
            except Exception as e:
                logger.debug("CTreePO uncertainty metadata failed (single): %s", e)
            metadata["ctreepo_sketch_dim"] = int(root_node.sketch.shape[0])

    async def process_unified(
        self,
        sample: DocumentSample,
        strategy: SummarizationStrategy,
    ) -> DocumentResult:
        """Process a document through the unified tree pipeline.

        Builds a single tree topology shared by both the sketch path (CTreePO)
        and the text path (LLM summarisation), with optional adaptive windowing
        and oracle feedback.

        Pipeline:
            1. Build unified tree (adaptive windows + embeddings)
            2. CTreePO forward pass (sketch all nodes)
            3. Populate sketch scores and confidence
            4. LLM summarisation (text-summarise all nodes using *strategy*)
            5. Select uncertain nodes for oracle audit (metadata only — actual
               oracle calls are deferred to the caller)
            6. Return result with both summary and sketch scores

        Args:
            sample: Document to process.
            strategy: SummarizationStrategy for LLM leaf/merge calls.

        Returns:
            DocumentResult with unified metadata.
        """
        from treepo._research.tree.embedding_tree import build_unified_tree, forward_ctreepo  # noqa: F811
        from treepo._research.tree.builder import TreeBuilder, BuildConfig

        doc_id = sample.doc_id
        start_time = time.time()

        result = DocumentResult(
            doc_id=doc_id,
            original_content=sample.text,
            reference_score=_extract_reference_score(sample),
            original_length=len(sample.text),
            metadata=_extract_metadata(sample),
        )

        semantic_payload = self._semantic_payload_for_document(
            text=sample.text,
            doc_id=doc_id,
            metadata=result.metadata,
        )
        semantic_prompt_payload = None
        if semantic_payload is not None:
            semantic_prompt_payload = {"neighbors": list(semantic_payload.get("neighbors", []))}
            result.metadata["semantic_memory_neighbor_count"] = int(semantic_payload.get("neighbor_count", 0) or 0)
            result.metadata["semantic_memory_retrieval_ms"] = float(semantic_payload.get("retrieval_ms", 0.0) or 0.0)
            self._semantic_payload_by_doc_id[str(doc_id)] = semantic_payload

        _meta_token = engram_document_metadata.set(result.metadata)
        _semantic_token = semantic_document_memory.set(semantic_prompt_payload)
        try:
            self._ensure_unified_components()

            # --- 1. Build unified tree ---
            sw_cfg = self._unified_ctreepo_settings
            score_cb = None
            if self._unified_mil_model is not None:
                mil = self._unified_mil_model
                score_cb = lambda embs: mil.get_mil_attention_scores(embs)

            nodes = build_unified_tree(
                sample.text,
                self._unified_emb_client,
                coarse_window_size=sw_cfg["coarse_window_size"],
                fine_window_size=sw_cfg["window_size"],
                merge_drift_threshold=sw_cfg["merge_drift_threshold"],
                adaptive=self.config.adaptive_windows,
                score_windows_callback=score_cb,
            )

            # --- 2 & 3. CTreePO forward pass + sketch scores ---
            if self._unified_ctreepo_model is not None:
                forward_ctreepo(self._unified_ctreepo_model, nodes)
                self._ctreepo_postprocess_single(nodes, result.metadata)

            # --- 4. LLM summarisation on shared topology ---
            build_config = BuildConfig(
                max_chunk_chars=self.config.max_chunk_chars,
                max_chunk_tokens=self.config.max_chunk_tokens,
                chunk_token_encoding=self.config.chunk_token_encoding,
                fail_on_degenerate_summary=self.config.fail_on_degenerate_summary,
                max_degenerate_leaf_fallbacks=self.config.max_degenerate_leaf_fallbacks,
                max_degenerate_merge_fallbacks=self.config.max_degenerate_merge_fallbacks,
                runtime_mode=self.config.runtime_mode,
                batch_plan_cache_name="batched_pipeline_unified_tree",
            )
            builder = TreeBuilder(strategy=strategy, config=build_config)
            await builder.summarize_unified_nodes(nodes, rubric=self.config.rubric)

            # Extract results from unified tree
            leaves = [n for n in nodes if n.is_leaf]
            root = nodes[-1]

            result.tree_height = root.level
            result.tree_leaves = len(leaves)
            result.final_summary = root.summary or ""
            result.summary_length = len(result.final_summary)
            result.compression_ratio = (
                result.original_length / max(result.summary_length, 1)
            )
            result.chunks = [n.text_span or "" for n in leaves]
            result.leaf_summaries = [n.summary or "" for n in leaves]

            # --- 5. Score root summary via oracle ---
            if (
                result.final_summary
                and self.config.prompt_builders.score
                and self.config.score_parser
            ):
                _client = getattr(strategy, "_client", None) or getattr(
                    strategy, "client", None
                )
                if _client is not None:
                    try:
                        score_request = BatchRequest(
                            request_id=f"{doc_id}_score",
                            messages=self.config.prompt_builders.score(
                                result.final_summary,
                                self.config.task_context,
                            ),
                            max_tokens=self.config.max_tokens_score,
                            temperature=0.0,
                            document_id=doc_id,
                            request_type="score",
                        )
                        await _client.submit(score_request)
                        score_response = await _client.await_response(
                            score_request.request_id,
                            timeout=float(self.config.await_response_timeout_seconds),
                        )
                        if score_response.error:
                            result.metadata["llm_score_failure_reason"] = str(score_response.error)
                        else:
                            result.estimated_score = self.config.score_parser(
                                score_response.content
                            )
                            if result.estimated_score is None:
                                result.metadata["llm_score_failure_reason"] = "score_parse_failed"
                                result.metadata["llm_score_failure_preview"] = str(
                                    score_response.content or ""
                                )[:240]
                        result.reasoning = score_response.content or ""
                    except Exception as e:
                        logger.debug("Oracle scoring failed for %s: %s", doc_id, e)

            # --- 6. Tree metadata ---
            result.metadata["unified_tree_node_count"] = len(nodes)
            result.metadata["unified_tree_leaf_count"] = len(leaves)
            result.metadata["unified_tree"] = True
            result.metadata["chunking"] = {
                "strategy": "unified_adaptive" if self.config.adaptive_windows else "unified_uniform",
                "fine_window_size": sw_cfg["window_size"],
                "coarse_window_size": sw_cfg["coarse_window_size"],
                "merge_drift_threshold": sw_cfg["merge_drift_threshold"],
                "adaptive": self.config.adaptive_windows,
                "mil_proxy": self._unified_mil_model is not None,
            }
            result.metadata["chunk_boundaries"] = [
                {"char_start": n.char_start, "char_end": n.char_end}
                for n in leaves
            ]

            requested_backends = self._resolve_representation_backends()
            primary_backend = self._resolve_primary_representation_backend()
            requested_backend_set = self._resolve_backend_request_set(
                requested_backends=requested_backends,
                primary_backend=primary_backend,
            )
            llm_family = str(_LLM_TEXT_PROGRAM_SPEC.program_family)
            embedding_family = str(_SEMANTIC_EMBEDDING_PROGRAM_SPEC.program_family)
            ctreepo_family = str(_CTREEPO_PROGRAM_SPEC.program_family)
            mergeable_family = str(_MERGEABLE_SKETCH_PROGRAM_SPEC.program_family)
            backend_scores: Dict[str, float] = {}
            llm_score = _safe_float(result.estimated_score)
            if llm_score is not None:
                backend_scores[llm_family] = float(llm_score)

            if ctreepo_family in requested_backend_set:
                ctreepo_rile = _safe_float(result.metadata.get("ctreepo_rile"))
                if (
                    ctreepo_rile is None
                    and self._is_representation_backend_configured("ctreepo")
                ):
                    try:
                        sketch, pred = self._get_ctreepo_prediction(sample.text)
                        if sketch is not None:
                            ctreepo_rile = _safe_float(pred)
                    except Exception as exc:
                        result.metadata["ctreepo_error"] = str(exc)
                        logger.debug("CTreePO prediction failed for %s: %s", doc_id, exc)
                if ctreepo_rile is not None:
                    backend_scores[ctreepo_family] = float(ctreepo_rile)
                    result.metadata["ctreepo_rile"] = round(float(ctreepo_rile), 4)

            if (
                mergeable_family in requested_backend_set
                and self._is_representation_backend_configured("mergeable_sketch")
            ):
                try:
                    pred01, pred_rile, window_count = self._get_mergeable_sketch_prediction(
                        text=sample.text,
                        leaf_texts=result.chunks,
                        doc_id=doc_id,
                        metadata=result.metadata,
                    )
                    if pred01 is not None:
                        result.metadata["mergeable_sketch_pred01"] = float(pred01)
                    if window_count is not None:
                        result.metadata["mergeable_sketch_window_count"] = int(window_count)
                    pred_rile_f = _safe_float(pred_rile)
                    if pred_rile_f is not None:
                        backend_scores[mergeable_family] = float(pred_rile_f)
                        result.metadata["mergeable_sketch_rile"] = round(float(pred_rile_f), 4)
                except Exception as exc:
                    result.metadata["mergeable_sketch_error"] = str(exc)
                    logger.debug("Mergeable sketch prediction failed for %s: %s", doc_id, exc)

            if embedding_family in requested_backend_set:
                embedding_score, embedding_support, embedding_source = self._score_from_embedding_sources(
                    payload=semantic_payload,
                    metadata=result.metadata,
                )
                result.metadata["semantic_program_support"] = int(embedding_support)
                result.metadata["semantic_program_source"] = str(embedding_source)
                if embedding_score is not None:
                    backend_scores[embedding_family] = float(embedding_score)
                    result.metadata["embedding_retrieval_score"] = round(float(embedding_score), 4)

            ensemble_diag: Dict[str, Any] | None = None
            ensemble_weights: Dict[str, float] | None = None
            if "ensemble" in requested_backend_set:
                ensemble_weights, ensemble_diag = self._resolve_ensemble_weights(
                    backend_scores,
                    metadata=result.metadata,
                )
                if ensemble_weights:
                    ensemble_score = 0.0
                    for backend, weight in ensemble_weights.items():
                        ensemble_score += float(weight) * float(backend_scores[backend])
                    backend_scores["ensemble"] = float(ensemble_score)

            selected_backend, selected_score, fallback_used = self._select_representation_score(
                backend_scores=backend_scores,
                requested_backends=requested_backends,
                primary_backend=primary_backend,
            )
            effective_selected_score, default_score_applied = self._resolve_effective_selected_score(
                selected_score=selected_score,
                metadata=result.metadata,
            )
            self._write_program_routing_metadata(
                metadata=result.metadata,
                requested_backends=requested_backends,
                primary_backend=primary_backend,
                backend_scores=backend_scores,
                selected_backend=selected_backend,
                selected_score=selected_score,
                effective_selected_score=effective_selected_score,
                fallback_used=fallback_used,
                ensemble_diag=ensemble_diag,
                ensemble_weights=ensemble_weights,
            )
            if default_score_applied and selected_backend is None:
                result.metadata["selected_program_family"] = "missing_score_default"
            result.estimated_score = effective_selected_score

        except Exception as e:
            logger.error("Error in unified processing for %s: %s", doc_id, e)
            result.error = str(e)
        finally:
            semantic_document_memory.reset(_semantic_token)
            engram_document_metadata.reset(_meta_token)

        result.processing_time = time.time() - start_time
        if semantic_payload is not None:
            try:
                query_vec = np.asarray(semantic_payload.get("query_vector"), dtype=np.float32)
            except Exception:
                query_vec = None
            self._semantic_write_after_score(result=result, query_vector=query_vec)
        return result

    async def process_batch_with_strategy(
        self,
        samples: List[DocumentSample],
        strategy: SummarizationStrategy,
        show_progress: bool = True,
    ) -> List[DocumentResult]:
        """
        Process multiple documents using a SummarizationStrategy.

        Routes through the BatchTreeOrchestrator for global batching.
        The strategy handles LLM access (e.g. DSPyStrategy uses DSPy LMs,
        BatchedStrategy uses AsyncBatchLLMClient).

        Args:
            samples: List of document samples
            strategy: SummarizationStrategy to use
            show_progress: Whether to show progress

        Returns:
            List of DocumentResult
        """
        return await self.process_batch_global_async(
            samples, show_progress=show_progress, strategy=strategy
        )

    async def process_batch_async(
        self,
        samples: List[DocumentSample],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        show_progress: Optional[bool] = None,
    ) -> List[DocumentResult]:
        """
        Process multiple documents with global pipelined batching.

        All LLM requests are pooled across documents for maximum throughput.

        Args:
            samples: List of document samples
            progress_callback: Optional progress callback(completed, total)
            show_progress: Override config.show_progress setting

        Returns:
            List of DocumentResults
        """
        self._reset_semantic_runtime_stats()
        # Convert simple callback to phase-aware callback
        phase_callback = None
        if progress_callback:
            def phase_callback(phase: str, completed: int, total: int):
                if phase == "chunk":
                    progress_callback(completed, total)

        return await self.process_batch_global_async(
            samples, phase_callback, show_progress
        )

    def process_batch(
        self,
        samples: List[DocumentSample],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        show_progress: Optional[bool] = None,
    ) -> List[DocumentResult]:
        """
        Sync wrapper for batch processing.

        Args:
            samples: List of manifesto samples
            progress_callback: Optional progress callback
            show_progress: Override config.show_progress setting

        Returns:
            List of DocumentResults
        """
        return asyncio.run(self.process_batch_async(samples, progress_callback, show_progress))

    def get_results(self) -> List[DocumentResult]:
        """Get all processed results."""
        return self._results

    async def process_batch_global_async(
        self,
        samples: List[DocumentSample],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        show_progress: Optional[bool] = None,
        *,
        strategy: Optional[SummarizationStrategy] = None,
    ) -> List[DocumentResult]:
        """
        Process documents using global pipelined batching for maximum throughput.

        This method processes ALL documents together:
        1. Chunks ALL documents
        2. Submits ALL leaf summaries together
        3. Schedules merges globally as soon as dependencies are ready
        4. Scores ALL documents together

        When ``strategy`` is provided (e.g. a DSPyStrategy for training),
        no LLM client is created — the strategy handles LLM access directly.

        Args:
            samples: List of document samples
            progress_callback: Optional callback(phase, completed, total)
            show_progress: Override config.show_progress setting
            strategy: Optional pre-built strategy. When provided, skips
                LLM client creation and scoring phases.

        Returns:
            List of DocumentResults
        """
        self._reset_semantic_runtime_stats()
        logger.info(f"Starting GLOBAL pipelined processing of {len(samples)} documents")
        start_time = time.time()

        use_progress = show_progress if show_progress is not None else self.config.show_progress

        requested_backends = self._resolve_representation_backends()
        primary_backend = self._resolve_primary_representation_backend()
        requested_backend_set = self._resolve_backend_request_set(
            requested_backends=requested_backends,
            primary_backend=primary_backend,
        )
        llm_score_requested = str(_LLM_TEXT_PROGRAM_SPEC.program_family) in requested_backend_set
        logger.info(
            "Program routing: families=%s primary=%s fallback=%s llm_text_path_enabled=%s",
            ",".join(requested_backends),
            primary_backend,
            bool(self.config.fallback_to_available_backend),
            bool(self.config.llm_text_path_enabled),
        )

        # When an external strategy is provided (e.g. DSPyStrategy), skip
        # client/metrics creation — the strategy manages LLM access itself.
        external_strategy = strategy is not None
        if not external_strategy and not bool(self.config.llm_text_path_enabled):
            strategy = self._build_local_text_strategy()
            external_strategy = True
            logger.info(
                "LLM text path disabled; using local deterministic summarize/merge strategy."
            )
        client = None
        llm_scoring_client = None
        metrics_collector = None

        if not external_strategy:
            server_urls = self.config.task_model_urls or [self.config.task_model_url]
            host, ports = _extract_host_and_ports(server_urls)
            if (self.config.metrics_poll_seconds or 0.0) > 0.0 and host and ports:
                try:
                    from treepo._research.core.vllm_metrics import VLLMMetricsCollector

                    metrics_collector = VLLMMetricsCollector(
                        ports=ports,
                        host=host,
                        poll_interval=float(self.config.metrics_poll_seconds or 0.0),
                    )
                except Exception:
                    metrics_collector = None

            if len(server_urls) > 1:
                logger.info(f"Using {len(server_urls)} servers for load balancing")
                client = MultiServerBatchClient(
                    servers=server_urls,
                    max_concurrent_per_server=self.config.max_concurrent_requests,
                    batch_size=self.config.batch_size,
                    batch_timeout=self.config.batch_timeout,
                    request_timeout=float(self.config.request_timeout_seconds),
                    recover_base_url_callback=self.config.task_model_recovery_callback,
                    recovery_cooldown_seconds=self.config.task_model_recovery_cooldown_seconds,
                    routing_policy=self.config.routing_policy,
                    metrics_collector=metrics_collector,
                    call_sink=self.config.call_trace_sink,
                )
            else:
                client = AsyncBatchLLMClient(
                    base_url=server_urls[0],
                    max_concurrent=self.config.max_concurrent_requests,
                    batch_size=self.config.batch_size,
                    batch_timeout=self.config.batch_timeout,
                    request_timeout=float(self.config.request_timeout_seconds),
                    recover_base_url_callback=self.config.task_model_recovery_callback,
                    recovery_cooldown_seconds=self.config.task_model_recovery_cooldown_seconds,
                    call_sink=self.config.call_trace_sink,
                )
        elif (
            llm_score_requested
            and self.config.prompt_builders.score
            and self.config.score_parser
        ):
            # External strategies (e.g., DSPyStrategy) own summarize/merge calls,
            # but we still need an LLM scoring client for representation backend "llm".
            server_urls = self.config.task_model_urls or [self.config.task_model_url]
            if len(server_urls) > 1:
                logger.info(
                    "External strategy in use; creating dedicated LLM scoring client across %d servers",
                    len(server_urls),
                )
                llm_scoring_client = MultiServerBatchClient(
                    servers=server_urls,
                    max_concurrent_per_server=self.config.max_concurrent_requests,
                    batch_size=self.config.batch_size,
                    batch_timeout=self.config.batch_timeout,
                    request_timeout=float(self.config.request_timeout_seconds),
                    recover_base_url_callback=self.config.task_model_recovery_callback,
                    recovery_cooldown_seconds=self.config.task_model_recovery_cooldown_seconds,
                    routing_policy=self.config.routing_policy,
                    metrics_collector=None,
                    call_sink=self.config.call_trace_sink,
                )
            else:
                logger.info(
                    "External strategy in use; creating dedicated LLM scoring client"
                )
                llm_scoring_client = AsyncBatchLLMClient(
                    base_url=server_urls[0],
                    max_concurrent=self.config.max_concurrent_requests,
                    batch_size=self.config.batch_size,
                    batch_timeout=self.config.batch_timeout,
                    request_timeout=float(self.config.request_timeout_seconds),
                    recover_base_url_callback=self.config.task_model_recovery_callback,
                    recovery_cooldown_seconds=self.config.task_model_recovery_cooldown_seconds,
                    call_sink=self.config.call_trace_sink,
                )

        results = []

        # Build stable, unique document IDs even when samples do not expose `doc_id`.
        sample_doc_ids: List[str] = []
        seen_doc_ids: set[str] = set()
        for idx, sample in enumerate(samples):
            raw_doc_id = (
                getattr(sample, "doc_id", None)
                or getattr(sample, "manifesto_id", None)
                or getattr(sample, "id", None)
            )
            doc_id = str(raw_doc_id).strip() if raw_doc_id is not None else ""
            if not doc_id:
                doc_id = f"doc_{idx}"
            if doc_id in seen_doc_ids:
                doc_id = f"{doc_id}_{idx}"
            seen_doc_ids.add(doc_id)
            sample_doc_ids.append(doc_id)

        doc_id_by_object_id = {
            id(sample): sample_doc_ids[idx] for idx, sample in enumerate(samples)
        }
        sample_by_id = {
            sample_doc_ids[idx]: sample for idx, sample in enumerate(samples)
        }
        clear_prompt_metadata_registry(sample_doc_ids)
        for idx, sample in enumerate(samples):
            doc_id = sample_doc_ids[idx]
            register_prompt_metadata_for_doc(doc_id, _extract_metadata(sample))
        semantic_prompt_payload_by_doc: Dict[str, Dict[str, Any]] = {}
        semantic_query_vector_by_doc: Dict[str, np.ndarray] = {}
        semantic_registry_doc_ids: List[str] = []
        clear_semantic_registry_fn: Optional[Callable[[Optional[List[str]]], None]] = None

        def get_doc_id(sample: Any) -> str:
            return doc_id_by_object_id.get(id(sample), "")

        if bool(getattr(self.config.semantic_memory, "enabled", False)):
            from treepo._research.core.semantic_prompting import (
                clear_semantic_memory_registry,
                register_semantic_memory_for_doc,
            )
            clear_semantic_registry_fn = clear_semantic_memory_registry

            clear_semantic_memory_registry(sample_doc_ids)
            for idx, sample in enumerate(samples):
                doc_id = sample_doc_ids[idx]
                metadata = _extract_metadata(sample)
                payload = self._semantic_payload_for_document(
                    text=str(getattr(sample, "text", "") or ""),
                    doc_id=doc_id,
                    metadata=metadata,
                )
                if payload is None:
                    continue
                prompt_payload = {"neighbors": list(payload.get("neighbors", []))}
                register_semantic_memory_for_doc(doc_id, prompt_payload)
                semantic_prompt_payload_by_doc[doc_id] = {
                    "neighbor_count": int(payload.get("neighbor_count", 0) or 0),
                    "retrieval_ms": float(payload.get("retrieval_ms", 0.0) or 0.0),
                }
                try:
                    semantic_query_vector_by_doc[doc_id] = np.asarray(
                        payload.get("query_vector"),
                        dtype=np.float32,
                    )
                except Exception:
                    pass
                semantic_registry_doc_ids.append(doc_id)
                self._semantic_payload_by_doc_id[doc_id] = payload

        async with AsyncExitStack() as stack:
            if metrics_collector is not None:
                await stack.enter_async_context(metrics_collector)

                async def _log_loop() -> None:
                    interval = max(5.0, float(self.config.metrics_poll_seconds or 0.0))
                    while True:
                        line = metrics_collector.snapshot().summary_line()
                        if "no reachable" in line:
                            logger.debug("Inference metrics: %s", line)
                        else:
                            logger.info("Inference metrics: %s", line)
                        await asyncio.sleep(interval)

                metrics_task = asyncio.create_task(_log_loop())
                stack.push_async_callback(cancel_tasks, [metrics_task])

            if client is not None:
                await stack.enter_async_context(client)
            if llm_scoring_client is not None:
                await stack.enter_async_context(llm_scoring_client)

            # --- Shared orchestrator setup ---
            if not external_strategy:
                strategy = BatchedStrategy(
                    client=client,
                    summarize_prompt_fn=self.config.prompt_builders.summarize,
                    merge_prompt_fn=self.config.prompt_builders.merge,
                    max_tokens=self.config.max_tokens_summary,
                    await_response_timeout=float(self.config.await_response_timeout_seconds),
                )
            config = BuildConfig(
                max_chunk_chars=self.config.max_chunk_chars,
                max_chunk_tokens=self.config.max_chunk_tokens,
                chunk_token_encoding=self.config.chunk_token_encoding,
                max_concurrent_requests=self.config.max_concurrent_requests,
                task_cancel_timeout=self.config.concurrency.task_cancel_timeout,
                document_retry_delay=self.config.concurrency.document_retry_delay,
                fail_on_degenerate_summary=self.config.fail_on_degenerate_summary,
                max_degenerate_leaf_fallbacks=self.config.max_degenerate_leaf_fallbacks,
                max_degenerate_merge_fallbacks=self.config.max_degenerate_merge_fallbacks,
                runtime_mode=self.config.runtime_mode,
                batch_plan_cache_name="batched_pipeline_global",
            )
            orchestrator = BatchTreeOrchestrator(strategy=strategy, config=config)

            unified_trees: Optional[Dict[int, List[Any]]] = None

            if self.config.unified_tree:
                # --- Unified path: build embedding trees, then feed into orchestrator ---
                self._ensure_unified_components()
                unified_trees = self._build_all_unified_trees(samples, sample_doc_ids)

                build_results = await orchestrator.process_documents_unified(
                    documents=samples,
                    rubric=self.config.rubric,
                    unified_trees=unified_trees,
                    get_text_fn=lambda s: s.text,
                    get_id_fn=get_doc_id,
                    progress_callback=progress_callback,
                )

                # CTreePO post-processing on unified nodes
                self._ctreepo_postprocess(unified_trees, build_results)
            else:
                # --- Standard path: chunk text and build trees via orchestrator ---
                build_results = await orchestrator.process_documents(
                    documents=samples,
                    rubric=self.config.rubric,
                    get_text_fn=lambda s: s.text,
                    get_id_fn=get_doc_id,
                    progress_callback=progress_callback,
                )

            scores = {}
            baselines = {}
            score_request_skip_reasons: Dict[int, str] = {}
            score_response_errors: Dict[int, str] = {}
            score_parse_failures: Dict[int, str] = {}
            active_llm_score_client = client if client is not None else llm_scoring_client
            score_runtime_telemetry = BatchTelemetry(runtime_mode=self.config.runtime_mode)
            score_plan_cache = get_named_plan_cache("batched_pipeline_scores")

            if llm_score_requested and active_llm_score_client is None:
                logger.info(
                    "LLM representation backend requested but no active LLM client is available; "
                    "llm score backend will be skipped."
                )

            if (
                llm_score_requested
                and active_llm_score_client is not None
                and self.config.prompt_builders.score
                and self.config.score_parser
            ):
                # Phase 2: Score ALL documents' summaries together
                logger.info(f"Phase 4: Scoring {len(build_results)} documents...")
                score_requests = []  # [(result_idx, request, summary_text)]
                score_skips: List[Dict[str, str]] = []

                for result_idx, build_result in enumerate(build_results):
                    doc_id = build_result.tree.metadata.get('doc_id', '')
                    summary_text = clean_summary_text(build_result.tree.final_summary)
                    if is_degenerate_summary_text(summary_text):
                        summary_text = ""
                    if not summary_text:
                        leaf_summaries = []
                        for n in build_result.tree.leaves:
                            candidate = clean_summary_text(n.summary)
                            if candidate and not is_degenerate_summary_text(candidate):
                                leaf_summaries.append(candidate)
                        if leaf_summaries:
                            combined = "\n\n".join(leaf_summaries)
                            summary_text = combined if len(combined) <= 2400 else combined[:2400].rstrip()
                            if build_result.tree.root is not None:
                                build_result.tree.root.summary = summary_text
                            build_result.tree.metadata["llm_score_summary_fallback"] = "leaf_concat"
                    skip_reason: Optional[str] = None
                    if build_result.errors:
                        skip_reason = "build_error"
                    elif not summary_text:
                        skip_reason = "empty_final_summary"
                    if skip_reason is not None:
                        score_request_skip_reasons[result_idx] = skip_reason
                        build_result.tree.metadata["llm_score_skipped_reason"] = skip_reason
                        score_skips.append({"doc_id": str(doc_id), "reason": skip_reason})
                        continue

                    sample_for_score = sample_by_id.get(doc_id)
                    score_metadata = _extract_metadata(sample_for_score) if sample_for_score is not None else {}
                    _score_meta_token = engram_document_metadata.set(score_metadata or None)
                    try:
                        score_messages = self.config.prompt_builders.score(
                            summary_text,
                            self.config.task_context,
                        )
                    finally:
                        engram_document_metadata.reset(_score_meta_token)

                    request = BatchRequest(
                        request_id=f"{doc_id}_score",
                        messages=score_messages,
                        max_tokens=self.config.max_tokens_score,
                        temperature=0.0,
                        document_id=doc_id,
                        request_type="score",
                    )
                    score_requests.append((result_idx, request, summary_text))

                logger.info(f"  Prepared {len(score_requests)} score requests...")
                if score_skips:
                    reason_counts = dict(Counter(item["reason"] for item in score_skips))
                    preview = ", ".join(
                        f"{item['doc_id']}({item['reason']})"
                        for item in score_skips[:10]
                    )
                    suffix = " ..." if len(score_skips) > 10 else ""
                    logger.warning(
                        "  Skipped %d/%d score requests due to missing scoring prerequisites: %s",
                        len(score_skips),
                        len(build_results),
                        reason_counts,
                    )
                    logger.warning("  Score-skip preview: %s%s", preview, suffix)

                async def _await_score_with_retries(result_idx: int, request: BatchRequest) -> Optional[float]:
                    max_attempts = 2
                    timeout_seconds = float(self.config.await_response_timeout_seconds)
                    last_error: Optional[str] = None
                    last_parse_preview: Optional[str] = None
                    current_request = request

                    for attempt in range(1, max_attempts + 1):
                        if attempt > 1:
                            current_request = BatchRequest(
                                request_id=f"{request.request_id}_retry{attempt - 1}",
                                messages=request.messages,
                                max_tokens=request.max_tokens,
                                temperature=0.0,
                                document_id=request.document_id,
                                request_type=request.request_type,
                            )
                            await active_llm_score_client.submit(current_request)

                        response = await active_llm_score_client.await_response(
                            current_request.request_id,
                            timeout=timeout_seconds,
                        )
                        if response.error:
                            last_error = str(response.error)
                            continue

                        parsed_score = self.config.score_parser(response.content)
                        if parsed_score is not None:
                            return parsed_score

                        last_parse_preview = str(response.content or "")[:240]

                    if last_error is not None:
                        score_response_errors[result_idx] = str(last_error)
                    if last_parse_preview is not None:
                        score_parse_failures[result_idx] = str(last_parse_preview)
                    return None

                async def _execute_score_requests_unified() -> None:
                    work_items: List[WorkItem] = []
                    request_lookup: Dict[str, Tuple[int, BatchRequest]] = {}
                    for result_idx, request, summary_text in score_requests:
                        item_id = f"score:{result_idx}"
                        request_lookup[item_id] = (result_idx, request)
                        estimated_tokens = max(1, len(str(summary_text or "")) // 4)
                        work_items.append(
                            WorkItem(
                                item_id=item_id,
                                backend_family="judge_score",
                                op_kind="score",
                                topology_signature="llm_score",
                                supervision_mask="score",
                                doc_id=str(request.document_id or ""),
                                payload=result_idx,
                                estimated_tokens=estimated_tokens,
                                estimated_nodes=1,
                                estimated_merge_ops=0,
                                padding_multiple=1,
                                padding_length=estimated_tokens,
                            )
                        )
                    batches = plan_work_batches(
                        work_items,
                        max_docs=max(1, int(self.config.max_concurrent_requests)),
                        max_total_tokens=0,
                        max_total_nodes=0,
                        max_total_merge_ops=0,
                        plan_cache=score_plan_cache,
                    )
                    for batch in batches:
                        batch_records = [request_lookup[item.item_id] for item in batch.items]
                        for _result_idx, request in batch_records:
                            await active_llm_score_client.submit(request)
                        parsed_scores = await asyncio.gather(
                            *(
                                _await_score_with_retries(result_idx, request)
                                for result_idx, request in batch_records
                            )
                        )
                        for (result_idx, _request), parsed in zip(batch_records, parsed_scores):
                            if parsed is not None:
                                scores[result_idx] = parsed
                        score_runtime_telemetry.add_batch(
                            batch,
                            token_budget=0,
                            node_budget=0,
                            max_docs_budget=max(1, int(self.config.max_concurrent_requests)),
                            fallback_reason="llm_score_batch",
                        )

                # Await all scores
                if self.config.runtime_mode == "unified_v2":
                    await _execute_score_requests_unified()
                else:
                    for result_idx, request, _summary_text in score_requests:
                        await active_llm_score_client.submit(request)
                    for result_idx, request, _summary_text in score_requests:
                        parsed = await _await_score_with_retries(result_idx, request)
                        if parsed is not None:
                            scores[result_idx] = parsed

                if score_response_errors or score_parse_failures:
                    logger.warning(
                        "  LLM score misses: response_errors=%d parse_failures=%d",
                        len(score_response_errors),
                        len(score_parse_failures),
                    )

                # Phase 3: Baseline scores (optional)
                if self.config.run_baseline:
                    logger.info(f"Phase 5: Computing baseline scores...")
                    baseline_requests = []

                    for result_idx, build_result in enumerate(build_results):
                        doc_id = build_result.tree.metadata.get('doc_id', '')
                        if build_result.errors:
                            continue

                        sample = sample_by_id.get(doc_id)
                        if not sample:
                            continue
                        baseline_text = sample.text  # Use full text - truncation corrupts results

                        baseline_metadata = _extract_metadata(sample) if sample is not None else {}
                        _baseline_meta_token = engram_document_metadata.set(baseline_metadata or None)
                        try:
                            baseline_messages = self.config.prompt_builders.score(
                                baseline_text,
                                self.config.task_context,
                            )
                        finally:
                            engram_document_metadata.reset(_baseline_meta_token)

                        request = BatchRequest(
                            request_id=f"{doc_id}_baseline",
                            messages=baseline_messages,
                            max_tokens=self.config.max_tokens_score,
                            temperature=0.0,
                            document_id=doc_id,
                            request_type="baseline",
                        )
                        baseline_requests.append((result_idx, request))
                        await active_llm_score_client.submit(request)

                    logger.info(f"  Submitted {len(baseline_requests)} baseline requests...")

                    for result_idx, request in baseline_requests:
                        response = await active_llm_score_client.await_response(
                            request.request_id,
                            timeout=float(self.config.await_response_timeout_seconds),
                        )
                        baselines[result_idx] = self.config.score_parser(response.content)

            # Convert BuildResults to DocumentResults
            for result_idx, build_result in enumerate(build_results):
                doc_id = build_result.tree.metadata.get('doc_id', '')
                sample = sample_by_id.get(doc_id)

                # Extract leaf summaries from tree.leaves
                leaf_summaries = [
                    leaf_node.summary or ""
                    for leaf_node in build_result.tree.leaves
                ]
                leaf_spans = [
                    leaf_node.raw_text_span or ""
                    for leaf_node in build_result.tree.leaves
                ]

                # Get original text length
                original_length = len(sample.text) if sample else 0
                final_summary = clean_summary_text(build_result.tree.final_summary or "")
                final_summary_fallback_used = False
                if is_degenerate_summary_text(final_summary):
                    final_summary = ""
                if not final_summary:
                    cleaned_leaf_summaries = []
                    for candidate in leaf_summaries:
                        cleaned = clean_summary_text(candidate)
                        if cleaned and not is_degenerate_summary_text(cleaned):
                            cleaned_leaf_summaries.append(cleaned)
                    if cleaned_leaf_summaries:
                        combined = "\n\n".join(cleaned_leaf_summaries)
                        final_summary = combined if len(combined) <= 2400 else combined[:2400].rstrip()
                        final_summary_fallback_used = True

                metadata = _extract_metadata(sample) if sample else {}
                if final_summary_fallback_used:
                    metadata["final_summary_fallback"] = "leaf_concat"
                cached_tree_path = self._persist_tree_artifact(
                    tree=build_result.tree,
                    doc_id=doc_id,
                    result_idx=result_idx,
                )
                if cached_tree_path is not None:
                    metadata["cached_tree_path"] = str(cached_tree_path)
                if build_result.tree.metadata.get("tree_plan"):
                    metadata["tree_plan"] = build_result.tree.metadata["tree_plan"]
                if build_result.tree.metadata.get("chunk_boundaries"):
                    metadata["chunk_boundaries"] = build_result.tree.metadata["chunk_boundaries"]
                if build_result.tree.metadata.get("chunking"):
                    metadata["chunking"] = build_result.tree.metadata["chunking"]
                if build_result.content_weights is not None:
                    metadata["content_weights"] = build_result.content_weights
                # Propagate unified tree / CTreePO metadata from orchestrator
                for _ukey in (
                    "unified_tree", "unified_tree_node_count",
                    "unified_tree_leaf_count", "unified_audit_candidates",
                    "ctreepo_rile", "ctreepo_sketch_dim",
                    "feedback_signal_count", "feedback_signals",
                ):
                    if build_result.tree.metadata.get(_ukey) is not None:
                        metadata[_ukey] = build_result.tree.metadata[_ukey]
                sem_prompt = semantic_prompt_payload_by_doc.get(doc_id)
                if sem_prompt is not None:
                    metadata["semantic_memory_neighbor_count"] = int(sem_prompt.get("neighbor_count", 0) or 0)
                    metadata["semantic_memory_retrieval_ms"] = float(sem_prompt.get("retrieval_ms", 0.0) or 0.0)

                llm_family = str(_LLM_TEXT_PROGRAM_SPEC.program_family)
                embedding_family = str(_SEMANTIC_EMBEDDING_PROGRAM_SPEC.program_family)
                ctreepo_family = str(_CTREEPO_PROGRAM_SPEC.program_family)
                mergeable_family = str(_MERGEABLE_SKETCH_PROGRAM_SPEC.program_family)
                backend_scores: Dict[str, float] = {}
                llm_score = _safe_float(scores.get(result_idx))
                if llm_score is not None:
                    backend_scores[llm_family] = float(llm_score)

                if ctreepo_family in requested_backend_set:
                    ctreepo_rile = _safe_float(metadata.get("ctreepo_rile"))
                    if (
                        ctreepo_rile is None
                        and sample is not None
                        and self._is_representation_backend_configured("ctreepo")
                    ):
                        try:
                            sketch, pred = self._get_ctreepo_prediction(sample.text)
                            if sketch is not None:
                                ctreepo_rile = _safe_float(pred)
                        except Exception as exc:
                            metadata["ctreepo_error"] = str(exc)
                            logger.debug("CTreePO prediction failed for %s: %s", doc_id, exc)
                    if ctreepo_rile is not None:
                        backend_scores[ctreepo_family] = float(ctreepo_rile)
                        metadata["ctreepo_rile"] = round(float(ctreepo_rile), 4)

                if (
                    mergeable_family in requested_backend_set
                    and sample is not None
                    and self._is_representation_backend_configured("mergeable_sketch")
                ):
                    try:
                        pred01, pred_rile, window_count = self._get_mergeable_sketch_prediction(
                            text=sample.text,
                            leaf_texts=leaf_spans,
                            doc_id=doc_id,
                            metadata=metadata,
                        )
                        if pred01 is not None:
                            metadata["mergeable_sketch_pred01"] = float(pred01)
                        if window_count is not None:
                            metadata["mergeable_sketch_window_count"] = int(window_count)
                        pred_rile_f = _safe_float(pred_rile)
                        if pred_rile_f is not None:
                            backend_scores[mergeable_family] = float(pred_rile_f)
                            metadata["mergeable_sketch_rile"] = round(float(pred_rile_f), 4)
                    except Exception as exc:
                        metadata["mergeable_sketch_error"] = str(exc)
                        logger.debug("Mergeable sketch prediction failed for %s: %s", doc_id, exc)

                if embedding_family in requested_backend_set:
                    sem_payload = self._semantic_payload_by_doc_id.get(doc_id)
                    if (
                        sem_payload is None
                        and sample is not None
                        and self._is_representation_backend_configured("embedding")
                    ):
                        try:
                            sem_payload = self._semantic_payload_for_document(
                                text=str(sample.text or ""),
                                doc_id=doc_id,
                                metadata=metadata,
                            )
                        except Exception as exc:
                            logger.debug("Semantic payload generation failed for %s: %s", doc_id, exc)
                            sem_payload = None
                    embedding_score, embedding_support, embedding_source = self._score_from_embedding_sources(
                        payload=sem_payload,
                        metadata=metadata,
                    )
                    metadata["semantic_program_support"] = int(embedding_support)
                    metadata["semantic_program_source"] = str(embedding_source)
                    if embedding_score is not None:
                        backend_scores[embedding_family] = float(embedding_score)
                        metadata["embedding_retrieval_score"] = round(float(embedding_score), 4)

                ensemble_diag: Dict[str, Any] | None = None
                ensemble_weights: Dict[str, float] | None = None
                if "ensemble" in requested_backend_set:
                    ensemble_weights, ensemble_diag = self._resolve_ensemble_weights(
                        backend_scores,
                        metadata=metadata,
                    )
                    if ensemble_weights:
                        ensemble_score = 0.0
                        for backend, weight in ensemble_weights.items():
                            ensemble_score += float(weight) * float(backend_scores[backend])
                        backend_scores["ensemble"] = float(ensemble_score)

                score_response_error = score_response_errors.get(result_idx)
                if score_response_error:
                    metadata["llm_score_failure_reason"] = str(score_response_error)
                score_parse_preview = score_parse_failures.get(result_idx)
                if score_parse_preview:
                    metadata["llm_score_failure_reason"] = "score_parse_failed"
                    metadata["llm_score_failure_preview"] = str(score_parse_preview)

                selected_backend, selected_score, fallback_used = self._select_representation_score(
                    backend_scores=backend_scores,
                    requested_backends=requested_backends,
                    primary_backend=primary_backend,
                )
                effective_selected_score, default_score_applied = self._resolve_effective_selected_score(
                    selected_score=selected_score,
                    metadata=metadata,
                )
                self._write_program_routing_metadata(
                    metadata=metadata,
                    requested_backends=requested_backends,
                    primary_backend=primary_backend,
                    backend_scores=backend_scores,
                    selected_backend=selected_backend,
                    selected_score=selected_score,
                    effective_selected_score=effective_selected_score,
                    fallback_used=fallback_used,
                    ensemble_diag=ensemble_diag,
                    ensemble_weights=ensemble_weights,
                )
                score_skip_reason = score_request_skip_reasons.get(result_idx)
                if score_skip_reason:
                    metadata["llm_score_skipped_reason"] = str(score_skip_reason)
                if default_score_applied and selected_backend is None:
                    metadata["selected_program_family"] = "missing_score_default"

                result_error = build_result.errors[0] if build_result.errors else None
                if (
                    result_error is None
                    and score_skip_reason is not None
                    and effective_selected_score is None
                ):
                    result_error = f"llm_scoring_skipped:{score_skip_reason}"

                result = DocumentResult(
                    doc_id=doc_id,
                    original_content=sample.text if sample else None,
                    reference_score=_extract_reference_score(sample) if sample else None,
                    original_length=original_length,
                    tree_height=build_result.tree.height,
                    tree_leaves=build_result.tree.leaf_count,
                    final_summary=final_summary,
                    summary_length=len(final_summary),
                    compression_ratio=original_length / max(len(final_summary), 1) if original_length else 1.0,
                    estimated_score=effective_selected_score,
                    baseline_score=baselines.get(result_idx),
                    error=result_error,
                    chunks=leaf_spans,
                    leaf_summaries=leaf_summaries,
                    level_history=None,  # Not available from BuildResult
                    metadata=metadata,
                )
                if bool(getattr(self.config.semantic_memory, "enabled", False)):
                    self._semantic_write_after_score(
                        result=result,
                        query_vector=semantic_query_vector_by_doc.get(doc_id),
                    )
                results.append(result)

            elapsed = time.time() - start_time
            logger.info(
                f"Global pipelined processing complete: {len(results)}/{len(samples)} in "
                f"{elapsed:.1f}s ({len(samples)/elapsed:.1f} samples/sec)"
            )

            if client is not None:
                if use_progress:
                    display_batch_summary(client.stats, title="Global Pipelined Batch Summary")
                else:
                    logger.info(f"LLM stats: {client.stats}")
                self._last_stats = client.stats
                self._last_diagnostics = getattr(client, "diagnostics", None)
            if self._last_diagnostics is None or not isinstance(self._last_diagnostics, dict):
                self._last_diagnostics = {}
            self._last_diagnostics["runtime_mode"] = str(self.config.runtime_mode)
            self._last_diagnostics["tree_build"] = dict(
                getattr(orchestrator, "_build_stats", {}) or {}
            )
            self._last_diagnostics["score_runtime"] = score_runtime_telemetry.as_dict()

            self._results.extend(results)
            if bool(getattr(self.config.semantic_memory, "enabled", False)):
                self._attach_semantic_diagnostics()
            clear_prompt_metadata_registry(sample_doc_ids)
            return results
