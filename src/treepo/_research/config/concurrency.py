"""
Centralized concurrency configuration for the OPS framework.

This module consolidates all threading/concurrency settings to avoid
magic numbers scattered throughout the codebase and prevent thread explosion
from nested ThreadPoolExecutors.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConcurrencyConfig:
    """
    Centralized configuration for all concurrency-related settings.

    This prevents the "nested ThreadPoolExecutor explosion" problem where
    parent executors spawn child executors, leading to O(parent × child) threads.

    Guidelines:
    - Total thread count should not exceed 2-4× CPU cores for IO-bound work
    - For nested execution, child_max_workers × parent_max_workers should be reasonable
    - Use shared_pool_size when possible to avoid nested pools entirely

    Example:
        config = ConcurrencyConfig()
        # For document processing (parent level)
        with ThreadPoolExecutor(max_workers=config.document_workers) as executor:
            # For chunk processing within documents (child level)
            # Use config.chunk_workers_per_doc which is capped
            ...
    """

    # ==========================================================================
    # Document-level concurrency (outer loop)
    # ==========================================================================

    # Maximum concurrent documents being processed
    # Canonical value aligned with BatchedPipelineConfig
    max_concurrent_documents: int = 30

    # ==========================================================================
    # Chunk-level concurrency (inner loop - CAPPED to prevent explosion)
    # ==========================================================================

    # Maximum workers per document for chunk summarization
    # IMPORTANT: This is intentionally low to prevent thread explosion
    # With 20 concurrent docs × 8 workers = 160 threads max
    chunk_workers_per_doc: int = 8

    # Maximum workers per document for merge operations
    merge_workers_per_doc: int = 4

    # ==========================================================================
    # Audit concurrency
    # ==========================================================================

    # Workers for parallel audit checks (sufficiency, merge, etc.)
    audit_max_workers: int = 20

    # ==========================================================================
    # LLM client concurrency
    # Canonical values aligned with BatchedPipelineConfig (src/pipelines/batched.py)
    # ==========================================================================

    # Maximum concurrent HTTP requests to LLM server
    max_concurrent_requests: int = 200

    # Batch size for request batching (smaller batches = lower latency)
    batch_size: int = 50

    # Batch timeout in seconds (how long to wait to fill a batch)
    # 20ms provides good balance between batching efficiency and latency
    batch_timeout: float = 0.02

    # ==========================================================================
    # Optimization concurrency
    # ==========================================================================

    # Threads for parallel metric evaluation during DSPy optimization
    optimizer_num_threads: int = 128

    # Maximum concurrent candidate evaluations
    optimizer_parallel_candidates: int = 4

    # Workers for oracle pre-caching
    precache_max_workers: int = 256

    # ==========================================================================
    # Timeouts - Operational
    # ==========================================================================

    # Base timeout for leaf summarization (5 minutes).
    # Actual timeout scales: max(base, per_chunk * num_chunks)
    # Conservative to prevent timeouts during large batch processing.
    # Typical chunk summarization takes 10-30s, but network issues can cause spikes.
    leaf_summarization_timeout: float = 300.0

    # Time budget per chunk during leaf summarization.
    # 120s is generous; typical chunk summarization takes 10-30s.
    # Scales with document complexity (more chunks = higher total timeout).
    # Formula: actual_timeout = max(base_timeout, per_chunk * num_chunks)
    leaf_timeout_per_chunk: float = 120.0

    # Base timeout for merge operations (seconds)
    merge_timeout: float = 300.0  # 5 minutes

    # Timeout per merge operation (seconds)
    merge_timeout_per_pair: float = 90.0  # 90 seconds per merge pair

    # Timeout per document processing (seconds)
    document_timeout: float = 600.0  # 10 minutes

    # ==========================================================================
    # Timeouts - Async Cleanup
    # These control how async tasks are cleaned up to prevent task pileup
    # ==========================================================================

    # Timeout for waiting on task cancellation during cleanup
    # Used when close() is called on async clients
    task_cancel_timeout: float = 30.0

    # Timeout for waiting on session cleanup (aiohttp)
    session_close_timeout: float = 10.0

    # Timeout for awaiting response after submit()
    # Used by AsyncBatchLLMClient.await_response() and similar
    await_response_timeout: float = 600.0  # 10 minutes

    # Timeout for GenRM requests (larger model = slower)
    genrm_request_timeout: float = 600.0  # 10 minutes

    # Batch fill timeout (how long to wait for batch to fill before sending)
    # Lower values = lower latency, higher values = better batching
    batch_fill_timeout: float = 0.1  # 100ms

    # ==========================================================================
    # Retry Settings
    # These control automatic retry behavior for failed documents
    # ==========================================================================

    # Maximum number of retry attempts for failed documents
    # Set to 0 to disable retries
    document_max_retries: int = 2

    # Delay in seconds between retry attempts
    # Helps avoid hammering the server when errors are transient
    document_retry_delay: float = 1.0

    # ==========================================================================
    # Computed properties
    # ==========================================================================

    @property
    def max_total_threads(self) -> int:
        """Theoretical maximum threads if all pools are fully utilized."""
        return self.max_concurrent_documents * max(
            self.chunk_workers_per_doc,
            self.merge_workers_per_doc
        )

    def get_chunk_workers(self, num_chunks: int) -> int:
        """Get appropriate worker count for chunk processing."""
        return min(num_chunks, self.chunk_workers_per_doc)

    def get_merge_workers(self, num_pairs: int) -> int:
        """Get appropriate worker count for merge processing."""
        return min(num_pairs, self.merge_workers_per_doc)

    def get_leaf_timeout(self, num_chunks: int) -> float:
        """
        Get timeout for leaf summarization that scales with chunk count.

        Returns the maximum of:
        - Base timeout (leaf_summarization_timeout)
        - Per-chunk timeout * number of chunks

        This prevents timeout errors when processing documents with many chunks.
        """
        scaled_timeout = self.leaf_timeout_per_chunk * num_chunks
        return max(self.leaf_summarization_timeout, scaled_timeout)

    def get_merge_timeout(self, num_pairs: int) -> float:
        """
        Get timeout for merge operations that scales with pair count.

        Returns the maximum of:
        - Base timeout (merge_timeout)
        - Per-pair timeout * number of pairs
        """
        scaled_timeout = self.merge_timeout_per_pair * num_pairs
        return max(self.merge_timeout, scaled_timeout)

    def validate(self) -> None:
        """Validate configuration and warn about potential issues."""
        import logging
        logger = logging.getLogger(__name__)

        max_threads = self.max_total_threads
        if max_threads > 500:
            logger.warning(
                f"High thread count possible: {max_threads} threads "
                f"({self.max_concurrent_documents} docs × "
                f"{max(self.chunk_workers_per_doc, self.merge_workers_per_doc)} workers). "
                f"Consider reducing max_concurrent_documents or chunk_workers_per_doc."
            )


# Global default configuration
# Import this and modify as needed, or create new instances
DEFAULT_CONCURRENCY = ConcurrencyConfig()


def get_concurrency_config() -> ConcurrencyConfig:
    """Get the default concurrency configuration."""
    return DEFAULT_CONCURRENCY


def create_low_resource_config() -> ConcurrencyConfig:
    """Create configuration for low-resource environments."""
    return ConcurrencyConfig(
        max_concurrent_documents=5,
        chunk_workers_per_doc=4,
        merge_workers_per_doc=2,
        audit_max_workers=8,
        max_concurrent_requests=20,
        optimizer_num_threads=32,
    )


def create_high_throughput_config() -> ConcurrencyConfig:
    """Create configuration for maximum throughput on powerful hardware."""
    return ConcurrencyConfig(
        max_concurrent_documents=50,
        chunk_workers_per_doc=8,  # Still capped to prevent explosion
        merge_workers_per_doc=4,
        audit_max_workers=32,
        max_concurrent_requests=200,
        batch_size=200,
        optimizer_num_threads=256,
        precache_max_workers=512,
    )
