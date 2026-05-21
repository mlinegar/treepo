"""Shared defaults and helpers for LLM batch transport.

The repo has two users of the same underlying batching substrate:

- tree construction through ``BatchTreeOrchestrator`` / ``BatchedDocPipeline``
- DSPy optimizer calls through ``BatchedDSPyLM``

Keeping the knobs here prevents those paths from drifting back to different
batch sizes, timeouts, or endpoint parsing rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence


DEFAULT_BATCH_MAX_CONCURRENT = 512
DEFAULT_BATCH_SIZE = 64
DEFAULT_BATCH_TIMEOUT_SECONDS = 0.02
DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS = 300.0
DEFAULT_BATCH_AWAIT_RESPONSE_TIMEOUT_SECONDS = 600.0
DEFAULT_BATCH_ROUTING_POLICY = "affinity_load_aware"


@dataclass(frozen=True)
class BatchTransportDefaults:
    max_concurrent: int = DEFAULT_BATCH_MAX_CONCURRENT
    batch_size: int = DEFAULT_BATCH_SIZE
    batch_timeout: float = DEFAULT_BATCH_TIMEOUT_SECONDS
    request_timeout: float = DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS
    await_response_timeout: Optional[float] = None
    routing_policy: str = DEFAULT_BATCH_ROUTING_POLICY

    def normalized(self) -> "BatchTransportDefaults":
        await_timeout = self.await_response_timeout
        return BatchTransportDefaults(
            max_concurrent=max(1, int(self.max_concurrent)),
            batch_size=max(1, int(self.batch_size)),
            batch_timeout=max(0.0, float(self.batch_timeout)),
            request_timeout=max(1.0, float(self.request_timeout)),
            await_response_timeout=(
                None if await_timeout is None else max(1.0, float(await_timeout))
            ),
            routing_policy=str(self.routing_policy or DEFAULT_BATCH_ROUTING_POLICY),
        )


def normalize_base_urls(
    *,
    api_base: Optional[Any] = None,
    api_bases: Optional[Sequence[Any]] = None,
) -> list[str]:
    """Return clean OpenAI-compatible base URLs from scalar/list/comma input."""
    if api_bases is None and isinstance(api_base, str) and "," in api_base:
        api_bases = [part.strip() for part in api_base.split(",")]
        api_base = None

    raw_values: Sequence[Any]
    if api_bases is not None:
        raw_values = list(api_bases)
    elif api_base is not None:
        raw_values = [api_base]
    else:
        raw_values = []

    return [str(url).strip().rstrip("/") for url in raw_values if str(url).strip()]
