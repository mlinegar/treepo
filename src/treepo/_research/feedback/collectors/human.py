"""
Human feedback collector.

Enqueues FeedbackRequests into a FeedbackStore for human review via the API.
Bridges to the existing ReviewQueue/FlaggedItem infrastructure.
"""

import asyncio
import logging
import time
from typing import Any, Optional

from treepo._research.feedback.collector import register_collector
from treepo._research.feedback.types import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)


@register_collector("human")
class HumanCollector:
    """Collector that queues requests for human review.

    Has two modes:
    - **Non-blocking** (default): enqueues and returns a pending response immediately.
      The actual response arrives later via the API.
    - **Blocking**: waits (polls) until a human submits a response via the store.

    Usage:
        from treepo._research.feedback.store import FeedbackStore

        store = FeedbackStore()
        collector = HumanCollector(store=store, blocking=False)

        # Non-blocking: enqueue and get a placeholder
        response = collector.collect(request)
        assert response.source == "human_pending"

        # Later, human submits via API -> store.submit(request_id, response)
        # Retrieve completed:
        dataset = store.to_feedback_dataset()
    """

    def __init__(
        self,
        store: Optional[Any] = None,
        blocking: bool = False,
        poll_interval: float = 1.0,
        timeout: float = 300.0,
    ):
        """
        Args:
            store: FeedbackStore instance. If None, creates one.
            blocking: If True, collect() blocks until a response is submitted.
            poll_interval: Seconds between polls in blocking mode.
            timeout: Maximum seconds to wait in blocking mode.
        """
        if store is None:
            from treepo._research.feedback.store import FeedbackStore
            store = FeedbackStore()
        self.store = store
        self.blocking = blocking
        self.poll_interval = poll_interval
        self.timeout = timeout

    def collect(
        self,
        request: FeedbackRequest,
        **kwargs: Any,
    ) -> FeedbackResponse:
        """Enqueue request for human review.

        In non-blocking mode: returns a pending placeholder immediately.
        In blocking mode: polls until a response is submitted or timeout.
        """
        self.store.enqueue(request)

        if not self.blocking:
            return FeedbackResponse(
                request_id=request.request_id,
                reasoning="Awaiting human review",
                source="human_pending",
                extra={"status": "pending"},
            )

        # Blocking: poll for response
        start = time.monotonic()
        while time.monotonic() - start < self.timeout:
            completed = self.store.get_completed(limit=10000)
            for req, resp in completed:
                if req.request_id == request.request_id:
                    return resp
            time.sleep(self.poll_interval)

        # Timeout
        return FeedbackResponse(
            request_id=request.request_id,
            reasoning=f"Human review timed out after {self.timeout}s",
            source="human_timeout",
            extra={"status": "timeout"},
        )

    async def collect_async(
        self,
        request: FeedbackRequest,
        **kwargs: Any,
    ) -> FeedbackResponse:
        """Async version: enqueue and optionally wait."""
        self.store.enqueue(request)

        if not self.blocking:
            return FeedbackResponse(
                request_id=request.request_id,
                reasoning="Awaiting human review",
                source="human_pending",
                extra={"status": "pending"},
            )

        # Async polling
        start = time.monotonic()
        while time.monotonic() - start < self.timeout:
            completed = self.store.get_completed(limit=10000)
            for req, resp in completed:
                if req.request_id == request.request_id:
                    return resp
            await asyncio.sleep(self.poll_interval)

        return FeedbackResponse(
            request_id=request.request_id,
            reasoning=f"Human review timed out after {self.timeout}s",
            source="human_timeout",
            extra={"status": "timeout"},
        )
