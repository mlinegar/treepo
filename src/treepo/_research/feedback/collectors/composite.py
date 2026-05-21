"""
Composite feedback collector.

Combines multiple collectors, aggregating their responses into a single
FeedbackResponse. Useful for getting complementary feedback -- e.g.,
LLM judge for preference + oracle for scalar score.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from treepo._research.feedback.collector import FeedbackCollector, register_collector
from treepo._research.feedback.types import FeedbackRequest, FeedbackResponse
from treepo._research.core.async_utils import to_thread

logger = logging.getLogger(__name__)


@register_collector("composite")
class CompositeCollector:
    """Combines multiple collectors, merging their responses.

    Each sub-collector produces a FeedbackResponse. The composite merges them
    using a simple priority rule: the first collector's preference/confidence
    wins, but scores and critiques are merged from all collectors.

    Usage:
        from treepo._research.feedback import get_collector

        oracle = get_collector("oracle", oracle_predict=my_fn)
        judge = get_collector("llm_judge", judge=my_judge)

        composite = CompositeCollector(
            collectors=[
                ("judge", judge),
                ("oracle", oracle),
            ],
            preference_source="judge",  # Use judge's preference
        )
        response = composite.collect(request)
        # response.preferred comes from judge
        # response.scores has both judge and oracle scores
    """

    def __init__(
        self,
        collectors: List[Tuple[str, Any]],
        preference_source: Optional[str] = None,
    ):
        """
        Args:
            collectors: List of (name, collector) tuples.
            preference_source: Name of the collector whose preference/confidence
                to use. If None, uses the first collector that provides one.
        """
        self.collectors = collectors
        self.preference_source = preference_source

    def collect(
        self,
        request: FeedbackRequest,
        **kwargs: Any,
    ) -> FeedbackResponse:
        """Collect feedback from all sub-collectors and merge responses."""
        responses: Dict[str, FeedbackResponse] = {}

        for name, collector in self.collectors:
            try:
                resp = collector.collect(request, **kwargs)
                responses[name] = resp
            except Exception as e:
                logger.warning("Collector '%s' failed: %s", name, e)

        if not responses:
            return FeedbackResponse(
                request_id=request.request_id,
                reasoning="All sub-collectors failed",
                source="composite",
            )

        return self._merge(request, responses)

    def _merge(
        self,
        request: FeedbackRequest,
        responses: Dict[str, FeedbackResponse],
    ) -> FeedbackResponse:
        """Merge multiple FeedbackResponses into one."""
        # Determine which response provides preference
        pref_resp: Optional[FeedbackResponse] = None
        if self.preference_source and self.preference_source in responses:
            pref_resp = responses[self.preference_source]
        else:
            # First response with a preference
            for resp in responses.values():
                if resp.preferred is not None:
                    pref_resp = resp
                    break

        # Merge scores: prefix each collector's scores with its name
        merged_scores: Dict[str, float] = {}
        for name, resp in responses.items():
            for score_key, score_val in resp.scores.items():
                merged_scores[f"{name}_{score_key}"] = score_val
            # Also add un-prefixed scores from the preference source
            if resp is pref_resp:
                merged_scores.update(resp.scores)

        # Merge critiques
        critiques = []
        for name, resp in responses.items():
            if resp.critique:
                critiques.append(f"[{name}] {resp.critique}")
        merged_critique = "\n".join(critiques) if critiques else ""

        # Merge reasoning
        reasonings = []
        for name, resp in responses.items():
            if resp.reasoning:
                reasonings.append(f"[{name}] {resp.reasoning}")
        merged_reasoning = "\n".join(reasonings) if reasonings else ""

        # Use pref_resp for preference fields, fallback to first response
        base = pref_resp or next(iter(responses.values()))

        # Collect all score estimates
        score_a = base.score_estimate_a
        score_b = base.score_estimate_b
        for resp in responses.values():
            if score_a is None and resp.score_estimate_a is not None:
                score_a = resp.score_estimate_a
            if score_b is None and resp.score_estimate_b is not None:
                score_b = resp.score_estimate_b

        # Source list
        source_names = list(responses.keys())

        return FeedbackResponse(
            request_id=request.request_id,
            preferred=base.preferred,
            scores=merged_scores,
            critique=merged_critique,
            reasoning=merged_reasoning,
            confidence=base.confidence,
            score_estimate_a=score_a,
            score_estimate_b=score_b,
            extra={
                "sub_collectors": source_names,
                "sub_responses": {
                    name: resp.to_dict() for name, resp in responses.items()
                },
            },
            source="composite",
            judge_model=base.judge_model,
        )

    async def collect_async(
        self,
        request: FeedbackRequest,
        **kwargs: Any,
    ) -> FeedbackResponse:
        """Async collection: run all sub-collectors concurrently."""
        responses: Dict[str, FeedbackResponse] = {}

        async def _run(name: str, collector: Any) -> Tuple[str, Optional[FeedbackResponse]]:
            try:
                if hasattr(collector, "collect_async"):
                    resp = await collector.collect_async(request, **kwargs)
                else:
                    resp = await to_thread(collector.collect, request, **kwargs)
                return name, resp
            except Exception as e:
                logger.warning("Collector '%s' failed (async): %s", name, e)
                return name, None

        tasks = [_run(name, collector) for name, collector in self.collectors]
        results = await asyncio.gather(*tasks)

        for name, resp in results:
            if resp is not None:
                responses[name] = resp

        if not responses:
            return FeedbackResponse(
                request_id=request.request_id,
                reasoning="All sub-collectors failed",
                source="composite",
            )

        return self._merge(request, responses)
