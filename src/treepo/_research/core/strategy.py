"""
Summarization Strategy Interface.

This module provides a unified async interface for summarization operations,
allowing the same tree-building logic to work with different backends:

1. BatchedStrategy: Uses AsyncBatchLLMClient for batched inference (foundation)
2. DSPyStrategy: Wraps DSPy modules, uses batching internally
3. CallableStrategy: Wraps a sync callable with async + temperature support
4. TournamentStrategy: Wraps any strategy with tournament selection for learning

Architecture:
    Batching is ALWAYS the foundation. DSPy is an optional layer for optimization.
    All strategies support temperature and candidate generation via batching.

Usage:
    # Batched inference (foundation)
    async with AsyncBatchLLMClient(url) as client:
        strategy = BatchedStrategy(client)

    # With DSPy for optimization (uses batching internally)
    strategy = DSPyStrategy(LeafSummarizer(), MergeSummarizer())

    # With tournament selection (for learning with preference collection)
    strategy = TournamentStrategy(
        base=BatchedStrategy(client),
        judge=LargeJudgeComparisonModule(),
    )

    # Same tree-building code works with all:
    summary = await strategy.summarize(content, rubric)
    merged = await strategy.merge(left, right, rubric)

    # Generate diverse candidates (all strategies support this)
    candidates = await strategy.generate_candidates(content, rubric, k=4)
"""

import asyncio
import logging
import random
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Protocol, TYPE_CHECKING, Literal, Callable, Mapping, Sequence

if TYPE_CHECKING:
    import dspy
    from treepo._research.core.batch_processor import AsyncBatchLLMClient, BatchRequest
    from treepo._research.training.judges.base import BaseJudge
    from treepo._research.training.supervision import BinaryComparison, ComparativeJudgment

from treepo._research.core.prompting import default_merge_prompt, default_summarize_prompt, default_unified_prompt
from treepo._research.core.prompting import clean_summary_text
from treepo._research.core.protocols import format_merge_input
from treepo._research.core.conditional_memory import canonical_hash
from treepo._research.core.async_utils import to_thread
from treepo._research.config.concurrency import get_concurrency_config

logger = logging.getLogger(__name__)

# Context for routing tournament preferences (e.g., batch doc IDs).
tournament_doc_id: ContextVar[Optional[str]] = ContextVar("tournament_doc_id", default=None)


def _filter_valid_candidates(
    results: List[Any],
    operation: str = "generation",
    *,
    warn_on_empty: bool = True,
) -> List[str]:
    """Filter async gather results to valid non-empty string candidates.

    Args:
        results: Results from asyncio.gather with return_exceptions=True
        operation: Description for logging (e.g., "Candidate generation", "Merge candidate")

    Returns:
        List of valid non-empty string candidates
    """
    candidates = []
    exception_count = 0
    non_string_count = 0
    exception_samples: List[str] = []
    for result in results:
        if isinstance(result, str) and result.strip():
            candidates.append(result)
        elif isinstance(result, Exception):
            exception_count += 1
            if len(exception_samples) < 3:
                detail = str(result).replace("\n", " ").strip()
                if len(detail) > 180:
                    detail = detail[:177] + "..."
                exception_samples.append(f"{type(result).__name__}: {detail}")
            logger.debug(f"{operation} failed: {result}")
        else:
            non_string_count += 1

    if (not candidates) and warn_on_empty:
        warn_budget = int(getattr(_filter_valid_candidates, "_warn_budget", 10))
        message = (
            "%s produced 0 valid candidates (exceptions=%d non_strings=%d total=%d). "
            "Downstream tournament preferences will be skipped."
        )
        if warn_budget > 0:
            logger.warning(
                message,
                operation,
                int(exception_count),
                int(non_string_count),
                int(len(results)),
            )
            if exception_samples:
                logger.warning(
                    "%s sample exceptions: %s",
                    operation,
                    " | ".join(exception_samples),
                )
            setattr(_filter_valid_candidates, "_warn_budget", warn_budget - 1)
        else:
            logger.debug(
                message,
                operation,
                int(exception_count),
                int(non_string_count),
                int(len(results)),
            )
    return candidates


# =============================================================================
# Strategy Protocol
# =============================================================================

class SummarizationStrategy(Protocol):
    """
    Protocol for summarization strategies.

    All strategies support:
    - temperature parameter for controlling diversity
    - candidate generation via batching at high temperature
    """

    async def summarize(
        self, content: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """Summarize content according to the rubric."""
        ...

    async def merge(
        self, left: str, right: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """Merge two summaries into one."""
        ...

    async def generate_candidates(
        self, content: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """
        Generate k diverse candidate summaries.

        Uses batching at high temperature for diversity.
        """
        ...

    async def generate_merge_candidates(
        self, left: str, right: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """
        Generate k diverse merge candidates.

        Uses batching at high temperature for diversity.
        """
        ...


# =============================================================================
# Batched Strategy (Foundation)
# =============================================================================

class BatchedStrategy:
    """
    Strategy using AsyncBatchLLMClient for batched inference.

    This is the FOUNDATION strategy - all other strategies build on batching.
    Submits requests to the batch client which handles pooling for optimal
    GPU utilization.

    Args:
        client: AsyncBatchLLMClient instance (must be started)
        max_tokens: Maximum tokens for summary responses
        summarize_prompt_fn: Function to build summarize prompts
        merge_prompt_fn: Function to build merge prompts
        unified_mode: If True, use single g function for both leaf and merge.
            This aligns with the theory where g : Strings -> Strings handles
            both cases. For merges, input is format_merge_input(left, right).
    """

    def __init__(
        self,
        client: "AsyncBatchLLMClient",
        max_tokens: int = 500,
        summarize_prompt_fn=None,
        merge_prompt_fn=None,
        unified_mode: bool = False,
        await_response_timeout: Optional[float] = None,
        disable_thinking: bool = True,
    ):
        self.client = client
        self.max_tokens = max_tokens
        self._counter = 0
        self.unified_mode = unified_mode
        self.disable_thinking = bool(disable_thinking)
        if await_response_timeout is None:
            self.await_response_timeout = None
        else:
            self.await_response_timeout = max(1.0, float(await_response_timeout))

        # Use default prompt builders if not provided
        if unified_mode:
            # Single g function for both leaf and merge
            self.summarize_prompt_fn = summarize_prompt_fn or default_unified_prompt
            self.merge_prompt_fn = None  # Not used in unified mode
        else:
            # Separate prompts (legacy mode)
            self.summarize_prompt_fn = summarize_prompt_fn or default_summarize_prompt
            self.merge_prompt_fn = merge_prompt_fn or default_merge_prompt

    async def _await_response(self, request_id: str):
        if self.await_response_timeout is None:
            return await self.client.await_response(request_id)
        return await self.client.await_response(
            request_id,
            timeout=self.await_response_timeout,
        )

    async def summarize(
        self, content: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """Summarize content using batched LLM client."""
        outputs = await self.summarize_many(
            [
                {
                    "content": content,
                    "rubric": rubric,
                    "temperature": temperature,
                    "doc_id": tournament_doc_id.get(),
                }
            ]
        )
        return outputs[0] if outputs else ""

    async def merge(
        self, left: str, right: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """Merge summaries using batched LLM client."""
        outputs = await self.merge_many(
            [
                {
                    "left": left,
                    "right": right,
                    "rubric": rubric,
                    "temperature": temperature,
                    "doc_id": tournament_doc_id.get(),
                }
            ]
        )
        return outputs[0] if outputs else ""

    def _make_summarize_request(
        self,
        *,
        content: str,
        rubric: str,
        temperature: float,
        doc_id: Optional[Any] = None,
    ):
        from treepo._research.core.batch_processor import BatchRequest

        self._counter += 1
        chat_template_kwargs = {"enable_thinking": False} if self.disable_thinking else None
        return BatchRequest(
            request_id=f"strategy_summarize_{self._counter}",
            messages=self.summarize_prompt_fn(content, rubric),
            max_tokens=self.max_tokens,
            request_type="summarize",
            temperature=temperature,
            document_id=str(doc_id) if doc_id is not None else None,
            chat_template_kwargs=chat_template_kwargs,
        )

    def _make_merge_request(
        self,
        *,
        left: str,
        right: str,
        rubric: str,
        temperature: float,
        doc_id: Optional[Any] = None,
    ):
        from treepo._research.core.batch_processor import BatchRequest

        self._counter += 1
        chat_template_kwargs = {"enable_thinking": False} if self.disable_thinking else None

        if self.unified_mode:
            combined = format_merge_input(left, right)
            messages = self.summarize_prompt_fn(combined, rubric)
        else:
            messages = self.merge_prompt_fn(left, right, rubric)

        return BatchRequest(
            request_id=f"strategy_merge_{self._counter}",
            messages=messages,
            max_tokens=self.max_tokens,
            request_type="merge",
            temperature=temperature,
            document_id=str(doc_id) if doc_id is not None else None,
            chat_template_kwargs=chat_template_kwargs,
        )

    async def _run_requests(self, requests: Sequence["BatchRequest"]) -> List[str]:
        if not requests:
            return []
        for request in requests:
            await self.client.submit(request)
        responses = await asyncio.gather(
            *(self._await_response(request.request_id) for request in requests),
            return_exceptions=True,
        )
        outputs: List[str] = []
        for response in responses:
            if isinstance(response, Exception):
                outputs.append("")
                continue
            outputs.append(clean_summary_text(response.content) if not response.error else "")
        return outputs

    async def summarize_many(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> List[str]:
        requests = [
            self._make_summarize_request(
                content=str(item.get("content", "") or ""),
                rubric=str(item.get("rubric", "") or ""),
                temperature=float(item.get("temperature", 0.7) or 0.7),
                doc_id=item.get("doc_id"),
            )
            for item in items
        ]
        return await self._run_requests(requests)

    async def merge_many(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> List[str]:
        requests = [
            self._make_merge_request(
                left=str(item.get("left", "") or ""),
                right=str(item.get("right", "") or ""),
                rubric=str(item.get("rubric", "") or ""),
                temperature=float(item.get("temperature", 0.7) or 0.7),
                doc_id=item.get("doc_id"),
            )
            for item in items
        ]
        return await self._run_requests(requests)

    async def _generate_candidates_impl(
        self,
        messages: List[Dict[str, str]],
        request_type: str,
        k: int,
        temperature: float,
    ) -> List[str]:
        """Common implementation for candidate generation.

        Args:
            messages: Prompt messages to send for each candidate
            request_type: Type identifier for requests (e.g., "candidate", "merge_candidate")
            k: Number of candidates to generate
            temperature: Sampling temperature for diversity

        Returns:
            List of generated candidate strings
        """
        from treepo._research.core.batch_processor import BatchRequest

        # Submit k requests in parallel
        requests = []
        doc_id = tournament_doc_id.get()
        chat_template_kwargs = {"enable_thinking": False} if self.disable_thinking else None
        for _ in range(k):
            self._counter += 1
            request = BatchRequest(
                request_id=f"strategy_{request_type}_{self._counter}",
                messages=messages,
                max_tokens=self.max_tokens,
                request_type=request_type,
                temperature=temperature,
                document_id=str(doc_id) if doc_id is not None else None,
                chat_template_kwargs=chat_template_kwargs,
            )
            requests.append(request)
            await self.client.submit(request)

        # Await all responses
        candidates = []
        for request in requests:
            response = await self._await_response(request.request_id)
            if response.content and not response.error:
                candidate = clean_summary_text(response.content)
                if candidate:
                    candidates.append(candidate)

        return candidates

    async def generate_candidates(
        self, content: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """Generate k diverse candidates via batched requests at high temperature."""
        return await self._generate_candidates_impl(
            messages=self.summarize_prompt_fn(content, rubric),
            request_type="candidate",
            k=k,
            temperature=temperature,
        )

    async def generate_merge_candidates(
        self, left: str, right: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """Generate k diverse merge candidates via batched requests at high temperature."""
        if self.unified_mode:
            # Use same prompt as leaf, just format input differently
            combined = format_merge_input(left, right)
            messages = self.summarize_prompt_fn(combined, rubric)
        else:
            messages = self.merge_prompt_fn(left, right, rubric)

        return await self._generate_candidates_impl(
            messages=messages,
            request_type="merge_candidate",
            k=k,
            temperature=temperature,
        )

    async def generate_candidates_many(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> List[List[str]]:
        from treepo._research.core.batch_processor import BatchRequest

        requests: List[BatchRequest] = []
        item_ranges: List[tuple[int, int]] = []
        chat_template_kwargs = {"enable_thinking": False} if self.disable_thinking else None
        for item in items:
            start = len(requests)
            messages = self.summarize_prompt_fn(
                str(item.get("content", "") or ""),
                str(item.get("rubric", "") or ""),
            )
            k = max(1, int(item.get("k", 4) or 4))
            temperature = float(item.get("temperature", 0.9) or 0.9)
            doc_id = item.get("doc_id")
            for _ in range(k):
                self._counter += 1
                requests.append(
                    BatchRequest(
                        request_id=f"strategy_candidate_{self._counter}",
                        messages=messages,
                        max_tokens=self.max_tokens,
                        request_type="candidate",
                        temperature=temperature,
                        document_id=str(doc_id) if doc_id is not None else None,
                        chat_template_kwargs=chat_template_kwargs,
                    )
                )
            item_ranges.append((start, len(requests)))
        outputs = await self._run_requests(requests)
        return [
            _filter_valid_candidates(outputs[start:end], "Candidate generation", warn_on_empty=False)
            for start, end in item_ranges
        ]

    async def generate_merge_candidates_many(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> List[List[str]]:
        from treepo._research.core.batch_processor import BatchRequest

        requests: List[BatchRequest] = []
        item_ranges: List[tuple[int, int]] = []
        chat_template_kwargs = {"enable_thinking": False} if self.disable_thinking else None
        for item in items:
            left = str(item.get("left", "") or "")
            right = str(item.get("right", "") or "")
            rubric = str(item.get("rubric", "") or "")
            if self.unified_mode:
                messages = self.summarize_prompt_fn(format_merge_input(left, right), rubric)
            else:
                messages = self.merge_prompt_fn(left, right, rubric)
            start = len(requests)
            k = max(1, int(item.get("k", 4) or 4))
            temperature = float(item.get("temperature", 0.9) or 0.9)
            doc_id = item.get("doc_id")
            for _ in range(k):
                self._counter += 1
                requests.append(
                    BatchRequest(
                        request_id=f"strategy_merge_candidate_{self._counter}",
                        messages=messages,
                        max_tokens=self.max_tokens,
                        request_type="merge_candidate",
                        temperature=temperature,
                        document_id=str(doc_id) if doc_id is not None else None,
                        chat_template_kwargs=chat_template_kwargs,
                    )
                )
            item_ranges.append((start, len(requests)))
        outputs = await self._run_requests(requests)
        return [
            _filter_valid_candidates(outputs[start:end], "Merge candidate generation", warn_on_empty=False)
            for start, end in item_ranges
        ]


# =============================================================================
# DSPy Strategy (wraps DSPy modules, can use batching internally)
# =============================================================================

class DSPyStrategy:
    """
    Strategy that wraps DSPy modules in async interface.

    Runs DSPy module calls in a thread pool to avoid blocking the event loop.
    For candidate generation, uses parallel calls at high temperature.

    Args:
        leaf_module: DSPy module for leaf summarization (content, rubric) -> str
        merge_module: DSPy module for merge summarization (left, right, rubric) -> str.
            Optional in unified mode.
        unified_mode: If True, use single module (leaf_module) for both leaf and merge.
            In this mode, merge operations format input via format_merge_input() and
            call leaf_module with content=formatted_input. This aligns with the theory
            where g : Strings -> Strings handles both cases.
    """

    def __init__(
        self,
        leaf_module: "dspy.Module",
        merge_module: Optional["dspy.Module"] = None,
        unified_mode: bool = False,
        *,
        default_temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        self.leaf_module = leaf_module
        self.unified_mode = unified_mode
        self.default_temperature = float(default_temperature)
        self.max_tokens = None if max_tokens is None else int(max_tokens)

        if unified_mode:
            # Use same module for both - merge just formats input differently
            self.merge_module = leaf_module
        else:
            # Legacy mode: separate modules (use leaf as fallback if merge not provided)
            self.merge_module = merge_module if merge_module is not None else leaf_module

    async def summarize(
        self, content: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """Summarize content using DSPy leaf module."""
        # TreeBuilder calls summarize() without a temperature argument; use the
        # configured default in that common case.
        if temperature == 0.7:
            temperature = self.default_temperature
        return await to_thread(
            self._call_with_temp,
            self.leaf_module,
            temperature,
            content=content,
            rubric=rubric,
        )

    async def merge(
        self, left: str, right: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """Merge summaries using DSPy module."""
        if temperature == 0.7:
            temperature = self.default_temperature
        if self.unified_mode:
            # Use same module as leaf, just format input differently
            # This is the theory's g(s_L * s_R) where * is format_merge_input
            combined = format_merge_input(left, right)
            return await to_thread(
                self._call_with_temp,
                self.leaf_module,
                temperature,
                content=combined,
                rubric=rubric,
            )
        else:
            # Legacy: use merge module with separate fields
            return await to_thread(
                self._call_with_temp,
                self.merge_module,
                temperature,
                left_summary=left,
                right_summary=right,
                rubric=rubric,
            )

    async def generate_candidates(
        self, content: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """
        Generate k diverse candidates using parallel DSPy calls at high temperature.
        """
        # Launch k calls in parallel
        tasks = [
            to_thread(
                self._call_with_temp,
                self.leaf_module,
                temperature,
                content=content,
                rubric=rubric,
            )
            for _ in range(k)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        candidates = _filter_valid_candidates(results, "Candidate generation", warn_on_empty=False)
        if candidates:
            return candidates

        # Fallback: try a single low-parallelism call so tree-building can proceed
        # even if concurrent candidate generation hits timeouts under load.
        try:
            summary = await self.summarize(content, rubric, temperature=self.default_temperature)
        except Exception as exc:
            _filter_valid_candidates(results + [exc], "Candidate generation")
            return []
        if isinstance(summary, str) and summary.strip():
            return [summary]

        _filter_valid_candidates(results, "Candidate generation")
        return []

    async def generate_merge_candidates(
        self, left: str, right: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """
        Generate k diverse merge candidates using parallel DSPy calls at high temperature.
        """
        if self.unified_mode:
            # Use same module as leaf, just format input differently
            combined = format_merge_input(left, right)
            tasks = [
                to_thread(
                    self._call_with_temp,
                    self.leaf_module,
                    temperature,
                    content=combined,
                    rubric=rubric,
                )
                for _ in range(k)
            ]
        else:
            tasks = [
                to_thread(
                    self._call_with_temp,
                    self.merge_module,
                    temperature,
                    left_summary=left,
                    right_summary=right,
                    rubric=rubric,
                )
                for _ in range(k)
            ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        candidates = _filter_valid_candidates(results, "Merge candidate generation", warn_on_empty=False)
        if candidates:
            return candidates

        # Fallback: single merge call to avoid empty merges when the parallel
        # candidate requests time out.
        try:
            merged = await self.merge(left, right, rubric, temperature=self.default_temperature)
        except Exception as exc:
            _filter_valid_candidates(results + [exc], "Merge candidate generation")
            return []
        if isinstance(merged, str) and merged.strip():
            return [merged]

        _filter_valid_candidates(results, "Merge candidate generation")
        return []

    async def summarize_many(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> List[str]:
        results = await asyncio.gather(
            *(
                self.summarize(
                    str(item.get("content", "") or ""),
                    str(item.get("rubric", "") or ""),
                    temperature=float(
                        item.get("temperature", self.default_temperature)
                        or self.default_temperature
                    ),
                )
                for item in items
            ),
            return_exceptions=True,
        )
        return [result if isinstance(result, str) else "" for result in results]

    async def merge_many(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> List[str]:
        results = await asyncio.gather(
            *(
                self.merge(
                    str(item.get("left", "") or ""),
                    str(item.get("right", "") or ""),
                    str(item.get("rubric", "") or ""),
                    temperature=float(
                        item.get("temperature", self.default_temperature)
                        or self.default_temperature
                    ),
                )
                for item in items
            ),
            return_exceptions=True,
        )
        return [result if isinstance(result, str) else "" for result in results]

    async def generate_candidates_many(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> List[List[str]]:
        results = await asyncio.gather(
            *(
                self.generate_candidates(
                    str(item.get("content", "") or ""),
                    str(item.get("rubric", "") or ""),
                    k=max(1, int(item.get("k", 4) or 4)),
                    temperature=float(item.get("temperature", 0.9) or 0.9),
                )
                for item in items
            ),
            return_exceptions=True,
        )
        return [result if isinstance(result, list) else [] for result in results]

    async def generate_merge_candidates_many(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> List[List[str]]:
        results = await asyncio.gather(
            *(
                self.generate_merge_candidates(
                    str(item.get("left", "") or ""),
                    str(item.get("right", "") or ""),
                    str(item.get("rubric", "") or ""),
                    k=max(1, int(item.get("k", 4) or 4)),
                    temperature=float(item.get("temperature", 0.9) or 0.9),
                )
                for item in items
            ),
            return_exceptions=True,
        )
        return [result if isinstance(result, list) else [] for result in results]

    def _call_with_temp(self, module, temperature: float, **kwargs) -> str:
        """Call DSPy module with specific temperature (sync, runs in thread)."""
        import dspy

        current_lm = dspy.settings.lm
        copy_kwargs = {"temperature": temperature}
        if self.max_tokens is not None:
            try:
                base_max_tokens = int(getattr(current_lm, "max_tokens", self.max_tokens) or self.max_tokens)
            except Exception:
                base_max_tokens = self.max_tokens
            copy_kwargs["max_tokens"] = int(min(self.max_tokens, base_max_tokens))

        try:
            lm_copy = current_lm.copy(**copy_kwargs)
        except TypeError:
            # Some LM implementations may not accept all copy kwargs; fall back
            # to temperature-only to preserve existing behavior.
            lm_copy = current_lm.copy(temperature=temperature)

        with dspy.context(lm=lm_copy):
            result = module(**kwargs)
            if isinstance(result, str):
                return clean_summary_text(result)
            for attr in ("summary", "merged_summary", "final_summary"):
                value = getattr(result, attr, None)
                if isinstance(value, str) and value.strip():
                    return clean_summary_text(value)
            return clean_summary_text(str(result))


# =============================================================================
# Callable Strategy (wraps a sync callable, temperature-aware if DSPy is configured)
# =============================================================================

class CallableStrategy:
    """
    Strategy that wraps a sync callable with the SummarizationStrategy interface.

    This is useful for integrating sync summarizers (e.g., DSPy modules) into
    async tree-building while preserving temperature control when DSPy is active.
    """

    def __init__(
        self,
        summarizer: Callable[..., Any],
        merge_fn: Optional[Callable[..., Any]] = None,
    ):
        """
        Initialize callable strategy.

        Args:
            summarizer: Sync function/content-based module (content, rubric) -> str
            merge_fn: Optional sync function for merges (left_summary, right_summary, rubric) -> str
        """
        self.summarizer = summarizer
        self.merge_fn = merge_fn

    async def summarize(
        self, content: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """Summarize content using the wrapped callable."""
        return await to_thread(
            self._call_with_temp,
            self.summarizer,
            temperature,
            content=content,
            rubric=rubric,
        )

    async def merge(
        self, left: str, right: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """Merge summaries using the wrapped callable."""
        if self.merge_fn is not None:
            return await to_thread(
                self._call_with_temp,
                self.merge_fn,
                temperature,
                left_summary=left,
                right_summary=right,
                rubric=rubric,
            )

        combined = format_merge_input(left, right)
        return await to_thread(
            self._call_with_temp,
            self.summarizer,
            temperature,
            content=combined,
            rubric=rubric,
        )

    async def generate_candidates(
        self, content: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """Generate k candidates using parallel calls at a fixed temperature."""
        tasks = [
            to_thread(
                self._call_with_temp,
                self.summarizer,
                temperature,
                content=content,
                rubric=rubric,
            )
            for _ in range(k)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        candidates = _filter_valid_candidates(results, "Candidate generation", warn_on_empty=False)
        if candidates:
            return candidates

        # Fallback: try a single low-parallelism call so tree-building can proceed
        # even if concurrent candidate generation hits timeouts under load.
        try:
            summary = await self.summarize(content, rubric, temperature=0.7)
        except Exception as exc:
            _filter_valid_candidates(results + [exc], "Candidate generation")
            return []
        if isinstance(summary, str) and summary.strip():
            return [summary]

        _filter_valid_candidates(results, "Candidate generation")
        return []

    async def generate_merge_candidates(
        self, left: str, right: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """Generate k merge candidates using parallel calls at a fixed temperature."""
        if self.merge_fn is not None:
            tasks = [
                to_thread(
                    self._call_with_temp,
                    self.merge_fn,
                    temperature,
                    left_summary=left,
                    right_summary=right,
                    rubric=rubric,
                )
                for _ in range(k)
            ]
        else:
            combined = f"{left}\n\n{right}"
            tasks = [
                to_thread(
                    self._call_with_temp,
                    self.summarizer,
                    temperature,
                    content=combined,
                    rubric=rubric,
                )
                for _ in range(k)
            ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        candidates = _filter_valid_candidates(results, "Merge candidate generation", warn_on_empty=False)
        if candidates:
            return candidates

        # Fallback: single merge call to avoid empty merges when the parallel
        # candidate requests time out.
        try:
            merged = await self.merge(left, right, rubric, temperature=0.7)
        except Exception as exc:
            _filter_valid_candidates(results + [exc], "Merge candidate generation")
            return []
        if isinstance(merged, str) and merged.strip():
            return [merged]

        _filter_valid_candidates(results, "Merge candidate generation")
        return []

    async def summarize_many(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> List[str]:
        results = await asyncio.gather(
            *(
                self.summarize(
                    str(item.get("content", "") or ""),
                    str(item.get("rubric", "") or ""),
                    temperature=float(item.get("temperature", 0.7) or 0.7),
                )
                for item in items
            ),
            return_exceptions=True,
        )
        return [result if isinstance(result, str) else "" for result in results]

    async def merge_many(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> List[str]:
        results = await asyncio.gather(
            *(
                self.merge(
                    str(item.get("left", "") or ""),
                    str(item.get("right", "") or ""),
                    str(item.get("rubric", "") or ""),
                    temperature=float(item.get("temperature", 0.7) or 0.7),
                )
                for item in items
            ),
            return_exceptions=True,
        )
        return [result if isinstance(result, str) else "" for result in results]

    async def generate_candidates_many(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> List[List[str]]:
        results = await asyncio.gather(
            *(
                self.generate_candidates(
                    str(item.get("content", "") or ""),
                    str(item.get("rubric", "") or ""),
                    k=max(1, int(item.get("k", 4) or 4)),
                    temperature=float(item.get("temperature", 0.9) or 0.9),
                )
                for item in items
            ),
            return_exceptions=True,
        )
        return [result if isinstance(result, list) else [] for result in results]

    async def generate_merge_candidates_many(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> List[List[str]]:
        results = await asyncio.gather(
            *(
                self.generate_merge_candidates(
                    str(item.get("left", "") or ""),
                    str(item.get("right", "") or ""),
                    str(item.get("rubric", "") or ""),
                    k=max(1, int(item.get("k", 4) or 4)),
                    temperature=float(item.get("temperature", 0.9) or 0.9),
                )
                for item in items
            ),
            return_exceptions=True,
        )
        return [result if isinstance(result, list) else [] for result in results]

    def _call_with_temp(self, fn, temperature: float, **kwargs) -> str:
        """Call wrapped function with DSPy temperature context when available."""
        try:
            import dspy
        except Exception:
            result = fn(**kwargs)
            return getattr(result, 'summary', str(result))

        current_lm = getattr(dspy.settings, 'lm', None)
        if current_lm is None:
            result = fn(**kwargs)
            return getattr(result, 'summary', str(result))

        with dspy.context(lm=current_lm.copy(temperature=temperature)):
            result = fn(**kwargs)
            value = getattr(result, 'summary', str(result))
            return clean_summary_text(value)


# =============================================================================
# Tournament Strategy (Wrapper for Learning Mode)
# =============================================================================

@dataclass
class TournamentConfig:
    """Configuration for tournament-based candidate selection."""
    k: int = 4  # Number of candidates to generate
    temperature: float = 0.9  # Temperature for candidate generation
    judge_retry_attempts: int = 1  # Additional judge retries per match on errors
    judge_retry_delay_seconds: float = 1.0  # Base delay before retrying failed matches


class TournamentStrategy:
    """
    Wraps any SummarizationStrategy with tournament selection.

    This strategy generates multiple candidate summaries using the base strategy's
    generate_candidates() method and uses either listwise or pairwise judging to
    select the best one. Pairwise preferences remain available as an optimizer-
    facing projection, while richer comparative judgments are collected when the
    judge can rank the full candidate set jointly.

    Usage:
        # Wrap any base strategy
        base = BatchedStrategy(client)
        strategy = TournamentStrategy(base, judge=...)

        # Use like any other strategy - tournament happens transparently
        summary = await strategy.summarize(content, rubric)

        # Get collected preferences (free byproduct!)
        preferences = strategy.get_preferences()

    The tournament wrapper is transparent to TreeBuilder - it doesn't know
    or care that tournament selection is happening internally.
    """

    def __init__(
        self,
        base: SummarizationStrategy,
        judge: Any,
        config: Optional[TournamentConfig] = None,
        feedback_collector: Optional[Any] = None,
    ):
        """
        Initialize tournament strategy.

        Args:
            base: Base summarization strategy to wrap
            judge: Comparison judge with either pairwise `.compare(...)` /
                DSPy-style `.forward(...)` or listwise `.rank_candidates(...)`
                support.
            config: Tournament configuration (k candidates, temperature)
            feedback_collector: Optional FeedbackCollector for enriched feedback.
                When set, tournament matches also produce FeedbackResponse objects
                in addition to PreferencePair objects.
        """
        self.base = base
        self.judge = judge
        self.config = config or TournamentConfig()
        self._preferences: List["BinaryComparison"] = []
        self._comparative_judgments: List["ComparativeJudgment"] = []
        self._feedback_responses: List[Any] = []
        self._segment_counter = 0
        self._feedback_collector = feedback_collector

    def reset_counter(self) -> None:
        """Reset the segment counter. Call between documents in sequential mode
        to keep segment IDs aligned with per-document tree node IDs."""
        self._segment_counter = 0

    def _extract_preference_fields(
        self,
        result: Any,
        is_dspy_module: bool,
        match_label: Optional[str] = None,
    ) -> tuple[str, str, float, Optional[float], Optional[float]]:
        """Normalize judge result into (preferred, reasoning, confidence, score_a, score_b)."""
        if result is None:
            return "tie", "", 0.0, None, None

        if is_dspy_module:
            preferred = getattr(result, "preference", getattr(result, "preferred", "tie"))
        else:
            preferred = getattr(result, "preferred", None)
            if preferred is None:
                error_message = getattr(result, "error_message", None)
                if error_message:
                    logger.warning(
                        "Tournament match %s returned error: %s",
                        match_label or "unknown",
                        error_message,
                    )
                    return "tie", f"Error: {error_message}", 0.0, None, None
                logger.warning(
                    "Tournament match %s returned result without preference",
                    match_label or "unknown",
                )
                return "tie", "", 0.0, None, None

        preferred = str(preferred).strip().upper()
        if preferred not in ("A", "B", "TIE"):
            logger.warning(
                "Tournament match %s returned invalid preference: %s",
                match_label or "unknown",
                preferred,
            )
            preferred = "TIE"
        preferred = "tie" if preferred == "TIE" else preferred

        reasoning = getattr(result, "reasoning", "")
        confidence = getattr(result, "confidence", 0.5)
        score_a = getattr(result, "helpfulness_a", None)
        score_b = getattr(result, "helpfulness_b", None)
        if score_a is None:
            score_a = getattr(result, "score_estimate_a", None)
        if score_b is None:
            score_b = getattr(result, "score_estimate_b", None)

        return preferred, reasoning, confidence, score_a, score_b

    def _augment_context_with_doc_metadata(self, context: str) -> str:
        """Append safe document metadata to judge context when available."""
        rendered = str(context or "").strip()
        try:
            from treepo._research.core.engram_prompting import format_prompt_metadata_block

            metadata_block = format_prompt_metadata_block()
        except Exception:
            metadata_block = ""

        if not metadata_block:
            return rendered
        if rendered:
            return f"{rendered}\n\n{metadata_block}"
        return metadata_block

    def _supports_listwise_judging(self) -> bool:
        """Whether the configured judge can rank all candidates in one call."""
        from treepo._research.training.supervision.judge_capabilities import (
            supports_direct_comparative_judging,
        )

        return supports_direct_comparative_judging(self.judge)

    def _judge_model_name(self) -> str:
        """Render a stable judge backend label for saved supervision records."""
        from treepo._research.training.supervision.judge_capabilities import judge_backend_name

        return judge_backend_name(self.judge)

    async def summarize(
        self, content: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """
        Summarize with tournament selection.

        Generates k candidates via base strategy, runs tournament, returns winner.
        Collects preferences as byproduct.

        Note: temperature param is ignored - we use config.temperature for candidates.
        """
        self._segment_counter += 1
        segment_id = f"leaf_{self._segment_counter}"
        candidates = await self.base.generate_candidates(
            content, rubric, k=self.config.k, temperature=self.config.temperature
        )
        return await self._select_from_candidates(
            candidates=candidates,
            original_text=content,
            rubric=rubric,
            segment_id=segment_id,
            law_type="sufficiency",
            doc_id=tournament_doc_id.get(),
        )

    async def merge(
        self, left: str, right: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """
        Merge with tournament selection.

        Generates k merge candidates via base strategy, runs tournament, returns winner.
        Collects preferences as byproduct.
        """
        self._segment_counter += 1
        segment_id = f"merge_{self._segment_counter}"
        candidates = await self.base.generate_merge_candidates(
            left, right, rubric, k=self.config.k, temperature=self.config.temperature
        )
        return await self._select_from_candidates(
            candidates=candidates,
            original_text=format_merge_input(left, right),
            rubric=rubric,
            segment_id=segment_id,
            law_type="merge",
            doc_id=tournament_doc_id.get(),
        )

    async def generate_candidates(
        self, content: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """Delegate to base strategy."""
        return await self.base.generate_candidates(content, rubric, k, temperature)

    async def generate_merge_candidates(
        self, left: str, right: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """Delegate to base strategy."""
        return await self.base.generate_merge_candidates(left, right, rubric, k, temperature)

    async def summarize_many(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> List[str]:
        base_items = [
            {
                "content": str(item.get("content", "") or ""),
                "rubric": str(item.get("rubric", "") or ""),
                "k": max(1, int(item.get("k", self.config.k) or self.config.k)),
                "temperature": float(
                    item.get("temperature", self.config.temperature)
                    or self.config.temperature
                ),
                "doc_id": item.get("doc_id"),
            }
            for item in items
        ]
        if hasattr(self.base, "generate_candidates_many"):
            candidate_sets = await self.base.generate_candidates_many(base_items)
        else:
            results = await asyncio.gather(
                *(
                    self.base.generate_candidates(
                        item["content"],
                        item["rubric"],
                        k=int(item["k"]),
                        temperature=float(item["temperature"]),
                    )
                    for item in base_items
                ),
                return_exceptions=True,
            )
            candidate_sets = [result if isinstance(result, list) else [] for result in results]

        tasks = []
        for item, candidates in zip(base_items, candidate_sets):
            self._segment_counter += 1
            tasks.append(
                self._select_from_candidates(
                    candidates=candidates,
                    original_text=item["content"],
                    rubric=item["rubric"],
                    segment_id=f"leaf_{self._segment_counter}",
                    law_type="sufficiency",
                    doc_id=item.get("doc_id"),
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [result if isinstance(result, str) else "" for result in results]

    async def merge_many(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> List[str]:
        base_items = [
            {
                "left": str(item.get("left", "") or ""),
                "right": str(item.get("right", "") or ""),
                "rubric": str(item.get("rubric", "") or ""),
                "k": max(1, int(item.get("k", self.config.k) or self.config.k)),
                "temperature": float(
                    item.get("temperature", self.config.temperature)
                    or self.config.temperature
                ),
                "doc_id": item.get("doc_id"),
            }
            for item in items
        ]
        if hasattr(self.base, "generate_merge_candidates_many"):
            candidate_sets = await self.base.generate_merge_candidates_many(base_items)
        else:
            results = await asyncio.gather(
                *(
                    self.base.generate_merge_candidates(
                        item["left"],
                        item["right"],
                        item["rubric"],
                        k=int(item["k"]),
                        temperature=float(item["temperature"]),
                    )
                    for item in base_items
                ),
                return_exceptions=True,
            )
            candidate_sets = [result if isinstance(result, list) else [] for result in results]

        tasks = []
        for item, candidates in zip(base_items, candidate_sets):
            self._segment_counter += 1
            tasks.append(
                self._select_from_candidates(
                    candidates=candidates,
                    original_text=format_merge_input(item["left"], item["right"]),
                    rubric=item["rubric"],
                    segment_id=f"merge_{self._segment_counter}",
                    law_type="merge",
                    doc_id=item.get("doc_id"),
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [result if isinstance(result, str) else "" for result in results]

    async def _select_from_candidates(
        self,
        *,
        candidates: List[str],
        original_text: str,
        rubric: str,
        segment_id: str,
        law_type: str,
        doc_id: Optional[Any],
    ) -> str:
        token = tournament_doc_id.set(str(doc_id)) if doc_id is not None else None
        try:
            if len(candidates) < 2:
                return candidates[0] if candidates else ""
            winner, prefs, comparative_records = await self._run_tournament_pipelined(
                candidates,
                original_text,
                rubric,
                segment_id,
                law_type=law_type,
            )
            self._preferences.extend(prefs)
            self._comparative_judgments.extend(comparative_records)
            return winner
        finally:
            if token is not None:
                tournament_doc_id.reset(token)

    async def _run_tournament_pipelined(
        self,
        candidates: List[str],
        original_text: str,
        rubric: str,
        segment_id: str,
        law_type: str = "sufficiency",
    ) -> tuple[str, List["BinaryComparison"], List["ComparativeJudgment"]]:
        """
        Run elimination tournament with pipelined execution.

        Submits new matches as soon as their prerequisites (parent matches)
        complete, rather than waiting for all round-N matches before starting
        round-N+1.

        For k=4 candidates:
        - Round 1: Match0 (A vs B), Match1 (C vs D)
        - Round 2: Match2 (winner0 vs winner1) - starts when BOTH Match0 and Match1 complete

        For k=8 candidates:
        - Round 1: Match0-3
        - Round 2: Match4 starts when Match0 AND Match1 complete (don't wait for Match2,3)
                   Match5 starts when Match2 AND Match3 complete
        - Round 3: Match6 starts when Match4 AND Match5 complete
        """
        from treepo._research.training.supervision import BinaryComparison

        if len(candidates) == 0:
            raise ValueError("No candidates provided")
        if len(candidates) == 1:
            return candidates[0], [], []

        if self._supports_listwise_judging():
            try:
                return await self._run_listwise_selection(
                    candidates=candidates,
                    original_text=original_text,
                    rubric=rubric,
                    segment_id=segment_id,
                    law_type=law_type,
                )
            except Exception as exc:
                logger.warning(
                    "Listwise tournament judging failed for %s (%s); falling back to pairwise bracket.",
                    segment_id,
                    exc,
                )

        # Build match structure upfront
        @dataclass
        class Match:
            id: int
            round: int
            left_idx: int  # Index into candidates (round 0) or match ID (later rounds)
            right_idx: int
            left_is_match: bool = False  # True if left_idx refers to a match result
            right_is_match: bool = False
            result: Optional[str] = None  # Winner summary when complete

        # Build tournament bracket
        matches: Dict[int, Match] = {}
        match_id = 0
        n = len(candidates)

        # Round 1: pair up candidates
        round_matches = []
        for i in range(0, n, 2):
            if i + 1 < n:
                matches[match_id] = Match(
                    id=match_id, round=1,
                    left_idx=i, right_idx=i + 1,
                    left_is_match=False, right_is_match=False
                )
                round_matches.append(match_id)
                match_id += 1

        # Handle odd candidate - it gets a "bye" (auto-advance to next round)
        bye_candidate_idx = n - 1 if n % 2 == 1 else None

        # Build subsequent rounds
        prev_round_matches = round_matches
        prev_bye = bye_candidate_idx
        round_num = 1

        while len(prev_round_matches) + (1 if prev_bye is not None else 0) > 1:
            round_num += 1
            round_matches = []

            # Pair up matches from previous round
            available = list(prev_round_matches)
            if prev_bye is not None:
                # Bye from previous round creates a "fake" match result
                # We'll handle this specially during execution
                pass

            for i in range(0, len(available), 2):
                if i + 1 < len(available):
                    matches[match_id] = Match(
                        id=match_id, round=round_num,
                        left_idx=available[i], right_idx=available[i + 1],
                        left_is_match=True, right_is_match=True
                    )
                    round_matches.append(match_id)
                    match_id += 1

            # New bye if odd number of matches
            if len(available) % 2 == 1:
                prev_bye = available[-1]  # Last match result is bye
            else:
                prev_bye = None

            prev_round_matches = round_matches

        # Track which matches are ready and completed
        preferences: List[BinaryComparison] = []
        completed: Dict[int, str] = {}  # match_id -> winner
        pending: Dict[int, asyncio.Task] = {}  # match_id -> task

        doc_id = tournament_doc_id.get()
        segment_tag = f"{doc_id}:{segment_id}" if doc_id is not None else segment_id

        judge_model = self._judge_model_name()

        async def execute_match(
            m: Match,
        ) -> tuple[int, str, "BinaryComparison"]:
            """Execute a single match and return normalized tournament artifacts."""
            from treepo._research.core.supervision_metadata import judgment_supervision_metadata
            from treepo._research.training.supervision import BinaryComparison
            from treepo._research.training.supervision.judge_capabilities import (
                invoke_pairwise_judgment_async,
            )

            # Get summaries for this match
            if m.left_is_match:
                summary_a = completed[m.left_idx]
            else:
                summary_a = candidates[m.left_idx]

            if m.right_is_match:
                summary_b = completed[m.right_idx]
            else:
                summary_b = candidates[m.right_idx]
            judge_context = self._augment_context_with_doc_metadata(rubric)

            max_attempts = 1 + max(0, int(getattr(self.config, "judge_retry_attempts", 0)))
            retry_delay = max(0.0, float(getattr(self.config, "judge_retry_delay_seconds", 0.0)))
            result: Any = None
            match_label = f"{segment_tag}_m{m.id}"
            for attempt in range(1, max_attempts + 1):
                try:
                    result = await invoke_pairwise_judgment_async(
                        self.judge,
                        context=judge_context,
                        original_text=original_text,
                        summary_a=summary_a,
                        summary_b=summary_b,
                        law_type=law_type,
                    )
                    if result.error_message is None:
                        break
                    error_message = result.error_message or (
                        f"missing/invalid preference in result type {type(result).__name__}"
                    )
                    if attempt < max_attempts:
                        logger.warning(
                            "Tournament match %s judge error on attempt %d/%d: %s. Retrying...",
                            match_label,
                            int(attempt),
                            int(max_attempts),
                            error_message,
                        )
                        if retry_delay > 0:
                            await asyncio.sleep(retry_delay * float(attempt))
                        continue
                    logger.error(
                        "Tournament match %s judge error on final attempt %d/%d: %s. Falling back to tie handling.",
                        match_label,
                        int(attempt),
                        int(max_attempts),
                        error_message,
                    )
                except Exception as exc:
                    if attempt < max_attempts:
                        logger.warning(
                            "Tournament match %s judge exception on attempt %d/%d: %s. Retrying...",
                            match_label,
                            int(attempt),
                            int(max_attempts),
                            exc,
                        )
                        if retry_delay > 0:
                            await asyncio.sleep(retry_delay * float(attempt))
                        continue
                    raise

            preferred = result.preferred
            reasoning = result.reasoning
            confidence = result.confidence
            score_a = result.score_estimate_a
            score_b = result.score_estimate_b

            if preferred == "A":
                winner = summary_a
            elif preferred == "B":
                winner = summary_b
            else:
                winner = summary_a if random.random() < 0.5 else summary_b

            pair = BinaryComparison(
                pair_id=f"tournament_{segment_tag}_m{m.id}",
                source_example_id=segment_tag,
                original_text=original_text,
                rubric=rubric,
                reference_score=0.0,
                summary_a=summary_a,
                summary_b=summary_b,
                preferred=preferred,
                reasoning=reasoning,
                confidence=confidence,
                law_type=law_type,
                preference_supervision=judgment_supervision_metadata(
                    application_name="tournament_preference_collection",
                    law_type=law_type,
                    comparison_signal_name=result.comparison_signal_name,
                    comparison_signal_min=result.comparison_signal_min,
                    comparison_signal_max=result.comparison_signal_max,
                    response_signal_name=result.response_signal_name,
                    response_signal_min=result.response_signal_min,
                    response_signal_max=result.response_signal_max,
                ),
                score_estimate_a=score_a,
                score_estimate_b=score_b,
                comparison_signal_value=result.comparison_signal_value,
                judge_model=judge_model,
            )

            return (m.id, winner, pair)

        def is_ready(m: Match) -> bool:
            """Check if a match's prerequisites are satisfied."""
            if m.left_is_match and m.left_idx not in completed:
                return False
            if m.right_is_match and m.right_idx not in completed:
                return False
            return True

        # Submit all ready matches (round 1)
        for m in matches.values():
            if is_ready(m) and m.id not in pending and m.id not in completed:
                pending[m.id] = asyncio.create_task(execute_match(m))

        try:
            # Process matches as they complete
            while pending:
                done, _ = await asyncio.wait(
                    pending.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    try:
                        match_id, winner, pair = await task
                    except Exception as e:
                        # Find which match failed
                        for mid, t in list(pending.items()):
                            if t is task:
                                logger.error(
                                    "Tournament match %s failed after retries: %s. Falling back to tie/random.",
                                    f"{segment_tag}_m{mid}",
                                    e,
                                )
                                # Treat as tie, pick randomly to avoid position bias.
                                m = matches[mid]
                                if m.left_is_match:
                                    summary_a = completed[m.left_idx]
                                else:
                                    summary_a = candidates[m.left_idx]
                                if m.right_is_match:
                                    summary_b = completed[m.right_idx]
                                else:
                                    summary_b = candidates[m.right_idx]
                                winner = summary_a if random.random() < 0.5 else summary_b
                                completed[mid] = winner
                                del pending[mid]
                                break
                        continue

                    # Record completion
                    completed[match_id] = winner
                    del pending[match_id]

                    preferences.append(pair)

                    # Collect enriched feedback if collector is configured
                    if self._feedback_collector is not None:
                        try:
                            from treepo._research.feedback.types import FeedbackRequest, FeedbackDimension

                            judge_reasoning = clean_summary_text(pair.reasoning)
                            if len(judge_reasoning) > 2400:
                                judge_reasoning = judge_reasoning[:2400].rstrip() + " ... (truncated)"

                            fb_request = FeedbackRequest(
                                request_id=pair.pair_id,
                                text_a=pair.summary_a,
                                text_b=pair.summary_b,
                                original_text=original_text,
                                rubric=rubric,
                                law_type=law_type,
                                node_id=segment_tag,
                                dimensions=[
                                    FeedbackDimension(kind="pairwise"),
                                    FeedbackDimension(kind="critique"),
                                ],
                                context={
                                    "match_label": f"{segment_tag}_m{match_id}",
                                    "match_id": match_id,
                                    "round": getattr(matches[match_id], "round", None),
                                    "left_idx": getattr(matches[match_id], "left_idx", None),
                                    "right_idx": getattr(matches[match_id], "right_idx", None),
                                    "left_is_match": getattr(matches[match_id], "left_is_match", None),
                                    "right_is_match": getattr(matches[match_id], "right_is_match", None),
                                    "judge_preferred": pair.preferred,
                                    "judge_confidence": pair.confidence,
                                    "judge_reasoning": judge_reasoning,
                                    "judge_score_estimate_a": pair.score_estimate_a,
                                    "judge_score_estimate_b": pair.score_estimate_b,
                                    "judge_model": judge_model,
                                },
                            )
                            fb_response = self._feedback_collector.collect(fb_request)
                            self._feedback_responses.append((fb_request, fb_response))
                        except Exception as e:
                            logger.debug("Feedback collector failed in tournament: %s", e)

                    # Check for newly ready matches
                    for m in matches.values():
                        if is_ready(m) and m.id not in pending and m.id not in completed:
                            pending[m.id] = asyncio.create_task(execute_match(m))

        finally:
            # CRITICAL: Cancel any remaining tasks on exception or early exit
            # This prevents orphaned tasks from continuing to run and slowing down the system
            if pending:
                config = get_concurrency_config()
                logger.debug(f"Cleaning up {len(pending)} pending tournament tasks...")
                for task in pending.values():
                    if not task.done():
                        task.cancel()
                # Wait for cancellation to complete (with configurable timeout)
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending.values(), return_exceptions=True),
                        timeout=config.task_cancel_timeout
                    )
                except asyncio.TimeoutError:
                    remaining = sum(1 for t in pending.values() if not t.done())
                    logger.warning(
                        f"Timeout ({config.task_cancel_timeout}s) waiting for "
                        f"{remaining}/{len(pending)} tournament tasks to cancel"
                    )

        # Find final winner
        if completed:
            final_match_id = max(completed.keys())
            return completed[final_match_id], preferences, []
        else:
            return candidates[0], preferences, []

    async def _run_listwise_selection(
        self,
        *,
        candidates: List[str],
        original_text: str,
        rubric: str,
        segment_id: str,
        law_type: str = "sufficiency",
    ) -> tuple[str, List["BinaryComparison"], List["ComparativeJudgment"]]:
        """Rank the full candidate set jointly and keep a conservative pairwise projection."""
        from treepo._research.core.logged_supervision import ObservationUnitKind, SamplingMetadata
        from treepo._research.core.supervision_metadata import judgment_supervision_metadata
        from treepo._research.training.supervision import (
            ComparativeCandidate,
            ComparativeJudgment,
            BinaryComparison,
        )

        if len(candidates) < 2:
            return candidates[0] if candidates else "", [], []
        if not self._supports_listwise_judging():
            raise ValueError("Configured judge does not support listwise ranking")

        doc_id = tournament_doc_id.get()
        segment_tag = f"{doc_id}:{segment_id}" if doc_id is not None else segment_id
        judge_context = self._augment_context_with_doc_metadata(rubric)
        judge_model = self._judge_model_name()
        match_label = f"{segment_tag}_listwise"
        max_attempts = 1 + max(0, int(getattr(self.config, "judge_retry_attempts", 0)))
        retry_delay = max(0.0, float(getattr(self.config, "judge_retry_delay_seconds", 0.0)))
        result = None
        last_error: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                from treepo._research.training.supervision.judge_capabilities import (
                    invoke_comparative_judgment_async,
                )

                candidate_result = await invoke_comparative_judgment_async(
                    self.judge,
                    context=judge_context,
                    original_text=original_text,
                    candidate_summaries=candidates,
                    law_type=law_type,
                )
                ordered_ids = [str(value).upper() for value in candidate_result.ordered_candidate_ids]
                if len(ordered_ids) < 2:
                    raise ValueError("comparative judge returned fewer than two ordered candidates")
                result = candidate_result
                break
            except Exception as exc:
                last_error = exc
                if attempt < max_attempts:
                    logger.warning(
                        "Tournament listwise judgment %s failed on attempt %d/%d: %s. Retrying...",
                        match_label,
                        int(attempt),
                        int(max_attempts),
                        exc,
                    )
                    if retry_delay > 0:
                        await asyncio.sleep(retry_delay * float(attempt))
                    continue
                raise

        if result is None:
            raise last_error or RuntimeError("Listwise tournament judge returned no result")

        ordered_ids = [str(value).upper() for value in result.ordered_candidate_ids]
        canonical_ids = [f"C{idx}" for idx in range(1, len(candidates) + 1)]
        seen_ids = set()
        normalized_order: List[str] = []
        for candidate_id in ordered_ids:
            if candidate_id in canonical_ids and candidate_id not in seen_ids:
                normalized_order.append(candidate_id)
                seen_ids.add(candidate_id)
        for candidate_id in canonical_ids:
            if candidate_id not in seen_ids:
                normalized_order.append(candidate_id)

        candidate_scores = {
            str(candidate_id).upper(): float(score)
            for candidate_id, score in dict(result.candidate_scores or {}).items()
        }
        if candidate_scores:
            default_positions = {
                candidate_id: idx
                for idx, candidate_id in enumerate(normalized_order)
            }
            normalized_order = sorted(
                normalized_order,
                key=lambda candidate_id: (
                    -float(candidate_scores.get(candidate_id, float("-inf"))),
                    default_positions.get(candidate_id, len(default_positions)),
                ),
            )
        if len(normalized_order) < 2:
            raise ValueError("Listwise tournament ranking did not identify at least two candidates")

        confidence = max(0.0, min(1.0, float(result.confidence or 0.5)))
        reasoning = str(result.reasoning or "")
        response_signal_name = str(
            result.response_signal_name or "listwise_candidate_score"
        )
        winner_id = normalized_order[0]
        runner_up_id = normalized_order[1]
        winner_index = int(winner_id[1:]) - 1
        runner_up_index = int(runner_up_id[1:]) - 1
        winner_summary = candidates[winner_index]
        runner_up_summary = candidates[runner_up_index]
        winner_score = candidate_scores.get(winner_id)
        runner_up_score = candidate_scores.get(runner_up_id)

        comparison_signal_name: Optional[str] = None
        comparison_signal_value: Optional[float] = None
        if winner_score is not None and runner_up_score is not None:
            comparison_signal_name = "listwise_score_margin"
            comparison_signal_value = float(winner_score) - float(runner_up_score)

        projection_pair = BinaryComparison(
            pair_id=f"tournament_{segment_tag}_listwise_top2",
            source_example_id=segment_tag,
            original_text=original_text,
            rubric=rubric,
            reference_score=0.0,
            summary_a=winner_summary,
            summary_b=runner_up_summary,
            preferred="A",
            reasoning=reasoning,
            confidence=max(0.5, confidence),
            law_type=law_type,
            preference_supervision=judgment_supervision_metadata(
                application_name="tournament_preference_collection",
                law_type=law_type,
                comparison_signal_name=comparison_signal_name,
                response_signal_name=response_signal_name,
                metadata={
                    "collection_mode": "listwise_projection",
                    "projection": "winner_vs_runner_up",
                    "source_record_id": f"tournament_{segment_tag}_listwise",
                    "num_candidates": len(candidates),
                },
            ),
            score_estimate_a=winner_score,
            score_estimate_b=runner_up_score,
            comparison_signal_value=comparison_signal_value,
            judge_model=judge_model,
        )

        supervision = judgment_supervision_metadata(
            application_name="tournament_preference_collection",
            law_type=law_type,
            comparison_signal_name=result.comparison_signal_name,
            comparison_signal_min=result.comparison_signal_min,
            comparison_signal_max=result.comparison_signal_max,
            response_signal_name=response_signal_name,
            response_signal_min=result.response_signal_min,
            response_signal_max=result.response_signal_max,
            metadata={
                "collection_mode": "listwise",
                "selection_strategy": "winner_take_all",
                "num_candidates": len(candidates),
            },
        ).with_updates(preference_family="groupwise")
        sampling = SamplingMetadata(
            unit_kind=ObservationUnitKind.PAIR,
            sampling_scheme="tournament_listwise_selection",
            policy_name=judge_model,
            metadata={
                "segment_id": segment_id,
                "source_example_id": segment_tag,
                "num_candidates": len(candidates),
            },
        )
        rank_by_id = {
            candidate_id: rank
            for rank, candidate_id in enumerate(normalized_order, start=1)
        }
        comparative_candidates: List[ComparativeCandidate] = []
        for idx, summary in enumerate(candidates, start=1):
            candidate_id = f"C{idx}"
            comparative_candidates.append(
                ComparativeCandidate(
                    candidate_id=candidate_id,
                    response=summary,
                    rank=rank_by_id.get(candidate_id, idx),
                    response_signal_value=candidate_scores.get(candidate_id),
                    source_pair_ids=[
                        projection_pair.pair_id
                    ] if candidate_id in {winner_id, runner_up_id} else [],
                    metadata={
                        "candidate_index": idx - 1,
                        "selected_winner": candidate_id == winner_id,
                        "selected_runner_up": candidate_id == runner_up_id,
                    },
                )
            )

        comparative_record = ComparativeJudgment(
            record_id=f"tournament_{segment_tag}_listwise",
            source_example_id=segment_tag,
            original_text=original_text,
            rubric=rubric,
            reference_score=0.0,
            law_type=law_type,
            candidates=comparative_candidates,
            sampling=sampling,
            preference_supervision=supervision,
            source_pair_ids=[projection_pair.pair_id],
            aggregate_sample_weight=sampling.ipw_weight(),
            judge_model=judge_model,
            comparison_signal_value=result.comparison_signal_value,
            metadata={
                "reasoning": reasoning,
                "confidence": confidence,
                "judge_payload": dict(result.raw_payload or {}),
            },
        )

        if self._feedback_collector is not None:
            try:
                from treepo._research.feedback.types import FeedbackDimension, FeedbackRequest

                judge_reasoning = clean_summary_text(reasoning)
                if len(judge_reasoning) > 2400:
                    judge_reasoning = judge_reasoning[:2400].rstrip() + " ... (truncated)"

                fb_request = FeedbackRequest(
                    request_id=comparative_record.record_id,
                    text_a=winner_summary,
                    text_b=runner_up_summary,
                    original_text=original_text,
                    rubric=rubric,
                    law_type=law_type,
                    node_id=segment_tag,
                    dimensions=[
                        FeedbackDimension(kind="pairwise"),
                        FeedbackDimension(kind="critique"),
                    ],
                    context={
                        "match_label": match_label,
                        "judge_preferred": "A",
                        "judge_confidence": confidence,
                        "judge_reasoning": judge_reasoning,
                        "judge_score_estimate_a": winner_score,
                        "judge_score_estimate_b": runner_up_score,
                        "judge_model": judge_model,
                        "listwise": True,
                        "ordered_candidate_ids": list(normalized_order),
                        "candidate_scores": dict(candidate_scores),
                        "num_candidates": len(candidates),
                    },
                )
                fb_response = self._feedback_collector.collect(fb_request)
                self._feedback_responses.append((fb_request, fb_response))
            except Exception as exc:
                logger.debug("Feedback collector failed in listwise tournament: %s", exc)

        return winner_summary, [projection_pair], [comparative_record]

    def get_preferences(self) -> List["BinaryComparison"]:
        """Get all collected preference pairs."""
        return self._preferences

    def get_comparative_judgments(self) -> List["ComparativeJudgment"]:
        """Get all collected comparative judgments."""
        return self._comparative_judgments

    def get_feedback_responses(self) -> List[Any]:
        """Get all collected feedback request/response pairs.

        Returns list of (FeedbackRequest, FeedbackResponse) tuples collected
        when a feedback_collector is configured.
        """
        return self._feedback_responses

    def reset_preferences(self) -> None:
        """Reset collected preferences (e.g., between documents)."""
        self._preferences = []
        self._comparative_judgments = []
        self._feedback_responses = []

    def get_preference_count(self) -> int:
        """Get number of collected preferences."""
        return len(self._preferences)

    def get_comparative_judgment_count(self) -> int:
        """Get number of collected comparative judgments."""
        return len(self._comparative_judgments)


# =============================================================================
# Strategy Registry
# =============================================================================

_STRATEGY_REGISTRY: Dict[str, type] = {}


def register_strategy(name: str):
    """
    Decorator to register a strategy class.

    Args:
        name: Name to register the strategy under

    Example:
        @register_strategy("custom")
        class CustomStrategy:
            ...
    """
    def decorator(cls):
        _STRATEGY_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_strategy(name: str, **kwargs) -> SummarizationStrategy:
    """
    Get a strategy by name from the registry.

    Args:
        name: Strategy name ("batched", "dspy", "callable", "tournament")
        **kwargs: Arguments passed to strategy constructor

    Returns:
        Configured strategy instance

    Raises:
        ValueError: If strategy name is not registered

    Example:
        strategy = get_strategy("batched", client=my_client)
        strategy = get_strategy("tournament", base=base_strategy, judge=my_judge)
    """
    name_lower = name.lower()
    if name_lower not in _STRATEGY_REGISTRY:
        available = list(_STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: '{name}'. Available: {available}")

    return _STRATEGY_REGISTRY[name_lower](**kwargs)


def list_strategies() -> List[str]:
    """Return list of registered strategy names."""
    return list(_STRATEGY_REGISTRY.keys())


# =============================================================================
# Gated Strategy (Engram WS3 — Context-aware gating)
# =============================================================================

class GatedStrategy:
    """Wraps any SummarizationStrategy with ConditionalMemory gating.

    Before each summarize/merge call, checks ConditionalMemory for a
    cached result. For "easy" chunks (high boilerplate ratio, low entity
    density), uses the cached summary or a deterministic fallback instead
    of calling the LLM.

    This is Engram's core insight: don't waste compute on trivial pattern
    reconstruction.  Expected impact: 25-40% reduction in LLM calls for
    repetitive corpora.

    Parameters
    ----------
    base : SummarizationStrategy
        The underlying strategy for actual LLM calls.
    memory : ConditionalMemory
        Shared memory instance for cache lookups.
    gate_threshold : float
        Minimum complexity score (0-1) below which cached results are used.
        0.0 = never gate (always call LLM), 1.0 = always gate if cached.
    """

    def __init__(
        self,
        base: SummarizationStrategy,
        memory: Any,  # ConditionalMemory (avoid import for TYPE_CHECKING)
        gate_threshold: float = 0.3,
    ):
        self.base = base
        self.memory = memory
        self.gate_threshold = gate_threshold
        self._gate_hits = 0
        self._gate_misses = 0

    @staticmethod
    def _complexity_score(text: str) -> float:
        """Estimate text complexity as a 0-1 score.

        Uses cheap heuristics:
        - Entity density (capitalized words / total words)
        - Vocabulary diversity (unique words / total words)
        - Boilerplate ratio (repeated bigrams)

        Returns a score where higher = more complex = should call LLM.
        """
        words = text.split()
        if len(words) < 5:
            return 1.0  # Short texts always go to LLM

        total = len(words)
        unique = len(set(w.lower() for w in words))
        capitalized = sum(1 for w in words if w and w[0].isupper())

        vocab_diversity = unique / total  # 0-1, higher = more diverse
        entity_density = capitalized / total  # 0-1, higher = more entities

        # Bigram repetition (boilerplate indicator)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        unique_bigrams = len(set(bigrams))
        bigram_diversity = unique_bigrams / max(1, len(bigrams))

        # Weighted combination: high diversity + high entities = complex
        score = 0.4 * vocab_diversity + 0.3 * entity_density + 0.3 * bigram_diversity
        return min(1.0, score)

    async def summarize(
        self, content: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """Summarize with gating: skip LLM for cached easy chunks."""
        namespace = f"gated_summarize:{self.memory.namespace_version}"
        key = canonical_hash(content)
        cached = self.memory.get_text(namespace, key)
        if cached:
            complexity = self._complexity_score(content)
            if complexity < self.gate_threshold:
                self._gate_hits += 1
                return cached

        self._gate_misses += 1
        result = await self.base.summarize(content, rubric, temperature)

        # Store result for future gating
        self.memory.set_text(namespace, key, result)
        return result

    async def merge(
        self, left: str, right: str, rubric: str, temperature: float = 0.7
    ) -> str:
        """Merge with gating: skip LLM for cached merge results."""
        merge_key = f"{left}\n---MERGE---\n{right}"
        namespace = f"gated_merge:{self.memory.namespace_version}"
        key = canonical_hash(merge_key)
        cached = self.memory.get_text(namespace, key)
        if cached:
            complexity = max(
                self._complexity_score(left),
                self._complexity_score(right),
            )
            if complexity < self.gate_threshold:
                self._gate_hits += 1
                return cached

        self._gate_misses += 1
        result = await self.base.merge(left, right, rubric, temperature)

        self.memory.set_text(namespace, key, result)
        return result

    async def generate_candidates(
        self, content: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """Delegate to base — candidate generation always calls LLM."""
        return await self.base.generate_candidates(content, rubric, k, temperature)

    async def generate_merge_candidates(
        self, left: str, right: str, rubric: str, k: int = 4, temperature: float = 0.9
    ) -> List[str]:
        """Delegate to base — candidate generation always calls LLM."""
        return await self.base.generate_merge_candidates(left, right, rubric, k, temperature)

    def gate_stats(self) -> Dict[str, Any]:
        """Return gating statistics."""
        total = self._gate_hits + self._gate_misses
        return {
            "gate_hits": self._gate_hits,
            "gate_misses": self._gate_misses,
            "gate_rate": self._gate_hits / total if total > 0 else 0.0,
        }


# Register built-in strategies
register_strategy("batched")(BatchedStrategy)
register_strategy("dspy")(DSPyStrategy)
register_strategy("callable")(CallableStrategy)
register_strategy("tournament")(TournamentStrategy)
register_strategy("gated")(GatedStrategy)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Protocol
    "SummarizationStrategy",
    # Registry
    "get_strategy",
    "list_strategies",
    "register_strategy",
    # Implementations
    "BatchedStrategy",
    "DSPyStrategy",
    "CallableStrategy",
    "TournamentStrategy",
    "TournamentConfig",
    "GatedStrategy",
    # Context
    "tournament_doc_id",
]
