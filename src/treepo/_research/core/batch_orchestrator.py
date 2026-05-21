"""
Batch Tree Orchestrator - Global pipelined tree building across documents.

This module provides BatchTreeOrchestrator for processing multiple documents
with optimal batching. Unlike per-document processing, this orchestrator:

1. Pre-chunks ALL documents
2. Submits ALL leaf summaries together (one big batch)
3. Schedules merges globally as dependencies become ready
4. Continues until all trees are complete

This keeps the underlying LLM server fed with ready work across docs and levels.

Usage:
    strategy = BatchedStrategy(client)
    orchestrator = BatchTreeOrchestrator(strategy)
    results = await orchestrator.process_documents(docs, rubric)
"""

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Dict, TYPE_CHECKING, Deque, Tuple, Sequence

if TYPE_CHECKING:
    from treepo._research.training.supervision import BinaryComparison

from treepo._research.core.data_models import Node, Tree, leaf, node
from treepo._research.preprocessing.chunker import TextChunk, chunk_for_ops as chunk
from treepo._research.preprocessing.visual_feedback import extract_content_weights_from_chunks
from treepo._research.core.strategy import SummarizationStrategy, TournamentStrategy, tournament_doc_id
from treepo._research.core.unified_runtime import (
    BatchTelemetry,
    RUNTIME_MODE_UNIFIED_V2,
    TopologyPlan,
    WorkItem,
    build_balanced_topology_plan,
    build_unified_topology_plan,
    get_named_plan_cache,
    resolve_runtime_mode,
    plan_work_batches,
)
from treepo._research.tree.builder import BuildConfig, BuildResult
from treepo._research.core.async_utils import cancel_tasks, to_thread
from treepo._research.core.prompting import clean_summary_text, is_degenerate_summary_text
from treepo._research.training.supervision import SupervisionDataset
from unified_g_v1.core.specs import build_llm_text_program_spec


logger = logging.getLogger(__name__)
_LLM_TEXT_PROGRAM_FAMILY = build_llm_text_program_spec(
    tokenizer_or_adapter_id="cl100k_base"
).program_family


class DegenerateSummaryFailure(RuntimeError):
    """Raised when degenerate summary fallbacks exceed configured limits."""


@dataclass
class DocumentState:
    """Tracks tree-building state for a single document during orchestration."""
    doc_id: str
    sample: Any  # Original document/sample object
    chunks: List[TextChunk] = field(default_factory=list)
    current_level: List[Node] = field(default_factory=list)
    level_num: int = 0
    error: Optional[str] = None
    leaf_failures: int = 0
    merge_failures: int = 0
    empty_leaf_fallbacks: int = 0
    empty_merge_fallbacks: int = 0
    degenerate_leaf_fallbacks: int = 0
    degenerate_merge_fallbacks: int = 0
    interpreter_leaf_recoveries: int = 0
    interpreter_merge_recoveries: int = 0
    plan: Optional[TopologyPlan] = None


class BatchTreeOrchestrator:
    """
    Orchestrates tree building across multiple documents with global pipelined batching.

    This orchestrator maximizes throughput by pooling LLM requests across all
    documents and scheduling merges as soon as dependencies are ready:

    1. Leaf summaries for ALL documents are batched together
    2. Merges across ALL documents are submitted as they become ready
    3. The LLM server sees a continuous stream of ready work

    Example:
        # Simple inference
        strategy = BatchedStrategy(client)
        orchestrator = BatchTreeOrchestrator(strategy)
        results = await orchestrator.process_documents(docs, rubric)

        # With tournament selection (learning mode)
        tournament = TournamentStrategy(base=strategy, judge=judge)
        orchestrator = BatchTreeOrchestrator(tournament)
        results = await orchestrator.process_documents(docs, rubric)
        # Get preferences from the tournament strategy
        preferences = tournament.get_preferences()
    """

    def __init__(
        self,
        strategy: SummarizationStrategy,
        config: Optional[BuildConfig] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            strategy: SummarizationStrategy for summarize/merge operations
            config: Build configuration (chunking, etc.)
        """
        self.strategy = strategy
        self.config = config or BuildConfig()
        self._build_stats = {
            'documents_processed': 0,
            'total_chunks': 0,
            'total_merges': 0,
            'total_levels': 0,
            'leaf_failures': 0,
            'merge_failures': 0,
            'degenerate_leaf_fallbacks': 0,
            'degenerate_merge_fallbacks': 0,
            'documents_with_failures': 0,
        }
        # Live progress counters (updated during cascading build)
        self._completed_leaves = 0
        self._total_leaves = 0
        self._completed_merges = 0
        self._total_merges = 0
        self._runtime_mode = resolve_runtime_mode(getattr(self.config, "runtime_mode", None))
        self._runtime_telemetry = BatchTelemetry(runtime_mode=self._runtime_mode)
        self._plan_cache = get_named_plan_cache(
            str(getattr(self.config, "batch_plan_cache_name", "batch_tree_orchestrator"))
        )

    @property
    def completion_fraction(self) -> float:
        """Fraction of total work completed (0.0 to 1.0).

        Exposed for overlapped GPU phase transitions (DualPath Opt 3):
        when completion_fraction exceeds a threshold (e.g. 0.85), the GPU
        orchestrator can begin prewarming GenRM while the tail of tree
        building finishes on the primary task server.
        """
        total = self._total_leaves + self._total_merges
        if total == 0:
            return 0.0
        done = self._completed_leaves + self._completed_merges
        return min(1.0, done / total)

    async def process_documents(
        self,
        documents: List[Any],
        rubric: str,
        get_text_fn: Optional[Callable[[Any], str]] = None,
        get_id_fn: Optional[Callable[[Any], str]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        max_retries: int = 0,
    ) -> List[BuildResult]:
        """
        Process multiple documents with global pipelined batching.

        Uses cascading dependency-driven execution for maximum throughput.

        Args:
            documents: List of documents to process
            rubric: Information preservation criteria
            get_text_fn: Function to extract text from document (default: str(doc))
            get_id_fn: Function to extract ID from document (default: index-based)
            progress_callback: Optional callback(phase, completed, total)
            max_retries: Number of retry attempts for failed documents (default: 0)

        Returns:
            List of BuildResult, one per document
        """
        # Default extractors
        if get_text_fn is None:
            get_text_fn = lambda doc: str(doc) if isinstance(doc, str) else getattr(doc, 'text', str(doc))
        if get_id_fn is None:
            get_id_fn = lambda doc: str(hash(doc))

        # Phase 1: Chunk all documents
        logger.info(f"Phase 1: Chunking {len(documents)} documents...")
        states = await self._chunk_all_documents(
            documents, get_text_fn, get_id_fn, progress_callback
        )

        return await self._build_and_finalize(
            states, documents, rubric, get_text_fn, get_id_fn,
            progress_callback, max_retries,
        )

    async def _chunk_all_documents(
        self,
        documents: List[Any],
        get_text_fn: Callable[[Any], str],
        get_id_fn: Callable[[Any], str],
        progress_callback: Optional[Callable],
    ) -> List[DocumentState]:
        """Chunk all documents upfront."""
        states = []
        total_chunks = 0

        for i, doc in enumerate(documents):
            doc_id = get_id_fn(doc)
            state_idx = len(states)
            try:
                text = get_text_fn(doc)
                if not text or len(text.strip()) == 0:
                    logger.warning(f"Document {doc_id} has no text, skipping")
                    states.append(DocumentState(
                        doc_id=doc_id,
                        sample=doc,
                        error="No text content",
                    ))
                    continue

                chunks = chunk(
                    text,
                    max_chars=self.config.max_chunk_chars,
                    max_tokens=self.config.max_chunk_tokens,
                    token_encoding=self.config.chunk_token_encoding,
                    overlap_tokens=self.config.chunk_overlap_tokens,
                    strategy=self.config.chunk_strategy,
                )

                if not chunks:
                    logger.warning(f"Document {doc_id} produced no chunks, skipping")
                    states.append(DocumentState(
                        doc_id=doc_id,
                        sample=doc,
                        error="Chunking failed",
                    ))
                    continue

                plan = self._create_doc_plan(state_idx, doc_id, chunks)
                states.append(DocumentState(
                    doc_id=doc_id,
                    sample=doc,
                    chunks=chunks,
                    plan=plan,
                ))
                total_chunks += len(chunks)

            except Exception as e:
                logger.error(f"Failed to chunk document {doc_id}: {e}")
                states.append(DocumentState(
                    doc_id=doc_id,
                    sample=doc,
                    error=str(e),
                ))

        self._build_stats['total_chunks'] = total_chunks
        logger.info(f"  Chunked {len(documents)} documents into {total_chunks} total chunks")

        if progress_callback:
            progress_callback("chunk", len(documents), len(documents))

        return states

    def _create_doc_plan(
        self,
        state_idx: int,
        doc_id: str,
        chunks: List[TextChunk],
    ) -> TopologyPlan:
        """Lower fixed/adaptive chunked text into the shared topology plan."""
        leaf_nodes = [
            {
                "id": f"d{state_idx}_leaf_{i}",
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "char_count": len(chunk.text),
                "token_count": chunk.token_count,
            }
            for i, chunk in enumerate(chunks)
        ]
        return build_balanced_topology_plan(
            doc_index=int(state_idx),
            doc_id=str(doc_id),
            leaf_metadata=leaf_nodes,
        )

    async def _build_and_finalize(
        self,
        states: List[DocumentState],
        documents: List[Any],
        rubric: str,
        get_text_fn: Callable[[Any], str],
        get_id_fn: Callable[[Any], str],
        progress_callback: Optional[Callable],
        max_retries: int = 0,
    ) -> List[BuildResult]:
        """Build trees from pre-chunked states and finalize results.

        Shared helper used by both ``process_documents()`` and
        ``process_documents_unified()``.  Runs cascading tree build,
        converts to ``BuildResult``, optionally retries failures, and
        logs a summary.
        """
        logger.info("Building trees with cascading execution...")
        await self._build_trees_cascading(states, rubric, progress_callback)

        results = self._create_results(states, rubric)

        if max_retries > 0:
            results = await self._retry_failed_documents(
                results, documents, rubric, get_text_fn, get_id_fn,
                progress_callback, max_retries,
            )

        self._log_failures(results, documents, get_id_fn)
        self._build_stats['documents_with_failures'] = sum(
            1 for state in states
            if state.error or state.leaf_failures > 0 or state.merge_failures > 0
        )
        self._build_stats['documents_processed'] = len(documents)
        logger.info("Batch processing complete: %d trees built", len(results))
        return results

    async def _build_trees_cascading(
        self,
        states: List[DocumentState],
        rubric: str,
        progress_callback: Optional[Callable],
    ) -> None:
        """
        Build trees with cascading execution across leaves and merges.

        This submits leaf summaries and merges as soon as their inputs are ready,
        allowing per-document tree construction to cascade without global barriers.
        """
        docs_to_build = [
            (idx, state)
            for idx, state in enumerate(states)
            if state.error is None and state.chunks
        ]

        if not docs_to_build:
            return

        plans: Dict[int, TopologyPlan] = {}
        remaining_deps_by_doc: Dict[int, Dict[int, int]] = {}
        completed_leaves: Dict[int, Dict[int, Node]] = {}
        completed_merges: Dict[int, Dict[int, Node]] = {}

        total_leaves = 0
        total_merges = 0
        max_levels = 0

        leaf_queue: Deque[Tuple[int, int, str, str]] = deque()
        ready_merges: Deque[Tuple[int, int]] = deque()
        failed_docs: set[int] = set()

        # Build dependency graphs and enqueue leaves
        for state_idx, state in docs_to_build:
            plan = state.plan or self._create_doc_plan(state_idx, state.doc_id, state.chunks)
            state.plan = plan
            plans[state_idx] = plan
            remaining_deps_by_doc[state_idx] = plan.copy_remaining_deps()

            completed_leaves[state_idx] = {}
            completed_merges[state_idx] = {}

            for leaf_idx, chunk_obj in enumerate(state.chunks):
                leaf_queue.append((state_idx, leaf_idx, chunk_obj.text, state.doc_id))

            total_leaves += len(state.chunks)
            total_merges += int(plan.internal_count)
            max_levels = max(max_levels, plan.max_level)

        # Expose totals for completion_fraction property
        self._total_leaves = total_leaves
        self._total_merges = total_merges
        self._completed_leaves = 0
        self._completed_merges = 0

        self._build_stats['total_merges'] += total_merges
        self._build_stats['total_levels'] = max(self._build_stats['total_levels'], max_levels)

        max_inflight = max(1, self.config.max_concurrent_requests)
        logger.info(
            "  Cascading build: leaves=%d merges=%d max_inflight=%d",
            total_leaves,
            total_merges,
            max_inflight,
        )
        leaf_fallback_char_limit = max(
            240,
            min(int(getattr(self.config, "max_chunk_chars", 8000) or 8000) // 2, 1200),
        )
        merge_fallback_char_limit = max(
            600,
            min(int(getattr(self.config, "max_chunk_chars", 8000) or 8000), 2200),
        )
        empty_leaf_warn_budget = 12
        empty_merge_warn_budget = 12
        leaf_retry_recoveries = 0
        merge_retry_recoveries = 0
        leaf_interpreter_recoveries = 0
        merge_interpreter_recoveries = 0
        degenerate_leaf_fallbacks = 0
        degenerate_merge_fallbacks = 0
        strict_retry_rubric = (
            f"{rubric}\n\n"
            "OUTPUT FORMAT REQUIREMENTS:\n"
            "- Return ONLY the summary text.\n"
            "- Do not discuss instructions, users, or formatting rules.\n"
            "- No preamble, no analysis, no markdown, no labels."
        )
        fail_on_degenerate_summary = bool(
            getattr(self.config, "fail_on_degenerate_summary", False)
        )
        max_degenerate_leaf_fallbacks = max(
            0, int(getattr(self.config, "max_degenerate_leaf_fallbacks", 0) or 0)
        )
        max_degenerate_merge_fallbacks = max(
            0, int(getattr(self.config, "max_degenerate_merge_fallbacks", 0) or 0)
        )
        interpreter_enabled = str(
            os.getenv("TT_ENABLE_DEGENERATE_SUMMARY_INTERPRETER", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}
        interpreter_source_char_limit = max(
            800,
            min(int(getattr(self.config, "max_chunk_chars", 8000) or 8000), 6000),
        )
        interpreter_api_base = str(
            os.getenv("TT_SUMMARY_INTERPRETER_BASE_URL", "")
        ).strip()
        interpreter_model = str(
            os.getenv("TT_SUMMARY_INTERPRETER_MODEL", "default")
        ).strip() or "default"
        interpreter_api_key = str(
            os.getenv("TT_SUMMARY_INTERPRETER_API_KEY", "EMPTY")
        ).strip() or "EMPTY"
        interpreter_max_tokens = max(
            128,
            int(os.getenv("TT_SUMMARY_INTERPRETER_MAX_TOKENS", "700") or 700),
        )
        interpreter_timeout_seconds = max(
            5.0,
            float(os.getenv("TT_SUMMARY_INTERPRETER_TIMEOUT_SECONDS", "120") or 120.0),
        )
        interpreter_disable_thinking = str(
            os.getenv("TT_SUMMARY_INTERPRETER_DISABLE_THINKING", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}
        interpreter_client: Optional[Any] = None
        if interpreter_enabled and interpreter_api_base:
            try:
                from treepo._research.core.llm_client import LLMClient, LLMConfig

                interpreter_client = LLMClient(
                    LLMConfig(
                        base_url=interpreter_api_base,
                        model=interpreter_model,
                        api_key=interpreter_api_key,
                        max_tokens=interpreter_max_tokens,
                        temperature=0.0,
                        max_retries=1,
                        timeout=interpreter_timeout_seconds,
                    ),
                    enable_cache=False,
                )
                logger.info(
                    "  Second-pass interpreter endpoint enabled: base_url=%s model=%s",
                    interpreter_api_base,
                    interpreter_model,
                )
            except Exception as exc:
                logger.warning(
                    "  Could not initialize second-pass interpreter endpoint (%s). "
                    "Falling back to primary strategy for interpreter pass.",
                    exc,
                )
                interpreter_client = None

        def _truncate_for_fallback(text: str, *, limit: int) -> str:
            raw = str(text or "").strip()
            if not raw:
                return ""
            if len(raw) <= limit:
                return raw
            marker = "\n...\n"
            remaining = max(1, limit - len(marker))
            head_chars = max(1, remaining // 2)
            tail_chars = max(1, remaining - head_chars)
            return f"{raw[:head_chars].rstrip()}{marker}{raw[-tail_chars:].lstrip()}"

        def _fallback_leaf_summary(text: str) -> str:
            cleaned = clean_summary_text(text)
            if cleaned:
                return _truncate_for_fallback(cleaned, limit=leaf_fallback_char_limit)
            return _truncate_for_fallback(str(text or ""), limit=leaf_fallback_char_limit)

        def _fallback_merge_summary(left_summary: str, right_summary: str) -> str:
            left_clean = clean_summary_text(left_summary) or str(left_summary or "").strip()
            right_clean = clean_summary_text(right_summary) or str(right_summary or "").strip()
            if left_clean and right_clean:
                combined = f"{left_clean}\n\n{right_clean}"
            else:
                combined = left_clean or right_clean
            return _truncate_for_fallback(combined, limit=merge_fallback_char_limit)

        async def _second_pass_interpret(
            *,
            mode: str,
            source_text: str,
            candidate_output: str,
        ) -> str:
            source_clean = _truncate_for_fallback(
                str(source_text or ""),
                limit=interpreter_source_char_limit,
            )
            candidate_clean = _truncate_for_fallback(
                str(candidate_output or ""),
                limit=2400,
            )
            if not source_clean:
                return ""
            draft_block = candidate_clean or "[EMPTY_OR_DEGENERATE_DRAFT_OUTPUT]"

            interpretation_prompt = (
                f"MODE: {mode}\n\n"
                f"RUBRIC:\n{rubric}\n\n"
                "TASK:\n"
                "- You are cleaning another model's draft output.\n"
                "- The draft may contain planning text (e.g., 'Thinking Process').\n"
                "- Recover the intended final summary and preserve rubric-relevant information.\n"
                "- If the draft is empty or unusable, produce a clean summary directly from SOURCE_TEXT.\n"
                "- Return ONLY the cleaned summary text.\n"
                "- Do not include preamble, analysis, labels, or markdown.\n\n"
                f"SOURCE_TEXT:\n{source_clean}\n\n"
                f"DRAFT_OUTPUT_FROM_MODEL:\n{draft_block}\n"
            )
            interpreted = ""
            if interpreter_client is not None:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You clean draft LLM outputs into final usable summaries."
                        ),
                    },
                    {"role": "user", "content": interpretation_prompt},
                ]
                try:
                    interpreter_kwargs: Dict[str, Any] = {
                        "max_tokens": interpreter_max_tokens,
                        "temperature": 0.0,
                    }
                    if interpreter_disable_thinking:
                        # Qwen 3.x/3.5 supports this via chat template kwargs.
                        interpreter_kwargs["extra_body"] = {
                            "chat_template_kwargs": {"enable_thinking": False}
                        }
                    llm_response = await to_thread(
                        interpreter_client.chat,
                        messages,
                        **interpreter_kwargs,
                    )
                    interpreted = str(getattr(llm_response, "content", "") or "")
                except Exception:
                    # Retry once without model-specific kwargs for compatibility.
                    try:
                        llm_response = await to_thread(
                            interpreter_client.chat,
                            messages,
                            max_tokens=interpreter_max_tokens,
                            temperature=0.0,
                        )
                        interpreted = str(getattr(llm_response, "content", "") or "")
                    except Exception:
                        interpreted = ""
            else:
                interpretation_rubric = (
                    f"{rubric}\n\n"
                    "SECOND-PASS CLEANUP TASK:\n"
                    "- The draft output may contain planning text (e.g., 'Thinking Process').\n"
                    "- Recover the intended final summary and preserve rubric-relevant information.\n"
                    "- Return ONLY the cleaned summary text.\n"
                    "- Do not include preamble, analysis, labels, or markdown.\n"
                    f"- Mode: {mode}.\n"
                )
                try:
                    interpreted = await self.strategy.summarize(
                        interpretation_prompt,
                        interpretation_rubric,
                        temperature=0.0,
                    )
                except Exception:
                    interpreted = ""
            cleaned = clean_summary_text(interpreted)
            if not cleaned or is_degenerate_summary_text(cleaned):
                return ""
            return cleaned

        def _preview_text(text: Optional[str], limit: int = 220) -> str:
            normalized = " ".join(str(text or "").split())
            if len(normalized) <= limit:
                return normalized
            return f"{normalized[: limit - 3].rstrip()}..."

        def _maybe_raise_degenerate_failure(
            *,
            kind: str,
            count: int,
            doc_id: str,
            item_idx: int,
            level: Optional[int] = None,
            raw_summary: Optional[str] = None,
        ) -> None:
            if kind == "leaf":
                limit = max_degenerate_leaf_fallbacks
            else:
                limit = max_degenerate_merge_fallbacks

            should_fail = False
            reason = ""
            if limit > 0 and count >= limit:
                should_fail = True
                reason = f"limit reached ({count}/{limit})"
            elif fail_on_degenerate_summary and count >= 1:
                should_fail = True
                reason = "fail_on_degenerate_summary enabled"

            if not should_fail:
                return

            level_suffix = "" if level is None else f" level={int(level)}"
            preview = _preview_text(raw_summary)
            raise DegenerateSummaryFailure(
                "Degenerate summary fallback triggered abort: "
                f"kind={kind} doc_id={doc_id} item={int(item_idx)}{level_suffix} "
                f"count={int(count)} reason={reason} raw_preview={preview!r}"
            )

        async def summarize_leaf(
            doc_idx: int,
            leaf_idx: int,
            text: str,
            doc_id: str,
        ) -> tuple[int, int, Node, Optional[str]]:
            nonlocal empty_leaf_warn_budget, leaf_retry_recoveries, leaf_interpreter_recoveries
            nonlocal degenerate_leaf_fallbacks
            token = tournament_doc_id.set(str(doc_id))
            try:
                raw_summary = await self.strategy.summarize(text, rubric)
                cleaned_summary = clean_summary_text(raw_summary)
                degenerate_output = is_degenerate_summary_text(cleaned_summary)
                summary = "" if degenerate_output else cleaned_summary
                error: Optional[str] = None
                retry_raw_summary: Optional[str] = None

                if not summary:
                    try:
                        retry_raw_summary = await self.strategy.summarize(
                            text,
                            strict_retry_rubric,
                            temperature=0.0,
                        )
                    except Exception:
                        retry_raw_summary = None
                    retry_cleaned_summary = clean_summary_text(retry_raw_summary)
                    retry_degenerate = is_degenerate_summary_text(retry_cleaned_summary)
                    if retry_cleaned_summary and not retry_degenerate:
                        summary = retry_cleaned_summary
                        leaf_retry_recoveries += 1

                if not summary and interpreter_enabled:
                    interpreted_summary = await _second_pass_interpret(
                        mode="leaf",
                        source_text=text,
                        candidate_output=retry_raw_summary or raw_summary or "",
                    )
                    if interpreted_summary:
                        summary = interpreted_summary
                        leaf_interpreter_recoveries += 1
                        states[doc_idx].interpreter_leaf_recoveries += 1

                if not summary:
                    summary = _fallback_leaf_summary(text)
                    error = (
                        "degenerate_leaf_summary_output"
                        if degenerate_output
                        else "empty_leaf_summary_output"
                    )
                    states[doc_idx].empty_leaf_fallbacks += 1
                    if degenerate_output:
                        degenerate_leaf_fallbacks += 1
                        states[doc_idx].degenerate_leaf_fallbacks += 1
                        self._build_stats['degenerate_leaf_fallbacks'] += 1
                        _maybe_raise_degenerate_failure(
                            kind="leaf",
                            count=degenerate_leaf_fallbacks,
                            doc_id=str(doc_id),
                            item_idx=int(leaf_idx),
                            raw_summary=raw_summary,
                        )
                    if empty_leaf_warn_budget > 0:
                        logger.warning(
                            "Leaf summary fallback for doc %s chunk %d (%s) raw_preview=%r",
                            str(doc_id),
                            int(leaf_idx),
                            "degenerate_model_output" if degenerate_output else "empty_model_output",
                            _preview_text(raw_summary),
                        )
                        empty_leaf_warn_budget -= 1
                node_id = f"d{doc_idx}_leaf_{leaf_idx}"
                return doc_idx, leaf_idx, leaf(text, summary=summary, node_id=node_id), error
            except Exception as e:
                logger.error(f"Leaf summarization failed for doc {doc_idx} chunk {leaf_idx}: {e}")
                fallback_summary = _fallback_leaf_summary(text)
                node_id = f"d{doc_idx}_leaf_{leaf_idx}"
                return doc_idx, leaf_idx, leaf(text, summary=fallback_summary, node_id=node_id), str(e)
            finally:
                tournament_doc_id.reset(token)

        async def execute_merge(doc_idx: int, merge_id: int) -> tuple[int, int, Node]:
            nonlocal empty_merge_warn_budget, merge_retry_recoveries, merge_interpreter_recoveries
            nonlocal degenerate_merge_fallbacks
            plan = plans[doc_idx]
            merge_task = plan.internal_nodes[merge_id]
            assert merge_task.left is not None and merge_task.right is not None

            left = (
                completed_merges[doc_idx][merge_task.left.index]
                if merge_task.left.is_internal
                else completed_leaves[doc_idx][merge_task.left.index]
            )
            right = (
                completed_merges[doc_idx][merge_task.right.index]
                if merge_task.right.is_internal
                else completed_leaves[doc_idx][merge_task.right.index]
            )

            token = tournament_doc_id.set(str(states[doc_idx].doc_id))
            try:
                merge_exception: Optional[Exception] = None
                merge_degenerate_output = False
                raw_summary: Optional[str] = None
                retry_raw_summary: Optional[str] = None
                try:
                    raw_summary = await self.strategy.merge(left.summary, right.summary, rubric)
                    cleaned_summary = clean_summary_text(raw_summary)
                    merge_degenerate_output = is_degenerate_summary_text(cleaned_summary)
                    summary = "" if merge_degenerate_output else cleaned_summary
                except Exception as exc:
                    merge_exception = exc
                    summary = ""

                if not summary:
                    retry_exception: Optional[Exception] = None
                    try:
                        retry_raw_summary = await self.strategy.merge(
                            left.summary,
                            right.summary,
                            strict_retry_rubric,
                            temperature=0.0,
                        )
                    except Exception as exc:
                        retry_exception = exc
                    retry_cleaned_summary = clean_summary_text(retry_raw_summary)
                    retry_degenerate_output = is_degenerate_summary_text(retry_cleaned_summary)
                    if retry_cleaned_summary and not retry_degenerate_output:
                        summary = retry_cleaned_summary
                        merge_retry_recoveries += 1
                    elif merge_exception is None and retry_exception is not None:
                        merge_exception = retry_exception

                if not summary and interpreter_enabled:
                    interpreted_summary = await _second_pass_interpret(
                        mode="merge",
                        source_text=f"SUMMARY 1:\n{left.summary}\n\nSUMMARY 2:\n{right.summary}",
                        candidate_output=retry_raw_summary or raw_summary or "",
                    )
                    if interpreted_summary:
                        summary = interpreted_summary
                        merge_interpreter_recoveries += 1
                        states[doc_idx].interpreter_merge_recoveries += 1

                if not summary:
                    summary = _fallback_merge_summary(left.summary, right.summary)
                    if not summary and merge_exception is not None:
                        raise merge_exception
                    states[doc_idx].empty_merge_fallbacks += 1
                    if merge_degenerate_output:
                        degenerate_merge_fallbacks += 1
                        states[doc_idx].degenerate_merge_fallbacks += 1
                        self._build_stats['degenerate_merge_fallbacks'] += 1
                        _maybe_raise_degenerate_failure(
                            kind="merge",
                            count=degenerate_merge_fallbacks,
                            doc_id=str(states[doc_idx].doc_id),
                            item_idx=int(merge_id),
                            level=int(merge_task.level),
                            raw_summary=raw_summary,
                        )
                    states[doc_idx].merge_failures += 1
                    self._build_stats['merge_failures'] += 1
                    if empty_merge_warn_budget > 0:
                        reason = (
                            f"exception={merge_exception}"
                            if merge_exception is not None
                            else (
                                "degenerate_model_output"
                                if merge_degenerate_output
                                else "empty_model_output"
                            )
                        )
                        logger.warning(
                            "Merge summary fallback for doc %s merge %d level %d (%s) raw_preview=%r",
                            str(states[doc_idx].doc_id),
                            int(merge_id),
                            int(merge_task.level),
                            reason,
                            _preview_text(raw_summary),
                        )
                        empty_merge_warn_budget -= 1

                return doc_idx, merge_id, node(
                    left=left,
                    right=right,
                    summary=summary,
                    node_id=str(merge_task.node_id),
                )
            finally:
                tournament_doc_id.reset(token)

        async def summarize_leaf_batch(
            batch_items: Sequence[Tuple[int, int, str, str]],
        ) -> List[tuple[int, int, Node, Optional[str]]]:
            nonlocal leaf_retry_recoveries, leaf_interpreter_recoveries
            nonlocal degenerate_leaf_fallbacks
            if not batch_items:
                return []
            payloads = [
                {
                    "content": text,
                    "rubric": rubric,
                    "temperature": 0.7,
                    "doc_id": doc_id,
                }
                for doc_idx, leaf_idx, text, doc_id in batch_items
            ]
            raw_summaries: List[str]
            bulk_summarize = getattr(self.strategy, "summarize_many", None)
            try:
                if callable(bulk_summarize):
                    raw_summaries = list(await bulk_summarize(payloads))
                else:
                    raw_summaries = list(
                        await asyncio.gather(
                            *(
                                self.strategy.summarize(text, rubric)
                                for _doc_idx, _leaf_idx, text, _doc_id in batch_items
                            ),
                            return_exceptions=False,
                        )
                    )
            except Exception as exc:
                logger.warning(
                    "Unified leaf batch failed (%s items): %s. Falling back to empty raw outputs.",
                    len(batch_items),
                    exc,
                )
                raw_summaries = ["" for _ in batch_items]

            results: List[tuple[int, int, Node, Optional[str]]] = []
            for item, raw_summary in zip(batch_items, raw_summaries):
                doc_idx, leaf_idx, text, doc_id = item
                token = tournament_doc_id.set(str(doc_id))
                try:
                    cleaned_summary = clean_summary_text(raw_summary)
                    degenerate_output = is_degenerate_summary_text(cleaned_summary)
                    summary = "" if degenerate_output else cleaned_summary
                    error: Optional[str] = None
                    retry_raw_summary: Optional[str] = None

                    if not summary:
                        try:
                            retry_raw_summary = await self.strategy.summarize(
                                text,
                                strict_retry_rubric,
                                temperature=0.0,
                            )
                        except Exception:
                            retry_raw_summary = None
                        retry_cleaned_summary = clean_summary_text(retry_raw_summary)
                        retry_degenerate = is_degenerate_summary_text(retry_cleaned_summary)
                        if retry_cleaned_summary and not retry_degenerate:
                            summary = retry_cleaned_summary
                            leaf_retry_recoveries += 1

                    if not summary and interpreter_enabled:
                        interpreted_summary = await _second_pass_interpret(
                            mode="leaf",
                            source_text=text,
                            candidate_output=retry_raw_summary or raw_summary or "",
                        )
                        if interpreted_summary:
                            summary = interpreted_summary
                            leaf_interpreter_recoveries += 1
                            states[doc_idx].interpreter_leaf_recoveries += 1

                    if not summary:
                        summary = _fallback_leaf_summary(text)
                        error = (
                            "degenerate_leaf_summary_output"
                            if degenerate_output
                            else "empty_leaf_summary_output"
                        )
                        states[doc_idx].empty_leaf_fallbacks += 1
                        if degenerate_output:
                            degenerate_leaf_fallbacks += 1
                            states[doc_idx].degenerate_leaf_fallbacks += 1
                            self._build_stats['degenerate_leaf_fallbacks'] += 1
                            _maybe_raise_degenerate_failure(
                                kind="leaf",
                                count=degenerate_leaf_fallbacks,
                                doc_id=str(doc_id),
                                item_idx=int(leaf_idx),
                                raw_summary=raw_summary,
                            )
                    node_id = f"d{doc_idx}_leaf_{leaf_idx}"
                    results.append((doc_idx, leaf_idx, leaf(text, summary=summary, node_id=node_id), error))
                except Exception as exc:
                    logger.error(
                        "Leaf summarization failed for doc %s chunk %s in unified batch: %s",
                        doc_idx,
                        leaf_idx,
                        exc,
                    )
                    fallback_summary = _fallback_leaf_summary(text)
                    node_id = f"d{doc_idx}_leaf_{leaf_idx}"
                    results.append((doc_idx, leaf_idx, leaf(text, summary=fallback_summary, node_id=node_id), str(exc)))
                finally:
                    tournament_doc_id.reset(token)
            return results

        async def execute_merge_batch(
            batch_items: Sequence[Tuple[int, int]],
        ) -> List[tuple[int, int, Node]]:
            nonlocal merge_retry_recoveries, merge_interpreter_recoveries
            nonlocal degenerate_merge_fallbacks
            if not batch_items:
                return []
            merge_contexts: List[Tuple[int, int, Node, Node, Any]] = []
            payloads: List[Dict[str, Any]] = []
            for doc_idx, merge_id in batch_items:
                plan = plans[doc_idx]
                merge_task = plan.internal_nodes[merge_id]
                assert merge_task.left is not None and merge_task.right is not None
                left = (
                    completed_merges[doc_idx][merge_task.left.index]
                    if merge_task.left.is_internal
                    else completed_leaves[doc_idx][merge_task.left.index]
                )
                right = (
                    completed_merges[doc_idx][merge_task.right.index]
                    if merge_task.right.is_internal
                    else completed_leaves[doc_idx][merge_task.right.index]
                )
                merge_contexts.append((doc_idx, merge_id, left, right, merge_task))
                payloads.append(
                    {
                        "left": left.summary,
                        "right": right.summary,
                        "rubric": rubric,
                        "temperature": 0.7,
                        "doc_id": states[doc_idx].doc_id,
                    }
                )
            bulk_merge = getattr(self.strategy, "merge_many", None)
            try:
                if callable(bulk_merge):
                    raw_summaries = list(await bulk_merge(payloads))
                else:
                    raw_summaries = list(
                        await asyncio.gather(
                            *(
                                self.strategy.merge(left.summary, right.summary, rubric)
                                for _doc_idx, _merge_id, left, right, _merge_task in merge_contexts
                            ),
                            return_exceptions=False,
                        )
                    )
            except Exception as exc:
                logger.warning(
                    "Unified merge batch failed (%s items): %s. Falling back to empty raw outputs.",
                    len(batch_items),
                    exc,
                )
                raw_summaries = ["" for _ in batch_items]

            results: List[tuple[int, int, Node]] = []
            for merge_ctx, raw_summary in zip(merge_contexts, raw_summaries):
                doc_idx, merge_id, left, right, merge_task = merge_ctx
                token = tournament_doc_id.set(str(states[doc_idx].doc_id))
                try:
                    merge_exception: Optional[Exception] = None
                    merge_degenerate_output = False
                    retry_raw_summary: Optional[str] = None
                    cleaned_summary = clean_summary_text(raw_summary)
                    merge_degenerate_output = is_degenerate_summary_text(cleaned_summary)
                    summary = "" if merge_degenerate_output else cleaned_summary

                    if not summary:
                        retry_exception: Optional[Exception] = None
                        try:
                            retry_raw_summary = await self.strategy.merge(
                                left.summary,
                                right.summary,
                                strict_retry_rubric,
                                temperature=0.0,
                            )
                        except Exception as exc:
                            retry_exception = exc
                        retry_cleaned_summary = clean_summary_text(retry_raw_summary)
                        retry_degenerate_output = is_degenerate_summary_text(retry_cleaned_summary)
                        if retry_cleaned_summary and not retry_degenerate_output:
                            summary = retry_cleaned_summary
                            merge_retry_recoveries += 1
                        elif retry_exception is not None:
                            merge_exception = retry_exception

                    if not summary and interpreter_enabled:
                        interpreted_summary = await _second_pass_interpret(
                            mode="merge",
                            source_text=f"SUMMARY 1:\n{left.summary}\n\nSUMMARY 2:\n{right.summary}",
                            candidate_output=retry_raw_summary or raw_summary or "",
                        )
                        if interpreted_summary:
                            summary = interpreted_summary
                            merge_interpreter_recoveries += 1
                            states[doc_idx].interpreter_merge_recoveries += 1

                    if not summary:
                        summary = _fallback_merge_summary(left.summary, right.summary)
                        if not summary and merge_exception is not None:
                            raise merge_exception
                        states[doc_idx].empty_merge_fallbacks += 1
                        if merge_degenerate_output:
                            degenerate_merge_fallbacks += 1
                            states[doc_idx].degenerate_merge_fallbacks += 1
                            self._build_stats['degenerate_merge_fallbacks'] += 1
                            _maybe_raise_degenerate_failure(
                                kind="merge",
                                count=degenerate_merge_fallbacks,
                                doc_id=str(states[doc_idx].doc_id),
                                item_idx=int(merge_id),
                                level=int(merge_task.level),
                                raw_summary=raw_summary,
                            )
                        states[doc_idx].merge_failures += 1
                        self._build_stats['merge_failures'] += 1

                    results.append(
                        (
                            doc_idx,
                            merge_id,
                            node(
                                left=left,
                                right=right,
                                summary=summary,
                                node_id=str(merge_task.node_id),
                            ),
                        )
                    )
                finally:
                    tournament_doc_id.reset(token)
            return results

        runtime_mode = resolve_runtime_mode(getattr(self.config, "runtime_mode", None))
        pending: Dict[asyncio.Task, Dict[str, Any]] = {}
        completed_leaves_count = 0
        completed_merges_count = 0
        prefer_merge = True
        max_pending_batches = max(1, min(8, max_inflight))
        progress_started = time.monotonic()
        last_progress_log = progress_started
        last_progress_completed = 0

        def _maybe_log_progress(force: bool = False) -> None:
            nonlocal last_progress_log, last_progress_completed
            completed_total = completed_leaves_count + completed_merges_count
            now = time.monotonic()
            should_log = force
            if not should_log:
                if (now - last_progress_log) >= 30.0:
                    should_log = True
                elif (completed_total - last_progress_completed) >= 250:
                    should_log = True
            if not should_log:
                return

            elapsed = max(1e-6, now - progress_started)
            rate = completed_total / elapsed
            total_tasks = total_leaves + total_merges
            stats = None
            try:
                stats = getattr(getattr(self.strategy, "client", None), "stats", None)
            except Exception:
                stats = None
            stats_str = f" stats={stats}" if stats is not None else ""

            logger.info(
                "  Cascading progress: leaves=%d/%d merges=%d/%d done=%d/%d pending=%d leaf_q=%d merge_q=%d rate=%.2f items/s%s",
                completed_leaves_count,
                total_leaves,
                completed_merges_count,
                total_merges,
                completed_total,
                total_tasks,
                len(pending),
                len(leaf_queue),
                len(ready_merges),
                rate,
                stats_str,
            )
            last_progress_log = now
            last_progress_completed = completed_total

        def _sort_ready_merges() -> None:
            """Sort ready merges by level (desc) to prioritize critical path.

            Higher-level merges are closer to the root and block more
            downstream work — scheduling them first reduces wall-clock time
            (DualPath Opt 6: size-aware merge scheduling).
            """
            if len(ready_merges) <= 1:
                return
            items = list(ready_merges)
            items.sort(
                key=lambda pair: (
                    -plans[pair[0]].internal_nodes[pair[1]].level,
                    -int(
                        plans[pair[0]].internal_nodes[pair[1]].metadata.get(
                            "estimated_input_tokens",
                            0,
                        )
                    ),
                )
            )
            ready_merges.clear()
            ready_merges.extend(items)

        def pump_ready_queue() -> None:
            nonlocal prefer_merge
            # Re-sort when new merges have been enqueued
            if len(ready_merges) > 1:
                _sort_ready_merges()

            if runtime_mode == RUNTIME_MODE_UNIFIED_V2:
                while len(pending) < max_pending_batches and (ready_merges or leaf_queue):
                    choose_merge = False
                    if ready_merges and leaf_queue:
                        choose_merge = prefer_merge
                    elif ready_merges:
                        choose_merge = True

                    if choose_merge and ready_merges:
                        candidate_items = [
                            (doc_idx, merge_id)
                            for doc_idx, merge_id in ready_merges
                            if doc_idx not in failed_docs
                        ]
                        if not candidate_items:
                            ready_merges.clear()
                            continue
                        work_items = []
                        merge_lookup: Dict[str, Tuple[int, int]] = {}
                        for doc_idx, merge_id in candidate_items:
                            merge_task = plans[doc_idx].internal_nodes[merge_id]
                            item_id = f"merge:{doc_idx}:{merge_id}"
                            merge_lookup[item_id] = (doc_idx, merge_id)
                            work_items.append(
                                WorkItem(
                                    item_id=item_id,
                                    backend_family=str(_LLM_TEXT_PROGRAM_FAMILY),
                                    op_kind="merge",
                                    topology_signature=plans[doc_idx].topology_signature,
                                    supervision_mask="merge",
                                    doc_id=str(states[doc_idx].doc_id),
                                    payload=(doc_idx, merge_id),
                                    estimated_tokens=int(
                                        merge_task.metadata.get("estimated_input_tokens", 0)
                                    ),
                                    estimated_nodes=1,
                                    estimated_merge_ops=1,
                                    padding_multiple=1,
                                    padding_length=max(
                                        1,
                                        int(merge_task.metadata.get("estimated_input_tokens", 0)),
                                    ),
                                )
                            )
                        batches = plan_work_batches(
                            work_items,
                            max_docs=max_inflight,
                            max_total_tokens=0,
                            max_total_nodes=0,
                            max_total_merge_ops=0,
                            plan_cache=self._plan_cache,
                        )
                        if not batches:
                            break
                        batch = batches[0]
                        batch_pairs = [merge_lookup[item.item_id] for item in batch.items]
                        selected = {pair for pair in batch_pairs}
                        ready_merges.clear()
                        ready_merges.extend(
                            [item for item in candidate_items if item not in selected]
                        )
                        task = asyncio.create_task(execute_merge_batch(batch_pairs))
                        pending[task] = {"kind": "merge_batch", "items": batch_pairs}
                        self._runtime_telemetry.add_batch(
                            batch,
                            token_budget=0,
                            node_budget=0,
                            max_docs_budget=max_inflight,
                            fallback_reason="llm_bulk_merge",
                        )
                        prefer_merge = False
                        continue

                    if leaf_queue:
                        candidate_items = [
                            (doc_idx, leaf_idx, text, doc_id)
                            for doc_idx, leaf_idx, text, doc_id in leaf_queue
                            if doc_idx not in failed_docs
                        ]
                        if not candidate_items:
                            leaf_queue.clear()
                            continue
                        work_items = []
                        leaf_lookup: Dict[str, Tuple[int, int, str, str]] = {}
                        for doc_idx, leaf_idx, text, doc_id in candidate_items:
                            item_id = f"leaf:{doc_idx}:{leaf_idx}"
                            leaf_lookup[item_id] = (doc_idx, leaf_idx, text, doc_id)
                            work_items.append(
                                WorkItem(
                                    item_id=item_id,
                                    backend_family=str(_LLM_TEXT_PROGRAM_FAMILY),
                                    op_kind="summarize",
                                    topology_signature=plans[doc_idx].topology_signature,
                                    supervision_mask="leaf",
                                    doc_id=str(doc_id),
                                    payload=(doc_idx, leaf_idx),
                                    estimated_tokens=max(1, len(text) // 4),
                                    estimated_nodes=1,
                                    estimated_merge_ops=0,
                                    padding_multiple=1,
                                    padding_length=max(1, len(text) // 4),
                                )
                            )
                        batches = plan_work_batches(
                            work_items,
                            max_docs=max_inflight,
                            max_total_tokens=0,
                            max_total_nodes=0,
                            max_total_merge_ops=0,
                            plan_cache=self._plan_cache,
                        )
                        if not batches:
                            break
                        batch = batches[0]
                        batch_items = [leaf_lookup[item.item_id] for item in batch.items]
                        selected = {(
                            doc_idx,
                            leaf_idx,
                        ) for doc_idx, leaf_idx, _text, _doc_id in batch_items}
                        leaf_queue.clear()
                        leaf_queue.extend(
                            [
                                item
                                for item in candidate_items
                                if (item[0], item[1]) not in selected
                            ]
                        )
                        task = asyncio.create_task(summarize_leaf_batch(batch_items))
                        pending[task] = {"kind": "leaf_batch", "items": batch_items}
                        self._runtime_telemetry.add_batch(
                            batch,
                            token_budget=0,
                            node_budget=0,
                            max_docs_budget=max_inflight,
                            fallback_reason="llm_bulk_leaf",
                        )
                        prefer_merge = True
                return

            while len(pending) < max_inflight and (ready_merges or leaf_queue):
                choose_merge = False
                if ready_merges and leaf_queue:
                    choose_merge = prefer_merge
                elif ready_merges:
                    choose_merge = True

                if choose_merge and ready_merges:
                    doc_idx, merge_id = ready_merges.popleft()
                    if doc_idx in failed_docs:
                        continue
                    task = asyncio.create_task(execute_merge(doc_idx, merge_id))
                    pending[task] = {"kind": "merge", "doc_idx": doc_idx, "item_id": merge_id}
                    prefer_merge = False
                    continue

                if leaf_queue:
                    doc_idx, leaf_idx, text, doc_id = leaf_queue.popleft()
                    if doc_idx in failed_docs:
                        continue
                    task = asyncio.create_task(summarize_leaf(doc_idx, leaf_idx, text, doc_id))
                    pending[task] = {"kind": "leaf", "doc_idx": doc_idx, "item_id": leaf_idx}
                    prefer_merge = True

        pump_ready_queue()

        try:
            while pending:
                _maybe_log_progress()
                done, _ = await asyncio.wait(
                    pending.keys(),
                    return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    meta = pending.pop(task)
                    kind = str(meta.get("kind", ""))
                    if task.cancelled():
                        continue

                    if kind == "leaf":
                        doc_idx = int(meta.get("doc_idx", -1))
                        item_id = int(meta.get("item_id", -1))
                        try:
                            doc_idx, leaf_idx, leaf_node, error = await task
                        except Exception as e:
                            if isinstance(e, DegenerateSummaryFailure):
                                raise
                            logger.error(f"Leaf task failed for doc {doc_idx} chunk {item_id}: {e}")
                            states[doc_idx].leaf_failures += 1
                            self._build_stats['leaf_failures'] += 1
                            continue

                        completed_leaves[doc_idx][leaf_idx] = leaf_node
                        completed_leaves_count += 1
                        self._completed_leaves = completed_leaves_count
                        if error:
                            states[doc_idx].leaf_failures += 1
                            self._build_stats['leaf_failures'] += 1

                        for dependent_id in plans[doc_idx].dependents_by_leaf.get(leaf_idx, []):
                            remaining_deps_by_doc[doc_idx][dependent_id] -= 1
                            if remaining_deps_by_doc[doc_idx][dependent_id] == 0:
                                ready_merges.append((doc_idx, dependent_id))

                        if progress_callback:
                            progress_callback("leaf", completed_leaves_count, total_leaves)

                    elif kind == "merge":
                        doc_idx = int(meta.get("doc_idx", -1))
                        item_id = int(meta.get("item_id", -1))
                        try:
                            doc_idx, merge_id, merged_node = await task
                        except Exception as e:
                            if isinstance(e, DegenerateSummaryFailure):
                                raise
                            logger.error(f"Cascading merge failed for doc {doc_idx} task {item_id}: {e}")
                            states[doc_idx].merge_failures += 1
                            self._build_stats['merge_failures'] += 1
                            failed_docs.add(doc_idx)

                            # Cancel any in-flight work for this document
                            tasks_to_cancel = [
                                t
                                for t, task_meta in pending.items()
                                if int(task_meta.get("doc_idx", -1)) == doc_idx
                            ]
                            for t in tasks_to_cancel:
                                t.cancel()
                                pending.pop(t, None)
                            if tasks_to_cancel:
                                await cancel_tasks(tasks_to_cancel, timeout=self.config.task_cancel_timeout)

                            if leaf_queue:
                                leaf_queue = deque([item for item in leaf_queue if item[0] != doc_idx])
                            if ready_merges:
                                ready_merges = deque([item for item in ready_merges if item[0] != doc_idx])
                            continue

                        completed_merges[doc_idx][merge_id] = merged_node
                        completed_merges_count += 1
                        self._completed_merges = completed_merges_count

                        for dependent_id in plans[doc_idx].dependents_by_internal.get(merge_id, []):
                            remaining_deps_by_doc[doc_idx][dependent_id] -= 1
                            if remaining_deps_by_doc[doc_idx][dependent_id] == 0:
                                ready_merges.append((doc_idx, dependent_id))

                        if progress_callback:
                            progress_callback("merge", completed_merges_count, total_merges)
                    elif kind == "leaf_batch":
                        batch_items = list(meta.get("items", []) or [])
                        try:
                            batch_results = await task
                        except Exception as e:
                            if isinstance(e, DegenerateSummaryFailure):
                                raise
                            logger.error(
                                "Unified leaf batch failed for %d items: %s",
                                len(batch_items),
                                e,
                            )
                            for doc_idx, _leaf_idx, _text, _doc_id in batch_items:
                                states[doc_idx].leaf_failures += 1
                                self._build_stats['leaf_failures'] += 1
                            continue
                        for doc_idx, leaf_idx, leaf_node, error in batch_results:
                            completed_leaves[doc_idx][leaf_idx] = leaf_node
                            completed_leaves_count += 1
                            self._completed_leaves = completed_leaves_count
                            if error:
                                states[doc_idx].leaf_failures += 1
                                self._build_stats['leaf_failures'] += 1
                            for dependent_id in plans[doc_idx].dependents_by_leaf.get(leaf_idx, []):
                                remaining_deps_by_doc[doc_idx][dependent_id] -= 1
                                if remaining_deps_by_doc[doc_idx][dependent_id] == 0:
                                    ready_merges.append((doc_idx, dependent_id))
                            if progress_callback:
                                progress_callback("leaf", completed_leaves_count, total_leaves)
                    elif kind == "merge_batch":
                        batch_items = list(meta.get("items", []) or [])
                        try:
                            batch_results = await task
                        except Exception as e:
                            if isinstance(e, DegenerateSummaryFailure):
                                raise
                            logger.error(
                                "Unified merge batch failed for %d items: %s",
                                len(batch_items),
                                e,
                            )
                            failed_doc_batch = {int(doc_idx) for doc_idx, _merge_id in batch_items}
                            for failed_doc_idx in failed_doc_batch:
                                states[failed_doc_idx].merge_failures += 1
                                self._build_stats['merge_failures'] += 1
                                failed_docs.add(failed_doc_idx)
                            continue
                        for doc_idx, merge_id, merged_node in batch_results:
                            completed_merges[doc_idx][merge_id] = merged_node
                            completed_merges_count += 1
                            self._completed_merges = completed_merges_count
                            for dependent_id in plans[doc_idx].dependents_by_internal.get(merge_id, []):
                                remaining_deps_by_doc[doc_idx][dependent_id] -= 1
                                if remaining_deps_by_doc[doc_idx][dependent_id] == 0:
                                    ready_merges.append((doc_idx, dependent_id))
                            if progress_callback:
                                progress_callback("merge", completed_merges_count, total_merges)

                pump_ready_queue()
        finally:
            if pending:
                await cancel_tasks(pending.keys(), timeout=self.config.task_cancel_timeout)

        # Finalize roots per document
        for state_idx, state in docs_to_build:
            plan = plans.get(state_idx)
            if plan is None:
                continue

            root_node: Optional[Node] = None
            if state_idx in failed_docs:
                if completed_merges[state_idx]:
                    root_node = completed_merges[state_idx][max(completed_merges[state_idx].keys())]
                elif completed_leaves[state_idx]:
                    root_node = completed_leaves[state_idx].get(0)
                    if root_node is None:
                        root_node = next(iter(completed_leaves[state_idx].values()), None)
            else:
                final_ref = plan.final_ref
                if final_ref.is_internal:
                    root_node = completed_merges[state_idx].get(final_ref.index)
                else:
                    root_node = completed_leaves[state_idx].get(final_ref.index)

            if root_node is None and completed_leaves[state_idx]:
                root_node = completed_leaves[state_idx].get(0)
                if root_node is None:
                    root_node = next(iter(completed_leaves[state_idx].values()), None)
                states[state_idx].merge_failures += 1
                self._build_stats['merge_failures'] += 1

            if root_node is not None:
                states[state_idx].current_level = [root_node]

        completed_docs = len(docs_to_build) - len(failed_docs)
        _maybe_log_progress(force=True)
        self._build_stats["runtime_mode"] = str(runtime_mode)
        self._build_stats["runtime_telemetry"] = self._runtime_telemetry.as_dict()
        self._build_stats["plan_cache"] = self._plan_cache.as_dict()
        logger.info(
            f"  Cascading build complete: {completed_docs}/{len(docs_to_build)} documents"
        )
        if leaf_retry_recoveries > 0 or merge_retry_recoveries > 0:
            logger.info(
                "  Strict retry recovered %d leaf and %d merge summaries before fallback",
                int(leaf_retry_recoveries),
                int(merge_retry_recoveries),
            )
        if leaf_interpreter_recoveries > 0 or merge_interpreter_recoveries > 0:
            logger.info(
                "  Second-pass interpreter recovered %d leaf and %d merge summaries",
                int(leaf_interpreter_recoveries),
                int(merge_interpreter_recoveries),
            )

        if progress_callback:
            progress_callback("pipelined_merge", completed_docs, len(docs_to_build))

    def _create_results(
        self,
        states: List[DocumentState],
        rubric: str,
    ) -> List[BuildResult]:
        """Convert document states to BuildResult objects."""
        results = []

        # Collect preferences if strategy supports it
        preferences = []
        if hasattr(self.strategy, 'get_preferences'):
            preferences = self.strategy.get_preferences()
        comparative_judgments = []
        if hasattr(self.strategy, "get_comparative_judgments"):
            comparative_judgments = self.strategy.get_comparative_judgments()

        for state in states:
            if state.error:
                # Create an empty result for failed documents
                results.append(BuildResult(
                    tree=Tree(root=leaf("", node_id="error"), rubric=rubric),
                    chunks_created=0,
                    nodes_created=0,
                    levels_created=0,
                    errors=[state.error],
                ))
                continue

            if not state.current_level:
                results.append(BuildResult(
                    tree=Tree(root=leaf("", node_id="empty"), rubric=rubric),
                    chunks_created=len(state.chunks),
                    nodes_created=0,
                    levels_created=0,
                    errors=["No nodes created"],
                ))
                continue

            # Get root node
            root = state.current_level[0]

            # Create tree
            tree = Tree(root=root, rubric=rubric)
            tree.metadata['doc_id'] = state.doc_id
            if state.plan and state.plan.plan_summary:
                tree.metadata['tree_plan'] = state.plan.plan_summary
            if state.leaf_failures or state.merge_failures:
                tree.metadata['leaf_failures'] = state.leaf_failures
                tree.metadata['merge_failures'] = state.merge_failures
            if state.empty_leaf_fallbacks or state.empty_merge_fallbacks:
                tree.metadata['empty_leaf_fallbacks'] = int(state.empty_leaf_fallbacks)
                tree.metadata['empty_merge_fallbacks'] = int(state.empty_merge_fallbacks)
            if state.degenerate_leaf_fallbacks or state.degenerate_merge_fallbacks:
                tree.metadata['degenerate_leaf_fallbacks'] = int(state.degenerate_leaf_fallbacks)
                tree.metadata['degenerate_merge_fallbacks'] = int(state.degenerate_merge_fallbacks)
            if state.interpreter_leaf_recoveries or state.interpreter_merge_recoveries:
                tree.metadata['interpreter_leaf_recoveries'] = int(state.interpreter_leaf_recoveries)
                tree.metadata['interpreter_merge_recoveries'] = int(state.interpreter_merge_recoveries)

            # Filter preferences for this document (if any)
            doc_id = str(state.doc_id)
            doc_preferences = [
                p for p in preferences
                if getattr(p, "source_example_id", "") == doc_id
                or getattr(p, "source_example_id", "").startswith(f"{doc_id}:")
            ] if preferences else []
            doc_comparative_judgments = [
                record
                for record in comparative_judgments
                if getattr(record, "source_example_id", "") == doc_id
                or getattr(record, "source_example_id", "").startswith(f"{doc_id}:")
            ] if comparative_judgments else []

            # Extract per-leaf info scores for content-weighted audit sampling.
            content_weights = extract_content_weights_from_chunks(state.chunks)

            results.append(BuildResult(
                tree=tree,
                chunks_created=len(state.chunks),
                nodes_created=tree.node_count,
                levels_created=tree.height + 1,
                errors=[],
                supervision=SupervisionDataset(
                    comparative_judgments=(
                        list(doc_comparative_judgments)
                        if doc_comparative_judgments
                        else [pair.to_comparative_judgment() for pair in doc_preferences]
                    )
                ),
                content_weights=content_weights,
            ))

        return results

    # -----------------------------------------------------------------
    # Unified tree integration
    # -----------------------------------------------------------------

    async def process_documents_unified(
        self,
        documents: List[Any],
        rubric: str,
        unified_trees: Dict[int, List[Any]],
        get_text_fn: Optional[Callable[[Any], str]] = None,
        get_id_fn: Optional[Callable[[Any], str]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        max_retries: int = 0,
    ) -> List[BuildResult]:
        """Process documents using pre-built unified tree topologies.

        This mirrors ``process_documents()`` but instead of chunking text via
        ``chunk_for_ops()``, it derives ``TextChunk`` objects and merge plans
        from pre-built ``EmbeddingTreeNode`` lists (one per document).

        The LLM leaf-summarisation and cascading merge calls go through the
        **same** ``_build_trees_cascading()`` infrastructure, ensuring all
        requests are globally pooled across documents.

        Args:
            documents: List of documents to process.
            rubric: Information preservation criteria.
            unified_trees: Mapping from document index to its pre-built
                ``EmbeddingTreeNode`` list (bottom-up order, as returned by
                ``build_unified_tree()``).
            get_text_fn: Extract text from document (default: ``str(doc)``).
            get_id_fn: Extract ID from document (default: index-based).
            progress_callback: Optional ``callback(phase, completed, total)``.
            max_retries: Retry count for failed documents.

        Returns:
            List of ``BuildResult``, one per document.
        """
        if get_text_fn is None:
            get_text_fn = lambda doc: str(doc) if isinstance(doc, str) else getattr(doc, 'text', str(doc))
        if get_id_fn is None:
            get_id_fn = lambda doc: str(hash(doc))

        # Phase 1: Convert unified trees into DocumentStates with TextChunks
        logger.info("Phase 1 (unified): Converting %d pre-built trees to chunks...", len(documents))
        states = self._chunk_from_unified_trees(
            documents, unified_trees, get_text_fn, get_id_fn, progress_callback,
        )

        results = await self._build_and_finalize(
            states, documents, rubric, get_text_fn, get_id_fn,
            progress_callback, max_retries,
        )

        # Enrich results with unified tree metadata
        for i, result in enumerate(results):
            nodes = unified_trees.get(i)
            if nodes and not result.errors:
                result.tree.metadata['unified_tree'] = True
                result.tree.metadata['unified_tree_node_count'] = len(nodes)
                leaves = [n for n in nodes if getattr(n, 'is_leaf', False)]
                result.tree.metadata['chunk_boundaries'] = [
                    {'char_start': n.char_start, 'char_end': n.char_end}
                    for n in leaves
                ]

        return results

    def _chunk_from_unified_trees(
        self,
        documents: List[Any],
        unified_trees: Dict[int, List[Any]],
        get_text_fn: Callable[[Any], str],
        get_id_fn: Callable[[Any], str],
        progress_callback: Optional[Callable],
    ) -> List[DocumentState]:
        """Convert pre-built unified trees into DocumentState objects.

        For each document whose index appears in *unified_trees*, leaf nodes
        are converted to ``TextChunk`` objects and a ``DocPlan`` merge graph
        is derived via the existing ``_create_doc_plan()`` helper.
        """
        states: List[DocumentState] = []
        total_chunks = 0

        for i, doc in enumerate(documents):
            doc_id = get_id_fn(doc)
            state_idx = len(states)

            nodes = unified_trees.get(i)
            if not nodes:
                # Fallback: chunk the text the standard way
                try:
                    text = get_text_fn(doc)
                    if not text or not text.strip():
                        states.append(DocumentState(doc_id=doc_id, sample=doc, error="No text content"))
                        continue
                    chunks = chunk(
                        text,
                        max_chars=self.config.max_chunk_chars,
                        max_tokens=self.config.max_chunk_tokens,
                        token_encoding=self.config.chunk_token_encoding,
                        overlap_tokens=self.config.chunk_overlap_tokens,
                        strategy=self.config.chunk_strategy,
                    )
                    if not chunks:
                        states.append(DocumentState(doc_id=doc_id, sample=doc, error="Chunking failed"))
                        continue
                    plan = self._create_doc_plan(state_idx, doc_id, chunks)
                    states.append(DocumentState(doc_id=doc_id, sample=doc, chunks=chunks, plan=plan))
                    total_chunks += len(chunks)
                except Exception as e:
                    logger.error("Failed to chunk document %s: %s", doc_id, e)
                    states.append(DocumentState(doc_id=doc_id, sample=doc, error=str(e)))
                continue

            # Build TextChunks from unified tree leaf nodes
            try:
                leaves = [n for n in nodes if getattr(n, 'is_leaf', False)]
                if not leaves:
                    states.append(DocumentState(doc_id=doc_id, sample=doc, error="Unified tree has no leaves"))
                    continue

                chunks = []
                for leaf_idx, leaf_node in enumerate(leaves):
                    text_span = getattr(leaf_node, 'text_span', '') or ''
                    chunks.append(TextChunk(
                        text=text_span,
                        chunk_index=leaf_idx,
                        start_char=getattr(leaf_node, 'char_start', 0),
                        end_char=getattr(leaf_node, 'char_end', len(text_span)),
                        token_count=len(text_span) // 4,  # rough estimate
                    ))

                plan = build_unified_topology_plan(
                    doc_index=int(state_idx),
                    doc_id=str(doc_id),
                    nodes=nodes,
                )
                states.append(DocumentState(doc_id=doc_id, sample=doc, chunks=chunks, plan=plan))
                total_chunks += len(chunks)

            except Exception as e:
                logger.error("Failed to convert unified tree for document %s: %s", doc_id, e)
                states.append(DocumentState(doc_id=doc_id, sample=doc, error=str(e)))

        self._build_stats['total_chunks'] = total_chunks
        logger.info("  Converted %d documents into %d total chunks (unified)", len(documents), total_chunks)

        if progress_callback:
            progress_callback("chunk", len(documents), len(documents))

        return states

    async def _retry_failed_documents(
        self,
        results: List[BuildResult],
        documents: List[Any],
        rubric: str,
        get_text_fn: Callable[[Any], str],
        get_id_fn: Callable[[Any], str],
        progress_callback: Optional[Callable],
        max_retries: int,
    ) -> List[BuildResult]:
        """
        Retry processing for failed documents.

        Args:
            results: Current results list (will be modified in place for successes)
            documents: Original documents
            rubric: Information preservation criteria
            get_text_fn: Function to extract text from document
            get_id_fn: Function to extract ID from document
            progress_callback: Optional progress callback
            max_retries: Number of retry attempts

        Returns:
            Updated results list with successful retries replaced
        """
        for attempt in range(1, max_retries + 1):
            # Find failed document indices
            failed_indices = [i for i, r in enumerate(results) if r.errors]

            if not failed_indices:
                logger.info("All documents processed successfully, no retries needed")
                break

            logger.info(f"Retry attempt {attempt}/{max_retries}: {len(failed_indices)} failed documents")

            # Brief delay before retry (from config)
            await asyncio.sleep(self.config.document_retry_delay)

            # Collect failed documents
            retry_docs = [documents[i] for i in failed_indices]

            # Re-chunk failed documents
            retry_states = await self._chunk_all_documents(
                retry_docs, get_text_fn, get_id_fn, None
            )

            # Build trees for retry batch
            await self._build_trees_cascading(retry_states, rubric, None)

            # Convert to results
            retry_results = self._create_results(retry_states, rubric)

            # Replace successful retries in original results
            successes = 0
            for orig_idx, retry_result in zip(failed_indices, retry_results):
                if not retry_result.errors:
                    results[orig_idx] = retry_result
                    successes += 1

            logger.info(f"  Retry attempt {attempt}: {successes}/{len(failed_indices)} recovered")

            if progress_callback:
                progress_callback(f"retry_{attempt}", successes, len(failed_indices))

        return results

    def _log_failures(
        self,
        results: List[BuildResult],
        documents: List[Any],
        get_id_fn: Callable[[Any], str],
    ) -> None:
        """
        Log summary of failed documents.

        Args:
            results: Processing results
            documents: Original documents
            get_id_fn: Function to extract ID from document
        """
        failed = [(i, r) for i, r in enumerate(results) if r.errors]

        if not failed:
            return

        logger.warning(f"\n{'='*50}")
        logger.warning(f"FAILED DOCUMENTS: {len(failed)}/{len(results)}")

        for idx, result in failed:
            doc_id = get_id_fn(documents[idx])
            # Get text length safely
            try:
                text_len = len(str(documents[idx]))
            except Exception:
                text_len = -1

            error = result.errors[0] if result.errors else "Unknown"
            logger.warning(f"  [{idx}] {doc_id}: {error} (len={text_len})")

        logger.warning(f"{'='*50}\n")

    def get_stats(self) -> dict:
        """Get orchestration statistics."""
        return dict(self._build_stats)

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._build_stats = {
            'documents_processed': 0,
            'total_chunks': 0,
            'total_merges': 0,
            'total_levels': 0,
            'leaf_failures': 0,
            'merge_failures': 0,
            'documents_with_failures': 0,
        }

    def reset(self) -> None:
        """Reset all state for reuse."""
        self.reset_stats()
        # Reset tournament preferences if strategy supports it
        if hasattr(self.strategy, 'reset_preferences'):
            self.strategy.reset_preferences()


# =============================================================================
# Convenience Functions
# =============================================================================

async def batch_build_trees(
    documents: List[Any],
    strategy: SummarizationStrategy,
    rubric: str,
    get_text_fn: Optional[Callable[[Any], str]] = None,
    get_id_fn: Optional[Callable[[Any], str]] = None,
    max_chunk_chars: int = 2000,
) -> List[BuildResult]:
    """
    Build trees for multiple documents with optimal batching.

    Convenience function that creates an orchestrator and processes documents.

    Args:
        documents: List of documents
        strategy: SummarizationStrategy to use
        rubric: Information preservation criteria
        get_text_fn: Function to extract text from document
        get_id_fn: Function to extract ID from document
        max_chunk_chars: Maximum chunk size

    Returns:
        List of BuildResult, one per document
    """
    config = BuildConfig(max_chunk_chars=max_chunk_chars)
    orchestrator = BatchTreeOrchestrator(strategy=strategy, config=config)
    return await orchestrator.process_documents(
        documents=documents,
        rubric=rubric,
        get_text_fn=get_text_fn,
        get_id_fn=get_id_fn,
    )
