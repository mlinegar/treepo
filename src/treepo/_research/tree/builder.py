"""
OPS Tree Builder - Constructs summarization trees from document chunks.

The builder creates trees bottom-up, starting from leaf nodes (text chunks)
and recursively summarizing pairs of nodes until a single root remains.

This module provides a unified TreeBuilder that works with any SummarizationStrategy:
- DSPyStrategy for optimization/training
- BatchedStrategy for high-throughput inference
- TournamentStrategy for learning with preference collection

The builder is async-first with a sync wrapper for compatibility.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Dict, Tuple
from pathlib import Path
import logging
import asyncio
import os

from treepo._research.training.supervision import SupervisionDataset

from treepo._research.core.data_models import (
    Node, Tree, leaf, node
)
from treepo._research.preprocessing.chunker import (
    AdaptiveChunkingConfig,
    ChunkFeedbackSignal,
    DocumentChunker,
    TextChunk,
    chunk_for_ops as chunk,
)
from treepo._research.core.strategy import SummarizationStrategy, TournamentStrategy, tournament_doc_id
from treepo._research.core.protocols import format_merge_input, Summarizer
from treepo._research.core.async_utils import gather_with_cleanup
from treepo._research.core.unified_runtime import RUNTIME_MODE_UNIFIED_V2, resolve_runtime_mode
from treepo._research.tree.async_operator import AsyncFromSummarizationStrategy
from treepo._research.tree.state_tree import state_tree_to_text_tree
from treepo._research.tree.state_tree_runner import arun_fixed_binary_state_tree


logger = logging.getLogger(__name__)


# =============================================================================
# Chunking Helpers
# =============================================================================

def chunk_binary(text: str, max_chars: int = 8000) -> List[str]:
    """
    Split text into exactly 2 chunks for mini-tree construction.

    Splits at the midpoint, preferring sentence boundaries.

    Args:
        text: Text to split
        max_chars: Maximum characters per chunk (hint only; no truncation)

    Returns:
        List of exactly 2 chunks
    """
    if not text or not text.strip():
        return ["", ""]

    text = text.strip()

    # Find midpoint
    midpoint = len(text) // 2

    # Look for sentence boundary near midpoint (within 20% of doc length)
    search_range = len(text) // 5
    best_split = midpoint

    # Search for sentence endings near midpoint
    for offset in range(0, search_range):
        # Check forward
        pos = midpoint + offset
        if pos < len(text) and text[pos] in '.!?\n':
            best_split = pos + 1
            break
        # Check backward
        pos = midpoint - offset
        if pos > 0 and text[pos] in '.!?\n':
            best_split = pos + 1
            break

    left = text[:best_split].strip()
    right = text[best_split:].strip()

    # Ensure we have two non-empty chunks
    if not left:
        left = right[:len(right)//2]
        right = right[len(right)//2:]
    if not right:
        right = left[len(left)//2:]
        left = left[:len(left)//2]

    return [left, right]


# =============================================================================
# Test/Mock Summarizers (for testing without LLM)
# =============================================================================

class IdentitySummarizer:
    """
    Summarizer that returns content unchanged.
    Useful for testing tree structure without LLM calls.
    """

    def __call__(self, content: str, rubric: str) -> str:
        return content


class ConcatenatingSummarizer:
    """
    Summarizer that concatenates with a separator.
    Useful for testing to see the full tree content.
    """

    def __init__(self, prefix: str = "[Summary] "):
        self.prefix = prefix

    def __call__(self, content: str, rubric: str) -> str:
        # Add a prefix to show summarization happened
        return f"{self.prefix}{content}"


class TruncatingSummarizer:
    """
    Summarizer that truncates content to a max length.
    Useful for testing with predictable output sizes.
    """

    def __init__(self, max_length: int = 100):
        self.max_length = max_length

    def __call__(self, content: str, rubric: str) -> str:
        if len(content) <= self.max_length:
            return content
        return content[:self.max_length - 3] + "..."


# =============================================================================
# Configuration and Results
# =============================================================================

@dataclass
class BuildConfig:
    """Configuration for tree building."""

    # Chunking settings
    max_chunk_chars: int = 2000
    max_chunk_tokens: Optional[int] = None
    chunk_token_encoding: str = "cl100k_base"
    chunk_overlap_tokens: int = 0
    min_chunk_chars: int = 100
    chunk_strategy: str = "axis"  # "axis" (default), "sentence", or "paragraph"
    adaptive_chunking: Optional[AdaptiveChunkingConfig] = None
    chunk_feedback_signals: Optional[List[ChunkFeedbackSignal]] = None

    # Tree settings
    merge_strategy: str = "binary"  # "binary" for 2-way merge

    # Tournament settings (used by TournamentStrategy)
    k: int = 4  # Number of candidates for tournament selection

    # Concurrency and cleanup
    max_concurrent_requests: int = 200
    task_cancel_timeout: float = 30.0
    document_retry_delay: float = 1.0
    runtime_mode: str = "legacy"
    batch_plan_cache_name: str = "tree_builder"

    # Degenerate-summary safeguards (manual validation / fail-fast mode)
    fail_on_degenerate_summary: bool = False
    max_degenerate_leaf_fallbacks: int = 0
    max_degenerate_merge_fallbacks: int = 0

    # Debug settings
    verbose: bool = False


@dataclass
class BuildResult:
    """Result of tree building operation."""
    tree: Tree
    chunks_created: int
    nodes_created: int
    levels_created: int
    errors: List[str] = field(default_factory=list)
    supervision: SupervisionDataset = field(default_factory=SupervisionDataset)
    content_weights: Optional[Dict[str, float]] = None  # per-leaf info scores for audit sampling


# =============================================================================
# Unified Tree Builder (async-first)
# =============================================================================

class TreeBuilder:
    """
    Unified async-first tree builder using SummarizationStrategy.

    This builder works with any strategy:
    - DSPyStrategy: For DSPy-based optimization/training
    - BatchedStrategy: For high-throughput batched inference
    - TournamentStrategy: Wraps any strategy with tournament selection

    The builder is async-first with a sync wrapper for compatibility.

    Example:
        # Async usage with batched strategy
        strategy = BatchedStrategy(client)
        builder = TreeBuilder(strategy)
        result = await builder.build(text, rubric)

        # Sync wrapper
        result = builder.build_sync(text, rubric)

        # With tournament selection (for learning)
        tournament = TournamentStrategy(base=strategy, judge=judge)
        builder = TreeBuilder(tournament)
        result = await builder.build(text, rubric)
        preferences = tournament.get_preferences()  # Free byproduct!
    """

    def __init__(
        self,
        strategy: SummarizationStrategy,
        config: Optional[BuildConfig] = None
    ):
        """
        Initialize the unified builder.

        Args:
            strategy: SummarizationStrategy for summarize/merge operations
            config: Build configuration
        """
        self.strategy = strategy
        self.config = config or BuildConfig()
        self._build_stats = {
            'summarizer_calls': 0,
            'total_input_chars': 0,
            'total_output_chars': 0
        }

    async def build(self, text: str, rubric: str = "") -> BuildResult:
        """
        Build a tree from raw text asynchronously.

        Args:
            text: Document text to process
            rubric: Information preservation criteria

        Returns:
            BuildResult containing the tree and statistics
        """
        if not text or not text.strip():
            raise ValueError("Cannot build tree from empty text")

        # Chunk the text
        chunks = chunk(
            text,
            max_chars=self.config.max_chunk_chars,
            max_tokens=self.config.max_chunk_tokens,
            token_encoding=self.config.chunk_token_encoding,
            overlap_tokens=self.config.chunk_overlap_tokens,
            strategy=self.config.chunk_strategy,
            adaptive_config=self.config.adaptive_chunking,
            feedback_signals=self.config.chunk_feedback_signals,
        )

        if not chunks:
            raise ValueError("Chunking produced no chunks")

        # Extract per-leaf info scores for auditor CONTENT_WEIGHTED sampling.
        from treepo._research.preprocessing.visual_feedback import extract_content_weights_from_chunks

        content_weights = extract_content_weights_from_chunks(chunks)

        result = await self.build_from_chunks(chunks, rubric)
        result.content_weights = content_weights
        return result

    async def build_from_chunks(
        self,
        chunks: List[TextChunk],
        rubric: str = ""
    ) -> BuildResult:
        """
        Build a tree from pre-chunked text asynchronously.

        Args:
            chunks: List of TextChunk objects
            rubric: Information preservation criteria

        Returns:
            BuildResult containing the tree and statistics
        """
        if not chunks:
            raise ValueError("Cannot build tree from empty chunks list")

        errors: List[str] = []

        # Canonical fixed-binary execution now lives in the `StateTree` runner.
        # We run a text-only operator lifted from the provided strategy, then
        # convert to the legacy `Tree` data model for downstream consumers.
        leaf_spans = [str(chunk_obj.text or "") for chunk_obj in chunks]
        operator = AsyncFromSummarizationStrategy(
            strategy=self.strategy,
            name=str(getattr(self.strategy, "name", "summarization_strategy")),
            max_concurrent=int(self.config.max_concurrent_requests),
        )
        state_result = await arun_fixed_binary_state_tree(
            operator,
            leaf_spans,
            rubric=str(rubric or ""),
            refine_rounds=0,
            max_concurrent=int(self.config.max_concurrent_requests),
        )
        tree = state_tree_to_text_tree(
            state_result.tree,
            rubric=str(rubric or ""),
            metadata={"state_tree_metadata": dict(state_result.tree.metadata or {})},
        )

        # Best-effort stats (kept for compatibility with existing diagnostics).
        try:
            all_nodes = list(tree.traverse_preorder())
            leaves = [n for n in all_nodes if n.is_leaf]
            internal = [n for n in all_nodes if not n.is_leaf]
            self._build_stats["summarizer_calls"] += len(all_nodes)
            self._build_stats["total_input_chars"] += sum(len(n.raw_text_span or "") for n in leaves)
            self._build_stats["total_input_chars"] += sum(
                len((n.left_child.summary if n.left_child else "")) + len((n.right_child.summary if n.right_child else ""))
                for n in internal
            )
            self._build_stats["total_output_chars"] += sum(len(n.summary or "") for n in all_nodes)
        except Exception:
            pass

        # Preserve explicit chunk-to-leaf lineage for downstream audit/training.
        tree.metadata.setdefault(
            "chunk_boundaries",
            [
                {
                    "chunk_index": chunk_obj.chunk_index,
                    "start_char": chunk_obj.start_char,
                    "end_char": chunk_obj.end_char,
                    "char_count": chunk_obj.char_count,
                    "token_count": chunk_obj.token_count,
                    "metadata": dict(chunk_obj.metadata or {}),
                }
                for chunk_obj in chunks
            ],
        )
        tree.metadata.setdefault(
            "chunking",
            {
                "strategy": self.config.chunk_strategy,
                "max_chunk_chars": self.config.max_chunk_chars,
                "adaptive_enabled": bool(
                    self.config.adaptive_chunking and self.config.adaptive_chunking.enabled
                ),
            },
        )
        # Attach support spans to nodes for unified downstream artifacts.
        try:
            from treepo._research.tree.unified_artifacts import attach_chunk_support

            attach_chunk_support(tree, overwrite=False)
        except Exception:
            logger.debug("Failed to attach chunk support spans to tree nodes", exc_info=True)

        # Collect supervision if strategy supports it (e.g., TournamentStrategy)
        binary_projection = []
        if hasattr(self.strategy, 'get_preferences'):
            binary_projection = self.strategy.get_preferences()
        comparative_judgments = []
        if hasattr(self.strategy, 'get_comparative_judgments'):
            comparative_judgments = self.strategy.get_comparative_judgments()

        current_doc_id = str(tournament_doc_id.get() or "").strip()
        if current_doc_id:
            binary_projection = [
                pair
                for pair in binary_projection
                if str(getattr(pair, "source_example_id", "")).startswith(f"{current_doc_id}:")
            ]
            comparative_judgments = [
                record
                for record in comparative_judgments
                if str(getattr(record, "source_example_id", "")).startswith(f"{current_doc_id}:")
            ]

        return BuildResult(
            tree=tree,
            chunks_created=len(chunks),
            nodes_created=tree.node_count,
            levels_created=tree.height + 1,
            errors=errors,
            supervision=SupervisionDataset(
                comparative_judgments=(
                    list(comparative_judgments)
                    if comparative_judgments
                    else [pair.to_comparative_judgment() for pair in binary_projection]
                )
            ),
        )

    async def build_from_file(self, filepath: Path, rubric: str = "") -> BuildResult:
        """
        Build a tree from a file asynchronously.

        Args:
            filepath: Path to text file
            rubric: Information preservation criteria

        Returns:
            BuildResult containing the tree and statistics
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        text = filepath.read_text(encoding='utf-8')
        result = await self.build(text, rubric)

        # Add file metadata
        result.tree.metadata['source_file'] = str(filepath)

        return result

    def build_sync(self, text: str, rubric: str = "") -> BuildResult:
        """
        Synchronous wrapper for build().

        Args:
            text: Document text to process
            rubric: Information preservation criteria

        Returns:
            BuildResult containing the tree and statistics
        """
        return asyncio.run(self.build(text, rubric))

    async def _build_leaves(
        self,
        chunks: List[TextChunk],
        rubric: str,
        errors: List[str],
    ) -> List[Node]:
        """Build leaf nodes with summaries in parallel."""
        runtime_mode = resolve_runtime_mode(getattr(self.config, "runtime_mode", None))
        bulk_summarize = getattr(self.strategy, "summarize_many", None)

        if runtime_mode == RUNTIME_MODE_UNIFIED_V2 and callable(bulk_summarize):
            doc_id = tournament_doc_id.get()
            try:
                summaries = list(
                    await bulk_summarize(
                        [
                            {
                                "content": chunk_obj.text,
                                "rubric": rubric,
                                "temperature": 0.7,
                                "doc_id": doc_id,
                            }
                            for chunk_obj in chunks
                        ]
                    )
                )
            except Exception as exc:
                errors.append(f"Bulk leaf summarization failed: {exc}")
                summaries = []

            if summaries:
                if len(summaries) < len(chunks):
                    summaries.extend([""] * (len(chunks) - len(summaries)))
                leaves: List[Node] = []
                for idx, (chunk_obj, summary) in enumerate(zip(chunks, summaries)):
                    rendered = str(summary or "")
                    self._build_stats['summarizer_calls'] += 1
                    self._build_stats['total_input_chars'] += len(chunk_obj.text)
                    self._build_stats['total_output_chars'] += len(rendered)
                    if rendered:
                        leaves.append(leaf(chunk_obj.text, summary=rendered, node_id=f"leaf_{idx}"))
                    else:
                        leaves.append(leaf(chunk_obj.text, node_id=f"leaf_{idx}"))
                return leaves

        async def summarize_leaf(idx: int, text: str) -> tuple[int, Node]:
            try:
                summary = await self.strategy.summarize(text, rubric)
                self._build_stats['summarizer_calls'] += 1
                self._build_stats['total_input_chars'] += len(text)
                self._build_stats['total_output_chars'] += len(summary)
                return idx, leaf(text, summary=summary, node_id=f"leaf_{idx}")
            except Exception as e:
                errors.append(f"Failed to summarize leaf {idx}: {e}")
                # Return leaf without summary as fallback
                return idx, leaf(text, node_id=f"leaf_{idx}")

        # Create all leaf tasks
        leaf_tasks = [
            summarize_leaf(i, chunk_obj.text)
            for i, chunk_obj in enumerate(chunks)
        ]

        # Await all leaf summarizations in parallel (with cleanup on cancellation)
        results = await gather_with_cleanup(leaf_tasks, return_exceptions=True)

        # Sort by index and extract nodes
        valid_results = []
        for item in results:
            if isinstance(item, Exception):
                errors.append(f"Leaf task failed: {item}")
                continue
            if isinstance(item, tuple) and len(item) == 2:
                valid_results.append(item)

        valid_results.sort(key=lambda x: x[0])
        return [n for _, n in valid_results]

    async def _merge_nodes(
        self,
        left: Node,
        right: Node,
        rubric: str,
        level: int
    ) -> Node:
        """Merge two nodes into a parent node asynchronously."""
        self._build_stats['summarizer_calls'] += 1
        self._build_stats['total_input_chars'] += len(left.summary) + len(right.summary)

        node_id = f"L{level}_{self._build_stats['summarizer_calls']}"
        summary = await self.strategy.merge(left.summary, right.summary, rubric)

        self._build_stats['total_output_chars'] += len(summary)

        return node(
            left=left,
            right=right,
            summary=summary,
            node_id=node_id
        )

    async def _build_tree_pipelined(
        self,
        leaves: List[Node],
        rubric: str,
        errors: List[str],
    ) -> Tree:
        """
        Build tree with pipelined execution - submit merges as soon as children ready.

        Submits merges as soon as their dependencies are satisfied, reducing
        latency by allowing work to overlap across tree levels.

        Pattern: Uses asyncio.wait(FIRST_COMPLETED) to process completions
        and submit newly-ready work immediately.
        """
        if len(leaves) == 1:
            return Tree(root=leaves[0], rubric=rubric)

        # Build merge dependency graph upfront
        @dataclass
        class MergeTask:
            id: int
            level: int
            left_idx: int   # Index into leaves (level 0) or merge ID (higher levels)
            right_idx: int
            left_is_merge: bool = False  # True if left_idx refers to a merge result
            right_is_merge: bool = False

        # Pre-compute all merges needed
        merges: Dict[int, MergeTask] = {}
        merge_id = 0

        # Level 0: pair up leaves
        current_refs: List[tuple[int, bool]] = [(i, False) for i in range(len(leaves))]  # (idx, is_merge)
        level = 0

        while len(current_refs) > 1:
            level += 1
            next_refs = []

            for i in range(0, len(current_refs), 2):
                if i + 1 < len(current_refs):
                    left_idx, left_is_merge = current_refs[i]
                    right_idx, right_is_merge = current_refs[i + 1]

                    merges[merge_id] = MergeTask(
                        id=merge_id,
                        level=level,
                        left_idx=left_idx,
                        right_idx=right_idx,
                        left_is_merge=left_is_merge,
                        right_is_merge=right_is_merge,
                    )
                    next_refs.append((merge_id, True))
                    merge_id += 1
                else:
                    # Odd node carries forward
                    next_refs.append(current_refs[i])

            current_refs = next_refs

        # Execute with pipelining
        completed: Dict[int, Node] = {}  # merge_id -> resulting Node
        pending: Dict[int, asyncio.Task] = {}  # merge_id -> task

        async def execute_merge(m: MergeTask) -> tuple[int, Node]:
            """Execute a single merge and return (merge_id, result_node)."""
            # Get left node
            if m.left_is_merge:
                left_node = completed[m.left_idx]
            else:
                left_node = leaves[m.left_idx]

            # Get right node
            if m.right_is_merge:
                right_node = completed[m.right_idx]
            else:
                right_node = leaves[m.right_idx]

            try:
                merged = await self._merge_nodes(left_node, right_node, rubric, m.level)
                return m.id, merged
            except Exception as e:
                error_msg = f"Pipelined merge {m.id} failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                raise

        def is_ready(m: MergeTask) -> bool:
            """Check if a merge's dependencies are satisfied."""
            if m.left_is_merge and m.left_idx not in completed:
                return False
            if m.right_is_merge and m.right_idx not in completed:
                return False
            return True

        runtime_mode = resolve_runtime_mode(getattr(self.config, "runtime_mode", None))
        bulk_merge = getattr(self.strategy, "merge_many", None)
        if runtime_mode == RUNTIME_MODE_UNIFIED_V2 and callable(bulk_merge):
            while len(completed) < len(merges):
                ready = [
                    m
                    for m in merges.values()
                    if is_ready(m) and m.id not in completed
                ]
                if not ready:
                    break

                payloads = []
                contexts: List[Tuple[MergeTask, Node, Node]] = []
                for m in ready:
                    left_node = completed[m.left_idx] if m.left_is_merge else leaves[m.left_idx]
                    right_node = completed[m.right_idx] if m.right_is_merge else leaves[m.right_idx]
                    contexts.append((m, left_node, right_node))
                    payloads.append(
                        {
                            "left": left_node.summary,
                            "right": right_node.summary,
                            "rubric": rubric,
                            "temperature": 0.7,
                            "doc_id": tournament_doc_id.get(),
                        }
                    )

                try:
                    merged_summaries = list(await bulk_merge(payloads))
                except Exception as exc:
                    errors.append(f"Bulk merge execution failed: {exc}")
                    merged_summaries = []

                if len(merged_summaries) < len(contexts):
                    merged_summaries.extend([""] * (len(contexts) - len(merged_summaries)))

                for (m, left_node, right_node), merged_summary in zip(contexts, merged_summaries):
                    rendered = str(merged_summary or "")
                    self._build_stats['summarizer_calls'] += 1
                    self._build_stats['total_input_chars'] += len(left_node.summary) + len(right_node.summary)
                    self._build_stats['total_output_chars'] += len(rendered)
                    completed[m.id] = node(
                        left=left_node,
                        right=right_node,
                        summary=rendered,
                        node_id=f"L{m.level}_{m.id}",
                    )

            if completed:
                final_id = max(completed.keys())
                return Tree(root=completed[final_id], rubric=rubric)
            return Tree(root=leaves[0], rubric=rubric)

        # Submit all initially ready merges (level 1 - leaf pairs)
        for m in merges.values():
            if is_ready(m) and m.id not in pending:
                pending[m.id] = asyncio.create_task(execute_merge(m))

        if self.config.verbose:
            logger.info(f"Pipelined build: {len(merges)} total merges, "
                       f"{len(pending)} initially ready")

        try:
            while pending:
                # Wait for any task to complete
                done, _ = await asyncio.wait(
                    pending.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    merge_id_result, result_node = await task
                    completed[merge_id_result] = result_node
                    del pending[merge_id_result]

                    if self.config.verbose:
                        logger.debug(f"Merge {merge_id_result} complete, "
                                    f"{len(completed)}/{len(merges)} done")

                    # Check for newly ready merges
                    for m in merges.values():
                        if is_ready(m) and m.id not in pending and m.id not in completed:
                            pending[m.id] = asyncio.create_task(execute_merge(m))

        finally:
            # Cancel remaining tasks on exception
            if pending:
                logger.debug(f"Cleaning up {len(pending)} pending pipelined merge tasks...")
                for task in pending.values():
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*pending.values(), return_exceptions=True)

        # Get final result - the last merge completed
        if completed:
            final_id = max(completed.keys())
            root = completed[final_id]
        else:
            # Shouldn't happen, but handle gracefully
            root = leaves[0]

        return Tree(root=root, rubric=rubric)

    def get_stats(self) -> dict:
        """Get build statistics."""
        return dict(self._build_stats)

    def reset_stats(self) -> None:
        """Reset build statistics."""
        self._build_stats = {
            'summarizer_calls': 0,
            'total_input_chars': 0,
            'total_output_chars': 0
        }

    def reset(self) -> None:
        """Reset all state for reuse."""
        self.reset_stats()
        # Reset tournament preferences if strategy supports it
        if hasattr(self.strategy, 'reset_preferences'):
            self.strategy.reset_preferences()

    # -----------------------------------------------------------------
    # Unified tree integration: LLM summarisation on shared topology
    # -----------------------------------------------------------------

    async def summarize_unified_nodes(
        self,
        nodes: List[Any],
        rubric: str = "",
    ) -> None:
        """Run LLM summarisation on a pre-built unified tree (in-place).

        This takes a list of ``EmbeddingTreeNode`` objects (from
        ``build_unified_tree()``) that already have shared topology and
        embeddings, and fills in the ``summary`` field using the builder's
        ``SummarizationStrategy``.

        - **Leaves**: summarised via ``strategy.summarize(text_span, rubric)``
        - **Internal nodes**: merged via ``strategy.merge(left.summary,
          right.summary, rubric)`` following the existing binary merge order.

        The tree topology (``children`` tuples, ``level``, ``char_start/end``)
        is **not modified** — only ``summary`` fields are written.

        Args:
            nodes: Flat list of EmbeddingTreeNode (bottom-up order, as returned
                by ``build_unified_tree()``).
            rubric: Information-preservation rubric for the LLM.
        """
        from treepo._research.core.async_utils import gather_with_cleanup
        runtime_mode = resolve_runtime_mode(getattr(self.config, "runtime_mode", None))

        # --- Phase 1: Summarise all leaves in parallel ---
        leaf_indices = [i for i, n in enumerate(nodes) if n.is_leaf]

        bulk_summarize = getattr(self.strategy, "summarize_many", None)
        if runtime_mode == RUNTIME_MODE_UNIFIED_V2 and callable(bulk_summarize) and leaf_indices:
            try:
                leaf_summaries = list(
                    await bulk_summarize(
                        [
                            {
                                "content": nodes[idx].text_span if nodes[idx].text_span else "",
                                "rubric": rubric,
                                "temperature": 0.7,
                                "doc_id": tournament_doc_id.get(),
                            }
                            for idx in leaf_indices
                        ]
                    )
                )
            except Exception as exc:
                logger.warning("Unified bulk leaf summarization failed: %s", exc)
                leaf_summaries = []

            if len(leaf_summaries) < len(leaf_indices):
                leaf_summaries.extend([""] * (len(leaf_indices) - len(leaf_summaries)))
            for idx, summary in zip(leaf_indices, leaf_summaries):
                nodes[idx].summary = str(summary or "")
                self._build_stats['summarizer_calls'] += 1
        else:
            async def _summarize_leaf(idx: int) -> Tuple[int, str]:
                node = nodes[idx]
                text = node.text_span if node.text_span else ""
                try:
                    summary = await self.strategy.summarize(text, rubric)
                    self._build_stats['summarizer_calls'] += 1
                    return idx, summary
                except Exception as e:
                    logger.warning("Leaf summarization failed for node %d: %s", idx, e)
                    return idx, text  # fall back to raw text

            leaf_tasks = [_summarize_leaf(i) for i in leaf_indices]
            leaf_results = await gather_with_cleanup(leaf_tasks, return_exceptions=True)

            for item in leaf_results:
                if isinstance(item, Exception):
                    logger.warning("Leaf summarization task failed: %s", item)
                    continue
                idx, summary = item
                nodes[idx].summary = summary

        # --- Phase 2: Merge internal nodes level-by-level ---
        pending_internal = {
            idx
            for idx, tree_node in enumerate(nodes)
            if (not tree_node.is_leaf) and tree_node.children is not None
        }
        bulk_merge = getattr(self.strategy, "merge_many", None)
        if runtime_mode == RUNTIME_MODE_UNIFIED_V2 and callable(bulk_merge):
            while pending_internal:
                ready_indices: List[int] = []
                payloads: List[Dict[str, Any]] = []
                for idx in list(pending_internal):
                    tree_node = nodes[idx]
                    left_idx, right_idx = tree_node.children
                    left_summary = nodes[left_idx].summary or nodes[left_idx].text_span
                    right_summary = nodes[right_idx].summary or nodes[right_idx].text_span
                    if not left_summary or not right_summary:
                        continue
                    ready_indices.append(idx)
                    payloads.append(
                        {
                            "left": left_summary,
                            "right": right_summary,
                            "rubric": rubric,
                            "temperature": 0.7,
                            "doc_id": tournament_doc_id.get(),
                        }
                    )

                if not ready_indices:
                    break

                try:
                    merged_summaries = list(await bulk_merge(payloads))
                except Exception as exc:
                    logger.warning("Unified bulk merge failed: %s", exc)
                    merged_summaries = []

                if len(merged_summaries) < len(ready_indices):
                    merged_summaries.extend([""] * (len(ready_indices) - len(merged_summaries)))

                for idx, merged_summary in zip(ready_indices, merged_summaries):
                    tree_node = nodes[idx]
                    left_idx, right_idx = tree_node.children
                    left_summary = nodes[left_idx].summary or nodes[left_idx].text_span or ""
                    right_summary = nodes[right_idx].summary or nodes[right_idx].text_span or ""
                    if left_idx == right_idx:
                        tree_node.summary = left_summary
                    else:
                        rendered = str(merged_summary or "")
                        if not rendered:
                            rendered = left_summary[:500] + "\n---\n" + right_summary[:500]
                        tree_node.summary = rendered
                        self._build_stats['summarizer_calls'] += 1
                    pending_internal.discard(idx)
        else:
            # Nodes are bottom-up, so processing in order guarantees children
            # are summarised before parents.
            for i, tree_node in enumerate(nodes):
                if tree_node.is_leaf:
                    continue
                if tree_node.children is None:
                    continue

                left_idx, right_idx = tree_node.children
                left_summary = nodes[left_idx].summary or nodes[left_idx].text_span
                right_summary = nodes[right_idx].summary or nodes[right_idx].text_span

                if left_idx == right_idx:
                    # Promoted odd node — just copy child summary
                    tree_node.summary = left_summary
                    continue

                try:
                    merged_summary = await self.strategy.merge(
                        left_summary, right_summary, rubric
                    )
                    self._build_stats['summarizer_calls'] += 1
                    tree_node.summary = merged_summary
                except Exception as e:
                    logger.warning("Merge failed for node %d: %s", i, e)
                    # Fallback: concatenate truncated children
                    tree_node.summary = left_summary[:500] + "\n---\n" + right_summary[:500]


# =============================================================================
# Helper Functions
# =============================================================================

async def async_build(
    text: str,
    rubric: str,
    strategy: SummarizationStrategy,
    max_chars: int = 2000,
) -> Tree:
    """
    Build an OPS tree asynchronously using a strategy.

    Args:
        text: Document text
        rubric: Information preservation criteria
        strategy: SummarizationStrategy to use
        max_chars: Maximum chunk size

    Returns:
        Tree
    """
    config = BuildConfig(max_chunk_chars=max_chars)
    builder = TreeBuilder(strategy=strategy, config=config)
    result = await builder.build(text, rubric)
    return result.tree


def build(
    text: str,
    rubric: str = "",
    summarizer: Optional[Summarizer] = None,
    max_chars: int = 2000
) -> Tree:
    """
    Build an OPS tree from text synchronously.

    For new code, prefer using TreeBuilder with build_sync().

    Args:
        text: Document text
        rubric: Information preservation criteria
        summarizer: Summarization function (defaults to identity)
        max_chars: Maximum chunk size

    Returns:
        Tree
    """
    if summarizer is None:
        summarizer = IdentitySummarizer()

    # Create a simple async wrapper for the sync summarizer
    class SyncSummarizerAdapter:
        def __init__(self, sync_fn: Summarizer):
            self._fn = sync_fn

        async def summarize(self, content: str, rubric: str) -> str:
            return self._fn(content, rubric)

        async def merge(self, left: str, right: str, rubric: str) -> str:
            combined = format_merge_input(left, right)
            return self._fn(combined, rubric)

    adapter = SyncSummarizerAdapter(summarizer)
    config = BuildConfig(max_chunk_chars=max_chars)
    builder = TreeBuilder(strategy=adapter, config=config)

    return builder.build_sync(text, rubric).tree


def build_test_tree(num_leaves: int = 4) -> Tree:
    """
    Build a simple test tree with predictable structure.

    Args:
        num_leaves: Number of leaf nodes

    Returns:
        Tree with numbered nodes
    """
    # Create simple numbered content
    chunks = [
        TextChunk(
            text=f"Chunk {i} content.",
            start_char=i*20,
            end_char=(i+1)*20,
            chunk_index=i
        )
        for i in range(num_leaves)
    ]

    summarizer = ConcatenatingSummarizer()

    # Use the sync adapter pattern
    class TestAdapter:
        async def summarize(self, content: str, rubric: str) -> str:
            return summarizer(content, rubric)

        async def merge(self, left: str, right: str, rubric: str) -> str:
            combined = format_merge_input(left, right)
            return summarizer(combined, rubric)

    config = BuildConfig(verbose=False)
    builder = TreeBuilder(strategy=TestAdapter(), config=config)

    async def _build():
        return await builder.build_from_chunks(chunks, rubric="Test rubric")

    result = asyncio.run(_build())
    return result.tree

# Backwards compatibility alias
AsyncTreeBuilder = TreeBuilder
