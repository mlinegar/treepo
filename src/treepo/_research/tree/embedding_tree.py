"""
Embedding tree construction for CTreePO.

Bridges the existing windowing / embedding infrastructure with the tree
structure needed by CTreePOModel. Given document text and an embedding
client, this module:

1. Splits text into windows (uniform axis windows).
2. Embeds each window via VLLMEmbeddingClient (cached in ConditionalMemory).
3. Optionally merges adjacent low-drift windows.
4. Builds a binary merge tree bottom-up.
5. Provides a forward_ctreepo() function that fills in sketch tensors.

The EmbeddingTreeNode mirrors learned_sketch.py's TreeNode but holds real
embedding vectors instead of synthetic indicator lists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:
    raise ImportError("PyTorch required for CTreePO. Install with: uv sync --extra torch")

from treepo._research.tree.ctreepo_model import CTreePOModel

logger = logging.getLogger(__name__)

# Optional imports for adaptive windowing (Phase 2 integration)
try:
    from treepo._research.preprocessing.adaptive_windows import (
        AxisWindow,
        adaptive_refine_windows,
        merge_adjacent_windows_by_embedding_drift,
    )
    _HAS_ADAPTIVE_WINDOWS = True
except ImportError:
    _HAS_ADAPTIVE_WINDOWS = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingTreeNode:
    """Unified node in a binary merge tree carrying both text and sketch paths.

    Every node has a shared structure (char offsets, level, children) and
    dual representations:
      - **Text path**: ``summary`` holds a human-readable summary (leaf:
        raw text, internal: LLM-merged summary).
      - **Sketch path**: ``embedding`` (leaf) → ``sketch`` (all nodes) via
        CTreePO LeafProjector + GatedMerge.

    Scores from multiple sources (oracle, sketch readout) live together
    on the same node so that feedback can flow between paths.
    """

    level: int                                      # 0 = leaf, increases upward
    text_span: str                                  # text covered by this node
    char_start: int                                 # start offset in original doc
    char_end: int                                   # end offset in original doc
    embedding: Optional[torch.Tensor | Sequence[float]] = None  # raw Qwen3 embedding (leaves only)
    sketch: Optional[torch.Tensor] = None           # set during forward pass
    children: Optional[Tuple[int, int]] = None      # indices of children (None for leaves)
    oracle_scores: Dict[str, float] = field(default_factory=dict)

    # --- Text-path fields (populated when LLM summarization runs) ---
    summary: str = ""                               # LLM summary (leaf: raw text; internal: merged)
    audit_result: Optional[Any] = None              # AuditResult from oracle scoring

    # --- Score fusion ---
    sketch_scores: Dict[str, float] = field(default_factory=dict)   # CTreePO readout predictions
    sketch_confidence: Optional[float] = None       # readout confidence (for audit targeting)

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    @property
    def text_len(self) -> int:
        return self.char_end - self.char_start


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def _uniform_windows(
    text_len: int,
    window_size: int,
    window_overlap: int = 0,
) -> List[Tuple[int, int]]:
    """Generate uniform (start, end) windows over text."""
    if text_len <= 0 or window_size <= 0:
        return [(0, max(text_len, 0))]
    if text_len <= window_size:
        return [(0, text_len)]

    step = max(1, window_size - window_overlap)
    windows: List[Tuple[int, int]] = []
    start = 0
    while start < text_len:
        end = min(start + window_size, text_len)
        windows.append((start, end))
        if end >= text_len:
            break
        start += step
    return windows


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------


def build_embedding_tree(
    text: str,
    embeddings: Sequence[Sequence[float] | torch.Tensor],
    windows: List[Tuple[int, int]],
) -> List[EmbeddingTreeNode]:
    """Build a binary merge tree from pre-computed window embeddings.

    Args:
        text: Full document text.
        embeddings: One embedding vector per window (from VLLMEmbeddingClient).
        windows: List of (char_start, char_end) tuples matching embeddings.

    Returns:
        Flat list of EmbeddingTreeNodes, bottom-up (leaves first, root last).
    """
    if len(embeddings) != len(windows):
        raise ValueError(
            f"Got {len(embeddings)} embeddings for {len(windows)} windows"
        )

    # Create leaf nodes
    from treepo._research.tree.packed_execution import canonicalize_leaf_embedding

    leaves: List[EmbeddingTreeNode] = []
    for (start, end), emb in zip(windows, embeddings):
        leaves.append(
            EmbeddingTreeNode(
                level=0,
                text_span=text[start:end],
                char_start=start,
                char_end=end,
                embedding=canonicalize_leaf_embedding(emb),
            )
        )

    nodes = list(leaves)

    # Build binary tree bottom-up (same pattern as learned_sketch.py)
    current_level_start = 0
    current_level_count = len(leaves)
    level = 1

    while current_level_count > 1:
        next_level_start = len(nodes)
        for i in range(0, current_level_count, 2):
            left_idx = current_level_start + i
            if i + 1 < current_level_count:
                right_idx = current_level_start + i + 1
                left_node = nodes[left_idx]
                right_node = nodes[right_idx]
                nodes.append(
                    EmbeddingTreeNode(
                        level=level,
                        text_span=text[left_node.char_start:right_node.char_end],
                        char_start=left_node.char_start,
                        char_end=right_node.char_end,
                        children=(left_idx, right_idx),
                    )
                )
            else:
                # Odd node: promote directly
                node = nodes[left_idx]
                nodes.append(
                    EmbeddingTreeNode(
                        level=level,
                        text_span=node.text_span,
                        char_start=node.char_start,
                        char_end=node.char_end,
                        children=(left_idx, left_idx),
                    )
                )
        current_level_start = next_level_start
        current_level_count = len(nodes) - next_level_start
        level += 1

    return nodes


def _get_model_device(model: CTreePOModel) -> torch.device:
    """Return the device of the first model parameter, or CPU."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def forward_ctreepo(
    model: CTreePOModel,
    nodes: List[EmbeddingTreeNode],
) -> None:
    """Run CTreePO forward pass, setting sketch on each node.

    Leaves are encoded in a single batched call.  Internal nodes are merged
    level-by-level, with all merges at a given level executed in one batched
    call.  This is semantically identical to the old per-node loop but much
    faster on GPU (and on CPU for large trees).
    """
    forward_ctreepo_batch(model, [nodes])


def forward_ctreepo_batch(
    model: CTreePOModel,
    tree_list: List[List[EmbeddingTreeNode]],
    max_batch_leaves: int = 8192,
) -> None:
    """Run CTreePO forward pass over multiple trees simultaneously.

    All leaf encodings across all trees are processed in a single batched
    call, and all merges at each tree level are batched across all trees.
    This is the primary entry point for efficient multi-document inference.

    Args:
        model: CTreePO model.
        tree_list: List of node lists (each from ``build_embedding_tree``).
        max_batch_leaves: Memory-safety cap on leaves per encoding batch.
            If total leaves exceed this, they are processed in sub-batches.
    """
    from treepo._research.tree.packed_execution import (
        build_packed_embedding_tree,
        build_packed_tree_batch,
        forward_packed_tree_batch,
    )

    if not tree_list:
        return

    packed_trees = [build_packed_embedding_tree(nodes) for nodes in tree_list]
    packed_batch = build_packed_tree_batch(
        packed_trees,
        device=_get_model_device(model),
    )
    forward_packed_tree_batch(
        model,
        packed_batch,
        max_batch_leaves=max_batch_leaves,
        materialize_nodes=True,
    )


# ---------------------------------------------------------------------------
# High-level helper (uses VLLMEmbeddingClient)
# ---------------------------------------------------------------------------


def build_tree_from_text(
    text: str,
    embedding_client: Any,
    window_size: int = 1200,
    window_overlap: int = 150,
    merge_drift_threshold: Optional[float] = None,
) -> List[EmbeddingTreeNode]:
    """Build an embedding tree from raw text.

    1. Create uniform windows.
    2. Embed each window via the client.
    3. Optionally merge adjacent low-drift windows.
    4. Build binary merge tree.

    Args:
        text: Document text.
        embedding_client: VLLMEmbeddingClient instance.
        window_size: Characters per window.
        window_overlap: Overlap between adjacent windows.
        merge_drift_threshold: If set, merge adjacent windows with cosine
            drift below this threshold (uses mean-pool for merged embedding).

    Returns:
        Flat list of EmbeddingTreeNodes (bottom-up).
    """
    windows = _uniform_windows(len(text), window_size, window_overlap)
    window_texts = [text[s:e] for s, e in windows]

    # Embed all windows (batched, cached in ConditionalMemory)
    embeddings = embedding_client.embed_texts(window_texts)

    # Optional: merge adjacent low-drift windows
    if merge_drift_threshold is not None and merge_drift_threshold > 0:
        windows, embeddings = _merge_low_drift_windows(
            windows, embeddings, merge_drift_threshold
        )

    return build_embedding_tree(text, embeddings, windows)


def _merge_low_drift_windows(
    windows: List[Tuple[int, int]],
    embeddings: List[List[float]],
    threshold: float,
) -> Tuple[List[Tuple[int, int]], List[List[float]]]:
    """Merge adjacent windows whose cosine distance is below threshold."""
    if len(windows) <= 1:
        return windows, embeddings

    mat = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    mat_normed = mat / norms

    merged_windows: List[Tuple[int, int]] = [windows[0]]
    merged_embs: List[np.ndarray] = [mat[0]]
    merge_counts: List[int] = [1]

    for i in range(1, len(windows)):
        prev_normed = mat_normed[i - 1]
        curr_normed = mat_normed[i]
        cosine_dist = 1.0 - float(np.dot(prev_normed, curr_normed))

        if cosine_dist <= threshold:
            # Merge with previous
            prev_start, _ = merged_windows[-1]
            _, curr_end = windows[i]
            merged_windows[-1] = (prev_start, curr_end)
            merged_embs[-1] = merged_embs[-1] + mat[i]
            merge_counts[-1] += 1
        else:
            merged_windows.append(windows[i])
            merged_embs.append(mat[i].copy())
            merge_counts.append(1)

    # Mean-pool merged embeddings
    final_embs = [
        (emb / count).tolist()
        for emb, count in zip(merged_embs, merge_counts)
    ]

    return merged_windows, final_embs


# ---------------------------------------------------------------------------
# Utility: collect all leaf/internal sketches
# ---------------------------------------------------------------------------


def collect_sketches(nodes: List[EmbeddingTreeNode]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Separate leaf sketches from internal node sketches.

    Returns:
        (leaf_sketches, internal_sketches)
    """
    leaf_sketches = []
    internal_sketches = []
    for node in nodes:
        if node.sketch is None:
            continue
        if node.is_leaf:
            leaf_sketches.append(node.sketch)
        else:
            internal_sketches.append(node.sketch)
    return leaf_sketches, internal_sketches


def get_root_sketch(nodes: List[EmbeddingTreeNode]) -> torch.Tensor:
    """Get the root node's sketch (last node in the list)."""
    if not nodes:
        raise ValueError("Empty node list")
    root = nodes[-1]
    if root.sketch is None:
        raise ValueError("Root sketch not set (call forward_ctreepo first)")
    return root.sketch


# ---------------------------------------------------------------------------
# Unified tree builder (shared windows for both text and sketch paths)
# ---------------------------------------------------------------------------


def _embedding_boundary_scores(
    embeddings: List[List[float]],
) -> List[float]:
    """Compute per-window boundary scores from embedding drift.

    Returns a score in [0, 1] for each window, where higher scores indicate
    windows near semantic boundaries (high embedding gradient).  Used as the
    ``score_windows`` callback for ``adaptive_refine_windows()``.

    The score for window *i* is the average cosine distance to its neighbours.
    Edge windows get the distance to their single neighbour.
    """
    n = len(embeddings)
    if n <= 1:
        return [0.5] * n

    mat = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = mat / norms

    scores: List[float] = []
    for i in range(n):
        dists: List[float] = []
        if i > 0:
            dists.append(1.0 - float(np.dot(normed[i], normed[i - 1])))
        if i < n - 1:
            dists.append(1.0 - float(np.dot(normed[i], normed[i + 1])))
        scores.append(float(np.mean(dists)) if dists else 0.5)

    # Normalise to [0, 1] for adaptive_refine_windows()
    s_min = min(scores)
    s_max = max(scores)
    span = s_max - s_min
    if span < 1e-12:
        return [0.5] * n
    return [(s - s_min) / span for s in scores]


def build_unified_tree(
    text: str,
    embedding_client: Any,
    *,
    coarse_window_size: int = 4000,
    fine_window_size: int = 1200,
    max_windows: int = 256,
    merge_drift_threshold: float = 0.03,
    adaptive: bool = True,
    feedback_signals: Optional[Sequence] = None,
    score_windows_callback: Optional[Callable] = None,
) -> List[EmbeddingTreeNode]:
    """Build a unified tree with adaptive, embedding-driven windows.

    This is the primary entry point for the unified architecture.  It replaces
    both ``build_tree_from_text()`` (fixed uniform windows) and the separate
    LLM chunking path by producing a single tree topology that both the sketch
    path (CTreePO) and the text path (LLM summarisation) can operate on.

    Pipeline:
        1. **Coarse embed**: Uniform windows at ``coarse_window_size`` → embed.
        2. **Score**: Compute embedding-drift boundary scores per window.
        3. **Refine** (if adaptive): ``adaptive_refine_windows()`` zooms into
           high-gradient / uncertain regions with ``fine_window_size`` windows.
        4. **Re-embed**: Embed the refined windows (only new ones need embedding;
           coarse windows that weren't refined keep their embeddings).
        5. **Drift merge**: Consolidate adjacent low-drift windows.
        6. **Build tree**: Binary merge tree from final windows.
        7. **Populate leaves**: Each leaf gets ``summary = raw text``
           ready for optional LLM summarisation.

    When ``adaptive=False``, falls back to uniform windows at
    ``fine_window_size`` (equivalent to the old ``build_tree_from_text``).

    Args:
        text: Full document text.
        embedding_client: VLLMEmbeddingClient (or any object with
            ``embed_texts(List[str]) -> List[List[float]]``).
        coarse_window_size: Characters per coarse window (Phase 2 adaptive).
        fine_window_size: Characters per fine window / fallback uniform size.
        max_windows: Cap on total windows after refinement.
        merge_drift_threshold: Cosine distance threshold for drift merging.
            Set to 0 to disable drift merging.
        adaptive: If True and adaptive_windows infrastructure is available,
            use coarse-to-fine refinement.  Otherwise uniform windows.
        feedback_signals: Optional ``ChunkFeedbackSignal`` list from a
            previous iteration's oracle audit (Phase 2 feedback loop).
        score_windows_callback: Optional callback that scores window
            embeddings for adaptive refinement.  Receives a list of embedding
            vectors and returns per-window scores in [0, 1].  When provided,
            these scores are **blended** with embedding-drift scores to guide
            adaptive refinement (e.g. MIL attention scores from Phase 5).

    Returns:
        Flat list of ``EmbeddingTreeNode`` (bottom-up, leaves first, root
        last).  Leaf nodes have ``embedding``, ``text_span``, and
        ``summary`` (= raw text) populated.
    """
    use_adaptive = adaptive and _HAS_ADAPTIVE_WINDOWS and len(text) > coarse_window_size

    if use_adaptive:
        nodes = _build_adaptive(
            text, embedding_client,
            coarse_window_size=coarse_window_size,
            fine_window_size=fine_window_size,
            max_windows=max_windows,
            merge_drift_threshold=merge_drift_threshold,
            feedback_signals=feedback_signals,
            score_windows_callback=score_windows_callback,
        )
    else:
        # Fall back to uniform windows (original build_tree_from_text path)
        nodes = build_tree_from_text(
            text, embedding_client,
            window_size=fine_window_size,
            window_overlap=int(fine_window_size * 0.125),
            merge_drift_threshold=merge_drift_threshold if merge_drift_threshold > 0 else None,
        )

    # Populate summary field on leaves (raw text, ready for LLM to override)
    for node in nodes:
        if node.is_leaf and not node.summary:
            node.summary = node.text_span

    return nodes


def _build_adaptive(
    text: str,
    embedding_client: Any,
    *,
    coarse_window_size: int,
    fine_window_size: int,
    max_windows: int,
    merge_drift_threshold: float,
    feedback_signals: Optional[Sequence],
    score_windows_callback: Optional[Callable] = None,
) -> List[EmbeddingTreeNode]:
    """Adaptive coarse-to-fine windowing backed by embedding drift scores.

    When *score_windows_callback* is provided (e.g. MIL attention scores),
    the final per-window score is a 50/50 blend of embedding-drift scores and
    the callback scores.  This lets learned importance guide refinement while
    still respecting semantic boundaries.
    """

    if not _HAS_ADAPTIVE_WINDOWS:
        raise RuntimeError("adaptive_windows module not available")

    # --- Step 1: Coarse embed ---
    coarse_wins = _uniform_windows(len(text), coarse_window_size, window_overlap=0)
    coarse_texts = [text[s:e] for s, e in coarse_wins]
    coarse_embs = embedding_client.embed_texts(coarse_texts)

    # --- Step 2: Score via embedding drift ---
    drift_scores = _embedding_boundary_scores(coarse_embs)

    # Optionally blend with external (e.g. MIL) scores
    if score_windows_callback is not None:
        mil_scores = score_windows_callback(coarse_embs)
        if len(mil_scores) == len(drift_scores):
            coarse_scores = [
                0.5 * d + 0.5 * m for d, m in zip(drift_scores, mil_scores)
            ]
        else:
            coarse_scores = drift_scores
    else:
        coarse_scores = drift_scores

    # Build AxisWindow list for adaptive_refine_windows
    axis_windows = [
        AxisWindow(start=s, end=e, unit="char")
        for s, e in coarse_wins
    ]

    # Build score lookup for the callback
    _score_cache: Dict[Tuple[int, int], float] = {
        (w.start, w.end): sc
        for w, sc in zip(axis_windows, coarse_scores)
    }

    def _score_callback(windows: Sequence[AxisWindow]) -> Sequence[float]:
        """Score callback: return cached coarse scores or 0.5 for new windows."""
        result = []
        for w in windows:
            key = (w.start, w.end)
            result.append(_score_cache.get(key, 0.5))
        return result

    # --- Step 3: Adaptive refinement ---
    # If we have feedback signals from a previous oracle audit, incorporate
    # them as extra refinement predicate (Phase 2 feedback loop)
    extra_predicate = None
    if feedback_signals:
        extra_predicate = _build_feedback_predicate(feedback_signals, len(text))

    refined_axis = adaptive_refine_windows(
        total_extent=len(text),
        coarse_window_size=coarse_window_size,
        fine_window_size=fine_window_size,
        max_windows=max_windows,
        score_windows=_score_callback,
        uncertainty_band=(0.35, 0.65),
        gradient_threshold=0.20,
        unit="char",
        extra_refine_predicate=extra_predicate,
    )

    # --- Step 4: Embed refined windows ---
    refined_tuples = [(w.start, w.end) for w in refined_axis]
    refined_texts = [text[s:e] for s, e in refined_tuples]
    refined_embs = embedding_client.embed_texts(refined_texts)

    # --- Step 5: Drift merge ---
    if merge_drift_threshold > 0 and len(refined_tuples) > 1:
        refined_tuples, refined_embs = _merge_low_drift_windows(
            refined_tuples, refined_embs, merge_drift_threshold
        )

    # --- Step 6 & 7: Build tree ---
    return build_embedding_tree(text, refined_embs, refined_tuples)


def _build_feedback_predicate(
    feedback_signals: Sequence,
    text_len: int,
) -> Any:
    """Build an extra_refine_predicate for adaptive_refine_windows from
    oracle feedback signals.

    Windows overlapping high-discrepancy feedback regions are refined
    regardless of their embedding score.
    """
    # Build a simple lookup: for each char offset, is there a high-discrepancy
    # feedback signal?
    HIGH_DISCREPANCY_THRESHOLD = 0.5

    refine_regions: List[Tuple[int, int]] = []
    for sig in feedback_signals:
        # ChunkFeedbackSignal has start_char, end_char, oracle_relevance_probability
        relevance = getattr(sig, "oracle_relevance_probability", None)
        low_info = getattr(sig, "low_info_probability", 0.0)
        # Refine if oracle marked this region as relevant but with high error,
        # or if it's a low-info region (may need different window size)
        if relevance is not None and relevance < HIGH_DISCREPANCY_THRESHOLD:
            refine_regions.append((
                getattr(sig, "start_char", 0),
                getattr(sig, "end_char", text_len),
            ))

    if not refine_regions:
        return None

    def _predicate(window: Any, score: float, gradient: float) -> bool:
        w_start = window.start
        w_end = window.end
        for rs, re in refine_regions:
            if w_start < re and w_end > rs:  # overlap
                return True
        return False

    return _predicate
