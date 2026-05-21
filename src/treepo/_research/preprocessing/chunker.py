"""
Document chunking for OPS tree construction.

Thin wrapper around langextract's chunking with tiktoken token counting.
Supports context-window-aware chunking via ContextWindowManager.
"""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, TYPE_CHECKING

from langextract.chunking import ChunkIterator
from langextract.core.tokenizer import RegexTokenizer

if TYPE_CHECKING:
    from treepo._research.config.context_window import ContextWindowManager


@dataclass
class TextChunk:
    """
    A chunk of text from a document.

    Attributes:
        text: The chunk content
        start_char: Starting character position in original document
        end_char: Ending character position in original document
        chunk_index: Index of this chunk in the sequence
        token_count: Number of tokens (if computed)
        metadata: Additional information about the chunk
    """
    text: str
    start_char: int = 0
    end_char: int = 0
    chunk_index: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        """Number of characters in this chunk."""
        return len(self.text)

    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"TextChunk({self.chunk_index}, tokens={self.token_count}, chars={self.char_count})"


@dataclass
class ChunkFeedbackSignal:
    """
    Feedback signal attached to a character span for adaptive chunking.

    low_info_probability:
        0.0 = highly informative, 1.0 = likely low-information.
    noise_probability:
        0.0 = low-noise span, 1.0 = likely noisy proxy signal.
    confidence:
        Trust score of the feedback source in [0, 1].
    """
    start_char: int
    end_char: int
    low_info_probability: float
    noise_probability: float = 0.0
    confidence: float = 1.0
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    oracle_relevance_probability: Optional[float] = None

    @property
    def width(self) -> int:
        """Character width of this feedback span."""
        return max(0, self.end_char - self.start_char)


@dataclass
class AdaptiveChunkingConfig:
    """
    Controls adaptive chunk sizing based on low-information/noise proxies.

    The policy expands chunk targets in low-information/noisy regions and
    compresses chunk targets in likely high-information regions.
    """
    enabled: bool = False
    min_chars: int = 400
    max_chars: int = 8000
    low_info_expansion_weight: float = 0.8
    noise_expansion_weight: float = 0.3
    high_info_compression_weight: float = 0.5
    min_target_scale: float = 0.6
    max_target_scale: float = 2.0
    proxy_blend: float = 0.5
    crossfit_folds: int = 1
    proxy_model: Optional[str] = None
    proxy_score_key: Optional[str] = None
    proxy_fallback_to_baseline: bool = True
    # Adapter id used for embedding-window scoring (text path defaults to char).
    window_adapter: str = "text_char"
    # If true, collapse adjacent low-drift embedding windows before feedback.
    window_merge_enabled: bool = True
    # Merge threshold for adjacent-window cosine distance in [0, 2].
    window_merge_max_cosine_distance: float = 0.03
    # Optional hard cap for merged windows in axis units (chars for text).
    window_merge_max_extent: Optional[int] = None


@dataclass
class HonestChunkingPolicy:
    """
    Honest split policy for adaptive chunking feedback.

    When enabled, chunk-boundary adaptation should only consume signals from
    the boundary split; oracle-quality evaluation should use the held-out
    evaluation split.
    """
    enabled: bool = False
    boundary_fraction: float = 0.5
    split_seed: int = 17
    boundary_role: str = "boundary"
    evaluation_role: str = "evaluation"


class AdaptiveChunkMemory:
    """
    Lightweight in-memory store for per-document chunk feedback.

    This lets callers update future chunking runs from prior label/prediction
    outcomes without coupling chunking to any specific oracle implementation.
    """

    def __init__(self):
        self._signals_by_doc: Dict[str, List[ChunkFeedbackSignal]] = {}

    def get_signals(
        self,
        doc_id: str,
        *,
        honest_role: Optional[str] = None,
    ) -> List[ChunkFeedbackSignal]:
        """
        Get cached signals for a document.

        If honest_role is provided, only signals tagged with that role are
        returned.
        """
        all_signals = list(self._signals_by_doc.get(doc_id, []))
        if honest_role is None:
            return all_signals
        return [
            signal
            for signal in all_signals
            if signal.metadata.get("honest_role") == honest_role
        ]

    def update_signals(
        self,
        doc_id: str,
        signals: Sequence[ChunkFeedbackSignal],
        *,
        honest_role: Optional[str] = None,
        replace_existing: bool = False,
        max_signals: int = 2048,
    ) -> None:
        """Append or replace feedback signals for a document."""
        updated_signals: List[ChunkFeedbackSignal] = []
        for signal in signals:
            if honest_role is not None:
                signal.metadata["honest_role"] = honest_role
            updated_signals.append(signal)

        if replace_existing or doc_id not in self._signals_by_doc:
            merged = updated_signals
        else:
            merged = self._signals_by_doc[doc_id] + updated_signals
        if max_signals > 0 and len(merged) > max_signals:
            merged = merged[-max_signals:]
        self._signals_by_doc[doc_id] = merged

    def get_signals_for_chunking(
        self,
        doc_id: str,
        *,
        honest_policy: Optional[HonestChunkingPolicy] = None,
    ) -> List[ChunkFeedbackSignal]:
        """
        Return signals allowed for chunk-boundary adaptation.

        If honesty is enabled, this filters to boundary-role signals only.
        """
        if honest_policy is None or not honest_policy.enabled:
            return self.get_signals(doc_id)
        return self.get_signals(doc_id, honest_role=honest_policy.boundary_role)

    def get_signals_for_evaluation(
        self,
        doc_id: str,
        *,
        honest_policy: Optional[HonestChunkingPolicy] = None,
    ) -> List[ChunkFeedbackSignal]:
        """
        Return signals allowed for oracle evaluation.

        If honesty is enabled, this filters to held-out evaluation-role signals.
        """
        if honest_policy is None or not honest_policy.enabled:
            return self.get_signals(doc_id)
        return self.get_signals(doc_id, honest_role=honest_policy.evaluation_role)

    def clear(self, doc_id: Optional[str] = None) -> None:
        """Clear all feedback or only one document's feedback."""
        if doc_id is None:
            self._signals_by_doc.clear()
            return
        self._signals_by_doc.pop(doc_id, None)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp value into [lower, upper]."""
    return max(lower, min(upper, value))


def assign_honest_split(
    sample_id: str,
    policy: Optional[HonestChunkingPolicy] = None,
) -> str:
    """
    Deterministically assign a sample to boundary/evaluation split.

    This avoids leakage between boundary adaptation and evaluation while
    remaining reproducible across runs.
    """
    if policy is None or not policy.enabled:
        return "all"

    frac = _clamp(policy.boundary_fraction)
    digest = hashlib.sha256(f"{policy.split_seed}:{sample_id}".encode("utf-8")).digest()
    draw = int.from_bytes(digest[:8], byteorder="big") / float(2**64)
    if draw < frac:
        return policy.boundary_role
    return policy.evaluation_role


def feedback_from_prediction_errors(
    chunks: Sequence[TextChunk],
    predicted_values: Sequence[float],
    target_values: Sequence[float],
    *,
    scale_min: float,
    scale_max: float,
    confidences: Optional[Sequence[float]] = None,
    honest_role: Optional[str] = None,
    source: str = "prediction_error",
) -> List[ChunkFeedbackSignal]:
    """
    Convert chunk-level prediction errors into adaptive feedback signals.

    Low-information probability is modeled as normalized absolute error:
    |pred - target| / (scale_max - scale_min), clipped to [0, 1].
    Noise probability defaults to (1 - confidence).
    """
    if len(chunks) != len(predicted_values) or len(chunks) != len(target_values):
        raise ValueError("chunks, predicted_values, and target_values must have the same length")
    if confidences is not None and len(confidences) != len(chunks):
        raise ValueError("confidences must match the number of chunks")

    scale_span = max(1e-9, scale_max - scale_min)
    signals: List[ChunkFeedbackSignal] = []

    for idx, chunk in enumerate(chunks):
        error = abs(float(predicted_values[idx]) - float(target_values[idx]))
        low_info = _clamp(error / scale_span)
        confidence = 1.0 if confidences is None else _clamp(float(confidences[idx]))
        signals.append(
            ChunkFeedbackSignal(
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                low_info_probability=low_info,
                noise_probability=_clamp(1.0 - confidence),
                confidence=confidence,
                source=source,
                metadata={
                    "chunk_index": chunk.chunk_index,
                    "predicted_value": float(predicted_values[idx]),
                    "target_value": float(target_values[idx]),
                    "normalized_error": low_info,
                },
            )
        )
        if honest_role is not None:
            signals[-1].metadata["honest_role"] = honest_role

    return signals


class Chunker:
    """
    Chunks documents using langextract's sentence-aware chunking with tiktoken.

    Supports two modes:
    1. Direct max_tokens: Pass a specific token limit
    2. Context-aware: Pass a ContextWindowManager to automatically calculate
       safe chunk sizes based on context window allocation

    Example:
        >>> # Direct mode
        >>> chunker = Chunker(max_tokens=2000)
        >>> chunks = chunker.chunk("Long document...")

        >>> # Context-aware mode (recommended)
        >>> from treepo._research.config.context_window import ContextWindowManager
        >>> manager = ContextWindowManager(context_window=32768)
        >>> chunker = Chunker(context_manager=manager)
        >>> chunks = chunker.chunk("Long document...")
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        model: str = "gpt-4",
        context_manager: Optional["ContextWindowManager"] = None,
        reserved_for_output: Optional[int] = None,
    ):
        """
        Initialize the chunker.

        Args:
            max_tokens: Maximum tokens per chunk. If None and context_manager
                       is provided, calculated automatically.
            model: Model name for token counting (e.g., "gpt-4", "qwen3")
            context_manager: Optional ContextWindowManager for context-aware
                            chunk sizing. Takes precedence over max_tokens.
            reserved_for_output: When using context_manager, tokens to reserve
                                for output. Defaults to manager.max_output_tokens.
        """
        self.model = model
        self._token_counter = None
        self._tokenizer = RegexTokenizer()
        self.context_manager = context_manager

        # Calculate max_tokens from context_manager if provided
        if context_manager is not None:
            self.max_tokens = context_manager.get_chunk_size(reserved_for_output)
        elif max_tokens is not None:
            self.max_tokens = max_tokens
        else:
            self.max_tokens = 2000  # Default

    def _get_token_counter(self):
        """Lazy load token counter."""
        if self._token_counter is None:
            from treepo._research.preprocessing.tokenizer import TokenCounter
            self._token_counter = TokenCounter(model=self.model)
        return self._token_counter

    def chunk(self, text: str) -> List[TextChunk]:
        """
        Chunk text using langextract's sentence-aware chunking.

        Args:
            text: Text to chunk

        Returns:
            List of TextChunk objects with token counts
        """
        if not text or not text.strip():
            return []

        counter = self._get_token_counter()

        # Estimate max_chars from max_tokens
        max_chars = counter.estimate_chars_from_tokens(self.max_tokens)

        chunk_iter = ChunkIterator(
            text=text,
            max_char_buffer=max_chars,
            tokenizer_impl=self._tokenizer
        )

        chunks = []
        for i, le_chunk in enumerate(chunk_iter):
            chunk_text = le_chunk.chunk_text
            char_interval = le_chunk.char_interval
            token_count = counter.count(chunk_text)

            chunks.append(TextChunk(
                text=chunk_text,
                start_char=char_interval.start_pos,
                end_char=char_interval.end_pos,
                chunk_index=i,
                token_count=token_count,
            ))

        return chunks

    def chunk_file(self, filepath: Path) -> List[TextChunk]:
        """
        Load and chunk a text file.

        Args:
            filepath: Path to the text file

        Returns:
            List of TextChunk objects
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        text = filepath.read_text(encoding='utf-8')
        chunks = self.chunk(text)

        for chunk in chunks:
            chunk.metadata['source_file'] = str(filepath)

        return chunks

    def iter_chunks(self, text: str) -> Iterator[TextChunk]:
        """Iterate over chunks."""
        for chunk in self.chunk(text):
            yield chunk


def chunk_text(
    text: str,
    max_tokens: int = 2000,
    model: str = "gpt-4"
) -> List[TextChunk]:
    """
    Convenience function for chunking text.

    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        model: Model name for token counting

    Returns:
        List of TextChunk objects
    """
    chunker = Chunker(max_tokens=max_tokens, model=model)
    return chunker.chunk(text)


def chunk_for_ops_token_budget(
    text: str,
    *,
    max_tokens: int,
    encoding: str = "cl100k_base",
    overlap_tokens: int = 0,
) -> List[TextChunk]:
    """
    Chunk text into contiguous windows by token budget.

    This is the simplest way to precommit to a fixed tree structure: the leaf
    partition is fully determined by (encoding, max_tokens, overlap_tokens)
    before any summarization happens.

    Notes:
    - The returned chunks form an exact partition of the input when
      overlap_tokens=0.
    - Character offsets are computed deterministically from the tokenization.
    - This chunker ignores the axis/sentence/paragraph strategies and is
      intended for token-budget-first workflows (e.g., paper examples).
    """
    if not text or not text.strip():
        return []

    max_tokens = max(1, int(max_tokens))
    overlap_tokens = max(0, int(overlap_tokens))
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be < max_tokens")

    from treepo._research.preprocessing.tokenizer import TokenCounter

    counter = TokenCounter(model=None, encoding=str(encoding))
    token_offsets = counter.encode_with_offsets(text)
    if not token_offsets:
        return []

    token_ids = [tok_id for tok_id, _, _ in token_offsets]
    step = max(1, max_tokens - overlap_tokens)

    chunks: List[TextChunk] = []
    chunk_idx = 0
    for start in range(0, len(token_ids), step):
        end = min(len(token_ids), start + max_tokens)
        if end <= start:
            break

        start_char = int(token_offsets[start][1])
        end_char = int(token_offsets[end - 1][2])
        if end_char <= start_char:
            continue

        chunks.append(
            TextChunk(
                text=text[start_char:end_char],
                start_char=start_char,
                end_char=end_char,
                chunk_index=chunk_idx,
                token_count=end - start,
                metadata={
                    "token_budget": max_tokens,
                    "overlap_tokens": overlap_tokens,
                    "encoding": counter.encoding_name,
                },
            )
        )
        chunk_idx += 1
        if end >= len(token_ids):
            break

    return chunks


def chunk_for_ops(
    text: str,
    max_chars: int = 2000,
    max_tokens: Optional[int] = None,
    token_encoding: str = "cl100k_base",
    overlap_tokens: int = 0,
    strategy: str = "axis",
    adaptive_config: Optional[AdaptiveChunkingConfig] = None,
    feedback_signals: Optional[Sequence[ChunkFeedbackSignal]] = None,
) -> List[TextChunk]:
    """
    Chunk text for OPS tree construction.

    Character-axis chunking for OPS with optional sentence/paragraph modes.

    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        max_tokens: Optional maximum tokens per chunk. When provided, token
            budget takes precedence and leaf boundaries are computed directly
            from the tokenization in one pass.
        token_encoding: Encoding used for token-budget chunking.
        overlap_tokens: Optional overlap for token-budget chunking.
        strategy: Chunking strategy:
            - "axis" (default): fixed-width axis bins without sentence parsing
            - "sentence": sentence boundary segmentation
            - "paragraph": paragraph boundary segmentation
        adaptive_config: Optional adaptive chunk policy configuration
        feedback_signals: Optional span-level low-info/noise feedback

    Returns:
        List of TextChunk objects
    """
    if not text or not text.strip():
        return []

    if max_tokens is not None:
        return chunk_for_ops_token_budget(
            text,
            max_tokens=max_tokens,
            encoding=str(token_encoding or "cl100k_base"),
            overlap_tokens=overlap_tokens,
        )

    axis_segment_chars = max(128, min(int(max_chars), 1200))
    segments = _segment_intervals(
        text,
        strategy,
        axis_segment_chars=axis_segment_chars,
    )
    if not segments:
        return []

    if adaptive_config is None or not adaptive_config.enabled:
        return _build_fixed_chunks(text, segments, max_chars=max_chars)

    return _build_adaptive_chunks(
        text=text,
        segments=segments,
        max_chars=max_chars,
        config=adaptive_config,
        feedback_signals=feedback_signals,
    )


def _axis_segment_intervals(text: str, axis_segment_chars: int) -> List[tuple[int, int]]:
    """Extract fixed-width axis intervals with light whitespace alignment."""
    if not text:
        return []

    n_chars = len(text)
    target = max(32, int(axis_segment_chars))
    intervals: List[tuple[int, int]] = []
    start = 0
    while start < n_chars:
        end = min(n_chars, start + target)
        if end < n_chars:
            # Favor nearby whitespace so boundaries are less jagged, but never
            # backtrack too aggressively.
            min_backtrack = start + max(1, target // 3)
            left = text.rfind(" ", min_backtrack, end + 1)
            if left >= min_backtrack:
                end = left + 1
            else:
                search_right = min(n_chars, end + max(16, target // 8))
                right = text.find(" ", end, search_right)
                if right != -1 and right > start:
                    end = right + 1
        if end <= start:
            end = min(n_chars, start + target)
        intervals.append((start, end))
        start = end
    return intervals


def _segment_intervals(
    text: str,
    strategy: str,
    *,
    axis_segment_chars: int,
) -> List[tuple[int, int]]:
    """Extract chunking segments for the requested strategy."""
    normalized = str(strategy or "axis").strip().lower()

    if normalized == "paragraph":
        pattern = r"(?:[^\n]|\n(?!\n))+"
        return [
            (match.start(), match.end())
            for match in re.finditer(pattern, text)
            if match.group(0).strip()
        ]

    if normalized == "sentence":
        pattern = r"[^.!?]+[.!?]?(?:\s+|$)"
        return [
            (match.start(), match.end())
            for match in re.finditer(pattern, text)
            if match.group(0).strip()
        ]

    return _axis_segment_intervals(text, axis_segment_chars=axis_segment_chars)


def _build_fixed_chunks(
    text: str,
    segments: Sequence[tuple[int, int]],
    *,
    max_chars: int,
) -> List[TextChunk]:
    """Original fixed-size chunking behavior."""
    chunks: List[TextChunk] = []
    current_start: Optional[int] = None
    current_end: Optional[int] = None

    for start, end in segments:
        if current_start is None:
            current_start = start
            current_end = end
            continue

        if end - current_start > max_chars:
            chunk_text = text[current_start:current_end]
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    start_char=current_start,
                    end_char=current_end,
                    chunk_index=len(chunks),
                )
            )
            current_start = start
            current_end = end
        else:
            current_end = end

    if current_start is not None and current_end is not None:
        chunk_text = text[current_start:current_end]
        chunks.append(
            TextChunk(
                text=chunk_text,
                start_char=current_start,
                end_char=current_end,
                chunk_index=len(chunks),
            )
        )

    return chunks


def _segment_low_info_noise_proxy(segment_text: str) -> tuple[float, float]:
    """
    Return coarse (low_info, noise) proxies in [0, 1] for a segment.

    This is intentionally lightweight and model-free:
    - lower lexical diversity + short/simple content -> higher low_info
    - punctuation-heavy segments -> higher noise
    """
    tokens = re.findall(r"[A-Za-z0-9_]+", segment_text.lower())
    if not tokens:
        return 1.0, 0.5

    token_count = len(tokens)
    unique_ratio = len(set(tokens)) / token_count
    long_token_ratio = sum(1 for tok in tokens if len(tok) >= 7) / token_count
    digit_ratio = sum(1 for ch in segment_text if ch.isdigit()) / max(1, len(segment_text))
    punct_ratio = sum(1 for ch in segment_text if ch in ",;:()[]{}<>/\\|") / max(
        1, len(segment_text)
    )

    info_proxy = (
        0.45 * unique_ratio
        + 0.25 * _clamp(token_count / 30.0)
        + 0.20 * long_token_ratio
        + 0.10 * _clamp(digit_ratio * 25.0)
    )
    info_proxy = _clamp(info_proxy)
    low_info_proxy = _clamp(1.0 - info_proxy)
    noise_proxy = _clamp(punct_ratio * 8.0)
    return low_info_proxy, noise_proxy


def _overlap_width(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Character overlap width between two half-open intervals."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _aggregate_feedback_for_span(
    start_char: int,
    end_char: int,
    feedback_signals: Optional[Sequence[ChunkFeedbackSignal]],
) -> tuple[float, float, float, float]:
    """
    Aggregate overlapping feedback for a segment.

    Returns:
        (low_info, noise, weight, oracle_relevance)
    """
    if not feedback_signals:
        return 0.0, 0.0, 0.0, 0.0

    span = max(1, end_char - start_char)
    low_sum = 0.0
    noise_sum = 0.0
    rel_sum = 0.0
    weight_sum = 0.0

    for signal in feedback_signals:
        overlap = _overlap_width(start_char, end_char, signal.start_char, signal.end_char)
        if overlap <= 0:
            continue
        weight = (overlap / span) * _clamp(float(signal.confidence))
        if weight <= 0:
            continue
        if signal.oracle_relevance_probability is not None:
            relevance = _clamp(float(signal.oracle_relevance_probability))
            low_value = _clamp(1.0 - relevance)
        else:
            low_value = _clamp(float(signal.low_info_probability))
            relevance = _clamp(1.0 - low_value)
        low_sum += weight * low_value
        noise_sum += weight * _clamp(float(signal.noise_probability))
        rel_sum += weight * relevance
        weight_sum += weight

    if weight_sum <= 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        low_sum / weight_sum,
        noise_sum / weight_sum,
        _clamp(weight_sum),
        rel_sum / weight_sum,
    )


def _adaptive_target_chars(
    base_chars: int,
    avg_low_info: float,
    avg_noise: float,
    config: AdaptiveChunkingConfig,
) -> tuple[int, float]:
    """Compute adaptive target size and scale factor."""
    low_info = _clamp(avg_low_info)
    noise = _clamp(avg_noise)
    info = 1.0 - low_info

    scale = (
        1.0
        + config.low_info_expansion_weight * low_info
        + config.noise_expansion_weight * noise
        - config.high_info_compression_weight * info
    )
    scale = _clamp(scale, config.min_target_scale, config.max_target_scale)

    min_chars = max(1, config.min_chars)
    max_chars = max(min_chars, config.max_chars)
    target = int(round(_clamp(base_chars * scale, float(min_chars), float(max_chars))))
    return target, scale


def _build_adaptive_chunks(
    text: str,
    segments: Sequence[tuple[int, int]],
    *,
    max_chars: int,
    config: AdaptiveChunkingConfig,
    feedback_signals: Optional[Sequence[ChunkFeedbackSignal]],
) -> List[TextChunk]:
    """Build chunks with adaptive target sizes."""
    chunks: List[TextChunk] = []
    current_start: Optional[int] = None
    current_end: Optional[int] = None

    low_info_weighted_sum = 0.0
    noise_weighted_sum = 0.0
    relevance_weighted_sum = 0.0
    char_weight_sum = 0.0
    chunk_feedback_weight = 0.0
    target_chars = max_chars
    target_scale = 1.0

    def finalize_current() -> None:
        nonlocal low_info_weighted_sum
        nonlocal noise_weighted_sum
        nonlocal relevance_weighted_sum
        nonlocal char_weight_sum
        nonlocal chunk_feedback_weight
        nonlocal target_chars
        nonlocal target_scale
        if current_start is None or current_end is None:
            return
        avg_low_info = low_info_weighted_sum / max(1.0, char_weight_sum)
        avg_noise = noise_weighted_sum / max(1.0, char_weight_sum)
        avg_relevance = relevance_weighted_sum / max(1.0, char_weight_sum)
        metadata = {
            "adaptive_policy": "low_info_noise_proxy",
            "avg_low_info_probability": avg_low_info,
            "avg_noise_probability": avg_noise,
            "avg_oracle_relevance_probability": avg_relevance,
            "adaptive_target_chars": target_chars,
            "adaptive_target_scale": target_scale,
            "feedback_weight": _clamp(chunk_feedback_weight),
        }
        chunks.append(
            TextChunk(
                text=text[current_start:current_end],
                start_char=current_start,
                end_char=current_end,
                chunk_index=len(chunks),
                metadata=metadata,
            )
        )

    for start, end in segments:
        segment_text = text[start:end]
        segment_chars = max(1, end - start)
        proxy_low, proxy_noise = _segment_low_info_noise_proxy(segment_text)
        fb_low, fb_noise, fb_weight, _fb_rel = _aggregate_feedback_for_span(start, end, feedback_signals)

        if fb_weight > 0:
            # As feedback coverage/confidence increases, rely more on learned
            # relevance/noise signals and less on coarse lexical heuristics.
            base_blend = _clamp(config.proxy_blend)
            effective_blend = _clamp(base_blend * (1.0 - _clamp(fb_weight)))
            seg_low = effective_blend * proxy_low + (1.0 - effective_blend) * fb_low
            seg_noise = effective_blend * proxy_noise + (1.0 - effective_blend) * fb_noise
            seg_rel = _clamp(1.0 - seg_low)
        else:
            seg_low = proxy_low
            seg_noise = proxy_noise
            seg_rel = _clamp(1.0 - seg_low)

        if current_start is None:
            current_start = start
            current_end = end
            low_info_weighted_sum = seg_low * segment_chars
            noise_weighted_sum = seg_noise * segment_chars
            relevance_weighted_sum = seg_rel * segment_chars
            char_weight_sum = float(segment_chars)
            chunk_feedback_weight = fb_weight
            avg_low = low_info_weighted_sum / char_weight_sum
            avg_noise = noise_weighted_sum / char_weight_sum
            target_chars, target_scale = _adaptive_target_chars(max_chars, avg_low, avg_noise, config)
            continue

        candidate_low_sum = low_info_weighted_sum + seg_low * segment_chars
        candidate_noise_sum = noise_weighted_sum + seg_noise * segment_chars
        candidate_rel_sum = relevance_weighted_sum + seg_rel * segment_chars
        candidate_char_sum = char_weight_sum + segment_chars
        avg_low = candidate_low_sum / candidate_char_sum
        avg_noise = candidate_noise_sum / candidate_char_sum
        candidate_target_chars, candidate_target_scale = _adaptive_target_chars(
            max_chars, avg_low, avg_noise, config
        )
        candidate_span = end - current_start

        if candidate_span > candidate_target_chars and current_end is not None:
            finalize_current()
            current_start = start
            current_end = end
            low_info_weighted_sum = seg_low * segment_chars
            noise_weighted_sum = seg_noise * segment_chars
            relevance_weighted_sum = seg_rel * segment_chars
            char_weight_sum = float(segment_chars)
            chunk_feedback_weight = fb_weight
            avg_low = low_info_weighted_sum / char_weight_sum
            avg_noise = noise_weighted_sum / char_weight_sum
            target_chars, target_scale = _adaptive_target_chars(max_chars, avg_low, avg_noise, config)
        else:
            current_end = end
            low_info_weighted_sum = candidate_low_sum
            noise_weighted_sum = candidate_noise_sum
            relevance_weighted_sum = candidate_rel_sum
            char_weight_sum = candidate_char_sum
            chunk_feedback_weight = _clamp((chunk_feedback_weight + fb_weight) / 2.0)
            target_chars = candidate_target_chars
            target_scale = candidate_target_scale

    finalize_current()
    return chunks


# =============================================================================
# Oracle → Feedback Signals (Phase 2: feedback loop)
# =============================================================================


def oracle_to_feedback_signals(
    nodes: Any,
    oracle_key: str = "rile",
    *,
    target_min: float = -100.0,
    target_max: float = 100.0,
    high_error_threshold: float = 0.3,
    source: str = "oracle_audit",
) -> List[ChunkFeedbackSignal]:
    """Convert oracle audit results on tree nodes to chunking feedback signals.

    For each node that has both an oracle score and a sketch prediction, this
    computes the discrepancy and maps it to a ``ChunkFeedbackSignal``:

    - **High discrepancy** (sketch disagrees with oracle) →
      ``oracle_relevance_probability`` is LOW, signaling that the window
      boundaries in this region should be **refined** (the sketch didn't
      capture the right information from these windows).
    - **Low discrepancy** (sketch agrees with oracle) →
      ``oracle_relevance_probability`` is HIGH, signaling the windows are
      **good** and can be merged/expanded.

    Args:
        nodes: List of ``EmbeddingTreeNode`` (from ``build_unified_tree``).
        oracle_key: Which oracle score to use (default ``"rile"``).
        target_min: Min value of the oracle scale (for normalisation).
        target_max: Max value of the oracle scale.
        high_error_threshold: Normalised error above which we flag for
            refinement (0.3 = 30% of scale range).
        source: Source tag for the feedback signals.

    Returns:
        List of ``ChunkFeedbackSignal`` covering audited regions.
    """
    scale = max(1.0, target_max - target_min)
    signals: List[ChunkFeedbackSignal] = []

    for node in nodes:
        oracle_val = node.oracle_scores.get(oracle_key) if hasattr(node, "oracle_scores") else None
        sketch_val = node.sketch_scores.get(oracle_key) if hasattr(node, "sketch_scores") else None

        if oracle_val is None:
            continue

        # Compute normalised error
        if sketch_val is not None:
            norm_error = abs(oracle_val - sketch_val) / scale
        else:
            # No sketch prediction → moderate uncertainty
            norm_error = 0.5

        # Map: high error → low oracle_relevance (needs refinement)
        # Low error → high oracle_relevance (windows are fine)
        oracle_relevance = max(0.0, min(1.0, 1.0 - norm_error))

        # Low info = inverse of relevance (high error means current windows
        # are not capturing the signal well)
        low_info_prob = max(0.0, min(1.0, norm_error))

        signals.append(ChunkFeedbackSignal(
            start_char=node.char_start,
            end_char=node.char_end,
            low_info_probability=low_info_prob,
            noise_probability=0.0,
            confidence=1.0 if sketch_val is not None else 0.5,
            source=source,
            oracle_relevance_probability=oracle_relevance,
        ))

    return signals


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================
DocumentChunker = Chunker
ParagraphChunker = Chunker  # Legacy name, Chunker handles paragraphs
