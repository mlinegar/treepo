"""
Bridge between VLM visual segments and the chunk feedback signal system.

Converts VisualSegment objects into ChunkFeedbackSignal objects that plug
directly into the existing adaptive chunking pipeline (chunk_for_ops).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from treepo._research.parsers.vlm_segmenter import VisualSegment
from treepo._research.preprocessing.chunker import ChunkFeedbackSignal


def map_segments_to_char_ranges(
    segments: List[VisualSegment],
    pages: List[str],
    page_char_ranges: Sequence[Tuple[int, int]],
) -> List[VisualSegment]:
    """
    Map visual segment bounding boxes to character positions in the joined document text.

    Uses the bbox vertical position (y0, y1) as a proxy for character position
    within the page text.  This is approximate but sufficient for feedback signal
    generation, since the adaptive chunking system blends overlapping signals
    via ``_aggregate_feedback_for_span()``.

    Args:
        segments: VLM-produced segments (char_start/char_end not yet set).
        pages: List of page texts.
        page_char_ranges: ``[(start, end), ...]`` mapping each page index to
            character positions in the joined document text (produced by
            ``PDFTextParser``).

    Returns:
        Same segment list with ``char_start`` and ``char_end`` populated.
    """
    for segment in segments:
        page_idx = segment.page_index
        if page_idx < 0 or page_idx >= len(page_char_ranges):
            segment.char_start = 0
            segment.char_end = 0
            continue

        page_start, page_end = page_char_ranges[page_idx]
        page_char_count = max(1, page_end - page_start)

        # Use bbox vertical position as proxy for character position.
        _, y0, _, y1 = segment.bbox
        y0_clamped = max(0.0, min(1.0, y0))
        y1_clamped = max(y0_clamped, min(1.0, y1))

        segment.char_start = page_start + int(y0_clamped * page_char_count)
        segment.char_end = page_start + int(y1_clamped * page_char_count)

        # Ensure nonzero width.
        if segment.char_end <= segment.char_start:
            segment.char_end = min(page_end, segment.char_start + 1)

    return segments


def segments_to_feedback_signals(
    segments: List[VisualSegment],
    *,
    source: str = "vlm_visual_segmentation",
) -> List[ChunkFeedbackSignal]:
    """
    Convert VLM visual segments into :class:`ChunkFeedbackSignal` objects.

    Mapping:

    * ``info_score`` → ``oracle_relevance_probability`` (direct)
    * ``info_score`` → ``low_info_probability = 1 - info_score`` (inverted)
    * ``confidence`` → ``confidence`` (direct)
    * ``1 - confidence`` → ``noise_probability``

    The returned signals plug directly into
    ``chunk_for_ops(feedback_signals=...)``.  The existing
    ``_aggregate_feedback_for_span()`` blends overlapping signals weighted
    by overlap width × confidence.
    """
    signals: List[ChunkFeedbackSignal] = []
    for segment in segments:
        if segment.char_end <= segment.char_start:
            continue
        signals.append(
            ChunkFeedbackSignal(
                start_char=segment.char_start,
                end_char=segment.char_end,
                low_info_probability=max(0.0, min(1.0, 1.0 - segment.info_score)),
                noise_probability=max(0.0, min(1.0, 1.0 - segment.confidence)),
                confidence=max(0.0, min(1.0, segment.confidence)),
                source=source,
                oracle_relevance_probability=max(0.0, min(1.0, segment.info_score)),
                metadata={
                    "segment_type": segment.segment_type,
                    "bbox": list(segment.bbox),
                    "page_index": segment.page_index,
                    "vlm_info_score": segment.info_score,
                },
            )
        )
    return signals


def extract_content_weights_from_chunks(
    chunks: List[Any],
) -> Dict[str, float]:
    """
    Extract per-leaf content weights from chunk metadata for audit sampling.

    Reads ``avg_oracle_relevance_probability`` set by
    ``_build_adaptive_chunks()`` in the chunker and returns a dict mapping
    ``leaf_<index>`` → info_score.

    Used by the ``CONTENT_WEIGHTED`` auditor sampling strategy to set PPS
    inclusion probabilities.
    """
    weights: Dict[str, float] = {}
    for i, chunk in enumerate(chunks):
        meta = getattr(chunk, "metadata", {}) if not isinstance(chunk, dict) else chunk
        if isinstance(meta, dict):
            score = meta.get("avg_oracle_relevance_probability", 0.5)
        else:
            score = 0.5
        weights[f"leaf_{i}"] = float(score)
    return weights
