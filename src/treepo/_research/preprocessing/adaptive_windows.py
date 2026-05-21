"""
Adaptive windowing utilities for embedding-driven segmentation.

The core abstraction is a 1D axis window (char index, time in ms, page index,
comment index, etc.) and a coarse-to-fine refinement policy based on local
scores and score gradients.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class AxisWindow:
    """
    A contiguous half-open interval on a 1D axis.

    Examples:
    - text chars: unit="char", start=0, end=1200
    - video ms: unit="ms", start=30_000, end=45_000
    - pdf pages: unit="page", start=10, end=14  (pages [10,14))
    - feed items: unit="item", start=80, end=120
    """

    start: int
    end: int
    unit: str = "char"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return max(0, int(self.end) - int(self.start))


def uniform_axis_windows(
    total_extent: int,
    *,
    window_size: int,
    overlap: int = 0,
    unit: str = "char",
) -> List[AxisWindow]:
    """Create fixed-width windows across a 1D axis."""
    total_extent = max(0, int(total_extent))
    if total_extent <= 0:
        return []

    window_size = max(1, int(window_size))
    overlap = max(0, min(int(overlap), window_size - 1))
    step = max(1, window_size - overlap)

    windows: List[AxisWindow] = []
    start = 0
    while start < total_extent:
        end = min(total_extent, start + window_size)
        if end > start:
            windows.append(AxisWindow(start=start, end=end, unit=unit))
        if end >= total_extent:
            break
        start += step
    return windows


def adaptive_refine_windows(
    *,
    total_extent: int,
    coarse_window_size: int,
    fine_window_size: int,
    max_windows: int,
    score_windows: Callable[[Sequence[AxisWindow]], Sequence[float]],
    uncertainty_band: Tuple[float, float] = (0.35, 0.65),
    gradient_threshold: float = 0.20,
    coarse_overlap_fraction: float = 0.10,
    fine_overlap_fraction: float = 0.125,
    unit: str = "char",
    extra_refine_predicate: Optional[Callable[[AxisWindow, float, float], bool]] = None,
) -> List[AxisWindow]:
    """
    Adaptive coarse-to-fine windowing.

    1. Build coarse windows.
    2. Score each coarse window with score_windows().
    3. Refine uncertain / high-gradient (and optionally custom) regions.
    4. Cap total windows by deterministic downsampling.
    """
    total_extent = max(0, int(total_extent))
    if total_extent <= 0:
        return []

    coarse_window_size = max(1, int(coarse_window_size))
    fine_window_size = max(1, min(int(fine_window_size), coarse_window_size))
    max_windows = max(1, int(max_windows))

    coarse_overlap = int(max(0.0, coarse_overlap_fraction) * coarse_window_size)
    coarse_windows = uniform_axis_windows(
        total_extent,
        window_size=coarse_window_size,
        overlap=coarse_overlap,
        unit=unit,
    )
    if len(coarse_windows) <= 1:
        return coarse_windows

    scores = [float(v) for v in score_windows(coarse_windows)]
    if len(scores) != len(coarse_windows):
        raise ValueError(
            f"score_windows length mismatch: expected {len(coarse_windows)}, got {len(scores)}"
        )

    lo_band = max(0.0, min(1.0, float(uncertainty_band[0])))
    hi_band = max(lo_band, min(1.0, float(uncertainty_band[1])))
    grad_th = max(0.0, float(gradient_threshold))

    refine_indices: set[int] = set()
    for idx, score in enumerate(scores):
        left = scores[idx - 1] if idx > 0 else score
        right = scores[idx + 1] if idx + 1 < len(scores) else score
        local_grad = 0.5 * (abs(score - left) + abs(right - score))
        default_refine = (lo_band <= score <= hi_band) or (local_grad >= grad_th)
        extra_refine = False
        if extra_refine_predicate is not None:
            try:
                extra_refine = bool(extra_refine_predicate(coarse_windows[idx], score, local_grad))
            except Exception:
                extra_refine = False
        if default_refine or extra_refine:
            refine_indices.add(idx)
            if idx > 0:
                refine_indices.add(idx - 1)
            if idx + 1 < len(coarse_windows):
                refine_indices.add(idx + 1)

    fine_overlap = int(max(0.0, fine_overlap_fraction) * fine_window_size)
    windows: List[AxisWindow] = []
    for idx, window in enumerate(coarse_windows):
        if idx not in refine_indices:
            windows.append(window)
            continue

        local_windows = uniform_axis_windows(
            window.width,
            window_size=fine_window_size,
            overlap=fine_overlap,
            unit=unit,
        )
        for local in local_windows:
            windows.append(
                AxisWindow(
                    start=window.start + local.start,
                    end=window.start + local.end,
                    unit=unit,
                )
            )

    # Deduplicate and sort deterministically.
    deduped: List[AxisWindow] = []
    seen = set()
    for w in sorted(windows, key=lambda x: (x.start, x.end)):
        key = (int(w.start), int(w.end), str(w.unit))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(w)

    if len(deduped) <= max_windows:
        return deduped

    stride = max(1, math.ceil(len(deduped) / float(max_windows)))
    reduced = deduped[::stride]

    # Ensure tail coverage.
    if reduced and reduced[-1].end < total_extent:
        start = max(0, total_extent - fine_window_size)
        tail = AxisWindow(start=start, end=total_extent, unit=unit)
        if (tail.start, tail.end, tail.unit) not in {
            (w.start, w.end, w.unit) for w in reduced
        }:
            reduced.append(tail)
            reduced.sort(key=lambda x: (x.start, x.end))
    return reduced


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine distance in [0, 2], with safe fallbacks for degenerate vectors."""
    if not left or not right:
        return 1.0
    if len(left) != len(right):
        return 1.0

    dot = 0.0
    left_norm_sq = 0.0
    right_norm_sq = 0.0
    for lval, rval in zip(left, right):
        l = float(lval)
        r = float(rval)
        dot += l * r
        left_norm_sq += l * l
        right_norm_sq += r * r

    if left_norm_sq <= 0.0 or right_norm_sq <= 0.0:
        return 1.0

    similarity = dot / (math.sqrt(left_norm_sq) * math.sqrt(right_norm_sq))
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def merge_adjacent_windows_by_embedding_drift(
    windows: Sequence[AxisWindow],
    embeddings: Sequence[Sequence[float]],
    *,
    max_cosine_distance: float = 0.03,
    max_merged_width: Optional[int] = None,
    max_gap: int = 0,
) -> List[AxisWindow]:
    """
    Merge adjacent windows when embedding drift is low.

    This is axis-agnostic and can be reused for text/page/time/item windows.
    """
    if len(windows) != len(embeddings):
        raise ValueError(
            f"windows/embeddings length mismatch: {len(windows)} vs {len(embeddings)}"
        )
    if not windows:
        return []

    max_cosine_distance = max(0.0, float(max_cosine_distance))
    max_gap = max(0, int(max_gap))
    merged_width_cap = None
    if max_merged_width is not None:
        merged_width_cap = max(1, int(max_merged_width))

    paired = sorted(
        zip(windows, embeddings),
        key=lambda pair: (int(pair[0].start), int(pair[0].end)),
    )

    first_window, first_embedding = paired[0]
    current = AxisWindow(
        start=int(first_window.start),
        end=int(first_window.end),
        unit=str(first_window.unit),
        metadata=dict(first_window.metadata),
    )
    current_count = 1
    current_max_internal_drift = 0.0
    previous_embedding = [float(v) for v in first_embedding]

    merged: List[AxisWindow] = []
    for next_window, next_embedding_raw in paired[1:]:
        next_embedding = [float(v) for v in next_embedding_raw]
        drift = _cosine_distance(previous_embedding, next_embedding)

        same_unit = str(next_window.unit) == str(current.unit)
        gap = int(next_window.start) - int(current.end)
        adjacent = gap <= max_gap
        next_end = max(int(current.end), int(next_window.end))
        next_width = max(0, next_end - int(current.start))
        width_ok = merged_width_cap is None or next_width <= merged_width_cap

        if same_unit and adjacent and width_ok and drift <= max_cosine_distance:
            current.end = next_end
            current_count += 1
            current_max_internal_drift = max(current_max_internal_drift, drift)
            previous_embedding = next_embedding
            continue

        if current_count > 1:
            current.metadata["merged_window_count"] = int(current_count)
            current.metadata["max_internal_cosine_distance"] = float(current_max_internal_drift)
        merged.append(current)

        current = AxisWindow(
            start=int(next_window.start),
            end=int(next_window.end),
            unit=str(next_window.unit),
            metadata=dict(next_window.metadata),
        )
        current_count = 1
        current_max_internal_drift = 0.0
        previous_embedding = next_embedding

    if current_count > 1:
        current.metadata["merged_window_count"] = int(current_count)
        current.metadata["max_internal_cosine_distance"] = float(current_max_internal_drift)
    merged.append(current)
    return merged
