"""
Adapters that map arbitrary samples onto a 1D window axis.

This lets adaptive windowing logic stay modality-agnostic while each adapter
defines how to materialize window content for scoring.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from treepo._research.preprocessing.adaptive_windows import AxisWindow, adaptive_refine_windows


@runtime_checkable
class AxisWindowAdapter(Protocol):
    """Protocol for mapping a sample to axis windows and content slices."""

    @property
    def axis_unit(self) -> str:
        """Axis unit name (e.g., char, ms, page, item)."""
        ...

    def total_extent(self, sample: Any) -> int:
        """Return total axis extent for this sample."""
        ...

    def materialize(self, sample: Any, window: AxisWindow) -> str:
        """Return model-ready content for a given window."""
        ...


def _default_text_getter(sample: Any) -> str:
    if isinstance(sample, str):
        return sample
    if isinstance(sample, dict):
        value = sample.get("text")
        return str(value or "")
    value = getattr(sample, "text", "")
    return str(value or "")


def _default_pages_getter(sample: Any) -> List[str]:
    if isinstance(sample, dict):
        pages = sample.get("pages")
    else:
        pages = getattr(sample, "pages", None)
    if isinstance(pages, Sequence) and not isinstance(pages, (str, bytes, bytearray)):
        return [str(page or "") for page in pages]
    text = _default_text_getter(sample)
    if not text:
        return []
    split_pages = [part for part in text.split("\f") if part.strip()]
    return split_pages or [text]


def _default_items_getter(sample: Any) -> List[Any]:
    if isinstance(sample, dict):
        items = sample.get("items")
    else:
        items = getattr(sample, "items", None)
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
        return list(items)

    text = _default_text_getter(sample)
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines or [text]


def _default_segments_getter(sample: Any) -> List[Any]:
    if isinstance(sample, dict):
        segments = sample.get("segments")
    else:
        segments = getattr(sample, "segments", None)
    if isinstance(segments, Sequence) and not isinstance(segments, (str, bytes, bytearray)):
        return list(segments)
    return []


@dataclass
class TextCharWindowAdapter:
    """Character-axis adapter for plain text samples."""

    text_getter: Optional[Callable[[Any], str]] = None
    _axis_unit: str = "char"

    @property
    def axis_unit(self) -> str:
        return self._axis_unit

    def _text(self, sample: Any) -> str:
        getter = self.text_getter or _default_text_getter
        return str(getter(sample) or "")

    def total_extent(self, sample: Any) -> int:
        return len(self._text(sample))

    def materialize(self, sample: Any, window: AxisWindow) -> str:
        text = self._text(sample)
        start = max(0, min(len(text), int(window.start)))
        end = max(start, min(len(text), int(window.end)))
        return text[start:end]


@dataclass
class TextPageWindowAdapter:
    """Page-axis adapter for page-oriented text sources (e.g., long PDFs)."""

    pages_getter: Optional[Callable[[Any], Sequence[str]]] = None
    page_delimiter: str = "\n\n"
    _axis_unit: str = "page"

    @property
    def axis_unit(self) -> str:
        return self._axis_unit

    def _pages(self, sample: Any) -> List[str]:
        getter = self.pages_getter or _default_pages_getter
        pages = getter(sample)
        return [str(page or "") for page in pages]

    def total_extent(self, sample: Any) -> int:
        return len(self._pages(sample))

    def materialize(self, sample: Any, window: AxisWindow) -> str:
        pages = self._pages(sample)
        start = max(0, min(len(pages), int(window.start)))
        end = max(start, min(len(pages), int(window.end)))
        if end <= start:
            return ""
        return self.page_delimiter.join(pages[start:end])


@dataclass
class SequenceItemWindowAdapter:
    """Item-axis adapter for ordered feeds (comments/posts/events)."""

    items_getter: Optional[Callable[[Any], Sequence[Any]]] = None
    item_renderer: Optional[Callable[[Any], str]] = None
    item_delimiter: str = "\n"
    _axis_unit: str = "item"

    @property
    def axis_unit(self) -> str:
        return self._axis_unit

    def _items(self, sample: Any) -> List[Any]:
        getter = self.items_getter or _default_items_getter
        return list(getter(sample))

    def _render_item(self, item: Any) -> str:
        if self.item_renderer is not None:
            return str(self.item_renderer(item) or "")
        return str(item or "")

    def total_extent(self, sample: Any) -> int:
        return len(self._items(sample))

    def materialize(self, sample: Any, window: AxisWindow) -> str:
        items = self._items(sample)
        start = max(0, min(len(items), int(window.start)))
        end = max(start, min(len(items), int(window.end)))
        if end <= start:
            return ""
        return self.item_delimiter.join(self._render_item(item) for item in items[start:end])


@dataclass
class TimeSegmentWindowAdapter:
    """
    Time-axis adapter for segment streams (e.g., video transcript windows).

    Expected segment schema:
    - dict/object with start/end keys in the configured time unit
    - optional text payload
    """

    segments_getter: Optional[Callable[[Any], Sequence[Any]]] = None
    start_key: str = "start"
    end_key: str = "end"
    text_key: str = "text"
    include_timestamps: bool = True
    _axis_unit: str = "ms"

    @property
    def axis_unit(self) -> str:
        return self._axis_unit

    def _segment_value(self, segment: Any, key: str, default: Any = None) -> Any:
        if isinstance(segment, dict):
            return segment.get(key, default)
        return getattr(segment, key, default)

    def _segments(self, sample: Any) -> List[Dict[str, Any]]:
        getter = self.segments_getter or _default_segments_getter
        raw_segments = getter(sample)
        normalized: List[Dict[str, Any]] = []
        for segment in raw_segments:
            start = self._segment_value(segment, self.start_key, None)
            end = self._segment_value(segment, self.end_key, None)
            if start is None:
                start = self._segment_value(segment, "start", None)
            if end is None:
                end = self._segment_value(segment, "end", None)
            if start is None or end is None:
                continue
            try:
                start_i = int(start)
                end_i = int(end)
            except (TypeError, ValueError):
                continue
            if end_i <= start_i:
                continue
            text = self._segment_value(segment, self.text_key, "")
            normalized.append(
                {
                    "start": start_i,
                    "end": end_i,
                    "text": str(text or ""),
                }
            )
        normalized.sort(key=lambda seg: (seg["start"], seg["end"]))
        return normalized

    def total_extent(self, sample: Any) -> int:
        segments = self._segments(sample)
        if not segments:
            return 0
        return max(int(seg["end"]) for seg in segments)

    def materialize(self, sample: Any, window: AxisWindow) -> str:
        segments = self._segments(sample)
        if not segments:
            return ""

        start = int(window.start)
        end = int(window.end)
        if end <= start:
            return ""

        parts: List[str] = []
        for seg in segments:
            seg_start = int(seg["start"])
            seg_end = int(seg["end"])
            if seg_end <= start or seg_start >= end:
                continue
            seg_text = seg["text"]
            if self.include_timestamps:
                parts.append(f"[{seg_start}-{seg_end}] {seg_text}")
            else:
                parts.append(seg_text)
        return "\n".join(parts)


@dataclass
class VisualRegionWindowAdapter:
    """Visual-region-axis adapter for VLM-segmented PDF pages."""

    _axis_unit: str = "visual_region"

    @property
    def axis_unit(self) -> str:
        return self._axis_unit

    def _get_segments(self, sample: Any) -> List[Dict[str, Any]]:
        """Get visual segments from sample metadata."""
        if isinstance(sample, dict):
            metadata = sample.get("metadata", {})
        else:
            metadata = getattr(sample, "metadata", {})
        if not isinstance(metadata, dict):
            return []
        segments = metadata.get("visual_segments", [])
        if not isinstance(segments, list):
            return []
        return segments

    def total_extent(self, sample: Any) -> int:
        return len(self._get_segments(sample))

    def materialize(self, sample: Any, window: AxisWindow) -> str:
        segments = self._get_segments(sample)
        start = max(0, min(len(segments), int(window.start)))
        end = max(start, min(len(segments), int(window.end)))
        if end <= start:
            return ""
        return "\n".join(
            str(seg.get("text_content", "") if isinstance(seg, dict) else "")
            for seg in segments[start:end]
        )


def build_window_adapter(adapter_name: str) -> AxisWindowAdapter:
    """
    Build a named window adapter.

    Names:
    - text_char / char / text
    - text_page / page / pdf_page
    - sequence_item / item / feed_item
    - time_segment / time / video_time
    """
    normalized = str(adapter_name or "text_char").strip().lower()
    if normalized in {"text_char", "char", "text"}:
        return TextCharWindowAdapter()
    if normalized in {"text_page", "page", "pdf_page"}:
        return TextPageWindowAdapter()
    if normalized in {"sequence_item", "item", "feed_item"}:
        return SequenceItemWindowAdapter()
    if normalized in {"time_segment", "time", "video_time"}:
        return TimeSegmentWindowAdapter()
    if normalized in {"visual_region", "vlm_region", "visual"}:
        return VisualRegionWindowAdapter()
    raise ValueError(
        f"Unknown adapter '{adapter_name}'. "
        "Expected one of: text_char, text_page, sequence_item, time_segment, visual_region"
    )


def build_adaptive_windows_for_sample(
    *,
    sample: Any,
    adapter: AxisWindowAdapter,
    score_materialized_windows: Callable[[Sequence[str], Sequence[AxisWindow]], Sequence[float]],
    coarse_window_size: int,
    fine_window_size: int,
    max_windows: int,
    uncertainty_band: tuple[float, float] = (0.35, 0.65),
    gradient_threshold: float = 0.20,
    coarse_overlap_fraction: float = 0.10,
    fine_overlap_fraction: float = 0.125,
    extra_refine_predicate: Optional[Callable[[AxisWindow, float, float], bool]] = None,
) -> list[AxisWindow]:
    """Build adaptive windows for a sample via the provided adapter."""
    total_extent = max(0, int(adapter.total_extent(sample)))
    if total_extent <= 0:
        return []

    def _score_windows(windows: Sequence[AxisWindow]) -> Sequence[float]:
        window_payloads = [adapter.materialize(sample, window) for window in windows]
        return score_materialized_windows(window_payloads, windows)

    return adaptive_refine_windows(
        total_extent=total_extent,
        coarse_window_size=coarse_window_size,
        fine_window_size=fine_window_size,
        max_windows=max_windows,
        score_windows=_score_windows,
        uncertainty_band=uncertainty_band,
        gradient_threshold=gradient_threshold,
        coarse_overlap_fraction=coarse_overlap_fraction,
        fine_overlap_fraction=fine_overlap_fraction,
        unit=adapter.axis_unit,
        extra_refine_predicate=extra_refine_predicate,
    )
