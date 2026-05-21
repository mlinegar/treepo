"""
VLM-based visual PDF segmentation.

Segments PDF pages into semantically meaningful regions (tables, figures,
equations, body text, headers) using a Vision Language Model, and scores
each region's information density for downstream adaptive chunking and
content-weighted audit sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple
import logging

logger = logging.getLogger(__name__)


# --- Data model ---

@dataclass
class VisualSegment:
    """A visually-identified region on a PDF page."""

    page_index: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) normalized [0,1]
    segment_type: str  # "table", "figure", "header", "body_text", "equation", etc.
    text_content: str  # VLM-extracted text within this region
    info_score: float  # [0, 1] — higher = more informative
    confidence: float  # VLM's confidence in the segmentation
    char_start: int = 0  # mapped to document char position (set by visual_feedback.map_segments_to_char_ranges)
    char_end: int = 0  # mapped to document char position
    metadata: Dict[str, Any] = field(default_factory=dict)


# --- Type weights for information scoring ---

DEFAULT_TYPE_WEIGHTS: Dict[str, float] = {
    "table": 0.92,
    "figure": 0.88,
    "equation": 0.85,
    "code": 0.82,
    "list": 0.65,
    "body_text": 0.50,
    "caption": 0.45,
    "header": 0.25,
    "footer": 0.10,
    "page_number": 0.05,
    "whitespace": 0.02,
}


def compute_info_score(
    segment_type: str,
    text_content: str,
    bbox: Tuple[float, float, float, float],
    confidence: float,
    *,
    type_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute information score for a visual segment.

    Combines segment type, content density, and VLM confidence.
    Returns a value in [0, 1] where higher = more informative.
    """
    weights = type_weights or DEFAULT_TYPE_WEIGHTS
    type_w = weights.get(segment_type, 0.50)

    # Content density: non-whitespace chars / bbox area
    x0, y0, x1, y1 = bbox
    bbox_area = max(1e-6, abs(x1 - x0) * abs(y1 - y0))
    non_ws = sum(1 for c in text_content if not c.isspace())
    # Normalize: ~500 chars per full-page bbox is moderate density
    density_raw = non_ws / (bbox_area * 2000)
    content_density = min(1.0, density_raw)

    # VLM confidence as visual complexity proxy
    visual_complexity = max(0.0, min(1.0, confidence))

    score = (
        0.45 * type_w
        + 0.30 * content_density
        + 0.25 * visual_complexity
    )
    return max(0.0, min(1.0, score))


# --- Backend protocol ---

class VLMSegmenterBackend(Protocol):
    """Protocol for VLM segmentation backends."""

    def segment_page(self, page_image: bytes, page_index: int) -> List[VisualSegment]:
        ...


class SmolDoclingBackend:
    """
    SmolDocling backend (256M params, 0.35s/page).

    Calls a SmolDocling inference endpoint that accepts page images and
    returns DocTags with bounding boxes and semantic type labels.

    Expected endpoint response format::

        {
            "status": "ok",
            "regions": [
                {
                    "type": "table",
                    "bbox": [x0, y0, x1, y1],
                    "text": "...",
                    "confidence": 0.95
                },
                ...
            ]
        }
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8010/v1/segment",
        timeout_seconds: float = 30.0,
        type_weights: Optional[Dict[str, float]] = None,
    ):
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.type_weights = type_weights

    def segment_page(self, page_image: bytes, page_index: int) -> List[VisualSegment]:
        """Send page image to SmolDocling endpoint and parse response."""
        import base64
        import json
        from urllib import request as urllib_request

        payload = json.dumps({
            "image": base64.b64encode(page_image).decode("ascii"),
            "page_index": page_index,
        }).encode("utf-8")

        req = urllib_request.Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib_request.urlopen(req, timeout=self.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        segments: List[VisualSegment] = []
        for region in data.get("regions", []):
            bbox_raw = region.get("bbox", [0, 0, 1, 1])
            bbox = (
                float(bbox_raw[0]),
                float(bbox_raw[1]),
                float(bbox_raw[2]),
                float(bbox_raw[3]),
            )
            seg_type = str(region.get("type", "body_text")).strip().lower()
            text = str(region.get("text", ""))
            conf = float(region.get("confidence", 0.5))

            info = compute_info_score(
                seg_type,
                text,
                bbox,
                conf,
                type_weights=self.type_weights,
            )

            segments.append(VisualSegment(
                page_index=page_index,
                bbox=bbox,
                segment_type=seg_type,
                text_content=text,
                info_score=info,
                confidence=conf,
            ))
        return segments


# --- Orchestrator ---

class VLMSegmenter:
    """
    Top-level VLM document segmenter.

    Renders each PDF page to an image, sends it to the VLM backend,
    and returns scored visual segments for the full document.
    """

    def __init__(
        self,
        backend: VLMSegmenterBackend,
        dpi: int = 150,
    ):
        self.backend = backend
        self.dpi = dpi

    def segment_document(self, pdf_path: Path) -> List[VisualSegment]:
        """Segment all pages of a PDF document."""
        page_images = self._render_pages(pdf_path)
        all_segments: List[VisualSegment] = []
        for page_index, image_bytes in enumerate(page_images):
            try:
                segments = self.backend.segment_page(image_bytes, page_index)
                all_segments.extend(segments)
            except Exception as exc:
                logger.warning(
                    "VLM segmentation failed for page %d of %s: %s",
                    page_index,
                    pdf_path,
                    exc,
                )
        return all_segments

    def _render_pages(self, pdf_path: Path) -> List[bytes]:
        """Render PDF pages to PNG images. Tries pymupdf first, falls back to pdf2image."""
        try:
            return self._render_with_pymupdf(pdf_path)
        except ImportError:
            return self._render_with_pdf2image(pdf_path)

    def _render_with_pymupdf(self, pdf_path: Path) -> List[bytes]:
        import fitz  # pymupdf

        doc = fitz.open(str(pdf_path))
        images: List[bytes] = []
        for page in doc:
            pix = page.get_pixmap(dpi=self.dpi)
            images.append(pix.tobytes("png"))
        doc.close()
        return images

    def _render_with_pdf2image(self, pdf_path: Path) -> List[bytes]:
        import io

        from pdf2image import convert_from_path

        pil_images = convert_from_path(str(pdf_path), dpi=self.dpi)
        images: List[bytes] = []
        for img in pil_images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images.append(buf.getvalue())
        return images
