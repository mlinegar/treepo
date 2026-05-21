"""
PDF dataset plugin.

Loads PDF files from a path (single file or directory) and emits page-aligned
DocumentSample objects with page char-range metadata for adaptive feedback.
"""

from __future__ import annotations

import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

from treepo._research.core.documents import DocumentSample
from treepo._research.parsers import PDFTextParser
from treepo._research.parsers.vlm_segmenter import VLMSegmenter
from treepo._research.preprocessing.visual_feedback import (
    map_segments_to_char_ranges,
    segments_to_feedback_signals,
)

from .base import DatasetInfo, register_dataset

logger = logging.getLogger(__name__)


def _normalize_ranges(ranges: List[tuple[int, int]]) -> List[List[int]]:
    """Serialize tuple ranges as JSON-safe integer lists."""
    out: List[List[int]] = []
    for start, end in ranges:
        out.append([int(start), int(end)])
    return out


def _has_routable_parser_hints(metadata: Dict[str, Any]) -> bool:
    """True if parser_feedback includes at least one actionable hint."""
    parser_feedback = metadata.get("parser_feedback")
    if not isinstance(parser_feedback, dict):
        return False
    hints = parser_feedback.get("axis_hints")
    if not isinstance(hints, list):
        return False
    for hint in hints:
        if not isinstance(hint, dict):
            continue
        recommended = hint.get("recommended_processors")
        if isinstance(recommended, list) and any(str(item or "").strip() for item in recommended):
            return True
        action = str(hint.get("action") or "").strip().lower()
        if action and action != "none":
            return True
    return False


@register_dataset("pdf")
class PDFDataset:
    """Dataset plugin for page-oriented PDF corpora."""

    def __init__(
        self,
        path: Optional[str] = None,
        recursive: bool = True,
        require_text: bool = True,
        parser_backends: Optional[List[str]] = None,
        vlm_segmenter: Optional[VLMSegmenter] = None,
    ):
        self.path = Path(path) if path else None
        self.recursive = bool(recursive)
        self.require_text = bool(require_text)
        self.parser_backends = list(parser_backends) if parser_backends else ["pypdf", "pymupdf"]
        self.vlm_segmenter = vlm_segmenter
        self._parser: Optional[PDFTextParser] = None

    @property
    def name(self) -> str:
        return "pdf"

    def get_info(self) -> DatasetInfo:
        return DatasetInfo(
            name=self.name,
            description="PDF dataset with page-aligned text extraction",
            supports_reference_scores=False,
        )

    def _get_parser(self) -> PDFTextParser:
        if self._parser is None:
            self._parser = PDFTextParser(
                backends=self.parser_backends,
                fallback_to_empty=True,
            )
        return self._parser

    def _resolve_files(self, path: Path) -> List[Path]:
        if path.is_file():
            if path.suffix.lower() != ".pdf":
                raise ValueError(f"Expected a .pdf file, got: {path}")
            return [path]
        if not path.is_dir():
            raise FileNotFoundError(f"PDF dataset path not found: {path}")

        pattern = "**/*.pdf" if self.recursive else "*.pdf"
        files = sorted(path.glob(pattern))
        return [file for file in files if file.is_file()]

    def load_samples(
        self,
        path: Optional[str] = None,
        limit: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        parser: Optional[PDFTextParser] = None,
        **kwargs: Any,
    ) -> List[DocumentSample]:
        data_path = Path(path) if path else self.path
        if data_path is None:
            raise ValueError("PDF dataset requires a path to a PDF file or directory")

        pdf_files = self._resolve_files(data_path)
        if shuffle:
            random.seed(seed)
            random.shuffle(pdf_files)
        if limit is not None:
            pdf_files = pdf_files[: max(0, int(limit))]

        parser_obj = parser or self._get_parser()
        samples: List[DocumentSample] = []
        for index, pdf_path in enumerate(pdf_files):
            try:
                parsed = parser_obj.parse_file(pdf_path)
            except Exception as exc:
                logger.warning("PDF parse failed for %s: %s", pdf_path, exc)
                continue

            metadata: Dict[str, Any] = dict(parsed.metadata or {})
            if self.require_text and not parsed.text.strip() and not _has_routable_parser_hints(metadata):
                continue

            if data_path.is_dir():
                doc_id = pdf_path.relative_to(data_path).with_suffix("").as_posix()
            else:
                doc_id = pdf_path.stem

            page_ranges = _normalize_ranges(parsed.page_char_ranges)
            metadata.update(
                {
                    "source_path": str(pdf_path),
                    "parser_backend": parsed.backend,
                    "page_count": len(parsed.pages),
                    "page_char_ranges": page_ranges,
                    "axis_char_ranges": {"page": page_ranges},
                }
            )

            # VLM visual segmentation enrichment (optional).
            if self.vlm_segmenter is not None:
                try:
                    visual_segments = self.vlm_segmenter.segment_document(pdf_path)
                    visual_segments = map_segments_to_char_ranges(
                        visual_segments,
                        list(parsed.pages),
                        parsed.page_char_ranges,
                    )
                    feedback_signals = segments_to_feedback_signals(visual_segments)
                    metadata["visual_segments"] = [
                        {
                            "page_index": seg.page_index,
                            "bbox": list(seg.bbox),
                            "segment_type": seg.segment_type,
                            "info_score": seg.info_score,
                            "confidence": seg.confidence,
                            "char_start": seg.char_start,
                            "char_end": seg.char_end,
                        }
                        for seg in visual_segments
                    ]
                    metadata["visual_feedback_signals"] = feedback_signals
                except Exception as exc:
                    logger.warning(
                        "VLM segmentation failed for %s: %s", pdf_path, exc,
                    )

            samples.append(
                DocumentSample(
                    doc_id=str(doc_id or f"pdf_{index}"),
                    text=parsed.text,
                    modality="text",
                    pages=list(parsed.pages),
                    reference_score=None,
                    metadata=metadata,
                )
            )
        return samples
