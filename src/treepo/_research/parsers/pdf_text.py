"""
PDF text parsing utilities.

The parser extracts page-aligned text and computes page->char span metadata so
downstream adaptive feedback can map page windows back to character spans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def _normalize_page_text(text: Any) -> str:
    """Normalize extracted page text into a stable plain-text form."""
    normalized = str(text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    return normalized.strip()


def _build_page_char_ranges(
    pages: Sequence[str],
    *,
    page_joiner: str,
) -> Tuple[str, List[Tuple[int, int]]]:
    """Join pages and compute char ranges for each page in joined text."""
    normalized_pages = [str(page or "") for page in pages]
    ranges: List[Tuple[int, int]] = []
    cursor = 0
    joiner = str(page_joiner or "")

    for idx, page_text in enumerate(normalized_pages):
        start = cursor
        cursor += len(page_text)
        end = cursor
        ranges.append((start, end))
        if idx + 1 < len(normalized_pages):
            cursor += len(joiner)

    return joiner.join(normalized_pages), ranges


def _non_whitespace_char_count(text: str) -> int:
    return sum(1 for char in str(text or "") if not char.isspace())


def _normalize_page_image_counts(
    image_counts: Sequence[int] | None,
    *,
    expected_len: int,
) -> List[int]:
    """Normalize image counts to a fixed-length non-negative integer list."""
    normalized: List[int] = []
    if isinstance(image_counts, Sequence):
        for count in image_counts:
            try:
                normalized.append(max(0, int(count)))
            except (TypeError, ValueError):
                normalized.append(0)
    if len(normalized) < expected_len:
        normalized.extend([0] * (expected_len - len(normalized)))
    elif len(normalized) > expected_len:
        normalized = normalized[:expected_len]
    return normalized


def _default_page_asset(source_path: Path, page_index: int, image_count: int) -> Dict[str, Any]:
    page_number = int(page_index) + 1
    return {
        "page_index": int(page_index),
        "page_number": page_number,
        "page_uri": f"pdf://{source_path.resolve().as_posix()}#page={page_number}",
        "backend_page_ref": f"page:{page_number}",
        "image_count": int(max(0, image_count)),
        "image_refs": [],
    }


def _normalize_page_assets(
    raw_assets: Sequence[Any] | None,
    *,
    expected_len: int,
    source_path: Path,
    image_counts: Sequence[int] | None = None,
) -> List[Dict[str, Any]]:
    """Normalize raw page asset metadata to one record per page."""
    normalized_counts = _normalize_page_image_counts(image_counts, expected_len=expected_len)
    base_assets = [
        _default_page_asset(source_path, idx, normalized_counts[idx] if idx < len(normalized_counts) else 0)
        for idx in range(expected_len)
    ]

    if not isinstance(raw_assets, Sequence) or isinstance(raw_assets, (str, bytes, bytearray)):
        return base_assets

    for idx, entry in enumerate(raw_assets):
        if idx >= len(base_assets):
            break
        if not isinstance(entry, Mapping):
            continue
        merged = dict(base_assets[idx])
        merged.update(dict(entry))

        image_refs = merged.get("image_refs")
        if isinstance(image_refs, Sequence) and not isinstance(image_refs, (str, bytes, bytearray)):
            merged["image_refs"] = [str(ref or "") for ref in image_refs if str(ref or "")]
        else:
            merged["image_refs"] = []

        try:
            merged["image_count"] = int(max(0, int(merged.get("image_count") or 0)))
        except (TypeError, ValueError):
            merged["image_count"] = 0

        merged["page_index"] = int(idx)
        merged["page_number"] = int(idx + 1)
        merged["page_uri"] = str(merged.get("page_uri") or base_assets[idx]["page_uri"])
        merged["backend_page_ref"] = str(
            merged.get("backend_page_ref") or base_assets[idx]["backend_page_ref"]
        )
        base_assets[idx] = merged

    return base_assets


def _build_page_parser_feedback(
    pages: Sequence[str],
    page_char_ranges: Sequence[Tuple[int, int]],
    *,
    page_image_counts: Sequence[int] | None = None,
    page_assets: Sequence[Mapping[str, Any]] | None = None,
    min_text_chars_for_visual_support: int = 96,
) -> Dict[str, Any]:
    """
    Build parser-level feedback hints over the page axis.

    This captures parser uncertainty and routing recommendations (OCR/VLM/vision
    embedding) rather than using text-density as a proxy for semantic relevance.
    """
    page_text_char_counts = [_non_whitespace_char_count(page) for page in pages]
    unit_count = len(page_text_char_counts)
    page_image_counts_norm = _normalize_page_image_counts(page_image_counts, expected_len=unit_count)

    normalized_assets: List[Dict[str, Any]] = []
    if isinstance(page_assets, Sequence) and not isinstance(page_assets, (str, bytes, bytearray)):
        for idx, asset in enumerate(page_assets):
            if idx >= unit_count:
                break
            normalized_assets.append(dict(asset) if isinstance(asset, Mapping) else {})
    if len(normalized_assets) < unit_count:
        normalized_assets.extend({} for _ in range(unit_count - len(normalized_assets)))

    nonempty_text_pages = sum(1 for count in page_text_char_counts if count > 0)
    pages_with_images = sum(1 for count in page_image_counts_norm if count > 0)
    total_text_chars = int(sum(page_text_char_counts))
    total_images = int(sum(page_image_counts_norm))

    axis_hints: List[Dict[str, Any]] = []
    pages_needing_ocr = 0
    pages_needing_vlm = 0
    pages_needing_visual_embedding = 0

    for page_index, text_chars in enumerate(page_text_char_counts):
        image_count = int(page_image_counts_norm[page_index])
        has_text = text_chars > 0
        has_images = image_count > 0
        extraction_is_empty = not has_text

        needs_ocr = extraction_is_empty and has_images
        extraction_uncertain = extraction_is_empty and not has_images
        visual_content_support = has_images and text_chars < int(max(1, min_text_chars_for_visual_support))

        if not (needs_ocr or extraction_uncertain or visual_content_support):
            continue

        char_start = 0
        char_end = 0
        if 0 <= page_index < len(page_char_ranges):
            char_start = int(page_char_ranges[page_index][0])
            char_end = int(page_char_ranges[page_index][1])

        page_asset = normalized_assets[page_index] if page_index < len(normalized_assets) else {}
        page_asset_ref = str(page_asset.get("page_uri") or "")
        page_image_refs = page_asset.get("image_refs")
        if isinstance(page_image_refs, Sequence) and not isinstance(page_image_refs, (str, bytes, bytearray)):
            page_image_refs = [str(ref or "") for ref in page_image_refs if str(ref or "")]
        else:
            page_image_refs = []

        recommended_processors: List[str] = []
        route_action = "none"
        source = "parser:pdf_extraction_hint"
        noise_probability = 0.25
        confidence = 0.60

        if needs_ocr:
            pages_needing_ocr += 1
            pages_needing_visual_embedding += 1
            recommended_processors = ["ocr", "vision_embedding"]
            route_action = "ocr_first_then_vision_embedding"
            source = "parser:pdf_needs_ocr"
            noise_probability = 0.95
            confidence = 0.92
        elif extraction_uncertain:
            pages_needing_vlm += 1
            recommended_processors = ["vlm_parse"]
            route_action = "vlm_parse"
            source = "parser:pdf_extraction_uncertain"
            noise_probability = 0.70
            confidence = 0.55
        elif visual_content_support:
            pages_needing_visual_embedding += 1
            recommended_processors = ["vision_embedding"]
            route_action = "augment_with_vision_embedding"
            source = "parser:pdf_visual_content"
            noise_probability = 0.55 if text_chars < 32 else 0.35
            confidence = 0.78 if text_chars < 32 else 0.68

        axis_hints.append(
            {
                "axis_unit": "page",
                "start": int(page_index),
                "end": int(page_index + 1),
                "char_start": char_start,
                "char_end": char_end,
                "text_chars": int(text_chars),
                "image_count": int(image_count),
                "page_asset_ref": page_asset_ref or None,
                "page_image_refs": page_image_refs,
                # This hint is about parser confidence/routing, not semantic low-info.
                "low_info_probability": 0.0,
                "noise_probability": float(max(0.0, min(1.0, noise_probability))),
                "confidence": confidence,
                "source": source,
                "action": route_action,
                "recommended_processors": recommended_processors,
            }
        )

    nonempty_text_fraction = float(nonempty_text_pages) / float(unit_count) if unit_count > 0 else 0.0
    page_with_images_fraction = float(pages_with_images) / float(unit_count) if unit_count > 0 else 0.0
    return {
        "schema_version": 3,
        "parser": "pdf_text",
        "strategy": "extraction_quality_routing",
        "axes": {
            "page": {
                "unit_count": int(unit_count),
                "text_nonempty_page_count": int(nonempty_text_pages),
                "text_nonempty_fraction": nonempty_text_fraction,
                "image_page_count": int(pages_with_images),
                "image_page_fraction": page_with_images_fraction,
                "total_text_chars": total_text_chars,
                "total_images": total_images,
            }
        },
        "routing_summary": {
            "pages_needing_ocr": int(pages_needing_ocr),
            "pages_needing_vlm_parse": int(pages_needing_vlm),
            "pages_recommended_for_vision_embedding": int(pages_needing_visual_embedding),
        },
        "axis_hints": axis_hints,
    }


@dataclass
class ParsedPDFDocument:
    """Parsed PDF output with page-aligned spans."""

    text: str
    pages: List[str]
    page_char_ranges: List[Tuple[int, int]]
    backend: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PDFTextParser:
    """
    Parse PDFs into page text with backend fallback.

    Backends are attempted in order. Supported backend names:
    - `pypdf`
    - `pymupdf`
    """

    def __init__(
        self,
        *,
        backends: Sequence[str] = ("pypdf", "pymupdf"),
        page_joiner: str = "\n\n",
        fallback_to_empty: bool = False,
    ):
        self.backends = [str(name or "").strip().lower() for name in backends if str(name or "").strip()]
        if not self.backends:
            self.backends = ["pypdf", "pymupdf"]
        self.page_joiner = str(page_joiner)
        self.fallback_to_empty = bool(fallback_to_empty)

    def parse_file(self, path: Path | str) -> ParsedPDFDocument:
        """Parse one PDF file into page text and char-range metadata."""
        pdf_path = Path(path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        errors: List[str] = []
        for backend in self.backends:
            try:
                extracted = self._extract_pages(pdf_path, backend=backend)
            except Exception as exc:
                errors.append(f"{backend}: {exc}")
                continue

            pages, page_image_counts, page_assets = self._normalize_extracted_pages(
                extracted,
                source_path=pdf_path,
            )
            normalized_pages = [_normalize_page_text(page) for page in pages]
            text, page_ranges = _build_page_char_ranges(
                normalized_pages,
                page_joiner=self.page_joiner,
            )
            parser_feedback = _build_page_parser_feedback(
                normalized_pages,
                page_ranges,
                page_image_counts=page_image_counts,
                page_assets=page_assets,
            )
            if text.strip() or parser_feedback.get("axis_hints"):
                return ParsedPDFDocument(
                    text=text,
                    pages=normalized_pages,
                    page_char_ranges=page_ranges,
                    backend=backend,
                    metadata={
                        "source_path": str(pdf_path),
                        "page_count": len(normalized_pages),
                        "page_joiner": self.page_joiner,
                        "page_assets": page_assets,
                        "parser_feedback": parser_feedback,
                    },
                )
            errors.append(f"{backend}: extracted empty text and no routable hints")

        if self.fallback_to_empty:
            return ParsedPDFDocument(
                text="",
                pages=[],
                page_char_ranges=[],
                backend=self.backends[0] if self.backends else "unknown",
                metadata={
                    "source_path": str(pdf_path),
                    "page_count": 0,
                    "page_joiner": self.page_joiner,
                    "page_assets": [],
                    "parse_errors": errors,
                },
            )

        attempted = ", ".join(self.backends)
        joined_errors = "; ".join(errors) if errors else "no parser backend succeeded"
        raise RuntimeError(
            f"Failed to parse PDF '{pdf_path}' using backends [{attempted}]: {joined_errors}"
        )

    def _extract_pages(self, path: Path, *, backend: str) -> Any:
        normalized = str(backend or "").strip().lower()
        if normalized == "pypdf":
            return self._extract_pages_with_pypdf(path)
        if normalized == "pymupdf":
            return self._extract_pages_with_pymupdf(path)
        raise ValueError(f"Unsupported PDF parser backend '{backend}'")

    @staticmethod
    def _normalize_extracted_pages(
        extracted: Any,
        *,
        source_path: Path,
    ) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
        """
        Normalize backend output to (pages, page_image_counts, page_assets).

        Backwards compatible with old backends/tests that return only pages.
        """
        if isinstance(extracted, tuple) and len(extracted) >= 3:
            raw_pages = extracted[0]
            raw_image_counts = extracted[1]
            raw_assets = extracted[2]
        elif isinstance(extracted, tuple) and len(extracted) >= 2:
            raw_pages = extracted[0]
            raw_image_counts = extracted[1]
            raw_assets = []
        else:
            raw_pages = extracted
            raw_image_counts = []
            raw_assets = []

        pages: List[str] = []
        if isinstance(raw_pages, Sequence) and not isinstance(raw_pages, (str, bytes, bytearray)):
            pages = [str(page or "") for page in raw_pages]
        elif raw_pages is not None:
            pages = [str(raw_pages)]

        image_counts = _normalize_page_image_counts(raw_image_counts, expected_len=len(pages))
        page_assets = _normalize_page_assets(
            raw_assets,
            expected_len=len(pages),
            source_path=source_path,
            image_counts=image_counts,
        )
        return pages, image_counts, page_assets

    @staticmethod
    def _count_images_in_pypdf_page(page: Any) -> int:
        """Best-effort image count for a PyPDF page object."""
        try:
            images = getattr(page, "images", None)
            if images is not None:
                return max(0, int(len(images)))
        except Exception:
            pass

        try:
            resources = page.get("/Resources")
            if resources is None:
                return 0
            xobjects = resources.get("/XObject")
            if xobjects is None:
                return 0
            count = 0
            for _, obj in xobjects.items():
                target = obj.get_object() if hasattr(obj, "get_object") else obj
                subtype = target.get("/Subtype") if hasattr(target, "get") else None
                if str(subtype) == "/Image":
                    count += 1
            return max(0, int(count))
        except Exception:
            return 0

    @staticmethod
    def _extract_pypdf_image_refs(page: Any) -> List[str]:
        """Best-effort extraction of stable image references from a PyPDF page."""
        refs: List[str] = []

        try:
            images = getattr(page, "images", None)
            if images is not None:
                for idx, image_obj in enumerate(images):
                    name = getattr(image_obj, "name", None) or getattr(image_obj, "id", None)
                    if name is None and isinstance(image_obj, Mapping):
                        name = image_obj.get("name") or image_obj.get("id")
                    refs.append(str(name or f"image:{idx}"))
        except Exception:
            pass

        if refs:
            deduped: List[str] = []
            for ref in refs:
                if ref not in deduped:
                    deduped.append(ref)
            return deduped

        try:
            resources = page.get("/Resources")
            if resources is None:
                return []
            xobjects = resources.get("/XObject")
            if xobjects is None:
                return []
            for name, obj in xobjects.items():
                target = obj.get_object() if hasattr(obj, "get_object") else obj
                subtype = target.get("/Subtype") if hasattr(target, "get") else None
                if str(subtype) == "/Image":
                    refs.append(str(name))
        except Exception:
            return []

        deduped = []
        for ref in refs:
            if ref not in deduped:
                deduped.append(ref)
        return deduped

    @staticmethod
    def _extract_pages_with_pypdf(path: Path) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
        try:
            from pypdf import PdfReader
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "pypdf backend unavailable. Install with: pip install pypdf"
            ) from exc

        reader = PdfReader(str(path))
        pages: List[str] = []
        image_counts: List[int] = []
        page_assets: List[Dict[str, Any]] = []

        for idx, page in enumerate(reader.pages):
            extracted = page.extract_text() or ""
            pages.append(str(extracted))

            image_refs = PDFTextParser._extract_pypdf_image_refs(page)
            image_count = len(image_refs)
            if image_count <= 0:
                image_count = PDFTextParser._count_images_in_pypdf_page(page)
            image_counts.append(int(max(0, image_count)))

            page_number = idx + 1
            page_assets.append(
                {
                    "page_index": idx,
                    "page_number": page_number,
                    "page_uri": f"pdf://{path.resolve().as_posix()}#page={page_number}",
                    "backend_page_ref": f"pypdf:page:{page_number}",
                    "image_count": int(max(0, image_count)),
                    "image_refs": image_refs,
                }
            )

        return pages, image_counts, page_assets

    @staticmethod
    def _extract_pages_with_pymupdf(path: Path) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
        try:
            import fitz
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "pymupdf backend unavailable. Install with: pip install pymupdf"
            ) from exc

        pages: List[str] = []
        image_counts: List[int] = []
        page_assets: List[Dict[str, Any]] = []

        doc = fitz.open(str(path))
        try:
            for page in doc:
                pages.append(str(page.get_text("text") or ""))

                image_refs: List[str] = []
                try:
                    image_info = page.get_images(full=False)
                    for info in image_info:
                        if isinstance(info, Sequence) and len(info) > 0:
                            xref = info[0]
                            image_refs.append(f"xref:{int(xref)}")
                except Exception:
                    image_refs = []

                image_count = len(image_refs)
                image_counts.append(int(max(0, image_count)))

                page_number = int(getattr(page, "number", len(page_assets))) + 1
                page_assets.append(
                    {
                        "page_index": page_number - 1,
                        "page_number": page_number,
                        "page_uri": f"pdf://{path.resolve().as_posix()}#page={page_number}",
                        "backend_page_ref": f"pymupdf:page:{page_number}",
                        "image_count": int(max(0, image_count)),
                        "image_refs": image_refs,
                    }
                )
        finally:
            doc.close()

        return pages, image_counts, page_assets
