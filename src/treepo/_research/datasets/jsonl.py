"""
Generic JSONL dataset plugin.

Each line is a JSON object with required "text" and optional "id"/"doc_id".
Optional numeric score fields:
- reference_score
- score
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from treepo._research.core.documents import DocumentSample
from .base import DatasetInfo, register_dataset


@register_dataset("jsonl")
class JSONLDataset:
    """JSONL dataset plugin for generic document collections."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None

    @property
    def name(self) -> str:
        return "jsonl"

    def get_info(self) -> DatasetInfo:
        return DatasetInfo(
            name=self.name,
            description="Generic JSONL dataset (id, text, optional reference_score)",
            supports_reference_scores=True,
        )

    def load_samples(
        self,
        path: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> List[DocumentSample]:
        data_path = Path(path) if path else self.path
        if data_path is None:
            raise ValueError("JSONL dataset requires a path")

        samples: List[DocumentSample] = []
        with open(data_path, "r") as handle:
            for idx, line in enumerate(handle):
                if limit is not None and len(samples) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = record.get("text") or record.get("content")
                if not text:
                    continue

                doc_id = (
                    record.get("doc_id")
                    or record.get("id")
                    or f"row_{idx}"
                )
                reference_score = record.get("reference_score")
                if reference_score is None:
                    reference_score = record.get("score")
                if reference_score is not None:
                    try:
                        reference_score = float(reference_score)
                    except (TypeError, ValueError):
                        reference_score = None

                metadata = record.get("metadata", {})
                # Preserve any extra fields
                for key, value in record.items():
                    if key in {"text", "content", "doc_id", "id", "reference_score", "score", "metadata"}:
                        continue
                    metadata.setdefault(key, value)

                pages = record.get("pages")
                if isinstance(pages, list):
                    pages = [str(page or "") for page in pages]
                else:
                    pages = None

                items = record.get("items")
                if not isinstance(items, list):
                    items = None

                segments = record.get("segments")
                if not isinstance(segments, list):
                    segments = None

                modality = str(record.get("modality") or "text")

                samples.append(
                    DocumentSample(
                        doc_id=str(doc_id),
                        text=text,
                        reference_score=reference_score,
                        modality=modality,
                        pages=pages,
                        items=items,
                        segments=segments,
                        metadata=metadata,
                    )
                )

        return samples
