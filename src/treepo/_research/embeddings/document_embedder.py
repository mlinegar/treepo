"""
Long-document embedding utilities.

We build *document vectors* from multilingual embeddings by:
  - slicing text into deterministic windows (char axis),
  - embedding each window,
  - pooling window embeddings into a single vector,
  - optionally embedding document metadata and late-fusing it.

This avoids "embed the entire document" failure modes due to context limits,
while still letting us represent the full document.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from treepo._research.core.doc_metadata import DocMetadata, format_doc_meta_embedding_text
from treepo._research.preprocessing.adaptive_windows import AxisWindow, uniform_axis_windows


class EmbeddingClient(Protocol):
    """Minimal protocol for embedding endpoints."""

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover - protocol
        ...


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vec) + 1e-12)
    return vec / denom


@dataclass(frozen=True)
class DocumentEmbeddingConfig:
    window_chars: int = 6000
    overlap_chars: int = 0
    max_windows: int = 8
    pooling: str = "mean"  # only mean supported for now
    l2_normalize: bool = True

    embed_metadata: bool = True
    text_weight: float = 1.0
    meta_weight: float = 0.25


@dataclass(frozen=True)
class DocumentEmbeddingResult:
    windows: List[AxisWindow]
    window_texts: List[str]
    window_embeddings: List[List[float]]

    text_vector: Optional[np.ndarray]
    meta_text: Optional[str]
    meta_vector: Optional[np.ndarray]
    combined_vector: Optional[np.ndarray]


class DocumentEmbedder:
    """Embeds long documents via deterministic windowing + pooling."""

    def __init__(self, embedding_client: EmbeddingClient, config: Optional[DocumentEmbeddingConfig] = None):
        self.client = embedding_client
        self.config = config or DocumentEmbeddingConfig()

    def build_windows(self, text: str) -> List[AxisWindow]:
        raw = str(text or "")
        total = len(raw)
        if total <= 0:
            return []

        window_chars = int(self.config.window_chars)
        overlap = int(self.config.overlap_chars)

        if window_chars <= 0:
            return [AxisWindow(start=0, end=total, unit="char")]

        windows = uniform_axis_windows(
            total,
            window_size=max(1, window_chars),
            overlap=max(0, min(overlap, max(0, window_chars - 1))),
            unit="char",
        )
        if not windows:
            return [AxisWindow(start=0, end=total, unit="char")]

        max_windows = int(self.config.max_windows)
        if max_windows > 0 and len(windows) > max_windows:
            stride = max(1, int(math.ceil(len(windows) / float(max_windows))))
            reduced = list(windows[::stride])
            # Ensure tail coverage (important for long docs with summaries/conclusions at end).
            if reduced and int(reduced[-1].end) < total:
                start = max(0, total - window_chars)
                tail = AxisWindow(start=start, end=total, unit="char")
                if len(reduced) >= max_windows:
                    reduced[-1] = tail
                else:
                    reduced.append(tail)
            # Dedup and sort deterministically.
            seen = set()
            out: List[AxisWindow] = []
            for w in sorted(reduced, key=lambda x: (int(x.start), int(x.end))):
                key = (int(w.start), int(w.end), str(w.unit))
                if key in seen:
                    continue
                seen.add(key)
                out.append(w)
            windows = out

        return list(windows)

    def embed_text(self, text: str) -> Tuple[List[AxisWindow], List[str], List[List[float]]]:
        raw = str(text or "")
        windows = self.build_windows(raw)
        window_texts = [raw[int(w.start) : int(w.end)] for w in windows]
        embeddings = self.client.embed_texts(window_texts) if window_texts else []
        return windows, window_texts, embeddings

    def pool_text_vector(self, window_embeddings: Sequence[Sequence[float]]) -> Optional[np.ndarray]:
        if not window_embeddings:
            return None
        pooling = str(self.config.pooling or "mean").strip().lower()
        mat = np.asarray(window_embeddings, dtype=np.float32)
        if mat.ndim != 2 or mat.shape[0] <= 0 or mat.shape[1] <= 0:
            return None
        if pooling != "mean":
            raise ValueError(f"Unsupported pooling: {pooling}")
        vec = mat.mean(axis=0)
        if self.config.l2_normalize:
            vec = _l2_normalize(vec)
        return vec.astype(np.float32, copy=False)

    def embed_metadata_vector(self, meta: Optional[DocMetadata]) -> tuple[Optional[str], Optional[np.ndarray]]:
        if not self.config.embed_metadata or meta is None:
            return None, None
        meta_text = format_doc_meta_embedding_text(meta)
        vec = self.client.embed_texts([meta_text])[0]
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] <= 0:
            return meta_text, None
        if self.config.l2_normalize:
            arr = _l2_normalize(arr)
        return meta_text, arr.astype(np.float32, copy=False)

    def combine_vectors(
        self,
        *,
        text_vector: Optional[np.ndarray],
        meta_vector: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if text_vector is None and meta_vector is None:
            return None
        if text_vector is None:
            return meta_vector
        if meta_vector is None:
            return text_vector
        combined = (float(self.config.text_weight) * text_vector) + (float(self.config.meta_weight) * meta_vector)
        if self.config.l2_normalize:
            combined = _l2_normalize(combined)
        return combined.astype(np.float32, copy=False)

    def embed_document(self, text: str, *, meta: Optional[DocMetadata] = None) -> DocumentEmbeddingResult:
        windows, window_texts, window_embeddings = self.embed_text(text)
        text_vec = self.pool_text_vector(window_embeddings)
        meta_text, meta_vec = self.embed_metadata_vector(meta)
        combined = self.combine_vectors(text_vector=text_vec, meta_vector=meta_vec)
        return DocumentEmbeddingResult(
            windows=windows,
            window_texts=window_texts,
            window_embeddings=list(window_embeddings),
            text_vector=text_vec,
            meta_text=meta_text,
            meta_vector=meta_vec,
            combined_vector=combined,
        )
