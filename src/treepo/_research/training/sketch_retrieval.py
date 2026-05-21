"""
Cross-language retrieval via CTreePO root sketches.

Given a trained CTreePOModel and a set of indexed documents, retrieves the
most similar documents by cosine similarity in sketch space. Since the sketch
space is learned to capture political position from multilingual embeddings,
retrieval works across languages automatically.

Usage:
    index = SketchIndex()
    index.add("11320_199809", root_sketch_tensor, metadata={"rile": -3.5, ...})
    results = index.query(query_sketch, top_k=3)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    doc_id: str
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SketchIndex:
    """In-memory index of document root sketches for retrieval.

    Stores L2-normalized sketch vectors and retrieves by cosine similarity.
    Can be persisted to/from JSON for cross-session reuse.
    """

    def __init__(self):
        self._ids: List[str] = []
        self._vectors: List[np.ndarray] = []
        self._metadata: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._ids)

    def add(
        self,
        doc_id: str,
        sketch: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a document sketch to the index.

        Args:
            doc_id: Unique document identifier.
            sketch: Root sketch vector (torch.Tensor, np.ndarray, or list).
            metadata: Optional metadata (rile, party, country, etc.).
        """
        if torch is not None and isinstance(sketch, torch.Tensor):
            vec = sketch.detach().cpu().numpy().astype(np.float32)
        elif isinstance(sketch, np.ndarray):
            vec = sketch.astype(np.float32)
        else:
            vec = np.array(sketch, dtype=np.float32)

        # L2 normalize
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec = vec / norm

        self._ids.append(doc_id)
        self._vectors.append(vec)
        self._metadata.append(metadata or {})

    def query(
        self,
        sketch: Any,
        top_k: int = 5,
        exclude_ids: Optional[set] = None,
    ) -> List[RetrievalResult]:
        """Find the top_k most similar documents by sketch cosine similarity.

        Args:
            sketch: Query sketch vector.
            top_k: Number of results to return.
            exclude_ids: Set of doc_ids to exclude from results.

        Returns:
            List of RetrievalResult sorted by descending similarity.
        """
        if not self._vectors:
            return []

        if torch is not None and isinstance(sketch, torch.Tensor):
            q = sketch.detach().cpu().numpy().astype(np.float32)
        elif isinstance(sketch, np.ndarray):
            q = sketch.astype(np.float32)
        else:
            q = np.array(sketch, dtype=np.float32)

        # L2 normalize query
        norm = float(np.linalg.norm(q))
        if norm > 1e-12:
            q = q / norm

        # Compute similarities (dot product on normalized vectors = cosine)
        mat = np.stack(self._vectors, axis=0)
        sims = mat @ q

        # Sort by descending similarity
        order = np.argsort(-sims)

        results = []
        for idx in order:
            doc_id = self._ids[idx]
            if exclude_ids and doc_id in exclude_ids:
                continue
            results.append(RetrievalResult(
                doc_id=doc_id,
                similarity=float(sims[idx]),
                metadata=self._metadata[idx],
            ))
            if len(results) >= top_k:
                break

        return results

    def save(self, path: Path) -> None:
        """Save index to JSON file."""
        path = Path(path)
        entries = []
        for doc_id, vec, meta in zip(self._ids, self._vectors, self._metadata):
            entries.append({
                "doc_id": doc_id,
                "sketch": vec.tolist(),
                "metadata": meta,
            })
        path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        logger.info("Saved sketch index (%d docs) to %s", len(entries), path)

    def load(self, path: Path) -> int:
        """Load index from JSON file. Returns number of entries loaded."""
        path = Path(path)
        entries = json.loads(path.read_text(encoding="utf-8"))
        count = 0
        for entry in entries:
            self.add(
                doc_id=entry["doc_id"],
                sketch=entry["sketch"],
                metadata=entry.get("metadata", {}),
            )
            count += 1
        logger.info("Loaded %d entries from %s", count, path)
        return count

    def get_all_similarities(self) -> np.ndarray:
        """Compute full pairwise similarity matrix."""
        if not self._vectors:
            return np.array([])
        mat = np.stack(self._vectors, axis=0)
        return mat @ mat.T
