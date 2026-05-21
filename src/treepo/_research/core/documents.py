"""
Document-level data models.

These models are dataset-agnostic and keep document metadata in a structured form.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocumentSample:
    """
    A single dataset-agnostic sample.

    `text` remains the common path for text-only pipelines.
    Additional optional fields (`pages`, `items`, `segments`) keep axis-window
    adapters modality-ready without forcing a new sample type.
    """
    doc_id: str
    text: str = ""
    reference_score: Optional[float] = None
    modality: str = "text"
    pages: Optional[List[str]] = None
    items: Optional[List[Any]] = None
    segments: Optional[List[Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like access for metadata."""
        return self.metadata.get(key, default)


@dataclass
class DocumentResult:
    """Result of processing a single document."""
    doc_id: str
    original_content: Optional[str] = None
    reference_score: Optional[float] = None
    estimated_score: Optional[float] = None
    baseline_score: Optional[float] = None

    final_summary: str = ""
    summary_length: int = 0
    original_length: int = 0
    compression_ratio: float = 1.0

    tree_height: Optional[int] = None
    tree_nodes: Optional[int] = None
    tree_leaves: Optional[int] = None

    chunks: list = field(default_factory=list)
    leaf_summaries: list = field(default_factory=list)
    level_history: list = field(default_factory=list)
    processing_time: float = 0.0

    error: Optional[str] = None
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Task-specific fields like 'left_indicators' should be stored in metadata

    @property
    def prediction_error(self) -> Optional[float]:
        if self.estimated_score is None or self.reference_score is None:
            return None
        return abs(self.estimated_score - self.reference_score)

    @property
    def baseline_error(self) -> Optional[float]:
        if self.baseline_score is None or self.reference_score is None:
            return None
        return abs(self.baseline_score - self.reference_score)
