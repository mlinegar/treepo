from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from treepo._research.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from treepo._research.core.preference_supervision import (
    PreferenceSupervisionMetadata,
    preference_supervision_metadata,
)

from .ids import stable_id
from .serialization import as_compact_str, to_jsonable


@dataclass
class PairwisePreference:
    """Backend-agnostic pairwise preference record."""

    example_id: str
    candidate_a: Any
    candidate_b: Any
    preferred: Literal["A", "B", "tie"]
    confidence: float

    input: Any = ""
    rubric: str = ""
    reference: Optional[float] = None
    score_a: Optional[float] = None
    score_b: Optional[float] = None
    reasoning: str = ""
    sampling: SamplingMetadata = field(
        default_factory=lambda: SamplingMetadata(unit_kind=ObservationUnitKind.PAIR)
    )
    preference_supervision: PreferenceSupervisionMetadata = field(
        default_factory=lambda: preference_supervision_metadata(
            application_name="ctreepo_opt_record"
        )
    )
    source_observation_ids: list[str] = field(default_factory=list)
    comparison_signal_value: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self) -> None:
        if not isinstance(self.sampling, SamplingMetadata):
            self.sampling = SamplingMetadata.from_dict(self.sampling)
        if self.sampling.unit_kind is None:
            self.sampling = self.sampling.with_updates(unit_kind=ObservationUnitKind.PAIR)
        if not isinstance(self.preference_supervision, PreferenceSupervisionMetadata):
            self.preference_supervision = PreferenceSupervisionMetadata.from_dict(
                self.preference_supervision
            )
        self.source_observation_ids = [str(value) for value in self.source_observation_ids]
        if self.comparison_signal_value is not None:
            self.comparison_signal_value = float(self.comparison_signal_value)

    def pair_id(self, *, n_chars: int = 16) -> str:
        payload = {
            "example_id": self.example_id,
            "rubric": self.rubric,
            "input": to_jsonable(self.input),
            "candidate_a": to_jsonable(self.candidate_a),
            "candidate_b": to_jsonable(self.candidate_b),
            "preferred": self.preferred,
        }
        return stable_id(payload, n_chars=n_chars)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id(),
            "example_id": self.example_id,
            "input": to_jsonable(self.input),
            "rubric": self.rubric,
            "candidate_a": to_jsonable(self.candidate_a),
            "candidate_b": to_jsonable(self.candidate_b),
            "preferred": self.preferred,
            "confidence": float(self.confidence),
            "reasoning": self.reasoning,
            "reference": self.reference,
            "score_a": self.score_a,
            "score_b": self.score_b,
            "sampling": self.sampling.to_dict(),
            "preference_supervision": self.preference_supervision.to_dict(),
            "source_observation_ids": list(self.source_observation_ids),
            "comparison_signal_value": self.comparison_signal_value,
            "sample_weight": self.sampling.ipw_weight(),
            "timestamp": self.timestamp,
        }

    def to_training_preference_pair(self) -> Any:
        """Convert to the repo's canonical binary comparison type (lazy import)."""
        from treepo._research.training.supervision import BinaryComparison

        reference_score = float(self.reference) if self.reference is not None else 0.0
        return BinaryComparison(
            pair_id=self.pair_id(),
            source_example_id=str(self.example_id),
            original_text=as_compact_str(self.input),
            rubric=str(self.rubric or ""),
            reference_score=reference_score,
            summary_a=as_compact_str(self.candidate_a),
            summary_b=as_compact_str(self.candidate_b),
            preferred=self.preferred,
            reasoning=str(self.reasoning or ""),
            confidence=float(self.confidence),
            sampling=self.sampling,
            preference_supervision=self.preference_supervision,
            source_observation_ids=list(self.source_observation_ids),
            comparison_signal_value=self.comparison_signal_value,
            score_estimate_a=self.score_a,
            score_estimate_b=self.score_b,
        )


__all__ = ["PairwisePreference", "SamplingMetadata"]
