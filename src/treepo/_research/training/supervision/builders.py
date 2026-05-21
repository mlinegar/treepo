"""Shared builders for dense numeric supervision datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from treepo._research.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from treepo._research.core.supervision_metadata import judgment_supervision_metadata
from treepo._research.training.supervision.contracts import (
    REPRESENTATION_DENSE_FEATURE_VECTOR,
    TARGET_SCALAR,
    TARGET_VECTOR,
    dense_supervision_metadata,
)
from treepo._research.training.supervision.types import ResponseJudgment, SupervisionDataset


@dataclass(frozen=True)
class DenseSupervisionExample:
    """One dense numeric training example for the supervision surface."""

    example_id: str
    features: Sequence[float]
    scalar_target: Optional[float] = None
    vector_target: Optional[Sequence[float]] = None
    original_text: Optional[str] = None
    rubric: str = "Predict a dense numeric target from a shared feature representation."
    response: str = "dense_candidate"
    response_id: Optional[str] = None
    unit_kind: ObservationUnitKind = ObservationUnitKind.DOCUMENT
    reference_score: float = 0.0
    source_doc_id: Optional[str] = None
    source_observation_ids: Sequence[str] = field(default_factory=tuple)
    truth_label_source: str = "oracle"
    sampling: Optional[SamplingMetadata] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scalar_target is None and self.vector_target is None:
            raise ValueError("dense supervision example requires scalar_target or vector_target")
        if self.scalar_target is not None and self.vector_target is not None:
            raise ValueError("dense supervision example must use exactly one target kind")


def build_dense_full_document_supervision_dataset(
    examples: Iterable[DenseSupervisionExample],
    *,
    application_name: str,
    supervision_signal_name: str,
    response_signal_name: str,
    law_type: str,
    split: str,
    response_signal_min: Optional[float] = None,
    response_signal_max: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SupervisionDataset:
    """Build a canonical full-document dense supervision dataset."""

    return build_dense_supervision_dataset(
        examples,
        application_name=application_name,
        supervision_channel_name="full_document_supervision",
        supervision_signal_name=supervision_signal_name,
        response_signal_name=response_signal_name,
        law_type=law_type,
        split=split,
        response_signal_min=response_signal_min,
        response_signal_max=response_signal_max,
        metadata=metadata,
    )


def build_dense_sampled_substructure_supervision_dataset(
    examples: Iterable[DenseSupervisionExample],
    *,
    application_name: str,
    supervision_signal_name: str,
    response_signal_name: str,
    law_type: str,
    split: str,
    response_signal_min: Optional[float] = None,
    response_signal_max: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SupervisionDataset:
    """Build a canonical sampled-substructure dense supervision dataset."""

    return build_dense_supervision_dataset(
        examples,
        application_name=application_name,
        supervision_channel_name="sampled_substructure_supervision",
        supervision_signal_name=supervision_signal_name,
        response_signal_name=response_signal_name,
        law_type=law_type,
        split=split,
        response_signal_min=response_signal_min,
        response_signal_max=response_signal_max,
        metadata=metadata,
    )


def build_dense_supervision_dataset(
    examples: Iterable[DenseSupervisionExample],
    *,
    application_name: str,
    supervision_channel_name: str,
    supervision_signal_name: str,
    response_signal_name: str,
    law_type: str,
    split: str,
    response_signal_min: Optional[float] = None,
    response_signal_max: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SupervisionDataset:
    """Build a canonical dense supervision dataset over numeric targets."""

    rows = list(examples)
    shared_metadata = dict(metadata or {})
    judgments: List[ResponseJudgment] = []
    for row in rows:
        sampling = row.sampling or SamplingMetadata(
            document_propensity=1.0,
            unit_propensity=1.0,
            label_propensity=1.0,
            sampling_scheme=str(supervision_channel_name),
            policy_name="all_documents",
            unit_kind=row.unit_kind,
            supports_ipw_estimation=True,
            metadata={
                "split": str(split),
                "representation_kind": "dense_feature_vector",
            },
        )
        dense_metadata = {
            **dense_supervision_metadata(
                representation_kind=REPRESENTATION_DENSE_FEATURE_VECTOR,
                target_kind=(
                    TARGET_VECTOR if row.vector_target is not None else TARGET_SCALAR
                ),
                metadata={
                    **shared_metadata,
                    **dict(row.metadata),
                    "split": str(split),
                    "feature_dim": int(len(tuple(row.features))),
                },
            ),
        }
        judgments.append(
            ResponseJudgment(
                judgment_id=f"{row.example_id}:{supervision_signal_name}",
                source_example_id=str(row.example_id),
                original_text=str(
                    row.original_text
                    if row.original_text is not None
                    else f"{application_name}::{row.example_id}"
                ),
                rubric=str(row.rubric),
                response=str(row.response),
                response_id=(
                    str(row.response_id)
                    if row.response_id is not None
                    else f"{row.example_id}:dense_candidate"
                ),
                reference_score=float(row.reference_score),
                law_type=str(law_type),
                source_doc_id=row.source_doc_id or str(row.example_id),
                truth_label_source=str(row.truth_label_source),
                sampling=sampling,
                supervision_metadata=judgment_supervision_metadata(
                    application_name=str(application_name),
                    supervision_channel_name=str(supervision_channel_name),
                    supervision_signal_name=str(supervision_signal_name),
                    response_signal_name=str(response_signal_name),
                    response_signal_min=response_signal_min,
                    response_signal_max=response_signal_max,
                    law_type=str(law_type),
                    metadata=dense_metadata,
                ),
                source_observation_ids=list(row.source_observation_ids),
                response_signal_value=(
                    float(row.scalar_target) if row.scalar_target is not None else None
                ),
                response_signal_vector=(
                    [float(value) for value in row.vector_target]
                    if row.vector_target is not None
                    else None
                ),
                candidate_features=[float(value) for value in row.features],
                metadata=dense_metadata,
            )
        )
    return SupervisionDataset(response_judgments=judgments)


__all__ = [
    "DenseSupervisionExample",
    "build_dense_sampled_substructure_supervision_dataset",
    "build_dense_supervision_dataset",
    "build_dense_full_document_supervision_dataset",
]
