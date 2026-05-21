"""Shared supervision acquisition and activation timing contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


ACQUISITION_OFFLINE_ARTIFACT = "offline_artifact"
ACQUISITION_NONE = "none"
ACQUISITION_SYNCHRONOUS_ORACLE = "synchronous_oracle"
ACQUISITION_ASYNC_FEEDBACK_QUEUE = "async_feedback_queue"
ACQUISITION_HUMAN_REVIEW = "human_review"
ACQUISITION_TEACHER_WORKER = "teacher_worker"
ACQUISITION_SYNCHRONOUS_OPTIMIZER_METRIC = "synchronous_optimizer_metric"

ACTIVATION_IMMEDIATE = "immediate"
ACTIVATION_TREE_PREP = "tree_prep_boundary"
ACTIVATION_BATCH_BOUNDARY = "batch_boundary"
ACTIVATION_EPOCH_BOUNDARY = "epoch_boundary"
ACTIVATION_ITERATION_BOUNDARY = "iteration_boundary"
ACTIVATION_NEXT_RUN = "next_run"

CONSUMER_CTREEPO_GRADIENT = "ctreepo_gradient_trainer"
CONSUMER_GEPA_OPTIMIZER = "gepa_optimizer"
CONSUMER_JUDGE_GEPA_OPTIMIZER = "judge_gepa_optimizer"
CONSUMER_DISTILLATION_FIT = "distillation_fit"


def default_label_lag_policy(activation_barrier: str) -> str:
    barrier = str(activation_barrier)
    if barrier == ACTIVATION_IMMEDIATE:
        return "completed_labels_active_immediately"
    if barrier == ACTIVATION_TREE_PREP:
        return "labels_collected_during_tree_preparation_active_from_epoch_0"
    if barrier == ACTIVATION_BATCH_BOUNDARY:
        return "completed_during_batch_k_active_after_batch_k"
    if barrier == ACTIVATION_EPOCH_BOUNDARY:
        return "completed_during_epoch_k_active_no_earlier_than_epoch_k_plus_1"
    if barrier == ACTIVATION_ITERATION_BOUNDARY:
        return "completed_during_iteration_k_active_no_earlier_than_iteration_k_plus_1"
    if barrier == ACTIVATION_NEXT_RUN:
        return "completed_labels_persist_for_later_artifact_replay"
    return f"activation_barrier={barrier}"


@dataclass(frozen=True)
class SupervisionTimingContract:
    """JSON-friendly contract for when supervision is acquired and consumed."""

    acquisition_policy: str
    activation_barrier: str
    consumer: str
    producer: Optional[str] = None
    delivery_mode: Optional[str] = None
    blocking: bool = False
    label_lag_policy: Optional[str] = None
    notes: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label_lag_policy is None:
            object.__setattr__(
                self,
                "label_lag_policy",
                default_label_lag_policy(self.activation_barrier),
            )

    def to_dict(self, *, prefix: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "acquisition_policy": str(self.acquisition_policy),
            "activation_barrier": str(self.activation_barrier),
            "consumer": str(self.consumer),
            "blocking": bool(self.blocking),
            "label_lag_policy": str(self.label_lag_policy),
        }
        if self.producer is not None:
            payload["producer"] = str(self.producer)
        if self.delivery_mode is not None:
            payload["delivery_mode"] = str(self.delivery_mode)
        if self.notes:
            payload["notes"] = [str(note) for note in self.notes]
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        if prefix is None:
            return payload
        return {f"{prefix}_{key}": value for key, value in payload.items()}


def supervision_timing_contract(
    *,
    acquisition_policy: str,
    activation_barrier: str,
    consumer: str,
    producer: Optional[str] = None,
    delivery_mode: Optional[str] = None,
    blocking: bool = False,
    label_lag_policy: Optional[str] = None,
    notes: Tuple[str, ...] = (),
    metadata: Optional[Dict[str, Any]] = None,
    prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a standardized supervision timing payload."""

    return SupervisionTimingContract(
        acquisition_policy=str(acquisition_policy),
        activation_barrier=str(activation_barrier),
        consumer=str(consumer),
        producer=None if producer is None else str(producer),
        delivery_mode=None if delivery_mode is None else str(delivery_mode),
        blocking=bool(blocking),
        label_lag_policy=label_lag_policy,
        notes=tuple(notes),
        metadata=dict(metadata or {}),
    ).to_dict(prefix=prefix)


__all__ = [
    "ACQUISITION_ASYNC_FEEDBACK_QUEUE",
    "ACQUISITION_HUMAN_REVIEW",
    "ACQUISITION_NONE",
    "ACQUISITION_OFFLINE_ARTIFACT",
    "ACQUISITION_SYNCHRONOUS_OPTIMIZER_METRIC",
    "ACQUISITION_SYNCHRONOUS_ORACLE",
    "ACQUISITION_TEACHER_WORKER",
    "ACTIVATION_BATCH_BOUNDARY",
    "ACTIVATION_EPOCH_BOUNDARY",
    "ACTIVATION_IMMEDIATE",
    "ACTIVATION_ITERATION_BOUNDARY",
    "ACTIVATION_NEXT_RUN",
    "ACTIVATION_TREE_PREP",
    "CONSUMER_CTREEPO_GRADIENT",
    "CONSUMER_DISTILLATION_FIT",
    "CONSUMER_GEPA_OPTIMIZER",
    "CONSUMER_JUDGE_GEPA_OPTIMIZER",
    "SupervisionTimingContract",
    "default_label_lag_policy",
    "supervision_timing_contract",
]
