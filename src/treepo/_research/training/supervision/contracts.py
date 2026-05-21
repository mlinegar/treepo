"""Standardized contracts for supervision-backed training and calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Dict, Optional


TRAINING_SURFACE_SUPERVISION_DATASET = "supervision_dataset"

REPRESENTATION_DENSE_FEATURE_VECTOR = "dense_feature_vector"
REPRESENTATION_EMBEDDING_VECTOR = "embedding_vector"
REPRESENTATION_SIMPLEX_VECTOR = "simplex_vector"
REPRESENTATION_RAW_SCALAR_SCORE = "raw_scalar_score"
REPRESENTATION_BAG_OF_EMBEDDING_VECTORS = "bag_of_embedding_vectors"

TARGET_SCALAR = "scalar"
TARGET_VECTOR = "vector"
TARGET_SIMPLEX_VECTOR = "simplex_vector"
TARGET_COMPARATIVE_JUDGMENT = "comparative_judgment"

OPTIMIZER_FAMILY_CLOSED_FORM_LINEAR = "closed_form_linear_regression"
OPTIMIZER_FAMILY_GRADIENT_DENSE = "gradient_based_dense_regression"
OPTIMIZER_FAMILY_TREE_ENSEMBLE = "tree_ensemble_regression"
OPTIMIZER_FAMILY_AFFINE_VECTOR_CALIBRATION = "affine_vector_calibration"
OPTIMIZER_FAMILY_BAG_LEVEL_GRADIENT = "bag_level_gradient_regression"
OPTIMIZER_FAMILY_GROUPWISE_COMPARATIVE = "groupwise_comparative_optimization"
OPTIMIZER_FAMILY_BINARY_PROJECTION = "binary_projection"


def infer_output_kind(*, target_kind: str) -> str:
    target = str(target_kind)
    if target == TARGET_SCALAR:
        return "scalar_prediction"
    if target == TARGET_SIMPLEX_VECTOR:
        return "simplex_prediction"
    if target == TARGET_VECTOR:
        return "vector_prediction"
    if target == TARGET_COMPARATIVE_JUDGMENT:
        return "comparative_policy"
    return f"{target}_artifact"


def infer_supervision_mode(
    *,
    representation_kind: str,
    target_kind: str,
    optimizer_family: str,
) -> str:
    target = str(target_kind)
    optimizer = str(optimizer_family)
    representation = str(representation_kind)

    if target == TARGET_SCALAR and optimizer in {
        OPTIMIZER_FAMILY_CLOSED_FORM_LINEAR,
        OPTIMIZER_FAMILY_GRADIENT_DENSE,
    }:
        return "dense_scalar_regression"
    if target == TARGET_SCALAR and optimizer == OPTIMIZER_FAMILY_BAG_LEVEL_GRADIENT:
        return "bagged_scalar_regression"
    if target == TARGET_VECTOR and optimizer in {
        OPTIMIZER_FAMILY_CLOSED_FORM_LINEAR,
        OPTIMIZER_FAMILY_GRADIENT_DENSE,
        OPTIMIZER_FAMILY_TREE_ENSEMBLE,
    }:
        return "dense_vector_regression"
    if target == TARGET_SIMPLEX_VECTOR and optimizer in {
        OPTIMIZER_FAMILY_GRADIENT_DENSE,
        OPTIMIZER_FAMILY_TREE_ENSEMBLE,
    }:
        return "dense_simplex_regression"
    if target == TARGET_VECTOR and optimizer == OPTIMIZER_FAMILY_AFFINE_VECTOR_CALIBRATION:
        return "dense_affine_vector_calibration"
    if target == TARGET_SIMPLEX_VECTOR and optimizer == OPTIMIZER_FAMILY_AFFINE_VECTOR_CALIBRATION:
        return "dense_affine_simplex_calibration"
    if target == TARGET_COMPARATIVE_JUDGMENT and optimizer == OPTIMIZER_FAMILY_GROUPWISE_COMPARATIVE:
        return "comparative_group_optimization"
    if target == TARGET_COMPARATIVE_JUDGMENT and optimizer == OPTIMIZER_FAMILY_BINARY_PROJECTION:
        return "binary_preference_projection"
    return f"{representation}__{target}__{optimizer}"


def dense_supervision_metadata(
    *,
    representation_kind: str = REPRESENTATION_DENSE_FEATURE_VECTOR,
    target_kind: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return canonical supervision metadata for dense numeric rows."""

    return {
        "representation_kind": str(representation_kind),
        "target_kind": str(target_kind),
        **dict(metadata or {}),
    }


@dataclass(frozen=True)
class SupervisionTrainingContract:
    """Canonical training/calibration contract for supervision-backed learners."""

    representation_kind: str
    target_kind: str
    optimizer_family: str
    optimizer_backend: str
    training_surface: str = TRAINING_SURFACE_SUPERVISION_DATASET
    supervision_mode: Optional[str] = None
    output_kind: Optional[str] = None
    selection_mode: Optional[str] = None
    selection_split: Optional[str] = None
    selection_metric_name: Optional[str] = None
    selection_metric_value: Optional[float] = None
    best_epoch: Optional[int] = None
    n_train_rows: Optional[int] = None
    n_val_rows: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.supervision_mode is None:
            object.__setattr__(
                self,
                "supervision_mode",
                infer_supervision_mode(
                    representation_kind=self.representation_kind,
                    target_kind=self.target_kind,
                    optimizer_family=self.optimizer_family,
                ),
            )
        if self.output_kind is None:
            object.__setattr__(
                self,
                "output_kind",
                infer_output_kind(target_kind=self.target_kind),
            )

    def to_dict(self, *, prefix: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "training_surface": str(self.training_surface),
            "representation_kind": str(self.representation_kind),
            "target_kind": str(self.target_kind),
            "optimizer_family": str(self.optimizer_family),
            "optimizer_backend": str(self.optimizer_backend),
            "supervision_mode": str(self.supervision_mode),
            "output_kind": str(self.output_kind),
        }
        if self.selection_mode is not None:
            payload["selection_mode"] = str(self.selection_mode)
        if self.selection_split is not None:
            payload["selection_split"] = str(self.selection_split)
        if self.selection_metric_name is not None:
            payload["selection_metric_name"] = str(self.selection_metric_name)
        if self.selection_metric_value is not None:
            payload["selection_metric_value"] = float(self.selection_metric_value)
        if self.best_epoch is not None:
            payload["best_epoch"] = int(self.best_epoch)
        if self.n_train_rows is not None:
            payload["supervision_rows"] = int(self.n_train_rows)
        if self.n_val_rows is not None:
            payload["validation_rows"] = int(self.n_val_rows)
        if self.metadata:
            payload["training_contract_metadata"] = dict(self.metadata)

        if prefix is None:
            return payload
        return {f"{prefix}_{key}": value for key, value in payload.items()}


def supervision_training_contract(
    *,
    representation_kind: str,
    target_kind: str,
    optimizer_family: str,
    optimizer_backend: str,
    prefix: Optional[str] = None,
    training_surface: str = TRAINING_SURFACE_SUPERVISION_DATASET,
    selection_mode: Optional[str] = None,
    selection_split: Optional[str] = None,
    selection_metric_name: Optional[str] = None,
    selection_metric_value: Optional[float] = None,
    best_epoch: Optional[int] = None,
    n_train_rows: Optional[int] = None,
    n_val_rows: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a standardized dict contract for supervision-backed training."""

    safe_metric_value: Optional[float]
    if selection_metric_value is None:
        safe_metric_value = None
    else:
        converted = float(selection_metric_value)
        safe_metric_value = converted if isfinite(converted) else converted
    return SupervisionTrainingContract(
        training_surface=str(training_surface),
        representation_kind=str(representation_kind),
        target_kind=str(target_kind),
        optimizer_family=str(optimizer_family),
        optimizer_backend=str(optimizer_backend),
        selection_mode=None if selection_mode is None else str(selection_mode),
        selection_split=None if selection_split is None else str(selection_split),
        selection_metric_name=(
            None if selection_metric_name is None else str(selection_metric_name)
        ),
        selection_metric_value=safe_metric_value,
        best_epoch=None if best_epoch is None else int(best_epoch),
        n_train_rows=None if n_train_rows is None else int(n_train_rows),
        n_val_rows=None if n_val_rows is None else int(n_val_rows),
        metadata=dict(metadata or {}),
    ).to_dict(prefix=prefix)


__all__ = [
    "OPTIMIZER_FAMILY_AFFINE_VECTOR_CALIBRATION",
    "OPTIMIZER_FAMILY_BAG_LEVEL_GRADIENT",
    "OPTIMIZER_FAMILY_BINARY_PROJECTION",
    "OPTIMIZER_FAMILY_CLOSED_FORM_LINEAR",
    "OPTIMIZER_FAMILY_GRADIENT_DENSE",
    "OPTIMIZER_FAMILY_GROUPWISE_COMPARATIVE",
    "OPTIMIZER_FAMILY_TREE_ENSEMBLE",
    "REPRESENTATION_BAG_OF_EMBEDDING_VECTORS",
    "REPRESENTATION_DENSE_FEATURE_VECTOR",
    "REPRESENTATION_EMBEDDING_VECTOR",
    "REPRESENTATION_RAW_SCALAR_SCORE",
    "REPRESENTATION_SIMPLEX_VECTOR",
    "SupervisionTrainingContract",
    "TARGET_COMPARATIVE_JUDGMENT",
    "TARGET_SCALAR",
    "TARGET_SIMPLEX_VECTOR",
    "TARGET_VECTOR",
    "TRAINING_SURFACE_SUPERVISION_DATASET",
    "dense_supervision_metadata",
    "infer_output_kind",
    "infer_supervision_mode",
    "supervision_training_contract",
]
