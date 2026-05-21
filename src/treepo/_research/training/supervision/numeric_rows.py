"""Shared adapters from canonical supervision to dense numeric training rows."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

import numpy as np

from treepo._research.training.supervision.types import (
    BinaryComparison,
    BinaryProjectionDataset,
    ComparativeDataset,
    ComparativeJudgment,
    SupervisionDataset,
    coerce_supervision_dataset,
)

DenseNumericSupervision = Union[
    SupervisionDataset,
    BinaryProjectionDataset,
    ComparativeDataset,
    Sequence[BinaryComparison],
    Sequence[ComparativeJudgment],
]
DenseScalarSupervision = DenseNumericSupervision
DenseVectorSupervision = DenseNumericSupervision


def dense_scalar_rows(supervision: DenseScalarSupervision) -> List[Dict[str, Any]]:
    """Return dense scalar rows from any supported supervision surface."""

    return coerce_supervision_dataset(supervision).to_dense_scalar_training_records()


def dense_vector_rows(supervision: DenseVectorSupervision) -> List[Dict[str, Any]]:
    """Return dense vector rows from any supported supervision surface."""

    return coerce_supervision_dataset(supervision).to_dense_vector_training_records()


def dense_scalar_rows_to_numpy(
    rows: Sequence[Dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert dense scalar rows to feature, target, and weight arrays."""

    if not rows:
        raise ValueError("dense scalar supervision requires at least one row")
    features = [list(row.get("features") or []) for row in rows]
    input_dim = len(features[0])
    if input_dim <= 0:
        raise ValueError("dense scalar supervision rows must include non-empty features")
    for index, row_features in enumerate(features):
        if len(row_features) != input_dim:
            raise ValueError(
                "dense scalar supervision rows must share a fixed feature dimension "
                f"(row 0 has {input_dim}, row {index} has {len(row_features)})"
            )
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray([float(row["score"]) for row in rows], dtype=np.float32)
    w = np.asarray([float(row.get("sample_weight", 1.0)) for row in rows], dtype=np.float32)
    return x, y, w


def dense_vector_rows_to_numpy(
    rows: Sequence[Dict[str, Any]],
    *,
    normalize_targets_to_simplex: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert dense vector rows to feature, target, and weight arrays."""

    if not rows:
        raise ValueError("dense vector supervision requires at least one row")
    features = [list(row.get("features") or []) for row in rows]
    input_dim = len(features[0])
    if input_dim <= 0:
        raise ValueError("dense vector supervision rows must include non-empty features")
    targets = [list(row.get("target") or []) for row in rows]
    output_dim = len(targets[0])
    if output_dim <= 0:
        raise ValueError("dense vector supervision rows must include non-empty targets")
    for index, row_features in enumerate(features):
        if len(row_features) != input_dim:
            raise ValueError(
                "dense vector supervision feature dimensions must match "
                f"(row 0 has {input_dim}, row {index} has {len(row_features)})"
            )
    for index, row_target in enumerate(targets):
        if len(row_target) != output_dim:
            raise ValueError(
                "dense vector supervision target dimensions must match "
                f"(row 0 has {output_dim}, row {index} has {len(row_target)})"
            )
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    if normalize_targets_to_simplex:
        row_sums = np.sum(y, axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-12)
        y = np.asarray(y / row_sums, dtype=np.float32)
    w = np.asarray([float(row.get("sample_weight", 1.0)) for row in rows], dtype=np.float32)
    return x, y, w


__all__ = [
    "DenseNumericSupervision",
    "DenseScalarSupervision",
    "DenseVectorSupervision",
    "dense_scalar_rows",
    "dense_scalar_rows_to_numpy",
    "dense_vector_rows",
    "dense_vector_rows_to_numpy",
]
