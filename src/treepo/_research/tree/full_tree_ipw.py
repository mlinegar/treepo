"""Helpers for full-tree IPW accounting with separate document-level supervision.

This module keeps two supervision channels distinct:

1. full-document labels, which are observed for every document; and
2. sampled node labels over the realized tree, which require logged propensities
   for unbiased finite-population estimation.

The node records are generic and can be produced by analytic simulations or by a
learned shared readout ``g`` applied to every realized tree node.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import random
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from treepo._research.core.local_law_adjustment import (
    LocalLawObservation,
    aggregate_local_law_observations,
)
from treepo._research.core.logged_supervision import SamplingMetadata
from treepo._research.tree.ipw import (
    NodeType,
    TreeSample,
    effective_sample_size,
    hajek_estimate,
    horvitz_thompson_mean,
    max_weight,
)


def squared_error(prediction: float, target: float) -> float:
    err = float(prediction) - float(target)
    return float(err * err)


def absolute_error(prediction: float, target: float) -> float:
    return abs(float(prediction) - float(target))


LossFn = Callable[[float, float], float]


DEFAULT_LAYERED_RATE_GRID: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0)
DEFAULT_TRADEOFF_RATE_GRID: tuple[float, ...] = (
    0.0,
    0.05,
    0.1,
    0.15,
    0.2,
    0.3,
    0.5,
    1.0,
)
@dataclass(frozen=True)
class FullTreeNodeRecord:
    """One realized node in a fully instantiated tree population."""

    doc_id: str
    node_id: str
    depth: int
    node_type: NodeType
    is_root: bool
    prediction: float
    target: float
    sampled: bool
    propensity: float
    objective_prediction: Optional[float] = None
    proxy_loss: Optional[float] = None
    oracle_loss: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.depth) < 0:
            raise ValueError(f"depth must be non-negative, got {self.depth!r}")
        if not isinstance(self.node_type, NodeType):
            object.__setattr__(self, "node_type", NodeType(str(self.node_type)))
        if not math.isfinite(float(self.prediction)):
            raise ValueError(f"prediction must be finite, got {self.prediction!r}")
        if not math.isfinite(float(self.target)):
            raise ValueError(f"target must be finite, got {self.target!r}")
        if self.objective_prediction is not None and not math.isfinite(
            float(self.objective_prediction)
        ):
            raise ValueError(
                "objective_prediction must be finite when provided, "
                f"got {self.objective_prediction!r}"
            )
        if self.proxy_loss is not None and not math.isfinite(float(self.proxy_loss)):
            raise ValueError(f"proxy_loss must be finite when provided, got {self.proxy_loss!r}")
        if self.oracle_loss is not None and not math.isfinite(float(self.oracle_loss)):
            raise ValueError(f"oracle_loss must be finite when provided, got {self.oracle_loss!r}")
        prop = float(self.propensity)
        if not math.isfinite(prop) or prop < 0.0 or prop > 1.0:
            raise ValueError(f"propensity must be finite and in [0, 1], got {self.propensity!r}")
        if bool(self.sampled) and prop <= 0.0:
            raise ValueError("sampled node records require strictly positive propensities")
        if not isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def loss_prediction(self) -> float:
        return (
            float(self.objective_prediction)
            if self.objective_prediction is not None
            else float(self.prediction)
        )


@dataclass(frozen=True)
class DocumentLevelPredictionRecord:
    """Always-observed document-level supervision target."""

    doc_id: str
    prediction: float
    target: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.prediction)):
            raise ValueError(f"prediction must be finite, got {self.prediction!r}")
        if not math.isfinite(float(self.target)):
            raise ValueError(f"target must be finite, got {self.target!r}")
        if not isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class EstimatorMomentSummary:
    mean: float
    bias: float
    variance: float
    rmse: float


@dataclass(frozen=True)
class FullTreeEstimatorMonteCarloSummary:
    policy_name: str
    trials: int
    true_full_node_mean: float
    document_top_loss: float
    document_top_mae: float
    naive: EstimatorMomentSummary
    ht: EstimatorMomentSummary
    hajek: EstimatorMomentSummary
    mean_sample_count: float
    mean_effective_sample_size: float
    mean_max_weight: float


@dataclass
class _StreamingBreakdownAccumulator:
    population_count: int = 0
    sampled_count: int = 0
    exact_loss_total: float = 0.0
    sampled_loss_total: float = 0.0
    ht_weighted_total: float = 0.0
    hajek_weighted_total: float = 0.0
    weight_sum: float = 0.0
    weight_sq_sum: float = 0.0
    max_weight_value: float = 0.0

    def update(self, record: FullTreeNodeRecord, *, loss_value: float) -> None:
        self.population_count += 1
        self.exact_loss_total += float(loss_value)
        if not bool(record.sampled) or float(record.propensity) <= 0.0:
            return
        weight = 1.0 / max(float(record.propensity), 1e-12)
        self.sampled_count += 1
        self.sampled_loss_total += float(loss_value)
        self.ht_weighted_total += float(weight * float(loss_value))
        self.hajek_weighted_total += float(weight * float(loss_value))
        self.weight_sum += float(weight)
        self.weight_sq_sum += float(weight * weight)
        self.max_weight_value = max(float(self.max_weight_value), float(weight))

    def _exact_mean_loss(self) -> float:
        if self.population_count <= 0:
            return float("nan")
        return float(self.exact_loss_total / float(self.population_count))

    def _naive_mean_loss(self) -> float:
        if self.sampled_count <= 0:
            return float("nan")
        return float(self.sampled_loss_total / float(self.sampled_count))

    def _ht_mean_loss(self) -> float:
        if self.population_count <= 0 or self.sampled_count <= 0:
            return float("nan")
        return float(self.ht_weighted_total / float(self.population_count))

    def _hajek_mean_loss(self) -> float:
        if self.weight_sum <= 0.0 or self.sampled_count <= 0:
            return float("nan")
        return float(self.hajek_weighted_total / float(self.weight_sum))

    def _effective_sample_size(self) -> float:
        if self.weight_sq_sum <= 0.0:
            return 0.0
        return float((self.weight_sum * self.weight_sum) / self.weight_sq_sum)

    def summary(self) -> Dict[str, Any]:
        return {
            "population_count": int(self.population_count),
            "sampled_count": int(self.sampled_count),
            "exact_mean_loss": self._exact_mean_loss(),
            "naive_mean_loss": self._naive_mean_loss(),
            "ht_mean_loss": self._ht_mean_loss(),
            "hajek_mean_loss": self._hajek_mean_loss(),
            "effective_sample_size": self._effective_sample_size(),
            "max_weight": float(self.max_weight_value) if self.sampled_count > 0 else 0.0,
        }


@dataclass
class FullTreeIPWSummaryAccumulator:
    node_loss_fn: LossFn = squared_error
    document_loss_fn: LossFn = squared_error
    node_stats: _StreamingBreakdownAccumulator = field(
        default_factory=_StreamingBreakdownAccumulator
    )
    corrected_local_law_observations: list[LocalLawObservation] = field(default_factory=list)
    type_groups: Dict[str, _StreamingBreakdownAccumulator] = field(default_factory=dict)
    depth_groups: Dict[str, _StreamingBreakdownAccumulator] = field(default_factory=dict)
    document_loss_total: float = 0.0
    document_mae_total: float = 0.0
    document_count: int = 0
    target_gap_total: float = 0.0
    prediction_gap_total: float = 0.0
    gap_count: int = 0
    _root_pairs_by_doc: Dict[str, tuple[float, float]] = field(default_factory=dict)
    _document_pairs_by_doc: Dict[str, tuple[float, float]] = field(default_factory=dict)

    def _maybe_commit_gap(self, doc_id: str) -> None:
        root_pair = self._root_pairs_by_doc.get(str(doc_id))
        document_pair = self._document_pairs_by_doc.get(str(doc_id))
        if root_pair is None or document_pair is None:
            return
        root_target, root_prediction = root_pair
        document_target, document_prediction = document_pair
        self.target_gap_total += abs(float(document_target) - float(root_target))
        self.prediction_gap_total += abs(float(document_prediction) - float(root_prediction))
        self.gap_count += 1
        self._root_pairs_by_doc.pop(str(doc_id), None)
        self._document_pairs_by_doc.pop(str(doc_id), None)

    def update_node_record(self, record: FullTreeNodeRecord) -> None:
        loss_value = _loss_value(record, self.node_loss_fn)
        self.node_stats.update(record, loss_value=float(loss_value))
        self.corrected_local_law_observations.append(
            _local_law_observation_from_node_record(record, loss_fn=self.node_loss_fn)
        )
        bucket = "root" if bool(record.is_root) else str(record.node_type.value)
        self.type_groups.setdefault(bucket, _StreamingBreakdownAccumulator()).update(
            record,
            loss_value=float(loss_value),
        )
        depth_key = str(int(record.depth))
        self.depth_groups.setdefault(depth_key, _StreamingBreakdownAccumulator()).update(
            record,
            loss_value=float(loss_value),
        )
        if bool(record.is_root):
            self._root_pairs_by_doc[str(record.doc_id)] = (
                float(record.target),
                float(record.loss_prediction),
            )
            self._maybe_commit_gap(str(record.doc_id))

    def update_document_record(self, record: DocumentLevelPredictionRecord) -> None:
        self.document_count += 1
        self.document_loss_total += float(_document_loss_value(record, self.document_loss_fn))
        self.document_mae_total += float(_document_loss_value(record, absolute_error))
        self._document_pairs_by_doc[str(record.doc_id)] = (
            float(record.target),
            float(record.prediction),
        )
        self._maybe_commit_gap(str(record.doc_id))

    def finalize(self) -> Dict[str, Any]:
        exact_mean_loss = self.node_stats._exact_mean_loss()
        naive_mean_loss = self.node_stats._naive_mean_loss()
        ht_mean_loss = self.node_stats._ht_mean_loss()
        hajek_mean_loss = self.node_stats._hajek_mean_loss()
        document_top_loss = (
            float(self.document_loss_total / float(self.document_count))
            if self.document_count > 0
            else float("nan")
        )
        document_top_mae = (
            float(self.document_mae_total / float(self.document_count))
            if self.document_count > 0
            else float("nan")
        )
        return {
            "population_size": int(self.node_stats.population_count),
            "sampled_nodes": int(self.node_stats.sampled_count),
            "sampled_fraction": (
                float(self.node_stats.sampled_count) / float(self.node_stats.population_count)
                if self.node_stats.population_count > 0
                else float("nan")
            ),
            "full_node_exact_mean_loss": exact_mean_loss,
            "sampled_node_naive_mean_loss": naive_mean_loss,
            "sampled_node_naive_signed_error": _estimator_error_summary(
                exact_mean_loss, naive_mean_loss
            )["signed_error"],
            "sampled_node_naive_abs_error": _estimator_error_summary(
                exact_mean_loss, naive_mean_loss
            )["abs_error"],
            "sampled_node_ht_mean_loss": ht_mean_loss,
            "sampled_node_ht_signed_error": _estimator_error_summary(
                exact_mean_loss, ht_mean_loss
            )["signed_error"],
            "sampled_node_ht_abs_error": _estimator_error_summary(
                exact_mean_loss, ht_mean_loss
            )["abs_error"],
            "sampled_node_hajek_mean_loss": hajek_mean_loss,
            "sampled_node_hajek_signed_error": _estimator_error_summary(
                exact_mean_loss, hajek_mean_loss
            )["signed_error"],
            "sampled_node_hajek_abs_error": _estimator_error_summary(
                exact_mean_loss, hajek_mean_loss
            )["abs_error"],
            "document_top_loss": document_top_loss,
            "document_top_mae": document_top_mae,
            "document_vs_root_node_target_gap_mae": (
                float(self.target_gap_total / float(self.gap_count))
                if self.gap_count > 0
                else float("nan")
            ),
            "document_vs_root_node_prediction_gap_mae": (
                float(self.prediction_gap_total / float(self.gap_count))
                if self.gap_count > 0
                else float("nan")
            ),
            "document_vs_root_node_pair_count": int(self.gap_count),
            "effective_sample_size": self.node_stats._effective_sample_size(),
            "max_weight": (
                float(self.node_stats.max_weight_value)
                if self.node_stats.sampled_count > 0
                else 0.0
            ),
            "weight_sum": float(self.node_stats.weight_sum),
            "corrected_local_law": aggregate_local_law_observations(
                self.corrected_local_law_observations,
                local_law_weight=1.0,
            ).to_dict(),
            "type_breakdown": {
                str(name): acc.summary()
                for name, acc in sorted(self.type_groups.items())
            },
            "depth_breakdown": {
                str(name): acc.summary()
                for name, acc in sorted(
                    self.depth_groups.items(),
                    key=lambda item: int(item[0]),
                )
            },
        }


def classify_layered_sampling_regime(
    *,
    leaf_rate: float,
    internal_rate: float,
) -> str:
    leaf = float(leaf_rate)
    internal = float(internal_rate)
    if abs(leaf) <= 1e-12 and abs(internal) <= 1e-12:
        return "doc_only"
    if abs(leaf - 1.0) <= 1e-12 and abs(internal - 1.0) <= 1e-12:
        return "full_tree"
    if abs(leaf - internal) <= 1e-12:
        return "uniform_ipw"
    if internal > leaf:
        return "internal_heavy"
    return "leaf_heavy"


def layered_propensity_policy(
    *,
    leaf_rate: float,
    internal_rate: float,
) -> Callable[[FullTreeNodeRecord], float]:
    leaf = min(1.0, max(0.0, float(leaf_rate)))
    internal = min(1.0, max(0.0, float(internal_rate)))

    def _policy(record: FullTreeNodeRecord) -> float:
        if record.node_type == NodeType.LEAF:
            return float(leaf)
        return float(internal)

    return _policy


def _finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _loss_value(record: FullTreeNodeRecord, loss_fn: LossFn) -> float:
    return float(loss_fn(record.loss_prediction, record.target))


def _local_law_observation_from_node_record(
    record: FullTreeNodeRecord,
    *,
    loss_fn: LossFn = squared_error,
) -> LocalLawObservation:
    exact_loss = _loss_value(record, loss_fn)
    proxy_loss = float(record.proxy_loss) if record.proxy_loss is not None else float(exact_loss)
    if bool(record.sampled) and record.oracle_loss is None:
        raise ValueError(
            f"sampled full-tree node record {record.doc_id}/{record.node_id} "
            "requires explicit oracle_loss"
        )
    observed = bool(record.sampled)
    oracle_loss = float(record.oracle_loss) if observed and record.oracle_loss is not None else None
    try:
        node_weight = float(record.metadata.get("node_weight", 1.0))
    except (TypeError, ValueError):
        node_weight = 1.0
    return LocalLawObservation(
        proxy_loss=float(proxy_loss),
        oracle_loss=oracle_loss,
        observed=bool(observed),
        propensity=float(record.propensity if observed else 0.0),
        depth=int(record.depth),
        node_weight=float(node_weight),
        metadata={
            "doc_id": str(record.doc_id),
            "node_id": str(record.node_id),
            "node_type": str(record.node_type.value),
            **dict(record.metadata),
        },
    )


def corrected_local_law_node_summary(
    records: Sequence[FullTreeNodeRecord],
    *,
    loss_fn: LossFn = squared_error,
    gamma_depth: float = 1.0,
    local_law_weight: float = 1.0,
) -> Dict[str, Any]:
    """Summarize dense-proxy plus sampled-oracle corrected node losses."""

    aggregate = aggregate_local_law_observations(
        [
            _local_law_observation_from_node_record(record, loss_fn=loss_fn)
            for record in records
        ],
        gamma_depth=float(gamma_depth),
        local_law_weight=float(local_law_weight),
    )
    return aggregate.to_dict()


def _document_loss_value(record: DocumentLevelPredictionRecord, loss_fn: LossFn) -> float:
    return float(loss_fn(record.prediction, record.target))


def _metadata_float(
    metadata: Mapping[str, Any],
    keys: Sequence[str],
    *,
    default: Optional[float] = None,
) -> Optional[float]:
    for key in keys:
        if key not in metadata:
            continue
        value = metadata.get(key)
        if value is None:
            continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(out):
            return out
    return default


def full_tree_node_records_from_state_tree(
    tree: Any,
    *,
    default_sampled: bool = False,
    default_propensity: Optional[float] = None,
) -> list[FullTreeNodeRecord]:
    """Project a rich ``StateTree`` full-tree trace into estimator records."""

    doc_id = str(getattr(tree, "metadata", {}).get("doc_id", "") or "")
    records: list[FullTreeNodeRecord] = []
    for node in tree.traverse_preorder():
        metadata = dict(getattr(node, "metadata", {}) or {})
        node_doc_id = str(metadata.get("doc_id", doc_id) or doc_id)
        node_type_name = str(metadata.get("node_type", "") or "")
        is_root = bool(metadata.get("is_root", getattr(node, "is_root", False)))
        is_leaf = bool(metadata.get("is_leaf", getattr(node, "is_leaf", False)))
        node_type = NodeType.LEAF if is_leaf else NodeType.MERGE
        if node_type_name in {NodeType.RESUMMARY.value, NodeType.SUBSTITUTION.value}:
            node_type = NodeType(node_type_name)
        prediction = _metadata_float(
            metadata,
            ("prediction", "objective_prediction", "readout_prediction", "scorer_output"),
        )
        target = _metadata_float(
            metadata,
            ("target", "oracle_target", "target_score", "target_raw", "target_1_7"),
        )
        if prediction is None or target is None:
            continue
        sampled = bool(metadata.get("observed", metadata.get("sampled", default_sampled)))
        propensity = _metadata_float(metadata, ("propensity", "logged_propensity"))
        if propensity is None:
            if bool(sampled):
                if bool(is_root):
                    propensity = 1.0
                else:
                    raise ValueError(
                        f"observed non-root trace node {node_doc_id}/{getattr(node, 'id', '')} "
                        "is missing propensity"
                    )
            else:
                propensity = 0.0 if default_propensity is None else float(default_propensity)
        depth = int(
            metadata.get(
                "depth",
                max(0, int(getattr(tree, "height", 0)) - int(getattr(node, "level", 0))),
            )
        )
        records.append(
            FullTreeNodeRecord(
                doc_id=node_doc_id,
                node_id=str(getattr(node, "id", "")),
                depth=depth,
                node_type=node_type,
                is_root=bool(is_root),
                prediction=float(prediction),
                target=float(target),
                sampled=bool(sampled),
                propensity=float(propensity),
                objective_prediction=_metadata_float(metadata, ("objective_prediction",)),
                proxy_loss=_metadata_float(metadata, ("proxy_loss",)),
                oracle_loss=_metadata_float(metadata, ("oracle_loss",)),
                metadata=metadata,
            )
        )
    return records


def document_record_from_state_tree(tree: Any) -> Optional[DocumentLevelPredictionRecord]:
    """Project the root of a full-tree trace into a document-level record."""

    metadata = dict(getattr(tree.root, "metadata", {}) or {})
    prediction = _metadata_float(
        metadata,
        ("document_prediction", "root_prediction", "prediction", "readout_prediction", "scorer_output"),
    )
    target = _metadata_float(
        metadata,
        ("document_target", "root_target", "target", "target_score", "target_raw"),
    )
    if prediction is None or target is None:
        return None
    tree_meta = dict(getattr(tree, "metadata", {}) or {})
    return DocumentLevelPredictionRecord(
        doc_id=str(metadata.get("doc_id", tree_meta.get("doc_id", "")) or ""),
        prediction=float(prediction),
        target=float(target),
        metadata={**tree_meta, **metadata},
    )


def local_law_observations_from_state_tree(tree: Any) -> list[LocalLawObservation]:
    """Project a rich full-tree trace directly into local-law loss rows."""

    observations: list[LocalLawObservation] = []
    for node in tree.traverse_preorder():
        metadata = dict(getattr(node, "metadata", {}) or {})
        proxy_loss = _metadata_float(metadata, ("proxy_loss", "loss_proxy"))
        if proxy_loss is None:
            prediction = _metadata_float(
                metadata,
                ("prediction", "objective_prediction", "readout_prediction", "scorer_output"),
            )
            proxy_target = _metadata_float(
                metadata,
                ("proxy_target", "target", "target_score", "target_raw", "target_1_7"),
            )
            if prediction is not None and proxy_target is not None:
                proxy_loss = float((float(prediction) - float(proxy_target)) ** 2)
        if proxy_loss is None:
            continue
        oracle_loss = _metadata_float(metadata, ("oracle_loss", "loss_oracle"))
        if oracle_loss is None:
            prediction = _metadata_float(
                metadata,
                ("prediction", "objective_prediction", "readout_prediction", "scorer_output"),
            )
            oracle_target = _metadata_float(
                metadata,
                ("oracle_target", "oracle_score", "oracle_target_score"),
            )
            if prediction is not None and oracle_target is not None:
                oracle_loss = float((float(prediction) - float(oracle_target)) ** 2)
        observed = bool(metadata.get("observed", metadata.get("sampled", False)))
        if oracle_loss is None:
            observed = False
        propensity = _metadata_float(metadata, ("propensity", "logged_propensity"))
        if propensity is None:
            is_root = bool(metadata.get("is_root", getattr(node, "is_root", False)))
            if observed:
                if is_root:
                    propensity = 1.0
                else:
                    raise ValueError(
                        f"observed non-root trace node {metadata.get('doc_id', '')}/{getattr(node, 'id', '')} "
                        "is missing propensity"
                    )
            else:
                propensity = 0.0
        depth_raw = metadata.get(
            "depth",
            max(0, int(getattr(tree, "height", 0)) - int(getattr(node, "level", 0))),
        )
        try:
            depth = int(depth_raw)
        except (TypeError, ValueError):
            depth = 0
        node_weight = _metadata_float(metadata, ("node_weight", "weight"), default=1.0)
        observations.append(
            LocalLawObservation(
                proxy_loss=float(proxy_loss),
                oracle_loss=None if oracle_loss is None else float(oracle_loss),
                observed=bool(observed),
                propensity=float(propensity),
                depth=max(0, int(depth)),
                node_weight=float(1.0 if node_weight is None else node_weight),
                metadata={
                    "doc_id": str(metadata.get("doc_id", getattr(tree, "metadata", {}).get("doc_id", "")) or ""),
                    "node_id": str(getattr(node, "id", "")),
                    **metadata,
                },
            )
        )
    return observations


def project_node_records_to_tree_samples(
    records: Sequence[FullTreeNodeRecord],
    *,
    loss_fn: LossFn = squared_error,
    violation_threshold: float = 0.0,
    only_sampled: bool = True,
) -> list[TreeSample]:
    """Project realized node records into ``TreeSample`` rows."""

    samples: list[TreeSample] = []
    for record in records:
        if only_sampled and not bool(record.sampled):
            continue
        if float(record.propensity) <= 0.0:
            if bool(record.sampled):
                raise ValueError(
                    f"sampled record {record.doc_id}/{record.node_id} has non-positive propensity"
                )
            continue
        loss_value = _loss_value(record, loss_fn)
        metadata = {
            "depth": int(record.depth),
            "is_root": bool(record.is_root),
            "prediction": float(record.prediction),
            "target": float(record.target),
            "objective_prediction": float(record.loss_prediction),
            **dict(record.metadata),
        }
        samples.append(
            TreeSample(
                doc_id=str(record.doc_id),
                node_id=str(record.node_id),
                node_type=record.node_type,
                violation=int(loss_value > float(violation_threshold)),
                preference_loss=float(loss_value),
                sampling=SamplingMetadata(
                    document_propensity=1.0,
                    unit_propensity=float(record.propensity),
                    label_propensity=1.0,
                    policy_name="full_tree_logged_sampling",
                    sampling_scheme="realized_tree_nodes",
                    supports_ipw_estimation=True,
                ),
                metadata=metadata,
            )
        )
    return samples


def exact_full_node_mean_loss(
    records: Sequence[FullTreeNodeRecord],
    *,
    loss_fn: LossFn = squared_error,
) -> float:
    if not records:
        return float("nan")
    return float(np.mean([_loss_value(record, loss_fn) for record in records]))


def sampled_naive_node_mean_loss(
    records: Sequence[FullTreeNodeRecord],
    *,
    loss_fn: LossFn = squared_error,
) -> float:
    sampled = [record for record in records if bool(record.sampled)]
    if not sampled:
        return float("nan")
    return float(np.mean([_loss_value(record, loss_fn) for record in sampled]))


def sampled_ht_node_mean_loss(
    records: Sequence[FullTreeNodeRecord],
    *,
    loss_fn: LossFn = squared_error,
    population_size: Optional[int] = None,
) -> float:
    samples = project_node_records_to_tree_samples(records, loss_fn=loss_fn, only_sampled=True)
    pop_n = int(population_size) if population_size is not None else int(len(records))
    if pop_n <= 0 or not samples:
        return float("nan")
    return float(horvitz_thompson_mean(samples, lambda s: float(s.preference_loss), float(pop_n)))


def sampled_hajek_node_mean_loss(
    records: Sequence[FullTreeNodeRecord],
    *,
    loss_fn: LossFn = squared_error,
) -> float:
    samples = project_node_records_to_tree_samples(records, loss_fn=loss_fn, only_sampled=True)
    if not samples:
        return float("nan")
    return float(hajek_estimate(samples, lambda s: float(s.preference_loss)))


def full_document_mean_loss(
    records: Sequence[DocumentLevelPredictionRecord],
    *,
    loss_fn: LossFn = squared_error,
) -> float:
    if not records:
        return float("nan")
    return float(np.mean([_document_loss_value(record, loss_fn) for record in records]))


def _breakdown_summary(
    records: Sequence[FullTreeNodeRecord],
    *,
    loss_fn: LossFn,
) -> Dict[str, Any]:
    samples = project_node_records_to_tree_samples(records, loss_fn=loss_fn, only_sampled=True)
    return {
        "population_count": int(len(records)),
        "sampled_count": int(sum(1 for record in records if bool(record.sampled))),
        "exact_mean_loss": exact_full_node_mean_loss(records, loss_fn=loss_fn),
        "naive_mean_loss": sampled_naive_node_mean_loss(records, loss_fn=loss_fn),
        "ht_mean_loss": sampled_ht_node_mean_loss(records, loss_fn=loss_fn),
        "hajek_mean_loss": sampled_hajek_node_mean_loss(records, loss_fn=loss_fn),
        "effective_sample_size": float(effective_sample_size(samples)) if samples else 0.0,
        "max_weight": float(max_weight(samples)) if samples else 0.0,
    }


def _estimator_error_summary(
    exact_value: float,
    estimate: float,
) -> Dict[str, float]:
    if not math.isfinite(float(exact_value)) or not math.isfinite(float(estimate)):
        return {
            "signed_error": float("nan"),
            "abs_error": float("nan"),
        }
    signed_error = float(estimate) - float(exact_value)
    return {
        "signed_error": float(signed_error),
        "abs_error": float(abs(signed_error)),
    }


def _document_vs_root_node_gap_summary(
    node_records: Sequence[FullTreeNodeRecord],
    document_records: Sequence[DocumentLevelPredictionRecord],
) -> Dict[str, float]:
    root_by_doc: Dict[str, FullTreeNodeRecord] = {}
    for record in node_records:
        if bool(record.is_root):
            root_by_doc[str(record.doc_id)] = record

    target_gaps: list[float] = []
    prediction_gaps: list[float] = []
    for document_record in document_records:
        root_record = root_by_doc.get(str(document_record.doc_id))
        if root_record is None:
            continue
        target_gaps.append(abs(float(document_record.target) - float(root_record.target)))
        prediction_gaps.append(
            abs(float(document_record.prediction) - float(root_record.loss_prediction))
        )

    return {
        "mean_target_gap": _finite_mean(target_gaps),
        "mean_prediction_gap": _finite_mean(prediction_gaps),
        "n_docs_with_root_pair": int(min(len(target_gaps), len(prediction_gaps))),
    }


def summarize_full_tree_ipw(
    node_records: Sequence[FullTreeNodeRecord],
    document_records: Sequence[DocumentLevelPredictionRecord],
    *,
    node_loss_fn: LossFn = squared_error,
    document_loss_fn: LossFn = squared_error,
) -> Dict[str, Any]:
    """Summarize the node-level estimand and the separate document-level loss."""

    samples = project_node_records_to_tree_samples(
        node_records,
        loss_fn=node_loss_fn,
        only_sampled=True,
    )
    type_groups: Dict[str, list[FullTreeNodeRecord]] = {}
    depth_groups: Dict[str, list[FullTreeNodeRecord]] = {}
    for record in node_records:
        bucket = "root" if bool(record.is_root) else str(record.node_type.value)
        type_groups.setdefault(bucket, []).append(record)
        depth_groups.setdefault(str(int(record.depth)), []).append(record)

    exact_mean_loss = exact_full_node_mean_loss(node_records, loss_fn=node_loss_fn)
    naive_mean_loss = sampled_naive_node_mean_loss(node_records, loss_fn=node_loss_fn)
    ht_mean_loss = sampled_ht_node_mean_loss(node_records, loss_fn=node_loss_fn)
    hajek_mean_loss = sampled_hajek_node_mean_loss(node_records, loss_fn=node_loss_fn)
    document_top_loss = full_document_mean_loss(document_records, loss_fn=document_loss_fn)
    document_top_mae = full_document_mean_loss(document_records, loss_fn=absolute_error)
    document_vs_root_gap = _document_vs_root_node_gap_summary(node_records, document_records)

    return {
        "population_size": int(len(node_records)),
        "sampled_nodes": int(len(samples)),
        "sampled_fraction": (
            float(len(samples)) / float(len(node_records)) if node_records else float("nan")
        ),
        "full_node_exact_mean_loss": exact_mean_loss,
        "sampled_node_naive_mean_loss": naive_mean_loss,
        "sampled_node_naive_signed_error": _estimator_error_summary(
            exact_mean_loss, naive_mean_loss
        )["signed_error"],
        "sampled_node_naive_abs_error": _estimator_error_summary(
            exact_mean_loss, naive_mean_loss
        )["abs_error"],
        "sampled_node_ht_mean_loss": ht_mean_loss,
        "sampled_node_ht_signed_error": _estimator_error_summary(
            exact_mean_loss, ht_mean_loss
        )["signed_error"],
        "sampled_node_ht_abs_error": _estimator_error_summary(
            exact_mean_loss, ht_mean_loss
        )["abs_error"],
        "sampled_node_hajek_mean_loss": hajek_mean_loss,
        "sampled_node_hajek_signed_error": _estimator_error_summary(
            exact_mean_loss, hajek_mean_loss
        )["signed_error"],
        "sampled_node_hajek_abs_error": _estimator_error_summary(
            exact_mean_loss, hajek_mean_loss
        )["abs_error"],
        "document_top_loss": document_top_loss,
        "document_top_mae": document_top_mae,
        "document_vs_root_node_target_gap_mae": float(document_vs_root_gap["mean_target_gap"]),
        "document_vs_root_node_prediction_gap_mae": float(
            document_vs_root_gap["mean_prediction_gap"]
        ),
        "document_vs_root_node_pair_count": int(document_vs_root_gap["n_docs_with_root_pair"]),
        "effective_sample_size": float(effective_sample_size(samples)) if samples else 0.0,
        "max_weight": float(max_weight(samples)) if samples else 0.0,
        "weight_sum": float(sum(sample.weight for sample in samples)),
        "corrected_local_law": corrected_local_law_node_summary(
            node_records,
            loss_fn=node_loss_fn,
            gamma_depth=1.0,
            local_law_weight=1.0,
        ),
        "type_breakdown": {
            str(name): _breakdown_summary(group, loss_fn=node_loss_fn)
            for name, group in sorted(type_groups.items())
        },
        "depth_breakdown": {
            str(name): _breakdown_summary(group, loss_fn=node_loss_fn)
            for name, group in sorted(depth_groups.items(), key=lambda item: int(item[0]))
        },
    }


def resample_full_tree_records(
    records: Sequence[FullTreeNodeRecord],
    *,
    propensity_fn: Callable[[FullTreeNodeRecord], float],
    rng: Optional[random.Random] = None,
) -> tuple[FullTreeNodeRecord, ...]:
    """Apply a Bernoulli node-sampling policy to a fully labeled node population."""

    local_rng = rng if rng is not None else random.Random()
    sampled_records = []
    for record in records:
        propensity = float(propensity_fn(record))
        if not math.isfinite(propensity):
            raise ValueError(f"propensity_fn returned a non-finite value for {record.node_id!r}")
        clipped = min(1.0, max(0.0, propensity))
        sampled_records.append(
            replace(
                record,
                sampled=bool(clipped > 0.0 and local_rng.random() < clipped),
                propensity=float(clipped),
            )
        )
    return tuple(sampled_records)


def _estimator_moments(values: Sequence[float], *, true_value: float) -> EstimatorMomentSummary:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        nan = float("nan")
        return EstimatorMomentSummary(mean=nan, bias=nan, variance=nan, rmse=nan)
    mean = float(np.mean(arr))
    bias = float(mean - float(true_value))
    variance = float(np.var(arr))
    rmse = float(np.sqrt(np.mean((arr - float(true_value)) ** 2)))
    return EstimatorMomentSummary(mean=mean, bias=bias, variance=variance, rmse=rmse)


def run_full_tree_estimator_monte_carlo(
    node_records: Sequence[FullTreeNodeRecord],
    document_records: Sequence[DocumentLevelPredictionRecord],
    *,
    propensity_fn: Callable[[FullTreeNodeRecord], float],
    n_trials: int,
    seed: int = 0,
    policy_name: str = "",
    node_loss_fn: LossFn = squared_error,
    document_loss_fn: LossFn = squared_error,
) -> FullTreeEstimatorMonteCarloSummary:
    """Estimate bias/variance/RMSE of naive, HT, and Hajek node estimators."""

    rng = random.Random(int(seed))
    true_full_node_mean = exact_full_node_mean_loss(node_records, loss_fn=node_loss_fn)
    document_top_loss = full_document_mean_loss(document_records, loss_fn=document_loss_fn)
    document_top_mae = full_document_mean_loss(document_records, loss_fn=absolute_error)

    naive_values: list[float] = []
    ht_values: list[float] = []
    hajek_values: list[float] = []
    sample_counts: list[float] = []
    ess_values: list[float] = []
    max_weights: list[float] = []

    for _ in range(int(max(0, n_trials))):
        sampled_records = resample_full_tree_records(
            node_records,
            propensity_fn=propensity_fn,
            rng=rng,
        )
        summary = summarize_full_tree_ipw(
            sampled_records,
            document_records,
            node_loss_fn=node_loss_fn,
            document_loss_fn=document_loss_fn,
        )
        naive_values.append(float(summary["sampled_node_naive_mean_loss"]))
        ht_values.append(float(summary["sampled_node_ht_mean_loss"]))
        hajek_values.append(float(summary["sampled_node_hajek_mean_loss"]))
        sample_counts.append(float(summary["sampled_nodes"]))
        ess_values.append(float(summary["effective_sample_size"]))
        max_weights.append(float(summary["max_weight"]))

    return FullTreeEstimatorMonteCarloSummary(
        policy_name=str(policy_name),
        trials=int(max(0, n_trials)),
        true_full_node_mean=float(true_full_node_mean),
        document_top_loss=float(document_top_loss),
        document_top_mae=float(document_top_mae),
        naive=_estimator_moments(naive_values, true_value=true_full_node_mean),
        ht=_estimator_moments(ht_values, true_value=true_full_node_mean),
        hajek=_estimator_moments(hajek_values, true_value=true_full_node_mean),
        mean_sample_count=_finite_mean(sample_counts),
        mean_effective_sample_size=_finite_mean(ess_values),
        mean_max_weight=_finite_mean(max_weights),
    )


__all__ = [
    "DocumentLevelPredictionRecord",
    "DEFAULT_LAYERED_RATE_GRID",
    "EstimatorMomentSummary",
    "FullTreeEstimatorMonteCarloSummary",
    "FullTreeIPWSummaryAccumulator",
    "FullTreeNodeRecord",
    "absolute_error",
    "classify_layered_sampling_regime",
    "corrected_local_law_node_summary",
    "document_record_from_state_tree",
    "exact_full_node_mean_loss",
    "full_document_mean_loss",
    "full_tree_node_records_from_state_tree",
    "layered_propensity_policy",
    "local_law_observations_from_state_tree",
    "project_node_records_to_tree_samples",
    "resample_full_tree_records",
    "run_full_tree_estimator_monte_carlo",
    "sampled_hajek_node_mean_loss",
    "sampled_ht_node_mean_loss",
    "sampled_naive_node_mean_loss",
    "squared_error",
    "summarize_full_tree_ipw",
]
