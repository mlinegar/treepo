"""
Training loop for CTreePO: learned mergeable sketches over multilingual embeddings.

Trains a CTreePOModel on manifesto documents with RILE supervision at the root
(empirical labels) and optionally at tree nodes (artifact labels, callbacks, or
online feedback).

Usage (from scripts/train_ctreepo.py):
    trainer = CTreePOTrainer(config)
    trainer.prepare_data(manifesto_ids)
    result = trainer.train()
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
except ImportError:
    raise ImportError("PyTorch required. Install with: pip install torch>=2.0.0")

from treepo._research.core.logged_supervision import (
    LoggedLabelObservation,
    ObservationUnitKind,
    SamplingMetadata,
    summarize_logged_observations,
    write_logged_observations_jsonl,
)
from treepo._research.tree.ctreepo_model import (
    CTreePOConfig,
    CTreePOModel,
    associativity_penalty,
    contrastive_loss,
    normalize_target,
    readout_aggregation_penalty,
)
from treepo._research.tree.embedding_tree import (
    EmbeddingTreeNode,
    build_tree_from_text,
    forward_ctreepo,
)
from treepo._research.tree.packed_execution import (
    PackedEmbeddingTree,
    PackedTreeBucketStore,
    PackedTreeBatch,
    PackedForwardResult,
    apply_runtime_mode_to_packed_trees,
    build_packed_tree_batch_from_bucket_store,
    build_packed_tree_bucket_stores,
    build_packed_embedding_tree,
    build_packed_tree_batch,
    forward_packed_tree_batch,
)
from treepo._research.core.provenance import (
    DATASET_SOURCE,
    normalize_truth_label_source,
)
from treepo._research.tree.compositional_learning import (
    CompositionalLearningProblemSpec,
    SupervisionDeliveryMode,
    shared_full_document_supervision_channel,
    shared_logged_substructure_observation,
    shared_protocol_problem_notes,
    shared_sampled_substructure_query_policy,
    shared_sampled_substructure_supervision_channel,
)
from treepo._research.training.tree_model_v2_trainer import (
    RealDocumentTaskAdapter,
    TreeModelV2Trainer,
    TreeModelV2TrainingConfig,
    TreeModelV2ObjectiveConfig,
    TreeModelV2ScoreTargetConfig,
)
from treepo._research.training.config_sections import (
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
    TrainConfig,
    ValidationConfig,
    config_to_dict,
)
from treepo._research.training.reproducibility import configure_reproducibility
from treepo._research.training.supervision.timing import (
    ACQUISITION_NONE,
    ACQUISITION_OFFLINE_ARTIFACT,
    ACQUISITION_SYNCHRONOUS_ORACLE,
    ACTIVATION_IMMEDIATE,
    ACTIVATION_TREE_PREP,
    CONSUMER_CTREEPO_GRADIENT,
    supervision_timing_contract,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class TreeOperatorDataConfig:
    window_size: int = 1200
    window_overlap: int = 150
    merge_drift_threshold: Optional[float] = None


@dataclass(frozen=True, kw_only=True)
class TreeOperatorObjectiveConfig:
    root_weight: float = 1.0
    merge_audit_weight: float = 0.5
    leaf_audit_weight: float = 0.0
    pseudo_weight: float = 0.1
    assoc_weight: float = 0.01
    idempotence_weight: float = 0.0
    contrastive_weight: float = 0.1
    consistency_weight: float = 0.05
    local_law_violation_threshold: float = 10.0
    fiber_pair_same_threshold: float = 10.0
    fiber_pair_diff_threshold: float = 30.0


@dataclass(frozen=True, kw_only=True)
class OnlineLocalLawSupervisionConfig:
    enabled: bool = False
    teacher_worker: bool = False
    worker_concurrency: int = 4


@dataclass(frozen=True, kw_only=True)
class LocalLawSupervisionConfig:
    require_local_law_supervision: bool = False
    n_audit: int = 5
    online: OnlineLocalLawSupervisionConfig = field(
        default_factory=OnlineLocalLawSupervisionConfig
    )


@dataclass(frozen=True, kw_only=True)
class TreeOperatorEvaluationConfig:
    uncertainty_z_score: float = 1.96
    min_interval_std: float = 0.5


@dataclass(frozen=True, kw_only=True)
class CTreePOTrainingConfig:
    """Sectioned training configuration for the C-TreePO tree operator."""

    model: CTreePOConfig = field(
        default_factory=lambda: CTreePOConfig(tree_model_version="v2")
    )
    data: TreeOperatorDataConfig = field(default_factory=TreeOperatorDataConfig)
    run: RunConfig = field(default_factory=RunConfig)
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(batch_size=4, epochs=50)
    )
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            learning_rate=1e-3,
            weight_decay=1e-4,
            optimizer="adamw",
            scheduler="cosine",
            min_learning_rate=1e-5,
            warmup_epochs=3,
            grad_clip_norm=1.0,
        )
    )
    validation: ValidationConfig = field(
        default_factory=lambda: ValidationConfig(
            eval_every=5,
            early_stopping_patience=12,
        )
    )
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    objective: TreeOperatorObjectiveConfig = field(
        default_factory=TreeOperatorObjectiveConfig
    )
    supervision: LocalLawSupervisionConfig = field(
        default_factory=LocalLawSupervisionConfig
    )
    evaluation: TreeOperatorEvaluationConfig = field(
        default_factory=TreeOperatorEvaluationConfig
    )


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Result of a CTreePO training run."""

    config: Dict[str, Any] = field(default_factory=dict)
    train_losses: List[float] = field(default_factory=list)
    eval_metrics: List[Dict[str, Any]] = field(default_factory=list)
    best_epoch: int = 0
    best_root_mae: float = float("inf")
    stopped_early: bool = False
    epochs_completed: int = 0
    training_time_seconds: float = 0.0
    local_law_summary: Dict[str, Any] = field(default_factory=dict)
    compositional_learning_problem: Dict[str, Any] = field(default_factory=dict)
    logged_observation_artifacts: Dict[str, Any] = field(default_factory=dict)
    reproducibility: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Eval metrics
# ---------------------------------------------------------------------------


@dataclass
class CTreePOEvalMetrics:
    epoch: int
    root_mae: float              # MAE on original RILE scale [-100, +100]
    root_mse: float
    root_mae_normalized: float
    interval_coverage_95: float
    interval_mean_width_95: float
    confidence_calibration_error: float
    node_oracle_label_rate: float
    node_oracle_mae: float
    leaf_oracle_mae: float
    merge_oracle_mae: float
    leaf_violation_rate: float
    merge_violation_rate: float
    leaf_oracle_count: int
    merge_oracle_count: int
    n_docs: int
    per_doc: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class CTreePOTrainer:
    """Trains a CTreePOModel on manifesto data with embedding trees."""

    def __init__(
        self,
        config: CTreePOTrainingConfig,
        embedding_client: Any = None,
        node_oracle_predictor: Optional[Callable[[str], float]] = None,
        node_oracle_source_kind: str = "none",
        node_oracle_source_spec: Optional[str] = None,
        online_node_oracle_queue: Optional[Any] = None,
        online_teacher_worker: bool = False,
        online_worker_concurrency: int = 4,
    ):
        self.config = config
        self.reproducibility = configure_reproducibility(int(config.run.seed))
        self.embedding_client = embedding_client
        self.node_oracle_predictor = node_oracle_predictor
        self.node_oracle_source_kind = str(node_oracle_source_kind or "none").strip().lower() or "none"
        self.node_oracle_source_spec = (
            str(node_oracle_source_spec).strip() if node_oracle_source_spec else None
        )
        self.online_node_oracle_queue = online_node_oracle_queue
        self.online_teacher_worker = bool(online_teacher_worker)
        self.online_worker_concurrency = int(max(1, online_worker_concurrency))
        self._online_worker_executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=1)
            if bool(online_teacher_worker) and online_node_oracle_queue is not None
            else None
        )
        self._online_worker_future: Optional[Future] = None
        self.model = CTreePOModel(config.model)
        self.device = self._resolve_device(config.runtime.device)
        self.model.to(self.device)
        self.rng = random.Random(config.run.seed)
        self._node_oracle_cache: Dict[str, float] = {}
        self._warned_missing_leaf_labels = False
        self._warned_missing_merge_labels = False
        self._train_logged_node_observations: List[LoggedLabelObservation[Any]] = []
        self._val_logged_node_observations: List[LoggedLabelObservation[Any]] = []
        self._use_shared_fiber_constraints = bool(
            getattr(self.model, "uses_tree_model_v2", False)
            and getattr(self.model, "has_phi", False)
        )
        self._task_adapter = RealDocumentTaskAdapter(
            max_leaf_targets_per_doc=(
                int(config.supervision.n_audit)
                if float(config.objective.leaf_audit_weight) > 0.0
                else 0
            ),
            max_internal_targets_per_doc=(
                int(config.supervision.n_audit)
                if float(config.objective.merge_audit_weight) > 0.0
                else 0
            ),
            rng=self.rng,
            enable_fiber_constraints=self._use_shared_fiber_constraints,
            fiber_same_threshold=float(config.objective.fiber_pair_same_threshold),
            fiber_diff_threshold=float(config.objective.fiber_pair_diff_threshold),
        )
        self._shared_trainer = TreeModelV2Trainer(
            model=self.model,
            adapter=self._task_adapter,
            forward_batch=self._forward_shared_batch,
            state_getter=self._node_state_from_batch_item,
            config=TreeModelV2TrainingConfig(
                score_targets=TreeModelV2ScoreTargetConfig(
                    target_min=float(config.model.target_min),
                    target_max=float(config.model.target_max),
                ),
                objective=TreeModelV2ObjectiveConfig(
                    root_weight=float(config.objective.root_weight),
                    leaf_scalar_weight=float(config.objective.leaf_audit_weight),
                    internal_scalar_weight=float(config.objective.merge_audit_weight),
                    fiber_same_weight=(
                        float(config.objective.contrastive_weight)
                        if self._use_shared_fiber_constraints
                        else 0.0
                    ),
                    fiber_different_weight=(
                        float(config.objective.contrastive_weight)
                        if self._use_shared_fiber_constraints
                        else 0.0
                    ),
                ),
            ),
            device=self.device,
        )
        self._last_shared_supervision_stats: Dict[str, Any] = {}
        self._last_wrapper_regularization_stats: Dict[str, Any] = {}
        self._last_train_step_stats: Dict[str, Any] = {}
        self._last_eval_runtime_stats: Dict[str, Any] = {}
        self._split_runtime_metadata: Dict[str, Dict[str, Any]] = {"train": {}, "val": {}}
        self._packed_tree_by_nodes_id: Dict[int, PackedEmbeddingTree] = {}
        self._packed_bucket_stores_by_split: Dict[str, Tuple[PackedTreeBucketStore, ...]] = {
            "train": tuple(),
            "val": tuple(),
        }
        self._packed_bucket_store_by_nodes_id: Dict[int, PackedTreeBucketStore] = {}
        self._packed_batch_cache: OrderedDict[Tuple[str, Tuple[int, ...]], PackedTreeBatch] = OrderedDict()
        self._packed_batch_cache_max_entries: int = 32

        # Pre-built trees (cached after prepare_data)
        self.train_trees: List[Tuple[List[EmbeddingTreeNode], float, str]] = []  # (nodes, rile, doc_id)
        self.val_trees: List[Tuple[List[EmbeddingTreeNode], float, str]] = []

    def _online_queue_enabled(self) -> bool:
        return self.online_node_oracle_queue is not None

    def _online_enqueue_epoch_requests(self, *, split: str, epoch: int) -> Dict[str, Any]:
        if not self._online_queue_enabled():
            return {}
        trees = self.train_trees if split == "train" else self.val_trees
        if not trees:
            return {}
        return self.online_node_oracle_queue.enqueue_epoch_requests(
            trees,
            split=split,
            epoch=int(epoch),
        )

    def _online_attach_completed_labels(self, *, split: str) -> Dict[str, Any]:
        if not self._online_queue_enabled():
            return {}
        trees = self.train_trees if split == "train" else self.val_trees
        if not trees:
            return {}
        result = self.online_node_oracle_queue.attach_completed(
            trees,
            split=split,
            truth_label_source=self.node_oracle_source_kind,
        )
        if result.observations:
            if split == "train":
                self._train_logged_node_observations.extend(result.observations)
            else:
                self._val_logged_node_observations.extend(result.observations)
        return result.to_dict()

    def _online_run_teacher_worker(self) -> Dict[str, Any]:
        if not self._online_queue_enabled() or not self.online_teacher_worker:
            return {}
        if self.node_oracle_predictor is None:
            return {"processed": 0, "submitted": 0, "failed": 0, "reason": "no_node_oracle_predictor"}
        return self.online_node_oracle_queue.run_teacher_worker(
            self.node_oracle_predictor,
            concurrency=self.online_worker_concurrency,
        )

    def _online_collect_worker_result(self, *, timeout_seconds: float = 0.0) -> Dict[str, Any]:
        if self._online_worker_future is None:
            return {}
        future = self._online_worker_future
        if not future.done():
            if float(timeout_seconds) <= 0.0:
                return {}
            try:
                result = future.result(timeout=float(timeout_seconds))
            except FutureTimeoutError:
                return {}
            except Exception as exc:
                self._online_worker_future = None
                logger.warning("Online node oracle worker failed: %s", exc)
                return {"processed": 0, "submitted": 0, "failed": 1, "error": str(exc)}
            self._online_worker_future = None
            return dict(result)
        self._online_worker_future = None
        try:
            return dict(future.result())
        except Exception as exc:
            logger.warning("Online node oracle worker failed: %s", exc)
            return {"processed": 0, "submitted": 0, "failed": 1, "error": str(exc)}

    def _online_start_teacher_worker(self) -> Dict[str, Any]:
        if not self._online_queue_enabled() or not self.online_teacher_worker:
            return {}
        if self._online_worker_future is not None and not self._online_worker_future.done():
            return {"worker_running": True}
        self._online_collect_worker_result()
        if self._online_worker_executor is None:
            self._online_worker_executor = ThreadPoolExecutor(max_workers=1)
        self._online_worker_future = self._online_worker_executor.submit(
            self._online_run_teacher_worker
        )
        return {"worker_started": True}

    def _local_law_supervision_timing(
        self,
        *,
        labeled_leaves: int = 0,
        labeled_internal: int = 0,
    ) -> Dict[str, Any]:
        """Describe how node-level local-law labels are acquired and activated."""

        if self._online_queue_enabled():
            timing = dict(self.online_node_oracle_queue.timing_contract())
            metadata = dict(timing.get("metadata", {}) or {})
            metadata.update(
                {
                    "active_labeled_leaves": int(labeled_leaves),
                    "active_labeled_internal": int(labeled_internal),
                    "teacher_worker_enabled": bool(self.online_teacher_worker),
                }
            )
            timing["metadata"] = metadata
            return timing

        if self.node_oracle_predictor is not None:
            return supervision_timing_contract(
                acquisition_policy=ACQUISITION_SYNCHRONOUS_ORACLE,
                activation_barrier=ACTIVATION_TREE_PREP,
                consumer=CONSUMER_CTREEPO_GRADIENT,
                producer=str(self.node_oracle_source_kind or "oracle_callback"),
                delivery_mode="direct_callback",
                blocking=True,
                notes=(
                    "Tree preparation calls the node oracle synchronously.",
                    "Collected labels are active before the first optimization epoch.",
                ),
                metadata={
                    "source_spec": self.node_oracle_source_spec,
                    "active_labeled_leaves": int(labeled_leaves),
                    "active_labeled_internal": int(labeled_internal),
                },
            )

        if int(labeled_leaves) > 0 or int(labeled_internal) > 0:
            return supervision_timing_contract(
                acquisition_policy=ACQUISITION_OFFLINE_ARTIFACT,
                activation_barrier=ACTIVATION_IMMEDIATE,
                consumer=CONSUMER_CTREEPO_GRADIENT,
                producer=str(self.node_oracle_source_kind or "artifact"),
                delivery_mode="loaded_tree_nodes",
                blocking=False,
                notes=(
                    "Node labels were already present on loaded tree artifacts.",
                    "No runtime oracle query is needed for these labels.",
                ),
                metadata={
                    "active_labeled_leaves": int(labeled_leaves),
                    "active_labeled_internal": int(labeled_internal),
                },
            )

        return supervision_timing_contract(
            acquisition_policy=ACQUISITION_NONE,
            activation_barrier=ACTIVATION_IMMEDIATE,
            consumer=CONSUMER_CTREEPO_GRADIENT,
            producer="none",
            delivery_mode="none",
            blocking=False,
            notes=("No node-level local-law supervision is active.",),
        )

    def _forward_shared_batch(
        self,
        model: CTreePOModel,
        batch_trees: Sequence[Tuple[List[EmbeddingTreeNode], float, str]],
    ) -> PackedForwardResult:
        packed_batch = self._packed_batch_for_items(batch_trees)
        return forward_packed_tree_batch(
            model,
            packed_batch,
            materialize_nodes=False,
        )

    @staticmethod
    def _node_state_from_batch_item(
        batch_item: Tuple[List[EmbeddingTreeNode], float, str],
        node_index: int,
    ) -> Optional[torch.Tensor]:
        nodes, _rile, _doc_id = batch_item
        if node_index < 0 or node_index >= len(nodes):
            return None
        return nodes[node_index].sketch

    @staticmethod
    def _tree_batch_has_oracle_labels(
        batch_trees: Sequence[Tuple[List[EmbeddingTreeNode], float, str]],
    ) -> tuple[bool, bool]:
        has_any_leaf_labels = False
        has_any_merge_labels = False
        for nodes, _rile, _doc_id in batch_trees:
            if any(node.is_leaf and "rile" in node.oracle_scores for node in nodes):
                has_any_leaf_labels = True
            if any((not node.is_leaf) and "rile" in node.oracle_scores for node in nodes):
                has_any_merge_labels = True
        return has_any_leaf_labels, has_any_merge_labels

    def _maybe_warn_missing_local_law_labels(
        self,
        *,
        has_any_leaf_labels: bool,
        has_any_merge_labels: bool,
    ) -> None:
        cfg = self.config
        if (
            float(cfg.objective.leaf_audit_weight) > 0.0
            and not has_any_leaf_labels
            and not self._warned_missing_leaf_labels
        ):
            logger.warning(
                "CTreePO leaf local-law weight is positive, but no leaf oracle scores were attached. "
                "Leaf preservation supervision is inactive."
            )
            self._warned_missing_leaf_labels = True
        if (
            float(cfg.objective.merge_audit_weight) > 0.0
            and not has_any_merge_labels
            and not self._warned_missing_merge_labels
        ):
            logger.warning(
                "CTreePO merge local-law weight is positive, but no internal-node oracle scores were attached. "
                "Merge preservation supervision is inactive."
            )
            self._warned_missing_merge_labels = True

    @staticmethod
    def _collect_root_batch(
        batch_trees: Sequence[Tuple[List[EmbeddingTreeNode], float, str]],
    ) -> tuple[List[torch.Tensor], List[float]]:
        root_sketches: List[torch.Tensor] = []
        root_targets: List[float] = []
        for nodes, rile, _doc_id in batch_trees:
            if not nodes:
                continue
            root = nodes[-1].sketch
            if root is None:
                raise ValueError("root sketch not available")
            root_sketches.append(root)
            root_targets.append(float(rile))
        return root_sketches, root_targets

    @staticmethod
    def _collect_root_batch_from_forward(
        batch_trees: Sequence[Tuple[List[EmbeddingTreeNode], float, str]],
        forward_result: PackedForwardResult,
    ) -> tuple[List[torch.Tensor], List[float]]:
        root_states = forward_result.state_batch.index_select(0, forward_result.root_indices)
        root_targets = [float(rile) for _nodes, rile, _doc_id in batch_trees]
        return [root_states[idx] for idx in range(int(root_states.shape[0]))], root_targets

    def _consistency_regularization_loss(
        self,
        batch_trees: Sequence[Tuple[List[EmbeddingTreeNode], float, str]],
        *,
        forward_result: Optional[PackedForwardResult] = None,
    ) -> tuple[torch.Tensor, int, Dict[str, Any]]:
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        weight = float(self.config.objective.consistency_weight)
        stats: Dict[str, Any] = {
            "consistency_regularization_active": bool(weight > 0.0),
            "consistency_term_count": 0,
        }
        if weight <= 0.0:
            return zero, 0, stats

        total = zero.clone()
        n_terms = 0
        if forward_result is not None:
            for level in forward_result.packed_batch.levels:
                if int(level.parent_indices.numel()) <= 0:
                    continue
                parent_states = forward_result.state_batch.index_select(0, level.parent_indices)
                left_states = forward_result.state_batch.index_select(0, level.left_indices)
                right_states = forward_result.state_batch.index_select(0, level.right_indices)
                parent_pred = self.model.predict_normalized_batch(parent_states, head="rile").reshape(-1)
                left_pred = self.model.predict_normalized_batch(left_states, head="rile").reshape(-1)
                right_pred = self.model.predict_normalized_batch(right_states, head="rile").reshape(-1)
                left_weight = level.left_weights.to(device=parent_pred.device, dtype=parent_pred.dtype)
                expected = left_weight * left_pred + (1.0 - left_weight) * right_pred
                total = total + weight * ((parent_pred - expected) ** 2).sum()
                n_terms += int(parent_pred.numel())
            stats["consistency_term_count"] = int(n_terms)
            if n_terms > 0:
                stats["consistency_regularization_loss"] = float(total.detach().cpu().item())
            return total, n_terms, stats

        for nodes, _rile, _doc_id in batch_trees:
            for node in nodes:
                if not node.is_leaf and node.children is not None:
                    left_idx, right_idx = node.children
                    if left_idx == right_idx:
                        continue
                    left_node = nodes[left_idx]
                    right_node = nodes[right_idx]
                    left_len = left_node.text_len
                    right_len = right_node.text_len
                    left_weight = left_len / max(left_len + right_len, 1)
                    c_loss = readout_aggregation_penalty(
                        self.model,
                        node.sketch,
                        left_node.sketch,
                        right_node.sketch,
                        left_weight=left_weight,
                        head="rile",
                    )
                    total = total + weight * c_loss
                    n_terms += 1

        stats["consistency_term_count"] = int(n_terms)
        if n_terms > 0:
            stats["consistency_regularization_loss"] = float(total.detach().cpu().item())
        return total, n_terms, stats

    def _associativity_regularization_loss(
        self,
        batch_trees: Sequence[Tuple[List[EmbeddingTreeNode], float, str]],
        *,
        forward_result: Optional[PackedForwardResult] = None,
    ) -> tuple[torch.Tensor, int, Dict[str, Any]]:
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        weight = float(self.config.objective.assoc_weight)
        stats: Dict[str, Any] = {
            "associativity_regularization_active": bool(weight > 0.0),
            "associativity_term_count": 0,
        }
        if weight <= 0.0:
            return zero, 0, stats

        total = zero.clone()
        n_terms = 0
        if forward_result is not None:
            for tree_index, tree in enumerate(forward_result.packed_batch.trees):
                if int(tree.leaf_count) < 3:
                    continue
                start = int(forward_result.node_offsets[tree_index])
                leaf_indices = torch.arange(
                    start,
                    start + int(tree.leaf_count),
                    dtype=torch.long,
                    device=forward_result.state_batch.device,
                )
                leaf_sketches = list(forward_result.state_batch.index_select(0, leaf_indices))
                a_loss = associativity_penalty(self.model, leaf_sketches, n_triplets=4)
                total = total + weight * a_loss
                n_terms += 1
            stats["associativity_term_count"] = int(n_terms)
            if n_terms > 0:
                stats["associativity_regularization_loss"] = float(total.detach().cpu().item())
            return total, n_terms, stats

        for nodes, _rile, _doc_id in batch_trees:
            leaf_sketches = [node.sketch for node in nodes if node.is_leaf and node.sketch is not None]
            if len(leaf_sketches) >= 3:
                a_loss = associativity_penalty(self.model, leaf_sketches, n_triplets=4)
                total = total + weight * a_loss
                n_terms += 1

        stats["associativity_term_count"] = int(n_terms)
        if n_terms > 0:
            stats["associativity_regularization_loss"] = float(total.detach().cpu().item())
        return total, n_terms, stats

    def _idempotence_regularization_loss(
        self,
        batch_trees: Sequence[Tuple[List[EmbeddingTreeNode], float, str]],
        *,
        forward_result: Optional[PackedForwardResult] = None,
    ) -> tuple[torch.Tensor, int, Dict[str, Any]]:
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        weight = float(self.config.objective.idempotence_weight)
        stats: Dict[str, Any] = {
            "idempotence_regularization_active": bool(weight > 0.0),
            "idempotence_term_count": 0,
            "idempotence_proxy_only": True,
        }
        if weight <= 0.0:
            return zero, 0, stats

        states: Optional[torch.Tensor] = None
        if forward_result is not None and int(forward_result.state_batch.numel()) > 0:
            states = forward_result.state_batch
        else:
            sketches: List[torch.Tensor] = []
            for nodes, _rile, _doc_id in batch_trees:
                sketches.extend(
                    node.sketch for node in nodes if node.sketch is not None
                )
            if sketches:
                states = torch.stack(sketches, dim=0).to(device=self.device)

        if states is None or int(states.shape[0]) <= 0:
            return zero, 0, stats

        merged = self.model.merge_batch(states, states)
        row_losses = torch.mean((merged - states) ** 2, dim=-1)
        total = weight * row_losses.sum()
        n_terms = int(row_losses.numel())
        stats["idempotence_term_count"] = int(n_terms)
        stats["idempotence_regularization_loss"] = float(total.detach().cpu().item())
        stats["idempotence_mean_squared_distance"] = float(
            row_losses.detach().mean().cpu().item()
        )
        return total, n_terms, stats

    def _legacy_contrastive_regularization_loss(
        self,
        root_sketches: Sequence[torch.Tensor],
        root_targets: Sequence[float],
    ) -> tuple[torch.Tensor, int, Dict[str, Any]]:
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        active = bool(
            float(self.config.objective.contrastive_weight) > 0.0
            and len(root_sketches) >= 2
            and not bool(getattr(self.model, "uses_tree_model_v2", False))
        )
        stats: Dict[str, Any] = {
            "legacy_sketch_contrastive_active": active,
            "legacy_sketch_contrastive_term_count": 0,
        }
        if not active:
            return zero, 0, stats

        raw_loss = contrastive_loss(
            list(root_sketches),
            list(root_targets),
            tau=0.1,
            similarity_threshold=15.0,
        )
        weighted_loss = float(self.config.objective.contrastive_weight) * raw_loss
        stats["legacy_sketch_contrastive_term_count"] = 1
        stats["legacy_sketch_contrastive_loss"] = float(weighted_loss.detach().cpu().item())
        return weighted_loss, 1, stats

    def _wrapper_regularization_loss(
        self,
        batch_trees: Sequence[Tuple[List[EmbeddingTreeNode], float, str]],
        root_sketches: Sequence[torch.Tensor],
        root_targets: Sequence[float],
        *,
        forward_result: Optional[PackedForwardResult] = None,
    ) -> tuple[torch.Tensor, int, Dict[str, Any]]:
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        total = zero.clone()
        n_terms = 0
        stats: Dict[str, Any] = {}

        for helper_output in (
            self._consistency_regularization_loss(
                batch_trees,
                forward_result=forward_result,
            ),
            self._associativity_regularization_loss(
                batch_trees,
                forward_result=forward_result,
            ),
            self._idempotence_regularization_loss(
                batch_trees,
                forward_result=forward_result,
            ),
            self._legacy_contrastive_regularization_loss(root_sketches, root_targets),
        ):
            helper_loss, helper_terms, helper_stats = helper_output
            total = total + helper_loss
            n_terms += int(helper_terms)
            stats.update(dict(helper_stats))

        stats["wrapper_regularization_term_count"] = int(n_terms)
        if n_terms > 0:
            stats["wrapper_regularization_loss"] = float(total.detach().cpu().item())
        return total, n_terms, stats

    def _predict_node_oracle_score(self, text: str) -> Optional[float]:
        if self.node_oracle_predictor is None:
            return None
        rendered = str(text or "")
        if not rendered.strip():
            return None
        cached = self._node_oracle_cache.get(rendered)
        if cached is not None:
            return float(cached)
        try:
            score = float(self.node_oracle_predictor(rendered))
        except Exception as exc:
            logger.warning("Node oracle predictor failed on span length=%d: %s", len(rendered), exc)
            return None
        self._node_oracle_cache[rendered] = score
        return score

    def _label_tree_nodes_with_oracle_scores(
        self,
        nodes: List[EmbeddingTreeNode],
        *,
        doc_id: str,
        split: str,
    ) -> Dict[str, int]:
        leaf_count = 0
        merge_count = 0
        if self.node_oracle_predictor is None:
            return {"leaf": 0, "merge": 0, "total": 0}

        observations: List[LoggedLabelObservation[Any]] = []

        for idx, node in enumerate(nodes):
            score = self._predict_node_oracle_score(node.text_span)
            if score is None:
                continue
            node.oracle_scores["rile"] = float(score)
            if node.is_leaf:
                leaf_count += 1
                unit_kind = ObservationUnitKind.LEAF
            else:
                merge_count += 1
                unit_kind = ObservationUnitKind.INTERNAL
            observations.append(
                shared_logged_substructure_observation(
                    document_id=str(doc_id),
                    unit_id=f"node_{idx}",
                    unit_kind=unit_kind,
                    label=float(score),
                    application_name="ctreepo_local_law_training",
                    supervision_signal_name="node_oracle_score",
                    truth_label_source=self.node_oracle_source_kind,
                    sampling=SamplingMetadata(
                        document_propensity=1.0,
                        unit_propensity=1.0,
                        label_propensity=1.0,
                        sampling_scheme="all_observed_tree_nodes",
                        policy_name="sampled_substructure_query_policy",
                        unit_kind=unit_kind,
                        supports_ipw_estimation=False,
                    ),
                    context={
                        "char_start": int(node.char_start),
                        "char_end": int(node.char_end),
                        "text_span": str(node.text_span),
                    },
                )
            )

        if split == "train":
            self._train_logged_node_observations = [
                obs for obs in self._train_logged_node_observations
                if obs.document_id != str(doc_id)
            ] + observations
        else:
            self._val_logged_node_observations = [
                obs for obs in self._val_logged_node_observations
                if obs.document_id != str(doc_id)
            ] + observations

        total = leaf_count + merge_count
        if total > 0:
            logger.info(
                "Labeled %s with node oracle scores: leaves=%d internal=%d",
                doc_id,
                leaf_count,
                merge_count,
            )
        return {"leaf": leaf_count, "merge": merge_count, "total": total}

    def _tree_local_law_summary(
        self,
        trees: Sequence[Tuple[List[EmbeddingTreeNode], float, str]],
    ) -> Dict[str, Any]:
        capability_report = (
            self.model.capability_report().to_dict()
            if hasattr(self.model, "capability_report")
            else None
        )
        total_nodes = 0
        total_leaves = 0
        total_internal = 0
        labeled_leaves = 0
        labeled_internal = 0

        for nodes, _rile, _doc_id in trees:
            total_nodes += len(nodes)
            for node in nodes:
                if node.is_leaf:
                    total_leaves += 1
                    if "rile" in node.oracle_scores:
                        labeled_leaves += 1
                else:
                    total_internal += 1
                    if "rile" in node.oracle_scores:
                        labeled_internal += 1

        supervision_timing = self._local_law_supervision_timing(
            labeled_leaves=int(labeled_leaves),
            labeled_internal=int(labeled_internal),
        )
        compositional_learning_problem = self._compositional_learning_problem(
            total_leaves=int(total_leaves),
            total_internal=int(total_internal),
            labeled_leaves=int(labeled_leaves),
            labeled_internal=int(labeled_internal),
            operator_capabilities=capability_report,
        )
        compositional_learning_problem["supervision_timing"] = dict(supervision_timing)
        logged_observations = (
            list(self._train_logged_node_observations)
            + list(self._val_logged_node_observations)
        )
        logged_summary = (
            summarize_logged_observations(logged_observations)
            if logged_observations
            else {
                "count": 0,
                "unit_kinds": [],
                "supports_ipw_estimation": False,
                "joint_propensity_min": 1.0,
                "joint_propensity_max": 1.0,
                "joint_propensity_mean": 1.0,
            }
        )
        root_supervision_source = (
            "labeled_tree_artifact"
            if str(self.node_oracle_source_kind) == "labeled_tree"
            else "empirical_root_labels"
        )
        if self._online_queue_enabled():
            node_supervision_source = "online_feedback_queue"
        elif self.node_oracle_predictor is not None:
            node_supervision_source = str(self.node_oracle_source_kind or "live_oracle")
        elif int(labeled_leaves) > 0 or int(labeled_internal) > 0:
            node_supervision_source = "labeled_tree_artifact"
        else:
            node_supervision_source = "none"
        overall_supervision_source = (
            root_supervision_source
            if node_supervision_source == "none"
            else "mixed_root_and_node"
        )

        return {
            "supervision_source": overall_supervision_source,
            "root_supervision_source": root_supervision_source,
            "node_supervision_source": node_supervision_source,
            "node_oracle_predictor_attached": bool(self.node_oracle_predictor is not None),
            "node_label_source_kind": str(self.node_oracle_source_kind),
            "node_label_source_spec": self.node_oracle_source_spec,
            "online_node_oracle_queue": (
                self.online_node_oracle_queue.summary()
                if self._online_queue_enabled()
                else {"enabled": False}
            ),
            "supervision_timing": dict(supervision_timing),
            "total_nodes": int(total_nodes),
            "total_leaves": int(total_leaves),
            "total_internal": int(total_internal),
            "labeled_leaves": int(labeled_leaves),
            "labeled_internal": int(labeled_internal),
            "leaf_label_rate": (
                float(labeled_leaves) / float(total_leaves) if total_leaves > 0 else 0.0
            ),
            "merge_label_rate": (
                float(labeled_internal) / float(total_internal) if total_internal > 0 else 0.0
            ),
            "requested_weights": {
                "root_weight": float(self.config.objective.root_weight),
                "leaf_audit_weight": float(self.config.objective.leaf_audit_weight),
                "merge_audit_weight": float(self.config.objective.merge_audit_weight),
                "consistency_weight": float(self.config.objective.consistency_weight),
                "assoc_weight": float(self.config.objective.assoc_weight),
                "idempotence_weight": float(self.config.objective.idempotence_weight),
                "contrastive_weight": float(self.config.objective.contrastive_weight),
            },
            "operator_capabilities": capability_report,
            "require_local_law_supervision": bool(
                self.config.supervision.require_local_law_supervision
            ),
            "objective": {
                "root_supervision": bool(float(self.config.objective.root_weight) > 0.0),
                "leaf_supervision": bool(
                    float(self.config.objective.leaf_audit_weight) > 0.0 and labeled_leaves > 0
                ),
                "merge_supervision": bool(
                    float(self.config.objective.merge_audit_weight) > 0.0 and labeled_internal > 0
                ),
                "idempotence_supervision": False,
                "proxy_idempotence_penalty": bool(
                    float(self.config.objective.idempotence_weight) > 0.0
                ),
                "proxy_readout_aggregation_penalty": bool(
                    float(self.config.objective.consistency_weight) > 0.0
                ),
                "proxy_associativity_penalty": bool(
                    float(self.config.objective.assoc_weight) > 0.0
                ),
            },
            "violation_threshold_raw": float(
                self.config.objective.local_law_violation_threshold
            ),
            "compositional_learning_problem": compositional_learning_problem,
            "logged_observations_summary": logged_summary,
            "logged_observation_artifacts": {},
            "notes": [
                "Root scalar supervision is valid on its own; node labels only activate local-law losses and node-level metrics.",
                "Leaf/internal node supervision is only active when node-span oracle labels are attached.",
                "Preferred local-law path: task-provided exact span oracle via --local-law-oracle task, or an explicit mechanical callback via --local-law-oracle.",
                "Model-backed teacher labeling is a fallback label source, not a requirement for neural-operator training.",
                "CTreePO remains a proxy-only operator for C2/L3: idempotence_weight penalizes ||g(z,z)-z||^2 in latent space, not theorem-domain decode/resummary equality.",
                "See operator_capabilities for the architecture-level theorem/proxy split.",
            ],
        }

    def _compositional_learning_problem(
        self,
        *,
        total_leaves: int,
        total_internal: int,
        labeled_leaves: int,
        labeled_internal: int,
        operator_capabilities: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        capability_report = (
            self.model.capability_report()
            if hasattr(self.model, "capability_report")
            else None
        )
        sampled_active = bool(labeled_leaves > 0 or labeled_internal > 0)
        sampled_label_source = normalize_truth_label_source(
            self.node_oracle_source_kind,
            default=(
                "oracle"
                if bool(self.node_oracle_predictor is not None or self._online_queue_enabled())
                else "unknown"
            ),
        )
        targeted_laws: List[Any] = []
        if float(self.config.objective.leaf_audit_weight) > 0.0:
            from treepo._research.core.ops_checks import LawKind

            targeted_laws.append(LawKind.L1_LEAF)
        if float(self.config.objective.merge_audit_weight) > 0.0:
            from treepo._research.core.ops_checks import LawKind

            targeted_laws.append(LawKind.L2_MERGE)

        problem = CompositionalLearningProblemSpec(
            name="ctreepo_local_law_training",
            document_type_name="documents",
            theorem_domain_name="span_summary_objects",
            operator_name=(
                capability_report.operator_name
                if capability_report is not None
                else type(self.model).__name__
            ),
            operator_capabilities=capability_report,
            supervision_channels=(
                shared_full_document_supervision_channel(
                    active=bool(float(self.config.objective.root_weight) > 0.0),
                    label_source=DATASET_SOURCE,
                    notes=(
                        "Whole-document labels supervise the root prediction directly in this application.",
                    ),
                ),
                shared_sampled_substructure_supervision_channel(
                    active=sampled_active,
                    label_source=sampled_label_source,
                    delivery_mode=(
                        SupervisionDeliveryMode.ONLINE_ORACLE_QUERY
                        if bool(self.node_oracle_predictor is not None or self._online_queue_enabled())
                        else SupervisionDeliveryMode.OFFLINE_LOGGED
                    ),
                    query_policy=(
                        shared_sampled_substructure_query_policy(
                            selection_strategy=(
                                "budgeted_random_node_feedback"
                                if self._online_queue_enabled()
                                else "all_observed_tree_nodes"
                            ),
                            adaptive=bool(self._online_queue_enabled()),
                            budget={
                                "observed_labeled_leaves": int(labeled_leaves),
                                "observed_labeled_internal": int(labeled_internal),
                            },
                            propensity_field_name="propensity",
                            logs_realized_propensities=bool(self._online_queue_enabled()),
                            supports_ipw_estimation=bool(self._online_queue_enabled()),
                            notes=(
                                (
                                    "Online mode queues sampled node-label requests through FeedbackStore and attaches completed labels at epoch boundaries."
                                    if self._online_queue_enabled()
                                    else "Current trainer queries the attached node oracle callback while preparing trees."
                                ),
                            ),
                        )
                        if bool(self.node_oracle_predictor is not None or self._online_queue_enabled())
                        else None
                    ),
                    targeted_laws=tuple(targeted_laws),
                    requires_propensity_logging=bool(self._online_queue_enabled()),
                    supports_unbiased_risk=bool(self._online_queue_enabled()),
                    notes=(
                        "Leaf and internal-node labels are attached to sampled tree units during training.",
                        "Current trainer uses sampled node supervision for optimization but does not persist per-sample propensities.",
                        "Idempotence is not directly supervised because there is no theorem-domain decode/resummary path.",
                    ),
                ),
            ),
            notes=shared_protocol_problem_notes(
                application_name="ctreepo_local_law_training",
                notes=(
                f"observed_labeled_leaves={int(labeled_leaves)}/{int(total_leaves)}",
                f"observed_labeled_internal={int(labeled_internal)}/{int(total_internal)}",
                "This spec is backend-agnostic: it records supervision channels separately from theorem-backed operator assumptions.",
                ),
            ),
        )
        payload = problem.to_dict()
        if operator_capabilities is not None and payload.get("operator_capabilities") is None:
            payload["operator_capabilities"] = dict(operator_capabilities)
        return payload

    def _validate_required_local_law_supervision(self) -> None:
        if not bool(self.config.supervision.require_local_law_supervision):
            return
        if self._online_queue_enabled():
            queue_stats = self.online_node_oracle_queue.summary()
            if int(queue_stats.get("pending", 0)) > 0 or int(queue_stats.get("completed", 0)) > 0:
                return

        train_summary = self._tree_local_law_summary(self.train_trees)
        missing: List[str] = []
        if (
            float(self.config.objective.leaf_audit_weight) > 0.0
            and int(train_summary["labeled_leaves"]) <= 0
        ):
            missing.append("leaf oracle labels")
        if (
            float(self.config.objective.merge_audit_weight) > 0.0
            and int(train_summary["labeled_internal"]) <= 0
        ):
            missing.append("internal-node oracle labels")

        if not missing:
            return

        source_status = (
            "attached" if bool(train_summary["node_oracle_predictor_attached"]) else "missing"
        )
        raise ValueError(
            "Local-law supervision was required but inactive for the training split: "
            f"missing {', '.join(missing)}; "
            f"node_label_source={str(train_summary.get('node_label_source_kind', 'none'))}; "
            f"node_oracle_predictor={source_status}; "
            f"leaf_audit_weight={float(self.config.objective.leaf_audit_weight):.6g}; "
            f"merge_audit_weight={float(self.config.objective.merge_audit_weight):.6g}; "
            f"labeled_leaves={int(train_summary['labeled_leaves'])}; "
            f"labeled_internal={int(train_summary['labeled_internal'])}. "
            "Attach a task-provided local-law oracle, supply an explicit callback, explicitly opt into model-backed teacher labeling, "
            "or set the corresponding local-law weights to zero."
        )

    @staticmethod
    def _resolve_device(requested: Any) -> torch.device:
        raw = str(requested or "auto").strip().lower()
        if raw and raw != "auto":
            return torch.device(raw)
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device="cuda")
                return torch.device("cuda")
            except Exception:
                logger.warning("CUDA auto-detected but allocation failed; falling back to CPU.")
        return torch.device("cpu")

    def _packed_tree_for_nodes(
        self,
        nodes: Sequence[EmbeddingTreeNode],
    ) -> PackedEmbeddingTree:
        cache_key = int(id(nodes))
        cached = self._packed_tree_by_nodes_id.get(cache_key)
        if cached is not None:
            return cached
        packed = build_packed_embedding_tree(nodes)
        self._packed_tree_by_nodes_id[cache_key] = packed
        return packed

    def _packed_trees_for_batch_items(
        self,
        batch_trees: Sequence[Tuple[List[EmbeddingTreeNode], float, str]],
    ) -> List[PackedEmbeddingTree]:
        return [self._packed_tree_for_nodes(nodes) for nodes, _rile, _doc_id in batch_trees]

    @staticmethod
    def _clone_cached_packed_batch(
        batch: PackedTreeBatch,
        *,
        cache_hit: bool,
    ) -> PackedTreeBatch:
        return replace(
            batch,
            runtime_stats={
                **dict(batch.runtime_stats),
                "packed_batch_cache_hit": bool(cache_hit),
            },
        )

    def _packed_bucket_store_for_trees(
        self,
        packed_trees: Sequence[PackedEmbeddingTree],
    ) -> Optional[PackedTreeBucketStore]:
        if not packed_trees:
            return None
        first_store = self._packed_bucket_store_by_nodes_id.get(int(id(packed_trees[0].nodes)))
        if first_store is None:
            return None
        for tree in packed_trees[1:]:
            if self._packed_bucket_store_by_nodes_id.get(int(id(tree.nodes))) is not first_store:
                return None
        return first_store

    def _store_exact_packed_batch_cache(
        self,
        *,
        cache_key: Optional[Tuple[str, Tuple[int, ...]]],
        packed_batch: PackedTreeBatch,
    ) -> PackedTreeBatch:
        if cache_key is None:
            packed_batch.runtime_stats["packed_batch_cache_hit"] = False
            return packed_batch
        cached_copy = self._clone_cached_packed_batch(packed_batch, cache_hit=False)
        self._packed_batch_cache[cache_key] = cached_copy
        self._packed_batch_cache.move_to_end(cache_key)
        while len(self._packed_batch_cache) > int(self._packed_batch_cache_max_entries):
            self._packed_batch_cache.popitem(last=False)
        return self._clone_cached_packed_batch(cached_copy, cache_hit=False)

    def _packed_batch_cache_key(
        self,
        batch_trees: Sequence[Tuple[List[EmbeddingTreeNode], float, str]],
        packed_trees: Sequence[PackedEmbeddingTree],
    ) -> Optional[Tuple[str, Tuple[int, ...]]]:
        if not batch_trees:
            return None
        runtime_modes = {str(tree.runtime_data_mode) for tree in packed_trees}
        if self.device.type == "cuda" and runtime_modes != {"resident"}:
            return None
        return (
            str(self.device),
            tuple(int(id(nodes)) for nodes, _rile, _doc_id in batch_trees),
        )

    def _packed_batch_for_items(
        self,
        batch_trees: Sequence[Tuple[List[EmbeddingTreeNode], float, str]],
    ) -> PackedTreeBatch:
        packed_trees = self._packed_trees_for_batch_items(batch_trees)
        cache_key = self._packed_batch_cache_key(batch_trees, packed_trees)
        if cache_key is not None:
            cached_batch = self._packed_batch_cache.get(cache_key)
            if cached_batch is not None:
                self._packed_batch_cache.move_to_end(cache_key)
                return self._clone_cached_packed_batch(cached_batch, cache_hit=True)

        bucket_store = self._packed_bucket_store_for_trees(packed_trees)
        if bucket_store is not None:
            packed_batch = build_packed_tree_batch_from_bucket_store(
                bucket_store,
                packed_trees,
                device=self.device,
            )
            return self._store_exact_packed_batch_cache(
                cache_key=cache_key,
                packed_batch=packed_batch,
            )

        packed_batch = build_packed_tree_batch(
            packed_trees,
            device=self.device,
        )
        return self._store_exact_packed_batch_cache(
            cache_key=cache_key,
            packed_batch=packed_batch,
        )

    def _configure_split_runtime(
        self,
        trees: Sequence[Tuple[List[EmbeddingTreeNode], float, str]],
        *,
        split: str,
    ) -> None:
        packed_trees = self._packed_trees_for_batch_items(trees)
        self._packed_batch_cache.clear()
        runtime_metadata = dict(
            apply_runtime_mode_to_packed_trees(
                packed_trees,
                device=self.device,
            )
        )
        bucket_stores = build_packed_tree_bucket_stores(
            packed_trees,
            device=self.device,
        )
        self._packed_bucket_stores_by_split[str(split)] = bucket_stores
        self._packed_bucket_store_by_nodes_id = {}
        for stores in self._packed_bucket_stores_by_split.values():
            for store in stores:
                for tree in store.trees:
                    self._packed_bucket_store_by_nodes_id[int(id(tree.nodes))] = store
        runtime_metadata["fixed_shape_bucket_store_count"] = int(len(bucket_stores))
        runtime_metadata["fixed_shape_dense_bucket_store_count"] = int(
            sum(1 for store in bucket_stores if str(store.bucket_store_mode) == "dense_resident")
        )
        runtime_metadata["fixed_shape_dense_bucket_store_bytes"] = int(
            sum(int(store.resident_dense_bytes) for store in bucket_stores)
        )
        self._split_runtime_metadata[str(split)] = runtime_metadata

    def _forward_packed_batch(
        self,
        batch_trees: Sequence[Tuple[List[EmbeddingTreeNode], float, str]],
        *,
        materialize_nodes: bool = False,
    ) -> PackedForwardResult:
        packed_batch = self._packed_batch_for_items(batch_trees)
        return forward_packed_tree_batch(
            self.model,
            packed_batch,
            materialize_nodes=materialize_nodes,
        )

    def prepare_trees_from_samples(
        self,
        samples: List[Any],
        split: str = "train",
    ) -> int:
        """Build embedding trees for a set of ManifestoSample objects.

        Args:
            samples: List of ManifestoSample (must have .text and .rile).
            split: "train" or "val".

        Returns:
            Number of trees built.
        """
        if self.embedding_client is None:
            raise ValueError("embedding_client required for prepare_trees_from_samples")

        trees = []
        for sample in samples:
            try:
                nodes = build_tree_from_text(
                    text=sample.text,
                    embedding_client=self.embedding_client,
                    window_size=self.config.data.window_size,
                    window_overlap=self.config.data.window_overlap,
                    merge_drift_threshold=self.config.data.merge_drift_threshold,
                )
                if not self._online_queue_enabled():
                    self._label_tree_nodes_with_oracle_scores(
                        nodes,
                        doc_id=str(sample.manifesto_id),
                        split=split,
                    )
                trees.append((nodes, float(sample.rile), sample.manifesto_id))
                logger.info(
                    "Built tree for %s: %d nodes, %d leaves, RILE=%.1f",
                    sample.manifesto_id,
                    len(nodes),
                    sum(1 for n in nodes if n.is_leaf),
                    sample.rile,
                )
            except Exception as e:
                logger.error("Failed to build tree for %s: %s", getattr(sample, "manifesto_id", "?"), e)

        if split == "train":
            self.train_trees = trees
        else:
            self.val_trees = trees
        self._configure_split_runtime(trees, split=split)
        if self._online_queue_enabled() and split == "train":
            self._online_attach_completed_labels(split=split)
            self._online_enqueue_epoch_requests(split=split, epoch=0)

        return len(trees)

    def _record_offline_labeled_tree_observations(
        self,
        nodes: Sequence[EmbeddingTreeNode],
        *,
        doc_id: str,
        split: str,
        label_source: str,
        artifact_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        observations: List[LoggedLabelObservation[Any]] = []
        source = normalize_truth_label_source(label_source, default="oracle")
        metadata = dict(artifact_metadata or {})
        for idx, node in enumerate(nodes):
            if "rile" not in node.oracle_scores:
                continue
            unit_kind = ObservationUnitKind.LEAF if node.is_leaf else ObservationUnitKind.INTERNAL
            observations.append(
                shared_logged_substructure_observation(
                    document_id=str(doc_id),
                    unit_id=f"node_{idx}",
                    unit_kind=unit_kind,
                    label=float(node.oracle_scores["rile"]),
                    application_name="ctreepo_labeled_node_distillation",
                    supervision_signal_name="offline_labeled_node_score",
                    truth_label_source=source,
                    sampling=SamplingMetadata(
                        document_propensity=1.0,
                        unit_propensity=1.0,
                        label_propensity=1.0,
                        sampling_scheme="offline_labeled_tree_artifact",
                        policy_name="teacher_trace_dump",
                        unit_kind=unit_kind,
                        supports_ipw_estimation=False,
                    ),
                    context={
                        "char_start": int(node.char_start),
                        "char_end": int(node.char_end),
                        "text_span": str(node.text_span),
                        "artifact_version": metadata.get("artifact_version"),
                        "leaf_size_chars": metadata.get("leaf_size_chars"),
                        "window_overlap_chars": metadata.get("window_overlap_chars"),
                    },
                )
            )

        if split == "train":
            self._train_logged_node_observations = [
                obs for obs in self._train_logged_node_observations
                if obs.document_id != str(doc_id)
            ] + observations
        else:
            self._val_logged_node_observations = [
                obs for obs in self._val_logged_node_observations
                if obs.document_id != str(doc_id)
            ] + observations

    def prepare_trees_from_labeled_trees(
        self,
        labeled_trees: Sequence[Any],
        split: str = "train",
    ) -> int:
        """Build embedding trees and attach offline labeled-node scores.

        This is the Stage-1 distillation path: no teacher calls happen here.
        Runtime nodes replay the artifact's exact leaf spans and child links
        before attaching ``node.oracle_scores["rile"]``.  This keeps teacher
        and student training aligned even if current CLI chunking flags differ
        from the Stage-0 artifact topology.
        """
        if self.embedding_client is None:
            raise ValueError("embedding_client required for prepare_trees_from_labeled_trees")

        from treepo._research.ctreepo.distillation import build_embedding_tree_from_labeled_tree

        trees: List[Tuple[List[EmbeddingTreeNode], float, str]] = []
        attach_totals = {"attached": 0, "missing": 0, "leaf_attached": 0, "internal_attached": 0}
        for labeled_tree in labeled_trees:
            doc_id = str(getattr(labeled_tree, "doc_id", ""))
            try:
                nodes, stats = build_embedding_tree_from_labeled_tree(
                    labeled_tree,
                    embedding_client=self.embedding_client,
                    head="rile",
                )
                for key in attach_totals:
                    attach_totals[key] += int(stats.get(key, 0))
                label_source = str(
                    getattr(labeled_tree, "label_source", "")
                    or (getattr(labeled_tree, "metadata", {}) or {}).get("label_source", "")
                    or "oracle"
                )
                self._record_offline_labeled_tree_observations(
                    nodes,
                    doc_id=doc_id,
                    split=split,
                    label_source=label_source,
                    artifact_metadata=dict(getattr(labeled_tree, "metadata", {}) or {}),
                )
                trees.append((nodes, float(labeled_tree.document_score), doc_id))
                logger.info(
                    "Replayed labeled tree for %s: %d nodes, attached=%d leaf=%d internal=%d",
                    doc_id,
                    len(nodes),
                    int(stats.get("attached", 0)),
                    int(stats.get("leaf_attached", 0)),
                    int(stats.get("internal_attached", 0)),
                )
            except Exception as e:
                logger.error("Failed to build labeled tree for %s: %s", doc_id or "?", e)

        if split == "train":
            self.train_trees = trees
        else:
            self.val_trees = trees
        self._configure_split_runtime(trees, split=split)
        logger.info(
            "Prepared %d %s labeled trees: attached=%d missing=%d leaves=%d internal=%d",
            len(trees),
            split,
            attach_totals["attached"],
            attach_totals["missing"],
            attach_totals["leaf_attached"],
            attach_totals["internal_attached"],
        )
        return len(trees)

    def prepare_trees_from_precomputed(
        self,
        embeddings_by_doc: Dict[str, Tuple[List[List[float]], List[Tuple[int, int]], str, float]],
        split: str = "train",
    ) -> int:
        """Build trees from pre-computed embeddings (no embedding client needed).

        Args:
            embeddings_by_doc: {doc_id: (embeddings, windows, text, rile)}
            split: "train" or "val"
        """
        from treepo._research.tree.embedding_tree import build_embedding_tree

        trees = []
        for doc_id, (embs, windows, text, rile) in embeddings_by_doc.items():
            nodes = build_embedding_tree(text, embs, windows)
            self._label_tree_nodes_with_oracle_scores(nodes, doc_id=str(doc_id), split=split)
            trees.append((nodes, rile, doc_id))

        if split == "train":
            self.train_trees = trees
        else:
            self.val_trees = trees
        self._configure_split_runtime(trees, split=split)
        return len(trees)

    def train_step(
        self,
        batch_trees: List[Tuple[List[EmbeddingTreeNode], float, str]],
        optimizer: optim.Optimizer,
    ) -> float:
        """One training step over a batch of document trees.

        Uses batched forward passes across all trees and batched loss
        computation for root, leaf (C1), and merge (C3) supervision.
        """
        self.model.train()
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device)
        n_terms = 0
        has_any_leaf_labels, has_any_merge_labels = self._tree_batch_has_oracle_labels(
            batch_trees
        )

        prepared = self._shared_trainer.prepare_batch(batch_trees)
        shared_loss, shared_terms, shared_stats = self._shared_trainer.compute_supervision_loss(
            prepared
        )
        total_loss = total_loss + shared_loss
        n_terms += int(shared_terms)
        self._last_shared_supervision_stats = dict(shared_stats)
        packed_forward = (
            prepared.forward_result
            if isinstance(prepared.forward_result, PackedForwardResult)
            else None
        )

        self._maybe_warn_missing_local_law_labels(
            has_any_leaf_labels=has_any_leaf_labels,
            has_any_merge_labels=has_any_merge_labels,
        )

        if packed_forward is not None:
            root_sketches, root_targets = self._collect_root_batch_from_forward(
                batch_trees,
                packed_forward,
            )
        else:
            root_sketches, root_targets = self._collect_root_batch(batch_trees)
        wrapper_loss, wrapper_terms, wrapper_stats = self._wrapper_regularization_loss(
            batch_trees,
            root_sketches,
            root_targets,
            forward_result=packed_forward,
        )
        total_loss = total_loss + wrapper_loss
        n_terms += int(wrapper_terms)
        self._last_wrapper_regularization_stats = dict(wrapper_stats)
        self._last_train_step_stats = {
            "tree_model_version": str(getattr(self.model, "tree_model_version", "legacy")),
            "shared": dict(shared_stats),
            "wrapper": dict(wrapper_stats),
            "shared_loss_term_count": int(shared_terms),
            "wrapper_loss_term_count": int(wrapper_terms),
            "total_loss_term_count": int(n_terms),
            "shared_loss": float(shared_loss.detach().cpu().item()),
            "wrapper_loss": float(wrapper_loss.detach().cpu().item()),
            "runtime": dict(packed_forward.runtime_stats) if packed_forward is not None else {},
        }

        if n_terms > 0:
            loss_val = total_loss / n_terms
            loss_val.backward()
            grad_clip = float(getattr(self.config, "grad_clip_norm", 0.0) or 0.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
            optimizer.step()
            self._last_train_step_stats["loss_value"] = float(loss_val.detach().cpu().item())
            return loss_val.item()
        self._last_train_step_stats["loss_value"] = 0.0
        return 0.0

    @torch.no_grad()
    def evaluate(
        self,
        trees: Optional[List[Tuple[List[EmbeddingTreeNode], float, str]]] = None,
        epoch: int = 0,
    ) -> CTreePOEvalMetrics:
        """Evaluate model on a set of document trees."""
        self.model.eval()
        if trees is None:
            trees = self.val_trees if self.val_trees else self.train_trees

        errors: List[float] = []
        sq_errors: List[float] = []
        norm_errors: List[float] = []
        covered_95: List[float] = []
        width_95: List[float] = []
        confidence_errors: List[float] = []
        all_node_oracle_errors: List[float] = []
        leaf_oracle_errors: List[float] = []
        merge_oracle_errors: List[float] = []
        per_doc: List[Dict[str, Any]] = []
        total_nodes = 0
        total_labeled_nodes = 0
        violation_threshold = float(self.config.objective.local_law_violation_threshold)

        forward_result = self._forward_packed_batch(trees, materialize_nodes=False)
        self._last_eval_runtime_stats = dict(forward_result.runtime_stats)

        root_states = forward_result.state_batch.index_select(0, forward_result.root_indices)
        pred_batch = self.model.predict_batch(root_states, "rile").reshape(-1)
        pred_norm_batch = self.model.predict_normalized_batch(root_states, "rile").reshape(-1)
        confidence_batch = self.model.predict_confidence_batch(root_states, "rile").reshape(-1)
        pred_norm_clamped = torch.clamp(
            pred_norm_batch,
            min=1e-6,
            max=1.0 - 1e-6,
        )
        span = float(self.config.model.target_max - self.config.model.target_min)
        std95_batch = torch.clamp(
            torch.sqrt(pred_norm_clamped * (1.0 - pred_norm_clamped)) * span,
            min=float(self.config.evaluation.min_interval_std),
        )
        lower95_batch = torch.clamp(
            pred_batch - float(self.config.evaluation.uncertainty_z_score) * std95_batch,
            min=self.config.model.target_min,
            max=self.config.model.target_max,
        )
        upper95_batch = torch.clamp(
            pred_batch + float(self.config.evaluation.uncertainty_z_score) * std95_batch,
            min=self.config.model.target_min,
            max=self.config.model.target_max,
        )

        pred_values = pred_batch.detach().cpu().tolist()
        pred_norm_values = pred_norm_batch.detach().cpu().tolist()
        confidence_values = confidence_batch.detach().cpu().tolist()
        lower_values = lower95_batch.detach().cpu().tolist()
        upper_values = upper95_batch.detach().cpu().tolist()
        std_values = std95_batch.detach().cpu().tolist()

        doc_leaf_oracle_counts = [0 for _ in trees]
        doc_merge_oracle_counts = [0 for _ in trees]
        packed_trees = list(forward_result.packed_batch.trees)
        total_nodes = int(sum(tree.node_count for tree in packed_trees))

        labeled_global_indices: List[int] = []
        labeled_targets: List[float] = []
        labeled_leaf_flags: List[bool] = []
        labeled_doc_indices: List[int] = []
        for doc_index, tree in enumerate(packed_trees):
            labeled_local_indices = torch.nonzero(
                tree.oracle_score_mask_cpu,
                as_tuple=False,
            ).reshape(-1)
            for local_idx in labeled_local_indices.tolist():
                labeled_global_indices.append(int(forward_result.global_index(doc_index, local_idx)))
                labeled_targets.append(float(tree.oracle_score_values_cpu[int(local_idx)].item()))
                labeled_leaf_flags.append(bool(tree.leaf_mask_cpu[int(local_idx)].item()))
                labeled_doc_indices.append(int(doc_index))

        total_labeled_nodes = int(len(labeled_global_indices))
        if labeled_global_indices:
            labeled_index_tensor = torch.tensor(
                labeled_global_indices,
                dtype=torch.long,
                device=self.device,
            )
            labeled_states = forward_result.state_batch.index_select(0, labeled_index_tensor)
            labeled_preds = self.model.predict_batch(labeled_states, "rile").reshape(-1).detach().cpu().tolist()
            for pred_val, target_val, is_leaf, doc_index in zip(
                labeled_preds,
                labeled_targets,
                labeled_leaf_flags,
                labeled_doc_indices,
            ):
                node_error = abs(float(pred_val) - float(target_val))
                all_node_oracle_errors.append(node_error)
                if bool(is_leaf):
                    doc_leaf_oracle_counts[int(doc_index)] += 1
                    leaf_oracle_errors.append(node_error)
                else:
                    doc_merge_oracle_counts[int(doc_index)] += 1
                    merge_oracle_errors.append(node_error)

        for doc_index, (tree_tuple, pred_val, pred_norm_val, confidence_val, lower_val, upper_val, std_val) in enumerate(
            zip(
                trees,
                pred_values,
                pred_norm_values,
                confidence_values,
                lower_values,
                upper_values,
                std_values,
            )
        ):
            _nodes, rile, doc_id = tree_tuple
            error = abs(float(pred_val) - float(rile))
            errors.append(error)
            sq_errors.append(error ** 2)
            target_norm = normalize_target(
                rile,
                self.config.model.target_min,
                self.config.model.target_max,
            )
            norm_error = abs(float(pred_norm_val) - float(target_norm))
            norm_errors.append(norm_error)
            width_val = max(0.0, float(upper_val - lower_val))
            width_95.append(width_val)
            covered_95.append(1.0 if (float(lower_val) <= float(rile) <= float(upper_val)) else 0.0)
            proxy_accuracy = max(0.0, min(1.0, 1.0 - norm_error))
            confidence_errors.append(abs(float(confidence_val) - proxy_accuracy))
            per_doc.append({
                "doc_id": doc_id,
                "rile_true": rile,
                "rile_pred": round(float(pred_val), 2),
                "abs_error": round(error, 2),
                "pred_norm": round(float(pred_norm_val), 4),
                "confidence": round(float(confidence_val), 4),
                "pred_interval_95": [round(float(lower_val), 2), round(float(upper_val), 2)],
                "pred_std_proxy": round(float(std_val), 4),
                "in_interval_95": bool(float(lower_val) <= float(rile) <= float(upper_val)),
                "oracle_labeled_leaves": int(doc_leaf_oracle_counts[doc_index]),
                "oracle_labeled_internal": int(doc_merge_oracle_counts[doc_index]),
            })

        mae = float(np.mean(errors)) if errors else 0.0
        mse = float(np.mean(sq_errors)) if sq_errors else 0.0
        mae_norm = float(np.mean(norm_errors)) if norm_errors else 0.0
        coverage_95 = float(np.mean(covered_95)) if covered_95 else 0.0
        mean_width_95 = float(np.mean(width_95)) if width_95 else 0.0
        conf_cal_err = float(np.mean(confidence_errors)) if confidence_errors else 0.0
        node_oracle_mae = float(np.mean(all_node_oracle_errors)) if all_node_oracle_errors else float("nan")
        leaf_oracle_mae = float(np.mean(leaf_oracle_errors)) if leaf_oracle_errors else float("nan")
        merge_oracle_mae = float(np.mean(merge_oracle_errors)) if merge_oracle_errors else float("nan")
        leaf_violation_rate = (
            float(np.mean(np.asarray(leaf_oracle_errors, dtype=np.float64) > violation_threshold))
            if leaf_oracle_errors
            else float("nan")
        )
        merge_violation_rate = (
            float(np.mean(np.asarray(merge_oracle_errors, dtype=np.float64) > violation_threshold))
            if merge_oracle_errors
            else float("nan")
        )
        node_oracle_label_rate = (
            float(total_labeled_nodes) / float(total_nodes) if total_nodes > 0 else 0.0
        )

        return CTreePOEvalMetrics(
            epoch=epoch,
            root_mae=mae,
            root_mse=mse,
            root_mae_normalized=mae_norm,
            interval_coverage_95=coverage_95,
            interval_mean_width_95=mean_width_95,
            confidence_calibration_error=conf_cal_err,
            node_oracle_label_rate=node_oracle_label_rate,
            node_oracle_mae=node_oracle_mae,
            leaf_oracle_mae=leaf_oracle_mae,
            merge_oracle_mae=merge_oracle_mae,
            leaf_violation_rate=leaf_violation_rate,
            merge_violation_rate=merge_violation_rate,
            leaf_oracle_count=len(leaf_oracle_errors),
            merge_oracle_count=len(merge_oracle_errors),
            n_docs=len(trees),
            per_doc=per_doc,
        )

    def _make_optimizer(self) -> optim.Optimizer:
        cfg = self.config.optimizer
        name = str(cfg.optimizer or "adamw").strip().lower()
        if name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=float(cfg.learning_rate),
                weight_decay=float(cfg.weight_decay),
            )
        return optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )

    def _make_scheduler(self, optimizer: optim.Optimizer):
        cfg = self.config.optimizer
        train_cfg = self.config.train
        mode = str(getattr(cfg, "scheduler", "none") or "none").strip().lower()
        if mode in {"", "none", "off"}:
            return None

        total_epochs = max(1, int(train_cfg.epochs))
        warmup_epochs = max(0, min(total_epochs - 1, int(cfg.warmup_epochs)))
        base_lr = max(1e-12, float(cfg.learning_rate))
        min_lr = max(0.0, min(float(cfg.min_learning_rate), base_lr))
        min_ratio = min_lr / base_lr

        def _lr_lambda(epoch_idx: int) -> float:
            epoch_idx = int(max(0, epoch_idx))
            if warmup_epochs > 0 and epoch_idx < warmup_epochs:
                return float(epoch_idx + 1) / float(warmup_epochs)

            denom = max(1, total_epochs - warmup_epochs - 1)
            progress = min(1.0, max(0.0, float(epoch_idx - warmup_epochs) / float(denom)))
            if mode == "linear":
                return float(min_ratio + (1.0 - min_ratio) * (1.0 - progress))
            if mode == "cosine":
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return float(min_ratio + (1.0 - min_ratio) * cosine)
            return 1.0

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # -----------------------------------------------------------------
    # Uncertainty-guided audit sampling (Phase 3)
    # -----------------------------------------------------------------

    @torch.no_grad()
    def select_audit_nodes(
        self,
        nodes: List[EmbeddingTreeNode],
        n_audit: int = 5,
        exploration_fraction: float = 0.3,
    ) -> List[int]:
        """Select internal nodes for oracle audit, preferring uncertain sketches.

        Instead of random sampling, this uses sketch readout confidence to
        prioritise nodes where the model is least sure.  A fraction of the
        budget is reserved for random exploration to avoid getting stuck.

        Args:
            nodes: Unified tree nodes (must have sketch set via forward_ctreepo).
            n_audit: Total number of nodes to select.
            exploration_fraction: Fraction of budget for random sampling.

        Returns:
            List of node indices to audit.
        """
        self.model.eval()
        internal_indices = [
            i for i, node in enumerate(nodes) if not node.is_leaf
        ]
        if not internal_indices or n_audit <= 0:
            return []

        n_to_select = min(n_audit, len(internal_indices))
        n_explore = max(1, int(n_to_select * exploration_fraction))
        n_uncertain = n_to_select - n_explore

        valid_indices: List[int] = []
        valid_sketches: List[torch.Tensor] = []
        confidences: List[Tuple[int, float]] = []
        for idx in internal_indices:
            node = nodes[idx]
            if node.sketch is None:
                confidences.append((idx, 0.0))  # no sketch → definitely audit
                continue
            valid_indices.append(idx)
            valid_sketches.append(node.sketch)
        if valid_sketches:
            sketch_batch = torch.stack(valid_sketches, dim=0)
            conf_values = (
                self.model.predict_confidence_batch(sketch_batch, "rile")
                .reshape(-1)
                .detach()
                .cpu()
                .tolist()
            )
            confidences.extend((idx, float(conf)) for idx, conf in zip(valid_indices, conf_values))

        # Sort by ascending confidence (lowest first = most uncertain)
        confidences.sort(key=lambda x: x[1])

        # Take top-n_uncertain most uncertain
        selected = set()
        for idx, _ in confidences[:n_uncertain]:
            selected.add(idx)

        # Random exploration from remaining
        remaining = [idx for idx, _ in confidences if idx not in selected]
        if remaining and n_explore > 0:
            explore = self.rng.sample(remaining, k=min(n_explore, len(remaining)))
            selected.update(explore)

        return list(selected)

    @torch.no_grad()
    def populate_sketch_scores(
        self,
        nodes: List[EmbeddingTreeNode],
        head: str = "rile",
    ) -> None:
        """Fill sketch_scores and sketch_confidence on all nodes in-place.

        After ``forward_ctreepo(model, nodes)`` has set sketches, this
        computes readout predictions and confidence for every node, writing
        them into the unified node fields.  Uses batched inference for
        efficiency.

        Args:
            nodes: Tree nodes with sketches already set.
            head: Which readout head to use.
        """
        self.model.eval()

        # Collect all nodes with sketches for batched prediction.
        valid_indices: List[int] = []
        sketches: List[torch.Tensor] = []
        for i, node in enumerate(nodes):
            if node.sketch is not None:
                valid_indices.append(i)
                sketches.append(node.sketch)

        if not sketches:
            return

        sketch_batch = torch.stack(sketches, dim=0)  # (N, sketch_dim)
        with torch.no_grad():
            preds = self.model.predict_batch(sketch_batch, head).reshape(-1)  # (N,)
            confs = self.model.predict_confidence_batch(sketch_batch, head).reshape(-1)  # (N,)
        pred_values = preds.detach().cpu().tolist()
        conf_values = confs.detach().cpu().tolist()

        for idx, pred_val, conf_val in zip(valid_indices, pred_values, conf_values):
            nodes[idx].sketch_scores[head] = round(float(pred_val), 2)
            nodes[idx].sketch_confidence = float(conf_val)

    def train(self, output_dir: Optional[Path] = None) -> TrainingResult:
        """Run the full training loop.

        Args:
            output_dir: If set, save checkpoints and metrics here.

        Returns:
            TrainingResult with training history.
        """
        if not self.train_trees:
            raise ValueError("No training trees. Call prepare_trees_from_samples() first.")

        cfg = self.config
        self._validate_required_local_law_supervision()
        optimizer = self._make_optimizer()
        scheduler = self._make_scheduler(optimizer)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        result = TrainingResult(config=config_to_dict(cfg))
        result.reproducibility = dict(self.reproducibility)
        result.local_law_summary = self._tree_local_law_summary(self.train_trees + self.val_trees)
        result.compositional_learning_problem = dict(
            result.local_law_summary.get("compositional_learning_problem", {}) or {}
        )
        if output_dir:
            logged_observations = (
                list(self._train_logged_node_observations)
                + list(self._val_logged_node_observations)
            )
            if logged_observations:
                artifact = write_logged_observations_jsonl(
                    output_dir / "node_oracle_logged_observations.jsonl",
                    logged_observations,
                    channel_name="sampled_substructure_supervision",
                )
                result.logged_observation_artifacts = {
                    artifact.channel_name: artifact.to_dict()
                }
                result.local_law_summary["logged_observation_artifacts"] = dict(
                    result.logged_observation_artifacts
                )
        best_mae = float("inf")
        best_epoch = 0
        best_epoch_metrics: Optional[Dict[str, Any]] = None
        epochs_since_improve = 0
        start_time = time.time()

        logger.info(
            "Starting CTreePO training: %d train docs, %d val docs, %d epochs (device=%s)",
            len(self.train_trees), len(self.val_trees), cfg.train.epochs,
            self.device,
        )

        for epoch in range(cfg.train.epochs):
            if self._online_queue_enabled():
                worker_stats = self._online_collect_worker_result(timeout_seconds=0.05)
                attach_stats = self._online_attach_completed_labels(split="train")
                if attach_stats.get("attached", 0) or worker_stats:
                    logger.info(
                        "Online node oracle epoch-boundary update: attached=%d worker=%s",
                        int(attach_stats.get("attached", 0) or 0),
                        worker_stats,
                    )

            # Shuffle training data
            indices = list(range(len(self.train_trees)))
            self.rng.shuffle(indices)

            epoch_losses: List[float] = []
            for batch_start in range(0, len(indices), cfg.train.batch_size):
                batch_idx = indices[batch_start:batch_start + cfg.train.batch_size]
                batch = [self.train_trees[i] for i in batch_idx]
                loss = self.train_step(batch, optimizer)
                epoch_losses.append(loss)

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            result.train_losses.append(avg_loss)
            result.epochs_completed = int(epoch + 1)
            current_lr = float(optimizer.param_groups[0]["lr"])

            # Evaluation
            if (
                (epoch + 1) % cfg.validation.eval_every == 0
                or epoch == cfg.train.epochs - 1
            ):
                eval_trees = self.val_trees if self.val_trees else self.train_trees
                metrics = self.evaluate(eval_trees, epoch=epoch)

                metrics_dict = {
                    "epoch": epoch,
                    "train_loss": round(avg_loss, 6),
                    "learning_rate": round(current_lr, 10),
                    "root_mae": round(metrics.root_mae, 2),
                    "root_mse": round(metrics.root_mse, 2),
                    "root_mae_norm": round(metrics.root_mae_normalized, 6),
                    "interval_coverage_95": round(metrics.interval_coverage_95, 4),
                    "interval_mean_width_95": round(metrics.interval_mean_width_95, 4),
                    "confidence_calibration_error": round(metrics.confidence_calibration_error, 6),
                    "local_law": {
                        "node_oracle_label_rate": round(metrics.node_oracle_label_rate, 6),
                        "node_oracle_mae": round(metrics.node_oracle_mae, 4)
                        if np.isfinite(metrics.node_oracle_mae)
                        else None,
                        "leaf_oracle_mae": round(metrics.leaf_oracle_mae, 4)
                        if np.isfinite(metrics.leaf_oracle_mae)
                        else None,
                        "merge_oracle_mae": round(metrics.merge_oracle_mae, 4)
                        if np.isfinite(metrics.merge_oracle_mae)
                        else None,
                        "leaf_violation_rate": round(metrics.leaf_violation_rate, 6)
                        if np.isfinite(metrics.leaf_violation_rate)
                        else None,
                        "merge_violation_rate": round(metrics.merge_violation_rate, 6)
                        if np.isfinite(metrics.merge_violation_rate)
                        else None,
                        "leaf_oracle_count": int(metrics.leaf_oracle_count),
                        "merge_oracle_count": int(metrics.merge_oracle_count),
                        "violation_threshold_raw": float(
                            cfg.objective.local_law_violation_threshold
                        ),
                    },
                    "per_doc": metrics.per_doc,
                    "runtime": dict(self._last_eval_runtime_stats),
                }
                result.eval_metrics.append(metrics_dict)

                logger.info(
                    "Epoch %d/%d: loss=%.4f lr=%.3e MAE=%.2f MSE=%.2f cov95=%.3f width95=%.2f cce=%.3f node_label_rate=%.3f leaf_viol=%s merge_viol=%s",
                    epoch + 1, cfg.train.epochs, avg_loss,
                    current_lr,
                    metrics.root_mae, metrics.root_mse,
                    metrics.interval_coverage_95,
                    metrics.interval_mean_width_95,
                    metrics.confidence_calibration_error,
                    metrics.node_oracle_label_rate,
                    (
                        f"{metrics.leaf_violation_rate:.3f}"
                        if np.isfinite(metrics.leaf_violation_rate)
                        else "na"
                    ),
                    (
                        f"{metrics.merge_violation_rate:.3f}"
                        if np.isfinite(metrics.merge_violation_rate)
                        else "na"
                    ),
                )

                if metrics.root_mae < (
                    best_mae - float(cfg.validation.early_stopping_min_delta)
                ):
                    best_mae = metrics.root_mae
                    best_epoch = epoch
                    best_epoch_metrics = dict(metrics_dict)
                    epochs_since_improve = 0
                    if output_dir:
                        torch.save(self.model.state_dict(), output_dir / "best.pt")
                        logger.info("  -> New best (MAE=%.2f), saved checkpoint", best_mae)
                else:
                    epochs_since_improve += 1
                    patience = int(max(0, cfg.validation.early_stopping_patience))
                    if patience > 0 and epochs_since_improve >= patience:
                        result.stopped_early = True
                        logger.info(
                            "Early stopping at epoch %d (no MAE improvement for %d evals; best epoch=%d MAE=%.2f)",
                            epoch + 1,
                            epochs_since_improve,
                            best_epoch + 1,
                            best_mae,
                        )
                        if scheduler is not None:
                            scheduler.step()
                        break

            if self._online_queue_enabled():
                enqueue_stats = self._online_enqueue_epoch_requests(
                    split="train",
                    epoch=epoch + 1,
                )
                worker_start = self._online_start_teacher_worker()
                if enqueue_stats.get("enqueued", 0) or worker_start:
                    logger.info(
                        "Online node oracle queued epoch=%d enqueued=%d worker=%s",
                        epoch + 1,
                        int(enqueue_stats.get("enqueued", 0) or 0),
                        worker_start,
                    )

            if scheduler is not None:
                scheduler.step()

        result.best_epoch = best_epoch
        result.best_root_mae = best_mae
        result.training_time_seconds = time.time() - start_time
        if self._online_worker_executor is not None:
            self._online_worker_executor.shutdown(wait=False)
        result.local_law_summary = self._tree_local_law_summary(self.train_trees + self.val_trees)
        result.compositional_learning_problem = dict(
            result.local_law_summary.get("compositional_learning_problem", {}) or {}
        )

        logger.info(
            "Training complete: best MAE=%.2f at epoch %d (%.1fs)",
            best_mae, best_epoch + 1, result.training_time_seconds,
        )

        if output_dir:
            logged_observations = (
                list(self._train_logged_node_observations)
                + list(self._val_logged_node_observations)
            )
            if logged_observations:
                artifact = write_logged_observations_jsonl(
                    output_dir / "node_oracle_logged_observations.jsonl",
                    logged_observations,
                    channel_name="sampled_substructure_supervision",
                )
                result.logged_observation_artifacts = {
                    artifact.channel_name: artifact.to_dict()
                }
                result.local_law_summary["logged_observation_artifacts"] = dict(
                    result.logged_observation_artifacts
                )
            # Save final model and training results
            torch.save(self.model.state_dict(), output_dir / "final.pt")
            if best_epoch_metrics is not None:
                (output_dir / "best_metrics.json").write_text(
                    json.dumps(best_epoch_metrics, indent=2, default=str),
                    encoding="utf-8",
                )
            (output_dir / "training_result.json").write_text(
                json.dumps(asdict(result), indent=2, default=str),
                encoding="utf-8",
            )

        return result


# ---------------------------------------------------------------------------
# Sketch extraction (inference mode)
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_root_sketch(
    model: CTreePOModel,
    text: str,
    embedding_client: Any,
    window_size: int = 1200,
    window_overlap: int = 150,
) -> Tuple[torch.Tensor, float]:
    """Extract root sketch and RILE prediction for a document.

    Returns:
        (root_sketch, rile_prediction)
    """
    model.eval()
    nodes = build_tree_from_text(
        text, embedding_client, window_size, window_overlap
    )
    forward_ctreepo(model, nodes)
    root = get_root_sketch(nodes)
    rile = model.predict(root, "rile").item()
    return root, rile
