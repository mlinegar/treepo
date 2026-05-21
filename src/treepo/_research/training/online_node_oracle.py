"""Online node-level oracle queue for C-TreePO local-law supervision."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import logging
import random
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from treepo._research.core.logged_supervision import (
    LoggedLabelObservation,
    ObservationUnitKind,
    SamplingMetadata,
)
from treepo._research.feedback.collectors.oracle import OracleCollector
from treepo._research.feedback.store import FeedbackStore
from treepo._research.feedback.types import FeedbackDimension, FeedbackRequest, FeedbackResponse
from treepo._research.training.supervision.timing import (
    ACQUISITION_ASYNC_FEEDBACK_QUEUE,
    ACTIVATION_EPOCH_BOUNDARY,
    CONSUMER_CTREEPO_GRADIENT,
    supervision_timing_contract,
)
from treepo._research.tree.compositional_learning import shared_logged_substructure_observation
from treepo._research.tree.embedding_tree import EmbeddingTreeNode

logger = logging.getLogger(__name__)

TreeItem = Tuple[List[EmbeddingTreeNode], float, str]


@dataclass
class OnlineNodeOracleQueueConfig:
    leaf_budget_per_epoch: int = 16
    merge_budget_per_epoch: int = 16
    target_name: str = "rile"
    request_prefix: str = "ctreepo_node_oracle"
    sampling_policy_name: str = "budgeted_random_node_feedback"
    source_kind: str = "oracle"
    source_spec: Optional[str] = None
    target_min: float = -100.0
    target_max: float = 100.0
    activation_barrier: str = ACTIVATION_EPOCH_BOUNDARY


@dataclass
class OnlineNodeOracleAttachResult:
    attached: int = 0
    leaf_attached: int = 0
    merge_attached: int = 0
    observations: List[LoggedLabelObservation[Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attached": int(self.attached),
            "leaf_attached": int(self.leaf_attached),
            "merge_attached": int(self.merge_attached),
        }


class OnlineNodeOracleQueue:
    """Bridge C-TreePO runtime nodes to the shared FeedbackStore queue."""

    def __init__(
        self,
        *,
        store: FeedbackStore,
        config: Optional[OnlineNodeOracleQueueConfig] = None,
        rng: Optional[random.Random] = None,
    ):
        self.store = store
        self.config = config or OnlineNodeOracleQueueConfig()
        self.rng = rng or random.Random()
        self.total_enqueued = 0
        self.total_attached = 0
        self.last_enqueue_stats: Dict[str, Any] = {}
        self.last_attach_stats: Dict[str, Any] = {}
        self.last_worker_stats: Dict[str, Any] = {}

    def request_id(
        self,
        *,
        split: str,
        doc_id: str,
        node_index: int,
        node: EmbeddingTreeNode,
    ) -> str:
        return (
            f"{self.config.request_prefix}:"
            f"{split}:{doc_id}:node_{int(node_index)}:"
            f"{int(node.char_start)}-{int(node.char_end)}"
        )

    def enqueue_epoch_requests(
        self,
        trees: Sequence[TreeItem],
        *,
        split: str,
        epoch: int,
    ) -> Dict[str, Any]:
        """Sample unlabeled nodes and enqueue non-blocking feedback requests."""
        self._reload()
        known_ids = self._known_request_ids()
        leaf_candidates: List[Tuple[TreeItem, int, EmbeddingTreeNode, str]] = []
        merge_candidates: List[Tuple[TreeItem, int, EmbeddingTreeNode, str]] = []
        for item in trees:
            nodes, _target, doc_id = item
            for idx, node in enumerate(nodes):
                if self.config.target_name in node.oracle_scores:
                    continue
                req_id = self.request_id(
                    split=split,
                    doc_id=str(doc_id),
                    node_index=idx,
                    node=node,
                )
                if req_id in known_ids:
                    continue
                target = leaf_candidates if node.is_leaf else merge_candidates
                target.append((item, idx, node, req_id))

        selected_leaf, leaf_propensity = self._sample_candidates(
            leaf_candidates,
            budget=int(self.config.leaf_budget_per_epoch),
        )
        selected_merge, merge_propensity = self._sample_candidates(
            merge_candidates,
            budget=int(self.config.merge_budget_per_epoch),
        )
        enqueued = 0
        for item, idx, node, req_id in selected_leaf:
            self.store.enqueue(
                self._build_request(
                    item=item,
                    node_index=idx,
                    node=node,
                    request_id=req_id,
                    split=split,
                    epoch=epoch,
                    unit_kind=ObservationUnitKind.LEAF,
                    unit_propensity=leaf_propensity,
                )
            )
            enqueued += 1
        for item, idx, node, req_id in selected_merge:
            self.store.enqueue(
                self._build_request(
                    item=item,
                    node_index=idx,
                    node=node,
                    request_id=req_id,
                    split=split,
                    epoch=epoch,
                    unit_kind=ObservationUnitKind.INTERNAL,
                    unit_propensity=merge_propensity,
                )
            )
            enqueued += 1
        self.total_enqueued += int(enqueued)
        self.last_enqueue_stats = {
            "epoch": int(epoch),
            "split": str(split),
            "leaf_candidates": int(len(leaf_candidates)),
            "merge_candidates": int(len(merge_candidates)),
            "leaf_enqueued": int(len(selected_leaf)),
            "merge_enqueued": int(len(selected_merge)),
            "enqueued": int(enqueued),
            "leaf_unit_propensity": float(leaf_propensity),
            "merge_unit_propensity": float(merge_propensity),
        }
        return dict(self.last_enqueue_stats)

    def attach_completed(
        self,
        trees: Sequence[TreeItem],
        *,
        split: str,
        truth_label_source: str = "oracle",
    ) -> OnlineNodeOracleAttachResult:
        """Attach completed feedback responses to matching runtime nodes."""
        self._reload()
        completed = {
            request.request_id: (request, response)
            for request, response in self.store.get_completed(limit=10_000_000)
        }
        result = OnlineNodeOracleAttachResult()
        for nodes, _target, doc_id in trees:
            for idx, node in enumerate(nodes):
                if self.config.target_name in node.oracle_scores:
                    continue
                req_id = self.request_id(
                    split=split,
                    doc_id=str(doc_id),
                    node_index=idx,
                    node=node,
                )
                pair = completed.get(req_id)
                if pair is None:
                    continue
                request, response = pair
                score = self._score_from_response(response)
                if score is None:
                    continue
                node.oracle_scores[self.config.target_name] = float(score)
                unit_kind = ObservationUnitKind.LEAF if node.is_leaf else ObservationUnitKind.INTERNAL
                sampling = self._sampling_for_response(request.sampling, response)
                result.observations.append(
                    shared_logged_substructure_observation(
                        document_id=str(doc_id),
                        unit_id=f"node_{idx}",
                        unit_kind=unit_kind,
                        label=float(score),
                        application_name="ctreepo_online_local_law_training",
                        supervision_signal_name="online_node_oracle_score",
                        truth_label_source=str(response.source or truth_label_source),
                        sampling=sampling,
                        context={
                            "request_id": req_id,
                            "char_start": int(node.char_start),
                            "char_end": int(node.char_end),
                            "text_span": str(node.text_span),
                            "response_source": str(response.source or ""),
                            "judge_model": str(response.judge_model or ""),
                        },
                    )
                )
                result.attached += 1
                if node.is_leaf:
                    result.leaf_attached += 1
                else:
                    result.merge_attached += 1

        self.total_attached += int(result.attached)
        self.last_attach_stats = result.to_dict()
        return result

    def run_teacher_worker(
        self,
        oracle_predictor: Callable[[str], float],
        *,
        limit: Optional[int] = None,
        concurrency: int = 4,
    ) -> Dict[str, Any]:
        """Answer pending scalar node requests with a model/task oracle."""
        if oracle_predictor is None:
            return {"processed": 0, "submitted": 0, "failed": 0}
        self._reload()
        pending = [
            request
            for request in self.store.get_pending(limit=int(limit or 10_000_000))
            if bool((request.context or {}).get("ctreepo_online_node_oracle"))
        ]
        if not pending:
            self.last_worker_stats = {"processed": 0, "submitted": 0, "failed": 0}
            return dict(self.last_worker_stats)

        collector = OracleCollector(
            oracle_predict=oracle_predictor,
            prefer_lower=False,
            scale_range=float(self.config.target_max) - float(self.config.target_min),
        )
        submitted = 0
        failed = 0
        max_workers = max(1, int(concurrency))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_request = {
                pool.submit(collector.collect, request): request
                for request in pending
            }
            for future in as_completed(future_to_request):
                request = future_to_request[future]
                try:
                    response = future.result()
                except Exception as exc:
                    logger.warning("Online node oracle worker failed for %s: %s", request.request_id, exc)
                    failed += 1
                    continue
                if self.store.submit(request.request_id, response):
                    submitted += 1

        self.last_worker_stats = {
            "processed": int(len(pending)),
            "submitted": int(submitted),
            "failed": int(failed),
        }
        return dict(self.last_worker_stats)

    def summary(self) -> Dict[str, Any]:
        self._reload()
        stats = self.store.get_statistics()
        return {
            "enabled": True,
            "supervision_timing": self.timing_contract(),
            "store_path": str(self.store.storage_path) if self.store.storage_path else None,
            "pending": int(stats.get("pending", 0)),
            "completed": int(stats.get("completed", 0)),
            "total": int(stats.get("total", 0)),
            "sources": dict(stats.get("sources", {}) or {}),
            "total_enqueued": int(self.total_enqueued),
            "total_attached": int(self.total_attached),
            "last_enqueue": dict(self.last_enqueue_stats),
            "last_attach": dict(self.last_attach_stats),
            "last_worker": dict(self.last_worker_stats),
        }

    def timing_contract(self) -> Dict[str, Any]:
        return supervision_timing_contract(
            acquisition_policy=ACQUISITION_ASYNC_FEEDBACK_QUEUE,
            activation_barrier=str(self.config.activation_barrier),
            consumer=CONSUMER_CTREEPO_GRADIENT,
            producer=str(self.config.source_kind),
            delivery_mode="feedback_store",
            blocking=False,
            notes=(
                "Trainer queues sampled node-label requests and keeps training with labels already active.",
                "Completed feedback is attached only at the configured activation barrier.",
            ),
            metadata={
                "target_name": str(self.config.target_name),
                "leaf_budget_per_epoch": int(self.config.leaf_budget_per_epoch),
                "merge_budget_per_epoch": int(self.config.merge_budget_per_epoch),
                "source_spec": self.config.source_spec,
            },
        )

    def _build_request(
        self,
        *,
        item: TreeItem,
        node_index: int,
        node: EmbeddingTreeNode,
        request_id: str,
        split: str,
        epoch: int,
        unit_kind: ObservationUnitKind,
        unit_propensity: float,
    ) -> FeedbackRequest:
        _nodes, document_score, doc_id = item
        sampling = SamplingMetadata(
            document_propensity=1.0,
            unit_propensity=max(1e-12, float(unit_propensity)),
            label_propensity=1.0,
            sampling_scheme="budgeted_random_without_replacement",
            policy_name=str(self.config.sampling_policy_name),
            unit_kind=unit_kind,
            supports_ipw_estimation=True,
            metadata={
                "leaf_budget_per_epoch": int(self.config.leaf_budget_per_epoch),
                "merge_budget_per_epoch": int(self.config.merge_budget_per_epoch),
            },
        )
        return FeedbackRequest(
            request_id=request_id,
            text_a=str(node.text_span),
            reference_score=float(document_score),
            dimensions=[
                FeedbackDimension(
                    kind="scalar",
                    name="score",
                    scale=(float(self.config.target_min), float(self.config.target_max)),
                )
            ],
            node_id=f"node_{int(node_index)}",
            tree_id=str(doc_id),
            source_doc_id=str(doc_id),
            law_type="leaf_preservation" if unit_kind == ObservationUnitKind.LEAF else "merge_preservation",
            sampling=sampling,
            priority=10 if unit_kind == ObservationUnitKind.LEAF else 5,
            context={
                "ctreepo_online_node_oracle": True,
                "target_name": str(self.config.target_name),
                "expected_response_scores_key": "score",
                "split": str(split),
                "epoch": int(epoch),
                "doc_id": str(doc_id),
                "node_index": int(node_index),
                "node_kind": unit_kind.value,
                "level": int(node.level),
                "char_start": int(node.char_start),
                "char_end": int(node.char_end),
                "source_kind": str(self.config.source_kind),
                "source_spec": self.config.source_spec,
                "supervision_timing": self.timing_contract(),
            },
        )

    def _sample_candidates(
        self,
        candidates: Sequence[Tuple[TreeItem, int, EmbeddingTreeNode, str]],
        *,
        budget: int,
    ) -> Tuple[List[Tuple[TreeItem, int, EmbeddingTreeNode, str]], float]:
        rows = list(candidates)
        if not rows or budget <= 0:
            return [], 1.0
        if int(budget) >= len(rows):
            return rows, 1.0
        propensity = float(budget) / float(len(rows))
        shuffled = list(rows)
        self.rng.shuffle(shuffled)
        return shuffled[: int(budget)], propensity

    def _known_request_ids(self) -> set[str]:
        pending_ids = {request.request_id for request in self.store.get_pending(limit=10_000_000)}
        completed_ids = {
            request.request_id
            for request, _response in self.store.get_completed(limit=10_000_000)
        }
        return pending_ids | completed_ids

    @staticmethod
    def _score_from_response(response: FeedbackResponse) -> Optional[float]:
        for key in ("score", "rile", "value"):
            if key in response.scores:
                try:
                    return float(response.scores[key])
                except (TypeError, ValueError):
                    return None
        if response.score_estimate_a is not None:
            try:
                return float(response.score_estimate_a)
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _sampling_for_response(
        sampling: SamplingMetadata,
        response: FeedbackResponse,
    ) -> SamplingMetadata:
        source = str(response.source or "").strip().lower()
        metadata = dict(sampling.metadata or {})
        metadata["label_response_source"] = str(response.source or "")
        if source == "human":
            return sampling.with_updates(
                supports_ipw_estimation=False,
                metadata=metadata,
            )
        return sampling.with_updates(metadata=metadata)

    def _reload(self) -> None:
        if getattr(self.store, "storage_path", None) is not None:
            self.store.reload()


__all__ = [
    "OnlineNodeOracleAttachResult",
    "OnlineNodeOracleQueue",
    "OnlineNodeOracleQueueConfig",
]
