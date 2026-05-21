"""
Compatibility layer for backend-specific proxy adapters.

The general theorem-facing abstractions now live in
``src.tree.compositional_operator``. This module keeps the existing import path
for CTreePO / mergeable-sketch proxy adapters and re-exports the general
operator types under their historical names.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from treepo._research.core.ops_checks import EvidenceStatus, OperatorCapabilityReport
from treepo._research.training.embedding_sketch import MergeableEmbeddingSketch, SketchState
from treepo._research.tree.compositional_operator import (
    CodecReductionOperatorAdapter,
    CompositionalOperator,
    CompositionalPredictorAdapter,
    FunctionalCompositionalOperator,
    FunctionalReductionOperator,
    FunctionalSketchLawOperator,
    ModelCompositionalOperatorAdapter,
    OperatorAssumptionBundle,
    OperatorPrediction,
    ProxyOperator,
    ReductionOperator,
    SketchLawOperator,
    StatePredictor,
    SummaryAutoencoderOperatorAdapter,
    attach_compositional_operator,
    attach_theorem_operator,
    make_deterministic_summary_operator,
    make_text_compositional_operator,
)
from treepo._research.tree.ctreepo_model import CTreePOModel


# Historical alias retained for compatibility.
NeuralOperator = ProxyOperator[Any]


class CTreePOOperatorAdapter:
    """Proxy-only adapter exposing CTreePO scalar readout."""

    def __init__(
        self,
        model: CTreePOModel,
        *,
        head: str = "rile",
        z_score: float = 1.96,
        min_std: float = 0.5,
    ):
        self.model = model
        self.name = "ctreepo"
        self.head = str(head)
        self.z_score = float(z_score)
        self.min_std = float(min_std)
        self.state_dim = int(model.config.sketch_dim)
        self.evidence_status = EvidenceStatus.PROXY_ONLY

    def encode_leaf(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.model.encode_leaf(embedding)

    def merge_states(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.model.merge(left, right)

    def capability_report(self) -> OperatorCapabilityReport:
        return self.model.capability_report()

    def with_compositional_operator(
        self,
        operator: CompositionalOperator[Any, torch.Tensor],
        *,
        name: Optional[str] = None,
    ) -> CompositionalPredictorAdapter[Any, torch.Tensor, torch.Tensor]:
        return attach_compositional_operator(self, operator, name=name)

    def with_theorem_operator(
        self,
        operator: CompositionalOperator[Any, torch.Tensor],
        *,
        name: Optional[str] = None,
    ) -> CompositionalPredictorAdapter[Any, torch.Tensor, torch.Tensor]:
        return self.with_compositional_operator(operator, name=name)

    @torch.no_grad()
    def predict_from_state(self, state: torch.Tensor, **_: Any) -> OperatorPrediction:
        mean, lower, upper, std = self.model.predict_interval(
            state,
            head=self.head,
            z_score=self.z_score,
            min_std=self.min_std,
        )
        norm = self.model.predict_normalized(state, head=self.head)
        conf = self.model.predict_confidence(state, head=self.head)
        return OperatorPrediction(
            mean=float(mean.item()),
            lower=float(lower.item()),
            upper=float(upper.item()),
            std=float(std.item()),
            normalized_mean=float(norm.item()),
            confidence=float(conf.item()),
            evidence_status=self.evidence_status,
            aux={"head": self.head, "evidence_status": self.evidence_status.value},
        )

    @torch.no_grad()
    def predict_from_state_batch(
        self,
        states: torch.Tensor,
    ) -> list[OperatorPrediction]:
        """Batched prediction from a stack of sketch states.

        Args:
            states: ``(N, sketch_dim)`` tensor of node sketches.

        Returns:
            List of N ``OperatorPrediction`` objects.
        """
        # predict_interval already handles batched input via nn.Linear broadcasting
        means, lowers, uppers, stds = self.model.predict_interval(
            states, head=self.head, z_score=self.z_score, min_std=self.min_std,
        )
        norms = self.model.predict_normalized_batch(states, head=self.head)
        confs = self.model.predict_confidence_batch(states, head=self.head)

        results: list[OperatorPrediction] = []
        for i in range(states.shape[0]):
            results.append(OperatorPrediction(
                mean=float(means[i].item()),
                lower=float(lowers[i].item()),
                upper=float(uppers[i].item()),
                std=float(stds[i].item()),
                normalized_mean=float(norms[i].item()),
                confidence=float(confs[i].item()),
                evidence_status=self.evidence_status,
                aux={"head": self.head, "evidence_status": self.evidence_status.value},
            ))
        return results


class MergeableSketchOperatorAdapter:
    """Proxy-only adapter exposing MergeableEmbeddingSketch scalar readout."""

    def __init__(
        self,
        model: MergeableEmbeddingSketch,
        *,
        target_min: float = -100.0,
        target_max: float = 100.0,
        z_score: float = 1.96,
        min_std: float = 0.5,
    ):
        self.model = model
        self.name = "mergeable_sketch"
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        self.z_score = float(z_score)
        self.min_std = float(min_std)
        self.state_dim = int(model.config.state_dim)
        self.evidence_status = EvidenceStatus.PROXY_ONLY

    def encode_windows(
        self,
        window_embeddings: torch.Tensor,
        *,
        counts: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> SketchState:
        return self.model.encode_windows(window_embeddings, counts=counts, mask=mask)

    def merge_states(self, left: SketchState, right: SketchState) -> SketchState:
        return left.merge(right)

    def capability_report(self) -> OperatorCapabilityReport:
        return self.model.capability_report()

    def with_compositional_operator(
        self,
        operator: CompositionalOperator[Any, SketchState],
        *,
        name: Optional[str] = None,
    ) -> CompositionalPredictorAdapter[Any, SketchState, SketchState]:
        return attach_compositional_operator(self, operator, name=name)

    def with_theorem_operator(
        self,
        operator: CompositionalOperator[Any, SketchState],
        *,
        name: Optional[str] = None,
    ) -> CompositionalPredictorAdapter[Any, SketchState, SketchState]:
        return self.with_compositional_operator(operator, name=name)

    @torch.no_grad()
    def predict_from_state(
        self,
        state: SketchState,
        *,
        meta_embeddings: Optional[torch.Tensor] = None,
        retrieval_features: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> OperatorPrediction:
        out = self.model.predict_from_state(
            state,
            meta_embeddings=meta_embeddings,
            retrieval_features=retrieval_features,
            return_dict=True,
        )
        norm = out.get("rile")
        if norm is None:
            raise RuntimeError("Mergeable sketch returned no rile prediction")
        norm = torch.clamp(norm, min=1e-6, max=1.0 - 1e-6)
        if norm.ndim > 0:
            norm = norm.reshape(-1)[0]
        norm_f = float(norm.item())
        span = self.target_max - self.target_min
        mean = self.target_min + span * norm_f
        std = max(self.min_std, span * float((norm * (1.0 - norm)).sqrt().item()))
        lower = max(self.target_min, mean - self.z_score * std)
        upper = min(self.target_max, mean + self.z_score * std)
        confidence = float(max(0.0, min(1.0, 1.0 - 2.0 * abs(norm_f - 0.5))))

        aux: dict[str, Any] = {"evidence_status": self.evidence_status.value}
        delta = out.get("delta")
        if delta is not None:
            delta_t = delta.reshape(-1)[0] if delta.ndim > 0 else delta
            aux["delta"] = float(delta_t.item())

        return OperatorPrediction(
            mean=float(mean),
            lower=float(lower),
            upper=float(upper),
            std=float(std),
            normalized_mean=float(norm_f),
            confidence=confidence,
            evidence_status=self.evidence_status,
            aux=aux,
        )


__all__ = [
    "CompositionalOperator",
    "ReductionOperator",
    "StatePredictor",
    "ProxyOperator",
    "NeuralOperator",
    "OperatorAssumptionBundle",
    "OperatorPrediction",
    "CompositionalPredictorAdapter",
    "attach_compositional_operator",
    "attach_theorem_operator",
    "SketchLawOperator",
    "FunctionalCompositionalOperator",
    "FunctionalReductionOperator",
    "FunctionalSketchLawOperator",
    "ModelCompositionalOperatorAdapter",
    "CodecReductionOperatorAdapter",
    "SummaryAutoencoderOperatorAdapter",
    "make_text_compositional_operator",
    "make_deterministic_summary_operator",
    "CTreePOOperatorAdapter",
    "MergeableSketchOperatorAdapter",
]
