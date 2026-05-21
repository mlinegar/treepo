"""
Generalized feedback types for ThinkingTrees.

This module provides type-agnostic feedback collection -- supporting pairwise
preferences, scalar ratings, written critiques, and arbitrary combinations.
All feedback types carry IPW propensity annotations from the audit sampling
design, enabling unbiased downstream estimation and training.

The key abstraction is the FeedbackRequest/FeedbackResponse pair:
- FeedbackRequest declares what feedback is wanted (via FeedbackDimension)
- FeedbackResponse carries whatever the collector provides

Responses are always convertible to:
- SupervisionDataset / ResponseJudgment / ComparativeJudgment
- BinaryComparison (backward-compatible binary projection)
- DSPy metric dict {'score': float, 'feedback': str} (optimizer-compatible)
- FlaggedItem update fields (human review bridge)

Usage:
    from treepo._research.feedback import FeedbackRequest, FeedbackResponse, FeedbackDimension

    # Create a pairwise request
    request = FeedbackRequest(
        request_id="req_1",
        text_a="Summary A...",
        text_b="Summary B...",
        original_text="Original document...",
        rubric="Preserve political positions",
    )

    # Or a scalar rating request
    request = FeedbackRequest(
        request_id="req_2",
        text_a="Summary to rate...",
        original_text="Original document...",
        rubric="Rate faithfulness 1-5",
        dimensions=[FeedbackDimension(kind="scalar", name="faithfulness", scale=(1.0, 5.0))],
    )

    # Responses are flexible
    response = FeedbackResponse(
        request_id="req_1",
        preferred="A",
        scores={"faithfulness": 4.2},
        critique="Summary A better preserves the key arguments.",
        confidence=0.85,
        source="llm_judge",
    )

    # Convert to DSPy metric
    metric = response.to_dspy_metric()
    # {'score': 4.2, 'feedback': 'Summary A better preserves the key arguments.'}

    # Convert to a binary comparison
    pair = response.to_binary_comparison(request)
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from treepo._research.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from treepo._research.core.supervision_metadata import judgment_supervision_metadata
from treepo._research.training.supervision import (
    BinaryComparison,
    BinaryProjectionDataset,
    ResponseJudgment,
    SupervisionDataset,
)

logger = logging.getLogger(__name__)

DEFAULT_PROPENSITY = 1.0
MIN_PROPENSITY = 1e-8


# =============================================================================
# FeedbackDimension
# =============================================================================

@dataclass
class FeedbackDimension:
    """A single dimension of feedback being requested.

    The system is agnostic about types -- these are the built-in kinds,
    but 'custom' is always available for arbitrary structured feedback.

    Built-in kinds:
        pairwise: Compare text_a vs text_b, return preferred A/B/tie
        scalar: Rate on a numeric scale, return score
        critique: Provide written feedback, return critique text
        custom: Arbitrary structured feedback via extra dict
    """
    kind: str  # "pairwise", "scalar", "critique", "custom"
    name: Optional[str] = None  # e.g., "helpfulness", "faithfulness"
    scale: Optional[Tuple[float, float]] = None  # (min, max) for scalar
    options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": self.kind}
        if self.name is not None:
            d["name"] = self.name
        if self.scale is not None:
            d["scale"] = list(self.scale)
        if self.options:
            d["options"] = self.options
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackDimension":
        d = dict(data)
        if "scale" in d and d["scale"] is not None:
            d["scale"] = tuple(d["scale"])
        return cls(**d)


# =============================================================================
# FeedbackRequest
# =============================================================================

@dataclass
class FeedbackRequest:
    """Context for requesting feedback on a tree node or comparison.

    Agnostic about what kind of feedback is requested. The ``dimensions``
    field declares what the requester wants; the collector provides what
    it can. If ``dimensions`` is empty, the request auto-infers from content:
    pairwise if ``text_b`` is set, scalar otherwise.

    IPW propensity fields are propagated from the audit sampling design
    (``AuditReport.inclusion_probability_map``) so that downstream
    estimators and training can apply inverse-probability weighting.
    """
    # Identity
    request_id: str

    # Content to evaluate
    text_a: str = ""
    text_b: Optional[str] = None
    original_text: str = ""
    rubric: str = ""
    reference_score: Optional[float] = None

    # What feedback is requested
    dimensions: List[FeedbackDimension] = field(default_factory=list)

    # Tree/audit context
    node_id: Optional[str] = None
    tree_id: Optional[str] = None
    source_doc_id: Optional[str] = None
    law_type: str = "sufficiency"

    # IPW propensity (propagated from audit)
    sampling: SamplingMetadata = field(
        default_factory=lambda: SamplingMetadata(unit_kind=ObservationUnitKind.PAIR)
    )

    # Metadata
    priority: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if not self.dimensions:
            if self.text_b is not None:
                self.dimensions = [FeedbackDimension(kind="pairwise")]
            else:
                self.dimensions = [FeedbackDimension(kind="scalar")]
        if not isinstance(self.sampling, SamplingMetadata):
            self.sampling = SamplingMetadata.from_dict(self.sampling)
        if self.sampling.unit_kind is None:
            self.sampling = self.sampling.with_updates(unit_kind=ObservationUnitKind.PAIR)

    @property
    def is_pairwise(self) -> bool:
        return any(d.kind == "pairwise" for d in self.dimensions)

    @property
    def joint_propensity(self) -> float:
        return self.sampling.effective_joint_propensity(min_propensity=0.0)

    @classmethod
    def from_flagged_item(cls, item: Any) -> "FeedbackRequest":
        """Create a FeedbackRequest from an existing FlaggedItem.

        Bridges the audit review queue to the generalized feedback system.
        """
        return cls(
            request_id=f"flag_{item.item_id}",
            text_a=item.input_a,
            text_b=item.input_b if item.input_b else None,
            rubric=item.rubric,
            node_id=item.node_id,
            tree_id=item.tree_id,
            law_type=item.check_type,
            priority=getattr(item, "priority", type("", (), {"value": 0})).value
            if hasattr(getattr(item, "priority", None), "value")
            else 0,
            dimensions=[
                FeedbackDimension(kind="pairwise")
                if item.input_b
                else FeedbackDimension(kind="scalar"),
                FeedbackDimension(kind="critique"),
            ],
            context={
                "approx_discrepancy": item.approx_discrepancy,
                "approx_reasoning": item.approx_reasoning,
                "node_level": getattr(item, "node_level", 0),
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "text_a": self.text_a,
            "text_b": self.text_b,
            "original_text": self.original_text,
            "rubric": self.rubric,
            "reference_score": self.reference_score,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "node_id": self.node_id,
            "tree_id": self.tree_id,
            "source_doc_id": self.source_doc_id,
            "law_type": self.law_type,
            "sampling": self.sampling.to_dict(),
            "priority": self.priority,
            "context": self.context,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackRequest":
        d = dict(data)
        if "dimensions" in d:
            d["dimensions"] = [FeedbackDimension.from_dict(dim) for dim in d["dimensions"]]
        if "sampling" not in d:
            d["sampling"] = {
                "document_propensity": d.pop("doc_propensity", DEFAULT_PROPENSITY),
                "unit_propensity": d.pop("node_propensity", DEFAULT_PROPENSITY),
                "label_propensity": d.pop("label_propensity", DEFAULT_PROPENSITY),
                "sampling_scheme": d.pop("sampling_scheme", None),
                "unit_kind": "pair",
            }
        return cls(**d)


# =============================================================================
# FeedbackResponse
# =============================================================================

@dataclass
class FeedbackResponse:
    """Multi-dimensional feedback response.

    Can carry any combination of:
    - Pairwise preference (A/B/tie)
    - Scalar rating(s) keyed by dimension name
    - Written critique / natural language feedback
    - Arbitrary structured data in ``extra``

    Always convertible to:
    - DSPy metric dict via ``to_dspy_metric()``
    - ResponseJudgment / ComparativeJudgment via supervision helpers
    - BinaryComparison via ``to_binary_comparison(request)``
    - PreferencePair via ``to_preference_pair(request)`` for compatibility
    - FlaggedItem update via ``to_flagged_item_update()``
    """
    # Identity (matches request)
    request_id: str

    # Pairwise preference
    preferred: Optional[Literal["A", "B", "tie"]] = None

    # Scalar ratings (dimension_name -> value)
    scores: Dict[str, float] = field(default_factory=dict)

    # Written critique
    critique: str = ""

    # Reasoning / confidence
    reasoning: str = ""
    confidence: float = 0.5
    score_estimate_a: Optional[float] = None
    score_estimate_b: Optional[float] = None

    # Arbitrary structured data
    extra: Dict[str, Any] = field(default_factory=dict)

    # Source metadata
    source: str = "unknown"
    judge_model: str = ""
    timestamp: Optional[str] = None
    raw_result: Optional[Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    @classmethod
    def from_human_feedback(
        cls,
        *,
        request_id: str,
        preferred: Optional[Literal["A", "B", "tie"]] = None,
        scores: Optional[Dict[str, float]] = None,
        critique: str = "",
        reasoning: str = "",
        confidence: float = 1.0,
        score_estimate_a: Optional[float] = None,
        score_estimate_b: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        judge_model: str = "",
    ) -> "FeedbackResponse":
        """Create a canonical human-sourced response for programmatic or API use."""
        return cls(
            request_id=request_id,
            preferred=preferred,
            scores=dict(scores or {}),
            critique=critique,
            reasoning=reasoning,
            confidence=confidence,
            score_estimate_a=score_estimate_a,
            score_estimate_b=score_estimate_b,
            extra=dict(extra or {}),
            source="human",
            judge_model=judge_model,
        )

    @classmethod
    def from_human_pairwise_feedback(
        cls,
        *,
        request_id: str,
        preferred: Literal["A", "B", "tie"],
        reasoning: str = "",
        critique: str = "",
        confidence: float = 1.0,
        score_estimate_a: Optional[float] = None,
        score_estimate_b: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "FeedbackResponse":
        """Create a human pairwise response that can project to binary/comparative supervision."""
        return cls.from_human_feedback(
            request_id=request_id,
            preferred=preferred,
            reasoning=reasoning,
            critique=critique,
            confidence=confidence,
            score_estimate_a=score_estimate_a,
            score_estimate_b=score_estimate_b,
            extra=extra,
        )

    @classmethod
    def from_human_scalar_feedback(
        cls,
        *,
        request_id: str,
        score: float,
        dimension_name: str = "score",
        reasoning: str = "",
        critique: str = "",
        confidence: float = 1.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "FeedbackResponse":
        """Create a human scalar response that can project to response supervision."""
        return cls.from_human_feedback(
            request_id=request_id,
            scores={str(dimension_name): float(score)},
            reasoning=reasoning,
            critique=critique,
            confidence=confidence,
            extra=extra,
        )

    def _combined_reasoning(self) -> str:
        reasoning = str(self.reasoning or "").strip()
        critique = str(self.critique or "").strip()
        parts: List[str] = []
        if reasoning:
            parts.append(reasoning)
        if critique and (not reasoning or critique not in reasoning):
            parts.append(critique)
        return "\n".join(parts).strip()

    # --- DSPy compatibility ---

    def to_dspy_metric(self) -> Dict[str, Any]:
        """Convert to DSPy metric format: {'score': float, 'feedback': str}.

        Score derivation priority:
        1. ``scores["score"]`` if present
        2. Mean of all entries in ``scores`` if non-empty
        3. Preference + confidence mapping if ``preferred`` is set
        4. 0.5 (neutral) as fallback
        """
        if "score" in self.scores:
            score = self.scores["score"]
        elif self.scores:
            score = sum(self.scores.values()) / len(self.scores)
        elif self.preferred is not None:
            if self.preferred == "A":
                score = 0.5 + self.confidence * 0.5
            elif self.preferred == "B":
                score = 0.5 - self.confidence * 0.5
            else:
                score = 0.5
        else:
            score = 0.5

        feedback = self._combined_reasoning()
        return {"score": score, "feedback": feedback}

    def to_response_judgment(
        self,
        request: "FeedbackRequest",
        *,
        response_id: Optional[str] = None,
        response_text: Optional[str] = None,
        score_value: Optional[float] = None,
    ) -> ResponseJudgment:
        """Convert scalar feedback into a canonical response judgment."""
        scalar_dimension = next(
            (dimension for dimension in request.dimensions if dimension.kind == "scalar"),
            None,
        )
        signal_name = (
            (scalar_dimension.name if scalar_dimension and scalar_dimension.name else None)
            or self.extra.get("response_signal_name")
            or next(iter(self.scores.keys()), None)
            or "response_score"
        )
        signal_min: Optional[float] = None
        signal_max: Optional[float] = None
        if scalar_dimension is not None and scalar_dimension.scale is not None:
            signal_min, signal_max = scalar_dimension.scale
        else:
            signal_min = self.extra.get("response_signal_min")
            signal_max = self.extra.get("response_signal_max")

        if score_value is None:
            if scalar_dimension and scalar_dimension.name and scalar_dimension.name in self.scores:
                score_value = self.scores[scalar_dimension.name]
            elif self.scores:
                score_value = next(iter(self.scores.values()))
            elif self.score_estimate_a is not None:
                score_value = self.score_estimate_a
            else:
                metric = self.to_dspy_metric()
                score_value = metric["score"]

        resolved_response_id = response_id or ("A" if request.text_b is None else "A")
        resolved_response_text = response_text or request.text_a
        return ResponseJudgment(
            judgment_id=f"{self.request_id}:{resolved_response_id}",
            source_example_id=request.node_id or request.request_id,
            original_text=request.original_text,
            rubric=request.rubric,
            response=resolved_response_text,
            response_id=resolved_response_id,
            reference_score=request.reference_score or 0.0,
            law_type=request.law_type,
            source_doc_id=request.source_doc_id,
            sampling=request.sampling,
            supervision_metadata=judgment_supervision_metadata(
                application_name="feedback_collection",
                law_type=request.law_type,
                response_signal_name=signal_name,
                response_signal_min=signal_min,
                response_signal_max=signal_max,
                metadata={
                    "request_id": request.request_id,
                    "feedback_source": self.source,
                },
            ),
            response_signal_value=score_value,
            judge_model=self.judge_model,
            timestamp=self.timestamp,
            truth_label_source=self.source,
            metadata={
                "reasoning": self.reasoning,
                "critique": self.critique,
                "source": self.source,
                **dict(self.extra),
            },
        )

    def to_comparative_judgment(
        self,
        request: "FeedbackRequest",
        pair_id: Optional[str] = None,
    ) -> Any:
        """Convert pairwise feedback into a canonical comparative judgment."""
        if request.text_b is None:
            raise ValueError("Comparative judgments require pairwise feedback with text_b.")
        return self.to_binary_comparison(request, pair_id=pair_id).to_comparative_judgment()

    # --- Binary comparison compatibility ---

    def to_binary_comparison(
        self,
        request: "FeedbackRequest",
        pair_id: Optional[str] = None,
    ) -> BinaryComparison:
        """Convert pairwise human/LLM/oracle feedback into a canonical binary comparison."""
        combined_reasoning = self._combined_reasoning()
        supervision = judgment_supervision_metadata(
            application_name="feedback_collection",
            supervision_channel_name="judgment_supervision",
            supervision_signal_name="judgment",
            preference_family="pairwise",
            law_type=request.law_type,
            comparison_signal_name=self.extra.get("comparison_signal_name"),
            comparison_signal_min=self.extra.get("comparison_signal_min"),
            comparison_signal_max=self.extra.get("comparison_signal_max"),
            response_signal_name=self.extra.get("response_signal_name"),
            response_signal_min=self.extra.get("response_signal_min"),
            response_signal_max=self.extra.get("response_signal_max"),
            metadata={
                "request_id": request.request_id,
                "feedback_source": self.source,
            },
        )

        return BinaryComparison(
            pair_id=pair_id or self.request_id,
            source_example_id=request.node_id or request.request_id,
            original_text=request.original_text,
            rubric=request.rubric,
            reference_score=request.reference_score or 0.0,
            summary_a=request.text_a,
            summary_b=request.text_b or "",
            preferred=self.preferred or "tie",
            reasoning=combined_reasoning,
            confidence=self.confidence,
            score_estimate_a=self.score_estimate_a,
            score_estimate_b=self.score_estimate_b,
            comparison_signal_value=self.extra.get("comparison_signal_value"),
            judge_model=self.judge_model,
            sampling=request.sampling,
            preference_supervision=supervision,
            source_doc_id=request.source_doc_id,
            truth_label_source=self.source,
            source_observation_ids=[request.request_id],
            law_type=request.law_type,
        )

    def to_preference_pair(
        self,
        request: "FeedbackRequest",
        pair_id: Optional[str] = None,
    ) -> Any:
        """Backward-compatible alias for ``to_binary_comparison``."""
        return self.to_binary_comparison(request, pair_id=pair_id)

    # --- FlaggedItem compatibility ---

    def to_flagged_item_update(self) -> Dict[str, Any]:
        """Return fields suitable for updating a FlaggedItem after review."""
        # Derive approval from preference or score
        if self.preferred is not None:
            approved = self.preferred != "B"
        elif self.scores:
            mean_score = sum(self.scores.values()) / len(self.scores)
            approved = mean_score >= 0.5
        else:
            approved = True

        review_reasoning = self._combined_reasoning()

        return {
            "reviewed": True,
            "review_result": approved,
            "review_reasoning": review_reasoning,
            "corrected_summary": self.extra.get("corrected_summary"),
            "reviewed_at": self.timestamp,
            "review_source": self.source,
        }

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "request_id": self.request_id,
            "preferred": self.preferred,
            "scores": self.scores,
            "critique": self.critique,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "score_estimate_a": self.score_estimate_a,
            "score_estimate_b": self.score_estimate_b,
            "extra": self.extra,
            "source": self.source,
            "judge_model": self.judge_model,
            "timestamp": self.timestamp,
        }
        # Skip raw_result in serialization (may not be JSON-safe)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackResponse":
        d = dict(data)
        d.pop("raw_result", None)
        return cls(**d)


# =============================================================================
# FeedbackDataset
# =============================================================================

class FeedbackDataset:
    """Collection of feedback request/response pairs with export capabilities.

    Provides conversion to SupervisionDataset, binary compatibility exports,
    DSPy examples, reward model format, and propensity diagnostics.
    """

    def __init__(
        self,
        items: Optional[List[Tuple[FeedbackRequest, FeedbackResponse]]] = None,
    ):
        self.items: List[Tuple[FeedbackRequest, FeedbackResponse]] = items or []

    def add(self, request: FeedbackRequest, response: FeedbackResponse) -> None:
        self.items.append((request, response))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[FeedbackRequest, FeedbackResponse]:
        return self.items[idx]

    # --- Filtering ---

    def filter_pairwise(self) -> "FeedbackDataset":
        """Return only pairwise items (where text_b is set and preferred is set)."""
        return FeedbackDataset([
            (req, resp) for req, resp in self.items
            if req.is_pairwise and resp.preferred is not None
        ])

    def filter_by_source(self, source: str) -> "FeedbackDataset":
        """Return items from a specific source."""
        return FeedbackDataset([
            (req, resp) for req, resp in self.items
            if resp.source == source
        ])

    def filter_by_confidence(self, min_confidence: float) -> "FeedbackDataset":
        """Return items above confidence threshold."""
        return FeedbackDataset([
            (req, resp) for req, resp in self.items
            if resp.confidence >= min_confidence
        ])

    # --- Export to existing formats ---

    def to_preference_dataset(self) -> Any:
        """Compatibility wrapper over the primary supervision dataset."""
        return self.to_binary_projection_dataset(projection="adjacent")

    def to_binary_projection_dataset(
        self,
        *,
        projection: str = "adjacent",
    ) -> BinaryProjectionDataset:
        """Convert completed feedback directly into the canonical binary projection dataset."""
        return self.to_supervision_dataset().project_binary(projection=projection)

    def to_supervision_dataset(
        self,
        *,
        include_pairwise_response_scores: bool = True,
    ) -> SupervisionDataset:
        """Convert mixed feedback into the primary supervision surface."""
        dataset = SupervisionDataset()
        for request, response in self.items:
            if request.is_pairwise and response.preferred is not None:
                dataset.add_comparative_judgment(
                    response.to_comparative_judgment(request)
                )
                if include_pairwise_response_scores:
                    if response.score_estimate_a is not None:
                        dataset.add_response_judgment(
                            response.to_response_judgment(
                                request,
                                response_id="A",
                                response_text=request.text_a,
                                score_value=response.score_estimate_a,
                            )
                        )
                    if (
                        request.text_b is not None
                        and response.score_estimate_b is not None
                    ):
                        dataset.add_response_judgment(
                            response.to_response_judgment(
                                request,
                                response_id="B",
                                response_text=request.text_b,
                                score_value=response.score_estimate_b,
                            )
                        )
            elif not request.is_pairwise:
                dataset.add_response_judgment(response.to_response_judgment(request))
        return dataset

    def to_dspy_examples(self) -> list:
        """Convert to DSPy examples with score/feedback fields.

        Each example includes the request context as inputs and the
        response score/feedback as outputs.
        """
        try:
            import dspy
        except ImportError:
            logger.warning("dspy not available; returning empty list")
            return []

        examples = []
        for request, response in self.items:
            metric = response.to_dspy_metric()
            example = dspy.Example(
                original_text=request.original_text,
                rubric=request.rubric,
                text_a=request.text_a,
                text_b=request.text_b or "",
                law_type=request.law_type,
                reference_score=request.reference_score or 0.0,
                score=metric["score"],
                feedback=metric["feedback"],
                preferred=response.preferred or "",
                confidence=response.confidence,
                sample_weight=1.0 / max(MIN_PROPENSITY, request.joint_propensity),
            ).with_inputs(
                "original_text", "rubric", "text_a", "text_b",
                "law_type", "reference_score",
            )
            examples.append(example)
        return examples

    def to_reward_model_format(self) -> List[Dict[str, Any]]:
        """Export scalar reward-model rows from the primary supervision surface."""
        return self.to_supervision_dataset().to_scalar_reward_records()

    # --- Propensity diagnostics ---

    def propensity_diagnostics(self) -> Dict[str, Any]:
        """Compute IPW diagnostics from request propensities."""
        if not self.items:
            return {
                "n_items": 0,
                "effective_sample_size": 0.0,
                "effective_sample_ratio": 0.0,
            }

        weights = []
        for request, _ in self.items:
            prop = max(MIN_PROPENSITY, request.joint_propensity)
            weights.append(1.0 / prop)

        n = len(weights)
        sum_w = sum(weights)
        sum_w_sq = sum(w * w for w in weights)
        neff = (sum_w * sum_w / sum_w_sq) if sum_w_sq > 0 else 0.0

        return {
            "n_items": n,
            "effective_sample_size": neff,
            "effective_sample_ratio": neff / n if n > 0 else 0.0,
            "mean_weight": sum_w / n,
            "min_weight": min(weights),
            "max_weight": max(weights),
        }

    # --- Serialization ---

    def save(self, path: Path) -> None:
        """Save dataset to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "n_items": len(self.items),
            "items": [
                {"request": req.to_dict(), "response": resp.to_dict()}
                for req, resp in self.items
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d feedback items to %s", len(self.items), path)

    @classmethod
    def load(cls, path: Path) -> "FeedbackDataset":
        """Load dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)
        items = []
        for entry in data.get("items", []):
            req = FeedbackRequest.from_dict(entry["request"])
            resp = FeedbackResponse.from_dict(entry["response"])
            items.append((req, resp))
        logger.info("Loaded %d feedback items from %s", len(items), path)
        return cls(items)

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics."""
        pairwise = [(req, resp) for req, resp in self.items if req.is_pairwise]
        scored = [(req, resp) for req, resp in self.items if resp.scores]
        with_critique = [(req, resp) for req, resp in self.items if resp.critique]
        sources = {}
        for _, resp in self.items:
            sources[resp.source] = sources.get(resp.source, 0) + 1

        diag = self.propensity_diagnostics()
        return {
            "total_items": len(self.items),
            "pairwise_items": len(pairwise),
            "scored_items": len(scored),
            "items_with_critique": len(with_critique),
            "sources": sources,
            "avg_confidence": (
                sum(resp.confidence for _, resp in self.items) / len(self.items)
                if self.items else 0.0
            ),
            "effective_sample_size": diag["effective_sample_size"],
            "effective_sample_ratio": diag["effective_sample_ratio"],
        }
