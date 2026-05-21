"""
Shared comparative and binary-projection types for supervision training.

This module contains the implementation for optimizer-facing binary projections,
comparative judgments, prompt rendering, and legacy deriver helpers. The
primary public abstraction still lives in ``src.training.supervision.types``.

Preference Learning Framework
-----------------------------
The framework is designed to be method-agnostic, supporting modern preference
learning methods beyond the original DPO (Direct Preference Optimization):

- **DPO** (Direct Preference Optimization): Pairwise preferences with sigmoid loss
- **GRPO** (Group Relative Policy Optimization): Group-wise rankings (DeepSeek style)
- **PPO** (Proximal Policy Optimization): Reward-based training
- **RLHF** (Reinforcement Learning from Human Feedback): Reward models

The primary public abstraction now lives in `src.training.supervision`.
`PreferencePair` and `PreferenceDataset` remain as compatibility projections
for binary optimizers and older callers.

Theoretical Foundation
----------------------
The preference learning guarantees are proven in the Lean formalization:
- PreferenceLearning.lean: Abstract preference learning framework
- DPO.lean: DPO as a concrete instance of the framework

When the local laws (L1, L2, L3) hold, preference learning on summarized
data is equivalent to preference learning on original data. This applies
to ANY preference learning method that satisfies oracle-measurability.

Available Derivers
------------------
- JudgeDeriver: Uses LLM judge (DSPy PairwiseJudge) for comparison
- GenRMDeriver: Uses NVIDIA GenRM model for comparison
- OracleDeriver: Uses oracle scores to derive preferences

Usage:
    from treepo._research.training.supervision.comparative_types import (
        get_deriver,
        JudgeDeriver,
        GenRMDeriver,
        OracleDeriver,
    )

    # Get a deriver by name
    deriver = get_deriver("genrm", judge=my_genrm_judge)

    # Derive preference
    result = deriver.derive(
        summary_a="...",
        summary_b="...",
        context="Preserve political position...",
        original_text="...",
    )

    # Export to various preference learning formats
    dataset = PreferenceDataset(pairs)
    dpo_data = dataset.to_preference_format(method="dpo")
    grpo_data = dataset.to_preference_format(method="grpo")
"""

import json
import logging
import math
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple, Type, runtime_checkable

import dspy

from treepo._research.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from treepo._research.core.supervision_metadata import (
    JudgmentSupervisionMetadata as PreferenceSupervisionMetadata,
    judgment_supervision_metadata as preference_supervision_metadata,
)
from treepo._research.core.prompting import default_summarize_prompt
from treepo._research.core.provenance import normalize_truth_label_source
from treepo._research.training.supervision.optimizer_metadata import (
    TreePORLRole,
    TreePOWeightingMode,
    build_treepo_optimizer_export_metadata,
)
from treepo._research.stats.sampling import (
    largest_remainder_allocation as _largest_remainder_allocation,
    pps_inclusion_probabilities as _pps_inclusion_probabilities,
    systematic_pps_sample_indices as _systematic_pps_sample_indices,
)

logger = logging.getLogger(__name__)

DEFAULT_GLOBAL_PROPENSITY = 1.0
MIN_PROPENSITY = 1e-8
MAX_PROPENSITY = 1.0


PromptBuilder = Callable[[str, str], Any]


def render_prompt(
    text: str,
    rubric: str,
    prompt_builder: Optional[PromptBuilder] = None,
) -> str:
    """Render a prompt string using a prompt builder or the default template."""
    builder = prompt_builder or default_summarize_prompt
    prompt = builder(text, rubric)
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for msg in prompt:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content", "")
                if role:
                    parts.append(f"{role}: {content}")
                else:
                    parts.append(str(content))
            else:
                parts.append(str(msg))
        return "\n".join(parts)
    return str(prompt)


def compute_propensity_diagnostics(
    pairs: List["PreferencePair"],
    *,
    include_ties: bool = True,
    min_propensity: float = MIN_PROPENSITY,
    max_weight: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute propensity/IPW diagnostics for a set of preference pairs.

    Used for reporting effective sample size and weight concentration in
    `final_stats.json` for judge/generator training subsets.
    """
    if include_ties:
        used_pairs = list(pairs)
    else:
        used_pairs = [pair for pair in pairs if pair.preferred != "tie"]

    n_total = len(pairs)
    n_used = len(used_pairs)

    if n_used == 0:
        return {
            "n_pairs_total": n_total,
            "n_pairs_used": 0,
            "n_ties_excluded": n_total,
            "include_ties": include_ties,
            "effective_sample_size": 0.0,
            "effective_sample_ratio": 0.0,
            "mean_joint_propensity": 0.0,
            "min_joint_propensity": 0.0,
            "max_joint_propensity": 0.0,
            "mean_sample_weight": 0.0,
            "min_sample_weight": 0.0,
            "max_sample_weight": 0.0,
            "sum_sample_weight": 0.0,
            "max_weight_clip": max_weight,
        }

    propensities = [
        pair.effective_joint_propensity(min_propensity=min_propensity)
        for pair in used_pairs
    ]
    weights = [
        pair.ipw_weight(min_propensity=min_propensity, max_weight=max_weight)
        for pair in used_pairs
    ]

    sum_w = sum(weights)
    sum_w_sq = sum(weight * weight for weight in weights)
    neff = (sum_w * sum_w / sum_w_sq) if sum_w_sq > 0 else 0.0
    neff_ratio = (neff / n_used) if n_used > 0 else 0.0

    return {
        "n_pairs_total": n_total,
        "n_pairs_used": n_used,
        "n_ties_excluded": n_total - n_used,
        "include_ties": include_ties,
        "effective_sample_size": neff,
        "effective_sample_ratio": neff_ratio,
        "mean_joint_propensity": sum(propensities) / n_used,
        "min_joint_propensity": min(propensities),
        "max_joint_propensity": max(propensities),
        "mean_sample_weight": sum_w / n_used,
        "min_sample_weight": min(weights),
        "max_sample_weight": max(weights),
        "sum_sample_weight": sum_w,
        "max_weight_clip": max_weight,
    }


# =============================================================================
# PreferenceDeriver Protocol
# =============================================================================

@dataclass
class PreferenceDerivationResult:
    """Result from preference derivation."""
    preferred: Literal["A", "B", "tie"]
    confidence: float  # 0.0 to 1.0
    reasoning: str = ""
    score_estimate_a: Optional[float] = None
    score_estimate_b: Optional[float] = None
    comparison_signal_value: Optional[float] = None
    comparison_signal_name: Optional[str] = None
    comparison_signal_min: Optional[float] = None
    comparison_signal_max: Optional[float] = None
    response_signal_name: Optional[str] = None
    response_signal_min: Optional[float] = None
    response_signal_max: Optional[float] = None
    raw_result: Optional[Any] = None


@runtime_checkable
class PreferenceDeriver(Protocol):
    """
    Protocol for preference derivation strategies.

    Derivers compare two summaries and determine which better preserves
    task-relevant information. Different implementations use different
    comparison mechanisms (LLM judge, GenRM, oracle scores).
    """

    def derive(
        self,
        summary_a: str,
        summary_b: str,
        context: str,
        original_text: str,
        reference_score: Optional[float] = None,
        law_type: str = "sufficiency",
        **kwargs,
    ) -> PreferenceDerivationResult:
        """
        Derive preference between two summaries.

        Args:
            summary_a: First candidate summary
            summary_b: Second candidate summary
            context: Description of what information to preserve (rubric)
            original_text: Original text being summarized
            reference_score: Ground truth score for original text (if available)
            law_type: OPS law type ("sufficiency", "idempotence", "merge")
            **kwargs: Additional arguments for specific derivers

        Returns:
            PreferenceDerivationResult with preference, confidence, and reasoning
        """
        ...


# =============================================================================
# Deriver Registry
# =============================================================================

_DERIVER_REGISTRY: Dict[str, Type["PreferenceDeriver"]] = {}


def register_deriver(name: str):
    """Decorator to register a deriver class."""
    def decorator(cls: Type[PreferenceDeriver]):
        _DERIVER_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_deriver(name: str, **kwargs) -> PreferenceDeriver:
    """
    Get a preference deriver by name.

    Args:
        name: Deriver name ("judge", "genrm", "oracle")
        **kwargs: Arguments passed to deriver constructor

    Returns:
        Configured deriver instance

    Raises:
        ValueError: If deriver name is not registered
    """
    name_lower = name.lower()
    if name_lower not in _DERIVER_REGISTRY:
        available = list(_DERIVER_REGISTRY.keys())
        raise ValueError(f"Unknown deriver: '{name}'. Available: {available}")

    return _DERIVER_REGISTRY[name_lower](**kwargs)


def list_derivers() -> List[str]:
    """Return list of registered deriver names."""
    return list(_DERIVER_REGISTRY.keys())


# =============================================================================
# Deriver Implementations
# =============================================================================

@register_deriver("judge")
class JudgeDeriver:
    """
    Preference deriver using LLM judge (DSPy PairwiseJudge).

    Uses chain-of-thought reasoning to determine which summary
    better preserves the target information.
    """

    def __init__(self, judge: Optional[Any] = None, use_cot: bool = True):
        """
        Initialize the judge deriver.

        Args:
            judge: Optional pre-initialized PairwiseJudge. If None, creates one.
            use_cot: Whether to use chain-of-thought reasoning
        """
        self.judge = judge
        self.use_cot = use_cot

    def _ensure_judge(self):
        """Lazily create judge if not provided."""
        if self.judge is None:
            from treepo._research.training.supervision.collector import PairwiseJudge
            self.judge = PairwiseJudge(use_cot=self.use_cot)
        return self.judge

    def derive(
        self,
        summary_a: str,
        summary_b: str,
        context: str,
        original_text: str,
        reference_score: Optional[float] = None,
        law_type: str = "sufficiency",
        **kwargs,
    ) -> PreferenceDerivationResult:
        """Derive preference using LLM judge."""
        judge = self._ensure_judge()

        result = judge.forward(
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            rubric=context,
            reference_score=reference_score or 0.0,
        )

        return PreferenceDerivationResult(
            preferred=result.get("preferred", "tie"),
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", ""),
            score_estimate_a=result.get("score_estimate_a"),
            score_estimate_b=result.get("score_estimate_b"),
            response_signal_name=(
                "judge_score_estimate"
                if (
                    result.get("score_estimate_a") is not None
                    or result.get("score_estimate_b") is not None
                )
                else None
            ),
            raw_result=result,
        )


@register_deriver("genrm")
class GenRMDeriver:
    """
    Preference deriver using NVIDIA GenRM model.

    Uses the special response_1/response_2 format for comparison
    with ranking scores (1-6) and helpfulness scores (1-5).
    """

    def __init__(self, judge: Any):
        """
        Initialize the GenRM deriver.

        Args:
            judge: GenRMJudge instance
        """
        self.judge = judge

    def derive(
        self,
        summary_a: str,
        summary_b: str,
        context: str,
        original_text: str,
        reference_score: Optional[float] = None,
        law_type: str = "sufficiency",
        **kwargs,
    ) -> PreferenceDerivationResult:
        """Derive preference using GenRM judge."""
        from treepo._research.training.preference.genrm import is_genrm_error

        result = self.judge.compare(
            context=context,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            law_type=law_type,
        )

        if is_genrm_error(result):
            return PreferenceDerivationResult(
                preferred="tie",
                confidence=0.0,
                reasoning=f"Error: {result.error_message}",
                raw_result=result,
            )

        # Map ranking score (1-6) to confidence (0-1)
        ranking_confidence = {
            1: 0.95, 2: 0.75, 3: 0.55,
            4: 0.55, 5: 0.75, 6: 0.95,
        }
        confidence = ranking_confidence.get(result.ranking_score, 0.5)

        return PreferenceDerivationResult(
            preferred=result.preferred,
            confidence=confidence,
            reasoning=result.reasoning,
            score_estimate_a=result.helpfulness_a,
            score_estimate_b=result.helpfulness_b,
            comparison_signal_value=float(result.ranking_score),
            comparison_signal_name="genrm_ranking_score",
            comparison_signal_min=1.0,
            comparison_signal_max=6.0,
            response_signal_name="genrm_helpfulness",
            response_signal_min=1.0,
            response_signal_max=5.0,
            raw_result=result,
        )


@register_deriver("oracle")
class OracleDeriver:
    """
    Preference deriver using oracle scoring function.

    Compares summaries by computing oracle scores for each and
    determining which has lower error relative to ground truth.
    """

    def __init__(
        self,
        oracle_predict: Callable[[str], float],
        tie_margin: float = 0.05,
        scale_range: Optional[float] = None,
    ):
        """
        Initialize the oracle deriver.

        Args:
            oracle_predict: Function that scores text
            tie_margin: Normalized error margin for ties (default 5%)
            scale_range: Range of the scale for normalization
        """
        self.oracle_predict = oracle_predict
        self.tie_margin = tie_margin
        self.scale_range = scale_range

    def derive(
        self,
        summary_a: str,
        summary_b: str,
        context: str,
        original_text: str,
        reference_score: Optional[float] = None,
        law_type: str = "sufficiency",
        **kwargs,
    ) -> PreferenceDerivationResult:
        """Derive preference using oracle scores."""
        # Get ground truth if not provided
        if reference_score is None:
            reference_score = self.oracle_predict(original_text)

        # Score both summaries
        score_a = self.oracle_predict(summary_a)
        score_b = self.oracle_predict(summary_b)

        # Compute errors
        error_a = abs(score_a - reference_score)
        error_b = abs(score_b - reference_score)

        # Normalize errors if scale_range provided
        if self.scale_range is not None and self.scale_range > 0:
            norm_error_a = error_a / self.scale_range
            norm_error_b = error_b / self.scale_range
        else:
            norm_error_a = error_a
            norm_error_b = error_b

        # Determine preference
        error_diff = norm_error_a - norm_error_b

        if abs(error_diff) <= self.tie_margin:
            preferred = "tie"
            confidence = 0.5
            reasoning = f"Tie: errors within margin. A={norm_error_a:.3f}, B={norm_error_b:.3f}"
        elif error_diff > 0:
            preferred = "B"
            confidence = min(0.95, 0.5 + abs(error_diff) * 2)
            reasoning = f"B has lower error ({norm_error_b:.3f} vs {norm_error_a:.3f})"
        else:
            preferred = "A"
            confidence = min(0.95, 0.5 + abs(error_diff) * 2)
            reasoning = f"A has lower error ({norm_error_a:.3f} vs {norm_error_b:.3f})"

        return PreferenceDerivationResult(
            preferred=preferred,
            confidence=confidence,
            reasoning=reasoning,
            score_estimate_a=score_a,
            score_estimate_b=score_b,
            response_signal_name="oracle_score_estimate",
            raw_result={
                "error_a": error_a,
                "error_b": error_b,
                "reference_score": reference_score,
            },
        )


@dataclass
class PreferencePair:
    """
    A single pairwise preference judgment.

    This is the optimizer-facing projection of a richer comparative judgment:
    it retains the winner/tie decision plus any raw comparative signal
    and any per-response score channel needed by downstream TRL-style
    objectives.
    """
    # Identifiers
    pair_id: str
    source_example_id: str

    # Input context
    original_text: str
    rubric: str
    reference_score: float

    # Candidate summaries
    summary_a: str
    summary_b: str

    # Judgment
    preferred: Literal["A", "B", "tie"]
    reasoning: str
    confidence: float

    # Fields with defaults (must come after required fields)
    law_type: str = "sufficiency"
    source_doc_id: Optional[str] = None
    three_layer_roles: Dict[str, str] = field(default_factory=dict)
    truth_label_source: str = "unknown"
    oracle_view: Optional[str] = None
    oracle_proxy_source: Optional[str] = None

    # Canonical TreePO/IPW metadata
    sampling: SamplingMetadata = field(
        default_factory=lambda: SamplingMetadata(unit_kind=ObservationUnitKind.PAIR)
    )
    preference_supervision: PreferenceSupervisionMetadata = field(
        default_factory=preference_supervision_metadata
    )
    source_observation_ids: List[str] = field(default_factory=list)

    # Optional audit alignment metadata (Phase 1.5 TreePO audit)
    audit_tree_id: Optional[str] = None
    audit_passed: Optional[bool] = None
    audit_violation_rate: Optional[float] = None
    audit_union_bound: Optional[float] = None
    audit_violation_ci_low: Optional[float] = None
    audit_violation_ci_high: Optional[float] = None

    # Score estimates from judge
    score_estimate_a: Optional[float] = None
    score_estimate_b: Optional[float] = None
    comparison_signal_value: Optional[float] = None
    oracle_error_a: Optional[float] = None
    oracle_error_b: Optional[float] = None

    # Metadata
    judge_model: str = ""
    timestamp: Optional[str] = None
    generation_config_a: Optional[Dict[str, Any]] = None
    generation_config_b: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

        self.truth_label_source = normalize_truth_label_source(self.truth_label_source)
        if self.source_doc_id is not None:
            self.source_doc_id = str(self.source_doc_id)
        if self.oracle_view is not None:
            self.oracle_view = str(self.oracle_view)
        if self.oracle_proxy_source is not None:
            self.oracle_proxy_source = str(self.oracle_proxy_source)
        if self.three_layer_roles is None:
            self.three_layer_roles = {}
        elif not isinstance(self.three_layer_roles, dict):
            self.three_layer_roles = dict(self.three_layer_roles)
        if not isinstance(self.sampling, SamplingMetadata):
            self.sampling = SamplingMetadata.from_dict(self.sampling)
        if self.sampling.unit_kind is None:
            self.sampling = self.sampling.with_updates(unit_kind=ObservationUnitKind.PAIR)
        if not isinstance(self.preference_supervision, PreferenceSupervisionMetadata):
            self.preference_supervision = PreferenceSupervisionMetadata.from_dict(
                self.preference_supervision
            )
        if self.preference_supervision.law_type is None and self.law_type:
            self.preference_supervision = self.preference_supervision.with_updates(
                law_type=self.law_type
            )
        elif not self.law_type and self.preference_supervision.law_type is not None:
            self.law_type = self.preference_supervision.law_type
        if self.comparison_signal_value is not None:
            self.comparison_signal_value = float(self.comparison_signal_value)
        self.source_observation_ids = [str(value) for value in self.source_observation_ids]

    def effective_joint_propensity(self, min_propensity: float = MIN_PROPENSITY) -> float:
        """Joint propensity with global-uniform fallback and numerical floor."""
        return self.sampling.effective_joint_propensity(min_propensity=min_propensity)

    def ipw_weight(
        self,
        min_propensity: float = MIN_PROPENSITY,
        max_weight: Optional[float] = None,
    ) -> float:
        """Inverse-propensity weight for this preference pair."""
        return self.sampling.ipw_weight(
            min_propensity=min_propensity,
            max_weight=max_weight,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pair_id": self.pair_id,
            "source_example_id": self.source_example_id,
            "original_text": self.original_text,
            "rubric": self.rubric,
            "reference_score": self.reference_score,
            "law_type": self.law_type,
            "source_doc_id": self.source_doc_id,
            "three_layer_roles": self.three_layer_roles,
            "truth_label_source": self.truth_label_source,
            "oracle_view": self.oracle_view,
            "oracle_proxy_source": self.oracle_proxy_source,
            "summary_a": self.summary_a,
            "summary_b": self.summary_b,
            "preferred": self.preferred,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "sampling": self.sampling.to_dict(),
            "preference_supervision": self.preference_supervision.to_dict(),
            "source_observation_ids": list(self.source_observation_ids),
            "audit_tree_id": self.audit_tree_id,
            "audit_passed": self.audit_passed,
            "audit_violation_rate": self.audit_violation_rate,
            "audit_union_bound": self.audit_union_bound,
            "audit_violation_ci_low": self.audit_violation_ci_low,
            "audit_violation_ci_high": self.audit_violation_ci_high,
            "score_estimate_a": self.score_estimate_a,
            "score_estimate_b": self.score_estimate_b,
            "comparison_signal_value": self.comparison_signal_value,
            "oracle_error_a": self.oracle_error_a,
            "oracle_error_b": self.oracle_error_b,
            "judge_model": self.judge_model,
            "timestamp": self.timestamp,
            "generation_config_a": self.generation_config_a,
            "generation_config_b": self.generation_config_b,
            "sample_weight": self.ipw_weight(),
        }

    def preference_supervision_metadata(self) -> Dict[str, Any]:
        metadata = {
            **self.preference_supervision.to_dict(),
            "source_doc_id": self.source_doc_id,
            "source_observation_ids": list(self.source_observation_ids),
            "truth_label_source": self.truth_label_source,
            "oracle_view": self.oracle_view,
            "oracle_proxy_source": self.oracle_proxy_source,
        }
        return {
            key: value
            for key, value in metadata.items()
            if value is not None and value != [] and value != {}
        }

    def comparative_signal_payload(self) -> Dict[str, Any]:
        payload = {
            "comparison_signal_name": self.preference_supervision.comparison_signal_name,
            "comparison_signal_value": self.comparison_signal_value,
            "comparison_signal_min": self.preference_supervision.comparison_signal_min,
            "comparison_signal_max": self.preference_supervision.comparison_signal_max,
            "response_signal_name": self.preference_supervision.response_signal_name,
            "response_signal_min": self.preference_supervision.response_signal_min,
            "response_signal_max": self.preference_supervision.response_signal_max,
            "response_signal_a": self.score_estimate_a,
            "response_signal_b": self.score_estimate_b,
        }
        return {
            key: value
            for key, value in payload.items()
            if value is not None and value != [] and value != {}
        }

    def optimization_metadata(
        self,
        *,
        response_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        rl_role: Optional[TreePORLRole] = None,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
    ) -> Dict[str, Any]:
        metadata = {
            "pair_id": self.pair_id,
            "reference_score": self.reference_score,
            "law_type": self.law_type,
            "preferred": self.preferred,
            "confidence": self.confidence,
            "preference_supervision": self.preference_supervision_metadata(),
            "comparative_signal": self.comparative_signal_payload(),
            "truth_label_source": self.truth_label_source,
            "oracle_view": self.oracle_view,
            "oracle_proxy_source": self.oracle_proxy_source,
            "source_doc_id": self.source_doc_id,
            "three_layer_roles": self.three_layer_roles,
            "treepo": self.treepo_metadata(
                rl_role=rl_role,
                tree_objective_weighting_mode=tree_objective_weighting_mode,
                discount_gamma=discount_gamma,
            ),
        }
        if response_id is not None:
            metadata["response_id"] = response_id
        if extra:
            metadata.update(extra)
        return {
            key: value
            for key, value in metadata.items()
            if value is not None and value != [] and value != {}
        }

    def treepo_metadata(
        self,
        *,
        rl_role: Optional[TreePORLRole] = None,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
    ) -> Dict[str, Any]:
        """Return compact TreePO metadata for downstream export formats."""
        optimizer_metadata = build_treepo_optimizer_export_metadata(
            fallback_node_id=self.pair_id,
            source_example_id=self.source_example_id,
            source_doc_id=self.source_doc_id,
            source_observation_ids=self.source_observation_ids,
            sampling=self.sampling,
            law_type=self.law_type,
            supervision_channel_name=self.preference_supervision.supervision_channel_name,
            supervision_signal_name=self.preference_supervision.supervision_signal_name,
            metadata_sources=(
                self.preference_supervision.metadata,
                self.sampling.metadata,
            ),
            weighting_mode=tree_objective_weighting_mode,
            discount_gamma=discount_gamma,
            rl_role=rl_role,
        ).to_dict()
        metadata = {
            **optimizer_metadata,
            "sampling": self.sampling.to_dict(),
            "sampling_scheme": self.sampling.sampling_scheme,
            "policy_name": self.sampling.policy_name,
            "unit_kind": (
                self.sampling.unit_kind.value
                if self.sampling.unit_kind is not None
                else None
            ),
            "supports_ipw_estimation": bool(self.sampling.supports_ipw_estimation),
            "audit_tree_id": self.audit_tree_id,
            "audit_passed": self.audit_passed,
            "audit_violation_rate": self.audit_violation_rate,
            "audit_union_bound": self.audit_union_bound,
            "audit_violation_ci_low": self.audit_violation_ci_low,
            "audit_violation_ci_high": self.audit_violation_ci_high,
            "truth_label_source": self.truth_label_source,
            "oracle_view": self.oracle_view,
            "oracle_proxy_source": self.oracle_proxy_source,
            "source_doc_id": self.source_doc_id,
            "source_observation_ids": list(self.source_observation_ids),
            "preference_supervision": self.preference_supervision_metadata(),
        }
        return {key: value for key, value in metadata.items() if value is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreferencePair':
        """Create from dictionary."""
        payload = dict(data)
        # Derived/runtime field; recomputed from propensities.
        payload.pop("sample_weight", None)
        supervision = payload.get("preference_supervision")
        if not isinstance(supervision, dict):
            supervision = {
                "law_type": payload.get("law_type"),
            }
        payload["preference_supervision"] = PreferenceSupervisionMetadata.from_dict(supervision)
        sampling = payload.get("sampling")
        if not isinstance(sampling, dict):
            sampling = {
                "document_propensity": payload.pop(
                    "document_propensity",
                    payload.pop("doc_propensity", DEFAULT_GLOBAL_PROPENSITY),
                ),
                "unit_propensity": payload.pop(
                    "unit_propensity",
                    payload.pop("node_propensity", DEFAULT_GLOBAL_PROPENSITY),
                ),
                "label_propensity": payload.pop(
                    "label_propensity",
                    payload.pop("label_propensity", DEFAULT_GLOBAL_PROPENSITY),
                ),
                "joint_propensity": payload.pop("joint_propensity", None),
                "sampling_scheme": payload.pop("sampling_scheme", None),
                "policy_name": payload.pop("policy_name", None),
                "unit_kind": payload.pop("unit_kind", payload.pop("node_type", "pair")),
                "supports_ipw_estimation": payload.pop("supports_ipw_estimation", True),
            }
        payload["sampling"] = SamplingMetadata.from_dict(sampling)
        return cls(**payload)

    def get_winner(self) -> Optional[str]:
        """Return the winning summary, or None for ties."""
        if self.preferred == "A":
            return self.summary_a
        elif self.preferred == "B":
            return self.summary_b
        return None

    def get_loser(self) -> Optional[str]:
        """Return the losing summary, or None for ties."""
        if self.preferred == "A":
            return self.summary_b
        elif self.preferred == "B":
            return self.summary_a
        return None

    def to_comparative_judgment(self) -> "ComparativeJudgmentRecord":
        """Project this binary preference into the general comparative format."""
        if self.preferred == "A":
            rank_a, rank_b = 1, 2
        elif self.preferred == "B":
            rank_a, rank_b = 2, 1
        else:
            rank_a = rank_b = 1

        candidates = [
            ComparativeCandidate(
                candidate_id=f"{self.pair_id}:A",
                response=self.summary_a,
                rank=rank_a,
                response_signal_value=self.score_estimate_a,
                source_pair_ids=[self.pair_id],
                metadata={
                    "response_id": "A",
                    "generation_config": self.generation_config_a,
                },
            ),
            ComparativeCandidate(
                candidate_id=f"{self.pair_id}:B",
                response=self.summary_b,
                rank=rank_b,
                response_signal_value=self.score_estimate_b,
                source_pair_ids=[self.pair_id],
                metadata={
                    "response_id": "B",
                    "generation_config": self.generation_config_b,
                },
            ),
        ]
        return ComparativeJudgmentRecord(
            record_id=self.pair_id,
            source_example_id=self.source_example_id,
            original_text=self.original_text,
            rubric=self.rubric,
            reference_score=self.reference_score,
            law_type=self.law_type,
            candidates=candidates,
            sampling=self.sampling,
            preference_supervision=self.preference_supervision,
            source_observation_ids=list(self.source_observation_ids),
            source_pair_ids=[self.pair_id],
            aggregate_sample_weight=self.ipw_weight(),
            source_doc_id=self.source_doc_id,
            three_layer_roles=dict(self.three_layer_roles),
            truth_label_source=self.truth_label_source,
            oracle_view=self.oracle_view,
            oracle_proxy_source=self.oracle_proxy_source,
            judge_model=self.judge_model,
            comparison_signal_value=self.comparison_signal_value,
            timestamp=self.timestamp,
            metadata={
                "reasoning": self.reasoning,
            },
        )


@dataclass
class ComparativeCandidate:
    """One candidate response inside a general comparative judgment record."""

    candidate_id: str
    response: str
    rank: Optional[int] = None
    response_signal_value: Optional[float] = None
    candidate_features: Optional[List[float]] = None
    source_pair_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.candidate_id = str(self.candidate_id)
        self.response = str(self.response)
        if self.rank is not None:
            self.rank = int(self.rank)
        if self.response_signal_value is not None:
            self.response_signal_value = float(self.response_signal_value)
        if self.candidate_features is None and isinstance(self.metadata, dict):
            payload = self.metadata.get("candidate_features")
            if payload is not None:
                self.candidate_features = [float(value) for value in payload]
        elif self.candidate_features is not None:
            self.candidate_features = [float(value) for value in self.candidate_features]
        self.source_pair_ids = [str(value) for value in self.source_pair_ids]
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "response": self.response,
            "rank": self.rank,
            "response_signal_value": self.response_signal_value,
            "candidate_features": list(self.candidate_features)
            if self.candidate_features is not None
            else None,
            "source_pair_ids": list(self.source_pair_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComparativeCandidate":
        return cls(**dict(data))


@dataclass
class ComparativeJudgmentRecord:
    """General N-candidate comparative supervision record."""

    record_id: str
    source_example_id: str
    original_text: str
    rubric: str
    reference_score: float
    law_type: str = "sufficiency"
    candidates: List[ComparativeCandidate] = field(default_factory=list)
    sampling: SamplingMetadata = field(
        default_factory=lambda: SamplingMetadata(unit_kind=ObservationUnitKind.PAIR)
    )
    preference_supervision: PreferenceSupervisionMetadata = field(
        default_factory=lambda: preference_supervision_metadata().with_updates(
            preference_family="groupwise"
        )
    )
    source_observation_ids: List[str] = field(default_factory=list)
    source_pair_ids: List[str] = field(default_factory=list)
    aggregate_sample_weight: float = 1.0
    source_doc_id: Optional[str] = None
    three_layer_roles: Dict[str, str] = field(default_factory=dict)
    truth_label_source: str = "unknown"
    oracle_view: Optional[str] = None
    oracle_proxy_source: Optional[str] = None
    judge_model: str = ""
    comparison_signal_value: Optional[float] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.record_id = str(self.record_id)
        self.source_example_id = str(self.source_example_id)
        self.original_text = str(self.original_text)
        self.rubric = str(self.rubric)
        self.reference_score = float(self.reference_score)
        if not isinstance(self.sampling, SamplingMetadata):
            self.sampling = SamplingMetadata.from_dict(self.sampling)
        if self.sampling.unit_kind is None:
            self.sampling = self.sampling.with_updates(unit_kind=ObservationUnitKind.PAIR)
        if not isinstance(self.preference_supervision, PreferenceSupervisionMetadata):
            self.preference_supervision = PreferenceSupervisionMetadata.from_dict(
                self.preference_supervision
            )
        if self.preference_supervision.law_type is None and self.law_type:
            self.preference_supervision = self.preference_supervision.with_updates(
                law_type=self.law_type
            )
        if self.preference_supervision.preference_family != "groupwise":
            self.preference_supervision = self.preference_supervision.with_updates(
                preference_family="groupwise"
            )
        self.candidates = [
            candidate
            if isinstance(candidate, ComparativeCandidate)
            else ComparativeCandidate.from_dict(candidate)
            for candidate in self.candidates
        ]
        self.source_observation_ids = [str(value) for value in self.source_observation_ids]
        self.source_pair_ids = [str(value) for value in self.source_pair_ids]
        self.truth_label_source = normalize_truth_label_source(self.truth_label_source)
        if self.aggregate_sample_weight is None:
            self.aggregate_sample_weight = 1.0
        self.aggregate_sample_weight = float(self.aggregate_sample_weight)
        if self.comparison_signal_value is not None:
            self.comparison_signal_value = float(self.comparison_signal_value)
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if not isinstance(self.three_layer_roles, dict):
            self.three_layer_roles = dict(self.three_layer_roles)
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)

    def comparative_signal_payload(self) -> Dict[str, Any]:
        payload = {
            "comparison_signal_name": self.preference_supervision.comparison_signal_name,
            "comparison_signal_value": self.comparison_signal_value,
            "comparison_signal_min": self.preference_supervision.comparison_signal_min,
            "comparison_signal_max": self.preference_supervision.comparison_signal_max,
            "response_signal_name": self.preference_supervision.response_signal_name,
            "response_signal_min": self.preference_supervision.response_signal_min,
            "response_signal_max": self.preference_supervision.response_signal_max,
        }
        return {
            key: value
            for key, value in payload.items()
            if value is not None and value != [] and value != {}
        }

    def preference_supervision_metadata(self) -> Dict[str, Any]:
        metadata = {
            **self.preference_supervision.to_dict(),
            "source_doc_id": self.source_doc_id,
            "source_observation_ids": list(self.source_observation_ids),
            "source_pair_ids": list(self.source_pair_ids),
            "truth_label_source": self.truth_label_source,
            "oracle_view": self.oracle_view,
            "oracle_proxy_source": self.oracle_proxy_source,
        }
        return {
            key: value
            for key, value in metadata.items()
            if value is not None and value != [] and value != {}
        }

    def treepo_metadata(
        self,
        *,
        rl_role: Optional[TreePORLRole] = None,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
        ipw_weight_override: Optional[float] = None,
        joint_propensity_override: Optional[float] = None,
        joint_propensity_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        optimizer_metadata = build_treepo_optimizer_export_metadata(
            fallback_node_id=self.record_id,
            source_example_id=self.source_example_id,
            source_doc_id=self.source_doc_id,
            source_observation_ids=self.source_observation_ids,
            sampling=self.sampling,
            law_type=self.law_type,
            supervision_channel_name=self.preference_supervision.supervision_channel_name,
            supervision_signal_name=self.preference_supervision.supervision_signal_name,
            metadata_sources=(
                self.preference_supervision.metadata,
                self.sampling.metadata,
                self.metadata,
            ),
            weighting_mode=tree_objective_weighting_mode,
            discount_gamma=discount_gamma,
            rl_role=rl_role,
            ipw_weight_override=ipw_weight_override,
            joint_propensity_override=joint_propensity_override,
            joint_propensity_source=joint_propensity_source,
        ).to_dict()
        metadata = {
            **optimizer_metadata,
            "sampling": self.sampling.to_dict(),
            "aggregate_sample_weight": self.aggregate_sample_weight,
            "source_observation_ids": list(self.source_observation_ids),
        }
        return {
            key: value
            for key, value in metadata.items()
            if value is not None and value != [] and value != {}
        }

    def optimization_metadata(
        self,
        *,
        rl_role: Optional[TreePORLRole] = None,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
        ipw_weight_override: Optional[float] = None,
        joint_propensity_override: Optional[float] = None,
        joint_propensity_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        metadata = {
            "record_id": self.record_id,
            "reference_score": self.reference_score,
            "law_type": self.law_type,
            "preference_supervision": self.preference_supervision_metadata(),
            "comparative_signal": self.comparative_signal_payload(),
            "truth_label_source": self.truth_label_source,
            "oracle_view": self.oracle_view,
            "oracle_proxy_source": self.oracle_proxy_source,
            "source_doc_id": self.source_doc_id,
            "three_layer_roles": self.three_layer_roles,
            "source_pair_ids": list(self.source_pair_ids),
            "num_candidates": len(self.candidates),
            "treepo": self.treepo_metadata(
                rl_role=rl_role,
                tree_objective_weighting_mode=tree_objective_weighting_mode,
                discount_gamma=discount_gamma,
                ipw_weight_override=ipw_weight_override,
                joint_propensity_override=joint_propensity_override,
                joint_propensity_source=joint_propensity_source,
            ),
        }
        if self.metadata:
            metadata["record_metadata"] = dict(self.metadata)
        return {
            key: value
            for key, value in metadata.items()
            if value is not None and value != [] and value != {}
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_example_id": self.source_example_id,
            "original_text": self.original_text,
            "rubric": self.rubric,
            "reference_score": self.reference_score,
            "law_type": self.law_type,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "sampling": self.sampling.to_dict(),
            "preference_supervision": self.preference_supervision.to_dict(),
            "source_observation_ids": list(self.source_observation_ids),
            "source_pair_ids": list(self.source_pair_ids),
            "aggregate_sample_weight": self.aggregate_sample_weight,
            "source_doc_id": self.source_doc_id,
            "three_layer_roles": dict(self.three_layer_roles),
            "truth_label_source": self.truth_label_source,
            "oracle_view": self.oracle_view,
            "oracle_proxy_source": self.oracle_proxy_source,
            "judge_model": self.judge_model,
            "comparison_signal_value": self.comparison_signal_value,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComparativeJudgmentRecord":
        payload = dict(data)
        payload["candidates"] = [
            ComparativeCandidate.from_dict(candidate)
            for candidate in list(payload.get("candidates", []) or [])
        ]
        return cls(**payload)

    def to_preference_pairs(
        self,
        *,
        projection: Literal["winner_vs_runner_up", "adjacent"] = "winner_vs_runner_up",
    ) -> List[PreferencePair]:
        """Project this groupwise judgment into one or more optimizer-facing binary pairs."""
        ordered_candidates = sorted(
            self.candidates,
            key=lambda candidate: (
                candidate.rank if candidate.rank is not None else 10**6,
                -(
                    float(candidate.response_signal_value)
                    if candidate.response_signal_value is not None
                    else float("-inf")
                ),
                candidate.candidate_id,
            ),
        )
        if len(ordered_candidates) < 2:
            return []

        if projection == "winner_vs_runner_up":
            pairings = [(ordered_candidates[0], ordered_candidates[1], 0)]
        elif projection == "adjacent":
            pairings = [
                (ordered_candidates[index], ordered_candidates[index + 1], index)
                for index in range(len(ordered_candidates) - 1)
            ]
        else:
            raise ValueError(
                f"Unknown comparative projection: {projection!r}. "
                "Expected 'winner_vs_runner_up' or 'adjacent'."
            )

        raw_confidence = self.metadata.get("confidence", 0.75)
        try:
            base_confidence = max(0.5, min(1.0, float(raw_confidence)))
        except (TypeError, ValueError):
            base_confidence = 0.75

        response_signal_range: Optional[float] = None
        if (
            self.preference_supervision.response_signal_min is not None
            and self.preference_supervision.response_signal_max is not None
        ):
            try:
                response_signal_range = (
                    float(self.preference_supervision.response_signal_max)
                    - float(self.preference_supervision.response_signal_min)
                )
            except (TypeError, ValueError):
                response_signal_range = None
        reasoning = str(self.metadata.get("reasoning", "") or "")
        projected_supervision = self.preference_supervision.with_updates(
            preference_family="pairwise",
            metadata={
                **dict(self.preference_supervision.metadata or {}),
                "projection": projection,
                "source_record_id": self.record_id,
            },
        )

        projected_pairs: List[PreferencePair] = []
        for preferred_candidate, other_candidate, projection_index in pairings:
            score_a = preferred_candidate.response_signal_value
            score_b = other_candidate.response_signal_value
            oracle_score_a = preferred_candidate.metadata.get("oracle_score")
            oracle_score_b = other_candidate.metadata.get("oracle_score")
            oracle_error_a = preferred_candidate.metadata.get("oracle_error")
            oracle_error_b = other_candidate.metadata.get("oracle_error")
            projected_preference = "A"
            confidence = base_confidence
            if (
                preferred_candidate.rank is not None
                and other_candidate.rank is not None
                and int(preferred_candidate.rank) == int(other_candidate.rank)
            ):
                projected_preference = "tie"
                confidence = 0.5
            elif score_a is not None and score_b is not None:
                score_gap = abs(float(score_a) - float(score_b))
                if response_signal_range is not None and response_signal_range > 0:
                    derived_confidence = 0.5 + min(0.5, score_gap / response_signal_range)
                else:
                    derived_confidence = 1.0 if score_gap > 1e-9 else 0.5
                confidence = max(confidence, min(1.0, max(0.5, float(derived_confidence))))

            comparison_signal_name = self.preference_supervision.comparison_signal_name
            comparison_signal_value = self.comparison_signal_value
            if score_a is not None and score_b is not None:
                comparison_signal_name = comparison_signal_name or "projected_response_signal_margin"
                comparison_signal_value = float(score_a) - float(score_b)

            projected_pairs.append(
                PreferencePair(
                    pair_id=f"{self.record_id}:proj:{projection}:{projection_index:03d}",
                    source_example_id=self.source_example_id,
                    original_text=self.original_text,
                    rubric=self.rubric,
                    reference_score=self.reference_score,
                    summary_a=preferred_candidate.response,
                    summary_b=other_candidate.response,
                    preferred=projected_preference,
                    reasoning=reasoning,
                    confidence=confidence,
                    law_type=self.law_type,
                    source_doc_id=self.source_doc_id,
                    three_layer_roles=dict(self.three_layer_roles),
                    truth_label_source=self.truth_label_source,
                    oracle_view=self.oracle_view,
                    oracle_proxy_source=self.oracle_proxy_source,
                    sampling=self.sampling,
                    preference_supervision=projected_supervision.with_updates(
                        comparison_signal_name=comparison_signal_name,
                    ),
                    source_observation_ids=list(self.source_observation_ids),
                    score_estimate_a=(
                        float(oracle_score_a)
                        if oracle_score_a is not None
                        else score_a
                    ),
                    score_estimate_b=(
                        float(oracle_score_b)
                        if oracle_score_b is not None
                        else score_b
                    ),
                    oracle_error_a=(
                        float(oracle_error_a)
                        if oracle_error_a is not None
                        else None
                    ),
                    oracle_error_b=(
                        float(oracle_error_b)
                        if oracle_error_b is not None
                        else None
                    ),
                    comparison_signal_value=comparison_signal_value,
                    judge_model=self.judge_model,
                    timestamp=self.timestamp,
                    generation_config_a=preferred_candidate.metadata.get("generation_config"),
                    generation_config_b=other_candidate.metadata.get("generation_config"),
                )
            )

        return projected_pairs

    def to_grouped_grpo_record(
        self,
        *,
        prompt_builder: Optional[PromptBuilder] = None,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
    ) -> Dict[str, Any]:
        ordered_candidates = sorted(
            self.candidates,
            key=lambda candidate: (
                candidate.rank if candidate.rank is not None else 10**6,
                -(
                    float(candidate.response_signal_value)
                    if candidate.response_signal_value is not None
                    else float("-inf")
                ),
                candidate.candidate_id,
            ),
        )
        responses = [candidate.response for candidate in ordered_candidates]
        ranks = [
            int(candidate.rank) if candidate.rank is not None else idx + 1
            for idx, candidate in enumerate(ordered_candidates)
        ]
        scores = [
            (
                float(candidate.response_signal_value)
                if candidate.response_signal_value is not None
                else None
            )
            for candidate in ordered_candidates
        ]
        treepo_metadata = self.treepo_metadata(
            rl_role="grpo_prompt",
            tree_objective_weighting_mode=tree_objective_weighting_mode,
            discount_gamma=discount_gamma,
            ipw_weight_override=self.aggregate_sample_weight,
            joint_propensity_override=(
                1.0 / float(self.aggregate_sample_weight)
                if float(self.aggregate_sample_weight) > 0.0
                else 1.0
            ),
            joint_propensity_source="aggregate_sample_weight",
        )
        return {
            "prompt": render_prompt(self.original_text, self.rubric, prompt_builder),
            "responses": responses,
            "ranks": ranks,
            "scores": scores,
            "k": len(responses),
            "sample_weight": float(treepo_metadata["effective_weight"]),
            "reference_score": self.reference_score,
            "original_text": self.original_text,
            "rubric": self.rubric,
            "law_type": self.law_type,
            "preference_supervision": self.preference_supervision_metadata(),
            "comparative_signal": self.comparative_signal_payload(),
            "metadata": self.optimization_metadata(
                rl_role="grpo_prompt",
                tree_objective_weighting_mode=tree_objective_weighting_mode,
                discount_gamma=discount_gamma,
                ipw_weight_override=self.aggregate_sample_weight,
                joint_propensity_override=(
                    1.0 / float(self.aggregate_sample_weight)
                    if float(self.aggregate_sample_weight) > 0.0
                    else 1.0
                ),
                joint_propensity_source="aggregate_sample_weight",
            ),
        }


class ComparativeDataset:
    """Dataset of general comparative judgments."""

    def __init__(self, records: Optional[List[ComparativeJudgmentRecord]] = None):
        self.records = records or []

    def __len__(self) -> int:
        return len(self.records)

    def add_record(self, record: ComparativeJudgmentRecord) -> None:
        self.records.append(record)

    def to_grouped_grpo_format(
        self,
        *,
        prompt_builder: Optional[PromptBuilder] = None,
        min_group_size: int = 2,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
    ) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for record in self.records:
            if len(record.candidates) < min_group_size:
                continue
            output.append(
                record.to_grouped_grpo_record(
                    prompt_builder=prompt_builder,
                    tree_objective_weighting_mode=tree_objective_weighting_mode,
                    discount_gamma=discount_gamma,
                )
            )
        return output

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "num_records": len(self.records),
            "records": [record.to_dict() for record in self.records],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved %d comparative records to %s", len(self.records), path)

    @classmethod
    def load(cls, path: Path) -> "ComparativeDataset":
        with open(path) as f:
            payload = json.load(f)
        records = [
            ComparativeJudgmentRecord.from_dict(record)
            for record in list(payload.get("records", []) or [])
        ]
        logger.info("Loaded %d comparative records from %s", len(records), path)
        return cls(records)


@dataclass
class GenerationConfig:
    """Configuration for generating candidate summaries."""
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 8192
    prompt_variant: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "prompt_variant": self.prompt_variant,
        }


class PreferenceDataset:
    """
    Dataset of preference pairs for training.

    Supports saving/loading, filtering, and conversion to training formats.
    """

    def __init__(
        self,
        pairs: Optional[List[PreferencePair]] = None,
        comparative_records: Optional[List[ComparativeJudgmentRecord]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            pairs: Initial list of preference pairs
        """
        self.pairs = pairs or []
        self.comparative_records = comparative_records or []

    def add_pair(self, pair: PreferencePair):
        """Add a preference pair to the dataset."""
        self.pairs.append(pair)

    def add_pairs(self, pairs: List[PreferencePair]):
        """Add multiple preference pairs."""
        self.pairs.extend(pairs)

    def add_comparative_record(self, record: ComparativeJudgmentRecord) -> None:
        """Add one direct comparative judgment record."""
        self.comparative_records.append(record)

    def add_comparative_records(self, records: List[ComparativeJudgmentRecord]) -> None:
        """Add multiple direct comparative judgment records."""
        self.comparative_records.extend(records)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PreferencePair:
        return self.pairs[idx]

    @staticmethod
    def _comparative_group_key(pair: PreferencePair) -> Tuple[Any, ...]:
        return (
            pair.source_example_id,
            pair.law_type,
            pair.original_text,
            pair.rubric,
            pair.reference_score,
            pair.source_doc_id,
        )

    @staticmethod
    def _comparative_record_group_key(record: ComparativeJudgmentRecord) -> Tuple[Any, ...]:
        return (
            record.source_example_id,
            record.law_type,
            record.original_text,
            record.rubric,
            record.reference_score,
            record.source_doc_id,
        )

    def to_comparative_dataset(
        self,
        law_type: Optional[str] = None,
    ) -> ComparativeDataset:
        """Aggregate binary preferences into general N-candidate comparative records."""
        direct_records = [
            record
            for record in self.comparative_records
            if law_type is None or record.law_type == law_type
        ]
        comparative_records: List[ComparativeJudgmentRecord] = list(direct_records)
        direct_group_keys = {
            self._comparative_record_group_key(record)
            for record in direct_records
        }
        grouped_pairs: Dict[Tuple[Any, ...], List[PreferencePair]] = defaultdict(list)
        for pair in self.pairs:
            if law_type is not None and pair.law_type != law_type:
                continue
            grouped_pairs[self._comparative_group_key(pair)].append(pair)

        for group_key, group_pairs in grouped_pairs.items():
            if not group_pairs:
                continue
            if group_key in direct_group_keys:
                continue
            if len(group_pairs) == 1:
                comparative_records.append(group_pairs[0].to_comparative_judgment())
                continue

            first = group_pairs[0]
            candidate_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
            ordered_keys: List[Tuple[str, str]] = []
            source_pair_ids: List[str] = []
            source_observation_ids: List[str] = []
            aggregate_sample_weight = 0.0
            comparison_signal_values: List[float] = []

            def ensure_candidate(
                *,
                response: str,
                response_id: str,
                generation_config: Optional[Dict[str, Any]],
            ) -> Dict[str, Any]:
                generation_key = json.dumps(generation_config, sort_keys=True, default=str)
                key = (str(response), generation_key)
                if key not in candidate_stats:
                    candidate_stats[key] = {
                        "response": str(response),
                        "response_id": str(response_id),
                        "generation_config": generation_config,
                        "wins": 0.0,
                        "response_signal_values": [],
                        "source_pair_ids": [],
                    }
                    ordered_keys.append(key)
                return candidate_stats[key]

            for pair in group_pairs:
                source_pair_ids.append(pair.pair_id)
                source_observation_ids.extend(pair.source_observation_ids)
                aggregate_sample_weight += pair.ipw_weight()
                if pair.comparison_signal_value is not None:
                    comparison_signal_values.append(float(pair.comparison_signal_value))

                cand_a = ensure_candidate(
                    response=pair.summary_a,
                    response_id="A",
                    generation_config=pair.generation_config_a,
                )
                cand_b = ensure_candidate(
                    response=pair.summary_b,
                    response_id="B",
                    generation_config=pair.generation_config_b,
                )
                cand_a["source_pair_ids"].append(pair.pair_id)
                cand_b["source_pair_ids"].append(pair.pair_id)

                if pair.score_estimate_a is not None:
                    cand_a["response_signal_values"].append(float(pair.score_estimate_a))
                if pair.score_estimate_b is not None:
                    cand_b["response_signal_values"].append(float(pair.score_estimate_b))

                if pair.preferred == "A":
                    cand_a["wins"] += 0.5 + 0.5 * float(pair.confidence)
                    cand_b["wins"] += 0.5 - 0.5 * float(pair.confidence)
                elif pair.preferred == "B":
                    cand_a["wins"] += 0.5 - 0.5 * float(pair.confidence)
                    cand_b["wins"] += 0.5 + 0.5 * float(pair.confidence)
                else:
                    cand_a["wins"] += 0.5
                    cand_b["wins"] += 0.5

            scored_candidates: List[Dict[str, Any]] = []
            for index, key in enumerate(ordered_keys, start=1):
                stats = candidate_stats[key]
                response_signal_values = list(stats["response_signal_values"])
                if response_signal_values:
                    aggregate_score = sum(response_signal_values) / len(response_signal_values)
                    aggregate_method = "mean_response_signal"
                else:
                    aggregate_score = float(stats["wins"])
                    aggregate_method = "pairwise_win_score"
                scored_candidates.append({
                    "candidate_id": f"{first.source_example_id}:cand_{index:03d}",
                    "response": stats["response"],
                    "aggregate_score": aggregate_score,
                    "source_pair_ids": list(dict.fromkeys(stats["source_pair_ids"])),
                    "metadata": {
                        "response_id": stats["response_id"],
                        "generation_config": stats["generation_config"],
                        "aggregate_method": aggregate_method,
                        "num_pairwise_observations": len(stats["source_pair_ids"]),
                    },
                })

            scored_candidates.sort(
                key=lambda item: (-float(item["aggregate_score"]), item["candidate_id"])
            )
            prev_score: Optional[float] = None
            current_rank = 0
            candidates: List[ComparativeCandidate] = []
            for idx, stats in enumerate(scored_candidates, start=1):
                score = float(stats["aggregate_score"])
                if prev_score is None or not math.isclose(
                    score,
                    prev_score,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    current_rank = idx
                    prev_score = score
                candidates.append(
                    ComparativeCandidate(
                        candidate_id=str(stats["candidate_id"]),
                        response=str(stats["response"]),
                        rank=current_rank,
                        response_signal_value=score,
                        source_pair_ids=list(stats["source_pair_ids"]),
                        metadata=dict(stats["metadata"]),
                    )
                )

            supervision = first.preference_supervision.with_updates(
                preference_family="groupwise"
            )
            comparative_records.append(
                ComparativeJudgmentRecord(
                    record_id=f"{first.source_example_id}:{first.law_type}:comparative",
                    source_example_id=first.source_example_id,
                    original_text=first.original_text,
                    rubric=first.rubric,
                    reference_score=first.reference_score,
                    law_type=first.law_type,
                    candidates=candidates,
                    sampling=first.sampling,
                    preference_supervision=supervision,
                    source_observation_ids=list(dict.fromkeys(source_observation_ids)),
                    source_pair_ids=list(dict.fromkeys(source_pair_ids)),
                    aggregate_sample_weight=aggregate_sample_weight,
                    source_doc_id=first.source_doc_id,
                    three_layer_roles=dict(first.three_layer_roles),
                    truth_label_source=first.truth_label_source,
                    oracle_view=first.oracle_view,
                    oracle_proxy_source=first.oracle_proxy_source,
                    judge_model=first.judge_model,
                    comparison_signal_value=(
                        sum(comparison_signal_values) / len(comparison_signal_values)
                        if comparison_signal_values
                        else None
                    ),
                    timestamp=first.timestamp,
                    metadata={
                        "aggregation_method": (
                            "mean_response_signal"
                            if any(
                                candidate.response_signal_value is not None
                                for candidate in candidates
                            )
                            else "pairwise_win_score"
                        ),
                        "num_source_pairs": len(group_pairs),
                    },
                )
            )

        return ComparativeDataset(comparative_records)

    def project_comparative_to_pairs(
        self,
        *,
        projection: Literal["winner_vs_runner_up", "adjacent"] = "adjacent",
        keep_existing: bool = True,
    ) -> "PreferenceDataset":
        """Return a pairwise projection of any direct comparative records."""
        projected_pairs: List[PreferencePair] = list(self.pairs) if keep_existing else []
        seen_pair_ids = {pair.pair_id for pair in projected_pairs}
        for record in self.comparative_records:
            for pair in record.to_preference_pairs(projection=projection):
                if pair.pair_id in seen_pair_ids:
                    continue
                projected_pairs.append(pair)
                seen_pair_ids.add(pair.pair_id)
        return PreferenceDataset(projected_pairs)

    def to_supervision_dataset(self) -> "SupervisionDataset":
        """Promote this legacy pairwise dataset to the primary supervision surface."""
        from treepo._research.training.supervision.types import SupervisionDataset

        comparative_judgments = [pair.to_comparative_judgment() for pair in self.pairs]
        comparative_judgments.extend(self.comparative_records)
        return SupervisionDataset(comparative_judgments=comparative_judgments)

    def get_sample_weights(
        self,
        min_propensity: float = MIN_PROPENSITY,
        max_weight: Optional[float] = None,
    ) -> List[float]:
        """Return IPW sample weights for all pairs."""
        return [
            pair.ipw_weight(min_propensity=min_propensity, max_weight=max_weight)
            for pair in self.pairs
        ]

    def propensity_diagnostics(
        self,
        *,
        include_ties: bool = True,
        min_propensity: float = MIN_PROPENSITY,
        max_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Dataset-level wrapper around `compute_propensity_diagnostics`."""
        return compute_propensity_diagnostics(
            self.pairs,
            include_ties=include_ties,
            min_propensity=min_propensity,
            max_weight=max_weight,
        )

    def resample_by_propensity(
        self,
        target_size: Optional[int] = None,
        seed: int = 42,
        max_weight: Optional[float] = None,
        min_propensity: float = MIN_PROPENSITY,
        strategy: Literal["multinomial", "pps_systematic", "stratified_multinomial"] = "pps_systematic",
        stratify_by: Optional[str] = None,
    ) -> 'PreferenceDataset':
        """
        Backward-compatible wrapper for propensity-based sampling.

        Historically this performed multinomial resampling with replacement.
        New strategies are available for efficiency and variance control.
        """
        return self.sample_by_propensity(
            target_size=target_size,
            seed=seed,
            max_weight=max_weight,
            min_propensity=min_propensity,
            strategy=strategy,
            stratify_by=stratify_by,
        )

    def sample_by_propensity(
        self,
        target_size: Optional[int] = None,
        seed: int = 42,
        max_weight: Optional[float] = None,
        min_propensity: float = MIN_PROPENSITY,
        strategy: Literal["multinomial", "pps_systematic", "stratified_multinomial"] = "pps_systematic",
        stratify_by: Optional[str] = None,
    ) -> 'PreferenceDataset':
        """
        Sample pairs according to propensity-derived weights.

        Strategies:
        - `multinomial`: with-replacement weighted sampling.
        - `pps_systematic`: fixed-size PPS without replacement when possible.
        - `stratified_multinomial`: weighted multinomial sampling within strata.
        """
        if not self.pairs:
            return PreferenceDataset([])

        size = int(target_size or len(self.pairs))
        if size <= 0:
            return PreferenceDataset([])

        rng = random.Random(seed)
        weights = self.get_sample_weights(
            min_propensity=min_propensity,
            max_weight=max_weight,
        )
        total_weight = sum(weights)
        if total_weight <= 0:
            return PreferenceDataset(self.pairs.copy())

        if strategy == "multinomial":
            sampled_pairs = rng.choices(self.pairs, weights=weights, k=size)
            return PreferenceDataset(sampled_pairs)

        if strategy == "pps_systematic":
            n = len(self.pairs)
            if size >= n:
                full = self.pairs.copy()
                extra = rng.choices(self.pairs, weights=weights, k=size - n)
                return PreferenceDataset(full + extra)

            inclusion_probs = _pps_inclusion_probabilities(weights, size)
            sampled_indices = _systematic_pps_sample_indices(inclusion_probs, size, rng)
            sampled_pairs = [self.pairs[index] for index in sampled_indices]
            return PreferenceDataset(sampled_pairs)

        if strategy == "stratified_multinomial":
            strata_key = stratify_by or "law_type"
            grouped_indices: Dict[str, List[int]] = defaultdict(list)
            for index, pair in enumerate(self.pairs):
                value = getattr(pair, strata_key, None)
                grouped_indices[str(value)].append(index)

            keys = list(grouped_indices.keys())
            if not keys:
                sampled_pairs = rng.choices(self.pairs, weights=weights, k=size)
                return PreferenceDataset(sampled_pairs)

            group_weight_sums = [
                sum(weights[index] for index in grouped_indices[key])
                for key in keys
            ]
            total_group_weight = sum(group_weight_sums)
            if total_group_weight <= 0:
                sampled_pairs = rng.choices(self.pairs, k=size)
                return PreferenceDataset(sampled_pairs)

            quotas = [
                size * (group_weight / total_group_weight)
                for group_weight in group_weight_sums
            ]
            allocation = _largest_remainder_allocation(size, quotas)
            sampled_pairs: List[PreferencePair] = []
            for key, group_size in zip(keys, allocation):
                if group_size <= 0:
                    continue
                group_indices = grouped_indices[key]
                group_pairs = [self.pairs[index] for index in group_indices]
                group_weights = [weights[index] for index in group_indices]
                sampled_pairs.extend(rng.choices(group_pairs, weights=group_weights, k=group_size))
            return PreferenceDataset(sampled_pairs)

        raise ValueError(
            f"Unknown sampling strategy: {strategy!r}. "
            "Expected one of {'multinomial', 'pps_systematic', 'stratified_multinomial'}."
        )

    def filter_by_confidence(self, min_confidence: float) -> 'PreferenceDataset':
        """Return new dataset with pairs above confidence threshold."""
        filtered = [p for p in self.pairs if p.confidence >= min_confidence]
        return PreferenceDataset(filtered)

    def filter_non_ties(self) -> 'PreferenceDataset':
        """Return new dataset excluding ties."""
        filtered = [p for p in self.pairs if p.preferred != "tie"]
        return PreferenceDataset(filtered)

    def split(
        self,
        train_ratio: float = 0.8,
        shuffle: bool = True,
    ) -> Tuple['PreferenceDataset', 'PreferenceDataset']:
        """
        Split into train and validation sets.

        Args:
            train_ratio: Fraction for training set
            shuffle: Whether to shuffle before splitting

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        pairs = self.pairs.copy()
        if shuffle:
            random.shuffle(pairs)

        split_idx = int(len(pairs) * train_ratio)
        return (
            PreferenceDataset(pairs[:split_idx]),
            PreferenceDataset(pairs[split_idx:]),
        )

    def to_dspy_examples(self) -> List[dspy.Example]:
        """
        Convert to DSPy examples for training.

        Returns:
            List of DSPy examples with inputs and preferred output
        """
        examples = []
        for pair in self.pairs:
            if pair.preferred == "tie":
                continue

            example = dspy.Example(
                law_type=pair.law_type,
                preference_supervision=pair.preference_supervision_metadata(),
                comparative_signal=pair.comparative_signal_payload(),
                rubric=pair.rubric,
                original_text=pair.original_text,
                summary_a=pair.summary_a,
                summary_b=pair.summary_b,
                reference_score=pair.reference_score,
                preferred=pair.preferred,
                reasoning=pair.reasoning,
                confidence=pair.confidence,
                sample_weight=pair.ipw_weight(),
                joint_propensity=pair.effective_joint_propensity(),
            ).with_inputs(
                "law_type", "rubric", "original_text", "summary_a", "summary_b", "reference_score"
            )
            examples.append(example)

        return examples

    def to_preference_format(
        self,
        method: Literal["dpo", "grpo", "rlhf", "general"] = "general",
        law_type: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Convert to preference learning format for various methods.

        This is the unified export method for preference learning data.
        The format depends on the downstream training method.

        Theoretical Foundation
        ----------------------
        From PreferenceLearning.lean: Any oracle-measurable preference learning
        method achieves equivalent results on oracle-preserving summaries.
        DPO, GRPO, PPO, etc. all satisfy this property when properly configured.

        Args:
            method: Target training method format
                - "dpo": Returns prompt/chosen/rejected for DPO-style training
                - "grpo": Returns group ranking format (placeholder for future)
                - "rlhf": Returns prompt/response/score for reward model training
                - "general": Returns full context with all fields
            law_type: Filter by OPS law type (sufficiency, merge, idempotence)
            prompt_builder: Optional prompt builder for generating prompts

        Returns:
            List of preference examples in the requested format
        """
        if method == "dpo":
            return self._to_dpo_format(
                law_type,
                prompt_builder,
                tree_objective_weighting_mode=tree_objective_weighting_mode,
                discount_gamma=discount_gamma,
            )
        elif method == "grpo":
            return self._to_grpo_format(
                law_type,
                prompt_builder,
                tree_objective_weighting_mode=tree_objective_weighting_mode,
                discount_gamma=discount_gamma,
            )
        elif method == "rlhf":
            return self._to_rlhf_format(
                law_type,
                prompt_builder,
                tree_objective_weighting_mode=tree_objective_weighting_mode,
                discount_gamma=discount_gamma,
            )
        else:
            return self._to_general_format(
                law_type,
                tree_objective_weighting_mode=tree_objective_weighting_mode,
                discount_gamma=discount_gamma,
            )

    def _to_dpo_format(
        self,
        law_type: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        *,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Convert to DPO (Direct Preference Optimization) format.

        DPO format uses pairwise preferences with prompt/chosen/rejected structure.
        This is the concrete instantiation of the abstract preference learning
        framework from PreferenceLearning.lean.

        Returns:
            List of dicts with prompt, chosen, rejected
        """
        dpo_data = []
        for pair in self.pairs:
            if pair.preferred == "tie":
                continue
            if law_type is not None and pair.law_type != law_type:
                continue

            prompt = render_prompt(pair.original_text, pair.rubric, prompt_builder)

            if pair.preferred == "A":
                chosen = pair.summary_a
                rejected = pair.summary_b
            else:
                chosen = pair.summary_b
                rejected = pair.summary_a

            metadata = pair.optimization_metadata(
                rl_role="dpo_pair",
                tree_objective_weighting_mode=tree_objective_weighting_mode,
                discount_gamma=discount_gamma,
            )

            dpo_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "sample_weight": float(metadata["treepo"]["effective_weight"]),
                "metadata": metadata,
            })

        return dpo_data

    def _to_grpo_format(
        self,
        law_type: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        *,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Convert to GRPO (Group Relative Policy Optimization) format.

        GRPO uses group-wise comparisons rather than strict pairwise preferences.
        This format is compatible with DeepSeek-style preference learning.

        Note: GRPO typically works with groups of K responses. For pairwise data,
        we structure it as a 2-element group with relative ranking.

        Returns:
            List of dicts with prompt, responses (ranked list), and ranking info
        """
        grpo_data = []
        for pair in self.pairs:
            if law_type is not None and pair.law_type != law_type:
                continue

            prompt = render_prompt(pair.original_text, pair.rubric, prompt_builder)

            # For GRPO, we provide ranked responses rather than chosen/rejected
            if pair.preferred == "A":
                ranked_responses = [pair.summary_a, pair.summary_b]
                ranks = [1, 2]
            elif pair.preferred == "B":
                ranked_responses = [pair.summary_b, pair.summary_a]
                ranks = [1, 2]
            else:  # tie
                ranked_responses = [pair.summary_a, pair.summary_b]
                ranks = [1, 1]  # Equal rank for ties

            metadata = pair.optimization_metadata(
                extra={"original_preferred": pair.preferred},
                rl_role="grpo_prompt",
                tree_objective_weighting_mode=tree_objective_weighting_mode,
                discount_gamma=discount_gamma,
            )
            grpo_data.append({
                "prompt": prompt,
                "responses": ranked_responses,
                "ranks": ranks,
                "confidence": pair.confidence,
                "sample_weight": float(metadata["treepo"]["effective_weight"]),
                "metadata": metadata,
            })

        return grpo_data

    def _to_rlhf_format(
        self,
        law_type: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        *,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Convert to RLHF (Reinforcement Learning from Human Feedback) format.

        RLHF format provides responses with scalar scores for reward model training.
        Confidence is converted to a relative score differential.

        Returns:
            List of dicts with prompt, response, score
        """
        rlhf_data = []
        for pair in self.pairs:
            if law_type is not None and pair.law_type != law_type:
                continue

            prompt = render_prompt(pair.original_text, pair.rubric, prompt_builder)

            # Generate score based on preference and confidence
            if pair.preferred == "A":
                score_a = 0.5 + pair.confidence * 0.5
                score_b = 0.5 - pair.confidence * 0.5
            elif pair.preferred == "B":
                score_a = 0.5 - pair.confidence * 0.5
                score_b = 0.5 + pair.confidence * 0.5
            else:  # tie
                score_a = 0.5
                score_b = 0.5

            rlhf_data.extend([
                {
                    "prompt": prompt,
                    "response": pair.summary_a,
                    "score": score_a,
                    "metadata": pair.optimization_metadata(
                        response_id="A",
                        rl_role="scalar_reward",
                        tree_objective_weighting_mode=tree_objective_weighting_mode,
                        discount_gamma=discount_gamma,
                    ),
                },
                {
                    "prompt": prompt,
                    "response": pair.summary_b,
                    "score": score_b,
                    "metadata": pair.optimization_metadata(
                        response_id="B",
                        rl_role="scalar_reward",
                        tree_objective_weighting_mode=tree_objective_weighting_mode,
                        discount_gamma=discount_gamma,
                    ),
                },
            ])
            for entry in rlhf_data[-2:]:
                entry["sample_weight"] = float(entry["metadata"]["treepo"]["effective_weight"])

        return rlhf_data

    def _to_general_format(
        self,
        law_type: Optional[str] = None,
        *,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Convert to general preference format with all fields.

        This format preserves all information and can be adapted to any
        preference learning method downstream.

        Returns:
            List of dicts with full preference pair information
        """
        general_data = []
        for pair in self.pairs:
            if law_type is not None and pair.law_type != law_type:
                continue

            general_data.append({
                "pair_id": pair.pair_id,
                "rubric": pair.rubric,
                "original_text": pair.original_text,
                "summary_a": pair.summary_a,
                "summary_b": pair.summary_b,
                "preferred": pair.preferred,
                "confidence": pair.confidence,
                "reasoning": pair.reasoning,
                "reference_score": pair.reference_score,
                "law_type": pair.law_type,
                "preference_supervision": pair.preference_supervision_metadata(),
                "truth_label_source": pair.truth_label_source,
                "oracle_view": pair.oracle_view,
                "oracle_proxy_source": pair.oracle_proxy_source,
                "source_doc_id": pair.source_doc_id,
                "three_layer_roles": pair.three_layer_roles,
                "sample_weight": float(
                    pair.treepo_metadata(
                        tree_objective_weighting_mode=tree_objective_weighting_mode,
                        discount_gamma=discount_gamma,
                    )["effective_weight"]
                ),
                "comparative_signal": pair.comparative_signal_payload(),
                "treepo": pair.treepo_metadata(
                    tree_objective_weighting_mode=tree_objective_weighting_mode,
                    discount_gamma=discount_gamma,
                ),
            })

        return general_data

    def to_reward_model_format(
        self,
        law_type: Optional[str] = None,
        include_oracle_scores: bool = True,
        prompt_builder: Optional[PromptBuilder] = None,
        *,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Convert to reward model training format.

        This format is optimized for training reward models that approximate
        the oracle/judge. Each response gets a scalar score derived from
        the pairwise comparisons.

        The score computation uses:
        - Oracle estimate scores if available (from GenRM or oracle scorer)
        - Fallback to preference + confidence-based scoring

        Args:
            law_type: Optional filter for specific law type
            include_oracle_scores: Include raw oracle score estimates if available
            prompt_builder: Optional prompt builder for generating prompts

        Returns:
            List of dicts with prompt, response, score, and optional oracle_estimate
        """
        rm_data = []
        for pair in self.pairs:
            if law_type is not None and pair.law_type != law_type:
                continue

            prompt = render_prompt(pair.original_text, pair.rubric, prompt_builder)

            # Use oracle estimate scores if available, else derive from preference
            if include_oracle_scores and pair.score_estimate_a is not None:
                score_a = pair.score_estimate_a
                score_b = pair.score_estimate_b if pair.score_estimate_b is not None else 0.5
            else:
                # Derive scores from preference and confidence
                if pair.preferred == "A":
                    score_a = 0.5 + pair.confidence * 0.5
                    score_b = 0.5 - pair.confidence * 0.5
                elif pair.preferred == "B":
                    score_a = 0.5 - pair.confidence * 0.5
                    score_b = 0.5 + pair.confidence * 0.5
                else:  # tie
                    score_a = 0.5
                    score_b = 0.5

            base_metadata = pair.optimization_metadata(
                rl_role="reward_pair",
                tree_objective_weighting_mode=tree_objective_weighting_mode,
                discount_gamma=discount_gamma,
            )

            rm_data.append({
                "prompt": prompt,
                "response": pair.summary_a,
                "score": score_a,
                "sample_weight": float(base_metadata["treepo"]["effective_weight"]),
                "oracle_estimate": pair.score_estimate_a,
                "oracle_error": pair.oracle_error_a,
                "metadata": {**base_metadata, "response_id": "A"},
            })
            rm_data.append({
                "prompt": prompt,
                "response": pair.summary_b,
                "score": score_b,
                "sample_weight": float(base_metadata["treepo"]["effective_weight"]),
                "oracle_estimate": pair.score_estimate_b,
                "oracle_error": pair.oracle_error_b,
                "metadata": {**base_metadata, "response_id": "B"},
            })

        return rm_data

    def to_grouped_grpo_format(
        self,
        law_type: Optional[str] = None,
        min_group_size: int = 2,
        prompt_builder: Optional[PromptBuilder] = None,
        *,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Convert to grouped GRPO format for k-wise rankings.

        Groups multiple responses for the same input (original_text + rubric)
        and provides rankings across all responses. This supports the
        Plackett-Luce style k-wise GRPO objective.

        Args:
            law_type: Optional filter for specific law type
            min_group_size: Minimum group size to include (default: 2)
            prompt_builder: Optional prompt builder for generating prompts

        Returns:
            List of dicts with prompt, responses (k items), and rankings
        """
        comparative_dataset = self.to_comparative_dataset(law_type=law_type)
        return comparative_dataset.to_grouped_grpo_format(
            prompt_builder=prompt_builder,
            min_group_size=min_group_size,
            tree_objective_weighting_mode=tree_objective_weighting_mode,
            discount_gamma=discount_gamma,
        )

    def to_dpo_format(
        self,
        law_type: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        *,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Convert to DPO (Direct Preference Optimization) format.

        .. deprecated::
            Use `to_preference_format(method='dpo')` instead for consistency
            with the generalized preference learning framework.

        Returns:
            List of dicts with prompt, chosen, rejected
        """
        import warnings
        warnings.warn(
            "to_dpo_format() is deprecated. Use to_preference_format(method='dpo') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._to_dpo_format(
            law_type,
            prompt_builder,
            tree_objective_weighting_mode=tree_objective_weighting_mode,
            discount_gamma=discount_gamma,
        )

    def save(self, path: Path):
        """Save dataset to JSON file."""
        warnings.warn(
            "PreferenceDataset.save() is a compatibility path. "
            "Prefer SupervisionDataset.save() for primary artifacts.",
            DeprecationWarning,
            stacklevel=2,
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "2.0",
            "created_at": datetime.now().isoformat(),
            "num_pairs": len(self.pairs),
            "pairs": [p.to_dict() for p in self.pairs],
            "num_comparative_records": len(self.comparative_records),
            "comparative_records": [record.to_dict() for record in self.comparative_records],
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(
            "Saved %d preference pairs and %d comparative records to %s",
            len(self.pairs),
            len(self.comparative_records),
            path,
        )

    @classmethod
    def load(cls, path: Path) -> 'PreferenceDataset':
        """Load dataset from JSON file."""
        warnings.warn(
            "PreferenceDataset.load() is a compatibility path. "
            "Prefer SupervisionDataset.load() for primary artifacts.",
            DeprecationWarning,
            stacklevel=2,
        )
        with open(path) as f:
            data = json.load(f)

        version = str(data.get("version", ""))
        if version.startswith("3.") or "response_judgments" in data or "comparative_judgments" in data:
            from treepo._research.training.supervision import SupervisionDataset

            return SupervisionDataset.load(path).project_binary(projection="adjacent")

        pairs = [PreferencePair.from_dict(p) for p in list(data.get("pairs", []) or [])]
        comparative_records = [
            ComparativeJudgmentRecord.from_dict(record)
            for record in list(data.get("comparative_records", []) or [])
        ]
        logger.info(
            "Loaded %d preference pairs and %d comparative records from %s",
            len(pairs),
            len(comparative_records),
            path,
        )

        return cls(pairs, comparative_records=comparative_records)

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics about the dataset."""
        non_ties = [p for p in self.pairs if p.preferred != "tie"]
        with_propensity = self.pairs
        with_audit_context = [
            p for p in self.pairs
            if getattr(p, "audit_tree_id", None) is not None
        ]
        propensity_stats = self.propensity_diagnostics(include_ties=True)

        return {
            "total_pairs": len(self.pairs),
            "total_comparative_records": len(self.comparative_records),
            "total_groupwise_records": len(self.to_comparative_dataset().records),
            "non_tie_pairs": len(non_ties),
            "tie_pairs": len(self.pairs) - len(non_ties),
            "prefer_a": sum(1 for p in self.pairs if p.preferred == "A"),
            "prefer_b": sum(1 for p in self.pairs if p.preferred == "B"),
            "avg_confidence": (
                sum(p.confidence for p in self.pairs) / len(self.pairs)
                if self.pairs else 0
            ),
            "high_confidence_pairs": sum(1 for p in self.pairs if p.confidence >= 0.8),
            "pairs_with_propensity": len(with_propensity),
            "pairs_with_audit_context": len(with_audit_context),
            "mean_joint_propensity": propensity_stats["mean_joint_propensity"],
            "mean_sample_weight": propensity_stats["mean_sample_weight"],
            "max_sample_weight": propensity_stats["max_sample_weight"],
            "effective_sample_size": propensity_stats["effective_sample_size"],
            "effective_sample_ratio": propensity_stats["effective_sample_ratio"],
        }
