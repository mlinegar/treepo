"""Primary supervision data surface for scalar and comparative judgments."""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

from treepo._research.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from treepo._research.core.supervision_metadata import (
    JudgmentSupervisionMetadata,
    judgment_supervision_metadata,
)
from treepo._research.core.provenance import normalize_truth_label_source
from treepo._research.training.supervision.comparative_types import (
    MAX_PROPENSITY,
    MIN_PROPENSITY,
    ComparativeCandidate as _LegacyComparativeCandidate,
    ComparativeDataset as _LegacyComparativeDataset,
    ComparativeJudgmentRecord as _LegacyComparativeJudgmentRecord,
    PreferenceDataset as _LegacyPreferenceDataset,
    PreferencePair as _LegacyPreferencePair,
    PromptBuilder,
    compute_propensity_diagnostics as _legacy_compute_propensity_diagnostics,
    render_prompt,
)
from treepo._research.training.supervision.optimizer_metadata import (
    TreePORLRole,
    TreePOWeightingMode,
    build_treepo_optimizer_export_metadata,
)

logger = logging.getLogger(__name__)

BinaryProjectionMode = Literal["winner_vs_runner_up", "adjacent"]
ComparativeCandidate = _LegacyComparativeCandidate
ComparativeJudgment = _LegacyComparativeJudgmentRecord
ComparativeDataset = _LegacyComparativeDataset
BinaryComparison = _LegacyPreferencePair


class BinaryProjectionDataset:
    """Optimizer-facing binary projection derived from canonical supervision."""

    def __init__(
        self,
        comparisons: Optional[List[BinaryComparison]] = None,
        comparative_judgments: Optional[List[ComparativeJudgment]] = None,
    ) -> None:
        self.comparisons = comparisons or []
        self.comparative_judgments = comparative_judgments or []

    def __len__(self) -> int:
        return len(self.comparisons)

    def __iter__(self):
        return iter(self.comparisons)

    def __getitem__(self, idx: int) -> BinaryComparison:
        return self.comparisons[idx]

    @property
    def pairs(self) -> List[BinaryComparison]:
        """Internal compatibility alias while callers finish the V2 cutover."""
        return self.comparisons

    @property
    def comparative_records(self) -> List[ComparativeJudgment]:
        """Internal compatibility alias while callers finish the V2 cutover."""
        return self.comparative_judgments

    def add_comparison(self, comparison: BinaryComparison) -> None:
        self.comparisons.append(comparison)

    def add_comparisons(self, comparisons: Iterable[BinaryComparison]) -> None:
        self.comparisons.extend(comparisons)

    def add_comparative_judgments(
        self,
        judgments: Iterable[ComparativeJudgment],
    ) -> None:
        self.comparative_judgments.extend(judgments)

    def _to_legacy_dataset(self) -> _LegacyPreferenceDataset:
        return _LegacyPreferenceDataset(
            list(self.comparisons),
            comparative_records=list(self.comparative_judgments),
        )

    @classmethod
    def _from_legacy_dataset(
        cls,
        dataset: _LegacyPreferenceDataset,
    ) -> "BinaryProjectionDataset":
        return cls(
            comparisons=list(dataset.pairs),
            comparative_judgments=list(dataset.comparative_records),
        )

    def to_comparative_dataset(
        self,
        law_type: Optional[str] = None,
    ) -> ComparativeDataset:
        return self._to_legacy_dataset().to_comparative_dataset(law_type=law_type)

    def get_sample_weights(
        self,
        min_propensity: float = MIN_PROPENSITY,
        max_weight: Optional[float] = None,
    ) -> List[float]:
        return [
            comparison.ipw_weight(
                min_propensity=min_propensity,
                max_weight=max_weight,
            )
            for comparison in self.comparisons
        ]

    def propensity_diagnostics(
        self,
        *,
        include_ties: bool = True,
        min_propensity: float = MIN_PROPENSITY,
        max_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        return _legacy_compute_propensity_diagnostics(
            list(self.comparisons),
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
        strategy: Literal[
            "multinomial",
            "pps_systematic",
            "stratified_multinomial",
        ] = "pps_systematic",
        stratify_by: Optional[str] = None,
    ) -> "BinaryProjectionDataset":
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
        strategy: Literal[
            "multinomial",
            "pps_systematic",
            "stratified_multinomial",
        ] = "pps_systematic",
        stratify_by: Optional[str] = None,
    ) -> "BinaryProjectionDataset":
        sampled = self._to_legacy_dataset().sample_by_propensity(
            target_size=target_size,
            seed=seed,
            max_weight=max_weight,
            min_propensity=min_propensity,
            strategy=strategy,
            stratify_by=stratify_by,
        )
        return self._from_legacy_dataset(sampled)

    def filter_by_confidence(self, min_confidence: float) -> "BinaryProjectionDataset":
        return self._from_legacy_dataset(
            self._to_legacy_dataset().filter_by_confidence(min_confidence)
        )

    def filter_non_ties(self) -> "BinaryProjectionDataset":
        return self._from_legacy_dataset(
            self._to_legacy_dataset().filter_non_ties()
        )

    def split(
        self,
        train_ratio: float = 0.8,
        shuffle: bool = True,
    ) -> Tuple["BinaryProjectionDataset", "BinaryProjectionDataset"]:
        train_set, val_set = self._to_legacy_dataset().split(
            train_ratio=train_ratio,
            shuffle=shuffle,
        )
        return (
            self._from_legacy_dataset(train_set),
            self._from_legacy_dataset(val_set),
        )

    def to_dspy_examples(self) -> List[Any]:
        return self._to_legacy_dataset().to_dspy_examples()

    def to_dpo_records(
        self,
        *,
        law_type: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
    ) -> List[Dict[str, Any]]:
        return self.to_preference_format(
            method="dpo",
            law_type=law_type,
            prompt_builder=prompt_builder,
            tree_objective_weighting_mode=tree_objective_weighting_mode,
            discount_gamma=discount_gamma,
        )

    def to_preference_format(
        self,
        *,
        method: Literal["dpo", "reward", "grpo"] = "dpo",
        law_type: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
    ) -> List[Dict[str, Any]]:
        return self._to_legacy_dataset().to_preference_format(
            method=method,
            law_type=law_type,
            prompt_builder=prompt_builder,
            tree_objective_weighting_mode=tree_objective_weighting_mode,
            discount_gamma=discount_gamma,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": "4.0",
            "num_binary_comparisons": len(self.comparisons),
            "binary_comparisons": [
                comparison.to_dict() for comparison in self.comparisons
            ],
            "num_comparative_judgments": len(self.comparative_judgments),
            "comparative_judgments": [
                judgment.to_dict() for judgment in self.comparative_judgments
            ],
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BinaryProjectionDataset":
        path = Path(path)
        with open(path) as handle:
            payload = json.load(handle)
        version = str(payload.get("version", ""))
        if version.startswith("4.") or "binary_comparisons" in payload:
            return cls(
                comparisons=[
                    BinaryComparison.from_dict(row)
                    for row in list(payload.get("binary_comparisons", []) or [])
                ],
                comparative_judgments=[
                    ComparativeJudgment.from_dict(row)
                    for row in list(payload.get("comparative_judgments", []) or [])
                ],
            )
        return cls._from_legacy_dataset(_LegacyPreferenceDataset.load(path))


@dataclass
class ResponseJudgment:
    """Canonical scalar supervision for one response."""

    judgment_id: str
    source_example_id: str
    original_text: str
    rubric: str
    response: str
    response_id: Optional[str] = None
    reference_score: float = 0.0
    law_type: str = "sufficiency"
    source_doc_id: Optional[str] = None
    three_layer_roles: Dict[str, str] = field(default_factory=dict)
    truth_label_source: str = "unknown"
    oracle_view: Optional[str] = None
    oracle_proxy_source: Optional[str] = None
    sampling: SamplingMetadata = field(
        default_factory=lambda: SamplingMetadata(unit_kind=ObservationUnitKind.PAIR)
    )
    supervision_metadata: JudgmentSupervisionMetadata = field(
        default_factory=judgment_supervision_metadata
    )
    source_observation_ids: List[str] = field(default_factory=list)
    response_signal_value: Optional[float] = None
    response_signal_vector: Optional[List[float]] = None
    candidate_features: Optional[List[float]] = None
    judge_model: str = ""
    timestamp: Optional[str] = None
    generation_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.judgment_id = str(self.judgment_id)
        self.source_example_id = str(self.source_example_id)
        self.original_text = str(self.original_text)
        self.rubric = str(self.rubric)
        self.response = str(self.response)
        if self.response_id is not None:
            self.response_id = str(self.response_id)
        self.reference_score = float(self.reference_score)
        self.law_type = str(self.law_type or "sufficiency")
        self.truth_label_source = normalize_truth_label_source(self.truth_label_source)
        if not isinstance(self.sampling, SamplingMetadata):
            self.sampling = SamplingMetadata.from_dict(self.sampling)
        if self.sampling.unit_kind is None:
            self.sampling = self.sampling.with_updates(unit_kind=ObservationUnitKind.PAIR)
        if not isinstance(self.supervision_metadata, JudgmentSupervisionMetadata):
            self.supervision_metadata = JudgmentSupervisionMetadata.from_dict(
                self.supervision_metadata
            )
        if self.supervision_metadata.law_type is None and self.law_type:
            self.supervision_metadata = self.supervision_metadata.with_updates(
                law_type=self.law_type
            )
        if self.response_signal_value is not None:
            self.response_signal_value = float(self.response_signal_value)
        if self.response_signal_vector is not None:
            self.response_signal_vector = [float(value) for value in self.response_signal_vector]
        if self.candidate_features is not None:
            self.candidate_features = [float(value) for value in self.candidate_features]
        self.source_observation_ids = [str(value) for value in self.source_observation_ids]
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if not isinstance(self.three_layer_roles, dict):
            self.three_layer_roles = dict(self.three_layer_roles)
        if self.generation_config is not None and not isinstance(self.generation_config, dict):
            self.generation_config = dict(self.generation_config)
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)

    def effective_joint_propensity(self, min_propensity: float = MIN_PROPENSITY) -> float:
        return self.sampling.effective_joint_propensity(min_propensity=min_propensity)

    def ipw_weight(
        self,
        min_propensity: float = MIN_PROPENSITY,
        max_weight: Optional[float] = None,
    ) -> float:
        return self.sampling.ipw_weight(
            min_propensity=min_propensity,
            max_weight=max_weight,
        )

    def supervision_metadata_payload(self) -> Dict[str, Any]:
        metadata = {
            **self.supervision_metadata.to_dict(),
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

    def scalar_signal_payload(self) -> Dict[str, Any]:
        payload = {
            "response_signal_name": self.supervision_metadata.response_signal_name,
            "response_signal_value": self.response_signal_value,
            "response_signal_vector": list(self.response_signal_vector)
            if self.response_signal_vector is not None
            else None,
            "response_signal_min": self.supervision_metadata.response_signal_min,
            "response_signal_max": self.supervision_metadata.response_signal_max,
        }
        return {
            key: value
            for key, value in payload.items()
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
            fallback_node_id=self.judgment_id,
            source_example_id=self.source_example_id,
            source_doc_id=self.source_doc_id,
            source_observation_ids=self.source_observation_ids,
            sampling=self.sampling,
            law_type=self.law_type,
            supervision_channel_name=self.supervision_metadata.supervision_channel_name,
            supervision_signal_name=self.supervision_metadata.supervision_signal_name,
            metadata_sources=(
                self.supervision_metadata.metadata,
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
            "judgment_id": self.judgment_id,
            "reference_score": self.reference_score,
            "law_type": self.law_type,
            "supervision_metadata": self.supervision_metadata_payload(),
            "scalar_signal": self.scalar_signal_payload(),
            "truth_label_source": self.truth_label_source,
            "oracle_view": self.oracle_view,
            "oracle_proxy_source": self.oracle_proxy_source,
            "source_doc_id": self.source_doc_id,
            "three_layer_roles": dict(self.three_layer_roles),
            "treepo": self.treepo_metadata(
                rl_role=rl_role,
                tree_objective_weighting_mode=tree_objective_weighting_mode,
                discount_gamma=discount_gamma,
                ipw_weight_override=ipw_weight_override,
                joint_propensity_override=joint_propensity_override,
                joint_propensity_source=joint_propensity_source,
            ),
        }
        if self.response_id is not None:
            metadata["response_id"] = self.response_id
        if self.generation_config is not None:
            metadata["generation_config"] = dict(self.generation_config)
        if self.metadata:
            metadata["judgment_metadata"] = dict(self.metadata)
        return {
            key: value
            for key, value in metadata.items()
            if value is not None and value != [] and value != {}
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "judgment_id": self.judgment_id,
            "source_example_id": self.source_example_id,
            "original_text": self.original_text,
            "rubric": self.rubric,
            "response": self.response,
            "response_id": self.response_id,
            "reference_score": self.reference_score,
            "law_type": self.law_type,
            "source_doc_id": self.source_doc_id,
            "three_layer_roles": dict(self.three_layer_roles),
            "truth_label_source": self.truth_label_source,
            "oracle_view": self.oracle_view,
            "oracle_proxy_source": self.oracle_proxy_source,
            "sampling": self.sampling.to_dict(),
            "supervision_metadata": self.supervision_metadata.to_dict(),
            "source_observation_ids": list(self.source_observation_ids),
            "response_signal_value": self.response_signal_value,
            "response_signal_vector": list(self.response_signal_vector)
            if self.response_signal_vector is not None
            else None,
            "candidate_features": list(self.candidate_features)
            if self.candidate_features is not None
            else None,
            "judge_model": self.judge_model,
            "timestamp": self.timestamp,
            "generation_config": dict(self.generation_config or {}) or None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResponseJudgment":
        payload = dict(data)
        payload["supervision_metadata"] = JudgmentSupervisionMetadata.from_dict(
            payload.get("supervision_metadata", {})
        )
        payload["sampling"] = SamplingMetadata.from_dict(payload.get("sampling", {}))
        return cls(**payload)


SupervisionInput = Union[
    "SupervisionDataset",
    BinaryProjectionDataset,
    ComparativeDataset,
    Sequence[ResponseJudgment],
    Sequence[ComparativeJudgment],
    Sequence[BinaryComparison],
    _LegacyPreferenceDataset,
    _LegacyComparativeDataset,
    Sequence[_LegacyComparativeJudgmentRecord],
    Sequence[_LegacyPreferencePair],
]


class SupervisionDataset:
    """Primary public supervision container for scalar and comparative judgments."""

    def __init__(
        self,
        response_judgments: Optional[List[ResponseJudgment]] = None,
        comparative_judgments: Optional[List[ComparativeJudgment]] = None,
    ) -> None:
        self.response_judgments = response_judgments or []
        self.comparative_judgments = comparative_judgments or []

    def __len__(self) -> int:
        return len(self.response_judgments) + len(self.comparative_judgments)

    def add_response_judgment(self, judgment: ResponseJudgment) -> None:
        self.response_judgments.append(judgment)

    def add_response_judgments(self, judgments: Iterable[ResponseJudgment]) -> None:
        self.response_judgments.extend(judgments)

    def add_comparative_judgment(self, judgment: ComparativeJudgment) -> None:
        self.comparative_judgments.append(judgment)

    def add_comparative_judgments(
        self,
        judgments: Iterable[ComparativeJudgment],
    ) -> None:
        self.comparative_judgments.extend(judgments)

    @staticmethod
    def _group_key_from_response(judgment: ResponseJudgment) -> Tuple[Any, ...]:
        return (
            judgment.source_example_id,
            judgment.law_type,
            judgment.original_text,
            judgment.rubric,
            judgment.reference_score,
            judgment.source_doc_id,
        )

    @staticmethod
    def _group_key_from_record(record: ComparativeJudgment) -> Tuple[Any, ...]:
        return (
            record.source_example_id,
            record.law_type,
            record.original_text,
            record.rubric,
            record.reference_score,
            record.source_doc_id,
        )

    def _response_grouped_records(
        self,
        *,
        law_type: Optional[str] = None,
        skip_group_keys: Optional[set[Tuple[Any, ...]]] = None,
    ) -> List[ComparativeJudgment]:
        grouped: Dict[Tuple[Any, ...], List[ResponseJudgment]] = defaultdict(list)
        for judgment in self.response_judgments:
            if law_type is not None and judgment.law_type != law_type:
                continue
            grouped[self._group_key_from_response(judgment)].append(judgment)

        records: List[ComparativeJudgment] = []
        skipped = skip_group_keys or set()
        for group_key, judgments in grouped.items():
            if not judgments or group_key in skipped:
                continue
            ordered = sorted(
                judgments,
                key=lambda value: (
                    -(
                        float(value.response_signal_value)
                        if value.response_signal_value is not None
                        else float("-inf")
                    ),
                    value.response_id or "",
                    value.judgment_id,
                ),
            )
            current_rank = 0
            previous_score: Optional[float] = None
            candidates: List[ComparativeCandidate] = []
            source_observation_ids: List[str] = []
            aggregate_sample_weight = 0.0
            for index, judgment in enumerate(ordered, start=1):
                score = (
                    float(judgment.response_signal_value)
                    if judgment.response_signal_value is not None
                    else float("-inf")
                )
                if previous_score is None or not math.isclose(
                    score,
                    previous_score,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    current_rank = index
                    previous_score = score
                candidates.append(
                    ComparativeCandidate(
                        candidate_id=judgment.response_id or judgment.judgment_id,
                        response=judgment.response,
                        rank=current_rank,
                        response_signal_value=judgment.response_signal_value,
                        candidate_features=(
                            list(judgment.candidate_features)
                            if judgment.candidate_features is not None
                            else None
                        ),
                        source_pair_ids=[],
                        metadata={
                            "response_id": judgment.response_id,
                            "generation_config": judgment.generation_config,
                            "source_judgment_id": judgment.judgment_id,
                            "aggregate_method": "direct_response_signal",
                        },
                    )
                )
                source_observation_ids.extend(judgment.source_observation_ids)
                aggregate_sample_weight += judgment.ipw_weight()

            first = ordered[0]
            records.append(
                ComparativeJudgment(
                    record_id=f"{first.source_example_id}:{first.law_type}:response_group",
                    source_example_id=first.source_example_id,
                    original_text=first.original_text,
                    rubric=first.rubric,
                    reference_score=first.reference_score,
                    law_type=first.law_type,
                    candidates=candidates,
                    sampling=first.sampling,
                    preference_supervision=first.supervision_metadata.with_updates(
                        preference_family="groupwise",
                        metadata={
                            **dict(first.supervision_metadata.metadata or {}),
                            "aggregation_method": "direct_response_signal",
                            "num_source_judgments": len(ordered),
                        },
                    ),
                    source_observation_ids=list(dict.fromkeys(source_observation_ids)),
                    aggregate_sample_weight=aggregate_sample_weight,
                    source_doc_id=first.source_doc_id,
                    three_layer_roles=dict(first.three_layer_roles),
                    truth_label_source=first.truth_label_source,
                    oracle_view=first.oracle_view,
                    oracle_proxy_source=first.oracle_proxy_source,
                    judge_model=first.judge_model,
                    comparison_signal_value=None,
                    timestamp=first.timestamp,
                    metadata={
                        "aggregation_method": "direct_response_signal",
                        "source_judgment_ids": [judgment.judgment_id for judgment in ordered],
                    },
                )
            )
        return records

    def _binary_projection_dataset(
        self,
        *,
        law_type: Optional[str] = None,
    ) -> BinaryProjectionDataset:
        projected_pairs: List[BinaryComparison] = []
        direct_groupwise_records: List[ComparativeJudgment] = []
        for record in self.comparative_judgments:
            if law_type is not None and record.law_type != law_type:
                continue
            if len(record.candidates) == 2:
                projected_pairs.extend(
                    record.to_preference_pairs(projection="winner_vs_runner_up")
                )
            else:
                direct_groupwise_records.append(record)
        return BinaryProjectionDataset(
            comparisons=projected_pairs,
            comparative_judgments=direct_groupwise_records,
        )

    def to_comparative_dataset(
        self,
        law_type: Optional[str] = None,
    ) -> ComparativeDataset:
        """Return groupwise comparative records for grouped optimizers."""
        binary_projection = self._binary_projection_dataset(law_type=law_type)
        comparative_dataset = binary_projection.to_comparative_dataset(law_type=law_type)
        existing_keys = {
            self._group_key_from_record(record)
            for record in comparative_dataset.records
        }
        comparative_dataset.records.extend(
            self._response_grouped_records(
                law_type=law_type,
                skip_group_keys=existing_keys,
            )
        )
        return comparative_dataset

    def project_binary(
        self,
        *,
        projection: BinaryProjectionMode = "adjacent",
    ) -> BinaryProjectionDataset:
        """Project canonical supervision to an optimizer-facing binary view."""
        projected_pairs: List[BinaryComparison] = []
        seen_pair_ids = set()

        for record in self.comparative_judgments:
            for pair in record.to_preference_pairs(projection=projection):
                if pair.pair_id in seen_pair_ids:
                    continue
                projected_pairs.append(pair)
                seen_pair_ids.add(pair.pair_id)

        for record in self._response_grouped_records():
            for pair in record.to_preference_pairs(projection=projection):
                if pair.pair_id in seen_pair_ids:
                    continue
                projected_pairs.append(pair)
                seen_pair_ids.add(pair.pair_id)

        return BinaryProjectionDataset(comparisons=projected_pairs)

    def to_dpo_records(
        self,
        *,
        law_type: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        projection: BinaryProjectionMode = "adjacent",
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
    ) -> List[Dict[str, Any]]:
        return self.project_binary(projection=projection).to_preference_format(
            method="dpo",
            law_type=law_type,
            prompt_builder=prompt_builder,
            tree_objective_weighting_mode=tree_objective_weighting_mode,
            discount_gamma=discount_gamma,
        )

    def to_reward_pairs(
        self,
        *,
        law_type: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        projection: BinaryProjectionMode = "adjacent",
        include_oracle_scores: bool = True,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
    ) -> List[Dict[str, Any]]:
        reward_pairs: List[Dict[str, Any]] = []
        dataset = self.project_binary(projection=projection)
        for pair in dataset.comparisons:
            if pair.preferred == "tie":
                continue
            if law_type is not None and pair.law_type != law_type:
                continue
            if pair.preferred == "A":
                chosen = pair.summary_a
                rejected = pair.summary_b
                chosen_score = pair.score_estimate_a
                rejected_score = pair.score_estimate_b
                chosen_error = pair.oracle_error_a
                rejected_error = pair.oracle_error_b
            else:
                chosen = pair.summary_b
                rejected = pair.summary_a
                chosen_score = pair.score_estimate_b
                rejected_score = pair.score_estimate_a
                chosen_error = pair.oracle_error_b
                rejected_error = pair.oracle_error_a
            reward_pairs.append(
                {
                    "prompt": render_prompt(pair.original_text, pair.rubric, prompt_builder),
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_score": chosen_score if include_oracle_scores else None,
                    "rejected_score": rejected_score if include_oracle_scores else None,
                    "chosen_error": chosen_error,
                    "rejected_error": rejected_error,
                    "sample_weight": float(
                        pair.treepo_metadata(
                            rl_role="reward_pair",
                            tree_objective_weighting_mode=tree_objective_weighting_mode,
                            discount_gamma=discount_gamma,
                        )["effective_weight"]
                    ),
                    "law_type": pair.law_type,
                    "preference_supervision": pair.preference_supervision_metadata(),
                    "comparative_signal": pair.comparative_signal_payload(),
                    "metadata": pair.optimization_metadata(
                        rl_role="reward_pair",
                        tree_objective_weighting_mode=tree_objective_weighting_mode,
                        discount_gamma=discount_gamma,
                    ),
                }
            )
        return reward_pairs

    def to_group_grpo_records(
        self,
        *,
        law_type: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        min_group_size: int = 2,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
    ) -> List[Dict[str, Any]]:
        return self.to_comparative_dataset(law_type=law_type).to_grouped_grpo_format(
            prompt_builder=prompt_builder,
            min_group_size=min_group_size,
            tree_objective_weighting_mode=tree_objective_weighting_mode,
            discount_gamma=discount_gamma,
        )

    def to_scalar_reward_records(
        self,
        *,
        law_type: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel",
        discount_gamma: float = 1.0,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for judgment in self.response_judgments:
            if law_type is not None and judgment.law_type != law_type:
                continue
            metadata = judgment.optimization_metadata(
                rl_role="scalar_reward",
                tree_objective_weighting_mode=tree_objective_weighting_mode,
                discount_gamma=discount_gamma,
            )
            rows.append(
                {
                    "prompt": render_prompt(
                        judgment.original_text,
                        judgment.rubric,
                        prompt_builder,
                    ),
                    "response": judgment.response,
                    "score": judgment.response_signal_value,
                    "sample_weight": float(metadata["treepo"]["effective_weight"]),
                    "metadata": metadata,
                }
            )

        for record in self.comparative_judgments:
            if law_type is not None and record.law_type != law_type:
                continue
            per_candidate_weight = float(record.aggregate_sample_weight) / max(
                1,
                len(record.candidates),
            )
            max_rank = max(
                (candidate.rank for candidate in record.candidates if candidate.rank is not None),
                default=len(record.candidates),
            )
            for candidate in record.candidates:
                score = candidate.response_signal_value
                if score is None:
                    rank = candidate.rank if candidate.rank is not None else max_rank
                    score = float(max_rank - rank + 1)
                treepo_metadata = record.treepo_metadata(
                    rl_role="scalar_reward",
                    tree_objective_weighting_mode=tree_objective_weighting_mode,
                    discount_gamma=discount_gamma,
                    ipw_weight_override=per_candidate_weight,
                    joint_propensity_override=(
                        1.0 / per_candidate_weight if per_candidate_weight > 0.0 else 1.0
                    ),
                    joint_propensity_source="aggregate_candidate_weight",
                )
                rows.append(
                    {
                        "prompt": render_prompt(
                            record.original_text,
                            record.rubric,
                            prompt_builder,
                        ),
                        "response": candidate.response,
                        "score": score,
                        "sample_weight": float(treepo_metadata["effective_weight"]),
                        "metadata": {
                            **record.optimization_metadata(
                                rl_role="scalar_reward",
                                tree_objective_weighting_mode=tree_objective_weighting_mode,
                                discount_gamma=discount_gamma,
                                ipw_weight_override=per_candidate_weight,
                                joint_propensity_override=(
                                    1.0 / per_candidate_weight
                                    if per_candidate_weight > 0.0
                                    else 1.0
                                ),
                                joint_propensity_source="aggregate_candidate_weight",
                            ),
                            "candidate_id": candidate.candidate_id,
                            "response_id": candidate.metadata.get("response_id"),
                            "treepo": treepo_metadata,
                        },
                    }
                )
        return rows

    def to_dense_scalar_training_records(
        self,
        *,
        law_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for judgment in self.response_judgments:
            if law_type is not None and judgment.law_type != law_type:
                continue
            if judgment.response_signal_value is None or judgment.candidate_features is None:
                continue
            rows.append(
                {
                    "features": list(judgment.candidate_features),
                    "score": float(judgment.response_signal_value),
                    "sample_weight": judgment.ipw_weight(),
                    "metadata": judgment.optimization_metadata(),
                }
            )

        for record in self.comparative_judgments:
            if law_type is not None and record.law_type != law_type:
                continue
            per_candidate_weight = float(record.aggregate_sample_weight) / max(
                1,
                len(record.candidates),
            )
            max_rank = max(
                (candidate.rank for candidate in record.candidates if candidate.rank is not None),
                default=len(record.candidates),
            )
            for candidate in record.candidates:
                features = candidate.candidate_features
                if not features:
                    continue
                score = candidate.response_signal_value
                if score is None:
                    rank = candidate.rank if candidate.rank is not None else max_rank
                    score = float(max_rank - rank + 1)
                rows.append(
                    {
                        "features": [float(value) for value in features],
                        "score": float(score),
                        "sample_weight": per_candidate_weight,
                        "metadata": {
                            **record.optimization_metadata(),
                            "candidate_id": candidate.candidate_id,
                            "response_id": candidate.metadata.get("response_id"),
                        },
                    }
                )
        return rows

    def to_dense_vector_training_records(
        self,
        *,
        law_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for judgment in self.response_judgments:
            if law_type is not None and judgment.law_type != law_type:
                continue
            if judgment.response_signal_vector is None or judgment.candidate_features is None:
                continue
            rows.append(
                {
                    "features": list(judgment.candidate_features),
                    "target": list(judgment.response_signal_vector),
                    "sample_weight": judgment.ipw_weight(),
                    "metadata": judgment.optimization_metadata(),
                }
            )
        return rows

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "3.0",
            "created_at": datetime.now().isoformat(),
            "num_response_judgments": len(self.response_judgments),
            "response_judgments": [
                judgment.to_dict() for judgment in self.response_judgments
            ],
            "num_comparative_judgments": len(self.comparative_judgments),
            "comparative_judgments": [
                record.to_dict() for record in self.comparative_judgments
            ],
        }
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)
        logger.info(
            "Saved %d response judgments and %d comparative judgments to %s",
            len(self.response_judgments),
            len(self.comparative_judgments),
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "SupervisionDataset":
        path = Path(path)
        with open(path) as handle:
            payload = json.load(handle)

        version = str(payload.get("version", ""))
        if version.startswith("3."):
            response_judgments = [
                ResponseJudgment.from_dict(judgment)
                for judgment in list(payload.get("response_judgments", []) or [])
            ]
            comparative_judgments = [
                ComparativeJudgment.from_dict(record)
                for record in list(payload.get("comparative_judgments", []) or [])
            ]
            logger.info(
                "Loaded %d response judgments and %d comparative judgments from %s",
                len(response_judgments),
                len(comparative_judgments),
                path,
            )
            return cls(
                response_judgments=response_judgments,
                comparative_judgments=comparative_judgments,
            )

        if "pairs" in payload or version.startswith("2.") or version.startswith("1."):
            dataset = _LegacyPreferenceDataset.load(path)
            return dataset.to_supervision_dataset()

        if "records" in payload:
            comparative_dataset = ComparativeDataset.load(path)
            return cls(comparative_judgments=list(comparative_dataset.records))

        raise ValueError(
            f"Unsupported supervision dataset format in {path}: version={version!r}"
        )

    def summary(self) -> Dict[str, Any]:
        comparative_dataset = self.to_comparative_dataset()
        projected_binary = self.project_binary()
        return {
            "total_response_judgments": len(self.response_judgments),
            "total_comparative_judgments": len(self.comparative_judgments),
            "total_groupwise_records": len(comparative_dataset.records),
            "total_binary_comparisons": len(projected_binary.comparisons),
            "scalar_judgments_with_scores": sum(
                1
                for judgment in self.response_judgments
                if judgment.response_signal_value is not None
            ),
        }


def coerce_supervision_dataset(supervision: SupervisionInput) -> SupervisionDataset:
    """Coerce legacy preference data or direct judgments into SupervisionDataset."""
    if isinstance(supervision, SupervisionDataset):
        return supervision
    if isinstance(supervision, BinaryProjectionDataset):
        return SupervisionDataset(
            comparative_judgments=[
                comparison.to_comparative_judgment()
                for comparison in supervision.comparisons
            ]
            + list(supervision.comparative_judgments)
        )
    if isinstance(supervision, _LegacyPreferenceDataset):
        return supervision.to_supervision_dataset()
    if isinstance(supervision, ComparativeDataset):
        return SupervisionDataset(comparative_judgments=list(supervision.records))

    values = list(supervision)
    if not values:
        return SupervisionDataset()

    response_judgments: List[ResponseJudgment] = []
    comparative_judgments: List[ComparativeJudgment] = []
    for value in values:
        if isinstance(value, ResponseJudgment):
            response_judgments.append(value)
        elif isinstance(value, ComparativeJudgment):
            comparative_judgments.append(value)
        elif isinstance(value, BinaryComparison):
            comparative_judgments.append(value.to_comparative_judgment())
        else:
            raise TypeError(
                "Unsupported supervision value "
                f"{type(value)!r}; expected ResponseJudgment, ComparativeJudgment, "
                "BinaryComparison, BinaryProjectionDataset, ComparativeDataset, or SupervisionDataset."
            )
    return SupervisionDataset(
        response_judgments=response_judgments,
        comparative_judgments=comparative_judgments,
    )


__all__ = [
    "BinaryComparison",
    "BinaryProjectionDataset",
    "BinaryProjectionMode",
    "ComparativeCandidate",
    "ComparativeDataset",
    "ComparativeJudgment",
    "ResponseJudgment",
    "SupervisionDataset",
    "coerce_supervision_dataset",
]
