"""Supervision collection utilities for StateTree runs.

This module provides a small, swap-friendly surface for collecting supervision
data from a `StateTree` run:

- run with *no* online oracle calls and emit "label requests" (unlabeled)
- optionally label sampled units immediately with a `ScoringOracle`

The output format is the canonical `SupervisionDataset` used by training code.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from treepo._research.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from treepo._research.core.protocols import format_merge_input
from treepo._research.core.scoring import ScoringOracle
from treepo._research.core.supervision_metadata import judgment_supervision_metadata
from treepo._research.training.supervision.types import ComparativeCandidate, ComparativeJudgment, ResponseJudgment, SupervisionDataset
from treepo._research.tree.state_tree import StateNode, StateTree


@dataclass(frozen=True)
class TreePOSupervisionSpec:
    """Configuration for sampling supervision units from a StateTree run."""

    mode: str = "off"  # "off" | "requests" | "label_now"
    labeler_kind: str = "oracle_score"  # "oracle_score" | "markov_toy_changepoints"
    supervision_kind: str = "scalar"  # "scalar" | "comparative"

    # Per-tree sampling: if < 1.0, only some trees emit supervision artifacts.
    doc_sample_probability: float = 0.0

    # Unit sampling within a selected tree.
    unit_selector: str = "all"  # "all" | "leaves" | "internal" | "root"
    max_units: int = 32

    # Deterministic sampling when set (recommended).
    random_seed: Optional[int] = 0

    # Unit sampling design (matches Auditor naming).
    sampling_strategy: str = "random"  # "random" | "level_weighted" | "content_weighted"
    unit_sampling_probability: float = 1.0  # second-stage independent gate

    # Content-weighted sampling configuration.
    content_weight_key: Optional[str] = None  # read node.metadata[key] if present
    content_weights: Optional[Dict[str, float]] = None  # node_id -> score
    content_weight_concentration: float = 2.0  # α exponent in score^α
    content_weight_propensity_floor: float = 0.01  # lower bound on inclusion probability

    # Output location for emitted SupervisionDataset JSON.
    output_dir: str = "outputs/treepo_supervision"

    # Label configuration (used for labeled datasets and for downstream consumers).
    response_signal_name: str = "preservation_similarity"
    response_signal_min: float = 0.0
    response_signal_max: float = 1.0

    truth_label_source: str = "oracle"

    # Comparative supervision: baseline candidate formatting.
    comparative_baseline_max_chars: int = 256

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TreePOSupervisionSpec":
        data = dict(payload or {})
        # DSL-friendly aliases.
        if "doc_sample_prob" in data and "doc_sample_probability" not in data:
            data["doc_sample_probability"] = data.pop("doc_sample_prob")
        if "sample_prob" in data and "unit_sampling_probability" not in data:
            data["unit_sampling_probability"] = data.pop("sample_prob")
        if "sampling_probability" in data and "unit_sampling_probability" not in data:
            data["unit_sampling_probability"] = data.pop("sampling_probability")
        if "unit_sample_prob" in data and "unit_sampling_probability" not in data:
            data["unit_sampling_probability"] = data.pop("unit_sample_prob")
        return cls(**data)

    def validate(self) -> None:
        mode = str(self.mode or "off").strip().lower()
        if mode not in {"off", "requests", "label_now"}:
            raise ValueError(f"Unsupported TreePOSupervisionSpec.mode={self.mode!r}.")
        labeler = str(self.labeler_kind or "oracle_score").strip().lower()
        if labeler not in {"oracle_score", "markov_toy_changepoints"}:
            raise ValueError(f"Unsupported TreePOSupervisionSpec.labeler_kind={self.labeler_kind!r}.")
        kind = str(self.supervision_kind or "scalar").strip().lower()
        if kind not in {"scalar", "comparative"}:
            raise ValueError(f"Unsupported TreePOSupervisionSpec.supervision_kind={self.supervision_kind!r}.")
        if kind == "comparative" and labeler != "oracle_score":
            raise ValueError(
                "TreePOSupervisionSpec.labeler_kind must be 'oracle_score' when supervision_kind='comparative'."
            )
        p = float(self.doc_sample_probability)
        if p < 0.0 or p > 1.0:
            raise ValueError("TreePOSupervisionSpec.doc_sample_probability must be in [0, 1].")
        if int(self.max_units) < 0:
            raise ValueError("TreePOSupervisionSpec.max_units must be non-negative.")
        strategy = str(self.sampling_strategy or "random").strip().lower()
        if strategy not in {"random", "level_weighted", "content_weighted"}:
            raise ValueError(f"Unsupported TreePOSupervisionSpec.sampling_strategy={self.sampling_strategy!r}.")
        gate = float(self.unit_sampling_probability)
        if gate < 0.0 or gate > 1.0:
            raise ValueError("TreePOSupervisionSpec.unit_sampling_probability must be in [0, 1].")
        floor = float(self.content_weight_propensity_floor)
        if floor < 0.0 or floor > 1.0:
            raise ValueError("TreePOSupervisionSpec.content_weight_propensity_floor must be in [0, 1].")
        concentration = float(self.content_weight_concentration)
        if concentration != concentration or concentration < 0.0:
            raise ValueError("TreePOSupervisionSpec.content_weight_concentration must be finite and non-negative.")
        if int(self.comparative_baseline_max_chars) <= 0:
            raise ValueError("TreePOSupervisionSpec.comparative_baseline_max_chars must be positive.")


@dataclass(frozen=True)
class TreePOLabelingPolicySpec:
    """Policy spec for labeling a saved `SupervisionDataset` later.

    This is a thin wrapper over `label_supervision_dataset(...)` so the same
    configuration can be used from TOML/JSON/DSL without threading through
    positional args.
    """

    policy_name: str = "treepo_supervision_labeler_v1"
    max_labels: Optional[int] = None
    label_probability: Optional[float] = None
    random_seed: Optional[int] = 0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TreePOLabelingPolicySpec":
        data = dict(payload or {})
        if "sample_prob" in data and "label_probability" not in data:
            data["label_probability"] = data.pop("sample_prob")
        if "sampling_probability" in data and "label_probability" not in data:
            data["label_probability"] = data.pop("sampling_probability")
        if "prob" in data and "label_probability" not in data:
            data["label_probability"] = data.pop("prob")
        return cls(**data)


def _compute_level_weights(nodes: Sequence[StateNode[Any, Any]]) -> List[float]:
    if not nodes:
        return []
    max_level = max(int(node.level) for node in nodes)
    raw = [(int(node.level) + 1) / float(max_level + 1) for node in nodes]
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(nodes)] * len(nodes)
    return [float(weight) / float(total) for weight in raw]


def _compute_content_weights(tree: StateTree[Any, Any], nodes: Sequence[StateNode[Any, Any]], spec: TreePOSupervisionSpec) -> List[float]:
    key = str(spec.content_weight_key) if spec.content_weight_key else None
    override = dict(spec.content_weights or {})
    tree_map = tree.metadata.get("content_weights")
    tree_weights = dict(tree_map) if isinstance(tree_map, Mapping) else {}
    alpha = float(spec.content_weight_concentration)

    weights: List[float] = []
    for node in nodes:
        score: Optional[float] = None
        if node.id in override:
            try:
                score = float(override[node.id])
            except (TypeError, ValueError):
                score = None
        if score is None and node.id in tree_weights:
            try:
                score = float(tree_weights[node.id])
            except (TypeError, ValueError):
                score = None
        if score is None and key is not None:
            try:
                raw = node.metadata.get(key)
                if raw is not None:
                    score = float(raw)
            except (TypeError, ValueError):
                score = None
        if score is None:
            score = 1.0
        safe = max(0.0, float(score))
        weights.append(float(safe**alpha) if alpha != 0.0 else 1.0)
    return weights


def _sample_nodes(
    tree: StateTree[Any, Any],
    nodes: Sequence[StateNode[Any, Any]],
    *,
    sample_size: int,
    spec: TreePOSupervisionSpec,
    rng: Any,
) -> List[Tuple[StateNode[Any, Any], float]]:
    if not nodes or sample_size <= 0:
        return []
    sample_size = min(int(sample_size), len(nodes))
    strategy = str(spec.sampling_strategy or "random").strip().lower()

    if strategy == "random":
        if len(nodes) > sample_size:
            sampled = list(rng.sample(list(nodes), k=sample_size))
        else:
            sampled = list(nodes)
        inclusion_prob = float(sample_size) / float(len(nodes)) if nodes else 0.0
        return [(node, float(inclusion_prob)) for node in sampled]

    from treepo._research.stats.sampling import pps_inclusion_probabilities, systematic_pps_sample_indices

    if strategy == "level_weighted":
        weights = _compute_level_weights(nodes)
        inclusion = pps_inclusion_probabilities(weights, sample_size)
        indices = systematic_pps_sample_indices(inclusion, sample_size, rng=rng)
        return [(nodes[index], float(inclusion[index])) for index in indices]

    if strategy == "content_weighted":
        weights = _compute_content_weights(tree, nodes, spec)
        inclusion = pps_inclusion_probabilities(weights, sample_size)
        floor = max(1e-12, float(spec.content_weight_propensity_floor))
        inclusion = [max(floor, float(p)) for p in inclusion]
        indices = systematic_pps_sample_indices(inclusion, sample_size, rng=rng)
        return [(nodes[index], float(inclusion[index])) for index in indices]

    raise ValueError(f"Unsupported sampling strategy: {strategy}")


def _stable_int_seed(*parts: str) -> int:
    payload = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _slug(value: str, *, fallback: str = "run", max_len: int = 64) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    text = text.strip("_")
    return text[:max_len] if text else fallback


def _select_nodes(tree: StateTree[Any, Any], selector: str) -> List[StateNode[Any, Any]]:
    mode = str(selector or "all").strip().lower()
    if mode == "root":
        return [tree.root]
    nodes = list(tree.traverse_preorder())
    if mode == "all":
        return nodes
    if mode == "leaves":
        return [node for node in nodes if node.is_leaf]
    if mode in {"internal", "internals"}:
        return [node for node in nodes if not node.is_leaf]
    raise ValueError(f"Unsupported TreePOSupervisionSpec.unit_selector={selector!r}.")


def _unit_kind_for_node(node: StateNode[Any, Any]) -> ObservationUnitKind:
    return ObservationUnitKind.LEAF if node.is_leaf else ObservationUnitKind.INTERNAL


def _law_type_for_node(node: StateNode[Any, Any]) -> str:
    return "sufficiency" if node.is_leaf else "merge_consistency"


def _span_text(span: Any) -> str:
    if span is None:
        return ""
    if isinstance(span, str):
        return span
    if isinstance(span, (list, tuple)) and all(isinstance(item, str) for item in span):
        return " ".join(str(item) for item in span)
    return str(span)


def _span_tokens(span: Any) -> List[str]:
    if span is None:
        return []
    if isinstance(span, (list, tuple)) and all(isinstance(item, str) for item in span):
        return [str(item) for item in span]
    if isinstance(span, str):
        return [token for token in str(span).split() if token]
    return []


def should_collect_supervision(
    spec: TreePOSupervisionSpec,
    *,
    document_id: Optional[str] = None,
) -> bool:
    spec.validate()
    mode = str(spec.mode or "off").strip().lower()
    if mode == "off":
        return False
    p = float(spec.doc_sample_probability)
    if p >= 1.0:
        return True
    if p <= 0.0:
        return False
    seed = spec.random_seed
    # If no seed is provided, fall back to wall-clock randomness.
    if seed is None:
        import random

        return random.random() < p
    doc_key = str(document_id or "")
    derived = _stable_int_seed(str(seed), doc_key, "treepo_supervision")
    import random

    rng = random.Random(derived)
    return rng.random() < p


def build_supervision_dataset_from_state_tree(
    tree: StateTree[Any, Any],
    *,
    rubric: str,
    spec: TreePOSupervisionSpec,
    document_id: Optional[str] = None,
    oracle: Optional[ScoringOracle] = None,
) -> SupervisionDataset:
    """Sample nodes from a tree and build a SupervisionDataset.

    When `spec.mode == "requests"`, the dataset contains unlabeled ResponseJudgment
    rows (response_signal_value=None).

    When `spec.mode == "label_now"`, `oracle` is required and each row is labeled
    with `oracle.score(original_text, response, rubric).score`.
    """
    spec.validate()
    supervision_kind = str(spec.supervision_kind or "scalar").strip().lower()
    mode = str(spec.mode or "off").strip().lower()
    labeler = str(spec.labeler_kind or "oracle_score").strip().lower()
    if mode == "off":
        return SupervisionDataset()
    if mode == "label_now" and labeler == "oracle_score" and oracle is None:
        raise ValueError(
            "build_supervision_dataset_from_state_tree requires oracle when "
            "mode='label_now' and labeler_kind='oracle_score'."
        )

    import random

    doc_key = str(document_id or tree.metadata.get("document_id") or tree.root.id)
    candidates = _select_nodes(tree, spec.unit_selector)
    if not candidates:
        return SupervisionDataset()

    max_units = int(spec.max_units)
    if max_units <= 0:
        return SupervisionDataset()

    # Deterministic unit sampling.
    if spec.random_seed is None:
        rng = random.Random()
    else:
        rng = random.Random(_stable_int_seed(str(spec.random_seed), doc_key, "treepo_units"))
    base_sample_size = min(max_units, len(candidates))
    sampled_with_probs = _sample_nodes(
        tree,
        candidates,
        sample_size=base_sample_size,
        spec=spec,
        rng=rng,
    )

    gate = float(spec.unit_sampling_probability)
    if gate <= 0.0:
        return SupervisionDataset()
    if gate < 1.0:
        gated: List[Tuple[StateNode[Any, Any], float]] = []
        for node, prob in sampled_with_probs:
            if rng.random() < gate:
                gated.append((node, min(1.0, float(prob) * gate)))
        sampled_with_probs = gated
    else:
        sampled_with_probs = [(node, min(1.0, float(prob))) for node, prob in sampled_with_probs]

    if not sampled_with_probs:
        return SupervisionDataset()

    if supervision_kind == "comparative":
        return _build_comparative_supervision_dataset_from_state_tree(
            tree,
            rubric=str(rubric or ""),
            spec=spec,
            document_id=document_id,
            oracle=oracle,
            sampled_with_probs=sampled_with_probs,
        )

    judgments: List[ResponseJudgment] = []
    # Even "requests" (unlabeled) support IPW estimation once labels are attached,
    # because we log document/unit propensities at collection time.
    mode_supports_ipw = True
    doc_propensity = float(spec.doc_sample_probability)
    if doc_propensity <= 0.0:
        doc_propensity = 1.0

    sampling_scheme = f"treepo_supervision_{str(spec.sampling_strategy or 'random').strip().lower()}"
    for node, unit_propensity in sampled_with_probs:
        unit_kind = _unit_kind_for_node(node)
        law_type = _law_type_for_node(node)
        original_text = _span_text(node.span)
        response_text = str(node.rendered or "")

        label_value: Optional[float] = None
        markov_sketch: Optional[Dict[str, Any]] = None
        if mode == "label_now":
            if labeler == "oracle_score":
                label_value = float(
                    oracle.score(original_text, response_text, str(rubric or "")).score  # type: ignore[union-attr]
                )
            elif labeler == "markov_toy_changepoints":
                from treepo._research.diffusion.markov_toy import encode_markov_path

                tokens = _span_tokens(node.span)
                sketch = encode_markov_path(tokens)
                label_value = float(sketch.changepoints)
                markov_sketch = {
                    "changepoints": int(sketch.changepoints),
                    "start_state": sketch.start_state,
                    "end_state": sketch.end_state,
                    "length": int(sketch.length),
                }

        sampling = SamplingMetadata(
            document_propensity=float(doc_propensity),
            unit_propensity=float(unit_propensity),
            label_propensity=1.0,
            sampling_scheme=sampling_scheme,
            policy_name="treepo_supervision_sampler_v2",
            unit_kind=unit_kind,
            supports_ipw_estimation=bool(mode_supports_ipw),
            metadata={
                "unit_selector": str(spec.unit_selector),
                "max_units": int(max_units),
                "candidate_count": int(len(candidates)),
                "unit_sampling_probability": float(spec.unit_sampling_probability),
                "label_cumulative_probability": 1.0 if mode == "label_now" else 0.0,
                "label_rounds": 0,
                "label_policy_family": "online_oracle" if mode == "label_now" else "unlabeled_requests",
            },
        )
        supervision_meta = judgment_supervision_metadata(
            law_type=str(law_type),
            response_signal_name=str(spec.response_signal_name),
            response_signal_min=float(spec.response_signal_min),
            response_signal_max=float(spec.response_signal_max),
            metadata={
                "document_id": doc_key,
                "node_id": str(node.id),
                "node_level": int(node.level),
                "unit_kind": unit_kind.value,
                "tree_mode": str(tree.metadata.get("mode", "") or ""),
            },
        )
        judgments.append(
            ResponseJudgment(
                judgment_id=uuid.uuid4().hex,
                source_example_id=str(doc_key),
                original_text=original_text,
                rubric=str(rubric or ""),
                response=response_text,
                response_id=str(node.id),
                reference_score=0.0,
                law_type=str(law_type),
                source_doc_id=str(doc_key),
                truth_label_source=str(spec.truth_label_source),
                sampling=sampling,
                supervision_metadata=supervision_meta,
                source_observation_ids=[],
                response_signal_value=label_value,
                metadata={
                    "state_tree_node_id": str(node.id),
                    "state_tree_node_level": int(node.level),
                    "state_tree_unit_kind": unit_kind.value,
                    "treepo_labeler_kind": str(labeler),
                    "markov_toy_sketch": markov_sketch,
                    "tree_metadata": dict(tree.metadata or {}),
                },
            )
        )

    return SupervisionDataset(response_judgments=judgments)


def _comparative_baseline_for_node(
    node: StateNode[Any, Any],
    *,
    original_text: str,
    max_chars: int,
) -> str:
    if not node.is_leaf and node.left_child is not None and node.right_child is not None:
        left = str(node.left_child.rendered or "")
        right = str(node.right_child.rendered or "")
        return format_merge_input(left, right)
    return str(original_text or "")[: max(1, int(max_chars))]


def _build_comparative_supervision_dataset_from_state_tree(
    tree: StateTree[Any, Any],
    *,
    rubric: str,
    spec: TreePOSupervisionSpec,
    document_id: Optional[str],
    oracle: Optional[ScoringOracle],
    sampled_with_probs: Sequence[Tuple[StateNode[Any, Any], float]],
) -> SupervisionDataset:
    mode = str(spec.mode or "off").strip().lower()
    if mode == "label_now" and oracle is None:
        raise ValueError("Comparative supervision requires oracle when mode='label_now'.")

    doc_key = str(document_id or tree.metadata.get("document_id") or tree.root.id)
    doc_propensity = float(spec.doc_sample_probability)
    if doc_propensity <= 0.0:
        doc_propensity = 1.0

    from treepo._research.core.supervision_metadata import judgment_supervision_metadata

    records: List[ComparativeJudgment] = []
    sampling_scheme = f"treepo_supervision_{str(spec.sampling_strategy or 'random').strip().lower()}"
    for node, unit_propensity in sampled_with_probs:
        unit_kind = _unit_kind_for_node(node)
        law_type = _law_type_for_node(node)
        original_text = _span_text(node.span)
        candidate_a = str(node.rendered or "")
        candidate_b = _comparative_baseline_for_node(
            node,
            original_text=original_text,
            max_chars=int(spec.comparative_baseline_max_chars),
        )

        candidates = [
            ComparativeCandidate(candidate_id=f"{node.id}:A", response=str(candidate_a), metadata={"kind": "tree_output"}),
            ComparativeCandidate(candidate_id=f"{node.id}:B", response=str(candidate_b), metadata={"kind": "baseline"}),
        ]

        if mode == "label_now":
            scored = [
                (idx, float(oracle.score(original_text, cand.response, str(rubric or "")).score))  # type: ignore[union-attr]
                for idx, cand in enumerate(candidates)
            ]
            for idx, score in scored:
                candidates[idx].response_signal_value = float(score)
            scored_sorted = sorted(scored, key=lambda row: row[1], reverse=True)
            for rank, (idx, _) in enumerate(scored_sorted, start=1):
                candidates[idx].rank = int(rank)

        sampling = SamplingMetadata(
            document_propensity=float(doc_propensity),
            unit_propensity=float(unit_propensity),
            label_propensity=1.0,
            sampling_scheme=sampling_scheme,
            policy_name="treepo_supervision_sampler_v2",
            unit_kind=unit_kind,
            supports_ipw_estimation=True,
            metadata={
                "unit_selector": str(spec.unit_selector),
                "max_units": int(spec.max_units),
                "unit_sampling_probability": float(spec.unit_sampling_probability),
                "label_cumulative_probability": 1.0 if mode == "label_now" else 0.0,
                "label_rounds": 0,
                "label_policy_family": "online_oracle" if mode == "label_now" else "unlabeled_requests",
                "supervision_kind": "comparative",
            },
        )

        preference_meta = judgment_supervision_metadata(
            law_type=str(law_type),
            response_signal_name=str(spec.response_signal_name),
            response_signal_min=float(spec.response_signal_min),
            response_signal_max=float(spec.response_signal_max),
            preference_family="groupwise",
            metadata={
                "document_id": doc_key,
                "node_id": str(node.id),
                "node_level": int(node.level),
                "unit_kind": unit_kind.value,
                "tree_mode": str(tree.metadata.get("mode", "") or ""),
            },
        )

        records.append(
            ComparativeJudgment(
                record_id=uuid.uuid4().hex,
                source_example_id=str(doc_key),
                original_text=str(original_text),
                rubric=str(rubric or ""),
                reference_score=0.0,
                law_type=str(law_type),
                candidates=candidates,
                sampling=sampling,
                preference_supervision=preference_meta,
                source_doc_id=str(doc_key),
                truth_label_source=str(spec.truth_label_source),
                metadata={
                    "state_tree_node_id": str(node.id),
                    "state_tree_node_level": int(node.level),
                    "state_tree_unit_kind": unit_kind.value,
                    "treepo_labeler_kind": "oracle_score" if mode == "label_now" else "requests",
                    "tree_metadata": dict(tree.metadata or {}),
                },
            )
        )

    return SupervisionDataset(comparative_judgments=records)


def persist_supervision_dataset(
    dataset: SupervisionDataset,
    *,
    spec: TreePOSupervisionSpec,
    document_id: Optional[str] = None,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Persist a dataset to disk and return the path."""
    out_dir = Path(str(spec.output_dir or "outputs/treepo_supervision"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save(path)
        return path

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    slug = _slug(str(document_id or ""), fallback="run")
    path = out_dir / f"supervision_{slug}_{stamp}_{uuid.uuid4().hex}.json"
    dataset.save(path)
    return path


def label_supervision_dataset(
    dataset: SupervisionDataset,
    *,
    oracle: ScoringOracle,
    rubric_fallback: str = "",
    max_labels: Optional[int] = None,
    label_probability: Optional[float] = None,
    random_seed: Optional[int] = None,
    policy_name: str = "treepo_supervision_labeler_v1",
) -> SupervisionDataset:
    """Fill missing scalar labels by calling a scoring oracle.

    This supports both:
    - full labeling (`label_probability=None` and `max_labels=None`)
    - partial labeling with a known label propensity, either via:
      - Bernoulli labeling (`label_probability=p`), or
      - fixed-size labeling (`max_labels=k`) with propensity `k / N_unlabeled`.

    The label propensity is recorded into `ResponseJudgment.sampling.label_propensity`
    for newly labeled rows, enabling correct IPW weights downstream.
    """
    if max_labels is not None and int(max_labels) < 0:
        raise ValueError("max_labels must be non-negative.")

    if label_probability is not None:
        p = float(label_probability)
        if p <= 0.0 or p > 1.0:
            raise ValueError("label_probability must be in (0, 1].")
    else:
        p = 1.0

    if max_labels is not None and label_probability is not None:
        raise ValueError("Provide at most one of max_labels or label_probability.")

    import random

    if dataset.response_judgments and dataset.comparative_judgments:
        raise ValueError(
            "label_supervision_dataset expects a dataset with either response_judgments or comparative_judgments, not both."
        )

    rng = random.Random(int(random_seed)) if random_seed is not None else random.Random()

    rows = list(dataset.response_judgments)
    records = list(dataset.comparative_judgments)
    if rows:
        total_units = len(rows)
        if total_units <= 0:
            return dataset

        candidates = [row for row in rows if row.response_signal_value is None]
        if not candidates:
            return dataset
    else:
        total_units = len(records)
        if total_units <= 0:
            return dataset

        def is_unlabeled(record: ComparativeJudgment) -> bool:
            return all(getattr(candidate, "rank", None) is None for candidate in list(record.candidates or []))

        candidates = [record for record in records if is_unlabeled(record)]
        if not candidates:
            return dataset

    selected: List[Any] = []
    policy_family = "full"
    if max_labels is not None:
        policy_family = "srswor_budget"
    elif label_probability is not None:
        policy_family = "bernoulli"

    prior_family: Optional[str] = None
    prior_cumulative: Optional[float] = None
    prior_rounds: int = 0
    if rows:
        for row in rows:
            meta = getattr(row.sampling, "metadata", {}) or {}
            candidate_family = meta.get("label_policy_family")
            if candidate_family:
                prior_family = str(candidate_family)
            candidate_cum = meta.get("label_cumulative_probability")
            if candidate_cum is not None:
                try:
                    prior_cumulative = float(candidate_cum)
                except (TypeError, ValueError):
                    prior_cumulative = None
            candidate_rounds = meta.get("label_rounds")
            if candidate_rounds is not None:
                try:
                    prior_rounds = max(prior_rounds, int(candidate_rounds))
                except (TypeError, ValueError):
                    pass
            if prior_family is not None and prior_cumulative is not None:
                break
    else:
        for record in records:
            meta = getattr(record.sampling, "metadata", {}) or {}
            candidate_family = meta.get("label_policy_family")
            if candidate_family:
                prior_family = str(candidate_family)
            candidate_cum = meta.get("label_cumulative_probability")
            if candidate_cum is not None:
                try:
                    prior_cumulative = float(candidate_cum)
                except (TypeError, ValueError):
                    prior_cumulative = None
            candidate_rounds = meta.get("label_rounds")
            if candidate_rounds is not None:
                try:
                    prior_rounds = max(prior_rounds, int(candidate_rounds))
                except (TypeError, ValueError):
                    pass
            if prior_family is not None and prior_cumulative is not None:
                break

    if prior_cumulative is None:
        prior_cumulative = 1.0 if not candidates else 0.0

    if prior_family and prior_family not in {"unlabeled_requests", "online_oracle", policy_family}:
        raise ValueError(
            f"label_supervision_dataset cannot mix label policies: prior={prior_family!r}, requested={policy_family!r}."
        )

    if max_labels is not None:
        budget = min(int(max_labels), len(candidates))
        if budget <= 0:
            return dataset
        selected = rng.sample(list(candidates), k=budget) if len(candidates) > budget else list(candidates)
    elif label_probability is not None and float(p) < 1.0:
        selected = [item for item in candidates if rng.random() < float(p)]
        if not selected:
            return dataset
    else:
        selected = list(candidates)

    if rows:
        selected_ids = {str(row.judgment_id) for row in selected}  # type: ignore[arg-type]
        for row in rows:
            if str(row.judgment_id) not in selected_ids:
                continue
            rubric = str(row.rubric or rubric_fallback or "")
            score = oracle.score(str(row.original_text or ""), str(row.response or ""), rubric).score
            row.response_signal_value = float(score)

        labeled_after = sum(1 for row in rows if row.response_signal_value is not None)
        if labeled_after <= 0:
            return dataset
    else:
        selected_ids = {str(record.record_id) for record in selected}  # type: ignore[arg-type]
        for record in records:
            if str(record.record_id) not in selected_ids:
                continue
            rubric = str(record.rubric or rubric_fallback or "")
            scored = []
            for idx, candidate in enumerate(list(record.candidates or [])):
                score = oracle.score(str(record.original_text or ""), str(candidate.response or ""), rubric).score
                record.candidates[idx].response_signal_value = float(score)
                scored.append((idx, float(score)))
            scored_sorted = sorted(scored, key=lambda row: row[1], reverse=True)
            for rank, (idx, _) in enumerate(scored_sorted, start=1):
                record.candidates[idx].rank = int(rank)

        labeled_after = sum(
            1
            for record in records
            if all(getattr(candidate, "rank", None) is not None for candidate in list(record.candidates or []))
        )
        if labeled_after <= 0:
            return dataset

    if policy_family == "full":
        cumulative = 1.0
    elif policy_family == "srswor_budget":
        cumulative = float(labeled_after) / float(total_units)
    else:
        cumulative = 1.0 - (1.0 - float(prior_cumulative)) * (1.0 - float(p))

    round_index = int(prior_rounds) + 1
    round_propensity = None
    if policy_family == "srswor_budget":
        round_propensity = float(min(int(max_labels or 0), len(candidates))) / float(len(candidates)) if candidates else None
    elif policy_family == "bernoulli":
        round_propensity = float(p)

    if rows:
        updated_rows: List[ResponseJudgment] = []
        for row in rows:
            try:
                row.sampling = row.sampling.with_updates(  # type: ignore[assignment]
                    label_propensity=float(cumulative),
                    supports_ipw_estimation=True,
                    metadata={
                        **dict(row.sampling.metadata or {}),
                        "label_policy_family": str(policy_family),
                        "label_policy_name": str(policy_name),
                        "label_round_index": int(round_index),
                        "label_round_propensity": float(round_propensity) if round_propensity is not None else None,
                        "label_random_seed": int(random_seed) if random_seed is not None else None,
                        "label_max_labels": int(max_labels) if max_labels is not None else None,
                        "label_probability": float(label_probability) if label_probability is not None else None,
                        "label_rounds": int(round_index),
                        "label_cumulative_probability": float(cumulative),
                        "label_cumulative_labeled_count": int(labeled_after),
                        "label_cumulative_total_count": int(total_units),
                    },
                )
            except Exception:
                pass
            updated_rows.append(row)
        dataset.response_judgments = updated_rows
        return dataset

    updated_records: List[ComparativeJudgment] = []
    for record in records:
        try:
            record.sampling = record.sampling.with_updates(  # type: ignore[assignment]
                label_propensity=float(cumulative),
                supports_ipw_estimation=True,
                metadata={
                    **dict(record.sampling.metadata or {}),
                    "label_policy_family": str(policy_family),
                    "label_policy_name": str(policy_name),
                    "label_round_index": int(round_index),
                    "label_round_propensity": float(round_propensity) if round_propensity is not None else None,
                    "label_random_seed": int(random_seed) if random_seed is not None else None,
                    "label_max_labels": int(max_labels) if max_labels is not None else None,
                    "label_probability": float(label_probability) if label_probability is not None else None,
                    "label_rounds": int(round_index),
                    "label_cumulative_probability": float(cumulative),
                    "label_cumulative_labeled_count": int(labeled_after),
                    "label_cumulative_total_count": int(total_units),
                },
            )
        except Exception:
            pass
        updated_records.append(record)

    dataset.comparative_judgments = updated_records
    return dataset


def label_supervision_dataset_with_policy(
    dataset: SupervisionDataset,
    *,
    oracle: ScoringOracle,
    policy: TreePOLabelingPolicySpec,
    rubric_fallback: str = "",
) -> SupervisionDataset:
    """Apply a labeling policy spec to an existing supervision dataset."""
    return label_supervision_dataset(
        dataset,
        oracle=oracle,
        rubric_fallback=str(rubric_fallback or ""),
        max_labels=policy.max_labels,
        label_probability=policy.label_probability,
        random_seed=policy.random_seed,
        policy_name=str(policy.policy_name or "treepo_supervision_labeler_v1"),
    )


__all__ = [
    "TreePOSupervisionSpec",
    "TreePOLabelingPolicySpec",
    "build_supervision_dataset_from_state_tree",
    "label_supervision_dataset",
    "label_supervision_dataset_with_policy",
    "persist_supervision_dataset",
    "should_collect_supervision",
]
