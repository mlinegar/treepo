"""Auditing utilities for text `StateTree` runs.

This module mirrors the behavior of `src/tree/auditor.py` but operates directly
on `StateTree[str, str]` instead of converting to the legacy `Tree` / `Node`
data model.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from treepo._research.core.protocols import format_merge_input
from treepo._research.core.scoring import ScoringOracle
from treepo._research.stats.sampling import pps_inclusion_probabilities, systematic_pps_sample_indices
from treepo._research.tree.auditor import AuditCheckResult, AuditConfig, AuditReport, SampledUnit, SamplingStrategy
from treepo._research.tree.state_tree import StateNode, StateTree

logger = logging.getLogger(__name__)


def _span_text(span: Any) -> str:
    if span is None:
        return ""
    return str(span)


def _traverse_inorder(node: StateNode[str, str]) -> List[StateNode[str, str]]:
    out: List[StateNode[str, str]] = []
    if node.left_child is not None:
        out.extend(_traverse_inorder(node.left_child))
    out.append(node)
    if node.right_child is not None:
        out.extend(_traverse_inorder(node.right_child))
    return out


@dataclass
class StateTreeAuditor:
    """Legacy-auditor-compatible checker that operates on `StateTree[str, str]`."""

    oracle: ScoringOracle
    config: AuditConfig
    summarizer: Optional[Callable[[str, str], str]] = None
    theorem_operator: Optional[Any] = None

    def __post_init__(self) -> None:
        # Use a per-auditor RNG to avoid clobbering the global random state.
        self._rng = random.Random(self.config.random_seed) if self.config.random_seed is not None else random.Random()
        self._last_inclusion_prob_map: Dict[str, float] = {}

    def _call_oracle(self, input_a: str, input_b: str, rubric: str) -> Tuple[bool, float, str]:
        result = self.oracle.score(input_a, input_b, rubric)
        discrepancy = 1.0 - float(result.score)
        is_congruent = discrepancy <= float(self.config.discrepancy_threshold)
        return bool(is_congruent), float(discrepancy), str(result.reasoning or "")

    def _combine_inputs(self, left: str, right: str, rubric: str) -> str:
        operator = self.theorem_operator
        if operator is not None and hasattr(operator, "combine"):
            return str(operator.combine(left, right, rubric=rubric))
        return format_merge_input(left, right)

    def _resummarize(self, text: str, rubric: str) -> str:
        operator = self.theorem_operator
        if operator is not None:
            if hasattr(operator, "resummarize"):
                return str(operator.resummarize(text, rubric=rubric))
            if hasattr(operator, "encode") and hasattr(operator, "decode"):
                encoded = operator.encode(text, rubric=rubric)
                return str(operator.decode(encoded, rubric=rubric))
        if self.summarizer is None:
            raise RuntimeError("No summarizer or theorem operator configured")
        return str(self.summarizer(text, rubric))

    def _apply_probability_gate(self, sampled_units: List[SampledUnit]) -> List[SampledUnit]:
        if not sampled_units:
            return []

        gate_prob = max(0.0, min(1.0, float(self.config.sampling_probability)))
        if gate_prob >= 1.0:
            return sampled_units

        gated: List[SampledUnit] = []
        for sampled in sampled_units:
            if self._rng.random() < gate_prob:
                gated.append(
                    SampledUnit(
                        item=sampled.item,
                        inclusion_probability=min(1.0, float(sampled.inclusion_probability) * gate_prob),
                    )
                )
        return gated

    @staticmethod
    def _compute_level_weights(nodes: Sequence[StateNode[str, str]]) -> List[float]:
        if not nodes:
            return []
        max_level = max(int(node.level) for node in nodes)
        raw = [(int(node.level) + 1) / float(max_level + 1) for node in nodes]
        total = sum(raw)
        if total <= 0:
            return [1.0 / len(nodes)] * len(nodes)
        return [float(weight) / float(total) for weight in raw]

    def _compute_content_weights(self, nodes: Sequence[StateNode[str, str]]) -> List[float]:
        if not nodes:
            return []

        alpha = max(0.0, float(self.config.content_weight_concentration))
        raw: List[float] = []
        for node in nodes:
            if self.config.content_weights:
                score = self.config.content_weights.get(node.id, 0.5)
            else:
                score = 0.5
            raw.append(max(1e-8, float(score) ** alpha))

        total = sum(raw)
        if total <= 0:
            return [1.0 / len(nodes)] * len(nodes)
        return [w / total for w in raw]

    def _sample_nodes(self, nodes: Sequence[StateNode[str, str]], budget: int) -> List[SampledUnit]:
        if not nodes or budget <= 0:
            return []

        nodes_list = list(nodes)
        budget = min(int(budget), len(nodes_list))

        if self.config.sampling_strategy == SamplingStrategy.RANDOM:
            sampled = self._rng.sample(nodes_list, budget)
            inclusion_prob = (budget / len(nodes_list)) if nodes_list else 0.0
            self._last_inclusion_prob_map.update({n.id: float(inclusion_prob) for n in nodes_list})
            return [SampledUnit(item=node, inclusion_probability=float(inclusion_prob)) for node in sampled]

        if self.config.sampling_strategy == SamplingStrategy.LEVEL_WEIGHTED:
            weights = self._compute_level_weights(nodes_list)
            inclusion_probs = pps_inclusion_probabilities(weights, budget)
            self._last_inclusion_prob_map.update({nodes_list[i].id: float(inclusion_probs[i]) for i in range(len(nodes_list))})
            sampled_indices = systematic_pps_sample_indices(inclusion_probs, budget, rng=self._rng)
            return [
                SampledUnit(item=nodes_list[index], inclusion_probability=float(inclusion_probs[index]))
                for index in sampled_indices
            ]

        if self.config.sampling_strategy == SamplingStrategy.CONTENT_WEIGHTED:
            weights = self._compute_content_weights(nodes_list)
            inclusion_probs = pps_inclusion_probabilities(weights, budget)
            floor = max(1e-6, float(self.config.content_weight_propensity_floor))
            inclusion_probs = [max(floor, float(p)) for p in inclusion_probs]
            self._last_inclusion_prob_map.update({nodes_list[i].id: float(inclusion_probs[i]) for i in range(len(nodes_list))})
            sampled_indices = systematic_pps_sample_indices(inclusion_probs, budget, rng=self._rng)
            return [
                SampledUnit(item=nodes_list[index], inclusion_probability=float(inclusion_probs[index]))
                for index in sampled_indices
            ]

        sampled = self._rng.sample(nodes_list, budget) if len(nodes_list) > budget else list(nodes_list)
        inclusion_prob = (budget / len(nodes_list)) if nodes_list else 0.0
        self._last_inclusion_prob_map.update({n.id: float(inclusion_prob) for n in nodes_list})
        return [SampledUnit(item=node, inclusion_probability=float(inclusion_prob)) for node in sampled]

    def _sample_adjacent_pairs(
        self,
        adjacent_pairs: Sequence[Tuple[StateNode[str, str], StateNode[str, str]]],
        budget: int,
    ) -> List[SampledUnit]:
        if not adjacent_pairs or budget <= 0:
            return []
        pairs = list(adjacent_pairs)
        sample_size = min(int(budget), len(pairs))
        sampled = self._rng.sample(pairs, sample_size)
        inclusion_prob = (sample_size / len(pairs)) if pairs else 0.0
        return [SampledUnit(item=pair, inclusion_probability=float(inclusion_prob)) for pair in sampled]

    def _check_sufficiency(self, node: StateNode[str, str], rubric: str) -> Tuple[AuditCheckResult, str, str]:
        input_a = _span_text(node.span)
        input_b = str(node.rendered or "")
        is_congruent, score, reasoning = self._call_oracle(input_a, input_b, rubric)
        passed = bool(is_congruent) and float(score) <= float(self.config.discrepancy_threshold)
        result = AuditCheckResult(
            node_id=str(node.id),
            check_type="sufficiency",
            passed=bool(passed),
            discrepancy_score=float(score),
            reasoning=str(reasoning),
            input_a=input_a,
            input_b=input_b,
        )
        return result, input_a, input_b

    def _check_merge_consistency(self, node: StateNode[str, str], rubric: str) -> Tuple[AuditCheckResult, str, str]:
        if node.span is not None:
            input_a = _span_text(node.span)
        else:
            left_span = _span_text(node.left_child.span) if node.left_child is not None else ""
            right_span = _span_text(node.right_child.span) if node.right_child is not None else ""
            if not left_span and node.left_child is not None:
                left_span = str(node.left_child.rendered or "")
            if not right_span and node.right_child is not None:
                right_span = str(node.right_child.rendered or "")
            input_a = self._combine_inputs(left_span, right_span, rubric)
        input_b = str(node.rendered or "")
        is_congruent, score, reasoning = self._call_oracle(input_a, input_b, rubric)
        passed = bool(is_congruent) and float(score) <= float(self.config.discrepancy_threshold)
        result = AuditCheckResult(
            node_id=str(node.id),
            check_type="merge_consistency",
            passed=bool(passed),
            discrepancy_score=float(score),
            reasoning=str(reasoning),
            input_a=input_a,
            input_b=input_b,
        )
        return result, input_a, input_b

    def _check_idempotence(self, node: StateNode[str, str], rubric: str) -> AuditCheckResult:
        if self.summarizer is None and self.theorem_operator is None:
            return AuditCheckResult(
                node_id=str(node.id),
                check_type="idempotence",
                passed=False,
                discrepancy_score=0.0,
                reasoning="Skipped: no summarizer or theorem operator configured",
                skipped=True,
                skip_reason="no_summarizer",
            )

        original_summary = str(node.rendered or "")
        try:
            re_summarized = self._resummarize(original_summary, rubric)
        except Exception as exc:
            return AuditCheckResult(
                node_id=str(node.id),
                check_type="idempotence",
                passed=False,
                discrepancy_score=1.0,
                reasoning=f"Summarizer error: {exc}",
            )

        is_congruent, score, reasoning = self._call_oracle(original_summary, re_summarized, rubric)
        passed = bool(is_congruent) and float(score) <= float(self.config.discrepancy_threshold)
        return AuditCheckResult(
            node_id=str(node.id),
            check_type="idempotence",
            passed=bool(passed),
            discrepancy_score=float(score),
            reasoning=f"Idempotence: {reasoning}",
            input_a=original_summary,
            input_b=str(re_summarized),
        )

    def _check_substitution(
        self,
        left_node: StateNode[str, str],
        right_node: StateNode[str, str],
        rubric: str,
    ) -> AuditCheckResult:
        if self.summarizer is None and self.theorem_operator is None:
            return AuditCheckResult(
                node_id=f"{left_node.id}+{right_node.id}",
                check_type="substitution",
                passed=False,
                discrepancy_score=0.0,
                reasoning="Skipped: no summarizer or theorem operator configured",
                skipped=True,
                skip_reason="no_summarizer",
            )

        raw_left = _span_text(left_node.span)
        raw_right = _span_text(right_node.span)

        joint_raw = self._combine_inputs(raw_left, raw_right, rubric)
        try:
            joint_summary = self._resummarize(joint_raw, rubric)
        except Exception as exc:
            return AuditCheckResult(
                node_id=f"{left_node.id}+{right_node.id}",
                check_type="substitution",
                passed=False,
                discrepancy_score=1.0,
                reasoning=f"Joint summarizer error: {exc}",
            )

        left_summary = str(left_node.rendered or "") or self._resummarize(raw_left, rubric)
        right_summary = str(right_node.rendered or "") or self._resummarize(raw_right, rubric)

        concat_summaries = self._combine_inputs(left_summary, right_summary, rubric)
        try:
            disjoint_summary = self._resummarize(concat_summaries, rubric)
        except Exception as exc:
            return AuditCheckResult(
                node_id=f"{left_node.id}+{right_node.id}",
                check_type="substitution",
                passed=False,
                discrepancy_score=1.0,
                reasoning=f"Disjoint summarizer error: {exc}",
            )

        is_congruent, score, reasoning = self._call_oracle(str(joint_summary), str(disjoint_summary), rubric)
        passed = bool(is_congruent) and float(score) <= float(self.config.discrepancy_threshold)
        return AuditCheckResult(
            node_id=f"{left_node.id}+{right_node.id}",
            check_type="substitution",
            passed=bool(passed),
            discrepancy_score=float(score),
            reasoning=f"Substitution (joint vs disjoint): {reasoning}",
            input_a=str(joint_summary),
            input_b=str(disjoint_summary),
        )

    @staticmethod
    def _get_adjacent_leaf_pairs(
        leaves: Sequence[StateNode[str, str]],
    ) -> List[Tuple[StateNode[str, str], StateNode[str, str]]]:
        if len(leaves) < 2:
            return []
        return [(leaves[i], leaves[i + 1]) for i in range(len(leaves) - 1)]

    def audit_tree(self, tree: StateTree[str, str], *, rubric: str = "") -> AuditReport:
        self._last_inclusion_prob_map = {}

        tree_id = str(tree.root.id) if tree.root is not None else "unknown"
        source_doc_id = None
        if isinstance(tree.metadata, dict):
            raw_doc_id = tree.metadata.get("document_id", None)
            if raw_doc_id is None:
                raw_doc_id = tree.metadata.get("doc_id", None)
            if raw_doc_id is not None:
                source_doc_id = str(raw_doc_id)

        all_nodes = list(tree.traverse_preorder())
        leaves = [n for n in all_nodes if n.is_leaf]
        internal = [n for n in all_nodes if not n.is_leaf]
        substitution_population = max(0, len(leaves) - 1)

        checks: List[AuditCheckResult] = []
        sufficiency_violations = 0
        sufficiency_samples = 0
        merge_violations = 0
        merge_samples = 0
        idempotence_violations = 0
        idempotence_samples = 0
        substitution_violations = 0
        substitution_samples = 0

        effective_budget = int(self.config.compute_sample_budget_for_guarantee())
        leaf_budget = effective_budget // 2 if bool(self.config.audit_internal) else effective_budget
        internal_budget = effective_budget - leaf_budget

        if bool(self.config.audit_leaves) and leaves:
            leaf_samples = self._apply_probability_gate(self._sample_nodes(leaves, leaf_budget))
            for sampled in leaf_samples:
                result, _, _ = self._check_sufficiency(sampled.item, rubric)
                result.inclusion_probability = float(sampled.inclusion_probability)
                result.sampling_design = str(self.config.sampling_strategy.value)
                checks.append(result)
                sufficiency_samples += 1
                if not result.passed:
                    sufficiency_violations += 1

        if bool(self.config.audit_internal) and internal:
            internal_samples = self._apply_probability_gate(self._sample_nodes(internal, internal_budget))
            for sampled in internal_samples:
                result, _, _ = self._check_merge_consistency(sampled.item, rubric)
                result.inclusion_probability = float(sampled.inclusion_probability)
                result.sampling_design = str(self.config.sampling_strategy.value)
                checks.append(result)
                merge_samples += 1
                if not result.passed:
                    merge_violations += 1

        if bool(self.config.audit_idempotence) and internal and (self.summarizer is not None or self.theorem_operator is not None):
            idem_samples = self._apply_probability_gate(self._sample_nodes(internal, int(self.config.idempotence_budget)))
            for sampled in idem_samples:
                result = self._check_idempotence(sampled.item, rubric)
                result.inclusion_probability = float(sampled.inclusion_probability)
                result.sampling_design = str(self.config.sampling_strategy.value)
                checks.append(result)
                idempotence_samples += 1
                if not result.passed:
                    idempotence_violations += 1
        elif bool(self.config.audit_idempotence) and internal and self.summarizer is None and self.theorem_operator is None:
            logger.warning(
                "Idempotence check (C2/L3) requested but no summarizer or theorem operator provided. "
                "Provide a summarizer or theorem operator to enable idempotence checking."
            )

        if bool(self.config.audit_substitution) and len(leaves) >= 2 and (self.summarizer is not None or self.theorem_operator is not None):
            leaves_in_order = [node for node in _traverse_inorder(tree.root) if node.is_leaf]
            adjacent_pairs = self._get_adjacent_leaf_pairs(leaves_in_order)
            if adjacent_pairs:
                sampled_pairs = self._apply_probability_gate(
                    self._sample_adjacent_pairs(adjacent_pairs, int(self.config.substitution_budget))
                )
                for sampled in sampled_pairs:
                    left_node, right_node = sampled.item
                    result = self._check_substitution(left_node, right_node, rubric)
                    result.inclusion_probability = float(sampled.inclusion_probability)
                    result.sampling_design = "adjacent_uniform"
                    checks.append(result)
                    substitution_samples += 1
                    if not result.passed:
                        substitution_violations += 1
        elif bool(self.config.audit_substitution) and len(leaves) >= 2 and self.summarizer is None and self.theorem_operator is None:
            logger.warning(
                "Substitution check (C3 Case A) requested but no summarizer or theorem operator provided. "
                "Provide a summarizer or theorem operator to enable substitution checking."
            )

        passed = sum(1 for c in checks if bool(c.passed))
        failed = len(checks) - passed
        failed_ids = [c.node_id for c in checks if not bool(c.passed)]

        report = AuditReport(
            tree_id=tree_id,
            source_doc_id=source_doc_id,
            total_nodes=len(all_nodes),
            nodes_audited=len(checks),
            nodes_passed=passed,
            nodes_failed=failed,
            failure_rate=(failed / len(checks)) if checks else 0.0,
            checks=checks,
            failed_node_ids=failed_ids,
            sufficiency_violations=sufficiency_violations,
            merge_violations=merge_violations,
            idempotence_violations=idempotence_violations,
            substitution_violations=substitution_violations,
            sufficiency_samples=sufficiency_samples,
            merge_samples=merge_samples,
            idempotence_samples=idempotence_samples,
            substitution_samples=substitution_samples,
            leaf_population=len(leaves),
            merge_population=len(internal),
            idempotence_population=len(internal),
            substitution_population=substitution_population,
            sampling_strategy=str(self.config.sampling_strategy.value),
            sampling_probability=float(self.config.sampling_probability),
            operator_capabilities=(
                self.theorem_operator.capability_report().to_dict()
                if self.theorem_operator is not None and hasattr(self.theorem_operator, "capability_report")
                else {}
            ),
            compositional_learning_problem={},
            logged_observations=[],
            logged_observation_artifacts={},
            inclusion_probability_map=dict(self._last_inclusion_prob_map),
        )
        report.logged_observations = report._build_logged_observations()
        return report


__all__ = ["StateTreeAuditor"]
