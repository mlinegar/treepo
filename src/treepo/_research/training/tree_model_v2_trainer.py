"""Shared supervision and trainer surface for tree-neural V2 models."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import torch
import torch.nn.functional as F

from treepo._research.training.supervision.local_law_torch import corrected_local_law_target_mse
from treepo._research.tree.packed_execution import PackedForwardResult
from treepo._research.tree.tree_model_v2 import TreeModelProtocol


SupervisionMode = Literal["dense_local_law", "sparse_local_law"]
FiberRelation = Literal["same", "different"]


@dataclass(frozen=True)
class TreeNodeRef:
    doc_index: int
    node_index: int


@dataclass(frozen=True)
class ScalarTarget:
    node_ref: TreeNodeRef
    value: float
    head: str = "rile"
    normalized: bool = False
    kind: str = "node"
    weight: float = 1.0
    proxy_value: Optional[float] = None
    oracle_value: Optional[float] = None
    observed: Optional[bool] = None
    propensity: Optional[float] = None
    local_law_adjustment: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FiberPairTarget:
    left_ref: TreeNodeRef
    right_ref: TreeNodeRef
    relation: FiberRelation
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FiberGroupTarget:
    node_refs: Tuple[TreeNodeRef, ...]
    relation: FiberRelation = "same"
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuxiliaryTarget:
    node_ref: TreeNodeRef
    name: str
    target: Any
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TreeSupervisionBatch:
    mode: SupervisionMode
    root_scalar_targets: Tuple[ScalarTarget, ...] = tuple()
    node_scalar_targets: Tuple[ScalarTarget, ...] = tuple()
    fiber_pair_targets: Tuple[FiberPairTarget, ...] = tuple()
    fiber_group_targets: Tuple[FiberGroupTarget, ...] = tuple()
    auxiliary_targets: Tuple[AuxiliaryTarget, ...] = tuple()
    adapter_name: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def all_node_refs(self) -> Tuple[TreeNodeRef, ...]:
        refs: List[TreeNodeRef] = []
        seen: set[TreeNodeRef] = set()

        def _append(ref: TreeNodeRef) -> None:
            if ref not in seen:
                seen.add(ref)
                refs.append(ref)

        for target in self.root_scalar_targets:
            _append(target.node_ref)
        for target in self.node_scalar_targets:
            _append(target.node_ref)
        for target in self.auxiliary_targets:
            _append(target.node_ref)
        for target in self.fiber_pair_targets:
            _append(target.left_ref)
            _append(target.right_ref)
        for group in self.fiber_group_targets:
            for ref in group.node_refs:
                _append(ref)
        return tuple(refs)


@runtime_checkable
class TreeTaskAdapter(Protocol):
    name: str
    head_name: str
    supervision_mode: SupervisionMode

    def build_supervision_batch(
        self,
        batch_items: Sequence[Any],
    ) -> TreeSupervisionBatch:
        ...

    def compute_auxiliary_losses(
        self,
        *,
        model: TreeModelProtocol,
        state_index: Mapping[TreeNodeRef, torch.Tensor],
        supervision_batch: TreeSupervisionBatch,
        device: torch.device,
    ) -> Mapping[str, torch.Tensor]:
        ...


TaskAdapter = TreeTaskAdapter


@dataclass(frozen=True, kw_only=True)
class TreeModelV2ScoreTargetConfig:
    target_min: float = -100.0
    target_max: float = 100.0


@dataclass(frozen=True, kw_only=True)
class TreeModelV2ObjectiveConfig:
    root_weight: float = 1.0
    leaf_scalar_weight: float = 0.0
    internal_scalar_weight: float = 0.0
    generic_node_scalar_weight: float = 0.0
    fiber_same_weight: float = 0.0
    fiber_different_weight: float = 0.0
    fiber_different_max_similarity: float = 0.25


@dataclass(frozen=True, kw_only=True)
class TreeModelV2TrainingConfig:
    score_targets: TreeModelV2ScoreTargetConfig = field(
        default_factory=TreeModelV2ScoreTargetConfig
    )
    objective: TreeModelV2ObjectiveConfig = field(
        default_factory=TreeModelV2ObjectiveConfig
    )


@dataclass(frozen=True)
class CompiledScalarTargetGroup:
    head: str
    state_indices: torch.Tensor
    target_values: torch.Tensor
    weights: torch.Tensor
    count: int
    corrected_local_law_mask: Optional[torch.Tensor] = None
    proxy_target_values: Optional[torch.Tensor] = None
    oracle_target_values: Optional[torch.Tensor] = None
    observed: Optional[torch.Tensor] = None
    propensities: Optional[torch.Tensor] = None
    corrected_local_law_count: int = 0


@dataclass(frozen=True)
class CompiledFiberPairGroup:
    relation: FiberRelation
    left_indices: torch.Tensor
    right_indices: torch.Tensor
    weights: torch.Tensor
    count: int


@dataclass(frozen=True)
class PreparedTreeBatch:
    supervision_batch: TreeSupervisionBatch
    state_index: Mapping[TreeNodeRef, torch.Tensor]
    state_batch: Optional[torch.Tensor] = None
    ref_index: Mapping[TreeNodeRef, int] = field(default_factory=dict)
    forward_result: Optional[Any] = None
    root_scalar_group: Optional[CompiledScalarTargetGroup] = None
    leaf_scalar_group: Optional[CompiledScalarTargetGroup] = None
    internal_scalar_group: Optional[CompiledScalarTargetGroup] = None
    generic_scalar_group: Optional[CompiledScalarTargetGroup] = None
    same_fiber_group: Optional[CompiledFiberPairGroup] = None
    different_fiber_group: Optional[CompiledFiberPairGroup] = None
    expanded_fiber_pair_count: int = 0


class TreeModelV2Trainer:
    """Generic local-law trainer/evaluator surface over tree-structured states."""

    def __init__(
        self,
        *,
        model: TreeModelProtocol,
        adapter: TreeTaskAdapter,
        forward_batch: Callable[[TreeModelProtocol, Sequence[Any]], Any],
        state_getter: Callable[[Any, int], Optional[torch.Tensor]],
        config: TreeModelV2TrainingConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.forward_batch = forward_batch
        self.state_getter = state_getter
        self.config = config
        self.device = device

    def prepare_batch(
        self,
        batch_items: Sequence[Any],
    ) -> PreparedTreeBatch:
        forward_output = self.forward_batch(self.model, batch_items)
        supervision_batch = self.adapter.build_supervision_batch(batch_items)
        state_index: Dict[TreeNodeRef, torch.Tensor] = {}
        ordered_refs = list(supervision_batch.all_node_refs)
        if isinstance(forward_output, PackedForwardResult):
            ref_index = {ref: idx for idx, ref in enumerate(ordered_refs)}
            state_batch = None
            if ordered_refs:
                global_indices = torch.tensor(
                    [
                        int(forward_output.global_index(ref.doc_index, ref.node_index))
                        for ref in ordered_refs
                    ],
                    dtype=torch.long,
                    device=forward_output.state_batch.device,
                )
                state_batch = forward_output.state_batch.index_select(0, global_indices)
                for idx, ref in enumerate(ordered_refs):
                    state_index[ref] = state_batch[idx]
            node_targets = list(supervision_batch.node_scalar_targets)
            leaf_targets = [target for target in node_targets if str(target.kind) == "leaf"]
            internal_targets = [
                target for target in node_targets if str(target.kind) == "internal"
            ]
            generic_targets = [
                target
                for target in node_targets
                if str(target.kind) not in {"leaf", "internal"}
            ]
            expanded_fiber_pairs = self._expanded_fiber_pairs(supervision_batch)
            same_pairs = [target for target in expanded_fiber_pairs if target.relation == "same"]
            diff_pairs = [target for target in expanded_fiber_pairs if target.relation == "different"]
            return PreparedTreeBatch(
                supervision_batch=supervision_batch,
                state_index=state_index,
                state_batch=state_batch,
                ref_index=ref_index,
                forward_result=forward_output,
                root_scalar_group=self._compile_scalar_target_group(
                    ref_index,
                    supervision_batch.root_scalar_targets,
                ),
                leaf_scalar_group=self._compile_scalar_target_group(ref_index, leaf_targets),
                internal_scalar_group=self._compile_scalar_target_group(ref_index, internal_targets),
                generic_scalar_group=self._compile_scalar_target_group(ref_index, generic_targets),
                same_fiber_group=self._compile_fiber_pair_group(
                    ref_index,
                    same_pairs,
                    relation="same",
                ),
                different_fiber_group=self._compile_fiber_pair_group(
                    ref_index,
                    diff_pairs,
                    relation="different",
                ),
                expanded_fiber_pair_count=int(len(expanded_fiber_pairs)),
            )
        for ref in ordered_refs:
            if ref in state_index:
                continue
            if ref.doc_index < 0 or ref.doc_index >= len(batch_items):
                raise IndexError(f"doc index out of range in supervision batch: {ref}")
            state = self.state_getter(batch_items[ref.doc_index], ref.node_index)
            if state is None:
                raise ValueError(f"state getter returned None for {ref}")
            state_index[ref] = state
        ref_index = {ref: idx for idx, ref in enumerate(ordered_refs)}
        state_batch = (
            torch.stack([state_index[ref] for ref in ordered_refs], dim=0)
            if ordered_refs
            else None
        )
        node_targets = list(supervision_batch.node_scalar_targets)
        leaf_targets = [target for target in node_targets if str(target.kind) == "leaf"]
        internal_targets = [
            target for target in node_targets if str(target.kind) == "internal"
        ]
        generic_targets = [
            target
            for target in node_targets
            if str(target.kind) not in {"leaf", "internal"}
        ]
        expanded_fiber_pairs = self._expanded_fiber_pairs(supervision_batch)
        same_pairs = [target for target in expanded_fiber_pairs if target.relation == "same"]
        diff_pairs = [target for target in expanded_fiber_pairs if target.relation == "different"]
        return PreparedTreeBatch(
            supervision_batch=supervision_batch,
            state_index=state_index,
            state_batch=state_batch,
            ref_index=ref_index,
            root_scalar_group=self._compile_scalar_target_group(
                ref_index,
                supervision_batch.root_scalar_targets,
            ),
            leaf_scalar_group=self._compile_scalar_target_group(ref_index, leaf_targets),
            internal_scalar_group=self._compile_scalar_target_group(ref_index, internal_targets),
            generic_scalar_group=self._compile_scalar_target_group(ref_index, generic_targets),
            same_fiber_group=self._compile_fiber_pair_group(
                ref_index,
                same_pairs,
                relation="same",
            ),
            different_fiber_group=self._compile_fiber_pair_group(
                ref_index,
                diff_pairs,
                relation="different",
            ),
            expanded_fiber_pair_count=int(len(expanded_fiber_pairs)),
        )

    def compute_supervision_loss(
        self,
        prepared: PreparedTreeBatch,
    ) -> tuple[torch.Tensor, int, Dict[str, Any]]:
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        total = zero.clone()
        n_terms = 0
        node_targets = list(prepared.supervision_batch.node_scalar_targets)
        stats: Dict[str, Any] = {
            "supervision_mode": str(prepared.supervision_batch.mode),
            "adapter_name": str(prepared.supervision_batch.adapter_name),
            "root_scalar_target_count": int(len(prepared.supervision_batch.root_scalar_targets)),
            "node_scalar_target_count": int(len(node_targets)),
            "leaf_scalar_target_count": int(
                0 if prepared.leaf_scalar_group is None else prepared.leaf_scalar_group.count
            ),
            "internal_scalar_target_count": int(
                0 if prepared.internal_scalar_group is None else prepared.internal_scalar_group.count
            ),
            "generic_node_scalar_target_count": int(
                0 if prepared.generic_scalar_group is None else prepared.generic_scalar_group.count
            ),
            "fiber_pair_target_count": int(len(prepared.supervision_batch.fiber_pair_targets)),
            "fiber_group_target_count": int(len(prepared.supervision_batch.fiber_group_targets)),
            "expanded_fiber_pair_count": int(prepared.expanded_fiber_pair_count),
            "same_fiber_pair_count": int(
                0 if prepared.same_fiber_group is None else prepared.same_fiber_group.count
            ),
            "different_fiber_pair_count": int(
                0 if prepared.different_fiber_group is None else prepared.different_fiber_group.count
            ),
            "auxiliary_target_count": int(len(prepared.supervision_batch.auxiliary_targets)),
            "uses_node_supervision": bool(node_targets),
            "uses_fiber_supervision": bool(prepared.expanded_fiber_pair_count),
        }

        # --- Root loss: SUM with count in n_terms (unchanged behaviour). ---
        root_loss = self._compiled_scalar_targets_loss(
            prepared,
            prepared.root_scalar_group,
        )
        if root_loss is not None:
            raw_loss, count = root_loss
            weighted = float(self.config.objective.root_weight) * raw_loss
            total = total + weighted
            n_terms += int(count)
            stats["root_scalar_loss"] = float(raw_loss.detach().cpu().item())

        # --- Local groups (leaf/internal/generic): normalise to per-sample
        # mean so their sample counts do NOT inflate n_terms and dilute the
        # root gradient.  The config weight controls relative importance. ---
        if node_targets:
            root_count = max(1, n_terms)  # use root count as the reference scale
            for key, group, weight in (
                (
                    "leaf_scalar_loss",
                    prepared.leaf_scalar_group,
                    float(self.config.objective.leaf_scalar_weight),
                ),
                (
                    "internal_scalar_loss",
                    prepared.internal_scalar_group,
                    float(self.config.objective.internal_scalar_weight),
                ),
                (
                    "generic_node_scalar_loss",
                    prepared.generic_scalar_group,
                    float(self.config.objective.generic_node_scalar_weight),
                ),
            ):
                if group is None or weight <= 0.0:
                    continue
                raw_loss, count = self._compiled_scalar_targets_loss(prepared, group)
                assert raw_loss is not None
                # Contribute as mean * root_count so the addition is
                # commensurate with the root sum, not proportional to the
                # local sample count.
                mean_loss = raw_loss / max(1, int(count))
                total = total + weight * mean_loss * float(root_count)
                # Do NOT add local count to n_terms — root count is enough.
                stats[key] = float(raw_loss.detach().cpu().item())
                if int(getattr(group, "corrected_local_law_count", 0) or 0) > 0:
                    stats[f"{key}_corrected_local_law_count"] = int(
                        group.corrected_local_law_count
                    )

        # --- Fiber pair losses: same treatment — normalise to mean, scale
        # by root_count so they don't inflate n_terms. ---
        root_count = max(1, n_terms)
        if (
            prepared.same_fiber_group is not None
            and float(self.config.objective.fiber_same_weight) > 0.0
        ):
            raw_loss, count = self._compiled_fiber_pair_loss(
                prepared,
                prepared.same_fiber_group,
            )
            mean_loss = raw_loss / max(1, int(count))
            total = total + float(self.config.objective.fiber_same_weight) * mean_loss * float(root_count)
            stats["fiber_same_loss"] = float(raw_loss.detach().cpu().item())
        if (
            prepared.different_fiber_group is not None
            and float(self.config.objective.fiber_different_weight) > 0.0
        ):
            raw_loss, count = self._compiled_fiber_pair_loss(
                prepared,
                prepared.different_fiber_group,
            )
            mean_loss = raw_loss / max(1, int(count))
            total = total + float(self.config.objective.fiber_different_weight) * mean_loss * float(root_count)
            stats["fiber_different_loss"] = float(raw_loss.detach().cpu().item())

        aux_losses = dict(
            self.adapter.compute_auxiliary_losses(
                model=self.model,
                state_index=prepared.state_index,
                supervision_batch=prepared.supervision_batch,
                device=self.device,
            )
            or {}
        )
        if aux_losses:
            actual_aux_count = 0
            for name, value in aux_losses.items():
                if value is None:
                    continue
                actual_aux_count += 1
                total = total + value
                stats[str(name)] = float(value.detach().cpu().item())
            if actual_aux_count > 0:
                n_terms += max(1, len(prepared.supervision_batch.auxiliary_targets))

        stats["loss_term_count"] = int(n_terms)
        return total, n_terms, stats

    def _scalar_targets_loss(
        self,
        state_index: Mapping[TreeNodeRef, torch.Tensor],
        targets: Sequence[ScalarTarget],
    ) -> Optional[tuple[torch.Tensor, int]]:
        if not targets:
            return None
        head = str(targets[0].head or self.adapter.head_name)
        grouped_targets: List[ScalarTarget] = list(targets)
        states = torch.stack([state_index[target.node_ref] for target in grouped_targets], dim=0)
        preds = self.model.predict_normalized_batch(states, head=head).reshape(-1)
        target_values = torch.as_tensor(
            [self._normalized_scalar_target(target) for target in grouped_targets],
            dtype=preds.dtype,
            device=preds.device,
        )
        weights = torch.as_tensor(
            [float(max(0.0, target.weight)) for target in grouped_targets],
            dtype=preds.dtype,
            device=preds.device,
        )
        loss = (((preds - target_values) ** 2) * weights).sum()
        return loss, int(len(grouped_targets))

    def _compile_scalar_target_group(
        self,
        ref_index: Mapping[TreeNodeRef, int],
        targets: Sequence[ScalarTarget],
    ) -> Optional[CompiledScalarTargetGroup]:
        if not targets:
            return None
        head = str(targets[0].head or self.adapter.head_name)
        adjustment_mask = [self._uses_corrected_local_law_target(target) for target in targets]
        has_adjustment = any(adjustment_mask)
        return CompiledScalarTargetGroup(
            head=head,
            state_indices=torch.as_tensor(
                [int(ref_index[target.node_ref]) for target in targets],
                dtype=torch.long,
                device=self.device,
            ),
            target_values=torch.as_tensor(
                [self._normalized_scalar_target(target) for target in targets],
                dtype=torch.float32,
                device=self.device,
            ),
            weights=torch.as_tensor(
                [float(max(0.0, target.weight)) for target in targets],
                dtype=torch.float32,
                device=self.device,
            ),
            count=int(len(targets)),
            corrected_local_law_mask=(
                torch.as_tensor(adjustment_mask, dtype=torch.bool, device=self.device)
                if has_adjustment
                else None
            ),
            proxy_target_values=(
                torch.as_tensor(
                    [
                        self._normalized_scalar_value(
                            self._target_proxy_value(target),
                            normalized=bool(target.normalized),
                        )
                        for target in targets
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                if has_adjustment
                else None
            ),
            oracle_target_values=(
                torch.as_tensor(
                    [
                        self._normalized_scalar_value(
                            self._target_oracle_value(target),
                            normalized=bool(target.normalized),
                        )
                        for target in targets
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                if has_adjustment
                else None
            ),
            observed=(
                torch.as_tensor(
                    [self._target_observed(target) for target in targets],
                    dtype=torch.float32,
                    device=self.device,
                )
                if has_adjustment
                else None
            ),
            propensities=(
                torch.as_tensor(
                    [self._target_propensity(target) for target in targets],
                    dtype=torch.float32,
                    device=self.device,
                )
                if has_adjustment
                else None
            ),
            corrected_local_law_count=int(sum(1 for flag in adjustment_mask if flag)),
        )

    def _compiled_scalar_targets_loss(
        self,
        prepared: PreparedTreeBatch,
        group: Optional[CompiledScalarTargetGroup],
    ) -> Optional[tuple[torch.Tensor, int]]:
        if group is None or prepared.state_batch is None or group.count <= 0:
            return None
        states = prepared.state_batch.index_select(0, group.state_indices)
        preds = self.model.predict_normalized_batch(states, head=group.head).reshape(-1)
        target_values = group.target_values.to(device=preds.device, dtype=preds.dtype)
        weights = group.weights.to(device=preds.device, dtype=preds.dtype)
        per_row_loss = (preds - target_values) ** 2
        if (
            group.corrected_local_law_mask is not None
            and group.proxy_target_values is not None
            and group.oracle_target_values is not None
            and group.observed is not None
            and group.propensities is not None
        ):
            corrected_rows = corrected_local_law_target_mse(
                predictions=preds,
                proxy_targets=group.proxy_target_values,
                oracle_targets=group.oracle_target_values,
                observed=group.observed,
                propensity=group.propensities,
                weights=None,
            )
            mask = group.corrected_local_law_mask.to(device=preds.device)
            per_row_loss = torch.where(mask, corrected_rows, per_row_loss)
        loss = (per_row_loss * weights).sum()
        return loss, int(group.count)

    def _normalized_scalar_target(self, target: ScalarTarget) -> float:
        return self._normalized_scalar_value(float(target.value), normalized=bool(target.normalized))

    def _normalized_scalar_value(self, value: float, *, normalized: bool) -> float:
        if bool(normalized):
            return float(value)
        span = float(self.config.score_targets.target_max) - float(self.config.score_targets.target_min)
        if span <= 0.0:
            return 0.5
        return (float(value) - float(self.config.score_targets.target_min)) / float(span)

    def _target_adjustment_payload(self, target: ScalarTarget) -> Mapping[str, Any]:
        metadata = dict(target.metadata or {})
        payload = metadata.get("local_law_adjustment")
        if isinstance(payload, Mapping):
            return payload
        payload = metadata.get("corrected_local_law")
        if isinstance(payload, Mapping):
            return payload
        return {}

    def _uses_corrected_local_law_target(self, target: ScalarTarget) -> bool:
        payload = self._target_adjustment_payload(target)
        enabled = bool(target.local_law_adjustment) or bool(payload.get("enabled", False))
        if not enabled:
            return False
        return self._target_proxy_value(target) is not None and self._target_oracle_value(target) is not None

    def _target_proxy_value(self, target: ScalarTarget) -> float:
        if target.proxy_value is not None:
            return float(target.proxy_value)
        payload = self._target_adjustment_payload(target)
        for key in ("proxy_value", "proxy_target", "proxy"):
            if key in payload and payload[key] is not None:
                return float(payload[key])
        return float(target.value)

    def _target_oracle_value(self, target: ScalarTarget) -> float:
        if target.oracle_value is not None:
            return float(target.oracle_value)
        payload = self._target_adjustment_payload(target)
        for key in ("oracle_value", "oracle_target", "oracle"):
            if key in payload and payload[key] is not None:
                return float(payload[key])
        return float(target.value)

    def _target_observed(self, target: ScalarTarget) -> bool:
        if target.observed is not None:
            return bool(target.observed)
        payload = self._target_adjustment_payload(target)
        for key in ("observed", "sampled", "node_observed"):
            if key in payload:
                return bool(payload[key])
        return True

    def _target_propensity(self, target: ScalarTarget) -> float:
        if target.propensity is not None:
            return float(target.propensity)
        payload = self._target_adjustment_payload(target)
        for key in ("propensity", "joint_propensity", "sampling_joint_inclusion_prob"):
            if key in payload and payload[key] is not None:
                return float(payload[key])
        return 1.0

    def _expanded_fiber_pairs(
        self,
        supervision_batch: TreeSupervisionBatch,
    ) -> List[FiberPairTarget]:
        expanded = list(supervision_batch.fiber_pair_targets)
        for group in supervision_batch.fiber_group_targets:
            refs = list(group.node_refs)
            for left_idx in range(len(refs)):
                for right_idx in range(left_idx + 1, len(refs)):
                    expanded.append(
                        FiberPairTarget(
                            left_ref=refs[left_idx],
                            right_ref=refs[right_idx],
                            relation=group.relation,
                            weight=float(group.weight),
                            metadata=dict(group.metadata),
                        )
                    )
        return expanded

    def _compile_fiber_pair_group(
        self,
        ref_index: Mapping[TreeNodeRef, int],
        pairs: Sequence[FiberPairTarget],
        *,
        relation: FiberRelation,
    ) -> Optional[CompiledFiberPairGroup]:
        if not pairs:
            return None
        return CompiledFiberPairGroup(
            relation=relation,
            left_indices=torch.as_tensor(
                [int(ref_index[pair.left_ref]) for pair in pairs],
                dtype=torch.long,
                device=self.device,
            ),
            right_indices=torch.as_tensor(
                [int(ref_index[pair.right_ref]) for pair in pairs],
                dtype=torch.long,
                device=self.device,
            ),
            weights=torch.as_tensor(
                [float(max(0.0, pair.weight)) for pair in pairs],
                dtype=torch.float32,
                device=self.device,
            ),
            count=int(len(pairs)),
        )

    def _fiber_pair_loss(
        self,
        state_index: Mapping[TreeNodeRef, torch.Tensor],
        pairs: Sequence[FiberPairTarget],
        *,
        relation: FiberRelation,
    ) -> tuple[torch.Tensor, int]:
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        if not pairs:
            return zero, 0
        unique_refs: List[TreeNodeRef] = []
        unique_index: Dict[TreeNodeRef, int] = {}
        for pair in pairs:
            for ref in (pair.left_ref, pair.right_ref):
                if ref not in unique_index:
                    unique_index[ref] = len(unique_refs)
                    unique_refs.append(ref)
        state_batch = torch.stack([state_index[ref] for ref in unique_refs], dim=0)
        features = None
        if getattr(self.model, "has_phi", False):
            try:
                features = self.model.phi_fiber(state_batch)
            except TypeError:
                features = None
        if features is None:
            try:
                features = self.model.phi_batch(state_batch)
            except TypeError:
                features = None
        if features is None:
            features = state_batch
        features = F.normalize(features, dim=-1, eps=1e-6)

        left_indices = torch.as_tensor(
            [unique_index[pair.left_ref] for pair in pairs],
            dtype=torch.long,
            device=features.device,
        )
        right_indices = torch.as_tensor(
            [unique_index[pair.right_ref] for pair in pairs],
            dtype=torch.long,
            device=features.device,
        )
        weights = torch.as_tensor(
            [float(max(0.0, pair.weight)) for pair in pairs],
            dtype=features.dtype,
            device=features.device,
        )
        cos = (features.index_select(0, left_indices) * features.index_select(0, right_indices)).sum(dim=-1)
        if relation == "same":
            loss = ((1.0 - cos) * weights).sum()
        else:
            margin = float(self.config.objective.fiber_different_max_similarity)
            loss = (torch.relu(cos - margin) ** 2 * weights).sum()
        return loss, int(len(pairs))

    def _compiled_fiber_pair_loss(
        self,
        prepared: PreparedTreeBatch,
        group: Optional[CompiledFiberPairGroup],
    ) -> tuple[torch.Tensor, int]:
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        if group is None or prepared.state_batch is None or group.count <= 0:
            return zero, 0
        state_batch = prepared.state_batch
        features = None
        if getattr(self.model, "has_phi", False):
            try:
                features = self.model.phi_fiber(state_batch)
            except TypeError:
                features = None
        if features is None:
            try:
                features = self.model.phi_batch(state_batch)
            except TypeError:
                features = None
        if features is None:
            features = state_batch
        features = F.normalize(features, dim=-1, eps=1e-6)
        left_indices = group.left_indices.to(device=features.device)
        right_indices = group.right_indices.to(device=features.device)
        weights = group.weights.to(device=features.device, dtype=features.dtype)
        cos = (
            features.index_select(0, left_indices)
            * features.index_select(0, right_indices)
        ).sum(dim=-1)
        if group.relation == "same":
            loss = ((1.0 - cos) * weights).sum()
        else:
            margin = float(self.config.objective.fiber_different_max_similarity)
            loss = (torch.relu(cos - margin) ** 2 * weights).sum()
        return loss, int(group.count)


class RealDocumentTaskAdapter:
    """Sparse local-law adapter for real-document RILE trees."""

    name = "real_document_rile"
    head_name = "rile"
    supervision_mode: SupervisionMode = "sparse_local_law"

    def __init__(
        self,
        *,
        max_leaf_targets_per_doc: int,
        max_internal_targets_per_doc: int,
        rng: Optional[random.Random] = None,
        enable_fiber_constraints: bool = False,
        fiber_same_threshold: float = 10.0,
        fiber_diff_threshold: float = 30.0,
    ) -> None:
        self.max_leaf_targets_per_doc = int(max(0, max_leaf_targets_per_doc))
        self.max_internal_targets_per_doc = int(max(0, max_internal_targets_per_doc))
        self.rng = rng or random.Random(0)
        self.enable_fiber_constraints = bool(enable_fiber_constraints)
        self.fiber_same_threshold = float(max(0.0, fiber_same_threshold))
        self.fiber_diff_threshold = float(max(self.fiber_same_threshold, fiber_diff_threshold))

    def build_supervision_batch(
        self,
        batch_items: Sequence[tuple[Sequence[Any], float, str]],
    ) -> TreeSupervisionBatch:
        root_targets: List[ScalarTarget] = []
        node_targets: List[ScalarTarget] = []
        doc_roots: List[tuple[TreeNodeRef, float, str]] = []

        for doc_index, (nodes, rile, doc_id) in enumerate(batch_items):
            if not nodes:
                continue
            root_ref = TreeNodeRef(doc_index, len(nodes) - 1)
            root_targets.append(
                ScalarTarget(
                    node_ref=root_ref,
                    value=float(rile),
                    head=self.head_name,
                    kind="root",
                    metadata={"doc_id": str(doc_id)},
                )
            )
            doc_roots.append((root_ref, float(rile), str(doc_id)))

            leaf_indices = [
                idx for idx, node in enumerate(nodes)
                if bool(getattr(node, "is_leaf", False))
                and "rile" in dict(getattr(node, "oracle_scores", {}) or {})
            ]
            internal_indices = [
                idx for idx, node in enumerate(nodes)
                if not bool(getattr(node, "is_leaf", False))
                and "rile" in dict(getattr(node, "oracle_scores", {}) or {})
            ]
            if self.max_leaf_targets_per_doc <= 0:
                leaf_indices = []
            elif len(leaf_indices) > self.max_leaf_targets_per_doc:
                leaf_indices = self.rng.sample(leaf_indices, k=self.max_leaf_targets_per_doc)
            if self.max_internal_targets_per_doc <= 0:
                internal_indices = []
            elif len(internal_indices) > self.max_internal_targets_per_doc:
                internal_indices = self.rng.sample(
                    internal_indices,
                    k=self.max_internal_targets_per_doc,
                )
            for idx in leaf_indices:
                node = nodes[idx]
                node_targets.append(
                    ScalarTarget(
                        node_ref=TreeNodeRef(doc_index, idx),
                        value=float(node.oracle_scores["rile"]),
                        head=self.head_name,
                        kind="leaf",
                        metadata={"doc_id": str(doc_id)},
                    )
                )
            for idx in internal_indices:
                node = nodes[idx]
                node_targets.append(
                    ScalarTarget(
                        node_ref=TreeNodeRef(doc_index, idx),
                        value=float(node.oracle_scores["rile"]),
                        head=self.head_name,
                        kind="internal",
                        metadata={"doc_id": str(doc_id)},
                    )
                )

        fiber_pairs: List[FiberPairTarget] = []
        if self.enable_fiber_constraints:
            for left_index in range(len(doc_roots)):
                left_ref, left_value, left_doc_id = doc_roots[left_index]
                for right_index in range(left_index + 1, len(doc_roots)):
                    right_ref, right_value, right_doc_id = doc_roots[right_index]
                    gap = abs(float(left_value) - float(right_value))
                    relation: Optional[FiberRelation] = None
                    if gap <= self.fiber_same_threshold:
                        relation = "same"
                    elif gap >= self.fiber_diff_threshold:
                        relation = "different"
                    if relation is None:
                        continue
                    fiber_pairs.append(
                        FiberPairTarget(
                            left_ref=left_ref,
                            right_ref=right_ref,
                            relation=relation,
                            metadata={
                                "left_doc_id": left_doc_id,
                                "right_doc_id": right_doc_id,
                                "target_gap": float(gap),
                            },
                        )
                    )

        return TreeSupervisionBatch(
            mode=self.supervision_mode,
            root_scalar_targets=tuple(root_targets),
            node_scalar_targets=tuple(node_targets),
            fiber_pair_targets=tuple(fiber_pairs),
            adapter_name=self.name,
            metadata={"n_docs": int(len(batch_items))},
        )

    def compute_auxiliary_losses(
        self,
        *,
        model: TreeModelProtocol,
        state_index: Mapping[TreeNodeRef, torch.Tensor],
        supervision_batch: TreeSupervisionBatch,
        device: torch.device,
    ) -> Mapping[str, torch.Tensor]:
        return {}


class MarkovTaskAdapter:
    """Dense local-law adapter exposing the Markov exact supervision surface."""

    name = "markov_count_tree"
    head_name = "count"
    supervision_mode: SupervisionMode = "dense_local_law"

    def __init__(
        self,
        *,
        theorem_feature_adapter_name: str,
        target_scale: float,
        include_auxiliary_targets: bool = True,
    ) -> None:
        self.theorem_feature_adapter_name = str(theorem_feature_adapter_name)
        self.target_scale = float(target_scale)
        self.include_auxiliary_targets = bool(include_auxiliary_targets)

    def build_supervision_batch(
        self,
        batch_items: Sequence[Any],
    ) -> TreeSupervisionBatch:
        from treepo._research.ctreepo.sim.core.markov_neural_operator_baselines import (
            _balanced_exact_sketch_targets,
        )
        from treepo._research.ctreepo.sim.core.theorem_feature_route import (
            build_theorem_feature_pair_sets,
            resolve_theorem_feature_adapter,
            theorem_feature_targets_from_markov_exact_targets,
        )

        theorem_feature_adapter = resolve_theorem_feature_adapter(
            self.theorem_feature_adapter_name
        )

        root_targets: List[ScalarTarget] = []
        node_targets: List[ScalarTarget] = []
        fiber_pairs: List[FiberPairTarget] = []
        auxiliary_targets: List[AuxiliaryTarget] = []

        for doc_index, doc in enumerate(batch_items):
            n_leaves = int(len(doc.leaf_token_ids))
            n_merges = int(len(doc.merge_counts_balanced))
            total_nodes = int(n_leaves + n_merges)
            if total_nodes <= 0:
                continue

            exact_targets = _balanced_exact_sketch_targets(
                leaf_counts=doc.leaf_counts,
                leaf_first_regimes=doc.leaf_first_regimes,
                leaf_last_regimes=doc.leaf_last_regimes,
            )
            feature_targets = theorem_feature_targets_from_markov_exact_targets(
                adapter=theorem_feature_adapter,
                exact_targets=exact_targets,
            )

            root_targets.append(
                ScalarTarget(
                    node_ref=TreeNodeRef(doc_index, total_nodes - 1),
                    value=float(doc.root_count),
                    head=self.head_name,
                    kind="root",
                    metadata={"n_leaves": int(n_leaves)},
                )
            )
            for leaf_index, count in enumerate(doc.leaf_counts):
                node_targets.append(
                    ScalarTarget(
                        node_ref=TreeNodeRef(doc_index, leaf_index),
                        value=float(count),
                        head=self.head_name,
                        kind="leaf",
                        metadata={"leaf_token_length": int(doc.leaf_token_lengths[leaf_index])},
                    )
                )
            for merge_index, count in enumerate(doc.merge_counts_balanced):
                node_targets.append(
                    ScalarTarget(
                        node_ref=TreeNodeRef(doc_index, n_leaves + merge_index),
                        value=float(count),
                        head=self.head_name,
                        kind="internal",
                    )
                )

            all_labels = tuple(feature_targets.leaf) + tuple(feature_targets.merge)
            pair_sets = build_theorem_feature_pair_sets(
                all_labels,
                adapter=theorem_feature_adapter,
            )
            for left_index, right_index in pair_sets.same_pairs:
                fiber_pairs.append(
                    FiberPairTarget(
                        left_ref=TreeNodeRef(doc_index, int(left_index)),
                        right_ref=TreeNodeRef(doc_index, int(right_index)),
                        relation="same",
                    )
                )
            for left_index, right_index in pair_sets.different_pairs:
                fiber_pairs.append(
                    FiberPairTarget(
                        left_ref=TreeNodeRef(doc_index, int(left_index)),
                        right_ref=TreeNodeRef(doc_index, int(right_index)),
                        relation="different",
                    )
                )

            if self.include_auxiliary_targets:
                for leaf_index, (_count, first_regime, last_regime) in enumerate(exact_targets["leaf"]):
                    auxiliary_targets.append(
                        AuxiliaryTarget(
                            node_ref=TreeNodeRef(doc_index, leaf_index),
                            name="first_regime",
                            target=int(first_regime),
                        )
                    )
                    auxiliary_targets.append(
                        AuxiliaryTarget(
                            node_ref=TreeNodeRef(doc_index, leaf_index),
                            name="last_regime",
                            target=int(last_regime),
                        )
                    )
                for merge_index, (_count, first_regime, last_regime) in enumerate(exact_targets["merge"]):
                    merge_ref = TreeNodeRef(doc_index, n_leaves + merge_index)
                    auxiliary_targets.append(
                        AuxiliaryTarget(
                            node_ref=merge_ref,
                            name="first_regime",
                            target=int(first_regime),
                        )
                    )
                    auxiliary_targets.append(
                        AuxiliaryTarget(
                            node_ref=merge_ref,
                            name="last_regime",
                            target=int(last_regime),
                        )
                    )
                for merge_index, join_bit in enumerate(exact_targets["merge_join_bits"]):
                    auxiliary_targets.append(
                        AuxiliaryTarget(
                            node_ref=TreeNodeRef(doc_index, n_leaves + merge_index),
                            name="join_bit",
                            target=int(join_bit),
                        )
                    )

        return TreeSupervisionBatch(
            mode=self.supervision_mode,
            root_scalar_targets=tuple(root_targets),
            node_scalar_targets=tuple(node_targets),
            fiber_pair_targets=tuple(fiber_pairs),
            auxiliary_targets=tuple(auxiliary_targets),
            adapter_name=self.name,
            metadata={"target_scale": float(self.target_scale)},
        )

    def compute_auxiliary_losses(
        self,
        *,
        model: TreeModelProtocol,
        state_index: Mapping[TreeNodeRef, torch.Tensor],
        supervision_batch: TreeSupervisionBatch,
        device: torch.device,
    ) -> Mapping[str, torch.Tensor]:
        return {}


def build_markov_dense_supervision_batch(
    docs: Sequence[Any],
    *,
    theorem_feature_adapter_name: str,
    target_scale: float,
) -> TreeSupervisionBatch:
    adapter = MarkovTaskAdapter(
        theorem_feature_adapter_name=theorem_feature_adapter_name,
        target_scale=target_scale,
    )
    return adapter.build_supervision_batch(docs)
