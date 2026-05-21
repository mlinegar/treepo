from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from treepo._research.unified_g_v1.core.contracts import MarkovRunSpec, Profile, SupervisionPolicy
from treepo._research.unified_g_v1.core.program import UnifiedFGProgram, UnifiedGContract, UnifiedGSurface
from treepo._research.unified_g_v1.core.specs import build_markov_token_fno_program_spec

from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import OPSCountConfig


DOC_TOKENS = 128


_UNIFIED_G_COMMON: dict[str, Any] = {
    "n_epochs": 40,
    "batch_size": 64,
    "lr": 5e-4,
    "weight_decay": 0.0,
    "state_dim": 128,
    "hidden_dim": 512,
    "task_objective_weight": 1.0,
    "local_law_weight": 0.8,
    "c1_relative_weight": 1.0,
    "c2_relative_weight": 1.0,
    "c3_relative_weight": 1.0,
    "tree_model_version": "unified_g",
    "tree_batch_runtime_mode": "unified_v2",
    "tree_batch_pack_mode": "fixed_fused",
    "tree_training_schedule": "two_stage",
    "tree_task_head_mode": "theorem_feature_scalar",
    "tree_theorem_surface_mode": "factorized_score_fiber",
    "tree_summary_spec_root_mode": "factored_theorem_readout",
    "tree_theorem_feature_dim": 48,
    "tree_theorem_feature_hidden_dim": 256,
    "tree_theorem_score_dim": 1,
    "tree_theorem_fiber_dim": 47,
    "tree_theorem_aux_dim": 0,
    "tree_leaf_fno_width": 128,
    "tree_leaf_fno_n_modes": 8,
    "tree_leaf_fno_n_layers": 4,
    "tree_root_supervision_kind": "mse",
    "tree_checkpoint_metric": "val_root_mae",
    "tree_stage1_checkpoint_metric": "val_theorem_bootstrap_direct",
    "tree_stage1_eval_mode": "per_epoch",
    "tree_stage1_epochs": 10,
    "tree_stage2_epochs": 30,
    "tree_stage1_root_weight": 0.0,
    "tree_join_bit_weight": 1.0,
    "tree_theorem_count_head_mode": "scalar_mse",
    "tree_theorem_count_dim": 8,
    "tree_theorem_first_dim": 8,
    "tree_theorem_last_dim": 8,
    "tree_theorem_count_ordinal_weight": 1.0,
    "tree_theorem_count_scalar_aux_weight": 0.25,
    "tree_theorem_count_threshold_balance": True,
    "tree_score_merge_mode": "exact_projected_sketch",
    "tree_phi_compose_weight": 0.0,
    "tree_phi_contrastive_weight": 0.0,
    "tree_phi_alignment_loss": "cosine_mse",
    "aligned_sketch_surface": "",
    "summary_spec_name": "markov_count_sketch",
    "slot_count": 4,
    "leaf_supervision_kind": "full_sketch",
    "leaf_label_rate": 1.0,
    "leaf_exact_supervision": False,
    "internal_supervision_kind": "full_sketch",
    "internal_label_rate": 1.0,
    "root_weight": 1.0,
    "depth_discount_gamma": 1.0,
    "schedule_consistency_weight": 0.0,
    "doc_sequence_train_fraction": 0.0,
}

_ROOT_ONLY_PROFILE: dict[str, Any] = {
    **_UNIFIED_G_COMMON,
    "tree_training_schedule": "single_stage",
    "tree_stage1_epochs": 0,
    "tree_stage2_epochs": 0,
    "tree_stage1_checkpoint_metric": "val_root_mae",
    "local_law_weight": 0.0,
    "c1_relative_weight": 0.0,
    "c2_relative_weight": 0.0,
    "c3_relative_weight": 0.0,
}

_FNO_CANARY_PROFILE: dict[str, Any] = {
    **_ROOT_ONLY_PROFILE,
    "tree_leaf_fno_n_modes": 16,
    "tree_root_supervision_kind": "count_ce",
    "include_fno_baseline": True,
}


def profile_overrides(profile: Profile) -> Mapping[str, Any]:
    if profile == Profile.STANDARD:
        return dict(_UNIFIED_G_COMMON)
    if profile == Profile.ROOT_ONLY:
        return dict(_ROOT_ONLY_PROFILE)
    if profile == Profile.FNO_CANARY:
        return dict(_FNO_CANARY_PROFILE)
    if profile == Profile.DUPLICATE_LOCAL_LABEL_ONE_LEAF:
        return dict(_UNIFIED_G_COMMON)
    raise ValueError(f"unsupported profile={profile!r}")


def _leaf_mass_eq_budget(root_share: int, leaf_tokens: int) -> dict[str, Any]:
    root_fraction = float(int(root_share)) / 100.0
    leaf_fraction = max(0.0, 1.0 - root_fraction)
    local_mass_per_call = float(int(leaf_tokens)) / float(DOC_TOKENS)
    if local_mass_per_call <= 0.0:
        raise ValueError("leaf_tokens must be positive for leaf_mass_eq")
    local_calls_per_doc = float(leaf_fraction) / float(local_mass_per_call)
    total_calls_per_doc = float(root_fraction) + float(local_calls_per_doc)
    full_doc_budget_share = (
        float(root_fraction) / float(total_calls_per_doc)
        if total_calls_per_doc > 0.0
        else 0.0
    )
    return {
        "budget_total_calls_per_doc": float(total_calls_per_doc),
        "full_doc_budget_share": float(full_doc_budget_share),
        "doc_consumption_mode": "root_only",
        "local_split_mode": "leaf_only",
        "local_allocation_policy": "breadth_first",
        "package_semantics": "leaf_mass_eq",
        "mass_target_per_doc": 1.0,
    }


def _root_only_budget(root_share: int) -> dict[str, Any]:
    root_fraction = float(int(root_share)) / 100.0
    return {
        "budget_total_calls_per_doc": float(root_fraction),
        "full_doc_budget_share": 1.0,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "inactive_for_family",
        "local_allocation_policy": "breadth_first",
        "package_semantics": "root_only",
        "mass_target_per_doc": float(root_fraction),
    }


def _duplicate_local_label_budget(root_share: int) -> dict[str, Any]:
    root_fraction = float(int(root_share)) / 100.0
    return {
        "budget_total_calls_per_doc": float(2.0 * root_fraction),
        "full_doc_budget_share": 0.5,
        "doc_consumption_mode": "root_only",
        "local_split_mode": "leaf_only",
        "local_allocation_policy": "breadth_first",
        "package_semantics": "duplicate_local_label_one_leaf",
        "mass_target_per_doc": float(2.0 * root_fraction),
    }


def supervision_budget_fields(spec: MarkovRunSpec) -> Mapping[str, Any]:
    if spec.profile == Profile.DUPLICATE_LOCAL_LABEL_ONE_LEAF:
        return _duplicate_local_label_budget(int(spec.root_share))
    if spec.supervision_policy == SupervisionPolicy.ROOT_ONLY:
        return _root_only_budget(int(spec.root_share))
    if spec.supervision_policy == SupervisionPolicy.LEAF_MASS_EQ:
        return _leaf_mass_eq_budget(int(spec.root_share), int(spec.leaf_tokens))
    raise ValueError(f"unsupported supervision_policy={spec.supervision_policy!r}")


def _markov_contract(spec: MarkovRunSpec) -> UnifiedGContract:
    profile_fields = dict(profile_overrides(spec.profile))
    operator_width = int(profile_fields.get("tree_leaf_fno_width", 128) or 128)
    operator_modes = int(profile_fields.get("tree_leaf_fno_n_modes", 8) or 8)
    program_spec = build_markov_token_fno_program_spec(
        feature_dim=operator_width,
        tokenizer_or_adapter_id="markov_token_ids",
        operator_width=operator_width,
        operator_modes=operator_modes,
    )
    return UnifiedGContract(
        name=f"markov_unified_g__{spec.profile.value}",
        surface=UnifiedGSurface(
            raw_input_kind="markov_token_sequence_or_child_state_pair",
            g_input_kind="markov_summary_surface",
            state_kind="markov_tree_state",
            output_kind="root_count_prediction",
            task_spec_kind="markov_run_spec",
            backend_family=str(program_spec.program_family),
            shared_g=True,
        ),
        leaf_adapter_name="token_sequence_to_summary_surface",
        merge_adapter_name="child_state_pair_to_summary_surface",
        g_name="shared_markov_summary_encoder",
        f_name="count_readout_head",
        comparator_refs=("official_fno", "one_leaf_canary"),
        notes=(
            "Markov Unified-G route: leaf and merge inputs land on one summary "
            "surface before a shared g produces the tree state."
        ),
        program_spec=program_spec,
        extra={
            "profile": spec.profile.value,
            "supervision_policy": spec.supervision_policy.value,
            "scope": spec.scope.value,
            "train_docs": int(spec.train_docs),
            "root_share": int(spec.root_share),
            "leaf_tokens": int(spec.leaf_tokens),
        },
    )


@dataclass(frozen=True)
class MarkovSummarySurface:
    source: str
    feature_vector: tuple[float, ...]
    root_share: int
    leaf_tokens: int
    profile: str
    supervision_policy: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "feature_vector": [float(value) for value in self.feature_vector],
            "root_share": int(self.root_share),
            "leaf_tokens": int(self.leaf_tokens),
            "profile": self.profile,
            "supervision_policy": self.supervision_policy,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MarkovTreeState:
    summary_surface: MarkovSummarySurface
    latent_vector: tuple[float, ...]
    count_estimate: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_surface": self.summary_surface.to_dict(),
            "latent_vector": [float(value) for value in self.latent_vector],
            "count_estimate": float(self.count_estimate),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MarkovPrediction:
    count_estimate: float
    state_width: int
    comparator_refs: tuple[str, ...]
    profile: str
    supervision_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "count_estimate": float(self.count_estimate),
            "state_width": int(self.state_width),
            "comparator_refs": [str(value) for value in self.comparator_refs],
            "profile": self.profile,
            "supervision_policy": self.supervision_policy,
        }


def _coerce_token_sequence(raw_input: Any) -> tuple[int, ...]:
    candidate: Any = raw_input
    if isinstance(raw_input, Mapping):
        for key in ("token_ids", "tokens", "sequence", "values"):
            if key in raw_input:
                candidate = raw_input[key]
                break
    if candidate is None:
        return tuple()
    if isinstance(candidate, str):
        values: list[int] = []
        for chunk in candidate.replace(",", " ").split():
            try:
                values.append(int(chunk))
            except ValueError:
                continue
        return tuple(values)
    if isinstance(candidate, Sequence) and not isinstance(candidate, (bytes, bytearray)):
        values = []
        for item in candidate:
            try:
                values.append(int(item))
            except (TypeError, ValueError):
                continue
        return tuple(values)
    try:
        return (int(candidate),)
    except (TypeError, ValueError):
        return tuple()


def _token_features(tokens: Sequence[int], *, width: int, leaf_tokens: int) -> tuple[float, ...]:
    used = tuple(int(value) for value in tokens) or (0,)
    width = max(4, int(width))
    n_tokens = float(len(used))
    max_token = float(max(max(used), 1))
    mean_token = float(sum(used)) / n_tokens
    variance = float(sum((value - mean_token) ** 2 for value in used)) / n_tokens
    std_token = variance ** 0.5
    unique_ratio = float(len(set(used))) / n_tokens
    transition_ratio = (
        float(sum(1 for left, right in zip(used, used[1:]) if left != right)) / float(max(1, len(used) - 1))
        if len(used) > 1
        else 0.0
    )
    features: list[float] = [
        min(1.0, n_tokens / float(max(1, int(leaf_tokens)))),
        unique_ratio,
        mean_token / max_token,
        min(1.0, std_token / max_token),
        transition_ratio,
    ]
    histogram_slots = max(0, width - len(features))
    if histogram_slots:
        histogram = [0.0] * histogram_slots
        for token in used:
            histogram[int(token) % histogram_slots] += 1.0
        features.extend(value / n_tokens for value in histogram)
    return tuple(float(value) for value in features[:width])


def _mix_features(
    values: Sequence[float],
    *,
    width: int,
) -> tuple[float, ...]:
    source = [float(value) for value in values]
    if not source:
        source = [0.0] * max(1, int(width))
    width = max(1, int(width))
    latent: list[float] = []
    for idx in range(width):
        here = source[idx % len(source)]
        prev_value = source[(idx - 1) % len(source)]
        next_value = source[(idx + 1) % len(source)]
        latent.append((0.5 * here) + (0.25 * prev_value) + (0.25 * next_value))
    return tuple(latent)


@dataclass(frozen=True)
class MarkovProgramRuntime:
    spec: MarkovRunSpec
    comparator_refs: tuple[str, ...]
    summary_width: int = 8

    def leaf_adapter(
        self,
        raw_input: Any,
        task_spec: MarkovRunSpec | None = None,
    ) -> MarkovSummarySurface:
        resolved_spec = task_spec or self.spec
        tokens = _coerce_token_sequence(raw_input)
        return MarkovSummarySurface(
            source="leaf",
            feature_vector=_token_features(
                tokens,
                width=int(self.summary_width),
                leaf_tokens=int(resolved_spec.leaf_tokens),
            ),
            root_share=int(resolved_spec.root_share),
            leaf_tokens=int(resolved_spec.leaf_tokens),
            profile=resolved_spec.profile.value,
            supervision_policy=resolved_spec.supervision_policy.value,
            metadata={"token_count": int(len(tokens))},
        )

    def merge_adapter(
        self,
        left_state: MarkovTreeState,
        right_state: MarkovTreeState,
        task_spec: MarkovRunSpec | None = None,
    ) -> MarkovSummarySurface:
        resolved_spec = task_spec or self.spec
        left_latent = list(left_state.latent_vector)
        right_latent = list(right_state.latent_vector)
        width = max(
            int(self.summary_width),
            len(left_latent) or 0,
            len(right_latent) or 0,
        )
        features: list[float] = []
        for idx in range(width):
            left_value = left_latent[idx % max(1, len(left_latent))] if left_latent else 0.0
            right_value = right_latent[idx % max(1, len(right_latent))] if right_latent else 0.0
            features.append((left_value + right_value) / 2.0)
        if features:
            features[0] = min(
                1.0,
                (float(left_state.count_estimate) + float(right_state.count_estimate))
                / float(max(1, 2 * int(resolved_spec.leaf_tokens))),
            )
        return MarkovSummarySurface(
            source="merge",
            feature_vector=tuple(float(value) for value in features[:width]),
            root_share=int(resolved_spec.root_share),
            leaf_tokens=int(resolved_spec.leaf_tokens),
            profile=resolved_spec.profile.value,
            supervision_policy=resolved_spec.supervision_policy.value,
            metadata={
                "left_count_estimate": float(left_state.count_estimate),
                "right_count_estimate": float(right_state.count_estimate),
            },
        )

    def g(
        self,
        summary_surface: MarkovSummarySurface,
        task_spec: MarkovRunSpec | None = None,
    ) -> MarkovTreeState:
        resolved_spec = task_spec or self.spec
        latent_vector = _mix_features(
            summary_surface.feature_vector,
            width=max(1, len(summary_surface.feature_vector)),
        )
        root_fraction = float(int(resolved_spec.root_share)) / 100.0
        count_estimate = float(resolved_spec.leaf_tokens) * (
            (0.35 * root_fraction)
            + (0.65 * sum(latent_vector[: min(4, len(latent_vector))]) / float(max(1, min(4, len(latent_vector)))))
        )
        return MarkovTreeState(
            summary_surface=summary_surface,
            latent_vector=latent_vector,
            count_estimate=float(count_estimate),
            metadata={
                "profile": resolved_spec.profile.value,
                "supervision_policy": resolved_spec.supervision_policy.value,
            },
        )

    def f(
        self,
        state: MarkovTreeState,
        task_spec: MarkovRunSpec | None = None,
    ) -> MarkovPrediction:
        resolved_spec = task_spec or self.spec
        return MarkovPrediction(
            count_estimate=float(state.count_estimate),
            state_width=int(len(state.latent_vector)),
            comparator_refs=tuple(str(value) for value in self.comparator_refs),
            profile=resolved_spec.profile.value,
            supervision_policy=resolved_spec.supervision_policy.value,
        )


MarkovUnifiedFGProgram = UnifiedFGProgram[
    Any,
    MarkovSummarySurface,
    MarkovTreeState,
    MarkovPrediction,
    MarkovRunSpec,
]


def build_markov_unified_fg_program(
    spec: MarkovRunSpec,
    *,
    binding: "MarkovUnifiedGBinding | None" = None,
) -> MarkovUnifiedFGProgram:
    resolved_binding = binding or resolve_markov_unified_g_binding(spec)
    summary_width = int(
        (resolved_binding.profile_fields.get("tree_theorem_count_dim", 8) or 8)
    )
    runtime = MarkovProgramRuntime(
        spec=spec,
        comparator_refs=tuple(str(value) for value in resolved_binding.contract.comparator_refs),
        summary_width=max(4, summary_width),
    )
    return UnifiedFGProgram(
        contract=resolved_binding.contract,
        leaf_adapter=runtime.leaf_adapter,
        merge_adapter=runtime.merge_adapter,
        g=runtime.g,
        f=runtime.f,
        decode=lambda state, _task_spec=None: state.to_dict(),
        runtime=runtime,
    )


@dataclass(frozen=True)
class MarkovUnifiedGBinding:
    spec: MarkovRunSpec
    contract: UnifiedGContract
    profile_fields: Mapping[str, Any]
    budget_fields: Mapping[str, Any]

    def apply(self, config: OPSCountConfig) -> OPSCountConfig:
        merged = {
            **asdict(config),
            **dict(self.profile_fields),
            **dict(self.budget_fields),
        }
        return OPSCountConfig(**merged)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract.to_dict(),
            "profile_fields": dict(self.profile_fields),
            "budget_fields": dict(self.budget_fields),
        }

    def to_program(self) -> MarkovUnifiedFGProgram:
        return build_markov_unified_fg_program(self.spec, binding=self)


def resolve_markov_unified_g_binding(spec: MarkovRunSpec) -> MarkovUnifiedGBinding:
    return MarkovUnifiedGBinding(
        spec=spec,
        contract=_markov_contract(spec),
        profile_fields=dict(profile_overrides(spec.profile)),
        budget_fields=dict(supervision_budget_fields(spec)),
    )


def resolve_markov_unified_fg_program(spec: MarkovRunSpec) -> MarkovUnifiedFGProgram:
    binding = resolve_markov_unified_g_binding(spec)
    return binding.to_program()
