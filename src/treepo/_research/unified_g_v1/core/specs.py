from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


VALID_SPACE_KINDS = frozenset(
    {
        "text",
        "token_id_sequence",
        "embedding_sequence",
        "numeric_sequence",
    }
)
VALID_PARTITION_UNITS = frozenset({"tokens", "windows", "raw_sequence"})
VALID_LAYOUT_KINDS = frozenset({"scalar_seq", "vector_seq", "matrix_seq", "text_seq"})
VALID_LEARNER_KINDS = frozenset({"llm", "fno", "mlp", "linear_head"})


def _normalize_token(value: Any, valid_values: frozenset[str], *, field_name: str) -> str:
    rendered = str(value or "").strip().lower()
    if rendered not in valid_values:
        raise ValueError(f"unsupported {field_name}={value!r}")
    return rendered


def build_program_family(
    *,
    space_kind: str,
    g_learner_kind: str,
    f_learner_kind: str,
) -> str:
    return "__".join(
        [
            _normalize_token(space_kind, VALID_SPACE_KINDS, field_name="space_kind"),
            _normalize_token(g_learner_kind, VALID_LEARNER_KINDS, field_name="g_learner_kind"),
            _normalize_token(f_learner_kind, VALID_LEARNER_KINDS, field_name="f_learner_kind"),
        ]
    )


@dataclass(frozen=True)
class SpaceSpec:
    space_kind: str
    partition_unit: str
    feature_dim: int
    tokenizer_or_adapter_id: str
    layout_kind: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "space_kind",
            _normalize_token(self.space_kind, VALID_SPACE_KINDS, field_name="space_kind"),
        )
        object.__setattr__(
            self,
            "partition_unit",
            _normalize_token(
                self.partition_unit,
                VALID_PARTITION_UNITS,
                field_name="partition_unit",
            ),
        )
        object.__setattr__(
            self,
            "layout_kind",
            _normalize_token(self.layout_kind, VALID_LAYOUT_KINDS, field_name="layout_kind"),
        )
        object.__setattr__(self, "feature_dim", max(0, int(self.feature_dim)))
        object.__setattr__(self, "tokenizer_or_adapter_id", str(self.tokenizer_or_adapter_id or ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "space_kind": self.space_kind,
            "partition_unit": self.partition_unit,
            "feature_dim": int(self.feature_dim),
            "tokenizer_or_adapter_id": self.tokenizer_or_adapter_id,
            "layout_kind": self.layout_kind,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class LearnerSpec:
    learner_kind: str
    shared_g: bool = True
    shared_f: bool = True
    width: int | None = None
    n_modes: int | None = None
    runtime_backend: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "learner_kind",
            _normalize_token(
                self.learner_kind,
                VALID_LEARNER_KINDS,
                field_name="learner_kind",
            ),
        )
        object.__setattr__(self, "shared_g", bool(self.shared_g))
        object.__setattr__(self, "shared_f", bool(self.shared_f))
        object.__setattr__(
            self,
            "width",
            None if self.width is None else max(1, int(self.width)),
        )
        object.__setattr__(
            self,
            "n_modes",
            None if self.n_modes is None else max(1, int(self.n_modes)),
        )
        object.__setattr__(self, "runtime_backend", str(self.runtime_backend or ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "learner_kind": self.learner_kind,
            "shared_g": bool(self.shared_g),
            "shared_f": bool(self.shared_f),
            "width": None if self.width is None else int(self.width),
            "n_modes": None if self.n_modes is None else int(self.n_modes),
            "runtime_backend": self.runtime_backend,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class UnifiedFGSpec:
    space: SpaceSpec
    g_learner: LearnerSpec
    f_learner: LearnerSpec
    leaf_space_adapter: str
    merge_space_adapter: str
    task_head: str
    program_family: str | None = None
    notes: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        program_family = self.program_family or build_program_family(
            space_kind=self.space.space_kind,
            g_learner_kind=self.g_learner.learner_kind,
            f_learner_kind=self.f_learner.learner_kind,
        )
        object.__setattr__(self, "program_family", str(program_family))
        object.__setattr__(self, "leaf_space_adapter", str(self.leaf_space_adapter or ""))
        object.__setattr__(self, "merge_space_adapter", str(self.merge_space_adapter or ""))
        object.__setattr__(self, "task_head", str(self.task_head or ""))
        object.__setattr__(self, "notes", str(self.notes or ""))

    @property
    def feature_dim(self) -> int:
        return int(self.space.feature_dim)

    @property
    def operator_width(self) -> int | None:
        return (
            int(self.f_learner.width)
            if self.f_learner.width is not None
            else (
                int(self.g_learner.width)
                if self.g_learner.width is not None
                else None
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_family": str(self.program_family),
            "space": self.space.to_dict(),
            "g_learner": self.g_learner.to_dict(),
            "f_learner": self.f_learner.to_dict(),
            "leaf_space_adapter": self.leaf_space_adapter,
            "merge_space_adapter": self.merge_space_adapter,
            "task_head": self.task_head,
            "space_kind": self.space.space_kind,
            "g_learner_kind": self.g_learner.learner_kind,
            "f_learner_kind": self.f_learner.learner_kind,
            "feature_dim": int(self.space.feature_dim),
            "operator_width": (
                None if self.operator_width is None else int(self.operator_width)
            ),
            "tokenizer_or_adapter_id": self.space.tokenizer_or_adapter_id,
            "notes": self.notes,
            "extra": dict(self.extra),
        }


def build_llm_text_program_spec(
    *,
    tokenizer_or_adapter_id: str = "text/plain",
) -> UnifiedFGSpec:
    return UnifiedFGSpec(
        space=SpaceSpec(
            space_kind="text",
            partition_unit="tokens",
            feature_dim=0,
            tokenizer_or_adapter_id=str(tokenizer_or_adapter_id),
            layout_kind="text_seq",
        ),
        g_learner=LearnerSpec(
            learner_kind="llm",
            shared_g=True,
            shared_f=True,
            runtime_backend="dspy_text",
        ),
        f_learner=LearnerSpec(
            learner_kind="llm",
            shared_g=True,
            shared_f=True,
            runtime_backend="dspy_text",
        ),
        leaf_space_adapter="identity_text_adapter",
        merge_space_adapter="format_merge_input",
        task_head="text_summary",
        notes="LLM text path with both g and f on the text summary surface.",
    )


def build_markov_token_fno_program_spec(
    *,
    feature_dim: int = 128,
    tokenizer_or_adapter_id: str = "markov_token_ids",
    operator_width: int = 128,
    operator_modes: int = 8,
) -> UnifiedFGSpec:
    return UnifiedFGSpec(
        space=SpaceSpec(
            space_kind="token_id_sequence",
            partition_unit="tokens",
            feature_dim=int(feature_dim),
            tokenizer_or_adapter_id=str(tokenizer_or_adapter_id),
            layout_kind="scalar_seq",
        ),
        g_learner=LearnerSpec(
            learner_kind="fno",
            shared_g=True,
            shared_f=True,
            width=int(operator_width),
            n_modes=int(operator_modes),
            runtime_backend="markov_neural_operator",
        ),
        f_learner=LearnerSpec(
            learner_kind="fno",
            shared_g=True,
            shared_f=True,
            width=int(operator_width),
            n_modes=int(operator_modes),
            runtime_backend="markov_neural_operator",
        ),
        leaf_space_adapter="token_sequence_to_summary_surface",
        merge_space_adapter="child_state_pair_to_summary_surface",
        task_head="root_count_prediction",
        notes="Trusted unified-g Markov token-sequence FNO/FNO route.",
    )


def build_token_sequence_fno_program_spec(
    *,
    feature_dim: int = 128,
    tokenizer_or_adapter_id: str = "token_ids",
    operator_width: int = 128,
    operator_modes: int = 8,
) -> UnifiedFGSpec:
    return UnifiedFGSpec(
        space=SpaceSpec(
            space_kind="token_id_sequence",
            partition_unit="tokens",
            feature_dim=int(feature_dim),
            tokenizer_or_adapter_id=str(tokenizer_or_adapter_id),
            layout_kind="scalar_seq",
        ),
        g_learner=LearnerSpec(
            learner_kind="fno",
            shared_g=True,
            shared_f=True,
            width=int(operator_width),
            n_modes=int(operator_modes),
            runtime_backend="token_sequence_neural_operator",
        ),
        f_learner=LearnerSpec(
            learner_kind="fno",
            shared_g=True,
            shared_f=True,
            width=int(operator_width),
            n_modes=int(operator_modes),
            runtime_backend="token_sequence_neural_operator",
        ),
        leaf_space_adapter="token_sequence_to_summary_surface",
        merge_space_adapter="child_state_pair_to_summary_surface",
        task_head="scalar_readout",
        notes="Token-sequence path with shared FNO g and FNO f.",
    )


def build_embedding_sequence_fno_program_spec(
    *,
    feature_dim: int,
    tokenizer_or_adapter_id: str,
    operator_width: int,
    operator_modes: int = 8,
) -> UnifiedFGSpec:
    return UnifiedFGSpec(
        space=SpaceSpec(
            space_kind="embedding_sequence",
            partition_unit="tokens",
            feature_dim=int(feature_dim),
            tokenizer_or_adapter_id=str(tokenizer_or_adapter_id),
            layout_kind="vector_seq",
        ),
        g_learner=LearnerSpec(
            learner_kind="fno",
            shared_g=True,
            shared_f=True,
            width=int(operator_width),
            n_modes=int(operator_modes),
            runtime_backend="embedding_neural_operator",
        ),
        f_learner=LearnerSpec(
            learner_kind="fno",
            shared_g=True,
            shared_f=True,
            width=int(operator_width),
            n_modes=int(operator_modes),
            runtime_backend="embedding_neural_operator",
        ),
        leaf_space_adapter="ordered_embedding_sequence_adapter",
        merge_space_adapter="state_pair_sequence_adapter",
        task_head="scalar_sequence_readout",
        notes="Ordered embedding-sequence path with FNO g and FNO f.",
    )


def build_semantic_embedding_program_spec(
    *,
    feature_dim: int = 0,
    tokenizer_or_adapter_id: str = "semantic_memory",
) -> UnifiedFGSpec:
    return UnifiedFGSpec(
        space=SpaceSpec(
            space_kind="embedding_sequence",
            partition_unit="windows",
            feature_dim=int(feature_dim),
            tokenizer_or_adapter_id=str(tokenizer_or_adapter_id),
            layout_kind="vector_seq",
        ),
        g_learner=LearnerSpec(
            learner_kind="linear_head",
            shared_g=False,
            shared_f=True,
            runtime_backend="semantic_retrieval",
        ),
        f_learner=LearnerSpec(
            learner_kind="linear_head",
            shared_g=False,
            shared_f=True,
            runtime_backend="semantic_retrieval",
        ),
        leaf_space_adapter="semantic_neighbor_lookup",
        merge_space_adapter="identity_merge_passthrough",
        task_head="retrieval_weighted_score",
        notes="Legacy semantic embedding retrieval preserved as a canonical program family.",
    )


def build_ctreepo_program_spec(
    *,
    feature_dim: int = 0,
    tokenizer_or_adapter_id: str = "embedding_tree",
) -> UnifiedFGSpec:
    return UnifiedFGSpec(
        space=SpaceSpec(
            space_kind="embedding_sequence",
            partition_unit="windows",
            feature_dim=int(feature_dim),
            tokenizer_or_adapter_id=str(tokenizer_or_adapter_id),
            layout_kind="vector_seq",
        ),
        g_learner=LearnerSpec(
            learner_kind="mlp",
            shared_g=True,
            shared_f=True,
            runtime_backend="ctreepo",
        ),
        f_learner=LearnerSpec(
            learner_kind="mlp",
            shared_g=True,
            shared_f=True,
            runtime_backend="ctreepo",
        ),
        leaf_space_adapter="embedding_tree_leaf_projector",
        merge_space_adapter="embedding_tree_gated_merge",
        task_head="ctreepo_readout",
        notes="Legacy CTreePO recast as an embedding-sequence program family.",
    )


def build_mergeable_sketch_program_spec(
    *,
    feature_dim: int = 0,
    tokenizer_or_adapter_id: str = "mergeable_sketch",
) -> UnifiedFGSpec:
    return UnifiedFGSpec(
        space=SpaceSpec(
            space_kind="numeric_sequence",
            partition_unit="raw_sequence",
            feature_dim=int(feature_dim),
            tokenizer_or_adapter_id=str(tokenizer_or_adapter_id),
            layout_kind="matrix_seq",
        ),
        g_learner=LearnerSpec(
            learner_kind="mlp",
            shared_g=False,
            shared_f=True,
            runtime_backend="mergeable_sketch",
        ),
        f_learner=LearnerSpec(
            learner_kind="linear_head",
            shared_g=False,
            shared_f=True,
            runtime_backend="mergeable_sketch",
        ),
        leaf_space_adapter="mergeable_window_matrix_adapter",
        merge_space_adapter="mergeable_numeric_merge_adapter",
        task_head="mergeable_scalar_head",
        notes="Legacy mergeable sketch preserved as a numeric/matrix sequence program family.",
    )


_LEGACY_ALIAS_BUILDERS = {
    "llm": build_llm_text_program_spec,
    "embedding": build_semantic_embedding_program_spec,
    "ctreepo": build_ctreepo_program_spec,
    "mergeable_sketch": build_mergeable_sketch_program_spec,
}


def resolve_program_spec_alias(
    value: Any,
) -> UnifiedFGSpec | None:
    rendered = str(value or "").strip().lower()
    if not rendered or rendered in {"auto", "ensemble"}:
        return None
    builder = _LEGACY_ALIAS_BUILDERS.get(rendered)
    if builder is not None:
        return builder()
    for candidate in (
        build_llm_text_program_spec(),
        build_semantic_embedding_program_spec(),
        build_ctreepo_program_spec(),
        build_mergeable_sketch_program_spec(),
        build_markov_token_fno_program_spec(),
    ):
        if str(candidate.program_family) == rendered:
            return candidate
    return None
