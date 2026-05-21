from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Mapping, TypeVar

from treepo._research.unified_g_v1.core.specs import UnifiedFGSpec


RawInputT = TypeVar("RawInputT")
GInputT = TypeVar("GInputT")
StateT = TypeVar("StateT")
OutputT = TypeVar("OutputT")
TaskSpecT = TypeVar("TaskSpecT")


@dataclass(frozen=True)
class UnifiedGSurface:
    """Describes the shared summary surface for a Unified-G program."""

    raw_input_kind: str
    g_input_kind: str
    state_kind: str
    output_kind: str
    task_spec_kind: str = "generic"
    backend_family: str = "generic"
    shared_g: bool = True
    shared_f: bool = True

    def __post_init__(self) -> None:
        if not bool(self.shared_g):
            raise ValueError("Unified-G V1 requires a single shared g across leaves and merges")


@dataclass(frozen=True)
class UnifiedGContract:
    """Stable metadata contract for a backend that obeys the shared-g invariant."""

    name: str
    surface: UnifiedGSurface
    leaf_adapter_name: str
    merge_adapter_name: str
    g_name: str = "g"
    f_name: str = "f"
    decode_name: str | None = None
    comparator_refs: tuple[str, ...] = ()
    notes: str = ""
    program_spec: UnifiedFGSpec | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        program_payload = self.program_spec.to_dict() if self.program_spec is not None else None
        return {
            "name": self.name,
            "surface": {
                "raw_input_kind": self.surface.raw_input_kind,
                "g_input_kind": self.surface.g_input_kind,
                "state_kind": self.surface.state_kind,
                "output_kind": self.surface.output_kind,
                "task_spec_kind": self.surface.task_spec_kind,
                "backend_family": self.surface.backend_family,
                "shared_g": bool(self.surface.shared_g),
                "shared_f": bool(self.surface.shared_f),
            },
            "leaf_adapter_name": self.leaf_adapter_name,
            "merge_adapter_name": self.merge_adapter_name,
            "g_name": self.g_name,
            "f_name": self.f_name,
            "decode_name": self.decode_name,
            "comparator_refs": [str(value) for value in self.comparator_refs],
            "notes": self.notes,
            "program_spec": program_payload,
            "program_family": None if program_payload is None else program_payload["program_family"],
            "space_kind": None if program_payload is None else program_payload["space_kind"],
            "g_learner_kind": None if program_payload is None else program_payload["g_learner_kind"],
            "f_learner_kind": None if program_payload is None else program_payload["f_learner_kind"],
            "feature_dim": None if program_payload is None else program_payload["feature_dim"],
            "operator_width": None if program_payload is None else program_payload["operator_width"],
            "tokenizer_or_adapter_id": (
                None if program_payload is None else program_payload["tokenizer_or_adapter_id"]
            ),
            "extra": dict(self.extra),
        }


@dataclass
class UnifiedGProgram(Generic[RawInputT, GInputT, StateT, OutputT, TaskSpecT]):
    """Executable unified-g program.

    The invariant is structural: both leaf and merge paths must land on the
    same ``GInput`` surface, and a single shared ``g`` maps that surface to the
    program state.
    """

    contract: UnifiedGContract
    leaf_adapter: Callable[[RawInputT, TaskSpecT | None], GInputT]
    merge_adapter: Callable[[StateT, StateT, TaskSpecT | None], GInputT]
    g: Callable[[GInputT, TaskSpecT | None], StateT]
    f: Callable[[StateT, TaskSpecT | None], OutputT] | None = None
    decode: Callable[[StateT, TaskSpecT | None], Any] | None = None
    runtime: Any = None

    def __post_init__(self) -> None:
        if not bool(self.contract.surface.shared_g):
            raise ValueError("UnifiedGProgram requires contract.surface.shared_g=True")

    def leaf_g_input(
        self,
        raw_input: RawInputT,
        task_spec: TaskSpecT | None = None,
    ) -> GInputT:
        return self.leaf_adapter(raw_input, task_spec)

    def merge_g_input(
        self,
        left_state: StateT,
        right_state: StateT,
        task_spec: TaskSpecT | None = None,
    ) -> GInputT:
        return self.merge_adapter(left_state, right_state, task_spec)

    def leaf_state(
        self,
        raw_input: RawInputT,
        task_spec: TaskSpecT | None = None,
    ) -> StateT:
        return self.g(self.leaf_g_input(raw_input, task_spec), task_spec)

    def merge_state(
        self,
        left_state: StateT,
        right_state: StateT,
        task_spec: TaskSpecT | None = None,
    ) -> StateT:
        return self.g(self.merge_g_input(left_state, right_state, task_spec), task_spec)

    def predict(
        self,
        state: StateT,
        task_spec: TaskSpecT | None = None,
    ) -> OutputT:
        if self.f is None:
            raise ValueError(
                f"{self.contract.name} does not expose an f head; "
                "use leaf_state/merge_state/render_state instead."
            )
        return self.f(state, task_spec)

    def predict_from_leaf(
        self,
        raw_input: RawInputT,
        task_spec: TaskSpecT | None = None,
    ) -> OutputT:
        return self.predict(self.leaf_state(raw_input, task_spec), task_spec)

    def render_state(
        self,
        state: StateT,
        task_spec: TaskSpecT | None = None,
    ) -> Any:
        if self.decode is None:
            return state
        return self.decode(state, task_spec)

    def render_leaf(
        self,
        raw_input: RawInputT,
        task_spec: TaskSpecT | None = None,
    ) -> Any:
        return self.render_state(self.leaf_state(raw_input, task_spec), task_spec)

    def render_merge(
        self,
        left_state: StateT,
        right_state: StateT,
        task_spec: TaskSpecT | None = None,
    ) -> Any:
        return self.render_state(self.merge_state(left_state, right_state, task_spec), task_spec)


@dataclass
class UnifiedFGProgram(UnifiedGProgram[RawInputT, GInputT, StateT, OutputT, TaskSpecT]):
    """Executable unified-f/g program.

    This is the stricter version of the shared-g contract: callers can rely on
    a concrete readout head ``f`` in addition to the shared summary function
    ``g``.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if not bool(self.contract.surface.shared_f):
            raise ValueError("UnifiedFGProgram requires contract.surface.shared_f=True")
        if self.f is None:
            raise ValueError("UnifiedFGProgram requires a non-null f head")

    def readout(
        self,
        state: StateT,
        task_spec: TaskSpecT | None = None,
    ) -> OutputT:
        return self.predict(state, task_spec)
