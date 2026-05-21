from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Type

from treepo._research.core.strategy import DSPyStrategy
from treepo._research.core.protocols import format_merge_input
from treepo._research.tasks.manifesto.lawstress_bootstrap_program import UnifiedG

from treepo._research.unified_g_v1.core.program import UnifiedFGProgram, UnifiedGContract, UnifiedGSurface
from treepo._research.unified_g_v1.core.specs import build_llm_text_program_spec


LLMTextUnifiedFGProgram = UnifiedFGProgram[str, str, str, str, str]
TextUnifiedGProgram = LLMTextUnifiedFGProgram


def build_llm_text_unified_fg_contract() -> UnifiedGContract:
    program_spec = build_llm_text_program_spec(tokenizer_or_adapter_id="cl100k_base")
    return UnifiedGContract(
        name="llm_text_unified_fg",
        surface=UnifiedGSurface(
            raw_input_kind="text",
            g_input_kind="text",
            state_kind="text_summary",
            output_kind="text_summary",
            task_spec_kind="rubric",
            backend_family=str(program_spec.program_family),
            shared_g=True,
            shared_f=True,
        ),
        leaf_adapter_name="identity_text_adapter",
        merge_adapter_name="format_merge_input",
        g_name="llm_summary_g",
        f_name="llm_readout_f",
        decode_name="identity_decode",
        notes=(
            "LLM-side text backend: both g and f stay on the text summary surface, "
            "with the same summary program used for leaves and merges."
        ),
        program_spec=program_spec,
        extra={
            "approach_kind": "llm_text",
            "summary_execution": "llm_text",
            "readout_execution": "llm_text",
        },
    )


def build_text_unified_g_contract() -> UnifiedGContract:
    return build_llm_text_unified_fg_contract()


@dataclass
class UnifiedGArtifact:
    """Stable string-to-string unified-g artifact wrapper."""

    module: Any
    module_cls: Type[Any] = UnifiedG

    @classmethod
    def baseline(cls) -> "UnifiedGArtifact":
        return cls(module=UnifiedG(), module_cls=UnifiedG)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        module_cls: Type[Any] | None = None,
    ) -> "UnifiedGArtifact":
        resolved_cls = module_cls or UnifiedG
        module = resolved_cls()
        module.load(str(path))
        return cls(module=module, module_cls=resolved_cls)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.module.save(str(output_path))
        return output_path

    def __call__(self, *, content: str, rubric: str) -> str:
        return str(self.module(content=str(content), rubric=str(rubric)) or "").strip()

    def summarize(self, content: str, rubric: str) -> str:
        return self(content=content, rubric=rubric)

    @property
    def contract(self) -> UnifiedGContract:
        return build_llm_text_unified_fg_contract()

    def to_program(self) -> LLMTextUnifiedFGProgram:
        return UnifiedFGProgram(
            contract=self.contract,
            leaf_adapter=lambda content, _rubric=None: str(content),
            merge_adapter=lambda left, right, _rubric=None: format_merge_input(
                str(left), str(right)
            ),
            g=lambda content, rubric=None: self.summarize(
                str(content),
                str(rubric or ""),
            ),
            f=lambda state, _rubric=None: str(state),
            decode=lambda state, _rubric=None: str(state),
            runtime={"artifact": self, "module": self.module},
        )


def resolve_text_unified_g_program(
    value: UnifiedGArtifact | LLMTextUnifiedFGProgram | None = None,
) -> LLMTextUnifiedFGProgram:
    if value is None:
        return UnifiedGArtifact.baseline().to_program()
    if isinstance(value, UnifiedGArtifact):
        return value.to_program()
    backend_family = str(value.contract.surface.backend_family)
    if backend_family not in {"text__llm__llm", "llm_text_unified_fg", "dspy_text_unified_g"}:
        raise ValueError(
            "expected a text Unified-G program for DSPy/LawStress usage; "
            f"got backend_family={backend_family!r}"
        )
    return value


def build_unified_dspy_strategy(
    artifact: UnifiedGArtifact | LLMTextUnifiedFGProgram | None = None,
) -> DSPyStrategy:
    if artifact is None:
        module = UnifiedGArtifact.baseline().module
    elif isinstance(artifact, UnifiedGArtifact):
        module = artifact.module
    else:
        runtime = artifact.runtime
        module = None
        if isinstance(runtime, Mapping):
            module = runtime.get("module")
        elif runtime is not None:
            module = getattr(runtime, "module", None)
        if module is None:
            raise ValueError(
                "build_unified_dspy_strategy requires a UnifiedGArtifact or a "
                "text Unified-G program whose runtime exposes the underlying DSPy module"
            )
    return DSPyStrategy(
        leaf_module=module,
        merge_module=None,
        unified_mode=True,
    )
