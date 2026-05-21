from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol


class DSPyTaskAdapter(Protocol):
    """Problem-specific hooks used by the generic DSPy method family."""

    def summary_instructions(self) -> str: ...

    def full_doc_f_task_context(self) -> str: ...

    def default_f_init_path(self, *, repo_root: Path) -> Optional[Path]: ...

    def new_scorer(self, *, max_output_tokens: int): ...


@dataclass(frozen=True)
class ManifestoDimensionDSPyAdapter:
    """Manifesto/Benoit implementation of the generic DSPy task adapter."""

    dimension: str
    f_init_path: Optional[str] = None

    def summary_instructions(self) -> str:
        from treepo._research.tasks.manifesto.dimensions import PolicyDimension, get_preservation_rubric
        from treepo._research.tasks.manifesto.scoring_contexts import get_scoring_context

        dim = PolicyDimension(self.dimension)
        rubric = str(get_preservation_rubric(dim) or "").strip()
        context = str(get_scoring_context(dim) or "").strip()
        return (
            "Generate a summary of the given political text that preserves all "
            f"information relevant to the {dim.value} dimension of Benoit's 1-7 "
            "policy scale. The summary will later be scored by a separate "
            "scoring model against expert annotations, so it must preserve "
            "every signal that distinguishes high vs low positions on this "
            "dimension.\n\n"
            f"{context}\n\n{rubric}"
        )

    def full_doc_f_task_context(self) -> str:
        from treepo._research.tasks.manifesto.benoit_scoring_contexts import get_benoit_scoring_context
        from treepo._research.tasks.manifesto.dimensions import PolicyDimension

        base = get_benoit_scoring_context(PolicyDimension(str(self.dimension)))
        return (
            base
            + "\n\nFor this supervised f-training benchmark, always return your best numeric estimate "
            "on the 1-7 scale. Do not return NA. If the document is short, partial, or ambiguous, "
            "infer the most likely expert-mean score from the available text and the training examples. "
            "Return only one number, allowing decimals when appropriate."
        )

    def default_f_init_path(self, *, repo_root: Path) -> Optional[Path]:
        if self.f_init_path is not None:
            if not self.f_init_path:
                return None
            return Path(self.f_init_path)
        candidate = (
            repo_root
            / "outputs"
            / "phase1_gepa_v2_rank"
            / str(self.dimension)
            / "optimized_scorer.json"
        )
        return candidate if candidate.exists() else None

    def new_scorer(self, *, max_output_tokens: int):
        from treepo._research.tasks.manifesto.dimension_scorer import DimensionScorer
        from treepo._research.tasks.manifesto.dimensions import PolicyDimension, get_dimension

        dim_spec = get_dimension(PolicyDimension(self.dimension))
        return DimensionScorer(
            dimension=dim_spec,
            max_output_tokens=int(max_output_tokens),
        )


def dspy_task_adapter(
    *,
    problem_id: str,
    dimension: str,
    f_init_path: Optional[str] = None,
) -> DSPyTaskAdapter:
    normalized = str(problem_id or "manifesto_benoit").strip().lower()
    if normalized in {"manifesto", "manifesto_benoit", "benoit"}:
        return ManifestoDimensionDSPyAdapter(
            dimension=str(dimension),
            f_init_path=f_init_path,
        )
    raise ValueError(f"unsupported DSPy problem_id: {problem_id!r}")
