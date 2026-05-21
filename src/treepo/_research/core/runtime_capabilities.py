"""Declarative runtime capability metadata for theorem and empirical families."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple

from treepo._research.core.engines import EngineSurface, EngineType


class RuntimeClaimStatus(Enum):
    """Claim/evidence status of a family on a particular engine surface."""

    CLAIM_BEARING = "claim_bearing"
    RESEARCH_ONLY = "research_only"
    INFRASTRUCTURE_ONLY = "infrastructure_only"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class FamilyRuntimeCapability:
    """Runtime support matrix kept separate from theorem/operator capability reports."""

    family_name: str
    supported_engine_surfaces: Tuple[EngineSurface, ...]
    recommended_engines: Tuple[EngineType, ...] = field(default_factory=tuple)
    claim_status_by_surface: Mapping[EngineSurface, RuntimeClaimStatus] = field(
        default_factory=dict
    )
    requires_external_engine: bool = True
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def claim_status(self, surface: EngineSurface | str) -> RuntimeClaimStatus:
        target = (
            surface
            if isinstance(surface, EngineSurface)
            else EngineSurface(str(surface).strip().lower())
        )
        return self.claim_status_by_surface.get(target, RuntimeClaimStatus.NOT_APPLICABLE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_name": self.family_name,
            "supported_engine_surfaces": tuple(
                surface.value for surface in self.supported_engine_surfaces
            ),
            "recommended_engines": tuple(engine.value for engine in self.recommended_engines),
            "claim_status_by_surface": {
                surface.value: status.value
                for surface, status in self.claim_status_by_surface.items()
            },
            "requires_external_engine": bool(self.requires_external_engine),
            "notes": tuple(self.notes),
        }


def default_family_runtime_capability(
    family_name: str,
    *,
    theorem_backed_symbolic: bool = False,
    recommended_engines: Sequence[EngineType] = (EngineType.VLLM, EngineType.SGLANG),
    notes: Sequence[str] = (),
) -> FamilyRuntimeCapability:
    """Conservative default for non-Markov/general families."""

    claim_status = {
        EngineSurface.CHAT_OPENAI: RuntimeClaimStatus.RESEARCH_ONLY,
        EngineSurface.EMBEDDING: RuntimeClaimStatus.RESEARCH_ONLY,
        EngineSurface.OPERATOR: RuntimeClaimStatus.RESEARCH_ONLY,
        EngineSurface.DIFFUSION_GENERATE: RuntimeClaimStatus.INFRASTRUCTURE_ONLY,
        EngineSurface.SYMBOLIC_EXACT: (
            RuntimeClaimStatus.CLAIM_BEARING
            if theorem_backed_symbolic
            else RuntimeClaimStatus.NOT_APPLICABLE
        ),
    }
    supported = [EngineSurface.CHAT_OPENAI, EngineSurface.EMBEDDING, EngineSurface.OPERATOR]
    if theorem_backed_symbolic:
        supported.append(EngineSurface.SYMBOLIC_EXACT)
    return FamilyRuntimeCapability(
        family_name=str(family_name),
        supported_engine_surfaces=tuple(supported),
        recommended_engines=tuple(recommended_engines),
        claim_status_by_surface=claim_status,
        requires_external_engine=not theorem_backed_symbolic,
        notes=tuple(notes)
        + (
            "General-family defaults keep chat engines available for empirical work.",
            "Embedding and operator surfaces are method-routing surfaces; they are not standalone scorers by default.",
            "Diffusion surfaces remain non-claim-bearing unless a theorem-backed operator is supplied explicitly.",
        ),
    )


def markov_family_runtime_capability(
    family_name: str,
    *,
    exact_family: Optional[str] = None,
    notes: Sequence[str] = (),
) -> FamilyRuntimeCapability:
    """Runtime support matrix for Markov families and exact counterexample lanes."""

    symbolic_claim = (
        RuntimeClaimStatus.CLAIM_BEARING if exact_family else RuntimeClaimStatus.NOT_APPLICABLE
    )
    supported = [
        EngineSurface.CHAT_OPENAI,
        EngineSurface.EMBEDDING,
        EngineSurface.OPERATOR,
        EngineSurface.DIFFUSION_GENERATE,
    ]
    if exact_family:
        supported.append(EngineSurface.SYMBOLIC_EXACT)
    return FamilyRuntimeCapability(
        family_name=str(family_name),
        supported_engine_surfaces=tuple(supported),
        recommended_engines=(
            (EngineType.SYMBOLIC_LOCAL, EngineType.SGLANG, EngineType.VLLM)
            if exact_family
            else (EngineType.VLLM, EngineType.SGLANG)
        ),
        claim_status_by_surface={
            EngineSurface.CHAT_OPENAI: RuntimeClaimStatus.RESEARCH_ONLY,
            EngineSurface.EMBEDDING: RuntimeClaimStatus.RESEARCH_ONLY,
            EngineSurface.OPERATOR: RuntimeClaimStatus.RESEARCH_ONLY,
            EngineSurface.DIFFUSION_GENERATE: RuntimeClaimStatus.INFRASTRUCTURE_ONLY,
            EngineSurface.SYMBOLIC_EXACT: symbolic_claim,
        },
        requires_external_engine=not bool(exact_family),
        notes=tuple(notes)
        + (
            "Markov exact/symbolic families remain the only claim-bearing runtime lane in the first pass.",
            "Off-the-shelf diffusion endpoints are tracked as orchestration or infrastructure surfaces unless a theorem-backed operator is supplied.",
        ),
    )


__all__ = [
    "FamilyRuntimeCapability",
    "RuntimeClaimStatus",
    "default_family_runtime_capability",
    "markov_family_runtime_capability",
]
