"""
Six policy dimensions from Benoit et al. (2026 AJPS).

LLM outputs use the common 1-7 axis from Benoit's prompts. The released expert
benchmark means are not uniformly stored on that axis; see
src/tasks/manifesto/expert_scale.py for the explicit raw-vs-normalized target
conversion used by our calibration and supervised-training paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict

from treepo._research.tasks.base import ScaleDefinition


class PolicyDimension(str, Enum):
    ECONOMIC = "economic"
    SOCIAL = "social"
    IMMIGRATION = "immigration"
    EU = "eu"
    ENVIRONMENT = "environment"
    DECENTRALIZATION = "decentralization"


@dataclass(frozen=True)
class DimensionSpec:
    """A single Benoit policy dimension on the common 1-7 scale."""

    dimension: PolicyDimension
    anchor_low: str
    anchor_high: str
    scale: ScaleDefinition
    ches_variable: str
    native_expert_scale: str
    benoit_issue_code: str  # as it appears in data_experts$issue / data_llms_all_*$issue


_SEVEN_POINT = lambda name, desc: ScaleDefinition(
    name=name,
    min_value=1.0,
    max_value=7.0,
    description=desc,
    higher_is_better=True,
    neutral_value=4.0,
)


BENOIT_DIMENSIONS: Dict[PolicyDimension, DimensionSpec] = {
    PolicyDimension.ECONOMIC: DimensionSpec(
        dimension=PolicyDimension.ECONOMIC,
        anchor_low="Strongly favors improving public services",
        anchor_high="Strongly favors reducing taxes",
        scale=_SEVEN_POINT(
            "economic",
            "Economic left-right: public services vs reducing taxes (Benoit Table 2).",
        ),
        ches_variable="lrecon",
        native_expert_scale="released expert_mean is survey-side/raw; derive 1-7 via expert_scale",
        benoit_issue_code="taxspend",
    ),
    PolicyDimension.SOCIAL: DimensionSpec(
        dimension=PolicyDimension.SOCIAL,
        anchor_low="Strongly supports liberal social policies",
        anchor_high="Strongly opposes liberal social policies",
        scale=_SEVEN_POINT("social", "Social liberalism (Benoit Table 2)."),
        ches_variable="galtan",
        native_expert_scale="released expert_mean is survey-side/raw; derive 1-7 via expert_scale",
        benoit_issue_code="social",
    ),
    PolicyDimension.IMMIGRATION: DimensionSpec(
        dimension=PolicyDimension.IMMIGRATION,
        anchor_low="Strongly opposes tough immigration policy",
        anchor_high="Strongly favors tough immigration policy",
        scale=_SEVEN_POINT("immigration", "Immigration policy (Benoit Table 2)."),
        ches_variable="immigrate_policy",
        native_expert_scale="released expert_mean is survey-side/raw; derive 1-7 via expert_scale",
        benoit_issue_code="immigration",
    ),
    PolicyDimension.EU: DimensionSpec(
        dimension=PolicyDimension.EU,
        anchor_low="Strongly opposed to European integration",
        anchor_high="Strongly in favor of European integration",
        scale=_SEVEN_POINT("eu", "Party orientation toward EU (Benoit Table 2)."),
        ches_variable="eu_position",
        native_expert_scale="released expert_mean is effectively native 1-7; expert_scale clamps to 1-7",
        benoit_issue_code="eu",
    ),
    PolicyDimension.ENVIRONMENT: DimensionSpec(
        dimension=PolicyDimension.ENVIRONMENT,
        anchor_low="Environmental protection even at cost of growth",
        anchor_high="Economic growth even at cost of environment",
        scale=_SEVEN_POINT("environment", "Environment vs growth (Benoit Table 2)."),
        ches_variable="enviro",
        native_expert_scale="released expert_mean is survey-side/raw; derive 1-7 via expert_scale",
        benoit_issue_code="environment",
    ),
    PolicyDimension.DECENTRALIZATION: DimensionSpec(
        dimension=PolicyDimension.DECENTRALIZATION,
        anchor_low="Strongly favors political decentralization",
        anchor_high="Strongly opposes political decentralization",
        scale=_SEVEN_POINT("decentralization", "Political decentralization (Benoit Table 2)."),
        ches_variable="regions",
        native_expert_scale="released expert_mean is survey-side/raw; derive 1-7 via expert_scale. Known to have structural"
        " manifesto-vs-expert mismatch (Benoit §4.4).",
        benoit_issue_code="decentralization",
    ),
}


_BY_ISSUE_CODE: Dict[str, PolicyDimension] = {
    spec.benoit_issue_code: dim for dim, spec in BENOIT_DIMENSIONS.items()
}


def from_benoit_issue_code(code: str) -> PolicyDimension:
    """Map Benoit's issue string (e.g. 'taxspend') to our enum."""
    return _BY_ISSUE_CODE[code]


def get_preservation_rubric(dim: PolicyDimension) -> str:
    """Return the per-dimension summarization rubric (what to preserve)."""
    from . import rubrics
    table = {
        PolicyDimension.ECONOMIC: rubrics.ECONOMIC_RUBRIC,
        PolicyDimension.SOCIAL: rubrics.SOCIAL_RUBRIC,
        PolicyDimension.IMMIGRATION: rubrics.IMMIGRATION_RUBRIC,
        PolicyDimension.EU: rubrics.EU_RUBRIC,
        PolicyDimension.ENVIRONMENT: rubrics.ENVIRONMENT_RUBRIC,
        PolicyDimension.DECENTRALIZATION: rubrics.DECENTRALIZATION_RUBRIC,
    }
    return table[dim]


def get_joint_rubric() -> str:
    """Return the union-rubric for summaries covering all 6 dimensions."""
    from . import rubrics
    return rubrics.JOINT_RUBRIC


def get_dimension(dim: PolicyDimension) -> DimensionSpec:
    return BENOIT_DIMENSIONS[dim]
