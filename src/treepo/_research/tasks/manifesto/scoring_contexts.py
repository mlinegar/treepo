"""
1-7 scoring task contexts per Benoit policy dimension.

A "scoring context" is the `task_context` argument passed to the scoring
signature (RILEScoreSignature-style). It describes the dimension, the 1-7
scale, end-scale anchor text, and the expert-scientist framing Benoit
adopted ("You are an expert political scientist with a PhD...").

Authoritative anchor text is from Benoit et al. 2026 Table 2 (p. 4 of the
main PDF). Full end-scale vignettes live in Supporting Information
Appendix A4 (pp. 6-11 of their SI); dimensions where we have not yet
transcribed those vignettes are marked with a TODO line the pilot script
should surface to the user before running.
"""

from __future__ import annotations

from typing import Dict

from .dimensions import PolicyDimension


_EXPERT_FRAMING = (
    "You are an expert political scientist with a PhD in political science."
    " Think carefully about your answer."
)


def _context(
    dimension_name: str,
    anchor_low: str,
    anchor_high: str,
    *,
    vignettes: str = "",
) -> str:
    return (
        f"{_EXPERT_FRAMING}\n\n"
        f"Task: Score the following manifesto summary on the {dimension_name} dimension"
        f" using a 1-7 integer scale.\n\n"
        f"Scale endpoints:\n"
        f"  1 = {anchor_low}\n"
        f"  7 = {anchor_high}\n"
        f"  4 = neutral / balanced between endpoints\n\n"
        f"{vignettes}"
        f"If the summary does not provide enough information to score this dimension,"
        f" return 'NA' rather than guessing.\n\n"
        f"Return an integer 1-7 (or 'NA') and brief reasoning."
    )


ECONOMIC_SCORING_CONTEXT = _context(
    "economic left-right",
    "Strongly favors improving public services",
    "Strongly favors reducing taxes",
    vignettes="",  # TODO: transcribe Benoit SI A4 p. 6 vignettes
)


SOCIAL_SCORING_CONTEXT = _context(
    "social liberalism",
    "Strongly supports liberal social policies",
    "Strongly opposes liberal social policies",
    vignettes="",  # TODO: Benoit SI A4 p. 7
)


IMMIGRATION_SCORING_CONTEXT = _context(
    "immigration policy",
    "Strongly opposes tough immigration policy",
    "Strongly favors tough immigration policy",
    vignettes="",  # TODO: Benoit SI A4 p. 8
)


EU_SCORING_CONTEXT = _context(
    "EU integration",
    "Strongly opposed to European integration",
    "Strongly in favor of European integration",
    vignettes="",  # TODO: Benoit SI A4 p. 9
)


ENVIRONMENT_SCORING_CONTEXT = _context(
    "environmental policy",
    "Environmental protection even at cost of economic growth",
    "Economic growth even at cost of environmental protection",
    vignettes="",  # TODO: Benoit SI A4 p. 10
)


DECENTRALIZATION_SCORING_CONTEXT = _context(
    "political decentralization",
    "Strongly favors political decentralization to regions/localities",
    "Strongly opposes political decentralization",
    vignettes="",  # TODO: Benoit SI A4 p. 11
)


SCORING_CONTEXTS: Dict[PolicyDimension, str] = {
    PolicyDimension.ECONOMIC: ECONOMIC_SCORING_CONTEXT,
    PolicyDimension.SOCIAL: SOCIAL_SCORING_CONTEXT,
    PolicyDimension.IMMIGRATION: IMMIGRATION_SCORING_CONTEXT,
    PolicyDimension.EU: EU_SCORING_CONTEXT,
    PolicyDimension.ENVIRONMENT: ENVIRONMENT_SCORING_CONTEXT,
    PolicyDimension.DECENTRALIZATION: DECENTRALIZATION_SCORING_CONTEXT,
}


def get_scoring_context(dimension: PolicyDimension) -> str:
    return SCORING_CONTEXTS[dimension]
