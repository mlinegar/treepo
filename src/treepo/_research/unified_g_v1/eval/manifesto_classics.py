"""The three canonical Manifesto RILE smoke examples.

These three UK party manifestos span the left-right spectrum and the
1983-1997 period. The historical smoke harness reported individual scores
on these three as a quick sanity check:

    manifesto_id   | pred | expert | gap | leaves

Keep the set small and fixed so results are directly comparable across runs.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassicExample:
    manifesto_id: str
    label: str
    expert_rile: float


CLASSIC_EXAMPLES: tuple[ClassicExample, ...] = (
    ClassicExample("51320_198306", "UK Labour 1983",       -39.2),
    ClassicExample("51620_198306", "UK Conservative 1983",  29.0),
    ClassicExample("51320_199705", "UK Labour 1997",         8.1),
)


def classic_doc_ids() -> list[str]:
    return [ex.manifesto_id for ex in CLASSIC_EXAMPLES]


def classic_expert_targets() -> dict[str, float]:
    return {ex.manifesto_id: float(ex.expert_rile) for ex in CLASSIC_EXAMPLES}
