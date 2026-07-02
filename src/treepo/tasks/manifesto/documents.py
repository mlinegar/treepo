"""Small packaged Manifesto/RILE documents for examples and tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ManifestoQSentence:
    qid: str
    text: str
    code: str = ""
    score: float | None = None
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManifestoDocument:
    doc_id: str
    country: str
    party: str
    year: int
    text: str
    rile_label: float
    qsentences: Sequence[ManifestoQSentence] = field(default_factory=tuple)
    replication: str = "manifesto_rile"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManifestoLeaf:
    text: str
    qid: str
    code: str = ""
    score: float | None = None
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManifestoReplicationTree:
    doc_id: str
    text: str
    leaves: Sequence[ManifestoLeaf]
    metadata: Mapping[str, Any] = field(default_factory=dict)


DEFAULT_MANIFESTO_REPLICATIONS: tuple[ManifestoDocument, ...] = (
    ManifestoDocument(
        doc_id="gb_lab_1983_sample",
        country="GBR",
        party="Labour",
        year=1983,
        text="Jobs, public ownership, social services, and unilateral nuclear disarmament.",
        rile_label=-34.0,
        qsentences=(
            ManifestoQSentence(
                "gb_lab_1983_q1",
                "Expand public employment and welfare provision.",
                "per503",
                -40.0,
            ),
            ManifestoQSentence(
                "gb_lab_1983_q2",
                "Oppose nuclear weapons and prioritize peace.",
                "per106",
                -25.0,
            ),
        ),
        replication="cmp_rile_root_label",
    ),
    ManifestoDocument(
        doc_id="gb_con_1983_sample",
        country="GBR",
        party="Conservative",
        year=1983,
        text="Lower inflation, private enterprise, strong defence, and limits on union power.",
        rile_label=28.0,
        qsentences=(
            ManifestoQSentence(
                "gb_con_1983_q1",
                "Support private enterprise and market discipline.",
                "per401",
                35.0,
            ),
            ManifestoQSentence(
                "gb_con_1983_q2",
                "Maintain strong defence commitments.",
                "per104",
                20.0,
            ),
        ),
        replication="cmp_rile_root_label",
    ),
    ManifestoDocument(
        doc_id="gb_lab_1997_sample",
        country="GBR",
        party="Labour",
        year=1997,
        text="Fiscal discipline, education investment, constitutional reform, and public service renewal.",
        rile_label=-5.0,
        qsentences=(
            ManifestoQSentence(
                "gb_lab_1997_q1",
                "Invest in education and public services.",
                "per506",
                -12.0,
            ),
            ManifestoQSentence(
                "gb_lab_1997_q2",
                "Commit to fiscal discipline and stable growth.",
                "per414",
                8.0,
            ),
        ),
        replication="cmp_rile_root_label",
    ),
)


__all__ = [
    "DEFAULT_MANIFESTO_REPLICATIONS",
    "ManifestoDocument",
    "ManifestoLeaf",
    "ManifestoQSentence",
    "ManifestoReplicationTree",
]
