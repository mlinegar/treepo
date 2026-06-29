"""Manifesto/RILE replication fixtures for methods examples.

The central shape is: one document tree has a root-level document label and
optional qsentence guidance for training or prompting unified ``g``. Real CMP
or manifesto corpora can be adapted into these records downstream; this module
keeps only a tiny dependency-light fixture and schema helpers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from treepo.tasks.manifesto.rile import clamp_rile


@dataclass(frozen=True)
class ManifestoQSentence:
    qid: str
    text: str
    code: str = ""
    guidance_score: float | None = None
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
    guidance_score: float | None = None
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
            ManifestoQSentence("gb_lab_1983_q1", "Expand public employment and welfare provision.", "per503", -40.0),
            ManifestoQSentence("gb_lab_1983_q2", "Oppose nuclear weapons and prioritize peace.", "per106", -25.0),
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
            ManifestoQSentence("gb_con_1983_q1", "Support private enterprise and market discipline.", "per401", 35.0),
            ManifestoQSentence("gb_con_1983_q2", "Maintain strong defence commitments.", "per104", 20.0),
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
            ManifestoQSentence("gb_lab_1997_q1", "Invest in education and public services.", "per506", -12.0),
            ManifestoQSentence("gb_lab_1997_q2", "Commit to fiscal discipline and stable growth.", "per414", 8.0),
        ),
        replication="cmp_rile_root_label",
    ),
)


def make_manifesto_replication_trees(
    documents: Sequence[ManifestoDocument] | None = None,
    *,
    split: str = "test",
) -> list[ManifestoReplicationTree]:
    docs = tuple(documents or DEFAULT_MANIFESTO_REPLICATIONS)
    trees: list[ManifestoReplicationTree] = []
    for doc in docs:
        label = clamp_rile(float(doc.rile_label))
        leaves = tuple(
            ManifestoLeaf(
                text=str(q.text),
                qid=str(q.qid),
                code=str(q.code),
                guidance_score=(None if q.guidance_score is None else float(q.guidance_score)),
                weight=float(q.weight),
                metadata=dict(q.metadata or {}),
            )
            for q in doc.qsentences
        )
        guidance = [
            {
                "qid": leaf.qid,
                "text": leaf.text,
                "code": leaf.code,
                "guidance_score": leaf.guidance_score,
                "weight": leaf.weight,
            }
            for leaf in leaves
        ]
        metadata = {
            "split": split,
            "doc_id": doc.doc_id,
            "country": doc.country,
            "party": doc.party,
            "year": int(doc.year),
            "replication": doc.replication,
            "text": doc.text,
            "teacher_score_1_7": label,
            "teacher_score_native": label,
            "expert_score_1_7": label,
            "expert_score_native": label,
            "expert_target_scale": "rile",
            "expert_score_for_objective": label,
            "root_label": label,
            "root_label_name": "rile",
            "g_guidance_qsentences": guidance,
        }
        metadata.update(dict(doc.metadata or {}))
        trees.append(
            ManifestoReplicationTree(
                doc_id=str(doc.doc_id),
                text=str(doc.text),
                leaves=leaves,
                metadata=metadata,
            )
        )
    return trees


def manifesto_oracle_predict_fn(*, tree: Any, **kwargs: Any) -> Mapping[str, float]:
    del kwargs
    meta = getattr(tree, "metadata", None) or {}
    return {"score": float(meta["teacher_score_1_7"])}


def manifesto_prompt_template() -> str:
    return (
        "Estimate the document-level RILE score. Return only one number.\n\n"
        "Document:\n{text}\n\nQ-sentence guidance for g:\n{qsentence_guidance}\n\nScore:"
    )


def qsentence_guidance_text(tree: Any) -> str:
    meta = getattr(tree, "metadata", None) or {}
    rows = meta.get("g_guidance_qsentences") or []
    rendered: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        score = row.get("guidance_score")
        score_text = "unknown" if score is None else f"{float(score):.3g}"
        rendered.append(
            f"- {row.get('qid', '')}: code={row.get('code', '')}, "
            f"guidance_score={score_text}, text={row.get('text', '')}"
        )
    return "\n".join(rendered)


def replication_payload(trees: Sequence[ManifestoReplicationTree]) -> list[Mapping[str, Any]]:
    return [
        {
            "doc_id": tree.doc_id,
            "metadata": dict(tree.metadata),
            "leaves": [asdict(leaf) for leaf in tree.leaves],
        }
        for tree in trees
    ]


__all__ = [
    "DEFAULT_MANIFESTO_REPLICATIONS",
    "ManifestoDocument",
    "ManifestoLeaf",
    "ManifestoQSentence",
    "ManifestoReplicationTree",
    "make_manifesto_replication_trees",
    "manifesto_oracle_predict_fn",
    "manifesto_prompt_template",
    "qsentence_guidance_text",
    "replication_payload",
]
