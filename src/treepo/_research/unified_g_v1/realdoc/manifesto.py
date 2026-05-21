from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

from treepo._research.unified_g_v1.core.artifact import (
    TextUnifiedGProgram,
    UnifiedGArtifact,
    build_unified_dspy_strategy,
)

from treepo._research.core.data_models import Tree
from treepo._research.tasks.manifesto.data_loader import ManifestoDataset
from treepo._research.tasks.manifesto.rubrics import RILE_PRESERVATION_RUBRIC
from treepo._research.tree.builder import BuildConfig, TreeBuilder


@dataclass(frozen=True)
class ManifestoAuditResult:
    doc_id: str
    chunk_size: int
    min_chunk_chars: int
    tree_height: int
    tree_nodes: int
    tree_leaves: int
    final_summary: str
    predicted_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_manifesto_text(doc_id: str) -> str:
    dataset = ManifestoDataset(require_text=True)
    sample = dataset.get_sample(str(doc_id))
    if sample is None or not str(getattr(sample, "text", "") or "").strip():
        raise ValueError(f"Could not load manifesto text for doc_id={doc_id!r}")
    return str(sample.text)


def build_manifesto_tree(
    text: str,
    *,
    artifact: UnifiedGArtifact | TextUnifiedGProgram,
    rubric: str = RILE_PRESERVATION_RUBRIC,
    chunk_size: int = 8000,
    min_chunk_chars: int = 400,
) -> Tree:
    strategy = build_unified_dspy_strategy(artifact)
    builder = TreeBuilder(
        strategy=strategy,
        config=BuildConfig(
            max_chunk_chars=int(chunk_size),
            min_chunk_chars=int(min_chunk_chars),
            chunk_strategy="axis",
        ),
    )
    return builder.build_sync(str(text), str(rubric)).tree


def audit_manifesto_document(
    doc_id: str,
    *,
    artifact: UnifiedGArtifact | TextUnifiedGProgram,
    rubric: str = RILE_PRESERVATION_RUBRIC,
    chunk_size: int = 8000,
    min_chunk_chars: int = 400,
    score_fn: Callable[[str], float] | None = None,
) -> ManifestoAuditResult:
    tree = build_manifesto_tree(
        load_manifesto_text(doc_id),
        artifact=artifact,
        rubric=rubric,
        chunk_size=int(chunk_size),
        min_chunk_chars=int(min_chunk_chars),
    )
    final_summary = str(tree.final_summary or "")
    predicted_score = (
        float(score_fn(final_summary))
        if score_fn is not None and final_summary.strip()
        else None
    )
    return ManifestoAuditResult(
        doc_id=str(doc_id),
        chunk_size=int(chunk_size),
        min_chunk_chars=int(min_chunk_chars),
        tree_height=int(tree.height),
        tree_nodes=int(tree.node_count),
        tree_leaves=int(tree.leaf_count),
        final_summary=final_summary,
        predicted_score=predicted_score,
    )


def run_manifesto_batch(
    doc_ids: Sequence[str],
    *,
    artifact: UnifiedGArtifact | TextUnifiedGProgram,
    rubric: str = RILE_PRESERVATION_RUBRIC,
    chunk_size: int = 8000,
    min_chunk_chars: int = 400,
    score_fn: Callable[[str], float] | None = None,
) -> list[dict[str, Any]]:
    return [
        audit_manifesto_document(
            str(doc_id),
            artifact=artifact,
            rubric=rubric,
            chunk_size=int(chunk_size),
            min_chunk_chars=int(min_chunk_chars),
            score_fn=score_fn,
        ).to_dict()
        for doc_id in doc_ids
    ]
