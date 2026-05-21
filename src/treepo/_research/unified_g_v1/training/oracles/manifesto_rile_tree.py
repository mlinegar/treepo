"""Manifesto RILE tree-text oracle for DSPy / tree-LLM training.

Builds fixed-token leaf chunks from the manifesto corpus, then attaches the
same per-span strict RILE targets used by the embedding-FNO path. For coded
manifestos we chunk the quasi-sentence-joined corpus text from
`RILECorpusIndex`; for uncoded manifestos we fall back to the raw text file and
leave strict per-node targets unset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo._research.unified_g_v1.core.splits import resolve_document_split_ids
from treepo._research.unified_g_v1.realdoc.rile_tree import RILETreeScaffold, build_rile_tree_scaffold
from treepo._research.unified_g_v1.training.tree_task import TreeExample


@dataclass
class ManifestoRileTreeOracle:
    phase1_data_path: str | Path | None = None
    split_ids_path: str | Path | None = None
    leaf_tokens: int = 1024
    token_encoding: str = "cl100k_base"
    enforce_local_laws: bool = True

    _train_items: list[TreeExample] = field(default_factory=list, init=False, repr=False)
    _val_items: list[TreeExample] = field(default_factory=list, init=False, repr=False)
    _split_source: str = field(default="", init=False, repr=False)
    _built: bool = field(default=False, init=False, repr=False)

    def _build_example(self, sample, *, chunker, rile_index) -> TreeExample:
        aligned_text = str(sample.text or "")
        span_rile_fn = None
        if rile_index is not None:
            codings = rile_index.manifesto(str(sample.manifesto_id))
            if codings is not None:
                aligned_text = str(codings.text)
                span_rile_fn = codings.span_rile

        leaf_chunks = chunker(
            aligned_text,
            leaf_tokens=int(self.leaf_tokens),
            token_encoding=str(self.token_encoding),
        )
        chunk_texts_with_spans = [
            (str(chunk.text or ""), int(chunk.start_char), int(chunk.end_char))
            for chunk in leaf_chunks
        ]
        if not chunk_texts_with_spans:
            chunk_texts_with_spans = [(aligned_text, 0, len(aligned_text))]

        scaffold = build_rile_tree_scaffold(
            doc_id=str(sample.manifesto_id),
            text=aligned_text,
            root_rile=float(sample.rile),
            chunk_texts_with_spans=chunk_texts_with_spans,
            span_rile_fn=span_rile_fn,
        )
        return TreeExample(
            leaves=[leaf.text for leaf in scaffold.leaves],
            target=float(sample.rile),
            extra={
                "doc_id": str(sample.manifesto_id),
                "scaffold": scaffold,
                "source_text": aligned_text,
                "strict_local_laws": bool(span_rile_fn is not None),
            },
        )

    def _ensure_built(self) -> None:
        if self._built:
            return
        from treepo._research.unified_g_v1.realdoc.embedding_fno_training import (
            _load_samples_for_doc_ids,
            _token_leaf_chunks,
        )
        from treepo._research.tasks.manifesto.data_loader import ManifestoDataset

        split_ids, split_source = resolve_document_split_ids(
            phase1_data_path=self.phase1_data_path,
            split_ids_path=self.split_ids_path,
        )
        self._split_source = str(split_source)
        dataset = ManifestoDataset(require_text=True)

        rile_index = None
        if self.enforce_local_laws:
            from treepo._research.tasks.manifesto.rile_codes import default_index

            rile_index = default_index()

        def _convert(doc_ids: Sequence[str]) -> list[TreeExample]:
            samples = _load_samples_for_doc_ids(doc_ids, dataset=dataset)
            return [
                self._build_example(
                    sample,
                    chunker=_token_leaf_chunks,
                    rile_index=rile_index,
                )
                for sample in samples
            ]

        self._train_items = _convert(split_ids.train_doc_ids)
        self._val_items = _convert(split_ids.val_doc_ids)
        self._built = True

    def train_examples(self) -> Sequence[TreeExample]:
        self._ensure_built()
        return self._train_items

    def val_examples(self) -> Sequence[TreeExample]:
        self._ensure_built()
        return self._val_items

    def metadata(self) -> Mapping[str, Any]:
        return {
            "oracle": "manifesto_rile_tree",
            "space_kind": "tree_text",
            "leaf_tokens": int(self.leaf_tokens),
            "token_encoding": str(self.token_encoding),
            "enforce_local_laws": bool(self.enforce_local_laws),
            "split_source": self._split_source,
        }


def tree_scaffold_from_example(example: TreeExample) -> RILETreeScaffold:
    scaffold = example.extra.get("scaffold")
    if not isinstance(scaffold, RILETreeScaffold):
        raise ValueError("tree example is missing extra['scaffold']")
    return scaffold
