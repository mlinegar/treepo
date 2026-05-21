"""Manifesto RILE embedding-sequence oracle.

Wraps the existing leaf-building machinery from `embedding_fno_training` and
exposes it behind the standard `TreeOracle` protocol. Each `TreeExample` has
`leaves = [EmbeddingSequenceTreeNode, ...]` and `target = RILE scalar`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from treepo._research.unified_g_v1.core.splits import resolve_document_split_ids
from treepo._research.unified_g_v1.training.tree_task import TreeExample


@dataclass
class ManifestoRileEmbeddingOracle:
    """Build RILE embedding-tree examples from a phase1 split.

    Heavy setup (embedding client, leaf cascade, tokenization) happens lazily
    on first `train_examples()` / `val_examples()` call, so constructing the
    oracle is cheap. Both calls share a `ManifestoDataset` and a
    `VLLMEmbeddingClient` instance.

    When `enforce_local_laws=True`, the oracle wires up a
    `RILECorpusIndex` from the Manifesto Project quasi-sentence codings
    and annotates every leaf and internal node of each tree with its
    span-level RILE target. These strict targets are then used by
    `ManifestoRileEmbeddingObjective` as C1/C3 supervision (in addition
    to C2 merge-commutativity that the objective always enforces).
    """

    embedding_api_base: str
    phase1_data_path: str | Path | None = None
    split_ids_path: str | Path | None = None
    embedding_model: str | None = None
    embedding_api_key: str = "EMPTY"
    embedding_timeout_seconds: float = 60.0
    embedding_batch_size: int = 32
    leaf_tokens: int = 1024
    subwindow_tokens: int = 128
    token_encoding: str = "cl100k_base"
    device: str = "auto"
    seed: int = 42
    enforce_local_laws: bool = True

    # Caches populated lazily.
    _train_items: list[TreeExample] = field(default_factory=list, init=False, repr=False)
    _val_items: list[TreeExample] = field(default_factory=list, init=False, repr=False)
    _embedding_dim: int | None = field(default=None, init=False, repr=False)
    _resolved_model_id: str = field(default="", init=False, repr=False)
    _split_source: str = field(default="", init=False, repr=False)
    _rile_index: Any | None = field(default=None, init=False, repr=False)

    def _ensure_built(self) -> None:
        if self._train_items and self._val_items:
            return
        # Lazy imports: these pull heavy deps (torch extensions, vllm client).
        from treepo._research.unified_g_v1.realdoc.embedding_fno_training import (
            _build_tree_examples,
            _load_samples_for_doc_ids,
            _resolve_device,
            _tree_examples_from_embedding_items,
        )
        from treepo._research.tasks.manifesto.data_loader import ManifestoDataset
        from treepo._research.training.embedding_proxy import VLLMEmbeddingClient

        random.seed(int(self.seed))
        torch.manual_seed(int(self.seed))
        device = _resolve_device(self.device)
        split_ids, split_source = resolve_document_split_ids(
            phase1_data_path=self.phase1_data_path,
            split_ids_path=self.split_ids_path,
        )
        self._split_source = str(split_source)
        dataset = ManifestoDataset(require_text=True)
        client = VLLMEmbeddingClient(
            api_base=str(self.embedding_api_base).rstrip("/"),
            model=None if self.embedding_model is None else str(self.embedding_model),
            api_key=str(self.embedding_api_key or "EMPTY"),
            timeout_seconds=float(self.embedding_timeout_seconds),
            batch_size=int(self.embedding_batch_size),
            cache_enabled=False,
        )
        self._resolved_model_id = str(client.resolve_model())

        # Build a per-sample span RILE fetcher when strict local-law
        # enforcement is requested. Samples whose manifesto_id isn't
        # granularly coded in the corpus_df fall back to a no-op fetcher
        # (returns None for every span), letting soft C1/C3 kick in for them.
        # Crucially, coded samples also switch their chunking source text to
        # the quasi-sentence-joined corpus text so the character coordinates
        # queried by `span_rile` are aligned with the text being chunked.
        span_rile_alignment_for_sample = None
        if self.enforce_local_laws:
            from treepo._research.tasks.manifesto.rile_codes import default_index

            self._rile_index = default_index()

            def _fn(sample):
                codings = self._rile_index.manifesto(str(sample.manifesto_id))
                if codings is None:
                    return None
                return str(codings.text), codings.span_rile

            span_rile_alignment_for_sample = _fn

        def _to_examples(doc_ids):
            samples = _load_samples_for_doc_ids(doc_ids, dataset=dataset)
            raw_items, _stats, dim = _build_tree_examples(
                samples,
                embedding_client=client,
                leaf_tokens=int(self.leaf_tokens),
                subwindow_tokens=int(self.subwindow_tokens),
                token_encoding=str(self.token_encoding),
                device=device,
                span_rile_alignment_for_sample=span_rile_alignment_for_sample,
            )
            if self._embedding_dim is None:
                self._embedding_dim = int(dim)
            return _tree_examples_from_embedding_items(raw_items)

        self._train_items = _to_examples(split_ids.train_doc_ids)
        self._val_items = _to_examples(split_ids.val_doc_ids)

    def train_examples(self) -> Sequence[TreeExample]:
        self._ensure_built()
        return self._train_items

    def val_examples(self) -> Sequence[TreeExample]:
        self._ensure_built()
        return self._val_items

    @property
    def embedding_dim(self) -> int | None:
        return self._embedding_dim

    def metadata(self) -> Mapping[str, Any]:
        return {
            "oracle": "manifesto_rile_embedding",
            "space_kind": "embedding_sequence",
            "embedding_api_base": str(self.embedding_api_base),
            "embedding_model": self._resolved_model_id or self.embedding_model,
            "leaf_tokens": int(self.leaf_tokens),
            "subwindow_tokens": int(self.subwindow_tokens),
            "split_source": self._split_source,
        }
