"""Observation-granularity seam: supplied leaf windows survive to the model.

Locks the two guarantees the RILE qsentence experiment (and any
leaf=observation-granularity workload) depends on:

1. **Leaves are the supplied windows, exactly.** ``build_labeled_tree_from_text``
   with ``explicit_char_windows`` makes one leaf per window (validated
   contiguous + full coverage, hard error otherwise), and the FNO family
   embeds those leaf texts as-is — ``leaf_size_tokens`` is capacity
   validation, never a re-chunker. What you observe is what gets scored.

2. **Trees are prepared/embedded once.** Tree building is expensive and the
   bundle is built once on disk (``write_labeled_trees_jsonl`` /
   ``load_labeled_trees``); in-memory, ``FNOFamily._prepare`` must not
   re-embed when callers pass the same trees in fresh list wrappers (the
   alternating evaluator does ``list(trees)`` per call). Keyed by per-tree
   identity, not ``id()`` of the transient list.
"""

from __future__ import annotations

from typing import List

import pytest

from treepo._research.ctreepo.distillation import build_labeled_tree_from_text

QSENTENCES = [
    "We will expand the welfare state.",
    "Taxes on enterprise must fall.",
    "Peace negotiations should continue.",
    "Law and order protects communities.",
    "Headline of section two.",
]


def _doc_and_windows() -> tuple[str, List[tuple[int, int]]]:
    text = " ".join(QSENTENCES)
    windows: List[tuple[int, int]] = []
    cursor = 0
    for i, sentence in enumerate(QSENTENCES):
        end = cursor + len(sentence)
        # Windows must be contiguous: each window absorbs the trailing
        # separator space, except the last.
        if i < len(QSENTENCES) - 1:
            end += 1
        windows.append((cursor, end))
        cursor = end
    return text, windows


def _qsentence_tree():
    text, windows = _doc_and_windows()
    return build_labeled_tree_from_text(
        doc_id="doc_qsent",
        text=text,
        document_score=0.5,
        split="train",
        score_fn=lambda _span: 0.5,
        window_size=len(text),
        explicit_char_windows=windows,
        label_source="test",
    ), text, windows


def test_explicit_windows_become_leaves_exactly() -> None:
    tree, text, windows = _qsentence_tree()
    leaf_ids = tree.levels[0]
    assert len(leaf_ids) == len(windows)
    leaf_texts = [tree.get_node(str(nid)).text for nid in leaf_ids]
    assert leaf_texts == [text[start:end] for start, end in windows]


def test_non_covering_windows_hard_error() -> None:
    text, windows = _doc_and_windows()
    gappy = list(windows)
    start, end = gappy[2]
    gappy[2] = (start + 1, end)  # introduce a 1-char gap
    with pytest.raises(ValueError, match="contiguous"):
        build_labeled_tree_from_text(
            doc_id="doc_gap",
            text=text,
            document_score=0.5,
            split="train",
            score_fn=lambda _span: 0.5,
            window_size=len(text),
            explicit_char_windows=gappy,
            label_source="test",
        )


class _CountingEmbeddingClient:
    """Records every text it embeds; deterministic fixed-dim output."""

    def __init__(self, dim: int = 16):
        self.dim = int(dim)
        self.embedded_texts: List[str] = []
        self.calls = 0

    def resolve_model(self) -> str:
        return f"counting:{self.dim}"

    def embed_texts(self, texts):
        self.calls += 1
        self.embedded_texts.extend(str(t) for t in texts)
        return [
            [float((len(str(t)) + j) % 5) for j in range(self.dim)] for t in texts
        ]


def _fno_family(client):
    from treepo._research.ctreepo.fno_family import FNOFamily, FNOFamilyConfig
    import torch

    return FNOFamily(
        config=FNOFamilyConfig(
            hidden_channels=8, n_modes=4, n_layers=1, head_hidden_dim=16,
            # Far larger than any qsentence: must NOT cause re-chunking.
            leaf_size_tokens=512,
            embedding_max_length_tokens=None,
            effective_embedding_dim=16,
        ),
        embedding_client=client,
        device=torch.device("cpu"),
    )


def test_fno_embeds_supplied_leaves_verbatim() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("neuralop")
    tree, text, windows = _qsentence_tree()
    client = _CountingEmbeddingClient()
    family = _fno_family(client)

    prepared, _dim = family._prepare([tree])

    assert len(prepared) == 1
    assert prepared[0].leaf_embeddings.shape[0] == len(windows)
    # The embedding surface saw exactly the supplied observation units —
    # one embedding per qsentence window, verbatim, no re-chunking.
    assert client.embedded_texts == [text[start:end] for start, end in windows]


def test_prepare_caches_across_fresh_list_wrappers() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("neuralop")
    tree, _text, _windows = _qsentence_tree()
    client = _CountingEmbeddingClient()
    family = _fno_family(client)

    trees = [tree]
    prepared_first, _ = family._prepare(trees)
    calls_after_first = client.calls
    # The alternating evaluator passes `list(trees)` — a fresh wrapper —
    # on every evaluation call. The same trees must not be re-embedded.
    prepared_second, _ = family._prepare(list(trees))

    assert client.calls == calls_after_first
    assert prepared_second is prepared_first
