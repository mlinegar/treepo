"""Shared text-chunking used by both the embedding-FNO and text-LLM data prep.

Both paths should see the *same* leaves for fair head-to-head comparison. This
module re-exports the internal `_token_leaf_chunks` helper under a stable
public name so the prep scripts can call it directly.
"""
from __future__ import annotations

from treepo._research.preprocessing.chunker import TextChunk

from treepo._research.unified_g_v1.realdoc.embedding_fno_training import _token_leaf_chunks


def leaf_chunks(
    text: str,
    *,
    leaf_tokens: int,
    token_encoding: str = "cl100k_base",
) -> list[TextChunk]:
    """Chunk a document into leaves of `leaf_tokens` tokens under `token_encoding`.

    This is the exact function the embedding-FNO path uses. The text-LLM prep
    should call this so both paths train on the same windows.
    """
    return _token_leaf_chunks(
        text,
        leaf_tokens=int(leaf_tokens),
        token_encoding=str(token_encoding),
    )


__all__ = ["TextChunk", "leaf_chunks"]
