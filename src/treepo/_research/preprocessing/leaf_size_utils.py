"""Exact-token leaf sizing via the EmbeddingGemma tokenizer.

This module owns the one place in the codebase that converts token budgets to
char spans for leaf chunking. The tokenizer is loaded once and cached as a
module-level singleton so the ~5s HF load cost is amortized across all callers.

The chunker guarantees that every returned char window maps to a token-id slice
of exactly ``leaf_size_tokens`` (except the final window which may be shorter).
No silent truncation — any text that can't fit into the requested token budget
is distributed across additional leaves.

Design notes:

- The tokenizer used is Google's EmbeddingGemma-300m at
  ``/mnt/data/models/google/embeddinggemma-300m``. It is a Gemma-family
  tokenizer (``vocab_size=262144``) sharing the base vocabulary with
  Gemma-4-31B-IT; differences in ``tokenizer.json`` are template / special-token
  only. This matters because the same tokenizer then serves both:
  (a) chunking documents into token-sized leaves for teacher trace generation,
  (b) the LM-context-budget check in LM-based families (DSPy, TRL).
  Using a single tokenizer avoids any mismatch between "how many tokens did we
  cut into a leaf" vs "how many tokens will the embedding / LM see".

- Char windows are computed from ``offset_mapping`` rather than from a fixed
  ``chars_per_token`` heuristic. This is exact for any Gemma-tokenized text,
  across languages.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

#: Absolute path to the on-disk EmbeddingGemma-300m model directory. Its
#: tokenizer files are the canonical Gemma-family tokenizer used everywhere a
#: token budget is evaluated.
DEFAULT_TOKENIZER_PATH = "/mnt/data/models/google/embeddinggemma-300m"

_TOKENIZER_CACHE: dict = {}
_TOKENIZER_LOCK = threading.Lock()


def _patch_importlib_metadata_once() -> None:
    """Guard against the Python 3.12 + transformers 5.3 import crash where
    ``importlib.metadata.packages_distributions()`` hits ``None['Name']``.

    Applied once, idempotently.
    """
    import collections
    import importlib.metadata as _md

    if getattr(_md.packages_distributions, "__ctreepo_patched__", False):
        return

    def _safe_packages_distributions():
        pkg_to_dist = collections.defaultdict(list)
        for dist in _md.distributions():
            try:
                meta = dist.metadata
                if meta is None:
                    continue
                name = meta["Name"]
            except Exception:
                continue
            for top in (
                _md._top_level_declared(dist) or _md._top_level_inferred(dist)
            ):
                pkg_to_dist[top].append(name)
        return dict(pkg_to_dist)

    _safe_packages_distributions.__ctreepo_patched__ = True  # type: ignore[attr-defined]
    _md.packages_distributions = _safe_packages_distributions


def get_gemma_tokenizer(model_path: Optional[str] = None):
    """Return a cached HuggingFace tokenizer for the EmbeddingGemma model.

    The same tokenizer is reused across token counting, chunking, and
    LM-budget checks. The first call pays a ~5s load cost; subsequent
    calls are instantaneous.
    """
    path = str(model_path or DEFAULT_TOKENIZER_PATH)
    cached = _TOKENIZER_CACHE.get(path)
    if cached is not None:
        return cached
    with _TOKENIZER_LOCK:
        cached = _TOKENIZER_CACHE.get(path)
        if cached is not None:
            return cached
        _patch_importlib_metadata_once()
        from transformers import AutoTokenizer  # type: ignore[import-not-found]

        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        _TOKENIZER_CACHE[path] = tok
        LOGGER.info("Loaded Gemma-family tokenizer from %s (vocab=%d)", path, tok.vocab_size)
        return tok


def count_tokens(text: str, *, model_path: Optional[str] = None) -> int:
    """Exact token count using the EmbeddingGemma tokenizer."""
    tok = get_gemma_tokenizer(model_path)
    if not text:
        return 0
    return int(len(tok.encode(str(text), add_special_tokens=False)))


def char_windows_from_token_budget(
    text: str,
    leaf_size_tokens: int,
    *,
    model_path: Optional[str] = None,
) -> List[Tuple[int, int]]:
    """Split ``text`` into non-overlapping char windows of ``leaf_size_tokens`` each.

    Uses the EmbeddingGemma tokenizer's ``offset_mapping`` to compute exact
    char boundaries. Every returned window corresponds to exactly
    ``leaf_size_tokens`` tokens from the tokenizer, except possibly the final
    window which may be shorter (the tail of the document).

    Guarantees:
    - Windows are non-overlapping and cover the entire input text.
    - ``sum(end - start for (start, end) in windows) == len(text)`` (to within
      whitespace that the tokenizer may skip between offsets; the final window
      always extends to ``len(text)``).
    - No silent drop: if tokenization produces more than ``leaf_size_tokens``
      tokens, additional windows are emitted; nothing is truncated.

    Raises ``ValueError`` for ``leaf_size_tokens <= 0``.
    """
    if int(leaf_size_tokens) <= 0:
        raise ValueError(f"leaf_size_tokens must be positive, got {leaf_size_tokens}")
    rendered = str(text or "")
    if not rendered:
        return [(0, 0)]
    tok = get_gemma_tokenizer(model_path)
    encoded = tok(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets: Sequence[Tuple[int, int]] = encoded.get("offset_mapping") or []
    if not offsets:
        return [(0, len(rendered))]
    windows: List[Tuple[int, int]] = []
    budget = int(leaf_size_tokens)
    n_tokens = len(offsets)
    i = 0
    prev_end = 0
    while i < n_tokens:
        j = min(i + budget, n_tokens)
        # Keep char windows contiguous so tree leaves cover the original text.
        # Whitespace between token offsets is attached to the preceding chunk;
        # that preserves exact token-slice boundaries while avoiding dropped
        # characters in downstream char-span replay.
        start = int(prev_end)
        end = int(offsets[j][0]) if j < n_tokens else int(offsets[j - 1][1])
        # Last window extends to end of text to absorb any trailing whitespace
        # that sits beyond the last token's offset.
        if j >= n_tokens:
            end = len(rendered)
        windows.append((start, end))
        prev_end = end
        i = j
    if windows:
        windows[0] = (0, windows[0][1])
        if windows[-1][1] != len(rendered):
            windows[-1] = (windows[-1][0], len(rendered))
    return windows


def leaf_size_tokens_to_approx_chars(
    leaf_size_tokens: int,
    *,
    chars_per_token: float = 4.0,
) -> int:
    """Coarse approximation when an exact chunker is unavailable.

    Prefer :func:`char_windows_from_token_budget` when you can afford a
    tokenizer call. This helper is for quick budget math (e.g. CLI arg
    validation) where approximate is fine.
    """
    return max(1, int(round(float(leaf_size_tokens) * float(chars_per_token))))


def assert_no_truncation(
    text: str,
    *,
    max_tokens: int,
    model_path: Optional[str] = None,
) -> None:
    """Raise ``RuntimeError`` if ``text`` would be truncated at ``max_tokens``.

    Use this at every boundary where an embedding or LM call is about to
    happen — it enforces the no-truncation invariant explicitly rather than
    relying on downstream ``truncation=True`` to silently drop content.
    """
    actual = count_tokens(text, model_path=model_path)
    if actual > int(max_tokens):
        raise RuntimeError(
            f"silent truncation would occur: text has {actual} tokens but "
            f"max_tokens={max_tokens}. Either (a) raise the per-call budget, "
            f"(b) split the text into smaller chunks before calling, or "
            f"(c) shrink leaf_size_tokens so chunks fit this embedding/LM."
        )
