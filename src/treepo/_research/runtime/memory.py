from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import tiktoken


@dataclass(frozen=True)
class TokenCounter:
    encoding_name: str = "cl100k_base"

    def _enc(self):
        return tiktoken.get_encoding(self.encoding_name)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc().encode(text))

    def truncate_tokens(self, text: str, *, max_tokens: int, keep: str = "tail") -> str:
        """Truncate to max_tokens using token boundaries.

        keep:
          - "tail": keep the last tokens (useful to preserve the question suffix)
          - "head": keep the first tokens
        """
        if max_tokens <= 0 or not text:
            return ""
        enc = self._enc()
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return text
        if keep == "head":
            kept = toks[:max_tokens]
        elif keep == "tail":
            kept = toks[-max_tokens:]
        else:
            raise ValueError(f"Unknown keep={keep!r}")
        return enc.decode(kept)


def chunk_text_tokens(
    text: str,
    *,
    counter: TokenCounter,
    chunk_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be > 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")

    enc = counter._enc()
    toks = enc.encode(text)
    if not toks:
        return [""]

    chunks: List[str] = []
    step = max(1, chunk_tokens - overlap_tokens)
    for start in range(0, len(toks), step):
        end = min(len(toks), start + chunk_tokens)
        chunk = enc.decode(toks[start:end])
        chunks.append(chunk)
        if end >= len(toks):
            break
    return chunks


def pairwise(iterable: Iterable[str]) -> List[tuple[str, str | None]]:
    items = list(iterable)
    out: List[tuple[str, str | None]] = []
    i = 0
    while i < len(items):
        a = items[i]
        b = items[i + 1] if i + 1 < len(items) else None
        out.append((a, b))
        i += 2
    return out

