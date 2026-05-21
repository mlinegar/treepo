from __future__ import annotations

import hashlib
from typing import Any

from .serialization import canonical_json


def stable_id(payload: Any, *, n_chars: int = 16) -> str:
    """Deterministic short id based on sha256(canonical_json(payload))."""
    if n_chars <= 0:
        raise ValueError("n_chars must be positive")
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[:n_chars]

