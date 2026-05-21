"""
Simple disk-backed response cache for OpenAI-compatible chat completions.

This is intentionally small and dependency-free:
- Keyed by a stable SHA256 over (model, messages, generation params).
- Stores JSON blobs on disk with atomic writes.

Primary use case: speed up repeated runs over the same documents/prompts
(e.g., CV folds, hyperparameter sweeps) by skipping identical LLM calls.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from treepo._research.core.conditional_memory import canonical_hash


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def make_chat_cache_key(
    *,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Compute a stable cache key for a chat completion request."""
    cache_data: Dict[str, Any] = {
        "model": str(model),
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if extra:
        # Only include JSON-serializable values; if something is not serializable
        # we fall back to string rendering.
        safe_extra: Dict[str, Any] = {}
        for key, value in extra.items():
            try:
                json.dumps(value)
                safe_extra[str(key)] = value
            except TypeError:
                safe_extra[str(key)] = str(value)
        cache_data["extra"] = safe_extra
    digest = canonical_hash(_stable_json_dumps(cache_data), normalize=False)
    return digest


@dataclass(frozen=True)
class CachedChatResponse:
    content: str
    usage: Dict[str, int]
    model: str
    created_at: str

    def to_json(self) -> str:
        return _stable_json_dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "CachedChatResponse":
        payload = json.loads(text)
        return cls(
            content=str(payload.get("content", "")),
            usage=dict(payload.get("usage", {}) or {}),
            model=str(payload.get("model", "")),
            created_at=str(payload.get("created_at", "")),
        )


class FileResponseCache:
    """
    File-per-key cache stored under a directory.

    Layout:
      <root>/<first2>/<key>.json
    """

    def __init__(self, root: Path):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        key = str(key)
        prefix = key[:2] if len(key) >= 2 else "__"
        return self.root / prefix / f"{key}.json"

    def get(self, key: str) -> Optional[CachedChatResponse]:
        path = self._path_for_key(key)
        try:
            data = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except Exception:
            return None
        try:
            return CachedChatResponse.from_json(data)
        except Exception:
            # Corrupt entry; best-effort cleanup.
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            return None

    def set(self, key: str, value: CachedChatResponse) -> None:
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(value.to_json(), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def now_iso() -> str:
        return datetime.now().isoformat()
