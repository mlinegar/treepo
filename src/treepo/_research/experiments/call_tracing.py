from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Mapping


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compact_call_row(payload: Mapping[str, Any]) -> Dict[str, Any]:
    raw = dict(payload or {})
    usage = dict(raw.get("usage") or {})
    cost = dict(raw.get("cost") or {})
    artifacts = dict(raw.get("artifacts") or {})
    return _json_safe(
        {
            "created_utc": str(raw.get("created_utc") or utc_now_iso()),
            "call_id": str(raw.get("call_id") or raw.get("request_id") or ""),
            "request_id": str(raw.get("request_id") or raw.get("call_id") or ""),
            "experiment_id": str(raw.get("experiment_id") or raw.get("run_id") or ""),
            "unit_id": str(raw.get("unit_id") or ""),
            "method_id": str(raw.get("method_id") or ""),
            "runner_id": str(raw.get("runner_id") or ""),
            "problem_id": str(raw.get("problem_id") or ""),
            "document_id": str(raw.get("document_id") or ""),
            "node_id": str(raw.get("node_id") or ""),
            "request_kind": str(raw.get("request_kind") or raw.get("request_type") or ""),
            "role": str(raw.get("role") or ""),
            "surface": str(raw.get("surface") or ""),
            "engine": str(raw.get("engine") or ""),
            "model": str(raw.get("model") or raw.get("model_id") or ""),
            "latency_ms": float(raw.get("latency_ms", 0.0) or 0.0),
            "usage": usage,
            "cost": cost,
            "error": str(raw.get("error") or ""),
            "artifacts": artifacts,
            "metadata": dict(raw.get("metadata") or {}),
        }
    )


def batch_request_call_row(
    request: Any,
    response: Any,
    *,
    base_url: str = "",
    model: str = "",
    surface: str = "chat_openai",
) -> Dict[str, Any]:
    meta = dict(getattr(request, "call_metadata", {}) or {})
    request_type = str(getattr(request, "request_type", "") or "")
    role = str(meta.get("role") or "")
    if not role:
        role = "scorer" if request_type in {"score", "baseline"} else "summarizer"
    return compact_call_row(
        {
            **meta,
            "request_id": str(getattr(request, "request_id", "") or ""),
            "document_id": str(getattr(request, "document_id", "") or meta.get("document_id", "") or ""),
            "request_kind": str(meta.get("request_kind") or request_type),
            "role": role,
            "surface": str(meta.get("surface") or surface),
            "engine": str(meta.get("engine") or ""),
            "model": str(meta.get("model") or model),
            "base_url": str(meta.get("base_url") or base_url),
            "latency_ms": float(getattr(response, "latency_ms", 0.0) or 0.0),
            "usage": dict(getattr(response, "usage", {}) or {}),
            "error": str(getattr(response, "error", "") or ""),
            "metadata": {
                **dict(meta.get("metadata") or {}),
                "base_url": str(meta.get("base_url") or base_url),
            },
        }
    )


class JsonlCallTraceSink:
    """Append compact call rows to JSONL without storing prompts or context."""

    def __init__(self, path: str | Path, *, defaults: Mapping[str, Any] | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        self.defaults = dict(defaults or {})
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, payload: Mapping[str, Any]) -> None:
        row = compact_call_row({**self.defaults, **dict(payload or {})})
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=False) + "\n")
                handle.flush()


__all__ = [
    "JsonlCallTraceSink",
    "batch_request_call_row",
    "compact_call_row",
]
