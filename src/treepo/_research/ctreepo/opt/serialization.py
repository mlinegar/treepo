from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of common python objects into JSON-serializable forms."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if dataclasses.is_dataclass(obj):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    # Avoid importing torch eagerly; detect tensor by module name.
    obj_mod = type(obj).__module__
    if obj_mod.startswith("torch") and hasattr(obj, "detach") and hasattr(obj, "cpu"):
        try:
            arr = obj.detach().cpu().numpy()
        except Exception:
            arr = None
        if arr is not None:
            return to_jsonable(arr)
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return to_jsonable(obj.to_dict())
    return str(obj)


def canonical_json(obj: Any) -> str:
    """Canonical JSON string for hashing/reproducibility (sorted keys, compact)."""
    payload = to_jsonable(obj)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def as_compact_str(obj: Any) -> str:
    """Readable, stable-ish string representation for storing in text-only records."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    payload = to_jsonable(obj)
    if payload is None:
        return ""
    if isinstance(payload, (dict, list)):
        return canonical_json(payload)
    return str(payload)
