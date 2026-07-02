"""Shared low-level helpers reused across the top-level treepo modules.

Holds schedule/audit-policy name literals, numeric guards, and the canonical
``jsonable`` serializer used to coerce arbitrary values into JSON-safe
structures before ``json.dumps`` or digesting.
"""

from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Tuple


ScheduleName = Literal["balanced", "left_to_right", "right_to_left"]
VALID_SCHEDULES: Tuple[ScheduleName, ...] = ("balanced", "left_to_right", "right_to_left")

AuditPolicyName = Literal["all", "fixed", "fraction", "sqrt", "log2"]
VALID_AUDIT_POLICIES: Tuple[AuditPolicyName, ...] = (
    "all",
    "fixed",
    "fraction",
    "sqrt",
    "log2",
)


def finite_float(value: float, *, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def jsonable(value: Any) -> Any:
    """Recursively coerce ``value`` into a JSON-serializable structure.

    Enums become their ``.value``, ``Path`` becomes ``str``, mappings/tuples/
    lists are recursed, and objects exposing ``to_dict`` or dataclasses are
    expanded (``to_dict`` preferred so custom coercion is honored). Anything
    else is returned unchanged.
    """

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return jsonable(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    return value


def audit_sample_count(
    internal_nodes: int,
    *,
    policy: AuditPolicyName,
    fixed_nodes: int = 0,
    fraction: float = 1.0,
    scale: float = 1.0,
) -> int:
    n = int(max(0, internal_nodes))
    if n <= 0:
        return 0

    pol = str(policy)
    if pol == "all":
        q = n
    elif pol == "fixed":
        q = int(max(0, fixed_nodes))
    elif pol == "fraction":
        q = int(math.ceil(float(fraction) * float(n)))
    elif pol == "sqrt":
        q = int(math.ceil(float(scale) * math.sqrt(float(n))))
    elif pol == "log2":
        q = int(math.ceil(float(scale) * math.log2(float(n) + 1.0)))
    else:
        raise ValueError(
            f"unsupported audit policy: {policy!r}; expected one of {VALID_AUDIT_POLICIES}"
        )
    return int(max(0, min(n, q)))


__all__ = [
    "AuditPolicyName",
    "ScheduleName",
    "VALID_AUDIT_POLICIES",
    "VALID_SCHEDULES",
    "audit_sample_count",
    "finite_float",
    "jsonable",
]
