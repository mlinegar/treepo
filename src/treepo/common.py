from __future__ import annotations

import math
from typing import Literal, Tuple


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
]
