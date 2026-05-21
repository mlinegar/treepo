from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


ROLE_SCORER = "scorer"
ROLE_SUMMARIZER = "summarizer"
ROLE_ORACLE = "oracle"
ROLE_EMBEDDER = "embedder"
ROLE_STATE_MODEL = "state_model"


@dataclass(frozen=True)
class RoleRef:
    role: str
    kind: str = ""
    model: str = ""
    endpoint: str = ""
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.metadata is None:
            data.pop("metadata", None)
        return {key: value for key, value in data.items() if value not in ("", None)}


def role_ref(
    role: str,
    *,
    kind: str = "",
    model: str = "",
    endpoint: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> RoleRef:
    return RoleRef(
        role=str(role),
        kind=str(kind or ""),
        model=str(model or ""),
        endpoint=str(endpoint or ""),
        metadata=dict(metadata or {}) or None,
    )


def roles_metadata(
    roles: Mapping[str, RoleRef | Mapping[str, Any]],
    *,
    oracle: RoleRef | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "roles": {
            str(name): value.to_dict() if isinstance(value, RoleRef) else dict(value)
            for name, value in dict(roles).items()
        }
    }
    if oracle is not None:
        out["oracle"] = oracle.to_dict() if isinstance(oracle, RoleRef) else dict(oracle)
    return out


__all__ = [
    "ROLE_EMBEDDER",
    "ROLE_ORACLE",
    "ROLE_SCORER",
    "ROLE_STATE_MODEL",
    "ROLE_SUMMARIZER",
    "RoleRef",
    "role_ref",
    "roles_metadata",
]
