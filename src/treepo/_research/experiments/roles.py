from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Mapping

from treepo._research.experiments.contracts import MethodRef


ROLE_SCORER = "scorer"
ROLE_SUMMARIZER = "summarizer"
ROLE_ORACLE = "oracle"
ROLE_EMBEDDER = "embedder"
ROLE_STATE_MODEL = "state_model"

CANONICAL_ROLES = (
    ROLE_SCORER,
    ROLE_SUMMARIZER,
    ROLE_EMBEDDER,
    ROLE_STATE_MODEL,
)


def _clean_mapping(payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in dict(payload or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and not value:
            continue
        out[str(key)] = value
    return out


def role_ref(
    *,
    role: str,
    surface: str = "",
    engine: str = "",
    model: str = "",
    base_url: str = "",
    endpoint: str = "",
    execution_mode: str = "",
    checkpoint_path: str = "",
    checkpoint: str = "",
    defaulted_from: str = "",
    metadata: Mapping[str, Any] | None = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Return compact public role metadata for a model/operator role."""

    endpoint_url = str(base_url or endpoint or "")
    checkpoint_value = str(checkpoint_path or checkpoint or "")
    return _clean_mapping(
        {
            "role": str(role),
            "surface": str(surface or ""),
            "engine": str(engine or ""),
            "model": str(model or ""),
            "base_url": endpoint_url,
            "execution_mode": str(execution_mode or ""),
            "checkpoint_path": checkpoint_value,
            "defaulted_from": str(defaulted_from or ""),
            "metadata": _clean_mapping(metadata),
            **dict(extra),
        }
    )


def chat_role_ref(
    *,
    role: str,
    engine: str = "",
    model: str = "",
    base_url: str = "",
    endpoint: str = "",
    defaulted_from: str = "",
    metadata: Mapping[str, Any] | None = None,
    **extra: Any,
) -> Dict[str, Any]:
    return role_ref(
        role=role,
        surface="chat_openai",
        engine=engine,
        model=model,
        base_url=base_url,
        endpoint=endpoint,
        defaulted_from=defaulted_from,
        metadata=metadata,
        **extra,
    )


def embedder_role_ref(
    *,
    engine: str = "",
    model: str = "",
    base_url: str = "",
    endpoint: str = "",
    metadata: Mapping[str, Any] | None = None,
    **extra: Any,
) -> Dict[str, Any]:
    return role_ref(
        role=ROLE_EMBEDDER,
        surface="embedding",
        engine=engine,
        model=model,
        base_url=base_url,
        endpoint=endpoint,
        metadata=metadata,
        **extra,
    )


def state_model_role_ref(
    *,
    engine: str = "",
    model: str = "",
    checkpoint_path: str = "",
    checkpoint: str = "",
    execution_mode: str = "",
    metadata: Mapping[str, Any] | None = None,
    **extra: Any,
) -> Dict[str, Any]:
    return role_ref(
        role=ROLE_STATE_MODEL,
        surface="operator",
        engine=engine,
        model=model,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        execution_mode=execution_mode,
        metadata=metadata,
        **extra,
    )


def oracle_ref(
    *,
    kind: str = "benchmark_labels",
    source: str = "",
    model: str = "",
    metadata: Mapping[str, Any] | None = None,
    **extra: Any,
) -> Dict[str, Any]:
    return _clean_mapping(
        {
            "kind": str(kind or "benchmark_labels"),
            "source": str(source or ""),
            "model": str(model or ""),
            "metadata": _clean_mapping(metadata),
            **dict(extra),
        }
    )


def normalize_roles(roles: Mapping[str, Any] | None) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    for role, raw_cfg in dict(roles or {}).items():
        if not isinstance(raw_cfg, Mapping):
            continue
        cfg = _clean_mapping(raw_cfg)
        if "role" not in cfg:
            cfg["role"] = str(role)
        normalized[str(role)] = cfg
    return normalized


def metadata_with_roles(
    metadata: Mapping[str, Any] | None = None,
    *,
    roles: Mapping[str, Any] | None = None,
    oracle: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    merged = dict(metadata or {})
    if roles is not None:
        existing_roles = normalize_roles(merged.get("roles") if isinstance(merged.get("roles"), Mapping) else {})
        existing_roles.update(normalize_roles(roles))
        merged["roles"] = existing_roles
    if oracle is not None:
        merged["oracle"] = _clean_mapping(oracle)
    return merged


def method_ref_with_roles(
    method_ref: MethodRef,
    *,
    roles: Mapping[str, Any] | None = None,
    oracle: Mapping[str, Any] | None = None,
) -> MethodRef:
    """Attach canonical role/oracle metadata to an existing MethodRef."""

    return replace(
        method_ref,
        metadata=metadata_with_roles(method_ref.metadata, roles=roles, oracle=oracle),
    )


__all__ = [
    "CANONICAL_ROLES",
    "ROLE_EMBEDDER",
    "ROLE_ORACLE",
    "ROLE_SCORER",
    "ROLE_STATE_MODEL",
    "ROLE_SUMMARIZER",
    "chat_role_ref",
    "embedder_role_ref",
    "metadata_with_roles",
    "method_ref_with_roles",
    "normalize_roles",
    "oracle_ref",
    "role_ref",
    "state_model_role_ref",
]
