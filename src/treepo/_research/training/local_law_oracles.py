from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class LocalLawOracleResolution:
    predictor: Callable[[str], float]
    source_kind: str
    source_spec: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_local_law_oracle_spec(raw: Any) -> Optional[str]:
    rendered = str(raw or "").strip()
    if not rendered:
        return None
    lowered = rendered.lower()
    if lowered in {"none", "off", "disabled"}:
        return None
    return rendered


def resolve_task_local_law_oracle(
    task: Any,
    *,
    backend_port: Optional[int] = None,
    backend_model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    strict_parse: Optional[bool] = None,
) -> Optional[LocalLawOracleResolution]:
    factory = getattr(task, "create_local_law_oracle", None)
    if not callable(factory):
        return None

    kwargs = {
        "port": None if backend_port is None else int(backend_port),
        "model": str(backend_model).strip() if backend_model else None,
        "max_tokens": None if max_tokens is None else int(max_tokens),
        "temperature": None if temperature is None else float(temperature),
        "strict_parse": strict_parse if strict_parse is None else bool(strict_parse),
    }
    try:
        signature = inspect.signature(factory)
        accepted = set(signature.parameters.keys())
        call_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in accepted and value is not None
        }
    except (TypeError, ValueError):
        call_kwargs = {key: value for key, value in kwargs.items() if value is not None}

    predictor = factory(**call_kwargs)
    if not callable(predictor):
        raise TypeError(
            f"Task local-law oracle factory returned non-callable object: {type(predictor).__name__}"
        )

    describe = getattr(task, "describe_local_law_oracle", None)
    metadata: Dict[str, Any] = {}
    if callable(describe):
        payload = describe()
        if isinstance(payload, dict):
            metadata = dict(payload)

    source_kind = str(metadata.get("kind") or "task_oracle").strip() or "task_oracle"
    source_spec = str(
        metadata.get("spec")
        or getattr(task, "name", "")
        or type(task).__name__
    ).strip() or None
    return LocalLawOracleResolution(
        predictor=predictor,
        source_kind=source_kind,
        source_spec=source_spec,
        metadata=metadata,
    )
