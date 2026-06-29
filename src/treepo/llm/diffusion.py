"""Diffusion/generate extension config.

OpenAI-compatible chat is the public text-generation path. ``/generate``
diffusion backends are optional application transports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


OPENAI_CHAT_ENGINES = frozenset({"openai", "openai_chat", "openai_compatible"})


@dataclass(frozen=True)
class DiffusionBackendConfig:
    """Configuration for optional generate transports.

    ``engine='openai'`` means callers should use the shared chat/text-generation
    surface. Other engines refer to optional generate transports.
    """

    engine: str = "openai"
    base_url: str | None = None
    base_urls: Sequence[str] | None = None
    max_concurrency: int = 8
    model: str | None = None
    timeout: float = 120.0
    generate_path: str | None = None
    default_payload: Mapping[str, Any] = field(default_factory=dict)
    use_unified_inference_engine: bool = False
    api_key: str = "EMPTY"


def is_openai_chat_engine(engine: str | None) -> bool:
    return str(engine or "").strip().lower() in OPENAI_CHAT_ENGINES


def build_diffusion_backend(
    config: DiffusionBackendConfig | Mapping[str, Any] | None = None,
    *,
    session: Any | None = None,
) -> Any:
    del session
    cfg = (
        config
        if isinstance(config, DiffusionBackendConfig)
        else DiffusionBackendConfig(**dict(config or {}))
    )
    if is_openai_chat_engine(cfg.engine):
        raise ValueError(
            "engine='openai' uses the shared chat/text-generation surface, not "
            "a /generate backend."
        )
    raise ImportError(
        "/generate diffusion backends are optional application transports and "
        "are not included in treepo. Register/build them from an external package."
    )


def __getattr__(name: str) -> Any:
    if name in {
        "DiffusionBackend",
        "DiffusionBatchResponse",
        "DiffusionGeneration",
        "HTTPGenerateDiffusionBackend",
        "SGLangDiffusionBackend",
        "VLLMOmniDiffusionBackend",
    }:
        raise ImportError(
            f"{name} is not included in treepo; import it "
            "from the external package that owns the generate transport."
        )
    raise AttributeError(name)


__all__ = [
    "DiffusionBackendConfig",
    "OPENAI_CHAT_ENGINES",
    "build_diffusion_backend",
    "is_openai_chat_engine",
]
