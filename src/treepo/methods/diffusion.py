"""Extension stub for optional diffusion/generate experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from treepo.llm.diffusion import DiffusionBackendConfig


@dataclass(frozen=True)
class DiffusionTextFamilyConfig:
    backend: DiffusionBackendConfig | Mapping[str, Any] = field(
        default_factory=DiffusionBackendConfig
    )
    prompt_template: str = (
        "Read the document and return only one numeric score.\n\n{text}\n\nScore:"
    )
    sampling_params: Mapping[str, Any] = field(default_factory=dict)
    engine_options: Mapping[str, Any] = field(default_factory=dict)
    score_regex: str = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    min_score: float | None = None
    max_score: float | None = None
    default_artifact: str = "diffusion-text-zero-shot"


class DiffusionTextFamily:
    name = "diffusion"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise ImportError(
            "family='diffusion' is an optional application family and is not "
            "included in treepo. Register a diffusion/generate family before "
            "using this scorer."
        )


def build_diffusion_family(backend_config: Mapping[str, Any]) -> DiffusionTextFamily:
    del backend_config
    return DiffusionTextFamily()


__all__ = [
    "DiffusionTextFamily",
    "DiffusionTextFamilyConfig",
    "build_diffusion_family",
]
