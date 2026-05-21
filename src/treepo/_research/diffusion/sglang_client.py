"""Compatibility wrapper for SGLang diffusion DLLM options."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from treepo._research.diffusion.backends import (
    DiffusionBatchResponse,
    DiffusionGeneration,
    SGLangDiffusionBackend,
)


class SGLangDiffusionClient(SGLangDiffusionBackend):
    """Backward-compatible SGLang client that still accepts DLLM alias fields."""

    def generate(
        self,
        texts: Sequence[str] | str,
        sampling_params: Optional[Mapping[str, Any]] = None,
        engine_options: Optional[Mapping[str, Any]] = None,
        dllm_algorithm: Optional[str] = None,
        dllm_algorithm_config: Optional[Mapping[str, Any] | str] = None,
    ) -> DiffusionBatchResponse:
        resolved_engine_options = dict(engine_options or {})
        if dllm_algorithm and "dllm_algorithm" not in resolved_engine_options:
            resolved_engine_options["dllm_algorithm"] = dllm_algorithm
        parsed_config = self._normalize_algorithm_config(dllm_algorithm_config)
        if parsed_config is not None and "dllm_algorithm_config" not in resolved_engine_options:
            resolved_engine_options["dllm_algorithm_config"] = parsed_config
        return super().generate(
            texts,
            sampling_params=sampling_params,
            engine_options=resolved_engine_options,
        )


__all__ = [
    "DiffusionBatchResponse",
    "DiffusionGeneration",
    "SGLangDiffusionBackend",
    "SGLangDiffusionClient",
]
