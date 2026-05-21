"""Standalone diffusion research prototype surfaces."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from treepo._research.diffusion.markov_toy import (
    MarkovToySketch,
    changepoint_count,
    chunk_states,
    count_only_feature,
    encode_markov_path,
    merge_markov_sketch,
    run_markov_toy_experiment,
)


__all__ = [
    "DiffusionBackend",
    "DiffusionBatchResponse",
    "DiffusionGeneration",
    "HTTPGenerateDiffusionBackend",
    "SGLangDiffusionBackend",
    "DiffusionOperationTrace",
    "DiffusionPromptTemplates",
    "DiffusionRunResult",
    "DiffusionTreeEngine",
    "FixedBinaryDiffusionTreeEngine",
    "MarkovToySketch",
    "VLLMOmniDiffusionBackend",
    "SGLangDiffusionClient",
    "build_diffusion_backend",
    "changepoint_count",
    "chunk_states",
    "count_only_feature",
    "encode_markov_path",
    "format_diffusion_chat_prompt",
    "merge_markov_sketch",
    "run_markov_toy_experiment",
]


def __getattr__(name: str) -> Any:
    if name in {
        "DiffusionBackend",
        "DiffusionBatchResponse",
        "DiffusionGeneration",
        "HTTPGenerateDiffusionBackend",
        "SGLangDiffusionBackend",
        "SGLangDiffusionClient",
        "VLLMOmniDiffusionBackend",
        "build_diffusion_backend",
    }:
        module = import_module("src.diffusion.backends")
        if name == "SGLangDiffusionClient":
            shim = import_module("src.diffusion.sglang_client")
            return getattr(shim, name)
        return getattr(module, name)
    if name in {
        "DiffusionOperationTrace",
        "DiffusionPromptTemplates",
        "DiffusionRunResult",
        "DiffusionTreeEngine",
        "FixedBinaryDiffusionTreeEngine",
        "format_diffusion_chat_prompt",
    }:
        module = import_module("src.diffusion.tree_engine")
        return getattr(module, name)
    raise AttributeError(name)
