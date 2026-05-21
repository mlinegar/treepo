"""Centralized path defaults so users don't have to edit source for their setup.

Three categories of paths live here:

1. **Model files** — local snapshots of HuggingFace-style models.
   Override with ``TREEPO_MODEL_DIR`` (default: ``~/models``).
2. **Runtime venvs** — separate Python environments where ``vllm`` /
   ``sglang`` are installed (research code launches them as subprocesses).
   Override with ``TREEPO_VLLM_VENV`` / ``TREEPO_SGLANG_VENV``
   (defaults: ``~/vllm-env`` and ``~/sglang-env``).
3. **The tokenizer model name** consumed by several family configs as
   their default tokenizer path. Constant ``DEFAULT_TOKENIZER_MODEL``;
   resolve to a filesystem path with ``default_tokenizer_path()``.

Why a module rather than buried defaults in each dataclass: a single
``export TREEPO_MODEL_DIR=/mnt/data/models`` in your shell now recovers
the original behavior across every consumer; previously you had to edit
several files.

The module is intentionally tiny — just constants + thin helpers — so
treepo.cld and the vendored ``_research/`` code can both import from it
without any additional dependency cost.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path | str) -> Path:
    """Resolve an environment variable to a Path, falling back to ``default``."""
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else Path(default).expanduser()


# ---------------------------------------------------------------------------
# Model directory
# ---------------------------------------------------------------------------

#: Root for local model snapshots. Override via the ``TREEPO_MODEL_DIR`` env var.
#:
#: Example: ``export TREEPO_MODEL_DIR=/mnt/data/models`` will resolve
#: ``model_path("google", "embeddinggemma-300m")`` to
#: ``/mnt/data/models/google/embeddinggemma-300m``.
MODEL_DIR: Path = _env_path("TREEPO_MODEL_DIR", Path.home() / "models")


def model_path(*parts: str) -> str:
    """Join ``MODEL_DIR`` with ``parts``, return a string path.

    >>> model_path("google", "embeddinggemma-300m")  # doctest: +SKIP
    '/home/mlinegar/models/google/embeddinggemma-300m'
    """
    return str(MODEL_DIR.joinpath(*parts))


#: HuggingFace-style name of the default tokenizer model used by
#: :class:`DSPyFamilyConfig`, :class:`FNOFamilyConfig`, and
#: :class:`TRLFamilyConfig` for budget calculations.
DEFAULT_TOKENIZER_MODEL: str = "google/embeddinggemma-300m"


def default_tokenizer_path() -> str:
    """Resolve :data:`DEFAULT_TOKENIZER_MODEL` against :data:`MODEL_DIR`."""
    return model_path(*DEFAULT_TOKENIZER_MODEL.split("/"))


# ---------------------------------------------------------------------------
# Runtime venvs (for subprocess-launched vLLM / sglang engines)
# ---------------------------------------------------------------------------

#: vLLM venv path. Override via the ``TREEPO_VLLM_VENV`` env var.
VLLM_VENV: Path = _env_path("TREEPO_VLLM_VENV", Path.home() / "vllm-env")

#: sglang venv path. Override via the ``TREEPO_SGLANG_VENV`` env var.
SGLANG_VENV: Path = _env_path("TREEPO_SGLANG_VENV", Path.home() / "sglang-env")


def vllm_venv_path() -> str:
    """Resolve the vLLM venv path as a string (re-read env each call)."""
    return str(_env_path("TREEPO_VLLM_VENV", Path.home() / "vllm-env"))


def sglang_venv_path() -> str:
    """Resolve the sglang venv path as a string (re-read env each call)."""
    return str(_env_path("TREEPO_SGLANG_VENV", Path.home() / "sglang-env"))


__all__ = [
    "MODEL_DIR",
    "DEFAULT_TOKENIZER_MODEL",
    "VLLM_VENV",
    "SGLANG_VENV",
    "model_path",
    "default_tokenizer_path",
    "vllm_venv_path",
    "sglang_venv_path",
]
