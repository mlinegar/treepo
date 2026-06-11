"""Lightweight GEPA default constants.

This module deliberately has no DSPy import. It is shared by the LLM
optimizer implementation and by package metadata that should remain importable
without the `llm` extra.
"""

from __future__ import annotations

from typing import Any


GEPA_STRONG_DEFAULT_KWARGS: dict[str, Any] = {
    "use_merge": True,
    "max_merge_invocations": 5,
    "track_stats": True,
    "add_format_failure_as_feedback": True,
    "reflection_minibatch_size": 8,
    "use_wandb": False,
    "use_mlflow": False,
}


__all__ = ["GEPA_STRONG_DEFAULT_KWARGS"]
