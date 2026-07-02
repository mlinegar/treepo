"""Trainer-adapter export wrapper for preference datasets.

A thin fan-out helper: run one preference ``dataset`` through each named
trainer adapter. ``export_for_adapter`` is injected by the caller so this stays
independent of the ``treepo.finetune`` module (and of the data model).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def export_adapter_views(
    export_for_adapter: Any,
    dataset: Any,
    output_dir: Path,
    *,
    adapters: Sequence[str] = (
        "embedding",
        "trl_sft",
        "trl_dpo",
        "trl_reward",
        "trl_scalar_reward",
        "trl_grpo",
        "dspy_examples",
    ),
    save_hf: bool = False,
) -> dict[str, Any]:
    """Export ``dataset`` through each named trainer adapter.

    ``export_for_adapter`` is injected so this helper stays independent of the
    ``treepo.finetune`` module. Returns a mapping of adapter name to its export
    artifacts, one subdirectory per adapter under ``output_dir``.
    """
    return {
        name: export_for_adapter(
            name,
            dataset,
            output_dir / name,
            save_hf=save_hf,
        )
        for name in adapters
    }


__all__ = ["export_adapter_views"]
