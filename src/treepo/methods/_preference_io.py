"""Preference dataset IO and export helpers.

Coerce arbitrary inputs into a ``PreferenceDataset``, write the dataset plus its
optimizer-facing record views to disk, preview those views, and fan the dataset
out through trainer adapters. These operate on the data model, so they import
it rather than the other way around.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from treepo.methods._preference_dataset import PreferenceDataset, PreferenceFormat
from treepo.methods._preference_normalize import _json_default


def normalize_preference_data(value: Any) -> PreferenceDataset:
    return PreferenceDataset.from_value(value)


def export_preference_records(
    value: Any,
    output_dir: Path | str,
    *,
    formats: Sequence[PreferenceFormat] = ("general", "supervised", "dpo", "reward", "grpo"),
    save_hf: bool = True,
) -> dict[str, Any]:
    dataset = PreferenceDataset.from_value(value)
    if len(dataset) == 0:
        return {}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset.save(out_dir / "preference_dataset.json")
    files: dict[str, str] = {
        "dataset": str(dataset_path),
    }
    if save_hf:
        hf_path = out_dir / "preference_hf_dataset"
        try:
            dataset.to_hf_dataset_dict().save_to_disk(str(hf_path))
        except Exception as exc:
            files["hf_dataset_error"] = f"{type(exc).__name__}: {exc}"
        else:
            files["hf_dataset"] = str(hf_path)
    counts: dict[str, int] = {
        "units": len(dataset.units),
        "candidates": len(dataset.candidates),
        "dataset": len(dataset),
    }
    for format_name in formats:
        records = dataset.to_records(format_name)
        suffix = "json" if format_name == "grpo" else "jsonl"
        path = out_dir / f"preference_{format_name}.{suffix}"
        if suffix == "jsonl":
            path.write_text(
                "".join(json.dumps(row, sort_keys=True, default=_json_default) + "\n" for row in records),
                encoding="utf-8",
            )
        else:
            path.write_text(json.dumps(records, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
        files[format_name] = str(path)
        counts[format_name] = len(records)
    return {
        "summary": dataset.summary(),
        "files": files,
        "counts": counts,
    }


def summarize_preference_views(
    preferences: Any,
    *,
    views: Sequence[str] = ("supervised", "dpo", "reward", "grpo"),
) -> dict[str, Any]:
    """Preview optimizer-facing record views for a preference dataset.

    Returns per-view record counts plus the first record of each non-empty
    view (as ``first_<view>``). ``None`` preferences yield an empty preview.
    """
    if preferences is None:
        return {}
    records = {name: preferences.to_records(name) for name in views}
    preview: dict[str, Any] = {"counts": {name: len(rows) for name, rows in records.items()}}
    for name, rows in records.items():
        if rows:
            preview[f"first_{name}"] = rows[0]
    return preview


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


__all__ = [
    "export_adapter_views",
    "export_preference_records",
    "normalize_preference_data",
    "summarize_preference_views",
]
