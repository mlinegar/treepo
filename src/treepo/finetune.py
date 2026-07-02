"""Trainer-neutral fine-tuning views over ``PreferenceDataset``.

This module deliberately stops at data projection. Embedding trainers,
sentence-transformers loops, TRL loops, and LoRA/PEFT configuration live in
optional downstream workflows that consume these exported rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from treepo.methods.preference import PreferenceDataset
from treepo.state import jsonable


FineTuneView = Literal[
    "embedding_pairs",
    "embedding_triplets",
    "embedding_ranked",
    "sft",
    "dpo",
    "reward",
    "grpo",
]

DEFAULT_FINETUNE_VIEWS: tuple[FineTuneView, ...] = (
    "embedding_pairs",
    "embedding_triplets",
    "embedding_ranked",
    "sft",
    "dpo",
    "reward",
    "grpo",
)


@dataclass(frozen=True)
class FineTuneAdapter:
    """Lightweight framework adapter over fine-tuning row views.

    Adapters in this package only prepare/export rows. Downstream packages can
    register adapters that also expose explicit training capabilities, but no
    trainer framework is imported here.
    """

    name: str
    framework: str
    required_views: tuple[str, ...]
    capabilities: tuple[str, ...] = ("prepare", "export")
    description: str = ""
    prepare_fn: Callable[..., dict[str, Any]] | None = None

    def prepare(
        self,
        preference_data: Any,
        output_dir: Path | str,
        *,
        save_hf: bool = True,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.prepare_fn is None:
            raise RuntimeError(f"fine-tune adapter {self.name!r} has no prepare function")
        return self.prepare_fn(
            preference_data,
            output_dir,
            adapter=self,
            save_hf=save_hf,
            config=dict(config or {}),
        )


_ADAPTERS: dict[str, FineTuneAdapter] = {}

_IDENTITY_FIELDS = (
    "tree_id",
    "doc_id",
    "node_id",
    "unit_id",
    "unit_type",
    "level",
    "position",
    "parent_id",
    "left_child_id",
    "right_child_id",
)


def build_finetune_views(
    preference_data: Any,
    *,
    views: Sequence[FineTuneView | str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Project one ``PreferenceDataset`` into trainer-ready row views.

    ``preference_data`` is normalized through ``PreferenceDataset.from_value``.
    The returned rows are plain JSONable dictionaries and keep tree/unit identity
    in each row's ``metadata``.
    """

    dataset = PreferenceDataset.from_value(preference_data)
    requested = _normalize_views(views)
    unit_index = _unit_index(dataset)
    out: dict[str, list[dict[str, Any]]] = {}

    for view in requested:
        if view == "embedding_pairs":
            out[view] = [
                _embedding_pair(row, unit_index=unit_index)
                for row in dataset.to_records("supervised")
            ]
        elif view == "embedding_triplets":
            out[view] = [
                _embedding_triplet(row, unit_index=unit_index)
                for row in dataset.to_records("reward")
            ]
        elif view == "embedding_ranked":
            out[view] = [
                _embedding_ranked(row, unit_index=unit_index)
                for row in dataset.to_records("grpo")
            ]
        elif view == "sft":
            out[view] = [_sft(row, unit_index=unit_index) for row in dataset.to_records("supervised")]
        elif view in {"dpo", "reward", "grpo"}:
            out[view] = [_enrich_record(row, unit_index=unit_index, format_name=view) for row in dataset.to_records(view)]  # type: ignore[arg-type]
        else:  # pragma: no cover - _normalize_views rejects this.
            raise ValueError(f"unsupported fine-tune view: {view}")

    return out


def export_finetune_views(
    preference_data: Any,
    output_dir: Path | str,
    *,
    views: Sequence[FineTuneView | str] | None = None,
    save_hf: bool = True,
) -> dict[str, Any]:
    """Write fine-tuning views as JSONL/JSON and, when available, HF datasets."""

    built = build_finetune_views(preference_data, views=views)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    counts: dict[str, int] = {}
    for name, rows in built.items():
        suffix = "json" if name in {"embedding_ranked", "grpo"} else "jsonl"
        path = out_dir / f"finetune_{name}.{suffix}"
        if suffix == "jsonl":
            _write_jsonl(path, rows)
        else:
            path.write_text(json.dumps(jsonable(rows), indent=2, sort_keys=True), encoding="utf-8")
        files[name] = str(path)
        counts[name] = len(rows)

    hf_path: Path | None = None
    if save_hf:
        hf_path = out_dir / "finetune_hf_dataset"
        try:
            _save_hf_dataset_dict(built, hf_path)
        except ImportError:
            hf_path = None
        else:
            files["hf_dataset"] = str(hf_path)

    return {
        "views": list(built.keys()),
        "counts": counts,
        "files": files,
        "summary": {
            "n_views": len(built),
            "n_rows": sum(counts.values()),
            "hf_dataset": None if hf_path is None else str(hf_path),
        },
    }



def register_finetune_adapter(
    adapter: FineTuneAdapter,
    *,
    replace: bool = False,
) -> FineTuneAdapter:
    """Register a framework adapter without importing the framework itself."""

    name = str(adapter.name).strip()
    if not name:
        raise ValueError("fine-tune adapter name must be non-empty")
    _normalize_views(adapter.required_views)
    if name in _ADAPTERS and not replace:
        raise ValueError(f"fine-tune adapter already registered: {name}")
    _ADAPTERS[name] = adapter
    return adapter


def get_finetune_adapter(name: str) -> FineTuneAdapter:
    """Return a registered adapter by name."""

    key = str(name).strip()
    try:
        return _ADAPTERS[key]
    except KeyError as exc:
        available = ", ".join(sorted(_ADAPTERS)) or "<none>"
        raise KeyError(f"unknown fine-tune adapter {key!r}; available: {available}") from exc


def list_finetune_adapters() -> list[FineTuneAdapter]:
    """List registered adapters in stable name order."""

    return [_ADAPTERS[name] for name in sorted(_ADAPTERS)]


def export_for_adapter(
    name: str,
    preference_data: Any,
    output_dir: Path | str,
    *,
    save_hf: bool = True,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare/export fine-tuning rows for one registered adapter."""

    adapter = get_finetune_adapter(name)
    return adapter.prepare(
        preference_data,
        output_dir,
        save_hf=save_hf,
        config=config,
    )


def _prepare_generic_adapter(
    preference_data: Any,
    output_dir: Path | str,
    *,
    adapter: FineTuneAdapter,
    save_hf: bool,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    requested = config.get("views")
    views = tuple(str(view) for view in requested) if requested else adapter.required_views
    artifacts = export_finetune_views(
        preference_data,
        output_dir,
        views=views,
        save_hf=save_hf,
    )
    return _adapter_result(
        adapter=adapter,
        views=artifacts.get("views", list(views)),
        counts=dict(artifacts.get("counts", {})),
        files=dict(artifacts.get("files", {})),
        summary=dict(artifacts.get("summary", {})),
    )


def _prepare_projected_adapter(
    preference_data: Any,
    output_dir: Path | str,
    *,
    adapter: FineTuneAdapter,
    save_hf: bool,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    del config
    built = build_finetune_views(preference_data, views=adapter.required_views)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    projected: dict[str, list[dict[str, Any]]] = {}
    files: dict[str, str] = {}
    counts: dict[str, int] = {}
    for view_name in adapter.required_views:
        rows = _adapter_rows(adapter.name, view_name, built.get(view_name, ()))
        projected[view_name] = rows
        path = out_dir / f"{adapter.name}_{view_name}.jsonl"
        _write_jsonl(path, rows)
        files[view_name] = str(path)
        counts[view_name] = len(rows)

    hf_path: Path | None = None
    if save_hf:
        hf_path = out_dir / f"{adapter.name}_hf_dataset"
        try:
            _save_hf_dataset_dict(projected, hf_path)
        except ImportError:
            hf_path = None
        else:
            files["hf_dataset"] = str(hf_path)

    return _adapter_result(
        adapter=adapter,
        views=list(projected),
        counts=counts,
        files=files,
        summary={
            "n_views": len(projected),
            "n_rows": sum(counts.values()),
            "hf_dataset": None if hf_path is None else str(hf_path),
        },
    )


def _adapter_result(
    *,
    adapter: FineTuneAdapter,
    views: Sequence[str],
    counts: Mapping[str, int],
    files: Mapping[str, str],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "adapter": adapter.name,
        "framework": adapter.framework,
        "capabilities": list(adapter.capabilities),
        "views": list(views),
        "counts": dict(counts),
        "files": dict(files),
        "summary": {
            **dict(summary),
            "adapter": adapter.name,
            "framework": adapter.framework,
            "trainer_imported": False,
        },
    }


def _adapter_rows(
    adapter_name: str,
    view_name: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if adapter_name == "embedding":
        return [jsonable(dict(row)) for row in rows]
    if adapter_name == "trl_sft":
        return [_trl_sft_row(row) for row in rows]
    if adapter_name == "trl_dpo":
        return [_trl_dpo_row(row) for row in rows]
    if adapter_name == "trl_reward":
        return [_trl_reward_row(row) for row in rows]
    if adapter_name == "trl_scalar_reward":
        return [row for raw in rows if (row := _trl_scalar_reward_row(raw)) is not None]
    if adapter_name == "trl_grpo":
        return [_trl_grpo_row(row) for row in rows]
    if adapter_name == "dspy_examples":
        return [_dspy_row(view_name, row) for row in rows]
    raise ValueError(f"unsupported built-in fine-tune adapter: {adapter_name}")


def _base_adapter_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_weight": _sample_weight(row),
        "metadata": jsonable(dict(row.get("metadata") or {})),
    }


def _trl_sft_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prompt": str(row.get("prompt") or ""),
        "completion": str(row.get("completion") or ""),
        **_base_adapter_row(row),
    }


def _trl_dpo_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prompt": str(row.get("prompt") or ""),
        "chosen": str(row.get("chosen") or ""),
        "rejected": str(row.get("rejected") or ""),
        **_base_adapter_row(row),
    }


def _trl_reward_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = _trl_dpo_row(row)
    if row.get("chosen_score") is not None:
        out["chosen_score"] = float(row["chosen_score"])
    if row.get("rejected_score") is not None:
        out["rejected_score"] = float(row["rejected_score"])
    return out


def _trl_scalar_reward_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    score = row.get("score")
    if score is None:
        return None
    return {
        "prompt": str(row.get("prompt") or ""),
        "response": str(row.get("completion") or row.get("response") or ""),
        "score": float(score),
        **_base_adapter_row(row),
    }


def _trl_grpo_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prompt": str(row.get("prompt") or ""),
        "responses": [str(item) for item in list(row.get("responses") or ())],
        "ranks": [int(rank) for rank in list(row.get("ranks") or ())],
        "scores": [None if score is None else float(score) for score in list(row.get("scores") or ())],
        **_base_adapter_row(row),
    }


def _dspy_row(view_name: str, row: Mapping[str, Any]) -> dict[str, Any]:
    base = _base_adapter_row(row)
    if view_name == "sft":
        return {
            "prompt": str(row.get("prompt") or ""),
            "completion": str(row.get("completion") or ""),
            "dspy_inputs": ["prompt"],
            **base,
        }
    if view_name == "dpo":
        return {
            "prompt": str(row.get("prompt") or ""),
            "summary_a": str(row.get("chosen") or ""),
            "summary_b": str(row.get("rejected") or ""),
            "preferred": "A",
            "dspy_inputs": ["prompt", "summary_a", "summary_b"],
            **base,
        }
    raise ValueError(f"dspy_examples does not support view {view_name!r}")



def _normalize_views(views: Sequence[FineTuneView | str] | None) -> tuple[str, ...]:
    if views is None:
        return tuple(DEFAULT_FINETUNE_VIEWS)
    valid = set(DEFAULT_FINETUNE_VIEWS)
    normalized = tuple(str(view) for view in views)
    invalid = sorted(set(normalized) - valid)
    if invalid:
        raise ValueError(f"unsupported fine-tune view(s): {', '.join(invalid)}")
    return normalized


def _unit_index(dataset: PreferenceDataset) -> dict[str, dict[str, Any]]:
    return {str(row.get("unit_id") or ""): dict(row) for row in dataset.to_records("general")}


def _embedding_pair(row: Mapping[str, Any], *, unit_index: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    metadata = _metadata_for_row(row, unit_index=unit_index, format_name="embedding_pairs")
    score = row.get("score")
    return {
        "anchor": str(row.get("prompt") or ""),
        "positive": str(row.get("completion") or ""),
        "score": 1.0 if score is None else float(score),
        "sample_weight": _sample_weight(row),
        "metadata": metadata,
    }


def _embedding_triplet(row: Mapping[str, Any], *, unit_index: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "anchor": str(row.get("prompt") or ""),
        "positive": str(row.get("chosen") or ""),
        "negative": str(row.get("rejected") or ""),
        "positive_score": _optional_float(row.get("chosen_score")),
        "negative_score": _optional_float(row.get("rejected_score")),
        "sample_weight": _sample_weight(row),
        "metadata": _metadata_for_row(row, unit_index=unit_index, format_name="embedding_triplets"),
    }


def _embedding_ranked(row: Mapping[str, Any], *, unit_index: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "anchor": str(row.get("prompt") or ""),
        "texts": [str(item) for item in list(row.get("responses") or ())],
        "scores": [
            None if score is None else float(score)
            for score in list(row.get("scores") or ())
        ],
        "ranks": [int(rank) for rank in list(row.get("ranks") or ())],
        "sample_weight": _sample_weight(row),
        "metadata": _metadata_for_row(row, unit_index=unit_index, format_name="embedding_ranked"),
    }


def _sft(row: Mapping[str, Any], *, unit_index: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    score = row.get("score")
    return {
        "prompt": str(row.get("prompt") or ""),
        "completion": str(row.get("completion") or ""),
        "score": None if score is None else float(score),
        "sample_weight": _sample_weight(row),
        "metadata": _metadata_for_row(row, unit_index=unit_index, format_name="sft"),
    }


def _enrich_record(
    row: Mapping[str, Any],
    *,
    unit_index: Mapping[str, Mapping[str, Any]],
    format_name: str,
) -> dict[str, Any]:
    out = jsonable(dict(row))
    out["metadata"] = _metadata_for_row(row, unit_index=unit_index, format_name=format_name)
    return out


def _metadata_for_row(
    row: Mapping[str, Any],
    *,
    unit_index: Mapping[str, Mapping[str, Any]],
    format_name: str,
) -> dict[str, Any]:
    existing = dict(row.get("metadata") or {})
    unit_id = str(row.get("unit_id") or existing.get("unit_id") or "")
    unit = dict(unit_index.get(unit_id) or {})
    metadata = dict(unit.get("metadata") or {})
    metadata.update(existing)
    for field in _IDENTITY_FIELDS:
        value = unit.get(field, metadata.get(field))
        if value is not None:
            metadata[field] = value
    if unit.get("target") is not None:
        metadata["target"] = unit.get("target")
    metadata["format"] = str(format_name)
    metadata["sample_weight"] = _sample_weight(row)
    return jsonable(metadata)


def _sample_weight(row: Mapping[str, Any]) -> float:
    return float(row.get("sample_weight", 1.0) or 1.0)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(jsonable(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _save_hf_dataset_dict(views: Mapping[str, Sequence[Mapping[str, Any]]], path: Path) -> None:
    from datasets import Dataset, DatasetDict

    dataset = DatasetDict(
        {
            name: Dataset.from_list([jsonable(dict(row)) for row in rows])
            for name, rows in views.items()
        }
    )
    dataset.save_to_disk(str(path))



# Built-in adapter specs: (name, framework, required_views, description).
# Every entry uses the projected prepare function except ``generic_jsonl``,
# which exports the raw views.
_BUILTIN_ADAPTER_SPECS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "generic_jsonl",
        "generic",
        DEFAULT_FINETUNE_VIEWS,
        "Export all requested fine-tuning views as JSONL/JSON artifacts.",
    ),
    (
        "embedding",
        "embedding",
        ("embedding_pairs", "embedding_triplets", "embedding_ranked"),
        "Prepare pairs, triplets, and ranked rows for embedding trainers.",
    ),
    (
        "trl_sft",
        "trl",
        ("sft",),
        "Prepare prompt/completion rows for TRL SFT training.",
    ),
    (
        "trl_dpo",
        "trl",
        ("dpo",),
        "Prepare prompt/chosen/rejected rows for TRL DPO training.",
    ),
    (
        "trl_reward",
        "trl",
        ("reward",),
        "Prepare chosen/rejected pair rows for TRL reward training.",
    ),
    (
        "trl_scalar_reward",
        "trl",
        ("sft",),
        "Prepare prompt/response/score rows for scalar reward regression.",
    ),
    (
        "trl_grpo",
        "trl",
        ("grpo",),
        "Prepare offline ranked prompt groups for TRL-compatible GRPO workflows.",
    ),
    (
        "dspy_examples",
        "dspy",
        ("sft", "dpo"),
        "Prepare DSPy-style supervised and pairwise examples without importing DSPy.",
    ),
)


def _register_builtin_adapters() -> None:
    """Register the built-in fine-tuning adapters from ``_BUILTIN_ADAPTER_SPECS``."""
    for name, framework, required_views, description in _BUILTIN_ADAPTER_SPECS:
        prepare_fn = (
            _prepare_generic_adapter
            if name == "generic_jsonl"
            else _prepare_projected_adapter
        )
        register_finetune_adapter(
            FineTuneAdapter(
                name=name,
                framework=framework,
                required_views=required_views,
                description=description,
                prepare_fn=prepare_fn,
            ),
            replace=True,
        )


_register_builtin_adapters()


__all__ = [
    "DEFAULT_FINETUNE_VIEWS",
    "FineTuneAdapter",
    "FineTuneView",
    "build_finetune_views",
    "export_finetune_views",
    "export_for_adapter",
    "get_finetune_adapter",
    "list_finetune_adapters",
    "register_finetune_adapter",
]
