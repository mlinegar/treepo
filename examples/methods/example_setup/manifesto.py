"""Manifesto/RILE setup helpers for examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .configs import ManifestoReplicationConfig


def manifesto_grid_cells(config: ManifestoReplicationConfig) -> tuple[dict[str, Any], ...]:
    doc_unit_kind = str(config.doc_unit_kind or "qsentence")
    leaf_counts = tuple(int(v) for v in (config.leaf_unit_counts or (config.leaf_unit_count,)))
    supervision_grid = tuple(str(v) for v in (config.supervision_grid or (config.preference_mode,)))
    mode_units = {
        "none": "root",
        "scores": doc_unit_kind,
        "pairwise": doc_unit_kind,
        "ranked": doc_unit_kind,
    }
    cells: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for mode in supervision_grid:
        supervision_unit = mode_units.get(mode)
        if supervision_unit is None:
            raise ValueError("preference_mode/supervision_grid entries must be one of: none, scores, pairwise, ranked")
        unit_leaf_counts = leaf_counts if supervision_unit == "root" else (1,)
        for leaf_count in unit_leaf_counts:
            normalized_leaf_count = max(1, int(leaf_count or 1))
            key = (mode, normalized_leaf_count)
            if key in seen:
                continue
            seen.add(key)
            cells.append(
                {
                    "preference_mode": mode,
                    "leaf_unit_count": normalized_leaf_count,
                    "doc_unit_kind": doc_unit_kind,
                    "supervision_unit": supervision_unit,
                }
            )
    return tuple(cells)


def run_manifesto_replication_cell(
    *,
    config: ManifestoReplicationConfig,
    output_dir: Path,
    leaf_unit_count: int,
    preference_mode: str,
    prompt_template: str,
) -> dict[str, Any]:
    from treepo import fit
    from treepo.methods.preference import export_preference_records
    from treepo.tasks.manifesto import (
        make_manifesto_preferences,
        make_manifesto_replication_trees,
        manifesto_document_unit_sampling_rows,
        manifesto_oracle_predict_fn,
        sample_manifesto_replication_trees,
    )
    from treepo.tasks.sampling_artifacts import write_sampling_artifacts

    output_dir.mkdir(parents=True, exist_ok=True)
    train_population = make_manifesto_replication_trees(
        split="train",
        leaf_unit_count=leaf_unit_count,
        doc_unit_kind=config.doc_unit_kind,
    )
    train, document_sampling_rows = sample_manifesto_replication_trees(
        train_population,
        sample_size=config.doc_sample_size,
        sample_rate=config.doc_sample_rate,
        seed=int(config.doc_sample_seed),
    )
    eval_trees = make_manifesto_replication_trees(
        split="test",
        leaf_unit_count=leaf_unit_count,
        doc_unit_kind=config.doc_unit_kind,
    )
    qsentence_sample_size, qsentence_sample_rate, qsentence_sample_seed = resolved_qsentence_sampling(config)
    qsentence_sampling_rows = (
        manifesto_document_unit_sampling_rows(
            train,
            sample_size=qsentence_sample_size,
            sample_rate=qsentence_sample_rate,
            seed=qsentence_sample_seed,
        )
        if preference_mode != "none"
        else []
    )
    sampling_artifacts = write_sampling_artifacts(
        output_dir,
        document_rows=document_sampling_rows,
        qsentence_rows=qsentence_sampling_rows,
    )
    backend = {
        "output_dir": str(output_dir / "fit"),
        "model": config.model,
        "prompt_template": prompt_template,
        "min_score": -100.0,
        "max_score": 100.0,
    }
    if config.use_oracle_predictor:
        backend["predict_fn"] = manifesto_oracle_predict_fn
    preferences = None
    preference_artifacts: dict[str, object] = {}
    if preference_mode != "none":
        preferences = make_manifesto_preferences(
            train,
            mode=preference_mode,
            scope=config.preference_scope,
            sample_size=qsentence_sample_size,
            sample_rate=qsentence_sample_rate,
            seed=qsentence_sample_seed,
        )
        preference_artifacts = {
            "mode": preference_mode,
            "scope": config.preference_scope,
            **export_preference_records(preferences, output_dir / "preference"),
        }
    fit_config = {
        "family": config.family,
        "train_data": train,
        "eval_data": eval_trees,
        "backend_config": backend,
        "axis": {
            "max_iterations": config.max_iterations,
            "axis_value": int(leaf_unit_count),
            "axis_kind": "leaf_unit_count",
        },
    }
    if preferences is not None:
        fit_config["preference_data"] = preferences
    return {
        "result": fit(fit_config),
        "train": train,
        "train_population": train_population,
        "eval_trees": eval_trees,
        "preference_artifacts": preference_artifacts,
        "sampling_artifacts": sampling_artifacts,
        "preference_mode": preference_mode,
        "leaf_unit_count": int(leaf_unit_count),
        "doc_unit_kind": str(config.doc_unit_kind or "unit"),
    }


def resolved_qsentence_sampling(config: ManifestoReplicationConfig) -> tuple[int | None, float | None, int]:
    sample_size = config.qsentence_sample_size if config.qsentence_sample_size is not None else config.sample_size
    sample_rate = config.qsentence_sample_rate if config.qsentence_sample_rate is not None else config.sample_rate
    if sample_size is not None and sample_rate is not None:
        raise ValueError("pass qsentence_sample_size/sample_size or qsentence_sample_rate/sample_rate, not both")
    seed = config.qsentence_sample_seed if config.qsentence_sample_seed is not None else config.sample_seed
    return sample_size, sample_rate, int(seed)


def sample_desc(*, sample_size: int | None, sample_rate: float | None) -> str:
    if sample_size is not None:
        return f"n={sample_size}"
    if sample_rate is not None:
        return f"rate={sample_rate:g}"
    return "all"


def manifesto_finetune_fixture(output_dir: Path) -> tuple[list[Any], Path, Any]:
    from treepo.methods.preference import PreferenceDataset, PreferenceRecord, preference_units_from_trees
    from treepo.tasks.manifesto import (
        make_manifesto_preferences,
        make_manifesto_replication_trees,
        manifesto_tree_records,
    )
    from treepo.tree import write_tree_records_jsonl

    train = make_manifesto_replication_trees(split="train", leaf_unit_count=1, doc_unit_kind="qsentence")
    tree_records = manifesto_tree_records(train)
    supervised = preference_units_from_trees(tree_records)
    pairwise_qsentences = _retag_records(
        make_manifesto_preferences(train, mode="pairwise", scope="qsentences"),
        PreferenceRecord=PreferenceRecord,
        suffix="pairwise",
        role="qsentence_pairwise_candidates",
    )
    ranked_qsentences = _retag_records(
        make_manifesto_preferences(train, mode="ranked", scope="qsentences"),
        PreferenceRecord=PreferenceRecord,
        suffix="ranked",
        role="qsentence_ranked_candidates",
    )
    combined = PreferenceDataset.from_records(
        [
            *[PreferenceRecord.from_mapping(row) for row in supervised.to_records("general")],
            *pairwise_qsentences,
            *ranked_qsentences,
        ]
    )
    tree_path = write_tree_records_jsonl(output_dir / "manifesto_tree_records.jsonl", tree_records)
    return train, tree_path, combined


def _retag_records(dataset: Any, *, PreferenceRecord: Any, suffix: str, role: str) -> list[Any]:
    records: list[Any] = []
    for row in dataset.to_records("general"):
        payload = dict(row)
        base_unit_id = str(payload["unit_id"])
        payload["unit_id"] = f"{base_unit_id}:{suffix}"
        metadata = dict(payload.get("metadata") or {})
        metadata.update({"base_unit_id": base_unit_id, "fine_tune_role": role})
        payload["metadata"] = metadata
        records.append(PreferenceRecord.from_mapping(payload))
    return records


def manifesto_finetune_summary(views: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    sft_rows = views["sft"]
    return {
        "f_sft_rows": sum(1 for row in sft_rows if row["metadata"].get("target") == "f"),
        "g_sft_rows": sum(1 for row in sft_rows if row["metadata"].get("target") == "g"),
        "qsentence_triplets": sum(
            1 for row in views["embedding_triplets"] if row["metadata"].get("unit_type") == "qsentence"
        ),
        "qsentence_ranked_groups": sum(
            1 for row in views["embedding_ranked"] if row["metadata"].get("unit_type") == "qsentence"
        ),
    }

