"""Small artifact-bundle helpers for treepo examples and adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo.local_law import LocalLawAuditRow, audit_local_laws
from treepo.methods.preference import PreferenceDataset, export_preference_records
from treepo.common import jsonable
from treepo.tree import (
    TreeRecord,
    local_law_rows_from_tree_records,
    tree_summary,
    write_tree_records_jsonl,
)


def write_artifact_bundle(
    output_dir: Path | str,
    *,
    trees: Sequence[Any] | None = None,
    preference_data: Any = None,
    local_law_rows: Sequence[LocalLawAuditRow | Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a compact bundle using treepo's canonical artifact shapes.

    The helper is intentionally small and example-oriented. It serializes tree
    records, preference projections, annotated local-law rows, and a manifest
    pointing at those files.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    payload: dict[str, Any] = {
        "version": "0.1",
        "files": files,
        "metadata": jsonable(dict(metadata or {})),
    }

    tree_records: list[TreeRecord] = []
    if trees is not None:
        tree_records = [TreeRecord.from_value(tree) for tree in trees]
        tree_path = write_tree_records_jsonl(out / "tree_records.jsonl", tree_records)
        files["tree_records"] = str(tree_path)
        summaries = [tree_summary(tree) for tree in tree_records]
        payload["trees"] = {
            "n_trees": len(tree_records),
            "n_nodes": sum(int(summary["n_nodes"]) for summary in summaries),
            "n_leaves": sum(int(summary["n_leaves"]) for summary in summaries),
            "validation_error_count": sum(len(summary["validation_errors"]) for summary in summaries),
            "summaries": summaries,
        }

    if preference_data is not None:
        preferences = PreferenceDataset.from_value(preference_data)
        if len(preferences) > 0:
            payload["preferences"] = export_preference_records(preferences, out / "preferences")

    rows = _coerce_rows(local_law_rows)
    if local_law_rows is None and tree_records:
        rows = list(local_law_rows_from_tree_records(tree_records))
    if rows:
        rows_path = out / "local_law_rows.jsonl"
        rows_path.write_text(
            "".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        files["local_law_rows"] = str(rows_path)
        payload["local_laws"] = audit_local_laws(rows)

    manifest_path = out / "treepo_artifact_bundle.json"
    files["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _coerce_rows(rows: Sequence[LocalLawAuditRow | Mapping[str, Any]] | None) -> list[LocalLawAuditRow]:
    if rows is None:
        return []
    out: list[LocalLawAuditRow] = []
    for row in rows:
        if isinstance(row, LocalLawAuditRow):
            out.append(row)
        else:
            out.append(LocalLawAuditRow(**dict(row)))
    return out


__all__ = ["write_artifact_bundle"]
