"""Shared IPW sampling-artifact writers for task examples.

The manifesto examples all persist the same population/observed sampling rows
and a small propensity summary alongside their run outputs. These helpers keep
that logic in one place: rows are written as sorted JSONL, the summary reports
population/observed counts and the distinct inclusion probabilities, and
``write_sampling_artifacts`` standardizes the on-disk layout by always writing
into a ``sampling/`` subdirectory of the run output directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_sampling_rows_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write sampling rows to ``path`` as one sorted JSON object per line."""
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sampling_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize sampling rows: population/observed counts and propensities."""
    observed = [row for row in rows if bool(row.get("observed"))]
    propensities = sorted(
        {
            float(row["inclusion_probability"])
            for row in rows
            if row.get("inclusion_probability") is not None
        }
    )
    return {
        "population_count": len(rows),
        "observed_count": len(observed),
        "propensities": propensities,
    }


def write_sampling_artifacts(
    output_dir: Path,
    *,
    document_rows: list[dict[str, Any]],
    qsentence_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write document/qsentence sampling rows into ``output_dir / "sampling"``.

    Returns a payload with per-population summaries and the written file paths.
    """
    sampling_dir = output_dir / "sampling"
    sampling_dir.mkdir(parents=True, exist_ok=True)
    document_path = sampling_dir / "document_sampling_rows.jsonl"
    qsentence_path = sampling_dir / "qsentence_sampling_rows.jsonl"
    write_sampling_rows_jsonl(document_path, document_rows)
    write_sampling_rows_jsonl(qsentence_path, qsentence_rows)
    return {
        "summary": {
            "documents": sampling_summary(document_rows),
            "qsentences": sampling_summary(qsentence_rows),
        },
        "files": {
            "documents": str(document_path),
            "qsentences": str(qsentence_path),
        },
    }


__all__ = [
    "sampling_summary",
    "write_sampling_artifacts",
    "write_sampling_rows_jsonl",
]
