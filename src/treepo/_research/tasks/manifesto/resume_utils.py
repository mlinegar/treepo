"""Shared resume helper for per-manifesto pipeline scripts.

Pattern:
    already, resuming = load_resume_rows(path, key_field="manifesto_id")
    rows = list(already.values())
    with path.open("a" if resuming else "w") as fp:
        for item in all_items:
            k = key_of(item)
            if k in already:
                continue
            ...  # compute
            fp.write(json.dumps(row) + "\n"); fp.flush()
            rows.append(row)
            already[k] = row

Used by phase0_economic_pilot.py (originally), phase2_combined_pipeline.py,
phase3_full_pipeline_optimize.py, phase3_combined_optimize.py, rescore_variants.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def load_resume_rows(
    path: Path,
    key_field: str = "manifesto_id",
    log_label: str | None = None,
) -> tuple[dict[str, dict], bool]:
    """Parse a newline-delimited JSON file into a dict keyed by `key_field`.

    Returns (already_done_dict, is_resuming). `is_resuming` is True iff any
    rows were loaded; callers should open the output file in "a" mode in that
    case and "w" otherwise.

    Malformed lines and rows missing the key field are silently dropped —
    worst case is that one datum re-runs, which matches the phase0 precedent.
    """
    already: dict[str, dict] = {}
    if not path.exists():
        return already, False
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = row.get(key_field)
        if k is not None and k != "":
            already[str(k)] = row
    if already:
        logger.info(
            "Resume%s: %d rows already in %s",
            f" [{log_label}]" if log_label else "",
            len(already),
            path,
        )
    return already, bool(already)


def resume_compound_key(row: dict, fields: Iterable[str]) -> str | None:
    """Build a compound key (join field values with '|') or return None if any field missing."""
    parts = []
    for f in fields:
        v = row.get(f)
        if v is None:
            return None
        parts.append(str(v))
    return "|".join(parts)
