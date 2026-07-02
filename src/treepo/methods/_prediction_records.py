"""Prediction-record artifact helpers for methods fit results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from treepo.methods._run_manifest import json_default


def collect_prediction_records(output_dir: Path) -> list[str]:
    """Return per-iteration prediction-record JSONL paths."""
    pred_dir = output_dir / "prediction_records"
    if not pred_dir.exists():
        return []
    return sorted(str(p) for p in pred_dir.glob("iter_*_post_eval.jsonl"))


def write_prediction_records(output_dir: Path, records: Sequence[Any]) -> None:
    pred_dir = output_dir / "prediction_records"
    for record in records:
        rows = (getattr(record, "extra", None) or {}).get("prediction_rows") or []
        if not rows:
            continue
        pred_dir.mkdir(parents=True, exist_ok=True)
        path = pred_dir / f"iter_{int(record.iteration):02d}_post_eval.jsonl"
        enriched = []
        for row in rows:
            payload = dict(row)
            payload.setdefault("iteration", int(record.iteration))
            payload.setdefault("stage_name", str(record.stage_name))
            payload.setdefault("family", str(record.family))
            enriched.append(payload)
        path.write_text(
            "\n".join(json.dumps(row, sort_keys=True, default=json_default) for row in enriched)
            + "\n",
            encoding="utf-8",
        )


__all__ = ["collect_prediction_records", "write_prediction_records"]
