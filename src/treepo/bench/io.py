from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from treepo import __version__


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
    except Exception:
        return None
    sha = out.decode("utf-8", errors="replace").strip()
    return sha if sha else None


def runtime_meta() -> Dict[str, object]:
    return {
        "treepo_version": str(__version__),
        "python_version": str(sys.version).split()[0],
        "numpy_version": str(np.__version__),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": platform.platform(),
        "git_sha": _git_sha(),
    }


def add_runtime_meta(payload: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    out["runtime_meta"] = runtime_meta()
    return out


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return

    keys: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k in seen:
                continue
            seen.add(k)
            keys.append(str(k))

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})


JsonLike = Union[Dict[str, Any], List[Any]]


def load_yaml_or_json(path: Path) -> JsonLike:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        return json.loads(raw)

    # YAML first; if unavailable or fails, fall back to JSON for robustness.
    try:
        import yaml  # type: ignore[import-not-found]

        return yaml.safe_load(raw)
    except Exception:
        return json.loads(raw)


def dump_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def summary_to_csv_rows_cardinality_recovery(summary: Any) -> List[Dict[str, object]]:
    from treepo.bench.cardinality_recovery import experiment_rows

    results = getattr(summary, "results", ())
    rows = experiment_rows(results)
    for row in rows:
        row["experiment"] = "cardinality_recovery"
    return rows


def summary_to_csv_rows_hll_merge_learning(summary: Any) -> List[Dict[str, object]]:
    from treepo.bench.hll_merge_learning import experiment_rows

    results = getattr(summary, "results", ())
    rows = experiment_rows(results)
    for row in rows:
        row["experiment"] = "hll_merge_learning"
    return rows


def summary_to_csv_rows_classical_sketches(summary: Any) -> List[Dict[str, object]]:
    from treepo.bench.classical_sketches import experiment_rows

    rows = experiment_rows(summary)
    for row in rows:
        row["experiment"] = "classical_sketches"
    return rows
