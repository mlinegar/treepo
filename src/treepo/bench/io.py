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


def summary_to_csv_row_segmented(summary: Any) -> Dict[str, object]:
    """
    Match `scripts/run_segmented_lda_ctreepo_simulation.py` CSV schema:
      - config_* fields
      - topic_* fields
      - per-policy metrics: <policy>_<metric>
      - decomposition_* fields
      - optional selection_audit_* fields
    """
    config = dict(getattr(summary, "config"))
    topic_meta = dict(getattr(summary, "topic_meta"))
    metrics = dict(getattr(summary, "metrics"))
    decomposition = getattr(summary, "decomposition")
    selection_audit = getattr(summary, "selection_audit")

    row: Dict[str, object] = {f"config_{k}": v for k, v in config.items()}
    row.update({f"topic_{k}": v for k, v in topic_meta.items()})
    row["calibration_samples"] = int(getattr(summary, "calibration_samples"))

    # metrics: dict[str, PolicyMetrics]
    for policy, m in metrics.items():
        md = m if isinstance(m, dict) else getattr(m, "__dict__", None)
        if md is None:
            try:
                from dataclasses import asdict

                md = asdict(m)
            except Exception:
                continue
        for k, v in dict(md).items():
            row[f"{policy}_{k}"] = v

    # decomposition: dataclass
    try:
        from dataclasses import asdict

        for k, v in asdict(decomposition).items():
            row[f"decomposition_{k}"] = v
    except Exception:
        for k, v in getattr(decomposition, "__dict__", {}).items():
            row[f"decomposition_{k}"] = v

    if selection_audit is not None:
        row["selection_audit_trials"] = int(getattr(selection_audit, "trials"))
        row["selection_audit_mean_sample_size"] = float(getattr(selection_audit, "mean_sample_size"))
        row["selection_audit_mean_effective_sample_size"] = float(getattr(selection_audit, "mean_effective_sample_size"))
        row["selection_audit_ipw_ci_coverage"] = float(getattr(selection_audit, "ipw_violation_ci_coverage"))
        row["selection_audit_ipw_ci_mean_radius"] = float(getattr(selection_audit, "ipw_violation_ci_mean_radius"))
    return row


def summary_to_csv_rows_ops(summary: Any) -> List[Dict[str, object]]:
    """
    Match `scripts/run_segment_lda_ops_weight_recovery_simulation.py` CSV schema:
    one row per method in summary.metrics.
    """
    cfg = dict(getattr(summary, "config"))
    geom = dict(getattr(summary, "training_geometry"))
    wt = dict(getattr(summary, "weight_truth"))

    out: List[Dict[str, object]] = []
    metrics = getattr(summary, "metrics")
    if not isinstance(metrics, dict):
        return out
    for name, m in metrics.items():
        if not isinstance(m, dict):
            continue
        row: Dict[str, object] = {
            "method": str(name),
            **{f"cfg_{k}": cfg.get(k) for k in cfg.keys()},
            **{f"train_{k}": geom.get(k) for k in geom.keys()},
            "relevant_topics": ",".join(str(x) for x in wt.get("relevant_topics", [])),
            "lambda_multiplier": wt.get("lambda_multiplier"),
        }
        row.update(m)
        out.append(row)
    return out


def summary_to_csv_row_ops(summary: Any) -> Any:
    # Backwards-compat alias: OPS summaries naturally emit multiple rows.
    return summary_to_csv_rows_ops(summary)


def summary_to_csv_row_learned_ops_g(summary: Any) -> Dict[str, object]:
    cfg = dict(getattr(summary, "config"))
    wt = dict(getattr(summary, "weight_truth", {}))
    geom = dict(getattr(summary, "training_geometry", {}))
    metrics = dict(getattr(summary, "metrics", {}))

    row: Dict[str, object] = {f"config_{k}": v for k, v in cfg.items()}
    row.update({f"weight_{k}": v for k, v in wt.items()})
    row.update({f"train_{k}": v for k, v in geom.items()})
    row.update({f"metric_{k}": v for k, v in metrics.items()})
    row["experiment"] = "learned_segment_lda_ops_g"
    return row


def summary_to_csv_row_learned_segmented_theta_g(summary: Any) -> Dict[str, object]:
    cfg = dict(getattr(summary, "config"))
    geom = dict(getattr(summary, "training_geometry", {}))
    metrics = dict(getattr(summary, "metrics", {}))

    row: Dict[str, object] = {f"config_{k}": v for k, v in cfg.items()}
    row.update({f"train_{k}": v for k, v in geom.items()})
    row.update({f"metric_{k}": v for k, v in metrics.items()})
    row["experiment"] = "learned_segmented_lda_theta_g"
    return row


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
