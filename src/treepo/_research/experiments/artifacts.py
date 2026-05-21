from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping


ARTIFACT_SUMMARY_JSON = "summary_json"
ARTIFACT_SUMMARY_CSV = "summary_csv"
ARTIFACT_METRICS_JSON = "metrics_json"
ARTIFACT_PREDICTIONS_JSONL = "predictions_jsonl"
ARTIFACT_PREDICTIONS_CSV = "predictions_csv"
ARTIFACT_CALLS_JSONL = "calls_jsonl"
ARTIFACT_STEPS_JSONL = "steps_jsonl"
ARTIFACT_FINAL_STATS_JSON = "final_stats_json"
ARTIFACT_FULL_TREE_METRICS_JSON = "full_tree_metrics_json"
ARTIFACT_FULL_TREE_STATE_BLOBS_DIR = "full_tree_state_blobs_dir"
ARTIFACT_FULL_TREE_TRACES_JSONL = "full_tree_traces_jsonl"
ARTIFACT_TRAINING_RESULT_JSON = "training_result_json"
ARTIFACT_REPRODUCIBILITY_MANIFEST_JSON = "reproducibility_manifest_json"
ARTIFACT_CHECKPOINT_PATH = "checkpoint_path"
ARTIFACT_BEST_CHECKPOINT_PATH = "best_checkpoint_path"
ARTIFACT_FINAL_CHECKPOINT_PATH = "final_checkpoint_path"
ARTIFACT_OUTPUT_DIR = "output_dir"


def prefixed_artifact_key(prefix: str, key: str) -> str:
    clean_prefix = str(prefix or "").strip().strip("_")
    clean_key = str(key or "").strip().strip("_")
    return f"{clean_prefix}_{clean_key}" if clean_prefix else clean_key


def existing_artifact_paths(path_map: Mapping[str, Any]) -> Dict[str, str]:
    """Return non-empty artifact paths whose target exists on disk."""

    out: Dict[str, str] = {}
    for key, raw_path in dict(path_map).items():
        text = str(raw_path or "").strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if path.exists():
            out[str(key)] = str(path)
    return out


__all__ = [
    "ARTIFACT_BEST_CHECKPOINT_PATH",
    "ARTIFACT_CALLS_JSONL",
    "ARTIFACT_CHECKPOINT_PATH",
    "ARTIFACT_FINAL_CHECKPOINT_PATH",
    "ARTIFACT_FINAL_STATS_JSON",
    "ARTIFACT_FULL_TREE_METRICS_JSON",
    "ARTIFACT_FULL_TREE_STATE_BLOBS_DIR",
    "ARTIFACT_FULL_TREE_TRACES_JSONL",
    "ARTIFACT_METRICS_JSON",
    "ARTIFACT_OUTPUT_DIR",
    "ARTIFACT_PREDICTIONS_CSV",
    "ARTIFACT_PREDICTIONS_JSONL",
    "ARTIFACT_REPRODUCIBILITY_MANIFEST_JSON",
    "ARTIFACT_STEPS_JSONL",
    "ARTIFACT_SUMMARY_CSV",
    "ARTIFACT_SUMMARY_JSON",
    "ARTIFACT_TRAINING_RESULT_JSON",
    "existing_artifact_paths",
    "prefixed_artifact_key",
]
