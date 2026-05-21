from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from treepo._research.runtime.contracts import utc_now_iso


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


@dataclass(frozen=True)
class StepEvent:
    run_id: str
    unit_id: str
    problem_id: str
    step_idx: int
    node_id: str
    action_type: str
    input_tokens: int
    output_tokens: int
    verifier_pass: bool
    failure_codes: List[str] = field(default_factory=list)
    repair_action: str = ""
    latency_ms: float = 0.0
    timestamp_utc: str = field(default_factory=utc_now_iso)

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["experiment_id"] = d.pop("run_id")
        # Flatten extras under a namespaced key.
        d["extra"] = dict(self.extra)
        return d


@dataclass(frozen=True)
class PredictionRecord:
    run_id: str
    unit_id: str
    phase_id: str
    benchmark: str
    task_id: str
    split: str
    max_seq_length: int
    seed: int
    method: str
    primary_metric: str

    problem_id: str
    prediction: str
    references: List[str]

    metrics: Dict[str, Any] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    failure: Optional[Dict[str, Any]] = None
    timestamp_utc: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["experiment_id"] = d.pop("run_id")
        return d


class TraceWriter:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

    def unit_steps_path(self, unit_id: str) -> Path:
        return self.run_dir / "units" / unit_id / "steps.jsonl"

    def unit_predictions_path(self, unit_id: str) -> Path:
        return self.run_dir / "units" / unit_id / "predictions.jsonl"

    def unit_calls_path(self, unit_id: str) -> Path:
        return self.run_dir / "units" / unit_id / "calls.jsonl"

    def write_step(self, unit_id: str, event: StepEvent) -> None:
        JsonlWriter(self.unit_steps_path(unit_id)).write(event.to_dict())

    def write_prediction(self, unit_id: str, record: PredictionRecord) -> None:
        JsonlWriter(self.unit_predictions_path(unit_id)).write(record.to_dict())

    def write_call_record(self, record: Dict[str, Any]) -> None:
        unit_id = str(record.get("unit_id") or "unknown")
        JsonlWriter(self.unit_calls_path(unit_id)).write(dict(record))
