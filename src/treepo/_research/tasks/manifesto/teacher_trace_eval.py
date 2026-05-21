"""Evaluation utilities for real-anchor teacher-trace local-law checks."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from treepo._research.tasks.manifesto.teacher_trace_generator import (
    TeacherTraceRecord,
    strict_same_side_raw,
)

RILE_RUBRIC = (
    "Preserve directional stance, policy signal intensity, factual commitments, and qualifying "
    "caveats when summarizing source content for information extraction."
)


class SummarizeFn(Protocol):
    def __call__(self, text: str, rubric: str, source_rile_raw: float, hop: int) -> str: ...


class MergeFn(Protocol):
    def __call__(self, left: str, right: str, rubric: str, source_rile_raw: float) -> str: ...


class ScoreFn(Protocol):
    def __call__(self, text: str) -> float: ...


@dataclass
class TeacherTraceEvalConfig:
    mode: str = "full"
    resummary_hops: int = 2

    c1_threshold_raw: float = 10.0
    c2_threshold_raw: float = 6.0
    c3_threshold_raw: float = 8.0
    neutral_raw: float = 0.0


@dataclass
class TeacherTracePrediction:
    example_id: str
    split: str
    source_manifesto_id: str
    source_bin_name: str
    source_rile_raw: float

    summary1: str
    summary2: str

    segment_a: str
    segment_b: str
    summary_a: str
    summary_b: str
    merged_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "split": self.split,
            "source_manifesto_id": self.source_manifesto_id,
            "source_bin_name": self.source_bin_name,
            "source_rile_raw": float(self.source_rile_raw),
            "summary1": self.summary1,
            "summary2": self.summary2,
            "segment_a": self.segment_a,
            "segment_b": self.segment_b,
            "summary_a": self.summary_a,
            "summary_b": self.summary_b,
            "merged_summary": self.merged_summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeacherTracePrediction":
        return cls(
            example_id=str(data.get("example_id", "")),
            split=str(data.get("split", "")),
            source_manifesto_id=str(data.get("source_manifesto_id", "")),
            source_bin_name=str(data.get("source_bin_name", "")),
            source_rile_raw=float(data.get("source_rile_raw", 0.0)),
            summary1=str(data.get("summary1", "")),
            summary2=str(data.get("summary2", "")),
            segment_a=str(data.get("segment_a", "")),
            segment_b=str(data.get("segment_b", "")),
            summary_a=str(data.get("summary_a", "")),
            summary_b=str(data.get("summary_b", "")),
            merged_summary=str(data.get("merged_summary", "")),
        )


@dataclass
class TeacherTraceEvalResult:
    example_id: str
    split: str
    source_manifesto_id: str
    source_bin_name: str

    source_rile_raw: float
    score_summary1_raw: float
    score_summary2_raw: float
    score_merge_raw: float

    score_segment_a_raw: float
    score_segment_b_raw: float
    score_merge_expected_raw: float

    c1_delta_raw: float
    c2_drift_raw: float
    c3_delta_raw: float

    c1_pass: bool
    c2_pass: bool
    c3_pass: bool

    same_side_summary1: bool
    same_side_summary2: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "split": self.split,
            "source_manifesto_id": self.source_manifesto_id,
            "source_bin_name": self.source_bin_name,
            "source_rile_raw": float(self.source_rile_raw),
            "score_summary1_raw": float(self.score_summary1_raw),
            "score_summary2_raw": float(self.score_summary2_raw),
            "score_merge_raw": float(self.score_merge_raw),
            "score_segment_a_raw": float(self.score_segment_a_raw),
            "score_segment_b_raw": float(self.score_segment_b_raw),
            "score_merge_expected_raw": float(self.score_merge_expected_raw),
            "c1_delta_raw": float(self.c1_delta_raw),
            "c2_drift_raw": float(self.c2_drift_raw),
            "c3_delta_raw": float(self.c3_delta_raw),
            "c1_pass": bool(self.c1_pass),
            "c2_pass": bool(self.c2_pass),
            "c3_pass": bool(self.c3_pass),
            "same_side_summary1": bool(self.same_side_summary1),
            "same_side_summary2": bool(self.same_side_summary2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeacherTraceEvalResult":
        return cls(
            example_id=str(data.get("example_id", "")),
            split=str(data.get("split", "")),
            source_manifesto_id=str(data.get("source_manifesto_id", "")),
            source_bin_name=str(data.get("source_bin_name", "")),
            source_rile_raw=float(data.get("source_rile_raw", 0.0)),
            score_summary1_raw=float(data.get("score_summary1_raw", 0.0)),
            score_summary2_raw=float(data.get("score_summary2_raw", 0.0)),
            score_merge_raw=float(data.get("score_merge_raw", 0.0)),
            score_segment_a_raw=float(data.get("score_segment_a_raw", 0.0)),
            score_segment_b_raw=float(data.get("score_segment_b_raw", 0.0)),
            score_merge_expected_raw=float(data.get("score_merge_expected_raw", 0.0)),
            c1_delta_raw=float(data.get("c1_delta_raw", 0.0)),
            c2_drift_raw=float(data.get("c2_drift_raw", 0.0)),
            c3_delta_raw=float(data.get("c3_delta_raw", 0.0)),
            c1_pass=bool(data.get("c1_pass", False)),
            c2_pass=bool(data.get("c2_pass", False)),
            c3_pass=bool(data.get("c3_pass", False)),
            same_side_summary1=bool(data.get("same_side_summary1", False)),
            same_side_summary2=bool(data.get("same_side_summary2", False)),
        )


def split_segments(text: str) -> Tuple[str, str]:
    rendered = str(text or "").strip()
    if not rendered:
        return "", ""
    if len(rendered) < 300:
        mid = len(rendered) // 2
        return rendered[:mid].strip(), rendered[mid:].strip()

    mid = len(rendered) // 2
    left = rendered.rfind("\n\n", 0, mid)
    right = rendered.find("\n\n", mid)
    candidates = [pos for pos in (left, right) if pos >= 0]
    if not candidates:
        cut = mid
    else:
        cut = min(candidates, key=lambda pos: abs(pos - mid))
    a = rendered[:cut].strip()
    b = rendered[cut:].strip()
    if not a or not b:
        cut = mid
        a = rendered[:cut].strip()
        b = rendered[cut:].strip()
    return a, b


def _length_weighted_mean(value_a: float, value_b: float, len_a: int, len_b: int) -> float:
    total = int(len_a) + int(len_b)
    if total <= 0:
        return 0.5 * (float(value_a) + float(value_b))
    return (float(value_a) * int(len_a) + float(value_b) * int(len_b)) / float(total)


def _resolve_worker_count(num_workers: int, workload_size: int, *, operation: str) -> int:
    requested = int(num_workers)
    if requested < 1:
        raise ValueError(f"{operation}: num_workers must be >= 1 (got {requested})")
    if int(workload_size) > 1 and requested < 2:
        raise ValueError(
            f"{operation}: single-worker mode is disabled for multi-item workloads. "
            f"Set num_workers >= 2 (got {requested}, workload_size={workload_size})."
        )
    return min(requested, max(1, int(workload_size)))


def build_predictions(
    records: Sequence[TeacherTraceRecord],
    *,
    summarize_fn: SummarizeFn,
    merge_fn: MergeFn,
    rubric: str = RILE_RUBRIC,
    resummary_hops: int = 2,
    num_workers: int = 1,
) -> List[TeacherTracePrediction]:
    hops = max(2, int(resummary_hops))

    def _predict_one(record: TeacherTraceRecord) -> TeacherTracePrediction:
        summary1 = str(
            summarize_fn(record.expanded_text, rubric, float(record.source_rile_raw), 1) or ""
        ).strip()
        summary2 = summary1
        for hop in range(2, hops + 1):
            summary2 = str(
                summarize_fn(summary2, rubric, float(record.source_rile_raw), hop) or ""
            ).strip()

        segment_a, segment_b = split_segments(record.expanded_text)
        summary_a = str(
            summarize_fn(segment_a, rubric, float(record.source_rile_raw), 1) or ""
        ).strip()
        summary_b = str(
            summarize_fn(segment_b, rubric, float(record.source_rile_raw), 1) or ""
        ).strip()
        merged = str(
            merge_fn(summary_a, summary_b, rubric, float(record.source_rile_raw)) or ""
        ).strip()

        return TeacherTracePrediction(
            example_id=record.example_id,
            split=record.split,
            source_manifesto_id=record.source_manifesto_id,
            source_bin_name=record.source_bin_name,
            source_rile_raw=float(record.source_rile_raw),
            summary1=summary1,
            summary2=summary2,
            segment_a=segment_a,
            segment_b=segment_b,
            summary_a=summary_a,
            summary_b=summary_b,
            merged_summary=merged,
        )

    worker_count = _resolve_worker_count(
        int(num_workers),
        len(records),
        operation="teacher_trace.build_predictions",
    )
    if worker_count <= 1 or len(records) <= 1:
        return [_predict_one(record) for record in records]

    slots: List[Optional[TeacherTracePrediction]] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_to_idx = {pool.submit(_predict_one, record): idx for idx, record in enumerate(records)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            slots[idx] = future.result()
    return [row for row in slots if row is not None]


def score_predictions(
    records: Sequence[TeacherTraceRecord],
    predictions: Sequence[TeacherTracePrediction],
    *,
    score_fn: ScoreFn,
    config: TeacherTraceEvalConfig,
    num_workers: int = 1,
) -> List[TeacherTraceEvalResult]:
    by_id: Dict[str, TeacherTraceRecord] = {record.example_id: record for record in records}

    def _score_one(prediction: TeacherTracePrediction) -> Optional[TeacherTraceEvalResult]:
        record = by_id.get(prediction.example_id)
        if record is None:
            return None

        y_true = float(record.source_rile_raw)
        score_s1 = float(score_fn(prediction.summary1))
        score_s2 = float(score_fn(prediction.summary2))
        score_merge = float(score_fn(prediction.merged_summary))
        score_seg_a = float(score_fn(prediction.segment_a)) if prediction.segment_a else 0.0
        score_seg_b = float(score_fn(prediction.segment_b)) if prediction.segment_b else 0.0

        expected_merge = _length_weighted_mean(
            score_seg_a,
            score_seg_b,
            len(prediction.segment_a),
            len(prediction.segment_b),
        )

        c1_delta = float(score_s1 - y_true)
        c2_drift = float(score_s2 - score_s1)
        c3_delta = float(score_merge - expected_merge)

        same_side_s1 = strict_same_side_raw(
            score_s1,
            y_true,
            neutral_raw=float(config.neutral_raw),
        )
        same_side_s2 = strict_same_side_raw(
            score_s2,
            y_true,
            neutral_raw=float(config.neutral_raw),
        )

        c1_pass = abs(c1_delta) <= float(config.c1_threshold_raw)
        c2_pass = bool(
            abs(c2_drift) <= float(config.c2_threshold_raw)
            and same_side_s1
            and same_side_s2
        )
        c3_pass = abs(c3_delta) <= float(config.c3_threshold_raw)

        return TeacherTraceEvalResult(
            example_id=prediction.example_id,
            split=prediction.split,
            source_manifesto_id=prediction.source_manifesto_id,
            source_bin_name=prediction.source_bin_name,
            source_rile_raw=y_true,
            score_summary1_raw=score_s1,
            score_summary2_raw=score_s2,
            score_merge_raw=score_merge,
            score_segment_a_raw=score_seg_a,
            score_segment_b_raw=score_seg_b,
            score_merge_expected_raw=float(expected_merge),
            c1_delta_raw=c1_delta,
            c2_drift_raw=c2_drift,
            c3_delta_raw=c3_delta,
            c1_pass=bool(c1_pass),
            c2_pass=bool(c2_pass),
            c3_pass=bool(c3_pass),
            same_side_summary1=bool(same_side_s1),
            same_side_summary2=bool(same_side_s2),
        )

    worker_count = _resolve_worker_count(
        int(num_workers),
        len(predictions),
        operation="teacher_trace.score_predictions",
    )
    if worker_count <= 1 or len(predictions) <= 1:
        return [row for row in (_score_one(prediction) for prediction in predictions) if row is not None]

    slots: List[Optional[TeacherTraceEvalResult]] = [None] * len(predictions)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_to_idx = {pool.submit(_score_one, prediction): idx for idx, prediction in enumerate(predictions)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            slots[idx] = future.result()
    return [row for row in slots if row is not None]


def _empty_metric_row() -> Dict[str, Any]:
    return {
        "n": 0,
        "c1_pass_rate": None,
        "c2_pass_rate": None,
        "c3_pass_rate": None,
        "avg_law_pass_rate": None,
        "same_side_rate": None,
        "same_side_summary2_rate": None,
        "c1_mae_raw": None,
        "summary2_mae_raw": None,
        "c2_drift_mae_raw": None,
        "c3_mae_raw": None,
    }


def compute_metric_row(results: Sequence[TeacherTraceEvalResult]) -> Dict[str, Any]:
    if not results:
        return _empty_metric_row()

    n = len(results)
    c1_rate = float(sum(1 for row in results if row.c1_pass) * 100.0 / n)
    c2_rate = float(sum(1 for row in results if row.c2_pass) * 100.0 / n)
    c3_rate = float(sum(1 for row in results if row.c3_pass) * 100.0 / n)
    avg_law = float((c1_rate + c2_rate + c3_rate) / 3.0)

    c1_mae = float(sum(abs(float(row.c1_delta_raw)) for row in results) / n)
    s2_mae = float(
        sum(abs(float(row.score_summary2_raw - row.source_rile_raw)) for row in results) / n
    )
    c2_mae = float(sum(abs(float(row.c2_drift_raw)) for row in results) / n)
    c3_mae = float(sum(abs(float(row.c3_delta_raw)) for row in results) / n)

    same_side_rate = float(sum(1 for row in results if row.same_side_summary1) * 100.0 / n)
    same_side_s2_rate = float(sum(1 for row in results if row.same_side_summary2) * 100.0 / n)

    return {
        "n": n,
        "c1_pass_rate": c1_rate,
        "c2_pass_rate": c2_rate,
        "c3_pass_rate": c3_rate,
        "avg_law_pass_rate": avg_law,
        "same_side_rate": same_side_rate,
        "same_side_summary2_rate": same_side_s2_rate,
        "c1_mae_raw": c1_mae,
        "summary2_mae_raw": s2_mae,
        "c2_drift_mae_raw": c2_mae,
        "c3_mae_raw": c3_mae,
    }


def _group_results(
    results: Sequence[TeacherTraceEvalResult],
    key_fn: Callable[[TeacherTraceEvalResult], str],
) -> Dict[str, List[TeacherTraceEvalResult]]:
    grouped: Dict[str, List[TeacherTraceEvalResult]] = defaultdict(list)
    for row in results:
        grouped[key_fn(row)].append(row)
    return dict(grouped)


def compute_group_metrics(results: Sequence[TeacherTraceEvalResult]) -> Dict[str, Any]:
    by_split = {
        split: compute_metric_row(rows)
        for split, rows in sorted(_group_results(results, lambda row: row.split).items())
    }
    by_bin = {
        bin_name: compute_metric_row(rows)
        for bin_name, rows in sorted(_group_results(results, lambda row: row.source_bin_name).items())
    }
    return {
        "split": by_split,
        "source_bin": by_bin,
    }


def build_eval_metrics(
    results: Sequence[TeacherTraceEvalResult],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    overall = compute_metric_row(results)
    groups = compute_group_metrics(results)
    metrics = {"overall": overall}
    return metrics, groups


def render_eval_report_markdown(metrics: Dict[str, Any], groups: Dict[str, Any]) -> str:
    overall = metrics.get("overall") or {}

    lines: List[str] = []
    lines.append("# Teacher-Trace Local-Law Evaluation Report")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- n: {overall.get('n')}")
    lines.append(f"- c1_pass_rate: {overall.get('c1_pass_rate')}")
    lines.append(f"- c2_pass_rate: {overall.get('c2_pass_rate')}")
    lines.append(f"- c3_pass_rate: {overall.get('c3_pass_rate')}")
    lines.append(f"- avg_law_pass_rate: {overall.get('avg_law_pass_rate')}")
    lines.append(f"- same_side_rate: {overall.get('same_side_rate')}")
    lines.append(f"- same_side_summary2_rate: {overall.get('same_side_summary2_rate')}")
    lines.append(f"- c1_mae_raw: {overall.get('c1_mae_raw')}")
    lines.append(f"- summary2_mae_raw: {overall.get('summary2_mae_raw')}")
    lines.append(f"- c2_drift_mae_raw: {overall.get('c2_drift_mae_raw')}")
    lines.append(f"- c3_mae_raw: {overall.get('c3_mae_raw')}")
    lines.append("")

    lines.append("## By Split")
    lines.append("")
    for split, row in sorted((groups.get("split") or {}).items()):
        lines.append(f"### {split}")
        lines.append(f"- n: {row.get('n')}")
        lines.append(f"- c1_pass_rate: {row.get('c1_pass_rate')}")
        lines.append(f"- c2_pass_rate: {row.get('c2_pass_rate')}")
        lines.append(f"- c3_pass_rate: {row.get('c3_pass_rate')}")
        lines.append(f"- avg_law_pass_rate: {row.get('avg_law_pass_rate')}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_predictions_jsonl(path: Path, predictions: Sequence[TeacherTracePrediction]) -> None:
    write_jsonl(path, (row.to_dict() for row in predictions))


def load_predictions_jsonl(path: Path) -> List[TeacherTracePrediction]:
    loaded: List[TeacherTracePrediction] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            loaded.append(TeacherTracePrediction.from_dict(json.loads(text)))
    return loaded


def write_eval_results_jsonl(path: Path, results: Sequence[TeacherTraceEvalResult]) -> None:
    write_jsonl(path, (row.to_dict() for row in results))


__all__ = [
    "TeacherTraceEvalConfig",
    "TeacherTraceEvalResult",
    "TeacherTracePrediction",
    "RILE_RUBRIC",
    "build_eval_metrics",
    "build_predictions",
    "compute_group_metrics",
    "compute_metric_row",
    "load_predictions_jsonl",
    "render_eval_report_markdown",
    "score_predictions",
    "split_segments",
    "write_eval_results_jsonl",
    "write_predictions_jsonl",
]
