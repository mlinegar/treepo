"""Evaluation utilities for local-law stress benchmark records."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from treepo._research.tasks.manifesto.lawstress_generator import LawStressRecord, normalize_rile

RILE_RUBRIC = (
    "Preserve directional stance, policy signal intensity, factual commitments, and qualifying caveats "
    "when summarizing source content for information extraction."
)


class SummarizeFn(Protocol):
    def __call__(self, text: str, rubric: str) -> str: ...


class MergeFn(Protocol):
    def __call__(self, left: str, right: str, rubric: str) -> str: ...


class ScoreFn(Protocol):
    def __call__(self, text: str) -> float: ...


class JudgeFn(Protocol):
    def __call__(
        self,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str,
    ) -> Any: ...


@dataclass
class LawStressEvalConfig:
    mode: str = "full"
    chunk_size: int = 2000
    resummary_hops: int = 2

    c1_threshold_norm: float = 0.10
    c2_threshold_norm: float = 0.06
    c3_threshold_norm: float = 0.08
    neutral_norm: float = 0.5

    abs_hard_same_side_min: float = 58.0
    abs_hard_c1_min: float = 52.0
    abs_hard_c2_min: float = 56.0
    abs_hard_c3_min: float = 50.0

    abs_control_same_side_min: float = 75.0
    abs_control_c1_min: float = 70.0
    abs_control_c2_min: float = 70.0
    abs_control_c3_min: float = 70.0

    rel_same_side_gain_min: float = 8.0
    rel_c_gain_min: float = 5.0
    rel_mae_gain_min: float = 0.01


@dataclass
class LawStressPrediction:
    example_id: str
    split: str
    difficulty: str
    law_target: str
    family: str
    bin_name: str
    summary1: str
    summary2: str
    summary_a: str
    summary_b: str
    merged_summary: str
    reference_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "split": self.split,
            "difficulty": self.difficulty,
            "law_target": self.law_target,
            "family": self.family,
            "bin_name": self.bin_name,
            "summary1": self.summary1,
            "summary2": self.summary2,
            "summary_a": self.summary_a,
            "summary_b": self.summary_b,
            "merged_summary": self.merged_summary,
            "reference_summary": self.reference_summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LawStressPrediction":
        return cls(
            example_id=str(data.get("example_id", "")),
            split=str(data.get("split", "")),
            difficulty=str(data.get("difficulty", "")),
            law_target=str(data.get("law_target", "")),
            family=str(data.get("family", "")),
            bin_name=str(data.get("bin_name", "")),
            summary1=str(data.get("summary1", "")),
            summary2=str(data.get("summary2", "")),
            summary_a=str(data.get("summary_a", "")),
            summary_b=str(data.get("summary_b", "")),
            merged_summary=str(data.get("merged_summary", "")),
            reference_summary=str(data.get("reference_summary", "")),
        )


@dataclass
class LawStressEvalResult:
    example_id: str
    split: str
    difficulty: str
    law_target: str
    family: str
    bin_name: str

    y_true_raw: float
    y_true_norm: float
    y_merge_expected_raw: float
    y_merge_expected_norm: float

    score_summary1_raw: float
    score_summary1_norm: float
    score_summary2_raw: float
    score_summary2_norm: float
    score_merge_raw: float
    score_merge_norm: float

    c1_pass: bool
    c2_pass: bool
    c3_pass: bool

    error_norm: float
    same_side_of_neutral: bool

    genrm_preferred: Optional[str] = None
    genrm_confidence: Optional[float] = None
    genrm_tie_or_win: Optional[bool] = None
    genrm_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "split": self.split,
            "difficulty": self.difficulty,
            "law_target": self.law_target,
            "family": self.family,
            "bin_name": self.bin_name,
            "y_true_raw": float(self.y_true_raw),
            "y_true_norm": float(self.y_true_norm),
            "y_merge_expected_raw": float(self.y_merge_expected_raw),
            "y_merge_expected_norm": float(self.y_merge_expected_norm),
            "score_summary1_raw": float(self.score_summary1_raw),
            "score_summary1_norm": float(self.score_summary1_norm),
            "score_summary2_raw": float(self.score_summary2_raw),
            "score_summary2_norm": float(self.score_summary2_norm),
            "score_merge_raw": float(self.score_merge_raw),
            "score_merge_norm": float(self.score_merge_norm),
            "c1_pass": bool(self.c1_pass),
            "c2_pass": bool(self.c2_pass),
            "c3_pass": bool(self.c3_pass),
            "error_norm": float(self.error_norm),
            "same_side_of_neutral": bool(self.same_side_of_neutral),
            "genrm_preferred": self.genrm_preferred,
            "genrm_confidence": self.genrm_confidence,
            "genrm_tie_or_win": self.genrm_tie_or_win,
            "genrm_error": self.genrm_error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LawStressEvalResult":
        return cls(
            example_id=str(data.get("example_id", "")),
            split=str(data.get("split", "")),
            difficulty=str(data.get("difficulty", "")),
            law_target=str(data.get("law_target", "")),
            family=str(data.get("family", "")),
            bin_name=str(data.get("bin_name", "")),
            y_true_raw=float(data.get("y_true_raw", 0.0)),
            y_true_norm=float(data.get("y_true_norm", 0.5)),
            y_merge_expected_raw=float(data.get("y_merge_expected_raw", 0.0)),
            y_merge_expected_norm=float(data.get("y_merge_expected_norm", 0.5)),
            score_summary1_raw=float(data.get("score_summary1_raw", 0.0)),
            score_summary1_norm=float(data.get("score_summary1_norm", 0.5)),
            score_summary2_raw=float(data.get("score_summary2_raw", 0.0)),
            score_summary2_norm=float(data.get("score_summary2_norm", 0.5)),
            score_merge_raw=float(data.get("score_merge_raw", 0.0)),
            score_merge_norm=float(data.get("score_merge_norm", 0.5)),
            c1_pass=bool(data.get("c1_pass", False)),
            c2_pass=bool(data.get("c2_pass", False)),
            c3_pass=bool(data.get("c3_pass", False)),
            error_norm=float(data.get("error_norm", 0.0)),
            same_side_of_neutral=bool(data.get("same_side_of_neutral", False)),
            genrm_preferred=data.get("genrm_preferred"),
            genrm_confidence=float(data["genrm_confidence"]) if data.get("genrm_confidence") is not None else None,
            genrm_tie_or_win=bool(data["genrm_tie_or_win"]) if data.get("genrm_tie_or_win") is not None else None,
            genrm_error=data.get("genrm_error"),
        )


def strict_same_side(pred_norm: float, true_norm: float, neutral_norm: float = 0.5) -> bool:
    pred_delta = float(pred_norm) - float(neutral_norm)
    if abs(pred_delta) <= 1e-9:
        return False
    true_delta = float(true_norm) - float(neutral_norm)
    return bool(pred_delta * true_delta > 0.0)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = float(len(xs))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    if den_x <= 0.0 or den_y <= 0.0:
        return None
    return float(numerator / math.sqrt(den_x * den_y))


def _average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


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
    records: Sequence[LawStressRecord],
    *,
    summarize_fn: SummarizeFn,
    merge_fn: MergeFn,
    rubric: str = RILE_RUBRIC,
    resummary_hops: int = 2,
    num_workers: int = 1,
) -> List[LawStressPrediction]:
    """Create model summaries needed for law checks."""

    hops = max(2, int(resummary_hops))

    def _predict_one(record: LawStressRecord) -> LawStressPrediction:
        summary1 = str(summarize_fn(record.text, rubric) or "").strip()
        summary2 = summary1
        for _ in range(hops - 1):
            summary2 = str(summarize_fn(summary2, rubric) or "").strip()

        summary_a = str(summarize_fn(record.segment_a, rubric) or "").strip()
        summary_b = str(summarize_fn(record.segment_b, rubric) or "").strip()
        merged_summary = str(merge_fn(summary_a, summary_b, rubric) or "").strip()

        return LawStressPrediction(
            example_id=record.example_id,
            split=record.split,
            difficulty=record.difficulty,
            law_target=record.law_target,
            family=record.family,
            bin_name=record.bin_name,
            summary1=summary1,
            summary2=summary2,
            summary_a=summary_a,
            summary_b=summary_b,
            merged_summary=merged_summary,
            reference_summary=record.reference_summary,
        )

    worker_count = _resolve_worker_count(
        int(num_workers),
        len(records),
        operation="lawstress.build_predictions",
    )
    if worker_count <= 1 or len(records) <= 1:
        return [_predict_one(record) for record in records]

    slots: List[Optional[LawStressPrediction]] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_to_idx = {pool.submit(_predict_one, record): idx for idx, record in enumerate(records)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            slots[idx] = future.result()
    return [row for row in slots if row is not None]


def _parse_judge_payload(payload: Any) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    preferred: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None

    if payload is None:
        return None, None, "empty_judge_result"

    if isinstance(payload, dict):
        preferred = payload.get("preferred")
        conf_raw = payload.get("confidence")
        if conf_raw is not None:
            try:
                confidence = float(conf_raw)
            except (TypeError, ValueError):
                confidence = None
        if payload.get("error"):
            error = str(payload.get("error"))
    else:
        preferred = getattr(payload, "preferred", None)
        conf_raw = getattr(payload, "confidence", None)
        if conf_raw is not None:
            try:
                confidence = float(conf_raw)
            except (TypeError, ValueError):
                confidence = None
        if hasattr(payload, "error_message") and getattr(payload, "error_message", None):
            error = str(getattr(payload, "error_message"))

    if preferred is not None:
        preferred = str(preferred).strip().upper()
        if preferred not in {"A", "B", "TIE"}:
            if preferred == "TIE":
                preferred = "TIE"
            elif preferred == "A":
                preferred = "A"
            elif preferred == "B":
                preferred = "B"
            else:
                preferred = None

    return preferred, confidence, error


def score_and_judge_predictions(
    records: Sequence[LawStressRecord],
    predictions: Sequence[LawStressPrediction],
    *,
    score_fn: ScoreFn,
    judge_fn: Optional[JudgeFn],
    config: LawStressEvalConfig,
    rubric: str = RILE_RUBRIC,
    num_workers: int = 1,
) -> List[LawStressEvalResult]:
    """Score summaries and compute local-law outcomes."""

    by_id: Dict[str, LawStressRecord] = {record.example_id: record for record in records}

    def _score_one(prediction: LawStressPrediction) -> Optional[LawStressEvalResult]:
        record = by_id.get(prediction.example_id)
        if record is None:
            return None

        score_s1_raw = float(score_fn(prediction.summary1))
        score_s2_raw = float(score_fn(prediction.summary2))
        score_merge_raw = float(score_fn(prediction.merged_summary))

        score_s1_norm = normalize_rile(score_s1_raw)
        score_s2_norm = normalize_rile(score_s2_raw)
        score_merge_norm = normalize_rile(score_merge_raw)

        y_true_norm = normalize_rile(record.y_raw)
        y_merge_expected_norm = normalize_rile(record.y_merge_expected_raw)

        c1_pass = abs(score_s1_norm - y_true_norm) <= float(config.c1_threshold_norm)
        c2_close = abs(score_s2_norm - score_s1_norm) <= float(config.c2_threshold_norm)
        c2_same_side = strict_same_side(score_s2_norm, score_s1_norm, neutral_norm=config.neutral_norm)
        c2_pass = bool(c2_close and c2_same_side)
        c3_pass = abs(score_merge_norm - y_merge_expected_norm) <= float(config.c3_threshold_norm)

        same_side = strict_same_side(score_s1_norm, y_true_norm, neutral_norm=config.neutral_norm)

        genrm_preferred: Optional[str] = None
        genrm_confidence: Optional[float] = None
        genrm_tie_or_win: Optional[bool] = None
        genrm_error: Optional[str] = None

        if judge_fn is not None and prediction.reference_summary:
            try:
                judge_payload = judge_fn(
                    rubric,
                    record.text,
                    prediction.summary1,
                    prediction.reference_summary,
                    "sufficiency",
                )
                parsed_preferred, parsed_confidence, parsed_error = _parse_judge_payload(judge_payload)
                genrm_preferred = parsed_preferred
                genrm_confidence = parsed_confidence
                genrm_error = parsed_error
                if parsed_preferred is not None:
                    genrm_tie_or_win = parsed_preferred in {"A", "TIE"}
            except Exception as exc:
                genrm_error = str(exc)
                genrm_tie_or_win = None

        return LawStressEvalResult(
            example_id=record.example_id,
            split=record.split,
            difficulty=record.difficulty,
            law_target=record.law_target,
            family=record.family,
            bin_name=record.bin_name,
            y_true_raw=float(record.y_raw),
            y_true_norm=float(y_true_norm),
            y_merge_expected_raw=float(record.y_merge_expected_raw),
            y_merge_expected_norm=float(y_merge_expected_norm),
            score_summary1_raw=float(score_s1_raw),
            score_summary1_norm=float(score_s1_norm),
            score_summary2_raw=float(score_s2_raw),
            score_summary2_norm=float(score_s2_norm),
            score_merge_raw=float(score_merge_raw),
            score_merge_norm=float(score_merge_norm),
            c1_pass=bool(c1_pass),
            c2_pass=bool(c2_pass),
            c3_pass=bool(c3_pass),
            error_norm=abs(score_s1_norm - y_true_norm),
            same_side_of_neutral=bool(same_side),
            genrm_preferred=genrm_preferred,
            genrm_confidence=genrm_confidence,
            genrm_tie_or_win=genrm_tie_or_win,
            genrm_error=genrm_error,
        )

    worker_count = _resolve_worker_count(
        int(num_workers),
        len(predictions),
        operation="lawstress.score_and_judge_predictions",
    )
    if worker_count <= 1 or len(predictions) <= 1:
        return [row for row in (_score_one(prediction) for prediction in predictions) if row is not None]

    slots: List[Optional[LawStressEvalResult]] = [None] * len(predictions)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_to_idx = {pool.submit(_score_one, prediction): idx for idx, prediction in enumerate(predictions)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            slots[idx] = future.result()
    return [row for row in slots if row is not None]


def _empty_metric_row() -> Dict[str, Any]:
    return {
        "n": 0,
        "mae": None,
        "pearson_r": None,
        "spearman_r": None,
        "within_5pct": None,
        "within_10pct": None,
        "same_side_of_neutral_pct": None,
        "c1_pass_rate": None,
        "c2_pass_rate": None,
        "c3_pass_rate": None,
        "genrm_tie_or_win_rate": None,
    }


def compute_metric_row(results: Sequence[LawStressEvalResult]) -> Dict[str, Any]:
    if not results:
        return _empty_metric_row()

    n = len(results)
    errors = [float(row.error_norm) for row in results]
    preds = [float(row.score_summary1_norm) for row in results]
    truth = [float(row.y_true_norm) for row in results]

    genrm_values = [row.genrm_tie_or_win for row in results if row.genrm_tie_or_win is not None]

    return {
        "n": n,
        "mae": float(sum(errors) / n),
        "pearson_r": _pearson(preds, truth),
        "spearman_r": _spearman(preds, truth),
        "within_5pct": float(sum(1 for error in errors if error <= 0.05) * 100.0 / n),
        "within_10pct": float(sum(1 for error in errors if error <= 0.10) * 100.0 / n),
        "same_side_of_neutral_pct": float(sum(1 for row in results if row.same_side_of_neutral) * 100.0 / n),
        "c1_pass_rate": float(sum(1 for row in results if row.c1_pass) * 100.0 / n),
        "c2_pass_rate": float(sum(1 for row in results if row.c2_pass) * 100.0 / n),
        "c3_pass_rate": float(sum(1 for row in results if row.c3_pass) * 100.0 / n),
        "genrm_tie_or_win_rate": (
            float(sum(1 for value in genrm_values if value) * 100.0 / len(genrm_values))
            if genrm_values
            else None
        ),
    }


def _group_results(
    results: Sequence[LawStressEvalResult],
    key_fn: Callable[[LawStressEvalResult], str],
) -> Dict[str, List[LawStressEvalResult]]:
    grouped: Dict[str, List[LawStressEvalResult]] = defaultdict(list)
    for row in results:
        grouped[key_fn(row)].append(row)
    return dict(grouped)


def compute_group_metrics(results: Sequence[LawStressEvalResult]) -> Dict[str, Any]:
    by_split = {
        split: compute_metric_row(rows)
        for split, rows in sorted(_group_results(results, lambda row: row.split).items())
    }
    by_difficulty = {
        difficulty: compute_metric_row(rows)
        for difficulty, rows in sorted(_group_results(results, lambda row: row.difficulty).items())
    }
    by_law = {
        law: compute_metric_row(rows)
        for law, rows in sorted(_group_results(results, lambda row: row.law_target).items())
    }
    by_family = {
        family: compute_metric_row(rows)
        for family, rows in sorted(_group_results(results, lambda row: row.family).items())
    }
    by_bin = {
        bin_name: compute_metric_row(rows)
        for bin_name, rows in sorted(_group_results(results, lambda row: row.bin_name).items())
    }
    by_split_difficulty = {
        key: compute_metric_row(rows)
        for key, rows in sorted(
            _group_results(results, lambda row: f"{row.split}:{row.difficulty}").items()
        )
    }

    return {
        "split": by_split,
        "difficulty": by_difficulty,
        "law_target": by_law,
        "family": by_family,
        "bin": by_bin,
        "split_difficulty": by_split_difficulty,
    }


def evaluate_success_criteria(
    *,
    overall: Dict[str, Any],
    group_metrics: Dict[str, Any],
    config: LawStressEvalConfig,
    baseline_overall: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hard = dict((group_metrics.get("difficulty") or {}).get("hard") or {})
    control = dict((group_metrics.get("difficulty") or {}).get("control") or {})

    abs_checks = {
        "hard_same_side": (hard.get("same_side_of_neutral_pct") is not None and hard["same_side_of_neutral_pct"] >= config.abs_hard_same_side_min),
        "hard_c1": (hard.get("c1_pass_rate") is not None and hard["c1_pass_rate"] >= config.abs_hard_c1_min),
        "hard_c2": (hard.get("c2_pass_rate") is not None and hard["c2_pass_rate"] >= config.abs_hard_c2_min),
        "hard_c3": (hard.get("c3_pass_rate") is not None and hard["c3_pass_rate"] >= config.abs_hard_c3_min),
        "control_same_side": (control.get("same_side_of_neutral_pct") is not None and control["same_side_of_neutral_pct"] >= config.abs_control_same_side_min),
        "control_c1": (control.get("c1_pass_rate") is not None and control["c1_pass_rate"] >= config.abs_control_c1_min),
        "control_c2": (control.get("c2_pass_rate") is not None and control["c2_pass_rate"] >= config.abs_control_c2_min),
        "control_c3": (control.get("c3_pass_rate") is not None and control["c3_pass_rate"] >= config.abs_control_c3_min),
    }

    relative: Dict[str, Any]
    if baseline_overall:
        current_same_side = overall.get("same_side_of_neutral_pct")
        current_c1 = overall.get("c1_pass_rate")
        current_c2 = overall.get("c2_pass_rate")
        current_c3 = overall.get("c3_pass_rate")
        current_mae = overall.get("mae")

        base_same_side = baseline_overall.get("same_side_of_neutral_pct")
        base_c1 = baseline_overall.get("c1_pass_rate")
        base_c2 = baseline_overall.get("c2_pass_rate")
        base_c3 = baseline_overall.get("c3_pass_rate")
        base_mae = baseline_overall.get("mae")

        relative_checks = {
            "same_side_gain": (
                current_same_side is not None
                and base_same_side is not None
                and (float(current_same_side) - float(base_same_side)) >= config.rel_same_side_gain_min
            ),
            "c1_gain": (
                current_c1 is not None
                and base_c1 is not None
                and (float(current_c1) - float(base_c1)) >= config.rel_c_gain_min
            ),
            "c2_gain": (
                current_c2 is not None
                and base_c2 is not None
                and (float(current_c2) - float(base_c2)) >= config.rel_c_gain_min
            ),
            "c3_gain": (
                current_c3 is not None
                and base_c3 is not None
                and (float(current_c3) - float(base_c3)) >= config.rel_c_gain_min
            ),
            "mae_gain": (
                current_mae is not None
                and base_mae is not None
                and (float(base_mae) - float(current_mae)) >= config.rel_mae_gain_min
            ),
        }
        relative = {
            "enabled": True,
            "checks": relative_checks,
            "pass": all(relative_checks.values()),
        }
    else:
        relative = {
            "enabled": False,
            "checks": {},
            "pass": True,
        }

    absolute_pass = all(abs_checks.values())
    overall_pass = bool(absolute_pass and relative["pass"])

    return {
        "absolute": {"checks": abs_checks, "pass": absolute_pass},
        "relative": relative,
        "overall_pass": overall_pass,
    }


def build_eval_metrics(
    results: Sequence[LawStressEvalResult],
    *,
    config: LawStressEvalConfig,
    baseline_overall: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    overall = compute_metric_row(results)
    group_metrics = compute_group_metrics(results)
    success = evaluate_success_criteria(
        overall=overall,
        group_metrics=group_metrics,
        config=config,
        baseline_overall=baseline_overall,
    )
    metrics = {
        "overall": overall,
        "success": success,
    }
    return metrics, group_metrics


def render_eval_report_markdown(
    metrics: Dict[str, Any],
    groups: Dict[str, Any],
) -> str:
    overall = metrics.get("overall") or {}
    success = metrics.get("success") or {}

    lines: List[str] = []
    lines.append("# LawStress Evaluation Report")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- n: {overall.get('n')}")
    lines.append(f"- mae: {overall.get('mae')}")
    lines.append(f"- pearson_r: {overall.get('pearson_r')}")
    lines.append(f"- spearman_r: {overall.get('spearman_r')}")
    lines.append(f"- within_5pct: {overall.get('within_5pct')}")
    lines.append(f"- within_10pct: {overall.get('within_10pct')}")
    lines.append(f"- same_side_of_neutral_pct: {overall.get('same_side_of_neutral_pct')}")
    lines.append(f"- c1_pass_rate: {overall.get('c1_pass_rate')}")
    lines.append(f"- c2_pass_rate: {overall.get('c2_pass_rate')}")
    lines.append(f"- c3_pass_rate: {overall.get('c3_pass_rate')}")
    lines.append(f"- genrm_tie_or_win_rate: {overall.get('genrm_tie_or_win_rate')}")
    lines.append("")

    lines.append("## Success Criteria")
    lines.append("")
    lines.append(f"- overall_pass: {success.get('overall_pass')}")
    absolute = (success.get("absolute") or {}).get("checks") or {}
    for key in sorted(absolute):
        lines.append(f"- abs/{key}: {absolute.get(key)}")
    relative = success.get("relative") or {}
    if relative.get("enabled"):
        for key, value in sorted((relative.get("checks") or {}).items()):
            lines.append(f"- rel/{key}: {value}")
    else:
        lines.append("- rel: disabled (no baseline provided)")
    lines.append("")

    lines.append("## By Difficulty")
    lines.append("")
    for difficulty, row in sorted((groups.get("difficulty") or {}).items()):
        lines.append(f"### {difficulty}")
        lines.append(f"- n: {row.get('n')}")
        lines.append(f"- mae: {row.get('mae')}")
        lines.append(f"- same_side_of_neutral_pct: {row.get('same_side_of_neutral_pct')}")
        lines.append(f"- c1_pass_rate: {row.get('c1_pass_rate')}")
        lines.append(f"- c2_pass_rate: {row.get('c2_pass_rate')}")
        lines.append(f"- c3_pass_rate: {row.get('c3_pass_rate')}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_predictions_jsonl(path: Path, predictions: Sequence[LawStressPrediction]) -> None:
    write_jsonl(path, (row.to_dict() for row in predictions))


def load_predictions_jsonl(path: Path) -> List[LawStressPrediction]:
    loaded: List[LawStressPrediction] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            loaded.append(LawStressPrediction.from_dict(json.loads(line)))
    return loaded


def write_eval_results_jsonl(path: Path, results: Sequence[LawStressEvalResult]) -> None:
    write_jsonl(path, (row.to_dict() for row in results))


__all__ = [
    "LawStressEvalConfig",
    "LawStressEvalResult",
    "LawStressPrediction",
    "RILE_RUBRIC",
    "build_eval_metrics",
    "build_predictions",
    "compute_group_metrics",
    "compute_metric_row",
    "evaluate_success_criteria",
    "load_predictions_jsonl",
    "render_eval_report_markdown",
    "score_and_judge_predictions",
    "strict_same_side",
    "write_eval_results_jsonl",
    "write_predictions_jsonl",
]
