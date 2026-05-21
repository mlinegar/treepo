"""Helpers for teacher-generated summary traces anchored to real manifestos."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from treepo._research.tasks.manifesto.data_loader import ManifestoDataset, ManifestoSample
from treepo._research.tasks.manifesto.lawstress_generator import get_rile_bin


def strict_same_side_raw(pred_raw: float, true_raw: float, neutral_raw: float = 0.0) -> bool:
    pred_delta = float(pred_raw) - float(neutral_raw)
    if abs(pred_delta) <= 1e-9:
        return False
    true_delta = float(true_raw) - float(neutral_raw)
    return bool(pred_delta * true_delta > 0.0)


_strict_same_side_raw = strict_same_side_raw


@dataclass(frozen=True)
class SeedManifestoDoc:
    """Real manifesto sample selected as anchor for synthetic expansion."""

    manifesto_id: str
    party_abbrev: str
    country_name: str
    year: int
    source_rile_raw: float
    source_bin_name: str
    source_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifesto_id": self.manifesto_id,
            "party_abbrev": self.party_abbrev,
            "country_name": self.country_name,
            "year": self.year,
            "source_rile_raw": self.source_rile_raw,
            "source_bin_name": self.source_bin_name,
            "source_text": self.source_text,
        }


@dataclass
class TeacherTraceRecord:
    """Accepted record containing synthetic expansion, summaries, and trace artifacts."""

    example_id: str
    split: str
    source_manifesto_id: str
    source_party_abbrev: str
    source_country_name: str
    source_year: int
    source_rile_raw: float
    source_bin_name: str
    source_text: str

    expanded_text: str
    expanded_score_raw: float
    expanded_delta_raw: float

    summary1: str
    summary1_score_raw: float
    summary1_delta_raw: float

    summary2: str
    summary2_score_raw: float
    summary2_delta_raw: float
    summary2_vs_summary1_delta_raw: float
    same_side_summary1: bool
    same_side_summary2: bool

    trace_critical_points: List[str]
    trace_entities: List[str]
    trace_qualifiers: List[str]
    trace_invariants: List[str]
    trace_notes: str

    attempts_used: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "split": self.split,
            "source_manifesto_id": self.source_manifesto_id,
            "source_party_abbrev": self.source_party_abbrev,
            "source_country_name": self.source_country_name,
            "source_year": int(self.source_year),
            "source_rile_raw": float(self.source_rile_raw),
            "source_bin_name": self.source_bin_name,
            "source_text": self.source_text,
            "expanded_text": self.expanded_text,
            "expanded_score_raw": float(self.expanded_score_raw),
            "expanded_delta_raw": float(self.expanded_delta_raw),
            "summary1": self.summary1,
            "summary1_score_raw": float(self.summary1_score_raw),
            "summary1_delta_raw": float(self.summary1_delta_raw),
            "summary2": self.summary2,
            "summary2_score_raw": float(self.summary2_score_raw),
            "summary2_delta_raw": float(self.summary2_delta_raw),
            "summary2_vs_summary1_delta_raw": float(self.summary2_vs_summary1_delta_raw),
            "same_side_summary1": bool(self.same_side_summary1),
            "same_side_summary2": bool(self.same_side_summary2),
            "trace_critical_points": list(self.trace_critical_points),
            "trace_entities": list(self.trace_entities),
            "trace_qualifiers": list(self.trace_qualifiers),
            "trace_invariants": list(self.trace_invariants),
            "trace_notes": self.trace_notes,
            "attempts_used": int(self.attempts_used),
        }

    def to_benchmark_doc(self) -> Dict[str, Any]:
        return {
            "id": self.example_id,
            "doc_id": self.example_id,
            "text": self.expanded_text,
            "reference_score": float(self.source_rile_raw),
            "score": float(self.source_rile_raw),
            "metadata": {
                "split": self.split,
                "source_manifesto_id": self.source_manifesto_id,
                "source_party_abbrev": self.source_party_abbrev,
                "source_country_name": self.source_country_name,
                "source_year": int(self.source_year),
                "source_bin_name": self.source_bin_name,
                "expanded_score_raw": float(self.expanded_score_raw),
                "expanded_delta_raw": float(self.expanded_delta_raw),
                "summary1_score_raw": float(self.summary1_score_raw),
                "summary2_score_raw": float(self.summary2_score_raw),
                "summary2_vs_summary1_delta_raw": float(self.summary2_vs_summary1_delta_raw),
                "same_side_summary1": bool(self.same_side_summary1),
                "same_side_summary2": bool(self.same_side_summary2),
            },
        }

    def to_summary_pair_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": f"{self.example_id}_hop1",
                "example_id": self.example_id,
                "split": self.split,
                "hop": 1,
                "input_text": self.expanded_text,
                "target_summary": self.summary1,
                "source_rile_raw": float(self.source_rile_raw),
                "target_score_raw": float(self.summary1_score_raw),
            },
            {
                "id": f"{self.example_id}_hop2",
                "example_id": self.example_id,
                "split": self.split,
                "hop": 2,
                "input_text": self.summary1,
                "target_summary": self.summary2,
                "source_rile_raw": float(self.source_rile_raw),
                "target_score_raw": float(self.summary2_score_raw),
            },
        ]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TeacherTraceRecord":
        return cls(
            example_id=str(payload.get("example_id", "")),
            split=str(payload.get("split", "")),
            source_manifesto_id=str(payload.get("source_manifesto_id", "")),
            source_party_abbrev=str(payload.get("source_party_abbrev", "")),
            source_country_name=str(payload.get("source_country_name", "")),
            source_year=int(payload.get("source_year", 0)),
            source_rile_raw=float(payload.get("source_rile_raw", 0.0)),
            source_bin_name=str(payload.get("source_bin_name", "")),
            source_text=str(payload.get("source_text", "")),
            expanded_text=str(payload.get("expanded_text", "")),
            expanded_score_raw=float(payload.get("expanded_score_raw", 0.0)),
            expanded_delta_raw=float(payload.get("expanded_delta_raw", 0.0)),
            summary1=str(payload.get("summary1", "")),
            summary1_score_raw=float(payload.get("summary1_score_raw", 0.0)),
            summary1_delta_raw=float(payload.get("summary1_delta_raw", 0.0)),
            summary2=str(payload.get("summary2", "")),
            summary2_score_raw=float(payload.get("summary2_score_raw", 0.0)),
            summary2_delta_raw=float(payload.get("summary2_delta_raw", 0.0)),
            summary2_vs_summary1_delta_raw=float(payload.get("summary2_vs_summary1_delta_raw", 0.0)),
            same_side_summary1=bool(payload.get("same_side_summary1", False)),
            same_side_summary2=bool(payload.get("same_side_summary2", False)),
            trace_critical_points=[str(v) for v in (payload.get("trace_critical_points") or [])],
            trace_entities=[str(v) for v in (payload.get("trace_entities") or [])],
            trace_qualifiers=[str(v) for v in (payload.get("trace_qualifiers") or [])],
            trace_invariants=[str(v) for v in (payload.get("trace_invariants") or [])],
            trace_notes=str(payload.get("trace_notes", "")),
            attempts_used=int(payload.get("attempts_used", 1)),
        )


def clip_source_text(text: str, max_chars: int) -> str:
    rendered = str(text or "").strip()
    if max_chars <= 0 or len(rendered) <= max_chars:
        return rendered
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return (
        f"{rendered[:head].rstrip()}\n\n"
        "[... SOURCE TEXT TRUNCATED FOR PROMPT LENGTH ...]\n\n"
        f"{rendered[-tail:].lstrip()}"
    )


def select_seed_manifestos(
    dataset: ManifestoDataset,
    *,
    n_docs: int,
    seed: int,
    min_source_chars: int = 1200,
    manifesto_ids: Optional[Sequence[str]] = None,
    balanced_bins: bool = True,
) -> List[SeedManifestoDoc]:
    """Select seed manifestos, optionally balanced by RILE bins."""

    if n_docs <= 0:
        return []

    rng = random.Random(int(seed))

    candidate_ids: List[str]
    if manifesto_ids:
        candidate_ids = [str(value) for value in manifesto_ids]
    else:
        candidate_ids = list(dataset.get_all_ids())
    rng.shuffle(candidate_ids)

    candidates: List[SeedManifestoDoc] = []
    for manifesto_id in candidate_ids:
        sample = dataset.get_sample(manifesto_id)
        if sample is None:
            continue
        text = str(sample.text or "").strip()
        if len(text) < int(min_source_chars):
            continue
        candidates.append(manifesto_sample_to_seed(sample))

    if not balanced_bins:
        return candidates[:n_docs]

    bin_order = sorted({seed_doc.source_bin_name for seed_doc in candidates})
    if not bin_order:
        return []

    by_bin: Dict[str, List[SeedManifestoDoc]] = {name: [] for name in bin_order}
    for seed_doc in candidates:
        by_bin.setdefault(seed_doc.source_bin_name, []).append(seed_doc)
    for rows in by_bin.values():
        rng.shuffle(rows)

    selected: List[SeedManifestoDoc] = []
    while len(selected) < n_docs:
        added = 0
        for bin_name in bin_order:
            rows = by_bin.get(bin_name) or []
            if not rows:
                continue
            selected.append(rows.pop())
            added += 1
            if len(selected) >= n_docs:
                break
        if added == 0:
            break

    return selected[:n_docs]


def build_split_labels(
    *,
    total_docs: int,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> List[str]:
    """Build deterministic split labels with exact requested counts."""

    total_docs = int(total_docs)
    if total_docs <= 0:
        return []

    train_size = max(0, int(train_size))
    val_size = max(0, int(val_size))
    test_size = max(0, int(test_size))

    requested_total = train_size + val_size + test_size
    if requested_total == 0:
        raise ValueError("At least one of train_size/val_size/test_size must be positive")

    if requested_total != total_docs:
        raise ValueError(
            f"Split counts must sum to total_docs ({train_size}+{val_size}+{test_size} != {total_docs})"
        )

    labels = (["train"] * train_size) + (["val"] * val_size) + (["test"] * test_size)
    rng = random.Random(int(seed))
    rng.shuffle(labels)
    return labels


def manifesto_sample_to_seed(sample: ManifestoSample) -> SeedManifestoDoc:
    return SeedManifestoDoc(
        manifesto_id=sample.manifesto_id,
        party_abbrev=str(sample.party_abbrev or ""),
        country_name=str(sample.country_name or ""),
        year=int(sample.year),
        source_rile_raw=float(sample.rile),
        source_bin_name=get_rile_bin(float(sample.rile)).name,
        source_text=str(sample.text or ""),
    )


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_teacher_trace_records_jsonl(path: Path, records: Sequence[TeacherTraceRecord]) -> None:
    write_jsonl(path, (record.to_dict() for record in records))


def load_teacher_trace_records_jsonl(path: Path) -> List[TeacherTraceRecord]:
    loaded: List[TeacherTraceRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            loaded.append(TeacherTraceRecord.from_dict(json.loads(line)))
    return loaded


def build_benchmark_docs(records: Sequence[TeacherTraceRecord]) -> List[Dict[str, Any]]:
    return [record.to_benchmark_doc() for record in records]


def build_summary_pair_rows(records: Sequence[TeacherTraceRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        rows.extend(record.to_summary_pair_rows())
    return rows


def summarize_teacher_trace_records(records: Sequence[TeacherTraceRecord]) -> Dict[str, Any]:
    if not records:
        return {"n": 0}

    def _avg(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    expanded_delta = [abs(row.expanded_delta_raw) for row in records]
    summary1_delta = [abs(row.summary1_delta_raw) for row in records]
    summary2_delta = [abs(row.summary2_delta_raw) for row in records]
    hop_delta = [abs(row.summary2_vs_summary1_delta_raw) for row in records]

    split_counts: Dict[str, int] = {}
    for row in records:
        split_counts[row.split] = split_counts.get(row.split, 0) + 1

    same_side_s1 = sum(1 for row in records if row.same_side_summary1)
    same_side_s2 = sum(1 for row in records if row.same_side_summary2)
    c1_tol_raw = 10.0
    c2_drift_tol_raw = 6.0
    c1_pass = sum(1 for row in records if abs(float(row.summary1_delta_raw)) <= c1_tol_raw)
    c2_pass = sum(
        1
        for row in records
        if abs(float(row.summary2_vs_summary1_delta_raw)) <= c2_drift_tol_raw
        and bool(row.same_side_summary1)
        and bool(row.same_side_summary2)
    )

    return {
        "n": len(records),
        "split_counts": split_counts,
        "expanded_mae_raw": _avg(expanded_delta),
        "summary1_mae_raw": _avg(summary1_delta),
        "summary2_mae_raw": _avg(summary2_delta),
        "summary2_vs_summary1_mae_raw": _avg(hop_delta),
        "same_side_summary1_pct": 100.0 * same_side_s1 / max(1, len(records)),
        "same_side_summary2_pct": 100.0 * same_side_s2 / max(1, len(records)),
        "c1_tol_raw": c1_tol_raw,
        "c2_drift_tol_raw": c2_drift_tol_raw,
        "c1_pass_pct": 100.0 * c1_pass / max(1, len(records)),
        "c2_pass_pct": 100.0 * c2_pass / max(1, len(records)),
    }


__all__ = [
    "SeedManifestoDoc",
    "TeacherTraceRecord",
    "build_benchmark_docs",
    "build_split_labels",
    "build_summary_pair_rows",
    "clip_source_text",
    "load_teacher_trace_records_jsonl",
    "select_seed_manifestos",
    "summarize_teacher_trace_records",
    "strict_same_side_raw",
    "write_jsonl",
    "write_teacher_trace_records_jsonl",
]
