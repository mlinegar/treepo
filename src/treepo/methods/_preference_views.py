"""Optimizer-facing preference projection helpers.

Given a unit's candidate rows, these functions build the supervised / DPO /
reward / GRPO record shapes: ordering, pairing, preferred-side resolution, rank
assignment, and per-record metadata. Pure projection logic over already
normalized rows; the ``PreferenceDataset`` methods call these to render views.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from typing import Any, Mapping, Sequence

from treepo.methods._preference_normalize import _bool, _json_text, _sample_weight


def _candidate_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("candidate_id") or ""),
        "candidate_id": str(row.get("candidate_id") or ""),
        "value": row.get("value"),
        "score": row.get("score"),
        "rank": row.get("rank"),
        "preferred": bool(row.get("preferred", False)),
        "metadata": dict(row.get("metadata") or {}),
    }


def _ordered_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(candidate) for candidate in candidates]
    if any(_bool(candidate.get("preferred")) for candidate in rows):
        return sorted(rows, key=lambda candidate: (0 if _bool(candidate.get("preferred")) else 1, str(candidate.get("candidate_id"))))
    if any(candidate.get("rank") is not None for candidate in rows):
        return sorted(rows, key=lambda candidate: (int(candidate.get("rank") or 10**9), str(candidate.get("candidate_id"))))
    if any(candidate.get("score") is not None for candidate in rows):
        return sorted(rows, key=lambda candidate: (-_score_sort_value(candidate), str(candidate.get("candidate_id"))))
    return rows


def _top_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in candidates]
    if not rows:
        return []
    preferred = [row for row in rows if _bool(row.get("preferred"))]
    if preferred:
        return preferred
    ranked = [row for row in rows if row.get("rank") is not None]
    if ranked:
        best_rank = min(int(row["rank"]) for row in ranked)
        return [row for row in ranked if int(row["rank"]) == best_rank]
    scored = [row for row in rows if row.get("score") is not None]
    if scored:
        best_score = max(float(row["score"]) for row in scored)
        return [row for row in scored if float(row["score"]) == best_score]
    return rows[:1]


def _pair_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows = _ordered_candidates(candidates)
    if len(rows) < 2:
        return []
    if any(_bool(row.get("preferred")) for row in rows):
        winners = [row for row in rows if _bool(row.get("preferred"))]
        losers = [row for row in rows if not _bool(row.get("preferred"))]
        return [(winner, loser) for winner in winners for loser in losers]
    return [(left, right) for left, right in zip(rows, rows[1:])]


def _preferred_side(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    if _bool(left.get("preferred")) and not _bool(right.get("preferred")):
        return "A"
    if _bool(right.get("preferred")) and not _bool(left.get("preferred")):
        return "B"
    if left.get("rank") is not None and right.get("rank") is not None:
        l_rank = int(left["rank"])
        r_rank = int(right["rank"])
        if l_rank == r_rank:
            return "tie"
        return "A" if l_rank < r_rank else "B"
    if left.get("score") is not None and right.get("score") is not None:
        l_score = float(left["score"])
        r_score = float(right["score"])
        if l_score == r_score:
            return "tie"
        return "A" if l_score > r_score else "B"
    return "tie"


def _grpo_ranks(candidates: Sequence[Mapping[str, Any]]) -> list[int]:
    rows = list(candidates)
    if any(_bool(row.get("preferred")) for row in rows):
        return [1 if _bool(row.get("preferred")) else 2 for row in rows]
    if any(row.get("rank") is not None for row in rows):
        return [
            int(row["rank"]) if row.get("rank") is not None else idx + 1
            for idx, row in enumerate(rows)
        ]
    if any(row.get("score") is not None for row in rows):
        ranks: list[int] = []
        previous_score: float | None = None
        current_rank = 1
        for idx, row in enumerate(rows):
            score = None if row.get("score") is None else float(row["score"])
            if idx == 0:
                current_rank = 1
            elif score is None or previous_score is None or score != previous_score:
                current_rank = idx + 1
            ranks.append(current_rank)
            previous_score = score
        return ranks
    return [1 for _ in rows]


def _reward_scores(chosen: Mapping[str, Any], rejected: Mapping[str, Any]) -> tuple[float, float]:
    if chosen.get("score") is not None and rejected.get("score") is not None:
        return float(chosen["score"]), float(rejected["score"])
    return 1.0, 0.0


def _score_sort_value(candidate: Mapping[str, Any]) -> float:
    score = candidate.get("score")
    return float(score) if score is not None else float("-inf")


def _export_metadata(
    unit: Mapping[str, Any],
    *candidates: Mapping[str, Any],
    format_name: str,
) -> dict[str, Any]:
    metadata = dict(unit.get("metadata") or {})
    metadata.update(
        {
            "unit_id": unit.get("unit_id"),
            "unit_type": unit.get("unit_type"),
            "target": unit.get("target"),
            "tree_id": unit.get("tree_id"),
            "doc_id": unit.get("doc_id"),
            "node_id": unit.get("node_id"),
            "law_type": metadata.get("law_type", metadata.get("law_kind", "preference")),
            "format": str(format_name),
            "sample_weight": _sample_weight(unit.get("weight", 1.0), unit.get("propensity", 1.0)),
        }
    )
    if candidates:
        metadata["candidate_ids"] = [candidate.get("candidate_id") for candidate in candidates]
    return metadata


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("value", "")
    if isinstance(value, str):
        return value
    return _json_text(value)


def _context_text(context: Any) -> str:
    if isinstance(context, str):
        return context
    if isinstance(context, MappingABC) and context.get("prompt"):
        return str(context["prompt"])
    return _json_text(context)


__all__ = [
    "_candidate_record",
    "_candidate_text",
    "_context_text",
    "_export_metadata",
    "_grpo_ranks",
    "_ordered_candidates",
    "_pair_candidates",
    "_preferred_side",
    "_reward_scores",
    "_score_sort_value",
    "_top_candidates",
]
