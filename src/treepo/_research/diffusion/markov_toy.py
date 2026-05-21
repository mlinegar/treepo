"""Theorem-valid toy Markov track for fixed-binary diffusion prototypes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar


State = str
T = TypeVar("T")


@dataclass(frozen=True)
class MarkovToySketch:
    """Exact mergeable sketch for changepoint-count Markov paths."""

    changepoints: int
    start_state: Optional[State]
    end_state: Optional[State]
    length: int


def changepoint_count(states: Sequence[State]) -> int:
    """Count state transitions in a realized Markov path."""
    return sum(1 for left, right in zip(states, states[1:]) if left != right)


def encode_markov_path(states: Sequence[State]) -> MarkovToySketch:
    """Exact leaf encoder for a realized Markov path segment."""
    items = list(states)
    if not items:
        return MarkovToySketch(changepoints=0, start_state=None, end_state=None, length=0)
    return MarkovToySketch(
        changepoints=changepoint_count(items),
        start_state=items[0],
        end_state=items[-1],
        length=len(items),
    )


def merge_markov_sketch(left: MarkovToySketch, right: MarkovToySketch) -> MarkovToySketch:
    """Exact merge for the changepoint-count theorem state."""
    if left.length == 0:
        return right
    if right.length == 0:
        return left
    boundary = 0 if left.end_state == right.start_state else 1
    return MarkovToySketch(
        changepoints=left.changepoints + right.changepoints + boundary,
        start_state=left.start_state,
        end_state=right.end_state,
        length=left.length + right.length,
    )


def count_only_feature(states: Sequence[State]) -> int:
    """Under-supported baseline that keeps only the raw changepoint count."""
    return changepoint_count(states)


def chunk_states(states: Sequence[State], chunk_size: int) -> List[List[State]]:
    """Split a path into fixed contiguous chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    values = list(states)
    return [values[index:index + chunk_size] for index in range(0, len(values), chunk_size)]


def _reduce_fixed_binary(values: Sequence[T], merge_fn) -> Tuple[T, List[Dict[str, Any]]]:
    if not values:
        raise ValueError("Fixed-binary reduction requires at least one value.")
    current = list(values)
    levels: List[Dict[str, Any]] = []
    level_index = 0
    while len(current) > 1:
        next_level: List[T] = []
        merged_indices: List[List[int]] = []
        carried_indices: List[int] = []
        index = 0
        while index + 1 < len(current):
            next_level.append(merge_fn(current[index], current[index + 1]))
            merged_indices.append([index, index + 1])
            index += 2
        if index < len(current):
            next_level.append(current[index])
            carried_indices.append(index)
        levels.append(
            {
                "level_index": level_index,
                "values": list(next_level),
                "merged_indices": merged_indices,
                "carried_indices": carried_indices,
            }
        )
        current = next_level
        level_index += 1
    return current[0], levels


def _serialize_levels(levels: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for level in levels:
        values = level["values"]
        serialized.append(
            {
                "level_index": int(level["level_index"]),
                "values": [
                    asdict(value) if isinstance(value, MarkovToySketch) else value
                    for value in values
                ],
                "merged_indices": [list(indices) for indices in level["merged_indices"]],
                "carried_indices": list(level["carried_indices"]),
            }
        )
    return serialized


def run_markov_toy_experiment(
    states: Sequence[State],
    *,
    chunk_size: int = 1,
    rounds: int = 1,
    eps_leaf: float = 0.0,
    eps_merge: float = 0.0,
    eps_idemp: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate exact and count-only fixed-binary schedules on a toy Markov path."""
    items = list(states)
    if not items:
        raise ValueError("run_markov_toy_experiment requires at least one state.")

    chunks = chunk_states(items, chunk_size)
    exact_leaf_states = [encode_markov_path(chunk) for chunk in chunks]
    exact_root_state, exact_levels = _reduce_fixed_binary(exact_leaf_states, merge_markov_sketch)

    full_path_state = encode_markov_path(items)
    count_leaf_states = [count_only_feature(chunk) for chunk in chunks]
    count_root_state, count_levels = _reduce_fixed_binary(count_leaf_states, lambda left, right: left + right)

    theorem_budget = round(
        float(eps_leaf) + float(eps_merge) + max(int(rounds) - 1, 0) * float(eps_idemp),
        12,
    )
    count_only_root_matches = count_root_state == full_path_state.changepoints

    return {
        "path": items,
        "chunk_size": int(chunk_size),
        "rounds": int(rounds),
        "chunks": chunks,
        "exact_leaf_states": [asdict(value) for value in exact_leaf_states],
        "exact_root_state": asdict(exact_root_state),
        "full_path_state": asdict(full_path_state),
        "exact_state_matches_full_path": exact_root_state == full_path_state,
        "count_only_leaf_states": count_leaf_states,
        "count_only_root_state": count_root_state,
        "count_only_full_path_value": full_path_state.changepoints,
        "count_only_matches_full_path": count_only_root_matches,
        "count_only_gap": int(count_root_state - full_path_state.changepoints),
        "theorem_budget": theorem_budget,
        "budget_formula": {
            "eps_leaf": float(eps_leaf),
            "eps_merge": float(eps_merge),
            "eps_idemp": float(eps_idemp),
            "value": theorem_budget,
            "shape": "eps_leaf + eps_merge + (rounds - 1) * eps_idemp",
        },
        "exact_schedule": _serialize_levels(exact_levels),
        "count_only_schedule": _serialize_levels(count_levels),
    }
