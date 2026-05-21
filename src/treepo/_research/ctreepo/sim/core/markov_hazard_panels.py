"""Paper-facing Markov hazard panels.

This module centralizes the sticky Markov DGP variants used in paper-facing
experiments.  The existing ``hazard_topic`` generator remains the backend; a
panel is just a named, stratified mixture of generator conditions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MarkovHazardCondition:
    condition_id: str
    paper_label: str
    doc_tokens: int
    n_regimes: int
    expected_boundaries: float
    hazard_switch_prob: float
    mixture_weight: float = 1.0
    role: str = "structural"
    segment_density_band: str = ""
    segment_min: int = 0
    segment_max: int = 0
    aliases: Tuple[str, ...] = tuple()

    @property
    def vocab_size(self) -> int:
        return int(4 * int(self.n_regimes))

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MarkovHazardPanel:
    panel_id: str
    display_name: str
    conditions: Tuple[MarkovHazardCondition, ...]
    stratification: str = "condition_weighted"
    aliases: Tuple[str, ...] = tuple()

    def to_dict(self) -> Dict[str, object]:
        return {
            "panel_id": str(self.panel_id),
            "display_name": str(self.display_name),
            "stratification": str(self.stratification),
            "aliases": [str(value) for value in self.aliases],
            "conditions": [condition.to_dict() for condition in self.conditions],
            "ops_config_overrides": panel_to_ops_overrides(self),
        }


def sticky_markov_mean_segment_length(
    *,
    doc_tokens: int,
    min_segments: int,
    max_segments: int,
) -> int:
    target_segments = 0.5 * (float(min_segments) + float(max_segments))
    target_boundaries = max(1.0, target_segments - 1.0)
    mean_segment_length = float(max(1, int(doc_tokens) - 1)) / target_boundaries
    return max(1, int(round(mean_segment_length)))


def sticky_markov_switch_probability(
    *,
    doc_tokens: int,
    expected_boundaries: float,
) -> float:
    return float(max(0.0, expected_boundaries) / max(1.0, float(int(doc_tokens) - 1)))


STICKY_STRUCTURAL_V2_CELL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "cell_id": "r4_p031",
        "legacy_aliases": ("r4_seg4to6",),
        "regime_count": 4,
        "segment_density_band": "lower_switch",
        "segment_min": 4,
        "segment_max": 6,
        "expected_boundaries": 4.0,
    },
    {
        "cell_id": "r12_p031",
        "legacy_aliases": ("r12_seg4to6",),
        "regime_count": 12,
        "segment_density_band": "lower_switch",
        "segment_min": 4,
        "segment_max": 6,
        "expected_boundaries": 4.0,
    },
    {
        "cell_id": "r4_p079",
        "legacy_aliases": ("r4_seg10to12",),
        "regime_count": 4,
        "segment_density_band": "higher_switch",
        "segment_min": 10,
        "segment_max": 12,
        "expected_boundaries": 10.0,
    },
    {
        "cell_id": "r12_p079",
        "legacy_aliases": ("r12_seg10to12",),
        "regime_count": 12,
        "segment_density_band": "higher_switch",
        "segment_min": 10,
        "segment_max": 12,
        "expected_boundaries": 10.0,
    },
)
DEFAULT_STICKY_STRUCTURAL_V2_CELL_ID = "r12_p079"
STICKY_STRUCTURAL_V2_LEGACY_ALIAS_MAP: dict[str, str] = {
    str(alias).lower(): str(spec["cell_id"])
    for spec in STICKY_STRUCTURAL_V2_CELL_SPECS
    for alias in tuple(spec.get("legacy_aliases", ()) or ())
}


def canonicalize_structural_v2_cell_id(cell_id: str) -> str:
    normalized = str(cell_id or "").strip().lower()
    if not normalized:
        return ""
    return str(STICKY_STRUCTURAL_V2_LEGACY_ALIAS_MAP.get(normalized, normalized))


def sticky_recoverable_config_overrides(*, doc_tokens: int) -> Dict[str, Any]:
    expected_boundaries = 5.0 if int(doc_tokens) == 128 else 20.0
    min_segments = 2 if int(doc_tokens) == 128 else 8
    max_segments = 6 if int(doc_tokens) == 128 else 24
    switch_prob = sticky_markov_switch_probability(
        doc_tokens=int(doc_tokens),
        expected_boundaries=float(expected_boundaries),
    )
    mean_seg_len = sticky_markov_mean_segment_length(
        doc_tokens=int(doc_tokens),
        min_segments=int(min_segments),
        max_segments=int(max_segments),
    )
    return {
        "generator_profile": "hazard_topic",
        "n_regimes": 4,
        "vocab_size": 16,
        "min_tokens": int(doc_tokens),
        "max_tokens": int(doc_tokens),
        "min_segments": int(min_segments),
        "max_segments": int(max_segments),
        "min_seg_len": int(mean_seg_len),
        "max_seg_len": int(mean_seg_len),
        "hazard_switch_prob": float(switch_prob),
        "train_docs": 10240 if int(doc_tokens) == 2048 else 1024,
        "val_docs": 1024 if int(doc_tokens) == 2048 else 128,
        "test_docs": 1024 if int(doc_tokens) == 2048 else 256,
    }


def sticky_recoverable_config_overrides_t2048() -> Dict[str, Any]:
    return sticky_recoverable_config_overrides(doc_tokens=2048)


def sticky_structural_config_overrides(
    *,
    doc_tokens: int,
    n_regimes: int,
    min_segments: int,
    max_segments: int,
    hazard_switch_prob: float,
) -> Dict[str, Any]:
    mean_seg_len = sticky_markov_mean_segment_length(
        doc_tokens=int(doc_tokens),
        min_segments=int(min_segments),
        max_segments=int(max_segments),
    )
    return {
        "generator_profile": "hazard_topic",
        "n_regimes": int(n_regimes),
        "vocab_size": int(4 * int(n_regimes)),
        "min_tokens": int(doc_tokens),
        "max_tokens": int(doc_tokens),
        "min_segments": int(min_segments),
        "max_segments": int(max_segments),
        "min_seg_len": int(mean_seg_len),
        "max_seg_len": int(mean_seg_len),
        "hazard_switch_prob": float(hazard_switch_prob),
        "train_docs": 1024,
        "val_docs": 128,
        "test_docs": 256,
    }


def _structural_condition_from_spec(
    spec: Mapping[str, Any],
    *,
    doc_tokens: int,
    panel_prefix: str,
    boundary_scale: float = 1.0,
) -> MarkovHazardCondition:
    cell_id = str(spec["cell_id"])
    expected_boundaries = float(spec["expected_boundaries"]) * float(boundary_scale)
    segment_min = int(round(float(spec["segment_min"]) * float(boundary_scale)))
    segment_max = int(round(float(spec["segment_max"]) * float(boundary_scale)))
    switch_prob = sticky_markov_switch_probability(
        doc_tokens=int(doc_tokens),
        expected_boundaries=float(expected_boundaries),
    )
    grid_alias = f"structural_core_v2_t{int(doc_tokens)}::{cell_id}"
    aliases = (grid_alias,)
    if int(doc_tokens) == 128:
        aliases = (
            cell_id,
            f"structural_core_v2::{cell_id}",
            grid_alias,
            *tuple(str(value) for value in spec.get("legacy_aliases", ()) or ()),
        )
    return MarkovHazardCondition(
        condition_id=f"{panel_prefix}_{cell_id}",
        paper_label=(
            f"{int(spec['regime_count'])} regimes, "
            f"{str(spec['segment_density_band']).replace('_', ' ')}"
        ),
        doc_tokens=int(doc_tokens),
        n_regimes=int(spec["regime_count"]),
        expected_boundaries=float(expected_boundaries),
        hazard_switch_prob=float(switch_prob),
        mixture_weight=1.0,
        role="structural",
        segment_density_band=str(spec["segment_density_band"]),
        segment_min=int(segment_min),
        segment_max=int(segment_max),
        aliases=tuple(aliases),
    )


def _recoverable_condition(*, doc_tokens: int) -> MarkovHazardCondition:
    expected = 5.0 if int(doc_tokens) == 128 else 20.0
    min_segments = 2 if int(doc_tokens) == 128 else 8
    max_segments = 6 if int(doc_tokens) == 128 else 24
    condition_id = f"recoverable_v5_t{int(doc_tokens)}"
    return MarkovHazardCondition(
        condition_id=condition_id,
        paper_label=f"recoverable sticky, {int(doc_tokens)} tokens",
        doc_tokens=int(doc_tokens),
        n_regimes=4,
        expected_boundaries=float(expected),
        hazard_switch_prob=sticky_markov_switch_probability(
            doc_tokens=int(doc_tokens),
            expected_boundaries=float(expected),
        ),
        mixture_weight=1.0,
        role="recoverable",
        segment_density_band="recoverable",
        segment_min=int(min_segments),
        segment_max=int(max_segments),
        aliases=(
            condition_id,
            "recoverable_v5" if int(doc_tokens) == 128 else f"recoverable_sticky_t{int(doc_tokens)}",
        ),
    )


def _make_panel_t128() -> MarkovHazardPanel:
    return MarkovHazardPanel(
        panel_id="paper_hazard_panel_v1_t128",
        display_name="Paper Hazard Panel v1 (128-token)",
        conditions=tuple(
            _structural_condition_from_spec(
                spec,
                doc_tokens=128,
                panel_prefix="paper_v1_t128",
                boundary_scale=1.0,
            )
            for spec in STICKY_STRUCTURAL_V2_CELL_SPECS
        ),
        aliases=("paper_hazard_panel_v1", "hazard_panel_v1_t128"),
    )


def _make_panel_t2048() -> MarkovHazardPanel:
    scale = math.sqrt(2048.0 / 128.0)
    return MarkovHazardPanel(
        panel_id="paper_hazard_panel_v1_t2048",
        display_name="Paper Hazard Panel v1 (2048-token composition stress)",
        conditions=tuple(
            _structural_condition_from_spec(
                spec,
                doc_tokens=2048,
                panel_prefix="paper_v1_t2048",
                boundary_scale=float(scale),
            )
            for spec in STICKY_STRUCTURAL_V2_CELL_SPECS
        ),
        aliases=("hazard_panel_v1_t2048",),
    )


def _make_single_condition_panel(condition: MarkovHazardCondition) -> MarkovHazardPanel:
    return MarkovHazardPanel(
        panel_id=str(condition.condition_id),
        display_name=str(condition.paper_label),
        conditions=(condition,),
        stratification="single_condition",
        aliases=tuple(condition.aliases),
    )


def _base_panels() -> Tuple[MarkovHazardPanel, ...]:
    return (
        _make_panel_t128(),
        _make_panel_t2048(),
        _make_single_condition_panel(_recoverable_condition(doc_tokens=128)),
        _make_single_condition_panel(_recoverable_condition(doc_tokens=2048)),
    )


def _condition_lookup() -> Dict[str, MarkovHazardCondition]:
    lookup: Dict[str, MarkovHazardCondition] = {}
    for panel in _base_panels():
        for condition in panel.conditions:
            keys = (condition.condition_id, *condition.aliases)
            for key in keys:
                normalized = str(key or "").strip().lower()
                if normalized:
                    lookup[normalized] = condition
    return lookup


def _panel_lookup() -> Dict[str, MarkovHazardPanel]:
    lookup: Dict[str, MarkovHazardPanel] = {}
    for panel in _base_panels():
        keys = (panel.panel_id, *panel.aliases)
        for key in keys:
            normalized = str(key or "").strip().lower()
            if normalized:
                lookup[normalized] = panel
        for condition in panel.conditions:
            single = _make_single_condition_panel(condition)
            for key in (condition.condition_id, *condition.aliases):
                normalized = str(key or "").strip().lower()
                if normalized:
                    lookup.setdefault(normalized, single)
    return lookup


def resolve_markov_hazard_condition(condition_id: str) -> MarkovHazardCondition:
    key = str(condition_id or "").strip().lower()
    lookup = _condition_lookup()
    if key not in lookup:
        valid = ", ".join(sorted(lookup))
        raise ValueError(
            f"unknown Markov hazard condition: {condition_id!r}; expected one of {valid}"
        )
    return lookup[key]


def resolve_markov_hazard_panel(panel_id: str) -> MarkovHazardPanel:
    key = str(panel_id or "paper_hazard_panel_v1_t128").strip().lower()
    lookup = _panel_lookup()
    if key not in lookup:
        valid = ", ".join(sorted(lookup))
        raise ValueError(f"unknown Markov hazard panel: {panel_id!r}; expected one of {valid}")
    return lookup[key]


def condition_to_ops_overrides(
    condition: MarkovHazardCondition,
    *,
    train_docs: int,
    val_docs: int,
    test_docs: int,
) -> Dict[str, Any]:
    return sticky_structural_config_overrides(
        doc_tokens=int(condition.doc_tokens),
        n_regimes=int(condition.n_regimes),
        min_segments=int(condition.segment_min),
        max_segments=int(condition.segment_max),
        hazard_switch_prob=float(condition.hazard_switch_prob),
    ) | {
        "train_docs": int(train_docs),
        "val_docs": int(val_docs),
        "test_docs": int(test_docs),
    }


def panel_to_ops_overrides(panel: MarkovHazardPanel) -> Dict[str, Any]:
    conditions = tuple(panel.conditions)
    if not conditions:
        return {}
    return {
        "generator_profile": "hazard_topic",
        "n_regimes": int(max(condition.n_regimes for condition in conditions)),
        "vocab_size": int(max(condition.vocab_size for condition in conditions)),
        "min_tokens": int(min(condition.doc_tokens for condition in conditions)),
        "max_tokens": int(max(condition.doc_tokens for condition in conditions)),
        "hazard_panel_id": str(panel.panel_id),
    }


def _stratified_condition_counts(
    total: int,
    conditions: Sequence[MarkovHazardCondition],
) -> Tuple[int, ...]:
    n = int(max(0, total))
    if n <= 0 or not conditions:
        return tuple(0 for _ in conditions)
    weights = np.asarray(
        [max(0.0, float(condition.mixture_weight)) for condition in conditions],
        dtype=np.float64,
    )
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(conditions), dtype=np.float64)
    weights = weights / float(weights.sum())
    raw = weights * float(n)
    counts = np.floor(raw).astype(np.int64)
    if n >= len(conditions):
        counts = np.maximum(counts, 1)
    while int(counts.sum()) > n:
        idx = int(np.argmax(counts - raw))
        counts[idx] -= 1
    remainder = int(n - int(counts.sum()))
    if remainder > 0:
        order = np.argsort(-(raw - np.floor(raw)))
        for idx in order[:remainder]:
            counts[int(idx)] += 1
    return tuple(int(value) for value in counts.tolist())


def _split_seed(seed: int, split: str) -> int:
    offsets = {"train": 0, "val": 100_000, "test": 200_000}
    return int(seed) + int(offsets.get(str(split), 300_000))


def _build_split_docs(
    panel: MarkovHazardPanel,
    *,
    split: str,
    n_docs: int,
    seed: int,
) -> tuple[Tuple[Any, ...], Tuple[str, ...], Dict[str, int]]:
    from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (
        OPSCountConfig,
        _build_generator_transitions,
        _generate_ops_count_docs,
    )

    conditions = tuple(panel.conditions)
    counts = _stratified_condition_counts(int(n_docs), conditions)
    docs_by_condition: list[list[Any]] = []
    ids_by_condition: list[list[str]] = []
    count_by_condition: Dict[str, int] = {}
    for idx, (condition, count) in enumerate(zip(conditions, counts)):
        count_by_condition[str(condition.condition_id)] = int(count)
        condition_docs: list[Any] = []
        condition_ids: list[str] = []
        if int(count) <= 0:
            docs_by_condition.append(condition_docs)
            ids_by_condition.append(condition_ids)
            continue
        overrides = condition_to_ops_overrides(
            condition,
            train_docs=int(count),
            val_docs=0,
            test_docs=int(count),
        )
        cfg = OPSCountConfig(**{key: value for key, value in overrides.items() if key != "hazard_panel_id"})
        condition_seed = _split_seed(int(seed), str(split)) + 7919 * int(idx + 1)
        transitions = _build_generator_transitions(cfg, seed=int(condition_seed))
        generated = _generate_ops_count_docs(
            cfg,
            n_docs=int(count),
            seed=int(condition_seed),
            transitions=transitions,
        )
        condition_docs = list(generated)
        condition_ids = [str(condition.condition_id)] * int(count)
        if condition_docs:
            rng = np.random.default_rng(
                _split_seed(int(seed), str(split)) + 17 + 7919 * int(idx + 1)
            )
            order = rng.permutation(len(condition_docs)).tolist()
            condition_docs = [condition_docs[int(i)] for i in order]
            condition_ids = [condition_ids[int(i)] for i in order]
        docs_by_condition.append(condition_docs)
        ids_by_condition.append(condition_ids)

    docs: list[Any] = []
    condition_ids: list[str] = []
    max_count = max((len(items) for items in docs_by_condition), default=0)
    for row_idx in range(max_count):
        for condition_docs, condition_id_values in zip(docs_by_condition, ids_by_condition):
            if row_idx >= len(condition_docs):
                continue
            docs.append(condition_docs[row_idx])
            condition_ids.append(condition_id_values[row_idx])
    return tuple(docs), tuple(condition_ids), count_by_condition


def build_markov_hazard_panel_data_bundle(
    panel_id: str,
    *,
    train_docs: int,
    val_docs: int,
    test_docs: int,
    seed: int,
):
    from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (
        MarkovOPSDataBundle,
        _markov_corpus_signature,
    )

    panel = resolve_markov_hazard_panel(panel_id)
    train, train_ids, train_counts = _build_split_docs(
        panel,
        split="train",
        n_docs=int(train_docs),
        seed=int(seed),
    )
    val, val_ids, val_counts = _build_split_docs(
        panel,
        split="val",
        n_docs=int(val_docs),
        seed=int(seed),
    )
    test, test_ids, test_counts = _build_split_docs(
        panel,
        split="test",
        n_docs=int(test_docs),
        seed=int(seed),
    )
    return MarkovOPSDataBundle(
        train_docs=tuple(train),
        val_docs=tuple(val),
        test_docs=tuple(test),
        train_corpus_signature=_markov_corpus_signature(tuple(train)),
        val_corpus_signature=_markov_corpus_signature(tuple(val)),
        test_corpus_signature=_markov_corpus_signature(tuple(test)),
        metadata={
            **panel.to_dict(),
            "hazard_panel_id": str(panel.panel_id),
            "seed": int(seed),
            "condition_ids": {
                "train": [str(value) for value in train_ids],
                "val": [str(value) for value in val_ids],
                "test": [str(value) for value in test_ids],
            },
            "condition_counts": {
                "train": dict(train_counts),
                "val": dict(val_counts),
                "test": dict(test_counts),
            },
        },
    )


__all__ = [
    "DEFAULT_STICKY_STRUCTURAL_V2_CELL_ID",
    "MarkovHazardCondition",
    "MarkovHazardPanel",
    "STICKY_STRUCTURAL_V2_CELL_SPECS",
    "STICKY_STRUCTURAL_V2_LEGACY_ALIAS_MAP",
    "build_markov_hazard_panel_data_bundle",
    "canonicalize_structural_v2_cell_id",
    "condition_to_ops_overrides",
    "panel_to_ops_overrides",
    "resolve_markov_hazard_condition",
    "resolve_markov_hazard_panel",
    "sticky_markov_mean_segment_length",
    "sticky_markov_switch_probability",
    "sticky_recoverable_config_overrides",
    "sticky_recoverable_config_overrides_t2048",
    "sticky_structural_config_overrides",
]
