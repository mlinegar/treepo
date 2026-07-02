"""Broad classical mergeable-sketch comparison.

This module is intentionally CPU-light. It validates that official Apache
DataSketches implementations can be routed through the same TreePO
leaf/merge/query surface.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from treepo.common import VALID_SCHEDULES
from treepo.bench.sketches import (
    make_count_min_adapter,
    make_cpc_adapter,
    make_frequent_strings_adapter,
    make_hll_adapter,
    make_kll_floats_adapter,
    make_quantiles_floats_adapter,
    make_req_floats_adapter,
    make_tdigest_double_adapter,
    make_theta_adapter,
    make_tuple_accumulator_adapter,
    make_varopt_strings_adapter,
)
from treepo.bench.sketches.adapters.datasketches_cardinality import (
    theta_a_not_b_estimate,
    theta_intersection_estimate,
    theta_union_estimate,
)
from treepo.bench.sketches.tree_reducer import fold_states


@dataclass(frozen=True)
class ClassicalSketchComparisonConfig:
    seed: int = 0
    capacity_label: str = "single"
    n_docs: int = 32
    min_tokens: int = 128
    max_tokens: int = 512
    universe_size: int = 4096
    leaf_unit_count: int = 64
    distinct_lg_k: int = 10
    theta_lg_k: int = 12
    cms_num_hashes: int = 5
    cms_num_buckets: int = 256
    frequent_lg_max_map_size: int = 8
    kll_k: int = 128
    quantiles_k: int = 128
    req_k: int = 12
    tdigest_k: int = 100
    tuple_lg_k: int = 12
    varopt_k: int = 64
    include_families: Tuple[str, ...] = ("distinct", "frequency", "quantile", "set", "sampling")
    quantile_queries: Tuple[float, ...] = (0.5, 0.95)


@dataclass(frozen=True)
class ClassicalSketchComparisonSummary:
    config: Dict[str, object]
    rows: Tuple[Dict[str, object], ...]

    def to_dict(self) -> Dict[str, object]:
        return {"config": dict(self.config), "rows": [dict(row) for row in self.rows]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _chunks(xs: Sequence, leaf_unit_count: int) -> List[List]:
    size = int(max(1, leaf_unit_count))
    return [list(xs[i : i + size]) for i in range(0, len(xs), size)] or [[]]


def _leaves(config: ClassicalSketchComparisonConfig, xs: Sequence) -> List[List]:
    return _chunks(xs, int(config.leaf_unit_count))


def _leaf_count_stats(config: ClassicalSketchComparisonConfig, docs: Sequence[Sequence]) -> Dict[str, object]:
    counts = [len(_leaves(config, doc)) for doc in docs]
    tokens_per_leaf = [
        float(len(doc)) / float(max(1, len(_leaves(config, doc))))
        for doc in docs
    ]
    if not counts:
        counts = [1]
    if not tokens_per_leaf:
        tokens_per_leaf = [float(config.leaf_unit_count)]
    return {
        "leaf_count_min": int(min(counts)),
        "leaf_count_mean": float(np.mean(np.asarray(counts, dtype=np.float64))),
        "leaf_count_max": int(max(counts)),
        "tokens_per_leaf_min": float(np.nanmin(np.asarray(tokens_per_leaf, dtype=np.float64))),
        "tokens_per_leaf_mean": float(np.nanmean(np.asarray(tokens_per_leaf, dtype=np.float64))),
        "tokens_per_leaf_max": float(np.nanmax(np.asarray(tokens_per_leaf, dtype=np.float64))),
    }


def _row_axes(
    config: ClassicalSketchComparisonConfig,
    docs: Sequence[Sequence],
) -> Dict[str, object]:
    leaf_stats = _leaf_count_stats(config, docs)
    return {
        "seed": int(config.seed),
        "capacity_label": str(config.capacity_label),
        "leaf_unit_count": int(config.leaf_unit_count),
        **leaf_stats,
        "distinct_lg_k": int(config.distinct_lg_k),
        "theta_lg_k": int(config.theta_lg_k),
        "cms_num_hashes": int(config.cms_num_hashes),
        "cms_num_buckets": int(config.cms_num_buckets),
        "frequent_lg_max_map_size": int(config.frequent_lg_max_map_size),
        "kll_k": int(config.kll_k),
        "quantiles_k": int(config.quantiles_k),
        "req_k": int(config.req_k),
        "tdigest_k": int(config.tdigest_k),
        "tuple_lg_k": int(config.tuple_lg_k),
        "varopt_k": int(config.varopt_k),
    }


def _safe_rel_error(pred: float, truth: float) -> float:
    return (float(pred) - float(truth)) / max(1.0, abs(float(truth)))


def _datasketches_rank_error(sketch_name: str, k: int) -> float | None:
    try:
        import datasketches as ds  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        if sketch_name == "kll_floats":
            return float(ds.kll_floats_sketch.get_normalized_rank_error(int(k), False))
        if sketch_name == "quantiles_floats":
            return float(ds.quantiles_floats_sketch.get_normalized_rank_error(int(k), False))
    except Exception:
        return None
    return None


def _metric_row(
    *,
    family: str,
    sketch: str,
    query: str,
    implementation_status: str,
    formal_status: str,
    truth: Sequence[float],
    pred_by_schedule: Mapping[str, Sequence[float]],
    memory_bytes: Sequence[float],
    theoretical_error: float | None,
) -> Dict[str, object]:
    truth_arr = np.asarray([float(x) for x in truth], dtype=np.float64)
    balanced = np.asarray([float(x) for x in pred_by_schedule["balanced"]], dtype=np.float64)
    abs_err = np.abs(balanced - truth_arr)
    rel_err = np.asarray([_safe_rel_error(p, t) for p, t in zip(balanced, truth_arr)], dtype=np.float64)
    spread = []
    for i in range(len(truth_arr)):
        vals = [float(pred_by_schedule[s][i]) for s in VALID_SCHEDULES]
        spread.append(max(vals) - min(vals))
    spread_arr = np.asarray(spread, dtype=np.float64)
    if theoretical_error is not None and math.isfinite(float(theoretical_error)):
        coverage = float(np.mean(np.abs(rel_err) <= 2.0 * float(theoretical_error)))
    else:
        coverage = float("nan")
    return {
        "family": str(family),
        "sketch": str(sketch),
        "query": str(query),
        "implementation_status": str(implementation_status),
        "formal_status": str(formal_status),
        "n_docs": int(len(truth_arr)),
        "mae": float(np.mean(abs_err)) if len(abs_err) else 0.0,
        "rmse": float(math.sqrt(float(np.mean(abs_err * abs_err)))) if len(abs_err) else 0.0,
        "relative_rmse": float(math.sqrt(float(np.mean(rel_err * rel_err)))) if len(rel_err) else 0.0,
        "mean_abs_rel_error": float(np.mean(np.abs(rel_err))) if len(rel_err) else 0.0,
        "schedule_spread_mean": float(np.mean(spread_arr)) if len(spread_arr) else 0.0,
        "schedule_spread_p95": float(np.percentile(spread_arr, 95.0)) if len(spread_arr) else 0.0,
        "bound_coverage_2sigma": coverage,
        "theoretical_error": float(theoretical_error) if theoretical_error is not None else float("nan"),
        "memory_bytes_mean": float(np.mean(np.asarray(memory_bytes, dtype=np.float64))) if memory_bytes else 0.0,
        "official_floor_rel_rmse": float("nan"),
        "distance_to_official_floor": float("nan"),
    }


def _attach_official_floors(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    floors: Dict[Tuple[str, str], float] = {}
    for row in rows:
        if str(row.get("implementation_status")) not in {"official_empirical", "lean_backed"}:
            continue
        key = (str(row.get("family")), str(row.get("query")))
        val = float(row.get("relative_rmse", float("nan")))
        if not math.isfinite(val):
            continue
        floors[key] = min(val, floors.get(key, val))
    out: List[Dict[str, object]] = []
    for row in rows:
        r = dict(row)
        key = (str(r.get("family")), str(r.get("query")))
        floor = floors.get(key, float("nan"))
        r["official_floor_rel_rmse"] = float(floor)
        val = float(r.get("relative_rmse", float("nan")))
        r["distance_to_official_floor"] = float(val - floor) if math.isfinite(floor) else float("nan")
        out.append(r)
    return out


def _tree_bundle_contract(config: ClassicalSketchComparisonConfig) -> Dict[str, object]:
    return {
        "f_init": "official_oracle",
        "g_init": "official_merge",
        "fg_schedule": "balanced",
        "reducer_contract": "bottom_up",
        "tree_bundle_leaf_policy": {
            "leaf_unit_count": int(config.leaf_unit_count),
            "min_tokens": int(config.min_tokens),
            "max_tokens": int(config.max_tokens),
        },
        "tree_bundle_state_contract": "oracle_state",
        "tree_bundle_summary_dim": None,
        "tree_bundle_state_dim": None,
    }


def _token_docs(config: ClassicalSketchComparisonConfig) -> List[List[int]]:
    rng = np.random.default_rng(int(config.seed))
    docs: List[List[int]] = []
    ranks = np.arange(1, int(config.universe_size) + 1, dtype=np.float64)
    probs = np.power(ranks, -1.05)
    probs /= float(probs.sum())
    for _ in range(int(config.n_docs)):
        n = int(rng.integers(int(config.min_tokens), int(config.max_tokens) + 1))
        toks = rng.choice(int(config.universe_size), size=n, replace=True, p=probs)
        docs.append([int(x) for x in toks.tolist()])
    return docs


def _float_docs(config: ClassicalSketchComparisonConfig) -> List[List[float]]:
    rng = np.random.default_rng(int(config.seed) + 97)
    docs: List[List[float]] = []
    for _ in range(int(config.n_docs)):
        n = int(rng.integers(int(config.min_tokens), int(config.max_tokens) + 1))
        mix = rng.uniform(size=n) < 0.15
        vals = rng.normal(0.0, 1.0, size=n)
        vals[mix] += rng.normal(3.0, 0.8, size=int(np.sum(mix)))
        docs.append([float(x) for x in vals.tolist()])
    return docs


def _run_distinct(config: ClassicalSketchComparisonConfig, docs: Sequence[Sequence[int]]) -> List[Dict[str, object]]:
    lg = int(config.distinct_lg_k)
    adapters = [
        (
            make_hll_adapter(backend="datasketches", precision=lg),
            "official_empirical",
            "lean_backed_family",
            1.04 / math.sqrt(float(1 << lg)),
        ),
        (
            make_cpc_adapter(lg_k=lg),
            "official_empirical",
            "empirical_only",
            2.0 / math.sqrt(float(1 << lg)),
        ),
        (
            make_theta_adapter(lg_k=int(config.theta_lg_k)),
            "official_empirical",
            "empirical_only",
            1.0 / math.sqrt(float(1 << int(config.theta_lg_k))),
        ),
    ]
    rows: List[Dict[str, object]] = []
    truth = [float(len(set(doc))) for doc in docs]
    for adapter, status, formal_status, theory in adapters:
        pred: Dict[str, List[float]] = {s: [] for s in VALID_SCHEDULES}
        mem: List[float] = []
        for doc in docs:
            leaves = _leaves(config, list(doc))
            leaf_states = [adapter.encode(leaf) for leaf in leaves]
            for sched in VALID_SCHEDULES:
                root = fold_states(leaf_states, adapter, schedule=sched)
                pred[sched].append(float(adapter.query(root, None)))
                if sched == "balanced":
                    mem.append(float(adapter.memory_bytes(root)))
        rows.append(
            _metric_row(
                family="distinct",
                sketch=str(adapter.name),
                query="cardinality",
                implementation_status=status,
                formal_status=formal_status,
                truth=truth,
                pred_by_schedule=pred,
                memory_bytes=mem,
                theoretical_error=theory,
            )
        )

    exact_pred = {s: truth for s in VALID_SCHEDULES}
    rows.append(
        _metric_row(
            family="distinct",
            sketch="exact_set",
            query="cardinality",
            implementation_status="control",
            formal_status="control",
            truth=truth,
            pred_by_schedule=exact_pred,
            memory_bytes=[float(len(set(doc)) * 8) for doc in docs],
            theoretical_error=0.0,
        )
    )
    wrong = [float(sum(len(set(leaf)) for leaf in _leaves(config, list(doc)))) for doc in docs]
    rows.append(
        _metric_row(
            family="distinct",
            sketch="sum_leaf_uniques",
            query="cardinality",
            implementation_status="negative_control",
            formal_status="negative_control",
            truth=truth,
            pred_by_schedule={s: wrong for s in VALID_SCHEDULES},
            memory_bytes=[0.0 for _ in docs],
            theoretical_error=None,
        )
    )
    return rows


def _run_frequency(config: ClassicalSketchComparisonConfig, docs: Sequence[Sequence[int]]) -> List[Dict[str, object]]:
    adapters = [
        (
            make_count_min_adapter(num_hashes=int(config.cms_num_hashes), num_buckets=int(config.cms_num_buckets)),
            "official_empirical",
            "lean_backed_family",
            1.0 / float(max(1, config.cms_num_buckets)),
        ),
        (
            make_frequent_strings_adapter(lg_max_map_size=int(config.frequent_lg_max_map_size)),
            "official_empirical",
            "empirical_only",
            None,
        ),
    ]
    rows: List[Dict[str, object]] = []
    for adapter, status, formal_status, theory in adapters:
        truth: List[float] = []
        pred: Dict[str, List[float]] = {s: [] for s in VALID_SCHEDULES}
        mem: List[float] = []
        for doc in docs:
            if str(adapter.name) == "count_min_datasketches":
                items = [int(x) for x in doc]
                counts = Counter(items)
                leaves = _leaves(config, items)
            else:
                items = [str(x) for x in doc]
                counts = Counter(items)
                leaves = _leaves(config, items)
            keys = [k for k, _ in counts.most_common(5)]
            leaf_states = [adapter.encode(leaf) for leaf in leaves]
            roots = {sched: fold_states(leaf_states, adapter, schedule=sched) for sched in VALID_SCHEDULES}
            for key in keys:
                truth.append(float(counts[key]))
                for sched, root in roots.items():
                    pred[sched].append(float(adapter.query(root, key)))
            mem.append(float(adapter.memory_bytes(roots["balanced"])))
        rows.append(
            _metric_row(
                family="frequency",
                sketch=str(adapter.name),
                query="top5_point_frequency",
                implementation_status=status,
                formal_status=formal_status,
                truth=truth,
                pred_by_schedule=pred,
                memory_bytes=mem,
                theoretical_error=theory,
            )
        )
    return rows


def _exact_rank(values: Sequence[float], x: float) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    if len(arr) == 0:
        return 0.0
    return float(np.searchsorted(arr, float(x), side="right")) / float(len(arr))


def _run_quantile(config: ClassicalSketchComparisonConfig, docs: Sequence[Sequence[float]]) -> List[Dict[str, object]]:
    adapters = [
        (
            make_kll_floats_adapter(k=int(config.kll_k)),
            "official_empirical",
            "lean_backed_family",
            _datasketches_rank_error("kll_floats", int(config.kll_k)),
        ),
        (
            make_quantiles_floats_adapter(k=int(config.quantiles_k)),
            "official_empirical",
            "empirical_only",
            _datasketches_rank_error("quantiles_floats", int(config.quantiles_k)),
        ),
        (
            make_req_floats_adapter(k=int(config.req_k), high_rank_accuracy=True),
            "official_empirical",
            "empirical_only",
            None,
        ),
        (make_tdigest_double_adapter(k=int(config.tdigest_k)), "official_empirical", "empirical_only", None),
    ]
    rows: List[Dict[str, object]] = []
    for adapter, status, formal_status, theory in adapters:
        for q in tuple(float(x) for x in config.quantile_queries):
            truth_rank: List[float] = []
            pred_rank: Dict[str, List[float]] = {s: [] for s in VALID_SCHEDULES}
            mem: List[float] = []
            for doc in docs:
                leaves = _leaves(config, list(doc))
                leaf_states = [adapter.encode(leaf) for leaf in leaves]
                roots = {sched: fold_states(leaf_states, adapter, schedule=sched) for sched in VALID_SCHEDULES}
                truth_rank.append(float(q))
                for sched, root in roots.items():
                    q_val = float(adapter.query(root, q))
                    pred_rank[sched].append(_exact_rank(doc, q_val))
                mem.append(float(adapter.memory_bytes(roots["balanced"])))
            rows.append(
                _metric_row(
                    family="quantile",
                    sketch=str(adapter.name),
                    query=f"rank_at_q{q:g}",
                    implementation_status=status,
                    formal_status=formal_status,
                    truth=truth_rank,
                    pred_by_schedule=pred_rank,
                    memory_bytes=mem,
                    theoretical_error=theory,
                )
            )
    return rows


def _run_set_ops(config: ClassicalSketchComparisonConfig, docs: Sequence[Sequence[int]]) -> List[Dict[str, object]]:
    adapter = make_theta_adapter(lg_k=int(config.theta_lg_k))
    pairs = list(zip(docs[0::2], docs[1::2]))
    rows: List[Dict[str, object]] = []
    for op_name in ("union", "intersection", "a_not_b"):
        truth: List[float] = []
        pred: Dict[str, List[float]] = {s: [] for s in VALID_SCHEDULES}
        mem: List[float] = []
        for a_doc, b_doc in pairs:
            a_set = set(a_doc)
            b_set = set(b_doc)
            if op_name == "union":
                truth.append(float(len(a_set | b_set)))
            elif op_name == "intersection":
                truth.append(float(len(a_set & b_set)))
            else:
                truth.append(float(len(a_set - b_set)))

            a_leaves = _leaves(config, list(a_doc))
            b_leaves = _leaves(config, list(b_doc))
            a_states = [adapter.encode(leaf) for leaf in a_leaves]
            b_states = [adapter.encode(leaf) for leaf in b_leaves]
            for sched in VALID_SCHEDULES:
                a_root = fold_states(a_states, adapter, schedule=sched)
                b_root = fold_states(b_states, adapter, schedule=sched)
                if op_name == "union":
                    pred[sched].append(theta_union_estimate(a_root, b_root, lg_k=int(config.theta_lg_k)))
                elif op_name == "intersection":
                    pred[sched].append(theta_intersection_estimate(a_root, b_root))
                else:
                    pred[sched].append(theta_a_not_b_estimate(a_root, b_root))
                if sched == "balanced":
                    mem.append(float(adapter.memory_bytes(a_root) + adapter.memory_bytes(b_root)))
        rows.append(
            _metric_row(
                family="set",
                sketch="theta_datasketches",
                query=op_name,
                implementation_status="official_empirical",
                formal_status="empirical_only",
                truth=truth,
                pred_by_schedule=pred,
                memory_bytes=mem,
                theoretical_error=1.0 / math.sqrt(float(1 << int(config.theta_lg_k))),
            )
        )
    return rows


def _run_sampling(config: ClassicalSketchComparisonConfig, docs: Sequence[Sequence[int]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    tuple_adapter = make_tuple_accumulator_adapter(lg_k=int(config.tuple_lg_k))
    varopt_adapter = make_varopt_strings_adapter(k=int(config.varopt_k))

    tuple_truth: List[float] = []
    tuple_pred: Dict[str, List[float]] = {s: [] for s in VALID_SCHEDULES}
    tuple_mem: List[float] = []
    for doc in docs:
        items = [(str(x), 1) for x in doc]
        leaves = _leaves(config, items)
        leaf_states = [tuple_adapter.encode(leaf) for leaf in leaves]
        tuple_truth.append(float(len(doc)))
        for sched in VALID_SCHEDULES:
            root = fold_states(leaf_states, tuple_adapter, schedule=sched)
            tuple_pred[sched].append(float(tuple_adapter.query(root, "summary_sum")))
            if sched == "balanced":
                tuple_mem.append(float(tuple_adapter.memory_bytes(root)))
    rows.append(
        _metric_row(
            family="sampling",
            sketch=str(tuple_adapter.name),
            query="accumulator_summary_sum",
            implementation_status="official_empirical",
            formal_status="empirical_only",
            truth=tuple_truth,
            pred_by_schedule=tuple_pred,
            memory_bytes=tuple_mem,
            theoretical_error=None,
        )
    )

    varopt_truth: List[float] = []
    varopt_pred: Dict[str, List[float]] = {s: [] for s in VALID_SCHEDULES}
    varopt_mem: List[float] = []
    for doc in docs:
        leaves = [[str(x) for x in leaf] for leaf in _leaves(config, list(doc))]
        leaf_states = [varopt_adapter.encode(leaf) for leaf in leaves]
        varopt_truth.append(float(len(doc)))
        for sched in VALID_SCHEDULES:
            root = fold_states(leaf_states, varopt_adapter, schedule=sched)
            varopt_pred[sched].append(float(varopt_adapter.query(root, None)))
            if sched == "balanced":
                varopt_mem.append(float(varopt_adapter.memory_bytes(root)))
    rows.append(
        _metric_row(
            family="sampling",
            sketch=str(varopt_adapter.name),
            query="total_weight",
            implementation_status="official_empirical",
            formal_status="empirical_only",
            truth=varopt_truth,
            pred_by_schedule=varopt_pred,
            memory_bytes=varopt_mem,
            theoretical_error=None,
        )
    )
    return rows


def run_classical_sketch_comparison(config: ClassicalSketchComparisonConfig) -> ClassicalSketchComparisonSummary:
    token_docs = _token_docs(config)
    float_docs = _float_docs(config)
    families = {str(x).strip().lower() for x in tuple(config.include_families)}
    rows: List[Dict[str, object]] = []
    if "distinct" in families:
        rows.extend(_run_distinct(config, token_docs))
    if "frequency" in families:
        rows.extend(_run_frequency(config, token_docs))
    if "quantile" in families:
        rows.extend(_run_quantile(config, float_docs))
    if "set" in families:
        rows.extend(_run_set_ops(config, token_docs))
    if "sampling" in families:
        rows.extend(_run_sampling(config, token_docs))
    rows = _attach_official_floors(rows)
    axes = _row_axes(config, list(token_docs) + list(float_docs))
    contract = _tree_bundle_contract(config)
    rows = [dict(row, **axes, **contract) for row in rows]
    return ClassicalSketchComparisonSummary(config=asdict(config), rows=tuple(rows))


def experiment_rows(summary: ClassicalSketchComparisonSummary) -> List[Dict[str, object]]:
    return [dict(r) for r in summary.rows]


__all__ = [
    "ClassicalSketchComparisonConfig",
    "ClassicalSketchComparisonSummary",
    "experiment_rows",
    "run_classical_sketch_comparison",
]
