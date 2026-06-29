from __future__ import annotations

import pytest

pytest.importorskip("datasketches")

from treepo.bench.classical_sketches import (  # noqa: E402
    ClassicalSketchComparisonConfig,
    run_classical_sketch_comparison,
)
from treepo.bench.sketches import (  # noqa: E402
    make_count_min_adapter,
    make_cpc_adapter,
    make_frequent_strings_adapter,
    make_kll_floats_adapter,
    make_quantiles_floats_adapter,
    make_req_floats_adapter,
    make_tdigest_double_adapter,
    make_theta_adapter,
    make_tuple_accumulator_adapter,
    make_varopt_strings_adapter,
)
from treepo.bench.sketches.tree_reducer import fold_states  # noqa: E402


def test_cardinality_adapters_update_merge_query_and_size() -> None:
    for adapter in (make_cpc_adapter(lg_k=8), make_theta_adapter(lg_k=8)):
        left = adapter.encode(range(100))
        right = adapter.encode(range(50, 175))
        merged = adapter.merge(left, right)
        est = adapter.query(merged, None)
        assert 120.0 <= est <= 220.0
        assert adapter.serialized_size_bytes(merged) > 0
        assert adapter.memory_bytes(merged) > 0


def test_frequency_adapters_estimate_merged_counts() -> None:
    for adapter in (
        make_count_min_adapter(num_hashes=5, num_buckets=128),
        make_frequent_strings_adapter(lg_max_map_size=8),
    ):
        left = adapter.encode(["a", "a", "b"])
        right = adapter.encode(["a", "c", "c"])
        merged = adapter.merge(left, right)
        assert adapter.query(merged, "a") >= 3.0
        assert adapter.query(merged, "c") >= 2.0
        assert adapter.serialized_size_bytes(merged) > 0


def test_frequency_adapters_support_weighted_updates() -> None:
    for adapter in (
        make_count_min_adapter(num_hashes=5, num_buckets=128),
        make_frequent_strings_adapter(lg_max_map_size=8),
    ):
        left = adapter.encode([("hamlet", 12), ("ghost", 2)])
        right = adapter.encode([("hamlet", 8), ("ophelia", 5)])
        merged = adapter.merge(left, right)
        assert adapter.query(merged, "hamlet") >= 20.0
        assert adapter.query(merged, "ghost") >= 2.0
        assert adapter.query(merged, "ophelia") >= 5.0


@pytest.mark.parametrize(
    "adapter",
    [
        make_kll_floats_adapter(k=128),
        make_quantiles_floats_adapter(k=128),
        make_req_floats_adapter(k=12),
        make_tdigest_double_adapter(k=100),
    ],
)
def test_quantile_adapters_merge_and_query(adapter) -> None:
    values = [float(i) for i in range(1000)]
    leaves = [values[i : i + 100] for i in range(0, len(values), 100)]
    states = [adapter.encode(leaf) for leaf in leaves]
    roots = [fold_states(states, adapter, schedule=s) for s in ("balanced", "left_to_right", "right_to_left")]
    estimates = [adapter.query(root, 0.5) for root in roots]
    assert all(420.0 <= x <= 580.0 for x in estimates)
    assert max(estimates) - min(estimates) <= 80.0
    assert adapter.serialized_size_bytes(roots[0]) > 0


def test_tuple_and_varopt_adapters_update_merge_query_and_size() -> None:
    tuple_adapter = make_tuple_accumulator_adapter(lg_k=10)
    left = tuple_adapter.encode([("a", 1), ("a", 2), ("b", 1)])
    right = tuple_adapter.encode([("a", 3), ("c", 4)])
    merged = tuple_adapter.merge(left, right)
    assert tuple_adapter.query(merged, None) == 3.0
    assert tuple_adapter.query(merged, "summary_sum") == 11.0
    assert tuple_adapter.serialized_size_bytes(merged) > 0

    varopt_adapter = make_varopt_strings_adapter(k=8)
    left_v = varopt_adapter.encode(["a", "b", "c"])
    right_v = varopt_adapter.encode(["d", "e", "f"])
    merged_v = varopt_adapter.merge(left_v, right_v)
    assert 3.0 <= varopt_adapter.query(merged_v, None) <= 9.0
    assert varopt_adapter.query(merged_v, "num_samples") > 0.0
    assert varopt_adapter.serialized_size_bytes(merged_v) > 0


def test_classical_sketch_comparison_smoke_covers_all_families() -> None:
    summary = run_classical_sketch_comparison(
        ClassicalSketchComparisonConfig(
            seed=3,
            n_docs=8,
            min_tokens=64,
            max_tokens=96,
            leaf_size=32,
            distinct_lg_k=8,
            theta_lg_k=8,
            cms_num_buckets=128,
            include_families=("distinct", "frequency", "quantile", "set", "sampling"),
        )
    )
    families = {str(r["family"]) for r in summary.rows}
    assert {"distinct", "frequency", "quantile", "set", "sampling"} <= families
    assert any(str(r["implementation_status"]) == "negative_control" for r in summary.rows)
    official_rows = [r for r in summary.rows if str(r["implementation_status"]) == "official_empirical"]
    assert official_rows
