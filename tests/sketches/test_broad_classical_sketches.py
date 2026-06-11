from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("datasketches")

from treepo.bench.classical_sketches import (  # noqa: E402
    ClassicalSketchComparisonConfig,
    run_classical_sketch_comparison,
)
from treepo.bench.reports.classical_sketches import (  # noqa: E402
    _aggregate,
    _is_exact_state_row,
    _is_projection_row,
    _markdown,
    _plot_leafsize_hll,
)
from treepo.bench.suites.classical_sketches import build_classical_sketches_suite  # noqa: E402
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
    assert any(str(r["implementation_status"]) == "lean_backed" for r in summary.rows)
    assert any(str(r["implementation_status"]) == "official_empirical" for r in summary.rows)


def test_treepo_bench_classical_sketches_suite_smoke(tmp_path) -> None:
    env = dict(os.environ)
    repo_treepo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["PYTHONPATH"] = repo_treepo + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        "-m",
        "treepo.bench.cli",
        "suite",
        "classical-sketches",
        "--out-root",
        str(tmp_path),
        "--jobs",
        "1",
        "--seeds",
        "0",
        "--leaf-sizes",
        "16,32",
        "--capacities",
        "small",
    ]
    proc = subprocess.run(cmd, cwd=str(tmp_path), env=env, text=True, capture_output=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["suite"] == "classical-sketches"
    assert payload["n_runs"] == 2
    assert all(result["status"] == "ok" for result in payload["results"])


def test_classical_sketch_suite_expands_learned_supervision_r_grid(tmp_path) -> None:
    specs = build_classical_sketches_suite(
        out_root=tmp_path,
        skip_existing=False,
        seeds="0",
        leaf_sizes="16",
        capacities="small",
        include_learned=True,
        learned_local_label_rates="0.1,1.0",
    )
    assert len(specs) == 2
    rates = sorted(float(spec.config["learned_leaf_query_rate"]) for spec in specs)
    assert rates == [0.1, 1.0]
    assert all(
        spec.config["learned_leaf_query_rate"] == spec.config["learned_internal_query_rate"]
        for spec in specs
    )
    paths = {str(spec.json_out) for spec in specs}
    assert any("/rootR100_leafR10_internalR10/" in path for path in paths)
    assert any("/R100/" in path for path in paths)


def test_classical_sketch_suite_expands_internal_only_supervision_r_grid(tmp_path) -> None:
    specs = build_classical_sketches_suite(
        out_root=tmp_path,
        skip_existing=False,
        seeds="0",
        leaf_sizes="16,64",
        capacities="small",
        include_learned=True,
        learned_leaf_query_rates="0.0",
        learned_internal_query_rates="0.1,0.3",
    )
    assert len(specs) == 4
    pairs = sorted(
        (
            float(spec.config["learned_root_query_rate"]),
            float(spec.config["learned_leaf_query_rate"]),
            float(spec.config["learned_internal_query_rate"]),
        )
        for spec in specs
    )
    assert pairs == [(1.0, 0.0, 0.1), (1.0, 0.0, 0.1), (1.0, 0.0, 0.3), (1.0, 0.0, 0.3)]
    paths = {str(spec.json_out) for spec in specs}
    assert any("/rootR100_leafR0_internalR10/" in path for path in paths)
    assert any("/rootR100_leafR0_internalR30/" in path for path in paths)


def test_classical_sketch_suite_expands_root_only_supervision_rate(tmp_path) -> None:
    specs = build_classical_sketches_suite(
        out_root=tmp_path,
        skip_existing=False,
        seeds="0",
        leaf_sizes="16",
        capacities="small",
        include_learned=True,
        learned_root_query_rates="0.9",
        learned_leaf_query_rates="0.0",
        learned_internal_query_rates="0.0",
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec.config["learned_root_query_rate"] == 0.9
    assert spec.config["learned_leaf_query_rate"] == 0.0
    assert spec.config["learned_internal_query_rate"] == 0.0
    assert "/rootR90_leafR0_internalR0/" in str(spec.json_out)


def test_classical_sketch_suite_expands_uniform_all_nodes_supervision_grid(tmp_path) -> None:
    specs = build_classical_sketches_suite(
        out_root=tmp_path,
        skip_existing=False,
        seeds="0",
        leaf_sizes="16",
        capacities="small",
        include_learned=True,
        learned_local_label_rates="0.1,0.3",
        learned_supervision_sampling_policy="uniform_all_nodes",
    )
    assert len(specs) == 2
    assert all(
        spec.config["learned_supervision_sampling_policy"] == "uniform_all_nodes"
        for spec in specs
    )
    assert all(
        spec.config["learned_root_query_rate"]
        == spec.config["learned_leaf_query_rate"]
        == spec.config["learned_internal_query_rate"]
        for spec in specs
    )
    assert all(
        spec.config["learned_leaf_query_rate"] == spec.config["learned_internal_query_rate"]
        for spec in specs
    )
    paths = {str(spec.json_out) for spec in specs}
    assert any("/uniform_all_nodes/R10/" in path for path in paths)
    assert any("/uniform_all_nodes/R30/" in path for path in paths)


def test_classical_sketch_suite_defaults_learned_target_jobs_to_auto(tmp_path) -> None:
    specs = build_classical_sketches_suite(
        out_root=tmp_path,
        skip_existing=False,
        seeds="0",
        leaf_sizes="16",
        capacities="small",
        include_learned=True,
    )
    assert len(specs) == 1
    assert specs[0].config["learned_target_jobs"] == "auto"


def test_classical_sketch_report_preserves_supervision_rate_axis() -> None:
    rows = []
    for internal_rate, value in ((0.1, 0.2), (0.3, 0.1)):
        rows.append(
            {
                "family": "distinct",
                "sketch": "learned_joint_exact_distinct",
                "query": "cardinality",
                "capacity_label": "small",
                "n_leaves": -1,
                "leaf_size": 16,
                "leaf_axis": "leaf_size",
                "learned_leaf_query_rate": 0.0,
                "learned_root_query_rate": 1.0,
                "learned_internal_query_rate": internal_rate,
                "learned_supervision_sampling_policy": "uniform_all_nodes",
                "leaf_count_min": 2,
                "leaf_count_mean": 3.0,
                "leaf_count_max": 4,
                "seed": 0,
                "implementation_status": "learned_empirical",
                "formal_status": "learned_empirical",
                "relative_rmse": value,
                "schedule_spread_mean": 0.0,
                "distance_to_official_floor": value,
                "official_floor_rel_rmse": 0.0,
                "bound_coverage_2sigma": 1.0,
                "theoretical_error": 0.0,
                "memory_bytes_mean": 128.0,
                "learned_variant": "fg",
            }
        )
    agg = _aggregate(rows)
    assert len(agg) == 2
    assert [row["learned_internal_query_rate"] for row in agg] == [0.1, 0.3]
    assert [row["learned_root_query_rate"] for row in agg] == [1.0, 1.0]
    assert {row["learned_supervision_sampling_policy"] for row in agg} == {"uniform_all_nodes"}
    assert [row["relative_rmse_mean"] for row in agg] == [0.2, 0.1]


def test_classical_sketch_report_keeps_leaf_size_rows_separate() -> None:
    rows = []
    for leaf_size, value in ((16, 0.2), (32, 0.1)):
        rows.append(
            {
                "family": "distinct",
                "sketch": "hll_datasketches",
                "query": "cardinality",
                "capacity_label": "large",
                "n_leaves": -1,
                "leaf_size": leaf_size,
                "leaf_axis": "leaf_size",
                "leaf_count_min": 2,
                "leaf_count_mean": 3.0,
                "leaf_count_max": 4,
                "seed": 0,
                "implementation_status": "official_empirical",
                "formal_status": "empirical_only",
                "relative_rmse": value,
                "schedule_spread_mean": 0.0,
                "distance_to_official_floor": 0.0,
                "official_floor_rel_rmse": value,
                "bound_coverage_2sigma": 1.0,
                "theoretical_error": 0.03,
                "memory_bytes_mean": 128.0,
            }
        )
    agg = _aggregate(rows)
    assert [row["leaf_size"] for row in agg] == [16, 32]
    assert all(row["n_leaves"] == -1 for row in agg)


def test_classical_sketch_report_preserves_projection_metadata() -> None:
    rows = []
    for leaf_size in (16, 32):
        rows.append(
            {
                "family": "distinct",
                "sketch": "learned_joint_exact_distinct",
                "query": "cardinality",
                "capacity_label": "large",
                "n_leaves": -1,
                "leaf_size": leaf_size,
                "leaf_axis": "leaf_size",
                "leaf_count_min": 2,
                "leaf_count_mean": 3.0,
                "leaf_count_max": 4,
                "seed": 0,
                "implementation_status": "learned_empirical",
                "formal_status": "learned_empirical",
                "relative_rmse": 0.1,
                "schedule_spread_mean": 0.0,
                "distance_to_official_floor": 0.1,
                "official_floor_rel_rmse": 0.0,
                "bound_coverage_2sigma": 1.0,
                "theoretical_error": 0.0,
                "memory_bytes_mean": 128.0,
                "learned_variant": "fg",
                "learned_target_kind": "exact_distinct",
                "projection_kind": "mergeable_projection",
                "state_space_kind": "projection_latent",
                "merge_kind": "learned_projection",
                "readout_kind": "learned_scalar",
                "leaf_feature_dim": 64,
                "learned_state_dim": 32,
                "g_input_dim": 64,
                "output_dim": 1,
            }
        )
    agg = _aggregate(rows)
    assert {row["leaf_size"] for row in agg} == {16, 32}
    assert all(row["projection_kind"] == "mergeable_projection" for row in agg)
    assert all(row["state_space_kind"] == "projection_latent" for row in agg)
    assert all(row["merge_kind"] == "learned_projection" for row in agg)
    assert all(row["readout_kind"] == "learned_scalar" for row in agg)
    assert all(row["learned_state_dim"] == 32 for row in agg)
    assert all(row["g_input_dim"] == 64 for row in agg)


def test_classical_sketch_report_splits_exact_state_and_projection_rows() -> None:
    projection = {
        "family": "distinct",
        "sketch": "learned_joint_exact_distinct",
        "query": "cardinality",
        "capacity_label": "large",
        "n_leaves": 2,
        "leaf_size": -1,
        "leaf_axis": "n_leaves",
        "implementation_status": "learned_empirical",
        "formal_status": "learned_empirical",
        "relative_rmse": 0.1,
        "projection_kind": "mergeable_projection",
        "memory_bytes_mean": 128.0,
    }
    exact = {
        **projection,
        "sketch": "learned_g_exact_distinct_union_state_space",
        "learned_target_kind": "exact_distinct_union_state_space",
        "projection_kind": "exact_distinct_union_oracle_state",
        "state_space_kind": "fixed_numeric_vector",
        "exact_state_mode": "structured_exact",
        "relative_rmse": 0.0,
    }

    agg = _aggregate([projection, exact])
    assert sum(1 for row in agg if _is_projection_row(row)) == 1
    assert sum(1 for row in agg if _is_exact_state_row(row)) == 1
    report = _markdown(agg)
    assert "Learned exact-state recovery rows: 1." in report
    assert "Learned mergeable-projection diagnostic rows: 1." in report


def test_official_only_hll_figure_title_does_not_claim_learned_merge(tmp_path) -> None:
    pypdf = pytest.importorskip("pypdf")
    rows = []
    for cap, lg_k in (("small", 8), ("medium", 10)):
        for leaf_size in (16, 32):
            rows.append(
                {
                    "family": "distinct",
                    "sketch": "hll_datasketches",
                    "query": "cardinality",
                    "capacity_label": cap,
                    "n_leaves": -1,
                    "leaf_size": leaf_size,
                    "leaf_axis": "leaf_size",
                    "leaf_count_min": 2,
                    "leaf_count_mean": 3.0,
                    "leaf_count_max": 4,
                    "tokens_per_leaf_mean": float(leaf_size),
                    "n_runs": 1,
                    "implementation_status": "official_empirical",
                    "formal_status": "lean_backed_family",
                    "relative_rmse_mean": 0.04,
                    "relative_rmse_ci95": 0.0,
                    "schedule_spread_mean": 0.0,
                    "theoretical_error_mean": 0.03,
                    "memory_bytes_mean": 128.0,
                    "distinct_lg_k": lg_k,
                }
            )
    stem = tmp_path / "hll"
    _plot_leafsize_hll(rows, stem)
    assert stem.with_suffix(".pdf").exists()
    assert stem.with_suffix(".png").exists()
    text = "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(str(stem.with_suffix(".pdf"))).pages)
    assert "Official HLL over leaf size" in text
    assert "learned merge" not in text.lower()
