"""Centralized dispatch — axis-factored, not Cartesian.

Four canonical methods (``fit`` / ``oracle`` / ``sketch`` / ``audit``).
Each one represents one orthogonal axis. Adding a new oracle, sketch
adapter, fixture, or family does *not* require a new method — the new
thing is passed by name through config.

These tests:

1. Enumerate ``list_methods()`` and confirm all four are present.
2. Verify each axis dispatches by name: multiple oracles through one
   ``"oracle"`` method, multiple sketch_kinds through one ``"sketch"``
   method.
3. Verify the registry catches unknown methods AND unknown config keys
   AND unknown oracle/sketch names AND missing required keys.
4. Verify ``fit`` accepts both ``spec`` and high-level kwargs.
5. Verify ``audit`` runs on real audit rows without going through fit.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from treepo._research.ctreepo.contracts import CTreePOFitResult, CTreePOLearningSpec
import treepo.cld
from treepo.cld import (
    LocalLawAuditRow,
    allowed_config_keys,
    list_methods,
    list_oracle_domains_with_fixtures,
    list_registered_oracles,
    list_sketch_kinds,
    method_info,
    run,
)


_EXPECTED_METHODS = {"fit", "oracle", "sketch", "audit", "probe"}


# --------------------------------------------------------------------------- #
# 1. Discovery surface
# --------------------------------------------------------------------------- #


def test_methods_is_minimal_axis_factored_set() -> None:
    """The canonical method set is exactly four axis names, not a
    Cartesian product of (data, oracle, sketch, family).
    """
    assert set(list_methods()) == _EXPECTED_METHODS


def test_method_info_for_each_axis() -> None:
    for method in list_methods():
        info = method_info(method)
        assert info["name"] == method
        assert isinstance(info["allowed_config_keys"], list)
        assert len(info["allowed_config_keys"]) >= 1


def test_oracle_discovery_returns_registered_oracles() -> None:
    """Adding a new ``register_oracle(...)`` elsewhere must be reachable
    by ``run("oracle", {"oracle_name": ...})`` without touching this file.
    """
    oracles = list_registered_oracles()
    for required in ("hll_exact", "leaf_local_mixture_target", "markov_changepoint_count"):
        assert required in oracles, f"oracle {required!r} not registered"


def test_sketch_discovery_includes_hll() -> None:
    kinds = list_sketch_kinds()
    assert "hll" in kinds
    assert "count_min" in kinds


def test_oracle_domain_fixtures_advertised() -> None:
    domains = list_oracle_domains_with_fixtures()
    assert "classical_sketch" in domains
    assert "lda" in domains


# --------------------------------------------------------------------------- #
# 2. Each axis dispatches by *config*, not by method name
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "oracle_name",
    ["hll_exact", "leaf_local_mixture_target"],
)
def test_oracle_method_dispatches_by_name(oracle_name: str, tmp_path: Path) -> None:
    result = run(
        "oracle",
        {"oracle_name": oracle_name, "seed": 0, "output_dir": str(tmp_path / oracle_name)},
    )
    assert isinstance(result, CTreePOFitResult)
    assert result.status == "success"
    # Oracle scored against precomputed teacher truth → MAE ≈ 0.
    assert result.metrics["internal_f_mae"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "sketch_kind,sketch_cfg",
    [
        ("hll", {"precision": 12}),
        ("hll", {"precision": 14}),
    ],
)
def test_sketch_method_dispatches_by_kind(
    sketch_kind: str,
    sketch_cfg: dict,
    tmp_path: Path,
) -> None:
    cfg = {
        "sketch_kind": sketch_kind,
        "n_trees": 4,
        "leaf_token_count": 16,
        "vocabulary_size": 64,
        "seed": 1,
        "output_dir": str(tmp_path),
        **sketch_cfg,
    }
    result = run("sketch", cfg)
    assert isinstance(result, CTreePOFitResult)
    assert result.status == "success"
    assert "internal_f_mae" in result.metrics


# --------------------------------------------------------------------------- #
# 3. Registry hygiene — failures are explicit, not silent.
# --------------------------------------------------------------------------- #


def test_unknown_method_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="unknown method"):
        run("does-not-exist", {})


def test_unknown_config_key_raises_valueerror_before_dispatch() -> None:
    with pytest.raises(ValueError, match="unknown config keys"):
        run("oracle", {"oracle_name": "hll_exact", "not_a_key": 1})


def test_oracle_method_rejects_missing_name() -> None:
    with pytest.raises(ValueError, match="oracle_name"):
        run("oracle", {})


def test_oracle_method_rejects_unknown_oracle_name() -> None:
    with pytest.raises(KeyError):
        run("oracle", {"oracle_name": "no-such-oracle"})


def test_sketch_method_rejects_missing_kind() -> None:
    with pytest.raises(ValueError, match="sketch_kind"):
        run("sketch", {})


def test_sketch_method_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown sketch_kind"):
        run("sketch", {"sketch_kind": "no-such-sketch"})


def test_oracle_method_auto_builds_markov_fixture(tmp_path: Path) -> None:
    """The Markov oracle now has a registered auto-fixture builder; calling
    without ``eval_data`` builds the canonical paper DGP corpus from the
    knobs in the config (or defaults).
    """
    result = run(
        "oracle",
        {
            "oracle_name": "markov_changepoint_count",
            "output_dir": str(tmp_path),
            # Tiny corpus so the test stays under a second.
            "train_docs": 0, "test_docs": 4,
            "n_regimes": 3,
            "min_tokens": 64, "max_tokens": 64, "max_seg_len": 16,
            "sinkhorn_iters": 5,
        },
    )
    assert getattr(result, "status", "?") in {"success", "fit_completed"}
    # MAE should be exactly 0 — the oracle counts the same true boundaries
    # the fixture wrote into metadata.
    assert float(dict(result.metrics).get("external_expert_mae", 1.0)) == 0.0


def test_register_method_rejects_duplicate_without_replace() -> None:
    with pytest.raises(ValueError, match="already registered"):
        treepo.cld.register_method(
            "fit",
            lambda cfg: None,
            allowed_config_keys=set(),
        )


# --------------------------------------------------------------------------- #
# 4. fit accepts both shapes
# --------------------------------------------------------------------------- #


def test_fit_method_accepts_full_spec(tmp_path: Path) -> None:
    from treepo.cld.fixtures import make_hll_token_trees

    spec = CTreePOLearningSpec(
        space_kind="fit_check",
        family="oracle",
        schedule="fg",
        initial_artifacts={"f": None, "g": None},
        train_data=[],
        eval_data=make_hll_token_trees(n_trees=4, seed=2),
        backend_config={"oracle_name": "hll_exact", "output_dir": str(tmp_path)},
        axis={"max_iterations": 0, "axis_value": 0},
    )
    result = run("fit", {"spec": spec})
    assert isinstance(result, CTreePOFitResult)
    assert result.status == "success"


def test_fit_method_accepts_high_level_kwargs(tmp_path: Path) -> None:
    from treepo.cld.fixtures import make_hll_token_trees

    result = run(
        "fit",
        {
            "family": "oracle",
            "eval_data": make_hll_token_trees(n_trees=4, seed=3),
            "backend_config": {"oracle_name": "hll_exact", "output_dir": str(tmp_path)},
        },
    )
    assert isinstance(result, CTreePOFitResult)
    assert result.status == "success"


def test_fit_method_requires_spec_or_family() -> None:
    with pytest.raises(ValueError, match="spec.*family"):
        run("fit", {})


def test_fit_method_learnable_constant_through_high_level_kwargs(tmp_path: Path) -> None:
    """The learnable_constant family is reachable through plain
    ``fit(family=..., train_data=...)`` — no separate "train" method
    needed.
    """
    train_trees: List[SimpleNamespace] = [
        SimpleNamespace(
            leaves=[SimpleNamespace(tokens=[0])],
            metadata={"split": "train", "teacher_score_1_7": 5.0, "observed": True, "propensity": 0.5},
        )
        for _ in range(4)
    ]
    result = run(
        "fit",
        {
            "family": "learnable_constant",
            "train_data": train_trees,
            "axis": {"max_iterations": 1, "axis_value": 0},
            "initial_artifacts": {"f": 0.0, "g": None},
            "backend_config": {"output_dir": str(tmp_path)},
        },
    )
    assert isinstance(result, CTreePOFitResult)
    assert result.status == "success"


# --------------------------------------------------------------------------- #
# 5. audit — post-hoc, no fit() call
# --------------------------------------------------------------------------- #


def test_audit_method_returns_objective_and_overlap() -> None:
    rows = [
        LocalLawAuditRow(
            row_id=f"r{i}",
            law_kind="c1_leaf",
            proxy_loss=0.0,
            oracle_loss=0.0,
            observed=True,
            propensity=0.5,
            node_weight=1.0,
        )
        for i in range(3)
    ]
    result = run("audit", {"rows": rows, "objective_mode": "corrected_local_law"})
    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert result["method"] == "audit"
    assert result["n_rows"] == 3
    assert result["influence_weighted_overlap"]["n_observed"] == 3


def test_audit_method_writes_optional_sidecar(tmp_path: Path) -> None:
    rows = [
        LocalLawAuditRow(
            row_id="r1",
            law_kind="c1",
            proxy_loss=0.0,
            oracle_loss=0.0,
            observed=True,
            propensity=0.5,
            node_weight=1.0,
        )
    ]
    result = run("audit", {"rows": rows, "output_dir": str(tmp_path)})
    assert (tmp_path / "audit_summary.json").exists()
    assert result["status"] == "success"


def test_audit_method_rejects_missing_rows() -> None:
    with pytest.raises(ValueError, match="rows"):
        run("audit", {})


# --------------------------------------------------------------------------- #
# 6. The smoke iterator — every method runs with a minimum config
# --------------------------------------------------------------------------- #


def test_smoke_every_method_runs(tmp_path: Path) -> None:
    """If a new method is added without a working handler, this test
    fails. The minimum configs reflect the axis-factored shape:
    one entry per method, parameterized by config (not method name).
    """
    from treepo.cld.fixtures import make_hll_token_trees

    minimum_configs: dict[str, dict] = {
        "fit": {
            "family": "oracle",
            "eval_data": make_hll_token_trees(n_trees=2, seed=10),
            "backend_config": {
                "oracle_name": "hll_exact",
                "output_dir": str(tmp_path / "fit"),
            },
        },
        "oracle": {
            "oracle_name": "hll_exact",
            "n_trees": 2,
            "output_dir": str(tmp_path / "oracle"),
        },
        "sketch": {
            "sketch_kind": "hll",
            "precision": 10,
            "n_trees": 2,
            "output_dir": str(tmp_path / "sketch"),
        },
        "audit": {
            "rows": [
                LocalLawAuditRow(
                    row_id="r0",
                    law_kind="c1",
                    proxy_loss=0.0,
                    oracle_loss=0.0,
                    observed=True,
                    propensity=0.5,
                    node_weight=1.0,
                )
            ],
            "output_dir": str(tmp_path / "audit"),
        },
    }
    # 'probe' subprocesses the paper script and needs CUDA; it's covered
    # by treepo.cld/tests/integration/test_probe_clean_unified_no_live.py.
    smoke_methods = _EXPECTED_METHODS - {"probe"}
    assert set(minimum_configs) == smoke_methods, (
        "this test must enumerate every smoke-runnable method; missing: "
        f"{smoke_methods - set(minimum_configs)}"
    )
    for method in smoke_methods:
        result = run(method, minimum_configs[method])
        if isinstance(result, CTreePOFitResult):
            assert result.status == "success", f"method {method!r} did not succeed"
        else:
            assert isinstance(result, dict)
            assert result.get("status") == "success", f"method {method!r} did not succeed"
