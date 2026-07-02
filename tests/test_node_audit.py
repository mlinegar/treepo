"""Tests for uniform node-audit designs and their application to audit rows."""

from __future__ import annotations

import pytest

from treepo.local_law import LocalLawAuditRow, audit_local_laws
from treepo.sampling import apply_node_audit, sample_node_audit


def _full_rows(n: int) -> list[LocalLawAuditRow]:
    return [
        LocalLawAuditRow(
            row_id=f"t:state:{idx}",
            law_kind="merge_preservation",
            proxy_loss=0.1 * idx,
            oracle_loss=0.2 * idx,
            observed=True,
            propensity=1.0,
            depth=idx % 3,
        )
        for idx in range(n)
    ]


def test_sample_node_audit_policies() -> None:
    everything = sample_node_audit(5, policy="all")
    assert everything.sample_size == 5 and everything.propensity == 1.0
    assert all(everything.observed)

    sqrt_design = sample_node_audit(5, policy="sqrt", seed=1)
    assert sqrt_design.sample_size == 3
    assert sqrt_design.propensity == pytest.approx(0.6)
    assert sum(sqrt_design.observed) == 3

    half = sample_node_audit(4, policy="fraction", fraction=0.5, seed=2)
    assert half.sample_size == 2 and half.propensity == pytest.approx(0.5)

    assert sample_node_audit(5, policy="sqrt", seed=1) == sqrt_design  # deterministic

    with pytest.raises(ValueError):
        sample_node_audit(0)
    with pytest.raises(ValueError):
        sample_node_audit(5, policy="fixed", fixed_nodes=0)


def test_apply_node_audit_realizes_design() -> None:
    rows = _full_rows(5)
    design = sample_node_audit(5, policy="sqrt", seed=3)
    audited = apply_node_audit(rows, design)
    assert len(audited) == 5
    for row, observed in zip(audited, design.observed):
        assert row.observed is bool(observed)
        assert row.propensity == pytest.approx(design.propensity)
        assert row.metadata["audit_policy"] == "sqrt"
        if observed:
            assert row.oracle_loss is not None
        else:
            assert row.oracle_loss is None
    # Audited rows feed the AIPW summary directly.
    summary = audit_local_laws(list(audited), gamma_depth=0.9)
    assert summary["local_law_objective"]["observed_count"] == design.sample_size
    assert summary["local_law_objective"]["row_count"] == 5


def test_apply_node_audit_validates_inputs() -> None:
    rows = _full_rows(4)
    design = sample_node_audit(5, policy="all")
    with pytest.raises(ValueError):
        apply_node_audit(rows, design)

    proxy_only = [
        LocalLawAuditRow(
            row_id="t:state:0",
            law_kind="leaf_preservation",
            proxy_loss=0.1,
            observed=False,
            propensity=1.0,
        )
    ]
    with pytest.raises(ValueError):
        apply_node_audit(proxy_only, sample_node_audit(1, policy="all"))
