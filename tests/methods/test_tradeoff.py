"""Tests for the compression-tradeoff curve record."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from treepo.methods.tradeoff import TradeoffCurve


def _rows() -> list[dict]:
    return [
        {"leaf_unit_count": 16, "root_mae": 1.5, "average_guess_mae": 2.0, "extra": "x"},
        {"leaf_unit_count": 8, "root_mae": 1.2, "average_guess_mae": 2.0},
        {"leaf_unit_count": 48, "root_mae": None, "average_guess_mae": "2.5"},
    ]


def test_from_rows_sorts_and_coerces() -> None:
    curve = TradeoffCurve.from_rows(
        _rows(), metric_keys=("root_mae", "average_guess_mae")
    )
    assert curve.axis_kind == "leaf_unit_count"
    assert [point["axis_value"] for point in curve.points] == [8.0, 16.0, 48.0]
    assert curve.points[0]["metrics"] == {"root_mae": 1.2, "average_guess_mae": 2.0}
    # Missing metrics stay None; string numbers coerce.
    assert curve.points[2]["metrics"] == {"root_mae": None, "average_guess_mae": 2.5}
    flat = curve.rows()
    assert flat[0] == {"leaf_unit_count": 8.0, "root_mae": 1.2, "average_guess_mae": 2.0}


def test_curve_validation() -> None:
    with pytest.raises(ValueError):
        TradeoffCurve.from_rows(_rows(), metric_keys=())
    with pytest.raises(ValueError):
        TradeoffCurve.from_rows(
            [{"leaf_unit_count": 8, "m": 1.0}, {"leaf_unit_count": 8, "m": 2.0}],
            metric_keys=("m",),
        )
    with pytest.raises(ValueError):
        TradeoffCurve.from_rows([{"m": 1.0}], metric_keys=("m",))


def test_write_outputs(tmp_path: Path) -> None:
    curve = TradeoffCurve.from_rows(
        _rows(), metric_keys=("root_mae",), metadata={"task": "markov"}
    )
    json_out = tmp_path / "curve.json"
    csv_out = tmp_path / "curve.csv"
    curve.write(json_out=json_out, csv_out=csv_out)
    payload = json.loads(json_out.read_text())
    assert payload["axis_kind"] == "leaf_unit_count"
    assert payload["metadata"] == {"task": "markov"}
    assert len(payload["points"]) == 3
    with csv_out.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["leaf_unit_count"] for row in rows] == ["8.0", "16.0", "48.0"]


def test_lazy_export() -> None:
    from treepo.methods import TradeoffCurve as exported

    assert exported is TradeoffCurve
