"""Phase 6 of the fit-grid plan: the per-cell ``results.json`` + cost schema.

Every ``fit()`` cell writes one results artifact with per-split/per-dimension
metrics (never pooled), the paired normalized L1 next to Pearson r, the
theta-regime/contextual-MAE pairing visible even when null, three separate
cost components, and resummary ops recorded even when zero.
"""

from __future__ import annotations

import json
from pathlib import Path

from treepo import fit
from treepo.methods._results import PAIRED_ROW_FIELDS, RESULTS_FILENAME
from treepo.tree import TreeNode, TreeRecord


def _tree(doc_id: str, *, offset: float, split: str) -> TreeRecord:
    leaves = [
        TreeNode(
            node_id=f"{doc_id}_l{i}",
            unit_type="leaf",
            text=f"{doc_id} unit {i} alpha beta gamma delta",
            level=0,
            position=i,
            label=0.1 * (i + 1) + offset,
        )
        for i in range(4)
    ]
    return TreeRecord(
        tree_id=doc_id,
        doc_id=doc_id,
        root_label=0.25 + offset,
        nodes=tuple(leaves),
        metadata={"split": split, "expert_score": 0.3 + offset},
    )


def _trees(n: int = 8) -> list[TreeRecord]:
    return [
        _tree(f"doc_{i:02d}", offset=0.05 * i, split="train" if i % 2 == 0 else "val")
        for i in range(n)
    ]


def _run_fit(tmp_path: Path, **spec_extra: object):
    trees = _trees()
    return fit(
        {
            "family": "neural_operator",
            "train_data": trees,
            "eval_data": trees,
            "axis": {"axis_kind": "leaf_count", "axis_value": 4},
            "seed": 11,
            "backend_config": {
                "operator_kind": "conv1d",
                "embedding_dim": 8,
                "hidden_channels": 4,
                "n_layers": 1,
                "head_hidden_dim": 8,
                "epochs_per_iteration": 1,
                "batch_size": 4,
                "learning_rate": 0.01,
                "device": "cpu",
                "output_dir": str(tmp_path),
            },
            **spec_extra,
        }
    )


def _load_results(tmp_path: Path) -> dict:
    path = tmp_path / RESULTS_FILENAME
    assert path.is_file(), "fit() must write results.json next to the manifest"
    return json.loads(path.read_text(encoding="utf-8"))


def test_results_json_written_and_linked(tmp_path: Path) -> None:
    result = _run_fit(tmp_path)
    payload = _load_results(tmp_path)
    assert payload["version"] == "0.1"
    assert payload["status"] == "success"
    assert result.summary["results_path"] == str(tmp_path / RESULTS_FILENAME)
    assert result.artifacts["results_json"] == str(tmp_path / RESULTS_FILENAME)


def test_results_cell_block_pins_the_grid_cell(tmp_path: Path) -> None:
    _run_fit(tmp_path, supervision_level="leaf")
    payload = _load_results(tmp_path)
    cell = payload["cell"]
    assert cell["family"] == "neural_operator"
    assert cell["seed"] == 11
    assert cell["supervision"]["level"] == "leaf"
    assert cell["grid_axes"]["local_label_mix"]["mix"] == "none"
    assert cell["axis"]["axis_kind"] == "leaf_count"


def test_results_metrics_are_per_split_and_never_pooled(tmp_path: Path) -> None:
    _run_fit(tmp_path)
    payload = _load_results(tmp_path)
    metrics = payload["metrics"]
    assert metrics["pooled_across_dimensions"] is False
    splits = metrics["splits"]
    assert {"all", "train", "val"} <= set(splits)
    for split_name in ("train", "val"):
        block = splits[split_name]
        external = block["external"]
        # Normalized L1 sits NEXT TO pearson, with bounds provenance.
        assert "pearson_r" in external and "normalized_abs_error" in external
        if external["normalized_abs_error"] is not None:
            assert external["scale_bounds_source"] == "observed_gold_range"
            lo, hi = external["scale_bounds"]
            assert hi > lo
        # The theta/contextual pairing is present (null: no sim channel here).
        assert block["sim"] == {
            "theta_first_regime_accuracy": None,
            "theta_last_regime_accuracy": None,
            "contextual_mae": None,
        }


def test_results_cost_components_are_separate_and_c2_stays_visible(tmp_path: Path) -> None:
    _run_fit(tmp_path, supervision_level="leaf")
    payload = _load_results(tmp_path)
    cost = payload["cost"]
    assert set(cost) == {"label_cost", "one_time_compute", "marginal_inference", "resummary_ops"}
    label = cost["label_cost"]
    assert label["node_label_source"] == "none"
    assert label["gold_node_labels_consumed"] > 0  # leaf supervision consumed gold rows
    assert label["distilled_node_labels_consumed"] == 0
    compute = cost["one_time_compute"]
    assert compute["fit_wall_seconds"] is not None and compute["fit_wall_seconds"] > 0
    assert cost["marginal_inference"]["n_eval_predictions"] == 8
    # The C2 stratum: zero resummary ops is recorded, never omitted.
    assert cost["resummary_ops"] == {"count": 0, "population": "empty_by_construction"}


def test_results_distilled_cell_reports_distilled_label_cost(tmp_path: Path) -> None:
    trees = _trees()
    rows_path = tmp_path / "teacher_node_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for i, tree in enumerate(trees):
            for node in tree.nodes:
                row = {
                    "doc_id": tree.doc_id,
                    "node_id": node.node_id,
                    "score_1_7": 1.0 + (i % 6) + 0.1 * (node.position or 0),
                }
                handle.write(json.dumps(row) + "\n")
    fit(
        {
            "family": "neural_operator",
            "train_data": trees,
            "eval_data": trees,
            "axis": {"axis_kind": "leaf_count", "axis_value": 4},
            "local_label_mix": "llm_distilled",
            "distilled_labels_path": str(rows_path),
            "supervision_level": "leaf",
            "backend_config": {
                "operator_kind": "conv1d",
                "embedding_dim": 8,
                "hidden_channels": 4,
                "n_layers": 1,
                "head_hidden_dim": 8,
                "epochs_per_iteration": 1,
                "batch_size": 4,
                "learning_rate": 0.01,
                "device": "cpu",
                "output_dir": str(tmp_path / "fit"),
            },
        }
    )
    payload = json.loads((tmp_path / "fit" / RESULTS_FILENAME).read_text(encoding="utf-8"))
    label = payload["cost"]["label_cost"]
    assert label["node_label_source"] == "llm_distilled"
    assert label["gold_node_labels_consumed"] == 0
    assert label["distilled_node_labels_consumed"] > 0


def test_results_paired_rows_point_at_prediction_files(tmp_path: Path) -> None:
    _run_fit(tmp_path)
    payload = _load_results(tmp_path)
    paired = payload["paired_rows"]
    assert paired["fields"] == PAIRED_ROW_FIELDS
    assert paired["files"], "paired per-document rows must be discoverable"
    first = Path(paired["files"][0])
    assert first.is_file()
    row = json.loads(first.read_text(encoding="utf-8").splitlines()[0])
    # The W-ledger ingest contract: key, prediction, gold, split all present.
    for field in PAIRED_ROW_FIELDS.values():
        assert field in row
