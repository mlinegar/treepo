"""Tests for the standalone tree visualization export."""

from __future__ import annotations

from pathlib import Path

import pytest

from treepo.tree import TreeRecord, local_law_rows_from_tree_records
from treepo.viz import tree_visualization_payload, write_tree_visualization_html


def _tree(idx: int) -> TreeRecord:
    return TreeRecord(
        tree_id=f"doc_{idx}",
        root_label=1.5,
        nodes=[
            {
                "node_id": "q1",
                "text": "first qsentence",
                "parent_id": "root",
                "level": 0,
                "position": 0,
                "label": 2.0,
                "state": {"kind": "policy", "text": "pro-market stance on trade"},
                "metadata": {
                    "llm_score": 1.8,
                    "proxy_loss": 0.04,
                    "llm_summary": "supports free trade",
                },
            },
            {
                "node_id": "q2",
                "text": "second qsentence",
                "parent_id": "root",
                "level": 0,
                "position": 1,
                "label": 1.0,
                "metadata": {"proxy_loss": 0.09},
            },
            {"node_id": "root", "unit_type": "root", "level": 1, "label": 1.5},
        ],
    )


def _sampling_rows(idx: int) -> list[dict[str, object]]:
    return [
        {
            "tree_id": f"doc_{idx}",
            "node_id": "q1",
            "observed": True,
            "joint_propensity": 0.5,
            "ipw_weight": 2.0,
        },
        {
            "tree_id": f"doc_{idx}",
            "node_id": "q2",
            "observed": False,
            "joint_propensity": 0.5,
        },
    ]


def test_payload_attaches_sampling_labels_and_laws() -> None:
    tree = _tree(0)
    law_rows = local_law_rows_from_tree_records([tree])
    payload = tree_visualization_payload(
        tree, sampling_rows=_sampling_rows(0), law_rows=law_rows
    )
    assert payload["tree_id"] == "doc_0"
    assert payload["n_leaves"] == 2
    assert payload["n_sampled_nodes"] == 1
    root = payload["roots"][0]
    assert root["node_id"] == "root"
    children = {child["node_id"]: child for child in root["children"]}
    assert list(children) == ["q1", "q2"]
    assert children["q1"]["sampling"]["observed"] is True
    assert children["q1"]["sampling"]["ipw_weight"] == 2.0
    assert children["q2"]["sampling"]["observed"] is False
    assert children["q1"]["labels"] == {"llm_score": 1.8}
    assert children["q1"]["label"] == 2.0
    assert children["q1"]["summaries"] == {
        "llm_summary": "supports free trade",
        "state": "pro-market stance on trade",
    }
    assert children["q1"]["laws"][0]["proxy_loss"] == 0.04
    # Depth follows the root-at-zero convention.
    assert children["q1"]["laws"][0]["depth"] == 1


def test_trace_law_rows_synthesize_merge_nodes() -> None:
    tree = TreeRecord(
        tree_id="doc_t",
        nodes=[
            {"node_id": "a", "text": "A", "parent_id": "root", "level": 0, "position": 0},
            {"node_id": "b", "text": "B", "parent_id": "root", "level": 0, "position": 1},
            {"node_id": "c", "text": "C", "parent_id": "root", "level": 0, "position": 2},
            {"node_id": "root", "unit_type": "root", "level": 1},
        ],
    )
    # Trace order for 3 leaves: a, b, c, merge(a,b), root state.
    law_rows = [
        {
            "row_id": f"doc_t:state:{idx}",
            "law_kind": "merge_preservation" if idx >= 3 else "leaf_preservation",
            "proxy_loss": 0.1 * (idx + 1),
            "oracle_loss": 0.1 * (idx + 1),
            "observed": True,
            "propensity": 1.0,
            "depth": [2, 2, 1, 1, 0][idx],
        }
        for idx in range(5)
    ]
    payload = tree_visualization_payload(tree, law_rows=law_rows)
    root = payload["roots"][0]
    child_ids = [child["node_id"] for child in root["children"]]
    assert child_ids == ["merge_3", "c"]
    merge = root["children"][0]
    assert [grand["node_id"] for grand in merge["children"]] == ["a", "b"]
    assert merge["laws"][0]["proxy_loss"] == pytest.approx(0.4)
    assert root["laws"][0]["proxy_loss"] == pytest.approx(0.5)
    assert root["children"][1]["laws"][0]["proxy_loss"] == pytest.approx(0.3)


def test_write_html_is_standalone_and_contains_nodes(tmp_path: Path) -> None:
    trees = [_tree(0), _tree(1)]
    rows = _sampling_rows(0) + _sampling_rows(1)
    out = write_tree_visualization_html(
        trees, tmp_path / "trees.html", sampling_rows=rows, title="test trees"
    )
    text = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in text
    assert "doc_0" in text and "doc_1" in text
    assert "first qsentence" in text
    assert "test trees" in text
    # Single file: no external script or stylesheet references.
    assert "src=" not in text and "href=" not in text
