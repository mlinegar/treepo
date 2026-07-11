"""Tests for the task-agnostic labeled-tree bundle loader."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from treepo.bundles import BundleFormatError, load_labeled_tree_bundle
from treepo.tree import TreeRecord

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "labeled_bundle"
FIXTURE_FILE = FIXTURE_DIR / "labeled_trees.jsonl"


def test_load_returns_tree_records_with_reconstructed_topology() -> None:
    records = load_labeled_tree_bundle(FIXTURE_DIR)
    assert [r.doc_id for r in records] == ["doc_a", "doc_b"]
    assert all(isinstance(r, TreeRecord) for r in records)

    record = records[0]
    assert record.root_label == pytest.approx(0.5)
    root = record.root()
    assert root is not None
    assert root.unit_type == "root"
    assert root.label == pytest.approx(0.5)
    assert root.left_child_id == "node_l0_00000"
    assert root.right_child_id == "node_l0_00001"

    leaves = record.leaves()
    assert len(leaves) == 2
    # Parent edges are reconstructed from child pointers.
    assert {leaf.parent_id for leaf in leaves} == {"node_l1_00000"}
    # Leaf order follows the bundle `levels` ordering.
    assert [leaf.node_id for leaf in leaves] == ["node_l0_00000", "node_l0_00001"]
    assert leaves[0].label == pytest.approx(0.75)


def test_load_accepts_jsonl_file_directly() -> None:
    from_dir = load_labeled_tree_bundle(FIXTURE_DIR)
    from_file = load_labeled_tree_bundle(FIXTURE_FILE)
    assert [r.doc_id for r in from_dir] == [r.doc_id for r in from_file]


def test_node_payload_is_preserved_losslessly() -> None:
    record = load_labeled_tree_bundle(FIXTURE_DIR)[0]
    leaf = record.get_node("node_l0_00000")
    assert leaf is not None
    meta = dict(leaf.metadata)
    # Span, cmp counts, qsentence totals, and index ranges all survive.
    assert meta["char_start"] == 0
    assert meta["char_end"] == 20
    assert meta["cmp_counts"] == {"402": 2}
    assert meta["total_non_header_qsentences"] == 2
    assert meta["sentence_start_index"] == 0
    assert meta["sentence_end_index"] == 1
    assert meta["qsentence_start_index"] == 0
    assert meta["qsentence_end_index"] == 2
    assert meta["g_training_role"] == "leaf"
    # Node-level typed fields are lifted into metadata so nothing is dropped.
    assert meta["dimension_scores"]["domain_2"] == pytest.approx(1.0)
    assert meta["confidence"] == pytest.approx(1.0)

    merge = record.get_node("node_l1_00000")
    assert merge is not None
    assert merge.unit_type == "root"  # the top merge node is the root
    assert dict(merge.metadata)["g_training_role"] == "merge"


def test_tree_metadata_records_schema_version_and_tolerates_unknown_fields() -> None:
    record = load_labeled_tree_bundle(FIXTURE_DIR)[0]
    meta = dict(record.metadata)
    assert meta["schema_version"] == "3.0"
    assert meta["label_source"] == "synthetic_fixture_v1"
    # An unknown extra field in the source is carried through, never rejected.
    assert meta["future_unknown_field"] == {"tolerated": True}


def test_split_filter_uses_pinned_split_ids() -> None:
    train = load_labeled_tree_bundle(FIXTURE_DIR, split="train")
    test = load_labeled_tree_bundle(FIXTURE_DIR, split="test")
    val = load_labeled_tree_bundle(FIXTURE_DIR, split="val")
    assert [r.doc_id for r in train] == ["doc_a"]
    assert [r.doc_id for r in test] == ["doc_b"]
    assert val == []


def test_pinned_split_ids_override_per_tree_metadata(tmp_path: Path) -> None:
    """split_ids.json is authoritative; per-tree metadata is never resampled over it."""

    rows = [json.loads(line) for line in FIXTURE_FILE.read_text(encoding="utf-8").splitlines()]
    # doc_a says split="train" in its own metadata; pin it to test instead.
    (tmp_path / "labeled_trees.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    (tmp_path / "split_ids.json").write_text(
        json.dumps({"train": ["doc_b"], "val": [], "test": ["doc_a"]}), encoding="utf-8"
    )
    train = load_labeled_tree_bundle(tmp_path, split="train")
    test = load_labeled_tree_bundle(tmp_path, split="test")
    assert [r.doc_id for r in train] == ["doc_b"]
    assert [r.doc_id for r in test] == ["doc_a"]


def test_split_falls_back_to_metadata_when_no_split_ids(tmp_path: Path) -> None:
    rows = [json.loads(line) for line in FIXTURE_FILE.read_text(encoding="utf-8").splitlines()]
    (tmp_path / "labeled_trees.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    # No split_ids.json here: loader uses per-tree metadata["split"].
    train = load_labeled_tree_bundle(tmp_path, split="train")
    assert [r.doc_id for r in train] == ["doc_a"]


def test_dimension_selects_target_scores() -> None:
    default = load_labeled_tree_bundle(FIXTURE_DIR)
    d1 = load_labeled_tree_bundle(FIXTURE_DIR, dimension="domain_1")
    # Root label switches to the requested dimension.
    assert default[0].root_label == pytest.approx(0.5)
    assert d1[0].root_label == pytest.approx(0.5)
    # Leaf labels differ per dimension.
    leaf0_default = default[0].get_node("node_l0_00000").label
    leaf0_domain1 = d1[0].get_node("node_l0_00000").label
    assert leaf0_default == pytest.approx(0.75)
    assert leaf0_domain1 == pytest.approx(0.0)


def test_unknown_dimension_raises() -> None:
    with pytest.raises(BundleFormatError, match="dimension"):
        load_labeled_tree_bundle(FIXTURE_DIR, dimension="not_a_dimension")


def test_missing_required_node_field_raises(tmp_path: Path) -> None:
    row = json.loads(FIXTURE_FILE.read_text(encoding="utf-8").splitlines()[0])
    # Drop a required node field.
    del row["nodes"]["node_l0_00000"]["node_id"]
    bad = tmp_path / "labeled_trees.jsonl"
    bad.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(BundleFormatError, match="missing required field 'node_id'"):
        load_labeled_tree_bundle(bad)


def test_missing_required_tree_field_raises(tmp_path: Path) -> None:
    row = json.loads(FIXTURE_FILE.read_text(encoding="utf-8").splitlines()[0])
    del row["nodes"]
    bad = tmp_path / "labeled_trees.jsonl"
    bad.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(BundleFormatError, match="missing required field 'nodes'"):
        load_labeled_tree_bundle(bad)


def test_non_json_line_raises(tmp_path: Path) -> None:
    bad = tmp_path / "labeled_trees.jsonl"
    bad.write_text("{not valid json}\n", encoding="utf-8")
    with pytest.raises(BundleFormatError, match="not valid JSON"):
        load_labeled_tree_bundle(bad)


def test_unknown_split_raises() -> None:
    with pytest.raises(BundleFormatError, match="split"):
        load_labeled_tree_bundle(FIXTURE_DIR, split="holdout")


def test_missing_path_raises() -> None:
    with pytest.raises(BundleFormatError, match="does not exist"):
        load_labeled_tree_bundle(FIXTURE_DIR.parent / "no_such_bundle")


# Optional smoke test against a real ThinkingTrees bundle. Skipped unless
# TREEPO_TT_BUNDLE points at a bundle directory or labeled_trees.jsonl. The
# package never hardcodes an absolute path (release hygiene forbids it).
_TT_BUNDLE = os.environ.get("TREEPO_TT_BUNDLE")


@pytest.mark.skipif(not _TT_BUNDLE, reason="set TREEPO_TT_BUNDLE to a real bundle to run")
def test_real_bundle_round_trips_node_targets() -> None:
    records = load_labeled_tree_bundle(_TT_BUNDLE)
    assert records, "real bundle produced no trees"
    record = records[0]
    assert record.root() is not None
    assert record.leaves()
    leaf = record.leaves()[0]
    assert "cmp_counts" in dict(leaf.metadata)
    assert "dimension_scores" in dict(leaf.metadata)
    # A pinned split partitions without overlap when split_ids is present.
    train = load_labeled_tree_bundle(_TT_BUNDLE, split="train")
    test = load_labeled_tree_bundle(_TT_BUNDLE, split="test")
    train_ids = {r.doc_id for r in train}
    test_ids = {r.doc_id for r in test}
    assert train_ids.isdisjoint(test_ids)
