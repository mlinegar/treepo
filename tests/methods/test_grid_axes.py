"""First-class supervision-grid axes: validation, pinned selection, provenance.

Covers Phase 3 of ``docs/treepo_fit_grid_upgrade_plan_2026_07_10.md``: the
``doc_gold_n`` / ``local_label_mix`` / ``seed`` spec axes, their deterministic
pinned selection, the grid-cell expander, and the provenance that ``fit()``
persists so every cell at the same ``(seed, n)`` trains on the identical,
nested subset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from treepo import fit
from treepo.methods._grid_axes import (
    GridAxes,
    apply_grid_axes,
    expand_grid_cells,
    resolve_label_mix,
    select_gold_docs,
)
from treepo.methods.fixtures import make_markov_changepoint_trees


@dataclass
class _Leaf:
    qid: str


@dataclass
class _Tree:
    doc_id: str
    leaves: tuple[_Leaf, ...] = field(default_factory=tuple)


def _docs(n: int, leaves_per: int = 3) -> list[_Tree]:
    return [
        _Tree(doc_id=f"d{i}", leaves=tuple(_Leaf(qid=f"q{i}_{j}") for j in range(leaves_per)))
        for i in range(n)
    ]


def _tiny_fno_config() -> dict[str, object]:
    return {
        "embedding_dim": 8,
        "hidden_channels": 4,
        "n_modes": 2,
        "n_layers": 1,
        "head_hidden_dim": 8,
        "epochs_per_iteration": 1,
        "batch_size": 4,
        "learning_rate": 0.01,
        "device": "cpu",
        "seed": 3,
    }


# --- axis validation -------------------------------------------------------


def test_grid_axes_validate_enum_and_ranges() -> None:
    with pytest.raises(ValueError):
        GridAxes(local_label_mix="bogus")
    with pytest.raises(ValueError):
        GridAxes(gold_fraction_p=1.5)
    with pytest.raises(ValueError):
        GridAxes(doc_gold_n=-1)
    ok = GridAxes(doc_gold_n=25, local_label_mix="gold_fraction", gold_fraction_p=0.5, seed=7)
    assert ok.doc_gold_n == 25 and ok.seed == 7


def test_grid_axes_from_spec_reads_first_class_fields() -> None:
    from treepo.methods.contracts import CTreePOLearningSpec

    spec = CTreePOLearningSpec.from_mapping(
        {
            "family": "fno",
            "doc_gold_n": 50,
            "local_label_mix": "gold_fraction",
            "gold_fraction_p": 0.25,
            "seed": 11,
        }
    )
    axes = GridAxes.from_spec(spec)
    assert (axes.doc_gold_n, axes.local_label_mix, axes.gold_fraction_p, axes.seed) == (
        50,
        "gold_fraction",
        0.25,
        11,
    )
    # Round-trips through to_dict for the manifest.
    assert spec.to_dict()["local_label_mix"] == "gold_fraction"


# --- doc_gold_n: pinned + nested -------------------------------------------


def test_doc_gold_n_is_pinned_and_nested_by_prefix() -> None:
    docs = _docs(20)
    sel25, prov25 = select_gold_docs(docs, doc_gold_n=5, seed=13)
    sel50, prov50 = select_gold_docs(docs, doc_gold_n=10, seed=13)
    sel100, prov100 = select_gold_docs(docs, doc_gold_n=20, seed=13)

    ids25 = set(prov25["selected_doc_ids"])
    ids50 = set(prov50["selected_doc_ids"])
    ids100 = set(prov100["selected_doc_ids"])
    assert ids25 < ids50 < ids100  # strictly nested
    assert len(sel25) == 5 and len(sel50) == 10
    # Same seed reselects the identical subset; a different seed differs.
    again, _ = select_gold_docs(docs, doc_gold_n=5, seed=13)
    other, _ = select_gold_docs(docs, doc_gold_n=5, seed=14)
    assert [t.doc_id for t in again] == [t.doc_id for t in sel25]
    assert set(t.doc_id for t in other) != ids25
    assert prov25["nested_prefix"] is True


def test_doc_gold_n_none_keeps_full_population() -> None:
    docs = _docs(8)
    selected, provenance = select_gold_docs(docs, doc_gold_n=None, seed=0)
    assert len(selected) == 8
    assert provenance["doc_gold_n"] is None
    assert provenance["nested_prefix"] is False
    assert provenance["selected_count"] == 8


# --- local_label_mix -------------------------------------------------------


def test_gold_fraction_selection_is_deterministic_and_pinned() -> None:
    docs = _docs(10, leaves_per=4)  # 40 nodes
    axes = GridAxes(local_label_mix="gold_fraction", gold_fraction_p=0.5, seed=5)
    a = resolve_label_mix(docs, axes=axes, backend_config={})
    b = resolve_label_mix(docs, axes=axes, backend_config={})
    assert a["selected_node_units"] == b["selected_node_units"]
    assert a["population_node_count"] == 40
    assert a["selected_node_count"] == 20
    assert a["state_source"] == "f_states"  # gold-leak guard

    none_axes = GridAxes(local_label_mix="gold_fraction", gold_fraction_p=0.0, seed=5)
    all_axes = GridAxes(local_label_mix="gold_fraction", gold_fraction_p=1.0, seed=5)
    assert resolve_label_mix(docs, axes=none_axes, backend_config={})["selected_node_count"] == 0
    assert resolve_label_mix(docs, axes=all_axes, backend_config={})["selected_node_count"] == 40


def test_none_mix_is_root_only() -> None:
    prov = resolve_label_mix(_docs(3), axes=GridAxes(local_label_mix="none"), backend_config={})
    assert prov["node_labels"] == "root_only"


def test_llm_distilled_requires_a_configured_source() -> None:
    axes = GridAxes(local_label_mix="llm_distilled")
    with pytest.raises(ValueError, match="cached-teacher"):
        resolve_label_mix(_docs(2), axes=axes, backend_config={})

    # A callable node predictor resolves it.
    def _predict(**_kwargs: Any) -> dict[str, float]:
        return {"score": 0.0}

    via_predictor = resolve_label_mix(
        _docs(2), axes=axes, backend_config={"node_oracle_predictor": _predict}
    )
    assert via_predictor["labels_source"] == "node_oracle_predictor"

    # An explicit cached-labels path resolves it too.
    via_path = resolve_label_mix(
        _docs(2),
        axes=GridAxes(local_label_mix="llm_distilled", distilled_labels_path="teacher_node_rows.jsonl"),
        backend_config={},
    )
    assert via_path["labels_source"] == "cached_jsonl"


# --- grid expansion --------------------------------------------------------


def test_expand_grid_cells_counts_and_gold_fraction_tagging() -> None:
    cells = expand_grid_cells(
        seeds=[0, 1, 2],
        doc_gold_ns=[None, 25, 50],
        local_label_mixes=["none", "gold_fraction"],
        leaf_unit_counts=[1, 8],
        gold_fraction_p=0.3,
    )
    assert len(cells) == 3 * 3 * 2 * 2
    gold_cells = [c for c in cells if c["local_label_mix"] == "gold_fraction"]
    assert gold_cells and all(c["gold_fraction_p"] == pytest.approx(0.3) for c in gold_cells)
    assert all("gold_fraction_p" not in c for c in cells if c["local_label_mix"] == "none")

    with pytest.raises(ValueError):
        expand_grid_cells(local_label_mixes=["bad_mix"])


def test_apply_grid_axes_returns_selected_and_provenance() -> None:
    docs = _docs(12)
    axes = GridAxes(doc_gold_n=4, local_label_mix="gold_fraction", gold_fraction_p=0.5, seed=2)
    selected, provenance = apply_grid_axes(docs, axes=axes, backend_config={})
    assert len(selected) == 4
    assert provenance["seed"] == 2
    assert provenance["doc_gold"]["selected_count"] == 4
    assert provenance["local_label_mix"]["mix"] == "gold_fraction"


# --- end-to-end provenance through fit() -----------------------------------


def _markov_trees(n: int, *, seed: int, split: str) -> list[Any]:
    return make_markov_changepoint_trees(
        n_trees=n,
        doc_tokens=32,
        leaf_unit_count=8,
        vocabulary_size=64,
        seed=seed,
        split=split,
    )


def test_fit_persists_grid_axes_provenance(tmp_path: Path) -> None:
    result = fit(
        {
            "family": "fno",
            "train_data": _markov_trees(6, seed=71, split="train"),
            "eval_data": _markov_trees(3, seed=72, split="test"),
            "backend_config": {**_tiny_fno_config(), "output_dir": str(tmp_path)},
            "axis": {"max_iterations": 2, "axis_value": 0},
            "doc_gold_n": 3,
            "seed": 9,
        }
    )
    assert result.status == "success"

    grid_axes = result.summary["grid_axes"]
    assert grid_axes["seed"] == 9
    doc_gold = grid_axes["doc_gold"]
    assert doc_gold["doc_gold_n"] == 3
    assert doc_gold["selected_count"] == 3
    assert len(doc_gold["selected_doc_ids"]) == 3
    assert doc_gold["nested_prefix"] is True

    evidence = result.artifacts["evidence"]
    assert evidence["grid_axes"]["present"] is True
    assert evidence["grid_axes"]["doc_gold"]["selected_doc_ids"] == doc_gold["selected_doc_ids"]

    # The pinned subset re-runs identically for the same (seed, n).
    again = fit(
        {
            "family": "fno",
            "train_data": _markov_trees(6, seed=71, split="train"),
            "eval_data": _markov_trees(3, seed=72, split="test"),
            "backend_config": {**_tiny_fno_config(), "output_dir": str(tmp_path / "again")},
            "axis": {"max_iterations": 2, "axis_value": 0},
            "doc_gold_n": 3,
            "seed": 9,
        }
    )
    assert (
        again.summary["grid_axes"]["doc_gold"]["selected_doc_ids"]
        == doc_gold["selected_doc_ids"]
    )


# --- example grid helper crosses the seeds axis ----------------------------


def _load_example_setup() -> Any:
    import sys

    examples_dir = Path(__file__).resolve().parents[2] / "examples" / "methods"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    import example_setup  # type: ignore

    return example_setup


def test_manifesto_grid_cells_crosses_seeds_and_axes() -> None:
    example_setup = _load_example_setup()
    config = example_setup.ManifestoReplicationConfig(
        supervision_grid=("none",),
        leaf_unit_counts=(1, 8),
        seeds=(0, 1, 2),
        doc_gold_ns=(None, 25),
        local_label_mixes=("none", "gold_fraction"),
        gold_fraction_p=0.5,
    )
    cells = example_setup.manifesto_grid_cells(config)
    # 3 seeds x 2 doc_gold_ns x 2 mixes x 2 leaf counts (root supervision).
    assert len(cells) == 3 * 2 * 2 * 2
    keys = {(c["seed"], c["doc_gold_n"], c["local_label_mix"], c["leaf_unit_count"]) for c in cells}
    assert len(keys) == len(cells)  # every cell fully specified and unique
    gold_cells = [c for c in cells if c["local_label_mix"] == "gold_fraction"]
    assert all(c["gold_fraction_p"] == pytest.approx(0.5) for c in gold_cells)
