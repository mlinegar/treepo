"""The RILE qsentence regime: only leaf (qsentence) and document labels observed.

Additive targets like RILE satisfy the rollup identity — the document value is
the (unit-count-weighted) mean of leaf values — so ``root_readout='leaf_mean'``
plus leaf+root supervision recovers the document construct with no merge labels
at all. These tests cover the rollup weight extraction, the model mechanics,
the identity report, the full leaf+root training regime, and (gated) the real
gold-qsentence bundle.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from treepo import fit
from treepo.bundles import (
    BundleFormatError,
    leaf_rollup_report,
    load_labeled_tree_bundle,
)
from treepo.methods._fno_config import NeuralOperatorFamilyConfig
from treepo.methods._fno_targets import _leaf_rollup_weights
from treepo.methods.families import resolve_family
from treepo.tree import TreeNode, TreeRecord


def _leaf(doc_id: str, i: int, label: float | None, *, meta: dict | None = None) -> TreeNode:
    return TreeNode(
        node_id=f"{doc_id}_l{i}",
        unit_type="leaf",
        text=f"{doc_id} leaf {i} alpha beta gamma",
        level=0,
        position=i,
        label=label,
        metadata=meta or {},
    )


def _leaf_root_tree(
    doc_id: str,
    leaf_labels: list[float],
    *,
    root_label: float | None = None,
    weights: list[float] | None = None,
    split: str = "train",
) -> TreeRecord:
    """Leaves + root label only — no internal merge nodes, like a qsentence doc."""

    leaves = [
        _leaf(
            doc_id,
            i,
            label,
            meta={"unit_count": weights[i]} if weights is not None else None,
        )
        for i, label in enumerate(leaf_labels)
    ]
    if root_label is None:
        if weights is None:
            root_label = sum(leaf_labels) / len(leaf_labels)
        else:
            root_label = sum(w * v for w, v in zip(weights, leaf_labels)) / sum(weights)
    return TreeRecord(
        tree_id=doc_id,
        root_label=root_label,
        nodes=tuple(leaves),
        metadata={"split": split},
    )


# --------------------------- weight extraction --------------------------- #


def test_rollup_weights_equal_by_default() -> None:
    config = NeuralOperatorFamilyConfig(root_readout="leaf_mean")
    tree = _leaf_root_tree("doc", [0.0, 1.0, 0.5])
    rows = _leaf_rollup_weights([tree], config, max_leaves=5)
    assert rows[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0.0, 0.0])


def test_rollup_weights_from_metadata_key() -> None:
    config = NeuralOperatorFamilyConfig(
        root_readout="leaf_mean", rollup_weight_key="unit_count"
    )
    tree = _leaf_root_tree("doc", [0.0, 1.0], weights=[3.0, 1.0])
    rows = _leaf_rollup_weights([tree], config, max_leaves=2)
    assert rows[0] == pytest.approx([0.75, 0.25])


def test_rollup_weights_missing_key_is_loud() -> None:
    config = NeuralOperatorFamilyConfig(
        root_readout="leaf_mean", rollup_weight_key="unit_count"
    )
    tree = _leaf_root_tree("doc", [0.0, 1.0])  # no unit_count metadata
    with pytest.raises(ValueError, match="rollup_weight_key='unit_count' is missing"):
        _leaf_rollup_weights([tree], config, max_leaves=2)


def test_rollup_config_validation() -> None:
    with pytest.raises(ValueError, match="root_readout must be"):
        resolve_family("neural_operator", {"operator_kind": "conv1d", "root_readout": "topk"})
    with pytest.raises(ValueError, match="only meaningful with"):
        resolve_family(
            "neural_operator",
            {"operator_kind": "conv1d", "rollup_weight_key": "unit_count"},
        )


# ----------------------------- model mechanics ---------------------------- #


def test_forward_rollup_is_weighted_mean_of_leaf_readouts() -> None:
    family = resolve_family(
        "neural_operator",
        {
            "operator_kind": "conv1d",
            "embedding_dim": 8,
            "hidden_channels": 4,
            "n_layers": 1,
            "head_hidden_dim": 8,
            "device": "cpu",
            "seed": 0,
            "root_readout": "leaf_mean",
        },
    )
    trees = [_leaf_root_tree("doc_a", [0.2, 0.8, 0.5]), _leaf_root_tree("doc_b", [0.1, 0.9, 0.4])]
    x, lengths = family._encode_trees(trees)
    family._ensure_model(output_dim=1)
    weights = family._rollup_weights_tensor(trees, max_leaves=int(x.shape[1]))
    torch = family._torch
    with torch.no_grad():
        rollup, traces = family._model.forward_rollup(x, lengths, weights)
        manual = (family._model.readout(family._model.leaf_operator(x)) * weights.unsqueeze(-1)).sum(dim=1)
    assert traces == []
    assert torch.allclose(rollup, manual)


# ----------------------------- identity report ---------------------------- #


def test_leaf_rollup_report_exact_identity() -> None:
    records = [
        _leaf_root_tree(f"doc_{i}", [0.1 * i, 0.5, 0.9 - 0.05 * i]) for i in range(5)
    ]
    report = leaf_rollup_report(records)
    assert report["n_docs"] == 5
    assert report["n_skipped"] == 0
    assert report["mean_abs_error"] == pytest.approx(0.0, abs=1e-12)
    assert report["pearson"] == pytest.approx(1.0)


def test_leaf_rollup_report_weighted_identity_and_skips() -> None:
    records = [
        _leaf_root_tree(f"doc_{i}", [0.0, 1.0], weights=[3.0, 1.0 + i]) for i in range(4)
    ]
    records.append(
        TreeRecord(tree_id="unlabeled", root_label=None, nodes=(_leaf("unlabeled", 0, None),))
    )
    report = leaf_rollup_report(records, weight_key="unit_count")
    assert report["n_docs"] == 4
    assert report["n_skipped"] == 1
    assert report["max_abs_error"] == pytest.approx(0.0, abs=1e-12)


def test_leaf_rollup_report_missing_weight_key_is_loud() -> None:
    with pytest.raises(BundleFormatError, match="no rollup weight"):
        leaf_rollup_report([_leaf_root_tree("doc", [0.0, 1.0])], weight_key="unit_count")


# ------------------------ leaf+root training regime ----------------------- #


def _backend(tmp_path: Path, **extra: object) -> dict[str, object]:
    return {
        "operator_kind": "conv1d",
        "embedding_dim": 16,
        "hidden_channels": 8,
        "n_layers": 1,
        "head_hidden_dim": 16,
        "epochs_per_iteration": 2,
        "batch_size": 8,
        "learning_rate": 0.01,
        "device": "cpu",
        "seed": 0,
        "output_dir": str(tmp_path),
        **extra,
    }


def test_leaf_root_only_regime_consumes_no_merge_rows(tmp_path: Path) -> None:
    trees = [_leaf_root_tree(f"doc_{i}", [0.1 * (i % 5), 0.5, 0.8]) for i in range(8)]
    result = fit(
        {
            "family": "neural_operator",
            "train_data": trees,
            "eval_data": trees,
            "root_weight": 1.0,
            "leaf_weight": 1.0,
            "backend_config": _backend(tmp_path, root_readout="leaf_mean"),
        }
    )
    assert result.status == "success"
    payload = result.artifacts["g"]["node_supervision"]
    assert payload["n_leaf_rows"] == 24  # 8 docs x 3 leaves
    assert payload["n_merge_rows"] == 0
    assert result.artifacts["g"]["root_readout"] == "leaf_mean"


WORDS = {1.0: "expand welfare protect equality", 0.0: "cut taxes defend markets"}


def _dgp_tree(doc_id: str, rng: random.Random, *, n_leaves: int = 8, split: str = "train") -> TreeRecord:
    labels = [rng.choice([0.0, 1.0]) for _ in range(n_leaves)]
    leaves = [
        TreeNode(
            node_id=f"{doc_id}_l{i}",
            unit_type="leaf",
            text=WORDS[label] + f" filler{i % 3}",
            level=0,
            position=i,
            label=label,
        )
        for i, label in enumerate(labels)
    ]
    return TreeRecord(
        tree_id=doc_id,
        root_label=sum(labels) / len(labels),
        nodes=tuple(leaves),
        metadata={"split": split},
    )


def test_additive_dgp_recovery_through_leaf_mean(tmp_path: Path) -> None:
    """The RILE mechanism end-to-end: leaf labels + doc label only, doc value

    additive in the leaves. Learning the leaf map and rolling up must recover
    held-out document scores nearly exactly (the rollup identity in-model).
    """

    rng = random.Random(0)
    train = [_dgp_tree(f"tr{i}", rng) for i in range(24)]
    test = [_dgp_tree(f"te{i}", rng, split="test") for i in range(16)]
    result = fit(
        {
            "family": "neural_operator",
            "train_data": train,
            "eval_data": test,
            "root_weight": 1.0,
            "leaf_weight": 1.0,
            "backend_config": _backend(
                tmp_path,
                root_readout="leaf_mean",
                epochs_per_iteration=30,
                learning_rate=0.02,
            ),
        }
    )
    assert result.status == "success"
    metrics = result.summary["split_metrics"]["test"]
    assert metrics["internal_f_pearson"] > 0.95
    assert metrics["internal_f_mae"] < 0.05


def test_weighted_rollup_regime_trains(tmp_path: Path) -> None:
    rng = random.Random(1)
    trees = []
    for i in range(8):
        labels = [rng.choice([0.0, 1.0]) for _ in range(4)]
        weights = [float(rng.randint(1, 5)) for _ in range(4)]
        trees.append(
            _leaf_root_tree(f"doc_{i}", labels, weights=weights, split="train")
        )
    result = fit(
        {
            "family": "neural_operator",
            "train_data": trees,
            "eval_data": trees,
            "root_weight": 1.0,
            "leaf_weight": 1.0,
            "backend_config": _backend(
                tmp_path, root_readout="leaf_mean", rollup_weight_key="unit_count"
            ),
        }
    )
    assert result.status == "success"
    assert result.artifacts["g"]["rollup_weight_key"] == "unit_count"


# ------------------------------ real bundle ------------------------------- #

_TT_QBUNDLE = os.environ.get("TREEPO_TT_QBUNDLE")
_TT_QBUNDLE_WEIGHT_KEY = os.environ.get("TREEPO_TT_QBUNDLE_WEIGHT_KEY") or None


@pytest.mark.skipif(not _TT_QBUNDLE, reason="set TREEPO_TT_QBUNDLE to a gold leaf-labeled bundle to run")
def test_real_bundle_rollup_identity_and_leaf_root_fit(tmp_path: Path) -> None:
    train = load_labeled_tree_bundle(_TT_QBUNDLE, split="train", dimension="rile")
    report = leaf_rollup_report(train, weight_key=_TT_QBUNDLE_WEIGHT_KEY)
    # Guard against placeholder artifacts (e.g. all-neutral 0.5 labels): a
    # zero-error identity over constant labels proves nothing.
    assert not report["degenerate_root_labels"], (
        "bundle root labels are constant — placeholder labels, not gold"
    )
    assert report["max_abs_error"] is not None and report["max_abs_error"] < 1e-6
    result = fit(
        {
            "family": "neural_operator",
            "train_data": train[:24],
            "eval_data": load_labeled_tree_bundle(_TT_QBUNDLE, split="val", dimension="rile"),
            "root_weight": 1.0,
            "leaf_weight": 1.0,
            "backend_config": _backend(
                tmp_path,
                root_readout="leaf_mean",
                rollup_weight_key=_TT_QBUNDLE_WEIGHT_KEY,
            ),
        }
    )
    assert result.status == "success"
    assert result.artifacts["g"]["node_supervision"]["n_merge_rows"] == 0
