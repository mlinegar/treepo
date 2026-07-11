#!/usr/bin/env python3
"""Load an external labeled-tree bundle and hand it to ``fit()``.

Producers (the ThinkingTrees Manifesto/RILE grids, and later Markov or
Benoit-econ tasks) publish labeled trees to the shared bundle contract described
in ``docs/bundle_contract.md``. This walkthrough loads such a bundle into
package-owned ``TreeRecord`` objects and shows the ``fit()`` wiring, without
porting any task into treepo.

Point it at a real bundle with ``bundle_path`` in the config (or ``--bundle``).
With no bundle configured it writes a tiny synthetic bundle to the output
directory and loads that, so the script runs anywhere with no external data.
``run_fit = true`` (the default) trains the configured family on the loaded
trees; ``supervision_level`` selects a named per-node supervision cell
(default/root/leaf/node/mix) for families that consume node labels
(neural_operator/fno).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - older interpreters
    tomllib = None  # type: ignore[assignment]

from example_setup import write_json

from treepo.bundles import load_labeled_tree_bundle
from treepo.tree import tree_summary


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists() or tomllib is None:
        return {}
    return dict(tomllib.loads(path.read_text(encoding="utf-8")))


def _write_synthetic_bundle(directory: Path) -> Path:
    """Write a minimal, contract-conformant bundle for the no-data path."""

    directory.mkdir(parents=True, exist_ok=True)

    def node(node_id, level, score, dims, left=None, right=None, is_leaf=True):
        return {
            "node_id": node_id,
            "level": level,
            "text": f"unit {node_id}",
            "score": score,
            "dimension_scores": dims,
            "left_child_id": left,
            "right_child_id": right,
            "metadata": {
                "is_leaf": is_leaf,
                "g_training_role": "leaf" if is_leaf else "merge",
                "cmp_counts": {"402": 1} if is_leaf else {"402": 2},
                "total_non_header_qsentences": 2,
                "sentence_start_index": 0,
                "sentence_end_index": 2,
            },
        }

    def tree(doc_id, split, leaf_scores, root_score):
        leaves = [
            node(f"node_l0_{i:05d}", 0, s, {"rile": s, "domain_1": 1.0 - s})
            for i, s in enumerate(leaf_scores)
        ]
        root = node(
            "node_l1_00000",
            1,
            root_score,
            {"rile": root_score, "domain_1": 1.0 - root_score},
            left=leaves[0]["node_id"],
            right=leaves[1]["node_id"],
            is_leaf=False,
        )
        nodes = {n["node_id"]: n for n in [*leaves, root]}
        return {
            "version": "3.0",
            "doc_id": doc_id,
            "document_text": "unit a unit b",
            "document_score": root_score,
            "nodes": nodes,
            "levels": [[leaf["node_id"] for leaf in leaves], [root["node_id"]]],
            "metadata": {"split": split, "artifact_version": "synthetic_example_v1"},
            "label_source": "synthetic_example_v1",
        }

    trees = [
        tree("doc_a", "train", [0.7, 0.3], 0.5),
        tree("doc_b", "test", [0.2, 0.8], 0.5),
    ]
    (directory / "labeled_trees.jsonl").write_text(
        "\n".join(json.dumps(t) for t in trees) + "\n", encoding="utf-8"
    )
    (directory / "split_ids.json").write_text(
        json.dumps({"train": ["doc_a"], "val": [], "test": ["doc_b"]}), encoding="utf-8"
    )
    return directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("manifesto_bundle.toml"))
    parser.add_argument("--bundle", type=Path, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(args.config)

    bundle_path = args.bundle or (Path(cfg["bundle_path"]) if cfg.get("bundle_path") else None)
    synthesized = False
    if bundle_path is None:
        bundle_path = _write_synthetic_bundle(output_dir / "synthetic_bundle")
        synthesized = True

    dimension = cfg.get("dimension") or None
    train = load_labeled_tree_bundle(bundle_path, split="train", dimension=dimension)
    eval_trees = load_labeled_tree_bundle(bundle_path, split="test", dimension=dimension)

    summary = {
        "bundle_path": str(bundle_path),
        "synthesized": synthesized,
        "dimension": dimension,
        "n_train": len(train),
        "n_eval": len(eval_trees),
        "train_trees": [tree_summary(record) for record in train[:3]],
    }

    # This is the wiring a grid driver uses: loaded trees flow straight into the
    # unified fit() surface as train_data / eval_data. Training is opt-in because
    # a real family backend (and its optional dependencies) is not needed just to
    # demonstrate the load.
    fit_config: dict[str, Any] = {
        "family": str(cfg.get("family", "learnable_constant")),
        "train_data": train,
        "eval_data": eval_trees,
        "axis": {"axis_kind": "leaf_count", "axis_value": int(cfg.get("leaf_count", 2))},
        "supervision_level": str(cfg.get("supervision_level", "default")),
    }
    # Explicit node weights cover regimes the named levels don't (e.g. the RILE
    # qsentence case: leaf + root observed, merges unlabeled -> root=1, leaf=1).
    for weight_field in ("root_weight", "leaf_weight", "merge_weight"):
        if cfg.get(weight_field) is not None:
            fit_config[weight_field] = float(cfg[weight_field])
    backend: dict[str, Any] = {}
    if cfg.get("root_readout"):
        backend["root_readout"] = str(cfg["root_readout"])
    if cfg.get("rollup_weight_key"):
        backend["rollup_weight_key"] = str(cfg["rollup_weight_key"])
    if backend:
        fit_config["backend_config"] = backend

    # Per-node supervision consumes the bundle's node labels directly: pick a
    # named level (root/leaf/node/mix) with a neural_operator/fno family, or
    # keep the default level for root-only families like learnable_constant.
    fit_result: dict[str, Any] | None = None
    if bool(cfg.get("run_fit", True)):
        from treepo import fit

        try:
            result = fit(fit_config)
            fit_result = result.to_dict()
            summary["fit_status"] = result.status
        except Exception as exc:  # noqa: BLE001 - demonstration surface
            summary["fit_status"] = f"error: {exc}"

    result_path = output_dir / "manifesto_bundle_result.json"
    write_json(
        result_path,
        {"summary": summary, "fit_config_family": fit_config["family"], "fit_result": fit_result},
    )

    print(
        f"bundle={'synthetic' if synthesized else 'external'} "
        f"train={len(train)} eval={len(eval_trees)} "
        f"dimension={dimension or 'default'} "
        f"run_fit={bool(cfg.get('run_fit', True))} output={result_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
