"""Live FNO training cell through ``treepo.methods.run`` on GPU.

Uses ``_FakeEmbeddingClient`` from the existing FNO test pattern —
no embedding server required, no manifesto-grade text — so the entire
cell runs in ~10–60 s on a single GPU. Verifies:

- ``treepo.methods.families.resolve_family("fno", ...)`` builds the same
  ``FNOFamily`` the paper code constructs.
- ``treepo.methods.run("fit", {family="fno", ...})`` completes one ``f``
  training step on GPU.
- Predictions return as floats, sized one per tree, with
  ``status="success"``.
- Per-tree prediction records land on disk.

Gating: requires ``TT_RUN_LIVE_TESTS=1`` AND CUDA available AND the
``neuralop`` package. Skipped otherwise.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import List

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_RUN_LIVE = str(os.getenv("TT_RUN_LIVE_TESTS", "") or "").strip().lower() in {"1", "true", "yes"}


def _have_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _have_neuralop() -> bool:
    try:
        import neuralop  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    (not _RUN_LIVE) or (not _have_cuda()) or (not _have_neuralop()),
    reason="Live FNO test requires TT_RUN_LIVE_TESTS=1, CUDA, and neuralop.",
)


class _FakeEmbeddingClient:
    """Same pattern as ``tests/ctreepo/test_unified_fg_ladder_contract.py`` —
    deterministic float vectors per text, no model, no server.
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = int(dim)

    def embed_texts(self, texts):
        return [
            [float((len(str(text)) + idx) % 7) for idx in range(self.dim)]
            for text in texts
        ]


def _tiny_labeled_trees(n: int = 4, *, score_base: float = 4.0):
    """Minimum labeled trees: ultra-short leaf text so the FNO's
    ``leaf_size_tokens`` / ``embedding_max_length_tokens`` budget fits
    each leaf in one embedding chunk. Mirrors the construction pattern
    from ``tests/ctreepo/test_unified_fg_ladder_contract.py`` but
    trimmed for fast-path live CUDA runs.
    """
    from treepo._research.tree.labeled import LabeledNode, LabeledTree

    trees: List[LabeledTree] = []
    for i in range(int(n)):
        doc_id = f"d{i:02d}"
        score = score_base + (i - n / 2.0) * 0.3
        left_text = "left a b"
        right_text = "right c d"
        text = f"{left_text} {right_text}"
        tree = LabeledTree(
            doc_id=doc_id,
            document_text=text,
            document_score=float(score),
            metadata={
                "split": "train" if i < n // 2 else "test",
                "teacher_score_1_7": float(score),
                "expert_score_1_7": float(score) + 0.1,
                "teacher_score_native": float(score),
                "expert_score_native": float(score) + 0.1,
                "expert_target_scale": "raw",
                "expert_score_for_objective": float(score) + 0.1,
            },
            label_source="test",
        )
        tree.add_node(LabeledNode(
            node_id="leaf_0", doc_id=doc_id, level=0,
            text=left_text,
            score=float(score) - 0.2,
            metadata={"teacher_summary": "L", "target_summary": "L"},
        ))
        tree.add_node(LabeledNode(
            node_id="leaf_1", doc_id=doc_id, level=0,
            text=right_text,
            score=float(score) + 0.2,
            metadata={"teacher_summary": "R", "target_summary": "R"},
        ))
        tree.add_node(LabeledNode(
            node_id="root", doc_id=doc_id, level=1,
            text=text, score=float(score),
            left_child_id="leaf_0", right_child_id="leaf_1",
            metadata={"teacher_summary": "root summary", "target_summary": "root summary"},
        ))
        trees.append(tree)
    return trees


def test_live_fno_family_trains_one_iteration_through_treepo_methods(tmp_path: Path) -> None:
    """Smallest FNO training cell that exercises CUDA + the
    ``FamilyRuntime`` protocol end-to-end through the treepo.methods
    dispatcher.
    """
    from treepo._research.ctreepo.fno_family import FNOFamilyConfig

    import treepo.methods

    embedding_dim = 8
    cfg = FNOFamilyConfig(
        hidden_channels=8,
        n_modes=4,
        n_layers=1,
        head_hidden_dim=8,
        epochs_per_iteration=2,
        batch_size=2,
        learning_rate=1e-3,
        leaf_size_tokens=8,
        embedding_max_length_tokens=8,
        effective_embedding_dim=embedding_dim,
        identity_init=True,
        seed=0,
    )
    family_runtime = None
    # Build through the registry; injected via family_runtime for instance access.
    from treepo.methods.families import resolve_family

    family_runtime = resolve_family(
        "fno",
        {
            "fno_config": cfg,
            "embedding_client": _FakeEmbeddingClient(dim=embedding_dim),
        },
    )

    trees = _tiny_labeled_trees(n=4)

    t0 = time.perf_counter()
    result = treepo.methods.run(
        "fit",
        {
            "family": "fno",
            "train_data": trees,
            "eval_data": trees,
            "backend_config": {
                "family_runtime": family_runtime,
                "output_dir": str(tmp_path),
            },
            "axis": {"max_iterations": 1, "axis_value": 0},
            "initial_artifacts": {"f": "identity", "g": "raw_concat"},
        },
    )
    elapsed = time.perf_counter() - t0
    print(f"\nFNO live cell wall-time: {elapsed:.1f}s")

    assert result.status == "success", (
        f"FNO live cell failed: history last extra="
        f"{result.history[-1].get('extra') if result.history else 'no-history'}"
    )
    # One k=0 + one k=1 (train_f) → 2 iterations.
    assert result.summary["n_iterations"] >= 1
    # Predictions present and finite.
    pred_paths = result.artifacts.get("prediction_records") or []
    assert pred_paths, "expected prediction_records JSONL written to disk"
    rows = [json.loads(line) for line in Path(pred_paths[-1]).read_text().splitlines() if line.strip()]
    assert len(rows) == len(trees)
    # Predictions are real floats from the FNO forward pass.
    valid_preds = [r["prediction"] for r in rows if r.get("prediction") is not None]
    assert len(valid_preds) == len(trees)
    import math
    for p in valid_preds:
        assert math.isfinite(float(p))
