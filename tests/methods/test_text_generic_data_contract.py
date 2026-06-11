"""Raw text is the universal data object of the unified ``fit()`` surface.

The contract this file locks: every family consumes the SAME raw-text
``LabeledTree`` objects — unchanged, no pre-embedding, no per-backend data
format. Whether a family prompts an LLM with the node text (dspy), embeds
it for a neural operator (fno), or ignores it for a scalar IPW estimator
(learnable_constant) is internal to the family, never a property of the
data. Transformers/LLMs, embedding operators, and future backends (e.g. a
diffusion LM family) are interchangeable behind ``register_family`` over
identical text trees.

This file asserts the property through the ``treepo.methods`` dispatch
surface (the parent workspace's ladder-contract test does the same at the
FamilyRuntime level).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import List

import pytest

from treepo._research.tree.labeled import LabeledNode, LabeledTree


def make_text_trees() -> List[LabeledTree]:
    """Plain-text labeled trees, one shared builder for every family."""
    trees: List[LabeledTree] = []
    for i in range(6):
        doc_id = f"doc_{i}"
        score = 3.0 + 0.4 * i
        left_text = f"{doc_id} left policy evidence about investment and jobs."
        right_text = f"{doc_id} right policy evidence about taxation and welfare."
        tree = LabeledTree(
            doc_id=doc_id,
            document_text=f"{left_text} {right_text}",
            document_score=score,
            metadata={
                "split": "train" if i % 2 == 0 else "test",
                "teacher_score_1_7": score,
                "expert_score_1_7": score + 0.1,
                "observed": True,
                "propensity": 1.0,
            },
            label_source="test",
        )
        tree.add_node(LabeledNode(node_id="leaf_0", doc_id=doc_id, level=0,
                                  text=left_text, score=score - 0.2))
        tree.add_node(LabeledNode(node_id="leaf_1", doc_id=doc_id, level=0,
                                  text=right_text, score=score + 0.2))
        tree.add_node(LabeledNode(node_id="root", doc_id=doc_id, level=1,
                                  text=f"{left_text} {right_text}", score=score,
                                  left_child_id="leaf_0", right_child_id="leaf_1"))
        trees.append(tree)
    return trees


class _HashingEmbeddingClient:
    """Deterministic offline embedding — the fno family's internal detail."""

    def __init__(self, dim: int = 64):
        self.dim = int(dim)

    def resolve_model(self) -> str:
        return f"hashing_embedding:{self.dim}"

    def embed_texts(self, texts):
        outputs = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in str(text or "").lower().split():
                digest = hashlib.blake2b(
                    token.encode("utf-8", errors="ignore"), digest_size=8
                ).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                vec[bucket] += -1.0 if (digest[4] & 1) else 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            outputs.append([v / norm for v in vec])
        return outputs


def test_fno_family_consumes_raw_text_trees(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("neuralop")
    import treepo.methods
    from treepo._research.ctreepo.fno_family import FNOFamilyConfig

    trees = make_text_trees()
    result = treepo.methods.run("fit", {
        "family": "fno",
        "train_data": trees,
        "eval_data": trees,
        "backend_config": {
            "fno_config": FNOFamilyConfig(
                hidden_channels=8, n_modes=4, n_layers=1, head_hidden_dim=16,
                leaf_size_tokens=64, epochs_per_iteration=1, batch_size=2,
                effective_embedding_dim=64, identity_init=True, seed=42,
                embedding_max_length_tokens=None,
            ),
            "embedding_client": _HashingEmbeddingClient(dim=64),
            "output_dir": str(tmp_path / "fit"),
        },
        "axis": {"max_iterations": 1, "axis_value": 0},
        "initial_artifacts": {"f": "identity", "g": "identity"},
    })
    assert result.status == "success"


def test_learnable_constant_family_consumes_same_trees(tmp_path: Path) -> None:
    from treepo._research.ctreepo.contracts import CTreePOLearningSpec
    from treepo.methods import fit

    trees = make_text_trees()
    spec = CTreePOLearningSpec(
        space_kind="learnable_constant",
        family="learnable_constant",
        schedule="fg",
        initial_artifacts={"f": 0.0, "g": None},
        train_data=trees,
        eval_data=[],
        backend_config={"output_dir": str(tmp_path)},
        axis={"max_iterations": 1, "axis_value": 0},
    )
    result = fit(spec)
    assert result.status == "success"


def test_dspy_llm_family_consumes_same_trees(monkeypatch) -> None:
    """The LLM family reduces the identical raw-text trees — node text goes
    straight into LM prompts; no embedding or other transformation of the
    data is required (LM calls faked, no server)."""
    pytest.importorskip("dspy")
    from treepo._research.ctreepo.dspy_family import DSPyFamily, DSPyFamilyConfig

    family = DSPyFamily(
        config=DSPyFamilyConfig(
            dimension="economic",
            num_threads=1,
            leaf_size_tokens=4,
            max_completion_tokens=8,
            lm_config={"model": "fake", "api_base": "http://localhost",
                       "api_key": "EMPTY"},
        )
    )
    monkeypatch.setattr(family, "_load_g_program", lambda artifact: object())
    monkeypatch.setattr(family, "_load_f_program", lambda artifact: object())
    seen_prompts: list[str] = []

    def fake_apply_g(_program, *, prompt: str) -> str:
        seen_prompts.append(prompt)
        return "merged_state"

    monkeypatch.setattr(family, "_apply_g", fake_apply_g)
    monkeypatch.setattr(
        family, "_apply_f_normalized", lambda _program, *, response: 0.5
    )

    trees = make_text_trees()
    scores = family.score_roots_with_f(
        f="trained_f.json", g="trained_g.json", trees=trees
    )

    assert len(scores) == len(trees)
    assert all(s is not None and math.isfinite(s) for s in scores)
    # The raw leaf text itself reached the LM prompt surface.
    assert any("investment and jobs" in prompt for prompt in seen_prompts)
