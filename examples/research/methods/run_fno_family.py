#!/usr/bin/env python3
"""Example: FNO family fit via `treepo.methods.run("fit", {family="fno", ...})`.

Pattern: load TOML directly into the upstream FNOFamilyConfig — no mirror.
Train/eval data are tiny in-memory `LabeledTree`s (the FNO family's input
type), embedded by a deterministic offline hashing client — no server, CPU
only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "configs/research/methods/fno_smoke.toml")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    import hashlib
    import math

    import treepo.methods
    from treepo.methods.canonical_defaults import load_dataclass
    from treepo._research.ctreepo.fno_family import FNOFamilyConfig
    from treepo._research.tree.labeled import LabeledNode, LabeledTree

    class HashingEmbeddingClient:
        """Deterministic offline embedding client (no server needed)."""

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

    def tiny_tree(doc_id: str, split: str, score: float) -> LabeledTree:
        left_text = f"{doc_id} left policy evidence about investment and jobs."
        right_text = f"{doc_id} right policy evidence about taxation and welfare."
        tree = LabeledTree(
            doc_id=doc_id,
            document_text=f"{left_text} {right_text}",
            document_score=score,
            metadata={"split": split,
                      "expert_score_1_7": score + 0.1,
                      "teacher_score_1_7": score},
            label_source="example",
        )
        tree.add_node(LabeledNode(node_id="leaf_0", doc_id=doc_id, level=0,
                                  text=left_text, score=score - 0.2))
        tree.add_node(LabeledNode(node_id="leaf_1", doc_id=doc_id, level=0,
                                  text=right_text, score=score + 0.2))
        tree.add_node(LabeledNode(node_id="root", doc_id=doc_id, level=1,
                                  text=f"{left_text} {right_text}", score=score,
                                  left_child_id="leaf_0", right_child_id="leaf_1"))
        return tree

    cfg = load_dataclass(args.config, FNOFamilyConfig,
                         overrides={"epochs_per_iteration": args.epochs})
    # Offline example: the leaves are tiny, so skip the no-truncation token
    # guard rather than require the local tokenizer checkpoint it loads.
    cfg.embedding_max_length_tokens = None
    trees = [tiny_tree(f"doc_{i}", "train" if i % 2 == 0 else "test",
                       3.0 + 0.5 * i) for i in range(6)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = treepo.methods.run("fit", {
        "family": "fno",
        "train_data": trees, "eval_data": trees,
        "backend_config": {"fno_config": cfg,
                           "embedding_client": HashingEmbeddingClient(),
                           "output_dir": str(args.output_dir / "fit")},
        "axis": {"max_iterations": 1, "axis_value": 0},
        "initial_artifacts": {"f": "identity", "g": "identity"},
    })
    print(f"status={result.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
