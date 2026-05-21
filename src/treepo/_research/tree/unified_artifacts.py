"""
Unified tree artifacts: support spans, embeddings, and mergeable sketch states.

This module is the glue between:
  - window selection / chunking (what text spans we "look at"),
  - LLM-based OPS trees (text summaries),
  - embedding/sketch-based representations (mergeable vectors).

Design goals:
  - Task-agnostic, purely structural/representational utilities.
  - Deterministic outputs given fixed inputs (good for caching + audits).
  - Strict mergeability for sketch states when desired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from treepo._research.core.data_models import Node, Tree


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover - protocol
        ...


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vec) + 1e-12)
    return vec / denom


@dataclass(frozen=True)
class MergeableVectorState:
    """Strictly mergeable vector summary."""

    sum_vec: np.ndarray  # [D]
    count: int

    def merge(self, other: "MergeableVectorState") -> "MergeableVectorState":
        if int(self.sum_vec.shape[0]) != int(other.sum_vec.shape[0]):
            raise ValueError("State dimension mismatch")
        return MergeableVectorState(sum_vec=self.sum_vec + other.sum_vec, count=int(self.count + other.count))

    def mean(self) -> np.ndarray:
        denom = max(1, int(self.count))
        return (self.sum_vec / float(denom)).astype(np.float32, copy=False)


def attach_chunk_support(
    tree: Tree,
    *,
    overwrite: bool = False,
) -> None:
    """
    Attach per-node support spans (char offsets) into Node.metadata.

    If tree.metadata contains a 'chunk_boundaries' list matching the number of
    leaves, we use it to set leaf support exactly. Otherwise, we fall back to
    cumulative widths of leaf raw_text_span.
    """
    leaves = tree.leaves
    boundaries = tree.metadata.get("chunk_boundaries")

    if isinstance(boundaries, list) and len(boundaries) == len(leaves):
        for idx, (leaf, row) in enumerate(zip(leaves, boundaries)):
            if not isinstance(row, dict):
                row = {}
            chunk_index = row.get("chunk_index", idx)
            start_char = row.get("start_char", None)
            end_char = row.get("end_char", None)

            if overwrite or "chunk_index" not in leaf.metadata:
                leaf.metadata["chunk_index"] = int(chunk_index) if isinstance(chunk_index, int) else int(idx)
            if start_char is not None and (overwrite or "char_start" not in leaf.metadata):
                leaf.metadata["char_start"] = int(start_char)
            if end_char is not None and (overwrite or "char_end" not in leaf.metadata):
                leaf.metadata["char_end"] = int(end_char)

            token_count = row.get("token_count", None)
            if token_count is not None and (overwrite or "token_count" not in leaf.metadata):
                try:
                    leaf.metadata["token_count"] = int(token_count)
                except (TypeError, ValueError):
                    pass
    else:
        cursor = 0
        for idx, leaf in enumerate(leaves):
            width = len(leaf.raw_text_span or "")
            if overwrite or "chunk_index" not in leaf.metadata:
                leaf.metadata["chunk_index"] = int(idx)
            if overwrite or "char_start" not in leaf.metadata:
                leaf.metadata["char_start"] = int(cursor)
            if overwrite or "char_end" not in leaf.metadata:
                leaf.metadata["char_end"] = int(cursor + width)
            cursor += width

    # Propagate support spans upward.
    for node in tree.traverse_postorder():
        if node.is_leaf:
            continue
        left = node.left_child
        right = node.right_child
        if left is None or right is None:
            continue
        l_start = left.metadata.get("char_start")
        l_end = left.metadata.get("char_end")
        r_start = right.metadata.get("char_start")
        r_end = right.metadata.get("char_end")
        if not all(isinstance(v, int) for v in [l_start, l_end, r_start, r_end]):
            continue
        if overwrite or "char_start" not in node.metadata:
            node.metadata["char_start"] = min(int(l_start), int(r_start))
        if overwrite or "char_end" not in node.metadata:
            node.metadata["char_end"] = max(int(l_end), int(r_end))


def embed_leaf_texts(
    tree: Tree,
    *,
    embedding_client: EmbeddingClient,
    text_field: str = "raw_text_span",
    embedding_key: str = "leaf_embedding",
    l2_normalize: bool = True,
    overwrite: bool = False,
) -> int:
    """
    Embed each leaf's text and store vectors in Node.metadata.

    Returns:
        Number of leaves embedded.
    """
    leaves = tree.leaves
    texts: List[str] = []
    target_leaves: List[Node] = []
    for leaf in leaves:
        if not overwrite and embedding_key in leaf.metadata:
            continue
        raw = getattr(leaf, text_field, None)
        texts.append(str(raw or ""))
        target_leaves.append(leaf)

    if not texts:
        return 0

    vectors = embedding_client.embed_texts(texts)
    if len(vectors) != len(texts):
        raise RuntimeError(f"Embedding client returned {len(vectors)} vectors for {len(texts)} texts")

    for leaf, vec in zip(target_leaves, vectors):
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] <= 0:
            raise RuntimeError("Invalid embedding vector shape")
        if l2_normalize:
            arr = _l2_normalize(arr)
        leaf.metadata[embedding_key] = [float(x) for x in arr.tolist()]

    return len(texts)


def build_mergeable_sum_state_from_embeddings(
    tree: Tree,
    *,
    leaf_embedding_key: str = "leaf_embedding",
    state_key: str = "embedding_sum_state",
    overwrite: bool = False,
) -> None:
    """
    Build a strictly mergeable (sum, count) state bottom-up from leaf embeddings.

    Stores state dicts under Node.metadata[state_key]:
        {"sum": [...], "count": int}
    """

    def _load_vec(node: Node) -> Optional[np.ndarray]:
        vec = node.metadata.get(leaf_embedding_key)
        if not isinstance(vec, list) or not vec:
            return None
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] <= 0:
            return None
        return arr

    def _store(node: Node, state: MergeableVectorState) -> None:
        node.metadata[state_key] = {"sum": [float(x) for x in state.sum_vec.tolist()], "count": int(state.count)}

    for node in tree.traverse_postorder():
        if (not overwrite) and state_key in node.metadata:
            continue
        if node.is_leaf:
            vec = _load_vec(node)
            if vec is None:
                continue
            _store(node, MergeableVectorState(sum_vec=vec, count=1))
            continue

        left = node.left_child
        right = node.right_child
        if left is None or right is None:
            continue
        left_state = left.metadata.get(state_key)
        right_state = right.metadata.get(state_key)
        if not isinstance(left_state, dict) or not isinstance(right_state, dict):
            continue
        l_sum = np.asarray(left_state.get("sum", []), dtype=np.float32)
        r_sum = np.asarray(right_state.get("sum", []), dtype=np.float32)
        if l_sum.ndim != 1 or r_sum.ndim != 1 or l_sum.shape[0] == 0 or r_sum.shape[0] == 0:
            continue
        if int(l_sum.shape[0]) != int(r_sum.shape[0]):
            raise ValueError("Embedding state dimension mismatch while merging")
        try:
            l_count = int(left_state.get("count", 0))
            r_count = int(right_state.get("count", 0))
        except (TypeError, ValueError):
            continue
        merged = MergeableVectorState(sum_vec=l_sum + r_sum, count=l_count + r_count)
        _store(node, merged)


def get_root_state(
    tree: Tree,
    *,
    state_key: str = "embedding_sum_state",
) -> Optional[MergeableVectorState]:
    """Load a MergeableVectorState from the root node metadata."""
    payload = tree.root.metadata.get(state_key) if tree.root is not None else None
    if not isinstance(payload, dict):
        return None
    vec = payload.get("sum", None)
    count = payload.get("count", None)
    if not isinstance(vec, list) or not vec:
        return None
    try:
        count_i = int(count)
    except (TypeError, ValueError):
        return None
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] <= 0:
        return None
    return MergeableVectorState(sum_vec=arr, count=count_i)


def build_mergeable_phi_state(
    tree: Tree,
    *,
    model: Any,
    leaf_embedding_key: str = "leaf_embedding",
    state_key: str = "phi_sum_state",
    overwrite: bool = False,
    device: str = "cpu",
) -> None:
    """
    Build a strictly mergeable (sum_phi, count) state using model.phi().

    This matches the MergeableEmbeddingSketch definition:
      state = (sum_i phi(e_i), count)
      merge = add

    Stores state dicts under Node.metadata[state_key]:
        {"sum": [...], "count": int}

    Notes:
      - This does not apply the model readout; it only computes the mergeable state.
      - The model is treated as an opaque object with a `.phi` module.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for build_mergeable_phi_state") from exc

    leaves = tree.leaves
    target: List[Node] = []
    embs: List[np.ndarray] = []
    for leaf in leaves:
        if (not overwrite) and state_key in leaf.metadata:
            continue
        vec = leaf.metadata.get(leaf_embedding_key)
        if not isinstance(vec, list) or not vec:
            continue
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] <= 0:
            continue
        target.append(leaf)
        embs.append(arr)

    if target:
        mat = np.stack(embs, axis=0).astype(np.float32, copy=False)
        with torch.no_grad():
            tensor = torch.from_numpy(mat).to(device=device, dtype=torch.float32)
            phi = model.phi(tensor).detach().cpu().numpy().astype(np.float32, copy=False)
        if phi.ndim != 2 or phi.shape[0] != len(target) or phi.shape[1] <= 0:
            raise RuntimeError("Unexpected phi output shape")
        for leaf, row in zip(target, phi):
            leaf.metadata[state_key] = {"sum": [float(x) for x in row.tolist()], "count": 1}

    # Merge bottom-up.
    for node in tree.traverse_postorder():
        if node.is_leaf:
            continue
        if (not overwrite) and state_key in node.metadata:
            continue
        left = node.left_child
        right = node.right_child
        if left is None or right is None:
            continue
        left_state = left.metadata.get(state_key)
        right_state = right.metadata.get(state_key)
        if not isinstance(left_state, dict) or not isinstance(right_state, dict):
            continue
        l_sum = np.asarray(left_state.get("sum", []), dtype=np.float32)
        r_sum = np.asarray(right_state.get("sum", []), dtype=np.float32)
        if l_sum.ndim != 1 or r_sum.ndim != 1 or l_sum.shape[0] == 0 or r_sum.shape[0] == 0:
            continue
        if int(l_sum.shape[0]) != int(r_sum.shape[0]):
            raise ValueError("Phi state dimension mismatch while merging")
        try:
            l_count = int(left_state.get("count", 0))
            r_count = int(right_state.get("count", 0))
        except (TypeError, ValueError):
            continue
        merged_sum = l_sum + r_sum
        node.metadata[state_key] = {"sum": [float(x) for x in merged_sum.tolist()], "count": int(l_count + r_count)}


def predict_root_from_phi_state(
    tree: Tree,
    *,
    model: Any,
    state_key: str = "phi_sum_state",
    meta_embedding: Optional[Sequence[float]] = None,
    device: str = "cpu",
) -> Optional[float]:
    """
    Predict normalized score in [0,1] from the root phi-state using model.readout.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for predict_root_from_phi_state") from exc

    payload = tree.root.metadata.get(state_key) if tree.root is not None else None
    if not isinstance(payload, dict):
        return None
    vec = payload.get("sum", None)
    count = payload.get("count", None)
    if not isinstance(vec, list) or not vec:
        return None
    try:
        count_i = int(count)
    except (TypeError, ValueError):
        return None

    sum_phi = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)  # [1,M]
    count_t = torch.tensor([float(count_i)], dtype=torch.float32, device=device)  # [1]

    from treepo._research.training.embedding_sketch import SketchState

    state = SketchState(sum_phi=sum_phi, count=count_t)

    meta_t = None
    if meta_embedding is not None:
        meta_arr = np.asarray(list(meta_embedding), dtype=np.float32)
        if meta_arr.ndim == 1 and meta_arr.shape[0] > 0:
            meta_t = torch.from_numpy(meta_arr).to(device=device, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        out = model.predict_from_state(state, meta_embeddings=meta_t)
        return float(out.detach().cpu().numpy().reshape(-1)[0])
