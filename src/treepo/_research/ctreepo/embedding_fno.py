"""Embedding-coordinate FNO distillation for tree-indexed labels.

This backend treats each embedding vector as a 1D signal over its coordinate
axis.  The leaf-count grid controls tree topology only; the FNO resolution is
the embedding dimension, e.g. 1024 for Qwen3-Embedding-0.6B.

FNO channel invariant (load-bearing):
- ``leaf_fno`` (f): ``in_channels=1``, ``out_channels=1``; operates on
  ``(batch, 1, embedding_dim)``.
- ``merge_fno`` (g): ``in_channels=2``, ``out_channels=1``; operates on
  ``(batch, 2, embedding_dim)`` = concat of two child embeddings along a new
  channel axis, producing a single embedding-dim-wide output.

This invariant is what makes ``merge(concat(a, b))`` literally "concatenate two
embeddings and produce one embedding", and what lets identity init
(``merge(a, a) = a``) be well-defined. ``state_channels`` is intentionally NOT
a tunable parameter of this module; ``hidden_channels`` inside the FNO is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from treepo._research.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r
from treepo._research.training.config_sections import (
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
    TestConfig,
    TrainConfig,
    ValidationConfig,
    config_to_dict,
)
from treepo._research.tree.labeled import LabeledNode, LabeledTree
from treepo._research.tree.state_tree import (
    explicit_oracle_trace_kwargs,
    local_law_trace_metadata,
    state_tree_skeleton_from_labeled_tree,
    state_tree_trace_metrics,
    update_state_tree_node,
    write_state_trees_jsonl,
)

import logging as _logging
_LOGGER = _logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class EmbeddingFNOModelConfig:
    hidden_channels: int = 32
    n_modes: int = 64
    n_layers: int = 2
    head_hidden_dim: int = 64
    target_min: float = 1.0
    target_max: float = 7.0


@dataclass(frozen=True, kw_only=True)
class EmbeddingFNOObjectiveConfig:
    root_weight: float = 1.0
    leaf_weight: float = 0.5
    merge_weight: float = 0.5


@dataclass(frozen=True, kw_only=True)
class EmbeddingFNOTrainConfig:
    run: RunConfig = field(default_factory=RunConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            learning_rate=1e-3,
            weight_decay=1e-4,
            optimizer="adamw",
            grad_clip_norm=1.0,
        )
    )
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    test: TestConfig = field(default_factory=TestConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: EmbeddingFNOModelConfig = field(default_factory=EmbeddingFNOModelConfig)
    objective: EmbeddingFNOObjectiveConfig = field(default_factory=EmbeddingFNOObjectiveConfig)


@dataclass
class EmbeddingFNOFitResult:
    output_dir: str
    embedding_dim: int
    train_count: int
    val_count: int
    test_count: int
    metrics: Dict[str, Any]
    artifacts: Dict[str, str]
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return config_to_dict(self)


@dataclass
class _PreparedTree:
    tree: LabeledTree
    split: str
    leaf_embeddings: torch.Tensor
    node_order: List[str]
    leaf_ranges: Dict[str, Tuple[int, int]]
    root_node_id: str


class EmbeddingCoordinateFNOTreeRegressor(nn.Module):
    """Tree model whose leaf and merge operators are FNOs over embedding dims."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        hidden_channels: int,
        n_modes: int,
        n_layers: int,
        head_hidden_dim: int,
        target_min: float,
        target_max: float,
    ) -> None:
        super().__init__()
        from neuralop.models import FNO

        self.embedding_dim = int(embedding_dim)
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        modes = max(1, min(int(n_modes), int(embedding_dim)))
        self.leaf_norm = nn.LayerNorm(int(embedding_dim))
        # f operator: (B, 1, embedding_dim) -> (B, 1, embedding_dim).
        self.leaf_fno = FNO(
            n_modes=(modes,),
            in_channels=1,
            out_channels=1,
            hidden_channels=int(hidden_channels),
            n_layers=int(n_layers),
        )
        # g operator: (B, 2, embedding_dim) -> (B, 1, embedding_dim).
        self.merge_fno = FNO(
            n_modes=(modes,),
            in_channels=2,
            out_channels=1,
            hidden_channels=int(hidden_channels),
            n_layers=int(n_layers),
        )
        self.score_head = nn.Sequential(
            nn.Linear(int(embedding_dim), int(head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(head_hidden_dim), 1),
        )
        _LOGGER.info(
            "FNO invariant: embedding_dim=%d, leaf_fno 1->1, merge_fno 2->1, "
            "hidden_channels=%d, n_modes=%d, n_layers=%d, head_hidden_dim=%d",
            int(embedding_dim), int(hidden_channels), int(n_modes),
            int(n_layers), int(head_hidden_dim),
        )

    def encode_leaves(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Residual bypass around the FNO: at identity init (zeroed FNO weights
        # and leaf_norm weight=1/bias=0), the output equals the raw input
        # embedding. Trained, the FNO learns a residual on top of the embedding.
        raw = embeddings.unsqueeze(1)
        normalized = self.leaf_norm(embeddings).unsqueeze(1)
        return raw + self.leaf_fno(normalized)

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        # Residual bypass around the merge FNO: at identity init the output is
        # 0.5 * (left + right); the FNO learns a residual correction.
        avg = 0.5 * (left + right)
        residual = self.merge_fno(torch.cat([left, right], dim=1))
        return avg + residual

    def predict_normalized(self, state: torch.Tensor) -> torch.Tensor:
        flat = state.squeeze(1)
        logits = self.score_head(flat).reshape(-1)
        return torch.sigmoid(logits)

    def predict_raw(self, state: torch.Tensor) -> torch.Tensor:
        norm = self.predict_normalized(state)
        return self.target_min + norm * (self.target_max - self.target_min)

    @torch.no_grad()
    def initialize_as_identity(self) -> None:
        """Set weights so the f/g paths reduce to the invariant's baseline.

        After this call:
        - ``encode_leaves(x)`` equals ``x.unsqueeze(1)`` for any ``x``.
        - ``merge(a, b)`` equals ``0.5 * (a + b)`` for any ``a``, ``b``.
        - ``predict_normalized`` returns 0.5 (mid-range), so ``predict_raw``
          returns ``target_min + 0.5 * (target_max - target_min)``.

        Subsequent training lets the FNOs learn residual corrections on top
        of these baselines, and the score head to move away from 0.5.
        """
        for p in self.leaf_fno.parameters():
            p.zero_()
        for p in self.merge_fno.parameters():
            p.zero_()
        nn.init.ones_(self.leaf_norm.weight)
        nn.init.zeros_(self.leaf_norm.bias)
        for m in self.score_head:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def _set_requires_grad(self, module: nn.Module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad = bool(flag)

    def freeze_for_f_training(self) -> None:
        """Train f-path params (leaf_fno + leaf_norm + score_head); freeze merge_fno."""
        self._set_requires_grad(self.leaf_fno, True)
        self._set_requires_grad(self.leaf_norm, True)
        self._set_requires_grad(self.score_head, True)
        self._set_requires_grad(self.merge_fno, False)

    def freeze_for_g_training(self) -> None:
        """Train g-path params (merge_fno only); freeze leaf_fno, leaf_norm, score_head."""
        self._set_requires_grad(self.leaf_fno, False)
        self._set_requires_grad(self.leaf_norm, False)
        self._set_requires_grad(self.score_head, False)
        self._set_requires_grad(self.merge_fno, True)

    def unfreeze_all(self) -> None:
        self._set_requires_grad(self, True)


def _normalize_score(value: float, *, target_min: float, target_max: float) -> float:
    span = float(target_max) - float(target_min)
    if span <= 0.0:
        return 0.5
    return max(0.0, min(1.0, (float(value) - float(target_min)) / span))


def _denormalize_score(value: float, *, target_min: float, target_max: float) -> float:
    return float(target_min) + max(0.0, min(1.0, float(value))) * (float(target_max) - float(target_min))


def _tree_split(tree: LabeledTree) -> str:
    return str((tree.metadata or {}).get("split", "") or "")


def _select_trees(trees: Sequence[LabeledTree], splits: Sequence[str]) -> List[LabeledTree]:
    keys = {str(value).lower() for value in splits}
    return [tree for tree in trees if _tree_split(tree).lower() in keys]


def _ordered_nodes(tree: LabeledTree) -> List[LabeledNode]:
    out: List[LabeledNode] = []
    seen: set[str] = set()
    for level_ids in list(tree.levels or []):
        for node_id in level_ids:
            node = tree.get_node(str(node_id))
            if node is not None and str(node.node_id) not in seen:
                out.append(node)
                seen.add(str(node.node_id))
    for node in sorted(tree.nodes.values(), key=lambda item: (int(item.level), str(item.node_id))):
        if str(node.node_id) not in seen:
            out.append(node)
            seen.add(str(node.node_id))
    return out


def _node_leaf_ranges(tree: LabeledTree) -> Dict[str, Tuple[int, int]]:
    leaves = list(tree.levels[0] if tree.levels else [])
    leaf_index = {str(node_id): idx for idx, node_id in enumerate(leaves)}
    memo: Dict[str, Tuple[int, int]] = {}

    def visit(node_id: str) -> Tuple[int, int]:
        node_id = str(node_id)
        if node_id in memo:
            return memo[node_id]
        node = tree.get_node(node_id)
        if node is None:
            raise ValueError(f"missing node id {node_id!r} in tree {tree.doc_id!r}")
        if int(node.level) == 0 or not node.left_child_id:
            idx = leaf_index[node_id]
            memo[node_id] = (idx, idx + 1)
            return memo[node_id]
        left = visit(str(node.left_child_id))
        right = visit(str(node.right_child_id or node.left_child_id))
        memo[node_id] = (min(left[0], right[0]), max(left[1], right[1]))
        return memo[node_id]

    for node in _ordered_nodes(tree):
        visit(str(node.node_id))
    return memo


def _prepare_trees(
    trees: Sequence[LabeledTree],
    *,
    embedding_client: Any,
    embedding_max_tokens: Optional[int] = None,
    chunks_per_leaf: int = 1,
    tokenizer_model_path: Optional[str] = None,
    enforce_no_truncation: bool = True,
) -> Tuple[List[_PreparedTree], int]:
    """Embed each leaf into a fixed-width coordinate vector.

    Per the no-truncation invariant: if a leaf is larger than the embedding
    model's max token length, split it into fixed chunk slots and concatenate
    those embeddings along the FNO spatial axis. This preserves the 1-channel
    f / 2-channel g invariant while allowing ``D_eff = K * D`` for future
    smaller-context embedding backends.
    """
    prepared: List[_PreparedTree] = []
    embedding_dim: Optional[int] = None
    chunks_per_leaf = max(1, int(chunks_per_leaf))
    if enforce_no_truncation and embedding_max_tokens is not None:
        from treepo._research.preprocessing.leaf_size_utils import (
            assert_no_truncation,
            char_windows_from_token_budget,
        )
    else:
        assert_no_truncation = None  # type: ignore[assignment]
        char_windows_from_token_budget = None  # type: ignore[assignment]
    for tree in trees:
        leaves = [tree.get_node(str(node_id)) for node_id in (tree.levels[0] if tree.levels else [])]
        leaf_nodes = [node for node in leaves if node is not None]
        if not leaf_nodes:
            continue
        leaf_texts = [str(node.text or "") for node in leaf_nodes]
        leaf_chunks: List[List[str]] = []
        for idx, text in enumerate(leaf_texts):
            if char_windows_from_token_budget is not None and embedding_max_tokens is not None:
                windows = char_windows_from_token_budget(
                    text,
                    int(embedding_max_tokens),
                    model_path=tokenizer_model_path,
                )
                chunks = [text[int(start): int(end)] for start, end in windows]
            else:
                chunks = [text]
            if len(chunks) > chunks_per_leaf:
                raise RuntimeError(
                    f"silent truncation in _prepare_trees: tree={tree.doc_id!r} leaf_idx={idx} "
                    f"needs {len(chunks)} embedding chunks but chunks_per_leaf={chunks_per_leaf}. "
                    "Increase leaf_size_tokens/chunks_per_leaf alignment or reduce leaf size."
                )
            if assert_no_truncation is not None and embedding_max_tokens is not None:
                for chunk_idx, chunk in enumerate(chunks):
                    try:
                        assert_no_truncation(
                            chunk,
                            max_tokens=int(embedding_max_tokens),
                            model_path=tokenizer_model_path,
                        )
                    except RuntimeError as exc:
                        raise RuntimeError(
                            f"silent truncation in _prepare_trees: tree={tree.doc_id!r} "
                            f"leaf_idx={idx} chunk_idx={chunk_idx} would overflow "
                            f"embedding_max_tokens={embedding_max_tokens}. Underlying error: {exc}"
                        ) from exc
            leaf_chunks.append(chunks or [""])
        flat_chunks = [chunk for chunks in leaf_chunks for chunk in chunks]
        chunk_embeddings = embedding_client.embed_texts(flat_chunks)
        if not chunk_embeddings:
            continue
        base_dim = int(len(chunk_embeddings[0]))
        if any(int(len(vec)) != base_dim for vec in chunk_embeddings):
            raise ValueError(f"embedding dimension changed across chunks for tree {tree.doc_id!r}")
        leaf_vectors: List[List[float]] = []
        cursor = 0
        zero = [0.0] * base_dim
        for chunks in leaf_chunks:
            count = len(chunks)
            parts = [list(vec) for vec in chunk_embeddings[cursor: cursor + count]]
            cursor += count
            while len(parts) < chunks_per_leaf:
                parts.append(list(zero))
            leaf_vectors.append([value for part in parts for value in part])
        tensor = torch.tensor(leaf_vectors, dtype=torch.float32)
        if tensor.ndim != 2:
            raise ValueError(f"embedding client returned non-matrix embeddings for {tree.doc_id!r}")
        if embedding_dim is None:
            embedding_dim = int(tensor.shape[1])
        elif int(tensor.shape[1]) != int(embedding_dim):
            raise ValueError(
                f"embedding dimension changed from {embedding_dim} to {tensor.shape[1]} "
                f"for tree {tree.doc_id!r}"
            )
        node_order = [str(node.node_id) for node in _ordered_nodes(tree)]
        root_node_id = str(tree.levels[-1][0]) if tree.levels and tree.levels[-1] else node_order[-1]
        prepared.append(
            _PreparedTree(
                tree=tree,
                split=_tree_split(tree),
                leaf_embeddings=tensor,
                node_order=node_order,
                leaf_ranges=_node_leaf_ranges(tree),
                root_node_id=root_node_id,
            )
        )
    if embedding_dim is None:
        raise ValueError("No trees could be embedded")
    return prepared, int(embedding_dim)


def _device_from_runtime(runtime: RuntimeConfig) -> torch.device:
    requested = str(runtime.device or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _forward_tree_states(
    model: EmbeddingCoordinateFNOTreeRegressor,
    item: _PreparedTree,
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    leaf_ids = list(item.tree.levels[0] if item.tree.levels else [])
    leaf_states = model.encode_leaves(item.leaf_embeddings.to(device))
    states: Dict[str, torch.Tensor] = {}
    for idx, node_id in enumerate(leaf_ids):
        states[str(node_id)] = leaf_states[idx : idx + 1]
    for node_id in item.node_order:
        if node_id in states:
            continue
        node = item.tree.get_node(node_id)
        if node is None:
            continue
        left = states[str(node.left_child_id)]
        right = states[str(node.right_child_id or node.left_child_id)]
        states[node_id] = model.merge(left, right)
    return states


def _node_weight(
    node: LabeledNode,
    *,
    root_node_id: str,
    objective: EmbeddingFNOObjectiveConfig,
) -> float:
    if str(node.node_id) == str(root_node_id):
        return float(objective.root_weight)
    if int(node.level) == 0:
        return float(objective.leaf_weight)
    return float(objective.merge_weight)


def _batch_loss(
    model: EmbeddingCoordinateFNOTreeRegressor,
    batch: Sequence[_PreparedTree],
    *,
    device: torch.device,
    cfg: EmbeddingFNOTrainConfig,
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    target_min = float(cfg.model.target_min)
    target_max = float(cfg.model.target_max)
    for item in batch:
        states = _forward_tree_states(model, item, device=device)
        for node_id in item.node_order:
            node = item.tree.get_node(node_id)
            if node is None:
                continue
            weight = _node_weight(node, root_node_id=item.root_node_id, objective=cfg.objective)
            if weight <= 0.0:
                continue
            pred_norm = model.predict_normalized(states[node_id]).reshape(())
            target_norm = torch.tensor(
                _normalize_score(float(node.score), target_min=target_min, target_max=target_max),
                dtype=torch.float32,
                device=device,
            )
            losses.append(float(weight) * F.mse_loss(pred_norm, target_norm))
    if not losses:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(losses).mean()


@torch.no_grad()
def _evaluate_split(
    model: EmbeddingCoordinateFNOTreeRegressor,
    items: Sequence[_PreparedTree],
    *,
    device: torch.device,
    cfg: EmbeddingFNOTrainConfig,
    output_path: Optional[Path] = None,
    full_tree_trace_path: Optional[Path] = None,
) -> Dict[str, Any]:
    model.eval()
    target_min = float(cfg.model.target_min)
    target_max = float(cfg.model.target_max)
    rows: List[Dict[str, Any]] = []
    root_preds: List[float] = []
    root_targets: List[float] = []
    root_experts: List[float] = []
    node_errors: List[float] = []
    leaf_errors: List[float] = []
    merge_errors: List[float] = []
    root_errors: List[float] = []
    full_tree_traces = []

    for item in items:
        states = _forward_tree_states(model, item, device=device)
        trace = state_tree_skeleton_from_labeled_tree(
            item.tree,
            method_family="embedding_fno",
            state_kind="embedding_fno_state",
            split=item.split,
        )
        expert_score = (item.tree.metadata or {}).get("expert_score_1_7")
        try:
            expert_value = float(expert_score)
        except (TypeError, ValueError):
            expert_value = float("nan")
        for node_id in item.node_order:
            node = item.tree.get_node(node_id)
            if node is None:
                continue
            pred_norm = float(model.predict_normalized(states[node_id]).detach().cpu().reshape(()).item())
            pred_raw = _denormalize_score(pred_norm, target_min=target_min, target_max=target_max)
            target_raw = float(node.score)
            error = abs(float(pred_raw) - float(target_raw))
            proxy_loss = float((pred_raw - target_raw) ** 2)
            is_root = str(node_id) == str(item.root_node_id)
            is_leaf = int(node.level) == 0
            oracle_kwargs = explicit_oracle_trace_kwargs(getattr(node, "metadata", {}) or {})
            law_metadata = local_law_trace_metadata(
                prediction=float(pred_raw),
                proxy_target=float(target_raw),
                proxy_loss=float(proxy_loss),
                oracle_target=oracle_kwargs["oracle_target"],
                oracle_loss=oracle_kwargs["oracle_loss"],
                observed=bool(oracle_kwargs["observed"]),
                sampled=bool(oracle_kwargs["sampled"]),
                propensity=oracle_kwargs["propensity"],
                node_weight=float(_node_weight(node, root_node_id=item.root_node_id, objective=cfg.objective)),
                law_channel="root" if is_root else ("leaf" if is_leaf else "merge"),
                state_kind="embedding_fno_state",
                label_source=str(oracle_kwargs["label_source"] or "proxy_score"),
            )
            node_errors.append(error)
            if is_root:
                root_errors.append(error)
                root_preds.append(float(pred_raw))
                root_targets.append(float(target_raw))
                if math.isfinite(expert_value):
                    root_experts.append(float(expert_value))
            elif is_leaf:
                leaf_errors.append(error)
            else:
                merge_errors.append(error)
            lo, hi = item.leaf_ranges.get(str(node_id), (0, 0))
            update_state_tree_node(
                trace,
                str(node_id),
                rendered=str(node.text or ""),
                state=states[node_id].detach().cpu(),
                metadata={
                    "prediction": float(pred_raw),
                    "readout_prediction": float(pred_raw),
                    "prediction_normalized": float(pred_norm),
                    "target": float(target_raw),
                    "target_1_7": float(target_raw),
                    **law_metadata,
                    "leaf_range": [int(lo), int(hi)],
                    "expert_score_1_7": expert_value if math.isfinite(expert_value) else None,
                },
            )
            rows.append(
                {
                    "doc_id": item.tree.doc_id,
                    "split": item.split,
                    "node_id": str(node_id),
                    "level": int(node.level),
                    "is_leaf": bool(is_leaf),
                    "is_root": bool(is_root),
                    "leaf_range": [int(lo), int(hi)],
                    "target_1_7": target_raw,
                    "prediction_1_7": float(pred_raw),
                    "abs_error_1_7": float(error),
                    "expert_score_1_7": expert_value if math.isfinite(expert_value) else None,
                }
            )
        full_tree_traces.append(trace)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(config_to_dict(row), sort_keys=True) + "\n")
    if full_tree_trace_path is not None:
        write_state_trees_jsonl(full_tree_traces, full_tree_trace_path)

    def _mae(values: Sequence[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    root_report = compute_corpus_pearson_r(root_preds, root_targets).as_dict() if len(root_preds) >= 4 else {"n": len(root_preds)}
    root_report["mae_1_7"] = _mae(root_errors)
    expert_report: Dict[str, Any]
    if len(root_experts) == len(root_preds) and len(root_preds) >= 4:
        expert_report = compute_corpus_pearson_r(root_preds, root_experts).as_dict()
    else:
        expert_report = {"n": min(len(root_preds), len(root_experts))}
    if root_experts and len(root_experts) == len(root_preds):
        expert_report["mae_1_7"] = float(np.mean([abs(p - t) for p, t in zip(root_preds, root_experts)]))
    else:
        expert_report["mae_1_7"] = None

    return {
        "count_trees": int(len(items)),
        "count_nodes": int(len(rows)),
        "node_mae_1_7": _mae(node_errors),
        "leaf_mae_1_7": _mae(leaf_errors),
        "merge_mae_1_7": _mae(merge_errors),
        "root_teacher_report": root_report,
        "root_expert_report": expert_report,
        "prediction_path": str(output_path) if output_path is not None else None,
        "full_tree_trace_path": (
            str(full_tree_trace_path) if full_tree_trace_path is not None else None
        ),
        "full_tree_trace_metrics": state_tree_trace_metrics(full_tree_traces),
    }


def fit_embedding_fno_node_regressor(
    labeled_trees: Sequence[LabeledTree],
    *,
    embedding_client: Any,
    config: Optional[EmbeddingFNOTrainConfig] = None,
) -> EmbeddingFNOFitResult:
    """Fit an embedding-coordinate FNO against tree node labels."""

    if embedding_client is None:
        raise ValueError("embedding_client is required for embedding-FNO fitting")
    cfg = config or EmbeddingFNOTrainConfig()
    output_dir = Path(cfg.run.output_dir or "outputs/embedding_fno_fit")
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(int(cfg.run.seed))
    np.random.seed(int(cfg.run.seed))
    torch.manual_seed(int(cfg.run.seed))
    device = _device_from_runtime(cfg.runtime)

    train_trees = _select_trees(labeled_trees, cfg.train.train_splits)
    val_trees = _select_trees(labeled_trees, cfg.validation.val_splits) if cfg.validation.enabled else []
    test_trees = _select_trees(labeled_trees, cfg.test.test_splits) if cfg.test.enabled else []
    if not train_trees:
        train_trees = list(labeled_trees)

    all_selected = list(train_trees) + list(val_trees) + list(test_trees)
    prepared_all, embedding_dim = _prepare_trees(all_selected, embedding_client=embedding_client)
    by_doc_split = {(item.tree.doc_id, item.split): item for item in prepared_all}
    train_items = [by_doc_split[(tree.doc_id, _tree_split(tree))] for tree in train_trees if (tree.doc_id, _tree_split(tree)) in by_doc_split]
    val_items = [by_doc_split[(tree.doc_id, _tree_split(tree))] for tree in val_trees if (tree.doc_id, _tree_split(tree)) in by_doc_split]
    test_items = [by_doc_split[(tree.doc_id, _tree_split(tree))] for tree in test_trees if (tree.doc_id, _tree_split(tree)) in by_doc_split]

    model = EmbeddingCoordinateFNOTreeRegressor(
        embedding_dim=embedding_dim,
        hidden_channels=int(cfg.model.hidden_channels),
        n_modes=int(cfg.model.n_modes),
        n_layers=int(cfg.model.n_layers),
        head_hidden_dim=int(cfg.model.head_hidden_dim),
        target_min=float(cfg.model.target_min),
        target_max=float(cfg.model.target_max),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optimizer.learning_rate),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    grad_clip = float(cfg.optimizer.grad_clip_norm or 0.0)
    best_val = float("inf")
    best_epoch = -1
    losses: List[Dict[str, Any]] = []
    start = time.time()

    for epoch in range(int(cfg.train.epochs)):
        model.train()
        order = list(range(len(train_items)))
        if cfg.train.shuffle:
            random.shuffle(order)
        epoch_losses: List[float] = []
        for start_idx in range(0, len(order), int(max(1, cfg.train.batch_size))):
            batch = [train_items[idx] for idx in order[start_idx : start_idx + int(max(1, cfg.train.batch_size))]]
            optimizer.zero_grad()
            loss = _batch_loss(model, batch, device=device, cfg=cfg)
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        epoch_payload: Dict[str, Any] = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
        }
        if val_items and ((epoch + 1) % int(max(1, cfg.validation.eval_every)) == 0 or epoch == int(cfg.train.epochs) - 1):
            val_metrics = _evaluate_split(model, val_items, device=device, cfg=cfg)
            val_mae = val_metrics.get("node_mae_1_7")
            epoch_payload["val_node_mae_1_7"] = val_mae
            if val_mae is not None and float(val_mae) < best_val:
                best_val = float(val_mae)
                best_epoch = int(epoch)
                torch.save(model.state_dict(), output_dir / "embedding_fno_best.pt")
        losses.append(epoch_payload)

    if best_epoch < 0:
        torch.save(model.state_dict(), output_dir / "embedding_fno_best.pt")
    torch.save(model.state_dict(), output_dir / "embedding_fno_final.pt")

    metrics = {
        "train": _evaluate_split(
            model,
            train_items,
            device=device,
            cfg=cfg,
            output_path=output_dir / "node_predictions_train.jsonl",
            full_tree_trace_path=output_dir / "full_tree_traces_train.jsonl",
        ),
        "val": _evaluate_split(
            model,
            val_items,
            device=device,
            cfg=cfg,
            output_path=output_dir / "node_predictions_val.jsonl",
            full_tree_trace_path=output_dir / "full_tree_traces_val.jsonl",
        ) if val_items else {},
        "test": _evaluate_split(
            model,
            test_items,
            device=device,
            cfg=cfg,
            output_path=output_dir / "node_predictions_test.jsonl",
            full_tree_trace_path=output_dir / "full_tree_traces_test.jsonl",
        ) if test_items else {},
        "losses": losses,
        "best_epoch": int(best_epoch),
        "best_val_node_mae_1_7": None if not math.isfinite(best_val) else float(best_val),
        "training_time_seconds": float(time.time() - start),
    }
    canonical_trace_path = output_dir / "full_tree_traces_test.jsonl"
    if not test_items and val_items:
        canonical_trace_path = output_dir / "full_tree_traces_val.jsonl"
    if not test_items and not val_items:
        canonical_trace_path = output_dir / "full_tree_traces_train.jsonl"
    artifacts = {
        "best_checkpoint": str(output_dir / "embedding_fno_best.pt"),
        "final_checkpoint": str(output_dir / "embedding_fno_final.pt"),
        "train_predictions": str(output_dir / "node_predictions_train.jsonl"),
        "val_predictions": str(output_dir / "node_predictions_val.jsonl"),
        "test_predictions": str(output_dir / "node_predictions_test.jsonl"),
        "train_full_tree_traces": str(output_dir / "full_tree_traces_train.jsonl"),
        "val_full_tree_traces": str(output_dir / "full_tree_traces_val.jsonl"),
        "test_full_tree_traces": str(output_dir / "full_tree_traces_test.jsonl"),
        "full_tree_traces_jsonl": str(canonical_trace_path),
        "metrics": str(output_dir / "embedding_fno_metrics.json"),
        "full_tree_metrics_json": str(output_dir / "embedding_fno_metrics.json"),
    }
    result = EmbeddingFNOFitResult(
        output_dir=str(output_dir),
        embedding_dim=int(embedding_dim),
        train_count=len(train_items),
        val_count=len(val_items),
        test_count=len(test_items),
        metrics=metrics,
        artifacts=artifacts,
        config=config_to_dict(cfg),
    )
    (output_dir / "embedding_fno_metrics.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "EmbeddingCoordinateFNOTreeRegressor",
    "EmbeddingFNOFitResult",
    "EmbeddingFNOModelConfig",
    "EmbeddingFNOObjectiveConfig",
    "EmbeddingFNOTrainConfig",
    "fit_embedding_fno_node_regressor",
]
