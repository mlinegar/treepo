"""FNO backend family for the alternating f/g optimization loop.

In this family, f and g share a single ``EmbeddingCoordinateFNOTreeRegressor``
instance because the model's parameters partition cleanly: ``leaf_fno +
leaf_norm + score_head`` are the f-path, ``merge_fno`` is the g-path. The
``FArtifact`` / ``GArtifact`` returned to the alternating trampoline is the
same underlying state_dict at each step — we just toggle which half trains
via ``freeze_for_{f,g}_training()``.

Supervision:
- ``train_f``: MSE between student f-predictions and teacher f-scores at every
  node (per-node supervision, unchanged from the legacy FNO trainer).
- ``train_g``: MSE between student-f applied to student-g-merged state and
  teacher f-scores at merge nodes. The scoring signal flows through the
  frozen current student f exactly as the alternating-semantics rule requires.

Identity init (``EmbeddingCoordinateFNOTreeRegressor.initialize_as_identity``)
makes the k=0 (``fg``) iteration a neutral baseline: every prediction is 4.0
(the 1-7 midpoint), consistent with ``leaf_fno = id`` and ``merge = avg``.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from treepo._research.ctreepo.alternating import FamilyRuntime
from treepo._research.ctreepo.embedding_fno import (
    EmbeddingCoordinateFNOTreeRegressor,
    _PreparedTree,
    _forward_tree_states,
    _ordered_nodes,
    _prepare_trees,
)
from treepo._research.tree.labeled import LabeledNode, LabeledTree
from treepo._research.tree.state_tree import (
    StateTree,
    explicit_oracle_trace_kwargs,
    local_law_trace_metadata,
    state_tree_skeleton_from_labeled_tree,
    state_tree_trace_metrics,
    update_state_tree_node,
    write_state_trees_jsonl,
)

LOGGER = logging.getLogger(__name__)

_FNO_IDENTITY_STATE_SENTINELS = frozenset({"identity", "raw_concat"})


def _is_identity_state_artifact(artifact: Any) -> bool:
    if artifact is None:
        return True
    return str(artifact).strip().lower() in _FNO_IDENTITY_STATE_SENTINELS


def _normalize_01(value: float, *, lo: float, hi: float) -> float:
    span = float(hi - lo)
    if span <= 0.0:
        return 0.5
    return max(0.0, min(1.0, (float(value) - float(lo)) / span))


def _denormalize(value: float, *, lo: float, hi: float) -> float:
    return float(lo) + max(0.0, min(1.0, float(value))) * float(hi - lo)


@dataclass
class FNOFamilyConfig:
    hidden_channels: int = 32
    n_modes: int = 64
    n_layers: int = 2
    head_hidden_dim: int = 64
    target_min: float = 1.0
    target_max: float = 7.0
    #: Number of epochs per ``train_f`` / ``train_g`` call.
    epochs_per_iteration: int = 8
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    #: Relative weighting of leaf / merge / root nodes in f-training (single-model MSE).
    root_weight: float = 1.0
    leaf_weight: float = 0.5
    merge_weight: float = 0.5
    #: If True, run ``initialize_as_identity()`` on first use.
    identity_init: bool = True
    seed: int = 42
    #: Configured leaf size in tokenizer tokens. This is the canonical row axis.
    leaf_size_tokens: int = 512
    #: Embedding model's max input length in tokens. When set, ``_prepare_trees``
    #: chunks a leaf into fixed slots if leaf_size_tokens exceeds this value,
    #: and otherwise hard-errors on any malformed oversized leaf.
    #: Default 2048 matches Google EmbeddingGemma-300m's max_position_embeddings.
    #: Set to ``None`` to disable the no-truncation enforcement (legacy / smoke).
    embedding_max_length_tokens: Optional[int] = 2048
    tokenizer_model_path: str = "/mnt/data/models/google/embeddinggemma-300m"
    #: Optional expected FNO spatial width after any within-leaf concatenation.
    #: For the EmbeddingGemma defaults this is 768. If a future embedding model
    #: has max length smaller than leaf_size_tokens, callers should set this to
    #: ``ceil(leaf_size_tokens / embedding_max_length_tokens) * base_embedding_dim``.
    effective_embedding_dim: Optional[int] = 768
    #: Canonical TreeBundle state-shape contract. ``summary_dim`` is the input
    #: representation width; ``state_dim`` must be wide enough to carry a pure
    #: concatenation of two child summaries/states. The legacy
    #: ``EmbeddingCoordinateFNOTreeRegressor`` still uses averaged
    #: embedding-width states internally, but publication-facing configs must
    #: satisfy this contract before they are accepted.
    summary_dim: Optional[int] = None
    state_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if int(self.leaf_size_tokens) <= 0:
            raise ValueError(
                f"leaf_size_tokens must be positive, got {self.leaf_size_tokens}"
            )
        if (
            self.embedding_max_length_tokens is not None
            and int(self.embedding_max_length_tokens) <= 0
        ):
            raise ValueError(
                "embedding_max_length_tokens must be positive when set, "
                f"got {self.embedding_max_length_tokens}"
            )
        if (
            self.effective_embedding_dim is not None
            and int(self.effective_embedding_dim) <= 0
        ):
            raise ValueError(
                "effective_embedding_dim must be positive when set, "
                f"got {self.effective_embedding_dim}"
            )
        if self.summary_dim is None and self.effective_embedding_dim is not None:
            self.summary_dim = int(self.effective_embedding_dim)
        if self.state_dim is None and self.summary_dim is not None:
            self.state_dim = 2 * int(self.summary_dim)
        if self.summary_dim is not None and int(self.summary_dim) <= 0:
            raise ValueError(f"summary_dim must be positive, got {self.summary_dim}")
        if self.state_dim is not None and int(self.state_dim) <= 0:
            raise ValueError(f"state_dim must be positive, got {self.state_dim}")
        if self.summary_dim is not None and self.state_dim is not None:
            if int(self.state_dim) < 2 * int(self.summary_dim):
                raise ValueError(
                    "state_dim must be at least 2 * summary_dim for canonical "
                    f"TreeBundle/FNO configs, got state_dim={self.state_dim}, "
                    f"summary_dim={self.summary_dim}"
                )

    @property
    def chunks_per_leaf(self) -> int:
        if self.embedding_max_length_tokens is None:
            return 1
        return max(
            1,
            int(
                math.ceil(
                    int(self.leaf_size_tokens) / float(int(self.embedding_max_length_tokens))
                )
            ),
        )


class FNOFamily(FamilyRuntime):
    """Alternating-optimization family using a shared EmbeddingCoordinateFNOTreeRegressor.

    The ``f_init`` and ``g_init`` handed to ``run_alternating_family`` are
    path strings pointing at a saved state_dict (or ``None`` / ``"identity"``
    to trigger identity initialization). After every training iteration we
    write a new state_dict snapshot and return its path as the next artifact.
    """

    name: str = "fno"

    def __init__(
        self,
        *,
        config: FNOFamilyConfig,
        embedding_client: Any,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.embedding_client = embedding_client
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model: Optional[EmbeddingCoordinateFNOTreeRegressor] = None
        self._embedding_dim: Optional[int] = None
        self._prepared_cache: Dict[int, List[_PreparedTree]] = {}
        self._last_full_tree_traces: List[StateTree[Any, Any]] = []

    # ------------------------------------------------------------------
    # BundleAwareFamilyRuntime protocol
    # ------------------------------------------------------------------

    @property
    def default_f(self) -> str:
        return "identity"

    @property
    def default_g(self) -> str:
        # FNO's merge path is initialized as the identity/average no-op, which
        # is the raw-concat-equivalent default in the shared ladder vocabulary.
        return "raw_concat"

    def expected_bundle(self) -> Mapping[str, Any]:
        expected: Dict[str, Any] = {
            "leaf_unit": "text_token",
            "state_kind": (
                "embedding",
                "embedding_coordinate",
                "embedding_coordinate_fno_state",
            ),
        }
        if self.config.summary_dim is not None:
            expected["summary_dim"] = int(self.config.summary_dim)
        if self.config.state_dim is not None:
            expected["state_dim_min"] = int(self.config.state_dim)
        return expected

    def supported_inits(self) -> Mapping[str, frozenset[str]]:
        return {
            "f": frozenset({"identity", "artifact"}),
            "g": frozenset({"identity", "raw_concat", "artifact"}),
        }

    def resolve_init(self, *, kind: str, spec: str) -> Any:
        axis = str(kind).strip().lower()
        text = str(spec).strip()
        lowered = text.lower()
        if axis not in {"f", "g"}:
            raise ValueError(f"FNO init kind must be 'f' or 'g', got {kind!r}")
        if not text:
            raise ValueError("FNO init spec must be non-empty")
        if lowered.startswith("artifact:"):
            path = text.partition(":")[2].strip()
            if not path:
                raise ValueError(f"FNO artifact init is missing a path: {spec!r}")
            return path
        if lowered == "identity":
            return "identity"
        if axis == "g" and lowered == "raw_concat":
            return "identity"
        supported = sorted(self.supported_inits().get(axis, frozenset()))
        raise ValueError(
            f"FNO {axis}-init {spec!r} is unsupported; expected one of {supported} "
            "or artifact:<path>"
        )

    def share_state_axes(self) -> frozenset[str]:
        return frozenset({"f", "g"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self, embedding_dim: int) -> EmbeddingCoordinateFNOTreeRegressor:
        if self._model is not None and self._embedding_dim == embedding_dim:
            return self._model
        if (
            self.config.effective_embedding_dim is not None
            and int(embedding_dim) != int(self.config.effective_embedding_dim)
        ):
            raise RuntimeError(
                "FNO effective embedding dimension mismatch: prepared trees produced "
                f"D_eff={embedding_dim}, but config expected "
                f"{self.config.effective_embedding_dim}. Check embedding dim, "
                "embedding_max_length_tokens, and chunks_per_leaf."
            )
        torch.manual_seed(int(self.config.seed))
        model = EmbeddingCoordinateFNOTreeRegressor(
            embedding_dim=embedding_dim,
            hidden_channels=self.config.hidden_channels,
            n_modes=self.config.n_modes,
            n_layers=self.config.n_layers,
            head_hidden_dim=self.config.head_hidden_dim,
            target_min=self.config.target_min,
            target_max=self.config.target_max,
        ).to(self.device)
        if self.config.identity_init:
            model.initialize_as_identity()
        self._model = model
        self._embedding_dim = int(embedding_dim)
        return model

    def _load_state(self, artifact: Any) -> None:
        if self._model is None:
            raise RuntimeError("model not initialized yet")
        if _is_identity_state_artifact(artifact):
            if self.config.identity_init:
                self._model.initialize_as_identity()
            return
        path = Path(str(artifact))
        if not path.exists():
            LOGGER.warning("FNO artifact %s missing; keeping current state", path)
            return
        state = torch.load(path, map_location=self.device, weights_only=False)
        # neuralop's FNO.state_dict() injects a non-parameter ``_metadata`` key;
        # drop it so strict loading passes.
        if isinstance(state, dict):
            state = {k: v for k, v in state.items() if k != "_metadata"}
        self._model.load_state_dict(state)

    def _save_state(self, output_dir: Path, tag: str) -> Path:
        assert self._model is not None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"fno_state_{tag}.pt"
        torch.save(self._model.state_dict(), path)
        return path

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        """Hard-check that an FNO state snapshot can be loaded."""
        if _is_identity_state_artifact(artifact):
            return
        path = Path(str(artifact))
        if not path.exists():
            raise RuntimeError(f"FNO {kind} artifact does not exist: {path}")
        state = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(state, dict):
            raise RuntimeError(f"FNO {kind} artifact is not a state dict: {path}")
        state = {k: v for k, v in state.items() if k != "_metadata"}
        if self._model is not None:
            self._model.load_state_dict(state)

    def _prepare(self, trees: Sequence[LabeledTree]) -> Tuple[List[_PreparedTree], int]:
        key = id(trees)
        if key in self._prepared_cache and self._embedding_dim is not None:
            return self._prepared_cache[key], self._embedding_dim
        prepared, embedding_dim = _prepare_trees(
            list(trees),
            embedding_client=self.embedding_client,
            embedding_max_tokens=self.config.embedding_max_length_tokens,
            chunks_per_leaf=int(self.config.chunks_per_leaf),
            tokenizer_model_path=str(self.config.tokenizer_model_path),
            enforce_no_truncation=self.config.embedding_max_length_tokens is not None,
        )
        # Keep the prepared leaf-coordinate tensors resident on the FNO device.
        # The same prepared objects are reused across f/g stages and repeated
        # evaluations, so paying the host->device transfer once avoids a copy in
        # every tree forward pass.
        for item in prepared:
            item.leaf_embeddings = item.leaf_embeddings.to(self.device)
        self._prepared_cache[key] = prepared
        return prepared, embedding_dim

    # ------------------------------------------------------------------
    # Training loops
    # ------------------------------------------------------------------

    def _train_step_loss_f(
        self,
        model: EmbeddingCoordinateFNOTreeRegressor,
        item: _PreparedTree,
    ) -> torch.Tensor:
        """MSE of predict_normalized vs normalized teacher score, weighted per role."""
        lo, hi = self.config.target_min, self.config.target_max
        states = _forward_tree_states(model, item, device=self.device)
        losses: List[torch.Tensor] = []
        for node_id in item.node_order:
            node = item.tree.get_node(node_id)
            if node is None or node.score is None:
                continue
            is_root = str(node_id) == str(item.root_node_id)
            is_leaf = int(node.level) == 0
            weight = (
                self.config.root_weight
                if is_root
                else (self.config.leaf_weight if is_leaf else self.config.merge_weight)
            )
            if weight <= 0.0:
                continue
            pred = model.predict_normalized(states[node_id]).reshape(())
            target = torch.tensor(
                _normalize_01(float(node.score), lo=lo, hi=hi),
                dtype=torch.float32,
                device=self.device,
            )
            losses.append(float(weight) * F.mse_loss(pred, target))
        if not losses:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        return torch.stack(losses).mean()

    def _train_step_loss_g(
        self,
        model: EmbeddingCoordinateFNOTreeRegressor,
        item: _PreparedTree,
    ) -> torch.Tensor:
        """MSE at MERGE NODES only: current-f applied to student-g-merged state
        should match the teacher's f-score at the merge node.

        leaf_fno, leaf_norm, and score_head are frozen; the gradient only
        flows to merge_fno, so this optimizes g against current f's reading.
        """
        lo, hi = self.config.target_min, self.config.target_max
        states = _forward_tree_states(model, item, device=self.device)
        losses: List[torch.Tensor] = []
        for node_id in item.node_order:
            node = item.tree.get_node(node_id)
            if node is None or node.score is None:
                continue
            if int(node.level) == 0:
                continue  # skip leaves; merge_fno wasn't invoked here
            pred = model.predict_normalized(states[node_id]).reshape(())
            target = torch.tensor(
                _normalize_01(float(node.score), lo=lo, hi=hi),
                dtype=torch.float32,
                device=self.device,
            )
            losses.append(F.mse_loss(pred, target))
        if not losses:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        return torch.stack(losses).mean()

    def _run_training(
        self,
        prepared: Sequence[_PreparedTree],
        *,
        mode: str,
        output_dir: Path,
    ) -> Path:
        model = self._model
        assert model is not None
        if mode == "f":
            model.freeze_for_f_training()
            loss_fn = self._train_step_loss_f
        elif mode == "g":
            model.freeze_for_g_training()
            loss_fn = self._train_step_loss_g
        else:
            raise ValueError(f"mode must be 'f' or 'g', got {mode!r}")

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            LOGGER.warning("No trainable params for mode=%s; skipping", mode)
            model.unfreeze_all()
            return self._save_state(output_dir, f"{mode}_noop")

        # g-training has no gradients at leaf_count=1 (no merge nodes exist).
        # Detect this by counting merge-level nodes across the prepared batch;
        # if none, the training is a no-op by construction.
        if mode == "g":
            merge_node_count = sum(
                1
                for item in prepared
                for node_id in item.node_order
                if (item.tree.get_node(node_id) is not None
                    and int(item.tree.get_node(node_id).level) > 0)
            )
            if merge_node_count == 0:
                LOGGER.info(
                    "No merge nodes in %d trees (leaf_count=1); g-training is a no-op",
                    len(prepared),
                )
                model.unfreeze_all()
                return self._save_state(output_dir, "g_noop_no_merge_nodes")

        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        model.train()
        batch_size = max(1, int(self.config.batch_size))
        n = len(prepared)
        train_losses: List[float] = []
        for epoch in range(int(self.config.epochs_per_iteration)):
            order = list(range(n))
            torch.manual_seed(int(self.config.seed) + epoch)
            random_perm = torch.randperm(n).tolist()
            epoch_losses: List[float] = []
            for start in range(0, n, batch_size):
                optimizer.zero_grad()
                batch_losses: List[torch.Tensor] = []
                for idx in random_perm[start : start + batch_size]:
                    per_loss = loss_fn(model, prepared[idx])
                    # Skip per-tree losses that have no autograd graph. For
                    # g-training, a tree with only leaves (1-leaf trees, short
                    # docs) produces a zero tensor with no grad_fn; calling
                    # backward() on such a batch raises. Skip these cases so
                    # mixed batches (some with merges, some without) train
                    # only on the trees that actually contribute gradients.
                    if per_loss.requires_grad and per_loss.grad_fn is not None:
                        batch_losses.append(per_loss)
                if not batch_losses:
                    continue
                loss = torch.stack(batch_losses).mean()
                loss.backward()
                if self.config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, float(self.config.grad_clip_norm))
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
            if epoch_losses:
                train_losses.append(sum(epoch_losses) / len(epoch_losses))
                LOGGER.debug(
                    "FNO %s epoch %d/%d mean_loss=%.6f",
                    mode, epoch + 1, self.config.epochs_per_iteration, train_losses[-1],
                )
        model.unfreeze_all()

        ckpt_path = self._save_state(output_dir, mode)
        history_path = Path(output_dir) / f"fno_{mode}_training_losses.json"
        history_path.write_text(
            json.dumps({"mode": mode, "epoch_mean_losses": train_losses}, indent=2) + "\n",
            encoding="utf-8",
        )
        return ckpt_path

    # ------------------------------------------------------------------
    # FamilyRuntime protocol
    # ------------------------------------------------------------------

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[LabeledTree],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        prepared, embedding_dim = self._prepare(traces)
        self._ensure_model(embedding_dim)
        # Load the state the model should start from. For FNO the combined
        # (f, g) state lives in one checkpoint; prefer g (the most-recent
        # artifact) since f_init may be stale relative to the current g.
        self._load_state(g if not _is_identity_state_artifact(g) else f_init)
        return self._run_training(prepared, mode="f", output_dir=Path(output_dir))

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[LabeledTree],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        prepared, embedding_dim = self._prepare(traces)
        self._ensure_model(embedding_dim)
        # Load the most recent f checkpoint so the frozen f-path matches the
        # current student f during g training.
        self._load_state(f if not _is_identity_state_artifact(f) else g_init)
        return self._run_training(prepared, mode="g", output_dir=Path(output_dir))

    @torch.no_grad()
    def full_tree_traces_with_f_g(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[LabeledTree],
    ) -> List[StateTree[Any, Any]]:
        """Return full per-node FNO traces for the supplied trees."""

        prepared, embedding_dim = self._prepare(trees)
        self._ensure_model(embedding_dim)
        # Prefer g as the most recent artifact; both are the same state_dict.
        self._load_state(g if not _is_identity_state_artifact(g) else f)
        model = self._model
        assert model is not None
        model.eval()
        traces_out: List[StateTree[Any, Any]] = []
        for item in prepared:
            states = _forward_tree_states(model, item, device=self.device)
            trace = state_tree_skeleton_from_labeled_tree(
                item.tree,
                method_family="fno",
                state_kind="embedding_coordinate_fno_state",
                split=item.split,
            )
            for node_id in item.node_order:
                node = item.tree.get_node(str(node_id))
                state = states.get(str(node_id))
                if node is None or state is None:
                    continue
                pred_norm = float(
                    model.predict_normalized(state).detach().cpu().reshape(()).item()
                )
                pred_raw = _denormalize(
                    pred_norm, lo=self.config.target_min, hi=self.config.target_max
                )
                target_raw = float(node.score)
                is_root = str(node_id) == str(item.root_node_id)
                is_leaf = int(node.level) == 0
                proxy_loss = float((pred_raw - target_raw) ** 2)
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
                    law_channel="root" if is_root else ("leaf" if is_leaf else "merge"),
                    state_kind="embedding_coordinate_fno_state",
                    label_source=str(oracle_kwargs["label_source"] or "proxy_score"),
                )
                update_state_tree_node(
                    trace,
                    str(node_id),
                    rendered=str(node.text or ""),
                    state=state.detach().cpu(),
                    metadata={
                        "prediction": float(pred_raw),
                        "readout_prediction": float(pred_raw),
                        "prediction_normalized": float(pred_norm),
                        "target": float(target_raw),
                        **law_metadata,
                    },
                )
            traces_out.append(trace)
        self._last_full_tree_traces = list(traces_out)
        return traces_out

    @torch.no_grad()
    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[LabeledTree],
    ) -> List[Optional[float]]:
        traces = self.full_tree_traces_with_f_g(f=f, g=g, trees=trees)
        pred_by_doc: Dict[str, Optional[float]] = {}
        for trace in traces:
            root_meta = dict(trace.root.metadata or {})
            doc_id = str(root_meta.get("doc_id", trace.metadata.get("doc_id", "")) or "")
            raw = root_meta.get("prediction")
            try:
                pred_by_doc[doc_id] = None if raw is None else float(raw)
            except (TypeError, ValueError):
                pred_by_doc[doc_id] = None
        return [pred_by_doc.get(str(tree.doc_id)) for tree in trees]

    def export_last_full_tree_traces(
        self,
        output_root: str | Path,
        *,
        split: str = "predict",
    ) -> Dict[str, Any]:
        """Persist the most recent full-tree traces emitted by ``score_roots_with_f``."""

        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        trace_path = root / f"full_tree_traces_{split}.jsonl"
        metrics_path = root / f"full_tree_metrics_{split}.json"
        write_state_trees_jsonl(self._last_full_tree_traces, trace_path)
        metrics = state_tree_trace_metrics(self._last_full_tree_traces)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "full_tree_traces_jsonl": str(trace_path),
            "full_tree_metrics_json": str(metrics_path),
        }
