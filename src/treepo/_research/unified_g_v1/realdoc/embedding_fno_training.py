from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from treepo._research.unified_g_v1.core.manifest import now_iso
from treepo._research.unified_g_v1.core.splits import (
    DocumentSplitIds,
    load_phase1_split_ids,
    resolve_document_split_ids,
)
from treepo._research.unified_g_v1.core.tensor_program import (
    EmbeddingOperatorModules,
    build_embedding_operator_unified_fg_program,
)
from treepo._research.unified_g_v1.training.fit import (
    FitResult,
    TrainerConfig,
    fit,
)
from treepo._research.unified_g_v1.training.backends.pytorch_loop import (
    PyTorchLoopConfig,
    run_pytorch_training,
)
from treepo._research.unified_g_v1.training.objectives import ManifestoRileEmbeddingObjective
from treepo._research.unified_g_v1.training.tree_task import TreeExample
from treepo._research.unified_g_v1.training.trainers.pytorch_tree import _TreeTaskSupervisionAdapter

from treepo._research.preprocessing.chunker import TextChunk, chunk_for_ops
from treepo._research.tasks.manifesto import RILE_SCALE
from treepo._research.tasks.manifesto.data_loader import ManifestoDataset, ManifestoSample
from treepo._research.training.embedding_proxy import VLLMEmbeddingClient


@dataclass
class EmbeddingSequenceTreeNode:
    text_len: int
    leaf_sequence: torch.Tensor | None = None
    children: tuple[int, int] | None = None
    sketch: torch.Tensor | None = None
    oracle_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return self.children is None


@dataclass(frozen=True)
class EmbeddingFNODocStats:
    doc_id: str
    text_chars: int
    leaf_count: int
    subwindow_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EmbeddingFNOTrainingConfig:
    output_dir: str
    embedding_api_base: str
    phase1_data_path: str | None = None
    split_ids_path: str | None = None
    embedding_model: str | None = None
    embedding_api_key: str = "EMPTY"
    embedding_timeout_seconds: float = 60.0
    embedding_batch_size: int = 32
    leaf_tokens: int = 1024
    subwindow_tokens: int = 128
    token_encoding: str = "cl100k_base"
    # summary_dim=None => auto-size to the embedding API's output dim.
    # state_dim=None   => auto-size to 2 * summary_dim (post-resolution).
    summary_dim: int | None = None
    state_dim: int | None = None
    adapter_hidden_dim: int | None = 512
    g_hidden_dim: int | None = 512
    head_width: int | None = None
    operator_modes: int | None = 32
    train_batch_size: int = 4
    n_epochs: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    seed: int = 42
    device: str = "auto"
    save_every_epoch: bool = False
    # Periodic resumable snapshot every N epochs (0 disables).
    checkpoint_every_n_epochs: int = 10
    # Explicit path to resume from; if None, looks for train_state.pt under output_dir.
    resume_from: str | None = None
    # LR auto-scaling: when True, `learning_rate` is treated as the base lr at
    # `lr_reference_summary_dim`; actual lr = base * sqrt(ref_dim / summary_dim).
    # This keeps training stable across the 768 -> 4096 head-dim range.
    auto_scale_lr: bool = True
    lr_reference_summary_dim: int = 768
    # Linear warmup from 0 -> target lr over the first N epochs (stabilizes
    # large-head + few-example cases like summary_dim=4096 with n_train=100).
    warmup_epochs: int = 2
    # Local-law balance. Canonical formula:
    # (1 - λ) · root_mse + λ · Σ ρᵢ · Cᵢ / Σ ρᵢ
    # where Cᵢ ∈ {C1 per-leaf RILE, C2 merge commutativity, C3 per-merge RILE}.
    # λ default matches the Markov publication-bundle setting (0.3); ρs default
    # equal so the local-law block is uniformly distributed.
    local_law_weight: float = 0.3
    c1_relative_weight: float = 1.0
    c2_relative_weight: float = 1.0
    c3_relative_weight: float = 1.0


EmbeddingFNOSplitIds = DocumentSplitIds


def _normalize_rile(raw_value: float) -> float:
    return float(max(0.0, min(1.0, RILE_SCALE.normalize(float(raw_value)))))


def _denormalize_rile(normalized_value: float) -> float:
    return float(RILE_SCALE.denormalize(float(normalized_value)))


def _resolve_device(device: str) -> torch.device:
    rendered = str(device or "auto").strip().lower()
    if rendered and rendered != "auto":
        return torch.device(rendered)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _token_leaf_chunks(
    text: str,
    *,
    leaf_tokens: int,
    token_encoding: str,
) -> list[TextChunk]:
    return list(
        chunk_for_ops(
            text,
            max_tokens=int(leaf_tokens),
            token_encoding=str(token_encoding),
            overlap_tokens=0,
            strategy="axis",
        )
    )


def _sequence_chunks_for_leaf(
    text: str,
    *,
    subwindow_tokens: int,
    token_encoding: str,
) -> list[TextChunk]:
    chunks = list(
        chunk_for_ops(
            text,
            max_tokens=int(subwindow_tokens),
            token_encoding=str(token_encoding),
            overlap_tokens=0,
            strategy="axis",
        )
    )
    if chunks:
        return chunks
    rendered = str(text or "")
    return [TextChunk(text=rendered, start_char=0, end_char=len(rendered))]


def _leaf_embedding_sequence(
    leaf_text: str,
    *,
    embedding_client: VLLMEmbeddingClient,
    subwindow_tokens: int,
    token_encoding: str,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    subchunks = _sequence_chunks_for_leaf(
        str(leaf_text or ""),
        subwindow_tokens=int(subwindow_tokens),
        token_encoding=str(token_encoding),
    )
    texts = [str(chunk.text or "") for chunk in subchunks if str(chunk.text or "").strip()]
    if not texts:
        texts = [str(leaf_text or "")]
    vectors = embedding_client.embed_texts(texts)
    if not vectors:
        raise ValueError("embedding client returned no vectors")
    return torch.as_tensor(vectors, dtype=torch.float32, device=device), int(len(texts))


def _annotate_rile_targets_on_tree(
    nodes: list[EmbeddingSequenceTreeNode],
    span_rile_fn,
) -> None:
    """Post-order walk: compute per-node char span and fetch span RILE.

    Leaves already carry `metadata["start_char"]` and `metadata["end_char"]`
    from chunking. Internal nodes inherit the union span from their children.
    For each node we call `span_rile_fn(start_char, end_char)` and stash the
    result on `node.oracle_scores["rile"]` (None → stays absent).
    """
    for node in nodes:
        if node.is_leaf:
            start = int(node.metadata.get("start_char", 0))
            end = int(node.metadata.get("end_char", 0))
        else:
            if node.children is None:
                continue
            left = nodes[int(node.children[0])]
            right = nodes[int(node.children[1])]
            start = int(
                min(
                    left.metadata.get("start_char", 0),
                    right.metadata.get("start_char", 0),
                )
            )
            end = int(
                max(
                    left.metadata.get("end_char", 0),
                    right.metadata.get("end_char", 0),
                )
            )
            node.metadata["start_char"] = start
            node.metadata["end_char"] = end
        rile = span_rile_fn(start, end)
        if rile is not None:
            node.oracle_scores["rile"] = float(rile)


def build_embedding_tree_example(
    sample: ManifestoSample,
    *,
    embedding_client: VLLMEmbeddingClient,
    leaf_tokens: int,
    subwindow_tokens: int,
    token_encoding: str,
    device: torch.device,
    source_text: str | None = None,
    span_rile_fn=None,
) -> tuple[tuple[list[EmbeddingSequenceTreeNode], float, str], EmbeddingFNODocStats]:
    rendered_text = str(sample.text if source_text is None else source_text or "")
    leaf_chunks = _token_leaf_chunks(
        rendered_text,
        leaf_tokens=int(leaf_tokens),
        token_encoding=str(token_encoding),
    )
    if not leaf_chunks:
        leaf_chunks = [TextChunk(text=rendered_text, start_char=0, end_char=len(rendered_text))]

    nodes: list[EmbeddingSequenceTreeNode] = []
    subwindow_count = 0
    current_level: list[int] = []
    for chunk in leaf_chunks:
        sequence, n_subwindows = _leaf_embedding_sequence(
            str(chunk.text or ""),
            embedding_client=embedding_client,
            subwindow_tokens=int(subwindow_tokens),
            token_encoding=str(token_encoding),
            device=device,
        )
        subwindow_count += int(n_subwindows)
        nodes.append(
            EmbeddingSequenceTreeNode(
                text_len=max(1, int(len(str(chunk.text or "")))),
                leaf_sequence=sequence,
                metadata={
                    "start_char": int(chunk.start_char),
                    "end_char": int(chunk.end_char),
                    "subwindow_count": int(n_subwindows),
                },
            )
        )
        current_level.append(len(nodes) - 1)

    while len(current_level) > 1:
        next_level: list[int] = []
        for index in range(0, len(current_level), 2):
            left_index = int(current_level[index])
            if index + 1 >= len(current_level):
                next_level.append(left_index)
                continue
            right_index = int(current_level[index + 1])
            nodes.append(
                EmbeddingSequenceTreeNode(
                    text_len=int(nodes[left_index].text_len + nodes[right_index].text_len),
                    children=(left_index, right_index),
                )
            )
            next_level.append(len(nodes) - 1)
        current_level = next_level

    # When a per-span RILE lookup is available, annotate every node (leaf
    # and internal) with its span RILE so the objective can enforce strict
    # C1/C3 supervision against Manifesto Project quasi-sentence codings.
    if span_rile_fn is not None:
        _annotate_rile_targets_on_tree(nodes, span_rile_fn)

    stats = EmbeddingFNODocStats(
        doc_id=str(sample.manifesto_id),
        text_chars=int(len(rendered_text)),
        leaf_count=int(sum(1 for node in nodes if node.is_leaf)),
        subwindow_count=int(subwindow_count),
    )
    return (nodes, float(sample.rile), str(sample.manifesto_id)), stats


class EmbeddingSequenceFNOTreeModel(nn.Module):
    tree_model_version = "unified_g"
    default_head = "rile"

    def __init__(
        self,
        *,
        embedding_dim: int,
        summary_dim: int | None = None,
        state_dim: int | None = None,
        adapter_hidden_dim: int | None = None,
        g_hidden_dim: int | None = None,
        head_width: int | None = None,
        operator_modes: int | None = None,
        target_min: float = -100.0,
        target_max: float = 100.0,
    ) -> None:
        super().__init__()
        program = build_embedding_operator_unified_fg_program(
            embedding_dim=int(embedding_dim),
            summary_dim=summary_dim,
            state_dim=state_dim,
            adapter_hidden_dim=adapter_hidden_dim,
            g_hidden_dim=g_hidden_dim,
            head_width=head_width,
            operator_modes=operator_modes,
            output_dim=1,
        )
        runtime = program.runtime
        if not isinstance(runtime, EmbeddingOperatorModules):
            raise TypeError("embedding operator runtime must be EmbeddingOperatorModules")
        self.leaf_adapter_module = runtime.leaf_adapter
        self.merge_adapter_module = runtime.merge_adapter
        self.g_module = runtime.g
        self.f_module = runtime.f
        self.contract = program.contract
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        self.program_contract = program.contract.to_dict()

    @property
    def state_dim(self) -> int:
        return int(self.g_module.state_dim)

    @property
    def leaf_state_dim(self) -> int:
        return int(self.g_module.state_dim)

    @property
    def has_phi(self) -> bool:
        return False

    def encode_leaf_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        summary = self.leaf_adapter_module(sequence)
        return self.g_module(summary)

    def encode_leaf_batch(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.encode_leaves(embeddings=embeddings)

    def encode_leaf_tokens_batch(
        self,
        token_id_batch: Sequence[Sequence[int]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        raise ValueError("EmbeddingSequenceFNOTreeModel does not accept token ids")

    def encode_leaves(
        self,
        *,
        embeddings: torch.Tensor | None = None,
        token_ids: Sequence[Sequence[int]] | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if embeddings is None:
            raise ValueError("EmbeddingSequenceFNOTreeModel requires embedding sequences")
        tensor = embeddings if device is None else embeddings.to(device)
        summary = self.leaf_adapter_module(tensor)
        return self.g_module(summary)

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.ndim == 1:
            left = left.unsqueeze(0)
        if right.ndim == 1:
            right = right.unsqueeze(0)
        summary = self.merge_adapter_module(left, right)
        return self.g_module(summary)

    def merge_batch(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.merge(left, right)

    def _normalized_from_state(self, states: torch.Tensor) -> torch.Tensor:
        logits = self.f_module(states).reshape(-1)
        return torch.sigmoid(logits)

    def predict(self, state: torch.Tensor, head: str = "rile") -> torch.Tensor:
        return self.predict_batch(
            state.unsqueeze(0) if state.ndim == 1 else state,
            head=head,
        ).reshape(-1)

    def predict_batch(self, states: torch.Tensor, head: str = "rile") -> torch.Tensor:
        normalized = self.predict_normalized_batch(states, head=head)
        return (
            normalized * float(self.target_max - self.target_min)
        ) + float(self.target_min)

    def predict_normalized(self, state: torch.Tensor, head: str = "rile") -> torch.Tensor:
        return self.predict_normalized_batch(
            state.unsqueeze(0) if state.ndim == 1 else state,
            head=head,
        ).reshape(-1)

    def predict_normalized_batch(self, states: torch.Tensor, head: str = "rile") -> torch.Tensor:
        del head
        tensor = states if states.ndim > 1 else states.unsqueeze(0)
        return self._normalized_from_state(tensor)

    def predict_confidence(self, state: torch.Tensor, head: str = "rile") -> torch.Tensor:
        return self.predict_confidence_batch(
            state.unsqueeze(0) if state.ndim == 1 else state,
            head=head,
        ).reshape(-1)

    def predict_confidence_batch(self, states: torch.Tensor, head: str = "rile") -> torch.Tensor:
        del head
        normalized = self.predict_normalized_batch(states)
        return 1.0 - (2.0 * torch.abs(normalized - 0.5))

    def phi(self, state: torch.Tensor) -> torch.Tensor | None:
        return state

    def phi_batch(self, states: torch.Tensor) -> torch.Tensor | None:
        return states

    def phi_score(self, state: torch.Tensor) -> torch.Tensor | None:
        return None

    def phi_fiber(self, state: torch.Tensor) -> torch.Tensor | None:
        return None

    def forward_tree(self, batch):
        """TreeModel protocol: fold leaves -> root.

        Returns `(root_state, prediction, forward_aux)`. `forward_aux` carries
        the per-node tensors the objective needs to enforce local laws:

        * `leaf_states`: list[Tensor] — one state vector per leaf node
        * `internal_states`: list[Tensor] — merge states at every internal
          (non-leaf, non-root) node plus the root
        * `commutativity_states`: list[Tensor] — merge(right, left) states,
          computed by swapping children at every internal node. The objective
          compares these against `internal_states` to enforce that the learned
          merge is permutation-invariant (C2 law, given RILE is a
          permutation-invariant statistic).

        Accepts a batch of `TreeExample` (with `.leaves` a list of
        `EmbeddingSequenceTreeNode`) or the legacy `(nodes, target, doc_id)` tuple.
        """
        tree_batch = []
        for entry in batch:
            if hasattr(entry, "leaves"):
                nodes = list(entry.leaves)
                tree_batch.append((nodes, float(entry.target), entry.extra.get("doc_id", "")))
            else:
                tree_batch.append(entry)
        _forward_embedding_tree_batch(self, tree_batch)
        root_states = torch.stack([item[0][-1].sketch for item in tree_batch], dim=0)
        prediction = self.predict_normalized_batch(root_states, head="rile").reshape(-1)
        # Collect per-document per-node predictions so the objective can
        # enforce C1 (leaf-root), C2 (merge commutativity), C3 (merge-root)
        # without holding a reference to the model.
        per_doc: list[dict[str, Any]] = []
        for nodes, _target, _doc_id in tree_batch:
            leaf_states: list[torch.Tensor] = []
            internal_states: list[torch.Tensor] = []
            commutativity_states: list[torch.Tensor] = []
            for node in nodes:
                if node.is_leaf:
                    if node.sketch is not None:
                        leaf_states.append(node.sketch.reshape(-1))
                    continue
                if node.sketch is None or node.children is None:
                    continue
                internal_states.append(node.sketch.reshape(-1))
                left_state = nodes[int(node.children[0])].sketch
                right_state = nodes[int(node.children[1])].sketch
                if left_state is None or right_state is None:
                    continue
                swapped = self.merge(right_state, left_state).reshape(-1)
                commutativity_states.append(swapped)

            doc_aux: dict[str, Any] = {}
            if leaf_states:
                leaf_stack = torch.stack(leaf_states, dim=0)
                doc_aux["leaf_predictions"] = self.predict_normalized_batch(
                    leaf_stack
                ).reshape(-1)
            if internal_states:
                internal_stack = torch.stack(internal_states, dim=0)
                doc_aux["internal_predictions"] = self.predict_normalized_batch(
                    internal_stack
                ).reshape(-1)
                if commutativity_states:
                    swap_stack = torch.stack(commutativity_states, dim=0)
                    doc_aux["commutativity_predictions"] = (
                        self.predict_normalized_batch(swap_stack).reshape(-1)
                    )
                    doc_aux["internal_states"] = internal_stack
                    doc_aux["commutativity_states"] = swap_stack
            per_doc.append(doc_aux)
        forward_aux = {"per_doc": per_doc}
        return root_states, prediction, forward_aux


def _forward_embedding_tree_batch(
    model: EmbeddingSequenceFNOTreeModel,
    batch_trees: Sequence[tuple[list[EmbeddingSequenceTreeNode], float, str]],
) -> None:
    for nodes, _target, _doc_id in batch_trees:
        for node in nodes:
            if node.is_leaf:
                if node.leaf_sequence is None:
                    raise ValueError("leaf node is missing leaf_sequence")
                node.sketch = model.encode_leaf_sequence(node.leaf_sequence).reshape(-1)
                continue
            if node.children is None:
                raise ValueError("internal node is missing children")
            left_state = nodes[int(node.children[0])].sketch
            right_state = nodes[int(node.children[1])].sketch
            if left_state is None or right_state is None:
                raise ValueError("child sketches must be available before merge")
            node.sketch = model.merge(left_state, right_state).reshape(-1)


def _node_state_from_batch_item(
    batch_item: tuple[list[EmbeddingSequenceTreeNode], float, str],
    node_index: int,
) -> torch.Tensor | None:
    nodes, _target, _doc_id = batch_item
    if node_index < 0 or node_index >= len(nodes):
        return None
    return nodes[node_index].sketch


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[list[Any]]:
    size = max(1, int(batch_size))
    for index in range(0, len(items), size):
        yield list(items[index:index + size])


def _evaluate_items(
    *,
    model: EmbeddingSequenceFNOTreeModel,
    items: Sequence[Any],
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    abs_errors_norm: list[float] = []
    abs_errors_raw: list[float] = []
    with torch.no_grad():
        for batch in _batched(items, batch_size):
            result = model.forward_tree(batch)
            if isinstance(result, tuple) and len(result) == 3:
                _root_states, pred_norm, _forward_aux = result
            else:
                _root_states, pred_norm = result
            pred_norm = pred_norm.reshape(-1)
            for offset, batch_item in enumerate(batch):
                if hasattr(batch_item, "leaves"):
                    _nodes = list(batch_item.leaves)
                    target_raw = float(batch_item.target)
                    doc_id = str((batch_item.extra or {}).get("doc_id", ""))
                else:
                    _nodes, target_raw, doc_id = batch_item
                predicted_norm = float(pred_norm[offset].detach().cpu().item())
                target_norm = _normalize_rile(float(target_raw))
                predicted_raw = _denormalize_rile(predicted_norm)
                abs_errors_norm.append(abs(predicted_norm - target_norm))
                abs_errors_raw.append(abs(predicted_raw - float(target_raw)))
                predictions.append(
                    {
                        "doc_id": str(doc_id),
                        "target_raw": float(target_raw),
                        "target_normalized": float(target_norm),
                        "predicted_raw": float(predicted_raw),
                        "predicted_normalized": float(predicted_norm),
                        "leaf_count": int(sum(1 for node in _nodes if node.is_leaf)),
                    }
                )
    mae_norm = float(sum(abs_errors_norm) / max(1, len(abs_errors_norm)))
    mae_raw = float(sum(abs_errors_raw) / max(1, len(abs_errors_raw)))
    return {
        "count": int(len(predictions)),
        "mae_normalized": float(mae_norm),
        "mae_raw": float(mae_raw),
        "predictions": predictions,
    }


def _load_samples_for_doc_ids(
    doc_ids: Sequence[str],
    *,
    dataset: ManifestoDataset,
) -> list[ManifestoSample]:
    samples: list[ManifestoSample] = []
    for doc_id in doc_ids:
        sample = dataset.get_sample(str(doc_id))
        if sample is not None and str(sample.text or "").strip():
            samples.append(sample)
    return samples


def _build_tree_examples(
    samples: Sequence[ManifestoSample],
    *,
    embedding_client: VLLMEmbeddingClient,
    leaf_tokens: int,
    subwindow_tokens: int,
    token_encoding: str,
    device: torch.device,
    span_rile_alignment_for_sample=None,
) -> tuple[list[tuple[list[EmbeddingSequenceTreeNode], float, str]], list[EmbeddingFNODocStats], int]:
    items: list[tuple[list[EmbeddingSequenceTreeNode], float, str]] = []
    stats: list[EmbeddingFNODocStats] = []
    embedding_dim: int | None = None
    for sample in samples:
        source_text: str | None = None
        span_rile_fn = None
        if span_rile_alignment_for_sample is not None:
            aligned = span_rile_alignment_for_sample(sample)
            if aligned is not None:
                source_text, span_rile_fn = aligned
        item, doc_stats = build_embedding_tree_example(
            sample,
            embedding_client=embedding_client,
            leaf_tokens=int(leaf_tokens),
            subwindow_tokens=int(subwindow_tokens),
            token_encoding=str(token_encoding),
            device=device,
            source_text=source_text,
            span_rile_fn=span_rile_fn,
        )
        first_leaf = next((node for node in item[0] if node.is_leaf and node.leaf_sequence is not None), None)
        if first_leaf is not None and embedding_dim is None:
            embedding_dim = int(first_leaf.leaf_sequence.shape[-1])
        items.append(item)
        stats.append(doc_stats)
    if embedding_dim is None:
        raise ValueError("no embedding sequences were built from the requested samples")
    return items, stats, int(embedding_dim)


def _tree_examples_from_embedding_items(
    raw_items: Sequence[tuple[list[EmbeddingSequenceTreeNode], float, str]],
) -> list[TreeExample]:
    out: list[TreeExample] = []
    for nodes, target, doc_id in raw_items:
        leaf_targets: list[float | None] = []
        internal_targets: list[float | None] = []
        for node in nodes:
            value = node.oracle_scores.get("rile")
            if node.is_leaf:
                leaf_targets.append(None if value is None else float(value))
            else:
                internal_targets.append(None if value is None else float(value))
        out.append(
            TreeExample(
                leaves=nodes,
                target=float(target),
                extra={
                    "doc_id": str(doc_id),
                    "leaf_rile_targets": leaf_targets,
                    "internal_rile_targets": internal_targets,
                },
            )
        )
    return out


def run_embedding_fno_training(
    config: EmbeddingFNOTrainingConfig,
) -> dict[str, Any]:
    random.seed(int(config.seed))
    torch.manual_seed(int(config.seed))

    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_ids, split_source = resolve_document_split_ids(
        phase1_data_path=config.phase1_data_path,
        split_ids_path=config.split_ids_path,
    )
    dataset = ManifestoDataset(require_text=True)
    device = _resolve_device(config.device)

    embedding_client = VLLMEmbeddingClient(
        api_base=str(config.embedding_api_base).rstrip("/"),
        model=None if config.embedding_model is None else str(config.embedding_model),
        api_key=str(config.embedding_api_key or "EMPTY"),
        timeout_seconds=float(config.embedding_timeout_seconds),
        batch_size=int(config.embedding_batch_size),
        cache_enabled=False,
    )
    resolved_model_id = embedding_client.resolve_model()

    train_samples = _load_samples_for_doc_ids(split_ids.train_doc_ids, dataset=dataset)
    val_samples = _load_samples_for_doc_ids(split_ids.val_doc_ids, dataset=dataset)

    use_strict_rile_targets = (
        float(config.local_law_weight) > 0.0
        and (
            float(config.c1_relative_weight) > 0.0
            or float(config.c3_relative_weight) > 0.0
        )
    )
    span_rile_alignment_for_sample = None
    if use_strict_rile_targets:
        from treepo._research.tasks.manifesto.rile_codes import default_index

        rile_index = default_index()

        def _aligned_span_rile(sample: ManifestoSample):
            codings = rile_index.manifesto(str(sample.manifesto_id))
            if codings is None:
                return None
            return str(codings.text), codings.span_rile

        span_rile_alignment_for_sample = _aligned_span_rile

    train_items, train_stats, embedding_dim = _build_tree_examples(
        train_samples,
        embedding_client=embedding_client,
        leaf_tokens=int(config.leaf_tokens),
        subwindow_tokens=int(config.subwindow_tokens),
        token_encoding=str(config.token_encoding),
        device=device,
        span_rile_alignment_for_sample=span_rile_alignment_for_sample,
    )
    val_items, val_stats, _ = _build_tree_examples(
        val_samples,
        embedding_client=embedding_client,
        leaf_tokens=int(config.leaf_tokens),
        subwindow_tokens=int(config.subwindow_tokens),
        token_encoding=str(config.token_encoding),
        device=device,
        span_rile_alignment_for_sample=span_rile_alignment_for_sample,
    )
    train_examples = _tree_examples_from_embedding_items(train_items)
    val_examples = _tree_examples_from_embedding_items(val_items)

    # Auto-size the FNO head to the embedding server's output dim so no
    # information is compressed away by an undersized summary head.
    # summary_dim defaults to embedding_dim; state_dim defaults to 2 * summary_dim.
    resolved_summary_dim = (
        int(embedding_dim) if config.summary_dim is None else int(config.summary_dim)
    )
    resolved_state_dim = (
        2 * resolved_summary_dim if config.state_dim is None else int(config.state_dim)
    )
    model = EmbeddingSequenceFNOTreeModel(
        embedding_dim=int(embedding_dim),
        summary_dim=resolved_summary_dim,
        state_dim=resolved_state_dim,
        adapter_hidden_dim=config.adapter_hidden_dim,
        g_hidden_dim=config.g_hidden_dim,
        head_width=config.head_width,
        operator_modes=config.operator_modes,
        target_min=float(RILE_SCALE.min_value),
        target_max=float(RILE_SCALE.max_value),
    ).to(device)

    objective = ManifestoRileEmbeddingObjective(
        rile_scale=RILE_SCALE,
        local_law_weight=float(config.local_law_weight),
        c1_relative_weight=float(config.c1_relative_weight),
        c2_relative_weight=float(config.c2_relative_weight),
        c3_relative_weight=float(config.c3_relative_weight),
    )
    supervision_adapter = _TreeTaskSupervisionAdapter(model=model, objective=objective)

    # Auto-scale learning rate by sqrt(ref_dim / summary_dim).
    # Large heads (summary_dim 4096) would be unstable at the small-head lr.
    if bool(getattr(config, "auto_scale_lr", True)):
        ref_dim = max(1, int(getattr(config, "lr_reference_summary_dim", 768)))
        import math as _math
        scale_factor = _math.sqrt(ref_dim / max(1, int(resolved_summary_dim)))
        resolved_lr = float(config.learning_rate) * float(scale_factor)
    else:
        resolved_lr = float(config.learning_rate)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=resolved_lr,
        weight_decay=float(config.weight_decay),
    )

    # Linear warmup over the first `warmup_epochs` epochs. The scheduler steps
    # once per epoch in the generic pytorch_loop; factor goes 1/W, 2/W, ..., 1.
    warmup_epochs = max(0, int(getattr(config, "warmup_epochs", 2)))
    if warmup_epochs > 0 and warmup_epochs < int(config.n_epochs):
        def _warmup_factor(epoch_idx: int) -> float:
            # epoch_idx starts at 0 after the first scheduler.step() (so epoch 1).
            current = epoch_idx + 1
            if current < warmup_epochs:
                return float(current) / float(warmup_epochs)
            return 1.0
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup_factor)
    else:
        lr_scheduler = None

    config_outer = config

    # Predict-train-mean baseline MAE on val — law-stress-scale denominator.
    # Items are (tree_nodes, target_raw, doc_id) tuples.
    from treepo._research.unified_g_v1.eval.law_stress import (
        DEFAULT_PRIMARY_GAIN_THRESHOLD,
        gain_frac as _rile_gain_frac,
    )
    if train_examples and val_examples:
        _train_mean = sum(float(item.target) for item in train_examples) / float(len(train_examples))
        baseline_val_mae = sum(
            abs(_train_mean - float(item.target)) for item in val_examples
        ) / float(len(val_examples))
    else:
        baseline_val_mae = 0.0
    objective.baseline_val_mae = float(baseline_val_mae)

    _base_evaluate = _evaluate_items

    def _evaluate_items_with_gain(*, model, items, batch_size):
        result = dict(_base_evaluate(model=model, items=items, batch_size=batch_size))
        if baseline_val_mae > 0.0 and "mae_raw" in result:
            gf = _rile_gain_frac(
                model_mae=float(result["mae_raw"]),
                baseline_mae=float(baseline_val_mae),
            )
            result["baseline_val_mae"] = float(baseline_val_mae)
            result["val_mae_gain_frac"] = float(gf)
            result["val_mae_pass"] = 1.0 if gf >= DEFAULT_PRIMARY_GAIN_THRESHOLD else 0.0
        return result

    # Custom trainer: closes over the already-constructed model/items/adapter
    # and drives the PyTorch loop directly. Passed via `cfg.trainer` so the
    # unified `fit()` entry point still dispatches here.
    def _embedding_fno_trainer(cfg, out_dir, _ds=None):
        loop = run_pytorch_training(
            model=model,
            optimizer=optimizer,
            train_items=train_examples,
            val_items=val_examples,
            supervision_adapter=supervision_adapter,
            evaluate_fn=_evaluate_items_with_gain,
            lr_scheduler=lr_scheduler,
            config=PyTorchLoopConfig(
                n_epochs=int(config_outer.n_epochs),
                train_batch_size=int(config_outer.train_batch_size),
                grad_clip_norm=float(config_outer.grad_clip_norm),
                seed=int(config_outer.seed),
                save_every_epoch=bool(config_outer.save_every_epoch),
                best_metric_key="mae_raw",
                checkpoint_every_n_epochs=int(
                    getattr(config_outer, "checkpoint_every_n_epochs", 10)
                ),
                resume_from=(
                    Path(config_outer.resume_from)
                    if getattr(config_outer, "resume_from", None)
                    else None
                ),
            ),
            output_dir=Path(out_dir),
            checkpoint_extra={
                "config": asdict(config_outer),
                "program_contract": model.program_contract,
                "embedding_dim": int(embedding_dim),
            },
        )
        return FitResult(
            backend="pytorch",
            summary={
                "backend": "pytorch",
                "best_epoch": int(loop["best_epoch"]),
                "best_metric_key": str(loop["best_metric_key"]),
                "best_metric_value": float(loop["best_metric_value"]),
            },
            artifacts={"best_checkpoint_path": str(loop["best_checkpoint_path"])},
            history=list(loop["history"]),
        )

    fit_result = fit(
        trainer_config=TrainerConfig(trainer=_embedding_fno_trainer),
        output_dir=output_dir,
    )
    history: list[dict[str, Any]] = list(fit_result.history)
    best_epoch = int(fit_result.summary["best_epoch"])
    best_val_mae = float(fit_result.summary["best_metric_value"])
    best_checkpoint_path = Path(fit_result.artifacts["best_checkpoint_path"])

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_train = _evaluate_items_with_gain(
        model=model,
        items=train_examples,
        batch_size=int(config.train_batch_size),
    )
    final_val = _evaluate_items_with_gain(
        model=model,
        items=val_examples,
        batch_size=int(config.train_batch_size),
    )

    best_gain_frac = None
    best_pass = None
    if baseline_val_mae > 0.0:
        best_gain_frac = float(
            _rile_gain_frac(model_mae=float(best_val_mae), baseline_mae=float(baseline_val_mae))
        )
        best_pass = bool(best_gain_frac >= DEFAULT_PRIMARY_GAIN_THRESHOLD)

    summary = {
        "generated_at": now_iso(),
        "run_kind": "embedding_sequence_fno_fno_training",
        "trainer_surface": "pytorch_tree_objective",
        "objective_name": "ManifestoRileEmbeddingObjective",
        "device": str(device),
        "config": asdict(config),
        "resolved_embedding_model": str(resolved_model_id),
        "embedding_dim": int(embedding_dim),
        "resolved_summary_dim": int(resolved_summary_dim),
        "resolved_state_dim": int(resolved_state_dim),
        "resolved_learning_rate": float(resolved_lr),
        "warmup_epochs": int(warmup_epochs),
        "program_contract": model.program_contract,
        "split_source": str(split_source),
        "split_doc_ids": split_ids.to_dict(),
        "doc_stats": {
            "train": [entry.to_dict() for entry in train_stats],
            "val": [entry.to_dict() for entry in val_stats],
        },
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_mae_raw": float(best_val_mae),
        "baseline_val_mae": float(baseline_val_mae),
        "best_val_mae_gain_frac": best_gain_frac,
        "best_val_mae_pass": best_pass,
        "final_train": final_train,
        "final_val": final_val,
        "artifacts": {
            "best_model": str(best_checkpoint_path),
            "history_json": str(output_dir / "history.json"),
            "summary_json": str(output_dir / "summary.json"),
            "train_predictions_json": str(output_dir / "train_predictions.json"),
            "val_predictions_json": str(output_dir / "val_predictions.json"),
        },
    }

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "train_predictions.json").write_text(
        json.dumps(final_train["predictions"], indent=2),
        encoding="utf-8",
    )
    (output_dir / "val_predictions.json").write_text(
        json.dumps(final_val["predictions"], indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
