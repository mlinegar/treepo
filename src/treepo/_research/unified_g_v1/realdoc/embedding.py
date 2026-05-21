from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

import torch

from treepo._research.unified_g_v1.core.manifest import now_iso
from treepo._research.unified_g_v1.core.tensor_program import build_embedding_operator_unified_fg_program

from treepo._research.preprocessing.chunker import TextChunk, chunk_for_ops
from treepo._research.tasks.manifesto.data_loader import ManifestoDataset
from treepo._research.training.embedding_proxy import VLLMEmbeddingClient


def _hash_embedding_vector(text: str, *, embedding_dim: int, salt: str) -> list[float]:
    values: list[float] = []
    counter = 0
    encoded_text = str(text or "").encode("utf-8")
    encoded_salt = str(salt or "").encode("utf-8")
    while len(values) < int(embedding_dim):
        digest = hashlib.sha256(
            encoded_salt + b"||" + encoded_text + b"||" + str(counter).encode("ascii")
        ).digest()
        for byte in digest:
            values.append((float(byte) / 127.5) - 1.0)
            if len(values) >= int(embedding_dim):
                break
        counter += 1
    return values


class EmbeddingClientProtocol(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class HashEmbeddingClient:
    embedding_dim: int = 64
    salt: str = "unified_g_v1"

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            _hash_embedding_vector(text, embedding_dim=int(self.embedding_dim), salt=self.salt)
            for text in texts
        ]


@dataclass(frozen=True)
class ManifestoEmbeddingSmokeResult:
    doc_id: str
    leaf_tokens: int
    subwindow_tokens: int
    tree_nodes: int
    tree_leaves: int
    root_state_dim: int
    operator_head_width: int
    predicted_rile_raw: float
    predicted_rile: float
    ground_truth_rile: float
    text_chars: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _make_embedding_client(
    *,
    embedding_backend: str,
    embedding_dim: int,
    salt: str,
    embedding_api_base: str | None,
    embedding_model: str | None,
    embedding_api_key: str,
    embedding_timeout_seconds: float,
    embedding_batch_size: int,
) -> tuple[Any, dict[str, Any]]:
    backend = str(embedding_backend or "hash").strip().lower()
    if backend == "hash":
        return HashEmbeddingClient(embedding_dim=int(embedding_dim), salt=str(salt)), {
            "kind": "hash",
            "salt": str(salt),
            "embedding_dim": int(embedding_dim),
        }
    if backend != "vllm":
        raise ValueError(f"unsupported embedding_backend={embedding_backend!r}")
    if not str(embedding_api_base or "").strip():
        raise ValueError("embedding_api_base is required for vllm embedding backend")
    client = VLLMEmbeddingClient(
        api_base=str(embedding_api_base),
        model=None if embedding_model is None else str(embedding_model),
        api_key=str(embedding_api_key or "EMPTY"),
        timeout_seconds=float(embedding_timeout_seconds),
        batch_size=int(embedding_batch_size),
        cache_enabled=False,
    )
    return client, {
        "kind": "vllm",
        "api_base": str(embedding_api_base),
        "model": str(embedding_model or ""),
    }


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
    return chunks or [TextChunk(text=str(text or ""), start_char=0, end_char=len(str(text or "")))]


def _leaf_embedding_sequence(
    leaf_text: str,
    *,
    embedding_client: Any,
    subwindow_tokens: int,
    token_encoding: str,
) -> torch.Tensor:
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
    return torch.as_tensor(vectors, dtype=torch.float32)


def reduce_embedding_tree(
    leaf_sequences: Sequence[torch.Tensor],
    *,
    embedding_dim: int = 64,
    summary_dim: int | None = None,
    state_dim: int | None = None,
    adapter_hidden_dim: int | None = None,
    g_hidden_dim: int | None = None,
    head_width: int | None = None,
    operator_modes: int | None = None,
    seed: int = 0,
    task_spec: str = "rile",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, int]]:
    torch.manual_seed(int(seed))
    if not leaf_sequences:
        raise ValueError("reduce_embedding_tree requires at least one leaf sequence")
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
    states = [program.leaf_state(sequence, task_spec) for sequence in leaf_sequences]
    tree_nodes = int(len(states))
    merge_count = 0
    while len(states) > 1:
        next_states: list[torch.Tensor] = []
        for index in range(0, len(states), 2):
            left_state = states[index]
            if index + 1 >= len(states):
                next_states.append(left_state)
                continue
            right_state = states[index + 1]
            next_states.append(program.merge_state(left_state, right_state, task_spec))
            merge_count += 1
        states = next_states
    root_state = states[0]
    raw_prediction = program.predict(root_state, task_spec)
    return root_state, raw_prediction, program.contract.to_dict(), {
        "tree_nodes": int(tree_nodes + merge_count),
        "tree_leaves": int(len(leaf_sequences)),
    }


def run_manifesto_embedding_smoke(
    doc_ids: Sequence[str],
    *,
    embedding_dim: int = 64,
    summary_dim: int | None = None,
    state_dim: int | None = None,
    leaf_tokens: int = 1024,
    subwindow_tokens: int = 128,
    token_encoding: str = "cl100k_base",
    head_width: int | None = None,
    operator_modes: int | None = None,
    embedding_backend: str = "hash",
    embedding_api_base: str | None = None,
    embedding_model: str | None = None,
    embedding_api_key: str = "EMPTY",
    embedding_timeout_seconds: float = 60.0,
    embedding_batch_size: int = 32,
    seed: int = 0,
    salt: str = "unified_g_v1",
) -> dict[str, Any]:
    dataset = ManifestoDataset(require_text=True)
    embedding_client, embedding_client_payload = _make_embedding_client(
        embedding_backend=str(embedding_backend),
        embedding_dim=int(embedding_dim),
        salt=str(salt),
        embedding_api_base=embedding_api_base,
        embedding_model=embedding_model,
        embedding_api_key=str(embedding_api_key or "EMPTY"),
        embedding_timeout_seconds=float(embedding_timeout_seconds),
        embedding_batch_size=int(embedding_batch_size),
    )
    results: list[dict[str, Any]] = []
    contract_payload: dict[str, Any] | None = None
    for doc_id in doc_ids:
        sample = dataset.get_sample(str(doc_id))
        if sample is None or not str(sample.text or "").strip():
            raise ValueError(f"Could not load manifesto sample for doc_id={doc_id!r}")
        leaf_chunks = _token_leaf_chunks(
            sample.text,
            leaf_tokens=int(leaf_tokens),
            token_encoding=str(token_encoding),
        )
        leaf_sequences = [
            _leaf_embedding_sequence(
                str(chunk.text or ""),
                embedding_client=embedding_client,
                subwindow_tokens=int(subwindow_tokens),
                token_encoding=str(token_encoding),
            )
            for chunk in leaf_chunks
        ]
        resolved_embedding_dim = int(leaf_sequences[0].shape[-1])
        root_state, raw_prediction, contract_payload, tree_stats = reduce_embedding_tree(
            leaf_sequences,
            embedding_dim=resolved_embedding_dim,
            summary_dim=summary_dim,
            state_dim=state_dim,
            adapter_hidden_dim=adapter_hidden_dim_from(summary_dim, resolved_embedding_dim),
            g_hidden_dim=g_hidden_dim_from(summary_dim, resolved_embedding_dim, state_dim),
            head_width=head_width,
            operator_modes=operator_modes,
            seed=int(seed),
            task_spec="rile",
        )
        raw_scalar = float(raw_prediction.reshape(-1)[0].detach().cpu().item())
        scaled_rile = float(torch.tanh(torch.tensor(raw_scalar)).item() * 100.0)
        contract_extra = dict((contract_payload or {}).get("extra") or {})
        results.append(
            ManifestoEmbeddingSmokeResult(
                doc_id=str(sample.manifesto_id),
                leaf_tokens=int(leaf_tokens),
                subwindow_tokens=int(subwindow_tokens),
                tree_nodes=int(tree_stats["tree_nodes"]),
                tree_leaves=int(tree_stats["tree_leaves"]),
                root_state_dim=int(root_state.shape[-1]),
                operator_head_width=int(contract_extra.get("operator_head_width") or 0),
                predicted_rile_raw=raw_scalar,
                predicted_rile=scaled_rile,
                ground_truth_rile=float(sample.rile),
                text_chars=int(len(sample.text)),
            ).to_dict()
        )
    resolved_contract_extra = dict((contract_payload or {}).get("extra") or {})
    return {
        "generated_at": now_iso(),
        "approach_kind": "embedding_sequence_fno_fno",
        "doc_ids": [str(doc_id) for doc_id in doc_ids],
        "embedding_dim": (
            int(resolved_contract_extra.get("embedding_dim"))
            if resolved_contract_extra.get("embedding_dim") is not None
            else int(embedding_dim)
        ),
        "summary_dim": None if summary_dim is None else int(summary_dim),
        "state_dim": None if state_dim is None else int(state_dim),
        "leaf_tokens": int(leaf_tokens),
        "subwindow_tokens": int(subwindow_tokens),
        "token_encoding": str(token_encoding),
        "operator_head_width": int(
            resolved_contract_extra.get("operator_head_width")
            or (0 if head_width is None else int(head_width))
        ),
        "operator_modes": int(
            resolved_contract_extra.get("operator_modes")
            or (0 if operator_modes is None else int(operator_modes))
        ),
        "seed": int(seed),
        "embedding_client": embedding_client_payload,
        "program_contract": contract_payload or {},
        "results": results,
    }


def adapter_hidden_dim_from(summary_dim: int | None, embedding_dim: int) -> int | None:
    if summary_dim is None:
        return None
    return max(32, int(summary_dim), int(embedding_dim))


def g_hidden_dim_from(
    summary_dim: int | None,
    embedding_dim: int,
    state_dim: int | None,
) -> int | None:
    if summary_dim is None and state_dim is None:
        return None
    candidates = [int(embedding_dim)]
    if summary_dim is not None:
        candidates.append(int(summary_dim))
    if state_dim is not None:
        candidates.append(int(state_dim))
    return max(32, *candidates)
