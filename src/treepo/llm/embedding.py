"""Canonical embedding-client interface and implementations.

The public contract stays deliberately small:

    embed_texts(Sequence[str]) -> list[list[float]]

Provider choice is a construction concern. Callers should depend on
``EmbeddingClient`` and receive a concrete client from ``build_embedding_client``
or from their runtime context.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
import threading
from typing import Any, Callable, List, Optional, Protocol, Sequence, runtime_checkable
import zlib


@runtime_checkable
class EmbeddingClient(Protocol):
    """Minimal protocol for text-embedding endpoints."""

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover - protocol
        ...


def _default_text_key(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _encode_embedding_f32z(vector: Sequence[float]) -> tuple[bytes, dict[str, Any]]:
    arr = array("f", (float(v) for v in vector))
    if sys.byteorder != "little":
        arr.byteswap()
    raw = arr.tobytes()
    compressed = zlib.compress(raw)
    return compressed, {"dtype": "float32", "byteorder": "little", "dim": int(len(arr))}


def _decode_embedding_f32z(blob: bytes) -> List[float]:
    raw = zlib.decompress(bytes(blob))
    arr = array("f")
    arr.frombytes(raw)
    if sys.byteorder != "little":
        arr.byteswap()
    return [float(v) for v in arr]


class HashingEmbeddingClient:
    """Deterministic local bag-of-token embedding fallback."""

    def __init__(self, dim: int = 32, *, salt: str = "") -> None:
        self.dim = int(max(2, dim))
        self.salt = str(salt or "")

    def resolve_model(self) -> str:
        return f"hashing:{self.dim}:{self.salt}" if self.salt else f"hashing:{self.dim}"

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for idx, token in enumerate(str(text or "").lower().split()):
                payload = token if not self.salt else f"{self.salt}\0{token}"
                digest = hashlib.sha256(payload.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:8], "big") % self.dim
                vec[bucket] += 1.0 + float(idx % 7) * 0.01
            vectors.append(vec)
        return vectors


class DenseHashEmbeddingClient:
    """Deterministic dense hash embedding for real-document smoke paths."""

    def __init__(self, embedding_dim: int = 64, *, salt: str = "unified_g_v1") -> None:
        self.embedding_dim = int(max(1, embedding_dim))
        self.salt = str(salt or "")

    def resolve_model(self) -> str:
        return f"dense_hash:{self.embedding_dim}:{self.salt}"

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [
            self._hash_embedding_vector(text, embedding_dim=self.embedding_dim, salt=self.salt)
            for text in texts
        ]

    @staticmethod
    def _hash_embedding_vector(text: str, *, embedding_dim: int, salt: str) -> List[float]:
        values: List[float] = []
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


HashEmbeddingClient = DenseHashEmbeddingClient


def _disk_cache_key(model_id: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(str(model_id).encode("utf-8"))
    h.update(b"\x00")
    h.update(str(text or "").encode("utf-8"))
    return h.hexdigest()


class DiskCachedEmbeddingClient:
    """Wrap any embedding client with a per-text on-disk cache."""

    def __init__(self, inner: EmbeddingClient, cache_dir: str | os.PathLike, *, model_id: str) -> None:
        self._inner = inner
        self._model_id = str(model_id)
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, List[float]] = {}
        self._loaded_shards: set[str] = set()
        self._lock = threading.Lock()

    def _load_shard(self, prefix: str) -> None:
        if prefix in self._loaded_shards:
            return
        path = self._dir / f"shard_{prefix}.pt"
        if path.exists():
            try:
                import torch

                data = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(data, dict):
                    self._mem.update(data)
            except Exception:
                pass
        self._loaded_shards.add(prefix)

    def _persist(self, by_prefix: dict[str, dict[str, List[float]]]) -> None:
        import torch

        for prefix, entries in by_prefix.items():
            path = self._dir / f"shard_{prefix}.pt"
            merged: dict[str, List[float]] = {}
            if path.exists():
                try:
                    existing = torch.load(path, map_location="cpu", weights_only=False)
                    if isinstance(existing, dict):
                        merged.update(existing)
                except Exception:
                    pass
            merged.update(entries)
            tmp = path.with_suffix(".pt.tmp")
            torch.save(merged, tmp)
            os.replace(tmp, path)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        keys = [_disk_cache_key(self._model_id, str(text or "")) for text in texts]
        with self._lock:
            for key in keys:
                self._load_shard(key[:2])
            missing_idx = [idx for idx, key in enumerate(keys) if key not in self._mem]
        if missing_idx:
            miss_texts = [str(texts[idx] or "") for idx in missing_idx]
            fresh = self._inner.embed_texts(miss_texts)
            if len(fresh) != len(miss_texts):
                raise RuntimeError(
                    "inner embed_texts returned "
                    f"{len(fresh)} vectors for {len(miss_texts)} texts"
                )
            by_prefix: dict[str, dict[str, List[float]]] = {}
            with self._lock:
                for idx, vec in zip(missing_idx, fresh):
                    key = keys[idx]
                    value = [float(x) for x in vec]
                    self._mem[key] = value
                    by_prefix.setdefault(key[:2], {})[key] = value
                self._persist(by_prefix)
        return [self._mem[key] for key in keys]


class OpenAICompatibleEmbeddingClient:
    """Client for OpenAI-compatible ``/v1/embeddings`` endpoints.

    This works for vLLM, OpenAI, custom HTTP servers, and any SGLang deployment
    that exposes the OpenAI embeddings route.
    """

    def __init__(
        self,
        *,
        api_base: str,
        model: Optional[str] = None,
        api_key: str = "EMPTY",
        timeout_seconds: float = 60.0,
        batch_size: int = 32,
        cache_enabled: bool = True,
        memory: Optional[Any] = None,
        session: Optional[Any] = None,
        text_key_fn: Optional[Callable[[str], str]] = None,
        verify_model: bool = True,
    ) -> None:
        self.api_base = str(api_base or "").rstrip("/")
        self.model = model
        self.api_key = api_key or "EMPTY"
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.batch_size = max(1, int(batch_size))
        self.cache_enabled = bool(cache_enabled)
        self.verify_model = bool(verify_model)
        self._cache: dict[str, List[float]] = {}
        self._memory = memory
        self._session = session
        self._text_key_fn = text_key_fn or _default_text_key
        self._model_verified = False

    @property
    def embeddings_url(self) -> str:
        if self.api_base.endswith("/v1"):
            return f"{self.api_base}/embeddings"
        return f"{self.api_base}/v1/embeddings"

    @property
    def models_url(self) -> str:
        if self.api_base.endswith("/v1"):
            return f"{self.api_base}/models"
        return f"{self.api_base}/v1/models"

    def _requests(self) -> Any:
        if self._session is not None:
            return self._session
        import requests

        return requests

    def resolve_model(self) -> str:
        if self.model and self._model_verified:
            return self.model
        if self.model and not self.verify_model:
            return str(self.model)

        response = self._requests().get(
            self.models_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])
        if not data:
            raise RuntimeError("Embedding endpoint returned no models")

        served_ids: List[str] = []
        for row in data:
            model_id = str(row.get("id", "")).strip()
            if model_id:
                served_ids.append(model_id)

        if not served_ids:
            raise RuntimeError("Embedding endpoint returned empty model ids")

        if self.model:
            requested = str(self.model).strip()
            if requested in served_ids:
                self._model_verified = True
                return requested
            if len(served_ids) == 1:
                self.model = served_ids[0]
                self._model_verified = True
                return self.model
            raise RuntimeError(
                "Requested embedding model id not served by endpoint. "
                f"requested={requested!r} served={served_ids!r}"
            )

        self.model = served_ids[0]
        self._model_verified = True
        return self.model

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        model = self.resolve_model()
        namespace = None
        if self._memory is not None:
            namespace_version = getattr(self._memory, "namespace_version", "v1")
            namespace = f"embed:{model}:{namespace_version}"

        outputs: List[Optional[List[float]]] = [None] * len(texts)
        pending_texts: List[str] = []
        pending_indices: List[int] = []
        pending_keys: List[str] = []

        for idx, text in enumerate(texts):
            raw_text = str(text or "")
            key = self._text_key_fn(raw_text)
            if self.cache_enabled and key in self._cache:
                outputs[idx] = list(self._cache[key])
                continue
            if self._memory is not None and namespace is not None:
                entry = self._memory.get(namespace, key)
                if entry is not None and getattr(entry, "value_type", None) == "f32z":
                    try:
                        vector = _decode_embedding_f32z(entry.value)
                    except Exception:
                        vector = None
                    if vector is not None:
                        outputs[idx] = vector
                        if self.cache_enabled:
                            self._cache[key] = vector
                        continue
            pending_texts.append(raw_text)
            pending_indices.append(idx)
            pending_keys.append(key)

        for start in range(0, len(pending_texts), self.batch_size):
            batch_texts = pending_texts[start : start + self.batch_size]
            batch_indices = pending_indices[start : start + self.batch_size]
            batch_keys = pending_keys[start : start + self.batch_size]

            response = self._requests().post(
                self.embeddings_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "input": batch_texts},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            data = sorted(payload.get("data", []), key=lambda row: int(row.get("index", 0)))
            if len(data) != len(batch_texts):
                raise RuntimeError(
                    f"Embedding response mismatch: expected {len(batch_texts)}, got {len(data)}"
                )

            for local_idx, row in enumerate(data):
                embedding = row.get("embedding")
                if not isinstance(embedding, list) or not embedding:
                    raise RuntimeError("Embedding response missing vector")
                vector = [float(v) for v in embedding]
                global_idx = batch_indices[local_idx]
                outputs[global_idx] = vector
                if self.cache_enabled:
                    self._cache[batch_keys[local_idx]] = vector
                if self._memory is not None and namespace is not None:
                    try:
                        blob, meta = _encode_embedding_f32z(vector)
                        self._memory.set(
                            namespace,
                            batch_keys[local_idx],
                            value_type="f32z",
                            value=blob,
                            meta=meta,
                        )
                    except Exception:
                        pass

        finalized: List[List[float]] = []
        for idx, vector in enumerate(outputs):
            if vector is None:
                raise RuntimeError(f"Missing embedding for index {idx}")
            finalized.append(vector)
        return finalized


VLLMEmbeddingClient = OpenAICompatibleEmbeddingClient
OpenAIEmbeddingClient = OpenAICompatibleEmbeddingClient


class TransformersEmbeddingClient:
    """Native Hugging Face transformers embedding client.

    Dependencies are imported lazily so the base package can still be used
    without installing torch/transformers.
    """

    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
        normalize: bool = True,
        pooling: str = "mean",
    ) -> None:
        self.model_name_or_path = str(model_name_or_path or "").strip()
        if not self.model_name_or_path:
            raise ValueError("TransformersEmbeddingClient requires model_name_or_path.")
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(1, int(max_length))
        self.normalize = bool(normalize)
        self.pooling = str(pooling or "mean").strip().lower()
        if self.pooling not in {"mean", "cls"}:
            raise ValueError("TransformersEmbeddingClient pooling must be 'mean' or 'cls'.")
        self._tokenizer = None
        self._model = None

    def resolve_model(self) -> str:
        return self.model_name_or_path

    def _ensure_loaded(self) -> tuple[Any, Any, Any]:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self._model is None:
            self._model = AutoModel.from_pretrained(self.model_name_or_path)
            resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(resolved_device)
            self._model.eval()
        return torch, self._tokenizer, self._model

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        torch, tokenizer, model = self._ensure_loaded()
        vectors: List[List[float]] = []
        device = next(model.parameters()).device
        for start in range(0, len(texts), self.batch_size):
            batch = [str(text or "") for text in texts[start : start + self.batch_size]]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                output = model(**encoded)
                hidden = output.last_hidden_state
                if self.pooling == "cls":
                    pooled = hidden[:, 0, :]
                else:
                    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                if self.normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            vectors.extend([[float(v) for v in row] for row in pooled.detach().cpu().tolist()])
        return vectors


@dataclass(frozen=True)
class EmbeddingClientConfig:
    engine: str = "openai_compatible"
    api_base: str = ""
    model: Optional[str] = None
    api_key: str = "EMPTY"
    timeout_seconds: float = 60.0
    batch_size: int = 32
    cache_enabled: bool = True


def build_embedding_client(
    engine: str = "openai_compatible",
    *,
    api_base: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: str = "EMPTY",
    timeout_seconds: float = 60.0,
    batch_size: int = 32,
    cache_enabled: bool = True,
    memory: Optional[Any] = None,
    session: Optional[Any] = None,
    text_key_fn: Optional[Callable[[str], str]] = None,
    verify_model: bool = True,
    embedding_dim: int = 32,
    salt: str = "",
    model_name_or_path: Optional[str] = None,
    device: Optional[str] = None,
    max_length: int = 512,
    normalize: bool = True,
    pooling: str = "mean",
) -> EmbeddingClient:
    """Build a concrete embedding client while keeping the call protocol stable."""

    engine_name = str(getattr(engine, "value", engine) or "openai_compatible")
    engine_name = engine_name.strip().lower().replace("-", "_")
    if engine_name in {"hash", "hashing", "mock", "deterministic"}:
        return HashingEmbeddingClient(dim=int(embedding_dim), salt=salt)
    if engine_name in {"dense_hash", "hash_dense", "unified_g_hash"}:
        return DenseHashEmbeddingClient(embedding_dim=int(embedding_dim), salt=salt or "unified_g_v1")
    if engine_name in {"transformers", "hf", "huggingface", "native_transformers"}:
        return TransformersEmbeddingClient(
            model_name_or_path=str(model_name_or_path or model or ""),
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize,
            pooling=pooling,
        )

    resolved_base = str(api_base or base_url or "").strip()
    if not resolved_base:
        raise ValueError(f"Embedding engine {engine_name!r} requires api_base/base_url.")
    return OpenAICompatibleEmbeddingClient(
        api_base=resolved_base,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        batch_size=batch_size,
        cache_enabled=cache_enabled,
        memory=memory,
        session=session,
        text_key_fn=text_key_fn,
        verify_model=verify_model,
    )


__all__ = [
    "DenseHashEmbeddingClient",
    "DiskCachedEmbeddingClient",
    "EmbeddingClient",
    "EmbeddingClientConfig",
    "HashEmbeddingClient",
    "HashingEmbeddingClient",
    "OpenAICompatibleEmbeddingClient",
    "OpenAIEmbeddingClient",
    "TransformersEmbeddingClient",
    "VLLMEmbeddingClient",
    "build_embedding_client",
]
