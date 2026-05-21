"""
Embedding-proxy training utilities backed by a vLLM OpenAI-compatible API.

The proxy is a lightweight ridge-regression head over embeddings served from
`/v1/embeddings`. This keeps inference cheap while allowing iterative refits as
new labels arrive.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import zlib

import numpy as np
import requests
from treepo._research.core.conditional_memory import canonical_hash, get_default_memory
from treepo._research.training.supervision import (
    DenseScalarRidgeModelConfig,
    DenseScalarRidgeTrainingConfig,
    DenseSupervisionExample,
    OPTIMIZER_FAMILY_BAG_LEVEL_GRADIENT,
    OPTIMIZER_FAMILY_CLOSED_FORM_LINEAR,
    OPTIMIZER_FAMILY_GRADIENT_DENSE,
    REPRESENTATION_BAG_OF_EMBEDDING_VECTORS,
    REPRESENTATION_EMBEDDING_VECTOR,
    TARGET_SCALAR,
    build_dense_full_document_supervision_dataset,
    fit_dense_scalar_ridge_regressor,
    supervision_training_contract,
)
try:
    import torch
except Exception:  # pragma: no cover - optional dependency path
    torch = None

if TYPE_CHECKING:
    from treepo._research.core.conditional_memory import ConditionalMemory

logger = logging.getLogger(__name__)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        converted = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(converted):
        return float(default)
    return converted


def _safe_optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if converted != converted:
        return None
    return converted


def _encode_embedding_f32z(vector: Sequence[float]) -> tuple[bytes, Dict[str, Any]]:
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


def _solve_linear_system(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Solve A x = b with Gaussian elimination + partial pivoting."""
    n = len(a)
    if n == 0 or any(len(row) != n for row in a) or len(b) != n:
        return None

    mat = [row[:] for row in a]
    rhs = b[:]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(mat[r][col]))
        pivot = mat[pivot_row][col]
        if abs(pivot) < 1e-12:
            return None

        if pivot_row != col:
            mat[col], mat[pivot_row] = mat[pivot_row], mat[col]
            rhs[col], rhs[pivot_row] = rhs[pivot_row], rhs[col]

        inv_pivot = 1.0 / mat[col][col]
        for j in range(col, n):
            mat[col][j] *= inv_pivot
        rhs[col] *= inv_pivot

        for r in range(n):
            if r == col:
                continue
            factor = mat[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n):
                mat[r][j] -= factor * mat[col][j]
            rhs[r] -= factor * rhs[col]

    return rhs


@dataclass
class LabeledEmbeddingExample:
    """Single supervision example for embedding-head proxy training."""

    doc_id: str
    text: str
    target_score: float
    truth_label_source: str = "unknown"


@dataclass
class EmbeddingRidgeProxyModel:
    """Ridge regression head over embedding vectors."""

    weights: List[float]
    bias: float
    embedding_dim: int
    embedding_model: str
    model_id: str = "embedding_proxy_v1"
    ridge_lambda: float = 1.0
    train_size: int = 0
    training_contract: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def predict_from_embedding(self, embedding: Sequence[float]) -> float:
        usable = min(len(embedding), len(self.weights))
        score = self.bias
        for idx in range(usable):
            score += float(embedding[idx]) * float(self.weights[idx])
        return _clamp01(score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "embedding_ridge_proxy",
            "model_id": self.model_id,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "ridge_lambda": self.ridge_lambda,
            "train_size": self.train_size,
            "training_contract": (
                dict(self.training_contract) if isinstance(self.training_contract, dict) else None
            ),
            "bias": self.bias,
            "weights": [float(w) for w in self.weights],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingRidgeProxyModel":
        weights = [float(w) for w in data.get("weights", [])]
        return cls(
            weights=weights,
            bias=float(data.get("bias", 0.0)),
            embedding_dim=int(data.get("embedding_dim", len(weights))),
            embedding_model=str(data.get("embedding_model", "unknown")),
            model_id=str(data.get("model_id", "embedding_proxy_v1")),
            ridge_lambda=float(data.get("ridge_lambda", 1.0)),
            train_size=int(data.get("train_size", 0)),
            training_contract=(
                dict(data["training_contract"])
                if isinstance(data.get("training_contract"), dict)
                else None
            ),
            created_at=str(data.get("created_at", datetime.now().isoformat())),
        )

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


@dataclass
class EmbeddingLinearSGDProxyModel:
    """Trainable linear head optimized with SGD/Adam over embeddings."""

    weights: List[float]
    bias: float
    embedding_dim: int
    embedding_model: str
    model_id: str = "embedding_proxy_linear_sgd_v1"
    epochs: int = 25
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    train_size: int = 0
    final_train_loss: Optional[float] = None
    training_contract: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def predict_from_embedding(self, embedding: Sequence[float]) -> float:
        usable = min(len(embedding), len(self.weights))
        logit = self.bias
        for idx in range(usable):
            logit += float(embedding[idx]) * float(self.weights[idx])
        score = 1.0 / (1.0 + math.exp(-max(min(logit, 40.0), -40.0)))
        return _clamp01(score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "embedding_linear_sgd_proxy",
            "model_id": self.model_id,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "train_size": self.train_size,
            "final_train_loss": self.final_train_loss,
            "training_contract": (
                dict(self.training_contract) if isinstance(self.training_contract, dict) else None
            ),
            "bias": self.bias,
            "weights": [float(w) for w in self.weights],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingLinearSGDProxyModel":
        weights = [float(w) for w in data.get("weights", [])]
        return cls(
            weights=weights,
            bias=float(data.get("bias", 0.0)),
            embedding_dim=int(data.get("embedding_dim", len(weights))),
            embedding_model=str(data.get("embedding_model", "unknown")),
            model_id=str(data.get("model_id", "embedding_proxy_linear_sgd_v1")),
            epochs=int(data.get("epochs", 25)),
            learning_rate=float(data.get("learning_rate", 5e-3)),
            weight_decay=float(data.get("weight_decay", 1e-4)),
            train_size=int(data.get("train_size", 0)),
            final_train_loss=_safe_optional_float(data.get("final_train_loss")),
            training_contract=(
                dict(data["training_contract"])
                if isinstance(data.get("training_contract"), dict)
                else None
            ),
            created_at=str(data.get("created_at", datetime.now().isoformat())),
        )

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


@dataclass
class EmbeddingMILSGDProxyModel:
    """
    Multi-instance (bag-level) proxy trained from doc-level scores.

    This model is intended to produce *span/window relevance* scores that are
    rubric-dependent without requiring span-level truth labels.

    Training objective:
      - Split each document into fixed char windows.
      - Predict per-window relevance probabilities p_i in [0,1].
      - Pool to a doc-level prediction with a monotone aggregator:
          y_hat = sigmoid(bag_bias + log1p(sum_i p_i))

    At inference time:
      - `predict_from_embedding()` returns p_i for a single window embedding,
        which plugs directly into the existing span-feedback path.
      - `predict_bag_from_embeddings()` returns y_hat for a document.
    """

    weights: List[float]
    bias: float
    bag_bias: float
    embedding_dim: int
    embedding_model: str
    model_id: str = "embedding_proxy_mil_sgd_v1"
    window_size_chars: int = 1200
    window_overlap_chars: int = 150
    epochs: int = 25
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    smoothness_lambda: float = 0.0
    sparsity_lambda: float = 0.0
    drift_temperature: float = 0.15
    train_size: int = 0
    final_train_loss: Optional[float] = None
    training_contract: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def _sigmoid(self, logit: float) -> float:
        clipped = max(min(float(logit), 40.0), -40.0)
        return 1.0 / (1.0 + math.exp(-clipped))

    def predict_from_embedding(self, embedding: Sequence[float]) -> float:
        usable = min(len(embedding), len(self.weights))
        logit = self.bias
        for idx in range(usable):
            logit += float(embedding[idx]) * float(self.weights[idx])
        return _clamp01(self._sigmoid(logit))

    def predict_bag_from_embeddings(self, embeddings: Sequence[Sequence[float]]) -> float:
        sum_p = 0.0
        for vec in embeddings:
            sum_p += self.predict_from_embedding(vec)
        bag_logit = float(self.bag_bias) + math.log1p(max(0.0, sum_p))
        return _clamp01(self._sigmoid(bag_logit))

    def bag_marginals_from_window_scores(self, window_scores: Sequence[float]) -> Dict[str, Any]:
        """
        Compute cheap leave-one-out marginals from per-window relevance scores.

        Returns:
          {"bag_score": y_hat, "deltas": [y_hat - y_hat(without i), ...]}
        """
        scores = [_clamp01(float(v)) for v in (window_scores or [])]
        sum_p = sum(scores)
        bag_logit = float(self.bag_bias) + math.log1p(max(0.0, sum_p))
        y_hat = _clamp01(self._sigmoid(bag_logit))
        deltas: List[float] = []
        for p_i in scores:
            sum_minus = max(0.0, sum_p - p_i)
            y_minus = _clamp01(self._sigmoid(float(self.bag_bias) + math.log1p(sum_minus)))
            deltas.append(_clamp01(y_hat - y_minus))
        return {"bag_score": y_hat, "deltas": deltas}

    def get_mil_attention_scores(
        self, embeddings: Sequence[Sequence[float]],
    ) -> List[float]:
        """Get per-window MIL relevance scores for adaptive windowing.

        Returns a list of per-window relevance probabilities in [0, 1], where
        high values indicate windows that are informative for the target task
        (e.g. RILE scoring) and should receive finer windowing.

        These scores can be used as the ``score_windows`` callback in
        ``build_unified_tree()`` to drive content-aware adaptive windowing.

        Args:
            embeddings: Per-window embedding vectors (one per window).

        Returns:
            Per-window relevance scores in [0, 1].
        """
        return [self.predict_from_embedding(emb) for emb in embeddings]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "embedding_mil_sgd_proxy",
            "model_id": self.model_id,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "window_size_chars": self.window_size_chars,
            "window_overlap_chars": self.window_overlap_chars,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "smoothness_lambda": self.smoothness_lambda,
            "sparsity_lambda": self.sparsity_lambda,
            "drift_temperature": self.drift_temperature,
            "train_size": self.train_size,
            "final_train_loss": self.final_train_loss,
            "training_contract": (
                dict(self.training_contract) if isinstance(self.training_contract, dict) else None
            ),
            "bag_bias": float(self.bag_bias),
            "bias": float(self.bias),
            "weights": [float(w) for w in self.weights],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingMILSGDProxyModel":
        weights = [float(w) for w in data.get("weights", [])]
        return cls(
            weights=weights,
            bias=float(data.get("bias", 0.0)),
            bag_bias=float(data.get("bag_bias", 0.0)),
            embedding_dim=int(data.get("embedding_dim", len(weights))),
            embedding_model=str(data.get("embedding_model", "unknown")),
            model_id=str(data.get("model_id", "embedding_proxy_mil_sgd_v1")),
            window_size_chars=int(data.get("window_size_chars", 1200)),
            window_overlap_chars=int(data.get("window_overlap_chars", 150)),
            epochs=int(data.get("epochs", 25)),
            learning_rate=float(data.get("learning_rate", 5e-3)),
            weight_decay=float(data.get("weight_decay", 1e-4)),
            smoothness_lambda=float(data.get("smoothness_lambda", 0.0)),
            sparsity_lambda=float(data.get("sparsity_lambda", 0.0)),
            drift_temperature=float(data.get("drift_temperature", 0.15)),
            train_size=int(data.get("train_size", 0)),
            final_train_loss=_safe_optional_float(data.get("final_train_loss")),
            training_contract=(
                dict(data["training_contract"])
                if isinstance(data.get("training_contract"), dict)
                else None
            ),
            created_at=str(data.get("created_at", datetime.now().isoformat())),
        )

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


def load_embedding_proxy_model(path: Path) -> Any:
    """
    Load an embedding proxy model saved with save_json().

    Supports ridge, linear_sgd, and mil_sgd proxy heads.
    """
    with open(path, "r") as f:
        data = json.load(f)

    model_type = str(data.get("model_type", "")).strip().lower()
    if model_type == "embedding_ridge_proxy":
        return EmbeddingRidgeProxyModel.from_dict(data)
    if model_type == "embedding_linear_sgd_proxy":
        return EmbeddingLinearSGDProxyModel.from_dict(data)
    if model_type == "embedding_mil_sgd_proxy":
        return EmbeddingMILSGDProxyModel.from_dict(data)

    # Backward-compatible fallback when model_type was not present.
    if "ridge_lambda" in data:
        return EmbeddingRidgeProxyModel.from_dict(data)
    return EmbeddingLinearSGDProxyModel.from_dict(data)


class VLLMEmbeddingClient:
    """Client for OpenAI-compatible embedding endpoints served by vLLM."""

    def __init__(
        self,
        *,
        api_base: str,
        model: Optional[str] = None,
        api_key: str = "EMPTY",
        timeout_seconds: float = 60.0,
        batch_size: int = 32,
        cache_enabled: bool = True,
        memory: Optional["ConditionalMemory"] = None,
    ):
        self.api_base = (api_base or "").rstrip("/")
        self.model = model
        self.api_key = api_key or "EMPTY"
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.batch_size = max(1, int(batch_size))
        self.cache_enabled = bool(cache_enabled)
        self._cache: Dict[str, List[float]] = {}
        self._memory = memory if memory is not None else get_default_memory()
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

    def resolve_model(self) -> str:
        response = None
        if self.model and self._model_verified:
            return self.model

        response = requests.get(
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
                logger.warning(
                    "Requested embedding model '%s' not found on endpoint; using served id '%s'.",
                    requested,
                    served_ids[0],
                )
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
        """Embed texts via vLLM API with deterministic local cache."""
        if not texts:
            return []

        model = self.resolve_model()
        namespace = None
        if self._memory is not None:
            namespace = f"embed:{model}:{self._memory.namespace_version}"
        outputs: List[Optional[List[float]]] = [None] * len(texts)
        pending_texts: List[str] = []
        pending_indices: List[int] = []
        pending_keys: List[str] = []

        for idx, text in enumerate(texts):
            raw_text = str(text or "")
            key = canonical_hash(raw_text)
            if self.cache_enabled and key in self._cache:
                outputs[idx] = list(self._cache[key])
                continue
            if self._memory is not None and namespace is not None:
                entry = self._memory.get(namespace, key)
                if entry is not None and entry.value_type == "f32z":
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

            response = requests.post(
                self.embeddings_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": batch_texts,
                },
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


def fit_embedding_ridge_proxy(
    examples: Sequence[LabeledEmbeddingExample],
    *,
    embedding_client: VLLMEmbeddingClient,
    ridge_lambda: float = 1.0,
    model_id: str = "embedding_proxy_v1",
) -> EmbeddingRidgeProxyModel:
    """Fit a ridge regression head over vLLM embeddings."""
    cleaned: List[LabeledEmbeddingExample] = []
    for ex in examples:
        target = _safe_optional_float(getattr(ex, "target_score", None))
        if target is None:
            continue
        text = str(getattr(ex, "text", "") or "")
        if not text:
            continue
        cleaned.append(
            LabeledEmbeddingExample(
                doc_id=str(getattr(ex, "doc_id", "")),
                text=text,
                target_score=_clamp01(target),
                truth_label_source=str(getattr(ex, "truth_label_source", "unknown")),
            )
        )

    if not cleaned:
        raise ValueError("No valid examples for embedding proxy fit")

    embeddings = embedding_client.embed_texts([ex.text for ex in cleaned])
    dim = len(embeddings[0])
    if dim <= 0:
        raise ValueError("Embedding dimension is zero")
    if any(len(vec) != dim for vec in embeddings):
        raise ValueError("Inconsistent embedding dimensions in fit data")
    supervision = build_dense_full_document_supervision_dataset(
        [
            DenseSupervisionExample(
                example_id=str(ex.doc_id or f"embedding_doc_{idx}"),
                features=list(vec),
                scalar_target=float(ex.target_score),
                original_text=f"embedding_proxy::{ex.doc_id}",
                rubric="Predict scalar document scores from fixed embedding features.",
                response="embedded_document_candidate",
                response_id=str(ex.doc_id or f"embedding_doc_{idx}"),
                source_doc_id=str(ex.doc_id or f"embedding_doc_{idx}"),
                truth_label_source=str(ex.truth_label_source),
                metadata={
                    "embedding_model": embedding_client.resolve_model(),
                    "embedding_dim": int(dim),
                },
            )
            for idx, (ex, vec) in enumerate(zip(cleaned, embeddings))
        ],
        application_name="embedding_proxy",
        supervision_signal_name="document_level_target",
        response_signal_name="document_score",
        law_type="document_level_target",
        split="train",
        response_signal_min=0.0,
        response_signal_max=1.0,
        metadata={
            "embedding_model": embedding_client.resolve_model(),
            "embedding_dim": int(dim),
            "training_application": "embedding_proxy",
        },
    )
    ridge = max(0.0, float(ridge_lambda))
    ridge_model, _fit_result = fit_dense_scalar_ridge_regressor(
        supervision,
        config=DenseScalarRidgeTrainingConfig(
            model=DenseScalarRidgeModelConfig(ridge_alpha=ridge)
        ),
    )
    weights_arr = np.asarray(ridge_model.weights, dtype=np.float64)
    bias = float(ridge_model.bias)
    if (not np.isfinite(bias)) or (not np.all(np.isfinite(weights_arr))):
        mean_target = sum(ex.target_score for ex in cleaned) / float(len(cleaned))
        logger.warning(
            "Embedding ridge solve produced non-finite values; using intercept-only fallback"
        )
        bias = float(mean_target)
        weights_arr = np.zeros((dim,), dtype=np.float64)

    weights = [float(w) for w in weights_arr.tolist()]
    return EmbeddingRidgeProxyModel(
        weights=weights,
        bias=bias,
        embedding_dim=dim,
        embedding_model=embedding_client.resolve_model(),
        model_id=model_id,
        ridge_lambda=ridge,
        train_size=len(cleaned),
        training_contract=supervision_training_contract(
            representation_kind=REPRESENTATION_EMBEDDING_VECTOR,
            target_kind=TARGET_SCALAR,
            optimizer_family=OPTIMIZER_FAMILY_CLOSED_FORM_LINEAR,
            optimizer_backend="closed_form_ridge",
            n_train_rows=len(cleaned),
        ),
    )


def fit_embedding_linear_sgd_proxy(
    examples: Sequence[LabeledEmbeddingExample],
    *,
    embedding_client: VLLMEmbeddingClient,
    epochs: int = 25,
    learning_rate: float = 5e-3,
    weight_decay: float = 1e-4,
    model_id: str = "embedding_proxy_linear_sgd_v1",
) -> EmbeddingLinearSGDProxyModel:
    """Fit a sigmoid-linear embedding head with gradient-based optimization."""
    if torch is None:
        raise ImportError(
            "PyTorch is required for linear_sgd embedding head training"
        )

    cleaned: List[LabeledEmbeddingExample] = []
    for ex in examples:
        target = _safe_optional_float(getattr(ex, "target_score", None))
        if target is None:
            continue
        text = str(getattr(ex, "text", "") or "")
        if not text:
            continue
        cleaned.append(
            LabeledEmbeddingExample(
                doc_id=str(getattr(ex, "doc_id", "")),
                text=text,
                target_score=_clamp01(target),
                truth_label_source=str(getattr(ex, "truth_label_source", "unknown")),
            )
        )

    if not cleaned:
        raise ValueError("No valid examples for embedding linear_sgd fit")

    embeddings = embedding_client.embed_texts([ex.text for ex in cleaned])
    dim = len(embeddings[0])
    if dim <= 0:
        raise ValueError("Embedding dimension is zero")
    if any(len(vec) != dim for vec in embeddings):
        raise ValueError("Inconsistent embedding dimensions in fit data")

    x_tensor = torch.tensor(embeddings, dtype=torch.float32)
    y_tensor = torch.tensor(
        [float(ex.target_score) for ex in cleaned],
        dtype=torch.float32,
    ).unsqueeze(1)

    head = torch.nn.Linear(dim, 1)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=max(1e-6, float(learning_rate)),
        weight_decay=max(0.0, float(weight_decay)),
    )
    loss_fn = torch.nn.MSELoss()
    epochs_i = max(1, int(epochs))

    final_loss = None
    for _ in range(epochs_i):
        optimizer.zero_grad(set_to_none=True)
        logits = head(x_tensor)
        preds = torch.sigmoid(logits)
        loss = loss_fn(preds, y_tensor)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())

    with torch.no_grad():
        weight = head.weight.detach().cpu().reshape(-1).tolist()
        bias = float(head.bias.detach().cpu().item())

    return EmbeddingLinearSGDProxyModel(
        weights=[float(v) for v in weight],
        bias=bias,
        embedding_dim=dim,
        embedding_model=embedding_client.resolve_model(),
        model_id=model_id,
        epochs=epochs_i,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        train_size=len(cleaned),
        final_train_loss=final_loss,
        training_contract=supervision_training_contract(
            representation_kind=REPRESENTATION_EMBEDDING_VECTOR,
            target_kind=TARGET_SCALAR,
            optimizer_family=OPTIMIZER_FAMILY_GRADIENT_DENSE,
            optimizer_backend="torch_linear",
            n_train_rows=len(cleaned),
        ),
    )


def fit_embedding_mil_sgd_proxy(
    examples: Sequence[LabeledEmbeddingExample],
    *,
    embedding_client: VLLMEmbeddingClient,
    window_size_chars: int = 1200,
    window_overlap_chars: int = 150,
    epochs: int = 10,
    learning_rate: float = 5e-3,
    weight_decay: float = 1e-4,
    smoothness_lambda: float = 0.0,
    sparsity_lambda: float = 0.0,
    drift_temperature: float = 0.15,
    max_windows_per_doc: int = 128,
    seed: int = 0,
    model_id: str = "embedding_proxy_mil_sgd_v1",
) -> EmbeddingMILSGDProxyModel:
    """Fit a MIL (doc-level) proxy that yields per-window relevance probabilities."""
    if torch is None:
        raise ImportError(
            "PyTorch is required for mil_sgd embedding head training"
        )

    window_size_chars = max(32, int(window_size_chars))
    window_overlap_chars = max(0, int(window_overlap_chars))
    if window_overlap_chars >= window_size_chars:
        window_overlap_chars = max(0, window_size_chars - 1)
    max_windows_per_doc = max(1, int(max_windows_per_doc))

    cleaned: List[LabeledEmbeddingExample] = []
    for ex in examples:
        target = _safe_optional_float(getattr(ex, "target_score", None))
        if target is None:
            continue
        text = str(getattr(ex, "text", "") or "")
        if not text:
            continue
        cleaned.append(
            LabeledEmbeddingExample(
                doc_id=str(getattr(ex, "doc_id", "")),
                text=text,
                target_score=_clamp01(target),
                truth_label_source=str(getattr(ex, "truth_label_source", "unknown")),
            )
        )

    if not cleaned:
        raise ValueError("No valid examples for embedding MIL proxy fit")

    from treepo._research.preprocessing.adaptive_windows import uniform_axis_windows

    def _window_texts(text: str) -> List[str]:
        total = len(text)
        windows = uniform_axis_windows(
            total,
            window_size=window_size_chars,
            overlap=window_overlap_chars,
            unit="char",
        )
        payloads: List[str] = []
        for w in windows[:max_windows_per_doc]:
            start = max(0, min(total, int(w.start)))
            end = max(start, min(total, int(w.end)))
            snippet = text[start:end]
            if snippet.strip():
                payloads.append(snippet)
        return payloads

    doc_windows: List[List[str]] = []
    doc_ids: List[str] = []
    targets: List[float] = []
    flat_payloads: List[str] = []
    slices: List[tuple[int, int]] = []

    for ex in cleaned:
        payloads = _window_texts(ex.text)
        if not payloads:
            continue
        start = len(flat_payloads)
        flat_payloads.extend(payloads)
        end = len(flat_payloads)
        slices.append((start, end))
        doc_windows.append(payloads)
        doc_ids.append(ex.doc_id)
        targets.append(float(ex.target_score))

    if not slices:
        raise ValueError("No non-empty MIL windows for embedding proxy fit")

    embeddings_flat = embedding_client.embed_texts(flat_payloads)
    dim = len(embeddings_flat[0])
    if dim <= 0:
        raise ValueError("Embedding dimension is zero")
    if any(len(vec) != dim for vec in embeddings_flat):
        raise ValueError("Inconsistent embedding dimensions in MIL fit data")

    import torch.nn.functional as F

    bags: List[torch.Tensor] = []
    for start, end in slices:
        bag = torch.tensor(embeddings_flat[start:end], dtype=torch.float32)
        if bag.numel() == 0:
            continue
        bag = F.normalize(bag, p=2, dim=1)
        bags.append(bag)

    if not bags:
        raise ValueError("No valid MIL bags after embedding")

    y_tensor = torch.tensor(targets[: len(bags)], dtype=torch.float32)

    torch.manual_seed(int(seed))
    w = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    bag_b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    optimizer = torch.optim.AdamW(
        [w, b, bag_b],
        lr=max(1e-6, float(learning_rate)),
        weight_decay=max(0.0, float(weight_decay)),
    )

    epochs_i = max(1, int(epochs))
    smooth_lambda = max(0.0, float(smoothness_lambda))
    sparse_lambda = max(0.0, float(sparsity_lambda))
    drift_temp = max(1e-6, float(drift_temperature))

    rng = __import__("random")
    rng.seed(int(seed))
    order = list(range(len(bags)))

    final_loss = None
    for _ in range(epochs_i):
        rng.shuffle(order)
        epoch_loss = 0.0
        for idx in order:
            bag = bags[idx]
            target = y_tensor[idx]
            logits = bag @ w + b
            p = torch.sigmoid(logits)
            sum_p = torch.sum(p)
            y_hat = torch.sigmoid(bag_b + torch.log1p(sum_p))
            loss = (y_hat - target) ** 2

            if sparse_lambda > 0.0:
                loss = loss + sparse_lambda * torch.mean(p)

            if smooth_lambda > 0.0 and bag.shape[0] >= 2:
                # drift in [0,2] since embeddings are L2-normalized.
                cos_sim = torch.sum(bag[:-1] * bag[1:], dim=1)
                drift = 1.0 - cos_sim
                weights = torch.exp(-drift / drift_temp)
                smooth = torch.mean(weights * (p[:-1] - p[1:]) ** 2)
                loss = loss + smooth_lambda * smooth

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu().item())

        final_loss = epoch_loss / max(1.0, float(len(order)))

    with torch.no_grad():
        w_out = w.detach().cpu().tolist()
        b_out = float(b.detach().cpu().item())
        bag_b_out = float(bag_b.detach().cpu().item())

    return EmbeddingMILSGDProxyModel(
        weights=[float(v) for v in w_out],
        bias=b_out,
        bag_bias=bag_b_out,
        embedding_dim=dim,
        embedding_model=embedding_client.resolve_model(),
        model_id=model_id,
        window_size_chars=window_size_chars,
        window_overlap_chars=window_overlap_chars,
        epochs=epochs_i,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        smoothness_lambda=smooth_lambda,
        sparsity_lambda=sparse_lambda,
        drift_temperature=drift_temp,
        train_size=len(bags),
        final_train_loss=final_loss,
        training_contract=supervision_training_contract(
            representation_kind=REPRESENTATION_BAG_OF_EMBEDDING_VECTORS,
            target_kind=TARGET_SCALAR,
            optimizer_family=OPTIMIZER_FAMILY_BAG_LEVEL_GRADIENT,
            optimizer_backend="torch_mil",
            n_train_rows=len(bags),
        ),
    )


def evaluate_embedding_proxy(
    model: Any,
    examples: Sequence[LabeledEmbeddingExample],
    *,
    embedding_client: VLLMEmbeddingClient,
) -> Dict[str, Any]:
    """Evaluate embedding proxy quality on labeled examples."""
    valid: List[LabeledEmbeddingExample] = []
    for ex in examples:
        target = _safe_optional_float(getattr(ex, "target_score", None))
        if target is None:
            continue
        text = str(getattr(ex, "text", "") or "")
        if not text:
            continue
        valid.append(
            LabeledEmbeddingExample(
                doc_id=str(getattr(ex, "doc_id", "")),
                text=text,
                target_score=_clamp01(target),
                truth_label_source=str(getattr(ex, "truth_label_source", "unknown")),
            )
        )

    if not valid:
        return {
            "n_examples": 0,
            "mae": None,
            "rmse": None,
            "mean_target": None,
            "mean_prediction": None,
        }

    abs_errors: List[float] = []
    sq_errors: List[float] = []
    targets: List[float] = []
    preds: List[float] = []

    if isinstance(model, EmbeddingMILSGDProxyModel):
        from treepo._research.preprocessing.adaptive_windows import uniform_axis_windows

        window_size = max(32, int(getattr(model, "window_size_chars", 1200)))
        window_overlap = max(0, int(getattr(model, "window_overlap_chars", 150)))
        if window_overlap >= window_size:
            window_overlap = max(0, window_size - 1)

        flat_payloads: List[str] = []
        slices: List[tuple[int, int]] = []
        for ex in valid:
            text = ex.text
            total = len(text)
            windows = uniform_axis_windows(
                total,
                window_size=window_size,
                overlap=window_overlap,
                unit="char",
            )
            payloads: List[str] = []
            for w in windows:
                start = max(0, min(total, int(w.start)))
                end = max(start, min(total, int(w.end)))
                snippet = text[start:end]
                if snippet.strip():
                    payloads.append(snippet)
            start_idx = len(flat_payloads)
            flat_payloads.extend(payloads)
            end_idx = len(flat_payloads)
            slices.append((start_idx, end_idx))

        embeddings_flat = embedding_client.embed_texts(flat_payloads) if flat_payloads else []
        for ex, (start, end) in zip(valid, slices):
            window_embeddings = embeddings_flat[start:end]
            pred = model.predict_bag_from_embeddings(window_embeddings)
            target = float(ex.target_score)
            err = pred - target
            abs_errors.append(abs(err))
            sq_errors.append(err * err)
            targets.append(target)
            preds.append(pred)
    else:
        embeddings = embedding_client.embed_texts([ex.text for ex in valid])
        for ex, vec in zip(valid, embeddings):
            target = float(ex.target_score)
            pred = model.predict_from_embedding(vec)
            err = pred - target
            abs_errors.append(abs(err))
            sq_errors.append(err * err)
            targets.append(target)
            preds.append(pred)

    n = float(len(valid))
    return {
        "n_examples": len(valid),
        "mae": sum(abs_errors) / n,
        "rmse": (sum(sq_errors) / n) ** 0.5,
        "mean_target": sum(targets) / n,
        "mean_prediction": sum(preds) / n,
    }


def export_embedding_finetune_dataset(
    examples: Sequence[LabeledEmbeddingExample],
    output_path: Path,
) -> Dict[str, Any]:
    """
    Export embedding fine-tune dataset as JSONL.

    Format per row:
      {"doc_id", "text", "target_score", "truth_label_source"}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(output_path, "w") as f:
        for ex in examples:
            target = _safe_optional_float(getattr(ex, "target_score", None))
            text = str(getattr(ex, "text", "") or "")
            if target is None or not text:
                continue
            row = {
                "doc_id": str(getattr(ex, "doc_id", "")),
                "text": text,
                "target_score": _clamp01(target),
                "truth_label_source": str(getattr(ex, "truth_label_source", "unknown")),
            }
            f.write(json.dumps(row) + "\n")
            n_rows += 1

    return {
        "path": str(output_path),
        "rows": n_rows,
    }
