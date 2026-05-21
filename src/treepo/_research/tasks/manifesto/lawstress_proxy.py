"""Embedding-proxy helpers for LawStress synthetic records.

This module trains a cheap "synthetic oracle" (embedding head) that can score
both full texts and summaries on a normalized RILE scale in [0, 1].

It is intentionally lightweight:
- Uses vLLM `/v1/embeddings` via `VLLMEmbeddingClient`
- Fits `EmbeddingRidgeProxyModel` (no torch required)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from treepo._research.tasks.manifesto.lawstress_generator import LawStressRecord, normalize_rile
from treepo._research.training.embedding_proxy import (
    EmbeddingRidgeProxyModel,
    LabeledEmbeddingExample,
    VLLMEmbeddingClient,
    fit_embedding_ridge_proxy,
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = float(len(xs))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    if den_x <= 0.0 or den_y <= 0.0:
        return None
    return float(numerator / math.sqrt(den_x * den_y))


def _average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _infer_text_type(doc_id: str) -> str:
    rendered = str(doc_id or "")
    if ":" in rendered:
        return rendered.split(":", 1)[1]
    return "unknown"


def build_proxy_training_examples(records: Sequence[LawStressRecord]) -> List[LabeledEmbeddingExample]:
    """Create ridge-proxy supervision from LawStress records.

    Each record yields multiple (text, label) pairs so the proxy learns to
    score both original texts and summary-style inputs.
    """

    examples: List[LabeledEmbeddingExample] = []

    def _add(example_id: str, text_type: str, text: str, target_raw: float, source: str) -> None:
        rendered = str(text or "").strip()
        if not rendered:
            return
        examples.append(
            LabeledEmbeddingExample(
                doc_id=f"{example_id}:{text_type}",
                text=rendered,
                target_score=_clamp01(normalize_rile(float(target_raw))),
                truth_label_source=str(source),
            )
        )

    for record in records:
        ex_id = str(record.example_id)
        _add(ex_id, "doc", record.text, float(record.teacher_score_doc), "teacher_doc")
        _add(ex_id, "segment_a", record.segment_a, float(record.teacher_score_segment_a), "teacher_segment_a")
        _add(ex_id, "segment_b", record.segment_b, float(record.teacher_score_segment_b), "teacher_segment_b")
        _add(ex_id, "naive_summary", record.naive_summary, float(record.naive_score_raw), "teacher_naive_summary")
        # For bootstrap: treat the reference summary as having the doc-level score.
        _add(ex_id, "reference_summary", record.reference_summary, float(record.teacher_score_doc), "teacher_doc_for_reference_summary")

    return examples


def train_embedding_proxy(
    examples: Sequence[LabeledEmbeddingExample],
    *,
    embedding_url: str,
    embedding_model: str,
    out_path: Path,
    ridge_lambda: float = 1.0,
    model_id: str = "lawstress_embedding_ridge_proxy_v1",
    api_key: str = "EMPTY",
    timeout_seconds: float = 60.0,
    batch_size: int = 32,
) -> EmbeddingRidgeProxyModel:
    """Fit and persist an EmbeddingRidgeProxyModel."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    embedding_client = VLLMEmbeddingClient(
        api_base=str(embedding_url),
        model=str(embedding_model),
        api_key=str(api_key or "EMPTY"),
        timeout_seconds=float(timeout_seconds),
        batch_size=int(batch_size),
    )

    trained = fit_embedding_ridge_proxy(
        examples,
        embedding_client=embedding_client,
        ridge_lambda=float(ridge_lambda),
        model_id=str(model_id),
    )
    trained.save_json(out_path)
    return trained


def load_embedding_proxy(path: Path) -> EmbeddingRidgeProxyModel:
    payload = Path(path).read_text(encoding="utf-8")
    import json

    return EmbeddingRidgeProxyModel.from_dict(json.loads(payload))


def evaluate_embedding_proxy(
    model: Any,
    *,
    embedding_client: Any,
    eval_examples: Sequence[LabeledEmbeddingExample],
) -> Dict[str, Any]:
    """Evaluate a proxy model on labeled examples.

    Args:
        model: Any object with `predict_from_embedding(embedding) -> float`.
        embedding_client: Any object with `embed_texts(list[str]) -> list[list[float]]`.
        eval_examples: Labeled examples (target_score in [0,1]).
    """

    texts = [str(ex.text or "") for ex in eval_examples]
    if not texts:
        return {"overall": {"n": 0}}

    embeddings = embedding_client.embed_texts(texts)
    preds: List[float] = []
    targets: List[float] = []
    types: List[str] = []

    for ex, emb in zip(eval_examples, embeddings):
        pred = getattr(model, "predict_from_embedding")(emb)
        preds.append(_clamp01(float(pred)))
        targets.append(_clamp01(float(ex.target_score)))
        types.append(_infer_text_type(ex.doc_id))

    def _metrics(rows: List[int]) -> Dict[str, Any]:
        if not rows:
            return {"n": 0, "mae": None, "pearson_r": None, "spearman_r": None}
        ys = [targets[i] for i in rows]
        yhat = [preds[i] for i in rows]
        n = len(rows)
        mae = sum(abs(a - b) for a, b in zip(ys, yhat)) / float(n)
        return {
            "n": int(n),
            "mae": float(mae),
            "pearson_r": _pearson(yhat, ys),
            "spearman_r": _spearman(yhat, ys),
        }

    all_rows = list(range(len(eval_examples)))
    by_type: Dict[str, List[int]] = {}
    for idx, t in enumerate(types):
        by_type.setdefault(t, []).append(idx)

    return {
        "overall": _metrics(all_rows),
        "by_type": {t: _metrics(rows) for t, rows in sorted(by_type.items())},
    }


__all__ = [
    "build_proxy_training_examples",
    "evaluate_embedding_proxy",
    "load_embedding_proxy",
    "train_embedding_proxy",
]

