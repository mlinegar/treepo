"""Small LDA helpers for synthetic topic-mixture fixtures.

The package includes a synthetic Dirichlet LDA fixture in
``treepo.methods.fixtures`` and a tiny sklearn baseline helper here. Larger
application recovery experiments should register from the package that owns
their data and evaluation loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SklearnLDABaselineResult:
    """Result from fitting sklearn LDA on generated document-token counts."""

    target_key: str
    target_topic: int
    topic_order: tuple[int, ...]
    mean_mae: float
    target_mae: float
    average_guess_target: float
    average_guess_target_mae: float
    average_guess_mean_mae: float
    mae_by_topic: tuple[float, ...]
    predictions: tuple[tuple[float, ...], ...] = field(default_factory=tuple)
    truth: tuple[tuple[float, ...], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_key": self.target_key,
            "target_topic": int(self.target_topic),
            "topic_order": list(self.topic_order),
            "mean_mae": float(self.mean_mae),
            "target_mae": float(self.target_mae),
            "average_guess_target": float(self.average_guess_target),
            "average_guess_target_mae": float(self.average_guess_target_mae),
            "average_guess_mean_mae": float(self.average_guess_mean_mae),
            "mae_by_topic": [float(x) for x in self.mae_by_topic],
            "predictions": [list(row) for row in self.predictions],
            "truth": [list(row) for row in self.truth],
        }


def fit_sklearn_lda_baseline(
    train_trees: Sequence[Any],
    eval_trees: Sequence[Any],
    *,
    n_topics: int,
    vocabulary_size: int,
    doc_topic_prior: float | None = None,
    topic_word_prior: float | None = None,
    max_iter: int = 100,
    random_state: int = 0,
    target_topic: int = 0,
) -> SklearnLDABaselineResult:
    """Fit official sklearn LDA and align inferred topics to generated topics."""

    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from sklearn.decomposition import LatentDirichletAllocation

    train_list = list(train_trees or [])
    eval_list = list(eval_trees or [])
    if not train_list or not eval_list:
        raise ValueError("train_trees and eval_trees must be non-empty")
    n_topics = int(n_topics)
    vocabulary_size = int(vocabulary_size)
    target_topic = int(target_topic)
    if n_topics <= 1 or vocabulary_size <= 0:
        raise ValueError("n_topics must be > 1 and vocabulary_size must be positive")
    if not 0 <= target_topic < n_topics:
        raise ValueError("target_topic must be in [0, n_topics)")

    model = LatentDirichletAllocation(
        n_components=n_topics,
        doc_topic_prior=doc_topic_prior,
        topic_word_prior=topic_word_prior,
        learning_method="batch",
        max_iter=int(max_iter),
        random_state=int(random_state),
    )
    model.fit(_count_matrix(train_list, vocabulary_size))
    inferred = model.transform(_count_matrix(eval_list, vocabulary_size))
    true_topic_words = _true_topic_word_distributions(train_list, n_topics, vocabulary_size)
    topic_order = _align_components(model.components_, true_topic_words, linear_sum_assignment, np)
    aligned = inferred[:, list(topic_order)]
    train_truth = np.asarray([_topic_proportions(tree, n_topics) for tree in train_list], dtype=float)
    truth = np.asarray([_topic_proportions(tree, n_topics) for tree in eval_list], dtype=float)
    abs_err = np.abs(aligned - truth)
    mae_by_topic = tuple(float(x) for x in abs_err.mean(axis=0).tolist())
    average_guess = train_truth.mean(axis=0)
    average_abs_err = np.abs(average_guess.reshape(1, -1) - truth)
    average_mae_by_topic = average_abs_err.mean(axis=0)
    return SklearnLDABaselineResult(
        target_key=f"topic_{target_topic}_proportion",
        target_topic=target_topic,
        topic_order=tuple(int(x) for x in topic_order),
        mean_mae=float(abs_err.mean()),
        target_mae=float(mae_by_topic[target_topic]),
        average_guess_target=float(average_guess[target_topic]),
        average_guess_target_mae=float(average_mae_by_topic[target_topic]),
        average_guess_mean_mae=float(average_abs_err.mean()),
        mae_by_topic=mae_by_topic,
        predictions=tuple(tuple(float(x) for x in row.tolist()) for row in aligned),
        truth=tuple(tuple(float(x) for x in row.tolist()) for row in truth),
    )


def _count_matrix(trees: Sequence[Any], vocabulary_size: int) -> Any:
    import numpy as np

    matrix = np.zeros((len(trees), int(vocabulary_size)), dtype=float)
    for row_idx, tree in enumerate(trees):
        for token in getattr(tree, "tokens", ()) or ():
            token_idx = int(token)
            if 0 <= token_idx < int(vocabulary_size):
                matrix[row_idx, token_idx] += 1.0
    return matrix


def _true_topic_word_distributions(
    trees: Sequence[Any],
    n_topics: int,
    vocabulary_size: int,
) -> Any:
    import numpy as np

    for tree in trees:
        meta = getattr(tree, "metadata", None) or {}
        value = meta.get("topic_word_distributions") if isinstance(meta, Mapping) else None
        if value is not None:
            arr = np.asarray(value, dtype=float)
            if arr.shape == (int(n_topics), int(vocabulary_size)):
                return arr / arr.sum(axis=1, keepdims=True)
    raise ValueError("LDA trees must carry metadata['topic_word_distributions'] for topic alignment")


def _topic_proportions(tree: Any, n_topics: int) -> tuple[float, ...]:
    values = getattr(tree, "topic_proportions", None)
    if values is None:
        meta = getattr(tree, "metadata", None) or {}
        values = meta.get("topic_proportions") if isinstance(meta, Mapping) else None
    if values is None:
        raise ValueError("LDA tree must carry topic_proportions")
    out = tuple(float(x) for x in values)
    if len(out) != int(n_topics):
        raise ValueError(f"expected {n_topics} topic proportions, got {len(out)}")
    return out


def _align_components(components: Any, true_topic_words: Any, linear_sum_assignment: Any, np: Any) -> tuple[int, ...]:
    estimated = np.asarray(components, dtype=float)
    estimated = estimated / estimated.sum(axis=1, keepdims=True)
    true = np.asarray(true_topic_words, dtype=float)
    true = true / true.sum(axis=1, keepdims=True)
    denom = np.linalg.norm(estimated, axis=1, keepdims=True) * np.linalg.norm(true, axis=1)
    similarity = estimated @ true.T / np.maximum(denom, 1e-12)
    component_rows, true_cols = linear_sum_assignment(-similarity)
    true_to_component = {int(true_topic): int(component) for component, true_topic in zip(component_rows, true_cols)}
    return tuple(true_to_component[idx] for idx in range(true.shape[0]))


__all__ = [
    "SklearnLDABaselineResult",
    "fit_sklearn_lda_baseline",
]
