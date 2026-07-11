"""Sklearn-LDA family and baseline report for synthetic topic fixtures.

``SklearnLDAFamily`` is the registered ``FamilyRuntime``
(``family='sklearn_lda'``) — the only place LDA training happens.
``fit_sklearn_lda_baseline`` is a report helper that routes its training
through :func:`treepo.fit` with that family and aggregates the topic-aligned
MAE table; it is not a second trainer.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo.local_law import LawKind, LocalLawAuditRow
from treepo.schedule import fold, merge_depths
from treepo.statistic import StatisticInfo
from treepo.tree import tree_leaves, tree_row_id


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


@dataclass(frozen=True)
class SklearnLDAFamilyConfig:
    n_topics: int = 2
    vocabulary_size: int = 0
    doc_topic_prior: float | None = None
    topic_word_prior: float | None = None
    max_iter: int = 100
    random_state: int = 0

    def __post_init__(self) -> None:
        if int(self.n_topics) <= 1 or int(self.vocabulary_size) <= 0:
            raise ValueError("n_topics must be > 1 and vocabulary_size must be positive")


class SklearnLDAFamily:
    """Sklearn LDA as a ``FamilyRuntime``: f is the aligned topic readout.

    ``train_f`` fits :class:`~sklearn.decomposition.LatentDirichletAllocation`
    on the training trees' token counts and aligns inferred components to the
    fixture's generating topics. g is the exact count-vector composition (no
    training), which is what the statistic surface audits.
    """

    name = "sklearn_lda"

    def __init__(self, config: SklearnLDAFamilyConfig) -> None:
        self.config = config
        self._model: Any = None
        self._topic_order: tuple[int, ...] | None = None

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
        del f_init, g, output_dir
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        from sklearn.decomposition import LatentDirichletAllocation

        train_list = list(traces or [])
        if not train_list:
            raise ValueError("sklearn_lda train_f requires non-empty traces")
        cfg = self.config
        model = LatentDirichletAllocation(
            n_components=int(cfg.n_topics),
            doc_topic_prior=cfg.doc_topic_prior,
            topic_word_prior=cfg.topic_word_prior,
            learning_method="batch",
            max_iter=int(cfg.max_iter),
            random_state=int(cfg.random_state),
        )
        model.fit(_count_matrix(train_list, int(cfg.vocabulary_size)))
        true_topic_words = _true_topic_word_distributions(
            train_list, int(cfg.n_topics), int(cfg.vocabulary_size)
        )
        self._topic_order = _align_components(
            model.components_, true_topic_words, linear_sum_assignment, np
        )
        self._model = model
        return {
            "kind": "treepo_sklearn_lda",
            "trained": "f",
            "iteration": int(iteration),
            "n_train": len(train_list),
            "topic_order": [int(x) for x in self._topic_order],
            "config": asdict(cfg),
        }

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
        del g_init, f, traces, output_dir
        # g is the exact additive count composition; there is nothing to fit.
        return {
            "kind": "treepo_sklearn_lda_g",
            "trained": "g",
            "iteration": int(iteration),
            "config": asdict(self.config),
        }

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[Any],
    ) -> list[list[float] | None]:
        del f, g
        tree_list = list(trees or [])
        if not tree_list:
            return []
        if self._model is None or self._topic_order is None:
            return [None] * len(tree_list)
        inferred = self._model.transform(
            _count_matrix(tree_list, int(self.config.vocabulary_size))
        )
        aligned = inferred[:, list(self._topic_order)]
        return [[float(x) for x in row] for row in aligned.tolist()]

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        if kind in {"f", "g"} and not isinstance(artifact, Mapping):
            raise TypeError(f"sklearn_lda {kind} artifact must be a mapping")

    def as_statistic(self, *, f: Any = None, g: Any = None) -> Any:
        del f, g
        return _LDACountStatistic(self)


class _LDACountStatistic:
    """Exact count-vector ComposableStatistic under the trained LDA readout.

    The composable state is the token-count vector — genuinely additive, so
    C2/C3 are exact indicator checks over real code paths; C1 audits each
    leaf's encoded counts against a direct recount of its tokens.
    """

    def __init__(self, family: SklearnLDAFamily) -> None:
        self.family = family
        self.info = StatisticInfo(
            name="sklearn_lda",
            state_kind="token_counts",
            exact=True,
            supports_local_laws=True,
            metadata={
                "config": asdict(family.config),
                "trained": family._model is not None,
            },
        )

    def encode_leaf(self, leaf: Any) -> Any:
        return _count_matrix([leaf], int(self.family.config.vocabulary_size))[0]

    def merge(self, left: Any, right: Any) -> Any:
        return left + right

    def readout(self, state: Any, query: Any = None) -> list[float] | None:
        del query
        family = self.family
        if family._model is None or family._topic_order is None:
            return None
        inferred = family._model.transform(state.reshape(1, -1))
        return [float(x) for x in inferred[0, list(family._topic_order)].tolist()]

    def predict_tree(self, tree: Any) -> list[float] | None:
        leaves = list(tree_leaves(tree) or ())
        if not leaves:
            return None
        states = [self.encode_leaf(leaf) for leaf in leaves]
        return self.readout(fold(states, self.merge, schedule="balanced"))

    def local_law_rows(
        self,
        units: Sequence[Any],
        *,
        query: Any = None,
        oracle: Any = None,
    ) -> Sequence[LocalLawAuditRow]:
        del query, oracle
        import numpy as np

        vocabulary_size = int(self.family.config.vocabulary_size)
        rows: list[LocalLawAuditRow] = []
        for idx, tree in enumerate(list(units or ())):
            leaves = list(tree_leaves(tree) or ())
            if not leaves:
                continue
            tree_id = tree_row_id(tree, idx, fallback_prefix="tree")
            base_metadata = {"statistic": self.info.name, "state_kind": self.info.state_kind}
            states = [self.encode_leaf(leaf) for leaf in leaves]
            depths = merge_depths(len(states), schedule="balanced")
            for leaf_idx, (leaf, state) in enumerate(zip(leaves, states)):
                recount = _count_matrix([leaf], vocabulary_size)[0]
                rows.append(
                    _exact_lda_row(
                        row_id=f"{tree_id}:leaf:{leaf_idx}",
                        law_kind=LawKind.C1_LEAF,
                        loss=0.0 if np.array_equal(state, recount) else 1.0,
                        depth=int(depths[leaf_idx]),
                        metadata={**base_metadata, "check": "leaf_recount", "law_facet": "c1_sufficiency"},
                    )
                )
            balanced = fold(states, self.merge, schedule="balanced")
            zero = np.zeros_like(balanced)
            rows.append(
                _exact_lda_row(
                    row_id=f"{tree_id}:idempotence",
                    law_kind=LawKind.C2_IDEMPOTENCE,
                    loss=0.0 if np.array_equal(self.merge(balanced, zero), balanced) else 1.0,
                    depth=0,
                    metadata={**base_metadata, "check": "zero_merge_identity", "law_facet": "c2_idempotence"},
                )
            )
            sequential = fold(states, self.merge, schedule="left_to_right")
            rows.append(
                _exact_lda_row(
                    row_id=f"{tree_id}:schedule",
                    law_kind=LawKind.C3_MERGE,
                    loss=0.0 if np.array_equal(balanced, sequential) else 1.0,
                    depth=0,
                    metadata={**base_metadata, "check": "schedule_invariance", "law_facet": "c3b_compositionality"},
                )
            )
            direct = _count_matrix([tree], vocabulary_size)[0]
            if float(direct.sum()) > 0.0:
                rows.append(
                    _exact_lda_row(
                        row_id=f"{tree_id}:root",
                        law_kind=LawKind.C3_MERGE,
                        loss=0.0 if np.array_equal(balanced, direct) else 1.0,
                        depth=0,
                        metadata={**base_metadata, "check": "composed_vs_direct", "law_facet": "c3a_joint_faithfulness"},
                    )
                )
        return tuple(rows)


def _exact_lda_row(
    *,
    row_id: str,
    law_kind: LawKind,
    loss: float,
    depth: int,
    metadata: Mapping[str, Any],
) -> LocalLawAuditRow:
    return LocalLawAuditRow(
        row_id=row_id,
        law_kind=law_kind,
        proxy_loss=float(loss),
        oracle_loss=float(loss),
        observed=True,
        propensity=1.0,
        depth=int(depth),
        metadata=dict(metadata),
    )


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
    """Fit sklearn LDA via ``fit()`` and report topic-aligned recovery MAE."""

    import numpy as np

    train_list = list(train_trees or [])
    eval_list = list(eval_trees or [])
    if not train_list or not eval_list:
        raise ValueError("train_trees and eval_trees must be non-empty")
    n_topics = int(n_topics)
    vocabulary_size = int(vocabulary_size)
    target_topic = int(target_topic)
    if not 0 <= target_topic < max(1, n_topics):
        raise ValueError("target_topic must be in [0, n_topics)")

    family = SklearnLDAFamily(
        SklearnLDAFamilyConfig(
            n_topics=n_topics,
            vocabulary_size=vocabulary_size,
            doc_topic_prior=doc_topic_prior,
            topic_word_prior=topic_word_prior,
            max_iter=int(max_iter),
            random_state=int(random_state),
        )
    )
    from treepo.learning import fit as _fit

    result = _fit(
        {
            "family": "sklearn_lda",
            "train_data": train_list,
            "eval_data": eval_list,
            "backend_config": {
                "family_runtime": family,
                "output_dir": tempfile.mkdtemp(prefix="treepo_lda_baseline_"),
            },
            "axis": {"max_iterations": 1, "axis_value": 0},
        },
    )
    if str(result.status) != "success":
        raise RuntimeError(f"sklearn_lda baseline fit failed: {result.status}")

    aligned = np.asarray(
        family.score_roots_with_f(f=None, g=None, trees=eval_list), dtype=float
    )
    topic_order = family._topic_order or ()
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
    "SklearnLDAFamily",
    "SklearnLDAFamilyConfig",
    "fit_sklearn_lda_baseline",
]
